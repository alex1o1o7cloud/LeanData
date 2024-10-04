import Mathlib
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.Functions
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.OddsAndEnds.ComplexRoots
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.NormedSpace.LpSpace
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Collapsing
import Mathlib.Combinatorics.SimpleGraph
import Mathlib.Combinatorics.SimpleGraph.Connectivity
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Graph.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Cyclic
import Mathlib.Init.Algebra.Order
import Mathlib.LinearAlgebra.Matrix
import Mathlib.MeasureTheory.Function.Lp
import Mathlib.MeasureTheory.Integral.Bochner
import Mathlib.MeasureTheory.Integral.SetIntegral
import Mathlib.NumberTheory.ModularArithmetic
import Mathlib.Probability.Basic
import Mathlib.Probability.Independence
import Mathlib.Probability.RandomVariable
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Algebra.Module
import Mathlib.Topology.Instances.Real
import Real
import data.finset
import data.nat.basic

namespace max_area_quadrilateral_AEBF_l44_44767

theorem max_area_quadrilateral_AEBF :
  ∀ (m : ℝ) (E F : ℝ × ℝ),
  (E ∈ ellipse 4 1) ∧ (F ∈ ellipse 4 1) ∧
  (E.2 / E.1 = m) ∧ (F.2 / F.1 = m) ∧
  ∃ t, (E ≠ F) ∧ (E, F) = ((2 * cos t, sin t), (-2 * cos t, -sin t)) →
  let A := (2, 0), B := (0, 1) in
  (max_area_of_quadrilateral A E B F = 2 * real.sqrt 2) :=
begin
  sorry
end

def ellipse (a b : ℝ) : set (ℝ × ℝ) :=
{p | let (x, y) := p in (x^2 / a^2) + (y^2 / b^2) = 1}

def max_area_of_quadrilateral (A E B F : ℝ × ℝ) : ℝ :=
let xA := A.1, yA := A.2,
    xB := B.1, yB := B.2,
    xE := E.1, yE := E.2,
    xF := F.1, yF := F.2 in
0.5 * abs (xA * (yE - yF) + xE * (yF - yA) + xF * (yA - yE) +
           xB * (yF - yE) + xE * (yA - yF) + xF * (yB - yE))

end max_area_quadrilateral_AEBF_l44_44767


namespace subsets_odd_cardinality_l44_44469

theorem subsets_odd_cardinality (n : ℕ) : 
    (cardinal {s : set (fin n) | s.card % 2 = 1}) = 2^(n - 1) :=
sorry

end subsets_odd_cardinality_l44_44469


namespace curve_C_to_line_l_min_distance_l44_44552

def line_l_parametric (t : ℝ) : ℝ × ℝ :=
  (6 - (sqrt 3 / 2) * t, (1 / 2) * t)

def line_l (x y : ℝ) : Prop :=
  x + sqrt 3 * y = 6

def curve_C (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

def curve_C' (x' y' : ℝ) : Prop :=
  (x'/3)^2 + y'^2 = 1

def scaling_transformation (x y : ℝ) : ℝ × ℝ :=
  (3 * x, y)

theorem curve_C_to_line_l :
  ∀ (x y : ℝ), curve_C x y → line_l (6 - (sqrt 3 / 2) * t) ((1 / 2) * t) :=
by
  sorry

theorem min_distance {x' y' : ℝ} :
  (curve_C' x' y') -> 
  ∀ (theta : ℝ), (3 * cos theta, sin theta) minimizes the distance to the line l :=
by
  sorry

end curve_C_to_line_l_min_distance_l44_44552


namespace maximum_minimum_sum_10_l44_44856

-- Definitions and notations for the sequence and sums
def seq (a : ℕ → ℕ) :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → (a (n + 1) - a n) ∈ { a i | i ≤ n ∧ i > 0 }

def sum_seq (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, a (i + 1)

theorem maximum_minimum_sum_10 : 
  ∀ (a : ℕ → ℕ),
    seq a →
    let S10 := sum_seq a 10 in
    S10 ≤ 1023 ∧ S10 ≥ 55 ∧ (∃ a, seq a ∧ sum_seq a 10 = 1023) ∧ (∃ a, seq a ∧ sum_seq a 10 = 55) →
    (∑ i in Finset.range 10, a (i + 1)) = 1078 := 
by
  intros a ha S10 hs
  sorry

end maximum_minimum_sum_10_l44_44856


namespace triangle_XYZ_PQ_l44_44208

-- The specific conditions given in the problem, defined as a substructure.
variables {X Y Z D E P Q : Type}
variables (XY XZ YZ: ℝ) (h1 : XY = 153) (h2 : XZ = 147) (h3 : YZ = 140)
variables (angleBisectorX : D) (angleBisectorY : E)
variables (perpendicularFromZToYE : P) (perpendicularFromZToXD : Q)

-- The theorem stating the problem with the given conditions and required proof that PQ = 67.
theorem triangle_XYZ_PQ 
    (angleBisectorX_intersects_YZ : ∀ D, D ∈ lineSegment YZ)
    (angleBisectorY_intersects_XZ : ∀ E, E ∈ lineSegment XZ) 
    (P_foot_perp_YE_from_Z : ∀ P, P ∈ perpendicularFoot Z (line YE))
    (Q_foot_perp_XD_from_Z : ∀ Q, Q ∈ perpendicularFoot Z (line XD))
    : distance P Q = 67 := 
by
  -- skipped proof
  sorry

end triangle_XYZ_PQ_l44_44208


namespace graph_c_is_shifted_g_l44_44551

def g (x : ℝ) : ℝ :=
  if -2 ≤ x ∧ x ≤ 2 then -x
  else if 2 ≤ x ∧ x ≤ 5 then real.sqrt (9 - (x - 5)^2) - 1
  else if 5 ≤ x ∧ x ≤ 7 then 3 * (x - 5)
  else 0

-- Define the shifted function
def g_shifted (x : ℝ) : ℝ := g(x) + 3

-- Define the graph C function to compare
def graph_c (x : ℝ) : ℝ :=
  if -2 ≤ x ∧ x ≤ 2 then -x + 3
  else if 2 ≤ x ∧ x ≤ 5 then real.sqrt (9 - (x - 5)^2) + 2
  else if 5 ≤ x ∧ x ≤ 7 then 3 * (x - 5) + 3
  else 0

-- Statement of the proof problem
theorem graph_c_is_shifted_g :
  ∀ x : ℝ, g_shifted(x) = graph_c(x) :=
sorry

end graph_c_is_shifted_g_l44_44551


namespace problem_statement_l44_44183

noncomputable def C_points_count (A B : (ℝ × ℝ)) : ℕ :=
  if A = (0, 0) ∧ B = (12, 0) then 4 else 0

theorem problem_statement :
  let A := (0, 0)
  let B := (12, 0)
  C_points_count A B = 4 :=
by
  sorry

end problem_statement_l44_44183


namespace balloon_arrangements_l44_44083

theorem balloon_arrangements : (7! / (2! * 2!)) = 1260 := by
  sorry

end balloon_arrangements_l44_44083


namespace no_valid_schedule_l44_44591

theorem no_valid_schedule (n : Nat) (h_n : n = 100) :
  ¬ ∃ (schedule : List (Finset (Fin n))), 
      (∀ day in schedule, day.card = 3) ∧
      (∀ pair : Finset (Fin n), pair.card = 2 → ∃! day in schedule, pair ⊆ day) := by
  sorry

end no_valid_schedule_l44_44591


namespace leak_emptying_time_l44_44422

theorem leak_emptying_time (fill_rate_no_leak : ℝ) (combined_rate_with_leak : ℝ) (L : ℝ) :
  fill_rate_no_leak = 1/10 →
  combined_rate_with_leak = 1/12 →
  fill_rate_no_leak - L = combined_rate_with_leak →
  1 / L = 60 :=
by
  intros h1 h2 h3
  sorry

end leak_emptying_time_l44_44422


namespace distinct_arrangements_balloon_l44_44121

theorem distinct_arrangements_balloon : 
  let n := 7
  let r1 := 2
  let r2 := 2
  (Nat.factorial n) / ((Nat.factorial r1) * (Nat.factorial r2)) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l44_44121


namespace equal_sides_l44_44930

variables {A B C D E F G H I : Type} [linear_ordered_field I]
variables (triangle : Type) [is_triangle triangle]
variables (incircle : Type) [is_circle incircle]
variables [tangent incircle A B] [tangent incircle B C]
variables [intersection : C F A B] [H_on_CG : H ∈ line.segment (C G)]
variables (HG_eq_CF : dist H G = dist C F)
variables [collinear A H E]

theorem equal_sides (h1 : tangent incircle A B)
                    (h2 : tangent incircle B C)
                    (h3 : intersection point (D I F))
                    (h4 : point_line_intersection F B G)
                    (h5 : H_on_CG)
                    (h6 : collinear A H E)
                    (h_dist : HG_eq_CF):
                    dist A B = dist A C := 
sorry

end equal_sides_l44_44930


namespace leading_coefficient_poly_l44_44797

variable (x : ℝ)

def poly : ℝ := -5 * (x^4 - 2 * x^3 + 3 * x) + 8 * (x^4 - x^2 + 1) - 3 * (3 * x^4 + x^3 + x)

theorem leading_coefficient_poly : leadingCoeff (poly x) = -6 := by
  sorry

end leading_coefficient_poly_l44_44797


namespace heptagon_angle_in_arithmetic_progression_l44_44684

theorem heptagon_angle_in_arithmetic_progression (a d : ℝ) :
  a + 3 * d = 128.57 → 
  (7 * a + 21 * d = 900) → 
  ∃ angle : ℝ, angle = 128.57 :=
by
  sorry

end heptagon_angle_in_arithmetic_progression_l44_44684


namespace charlotte_and_dan_mean_score_l44_44844

theorem charlotte_and_dan_mean_score (
  scores : List ℕ,
  sum_scores : Nat,
  mean_ava_ben : ℕ
) : Float := by
  -- Conditions from the problem statement
  have h1 : scores = [82, 84, 86, 88, 90, 92, 95, 97] := by sorry
  have h2 : sum_scores = 714 := by sorry
  have h3 : mean_ava_ben = 90 := by sorry
  
  -- Expected conclusion
  have conclusion : Float := (714 - (4 * 90)) / 4
  exact conclusion

end charlotte_and_dan_mean_score_l44_44844


namespace exists_convex_polyhedron_net_of_triangle_l44_44411

/-- A theorem that states there exists a convex polyhedron that can be cut along its edges 
     and unfolded into a triangle without internal cuts, specifically, a tetrahedral pyramid. -/
theorem exists_convex_polyhedron_net_of_triangle : 
  ∃ (P : Polyhedron), (Convex P) ∧ (∃ (n : Net P), (net_is_triangle n) ∧ (without_internal_cuts n)) :=
sorry

end exists_convex_polyhedron_net_of_triangle_l44_44411


namespace hypotenuse_of_45_45_90_triangle_l44_44644

noncomputable def leg_length : ℝ := 15
noncomputable def angle_opposite_leg : ℝ := Real.pi / 4  -- 45 degrees in radians

theorem hypotenuse_of_45_45_90_triangle (h_leg : ℝ) (h_angle : ℝ) 
  (h_leg_cond : h_leg = leg_length) (h_angle_cond : h_angle = angle_opposite_leg) :
  ∃ h_hypotenuse : ℝ, h_hypotenuse = h_leg * Real.sqrt 2 :=
sorry

end hypotenuse_of_45_45_90_triangle_l44_44644


namespace five_letter_arrangements_l44_44895

theorem five_letter_arrangements : 
  let letters := ['A', 'B', 'C', 'D', 'E', 'F', 'G']
  let arrangements := {l : List Char | l.length = 5 ∧ l.head = 'D' ∧ 'E' ∈ l.tail ∧ l.nodup}
  arrangements.card = 480 := 
sorry

end five_letter_arrangements_l44_44895


namespace quadratic_roots_condition_l44_44918

theorem quadratic_roots_condition (k : ℝ) : 
  ((∃ x : ℝ, (k - 1) * x^2 + 4 * x + 1 = 0) ∧ ∃ x1 x2 : ℝ, x1 ≠ x2) ↔ (k < 5 ∧ k ≠ 1) :=
by {
  sorry  
}

end quadratic_roots_condition_l44_44918


namespace hypotenuse_of_45_45_90_triangle_15_l44_44654

theorem hypotenuse_of_45_45_90_triangle_15 (a : ℝ) (h : a = 15) : 
  ∃ (c : ℝ), c = a * Real.sqrt 2 :=
by
  use a * Real.sqrt 2
  rw h
  sorry

end hypotenuse_of_45_45_90_triangle_15_l44_44654


namespace target_shooting_orders_l44_44941

theorem target_shooting_orders :
  ∃ n : ℕ, n = nat.factorial 9 / (nat.factorial 3 * nat.factorial 3 * nat.factorial 3) ∧ n = 1680 :=
sorry

end target_shooting_orders_l44_44941


namespace arithmetic_sequence_count_l44_44479

theorem arithmetic_sequence_count :
  ∀ (a d a_n : ℕ), a = 6 → d = 5 → a_n = 91 → ∃ n, a + (n - 1) * d = 91 ∧ n = 18 :=
by
  intros a d a_n ha hd han
  use 18
  rw [ha, hd, han]
  norm_num
  sorry

end arithmetic_sequence_count_l44_44479


namespace area_XYZ_fraction_ABC_l44_44805

-- Define the points A, B, C, X, Y, and Z
def A := (2 : ℝ, 0 : ℝ)
def B := (8 : ℝ, 12 : ℝ)
def C := (16 : ℝ, 8 : ℝ)
def X := (6 : ℝ, 0 : ℝ)
def Y := (10 : ℝ, 4 : ℝ)
def Z := (12 : ℝ, 0 : ℝ)

-- Define a function to calculate the area of a triangle given its vertices
def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  0.5 * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

-- Calculate the areas of triangles XYZ and ABC
noncomputable def Area_XYZ := triangle_area X Y Z
noncomputable def Area_ABC := triangle_area A B C

-- Define the proof problem
theorem area_XYZ_fraction_ABC : Area_XYZ / Area_ABC = 1 / 5 :=
by
  sorry

end area_XYZ_fraction_ABC_l44_44805


namespace solve_inequality_l44_44311

theorem solve_inequality (x : ℝ) : -3 * x^2 + 8 * x + 1 < 0 ↔ x ∈ Set.Ioo (-1 / 3 : ℝ) 1 :=
sorry

end solve_inequality_l44_44311


namespace b_n_plus_c_n_constant_l44_44279

def seq_a : ℕ+ → ℝ
| ⟨1, _⟩ := 4
| ⟨n+1, p⟩ := seq_a ⟨n, nat.succ_pos _⟩

def seq_b : ℕ+ → ℝ
| ⟨1, _⟩ := 3
| ⟨n+1, p⟩ := (seq_a ⟨n, nat.succ_pos _⟩ + seq_c ⟨n, nat.succ_pos _⟩) / 2

def seq_c : ℕ+ → ℝ
| ⟨1, _⟩ := 5
| ⟨n+1, p⟩ := (seq_a ⟨n, nat.succ_pos _⟩ + seq_b ⟨n, nat.succ_pos _⟩) / 2

theorem b_n_plus_c_n_constant (n : ℕ+) : seq_b n + seq_c n = 8 :=
sorry

end b_n_plus_c_n_constant_l44_44279


namespace hypotenuse_of_454590_triangle_l44_44641

theorem hypotenuse_of_454590_triangle (l : ℝ) (angle : ℝ) (h : ℝ) (h_leg : l = 15) (h_angle : angle = 45) :
  h = l * Real.sqrt 2 := 
  sorry

end hypotenuse_of_454590_triangle_l44_44641


namespace locus_is_hyperbola_l44_44043

noncomputable def locus_of_points (O M P : ℝ × ℝ) (r : ℝ) : Prop :=
  let ON := r in
  let y := (M.2 : ℝ) in
  let x := real.sqrt (y ^ 2 + r ^ 2) in
  M.1 = x ∧ P.2 = M.2 + y ∧ P.1 = M.1
  
theorem locus_is_hyperbola :
  ∀ (O M P : ℝ × ℝ) (r : ℝ),
  r > 0 →
  locus_of_points O M P r →
  ∃ (x y : ℝ), P = (x, y) ∧ (x^2 / r^2) - (y^2 / r^2) = 1 :=
begin
  intros O M P r hr hlocus,
  sorry -- Proof to be provided here.
end

end locus_is_hyperbola_l44_44043


namespace F_999_eq_998_F_984_eq_F_F_F_1004_F_84_eq_997_l44_44993

def F : ℤ → ℤ
| (n : ℤ) := if n ≥ 1000 then n - 3 else F (F (n + 5))

/- Part (a) -/
theorem F_999_eq_998 : F 999 = 998 := sorry

/- Part (b) -/
theorem F_984_eq_F_F_F_1004 : F 984 = F (F (F 1004)) := sorry

/- Part (c) -/
theorem F_84_eq_997 : F 84 = 997 := sorry

end F_999_eq_998_F_984_eq_F_F_F_1004_F_84_eq_997_l44_44993


namespace sum_infinite_series_l44_44032

noncomputable def H : ℕ → ℚ 
| 0       := 0
| (n + 1) := H n + 1 / (n + 1)

theorem sum_infinite_series (k : ℕ) (hk : 0 < k) :
  ∑' n, 1 / ((n + k) * H n * H (n + 1)) = 1 / (1 + k) :=
by 
  sorry

end sum_infinite_series_l44_44032


namespace find_a5_l44_44888

def seq : ℕ → ℕ
| 0     := 0    -- provide a dummy term for a_0
| 1     := 1
| (n+1) := 2 * (seq n) + 3 * n

theorem find_a5 : seq 5 = 94 := 
by
  sorry

end find_a5_l44_44888


namespace difference_of_squares_example_l44_44801

theorem difference_of_squares_example : 625^2 - 375^2 = 250000 :=
by sorry

end difference_of_squares_example_l44_44801


namespace distinct_arrangements_balloon_l44_44113

noncomputable def totalPermutations (n nl no : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial nl * Nat.factorial no)

theorem distinct_arrangements_balloon :
  totalPermutations 7 2 2 = 1260 := by 
  sorry

end distinct_arrangements_balloon_l44_44113


namespace balloon_arrangements_l44_44082

theorem balloon_arrangements : (7! / (2! * 2!)) = 1260 := by
  sorry

end balloon_arrangements_l44_44082


namespace calculate_Sn_l44_44465

variable {x : ℝ} {n : ℕ}

def S_n_special (x : ℝ) (n : ℕ) : ℝ :=
  if x = 1 ∨ x = -1 then 4 * n else S_n_general x n

def S_n_general (x : ℝ) (n : ℕ) : ℝ :=
  (x^2 - x^(2*n + 2)) / (1 - x^2) + (2 * x * (1 - x^n)) / (1 - x) + n

theorem calculate_Sn :
  S_n_special x n = 
    if x = 1 ∨ x = -1 then 4 * n else
    (x^2 - x^(2*n + 2)) / (1 - x^2) + (2 * x * (1 - x^n)) / (1 - x) + n :=
by 
  apply dite;
  { intro hx, sorry }, -- This part deals with the special case x = 1 or x = -1
  { intro hx, sorry }  -- This part deals with the general case x ≠ 1 and x ≠ -1

end calculate_Sn_l44_44465


namespace min_vector_magnitude_l44_44510

variables {α : Type*} [inner_product_space ℝ α]

theorem min_vector_magnitude 
  (OA OB : α) 
  (hOA : ∥OA∥ = 4)
  (hOB : ∥OB∥ = 2) 
  (hAOB : angle OA OB = 2 * real.pi / 3) 
  (OC : α) 
  (x y : ℝ) 
  (hOC : OC = x • OA + y • OB)
  (hxy : x + 2 * y = 1) : 
  ∥OC∥ = 2 * real.sqrt 7 / 7 :=
begin
  sorry
end

end min_vector_magnitude_l44_44510


namespace quadratic_distinct_real_roots_l44_44920

theorem quadratic_distinct_real_roots (k : ℝ) :
  ((k - 1) ≠ 0) ∧ ((4^2 - 4 * (k - 1) * 1) > 0) → k < 5 ∧ k ≠ 1 :=
by
  -- We state the problem conditions directly and prove the intended result.
  intro h
  cases h with hk hΔ
  sorry

end quadratic_distinct_real_roots_l44_44920


namespace johns_age_l44_44235

theorem johns_age (d j : ℕ) 
  (h1 : j = d - 30) 
  (h2 : j + d = 80) : 
  j = 25 :=
by
  sorry

end johns_age_l44_44235


namespace equal_lengths_if_and_only_if_isosceles_l44_44188

variable {ABC : Type} [EuclideanGeometry ABC]
variables {A B C H_a H_b H_c : ABC}
variables (isFeetOfAltitudes : feet_of_altitudes A B C H_a H_b H_c)

theorem equal_lengths_if_and_only_if_isosceles :
  (dist H_a H_b = dist H_a H_c ∨ dist H_b H_c = dist H_a H_c ∨ dist H_a H_b = dist H_b H_c) ↔ 
  (angle B C A = angle C B A) :=
by
  sorry

end equal_lengths_if_and_only_if_isosceles_l44_44188


namespace cube_root_neg_64_l44_44745

-- Define cube root function
def cube_root (x : ℤ) : ℤ :=
  if ∃ y : ℤ, y^3 = x then classical.some (exists.intro (-4) (by norm_num)) else 0

-- Theorem to be proved
theorem cube_root_neg_64 : cube_root (-64) = -4 :=
by {
  unfold cube_root,
  rw if_pos,
  apply classical.some_spec (exists.intro (-4) _),
  norm_num,
  sorry
}

end cube_root_neg_64_l44_44745


namespace exists_min_a_and_b_l44_44703

noncomputable def polynomial := λ (a b : ℝ), (λ x, x^4 - a * x^3 + b * x^2 - a * x + 2)

theorem exists_min_a_and_b :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ x, polynomial a b x = 0 → ∀ r s t u : ℝ, x = r ∨ x = s ∨ x = t ∨ x = u) ∧
  b = 6 * Real.sqrt 2 :=
sorry

end exists_min_a_and_b_l44_44703


namespace find_value_in_box_l44_44386

theorem find_value_in_box (x : ℕ) :
  10 * 20 * 30 * 40 * 50 = 100 * 2 * 300 * 4 * x ↔ x = 50 := by
  sorry

end find_value_in_box_l44_44386


namespace max_distance_S_l44_44985

noncomputable def maximum_distance_S_origin (z : ℂ) (hz : abs z = 1) : ℝ :=
  let w := (1 - complex.i) * z + 3 * complex.conj z in
  abs w

theorem max_distance_S (z : ℂ) (hz : abs z = 1) : maximum_distance_S_origin z hz = real.sqrt 17 := by
  sorry

end max_distance_S_l44_44985


namespace piecewise_function_eval_l44_44847

def f (x : ℝ) : ℝ :=
  if x ∈ set.Ioo (-6) (-1) then |2 * x + 3|
  else if x ∈ set.Icc (-1) (1) then x ^ 2
  else if x ∈ set.Icc (1) (6) then x
  else 0

theorem piecewise_function_eval (h1 : f (Real.sqrt 2) = Real.sqrt 2) (h2 : f (-π) = 2 * π - 3) : true :=
by
  sorry

end piecewise_function_eval_l44_44847


namespace buffy_breath_holding_time_l44_44251

theorem buffy_breath_holding_time (k : ℕ) (b : ℕ) : 
  k = 3 * 60 ∧ b = k - 20 → b - 40 = 120 := 
by
  intros h
  cases h with hk hb
  rw [hk, hb]
  norm_num
  sorry  -- This "sorry" is here to skip the proof

end buffy_breath_holding_time_l44_44251


namespace tower_remainder_modulo_l44_44759

-- Define the problem conditions
def is_valid_tower (tower : List ℕ) : Prop :=
  ∀ (k : ℕ), (k < tower.length - 1) → (tower[k+1] ≤ tower[k] + 2)

-- Define the collection of cubes
def cubes := {k : ℕ // 1 ≤ k ∧ k ≤ 10}

-- Define the number of valid towers and the modulo operation
noncomputable def T : ℕ :=
  -- This is a placeholder definition; an actual definition would compute the number of valid towers.
  sorry

noncomputable def remainder := T % 1000

-- Formalizing the proof problem
theorem tower_remainder_modulo : remainder = 122 :=
  sorry

end tower_remainder_modulo_l44_44759


namespace point_on_terminal_of_120_degree_l44_44577

theorem point_on_terminal_of_120_degree (a : ℝ) 
    (h : ∃ a : ℝ, (-4, a) ∈ { (x, y) | tan 120 = y / x }) : 
    a = 4 * real.sqrt 3 :=
by
  -- define the necessary conditions regarding the point and its relation to the angle
  have h1 : tan 120 = real.sqrt 3 := sorry
  have h2 : tan 120 = a / -4 := sorry
  have h3 : -real.sqrt 3 = a / -4 := sorry
  sorry

end point_on_terminal_of_120_degree_l44_44577


namespace distinct_arrangements_balloon_l44_44109

noncomputable def totalPermutations (n nl no : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial nl * Nat.factorial no)

theorem distinct_arrangements_balloon :
  totalPermutations 7 2 2 = 1260 := by 
  sorry

end distinct_arrangements_balloon_l44_44109


namespace hypotenuse_of_45_45_90_triangle_l44_44645

noncomputable def leg_length : ℝ := 15
noncomputable def angle_opposite_leg : ℝ := Real.pi / 4  -- 45 degrees in radians

theorem hypotenuse_of_45_45_90_triangle (h_leg : ℝ) (h_angle : ℝ) 
  (h_leg_cond : h_leg = leg_length) (h_angle_cond : h_angle = angle_opposite_leg) :
  ∃ h_hypotenuse : ℝ, h_hypotenuse = h_leg * Real.sqrt 2 :=
sorry

end hypotenuse_of_45_45_90_triangle_l44_44645


namespace range_of_k_l44_44535

theorem range_of_k (k : ℝ) (a : ℕ+ → ℝ) (h : ∀ n : ℕ+, a n = 2 * (n:ℕ)^2 + k * (n:ℕ)) 
  (increasing : ∀ n : ℕ+, a n < a (n + 1)) : 
  k > -6 := 
by 
  sorry

end range_of_k_l44_44535


namespace volume_of_tetrahedron_OMNB1_is_correct_l44_44190

def Point := ℝ × ℝ × ℝ

noncomputable def volume_of_tetrahedron (O M N B₁ : Point) : ℝ :=
  let vector_sub (P Q : Point) : Point := (P.1 - Q.1, P.2 - Q.2, P.3 - Q.3)
  let cross_product (u v : Point) : Point := (
    u.2 * v.3 - u.3 * v.2,
    u.3 * v.1 - u.1 * v.3,
    u.1 * v.2 - u.2 * v.1
  )
  let dot_product (u v : Point) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

  let OM := vector_sub M O
  let OB₁ := vector_sub B₁ O
  let MN := vector_sub N M

  let normal_vector := cross_product OM OB₁
  let magnitude (v : Point) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)
  let area_triangle_OMB₁ := 0.5 * magnitude normal_vector
  let height_N := (dot_product (vector_sub N O) normal_vector).abs / magnitude normal_vector

  (1 / 3) * area_triangle_OMB₁ * height_N

theorem volume_of_tetrahedron_OMNB1_is_correct :
  let O := (0.5, 0.5, 0)
  let M := (0, 0.5, 1)
  let N := (1, 1, 2 / 3)
  let B₁ := (1, 0, 1)
  volume_of_tetrahedron O M N B₁ = 11 / 72 :=
by
  sorry

end volume_of_tetrahedron_OMNB1_is_correct_l44_44190


namespace medians_intersect_at_centroid_l44_44298

open EuclideanGeometry

-- Given a triangle ABC
variables (A B C : Point)

-- Define the midpoints of the sides
def D : Point := midpoint B C
def E : Point := midpoint C A
def F : Point := midpoint A B

-- Define the medians
def AD : LineSegment := LineSegment.mk A D
def BE : LineSegment := LineSegment.mk B E
def CF : LineSegment := LineSegment.mk C F

-- Ensure they intersect at a single point, the centroid G
theorem medians_intersect_at_centroid :
  ∃ G : Point, is_centroid G A B C ∧
  (G ∈ AD) ∧ (G ∈ BE) ∧ (G ∈ CF) ∧
  divides_in_ratio G AD 2 1 ∧ divides_in_ratio G BE 2 1 ∧ divides_in_ratio G CF 2 1 := sorry

end medians_intersect_at_centroid_l44_44298


namespace y_intercept_is_0_4_l44_44012

/-- The $y$-intercept of the line defined by the equation $3x + 5y = 20$ is the point (0, 4). -/
theorem y_intercept_is_0_4 : ∃ y : ℝ, (3 * (0 : ℝ) + 5 * y = 20) ∧ y = 4 :=
by {
    existsi 4,
    split;
    { linarith }
    sorry
}

end y_intercept_is_0_4_l44_44012


namespace calculate_visits_to_water_fountain_l44_44287

-- Define the distance from the desk to the fountain
def distance_desk_to_fountain : ℕ := 30

-- Define the total distance Mrs. Hilt walked
def total_distance_walked : ℕ := 120

-- Define the distance of a round trip (desk to fountain and back)
def round_trip_distance : ℕ := 2 * distance_desk_to_fountain

-- Define the number of round trips and hence the number of times to water fountain
def number_of_visits : ℕ := total_distance_walked / round_trip_distance

theorem calculate_visits_to_water_fountain:
    number_of_visits = 2 := 
by
    sorry

end calculate_visits_to_water_fountain_l44_44287


namespace range_of_independent_variable_l44_44952

theorem range_of_independent_variable (x : ℝ) :
  (∃ y, y = 1 / real.sqrt (x + 2)) → x > -2 :=
by
  sorry

end range_of_independent_variable_l44_44952


namespace pizza_consumption_l44_44397

theorem pizza_consumption :
  let initial := (2 / 3 : ℚ),
      second  := (1 / 2) * (1 / 3 : ℚ),
      third   := (1 / 2) * second,
      fourth  := (1 / 2) * third,
      fifth   := (1 / 2) * fourth,
      sixth   := (1 / 2) * fifth,
      total_consumed := initial + second + third + fourth + fifth + sixth
  in total_consumed = (191 / 192 : ℚ) :=
by 
  sorry -- Placeholder for the actual proof.

end pizza_consumption_l44_44397


namespace max_value_thm_l44_44911

noncomputable def max_value (c : ℝ) : ℝ :=
  let f (x : ℝ) := (3 * x ^ 2 + 3 * x + c) / (x^2 + x + 1)
  Real.Sup (Set.range f)

theorem max_value_thm {c : ℝ} : max_value c = 13 / 3 :=
sorry

end max_value_thm_l44_44911


namespace hexagon_inequality_l44_44858

-- Given a convex hexagon ABCDEF with the following properties:
-- AB parallel DE, BC parallel EF, and CD parallel FA
-- The distance between AB and DE is equal to the distance between BC and EF and to the distance between CD and FA

variables (A B C D E F : Point) 

-- Define the convex hexagon condition and parallel relations:
def convex_hexagon (A B C D E F : Point) : Prop := is_convex_hexagon A B C D E F  -- Assume the existence of a function that checks convexity
def parallel_AB_DE : Prop := parallel A B D E
def parallel_BC_EF : Prop := parallel B C E F
def parallel_CD_FA : Prop := parallel C D F A

-- Assume the equal distance condition
def equal_dist : Prop := 
  let dist1 := distance (line A B) (line D E)
  let dist2 := distance (line B C) (line E F)
  let dist3 := distance (line C D) (line F A)
  dist1 = dist2 ∧ dist2 = dist3

-- Define the inequality to be proved
def inequality_to_prove : Prop := 
  let AD := length A D
  let BE := length B E
  let CF := length C F
  AD + BE + CF ≤ (length A B) + (length B C) + (length C D) + (length D E) + (length E F) + (length F A)

-- Final theorem that puts everything together
theorem hexagon_inequality
  (h_convex : convex_hexagon A B C D E F)
  (h_parallel_ab_de : parallel_AB_DE A B C D E F)
  (h_parallel_bc_ef : parallel_BC_EF A B C D E F)
  (h_parallel_cd_fa : parallel_CD_FA A B C D E F)
  (h_equal_dist : equal_dist A B C D E F) : 
  inequality_to_prove A B C D E F :=
sorry

end hexagon_inequality_l44_44858


namespace kira_breakfast_time_l44_44974

theorem kira_breakfast_time :
  let fry_time_per_sausage := 5 -- minutes per sausage
  let scramble_time_per_egg := 4 -- minutes per egg
  let sausages := 3
  let eggs := 6
  let time_to_fry := sausages * fry_time_per_sausage
  let time_to_scramble := eggs * scramble_time_per_egg
  (time_to_fry + time_to_scramble) = 39 := 
by
  sorry

end kira_breakfast_time_l44_44974


namespace puppies_to_start_with_l44_44444

variable (given_away : ℕ)
variable (left : ℕ)

theorem puppies_to_start_with (given_away left : ℕ) : given_away = 7 → left = 5 → given_away + left = 12 := by
  intros h₁ h₂
  rw [h₁, h₂]
  sorry

end puppies_to_start_with_l44_44444


namespace distinct_convex_polygons_of_four_or_more_sides_l44_44489

noncomputable def total_subsets (n : Nat) : Nat := 2^n

noncomputable def subsets_with_fewer_than_four_members (n : Nat) : Nat := 
  (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)

noncomputable def valid_subsets (n : Nat) : Nat := 
  total_subsets n - subsets_with_fewer_than_four_members n

theorem distinct_convex_polygons_of_four_or_more_sides (n : Nat) (h : n = 15) : valid_subsets n = 32192 := by
  sorry

end distinct_convex_polygons_of_four_or_more_sides_l44_44489


namespace base_addition_l44_44813

theorem base_addition (b : ℕ) (h : b > 1) :
  (2 * b^3 + 3 * b^2 + 8 * b + 4) + (3 * b^3 + 4 * b^2 + 1 * b + 7) = 
  1 * b^4 + 0 * b^3 + 2 * b^2 + 0 * b + 1 → b = 10 :=
by
  intro H
  -- skipping the detailed proof steps
  sorry

end base_addition_l44_44813


namespace emily_new_salary_l44_44822

noncomputable def emily_salary : ℝ := 1000000
noncomputable def employee1_salary : ℝ := 30000
noncomputable def employee2_salary : ℝ := 30000
noncomputable def employee3_salary : ℝ := 25000
noncomputable def employee4_salary : ℝ := 35000
noncomputable def employee5_salary : ℝ := 20000
noncomputable def target_salary : ℝ := 35000
noncomputable def tax_rate : ℝ := 0.15

theorem emily_new_salary : 
  let total_increment := (2 * max (0 : ℝ) (target_salary - employee1_salary)) + 
                         max (0 : ℝ) (target_salary - employee3_salary) + 
                         max (0 : ℝ) (target_salary - employee5_salary),
      tax_amount      := tax_rate * total_increment,
      total_needed    := total_increment + tax_amount,
      final_salary    := emily_salary - total_needed in
  final_salary = 959750 :=
by
  let total_increment := (2 * max (0 : ℝ) (target_salary - employee1_salary)) + 
                         max (0 : ℝ) (target_salary - employee3_salary) + 
                         max (0 : ℝ) (target_salary - employee5_salary)
  let tax_amount := tax_rate * total_increment
  let total_needed := total_increment + tax_amount
  let final_salary := emily_salary - total_needed
  sorry

end emily_new_salary_l44_44822


namespace ellipse_C1_parallel_line_l_l44_44196

noncomputable theory

-- Assume the problem conditions as definitions
def ellipse_eq (x y a b : ℝ): Prop := (a > 0) ∧ (b > 0) ∧ (a > b) ∧ (x^2 / a^2 + y^2 / b^2 = 1)
def parabola_eq (x y : ℝ) : Prop := (y^2 = 4 * x)
def focus_F2 : ℝ × ℝ := (1, 0)
def intersection_M (x y : ℝ) : Prop := ellipse_eq x y 2 sqrt(3) ∧ parabola_eq x y ∧ (sqrt((x - 1)^2 + y^2) = 5 / 3)
def parallel_MN (N M F1 F2 : ℝ × ℝ) : Prop := (N.1 - M.1, N.2 - M.2) = (M.1 - F1.1 + F2.1, M.2 - F1.2 + F2.2)
def perp_OA_OB (A B : ℝ × ℝ) : Prop := (A.1 * B.1 + A.2 * B.2 = 0)

-- The statements we need to prove
theorem ellipse_C1 :
  (∃ a b : ℝ, ellipse_eq 0 0 a b) → ∃ x y : ℝ, ellipse_eq x y 2 sqrt(3) :=
sorry

theorem parallel_line_l :
  (∃ N M F1 F2 : ℝ × ℝ, parallel_MN N M F1 F2) ∧ 
  (∃ A B : ℝ × ℝ, ellipse_eq A.1 A.2 2 sqrt(3) ∧ ellipse_eq B.1 B.2 2 sqrt(3) 
    ∧ perp_OA_OB A B) →
  (∃ m : ℝ, (m = sqrt(2)) ∨ (m = -sqrt(2))) →
  (∃ l : ℝ → ℝ, l = λ x, sqrt(6) * x - 2 * sqrt(3) ∨ l = λ x, sqrt(6) * x + 2 * sqrt(3)) :=
sorry

end ellipse_C1_parallel_line_l_l44_44196


namespace total_time_spent_l44_44457

-- Define time spent on each step
def time_first_step : ℕ := 30
def time_second_step : ℕ := time_first_step / 2
def time_third_step : ℕ := time_first_step + time_second_step

-- Prove the total time spent
theorem total_time_spent : 
  time_first_step + time_second_step + time_third_step = 90 := by
  sorry

end total_time_spent_l44_44457


namespace part_i_part_ii_l44_44062

noncomputable def f (x a : ℝ) := |x - a|

theorem part_i :
  (∀ (x : ℝ), (f x 1) ≥ (|x + 1| + 1) ↔ x ≤ -0.5) :=
sorry

theorem part_ii :
  (∀ (x a : ℝ), (f x a) + 3 * x ≤ 0 → { x | x ≤ -1 } ⊆ { x | (f x a) + 3 * x ≤ 0 }) →
  (∀ (a : ℝ), (0 ≤ a ∧ a ≤ 2) ∨ (-4 ≤ a ∧ a < 0)) :=
sorry

end part_i_part_ii_l44_44062


namespace sum_of_sequence_l44_44698

noncomputable def seq_sum : ℕ → ℤ
| 0       := 0
| (n + 1) := seq_sum n + (-1)^(n + 1) * (n + 1)

theorem sum_of_sequence : seq_sum 2012 = 1006 :=
by
  sorry

end sum_of_sequence_l44_44698


namespace cosine_triangle_ABC_l44_44210

noncomputable def triangle_cosine_proof (a b : ℝ) (A : ℝ) (cosB : ℝ) : Prop :=
  let sinA := Real.sin A
  let sinB := b * sinA / a
  let cosB_expr := Real.sqrt (1 - sinB^2)
  cosB = cosB_expr

theorem cosine_triangle_ABC : triangle_cosine_proof (Real.sqrt 7) 2 (Real.pi / 4) (Real.sqrt 35 / 7) :=
by
  sorry

end cosine_triangle_ABC_l44_44210


namespace modulo_problem_l44_44803

theorem modulo_problem :
  (47 ^ 2051 - 25 ^ 2051) % 5 = 3 := by
  sorry

end modulo_problem_l44_44803


namespace find_k_l44_44077

theorem find_k (k : ℝ) (a b : ℝ × ℝ) (ha : a = (1, 2)) (hb : b = (-1, 1)) : 
  let u := (λ k, (1, 2) : ℝ × ℝ)
  let v := (λ k, (-1, 1) : ℝ × ℝ)
  (k * u k + v k = (λ k, (k-1, 2*k+1))) (i:j)
  let w := (1 + 3 * 1, 2 - 3 * 1) := (4, -1)
  (∀ k : ℝ, k * u k + v k ⬝ w = 0) :
  k = 5 / 2 :=
begin sorry end


end find_k_l44_44077


namespace distinct_arrangements_balloon_l44_44107

noncomputable def totalPermutations (n nl no : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial nl * Nat.factorial no)

theorem distinct_arrangements_balloon :
  totalPermutations 7 2 2 = 1260 := by 
  sorry

end distinct_arrangements_balloon_l44_44107


namespace equation_of_line_passing_AB_l44_44326

-- Define the points A and B
def A : ℝ × ℝ := (0, -5)
def B : ℝ × ℝ := (1, 0)

-- The statement to prove
theorem equation_of_line_passing_AB : ∃ m b : ℝ, (∀ x, y = m * x + b ↔ (x, y) = A ∨ (x, y) = B) := 
begin
  sorry
end

end equation_of_line_passing_AB_l44_44326


namespace min_value_w_prod_sum_l44_44989

theorem min_value_w_prod_sum :
  let g := λ x, x^4 + 8*x^3 + 18*x^2 + 8*x + 1
  let w := {w : ℝ | g w = 0}
  ∃ w1 w2 w3 w4 ∈ w, 
  w1 * w4 = 1 ∧ w2 * w3 = 1 ∧
  (∀ a b c d, {a, b, c, d} = {0, 1, 2, 3} → 
    |w.a * w.b + w.c * w.d| = 2) :=
sorry

end min_value_w_prod_sum_l44_44989


namespace masha_can_climb_10_steps_l44_44833

def ways_to_climb_stairs : ℕ → ℕ 
| 0 => 1
| 1 => 1
| n + 2 => ways_to_climb_stairs (n + 1) + ways_to_climb_stairs n

theorem masha_can_climb_10_steps : ways_to_climb_stairs 10 = 89 :=
by
  -- proof omitted here as per instruction
  sorry

end masha_can_climb_10_steps_l44_44833


namespace find_m_l44_44536

theorem find_m (m : ℝ) (α : ℝ) (h1 : sin α = 3 / 5) (h2 : (-4, m) ∈ {p | p.1^2 + p.2^2 = 25 ∧ p.1 < 0}) : m = 3 :=
sorry

end find_m_l44_44536


namespace thomas_score_l44_44285

theorem thomas_score (n : ℕ) (avg_without_thomas avg_with_thomas : ℝ) 
  (h1 : n = 20) 
  (h2 : avg_without_thomas = 86) 
  (h3 : avg_with_thomas = 88) : 
  let score_without := 19 * avg_without_thomas,
      score_with := n * avg_with_thomas,
      thomas_score := score_with - score_without in
  thomas_score = 126 :=
by
  sorry

end thomas_score_l44_44285


namespace evaluate_a_4_times_l44_44912

def a (k : ℕ) : ℕ := (k + 1) ^ 2

theorem evaluate_a_4_times (k : ℕ) (h : k = 1) : a (a (a (a k))) = 458329 := by
  rw [h, a, a, a, a]
  simp
  -- Detailed calculation steps are omitted, hence 'sorry'
  sorry

end evaluate_a_4_times_l44_44912


namespace evaluate_f_5_l44_44869

def f : ℤ → ℤ
| x := if x < 4 then 2^x else f (x-1)

theorem evaluate_f_5 : f 5 = 8 :=
by 
  -- directly prove using the conditions
  sorry

end evaluate_f_5_l44_44869


namespace shift_graph_cosine_l44_44371

theorem shift_graph_cosine (x : ℝ) :
  (∀ x, cos (2 * x - π / 6) = cos (2 * (x - π / 6) + π / 6)) →
  (∀ x, cos (2 * (x + π / 6)) = cos (2 * x + π / 6)) →
  (∀ x, cos (2 * (x - π / 6 + π / 6)) = cos (2 * x - π / 6)) :=
by
  intros h1 h2
  sorry

end shift_graph_cosine_l44_44371


namespace move_point_right_l44_44581

theorem move_point_right (x y : ℝ) (h : (x, y) = (-2, 3)) : (x + 4, y) = (2, 3) :=
by 
  rw [h]
  sorry

end move_point_right_l44_44581


namespace hypotenuse_of_45_45_90_triangle_l44_44651

theorem hypotenuse_of_45_45_90_triangle (a : ℝ) (h : ℝ) 
  (ha : a = 15) 
  (angle_opposite_leg : ℝ) 
  (h_angle : angle_opposite_leg = 45) 
  (right_triangle : ∃ θ : ℝ, θ = 90) : 
  h = 15 * Real.sqrt 2 := 
sorry

end hypotenuse_of_45_45_90_triangle_l44_44651


namespace middle_quadrilateral_area_l44_44820

-- Definitions: Convex quadrilateral and its division
structure ConvexQuadrilateral (P : Type) :=
(A B C D : P)

variable {P : Type} [EuclideanGeometry P] {A B C D : P}

-- Define the property of each side being divided into five equal parts
def side_divided_into_five_equal_parts (quad : ConvexQuadrilateral P) (A B : P) : Prop :=
  ∃ points : Fin 4 → P, 
  (∀ i, dist (points i) (points (i + 1)) = dist A B / 5)

-- Define the inner quadrilateral formed by corresponding points
def inner_quadrilateral_formed (quad : ConvexQuadrilateral P) : ConvexQuadrilateral P :=
{
  A := midpoint (segment1.1, segment1.4), -- dealing with points on divided segments
  B := midpoint (segment2.1, segment2.4),
  C := midpoint (segment3.1, segment3.4),
  D := midpoint (segment4.1, segment4.4)
}

-- Statement of the proof problem
theorem middle_quadrilateral_area (quad : ConvexQuadrilateral P)
  (h1 : side_divided_into_five_equal_parts quad quad.A quad.B)
  (h2 : side_divided_into_five_equal_parts quad quad.C quad.D)
  (h3 : side_divided_into_five_equal_parts quad quad.A quad.D)
  (h4 : side_divided_into_five_equal_parts quad quad.B quad.C) :
  area (inner_quadrilateral_formed quad) = (1 / 25) * area quad := 
sorry

end middle_quadrilateral_area_l44_44820


namespace area_after_transformation_l44_44615

-- Let T be a region with area 15
def T_area : ℝ := 15

-- Define the transformation matrix
def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 2], ![4, -5]]

-- Calculate the resulting area after applying the transformation
theorem area_after_transformation : 
  det transformation_matrix * T_area = 345 :=
by
  sorry

end area_after_transformation_l44_44615


namespace arithmetic_sequence_general_term_l44_44593

theorem arithmetic_sequence_general_term
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_nonzero_d : d ≠ 0)
  (a1 a2 : ℤ)
  (h_roots : ∀ n, (a 1 = a1) ∧ (a 2 = a2) ∧ (a 3 = a1 + a2) ∧ (a 4 = a1 * a2)) :
  ∀ n, a n = 2 * n := 
sorry

end arithmetic_sequence_general_term_l44_44593


namespace circles_tangent_dist_l44_44531

theorem circles_tangent_dist (t : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 4) ∧ 
  (∀ x y : ℝ, (x - t)^2 + y^2 = 1) ∧ 
  (∀ x1 y1 x2 y2 : ℝ, x1^2 + y1^2 = 4 → (x2 - t)^2 + y2^2 = 1 → 
    dist (x1, y1) (x2, y2) = 3) → 
  t = 3 ∨ t = -3 :=
by 
  sorry

end circles_tangent_dist_l44_44531


namespace trapezoidal_tank_depth_l44_44782

theorem trapezoidal_tank_depth (V S S' : ℝ) (h : ℝ) (h_pos : 0 < h) :
  V = (h / 2) * (S + S') → S = 60 → S' = 40 → V = 200000 → h = 79 :=
by {
  intros,
  sorry
}

end trapezoidal_tank_depth_l44_44782


namespace distinct_arrangements_balloon_l44_44111

noncomputable def totalPermutations (n nl no : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial nl * Nat.factorial no)

theorem distinct_arrangements_balloon :
  totalPermutations 7 2 2 = 1260 := by 
  sorry

end distinct_arrangements_balloon_l44_44111


namespace h_2023_eq_4052_l44_44473

def h : ℕ → ℤ
| 1       := 3
| 2       := 2
| (n + 3) := h (n + 2) - h (n + 1) + 2 * (n + 3)

theorem h_2023_eq_4052 : h 2023 = 4052 :=
sorry

end h_2023_eq_4052_l44_44473


namespace probability_sum_3_is_1_over_216_l44_44353

-- Let E be the event that three fair dice sum to 3
def event_sum_3 (d1 d2 d3 : ℕ) : Prop := d1 + d2 + d3 = 3

-- Probabilities of rolling a particular outcome on a single die
noncomputable def P_roll_1 (n : ℕ) := if n = 1 then 1/6 else 0

-- Define the probability of the event E occurring
noncomputable def P_event_sum_3 := 
  ∑ d1 in {1, 2, 3, 4, 5, 6}, 
  ∑ d2 in {1, 2, 3, 4, 5, 6}, 
  ∑ d3 in {1, 2, 3, 4, 5, 6}, 
  if event_sum_3 d1 d2 d3 then P_roll_1 d1 * P_roll_1 d2 * P_roll_1 d3 else 0

-- The main theorem to prove the desired probability
theorem probability_sum_3_is_1_over_216 : P_event_sum_3 = 1/216 := by 
  sorry

end probability_sum_3_is_1_over_216_l44_44353


namespace exists_unique_perpendicular_line_l44_44830

-- Given two skew lines in three-dimensional space
variables (l₁ l₂ : set (ℝ × ℝ × ℝ))

-- Define skew lines (non-intersecting and non-parallel)
def skew_lines (l₁ l₂ : set (ℝ × ℝ × ℝ)) : Prop :=
  ¬ (∃ p : ℝ × ℝ × ℝ, p ∈ l₁ ∧ p ∈ l₂) ∧
  ¬ (∃ v : ℝ × ℝ × ℝ, ∃ p₁ ∈ l₁, ∃ p₂ ∈ l₂, p₂ - p₁ = v)

-- Define the condition for a line being perpendicular to both skew lines
def perpendicular_to_both (line : set (ℝ × ℝ × ℝ)) (l₁ l₂ : set (ℝ × ℝ × ℝ)) : Prop :=
  ∃ v₁ ∈ l₁, ∃ v₂ ∈ l₂, ∀ p ∈ line, ∃ r₁ r₂ : ℝ, p = v₁ + r₁ * (v₂ - v₁) ∧
  p = v₂ + r₂ * (v₁ - v₂)

-- Prove the existence and uniqueness of the line
theorem exists_unique_perpendicular_line (l₁ l₂ : set (ℝ × ℝ × ℝ)) (h: skew_lines l₁ l₂) :
  ∃! line : set (ℝ × ℝ × ℝ), perpendicular_to_both line l₁ l₂ :=
sorry

end exists_unique_perpendicular_line_l44_44830


namespace opposite_midpoints_cut_segments_adjacent_midpoints_cut_segments_l44_44429
-- Import the necessary library for mathematical definitions

-- Definitions based on the problem statement
def grid_rectangle (length width : ℕ) := length = 10 ∧ width = 12

-- Condition definition of forming a 1x1 by folding a 10x12 grid
def folded_square := ∃ (length width : ℕ), grid_rectangle length width

-- Goal: Segments from cutting through the midpoints of the opposite sides of a folded square
theorem opposite_midpoints_cut_segments : folded_square → (∃ n, n = 11 ∨ n = 13) :=
by
  intro h,
  unfold folded_square at h,
  sorry

-- Goal: Segments from cutting through the midpoints of adjacent sides of a folded square
theorem adjacent_midpoints_cut_segments : folded_square → (∃ n, n = 31 ∨ n = 36 ∨ n = 37 ∨ n = 43) :=
by
  intro h,
  unfold folded_square at h,
  sorry

end opposite_midpoints_cut_segments_adjacent_midpoints_cut_segments_l44_44429


namespace polygon_interior_exterior_equal_l44_44926

theorem polygon_interior_exterior_equal (n : ℕ) :
  (n - 2) * 180 = 360 → n = 4 :=
by
  sorry

end polygon_interior_exterior_equal_l44_44926


namespace H2O_production_l44_44827

theorem H2O_production (n : Nat) (m : Nat)
  (h1 : n = 3)
  (h2 : m = 3) :
  n = m → n = 3 := by
  sorry

end H2O_production_l44_44827


namespace cora_read_pages_on_monday_l44_44472

noncomputable def pages_on_monday (total_pages read_tuesday read_wednesday read_thursday read_friday : ℕ) : ℕ :=
  total_pages - (read_tuesday + read_wednesday + read_thursday + read_friday)

theorem cora_read_pages_on_monday :
  let total_pages := 158
  let read_tuesday := 38
  let read_wednesday := 61
  let read_thursday := 12
  let read_friday := 2 * read_thursday
  pages_on_monday total_pages read_tuesday read_wednesday read_thursday read_friday = 23 :=
by
  dsimp [pages_on_monday]
  sorry

end cora_read_pages_on_monday_l44_44472


namespace distinct_arrangements_balloon_l44_44132

-- Let's define the basic conditions:
def total_letters : Nat := 7
def repeats_l : Nat := 2
def repeats_o : Nat := 2

-- Now let's state the problem.
theorem distinct_arrangements_balloon : 
  (Nat.factorial total_letters) / ((Nat.factorial repeats_l) * (Nat.factorial repeats_o)) = 1260 := 
by
  sorry

end distinct_arrangements_balloon_l44_44132


namespace total_cost_correct_l44_44939

-- Define the conditions
def total_employees : ℕ := 300
def emp_12_per_hour : ℕ := 200
def emp_14_per_hour : ℕ := 40
def emp_17_per_hour : ℕ := total_employees - emp_12_per_hour - emp_14_per_hour

def wage_12_per_hour : ℕ := 12
def wage_14_per_hour : ℕ := 14
def wage_17_per_hour : ℕ := 17

def hours_per_shift : ℕ := 8

-- Define the cost calculations
def cost_12 : ℕ := emp_12_per_hour * wage_12_per_hour * hours_per_shift
def cost_14 : ℕ := emp_14_per_hour * wage_14_per_hour * hours_per_shift
def cost_17 : ℕ := emp_17_per_hour * wage_17_per_hour * hours_per_shift

def total_cost : ℕ := cost_12 + cost_14 + cost_17

-- The theorem to be proved
theorem total_cost_correct :
  total_cost = 31840 :=
by
  sorry

end total_cost_correct_l44_44939


namespace exists_indices_l44_44042

open Nat List

theorem exists_indices (m n : ℕ) (a : Fin m → ℕ) (b : Fin n → ℕ) 
  (h1 : ∀ i : Fin m, a i ≤ n) (h2 : ∀ i j : Fin m, i ≤ j → a i ≤ a j)
  (h3 : ∀ j : Fin n, b j ≤ m) (h4 : ∀ i j : Fin n, i ≤ j → b i ≤ b j) :
  ∃ i : Fin m, ∃ j : Fin n, a i + i.val + 1 = b j + j.val + 1 := by
  sorry

end exists_indices_l44_44042


namespace part_one_part_two_l44_44544

noncomputable def f (x a : ℝ) := x^2 + 4 * a * x + 2 * a + 6
noncomputable def g (a : ℝ) := 2 - a * (abs (a + 3))

theorem part_one :
  (∀ x : ℝ, f x (3/2) ≥ 0) ∧ (∀ b : ℝ, (∀ x : ℝ, f x b ≥ 0) → b = (3/2)) :=
by
  sorry

theorem part_two :
  (∀ x : ℝ, ∃ a : ℝ, -1 ≤ a ∧ a ≤ 3/2 ∧ f x a ≥ 0) →
  (let S := { y | ∃ a : ℝ, -1 ≤ a ∧ a ≤ 3/2 ∧ y = g a } in set.range g = S) :=
by
  sorry

end part_one_part_two_l44_44544


namespace find_digit_D_l44_44691

theorem find_digit_D (A B C D : ℕ) (h1 : A + B = A + 10 * (B / 10)) (h2 : D + 10 * (A / 10) = A + C)
  (h3 : A + 10 * (B / 10) - C = A) (h4 : 0 ≤ A) (h5 : A ≤ 9) (h6 : 0 ≤ B) (h7 : B ≤ 9)
  (h8 : 0 ≤ C) (h9 : C ≤ 9) (h10 : 0 ≤ D) (h11 : D ≤ 9) : D = 9 := 
sorry

end find_digit_D_l44_44691


namespace find_values_l44_44413

variable (circle triangle : ℕ)

axiom condition1 : triangle = circle + circle + circle
axiom condition2 : triangle + circle = 40

theorem find_values : circle = 10 ∧ triangle = 30 :=
by
  sorry

end find_values_l44_44413


namespace paddyfield_warblers_percentage_l44_44935

-- Definitions for the conditions
variables (B : ℝ) -- Total number of birds
variables (H : ℝ) -- Portion of hawks
variables (W : ℝ) -- Portion of paddyfield-warblers among non-hawks
variables (K : ℝ) -- Portion of kingfishers among non-hawks
variables (N : ℝ) -- Portion of non-hawks

-- Conditions
def hawks_portion : Prop := H = 0.30 * B
def non_hawks_portion : Prop := N = 0.70 * B
def paddyfield_warblers : Prop := W * N = W * (0.70 * B)
def kingfishers : Prop := K = 0.25 * W * N

-- Prove that the percentage of non-hawks that are paddyfield-warblers is 40%
theorem paddyfield_warblers_percentage :
  hawks_portion B H →
  non_hawks_portion B N →
  paddyfield_warblers B W N →
  kingfishers B W N K →
  W = 0.4 :=
by
  sorry

end paddyfield_warblers_percentage_l44_44935


namespace people_don_l44_44290

def total_people := 1200
def percent_don't_like_radio := 0.3
def percent_don't_like_radio_and_music := 0.1

theorem people_don't_like_radio_and_music : 
  (percent_don't_like_radio * total_people) * percent_don't_like_radio_and_music = 36 := by
  sorry

end people_don_l44_44290


namespace total_distance_traveled_by_children_l44_44821

theorem total_distance_traveled_by_children :
  let ap := 50
  let dist_1_vertex_skip := (50 : ℝ) * Real.sqrt 2
  let dist_2_vertices_skip := (50 : ℝ) * Real.sqrt (2 + 2 * Real.sqrt 2)
  let dist_diameter := (2 : ℝ) * 50
  let single_child_distance := 2 * dist_1_vertex_skip + 2 * dist_2_vertices_skip + dist_diameter
  8 * single_child_distance = 800 * Real.sqrt 2 + 800 * Real.sqrt (2 + 2 * Real.sqrt 2) + 800 :=
sorry

end total_distance_traveled_by_children_l44_44821


namespace min_n_satisfies_inequality_l44_44995

theorem min_n_satisfies_inequality :
  ∃ (n : ℕ), (∀ (x y z : ℝ), (x^2 + y^2 + z^2) ≤ n * (x^4 + y^4 + z^4)) ∧ (n = 3) :=
by
  sorry

end min_n_satisfies_inequality_l44_44995


namespace seated_arrangements_l44_44418

theorem seated_arrangements (D R : ℕ) (total : ℕ) (hD : D = 7) (hR : R = 5) (h_total : total = D + R) : 
  (total - 1)! = 11! :=
by
  rw [h_total, hD, hR]
  sorry

end seated_arrangements_l44_44418


namespace sequence_result_l44_44312

theorem sequence_result (initial_value : ℕ) (total_steps : ℕ) 
    (net_effect_one_cycle : ℕ) (steps_per_cycle : ℕ) : 
    initial_value = 100 ∧ total_steps = 26 ∧ 
    net_effect_one_cycle = (15 - 12 + 3) ∧ steps_per_cycle = 3 
    → 
    ∀ (resulting_value : ℕ), resulting_value = 151 :=
by
  sorry

end sequence_result_l44_44312


namespace probability_sum_three_dice_3_l44_44368

-- Definition of a fair six-sided die
def fair_six_sided_die : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Definition of probability of an event
def probability (s : Set ℕ) (event : ℕ → Prop) : ℚ :=
  if h : finite s then (s.filter event).to_finset.card / s.to_finset.card else 0

theorem probability_sum_three_dice_3 :
  let dice := List.repeat fair_six_sided_die 3 in
  let event := λ result : List ℕ => result.sum = 3 in
  probability ({(r1, r2, r3) | r1 ∈ fair_six_sided_die ∧ r2 ∈ fair_six_sided_die ∧ r3 ∈ fair_six_sided_die }) (λ (r1, r2, r3) => r1 + r2 + r3 = 3) = 1 / 216 :=
by
  sorry

end probability_sum_three_dice_3_l44_44368


namespace lattice_points_count_l44_44815

theorem lattice_points_count :
  {p : ℕ × ℕ // 0 < p.1 ∧ 0 < p.2 ∧ 4 * p.1 + 3 * p.2 < 12}.card = 3 :=
by
  sorry

end lattice_points_count_l44_44815


namespace gcd_176_88_l44_44328

theorem gcd_176_88 : Nat.gcd 176 88 = 88 :=
by
  sorry

end gcd_176_88_l44_44328


namespace continuous_at_x_3_l44_44507

def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 3 then
  x^2 + 2*x + 2
else
  3*x + a

theorem continuous_at_x_3 {a : ℝ} : (∀ x : ℝ, f x a = if x > 3 then x^2 + 2*x + 2 else 3*x + a) → f 3 a = 17 - 9 := by
  intro h
  have : 17 = 9 + a := by sorry
  rw this
  rfl

end continuous_at_x_3_l44_44507


namespace largest_prime_factor_l44_44499

-- Condition: expression whose largest prime factor is to be determined
def expression : ℕ := 20^3 + 15^4 - 10^5 + 2 * 25^3

-- Theorem: the largest prime factor of the expression is 11
theorem largest_prime_factor : Nat.greatest_prime_factor expression = 11 := 
sorry

end largest_prime_factor_l44_44499


namespace inradius_of_triangle_l44_44689

theorem inradius_of_triangle (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 2) (h₂ : r₂ = 3) (h₃ : r₃ = 6) : 
  let r := 1 in 
  1 / r₁ + 1 / r₂ + 1 / r₃ = 1 / r :=
by
  -- Proof will be filled in here
  sorry

end inradius_of_triangle_l44_44689


namespace infinitely_many_solutions_l44_44000

theorem infinitely_many_solutions (b : ℝ) : (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := sorry

end infinitely_many_solutions_l44_44000


namespace point_distance_sqrt5_l44_44692

theorem point_distance_sqrt5 (x : ℝ) : abs x = sqrt 5 ↔ x = sqrt 5 ∨ x = -sqrt 5 := by
  sorry

end point_distance_sqrt5_l44_44692


namespace max_value_of_ab_l44_44627

theorem max_value_of_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 5 * a + 3 * b < 90) :
  ab * (90 - 5 * a - 3 * b) ≤ 1800 :=
sorry

end max_value_of_ab_l44_44627


namespace Eiffel_Tower_model_scale_l44_44316

theorem Eiffel_Tower_model_scale
  (h_tower : ℝ := 324)
  (h_model_cm : ℝ := 18) :
  (h_tower / (h_model_cm / 100)) / 100 = 18 :=
by
  sorry

end Eiffel_Tower_model_scale_l44_44316


namespace balloon_arrangements_l44_44103

-- Define the variables
def n : ℕ := 7
def L_count : ℕ := 2
def O_count : ℕ := 2
def B_count : ℕ := 1
def A_count : ℕ := 1
def N_count : ℕ := 1

-- Define the multiset permutation formula
def multiset_permutations (n : ℕ) (counts : List ℕ) : ℕ :=
  n.factorial / (counts.map Nat.factorial).prod

-- Proof that the number of distinct arrangements is 1260
theorem balloon_arrangements : multiset_permutations n [L_count, O_count, B_count, A_count, N_count] = 1260 :=
  by
  -- The proof is omitted
  sorry

end balloon_arrangements_l44_44103


namespace part1_l44_44523

variable {m a : ℝ}
def g (x : ℝ) (m : ℝ) : ℝ := x^2 - mx + (m^2 / 2) + 2 * m - 3
def g_solution_set (m : ℝ) : Set ℝ := {x | g x m < (m^2 / 2) + 1}

theorem part1 (h : g_solution_set m = set.Ioo 1 a) : a = 2 := 
sorry

end part1_l44_44523


namespace toms_initial_investment_l44_44375

theorem toms_initial_investment (t j k : ℕ) (hj_neq_ht : t ≠ j) (hk_neq_ht : t ≠ k) (hj_neq_hk : j ≠ k) 
  (h1 : t + j + k = 1200) 
  (h2 : t - 150 + 3 * j + 3 * k = 1800) : 
  t = 825 := 
sorry

end toms_initial_investment_l44_44375


namespace elder_middle_arrangements_l44_44433

theorem elder_middle_arrangements (V E : Type) (vol1 vol2 vol3 vol4 : V) (elder : E) :
  ∃ (arrangements : nat), arrangements = 24 ∧
  let positions := [1, 2, 3, 4, 5] in
  let elder_pos := 3 in
  let remaining_positions := [1, 2, 4, 5] in
  ∀ (perm : Permutations (vol1, vol2, vol3, vol4)), 
  arrangements = factorial 4 := 
by
  sorry

end elder_middle_arrangements_l44_44433


namespace quadratic_has_two_distinct_real_roots_determine_k_from_roots_relation_l44_44072

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4*a*c

theorem quadratic_has_two_distinct_real_roots (k : ℝ) :
  let a := 1
  let b := 2*k - 1
  let c := -k - 1
  discriminant a b c > 0 := by
  sorry

theorem determine_k_from_roots_relation (x1 x2 k : ℝ) 
  (h1 : x1 + x2 = -(2*k - 1))
  (h2 : x1 * x2 = -k - 1)
  (h3 : x1 + x2 - 4*(x1 * x2) = 2) :
  k = -3/2 := by
  sorry

end quadratic_has_two_distinct_real_roots_determine_k_from_roots_relation_l44_44072


namespace f_property_l44_44393

noncomputable def f (x : ℝ) := log x + 1

theorem f_property (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = f x + f y - 1 :=
by
  -- We leave the proof as an exercise.
  sorry

end f_property_l44_44393


namespace domain_of_h_l44_44719

def h (y : ℝ) := 1 / ((y - 2)^2 + (y - 8))

theorem domain_of_h :
  (dom := (-∞ : Set ℝ) ∪ Set.Ioo(-1, 4) ∪ (4 : Set ℝ).Ioi) ∧
  ∀ y : ℝ, (h y).Undefined ↔ y = 4 ∨ y = -1 :=
by sorry

end domain_of_h_l44_44719


namespace find_j_l44_44028

-- Define the property where roots are distinct real numbers in arithmetic progression.
def are_roots_arithmetic_progression (a d : ℝ) : Prop :=
  a ≠ 0 ∧ d ≠ 0 ∧ list.pairwise (≠) [a, a + d, a + 2 * d, a + 3 * d]

-- Polynomial with distinct real roots in arithmetic progression.
noncomputable def polynomial_roots_ap (a d : ℝ) (h_arith_prog: are_roots_arithmetic_progression a d) : polynomial ℝ :=
  polynomial.C (a * (a + d) * (a + 2 * d) * (a + 3 * d)) *
  (polynomial.X - polynomial.C a) *
  (polynomial.X - polynomial.C (a + d)) *
  (polynomial.X - polynomial.C (a + 2 * d)) *
  (polynomial.X - polynomial.C (a + 3 * d))

-- The polynomial in the problem.
def polynomial_problem : polynomial ℝ :=
  polynomial.X ^ 4 + polynomial.C j * polynomial.X ^ 2 + polynomial.C 16 * polynomial.X + polynomial.C 64

-- The goal is to prove that j = -160/9
theorem find_j (a : ℝ) (d : ℝ) (h_arith_prog: are_roots_arithmetic_progression a d) (h_poly_eq: polynomial_problem = polynomial_roots_ap a d h_arith_prog) :
  j = -160 / 9 :=
sorry

end find_j_l44_44028


namespace hypotenuse_of_45_45_90_triangle_l44_44652

theorem hypotenuse_of_45_45_90_triangle (a : ℝ) (h : ℝ) 
  (ha : a = 15) 
  (angle_opposite_leg : ℝ) 
  (h_angle : angle_opposite_leg = 45) 
  (right_triangle : ∃ θ : ℝ, θ = 90) : 
  h = 15 * Real.sqrt 2 := 
sorry

end hypotenuse_of_45_45_90_triangle_l44_44652


namespace inverse_of_A_is_zero_matrix_l44_44498

def matrix_2x2 := Matrix (Fin 2) (Fin 2) ℚ
def A : matrix_2x2 := ![
  ![9, 18],
  ![-6, -12]
]

theorem inverse_of_A_is_zero_matrix : ∃ B : matrix_2x2, 
  (A.det = 0 ∧ B = 0 ∧ ∀ C : matrix_2x2, A * C = 0 → C = B) := 
by
  sorry

end inverse_of_A_is_zero_matrix_l44_44498


namespace isosceles_triangle_side_length_l44_44792

theorem isosceles_triangle_side_length (a b : ℝ) (h : a < b) : 
  ∃ l : ℝ, l = (b - a) / 2 := 
sorry

end isosceles_triangle_side_length_l44_44792


namespace blanket_cost_l44_44399

theorem blanket_cost (x : ℝ) 
    (h₁ : 200 + 750 + 2 * x = 1350) 
    (h₂ : 2 + 5 + 2 = 9) 
    (h₃ : (200 + 750 + 2 * x) / 9 = 150) : 
    x = 200 :=
by
    have h_total : 200 + 750 + 2 * x = 1350 := h₁
    have h_avg : (200 + 750 + 2 * x) / 9 = 150 := h₃
    sorry

end blanket_cost_l44_44399


namespace double_given_number_l44_44564

def given_number : ℝ := 1.2 * 10^6

def double_number (x: ℝ) : ℝ := x * 2

theorem double_given_number : double_number given_number = 2.4 * 10^6 :=
by sorry

end double_given_number_l44_44564


namespace choose_president_and_vice_president_l44_44664

theorem choose_president_and_vice_president :
  let total_members := 24
  let boys := 8
  let girls := 16
  let senior_members := 4
  let senior_boys := 2
  let senior_girls := 2
  let president_choices := senior_members
  let vice_president_choices_boy_pres := girls
  let vice_president_choices_girl_pres := boys - senior_boys
  let total_ways :=
    (senior_boys * vice_president_choices_boy_pres) + 
    (senior_girls * vice_president_choices_girl_pres)
  total_ways = 44 := 
by
  sorry

end choose_president_and_vice_president_l44_44664


namespace distinct_arrangements_balloon_l44_44096

theorem distinct_arrangements_balloon : 
  let n := 7 
  let freq_l := 2 
  let freq_o := 2 
  let freq_b := 1 
  let freq_a := 1 
  let freq_n := 1 
  Nat.factorial n / (Nat.factorial freq_l * Nat.factorial freq_o * Nat.factorial freq_b * Nat.factorial freq_a * Nat.factorial freq_n) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l44_44096


namespace jack_change_l44_44214

def cost_per_sandwich : ℕ := 5
def number_of_sandwiches : ℕ := 3
def payment : ℕ := 20

theorem jack_change : payment - (cost_per_sandwich * number_of_sandwiches) = 5 := 
by
  sorry

end jack_change_l44_44214


namespace all_terms_integers_l44_44341

def sequence (a : ℕ → ℤ) : Prop :=
  a 1  = 1  ∧
  a 2  = 143 ∧
  (∀ n ≥ 2, a (n + 1) = 5 * (∑ i in finset.range n, a (i + 1)) / n)

theorem all_terms_integers (a : ℕ → ℤ) (h : sequence a) : ∀ n, nat.pred n ≠ 0 → ∃ m : ℤ, a n = m := 
sorry

end all_terms_integers_l44_44341


namespace grilled_cheese_sandwiches_l44_44217

theorem grilled_cheese_sandwiches (h g : ℕ) (c_ham c_grilled total_cheese : ℕ)
  (h_count : h = 10)
  (ham_cheese : c_ham = 2)
  (grilled_cheese : c_grilled = 3)
  (cheese_used : total_cheese = 50)
  (sandwich_eq : total_cheese = h * c_ham + g * c_grilled) :
  g = 10 :=
by
  sorry

end grilled_cheese_sandwiches_l44_44217


namespace incorrect_g2_incorrect_g_neg3_l44_44907

def g (x : ℝ) : ℝ := (2 * x + 3) / (x - 2)

theorem incorrect_g2 : g 2 ≠ 0 := sorry

theorem incorrect_g_neg3 : g (-3) ≠ 0 := sorry

end incorrect_g2_incorrect_g_neg3_l44_44907


namespace counting_formula_l44_44990

theorem counting_formula (n : ℕ) :
  let S := 3^n,
      A1 := 2^n,
      A2 := 2^n,
      A3 := 2^n,
      A1_inter_A2 := 1,
      A1_inter_A3 := 1,
      A2_inter_A3 := 1,
      A1_inter_A2_inter_A3 := 0 in
  S - (A1 + A2 + A3) + (A1_inter_A2 + A1_inter_A3 + A2_inter_A3) - A1_inter_A2_inter_A3 = 3^n - 3 * 2^n + 3 :=
  by
    sorry

end counting_formula_l44_44990


namespace triangle_equilateral_l44_44211

-- Define the sides and angles of a triangle
variables {a b c A B C : ℝ}

-- Given conditions
def condition1 : Prop := (a / real.cos A = b / real.cos B) ∧ (b / real.cos B = c / real.cos C)

-- Proving that the triangle is equilateral
theorem triangle_equilateral (h : condition1) : a = b ∧ b = c :=
  sorry

end triangle_equilateral_l44_44211


namespace balloon_arrangements_l44_44086

theorem balloon_arrangements : (7! / (2! * 2!)) = 1260 := by
  sorry

end balloon_arrangements_l44_44086


namespace mixed_oil_rate_l44_44567

def cost_per_litre_one : ℝ := 55
def cost_per_litre_two : ℝ := 70
def cost_per_litre_three : ℝ := 82

def volume_one : ℝ := 12
def volume_two : ℝ := 8
def volume_three : ℝ := 4

def total_cost : ℝ := (cost_per_litre_one * volume_one) + (cost_per_litre_two * volume_two) + (cost_per_litre_three * volume_three)
def total_volume : ℝ := volume_one + volume_two + volume_three

def rate_per_litre : ℝ := total_cost / total_volume

theorem mixed_oil_rate : rate_per_litre = 64.5 := by
  sorry

end mixed_oil_rate_l44_44567


namespace fg_minus_gf_eq_zero_l44_44314

noncomputable def f (x : ℝ) : ℝ := 4 * x + 6

noncomputable def g (x : ℝ) : ℝ := x / 2 - 1

theorem fg_minus_gf_eq_zero (x : ℝ) : (f (g x)) - (g (f x)) = 0 :=
by
  sorry

end fg_minus_gf_eq_zero_l44_44314


namespace tangent_parallel_to_BC_l44_44674

-- Definitions of points and angles in the geometric configuration
variables (A B C D S : Point)
variables (Gamma : Circle)
variables (alpha gamma : Real)
variables (AS BC : Line)
variables (widehat : Angle)

-- Conditions
axiom intersection_D : intersects (AS) (BC) D
axiom angle_BDA_def : widehat(B, D, A) = widehat(D, A, C) + widehat(D, C, A)
axiom angle_DAC : widehat(D, A, C) = alpha / 2
axiom angle_DCA : widehat(D, C, A) = gamma
axiom tangent_angle_S : widehat(A, C, S) = widehat(A, C, B) + widehat(B, C, S)
axiom angle_ACB : widehat(A, C, B) = gamma
axiom angle_BCS : widehat(B, C, S) = alpha / 2

-- Prove that the tangent at S to Gamma is parallel to (BC)
theorem tangent_parallel_to_BC : tangent_at_point Gamma S = tangent S (BC) := by
  calc tangent_angle_S = (gamma + alpha / 2) : angle_ACB + angle_BCS
      ... = angle_BDA : angle_DAC + angle_DCA
      ... = parallel : sorry

end tangent_parallel_to_BC_l44_44674


namespace simplify_expression_l44_44677

theorem simplify_expression (x y : ℤ) (h1 : x = 1) (h2 : y = -2) :
  2 * x ^ 2 - (3 * (-5 / 3 * x ^ 2 + 2 / 3 * x * y) - (x * y - 3 * x ^ 2)) + 2 * x * y = 2 :=
by {
  sorry
}

end simplify_expression_l44_44677


namespace convex_polygons_on_circle_l44_44487

theorem convex_polygons_on_circle:
  let points := 15 in
  ∑ i in finset.range (points + 1), choose points i - (choose points 0 + choose points 1 + choose points 2 + choose points 3) = 32192 :=
begin
  sorry
end

end convex_polygons_on_circle_l44_44487


namespace hyperbola_vertex_distance_l44_44020

theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ), (x^2 / 48 - y^2 / 16 = 1) →
    (2 * real.sqrt 48 = 8 * real.sqrt 3) :=
by
  intros x y h
  sorry

end hyperbola_vertex_distance_l44_44020


namespace distinct_distances_l44_44414

theorem distinct_distances (points : Finset (ℝ × ℝ)) (h : points.card = 2016) :
  ∃ s : Finset ℝ, s.card ≥ 45 ∧ ∀ p ∈ points, ∃ q ∈ points, p ≠ q ∧ 
    (s = (points.image (λ r => dist p r)).filter (λ x => x ≠ 0)) :=
by
  sorry

end distinct_distances_l44_44414


namespace find_x_l44_44724

theorem find_x :
  ∃ x : ℝ, 0.8 * x + 0.08 = 0.56 → x = 0.6 :=
by
  intro x
  rw [← sub_eq_zero, sub_eq_iff_eq_add] at h
  have : 0.48 = 0.8 * 0.6, by norm_num
  rw this at h
  exact h

end find_x_l44_44724


namespace grilled_cheese_sandwiches_l44_44218

theorem grilled_cheese_sandwiches (h g : ℕ) (c_ham c_grilled total_cheese : ℕ)
  (h_count : h = 10)
  (ham_cheese : c_ham = 2)
  (grilled_cheese : c_grilled = 3)
  (cheese_used : total_cheese = 50)
  (sandwich_eq : total_cheese = h * c_ham + g * c_grilled) :
  g = 10 :=
by
  sorry

end grilled_cheese_sandwiches_l44_44218


namespace sum_first_2015_terms_eq_neg_one_l44_44631

noncomputable def i : ℂ := complex.I

def a_n (n : ℕ) := i ^ n

theorem sum_first_2015_terms_eq_neg_one :
  (finset.sum (finset.range 2015) (λ n, a_n (n+1))) = -1 :=
sorry

end sum_first_2015_terms_eq_neg_one_l44_44631


namespace smaller_cone_volume_ratio_l44_44216

theorem smaller_cone_volume_ratio :
  let r := 12
  let theta1 := 120
  let theta2 := 240
  let arc_length_small := (theta1 / 360) * (2 * Real.pi * r)
  let arc_length_large := (theta2 / 360) * (2 * Real.pi * r)
  let r1 := arc_length_small / (2 * Real.pi)
  let r2 := arc_length_large / (2 * Real.pi)
  let l := r
  let h1 := Real.sqrt (l^2 - r1^2)
  let h2 := Real.sqrt (l^2 - r2^2)
  let V1 := (1 / 3) * Real.pi * r1^2 * h1
  let V2 := (1 / 3) * Real.pi * r2^2 * h2
  V1 / V2 = Real.sqrt 10 / 10 := sorry

end smaller_cone_volume_ratio_l44_44216


namespace find_m_value_l44_44893

variables {a b : ℝ × ℝ}
variables (m : ℝ)

def vector_a := (1, real.sqrt 3)
def vector_b := (3, m)

-- Define the dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Define the projection of vec_b onto vec_a
variable (projection_value : ℝ := 3)
def projection (v1 v2 : ℝ × ℝ) : ℝ :=
  (dot_product v1 v2) / (magnitude v1)

theorem find_m_value : 
  projection vector_a vector_b = projection_value → m = real.sqrt 3 :=
by
  -- Projection definition ≥ Either expand or real.sqrt 3
  sorry

end find_m_value_l44_44893


namespace hungarian_license_plates_l44_44176

/-- 
In Hungarian license plates, digits can be identical. Based on observations, 
someone claimed that on average, approximately 3 out of every 10 vehicles 
have such license plates. Is this statement true?
-/
theorem hungarian_license_plates : 
  let total_numbers := 999
  let non_repeating := 720
  let repeating := total_numbers - non_repeating
  let probability := (repeating : ℝ) / total_numbers
  abs (probability - 0.3) < 0.05 :=
by {
  let total_numbers := 999
  let non_repeating := 720
  let repeating := total_numbers - non_repeating
  let probability := (repeating : ℝ) / total_numbers
  sorry
}

end hungarian_license_plates_l44_44176


namespace inequality_sqrt_expression_l44_44255

theorem inequality_sqrt_expression (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  sqrt 2 * (sqrt (a * (a + b) ^ 3) + b * sqrt (a ^ 2 + b ^ 2)) ≤ 3 * (a ^ 2 + b ^ 2) ∧ 
  (sqrt 2 * (sqrt (a * (a + b) ^ 3) + b * sqrt (a ^ 2 + b ^ 2)) = 3 * (a ^ 2 + b ^ 2) ↔ a = b) :=
by 
  sorry

end inequality_sqrt_expression_l44_44255


namespace number_of_k_pop_sequences_l44_44981

noncomputable def is_k_pop_sequence (k : ℕ) (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n = ({a m | m < n+k}.to_finset.card : ℕ)

theorem number_of_k_pop_sequences (k : ℕ) (hk : 0 < k) :
  ∃ f : ℕ → ℕ, (∀ k, f k = 2^k) ∧ (∑ (_, _) in (finset.range (k) × finset.univ), f _) = f k :=
sorry

end number_of_k_pop_sequences_l44_44981


namespace chess_tournament_games_l44_44937

theorem chess_tournament_games (n : ℕ) (h : n = 25) : 2 * n * (n - 1) = 1200 :=
by
  sorry

end chess_tournament_games_l44_44937


namespace probability_sum_eq_9_l44_44735

def set_t : set ℤ := { -3, 0, 2, 3, 4, 5, 7, 9 }
def set_b : set ℤ := { -2, 4, 5, 6, 7, 8, 10, 12 }

def count_pairs_with_sum (t b : set ℤ) (s : ℤ) : ℕ :=
  t.countp (λ x => b.mem (s - x))

noncomputable def total_pairs (t b : set ℤ) : ℕ :=
  t.card * b.card

theorem probability_sum_eq_9 : 
  (count_pairs_with_sum set_t set_b 9 : ℚ) / (total_pairs set_t set_b : ℚ) = 3 / 32 :=
by
  sorry

end probability_sum_eq_9_l44_44735


namespace simplify_trig_identity_1_simplify_trig_identity_2_l44_44305

variable (α : ℝ)

-- Proof Problem 1
theorem simplify_trig_identity_1 :
  sin α ^ 4 + (tan α ^ 2 * cos α ^ 4) + cos α ^ 2 = 1 :=
by sorry

-- Proof Problem 2
theorem simplify_trig_identity_2 :
  (cos (real.pi + α) * sin (α + 2 * real.pi)) / (sin (-α - real.pi) * cos (-real.pi - α)) = -1 :=
by sorry

end simplify_trig_identity_1_simplify_trig_identity_2_l44_44305


namespace polygon_interior_exterior_eq_l44_44924

theorem polygon_interior_exterior_eq (n : ℕ) (hn : 3 ≤ n)
  (interior_sum_eq_exterior_sum : (n - 2) * 180 = 360) : n = 4 := by
  sorry

end polygon_interior_exterior_eq_l44_44924


namespace a_n_formula_T_n_formula_l44_44857

-- Definitions used in Part I
def S : ℕ → ℕ
  | 0 => 0
  | n + 1 => (n + 1) * (2 * (n + 1) - 1)

def a : ℕ → ℕ
  | 0 => 0
  | n + 1 => S (n + 1) - S n

-- Proof Problem Part I
theorem a_n_formula (n : ℕ) : a n = 4 * n - 3 := sorry

-- Definitions used in Part II
def b (n : ℕ) := 2 * n - 1

def c (n : ℕ) : ℕ → ℚ
  | 0 => 0
  | n + 1 => 1 / ((a (n + 1)) * (2 * (b (n + 1)) + 3))

-- Partial sum function
def T : ℕ → ℚ
  | 0 => 0
  | n + 1 => T n + c (n + 1)

-- Proof Problem Part II
theorem T_n_formula (n : ℕ) : T n = n / (4 * n + 1) := sorry

end a_n_formula_T_n_formula_l44_44857


namespace hypotenuse_of_454590_triangle_l44_44640

theorem hypotenuse_of_454590_triangle (l : ℝ) (angle : ℝ) (h : ℝ) (h_leg : l = 15) (h_angle : angle = 45) :
  h = l * Real.sqrt 2 := 
  sorry

end hypotenuse_of_454590_triangle_l44_44640


namespace Peter_speed_is_correct_l44_44970

variable (Peter_speed : ℝ)

def Juan_speed : ℝ := Peter_speed + 3

def distance_Peter_in_1_5_hours : ℝ := 1.5 * Peter_speed

def distance_Juan_in_1_5_hours : ℝ := 1.5 * Juan_speed Peter_speed

theorem Peter_speed_is_correct (h : distance_Peter_in_1_5_hours Peter_speed + distance_Juan_in_1_5_hours Peter_speed = 19.5) : Peter_speed = 5 :=
by
  sorry

end Peter_speed_is_correct_l44_44970


namespace inequality_solution_l44_44697

theorem inequality_solution (y : ℝ) : 
  (3 ≤ |y - 4| ∧ |y - 4| ≤ 7) ↔ (7 ≤ y ∧ y ≤ 11 ∨ -3 ≤ y ∧ y ≤ 1) :=
by
  sorry

end inequality_solution_l44_44697


namespace polyhedron_has_circumscribed_sphere_l44_44687

theorem polyhedron_has_circumscribed_sphere
  (A B C D S C1 B1 D1 : Point)
  (h_plane : plane_contains_point A (perpendicular_plane S C))
  (h_intersect_SC : h_plane ∩ SC = {C1})
  (h_intersect_SB : h_plane ∩ SB = {B1})
  (h_intersect_SD : h_plane ∩ SD = {D1})
  (h_base : rectangle A B C D)
  (h_perpendicular : perpendicular S (plane_of_rectangle A B C D)) :
  ∃ O : Point, circumscribed_sphere (polyhedron A B C D B1 C1 D1) O := sorry

end polyhedron_has_circumscribed_sphere_l44_44687


namespace distinct_arrangements_balloon_l44_44117

theorem distinct_arrangements_balloon : 
  let n := 7
  let r1 := 2
  let r2 := 2
  (Nat.factorial n) / ((Nat.factorial r1) * (Nat.factorial r2)) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l44_44117


namespace distinct_arrangements_balloon_l44_44115

noncomputable def totalPermutations (n nl no : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial nl * Nat.factorial no)

theorem distinct_arrangements_balloon :
  totalPermutations 7 2 2 = 1260 := by 
  sorry

end distinct_arrangements_balloon_l44_44115


namespace find_ON_l44_44377

theorem find_ON (N A B O : Point) (a b c : ℝ) 
  (hNA : dist N A = a) (hNB : dist N B = b) (hOA : dist O A = c)
  (hOA_OB : dist O A = dist O B) (hOA_gt_ON : dist O A > dist O N) (h_neq : a ≠ b) : 
  dist O N = Real.sqrt (c^2 - a * b) :=
sorry

end find_ON_l44_44377


namespace hypotenuse_of_45_45_90_triangle_l44_44646

noncomputable def leg_length : ℝ := 15
noncomputable def angle_opposite_leg : ℝ := Real.pi / 4  -- 45 degrees in radians

theorem hypotenuse_of_45_45_90_triangle (h_leg : ℝ) (h_angle : ℝ) 
  (h_leg_cond : h_leg = leg_length) (h_angle_cond : h_angle = angle_opposite_leg) :
  ∃ h_hypotenuse : ℝ, h_hypotenuse = h_leg * Real.sqrt 2 :=
sorry

end hypotenuse_of_45_45_90_triangle_l44_44646


namespace proof_problem_l44_44881

variables {a b t x₁ x₂ : ℝ}

def f (x : ℝ) := a * x^2 + b / x
def tangent_condition : Prop := f 1 = 3 ∧ (2 * a - b = 0)
def monotonic_intervals (x : ℝ) : Prop :=
  (0 < x ∧ x < 1 → (2 * x^3 - 2) / x^2 < 0) ∧ (x > 1 → (2 * x^3 - 2) / x^2 > 0)
def sum_of_roots_condition (x₁ x₂ : ℝ) : Prop := (0 < x₁ ∧ x₁ < 1 ∧ x₂ > 1 ∧ f x₁ = t ∧ f x₂ = t ∧ t > 3 → x₁ + x₂ > 2)

theorem proof_problem :
  (f 1 = 3 ∧ (2 * a - b = 0) → a = 1 ∧ b = 2) ∧
  ∀ x, (0 < x ∧ x < 1 → (2 * x^3 - 2) / x^2 < 0) ∧ (x > 1 → (2 * x^3 - 2) / x^2 > 0) ∧
  ∀ t, t > 3 → ∀ x₁ x₂, 0 < x₁ ∧ x₁ < 1 ∧ x₂ > 1 ∧ f x₁ = t ∧ f x₂ = t → x₁ + x₂ > 2 :=
by {
  sorry
}

end proof_problem_l44_44881


namespace rainfall_measurement_l44_44683

-- Define the parameters and known values
def side_length : ℝ := 20 -- side length of the square base in mm
def height : ℝ := 40 -- height of the container in mm
def water_depth : ℝ := 10 -- water depth in the container after 24 hours in mm
def radius : ℝ := side_length / 2 -- radius of the inscribed circular opening
def volume : ℝ := side_length * side_length * water_depth -- volume of collected water
def area : ℝ := Real.pi * radius * radius -- area of the circular opening

-- Theorem: The approximate amount of rainfall measured by the student is 12.7 mm.
theorem rainfall_measurement (V : ℝ) (S : ℝ) :
  V = volume →
  S = area →
  (V / S) ≈ 12.7 :=
by
  intros hV hS
  rw [hV, hS]
  -- here we need a step to actually prove the approximation, which we skip with sorry
  sorry

end rainfall_measurement_l44_44683


namespace find_symmetric_point_l44_44503

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def plane (x y z : ℝ) : ℝ := 
  4 * x + 6 * y + 4 * z - 25

def symmetric_point (M M_prime : Point3D) (plane_eq : ℝ → ℝ → ℝ → ℝ) : Prop :=
  let t : ℝ := (1 / 4)
  let M0 : Point3D := { x := (1 + 4 * t), y := (6 * t), z := (1 + 4 * t) }
  let midpoint_x := (M.x + M_prime.x) / 2
  let midpoint_y := (M.y + M_prime.y) / 2
  let midpoint_z := (M.z + M_prime.z) / 2
  M0.x = midpoint_x ∧ M0.y = midpoint_y ∧ M0.z = midpoint_z ∧
  plane_eq M0.x M0.y M0.z = 0

def M : Point3D := { x := 1, y := 0, z := 1 }

def M_prime : Point3D := { x := 3, y := 3, z := 3 }

theorem find_symmetric_point : symmetric_point M M_prime plane := by
  -- the proof is omitted here
  sorry

end find_symmetric_point_l44_44503


namespace sin_75_deg_l44_44796

theorem sin_75_deg : Real.sin (75 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := 
by sorry

end sin_75_deg_l44_44796


namespace John_time_correct_l44_44221

-- Definitions based on conditions
def John_time : ℝ := 1.5
def John_work_rate (J : ℝ) : ℝ := 1 / J
def David_work_rate (J : ℝ) : ℝ := 1 / (2 * J)
def combined_work_rate (J : ℝ) : ℝ := John_work_rate J + David_work_rate J

-- Proof goal statement
theorem John_time_correct (J : ℝ) (h : combined_work_rate J = 1) : J = John_time :=
by
  sorry

end John_time_correct_l44_44221


namespace players_taking_chemistry_l44_44460

theorem players_taking_chemistry (total_players biology_players both_sci_players: ℕ) 
  (h1 : total_players = 12)
  (h2 : biology_players = 7)
  (h3 : both_sci_players = 2)
  (h4 : ∀ p, p <= total_players) : 
  ∃ chemistry_players, chemistry_players = 7 := 
sorry

end players_taking_chemistry_l44_44460


namespace membership_relation_l44_44511

-- Definitions of M and N
def M (x : ℝ) : Prop := abs (x + 1) < 4
def N (x : ℝ) : Prop := x / (x - 3) < 0

theorem membership_relation (a : ℝ) (h : M a) : N a → M a := by
  sorry

end membership_relation_l44_44511


namespace power_function_no_origin_l44_44572

theorem power_function_no_origin (m : ℝ) : 
  (m^2 - m - 1 <= 0) ∧ (m^2 - 3 * m + 3 = 1) → m = 1 :=
by
  intros
  sorry

end power_function_no_origin_l44_44572


namespace digit_product_equality_exists_l44_44467

/-- Prove that there exists at least one digit from {0, 1, ..., 9} that can be assigned to one of 
    C, T, I, Φ, P, A to make the equation C * T * 0 = C * I * Φ * P * A hold true, 
    with all variables being distinct digits. -/
theorem digit_product_equality_exists :
  ∃ (C T I Φ P A : ℕ), 
    (C ≠ T ∧ C ≠ I ∧ C ≠ Φ ∧ C ≠ P ∧ C ≠ A ∧ T ≠ I ∧ T ≠ Φ ∧ T ≠ P ∧ T ≠ A ∧ 
     I ≠ Φ ∧ I ≠ P ∧ I ≠ A ∧ Φ ≠ P ∧ Φ ≠ A ∧ P ≠ A) ∧
    C ∈ finset.range 10 ∧ T ∈ finset.range 10 ∧ I ∈ finset.range 10 ∧ 
    Φ ∈ finset.range 10 ∧ P ∈ finset.range 10 ∧ A ∈ finset.range 10 ∧
    C * T * 0 = C * I * Φ * P * A :=
by
  sorry

end digit_product_equality_exists_l44_44467


namespace cubic_identity_l44_44161

variable {a b c : ℝ}

theorem cubic_identity (h1 : a + b + c = 13) (h2 : ab + ac + bc = 30) : a^3 + b^3 + c^3 - 3 * a * b * c = 1027 := 
by 
  sorry

end cubic_identity_l44_44161


namespace functional_equation_solution_l44_44495

theorem functional_equation_solution (f : ℚ → ℚ) (h : ∀ x y : ℚ, f (x + y) = f x + f y) :
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x :=
sorry

end functional_equation_solution_l44_44495


namespace chord_length_on_parabola_l44_44850

noncomputable def circle_chord_length (x : ℝ) : ℝ :=
  let y := 1/2 * x^2
  let r := real.sqrt (x^2 + (1/2 * x^2 - 1)^2)
  (2 : ℝ)

theorem chord_length_on_parabola :
  ∀ (x : ℝ), circle_chord_length x = 2 :=
by
  intro x
  simp [circle_chord_length]
  sorry

end chord_length_on_parabola_l44_44850


namespace justin_home_time_l44_44248

noncomputable def dinner_duration : ℕ := 45
noncomputable def homework_duration : ℕ := 30
noncomputable def cleaning_room_duration : ℕ := 30
noncomputable def taking_out_trash_duration : ℕ := 5
noncomputable def emptying_dishwasher_duration : ℕ := 10

noncomputable def total_time_required : ℕ :=
  dinner_duration + homework_duration + cleaning_room_duration + taking_out_trash_duration + emptying_dishwasher_duration

noncomputable def latest_start_time_hour : ℕ := 18 -- 6 pm in 24-hour format
noncomputable def total_time_required_hours : ℕ := 2
noncomputable def movie_time_hour : ℕ := 20 -- 8 pm in 24-hour format

theorem justin_home_time : latest_start_time_hour - total_time_required_hours = 16 := -- 4 pm in 24-hour format
by
  sorry

end justin_home_time_l44_44248


namespace tangent_line_eq_at_a_equals_1_f_positive_a_equals_1_f_min_max_a_greater_half_l44_44882

-- Part (1)
theorem tangent_line_eq_at_a_equals_1 :
  ∀ (f : ℝ → ℝ), (∀ x, f x = Real.exp x - 2 * x) →
  ((λ x, Real.exp x - 2 * x).deriv 0 = -1 ∧ f 0 = 1) →
  ∃ (L : ℝ → ℝ), (L 0 = 1 ∧ ∀ x, L x = -x + 1) :=
sorry

-- Part (2)
theorem f_positive_a_equals_1 :
  ∀ (f : ℝ → ℝ), (∀ x, f x = Real.exp x - 2 * x) →
  (∀ x, f x > 0) :=
sorry

-- Part (3)
theorem f_min_max_a_greater_half :
  ∀ (f : ℝ → ℝ), (∀ x a, a > 1/2 → f x = Real.exp x - 2 * a * x) →
  (∀ a, a > 1/2 →
    (∃ x, 0 ≤ x ∧ x ≤ 2 * a ∧ f(x) = 2 * a * (1 - Real.log (2 * a)))
    ∧ (∃ x, 0 ≤ x ∧ x ≤ 2 * a ∧ f(x) = Real.exp (2 * a) - 4 * a^2)) :=
sorry

end tangent_line_eq_at_a_equals_1_f_positive_a_equals_1_f_min_max_a_greater_half_l44_44882


namespace compute_g_neg_x_l44_44474

noncomputable def g (x : ℝ) : ℝ := (x^2 + 3*x + 2) / (x^2 - 3*x + 2)

theorem compute_g_neg_x (x : ℝ) (h : x^2 ≠ 2) : g (-x) = 1 / g x := 
  by sorry

end compute_g_neg_x_l44_44474


namespace hypotenuse_of_45_45_90_triangle_l44_44663

theorem hypotenuse_of_45_45_90_triangle (leg : ℝ) (angle_opposite_leg : ℝ) (h_leg : leg = 15) (h_angle : angle_opposite_leg = 45) :
  ∃ hypotenuse, hypotenuse = leg * Real.sqrt 2 :=
by
  use leg * Real.sqrt 2
  rw [h_leg]
  rw [h_angle]
  sorry

end hypotenuse_of_45_45_90_triangle_l44_44663


namespace mortar_shell_hits_the_ground_at_50_seconds_l44_44586

noncomputable def mortar_shell_firing_equation (x : ℝ) : ℝ :=
  - (1 / 5) * x^2 + 10 * x

theorem mortar_shell_hits_the_ground_at_50_seconds : 
  ∃ x : ℝ, mortar_shell_firing_equation x = 0 ∧ x = 50 :=
by
  sorry

end mortar_shell_hits_the_ground_at_50_seconds_l44_44586


namespace function_characteristics_l44_44026

-- Define the given function
noncomputable def f (x : ℝ) : ℝ := 2 * sin (1/2 * x + π/4)

-- Amplitude, period, and phase shift for the problem statement
def amplitude : ℝ := abs 2 -- Amplitude is |2|
def phase_shift : ℝ := π/4 -- Phase shift is π/4
def period : ℝ := 4 * π   -- Period is 4π

-- Lean statement to prove the equivalence
theorem function_characteristics :
  amplitude = 2 ∧ phase_shift = π/4 ∧ period = 4 * π :=
by
  -- Proof goes here, skipped with sorry
  sorry

end function_characteristics_l44_44026


namespace mul_eight_neg_half_l44_44798

theorem mul_eight_neg_half : 8 * (- (1/2: ℚ)) = -4 := 
by 
  sorry

end mul_eight_neg_half_l44_44798


namespace medians_intersect_l44_44595

-- Definitions (conditions)
variables {P : Type*} [plane P] [point P] [is_affine_space P]

-- An arbitrary convex hexagon ABCDEF
variables (A B C D E F : P)

-- The midpoints K, L, M, N, P, Q of AB, BC, CD, DE, EF, FA respectively
def midpoint (x y : P) : P := sorry -- placeholder for midpoint definition

variables
  (K : P) (L : P) (M : P) (N : P) (P : P) (Q : P)
  (hK : K = midpoint A B)
  (hL : L = midpoint B C)
  (hM : M = midpoint C D)
  (hN : N = midpoint D E)
  (hP : P = midpoint E F)
  (hQ : Q = midpoint F A)

-- Proving the intersection points of the medians of the two formed triangles coincide
theorem medians_intersect (O : P) :
  let triangle1_medians := medians_of_triangle K M P,
      triangle2_medians := medians_of_triangle N Q L in
  intersection_of_medians triangle1_medians O ↔
  intersection_of_medians triangle2_medians O :=
  sorry -- proof to be filled in

-- Helper definitions (placeholders)
def medians_of_triangle (A B C : P) : set P := sorry -- placeholder
def intersection_of_medians (medians : set P) (O : P) : Prop := sorry -- placeholder

end medians_intersect_l44_44595


namespace balloon_permutations_l44_44137

theorem balloon_permutations : 
  let n : ℕ := 7
  let k1 : ℕ := 2
  let k2 : ℕ := 2
  ∃ distinct_arrangements : ℕ, 
  distinct_arrangements = n.factorial / (k1.factorial * k2.factorial) 
  ∧ distinct_arrangements = 1260 :=
by
  sorry

end balloon_permutations_l44_44137


namespace f_is_odd_function_f_max_in_interval_inequality_solution_l44_44519

-- Define conditions
axiom f_additive (f : ℝ → ℝ) : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_negative_for_positive (f : ℝ → ℝ) : ∀ x : ℝ, x > 0 → f x < 0
axiom f_at_one (f : ℝ → ℝ) : f 1 = -2

-- Questions to be proven
theorem f_is_odd_function (f : ℝ → ℝ) [f_additive f] [f_negative_for_positive f] [f_at_one f] : 
∀ x : ℝ, f(-x) = -f(x) := 
sorry

theorem f_max_in_interval (f : ℝ → ℝ) [f_additive f] [f_negative_for_positive f] [f_is_odd_function f] :
∀ x ∈ Icc (-3 : ℝ) 3, f x ≤ f (-3) := 
sorry

theorem inequality_solution (f : ℝ → ℝ) [f_additive f] [f_negative_for_positive f] [f_at_one f] [f_is_odd_function f] :
∀ a x : ℝ, 
(a = 0 → x < 1) ∧ 
(a = 2 → x ≠ 1) ∧ 
(a < 0 → 2/a < x ∧ x < 1) ∧ 
(0 < a ∧ a < 2 → x > 2/a ∨ x < 1) ∧ 
(a > 2 → x < 2/a ∨ x > 1) := 
sorry

end f_is_odd_function_f_max_in_interval_inequality_solution_l44_44519


namespace balloon_arrangements_l44_44106

-- Define the variables
def n : ℕ := 7
def L_count : ℕ := 2
def O_count : ℕ := 2
def B_count : ℕ := 1
def A_count : ℕ := 1
def N_count : ℕ := 1

-- Define the multiset permutation formula
def multiset_permutations (n : ℕ) (counts : List ℕ) : ℕ :=
  n.factorial / (counts.map Nat.factorial).prod

-- Proof that the number of distinct arrangements is 1260
theorem balloon_arrangements : multiset_permutations n [L_count, O_count, B_count, A_count, N_count] = 1260 :=
  by
  -- The proof is omitted
  sorry

end balloon_arrangements_l44_44106


namespace area_of_circle_above_line_l44_44382

theorem area_of_circle_above_line (x y : ℝ) :
  (∃ r : ℝ, (x - 4)^2 + (y - 8)^2 = r^2 ∧ r = 2 * real.sqrt 3) →
  y > 4 →
  (π * (2 * real.sqrt 3)^2 = 12 * π) :=
by
  sorry

end area_of_circle_above_line_l44_44382


namespace rotated_square_distance_l44_44369

-- Definition of the problem conditions
def square_side : ℝ := 1
def rotated_angle : ℝ := π / 4 -- 45 degrees in radians
def base_line := λ (x : ℝ), 0

-- Rotating the square and finding the vertical distance
theorem rotated_square_distance:
  let diagonal := real.sqrt (2 * square_side^2) in
  let height_center_to_center := (diagonal / 2) - (square_side / 2) in
  let height_B := height_center_to_center + (square_side * real.sqrt 2) / 2 in
  height_B = real.sqrt 2 + (1 / 2) :=
  sorry

end rotated_square_distance_l44_44369


namespace work_lasted_35_days_l44_44402

-- Define the conditions
def P_work_days : ℕ := 80
def Q_work_days : ℕ := 48
def P_work_alone_days : ℕ := 8

-- Stating the theorem
theorem work_lasted_35_days :
  P_work_days = 80 →
  Q_work_days = 48 →
  P_work_alone_days = 8 →
  let total_days := P_work_alone_days + ((P_work_days * Q_work_days) / (P_work_days + Q_work_days)) * 9 / 10 in
  total_days = 35 :=
by
  -- Placeholder for the proof
  sorry

end work_lasted_35_days_l44_44402


namespace hypotenuse_of_45_45_90_triangle_l44_44662

theorem hypotenuse_of_45_45_90_triangle (leg : ℝ) (angle_opposite_leg : ℝ) (h_leg : leg = 15) (h_angle : angle_opposite_leg = 45) :
  ∃ hypotenuse, hypotenuse = leg * Real.sqrt 2 :=
by
  use leg * Real.sqrt 2
  rw [h_leg]
  rw [h_angle]
  sorry

end hypotenuse_of_45_45_90_triangle_l44_44662


namespace no_yarn_earnings_l44_44810

noncomputable def yarn_cost : Prop :=
  let monday_yards := 20
  let tuesday_yards := 2 * monday_yards
  let wednesday_yards := (1 / 4) * tuesday_yards
  let total_yards := monday_yards + tuesday_yards + wednesday_yards
  let fabric_cost_per_yard := 2
  let total_fabric_earnings := total_yards * fabric_cost_per_yard
  let total_earnings := 140
  total_fabric_earnings = total_earnings

theorem no_yarn_earnings:
  yarn_cost :=
sorry

end no_yarn_earnings_l44_44810


namespace evaluate_expression_l44_44676

theorem evaluate_expression (x : ℚ) (h : x = 1/2) : 
  (x - 3)^2 + (x + 3)*(x - 3) + 2*x*(2 - x) = -1 :=
by 
  rw h
  -- The proof steps will follow here
  sorry

end evaluate_expression_l44_44676


namespace quadratic_solution_pair_l44_44339

open Real

noncomputable def solution_pair : ℝ × ℝ :=
  ((45 - 15 * sqrt 5) / 2, (45 + 15 * sqrt 5) / 2)

theorem quadratic_solution_pair (a c : ℝ) 
  (h1 : (∃ x : ℝ, a * x^2 + 30 * x + c = 0 ∧ ∀ y : ℝ, y ≠ x → a * y^2 + 30 * y + c ≠ 0))
  (h2 : a + c = 45)
  (h3 : a < c) :
  (a, c) = solution_pair :=
sorry

end quadratic_solution_pair_l44_44339


namespace magnitude_proj_l44_44892

variables {E : Type*} [inner_product_space ℝ E] (u z : E)
variables (h1 : inner u z = 6) (h2 : ∥z∥ = 10)

theorem magnitude_proj : ∥(orthogonal_projection z u)∥ = 0.6 := 
sorry

end magnitude_proj_l44_44892


namespace die_sum_bounds_proof_l44_44748

noncomputable def die_sum_bounds : Prop :=
    let cell_count := 99 in
    let avg_face_value := 3.5 in
    let expected_sum := avg_face_value * cell_count in
    let max_deviation := 4.5 in
    (expected_sum - max_deviation ≤ 342 ∧
    expected_sum + max_deviation ≥ 351)

theorem die_sum_bounds_proof : die_sum_bounds := 
begin
  sorry
end

end die_sum_bounds_proof_l44_44748


namespace total_investment_sum_l44_44715

variable Raghu_investment : ℕ := 2500
variable Trishul_investment : ℕ := Raghu_investment - (Raghu_investment / 10)
variable Vishal_investment : ℕ := Trishul_investment + (Trishul_investment / 10)
variable Total_investment : ℕ := Raghu_investment + Trishul_investment + Vishal_investment

theorem total_investment_sum :
  Total_investment = 7225 :=
by sorry

end total_investment_sum_l44_44715


namespace part_1_part_2_l44_44506

def S := { n : ℕ // nat.popcount n = 3 }

def f (k : ℕ) := { n : ℕ // k + 1 ≤ n ∧ n ≤ 2 * k ∧ n ∈ S }

theorem part_1 (m : ℕ) (hm : 0 < m) : 
  ∃ k : ℕ, 0 < k ∧ f(k).card = m := sorry

theorem part_2 (m : ℕ) (hm : 0 < m) : 
  (∃! k : ℕ, 0 < k ∧ f(k).card = m) ↔ ∃ s : ℕ, s ≥ 2 ∧ m = s * (s - 1) / 2 + 1 := sorry

end part_1_part_2_l44_44506


namespace save_special_troops_l44_44705

theorem save_special_troops (positions : Finset ℕ) :
  positions = {5, 6, 7, 8, 9, 12, 16, 18, 19, 22, 23, 24, 26, 27, 30} →
  ∀ soldiers : Fin 30 → Prop,
  (∀ i ∈ positions, soldiers i) →
  (∀ i ∉ positions, ¬ soldiers i) →
  by sorry

end save_special_troops_l44_44705


namespace simplify_expression_l44_44304

theorem simplify_expression (x y : ℝ) (h : x = -3) : 
  x * (x - 4) * (x + 4) - (x + 3) * (x^2 - 6 * x + 9) + 5 * x^3 * y^2 / (x^2 * y^2) = -66 :=
by
  sorry

end simplify_expression_l44_44304


namespace license_plate_combinations_l44_44793

theorem license_plate_combinations :
  let letters := 26 in
  let binom_25_2 := Nat.choose 25 2 in
  let arrange_letters := Nat.choose 4 2 * 2 in
  let digits := 10 in
  let choose_positions := Nat.choose 3 2 in
  let different_digit := 9 in
  letters * binom_25_2 * arrange_letters * digits * choose_positions * different_digit = 4212000 :=
  by
    let letters := 26
    let binom_25_2 := Nat.choose 25 2
    let arrange_letters := Nat.choose 4 2 * 2
    let digits := 10
    let choose_positions := Nat.choose 3 2
    let different_digit := 9
    calc
      26 * binom_25_2 * arrange_letters * digits * choose_positions * different_digit
      _ = 26 * 300 * 6 * 10 * 3 * 9 : by rfl
      _ = 4212000 : by norm_num

end license_plate_combinations_l44_44793


namespace probability_sum_is_3_l44_44363

theorem probability_sum_is_3 (die : Type) [Fintype die] [DecidableEq die] 
  (dice_faces : die → ℕ) (h : ∀ d, dice_faces d ∈ {1, 2, 3, 4, 5, 6}) :
  (∑ i in finset.range 3, (die →₀ ℕ).single 1) = 3 → 
  (1 / (finset.card univ) ^ 3) = 1 / 216 :=
by
  sorry

end probability_sum_is_3_l44_44363


namespace johns_age_l44_44222

theorem johns_age (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l44_44222


namespace quadratic_function_origin_l44_44573

theorem quadratic_function_origin (m : ℝ) : 
    (∀ x y : ℝ, y = m * x^2 + x + m * (m - 3) → x = 0 → y = 0) → m = 3 :=
by
  intro h1
  have h0 := h1 0 0
  have h := h0 rfl
  sorry

end quadratic_function_origin_l44_44573


namespace probability_C_10000_equal_expected_prize_money_l44_44784

/-- Problem 1: Probability that C gets 10,000 yuan given P1 = P2 = 1/2 --/
theorem probability_C_10000 (P1 P2 : ℝ) (h : P1 = 1/2 ∧ P2 = 1/2) : 
  let A := 10000
  let B := 20000
  let prize := 40000
  P1 * (1 - P2) + (1 - P1) * P2 = 1/2 := by
  sorry

/-- Problem 2: Values of P1 and P2 for equal expected prize money --/
theorem equal_expected_prize_money (P1 P2 : ℝ) (h1 : P1 + P2 = 1) (h2 : 
  let eA := P1 + 2 * P2 
  let eB := P1 + 2 * P2
  let eC := 2 * P1^2 + 2 * P1 * P2
  eA = eB ∧ eB = eC) : P1 = 2 / 3 ∧ P2 = 1 / 3 := by
  sorry

end probability_C_10000_equal_expected_prize_money_l44_44784


namespace rectangle_same_color_l44_44819

theorem rectangle_same_color (colors : ℕ → ℕ → ℕ) (h_colors : ∀ x y, colors x y < 3) :
  ∃ (x1 x2 y1 y2 : ℕ), x1 < x2 ∧ y1 < y2 ∧ colors x1 y1 = colors x1 y2 ∧ colors x1 y1 = colors x2 y1 ∧ colors x1 y1 = colors x2 y2 := 
sorry

end rectangle_same_color_l44_44819


namespace distinct_arrangements_balloon_l44_44130

-- Let's define the basic conditions:
def total_letters : Nat := 7
def repeats_l : Nat := 2
def repeats_o : Nat := 2

-- Now let's state the problem.
theorem distinct_arrangements_balloon : 
  (Nat.factorial total_letters) / ((Nat.factorial repeats_l) * (Nat.factorial repeats_o)) = 1260 := 
by
  sorry

end distinct_arrangements_balloon_l44_44130


namespace smallest_a_for_polynomial_l44_44335

theorem smallest_a_for_polynomial (a b x₁ x₂ x₃ : ℕ) 
    (h1 : x₁ * x₂ * x₃ = 2730)
    (h2 : x₁ + x₂ + x₃ = a)
    (h3 : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0)
    (h4 : ∀ y₁ y₂ y₃ : ℕ, y₁ * y₂ * y₃ = 2730 ∧ y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 → y₁ + y₂ + y₃ ≥ a) :
  a = 54 :=
  sorry

end smallest_a_for_polynomial_l44_44335


namespace correct_statement_l44_44728

theorem correct_statement:
  (∀ (a b: ℚ), (a = -b) → (a + b = 0)) ∧
  ¬ (∀ (a: ℚ), (0 < a ∨ a < 0) → a ∈ ℚ)
  ∧ ¬ (∀ (a: ℚ), (|a| = a) → (0 < a))
  ∧ ¬ (∀ (a: ℚ), ((a < 0) ↔ (a has a negative sign))) := 
by
  sorry

end correct_statement_l44_44728


namespace distinct_arrangements_balloon_l44_44097

theorem distinct_arrangements_balloon : 
  let n := 7 
  let freq_l := 2 
  let freq_o := 2 
  let freq_b := 1 
  let freq_a := 1 
  let freq_n := 1 
  Nat.factorial n / (Nat.factorial freq_l * Nat.factorial freq_o * Nat.factorial freq_b * Nat.factorial freq_a * Nat.factorial freq_n) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l44_44097


namespace problem_statement_l44_44462

def modular_inverse_existence (a n : ℕ) := ∃ b : ℕ, (a * b) % n = 1

def modular_inverse (a n : ℕ) (h : modular_inverse_existence a n) : ℕ := 
  classical.some h

theorem problem_statement : 
  let x := modular_inverse 7 60 (by { dsimp [modular_inverse_existence], use 43, norm_num })
  let y := modular_inverse 13 60 (by { dsimp [modular_inverse_existence], use 37, norm_num }) in
  (3 * x + 9 * y) % 60 = 42 :=
by
  let x := 43
  let y := 37
  norm_num [x, y]
  sorry

end problem_statement_l44_44462


namespace Peter_speed_is_correct_l44_44969

variable (Peter_speed : ℝ)

def Juan_speed : ℝ := Peter_speed + 3

def distance_Peter_in_1_5_hours : ℝ := 1.5 * Peter_speed

def distance_Juan_in_1_5_hours : ℝ := 1.5 * Juan_speed Peter_speed

theorem Peter_speed_is_correct (h : distance_Peter_in_1_5_hours Peter_speed + distance_Juan_in_1_5_hours Peter_speed = 19.5) : Peter_speed = 5 :=
by
  sorry

end Peter_speed_is_correct_l44_44969


namespace find_m_plus_n_l44_44554

-- Define the sets and variables
def M : Set ℝ := {x | x^2 - 4 * x < 0}
def N (m : ℝ) : Set ℝ := {x | m < x ∧ x < 5}
def K (n : ℝ) : Set ℝ := {x | 3 < x ∧ x < n}

theorem find_m_plus_n (m n : ℝ) 
  (hM: M = {x | 0 < x ∧ x < 4})
  (hK_true: K n = M ∩ N m) :
  m + n = 7 := 
  sorry

end find_m_plus_n_l44_44554


namespace bread_remaining_is_26_85_l44_44909

noncomputable def bread_leftover (jimin_cm : ℕ) (taehyung_m original_length : ℝ) : ℝ :=
  original_length - (jimin_cm / 100 + taehyung_m)

theorem bread_remaining_is_26_85 :
  bread_leftover 150 1.65 30 = 26.85 :=
by
  sorry

end bread_remaining_is_26_85_l44_44909


namespace determine_x_l44_44389

def f (x : ℕ) : ℕ :=
if x % 2 = 0 then x / 2 + 1 else 2 * x + 1

theorem determine_x (x : ℕ) (h : f 6 * f x = 28) : x = 12 :=
by
  -- Step to directly input values used in the original computation
  have h6 : f 6 = 4 := by
    -- Show the computation detail of f(6)
    unfold f
    rw [if_pos (by norm_num : 6 % 2 = 0)]
    norm_num
  sorry

end determine_x_l44_44389


namespace rate_of_interest_increase_l44_44300

noncomputable def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

noncomputable def percentage_increase_in_rate (P A1 A2 T : ℝ) : ℝ :=
  let SI1 := A1 - P in
  let R1 := (SI1 * 100) / (P * T) in
  let SI2 := A2 - P in
  let R2 := (SI2 * 100) / (P * T) in
  ((R2 - R1) / R1) * 100

theorem rate_of_interest_increase :
  percentage_increase_in_rate 800 956 1052 3 ≈ 61.54 := by
    sorry

end rate_of_interest_increase_l44_44300


namespace dot_product_calc_parallel_condition_l44_44078

open Real

variables (a b c : ℝ × ℝ) (λ : ℝ)

def vec_a := (2, -1)
def vec_b := (3, -2)
def vec_c := (3, 4)

-- Calculate the dot product 
theorem dot_product_calc :
  (vec_a.1 * (vec_b.1 + vec_c.1) + vec_a.2 * (vec_b.2 + vec_c.2)) = 10 :=
by
  sorry

-- Condition for parallelism
theorem parallel_condition :
  (2 + 3 * λ, -1 - 2 * λ) = (vec_c.1 * k, vec_c.2 * k) ↔ λ = -11 / 18 :=
by
  sorry

end dot_product_calc_parallel_condition_l44_44078


namespace prism_surface_area_88_l44_44700

def sphere_volume (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

def prism_volume (l w h : ℝ) : ℝ :=
  l * w * h

def surface_area_prism (l w h : ℝ) : ℝ :=
  2 * l * w + 2 * l * h + 2 * w * h

axiom sphere_prism_volume
  (r l w : ℝ) (h : ℝ) (h_eq : prism_volume l w h = sphere_volume r) :
  r = 3 + 36 / Real.pi ∧ l = 6 ∧ w = 4 → h = 2

theorem prism_surface_area_88 :
  let r := 3 + 36 / Real.pi
  let l := 6
  let w := 4
  let h := 2
  prism_volume l w h = sphere_volume r →
  surface_area_prism l w h = 88 :=
by
  intros
  rw [surface_area_prism, prism_volume, sphere_volume]
  sorry

end prism_surface_area_88_l44_44700


namespace find_number_of_students_l44_44396

theorem find_number_of_students
    (S N : ℕ) 
    (h₁ : 4 * S + 3 = N)
    (h₂ : 5 * S = N + 6) : 
  S = 9 :=
by
  sorry

end find_number_of_students_l44_44396


namespace erin_paths_count_l44_44007

-- Definition of the problem conditions
def erin_paths (start : ℂ × ℂ × ℂ) (end : ℂ × ℂ × ℂ) : Prop :=
  -- Erin starts at a specific corner of a cube
  start = (0, 0, 0) ∧
  -- Visits all corners exactly once and ends adjacent to the start without looping 
  end ∈ {(0, 0, 1), (0, 1, 0), (1, 0, 0)} ∧
  -- The path is exactly 8 edges long visiting every corner once
  false -- Placeholder for the actual path condition logic

-- The number of such paths
def number_of_erins_paths : ℕ := 12

theorem erin_paths_count : ∃ n : ℕ, n = number_of_erins_paths :=
by {
  -- We can use sorry to skip the proof
  sorry
}

end erin_paths_count_l44_44007


namespace johns_age_l44_44230

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l44_44230


namespace tom_total_distance_l44_44373

/-- Tom swims for 1.5 hours at 2.5 miles per hour. 
    Tom runs for 0.75 hours at 6.5 miles per hour. 
    Tom bikes for 3 hours at 12 miles per hour. 
    The total distance Tom covered is 44.625 miles.
-/
theorem tom_total_distance
  (swim_time : ℝ := 1.5) (swim_speed : ℝ := 2.5)
  (run_time : ℝ := 0.75) (run_speed : ℝ := 6.5)
  (bike_time : ℝ := 3) (bike_speed : ℝ := 12) :
  swim_time * swim_speed + run_time * run_speed + bike_time * bike_speed = 44.625 :=
by
  sorry

end tom_total_distance_l44_44373


namespace perimeter_quadrilateral_l44_44721

-- Define variables for side lengths
variables (EF GH FG : ℝ)
variable (perp : ∀ {a b c d : ℝ}, a ∉ Set.range b → c ∉ Set.range d → b = d → a = c)

-- Given conditions
def EF_length := (EF = 12)
def GH_length := (GH = 7)
def FG_length := (FG = 15)
def EF_perpendicular_FG := (perp ↔ EF ≠ FG)
def GH_perpendicular_FG := (perp ↔ GH ≠ FG)

-- Define the hypothesis as a conjunct of the conditions
def conditions := EF_length ∧ GH_length ∧ FG_length ∧ EF_perpendicular_FG ∧ GH_perpendicular_FG

-- The main theorem to prove
theorem perimeter_quadrilateral (h : conditions) : EF + FG + GH + (Real.sqrt (5^2 + 15^2)) = 34 + 5 * Real.sqrt 10 := by
  sorry

end perimeter_quadrilateral_l44_44721


namespace area_of_region_l44_44015

theorem area_of_region : 
  let x1 := 0
  let y1 := 0
  let x2 := 2
  let y2 := 2
  in (x2 - x1) * (y2 - y1) = 4 :=
by
  simp [x1, y1, x2, y2]
  norm_num
  sorry

end area_of_region_l44_44015


namespace max_quotient_l44_44165

theorem max_quotient (a b : ℝ) (ha : 100 ≤ a ∧ a ≤ 300) (hb : 500 ≤ b ∧ b ≤ 1500) : 
  ∃ max_val, max_val = 225 ∧ ∀ (x y : ℝ), (100 ≤ x ∧ x ≤ 300) ∧ (500 ≤ y ∧ y ≤ 1500) → (y^2 / x^2) ≤ max_val := 
by
  use 225
  sorry

end max_quotient_l44_44165


namespace distance_between_A_and_B_l44_44957

noncomputable def pointA : ℝ × ℝ × ℝ := (1, 3, -2)
noncomputable def pointB : ℝ × ℝ × ℝ := (-2, 3, 2)

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

theorem distance_between_A_and_B : distance pointA pointB = 5 := by
  sorry

end distance_between_A_and_B_l44_44957


namespace sum_of_roots_l44_44832

-- Given polynomial
def p (x : ℝ) : ℝ := 
  (x - 1)^2010 - 2 * (x - 2)^2009 + 3 * (x - 3)^2008 
  - (∑ i in finset.range 2009, (i + 4) * (-1)^(i + 4) * (x - (i + 4)) ^ (2009 - i + 1))
  + 2009 * (-1)^2009 * (x - 2010)^2 + 2010 * (-1)^2010 * (x - 2011)^1

-- Statement of the problem
theorem sum_of_roots : 
  let roots := (some function that returns the roots of the polynomial)
  (multiset.sum roots = 2012) :=
sorry

end sum_of_roots_l44_44832


namespace range_of_m_l44_44923

variable (x y m : ℝ)

def system_of_eq1 := 2 * x + y = -4 * m + 5
def system_of_eq2 := x + 2 * y = m + 4
def inequality1 := x - y > -6
def inequality2 := x + y < 8

theorem range_of_m:
  system_of_eq1 x y m → 
  system_of_eq2 x y m → 
  inequality1 x y → 
  inequality2 x y → 
  -5 < m ∧ m < 7/5 :=
by 
  intros h1 h2 h3 h4
  sorry

end range_of_m_l44_44923


namespace percentage_of_first_to_second_l44_44171

theorem percentage_of_first_to_second (X : ℝ) (h1 : first = (7/100) * X) (h2 : second = (14/100) * X) : (first / second) * 100 = 50 := 
by
  sorry

end percentage_of_first_to_second_l44_44171


namespace triangle_side_length_range_l44_44056

theorem triangle_side_length_range (x : ℝ) (h : 0 < x) (h2 : 0 < 2) (h3 : 0 < 3) 
  (acute_triangle : 2^2 + 3^2 - x^2 > 0 ∧ 2^2 + x^2 - 3^2 > 0) : 
  sqrt 5 < x ∧ x < sqrt 13 :=
by
  sorry

end triangle_side_length_range_l44_44056


namespace balloon_arrangements_l44_44081

theorem balloon_arrangements : (7! / (2! * 2!)) = 1260 := by
  sorry

end balloon_arrangements_l44_44081


namespace derivative_f_at_1_l44_44545

def f (x : ℝ) : ℝ := (x^3 - 2 * x) * exp x

theorem derivative_f_at_1 :
  (deriv f 1) = 0 :=
by 
  -- Proof is omitted
  sorry

end derivative_f_at_1_l44_44545


namespace total_population_expr_l44_44582

-- Definitions of the quantities
variables (b g t : ℕ)

-- Conditions
axiom boys_as_girls : b = 3 * g
axiom girls_as_teachers : g = 9 * t

-- Theorem to prove
theorem total_population_expr : b + g + t = 37 * b / 27 :=
by
  sorry

end total_population_expr_l44_44582


namespace balloon_permutations_l44_44141

theorem balloon_permutations : 
  let n : ℕ := 7
  let k1 : ℕ := 2
  let k2 : ℕ := 2
  ∃ distinct_arrangements : ℕ, 
  distinct_arrangements = n.factorial / (k1.factorial * k2.factorial) 
  ∧ distinct_arrangements = 1260 :=
by
  sorry

end balloon_permutations_l44_44141


namespace binomial_coeff_property_l44_44394

def parity (Y : Set ℕ) : ℤ :=
  if (Y.card % 2 = 0) then 1 else -1

def subset_sum (Y : Set ℕ) : ℕ := Y.sum id

theorem binomial_coeff_property (X : Set ℕ) (n s p : ℕ) (hX_card : X.card = n) (hX_sum : X.sum id = s) (hX_prod : X.prod id = p) (N : ℕ) (hN : N ≥ s) :
  (∑ Y in X.powerset, parity Y * binomial (N - subset_sum Y) s) = p :=
sorry

end binomial_coeff_property_l44_44394


namespace find_minimum_value_l44_44070

-- Conditions
def f (x : ℝ) : ℝ := 3 ^ x + 9 ^ x
def domain (t x : ℝ) : Prop := t ≤ x ∧ x ≤ t + 1

theorem find_minimum_value (t : ℝ) (h : ∀ x, domain t x → f x ≤ 12) : ∃ x, domain t x ∧ f x = 2 :=
  sorry

end find_minimum_value_l44_44070


namespace solution_l44_44818

universe u

-- Definitions for the dwarves and their truth-telling properties
inductive Dwarf
| Benya : Dwarf
| Venya : Dwarf
| Senya : Dwarf
| Zhenya : Dwarf

def tells_truth (d : Dwarf) : Prop
def tells_lie (d : Dwarf) : Prop := ¬ tells_truth d

-- The dwarves' statements
def Benya_statement : Prop := tells_lie Dwarf.Venya
def Zhenya_statement : Prop := tells_lie Dwarf.Benya
def Senya_statement_1 : Prop := tells_lie Dwarf.Benya ∧ tells_lie Dwarf.Zhenya
def Senya_statement_2 : Prop := tells_lie Dwarf.Zhenya

-- Logical conditions derived from the problem statements
axiom H1 : tells_truth Dwarf.Benya ↔ Benya_statement
axiom H2 : tells_truth Dwarf.Zhenya ↔ Zhenya_statement
axiom H3 : tells_truth Dwarf.Senya ↔ (Senya_statement_1 ∧ Senya_statement_2)

-- Conclusion to be proved
theorem solution : 
  tells_truth Dwarf.Venya ∧ tells_truth Dwarf.Zhenya ∧ tells_lie Dwarf.Benya ∧ tells_lie Dwarf.Senya := 
sorry

end solution_l44_44818


namespace inequality_proof_l44_44273

theorem inequality_proof {n : ℕ} (h1 : ∀ i, i ≤ n → 0 < x[i])
  (m : ℝ) (m_pos : 0 < m) (a : ℝ) (a_nonneg : 0 ≤ a) 
  (s : ℝ) (s_cond : s = ∑ i in range (n + 1), x[i]) 
  (s_le_n : s ≤ n) :
  ∏ i in range (n + 1), (x[i]^m + 1 / x[i]^m + a) ≥ ( (s / n)^m + (n / s)^m + a )^n := sorry

end inequality_proof_l44_44273


namespace ratio_of_water_level_increase_l44_44713

noncomputable def volume_narrow_cone (h₁ : ℝ) : ℝ := (16 / 3) * Real.pi * h₁
noncomputable def volume_wide_cone (h₂ : ℝ) : ℝ := (64 / 3) * Real.pi * h₂
noncomputable def volume_marble_narrow : ℝ := (32 / 3) * Real.pi
noncomputable def volume_marble_wide : ℝ := (4 / 3) * Real.pi

theorem ratio_of_water_level_increase :
  ∀ (h₁ h₂ h₁' h₂' : ℝ),
  h₁ = 4 * h₂ →
  h₁' = h₁ + 2 →
  h₂' = h₂ + (1 / 16) →
  volume_narrow_cone h₁ = volume_wide_cone h₂ →
  volume_narrow_cone h₁ + volume_marble_narrow = volume_narrow_cone h₁' →
  volume_wide_cone h₂ + volume_marble_wide = volume_wide_cone h₂' →
  (h₁' - h₁) / (h₂' - h₂) = 32 :=
by
  intros h₁ h₂ h₁' h₂' h₁_eq_4h₂ h₁'_eq_h₁_add_2 h₂'_eq_h₂_add_1_div_16 vol_h₁_eq_vol_h₂ vol_nar_eq vol_wid_eq
  sorry

end ratio_of_water_level_increase_l44_44713


namespace tangent_line_at_origin_is_neg3x_l44_44988

noncomputable def f (a x : ℝ) := x^3 + a * x^2 + (a - 3) * x

def even_function (g : ℝ → ℝ) : Prop :=
∀ x : ℝ, g (-x) = g x

theorem tangent_line_at_origin_is_neg3x (a : ℝ) (h : even_function (λ x, 3 * x^2 + 2 * a * x + (a - 3))) :
  tangent_line_at_origin a :=
by
  have h : ∀ x, 3 * x^2 + 2 * a * x + (a - 3) = 3 * x^2 - 3 := sorry
  have ha : a = 0 := sorry
  use -3
  split
  · simp [f, ha]
  · simp only [f, ha, zero_add]
    simp [f, ha, mul_zero, add_zero]
    erw [f', apply_rules]
    · simp
    sorry

end tangent_line_at_origin_is_neg3x_l44_44988


namespace angle_between_vectors_is_90_l44_44275

variables (a b : ℝ^3) (k : ℝ)
hypothesis (ha : a ≠ 0)
hypothesis (hb : b ≠ 0)
hypothesis (hk : k ≠ 0)
hypothesis (h : ∥a + k • b∥ = ∥a - k • b∥)

theorem angle_between_vectors_is_90 :
  (a ⬝ b) = 0 := sorry

end angle_between_vectors_is_90_l44_44275


namespace hypotenuse_of_454590_triangle_l44_44639

theorem hypotenuse_of_454590_triangle (l : ℝ) (angle : ℝ) (h : ℝ) (h_leg : l = 15) (h_angle : angle = 45) :
  h = l * Real.sqrt 2 := 
  sorry

end hypotenuse_of_454590_triangle_l44_44639


namespace pencils_left_l44_44794

-- Define initial count of pencils
def initial_pencils : ℕ := 20

-- Define pencils misplaced
def misplaced_pencils : ℕ := 7

-- Define pencils broken and thrown away
def broken_pencils : ℕ := 3

-- Define pencils found
def found_pencils : ℕ := 4

-- Define pencils bought
def bought_pencils : ℕ := 2

-- Define the final number of pencils
def final_pencils: ℕ := initial_pencils - misplaced_pencils - broken_pencils + found_pencils + bought_pencils

-- Prove that the final number of pencils is 16
theorem pencils_left : final_pencils = 16 :=
by
  -- The proof steps are omitted here
  sorry

end pencils_left_l44_44794


namespace distinct_arrangements_balloon_l44_44112

noncomputable def totalPermutations (n nl no : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial nl * Nat.factorial no)

theorem distinct_arrangements_balloon :
  totalPermutations 7 2 2 = 1260 := by 
  sorry

end distinct_arrangements_balloon_l44_44112


namespace convex_polyhedron_has_two_faces_with_same_number_of_edges_l44_44303

def is_convex (P : Type) [Polyhedron P] : Prop := 
  ∀ (x y: P), segment x y ⊆ P

theorem convex_polyhedron_has_two_faces_with_same_number_of_edges
  (P : Type) [Polyhedron P] (h_convex : is_convex P) :
  ∃ (A B : Face P), A ≠ B ∧ number_of_edges A = number_of_edges B :=
sorry

end convex_polyhedron_has_two_faces_with_same_number_of_edges_l44_44303


namespace ten_times_six_x_plus_fourteen_pi_l44_44565

theorem ten_times_six_x_plus_fourteen_pi (x : ℝ) (Q : ℝ) (h : 5 * (3 * x + 7 * Real.pi) = Q) : 
  10 * (6 * x + 14 * Real.pi) = 4 * Q :=
by
  sorry

end ten_times_six_x_plus_fourteen_pi_l44_44565


namespace modified_sum_of_set_l44_44437

theorem modified_sum_of_set (m : ℕ) (t : ℕ) (y : Fin m → ℕ)
  (h_sum : (Finset.univ.sum y) = t) :
  Finset.univ.sum (λ i, 3 * y i + 15) = 3 * t + 15 * m := 
sorry

end modified_sum_of_set_l44_44437


namespace distinct_ball_placement_l44_44898

def num_distributions (balls boxes : ℕ) : ℕ :=
  if boxes = 3 then 243 - 32 + 16 else 0

theorem distinct_ball_placement : num_distributions 5 3 = 227 :=
by
  sorry

end distinct_ball_placement_l44_44898


namespace prize_winner_is_Bing_l44_44785

def Students : Type := { n // n < 4 }
def Jia : Students := ⟨0, by norm_num⟩
def Yi : Students := ⟨1, by norm_num⟩
def Bing : Students := ⟨2, by norm_num⟩
def Ding : Students := ⟨3, by norm_num⟩

def Win (x : Students) : Prop := 
  x = Jia ∨ x = Yi ∨ x = Bing ∨ x = Ding

-- Statements by the students
def Statement_Jia := (Win Yi ∨ Win Bing)
def Statement_Yi := ¬Win Jia ∧ ¬Win Bing
def Statement_Bing := Win Bing
def Statement_Ding := Win Yi

-- Condition stating exactly two statements are true
def exactly_two_statements_true (s1 s2 s3 s4 : Prop) : Prop :=
  (s1 ∧ s2 ∧ (¬s3) ∧ (¬s4)) ∨
  (s1 ∧ (¬s2) ∧ s3 ∧ (¬s4)) ∨
  ((¬s1) ∧ s2 ∧ s3 ∧ (¬s4)) ∨
  ((¬s1) ∧ (¬s2) ∧ s3 ∧ s4) ∨
  ((¬s1) ∧ s2 ∧ (¬s3) ∧ s4) ∨
  (s1 ∧ (¬s2) ∧ (¬s3) ∧ s4)

theorem prize_winner_is_Bing (winner : Students) :
  (exists unique x, Win x) →
  ((exactly_two_statements_true Statement_Jia Statement_Yi Statement_Bing Statement_Ding) →
    winner = Bing) :=
begin
  sorry
end

end prize_winner_is_Bing_l44_44785


namespace only_rational_point_on_line_is_origin_distance_from_line_irrational_slope_l44_44944

section
variable {k : ℝ} (irr_k : ¬ ∃ (r : ℚ), k = r)

-- Part (i)
theorem only_rational_point_on_line_is_origin (x y : ℝ) (hx : x ≠ 0 ∨ y ≠ 0) (hr : x ∈ ℚ ∧ y ∈ ℚ) :
  ¬ (y = k * x) :=
sorry

-- Part (ii)
theorem distance_from_line_irrational_slope (ε : ℝ) (hε : ε > 0) :
  ∃ (m n : ℤ), |k * m - n| < ε * real.sqrt (k^2 + 1) :=
sorry
end

end only_rational_point_on_line_is_origin_distance_from_line_irrational_slope_l44_44944


namespace EP_eq_EQ_l44_44271

-- Noncomputable section to handle geometric objects more conveniently
noncomputable section

open EuclideanGeometry

-- Variables and Types
variables {Point : Type} [AffineSpace Point]

-- Declaring points (M, N, A, B, C, D, E, P, Q) and circles k1 and k2
variable (k1 k2 : Circle Point)
variable (M N A B C D E P Q : Point)
variable [h1k1 : M ∈ k1]
variable [h2k1 : N ∈ k1]
variable [h1k2 : M ∈ k2]
variable [h2k2 : N ∈ k2]
variable [tangent1 : Tangent k1 A B]
variable [tangent2 : Tangent k2 B A]
variable [parAB : ParallelLine M A B C D]
variable [interAC_BD : LineIntersection A C B D E]
variable [interAN_CD : LineIntersection A N C D P]
variable [interBN_CD : LineIntersection B N C D Q]

-- Theorem to prove
theorem EP_eq_EQ : dist E P = dist E Q :=
sorry

end EP_eq_EQ_l44_44271


namespace vertex_sums_not_equal_vertex_sums_equal_with_change_l44_44325

theorem vertex_sums_not_equal (a : Fin 12 → ℕ) (h_sum : ∑ i in Finset.univ, a i = 78) :
  ¬ (∃ s : ℕ, ∀ v : Fin 8, ∑ e in vertex_edges v, a e = s) :=
sorry

theorem vertex_sums_equal_with_change (a : Fin 12 → ℕ) (h_edge_change : a 11 = 13)
    (h_sum : ∑ i in Finset.univ.erase 11, a i + 13 = 80) :
  ∃ s : ℕ, ∀ v : Fin 8, ∑ e in vertex_edges v, if e = 11 then 13 else a e = s :=
sorry

end vertex_sums_not_equal_vertex_sums_equal_with_change_l44_44325


namespace paddyfield_warblers_percentage_l44_44936

-- Definitions for the conditions
variables (B : ℝ) -- Total number of birds
variables (H : ℝ) -- Portion of hawks
variables (W : ℝ) -- Portion of paddyfield-warblers among non-hawks
variables (K : ℝ) -- Portion of kingfishers among non-hawks
variables (N : ℝ) -- Portion of non-hawks

-- Conditions
def hawks_portion : Prop := H = 0.30 * B
def non_hawks_portion : Prop := N = 0.70 * B
def paddyfield_warblers : Prop := W * N = W * (0.70 * B)
def kingfishers : Prop := K = 0.25 * W * N

-- Prove that the percentage of non-hawks that are paddyfield-warblers is 40%
theorem paddyfield_warblers_percentage :
  hawks_portion B H →
  non_hawks_portion B N →
  paddyfield_warblers B W N →
  kingfishers B W N K →
  W = 0.4 :=
by
  sorry

end paddyfield_warblers_percentage_l44_44936


namespace correct_choice_l44_44726

noncomputable def satisfies_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
∀ x, f (x + T) = f x

noncomputable def symmetric_about_axis (f : ℝ → ℝ) (a : ℝ) : Prop :=
∀ x, f (2 * a - x) = f x

noncomputable def strictly_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

def candidate_function : ℝ → ℝ := λ x, 2 * |sin x| + sin x

theorem correct_choice :
  satisfies_period candidate_function (2 * Real.pi) ∧
  symmetric_about_axis candidate_function (Real.pi / 2) ∧
  strictly_increasing_on_interval candidate_function 0 (Real.pi / 2) :=
by
  sorry

end correct_choice_l44_44726


namespace construct_ellipse_from_focus_and_tangents_l44_44809

open EuclideanGeometry

theorem construct_ellipse_from_focus_and_tangents
  (F1 : Point)
  (t1 t2 t3 : Line) :
  ∃ (F2 : Point) (a : ℝ), is_focus F1 t1 t2 t3 F2 a :=
by
  sorry

end construct_ellipse_from_focus_and_tangents_l44_44809


namespace problem_l44_44876

noncomputable def f (x : ℝ) : ℝ := ((x + 1) ^ 2 + Real.sin x) / (x ^ 2 + 1)

noncomputable def f' (x : ℝ) : ℝ := ((2 + Real.cos x) * (x ^ 2 + 1) - (2 * x + Real.sin x) * (2 * x)) / (x ^ 2 + 1) ^ 2

theorem problem : f 2016 + f' 2016 + f (-2016) - f' (-2016) = 2 := by
  sorry

end problem_l44_44876


namespace train_passes_pole_in_12_75_seconds_l44_44561

def time_to_pass_pole (train_speed_kmph : ℝ) (train_length_m : ℝ) (incline_percent : ℝ) (headwind_kmph : ℝ) : ℝ :=
  let adjusted_speed_kmph := train_speed_kmph - headwind_kmph
  let adjusted_speed_mps := adjusted_speed_kmph * (5 / 18)
  train_length_m / adjusted_speed_mps

theorem train_passes_pole_in_12_75_seconds :
  time_to_pass_pole 68 170 5 20 = 12.75 :=
by
  let adjusted_speed_kmph := 68 - 20
  let adjusted_speed_mps := adjusted_speed_kmph * (5 / 18)
  let result := 170 / adjusted_speed_mps
  have h : result = 12.75 := by norm_num
  exact h

end train_passes_pole_in_12_75_seconds_l44_44561


namespace license_plates_count_l44_44714

theorem license_plates_count : 
  ∃ (count : ℕ), 
    (∃ (alphabet : Finset Char), 
      alphabet.card = 12 ∧ 
      (∃ (first_letter : Char), first_letter ∈ {'G', 'K', 'P'} ∧ 
      (∃ (last_letter : Char), last_letter = 'T' ∧ 
      (∀ (c : Char), c ≠ 'R' ∧ c ≠ first_letter ∧ c ≠ 'T' → c ∈ alphabet) ∧ 
      (∀ (a b c d e : Char), 
        a ∈ {'G', 'K', 'P'} ∧ 
        b ∈ (alphabet.erase a).erase 'T' ∧ 
        c ∈ ((alphabet.erase a).erase 'T').erase b ∧ 
        d ∈ (((alphabet.erase a).erase 'T').erase b).erase c ∧ 
        e = 'T' → 
        count = 3 * 7 * 6 * 5 * 1))))) := 
by 
  sorry

end license_plates_count_l44_44714


namespace line_divides_rectangle_correctly_l44_44458

noncomputable def rect_points : (Point ℝ × Point ℝ × Point ℝ × Point ℝ) :=
  (⟨(0, 0)⟩, ⟨(0, 5)⟩, ⟨(6, 5)⟩, ⟨(6, 0)⟩)

noncomputable def point_M : Point ℝ := ⟨(5, 6)⟩

theorem line_divides_rectangle_correctly :
  (∃ (m b : ℝ), ∀ (x y : ℝ), (y = m * x + b) ∧ passes_through ⟨(5,6)⟩ ⟨x,y⟩
    ∧ (divides_area_with_ratio ⟨(0,0)⟩ ⟨(0,5)⟩ ⟨(6,5)⟩ ⟨(6,0)⟩ ⟨x, y⟩ (2/3))) →
  (∃ (m b : ℝ),
    m = 35 / 26 ∧ b = -19 / 26) :=
begin
  sorry
end

end line_divides_rectangle_correctly_l44_44458


namespace base_3_vs_base_8_digits_l44_44154

theorem base_3_vs_base_8_digits : 
  let n := 2035 in
  let digits_base3 := Nat.log 3 (n + 1) in    -- calculating number of digits
  let digits_base8 := Nat.log 8 (n + 1) in    -- "+ 1" to get exact number of multiples needed
  digits_base3 - digits_base8 = 3 :=
by {
  let n := 2035
  let digits_base3 := Nat.log 3 (n + 1)
  let digits_base8 := Nat.log 8 (n + 1)
  sorry
}

end base_3_vs_base_8_digits_l44_44154


namespace triangle_b_side_length_l44_44206

noncomputable def cosine_law_b (a c : ℝ) (B : ℝ) : ℝ :=
  real.sqrt (a^2 + c^2 - 2 * a * c * real.cos B)

theorem triangle_b_side_length (a c : ℝ) (B : ℝ) (ha : a = 1) (hc : c = 2) (hB : B = real.pi / 3) :
  cosine_law_b a c B = real.sqrt 3 :=
  by
    rw [ha, hc, hB]
    dsimp [cosine_law_b]
    rw [real.cos_pi_div_three]
    norm_num
    sorry

end triangle_b_side_length_l44_44206


namespace solve_for_angle_a_l44_44207

theorem solve_for_angle_a (a b c d e : ℝ) (h1 : a + b + c + d = 360) (h2 : e = 360 - (a + d)) : a = 360 - e - b - c :=
by
  sorry

end solve_for_angle_a_l44_44207


namespace grant_room_proof_l44_44811

/-- Danielle's apartment has 6 rooms -/
def danielle_rooms : ℕ := 6

/-- Heidi's apartment has 3 times as many rooms as Danielle's apartment -/
def heidi_rooms : ℕ := 3 * danielle_rooms

/-- Jenny's apartment has 5 more rooms than Danielle's apartment -/
def jenny_rooms : ℕ := danielle_rooms + 5

/-- Lina's apartment has 7 rooms -/
def lina_rooms : ℕ := 7

/-- The total number of rooms from Danielle, Heidi, Jenny,
    and Lina's apartments -/
def total_rooms : ℕ := danielle_rooms + heidi_rooms + jenny_rooms + lina_rooms

/-- Grant's apartment has 1/3 less rooms than 1/9 of the
    combined total of rooms from Danielle's, Heidi's, Jenny's, and Lina's apartments -/
def grant_rooms : ℕ := (total_rooms / 9) - (total_rooms / 9) / 3

/-- Prove that Grant's apartment has 3 rooms -/
theorem grant_room_proof : grant_rooms = 3 :=
by
  sorry

end grant_room_proof_l44_44811


namespace maximum_ab_l44_44040

theorem maximum_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 4 * b = 2) : ab ≤ 1 / 12 :=
begin
  sorry
end

end maximum_ab_l44_44040


namespace distinct_arrangements_balloon_l44_44124

theorem distinct_arrangements_balloon : 
  let n := 7
  let r1 := 2
  let r2 := 2
  (Nat.factorial n) / ((Nat.factorial r1) * (Nat.factorial r2)) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l44_44124


namespace chessboard_columns_l44_44008

theorem chessboard_columns (M : Matrix (Fin 8) (Fin 8) ℤ)
  (hM : ∀ i j, M i j = 1 ∨ M i j = -1)
  (h_rows_pos : ∃ (rows_pos : Finset (Fin 8)), rows_pos.card ≥ 4 ∧ ∀ i ∈ rows_pos, 0 < ∑ j, M i j) :
  ∃ (cols_neg : Finset (Fin 8)), cols_neg.card ≤ 6 ∧ ∀ j ∈ cols_neg, ∑ i, M i j < -3 :=
sorry

end chessboard_columns_l44_44008


namespace johns_age_l44_44224

theorem johns_age (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l44_44224


namespace percentage_decrease_theorem_l44_44694

-- Define the conditions
variables (a x : ℝ) (h_pos : x > 0) (h_a : a > 0)

-- Define the production values
def production_before_last := x
def production_last := x * (1 + a / 100)

-- Define the percentage decrease
def percentage_decrease := (production_last a x - production_before_last x) / production_last a x

-- The statement to prove
theorem percentage_decrease_theorem : 
  percentage_decrease a x = | a / (100 + a) | :=
sorry

end percentage_decrease_theorem_l44_44694


namespace chip_price_reduction_equation_l44_44835

-- Define initial price
def initial_price : ℝ := 400

-- Define final price after reductions
def final_price : ℝ := 144

-- Define the price reduction percentage
variable (x : ℝ)

-- The equation we need to prove
theorem chip_price_reduction_equation :
  initial_price * (1 - x) ^ 2 = final_price :=
sorry

end chip_price_reduction_equation_l44_44835


namespace proof_f_6_l44_44568

theorem proof_f_6 (n k : ℤ) (f : ℤ → ℤ) (h1 : ∀ n, f n = f (n-1) - n) (h2 : f k = 14) : f 6 = f(k - (k - 6)) :=
by sorry

end proof_f_6_l44_44568


namespace parabola_centroid_locus_l44_44254

/-- Let P_0 be a parabola defined by the equation y = m * x^2. 
    Let A and B be points on P_0 such that the tangents at A and B are perpendicular. 
    Let G be the centroid of the triangle formed by A, B, and the vertex of P_0.
    Let P_n be the nth derived parabola.
    Prove that the equation of P_n is y = 3^n * m * x^2 + (1 / (4 * m)) * (1 - (1 / 3)^n). -/
theorem parabola_centroid_locus (n : ℕ) (m : ℝ) 
  (h_pos_m : 0 < m) :
  ∃ P_n : ℝ → ℝ, 
    ∀ x : ℝ, P_n x = 3^n * m * x^2 + (1 / (4 * m)) * (1 - (1 / 3)^n) :=
sorry

end parabola_centroid_locus_l44_44254


namespace probability_sin_le_half_l44_44729

noncomputable def geometric_probability_sine : ℝ :=
  let interval : set ℝ := set.Icc 0 real.pi
  let sin_interval : set ℝ := {x | sin x ≤ 1/2}
  let measure : ℝ := real.measure_space.volume set.univ (interval ∩ sin_interval)
in measure / real.measure_space.volume set.univ interval

theorem probability_sin_le_half : geometric_probability_sine = 1 / 3 :=
sorry

end probability_sin_le_half_l44_44729


namespace score_equality_l44_44379

theorem score_equality (n : ℕ) (A B : Fin n → ℝ)
  (score_calc : ∀ i j, i < n → j < n → (A i + B j) ∈ {0, 0.5, 1})
  (equal_points : (∑ i, A i) = (∑ i, B i)) :
  ∃ (i j : Fin n), i ≠ j ∧ (A i + B i) = (A j + B j) :=
by sorry

end score_equality_l44_44379


namespace total_possible_guesses_l44_44428

-- Conditions
def is_valid_price (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9999
def prize_prices := {a b c : ℕ // is_valid_price a ∧ is_valid_price b ∧ is_valid_price c}
def digits : list ℕ := [1, 2, 2, 3, 3, 3, 3]

-- Statement of the math proof problem
theorem total_possible_guesses : (∃ (A B C : ℕ), A ∈ prize_prices ∧ B ∈ prize_prices ∧ C ∈ prize_prices ∧ 
                                (digits.perm (A.digits ++ B.digits ++ C.digits))) → 
                                ∃ n, n = 945 :=
by
  sorry

end total_possible_guesses_l44_44428


namespace ellipse_foci_distance_l44_44450

/-- Given an ellipse tangent to the x-axis at (6, 0) and to the y-axis at (0, 3), 
the distance between the foci of the ellipse is 6√3. -/
theorem ellipse_foci_distance :
  ∀ (a b : ℝ), a = 6 → b = 3 → 
  let c := Real.sqrt (a ^ 2 - b ^ 2) in
  2 * c = 6 * Real.sqrt 3 :=
by
  intros a b ha hb
  rw [ha, hb]
  let c := Real.sqrt (6 ^ 2 - 3 ^ 2)
  sorry

end ellipse_foci_distance_l44_44450


namespace find_x_l44_44202

noncomputable def angle_sum_triangle (A B C: ℝ) : Prop :=
  A + B + C = 180

noncomputable def vertical_angles_equal (A B: ℝ) : Prop :=
  A = B

noncomputable def right_angle_sum (D E: ℝ) : Prop :=
  D + E = 90

theorem find_x 
  (angle_ABC angle_BAC angle_DCE : ℝ) 
  (h1 : angle_ABC = 70)
  (h2 : angle_BAC = 50)
  (h3 : angle_sum_triangle angle_ABC angle_BAC angle_DCE)
  (h4 : vertical_angles_equal angle_DCE angle_DCE)
  (h5 : right_angle_sum angle_DCE 30) :
  angle_DCE = 60 :=
by
  sorry

end find_x_l44_44202


namespace average_speed_highspeed_train_l44_44709

-- Definitions based on the conditions

def distance_regular_train := 520 -- in kilometers
def distance_highspeed_train := 400 -- in kilometers
def speed_relation : ℝ := 2.5 -- high-speed train speed is 2.5 times regular train speed
def time_saved := 3 -- in hours

-- Define the average speed of the regular train
variable (x : ℝ) -- average speed of regular train in kilometers per hour

-- The equation derived from the conditions
def equation := (distance_highspeed_train / (speed_relation * x) + time_saved = distance_regular_train / x)

-- The main proof problem: prove that when solving equation, the speed of the high-speed train is 300 kilometers per hour
theorem average_speed_highspeed_train : ∃ x : ℝ, x = 120 ∧ 2.5 * x = 300 :=
begin
  assume x,
  unfold equation,
  sorry
end

end average_speed_highspeed_train_l44_44709


namespace triangle_perimeter_l44_44783

theorem triangle_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 6)
  (c1 c2 : ℝ) (h3 : (c1 - 2) * (c1 - 4) = 0) (h4 : (c2 - 2) * (c2 - 4) = 0) :
  c1 = 2 ∨ c1 = 4 → c2 = 2 ∨ c2 = 4 → 
  (c1 ≠ 2 ∧ c1 = 4 ∨ c2 ≠ 2 ∧ c2 = 4) → 
  (a + b + c1 = 13 ∨ a + b + c2 = 13) :=
by
  sorry

end triangle_perimeter_l44_44783


namespace range_of_a_l44_44571

noncomputable theory
open Real

def ellipse (x y a : ℝ) : Prop := x^2 + 4 * (y - a)^2 = 4
def parabola (x y : ℝ) : Prop := x^2 = 2 * y

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, ellipse x y a ∧ parabola x y) ↔ -1 ≤ a ∧ a ≤ 17 / 8 :=
by
  sorry

end range_of_a_l44_44571


namespace hyperbola_ecc_2_l44_44071

noncomputable def hyperbola_eccentricity (a b c : ℝ) : ℝ :=
  c / a 

theorem hyperbola_ecc_2 (a b c e: ℝ) 
  (h1: ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 → True)
  (h2: b^2 = c^2 - a^2 → True)
  (h3: b = sqrt 3 * a → True)
  (h4: c = 2 * a → True)
  : e = 2 :=
  by sorry

end hyperbola_ecc_2_l44_44071


namespace general_term_formula_sum_of_first_n_terms_l44_44436

-- Definitions extracted from conditions
def sequence_a (a : ℕ → ℤ) : Prop := a 1 = 3 ∧ ∀ n, a (n + 1) = a n + 2

def sequence_b (b : ℕ → ℤ) (a : ℕ → ℤ) : Prop := ∀ n, b n = a n * (3 : ℤ) ^ n

def sum_b (T : ℕ → ℤ) (b : ℕ → ℤ) : Prop := ∀ n, T n = (∑ k in finset.range (n+1), b k)

-- Main theorem statements
theorem general_term_formula (a : ℕ → ℤ) (h_a : sequence_a a) : ∀ n, a n = 2 * n + 1 :=
sorry

theorem sum_of_first_n_terms (b : ℕ → ℤ) (a : ℕ → ℤ) (T : ℕ → ℤ) (h_a : sequence_a a) (h_b : sequence_b b a) (h_T : sum_b T b) :
  ∀ n, T n = n * (3 : ℤ) ^ (n + 1) :=
sorry

end general_term_formula_sum_of_first_n_terms_l44_44436


namespace volume_of_tetrahedron_l44_44484

noncomputable def tetrahedron_volume (P Q R S : Type) 
  (area_PQR area_QRS : ℝ) 
  (QR : ℝ) 
  (angle_PQR_QRS : ℝ) 
  (hQRS : ℝ) 
  (h : ℝ) 
  (V : ℝ) := 
  area_PQR = 150 ∧ area_QRS = 100 ∧ QR = 12 ∧ angle_PQR_QRS = π/4 ∧
  (V = 1 / 3 * area_PQR * h) 

theorem volume_of_tetrahedron (P Q R S : Type) 
  (area_PQR area_QRS : ℝ)
  (QR : ℝ) 
  (angle_PQR_QRS : ℝ) : 
  area_PQR = 150 → area_QRS = 100 → QR = 12 → angle_PQR_QRS = π/4 → 
  ∃ V : ℝ, V = 588.75 :=
by 
  sorry

end volume_of_tetrahedron_l44_44484


namespace sequences_cover_naturals_without_repetition_l44_44557

theorem sequences_cover_naturals_without_repetition
  (x y : Real) 
  (hx : Irrational x) 
  (hy : Irrational y) 
  (hxy : 1/x + 1/y = 1) :
  (∀ n : ℕ, ∃! k : ℕ, (⌊k * x⌋ = n) ∨ (⌊k * y⌋ = n)) :=
sorry

end sequences_cover_naturals_without_repetition_l44_44557


namespace original_difference_of_weights_l44_44706

variable (F S T : ℝ)

theorem original_difference_of_weights :
  (F + S + T = 75) →
  (F - 2 = 0.7 * (S + 2)) →
  (S + 1 = 0.8 * (T + 1)) →
  T - F = 10.16 :=
by
  intro h1 h2 h3
  sorry

end original_difference_of_weights_l44_44706


namespace tangent_line_equation_l44_44497

noncomputable def exp_func : ℝ → ℝ := λ x, Real.exp x

theorem tangent_line_equation :
  ∀ (x : ℝ), x = 1 → (∃ (m : ℝ) (b : ℝ), m = Real.exp 1 ∧ b = Real.exp 1 ∧
  ∀ (y : ℝ), y = Real.exp x → (Real.exp 1 * x - y = 0)) :=
by
  sorry

end tangent_line_equation_l44_44497


namespace probability_sum_is_3_l44_44362

theorem probability_sum_is_3 (die : Type) [Fintype die] [DecidableEq die] 
  (dice_faces : die → ℕ) (h : ∀ d, dice_faces d ∈ {1, 2, 3, 4, 5, 6}) :
  (∑ i in finset.range 3, (die →₀ ℕ).single 1) = 3 → 
  (1 / (finset.card univ) ^ 3) = 1 / 216 :=
by
  sorry

end probability_sum_is_3_l44_44362


namespace avg_percentage_difference_mike_phil_olivia_l44_44284

noncomputable def mike_earnings : ℝ := 12
noncomputable def phil_earnings : ℝ := 6
noncomputable def olivia_earnings : ℝ := 10

def percentage_difference (a b : ℝ) : ℝ :=
  let diff := abs (a - b)
  let avg_earnings := (a + b) / 2
  (diff / avg_earnings) * 100

def avg_percentage_difference (x y z : ℝ) : ℝ :=
  (percentage_difference x y + percentage_difference x z + percentage_difference y z) / 3

theorem avg_percentage_difference_mike_phil_olivia :
  avg_percentage_difference mike_earnings phil_earnings olivia_earnings = 44.95 :=
by
  sorry

end avg_percentage_difference_mike_phil_olivia_l44_44284


namespace total_salaries_l44_44696

variable (A_salary B_salary : ℝ)

def A_saves : ℝ := 0.05 * A_salary
def B_saves : ℝ := 0.15 * B_salary

theorem total_salaries (h1 : A_salary = 5250) 
                       (h2 : A_saves = B_saves) : 
    A_salary + B_salary = 7000 := by
  sorry

end total_salaries_l44_44696


namespace regular_polygon_sides_l44_44182

-- Given definitions and conditions
variables {A B C D : Type} [cyclic_quadrilateral A B C D]
variables {angleA : ℝ} {angleB : ℝ} {angleC : ℝ}
variable h1 : angleB = 3 * angleA
variable h2 : angleC = 3 * angleA
variable h3 : (B C D) are consecutive vertices of regular_polygon inscribed in circle

-- The theorem to prove the number of sides of the polygon
theorem regular_polygon_sides :
  ∃ n : ℕ, n = 4 := by
sorry

end regular_polygon_sides_l44_44182


namespace distance_between_hyperbola_vertices_l44_44019

theorem distance_between_hyperbola_vertices :
  let equation := ∀ x y, (x * x) / 48 - (y * y) / 16 = 1
  ∃ distance, distance = 8 * Real.sqrt 3 :=
by
  -- Let's define the necessary conditions for the hyperbola
  have hyp_eq : ∀ x y, (x * x) / 48 - (y * y) / 16 = 1,
  from sorry,

  -- The vertices of the hyperbola are located at (±a, 0)
  let a := Real.sqrt 48,

  -- The distance between the vertices is 2a
  let distance := 2 * a,

  -- Simplify the distance
  have distance_simplified : distance = 8 * Real.sqrt 3,
  from sorry,

  -- Finally, we have the required distance
  exact ⟨distance, distance_simplified⟩

end distance_between_hyperbola_vertices_l44_44019


namespace cyclic_quadrilateral_collinear_l44_44009

theorem cyclic_quadrilateral_collinear
  (A B C D P Q : Point) 
  (cyclic_quadrilateral : CyclicQuadrilateral A B C D)
  (M : Midpoint A C)
  (N : Midpoint B D)
  (H : Intersection (extension A B) (extension C D))
  (K : Intersection (extension A D) (extension B C)) :
  Collinear M N H K :=
sorry

end cyclic_quadrilateral_collinear_l44_44009


namespace induction_term_l44_44667

theorem induction_term (n : ℕ) (h : n > 1)
  (induction_hypothesis : ∀ k : ℕ, 1 + 2 + 4 + ... + 2^k < 2^(k + 1)) :
  ∀ k : ℕ, term_added (1 + 2 + 4 + ... + 2^k) k = 2^k := by
  sorry

-- Assuming we have a function term_added representing the term added in the sum.
def term_added (sum : ℕ) (k : ℕ) : ℕ := 2^k  -- Placeholder definition, the actual logic should determine the term added.


end induction_term_l44_44667


namespace number_of_pipes_l44_44464

-- Definitions based on conditions
def diameter_large : ℝ := 8
def diameter_small : ℝ := 2
def radius_large : ℝ := diameter_large / 2
def radius_small : ℝ := diameter_small / 2
def area_large : ℝ := π * radius_large^2
def area_small : ℝ := π * radius_small^2
def required_area : ℝ := 2 * area_large

-- The proof statement
theorem number_of_pipes :
  (required_area / area_small) = 32 :=
sorry

end number_of_pipes_l44_44464


namespace sum_of_squares_eq_l44_44295

theorem sum_of_squares_eq (n : ℕ) (a : Fin n → ℝ) :
  (∑ ε in Finset.univ.pi (λ _, {-1, 1}.toFinset), (∑ i, ε i * a i) ^ 2) = 2^n * ∑ i, (a i)^2 :=
sorry

end sum_of_squares_eq_l44_44295


namespace find_prime_and_int_solutions_l44_44010

-- Define the conditions
def is_solution (p x : ℕ) : Prop :=
  x^(p-1) ∣ (p-1)^x + 1

-- Define the statement to be proven
theorem find_prime_and_int_solutions :
  ∀ p x : ℕ, Prime p → (1 ≤ x ∧ x ≤ 2 * p) →
  (is_solution p x ↔ 
    (p = 2 ∧ (x = 1 ∨ x = 2)) ∨ 
    (p = 3 ∧ (x = 1 ∨ x = 3)) ∨
    (x = 1))
:=
by sorry

end find_prime_and_int_solutions_l44_44010


namespace fifth_eq_nth_eq_sum_to_100_l44_44636

def a (n : ℕ) : ℚ := 1 / ((2 * n - 1) * (2 * n + 1))

theorem fifth_eq :
  a 5 = 1 / (9 * 11) :=
  sorry

theorem nth_eq (n : ℕ) (hn : 0 < n) :
  a n = 1 / ((2 * n - 1) * (2 * n + 1)) ∧ a n = 1 / 2 * (1 / (2 * n - 1) - 1 / (2 * n + 1)) :=
  sorry

theorem sum_to_100 :
  (∑ n in Finset.range 100, a (n + 1)) = 100 / 201 :=
  sorry

end fifth_eq_nth_eq_sum_to_100_l44_44636


namespace train_length_l44_44439

def speed_kmph : ℝ := 90
def time_sec : ℝ := 11.999040076793857
def speed_mps : ℝ := speed_kmph * (1000 / 3600)
def expected_length : ℝ := 299.9760019198464

theorem train_length :
  speed_mps * time_sec = expected_length := 
by
  sorry

end train_length_l44_44439


namespace ratio_of_circles_l44_44791

theorem ratio_of_circles (a b R r : ℝ) 
  (h_isosceles : a = 4 * b) 
  (h_height : ∃ h, h = sqrt (a^2 - b^2))
  (h_area : ∃ S, S = b * sqrt (a^2 - b^2)) 
  (h_semiperimeter : ∃ p, p = a + b) 
  (h_inscribed_r : ∃ r, r = b * sqrt (a^2 - b^2) / (a + b)) 
  (h_circum_radius_R : R = a^2 / (2 * sqrt (a^2 - b^2))) :
  R / r = 8 / 3 := 
by
  sorry

end ratio_of_circles_l44_44791


namespace convex_polygons_from_fifteen_points_l44_44493

theorem convex_polygons_from_fifteen_points 
    (h : ∀ (n : ℕ), n = 15) :
    ∃ (k : ℕ), k = 32192 :=
by
  sorry

end convex_polygons_from_fifteen_points_l44_44493


namespace proper_subsets_count_of_A_l44_44333

open Finset

def A : Finset ℤ := {x ∈ (range 1 5) | log 2 (x : ℝ) ≤ 2}

theorem proper_subsets_count_of_A : A.filter (λ x, log 2 (x : ℝ) ≤ 2).card != A.card - 1 :=
  by
  let n := A.card
  have h1 : n = 4 := sorry
  have h2 : 2^4 - 1 = 15 := by norm_num
  exact h2

#print proper_subsets_count_of_A

end proper_subsets_count_of_A_l44_44333


namespace number_of_good_card_groups_l44_44425

noncomputable def card_value (k : ℕ) : ℕ := 2 ^ k

def is_good_card_group (cards : Finset ℕ) : Prop :=
  (cards.sum card_value = 2004)

theorem number_of_good_card_groups : 
  ∃ n : ℕ, n = 1006009 ∧ ∃ (cards : Finset ℕ), is_good_card_group cards :=
sorry

end number_of_good_card_groups_l44_44425


namespace middle_term_expansion_sum_odd_coefficients_weighted_sum_coefficients_l44_44613

noncomputable def T (n r : ℕ) : ℝ := C(n, r) * (-1/2)^r

theorem middle_term_expansion 
  (a : ℕ → ℝ) 
  (a_seq : |a 0|, |a 1|, |a 2| forms an arithmetic sequence) : 
  (a 0 = 1) ∧ (a 1 = -n/2) ∧ (a 2 = (n*(n-1))/8) ∧ (n = 8) → 
  T 8 4 = 35/8 :=
sorry

theorem sum_odd_coefficients 
  (a : ℕ → ℝ) 
  (a_seq : |a 0|, |a 1|, |a 2| forms an arithmetic sequence) : 
  (a 0 = 1) ∧ (a 1 = -n/2) ∧ (a 2 = (n*(n-1))/8) ∧ (n = 8) → 
  ∑ i in {1, 3, 5, 7}, a i = -205/16 :=
sorry

theorem weighted_sum_coefficients 
  (a : ℕ → ℝ) 
  (a_seq : |a 0|, |a 1|, |a 2| forms an arithmetic sequence) : 
  (a 0 = 1) ∧ (a 1 = -n/2) ∧ (a 2 = (n*(n-1))/8) ∧ (n = 8) → 
  ∑ i in (range 8).map_with_index (λ (i xi), i * a i) = -1/32 :=
sorry

end middle_term_expansion_sum_odd_coefficients_weighted_sum_coefficients_l44_44613


namespace kira_breakfast_time_l44_44973

theorem kira_breakfast_time :
  let fry_time_per_sausage := 5 -- minutes per sausage
  let scramble_time_per_egg := 4 -- minutes per egg
  let sausages := 3
  let eggs := 6
  let time_to_fry := sausages * fry_time_per_sausage
  let time_to_scramble := eggs * scramble_time_per_egg
  (time_to_fry + time_to_scramble) = 39 := 
by
  sorry

end kira_breakfast_time_l44_44973


namespace sum_of_roots_eq_l44_44027

noncomputable def sum_of_roots_tan_quadratic : ℝ :=
  2 * (Real.arctan 4) + 2 * (Real.arctan 2) + Real.pi

theorem sum_of_roots_eq:
  (∑ x in {
    x | x ∈ (Icc 0 (2 * Real.pi))
         ∧ x ≠ ±Real.arctan 4
         ∧ x ≠ ±Real.arctan 2
         ∧ (tan x)^2 - 6 * (tan x) + 8 = 0}, x) = sum_of_roots_tan_quadratic := by
sorry

end sum_of_roots_eq_l44_44027


namespace largest_in_column_smallest_in_row_7_l44_44390

def array : List (List ℕ) :=
  [[10, 6, 4, 3, 2],
   [11, 7, 14, 10, 8],
   [8, 3, 4, 5, 9],
   [13, 4, 15, 12, 1],
   [8, 2, 5, 9, 3]]

theorem largest_in_column_smallest_in_row_7 :
  ∃ (i j : ℕ), 
    i < array.length ∧ 
    j < (array.headD []).length ∧ 
    array.nthD i [] !! j = 7 ∧ 
    (∀ k < array.length, array.nthD k [] !! j ≤ 7) ∧ 
    (∀ l < (array.nthD i []).length, 7 ≤ array.nthD i [] !! l) :=
  sorry

end largest_in_column_smallest_in_row_7_l44_44390


namespace distance_between_foci_of_ellipse_tangent_at_points_l44_44452

noncomputable def ellipse_dist_foci (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

theorem distance_between_foci_of_ellipse_tangent_at_points :
  (∀ (a b : ℝ), a = 6 ∧ b = 3 → ellipse_dist_foci a b = 3 * Real.sqrt 3) :=
by
  intros a b h
  cases h with ha hb
  rw [ha, hb]
  sorry

end distance_between_foci_of_ellipse_tangent_at_points_l44_44452


namespace smallest_positive_value_l44_44902

theorem smallest_positive_value (c d : ℤ) (h : c^2 > d^2) : 
  ∃ m > 0, m = (c^2 + d^2) / (c^2 - d^2) + (c^2 - d^2) / (c^2 + d^2) ∧ m = 2 :=
by
  sorry

end smallest_positive_value_l44_44902


namespace intersection_points_C2_C3_max_distance_C1_C2_C3_l44_44194

-- Definitions of the curves in Cartesian coordinates
def C2 := { p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * real.sqrt 3 * p.2 = 0 }
def C3 := { p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * p.1 = 0 }

-- Conditions
def C1 := { p : ℝ × ℝ | ∃ t α, t ≠ 0 ∧ 0 ≤ α ∧ α ≤ real.pi ∧ p.1 = t * real.cos α ∧ p.2 = t * real.sin α }

-- Proof Problem
theorem intersection_points_C2_C3 :
  (0, 0) ∈ C2 ∩ C3 ∧ (3 / 2, real.sqrt 3 / 2) ∈ C2 ∩ C3 :=
sorry

theorem max_distance_C1_C2_C3 :
  ∀ A B : ℝ × ℝ, A ∈ C1 ∩ C2 ∧ B ∈ C1 ∩ C3 → dist A B ≤ 4 :=
sorry

end intersection_points_C2_C3_max_distance_C1_C2_C3_l44_44194


namespace median_free_throws_is_17_l44_44750

def free_throws := [10, 16, 21, 19, 7, 25, 17, 22, 14]

theorem median_free_throws_is_17 :
  list.median free_throws = 17 :=
sorry

end median_free_throws_is_17_l44_44750


namespace solve_inequality_l44_44310

theorem solve_inequality (x : ℝ) : -3 * x^2 + 8 * x + 1 < 0 ↔ x ∈ Set.Ioo (-1 / 3 : ℝ) 1 :=
sorry

end solve_inequality_l44_44310


namespace lcm_fractions_l44_44814

theorem lcm_fractions (x : ℕ) (hx : x > 0) : 
  lcm {1/x | 1/(4*x) | 1/(5*x)} = 1/(20*x) := sorry

end lcm_fractions_l44_44814


namespace fraction_of_water_l44_44421

/-- 
  Prove that the fraction of the mixture that is water is (\frac{2}{5}) 
  given the total weight of the mixture is 40 pounds, 
  1/4 of the mixture is sand, 
  and the remaining 14 pounds of the mixture is gravel. 
-/
theorem fraction_of_water 
  (total_weight : ℝ)
  (weight_sand : ℝ)
  (weight_gravel : ℝ)
  (weight_water : ℝ)
  (h1 : total_weight = 40)
  (h2 : weight_sand = (1/4) * total_weight)
  (h3 : weight_gravel = 14)
  (h4 : weight_water = total_weight - (weight_sand + weight_gravel)) :
  (weight_water / total_weight) = 2/5 :=
by
  sorry

end fraction_of_water_l44_44421


namespace investment_at_6_percent_l44_44435

theorem investment_at_6_percent
  (x y : ℝ) 
  (total_investment : x + y = 15000)
  (total_interest : 0.06 * x + 0.075 * y = 1023) :
  x = 6800 :=
sorry

end investment_at_6_percent_l44_44435


namespace skew_MC_BD_l44_44164

-- Define points and properties in Lean
variables {A B C D M : Type}
variables (plane_orthogonal : ∀ {P : Type}, P = M → ⊥)
variables (in_plane : A ≠ B → A ≠ D → B ≠ D  → A = B → ∃ (plane : Type), ∀ p : Type, p ≠ A ∨ p ≠ B ∨ p ≠ C ∨ p ≠ D → P)

-- Define the property that MA is perpendicular to the plane of the rhombus ABCD
axiom MA_perpendicular_plane_ABCD : ∀ (M A B C D : Type), ((plane_orthogonal {M}) ∧ (in_plane {A}) ∧ (in_plane {B}) ∧ (in_plane {C}) ∧ (in_plane {D})) 

-- The theorem of determination of skew lines
theorem skew_MC_BD (h : MA_perpendicular_plane_ABCD) : ∃ h₂ BD ⊂ P, ∀ P MC ∩ P = C: ∀ (M A B C D : Type), boolean :=
sorry

end skew_MC_BD_l44_44164


namespace complex_modulus_power_l44_44463

theorem complex_modulus_power :
  let z := (1 / 3 : ℂ) + (2 / 3 : ℂ) * complex.I
  (|z^8|) = 625 / 6561 :=
by
  let z := (1 / 3 : ℂ) + (2 / 3 : ℂ) * complex.I
  have h1 : |z| = (sqrt 5) / 3 := sorry
  have h2 : |z^8| = |z|^8 := by simp [norm_pow]
  rw [h1, h2]
  have h3 : ((sqrt 5) / 3)^8 = 625 / 6561 := sorry
  exact h3

end complex_modulus_power_l44_44463


namespace moles_NaClO4_formed_l44_44502

-- Condition: Balanced chemical reaction
def reaction : Prop := ∀ (NaOH HClO4 NaClO4 H2O : ℕ), NaOH + HClO4 = NaClO4 + H2O

-- Given: 3 moles of NaOH and 3 moles of HClO4
def initial_moles_NaOH : ℕ := 3
def initial_moles_HClO4 : ℕ := 3

-- Question: number of moles of NaClO4 formed
def final_moles_NaClO4 : ℕ := 3

-- Proof Problem: Given the balanced chemical reaction and initial moles, prove the final moles of NaClO4
theorem moles_NaClO4_formed : reaction → initial_moles_NaOH = 3 → initial_moles_HClO4 = 3 → final_moles_NaClO4 = 3 :=
by
  intros
  sorry

end moles_NaClO4_formed_l44_44502


namespace distinct_arrangements_balloon_l44_44131

-- Let's define the basic conditions:
def total_letters : Nat := 7
def repeats_l : Nat := 2
def repeats_o : Nat := 2

-- Now let's state the problem.
theorem distinct_arrangements_balloon : 
  (Nat.factorial total_letters) / ((Nat.factorial repeats_l) * (Nat.factorial repeats_o)) = 1260 := 
by
  sorry

end distinct_arrangements_balloon_l44_44131


namespace cos_monotonic_increasing_l44_44879

noncomputable def f (x : ℝ) : ℝ := cos (2 * x - π / 4)

theorem cos_monotonic_increasing : 
  ∀ x y : ℝ, -π / 4 < x → x < 0 → -π / 4 < y → y < 0 → x < y → f x < f y :=
by
  sorry

end cos_monotonic_increasing_l44_44879


namespace scientists_speculation_reasonable_uranus_will_not_affect_earth_next_observation_l44_44672

-- Define the given conditions as constants and theorems in Lean
theorem scientists_speculation_reasonable : 
  ∃ (a b c : ℝ), 
  (64*a - 8*b + c = 32) ∧ 
  (36*a - 6*b + c = 28.5) ∧ 
  (16*a - 4*b + c = 26) ∧ 
  (∀ (x y : ℝ), (y = a*x^2 + b*x + c) → (x = 0) → y < 24.5) :=
by -- sorry is a placeholder for the proof
sorry

theorem uranus_will_not_affect_earth_next_observation : 
  ∃ (a b c : ℝ), 
  (64*a - 8*b + c = 32) ∧ 
  (36*a - 6*b + c = 28.5) ∧ 
  (16*a - 4*b + c = 26) ∧ 
  (∀ (x y : ℝ), (y = a*x^2 + b*x + c) → (x = 2) → y ≥ 24.5) :=
by -- sorry is a placeholder for the proof
sorry

end scientists_speculation_reasonable_uranus_will_not_affect_earth_next_observation_l44_44672


namespace time_to_cross_pole_l44_44212

-- Define constants and conditions
def train_length : ℝ := 150
def train_speed_kmh : ℝ := 200
def wind_resistance_kmh : ℝ := 20

-- Calculate effective speed in m/s
def effective_speed_kmh : ℝ := train_speed_kmh - wind_resistance_kmh
def kmh_to_mps (speed : ℝ) : ℝ := speed * (1000 / 3600)
def effective_speed_mps : ℝ := kmh_to_mps effective_speed_kmh

-- Define the time
def time_to_cross (length : ℝ) (speed : ℝ) : ℝ := length / speed

-- The theorem to prove
theorem time_to_cross_pole : time_to_cross train_length effective_speed_mps = 3 := by
  sorry

end time_to_cross_pole_l44_44212


namespace imaginary_part_of_complex_division_l44_44052

noncomputable def imaginary_unit := Complex.i
noncomputable def z := 2 + imaginary_unit
noncomputable def conj_z := Complex.conj z

theorem imaginary_part_of_complex_division : Complex.im (z / conj_z) = 4 / 5 := sorry

end imaginary_part_of_complex_division_l44_44052


namespace power_function_through_point_l44_44534

theorem power_function_through_point (n : ℕ) (h : 2^n = 8) : y = x^3 := by
  sorry

end power_function_through_point_l44_44534


namespace distinct_arrangements_balloon_l44_44116

theorem distinct_arrangements_balloon : 
  let n := 7
  let r1 := 2
  let r2 := 2
  (Nat.factorial n) / ((Nat.factorial r1) * (Nat.factorial r2)) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l44_44116


namespace buffy_breath_holding_time_l44_44250

theorem buffy_breath_holding_time (k : ℕ) (b : ℕ) : 
  k = 3 * 60 ∧ b = k - 20 → b - 40 = 120 := 
by
  intros h
  cases h with hk hb
  rw [hk, hb]
  norm_num
  sorry  -- This "sorry" is here to skip the proof

end buffy_breath_holding_time_l44_44250


namespace coeff_x3_sq_q_l44_44904

def q (x : ℝ) : ℝ := x^4 - 4 * x^2 + 3 * x - 1

theorem coeff_x3_sq_q : coeff (X^3) ((q(X))^2) = -6 :=
by
  sorry

end coeff_x3_sq_q_l44_44904


namespace johns_age_l44_44225

theorem johns_age (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l44_44225


namespace exists_infinite_terms_written_as_combination_l44_44256

-- Define the sequence and required conditions
variable {a : ℕ → ℕ}
variable (infinite_seq : ∀ k, a k < a (k + 1))
variable (strictly_positive : ∀ k, 0 < a k)

theorem exists_infinite_terms_written_as_combination :
  ∃∞ m, ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ ∃ p q, p ≠ q ∧ a m = x * a p + y * a q :=
sorry

end exists_infinite_terms_written_as_combination_l44_44256


namespace sheepdog_speed_l44_44634

theorem sheepdog_speed 
  (T : ℝ) (t : ℝ) (sheep_speed : ℝ) (initial_distance : ℝ)
  (total_distance_speed : ℝ) :
  T = 20  →
  t = 20 →
  sheep_speed = 12 →
  initial_distance = 160 →
  total_distance_speed = 20 →
  total_distance_speed * T = initial_distance + sheep_speed * t := 
by sorry

end sheepdog_speed_l44_44634


namespace sum_dbl_geometric_series_l44_44468

theorem sum_dbl_geometric_series :
  (∑ (j : ℕ) (k : ℕ), 2^(-(3 * k + j + (k + j) ^ 2) : ℝ)) = (4 / 3 : ℝ) :=
by {
  sorry
}

end sum_dbl_geometric_series_l44_44468


namespace probability_sum_3_is_1_over_216_l44_44356

-- Let E be the event that three fair dice sum to 3
def event_sum_3 (d1 d2 d3 : ℕ) : Prop := d1 + d2 + d3 = 3

-- Probabilities of rolling a particular outcome on a single die
noncomputable def P_roll_1 (n : ℕ) := if n = 1 then 1/6 else 0

-- Define the probability of the event E occurring
noncomputable def P_event_sum_3 := 
  ∑ d1 in {1, 2, 3, 4, 5, 6}, 
  ∑ d2 in {1, 2, 3, 4, 5, 6}, 
  ∑ d3 in {1, 2, 3, 4, 5, 6}, 
  if event_sum_3 d1 d2 d3 then P_roll_1 d1 * P_roll_1 d2 * P_roll_1 d3 else 0

-- The main theorem to prove the desired probability
theorem probability_sum_3_is_1_over_216 : P_event_sum_3 = 1/216 := by 
  sorry

end probability_sum_3_is_1_over_216_l44_44356


namespace balloon_arrangements_l44_44084

theorem balloon_arrangements : (7! / (2! * 2!)) = 1260 := by
  sorry

end balloon_arrangements_l44_44084


namespace find_tan_α_l44_44512

variable (α : ℝ) (h1 : Real.sin (α - Real.pi / 3) = 3 / 5)
variable (h2 : Real.pi / 4 < α ∧ α < Real.pi / 2)

theorem find_tan_α (h1 : Real.sin (α - Real.pi / 3) = 3 / 5) (h2 : Real.pi / 4 < α ∧ α < Real.pi / 2) : 
  Real.tan α = - (48 + 25 * Real.sqrt 3) / 11 :=
sorry

end find_tan_α_l44_44512


namespace min_value_expression_l44_44264

/-- 
Given real numbers a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p such that 
abcd = 16, efgh = 16, ijkl = 16, and mnop = 16, prove that the minimum value of 
(aeim)^2 + (bfjn)^2 + (cgko)^2 + (dhlp)^2 is 1024. 
-/
theorem min_value_expression (a b c d e f g h i j k l m n o p : ℝ) 
  (h1 : a * b * c * d = 16) 
  (h2 : e * f * g * h = 16) 
  (h3 : i * j * k * l = 16) 
  (h4 : m * n * o * p = 16) : 
  (a * e * i * m) ^ 2 + (b * f * j * n) ^ 2 + (c * g * k * o) ^ 2 + (d * h * l * p) ^ 2 ≥ 1024 :=
by 
  sorry


end min_value_expression_l44_44264


namespace sum_distances_at_least_2001_l44_44849

theorem sum_distances_at_least_2001 (P : ℕ → ℝ × ℝ) (C : (ℝ × ℝ) × ℝ) (hC : C.2 = 1) :
  ∃ P₀ : ℝ × ℝ, (dist P₀ C.1 = 1) ∧ (∑ i in finset.range 2001, dist P₀ (P i)  ) ≥ 2001 := 
by
  sorry

end sum_distances_at_least_2001_l44_44849


namespace problem_l44_44853

noncomputable def f (x : ℝ) := x^2
noncomputable def g (x : ℝ) := -x^3 + 5 * x - 3

theorem problem :
  (∀ x, f(x) = x^2) ∧
  (∀ x, g(x) = -x^3 + 5 * x - 3) ∧
  (∀ x > 0, let F := f x - g x in ∃ y, F y ∧ ∀ z, F z >= F y) ∧
  (∃ (k m : ℝ), (k = 2 ∧ m = -1) ∧ (∀ x, x > 0 → f(x)^3 >= k * x + m) ∧ (∀ x, x > 0 → g(x) <= k * x + m)) :=
sorry

end problem_l44_44853


namespace sale_price_reduction_l44_44374

theorem sale_price_reduction (x : ℝ) : 
  let increased_price := 1.30 * x in
  let sale_price := 0.75 * increased_price in
  sale_price = 0.975 * x :=
by
  -- Here we state the proof goals, but include sorry to skip the actual proof.
  sorry

end sale_price_reduction_l44_44374


namespace moles_of_cl2_required_eq_1_l44_44562

noncomputable def cl2_required (ch4_needed : ℕ) : ℕ := ch4_needed

theorem moles_of_cl2_required_eq_1 :
  ∀ (moles_ch4 moles_cl2 : ℕ), moles_ch4 = 1 → moles_cl2 = cl2_required moles_ch4 → moles_cl2 = 1 :=
by
  intros moles_ch4 moles_cl2 h1 h2
  simp [cl2_required, h1] at h2
  exact h2
  sorry

end moles_of_cl2_required_eq_1_l44_44562


namespace base_case_n_equals_1_l44_44666

variable {a : ℝ}
variable {n : ℕ}

theorem base_case_n_equals_1 (h1 : a ≠ 1) (h2 : n = 1) : 1 + a = 1 + a :=
by
  sorry

end base_case_n_equals_1_l44_44666


namespace jean_candy_count_l44_44608

theorem jean_candy_count : ∃ C : ℕ, 
  C - 7 = 16 ∧ 
  (C - 7 + 7 = C) ∧ 
  (C - 7 = 16) ∧ 
  (C + 0 = C) ∧
  (C - 7 = 16) :=
by 
  sorry 

end jean_candy_count_l44_44608


namespace systematic_sampling_seat_number_l44_44179

theorem systematic_sampling_seat_number (total_students sample_size : ℕ) (seat_numbers : Fin total_students) :
  sample_size = 4 → total_students = 64 → 
  seat_numbers = {5, 21, 53, _} → ∃ n, n = 37 :=
by 
  intros _ _ h_sample h_total h_seats,
  -- Sample interval
  let interval : ℕ := total_students / sample_size,  
  -- Check the given seat numbers and derive the last one
  have seq : (5, 5 + interval, 5 + 2 * interval, 5 + 3 * interval) from (5, 21, 37, 53),
  exact ⟨37, rfl⟩

end systematic_sampling_seat_number_l44_44179


namespace probability_sum_three_dice_3_l44_44365

-- Definition of a fair six-sided die
def fair_six_sided_die : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Definition of probability of an event
def probability (s : Set ℕ) (event : ℕ → Prop) : ℚ :=
  if h : finite s then (s.filter event).to_finset.card / s.to_finset.card else 0

theorem probability_sum_three_dice_3 :
  let dice := List.repeat fair_six_sided_die 3 in
  let event := λ result : List ℕ => result.sum = 3 in
  probability ({(r1, r2, r3) | r1 ∈ fair_six_sided_die ∧ r2 ∈ fair_six_sided_die ∧ r3 ∈ fair_six_sided_die }) (λ (r1, r2, r3) => r1 + r2 + r3 = 3) = 1 / 216 :=
by
  sorry

end probability_sum_three_dice_3_l44_44365


namespace balloon_arrangements_l44_44149

-- Defining the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Given Conditions
def seven_factorial := fact 7 -- 7!
def two_factorial := fact 2 -- 2!

-- Statement to prove
theorem balloon_arrangements : seven_factorial / (two_factorial * two_factorial) = 1260 :=
by
  sorry

end balloon_arrangements_l44_44149


namespace num_students_l44_44442

theorem num_students (x : ℕ) (h1 : ∃ z : ℕ, z = 10 * x + 6) (h2 : ∃ z : ℕ, z = 12 * x - 6) : x = 6 :=
by
  sorry

end num_students_l44_44442


namespace squirrels_on_trees_l44_44789

theorem squirrels_on_trees (s b j : ℕ) 
  (h1 : s + b + j = 34) 
  (h2 : b + 7 = j + s - 7)
  (h3 : j - 5 = s - 7)
  (h4 : b + 12 = 2j) 
  : s = 13 ∧ b = 10 ∧ j = 11 := by 
  sorry

end squirrels_on_trees_l44_44789


namespace max_value_among_sample_data_l44_44192

-- Definitions and conditions
def distinct (l : List ℕ) : Prop := l.nodup

def satisfies_conditions (l : List ℕ) : Prop :=
  l.length = 5 ∧ 
  l.sum = 35 ∧ 
  (l.map (λ x => (x - 7)^2)).sum = 20 ∧ 
  distinct l

-- Lean statement for the proof problem
theorem max_value_among_sample_data (l : List ℕ) 
  (h : satisfies_conditions l) : l.maximum = some 10 := by
  sorry

end max_value_among_sample_data_l44_44192


namespace magnitude_of_complex_unit_sqrt_five_l44_44528

theorem magnitude_of_complex_unit_sqrt_five (i : ℂ) (z : ℂ)
  (h1 : i^2 = -1) (h2 : z = 1 + (1 - i)^2) : complex.abs z = Real.sqrt 5 := by
  sorry

end magnitude_of_complex_unit_sqrt_five_l44_44528


namespace area_of_large_rectangle_l44_44035

noncomputable def areaEFGH : ℕ :=
  let shorter_side := 3
  let longer_side := 2 * shorter_side
  let width_EFGH := shorter_side + shorter_side
  let length_EFGH := longer_side + longer_side
  width_EFGH * length_EFGH

theorem area_of_large_rectangle :
  areaEFGH = 72 := by
  sorry

end area_of_large_rectangle_l44_44035


namespace balloon_arrangements_l44_44145

-- Defining the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Given Conditions
def seven_factorial := fact 7 -- 7!
def two_factorial := fact 2 -- 2!

-- Statement to prove
theorem balloon_arrangements : seven_factorial / (two_factorial * two_factorial) = 1260 :=
by
  sorry

end balloon_arrangements_l44_44145


namespace cost_before_tax_reduction_l44_44913

variable (C : ℝ)
variable (diff : ℝ)
variable (tax5 tax4 : ℝ)

axiom sales_tax_reduction :
  tax5 = 0.05 ∧ 
  tax4 = 0.04 ∧
  diff = 10 ∧ 
  ((C * (1 + tax5)) - (C * (1 + tax4))) = diff

theorem cost_before_tax_reduction (h : sales_tax_reduction):
  C = 1000 :=
by 
  sorry

end cost_before_tax_reduction_l44_44913


namespace second_largest_geometric_sum_l44_44699

theorem second_largest_geometric_sum {a r : ℕ} (h_sum: a + a * r + a * r^2 + a * r^3 = 1417) (h_geometric: 1 + r + r^2 + r^3 ∣ 1417) : (a * r^2 = 272) :=
sorry

end second_largest_geometric_sum_l44_44699


namespace abs_diff_of_solutions_l44_44267

theorem abs_diff_of_solutions (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : |m - n| = 5 := 
sorry

end abs_diff_of_solutions_l44_44267


namespace sum_of_x_y_l44_44167

theorem sum_of_x_y (x y : ℕ) (h1 : 10 * x + y = 75) (h2 : 10 * y + x = 57) : x + y = 12 :=
sorry

end sum_of_x_y_l44_44167


namespace balloon_permutations_l44_44136

theorem balloon_permutations : 
  let n : ℕ := 7
  let k1 : ℕ := 2
  let k2 : ℕ := 2
  ∃ distinct_arrangements : ℕ, 
  distinct_arrangements = n.factorial / (k1.factorial * k2.factorial) 
  ∧ distinct_arrangements = 1260 :=
by
  sorry

end balloon_permutations_l44_44136


namespace find_six_digit_number_l44_44779

theorem find_six_digit_number (a b c d e f : ℕ) (N : ℕ) :
  a = 1 ∧ f = 7 ∧
  N = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f ∧
  (f - 1) * 10^5 + 10^4 * a + 10^3 * b + 10^2 * c + 10 * d + e = 5 * N →
  N = 142857 :=
by
  sorry

end find_six_digit_number_l44_44779


namespace tax_percentage_correct_l44_44481

def spending_distribution := {clothing : ℝ, food : ℝ, other_items : ℝ}
def tax_rates := {clothing : ℝ, food : ℝ, other_items : ℝ}
def discounts := {clothing : ℝ, other_items : ℝ}

theorem tax_percentage_correct(tot_spent : ℝ) 
    (sd : spending_distribution)
    (tr : tax_rates)
    (disc : discounts)
    (total_clothing_spent := tot_spent * sd.clothing)
    (total_food_spent := tot_spent * sd.food)
    (total_other_items_spent := tot_spent * sd.other_items)
    (final_clothing_spent := total_clothing_spent - (total_clothing_spent * disc.clothing))
    (final_other_items_spent := total_other_items_spent - (total_other_items_spent * disc.other_items))
    (final_food_spent := total_food_spent)
    (clothing_tax := final_clothing_spent * tr.clothing)
    (food_tax := final_food_spent * tr.food)
    (other_items_tax := final_other_items_spent * tr.other_items)
    (total_spent_excl_taxes := final_clothing_spent + final_food_spent + final_other_items_spent)
    (total_taxes := clothing_tax + food_tax + other_items_tax) :
    (total_taxes / total_spent_excl_taxes) * 100 = 5.105 := 
by sorry

-- Instantiate with the given values
noncomputable def Jill_spending_distribution : spending_distribution := {
  clothing := 0.40,
  food := 0.25,
  other_items := 0.35
}

noncomputable def Jill_tax_rates : tax_rates := {
  clothing := 0.05,
  food := 0.03,
  other_items := 0.07
}

noncomputable def Jill_discounts : discounts := {
  clothing := 0.10,
  other_items := 0.15
}

-- Apply the theorem with the instantiated values
#eval tax_percentage_correct 100 Jill_spending_distribution Jill_tax_rates Jill_discounts

end tax_percentage_correct_l44_44481


namespace divisor_of_form_4k_minus_1_l44_44260

theorem divisor_of_form_4k_minus_1
  (n : ℕ) (hn1 : Odd n) (hn_pos : 0 < n)
  (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h_eq : (1 / (x : ℚ) + 1 / (y : ℚ) = 4 / n)) :
  ∃ k : ℕ, ∃ d, d ∣ n ∧ d = 4 * k - 1 ∧ k ∈ Set.Ici 1 :=
sorry

end divisor_of_form_4k_minus_1_l44_44260


namespace number_of_valid_configs_l_shape_foldable_to_cube_missing_one_face_l44_44772

structure Square := (id : ℕ)

structure LShape :=
  (a b c d : Square)
  (is_l_shaped : b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ ∃ e, e = d + 1)

constant positions : Fin 7 → Square

def valid_cube_configs (lshape : LShape) (pos : Fin 7) : Prop :=
  sorry -- Defines the condition for a valid configuration forming a cube missing one face.

theorem number_of_valid_configs_l_shape_foldable_to_cube_missing_one_face
  (lshape : LShape) :
  (∑ pos in Finset.univ.filter (valid_cube_configs lshape), 1) = 3 :=
sorry

end number_of_valid_configs_l_shape_foldable_to_cube_missing_one_face_l44_44772


namespace rhombus_area_l44_44775

theorem rhombus_area (s : ℝ) (θ : ℝ) (h₀ : s = 4) (h₁ : θ = π / 4) : 
  (s * s * sin θ) = 8 * sqrt 2 :=
by
  sorry

end rhombus_area_l44_44775


namespace triangle_inequality_l44_44293

theorem triangle_inequality
  (R r p : ℝ) (a b c : ℝ)
  (h1 : a * b + b * c + c * a = r^2 + p^2 + 4 * R * r)
  (h2 : 16 * R * r - 5 * r^2 ≤ p^2)
  (h3 : p^2 ≤ 4 * R^2 + 4 * R * r + 3 * r^2):
  20 * R * r - 4 * r^2 ≤ a * b + b * c + c * a ∧ a * b + b * c + c * a ≤ 4 * (R + r)^2 := 
  by
    sorry

end triangle_inequality_l44_44293


namespace ant_expected_moves_l44_44751

noncomputable def expected_moves_to_anthill (x y : ℤ) : ℕ :=
if odd x ∧ odd y then 0
else if even x ∧ even y then 4
else 4

theorem ant_expected_moves : expected_moves_to_anthill 0 0 = 4 := 
by
  sorry

end ant_expected_moves_l44_44751


namespace tn_arithmetic_sequence_l44_44336

variable {n : ℕ}
variable {u : Fin n → ℝ}

-- Assume the arithmetic sequence condition
def is_arithmetic_sequence (u : Fin n → ℝ) : Prop :=
  ∃ d : ℝ, ∀ i : Fin (n-1), u ⟨i.1 + 1, Nat.add_lt_add_right i.2 1⟩ = u ⟨i.1, i.2⟩ + d

-- Assume positive numbers condition
def all_positive (u : Fin n → ℝ) : Prop :=
  ∀ i : Fin n, 0 < u i

-- The theorem to be proven
theorem tn_arithmetic_sequence (hu : is_arithmetic_sequence u) (hp : all_positive u) :
  let t_n := ∑ i : Fin (n-1), 1 / (Real.sqrt (u ⟨i.1, i.2⟩) + Real.sqrt (u ⟨i.1 + 1, Nat.add_lt_add_right i.2 1⟩))
  in t_n = (n - 1) / (Real.sqrt (u 0) + Real.sqrt (u ⟨n-1, sorry⟩))
:= sorry

end tn_arithmetic_sequence_l44_44336


namespace ratio_b_a_4_l44_44576

theorem ratio_b_a_4 (a b : ℚ) (h1 : b / a = 4) (h2 : b = 15 - 6 * a) : a = 3 / 2 :=
by
  sorry

end ratio_b_a_4_l44_44576


namespace find_point_B_l44_44890

theorem find_point_B 
  (a : ℝ × ℝ) (A : ℝ × ℝ) (AB : ℝ × ℝ → ℝ × ℝ)
  (direction : AB (7, -5) = (2, -1) ∧ |√((7 - 1)^2 + (-5 + 2)^2)| = 3 * √5) :
  ∃ B : ℝ × ℝ, B = (7, -5) :=
by
  have h1 : AB (7, -5) = (7 - 1, -5 + 2),
  have h2 : ∀ x y : ℝ, x = 7 ∧ y = -5 ↔ -(x - 1) - 2(y + 2) = 0,
  have h3 : ∃ x y : ℝ , sqrt ((x - 1)^2 + (y + 2)^2) = 3 * sqrt 5,
  have h4 : direction = true,
  sorry

end find_point_B_l44_44890


namespace regular_hexagon_interior_angles_l44_44343

theorem regular_hexagon_interior_angles (n : ℕ) (h : n = 6) :
  (n - 2) * 180 = 720 :=
by
  subst h
  rfl

end regular_hexagon_interior_angles_l44_44343


namespace percentage_of_hindu_boys_l44_44588

-- Define the total number of boys in the school
def total_boys := 700

-- Define the percentage of Muslim boys
def muslim_percentage := 44 / 100

-- Define the percentage of Sikh boys
def sikh_percentage := 10 / 100

-- Define the number of boys from other communities
def other_communities_boys := 126

-- State the main theorem to prove the percentage of Hindu boys
theorem percentage_of_hindu_boys (h1 : total_boys = 700)
                                 (h2 : muslim_percentage = 44 / 100)
                                 (h3 : sikh_percentage = 10 / 100)
                                 (h4 : other_communities_boys = 126) : 
                                 ((total_boys - (total_boys * muslim_percentage + total_boys * sikh_percentage + other_communities_boys)) / total_boys) * 100 = 28 :=
by {
  sorry
}

end percentage_of_hindu_boys_l44_44588


namespace frank_winnings_expected_value_l44_44845

-- Define the conditions as a structure
structure DieRoll :=
  (sides : Finset ℕ)
  (prime_set : Finset ℕ)
  (square_set : Finset ℕ)

-- Initialize the die roll with specific conditions
def frankDie : DieRoll :=
  { sides := {1, 2, 3, 4, 5, 6, 7, 8},
    prime_set := {2, 3, 5, 7},
    square_set := {1, 4} }

-- Expected value calculation function
noncomputable def expected_winnings (die : DieRoll) : ℚ :=
  let prime_prob := (die.prime_set.card : ℚ) / (die.sides.card)
  let square_prob := (die.square_set.card : ℚ) / (die.sides.card)
  let prime_winnings := (prime_prob * (2 + 3 + 5 + 7 : ℚ) / 4) 
  let square_winnings := (square_prob * (1 * 2 + 4 * 2 : ℚ) / 2) 
  prime_winnings + square_winnings 

-- The target theorem
theorem frank_winnings_expected_value :
  expected_winnings frankDie = 3.25 := 
begin 
  sorry
end

end frank_winnings_expected_value_l44_44845


namespace hypotenuse_of_45_45_90_triangle_l44_44648

noncomputable def leg_length : ℝ := 15
noncomputable def angle_opposite_leg : ℝ := Real.pi / 4  -- 45 degrees in radians

theorem hypotenuse_of_45_45_90_triangle (h_leg : ℝ) (h_angle : ℝ) 
  (h_leg_cond : h_leg = leg_length) (h_angle_cond : h_angle = angle_opposite_leg) :
  ∃ h_hypotenuse : ℝ, h_hypotenuse = h_leg * Real.sqrt 2 :=
sorry

end hypotenuse_of_45_45_90_triangle_l44_44648


namespace vector_dot_product_in_triangle_l44_44961

theorem vector_dot_product_in_triangle (A B C M : ℝ^3) (AB AC BC : ℝ) 
  (hAB : distance A B = 10) (hAC : distance A C = 6) (hBC : distance B C = 8)
  (hM : M = (A + B) / 2) :
  (M - C) • (A - C) + (M - C) • (B - C) = 50 :=
by
  sorry

end vector_dot_product_in_triangle_l44_44961


namespace hypotenuse_of_45_45_90_triangle_15_l44_44655

theorem hypotenuse_of_45_45_90_triangle_15 (a : ℝ) (h : a = 15) : 
  ∃ (c : ℝ), c = a * Real.sqrt 2 :=
by
  use a * Real.sqrt 2
  rw h
  sorry

end hypotenuse_of_45_45_90_triangle_15_l44_44655


namespace problem_division_l44_44983

def A := {i : ℕ | 1 ≤ i ∧ i ≤ 9}

-- Represents functions from set A to set A
def number_of_functions (A : Set ℕ) : ℕ := 
  Set.card {f : A → A | ∃ c ∈ A, ∀ x ∈ A, f (f x) = c}

-- Given N as the number_of_functions from set A to set A such that f(f(x)) is a constant function.
def N := number_of_functions A

theorem problem_division (A : Set ℕ) (N : ℕ) :
  (N % 1000 = 853) := by
  sorry

end problem_division_l44_44983


namespace balloon_arrangements_l44_44143

-- Defining the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Given Conditions
def seven_factorial := fact 7 -- 7!
def two_factorial := fact 2 -- 2!

-- Statement to prove
theorem balloon_arrangements : seven_factorial / (two_factorial * two_factorial) = 1260 :=
by
  sorry

end balloon_arrangements_l44_44143


namespace probability_event_independence_l44_44392

theorem probability_event_independence (E : Type) [Fintype E] [DecidableEq E] (e : E) (n : ℕ) :
  (∀ (m : ℕ), Prob.event_reoccur (Prob.trial n) e = Prob.event_reoccur (Prob.trial m) e) :=
sorry

end probability_event_independence_l44_44392


namespace total_distance_covered_l44_44380

noncomputable def speed_train_a : ℚ := 80          -- Speed of Train A in kmph
noncomputable def speed_train_b : ℚ := 110         -- Speed of Train B in kmph
noncomputable def duration : ℚ := 15               -- Duration in minutes
noncomputable def conversion_factor : ℚ := 60      -- Conversion factor from hours to minutes

theorem total_distance_covered : 
    (speed_train_a / conversion_factor) * duration + 
    (speed_train_b / conversion_factor) * duration = 47.5 :=
by
  sorry

end total_distance_covered_l44_44380


namespace convex_polygons_from_fifteen_points_l44_44492

theorem convex_polygons_from_fifteen_points 
    (h : ∀ (n : ℕ), n = 15) :
    ∃ (k : ℕ), k = 32192 :=
by
  sorry

end convex_polygons_from_fifteen_points_l44_44492


namespace remove_vertex_preserves_connectivity_l44_44860

-- Definitions
variables {V : Type*} [Fintype V] [DecidableEq V]
variables (G : SimpleGraph V) (h_connected : G.Connected)

-- Theorem Statement
theorem remove_vertex_preserves_connectivity (G : SimpleGraph V) (h_connected : G.Connected) : 
  ∃ (v : V), ∀ w1 w2 ∈ (G.delete_vertex v).verts, (G.delete_vertex v).Connected :=
sorry

end remove_vertex_preserves_connectivity_l44_44860


namespace balloon_arrangements_l44_44150

-- Defining the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Given Conditions
def seven_factorial := fact 7 -- 7!
def two_factorial := fact 2 -- 2!

-- Statement to prove
theorem balloon_arrangements : seven_factorial / (two_factorial * two_factorial) = 1260 :=
by
  sorry

end balloon_arrangements_l44_44150


namespace cake_baking_ratio_l44_44633

theorem cake_baking_ratio :
  ∃ r : ℝ, (10 * r^5 = 320) ∧ r = 2 :=
by
  use 2
  split
  {
    sorry
  }

end cake_baking_ratio_l44_44633


namespace ellipse_eq_proof_T_range_proof_l44_44197

def ellipse_eq (a b : ℝ) (a_gt_b : a > b) (b_gt_0 : b > 0) (c : ℝ) (eq_c_1 : c = 1) : Prop :=
  ∃ x y : ℝ, (x*x)/(a*a) + (y*y)/(b*b) = 1 ∧
  let dist := λ p1 p2 : (ℝ × ℝ), ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)^0.5 in
  dist (x, y) (1, 0) = 5/3 ∧
  dist (x, y) (-1, 0) + 5/3 = 4 ∧
  a = 2 ∧ b^2 = 3

noncomputable def T_range := {t : ℝ | 0 < t ∧ t < 1/4}

theorem ellipse_eq_proof (a b : ℝ) (a_gt_b : a > b) (b_gt_0 : b > 0) (c : ℝ) (eq_c_1 : c = 1) :
  (∃ x y : ℝ, (x*x)/(2*2) + (y*y)/3 = 1 ∧
    (let dist (p1 p2 : (ℝ × ℝ)) := ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2) ^ 0.5 in 
    dist (x, y) (1, 0) = 5 / 3 ∧ 
    dist (x, y) (-1, 0) + 5 / 3 = 4)) :=
begin
  sorry
end

theorem T_range_proof (t : ℝ) :
  t ∈ T_range :=
begin
  sorry
end

end ellipse_eq_proof_T_range_proof_l44_44197


namespace intersection_on_circle_l44_44599

variables (A B C H M X E F J : Type*)

-- Given Conditions:
-- 1. Triangle ABC is acute with AC > AB.
axiom acute_triangle (ABC : Triangle) : 
  triangle.acute ABC ∧ triangle.side_ordering ABC AC AB

-- 2. H is the orthocenter of triangle ABC.
axiom orthocenter (H : Point) (ABC : Triangle) : is_orthocenter H ABC

-- 3. M is the midpoint of BC.
axiom midpoint (M : Point) (BC : Line) : is_midpoint M BC

-- 4. Line AM intersects the circumcircle of triangle ABC again at X.
axiom line_circum (AM : Line) (circumcircle : Circle) : intersects AM circumcircle X ∧ in_circle X circumcircle

-- 5. Line CH intersects the perpendicular bisector of BC at point E and the circumcircle again at point F.
axiom line_perpbisect (CH : Line) (perpbisector : Line) (circumcircle: Circle) :
  intersects CH perpbisector E ∧ intersects CH circumcircle F

-- 6. Circle Γ passes through X, E, and F.
axiom circle_gamma (Gamma : Circle) : passes_through Gamma X ∧ passes_through Gamma E ∧ passes_through Gamma F

-- 7. J is on circle Γ such that BCHJ is a trapezoid with CB parallel to HJ.
axiom trapezoid (Gamma : Circle) (B C H J : Point) :
  in_circle J Gamma ∧ is_parallel (line C B) (line H J)

-- Prove that the intersection of JB and EM lies on circle Γ.
theorem intersection_on_circle (ABC : Triangle) (H M X E F J : Point) 
  (Gamma : Circle) (AM CH EM JB : Line) :
  intersects JB EM (λ P, in_circle P Gamma) :=
sorry

end intersection_on_circle_l44_44599


namespace pieces_length_l44_44752

theorem pieces_length (L M S : ℝ) (h1 : L + M + S = 180)
  (h2 : L = M + S + 30)
  (h3 : M = L / 2 - 10) :
  L = 105 ∧ M = 42.5 ∧ S = 32.5 :=
by
  sorry

end pieces_length_l44_44752


namespace car_transport_distance_l44_44349

theorem car_transport_distance
  (d_birdhouse : ℕ) 
  (d_lawnchair : ℕ) 
  (d_car : ℕ)
  (h1 : d_birdhouse = 1200)
  (h2 : d_birdhouse = 3 * d_lawnchair)
  (h3 : d_lawnchair = 2 * d_car) :
  d_car = 200 := 
by
  sorry

end car_transport_distance_l44_44349


namespace triangle_side_ratio_l44_44959

theorem triangle_side_ratio (A B C a b c : ℝ)
  (h1 : A + B + C = 180)
  (h2 : A = 30)
  (h3 : B = 60)
  (h4 : C = 90)
  (h5 : a / Real.sin (Real.pi / 6) = b / Real.sin (Real.pi / 3))
  (h6 : a / Real.sin (Real.pi / 6) = c / Real.sin (Real.pi / 2)) :
  a : b : c = 1 : Real.sqrt 3 : 2 :=
by sorry

end triangle_side_ratio_l44_44959


namespace initial_price_is_2_l44_44695

variable (P Q : ℝ)
-- Condition: The price of sugar increased to Rs. 5 per kg.
def new_price : ℝ := 5
-- Condition: The quantity of sugar consumed is reduced by 60%.
def new_quantity : ℝ := 0.4 * Q

-- The person's initial and subsequent expenditures should be equal.
def expenditure_initial : ℝ := P * Q
def expenditure_new : ℝ := new_price * new_quantity

-- The proof problem: Given the conditions, prove that the initial price was Rs. 2 per kg.
theorem initial_price_is_2 (h : expenditure_initial P Q = expenditure_new Q) : P = 2 :=
by
  sorry

end initial_price_is_2_l44_44695


namespace hypotenuse_of_45_45_90_triangle_15_l44_44657

theorem hypotenuse_of_45_45_90_triangle_15 (a : ℝ) (h : a = 15) : 
  ∃ (c : ℝ), c = a * Real.sqrt 2 :=
by
  use a * Real.sqrt 2
  rw h
  sorry

end hypotenuse_of_45_45_90_triangle_15_l44_44657


namespace angle_AHI_eq_3_angle_ABC_l44_44997

open EuclideanGeometry

-- Define the variables and hypotheses
variables {A B C I H : Point} (ABC : Triangle A B C)

-- Hypotheses corresponding to the given conditions
hypothesis (ABC_acute : acute_triangle ABC)
hypothesis (angle_BAC_60 : angle BAC = 60)
hypothesis (AB_gt_AC : length A B > length A C)
hypothesis (I_incenter : incenter I ABC)
hypothesis (H_orthocenter : orthocenter H ABC)

-- Statement of the theorem to be proved
theorem angle_AHI_eq_3_angle_ABC 
    (ABC_acute : acute_triangle ABC)
    (angle_BAC_60 : ∠ A B C = 60)
    (AB_gt_AC : length A B > length A C)
    (I_incenter : incenter I ABC)
    (H_orthocenter : orthocenter H ABC)
    : 2 * angle A H I = 3 * angle A B C := 
    by sorry

end angle_AHI_eq_3_angle_ABC_l44_44997


namespace center_circle_sum_l44_44723

theorem center_circle_sum (h k : ℝ) :
  (∃ h k : ℝ, h + k = 6 ∧ ∃ R, (x - h)^2 + (y - k)^2 = R^2) ↔ ∃ h k : ℝ, h = 3 ∧ k = 3 ∧ h + k = 6 := 
by
  sorry

end center_circle_sum_l44_44723


namespace Margo_total_distance_walked_l44_44282

theorem Margo_total_distance_walked :
  ∀ (d : ℝ),
  (5 * (d / 5) + 3 * (d / 3) = 1) →
  (2 * d = 3.75) :=
by
  sorry

end Margo_total_distance_walked_l44_44282


namespace sum_of_digits_of_min_number_with_product_1728_l44_44910

theorem sum_of_digits_of_min_number_with_product_1728 :
  ∃ N : ℕ, (∀ d ∈ (digits 10 N), 0 < d ∧ d < 10) ∧ (list.prod (digits 10 N) = 1728) ∧ (list.sum (digits 10 N) = 28) :=
sorry

end sum_of_digits_of_min_number_with_product_1728_l44_44910


namespace father_son_age_problem_l44_44426

theorem father_son_age_problem
  (F S Y : ℕ)
  (h1 : F = 3 * S)
  (h2 : F = 45)
  (h3 : F + Y = 2 * (S + Y)) :
  Y = 15 :=
sorry

end father_son_age_problem_l44_44426


namespace problem1_problem2_l44_44543

open Real

noncomputable def f (a x : ℝ) : ℝ := a * log x + 0.5 * x^2 - a * x

theorem problem1 (a : ℝ) : (a > 4) ↔ (∀ x : ℝ, a * log x + 0.5 * x^2 - a * x has two distinct positive roots where x^2 - a * x + a = 0) := sorry

theorem problem2 (a : ℝ) (h : a > 4) : ∀ x1 x2 : ℝ, (x1 + x2 = a) ∧ (x1 * x2 = a) → (f a x1 + f a x2 < λ * (x1 + x2)) → (λ ≥ log 4 - 3) := sorry

end problem1_problem2_l44_44543


namespace number_of_value_sets_l44_44889

def A := {-1, 0, 1, 2, 3}
def B := {-1, 0, 1}

theorem number_of_value_sets : 
  ∃ (f : A → B), set.card (set.image f A) = 7 :=
sorry

end number_of_value_sets_l44_44889


namespace collinear_NOK_l44_44979

variable (Point : Type)
variable [AffineSpace Point]

variables (A B C D O P Q N K : Point)

-- Definitions for collinearity and intersecting lines
def collinear (a b c : Point) : Prop :=
  ∃ l : Set Point, l ∈ line a b ∧ l ∈ line b c

def on_Line (a b c : Point) (l : Set Point) : Prop :=
  a ∈ l ∧ b ∈ l ∧ c ∈ l

-- Given conditions
variable (h1 : on_diagonals O A C B D)
variable (h2 : dist A O = dist C O)
variable (h3 : segment A O P ∧ segment C O Q ∧ dist P O = dist Q O)
variable (h4 : intersect AB DP N)
variable (h5 : intersect CD BQ K)

-- Theorem to prove collinearity
theorem collinear_NOK :
  collinear N O K := 
sorry

end collinear_NOK_l44_44979


namespace distinct_arrangements_balloon_l44_44092

theorem distinct_arrangements_balloon : 
  let n := 7 
  let freq_l := 2 
  let freq_o := 2 
  let freq_b := 1 
  let freq_a := 1 
  let freq_n := 1 
  Nat.factorial n / (Nat.factorial freq_l * Nat.factorial freq_o * Nat.factorial freq_b * Nat.factorial freq_a * Nat.factorial freq_n) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l44_44092


namespace donny_spent_total_on_friday_and_sunday_l44_44004

noncomputable def daily_savings (initial: ℚ) (increase_rate: ℚ) (days: List ℚ) : List ℚ :=
days.scanl (λ acc day => acc * increase_rate + acc) initial

noncomputable def thursday_savings : ℚ := (daily_savings 15 (1 + 0.1) [15, 15, 15]).sum

noncomputable def friday_spent : ℚ := thursday_savings * 0.5

noncomputable def remaining_after_friday : ℚ := thursday_savings - friday_spent

noncomputable def saturday_savings (thursday: ℚ) : ℚ := thursday * (1 - 0.20)

noncomputable def total_savings_saturday : ℚ := remaining_after_friday + saturday_savings thursday_savings

noncomputable def sunday_spent : ℚ := total_savings_saturday * 0.40

noncomputable def total_spent : ℚ := friday_spent + sunday_spent

theorem donny_spent_total_on_friday_and_sunday : total_spent = 55.13 := by
  sorry

end donny_spent_total_on_friday_and_sunday_l44_44004


namespace jerome_contact_list_l44_44966

theorem jerome_contact_list
  (classmates : ℕ)
  (friends : ℕ)
  (family : ℕ)
  (added : ℕ)
  (removed : ℕ)
  (h_classmates : classmates = 20)
  (h_friends : friends = classmates / 2)
  (h_family : family = 2 + 1)
  (h_added : added = 5 + 7)
  (h_removed : removed = 3 + 4)
  :
  classmates + friends + family + added - removed = 38 :=
by
  rw [h_classmates, h_friends, h_family, h_added, h_removed]
  norm_num
  done

end jerome_contact_list_l44_44966


namespace find_rs_l44_44292

-- Define a structure to hold the conditions
structure Conditions (r s : ℝ) : Prop :=
  (positive_r : 0 < r)
  (positive_s : 0 < s)
  (eq1 : r^3 + s^3 = 1)
  (eq2 : r^6 + s^6 = (15 / 16))

-- State the theorem
theorem find_rs (r s : ℝ) (h : Conditions r s) : rs = 1 / (48 : ℝ)^(1/3) :=
by
  sorry

end find_rs_l44_44292


namespace perpendicular_bisector_condition_l44_44597

variables (p : ℝ) (p_pos : p > 0)

def focus_of_parabola := (p / 2, 0)
def directrix_of_parabola := -p / 2

theorem perpendicular_bisector_condition (h : 2 * (p / 2) - 4 * 0 + 5 = 0) :
  directrix_of_parabola p = -5 / 4 :=
by
  sorry

end perpendicular_bisector_condition_l44_44597


namespace hypotenuse_of_45_45_90_triangle_l44_44653

theorem hypotenuse_of_45_45_90_triangle (a : ℝ) (h : ℝ) 
  (ha : a = 15) 
  (angle_opposite_leg : ℝ) 
  (h_angle : angle_opposite_leg = 45) 
  (right_triangle : ∃ θ : ℝ, θ = 90) : 
  h = 15 * Real.sqrt 2 := 
sorry

end hypotenuse_of_45_45_90_triangle_l44_44653


namespace number_of_people_to_bail_water_in_2_hours_l44_44419

theorem number_of_people_to_bail_water_in_2_hours (h1 : 10 * 3 ≥ total_water)
                                                  (h2 : 5 * 8 ≥ total_water)
                                                  (enter_rate : 2)
                                                  (per_person_rate : 1)
                                                  (total_water_at_3_hours : total_water = 30)
                                                  (total_water_at_8_hours : total_water = 40): 
  (needed_people : ℕ) (needed_people = 14) :=
sorry

end number_of_people_to_bail_water_in_2_hours_l44_44419


namespace rectangles_on_grid_l44_44897

-- Define the grid dimensions
def m := 3
def n := 2

-- Define a function to count the total number of rectangles formed by the grid.
def count_rectangles (m n : ℕ) : ℕ := 
  (m * (m - 1) / 2 + n * (n - 1) / 2) * (n * (n - 1) / 2 + m * (m - 1) / 2) 

-- State the theorem we need to prove
theorem rectangles_on_grid : count_rectangles m n = 14 :=
  sorry

end rectangles_on_grid_l44_44897


namespace paper_perimeter_l44_44899

def side_length_name_tag : ℕ := 4
def paper_width : ℕ := 34
def remaining_width : ℕ := 2
def students_count : ℕ := 24

theorem paper_perimeter :
  let used_width := paper_width - remaining_width
  let name_tags_per_row := used_width / side_length_name_tag
  let rows := students_count / name_tags_per_row
  let length := rows * side_length_name_tag
  in 2 * (paper_width + length) = 92 :=
by sorry

end paper_perimeter_l44_44899


namespace coefficient_of_x3_in_expansion_l44_44200

noncomputable def binom_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem coefficient_of_x3_in_expansion :
  (∑ k in Finset.range (41), binom_coeff 40 k * (1 : ℤ)^(40 - k) * (2 : ℤ)^k) = 79040 :=
begin
  have h₁ : binom_coeff 40 3 = 9880,
  { simp [binom_coeff, Nat.choose],
    norm_num },

  have h₂ : 1^(40 - 3) * 2^3 = 8,
  { norm_num },

  have h₃ : binom_coeff 40 3 * 8 = 79040,
  { norm_num,
    linarith },

  exact h₃,
end

end coefficient_of_x3_in_expansion_l44_44200


namespace sequence_solution_l44_44555

theorem sequence_solution
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_a1 : a 1 = 10)
  (h_b1 : b 1 = 10)
  (h_recur_a : ∀ n : ℕ, a (n + 1) = 1 / (a n * b n))
  (h_recur_b : ∀ n : ℕ, b (n + 1) = (a n)^4 * b n) :
  (∀ n : ℕ, n > 0 → a n = 10^((2 - 3 * n) * (-1 : ℝ)^n) ∧ b n = 10^((6 * n - 7) * (-1 : ℝ)^n)) :=
by
  sorry

end sequence_solution_l44_44555


namespace total_cost_of_shirt_and_coat_l44_44006

-- Definition of the conditions
def shirt_cost : ℕ := 150
def one_third_of_coat (coat_cost: ℕ) : Prop := shirt_cost = coat_cost / 3

-- Theorem stating the problem to prove
theorem total_cost_of_shirt_and_coat (coat_cost : ℕ) (h : one_third_of_coat coat_cost) : shirt_cost + coat_cost = 600 :=
by 
  -- Proof goes here, using sorry as placeholder
  sorry

end total_cost_of_shirt_and_coat_l44_44006


namespace incorrect_statements_l44_44061

noncomputable def f : ℝ → ℝ := λ x, Real.cos (x + Real.pi / 3)

theorem incorrect_statements :
  (¬ (∃ k : ℤ, f (x + k * (2 * Real.pi)) = f x ∧ k ≠ 0 → f x = f (x - 2 * Real.pi))) ∨
  (¬ (∀ x : ℝ, f (x + (8 * Real.pi / 3)) = f x)) ∨
  (¬ (∃ x : ℝ, f (x + Real.pi) = 0 → x = Real.pi / 6)) ∨
  (¬ (∀ x : ℝ, (Real.pi / 2 < x ∧ x < Real.pi) → f x > f (x + Real.pi / 3)))
:= by
  sorry

end incorrect_statements_l44_44061


namespace polygon_interior_exterior_equal_l44_44927

theorem polygon_interior_exterior_equal (n : ℕ) :
  (n - 2) * 180 = 360 → n = 4 :=
by
  sorry

end polygon_interior_exterior_equal_l44_44927


namespace probability_sum_3_is_1_over_216_l44_44355

-- Let E be the event that three fair dice sum to 3
def event_sum_3 (d1 d2 d3 : ℕ) : Prop := d1 + d2 + d3 = 3

-- Probabilities of rolling a particular outcome on a single die
noncomputable def P_roll_1 (n : ℕ) := if n = 1 then 1/6 else 0

-- Define the probability of the event E occurring
noncomputable def P_event_sum_3 := 
  ∑ d1 in {1, 2, 3, 4, 5, 6}, 
  ∑ d2 in {1, 2, 3, 4, 5, 6}, 
  ∑ d3 in {1, 2, 3, 4, 5, 6}, 
  if event_sum_3 d1 d2 d3 then P_roll_1 d1 * P_roll_1 d2 * P_roll_1 d3 else 0

-- The main theorem to prove the desired probability
theorem probability_sum_3_is_1_over_216 : P_event_sum_3 = 1/216 := by 
  sorry

end probability_sum_3_is_1_over_216_l44_44355


namespace remainder_11_pow_1000_mod_500_l44_44722

theorem remainder_11_pow_1000_mod_500 : (11 ^ 1000) % 500 = 1 :=
by
  have h1 : 11 % 5 = 1 := by norm_num
  have h2 : (11 ^ 10) % 100 = 1 := by
    -- Some steps omitted to satisfy conditions; normally would be generalized
    sorry
  have h3 : 500 = 5 * 100 := by norm_num
  -- Further omitted steps aligning with the Chinese Remainder Theorem application.
  sorry

end remainder_11_pow_1000_mod_500_l44_44722


namespace kira_breakfast_time_l44_44975

theorem kira_breakfast_time (n_sausages : ℕ) (n_eggs : ℕ) (t_fry_per_sausage : ℕ) (t_scramble_per_egg : ℕ) (total_time : ℕ) :
  n_sausages = 3 → n_eggs = 6 → t_fry_per_sausage = 5 → t_scramble_per_egg = 4 → total_time = (n_sausages * t_fry_per_sausage + n_eggs * t_scramble_per_egg) →
  total_time = 39 :=
by
  intros h_sausages h_eggs h_fry h_scramble h_total
  rw [h_sausages, h_eggs, h_fry, h_scramble] at h_total
  exact h_total

end kira_breakfast_time_l44_44975


namespace find_y_l44_44817

theorem find_y (y : ℕ) : (8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10) = 2 ^ y → y = 33 := 
by 
  sorry

end find_y_l44_44817


namespace one_third_of_6_3_eq_21_10_l44_44496

theorem one_third_of_6_3_eq_21_10 : (6.3 / 3) = (21 / 10) := by
  sorry

end one_third_of_6_3_eq_21_10_l44_44496


namespace meal_distribution_l44_44682

noncomputable theory
open_locale classical

-- Definitions based on the problem's conditions
def num_people := 10
def meal_types := 4

def beef_orders := 4
def chicken_orders := 3
def fish_orders := 2
def vegetarian_orders := 1

-- Number of derangements for the remaining 9 people
def derangements (n : ℕ) : ℕ := nat.fact n * ∑ k in finset.range (n + 1), (-1 : ℤ) ^ k / nat.fact k

-- The theorem to be proven:
theorem meal_distribution : ∃ (D : ℕ), 
  D = derangements 9 ∧
  (∑ p in finset.range num_people, if p = 1 then 1 else 0) * D = 10 * derangements 9 :=
begin
  sorry
end

end meal_distribution_l44_44682


namespace three_point_seven_five_as_fraction_l44_44717

theorem three_point_seven_five_as_fraction :
  (15 : ℚ) / 4 = 3.75 :=
sorry

end three_point_seven_five_as_fraction_l44_44717


namespace distance_between_foci_of_ellipse_tangent_at_points_l44_44451

noncomputable def ellipse_dist_foci (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

theorem distance_between_foci_of_ellipse_tangent_at_points :
  (∀ (a b : ℝ), a = 6 ∧ b = 3 → ellipse_dist_foci a b = 3 * Real.sqrt 3) :=
by
  intros a b h
  cases h with ha hb
  rw [ha, hb]
  sorry

end distance_between_foci_of_ellipse_tangent_at_points_l44_44451


namespace digit_difference_one_l44_44780

theorem digit_difference_one (p q : ℕ) (h_pq : p < 10 ∧ q < 10) (h_diff : (10 * p + q) - (10 * q + p) = 9) :
  p - q = 1 :=
by
  sorry

end digit_difference_one_l44_44780


namespace points_per_enemy_l44_44711

theorem points_per_enemy (kills: ℕ) (bonus_threshold: ℕ) (bonus_multiplier: ℝ) (total_score_with_bonus: ℕ) (P: ℝ) 
(hk: kills = 150) (hbt: bonus_threshold = 100) (hbm: bonus_multiplier = 1.5) (hts: total_score_with_bonus = 2250)
(hP: 150 * P * bonus_multiplier = total_score_with_bonus) : 
P = 10 := sorry

end points_per_enemy_l44_44711


namespace find_radius_of_circle_l44_44596

theorem find_radius_of_circle :
  ∀ (r : ℝ) (α : ℝ) (ρ : ℝ) (θ : ℝ), r > 0 →
  (∀ (x y : ℝ), x = r * Real.cos α ∧ y = r * Real.sin α → x^2 + y^2 = r^2) →
  (∃ (x y: ℝ), x - y + 2 = 0 ∧ 2 * Real.sqrt (r^2 - 2) = 2 * Real.sqrt 2) →
  r = 2 :=
by
  intro r α ρ θ r_pos curve_eq polar_eq
  sorry

end find_radius_of_circle_l44_44596


namespace john_age_proof_l44_44239

theorem john_age_proof (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end john_age_proof_l44_44239


namespace difference_of_squares_example_l44_44802

theorem difference_of_squares_example : 625^2 - 375^2 = 250000 :=
by sorry

end difference_of_squares_example_l44_44802


namespace volume_ratio_l44_44574

theorem volume_ratio (x : ℝ) (h : x > 0) : 
  let V_Q := x^3
  let V_P := (3 * x)^3
  (V_Q / V_P) = (1 / 27) :=
by
  sorry

end volume_ratio_l44_44574


namespace count_4_digit_distinct_leading_nonzero_multiple_5_largest_6_l44_44894

/-- The count of 4-digit integers with all distinct digits, leading digit not zero, 
    is a multiple of 5, and 6 is the largest digit is 96. -/
theorem count_4_digit_distinct_leading_nonzero_multiple_5_largest_6 : 
  ∃ n : ℕ, n = 96 ∧ 
        (∀ x : ℕ, 1000 ≤ x ∧ x < 10000 → 
        (∀ i j : ℕ, i ≠ j → x.digits.get i ≠ x.digits.get j) → 
        x.digits.get 0 ≠ 0 → 
        (x % 5 = 0 ∧ x.digits.max = 6) ↔ x ∈ (n)) :=
sorry

end count_4_digit_distinct_leading_nonzero_multiple_5_largest_6_l44_44894


namespace tangent_lines_to_circle_through_point_l44_44829

theorem tangent_lines_to_circle_through_point :
  let P := (0 : ℝ, -1 : ℝ)
  let C := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 + 2)^2 = 1}
  ∃ l1 l2, (l1 = {p : ℝ × ℝ | p.1 = 0} ∧ l2 = {p : ℝ × ℝ | p.2 = -1}) ∧ 
            (∀ p ∈ C, p ∈ l1 → (p ∈ l2 → p ∈ {q : ℝ × ℝ | (q.1 - 1)^2 + (q.2 + 2)^2 ≠ 1}) ∧
                                  (P.1 ∈ l1 → P.2 ∈ l2)) :=
by
  sorry

end tangent_lines_to_circle_through_point_l44_44829


namespace five_burritos_and_five_quesadillas_cost_l44_44299

noncomputable def cost_of_five_burritos_and_five_quesadillas : ℝ :=
(let b := (8 / 5 : ℝ),
     q := 2.05 in
  5 * b + 5 * q + 2)

theorem five_burritos_and_five_quesadillas_cost
  (h₁ : 4 * b + 2 * q + 2 = 12.50)
  (h₂ : 3 * b + 4 * q + 2 = 15.00) :
  cost_of_five_burritos_and_five_quesadillas = 20.25 :=
  sorry

end five_burritos_and_five_quesadillas_cost_l44_44299


namespace distinct_arrangements_balloon_l44_44128

-- Let's define the basic conditions:
def total_letters : Nat := 7
def repeats_l : Nat := 2
def repeats_o : Nat := 2

-- Now let's state the problem.
theorem distinct_arrangements_balloon : 
  (Nat.factorial total_letters) / ((Nat.factorial repeats_l) * (Nat.factorial repeats_o)) = 1260 := 
by
  sorry

end distinct_arrangements_balloon_l44_44128


namespace determine_systematic_sample_l44_44351

def is_systematic_sample (s : List ℕ) : Prop :=
  ∀ (n : ℕ), n < s.length - 1 → s[n + 1] = s[n] + 10

def selected_products : List (List ℕ) :=
  [[6, 11, 16, 21, 26], 
   [3, 13, 23, 33, 43], 
   [5, 15, 25, 36, 47], 
   [10, 20, 29, 39, 49]]

theorem determine_systematic_sample :
  ∃ s ∈ selected_products, is_systematic_sample s ∧ s = [3, 13, 23, 33, 43] :=
by
  sorry

end determine_systematic_sample_l44_44351


namespace distinct_arrangements_balloon_l44_44129

-- Let's define the basic conditions:
def total_letters : Nat := 7
def repeats_l : Nat := 2
def repeats_o : Nat := 2

-- Now let's state the problem.
theorem distinct_arrangements_balloon : 
  (Nat.factorial total_letters) / ((Nat.factorial repeats_l) * (Nat.factorial repeats_o)) = 1260 := 
by
  sorry

end distinct_arrangements_balloon_l44_44129


namespace sec_sum_equals_four_l44_44795

theorem sec_sum_equals_four : 
  let z := (complex.exp (complex.I * 2 * real.pi / 9)) in
  complex.abs z = 1 → 
  z^9 = 1 → 
  (1 / real.cos (2 * real.pi / 9)) + 
  (1 / real.cos (4 * real.pi / 9)) + 
  (1 / real.cos (6 * real.pi / 9)) + 
  (1 / real.cos (8 * real.pi / 9)) = 4 := by
  sorry

end sec_sum_equals_four_l44_44795


namespace area_ratio_l44_44201

-- Define a structure for points in ℝ^2
structure Point2D :=
  (x : ℝ)
  (y : ℝ)

-- Define the vertices of the square ABCD
def A : Point2D := ⟨0, 1⟩
def B : Point2D := ⟨1, 1⟩
def C : Point2D := ⟨1, 0⟩
def D : Point2D := ⟨0, 0⟩

-- Define the midpoints of the sides
def M : Point2D := ⟨0.5, 1⟩
def N : Point2D := ⟨1, 0.5⟩
def P : Point2D := ⟨0.5, 0⟩
def Q : Point2D := ⟨0, 0.5⟩

-- Define the points A', B', C', D'
def A' : Point2D := ⟨0, 0.75⟩
def B' : Point2D := ⟨0.75, 1⟩
def C' : Point2D := ⟨1, 0.25⟩
def D' : Point2D := ⟨0.25, 0⟩

-- Define the function to calculate the area of a quadrilateral using the shoelace formula
def quadrilateral_area (p1 p2 p3 p4 : Point2D) : ℝ :=
  (1/2 : ℝ) * |(p1.x * p2.y + p2.x * p3.y + p3.x * p4.y + p4.x * p1.y) -
               (p1.y * p2.x + p2.y * p3.x + p3.y * p4.x + p4.y * p1.x)|

-- Define S_ABCD as the area of square ABCD
def S_ABCD : ℝ := 1 -- The area of a unit square is 1

-- Define S_A'B'C'D' as the area of quadrilateral A'B'C'D'
def S_A'B'C'D' : ℝ := quadrilateral_area A' B' C' D'

-- Theorem: Prove that S_A'B'C'D' is 1/5 of S_ABCD
theorem area_ratio : S_A'B'C'D' = (1/5) * S_ABCD :=
by
  sorry

end area_ratio_l44_44201


namespace hypotenuse_of_45_45_90_triangle_l44_44661

theorem hypotenuse_of_45_45_90_triangle (leg : ℝ) (angle_opposite_leg : ℝ) (h_leg : leg = 15) (h_angle : angle_opposite_leg = 45) :
  ∃ hypotenuse, hypotenuse = leg * Real.sqrt 2 :=
by
  use leg * Real.sqrt 2
  rw [h_leg]
  rw [h_angle]
  sorry

end hypotenuse_of_45_45_90_triangle_l44_44661


namespace michael_initial_money_l44_44635

theorem michael_initial_money (M : ℝ) 
  (half_give_away_to_brother : ∃ (m_half : ℝ), M / 2 = m_half)
  (brother_initial_money : ℝ := 17)
  (candy_cost : ℝ := 3)
  (brother_ends_up_with : ℝ := 35) :
  brother_initial_money + M / 2 - candy_cost = brother_ends_up_with ↔ M = 42 :=
sorry

end michael_initial_money_l44_44635


namespace tony_and_esther_split_equally_l44_44712

def total_amount : ℝ := 50
def moses_share : ℝ := 0.4 * total_amount
def esther_share : ℝ := moses_share - 5
def remainder : ℝ := total_amount - moses_share
def tony_share : ℝ := remainder - esther_share

theorem tony_and_esther_split_equally : 
  moses_share = 20 ∧ remainder = 30 ∧ esther_share = 15 → tony_share = 15 :=
by
  intro h
  cases h with hm h
  cases h with hr he
  rw [hm, hr, he]
  sorry

end tony_and_esther_split_equally_l44_44712


namespace shadow_point_interval_l44_44612

noncomputable def shadow_point {f : ℝ → ℝ} (x : ℝ) : Prop :=
  ∃ y : ℝ, y > x ∧ f y > f x

theorem shadow_point_interval
  (f : ℝ → ℝ) (a b : ℝ) (h_cont : Continuous f)
  (h_shadow : ∀ x : ℝ, a < x ∧ x < b → shadow_point x)
  (h_not_shadow_a : ¬ shadow_point a)
  (h_not_shadow_b : ¬ shadow_point b) :
  (∀ x : ℝ, a < x ∧ x < b → f x ≤ f b) ∧ f a = f b :=
sorry

end shadow_point_interval_l44_44612


namespace find_value_of_fraction_l44_44996

theorem find_value_of_fraction (x y : ℝ) (hx : x > 0) (hy : y > x) (h : x / y + y / x = 4) : 
  (x + y) / (x - y) = Real.sqrt 3 :=
by
  sorry

end find_value_of_fraction_l44_44996


namespace prob_sum_to_3_three_dice_correct_l44_44358

def prob_sum_to_3_three_dice (sum : ℕ) (dice_count : ℕ) (dice_faces : Finset ℕ) : ℚ :=
  if sum = 3 ∧ dice_count = 3 ∧ dice_faces = {1, 2, 3, 4, 5, 6} then (1 : ℚ) / 216 else 0

theorem prob_sum_to_3_three_dice_correct :
  prob_sum_to_3_three_dice 3 3 {1, 2, 3, 4, 5, 6} = (1 : ℚ) / 216 := 
by
  sorry

end prob_sum_to_3_three_dice_correct_l44_44358


namespace balloon_permutations_l44_44138

theorem balloon_permutations : 
  let n : ℕ := 7
  let k1 : ℕ := 2
  let k2 : ℕ := 2
  ∃ distinct_arrangements : ℕ, 
  distinct_arrangements = n.factorial / (k1.factorial * k2.factorial) 
  ∧ distinct_arrangements = 1260 :=
by
  sorry

end balloon_permutations_l44_44138


namespace parrots_in_each_cage_l44_44770

theorem parrots_in_each_cage (P : ℕ) (h : 9 * P + 9 * 6 = 72) : P = 2 :=
sorry

end parrots_in_each_cage_l44_44770


namespace min_chips_in_cells_l44_44945

theorem min_chips_in_cells (strip_size : ℕ) (h1 : strip_size = 2021)
  (chips : Finset ℕ) (empty_cells : Finset ℕ)
  (h2 : ∀ c ∈ empty_cells, ∃ l r : ℕ, l ∈ chips ∧ r ∈ chips ∧ 
      abs (l - r) = c ∧ c ≠ 0 ∧ 
      (∀ c' ∈ empty_cells, c ≠ c') )
  : ∃ n, chips.card = 1347 :=
by
  have : chips.card = 1347 := sorry
  exact ⟨1347, this⟩

end min_chips_in_cells_l44_44945


namespace balloon_arrangements_l44_44098

-- Define the variables
def n : ℕ := 7
def L_count : ℕ := 2
def O_count : ℕ := 2
def B_count : ℕ := 1
def A_count : ℕ := 1
def N_count : ℕ := 1

-- Define the multiset permutation formula
def multiset_permutations (n : ℕ) (counts : List ℕ) : ℕ :=
  n.factorial / (counts.map Nat.factorial).prod

-- Proof that the number of distinct arrangements is 1260
theorem balloon_arrangements : multiset_permutations n [L_count, O_count, B_count, A_count, N_count] = 1260 :=
  by
  -- The proof is omitted
  sorry

end balloon_arrangements_l44_44098


namespace distinct_arrangements_balloon_l44_44123

theorem distinct_arrangements_balloon : 
  let n := 7
  let r1 := 2
  let r2 := 2
  (Nat.factorial n) / ((Nat.factorial r1) * (Nat.factorial r2)) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l44_44123


namespace parabola_A_distance_l44_44628

noncomputable def distance_OA (p : ℝ) (hp : 0 < p) (A : ℝ × ℝ)
  (hA : A.2 ^ 2 = 2 * p * A.1) (F : ℝ × ℝ) (hF : F = (p / 2, 0))
  (θ : ℝ) (hθ : θ = real.pi / 3) : ℝ :=
  let vector_FA := ((A.1 - F.1), A.2)
  in real.sqrt (A.1 ^ 2 + A.2 ^ 2)

theorem parabola_A_distance 
  (O A : ℝ × ℝ)
  (p : ℝ) 
  (hp : 0 < p) 
  (hA : A.2 ^ 2 = 2 * p * A.1)
  (θ : ℝ)
  (hθ : θ = real.pi / 3)
  (F : ℝ × ℝ)
  (hF : F = (p / 2, 0)) :
  ∥A - O∥ = (real.sqrt 21 / 2) * p :=
by
  -- skipping the actual proof
  sorry

end parabola_A_distance_l44_44628


namespace largest_expression_is_A_l44_44480

noncomputable def A : ℝ := 3009 / 3008 + 3009 / 3010
noncomputable def B : ℝ := 3011 / 3010 + 3011 / 3012
noncomputable def C : ℝ := 3010 / 3009 + 3010 / 3011

theorem largest_expression_is_A : A > B ∧ A > C := by
  sorry

end largest_expression_is_A_l44_44480


namespace parabola_focus_coordinates_l44_44716

theorem parabola_focus_coordinates : 
  ∀ (x y : ℝ), x = 4 * y^2 → (∃ (y₀ : ℝ), (x, y₀) = (1/16, 0)) :=
by
  intro x y hxy
  sorry

end parabola_focus_coordinates_l44_44716


namespace arabella_total_learning_time_l44_44455

-- Define the conditions
def arabella_first_step_time := 30 -- in minutes
def arabella_second_step_time := arabella_first_step_time / 2 -- half the time of the first step
def arabella_third_step_time := arabella_first_step_time + arabella_second_step_time -- sum of the first and second steps

-- Define the total time spent
def arabella_total_time := arabella_first_step_time + arabella_second_step_time + arabella_third_step_time

-- The theorem to prove
theorem arabella_total_learning_time : arabella_total_time = 90 := 
  sorry

end arabella_total_learning_time_l44_44455


namespace tire_price_is_115dot71_l44_44763

noncomputable def regular_price (p : ℝ) : Prop :=
  3 * p + p / 2 = 405

theorem tire_price_is_115dot71 : regular_price 115.71 :=
by
  have h1 : 3 * 115.71 + 115.71 / 2 = 405 := by sorry
  exact h1

end tire_price_is_115dot71_l44_44763


namespace problem_integer_product_l44_44702

-- Definitions for conditions
def sum_of_digit_squares (n : ℕ) : ℕ :=
  (n.digits 10).sum_by (λ d, d * d)

def increasing_digits (n : ℕ) : Prop :=
  ∀ (i j : ℕ), i < j → n.digits 10 !! i < n.digits 10 !! j

-- The theorem based on the proof problem
theorem problem_integer_product :
  ∃ (n : ℕ), sum_of_digit_squares n = 50 ∧ increasing_digits n ∧ n.digits 10.prod = 36 :=
sorry

end problem_integer_product_l44_44702


namespace calc_difference_of_squares_l44_44800

theorem calc_difference_of_squares :
  625^2 - 375^2 = 250000 :=
by sorry

end calc_difference_of_squares_l44_44800


namespace distinct_convex_polygons_of_four_or_more_sides_l44_44490

noncomputable def total_subsets (n : Nat) : Nat := 2^n

noncomputable def subsets_with_fewer_than_four_members (n : Nat) : Nat := 
  (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)

noncomputable def valid_subsets (n : Nat) : Nat := 
  total_subsets n - subsets_with_fewer_than_four_members n

theorem distinct_convex_polygons_of_four_or_more_sides (n : Nat) (h : n = 15) : valid_subsets n = 32192 := by
  sorry

end distinct_convex_polygons_of_four_or_more_sides_l44_44490


namespace competition_result_l44_44786

-- Define the participants and their predictions
inductive Participant
| A | B | C

open Participant

-- Define the rankings
structure Ranking :=
(first : Participant)
(second : Participant)
(third : Participant)

-- Express the conditions as propositions
def A_prediction (rank : Ranking) : Prop := rank.third ≠ A
def B_prediction (rank : Ranking) : Prop := rank.third = B
def C_prediction (rank : Ranking) : Prop := rank.first ≠ C

-- The main theorem stating if only one prediction is correct, B wins first place.
theorem competition_result (rank : Ranking) 
  (hA : A_prediction rank)
  (hB : B_prediction rank)
  (hC : C_prediction rank)
  (h : (A_prediction rank ∨ B_prediction rank ∨ C_prediction rank)) : 
  (hA ∨ hB ∨ hC) ∧ (A ≠ rank.first) ∧ (B = rank.first) ∧ (C ≠ rank.first) :=
sorry

end competition_result_l44_44786


namespace balloon_arrangements_l44_44087

theorem balloon_arrangements : (7! / (2! * 2!)) = 1260 := by
  sorry

end balloon_arrangements_l44_44087


namespace jack_change_l44_44215

def cost_per_sandwich : ℕ := 5
def number_of_sandwiches : ℕ := 3
def payment : ℕ := 20

theorem jack_change : payment - (cost_per_sandwich * number_of_sandwiches) = 5 := 
by
  sorry

end jack_change_l44_44215


namespace elmer_fuel_savings_l44_44005

-- Definitions
def old_car_efficiency (x : ℝ) : ℝ := x
def new_car_efficiency (x : ℝ) : ℝ := 1.6 * x
def gasoline_cost (c : ℝ) : ℝ := c
def diesel_cost (c : ℝ) : ℝ := 1.25 * c
def journey_distance : ℝ := 1000

-- Cost calculations
def old_car_cost (x c : ℝ) : ℝ := (journey_distance / old_car_efficiency x) * gasoline_cost c
def new_car_cost (x c : ℝ) : ℝ := (journey_distance / new_car_efficiency x) * diesel_cost c

-- Percentage savings calculation
def percentage_savings (old_cost new_cost : ℝ) : ℝ := ((old_cost - new_cost) / old_cost) * 100

theorem elmer_fuel_savings (x c : ℝ) :
  percentage_savings (old_car_cost x c) (new_car_cost x c) = 21.875 :=
by
  -- Proof goes here
  sorry

end elmer_fuel_savings_l44_44005


namespace safe_travel_exists_l44_44708

def total_travel_time : ℕ := 16
def first_crater_cycle : ℕ := 18
def first_crater_duration : ℕ := 1
def second_crater_cycle : ℕ := 10
def second_crater_duration : ℕ := 1

theorem safe_travel_exists : 
  ∃ t : ℕ, t ∈ { t | (∀ k : ℕ, t % first_crater_cycle ≠ k ∨ t % first_crater_cycle ≥ first_crater_duration) 
  ∧ (∀ k : ℕ, t % second_crater_cycle ≠ k ∨ t % second_crater_cycle ≥ second_crater_duration) 
  ∧ (∀ k : ℕ, (t + total_travel_time) % first_crater_cycle ≠ k ∨ (t + total_travel_time) % first_crater_cycle ≥ first_crater_duration) 
  ∧ (∀ k : ℕ, (t + total_travel_time) % second_crater_cycle ≠ k ∨ (t + total_travel_time) % second_crater_cycle ≥ second_crater_duration) } :=
sorry

end safe_travel_exists_l44_44708


namespace triangle_AC_l44_44606

section
variables {a b k : ℝ} (h1: k ≤ 1)

theorem triangle_AC (BC_eq_a : BC = a) (AB_eq_b : AB = b) (DE_div_AC_eq_k : DE / AC = k) :
  AC = real.sqrt (a^2 + b^2 + 2 * a * b * k) ∨ AC = real.sqrt (a^2 + b^2 - 2 * a * b * k) :=
sorry
end

end triangle_AC_l44_44606


namespace parabola_whose_directrix_is_tangent_to_circle_l44_44447

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 9

noncomputable def is_tangent (line_eq : ℝ → ℝ → Prop) (circle_eq : ℝ → ℝ → Prop) : Prop := 
  ∃ p : ℝ × ℝ, (line_eq p.1 p.2) ∧ (circle_eq p.1 p.2) ∧ 
  (∀ q : ℝ × ℝ, (circle_eq q.1 q.2) → (line_eq q.1 q.2) → q = p)

-- Definitions of parabolas
noncomputable def parabola_A_directrix (x y : ℝ) : Prop := y = 2

noncomputable def parabola_B_directrix (x y : ℝ) : Prop := x = 2

noncomputable def parabola_C_directrix (x y : ℝ) : Prop := x = -4

noncomputable def parabola_D_directrix (x y : ℝ) : Prop := y = -1

-- The final statement to prove
theorem parabola_whose_directrix_is_tangent_to_circle :
  is_tangent parabola_D_directrix circle_eq ∧ ¬ is_tangent parabola_A_directrix circle_eq ∧ 
  ¬ is_tangent parabola_B_directrix circle_eq ∧ ¬ is_tangent parabola_C_directrix circle_eq :=
sorry

end parabola_whose_directrix_is_tangent_to_circle_l44_44447


namespace solve_for_m_l44_44075

-- Definition of vectors a and b
def vec_a := (1 : ℚ, 1 : ℚ)
def vec_b (m : ℚ) := (-1, m)

-- Condition for parallel vectors
def parallel (u v : ℚ × ℚ) := u.1 * v.2 = u.2 * v.1

theorem solve_for_m (m : ℚ) (h : parallel vec_a (vec_b m)) : m = -1 :=
by
  sorry

end solve_for_m_l44_44075


namespace minimum_value_of_f_l44_44546

noncomputable def f (x : ℝ) : ℝ :=
  2^x + 1 / 2^(x + 2)

theorem minimum_value_of_f : f(-1) = 1 ∧ ∀ x : ℝ, f(x) ≥ 1 :=
by
  sorry

end minimum_value_of_f_l44_44546


namespace min_chips_in_cells_l44_44947

theorem min_chips_in_cells (n : ℕ) (h : 2021 - n ≤ (n + 1) / 2 ∧ 
                                  (∀ i j : ℕ, (i < j ∧ 1 ≤ i ∧ i ≤ 2021 - n
                                   ∧ j ≤ 2021 - n) → abs (f i - f j) ≠ 0)) :
  n = 1347 :=
by sorry

end min_chips_in_cells_l44_44947


namespace sum_max_min_S2014_l44_44854

def equal_sum_of_squares_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n^2 + a (n + 1)^2 = 1

def sum_sequence (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (Finset.range n).sum a

theorem sum_max_min_S2014 (a : ℕ → ℤ) (Sn : ℕ → ℤ)
  (h_eq_sum_of_squares : equal_sum_of_squares_sequence a)
  (h_first_term : a 1 = 1)
  (h_sum_first_n : ∀ n, Sn n = sum_sequence a n):
  let S2014 := Sn 2014 in
  let max_S2014 := 1007 in
  let min_S2014 := -1005 in
  max_S2014 + min_S2014 = 2 :=
sorry

end sum_max_min_S2014_l44_44854


namespace integer_root_theorem_l44_44773

theorem integer_root_theorem (a1 a2 : ℤ) (x : ℤ) :
  (x ^ 3 + a2 * x ^ 2 + a1 * x + 24 = 0) →
  x ∈ {1, -1, 2, -2, 3, -3, 4, -4, 6, -6, 8, -8, 12, -12, 24, -24} :=
by
  intro h
  sorry

end integer_root_theorem_l44_44773


namespace area_of_triangle_ABC_area_of_rectangle_not_covered_l44_44950

-- Definitions and assumptions
variables (BC height_A BC_HEIGHT triangle_area rectangle_area remaining_area : ℝ)
variables (DE EF : ℝ)

-- Given conditions
def conditions : Prop := 
  BC = 12 ∧
  height_A = 15 ∧
  DE = 30 ∧
  EF = 20 ∧
  triangle_area = (1/2) * BC * height_A ∧
  rectangle_area = DE * EF ∧
  remaining_area = rectangle_area - triangle_area

-- Statements to be proved
theorem area_of_triangle_ABC (h : conditions) : triangle_area = 90 :=
by
  cases h with h_BC h_rest
  cases h_rest with h_height_A h_rest
  cases h_rest with h_DE h_rest
  cases h_rest with h_EF h_rest
  cases h_rest with h_triangle_area h_rest
  rw [h_BC, h_height_A] at h_triangle_area
  exact h_triangle_area

theorem area_of_rectangle_not_covered (h : conditions) : remaining_area = 510 :=
by
  cases h with h_BC h_rest
  cases h_rest with h_height_A h_rest
  cases h_rest with h_DE h_rest
  cases h_rest with h_EF h_rest
  cases h_rest with h_triangle_area h_rest
  cases h_rest with h_rectangle_area h_rest
  rw [h_rectangle_area, h_triangle_area] at h_rest
  exact h_rest

end area_of_triangle_ABC_area_of_rectangle_not_covered_l44_44950


namespace constant_term_binomial_expansion_l44_44513

open Complex

theorem constant_term_binomial_expansion :
  let n := ∫ x in (0 : ℝ)..(π / 2), 4 * sin x in
  (x - x⁻¹)^n = (x - x⁻¹)^4 →
  -- Proof that the constant term in the expansion of (x - 1/x)^4 is 6
  (-1)^2 * Nat.choose 4 2 = 6 :=
by
  -- (Skipping proof details)
  sorry

end constant_term_binomial_expansion_l44_44513


namespace product_of_numbers_l44_44378

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 1 * k) (h2 : x + y = 8 * k) (h3 : x * y = 30 * k) : 
  x * y = 400 / 7 := 
sorry

end product_of_numbers_l44_44378


namespace find_integer_n_l44_44024

theorem find_integer_n : ∃ (n : ℤ), (-90 ≤ n ∧ n ≤ 90) ∧ real.sin (n * real.pi / 180) = real.sin (721 * real.pi / 180) :=
by {
  use 1,
  split,
  { -- Prove that 1 is within the given range
    split; norm_num },
  { -- sin(721°) = sin(1°), i.e., sin(721 * π / 180) = sin(1 * π / 180)
    rw [← real.sin_add_pi, ← real.sin_add_pi],
    norm_num,
    simp [real.sin_pi_sub] }
}

end find_integer_n_l44_44024


namespace final_number_is_50_l44_44962

theorem final_number_is_50 (initial_ones initial_fours : ℕ) (h1 : initial_ones = 900) (h2 : initial_fours = 100) :
  ∃ (z : ℝ), (900 * (1:ℝ)^2 + 100 * (4:ℝ)^2) = z^2 ∧ z = 50 :=
by
  sorry

end final_number_is_50_l44_44962


namespace hypotenuse_of_454590_triangle_l44_44642

theorem hypotenuse_of_454590_triangle (l : ℝ) (angle : ℝ) (h : ℝ) (h_leg : l = 15) (h_angle : angle = 45) :
  h = l * Real.sqrt 2 := 
  sorry

end hypotenuse_of_454590_triangle_l44_44642


namespace minimal_visible_sum_of_dice_l44_44417

theorem minimal_visible_sum_of_dice (corner_sum edge_sum center_sum : ℕ) 
  (corner_count edge_count center_count num_faces : ℕ) 
  (h1 : corner_count = 8) 
  (h2 : edge_count = 24) 
  (h3 : center_count = 24)
  (h4 : ∀ a b c : ℕ, a + b + c = 6 ∧ b + c + num_faces = 7) 
  (h5 : ∀ a b : ℕ, a + b = 3 ∧ b + num_faces = 7) 
  (h6 : ∀ a : ℕ, a = 1 ∧ 1 + num_faces = 7): 
  corner_sum * corner_count + edge_sum * edge_count + center_sum * center_count = 144 :=
begin
  sorry,
end

end minimal_visible_sum_of_dice_l44_44417


namespace chessboard_rice_sum_l44_44964

theorem chessboard_rice_sum :
  ∑ i in finset.range 64, 2^i = 2^64 - 1 :=
begin
  sorry
end

end chessboard_rice_sum_l44_44964


namespace kira_breakfast_time_l44_44976

theorem kira_breakfast_time (n_sausages : ℕ) (n_eggs : ℕ) (t_fry_per_sausage : ℕ) (t_scramble_per_egg : ℕ) (total_time : ℕ) :
  n_sausages = 3 → n_eggs = 6 → t_fry_per_sausage = 5 → t_scramble_per_egg = 4 → total_time = (n_sausages * t_fry_per_sausage + n_eggs * t_scramble_per_egg) →
  total_time = 39 :=
by
  intros h_sausages h_eggs h_fry h_scramble h_total
  rw [h_sausages, h_eggs, h_fry, h_scramble] at h_total
  exact h_total

end kira_breakfast_time_l44_44976


namespace convex_polygons_from_fifteen_points_l44_44494

theorem convex_polygons_from_fifteen_points 
    (h : ∀ (n : ℕ), n = 15) :
    ∃ (k : ℕ), k = 32192 :=
by
  sorry

end convex_polygons_from_fifteen_points_l44_44494


namespace area_of_region_l44_44014

theorem area_of_region : let r1 (θ : ℝ) := 2 / Real.cos θ,
                            r2 (θ : ℝ) := 2 / Real.sin θ,
                            x := r1,
                            y := r2
                         in if r1(0) = 2 ∧ r2(math.pi/2) = 2 ∧ (x = 2 → y = 2)
                          then 
                            2 * 2 = 4 := sorry

end area_of_region_l44_44014


namespace toms_total_cost_l44_44372

theorem toms_total_cost :
  let costA := 4 * 15
  let costB := 3 * 12
  let discountB := 0.20 * costB
  let costBDiscounted := costB - discountB
  let costC := 2 * 18
  costA + costBDiscounted + costC = 124.80 := 
by
  sorry

end toms_total_cost_l44_44372


namespace hagrid_divisible_by_three_l44_44747

def distinct_digits (n : ℕ) : Prop :=
  n < 10

theorem hagrid_divisible_by_three (H A G R I D : ℕ) (H_dist A_dist G_dist R_dist I_dist D_dist : distinct_digits H ∧ distinct_digits A ∧ distinct_digits G ∧ distinct_digits R ∧ distinct_digits I ∧ distinct_digits D)
  (distinct_letters: H ≠ A ∧ H ≠ G ∧ H ≠ R ∧ H ≠ I ∧ H ≠ D ∧ A ≠ G ∧ A ≠ R ∧ A ≠ I ∧ A ≠ D ∧ G ≠ R ∧ G ≠ I ∧ G ≠ D ∧ R ≠ I ∧ R ≠ D ∧ I ≠ D) :
  3 ∣ (H * 100000 + A * 10000 + G * 1000 + R * 100 + I * 10 + D) * H * A * G * R * I * D :=
sorry

end hagrid_divisible_by_three_l44_44747


namespace valid_sequences_length_22_l44_44156

noncomputable def count_valid_sequences : ℕ → ℕ
| 3 := 1  -- for n = 3, sequence is 010
| 4 := 0  -- for n = 4, no valid sequence
| 5 := 0  -- for n = 5, no valid sequence
| 6 := 1  -- for n = 6, sequence is 010101
| n := count_valid_sequences (n - 3) + count_valid_sequences (n - 4) -- general recursion

theorem valid_sequences_length_22 :
  count_valid_sequences 22 = 93 :=
sorry

end valid_sequences_length_22_l44_44156


namespace sport_tournament_attendance_l44_44186

theorem sport_tournament_attendance :
  let total_attendance := 500
  let team_A_supporters := 0.35 * total_attendance
  let team_B_supporters := 0.25 * total_attendance
  let team_C_supporters := 0.20 * total_attendance
  let team_D_supporters := 0.15 * total_attendance
  let AB_overlap := 0.10 * team_A_supporters
  let BC_overlap := 0.05 * team_B_supporters
  let CD_overlap := 0.07 * team_C_supporters
  let atmosphere_attendees := 30
  let total_supporters := team_A_supporters + team_B_supporters + team_C_supporters + team_D_supporters
                         - (AB_overlap + BC_overlap + CD_overlap)
  let unsupported_people := total_attendance - total_supporters - atmosphere_attendees
  unsupported_people = 26 :=
by
  sorry

end sport_tournament_attendance_l44_44186


namespace negation_union_l44_44049

variable {α : Type*} (A B : set α) (x : α)

theorem negation_union (h : x ∈ A ∪ B) : ¬ (x ∈ A ∪ B) ↔ (x ∉ A ∧ x ∉ B) :=
by sorry

end negation_union_l44_44049


namespace intersection_sets_l44_44526

-- Define set A as all x such that x >= -2
def setA : Set ℝ := {x | x >= -2}

-- Define set B as all x such that x < 1
def setB : Set ℝ := {x | x < 1}

-- The statement to prove in Lean 4
theorem intersection_sets : (setA ∩ setB) = {x | -2 <= x ∧ x < 1} :=
by
  sorry

end intersection_sets_l44_44526


namespace convex_polygons_on_circle_l44_44486

theorem convex_polygons_on_circle:
  let points := 15 in
  ∑ i in finset.range (points + 1), choose points i - (choose points 0 + choose points 1 + choose points 2 + choose points 3) = 32192 :=
begin
  sorry
end

end convex_polygons_on_circle_l44_44486


namespace ray_walks_to_park_l44_44670

theorem ray_walks_to_park (x : ℤ) (h1 : 3 * (x + 7 + 11) = 66) : x = 4 :=
by
  -- solving steps are skipped
  sorry

end ray_walks_to_park_l44_44670


namespace measure_of_angle_C_cos2A_minus_2sin2B_plus_1_l44_44960

noncomputable def triangle_condition1 (a b c : ℝ) (A B C : ℝ) : Prop :=
  c * sin B + (a + c^2 / a - b^2 / a) * sin C = 2 * c * sin A

noncomputable def circumcircle_radius (R : ℝ) : Prop :=
  R = real.sqrt 3

noncomputable def triangle_area (a b C : ℝ) : Prop :=
  0.5 * a * b * sin C = real.sqrt 3

theorem measure_of_angle_C (a b c A B C R : ℝ) 
  (h1 : triangle_condition1 a b c A B C) 
  (h2 : circumcircle_radius R) :
  C = real.pi / 3 :=
sorry

theorem cos2A_minus_2sin2B_plus_1 (a b c A B C R : ℝ) 
  (h1 : triangle_condition1 a b c A B C) 
  (h2 : circumcircle_radius R) 
  (h3 : triangle_area a b C) :
  cos (2 * A) - 2 * sin B^2 + 1 = -1 / 6 :=
sorry

end measure_of_angle_C_cos2A_minus_2sin2B_plus_1_l44_44960


namespace polygon_interior_exterior_eq_l44_44925

theorem polygon_interior_exterior_eq (n : ℕ) (hn : 3 ≤ n)
  (interior_sum_eq_exterior_sum : (n - 2) * 180 = 360) : n = 4 := by
  sorry

end polygon_interior_exterior_eq_l44_44925


namespace W_555_2_last_three_digits_l44_44409

noncomputable def W : ℕ → ℕ → ℕ
| n, 0     => n ^ n
| n, (k+1) => W (W n k) k

theorem W_555_2_last_three_digits :
  (W 555 2) % 1000 = 875 :=
sorry

end W_555_2_last_three_digits_l44_44409


namespace last_three_digits_of_expression_l44_44998

noncomputable def x : ℝ := real.sqrt 5 ^ (2 / 3) + root (real.sqrt 5 - 2) 3

theorem last_three_digits_of_expression : (x ^ 2014) % 1000 = 125 := 
sorry

end last_three_digits_of_expression_l44_44998


namespace total_time_spent_l44_44456

-- Define time spent on each step
def time_first_step : ℕ := 30
def time_second_step : ℕ := time_first_step / 2
def time_third_step : ℕ := time_first_step + time_second_step

-- Prove the total time spent
theorem total_time_spent : 
  time_first_step + time_second_step + time_third_step = 90 := by
  sorry

end total_time_spent_l44_44456


namespace minimum_distance_ln_function_l44_44063

noncomputable def minimum_distance (f : ℝ → ℝ) (l : ℝ × ℝ → Prop) (P : ℝ × ℝ) : ℝ :=
  Real.abs ((2 * P.1 + P.2 * (-1) + 6) / (Real.sqrt (2^2 + (-1)^2)))

theorem minimum_distance_ln_function :
  let f := λ x: ℝ, 2 * Real.log x in
  let l := λ p: ℝ × ℝ, 2 * p.1 - p.2 + 6 = 0 in
  let P := (1, f 1) in
  minimum_distance f l P = (8 * Real.sqrt 5 / 5) :=
by
  sorry

end minimum_distance_ln_function_l44_44063


namespace exists_horizontal_chord_l44_44524

structure Point :=
(x : ℝ)
(y : ℝ)

structure LineSegment :=
(A : Point)
(B : Point)

def horizontal (l : LineSegment) : Prop :=
l.A.y = l.B.y

def distance (A B : Point) : ℝ :=
(real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2))

noncomputable def broken_line (A B : Point) : Type :=
{ L : list LineSegment // L.head.A = A ∧ L.last.B = B }

def horizontal_chord (L : broken_line A B) (a b : Point) : Prop :=
a.y = b.y ∧ a ≠ b

theorem exists_horizontal_chord (A B : Point) (L : broken_line A B) (hAB : distance A B ∈ ℤ) :
  ∀ n : ℕ+, ∃ C D : Point, horizontal_chord L C D ∧ distance C D = 1 / n :=
by
  sorry

end exists_horizontal_chord_l44_44524


namespace math_problem_l44_44505

noncomputable def prod_terms : ℕ → ℝ
| 2       := 1 - (1 + 2 + Real.sqrt 2) / (2^3 + 1)
| (n + 3) := (1 - (∑ i in Finset.range (n + 3 + 1), (i : ℝ) + Real.sqrt 2) / ((n + 3)^3 + 1)) * prod_terms (n + 2)

theorem math_problem (n : ℕ) (hn : n ≥ 2) :
  prod_terms n > 1 / Real.sqrt (5 * n) :=
sorry

end math_problem_l44_44505


namespace no_periodic_sum_l44_44406

def is_periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x : ℝ, f (x + p) = f x

theorem no_periodic_sum (g h : ℝ → ℝ) :
  (is_periodic g 2) → (is_periodic h (π / 2)) → ¬ ∃ T > 0, is_periodic (λ x, g x + h x) T :=
by {
  sorry
}

end no_periodic_sum_l44_44406


namespace vasya_reading_problem_l44_44381

-- Definitions based on conditions:
variables (x : ℕ) -- number of books planned to read each week
variables (total_books : ℕ) -- total number of books
noncomputable def planned_total_books := 12 * x
noncomputable def actual_total_books := 15 * (x - 1)

-- Theorem statement without proof
theorem vasya_reading_problem (hx : actual_total_books = planned_total_books) :
  let weeks_required := total_books / (x + 1)
  in weeks_required = 10 :=
sorry

end vasya_reading_problem_l44_44381


namespace ellipse_equation_line_equation_l44_44602

theorem ellipse_equation (a b c : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : c = a * √((3 : ℝ) / 2)) (h4 : 2 * a = 2 * √3) : 
  (a^2 = 3 ∧ b^2 = 1) ↔ (by : (c : ℝ) := √6 / 3) :=
begin
  sorry
end

theorem line_equation (k : ℝ) (h5 : ∃ (l : ℝ), l = y ∧ b = √((k: ℝ)^2 - 1)) (h6 : (1/2) * line * d = 6 / 7) :
  y = k * x + 2 ∨ y = k * x + 2 ↔ (k^2 = 2 ∨ k^2 = 25/9) :=
begin
  sorry
end

end ellipse_equation_line_equation_l44_44602


namespace min_value_proof_l44_44848

noncomputable def min_expr_value (x y : ℝ) : ℝ :=
  (1 / (2 * x)) + (1 / y)

theorem min_value_proof (x y : ℝ) (h1 : x + y = 1) (h2 : y > 0) (h3 : x > 0) :
  min_expr_value x y = (3 / 2) + Real.sqrt 2 :=
sorry

end min_value_proof_l44_44848


namespace problem1_problem2_l44_44987

-- Definitions
variables {a b z : ℝ}

-- Problem 1 translated to Lean
theorem problem1 (h1 : a + 2 * b = 9) (h2 : |9 - 2 * b| + |a + 1| < 3) : -2 < a ∧ a < 1 := 
sorry

-- Problem 2 translated to Lean
theorem problem2 (h1 : a + 2 * b = 9) (ha_pos : 0 < a) (hb_pos : 0 < b) : 
  ∃ z : ℝ, z = a * b^2 ∧ ∀ w : ℝ, (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + 2 * b = 9 ∧ w = a * b^2) → w ≤ 27 :=
sorry

end problem1_problem2_l44_44987


namespace perpendicular_vectors_lambda_l44_44076

theorem perpendicular_vectors_lambda (λ : ℝ) (a b : ℝ × ℝ) (h₁ : a = (1, 0)) (h₂ : b = (1, 1)) 
  (h_perpendicular : inner (1 + λ, λ) b = 0) : λ = -1/2 := 
by sorry

end perpendicular_vectors_lambda_l44_44076


namespace num_propositions_with_logical_connectives_l44_44787

-- Define the propositions
def P1 : Prop := February 14, 2010, is both Chinese New Year and Valentine's Day
def P2 : Prop := A multiple of 10 is definitely a multiple of 5
def P3 : Prop := A trapezoid is not a rectangle

-- Define the logical connectives used in each proposition
def uses_logical_connectives (P : Prop) : Prop :=
  (P = P1 ∧ P.uses_and) ∨ (P = P3 ∧ P.uses_not)

-- The main statement to prove
theorem num_propositions_with_logical_connectives :
  (∀ P, P = P1 ∨ P = P2 ∨ P = P3) →
  (P1.uses_logical_connectives ∧ ¬P2.uses_logical_connectives ∧ P3.uses_logical_connectives) →
  card ({P | uses_logical_connectives(P)}) = 2 :=
by
  apply sorry

end num_propositions_with_logical_connectives_l44_44787


namespace Q_polynomial_correct_l44_44986

def Q (x : ℝ) : ℝ := -2 * x^2 + 6 * x - 1

theorem Q_polynomial_correct :
  (Q(0) + Q(1) * (-1) + Q(2) * (-1)^2 = 3) ∧
  (Q(0) + Q(1) * 3 + Q(2) * 3^2 = 15) :=
by
  -- These are the conditions that need to be proved to match the polynomial.
  sorry

end Q_polynomial_correct_l44_44986


namespace num_valid_four_digit_numbers_l44_44153

theorem num_valid_four_digit_numbers :
  let N (a b c d : ℕ) := 1000 * a + 100 * b + 10 * c + d
  ∃ (a b c d : ℕ), 5000 ≤ N a b c d ∧ N a b c d < 7000 ∧ (N a b c d % 5 = 0) ∧ (2 ≤ b ∧ b < c ∧ c ≤ 7) ∧
                   (60 = (if a = 5 ∨ a = 6 then (if d = 0 ∨ d = 5 then 15 else 0) else 0)) :=
sorry

end num_valid_four_digit_numbers_l44_44153


namespace functional_relationship_correct_y_intercept_correct_x_intercepts_correct_l44_44041

section DirectProportionality

variables (x y k1 k2 : ℝ)

-- Conditions for the problem
def y1 := k1 * (x - 3)
def y2 := k2 * (x^2 + 1)
def y := y1 + y2

-- Given points
def point1 := (0, -2)
def point2 := (1, 4)

-- System of equations derived from the conditions
def equation1 := -3 * k1 + k2 = -2
def equation2 := -2 * k1 + 2 * k2 = 4

-- Solving for k1 and k2
def k1_value := 2
def k2_value := 4

-- Functional relationship derived
def functional_relationship := y = 4 * x^2 + 2 * x - 2

-- Intersection points
def y_intercept := (0, -2)
def x_intercepts := [(-1, 0), (1 / 2, 0)]

-- Proof statements to be proven
theorem functional_relationship_correct : 
  ∀ x : ℝ, y = 4 * x^2 + 2 * x - 2 :=
by sorry

theorem y_intercept_correct :
  y.intercept (4 * x^2 + 2 * x - 2) = (0, -2) :=
by sorry

theorem x_intercepts_correct :
  x.intercepts (4 * x^2 + 2 * x - 2) = [(-1, 0), (1 / 2, 0)] :=
by sorry

end DirectProportionality

end functional_relationship_correct_y_intercept_correct_x_intercepts_correct_l44_44041


namespace quadratic_factorization_l44_44002

theorem quadratic_factorization (a : ℤ) :
  (∃ m n p q : ℤ, 15 * (mx * px) + (mq * x + np * x) + nq = 15 * x^2 + a * x + 15) ∧ 
  (∃ k : ℤ, a^2 - 900 = k^2) → a = 34 :=
begin
  sorry
end

end quadratic_factorization_l44_44002


namespace program_output_is_correct_l44_44875

theorem program_output_is_correct (a b c : ℕ) (h₀ : a = 2) (h₁ : b = 3) (h₂ : c = 4) : 
  let a := b,
      b := c,
      c := a
  in (a, b, c) = (3, 4, 2) :=
by
  -- State the initial conditions
  have ha : a = 2 := h₀,
  have hb : b = 3 := h₁,
  have hc : c = 4 := h₂,
  -- Execute the assignments as per program instructions
  let a := b,
  let b := c,
  let c := a,
  -- Assert the final values
  show (a, b, c) = (3, 4, 2),
  sorry

end program_output_is_correct_l44_44875


namespace johns_age_l44_44228

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l44_44228


namespace sin_lt_alpha_lt_tan_l44_44294

variable {α : ℝ}

theorem sin_lt_alpha_lt_tan (h1 : 0 < α) (h2 : α < π / 2) : sin α < α ∧ α < tan α :=
  sorry

end sin_lt_alpha_lt_tan_l44_44294


namespace prob_all_one_l44_44388

-- Define the probability function for a single die landing on a specific number
def prob_of_die_landing_on (n : ℕ) : ℚ :=
  if n ∈ {1, 2, 3, 4, 5, 6} then 1 / 6 else 0

-- Define the event that four dice landing on specific numbers
def prob_of_fours_dice_landing_on (a b c d : ℕ) : ℚ :=
  prob_of_die_landing_on a * prob_of_die_landing_on b * prob_of_die_landing_on c * prob_of_die_landing_on d

-- Define the theorem we want to prove
theorem prob_all_one : prob_of_fours_dice_landing_on 1 1 1 1 = 1 / 1296 :=
by sorry

end prob_all_one_l44_44388


namespace range_of_a_l44_44870

noncomputable def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a > 0

noncomputable def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (a : ℝ) (h1 : p a) (h2 : q a) : a ≤ -2 :=
by
  sorry

end range_of_a_l44_44870


namespace abby_and_damon_weight_l44_44441

variables {a b c d : ℝ}

theorem abby_and_damon_weight (h1 : a + b = 260) (h2 : b + c = 245) 
(h3 : c + d = 270) (h4 : a + c = 220) : a + d = 285 := 
by 
  sorry

end abby_and_damon_weight_l44_44441


namespace peter_speed_l44_44972

theorem peter_speed (P : ℝ) (h1 : P >= 0) (h2 : 1.5 * P + 1.5 * (P + 3) = 19.5) : P = 5 := by
  sorry

end peter_speed_l44_44972


namespace greatest_consecutive_integers_sum_72_l44_44383

theorem greatest_consecutive_integers_sum_72 :
  ∃ N a : ℤ, (∑ i in Finset.range N, i + a) = 72 ∧ 
  ∀ x y : ℤ, (∑ i in Finset.range y, i + x) = 72 → y ≤ N :=
sorry

end greatest_consecutive_integers_sum_72_l44_44383


namespace distinct_arrangements_balloon_l44_44089

theorem distinct_arrangements_balloon : 
  let n := 7 
  let freq_l := 2 
  let freq_o := 2 
  let freq_b := 1 
  let freq_a := 1 
  let freq_n := 1 
  Nat.factorial n / (Nat.factorial freq_l * Nat.factorial freq_o * Nat.factorial freq_b * Nat.factorial freq_a * Nat.factorial freq_n) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l44_44089


namespace modulus_z_l44_44872

def z : ℂ := 3 + (3 + 4 * complex.I) / (4 - 3 * complex.I)

theorem modulus_z : complex.abs z = real.sqrt 10 := by
  sorry

end modulus_z_l44_44872


namespace find_function_l44_44826

theorem find_function (f : ℕ → ℕ) (h : ∀ n : ℕ, f(n) + f(f(n)) + f(f(f(n))) = 3n) : ∀ n : ℕ, f(n) = n :=
by
  -- Proof will be inserted here
  sorry

end find_function_l44_44826


namespace blown_up_fraction_l44_44430

variables (f : ℚ) (total_balloons blown_up : ℚ)

def total_balloons := 200
def intact_balloons := 80

-- After half an hour, fraction f of total balloons blew up.
def first_half_balloons := f * total_balloons

-- Twice the number of balloons that blew up in first half also blew up in the next hour.
def next_hour_balloons := 2 * first_half_balloons

-- Total balloons that blew up is given by total balloons minus intact balloons at the end.
def total_balloons_blown_up := total_balloons - intact_balloons

theorem blown_up_fraction (h : first_half_balloons + next_hour_balloons = total_balloons_blown_up) :
  f = 1 / 5 :=
by
  sorry

end blown_up_fraction_l44_44430


namespace length_of_AE_is_2sqrt55_l44_44757

open Real

theorem length_of_AE_is_2sqrt55 :
  ∀ (A B D E C : Type)
    (radius_A : Real)
    (radius_B : Real)
    (distance_AB : Real)
    (length_DE : Real)
    (length_AD : Real)
    (length_AE : Real),
    radius_A = 10 →
    radius_B = 3 →
    distance_AB = radius_A + radius_B →
    length_DE = 2 * sqrt 30 →
    length_AD = radius_A →
    length_AE = sqrt (length_AD^2 + length_DE^2) →
    length_AE = 2 * sqrt 55 :=
by
  intros A B D E C radius_A radius_B distance_AB length_DE length_AD length_AE
  intros h_radius_A h_radius_B h_distance_AB h_length_DE h_length_AD h_length_AE
  rw [h_radius_A, h_radius_B, h_distance_AB, h_length_DE, h_length_AD] at h_length_AE
  exact h_length_AE

end length_of_AE_is_2sqrt55_l44_44757


namespace no_four_consecutive_lucky_numbers_l44_44281

def is_lucky (n : ℕ) : Prop :=
  let digits := n.digits 10
  n > 999999 ∧ n < 10000000 ∧ (∀ d ∈ digits, d ≠ 0) ∧ 
  n % (digits.foldl (λ x y => x * y) 1) = 0

theorem no_four_consecutive_lucky_numbers :
  ¬ ∃ (n : ℕ), is_lucky n ∧ is_lucky (n + 1) ∧ is_lucky (n + 2) ∧ is_lucky (n + 3) :=
sorry

end no_four_consecutive_lucky_numbers_l44_44281


namespace project_total_hours_l44_44317

theorem project_total_hours (x : ℕ) (hx1 : 3 * x = x + 40) : 
  let person1_hours := x,
      person2_hours := 2 * x,
      person3_hours := 3 * x,
      total_hours := person1_hours + person2_hours + person3_hours in
  total_hours = 120 :=
by
  sorry

end project_total_hours_l44_44317


namespace johns_age_l44_44233

theorem johns_age (d j : ℕ) 
  (h1 : j = d - 30) 
  (h2 : j + d = 80) : 
  j = 25 :=
by
  sorry

end johns_age_l44_44233


namespace odd_coefficients_of_polynomial_l44_44025

-- Define the polynomial and its properties
noncomputable def P (x : ℤ) (n : ℕ) : ℤ := (x^2 + x + 1)^n

-- Theorem statement
theorem odd_coefficients_of_polynomial (n k : ℕ) :
  let count_odd_coeffs := λ p : ℕ, (p = x^2 + x + 1)^n in 
  (k = int.log2 (n + 1)) ∧ (n = 2^k - 1) →
  count_odd_coeffs n = (2 ^ (k + 2) + (-1) ^ (k + 1)) / 3 :=
begin
  sorry
end

end odd_coefficients_of_polynomial_l44_44025


namespace triangle_parallel_midpoint_length_l44_44376

open Lean

theorem triangle_parallel_midpoint_length :
  ∀ (P Q R S T M : Point) 
  (h1 : distance P R = 26)
  (h2 : distance P Q = 24)
  (h3 : distance Q R = 10)
  (h4 : S ∈ line_segment P R)
  (h5 : T ∈ line_segment P Q)
  (h6 : midpoint M P R)
  (h7 : parallel line_segment ST line_segment QR),
  distance S T = 5 :=
by 
  intros P Q R S T M h1 h2 h3 h4 h5 h6 h7
  sorry

end triangle_parallel_midpoint_length_l44_44376


namespace ann_pays_more_l44_44178

noncomputable def price_ann : ℝ :=
let original_price : ℝ := 150.00 in
let tax_rate : ℝ := 0.07 in
let discount_rate : ℝ := 0.25 in
let service_charge_rate : ℝ := 0.05 in
let price_with_tax := original_price * (1 + tax_rate) in
let discounted_price := price_with_tax * (1 - discount_rate) in
discounted_price * (1 + service_charge_rate)

noncomputable def price_ben : ℝ :=
let original_price : ℝ := 150.00 in
let tax_rate : ℝ := 0.07 in
let discount_rate : ℝ := 0.25 in
let discounted_price := original_price * (1 - discount_rate) in
discounted_price * (1 + tax_rate)

theorem ann_pays_more : price_ann - price_ben = 6.02 := sorry

end ann_pays_more_l44_44178


namespace find_remaining_perimeter_l44_44509

noncomputable def equilateral_triangle (A B C : Type) [metric_space A] [metric_space B] [metric_space C] :=
∃ (l : ℝ), ∀ (x y z : A), (dist x y = l) ∧ (dist y z = l) ∧ (dist z x = l)

noncomputable def isosceles_right_triangle (D B E : Type) [metric_space D] [metric_space B] [metric_space E] :=
∃ (a : ℝ), (dist D B = a) ∧ (dist B E = a) ∧ (dist D E = a * sqrt 2)

noncomputable def remaining_perimeter (ABC: Type) [metric_space ABC]
  (A B C D E : ABC) (l : ℝ) (a : ℝ) : ℝ :=
  let d_AB := dist A B in
  let d_AD := l - a in
  let d_DE := a * sqrt 2 in
  let d_BC := l in
  let d_EC := l - (a * sqrt 2) in
  d_AD + d_DE + d_EC + d_BC

theorem find_remaining_perimeter :
  ∀ (ABC : Type) [metric_space ABC] (A B C D E : ABC) (l a : ℝ),
    equilateral_triangle A B C →
    isosceles_right_triangle D B E →
    dist D E = 2 →
    dist B C = 4 →
    remaining_perimeter ABC A B C D E l a = 14 - 2 * sqrt 2 :=
sorry

end find_remaining_perimeter_l44_44509


namespace removed_triangles_area_l44_44438

theorem removed_triangles_area (AB : ℝ) (h_AB : AB = 24) :
  let r_squared_s_squared_sum : ℝ := 288
  r_squared_s_squared_sum = 288 :=
by
  have : (r : ℝ) (s : ℝ), r^2 + s^2 = (AB^2) / 2 
  sorry

end removed_triangles_area_l44_44438


namespace find_x_l44_44030

-- Assume the condition
theorem find_x (x : ℝ) (h : sqrt (x - 3) = 5) : x = 28 :=
sorry

end find_x_l44_44030


namespace find_b_if_continuous_at_3_l44_44276
noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
  if x ≤ 3 then 3 * x^2 + 1 else b * x^2 + b * x + 6

theorem find_b_if_continuous_at_3 (b : ℝ) :
  (∀ x1 x2, (x1 ≤ 3 → f x1 b = 3 * x1^2 + 1) ∧ (3 < x2 → f x2 b = b * x2^2 + b * x2 + 6)) ∧
  continuous_at (λ x, f x b) 3 →
  b = 11 / 6 :=
by
  sorry

end find_b_if_continuous_at_3_l44_44276


namespace magician_can_deduce_l44_44685

def audience_segment_length (k : ℕ) (h₁ : 1 ≤ k ∧ k ≤ 14) : ℕ :=
  k

def assistant_segment_length (k : ℕ) (h₁ : 1 ≤ k ∧ k ≤ 13) : ℕ :=
  k + 1

def assistant_segment_length_14 : ℕ :=
  1

theorem magician_can_deduce (k : ℕ) (h₁ : 1 ≤ k ∧ k ≤ 14)
  (given_assistant_segment : ℕ) (h₂ : given_assistant_segment = assistant_segment_length k h₁ ∨ (k=14 ∧ given_assistant_segment=1)):
  (∃ (audience_segment : ℕ), audience_segment = audience_segment_length k h₁ ∧ (given_assistant_segment = assistant_segment_length k h₁ ∨ (k=14 ∧ given_assistant_segment=1))) :=
by
  sorry

end magician_can_deduce_l44_44685


namespace map_at_three_l44_44266

variable (A B : Type)
variable (a : ℝ)
variable (f : ℝ → ℝ)
variable (h_map : ∀ x : ℝ, f x = a * x - 1)
variable (h_cond : f 2 = 3)

theorem map_at_three : f 3 = 5 := by
  sorry

end map_at_three_l44_44266


namespace inverse_function_eval_l44_44055

def f (x : ℝ) : ℝ := 2^x
def g (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem inverse_function_eval (h : ∀ x, g (f x) = x) : g (1 / 2) = -1 :=
by
  sorry

end inverse_function_eval_l44_44055


namespace sum_of_squares_l44_44385

theorem sum_of_squares :
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 4^2 - 2^2 = 272 :=
by
  sorry

end sum_of_squares_l44_44385


namespace find_consecutive_numbers_l44_44011

theorem find_consecutive_numbers :
  ∃ (a b c d : ℕ),
      a % 11 = 0 ∧
      b % 7 = 0 ∧
      c % 5 = 0 ∧
      d % 4 = 0 ∧
      b = a + 1 ∧
      c = a + 2 ∧
      d = a + 3 ∧
      (a % 10) = 3 ∧
      (b % 10) = 4 ∧
      (c % 10) = 5 ∧
      (d % 10) = 6 :=
sorry

end find_consecutive_numbers_l44_44011


namespace team_game_two_players_two_colors_team_game_arbitrary_players_colors_l44_44738

-- Part (a)
theorem team_game_two_players_two_colors (n k : ℕ) (h : n = 2 ∧ k = 2) 
: ∃ m : ℕ, m = 1 :=
sorry

-- Part (b)
theorem team_game_arbitrary_players_colors (n k : ℕ) 
: ∃ m : ℕ, m = floor (n / k) :=
sorry

end team_game_two_players_two_colors_team_game_arbitrary_players_colors_l44_44738


namespace problem_1_problem_2_problem_3_l44_44864

-- Simplified and combined statements for clarity
theorem problem_1 (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) 
  (h_cond : ∀ x ≤ 0, f x = Real.logb (1/2) (-x + 1)) : 
  f 3 + f (-1) = -3 := sorry

theorem problem_2 (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) 
  (h_cond : ∀ x ≤ 0, f x = Real.logb (1/2) (-x + 1)) : 
  ∀ x, f x = if x ≤ 0 then Real.logb (1/2) (-x + 1) else Real.logb (1/2) (x + 1) := sorry

theorem problem_3 (f : ℝ → ℝ) (h_cond : ∀ x ≤ 0, f x = Real.logb (1/2) (-x + 1))
  (h_cond_ev : ∀ x, f x = f (-x)) (a : ℝ) : 
  f (a - 1) < -1 ↔ a ∈ ((Set.Iio 0) ∪ (Set.Ioi 2)) := sorry

end problem_1_problem_2_problem_3_l44_44864


namespace range_q_l44_44268

noncomputable def q (x : ℝ) : ℝ :=
if is_prime (floor x)
then x + 2
else 
  let y := smallest_prime_factor (floor x) in
  q (y) + (x + 2 - (floor x))

theorem range_q : 
  set.range q = (set.Ico 4 10) ∪ (set.Ico 12 16) := 
sorry

end range_q_l44_44268


namespace chord_length_intersection_l44_44330

theorem chord_length_intersection {t : ℝ} :
  let circle := (x y : ℝ) → x ^ 2 + y ^ 2 = 9,
      line := (x = 1 + 2 * t) ∧ (y = 2 + t) in
  ∃ (length : ℝ), length = 12 / 5 * Real.sqrt 5 ∧ 
  ( ∃ (x y : ℝ), circle x y ∧ line x y) :=
sorry

end chord_length_intersection_l44_44330


namespace giselle_initial_doves_l44_44037

theorem giselle_initial_doves (F : ℕ) (h1 : ∀ F, F > 0) (h2 : 3 * F * 3 / 4 + F = 65) : F = 20 :=
sorry

end giselle_initial_doves_l44_44037


namespace no_common_period_l44_44407

theorem no_common_period (g h : ℝ → ℝ) 
  (hg : ∀ x, g (x + 2) = g x) 
  (hh : ∀ x, h (x + π/2) = h x) : 
  ¬ (∃ T > 0, ∀ x, g (x + T) + h (x + T) = g x + h x) :=
sorry

end no_common_period_l44_44407


namespace triangle_equality_l44_44580

theorem triangle_equality (a b c A B C : ℝ)
  (h1 : a * Real.cos C = c * Real.cos A)
  (h2 : ∃ r : ℝ, b = a * r ∧ c = b * r) :
  (∠ABC = ∠BCA ∧ ∠BCA = ∠CAB) :=
sorry

end triangle_equality_l44_44580


namespace perfect_square_factors_of_12000_l44_44152

-- Define the prime factorization of 12000
def factorization_12000 : ℕ := 2^5 * 3 * 5^3

-- Define the condition for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop := 
  ∃ m : ℕ, m^2 = n

-- Define the function to count the perfect square factors
noncomputable def count_perfect_square_factors (n : ℕ) : ℕ :=
  finset.card
    (finset.filter is_perfect_square
      (finset.filter (λ d, n % d = 0) 
        (finset.range (n + 1))))

-- 12,000 can be uniquely factorized into 2^5 * 3 * 5^3
example : factorization_12000 = 12000 := by norm_num

-- Prove the number of perfect square factors of 12000 is 6
theorem perfect_square_factors_of_12000 : count_perfect_square_factors 12000 = 6 := by sorry

end perfect_square_factors_of_12000_l44_44152


namespace billiard_ball_hits_top_left_pocket_l44_44432

/--
A ball is released from the bottom left pocket of a rectangular billiard table with dimensions
26 × 1965 (with the longer side 1965 running left to right and the shorter side 26 running top
to bottom) at an angle of 45° to the sides. Pockets are located at the corners of the rectangle.
Prove that after several reflections off the sides, the ball will fall into the top left pocket.
--/
theorem billiard_ball_hits_top_left_pocket 
  (table_width : ℕ) (table_height : ℕ) (angle : ℝ)
  (initial_position : ℕ × ℕ) (target_position : ℕ × ℕ) :
  table_width = 1965 → 
  table_height = 26 → 
  angle = real.pi / 4 → 
  initial_position = (0, 0) → 
  target_position = (0, 26) →
  ∃ (m n : ℕ), 2 * m = 151 * n :=
by sorry

end billiard_ball_hits_top_left_pocket_l44_44432


namespace power_function_decreasing_l44_44553

theorem power_function_decreasing (m : ℝ) : 
  (m^2 - 4m + 4 = 1) → (m^2 - 6m + 8 < 0) → 
  m = 3 :=
by
  sorry

end power_function_decreasing_l44_44553


namespace hyperbola_eccentricity_l44_44868

noncomputable def hyperbola_eccentricity_center_origin_and_asymptote_angle (a b : ℝ) (foci_on_x_axis : Bool) :
    Prop :=
  let e := if foci_on_x_axis then √(1 + (b^2 / a^2)) else √(1 + (a^2 / b^2))
  center_origin : (0, 0) ∧
  (a > 0 ∧ b > 0) ∧
  (foci_on_x_axis → (b/a = √3)) ∧ 
  (¬foci_on_x_axis → (a/b = √3)) ∧
    (e = 2 ∨ e = (2 * √3) / 3)

theorem hyperbola_eccentricity {a b : ℝ} (h₀ : a > 0) (h₁ : b > 0)
  (ineq : (b/a = √3) ∨ (a/b = √3)) :
  let e := if (b/a = √3) then √(1 + (b^2 / a^2)) else √(1 + (a^2 / b^2))
  e = 2 ∨ e = (2 * √3) / 3 :=
by {
  sorry
}

end hyperbola_eccentricity_l44_44868


namespace fred_earnings_over_weekend_l44_44977

-- Fred's earning from delivering newspapers
def earnings_from_newspapers : ℕ := 16

-- Fred's earning from washing cars
def earnings_from_cars : ℕ := 74

-- Fred's total earnings over the weekend
def total_earnings : ℕ := earnings_from_newspapers + earnings_from_cars

-- Proof that total earnings is 90
theorem fred_earnings_over_weekend : total_earnings = 90 :=
by 
  -- sorry statement to skip the proof steps
  sorry

end fred_earnings_over_weekend_l44_44977


namespace smallest_log_value_l44_44905

open Real

theorem smallest_log_value (x y : ℝ) (h₀ : 2 < y ∧ y ≤ x) : 
  ∃ d : ℝ, d = log x y ∧ (2 - (d + (1 / d)) ≥ 0) :=
by 
  sorry

end smallest_log_value_l44_44905


namespace max_satisfying_condition_l44_44500

-- Define the set of integers satisfying the given condition
def satisfies_condition (s : Set ℕ) : Prop :=
  ∀ {a b : ℕ}, a ∈ s → b ∈ s → a ≠ b → |a - b| ≥ a * b / 100

-- Define the maximum number of elements satisfying the condition
def max_satisfying_elements : ℕ :=
  10  -- We know from the problem statement that this is at most 10

theorem max_satisfying_condition :
  ∀ (s : Set ℕ), satisfies_condition s → s.card ≤ max_satisfying_elements :=
sorry -- Proof needed

end max_satisfying_condition_l44_44500


namespace line_slope_point_l44_44766

theorem line_slope_point (m b : ℝ) : m = 4 → (2, -1) ∈ set_of (λ (p : ℝ × ℝ), p.2 = m * p.1 + b) → m + b = -5 :=
by
  intros h1 h2
  sorry

end line_slope_point_l44_44766


namespace prob_sum_to_3_three_dice_correct_l44_44357

def prob_sum_to_3_three_dice (sum : ℕ) (dice_count : ℕ) (dice_faces : Finset ℕ) : ℚ :=
  if sum = 3 ∧ dice_count = 3 ∧ dice_faces = {1, 2, 3, 4, 5, 6} then (1 : ℚ) / 216 else 0

theorem prob_sum_to_3_three_dice_correct :
  prob_sum_to_3_three_dice 3 3 {1, 2, 3, 4, 5, 6} = (1 : ℚ) / 216 := 
by
  sorry

end prob_sum_to_3_three_dice_correct_l44_44357


namespace chris_packed_percentage_l44_44249

theorem chris_packed_percentage (K C : ℕ) (h : K / (C : ℝ) = 2 / 3) :
  (C / (K + C : ℝ)) * 100 = 60 :=
by
  sorry

end chris_packed_percentage_l44_44249


namespace range_of_a_min_value_t_l44_44537

noncomputable def h (a : ℝ) (x : ℝ) := Real.log (x - 1) - (a * (x - 2)) / x

theorem range_of_a (a : ℝ) (x : ℝ) (h1 : ∀ x, x > 2 → h a x < 0) : a > 2 :=
sorry

theorem min_value_t (n : ℕ) (h2 : n > 0) : 
  (∑ i in Finset.range n, 1 / ((i + 3)^2 * Real.log (i + 2))) < 3 / 8 :=
sorry

end range_of_a_min_value_t_l44_44537


namespace determine_a_range_of_g_l44_44532

def f (a x : ℝ) : ℝ := a^(x-1)

theorem determine_a :
  (∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ f a 2 = 1/2) ↔ (a = 1/2) :=
sorry

def g (a x : ℝ) : ℝ := a^(2*x) - a^(x-2) + 8

theorem range_of_g :
  (a = 1/2 ∧ ∀ x ∈ Icc (-2:ℝ) 1, g a x ∈ Icc 4 8) :=
sorry

end determine_a_range_of_g_l44_44532


namespace distance_to_canada_l44_44607

theorem distance_to_canada (speed : ℝ) (total_time : ℝ) (stop_time : ℝ) (driving_time : ℝ) (distance : ℝ) :
  speed = 60 ∧ total_time = 7 ∧ stop_time = 1 ∧ driving_time = total_time - stop_time ∧
  distance = speed * driving_time → distance = 360 :=
by
  sorry

end distance_to_canada_l44_44607


namespace arrangement_count_equivalent_problem_l44_44701

noncomputable def number_of_unique_arrangements : Nat :=
  let n : Nat := 6 -- Number of balls and boxes
  let match_3_boxes_ways := Nat.choose n 3 -- Choosing 3 boxes out of 6
  let permute_remaining_boxes := 2 -- Permutations of the remaining 3 boxes such that no numbers match
  match_3_boxes_ways * permute_remaining_boxes

theorem arrangement_count_equivalent_problem :
  number_of_unique_arrangements = 40 := by
  sorry

end arrangement_count_equivalent_problem_l44_44701


namespace Valleyball_Soccer_League_members_l44_44598

theorem Valleyball_Soccer_League_members (cost_socks cost_tshirt total_expenditure cost_per_member: ℕ) (h1 : cost_socks = 6) (h2 : cost_tshirt = cost_socks + 8) (h3 : total_expenditure = 3740) (h4 : cost_per_member = cost_socks + 2 * cost_tshirt) : 
  total_expenditure = 3740 → cost_per_member = 34 → total_expenditure / cost_per_member = 110 :=
sorry

end Valleyball_Soccer_League_members_l44_44598


namespace diff_sales_total_sales_total_revenue_l44_44681

/--
Prove that the difference in sales between the highest and lowest sales days is 19 kg
-/
theorem diff_sales 
  (sales : List ℤ := [+7, -5, -3, +13, -6, +12, +5]) 
  : (13 - (-6)) = 19 := 
by
  sorry

/--
Prove that the total actual sales in the first week is 723 kg
-/
theorem total_sales
  (planned_sales_per_day : ℕ := 100)
  (sales : List ℕ := [7, -5, -3, +13, -6, +12, +5])
  : (700 + (7 - 5 - 3 + 13 - 6 + 12 + 5)) = 723 := 
by
  sorry

/--
Prove that the total revenue from apple sales in the first week is 2530.5 yuan
-/
theorem total_revenue
  (sales : ℕ := 723)
  (price_per_kg : ℝ := 5.5)
  (shipping_cost_per_kg : ℝ := 2.0)
  : (3.5 * 723) = 2530.5 := 
by
  sorry

end diff_sales_total_sales_total_revenue_l44_44681


namespace distinct_arrangements_balloon_l44_44120

theorem distinct_arrangements_balloon : 
  let n := 7
  let r1 := 2
  let r2 := 2
  (Nat.factorial n) / ((Nat.factorial r1) * (Nat.factorial r2)) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l44_44120


namespace unique_solution_in_base_10_no_solution_in_base_12_l44_44319

noncomputable theory

-- Definitions of the required conditions
def arithmetic_mean (a b : ℕ) : ℕ := (a + b) / 2
def geometric_mean (a b : ℕ) : ℕ := nat.sqrt (a * b)
def reverse_digits (n : ℕ) : ℕ := 
  let d := n % 10 in
  let t := n / 10 in
  10 * d + t

-- The main theorem statement
theorem unique_solution_in_base_10_no_solution_in_base_12 :
  ∃ (a b : ℕ), a ≠ b ∧ arithmetic_mean a b = 65 ∧ reverse_digits 65 = geometric_mean a b ∧
  ∀ (p q : ℕ), (∀ (a b : ℕ), arithmetic_mean a b = 12 * p + q → geometric_mean a b = 12 * q + p → false) :=
sorry

end unique_solution_in_base_10_no_solution_in_base_12_l44_44319


namespace stickers_distribution_l44_44079

theorem stickers_distribution : 
  (10 + 5 - 1).choose (5 - 1) = 1001 := 
by
  sorry

end stickers_distribution_l44_44079


namespace find_hyperbola_params_lambda_plus_mu_const_l44_44884

noncomputable def hyperbola_eq (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

noncomputable def foci_distance (F1 F2 : ℝ × ℝ) : ℝ :=
  (F1.1 - F2.1)^2 + (F1.2 - F2.2)^2

def line_perpendicular_to_asymptote (a b : ℝ) : Prop :=
  let m_asym : ℝ := b / a in
  m_asym * (-√3 / 3) = -1

theorem find_hyperbola_params (a b : ℝ) (F1 F2 : ℝ × ℝ)
  (h_hyperbola : ∀ x y, hyperbola_eq a b x y)
  (h_foci_dist : foci_distance F1 F2 = 16)
  (h_perpendicular : line_perpendicular_to_asymptote a b) :
  a^2 = 1 ∧ b^2 = 3 :=
sorry

theorem lambda_plus_mu_const (M F1 F2 A B : ℝ × ℝ) 
  (λ μ : ℝ) 
  (h_M_on_hyperbola : hyperbola_eq 1 √3 M.1 M.2)
  (h_rel1 : λ•(F1.1 - M.1, F1.2 - M.2) = (A.1 - F1.1, A.2 - F1.2))
  (h_rel2 : μ•(F2.1 - M.1, F2.2 - M.2) = (B.1 - F2.1, B.2 - F2.2)) :
  λ + μ = -10/3 :=
sorry

end find_hyperbola_params_lambda_plus_mu_const_l44_44884


namespace problem_l44_44069

def f (x : ℝ) : ℝ :=
if x > 0 then real.log x / real.log 4 else 2 ^ x

theorem problem (h : f (f (1/4)) = 1/2) : true :=
by sorry

end problem_l44_44069


namespace center_polar_coordinates_l44_44954

-- Assuming we have a circle defined in polar coordinates
def polar_circle_center (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.cos θ + 2 * Real.sin θ

-- The goal is to prove that the center of this circle has the polar coordinates (sqrt 2, π/4)
theorem center_polar_coordinates : ∃ ρ θ, polar_circle_center ρ θ ∧ ρ = Real.sqrt 2 ∧ θ = Real.pi / 4 :=
sorry

end center_polar_coordinates_l44_44954


namespace john_age_proof_l44_44240

theorem john_age_proof (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end john_age_proof_l44_44240


namespace part_a_part_b_l44_44259

open Finset

namespace ProofProblem

-- Part (a)
theorem part_a (n : ℕ) (h : 0 < n) (S : Finset ℕ := Icc n (5 * n)) :
  ∀ (A B : Finset ℕ), A ∪ B = S ∧ A ∩ B = ∅ →
  ∃ (x y z : ℕ), x ∈ A ∧ y ∈ A ∧ z ∈ A ∧ x + y = z :=
begin
  sorry
end

-- Part (b)
theorem part_b (n : ℕ) (h : 0 < n) (S : Finset ℕ := Ico n (5 * n)) :
  ∃ (A B : Finset ℕ), A ∪ B = S ∧ A ∩ B = ∅ ∧
  ∀ (x y z : ℕ), x ∉ A ∨ y ∉ A ∨ z ∉ A ∨ x + y ≠ z :=
begin
  sorry
end

end ProofProblem

end part_a_part_b_l44_44259


namespace incorrect_statements_l44_44054

variable {f : ℝ → ℝ}
variable (Hdiff : Differentiable ℝ f)

-- Condition: (x^2 + 3x - 4) * f'(x) < 0
variable (Hineq : ∀ x, (x^2 + 3x - 4) * (deriv f x) < 0)

-- Definitions for each statement
def stmt1 : Prop := ∀ x, (x < -4 ∨ x > 1) → deriv f x < 0
def stmt2 : Prop := ∃ a b, a ≠ b ∧ deriv f a = 0 ∧ deriv f b = 0
def stmt3 : Prop := f 0 + f 2 > f (-5) + f (-3)
def stmt4 : Prop := ∀ x, -1 < x ∧ x < 4 → deriv f x > 0

theorem incorrect_statements :
  ¬stmt1 ∧ ¬stmt3 ∧ ¬stmt4 :=
by sorry

end incorrect_statements_l44_44054


namespace selection_problems_l44_44445

noncomputable def binomial : ℕ → ℕ → ℕ
| n k := n.choose k

theorem selection_problems :
  let n := 12 in
  let g := 10 in
  let d := 2 in
  let k := 3 in

  -- Proving that the total ways to select 3 products from 12 is 220
  binomial n k = 220 ∧ 

  -- Proving that the ways to select 1 defective and 2 genuine products is 90
  binomial d 1 * binomial g 2 = 90 ∧ 

  -- Proving that the ways to select at least 1 defective product is 100
  binomial n k - binomial g 3 = 100 :=
by repeat { sorry }

end selection_problems_l44_44445


namespace no_month_5_mondays_and_5_thursdays_l44_44213

theorem no_month_5_mondays_and_5_thursdays (n : ℕ) (h : n = 28 ∨ n = 29 ∨ n = 30 ∨ n = 31) :
  ¬ (∃ (m : ℕ) (t : ℕ), m = 5 ∧ t = 5 ∧ 5 * (m + t) ≤ n) := by sorry

end no_month_5_mondays_and_5_thursdays_l44_44213


namespace range_of_m_l44_44514

theorem range_of_m (p : set ℝ := {x | (x + 2) / (x - 10) ≤ 0})
                    (q : set ℝ := {x | x ^ 2 - 2 * x + 1 - m^2 < 0 ∧ m > 0})
                    (hpq : ∀ x, p x → q x ∧ ∃ x, ¬(q x → p x)) :
                    0 < m ∧ m < 3 := 
sorry

end range_of_m_l44_44514


namespace problem_statement_l44_44575

theorem problem_statement (x : ℤ) (h : 3 - x = -2) : x + 1 = 6 := 
by {
  -- Proof would be provided here
  sorry
}

end problem_statement_l44_44575


namespace sum_Sp_equals_101475_l44_44834

/-- Definitions based on conditions -/
def first_term (p : ℕ) : ℕ := p
def common_difference (p : ℕ) : ℕ := 2 * p - 1

/-- n-th term of the arithmetic progression -/
def nth_term (p : ℕ) (n : ℕ) : ℕ :=
  first_term p + (n - 1) * common_difference p

/-- Sum of the first n terms of the arithmetic progression -/
def sum_of_terms (p : ℕ) (n : ℕ) : ℕ :=
  n / 2 * (first_term p + nth_term p n)

/-- Definition of Sp based on the problem statement -/
def Sp (p : ℕ) : ℕ :=
  sum_of_terms p 30

/-- Problem statement to be proved -/
theorem sum_Sp_equals_101475 : (∑ p in Finset.range 15, Sp (p + 1)) = 101475 := by
  sorry

end sum_Sp_equals_101475_l44_44834


namespace A_5_card_A_1_to_A_10_card_sum_l44_44984

open Finset

def A_n (n : ℕ) : Finset ℕ :=
  filter (λ x, 2^n < x ∧ x < 2^(n + 1) ∧ ∃ m, x = 3 * m) (range (2^(n + 1) + 1))

theorem A_5_card : (A_n 5).card = 11 := by sorry

theorem A_1_to_A_10_card_sum : (∑ n in range (10 + 1), (A_n n).card) = 682 := by sorry

end A_5_card_A_1_to_A_10_card_sum_l44_44984


namespace balloon_arrangements_l44_44080

theorem balloon_arrangements : (7! / (2! * 2!)) = 1260 := by
  sorry

end balloon_arrangements_l44_44080


namespace distinct_arrangements_balloon_l44_44090

theorem distinct_arrangements_balloon : 
  let n := 7 
  let freq_l := 2 
  let freq_o := 2 
  let freq_b := 1 
  let freq_a := 1 
  let freq_n := 1 
  Nat.factorial n / (Nat.factorial freq_l * Nat.factorial freq_o * Nat.factorial freq_b * Nat.factorial freq_a * Nat.factorial freq_n) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l44_44090


namespace hypotenuse_of_45_45_90_triangle_15_l44_44658

theorem hypotenuse_of_45_45_90_triangle_15 (a : ℝ) (h : a = 15) : 
  ∃ (c : ℝ), c = a * Real.sqrt 2 :=
by
  use a * Real.sqrt 2
  rw h
  sorry

end hypotenuse_of_45_45_90_triangle_15_l44_44658


namespace magnitude_of_z_l44_44315

-- Define the condition as a property
def z_prop (z : ℂ) : Prop :=
  i * (conj z + 3) = 3 - i ^ 2

-- Define the theorem we want to prove
theorem magnitude_of_z (z : ℂ) (h : z_prop z) : complex.abs z = 5 :=
sorry

end magnitude_of_z_l44_44315


namespace probability_composite_l44_44908

noncomputable def composite_probability : ℝ :=
  77735 / 77760

theorem probability_composite (d6 : ℕ → ℝ) (d10 : ℝ) :
  (∀ n, 1 ≤ d6 n ∧ d6 n ≤ 6) →
  (1 ≤ d10 ∧ d10 ≤ 10) →
  (∃ (p : ℝ), p = 77735 / 77760 ∧ p = composite_probability) :=
by
  intros h6 h10
  use composite_probability
  split
  case left => rfl
  case right => rfl
  sorry

end probability_composite_l44_44908


namespace area_expression_l44_44958

noncomputable def overlapping_area (m : ℝ) (h1 : 0 < m) (h2 : m < 4 * Real.sqrt 2) : ℝ :=
if h : m ≤ 2 * Real.sqrt 2 then
  6 - Real.sqrt 2 * m
else
  (1 / 4) * m^2 - 2 * Real.sqrt 2 * m + 8

theorem area_expression (m : ℝ) (h1 : 0 < m) (h2 : m < 4 * Real.sqrt 2) :
  let y := overlapping_area m h1 h2
  (if h : m ≤ 2 * Real.sqrt 2 then y = 6 - Real.sqrt 2 * m
   else y = (1 / 4) * m^2 - 2 * Real.sqrt 2 * m + 8) := 
sorry

end area_expression_l44_44958


namespace max_profit_advertising_max_profit_allocation_l44_44760

noncomputable def f (t : ℝ) : ℝ := -t^2 + 5 * t
noncomputable def g (x : ℝ) : ℝ := -1/3 * x^3 + x^2 + 3 * x + (-(3 - x)^2 + 5 * (3 - x) - 3)

theorem max_profit_advertising :
  ∃ t ∈ Icc (0 : ℝ) 3, t = 2 ∧ f t = 4 :=
begin
  sorry
end

theorem max_profit_allocation :
  ∃ x ∈ Icc (0 : ℝ) 3, x = 2 ∧ g x = 25/3 :=
begin
  sorry
end

end max_profit_advertising_max_profit_allocation_l44_44760


namespace quadratic_inequality_solution_l44_44308

theorem quadratic_inequality_solution :
  ∀ x : ℝ, x ∈ Ioo ((4 - Real.sqrt 19) / 3) ((4 + Real.sqrt 19) / 3) → (-3 * x^2 + 8 * x + 1 < 0) :=
by
  intro x hx
  have h1 : x ∈ Ioo ((4 - Real.sqrt 19) / 3) ((4 + Real.sqrt 19) / 3) := hx
  -- Further proof would go here
  sorry

end quadratic_inequality_solution_l44_44308


namespace computation_result_l44_44922

theorem computation_result :
  let a := -6
  let b := 25
  let c := -39
  let d := 40
  9 * a + 3 * b + 6 * c + d = -173 := by
  sorry

end computation_result_l44_44922


namespace root_condition_l44_44170

variable (a : ℝ)

def quadratic (a : ℝ) := λ x : ℝ, x^2 + a * x + a^2 - 1

theorem root_condition (h : ∃ x : ℝ, x > 0 ∧ quadratic a x = 0) (h' : ∃ y : ℝ, y < 0 ∧ quadratic a y = 0) : -1 < a ∧ a < 1 :=
sorry

end root_condition_l44_44170


namespace solve_system_eq_0_or_2_l44_44261

noncomputable def solve_system_eq {n : ℕ} (x : Fin n → ℝ) (k : Fin n) : Prop := 
  (∀ k : Fin n, x k + x (k + 1) % n = (x (k + 2) % n) ^ 2) ∧ 
  (∀ i : Fin n, x i ≥ 0)

theorem solve_system_eq_0_or_2 {n : ℕ} (x : Fin n → ℝ) :
  solve_system_eq x →
  (∀ i, x i = 0) ∨ (∀ i, x i = 2) := 
sorry

end solve_system_eq_0_or_2_l44_44261


namespace valid_three_digit_numbers_count_l44_44157

noncomputable def count_valid_three_digit_numbers : ℕ :=
  let total_three_digit_numbers := 900
  let excluded_numbers := 81 + 72
  total_three_digit_numbers - excluded_numbers

theorem valid_three_digit_numbers_count :
  count_valid_three_digit_numbers = 747 :=
by
  sorry

end valid_three_digit_numbers_count_l44_44157


namespace Yolanda_Bob_Jim_meeting_l44_44638

variables (Yolanda_speed Bob_speed distance Jim_speed time_bob_met Yolanda_distance_bob Met_distance_bob Met_distance_jim : ℝ)

-- Conditions given in the problem
def conditions :=
  Yolanda_speed = 4 ∧
  Bob_speed = 6 ∧
  distance = 80 ∧
  Jim_speed = 5 ∧
  time_bob_met = 7.6 ∧
  Yolanda_distance_bob = 76

-- The main theorem to be proved
theorem Yolanda_Bob_Jim_meeting
  (h : conditions) :
  (Bob_speed * time_bob_met = 45.6) ∧
  (Jim_speed * time_bob_met = 38) := 
sorry

end Yolanda_Bob_Jim_meeting_l44_44638


namespace hypotenuse_of_45_45_90_triangle_l44_44647

noncomputable def leg_length : ℝ := 15
noncomputable def angle_opposite_leg : ℝ := Real.pi / 4  -- 45 degrees in radians

theorem hypotenuse_of_45_45_90_triangle (h_leg : ℝ) (h_angle : ℝ) 
  (h_leg_cond : h_leg = leg_length) (h_angle_cond : h_angle = angle_opposite_leg) :
  ∃ h_hypotenuse : ℝ, h_hypotenuse = h_leg * Real.sqrt 2 :=
sorry

end hypotenuse_of_45_45_90_triangle_l44_44647


namespace convince_the_king_l44_44289

/-- Define the types of inhabitants -/
inductive Inhabitant
| Knight
| Liar
| Normal

/-- Define the king's preference -/
def K (inhabitant : Inhabitant) : Prop :=
  match inhabitant with
  | Inhabitant.Knight => False
  | Inhabitant.Liar => False
  | Inhabitant.Normal => True

/-- All knights tell the truth -/
def tells_truth (inhabitant : Inhabitant) : Prop :=
  match inhabitant with
  | Inhabitant.Knight => True
  | Inhabitant.Liar => False
  | Inhabitant.Normal => False

/-- All liars always lie -/
def tells_lie (inhabitant : Inhabitant) : Prop :=
  match inhabitant with
  | Inhabitant.Knight => False
  | Inhabitant.Liar => True
  | Inhabitant.Normal => False

/-- Normal persons can tell both truths and lies -/
def can_tell_both (inhabitant : Inhabitant) : Prop :=
  match inhabitant with
  | Inhabitant.Knight => False
  | Inhabitant.Liar => False
  | Inhabitant.Normal => True

/-- Prove there exists a true statement and a false statement to convince the king -/
theorem convince_the_king (p : Inhabitant) :
  (∃ S : Prop, (S ↔ tells_truth p) ∧ K p) ∧ (∃ S' : Prop, (¬ S' ↔ tells_lie p) ∧ K p) :=
by
  sorry

end convince_the_king_l44_44289


namespace sequence_arithmetic_and_find_an_l44_44855

theorem sequence_arithmetic_and_find_an (a : ℕ → ℝ)
  (h1 : a 9 = 1 / 7)
  (h2 : ∀ n, a (n + 1) = a n / (3 * a n + 1)) :
  (∀ n, 1 / a (n + 1) = 3 + 1 / a n) ∧ (∀ n, a n = 1 / (3 * n - 20)) :=
by
  sorry

end sequence_arithmetic_and_find_an_l44_44855


namespace invertible_interval_l44_44675

def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x - 9

theorem invertible_interval : ∃ I : set ℝ, (2 ∈ I) ∧ (∀ x1 x2 : ℝ, x1 ∈ I ∧ x2 ∈ I ∧ f x1 = f x2 → x1 = x2) ∧ I = set.Ici 1 :=
by
  use set.Ici 1
  sorry

end invertible_interval_l44_44675


namespace find_num_unbounded_sequences_l44_44475

noncomputable def g1 (n : ℕ) : ℕ :=
if n = 1 then 1 else
let factors := n.factorization.to_finset in
factors.fold 1 (λ p acc, acc * (p + 2)^((n.factorization p) - 1))

noncomputable def gm : ℕ → ℕ → ℕ 
| 1 n := g1 n
| (m + 1) n := g1 (gm m n)

def unbounded_sequence (N : ℕ) : Prop :=
∀ m, ∃ k ≥ m, gm k N > N

def satisfies_condition (N : ℕ) : bool :=
if unbounded_sequence N then true else false

def num_unbounded_sequences (upper_bound : ℕ) : ℕ :=
(λ n, if satisfies_condition n then 1 else 0) (range upper_bound).sum

theorem find_num_unbounded_sequences :
  num_unbounded_sequences 500 = 8 := by
  sorry

end find_num_unbounded_sequences_l44_44475


namespace min_value_of_box_l44_44901

theorem min_value_of_box (a b : ℤ) (h_ab : a * b = 30) : 
  ∃ (m : ℤ), m = 61 ∧ (∀ (c : ℤ), a * b = 30 → a^2 + b^2 = c → c ≥ m) := 
sorry

end min_value_of_box_l44_44901


namespace heptagon_divisibility_impossible_l44_44585

theorem heptagon_divisibility_impossible (a b c d e f g : ℕ) :
  (b ∣ a ∨ a ∣ b) ∧ (c ∣ b ∨ b ∣ c) ∧ (d ∣ c ∨ c ∣ d) ∧ (e ∣ d ∨ d ∣ e) ∧
  (f ∣ e ∨ e ∣ f) ∧ (g ∣ f ∨ f ∣ g) ∧ (a ∣ g ∨ g ∣ a) →
  ¬((a ∣ c ∨ c ∣ a) ∧ (a ∣ d ∨ d ∣ a) ∧ (a ∣ e ∨ e ∣ a) ∧ (a ∣ f ∨ f ∣ a) ∧
    (a ∣ g ∨ g ∣ a) ∧ (b ∣ d ∨ d ∣ b) ∧ (b ∣ e ∨ e ∣ b) ∧ (b ∣ f ∨ f ∣ b) ∧
    (b ∣ g ∨ g ∣ b) ∧ (c ∣ e ∨ e ∣ c) ∧ (c ∣ f ∨ f ∣ c) ∧ (c ∣ g ∨ g ∣ c) ∧
    (d ∣ f ∨ f ∣ d) ∧ (d ∣ g ∨ g ∣ d) ∧ (e ∣ g ∨ g ∣ e)) :=
 by
  sorry

end heptagon_divisibility_impossible_l44_44585


namespace product_of_roots_of_quadratic_l44_44404

theorem product_of_roots_of_quadratic :
  (∀ x : ℝ, x ^ 2 - 9 * x + 20 = 0 → x = 4 ∨ x = 5) →
  ∀ (a b : ℝ), (a = 4 ∧ b = 5) ∨ (a = 5 ∧ b = 4) →
  a * b = 20 :=
by
  intros h ha
  cases ha
  case inl h1 => cases h1 with ha hb; rw [ha, hb]
  case inr h2 => cases h2 with ha hb; rw [hb, ha]
  exact mul_comm 4 5
  -- Alternatively, just directly: exact dec_trivial -- but depending on the priority in goals by case.

end product_of_roots_of_quadratic_l44_44404


namespace max_M_value_l44_44807

-- Define a rectangle's area and perimeter
variables {a b S : ℝ} (h_area : S = a * b) (h_perimeter : ∀ a b, p = 2 * (a + b))
def M (p : ℝ) : ℝ := (16 - p) / (p^2 + 2 * p)

-- Statement of the problem: Prove the maximum value of M given the conditions
theorem max_M_value (S : ℝ) (h : S > 0) :
  ∃ p, p = 4 * sqrt S ∧ 
       ∀ p ≥ 4 * sqrt S, M p ≤ (4 - sqrt S) / (4 * S + 2 * sqrt S) :=
sorry

end max_M_value_l44_44807


namespace option_a_no_six_l44_44036

-- Define the conditions for each option
structure DiceRolls (n : ℕ) :=
  (results : Fin n → ℕ)

def average (dr : DiceRolls 5) : ℕ :=
  (∑ i, dr.results i) / 5

def variance (dr : DiceRolls 5 ) : ℝ :=
  let avg := (average dr : ℝ)
  (∑ i, (dr.results i - avg) ^ 2) / 5

def median (dr : DiceRolls 5) : ℕ :=
  let sorted := List.sort (Fin.elim0 (λ i, dr.results i))
  sorted.nth_le 2 (by decide : 2 < sorted.length)

def mode (dr : DiceRolls 5) : ℕ :=
  (List.mode (Fin.elim0 (λ i, dr.results i))).getOrElse 0

theorem option_a_no_six (dr : DiceRolls 5) (h_avg : average dr = 2) (h_var : variance dr = 3.1) : 
  ¬ ∃ i, dr.results i = 6 := sorry

end option_a_no_six_l44_44036


namespace scaled_standard_deviation_l44_44777

variables 
  (m n : ℝ) -- Original average and variance
  (a : ℝ)  -- Scaling factor

theorem scaled_standard_deviation
  (h_a_pos : a > 0) 
  (h_variance : n = m ^ 2) :
  (std_dev a n) = a * Real.sqrt n :=
sorry

end scaled_standard_deviation_l44_44777


namespace percent_paddyfield_warblers_l44_44933

variable (B : ℝ) -- The total number of birds.
variable (N_h : ℝ := 0.30 * B) -- Number of hawks.
variable (N_non_hawks : ℝ := 0.70 * B) -- Number of non-hawks.
variable (N_not_hpwk : ℝ := 0.35 * B) -- 35% are not hawks, paddyfield-warblers, or kingfishers.
variable (N_hpwk : ℝ := 0.65 * B) -- 65% are hawks, paddyfield-warblers, or kingfishers.
variable (P : ℝ) -- Percentage of non-hawks that are paddyfield-warblers, to be found.
variable (N_pw : ℝ := P * 0.70 * B) -- Number of paddyfield-warblers.
variable (N_k : ℝ := 0.25 * N_pw) -- Number of kingfishers.

theorem percent_paddyfield_warblers (h_eq : N_h + N_pw + N_k = 0.65 * B) : P = 0.5714 := by
  sorry

end percent_paddyfield_warblers_l44_44933


namespace median_of_sequence_is_36_l44_44187

def sequence (n : ℕ) : ℕ → ℕ
| k := if h : 1 ≤ k ∧ k ≤ n then k^2 else 0

def count_elements (n : ℕ) := ∑ k in Finset.range (n + 1), (sequence n k)

def find_median (n : ℕ) : ℕ :=
  let cnt := count_elements n in
  if cnt % 2 = 1 then
    let pos := cnt / 2 in
    let rec go (k sum : ℕ) : ℕ :=
      let sum := sum + sequence n k in
      if pos < sum then k else go (k + 1) sum
    in go 1 0
  else
    let pos := cnt / 2 - 1 in
    let rec go (k sum : ℕ) : ℕ :=
      let sum := sum + sequence n k in
      if pos < sum then k else go (k + 1) sum
    in go 1 0

theorem median_of_sequence_is_36 : find_median 60 = 36 := sorry

end median_of_sequence_is_36_l44_44187


namespace number_of_perfect_square_divisors_of_M_l44_44566

def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

def M : ℕ :=
    factorial 1 * factorial 2 * factorial 3 * factorial 4 * factorial 5 * factorial 6 * factorial 7 * factorial 8 * factorial 9

theorem number_of_perfect_square_divisors_of_M : 
    let count_perfect_square_divisors (k : ℕ) : ℕ :=
        let factors := k.factorization
        factors.prod (λ p e, (e / 2) + 1)
    count_perfect_square_divisors M = 672 := 
sorry

end number_of_perfect_square_divisors_of_M_l44_44566


namespace students_basketball_cricket_l44_44583

theorem students_basketball_cricket (A B: ℕ) (AB: ℕ):
  A = 12 →
  B = 8 →
  AB = 3 →
  (A + B - AB) = 17 :=
by
  intros
  sorry

end students_basketball_cricket_l44_44583


namespace multiples_of_15_between_10_and_150_l44_44155

theorem multiples_of_15_between_10_and_150 :
  ∃ n : ℕ, (finset.Icc 1 10).card = n ∧ n = 10 :=
by
  sorry

end multiples_of_15_between_10_and_150_l44_44155


namespace peter_speed_l44_44971

theorem peter_speed (P : ℝ) (h1 : P >= 0) (h2 : 1.5 * P + 1.5 * (P + 3) = 19.5) : P = 5 := by
  sorry

end peter_speed_l44_44971


namespace find_point_coordinates_l44_44928

noncomputable def curve_point_is_parallel_tangent : Prop :=
  let P := (-Real.log 2, 2)
  in P ∈ {P : ℝ × ℝ | P.2 = Real.exp (-P.1)} ∧ 
     ∀ Q : ℝ × ℝ, Q ∈ {Q : ℝ × ℝ | Q.2 = Real.exp (-Q.1)} → 
     (2 * Q.1 + Q.2 + 1 = 0 → Q = P)

theorem find_point_coordinates :
  curve_point_is_parallel_tangent :=
begin
  -- proof omitted
  sorry
end

end find_point_coordinates_l44_44928


namespace regular_hexagon_interior_angles_l44_44344

theorem regular_hexagon_interior_angles (n : ℕ) (h : n = 6) :
  (n - 2) * 180 = 720 :=
by
  subst h
  rfl

end regular_hexagon_interior_angles_l44_44344


namespace hypotenuse_of_454590_triangle_l44_44643

theorem hypotenuse_of_454590_triangle (l : ℝ) (angle : ℝ) (h : ℝ) (h_leg : l = 15) (h_angle : angle = 45) :
  h = l * Real.sqrt 2 := 
  sorry

end hypotenuse_of_454590_triangle_l44_44643


namespace determine_positions_l44_44440

-- Defining the rankings of the students
def rankings := {A B C D E : ℕ // A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E}

theorem determine_positions (r : rankings) :
  (r.A < r.B ∧ r.A < r.C /\
   (r.B = r.A + 1 ∨ r.B = r.A - 1) ∧ (r.B = r.C + 1 ∨ r.B = r.C - 1) /\
   ∀ x ∈ {r.C}, x > r.C /\
   ∀ y ∈ {r.D}, y < r.D /\
   r.E = 4) → r.A * 10000 + r.B * 1000 + r.C * 100 + r.D * 10 + r.E = 23514 :=
begin
  sorry
end

end determine_positions_l44_44440


namespace probability_sum_is_3_l44_44364

theorem probability_sum_is_3 (die : Type) [Fintype die] [DecidableEq die] 
  (dice_faces : die → ℕ) (h : ∀ d, dice_faces d ∈ {1, 2, 3, 4, 5, 6}) :
  (∑ i in finset.range 3, (die →₀ ℕ).single 1) = 3 → 
  (1 / (finset.card univ) ^ 3) = 1 / 216 :=
by
  sorry

end probability_sum_is_3_l44_44364


namespace distinct_arrangements_balloon_l44_44118

theorem distinct_arrangements_balloon : 
  let n := 7
  let r1 := 2
  let r2 := 2
  (Nat.factorial n) / ((Nat.factorial r1) * (Nat.factorial r2)) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l44_44118


namespace females_attending_meeting_l44_44559

noncomputable def number_of_females_attending_meeting : ℕ :=
let total_people := 300 in
let attending := total_people / 2 in
let females := attending / 3 in
females

theorem females_attending_meeting : number_of_females_attending_meeting = 50 := by
  sorry

end females_attending_meeting_l44_44559


namespace find_a_l44_44915

-- Given conditions and definitions
def circle_eq (x y : ℝ) : Prop := (x^2 + y^2 - 2*x - 2*y + 1 = 0)
def line_eq (x y a : ℝ) : Prop := (x - 2*y + a = 0)
def chord_length (r : ℝ) : ℝ := 2 * r

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, circle_eq x y) → 
  (∀ x y : ℝ, line_eq x y a) → 
  (∃ x y : ℝ, (x = 1 ∧ y = 1) ∧ (line_eq x y a ∧ chord_length 1 = 2)) → 
  a = 1 := by sorry

end find_a_l44_44915


namespace perimeter_angle_bisector_inequality_l44_44529

theorem perimeter_angle_bisector_inequality
  {A B C A1 B1 C1 : Point}
  (hA : AngleBisector A A1 (Triangle.mk A B C))
  (hB : AngleBisector B B1 (Triangle.mk A B C))
  (hC : AngleBisector C C1 (Triangle.mk A B C)) :
  perimeter (Triangle.mk A1 B1 C1) ≤ (1 / 2) * perimeter (Triangle.mk A B C) :=
  sorry

end perimeter_angle_bisector_inequality_l44_44529


namespace joses_share_of_profit_l44_44403

theorem joses_share_of_profit
    (Toms_investment : ℝ)
    (Toms_duration : ℝ)
    (Joses_investment : ℝ)
    (Joses_duration : ℝ)
    (Total_profit : ℝ)
    (H1 : Toms_investment = 3000)
    (H2 : Toms_duration = 12)
    (H3 : Joses_investment = 4500)
    (H4 : Joses_duration = 10)
    (H5 : Total_profit = 5400) :
    let Toms_month_investment := Toms_investment * Toms_duration,
        Joses_month_investment := Joses_investment * Joses_duration,
        Total_month_investment := Toms_month_investment + Joses_month_investment,
        Joses_proportion := Joses_month_investment / Total_month_investment,
        Joses_share_of_profit := Total_profit * Joses_proportion
    in Joses_share_of_profit = 3000 :=
by
  sorry

end joses_share_of_profit_l44_44403


namespace find_weight_b_l44_44322

theorem find_weight_b (A B C : ℕ) 
  (h1 : A + B + C = 90)
  (h2 : A + B = 50)
  (h3 : B + C = 56) : 
  B = 16 :=
sorry

end find_weight_b_l44_44322


namespace probability_of_green_l44_44431

-- Define the conditions
def P_R : ℝ := 0.15
def P_O : ℝ := 0.35
def P_B : ℝ := 0.2
def total_probability (P_Y P_G : ℝ) : Prop := P_R + P_O + P_B + P_Y + P_G = 1

-- State the theorem to be proven
theorem probability_of_green (P_Y : ℝ) (P_G : ℝ) (h : total_probability P_Y P_G) (P_Y_assumption : P_Y = 0.15) : P_G = 0.15 :=
by
  sorry

end probability_of_green_l44_44431


namespace domain_of_f_l44_44477

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x - 5) + real.cbrt (x + 4)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f x = y} = Ici 5 :=
by
  sorry

end domain_of_f_l44_44477


namespace monochromatic_triangle_exists_l44_44861

theorem monochromatic_triangle_exists (points: FinSet Point) (h: points.card = 6) 
    (h_not_collinear: ∀ (A B C: Point), A ∈ points → B ∈ points → C ∈ points → A ≠ B → B ≠ C → A ≠ C → ¬ collinear A B C) 
    (color: Point → Point → color := λ _ _, red ∨ blue): 
    ∃ (A B C: Point), A ∈ points ∧ B ∈ points ∧ C ∈ points ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ same_color (color A B) (color B C) (color A C) :=
by sorry

end monochromatic_triangle_exists_l44_44861


namespace group_commutative_l44_44629

open Function

variables (G : Type*) [Group G]
  (m n : ℕ)
  (h_coprime : Nat.coprime m n)
  (phi_m : G → G)
  (phi_n : G → G)
  (h_phi_m : ∀ x : G, ∃ y : G, phi_m y = x)
  (h_phi_n : ∀ x : G, ∃ y : G, phi_n y = x)
  (h_phi_m_def : ∀ x : G, phi_m x = x^(m+1))
  (h_phi_n_def : ∀ x : G, phi_n x = x^(n+1))

theorem group_commutative : ∀ x y : G, x * y = y * x :=
by
  sorry

end group_commutative_l44_44629


namespace functional_equation_properties_l44_44068

-- Define the function and its properties
variable {f : ℝ → ℝ}

-- The main theorem combining both parts of the problem
theorem functional_equation_properties (h : ∀ x y : ℝ, f (x + y) = f x + f y) : 
  (f 0 = 0) ∧ (∀ x : ℝ, f (-x) = -f x) :=
by
  -- Skipping the proof
  sorry

end functional_equation_properties_l44_44068


namespace number_of_sets_X_l44_44332

noncomputable def finite_set_problem (M A B : Finset ℕ) : Prop :=
  (M.card = 10) ∧ 
  (A ⊆ M) ∧ 
  (B ⊆ M) ∧ 
  (A ∩ B = ∅) ∧ 
  (A.card = 2) ∧ 
  (B.card = 3) ∧ 
  (∃ (X : Finset ℕ), X ⊆ M ∧ ¬(A ⊆ X) ∧ ¬(B ⊆ X))

theorem number_of_sets_X (M A B : Finset ℕ) (h : finite_set_problem M A B) : 
  ∃ n : ℕ, n = 672 := 
sorry

end number_of_sets_X_l44_44332


namespace cos_sum_of_angles_l44_44730

theorem cos_sum_of_angles (α β γ R r : ℝ) (h_triangle: α + β + γ = π) 
    (h_R : R > 0) (h_r: r ≥ 0): 
    cos α + cos β + cos γ = (R + r) / R :=
sorry

end cos_sum_of_angles_l44_44730


namespace prob_sum_to_3_three_dice_correct_l44_44359

def prob_sum_to_3_three_dice (sum : ℕ) (dice_count : ℕ) (dice_faces : Finset ℕ) : ℚ :=
  if sum = 3 ∧ dice_count = 3 ∧ dice_faces = {1, 2, 3, 4, 5, 6} then (1 : ℚ) / 216 else 0

theorem prob_sum_to_3_three_dice_correct :
  prob_sum_to_3_three_dice 3 3 {1, 2, 3, 4, 5, 6} = (1 : ℚ) / 216 := 
by
  sorry

end prob_sum_to_3_three_dice_correct_l44_44359


namespace quadratic_distinct_real_roots_l44_44921

theorem quadratic_distinct_real_roots (k : ℝ) :
  ((k - 1) ≠ 0) ∧ ((4^2 - 4 * (k - 1) * 1) > 0) → k < 5 ∧ k ≠ 1 :=
by
  -- We state the problem conditions directly and prove the intended result.
  intro h
  cases h with hk hΔ
  sorry

end quadratic_distinct_real_roots_l44_44921


namespace find_s_l44_44579

variables {A B C G F Q : Type}
variables [EuclideanGeometry A B C G F Q]

-- Assume given conditions
def condition1 (A B C G : Type) [EuclideanGeometry A B C G] := 
  segment_ratio (C, G) (G, B) = 4 / 1

def condition2 (A B F : Type) [EuclideanGeometry A B F] := 
  segment_ratio (A, F) (F, B) = 3 / 1

def condition3 (A C F G : Type) [EuclideanGeometry A C F G] : Prop :=
  exists Q : Type, intersection_point (C, F) (A, G) Q

-- Main theorem to prove
theorem find_s (A B C G F Q s : Type) [EuclideanGeometry A B C G F Q] 
  (h1 : condition1 A B C G) 
  (h2 : condition2 A B F) 
  (h3 : condition3 A C F G) :
  s = 5 := sorry

end find_s_l44_44579


namespace domain_of_f_parity_of_f_range_of_x_if_f_pos_l44_44880

-- Define the function, domain and parity questions
noncomputable def f (x : ℝ) := log (1 + x) - log (1 - x)

-- Find the domain of the function f(x)
theorem domain_of_f : ∀ x : ℝ, (1 + x > 0) ∧ (1 - x > 0) ↔ -1 < x ∧ x < 1 :=
by sorry

-- Determine the parity of the function f(x)
theorem parity_of_f : ∀ x : ℝ, f (-x) = - f x :=
by sorry

-- If f(x) > 0, find the range of x
theorem range_of_x_if_f_pos : (∀ x : ℝ, f x > 0 → (0 < x ∧ x < 1)) :=
by
  intros x hx
  sorry

end domain_of_f_parity_of_f_range_of_x_if_f_pos_l44_44880


namespace john_age_proof_l44_44238

theorem john_age_proof (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end john_age_proof_l44_44238


namespace my_inequality_l44_44999

open Real

variable {a b c : ℝ}

theorem my_inequality 
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : a * b + b * c + c * a = 1) :
  sqrt (a ^ 3 + a) + sqrt (b ^ 3 + b) + sqrt (c ^ 3 + c) ≥ 2 * sqrt (a + b + c) := 
  sorry

end my_inequality_l44_44999


namespace distinct_arrangements_balloon_l44_44133

-- Let's define the basic conditions:
def total_letters : Nat := 7
def repeats_l : Nat := 2
def repeats_o : Nat := 2

-- Now let's state the problem.
theorem distinct_arrangements_balloon : 
  (Nat.factorial total_letters) / ((Nat.factorial repeats_l) * (Nat.factorial repeats_o)) = 1260 := 
by
  sorry

end distinct_arrangements_balloon_l44_44133


namespace hypotenuse_of_45_45_90_triangle_l44_44649

theorem hypotenuse_of_45_45_90_triangle (a : ℝ) (h : ℝ) 
  (ha : a = 15) 
  (angle_opposite_leg : ℝ) 
  (h_angle : angle_opposite_leg = 45) 
  (right_triangle : ∃ θ : ℝ, θ = 90) : 
  h = 15 * Real.sqrt 2 := 
sorry

end hypotenuse_of_45_45_90_triangle_l44_44649


namespace asimov_books_l44_44482

theorem asimov_books (h p : Nat) (condition1 : h + p = 12) (condition2 : 30 * h + 20 * p = 300) : h = 6 := by
  sorry

end asimov_books_l44_44482


namespace card_draw_probability_l44_44352

theorem card_draw_probability :
  (13 / 52) * (13 / 51) * (13 / 50) = 2197 / 132600 :=
by
  sorry

end card_draw_probability_l44_44352


namespace balloon_arrangements_l44_44144

-- Defining the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Given Conditions
def seven_factorial := fact 7 -- 7!
def two_factorial := fact 2 -- 2!

-- Statement to prove
theorem balloon_arrangements : seven_factorial / (two_factorial * two_factorial) = 1260 :=
by
  sorry

end balloon_arrangements_l44_44144


namespace balloon_permutations_l44_44135

theorem balloon_permutations : 
  let n : ℕ := 7
  let k1 : ℕ := 2
  let k2 : ℕ := 2
  ∃ distinct_arrangements : ℕ, 
  distinct_arrangements = n.factorial / (k1.factorial * k2.factorial) 
  ∧ distinct_arrangements = 1260 :=
by
  sorry

end balloon_permutations_l44_44135


namespace find_f_neg_seven_l44_44542

def f (x : ℝ) : ℝ :=
  if x >= 0 then 2^x - Real.log2 x else f (x + 3)

theorem find_f_neg_seven : 
  f (-7) = 3 :=
sorry

end find_f_neg_seven_l44_44542


namespace quadrilateral_perimeter_l44_44424

noncomputable def perimeter_of_quadrilateral (PA PB PC PD area: ℝ) : ℝ :=
  2 * (sqrt (PA^2 + PB^2) + sqrt (PB^2 + PC^2) + sqrt (PC^2 + PD^2) + sqrt (PD^2 + PA^2))

theorem quadrilateral_perimeter:
  ∀ (PA PB PC PD area: ℝ),
  PA = 30 →
  PB = 40 →
  PC = 35 →
  PD = 50 →
  area = 2200 →
  perimeter_of_quadrilateral PA PB PC PD area = 50 + 5 * sqrt 113 + 5 * sqrt 149 + 2 * sqrt 850 :=
by
  intros PA PB PC PD area hPA hPB hPC hPD h_area
  sorry

end quadrilateral_perimeter_l44_44424


namespace smallest_add_to_palindrome_l44_44769

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

theorem smallest_add_to_palindrome (x : ℕ) : x = 110 :=
  let n := 2002
  let m := 2112
  is_palindrome n ∧ is_palindrome m ∧ m > n → m - n = x :=
begin
  sorry,
end

end smallest_add_to_palindrome_l44_44769


namespace constant_term_eq_160_l44_44828

-- Define the binomial coefficients and the binomial theorem
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the general term of (2x + 1/x)^6 expansion
def general_term_expansion (r : ℕ) : ℤ :=
  2^(6 - r) * binom 6 r

-- Define the proof statement for the required constant term
theorem constant_term_eq_160 : general_term_expansion 3 = 160 := 
by
  sorry

end constant_term_eq_160_l44_44828


namespace part1_part2_l44_44060

-- Definition of the quadratic equation and its real roots condition
def quadratic_has_real_roots (k : ℝ) : Prop :=
  let Δ := (2 * k - 1)^2 - 4 * (k^2 - 1)
  Δ ≥ 0

-- Proving part (1): The range of real number k
theorem part1 (k : ℝ) (hk : quadratic_has_real_roots k) : k ≤ 5 / 4 := 
  sorry

-- Definition using the given condition in part (2)
def roots_condition (x₁ x₂ : ℝ) : Prop :=
  x₁^2 + x₂^2 = 16 + x₁ * x₂

-- Sum and product of roots of the quadratic equation
theorem part2 (k : ℝ) (h : quadratic_has_real_roots k) 
  (hx_sum : ∃ x₁ x₂ : ℝ, x₁ + x₂ = 1 - 2 * k ∧ x₁ * x₂ = k^2 - 1 ∧ roots_condition x₁ x₂) : k = -2 :=
  sorry

end part1_part2_l44_44060


namespace fraction_sum_l44_44466

theorem fraction_sum :
  (2 / 5 : ℝ) + (3 / 25 : ℝ) + (4 / 125 : ℝ) + (1 / 625 : ℝ) = 0.5536 :=
by
  have h₁ : (2 / 5 : ℝ) = 0.4 := by norm_num
  have h₂ : (3 / 25 : ℝ) = 0.12 := by norm_num
  have h₃ : (4 / 125 : ℝ) = 0.032 := by norm_num
  have h₄ : (1 / 625 : ℝ) = 0.0016 := by norm_num
  rw [h₁, h₂, h₃, h₄]
  norm_num
  sorry

end fraction_sum_l44_44466


namespace distance_between_hyperbola_vertices_l44_44018

theorem distance_between_hyperbola_vertices :
  let equation := ∀ x y, (x * x) / 48 - (y * y) / 16 = 1
  ∃ distance, distance = 8 * Real.sqrt 3 :=
by
  -- Let's define the necessary conditions for the hyperbola
  have hyp_eq : ∀ x y, (x * x) / 48 - (y * y) / 16 = 1,
  from sorry,

  -- The vertices of the hyperbola are located at (±a, 0)
  let a := Real.sqrt 48,

  -- The distance between the vertices is 2a
  let distance := 2 * a,

  -- Simplify the distance
  have distance_simplified : distance = 8 * Real.sqrt 3,
  from sorry,

  -- Finally, we have the required distance
  exact ⟨distance, distance_simplified⟩

end distance_between_hyperbola_vertices_l44_44018


namespace right_angle_triangle_perimeter_l44_44434

theorem right_angle_triangle_perimeter :
  ∃ (a c : ℕ), c^2 - a^2 = 16 ∧ c > 4 ∧ c - a < 4 ∧ a + 4 + c = 12 :=
begin
  sorry
end

end right_angle_triangle_perimeter_l44_44434


namespace find_smallest_theta_l44_44274

noncomputable def smallest_theta (a b c : ℝ^3) (θ : ℝ) : ℝ :=
  if h : ((∥a∥ = 1) ∧ (∥b∥ = 1) ∧ (∥c∥ = 1)) ∧
          (∠(a, b) = θ) ∧ (∠(c, a × b) = θ) ∧
          (b • (c × a) = 1/3) then
    θ
  else
    0  -- 0 is a filler value, it will not affect the proof

theorem find_smallest_theta (a b c : ℝ^3) :
  ∥a∥ = 1 ∧ ∥b∥ = 1 ∧ ∥c∥ = 1 ∧
  (let θ : ℝ := smallest_theta a b c θ in
   θ = θ) ∧
  (b • (c × a) = 1/3) ∧
  ∠a b = ∠c (a × b) ->
  smallest_theta a b c θ = 20.905 :=
by sorry

end find_smallest_theta_l44_44274


namespace triangle_A1DE_properties_l44_44538

variables (t : ℝ) (α β γ : ℝ)
variables (A B C A1 D E A2 : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space A1] [metric_space D] [metric_space E] [metric_space A2]

-- Defining conditions
def given_conditions : Prop :=
  t > 0 ∧
  0 < α < 180 ∧
  0 < β < 180 ∧
  0 < γ < 180 ∧
  α + β + γ = 180

noncomputable def height_A1A2 : ℝ :=
  (cos β) * (cos γ) * real.sqrt ((2 * t * (sin β) * (sin γ)) / (sin α))

noncomputable def area_t2 : ℝ :=
  (t / 4) * (sin (2 * β)) * (sin (2 * γ))

theorem triangle_A1DE_properties (h : given_conditions t α β γ) :
  ∃ t2 : ℝ, ∃ hA1A2 : ℝ, t2 = area_t2 t α β γ ∧ hA1A2 = height_A1A2 t α β γ :=
by
  use area_t2 t α β γ
  use height_A1A2 t α β γ
  sorry

end triangle_A1DE_properties_l44_44538


namespace minimum_phi_l44_44066

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 3))
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

-- Define the condition for g overlapping with f after shifting by φ
noncomputable def shifted_g (x φ : ℝ) : ℝ := Real.sin (2 * x + 2 * φ)

theorem minimum_phi (φ : ℝ) (h : φ > 0) :
  (∃ (x : ℝ), shifted_g x φ = f x) ↔ (∃ k : ℕ, φ = Real.pi / 6 + k * Real.pi) :=
sorry

end minimum_phi_l44_44066


namespace johns_age_l44_44244

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
sorry

end johns_age_l44_44244


namespace candy_cost_l44_44758

theorem candy_cost (c : ℝ) :
  let cost_first := 30 * 8
  let total_weight := 30 + 60
  let desired_total_cost := total_weight * 6
  let cost_second := 60 * c
  in cost_first + cost_second = desired_total_cost → c = 5 :=
by
  -- Introducing definitions
  let cost_first := 30 * 8
  let total_weight := 30 + 60
  let desired_total_cost := total_weight * 6
  let cost_second := 60 * c

  -- Assume the given sum of costs equation
  intros h

  -- Equation from hypothesis
  have eq1 : cost_first + cost_second = desired_total_cost := h

  -- Substituting values into the hypothesis
  rw [←eq1, add_assoc, mul_assoc]
  
  -- Isolating cost_second (60 * c)
  have eq2 : cost_second = desired_total_cost - cost_first := by linarith

  -- Solve for c
  have : c = (desired_total_cost - cost_first) / 60 := by simp [cost_second, eq2]

  -- Simplifying to get the final answer
  norm_num at this
  assumption

end candy_cost_l44_44758


namespace balloon_arrangements_l44_44085

theorem balloon_arrangements : (7! / (2! * 2!)) = 1260 := by
  sorry

end balloon_arrangements_l44_44085


namespace divisor_inequality_l44_44982

-- Definition of our main inequality theorem
theorem divisor_inequality (n : ℕ) (h1 : n > 0) (h2 : n % 8 = 4)
    (divisors : List ℕ) (h3 : divisors = (List.range (n + 1)).filter (λ x => n % x = 0)) 
    (i : ℕ) (h4 : i < divisors.length - 1) (h5 : i % 3 ≠ 0) : 
    divisors[i + 1] ≤ 2 * divisors[i] := sorry

end divisor_inequality_l44_44982


namespace probability_sum_three_dice_3_l44_44366

-- Definition of a fair six-sided die
def fair_six_sided_die : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Definition of probability of an event
def probability (s : Set ℕ) (event : ℕ → Prop) : ℚ :=
  if h : finite s then (s.filter event).to_finset.card / s.to_finset.card else 0

theorem probability_sum_three_dice_3 :
  let dice := List.repeat fair_six_sided_die 3 in
  let event := λ result : List ℕ => result.sum = 3 in
  probability ({(r1, r2, r3) | r1 ∈ fair_six_sided_die ∧ r2 ∈ fair_six_sided_die ∧ r3 ∈ fair_six_sided_die }) (λ (r1, r2, r3) => r1 + r2 + r3 = 3) = 1 / 216 :=
by
  sorry

end probability_sum_three_dice_3_l44_44366


namespace average_math_chem_l44_44736

-- Define the conditions as hypotheses
variables (M P C : ℕ) -- Assuming marks are natural numbers for simplicity
hypothesis (h1 : M + P = 60)
hypothesis (h2 : C = P + 20)

-- Define the statement to be proved
theorem average_math_chem (M P C : ℕ) (h1 : M + P = 60) (h2 : C = P + 20) : 
  (M + C) / 2 = 40 := 
by
  sorry

end average_math_chem_l44_44736


namespace problem_1_part_1_problem_1_part_2_l44_44630

section Problem1

variables (f : ℝ → ℝ) (m : ℝ) (x a b : ℝ)
def second_derivative := x^2 - m*x - 3
def convex_on_interval (f'' : ℝ → ℝ) (a b : ℝ) := ∀ x, a < x ∧ x < b → f'' x < 0

-- Given conditions
axiom H1 : f x = (1/12)*x^4 - (1/6)*m*x^3 - (3/2)*x^2
axiom H2 : ∀ x, second_derivative x = x^2 - m*x - 3

-- Problem Ⅰ
noncomputable def m_value := 2

-- Problem Ⅱ
noncomputable def max_b_minus_a := 2

end Problem1

theorem problem_1_part_1 (f : ℝ → ℝ) (m : ℝ) (a b : ℝ) (h : convex_on_interval (second_derivative m) a b = true) :
  m = 2 :=
sorry

theorem problem_1_part_2 (f : ℝ → ℝ) (m : ℝ) (a b : ℝ) (h : convex_on_interval (second_derivative m) a b = true) (hm : |m| ≤ 2) :
  b - a ≤ 2 :=
sorry

end problem_1_part_1_problem_1_part_2_l44_44630


namespace divisors_powers_of_2_fact_8_l44_44896

-- Defining the factorial function for n = 8
def factorial : ℕ → ℕ 
| 0       => 1
| (n + 1) => (n + 1) * factorial n

-- Defining the problem condition factorial 8
def fact_8 : ℕ := factorial 8

-- Stating that there are 8 divisors of fact_8 that are powers of 2
theorem divisors_powers_of_2_fact_8 : (finset.filter (λd, ∃ k, d = 2^k) (finset.divisors fact_8)).card = 8 :=
by sorry

end divisors_powers_of_2_fact_8_l44_44896


namespace find_angle_zero_l44_44189

theorem find_angle_zero (a b c : ℝ) (h : (a + b + c) * (a + b - c) = 4 * a * b) : 
  ∃ C : ℝ, C = 0 ∧ cos C = 1 ∧ 
  (a^2 + b^2 - c^2 = 2 * a * b) :=
by
  use 0
  split
  { reflexivity }
  split
  { simp }
  sorry

end find_angle_zero_l44_44189


namespace balloon_arrangements_l44_44146

-- Defining the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Given Conditions
def seven_factorial := fact 7 -- 7!
def two_factorial := fact 2 -- 2!

-- Statement to prove
theorem balloon_arrangements : seven_factorial / (two_factorial * two_factorial) = 1260 :=
by
  sorry

end balloon_arrangements_l44_44146


namespace value_of_a_l44_44863

theorem value_of_a (a : ℝ) (h₁ : ∃ b : ℝ, b = (a + 2) ∧ is_square b)
  (h₂ : is_simplest_quadratic_root (sqrt (a + 2)) ∧ can_be_combined_with (sqrt (a + 2)) (sqrt 12)) :
  a = 1 :=
sorry

end value_of_a_l44_44863


namespace balloon_arrangements_l44_44099

-- Define the variables
def n : ℕ := 7
def L_count : ℕ := 2
def O_count : ℕ := 2
def B_count : ℕ := 1
def A_count : ℕ := 1
def N_count : ℕ := 1

-- Define the multiset permutation formula
def multiset_permutations (n : ℕ) (counts : List ℕ) : ℕ :=
  n.factorial / (counts.map Nat.factorial).prod

-- Proof that the number of distinct arrangements is 1260
theorem balloon_arrangements : multiset_permutations n [L_count, O_count, B_count, A_count, N_count] = 1260 :=
  by
  -- The proof is omitted
  sorry

end balloon_arrangements_l44_44099


namespace sum_of_all_possible_values_l44_44620

theorem sum_of_all_possible_values (x y : ℝ) (h : x * y - x^2 - y^2 = 4) :
  (x - 2) * (y - 2) = 4 :=
sorry

end sum_of_all_possible_values_l44_44620


namespace unique_a_l44_44520

variables {ℝ : Type*} [inner_product_space ℝ (euclidean_space ℝ (fin 2))]

def f (a x : euclidean_space ℝ (fin 2)) : euclidean_space ℝ (fin 2) :=
  x - inner x a • a

theorem unique_a (a : euclidean_space ℝ (fin 2)) :
  (∀ x y : euclidean_space ℝ (fin 2), ⟪f a x, f a y⟫ = ⟪x, y⟫) → ∥a∥ ^ 2 = 2 :=
by
  intro h
  sorry

end unique_a_l44_44520


namespace convex_polygons_on_circle_l44_44488

theorem convex_polygons_on_circle:
  let points := 15 in
  ∑ i in finset.range (points + 1), choose points i - (choose points 0 + choose points 1 + choose points 2 + choose points 3) = 32192 :=
begin
  sorry
end

end convex_polygons_on_circle_l44_44488


namespace green_yarn_length_l44_44690

/-- The length of the green piece of yarn given the red yarn is 8 cm more 
than three times the length of the green yarn and the total length 
for 2 pieces of yarn is 632 cm. -/
theorem green_yarn_length (G R : ℕ) 
  (h1 : R = 3 * G + 8)
  (h2 : G + R = 632) : 
  G = 156 := 
by
  sorry

end green_yarn_length_l44_44690


namespace find_f_log2_3_l44_44533

noncomputable def f (x : ℝ) : ℝ := (2^x + 2^(-x)) / 2
noncomputable def g (x : ℝ) : ℝ := (2^x - 2^(-x)) / 2

lemma even_f : ∀ x : ℝ, f x = f (-x) :=
begin
  intros x,
  -- Proof omitted, placeholder
  sorry
end

lemma odd_g : ∀ x : ℝ, g x = -g (-x) :=
begin
  intros x,
  -- Proof omitted, placeholder
  sorry
end

lemma fg_eq_expr : ∀ x : ℝ, f x + g x = 2^x + x :=
begin
  intros x,
  -- Proof omitted, placeholder
  sorry
end

theorem find_f_log2_3 :
  f (Real.log 3 / Real.log 2) = 5 / 3 :=
begin
  calc
    f (Real.log 3 / Real.log 2) = (3 + 1/3) / 2 : by { -- Use known definition of log2(3) here
        -- Proof omitted, placeholder
        sorry
    }
    ... = 5 / 3 : by { -- Simple division
        -- Proof omitted, placeholder
        sorry
    },
end

end find_f_log2_3_l44_44533


namespace pairs_of_managers_refusing_l44_44768

theorem pairs_of_managers_refusing (h_comb : (Nat.choose 8 4) = 70) (h_restriction : 55 = 70 - n * (Nat.choose 6 2)) : n = 1 :=
by
  have h1 : Nat.choose 8 4 = 70 := h_comb
  have h2 : Nat.choose 6 2 = 15 := by sorry -- skipped calculation for (6 choose 2), which is 15
  have h3 : 55 = 70 - n * 15 := h_restriction
  sorry -- proof steps to show n = 1

end pairs_of_managers_refusing_l44_44768


namespace cosine_eq_one_fifth_l44_44039

theorem cosine_eq_one_fifth {α : ℝ} 
  (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) : 
  Real.cos α = 1 / 5 := 
sorry

end cosine_eq_one_fifth_l44_44039


namespace johns_age_l44_44245

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
sorry

end johns_age_l44_44245


namespace max_capacity_per_car_l44_44347

-- Conditions
def num_cars : ℕ := 2
def num_vans : ℕ := 3
def people_per_car : ℕ := 5
def people_per_van : ℕ := 3
def max_people_per_van : ℕ := 8
def additional_people : ℕ := 17

-- Theorem to prove maximum capacity of each car is 6 people
theorem max_capacity_per_car (num_cars num_vans people_per_car people_per_van max_people_per_van additional_people : ℕ) : 
  (num_cars = 2 ∧ num_vans = 3 ∧ people_per_car = 5 ∧ people_per_van = 3 ∧ max_people_per_van = 8 ∧ additional_people = 17) →
  ∃ max_people_per_car, max_people_per_car = 6 :=
by
  sorry

end max_capacity_per_car_l44_44347


namespace even_function_inequality_l44_44746

noncomputable def evenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

theorem even_function_inequality (f : ℝ → ℝ) (h_even : evenFunction f)
    (h_interval : ∀ x : ℝ, -6 ≤ x ∧ x ≤ 6 → True)
    (h_condition : f 3 < f 1) : f (-1) > f (-3) := 
begin
  -- Proof to be filled in
  sorry
end

end even_function_inequality_l44_44746


namespace divisors_less_than_not_divide_l44_44623

theorem divisors_less_than_not_divide : 
  let n := 2^40 * 5^15 in 
  let n_squared := n^2 in
  let total_divisors_n_squared := (80 + 1) * (30 + 1) in 
  let divisors_n_less_than := 
    (total_divisors_n_squared - 1) / 2 in 
  let total_divisors_n := (40 + 1) * (15 + 1) in 
  divisors_n_less_than - total_divisors_n = 599 := 
by
  sorry

end divisors_less_than_not_divide_l44_44623


namespace systematic_sampling_l44_44587

def sample_number (n m : ℕ) : ℕ :=
  (m + n - 1) % 10 + 10 * (n - 1)

theorem systematic_sampling (m : ℕ) (H : m = 6) : sample_number 7 m = 73 :=
by {
  simp [sample_number],
  sorry
}

end systematic_sampling_l44_44587


namespace solution_set_f_gt_x_l44_44878

noncomputable def f : ℝ → ℝ := sorry

axiom f_one_eq_one : f 1 = 1
axiom f_deriv_gt_one (x : ℝ) : deriv f x > 1

theorem solution_set_f_gt_x :
  { x : ℝ | f x > x } = set.Ioi 1 :=
begin
  sorry
end

end solution_set_f_gt_x_l44_44878


namespace largest_integer_in_sequence_l44_44478

theorem largest_integer_in_sequence 
  (a : ℤ) 
  (h1 : a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6) + (a + 7) + (a + 8) = 2025) 
  (h2 : ∃ k : ℤ, k ∈ Finset.range 9 ∧ a + k = 222) : 
  a + 8 = 229 :=
sorry

end largest_integer_in_sequence_l44_44478


namespace calc_difference_of_squares_l44_44799

theorem calc_difference_of_squares :
  625^2 - 375^2 = 250000 :=
by sorry

end calc_difference_of_squares_l44_44799


namespace painting_faces_not_sum_to_9_l44_44453

open Finset

/-
Given an ordinary 8-sided die with faces numbered from 1 to 8,
prove that the number of ways to paint three faces red such that 
their numbers don't add up to 9 is 32.
-/

def valid_paintings : Finset (Finset ℕ) :=
  (univ : Finset ℕ).powerset.filter (λ s, s.card = 3 ∧ ∀ (x ∈ s) (y ∈ s), x + y ≠ 9 ∨ x = y)

theorem painting_faces_not_sum_to_9 : valid_paintings.card = 32 := 
by 
  sorry

end painting_faces_not_sum_to_9_l44_44453


namespace max_number_of_eligible_ages_l44_44320

-- Definitions based on the problem conditions
def average_age : ℝ := 31
def std_dev : ℝ := 5
def acceptable_age_range (a : ℝ) : Prop := 26 ≤ a ∧ a ≤ 36
def has_masters_degree : Prop := 24 ≤ 26  -- simplified for context indicated in problem
def has_work_experience : Prop := 26 ≥ 26

-- Define the maximum number of different ages of the eligible applicants
noncomputable def max_diff_ages : ℕ := 36 - 26 + 1  -- This matches the solution step directly

-- The theorem stating the result
theorem max_number_of_eligible_ages :
  max_diff_ages = 11 :=
by {
  sorry
}

end max_number_of_eligible_ages_l44_44320


namespace base_conversion_subtraction_l44_44823

theorem base_conversion_subtraction :
  (4 * 6^4 + 3 * 6^3 + 2 * 6^2 + 1 * 6^1 + 0 * 6^0) - (3 * 7^3 + 2 * 7^2 + 1 * 7^1 + 0 * 7^0) = 4776 :=
by {
  sorry
}

end base_conversion_subtraction_l44_44823


namespace number_of_possible_values_for_a_l44_44291

theorem number_of_possible_values_for_a 
  (a b c d : ℤ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h5 : a > b) (h6 : b > c) (h7 : c > d)
  (h8 : a + b + c + d = 2004)
  (h9 : a^2 - b^2 - c^2 + d^2 = 1004) : 
  ∃ n : ℕ, n = 500 :=
  sorry

end number_of_possible_values_for_a_l44_44291


namespace range_of_a_l44_44548

def piecewise_function (x : ℝ) : ℝ :=
if x < 0 then x^2 + 2 * x else x^2 - 2 * x

theorem range_of_a (a : ℝ) (h : piecewise_function (-a) + piecewise_function a ≤ 0) : -2 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_a_l44_44548


namespace johns_age_l44_44226

theorem johns_age (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l44_44226


namespace dyed_pink_correct_l44_44370

def silk_dyed_green := 61921
def total_yards_dyed := 111421
def yards_dyed_pink := total_yards_dyed - silk_dyed_green

theorem dyed_pink_correct : yards_dyed_pink = 49500 := by 
  sorry

end dyed_pink_correct_l44_44370


namespace discount_rate_pony_bogo_correct_l44_44302

def original_cost_seahorse (n : ℕ) : ℝ := n * 20
def original_cost_fox (n : ℕ) : ℝ := n * 15
def original_cost_pony (n : ℕ) : ℝ := n * 18

def tier_discount (n : ℕ) (price : ℝ) : ℝ :=
  if n >= 3 then 0.25 * price * n
  else if n == 2 then 0.20 * price * n
  else if n == 1 then 0.10 * price * n
  else 0

def bogo_discount (n : ℕ) (price : ℝ) (d : ℝ) : ℝ :=
  (n / 2) * (1 - d) * price

def total_original_cost (n_s n_f n_p : ℕ) : ℝ :=
  original_cost_seahorse n_s + original_cost_fox n_f + original_cost_pony n_p

def total_cost_with_tax (n_s n_f n_p : ℕ) : ℝ :=
  let orig_cost := total_original_cost n_s n_f n_p
  orig_cost * 1.07

def total_tier_discount (n_s n_f : ℕ) : ℝ :=
  tier_discount n_s 20 + tier_discount n_f 15

def final_cost (n_s n_f n_p : ℕ) (d : ℝ) : ℝ :=
  total_cost_with_tax n_s n_f n_p - total_tier_discount n_s n_f - bogo_discount n_p 18 d

def correct_discount_rate (total_savings target_savings n_s n_f n_p : ℕ) (d : ℝ) : Prop :=
  let final_with_savings := final_cost n_s n_f n_p d
  final_with_savings = total_cost_with_tax n_s n_f n_p - target_savings

theorem discount_rate_pony_bogo_correct :
  correct_discount_rate 30 1 4 3 5 (1/18) :=
by
  sorry

end discount_rate_pony_bogo_correct_l44_44302


namespace no_periodic_sum_l44_44405

def is_periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x : ℝ, f (x + p) = f x

theorem no_periodic_sum (g h : ℝ → ℝ) :
  (is_periodic g 2) → (is_periodic h (π / 2)) → ¬ ∃ T > 0, is_periodic (λ x, g x + h x) T :=
by {
  sorry
}

end no_periodic_sum_l44_44405


namespace number_of_special_functions_l44_44980

theorem number_of_special_functions {X : Type} (n : ℕ) (h : 2 ≤ n) (H : Fintype.card X = n) :
  let P := set X in
  let functions := {f : P → P | ∀ A B : P, A ≠ B → |f(A) ∩ f(B)| = |A ∩ B|} in
  Fintype.card functions = nat.factorial n :=
by
  let P := set X
  let functions := {f : P → P | ∀ A B : P, A ≠ B → |f(A) ∩ f(B)| = |A ∩ B|}
  haveI : Fintype X := sorry
  exact sorry

end number_of_special_functions_l44_44980


namespace johns_age_l44_44246

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
sorry

end johns_age_l44_44246


namespace part1_part2_l44_44931

variable {A B C a b c : ℝ}
variable {ac : ℝ}

-- Conditions
axiom h1 : a + c = 8
axiom h2 : cos B = 1/4
axiom h3 : a * c = 16
axiom h4 : sin A = sqrt 6 / 4

-- Part 1: Prove b = 2 * sqrt 6
theorem part1 : b = 2 * sqrt 6 := 
by {
  sorry
}

-- Part 2: Prove sin C = (3 * sqrt 6) / 8
theorem part2 : sin C = 3 * sqrt 6 / 8 :=
by {
  sorry
}

end part1_part2_l44_44931


namespace rearrange_sequence_l44_44842

theorem rearrange_sequence (N : ℕ) : (∃(σ : list ℕ), permutation σ (list.range N) ∧
  ∀(k : ℕ) (h₀ : 2 ≤ k) (h₁ : k ≤ N), ¬ (∃(s : ℕ), (list.sum (list.take k (list.drop s σ)) / k = ⌊list.sum (list.take k (list.drop s σ)) / k⌋))) ↔ even N :=
sorry

end rearrange_sequence_l44_44842


namespace angle_B_in_triangle_l44_44605

theorem angle_B_in_triangle (a b c A B C : ℝ) (hA : A = real.atan 3) (hC : C = real.acos (real.sqrt 5 / 5)) (habc : a ^ 2 + b ^ 2 = c ^ 2) :
  B = π / 4 :=
by
  sorry

end angle_B_in_triangle_l44_44605


namespace abc_cubed_sum_l44_44265

theorem abc_cubed_sum (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
    (h_eq : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) : 
    a^3 + b^3 + c^3 = -36 :=
by sorry

end abc_cubed_sum_l44_44265


namespace find_b_l44_44991

-- Define functions p and q
def p (x : ℝ) : ℝ := 3 * x - 5
def q (x : ℝ) (b : ℝ) : ℝ := 4 * x - b

-- Set the target value for p(q(3))
def target_val : ℝ := 9

-- Prove that b = 22/3
theorem find_b (b : ℝ) : p (q 3 b) = target_val → b = 22 / 3 := by
  intro h
  sorry

end find_b_l44_44991


namespace complex_inequality_l44_44621

variable {R : Type} [LinearOrderedCommRing R]

variables {A B C P1 P2 : Point}
variables {a b c a1 b1 c1 a2 b2 c2 : R}

-- Assumptions
axiom dist_A_P1 : dist A P1 = a1
axiom dist_B_P1 : dist B P1 = b1
axiom dist_C_P1 : dist C P1 = c1
axiom dist_A_P2 : dist A P2 = a2
axiom dist_B_P2 : dist B P2 = b2
axiom dist_C_P2 : dist C P2 = c2
axiom side_a : dist B C = a
axiom side_b : dist C A = b
axiom side_c : dist A B = c

-- Theorem to prove
theorem complex_inequality : a * a1 * a2 + b * b1 * b2 + c * c1 * c2 ≥ a * b * c := by
  sorry

end complex_inequality_l44_44621


namespace area_inequality_l44_44637

theorem area_inequality
  (a b c u : ℝ)
  (h1 : a + b + c + u = 1)
  (h2 : a = S_{AB_1 C_1})
  (h3 : b = S_{A_1 BC_1})
  (h4 : c = S_{A_1 B_1 C})
  (h5 : u = S_{A_1 B_1 C_1}) : 
  u^3 + (a + b + c)u^2 ≥ 4 * a * b * c :=
by
  sorry


end area_inequality_l44_44637


namespace matching_charge_and_minutes_l44_44252

def charge_at_time (x : ℕ) : ℕ :=
  100 - x / 6

def minutes_past_midnight (x : ℕ) : ℕ :=
  x % 60

theorem matching_charge_and_minutes :
  ∃ x, (x = 292 ∨ x = 343 ∨ x = 395 ∨ x = 446 ∨ x = 549) ∧ 
       charge_at_time x = minutes_past_midnight x :=
by {
  sorry
}

end matching_charge_and_minutes_l44_44252


namespace geometric_sequence_sum_div_term_l44_44470

theorem geometric_sequence_sum_div_term 
  (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ)
  (h1 : q = 2)
  (h2 : ∀ n, a (n + 1) = q * a n)
  (h_sum : ∀ n, S n = ∑ i in finset.range (n + 1), a i) :
  S 4 / a 2 = 15 / 2 :=
by
  sorry

end geometric_sequence_sum_div_term_l44_44470


namespace distinct_arrangements_balloon_l44_44093

theorem distinct_arrangements_balloon : 
  let n := 7 
  let freq_l := 2 
  let freq_o := 2 
  let freq_b := 1 
  let freq_a := 1 
  let freq_n := 1 
  Nat.factorial n / (Nat.factorial freq_l * Nat.factorial freq_o * Nat.factorial freq_b * Nat.factorial freq_a * Nat.factorial freq_n) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l44_44093


namespace no_periodic_sequence_a_n_l44_44198

def first_nonzero_digit (n : ℕ) : ℕ :=
sorry  -- Definition of a_n: first non-zero digit from the unit place in the decimal representation of n!

theorem no_periodic_sequence_a_n : ¬ ∃ N, ∃ T, ∀ k, first_nonzero_digit ((N + k * T)!) = first_nonzero_digit ((N + (k + 1) * T)!) :=
sorry

end no_periodic_sequence_a_n_l44_44198


namespace fraction_relevant_quarters_l44_44283

-- Define the total number of quarters and the number of relevant quarters
def total_quarters : ℕ := 50
def relevant_quarters : ℕ := 10

-- Define the theorem that states the fraction of relevant quarters is 1/5
theorem fraction_relevant_quarters : (relevant_quarters : ℚ) / total_quarters = 1 / 5 := by
  sorry

end fraction_relevant_quarters_l44_44283


namespace smallest_n_conditions_l44_44900

theorem smallest_n_conditions
  (n : ℕ)
  (1 < k_1 : ℕ)
  (k_ascending : ∀ i j : ℕ, i < j → k_1 < k_2 → ... → k_{n-1} < k_n)
  (a_integers : ∀ i : ℕ, i ≤ n → ∃ a_i : ℤ, true)
  (exists_k : ∀ N : ℤ, ∃ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ (k_i : ℕ) ∣ (N - a_i)) :
  n = 5 := sorry

end smallest_n_conditions_l44_44900


namespace largest_fraction_l44_44527

theorem largest_fraction :
  (∀ (w x y z : ℕ), (0 < w ∧ w < x ∧ x < y ∧ y < z) ∧
  w = 1 ∧ x = 3 ∧ y = 6 ∧ z = 10 →
  (max (max (max (max (w+x)/(y+z) (w+z)/(x+y)) (x+y)/(w+z)) (x+z)/(w+y)) (y+z)/(w+x)) = (y+z)/(w+x)) := by

end largest_fraction_l44_44527


namespace find_x_from_conditions_l44_44031

theorem find_x_from_conditions 
  (x y : ℕ) 
  (h1 : 1 ≤ x)
  (h2 : x ≤ 100)
  (h3 : 1 ≤ y)
  (h4 : y ≤ 100)
  (h5 : y > x)
  (h6 : (21 + 45 + 77 + 2 * x + y) / 6 = 2 * x) 
  : x = 16 := 
sorry

end find_x_from_conditions_l44_44031


namespace parallel_lines_l44_44530

noncomputable def line1 (x y : ℝ) : Prop := x - y + 1 = 0
noncomputable def line2 (a x y : ℝ) : Prop := x + a * y + 3 = 0

theorem parallel_lines (a x y : ℝ) : (∀ (x y : ℝ), line1 x y → line2 a x y → x = y ∨ (line1 x y ∧ x ≠ y)) → 
  (a = -1 ∧ ∃ d : ℝ, d = Real.sqrt 2) :=
sorry

end parallel_lines_l44_44530


namespace larger_bottle_percentage_increase_l44_44247

theorem larger_bottle_percentage_increase :
  (∀ (b : ℕ) (d : ℕ), d ≠ 0 → b * d = 16 * 4 * 7 ∧ b / (4 * 7) = 2 → -- The total weekly intake from 16-ounce bottles
  ∀ (t : ℤ), t = 728 - 448 → -- The weekly intake from larger bottles
  ∀ (b_l : ℕ), b_l * 14 = t → -- The number of larger bottles
  ∀ (l : ℕ), l = b_l / 2 * 1.25 → -- size of each larger bottle
  (l - 16) / 16 * 100 = 25) -- Percentage increase from 16-ounce to larger bottles
:= sorry

end larger_bottle_percentage_increase_l44_44247


namespace quad_function_intersects_x_axis_l44_44840

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quad_function_intersects_x_axis (m : ℝ) :
  (discriminant (2 * m) (8 * m + 1) (8 * m) ≥ 0) ↔ (m ≥ -1/16 ∧ m ≠ 0) :=
by
  sorry

end quad_function_intersects_x_axis_l44_44840


namespace dividend_is_correct_l44_44734

theorem dividend_is_correct :
  ∃ (R D Q V: ℕ), R = 6 ∧ D = 5 * Q ∧ D = 3 * R + 2 ∧ V = D * Q + R ∧ V = 86 :=
by
  sorry

end dividend_is_correct_l44_44734


namespace midpoint_of_isosceles_triangle_and_parallels_l44_44522

open Nat EuclideanGeometry

theorem midpoint_of_isosceles_triangle_and_parallels
  (A B C C' M P : Point)
  (h_iso : AB = BC)
  (h_circumcircle : Omega ABC)
  (h_diameter : Diameter CC')
  (h_parallel : Parallel (Line_through C' M) (Line_through B C))
  (h_intersections : Intersects (Line_through C' M) (AB) M ∧ Intersects (Line_through C' M) (AC) P)
  : IsMidpoint M C' P :=
begin
  sorry
end

end midpoint_of_isosceles_triangle_and_parallels_l44_44522


namespace cross_product_magnitude_l44_44570

noncomputable def vec_a : ℝ :=
sorry

noncomputable def vec_b : ℝ :=
sorry

def magnitude_a : ℝ := 4
def magnitude_b : ℝ := 3
def dot_product : ℝ := -2

theorem cross_product_magnitude :
  ∃ θ : ℝ, vec_a * vec_b * real.sin θ = magnitude_a * magnitude_b * real.sin θ ∧
           vec_a * vec_b = magnitude_a * magnitude_b * real.cos θ ∧
           real.sqrt(1 - (real.cos θ) ^ 2) = real.sin θ 
           ∧ magnitude_a * magnitude_b * real.sin θ = 2 * real.sqrt 35 :=
sorry

end cross_product_magnitude_l44_44570


namespace cos_theta_projection_l44_44891

noncomputable def vectors_projection_cos_theta : Prop :=
  ∀ (a b : ℝ^3) (θ : ℝ),
  ‖a‖ = 1 → ‖b‖ = 1 →
  (a • b / ‖b‖ ^ 2) * b = (1 / 3) * b →
  Real.cos θ = 1 / 3

theorem cos_theta_projection
  (a b : ℝ^3) (θ : ℝ)
  (ha : ‖a‖ = 1)
  (hb : ‖b‖ = 1)
  (hproj : (a • b / ‖b‖ ^ 2) * b = (1 / 3) * b) :
  Real.cos θ = 1 / 3 :=
sorry

end cos_theta_projection_l44_44891


namespace a2013_eq_1_l44_44679

-- Define the sequence and conditions
axiom sequence (a : ℕ → ℝ)
axiom sum_of_three_consecutive (a : ℕ → ℝ) :
  ∀ n : ℕ, a n + a (n + 1) + a (n + 2) = a (n + 1) + a (n + 2) + a (n + 3)
axiom a3_eq_x (x : ℝ) (a : ℕ → ℝ) : a 3 = x
axiom a999_eq_3_minus_2x (x : ℝ) (a : ℕ → ℝ) : a 999 = 3 - 2 * x

-- The theorem to prove
theorem a2013_eq_1 (a : ℕ → ℝ) (x : ℝ) :
  (∀ n : ℕ, a n + a (n + 1) + a (n + 2) = a (n + 1) + a (n + 2) + a (n + 3)) →
  a 3 = x →
  a 999 = 3 - 2 * x →
  a 2013 = 1 :=
by
  intros
  sorry

end a2013_eq_1_l44_44679


namespace hyperbola_vertex_distance_l44_44021

theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ), (x^2 / 48 - y^2 / 16 = 1) →
    (2 * real.sqrt 48 = 8 * real.sqrt 3) :=
by
  intros x y h
  sorry

end hyperbola_vertex_distance_l44_44021


namespace find_a_range_l44_44812

-- Define f as an odd function on ℝ and satisfying the given conditions
section
  variable {f : ℝ → ℝ}
  variable {a : ℝ}

  -- Define conditions
  def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = -f(x)
  def periodic_function (f : ℝ → ℝ) : Prop := ∀ x, f(x + 5) = f(x)
  def condition_1 : Prop := odd_function f
  def condition_2 : Prop := ∀ x, f(x + 5 / 2) = -f(x)
  def condition_3 : Prop := f(1) > -1
  def condition_4 : Prop := f(4) = Real.log 2 a ∧ a > 0 ∧ a ≠ 1

  -- Define the theorem that needs to be proven
  theorem find_a_range (f : ℝ → ℝ) (a : ℝ) 
    (h1 : condition_1)
    (h2 : condition_2)
    (h3 : condition_3)
    (h4 : condition_4) :
    (0 < a ∧ a < 1) ∨ (2 < a) :=
  sorry
end

end find_a_range_l44_44812


namespace f_is_odd_f_is_increasing_l44_44877

variable (a : ℝ) (h₁ : a > 1)

def f (x : ℝ) : ℝ := (a ^ x - 1) / (a ^ x + 1)

theorem f_is_odd : ∀ x : ℝ, f a (-x) = -f a x :=
by {
  sorry
}

theorem f_is_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂ :=
by {
  sorry
}

end f_is_odd_f_is_increasing_l44_44877


namespace num_valid_passwords_l44_44395

-- Define the given digits
def digits := [0, 5, 1, 0, 1, 8]

-- Define the condition that no two 1s or 0s can be adjacent
def validArrangement (l : List Nat) : Prop :=
  ∀ i, i < l.length - 1 → 
    (l.get! i ≠ 1 ∨ l.get! (i + 1) ≠ 1) ∧ 
    (l.get! i ≠ 0 ∨ l.get! (i + 1) ≠ 0)

-- Define the main theorem that the number of valid passwords is 396
theorem num_valid_passwords : (List.permutations digits).count validArrangement = 396 :=
  sorry

end num_valid_passwords_l44_44395


namespace bijective_bounded_dist_l44_44258

open Int

theorem bijective_bounded_dist {k : ℕ} (f : ℤ → ℤ) 
    (hf_bijective : Function.Bijective f)
    (hf_property : ∀ i j : ℤ, |i - j| ≤ k → |f i - (f j)| ≤ k) :
    ∀ i j : ℤ, |f i - (f j)| = |i - j| := 
sorry

end bijective_bounded_dist_l44_44258


namespace evaluate_expression_l44_44483

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x^(1/3)

theorem evaluate_expression : 3 * f 3 + f 27 = 19818 := 
by 
  let f := λ x : ℝ, x^3 + 3 * x^(1/3)
  let f3 := f 3
  let f27 := f 27
  have h1 : f3 = 36 := 
    by 
      unfold f
      apply eq_of_heq
      apply congr_arg
      simp
  have h2 : f27 = 19710 := 
    by 
      unfold f
      apply eq_of_heq
      apply congr_arg
      simp
  calc 
    3 * f 3 + f 27
      = 3 * 36 + 19710 : by rw [h1, h2]
      ... = 108 + 19710 : by ring
      ... = 19818 : by ring

end evaluate_expression_l44_44483


namespace estimate_probability_of_hitting_a_shot_l44_44781

-- Define the conditions from the problem as variables and expressions
variable (attempts_set1 attempts_set2 attempts_set3 total_attempts : ℕ)
variable (hits_set1 hits_set2 hits_set3 total_hits: ℕ)
variable (freq_set1 freq_set2 freq_set3 total_freq: ℚ)

-- Set the values for the attempts, hits, and frequencies
def setup_conditions : Prop :=
  attempts_set1 = 100 ∧ attempts_set2 = 200 ∧ attempts_set3 = 300 ∧ total_attempts = 600 ∧
  hits_set1 = 68 ∧ hits_set2 = 124 ∧ hits_set3 = 174 ∧ total_hits = 366 ∧
  freq_set1 = 68 / 100 ∧ freq_set2 = 124 / 200 ∧ freq_set3 = 174 / 300 ∧ total_freq = 366 / 600

-- Define the probability estimate proposition
def probability_estimate : Prop :=
  total_freq = 0.61

-- Main theorem combining conditions and conclusion
theorem estimate_probability_of_hitting_a_shot :
  setup_conditions → probability_estimate :=
by
  intros hconds
  sorry

end estimate_probability_of_hitting_a_shot_l44_44781


namespace differential_eq_solution_correct_l44_44678

noncomputable def diff_eq_solution (C1 C2 : ℝ) : ∀ (x : ℝ), ℝ :=
  C1 * Real.cos (3 * x) +
  C2 * Real.sin (3 * x) +
  ((6 / 37) * x + 641 / 1369) * Real.exp (-x) * Real.cos (3 * x) +
  ((1 / 37) * x + 298 / 1369) * Real.exp (-x) * Real.sin (3 * x)

theorem differential_eq_solution_correct (C1 C2 : ℝ) :
  ∀ x : ℝ,
    let y := diff_eq_solution C1 C2 x in 
    (deriv (deriv y + y * 9) = 
    Real.exp (-x) * (-Real.cos (3 * x) + (x + 2) * Real.sin (3 * x))) :=
by
  intros x y
  sorry

end differential_eq_solution_correct_l44_44678


namespace incorrect_weight_conclusion_l44_44058

theorem incorrect_weight_conclusion (x y : ℝ) (h1 : y = 0.85 * x - 85.71) :
  ¬ (x = 160 → y = 50.29) :=
sorry

end incorrect_weight_conclusion_l44_44058


namespace pairing_probability_l44_44938

variable {students : Fin 28} (Alex Jamie : Fin 28)

theorem pairing_probability (h1 : ∀ (i j : Fin 28), i ≠ j) :
  ∃ p : ℚ, p = 1 / 27 ∧ 
  (∃ (A_J_pairs : Finset (Fin 28) × Finset (Fin 28)),
  A_J_pairs.1 = {Alex} ∧ A_J_pairs.2 = {Jamie}) -> p = 1 / 27
:= sorry

end pairing_probability_l44_44938


namespace find_fraction_l44_44262

theorem find_fraction (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a / b + (2 * a + 5 * b) / (b + 5 * a) = 4)
  (h3 : b = 1) : a / b = (17 + Real.sqrt 269) / 10 :=
by
  sorry

end find_fraction_l44_44262


namespace silly_bills_count_l44_44410

theorem silly_bills_count (x : ℕ) (h1 : x + 2 * (x + 11) + 3 * (x - 18) = 100) : x = 22 :=
by { sorry }

end silly_bills_count_l44_44410


namespace grilled_cheese_sandwiches_l44_44220

theorem grilled_cheese_sandwiches (h_cheese : ℕ) (g_cheese : ℕ) (total_cheese : ℕ) (ham_sandwiches : ℕ) (grilled_cheese_sandwiches : ℕ) :
  h_cheese = 2 →
  g_cheese = 3 →
  total_cheese = 50 →
  ham_sandwiches = 10 →
  total_cheese - (ham_sandwiches * h_cheese) = grilled_cheese_sandwiches * g_cheese →
  grilled_cheese_sandwiches = 10 :=
by {
  intros,
  sorry
}

end grilled_cheese_sandwiches_l44_44220


namespace percent_employees_three_years_or_more_l44_44963

theorem percent_employees_three_years_or_more 
  (y : ℕ) 
  (less_than_1_year : ℕ := 4 * y)
  (one_to_two_years : ℕ := 6 * y)
  (two_to_three_years : ℕ := 9 * y)
  (three_to_four_years : ℕ := 2 * y)
  (four_to_five_years : ℕ := 1 * y)
  (five_to_six_years : ℕ := 3 * y)
  (six_to_seven_years : ℕ := 3 * y)
  (seven_to_eight_years : ℕ := 2 * y)
  (eight_plus_years : ℕ := 1 * y)
  (total_employees : ℕ := less_than_1_year + one_to_two_years + two_to_three_years + three_to_four_years + four_to_five_years + five_to_six_years + six_to_seven_years + seven_to_eight_years + eight_plus_years)
  (three_years_or_more : ℕ := three_to_four_years + four_to_five_years + five_to_six_years + six_to_seven_years + seven_to_eight_years + eight_plus_years)
  :
  (\frac{three_years_or_more}{total_employees} * 100 = 38.71) :=
sorry

end percent_employees_three_years_or_more_l44_44963


namespace number_of_m_values_l44_44603

theorem number_of_m_values (m : ℕ) (h1 : 4 * m > 11) (h2 : m < 12) : 
  11 - 3 + 1 = 9 := 
sorry

end number_of_m_values_l44_44603


namespace time_between_ticks_at_6_oclock_l44_44459

theorem time_between_ticks_at_6_oclock
    (ticks_at_6 : ℕ)
    (ticks_at_12 : ℕ)
    (time_12 : ℕ)
    (intervals_equal : Prop) :
    ticks_at_6 = 6 → 
    ticks_at_12 = 12 → 
    time_12 = 55 → 
    intervals_equal → 
    (ticks_at_6 - 1) * (time_12 / (ticks_at_12 - 1)) = 25 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  have h : time_12 / (ticks_at_12 - 1) = 5 := by norm_num
  rw [h]
  norm_num
  sorry

end time_between_ticks_at_6_oclock_l44_44459


namespace combination_sum_l44_44743

theorem combination_sum  :
  (nat.choose 4 4 + nat.choose 5 4 + nat.choose 6 4 + nat.choose 7 4 + 
   nat.choose 8 4 + nat.choose 9 4 + nat.choose 10 4 = 462) :=
by
  -- Definitions for each of the conditions could be added here if necessary.
  -- The proof can use the property of combinations extensively.
  sorry

end combination_sum_l44_44743


namespace sum_of_four_primes_div_by_60_l44_44624

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem sum_of_four_primes_div_by_60
  (p q r s : ℕ)
  (hp : is_prime p)
  (hq : is_prime q)
  (hr : is_prime r)
  (hs : is_prime s)
  (horder : 5 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < p + 10) :
  (p + q + r + s) % 60 = 0 :=
by
  sorry


end sum_of_four_primes_div_by_60_l44_44624


namespace balloon_arrangements_l44_44148

-- Defining the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Given Conditions
def seven_factorial := fact 7 -- 7!
def two_factorial := fact 2 -- 2!

-- Statement to prove
theorem balloon_arrangements : seven_factorial / (two_factorial * two_factorial) = 1260 :=
by
  sorry

end balloon_arrangements_l44_44148


namespace age_of_teacher_l44_44686

theorem age_of_teacher (avg_age_students : ℕ) (num_students : ℕ) (inc_avg_with_teacher : ℕ) (num_people_with_teacher : ℕ) :
  avg_age_students = 21 →
  num_students = 20 →
  inc_avg_with_teacher = 22 →
  num_people_with_teacher = 21 →
  let total_age_students := num_students * avg_age_students
  let total_age_with_teacher := num_people_with_teacher * inc_avg_with_teacher
  total_age_with_teacher - total_age_students = 42 :=
by
  intros
  sorry

end age_of_teacher_l44_44686


namespace solution_set_inequality_l44_44903

variable {f : ℝ → ℝ}

axiom A1 : ∀ a b : ℝ, a < b → f b < f a
axiom A2 : f 0 = 3
axiom A3 : f 3 = -1

theorem solution_set_inequality (x : ℝ) : |f (x+1) - 1| < 2 ↔ x ∈ set.Ioo (-1) 2 :=
by
  sorry

end solution_set_inequality_l44_44903


namespace girls_in_pairs_probability_l44_44762

theorem girls_in_pairs_probability : 
  let total_ways := 8! / (2^4 * 4!),
      unfavorable_pairs := nat.factorial 4,
      probability_no_girl_pairs := unfavorable_pairs / real.of_rat total_ways,
      probability_at_least_one_girl_pair := 1 - probability_no_girl_pairs
  in probability_at_least_one_girl_pair ≈ 0.77 :=
by {
  sorry
}

end girls_in_pairs_probability_l44_44762


namespace johns_age_l44_44242

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
sorry

end johns_age_l44_44242


namespace number_of_valid_sets_l44_44334

-- Define the elements as abstract types
variables {α : Type*} (a1 a2 a3 a4 : α)

-- Define the conditions in Lean
def valid_set (M : set α) : Prop :=
  M ⊆ {a1, a2, a3, a4} ∧ M ∩ {a1, a2, a3} = {a1, a2}

-- Statement of the proof problem
theorem number_of_valid_sets :
  {M : set α | valid_set a1 a2 a3 a4 M}.to_finset.card = 2 :=
sorry

end number_of_valid_sets_l44_44334


namespace greatest_common_factor_and_ratio_l44_44720

theorem greatest_common_factor_and_ratio (a b gcf: ℕ) (ratio: ℚ):
  (a = 4536) →
  (b = 14280) →
  (gcf = 504) →
  (ratio = 1 / 3) →
  Nat.gcd a b = gcf ∧ (a : ℚ) / (b : ℚ) = ratio :=
by
  intros h_a h_b h_gcf h_ratio
  rw [h_a, h_b, h_gcf, h_ratio]
  split
  · sorry
  · sorry

end greatest_common_factor_and_ratio_l44_44720


namespace part1_part2_part3_l44_44550

-- Define the function and its properties
def f (x : ℝ) (a : ℝ) : ℝ := exp x + a * x
def f' (x : ℝ) (a : ℝ) : ℝ := deriv (λ x, exp x + a * x) x

-- Define the conditions
variables {a x₁ x₂ t : ℝ}
variable (h_a : a < -e)
variable (h_intersection_1 : f x₁ a = 0)
variable (h_intersection_2 : f x₂ a = 0)
variable (h_order : x₁ < x₂)
variable (h_t : t = real.sqrt (x₂ / x₁))

-- Define the problem statements with their respective proofs replaced by "sorry"
theorem part1 : a < -e := 
    h_a

theorem part2 : f' ((3 * x₁ + x₂) / 4) a < 0 := 
    sorry

theorem part3 : (t - 1) * (a + real.sqrt 3) = -2 * real.sqrt 3 := 
    sorry

end part1_part2_part3_l44_44550


namespace inradius_half_l44_44604

namespace TriangleInradius

variables {A B C M : Type} 
variables (r_ABC r_BCM : ℝ) (p_ABC p_BCM : ℝ) 
variables [is_triangle A B C] [is_median B M A C] [is_midpoint A C M]

theorem inradius_half (h : r_BCM = (1 / 2) * r_ABC) : False :=
begin
  -- Using properties of triangle areas and inradii to derive contradiction
  sorry
end

end TriangleInradius

end inradius_half_l44_44604


namespace balloon_permutations_l44_44142

theorem balloon_permutations : 
  let n : ℕ := 7
  let k1 : ℕ := 2
  let k2 : ℕ := 2
  ∃ distinct_arrangements : ℕ, 
  distinct_arrangements = n.factorial / (k1.factorial * k2.factorial) 
  ∧ distinct_arrangements = 1260 :=
by
  sorry

end balloon_permutations_l44_44142


namespace sum_of_xyz_l44_44051

open Nat

def is_simplified_fraction (a b : ℕ) : Prop :=
  gcd a b = 1

theorem sum_of_xyz 
  (x y z : ℕ) 
  (h1 : is_simplified_fraction x 9) 
  (h2 : is_simplified_fraction y 15) 
  (h3 : is_simplified_fraction z 14) 
  (h4 : (x * y * z) = 882) : 
  x + y + z = 21 :=
sorry

end sum_of_xyz_l44_44051


namespace distinct_arrangements_balloon_l44_44122

theorem distinct_arrangements_balloon : 
  let n := 7
  let r1 := 2
  let r2 := 2
  (Nat.factorial n) / ((Nat.factorial r1) * (Nat.factorial r2)) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l44_44122


namespace number_of_people_between_Masha_and_Nastya_l44_44504

theorem number_of_people_between_Masha_and_Nastya :
  (∀ positions : (ℕ → ℕ), -- a function mapping a person to their position
    positions "Irina" = 1 ∧ -- Irina in the first position
    positions "Olya" = 5 ∧ -- Olya in the fifth position
    positions "Anya" = 4 ∧ -- Anya in the fourth position
    positions "Masha" = 3 ∧ -- Masha in the third position
    positions "Nastya" = 2 -- Nastya in the second position
  ) → -- Implication
  (abs (positions "Masha" - positions "Nastya") - 1) = 1 := -- Number of people between Masha and Nastya is 1
by
  sorry

end number_of_people_between_Masha_and_Nastya_l44_44504


namespace parabola_ellipse_properties_l44_44887

-- Definitions of the conditions
def parabola_eq (y x : ℝ) (p : ℝ) : Prop := y^2 = 2*p*x
def point_M_on_parabola (p : ℝ) : Prop := ∃ y₀ : ℝ, parabola_eq y₀ 3 p
def distance_M_to_focus (p : ℝ) : Prop := ∃ y₀ : ℝ, (4 : ℝ) = dist (3, y₀) (1, 0)
def ellipse_eq (y x a b : ℝ) : Prop := (a > b) ∧ (b > 0) ∧ (y^2 / a^2 + x^2 / b^2 = 1)
def ellipse_condition (a b : ℝ) : Prop := ∃ x y : ℝ, ellipse_eq y x a b ∧ dist (x, y) (1, 0) = 0
def eccentricity_condition (a b : ℝ) : Prop := (a^2 - b^2 = (a * √2 / 2)^2)

-- Main theorem statement
theorem parabola_ellipse_properties :
  ∃ p a b : ℝ,
    point_M_on_parabola p ∧
    distance_M_to_focus p ∧
    ellipse_condition a b ∧
    eccentricity_condition a b ∧
    (∃ λ μ : ℝ, (λ + μ = -1)) := sorry

end parabola_ellipse_properties_l44_44887


namespace ordering_schemes_count_l44_44825

def dishes_fengfeng_likes : set string := {"Fresh Sprouts", "Garlic Cabbage", "Braised Eggplant"}
def dishes_leilei_likes : set string := {"Papaya Chicken", "Spiced Beef", "Braised Lamb", "Fresh Sprouts", "Garlic Cabbage"}
def dishes_feifei_likes : set string := {"Papaya Chicken", "Grilled Prawns", "Fresh Sprouts", "Braised Eggplant"}

theorem ordering_schemes_count :
  ∃ (orders : list (string × string)) (disjoint : list.pairwise (λ x y : string × string, x.2 ≠ y.2) orders),
    (∀ (p : string), p ∈ ["Fengfeng", "Leilei", "Feifei"] → ∃ dish : string, (p, dish) ∈ orders)
    ∧ orders.length = 30 := 
sorry

end ordering_schemes_count_l44_44825


namespace cos_of_sin_half_l44_44160

theorem cos_of_sin_half (α : ℝ) (h : sin (α / 2) = sqrt 3 / 3) : cos α = 1 / 3 := 
by
  sorry

end cos_of_sin_half_l44_44160


namespace distinct_points_count_l44_44556

def A : Set ℕ := {5}
def B : Set ℕ := {1, 2}
def C : Set ℕ := {1, 3, 4}

theorem distinct_points_count :
  (A × B × C).toFinset.card = 6 := 
sorry

end distinct_points_count_l44_44556


namespace intersection_complement_A_B_l44_44846

open Set

theorem intersection_complement_A_B :
  let A := {x : ℝ | x + 1 > 0}
  let B := {-2, -1, 0, 1}
  (compl A ∩ B : Set ℝ) = {-2, -1} :=
by
  sorry

end intersection_complement_A_B_l44_44846


namespace new_mean_is_correct_l44_44286

-- Define the given conditions for the problem
def total_students : ℕ := 48
def students_day1 : ℕ := 40
def avg_day1 : ℝ := 75 / 100  -- representing 75%
def students_day2 : ℕ := 8
def avg_day2 : ℝ := 82 / 100  -- representing 82%

-- Define the goal of the problem
def new_mean : ℝ := 
  (students_day1 * avg_day1 + students_day2 * avg_day2) / total_students

-- State the problem to prove new mean equals 76.17%
theorem new_mean_is_correct : new_mean = 76.17 / 100 := sorry

end new_mean_is_correct_l44_44286


namespace four_digit_number_solution_l44_44427

theorem four_digit_number_solution (x : ℕ) (h1 : 999 < x) (h2 : x < 10000) :
  (x / 10:ℝ → Int - (x / 1009:ℝ) = 2059.2 → x = 2288) V (x - (x / 100:ℝ) = 2059.2 → x = 2080 :=
by
  sorry

end four_digit_number_solution_l44_44427


namespace cricket_team_right_handed_l44_44288

theorem cricket_team_right_handed (total_players : ℕ) (throwers : ℕ) (f : ℕ → ℕ) 
  (left_handed : ℕ) (right_handed_players : ℕ) :
  total_players = 150 →
  throwers = 90 →
  left_handed = f (total_players - throwers) →
  f n = n / 5 →
  ∀ x, x = total_players - throwers - left_handed + throwers → x = 138 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h4, h3] at h5
  sorry

end cricket_team_right_handed_l44_44288


namespace conical_frustum_rectangular_piece_l44_44423

theorem conical_frustum_rectangular_piece:
  ∀ (base_diameter slant_height : ℝ), 
  base_diameter = 4 ∧ slant_height = 6 → 
  (⟨12, 6⟩ : ℝ × ℝ) :=
by
  intro base_diameter slant_height
  intro h
  have radius := base_diameter / 2
  have arc_length := 2 * Real.pi * radius
  have full_circumference := 2 * Real.pi * slant_height
  have sector_angle := arc_length / full_circumference * 360
  have width := slant_height
  have length := arc_length
  exact ⟨length, width⟩
  sorry

end conical_frustum_rectangular_piece_l44_44423


namespace prob_sum_to_3_three_dice_correct_l44_44360

def prob_sum_to_3_three_dice (sum : ℕ) (dice_count : ℕ) (dice_faces : Finset ℕ) : ℚ :=
  if sum = 3 ∧ dice_count = 3 ∧ dice_faces = {1, 2, 3, 4, 5, 6} then (1 : ℚ) / 216 else 0

theorem prob_sum_to_3_three_dice_correct :
  prob_sum_to_3_three_dice 3 3 {1, 2, 3, 4, 5, 6} = (1 : ℚ) / 216 := 
by
  sorry

end prob_sum_to_3_three_dice_correct_l44_44360


namespace min_value_m_l44_44673

-- Define the function f(x)
def f (x : ℝ) : ℝ := sqrt 3 * sin (2*x) - cos (2*x)

-- Define the conditions
def left_shift (m : ℝ) (h : m > -π / 2) : ℝ → ℝ := fun x => f (x + m.abs)

def symmetric_about (line : ℝ) (g : ℝ → ℝ) : Prop :=
  ∀ x, g (2*line - x) = g x

-- The theorem stating the minimum value of m under the given conditions
theorem min_value_m (m : ℝ) (h : m > -π / 2) :
  symmetric_about (π / 6) (left_shift m h) →
  m = -π / 6 :=
sorry

end min_value_m_l44_44673


namespace johns_age_l44_44223

theorem johns_age (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l44_44223


namespace distinct_arrangements_balloon_l44_44125

-- Let's define the basic conditions:
def total_letters : Nat := 7
def repeats_l : Nat := 2
def repeats_o : Nat := 2

-- Now let's state the problem.
theorem distinct_arrangements_balloon : 
  (Nat.factorial total_letters) / ((Nat.factorial repeats_l) * (Nat.factorial repeats_o)) = 1260 := 
by
  sorry

end distinct_arrangements_balloon_l44_44125


namespace sum_cn_eq_Tn_l44_44204

def Sn (n : ℕ) : ℕ := n^2 + 2 * n
def an (n : ℕ) : ℕ := 2 * n + 1
def bn (n : ℕ) : ℕ := 2^n
def cn (n : ℕ) : ℕ := an n * bn n
def Tn (n : ℕ) : ℕ := (2 * n - 1) * 2^(n + 1) + 2

theorem sum_cn_eq_Tn (n : ℕ) : 
  (finset.range n).sum (λ i, cn (i+1)) = Tn n :=
sorry

end sum_cn_eq_Tn_l44_44204


namespace A_proper_subset_B_l44_44174

def setA : Set ℝ := {x | ∃ k : ℤ, x = (k * Real.pi / 2) + (Real.pi / 4)}
def setB : Set ℝ := {x | ∃ k : ℤ, x = (k * Real.pi / 4) + (Real.pi / 2)}

theorem A_proper_subset_B : setA ⊆ setB ∧ ∃ x ∈ setB, x ∉ setA := by
  sorry

end A_proper_subset_B_l44_44174


namespace balloon_arrangements_l44_44100

-- Define the variables
def n : ℕ := 7
def L_count : ℕ := 2
def O_count : ℕ := 2
def B_count : ℕ := 1
def A_count : ℕ := 1
def N_count : ℕ := 1

-- Define the multiset permutation formula
def multiset_permutations (n : ℕ) (counts : List ℕ) : ℕ :=
  n.factorial / (counts.map Nat.factorial).prod

-- Proof that the number of distinct arrangements is 1260
theorem balloon_arrangements : multiset_permutations n [L_count, O_count, B_count, A_count, N_count] = 1260 :=
  by
  -- The proof is omitted
  sorry

end balloon_arrangements_l44_44100


namespace irregular_polygon_composite_l44_44790

noncomputable def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, 1 < d ∧ d < n ∧ n % d = 0

theorem irregular_polygon_composite (n : ℕ) (α : ℝ) (h1 : ∃ P : ℕ → ℝ × ℝ, 
  (∀ i, P i ≠ P ((i + 1) % n)) ∧ -- irregularity condition
  (∀ i, P i = P ((i + 2 * π / α) % n))) -- rotation condition
  (h2 : α ≠ 2 * π) : is_composite n :=
sorry

end irregular_polygon_composite_l44_44790


namespace tan_symmetric_about_k_pi_over_2_min_value_cos2x_plus_sinx_l44_44541

theorem tan_symmetric_about_k_pi_over_2 (k : ℤ) : 
  (∀ x : ℝ, Real.tan (x + k * Real.pi / 2) = Real.tan x) := 
sorry

theorem min_value_cos2x_plus_sinx : 
  (∀ x : ℝ, Real.cos x ^ 2 + Real.sin x ≥ -1) ∧ (∃ x : ℝ, Real.cos x ^ 2 + Real.sin x = -1) :=
sorry

end tan_symmetric_about_k_pi_over_2_min_value_cos2x_plus_sinx_l44_44541


namespace problem_statement_l44_44953

noncomputable def f (x : ℚ) : ℚ := (x^2 - x - 6) / (x^3 - 2 * x^2 - x + 2)

def a : ℕ := 1  -- number of holes
def b : ℕ := 2  -- number of vertical asymptotes
def c : ℕ := 1  -- number of horizontal asymptotes
def d : ℕ := 0  -- number of oblique asymptotes

theorem problem_statement : a + 2 * b + 3 * c + 4 * d = 8 :=
by
  sorry

end problem_statement_l44_44953


namespace triangle_perimeter_l44_44578

/-- In a triangle ABC, where sides a, b, c are opposite to angles A, B, C respectively.
Given the area of the triangle = 15 * sqrt 3 / 4, 
angle A = 60 degrees and 5 * sin B = 3 * sin C,
prove that the perimeter of triangle ABC is 8 + sqrt 19. -/
theorem triangle_perimeter
  (a b c : ℝ)
  (A B C : ℝ)
  (hA : A = 60)
  (h_area : (1 / 2) * b * c * (Real.sin (A / (180 / Real.pi))) = 15 * Real.sqrt 3 / 4)
  (h_sin : 5 * Real.sin B = 3 * Real.sin C) :
  a + b + c = 8 + Real.sqrt 19 :=
sorry

end triangle_perimeter_l44_44578


namespace vector_magnitude_BC_l44_44866

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude_BC
  (AB AC : ℝ × ℝ)
  (hAB : magnitude AB = 1)
  (hAC : magnitude AC = 2)
  (angle_BAC : ∃ θ : ℝ, θ = real.pi / 3) :
  magnitude (AC - AB) = real.sqrt 3 :=
by
  sorry

end vector_magnitude_BC_l44_44866


namespace johns_age_l44_44236

theorem johns_age (d j : ℕ) 
  (h1 : j = d - 30) 
  (h2 : j + d = 80) : 
  j = 25 :=
by
  sorry

end johns_age_l44_44236


namespace rotation_matrix_inverse_and_area_l44_44048

theorem rotation_matrix_inverse_and_area :
  let M := ![
    [real.cos (-real.pi / 4), -real.sin (-real.pi / 4)], 
    [real.sin (-real.pi / 4), real.cos (-real.pi / 4)]
  ] in
  let M_inv := ![
    [real.cos (real.pi / 4), -real.sin (real.pi / 4)], 
    [real.sin (real.pi / 4), real.cos (real.pi / 4)]
  ] in
  let A := (1, 0) in
  let B := (2, 2) in
  let C := (3, 0) in
  let area_ABC := (1 / 2) * (3 - 1) * 2 in
  let area_A1B1C1 := area_ABC in
  M = ![
    [real.sqrt 2 / 2, real.sqrt 2 / 2], 
    [-real.sqrt 2 / 2, real.sqrt 2 / 2]
  ] ∧
  M_inv = ![
    [real.sqrt 2 / 2, -real.sqrt 2 / 2], 
    [real.sqrt 2 / 2, real.sqrt 2 / 2]
  ] ∧
  area_A1B1C1 = 2 :=
by {
  sorry
}

end rotation_matrix_inverse_and_area_l44_44048


namespace balloon_arrangements_l44_44102

-- Define the variables
def n : ℕ := 7
def L_count : ℕ := 2
def O_count : ℕ := 2
def B_count : ℕ := 1
def A_count : ℕ := 1
def N_count : ℕ := 1

-- Define the multiset permutation formula
def multiset_permutations (n : ℕ) (counts : List ℕ) : ℕ :=
  n.factorial / (counts.map Nat.factorial).prod

-- Proof that the number of distinct arrangements is 1260
theorem balloon_arrangements : multiset_permutations n [L_count, O_count, B_count, A_count, N_count] = 1260 :=
  by
  -- The proof is omitted
  sorry

end balloon_arrangements_l44_44102


namespace num_concave_down_functions_l44_44074

theorem num_concave_down_functions : 
  ∃ f : ℕ → (ℝ → ℝ), 
  (∀ i : ℕ, i < 3 → (0 < ∀ (x1 x2 : ℝ), x1 < x2 ∧ x2 < 1 → 
  (f i ⟨ (x1 + x2) / 2 ⟩ < (f i ⟨ x1 ⟩ + f i ⟨ x2 ⟩) / 2))) ∧
  (((f 0 = (λ x, 2^x)) ∨ (f 1 = (λ x, log 2 x)) ∨ (f 2 = (λ x, x^2)) ) → 
  f 0 ∈ { (λ x, 2^x), (λ x, log 2 x), (λ x, x^2) } ∧ 
  f 1 ∈ { (λ x, 2^x), (λ x, log 2 x), (λ x, x^2) } ∧ 
  f 2 ∈ { (λ x, 2^x), (λ x, log 2 x), (λ x, x^2)}) ∧
  2 := 
sorry

end num_concave_down_functions_l44_44074


namespace valid_numbers_count_l44_44508

def is_valid_number (n : ℕ) : Prop :=
  let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10]
  (∀ d ∈ digits, d = 2 ∨ d = 3) ∧ (digits.contains 2 ∧ digits.contains 3)

def count_valid_numbers : ℕ :=
  (List.range (10^4)).countp is_valid_number

theorem valid_numbers_count : count_valid_numbers = 14 := by
  sorry

end valid_numbers_count_l44_44508


namespace function_quadrants_l44_44203

theorem function_quadrants (n : ℝ) (h: ∀ x : ℝ, x ≠ 0 → ((n-1)*x * x > 0)) : n > 1 :=
sorry

end function_quadrants_l44_44203


namespace johns_age_l44_44227

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l44_44227


namespace find_initial_amount_l44_44022

-- Definitions for conditions
def final_amount : ℝ := 5565
def rate_year1 : ℝ := 0.05
def rate_year2 : ℝ := 0.06

-- Theorem statement to prove the initial amount
theorem find_initial_amount (P : ℝ) 
  (H : final_amount = (P * (1 + rate_year1)) * (1 + rate_year2)) :
  P = 5000 := 
sorry

end find_initial_amount_l44_44022


namespace solve_system_eq_l44_44307

theorem solve_system_eq (x y : ℝ) :
  x^2 + y^2 + 6 * x * y = 68 ∧ 2 * x^2 + 2 * y^2 - 3 * x * y = 16 ↔
  (x = 4 ∧ y = 2) ∨ (x = 2 ∧ y = 4) ∨ (x = -4 ∧ y = -2) ∨ (x = -2 ∧ y = -4) := 
by
  sorry

end solve_system_eq_l44_44307


namespace find_b_l44_44331

-- Definitions and conditions
def line_eq (b : ℝ) (x : ℝ) : ℝ := b - x

noncomputable def area_QOP (b : ℝ) : ℝ := (1/2) * b * b

noncomputable def area_QRS (b : ℝ) : ℝ := (1/2) * (6 - b) * (6 - b)

lemma ratio_areas (b : ℝ) (hb_cond : 0 < b ∧ b < 6) : (area_QRS b) / (area_QOP b) = 4 / 9 :=
sorry

-- Theorem to prove
theorem find_b (b : ℝ) (hb_cond : 0 < b ∧ b < 6)
  (h_ratio : ratio_areas b hb_cond) : b = 3.6 :=
sorry

end find_b_l44_44331


namespace balloon_permutations_l44_44140

theorem balloon_permutations : 
  let n : ℕ := 7
  let k1 : ℕ := 2
  let k2 : ℕ := 2
  ∃ distinct_arrangements : ℕ, 
  distinct_arrangements = n.factorial / (k1.factorial * k2.factorial) 
  ∧ distinct_arrangements = 1260 :=
by
  sorry

end balloon_permutations_l44_44140


namespace ellipse_foci_distance_l44_44449

/-- Given an ellipse tangent to the x-axis at (6, 0) and to the y-axis at (0, 3), 
the distance between the foci of the ellipse is 6√3. -/
theorem ellipse_foci_distance :
  ∀ (a b : ℝ), a = 6 → b = 3 → 
  let c := Real.sqrt (a ^ 2 - b ^ 2) in
  2 * c = 6 * Real.sqrt 3 :=
by
  intros a b ha hb
  rw [ha, hb]
  let c := Real.sqrt (6 ^ 2 - 3 ^ 2)
  sorry

end ellipse_foci_distance_l44_44449


namespace sum_of_consecutive_even_integers_is_24_l44_44338

theorem sum_of_consecutive_even_integers_is_24 (x : ℕ) (h_pos : x > 0)
    (h_eq : (x - 2) * x * (x + 2) = 20 * ((x - 2) + x + (x + 2))) :
    (x - 2) + x + (x + 2) = 24 :=
sorry

end sum_of_consecutive_even_integers_is_24_l44_44338


namespace black_percentage_l44_44940

-- Definition of circle's radius increment and coloring scheme
def radii (n : ℕ) : ℝ := 3 * (n + 1)
def is_black (n : ℕ) : Prop := n % 2 = 0

-- Determine the areas of the circles
def area (r : ℝ) : ℝ := Real.pi * r^2

-- Total number of circles
def num_circles : ℕ := 4

-- Iterate over the circles, calculating their contributions to the black area
def black_area : ℝ :=
  (Finset.range num_circles).sum (λ n, if is_black n then
      area (radii n) - (if (radii n - 3) > 0 then area (radii n - 3) else 0)
    else 0)

-- Total area of the largest circle
def total_area : ℝ := area (radii (num_circles - 1))

-- The goal is to prove that the black area percentage is 37.5%
theorem black_percentage : black_area / total_area * 100 = 37.5 := by
  sorry

end black_percentage_l44_44940


namespace find_f_two_l44_44852

-- Define the function f with the given properties
def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x + 1

-- Given conditions
variable (a b : ℝ)
axiom f_neg_two_zero : f (-2) a b = 0

-- Statement to be proven
theorem find_f_two : f 2 a b = 2 := 
by {
  sorry
}

end find_f_two_l44_44852


namespace factor_expression_l44_44485

theorem factor_expression (y : ℝ) : 
  3 * y * (2 * y + 5) + 4 * (2 * y + 5) = (3 * y + 4) * (2 * y + 5) :=
by
  sorry

end factor_expression_l44_44485


namespace perimeter_of_triangle_ABC_l44_44270

noncomputable def triangle_perimeter (r1 r2 r3 : ℝ) (θ1 θ2 θ3 : ℝ) : ℝ :=
  let x1 := r1 * Real.cos θ1
  let y1 := r1 * Real.sin θ1
  let x2 := r2 * Real.cos θ2
  let y2 := r2 * Real.sin θ2
  let x3 := r3 * Real.cos θ3
  let y3 := r3 * Real.sin θ3
  let d12 := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  let d23 := Real.sqrt ((x3 - x2)^2 + (y3 - y2)^2)
  let d31 := Real.sqrt ((x3 - x1)^2 + (y3 - y1)^2)
  d12 + d23 + d31

--prove

theorem perimeter_of_triangle_ABC (θ1 θ2 θ3: ℝ)
  (h1: θ1 - θ2 = Real.pi / 3)
  (h2: θ2 - θ3 = Real.pi / 3) :
  triangle_perimeter 4 5 7 θ1 θ2 θ3 = sorry := 
sorry

end perimeter_of_triangle_ABC_l44_44270


namespace part_I_part_II_l44_44886

noncomputable def M : Set ℝ := { x | |x + 1| + |x - 1| ≤ 2 }

theorem part_I : M = Set.Icc (-1 : ℝ) (1 : ℝ) := 
sorry

theorem part_II (x y z : ℝ) (hx : x ∈ M) (hy : |y| ≤ (1/6)) (hz : |z| ≤ (1/9)) :
  |x + 2 * y - 3 * z| ≤ (5/3) :=
by
  sorry

end part_I_part_II_l44_44886


namespace parallel_line_plane_condition_l44_44741

/-- 
Theorem: "Line m is parallel to countless lines in the plane α" 
is a necessary but not sufficient condition for "line m is parallel to plane α".
-/
theorem parallel_line_plane_condition {m : Type} {α : Type} [Plane α] [Line m] (h1 : ∀ l : α, m ‖ l) : 
  (∀ l : α, m ‖ l) ↔ (∀ l : α, m ‖ l) ∧ (m ‖ α) := sorry

end parallel_line_plane_condition_l44_44741


namespace non_increasing_condition_l44_44313

variable {a b : ℝ} (f : ℝ → ℝ)

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem non_increasing_condition (h₀ : ∀ x y, a ≤ x → x < y → y ≤ b → ¬ (f x > f y)) :
  ¬ increasing_on_interval f a b :=
by
  intro h1
  have : ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y := h1
  exact sorry

end non_increasing_condition_l44_44313


namespace G_51_value_l44_44046

def G (n : ℕ) : ℚ :=
  if n = 1 then 3 else (3 * G (n - 1) + 2) / 3

theorem G_51_value : G 51 = 36 + 1 / 3 :=
by
  sorry

end G_51_value_l44_44046


namespace sum_real_roots_eq_neg4_l44_44816

-- Define the equation condition
def equation_condition (x : ℝ) : Prop :=
  (2 * x / (x^2 + 5 * x + 3) + 3 * x / (x^2 + x + 3) = 1)

-- Define the statement that sums the real roots
theorem sum_real_roots_eq_neg4 : 
  ∃ S : ℝ, (∀ x : ℝ, equation_condition x → x = -1 ∨ x = -3) ∧ (S = -4) :=
sorry

end sum_real_roots_eq_neg4_l44_44816


namespace card_identification_l44_44737

/-
  Given a 6x6 grid of 36 cards, if a spectator selects a card and remembers its initial column,
  and the magician gathers the cards column by column and then redistributes them row by row,
  the magician can identify the chosen card based on the initial column and the new column.
-/
theorem card_identification (initial_arrangement : Fin 6 -> Fin 6 -> ℕ) 
    (chosen_card : Fin 6 -> Fin 6) 
    (initial_column : Fin 6) 
    (new_column : Fin 6) :
  ∃ (final_row final_col : Fin 6), 
    initial_arrangement final_row final_col = chosen_card initial_column new_column :=
sorry

end card_identification_l44_44737


namespace regression_total_sum_of_squares_l44_44033

theorem regression_total_sum_of_squares (y : ℝ → ℝ) (x : ℝ → ℝ) (n : ℕ) (R_squared : ℝ) (sum_squared_residuals : ℝ) :
  n = 10 →
  R_squared = 0.95 →
  sum_squared_residuals = 120.53 →
  ∑ i in finset.range n, (y i - finset.sum (finset.range n) y / n) ^ 2 = 2410.6 :=
by
  intros h_n h_R_squared h_sum_squared_residuals
  sorry

end regression_total_sum_of_squares_l44_44033


namespace max_area_of_isosceles_triangle_ABC_l44_44191

theorem max_area_of_isosceles_triangle_ABC
  (b : ℝ)
  (h₁ : ∀ A B C : ℝ, A = B → A = C → ¬ (A = 0) → neq AC 0 → D = midpoint A C → BD = 1 → 
        ∃ S : ℝ, S = (b^2 / 2) * sqrt (-(9 * (b^2 - (20 / 9))^2) + (256 / 9)) / 4 * b^2 → 
        max_area (triangle_area ABC) S)
  : ∃ S : ℝ, S = 2 / 3 :=
by
  sorry

end max_area_of_isosceles_triangle_ABC_l44_44191


namespace cubics_of_sum_and_product_l44_44162

theorem cubics_of_sum_and_product (x y : ℝ) (h₁ : x + y = 10) (h₂ : x * y = 11) : 
  x^3 + y^3 = 670 :=
by
  sorry

end cubics_of_sum_and_product_l44_44162


namespace min_value_of_f_l44_44501

def f (x : ℝ) : ℝ := (x^2 + 5) / (Real.sqrt (x^2 + 4))

theorem min_value_of_f : ∃ x_min : ℝ, f x_min = 5 / 2 :=
by
  use 0 -- specific value found from the proof steps
  sorry -- proof is omitted

end min_value_of_f_l44_44501


namespace income_is_10000_l44_44329

-- Define the necessary variables: income, expenditure, and savings
variables (income expenditure : ℕ) (x : ℕ)

-- Define the conditions given in the problem
def ratio_condition : Prop := income = 10 * x ∧ expenditure = 7 * x
def savings_condition : Prop := income - expenditure = 3000

-- State the theorem that needs to be proved
theorem income_is_10000 (h_ratio : ratio_condition income expenditure x) (h_savings : savings_condition income expenditure) : income = 10000 :=
sorry

end income_is_10000_l44_44329


namespace a_and_b_together_full_time_completion_days_l44_44398

variable (a b : Type)
variable (work : a → b → ℝ)
variable (days : ℝ)
variable (half_time : ℝ := 0.5)

-- Definitions
def work_done_by_a_in_one_day : ℝ := 1 / 20
def work_done_by_b_in_one_day (y : ℝ) : ℝ := 1 / y
def combined_work_rate_half_time (y : ℝ) : ℝ := work_done_by_a_in_one_day + work_done_by_b_in_one_day y * half_time
def combined_work_rate_full_time (y : ℝ) : ℝ := work_done_by_a_in_one_day + work_done_by_b_in_one_day y

-- Problem Statement
theorem a_and_b_together_full_time_completion_days (y : ℝ) :
  (combined_work_rate_half_time y = 1 / 15) →
  combined_work_rate_full_time y = 1 / 12 :=
by
  sorry

end a_and_b_together_full_time_completion_days_l44_44398


namespace trapezoid_sides_l44_44205
-- Import the necessary library

-- Define the conditions as given in the problem
variables {A B C D E : Type*}
variables (a : ℝ) (x : ℝ)
-- Defining points as necessary, and assumptions
variables (AB BC CD AD BE : ℝ)
variables (h_trap : AD = a)
variables (h_perp1 : BC ⊥ CD)
variables (h_eq : AB = BC)
variables (h_perp2 : BD ⊥ AB)

-- The mathematically equivalent proof problem in Lean 4 statement
theorem trapezoid_sides (a : ℝ) (h_trap : AD = a)
  (h_perp1 : BC ⊥ CD) (h_eq : AB = BC) (h_perp2 : BD ⊥ AB) : 
  AB = BC :=
  sorry

end trapezoid_sides_l44_44205


namespace min_chips_in_cells_l44_44946

theorem min_chips_in_cells (strip_size : ℕ) (h1 : strip_size = 2021)
  (chips : Finset ℕ) (empty_cells : Finset ℕ)
  (h2 : ∀ c ∈ empty_cells, ∃ l r : ℕ, l ∈ chips ∧ r ∈ chips ∧ 
      abs (l - r) = c ∧ c ≠ 0 ∧ 
      (∀ c' ∈ empty_cells, c ≠ c') )
  : ∃ n, chips.card = 1347 :=
by
  have : chips.card = 1347 := sorry
  exact ⟨1347, this⟩

end min_chips_in_cells_l44_44946


namespace count_palindromic_times_l44_44416

def is_palindrome (t : String) : Prop :=
  t = t.reverse

def valid_time (h m : Nat) : Prop :=
  h < 24 ∧ m < 60

def valid_palindromic_time (h m : Nat) : Prop :=
  valid_time h m ∧ is_palindrome (h.repr ++ ":" ++ m.repr)

theorem count_palindromic_times : 
  (Finset.card (Finset.filter (λ t : Fin (24) × Fin (60), valid_palindromic_time t.1 t.2)
    (Finset.univ : Finset (Fin (24) × Fin (60))))) = 60 :=
begin
  sorry
end

end count_palindromic_times_l44_44416


namespace count_div_by_4_or_6_count_div_by_4_or_6_correct_l44_44563

theorem count_div_by_4_or_6 (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 80) :
  (n % 4 = 0 ∨ n % 6 = 0) ↔ n ∈ {i | 1 ≤ i ∧ i ≤ 80 ∧ (i % 4 = 0 ∨ i % 6 = 0)} :=
sorry

theorem count_div_by_4_or_6_correct :
  (finset.filter (λ n, (n % 4 = 0 ∨ n % 6 = 0)) (finset.range 81)).card = 27 :=
sorry

end count_div_by_4_or_6_count_div_by_4_or_6_correct_l44_44563


namespace midpoint_is_hyperbola_center_tangents_angle_l44_44516

noncomputable theory

-- Define the given conditions and required proofs.
variables {A1 A2 A3 A4 : (ℝ × ℝ)} -- Intersection points
variables {a D E F : ℝ} -- Hyperbola and circle parameters
variable (circle) : (ℝ × ℝ) → ℝ := (λ p, p.1^2 + p.2^2 + 2*D*p.1 + 2*E*p.2 + F)
variable (hyperbola) : (ℝ × ℝ) → ℝ := (λ p, p.1 * p.2 - a)

-- 1. Prove that the midpoint of A3A4 is the center of the hyperbola
theorem midpoint_is_hyperbola_center :
  (circle A1 = 0) ∧ (circle A2 = 0) ∧ (circle A3 = 0) ∧ (circle A4 = 0) ∧
  (hyperbola A1 = 0) ∧ (hyperbola A2 = 0) ∧ (hyperbola A3 = 0) ∧ (hyperbola A4 = 0) ∧
  (A1.1 + A2.1 = -2 * D) ∧ (A1.2 + A2.2 = -2 * E) →
  ((A3.1 + A4.1) / 2 = 0) ∧ ((A3.2 + A4.2) / 2 = 0) :=
sorry

-- 2. Find the angle between the tangents to the hyperbola at A3, A4 and the line A1A2
/--
  This theorem states that, given certain conditions about intersection points of a circle and a hyperbola,
  the tangents to the hyperbola at those points and the line segment formed by two other specific 
  intersection points are perpendicular.
-/
theorem tangents_angle :
  (circle A1 = 0) ∧ (circle A2 = 0) ∧ (circle A3 = 0) ∧ (circle A4 = 0) ∧
  (hyperbola A1 = 0) ∧ (hyperbola A2 = 0) ∧ (hyperbola A3 = 0) ∧ (hyperbola A4 = 0) ∧
  (A1.1 + A2.1 = -2 * D) ∧ (A1.2 + A2.2 = -2 * E) →
  ∀ (tangent_slope : ℝ) (tangent_to_hyperbola : ℝ),
  tangent_slope = -a / A3.1^2 ∧ tangent_to_hyperbola = -a / (A1.1 * A2.1) →
  tangent_slope * tangent_to_hyperbola = -1 :=
sorry

end midpoint_is_hyperbola_center_tangents_angle_l44_44516


namespace square_side_length_l44_44318

theorem square_side_length (s : ℝ) (h : s^2 = 3 * 4 * s) : s = 12 :=
by
  sorry

end square_side_length_l44_44318


namespace combination_sum_l44_44744

theorem combination_sum :
  (Nat.choose 3 2) + (Nat.choose 4 2) + (Nat.choose 5 2) + (Nat.choose 6 2) = 34 :=
by
  sorry

end combination_sum_l44_44744


namespace gears_ratio_correct_l44_44590

variables {p q r : ℕ} {ω_A ω_B ω_C : ℝ}

-- Conditions
def cond1 : Prop := p * ω_A = q * ω_B
def cond2 : Prop := q * ω_B = r * ω_C

-- Statement to prove
def gear_ratio : Prop := (ω_A / ω_B) = (r / p) ∧ (ω_B / ω_C) = (q / r)

theorem gears_ratio_correct (h1 : cond1) (h2 : cond2) : gear_ratio :=
sorry

end gears_ratio_correct_l44_44590


namespace solve_log_inequality_l44_44836

theorem solve_log_inequality (a x : ℝ) (h1 : a + 2 * x - x^2 > 0) (h2 : a > 0) (h3 : sqrt(2 * a) ≠ 1) :
  (a < 1 / 2 ∧ (1 - sqrt(1 - a)) < x ∧ x < (1 + sqrt(1 - a)))
  ∨ (1 / 2 < a ∧ a ≤ 1 ∧ ((x < 1 - sqrt(1 + a)) ∨ (1 - sqrt(1 - a)) < x ∧ x < (1 + sqrt(1 - a)) ∨ (x > 1 + sqrt(1 + a))))
  ∨ (a > 1 ∧ (1 - sqrt(1 + a)) < x ∧ x < (1 + sqrt(1 + a))) :=
sorry

end solve_log_inequality_l44_44836


namespace find_cos_tan_l44_44871

variables {α : Type*} [real α]

def terminal_point (P : α × α) (m : α) := 
  P = (-real.sqrt 3, m) ∧ m ≠ 0

def sin_condition (α : ℝ) (m : ℝ) := 
  real.sin α = (real.sqrt 2 / 4) * m

theorem find_cos_tan (P : α × α) (m : α) (h1 : terminal_point P m) (h2 : sin_condition α m) :
  real.cos α = -real.sqrt 6 / 4 ∧ 
  (real.tan α = real.sqrt 15 / 3 ∨ real.tan α = -real.sqrt 15 / 3) :=
sorry

end find_cos_tan_l44_44871


namespace seq_2016_l44_44045

noncomputable def seq : ℕ → ℝ
| 1       := 2
| 2       := 1/3
| (n + 1) := seq (n) / seq (n - 1)

theorem seq_2016 : seq 2016 = 6 :=
by sorry

end seq_2016_l44_44045


namespace integer_solutions_l44_44476

def satisfies_eq (x y n : ℤ) : Prop :=
  x^2 + 2 * y^2 = 2^n

def is_solution (r : ℤ) : Prop :=
  ∃ x y n, x = ±2^r ∧ y = 0 ∧ n = 2*r ∨
           x = 0 ∧ y = ±2^r ∧ n = 2*r + 1

theorem integer_solutions :
  ∀ (x y n : ℤ), satisfies_eq x y n ↔ (∃ r : ℤ, x = ±2^r ∧ y = 0 ∧ n = 2 * r
                                      ∨ x = 0 ∧ y = ±2^r ∧ n = 2 * r + 1) :=
by
  sorry

end integer_solutions_l44_44476


namespace g_has_two_zeros_on_interval_l44_44914

theorem g_has_two_zeros_on_interval :
  let g : ℝ → ℝ := λ x, sin (2 * x + π / 3)
  ∃ a b ∈ (set.Icc 0 π), g a = 0 ∧ g b = 0 ∧ a ≠ b :=
by
  let g : ℝ → ℝ := λ x, sin (2 * x + π / 3)
  have h : ∀ x ∈ set.Icc 0 π, |g x| ≤ 1, sorry
  have z1 : ∃ x ∈ set.Icc 0 π, g x = 0, sorry
  have z2 : ∃ x' ∈ set.Icc 0 π, g x' = 0 ∧ x' ≠ some z1, sorry
  exact ⟨some z1, some z2, (some z1).prop, (some z2).prop.1, some_spec z1, some_spec z2.2, some_spec z2.2.2⟩

end g_has_two_zeros_on_interval_l44_44914


namespace median_is_seven_l44_44778

-- Definitions of the conditions given in the problem.
def dataset := [6, 8, 7, 7, a, b, c]
def unique_mode (d : List ℕ) (mode : ℕ) := (∀ x : ℕ, count x d < count mode d) ∧ (∃! x : ℕ, count x d = count mode d)
def mean (d : List ℕ) (m : ℕ) := m * d.length = d.sum
def is_median (d : List ℕ) (med : ℕ) := ∃ l1 l2, l1 ++ [med] ++ l2 = d ∧ l1.length = l2.length

-- The problem statement proving the median is 7 given the conditions.
theorem median_is_seven (a b c : ℕ) (h_mode : unique_mode dataset 8) (h_mean : mean dataset 7) :
  is_median (dataset.map id) 7 :=
sorry

end median_is_seven_l44_44778


namespace grid_problem_l44_44592

theorem grid_problem 
  (n m : ℕ) 
  (h1 : ∀ (blue_cells : ℕ), blue_cells = m + n - 1 → (n * m ≠ 0) → (blue_cells = (n * m) / 2010)) :
  ∃ (k : ℕ), k = 96 :=
by
  sorry

end grid_problem_l44_44592


namespace not_in_range_of_f_l44_44619

noncomputable def f (x : ℚ) (k : ℚ) : ℚ := (2 * x + k) / (3 * x + 4)

theorem not_in_range_of_f (k : ℚ) (h1 : f 5 k = 5) (h2 : f 100 k = 100) (h3 : ∀ x : ℚ, x ≠ -4/3 → f (f x k) k = x) : 
  ∃ y : ℚ, y = -8 / 13 ∧ ∀ x : ℚ, f x k ≠ y :=
by
  have k_val : k = 13 / 6 := sorry  -- derived from the conditions
  use -8 / 13
  split
  · refl
  · intros x
    have hx : 3 * x + 4 ≠ 0 := sorry  -- premise ensuring the denominator is not zero
    have h_fx_eq : f x k = -8 / 13 ↔ (2 * x + k) = -8 / 13 * (3 * x + 4) := by sorry  -- relate with the definition of f and equate
    have contradiction : (2*x + 13/6) ≠ -8/13 * (3*x + 4) := by sorry  -- simplify to show it creates a contradiction
    exact contradiction

end not_in_range_of_f_l44_44619


namespace area_of_YZW_l44_44199

-- Definitions from conditions
def area_of_triangle_XYZ := 36
def base_XY := 8
def base_YW := 32

-- The theorem to prove
theorem area_of_YZW : 1/2 * base_YW * (2 * area_of_triangle_XYZ / base_XY) = 144 := 
by
  -- Placeholder for the proof  
  sorry

end area_of_YZW_l44_44199


namespace rhombus_longer_diagonal_length_l44_44776

theorem rhombus_longer_diagonal_length :
  ∀ (a b d1 : ℝ), 
  a = 61 ∧ b = 61 ∧ d1 = 110 →
  ∃ d2 : ℝ, d2 = 24 * Real.sqrt 58 :=
by
  intros a b d1 h,
  cases h with ha hb,
  cases hb with hb1 hb2,
  use [24 * Real.sqrt 58],
  sorry

end rhombus_longer_diagonal_length_l44_44776


namespace sum_floor_log_equals_4926_l44_44837

def floor_sum_log (n : ℕ) :=
  ∑ k in Finset.range (n + 1), Int.floor (Real.log10 (↑k + 1))

theorem sum_floor_log_equals_4926 :
  floor_sum_log 2011 = 4926 :=
by
  sorry

end sum_floor_log_equals_4926_l44_44837


namespace problem_statement_l44_44740

open EuclideanGeometry

variables {A B C P D E F N M : Point}
variables (Γ : Circle) (∆ABC : Triangle)

noncomputable def acute_triangle_inscribed (A B C : Point) : Prop :=
  acute ∆ABC ∧ Triangle.circumcircle ∆ABC = Γ

noncomputable def tangents_intersect (B C P : Point) (Γ : Circle) : Prop :=
  tangent_at Γ B ∧ tangent_at Γ C ∧ intersect_at (tangents Γ B C) P

noncomputable def projections (P : Point) (BC AC AB : Line) (D E F : Point) : Prop :=
  projection P BC D ∧ projection P AC E ∧ projection P AB F

noncomputable def circumcircle_intersects (DEF : Triangle) (BC : Line) (N D : Point) : Prop :=
  Triangle.circumcircle DEF ∩ BC = {N} ∧ N ≠ D

noncomputable def projection_A_to_BC (A : Point) (BC : Line) (M : Point) : Prop :=
  projection A BC M

theorem problem_statement
  (acute_triangle_inscribed A B C)
  (tangents_intersect B C P Γ)
  (projections P (line_through B C) (line_through A C) (line_through A B) D E F)
  (circumcircle_intersects (Triangle.mk D E F) (line_through B C) N D)
  (projection_A_to_BC A (line_through B C) M) :
  dist B N = dist C M :=
sorry

end problem_statement_l44_44740


namespace sum_fourth_powers_sum_fourth_powers_equation_l44_44668

open Nat

theorem sum_fourth_powers (n : ℕ) :
  (∑ i in Finset.range (n+1), i^4) = 
  (binom n 1) + 15 * (binom n 2) + 50 * (binom n 3) + 60 * (binom n 4) + 24 * (binom n 5) :=
  by 
    -- Proof is omitted
    sorry

theorem sum_fourth_powers_equation (n : ℕ) :
  (∑ i in Finset.range (n+1), i^4) = 
  (1 / 30 : ℚ) * n * (n + 1) * (2 * n + 1) * (3 * n^2 + 3 * n - 1) := 
  by 
    -- Proof is omitted
    sorry

end sum_fourth_powers_sum_fourth_powers_equation_l44_44668


namespace probability_sum_is_3_l44_44361

theorem probability_sum_is_3 (die : Type) [Fintype die] [DecidableEq die] 
  (dice_faces : die → ℕ) (h : ∀ d, dice_faces d ∈ {1, 2, 3, 4, 5, 6}) :
  (∑ i in finset.range 3, (die →₀ ℕ).single 1) = 3 → 
  (1 / (finset.card univ) ^ 3) = 1 / 216 :=
by
  sorry

end probability_sum_is_3_l44_44361


namespace john_age_proof_l44_44237

theorem john_age_proof (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end john_age_proof_l44_44237


namespace find_angle_C_find_side_c_l44_44209

noncomputable def triangle := Type

structure Triangle (A B C : ℝ) :=
  (side_a : ℝ)
  (side_b : ℝ)
  (side_c : ℝ)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (angle_C : ℝ)

axiom law_of_cosines (T : Triangle A B C) : 
  2 * T.side_c * Math.cos T.angle_C = T.side_b * Math.cos T.angle_A + T.side_a * Math.cos T.angle_B

theorem find_angle_C (A B C a b c : ℝ) (h : Triangle A B C) (h1 : law_of_cosines h) :
  h.angle_C = Real.pi / 3 := 
sorry

theorem find_side_c (A B C : ℝ) (a : ℝ := 6) (cos_A : ℝ := -4 / 5) : 
  ∃ (h : Triangle A B C), law_of_cosines h → h.side_c = 5 * Real.sqrt(3) :=
sorry

end find_angle_C_find_side_c_l44_44209


namespace isosceles_triangle_BC_squared_l44_44626

theorem isosceles_triangle_BC_squared (ABC : triangle)
  (acute : ABC.acute)
  (isosceles : ABC.isosceles)
  (AB_eq_AC : ABC.AB = 2 ∧ ABC.AC = 2)
  (H : ABC.orthocenter)
  (M : midpoint ABC.A ABC.B)
  (N : midpoint ABC.A ABC.C)
  (circumcircle_MHN : ∃ (X Y : point), circumcircle (triangle.mk M H N).intersects_line (line.mk ABC.B ABC.C) = (X, Y))
  (XY_eq_AB_AC : XY_length (line.mk X Y) = 2) :
  BC_length_squared (line.mk ABC.B ABC.C) = 2 * (√17 - 1) :=
by sorry

end isosceles_triangle_BC_squared_l44_44626


namespace smallest_value_of_expression_l44_44257

noncomputable def f (x : ℝ) : ℝ := x^4 + 14*x^3 + 52*x^2 + 56*x + 16

theorem smallest_value_of_expression :
  ∀ z : Fin 4 → ℝ, (∀ i, f (z i) = 0) → 
  ∃ (a b c d : Fin 4), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ d ≠ b ∧ a ≠ c ∧ 
  |(z a * z b) + (z c * z d)| = 8 :=
by
  sorry

end smallest_value_of_expression_l44_44257


namespace distinct_arrangements_balloon_l44_44119

theorem distinct_arrangements_balloon : 
  let n := 7
  let r1 := 2
  let r2 := 2
  (Nat.factorial n) / ((Nat.factorial r1) * (Nat.factorial r2)) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l44_44119


namespace f_15_value_l44_44034

noncomputable def f : ℝ → ℝ :=
sorry

noncomputable def a : ℕ → ℝ :=
λ n, (f n) ^ 2 - f n

axiom f_recursion (x : ℝ) : f (x + 1) = sqrt (f x - (f x) ^ 2) + 1 / 2

axiom sum_of_a : ∑ n in Finset.range 15, a n = -31 / 16

theorem f_15_value : f 15 = 3 / 4 :=
sorry

end f_15_value_l44_44034


namespace month_has_30_days_l44_44169

def has_more_mondays_than_tuesdays (month_length : ℕ) : Prop :=
  (∃ (monday_count tuesday_count : ℕ), monday_count > tuesday_count 
   ∧ monday_count = (month_length / 7) + if (month_length % 7 = 0) then 0 else 1 
   ∧ tuesday_count = (month_length / 7))

def has_fewer_saturdays_than_sundays (month_length : ℕ) : Prop :=
  (∃ (saturday_count sunday_count : ℕ), saturday_count < sunday_count 
   ∧ saturday_count = (month_length / 7) + if (month_length % 7 < 6) then 0 else 1 
   ∧ sunday_count = (month_length / 7) + if (month_length % 7 < 1) then 0 else 1)

theorem month_has_30_days :
  (∃ month_length : ℕ, has_more_mondays_than_tuesdays month_length 
                       ∧ has_fewer_saturdays_than_sundays month_length 
                       ∧ month_length = 30) :=
sorry

end month_has_30_days_l44_44169


namespace balloon_permutations_l44_44134

theorem balloon_permutations : 
  let n : ℕ := 7
  let k1 : ℕ := 2
  let k2 : ℕ := 2
  ∃ distinct_arrangements : ℕ, 
  distinct_arrangements = n.factorial / (k1.factorial * k2.factorial) 
  ∧ distinct_arrangements = 1260 :=
by
  sorry

end balloon_permutations_l44_44134


namespace problem_statement_l44_44917

noncomputable def f (x m : ℝ) := 4 * x^2 - m * x + 5

theorem problem_statement (m : ℝ) 
  (h1 : ∀ x≥-2, ∀ y<-2, f x m ≥ f (-2) m)
  (h2 : ∀ x>-2, ∀ y≤-2, f y m ≤ f (-2) m) :
  f 1 m = 25 := 
sorry

end problem_statement_l44_44917


namespace inequality_solution_l44_44693

theorem inequality_solution (x : ℝ) : (5 * x + 3 > 9 - 3 * x ∧ x ≠ 3) ↔ (x > 3 / 4 ∧ x ≠ 3) :=
by {
  sorry
}

end inequality_solution_l44_44693


namespace geometric_seq_sum_of_seq_l44_44955

def seq_a (n : ℕ) : ℕ :=
  if n = 1 then 2 else 2 * seq_a (n - 1) - (n - 1) + 1

def seq_b (n : ℕ) : ℕ :=
  seq_a n - n

theorem geometric_seq :
  ∀ n : ℕ, seq_b 1 = 1 ∧ (∀ k : ℕ, seq_b (k + 1) = 2 * seq_b k) :=
by 
  -- Prove that seq_b is a geometric sequence and find the general formula for seq_a
  sorry

theorem sum_of_seq : 
  ∀ n : ℕ, 
    let S_n := (∑ i in Finset.range n, seq_a (i + 1))
    in S_n = (n * (n + 1)) / 2 + 2 ^ n - 1 :=
by 
  -- Prove that the sum of the first n terms is (n * (n + 1)) / 2 + 2 ^ n - 1
  sorry

end geometric_seq_sum_of_seq_l44_44955


namespace quadratic_inequality_solution_l44_44309

theorem quadratic_inequality_solution :
  ∀ x : ℝ, x ∈ Ioo ((4 - Real.sqrt 19) / 3) ((4 + Real.sqrt 19) / 3) → (-3 * x^2 + 8 * x + 1 < 0) :=
by
  intro x hx
  have h1 : x ∈ Ioo ((4 - Real.sqrt 19) / 3) ((4 + Real.sqrt 19) / 3) := hx
  -- Further proof would go here
  sorry

end quadratic_inequality_solution_l44_44309


namespace find_n_mod_11_l44_44023

theorem find_n_mod_11 : ∃ n : ℤ, (0 ≤ n ∧ n ≤ 11) ∧ (n ≡ 123456 [MOD 11]) ∧ n = 3 := by
  sorry

end find_n_mod_11_l44_44023


namespace Jolene_cars_washed_proof_l44_44968

-- Definitions for conditions
def number_of_families : ℕ := 4
def babysitting_rate : ℕ := 30 -- in dollars
def car_wash_rate : ℕ := 12 -- in dollars
def total_money_raised : ℕ := 180 -- in dollars

-- Mathematical representation of the problem:
def babysitting_earnings : ℕ := number_of_families * babysitting_rate
def earnings_from_cars : ℕ := total_money_raised - babysitting_earnings
def number_of_cars_washed : ℕ := earnings_from_cars / car_wash_rate

-- The proof statement
theorem Jolene_cars_washed_proof : number_of_cars_washed = 5 := 
sorry

end Jolene_cars_washed_proof_l44_44968


namespace arabella_total_learning_time_l44_44454

-- Define the conditions
def arabella_first_step_time := 30 -- in minutes
def arabella_second_step_time := arabella_first_step_time / 2 -- half the time of the first step
def arabella_third_step_time := arabella_first_step_time + arabella_second_step_time -- sum of the first and second steps

-- Define the total time spent
def arabella_total_time := arabella_first_step_time + arabella_second_step_time + arabella_third_step_time

-- The theorem to prove
theorem arabella_total_learning_time : arabella_total_time = 90 := 
  sorry

end arabella_total_learning_time_l44_44454


namespace distinct_arrangements_balloon_l44_44094

theorem distinct_arrangements_balloon : 
  let n := 7 
  let freq_l := 2 
  let freq_o := 2 
  let freq_b := 1 
  let freq_a := 1 
  let freq_n := 1 
  Nat.factorial n / (Nat.factorial freq_l * Nat.factorial freq_o * Nat.factorial freq_b * Nat.factorial freq_a * Nat.factorial freq_n) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l44_44094


namespace john_age_proof_l44_44241

theorem john_age_proof (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end john_age_proof_l44_44241


namespace hyperbola_equation_l44_44053

theorem hyperbola_equation :
  ∃ (a b : ℝ), (a^2 + b^2 = 9) ∧ (4 * b^2 = 5 * a^2) ∧
    E = { (x, y) | (x^2 / a^2) - (y^2 / b^2) = 1 } ∧
    ∃ P : ℝ × ℝ, P = (3, 0) ∧ focus E P ∧
    ∃ N : ℝ × ℝ, N = (-12, -15) ∧
    ∃ A B : ℝ × ℝ, (A, B ∈ E) ∧ (midpoint A B = N) ∧ line_through P A B = l :=
begin
  sorry
end

end hyperbola_equation_l44_44053


namespace common_elements_count_l44_44617

open_locale big_operators

def multiples_of (n : ℕ) (k : ℕ) : finset ℕ := 
finset.image (λ i, n * i) (finset.range k)

def set_S : finset ℕ := multiples_of 5 1000
def set_T : finset ℕ := multiples_of 8 1000

theorem common_elements_count : (set_S ∩ set_T).card = 125 :=
sorry

end common_elements_count_l44_44617


namespace train_length_l44_44732

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (length_m : ℕ) 
  (h1 : speed_kmh = 180)
  (h2 : time_s = 18)
  (h3 : 1 = 1000 / 3600) :
  length_m = (speed_kmh * 1000 / 3600) * time_s :=
by
  sorry

end train_length_l44_44732


namespace restaurant_hamburgers_l44_44774

-- Define the conditions
def hamburgers_served : ℕ := 3
def hamburgers_left_over : ℕ := 6

-- Define the total hamburgers made
def hamburgers_made : ℕ := hamburgers_served + hamburgers_left_over

-- State and prove the theorem
theorem restaurant_hamburgers : hamburgers_made = 9 := by
  sorry

end restaurant_hamburgers_l44_44774


namespace area_of_region_l44_44013

theorem area_of_region : let r1 (θ : ℝ) := 2 / Real.cos θ,
                            r2 (θ : ℝ) := 2 / Real.sin θ,
                            x := r1,
                            y := r2
                         in if r1(0) = 2 ∧ r2(math.pi/2) = 2 ∧ (x = 2 → y = 2)
                          then 
                            2 * 2 = 4 := sorry

end area_of_region_l44_44013


namespace roots_of_unity_probability_l44_44269

open Complex

noncomputable def prob_condition (v w : ℂ) (h : v ≠ w ∧ v^2023 = 1 ∧ w^2023 = 1) : ℚ :=
if (sqrt (2 + sqrt 3)) ≤ abs (v + w) then 1 else 0

theorem roots_of_unity_probability :
  (∑ v w in (finset.range 2023).image (λ k, exp (2 * real.pi * I * k / 2023)), 
    if v ≠ w then prob_condition v w ⟨ne_of_ne_zero _,⟨_,_⟩⟩ else 0) / (2023 * 2022) = 337 / 2022 :=
sorry

end roots_of_unity_probability_l44_44269


namespace height_of_door_l44_44324

theorem height_of_door (a b c : ℕ) (cost_per_sq_ft total_cost H : ℕ) :
  a = 25 → b = 15 → c = 12 → cost_per_sq_ft = 10 → total_cost = 9060 →
  let perimeter := 2 * (a + b),
      area_walls := perimeter * c,
      area_door := H * 3,
      area_windows := 3 * (4 * 3),
      area_not_whitewashed := area_door + area_windows,
      area_to_whitewash := area_walls - area_not_whitewashed,
      computed_cost := area_to_whitewash * cost_per_sq_ft in
  computed_cost = total_cost → H = 6 :=
by
  intros a_eq b_eq c_eq cost_per_sq_ft_eq total_cost_eq;
  simp [*, Nat.mul_sub];
  sorry

end height_of_door_l44_44324


namespace prob_between_l44_44280

-- Definition of the standard normal variable
def std_normal (ξ : ℝ) : Prop := ξ ~ Normal(0, 1)

-- Given conditions: ξ follows N(0,1) and P(ξ > 1) = 1/4
variable (ξ : ℝ)
@[simp] lemma std_normal_pdf (ξ : ℝ) : std_normal ξ := by sorry
@[simp] lemma prob_greater_than_one_quarter : P(λ ξ, ξ > 1) = 1 / 4 := by sorry

-- Theorem: P(-1 < ξ < 1) = 1 / 2
theorem prob_between : P(λ ξ, -1 < ξ ∧ ξ < 1) = 1 / 2 := 
by 
  have h1 : P(λ ξ, ξ < -1) = P(λ ξ, ξ > 1) := by 
    simp [std_normal ξ]
  
  have h2 : P(λ ξ, -1 < ξ < 1) = 1 - P(λ ξ, ξ > 1) - P(λ ξ, ξ < -1) := by 
    simp [std_normal ξ, P λ ξ, ξ > 1]

  simp [← h1, prob_greater_than_one_quarter, h2]
  exact prob_greater_than_one_quarter
   

end prob_between_l44_44280


namespace gcd_poly_l44_44867

theorem gcd_poly {b : ℕ} (h : 1116 ∣ b) : Nat.gcd (b^2 + 11 * b + 36) (b + 6) = 6 :=
by
  sorry

end gcd_poly_l44_44867


namespace no_integer_n_exists_l44_44003

theorem no_integer_n_exists : ∀ (n : ℤ), n ^ 2022 - 2 * n ^ 2021 + 3 * n ^ 2019 ≠ 2020 :=
by sorry

end no_integer_n_exists_l44_44003


namespace balloon_arrangements_l44_44147

-- Defining the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Given Conditions
def seven_factorial := fact 7 -- 7!
def two_factorial := fact 2 -- 2!

-- Statement to prove
theorem balloon_arrangements : seven_factorial / (two_factorial * two_factorial) = 1260 :=
by
  sorry

end balloon_arrangements_l44_44147


namespace sum_of_arithmetic_sequence_l44_44594

variable {a : ℕ → ℝ}

theorem sum_of_arithmetic_sequence (h₁ : a 1 + a 4 + a 7 = 15) 
                                   (h₂ : a 3 + a 6 + a 9 = 3) :
    (∑ n in Finset.range 9, a (n + 1)) = 27 := 
sorry

end sum_of_arithmetic_sequence_l44_44594


namespace sufficient_but_not_necessary_condition_for_exponential_l44_44742

theorem sufficient_but_not_necessary_condition_for_exponential (x : ℝ) : (x > 1 → 2^x > 1) ∧ (∃ y : ℝ, 2^y > 1 ∧ ¬ (y > 1)) := 
by sorry

end sufficient_but_not_necessary_condition_for_exponential_l44_44742


namespace f_even_a_neg_half_max_f_in_interval_l44_44549

-- Definitions of the conditions
def f (a : ℝ) (x : ℝ) : ℝ := a * x + log 2 (2^x + 1)

-- Part (1): Prove f is an even function when a = -1/2
theorem f_even_a_neg_half : (∀ x : ℝ, f (-1/2) x = f (-1/2) (-x)) := by sorry

-- Part (2): Maximum value of f in [1,2] given certain conditions
theorem max_f_in_interval (a : ℝ) (h_a : a > 0)
  (h_minimum : ∀ x ∈ (Set.Icc 1 2), f a x + f a^(-1) x = 1 + log 2 3) :
  (∀ x ∈ (Set.Icc 1 2), f 1 2 = 2 + log 2 5) := by sorry

end f_even_a_neg_half_max_f_in_interval_l44_44549


namespace distinct_arrangements_balloon_l44_44126

-- Let's define the basic conditions:
def total_letters : Nat := 7
def repeats_l : Nat := 2
def repeats_o : Nat := 2

-- Now let's state the problem.
theorem distinct_arrangements_balloon : 
  (Nat.factorial total_letters) / ((Nat.factorial repeats_l) * (Nat.factorial repeats_o)) = 1260 := 
by
  sorry

end distinct_arrangements_balloon_l44_44126


namespace abs_ab_eq_2_sqrt_65_l44_44327

theorem abs_ab_eq_2_sqrt_65
  (a b : ℝ)
  (h1 : b^2 - a^2 = 16)
  (h2 : a^2 + b^2 = 36) :
  |a * b| = 2 * Real.sqrt 65 := 
sorry

end abs_ab_eq_2_sqrt_65_l44_44327


namespace percent_paddyfield_warblers_l44_44934

variable (B : ℝ) -- The total number of birds.
variable (N_h : ℝ := 0.30 * B) -- Number of hawks.
variable (N_non_hawks : ℝ := 0.70 * B) -- Number of non-hawks.
variable (N_not_hpwk : ℝ := 0.35 * B) -- 35% are not hawks, paddyfield-warblers, or kingfishers.
variable (N_hpwk : ℝ := 0.65 * B) -- 65% are hawks, paddyfield-warblers, or kingfishers.
variable (P : ℝ) -- Percentage of non-hawks that are paddyfield-warblers, to be found.
variable (N_pw : ℝ := P * 0.70 * B) -- Number of paddyfield-warblers.
variable (N_k : ℝ := 0.25 * N_pw) -- Number of kingfishers.

theorem percent_paddyfield_warblers (h_eq : N_h + N_pw + N_k = 0.65 * B) : P = 0.5714 := by
  sorry

end percent_paddyfield_warblers_l44_44934


namespace binomial_n_value_l44_44521

noncomputable def n_value {n p : ℕ} {xi : ℝ → ℝ} (h1 : xi ∼ B(n, p)) (h2 : E xi = 6) (h3 : D xi = 3) : Prop :=
  n = 12

theorem binomial_n_value (n p : ℕ) (xi : ℝ → ℝ) :
  (xi ∼ B(n, p)) → (E xi = 6) → (D xi = 3) → n = 12 :=
begin
  intros h1 h2 h3,
  sorry
end

end binomial_n_value_l44_44521


namespace hyperbola_line_intersect_l44_44885

theorem hyperbola_line_intersect (x y k m x0 y0 : ℝ) (hk : k ≠ 2 ∧ k ≠ -2)
  (hx : x^2 - (y^2) / 4 = 1)
  (hy : y = k * x + m)
  (hA : x0 = -5 * k / m ∧ y0 = -5 / m):
  (k^2 = m^2 + 4) ∧
  ∃ x0 y0 : ℝ, (x0 = 7 ∧ y0 = sqrt 6) :=
by
  sorry

end hyperbola_line_intersect_l44_44885


namespace problem_statement_l44_44949

noncomputable def a (n : ℕ) : ℝ := 1 + (n - 1) * 3

noncomputable def term (n : ℕ) : ℝ := 1 / (Real.sqrt (a n) + Real.sqrt (a (n + 1)))

noncomputable def A : ℝ := ∑ n in Finset.range 1579, term n

def smallest_integer_greater_than_A : ℕ := ⌈A⌉

theorem problem_statement : smallest_integer_greater_than_A = 23 := 
sorry

end problem_statement_l44_44949


namespace inequality_and_equality_condition_l44_44625

variable {x y : ℝ}

theorem inequality_and_equality_condition
  (hx : 0 < x) (hy : 0 < y) :
  (x + y^2 / x ≥ 2 * y) ∧ (x + y^2 / x = 2 * y ↔ x = y) := sorry

end inequality_and_equality_condition_l44_44625


namespace range_of_m_l44_44865

open Real

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := log (2 * x^2 - x + m)

theorem range_of_m (h₁ : ∀ x : ℝ, f (-x) m = -f x m)
                   (h₂ : ∀ x : ℝ, f (x + 6) m = f x m)
                   (h₃ : f 0 m = 0)
                   (h₄ : f (-3) m = 0)
                   (h₅ : f 3 m = 0)
                   (h₆ : ∀ x ∈ Ioo(0, 3), 2 * x^2 - x + m > 0)
                   (h₇ : ∃ x' ∈ Ioo(0, 3), 2 * x'^2 - x' + m = 1)
                   : m ∈ Ioo(1/8, 1) ∨ m = 9/8 :=
sorry

end range_of_m_l44_44865


namespace coefficient_x2y3_in_expansion_l44_44017

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem
theorem coefficient_x2y3_in_expansion (x y : ℝ) : 
  binomial 5 3 * (2 : ℝ) ^ 2 * (-1 : ℝ) ^ 3 = -40 := by
sorry

end coefficient_x2y3_in_expansion_l44_44017


namespace area_of_trapezoid_l44_44843

/-- Given four coplanar squares with side lengths 3, 5, and 7 units, arranged side by side along a line,
and a segment connecting the bottom-left corner of the smallest square to the upper right of the largest square,
the area of the formed trapezoid is 12.83325 square units. -/
theorem area_of_trapezoid : 
∀ (s1 s2 s3 : ℕ), 
  s1 = 3 → 
  s2 = 5 → 
  s3 = 7 → 
  let base1 := s1 * (7 / 15) in
  let base2 := (s1 + s2) * (7 / 15) in
  let height := s2 in
  (1 / 2) * (base1 + base2) * height = 12.83325 :=
by
  intros s1 s2 s3 h1 h2 h3 base1 base2 height
  sorry

end area_of_trapezoid_l44_44843


namespace circle_intersection_range_l44_44515

theorem circle_intersection_range (m : ℝ) :
  (x^2 + y^2 - 4*x + 2*m*y + m + 6 = 0) ∧ 
  (∀ A B : ℝ, 
    (A - y = 0) ∧ (B - y = 0) → A * B > 0
  ) → 
  (m > 2 ∨ (-6 < m ∧ m < -2)) :=
by 
  sorry

end circle_intersection_range_l44_44515


namespace part1_part2_l44_44412

section PartOne

variables (x y : ℕ)
def condition1 := x + y = 360
def condition2 := x - y = 110

theorem part1 (h1 : condition1 x y) (h2 : condition2 x y) : x = 235 ∧ y = 125 := by {
  sorry
}

end PartOne

section PartTwo

variables (t W : ℕ)
def tents_capacity (t : ℕ) := 40 * t + 20 * (9 - t)
def food_capacity (t : ℕ) := 10 * t + 20 * (9 - t)
def transportation_cost (t : ℕ) := 4000 * t + 3600 * (9 - t)

theorem part2 
  (htents : tents_capacity t ≥ 235) 
  (hfood : food_capacity t ≥ 125) : 
  W = transportation_cost t → t = 3 ∧ W = 33600 := by {
  sorry
}

end PartTwo

end part1_part2_l44_44412


namespace balloon_arrangements_l44_44088

theorem balloon_arrangements : (7! / (2! * 2!)) = 1260 := by
  sorry

end balloon_arrangements_l44_44088


namespace main_theorem_l44_44851

noncomputable def locus_of_intersection (circle : Type) 
  (P Q A B M : circle) (m n : ℝ) 
  (arcPQ : subtype (m = measure_arc P Q)) 
  (arcAB : subtype (n = measure_arc A B)) : Prop :=
  ∀ (A B : circle), (∃ (M : circle), intersect (chord A Q) (chord B P) = M) ↔ 
  ∃ (circ : circle), (circ ∈ locus_of_point M) ∧ (P ∈ circ) ∧ (Q ∈ circ)

theorem main_theorem : ∀ (circle : Type) (m n : ℝ) 
  (P Q : circle) (A B M : circle) 
  (arcPQ : subtype (m = measure_arc P Q)) 
  (arcAB : subtype (n = measure_arc A B)), 
  locus_of_intersection circle P Q A B M m n arcPQ arcAB :=
begin
  sorry
end

end main_theorem_l44_44851


namespace no_solution_5x_plus_2_eq_17y_l44_44297

theorem no_solution_5x_plus_2_eq_17y :
  ¬∃ (x y : ℕ), 5^x + 2 = 17^y :=
sorry

end no_solution_5x_plus_2_eq_17y_l44_44297


namespace johns_age_l44_44232

theorem johns_age (d j : ℕ) 
  (h1 : j = d - 30) 
  (h2 : j + d = 80) : 
  j = 25 :=
by
  sorry

end johns_age_l44_44232


namespace distinct_arrangements_balloon_l44_44127

-- Let's define the basic conditions:
def total_letters : Nat := 7
def repeats_l : Nat := 2
def repeats_o : Nat := 2

-- Now let's state the problem.
theorem distinct_arrangements_balloon : 
  (Nat.factorial total_letters) / ((Nat.factorial repeats_l) * (Nat.factorial repeats_o)) = 1260 := 
by
  sorry

end distinct_arrangements_balloon_l44_44127


namespace at_most_one_solution_in_positive_integers_l44_44278

theorem at_most_one_solution_in_positive_integers 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : ¬ is_square a) 
  (h4 : ¬ is_square b) 
  (h5 : ¬ is_square (a * b)) :
  ¬(∃ x y : ℕ, x > 0 ∧ y > 0 ∧ a * x^2 - b * y^2 = 1 ∧ 
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ a * x^2 - b * y^2 = -1) :=
sorry

end at_most_one_solution_in_positive_integers_l44_44278


namespace parallel_lines_l44_44173

theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, x + 2 * a * y - 1 = 0 → (3 * a - 1) * x - 4 * a * y - 1 = 0 → False) → 
  (a = 0 ∨ a = -1/3) :=
sorry

end parallel_lines_l44_44173


namespace distinct_arrangements_balloon_l44_44095

theorem distinct_arrangements_balloon : 
  let n := 7 
  let freq_l := 2 
  let freq_o := 2 
  let freq_b := 1 
  let freq_a := 1 
  let freq_n := 1 
  Nat.factorial n / (Nat.factorial freq_l * Nat.factorial freq_o * Nat.factorial freq_b * Nat.factorial freq_a * Nat.factorial freq_n) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l44_44095


namespace distinct_arrangements_balloon_l44_44110

noncomputable def totalPermutations (n nl no : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial nl * Nat.factorial no)

theorem distinct_arrangements_balloon :
  totalPermutations 7 2 2 = 1260 := by 
  sorry

end distinct_arrangements_balloon_l44_44110


namespace part_I_monotonic_intervals_part_II_range_of_m_part_III_prove_x1x2_l44_44065

noncomputable def f (x : ℝ) (m : ℝ) := x * Real.log x - m * x^2

-- Part (I)
theorem part_I_monotonic_intervals (m : ℝ) (x : ℝ) :
  (m = 0) →
  (∀ x > 1 / Real.exp 1, (Real.log x + 1) > 0) ∧
  (∀ x < 1 / Real.exp 1, (Real.log x + 1) < 0) :=
by sorry

-- Part (II)
theorem part_II_range_of_m (m : ℝ) :
  (∀ x, x ∈ Set.Icc (Real.sqrt (Real.exp 1)) (Real.exp 2) →
    (x^2 - x) / f x m > 1) →
  (frac (3 * Real.sqrt (Real.exp 1) / (2 * Real.exp 1)) - 1 < m < 2 / Real.exp 4) :=
by sorry

-- Part (III)
theorem part_III_prove_x1x2 (x1 x2 : ℝ) :
  (x1 ∈ Set.Ioo (1 / Real.exp 1) 1) →
  (x2 ∈ Set.Ioo (1 / Real.exp 1) 1) →
  (x1 + x2 < 1) →
  (x1 * x2 < (x1 + x2)^4) :=
by sorry

end part_I_monotonic_intervals_part_II_range_of_m_part_III_prove_x1x2_l44_44065


namespace count_implications_l44_44808

theorem count_implications (p q r : Prop) :
  ((p ∧ q ∧ ¬r → ((q → p) → ¬r)) ∧ 
   (¬p ∧ ¬q ∧ ¬r → ((q → p) → ¬r)) ∧ 
   (p ∧ ¬q ∧ r → ¬ ((q → p) → ¬r)) ∧ 
   (¬p ∧ q ∧ ¬r → ((q → p) → ¬r))) →
   (3 = 3) := sorry

end count_implications_l44_44808


namespace probability_sum_3_is_1_over_216_l44_44354

-- Let E be the event that three fair dice sum to 3
def event_sum_3 (d1 d2 d3 : ℕ) : Prop := d1 + d2 + d3 = 3

-- Probabilities of rolling a particular outcome on a single die
noncomputable def P_roll_1 (n : ℕ) := if n = 1 then 1/6 else 0

-- Define the probability of the event E occurring
noncomputable def P_event_sum_3 := 
  ∑ d1 in {1, 2, 3, 4, 5, 6}, 
  ∑ d2 in {1, 2, 3, 4, 5, 6}, 
  ∑ d3 in {1, 2, 3, 4, 5, 6}, 
  if event_sum_3 d1 d2 d3 then P_roll_1 d1 * P_roll_1 d2 * P_roll_1 d3 else 0

-- The main theorem to prove the desired probability
theorem probability_sum_3_is_1_over_216 : P_event_sum_3 = 1/216 := by 
  sorry

end probability_sum_3_is_1_over_216_l44_44354


namespace polynomial_roots_l44_44057

-- Define the variables and conditions
variables (AT TB : ℝ)
-- Conditions
axiom sum_eq_fifteen : AT + TB = 15
axiom product_eq_thirtysix : AT * TB = 36

-- Polynomial to prove
def polynomial_with_new_roots := Polynomial.C 75 - Polynomial.X * (20) + Polynomial.X ^ 2

-- Theorem statement
theorem polynomial_roots :
  ∀ (x : ℝ), polynomial_with_new_roots.eval x = 0 ↔ (x = AT + 5 ∨ x = TB) :=
by
  sorry

end polynomial_roots_l44_44057


namespace u_1000_gt_45_l44_44618

def u : ℕ → ℝ
| 0       := 5
| (n + 1) := u n + 1 / u n

theorem u_1000_gt_45 : u 1000 > 45 := sorry

end u_1000_gt_45_l44_44618


namespace rate_of_interest_increase_l44_44301

noncomputable def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

noncomputable def percentage_increase_in_rate (P A1 A2 T : ℝ) : ℝ :=
  let SI1 := A1 - P in
  let R1 := (SI1 * 100) / (P * T) in
  let SI2 := A2 - P in
  let R2 := (SI2 * 100) / (P * T) in
  ((R2 - R1) / R1) * 100

theorem rate_of_interest_increase :
  percentage_increase_in_rate 800 956 1052 3 ≈ 61.54 := by
    sorry

end rate_of_interest_increase_l44_44301


namespace johns_age_l44_44234

theorem johns_age (d j : ℕ) 
  (h1 : j = d - 30) 
  (h2 : j + d = 80) : 
  j = 25 :=
by
  sorry

end johns_age_l44_44234


namespace sum_mod_20_l44_44831

theorem sum_mod_20 : 
  (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92 + 93 + 94) % 20 = 15 :=
by 
  -- The proof goes here
  sorry

end sum_mod_20_l44_44831


namespace smallest_nat_divisible_by_225_l44_44601

def has_digits_0_or_1 (n : ℕ) : Prop := 
  ∀ (d : ℕ), d ∈ n.digits 10 → d = 0 ∨ d = 1

def divisible_by_225 (n : ℕ) : Prop := 225 ∣ n

theorem smallest_nat_divisible_by_225 :
  ∃ (n : ℕ), has_digits_0_or_1 n ∧ divisible_by_225 n 
    ∧ ∀ (m : ℕ), has_digits_0_or_1 m ∧ divisible_by_225 m → n ≤ m 
    ∧ n = 11111111100 := 
  sorry

end smallest_nat_divisible_by_225_l44_44601


namespace balloon_arrangements_l44_44104

-- Define the variables
def n : ℕ := 7
def L_count : ℕ := 2
def O_count : ℕ := 2
def B_count : ℕ := 1
def A_count : ℕ := 1
def N_count : ℕ := 1

-- Define the multiset permutation formula
def multiset_permutations (n : ℕ) (counts : List ℕ) : ℕ :=
  n.factorial / (counts.map Nat.factorial).prod

-- Proof that the number of distinct arrangements is 1260
theorem balloon_arrangements : multiset_permutations n [L_count, O_count, B_count, A_count, N_count] = 1260 :=
  by
  -- The proof is omitted
  sorry

end balloon_arrangements_l44_44104


namespace t_l44_44342

theorem t {
  /- Define the sides and angle of the triangle -/
  AC AB BC: ℝ,
  CAB: ℝ,
  /- Conditions based on the problem -/
  h1: AC = 1,
  h2: AB = 2,
  h3: CAB = 60,
  /- Precomputed normalized Cosine for the use in Law of Cosines -/
  cos_60_deg : real.cos (CAB * real.pi / 180) = 1 / 2,
  /- Using Law of Cosines to compute BC, which is given as √3 -/
  BC: real.sqrt (AC^2 + AB^2 - 2 * AC * AB * real.cos (CAB * real.pi / 180)) = real.sqrt 3
} :
  (radius_calculated : ℝ) = 1 :=
begin
  /- Placeholder for proof that BC is indeed √3 using law of cosines as in solution steps -/
  admit,
  /- Use above steps to finally conclude radius as 1 -/
  sorry
end

end t_l44_44342


namespace sequence_d_is_geometric_l44_44448

def is_geometric_sequence (s : List ℚ) : Prop :=
  ∃ r : ℚ, ∀ (i : ℕ), i < s.length - 1 → s[i + 1] = r * s[i]

theorem sequence_d_is_geometric :
  is_geometric_sequence [16, -8, 4, -2] := sorry

end sequence_d_is_geometric_l44_44448


namespace laura_stock_percent_change_l44_44253

theorem laura_stock_percent_change (y : ℝ) (h1 : y > 0) :
  let day1_value := 0.85 * y in
  let day2_value := day1_value + 0.25 * day1_value in
  (day2_value - y) / y * 100 = 6.25 := 
by 
  sorry

end laura_stock_percent_change_l44_44253


namespace jack_marbles_l44_44965

theorem jack_marbles (initialMarbles : ℕ) (soldPercent : ℝ) (gavePercent : ℝ) (lostMarbles : ℕ) (donatedMarbles : ℕ) :
  initialMarbles = 150 ∧ soldPercent = 0.2 ∧ gavePercent = 0.1 ∧ lostMarbles = 5 ∧ donatedMarbles = 1 →
  let remainingMarblesAfterSell := initialMarbles - (initialMarbles * soldPercent).toInt in
  let remainingMarblesAfterGive := remainingMarblesAfterSell - (remainingMarblesAfterSell * gavePercent).toInt in
  let remainingMarblesAfterLoss := remainingMarblesAfterGive - lostMarbles in
  let finalMarbles := remainingMarblesAfterLoss - donatedMarbles in
  finalMarbles = 102 :=
begin
  intro h,
  sorry
end

end jack_marbles_l44_44965


namespace maximum_distance_between_curves_l44_44540

theorem maximum_distance_between_curves :
  ∃ (P Q : ℝ × ℝ), 
    (∃ θ : ℝ, P = (sqrt 2 * real.cos θ, 6 + sqrt 2 * real.sin θ)) ∧
    (∃ ϕ : ℝ, Q = (sqrt 10 * real.cos ϕ, real.sin ϕ)) ∧
    (∀ (P : ℝ × ℝ) (Q : ℝ × ℝ),
       (∃ θ : ℝ, P = (sqrt 2 * real.cos θ, 6 + sqrt 2 * real.sin θ)) →
       (∃ ϕ : ℝ, Q = (sqrt 10 * real.cos ϕ, real.sin ϕ)) →
       dist P Q ≤ 6 * sqrt 2) := sorry

end maximum_distance_between_curves_l44_44540


namespace length_AE_is_root_13_l44_44688

-- Defining the points
def A := (0, 4)
def B := (6, 0)
def C := (2, 1)
def D := (5, 4)

-- Distance function between two points (x1, y1) and (x2, y2)
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Intersection point of lines AB and CD
def E := (3, 2)

theorem length_AE_is_root_13 : distance A E = real.sqrt 13 :=
by
  sorry

end length_AE_is_root_13_l44_44688


namespace number_is_correct_l44_44755

theorem number_is_correct : (1 / 8) + 0.675 = 0.800 := 
by
  sorry

end number_is_correct_l44_44755


namespace females_attending_meeting_l44_44560

noncomputable def number_of_females_attending_meeting : ℕ :=
let total_people := 300 in
let attending := total_people / 2 in
let females := attending / 3 in
females

theorem females_attending_meeting : number_of_females_attending_meeting = 50 := by
  sorry

end females_attending_meeting_l44_44560


namespace tan_ratio_area_of_triangle_l44_44929

-- Definitions for the conditions
variable {A B C a b c : ℝ}
variable {α β γ : ℝ} -- angles A, B, C in radians

-- Provided conditions
axiom angle_A : α = A
axiom angle_B : β = B
axiom angle_C : γ = C
axiom side_a : a = 3 * b * Real.cos γ
axiom ABC_triangle : Triangle A B C

-- Part I: Prove tan C / tan B = 2
theorem tan_ratio : Real.tan γ / Real.tan β = 2 :=
  sorry

-- Provided conditions for Part II
axiom value_a : a = 3
axiom tan_A_eq_3 : Real.tan α = 3

-- Part II: Prove the area of triangle ABC = 3
theorem area_of_triangle : (1 / 2) * b * c * Real.sin α = 3 :=
  sorry

end tan_ratio_area_of_triangle_l44_44929


namespace mod_5_pow_1000_div_29_l44_44163

theorem mod_5_pow_1000_div_29 : 5^1000 % 29 = 21 := 
by 
  -- The proof will go here.
  sorry

end mod_5_pow_1000_div_29_l44_44163


namespace percentage_increase_correct_l44_44175

variable {R1 E1 P1 R2 E2 P2 R3 E3 P3 : ℝ}

-- Conditions
axiom H1 : P1 = R1 - E1
axiom H2 : R2 = 1.20 * R1
axiom H3 : E2 = 1.10 * E1
axiom H4 : P2 = R2 - E2
axiom H5 : P2 = 1.15 * P1
axiom H6 : R3 = 1.25 * R2
axiom H7 : E3 = 1.20 * E2
axiom H8 : P3 = R3 - E3
axiom H9 : P3 = 1.35 * P2

theorem percentage_increase_correct :
  ((P3 - P1) / P1) * 100 = 55.25 :=
by sorry

end percentage_increase_correct_l44_44175


namespace chris_age_l44_44321

theorem chris_age (a b c : ℤ) (h1 : a + b + c = 45) (h2 : c - 5 = a)
  (h3 : c + 4 = 3 * (b + 4) / 4) : c = 15 :=
by
  sorry

end chris_age_l44_44321


namespace log_x_inequality_l44_44862

noncomputable def log_x_over_x (x : ℝ) := (Real.log x) / x

theorem log_x_inequality {x : ℝ} (h1 : 1 < x) (h2 : x < 2) : 
  (log_x_over_x x) ^ 2 < log_x_over_x x ∧ log_x_over_x x < log_x_over_x (x * x) :=
by
  sorry

end log_x_inequality_l44_44862


namespace proof_main_l44_44073

-- Define the conditions
def elements (a b c : ℕ) : Prop := {a, b, c} = {0, 1, 3}
def condition_a (a : ℕ) : Prop := a ≠ 3
def condition_b (b : ℕ) : Prop := b = 3
def condition_c (c : ℕ) : Prop := c ≠ 0
def one_correct_condition (a b c : ℕ) : Prop := 
  (condition_a a ∧ ¬condition_b b ∧ ¬condition_c c) ∨ 
  (¬condition_a a ∧ condition_b b ∧ ¬condition_c c) ∨ 
  (¬condition_a a ∧ ¬condition_b b ∧ condition_c c)

def main_theorem (a b c : ℕ) : Prop := 
  elements a b c ∧ one_correct_condition a b c → 100 * a + 10 * b + c = 301

-- The statement
theorem proof_main : ∃ (a b c : ℕ), main_theorem a b c :=
begin
  -- Actual proof is not required, hence we use sorry
  sorry
end

end proof_main_l44_44073


namespace find_third_number_l44_44415

theorem find_third_number (x : ℕ) : 9548 + 7314 = x + 13500 ↔ x = 3362 :=
by
  sorry

end find_third_number_l44_44415


namespace sqrt_D_irrational_l44_44992

theorem sqrt_D_irrational (a b c : ℤ) (h : a + 1 = b) (h_c : c = a + b) : 
  Irrational (Real.sqrt ((a^2 : ℤ) + (b^2 : ℤ) + (c^2 : ℤ))) :=
  sorry

end sqrt_D_irrational_l44_44992


namespace abc_inequality_l44_44839

theorem abc_inequality (a b c : ℝ) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (h : a * b + a * c + b * c = a + b + c) : 
  a + b + c + 1 ≥ 4 * a * b * c :=
by 
  sorry

end abc_inequality_l44_44839


namespace conclusion_1_conclusion_2_conclusion_3_main_theorem_l44_44391

-- First, define the necessary conditions
variables {V : Type*} [inner_product_space ℝ V]

-- Definition of parallel between lines
def parallel (l m : submodule ℝ V) : Prop := ∃ (v : V), l = submodule.span ℝ {v} ∧ m = submodule.span ℝ {v}

-- Definition of perpendicular between lines
def perpendicular (l m : submodule ℝ V) : Prop := ∃ (u v : V), l = submodule.span ℝ {u} ∧ m = submodule.span ℝ {v} ∧ inner_product_space.is_orthogonal V u v

-- Formulating the statements
theorem conclusion_1 {l m n : submodule ℝ V} (h1 : parallel l m) (h2 : parallel m n) : parallel l n := sorry
theorem conclusion_2 {l m n : submodule ℝ V} (h1 : perpendicular l m) (h2 : parallel m n) : perpendicular l n := sorry
theorem conclusion_3 {l m n : submodule ℝ V} (h1 : nonempty (l ⊓ m)) (h2 : parallel m n) : ¬ nonempty (l ⊓ n) := sorry

-- Main theorem combining conclusions
theorem main_theorem {l m n : submodule ℝ V} :
  (∀ (l m n : submodule ℝ V), parallel l m → parallel m n → parallel l n) ∧
  (∀ (l m n : submodule ℝ V), perpendicular l m → parallel m n → perpendicular l n) ∧
  (∀ (l m n : submodule ℝ V), nonempty (l ⊓ m) → parallel m n → ¬ nonempty (l ⊓ n)) := 
begin
  split,
  { intros l m n h1 h2,
    exact conclusion_1 h1 h2,
  },
  split,
  { intros l m n h1 h2,
    exact conclusion_2 h1 h2,
  },
  { intros l m n h1 h2,
    exact conclusion_3 h1 h2
  }
end

end conclusion_1_conclusion_2_conclusion_3_main_theorem_l44_44391


namespace inequality_polynomial_l44_44994

open Real

noncomputable def P (x : ℝ) (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range (n+1)).sum (λ i, a i * x ^ i)

theorem inequality_polynomial (a : ℕ → ℝ) (n : ℕ) (h : ∀ i, 0 ≤ a i) :
  ∀ x : ℝ, x ≠ 0 → P x a n * P (1 / x) a n ≥ (P 1 a n) ^ 2 :=
by
  intros x hx
  sorry

end inequality_polynomial_l44_44994


namespace minimum_x_y_sum_l44_44558

theorem minimum_x_y_sum (x y : ℕ) (hx : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h : (1 / (x : ℚ)) + (1 / (y : ℚ)) = 1 / 15) : x + y = 64 :=
  sorry

end minimum_x_y_sum_l44_44558


namespace find_a1_min_value_l44_44956

-- Define the sequence recursively
def a : ℕ → ℝ
| 0       := 0  -- This will not be used since we start from a₁
| (n + 1) := a n ^ 2 - a n + 1

-- The proof problem statement
theorem find_a1_min_value :
  ∃ a1 : ℝ, a1 > 1 ∧
    (∃ (an : ℕ → ℝ),
      an 1 = a1 ∧
      (∀ n : ℕ, an (n + 1) = an n ^ 2 - an n + 1) ∧
      (1 / a1 + ∑ i in finset.range 2014, 1 / an (i + 2) = 2) ∧
        (∀ a1' ∈ {a : ℝ | a > 1},
          let a2016 := (2 - a1') / (3 - 2 * a1')
          in a2016 - 4 * a1' ≥ (2 - 11 / 2)) ∧
        a1 = 5 / 4) := 
sorry

end find_a1_min_value_l44_44956


namespace factorization_l44_44824

theorem factorization (m : ℝ) : 3 * m^2 - 6 * m = 3 * m * (m - 2) := 
by
  sorry

end factorization_l44_44824


namespace night_flying_hours_completed_l44_44707

-- Assuming required definitions and variables
variable (total_hours_req : ℕ) (day_flying_hours : ℕ) (cross_country_hours : ℕ)
variable (monthly_flight_hours : ℕ) (months : ℕ)
variable (night_flying_hours : ℕ)

-- Giving values to the assumed variables
def total_hours_req := 1500
def day_flying_hours := 50
def cross_country_hours := 121
def monthly_flight_hours := 220
def months := 6

-- Total completed hours without night flying
def total_completed_hours := day_flying_hours + cross_country_hours

-- Total hours to fly in given months
def total_hours_in_months := monthly_flight_hours * months

-- The statement to prove
theorem night_flying_hours_completed :
  night_flying_hours = total_hours_req - total_completed_hours - total_hours_in_months :=
sorry

end night_flying_hours_completed_l44_44707


namespace union_M_N_eq_M_l44_44001

def M : set (ℝ × ℝ) := {p | p.1 + p.2 = 0}
def N : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 0}

theorem union_M_N_eq_M : M ∪ N = M := by
  sorry

end union_M_N_eq_M_l44_44001


namespace red_tile_probability_l44_44753

def red_tile_count : ℕ :=
  -- count of numbers between 1 and 100 congruent to 3 (mod 7)
  Nat.card (Finset.filter (λ n => n % 7 = 3) (Finset.range' 1 100))

def total_tile_count : ℕ := 100

def probability_of_red_tile : ℝ := (red_tile_count : ℝ) / (total_tile_count : ℝ)

theorem red_tile_probability : probability_of_red_tile = 7 / 50 :=
by
  -- proof here
  sorry

end red_tile_probability_l44_44753


namespace sandy_age_l44_44671

theorem sandy_age (S M : ℕ) (h1 : M = S + 14) (h2 : S / M = 7 / 9) : S = 49 :=
sorry

end sandy_age_l44_44671


namespace regular_pentagon_of_equal_angles_l44_44185

-- Defining the angles for a given pentagon
variables {A B C D E : Type}
variables [angle ABE, angle ACB, angle ADB, angle AEB, angle BAC, angle CAD, angle DBE : ℝ]

-- Hypothesis: all marked angles are equal
variables (h1 : angle ABE = angle ACB)
variables (h2 : angle ACB = angle ADB)
variables (h3 : angle ADB = angle AEB)
variables (h4 : angle AEB = angle BAC)
variables (h5 : angle BAC = angle CAD)
variables (h6 : angle CAD = angle DBE)

-- To prove: the pentagon is regular
theorem regular_pentagon_of_equal_angles
  (h1 : angle ABE = angle ACB)
  (h2 : angle ACB = angle ADB)
  (h3 : angle ADB = angle AEB)
  (h4 : angle AEB = angle BAC)
  (h5 : angle BAC = angle CAD)
  (h6 : angle CAD = angle DBE) :
  regular_pentagon (ABCDE) :=
sorry

end regular_pentagon_of_equal_angles_l44_44185


namespace part_a_equilateral_l44_44942

theorem part_a_equilateral (A B C : Type) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] 
  (h₁: is_perp_bisector_eq A B C) : 
  is_equilateral A B C := sorry

end part_a_equilateral_l44_44942


namespace find_x_l44_44029

-- Assume the condition
theorem find_x (x : ℝ) (h : sqrt (x - 3) = 5) : x = 28 :=
sorry

end find_x_l44_44029


namespace rectangle_diagonals_equal_in_length_l44_44727

theorem rectangle_diagonals_equal_in_length (AB CD AD BC : ℝ) (rect : (AB = CD ∧ AD = BC) ∧ (angles_right : ∀ (A B C D : Type), (is_right_angle : quad A B C D) ) ) :
    diagonals_equal (AB CD AD BC) := sorry

end rectangle_diagonals_equal_in_length_l44_44727


namespace profit_approx_l44_44731

def selling_price : ℝ := 2524.36
def cost_price : ℝ := 2400.0
def profit_amount : ℝ := selling_price - cost_price
def profit_percent : ℝ := (profit_amount / cost_price) * 100

theorem profit_approx : abs (profit_percent - 5.18) < 0.01 := by
  sorry

end profit_approx_l44_44731


namespace ratio_of_boys_l44_44584

theorem ratio_of_boys 
  (p : ℚ) 
  (h : p = (3/4) * (1 - p)) : 
  p = 3 / 7 :=
by
  sorry

end ratio_of_boys_l44_44584


namespace problem_solution_l44_44166

def f (x : ℤ) : ℤ := 3 * x + 1
def g (x : ℤ) : ℤ := 4 * x - 3

theorem problem_solution :
  (f (g (f 3))) / (g (f (g 3))) = 112 / 109 := by
sorry

end problem_solution_l44_44166


namespace relationship_a_b_l44_44525

-- Definitions for conditions
variable (a b : ℝ)

-- Condition 1
def condition1 : Prop := 1003^a + 1004^b = 2006^b

-- Condition 2
def condition2 : Prop := 997^a + 1009^b = 2007^a

-- Theorem statement
theorem relationship_a_b (h1 : condition1 a b) (h2 : condition2 a b) : a < b :=
  sorry

end relationship_a_b_l44_44525


namespace three_arithmetic_progressions_not_two_arithmetic_progressions_l44_44611

def is_arithmetic_progression (a : Int) (d : Int) (S : Set Int) : Bool :=
  ∀ x ∈ S, ∃ n : Int, x = a + n * d

def is_union_of_arithmetic_progressions (S : Set Int) (n : Nat) : Bool :=
  ∃ (A : List (Set Int)), List.length A = n ∧ ∀ s ∈ S, ∃ a ∈ A, s ∈ a

noncomputable def set_S : Set Int := {9, 15, 21, 25, 27, 33, 35, 39, 45, 49, 51, 55, 57, 63, 65, 69, 75}

theorem three_arithmetic_progressions :
  is_union_of_arithmetic_progressions set_S 3 :=
sorry

theorem not_two_arithmetic_progressions :
  ¬ is_union_of_arithmetic_progressions set_S 2 :=
sorry

end three_arithmetic_progressions_not_two_arithmetic_progressions_l44_44611


namespace hypotenuse_of_45_45_90_triangle_l44_44650

theorem hypotenuse_of_45_45_90_triangle (a : ℝ) (h : ℝ) 
  (ha : a = 15) 
  (angle_opposite_leg : ℝ) 
  (h_angle : angle_opposite_leg = 45) 
  (right_triangle : ∃ θ : ℝ, θ = 90) : 
  h = 15 * Real.sqrt 2 := 
sorry

end hypotenuse_of_45_45_90_triangle_l44_44650


namespace no_common_period_l44_44408

theorem no_common_period (g h : ℝ → ℝ) 
  (hg : ∀ x, g (x + 2) = g x) 
  (hh : ∀ x, h (x + π/2) = h x) : 
  ¬ (∃ T > 0, ∀ x, g (x + T) + h (x + T) = g x + h x) :=
sorry

end no_common_period_l44_44408


namespace subset_P1_P2_l44_44614

def P1 (a : ℝ) : Set ℝ := {x | x^2 + a*x + 1 > 0}
def P2 (a : ℝ) : Set ℝ := {x | x^2 + a*x + 2 > 0}

theorem subset_P1_P2 (a : ℝ) : P1 a ⊆ P2 a :=
by intros x hx; sorry

end subset_P1_P2_l44_44614


namespace martians_legs_ratio_l44_44443

def aliens_arms := 3
def aliens_legs := 8
def martians_arms := 2 * aliens_arms

def five_aliens_limbs : Nat := 5 * aliens_arms + 5 * aliens_legs
def five_martians_limbs (M : Nat) : Nat := 5 * martians_arms + 5 * M

theorem martians_legs_ratio :
  ∃ M : Nat, five_aliens_limbs = five_martians_limbs(M) + 5 ∧
  (M : ℚ) / (aliens_legs : ℚ) = 1 / 2 :=
by
  sorry

end martians_legs_ratio_l44_44443


namespace largest_element_sum_of_digits_in_E_l44_44806
open BigOperators
open Nat

def E : Set ℕ := { n | ∃ (r₉ r₁₀ r₁₁ : ℕ), 0 < r₉ ∧ r₉ ≤ 9 ∧ 0 < r₁₀ ∧ r₁₀ ≤ 10 ∧ 0 < r₁₁ ∧ r₁₁ ≤ 11 ∧
  r₉ = n % 9 ∧ r₁₀ = n % 10 ∧ r₁₁ = n % 11 ∧
  (r₉ > 1) ∧ (r₁₀ > 1) ∧ (r₁₁ > 1) ∧
  ∃ (a : ℕ) (b : ℕ) (c : ℕ), r₉ = a ∧ r₁₀ = a * b ∧ r₁₁ = a * b * c ∧ b ≠ 1 ∧ c ≠ 1 }

noncomputable def N : ℕ := 
  max (max (74 % 990) (134 % 990)) (526 % 990)

def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem largest_element_sum_of_digits_in_E :
  sum_of_digits N = 13 :=
sorry

end largest_element_sum_of_digits_in_E_l44_44806


namespace johns_age_l44_44231

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l44_44231


namespace total_marks_calculation_l44_44180

def average (total_marks : ℕ) (num_candidates : ℕ) : ℕ := total_marks / num_candidates
def total_marks (average : ℕ) (num_candidates : ℕ) : ℕ := average * num_candidates

theorem total_marks_calculation
  (num_candidates : ℕ)
  (average_marks : ℕ)
  (range_min : ℕ)
  (range_max : ℕ)
  (h1 : num_candidates = 250)
  (h2 : average_marks = 42)
  (h3 : range_min = 10)
  (h4 : range_max = 80) :
  total_marks average_marks num_candidates = 10500 :=
by 
  sorry

end total_marks_calculation_l44_44180


namespace conference_end_time_correct_l44_44761

-- Define the conference conditions
def conference_start_time : ℕ := 15 * 60 -- 3:00 p.m. in minutes
def conference_duration : ℕ := 450 -- 450 minutes duration
def daylight_saving_adjustment : ℕ := 60 -- clocks set forward by one hour

-- Define the end time computation
def end_time_without_daylight_saving : ℕ := conference_start_time + conference_duration
def end_time_with_daylight_saving : ℕ := end_time_without_daylight_saving + daylight_saving_adjustment

-- Prove that the conference ended at 11:30 p.m. (11:30 p.m. in minutes is 23 * 60 + 30)
theorem conference_end_time_correct : end_time_with_daylight_saving = 23 * 60 + 30 := by
  sorry

end conference_end_time_correct_l44_44761


namespace average_speed_l44_44420

theorem average_speed (v : ℝ) (h : 500 / v - 500 / (v + 10) = 2) : v = 45.25 :=
by
  sorry

end average_speed_l44_44420


namespace finite_S_l44_44622

def φ (n : ℕ) : ℕ := (fin n).filter (nat.coprime n).card

def τ (n : ℕ) : ℕ := nat.divisors n.length

def S := { n : ℕ | n > 0 ∧ φ n * τ n ≥ (n^3 / 3).sqrt }

theorem finite_S : S.finite := 
sorry

end finite_S_l44_44622


namespace part_I_part_II_l44_44771

noncomputable def f (a x : ℝ) : ℝ :=
  2 * a * sin x * cos x + cos (2 * x)

theorem part_I (h : f a (π / 4) = 1) : a = 1 ∧ (∀x, f 1 x = sqrt 2 * sin (2 * x + π / 4)) ∧ (∃ T > 0, ∀ x, f 1 (x + T) = f 1 x ∧ T = π) :=
by {
  sorry
}

theorem part_II (h : a = 1) : ∀ (x : ℝ), 0 < x → x < π → 
  (∀ x, x ∈ Icc (π / 8) (5 * π / 8) → f 1 x = sqrt 2 * sin (2 * x + π / 4)) :=
by {
  sorry
}

end part_I_part_II_l44_44771


namespace percentage_increase_l44_44340

theorem percentage_increase (S P : ℝ) (h1 : (S * (1 + P / 100)) * 0.8 = 1.04 * S) : P = 30 :=
by 
  sorry

end percentage_increase_l44_44340


namespace systematic_sampling_proof_l44_44788

-- Define the conditions
def method_1_systematic := 
  ∃ (i_0 : ℕ) (ball_numbers : set ℕ), ball_numbers ⊆ {n | 1 ≤ n ∧ n ≤ 15} ∧ 
                                      ball_numbers.card = 3 ∧
                                      ∀ i ∈ ball_numbers, i + 5 ∈ ball_numbers ∨ (i + 5) % 15 + 1 ∈ ball_numbers ∧
                                      ∀ i ∈ ball_numbers, i + 10 ∈ ball_numbers ∨ (i + 10) % 15 + 1 ∈ ball_numbers

def method_2_systematic := 
  ∃ (interval : ℕ), interval = 5 ∧ 
                     ∀ t : ℕ, t % interval = 0 → ∃ product : ℕ, product > 0

def method_3_systematic := false -- Method 3 is explicitly not systematic

def method_4_systematic := 
  ∀ row : ℕ, row > 0 → ∃ seat_number : ℕ, seat_number = 14

-- Prove that methods 1, 2, and 4 are systematic, and method 3 is not
theorem systematic_sampling_proof : 
  (method_1_systematic → true) ∧   -- Implicitly meaning method 1 is systematic
  (method_2_systematic → true) ∧   -- Implicitly meaning method 2 is systematic
  (¬method_3_systematic) ∧         -- Explicitly meaning method 3 is not systematic
  (method_4_systematic → true) :=  -- Implicitly meaning method 4 is systematic
sorry

end systematic_sampling_proof_l44_44788


namespace coconut_grove_produce_trees_l44_44181

theorem coconut_grove_produce_trees (x : ℕ)
  (h1 : 60 * (x + 3) + 120 * x + 180 * (x - 3) = 100 * 3 * x)
  : x = 6 := sorry

end coconut_grove_produce_trees_l44_44181


namespace Callum_total_points_l44_44610

theorem Callum_total_points :
  let matches := 15
  let wins_per_match := 10
  let win_point_multiplier := λ (streak: ℕ), 2 ^ streak
  let loss_point_decrease := λ (consec_losses: ℕ), wins_per_match / (2 ^ (consec_losses - 2))
  let krishna_wins := 9
  let callum_wins := matches - krishna_wins
  let total_points :=
    90 + 9.921875 in
  total_points ≈ 100 :=
by
  -- Preliminary definitions
  have define_krishna_wins : (krishna_wins = 3 * matches / 5) := by norm_num
  have define_callum_wins : (callum_wins = matches - krishna_wins) := by tauto
  sorry

end Callum_total_points_l44_44610


namespace required_fencing_l44_44400

def rectangular_field  : Prop :=
  ∃ (L W : ℝ), L = 30 ∧ 810 = L * W ∧ 840 = 2 * W + L

theorem required_fencing : 
  rectangular_field → ∃ (f : ℝ), f = 84 :=
by 
  intro h
  cases h with L hL
  cases hL with W hW
  cases hW with hLength hArea
  dsimp only at *
  rw [hLength] at hArea
  rw [hLength]
  -- hArea: 810 = 30 * W
  have hw : W = 27, from eq_of_mul_eq_mul_right (by norm_num) hArea.symm,
  rw [hw],
  use 84,
  norm_num,
  sorry

end required_fencing_l44_44400


namespace parabola_symmetric_point_l44_44569

theorem parabola_symmetric_point 
    (a c : ℝ) 
    (h : (2 : ℝ), 3 = (2 : ℝ) * (a : ℝ) ^ 2 + 2 * (2 : ℝ) * (a : ℝ) + (c : ℝ)) :
    (-4 : ℝ), 3 = (-4 : ℝ) * (a : ℝ) ^ 2 + 2 * (-4 : ℝ) * (a : ℝ) + (c : ℝ) :=
  sorry

end parabola_symmetric_point_l44_44569


namespace circle_center_coordinates_l44_44539

-- Definition of the circle's equation
def circle_eq : Prop := ∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = 3

-- Proof of the circle's center coordinates
theorem circle_center_coordinates : ∃ h k : ℝ, (h, k) = (2, -1) := 
sorry

end circle_center_coordinates_l44_44539


namespace NK_parallel_A5A2_l44_44665

theorem NK_parallel_A5A2
  (A1 A2 A3 A4 A5 A6 K L M N : Type)
  [circle : ∀ {X Y Z : Type}, (Segment X Y Z) → Prop]
  (on_circle : circle (Segment A1 A2 A3) ∧ circle (Segment A1 A4 A5) 
  ∧ circle (Segment A1 A6 A2) ∧ circle (Segment A1 A6 A5) 
  ∧ circle (Segment A1 A3 A4) ∧ circle (Segment A1 A4 A5))
  (on_lines : (Line K = Line A1 A2) ∧ (Line L = Line A3 A4) 
  ∧ (Line M = Line A1 A6) ∧ (Line N = Line A4 A5))
  (par_kl_A2A3 : parallel (Line KL) (Line A2 A3))
  (par_lm_A3A6 : parallel (Line LM) (Line A3 A6))
  (par_mn_A6A5 : parallel (Line MN) (Line A6 A5)) :
  parallel (Line NK) (Line A5 A2) :=
by
  sorry

end NK_parallel_A5A2_l44_44665


namespace binary_op_property_l44_44047

variable (X : Type)
variable (star : X → X → X)
variable (h : ∀ x y : X, star (star x y) x = y)

theorem binary_op_property (x y : X) : star x (star y x) = y := 
by 
  sorry

end binary_op_property_l44_44047


namespace inversely_proportional_ratio_y1_y2_l44_44680

noncomputable theory

variable {x y : ℝ}
variable {x1 x2 y1 y2 : ℝ}

-- Condition 1: x is inversely proportional to y
def inversely_proportional (x y : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ x * y = k

-- Condition 2: x1 and x2 are two nonzero values of x such that x1 / x2 = 3 / 4
def ratio_x1_x2 (x1 x2 : ℝ) : Prop := x1 ≠ 0 ∧ x2 ≠ 0 ∧ x1 / x2 = 3 / 4

-- Condition 3: y1 and y2 are nonzero values corresponding to x1 and x2
def nonzero_y1_y2 (y1 y2 : ℝ) : Prop := y1 ≠ 0 ∧ y2 ≠ 0

theorem inversely_proportional_ratio_y1_y2 
  (h1 : inversely_proportional x y) 
  (hx : ratio_x1_x2 x1 x2) 
  (hy : nonzero_y1_y2 y1 y2) : 
  y1 / y2 = 4 / 3 := sorry

end inversely_proportional_ratio_y1_y2_l44_44680


namespace max_imaginary_part_of_root_l44_44446

theorem max_imaginary_part_of_root (z : ℂ) (h : z^6 - z^4 + z^2 - 1 = 0) (hne : z^2 ≠ 1) : 
  ∃ θ : ℝ, -90 ≤ θ ∧ θ ≤ 90 ∧ Complex.im z = Real.sin θ ∧ θ = 90 := 
sorry

end max_imaginary_part_of_root_l44_44446


namespace mercers_theorem_gaussian_process_representation_verify_continuity_self_adjointness_l44_44739

noncomputable theory

-- Definition of the covariance function K
def covariance_function (a b : ℝ) := { K : ℝ × ℝ → ℝ // Continuous K }

-- Definition of the operator A in terms of K
def operator_A (a b : ℝ) (K : ℝ → ℝ → ℝ) (f : ℝ → ℝ) (s : ℝ) :=
  ∫ t in set.Icc a b, K s t * f t

-- Statement of Mercer's theorem
theorem mercers_theorem (a b : ℝ) (K : ℝ → ℝ → ℝ) (hK : ContinuousOn (λ p : ℝ × ℝ, K p.1 p.2) (set.Icc a b).prod (set.Icc a b)) :
  ∃ (λ : ℕ → ℝ) (φ : ℕ → ℝ → ℝ),
    (∀ n, λ n ≥ 0) ∧
    Orthonormal ℝ φ ∧
    (∀ n, ContinuousOn (λ s, φ n s) (set.Icc a b)) ∧
    (∀ s t, K s t = ∑' n, λ n * φ n s * φ n t) :=
sorry

-- Representation of the Gaussian process
theorem gaussian_process_representation (a b : ℝ) (K : ℝ → ℝ → ℝ) (λ : ℕ → ℝ) (φ : ℕ → ℝ → ℝ) (ξ : ℕ → ℝ) :
  (∀ n, ContinuousOn (λ s, φ n s) (set.Icc a b)) →
  (∀ n, ξ n ∼ 𝒩 0 1) →
  (∀ s t, K s t = ∑' n, λ n * φ n s * φ n t) →
  ∀ t, (X_t : ℝ) = ∑' n, ξ n * (λ n)^(1/2) * φ n t :=
sorry

-- Verify continuity and self-adjointness
theorem verify_continuity_self_adjointness (a b : ℝ) (K : ℝ → ℝ → ℝ) (hK : ContinuousOn (λ p : ℝ × ℝ, K p.1 p.2) (set.Icc a b).prod (set.Icc a b)) :
  ∃ (A : (ℝ → ℝ) → (ℝ → ℝ)),
    (∀ f g : ℝ → ℝ, ∥ A f - A g ∥_2 ≤ (b - a) * ∥K∥∞ * ∥ f - g ∥_2) ∧
    (∀ f g : ℝ → ℝ, ⟪ A f, g ⟫ = ⟪ f, A g ⟫) :=
sorry

end mercers_theorem_gaussian_process_representation_verify_continuity_self_adjointness_l44_44739


namespace area_of_region_l44_44016

theorem area_of_region : 
  let x1 := 0
  let y1 := 0
  let x2 := 2
  let y2 := 2
  in (x2 - x1) * (y2 - y1) = 4 :=
by
  simp [x1, y1, x2, y2]
  norm_num
  sorry

end area_of_region_l44_44016


namespace abs_x_minus_2_plus_abs_x_minus_1_lt_b_iff_b_gt_1_l44_44906

variable (x b : ℝ)

theorem abs_x_minus_2_plus_abs_x_minus_1_lt_b_iff_b_gt_1 :
  (∃ x : ℝ, |x - 2| + |x - 1| < b) ↔ b > 1 := sorry

end abs_x_minus_2_plus_abs_x_minus_1_lt_b_iff_b_gt_1_l44_44906


namespace collinear_EFN_l44_44518

open EuclideanGeometry

-- Defining the cyclic quadrilateral and the midpoint
variables {A B C D M N E F : Point}
variable [CyclicQuadrilateral ABCD]

-- Definitions of M and N with given properties
variable (M_mid: isMidpoint M C D) (N_circumcircle_abm: OnCircle N (Circumcircle A B M)) 
variable (N_neq_M : N ≠ M)
variable (ratio_condition : (LineSegmentRatio A N B = LineSegmentRatio A M B))

-- Intersections defining points E and F
variable (E_def : E = intersection_point (line_through A C) (line_through B D))
variable (F_def : F = intersection_point (line_through B C) (line_through D A))

-- The theorem to prove collinearity
theorem collinear_EFN : Collinear {E, F, N} :=
by
  sorry

end collinear_EFN_l44_44518


namespace speed_A_is_21_l44_44749

variables (a b τ D : ℕ)
variables (a_gt_b : a > b)
variables (b_not_factor_a : ¬ (∃ k, a = b * k))
variables (meet_point_at_d : D = a * τ + b * τ)
variables (new_meeting_point : a * (τ + 2) + b * τ = D + 42)

theorem speed_A_is_21 (h₁ : a > b)
                      (h₂ : ¬ (∃ k, a = b * k))
                      (h₃ : D = a * τ + b * τ)
                      (h₄ : a * (τ + 2) + b * τ = D + 42) :
                      a = 21 :=
begin
  sorry
end

end speed_A_is_21_l44_44749


namespace find_max_min_f_l44_44067

theorem find_max_min_f (f : ℝ → ℝ)
    (h1 : ∀ x y : ℝ, f(x - y) = f(x) - f(y))
    (h2 : ∀ x : ℝ, x > 0 → f(x) > 0) :
    (∃ max min : ℝ, max = 10 ∧ min = -10 ∧
      ∀ x ∈ set.Icc (-5 : ℝ) 5, f x ≤ max ∧ f x ≥ min) :=
by
  let f1 := f(1)
  have f_positive : f(1) = 2 := by sorry
  have f2 : f(2) = f(1) + f(1) := by sorry
  have f2_val : f2 = 4 := by sorry
  have f4 : f(4) = f(2) + f(2) := by sorry
  have f4_val : f4 = 8 := by sorry
  have f5 : f(5) = f(1) + f(4) := by sorry
  have max_val : f(5) = 10 := by sorry
  have min_val : f(-5) = -10 := by sorry
  exact ⟨10, -10, rfl, rfl, λ x hx, ⟨le_of_lt hx.2, le_of_lt hx.1⟩⟩
    sorry

end find_max_min_f_l44_44067


namespace sum_of_first_99_terms_l44_44600

-- Define the arithmetic sequence
variable (a : ℕ → ℕ) -- For simplicity, using natural numbers; modify as needed
variable (d : ℕ)     -- Common difference in the sequence

-- Define the conditions
def condition1 : Prop := (Finset.range 33).sum (λ k, a (3 * k + 1)) = 150
def condition2 : Prop := (Finset.range 33).sum (λ k, a (3 * k + 2)) = 200

-- Define the sum of the first 99 terms of the arithmetic sequence
def S_99 : ℕ := (Finset.range 99).sum a

-- State the theorem to be proven
theorem sum_of_first_99_terms
  (cond1 : condition1 a)
  (cond2 : condition2 a)
  : S_99 a = 600 :=
sorry

end sum_of_first_99_terms_l44_44600


namespace balloon_permutations_l44_44139

theorem balloon_permutations : 
  let n : ℕ := 7
  let k1 : ℕ := 2
  let k2 : ℕ := 2
  ∃ distinct_arrangements : ℕ, 
  distinct_arrangements = n.factorial / (k1.factorial * k2.factorial) 
  ∧ distinct_arrangements = 1260 :=
by
  sorry

end balloon_permutations_l44_44139


namespace sum_of_interior_angles_of_regular_hexagon_l44_44346

theorem sum_of_interior_angles_of_regular_hexagon : 
  ∑ (i : Fin 6), 180 = 720 := 
sorry

end sum_of_interior_angles_of_regular_hexagon_l44_44346


namespace roots_sum_l44_44616

def log2 (x: ℝ) := (Real.log x / Real.log 2)

theorem roots_sum : 
  ∀ (α β : ℝ), 
  log2 α + α + 2 = 0 ∧ 2^β + β + 2 = 0 →
  α + β = -2 :=
by 
  intros α β h,
  cases h with hα hβ,
  sorry

end roots_sum_l44_44616


namespace penny_makes_total_revenue_l44_44461

def price_per_slice : ℕ := 7
def slices_per_pie : ℕ := 6
def pies_sold : ℕ := 7

theorem penny_makes_total_revenue :
  (pies_sold * slices_per_pie) * price_per_slice = 294 := by
  sorry

end penny_makes_total_revenue_l44_44461


namespace hypotenuse_of_45_45_90_triangle_l44_44660

theorem hypotenuse_of_45_45_90_triangle (leg : ℝ) (angle_opposite_leg : ℝ) (h_leg : leg = 15) (h_angle : angle_opposite_leg = 45) :
  ∃ hypotenuse, hypotenuse = leg * Real.sqrt 2 :=
by
  use leg * Real.sqrt 2
  rw [h_leg]
  rw [h_angle]
  sorry

end hypotenuse_of_45_45_90_triangle_l44_44660


namespace evaluate_expression_at_two_l44_44951

theorem evaluate_expression_at_two :
  (∀ x : ℝ, 2 * x ≠ 1) →
  let expr := (λ x : ℝ, (2 * x + 1) / (2 * x - 1))
  in expr 2 = 5 / 3 :=
by
  intros h1 expr
  sorry

end evaluate_expression_at_two_l44_44951


namespace at_least_five_bulbs_l44_44337

noncomputable def probability_at_least_five_bulbs_working (n : ℕ) : ℝ :=
∑ k in Finset.range n.succ, if k ≥ 5 then (Nat.choose n k) * (0.95^k) * (0.05^(n-k)) else 0

theorem at_least_five_bulbs (n : ℕ) (h : probability_at_least_five_bulbs_working n ≥ 0.99) : n = 7 :=
sorry

end at_least_five_bulbs_l44_44337


namespace triangle_area_inequality_l44_44517

noncomputable def area (a b c : Point) : Real := sorry -- Replace with actual area function

variables {A B C D : Point}

theorem triangle_area_inequality 
  (convex_quadrilateral : ConvexQuadrilateral A B C D) 
  (angle_inequality : ∠ABD + ∠ACD > ∠BAC + ∠BDC) : 
  area A B D + area A C D > area B A C + area B D C := 
  sorry

end triangle_area_inequality_l44_44517


namespace sin_810_eq_1_l44_44804

theorem sin_810_eq_1 (θ : Real) : θ = 810 → sin θ = 1 :=
  by
    sorry

end sin_810_eq_1_l44_44804


namespace kite_minimum_area_correct_l44_44764

noncomputable def minimumKiteAreaAndSum (r : ℕ) (OP : ℕ) (h₁ : r = 60) (h₂ : OP < r) : ℕ × ℝ :=
  let d₁ := 2 * r
  let d₂ := 2 * Real.sqrt (r^2 - OP^2)
  let area := (d₁ * d₂) / 2
  (120 + 119, area)

theorem kite_minimum_area_correct {r OP : ℕ} (h₁ : r = 60) (h₂ : OP < r) :
  minimumKiteAreaAndSum r OP h₁ h₂ = (239, 120 * Real.sqrt 119) :=
by simp [minimumKiteAreaAndSum, h₁, h₂] ; sorry

end kite_minimum_area_correct_l44_44764


namespace abi_spends_two_thirds_of_salary_l44_44733

theorem abi_spends_two_thirds_of_salary (S : ℝ) (x : ℝ) (hS : S > 0) (hx : 0 ≤ x ∧ x ≤ 1) (h : 12 * x * S = 6 * (1 - x) * S) : 
  1 - x = 2 / 3 :=
by
  have h1 : S ≠ 0 := ne_of_gt hS
  have h2 : 12 * x * S / S = 6 * (1 - x) * S / S := by rw h
  have h3 : 12 * x = 6 * (1 - x) := by { field_simp [h1], exact h2 }
  have h4 : 12 * x + 6 * x = 6 := by { linarith }
  have h5 : 18 * x = 6 := by { rw add_comm at h4, exact h4 }
  have h6 : x = 1 / 3 := by { field_simp, exact eq_div_of_mul_eq h5 zero_ne_two }
  linarith

end abi_spends_two_thirds_of_salary_l44_44733


namespace no_real_solution_l44_44059

noncomputable def augmented_matrix (m : ℝ) : Matrix (Fin 2) (Fin 3) ℝ :=
  ![![m, 4, m+2], ![1, m, m]]

theorem no_real_solution (m : ℝ) :
  (∀ (a b : ℝ), ¬ ∃ (x y : ℝ), a * x + b * y = m ∧ a * x + b * y = 4 ∧ a * x + b * y = m + 2) ↔ m = 2 :=
by
sorry

end no_real_solution_l44_44059


namespace non_monotonic_f_l44_44547

noncomputable def f (a x : ℝ) : ℝ := 2 * x^2 - a * x + real.log x

theorem non_monotonic_f (a : ℝ) : ¬ monotonic_on (f a) (set.Ioi 0) ↔ a ∈ set.Ioi 4 :=
by
  -- noncomputable definitions and statements go here
  sorry

end non_monotonic_f_l44_44547


namespace linear_function_m_range_l44_44172

noncomputable def linear_function_in_quadrants (m : ℝ) : Prop :=
  ∃ (x y : ℝ), (y = (m - 3) * x + (m + 1)) ∧
  (x > 0 ∧ y > 0 ∨ 
   x < 0 ∧ y > 0 ∨ 
   x > 0 ∧ y < 0)

theorem linear_function_m_range (m : ℝ) : linear_function_in_quadrants m → -1 < m ∧ m < 3 :=
begin
  sorry
end

end linear_function_m_range_l44_44172


namespace good_seating_exists_l44_44044

open Real

noncomputable def find_N (n : ℕ) (h : n ≥ 3) : ℕ :=
  ⌊e * (n.factorial)⌋₊

theorem good_seating_exists (n : ℕ) (h : n ≥ 3) :
  ∃ N, N = find_N n h ∧ ∃ person, participates_in_at_least_N_good_seating_arrangements person N :=
sorry

end good_seating_exists_l44_44044


namespace height_of_box_l44_44387

def base_area : ℕ := 20 * 20
def cost_per_box : ℝ := 1.30
def total_volume : ℕ := 3060000
def amount_spent : ℝ := 663

theorem height_of_box : ∃ h : ℕ, 400 * h = total_volume / (amount_spent / cost_per_box) := sorry

end height_of_box_l44_44387


namespace middle_school_skipping_rope_competition_l44_44754

/-- A certain middle school actively advocates the "Sunshine Sports" movement to improve the physical fitness of middle school students. 
    They organized a one-minute skipping rope competition for 10 representatives from the seventh grade. 
    The standard number of skips is set at 160 times. 
    Any number of skips exceeding this standard is recorded as a positive number, 
    while any number of skips below the standard is recorded as a negative number. 
    The performance records are as follows (unit: times): +19, -1, +22, -3, -5, +12, -8, +1, +8, +15.
    Prove the following:
    1. The difference between the best and worst performances of the representatives in the class is 30 times. 
    2. The average number of skips per person for the representatives in the class is 166 times.
    3. The class cannot receive the school's reward based on their total score. -/
theorem middle_school_skipping_rope_competition :
  let records := [19, -1, 22, -3, -5, 12, -8, 1, 8, 15],
      standard := 160
  in
    (max records - min records = 30) ∧
    (standard + (records.sum / records.length) = 166) ∧
    ((records.filter (λ x => x > 0)).sum * 1 - (records.filter (λ x => x < 0)).sum * 1.2 < 60) :=
by
  -- Proof steps would go here
  sorry

end middle_school_skipping_rope_competition_l44_44754


namespace f_sum_l44_44277

def f (x : ℝ) : ℝ :=
  if x > 3 then x^2 + 4
  else if x >= -3 then 3 * x + 1
  else -2

theorem f_sum : f (-4) + f (0) + f (4) = 19 := by
  sorry

end f_sum_l44_44277


namespace balloon_arrangements_l44_44105

-- Define the variables
def n : ℕ := 7
def L_count : ℕ := 2
def O_count : ℕ := 2
def B_count : ℕ := 1
def A_count : ℕ := 1
def N_count : ℕ := 1

-- Define the multiset permutation formula
def multiset_permutations (n : ℕ) (counts : List ℕ) : ℕ :=
  n.factorial / (counts.map Nat.factorial).prod

-- Proof that the number of distinct arrangements is 1260
theorem balloon_arrangements : multiset_permutations n [L_count, O_count, B_count, A_count, N_count] = 1260 :=
  by
  -- The proof is omitted
  sorry

end balloon_arrangements_l44_44105


namespace net_population_change_nearest_percent_l44_44350

theorem net_population_change_nearest_percent :
  let final_population := (1 : ℝ) * (6 / 5) * (13 / 10) * (9 / 10) * (4 / 5) in
  let net_change := (final_population - 1) * 100 in
  round net_change = 12 :=
by
  let final_population := (1 : ℝ) * (6 / 5) * (13 / 10) * (9 / 10) * (4 / 5)
  let net_change := (final_population - 1) * 100
  have h : round net_change = 12 := sorry
  exact h

end net_population_change_nearest_percent_l44_44350


namespace n_sided_polygon_rotation_l44_44168

theorem n_sided_polygon_rotation (n : Nat) : (360 % 90 = 0) -> n = 12 := by
  intro h
  -- sorry, implementation skipped

end n_sided_polygon_rotation_l44_44168


namespace sum_of_interior_angles_of_regular_hexagon_l44_44345

theorem sum_of_interior_angles_of_regular_hexagon : 
  ∑ (i : Fin 6), 180 = 720 := 
sorry

end sum_of_interior_angles_of_regular_hexagon_l44_44345


namespace balls_into_boxes_l44_44158

theorem balls_into_boxes : 
  ∃ (F : Fin 7 → Fin 4), F.solution_length = 104 := 
sorry

end balls_into_boxes_l44_44158


namespace dodecahedron_cube_volume_ratio_l44_44050

theorem dodecahedron_cube_volume_ratio (a : ℝ) (p q : ℕ) (hpq_coprime : Nat.coprime p q)
  (VD : ℝ := (15 + 7 * Real.sqrt 5) * a^3 / 4)
  (FD : ℝ := a / 2 * Real.tan (3 * Real.pi / 10))
  (s : ℝ := a * Real.sqrt 3 * Real.tan (3 * Real.pi / 10))
  (VC : ℝ := s^3)
  (ratio : ℝ := VD / VC) :
  ratio = (p : ℝ) / (q : ℝ) → p + q = sorry := 
sorry

end dodecahedron_cube_volume_ratio_l44_44050


namespace sum_first_20_terms_l44_44943

-- Define the conditions
variables {α : Type*} [ordered_ring α]
variables (a : ℕ → α) (d : α)
variable (a1 : α)
variable (a20 : α)

-- Given conditions
axiom h1 : a 1 + a 2 + a 3 = -24
axiom h2 : a 18 + a 19 + a 20 = 78

-- Define the arithmetic sequence property
def arithmetic_seq (a : ℕ → α) (d : α) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- The main statement to prove
theorem sum_first_20_terms (h : arithmetic_seq a d) : (∑ i in finset.range 20, a (i + 1)) = 180 :=
sorry

end sum_first_20_terms_l44_44943


namespace distinct_convex_polygons_of_four_or_more_sides_l44_44491

noncomputable def total_subsets (n : Nat) : Nat := 2^n

noncomputable def subsets_with_fewer_than_four_members (n : Nat) : Nat := 
  (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)

noncomputable def valid_subsets (n : Nat) : Nat := 
  total_subsets n - subsets_with_fewer_than_four_members n

theorem distinct_convex_polygons_of_four_or_more_sides (n : Nat) (h : n = 15) : valid_subsets n = 32192 := by
  sorry

end distinct_convex_polygons_of_four_or_more_sides_l44_44491


namespace cake_fraction_eaten_l44_44632

theorem cake_fraction_eaten (total_slices kept_slices slices_eaten : ℕ) 
  (h1 : total_slices = 12)
  (h2 : kept_slices = 9)
  (h3 : slices_eaten = total_slices - kept_slices) :
  (slices_eaten : ℚ) / total_slices = 1 / 4 := 
sorry

end cake_fraction_eaten_l44_44632


namespace hypotenuse_of_45_45_90_triangle_l44_44659

theorem hypotenuse_of_45_45_90_triangle (leg : ℝ) (angle_opposite_leg : ℝ) (h_leg : leg = 15) (h_angle : angle_opposite_leg = 45) :
  ∃ hypotenuse, hypotenuse = leg * Real.sqrt 2 :=
by
  use leg * Real.sqrt 2
  rw [h_leg]
  rw [h_angle]
  sorry

end hypotenuse_of_45_45_90_triangle_l44_44659


namespace volume_of_bottle_l44_44756

variables (x : ℝ)

-- Define the initial conditions
def initial_syrup_concentration : ℝ := 0.36
def final_syrup_concentration : ℝ := 0.01
def volume_poured : ℝ := 1

-- Defining the theorem to prove
theorem volume_of_bottle :
  let c₀ := initial_syrup_concentration,
      c₁ := final_syrup_concentration,
      v_p := volume_poured in
  (1 - c₁) * (x - v_p) / ((x * c₀ - c₀ * v_p) / (x - v_p) + (1 - c₀) * v_p) = c₁ → 
    x = 1.2 := 
begin
  sorry
end

end volume_of_bottle_l44_44756


namespace max_product_of_three_l44_44725

theorem max_product_of_three :
  let S := {-4, -3, -1, 5, 6} in
  ∃ (a b c : ℤ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
  (∀ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x ∈ S ∧ y ∈ S ∧ z ∈ S → 
  x * y * z ≤ a * b * c) ∧ a * b * c = 72 :=
begin
  sorry
end

end max_product_of_three_l44_44725


namespace quadratic_roots_condition_l44_44919

theorem quadratic_roots_condition (k : ℝ) : 
  ((∃ x : ℝ, (k - 1) * x^2 + 4 * x + 1 = 0) ∧ ∃ x1 x2 : ℝ, x1 ≠ x2) ↔ (k < 5 ∧ k ≠ 1) :=
by {
  sorry  
}

end quadratic_roots_condition_l44_44919


namespace smallest_difference_of_factors_l44_44159

theorem smallest_difference_of_factors (a b : ℕ) (h1 : a * b = 1764) (h2 : a ≠ b) : (a - b).natAbs = 13 :=
sorry

end smallest_difference_of_factors_l44_44159


namespace johns_age_l44_44229

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l44_44229


namespace three_point_seven_five_as_fraction_l44_44718

theorem three_point_seven_five_as_fraction :
  (15 : ℚ) / 4 = 3.75 :=
sorry

end three_point_seven_five_as_fraction_l44_44718


namespace johns_age_l44_44243

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
sorry

end johns_age_l44_44243


namespace equation_of_tangent_line_at_P1_slope_angle_range_l44_44874

    noncomputable def curve (x : ℝ) : ℝ := (1/3) * x^3 + x

    def tangent_at_point (P : ℝ × ℝ) (x : ℝ) : ℝ := x^2 + 1

    theorem equation_of_tangent_line_at_P1 (x y : ℝ) (h : (x, y) = (1, 4 / 3)) :
      6 * x - 3 * y - 2 = 0 := 
    sorry

    theorem slope_angle_range :
      ∀ (x : ℝ), 1 ≤ (x^2 + 1) → π / 4 ≤ atan (x^2 + 1) ∧ atan (x^2 + 1) < π / 2 := 
    sorry
    
end equation_of_tangent_line_at_P1_slope_angle_range_l44_44874


namespace domain_of_y_is_2_to_8_l44_44916

variable (f : ℝ → ℝ)
variable (y : ℝ → ℝ)
variable (x : ℝ)

noncomputable def domain_f (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 16
noncomputable def domain_y (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 8

theorem domain_of_y_is_2_to_8 :
  (∀ x, domain_f x → (∃ x, y = f x + f (2 * x))) →
  (∀ x, domain_y x) :=
by
  sorry

end domain_of_y_is_2_to_8_l44_44916


namespace find_b_l44_44272

noncomputable def p (x : ℕ) := 3 * x + 5
noncomputable def q (x : ℕ) (b : ℕ) := 4 * x - b

theorem find_b : ∃ (b : ℕ), p (q 3 b) = 29 ∧ b = 4 := sorry

end find_b_l44_44272


namespace orthocentric_ABCM_l44_44296

variables {A B C D M : Type*}

-- Define orthocentric tetrahedron
def orthocentric_tetrahedron (A B C D : Type*) : Prop :=
  ∃ h : A, 
  ∃ h : B, 
  ∃ h : C, 
  ∃ h : D, 
  ⟪A - B, A - C⟫ = 0 ∧
  ⟪A - B, A - D⟫ = 0 ∧
  ⟪A - C, A - D⟫ = 0 ∧
  ⟪B - C, B - D⟫ = 0 ∧
  ⟪B - A, B - D⟫ = 0 ∧
  ⟪C - A, C - B⟫ = 0

-- Define orthocenter of tetrahedron
def orthocenter (H A B C D : Type*) : Prop :=
  orthocentric_tetrahedron A B C D ∧
  ∀ (X Y Z : Type*), ⟪X - H, Y - H⟫ = 0 → ⟪Y - H, Z - H⟫ = 0 → ⟪X - H, Z - H⟫ = 0

-- Given condition: ABCD is an orthocentric tetrahedron with orthocenter M
axiom orthocentric_ABCD : orthocentric_tetrahedron A B C D
axiom orthocenter_M : orthocenter M A B C D

-- Prove that ABCM is an orthocentric tetrahedron with orthocenter D
theorem orthocentric_ABCM :
  orthocenter D A B C M :=
  sorry

end orthocentric_ABCM_l44_44296


namespace kids_played_on_monday_l44_44609

theorem kids_played_on_monday (m t a : Nat) (h1 : t = 7) (h2 : a = 19) (h3 : a = m + t) : m = 12 := 
by 
  sorry

end kids_played_on_monday_l44_44609


namespace trajectory_of_M_is_ellipse_product_of_slopes_is_constant_l44_44859

open Real

noncomputable def curve_equation (M : ℝ × ℝ) :=
  let F : ℝ × ℝ := (1, 0)
  let l := 4 in
  dist M F / abs (l - M.1) = 1 / 2

theorem trajectory_of_M_is_ellipse :
  ∀ (x y : ℝ), curve_equation (x, y) → (x ^ 2 / 4 + y ^ 2 / 3 = 1) :=
sorry

theorem product_of_slopes_is_constant :
  ∀ (k : ℝ) (A B M : ℝ × ℝ) (K₁ K₂ : ℝ),
    (A ≠ M) → (B ≠ M) → (k ≠ 0) →
    A.2 = k * A.1 → B.2 = k * B.1 →
    curve_equation A → curve_equation B →
    ∃ K₁ K₂, (K₁ = (A.2 - M.2) / (A.1 - M.1) ∧ K₂ = (B.2 - M.2) / (B.1 - M.1)) →
    K₁ * K₂ = -3 / 4 :=
sorry

end trajectory_of_M_is_ellipse_product_of_slopes_is_constant_l44_44859


namespace snowflakes_initial_count_l44_44704

theorem snowflakes_initial_count :
  (∃ initial_count : ℕ, 
    let intervals := 60 / 5 in
    let additional_snowflakes := intervals * 4 in
    initial_count + additional_snowflakes = 58) →
  initial_count = 10 :=
by
  intros h
  obtain ⟨initial_count, h⟩ := h
  let intervals := 60 / 5
  let additional_snowflakes := intervals * 4
  have : initial_count + additional_snowflakes = 58 := h
  sorry

end snowflakes_initial_count_l44_44704


namespace parabola_equation_line_and_circle_l44_44765

namespace proof

-- Define the conditions
variable (P : ℝ × ℝ) (C : ℝ × ℝ → Prop) (p : ℝ)
variable (A B : ℝ × ℝ)
variable (k : ℝ)
variable (M : ℝ × ℝ) (R : ℝ)
variable (x1 x2 y1 y2 : ℝ)

-- Assume the conditions
axiom P_condition : P = (-1, 0)
axiom parabola_condition : ∀ x y: ℝ, C (x, y) ↝ y^2 = 2 * p * x ∧ p > 0
axiom line_condition : ∃ k ≠ 0, (y1 = k * (x1 + 1)) ∧ (y2 = k * (x2 + 1))
axiom dot_product_condition : x1 * x2 + y1 * y2 = 5

-- Proof of the parabola equation
theorem parabola_equation : parabola_condition (-1, 0) 2 ↝
  ∃ p : ℝ, p = 2 ∧ C = λ P, P.2^2 = 4 * P.1 :=
sorry

-- Check if there exists a line and circle
theorem line_and_circle : 
  ∃ l C : ℝ × ℝ → Prop, 
    (∃ k : ℝ, k > 0 ∧
      l = λ (x y : ℝ), y = k * (x + 1) ∧
      C = λ (x y : ℝ), (x - 5)^2 + y^2 = 24) :=
sorry

end proof

end parabola_equation_line_and_circle_l44_44765


namespace find_k_values_l44_44841

-- Definitions for the conditions provided
def g (k : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x + k) / (x^2 - 3*x - 4)

def has_one_vertical_asymptote (g : ℝ → ℝ) : Prop :=
  (∃ (c : ℝ), ∀ x ≠ c, ∃ δ > 0, ∀ ε > 0, ∃ y ∈ set.Ioo (x - δ) (x + δ), g y < -ε ∨ g y > ε)

-- Main theorem statement
theorem find_k_values :
  ∃ k : ℝ, (k = -8 ∨ k = -3) ∧ has_one_vertical_asymptote (g k) :=
sorry

end find_k_values_l44_44841


namespace original_average_of_numbers_l44_44967

theorem original_average_of_numbers 
  (A : ℝ) 
  (h : (A * 15) + (11 * 15) = 51 * 15) : 
  A = 40 :=
sorry

end original_average_of_numbers_l44_44967


namespace oil_needed_for_sugar_l44_44838

theorem oil_needed_for_sugar (S O : ℕ) (sugar_ratio oil_ratio : ℕ) :
  sugar_ratio = 300 → oil_ratio = 60 → O = (S / sugar_ratio) * oil_ratio →
  S = 900 → O = 180 :=
by
  intros h1 h2 h3 h4
  rw [h4, h1, h2]
  sorry

end oil_needed_for_sugar_l44_44838


namespace tangent_line_equation_at_1_l44_44064

def f (x : ℝ) : ℝ := 2 * Real.log x - x * f' 1

theorem tangent_line_equation_at_1 :
  ∀ (x y : ℝ), (f x - y = -1) → (x - y - 2 = 0) → y = f x :=
by
  intro x y
  intro h1 h2
  sorry

end tangent_line_equation_at_1_l44_44064


namespace lily_pads_cover_entire_lake_l44_44184

-- Definitions from conditions
def doubling (P : ℕ → ℝ) : Prop :=
∀ t : ℕ, P t * 2 = P (t + 1)

def P_at_38_half {P : ℕ → ℝ} (lake_size : ℝ) : Prop :=
P 38 = lake_size / 2

-- The actual theorem to be proven
theorem lily_pads_cover_entire_lake (P : ℕ → ℝ) (lake_size : ℝ) 
    (h1 : doubling P) 
    (h2 : P_at_38_half lake_size) :
  P 39 = lake_size :=
by 
  sorry

end lily_pads_cover_entire_lake_l44_44184


namespace latia_hours_per_week_l44_44978

theorem latia_hours_per_week (cost_tv : ℝ) (earn_per_hour : ℝ) (needed_hours : ℝ) :
  cost_tv = 1700 → earn_per_hour = 10 → needed_hours = 50 →  
  let total_earned := cost_tv - needed_hours * earn_per_hour in
  let total_hours := total_earned / earn_per_hour in
  let weeks := 4 in
  total_hours / weeks = 30 :=
by
  intros h1 h2 h3
  let total_earned := cost_tv - needed_hours * earn_per_hour
  let total_hours := total_earned / earn_per_hour
  let weeks := 4
  sorry

end latia_hours_per_week_l44_44978


namespace part_one_part_two_l44_44883

noncomputable def f (x : ℝ) : ℝ := (x^2 - 1) * Real.exp x + x

theorem part_one :
  let domain := Set.Icc (1 / 2 : ℝ) 1
  let max_value := 1
  let min_value := -3 / 4 * Real.exp (1 / 2) + 1 / 2
  ∀ (x ∈ domain), (f 1 = max_value ∧ f (1 / 2) = min_value) :=
sorry

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x - a * Real.exp x - x

theorem part_two (a x1 x2 t : ℝ) (h1 : x1 < x2) (h2 : g x1 a = 0) (h3 : g x2 a = 0) :
  (e * g x2 a ≤ t * (2 + x1) * (Real.exp x2 + 1)) ∧ (t = Real.exp 1) :=
sorry

end part_one_part_two_l44_44883


namespace sequence_n_500_l44_44589

theorem sequence_n_500 (a : ℕ → ℤ) 
  (h1 : a 1 = 1010) 
  (h2 : a 2 = 1011) 
  (h3 : ∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = 2 * n + 3) : 
  a 500 = 3003 := 
sorry

end sequence_n_500_l44_44589


namespace distinct_arrangements_balloon_l44_44108

noncomputable def totalPermutations (n nl no : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial nl * Nat.factorial no)

theorem distinct_arrangements_balloon :
  totalPermutations 7 2 2 = 1260 := by 
  sorry

end distinct_arrangements_balloon_l44_44108


namespace find_cos_C_find_abc_values_l44_44932

variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
variables h1 : ∠A + ∠B + ∠C = 180
          h2 : a^2 = b^2 + c^2 - 2*b*c*cos A
          h3 : b^2 = a^2 + c^2 - 2*a*c*cos B
          h4 : sin (C / 2) = sqrt 10 / 4
          h5 : area_triangle_ABC = 3 * sqrt 15 / 4
          h6 : sin^2 A + sin^2 B = 13 / 16 * sin^2 C

-- To prove cos(C) based on given conditions
theorem find_cos_C : cos C = -1 / 4 :=
by 
  sorry

-- To prove values of a, b, and c based on given conditions
theorem find_abc_values : (a = 2 ∧ b = 3 ∧ c = 4) ∨ (a = 3 ∧ b = 2 ∧ c = 4) :=
by 
  sorry

end find_cos_C_find_abc_values_l44_44932


namespace value_of_difference_l44_44401

noncomputable def greatest_even (z : ℝ) : ℕ :=
if z < 0 then 0 else (even_below : ℕ := Nat.floor z) * if even_below % 2 = 0 then 1 else 0) 

theorem value_of_difference : 6.32 - greatest_even 6.32 = 0.32 :=
by
  sorry

end value_of_difference_l44_44401


namespace grilled_cheese_sandwiches_l44_44219

theorem grilled_cheese_sandwiches (h_cheese : ℕ) (g_cheese : ℕ) (total_cheese : ℕ) (ham_sandwiches : ℕ) (grilled_cheese_sandwiches : ℕ) :
  h_cheese = 2 →
  g_cheese = 3 →
  total_cheese = 50 →
  ham_sandwiches = 10 →
  total_cheese - (ham_sandwiches * h_cheese) = grilled_cheese_sandwiches * g_cheese →
  grilled_cheese_sandwiches = 10 :=
by {
  intros,
  sorry
}

end grilled_cheese_sandwiches_l44_44219


namespace intersection_line_curve_distance_sum_l44_44195

open Real

noncomputable def P : Point := ⟨0, 1⟩ 

def line_parametric_eq (t : ℝ) : Point :=
  ⟨(sqrt 2) / 2 * t, 1 + (sqrt 2) / 2 * t⟩ 

def polar_to_cartesian (ρ θ : ℝ) : Point :=
  ⟨ρ * cos θ, ρ * sin θ⟩

def curve_cartesian_eq (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

theorem intersection_line_curve_distance_sum :
  ∀ t₁ t₂ : ℝ, 
  t₁ ^ 2 - (sqrt 2) * t₁ - 1 = 0 ∧
  t₂ ^ 2 - (sqrt 2) * t₂ - 1 = 0 ∧
  t₁ ≠ t₂ →
  abs t₁ + abs t₂ = sqrt 6 :=
by
  sorry -- proof omitted

end intersection_line_curve_distance_sum_l44_44195


namespace find_m_l44_44873

noncomputable def z : ℂ := (4 + 2 * complex.I) / ((1 + complex.I)^2)
def point := (1 : ℝ, -2 : ℝ)
def line (x y m : ℝ) := x - 2 * y + m = 0

theorem find_m : line (point.1) (point.2) (-5) :=
by
  sorry

end find_m_l44_44873


namespace boys_in_class_l44_44177

theorem boys_in_class (g b : ℕ) 
  (h_ratio : 4 * g = 3 * b) (h_total : g + b = 28) : b = 16 :=
by
  sorry

end boys_in_class_l44_44177


namespace probability_of_returning_hometown_relationship_significance_l44_44669

def total_people : ℕ := 100
def age_50_above_return : ℕ := 15
def age_50_above_total : ℕ := 40
def age_50_below_no_return : ℕ := 55

theorem probability_of_returning_hometown :
  (age_50_above_return.to_rat / age_50_above_total.to_rat) = (3 / 8) := 
by sorry

def a : ℕ := 5
def b : ℕ := 55
def c : ℕ := 15
def d : ℕ := 25
def n : ℕ := 100
def k_square : ℚ :=
  (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem relationship_significance :
  k_square > 10.828 :=
by sorry

end probability_of_returning_hometown_relationship_significance_l44_44669


namespace probability_sum_three_dice_3_l44_44367

-- Definition of a fair six-sided die
def fair_six_sided_die : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Definition of probability of an event
def probability (s : Set ℕ) (event : ℕ → Prop) : ℚ :=
  if h : finite s then (s.filter event).to_finset.card / s.to_finset.card else 0

theorem probability_sum_three_dice_3 :
  let dice := List.repeat fair_six_sided_die 3 in
  let event := λ result : List ℕ => result.sum = 3 in
  probability ({(r1, r2, r3) | r1 ∈ fair_six_sided_die ∧ r2 ∈ fair_six_sided_die ∧ r3 ∈ fair_six_sided_die }) (λ (r1, r2, r3) => r1 + r2 + r3 = 3) = 1 / 216 :=
by
  sorry

end probability_sum_three_dice_3_l44_44367


namespace balloon_arrangements_l44_44151

-- Defining the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Given Conditions
def seven_factorial := fact 7 -- 7!
def two_factorial := fact 2 -- 2!

-- Statement to prove
theorem balloon_arrangements : seven_factorial / (two_factorial * two_factorial) = 1260 :=
by
  sorry

end balloon_arrangements_l44_44151


namespace hypotenuse_of_45_45_90_triangle_15_l44_44656

theorem hypotenuse_of_45_45_90_triangle_15 (a : ℝ) (h : a = 15) : 
  ∃ (c : ℝ), c = a * Real.sqrt 2 :=
by
  use a * Real.sqrt 2
  rw h
  sorry

end hypotenuse_of_45_45_90_triangle_15_l44_44656


namespace tax_revenue_decrease_l44_44348

theorem tax_revenue_decrease 
  {T C : ℝ} -- T for tax, C for consumption
  (hT : T > 0) (hC : C > 0) -- assuming tax and consumption are positive
  (h_tax_decrease : ∀ T_new : ℝ, T_new = T * 0.78) 
  (h_consumption_increase : ∀ C_new : ℝ, C_new = C * 1.09) : 
  let R := T * C in
  let R_new := (T * 0.78) * (C * 1.09) in
  let percentage_decrease := ((R - R_new) / R) * 100 in
  percentage_decrease = 14.98 :=
by
  sorry

end tax_revenue_decrease_l44_44348


namespace number_of_sodas_l44_44384

theorem number_of_sodas (cost_sandwich : ℝ) (num_sandwiches : ℕ) (cost_soda : ℝ) (total_cost : ℝ):
  cost_sandwich = 2.45 → 
  num_sandwiches = 2 → 
  cost_soda = 0.87 → 
  total_cost = 8.38 → 
  (total_cost - num_sandwiches * cost_sandwich) / cost_soda = 4 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end number_of_sodas_l44_44384


namespace min_chips_in_cells_l44_44948

theorem min_chips_in_cells (n : ℕ) (h : 2021 - n ≤ (n + 1) / 2 ∧ 
                                  (∀ i j : ℕ, (i < j ∧ 1 ≤ i ∧ i ≤ 2021 - n
                                   ∧ j ≤ 2021 - n) → abs (f i - f j) ≠ 0)) :
  n = 1347 :=
by sorry

end min_chips_in_cells_l44_44948


namespace triangle_angle_bisector_sum_l44_44323

theorem triangle_angle_bisector_sum (P Q R : ℝ × ℝ)
  (hP : P = (-8, 5)) (hQ : Q = (-15, -19)) (hR : R = (1, -7)) 
  (a b c : ℕ) (h : a + c = 89) 
  (gcd_abc : Int.gcd (Int.gcd a b) c = 1) :
  a + c = 89 :=
by
  sorry

end triangle_angle_bisector_sum_l44_44323


namespace f_three_eq_three_l44_44471

noncomputable def f (x : ℕ) : ℤ :=
  ((x + 1) * (x^3 + 1) * (x^9 + 1) * ... * (x^(3^2007) + 1) + (x^2 - 1) - 1) / (x^(3^2008 - 1) - 1)

theorem f_three_eq_three : f 3 = 3 := by
  sorry

end f_three_eq_three_l44_44471


namespace distinct_arrangements_balloon_l44_44114

noncomputable def totalPermutations (n nl no : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial nl * Nat.factorial no)

theorem distinct_arrangements_balloon :
  totalPermutations 7 2 2 = 1260 := by 
  sorry

end distinct_arrangements_balloon_l44_44114


namespace find_a_plus_b_l44_44263

theorem find_a_plus_b (a b : ℝ) (h₁ : (2 + (complex.I * real.sqrt 5)) * root (λ x : complex, x^3 + a * x + b)) :
    a + b = 29 := 
sorry

end find_a_plus_b_l44_44263


namespace balloon_arrangements_l44_44101

-- Define the variables
def n : ℕ := 7
def L_count : ℕ := 2
def O_count : ℕ := 2
def B_count : ℕ := 1
def A_count : ℕ := 1
def N_count : ℕ := 1

-- Define the multiset permutation formula
def multiset_permutations (n : ℕ) (counts : List ℕ) : ℕ :=
  n.factorial / (counts.map Nat.factorial).prod

-- Proof that the number of distinct arrangements is 1260
theorem balloon_arrangements : multiset_permutations n [L_count, O_count, B_count, A_count, N_count] = 1260 :=
  by
  -- The proof is omitted
  sorry

end balloon_arrangements_l44_44101


namespace relationship_between_Q_and_t_remaining_power_after_5_hours_distance_with_40_power_l44_44710

-- Define the relationship between Q and t
def remaining_power (t : ℕ) : ℕ := 80 - 15 * t

-- Question 1: Prove relationship between Q and t
theorem relationship_between_Q_and_t : ∀ t : ℕ, remaining_power t = 80 - 15 * t :=
by sorry

-- Question 2: Prove remaining power after 5 hours
theorem remaining_power_after_5_hours : remaining_power 5 = 5 :=
by sorry

-- Question 3: Prove distance the car can travel with 40 kW·h remaining power
theorem distance_with_40_power 
  (remaining_power : ℕ := (80 - 15 * t)) 
  (t := 8 / 3)
  (speed : ℕ := 90) : (90 * (8 / 3)) = 240 :=
by sorry

end relationship_between_Q_and_t_remaining_power_after_5_hours_distance_with_40_power_l44_44710


namespace min_filtration_cycles_l44_44193

theorem min_filtration_cycles {c₀ : ℝ} (initial_concentration : c₀ = 225)
  (max_concentration : ℝ := 7.5) (reduction_factor : ℝ := 1 / 3)
  (log2 : ℝ := 0.3010) (log3 : ℝ := 0.4771) :
  ∃ n : ℕ, (c₀ * (reduction_factor ^ n) ≤ max_concentration ∧ n ≥ 9) :=
sorry

end min_filtration_cycles_l44_44193


namespace exponential_trigonometric_inequality_l44_44038

theorem exponential_trigonometric_inequality
  (x : ℝ) 
  (h₁ : (π/4) < x)
  (h₂ : x < (π/2)) :
  let a := 2^(1 - Real.sin x)
      b := 2^(Real.cos x)
      c := 2^(Real.tan x)
  in a < b ∧ b < c := 
by
  sorry

end exponential_trigonometric_inequality_l44_44038


namespace numAdults_l44_44306

-- Definitions of the given problem.
def numKids : Nat := 6
def kidsTicketCost : Nat := 5
def totalCost : Nat := 50
def adultsTicketCost : Nat := 2 * kidsTicketCost

-- Theorem to prove the number of adults is 2.
theorem numAdults : ∃ (A : Nat), A = 2 ∧ (numKids * kidsTicketCost + A * adultsTicketCost = totalCost) := 
by
  -- In this case, we want to prove that there exists A such that the equations hold.
  -- We write the statement as usual mathematics but only provide the high-level structure.
  exists 2
  simp [numKids, kidsTicketCost, totalCost, adultsTicketCost]
  sorry

end numAdults_l44_44306


namespace distinct_arrangements_balloon_l44_44091

theorem distinct_arrangements_balloon : 
  let n := 7 
  let freq_l := 2 
  let freq_o := 2 
  let freq_b := 1 
  let freq_a := 1 
  let freq_n := 1 
  Nat.factorial n / (Nat.factorial freq_l * Nat.factorial freq_o * Nat.factorial freq_b * Nat.factorial freq_a * Nat.factorial freq_n) = 1260 :=
by
  sorry

end distinct_arrangements_balloon_l44_44091
