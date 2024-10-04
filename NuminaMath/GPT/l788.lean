import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Basic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GCDMonoid.Basic
import Mathlib.Algebra.GcdMonoid.Basic
import Mathlib.Algebra.GeomSeq
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Matrix.Basic
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Order.Sqrt
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Fin.VecNotation
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.Div
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Gcd.Basic
import Mathlib.Data.Nat.LCM
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.LinearAlgebra.Projection
import Mathlib.Logic.Basic
import Mathlib.NumberTheory.Divisors
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic
import Mathlib.Topology.Sequences

namespace prime_factors_upper_bound_l788_788540

theorem prime_factors_upper_bound (k n : ℕ) (h1 : k ≥ 2)
  (h2 : ∀ m : ℕ, (1 ≤ m ∧ m < (n:ℚ)^(1/k : ℚ)) → m ∣ n) : 
  ∃ t : ℕ, t ≤ 2 * k - 1 ∧ ∀ p ∈ (nat.factors n).to_finset, is_prime p :=
by
  sorry

end prime_factors_upper_bound_l788_788540


namespace parabola_and_circle_eq_line_A2A3_tangent_l788_788625

-- Define the conditions of the problem
-- Vertex of the parabola at the origin and focus on the x-axis
def parabola_eq : Prop := ∃ p > 0, ∀ x y : ℝ, (y^2 = 2 * p * x ↔ (x, y) ∈ C)

-- Define line l: x = 1
def line_l (x y : ℝ) : Prop := x = 1

-- Define the parabola C and the points of intersection P and Q
def intersection_points (y : ℝ) : Prop := (1, y) ∈ C

-- Define the perpendicularity condition OP ⊥ OQ
def perpendicular_condition (P Q : ℝ × ℝ) : Prop := (∃ p > 0, P = (1, sqrt p) ∧ Q = (1, -sqrt p))

-- Define the point M and its associated circle M tangent to line l
def point_M : ℝ × ℝ := (2, 0)

def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the points A1, A2, A3 on parabola C
def on_parabola (A : ℝ × ℝ) : Prop := (∃ p > 0, A.2^2 = 2 * p * A.1)

-- Define that lines A1A2 and A1A3 are tangent to circle M
def tangent_to_circle (A₁ A₂ : ℝ × ℝ) : Prop := sorry

-- Prove the equation of parabola C and circle M
theorem parabola_and_circle_eq : (∀ x y : ℝ, y^2 = x ∧ (x - 2)^2 + y^2 = 1) :=
by
  sorry

-- Prove the position relationship between line A2A3 and circle M
theorem line_A2A3_tangent (A₁ A₂ A₃ : ℝ × ℝ) :
    on_parabola A₁ ∧ on_parabola A₂ ∧ on_parabola A₃ ∧ tangent_to_circle A₁ A₂ ∧ tangent_to_circle A₁ A₃ →
    (∃ l_tangent : ℝ, tangent_to_circle A₂ A₃) :=
by
  sorry

end parabola_and_circle_eq_line_A2A3_tangent_l788_788625


namespace angle_kpm_45_l788_788587

/-- Given three mutually externally tangent circles centered at points A, B, C 
    such that ∠ABC = 90°, with points K, P, M being the points of tangency, 
    where P lies on the side AC, the angle ∠KPM is 45°. -/
theorem angle_kpm_45
  (A B C K P M : Type)
  (h1 : ∠ B A C = 90)
  (h2 : P ∈ line A C) :
  ∠ K P M = 45 := 
sorry

end angle_kpm_45_l788_788587


namespace hyperbola_eccentricity_l788_788844

/-- The geometry problem regarding the hyperbola and parabola. -/
theorem hyperbola_eccentricity
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (has : ∃ x y, x^2 / a^2 - y^2 / b^2 = 1 ∧ y = b / a * x ∧ x^2 = y - 1) :
  let c := sqrt (a^2 + b^2) in
  let e := c / a in
  e = sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_l788_788844


namespace compute_expression_l788_788933

variable (a b : ℚ)
variable (h₁ : a = 3 / 5)
variable (h₂ : b = 2 / 3)

theorem compute_expression : a^2 * b^(-3) = 243 / 200 :=
by
  rw [h₁, h₂]
  sorry

end compute_expression_l788_788933


namespace minimum_distance_on_C2_and_line_l_l788_788494

noncomputable def curve_C1 (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

noncomputable def scaling_transform (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 / 2, p.2)

noncomputable def curve_C2 (α : ℝ) : ℝ × ℝ :=
  (Real.cos α, 2 * Real.sin α)

noncomputable def line_l (x y : ℝ) : Prop :=
  x + y + 6 = 0

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem minimum_distance_on_C2_and_line_l :
  ∃ P Q : ℝ × ℝ, curve_C2 (P.1) ∧ line_l Q.1 Q.2 ∧ 
  ∀ p', curve_C2 (p'.1) → ∀ q', line_l q'.1 q'.2 → 
  distance P Q ≤ distance p' q' := 
  ∃ PQ_min_distance_value : ℝ, PQ_min_distance_value = 3 * Real.sqrt 2 - Real.sqrt 10 / 2 := sorry

end minimum_distance_on_C2_and_line_l_l788_788494


namespace projection_onto_vector_is_expected_l788_788309

def projection_matrix (u: ℝ × ℝ) : matrix (fin 2) (fin 2) ℝ :=
  let ⟨x, y⟩ := u in 
  (1 / (x^2 + y^2)) • (matrix.col_vec u ⬝ (matrix.transpose (matrix.row_vec u)))

def expected_matrix : matrix (fin 2) (fin 2) ℝ :=
  ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]]

theorem projection_onto_vector_is_expected :
  projection_matrix (3, 4) = expected_matrix := by
  sorry

end projection_onto_vector_is_expected_l788_788309


namespace sales_tax_is_5_percent_l788_788241

theorem sales_tax_is_5_percent :
  let cost_tshirt := 8
  let cost_sweater := 18
  let cost_jacket := 80
  let discount := 0.10
  let num_tshirts := 6
  let num_sweaters := 4
  let num_jackets := 5
  let total_cost_with_tax := 504
  let total_cost_before_discount := (num_jackets * cost_jacket)
  let discount_amount := discount * total_cost_before_discount
  let discounted_cost_jackets := total_cost_before_discount - discount_amount
  let total_cost_before_tax := (num_tshirts * cost_tshirt) + (num_sweaters * cost_sweater) + discounted_cost_jackets
  let sales_tax := (total_cost_with_tax - total_cost_before_tax)
  let sales_tax_percentage := (sales_tax / total_cost_before_tax) * 100
  sales_tax_percentage = 5 := by
  sorry

end sales_tax_is_5_percent_l788_788241


namespace number_of_distinct_prime_factors_30_factorial_l788_788855

theorem number_of_distinct_prime_factors_30_factorial : 
  ∃ s : Finset ℕ, (∀ p ∈ s, Prime p) ∧ (∏ x in s, x ≤ 30) ∧ (s.card = 10) :=
sorry

end number_of_distinct_prime_factors_30_factorial_l788_788855


namespace projection_matrix_correct_l788_788317

variables {R : Type*} [field R] [decidable_eq R]
variables (x y : R)
def vector_v : matrix (fin 2) (fin 1) R := ![![3], ![4]]
def vector_u : matrix (fin 2) (fin 1) R := ![![x], ![y]]
def projection_matrix : matrix (fin 2) (fin 2) R := ![![9/25, 12/25], ![12/25, 16/25]]

theorem projection_matrix_correct :
  (projection_matrix R) ⬝ (vector_u x y) = (25 : R)⁻¹ • (transpose (vector_v 3 4) ⬝ (vector_u x y)) ⬝ (vector_v 3 4) := 
sorry

end projection_matrix_correct_l788_788317


namespace arctan_asymptotic_equivalent_l788_788561

theorem arctan_asymptotic_equivalent (c : ℝ) :
  tendsto (λ x : ℝ, (arctan (c * x)) / (c * x)) (𝓝 0) (𝓝 1) :=
sorry

end arctan_asymptotic_equivalent_l788_788561


namespace tan_alpha_plus_pi_over_4_l788_788408

theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h : 2 * sin (2 * α) = 1 + cos (2 * α)) : 
  tan (α + real.pi / 4) = -1 ∨ tan (α + real.pi / 4) = 3 :=
by {
  -- Proof goes here
  sorry
}

end tan_alpha_plus_pi_over_4_l788_788408


namespace limit_r_as_m_to_zero_l788_788766

noncomputable def L (m : ℝ) := -real.sqrt (m + 4)

theorem limit_r_as_m_to_zero : 
  tendsto (λ m, (L (-m) - L m) / m) (nhds 0) (nhds (1 / 2)) := 
sorry

end limit_r_as_m_to_zero_l788_788766


namespace chips_per_bag_l788_788127

theorem chips_per_bag
  (calories_per_chip : ℕ)
  (cost_per_bag : ℕ)
  (target_calories : ℕ)
  (total_cost : ℕ)
  (calories_per_chip_eq : calories_per_chip = 10)
  (cost_per_bag_eq : cost_per_bag = 2)
  (target_calories_eq : target_calories = 480)
  (total_cost_eq : total_cost = 4)
  : let chips_needed := target_calories / calories_per_chip
    let bags_needed := total_cost / cost_per_bag
    let chips_per_bag := chips_needed / bags_needed
    in chips_per_bag = 24 := by
  sorry

end chips_per_bag_l788_788127


namespace number_of_factors_of_m_l788_788537

theorem number_of_factors_of_m :
  let m := 2^5 * 3^6 * 5^7 in
  ∀ n : ℕ, n = (5+1)*(6+1)*(7+1) → ∃ d : ℕ, d ∣ m ∧ (6 * 7 * 8 = n) :=
by
  sorry

end number_of_factors_of_m_l788_788537


namespace area_of_ZPQ_l788_788897

theorem area_of_ZPQ
  (XYZ : Type) [triangle XYZ]
  (P Q : XYZ)
  (XZ XY : XYZ)
  (midpoint_XZ_P : midpoint XZ P)
  (midpoint_XY_Q : midpoint XY Q)
  (area_XYZ : area XYZ = 36) :
  area (triangle ZPQ) = 9 := 
sorry

end area_of_ZPQ_l788_788897


namespace triangle_equilateral_l788_788451

variables {V : Type*} [inner_product_space ℝ V]

def is_equilateral_triangle (A B C : V) := 
  dist A B = dist B C ∧ dist B C = dist C A

theorem triangle_equilateral
  (A B C : V)
  (a := B - C)
  (b := C - A)
  (c := A - B)
  (h1 : inner a b = 0)
  (h2 : inner b c = 0)
  (h3 : inner c a = 0) :
  is_equilateral_triangle A B C :=
  sorry

end triangle_equilateral_l788_788451


namespace prob_neq_zero_l788_788680

noncomputable def probability_no_one (a b c d : ℕ) : ℚ :=
  if 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ 1 ≤ d ∧ d ≤ 6 
  then (5/6)^4 
  else 0

theorem prob_neq_zero (a b c d : ℕ) :
  (1 ≤ a) ∧ (a ≤ 6) ∧ (1 ≤ b) ∧ (b ≤ 6) ∧ (1 ≤ c) ∧ (c ≤ 6) ∧ (1 ≤ d) ∧ (d ≤ 6) →
  (a - 1) * (b - 1) * (c - 1) * (d - 1) ≠ 0 ↔ 
  probability_no_one a b c d = 625/1296 :=
by
  sorry

end prob_neq_zero_l788_788680


namespace range_c_of_sets_l788_788443

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem range_c_of_sets (c : ℝ) (h₀ : c > 0)
  (A := { x : ℝ | log2 x < 1 })
  (B := { x : ℝ | 0 < x ∧ x < c })
  (hA_union_B_eq_B : A ∪ B = B) :
  c ≥ 2 :=
by
  -- Minimum outline is provided, the proof part is replaced with "sorry" to indicate the point to be proved
  sorry

end range_c_of_sets_l788_788443


namespace bounds_T_n_l788_788810

-- Define given sequence
def sequence_a : ℕ → ℝ × ℝ 
  | k => 
    let Δ := (2^k + 3*k)² - 4 * 3*k * 2^k
    let root1 := ((2^k + 3*k) + real.sqrt Δ) / 2
    let root2 := ((2^k + 3*k) - real.sqrt Δ) / 2
    if root1 <= root2 then (root1, root2) else (root2, root1)

-- Define f(n)
def f (n : ℕ) : ℝ := 
  (1 / 2) * ((real.abs (real.sin n) / real.sin n) + 3)

-- Define T_n
def T_n (n : ℕ) : ℝ :=
  ∑ i in finset.range n, (-1)^(f (i + 2)) / (sequence_a (i+1)).1 / (sequence_a (i+1)).2

-- State the theorem
theorem bounds_T_n (n : ℕ) (h : n > 0) : 
  (1 / 6) ≤ T_n n ∧ T_n n ≤ (5 / 24) := 
sorry

end bounds_T_n_l788_788810


namespace OH_squared_l788_788926

/-- 
Given:
  O is the circumcenter of triangle ABC.
  H is the orthocenter of triangle ABC.
  a, b, and c are the side lengths of triangle ABC.
  R is the circumradius of triangle ABC.
  R = 5.
  a^2 + b^2 + c^2 = 50.

Prove:
  OH^2 = 175.
-/
theorem OH_squared (a b c R : ℝ) (hR : R = 5) (habc : a^2 + b^2 + c^2 = 50) :
  let OH_squared := 9 * R^2 - (a^2 + b^2 + c^2) in
  OH_squared = 175 :=
by
  sorry

end OH_squared_l788_788926


namespace acute_triangle_altitude_inequality_l788_788483

theorem acute_triangle_altitude_inequality (a b c d e f : ℝ) 
  (A B C : ℝ) 
  (acute_triangle : (d = b * Real.sin C) ∧ (d = c * Real.sin B) ∧
                    (e = a * Real.sin C) ∧ (f = a * Real.sin B))
  (projections : (de = b * Real.cos B) ∧ (df = c * Real.cos C))
  : (de + df ≤ a) := 
sorry

end acute_triangle_altitude_inequality_l788_788483


namespace sum_of_three_pairwise_rel_prime_integers_l788_788654

theorem sum_of_three_pairwise_rel_prime_integers (a b c : ℕ)
  (h1: 1 < a) (h2: 1 < b) (h3: 1 < c)
  (prod: a * b * c = 216000)
  (rel_prime_ab : Nat.gcd a b = 1)
  (rel_prime_ac : Nat.gcd a c = 1)
  (rel_prime_bc : Nat.gcd b c = 1) : 
  a + b + c = 184 := 
sorry

end sum_of_three_pairwise_rel_prime_integers_l788_788654


namespace max_segments_no_tetrahedron_l788_788407

-- Define the problem conditions
structure Point (α : Type) :=
  (A B C D E F : α)

def no_four_points_coplanar (α : Type) [PlaneSpace α] (p : Point α) : Prop :=
  -- This definition can be nuanced, but conceptually ensures no set of 4 points are coplanar
  ∀ (subset : finite_set_of_4_points p), ¬ coplanar subset

-- Define the maximum number of line segments without forming a tetrahedron
def max_non_tetrahedron_segments : ℕ := 12

-- The theorem statement
theorem max_segments_no_tetrahedron {α : Type} [PlaneSpace α] (p : Point α) 
  (h : no_four_points_coplanar α p) : 
  ∃ (segments : list (α × α)), 
    (forall_segment_valid segments p) ∧ 
    (segment_count segments = max_non_tetrahedron) ∧ 
    (no_tetrahedron_formed segments) :=
by
  sorry

end max_segments_no_tetrahedron_l788_788407


namespace average_growth_rate_equation_l788_788229

-- Define the current and target processing capacities
def current_capacity : ℝ := 1000
def target_capacity : ℝ := 1200

-- Define the time period in months
def months : ℕ := 2

-- Define the monthly average growth rate
variable (x : ℝ)

-- The statement to be proven: current capacity increased by the growth rate over 2 months equals the target capacity 
theorem average_growth_rate_equation :
  current_capacity * (1 + x) ^ months = target_capacity :=
sorry

end average_growth_rate_equation_l788_788229


namespace divide_stones_into_heaps_l788_788955

-- Definitions based on the identified conditions
variable (Heaps : List ℕ) -- list of heap sizes
variable (n_stones : ℕ) -- total number of stones
variable (n_heaps : ℕ) -- number of heaps

-- Conditions
axiom total_stones : n_stones = 660
axiom total_heaps : n_heaps = 30
axiom heap_size_condition : ∀ (h₁ h₂ : ℕ), h₁ ∈ Heaps → h₂ ∈ Heaps → h₁ ≤ 2 * h₂

-- Theorem statement
theorem divide_stones_into_heaps : 
  ∃ Heaps, Heaps.length = n_heaps ∧ Heaps.sum = n_stones ∧
  ∀ (h₁ h₂ : ℕ), h₁ ∈ Heaps → h₂ ∈ Heaps → h₁ ≤ 2 * h₂ :=
by
  sorry

end divide_stones_into_heaps_l788_788955


namespace wendy_time_per_piece_l788_788667

noncomputable def time_per_piece_of_furniture 
  (chairs : ℕ) (tables : ℕ) (total_time : ℕ) : ℕ :=
total_time / (chairs + tables)

theorem wendy_time_per_piece 
  (chairs : ℕ) (tables : ℕ) (total_time : ℕ)
  (h_chairs : chairs = 4) 
  (h_tables : tables = 4) 
  (h_total_time : total_time = 48) :
  time_per_piece_of_furniture chairs tables total_time = 6 :=
by
  simp [time_per_piece_of_furniture, h_chairs, h_tables, h_total_time]
  sorry

end wendy_time_per_piece_l788_788667


namespace projection_matrix_is_correct_l788_788355

noncomputable def projectionMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  let v : Fin 2 → ℝ := ![3, 4]
  (1 / (v 0 ^ 2 + v 1 ^ 2)) • (λ i j, v i * v j)

theorem projection_matrix_is_correct :
  projectionMatrix = ![![9/25, 12/25], ![12/25, 16/25]] :=
by
  sorry

end projection_matrix_is_correct_l788_788355


namespace parabola_equation_circle_m_equation_line_a2a3_tangent_to_circle_m_l788_788634

noncomputable theory
open_locale classical

-- Definitions and conditions
def parabola_vertex_origin (x y : ℝ) : Prop := ∃ p : ℝ, p > 0 ∧ y^2 = 2 * p * x
def line_intersects_parabola_perpendicularly : Prop :=
  ∃ p : ℝ, p = 1 / 2 ∧ parabola_vertex_origin 1 p

def circle_m_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1
def line_tangent_to_circle_m (l : ℝ → ℝ) : Prop := ∀ x y : ℝ, circle_m_eq x y → l x = y

def points_on_parabola_and_tangent (A1 A2 A3 : ℝ × ℝ) : Prop :=
  parabola_vertex_origin A1.1 A1.2 ∧
  parabola_vertex_origin A2.1 A2.2 ∧
  parabola_vertex_origin A3.1 A3.2 ∧
  line_tangent_to_circle_m (λ y, A1.1 * y + A1.2) ∧
  line_tangent_to_circle_m (λ y, A1.1 * y + A3.2)

-- Statements to prove
theorem parabola_equation : ∃ C : ℝ → ℝ → Prop, (C = parabola_vertex_origin) := sorry
theorem circle_m_equation : ∃ M : ℝ → ℝ → Prop, (M = circle_m_eq) := sorry
theorem line_a2a3_tangent_to_circle_m :
  ∀ A1 A2 A3 : ℝ × ℝ, 
  (points_on_parabola_and_tangent A1 A2 A3) →
  ∃ l : ℝ → ℝ, line_tangent_to_circle_m l := sorry

end parabola_equation_circle_m_equation_line_a2a3_tangent_to_circle_m_l788_788634


namespace similarity_coefficients_of_triangle_l788_788247

-- Initial triangle sides
variables (a b c : ℝ)
-- Condition: the triangle has sides 2, 3, and 3
axiom sides_eq : a = 2 ∧ b = 3 ∧ c = 3
-- Similarity coefficients to prove
noncomputable def similarity_coefficients (k₁ k₂ k₃ k₄ : ℝ) : Prop :=
  k₁ = 1/2 ∨ (k₁ = 6/13 ∧ k₂ = 4/13 ∧ k₃ = 9/13 ∧ k₄ = 6/13)

-- Problem statement: proving the similarity coefficients
theorem similarity_coefficients_of_triangle :
  ∃ (k₁ k₂ k₃ k₄ : ℝ), similarity_coefficients k₁ k₂ k₃ k₄ :=
begin
  have eq_sides : a = 2 ∧ b = 3 ∧ c = 3 := sides_eq,
  sorry
end

end similarity_coefficients_of_triangle_l788_788247


namespace find_point_on_xaxis_l788_788400

theorem find_point_on_xaxis (x : ℝ) :
  let A := (-1, 3) in
  let B := (2, 6) in
  let P := (x, 0) in
  dist P A = dist P B → P = (5, 0) :=
by
  sorry

end find_point_on_xaxis_l788_788400


namespace original_number_is_correct_l788_788238

noncomputable def original_number : ℝ :=
  let x := 11.26666666666667
  let y := 30.333333333333332
  x + y

theorem original_number_is_correct (x y : ℝ) (h₁ : 10 * x + 22 * y = 780) (h₂ : y = 30.333333333333332) : 
  original_number = 41.6 :=
by
  sorry

end original_number_is_correct_l788_788238


namespace number_of_difference_focused_permutations_l788_788268

def is_difference_focused (b : Fin 6 → ℕ) : Prop :=
  b 0 + b 1 + b 2 - b 3 - b 4 - b 5 > 0

def permutations (s : Set (Fin 6 → ℕ)) : Set (Fin 6 → ℕ) :=
  { b | ∃ (l : List (Fin 6)), l.nodup ∧ (∀ i, b i = l.nthLe i (by simp [Fin.size])) ∧ (List.perm l.toList [1, 2, 3, 4, 5, 6]) }

theorem number_of_difference_focused_permutations :
  (permutations { b | is_difference_focused b}).count = 1080 := sorry

end number_of_difference_focused_permutations_l788_788268


namespace compare_abc_l788_788801

/- 
Define constants a, b, and c based on given conditions
-/
def a := - ((0.3)^2)
def b := (3:ℝ)⁻¹
def c := (-1/3)^0
-- Prove that a < b < c
theorem compare_abc : a < b ∧ b < c := by
  sorry

end compare_abc_l788_788801


namespace parabola_directrix_l788_788593

theorem parabola_directrix (y x : ℝ) (h : y = x^2) : 4 * y + 1 = 0 :=
sorry

end parabola_directrix_l788_788593


namespace cosine_dihedral_angle_value_l788_788476

noncomputable def vector_a : ℝ × ℝ × ℝ := (0, -1, 3)
noncomputable def vector_b : ℝ × ℝ × ℝ := (2, 2, 4)

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

noncomputable def cosine_dihedral_angle : ℝ :=
  dot_product vector_a vector_b / (magnitude vector_a * magnitude vector_b)

theorem cosine_dihedral_angle_value :
  cosine_dihedral_angle = (real.sqrt 15 / 6) ∨ cosine_dihedral_angle = -(real.sqrt 15 / 6) :=
by sorry

end cosine_dihedral_angle_value_l788_788476


namespace flyers_left_l788_788505

theorem flyers_left (total_flyers : ℕ) (jack_flyers : ℕ) (rose_flyers : ℕ) (h1 : total_flyers = 1236) (h2 : jack_flyers = 120) (h3 : rose_flyers = 320) : (total_flyers - (jack_flyers + rose_flyers) = 796) := 
by
  sorry

end flyers_left_l788_788505


namespace nigel_gave_away_l788_788122

theorem nigel_gave_away :
  ∀ (original : ℕ) (gift_from_mother : ℕ) (final : ℕ) (money_given_away : ℕ),
    original = 45 →
    gift_from_mother = 80 →
    final = 2 * original + 10 →
    final = original - money_given_away + gift_from_mother →
    money_given_away = 25 :=
by
  intros original gift_from_mother final money_given_away
  sorry

end nigel_gave_away_l788_788122


namespace find_a_l788_788429

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x / (1 + a * 2^x)

theorem find_a (a : ℝ) (f : ℝ → ℝ) (h_f_def : ∀ x, f x = 2^x / (1 + a * 2^x))
  (h_symm : ∀ x, f x + f (-x) = 1) : a = 1 :=
sorry

end find_a_l788_788429


namespace exists_odd_card_H_l788_788546

-- Define the context and conditions
variables {M : Type*} [nonempty M]
variable [fintype M]
variable (H : M → set M)

-- Assume the given conditions
axiom cond1 : ∀ x : M, x ∈ H x
axiom cond2 : ∀ x y : M, y ∈ H x ↔ x ∈ H y
axiom odd_card_M : fintype.card M % 2 = 1

-- The main theorem
theorem exists_odd_card_H : 
  ∃ x : M, fintype.card (H x) % 2 = 1 :=
sorry

end exists_odd_card_H_l788_788546


namespace count_integers_le_zero_l788_788271

def P (x : ℤ) : ℤ := (x - 16) * (x - 36) * (x - 64) * (x - 100) *
  (x - 144) * (x - 196) * (x - 256) * (x - 324) * (x - 400) * (x - 484) *
  (x - 576) * (x - 676) * (x - 784) * (x - 900) * (x - 1024) * (x - 1156) *
  (x - 1296) * (x - 1444) * (x - 1600) * (x - 1764) * (x - 1936) * (x - 2116) *
  (x - 2304) * (x - 2500)

theorem count_integers_le_zero (n : ℤ) :
  P(n) ≤ 0 → (finset.range 2765).card - 1 = 2764 :=
by
  sorry

end count_integers_le_zero_l788_788271


namespace projection_matrix_3_4_l788_788331

theorem projection_matrix_3_4 :
  let v := λ α : Type, @vector α 2 := ![3, 4]
  let proj := λ x : vector ℝ 2, (v ℝ ⬝ x) / (v ℝ ⬝ v ℝ) • v ℝ
  proj = (λ x : vector ℝ 2, matrix.mul_vec ![
     ![9 / 25, 12 / 25],
     ![12 / 25, 16 / 25]
  ] x) :=
by sorry

end projection_matrix_3_4_l788_788331


namespace projection_matrix_3_4_l788_788332

theorem projection_matrix_3_4 :
  let v := λ α : Type, @vector α 2 := ![3, 4]
  let proj := λ x : vector ℝ 2, (v ℝ ⬝ x) / (v ℝ ⬝ v ℝ) • v ℝ
  proj = (λ x : vector ℝ 2, matrix.mul_vec ![
     ![9 / 25, 12 / 25],
     ![12 / 25, 16 / 25]
  ] x) :=
by sorry

end projection_matrix_3_4_l788_788332


namespace month_days_l788_788612

theorem month_days (letters_per_day packages_per_day total_mail six_months : ℕ) (h1 : letters_per_day = 60) (h2 : packages_per_day = 20) (h3 : total_mail = 14400) (h4 : six_months = 6) : 
  total_mail / (letters_per_day + packages_per_day) / six_months = 30 :=
by sorry

end month_days_l788_788612


namespace maximum_value_of_f_l788_788608

noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.cos x

theorem maximum_value_of_f :
  ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = (Real.pi / 6) + Real.sqrt 3 ∧ 
  ∀ y ∈ Set.Icc 0 (Real.pi / 2), f y ≤ f (Real.pi / 6) :=
by
  sorry

end maximum_value_of_f_l788_788608


namespace thales_circle_locus_l788_788853

variable (a b : Line)
variable (A M B : Point)
variable [parallel : Parallel a b]
variable [perpendicular : Perpendicular AB a]
variable [M_on_a : On M a]
variable (k : Circle) [diameter : Diameter k A M]
variable (k1 : Circle) [thales_circle : ThalesCircle k1 A B]
variable (P : Point) [P_on_b : On P b]
variable (Q R : Point)

def locus_QR_is_Thales_circle : Prop :=
  locus_of_points (λ M, (Q R).pair) (move_along_line M_on_a) = thalesCircle_centered_at A passing_through B

theorem thales_circle_locus (M : Point) : locus_QR_is_Thales_circle a b A M B k k1 P Q R sorry :=
sorry

end thales_circle_locus_l788_788853


namespace money_last_weeks_l788_788221

-- Define the amounts of money earned and spent per week
def money_mowing : ℕ := 5
def money_weed_eating : ℕ := 58
def weekly_spending : ℕ := 7

-- Define the total money earned
def total_money : ℕ := money_mowing + money_weed_eating

-- Define the number of weeks the money will last
def weeks_last (total : ℕ) (weekly : ℕ) : ℕ := total / weekly

-- Theorem stating the number of weeks the money will last
theorem money_last_weeks : weeks_last total_money weekly_spending = 9 := by
  sorry

end money_last_weeks_l788_788221


namespace sequence_a_100_l788_788613

theorem sequence_a_100 (a : ℕ → ℤ) (h₁ : a 1 = 3) (h₂ : ∀ n : ℕ, a (n + 1) = a n - 2) : a 100 = -195 :=
by
  sorry

end sequence_a_100_l788_788613


namespace probability_S4_gt_0_l788_788655

noncomputable def probability_of_heads : ℝ := 1 / 2

def a_n (n : ℕ) (coin_toss : ℕ → bool) : ℤ := 
  if coin_toss n then 1 else -1

def S_n (n : ℕ) (coin_toss : ℕ → bool) : ℤ := 
  ∑ i in finset.range n, a_n i coin_toss

theorem probability_S4_gt_0 
  (h_fair : ∀ n, (probability_of_heads = 1 / 2)) :
  (∑ k in finset.filter (λ k, k>0) (finset.range 5), 
    ((nat.choose 4 k) : ℝ)
    * probability_of_heads ^ k 
    * (1 - probability_of_heads) ^ (4 - k)) = 5 / 16 := 
sorry

end probability_S4_gt_0_l788_788655


namespace peter_score_l788_788220

theorem peter_score (e m h : ℕ) (total_problems points : ℕ)
  (easy_solved medium_solved hard_solved: ℕ → ℕ := λ x, x) : 
  e + m + h = total_problems ∧ 
  2 * e + 3 * m + 5 * h = points ∧ 
  total_problems = 25 ∧ 
  points = 84 ∧ 
  easy_solved e = e ∧ 
  medium_solved m = m / 2 ∧ 
  hard_solved h = h / 3 → 
  easy_solved e * 2 + medium_solved m * 3 + hard_solved h * 5 = 40 := 
by 
  sorry

end peter_score_l788_788220


namespace random_event_is_option_D_l788_788683

-- Definitions based on conditions
def rains_without_clouds : Prop := false
def like_charges_repel : Prop := true
def seeds_germinate_without_moisture : Prop := false
def draw_card_get_1 : Prop := true

-- Proof statement
theorem random_event_is_option_D : 
  (¬ rains_without_clouds ∧ like_charges_repel ∧ ¬ seeds_germinate_without_moisture ∧ draw_card_get_1) →
  (draw_card_get_1 = true) :=
by sorry

end random_event_is_option_D_l788_788683


namespace checker_rectangle_l788_788686

-- Define the problem setup in Lean
def checker := bool  -- Checkers can be represented as booleans: false for white, true for black.

def in_rectangle (grid : list (list checker)) : Prop :=
  ∃ r1 r2 c1 c2, r1 < r2 ∧ c1 < c2 ∧ 
  (grid[r1][c1] = grid[r1][c2] ∧ grid[r1][c1] = grid[r2][c1] ∧ grid[r1][c1] = grid[r2][c2])

-- Formal statement of the problem
theorem checker_rectangle (grid : list (list checker)) (h_length : grid.length = 3) (h_width : ∀ row, row ∈ grid → row.length = 7) :
  in_rectangle grid :=
by
  sorry

end checker_rectangle_l788_788686


namespace divide_660_stones_into_30_piles_l788_788978

theorem divide_660_stones_into_30_piles :
  ∃ (heaps : Fin 30 → ℕ),
    (∑ i, heaps i = 660) ∧
    ∀ i j, heaps i ≤ 2 * heaps j :=
sorry

end divide_660_stones_into_30_piles_l788_788978


namespace flyers_left_l788_788512

theorem flyers_left (total_flyers : ℕ) (jack_flyers : ℕ) (rose_flyers : ℕ) :
  total_flyers = 1236 → jack_flyers = 120 → rose_flyers = 320 → total_flyers - (jack_flyers + rose_flyers) = 796 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact eq.refl _

end flyers_left_l788_788512


namespace area_of_triangle_l788_788887

noncomputable def complex_area (z : ℂ) : ℝ :=
  (1 / 2) * complex.abs (z * (z - 1))

theorem area_of_triangle (z : ℂ) (hz : complex.abs z = 1) : complex_area z = 1 :=
begin
  sorry
end

end area_of_triangle_l788_788887


namespace OH_squared_l788_788925

/-- 
Given:
  O is the circumcenter of triangle ABC.
  H is the orthocenter of triangle ABC.
  a, b, and c are the side lengths of triangle ABC.
  R is the circumradius of triangle ABC.
  R = 5.
  a^2 + b^2 + c^2 = 50.

Prove:
  OH^2 = 175.
-/
theorem OH_squared (a b c R : ℝ) (hR : R = 5) (habc : a^2 + b^2 + c^2 = 50) :
  let OH_squared := 9 * R^2 - (a^2 + b^2 + c^2) in
  OH_squared = 175 :=
by
  sorry

end OH_squared_l788_788925


namespace slower_train_speed_l788_788190

theorem slower_train_speed 
  (length_train1 length_train2 : ℕ) 
  (time_crossing : ℝ) 
  (speed_faster_train_km_hr : ℝ)
  (length_train1 = 250)
  (length_train2 = 500)
  (time_crossing = 26.99784017278618)
  (speed_faster_train_km_hr = 60) :
  (slower_train_speed_km_hr : ℝ) : slower_train_speed_km_hr ≈ 40.017 := 
by
  sorry

end slower_train_speed_l788_788190


namespace expand_product_l788_788290

theorem expand_product (x : ℤ) : 
  (3 * x + 4) * (2 * x - 6) = 6 * x^2 - 10 * x - 24 :=
by
  sorry

end expand_product_l788_788290


namespace triangle_area_iso_l788_788189

open Real

noncomputable def area_triangle (A B C : Point) : ℚ :=
  1 / 2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

structure Point :=
  (x y : ℚ)

def line1 (P : Point) : ℚ := 3 / 4 * P.x + 3 / 4
def line2 (P : Point) : ℚ := 1 / 3 * P.x + 2
def line3 (P : Point) : Prop := P.x + P.y = 12

def point_A : Point := ⟨3, 3⟩
def point_B : Point := ⟨7.5, 4.5⟩ -- These are calculated satisfactions of the lines
def point_C : Point := ⟨6.42857, 5.57143⟩

theorem triangle_area_iso :
  area_triangle point_A point_B point_C = 3.214285 :=
sorry

end triangle_area_iso_l788_788189


namespace oh_squared_l788_788923

theorem oh_squared (O H : ℝ) (a b c R : ℝ) (h1 : R = 5) (h2 : a^2 + b^2 + c^2 = 50) :
  let OH := H - O in
  OH ^ 2 = 175 :=
by
  sorry

end oh_squared_l788_788923


namespace ratio_of_inscribed_and_circumscribed_spheres_l788_788167

noncomputable def radius_ratio (α β γ : ℝ) : ℝ :=
  (3 - Real.cos α - Real.cos β - Real.cos γ) / 
  (3 + Real.cos α + Real.cos β + Real.cos γ)

theorem ratio_of_inscribed_and_circumscribed_spheres
  (α β γ : ℝ)
  (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ) 
  (h_sum : α + β + γ < real.pi) : 
  radius_ratio α β γ =
  (3 - Real.cos α - Real.cos β - Real.cos γ) / 
  (3 + Real.cos α + Real.cos β + Real.cos γ) := 
  sorry

end ratio_of_inscribed_and_circumscribed_spheres_l788_788167


namespace high_jump_sneakers_cost_l788_788901

def lawn_earnings (lawns mowed : ℕ) (pay_per_lawn : ℝ) : ℝ :=
  lawns_mowed * pay_per_lawn

def figure_earnings (figures_sold : ℕ) (pay_per_figure : ℝ) : ℝ :=
  figures_sold * pay_per_figure

def job_earnings (hours_worked : ℕ) (pay_per_hour : ℝ) : ℝ :=
  hours_worked * pay_per_hour

def total_earnings (lawn_earnings : ℝ) (figure_earnings : ℝ) (job_earnings : ℝ) : ℝ :=
  lawn_earnings + figure_earnings + job_earnings

theorem high_jump_sneakers_cost :
  let lawns_mowed := 3
  let pay_per_lawn := 8
  let figures_sold := 2
  let pay_per_figure := 9
  let hours_worked := 10
  let pay_per_hour := 5
  let cost_of_sneakers := 92
  total_earnings (lawn_earnings lawns_mowed pay_per_lawn) (figure_earnings figures_sold pay_per_figure) (job_earnings hours_worked pay_per_hour) = cost_of_sneakers :=
by
  sorry

end high_jump_sneakers_cost_l788_788901


namespace power_of_fraction_l788_788760

theorem power_of_fraction :
  (3 / 4) ^ 5 = 243 / 1024 :=
by sorry

end power_of_fraction_l788_788760


namespace composite_expression_l788_788818

theorem composite_expression (a b c d m n : ℕ) (ha : a > b) (hb : b > c) (hc : c > d) (pos: 0 < d) 
    (hdiv : a + b - c + d ∣ a * c + b * d)
    (hposm : 0 < m) (odd_n : n % 2 = 1) :
    ¬(nat.prime (a^n * b^m + c^m * d^n)) :=
sorry

end composite_expression_l788_788818


namespace product_of_numbers_l788_788178

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 26) (h2 : x - y = 8) : x * y = 153 :=
sorry

end product_of_numbers_l788_788178


namespace projection_onto_vector_l788_788323

noncomputable def projection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  ![\[9 / 25, 12 / 25\], \[12 / 25, 16 / 25\]]

theorem projection_onto_vector:
    ∀ (x y : ℚ), (Matrix.mul_vec projection_matrix ![\x, \y]) = ![(9 * x + 12 * y) / 25, (12 * x + 16 * y) / 25] := by
  sorry

end projection_onto_vector_l788_788323


namespace crayons_given_proof_l788_788126

def initial_crayons : ℕ := 110
def total_lost_crayons : ℕ := 412
def more_lost_than_given : ℕ := 322

def G : ℕ := 45 -- This is the given correct answer to prove.

theorem crayons_given_proof :
  ∃ G : ℕ, (G + (G + more_lost_than_given)) = total_lost_crayons ∧ G = 45 :=
by
  sorry

end crayons_given_proof_l788_788126


namespace positive_area_triangles_count_l788_788857

/-- 
  The total number of triangles with positive area, whose vertices are points 
  in the xy-plane with integer coordinates satisfying 1 ≤ x ≤ 5 and 1 ≤ y ≤ 3, is 416.
-/
theorem positive_area_triangles_count : 
  (∃ (points : List (ℤ × ℤ)), 
   points.length = 15 ∧ 
   (∀ p, p ∈ points → 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 3) ∧ 
   triangles_with_positive_area points = 416) := 
sorry

end positive_area_triangles_count_l788_788857


namespace average_of_eight_digits_l788_788181

theorem average_of_eight_digits 
  (a b c d e f g h : ℝ)
  (h1 : (a + b + c + d + e) / 5 = 12)
  (h2 : (f + g + h) / 3 ≈ 33.333333333333336) :
  (a + b + c + d + e + f + g + h) / 8 = 20 :=
sorry

end average_of_eight_digits_l788_788181


namespace concurrency_of_lines_l788_788099

noncomputable def triangle (A B C : Type) := 
{A : A, B : B, C : C}

noncomputable def points_on_sides (ABC : triangle ℝ) : Prop :=
∃ A1 B1 C1 : ℝ, (A1 ∈ line (BC (ABC.A ABC.B))) ∧ 
                 (B1 ∈ line (CA (ABC.B ABC.C))) ∧ 
                 (C1 ∈ line (AB (ABC.A ABC.B))) ∧ 
                 (concurrent [line (ABC.A A1), line (ABC.B B1), line (ABC.C C1)])

noncomputable def circle_intersections (ABC : triangle ℝ) (A1 B1 C1 : ℝ) : Prop :=
∃ A2 B2 C2 : ℝ, 
                 (A2 ∈ circle_through [A1, B1, C1]) ∧ 
                 (B2 ∈ circle_through [A1, B1, C1]) ∧ 
                 (C2 ∈ circle_through [A1, B1, C1]) ∧ 
                 (A2 ∈ line (BC (ABC.A ABC.B))) ∧ 
                 (B2 ∈ line (CA (ABC.B ABC.C))) ∧ 
                 (C2 ∈ line (AB (ABC.A ABC.B)))

theorem concurrency_of_lines (ABC : triangle ℝ) 
  (h1 : points_on_sides ABC)
  (h2 : ∃ A1 B1 C1 : ℝ, A1 ∈ line (BC (ABC.A ABC.B)) ∧ 
                        B1 ∈ line (CA (ABC.B ABC.C)) ∧ 
                        C1 ∈ line (AB (ABC.A ABC.B)) ∧ 
                       (circle_intersections ABC A1 B1 C1)):
  ∃ A2 B2 C2 : ℝ, concurrent [line (ABC.A A2), line (ABC.B B2), line (ABC.C C2)] :=
sorry

end concurrency_of_lines_l788_788099


namespace flyers_left_l788_788504

theorem flyers_left (total_flyers : ℕ) (jack_flyers : ℕ) (rose_flyers : ℕ) (h1 : total_flyers = 1236) (h2 : jack_flyers = 120) (h3 : rose_flyers = 320) : (total_flyers - (jack_flyers + rose_flyers) = 796) := 
by
  sorry

end flyers_left_l788_788504


namespace marissa_tied_boxes_l788_788116

def Total_ribbon : ℝ := 4.5
def Leftover_ribbon : ℝ := 1
def Ribbon_per_box : ℝ := 0.7

theorem marissa_tied_boxes : (Total_ribbon - Leftover_ribbon) / Ribbon_per_box = 5 := by
  sorry

end marissa_tied_boxes_l788_788116


namespace solve_equation_l788_788373

noncomputable def smallest_solution : Rat :=
  (8 - Real.sqrt 145) / 3

theorem solve_equation : 
  ∃ x : ℝ, (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ x = smallest_solution := sorry

end solve_equation_l788_788373


namespace divide_stones_into_heaps_l788_788949

-- Definitions based on the identified conditions
variable (Heaps : List ℕ) -- list of heap sizes
variable (n_stones : ℕ) -- total number of stones
variable (n_heaps : ℕ) -- number of heaps

-- Conditions
axiom total_stones : n_stones = 660
axiom total_heaps : n_heaps = 30
axiom heap_size_condition : ∀ (h₁ h₂ : ℕ), h₁ ∈ Heaps → h₂ ∈ Heaps → h₁ ≤ 2 * h₂

-- Theorem statement
theorem divide_stones_into_heaps : 
  ∃ Heaps, Heaps.length = n_heaps ∧ Heaps.sum = n_stones ∧
  ∀ (h₁ h₂ : ℕ), h₁ ∈ Heaps → h₂ ∈ Heaps → h₁ ≤ 2 * h₂ :=
by
  sorry

end divide_stones_into_heaps_l788_788949


namespace difference_smallest_integers_mod_1_13_l788_788859

noncomputable def lcm_1_to_13 : ℕ :=
  Nat.lcm 1 (Nat.lcm (2) (Nat.lcm (3) (Nat.lcm (4) (Nat.lcm (5) (Nat.lcm (6) (Nat.lcm (7) (Nat.lcm (8) (Nat.lcm (9) (Nat.lcm (10) (Nat.lcm (11) (Nat.lcm (12) (Nat.lcm (13) (1)))))))))))))))

theorem difference_smallest_integers_mod_1_13 : 
  ∃ n1 n2, (∀ k ∈ {1, 2, ..., 13}, (n1 > 1 ∧ n2 > 1) ∧ (n1 % k = 1) ∧ (n2 % k = 1) ∧ n1 < n2 ∧ n2 = n1 + lcm_1_to_13) → 
  n2 - n1 = 720720 :=
by
  sorry

end difference_smallest_integers_mod_1_13_l788_788859


namespace elizabeth_revenue_per_investment_l788_788554

theorem elizabeth_revenue_per_investment :
  ∀ (revenue_per_investment_banks revenue_difference total_investments_banks total_investments_elizabeth : ℕ),
    revenue_per_investment_banks = 500 →
    total_investments_banks = 8 →
    total_investments_elizabeth = 5 →
    revenue_difference = 500 →
    ((revenue_per_investment_banks * total_investments_banks) + revenue_difference) / total_investments_elizabeth = 900 :=
by
  intros revenue_per_investment_banks revenue_difference total_investments_banks total_investments_elizabeth
  intros h_banks_revenue h_banks_investments h_elizabeth_investments h_revenue_difference
  sorry

end elizabeth_revenue_per_investment_l788_788554


namespace sqrt17_minus_5_l788_788602

def greatest_integer (x : ℝ) : ℤ := ⌊x⌋

theorem sqrt17_minus_5 :
  greatest_integer (Real.sqrt 17 - 5) = -1 :=
by
  -- Definitions based on provided conditions
  have h1 : 4 < Real.sqrt 17 := by sorry
  have h2 : Real.sqrt 17 < 5 := by sorry
  -- Combining these, we get the intermediate condition
  have h3 : -1 < Real.sqrt 17 - 5 := by sorry
  have h4 : Real.sqrt 17 - 5 < 0 := by sorry
  -- Concluding the statement
  -- which leads us to:
  have h5 : greatest_integer (Real.sqrt 17 - 5) = -1 := by sorry
  exact h5

end sqrt17_minus_5_l788_788602


namespace part1_part2_l788_788842

noncomputable def f (a x : ℝ) : ℝ :=
  a * Real.sin x - 1/2 * Real.cos (2 * x) + a - 3/a + 1/2

theorem part1 (a : ℝ) (h₀ : a ≠ 0) :
  (∀ x : ℝ, f a x ≤ 0) → a ∈ Set.Icc 0 1 := sorry

theorem part2 (a : ℝ) (h₀ : a ≠ 0) (h₁ : a ≥ 2) :
  (∃ x : ℝ, f a x ≤ 0) → a ∈ Set.Icc 2 3 := sorry

end part1_part2_l788_788842


namespace sum_of_digits_second_smallest_multiple_l788_788086

theorem sum_of_digits_second_smallest_multiple :
  sum_of_digits (2 * nat.lcm (list.range 9).tail) = 15 :=
by
  sorry

end sum_of_digits_second_smallest_multiple_l788_788086


namespace area_of_convex_quadrilateral_l788_788881

-- Define the conditions of the problem in a way usable for the proof
variables {A B C D : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables (AB BC CD DA : ℝ)
variables (angle_BCD : ℝ)
variables (a b c : ℕ)

-- State the given conditions
def problem_conditions : Prop :=
  AB = 12 ∧ BC = 6 ∧ CD = 13 ∧ DA = 13 ∧ angle_BCD = real.pi / 2

-- State the question in terms of the proof problem
theorem area_of_convex_quadrilateral (h : problem_conditions AB BC CD DA angle_BCD) :
  ∃ a b c, a + b + c = 690 := by
sorry

end area_of_convex_quadrilateral_l788_788881


namespace product_of_decimal_numbers_l788_788691

theorem product_of_decimal_numbers 
  (h : 213 * 16 = 3408) : 
  1.6 * 21.3 = 34.08 :=
by
  sorry

end product_of_decimal_numbers_l788_788691


namespace least_n_factorial_l788_788670

theorem least_n_factorial (n : ℕ) : (∃ n, n ≥ 1 ∧ ∀ m, m < n → ¬(9450 ∣ factorial m) ) ∧ (9450 ∣ factorial 10) :=
by
  sorry

end least_n_factorial_l788_788670


namespace projection_onto_3_4_matrix_l788_788339

def projection_matrix := λ (u : ℝ) (v : ℝ), (3 * u + 4 * v) / 25

theorem projection_onto_3_4_matrix :
  ∀ (x y : ℝ),
  (λ (u v : ℝ), (3 * x + 4 * y) / 25) = (λ (u v : ℝ), (\(u * 9 / 25) + (v * 12 / 25), (u * 12 / 25) + (v * 16 / 25))) :=
by
  sorry

end projection_onto_3_4_matrix_l788_788339


namespace smallest_solution_to_equation_l788_788367

noncomputable def smallest_solution := (11 - Real.sqrt 445) / 6

theorem smallest_solution_to_equation:
  ∃ x : ℝ, (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ (x = smallest_solution) :=
sorry

end smallest_solution_to_equation_l788_788367


namespace transformation_constants_l788_788165

noncomputable def f : ℝ → ℝ
| x := if h : -3 ≤ x ∧ x < 0 then -2 - x
       else if h : 0 ≤ x ∧ x < 2 then sqrt (4 - (x - 2) ^ 2) - 2
       else if h : 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
       else 0 -- default value outside the defined intervals

def g (a b c : ℝ) (x : ℝ) : ℝ := a * f (b * x) + c

theorem transformation_constants :
  ∃ a b c : ℝ, a = 1 ∧ b = 1 / 3 ∧ c = -3 ∧ ∀ x : ℝ, g a b c x = f (x / 3) - 3 :=
begin
  use [1, 1 / 3, -3],
  split, refl,
  split, refl,
  split, refl,
  intro x,
  rw [g, f, f],
  sorry
end

end transformation_constants_l788_788165


namespace camp_boys_count_l788_788481

/-- The ratio of boys to girls and total number of individuals in the camp including teachers
is given, we prove the number of boys is 26. -/
theorem camp_boys_count 
  (b g t : ℕ) -- b = number of boys, g = number of girls, t = number of teachers
  (h1 : b = 3 * (t - 5))  -- boys count related to some integer "t" minus teachers
  (h2 : g = 4 * (t - 5))  -- girls count related to some integer "t" minus teachers
  (total_individuals : t = 65) : 
  b = 26 :=
by
  have h : 3 * (t - 5) + 4 * (t - 5) + 5 = 65 := sorry
  sorry

end camp_boys_count_l788_788481


namespace trigonometric_order_l788_788545

-- a = sin(-1)
def a : ℝ := Real.sin (-1)

-- b = cos(-1)
def b : ℝ := Real.cos (-1)

-- c = tan(-1)
def c : ℝ := Real.tan (-1)

-- Prove c < a < b
theorem trigonometric_order : c < a ∧ a < b := 
by sorry

end trigonometric_order_l788_788545


namespace part1_part2_l788_788442

-- Part 1: proving intersection of sets A and B
theorem part1 (m : ℝ) (h : m = -1) : 
  let A := {x : ℝ | 1 < x ∧ x < 3 },
      B := {x : ℝ | -2 < x ∧ x < 2 } in 
  A ∩ B = { x : ℝ | 1 < x ∧ x < 2 } :=
by sorry

-- Part 2: proving the range of m such that A ⊆ B
theorem part2 (m : ℝ) :
  let A := {x : ℝ | 1 < x ∧ x < 3 },
      B := {x : ℝ | 2 * m < x ∧ x < 1 - m } in  
  (A ⊆ B) → m ≤ -2 :=
by sorry

end part1_part2_l788_788442


namespace pile_division_660_stones_l788_788974

theorem pile_division_660_stones (n : ℕ) (heaps : List ℕ) :
  n = 660 ∧ heaps.length = 30 ∧ ∀ x ∈ heaps, ∀ y ∈ heaps, (x ≤ 2 * y ∧ y ≤ 2 * x) →
  ∃ heaps : List ℕ, n = 660 ∧ heaps.length = 30 ∧ (∀ x y ∈ heaps, x ≤ 2 * y ∧ y ≤ 2 * x) :=
by
  sorry

end pile_division_660_stones_l788_788974


namespace five_star_three_l788_788273

def star (a b : ℤ) : ℤ := a^2 - 2 * a * b + b^2

theorem five_star_three : star 5 3 = 4 := by
  sorry

end five_star_three_l788_788273


namespace inequality_sqrt_sum_ge_two_l788_788459

theorem inequality_sqrt_sum_ge_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (abc_eq_one : a * b * c = 1) : 
    1 / Real.sqrt (b + 1 / a + 1 / 2) + 
    1 / Real.sqrt (c + 1 / b + 1 / 2) + 
    1 / Real.sqrt (a + 1 / c + 1 / 2) >= Real.sqrt 2 :=
begin
  sorry
end

end inequality_sqrt_sum_ge_two_l788_788459


namespace max_students_before_new_year_l788_788703

theorem max_students_before_new_year (N M k l : ℕ) (h1 : 100 * M = k * N) (h2 : 100 * (M + 1) = l * (N + 3)) (h3 : 3 * l < 300) :
      N ≤ 197 := by
  sorry

end max_students_before_new_year_l788_788703


namespace sum_of_squares_l788_788465

theorem sum_of_squares :
  ∃ p q r s t u : ℤ, (∀ x : ℤ, 729 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) ∧ 
    (p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 8210) :=
sorry

end sum_of_squares_l788_788465


namespace area_between_tangent_circles_l788_788656

theorem area_between_tangent_circles (r : ℝ) (h_r : r > 0) :
  let area_trapezoid := 4 * r^2 * Real.sqrt 3
  let area_sector1 := π * r^2 / 3
  let area_sector2 := 3 * π * r^2 / 2
  area_trapezoid - (area_sector1 + area_sector2) = r^2 * (24 * Real.sqrt 3 - 11 * π) / 6 := by
  sorry

end area_between_tangent_circles_l788_788656


namespace divide_660_stones_into_30_piles_l788_788981

theorem divide_660_stones_into_30_piles :
  ∃ (heaps : Fin 30 → ℕ),
    (∑ i, heaps i = 660) ∧
    ∀ i j, heaps i ≤ 2 * heaps j :=
sorry

end divide_660_stones_into_30_piles_l788_788981


namespace length_of_train_is_135_l788_788744

noncomputable def length_of_train (v : ℝ) (t : ℝ) : ℝ :=
  ((v * 1000) / 3600) * t

theorem length_of_train_is_135 :
  length_of_train 140 3.4711508793582233 = 135 :=
sorry

end length_of_train_is_135_l788_788744


namespace popsicle_count_l788_788720

theorem popsicle_count (r : ℝ) (n : ℕ) (h1 : ∀ k, k ≥ 1 → melting_rate (k + 1) = 2 * melting_rate k)
  (h2 : melting_rate n = 32 * melting_rate 1) : n = 6 :=
by
  sorry

end popsicle_count_l788_788720


namespace least_length_XZ_l788_788692

open Real EuclideanGeometry

-- Define the conditions of the triangle PQR with given lengths.
noncomputable def PQR (P Q R : Point) : Prop :=
  ∠Q = π / 2 ∧ dist P Q = 3 ∧ dist Q R = 8

-- Define X as a variable point on PQ.
def on_PQ (P Q X : Point) : Prop :=
  X ∈ line_through P Q 

-- Define Y such that XY is parallel to QR.
def Y_parallel_QR (X Y Q R : Point) : Prop :=
  X ≠ Y ∧ parallel (line_through X Y) (line_through Q R)

-- Define Z such that YZ is parallel to PQ.
def Z_parallel_PQ (Y Z P Q : Point) : Prop :=
  Y ≠ Z ∧ parallel (line_through Y Z) (line_through P Q)

-- The least possible length of XZ
theorem least_length_XZ (P Q R X Y Z : Point) 
  (hPQR : PQR P Q R) (hX : on_PQ P Q X) (hY : Y_parallel_QR X Y Q R) (hZ : Z_parallel_PQ Y Z P Q) :
  0 ≤ dist X Z ∧ ∀ X' ∈ line_through P Q, (dist X' Z < dist X Z → X' = P) → dist X Z = 0 := 
sorry

end least_length_XZ_l788_788692


namespace expression_evaluation_l788_788762

theorem expression_evaluation : 3 * 257 + 4 * 257 + 2 * 257 + 258 = 2571 := by
  sorry

end expression_evaluation_l788_788762


namespace projection_matrix_is_correct_l788_788353

noncomputable def projectionMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  let v : Fin 2 → ℝ := ![3, 4]
  (1 / (v 0 ^ 2 + v 1 ^ 2)) • (λ i j, v i * v j)

theorem projection_matrix_is_correct :
  projectionMatrix = ![![9/25, 12/25], ![12/25, 16/25]] :=
by
  sorry

end projection_matrix_is_correct_l788_788353


namespace find_a_range_l788_788819

def prop_p (a : ℝ) : Prop := ∀ m ∈ set.Icc (-1 : ℝ) 1, a^2 - 5 * a - 3 ≥ real.sqrt (m^2 + 8)

def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + a * x + 2 < 0

theorem find_a_range (a : ℝ) (hp : prop_p a) (hq_false : ¬ prop_q a) : -2 * real.sqrt 2 ≤ a ∧ a ≤ -1 :=
sorry

end find_a_range_l788_788819


namespace find_c_for_min_value_l788_788384

theorem find_c_for_min_value :
  ∀ (c : ℝ), (∃ (x : ℝ), -3 ≤ x ∧ x ≤ 2 ∧ y = -x^2 - 2x + c ∧ y = -5) → c = 3 :=
by
  assume c
  assume h
  sorry

end find_c_for_min_value_l788_788384


namespace find_4a_add_c_find_2a_sub_2b_sub_c_l788_788005

variables {R : Type*} [CommRing R]

theorem find_4a_add_c (a b c : ℝ) (h : ∀ x : ℝ, (x^3 + a * x^2 + b * x + c) = (x^2 + 3 * x - 4) * (x + (a - 3) - b + 4 - c)) :
  4 * a + c = 12 :=
sorry

theorem find_2a_sub_2b_sub_c (a b c : ℝ) (h : ∀ x : ℝ, (x^3 + a * x^2 + b * x + c) = (x^2 + 3 * x - 4) * (x + (a - 3) - b + 4 - c)) :
  2 * a - 2 * b - c = 14 :=
sorry

end find_4a_add_c_find_2a_sub_2b_sub_c_l788_788005


namespace quotient_larger_than_dividend_l788_788254

noncomputable def division_results :=
  [ (7.9 / 1.6, 7.9),
    (23.7 / 1, 23.7),
    (5.4 / 0.8, 5.4),
    (5.5 / 1.3, 5.5) ]

theorem quotient_larger_than_dividend :
  (5.4 / 0.8) > 5.4 :=
by
  unfold division_results
  sorry

end quotient_larger_than_dividend_l788_788254


namespace dot_product_of_BC_and_CA_l788_788473

variables (a b : ℝ)
variables (C : ℝ)
variables (BC CA : E)

theorem dot_product_of_BC_and_CA (h1 : a = 5) (h2 : b = 8) (h3 : C = 60 * (Real.pi / 180)) :
  (|BC| = a) → (|CA| = b) → (BC • CA = a * b * Real.cos (Real.pi - C / 180)) :=
by
  intros
  have angle_eq : Real.pi - C = Real.pi - 60 * (Real.pi / 180) :=
    by { sorry }
  exact a * b * - (1 / 2) = -20
  sorry

end dot_product_of_BC_and_CA_l788_788473


namespace targets_breaking_order_count_l788_788056

theorem targets_breaking_order_count : let n : ℕ := 9 in
  let m : ℕ := 3 in
  (n.factorial / (m.factorial * m.factorial * m.factorial)) = 1680 := by
  sorry

end targets_breaking_order_count_l788_788056


namespace joshua_borrowed_cents_l788_788520

-- Definitions based on conditions
def cost_pen_cents : ℕ := 600
def joshua_cents : ℕ := 500
def additional_cents_needed : ℕ := 32

-- Mathematically equivalent proof problem statement
theorem joshua_borrowed_cents : ∀ (borrowed_cents : ℕ), 
  borrowed_cents = cost_pen_cents + additional_cents_needed - joshua_cents → 
  borrowed_cents = 132 :=
by
  intro borrowed_cents
  assume h : borrowed_cents = cost_pen_cents + additional_cents_needed - joshua_cents
  sorry

end joshua_borrowed_cents_l788_788520


namespace max_students_before_new_year_l788_788701

theorem max_students_before_new_year (N M k l : ℕ) (h1 : 100 * M = k * N) (h2 : 100 * (M + 1) = l * (N + 3)) (h3 : 3 * l < 300) :
      N ≤ 197 := by
  sorry

end max_students_before_new_year_l788_788701


namespace attendance_calculation_l788_788452

theorem attendance_calculation (total_students : ℕ) (attendance_rate : ℚ)
  (h1 : total_students = 120)
  (h2 : attendance_rate = 0.95) :
  total_students * attendance_rate = 114 := 
  sorry

end attendance_calculation_l788_788452


namespace chess_game_probabilities_l788_788129

theorem chess_game_probabilities :
  let p_draw := 1 / 2
  let p_b_win := 1 / 3
  let p_sum := 1
  let p_a_win := p_sum - p_draw - p_b_win
  let p_a_not_lose := p_draw + p_a_win
  let p_b_not_lose := p_draw + p_b_win
  A := p_a_win = 1 / 6
  B := p_a_not_lose = 1 / 2
  C := p_a_win = 2 / 3
  D := p_b_not_lose = 1 / 2
  in ¬ (p_a_win = 1 / 6 ∧ p_a_not_lose ≠ 1 / 2 ∧ p_a_win ≠ 2 / 3 ∧ p_b_not_lose ≠ 1 / 2)
:=
sorry

end chess_game_probabilities_l788_788129


namespace projection_onto_vector_is_expected_l788_788302

def projection_matrix (u: ℝ × ℝ) : matrix (fin 2) (fin 2) ℝ :=
  let ⟨x, y⟩ := u in 
  (1 / (x^2 + y^2)) • (matrix.col_vec u ⬝ (matrix.transpose (matrix.row_vec u)))

def expected_matrix : matrix (fin 2) (fin 2) ℝ :=
  ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]]

theorem projection_onto_vector_is_expected :
  projection_matrix (3, 4) = expected_matrix := by
  sorry

end projection_onto_vector_is_expected_l788_788302


namespace projection_matrix_is_correct_l788_788356

noncomputable def projectionMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  let v : Fin 2 → ℝ := ![3, 4]
  (1 / (v 0 ^ 2 + v 1 ^ 2)) • (λ i j, v i * v j)

theorem projection_matrix_is_correct :
  projectionMatrix = ![![9/25, 12/25], ![12/25, 16/25]] :=
by
  sorry

end projection_matrix_is_correct_l788_788356


namespace min_trucks_required_to_transport_l788_788648

def A : Type := { weights := 5, count := 4 }
def B : Type := { weights := 4, count := 6 }
def C : Type := { weights := 3, count := 11 }
def D : Type := { weights := 1, count := 7 }
def truck_capacity : ℕ := 6

theorem min_trucks_required_to_transport :
  let total_weight := (A.weights * A.count) + (B.weights * B.count) + (C.weights * C.count) + (D.weights * D.count) in
  total_weight = 84 →
  ∃ trucks : ℕ, trucks = 16 ∧ trucks * truck_capacity ≥ total_weight := 
by
  sorry

end min_trucks_required_to_transport_l788_788648


namespace max_dist_from_curve_to_point_l788_788480

-- Defining the polar equation and the specific point
def polar_curve (ρ θ : ℝ) : Prop := ρ = 2 * cos θ

def point_in_polar_coords : ℝ × ℝ := (1, π)

-- The conversion of the polar curve to rectangular coordinates
def curve_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- The maximum distance calculation
def max_distance (x y : ℝ) : ℝ := 
  let dist_to_point := 2 -- distance from (1, 0) to (-1, 0)
  let radius := 1
  radius + dist_to_point

-- The main statement: prove the maximum distance is 3
theorem max_dist_from_curve_to_point : ∀ x y, curve_eq x y → max_distance x y = 3 := 
by sorry

end max_dist_from_curve_to_point_l788_788480


namespace expected_num_games_ends_l788_788478

-- Definitions for the probabilities
def prob_winning_A (n : ℕ) : ℚ := if (n % 2 = 1) then 3/5 else 2/5
def prob_winning_B (n : ℕ) : ℚ := if (n % 2 = 0) then 3/5 else 2/5

-- Condition for competition to end
def ends_condition (win_A win_B : ℕ) : Prop := (win_A = win_B + 2) ∨ (win_B = win_A + 2)

-- Mathematics proof problem: expected number of games when match ends
theorem expected_num_games_ends : expected_value (games_till_end prob_winning_A prob_winning_B ends_condition) = 25/6 := sorry

end expected_num_games_ends_l788_788478


namespace students_before_new_year_le_197_l788_788705

variable (N M k ℓ : ℕ)

-- Conditions
axiom condition_1 : M = (k * N) / 100
axiom condition_2 : 100 * M = k * N
axiom condition_3 : 100 * (M + 1) = ℓ * (N + 3)
axiom condition_4 : ℓ < 100

-- The theorem to prove
theorem students_before_new_year_le_197 :
  N ≤ 197 :=
by
  sorry

end students_before_new_year_le_197_l788_788705


namespace problem_l788_788439

noncomputable def p : Prop :=
  ∀ x : ℝ, (0 < x) → Real.exp x > 1 + x

def q (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-x) + 2 = -(f x + 2)) → ∀ x : ℝ, f (-x) = f x - 4

theorem problem (f : ℝ → ℝ) : p ∨ q f :=
  sorry

end problem_l788_788439


namespace sandy_total_marks_l788_788573

theorem sandy_total_marks : 
  let c := 22 in
  let i := 30 - 22 in
  (3 * c - 2 * i) = 50 := 
by {
  let c := 22
  let i := 30 - 22
  calc (3 * c - 2 * i)
      = (3 * 22 - 2 * (30 - 22)) : by rfl
    ... = (66 - 2 * 8)           : by rfl
    ... = (66 - 16)              : by rfl
    ... = 50                     : by rfl
}

end sandy_total_marks_l788_788573


namespace projection_matrix_correct_l788_788312

variables {R : Type*} [field R] [decidable_eq R]
variables (x y : R)
def vector_v : matrix (fin 2) (fin 1) R := ![![3], ![4]]
def vector_u : matrix (fin 2) (fin 1) R := ![![x], ![y]]
def projection_matrix : matrix (fin 2) (fin 2) R := ![![9/25, 12/25], ![12/25, 16/25]]

theorem projection_matrix_correct :
  (projection_matrix R) ⬝ (vector_u x y) = (25 : R)⁻¹ • (transpose (vector_v 3 4) ⬝ (vector_u x y)) ⬝ (vector_v 3 4) := 
sorry

end projection_matrix_correct_l788_788312


namespace minimum_a_condition_l788_788003

theorem minimum_a_condition (a : ℝ) (h₀ : 0 < a) 
  (h₁ : ∀ x : ℝ, 1 < x → x + a / (x - 1) ≥ 5) :
  4 ≤ a :=
sorry

end minimum_a_condition_l788_788003


namespace distance_between_centers_is_8_l788_788420

-- Definitions from conditions
def radius_sphere : ℝ := 10
def area_cross_sectional_circle : ℝ := 36 * real.pi

-- Distance to prove
def distance_center_to_center (r_s : ℝ) (a_c : ℝ) : ℝ :=
  real.sqrt (r_s^2 - (real.sqrt (a_c / real.pi))^2)

-- Theorem statement
theorem distance_between_centers_is_8 :
  distance_center_to_center radius_sphere area_cross_sectional_circle = 8 := 
sorry

end distance_between_centers_is_8_l788_788420


namespace other_endpoint_of_diameter_l788_788265

theorem other_endpoint_of_diameter :
  ∀ (C A : ℝ × ℝ), C = (5, -4) → A = (0, -9) → ∃ Q : ℝ × ℝ, Q = (10, 1) :=
by
  intros C A hC hA
  use (10, 1)
  rw [hC, hA]
  sorry

end other_endpoint_of_diameter_l788_788265


namespace maximum_value_expression_l788_788276

theorem maximum_value_expression (x : ℝ) :
  (∃ x : ℝ, (4 * x^2 + 12 * x + 29) / (4 * x^2 + 12 * x + 5) = 97) :=
begin
  sorry
end

end maximum_value_expression_l788_788276


namespace length_of_bridge_correct_l788_788694

noncomputable def length_of_bridge (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_seconds : ℝ) : ℝ :=
  let train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600
  let total_distance : ℝ := train_speed_ms * crossing_time_seconds
  total_distance - train_length

theorem length_of_bridge_correct :
  length_of_bridge 500 42 60 = 200.2 :=
by
  sorry -- Proof of the theorem

end length_of_bridge_correct_l788_788694


namespace find_real_values_x_l788_788785

noncomputable def p : Set ℝ := { x | (2 * x^3 + x^4 - 4 * x^5) / (2 * x + 2 * x^2 - 4 * x^4) ≥ 1 }

theorem find_real_values_x :
  ∀ x ∈ p, x ∈ Set.Iio (-2) ∪ Set.Ioo 0 (1 / 2) ∪ Set.Ioi (1 / 2) :=
begin
  sorry
end

end find_real_values_x_l788_788785


namespace projection_matrix_is_correct_l788_788354

noncomputable def projectionMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  let v : Fin 2 → ℝ := ![3, 4]
  (1 / (v 0 ^ 2 + v 1 ^ 2)) • (λ i j, v i * v j)

theorem projection_matrix_is_correct :
  projectionMatrix = ![![9/25, 12/25], ![12/25, 16/25]] :=
by
  sorry

end projection_matrix_is_correct_l788_788354


namespace problem_inequality_l788_788383

variable {x y : ℝ}

theorem problem_inequality (hx : 2 < x) (hy : 2 < y) : 
  (x^2 - x) / (y^2 + y) + (y^2 - y) / (x^2 + x) > 2 / 3 := 
  sorry

end problem_inequality_l788_788383


namespace probability_two_absent_one_present_l788_788053

theorem probability_two_absent_one_present (P_absent P_present : ℚ)
  (h_absent : P_absent = 1 / 20)
  (h_present : P_present = 19 / 20) :
  (3 * (P_absent * P_absent * P_present) * 100).round / 100 = 0.7 := by
sorry

end probability_two_absent_one_present_l788_788053


namespace sufficient_condition_perpendicular_l788_788159

-- Define the equations of the lines
def line1 (m : ℝ) (x y : ℝ) := m * x + (2 * m - 1) * y + 1 = 0
def line2 (m : ℝ) (x y : ℝ) := 3 * x + m * y + 3 = 0

-- Definition to check if two lines are perpendicular
def perpendicular (slope1 slope2 : ℝ) := slope1 * slope2 = -1

-- Prove that m = -1 is a sufficient condition for the lines to be perpendicular
theorem sufficient_condition_perpendicular (m : ℝ) :
    (∀ x y : ℝ, line1 m x y) ∧ (∀ x y : ℝ, line2 m x y) → m = -1 → (∃ x y : ℝ, ⊢ perpendicular (-m / (2*m - 1)) (-3 / m)) :=
by
  intros
  sorry

end sufficient_condition_perpendicular_l788_788159


namespace John_distance_proof_l788_788081

def initial_running_time : ℝ := 8
def increase_percentage : ℝ := 0.75
def initial_speed : ℝ := 8
def speed_increase : ℝ := 4

theorem John_distance_proof : 
  (initial_running_time + initial_running_time * increase_percentage) * (initial_speed + speed_increase) = 168 := 
by
  -- Proof can be completed here
  sorry

end John_distance_proof_l788_788081


namespace marissa_tied_boxes_l788_788113

theorem marissa_tied_boxes 
  (r_total : ℝ) (r_per_box : ℝ) (r_left : ℝ) (h_total : r_total = 4.5)
  (h_per_box : r_per_box = 0.7) (h_left : r_left = 1) :
  (r_total - r_left) / r_per_box = 5 :=
by
  sorry

end marissa_tied_boxes_l788_788113


namespace projection_onto_3_4_matrix_l788_788336

def projection_matrix := λ (u : ℝ) (v : ℝ), (3 * u + 4 * v) / 25

theorem projection_onto_3_4_matrix :
  ∀ (x y : ℝ),
  (λ (u v : ℝ), (3 * x + 4 * y) / 25) = (λ (u v : ℝ), (\(u * 9 / 25) + (v * 12 / 25), (u * 12 / 25) + (v * 16 / 25))) :=
by
  sorry

end projection_onto_3_4_matrix_l788_788336


namespace kelly_chris_boxes_ratio_l788_788907

theorem kelly_chris_boxes_ratio (X : ℝ) (h : X > 0) :
  (0.4 * X) / (0.6 * X) = 2 / 3 :=
by sorry

end kelly_chris_boxes_ratio_l788_788907


namespace john_run_distance_l788_788078

theorem john_run_distance :
  ∀ (initial_hours : ℝ) (increase_time_percent : ℝ) (initial_speed : ℝ) (increase_speed : ℝ),
  initial_hours = 8 → increase_time_percent = 0.75 → initial_speed = 8 → increase_speed = 4 →
  let increased_hours := initial_hours * increase_time_percent,
      total_hours := initial_hours + increased_hours,
      new_speed := initial_speed + increase_speed,
      distance := total_hours * new_speed in
  distance = 168 := 
by
  intros initial_hours increase_time_percent initial_speed increase_speed h_hours h_time h_speed h_increase
  let increased_hours := initial_hours * increase_time_percent
  let total_hours := initial_hours + increased_hours
  let new_speed := initial_speed + increase_speed
  let distance := total_hours * new_speed
  sorry

end john_run_distance_l788_788078


namespace regression_decrease_by_three_l788_788848

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 2 - 3 * x

-- Prove that when the explanatory variable increases by 1 unit, the predicted variable decreases by 3 units
theorem regression_decrease_by_three : ∀ x : ℝ, regression_equation (x + 1) = regression_equation x - 3 :=
by
  intro x
  unfold regression_equation
  sorry

end regression_decrease_by_three_l788_788848


namespace boat_distance_downstream_l788_788716

-- Definitions of the given conditions
def boat_speed_still_water : ℝ := 13
def stream_speed : ℝ := 4
def travel_time_downstream : ℝ := 4

-- Mathematical statement to be proved
theorem boat_distance_downstream : 
  let effective_speed_downstream := boat_speed_still_water + stream_speed
  in effective_speed_downstream * travel_time_downstream = 68 :=
by
  sorry

end boat_distance_downstream_l788_788716


namespace linda_buttons_minimum_l788_788550

theorem linda_buttons_minimum : ∃ m : ℕ, 
  (∀ W : ℕ, W > 1 ∧ W < m → m % W = 0 → W ≠ m / W) ∧ 
  (nat.totient m = 17) ∧ 
  (m = 2916) :=
by
  sorry

end linda_buttons_minimum_l788_788550


namespace vectors_perpendicular_implies_x_l788_788854

variables (x : ℝ)

def vector_a : ℝ × ℝ := (1, 3)
def vector_b : ℝ × ℝ := (x, 1)

def perpendicular_vectors (v1 v2 : ℝ × ℝ) : Prop :=
  v1.fst * v2.fst + v1.snd * v2.snd = 0

theorem vectors_perpendicular_implies_x :
  perpendicular_vectors vector_a vector_b → x = -3 :=
by
  sorry

end vectors_perpendicular_implies_x_l788_788854


namespace parabola_circle_properties_l788_788638

section ParabolaCircleTangent

variables {A1 A2 A3 P Q M : Point} 
variables {parabola : Parabola} 
variables {circle : Circle} 
variables {line_l : Line}

-- Definitions of points
def O := Point.mk 0 0
def M := Point.mk 2 0
def P := Point.mk 1 (Real.sqrt (2 * (1 / 2)))
def Q := Point.mk 1 (-Real.sqrt (2 * (1 / 2)))

-- Definition of geometrical constructs
def parabola := {p : Point // p.y^2 = p.x}
def circle := {c : Point // (c.x - 2)^2 + c.y^2 = 1}
def line_l := {l : Line // l.slope = ⊤ ∧ l.x_intercept = 1 }

-- Tangent properties for lines A1A2 and A1A3
def is_tangent {A B : Point} (l : Line) (circle : Circle) : Prop :=
  ∃ r: Real, (∥circle.center - A∥ = r) ∧ (∥circle.center - B∥ = r) ∧ (∥circle.center - (line.foot circle.center)∥ = r)

-- Theorem/Statement to prove:
theorem parabola_circle_properties :
  (parabola = {p : Point // p.y^2 = p.x}) →
  (circle = {c : Point // (c.x - 2)^2 + c.y^2 = 1}) →
  (∀ A1 A2 A3 : Point, A1 ∈ parabola → A2 ∈ parabola → A3 ∈ parabola → 
    (is_tangent (line_through A1 A2) circle) → (is_tangent (line_through A1 A3) circle) → 
    ⊥ ≤ distance_from_point_to_line (line_through A2 A3) circle.center = 1 ) :=
sorry

end ParabolaCircleTangent

end parabola_circle_properties_l788_788638


namespace prove_a_value_l788_788047

noncomputable def a_value (a : ℝ) : Prop :=
  (∃ (a : ℝ), a > 0 ∧
     (∃ (x y : ℝ), (x ^ 2 / a ^ 2 - y ^ 2 / 3 ^ 2 = 1) ∧  -- Equation of the hyperbola
                   ((a ^ 2 + 3 ^ 2) / a ^ 2 = 4)))         -- Condition using eccentricity

theorem prove_a_value : a_value (real.sqrt 3) :=
by
  unfold a_value
  use real.sqrt 3
  split
  { exact real.sqrt_pos.2 zero_lt_three }
  use [1, 1]  -- Example values to satisfy the hyperbola (placeholders)
  split
  { sorry }  -- Proof the example values satisfy the hyperbola equation
  { sorry }  -- Proof that (a^2 + 3^2) / a^2 = 4

end prove_a_value_l788_788047


namespace binomial_coeff_sum_eq_16_l788_788158

open Nat

theorem binomial_coeff_sum_eq_16 (n: ℕ) (h: n = 4) :
  (∑ k in range (n + 1), binomial n k) = 16 :=
by
  rw h
  sorry

end binomial_coeff_sum_eq_16_l788_788158


namespace OH_squared_l788_788919

variables {A B C O H : Type}
variables (a b c R : ℝ)

-- Define the conditions
def IsCircumcenter (O : Type) := true -- placeholder, requires precise definition
def IsOrthocenter (H : Type) := true -- placeholder, requires precise definition
def sideLengths (a b c : ℝ) := true -- placeholder, requires precise definition
def circumradius (R : ℝ) := R = 5
def sumOfSquareSides (a b c : ℝ) := a^2 + b^2 + c^2 = 50

-- The main statement to be proven
theorem OH_squared (h1 : IsCircumcenter O)
                   (h2 : IsOrthocenter H)
                   (h3 : sideLengths a b c)
                   (h4 : circumradius R)
                   (h5 : sumOfSquareSides a b c) :
    let OH_squared := 9 * R^2 - (a^2 + b^2 + c^2)
    in OH_squared = 175 := sorry

end OH_squared_l788_788919


namespace coeff_a_zero_l788_788467

-- Define the problem in Lean 4

theorem coeff_a_zero (a b c : ℝ) (h : ∀ p : ℝ, 0 < p → ∀ x, a * x^2 + b * x + c + p = 0 → 0 < x) :
  a = 0 :=
sorry

end coeff_a_zero_l788_788467


namespace constant_expenditure_reduction_l788_788049

theorem constant_expenditure_reduction:
  (fuel_price_increase : String → ℝ) 
  (fuel_reduction_needed : String → ℝ) 
  (cond_petrol : fuel_price_increase "petrol" = 0.40) 
  (cond_diesel : fuel_price_increase "diesel" = 0.25) 
  (cond_natural_gas : fuel_price_increase "natural_gas" = 0.15) :
  (abs (fuel_reduction_needed "petrol" - 0.2857) < 0.01) ∧
  (fuel_reduction_needed "diesel" = 0.20) ∧
  (abs (fuel_reduction_needed "natural_gas" - 0.1304) < 0.01) := 
sorry

end constant_expenditure_reduction_l788_788049


namespace sniper_B_has_greater_chance_of_winning_l788_788057

def pA (n : ℕ) : ℝ :=
  if n = 1 then 0.4 else if n = 2 then 0.1 else if n = 3 then 0.5 else 0

def pB (n : ℕ) : ℝ :=
  if n = 1 then 0.1 else if n = 2 then 0.6 else if n = 3 then 0.3 else 0

noncomputable def expected_score (p : ℕ → ℝ) : ℝ :=
  (1 * p 1) + (2 * p 2) + (3 * p 3)

theorem sniper_B_has_greater_chance_of_winning :
  expected_score pB > expected_score pA :=
by
  sorry

end sniper_B_has_greater_chance_of_winning_l788_788057


namespace original_number_conditions_l788_788722

theorem original_number_conditions (a : ℕ) :
  ∃ (y1 y2 : ℕ), (7 * a = 10 * 9 + y1) ∧ (9 * 9 = 10 * 8 + y2) ∧ y2 = 1 ∧ (a = 13 ∨ a = 14) := sorry

end original_number_conditions_l788_788722


namespace sum_of_valid_m_values_l788_788796

def valid_m_values (m : ℤ) : Prop :=
  ∃ x : ℕ, x > 0 ∧ (6 - 3 * (x - 1) = m * x - 9)

theorem sum_of_valid_m_values : 
  (∑ m in Finset.filter valid_m_values (Finset.range 20), m) = 21 :=
by sorry

end sum_of_valid_m_values_l788_788796


namespace perimeter_lt_pi_d_l788_788564

theorem perimeter_lt_pi_d {P : ℝ} {d : ℝ} (h : ∀ (s : ℝ), s ∈ sides ∨ s ∈ diagonals → s < d) : P < π * d :=
sorry

end perimeter_lt_pi_d_l788_788564


namespace product_of_decimals_l788_788205

def x : ℝ := 0.8
def y : ℝ := 0.12

theorem product_of_decimals : x * y = 0.096 :=
by
  sorry

end product_of_decimals_l788_788205


namespace smaller_angle_at_945_l788_788259

-- Definitions of conditions
def minute_hand_angle (time : ℕ) : ℝ :=
  if time = 9 * 60 + 45 then (3 / 4) * 360 else 0  -- Angle for minute hand at 9:45

def hour_hand_angle (hours minutes : ℕ) : ℝ :=
  if hours = 9 ∧ minutes = 45 then 9 * 30 + (30 * (minutes / 60.0)) else 0  -- Angle calculation for the hour hand at 9:45

-- Main theorem to prove
theorem smaller_angle_at_945 :
  let time := 9 * 60 + 45 in
  let hours := 9 in
  let minutes := 45 in
  let minute_angle := minute_hand_angle time in
  let hour_angle := hour_hand_angle hours minutes in
  abs (minute_angle - hour_angle) = 22.5 :=
by 
  sorry

end smaller_angle_at_945_l788_788259


namespace monotonic_intervals_and_range_of_a_l788_788430

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - (a + 2) * x + log x + 2

theorem monotonic_intervals_and_range_of_a (a : ℝ) (h : a ≤ 2) :
  (monotonic_intervals_of_f a) ∧ ((∀ x ∈ [1, 2], f a x ≥ 0) → (1 - 2 * log 2 ≤ a ∧ a ≤ 1/2) ∨ (a ≥ 1)) :=
by sorry

end monotonic_intervals_and_range_of_a_l788_788430


namespace marissa_tied_boxes_l788_788114

theorem marissa_tied_boxes 
  (r_total : ℝ) (r_per_box : ℝ) (r_left : ℝ) (h_total : r_total = 4.5)
  (h_per_box : r_per_box = 0.7) (h_left : r_left = 1) :
  (r_total - r_left) / r_per_box = 5 :=
by
  sorry

end marissa_tied_boxes_l788_788114


namespace photos_per_day_in_january_l788_788144

theorem photos_per_day_in_january (P_total : ℕ) (P_week : ℕ) (W_Feb : ℕ) (D_Jan : ℕ) (h1 : P_total = 146) (h2 : P_week = 21) (h3 : W_Feb = 4) (h4 : D_Jan = 31) :
  (P_total - P_week * W_Feb) / D_Jan = 2 :=
by
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end photos_per_day_in_january_l788_788144


namespace sum_divisible_by_seventeen_l788_788362

theorem sum_divisible_by_seventeen :
  (90 + 91 + 92 + 93 + 94 + 95 + 96 + 97) % 17 = 0 := 
by 
  sorry

end sum_divisible_by_seventeen_l788_788362


namespace projection_matrix_3_4_l788_788330

theorem projection_matrix_3_4 :
  let v := λ α : Type, @vector α 2 := ![3, 4]
  let proj := λ x : vector ℝ 2, (v ℝ ⬝ x) / (v ℝ ⬝ v ℝ) • v ℝ
  proj = (λ x : vector ℝ 2, matrix.mul_vec ![
     ![9 / 25, 12 / 25],
     ![12 / 25, 16 / 25]
  ] x) :=
by sorry

end projection_matrix_3_4_l788_788330


namespace projection_matrix_correct_l788_788310

variables {R : Type*} [field R] [decidable_eq R]
variables (x y : R)
def vector_v : matrix (fin 2) (fin 1) R := ![![3], ![4]]
def vector_u : matrix (fin 2) (fin 1) R := ![![x], ![y]]
def projection_matrix : matrix (fin 2) (fin 2) R := ![![9/25, 12/25], ![12/25, 16/25]]

theorem projection_matrix_correct :
  (projection_matrix R) ⬝ (vector_u x y) = (25 : R)⁻¹ • (transpose (vector_v 3 4) ⬝ (vector_u x y)) ⬝ (vector_v 3 4) := 
sorry

end projection_matrix_correct_l788_788310


namespace dara_half_age_of_jane_in_6_years_l788_788610

-- Definitions from conditions
def jane_current_age : ℕ := 28
def dara_future_years : ℕ := 14
def dara_future_age : ℕ := 25

-- Helper to find Dara's current age
def dara_current_age : ℕ := dara_future_age - dara_future_years

-- The proof statement: prove that Dara will be half the age of Jane in 6 years
theorem dara_half_age_of_jane_in_6_years :
  ∃ x : ℕ, dara_current_age + x = (jane_current_age + x) / 2 ∧ x = 6 :=
by
  use 6
  simp [jane_current_age, dara_current_age]
  conv_rhs { rw [←add_succ, ←Nat.add_assoc, Nat.add_sub_cancel_left] }
  exact Nat.succ_ne_zero 2
  sorry

end dara_half_age_of_jane_in_6_years_l788_788610


namespace total_charge_for_trip_l788_788213

-- Define the initial fee
def initial_fee : ℝ := 2.25

-- Define the additional charge per 2/5 mile increment
def additional_charge_per_increment : ℝ := 0.25

-- Define the distance of the trip in miles
def trip_distance : ℝ := 3.6

-- Define the length of each increment in miles
def increment_length : ℝ := 2 / 5

-- Define the total number of increments for the given trip
noncomputable def number_of_increments : ℝ := trip_distance / increment_length

-- Define the total additional charge based on the number of increments
noncomputable def total_additional_charge : ℝ := number_of_increments * additional_charge_per_increment

-- Define the total charge for the trip
noncomputable def total_charge : ℝ := initial_fee + total_additional_charge

-- State the theorem that the total charge for a trip of 3.6 miles is $6.30
theorem total_charge_for_trip : total_charge = 6.30 := by
  sorry

end total_charge_for_trip_l788_788213


namespace projection_onto_vector_l788_788318

noncomputable def projection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  ![\[9 / 25, 12 / 25\], \[12 / 25, 16 / 25\]]

theorem projection_onto_vector:
    ∀ (x y : ℚ), (Matrix.mul_vec projection_matrix ![\x, \y]) = ![(9 * x + 12 * y) / 25, (12 * x + 16 * y) / 25] := by
  sorry

end projection_onto_vector_l788_788318


namespace calc_num_articles_l788_788863

-- Definitions based on the conditions
def cost_price (C : ℝ) : ℝ := C
def selling_price (C : ℝ) : ℝ := 1.10000000000000004 * C
def num_articles (n : ℝ) (C : ℝ) (S : ℝ) : Prop := 55 * C = n * S

-- Proof Statement
theorem calc_num_articles (C : ℝ) : ∃ n : ℝ, num_articles n C (selling_price C) ∧ n = 50 :=
by sorry

end calc_num_articles_l788_788863


namespace womenInBusinessClass_l788_788574

-- Given conditions
def totalPassengers : ℕ := 300
def percentageWomen : ℚ := 70 / 100
def percentageWomenBusinessClass : ℚ := 15 / 100

def numberOfWomen (totalPassengers : ℕ) (percentageWomen : ℚ) : ℚ := 
  totalPassengers * percentageWomen

def numberOfWomenBusinessClass (numberOfWomen : ℚ) (percentageWomenBusinessClass : ℚ) : ℚ := 
  numberOfWomen * percentageWomenBusinessClass

-- Theorem to prove
theorem womenInBusinessClass (totalPassengers : ℕ) (percentageWomen : ℚ) (percentageWomenBusinessClass : ℚ) :
  numberOfWomenBusinessClass (numberOfWomen totalPassengers percentageWomen) percentageWomenBusinessClass = 32 := 
by 
  -- The proof steps would go here
  sorry

end womenInBusinessClass_l788_788574


namespace triangle_DOG_angle_GHG_l788_788070

-- Define the triangle DOG with given properties
structure Triangle :=
  (D G O : Type*)
  [inst : HasAngle D]
  [inst : HasAngle G]
  [inst : HasAngle O]
  (DOG : Angle)
  (DGO : Angle)
  (GOD : Angle)
  (DH : AngleBisector D G O DOG)
  (DG' : AngleBisector D G O DGO)

noncomputable def angle_GHG'_eq_42 : Prop :=
  ∃ (DOG DGO GOD : Angle) (DH DG' : Triangle),
  DOG.measure = 48 ∧
  DGO.measure = 48 ∧
  GOD.measure = 84 ∧
  DH.is_bisector DOG ∧
  DG'.is_bisector DGO ∧ 
  ∠ GHG' = 42

theorem triangle_DOG_angle_GHG'_eq_42
  (triangle_DOG : Triangle):
  angle_GHG'_eq_42 :=
sorry

end triangle_DOG_angle_GHG_l788_788070


namespace rods_to_furlongs_l788_788002

theorem rods_to_furlongs : ∀ (rods : ℕ), rods = 1000 → rols / 50 = 20 :=
by
  intros rods h1
  rw h1
  exact Nat.div_eq_of_eq_mul_right (by decide) rfl

end rods_to_furlongs_l788_788002


namespace projection_onto_3_4_matrix_l788_788335

def projection_matrix := λ (u : ℝ) (v : ℝ), (3 * u + 4 * v) / 25

theorem projection_onto_3_4_matrix :
  ∀ (x y : ℝ),
  (λ (u v : ℝ), (3 * x + 4 * y) / 25) = (λ (u v : ℝ), (\(u * 9 / 25) + (v * 12 / 25), (u * 12 / 25) + (v * 16 / 25))) :=
by
  sorry

end projection_onto_3_4_matrix_l788_788335


namespace largest_red_socks_l788_788726

noncomputable def maxRedSocks : Nat :=
  let r := 897
  let b := 701
  let y := 702
  let total := r + b + y
  have cond1 : total ≤ 2300 := by sorry
  have prob := (Nat.choose r 3 + Nat.choose b 3 + Nat.choose y 3) * 3 = Nat.choose total 3 := by sorry
  r

-- Statement of the theorem
theorem largest_red_socks (r b y : ℕ) (h : r + b + y ≤ 2300)
  (hprob : (Nat.choose r 3 + Nat.choose b 3 + Nat.choose y 3) * 3 = Nat.choose (r + b + y) 3) : r ≤ 897 := by
  sorry

end largest_red_socks_l788_788726


namespace projection_vector_of_a_onto_b_l788_788472

open Real

-- Definitions of vectors and operations
def vec_a : ℝ × ℝ := (sqrt 3, 3)
def vec_b : ℝ × ℝ := (-2, 0)

-- Dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Squared magnitude of a vector
def magnitude_squared (v : ℝ × ℝ) : ℝ := v.1 * v.1 + v.2 * v.2

-- Projection of vector a onto vector b
def projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  let coeff := (dot_product a b) / (magnitude_squared b)
  (coeff * b.1, coeff * b.2)

-- The theorem statement
theorem projection_vector_of_a_onto_b : projection vec_a vec_b = (sqrt 3, 0) :=
by sorry

end projection_vector_of_a_onto_b_l788_788472


namespace smallest_solution_to_equation_l788_788370

noncomputable def smallest_solution := (11 - Real.sqrt 445) / 6

theorem smallest_solution_to_equation:
  ∃ x : ℝ, (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ (x = smallest_solution) :=
sorry

end smallest_solution_to_equation_l788_788370


namespace number_of_integer_points_l788_788474

theorem number_of_integer_points : 
  let region := {p : ℤ × ℤ | 3 * p.1 ≤ p.2 ∧ p.1 ≤ 3 * p.2 ∧ p.1 + p.2 ≤ 100}
  in (region.card = 2551) :=
sorry

end number_of_integer_points_l788_788474


namespace projection_matrix_l788_788343

theorem projection_matrix
  (x y : ℝ) :
  let v := ![3, 4]
  let proj_v := (v ⬝ ![x, y]) / (v ⬝ v) • v
  let proj_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
    ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]] in
  proj_v = proj_matrix.mul_vec ![x, y] :=
by
  let v := ![3, 4]
  let proj_v := (v ⬝ ![x, y]) / (v ⬝ v) • v
  let proj_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]]
  sorry

end projection_matrix_l788_788343


namespace flyers_left_l788_788507

theorem flyers_left (total_flyers : ℕ) (jack_flyers : ℕ) (rose_flyers : ℕ) (h1 : total_flyers = 1236) (h2 : jack_flyers = 120) (h3 : rose_flyers = 320) : (total_flyers - (jack_flyers + rose_flyers) = 796) := 
by
  sorry

end flyers_left_l788_788507


namespace max_value_f_l788_788414

-- Definitions and Conditions
def f (x : ℝ) (b c : ℝ) : ℝ := (1/2) * x^2 + b / x + c
def g (x : ℝ) : ℝ := (1/4) * x + 1 / x
def M : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}

-- Problem Statement
theorem max_value_f (b c x0 : ℝ) (hM : x0 ∈ M)
  (h1 : ∀ x ∈ M, f x b (c+1/2) ≥ f x0 b (c+1/2))
  (h2 : ∀ x ∈ M, g x ≥ g x0)
  (h3 : f x0 b (c+1/2) = g x0):
  f 4 8 (-5) = 5 := sorry

end max_value_f_l788_788414


namespace odd_function_expression_l788_788936

def f : ℝ → ℝ := sorry -- The definition of f is implied in the proof below.

theorem odd_function_expression (f_odd : ∀ x, f (-x) = -f x)
  (pos_expr : ∀ x, 0 < x → f x = -x * log (1 + x)) :
  ∀ x, x < 0 → f x = -x * log (1 - x) :=
sorry

end odd_function_expression_l788_788936


namespace quadratic_poly_coeffs_l788_788787

theorem quadratic_poly_coeffs:
  ∀ (m n : ℝ),
  (∀ x : ℝ, polynomial.eval x (polynomial.C 1 * polynomial.X^2 + polynomial.C m * polynomial.X + polynomial.C n) = polynomial.eval x (polynomial.X - polynomial.C m) → polynomial.eval x (polynomial.C m)) ∧
  (∀ x : ℝ, polynomial.eval x (polynomial.X^2 + polynomial.C m * polynomial.X + polynomial.C n) = polynomial.eval x (polynomial.X - polynomial.C n) → polynomial.eval x (polynomial.C n)) →
  (m = 0 ∧ n = 0) ∨ (m = 1 / 2 ∧ n = 0) ∨ (m = 1 ∧ n = -1) :=
by
  sorry

end quadratic_poly_coeffs_l788_788787


namespace diametrically_opposite_to_11_is_1_l788_788603

theorem diametrically_opposite_to_11_is_1
    (arrangement : Fin 20 → Fin 20)
    (A B : Fin 20 → Nat)
    (hA : ∀ k : Fin 20, A k = (Finset.filter (· < k) (Finset.range 9).image (arrangement ∘ (k + ·))) .card)
    (hB : ∀ k : Fin 20, B k = (Finset.filter (· < k) (Finset.range 9).image (arrangement ∘ (k - ·))) .card)
    (h : ∀ k : Fin 20, A k = B k) :
    arrangement 11 = 1 := sorry

end diametrically_opposite_to_11_is_1_l788_788603


namespace percent_students_own_cats_l788_788875

theorem percent_students_own_cats 
  (total_students : ℕ) (cat_owners : ℕ) (h1 : total_students = 300) (h2 : cat_owners = 45) :
  (cat_owners : ℚ) / total_students * 100 = 15 := 
by
  sorry

end percent_students_own_cats_l788_788875


namespace number_of_positive_integers_l788_788794

theorem number_of_positive_integers :
  let S := {x : ℕ | 30 < x^2 + 8 * x + 16 ∧ x^2 + 8 * x + 16 < 55} in
  S.card = 2 :=
by
  sorry

end number_of_positive_integers_l788_788794


namespace problem_dihedral_angle_l788_788879

-- Assume noncomputable definitions for calculations involving non-rational trigonometric functions
noncomputable def dihedral_angle_between_planes (a : ℝ) : ℝ :=
  Real.arctan (2 / 3)

theorem problem_dihedral_angle :
  let BB1 BC BL CM DN : ℝ := 5 * a, 3 * a, 3 * a, 2 * a, a in
  let angle := dihedral_angle_between_planes a in
  angle = Real.arctan (2 / 3) :=
by
  -- Proof is omitted
  sorry

end problem_dihedral_angle_l788_788879


namespace cogs_produced_after_speed_increase_l788_788255

-- Define the initial conditions of the problem
def initial_cogs := 60
def initial_rate := 15
def increased_rate := 60
def average_output := 24

-- Variables to represent the number of cogs produced after the speed increase and the total time taken for each phase
variable (x : ℕ)

-- Assuming the equations representing the conditions
def initial_time := initial_cogs / initial_rate
def increased_time := x / increased_rate

def total_cogs := initial_cogs + x
def total_time := initial_time + increased_time

-- Define the overall average output equation
def average_eq := average_output * total_time = total_cogs

-- The proposition we want to prove
theorem cogs_produced_after_speed_increase : x = 60 :=
by
  -- Using the equation from the conditions
  have h1 : average_eq := sorry
  sorry

end cogs_produced_after_speed_increase_l788_788255


namespace divide_660_stones_into_30_heaps_l788_788961

theorem divide_660_stones_into_30_heaps :
    ∃ (heaps : Fin 30 → ℕ), (∑ i, heaps i = 660) ∧ (∀ i j, heaps i < 2 * heaps j) ∨ (heaps j < 2 * heaps i) := 
sorry

end divide_660_stones_into_30_heaps_l788_788961


namespace projection_matrix_3_4_l788_788328

theorem projection_matrix_3_4 :
  let v := λ α : Type, @vector α 2 := ![3, 4]
  let proj := λ x : vector ℝ 2, (v ℝ ⬝ x) / (v ℝ ⬝ v ℝ) • v ℝ
  proj = (λ x : vector ℝ 2, matrix.mul_vec ![
     ![9 / 25, 12 / 25],
     ![12 / 25, 16 / 25]
  ] x) :=
by sorry

end projection_matrix_3_4_l788_788328


namespace no_such_function_exists_l788_788136

open Classical

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (f 0 > 0) ∧ (∀ (x y : ℝ), f (x + y) ≥ f x + y * f (f x)) :=
sorry

end no_such_function_exists_l788_788136


namespace cylindrical_pencils_common_point_l788_788405

theorem cylindrical_pencils_common_point :
  ∃ P : fin 6 → ℝ × ℝ × ℝ, ∀ i j : fin 6, i ≠ j → ∃ p : ℝ × ℝ × ℝ, on_boundary (P i) (d) p ∧ on_boundary (P j) (d) p :=
sorry

-- Definitions for "on_boundary" must be provided, assuming the standard definition of touching the boundary of the cylindrical pencil.

end cylindrical_pencils_common_point_l788_788405


namespace coplanar_AD_eq_linear_combination_l788_788814

-- Define the points
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨4, 1, 3⟩
def B : Point3D := ⟨2, 3, 1⟩
def C : Point3D := ⟨3, 7, -5⟩
def D : Point3D := ⟨11, -1, 3⟩

-- Define the vectors
def vector (P Q : Point3D) : Point3D := ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

def AB := vector A B
def AC := vector A C
def AD := vector A D

-- Coplanar definition: AD = λ AB + μ AC
theorem coplanar_AD_eq_linear_combination (lambda mu : ℝ) :
  AD = ⟨lambda * 2 + mu * (-1), lambda * (-2) + mu * 6, lambda * (-2) + mu * (-8)⟩ :=
sorry

end coplanar_AD_eq_linear_combination_l788_788814


namespace triangle_ratio_sum_eq_one_l788_788931

theorem triangle_ratio_sum_eq_one 
  (A B C O D E F : Type) 
  (h1 : point_in_triangle ABC O)
  (h2 : parallel (line_through_point O parallel_to BC) CA D)
  (h3 : parallel (line_through_point O parallel_to CA) AB E)
  (h4 : parallel (line_through_point O parallel_to AB) BC F) :
  (BF / BC) + (AE / AB) + (CD / AC) = 1 := 
  sorry

end triangle_ratio_sum_eq_one_l788_788931


namespace watch_correction_l788_788248

theorem watch_correction :
  ∀ (rate_loss_per_day : ℚ) (initial_time : ℚ) (final_time : ℚ) (days_passed : ℚ),
  rate_loss_per_day = 13 / 4 →
  initial_time = 0 →
  final_time = 188 →
  days_passed = 7 →
  let hourly_loss := rate_loss_per_day / 24 in
  let total_loss := final_time * hourly_loss in
  total_loss = 25 + 17 / 96 :=

by
  intros rate_loss_per_day initial_time final_time days_passed
  intro h1
  intro h2
  intro h3
  intro h4
  let hourly_loss := rate_loss_per_day / 24
  let total_loss := final_time * hourly_loss
  have : total_loss = 25 + 17 / 96
  sorry

end watch_correction_l788_788248


namespace num_pairs_bound_l788_788543

noncomputable def number_of_pairs (n : ℕ) (α : Fin n → E) [InnerProductSpace ℝ E] : ℕ :=
  Finset.card ((Finset.univ : Finset (Fin n)).filter (λ i => (Finset.univ \ Finset.range i).card > 0 ∧ 
  ∃ j, i < j ∧ ⟪α i, α j⟫ < 0))

theorem num_pairs_bound (n : ℕ) (α : Fin n → E) [InnerProductSpace ℝ E] (h : n ≥ 2) :
  number_of_pairs n α ≤ n^2 / 3 :=
sorry

end num_pairs_bound_l788_788543


namespace problem1_problem2_problem3_l788_788841

namespace ProofProblems

def f_k (k : ℤ) (x : ℝ) : ℝ := 2^x - (k-1)*2^(-x)
def g (x : ℝ) : ℝ := (f_k 2 x) / (f_k 0 x)

-- Problem (1)
theorem problem1 (x : ℝ) : (f_k 2 x = 2) → x = Real.log (Real.sqrt 2 + 1) :=
sorry

-- Problem (2)
theorem problem2 : ∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ < g x₂ :=
sorry

-- Problem (3)
theorem problem3 (m : ℝ) : (∀ x : ℝ, 1 ≤ x → ∃ y : ℝ, y = f_k 0 (2*x) + 2*m*(f_k 2 x) ∧ y = 0) → m ≤ -17 / 12 :=
sorry

end ProofProblems

end problem1_problem2_problem3_l788_788841


namespace sequence_b_two_l788_788172

theorem sequence_b_two (b : ℕ → ℝ) 
  (h₁ : b 1 = 25) 
  (h₂ : b 10 = 125) 
  (h₃ : ∀ n, n ≥ 3 → b n = (∑ i in finset.range (n-1), b (i+1)) / (n-1)) :
  b 2 = 225 :=
sorry

end sequence_b_two_l788_788172


namespace divide_660_stones_into_30_heaps_l788_788957

theorem divide_660_stones_into_30_heaps :
    ∃ (heaps : Fin 30 → ℕ), (∑ i, heaps i = 660) ∧ (∀ i j, heaps i < 2 * heaps j) ∨ (heaps j < 2 * heaps i) := 
sorry

end divide_660_stones_into_30_heaps_l788_788957


namespace fifteenth_battery_replacement_month_l788_788748

theorem fifteenth_battery_replacement_month :
  (98 % 12) + 1 = 4 :=
by
  sorry

end fifteenth_battery_replacement_month_l788_788748


namespace min_max_value_of_F_l788_788596

theorem min_max_value_of_F :
  ∀ (A B : ℝ), (∀ x ∈ Icc 0 (3 * π / 2),
    abs (cos x ^ 2 + 2 * sin x * cos x - sin x ^ 2 + A * x + B) ≤ sqrt 2) ↔ (A = 0 ∧ B = 0) := sorry

end min_max_value_of_F_l788_788596


namespace dissimilar_terms_expansion_count_l788_788773

noncomputable def num_dissimilar_terms_in_expansion (a b c d : ℝ) : ℕ :=
  let n := 8
  let k := 4
  Nat.choose (n + k - 1) (k - 1)

theorem dissimilar_terms_expansion_count : 
  num_dissimilar_terms_in_expansion a b c d = 165 := by
  sorry

end dissimilar_terms_expansion_count_l788_788773


namespace coprime_count_multiple_n_l788_788942

theorem coprime_count_multiple_n (A n : ℕ) (hA : A > 1) (hn : n > 1) :
  ∃ k : ℕ, k * n = (nat.totient (A^n - 1)) :=
by sorry

end coprime_count_multiple_n_l788_788942


namespace divide_660_stones_into_30_piles_l788_788977

theorem divide_660_stones_into_30_piles :
  ∃ (heaps : Fin 30 → ℕ),
    (∑ i, heaps i = 660) ∧
    ∀ i j, heaps i ≤ 2 * heaps j :=
sorry

end divide_660_stones_into_30_piles_l788_788977


namespace find_k_l788_788715

theorem find_k :
  ∃ k : ℕ, (k > 0) ∧ ((24 / (8 + k) - k / (8 + k) = 1) → k = 8) :=
by
  use 8
  split
  · trivial
  · intro h
    sorry

end find_k_l788_788715


namespace sequence_general_formula_sum_of_b_l788_788932

/-- Let {a_n} be a sequence of positive terms with a common difference such that
    a_3 = 3, and a_2, a_5 - 1, a_6 + 2 form a geometric sequence. Prove that 
    a_n = n for all natural numbers n. -/
theorem sequence_general_formula (a : ℕ → ℕ) (h : ∀ n, a (n + 1) - a n = a 2 - a 1) 
  (h3 : a 3 = 3) (h_geo : (a 2, a 5 - 1, a 6 + 2) = (a 5 - 1)^2 = a 2 * (a 6 + 2))
  : ∀ n, a n = n := 
sorry

/-- Given the general formula for the sequence {a_n}, prove that if S_n denotes the 
    sum of the first n terms of {a_n}, and b_n = 1 / S_n, then the sum of the first n 
    terms of {b_n}, denoted by T_n, equals 2n / (n + 1). -/
theorem sum_of_b (a : ℕ → ℕ) (h : ∀ n, a n = n) (S : ℕ → ℕ) (h_sum : ∀ n, S n = n * (n + 1) / 2)
  (b : ℕ → ℕ) (h_b : ∀ n, b n = 2 / (n * (n + 1))) (T : ℕ → ℕ) 
  (h_T : T = ∑ i in range n, b i)
  : ∀ n, T n = 2 * n / (n + 1) := 
sorry

end sequence_general_formula_sum_of_b_l788_788932


namespace categorize_numbers_l788_788292

def given_numbers : List ℝ := [-3, -1/3, -|-3|, Real.pi, -0.3, 0, Real.cbrt 16, 1.1010010001]

def is_integer (x : ℝ) : Prop :=
  ∃ (n : ℤ), x = n

def is_negative_fraction (x : ℝ) : Prop :=
  ∃ (a b : ℤ), b ≠ 0 ∧ a < 0 ∧ x = a / b

def is_irrational (x : ℝ) : Prop :=
  ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem categorize_numbers :
  ∃ ints neg_fracs irrats : List ℝ,
    ints = [-3, -|-3|, 0] ∧
    neg_fracs = [-1/3, -0.3] ∧
    irrats = [Real.pi, Real.cbrt 16] ∧
    (∀ x ∈ ints, is_integer x) ∧
    (∀ x ∈ neg_fracs, is_negative_fraction x) ∧
    (∀ x ∈ irrats, is_irrational x) :=
by
  sorry

end categorize_numbers_l788_788292


namespace john_run_distance_l788_788080

theorem john_run_distance :
  ∀ (initial_hours : ℝ) (increase_time_percent : ℝ) (initial_speed : ℝ) (increase_speed : ℝ),
  initial_hours = 8 → increase_time_percent = 0.75 → initial_speed = 8 → increase_speed = 4 →
  let increased_hours := initial_hours * increase_time_percent,
      total_hours := initial_hours + increased_hours,
      new_speed := initial_speed + increase_speed,
      distance := total_hours * new_speed in
  distance = 168 := 
by
  intros initial_hours increase_time_percent initial_speed increase_speed h_hours h_time h_speed h_increase
  let increased_hours := initial_hours * increase_time_percent
  let total_hours := initial_hours + increased_hours
  let new_speed := initial_speed + increase_speed
  let distance := total_hours * new_speed
  sorry

end john_run_distance_l788_788080


namespace andrew_age_l788_788754

/-- 
Andrew and his five cousins are ages 4, 6, 8, 10, 12, and 14. 
One afternoon two of his cousins whose ages sum to 18 went to the movies. 
Two cousins younger than 12 but not including the 8-year-old went to play baseball. 
Andrew and the 6-year-old stayed home. How old is Andrew?
-/
theorem andrew_age (ages : Finset ℕ) (andrew_age: ℕ)
  (h_ages : ages = {4, 6, 8, 10, 12, 14})
  (movies : Finset ℕ) (baseball : Finset ℕ)
  (h_movies1 : movies.sum id = 18)
  (h_baseball1 : ∀ x ∈ baseball, x < 12 ∧ x ≠ 8)
  (home : Finset ℕ) (h_home : home = {6, andrew_age}) :
  andrew_age = 12 :=
sorry

end andrew_age_l788_788754


namespace simplify_equation_l788_788580

variable {x : ℝ}

theorem simplify_equation : (1 / (x - 1) + 3 = 3 * x / (1 - x)) → 1 + 3 * (x - 1) = -3 * x :=
by
  sorry

end simplify_equation_l788_788580


namespace hansels_raise_percentage_l788_788453

noncomputable def initial_salary_hansel : ℕ := 30000
noncomputable def initial_salary_gretel : ℕ := 30000
def raise_percentage_gretel : ℝ := 0.15
noncomputable def new_salary_gretel := initial_salary_gretel + (raise_percentage_gretel * initial_salary_gretel)
noncomputable def salary_difference : ℕ := 1500
noncomputable def new_salary_hansel := new_salary_gretel - salary_difference

theorem hansels_raise_percentage :
  let raise_amount_hansel := new_salary_hansel - initial_salary_hansel in
  let raise_percentage_hansel := (raise_amount_hansel : ℝ) / initial_salary_hansel.to_real * 100 in
  raise_percentage_hansel = 10 :=
sorry

end hansels_raise_percentage_l788_788453


namespace stones_partition_l788_788996

theorem stones_partition (total_stones : ℕ) (piles : ℕ) (heaps : ℕ → ℕ) 
  (h_total : total_stones = 660)
  (h_piles : piles = 30)
  (h_sum_heaps : ∑ i in range piles, heaps i = 660)
  (h_factor : ∀ i j, i < piles → j < piles → heaps i ≤ 2 * heaps j) :
  ∃ heaps : Π i : ℕ, i < piles → ℕ,
    (∑ i in range piles, heaps i = 660) ∧
    (∀ i j, i < piles → j < piles → heaps i ≤ 2 * heaps j) :=
  sorry

end stones_partition_l788_788996


namespace abs_diff_ge_abs_sum_iff_non_positive_prod_l788_788800

theorem abs_diff_ge_abs_sum_iff_non_positive_prod (a b : ℝ) : 
  |a - b| ≥ |a| + |b| ↔ a * b ≤ 0 := 
by sorry

end abs_diff_ge_abs_sum_iff_non_positive_prod_l788_788800


namespace projection_onto_vector_l788_788320

noncomputable def projection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  ![\[9 / 25, 12 / 25\], \[12 / 25, 16 / 25\]]

theorem projection_onto_vector:
    ∀ (x y : ℚ), (Matrix.mul_vec projection_matrix ![\x, \y]) = ![(9 * x + 12 * y) / 25, (12 * x + 16 * y) / 25] := by
  sorry

end projection_onto_vector_l788_788320


namespace exists_large_n_fractional_part_gt_999999_l788_788499

theorem exists_large_n_fractional_part_gt_999999 :
  ∃ n : ℕ, (let base : ℝ := 2 + Real.sqrt 2 in
            let frac_part := base^n - Real.floor (base^n) in
            frac_part > 0.999999) :=
begin
  sorry
end

end exists_large_n_fractional_part_gt_999999_l788_788499


namespace projection_matrix_l788_788344

theorem projection_matrix
  (x y : ℝ) :
  let v := ![3, 4]
  let proj_v := (v ⬝ ![x, y]) / (v ⬝ v) • v
  let proj_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
    ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]] in
  proj_v = proj_matrix.mul_vec ![x, y] :=
by
  let v := ![3, 4]
  let proj_v := (v ⬝ ![x, y]) / (v ⬝ v) • v
  let proj_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]]
  sorry

end projection_matrix_l788_788344


namespace kendalls_nickels_l788_788522

theorem kendalls_nickels :
  ∀ (n_quarters n_dimes n_nickels : ℕ),
  (n_quarters = 10) →
  (n_dimes = 12) →
  ((n_quarters * 25) + (n_dimes * 10) + (n_nickels * 5) = 400) →
  n_nickels = 6 :=
by
  intros n_quarters n_dimes n_nickels hq hd heq
  sorry

end kendalls_nickels_l788_788522


namespace sequence_eventually_constant_l788_788542

def sequence_a (n : ℕ) (a : ℕ → ℕ) (k : ℕ) : ℕ :=
  if k = 1 then n
  else 
    let sum_to_k_minus_1 := (Finset.range (k - 1)).sum (λ i, a (i + 1)) in
    (Finset.range k).filter 
      (λ ak, (sum_to_k_minus_1 + ak) % k = 0).nth 0 |>.getD 0

theorem sequence_eventually_constant (n : ℕ) (h : 0 < n) :
  ∃ b N, ∀ k ≥ N, sequence_a n (sequence_a n) k = b := 
sorry

end sequence_eventually_constant_l788_788542


namespace medians_sum_of_sides_l788_788204

noncomputable def square (x : ℝ) := x * x

def medians_sum_squares (a b c : ℝ) (m_a m_b m_c : ℝ) : ℝ :=
  square m_a + square m_b + square m_c

theorem medians_sum_of_sides (a b c : ℝ) (m_a m_b m_c : ℝ) :
  a = 13 → b = 14 → c = 15 →
  2 * (square m_a + square m_b + square m_c) = 2 * (square a + square b + square c) →
  medians_sum_squares a b c m_a m_b m_c = 590 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  simp only [square] at h4
  -- sum of side lengths squared
  have side_sum_squares : 15 * 15 + 14 * 14 + 13 * 13 = 590 := by norm_num
  rw side_sum_squares at h4
  -- Simplify
  linarith

end medians_sum_of_sides_l788_788204


namespace eggs_left_l788_788180

theorem eggs_left (total_eggs: ℕ) (eggs_taken: ℕ) : total_eggs = 47 → eggs_taken = 5 → (total_eggs - eggs_taken) = 42 :=
by
  intros h₁ h₂
  rw [h₁, h₂]
  exact rfl

end eggs_left_l788_788180


namespace compute_expression_l788_788090

def f (x : ℝ) := x - 3
def g (x : ℝ) := x / 2
def f_inv (x : ℝ) := x + 3
def g_inv (x : ℝ) := x * 2

theorem compute_expression : 
  f (g_inv (f_inv (g (f_inv (g (f 23)))))) = 16 :=
by
  sorry

end compute_expression_l788_788090


namespace abs_diff_gt_1_probability_l788_788139

def fair_coin_flip : ℕ := sorry  -- Abstractly represent a fair coin flip
def choose_number : ℕ → ℝ := sorry  -- Represent the number choosing process based on a coin flip

-- Function to choose a number based on the procedure
noncomputable def select_number (flip: ℕ) : ℝ :=
  match flip with
  | 0 => if fair_coin_flip = 0 then 0 else 2
  | _ => choose_number (flip - 1)

-- Function to implement the selection of two independent numbers
noncomputable def random_pair : ℝ × ℝ :=
  let flip1 := fair_coin_flip in
  let flip2 := fair_coin_flip in
  (select_number flip1, select_number flip2)

-- Probability calculation placeholder
noncomputable def probability_abs_diff_gt_1 : ℚ := 
  let (x, y) := random_pair
  in sorry

theorem abs_diff_gt_1_probability : probability_abs_diff_gt_1 = 5 / 8 := 
  sorry

end abs_diff_gt_1_probability_l788_788139


namespace flyers_left_to_hand_out_l788_788508

-- Definitions for given conditions
def total_flyers : Nat := 1236
def jack_handout : Nat := 120
def rose_handout : Nat := 320

-- Statement of the problem
theorem flyers_left_to_hand_out : total_flyers - (jack_handout + rose_handout) = 796 :=
by
  -- proof goes here
  sorry

end flyers_left_to_hand_out_l788_788508


namespace xyz_expr_min_max_l788_788946

open Real

theorem xyz_expr_min_max (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h_sum : x + y + z = 1) :
  ∃ m M : ℝ, m = 0 ∧ M = 1/4 ∧
    (∀ x y z : ℝ, x + y + z = 1 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 →
      xy + yz + zx - 3 * xyz ≥ m ∧ xy + yz + zx - 3 * xyz ≤ M) :=
sorry

end xyz_expr_min_max_l788_788946


namespace no_integer_x_exists_l788_788065

-- Define the problem conditions.
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

theorem no_integer_x_exists (n : ℕ) (x : ℤ) :
  n = 343 → is_three_digit n → ∀ x : ℤ, ¬(log n 3 + log n (x) = log n n) :=
by
  intro h1 h2 x
  sorry

end no_integer_x_exists_l788_788065


namespace problem_statement_l788_788843

variable {a : ℝ} {f : ℝ → ℝ} {x1 x2 x3 : ℝ}

-- Given the function definition and conditions
def function_def (x : ℝ) : ℝ := abs (x + 1) * real.exp (-1 / x) - a

-- Condition that f(x) = 0 has exactly three roots
def has_exactly_three_roots (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0

-- Main problem: Prove that x2 - x1 < a under the given conditions
theorem problem_statement (h1 : a > 0) (h2 : has_exactly_three_roots a function_def) : x2 - x1 < a := 
  sorry

end problem_statement_l788_788843


namespace translated_point_B_coords_l788_788882

-- Define the initial point A
def point_A : ℝ × ℝ := (-2, 2)

-- Define the translation operations
def translate_down (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 - d)

def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 + d, p.2)

-- Define the translation of point A to point B
def point_B :=
  translate_right (translate_down point_A 4) 3

-- The proof statement
theorem translated_point_B_coords : point_B = (1, -2) :=
  by sorry

end translated_point_B_coords_l788_788882


namespace cyclic_quadrilateral_maximal_product_l788_788792

theorem cyclic_quadrilateral_maximal_product
  (O A B C D E : Type)
  [euclidean_geometry O]
  (c_circle : circle O ↔ ∃ A, ∃ B, diameter A B)
  (cC_on_circle : point_on_circle C c_circle)
  (D_not_AB : D ≠ A ∧ D ≠ B)
  (D_on_arc : ∃ arc, D ∈ arc ∧ arc ∉ (arc_not_containing C))
  (E_on_CD_perp : E_on_line_CD ∧ B_perp_CD :
  (CE : length(CE) = max (CE : 取‘D)) :
  (BOED_cyclic : ∃ O A B D, cyclic_quadrilateral BOED ↔ is_right_angle (angle BOD)) :
  maximal_product : CE * ED = max {CE * ED | E_on_CD_possible}) := sorry

end cyclic_quadrilateral_maximal_product_l788_788792


namespace acute_angle_at_3_25_l788_788199

noncomputable def angle_between_hour_and_minute_hands (hour minute : ℕ) : ℝ :=
  let hour_angle := (hour % 12) * 30 + (minute / 60 * 30)
  let minute_angle := (minute / 60 * 360)
  let diff := abs (minute_angle - hour_angle)
  if diff > 180 then 360 - diff else diff

theorem acute_angle_at_3_25 : angle_between_hour_and_minute_hands 3 25 = 47.5 :=
by 
  sorry

end acute_angle_at_3_25_l788_788199


namespace projection_matrix_3_4_l788_788329

theorem projection_matrix_3_4 :
  let v := λ α : Type, @vector α 2 := ![3, 4]
  let proj := λ x : vector ℝ 2, (v ℝ ⬝ x) / (v ℝ ⬝ v ℝ) • v ℝ
  proj = (λ x : vector ℝ 2, matrix.mul_vec ![
     ![9 / 25, 12 / 25],
     ![12 / 25, 16 / 25]
  ] x) :=
by sorry

end projection_matrix_3_4_l788_788329


namespace incorrect_propositions_l788_788253

/-- 
Among the following five propositions:
  ① If a ⟂ b, b ⟂ c, then a ⟂ c;  
  ② If a, b form equal angles with c, then a ∥ b;  
  ③ If a ∥ α, b ∥ α, then a ∥ b;  
  ④ If α ∩ β = l, a ⊆ α, b ⊆ β, then a, b are parallel or skew;  
  ⑤ If within plane α there are three points not on the same line that are equidistant from plane β, then α ∥ β;  
-/
theorem incorrect_propositions :
  (¬ ∀ (a b c : Type), (a ⟂ b) → (b ⟂ c) → (a ⟂ c)) ∧
  (¬ ∀ (a b c : Type), (angle a c = angle b c) → (a ∥ b)) ∧
  (¬ ∀ (a b α : Type), (a ∥ α) → (b ∥ α) → (a ∥ b)) ∧
  (¬ ∀ (α β a b : Type), (α ∩ β = l) → (a ⊆ α) → (b ⊆ β) → (a are_parallel_or_skew b)) ∧
  (¬ ∀ (α β : Type), (∃ (p1 p2 p3 : α), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ (distance p1 β = distance p2 β) ∧ (distance p2 β = distance p3 β)) → (α ∥ β)) :=
by sorry

end incorrect_propositions_l788_788253


namespace cyclic_A_D_Q_E_l788_788487

noncomputable def Point := ℝ × ℝ
noncomputable def Line (A B: Point) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

variables (A B C T S K H D E Q M : Point)

-- Conditions for the triangle and additional geometric constructions
axiom acute_angled_triangle (h1 : ∃ t : Line A B, ∃ s : Line A C, ∃ k : Line B C, (t.1 ≠ 0 → k.1*t.1 + k.2*t.2 > 0) ∧ (s.1 ≠ 0 → k.1*s.1 + k.2*s.2 > 0) ∧ t ≠ s ∧ s ≠ k ∧ t ≠ k)
axiom AB_neq_AC : A.1 ≠ B.1 ∨ A.2 ≠ B.2
axiom midpoint_M : M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
axiom BM_eq_CM (h1 : midpoint_M M) : (B.1 - M.1)^2 + (B.2 - M.2)^2 = (C.1 - M.1)^2 + (C.2 - M.2)^2 
axiom altitudes (h1 : T = (A.1, (B.2 + C.2) / 2)) (h2 : S = (B.1, (A.2 + C.2) / 2)) (h3 : K = (C.1, (A.2 + B.2) / 2)) 
axiom orthocenter: (Line B H) = (Line A S) ∧ (Line C H) = (Line A T)
axiom DE : (D.1 ≠ B.1 ∨ D.2 ≠ B.2) → (E.1 ≠ C.1 ∨ E.2 ≠ C.2) ∧ (Line A D) = (Line A E) 
axiom AE_eq_AD : (D.1 - A.1)^2 + (D.2 - A.2)^2 = (E.1 - A.1)^2 + (E.2 - A.2)^2
axiom angle_bisector_AQ : (Line A Q) = (Line A (.5 * (A.1 + B.1), .5 * (A.2 + C.2)))

theorem cyclic_A_D_Q_E :
  ∃ O : Point, (D.1 - O.1) ^ 2 + (D.2 - O.2) ^ 2 = (A.1 - O.1) ^ 2 + (A.2 - O.2) ^ 2 ∧ 
              (D.1 - O.1) ^ 2 + (D.2 - O.2) ^ 2 = (E.1 - O.1) ^ 2 + (E.2 - O.2) ^ 2 ∧ 
              (D.1 - O.1) ^ 2 + (D.2 - O.2) ^ 2 = (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 := 
BY sorry

end cyclic_A_D_Q_E_l788_788487


namespace parabola_equation_circle_m_equation_line_a2a3_tangent_to_circle_m_l788_788629

noncomputable theory
open_locale classical

-- Definitions and conditions
def parabola_vertex_origin (x y : ℝ) : Prop := ∃ p : ℝ, p > 0 ∧ y^2 = 2 * p * x
def line_intersects_parabola_perpendicularly : Prop :=
  ∃ p : ℝ, p = 1 / 2 ∧ parabola_vertex_origin 1 p

def circle_m_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1
def line_tangent_to_circle_m (l : ℝ → ℝ) : Prop := ∀ x y : ℝ, circle_m_eq x y → l x = y

def points_on_parabola_and_tangent (A1 A2 A3 : ℝ × ℝ) : Prop :=
  parabola_vertex_origin A1.1 A1.2 ∧
  parabola_vertex_origin A2.1 A2.2 ∧
  parabola_vertex_origin A3.1 A3.2 ∧
  line_tangent_to_circle_m (λ y, A1.1 * y + A1.2) ∧
  line_tangent_to_circle_m (λ y, A1.1 * y + A3.2)

-- Statements to prove
theorem parabola_equation : ∃ C : ℝ → ℝ → Prop, (C = parabola_vertex_origin) := sorry
theorem circle_m_equation : ∃ M : ℝ → ℝ → Prop, (M = circle_m_eq) := sorry
theorem line_a2a3_tangent_to_circle_m :
  ∀ A1 A2 A3 : ℝ × ℝ, 
  (points_on_parabola_and_tangent A1 A2 A3) →
  ∃ l : ℝ → ℝ, line_tangent_to_circle_m l := sorry

end parabola_equation_circle_m_equation_line_a2a3_tangent_to_circle_m_l788_788629


namespace triangle_obtuse_l788_788614

theorem triangle_obtuse (x : ℝ) (hx : x > 0) : 
  let a := 3 * x
  let b := 4 * x
  let c := 6 * x
  in a^2 + b^2 < c^2 := 
by 
  let a := 3 * x
  let b := 4 * x
  let c := 6 * x
  calc
    a^2 + b^2 = (3 * x)^2 + (4 * x)^2 : by rw [pow_two, pow_two]
           ... = 9 * x^2 + 16 * x^2 : by norm_num
           ... = 25 * x^2 : by ring
    c^2 = (6 * x)^2 : by rw pow_two
      ... = 36 * x^2 : by ring
    show 25 * x^2 < 36 * x^2, by linarith

end triangle_obtuse_l788_788614


namespace total_number_of_candles_l788_788903

theorem total_number_of_candles
  (candles_bedroom : ℕ)
  (candles_living_room : ℕ)
  (candles_donovan : ℕ)
  (h1 : candles_bedroom = 20)
  (h2 : candles_bedroom = 2 * candles_living_room)
  (h3 : candles_donovan = 20) :
  candles_bedroom + candles_living_room + candles_donovan = 50 :=
by
  sorry

end total_number_of_candles_l788_788903


namespace bacteria_growth_rate_l788_788898

-- Define the existence of the growth rate and the initial amount of bacteria
variable (B : ℕ → ℝ) (B0 : ℝ) (r : ℝ)

-- State the conditions from the problem
axiom bacteria_growth_model : ∀ t : ℕ, B t = B0 * r ^ t
axiom day_30_full : B 30 = B0 * r ^ 30
axiom day_26_sixteenth : B 26 = (1 / 16) * B 30

-- Theorem stating that the growth rate r of the bacteria each day is 2
theorem bacteria_growth_rate : r = 2 := by
  sorry

end bacteria_growth_rate_l788_788898


namespace find_x_l788_788156

-- Necessary definitions based on the conditions
def average (a b c : ℝ) : ℝ := (a + b + c) / 3

theorem find_x:
  let avg1 := average 20 40 60 in
  let avg2 (x : ℝ) := average 10 50 x in
  avg1 = 40 → (∀ x : ℝ, avg1 = avg2 x + 5 → x = 45) :=
by
  intros avg1 avg2 h₁ x h₂
  -- avg1 is defined as 40 from the problem condition
  subst avg1
  -- we now have 40 = avg2 x + 5
  sorry

end find_x_l788_788156


namespace pile_division_660_stones_l788_788971

theorem pile_division_660_stones (n : ℕ) (heaps : List ℕ) :
  n = 660 ∧ heaps.length = 30 ∧ ∀ x ∈ heaps, ∀ y ∈ heaps, (x ≤ 2 * y ∧ y ≤ 2 * x) →
  ∃ heaps : List ℕ, n = 660 ∧ heaps.length = 30 ∧ (∀ x y ∈ heaps, x ≤ 2 * y ∧ y ≤ 2 * x) :=
by
  sorry

end pile_division_660_stones_l788_788971


namespace projection_onto_vector_l788_788324

noncomputable def projection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  ![\[9 / 25, 12 / 25\], \[12 / 25, 16 / 25\]]

theorem projection_onto_vector:
    ∀ (x y : ℚ), (Matrix.mul_vec projection_matrix ![\x, \y]) = ![(9 * x + 12 * y) / 25, (12 * x + 16 * y) / 25] := by
  sorry

end projection_onto_vector_l788_788324


namespace highest_annual_income_is_stock_B_l788_788583

-- Define the conditions
def investment_amount := 6800
def stock_A_price := 136
def stock_A_dividend_rate := 0.10
def stock_B_price := 150
def stock_B_dividend_rate := 0.12
def stock_C_price := 100
def stock_C_dividend_rate := 0.08

-- Define the number of shares that can be bought of each stock
def shares_stock_A : Nat := investment_amount / stock_A_price
def shares_stock_B : Nat := investment_amount / stock_B_price
def shares_stock_C : Nat := investment_amount / stock_C_price

-- Define the total annual income from each stock
def annual_income_stock_A := stock_A_dividend_rate * investment_amount
def annual_income_stock_B := stock_B_dividend_rate * (shares_stock_B * stock_B_price)
def annual_income_stock_C := stock_C_dividend_rate * investment_amount

-- Prove that the stock yielding the highest annual income is Stock B
theorem highest_annual_income_is_stock_B :
  max annual_income_stock_A (max annual_income_stock_B annual_income_stock_C) = annual_income_stock_B :=
  by 
    -- placeholder for the actual proof
    sorry

end highest_annual_income_is_stock_B_l788_788583


namespace product_of_distances_l788_788027

-- Define the problem's conditions
def line (x y : ℝ) : Prop := x + y = 1
def parabola (x y : ℝ) : Prop := y = x^2
def M : ℝ × ℝ := (-1, 2)

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
  line x1 y1 ∧ parabola x1 y1 ∧ line x2 y2 ∧ parabola x2 y2 ∧
  A = (x1, y1) ∧ B = (x2, y2) ∧ A ≠ B

-- Define the distance function
def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

-- Define the proof problem
theorem product_of_distances : 
  ∀ A B : ℝ × ℝ, intersection_points A B → distance M A * distance M B = 2 :=
by
  intros
  sorry

end product_of_distances_l788_788027


namespace problem_l788_788017

noncomputable def f (x : ℝ) : ℝ := x^3 - x + 1

def tangent_line_at (p : ℝ × ℝ) (f : ℝ → ℝ) : ℝ → ℝ :=
  let df := deriv f in
  λ x, df p.1 * (x - p.1) + p.2

def triangle_area (a b : ℝ) : ℝ :=
  1 / 2 * a * b

theorem problem (p : ℝ × ℝ) (h : p = (0 : ℝ, 1)) :
  triangle_area 1 1 = 1 / 2 :=
by
  sorry

end problem_l788_788017


namespace length_of_bridge_is_205_l788_788216

-- Definition of the conditions
def train_length : ℝ := 170
def speed_kmh : ℝ := 45
def time_seconds : ℝ := 30

-- Conversion from km/hr to m/s
def speed_mps : ℝ := speed_kmh * 1000 / 3600

-- Calculation of total distance covered in 30 seconds
def total_distance : ℝ := speed_mps * time_seconds

-- Target: the length of the bridge
def bridge_length : ℝ := total_distance - train_length

-- The theorem statement
theorem length_of_bridge_is_205 : bridge_length = 205 :=
by sorry

end length_of_bridge_is_205_l788_788216


namespace tangent_line_equation_at_0_1_l788_788297

def f (x : ℝ) : ℝ := Real.exp x + 5 * Real.sin x

def f' (x : ℝ) : ℝ := Real.exp x + 5 * Real.cos x

theorem tangent_line_equation_at_0_1 : 
  let k := f' 0 in 
  k = 6 → 
  ∀ x y : ℝ, y = f 0 + k * x - k * 0 → y = 6 * x + 1 :=
by 
  intros k hk x y h
  simp [f, f'] at hk
  simp at h
  rw h
  exact hk

end tangent_line_equation_at_0_1_l788_788297


namespace divide_stones_l788_788966

/-- A pile of 660 stones can be divided into 30 piles where the sizes of the piles differ by less than a factor of 2. -/
theorem divide_stones (n : ℕ) (p : ℕ) (stones : ℕ) :
  stones = 660 → p = 30 →
  ∃ (heaps : Fin p → ℕ),
    (∑ i, heaps i = stones) ∧ (∀ i j, heaps i ≤ 2 * heaps j ∧ heaps j ≤ 2 * heaps i) :=
by
  intros h1 h2
  sorry

end divide_stones_l788_788966


namespace minimumJumpsToBlueIsFour_l788_788765

-- Define the grid as a type alias
def Grid := Array (Array Bool)

-- Define a function to simulate the jump effect given grid and coordinates
def jump (g : Grid) (x y : Nat) : Grid := sorry -- This function will implement the change logic according to the conditions.

-- Define the initial state of the grid - all red (e.g., all False)
def initialGrid : Grid := Array.repeat (Array.repeat false 4) 4

-- Define a function to check if the grid is entirely blue (e.g., all True)
def isBlue (g : Grid) : Bool := g.all (λ row => row.all (λ cell => cell))

-- A helper function to count the jumps (this function is our focus)
def minimumJumpsToBlue (initial : Grid) : Nat := sorry -- This would be a function that calculates the minimum jumps to blue grid.

-- The theorem statement specifying what needs to be proven
theorem minimumJumpsToBlueIsFour : minimumJumpsToBlue initialGrid = 4 := by
  sorry

end minimumJumpsToBlueIsFour_l788_788765


namespace pit_A_no_replant_exactly_one_pit_no_replant_at_least_one_replant_l788_788713

noncomputable def pit_a_no_replant_prob : ℝ := 0.875
noncomputable def one_pit_no_replant_prob : ℝ := 0.713
noncomputable def at_least_one_pit_replant_prob : ℝ := 0.330

theorem pit_A_no_replant (p : ℝ) (h1 : p = 0.5) : pit_a_no_replant_prob = 1 - (1 - p)^3 := by
  sorry

theorem exactly_one_pit_no_replant (p : ℝ) (h1 : p = 0.5) : one_pit_no_replant_prob = 1 - 3 * (1 - p)^3 * (p^3)^(2) := by
  sorry

theorem at_least_one_replant (p : ℝ) (h1 : p = 0.5) : at_least_one_pit_replant_prob = 1 - (1 - (1 - p)^3)^3 := by
  sorry

end pit_A_no_replant_exactly_one_pit_no_replant_at_least_one_replant_l788_788713


namespace piece_attacks_given_square_X_from_no_more_than_20_squares_place_20_pieces_no_threat_l788_788598

-- Definition and Assumptions
open Set

noncomputable def piece_attacks_at_most_20_squares (F : Type*) (board : set (ℕ × ℕ)) :=
  ∀ (sq : ℕ × ℕ), sq ∈ board → (∃ (attack_squares : set (ℕ × ℕ)), attack_squares ⊆ board ∧ attack_squares.size ≤ 20 ∧ sq ∈ attack_squares)

-- Problem (a)
theorem piece_attacks_given_square_X_from_no_more_than_20_squares (F : Type*) (board : set (ℕ × ℕ)) 
  (h_attack : piece_attacks_at_most_20_squares F board) (X : ℕ × ℕ) :
  ∃ (attackers : set (ℕ × ℕ)), attackers ⊆ board ∧ attackers.size ≤ 20 ∧ (∀ sq ∈ attackers, attacks F sq X) :=
sorry

-- Problem (b)
theorem place_20_pieces_no_threat (F : Type*) (pieces : fin 20 → F) (board : set (ℕ × ℕ))
  (h_attack : ∀ p, piece_attacks_at_most_20_squares (pieces p) board) : 
  ∃ (placement : fin 20 → ℕ × ℕ), (∀ i j, i ≠ j → ¬attacks (pieces i) (placement i) (placement j)) :=
sorry

end piece_attacks_given_square_X_from_no_more_than_20_squares_place_20_pieces_no_threat_l788_788598


namespace divide_stones_l788_788964

/-- A pile of 660 stones can be divided into 30 piles where the sizes of the piles differ by less than a factor of 2. -/
theorem divide_stones (n : ℕ) (p : ℕ) (stones : ℕ) :
  stones = 660 → p = 30 →
  ∃ (heaps : Fin p → ℕ),
    (∑ i, heaps i = stones) ∧ (∀ i j, heaps i ≤ 2 * heaps j ∧ heaps j ≤ 2 * heaps i) :=
by
  intros h1 h2
  sorry

end divide_stones_l788_788964


namespace train_journey_time_eq_l788_788245

variable (a b c : ℝ)

theorem train_journey_time_eq :
  (2 * a) / 30 + (3 * b) / 50 + (4 * c) / 70 = (140 * a + 126 * b + 120 * c) / 2100 := by
start
  -- sorry added to the proof step since the detailed proof is not required.
  sorry
end

end train_journey_time_eq_l788_788245


namespace cost_of_milk_l788_788616

-- Given conditions
def total_cost_of_groceries : ℕ := 42
def cost_of_bananas : ℕ := 12
def cost_of_bread : ℕ := 9
def cost_of_apples : ℕ := 14

-- Prove that the cost of milk is $7
theorem cost_of_milk : total_cost_of_groceries - (cost_of_bananas + cost_of_bread + cost_of_apples) = 7 := 
by 
  sorry

end cost_of_milk_l788_788616


namespace projection_of_u_l788_788240

-- Define the vectors
def v1 : ℝ × ℝ := (3, 3)
def v2 : ℝ × ℝ := (45 / 10, 15 / 10)
def u : ℝ × ℝ := (1, -1)

-- Define the projection function onto (3, 1)
def proj (x y : ℝ × ℝ) : ℝ × ℝ :=
  let k := (x.1 * y.1 + x.2 * y.2) / (y.1 * y.1 + y.2 * y.2)
  (k * y.1, k * y.2)

-- Define known result of projection
def proj_result : ℝ × ℝ := proj v1 (3, 1)

-- State the theorem to prove
theorem projection_of_u : proj u (3, 1) = (0.6, 0.2) :=
  by
  sorry

end projection_of_u_l788_788240


namespace digit_difference_l788_788591

open Nat

theorem digit_difference (x y : ℕ) (h₁ : 10 * x + y - (10 * y + x) = 81) (h₂ : Prime (x + y)) : x - y = 9 :=
by
  sorry

end digit_difference_l788_788591


namespace isosceles_triangle_height_l788_788257

theorem isosceles_triangle_height (s h : ℝ) (eq_areas : (2 * s * s) = (1/2 * s * h)) : h = 4 * s :=
by
  sorry

end isosceles_triangle_height_l788_788257


namespace chess_probability_l788_788131

theorem chess_probability (P_draw P_B_win : ℚ) (h_draw : P_draw = 1/2) (h_B_win : P_B_win = 1/3) :
  (1 - P_draw - P_B_win = 1/6) ∧ -- Statement A is correct
  (P_draw + (1 - P_draw - P_B_win) ≠ 1/2) ∧ -- Statement B is incorrect as it's not 1/2
  (1 - P_draw - P_B_win ≠ 2/3) ∧ -- Statement C is incorrect as it's not 2/3
  (P_draw + P_B_win ≠ 1/2) := -- Statement D is incorrect as it's not 1/2
by
  -- Insert proof here
  sorry

end chess_probability_l788_788131


namespace abs_linear_combination_l788_788295

theorem abs_linear_combination (a b : ℝ) :
  (∀ x y : ℝ, |a * x + b * y| + |b * x + a * y| = |x| + |y|) →
  (a = 1 ∧ b = 0) ∨ (a = 0 ∧ b = 1) ∨ (a = 0 ∧ b = -1) ∨ (a = -1 ∧ b = 0) :=
by {
  sorry
}

end abs_linear_combination_l788_788295


namespace alice_safe_paths_l788_788749

/-
Define the coordinate system and conditions.
-/

def total_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

def paths_through_dangerous_area : ℕ :=
  (total_paths 2 2) * (total_paths 2 1)

def safe_paths : ℕ :=
  total_paths 4 3 - paths_through_dangerous_area

theorem alice_safe_paths : safe_paths = 17 := by
  sorry

end alice_safe_paths_l788_788749


namespace constant_term_of_expansion_l788_788832

noncomputable def sum_of_coefficients : ℕ := 96
noncomputable def constant_term := 15

theorem constant_term_of_expansion :
  (∑ n in Finset.range 6, binomial 5 n * (1 ^ (5 - 2 * n) + (1 + 1 + a * 1 ^ 3) * 1 ^ (5 - 2 * n))) = sum_of_coefficients →
  let a := 1 in
  let T := ∑ n in Finset.range 6, binomial 5 n * (1 ^ (5 - 2 * n) + (1 + 1 + a * 1 ^ 3) * 1 ^ (5 - 2 * n)) in
  T = constant_term := 
sorry

end constant_term_of_expansion_l788_788832


namespace count_valid_a1_l788_788535

def satisfies_condition (a1 : ℕ) : Prop :=
  let a2 := if a1 % 2 = 0 then a1 / 2 else 3 * a1 + 1 in
  let a3 := if a2 % 2 = 0 then a2 / 2 else 3 * a2 + 1 in
  let a4 := if a3 % 2 = 0 then a3 / 2 else 3 * a3 + 1 in
  a1 < a2 ∧ a1 < a3 ∧ a1 < a4

theorem count_valid_a1 : (Finset.range 2501).filter (λ a1 => ∃ k, a1 = 4 * k + 3 ∧ satisfies_condition a1).card = 625 :=
  sorry

end count_valid_a1_l788_788535


namespace area_of_square_l788_788062

theorem area_of_square (ABCD : Type) [linear_ordered_comm_ring ABCD] 
  (A B C D F E : ABCD) 
  (midpoint_AD : midpoint A D = F) 
  (midpoint_CD : midpoint C D = E) 
  (area_FED : triangle_area F E D = 2) 
  (interior_angle_D : interior_angle F E D = 30) 
  : area_sq ABCD = 16 := 
sorry

end area_of_square_l788_788062


namespace correct_propositions_l788_788939

variables (m n : Line) (α β γ : Plane)
variables (h_diff_lines : m ≠ n) (h_diff_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)

-- Proposition 1: (α ∥ β ∧ α ∥ γ) → β ∥ γ
def Proposition1 : Prop := (α ∥ β ∧ α ∥ γ) → β ∥ γ

-- Proposition 2: (α ⟂ β ∧ m ∥ α) → m ⟂ β
def Proposition2 : Prop := (α ⟂ β ∧ m ∥ α) → m ⟂ β

-- Proposition 3: (m ⟂ α ∧ m ∥ β) → α ⟂ β
def Proposition3 : Prop := (m ⟂ α ∧ m ∥ β) → α ⟂ β

-- Proposition 4: (m ∥ n ∧ n ⊆ α) → m ∥ α
def Proposition4 : Prop := (m ∥ n ∧ n ⊆ α) → m ∥ α

theorem correct_propositions :
  (Proposition1 α β γ ∧ Proposition3 m α β) ∧ ¬(Proposition2 m α β) ∧ ¬(Proposition4 m n α) := 
begin
  sorry
end

end correct_propositions_l788_788939


namespace preceding_integer_l788_788462

def bin_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc bit => 2 * acc + if bit then 1 else 0) 0

theorem preceding_integer : bin_to_nat [true, true, false, false, false] - 1 = bin_to_nat [true, false, true, true, true] := by
  sorry

end preceding_integer_l788_788462


namespace cosine_sine_inequality_theorem_l788_788419

theorem cosine_sine_inequality_theorem (θ : ℝ) :
  (∀ x : ℝ, 0 ≤ x → x ≤ 1 → 
    x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) ↔
    (π / 12 < θ ∧ θ < 5 * π / 12) :=
by
  sorry

end cosine_sine_inequality_theorem_l788_788419


namespace find_fx_for_negative_interval_l788_788092

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def periodic_function (f : ℝ → ℝ) (P : ℝ) : Prop :=
  ∀ x : ℝ, f (x + P) = f x

def function_on_interval (f : ℝ → ℝ) (I : Set ℝ) (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ∈ I → f x = g x

theorem find_fx_for_negative_interval (f : ℝ → ℝ) :
  even_function f →
  periodic_function f 2 →
  function_on_interval f (Set.Icc 2 3) (id) →
  function_on_interval f (Set.Icc (-2) 0) (λ x, 3 - |x + 1|) :=
by
  intros h_even h_periodic h_interval
  sorry

end find_fx_for_negative_interval_l788_788092


namespace projection_onto_3_4_matrix_l788_788340

def projection_matrix := λ (u : ℝ) (v : ℝ), (3 * u + 4 * v) / 25

theorem projection_onto_3_4_matrix :
  ∀ (x y : ℝ),
  (λ (u v : ℝ), (3 * x + 4 * y) / 25) = (λ (u v : ℝ), (\(u * 9 / 25) + (v * 12 / 25), (u * 12 / 25) + (v * 16 / 25))) :=
by
  sorry

end projection_onto_3_4_matrix_l788_788340


namespace flyers_left_l788_788514

theorem flyers_left (total_flyers : ℕ) (jack_flyers : ℕ) (rose_flyers : ℕ) :
  total_flyers = 1236 → jack_flyers = 120 → rose_flyers = 320 → total_flyers - (jack_flyers + rose_flyers) = 796 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact eq.refl _

end flyers_left_l788_788514


namespace airline_cities_connectivity_l788_788054

theorem airline_cities_connectivity:
  ∀ (cities : ℕ) (airlines : ℕ) (connections : ℕ → ℕ → ℕ),
  cities = 800 →
  airlines = 8 →
  (∀ i j : ℕ, 1 ≤ i → i ≤ cities → 1 ≤ j → j ≤ cities → 1 ≤ connections i j ∧ connections i j ≤ airlines) →
  ¬(∃ airline : ℕ, 1 ≤ airline ∧ airline ≤ airlines ∧
    ∃ subset : list ℕ, subset.length > 200 ∧
    (∀ (x y : ℕ), x ∈ subset → y ∈ subset → 
     (x = y) ∨ connections x y = airline ∨ (∃ path : list ℕ, path.head = x ∧ path.last = some y ∧ ∀ p ∈ path, 1 ≤ connections p p ∧ connections p p ≤ airlines))) :=
by {
  intro cities airlines connections hcities hairlines hconnect,
  sorry
}

end airline_cities_connectivity_l788_788054


namespace speed_of_current_is_1_75_l788_788227

-- Define the upstream and downstream travel times and distances
def distance : ℝ := 1 -- in kilometers

def upstream_time : ℝ := 40 / 60 -- 40 minutes converted to hours
def downstream_time : ℝ := 12 / 60 -- 12 minutes converted to hours

-- Define the upstream and downstream speeds
def upstream_speed : ℝ := distance / upstream_time
def downstream_speed : ℝ := distance / downstream_time

-- The speed of the current to be proved
def speed_of_current : ℝ := (downstream_speed - upstream_speed) / 2

theorem speed_of_current_is_1_75 :
  speed_of_current = 1.75 := by
  -- sorry is used here to indicate we are skipping the proof
  sorry

end speed_of_current_is_1_75_l788_788227


namespace find_number_l788_788041

theorem find_number (x : ℤ) (h : 5 * x - 28 = 232) : x = 52 :=
by
  sorry

end find_number_l788_788041


namespace polar_eq_line_l1_polar_eq_curve_C_area_triangle_CMN_intersection_l1_l2_l788_788884

-- Definitions for the problem conditions.
def line_l1 (ρ θ : ℝ) : Prop := ρ * Real.cos θ + 2 = 0

def curve_C (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

def line_l2 (θ : ℝ) : Prop := θ = π / 4

-- Question and required mathematical proofs
theorem polar_eq_line_l1 : 
  ∀ θ, ∃ ρ, line_l1 ρ θ :=
by sorry

theorem polar_eq_curve_C : 
  ∀ θ, ∃ ρ, curve_C ρ θ :=
by sorry

theorem area_triangle_CMN 
  (θ : ℝ) (hθ : line_l2 θ) 
  (ρ1 ρ2 : ℝ) (hρ1 : curve_C ρ1 θ) (hρ2 : curve_C ρ2 θ) :
  2 :=
by sorry

theorem intersection_l1_l2 : 
  ∃ ρ, line_l1 ρ (π / 4) ∧ line_l2 (π / 4) :=
by sorry

end polar_eq_line_l1_polar_eq_curve_C_area_triangle_CMN_intersection_l1_l2_l788_788884


namespace stones_partition_l788_788994

theorem stones_partition (total_stones : ℕ) (piles : ℕ) (heaps : ℕ → ℕ) 
  (h_total : total_stones = 660)
  (h_piles : piles = 30)
  (h_sum_heaps : ∑ i in range piles, heaps i = 660)
  (h_factor : ∀ i j, i < piles → j < piles → heaps i ≤ 2 * heaps j) :
  ∃ heaps : Π i : ℕ, i < piles → ℕ,
    (∑ i in range piles, heaps i = 660) ∧
    (∀ i j, i < piles → j < piles → heaps i ≤ 2 * heaps j) :=
  sorry

end stones_partition_l788_788994


namespace AC_in_right_triangle_theorem_l788_788052

noncomputable def AC_in_right_triangle : Prop :=
  ∀ (A B C : Type) [InnerProductSpace ℝ A],
  (angle_eq A B C 90) → 
  (AB = 10) → 
  (BC = 8) → 
  (AC = 6)

theorem AC_in_right_triangle_theorem : AC_in_right_triangle :=
by
  sorry

end AC_in_right_triangle_theorem_l788_788052


namespace max_points_no_three_collinear_not_obtuse_l788_788772

def no_three_collinear (points : List (EuclideanSpace ℝ 3)) : Prop :=
  ∀ (P1 P2 P3 : EuclideanSpace ℝ 3), 
    P1 ≠ P2 → P2 ≠ P3 → P1 ≠ P3 → 
    (P1, P2) ∈ points.choice → (P2, P3) ∈ points.choice → (P1, P3) ∈ points.choice → 
    ¬(P1, P2, P3 : AffineSpan ℝ).

def not_obtuse_triangle (P1 P2 P3 : EuclideanSpace ℝ 3) : Prop :=
  ∀ (θ1 θ2 θ3 : ℝ), 
    angle P1 P2 P3 θ1 ∧ angle P2 P3 P1 θ2 ∧ angle P3 P1 P2 θ3 → 
    θ1 < π/2 ∧ θ2 < π/2 ∧ θ3 < π/2

theorem max_points_no_three_collinear_not_obtuse : 
  ∃ (n : ℕ), 
  (∀ (points : List (EuclideanSpace ℝ 3)), points.length = n → no_three_collinear points → 
  (∀ (i j k : Fin n), 1 ≤ i.1 ∧ i.1 < j.1 ∧ j.1 < k.1 ≤ n → not_obtuse_triangle (points[i.1]) (points[j.1]) (points[k.1]))) 
  ∧ (∀ (m : ℕ), m > n → ¬(∃ (points : List (EuclideanSpace ℝ 3)), points.length = m ∧ no_three_collinear points ∧ 
  (∀ (i j k : Fin m), 1 ≤ i.1 ∧ i.1 < j.1 ∧ j.1 < k.1 ≤ m → not_obtuse_triangle (points[i.1]) (points[j.1]) (points[k.1])))) :=
  sorry

end max_points_no_three_collinear_not_obtuse_l788_788772


namespace problem_l788_788067

noncomputable def a : ℕ → ℝ
| 1 => 1 / 3
| n => sorry  -- This will be used to define the rest of the sequence based on the conditions.

def S (n : ℕ) : ℝ := n * (2 * n - 1) * a n

theorem problem (n : ℕ) : 
  (a 2 = 1 / 15) ∧ (a 3 = 1 / 35) ∧ (a 4 = 1 / 63) ∧ 
  (∀ n, a n = 1 / ((2 * n - 1) * (2 * n + 1))) :=
by
  sorry

end problem_l788_788067


namespace divide_660_stones_into_30_heaps_l788_788956

theorem divide_660_stones_into_30_heaps :
    ∃ (heaps : Fin 30 → ℕ), (∑ i, heaps i = 660) ∧ (∀ i j, heaps i < 2 * heaps j) ∨ (heaps j < 2 * heaps i) := 
sorry

end divide_660_stones_into_30_heaps_l788_788956


namespace angle_in_second_quadrant_l788_788708

-- Define the problem statement
theorem angle_in_second_quadrant (θ : ℝ) 
  (θ_eq : θ = 29 * real.pi / 6) : 
  θ = 29 * real.pi / 6 → (π < θ ∧ θ < 2 * π) :=
begin
  intros,
  sorry
end

end angle_in_second_quadrant_l788_788708


namespace angle_BPC_ninety_l788_788910

-- Given definitions from the conditions
variables {A B C M N P : Point}
variable (ABC : Triangle A B C)
variable {incircle : Circle}
variable {incenter : Point}
variable {angle_bisector_P : Line}
variable (M_touch : incircle.Touches M AB)
variable (N_touch : incircle.Touches N AC)
variable (P_on_MN : P ∈ Line.mk M N)
variable (P_on_angle_bisector : P ∈ angle_bisector_P)

theorem angle_BPC_ninety (h : angle_bisector_P ∈ Triangle.angleBisectorAt ABC B) : 
  ∠ B P C = 90° :=
sorry

end angle_BPC_ninety_l788_788910


namespace divide_660_stones_into_30_piles_l788_788989

theorem divide_660_stones_into_30_piles :
  ∃ (heaps : Fin 30 → ℕ),
    (∑ i, heaps i = 660) ∧
    (∀ i j, heaps i < 2 * heaps j ∨ heaps j < 2 * heaps i) :=
sorry

end divide_660_stones_into_30_piles_l788_788989


namespace smallest_value_abs_w3_plus_z3_l788_788389

theorem smallest_value_abs_w3_plus_z3 (w z : ℂ) 
  (h1: complex.abs (w + z) = 2) 
  (h2: complex.abs (w^2 + z^2) = 8) : 
  complex.abs (w^3 + z^3) = 20 :=
sorry

end smallest_value_abs_w3_plus_z3_l788_788389


namespace students_before_new_year_le_197_l788_788704

variable (N M k ℓ : ℕ)

-- Conditions
axiom condition_1 : M = (k * N) / 100
axiom condition_2 : 100 * M = k * N
axiom condition_3 : 100 * (M + 1) = ℓ * (N + 3)
axiom condition_4 : ℓ < 100

-- The theorem to prove
theorem students_before_new_year_le_197 :
  N ≤ 197 :=
by
  sorry

end students_before_new_year_le_197_l788_788704


namespace sequence_difference_l788_788175

theorem sequence_difference {a : ℕ → ℕ} (h : ∀ n, (a (n + 1) = if even (a n) then a n / 2 else 3 * (a n) + 1) ∧ a 7 = 2) :
  let S := 254 in let T := 190 in (S - T = 64) :=
by
  sorry

end sequence_difference_l788_788175


namespace analysis_method_proves_sufficient_condition_l788_788753

-- Definitions and conditions from part (a)
def analysis_method_traces_cause_from_effect : Prop := true
def analysis_method_seeks_sufficient_conditions : Prop := true
def analysis_method_finds_conditions_for_inequality : Prop := true

-- The statement to be proven
theorem analysis_method_proves_sufficient_condition :
  analysis_method_finds_conditions_for_inequality →
  analysis_method_traces_cause_from_effect →
  analysis_method_seeks_sufficient_conditions →
  (B = "Sufficient condition") :=
by 
  sorry

end analysis_method_proves_sufficient_condition_l788_788753


namespace inclination_angle_of_line_l788_788669

theorem inclination_angle_of_line (m : ℝ) (b : ℝ) (h : b = -3) (h_line : ∀ x : ℝ, x - 3 = m * x + b) : 
  (Real.arctan m * 180 / Real.pi) = 45 := 
by sorry

end inclination_angle_of_line_l788_788669


namespace problem_l788_788929

def floor (z : ℝ) : ℤ := Int.floor z

theorem problem :
  ∀ u : ℝ, ∀ v : ℝ, v = 4 * (floor u) + 5 → v = 5 * (floor (u - 3)) + 9 → (u ∈ set.Ioo 11 12) →
  (u + v ∈ set.Ioo 60 61) :=
by
  intros u v h₁ h₂ hu
  sorry

end problem_l788_788929


namespace period_of_sine_function_l788_788024

theorem period_of_sine_function {t : ℝ} (h : 0 ≠ t)
  (h1 : ∀ x : ℝ, sin (π * (x + t) + φ) = sin (π * x + φ)) : t = 2 :=
sorry

end period_of_sine_function_l788_788024


namespace placement_of_6_5_l788_788184

-- Define the problem context and conditions
def slip_numbers := [1, 1.5, 2, 2.5, 3, 3.5, 4, 4, 4.5, 5, 5, 5.5, 6, 6.5, 7, 7.5]
def cups := {A, B, C, D}
def even_integers := {n : ℤ | n % 2 = 0} -- Set of even integers
def sum_is_even (s : Set ℝ) := (finset.sum (multiset.to_finset (list.to_multiset s))).val % 2 = 0
def consecutive_even_sums (s : ℕ → ℤ) := ∀ n, s (n + 1) = s n + 2
def sum_of_slips := list.sum slip_numbers = 68
def slip5_in_D := 5
def slip4_in_B := 4

-- Define the correct answer constraint
def correct_answer (cup_with_6_5 : char) := cup_with_6_5 = 'C'

-- The proof problem statement
theorem placement_of_6_5 :
  slip_numbers ∈ cups ∧
  sum_is_even slip_numbers ∧
  consecutive_even_sums (λ n, if n = 'A' then 12 else if n = 'B' then 14 else if n = 'C' then 16 else 18) ∧
  slip5_in_D ∧
  slip4_in_B →
  correct_answer 'C' :=
sorry

end placement_of_6_5_l788_788184


namespace john_ultramarathon_distance_l788_788076

theorem john_ultramarathon_distance :
  let initial_time := 8
  let time_increase_percentage := 0.75
  let speed_increase := 4
  let initial_speed := 8
  initial_time * (1 + time_increase_percentage) * (initial_speed + speed_increase) = 168 :=
by
  let initial_time := 8
  let time_increase_percentage := 0.75
  let speed_increase := 4
  let initial_speed := 8
  sorry

end john_ultramarathon_distance_l788_788076


namespace john_run_distance_l788_788079

theorem john_run_distance :
  ∀ (initial_hours : ℝ) (increase_time_percent : ℝ) (initial_speed : ℝ) (increase_speed : ℝ),
  initial_hours = 8 → increase_time_percent = 0.75 → initial_speed = 8 → increase_speed = 4 →
  let increased_hours := initial_hours * increase_time_percent,
      total_hours := initial_hours + increased_hours,
      new_speed := initial_speed + increase_speed,
      distance := total_hours * new_speed in
  distance = 168 := 
by
  intros initial_hours increase_time_percent initial_speed increase_speed h_hours h_time h_speed h_increase
  let increased_hours := initial_hours * increase_time_percent
  let total_hours := initial_hours + increased_hours
  let new_speed := initial_speed + increase_speed
  let distance := total_hours * new_speed
  sorry

end john_run_distance_l788_788079


namespace max_sequence_length_l788_788441

theorem max_sequence_length (b : ℕ → ℤ) (y : ℤ) :
  (b 1 = 5000) →
  (b 2 = y) →
  (∀ n : ℕ, n ≥ 2 → b (n + 1) = b (n - 1) - b n) →
  (∀ n : ℕ, b n < 20000) →
  0 < y →
  y = 3333 :=
begin
  sorry
end

end max_sequence_length_l788_788441


namespace not_divisible_by_n_only_prime_3_l788_788224

-- Problem 1: Prove that for any natural number \( n \) greater than 1, \( 2^n - 1 \) is not divisible by \( n \)
theorem not_divisible_by_n (n : ℕ) (h1 : 1 < n) : ¬ (n ∣ (2^n - 1)) :=
sorry

-- Problem 2: Prove that the only prime number \( n \) such that \( 2^n + 1 \) is divisible by \( n^2 \) is \( n = 3 \)
theorem only_prime_3 (n : ℕ) (hn : Nat.Prime n) (hdiv : n^2 ∣ (2^n + 1)) : n = 3 :=
sorry

end not_divisible_by_n_only_prime_3_l788_788224


namespace min_omega_value_l788_788410

theorem min_omega_value (ω : ℝ) (hω : ω > 0)
  (h_shift : ∀ x, sin (ω * (x - 4 * π / 3) + π / 3) + 2 = sin (ω * x + π / 3) + 2) :
  ω = 3 / 2 :=
by
  sorry

end min_omega_value_l788_788410


namespace volume_of_tetrahedron_l788_788058

def tetrahedron_volume (AB CD : ℝ) (dist : ℝ) (angle : ℝ) := 
  ∃ V : ℝ, V = 1/2 ∧ 
    AB = 1 ∧ 
    CD = sqrt 3 ∧ 
    dist = 2 ∧ 
    angle = real.pi / 3

theorem volume_of_tetrahedron : tetrahedron_volume 1 (sqrt 3) 2 (real.pi / 3) :=
  sorry

end volume_of_tetrahedron_l788_788058


namespace sum_of_distances_to_orthocenter_leq_twice_largest_altitude_l788_788524

noncomputable def acute_triangle (ABC : Type) [triangle ABC] : Prop :=
  is_acute_triangle ABC

noncomputable def orthocenter (ABC : Type) [triangle ABC] : Point ABC :=
  classical.ortho_center ABC

noncomputable def h_max (ABC : Type) [triangle ABC] : ℝ :=
  largest_altitude ABC

theorem sum_of_distances_to_orthocenter_leq_twice_largest_altitude
  (ABC : Type) [triangle ABC] (H : Point ABC) (h_max : ℝ)
  (H_ortho : H = orthocenter ABC) (acute_ABC : acute_triangle ABC) :
  altitude AH + altitude BH + altitude CH ≤ 2 * h_max := 
sorry

end sum_of_distances_to_orthocenter_leq_twice_largest_altitude_l788_788524


namespace find_k_l788_788594

theorem find_k 
  (A B X Y : ℝ × ℝ)
  (hA : A = (-3, 0))
  (hB : B = (0, -3))
  (hX : X = (0, 9))
  (Yx : Y.1 = 15)
  (hXY_parallel : (Y.2 - X.2) / (Y.1 - X.1) = (B.2 - A.2) / (B.1 - A.1)) :
  Y.2 = -6 := by
  -- proofs are omitted as per the requirements
  sorry

end find_k_l788_788594


namespace pentadecagon_diagonals_l788_788360

def number_of_diagonals (n : Nat) : Nat :=
  (n * (n - 3)) / 2

theorem pentadecagon_diagonals :
  number_of_diagonals 15 = 90 := 
by
  sorry

end pentadecagon_diagonals_l788_788360


namespace inequality_sum_geometric_series_l788_788541

-- Lean 4 statement for the given problem
theorem inequality_sum_geometric_series
  {n : ℕ} (hn : 0 < n) 
  {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy: x^n + y^n = 1) :
  (∑ k in Finset.range n, (1 + x^(2 * (k + 1))) / (1 + x^(4 * (k + 1))))
  * (∑ k in Finset.range n, (1 + y^(2 * (k + 1))) / (1 + y^(4 * (k + 1)))) < 
  1 / ((1 - x) * (1 - y)) :=
sorry

end inequality_sum_geometric_series_l788_788541


namespace induction_step_product_l788_788663

/-- Induction hypothesis: we assume that (k+1)*(k+2)*...*(2k) = 2^k * 1 * 3 * ... * (2k-1) -/
theorem induction_step_product (k : ℕ) (h : (k+1) * (k+2) * ... * (2*k) = 2^k * ∏ i in range k, (2*i + 1)) 
: (k+2) * (k+3) * ... * (2*k + 1) * (2*k + 2) = 2^(k+1) * ∏ i in range (k + 1), (2*i + 1) :=
sorry

end induction_step_product_l788_788663


namespace midline_double_l788_788282

-- Let point M be the midpoint of BC
def is_midpoint (M B C : Point) : Prop := dist B M = dist C M

-- Let points N and P be the midpoints of AC and AB respectively
def is_midpoint_AC (N A C : Point) : Prop := dist A N = dist C N
def is_midpoint_AB (P A B : Point) : Prop := dist A P = dist B P

-- Extend AM to a point D such that MD = AM; hence AD = 2AM
def extends_to (A M D : Point) : Prop := dist A D = 2 * dist A M

-- Extend BC to points E and F such that EB = BC and CF = BC
def extends_bc (B C E F : Point) : Prop := dist B E = dist B C ∧ dist C F = dist B C

-- Define the problem statement
theorem midline_double (A B C M N P D E F : Point) :
  is_midpoint M B C →
  is_midpoint_AC N A C →
  is_midpoint_AB P A B →
  extends_to A M D →
  extends_bc B C E F →
  dist A D = 2 * dist A M ∧
  dist A E = 2 * dist B N ∧
  dist A F = 2 * dist P C :=
sorry

end midline_double_l788_788282


namespace trapezoid_UW_eq_RT_l788_788590

-- Define trapezoid and properties
variables {R S Q T A U W : ℝ}
variables (RS QT RT : ℝ)
variables (angle_R_is_right : ∠ R = π / 2)
variables (RS_longer_QT : RS > QT)
variables (diagonals_intersect_right_angle : ∠ (diagonal RS QT) = π / 2)
variables (bisector_intersects_RT_at_U : ∃ U, bisector ∠RAT ∩ RT = U)
variables (parallel_line_U_RS_W : ∃ W, line_through_U || RS ∩ SQ = W)

-- Required proposition
theorem trapezoid_UW_eq_RT
  (h₁ : angle_R_is_right)
  (h₂ : RS_longer_QT)
  (h₃ : diagonals_intersect_right_angle)
  (h₄ : bisector_intersects_RT_at_U)
  (h₅ : parallel_line_U_RS_W)
  : distance U W = RT := 
sorry

end trapezoid_UW_eq_RT_l788_788590


namespace number_of_possible_values_of_a_is_520_l788_788132

noncomputable def count_possible_values_of_a : ℕ :=
  let a_b_c_d_values := 
    {p : list ℕ | p.length = 4 ∧ 
                  p.sorted (>) ∧ 
                  p.sum = 2080 ∧ 
                  (p.head! ^ 2 - p.tail.head! ^ 2 + p.tail.tail.head! ^ 2 - p.tail.tail.tail.head! ^ 2) = 2040} in
  (a_b_c_d_values.map (λ p, p.head!)).nodup.length

theorem number_of_possible_values_of_a_is_520 : 
  count_possible_values_of_a = 520 := 
by sorry

end number_of_possible_values_of_a_is_520_l788_788132


namespace flyers_left_l788_788513

theorem flyers_left (total_flyers : ℕ) (jack_flyers : ℕ) (rose_flyers : ℕ) :
  total_flyers = 1236 → jack_flyers = 120 → rose_flyers = 320 → total_flyers - (jack_flyers + rose_flyers) = 796 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact eq.refl _

end flyers_left_l788_788513


namespace binom_sum_identity_l788_788577

-- Defining the floor function for easier reference.
def floor (x : ℝ) : ℤ := Int.floor x

-- Defining binomial coefficient.
def binom (n r : ℕ) : ℕ := Nat.choose n r

theorem binom_sum_identity (n : ℕ) (hn : 0 < n) :
  (∑ r in Finset.range ((floor ((n - 1 : ℕ) / 2) + 1).to_nat), 
    (((n - 2 * r : ℕ) / n) * (binom n r))^2) = 
  (1 / n) * binom (2 * n - 2) (n - 1) := 
sorry

end binom_sum_identity_l788_788577


namespace max_volume_pyramid_l788_788560

theorem max_volume_pyramid (n : ℕ) (S : ℝ) :
  ∃(r h : ℝ), 
  (∀(V : ℝ), V = (n / 3) * tan (π / n) * r^2 * h) ∧ 
  (S = n * tan (π / n) * (r^2 + r * sqrt(h^2 + r^2))) →
  V is max when dihedral_angle_base_edge = dihedral_angle_regular_tetrahedron 
:= 
sorry

end max_volume_pyramid_l788_788560


namespace fence_poles_placement_l788_788232

def total_bridges_length (bridges : List ℕ) : ℕ :=
  bridges.sum

def effective_path_length (path_length : ℕ) (bridges_length : ℕ) : ℕ :=
  path_length - bridges_length

def poles_on_one_side (effective_length : ℕ) (interval : ℕ) : ℕ :=
  effective_length / interval

def total_poles (path_length : ℕ) (interval : ℕ) (bridges : List ℕ) : ℕ :=
  let bridges_length := total_bridges_length bridges
  let effective_length := effective_path_length path_length bridges_length
  let poles_one_side := poles_on_one_side effective_length interval
  2 * poles_one_side + 2

theorem fence_poles_placement :
  total_poles 2300 8 [48, 58, 62] = 534 := by
  sorry

end fence_poles_placement_l788_788232


namespace circle_and_line_equations_l788_788805

theorem circle_and_line_equations :
  ∀ (C : Set (ℝ × ℝ)) (l m : Set (ℝ × ℝ)),
  (∃ A B : ℝ × ℝ, A = (0, 2) ∧ B = (2, -2) ∧
  (∀ p, p ∈ C ↔ (p.1 + 3)^2 + (p.2 + 2)^2 = 25)) ∧
  (∃ center : ℝ × ℝ, center ∈ l ∧ l = {p : ℝ × ℝ | p.1 - p.2 + 1 = 0 }) ∧
  (∃ (p : ℝ × ℝ) (d : ℝ), p = (1, 4) ∧ 2 * d = 6 ∧ 
  (∀ radio : ℝ, radio = √(25 - 16)) ∧
  (p ∈ m ∧ (m = {p : ℝ × ℝ | p.1 = 1} ∨ m = {p : ℝ × ℝ | (5/12)*p.1 - p.2 + 43/12 = 0}))) :=
begin
  sorry
end

end circle_and_line_equations_l788_788805


namespace production_line_B_l788_788727

noncomputable def total_units : ℕ := 5000
noncomputable def ratio_A : ℕ := 1
noncomputable def ratio_B : ℕ := 2
noncomputable def ratio_C : ℕ := 2

def total_ratio : ℕ := ratio_A + ratio_B + ratio_C
noncomputable def units_B : ℕ := total_units * ratio_B / total_ratio

theorem production_line_B:
  units_B = 2000 :=
sorry

end production_line_B_l788_788727


namespace find_f_minus_3_l788_788091

def rational_function (f : ℚ → ℚ) : Prop :=
  ∀ x : ℚ, x ≠ 0 → 4 * f (1 / x) + (3 * f x / x) = 2 * x^2

theorem find_f_minus_3 (f : ℚ → ℚ) (h : rational_function f) : 
  f (-3) = 494 / 117 :=
by
  sorry

end find_f_minus_3_l788_788091


namespace value_of_b_minus_a_l788_788418

def f (x : ℝ) : ℝ := 2 * Real.sin x

theorem value_of_b_minus_a (a b : ℝ) 
  (h1 : ∀ x, a ≤ x ∧ x ≤ b → -1 ≤ f x ∧ f x ≤ 2) 
  (h2 : b - a = 5 * Real.pi / 3) : False :=
sorry

end value_of_b_minus_a_l788_788418


namespace math_problem_l788_788791

noncomputable def root_cube_64 : ℝ := 64^(1/3)
noncomputable def root_8 : ℝ := 8^(1/2)

theorem math_problem : (root_cube_64 - root_8)^2 = 24 - 16 * real.sqrt 2 := 
by 
  let a := (4 : ℝ)
  let b := (2 * real.sqrt 2 : ℝ)
  have h1 : root_cube_64 = a := 
    by 
      have : (64 : ℝ) = (4 * 4 * 4 * 4 * 4 * 4) := 
        by
          norm_num
      rw [real.rpow_nat_cast (4 : ℝ) (6 : ℝ)]
      norm_num at this
  have h2 : root_8 = b :=
    by 
      have : (8 : ℝ) = (2 * 2 * 2) := 
        by 
          norm_num
      exact real.sqrt_two (8 : ℝ) (real.sqrt 8 = 2 * real.sqrt 2)
  have : (a - b)^2 = (4 - 2 * real.sqrt 2)^2 :=
    by 
      exact congr_arg (λ x, x) (h1.symm) h2.symm
  rw [this]
  norm_num
  sorry

end math_problem_l788_788791


namespace volume_of_orthocentric_tetrahedron_l788_788894

-- Definition of an orthocentric tetrahedron
structure OrthocentricTetrahedron (A B C D : Type) : Prop :=
  (is_tetrahedron : True) -- Placeholder for proper tetrahedron definition
  (altitudes_concurrent : True) -- Placeholder for the concurrency of altitudes

-- Define the areas of faces
variables {A B C D : Type}
variables (S1 S2 S3 : ℝ)

-- Define the right angle condition at vertex B
def right_angle_at_B (A B C D : Type) : Prop :=
  ∠(A, B, C) = 90  -- Placeholder for actual angle definition

-- Define the volume of tetrahedron in terms of face areas
theorem volume_of_orthocentric_tetrahedron
  (h : OrthocentricTetrahedron A B C D)
  (right_angle : right_angle_at_B A B C D)
  (S1_pos : 0 < S1) (S2_pos : 0 < S2) (S3_pos : 0 < S3) :
  volume A B C D = (1/3) * sqrt (2 * S1 * S2 * S3) :=
  sorry

end volume_of_orthocentric_tetrahedron_l788_788894


namespace sum_of_series_base6_l788_788376

theorem sum_of_series_base6 :
  let series_sum_base6 (start end step : ℕ) (base : ℕ) : ℕ :=
    let num_terms := (end - start) / step + 1 in
    (num_terms * (start + end)) / 2
  in series_sum_base6 2 50 2 6 = 1040 :=
by
sorry

end sum_of_series_base6_l788_788376


namespace max_rides_day1_max_rides_day2_l788_788758

open List 

def daily_budget : ℤ := 10

def ride_prices_day1 : List (String × ℤ) := 
  [("Ferris wheel", 4), ("Roller coaster", 5), ("Bumper cars", 3), ("Carousel", 2), ("Log flume", 6)]

def ride_prices_day2 : List (String × ℤ) := 
  [("Ferris wheel", 4), ("Roller coaster", 7), ("Bumper cars", 3), ("Carousel", 2), ("Log flume", 6), ("Haunted house", 4)]

def max_rides (budget : ℤ) (prices : List (String × ℤ)) : ℤ :=
  sorry -- We'll assume this calculates the max number of rides correctly based on the given budget and prices.

theorem max_rides_day1 : max_rides daily_budget ride_prices_day1 = 3 := by
  sorry 

theorem max_rides_day2 : max_rides daily_budget ride_prices_day2 = 3 := by
  sorry 

end max_rides_day1_max_rides_day2_l788_788758


namespace tangent_to_ln_curve_l788_788012

theorem tangent_to_ln_curve (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ y = ln x ∧ y = a * x)
  -> a = 1 / Real.exp 1 :=
by
  sorry

end tangent_to_ln_curve_l788_788012


namespace divide_stones_l788_788965

/-- A pile of 660 stones can be divided into 30 piles where the sizes of the piles differ by less than a factor of 2. -/
theorem divide_stones (n : ℕ) (p : ℕ) (stones : ℕ) :
  stones = 660 → p = 30 →
  ∃ (heaps : Fin p → ℕ),
    (∑ i, heaps i = stones) ∧ (∀ i j, heaps i ≤ 2 * heaps j ∧ heaps j ≤ 2 * heaps i) :=
by
  intros h1 h2
  sorry

end divide_stones_l788_788965


namespace projection_matrix_is_correct_l788_788352

noncomputable def projectionMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  let v : Fin 2 → ℝ := ![3, 4]
  (1 / (v 0 ^ 2 + v 1 ^ 2)) • (λ i j, v i * v j)

theorem projection_matrix_is_correct :
  projectionMatrix = ![![9/25, 12/25], ![12/25, 16/25]] :=
by
  sorry

end projection_matrix_is_correct_l788_788352


namespace projection_onto_vector_l788_788319

noncomputable def projection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  ![\[9 / 25, 12 / 25\], \[12 / 25, 16 / 25\]]

theorem projection_onto_vector:
    ∀ (x y : ℚ), (Matrix.mul_vec projection_matrix ![\x, \y]) = ![(9 * x + 12 * y) / 25, (12 * x + 16 * y) / 25] := by
  sorry

end projection_onto_vector_l788_788319


namespace find_radius_of_tangent_circle_l788_788883

def tangent_circle_radius : Prop :=
  ∃ (r : ℝ), 
    (r > 0) ∧ 
    (∀ (θ : ℝ),
      (∃ (x y : ℝ),
        x = 1 + r * Real.cos θ ∧ 
        y = 1 + r * Real.sin θ ∧ 
        x + y - 1 = 0))
    → r = (Real.sqrt 2) / 2

theorem find_radius_of_tangent_circle : tangent_circle_radius :=
sorry

end find_radius_of_tangent_circle_l788_788883


namespace not_curious_60_62_823_l788_788597

def curious (f : ℤ → ℤ) (a : ℤ) : Prop := ∀ x : ℤ, f(x) = f(a - x)

theorem not_curious_60_62_823 (f : ℤ → ℤ) (h1 : ∀ x : ℤ, f(x) ≠ x) :
  ¬ (curious f 60 ∨ curious f 62 ∨ curious f 823) :=
by 
  sorry

end not_curious_60_62_823_l788_788597


namespace max_distance_from_circle_to_line_l788_788607

theorem max_distance_from_circle_to_line :
  let Cx := -2
  let Cy := 1
  let radius := (√5 / 5)
  let d := (| -2 * 3 + 4 * 1 | / √(9 + 16))
  let max_distance := d + radius
  ((∀ (x y :ℝ), x^2 + y^2 + 4*x - 2*y + (24 / 5) = 0 → ∀ (x y:ℝ), 3*x + 4*y = 0 → max_distance = (2 + √5) / 5) :=
sorry

end max_distance_from_circle_to_line_l788_788607


namespace smallest_nine_digit_divisible_by_11_l788_788664

theorem smallest_nine_digit_divisible_by_11 :
  ∃ (n : ℕ), (∀ (d : ℕ), 1 ≤ d ∧ d ≤ 9 → ∃! (i : ℕ), 0 ≤ i ∧ i < 9 ∧ d = (nat.digits 10 n).nth i) ∧
  n % 11 = 0 ∧ n = 123475869 :=
sorry

end smallest_nine_digit_divisible_by_11_l788_788664


namespace general_term_and_T_n_bounds_l788_788010

-- Given conditions as definitions
def geom_seq_first_term (a : ℕ → ℝ) := a 1 = 3 / 2
def geom_seq_not_decreasing (a : ℕ → ℝ) := ∀ n, a n ≤ a (n + 1)
def S (a : ℕ → ℝ) : ℕ → ℝ
| 1       := a 1
| (n + 1) := S a n + a (n + 1)
def arithmetic_sequence (a S : ℕ → ℝ) := S 3 + a 3 + S 5 + a 5 = 2 * (S 4 + a 4)

-- Problem statement to be solved
theorem general_term_and_T_n_bounds (a : ℕ → ℝ) (S : ℕ → ℝ) (T : ℕ → ℝ) :
  geom_seq_first_term a →
  geom_seq_not_decreasing a →
  arithmetic_sequence a S →
  (∀ n, a n = (-1)^(n - 1) * 3 / 2^n) ∧
  (∀ n, T n = S n - 1 / S n → 
       (-7 / 12 ≤ T n ∧ T n ≤ 5 / 6)) :=
  by sorry

end general_term_and_T_n_bounds_l788_788010


namespace simplify_evaluate_expression_l788_788579

theorem simplify_evaluate_expression (a b : ℤ) (h1 : a = -2) (h2 : b = 4) : 
  (-(3 * a)^2 + 6 * a * b - (a^2 + 3 * (a - 2 * a * b))) = 14 :=
by
  rw [h1, h2]
  sorry

end simplify_evaluate_expression_l788_788579


namespace projection_matrix_correct_l788_788314

variables {R : Type*} [field R] [decidable_eq R]
variables (x y : R)
def vector_v : matrix (fin 2) (fin 1) R := ![![3], ![4]]
def vector_u : matrix (fin 2) (fin 1) R := ![![x], ![y]]
def projection_matrix : matrix (fin 2) (fin 2) R := ![![9/25, 12/25], ![12/25, 16/25]]

theorem projection_matrix_correct :
  (projection_matrix R) ⬝ (vector_u x y) = (25 : R)⁻¹ • (transpose (vector_v 3 4) ⬝ (vector_u x y)) ⬝ (vector_v 3 4) := 
sorry

end projection_matrix_correct_l788_788314


namespace no_oper_yields_4_l788_788423

theorem no_oper_yields_4 : 
  ∀ (op : ℝ → ℝ → ℝ), (op ≠ (· + ·) ∧ op ≠ (· - ·) ∧ op ≠ (· * ·) ∧ op ≠ (· / ·)) →
  (op 9 3 ≠ 4) :=
begin
  intro op,
  intro h,
  cases h with h1 h_remaining,
  cases h_remaining with h2 h_remaining,
  cases h_remaining with h3 h_rest,
  have h_add : (9 + 3) ≠ 4, by norm_num,
  have h_sub : (9 - 3) ≠ 4, by norm_num,
  have h_mul : (9 * 3) ≠ 4, by norm_num,
  have h_div : (9 / 3) ≠ 4, by norm_num,
  cases h_rest with h4 _,
  exact h4,
  sorry,
end

end no_oper_yields_4_l788_788423


namespace slope_ratio_constant_l788_788100

variable (a : ℝ)
variable (C : Ellipse (0, 0) a (sqrt (a^2 - 1)))
variable (F : Point (1, 0))
variable (l : Line)
variable (P Q : Point)
variable (A : Point (0, 0))
variable (B : Point (a, 0))
variable (k1 k2 : ℝ)

-- Conditions
axiom h1 : l ≠ horizontal
axiom h2 : l ∋ F
axiom h3 : ∀ P₁ P₂, LineSegment B P₁.isOn (C) → LineSegment A P₂.isOn (C) → 
                    l ∋ P₁ → l ∋ P₂ → (P = P₁ ∨ P = P₂) ∧ (Q = P₁ ∨ Q = P₂) ∧ P ≠ Q  
axiom h4 : slope (LineSegment A P) = k1
axiom h5 : slope (LineSegment B Q) = k2

-- Proof of \( \frac{k1}{k2} \) being a constant in terms of \( a \)
theorem slope_ratio_constant : (k1 / k2) = (a - 1) / (a + 1) := sorry

end slope_ratio_constant_l788_788100


namespace parabola_and_circle_eq_line_A2A3_tangent_l788_788624

-- Define the conditions of the problem
-- Vertex of the parabola at the origin and focus on the x-axis
def parabola_eq : Prop := ∃ p > 0, ∀ x y : ℝ, (y^2 = 2 * p * x ↔ (x, y) ∈ C)

-- Define line l: x = 1
def line_l (x y : ℝ) : Prop := x = 1

-- Define the parabola C and the points of intersection P and Q
def intersection_points (y : ℝ) : Prop := (1, y) ∈ C

-- Define the perpendicularity condition OP ⊥ OQ
def perpendicular_condition (P Q : ℝ × ℝ) : Prop := (∃ p > 0, P = (1, sqrt p) ∧ Q = (1, -sqrt p))

-- Define the point M and its associated circle M tangent to line l
def point_M : ℝ × ℝ := (2, 0)

def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the points A1, A2, A3 on parabola C
def on_parabola (A : ℝ × ℝ) : Prop := (∃ p > 0, A.2^2 = 2 * p * A.1)

-- Define that lines A1A2 and A1A3 are tangent to circle M
def tangent_to_circle (A₁ A₂ : ℝ × ℝ) : Prop := sorry

-- Prove the equation of parabola C and circle M
theorem parabola_and_circle_eq : (∀ x y : ℝ, y^2 = x ∧ (x - 2)^2 + y^2 = 1) :=
by
  sorry

-- Prove the position relationship between line A2A3 and circle M
theorem line_A2A3_tangent (A₁ A₂ A₃ : ℝ × ℝ) :
    on_parabola A₁ ∧ on_parabola A₂ ∧ on_parabola A₃ ∧ tangent_to_circle A₁ A₂ ∧ tangent_to_circle A₁ A₃ →
    (∃ l_tangent : ℝ, tangent_to_circle A₂ A₃) :=
by
  sorry

end parabola_and_circle_eq_line_A2A3_tangent_l788_788624


namespace polynomial_expression_l788_788582

noncomputable def p (x : ℝ) : ℝ := -(x^5) + 4 * (x^3) + 24 * (x^2) + 16 * x + 1

theorem polynomial_expression (x : ℝ) :
  p(x) + (x^5 + 3 * (x^3) + 9 * x) = 7 * (x^3) + 24 * (x^2) + 25 * x + 1 :=
by
  rw [p]
  ring
  sorry

end polynomial_expression_l788_788582


namespace projection_matrix_correct_l788_788313

variables {R : Type*} [field R] [decidable_eq R]
variables (x y : R)
def vector_v : matrix (fin 2) (fin 1) R := ![![3], ![4]]
def vector_u : matrix (fin 2) (fin 1) R := ![![x], ![y]]
def projection_matrix : matrix (fin 2) (fin 2) R := ![![9/25, 12/25], ![12/25, 16/25]]

theorem projection_matrix_correct :
  (projection_matrix R) ⬝ (vector_u x y) = (25 : R)⁻¹ • (transpose (vector_v 3 4) ⬝ (vector_u x y)) ⬝ (vector_v 3 4) := 
sorry

end projection_matrix_correct_l788_788313


namespace max_value_xyz_l788_788538

theorem max_value_xyz (x y z : ℝ) (h : x + y + 2 * z = 5) : 
  (∃ x y z : ℝ, x + y + 2 * z = 5 ∧ xy + xz + yz = 25/6) :=
sorry

end max_value_xyz_l788_788538


namespace factorization_of_1386_l788_788457

-- We start by defining the number and the requirements.
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def factors_mult (a b : ℕ) : Prop := a * b = 1386
def factorization_count (count : ℕ) : Prop :=
  ∃ (a b : ℕ), is_two_digit a ∧ is_two_digit b ∧ factors_mult a b ∧ 
  (∀ c d, is_two_digit c ∧ is_two_digit d ∧ factors_mult c d → 
  (c = a ∧ d = b ∨ c = b ∧ d = a) → c = a ∧ d = b ∨ c = b ∧ d = a) ∧
  count = 4

-- Now, we state the theorem.
theorem factorization_of_1386 : factorization_count 4 :=
sorry

end factorization_of_1386_l788_788457


namespace hiker_walks_18_miles_on_first_day_l788_788732

noncomputable def miles_walked_first_day (h : ℕ) : ℕ := 3 * h

def total_miles_walked (h : ℕ) : ℕ := (3 * h) + (4 * (h - 1)) + (4 * h)

theorem hiker_walks_18_miles_on_first_day :
  (∃ h : ℕ, total_miles_walked h = 62) → miles_walked_first_day 6 = 18 :=
by
  sorry

end hiker_walks_18_miles_on_first_day_l788_788732


namespace projection_matrix_is_correct_l788_788351

noncomputable def projectionMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  let v : Fin 2 → ℝ := ![3, 4]
  (1 / (v 0 ^ 2 + v 1 ^ 2)) • (λ i j, v i * v j)

theorem projection_matrix_is_correct :
  projectionMatrix = ![![9/25, 12/25], ![12/25, 16/25]] :=
by
  sorry

end projection_matrix_is_correct_l788_788351


namespace tangent_lines_l788_788771

def P : (ℚ × ℚ) := (3/5, 14/5)
def ellipse (x y : ℚ) : Prop := 4 * x^2 + 9 * y^2 = 36

theorem tangent_lines (tangent1 tangent2 : ℚ × ℚ → Prop) :
  (∀ x y : ℚ, tangent1 (x, y) ↔ (8 * x + 9 * y = 30)) ∧
  (∀ x y : ℚ, tangent2 (x, y) ↔ (x - 2 * y = -5)) → 
  (∃ t1 t2 : ℚ × ℚ → Prop, 
    (∀ x y : ℚ, t1 (x, y) ↔ (8 * x + 9 * y = 30)) ∧ 
    (∀ x y : ℚ, t2 (x, y) ↔ (x - 2 * y = -5))) → 
  (t1 p → ellipse p) ∧ (t2 p → ellipse p) := 
sorry

end tangent_lines_l788_788771


namespace compound_interest_correct_l788_788576

def total_savings : ℝ := 2750
def principal_simple_interest : ℝ := total_savings / 2
def simple_interest_received : ℝ := 550
def time_years : ℝ := 2
def principal_compound_interest : ℝ := total_savings / 2

noncomputable def interest_rate : ℝ := simple_interest_received / (principal_simple_interest * time_years)

theorem compound_interest_correct :
  let r := interest_rate in
  let A := principal_compound_interest * (1 + r)^time_years in
  let CI := A - principal_compound_interest in
  CI = 605 :=
by
  sorry

end compound_interest_correct_l788_788576


namespace tan_405_eq_1_l788_788267

theorem tan_405_eq_1 : Real.tan (405 * Real.pi / 180) = 1 :=
by
  have h1 : (405 * Real.pi / 180) = (45 * Real.pi / 180 + 2 * Real.pi), by norm_num
  rw [h1, Real.tan_add_two_pi]
  exact Real.tan_pi_div_four.symm

end tan_405_eq_1_l788_788267


namespace sugar_ratio_l788_788289

theorem sugar_ratio (r : ℝ) (H1 : 24 * r^3 = 3) : (24 * r / 24 = 1 / 2) :=
by
  sorry

end sugar_ratio_l788_788289


namespace range_of_function_a_eq_2_b_eq_2_exists_positive_b_for_even_function_range_of_a_strictly_increasing_l788_788433

section
variables {x : ℝ} (a b : ℝ) (y : ℝ)

-- 1. Prove range of the function when a = b = 2 is (1/2, 1)
theorem range_of_function_a_eq_2_b_eq_2 (x : ℝ) : 
  (∃ y, y = (1 - 1 / (2^x + 2)) ∧ ∃ x, y = (1 - (2^x + 1) / (2^x + 2))) :=
begin
  sorry
end

-- 2. Prove that there exists a positive number b such that the function is even when a = 0
theorem exists_positive_b_for_even_function (a : ℝ) : 
  a = 0 → ∃ b > 0, (∀ x, (b^x + 1) / (2^x) = (b^(-x) + 1) / (2^(-x))) :=
begin
  sorry
end

-- 3. Prove the range of a for the function to be strictly increasing on [-1, +∞) when a > 0 and b = 4
theorem range_of_a_strictly_increasing (a : ℝ) : 
  (∀ x1 x2 ∈ Ici (-1), x1 < x2 → (4^x1 + 1) / (2^x1 + a) < (4^x2 + 1) / (2^x2 + a)) 
  ↔ (a ≥ 3/4) :=
begin
  sorry
end

end

end range_of_function_a_eq_2_b_eq_2_exists_positive_b_for_even_function_range_of_a_strictly_increasing_l788_788433


namespace number_of_numbers_with_zero_digit_l788_788856

-- Define a function that checks if a given positive integer contains the digit 0.
def containsZeroDigit (n : Nat) : Bool :=
  n.digits 10 |> List.contains 0

-- Define the range of numbers we are considering.
def range := List.range' 1 3020

-- Define the problem statement.
theorem number_of_numbers_with_zero_digit : 
  List.countp containsZeroDigit range = 572 :=
by
  sorry

end number_of_numbers_with_zero_digit_l788_788856


namespace polygon_self_intersect_l788_788687

theorem polygon_self_intersect (n : ℕ) (P : ℤ → (ℤ × ℤ)) :
  (∀ i : ℕ, (i < n) → (P (2*i)).1 = (P (2*i + 1)).1 ∧ (P (2*i)).2 > (P (2*i + 1)).2)
  → (∀ i : ℕ, (i < n-1) → ((P (2*i + 1)).2 = (P (2*(i+1))).2))
  → ∃ i j : ℕ, i ≠ j ∧ line_intersect (P i) (P (i+1)) (P j) (P (j+1)) :=
sorry

end polygon_self_intersect_l788_788687


namespace sale_in_second_month_l788_788731

-- Define the constants for known sales and average requirement
def sale_first_month : Int := 8435
def sale_third_month : Int := 8855
def sale_fourth_month : Int := 9230
def sale_fifth_month : Int := 8562
def sale_sixth_month : Int := 6991
def average_sale_per_month : Int := 8500
def number_of_months : Int := 6

-- Define the total sales required for six months
def total_sales_required : Int := average_sale_per_month * number_of_months

-- Define the total known sales excluding the second month
def total_known_sales : Int := sale_first_month + sale_third_month + sale_fourth_month + sale_fifth_month + sale_sixth_month

-- The statement to prove: the sale in the second month is 8927
theorem sale_in_second_month : 
  total_sales_required - total_known_sales = 8927 := 
by
  sorry

end sale_in_second_month_l788_788731


namespace sam_total_cans_l788_788707

theorem sam_total_cans (bags_saturday bags_sunday bags_total cans_per_bag total_cans : ℕ)
    (h1 : bags_saturday = 3)
    (h2 : bags_sunday = 4)
    (h3 : bags_total = bags_saturday + bags_sunday)
    (h4 : cans_per_bag = 9)
    (h5 : total_cans = bags_total * cans_per_bag) : total_cans = 63 :=
sorry

end sam_total_cans_l788_788707


namespace average_difference_is_neg3_l788_788677

-- Define the total number of data points
def n : ℕ := 30

-- Define the actual and incorrect data point
def actual_value : ℕ := 105
def incorrect_value : ℕ := 15

-- Define the resulting undercount
def undercount : ℕ := actual_value - incorrect_value := 90

-- Define the difference in average due to the error
def average_difference : ℤ := - (undercount / n)

/-- 
  Theorem: Difference between the calculated average and the actual average
  Given a mistake in data entry where one data point of 105 was entered as 15 out of 30 data points,
  the difference between the calculated average and the actual average is -3.
-/
theorem average_difference_is_neg3 : average_difference = -3 := 
by
  sorry

end average_difference_is_neg3_l788_788677


namespace stones_partition_l788_788991

theorem stones_partition (total_stones : ℕ) (piles : ℕ) (heaps : ℕ → ℕ) 
  (h_total : total_stones = 660)
  (h_piles : piles = 30)
  (h_sum_heaps : ∑ i in range piles, heaps i = 660)
  (h_factor : ∀ i j, i < piles → j < piles → heaps i ≤ 2 * heaps j) :
  ∃ heaps : Π i : ℕ, i < piles → ℕ,
    (∑ i in range piles, heaps i = 660) ∧
    (∀ i j, i < piles → j < piles → heaps i ≤ 2 * heaps j) :=
  sorry

end stones_partition_l788_788991


namespace count_valid_lists_l788_788606

def is_valid_list (a b c d e : ℕ) : Prop :=
  a < b ∧ b < c ∧ c < d ∧ d < e ∧
  c = a + b ∧
  d = b + c ∧
  e = c + d

theorem count_valid_lists : 
  (∃ (a b c d e : ℕ), is_valid_list a b c d e ∧ e = 124) =
  8 := 
sorry

end count_valid_lists_l788_788606


namespace locus_is_circle_l788_788660

noncomputable def locus_of_centers_of_gravity (R a : ℝ) : set (ℝ × ℝ) :=
{p | ∃ θ φ ψ, p = ((2 * a) / 3 + (2 / 3) * R * (Real.cos θ + Real.cos φ + Real.cos ψ), (2 / 3) * R * (Real.sin θ + Real.sin φ + Real.sin ψ))}

theorem locus_is_circle (R a : ℝ) : 
  locus_of_centers_of_gravity R a = {p | ∃ t : ℝ, p = ((2 * a) / 3, 0) + (4 * R / 3) * (Real.cos t, Real.sin t)} :=
sorry

end locus_is_circle_l788_788660


namespace find_a4_b4_c4_l788_788534

-- Define the roots of the polynomial
variables {a b c : ℝ}

-- Define the polynomial conditions given
def polynomial_condition : Prop :=
  ∀ x : ℝ, (x = a ∨ x = b ∨ x = c) → x^3 - 2 * x^2 + 3 * x - 4 = 0

-- Define Vieta's formulas as conditions from the sum, product and sum of products of the roots
def vieta_conditions : Prop :=
  a + b + c = 2 ∧ a * b + a * c + b * c = 3 ∧ a * b * c = 4

-- State the final goal based on the conditions
theorem find_a4_b4_c4 (h_poly: polynomial_condition) (h_vieta: vieta_conditions) : 
  a^4 + b^4 + c^4 = 18 :=
sorry

end find_a4_b4_c4_l788_788534


namespace pure_ghee_percentage_l788_788061

theorem pure_ghee_percentage (Q : ℝ) (P : ℝ) (H1 : Q = 10) (H2 : (P / 100) * Q + 10 = 0.80 * (Q + 10)) :
  P = 60 :=
sorry

end pure_ghee_percentage_l788_788061


namespace inequality_f_bound_f_iter_bound_l788_788999

noncomputable def f : ℝ → ℝ := sorry

def increasing (f : ℝ → ℝ) := ∀ x y, x < y → f(x) < f(y)

lemma f_increasing : increasing f := sorry

axiom f_add_3 (x : ℝ) : f(x + 1) = f(x) + 3

def f_iter : ℕ → (ℝ → ℝ)
| 1       := f
| (n + 1) := f ∘ f_iter n

theorem inequality_f_bound (x : ℝ) : 3 * x + f(0) - 3 ≤ f(x) ∧ f(x) ≤ 3 * x + f(0) + 3 := sorry

theorem f_iter_bound (n : ℕ) (x y : ℝ) : abs (f_iter n x - f_iter n y) ≤ 3^n * (abs (x - y) + 3) := sorry

end inequality_f_bound_f_iter_bound_l788_788999


namespace divide_stones_into_heaps_l788_788951

-- Definitions based on the identified conditions
variable (Heaps : List ℕ) -- list of heap sizes
variable (n_stones : ℕ) -- total number of stones
variable (n_heaps : ℕ) -- number of heaps

-- Conditions
axiom total_stones : n_stones = 660
axiom total_heaps : n_heaps = 30
axiom heap_size_condition : ∀ (h₁ h₂ : ℕ), h₁ ∈ Heaps → h₂ ∈ Heaps → h₁ ≤ 2 * h₂

-- Theorem statement
theorem divide_stones_into_heaps : 
  ∃ Heaps, Heaps.length = n_heaps ∧ Heaps.sum = n_stones ∧
  ∀ (h₁ h₂ : ℕ), h₁ ∈ Heaps → h₂ ∈ Heaps → h₁ ≤ 2 * h₂ :=
by
  sorry

end divide_stones_into_heaps_l788_788951


namespace divide_660_stones_into_30_piles_l788_788984

theorem divide_660_stones_into_30_piles :
  ∃ (heaps : Fin 30 → ℕ),
    (∑ i, heaps i = 660) ∧
    (∀ i j, heaps i < 2 * heaps j ∨ heaps j < 2 * heaps i) :=
sorry

end divide_660_stones_into_30_piles_l788_788984


namespace dice_probability_ne_zero_l788_788679

theorem dice_probability_ne_zero :
  let outcomes := {[1, 2, 3, 4, 5, 6]} in
  ∃ (a b c d : ℕ) (h1 : a ∈ outcomes) (h2 : b ∈ outcomes) (h3 : c ∈ outcomes) (h4 : d ∈ outcomes),
  ((a - 1) * (b - 1) * (c - 1) * (d - 1) ≠ 0) →
  (prob_of_event := (5/6)^4) →
  prob_of_event = 625 / 1296 := 
sorry

end dice_probability_ne_zero_l788_788679


namespace tangent_line_eq_extreme_values_range_of_a_l788_788431

noncomputable def f (x : ℝ) (a: ℝ) : ℝ := x^2 - a * Real.log x

-- (I) Proving the tangent line equation is y = x for a = 1 at x = 1.
theorem tangent_line_eq (h : ∀ x, f x 1 = x^2 - Real.log x) :
  ∃ y : (ℝ → ℝ), y = id ∧ y 1 = x :=
sorry

-- (II) Proving extreme values of the function f(x).
theorem extreme_values (a: ℝ) :
  (∃ x_min : ℝ, f x_min a = (a/2) - (a/2) * Real.log (a/2)) ∧ 
  (∀ x, ¬∃ x_max : ℝ, f x_max a > f x a) :=
sorry

-- (III) Proving the range of values for a.
theorem range_of_a :
  (∀ x, 2*x - (a/x) ≥ 0 → 2 < x) → a ≤ 8 :=
sorry

end tangent_line_eq_extreme_values_range_of_a_l788_788431


namespace sector_central_angle_in_radians_l788_788829

/-- 
Given a sector of a circle where the perimeter is 4 cm 
and the area is 1 cm², prove that the central angle 
of the sector in radians is 2.
-/
theorem sector_central_angle_in_radians 
  (r l : ℝ) 
  (h_perimeter : 2 * r + l = 4) 
  (h_area : (1 / 2) * l * r = 1) : 
  l / r = 2 :=
by
  sorry

end sector_central_angle_in_radians_l788_788829


namespace area_enclosed_set_S_l788_788097

   open Complex
   
   noncomputable def area_of_S : ℝ :=
     pi * 4^2 * (7/8) * (9/8)

   theorem area_enclosed_set_S :
     ∀ (z w : ℂ), z = w - 2 / w ∧ abs w = 4 →
     area_of_S = 63 * pi / 4 :=
   by
     sorry
   
end area_enclosed_set_S_l788_788097


namespace inverse_proportion_quadrants_l788_788166

-- Define the inverse proportion function and its constant.
def inverse_proportion_function (k : ℝ) : ℝ → ℝ :=
  λ x, -k / x

-- Given conditions
variable (k : ℝ)
variable (hk : k < 0)

-- Quadrant determination.
def is_in_quadrant_II (f : ℝ → ℝ) : Prop :=
  ∀ x, x < 0 → f x > 0

def is_in_quadrant_IV (f : ℝ → ℝ) : Prop :=
  ∀ x, x > 0 → f x < 0

-- The theorem statement
theorem inverse_proportion_quadrants : 
  is_in_quadrant_II (inverse_proportion_function k) hk ∧ 
  is_in_quadrant_IV (inverse_proportion_function k) hk :=
sorry

end inverse_proportion_quadrants_l788_788166


namespace find_x_of_series_eq_15_l788_788279

noncomputable def infinite_series (x : ℝ) : ℝ :=
  5 + (5 + x) / 3 + (5 + 2 * x) / 3^2 + (5 + 3 * x) / 3^3 + ∑' n, (5 + (n + 1) * x) / 3 ^ (n + 1)

theorem find_x_of_series_eq_15 (x : ℝ) (h : infinite_series x = 15) : x = 10 :=
sorry

end find_x_of_series_eq_15_l788_788279


namespace shaded_fraction_is_5_over_8_l788_788557

def triangle_area (b h : ℝ) : ℝ :=
  (1 / 2) * b * h

def smaller_triangle_area (b h : ℝ) : ℝ :=
  (1 / 2) * (b / 2) * (h / 2)

def shaded_area_fraction (b h : ℝ) : ℝ :=
  (triangle_area b h - smaller_triangle_area b h) / (triangle_area b h)

theorem shaded_fraction_is_5_over_8 (b h : ℝ) (hb : b > 0) (hh : h > 0) :
  shaded_area_fraction b h = 5 / 8 :=
by
  sorry

end shaded_fraction_is_5_over_8_l788_788557


namespace problem_statement_l788_788064

-- Define the given parametric equations and curve.
def param_x (t : ℝ) : ℝ := -2 - t
def param_y (t : ℝ) : ℝ := 2 - (sqrt 3) * t

-- Define the curve C.
def curve_C (x y : ℝ) : Prop := (y - 2)^2 - x^2 = 1

-- Define line l based on the parametric equations.
def line_l (t : ℝ) : Prop := curve_C (param_x t) (param_y t)

-- Define the coordinates of point P in polar and Cartesian coordinates.
def polar_to_cartesian (r θ : ℝ) : (ℝ × ℝ) :=
  (r * cos θ, r * sin θ)

-- Define point P.
def P : (ℝ × ℝ) := polar_to_cartesian (2 * sqrt 2) (3 * Real.pi / 4)

-- Coordinates of point A and point B
def point_A (x1 y1 : ℝ) : Prop := line_l x1 ∧ line_l y1
def point_B (x2 y2 : ℝ) : Prop := line_l x2 ∧ line_l y2

-- Define the distance function.
def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Midpoint M of segment AB.
def midpoint (x1 y1 x2 y2 : ℝ) : (ℝ × ℝ) :=
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Hypothesis for the value of |AB|
def length_AB (x1 y1 x2 y2: ℝ) : ℝ := 2 * sqrt 14

-- Hypotheses for midpoint M
def M : (ℝ × ℝ) := midpoint (-3) (2 - sqrt 3) (-3) (2 + sqrt 3) -- simplified

-- Define the distance between P and M.
def distance_PM (P M : (ℝ × ℝ)) : ℝ :=
  match P, M with
  | (xp, yp), (xm, ym) => distance xp yp xm ym

theorem problem_statement :
  ∃ (AB : ℝ), AB = 2 * sqrt 14 ∧
  ∃ (M : ℝ × ℝ), M = midpoint (-3) (2 - sqrt 3) (-3) (2 + sqrt 3) ∧
  ∃ (PM : ℝ), PM = distance_PM P M ∧ PM = 2 := by
  sorry

end problem_statement_l788_788064


namespace find_integer_l788_788790

theorem find_integer (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) (h3 : Real.cos (n * Real.pi / 180) = Real.sin (312 * Real.pi / 180)) :
  n = 42 :=
by
  sorry

end find_integer_l788_788790


namespace find_a8_in_arithmetic_sequence_l788_788488

variable {a : ℕ → ℕ} -- Define a as a function from natural numbers to natural numbers

-- Assume a is an arithmetic sequence
axiom arithmetic_sequence (a : ℕ → ℕ) : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_a8_in_arithmetic_sequence (h : a 4 + a 6 + a 8 + a 10 + a 12 = 120) : a 8 = 24 :=
by
  sorry  -- Proof to be filled in separately

end find_a8_in_arithmetic_sequence_l788_788488


namespace gcd_of_factorials_l788_788299

theorem gcd_of_factorials (n m : ℕ) (h1 : n = 7) (h2 : m = 8) :
  Nat.gcd (n.factorial) (m.factorial) = 5040 := by
  have fact7 : 7.factorial = 5040 := by
    norm_num
  rw [h1, h2]
  rw [Nat.factorial_succ]
  rw [<-mul_comm 8 7.factorial, fact7]
  exact Nat.gcd_mul_left 8 5040 1

end gcd_of_factorials_l788_788299


namespace gavin_blue_shirts_l788_788799

theorem gavin_blue_shirts (total_shirts green_shirts : ℕ) (h1 : total_shirts = 23) (h2 : green_shirts = 17) : 
  total_shirts - green_shirts = 6 :=
by
  rw [h1, h2]
  norm_num
  sorry

end gavin_blue_shirts_l788_788799


namespace pause_point_l788_788763

-- Definitions
def total_movie_length := 60 -- In minutes
def remaining_time := 30 -- In minutes

-- Theorem stating the pause point in the movie
theorem pause_point : total_movie_length - remaining_time = 30 := by
  -- This is the original solution in mathematical terms, omitted in lean statement.
  -- total_movie_length - remaining_time = 60 - 30 = 30
  sorry

end pause_point_l788_788763


namespace stones_partition_l788_788993

theorem stones_partition (total_stones : ℕ) (piles : ℕ) (heaps : ℕ → ℕ) 
  (h_total : total_stones = 660)
  (h_piles : piles = 30)
  (h_sum_heaps : ∑ i in range piles, heaps i = 660)
  (h_factor : ∀ i j, i < piles → j < piles → heaps i ≤ 2 * heaps j) :
  ∃ heaps : Π i : ℕ, i < piles → ℕ,
    (∑ i in range piles, heaps i = 660) ∧
    (∀ i j, i < piles → j < piles → heaps i ≤ 2 * heaps j) :=
  sorry

end stones_partition_l788_788993


namespace white_tshirts_l788_788685

theorem white_tshirts (packages shirts_per_package : ℕ) (h1 : packages = 71) (h2 : shirts_per_package = 6) : packages * shirts_per_package = 426 := 
by 
  sorry

end white_tshirts_l788_788685


namespace total_limes_l788_788251

-- Define the number of limes picked by Alyssa, Mike, and Tom's plums
def alyssa_limes : ℕ := 25
def mike_limes : ℕ := 32
def tom_plums : ℕ := 12

theorem total_limes : alyssa_limes + mike_limes = 57 := by
  -- The proof is omitted as per the instruction
  sorry

end total_limes_l788_788251


namespace total_resistance_l788_788278

theorem total_resistance (R₀ : ℝ) (h : R₀ = 10) : 
  let R₃ := R₀; let R₄ := R₀; let R₃₄ := R₃ + R₄;
  let R₂ := R₀; let R₅ := R₀; let R₂₃₄ := 1 / (1 / R₂ + 1 / R₃₄ + 1 / R₅);
  let R₁ := R₀; let R₆ := R₀; let R₁₂₃₄ := R₁ + R₂₃₄ + R₆;
  R₁₂₃₄ = 13.33 :=
by 
  sorry

end total_resistance_l788_788278


namespace projection_matrix_is_correct_l788_788357

noncomputable def projectionMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  let v : Fin 2 → ℝ := ![3, 4]
  (1 / (v 0 ^ 2 + v 1 ^ 2)) • (λ i j, v i * v j)

theorem projection_matrix_is_correct :
  projectionMatrix = ![![9/25, 12/25], ![12/25, 16/25]] :=
by
  sorry

end projection_matrix_is_correct_l788_788357


namespace lattice_points_in_region_l788_788733

def is_lattice_point (x y : ℝ) : Prop :=
  x = Int.ofNat (Nat.abs (Int.ofNat (Nat.abs (Int.floor x)))) ∧
  y = Int.ofNat (Nat.abs (Int.ofNat (Nat.abs (Int.floor y))))

def region (x y : ℝ) : Prop :=
  y = abs x ∨ y = -x^3 + 6*x + 3

theorem lattice_points_in_region : Nat :=
by {
  -- The correct proof logic should be filled
  -- Right now, the solution is directly assigned based on the problem conclusion
  exact 19
}

end lattice_points_in_region_l788_788733


namespace solve_circle_sum_l788_788293

def circle_sum_property : Prop :=
  ∃ (a b c d e f : ℕ), 
    a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} ∧ 
    c ∈ {1, 2, 3, 4, 5, 6} ∧ d ∈ {1, 2, 3, 4, 5, 6} ∧ 
    e ∈ {1, 2, 3, 4, 5, 6} ∧ f ∈ {1, 2, 3, 4, 5, 6} ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
    d ≠ e ∧ d ≠ f ∧ 
    e ≠ f ∧ 
    a + b + c = 10 ∧ a + d + e = 10 ∧ b + f + d = 10

theorem solve_circle_sum : circle_sum_property :=
sorry

end solve_circle_sum_l788_788293


namespace circle_geometry_problem_l788_788412

theorem circle_geometry_problem
  (A B C P D E F K: Point)
  (Γ Γ₁ Γ₂: Circle)
  (h1: IsInscribed (Triangle.mk A B C) Γ)
  (h2: IsSecant (Line.mk P B C) Γ)
  (h3: IsTangent (Line.mk P A) Γ)
  (h4: SymmetricPoint D A P)
  (h5: Circumcircle (Triangle.mk D A C) = Γ₁)
  (h6: Circumcircle (Triangle.mk P A B) = Γ₂)
  (h7: second_intersection Γ₁ Γ₂ = E)
  (h8: second_intersection (Line.mk E B) Γ₁ = F)
  (h9: intersects_extension (Line.mk C P) Γ₁ K) :
  CF = AB :=
sorry

end circle_geometry_problem_l788_788412


namespace find_DG_l788_788140

-- Define the constants and conditions.
variables (a b : ℕ) (S : ℕ)
variables (DG : ℕ) (BC : ℕ := 43)
variables (h_area_eq : S = 43 * (a + b))
variables (h_int_sides : ∀ (x y : ℕ), x ∣ S → y ∣ S → Nat.gcd x y = 1 → rect_sides x y)
variables (h_DG : S = a * DG)

-- The main theorem to prove DG = 1892
theorem find_DG (h_area_eq : S = 43 * (a + b)) (h_int_sides : ∀ (x y : ℕ), x ∣ S → y ∣ S → Nat.gcd x y = 1 → rect_sides x y)
    (h_DG : S = a * DG) : DG = 1892 := 
begin
  sorry
end

end find_DG_l788_788140


namespace lines_through_P_with_angle_30_eq_2_l788_788448

noncomputable def number_of_lines_through_P_with_angle_30 
  (a b : Line) (P : Point)
  (h1 : Skew a b) 
  (h2 : angle a b = 50) 
  (h3 : Point P) 
  : ℕ := 
  2

theorem lines_through_P_with_angle_30_eq_2 
  (a b : Line) (P : Point) 
  (h1 : Skew a b) 
  (h2 : angle a b = 50) 
  (h3 : Point P) : number_of_lines_through_P_with_angle_30 a b P h1 h2 h3 = 2 := 
sorry

end lines_through_P_with_angle_30_eq_2_l788_788448


namespace inspection_time_l788_788233

theorem inspection_time 
  (num_digits : ℕ) (num_letters : ℕ) 
  (letter_opts : ℕ) (start_digits : ℕ) 
  (inspection_time_three_hours : ℕ) 
  (probability : ℝ) 
  (num_vehicles : ℕ) 
  (vehicles_inspected : ℕ)
  (cond1 : num_digits = 4)
  (cond2 : num_letters = 2)
  (cond3 : letter_opts = 3)
  (cond4 : start_digits = 2)
  (cond5 : inspection_time_three_hours = 180) 
  (cond6 : probability = 0.02)
  (cond7 : num_vehicles = 900)
  (cond8 : vehicles_inspected = num_vehicles * probability) :
  vehicles_inspected = (inspection_time_three_hours / 10) :=
  sorry

end inspection_time_l788_788233


namespace solve_for_r_l788_788146

theorem solve_for_r (r s : ℚ) (h : (2 * (r - 45)) / 3 = (3 * s - 2 * r) / 4) (s_val : s = 20) :
  r = 270 / 7 :=
by
  sorry

end solve_for_r_l788_788146


namespace least_adjacent_probability_l788_788191

theorem least_adjacent_probability (n : ℕ) 
    (h₀ : 0 < n)
    (h₁ : (∀ m : ℕ, 0 < m ∧ m < n → (4 * m^2 - 4 * m + 8) / (m^2 * (m^2 - 1)) ≥ 1 / 2015)) : 
    (4 * n^2 - 4 * n + 8) / (n^2 * (n^2 - 1)) < 1 / 2015 := by
  sorry

end least_adjacent_probability_l788_788191


namespace find_d_l788_788604

theorem find_d 
  (x y : ℝ)
  (t : ℝ)
  (h1 : y = (4 * x - 8) / 5)
  (h2 : ∃ v d, v = ⟨5, 2⟩ ∧ (∀ x ≥ 5, ∥⟨x, y⟩ - ⟨5, 2⟩ = t * d)) :
  ∃ d, d = ⟨5 / Real.sqrt 41, 4 / Real.sqrt 41⟩ :=
sorry

end find_d_l788_788604


namespace find_third_side_length_l788_788482

noncomputable def third_side_length (a b : ℝ) (θ : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2 - 2 * a * b * real.cos θ)

theorem find_third_side_length :
  third_side_length 10 12 (150 * real.pi / 180) = real.sqrt (244 + 120 * real.sqrt 3) :=
by
  sorry

end find_third_side_length_l788_788482


namespace dice_probability_ne_zero_l788_788678

theorem dice_probability_ne_zero :
  let outcomes := {[1, 2, 3, 4, 5, 6]} in
  ∃ (a b c d : ℕ) (h1 : a ∈ outcomes) (h2 : b ∈ outcomes) (h3 : c ∈ outcomes) (h4 : d ∈ outcomes),
  ((a - 1) * (b - 1) * (c - 1) * (d - 1) ≠ 0) →
  (prob_of_event := (5/6)^4) →
  prob_of_event = 625 / 1296 := 
sorry

end dice_probability_ne_zero_l788_788678


namespace pile_division_660_stones_l788_788975

theorem pile_division_660_stones (n : ℕ) (heaps : List ℕ) :
  n = 660 ∧ heaps.length = 30 ∧ ∀ x ∈ heaps, ∀ y ∈ heaps, (x ≤ 2 * y ∧ y ≤ 2 * x) →
  ∃ heaps : List ℕ, n = 660 ∧ heaps.length = 30 ∧ (∀ x y ∈ heaps, x ≤ 2 * y ∧ y ≤ 2 * x) :=
by
  sorry

end pile_division_660_stones_l788_788975


namespace right_triangle_ABC_l788_788815

noncomputable def parabola : set (ℝ × ℝ) := {p | p.2^2 = 4 * p.1}

noncomputable def A : ℝ × ℝ := (1, 2)

def line_intersects_parabola (B C : ℝ × ℝ) (hB : B ∈ parabola) (hC : C ∈ parabola) (p : ℝ × ℝ) (hP : p = (5, -2)) : Prop :=
∃ k : ℝ, ∃ l : ℝ, (B = ((k+1) * (k-1), 2 * k) ∧ C = ((l+1) * (l-1), 2 * l) ∧ hP.1 + (k + l) * hP.2 / 2 + (k * l + 1) = 5)

theorem right_triangle_ABC (B C : ℝ × ℝ) (hB : B ∈ parabola) (hC : C ∈ parabola)
(hL : line_intersects_parabola B C (5, -2) ⟨rfl, rfl⟩) :
  let k := (2 * B.2 - 2) / (B.1 - 1),
      l := (2 * C.2 - 2) / (C.1 - 1) in
  k * l = -1 :=
sorry

end right_triangle_ABC_l788_788815


namespace michael_passes_donovan_in_laps_l788_788281

theorem michael_passes_donovan_in_laps :
  ∀ (track_length lap_time_donovan lap_time_michael : ℕ),
    track_length = 300 →
    lap_time_donovan = 45 →
    lap_time_michael = 40 →
    ∃ laps : ℕ, laps = 9 ∧ (laps * lap_time_michael) - (laps * lap_time_donovan) ≥ track_length :=
begin
  intros track_length lap_time_donovan lap_time_michael h1 h2 h3,
  use 9,
  split,
  exact rfl,
  sorry
end

end michael_passes_donovan_in_laps_l788_788281


namespace basis_transformation_coplanar_condition_l788_788817

open Function

namespace MathProof

variables {V : Type*} [InnerProductSpace ℝ V]
variables (O A B C M : V)
variables (OA OB OC : V)
variables {x y z : ℝ}

-- The given vectors form a basis for space
axiom OA_basis : Basis (Fin 3) ℝ V
axiom OB_basis : Basis (Fin 3) ℝ V
axiom OC_basis : Basis (Fin 3) ℝ V

-- Given condition for OM vector
def OM := x • OA + y • OB + z • OC

-- The mathematically equivalent problem
theorem basis_transformation (h_basis : LinearlyIndependent ℝ ![OA, OB, OC]) :
  LinearlyIndependent ℝ ![OA + OB, OA - OB, OC] :=
sorry

theorem coplanar_condition (h_om : OM = x • OA + y • OB + z • OC) :
  (x + y + z = 1) ↔ (∃ (a b c : ℝ), M = a • A + b • B + c • C ∧ a + b + c = 1) :=
sorry

end MathProof

end basis_transformation_coplanar_condition_l788_788817


namespace arithmetic_sequence_general_term_l788_788828

noncomputable def general_term (a : ℕ → ℤ) (d : ℤ) (t : ℤ) : ℕ → ℤ :=
  λ n, (2 * n - 1)

theorem arithmetic_sequence_general_term (a : ℕ → ℤ) (d : ℤ) (t : ℤ) (h1 : ∀ n : ℕ, a (n + 1) = a n + d)
  (h2 : 0 < d) (h3 : a 1 = 1) (h4 : ∀ n : ℕ, 2 * (a n * a (n + 1) + 1) = t * n * (1 + a n)) : 
  ∀ n : ℕ, a n = 2 * n - 1 :=
by
  sorry

end arithmetic_sequence_general_term_l788_788828


namespace divide_660_stones_into_30_piles_l788_788983

theorem divide_660_stones_into_30_piles :
  ∃ (heaps : Fin 30 → ℕ),
    (∑ i, heaps i = 660) ∧
    ∀ i j, heaps i ≤ 2 * heaps j :=
sorry

end divide_660_stones_into_30_piles_l788_788983


namespace evaluate_power_sum_l788_788288

noncomputable def i : ℂ := complex.I

theorem evaluate_power_sum :
  (i^14 + i^19 + i^24 + i^29 + i^34 + i^39) = -1 - i :=
by {
  have h1 : i^2 = -1 := by sorry,
  have h2 : i^4 = 1 := by sorry,
  sorry
}

end evaluate_power_sum_l788_788288


namespace symmetry_about_origin_l788_788600

def f (x : ℝ) : ℝ := -Real.exp (-x)
def g (x : ℝ) : ℝ := Real.exp x

theorem symmetry_about_origin : 
  ∀ x : ℝ, f(-x) = -f(x) ∧ g(-x) = -g(x) → (∀ y : ℝ, f y = -g y) → f(-x) = f x ∧ g(-x) = -g x :=
sorry

end symmetry_about_origin_l788_788600


namespace parabola_and_circle_tangency_l788_788619

open Real

noncomputable def parabola_eq : Prop :=
  (parabola : {x : ℝ → ℝ | ∃ y: ℝ, y^2 = x})

noncomputable def circle_eq : Prop :=
  (circle : {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2)^2 = 1})

theorem parabola_and_circle_tangency:
  (∀ x y : ℝ, ∃ p, y^2 = x ↔ p ∈ parabola_eq) →
  ((x - 2)^2 + y^2 = 1) →
  (∀ A1 A2 A3 : ℝ × ℝ,
    A1 ∈ parabola_eq ∧ A2 ∈ parabola_eq ∧ A3 ∈ parabola_eq →
    (tangential A1 A2 circle ∧ tangential A1 A3 circle →
    tangential A2 A3 circle
  )) := sorry

end parabola_and_circle_tangency_l788_788619


namespace collinear_UVW_l788_788571

theorem collinear_UVW
  (A B C D X Y U V W : Point)
  (AB CD : Line)
  (h_cyclic : cyclic_quadrilateral A B C D)
  (h_circumscribed : circumscribed_quadrilateral A B C D)
  (h_incircle_touch_AB_X : touches_incircle A B X)
  (h_incircle_touch_CD_Y : touches_incircle C D Y)
  (h_perpendicular_AU : perpendicular (line_through A D) (line_through A U))
  (h_perpendicular_DW : perpendicular (line_through D C) (line_through D W))
  (h_perpendicular_XV : perpendicular (line_through X Y) (line_through X V))
  (h_perpendicular_YV : perpendicular (line_through Y X) (line_through Y V))
  (h_perpendicular_BW : perpendicular (line_through B D) (line_through B W))
  (h_perpendicular_CW : perpendicular (line_through C A) (line_through C W)) :
  collinear U V W :=
sorry

end collinear_UVW_l788_788571


namespace John_distance_proof_l788_788083

def initial_running_time : ℝ := 8
def increase_percentage : ℝ := 0.75
def initial_speed : ℝ := 8
def speed_increase : ℝ := 4

theorem John_distance_proof : 
  (initial_running_time + initial_running_time * increase_percentage) * (initial_speed + speed_increase) = 168 := 
by
  -- Proof can be completed here
  sorry

end John_distance_proof_l788_788083


namespace largest_of_seven_consecutive_integers_l788_788176

-- Define the main conditions as hypotheses
theorem largest_of_seven_consecutive_integers (n : ℕ) (h_sum : 7 * n + 21 = 2401) : 
  n + 6 = 346 :=
by
  -- Conditions from the problem are utilized here
  sorry

end largest_of_seven_consecutive_integers_l788_788176


namespace complex_in_second_quadrant_l788_788490

-- Define the complex number
def complex_number : ℂ := (-2 - 3 * complex.I) / complex.I

-- Define the condition for the quadrant
def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- State the theorem to prove the location of the complex number
theorem complex_in_second_quadrant : in_second_quadrant complex_number :=
by
  -- We skip the actual proof
  sorry

end complex_in_second_quadrant_l788_788490


namespace area_circle_outside_triangle_l788_788525

open Real

-- Defining the problem conditions
variables (A B C X Y : ℝ) (r : ℝ)
variables (angle_BAC : ∠ A B C = π / 2)
variables (AB AC BC : ℝ)
variables (circle_tangent_AB_AC : Circle (midpoint A (midpoint B C)) r)
variables (AB_val : AB = 9)
variables (AC_val : AC = 12)

-- The proof problem statement
theorem area_circle_outside_triangle :
  let area := (1 / 4) * π * r^2 - (1 / 2) * r^2 in
  AB = 9 → AC = 12 → BC = sqrt (AB^2 + AC^2) → r = 3 → area = (9 * (π - 2)) / 4 :=
by
  -- Acknowledge the variables
  intros x y :: sorry

end area_circle_outside_triangle_l788_788525


namespace area_of_rectangle_l788_788723

-- Define the problem conditions in Lean
def circle_radius := 7
def circle_diameter := 2 * circle_radius
def width_of_rectangle := circle_diameter
def length_to_width_ratio := 3
def length_of_rectangle := length_to_width_ratio * width_of_rectangle

-- Define the statement to be proved (area of the rectangle)
theorem area_of_rectangle : 
  (length_of_rectangle * width_of_rectangle) = 588 := by
  sorry

end area_of_rectangle_l788_788723


namespace pile_division_660_stones_l788_788972

theorem pile_division_660_stones (n : ℕ) (heaps : List ℕ) :
  n = 660 ∧ heaps.length = 30 ∧ ∀ x ∈ heaps, ∀ y ∈ heaps, (x ≤ 2 * y ∧ y ≤ 2 * x) →
  ∃ heaps : List ℕ, n = 660 ∧ heaps.length = 30 ∧ (∀ x y ∈ heaps, x ≤ 2 * y ∧ y ≤ 2 * x) :=
by
  sorry

end pile_division_660_stones_l788_788972


namespace max_value_ab_c_l788_788947

noncomputable def max_expression (a b c : ℝ) : ℝ := 2 * a * b * real.sqrt 3 + 2 * a * c 

theorem max_value_ab_c (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a^2 + b^2 + c^2 = 1) :
  max_expression a b c ≤ real.sqrt 3 :=
by
  sorry

end max_value_ab_c_l788_788947


namespace intersection_A_B_l788_788402

-- Define set A
def A : Set Int := { x | x^2 - x - 2 ≤ 0 }

-- Define set B
def B : Set Int := { x | x < 1 }

-- Define the intersection set
def intersection_AB : Set Int := { -1, 0 }

-- Formalize the proof statement
theorem intersection_A_B : (A ∩ B) = intersection_AB :=
by sorry

end intersection_A_B_l788_788402


namespace weight_of_person_replaced_l788_788055

variable (W total_weight : ℝ) (person_replaced new_person : ℝ)

-- Given conditions
axiom avg_weight_increase : (total_weight / 8) + 6 = (total_weight - person_replaced + new_person) / 8
axiom new_person_weight : new_person = 88

theorem weight_of_person_replaced : person_replaced = 40 :=
by 
  have h₁ : total_weight + 48 = total_weight - person_replaced + new_person :=
    (avg_weight_increase).symm.trans $ by norm_num
  rw new_person_weight at h₁
  have h₂ : total_weight + 48 = total_weight + (88 - person_replaced) := h₁
  have h₃ : 48 = 88 - person_replaced := by linarith
  exact eq_sub_of_add_eq h₃.symm

end weight_of_person_replaced_l788_788055


namespace number_of_students_before_new_year_l788_788699

variables (M N k ℓ : ℕ)
hypotheses (h1 : 100 * M = k * N)
             (h2 : 100 * (M + 1) = ℓ * (N + 3))
             (h3 : ℓ < 100)

theorem number_of_students_before_new_year (h1 : 100 * M = k * N)
                                             (h2 : 100 * (M + 1) = ℓ * (N + 3))
                                             (h3 : ℓ < 100) :
  N ≤ 197 :=
sorry

end number_of_students_before_new_year_l788_788699


namespace range_of_m_l788_788277

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, 4 * real.cos x - real.cos x ^ 2 + m - 3 = 0) ↔
  0 ≤ m ∧ m ≤ 8 := 
sorry

end range_of_m_l788_788277


namespace marissa_tied_boxes_l788_788115

def Total_ribbon : ℝ := 4.5
def Leftover_ribbon : ℝ := 1
def Ribbon_per_box : ℝ := 0.7

theorem marissa_tied_boxes : (Total_ribbon - Leftover_ribbon) / Ribbon_per_box = 5 := by
  sorry

end marissa_tied_boxes_l788_788115


namespace circle_eq_concentric_with_given_and_passes_through_point_l788_788789

noncomputable def given_circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y + 3 = 0
noncomputable def new_circle_center := (2 : ℝ, -3 : ℝ)
noncomputable def point_on_new_circle := (-1 : ℝ, 1 : ℝ)

theorem circle_eq_concentric_with_given_and_passes_through_point :
  ∃ m : ℝ, (∀ x y : ℝ, (x - 2) ^ 2 + (y + 3) ^ 2 = m) ∧ 
           ((point_on_new_circle.1 - 2)^2 + (point_on_new_circle.2 + 3)^2 = m) ∧ 
           m = 25 :=
by
  sorry

end circle_eq_concentric_with_given_and_passes_through_point_l788_788789


namespace projection_matrix_3_4_l788_788333

theorem projection_matrix_3_4 :
  let v := λ α : Type, @vector α 2 := ![3, 4]
  let proj := λ x : vector ℝ 2, (v ℝ ⬝ x) / (v ℝ ⬝ v ℝ) • v ℝ
  proj = (λ x : vector ℝ 2, matrix.mul_vec ![
     ![9 / 25, 12 / 25],
     ![12 / 25, 16 / 25]
  ] x) :=
by sorry

end projection_matrix_3_4_l788_788333


namespace correct_propositions_l788_788834

-- Definition for an acute angle between two vectors
def acute_angle (u v : Vector ℝ) : Prop :=
  (u.dot v) > 0

-- Definition of a function f such that f(x) = x
def f_eq_x (x : ℝ) : ℝ := x

-- Definition for close functions
def close_functions (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ set.Icc a b, abs (f x - g x) ≤ 1

-- Specific functions f(x) and g(x)
def f (x : ℝ) : ℝ := x^2 - 3*x + 4
def g (x : ℝ) : ℝ := 2*x - 3

-- The theorem to prove that propositions 1 and 3 are true
theorem correct_propositions :
  (∀ u v : Vector ℝ, acute_angle u v) ∧ close_functions f g 2 3 :=
by
  sorry

end correct_propositions_l788_788834


namespace flyers_left_l788_788500

theorem flyers_left (initial_flyers : ℕ) (jack_flyers : ℕ) (rose_flyers : ℕ) (left_flyers : ℕ) :
  initial_flyers = 1236 →
  jack_flyers = 120 →
  rose_flyers = 320 →
  left_flyers = 796 →
  initial_flyers - (jack_flyers + rose_flyers) = left_flyers := 
by
  intros h_initial h_jack h_rose h_left
  rw [h_initial, h_jack, h_rose, h_left]
  simp
  sorry

end flyers_left_l788_788500


namespace f_2012_l788_788934

theorem f_2012 (m n α₁ α₂ : ℝ) (h : m ≠ 0) (h' : n ≠ 0)
  (h₁ : m * sin(2011 * π + α₁) + n * cos(2011 * π + α₂) = 1) :
  m * sin(2012 * π + α₁) + n * cos(2012 * π + α₂) = -1 :=
by
  sorry

end f_2012_l788_788934


namespace amelia_money_left_l788_788252

theorem amelia_money_left :
  let first_course := 15
  let second_course := first_course + 5
  let dessert := 0.25 * second_course
  let total_first_three_courses := first_course + second_course + dessert
  let drink := 0.20 * total_first_three_courses
  let pre_tip_total := total_first_three_courses + drink
  let tip := 0.15 * pre_tip_total
  let total_bill := pre_tip_total + tip
  let initial_money := 60
  let money_left := initial_money - total_bill
  money_left = 4.8 :=
by
  sorry

end amelia_money_left_l788_788252


namespace probability_one_student_two_unqualified_expected_value_unqualified_items_l788_788283
noncomputable def foodTypeA_pass_rate : ℝ := 0.90
noncomputable def foodTypeB_pass_rate : ℝ := 0.80
noncomputable def foodTypeA_fail_rate : ℝ := 1 - foodTypeA_pass_rate
noncomputable def foodTypeB_fail_rate : ℝ := 1 - foodTypeB_pass_rate

/-
 Question 1: Probability that exactly one student gets two items that are both unqualified.
-/
theorem probability_one_student_two_unqualified :
  (3 * ((1 - foodTypeA_fail_rate * foodTypeB_fail_rate)^2 * (foodTypeA_fail_rate * foodTypeB_fail_rate))) = 0.0576 := 
sorry

/-
 Question 2: Expected value of the number of unqualified items purchased.
-/
def probability_xi_0 : ℝ := foodTypeA_pass_rate * foodTypeB_pass_rate
def probability_xi_1 : ℝ := (1 - foodTypeA_pass_rate) * foodTypeB_pass_rate + foodTypeA_pass_rate * (1 - foodTypeB_pass_rate)
def probability_xi_2 : ℝ := 1 - probability_xi_0 - probability_xi_1

theorem expected_value_unqualified_items :
  (0 * probability_xi_0 + 1 * probability_xi_1 + 2 * probability_xi_2) = 0.30 :=
sorry

end probability_one_student_two_unqualified_expected_value_unqualified_items_l788_788283


namespace distance_is_one_l788_788162

-- Define the distance formula for two parallel lines
def distance_between_parallel_lines (A B C1 C2 : ℝ) : ℝ :=
  abs (C2 - C1) / real.sqrt (A ^ 2 + B ^ 2)

-- Define the specific problem conditions
def A : ℝ := 4
def B : ℝ := 3
def C1 : ℝ := 5
def C2 : ℝ := 10

-- Theorem to state the distance between given lines
theorem distance_is_one : distance_between_parallel_lines A B C1 C2 = 1 :=
by
  -- Substitute the values and show the calculation
  sorry

end distance_is_one_l788_788162


namespace limit_log_div_x_alpha_l788_788783

open Real

theorem limit_log_div_x_alpha (α : ℝ) (hα : α > 0) :
  (Filter.Tendsto (fun x => (log x) / (x^α)) Filter.atTop (nhds 0)) :=
by
  sorry

end limit_log_div_x_alpha_l788_788783


namespace minimum_cards_for_even_product_l788_788777

noncomputable def ensure_even_product (cards : List Int) : Bool :=
  let even_count := cards.filter (λ x => x % 2 = 0).length
  let odd_count := cards.filter (λ x => x % 2 ≠ 0).length
  even_count ≥ 1

theorem minimum_cards_for_even_product : ∃ (cards : List Int), (cards.length = 3) ∧ 
  (cards.filter (λ x => x % 2 = 0).length ≥ 2) ∧ 
  (cards.filter (λ x => x % 2 ≠ 0).length ≥ 1) ∧ 
  ensure_even_product cards :=
by
  sorry

end minimum_cards_for_even_product_l788_788777


namespace sum_of_common_ratios_l788_788938

variable {k p r : ℝ}

theorem sum_of_common_ratios (h1 : k ≠ 0)
                             (h2 : p ≠ r)
                             (h3 : k * p^2 - k * r^2 = 5 * (k * p - k * r)) :
                             p + r = 5 := 
by
  sorry

end sum_of_common_ratios_l788_788938


namespace chess_probability_l788_788130

theorem chess_probability (P_draw P_B_win : ℚ) (h_draw : P_draw = 1/2) (h_B_win : P_B_win = 1/3) :
  (1 - P_draw - P_B_win = 1/6) ∧ -- Statement A is correct
  (P_draw + (1 - P_draw - P_B_win) ≠ 1/2) ∧ -- Statement B is incorrect as it's not 1/2
  (1 - P_draw - P_B_win ≠ 2/3) ∧ -- Statement C is incorrect as it's not 2/3
  (P_draw + P_B_win ≠ 1/2) := -- Statement D is incorrect as it's not 1/2
by
  -- Insert proof here
  sorry

end chess_probability_l788_788130


namespace q_minus_r_max_value_l788_788215

theorem q_minus_r_max_value :
  ∃ (q r : ℕ), q > 99 ∧ q < 1000 ∧ r > 99 ∧ r < 1000 ∧ 
    q = 100 * (q / 100) + 10 * ((q / 10) % 10) + (q % 10) ∧ 
    r = 100 * (q % 10) + 10 * ((q / 10) % 10) + (q / 100) ∧ 
    q - r = 297 :=
by sorry

end q_minus_r_max_value_l788_788215


namespace distinct_prime_factors_of_sigma_n_gcd_of_sigma_n_and_n_l788_788769

open Nat

namespace MathProof

def n : ℕ := 450

def prime_factors (n : ℕ) : Finset ℕ :=
  (n.factorization.support : Finset ℕ)

def sigma (n : ℕ) : ℕ := 
  (divisors n).sum

def gcd_of_sigma_and_n (n : ℕ) : ℕ :=
  gcd n (sigma n)

theorem distinct_prime_factors_of_sigma_n : prime_factors (sigma n).card = 3 := sorry

theorem gcd_of_sigma_n_and_n : gcd_of_sigma_and_n n = 3 := sorry

end MathProof

end distinct_prime_factors_of_sigma_n_gcd_of_sigma_n_and_n_l788_788769


namespace stones_partition_l788_788995

theorem stones_partition (total_stones : ℕ) (piles : ℕ) (heaps : ℕ → ℕ) 
  (h_total : total_stones = 660)
  (h_piles : piles = 30)
  (h_sum_heaps : ∑ i in range piles, heaps i = 660)
  (h_factor : ∀ i j, i < piles → j < piles → heaps i ≤ 2 * heaps j) :
  ∃ heaps : Π i : ℕ, i < piles → ℕ,
    (∑ i in range piles, heaps i = 660) ∧
    (∀ i j, i < piles → j < piles → heaps i ≤ 2 * heaps j) :=
  sorry

end stones_partition_l788_788995


namespace parabola_equation_and_fixed_point_and_slope_range_l788_788825

noncomputable theory

-- Definitions related to the parabola and lines
def point (x y : ℝ) : Type :=
⟨x, y⟩

def parabola (E : Type) : Type :=
  ∀ (x y : ℝ), y^2 = 4 * x

def line_through (l : Type) (M : Type) (slope : ℝ) : line :=
  l.includes M ∧ l.slope = slope

def condition (M : point (-1, 1)) (p : ℝ) (p_pos : p > 0)
  (l1 : line_through (M (-1,1)) 2) (l2 : line_through (M (-1,1)) 2) : Prop :=
  parabola y^2 = 2 * p * x

-- Statement requiring proof
theorem parabola_equation_and_fixed_point_and_slope_range :
  ∀ (M : point (-1, 1)) (p : ℝ) (p_pos : p > 0)
    (l1 : line_through M 2) (l2 : line_through M 2),
    ∃ E : parabola, E = ∀ x y, y^2 = 4 * x ∧
    ∃ H : point, H = point (3 / 2) 1 ∧
    ∃ k : ℝ,  (S ≤ 5) → (k ∈ (Set.Icc (-(sqrt 5 + 1) / 2) -1 ∪ Set.Icc 1/2 (sqrt 5 - 1)/2))
sorry

end parabola_equation_and_fixed_point_and_slope_range_l788_788825


namespace car_cost_difference_l788_788219

def total_cost (initial_cost fuel_cost insurance_cost maintenance_cost resale_value years : ℕ) : ℕ :=
  initial_cost + fuel_cost * years + insurance_cost * years + maintenance_cost * years - resale_value

def cost_of_car_A : ℕ :=
  total_cost 900000
             (15000 / 100 * 9 * 40)  -- Annual fuel cost for car 'A'
             35000 -- Annual insurance cost for car 'A'
             25000 -- Annual maintenance cost for car 'A'
             500000 -- Resale value of car 'A'
             5 -- Usage period

def cost_of_car_B : ℕ :=
  total_cost 600000
             (15000 / 100 * 10 * 40)  -- Annual fuel cost for car 'B'
             32000 -- Annual insurance cost for car 'B'
             20000 -- Annual maintenance cost for car 'B'
             350000 -- Resale value of car 'B'
             5 -- Usage period

theorem car_cost_difference :
  cost_of_car_A - cost_of_car_B = 160000 :=
by
  have h_car_A : cost_of_car_A = 970000 := sorry
  have h_car_B : cost_of_car_B = 810000 := sorry
  rw [h_car_A, h_car_B]
  rfl

end car_cost_difference_l788_788219


namespace triangle_concurrency_or_parallel_l788_788105

theorem triangle_concurrency_or_parallel 
  (A B C A' D E F P Q : Point)
  (h_reflection : A' = reflect A BC)
  (hD : D ∈ Segment BC ∧ D ≠ B ∧ D ≠ C)
  (hE : ∃ O1, O1 ∈ circumcircle (A B D) ∧ E ∈ Segment AC ∧ E ∈ O1)
  (hF : ∃ O2, O2 ∈ circumcircle (A C D) ∧ F ∈ Segment AB ∧ F ∈ O2)
  (hP : ∃ intP, intP ∈ line_intersection (A' C) (D E) ∧ P = intP)
  (hQ : ∃ intQ, intQ ∈ line_intersection (A' B) (D F) ∧ Q = intQ) :
  concurrent_or_parallel (A D) (B P) (C Q) :=
sorry

end triangle_concurrency_or_parallel_l788_788105


namespace projection_onto_vector_is_expected_l788_788306

def projection_matrix (u: ℝ × ℝ) : matrix (fin 2) (fin 2) ℝ :=
  let ⟨x, y⟩ := u in 
  (1 / (x^2 + y^2)) • (matrix.col_vec u ⬝ (matrix.transpose (matrix.row_vec u)))

def expected_matrix : matrix (fin 2) (fin 2) ℝ :=
  ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]]

theorem projection_onto_vector_is_expected :
  projection_matrix (3, 4) = expected_matrix := by
  sorry

end projection_onto_vector_is_expected_l788_788306


namespace flyers_left_to_hand_out_l788_788509

-- Definitions for given conditions
def total_flyers : Nat := 1236
def jack_handout : Nat := 120
def rose_handout : Nat := 320

-- Statement of the problem
theorem flyers_left_to_hand_out : total_flyers - (jack_handout + rose_handout) = 796 :=
by
  -- proof goes here
  sorry

end flyers_left_to_hand_out_l788_788509


namespace divide_660_stones_into_30_piles_l788_788980

theorem divide_660_stones_into_30_piles :
  ∃ (heaps : Fin 30 → ℕ),
    (∑ i, heaps i = 660) ∧
    ∀ i j, heaps i ≤ 2 * heaps j :=
sorry

end divide_660_stones_into_30_piles_l788_788980


namespace fraction_ratio_l788_788298

theorem fraction_ratio (x : ℚ) : 
  (x : ℚ) / (2/6) = (3/4) / (1/2) -> (x = 1/2) :=
by {
  sorry
}

end fraction_ratio_l788_788298


namespace parabola_and_circle_tangency_l788_788622

open Real

noncomputable def parabola_eq : Prop :=
  (parabola : {x : ℝ → ℝ | ∃ y: ℝ, y^2 = x})

noncomputable def circle_eq : Prop :=
  (circle : {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2)^2 = 1})

theorem parabola_and_circle_tangency:
  (∀ x y : ℝ, ∃ p, y^2 = x ↔ p ∈ parabola_eq) →
  ((x - 2)^2 + y^2 = 1) →
  (∀ A1 A2 A3 : ℝ × ℝ,
    A1 ∈ parabola_eq ∧ A2 ∈ parabola_eq ∧ A3 ∈ parabola_eq →
    (tangential A1 A2 circle ∧ tangential A1 A3 circle →
    tangential A2 A3 circle
  )) := sorry

end parabola_and_circle_tangency_l788_788622


namespace f_c_is_odd_f_a_not_odd_f_b_not_odd_f_d_not_odd_l788_788210

-- Define the functions
def fₐ (x : ℝ) := Real.log x
def f_b (x : ℝ) := Real.exp x
def f_c (x : ℝ) := x + Real.sin x
def f_d (x : ℝ) := Real.cos x + x^2

-- Prove that f_c is an odd function
theorem f_c_is_odd : ∀ x : ℝ, f_c (-x) = -f_c x := by
  intro x
  sorry

-- Definitions for conditions
def f_not_odd : ∀ (f : ℝ → ℝ), Prop := 
  λ f, ∃ x : ℝ, f (-x) ≠ -f x

-- State the non-odd functions
theorem f_a_not_odd : f_not_odd fₐ := by
  sorry

theorem f_b_not_odd : f_not_odd f_b := by
  sorry

theorem f_d_not_odd : f_not_odd f_d := by
  sorry

end f_c_is_odd_f_a_not_odd_f_b_not_odd_f_d_not_odd_l788_788210


namespace three_digit_number_count_l788_788456

def total_three_digit_numbers : ℕ := 900

def count_ABA : ℕ := 9 * 9  -- 81

def count_ABC : ℕ := 9 * 9 * 8  -- 648

def valid_three_digit_numbers : ℕ := total_three_digit_numbers - (count_ABA + count_ABC)

theorem three_digit_number_count :
  valid_three_digit_numbers = 171 := by
  sorry

end three_digit_number_count_l788_788456


namespace truck_600_units_time_l788_788044

noncomputable def truck_travel_time (speed_kmh : ℝ) (distance_units : ℝ) (U : ℝ) : ℝ :=
  distance_units / ((speed_kmh * U) / 3600)

theorem truck_600_units_time (U : ℝ) (hU : U ≠ 0) : 
  truck_travel_time 108 600 U = 20000 / U := 
by
  unfold truck_travel_time
  have h1 : (108 * U) / 3600 = 0.03 * U := by norm_num
  rw [h1, mul_div_assoc, div_eq_mul_one_div]
  field_simp [hU]
  ring


end truck_600_units_time_l788_788044


namespace circles_tangent_to_both_l788_788912

-- Definitions for the circles and their properties
structure Circle (α : Type) :=
(center : α × α)
(radius : ℝ)

def tangent (C1 C2 : Circle ℝ) : Prop :=
  let d := ((C1.center.1 - C2.center.1) ^ 2 + (C1.center.2 - C2.center.2) ^ 2).sqrt
  d = C1.radius + C2.radius

-- Constants for circles C1 and C2
def C1 : Circle ℝ := ⟨(0, 0), 2⟩
def C2 : Circle ℝ := ⟨(4, 0), 2⟩

-- Condition that C1 and C2 are tangent
lemma C1_C2_tangent : tangent C1 C2 :=
by {
  unfold tangent,
  simp,
  sorry
}

-- Main theorem statement
theorem circles_tangent_to_both : ∃ C : set (Circle ℝ), 
  C.count = 6 ∧ ∀ C' ∈ C, C'.radius = 4 ∧ (tangent C1 C' ∧ tangent C2 C') :=
begin
  sorry
end

end circles_tangent_to_both_l788_788912


namespace pile_division_660_stones_l788_788970

theorem pile_division_660_stones (n : ℕ) (heaps : List ℕ) :
  n = 660 ∧ heaps.length = 30 ∧ ∀ x ∈ heaps, ∀ y ∈ heaps, (x ≤ 2 * y ∧ y ≤ 2 * x) →
  ∃ heaps : List ℕ, n = 660 ∧ heaps.length = 30 ∧ (∀ x y ∈ heaps, x ≤ 2 * y ∧ y ≤ 2 * x) :=
by
  sorry

end pile_division_660_stones_l788_788970


namespace smallest_possible_n_l788_788208

theorem smallest_possible_n (n : ℕ) (h : lcm 60 n / gcd 60 n = 60) : n = 16 :=
sorry

end smallest_possible_n_l788_788208


namespace quotient_of_integers_l788_788207

variable {x y : ℤ}

theorem quotient_of_integers (h : 1996 * x + y / 96 = x + y) : 
  (x / y = 1 / 2016) ∨ (y / x = 2016) := by
  sorry

end quotient_of_integers_l788_788207


namespace min_value_a_2b_3c_l788_788401

theorem min_value_a_2b_3c (a b c : ℝ)
  (h : ∀ x y : ℝ, x + 2 * y - 3 ≤ a * x + b * y + c ∧ a * x + b * y + c ≤ x + 2 * y + 3) :
  a + 2 * b - 3 * c ≥ -2 :=
sorry

end min_value_a_2b_3c_l788_788401


namespace final_concentration_is_10_percent_l788_788725

variable (V_sal : ℝ) (V_cup : ℝ) (V_large : ℝ) (V_medium : ℝ)
variable (V_small : ℝ) (C_initial : ℝ) (C_final : ℝ)

-- Conditions
def saline_initial_concentration : Prop := C_initial = 0.15

def volume_ratios : Prop := V_large / V_medium = 2 ∧ 
                           V_medium / V_small = 5 / 3

def small_ball_displacement : Prop := 0.1 * V_cup = V_small

def volume_cup : Prop := V_cup = 30 * V_small

-- Expected final concentration
def final_saline_concentration : Prop := C_final = 0.10

-- Prove that the final concentration is 10% given all conditions
theorem final_concentration_is_10_percent :
  saline_initial_concentration →
  volume_ratios →
  small_ball_displacement →
  volume_cup →
  final_saline_concentration :=
  by
    intros
    sorry

end final_concentration_is_10_percent_l788_788725


namespace det_rotation_75_degrees_l788_788529

def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]

theorem det_rotation_75_degrees :
  Matrix.det (rotation_matrix (Real.pi / 180 * 75)) = 1 :=
by
  sorry

end det_rotation_75_degrees_l788_788529


namespace find_f_l788_788536

theorem find_f (f : ℝ → ℝ) (h₀ : f 0 = 1) (h₁ : ∀ x y, f (x * y) = f ((x^2 + y^2) / 2) + (x - y)^2) : 
  ∀ x, f x = 1 - 2 * x :=
by
  sorry  -- Proof not required

end find_f_l788_788536


namespace largest_non_representable_intro_l788_788890

-- Define the coin denominations
def coin_denominations (n : ℕ) : List ℕ :=
  List.map (λ i => 2^(n-i) * 3^i) (List.range (n+1))

-- Define when a number s is n-representable
def n_representable (s n : ℕ) : Prop :=
  ∃ (counts : List ℕ), counts.length = n + 1 ∧
    s = List.sum (List.map (λ (i : ℕ × ℕ) => (coin_denominations n).nthLE i.1 sorry * i.2) counts.enum)

-- Define the largest non-representable amount
def largest_non_representable (n : ℕ) : ℕ :=
  3^(n+1) - 2^(n+2)

-- The proof statement
theorem largest_non_representable_intro (n : ℕ) :
  ∀ s, (s < largest_non_representable n → ¬ n_representable s n) ∧ 
       (s > largest_non_representable n → n_representable s n) := 
sorry

end largest_non_representable_intro_l788_788890


namespace smallest_solution_is_39_over_8_l788_788366

noncomputable def smallest_solution (x : ℝ) : Prop :=
  (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ (x ≠ 0) ∧ (x ≠ 3)

theorem smallest_solution_is_39_over_8 : ∃ x > 0, smallest_solution x ∧ x = 39 / 8 :=
by
  sorry

end smallest_solution_is_39_over_8_l788_788366


namespace sum_of_squared_residuals_l788_788440

theorem sum_of_squared_residuals :
  let y_pred := λ x : ℝ, 2 * x + 1
  let data := [(2, 4.9), (3, 7.1), (4, 9.1)]
  let residuals := data.map (λ (xy : ℝ × ℝ), xy.snd - y_pred xy.fst)
  let squared_residuals := residuals.map (λ e, e ^ 2)
  squared_residuals.sum = 0.03 :=
by
  sorry

end sum_of_squared_residuals_l788_788440


namespace sin_double_angle_l788_788808

theorem sin_double_angle (a : ℝ) (ha : a > 0) :
  let P := (-4 * a, 3 * a) in
  ∃ (θ : ℝ), 
    let sinθ := (3 * a) / (Real.sqrt ((-4 * a) ^ 2 + (3 * a) ^ 2)) in
    let cosθ := (-4 * a) / (Real.sqrt ((-4 * a) ^ 2 + (3 * a) ^ 2)) in
    sin (2 * θ) = -24 / 25 :=
by
  sorry

end sin_double_angle_l788_788808


namespace calculate_perimeter_of_staircase_region_l788_788068

-- Define the properties and dimensions of the staircase-shaped region
def is_right_angle (angle : ℝ) : Prop := angle = 90

def congruent_side_length : ℝ := 1

def bottom_base_length : ℝ := 12

def total_area : ℝ := 78

def perimeter_region : ℝ := 34.5

theorem calculate_perimeter_of_staircase_region
  (is_right_angle : ∀ angle, is_right_angle angle)
  (congruent_sides_count : ℕ := 12)
  (total_congruent_side_length : ℝ := congruent_sides_count * congruent_side_length)
  (bottom_base_length : ℝ)
  (total_area : ℝ)
  : bottom_base_length = 12 ∧ total_area = 78 → 
    ∃ perimeter : ℝ, perimeter = 34.5 :=
by
  admit -- Proof goes here

end calculate_perimeter_of_staircase_region_l788_788068


namespace a_can_work_alone_in_14_days_l788_788212

-- Definitions for conditions given in the problem
def B_days_work := 10.5
def together_days_work := 6

-- Main theorem to prove that A can do the work alone in 14 days
theorem a_can_work_alone_in_14_days :
  ∃ (A_days_work : ℝ),
    (B_days_work = 10.5 ∧ together_days_work = 6) →
    A_days_work = 14 := 
begin
  sorry
end

end a_can_work_alone_in_14_days_l788_788212


namespace average_score_is_7_stddev_is_2_l788_788246

-- Define the scores list
def scores : List ℝ := [7, 8, 7, 9, 5, 4, 9, 10, 7, 4]

-- Proof statement for average score
theorem average_score_is_7 : (scores.sum / scores.length) = 7 :=
by
  simp [scores]
  sorry

-- Proof statement for standard deviation
theorem stddev_is_2 : Real.sqrt ((scores.map (λ x => (x - (scores.sum / scores.length))^2)).sum / scores.length) = 2 :=
by
  simp [scores]
  sorry

end average_score_is_7_stddev_is_2_l788_788246


namespace product_not_perfect_square_l788_788569

theorem product_not_perfect_square :
  ∀ a b : ℤ, a = 2^1917 + 1 → b = 2^1991 - 1 → 
  let product := ∏ k in Finset.range (b + 1 - a).natAbs, a + k in
  ¬∃ (n : ℤ), product = n^2 :=
by
  intros a b ha hb
  have product := ∏ k in Finset.range (b + 1 - a).natAbs, a + k
  sorry

end product_not_perfect_square_l788_788569


namespace find_a_l788_788109

open ProbabilityTheory

noncomputable def random_variable : Type := sorry -- Assuming we define a random variable type

def xi (rv : random_variable) : distribution := normal 2 4 -- xi is normally distributed with mean 2 and variance 4

theorem find_a (rv : random_variable) (h1 : xi rv = normal 2 4) :
  ∃ a : ℝ, a = 5/3 ∧
  P(xi rv > a + 2) = P(xi rv < 2 * a - 3) :=
sorry

end find_a_l788_788109


namespace piecewise_function_solution_l788_788107

def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then sqrt x else
  if x ≥ 1 then 2 * (x - 1) else 0

theorem piecewise_function_solution (a : ℝ) (h : f a = f (a + 1)) :
  a = 1 / 4 ∧ f (1 / a) = 6 :=
by
  sorry

end piecewise_function_solution_l788_788107


namespace solve_quadratic_equation_l788_788147

theorem solve_quadratic_equation (x : ℝ) :
  (6 * x^2 - 3 * x - 1 = 2 * x - 2) ↔ (x = 1 / 3 ∨ x = 1 / 2) :=
by sorry

end solve_quadratic_equation_l788_788147


namespace geometric_progression_common_ratio_l788_788876

-- Definitions and theorems
variable {α : Type*} [OrderedCommRing α]

theorem geometric_progression_common_ratio
  (a : α) (r : α)
  (h_pos : a > 0)
  (h_geometric : ∀ n : ℕ, a * r^n = (a * r^(n + 1)) * (a * r^(n + 2))):
  r = 1 := by
  sorry

end geometric_progression_common_ratio_l788_788876


namespace largest_non_representable_intro_l788_788891

-- Define the coin denominations
def coin_denominations (n : ℕ) : List ℕ :=
  List.map (λ i => 2^(n-i) * 3^i) (List.range (n+1))

-- Define when a number s is n-representable
def n_representable (s n : ℕ) : Prop :=
  ∃ (counts : List ℕ), counts.length = n + 1 ∧
    s = List.sum (List.map (λ (i : ℕ × ℕ) => (coin_denominations n).nthLE i.1 sorry * i.2) counts.enum)

-- Define the largest non-representable amount
def largest_non_representable (n : ℕ) : ℕ :=
  3^(n+1) - 2^(n+2)

-- The proof statement
theorem largest_non_representable_intro (n : ℕ) :
  ∀ s, (s < largest_non_representable n → ¬ n_representable s n) ∧ 
       (s > largest_non_representable n → n_representable s n) := 
sorry

end largest_non_representable_intro_l788_788891


namespace part1_part2_l788_788849

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  (a 1 = 1) ∧ 
  (a 2 = 8) ∧ 
  (∀ n ≥ 2, a (n + 1) = (4 / n) * a n + a (n - 1))

theorem part1 (a : ℕ → ℝ) (h : sequence a) : 
  ∃ c : ℝ, ∀ n : ℕ, n ≥ 1 → a n ≤ c * n^2 :=
by {
  sorry
}

theorem part2 (a : ℕ → ℝ) (h : sequence a) :
  ∀ n : ℕ, n > 0 → a (n + 1) - a n ≤ 4 * n + 3 :=
by {
  sorry
}

end part1_part2_l788_788849


namespace div_power_sub_one_l788_788558

theorem div_power_sub_one : 11 * 31 * 61 ∣ 20^15 - 1 := 
by
  sorry

end div_power_sub_one_l788_788558


namespace minimize_triangle_area_eqn_l788_788234

/-- 
Given a line l passing through point P(2,1) and intersecting the positive half-axes of the coordinate axes at points A and B,
prove that the equation of the line when the area of triangle AOB is minimized is x + 2y - 4 = 0. 
--/
theorem minimize_triangle_area_eqn (a b : ℝ) 
  (h1 : 0 < a)
  (h2 : 0 < b) 
  (h3 : (2 : ℝ) / a + 1 / b = 1)
  (h4 : a * b = 8) :
  ∃ (c d e : ℝ), (c = 1) ∧ (d = 2) ∧ (e = 4) ∧ (x + 2 * y - 4 = 0) :=
begin
  sorry
end

end minimize_triangle_area_eqn_l788_788234


namespace bus_time_l788_788112

variable (t1 t2 t3 t4 : ℕ)

theorem bus_time
  (h1 : t1 = 25)
  (h2 : t2 = 40)
  (h3 : t3 = 15)
  (h4 : t4 = 10) :
  t1 + t2 + t3 + t4 = 90 := by
  sorry

end bus_time_l788_788112


namespace arithmetic_mean_calculation_l788_788269

-- Let n be a natural number greater than 2, and let the set contain n numbers.
-- Two of these numbers are 1 - 1/n and 1 - 2/n, and the remaining n-2 numbers are all 1.
-- Prove that the arithmetic mean of these n numbers is 1 - 3/n^2.

theorem arithmetic_mean_calculation (n : ℕ) (h : n > 2) :
  let a := (1 - 1 / (n : ℝ))
  let b := (1 - 2 / (n : ℝ))
  let c := (of_nat (n-2) : ℝ)
  let total := c + a + b
  let mean := total / (n : ℝ)
  mean = 1 - 3 / (n : ℝ)^2 :=
by sorry
 
end arithmetic_mean_calculation_l788_788269


namespace projection_onto_vector_l788_788325

noncomputable def projection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  ![\[9 / 25, 12 / 25\], \[12 / 25, 16 / 25\]]

theorem projection_onto_vector:
    ∀ (x y : ℚ), (Matrix.mul_vec projection_matrix ![\x, \y]) = ![(9 * x + 12 * y) / 25, (12 * x + 16 * y) / 25] := by
  sorry

end projection_onto_vector_l788_788325


namespace right_triangle_area_and_perimeter_l788_788878

theorem right_triangle_area_and_perimeter (leg1 leg2 : ℝ) (h1 : leg1 = 30) (h2 : leg2 = 45) :
  let area := (1 / 2) * leg1 * leg2
  let hypotenuse := Real.sqrt (leg1^2 + leg2^2)
  let perimeter := leg1 + leg2 + hypotenuse
  area = 675 ∧ perimeter = 129 := 
by
  unfold area hypotenuse perimeter
  rw [h1, h2]
  norm_num
  sorry

end right_triangle_area_and_perimeter_l788_788878


namespace no_fixed_points_range_l788_788795

def no_fixed_points (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 1 ≠ x

theorem no_fixed_points_range (a : ℝ) : no_fixed_points a ↔ -1 < a ∧ a < 3 := by
  sorry

end no_fixed_points_range_l788_788795


namespace allocation_methods_count_l788_788230

theorem allocation_methods_count (total_warriors : ℕ) (tasks : ℕ) (capt_vs_vice : Fin 2 → Fin 6) 
  (remaining_warriors : Fin 4 → Fin 4) : 
  total_warriors = 6 ∧ tasks = 4 → 
  (∃ capt_vs_vice_method : Bool, ∃ remaining_warriors_method : Fin 3 → Fin 4, 
  ∃ task_assignment_method : Fin 4 → Fin 4, 
  2 * 4.choose(3) * 4.factorial = 192) :=
by
  intro h
  use (true) -- Choose one of the captain or vice-captain
  use (λ _, ⟨0, λ _, 0⟩) -- Choose 3 people from the remaining 4
  use (λ _, ⟨0, λ _, 0⟩) -- Assign these 4 to 4 tasks
  sorry

end allocation_methods_count_l788_788230


namespace total_number_of_candles_l788_788904

theorem total_number_of_candles
  (candles_bedroom : ℕ)
  (candles_living_room : ℕ)
  (candles_donovan : ℕ)
  (h1 : candles_bedroom = 20)
  (h2 : candles_bedroom = 2 * candles_living_room)
  (h3 : candles_donovan = 20) :
  candles_bedroom + candles_living_room + candles_donovan = 50 :=
by
  sorry

end total_number_of_candles_l788_788904


namespace original_sandbox_capacity_l788_788228

theorem original_sandbox_capacity :
  ∃ (L W H : ℝ), 8 * (L * W * H) = 80 → L * W * H = 10 :=
by
  sorry

end original_sandbox_capacity_l788_788228


namespace count_four_digit_integers_l788_788454

theorem count_four_digit_integers :
  let is_valid (n : ℤ) := 
    1000 ≤ n ∧ n < 10000 ∧ 
    n % 7 = 3 ∧ 
    n % 10 = 6 ∧ 
    n % 12 = 8 ∧ 
    n % 13 = 2
  in {n : ℕ | is_valid n}.card = PICK_AMONG_CHOICES
:= sorry

end count_four_digit_integers_l788_788454


namespace parabola_equation_circle_m_equation_line_a2a3_tangent_to_circle_m_l788_788630

noncomputable theory
open_locale classical

-- Definitions and conditions
def parabola_vertex_origin (x y : ℝ) : Prop := ∃ p : ℝ, p > 0 ∧ y^2 = 2 * p * x
def line_intersects_parabola_perpendicularly : Prop :=
  ∃ p : ℝ, p = 1 / 2 ∧ parabola_vertex_origin 1 p

def circle_m_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1
def line_tangent_to_circle_m (l : ℝ → ℝ) : Prop := ∀ x y : ℝ, circle_m_eq x y → l x = y

def points_on_parabola_and_tangent (A1 A2 A3 : ℝ × ℝ) : Prop :=
  parabola_vertex_origin A1.1 A1.2 ∧
  parabola_vertex_origin A2.1 A2.2 ∧
  parabola_vertex_origin A3.1 A3.2 ∧
  line_tangent_to_circle_m (λ y, A1.1 * y + A1.2) ∧
  line_tangent_to_circle_m (λ y, A1.1 * y + A3.2)

-- Statements to prove
theorem parabola_equation : ∃ C : ℝ → ℝ → Prop, (C = parabola_vertex_origin) := sorry
theorem circle_m_equation : ∃ M : ℝ → ℝ → Prop, (M = circle_m_eq) := sorry
theorem line_a2a3_tangent_to_circle_m :
  ∀ A1 A2 A3 : ℝ × ℝ, 
  (points_on_parabola_and_tangent A1 A2 A3) →
  ∃ l : ℝ → ℝ, line_tangent_to_circle_m l := sorry

end parabola_equation_circle_m_equation_line_a2a3_tangent_to_circle_m_l788_788630


namespace circumcenter_AEF_on_AB_l788_788812

open EuclideanGeometry

variables {A B C H E F : Point}
variable {ABC : Triangle A B C}

-- Conditions
axiom h_acute_angled (ABC : Triangle A B C) : acute_triangle ABC
axiom h_ab_gt_ac (ABC : Triangle A B C) : length_side AB > length_side AC
axiom h_orthocenter (H : Point) (ABC : Triangle A B C) : orthocenter ABC H
axiom h_reflect_C (E : Point) (C H : Point) (l : Line) : is_reflection C l E
axiom h_ef_intersect_ac (F : Point) (E H : Point) (AC : Line) : line_intersection EH AC F

-- Prove that the circumcenter of triangle AEF lies on line AB
theorem circumcenter_AEF_on_AB (ABC : Triangle A B C) (H E F : Point) :
  orthocenter ABC H →
  is_reflection C (altitude ABC H) E →
  line_intersection (line_through E H) (line_through A C) F →
  circumcenter (triangle A E F) ∈ line_through A B :=
begin
  sorry
end

end circumcenter_AEF_on_AB_l788_788812


namespace wxyz_sum_l788_788094

noncomputable def wxyz (w x y z : ℕ) := 2^w * 3^x * 5^y * 7^z

theorem wxyz_sum (w x y z : ℕ) (h : wxyz w x y z = 1260) : w + 2 * x + 3 * y + 4 * z = 13 :=
sorry

end wxyz_sum_l788_788094


namespace flyers_left_l788_788503

theorem flyers_left (initial_flyers : ℕ) (jack_flyers : ℕ) (rose_flyers : ℕ) (left_flyers : ℕ) :
  initial_flyers = 1236 →
  jack_flyers = 120 →
  rose_flyers = 320 →
  left_flyers = 796 →
  initial_flyers - (jack_flyers + rose_flyers) = left_flyers := 
by
  intros h_initial h_jack h_rose h_left
  rw [h_initial, h_jack, h_rose, h_left]
  simp
  sorry

end flyers_left_l788_788503


namespace find_k_l788_788712

-- Define the conditions
variables (x y k : ℕ)
axiom part_sum : x + y = 36
axiom first_part : x = 19
axiom value_eq : 8 * x + k * y = 203

-- Prove that k is 3
theorem find_k : k = 3 :=
by
  -- Insert your proof here
  sorry

end find_k_l788_788712


namespace sum_of_coprime_numbers_l788_788225

theorem sum_of_coprime_numbers (A B C : ℕ)
  (h_coprime1 : Nat.coprime A B)
  (h_coprime2 : Nat.coprime B C)
  (h_prod1 : A * B = 551)
  (h_prod2 : B * C = 1073) :
  A + B + C = 85 :=
sorry

end sum_of_coprime_numbers_l788_788225


namespace claire_needs_80_tiles_l788_788764

def room_length : ℕ := 14
def room_width : ℕ := 18
def border_width : ℕ := 2
def small_tile_side : ℕ := 1
def large_tile_side : ℕ := 3

def num_small_tiles : ℕ :=
  let perimeter_length := (2 * (room_width - 2 * border_width))
  let perimeter_width := (2 * (room_length - 2 * border_width))
  let corner_tiles := (2 * border_width) * 4
  perimeter_length + perimeter_width + corner_tiles

def num_large_tiles : ℕ :=
  let inner_length := room_length - 2 * border_width
  let inner_width := room_width - 2 * border_width
  let inner_area := inner_length * inner_width
  Nat.ceil (inner_area / (large_tile_side * large_tile_side))

theorem claire_needs_80_tiles : num_small_tiles + num_large_tiles = 80 :=
by sorry

end claire_needs_80_tiles_l788_788764


namespace parabola_focus_directrix_eqn_l788_788735

theorem parabola_focus_directrix_eqn :
  let focus : (ℝ × ℝ) := (2, -1)
  let directrix : ℝ × ℝ → Prop := λ p, 5 * p.1 + 4 * p.2 = 20
  ∃ a b c d e f : ℤ, a > 0 ∧ Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.natAbs a) (Int.natAbs b)) (Int.natAbs c)) (Int.natAbs d)) (Int.natAbs e)) (Int.natAbs f) = 1 ∧
  (∀ x y : ℝ, (41 * x^2 + 41 * y^2 - 164 * x + 82 * y + 205 = 25 * x^2 + 40 * x * y + 16 * y^2 - 200 * x - 160 * y + 400) →
  16 * x^2 + 25 * y^2 + 36 * x + 242 * y - 195 = 0) :=
by
  let focus := (2, -1)
  let directrix := λ p, 5 * p.1 + 4 * p.2 = 20
  existsi 16, 0, 25, 36, 242, -195
  refine ⟨by decide, by decide,
    (λ x y (h : 41 * x^2 + 41 * y^2 - 164 * x + 82 * y + 205 = 25 * x^2 + 40 * x * y + 16 * y^2 - 200 * x - 160 * y + 400), _⟩,
  sorry

end parabola_focus_directrix_eqn_l788_788735


namespace jessica_balloons_l788_788900

-- Defining the number of blue balloons Joan, Sally, and the total number.
def balloons_joan : ℕ := 9
def balloons_sally : ℕ := 5
def balloons_total : ℕ := 16

-- The statement to prove that Jessica has 2 blue balloons
theorem jessica_balloons : balloons_total - (balloons_joan + balloons_sally) = 2 :=
by
  -- Using the given information and arithmetic, we can show the main statement
  sorry

end jessica_balloons_l788_788900


namespace perimeter_of_square_l788_788154

-- Given conditions
variables {x y : ℝ} (h1 : x - y = 5) (h2 : x * y > 0)

theorem perimeter_of_square (h : (∃ s : ℝ, s^2 = 5 * (x * y))) : 
  ∃ p : ℝ, p = 4 * Real.sqrt (5 * x * y) :=
by
  obtain ⟨s, hs⟩ := h
  use 4 * s
  rw hs
  congr
  field_simp [Real.sqrt_mul (by norm_num : (5 : ℝ)) (x * y)]
  sorry

end perimeter_of_square_l788_788154


namespace tourist_growth_rate_l788_788776

theorem tourist_growth_rate (F : ℝ) (x : ℝ) 
    (hMarch : F * 0.6 = 0.6 * F)
    (hApril : F * 0.6 * 0.5 = 0.3 * F)
    (hMay : 2 * F = 2 * F):
    (0.6 * 0.5 * (1 + x) = 2) :=
by
  sorry

end tourist_growth_rate_l788_788776


namespace inequality_with_distances_equality_condition_l788_788948

noncomputable def distances_from_point (P A B C : Point) : ℝ × ℝ × ℝ :=
  let u := distance P A
  let v := distance P B
  let w := distance P C
  (u, v, w)

noncomputable def area_of_triangle (A B C : Point) : ℝ := sorry

noncomputable def tan_of_angles (A B C : Point) : ℝ × ℝ × ℝ := sorry

theorem inequality_with_distances (P A B C : Point) (h_acute : is_acute_triangle A B C) :
  let (u, v, w) := distances_from_point P A B C
  let (tanA, tanB, tanC) := tan_of_angles A B C
  u^2 * tanA + v^2 * tanB + w^2 * tanC ≥ 4 * area_of_triangle A B C :=
sorry

theorem equality_condition (P A B C : Point) (h_acute : is_acute_triangle A B C) :
  let (u, v, w) := distances_from_point P A B C
  let (tanA, tanB, tanC) := tan_of_angles A B C
  u^2 * tanA + v^2 * tanB + w^2 * tanC = 4 * area_of_triangle A B C ↔ P = orthocenter_of_triangle A B C :=
sorry

end inequality_with_distances_equality_condition_l788_788948


namespace OH_squared_l788_788915

variables {O H A B C : Type} [inner_product_space ℝ O]

def circumcenter (a b c : ℝ) : Type := -- Definition of circumcenter (e.g., type class for properties)
 sorry -- shared space with orthocenter and triangle sides

def orthocenter (a b c : ℝ) : Type := -- Definition of orthocenter (e.g., type class for properties)
 sorry -- shared space with circumcenter and triangle sides

variables (a b c R : ℝ) (triangle : circumcenter a b c) -- Defining triangle properties
variables (orthotriangle : orthocenter a b c) -- Defining orthotriangle within the triangle properties

theorem OH_squared 
  (hR : R = 5)
  (h_side_sum : a^2 + b^2 + c^2 = 50) : 
  let OH_squared := 
    (3 * R^2 + 2 * (R^2 - (a^2 + b^2 + c^2) / 2)) in
  OH_squared = 75 :=
by
  sorry

end OH_squared_l788_788915


namespace inequality_properties_l788_788043

variable {a b c : ℝ}

theorem inequality_properties (h1 : a > b) (h2 : b > 0) : 
  (a + c > b + c) ∧ 
  (a^2 > b^2) ∧ 
  (sqrt a > sqrt b) ∧ 
  ¬ (ac > bc) := by
  sorry

end inequality_properties_l788_788043


namespace least_number_of_cans_l788_788688

theorem least_number_of_cans (maaza : ℕ) (pepsi : ℕ) (sprite : ℕ) (gcd_val : ℕ) (total_cans : ℕ)
  (h1 : maaza = 50) (h2 : pepsi = 144) (h3 : sprite = 368) (h_gcd : gcd maaza (gcd pepsi sprite) = gcd_val)
  (h_total_cans : total_cans = maaza / gcd_val + pepsi / gcd_val + sprite / gcd_val) :
  total_cans = 281 :=
sorry

end least_number_of_cans_l788_788688


namespace problem_statement_l788_788756

noncomputable def f (x : ℝ) : ℝ := 3^x + 3^(-x)

noncomputable def g (x : ℝ) : ℝ := 3^x - 3^(-x)

theorem problem_statement : 
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, g (-x) = -g x) :=
by {
  sorry
}

end problem_statement_l788_788756


namespace lighthouse_height_l788_788661

theorem lighthouse_height : 
  ∃ (h : ℝ), 
  let d1 := h * Real.sqrt 3,
  let d2 := h in
  d1 + d2 = 273.2050807568877 ↔ h = 100 :=
by
  sorry

end lighthouse_height_l788_788661


namespace train_crossing_time_approx_l788_788244

noncomputable def train_length : ℝ := 80  -- Length of the train in meters
noncomputable def train_speed_kmh : ℝ := 48  -- Speed of the train in km/hr
noncomputable def conversion_factor := 1000 / 3600  -- Conversion factor from km/hr to m/s
noncomputable def train_speed_ms : ℝ := train_speed_kmh * conversion_factor  -- Speed of the train in m/s
noncomputable def crossing_time : ℝ := train_length / train_speed_ms  -- Time to cross the telegraph post in seconds

-- The theorem to prove
theorem train_crossing_time_approx : crossing_time ≈ 6 :=
by
  sorry

end train_crossing_time_approx_l788_788244


namespace logan_drove_5_hours_l788_788585

open Real

/-- Conditions from the problem -/
def t_tamika : ℝ := 8
def s_tamika : ℝ := 45
def s_logan : ℝ := 55
def d_tamika : ℝ := t_tamika * s_tamika := by
  unfold t_tamika s_tamika
  exact 45 * 8 -- Reason: just unfolding values

def d_logan : ℝ := d_tamika - 85 := by
  unfold d_tamika
  exact (45 * 8) - 85 -- Reason: just unfolding values

def hours_drove_logan (t_logan : ℝ) := t_logan = d_logan / s_logan

theorem logan_drove_5_hours : hours_drove_logan 5 :=
  by
  unfold hours_drove_logan d_logan s_logan
  sorry

end logan_drove_5_hours_l788_788585


namespace remainder_of_poly1_div_poly2_l788_788672

-- Definitions for the problem.
def poly1 : Polynomial ℤ := (Polynomial.X + 1) ^ 2011
def poly2 : Polynomial ℤ := Polynomial.X ^ 2 - Polynomial.X + 1

-- Result statement which states the remainder when poly1 is divided by poly2
theorem remainder_of_poly1_div_poly2 : (poly1 % poly2) = Polynomial.X :=
by 
  sorry

end remainder_of_poly1_div_poly2_l788_788672


namespace largest_divisor_poly_l788_788198

-- Define the polynomial and the required properties
def poly (n : ℕ) : ℕ := (n+1) * (n+3) * (n+5) * (n+7) * (n+11)

-- Define the conditions and the proof statement
theorem largest_divisor_poly (n : ℕ) (h_even : n % 2 = 0) : ∃ d, d = 15 ∧ ∀ m, m ∣ poly n → m ≤ d :=
by
  sorry

end largest_divisor_poly_l788_788198


namespace projection_matrix_correct_l788_788315

variables {R : Type*} [field R] [decidable_eq R]
variables (x y : R)
def vector_v : matrix (fin 2) (fin 1) R := ![![3], ![4]]
def vector_u : matrix (fin 2) (fin 1) R := ![![x], ![y]]
def projection_matrix : matrix (fin 2) (fin 2) R := ![![9/25, 12/25], ![12/25, 16/25]]

theorem projection_matrix_correct :
  (projection_matrix R) ⬝ (vector_u x y) = (25 : R)⁻¹ • (transpose (vector_v 3 4) ⬝ (vector_u x y)) ⬝ (vector_v 3 4) := 
sorry

end projection_matrix_correct_l788_788315


namespace order_of_magnitude_l788_788040

theorem order_of_magnitude (a b : ℝ) (h1 : a > 0) (h2 : b < 0) (h3 : |a| < |b|) :
  -b > a ∧ a > -a ∧ -a > b := by
  sorry

end order_of_magnitude_l788_788040


namespace measure_of_y_l788_788200

theorem measure_of_y (y : ℕ) (h₁ : 40 + 2 * y + y = 180) : y = 140 / 3 :=
by
  sorry

end measure_of_y_l788_788200


namespace train_passing_time_l788_788243

-- Define the conditions
def length_of_train : ℝ := 100
def speed_of_train : ℝ := 68
def speed_of_man : ℝ := 8
def relative_speed := (speed_of_train - speed_of_man) * 1000 / 3600  -- converting to m/s

-- Define the correct answer
def time_to_pass := length_of_train / relative_speed

-- The claim that needs to be proved
theorem train_passing_time : abs (time_to_pass - 6) < 1 :=
by
  sorry

end train_passing_time_l788_788243


namespace projection_onto_vector_is_expected_l788_788305

def projection_matrix (u: ℝ × ℝ) : matrix (fin 2) (fin 2) ℝ :=
  let ⟨x, y⟩ := u in 
  (1 / (x^2 + y^2)) • (matrix.col_vec u ⬝ (matrix.transpose (matrix.row_vec u)))

def expected_matrix : matrix (fin 2) (fin 2) ℝ :=
  ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]]

theorem projection_onto_vector_is_expected :
  projection_matrix (3, 4) = expected_matrix := by
  sorry

end projection_onto_vector_is_expected_l788_788305


namespace sum_of_three_pairwise_rel_prime_integers_l788_788653

theorem sum_of_three_pairwise_rel_prime_integers (a b c : ℕ)
  (h1: 1 < a) (h2: 1 < b) (h3: 1 < c)
  (prod: a * b * c = 216000)
  (rel_prime_ab : Nat.gcd a b = 1)
  (rel_prime_ac : Nat.gcd a c = 1)
  (rel_prime_bc : Nat.gcd b c = 1) : 
  a + b + c = 184 := 
sorry

end sum_of_three_pairwise_rel_prime_integers_l788_788653


namespace problem1_problem2_l788_788822

-- Define what it means for a function to be monotonically increasing or decreasing on a domain
def is_monotonically_increasing (f : ℝ → ℝ) (D : set ℝ) : Prop :=
  ∀ {x y}, x ∈ D → y ∈ D → x < y → f x < f y

def is_monotonically_decreasing (f : ℝ → ℝ) (D : set ℝ) : Prop :=
  ∀ {x y}, x ∈ D → y ∈ D → x < y → f x > f y

-- Define what it means for a function to be closed on a domain
def is_closed_function (f : ℝ → ℝ) (D : set ℝ) (a b : ℝ) : Prop :=
  is_monotonically_increasing f D ∨ is_monotonically_decreasing f D ∧
  [a, b] ⊆ D ∧
  ∀ y ∈ [f a, f b], ∃ x ∈ [a, b], f x = y

-- Proof problems
theorem problem1 :
  ¬ is_closed_function (λ x : ℝ, 3 ^ x) (set.Ioi 0) 0 0 :=
sorry

theorem problem2 (k : ℝ) (h : k < 0) :
  is_closed_function (λ x : ℝ, k + real.sqrt x) (set.Ioi 0) (k + real.sqrt 0) (k + real.sqrt 1) ↔ (-1 / 4 < k ∧ k < 0) :=
sorry

end problem1_problem2_l788_788822


namespace find_f_prime_one_l788_788386

def f (x : ℝ) : ℝ := 2 * x * (f 1) + Real.log x

theorem find_f_prime_one :
  deriv f 1 = -1 := sorry

end find_f_prime_one_l788_788386


namespace sum_of_squares_l788_788135

theorem sum_of_squares (n : ℕ) : ∃ k : ℤ, (∃ a b : ℤ, k = a^2 + b^2) ∧ (∃ d : ℕ, d ≥ n) :=
by
  sorry

end sum_of_squares_l788_788135


namespace sum_S10_l788_788164

def sequence (n : ℕ) : ℚ := 1 / (n^2 + 2 * n)

def sum_sequence (n : ℕ) : ℚ := ∑ k in Finset.range (n + 1), sequence k

theorem sum_S10 :
  sum_sequence 10 = 1 / 2 * (3 / 2 - 1 / 11 - 1 / 12) :=
by
  sorry

end sum_S10_l788_788164


namespace valid_sequences_count_l788_788087

def Transformation (p : (ℝ × ℝ)) : Type :=
  { f : (ℝ × ℝ) → (ℝ × ℝ) // ∃ p': ℝ × ℝ, f p = p' }

def rotate90 (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, p.1)
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
def rotate270 (p : ℝ × ℝ) : ℝ × ℝ := (p.2, -p.1)
def reflectX (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def reflectY (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def scale1 (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2)

def isometries : list ((ℝ × ℝ) → (ℝ × ℝ)) :=
  [rotate90, rotate180, rotate270, reflectX, reflectY, scale1]

def Triangle (p1 p2 p3 : ℝ × ℝ) : Type := (p1, p2, p3)

noncomputable def T := Triangle (0, 0) (3, 0) (0, 4)

 def transformation_sequences (n : ℕ) : list (list ((ℝ × ℝ) → (ℝ × ℝ))) :=
  list.replicate n isometries

def apply_transformation (f: ((ℝ × ℝ) -> (ℝ × ℝ))) (T : Triangle): Triangle := 
  let ⟨p1, p2, p3⟩ := T 
  in (f p1, f p2, f p3)

noncomputable def apply_sequence (seq : list ((ℝ × ℝ) → (ℝ × ℝ))) (T : Triangle) : Triangle :=
  seq.foldl (λ t f, apply_transformation f t) T

noncomputable def return_to_original (seq : list ((ℝ × ℝ) → (ℝ × ℝ))) (T : Triangle) : bool :=
  apply_sequence seq T = T

noncomputable def count_valid_sequences (n : ℕ) (T : Triangle) : ℕ :=
  (transformation_sequences n).count (λ seq, return_to_original seq T)

theorem valid_sequences_count :
  count_valid_sequences 4 T = 30 :=
sorry

end valid_sequences_count_l788_788087


namespace complex_number_calculation_l788_788821

theorem complex_number_calculation (i : ℂ) (h : i * i = -1) : i^7 - 2/i = i := 
by 
  sorry

end complex_number_calculation_l788_788821


namespace min_value_frac_sum_min_value_squared_sum_minimum_value_frac_sum_minimum_value_squared_sum_l788_788826

theorem min_value_frac_sum :
  ∀ (x y : ℝ), 0 < x ∧ 0 < y ∧ 2 * x + y = 1 → (∀ v, ∃ x y: ℝ, v = (frac_sum)) :=
  
 theorem min_value_squared_sum :
  ∀ (x y : ℝ), 0 < x ∧ 0 < y ∧ 2 * x + y = 1 → (∀ v, ∃ x y: ℝ, v = (squared_sum)) :=

def frac_sum(x y : ℝ) := 2 / x + 1 / y

def squared_sum(x y : ℝ):= 4 * x ^ 2 + y ^ 2

theorem minimum_value_frac_sum :
  ∀ (x y : ℝ), 0 < x ∧ 0 < y ∧ 2 * x + y = 1 → 9 ≤ 2 / x + 1 / y :=
sorry

theorem minimum_value_squared_sum:
  ∀ (x y : ℝ), 0 < x ∧ 0 < y ∧ 2 * x + y = 1 → 4 * x ^ 2 + y ^ 2 ≥ 0.5 :=
sorry

end min_value_frac_sum_min_value_squared_sum_minimum_value_frac_sum_minimum_value_squared_sum_l788_788826


namespace paint_needed_ratio_l788_788657

theorem paint_needed_ratio (d : ℝ) : 
  let D := 7 * d in
  let A_small := Real.pi * (d / 2) ^ 2 in
  let A_large := Real.pi * (D / 2) ^ 2 in
  A_large = 49 * A_small :=
by
  let D := 7 * d
  let A_small := Real.pi * (d / 2) ^ 2
  let A_large := Real.pi * (D / 2) ^ 2
  sorry

end paint_needed_ratio_l788_788657


namespace c_perp_d_l788_788868

variables {ℝ : Type*} [linear_ordered_field ℝ]

variables (a b c d : ℝ^3) (λ μ : ℝ)
-- Assumptions
def c_perp_a : Prop := c ⬝ a = 0
def c_perp_b : Prop := c ⬝ b = 0
def d_eq : Prop := d = λ • a + μ • b
def λ_ne_0 : Prop := λ ≠ 0
def μ_ne_0 : Prop := μ ≠ 0

-- The theorem to prove
theorem c_perp_d (h1 : c_perp_a c a) (h2 : c_perp_b c b) (h3 : d_eq d (λ • a + μ • b)) (h4 : λ_ne_0 λ) (h5 : μ_ne_0 μ) :
  c ⬝ d = 0 := 
sorry

end c_perp_d_l788_788868


namespace problem1_problem2_problem3_l788_788425

open Real

-- Define the function f
def f (x : ℝ) : ℝ := 4^x - 2^x

-- Define the conditions for s and t
variables (s t : ℝ)
axiom h_condition : f s + f t = 0

-- Define a and b
def a : ℝ := 2^s + 2^t
def b : ℝ := 2^(s + t)

/-- Problem 1: The range of f(x) over [-1, 1] is [-1/4, 2] -/
theorem problem1 : set.range (λ (x : ℝ), f x) = Icc (-1 / 4) 2 :=
sorry

/-- Problem 2: The relationship is b = (a^2 - a) / 2 and the domain of a is (1, 2] -/
theorem problem2 : b = (a^2 - a) / 2 ∧ (1 < a ∧ a ≤ 2) :=
sorry

/-- Problem 3: The range of 8^s + 8^t is (1, 2] -/
theorem problem3 : set.range (λ (p : ℝ × ℝ), 8^p.1 + 8^p.2) = Icc 1 2 :=
sorry

end problem1_problem2_problem3_l788_788425


namespace joan_took_marbles_l788_788143

-- Each condition is used as a definition.
def original_marbles : ℕ := 86
def remaining_marbles : ℕ := 61

-- The theorem states that the number of marbles Joan took equals 25.
theorem joan_took_marbles : (original_marbles - remaining_marbles) = 25 := by
  sorry    -- Add sorry to skip the proof.

end joan_took_marbles_l788_788143


namespace sequence_formula_l788_788174

theorem sequence_formula (a : ℕ → ℚ) (h1 : a 1 = 1 / 2)
    (h2 : ∀ n ≥ 2, (∑ i in Finset.range n.succ, a (i + 1)) = n^2 * a n) :
    ∀ n : ℕ, n ≥ 1 → a n = 1 / (n * (n + 1)) :=
by
  sorry

end sequence_formula_l788_788174


namespace divide_660_stones_into_30_piles_l788_788986

theorem divide_660_stones_into_30_piles :
  ∃ (heaps : Fin 30 → ℕ),
    (∑ i, heaps i = 660) ∧
    (∀ i j, heaps i < 2 * heaps j ∨ heaps j < 2 * heaps i) :=
sorry

end divide_660_stones_into_30_piles_l788_788986


namespace de_morgan_union_de_morgan_inter_l788_788559

open Set

variable {α : Type*} (A B : Set α)

theorem de_morgan_union : ∀ (A B : Set α), 
  compl (A ∪ B) = compl A ∩ compl B := 
by 
  intro A B
  sorry

theorem de_morgan_inter : ∀ (A B : Set α), 
  compl (A ∩ B) = compl A ∪ compl B := 
by 
  intro A B
  sorry

end de_morgan_union_de_morgan_inter_l788_788559


namespace quadratic_two_distinct_real_roots_l788_788846

theorem quadratic_two_distinct_real_roots (k : ℝ) : 4 - 4 * k > 0 → k < 1 :=
by
  intro h
  have : 4 - 4 * k = 4 * (1 - k),
  rw [mul_sub],
  rw [mul_one],
  linarith,
  sorry

end quadratic_two_distinct_real_roots_l788_788846


namespace projection_matrix_l788_788347

theorem projection_matrix
  (x y : ℝ) :
  let v := ![3, 4]
  let proj_v := (v ⬝ ![x, y]) / (v ⬝ v) • v
  let proj_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
    ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]] in
  proj_v = proj_matrix.mul_vec ![x, y] :=
by
  let v := ![3, 4]
  let proj_v := (v ⬝ ![x, y]) / (v ⬝ v) • v
  let proj_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]]
  sorry

end projection_matrix_l788_788347


namespace solve_eq_norm_l788_788274

def vec_norm (v : ℝ × ℝ) := real.sqrt (v.1 * v.1 + v.2 * v.2)

def vec_sub (v1 v2 : ℝ × ℝ) := (v1.1 - v2.1, v1.2 - v2.2)

theorem solve_eq_norm (k : ℝ) :
  vec_norm (vec_sub (k * 3, k * -4) (5, 8)) = 5 * real.sqrt 13 →
  (k = 123 / 50 ∨ k = -191 / 50) :=
  sorry

end solve_eq_norm_l788_788274


namespace imaginary_part_of_exp_neg_pi_div_6_eq_neg_one_half_l788_788586

theorem imaginary_part_of_exp_neg_pi_div_6_eq_neg_one_half :
  (Complex.exp (-Complex.I * Real.pi / 6)).im = -1/2 := by
sorry

end imaginary_part_of_exp_neg_pi_div_6_eq_neg_one_half_l788_788586


namespace number_of_students_before_new_year_l788_788700

variables (M N k ℓ : ℕ)
hypotheses (h1 : 100 * M = k * N)
             (h2 : 100 * (M + 1) = ℓ * (N + 3))
             (h3 : ℓ < 100)

theorem number_of_students_before_new_year (h1 : 100 * M = k * N)
                                             (h2 : 100 * (M + 1) = ℓ * (N + 3))
                                             (h3 : ℓ < 100) :
  N ≤ 197 :=
sorry

end number_of_students_before_new_year_l788_788700


namespace cost_price_of_article_l788_788743

theorem cost_price_of_article
  (C SP1 SP2 : ℝ)
  (h1 : SP1 = 0.8 * C)
  (h2 : SP2 = 1.05 * C)
  (h3 : SP2 = SP1 + 100) : 
  C = 400 := 
sorry

end cost_price_of_article_l788_788743


namespace drive_time_is_eleven_hours_l788_788258

-- Define the distances and speed as constants
def distance_salt_lake_to_vegas : ℕ := 420
def distance_vegas_to_los_angeles : ℕ := 273
def average_speed : ℕ := 63

-- Calculate the total distance
def total_distance : ℕ := distance_salt_lake_to_vegas + distance_vegas_to_los_angeles

-- Calculate the total time required
def total_time : ℕ := total_distance / average_speed

-- Theorem stating Andy wants to complete the drive in 11 hours
theorem drive_time_is_eleven_hours : total_time = 11 := sorry

end drive_time_is_eleven_hours_l788_788258


namespace binom_expr_value_l788_788270

variables (x : ℝ) (k : ℕ)

noncomputable def binom_real (x : ℝ) (k : ℕ) : ℝ :=
  if h : k = 0 then 1
  else (list.range k).prod (λ i, x - i) / (k.factorial : ℝ)

theorem binom_expr_value :
  (binom_real (3/2) 10 * 3^10 / binom_real 20 10) = -1.243 :=
sorry

end binom_expr_value_l788_788270


namespace perpendicular_lines_value_of_a_l788_788045

theorem perpendicular_lines_value_of_a (a : ℝ) :
    (∃ a : ℝ, (∀ x y : ℝ, ax + 2y + 6 = 0) → (x + (a-1)y - 1 = 0))
    → a = 2/3 :=
by 
    sorry

end perpendicular_lines_value_of_a_l788_788045


namespace loaned_out_books_l788_788736

def initial_books : ℕ := 75
def added_books : ℕ := 10 + 15 + 6
def removed_books : ℕ := 3 + 2 + 4
def end_books : ℕ := 90
def return_percentage : ℝ := 0.80

theorem loaned_out_books (L : ℕ) :
  (end_books - initial_books = added_books - removed_books - ⌊(1 - return_percentage) * L⌋) →
  (L = 35) :=
sorry

end loaned_out_books_l788_788736


namespace zero_in_interval_l788_788426

noncomputable def f (a b x : ℝ) := log a x + x - b

theorem zero_in_interval (a b x₀ : ℝ) (h₀ : 2 < a) (h₁ : a < 3) (h₂ : 3 < b) (h₃ : b < 4)
(h₄ : f a b x₀ = 0) : 2 < x₀ ∧ x₀ < 3 :=
sorry

end zero_in_interval_l788_788426


namespace divide_660_stones_into_30_heaps_l788_788960

theorem divide_660_stones_into_30_heaps :
    ∃ (heaps : Fin 30 → ℕ), (∑ i, heaps i = 660) ∧ (∀ i j, heaps i < 2 * heaps j) ∨ (heaps j < 2 * heaps i) := 
sorry

end divide_660_stones_into_30_heaps_l788_788960


namespace chewing_gums_count_l788_788117

-- Given conditions
def num_chocolate_bars : ℕ := 55
def num_candies : ℕ := 40
def total_treats : ℕ := 155

-- Definition to be proven
def num_chewing_gums : ℕ := total_treats - (num_chocolate_bars + num_candies)

-- Theorem statement
theorem chewing_gums_count : num_chewing_gums = 60 :=
by 
  -- here would be the proof steps, but it's omitted as per the instruction
  sorry

end chewing_gums_count_l788_788117


namespace tangential_quadrilateral_and_incenter_l788_788892

variables {A B C D E K L M N : Type} 

-- Conditions: Let ABCD be a cyclic quadrilateral
def cyclic_quad (A B C D : Type) : Prop := sorry

-- Conditions: Let E be the intersection of diagonals AC and BD
def diagonals_intersect (A B C D E : Type) : Prop := sorry

-- Conditions: K, L, M, N are the reflections of E over the sides AB, BC, CD, and DA respectively
def reflections (A B C D E K L M N: Type) : Prop := sorry

-- Statement to prove
theorem tangential_quadrilateral_and_incenter 
  (A B C D E K L M N: Type) 
  (hCyclic: cyclic_quad A B C D) 
  (hIntersect: diagonals_intersect A B C D E) 
  (hReflections: reflections A B C D E K L M N) 
  : 
  (KLMN_tangential : tangential_quadrilateral K L M N) 
  (incenter_is_E : incenter K L M N = E) := 
sorry

end tangential_quadrilateral_and_incenter_l788_788892


namespace parabola_circle_properties_l788_788639

section ParabolaCircleTangent

variables {A1 A2 A3 P Q M : Point} 
variables {parabola : Parabola} 
variables {circle : Circle} 
variables {line_l : Line}

-- Definitions of points
def O := Point.mk 0 0
def M := Point.mk 2 0
def P := Point.mk 1 (Real.sqrt (2 * (1 / 2)))
def Q := Point.mk 1 (-Real.sqrt (2 * (1 / 2)))

-- Definition of geometrical constructs
def parabola := {p : Point // p.y^2 = p.x}
def circle := {c : Point // (c.x - 2)^2 + c.y^2 = 1}
def line_l := {l : Line // l.slope = ⊤ ∧ l.x_intercept = 1 }

-- Tangent properties for lines A1A2 and A1A3
def is_tangent {A B : Point} (l : Line) (circle : Circle) : Prop :=
  ∃ r: Real, (∥circle.center - A∥ = r) ∧ (∥circle.center - B∥ = r) ∧ (∥circle.center - (line.foot circle.center)∥ = r)

-- Theorem/Statement to prove:
theorem parabola_circle_properties :
  (parabola = {p : Point // p.y^2 = p.x}) →
  (circle = {c : Point // (c.x - 2)^2 + c.y^2 = 1}) →
  (∀ A1 A2 A3 : Point, A1 ∈ parabola → A2 ∈ parabola → A3 ∈ parabola → 
    (is_tangent (line_through A1 A2) circle) → (is_tangent (line_through A1 A3) circle) → 
    ⊥ ≤ distance_from_point_to_line (line_through A2 A3) circle.center = 1 ) :=
sorry

end ParabolaCircleTangent

end parabola_circle_properties_l788_788639


namespace probability_blue_then_green_l788_788152

-- Definitions based on the conditions
def faces := 12
def red_faces := 5
def blue_faces := 4
def yellow_faces := 2
def green_faces := 1

-- Probabilities based on the problem setup
def probability_blue := blue_faces / faces
def probability_green := green_faces / faces

-- Proof statement
theorem probability_blue_then_green :
  (probability_blue * probability_green) = (1 / 36) :=
by
  sorry

end probability_blue_then_green_l788_788152


namespace interest_percentage_of_selling_price_l788_788523

-- Define the given conditions
def face_value : ℝ := 5000
def interest_rate : ℝ := 0.05
def selling_price : ℝ := 3846.153846153846

-- Define the interest calculation
def interest : ℝ := face_value * interest_rate

-- The theorem to prove
theorem interest_percentage_of_selling_price :
  (interest / selling_price) * 100 ≈ 6.5 := by
  sorry

end interest_percentage_of_selling_price_l788_788523


namespace original_price_of_dish_l788_788518

theorem original_price_of_dish:
  ∃ P : ℝ, 
    let J_total := 0.9 * P + 0.15 * P in
    let J := 0.9 * P in
    let T := 0.15 * J in
    let Jane_total := J + T in
    J_total = Jane_total + 0.36 ∧ P = 24 :=
by
  sorry

end original_price_of_dish_l788_788518


namespace smallest_solution_to_equation_l788_788368

noncomputable def smallest_solution := (11 - Real.sqrt 445) / 6

theorem smallest_solution_to_equation:
  ∃ x : ℝ, (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ (x = smallest_solution) :=
sorry

end smallest_solution_to_equation_l788_788368


namespace parabola_circle_properties_l788_788640

section ParabolaCircleTangent

variables {A1 A2 A3 P Q M : Point} 
variables {parabola : Parabola} 
variables {circle : Circle} 
variables {line_l : Line}

-- Definitions of points
def O := Point.mk 0 0
def M := Point.mk 2 0
def P := Point.mk 1 (Real.sqrt (2 * (1 / 2)))
def Q := Point.mk 1 (-Real.sqrt (2 * (1 / 2)))

-- Definition of geometrical constructs
def parabola := {p : Point // p.y^2 = p.x}
def circle := {c : Point // (c.x - 2)^2 + c.y^2 = 1}
def line_l := {l : Line // l.slope = ⊤ ∧ l.x_intercept = 1 }

-- Tangent properties for lines A1A2 and A1A3
def is_tangent {A B : Point} (l : Line) (circle : Circle) : Prop :=
  ∃ r: Real, (∥circle.center - A∥ = r) ∧ (∥circle.center - B∥ = r) ∧ (∥circle.center - (line.foot circle.center)∥ = r)

-- Theorem/Statement to prove:
theorem parabola_circle_properties :
  (parabola = {p : Point // p.y^2 = p.x}) →
  (circle = {c : Point // (c.x - 2)^2 + c.y^2 = 1}) →
  (∀ A1 A2 A3 : Point, A1 ∈ parabola → A2 ∈ parabola → A3 ∈ parabola → 
    (is_tangent (line_through A1 A2) circle) → (is_tangent (line_through A1 A3) circle) → 
    ⊥ ≤ distance_from_point_to_line (line_through A2 A3) circle.center = 1 ) :=
sorry

end ParabolaCircleTangent

end parabola_circle_properties_l788_788640


namespace modern_art_museum_l788_788647

theorem modern_art_museum (V E U : ℕ) (h1 : V = (3/4 : ℚ) * V + 110) (h2 : E = U)
  (h3 : 110 = V - (3/4 : ℚ) * V) : V = 440 :=
by {
  -- Defining variables and conditions from the problem
  let total_visitors := V,
  let enjoyed_and_understood := (3/4 : ℚ) * V,
  let not_enjoyed_not_understood := 110,

  -- Given conditions
  have hf1 : total_visitors = enjoyed_and_understood + not_enjoyed_not_understood, from h1,
  have hf2 : E = U, from h2,
  have hf3 : not_enjoyed_not_understood = V - enjoyed_and_understood, from h3,

  -- From conditions, solve for V
  -- V - (3/4) * V = 110 reduces to (1/4) * V = 110
  have hf4 : (1 / 4 : ℚ) * V = 110, by linarith,
  have hf5 : V = 440, from eq_of_mul_eq_mul_left (one_ne_zero : (1 / 4 : ℚ) ≠ 0) (by norm_num : (1 / 4 : ℚ) * 440 = 110),

  -- Conclusion
  exact hf5,
}

end modern_art_museum_l788_788647


namespace prove_sum_l788_788477

variables {a : ℕ → ℝ} {r : ℝ}
variable (pos : ∀ n, 0 < a n)

-- Defining the conditions
def geom_seq (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = a n * r

def condition1 (a : ℕ → ℝ) (r : ℝ) : Prop := a 0 + a 1 + a 2 = 2
def condition2 (a : ℕ → ℝ) (r : ℝ) : Prop := a 2 + a 3 + a 4 = 8

-- The main theorem statement
theorem prove_sum (a : ℕ → ℝ) (r : ℝ) (pos : ∀ n, 0 < a n)
  (geom : geom_seq a r) (h1 : condition1 a r) (h2 : condition2 a r) :
  a 3 + a 4 + a 5 = 16 :=
sorry

end prove_sum_l788_788477


namespace circles_externally_tangent_l788_788170

-- Definition of Circle 1 given in the problem
def Circle1 (x y : ℝ) := x^2 + y^2 + 2 * x + 2 * y - 2 = 0

-- Definition of Circle 2 given in the problem
def Circle2 (x y : ℝ) := x^2 + y^2 - 6 * x + 2 * y + 6 = 0

-- Proof statement for the positional relationship between Circle1 and Circle2
theorem circles_externally_tangent :
  ∀ x y : ℝ, Circle1 x y → Circle2 x y → (dist (-1, -1) (3, -1) = 4) := 
begin
  -- (Proof steps would go here, but they are omitted as per instructions)
  sorry
end

end circles_externally_tangent_l788_788170


namespace find_min_value_l788_788359

-- Define the function y given x
def y (x : ℝ) : ℝ := 
  Real.tan (x + 2 * Real.pi / 3) - Real.tan (x + Real.pi / 6) + 
  Real.cos (x + Real.pi / 6) + Real.sin (x + Real.pi / 6)

-- Define the conditions for x
def x_in_bounds (x : ℝ) : Prop := 
  -Real.pi / 4 ≤ x ∧ x ≤ -Real.pi / 6

-- Define the minimum value to be proved
def y_min_value (x : ℝ) : Prop :=
  y x = sqrt(2)

-- The statement of the proof problem
theorem find_min_value : ∃ x : ℝ, x_in_bounds x → y_min_value x :=
sorry   -- proof to be provided

end find_min_value_l788_788359


namespace projection_onto_vector_is_expected_l788_788308

def projection_matrix (u: ℝ × ℝ) : matrix (fin 2) (fin 2) ℝ :=
  let ⟨x, y⟩ := u in 
  (1 / (x^2 + y^2)) • (matrix.col_vec u ⬝ (matrix.transpose (matrix.row_vec u)))

def expected_matrix : matrix (fin 2) (fin 2) ℝ :=
  ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]]

theorem projection_onto_vector_is_expected :
  projection_matrix (3, 4) = expected_matrix := by
  sorry

end projection_onto_vector_is_expected_l788_788308


namespace distance_parallel_lines_l788_788592

-- Define the two lines l1 and l2
def line1 (x y : ℝ) : Prop := x - y + 1 = 0
def line2 (x y : ℝ) : Prop := 3 * x - 3 * y + 1 = 0

-- Definition of distance between two parallel lines
def distance_between_lines (a b c1 c2 : ℝ) : ℝ :=
  abs (c1 - c2) / real.sqrt (a^2 + b^2)

-- Given conditions line1 and line2, prove the distance is sqrt(2)/3
theorem distance_parallel_lines : 
  let d := distance_between_lines 1 (-1) 1 (1/3) in
  d = real.sqrt 2 / 3 :=
by
  -- This will be where the proof goes
  sorry

end distance_parallel_lines_l788_788592


namespace power_multiplication_l788_788759

theorem power_multiplication : 3000 * (3000 ^ 2999) = 3000 ^ 3000 := by
  sorry

end power_multiplication_l788_788759


namespace total_apples_purchased_l788_788747

theorem total_apples_purchased (M : ℝ) (T : ℝ) (W : ℝ) 
    (hM : M = 15.5)
    (hT : T = 3.2 * M)
    (hW : W = 1.05 * T) :
    M + T + W = 117.18 := by
  sorry

end total_apples_purchased_l788_788747


namespace height_of_right_triangle_l788_788568

theorem height_of_right_triangle (a b c : ℝ) (h : ℝ) (h_right : a^2 + b^2 = c^2) (h_area : h = (a * b) / c) : h = (a * b) / c := 
by
  sorry

end height_of_right_triangle_l788_788568


namespace divide_stones_l788_788969

/-- A pile of 660 stones can be divided into 30 piles where the sizes of the piles differ by less than a factor of 2. -/
theorem divide_stones (n : ℕ) (p : ℕ) (stones : ℕ) :
  stones = 660 → p = 30 →
  ∃ (heaps : Fin p → ℕ),
    (∑ i, heaps i = stones) ∧ (∀ i j, heaps i ≤ 2 * heaps j ∧ heaps j ≤ 2 * heaps i) :=
by
  intros h1 h2
  sorry

end divide_stones_l788_788969


namespace OH_squared_l788_788917

variables {A B C O H : Type}
variables (a b c R : ℝ)

-- Define the conditions
def IsCircumcenter (O : Type) := true -- placeholder, requires precise definition
def IsOrthocenter (H : Type) := true -- placeholder, requires precise definition
def sideLengths (a b c : ℝ) := true -- placeholder, requires precise definition
def circumradius (R : ℝ) := R = 5
def sumOfSquareSides (a b c : ℝ) := a^2 + b^2 + c^2 = 50

-- The main statement to be proven
theorem OH_squared (h1 : IsCircumcenter O)
                   (h2 : IsOrthocenter H)
                   (h3 : sideLengths a b c)
                   (h4 : circumradius R)
                   (h5 : sumOfSquareSides a b c) :
    let OH_squared := 9 * R^2 - (a^2 + b^2 + c^2)
    in OH_squared = 175 := sorry

end OH_squared_l788_788917


namespace man_age_twice_son_age_in_2_years_l788_788235

variable (currentAgeSon : ℕ)
variable (currentAgeMan : ℕ)
variable (Y : ℕ)

-- Given conditions
def sonCurrentAge : Prop := currentAgeSon = 23
def manCurrentAge : Prop := currentAgeMan = currentAgeSon + 25
def manAgeTwiceSonAgeInYYears : Prop := currentAgeMan + Y = 2 * (currentAgeSon + Y)

-- Theorem to prove
theorem man_age_twice_son_age_in_2_years :
  sonCurrentAge currentAgeSon →
  manCurrentAge currentAgeSon currentAgeMan →
  manAgeTwiceSonAgeInYYears currentAgeSon currentAgeMan Y →
  Y = 2 :=
by
  intros h_son_age h_man_age h_age_relation
  sorry

end man_age_twice_son_age_in_2_years_l788_788235


namespace vector_scalar_m_eq_l788_788450

theorem vector_scalar_m_eq :
  ∃ (m : ℝ), ∀ (a b : ℝ × ℝ), a = (4, 2) → b = (m, 3) → ∃ (λ : ℝ), a = λ • b → m = 6 :=
by
  sorry

end vector_scalar_m_eq_l788_788450


namespace projection_matrix_correct_l788_788311

variables {R : Type*} [field R] [decidable_eq R]
variables (x y : R)
def vector_v : matrix (fin 2) (fin 1) R := ![![3], ![4]]
def vector_u : matrix (fin 2) (fin 1) R := ![![x], ![y]]
def projection_matrix : matrix (fin 2) (fin 2) R := ![![9/25, 12/25], ![12/25, 16/25]]

theorem projection_matrix_correct :
  (projection_matrix R) ⬝ (vector_u x y) = (25 : R)⁻¹ • (transpose (vector_v 3 4) ⬝ (vector_u x y)) ⬝ (vector_v 3 4) := 
sorry

end projection_matrix_correct_l788_788311


namespace find_line_through_P_l788_788807

theorem find_line_through_P (P A B : ℝ × ℝ)
  (hP : P = (2, 1))
  (hA : ∃ a b, A = (a, b) ∧ 2 * a - b - 1 = 0)
  (hB : ∃ m n, B = (m, n) ∧ m + n + 2 = 0)
  (hMid : P = ((fst A + fst B) / 2, (snd A + snd B) / 2)) :
  ∃ l : ℝ × ℝ, (fst l = 4 * fst l - snd l - 7 = 0) :=
begin
  sorry
end

end find_line_through_P_l788_788807


namespace OH_squared_l788_788927

/-- 
Given:
  O is the circumcenter of triangle ABC.
  H is the orthocenter of triangle ABC.
  a, b, and c are the side lengths of triangle ABC.
  R is the circumradius of triangle ABC.
  R = 5.
  a^2 + b^2 + c^2 = 50.

Prove:
  OH^2 = 175.
-/
theorem OH_squared (a b c R : ℝ) (hR : R = 5) (habc : a^2 + b^2 + c^2 = 50) :
  let OH_squared := 9 * R^2 - (a^2 + b^2 + c^2) in
  OH_squared = 175 :=
by
  sorry

end OH_squared_l788_788927


namespace sequence_append_positive_integers_l788_788861

open Function

noncomputable def p (S : List ℕ) : ℕ := S.foldl (· * ·) 1

noncomputable def m (S : List ℕ) : ℚ :=
  let non_empty_subsets := (List.powerset S).filter (· ≠ [])
  (non_empty_subsets.map p).sum / non_empty_subsets.length

set_option pp.explicit true
set_option pp.generalizedFieldNotations true

theorem sequence_append_positive_integers 
  (S : List ℕ) 
  (exists_n : ∃ n : ℕ, S.length = n ∧ m S = 13) 
  (a_n1 : ℕ) 
  (S' := S ++ [a_n1])
  (m_Sp_eq_49 : m S' = 49) : 
  S' = [1, 1, 7, 22] :=
sorry

end sequence_append_positive_integers_l788_788861


namespace totalCandlesInHouse_l788_788905

-- Definitions for the problem's conditions
def bedroomCandles : ℕ := 20
def livingRoomCandles : ℕ := bedroomCandles / 2
def donovanCandles : ℕ := 20

-- Problem to prove
theorem totalCandlesInHouse : bedroomCandles + livingRoomCandles + donovanCandles = 50 := by
  sorry

end totalCandlesInHouse_l788_788905


namespace chess_game_probabilities_l788_788128

theorem chess_game_probabilities :
  let p_draw := 1 / 2
  let p_b_win := 1 / 3
  let p_sum := 1
  let p_a_win := p_sum - p_draw - p_b_win
  let p_a_not_lose := p_draw + p_a_win
  let p_b_not_lose := p_draw + p_b_win
  A := p_a_win = 1 / 6
  B := p_a_not_lose = 1 / 2
  C := p_a_win = 2 / 3
  D := p_b_not_lose = 1 / 2
  in ¬ (p_a_win = 1 / 6 ∧ p_a_not_lose ≠ 1 / 2 ∧ p_a_win ≠ 2 / 3 ∧ p_b_not_lose ≠ 1 / 2)
:=
sorry

end chess_game_probabilities_l788_788128


namespace divide_660_stones_into_30_piles_l788_788985

theorem divide_660_stones_into_30_piles :
  ∃ (heaps : Fin 30 → ℕ),
    (∑ i, heaps i = 660) ∧
    (∀ i j, heaps i < 2 * heaps j ∨ heaps j < 2 * heaps i) :=
sorry

end divide_660_stones_into_30_piles_l788_788985


namespace dot_product_zero_l788_788869

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V) (k : ℝ)

-- Define the conditions
def a_parallel_b : Prop := ∃ k : ℝ, b = k • a
def a_perp_c : Prop := ⟪a, c⟫ = 0

-- The theorem to be proved
theorem dot_product_zero (h1 : a_parallel_b a b) (h2 : a_perp_c a c) :
  ⟪c, a + 2 • b⟫ = 0 :=
sorry

end dot_product_zero_l788_788869


namespace count_right_triangles_with_given_conditions_l788_788399

-- Define the type of our points as a pair of integers
def Point := (ℤ × ℤ)

-- Define the orthocenter being a specific point
def isOrthocenter (P : Point) := P = (-1, 7)

-- Define that a given triangle has a right angle at the origin
def rightAngledAtOrigin (O A B : Point) :=
  O = (0, 0) ∧
  (A.fst = 0 ∨ A.snd = 0) ∧
  (B.fst = 0 ∨ B.snd = 0) ∧
  (A.fst ≠ 0 ∨ A.snd ≠ 0) ∧
  (B.fst ≠ 0 ∨ B.snd ≠ 0)

-- Define that the points are lattice points
def areLatticePoints (O A B : Point) :=
  ∃ t k : ℤ, (A = (3 * t, 4 * t) ∧ B = (-4 * k, 3 * k)) ∨
            (B = (3 * t, 4 * t) ∧ A = (-4 * k, 3 * k))

-- Define the number of right triangles given the constraints
def numberOfRightTriangles : ℕ := 2

-- Statement of the problem
theorem count_right_triangles_with_given_conditions :
  ∃ (O A B : Point),
    rightAngledAtOrigin O A B ∧
    isOrthocenter (-1, 7) ∧
    areLatticePoints O A B ∧
    numberOfRightTriangles = 2 :=
  sorry

end count_right_triangles_with_given_conditions_l788_788399


namespace cheese_piece_volume_l788_788261

theorem cheese_piece_volume (d h : ℝ) (V_piece : ℝ) (one_third : V_piece = (1 / 3 * π * (d / 2)^2 * h)) (d_val : d = 5) (h_val : h = 1.5) : abs (V_piece - 5.9) < 0.1 :=
by
  have radius : ℝ := d / 2
  have full_volume : ℝ := π * radius^2 * h
  have piece_volume : ℝ := (1 / 3) * full_volume
  rw [d_val, h_val, one_third]
  norm_num
  sorry

end cheese_piece_volume_l788_788261


namespace range_of_a_l788_788171

-- Definitions based on the given conditions
def p (a : ℝ) : Prop := ∀ x ∈ set.Icc 1 2, x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

-- Proposition statement to prove the range of 'a'
theorem range_of_a (a : ℝ) : p a ∧ q a ↔ a ≤ -2 ∨ a = 1 :=
by
  sorry

end range_of_a_l788_788171


namespace eventually_return_to_initial_state_l788_788575

theorem eventually_return_to_initial_state (n : ℕ) (boxes : Fin n → ℕ) :
  ∃ k, ( ∃ f: (ℕ → Fin n → ℕ), f 0 = boxes ∧ (∀ m, f (m + 1) = update_boxes (f m) )
    → f (k) = boxes)
:=
sorry

end eventually_return_to_initial_state_l788_788575


namespace find_n_in_range_l788_788300

theorem find_n_in_range :
  ({n : ℤ | -20 ≤ n ∧ n ≤ 20 ∧ n ≡ -127 [MOD 7]} = {-13, 1, 15}) :=
by
  sorry

end find_n_in_range_l788_788300


namespace slope_angle_AB_l788_788816

noncomputable def A : ℝ × ℝ := (0, 1)
noncomputable def B : ℝ × ℝ := (1, 0)

theorem slope_angle_AB :
  let θ := Real.arctan (↑(B.2 - A.2) / ↑(B.1 - A.1))
  θ = 3 * Real.pi / 4 := 
by
  -- Proof goes here
  sorry

end slope_angle_AB_l788_788816


namespace number_of_triangles_l788_788711

/-!
# Problem Statement
Given a square with 20 interior points connected such that the lines do not intersect and divide the square into triangles,
prove that the number of triangles formed is 42.
-/

theorem number_of_triangles (V E F : ℕ) (hV : V = 24) (hE : E = (3 * F + 1) / 2) (hF : V - E + F = 2) :
  (F - 1) = 42 :=
by
  sorry

end number_of_triangles_l788_788711


namespace rakesh_salary_l788_788572

variable (S : ℝ) -- The salary S is a real number
variable (h : 0.595 * S = 2380) -- Condition derived from the problem

theorem rakesh_salary : S = 4000 :=
by
  sorry

end rakesh_salary_l788_788572


namespace parabola_focus_coordinates_l788_788160

theorem parabola_focus_coordinates :
  let p := 1 in
  ∃ (x y : ℝ), (y^2 = 4 * x) ∧ (x = p) ∧ (y = 0) :=
by
  let p := 1
  use p, 0
  split
  sorry

end parabola_focus_coordinates_l788_788160


namespace intersection_when_a_eq_4_range_for_A_subset_B_l788_788032

-- Define the conditions
def setA : Set ℝ := { x | (1 - x) / (x - 7) > 0 }
def setB (a : ℝ) : Set ℝ := { x | x^2 - 2 * x - a^2 - 2 * a < 0 }

-- First proof goal: When a = 4, find A ∩ B
theorem intersection_when_a_eq_4 :
  setA ∩ (setB 4) = { x : ℝ | 1 < x ∧ x < 6 } :=
sorry

-- Second proof goal: Find the range for a such that A ⊆ B
theorem range_for_A_subset_B :
  { a : ℝ | setA ⊆ setB a } = { a : ℝ | a ≤ -7 ∨ a ≥ 5 } :=
sorry

end intersection_when_a_eq_4_range_for_A_subset_B_l788_788032


namespace concurrency_AM_EN_FP_l788_788492

variable {α : Type*} [EuclideanGeometry α]
variables {A B C D E F M N P O : α}

-- Define the properties and points of the equilateral triangle and midpoints
def is_equilateral (A B C : α) : Prop := 
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

def midpoint (X Y M : α) : Prop := 
  dist M X = dist M Y ∧ dist X Y = 2 * dist M X

-- Assumptions
variables (hABC : is_equilateral A B C)
variables (hD_mid : midpoint B C D)
variables (hE_mid : midpoint C A E)
variables (hF_mid : midpoint A B F)
variables (hM_mid : midpoint F D M)
variables (hN_mid : midpoint F B N)
variables (hP_mid : midpoint D C P)

-- Concurrency statement
theorem concurrency_AM_EN_FP 
  (hAM : line_through_points A M)
  (hEN : line_through_points E N)
  (hFP : line_through_points F P)
  : concurrent A M E N F P :=
sorry

end concurrency_AM_EN_FP_l788_788492


namespace original_number_j_l788_788521

noncomputable def solution (n : ℚ) : ℚ := (3 * (n + 3) - 5) / 3

theorem original_number_j { n : ℚ } (h : solution n = 10) : n = 26 / 3 :=
by
  sorry

end original_number_j_l788_788521


namespace find_k_l788_788937

noncomputable def series (k : ℝ) : ℝ := ∑' n, (7 * n - 2) / k^n

theorem find_k (k : ℝ) (h₁ : 1 < k) (h₂ : series k = 17 / 2) : k = 17 / 7 :=
by
  sorry

end find_k_l788_788937


namespace largest_sum_valid_set_l788_788111

-- Define the conditions for the set S
def valid_set (S : Finset ℕ) : Prop :=
  (∀ x ∈ S, 0 < x ∧ x ≤ 15) ∧
  ∀ (A B : Finset ℕ), A ⊆ S → B ⊆ S → A ≠ B → A ∩ B = ∅ → A.sum id ≠ B.sum id

-- The theorem stating the largest sum of such a set
theorem largest_sum_valid_set : ∃ (S : Finset ℕ), valid_set S ∧ S.sum id = 61 :=
sorry

end largest_sum_valid_set_l788_788111


namespace geometric_series_sum_n_eq_6_l788_788495

theorem geometric_series_sum_n_eq_6 : 
  ∃ n : ℕ, (S_n = 126) ∧ (n = 6) :=
sorry
  where
    a : ℕ → ℕ
    | 1        := 2
    | (n + 1)  := 2 * a n

    S_n (n : ℕ) : ℕ := (∑ i in finset.range n.succ, a (i+1))

-- Additional definitions and import might be necessary for handling mathematical operations.

end geometric_series_sum_n_eq_6_l788_788495


namespace max_value_of_4x_plus_3y_l788_788415

noncomputable def maxFourXPlusThreeY : ℝ :=
  let eqn := λ x y : ℝ, x^2 + y^2 = 10 * x + 8 * y + 10 in
  let maxW := 70 in
  maxW

theorem max_value_of_4x_plus_3y (x y : ℝ) (h : x^2 + y^2 = 10 * x + 8 * y + 10) : 4 * x + 3 * y ≤ maxFourXPlusThreeY :=
begin
  sorry -- proof steps would go here
end

end max_value_of_4x_plus_3y_l788_788415


namespace find_n_find_largest_terms_in_expansion_l788_788827

/-- Given that the binomial coefficients of the sixth and seventh terms in the expansion of (1 + 2x)^n are the largest, prove n = 11. -/
theorem find_n (n : ℕ) (hn: binomial_coeff ((1 : ℝ) + 2* (x : ℝ)) n = largest_coeff_sixth_seventh) : 
  n = 11 := 
sorry

/-- Find the terms with the largest coefficients in the expansion of (1 + 2x)^(11) -/
theorem find_largest_terms_in_expansion :
  let T_8 := 42240 * x^7,
      T_9 := 42240 * x^8 in
  largest_coeff_terms ((1 : ℝ) + 2* (x : ℝ)) 11 = [T_8, T_9] :=
sorry

end find_n_find_largest_terms_in_expansion_l788_788827


namespace smallest_number_of_ducks_l788_788553

theorem smallest_number_of_ducks (n_ducks n_cranes : ℕ) (h1 : n_ducks = n_cranes) : 
  ∃ n, n_ducks = n ∧ n_cranes = n ∧ n = Nat.lcm 13 17 := by
  use 221
  sorry

end smallest_number_of_ducks_l788_788553


namespace consignment_fee_correct_l788_788793

noncomputable def consignment_fee (x : ℝ) : ℝ :=
if x ≤ 50 then 0.53 * x
else 50 * 0.53 + (x - 50) * 0.85

theorem consignment_fee_correct (x : ℝ) (h : x > 50) :
  consignment_fee x = 50 * 0.53 + (x - 50) * 0.85 :=
by
  unfold consignment_fee
  simp [if_neg (not_le_of_gt h)]
  sorry  -- Proof can be completed later

end consignment_fee_correct_l788_788793


namespace value_2_stddevs_less_than_mean_l788_788155

-- Definitions based on the conditions
def mean : ℝ := 10.5
def stddev : ℝ := 1
def value := mean - 2 * stddev

-- Theorem we aim to prove
theorem value_2_stddevs_less_than_mean : value = 8.5 := by
  -- proof will go here
  sorry

end value_2_stddevs_less_than_mean_l788_788155


namespace range_f_x1_x2_l788_788836

noncomputable def f (c x : ℝ) : ℝ := 2 * x ^ 3 - 3 * x ^ 2 + c * x + 1

theorem range_f_x1_x2 (c x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 < x2) 
  (h4 : 36 - 24 * c > 0) (h5 : ∀ x, f c x = 2 * x ^ 3 - 3 * x ^ 2 + c * x + 1) :
  1 < f c x1 / x2 ∧ f c x1 / x2 < 5 / 2 :=
sorry

end range_f_x1_x2_l788_788836


namespace divide_stones_into_heaps_l788_788952

-- Definitions based on the identified conditions
variable (Heaps : List ℕ) -- list of heap sizes
variable (n_stones : ℕ) -- total number of stones
variable (n_heaps : ℕ) -- number of heaps

-- Conditions
axiom total_stones : n_stones = 660
axiom total_heaps : n_heaps = 30
axiom heap_size_condition : ∀ (h₁ h₂ : ℕ), h₁ ∈ Heaps → h₂ ∈ Heaps → h₁ ≤ 2 * h₂

-- Theorem statement
theorem divide_stones_into_heaps : 
  ∃ Heaps, Heaps.length = n_heaps ∧ Heaps.sum = n_stones ∧
  ∀ (h₁ h₂ : ℕ), h₁ ∈ Heaps → h₂ ∈ Heaps → h₁ ≤ 2 * h₂ :=
by
  sorry

end divide_stones_into_heaps_l788_788952


namespace projection_matrix_3_4_l788_788326

theorem projection_matrix_3_4 :
  let v := λ α : Type, @vector α 2 := ![3, 4]
  let proj := λ x : vector ℝ 2, (v ℝ ⬝ x) / (v ℝ ⬝ v ℝ) • v ℝ
  proj = (λ x : vector ℝ 2, matrix.mul_vec ![
     ![9 / 25, 12 / 25],
     ![12 / 25, 16 / 25]
  ] x) :=
by sorry

end projection_matrix_3_4_l788_788326


namespace parabola_and_circle_tangency_l788_788621

open Real

noncomputable def parabola_eq : Prop :=
  (parabola : {x : ℝ → ℝ | ∃ y: ℝ, y^2 = x})

noncomputable def circle_eq : Prop :=
  (circle : {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2)^2 = 1})

theorem parabola_and_circle_tangency:
  (∀ x y : ℝ, ∃ p, y^2 = x ↔ p ∈ parabola_eq) →
  ((x - 2)^2 + y^2 = 1) →
  (∀ A1 A2 A3 : ℝ × ℝ,
    A1 ∈ parabola_eq ∧ A2 ∈ parabola_eq ∧ A3 ∈ parabola_eq →
    (tangential A1 A2 circle ∧ tangential A1 A3 circle →
    tangential A2 A3 circle
  )) := sorry

end parabola_and_circle_tangency_l788_788621


namespace m_value_l788_788262

theorem m_value (m : ℝ) (h : (243:ℝ) ^ (1/3) = 3 ^ m) : m = 5 / 3 :=
sorry

end m_value_l788_788262


namespace totalCandies_l788_788072

def bobCandies : Nat := 10
def maryCandies : Nat := 5
def sueCandies : Nat := 20
def johnCandies : Nat := 5
def samCandies : Nat := 10

theorem totalCandies : bobCandies + maryCandies + sueCandies + johnCandies + samCandies = 50 := 
by
  sorry

end totalCandies_l788_788072


namespace problem_statement_l788_788468

-- Assuming l is a line and alpha is a plane. 
-- l is not parallel to alpha and l is not contained in alpha implies 
-- there are no lines in alpha that are parallel to l.

structure Line : Type := (exists_point : ℝ → ℝ → ℝ)

structure Plane : Type := (contains_line : Line → Prop)

variables (l : Line) (α : Plane)

def is_not_parallel_to_plane (l : Line) (α : Plane) : Prop :=
  ¬ ∀ (l₂ : Line), α.contains_line l₂ → ∀ p, l.exists_point p = l₂.exists_point p

def is_not_subset_of_plane (l : Line) (α : Plane) : Prop :=
  ¬ ∀ p, l.exists_point p ∈ α.contains_line

def no_line_in_plane_parallel_to_line (l : Line) (α : Plane) : Prop :=
  ∀ (l₂ : Line), α.contains_line l₂ → ¬ ∀ p, l.exists_point p = l₂.exists_point p

theorem problem_statement
  (h1 : is_not_parallel_to_plane l α)
  (h2 : is_not_subset_of_plane l α) :
  no_line_in_plane_parallel_to_line l α :=
sorry

end problem_statement_l788_788468


namespace find_term_number_l788_788223

-- Define the arithmetic sequence
def arithmetic_seq (a d : Int) (n : Int) := a + (n - 1) * d

-- Define the condition: first term and common difference
def a1 := 4
def d := 3

-- Prove that the 672nd term is 2017
theorem find_term_number (n : Int) (h : arithmetic_seq a1 d n = 2017) : n = 672 := by
  sorry

end find_term_number_l788_788223


namespace convex_quadrilateral_inequality_l788_788562

variable (a b c d : ℝ) -- lengths of sides of quadrilateral
variable (S : ℝ) -- Area of the quadrilateral

-- Given condition: a, b, c, d are lengths of the sides of a convex quadrilateral
def is_convex_quadrilateral (a b c d : ℝ) (S : ℝ) : Prop :=
  S ≤ (a^2 + b^2 + c^2 + d^2) / 4

theorem convex_quadrilateral_inequality (a b c d : ℝ) (S : ℝ) 
  (h : is_convex_quadrilateral a b c d S) : 
  S ≤ (a^2 + b^2 + c^2 + d^2) / 4 := 
by
  sorry

end convex_quadrilateral_inequality_l788_788562


namespace perimeter_convex_polygon_lt_pi_d_l788_788566

theorem perimeter_convex_polygon_lt_pi_d (n : ℕ) (d : ℝ) (h : d > 0) 
  (lengths : fin n → ℝ) 
  (convex : ∀ (i j : fin n), lengths i = lengths j) 
  (side_cond : ∀ (i : fin n), lengths i < d) 
  (diagonal_cond : ∀ (i j : fin n), i ≠ j → (lengths i + lengths j) < d) :
  ∑ i, lengths i < π * d :=
by
  sorry

end perimeter_convex_polygon_lt_pi_d_l788_788566


namespace foci_distance_of_hyperbola_l788_788788

theorem foci_distance_of_hyperbola :
  let a_sq := 25
  let b_sq := 9
  let c := Real.sqrt (a_sq + b_sq)
  2 * c = 2 * Real.sqrt 34 :=
by
  let a_sq := 25
  let b_sq := 9
  let c := Real.sqrt (a_sq + b_sq)
  sorry

end foci_distance_of_hyperbola_l788_788788


namespace max_students_before_new_year_l788_788702

theorem max_students_before_new_year (N M k l : ℕ) (h1 : 100 * M = k * N) (h2 : 100 * (M + 1) = l * (N + 3)) (h3 : 3 * l < 300) :
      N ≤ 197 := by
  sorry

end max_students_before_new_year_l788_788702


namespace intersection_point_exists_l788_788845

theorem intersection_point_exists :
  ∃ (x y : ℝ), (2 * x + y - 5 = 0) ∧ (y = 2 * x^2 + 1) ∧ (-1 ≤ x ∧ x ≤ 1) ∧ (x = 1) ∧ (y = 3) :=
by
  use 1, 3
  split; norm_num
  split
  · exact -1 ≤ 1 ∧ 1 ≤ 1
  · norm_num
  · norm_num
  sorry

end intersection_point_exists_l788_788845


namespace cubic_product_roots_l788_788802

noncomputable def cubic_function (a b c x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

theorem cubic_product_roots (a b c x1 x2 : ℝ)
  (h1 : cubic_function a b c 0 = 0)
  (h2 : (∃ x1 x2, cubic_function a b c x1 = 0 ∧ cubic_function a b c x2 = 0 ∧ x1 ≠ 0 ∧ x2 ≠ 0))
  (h3 : f' x = 3 * a * x^2 + 2 * b * x + c)
  (h4 : f' (3 - sqrt(3))/3 = 0)
  (h5 : f' (3 + sqrt(3))/3 = 0) : 
  x1 * x2 = 2 :=
by
sory

end cubic_product_roots_l788_788802


namespace problem_builds_S2006_l788_788803

noncomputable def S (n : ℕ) : ℝ :=
  1 + ∑ i in finset.range n, 1 / (∑ j in finset.range (i + 1), 1 / (((j * (j + 1)) / 2) : ℝ))

noncomputable def k (n : ℕ) : ℝ := (n * (n + 1) / 2 : ℝ)

theorem problem_builds_S2006 : 
  let T0 := 1006 in
  S 2006 = T0 :=
begin
  sorry
end

end problem_builds_S2006_l788_788803


namespace unique_polynomial_l788_788395

noncomputable theory

def polynomial_function (f : ℝ → ℝ) : Prop :=
  ∃ (a_3 a_2 a_1 a_0 : ℝ), a_3 ≠ 0 ∧ f = λ x, a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0

theorem unique_polynomial (f : ℝ → ℝ) :
  polynomial_function f →
  (∀ x, f(x ^ 2) = (f x) ^ 2) →
  (∀ x, f(x ^ 2) = f(f x)) →
  f 1 = 1 →
  f = λ x, x ^ 3 :=
begin
  sorry,
end

end unique_polynomial_l788_788395


namespace smallest_possible_median_is_4_l788_788675

noncomputable def smallest_median (x : ℕ) (hx : x > 0) : ℕ :=
if 2 < x ∧ x < 4 then 2 * x else if 4 ≤ x ∧ x < 7 then x else 9

theorem smallest_possible_median_is_4 : 
(∃ x : ℕ, x > 0 ∧ smallest_median x (by apply_instance) = 4) :=
begin
  use 4,
  split,
  { exact nat.one_lt_bit0 nat.one_pos },
  { simp [smallest_median] },
end

end smallest_possible_median_is_4_l788_788675


namespace min_AP_squared_sum_value_l788_788710

-- Definitions based on given problem conditions
def A : ℝ := 0
def B : ℝ := 2
def C : ℝ := 4
def D : ℝ := 7
def E : ℝ := 15

def distance_squared (x y : ℝ) : ℝ := (x - y)^2

noncomputable def min_AP_squared_sum (r : ℝ) : ℝ :=
  r^2 + distance_squared r B + distance_squared r C + distance_squared r D + distance_squared r E

theorem min_AP_squared_sum_value : ∃ (r : ℝ), (min_AP_squared_sum r) = 137.2 :=
by
  existsi 5.6
  sorry

end min_AP_squared_sum_value_l788_788710


namespace max_M_value_l788_788138

noncomputable def M (x y z w : ℝ) : ℝ :=
  x * w + 2 * y * w + 3 * x * y + 3 * z * w + 4 * x * z + 5 * y * z

theorem max_M_value (x y z w : ℝ) (h : x + y + z + w = 1) :
  (M x y z w) ≤ 3 / 2 :=
sorry

end max_M_value_l788_788138


namespace fraction_sum_of_roots_l788_788096

theorem fraction_sum_of_roots (x1 x2 : ℝ) (h1 : 5 * x1^2 - 3 * x1 - 2 = 0) (h2 : 5 * x2^2 - 3 * x2 - 2 = 0) (hx : x1 ≠ x2) :
  (1 / x1 + 1 / x2 = -3 / 2) :=
by
  sorry

end fraction_sum_of_roots_l788_788096


namespace final_answer_l788_788909

-- Given conditions for the triangle and points
variables (x : ℝ) (A B C W X Y Z : Point)
hypothesis AB_eq : distance A B = 6 * x ^ 2 + 1
hypothesis AC_eq : distance A C = 2 * x ^ 2 + 2 * x
hypothesis AW_eq : distance A W = x
hypothesis WX_eq : distance W X = x + 4
hypothesis AY_eq : distance A Y = x + 1
hypothesis YZ_eq : distance Y Z = x

-- The transformation function for lines not intersecting BC
def f (ℓ : Line) : Point := sorry

-- Further given conditions and intersections
hypothesis f_WY_XY_B : intersection (LineThrough (f (lineThrough W Y)) (f (lineThrough X Y))) B
hypothesis f_WZ_XZ_B : intersection (LineThrough (f (lineThrough W Z)) (f (lineThrough X Z))) B
hypothesis f_WZ_WY_C : intersection (LineThrough (f (lineThrough W Z)) (f (lineThrough W Y))) C
hypothesis f_XY_XZ_C : intersection (LineThrough (f (lineThrough X Y)) (f (lineThrough X Z))) C

-- Expression of BC in the required form
structure bc_form :=
(a b c d : ℤ)
(sq_free : squarefree c)
(gcd_bd : Int.gcd b d = 1)
(comb : 100 * a + b + c + d = 413)

-- The main theorem stating our goal
theorem final_answer (h : bc_form) : 
  (∃ (a b c d : ℤ), squarefree c ∧ Int.gcd b d = 1 ∧ 
    (distance B C = a + (b * real.sqrt c) / d) ∧ 100 * a + b + c + d = 413) := 
sorry

end final_answer_l788_788909


namespace find_h_l788_788601

theorem find_h (h j k : ℤ) (y_intercept1 : 3 * h ^ 2 + j = 2013) 
  (y_intercept2 : 2 * h ^ 2 + k = 2014)
  (x_intercepts1 : ∃ (y : ℤ), j = -3 * y ^ 2)
  (x_intercepts2 : ∃ (x : ℤ), k = -2 * x ^ 2) :
  h = 36 :=
by sorry

end find_h_l788_788601


namespace clerical_staff_percentage_l788_788123

theorem clerical_staff_percentage (total_employees : ℕ)
  (initial_clerical_fraction : ℚ) (reduction_fraction : ℚ) :
  total_employees = 3600 → initial_clerical_fraction = (1/3 : ℚ) → reduction_fraction = (1/6 : ℚ) →
  ((floor (((initial_clerical_fraction * total_employees - reduction_fraction * initial_clerical_fraction * total_employees) / (total_employees - reduction_fraction * initial_clerical_fraction * total_employees)) * 1000)).toReal / 10 = 29.4) :=
sorry

end clerical_staff_percentage_l788_788123


namespace divide_stones_into_heaps_l788_788950

-- Definitions based on the identified conditions
variable (Heaps : List ℕ) -- list of heap sizes
variable (n_stones : ℕ) -- total number of stones
variable (n_heaps : ℕ) -- number of heaps

-- Conditions
axiom total_stones : n_stones = 660
axiom total_heaps : n_heaps = 30
axiom heap_size_condition : ∀ (h₁ h₂ : ℕ), h₁ ∈ Heaps → h₂ ∈ Heaps → h₁ ≤ 2 * h₂

-- Theorem statement
theorem divide_stones_into_heaps : 
  ∃ Heaps, Heaps.length = n_heaps ∧ Heaps.sum = n_stones ∧
  ∀ (h₁ h₂ : ℕ), h₁ ∈ Heaps → h₂ ∈ Heaps → h₁ ≤ 2 * h₂ :=
by
  sorry

end divide_stones_into_heaps_l788_788950


namespace f_decreasing_on_neg_infty_2_l788_788595

def f (x : ℝ) := x^2 - 4 * x + 3

theorem f_decreasing_on_neg_infty_2 :
  ∀ x y : ℝ, x < y → y ≤ 2 → f y < f x :=
by
  sorry

end f_decreasing_on_neg_infty_2_l788_788595


namespace bob_should_give_l788_788750

theorem bob_should_give (alice_paid bob_paid charlie_paid : ℕ)
  (h_alice : alice_paid = 120)
  (h_bob : bob_paid = 150)
  (h_charlie : charlie_paid = 180) :
  bob_paid - (120 + 150 + 180) / 3 = 0 := 
by
  sorry

end bob_should_give_l788_788750


namespace derivative_at_pi_over_three_l788_788837

def f (x : ℝ) : ℝ := Real.sin x + Real.cos x + 2 * (f '(Float.pi / 3)) * x + 3 

theorem derivative_at_pi_over_three :
  deriv f (Float.pi / 3) = (Real.sqrt 3 - 1) / 2 :=
  sorry

end derivative_at_pi_over_three_l788_788837


namespace boots_sold_on_monday_eq_24_l788_788153

noncomputable def price_of_shoes : ℝ := 2
noncomputable def price_of_boots (price_of_shoes : ℝ) : ℝ := price_of_shoes + 15

theorem boots_sold_on_monday_eq_24 :
  let S := price_of_shoes,
      B := price_of_boots S,
      monday_revenue := 460,
      tuesday_revenue := 560 in
  ∃ (x : ℕ), 22 * S + x * B = monday_revenue ∧
             8 * S + 32 * B = tuesday_revenue ∧
             x = 24 :=
by
  sorry

end boots_sold_on_monday_eq_24_l788_788153


namespace count_sister_point_pairs_l788_788063

def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + 2*x else (x + 1) / Real.exp 1

def is_on_graph (A : ℝ × ℝ) (f : ℝ → ℝ) : Prop :=
  A.snd = f A.fst

def is_symmetric_about_origin (A B : ℝ × ℝ) : Prop :=
  B = (-A.fst, -A.snd)

def is_sister_point_pair (A B : ℝ × ℝ) (f : ℝ → ℝ) : Prop :=
  is_on_graph A f ∧ is_on_graph B f ∧ is_symmetric_about_origin A B ∧ A ≠ B

theorem count_sister_point_pairs :
  { p : ℝ × ℝ | is_sister_point_pair p.1 p.2 f }.to_finset.card = 2 :=
sorry

end count_sister_point_pairs_l788_788063


namespace range_of_m_l788_788163

def f (x : ℝ) : ℝ := -x^3 - 2*x^2 + 4*x

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≥ m^2 - 14 * m) ↔ 3 ≤ m ∧ m ≤ 11 :=
by
  sorry

end range_of_m_l788_788163


namespace prove_inequality_l788_788382

theorem prove_inequality (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  x^2 + x * y + y^2 ≤ 3 * (x - real.sqrt (x * y) + y)^2 :=
sorry

end prove_inequality_l788_788382


namespace probability_less_than_10_l788_788179

def num_provinces : ℕ := 10

def per_capita_GDP : Fin num_provinces → ℝ
| ⟨0, _⟩ := 18.39
| ⟨1, _⟩ := 17.38
| ⟨2, _⟩ := 13.73
| ⟨3, _⟩ := 11.75
| ⟨4, _⟩ := 11.39
| ⟨5, _⟩ := 11.32
| ⟨6, _⟩ := 9.87
| ⟨7, _⟩ := 8.7
| ⟨8, _⟩ := 8.66
| ⟨9, _⟩ := 8.53

theorem probability_less_than_10 :
  (Fin.num_of (λ i : Fin num_provinces, per_capita_GDP i < 10) 
   (by decide) : ℝ) / num_provinces = 0.4 :=
sorry

end probability_less_than_10_l788_788179


namespace eval_polynomial_l788_788780

theorem eval_polynomial (x : ℝ) (h : x^2 - 3 * x - 9 = 0) : x^3 - 3 * x^2 - 9 * x + 27 = 27 := 
by
  sorry

end eval_polynomial_l788_788780


namespace hyperbola_asymptotes_l788_788786

theorem hyperbola_asymptotes :
  ∀ x y : ℝ,
  (x ^ 2 / 4 - y ^ 2 / 16 = 1) → (y = 2 * x) ∨ (y = -2 * x) :=
sorry

end hyperbola_asymptotes_l788_788786


namespace sum_of_three_integers_l788_788652

theorem sum_of_three_integers (a b c : ℕ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c)
  (h4 : a * b * c = 216000) (h5 : Nat.coprime a b) (h6 : Nat.coprime a c) (h7 : Nat.coprime b c) :
  a + b + c = 184 :=
sorry

end sum_of_three_integers_l788_788652


namespace Vasya_sums_l788_788665

theorem Vasya_sums :
  ∃ (a b c d : ℝ),
    a + b = 2 ∧ a + c = 6 ∧
    ∃ (p q r s : ℝ),
      {p + q, p + r, p + s, q + r, q + s, r + s}.max = 20 ∧
      {p + q, p + r, p + s, q + r, q + s, r + s}.erase (20).max = 16 ∧
      {p + q, p + r, p + s, q + r, q + s, r + s}.erase (20).erase (16).max = 13 ∧
      {p + q, p + r, p + s, q + r, q + s, r + s}.erase (20).erase (16).erase (13).max = 9 :=
begin
  sorry

end Vasya_sums_l788_788665


namespace projection_matrix_l788_788346

theorem projection_matrix
  (x y : ℝ) :
  let v := ![3, 4]
  let proj_v := (v ⬝ ![x, y]) / (v ⬝ v) • v
  let proj_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
    ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]] in
  proj_v = proj_matrix.mul_vec ![x, y] :=
by
  let v := ![3, 4]
  let proj_v := (v ⬝ ![x, y]) / (v ⬝ v) • v
  let proj_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]]
  sorry

end projection_matrix_l788_788346


namespace projection_matrix_is_correct_l788_788350

noncomputable def projectionMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  let v : Fin 2 → ℝ := ![3, 4]
  (1 / (v 0 ^ 2 + v 1 ^ 2)) • (λ i j, v i * v j)

theorem projection_matrix_is_correct :
  projectionMatrix = ![![9/25, 12/25], ![12/25, 16/25]] :=
by
  sorry

end projection_matrix_is_correct_l788_788350


namespace impossible_grid_placement_l788_788071

theorem impossible_grid_placement :
  ¬ ∃ f : ℕ × ℕ → ℕ, ∀ m n : ℕ, m > 100 → n > 100 →
    (∃ s : ℕ, s = ∑ i in finset.range m, ∑ j in finset.range n, f (i, j) ∧ (m + n) ∣ s) := 
sorry

end impossible_grid_placement_l788_788071


namespace prove_total_number_of_apples_l788_788264

def avg_price (light_price heavy_price : ℝ) (light_proportion heavy_proportion : ℝ) : ℝ :=
  light_proportion * light_price + heavy_proportion * heavy_price

def weighted_avg_price (prices proportions : List ℝ) : ℝ :=
  (List.map (λ ⟨p, prop⟩ => p * prop) (List.zip prices proportions)).sum

noncomputable def total_num_apples (total_earnings weighted_price : ℝ) : ℝ :=
  total_earnings / weighted_price

theorem prove_total_number_of_apples : 
  let light_proportion := 0.6
  let heavy_proportion := 0.4
  let prices := [avg_price 0.4 0.6 light_proportion heavy_proportion, 
                 avg_price 0.1 0.15 light_proportion heavy_proportion,
                 avg_price 0.25 0.35 light_proportion heavy_proportion,
                 avg_price 0.15 0.25 light_proportion heavy_proportion,
                 avg_price 0.2 0.3 light_proportion heavy_proportion,
                 avg_price 0.05 0.1 light_proportion heavy_proportion]
  let proportions := [0.4, 0.2, 0.15, 0.1, 0.1, 0.05]
  let weighted_avg := weighted_avg_price prices proportions
  total_num_apples 120 weighted_avg = 392 :=
by
  sorry

end prove_total_number_of_apples_l788_788264


namespace moving_circle_passes_fixed_point_l788_788237

noncomputable def moving_circle_center_on_parabola (h k : ℝ) : Prop :=
k^2 = 4 * h

noncomputable def circle_tangent_to_line (h k : ℝ) (r : ℝ) : Prop :=
h + r = -1

theorem moving_circle_passes_fixed_point :
  ∀ (h k r : ℝ), moving_circle_center_on_parabola h k → circle_tangent_to_line h k r → (1 - h)^2 + k^2 = r^2 :=
by {
  intros h k r hc ht,
  sorry
}

end moving_circle_passes_fixed_point_l788_788237


namespace gas_station_distance_l788_788125

theorem gas_station_distance (x : ℝ) :
  (x < 10 ∧ x > 7 ∧ x > 5 ∧ x > 9) → x ∈ set.Ioi 9 := by
  sorry

end gas_station_distance_l788_788125


namespace necessary_condition_not_sufficient_condition_l788_788004

noncomputable def zero_point (a : ℝ) : Prop :=
  ∃ x : ℝ, 3^x + a - 1 = 0

noncomputable def decreasing_log (a : ℝ) : Prop :=
  0 < a ∧ a < 1

theorem necessary_condition (a : ℝ) (h : zero_point a) : 0 < a ∧ a < 1 := sorry

theorem not_sufficient_condition (a : ℝ) (h : 0 < a ∧ a < 1) : ¬(zero_point a) := sorry

end necessary_condition_not_sufficient_condition_l788_788004


namespace projection_onto_vector_l788_788321

noncomputable def projection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  ![\[9 / 25, 12 / 25\], \[12 / 25, 16 / 25\]]

theorem projection_onto_vector:
    ∀ (x y : ℚ), (Matrix.mul_vec projection_matrix ![\x, \y]) = ![(9 * x + 12 * y) / 25, (12 * x + 16 * y) / 25] := by
  sorry

end projection_onto_vector_l788_788321


namespace det_rotation_matrix_75_degrees_l788_788531

theorem det_rotation_matrix_75_degrees :
  let θ : ℝ := 75 * (Real.pi / 180)
  let S := Matrix.vec₁ 2 2
  S = ![
    [Real.cos θ, -Real.sin θ],
    [Real.sin θ,  Real.cos θ]
  ]
  Matr.det S = 1 :=
by
  sorry

end det_rotation_matrix_75_degrees_l788_788531


namespace domain_v_l788_788668

noncomputable def v (x : ℝ) : ℝ := 1 / (Real.sqrt (x^2 + 1))

theorem domain_v : ∀ x : ℝ, ∃ y : ℝ, v x = y :=
by
  intro x
  use v x
  unfold v
  sorry

end domain_v_l788_788668


namespace proof_l788_788102

variable {n : ℕ} (n_pos : 0 < n)
variable {a b : Fin n → ℝ} {A B : ℝ}
variable (h1 : ∀ i, 0 ≤ i < n → 0 < a i ∧ 0 < b i ∧ a i ≤ b i)
variable (h2 : ∀ i, 0 ≤ i < n → a i ≤ A)
variable (h3 : (∏ i in Finset.range n, b i) / (∏ i in Finset.range n, a i) ≤ B / A)

theorem proof :
  (∏ i in Finset.range n, b i + 1) / (∏ i in Finset.range n, a i + 1) ≤ (B + 1) / (A + 1) :=
sorry

end proof_l788_788102


namespace regular_pentagon_of_convex_equal_sides_and_angles_l788_788133

theorem regular_pentagon_of_convex_equal_sides_and_angles 
  (ABCDE : polygon)
  (h_convex : convex ABCDE)
  (h_equal_sides : ∀ i j, side_length ABCDE i = side_length ABCDE j)
  (h_angles : ∀ i j, interior_angle ABCDE i ≥ interior_angle ABCDE j ∨ interior_angle ABCDE i = interior_angle ABCDE j) :
  (∀ i j, interior_angle ABCDE i = interior_angle ABCDE j) :=
sorry

end regular_pentagon_of_convex_equal_sides_and_angles_l788_788133


namespace OH_squared_l788_788918

variables {A B C O H : Type}
variables (a b c R : ℝ)

-- Define the conditions
def IsCircumcenter (O : Type) := true -- placeholder, requires precise definition
def IsOrthocenter (H : Type) := true -- placeholder, requires precise definition
def sideLengths (a b c : ℝ) := true -- placeholder, requires precise definition
def circumradius (R : ℝ) := R = 5
def sumOfSquareSides (a b c : ℝ) := a^2 + b^2 + c^2 = 50

-- The main statement to be proven
theorem OH_squared (h1 : IsCircumcenter O)
                   (h2 : IsOrthocenter H)
                   (h3 : sideLengths a b c)
                   (h4 : circumradius R)
                   (h5 : sumOfSquareSides a b c) :
    let OH_squared := 9 * R^2 - (a^2 + b^2 + c^2)
    in OH_squared = 175 := sorry

end OH_squared_l788_788918


namespace binomial_odd_sum_l788_788177

theorem binomial_odd_sum (n : ℕ) (h : (2:ℕ)^(n - 1) = 64) : n = 7 :=
by
  sorry

end binomial_odd_sum_l788_788177


namespace cones_slant_height_angle_l788_788659

theorem cones_slant_height_angle :
  ∀ (α: ℝ),
  α = 2 * Real.arccos (Real.sqrt (2 / (2 + Real.sqrt 2))) :=
by
  sorry

end cones_slant_height_angle_l788_788659


namespace percentage_invalid_votes_l788_788059

theorem percentage_invalid_votes (V : ℕ) (VB : ℕ) (A_exceeds : ℕ) (P : ℝ)
  (hV : V = 6720) (hVB : VB = 2184) (hA_exceeds : A_exceeds = 0.15 * V)
  (h_valid_votes_eq : VB + (VB + A_exceeds) = (100 - P) / 100 * V) :
  P = 20 :=
by
  sorry

end percentage_invalid_votes_l788_788059


namespace sum_of_three_integers_l788_788651

theorem sum_of_three_integers (a b c : ℕ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c)
  (h4 : a * b * c = 216000) (h5 : Nat.coprime a b) (h6 : Nat.coprime a c) (h7 : Nat.coprime b c) :
  a + b + c = 184 :=
sorry

end sum_of_three_integers_l788_788651


namespace integer_solutions_l788_788784

theorem integer_solutions (x y k : ℤ) :
  21 * x + 48 * y = 6 ↔ ∃ k : ℤ, x = -2 + 16 * k ∧ y = 1 - 7 * k :=
by
  sorry

end integer_solutions_l788_788784


namespace emma_garden_area_l788_788779

-- Define the given conditions
def EmmaGarden (total_posts : ℕ) (posts_on_shorter_side : ℕ) (posts_on_longer_side : ℕ) (distance_between_posts : ℕ) : Prop :=
  total_posts = 24 ∧
  distance_between_posts = 6 ∧
  (posts_on_longer_side + 1) = 3 * (posts_on_shorter_side + 1) ∧
  2 * (posts_on_shorter_side + 1 + posts_on_longer_side + 1) = 24

-- The theorem to prove
theorem emma_garden_area : ∃ (length width : ℕ), EmmaGarden 24 2 8 6 ∧ (length = 6 * (2) ∧ width = 6 * (8 - 1)) ∧ (length * width = 576) :=
by
  -- proof goes here
  sorry

end emma_garden_area_l788_788779


namespace irrational_sqrt2_among_others_l788_788752

theorem irrational_sqrt2_among_others : 
  let a := (22/7 : ℚ),
      b := (0 : ℚ),
      c := (sqrt 2 : ℝ),
      d := (2 / 10 : ℚ) in
  irrational c ∧ rational a ∧ rational b ∧ rational d :=
by
  sorry

end irrational_sqrt2_among_others_l788_788752


namespace triangle_problem_l788_788069

theorem triangle_problem (A B C D : Point) 
  (h1 : ∃ (α : angle), sin α = 4/5 ∧ α < π/2 ∧ A = vertex α B C)
  (h2 : outside_triangle A B C D)
  (h3 : ∠BAD = ∠DAC)
  (h4 : ∠BDC = π/2)
  (h5 : distance A D = 1)
  (h6 : distance B D / distance C D = 3/2) :
  ∃ (a b c : ℕ), pairwise_rel_prime a b c ∧ AB + AC = a * sqrt b / c ∧ a + b + c = 34 := by
      sorry

end triangle_problem_l788_788069


namespace beta_cannot_be_determined_l788_788463

variables (α β : ℝ)
def consecutive_interior_angles (α β : ℝ) : Prop := -- define what it means for angles to be consecutive interior angles
  α + β = 180  -- this is true for interior angles, for illustrative purposes.

theorem beta_cannot_be_determined
  (h1 : consecutive_interior_angles α β)
  (h2 : α = 55) :
  ¬(∃ β, β = α) :=
by
  sorry

end beta_cannot_be_determined_l788_788463


namespace sum_first_25_AP_l788_788470

theorem sum_first_25_AP (a d : ℝ) (h : a + 7 * d = 4) : 
    let S25 := (25 / 2) * (2 * a + 24 * d)
    in S25 = 100 + 125 * d := 
by
  sorry

end sum_first_25_AP_l788_788470


namespace clock_angle_78_at_7_24_and_7_52_l788_788377

def hour_hand_angle (h m : ℕ) : ℝ := 30 * h + 0.5 * m
def minute_hand_angle (m : ℕ) : ℝ := 6 * m

def angle_between_hands (h m : ℕ) : ℝ := 
  let angle_diff := abs (hour_hand_angle h m - minute_hand_angle m)
  if angle_diff > 180 then 360 - angle_diff else angle_diff

theorem clock_angle_78_at_7_24_and_7_52
  (angle_diff_24 : angle_between_hands 7 24 = 78)
  (angle_diff_52 : angle_between_hands 7 52 = 78) :
  true := sorry

end clock_angle_78_at_7_24_and_7_52_l788_788377


namespace parabola_and_circle_tangency_l788_788620

open Real

noncomputable def parabola_eq : Prop :=
  (parabola : {x : ℝ → ℝ | ∃ y: ℝ, y^2 = x})

noncomputable def circle_eq : Prop :=
  (circle : {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2)^2 = 1})

theorem parabola_and_circle_tangency:
  (∀ x y : ℝ, ∃ p, y^2 = x ↔ p ∈ parabola_eq) →
  ((x - 2)^2 + y^2 = 1) →
  (∀ A1 A2 A3 : ℝ × ℝ,
    A1 ∈ parabola_eq ∧ A2 ∈ parabola_eq ∧ A3 ∈ parabola_eq →
    (tangential A1 A2 circle ∧ tangential A1 A3 circle →
    tangential A2 A3 circle
  )) := sorry

end parabola_and_circle_tangency_l788_788620


namespace relationship_between_y_values_l788_788830

theorem relationship_between_y_values 
  (m : ℝ) 
  (y1 y2 y3 : ℝ)
  (h1 : y1 = (-1 : ℝ) ^ 2 + 2 * (-1 : ℝ) + m) 
  (h2 : y2 = (3 : ℝ) ^ 2 + 2 * (3 : ℝ) + m) 
  (h3 : y3 = ((1 / 2) : ℝ) ^ 2 + 2 * ((1 / 2) : ℝ) + m) : 
  y2 > y3 ∧ y3 > y1 := 
by 
  sorry

end relationship_between_y_values_l788_788830


namespace projection_onto_3_4_matrix_l788_788334

def projection_matrix := λ (u : ℝ) (v : ℝ), (3 * u + 4 * v) / 25

theorem projection_onto_3_4_matrix :
  ∀ (x y : ℝ),
  (λ (u v : ℝ), (3 * x + 4 * y) / 25) = (λ (u v : ℝ), (\(u * 9 / 25) + (v * 12 / 25), (u * 12 / 25) + (v * 16 / 25))) :=
by
  sorry

end projection_onto_3_4_matrix_l788_788334


namespace greatest_odd_three_digit_non_divisor_of_factorial_l788_788195

theorem greatest_odd_three_digit_non_divisor_of_factorial :
  ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 2 = 1) ∧
    (let k := (n - 1) / 2 in ¬ (k * (k + 1)) ∣ n!) ∧
    (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 2 = 1 ∧ (let k := (m - 1) / 2 in ¬ (k * (k + 1)) ∣ m!)) → m ≤ n) :=
begin
  use 999,
  split,
  {split,
   {exact nat.le_of_lt dec_trivial,},
   {exact nat.le_refl 999,}},
  split,
  {exact dec_trivial,},
  split,
  {let k := (999 - 1) / 2,
   simp only [nat.factorial],
   exact sorry,},
  intros m hm,
  have h_m999 : m <= 999 := by sorry,
  exact h_m999,
end

end greatest_odd_three_digit_non_divisor_of_factorial_l788_788195


namespace shaded_triangle_area_proof_l788_788584

theorem shaded_triangle_area_proof :
  ∀ (side_length : ℝ) (initial_area : ℝ) (ratio : ℝ) (iterations : ℕ),
  side_length = 12 →
  initial_area = (sqrt 3 / 4) * 12^2 →
  ratio = 1 / 9 →
  iterations = 50 →
  let shaded_area : ℝ := initial_area * ratio / (1 - ratio) in
  shaded_area = 4.5 * sqrt 3 :=
by 
  intros side_length initial_area ratio iterations h1 h2 h3 h4 shaded_area_def
  sorry

end shaded_triangle_area_proof_l788_788584


namespace Ruby_apples_remaining_l788_788142

def Ruby_original_apples : ℕ := 6357912
def Emily_takes_apples : ℕ := 2581435
def Ruby_remaining_apples (R E : ℕ) : ℕ := R - E

theorem Ruby_apples_remaining : Ruby_remaining_apples Ruby_original_apples Emily_takes_apples = 3776477 := by
  sorry

end Ruby_apples_remaining_l788_788142


namespace sufficient_but_not_necessary_condition_for_constant_term_in_binomial_expansion_l788_788489

theorem sufficient_but_not_necessary_condition_for_constant_term_in_binomial_expansion :
  ∀ (n : ℕ), (C: ℕ) (x : ℝ), (n = 6 → ∃ (C : ℕ), ∃ (x : ℝ), ∑ i in finset.range (n+1), C * x ^ (n - 2 * i) = C * x^0)
  ∧ (∀ (n : ℕ), (even n → ∃ (C : ℕ), ∃ (x : ℝ), ∑ i in finset.range (n+1), C * x ^ (n - 2 * i) = C * x^0)) :=
by
  sorry

end sufficient_but_not_necessary_condition_for_constant_term_in_binomial_expansion_l788_788489


namespace c_younger_by_10_l788_788108

def ages (a b c d : ℕ) : Prop :=
  (a + b = b + c + 10) ∧
  (c + d = a + d - 15) ∧
  (a = (7 * d) / 4)

theorem c_younger_by_10 (a b c d : ℕ) (h : ages a b c d) : a - c = 10 :=
by
  cases h with
  | intro h1 h' =>
  cases h' with
  | intro h2 h3 =>
  rw [h1, h2, h3]
  sorry

end c_younger_by_10_l788_788108


namespace find_a_l788_788417

theorem find_a (a : ℝ) (h : ∃ x, x = -1 ∧ 4 * x^3 + 2 * a * x = 8) : a = -6 :=
sorry

end find_a_l788_788417


namespace parabola_equation_circle_m_equation_line_a2a3_tangent_to_circle_m_l788_788631

noncomputable theory
open_locale classical

-- Definitions and conditions
def parabola_vertex_origin (x y : ℝ) : Prop := ∃ p : ℝ, p > 0 ∧ y^2 = 2 * p * x
def line_intersects_parabola_perpendicularly : Prop :=
  ∃ p : ℝ, p = 1 / 2 ∧ parabola_vertex_origin 1 p

def circle_m_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1
def line_tangent_to_circle_m (l : ℝ → ℝ) : Prop := ∀ x y : ℝ, circle_m_eq x y → l x = y

def points_on_parabola_and_tangent (A1 A2 A3 : ℝ × ℝ) : Prop :=
  parabola_vertex_origin A1.1 A1.2 ∧
  parabola_vertex_origin A2.1 A2.2 ∧
  parabola_vertex_origin A3.1 A3.2 ∧
  line_tangent_to_circle_m (λ y, A1.1 * y + A1.2) ∧
  line_tangent_to_circle_m (λ y, A1.1 * y + A3.2)

-- Statements to prove
theorem parabola_equation : ∃ C : ℝ → ℝ → Prop, (C = parabola_vertex_origin) := sorry
theorem circle_m_equation : ∃ M : ℝ → ℝ → Prop, (M = circle_m_eq) := sorry
theorem line_a2a3_tangent_to_circle_m :
  ∀ A1 A2 A3 : ℝ × ℝ, 
  (points_on_parabola_and_tangent A1 A2 A3) →
  ∃ l : ℝ → ℝ, line_tangent_to_circle_m l := sorry

end parabola_equation_circle_m_equation_line_a2a3_tangent_to_circle_m_l788_788631


namespace det_rotation_matrix_75_degrees_l788_788532

theorem det_rotation_matrix_75_degrees :
  let θ : ℝ := 75 * (Real.pi / 180)
  let S := Matrix.vec₁ 2 2
  S = ![
    [Real.cos θ, -Real.sin θ],
    [Real.sin θ,  Real.cos θ]
  ]
  Matr.det S = 1 :=
by
  sorry

end det_rotation_matrix_75_degrees_l788_788532


namespace flyers_left_l788_788502

theorem flyers_left (initial_flyers : ℕ) (jack_flyers : ℕ) (rose_flyers : ℕ) (left_flyers : ℕ) :
  initial_flyers = 1236 →
  jack_flyers = 120 →
  rose_flyers = 320 →
  left_flyers = 796 →
  initial_flyers - (jack_flyers + rose_flyers) = left_flyers := 
by
  intros h_initial h_jack h_rose h_left
  rw [h_initial, h_jack, h_rose, h_left]
  simp
  sorry

end flyers_left_l788_788502


namespace find_result_l788_788867

theorem find_result :
  ∀ (n : ℝ), n = 3 → (5 * n + 4 = 19) :=
by
  intros n h
  rw h
  sorry

end find_result_l788_788867


namespace find_integer_root_of_polynomial_l788_788169

variables {b c : ℚ} {x : ℝ}

noncomputable def polynomial_has_integer_root (b c : ℚ) (r : ℝ) : Prop := x^3 + b*x + c = 0

theorem find_integer_root_of_polynomial (b c : ℚ) (h1 : polynomial_has_integer_root b c (5 - Real.sqrt 11)) 
  (h2 : ∀ r : ℚ, polynomial_has_integer_root b c r → r ∈ ℚ) : 
  ∃ r : ℝ, r = -10 ∧ polynomial_has_integer_root b c r :=
sorry

end find_integer_root_of_polynomial_l788_788169


namespace projection_matrix_l788_788342

theorem projection_matrix
  (x y : ℝ) :
  let v := ![3, 4]
  let proj_v := (v ⬝ ![x, y]) / (v ⬝ v) • v
  let proj_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
    ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]] in
  proj_v = proj_matrix.mul_vec ![x, y] :=
by
  let v := ![3, 4]
  let proj_v := (v ⬝ ![x, y]) / (v ⬝ v) • v
  let proj_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]]
  sorry

end projection_matrix_l788_788342


namespace tunnel_length_in_km_l788_788745

-- Definitions for given conditions
def train_length : ℝ := 100  -- Length of the train in meters
def train_speed_kmph : ℝ := 72  -- Speed of the train in kilometers per hour
def pass_time_minutes : ℝ := 2.5  -- Time to pass through the tunnel in minutes

-- Convert given data to consistent units
def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)  -- Speed in meters per second
def pass_time_seconds : ℝ := pass_time_minutes * 60  -- Time in seconds

-- Calculate the total distance traveled while passing through the tunnel
def total_distance_traveled : ℝ := train_speed_mps * pass_time_seconds  -- Distance in meters

-- Theorem stating the length of the tunnel
theorem tunnel_length_in_km (h1 : train_length = 100)
                            (h2 : train_speed_kmph = 72)
                            (h3 : pass_time_minutes = 2.5) :
  (total_distance_traveled - train_length) / 1000 = 2.9 :=
by
  sorry

end tunnel_length_in_km_l788_788745


namespace ellipse_major_minor_ratio_l788_788833

theorem ellipse_major_minor_ratio (m : ℝ) (x y : ℝ) (h1 : x^2 + y^2 / m = 1) (h2 : 2 * 1 = 4 * Real.sqrt m) 
  : m = 1 / 4 :=
sorry

end ellipse_major_minor_ratio_l788_788833


namespace find_a_and_b_and_intervals_of_monotonicity_l788_788432

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^3 - 3 * a * x^2 + 2 * b * x

theorem find_a_and_b_and_intervals_of_monotonicity :
  (∃ a b : ℝ, (∀ x : ℝ, f 1 a b = -1 ∧ deriv (f x a b) 1 = 0)
  → a = 1/3 ∧ b = -1/2) ∧
  (∀ x : ℝ, (deriv (λ (x : ℝ), f x (1/3) (-1/2)) x > 0 ↔ x ∈ set.union (set.Iio (-1/3)) (set.Ioi 1) ∧
  deriv (λ (x : ℝ), f x (1/3) (-1/2)) x < 0 ↔ x ∈ set.Ioo (-1/3) 1))) :=
begin
  sorry
end

end find_a_and_b_and_intervals_of_monotonicity_l788_788432


namespace units_digit_F_F10_l788_788768

def F : ℕ → ℕ
| 0       := 3
| 1       := 2
| (n + 2) := F n + F (n + 1)

-- To show that the units digit of F_{F_{10}} is 1
theorem units_digit_F_F10 : (F (F 10)) % 10 = 1 := sorry

end units_digit_F_F10_l788_788768


namespace perimeter_lt_pi_d_l788_788563

theorem perimeter_lt_pi_d {P : ℝ} {d : ℝ} (h : ∀ (s : ℝ), s ∈ sides ∨ s ∈ diagonals → s < d) : P < π * d :=
sorry

end perimeter_lt_pi_d_l788_788563


namespace octagon_enclosed_area_l788_788157

theorem octagon_enclosed_area :
  let s := 3
      octagon_area := 2 * (1 + Real.sqrt 2) * s^2
      sector_area := 8 * (1 / 2) * Real.pi
      total_enclosed_area := octagon_area + sector_area
  in total_enclosed_area = 54 + 54 * Real.sqrt 2 + 4 * Real.pi :=
by 
  let s := 3
  let octagon_area := 2 * (1 + Real.sqrt 2) * s^2
  let sector_area := 8 * (1 / 2) * Real.pi
  let total_enclosed_area := octagon_area + sector_area
  have h1 : octagon_area = 54 + 54 * Real.sqrt 2 := sorry
  have h2 : sector_area = 4 * Real.pi := sorry
  have h3 : total_enclosed_area = 54 + 54 * Real.sqrt 2 + 4 * Real.pi := sorry
  exact h3

end octagon_enclosed_area_l788_788157


namespace angle_ABC_plus_angle_ADC_eq_180_l788_788491

-- Definitions of points and angles
variables (A B C D E F : Type) 
           [IsPoint A] [IsPoint B] [IsPoint C]
           [IsPoint D] [IsPoint E]

-- Definitions of line segments and angles
variables (AC BD AB AD CE EB DF FC AF : LineSegment)
           (angle_B ADCE : Angle)

-- Conditions
axiom AC_bisects_BAD : Bisects AC angle_BAD
axiom CE_perp_AB_at_E : Perpendicular CE AB E 
axiom AE_equation : AE = (1 / 2) * (AB + AD)

-- The theorem to be proved: Relationship between angles ABC and ADC
theorem angle_ABC_plus_angle_ADC_eq_180 :
  angle_ABC + angle_ADC = 180 :=
sorry

end angle_ABC_plus_angle_ADC_eq_180_l788_788491


namespace find_lambda_mu_l788_788416

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem find_lambda_mu (A B C P : V) (λ μ : ℝ)
  (h1 : A = λ • (B + C))
  (h2 : B = (1 - 2 * μ) • (C - B)) :
  λ + μ = 3 / 4 :=
sorry

end find_lambda_mu_l788_788416


namespace actual_diameter_of_tissue_l788_788218

variable (magnified_diameter : ℝ) (magnification_factor : ℝ)

theorem actual_diameter_of_tissue 
    (h1 : magnified_diameter = 0.2) 
    (h2 : magnification_factor = 1000) : 
    magnified_diameter / magnification_factor = 0.0002 := 
  by
    sorry

end actual_diameter_of_tissue_l788_788218


namespace not_right_angle_l788_788209

-- Define the angles in the triangle and their ratios
variables {α β γ : ℝ} (h_ratio : α / β = 3 / 4) (h_sum : α + β + γ = 180)

-- The statement to be proved
theorem not_right_angle :
  α / β = 3 / 4 ∧ α + β + γ = 180 → ¬(γ = 90) :=
begin
  sorry
end

end not_right_angle_l788_788209


namespace intersection_M_N_l788_788046

def M : Set ℝ := {x | (x - 1) * (x - 4) = 0}
def N : Set ℝ := {x | (x + 1) * (x - 3) < 0}

theorem intersection_M_N :
  M ∩ N = {1} :=
sorry

end intersection_M_N_l788_788046


namespace sum_of_roots_of_quadratic_l788_788203

theorem sum_of_roots_of_quadratic (a b c : ℝ) (h_eq : a = 3 ∧ b = 6 ∧ c = -9) :
  (-b / a) = -2 :=
by
  rcases h_eq with ⟨ha, hb, hc⟩
  -- Proof goes here, but we can use sorry to skip it
  sorry

end sum_of_roots_of_quadratic_l788_788203


namespace smallest_n_for_good_sequence_l788_788742

def is_good_sequence (a : ℕ → ℝ) : Prop :=
   (∃ (a_0 : ℕ), a 0 = a_0) ∧
   (∀ i : ℕ, a (i+1) = 2 * a i + 1 ∨ a (i+1) = a i / (a i + 2)) ∧
   (∃ k : ℕ, a k = 2014)

theorem smallest_n_for_good_sequence : 
  ∀ (a : ℕ → ℝ), is_good_sequence a → ∃ n : ℕ, a n = 2014 ∧ ∀ m : ℕ, m < n → a m ≠ 2014 :=
sorry

end smallest_n_for_good_sequence_l788_788742


namespace third_side_tangent_l788_788662

theorem third_side_tangent {p q a b c : ℝ} 
    (A_on_parabola : (2*p*a^2, 2*p*a))
    (B_on_parabola : (2*p*b^2, 2*p*b))
    (C_on_parabola : (2*p*c^2, 2*p*c))
    (AB_tangent : (2*q)^2 - 4*((a+b)*((2*p)*(a^2)*(b))) = 0)
    (BC_tangent : (2*q)^2 - 4*((b+c)*((2*p)*(b^2)*(c))) = 0) : 
    (2*q)^2 - 4*((a+c)*((2*p)*(a^2)*(c))) = 0 := 
sorry

end third_side_tangent_l788_788662


namespace maximum_value_expression_l788_788358

theorem maximum_value_expression (θ₁ θ₂ θ₃ θ₄ θ₅ φ : ℝ) :
  (cos (θ₁ + φ) * sin (θ₂ + φ) + cos (θ₂ + φ) * sin (θ₃ + φ) +
   cos (θ₃ + φ) * sin (θ₄ + φ) + cos (θ₄ + φ) * sin (θ₅ + φ) +
   cos (θ₅ + φ) * sin (θ₁ + φ)) ≤ 5 / 2 :=
by
  sorry

end maximum_value_expression_l788_788358


namespace largest_non_representable_correct_largest_non_representable_not_provable_l788_788889

noncomputable def largest_non_representable (n : ℕ) : ℕ :=
  3^(n + 1) - 2^(n + 2)

theorem largest_non_representable_correct (n : ℕ) : 
  ∀ (s : ℕ), (s > 3^(n + 1) - 2^(n+2)) -> (∃ a b : ℕ, s = 2^n * a + b * 2^(n-1) * 3 ∨
  s = 2^(n-2) * (3^2 * b) ∨ s = 2^(n-3) * 3^3 ∨ ... ∨ s = 2 * 3^(n-1) ∨ s = 3^n) :=
    sorry

theorem largest_non_representable_not_provable (n : ℕ) :
  ¬ ∃ (s ≥ 0), s = 3^(n + 1) - 2^(n + 2) :=
    sorry

end largest_non_representable_correct_largest_non_representable_not_provable_l788_788889


namespace part_one_part_two_l788_788839

def f (x : ℝ) (a : ℝ) : ℝ := |2 * x + a| - |2 * x + 3|
def g (x : ℝ) : ℝ := |x - 1| - 3

theorem part_one (x : ℝ) : 
  |g(x)| < 2 ↔ (2 < x ∧ x < 6) ∨ (-4 < x ∧ x < 0) := 
sorry

theorem part_two (a : ℝ) : 
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f x₁ a = g x₂) ↔ (0 ≤ a ∧ a ≤ 6) :=
sorry

end part_one_part_two_l788_788839


namespace parabola_circle_properties_l788_788635

section ParabolaCircleTangent

variables {A1 A2 A3 P Q M : Point} 
variables {parabola : Parabola} 
variables {circle : Circle} 
variables {line_l : Line}

-- Definitions of points
def O := Point.mk 0 0
def M := Point.mk 2 0
def P := Point.mk 1 (Real.sqrt (2 * (1 / 2)))
def Q := Point.mk 1 (-Real.sqrt (2 * (1 / 2)))

-- Definition of geometrical constructs
def parabola := {p : Point // p.y^2 = p.x}
def circle := {c : Point // (c.x - 2)^2 + c.y^2 = 1}
def line_l := {l : Line // l.slope = ⊤ ∧ l.x_intercept = 1 }

-- Tangent properties for lines A1A2 and A1A3
def is_tangent {A B : Point} (l : Line) (circle : Circle) : Prop :=
  ∃ r: Real, (∥circle.center - A∥ = r) ∧ (∥circle.center - B∥ = r) ∧ (∥circle.center - (line.foot circle.center)∥ = r)

-- Theorem/Statement to prove:
theorem parabola_circle_properties :
  (parabola = {p : Point // p.y^2 = p.x}) →
  (circle = {c : Point // (c.x - 2)^2 + c.y^2 = 1}) →
  (∀ A1 A2 A3 : Point, A1 ∈ parabola → A2 ∈ parabola → A3 ∈ parabola → 
    (is_tangent (line_through A1 A2) circle) → (is_tangent (line_through A1 A3) circle) → 
    ⊥ ≤ distance_from_point_to_line (line_through A2 A3) circle.center = 1 ) :=
sorry

end ParabolaCircleTangent

end parabola_circle_properties_l788_788635


namespace t_range_find_t_max_value_of_m_l788_788019
open Real

def f (x t : ℝ) := (x ^ 3 - 6 * x ^ 2 + 3 * x + t) * exp x

theorem t_range (f : ℝ → ℝ → ℝ) (a b c t : ℝ) (h_extreme: f a t = 0 ∧ f b t = 0 ∧ f c t = 0) (h_order: a < b ∧ b < c):
  -8 < t ∧ t < 24 :=
sorry

theorem find_t (f : ℝ → ℝ → ℝ) (a b c t : ℝ) (h_extreme: f a t = 0 ∧ f b t = 0 ∧ f c t = 0) (h_order: a < b ∧ b < c) (h_eq: a + c = 2 * b^2):
  t = 8 :=
sorry

theorem max_value_of_m (f : ℝ → ℝ → ℝ) (m : ℝ) (t : ℝ) (h_m_range: 1 ≤ m ∧ m ≤ 5) (h_t_range: 0 ≤ t ∧ t ≤ 2)
  (h_ineq: ∀ x ∈ Icc 1 m, f x t ≤ x):
  m = 5 :=
sorry

end t_range_find_t_max_value_of_m_l788_788019


namespace solve_equation_l788_788528

noncomputable def floor (x : ℝ) : ℤ := int.floor x
noncomputable def ceil (x : ℝ) : ℤ := int.ceil x
noncomputable def round (x : ℝ) : ℤ := if x - x.floor < 0.5 then x.floor else x.ceil

theorem solve_equation (x : ℝ) (h₁ : 1 < x) (h₂ : x < 1.5) : 
  3 * (floor x) + 2 * (ceil x) + (round x) = 8 :=
by
  sorry

end solve_equation_l788_788528


namespace min_value_inv_sum_l788_788945

open Real

theorem min_value_inv_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) :
  3 ≤ (1 / x) + (1 / y) + (1 / z) :=
sorry

end min_value_inv_sum_l788_788945


namespace projection_onto_vector_is_expected_l788_788307

def projection_matrix (u: ℝ × ℝ) : matrix (fin 2) (fin 2) ℝ :=
  let ⟨x, y⟩ := u in 
  (1 / (x^2 + y^2)) • (matrix.col_vec u ⬝ (matrix.transpose (matrix.row_vec u)))

def expected_matrix : matrix (fin 2) (fin 2) ℝ :=
  ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]]

theorem projection_onto_vector_is_expected :
  projection_matrix (3, 4) = expected_matrix := by
  sorry

end projection_onto_vector_is_expected_l788_788307


namespace Mia_average_speed_is_18_26_l788_788119

def totalDistance : ℝ := 45 + 15 + 10
def time1 : ℝ := 45 / 15
def time2 : ℝ := 15 / 45
def time3 : ℝ := 10 / 20
def totalTime : ℝ := time1 + time2 + time3
def averageSpeed : ℝ := totalDistance / totalTime

theorem Mia_average_speed_is_18_26 :
  averageSpeed = 18.26 := 
  sorry

end Mia_average_speed_is_18_26_l788_788119


namespace divide_660_stones_into_30_heaps_l788_788962

theorem divide_660_stones_into_30_heaps :
    ∃ (heaps : Fin 30 → ℕ), (∑ i, heaps i = 660) ∧ (∀ i j, heaps i < 2 * heaps j) ∨ (heaps j < 2 * heaps i) := 
sorry

end divide_660_stones_into_30_heaps_l788_788962


namespace complement_intersection_l788_788471

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2, 4}
def N : Set ℕ := {3, 4, 5}

theorem complement_intersection (U M N : Set ℕ) :
  U = {1, 2, 3, 4, 5} → M = {1, 2, 4} → N = {3, 4, 5} →
  compl (M ∩ N) = {1, 2, 3, 5} :=
by
  intros hU hM hN
  rw [hU, hM, hN]
  sorry

end complement_intersection_l788_788471


namespace max_min_values_f_monotonicity_range_of_a_l788_788022

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (a + 1) / 2 * x^2 + 1

-- (1) Maximum and minimum values when a = -1/2
theorem max_min_values (a : ℝ) : 
   a = -1/2 →
   (∃ x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1), ∀ y ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1), f a y ≤ f a x) ∧ 
   (∃ x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1), ∀ y ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1), f a x ≤ f a y) ∧ 
   (∀ x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1), f a x = if x = Real.exp 1 then (1 / 2 + (Real.exp 1)^2 / 4) else if x = 1 then 5 / 4 else f a x) :=
begin
  sorry
end

-- (2) Monotonicity discussion of f(x)
theorem f_monotonicity (a x : ℝ) (hx : 0 < x): 
   ((a <= -1) → ((∃μ : ℝ, μ = a^2 ↔ (∀ x : ℝ, x≠ μ→f a x > 0)) ↔ (∀ x : ℝ, f a x < 0)) ) ∧
   ((a >= 0) → (∀ x : ℝ, f a x >0)):=
    
begin  
  sorry
end

-- (3) Range of a for inequality
theorem range_of_a (a x : ℝ) (h1: -1 < a) (h2 : a < 0) (hx : 0 < x): 
   1 + a / 2 * Real.log (-a) < ∀ x > 0 → 
   (a > (1 / Real.exp 1) - 1) ∧ (a < 0) :=
begin
    sorry
end

end max_min_values_f_monotonicity_range_of_a_l788_788022


namespace divide_stones_l788_788968

/-- A pile of 660 stones can be divided into 30 piles where the sizes of the piles differ by less than a factor of 2. -/
theorem divide_stones (n : ℕ) (p : ℕ) (stones : ℕ) :
  stones = 660 → p = 30 →
  ∃ (heaps : Fin p → ℕ),
    (∑ i, heaps i = stones) ∧ (∀ i j, heaps i ≤ 2 * heaps j ∧ heaps j ≤ 2 * heaps i) :=
by
  intros h1 h2
  sorry

end divide_stones_l788_788968


namespace shoes_total_price_l788_788516

-- Define the variables involved
variables (S J : ℝ)

-- Define the conditions
def condition1 : Prop := J = (1 / 4) * S
def condition2 : Prop := 6 * S + 4 * J = 560

-- Define the total price calculation
def total_price : ℝ := 6 * S

-- State the theorem and proof goal
theorem shoes_total_price (h1 : condition1 S J) (h2 : condition2 S J) : total_price S = 480 := 
sorry

end shoes_total_price_l788_788516


namespace changing_quantities_l788_788556

theorem changing_quantities (A B P Q : Point) 
  (h1 : parallel_line A B P Q) 
  (h2 : midpoint P A B) :
  count_changes A B P Q = 3 := by
  sorry 

end changing_quantities_l788_788556


namespace symmetric_jensen_inequality_l788_788781

variable {n : ℕ}
variable {F : (Fin n → ℝ) → ℝ}
variable {x y : Fin n → ℝ}

def convex_function (F : (Fin n → ℝ) → ℝ) : Prop :=
  ∀ (x y : Fin n → ℝ) (q₁ q₂ : ℝ) (hq : q₁ + q₂ = 1),
    F (λ i, q₁ * x i + q₂ * y i) ≤ q₁ * F x + q₂ * F y

theorem symmetric_jensen_inequality :
  convex_function F →
  F (λ i, (x i + y i) / 2) ≤ (F x + F y) / 2 :=
sorry

end symmetric_jensen_inequality_l788_788781


namespace min_value_of_rational_function_l788_788379

theorem min_value_of_rational_function : ∀ (x : ℝ), 0 ≤ x → ( ∃ (y : ℝ), y = 2 ∧ ∀ (x : ℝ), 0 ≤ x → \frac {4*x^2 + 8*x + 13}{6*(1 + x)} ≥ 2) :=
by
  sorry

end min_value_of_rational_function_l788_788379


namespace correct_calculation_l788_788682

theorem correct_calculation (a : ℝ) : (3 * a^3)^2 = 9 * a^6 :=
by sorry

end correct_calculation_l788_788682


namespace OH_squared_l788_788914

variables {O H A B C : Type} [inner_product_space ℝ O]

def circumcenter (a b c : ℝ) : Type := -- Definition of circumcenter (e.g., type class for properties)
 sorry -- shared space with orthocenter and triangle sides

def orthocenter (a b c : ℝ) : Type := -- Definition of orthocenter (e.g., type class for properties)
 sorry -- shared space with circumcenter and triangle sides

variables (a b c R : ℝ) (triangle : circumcenter a b c) -- Defining triangle properties
variables (orthotriangle : orthocenter a b c) -- Defining orthotriangle within the triangle properties

theorem OH_squared 
  (hR : R = 5)
  (h_side_sum : a^2 + b^2 + c^2 = 50) : 
  let OH_squared := 
    (3 * R^2 + 2 * (R^2 - (a^2 + b^2 + c^2) / 2)) in
  OH_squared = 75 :=
by
  sorry

end OH_squared_l788_788914


namespace monotonicity_condition_l788_788864

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x

theorem monotonicity_condition (a : ℝ) :
  (∀ x ∈ Set.Ioo 1 2, f a x ≥ f a 1) ↔ a ∈ Set.Ici 2 :=
by
  sorry

end monotonicity_condition_l788_788864


namespace number_of_trees_in_yard_l788_788874

theorem number_of_trees_in_yard :
  ∀ (yard_length tree_distance : ℕ), yard_length = 360 ∧ tree_distance = 12 → 
  (yard_length / tree_distance + 1 = 31) :=
by
  intros yard_length tree_distance h
  have h1 : yard_length = 360 := h.1
  have h2 : tree_distance = 12 := h.2
  sorry

end number_of_trees_in_yard_l788_788874


namespace problem1_problem2_l788_788148

-- Problem 1: Prove that the solutions of x^2 + 6x - 7 = 0 are x = -7 and x = 1
theorem problem1 (x : ℝ) : x^2 + 6*x - 7 = 0 ↔ (x = -7 ∨ x = 1) := by
  -- Proof omitted
  sorry

-- Problem 2: Prove that the solutions of 4x(2x+1) = 3(2x+1) are x = -1/2 and x = 3/4
theorem problem2 (x : ℝ) : 4*x*(2*x + 1) = 3*(2*x + 1) ↔ (x = -1/2 ∨ x = 3/4) := by
  -- Proof omitted
  sorry

end problem1_problem2_l788_788148


namespace biology_books_count_l788_788649

-- Defining combination function C(n, k)
def combination (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

-- Given conditions
def chem_books : ℕ := 8
def ways_to_pick_chem_books : ℕ := combination chem_books 2
def total_ways : ℕ := 1260

-- Prove that the number of biology books is 10
theorem biology_books_count (B : ℕ) (h1 : combination chem_books 2 = 28) (h2 : combination B 2 * 28 = total_ways) :
  B = 10 :=
by
  sorry

end biology_books_count_l788_788649


namespace sin_15_minus_sin_75_eq_neg_sqrt_6_div_2_l788_788578

-- We define the known trigonometric values
def cos_45 := real.sqrt 2 / 2
def sin_45 := real.sqrt 2 / 2
def cos_30 := real.sqrt 3 / 2
def sin_30 := 1 / 2

-- Use the trigonometric identities for the problem
def sin_15 : ℝ := cos_75
def sin_75 : ℝ := cos_15

-- Definitions for the components in terms of cosine addition formulas
def cos_75 := cos_45 * cos_30 - sin_45 * sin_30
def cos_15 := cos_45 * cos_30 + sin_45 * sin_30

-- Theorem statement
theorem sin_15_minus_sin_75_eq_neg_sqrt_6_div_2 : 
  sin_15 - sin_75 = -real.sqrt 6 / 2 := sorry

end sin_15_minus_sin_75_eq_neg_sqrt_6_div_2_l788_788578


namespace arithmetic_sequence_value_l788_788014

theorem arithmetic_sequence_value :
  ∀ (a_n : ℕ → ℤ) (d : ℤ),
    (∀ n : ℕ, a_n n = a_n 0 + ↑n * d) →
    a_n 2 = 4 →
    a_n 4 = 8 →
    a_n 10 = 20 :=
by
  intros a_n d h_arith h_a3 h_a5
  --
  sorry

end arithmetic_sequence_value_l788_788014


namespace problem_solution_l788_788263

noncomputable def problem_statement : Prop :=
  (1/2)^(-2) + Real.log 2 - Real.log (1/5) = 5

theorem problem_solution : problem_statement :=
by
  sorry

end problem_solution_l788_788263


namespace system_of_equations_solution_l788_788149

theorem system_of_equations_solution (x y z u v : ℤ) 
  (h1 : x + y + z + u = 5)
  (h2 : y + z + u + v = 1)
  (h3 : z + u + v + x = 2)
  (h4 : u + v + x + y = 0)
  (h5 : v + x + y + z = 4) :
  v = -2 ∧ x = 2 ∧ y = 1 ∧ z = 3 ∧ u = -1 := 
by 
  sorry

end system_of_equations_solution_l788_788149


namespace sin_cos_identity_l788_788039

-- Define the conditions and conclude with the condition to prove
theorem sin_cos_identity (θ : ℝ) (a : ℝ) (hθ : 0 < θ ∧ θ < π / 2) (h : cos (2 * θ) = a) : 
  sin θ * cos θ = (sqrt (1 - a^2)) / 2 :=
sorry

end sin_cos_identity_l788_788039


namespace choose_4_numbers_from_1_to_8_sum_even_l788_788871

theorem choose_4_numbers_from_1_to_8_sum_even : 
  (finset.univ : finset ℕ).filter (λ s, s.card = 4 ∧ s.sum % 2 = 0).card = 38 := by
  sorry

end choose_4_numbers_from_1_to_8_sum_even_l788_788871


namespace least_number_to_add_l788_788696

theorem least_number_to_add (a b : ℕ) (h : a = 1056) (h2 : b = 23) : ∃ n, (a + n) % b = 0 ∧ n = 2 :=
by
  have h3 : a % b = 21 := by sorry -- From the solution steps, we know 1056 % 23 = 21
  have n : ℕ := b - (a % b)
  have h4 : n = 2 := by sorry -- Calculation from the solution: 23 - 21 = 2
  use n
  split
  · have h5 : (a + n) % b = ((a % b) + n % b) % b := by sorry -- Apply modular arithmetic properties
    rw [h3, h4]
    sorry -- Complete the proof that (a + n) % b = 0
  · exact h4

end least_number_to_add_l788_788696


namespace system_solution_l788_788797

theorem system_solution (m n : ℚ) (x y : ℚ) 
  (h₁ : 2 * x + m * y = 5) 
  (h₂ : n * x - 3 * y = 2) 
  (h₃ : x = 3)
  (h₄ : y = 1) : 
  m / n = -3 / 5 :=
by sorry

end system_solution_l788_788797


namespace function_machine_output_l788_788893

theorem function_machine_output (input : ℕ) (is_input_12 : input = 12) : 
  let step1 := input * 3,
      step2 := if step1 > 20 then step1 - 7 else step1 + 10
  in step2 = 29 :=
by
  simp [is_input_12, Nat.mul_comm, Nat.succ_le_succ]
  sorry

end function_machine_output_l788_788893


namespace parabola_equation_circle_m_equation_line_a2a3_tangent_to_circle_m_l788_788632

noncomputable theory
open_locale classical

-- Definitions and conditions
def parabola_vertex_origin (x y : ℝ) : Prop := ∃ p : ℝ, p > 0 ∧ y^2 = 2 * p * x
def line_intersects_parabola_perpendicularly : Prop :=
  ∃ p : ℝ, p = 1 / 2 ∧ parabola_vertex_origin 1 p

def circle_m_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1
def line_tangent_to_circle_m (l : ℝ → ℝ) : Prop := ∀ x y : ℝ, circle_m_eq x y → l x = y

def points_on_parabola_and_tangent (A1 A2 A3 : ℝ × ℝ) : Prop :=
  parabola_vertex_origin A1.1 A1.2 ∧
  parabola_vertex_origin A2.1 A2.2 ∧
  parabola_vertex_origin A3.1 A3.2 ∧
  line_tangent_to_circle_m (λ y, A1.1 * y + A1.2) ∧
  line_tangent_to_circle_m (λ y, A1.1 * y + A3.2)

-- Statements to prove
theorem parabola_equation : ∃ C : ℝ → ℝ → Prop, (C = parabola_vertex_origin) := sorry
theorem circle_m_equation : ∃ M : ℝ → ℝ → Prop, (M = circle_m_eq) := sorry
theorem line_a2a3_tangent_to_circle_m :
  ∀ A1 A2 A3 : ℝ × ℝ, 
  (points_on_parabola_and_tangent A1 A2 A3) →
  ∃ l : ℝ → ℝ, line_tangent_to_circle_m l := sorry

end parabola_equation_circle_m_equation_line_a2a3_tangent_to_circle_m_l788_788632


namespace dodecahedron_to_cuboids_ratio_is_8_l788_788001

noncomputable def volume_ratio_dodecahedron_cuboids (s d : ℝ) (is_regular_dodecahedron : Prop)
  (horizontal_length_eq_twice_depth : Prop) (height_eq_depth : Prop) (half_volume_relationship : Prop) : Prop :=
  let V_D := (15 + 7 * Real.sqrt 5) / 4 * s^3 in
  let V_R := 2 * d^3 in
  let total_V_R := 12 * V_R in
  horizontal_length_eq_twice_depth ∧ height_eq_depth ∧ half_volume_relationship →
  V_D / total_V_R = 8

-- Now, we can define a corresponding Lean theorem to state our problem:

theorem dodecahedron_to_cuboids_ratio_is_8
  (s d : ℝ)
  (is_regular_dodecahedron : Prop)
  (horizontal_length_eq_twice_depth : Prop := ∀ d : ℝ, d > 0 → horizontal_length = 2 * d)
  (height_eq_depth : Prop := ∀ d : ℝ, d > 0 → height = d)
  (half_volume_relationship : Prop := ∀ V_D V_R : ℝ, total_V_R = 24 * d^3 ∧ total_V_R = (1/2) * V_D) :
  volume_ratio_dodecahedron_cuboids s d is_regular_dodecahedron horizontal_length_eq_twice_depth height_eq_depth half_volume_relationship :=
by
  sorry  -- Proof will be constructed here

end dodecahedron_to_cuboids_ratio_is_8_l788_788001


namespace max_value_of_f_l788_788381

def f (x : ℝ) : ℝ := 9 * x - 4 * x^2

theorem max_value_of_f :
  (∀ x : ℝ, f x ≤ 5.0625) ∧ (∃ x : ℝ, f x = 5.0625) :=
by
  sorry

end max_value_of_f_l788_788381


namespace symmetrical_placement_exists_l788_788404

-- Definitions based on the given conditions
def Pencil := ℝ × ℝ × ℝ -- A pencil's position in 3D space

-- A condition to define the pencils being identical straight circular cylinders can be represented
noncomputable def identical_cylindrical_pencils (p1 p2 : Pencil) : Prop :=
  p1 = p2

-- Now create a higher-level predicate indicating mutual touching
def mutual_touching (pencils : List Pencil) : Prop :=
  ∀ (p1 p2 : Pencil), p1 ∈ pencils → p2 ∈ pencils → p1 ≠ p2 → (∃ c : ℝ × ℝ × ℝ, common_boundary_point p1 p2 c)

-- Define the condition of common boundary point (example, this part can be highly simplified)
def common_boundary_point (p1 p2 : Pencil) (c : ℝ × ℝ × ℝ) : Prop :=
  -- Implement geometrically appropriate condition
  sorry

-- Now the main statement for proof
theorem symmetrical_placement_exists :
  ∃ (arrangement : List Pencil), arrangement.length = 6 ∧ mutual_touching arrangement :=
by
  sorry

end symmetrical_placement_exists_l788_788404


namespace _l788_788666

-- Definitions of conditions
def chessboard : Type := fin 8 × fin 8

def selects_eight_non_attacking (V : finset chessboard) :=
  V.card = 8 ∧ (∀ (x y ∈ V), x ≠ y → x.1 ≠ y.1 ∧ x.2 ≠ y.2)

def places_eight_non_attacking_rooks (P : finset chessboard) :=
  P.card = 8 ∧ (∀ (x y ∈ P), x ≠ y → x.1 ≠ y.1 ∧ x.2 ≠ y.2)

def rooks_on_selected_squares (V P : finset chessboard) : ℕ :=
  finset.card (V ∩ P)

noncomputable theorem minimum_turns_to_guarantee_win (V : finset chessboard) :
  selects_eight_non_attacking V →
  ∃ k : ℕ, k ≤ 2 ∧ ∀ P : finset chessboard, places_eight_non_attacking_rooks P → 
  if even (rooks_on_selected_squares V P) then true else ∃ P' : finset chessboard, places_eight_non_attacking_rooks P' → true :=
by
  sorry

end _l788_788666


namespace z_in_third_quadrant_l788_788006

def i := Complex.I

def z := i + 2 * (i^2) + 3 * (i^3)

theorem z_in_third_quadrant : 
    let z_real := Complex.re z
    let z_imag := Complex.im z
    z_real < 0 ∧ z_imag < 0 :=
by
  sorry

end z_in_third_quadrant_l788_788006


namespace prove_f_properties_l788_788998

theorem prove_f_properties :
  ∀ (A ω φ : ℝ), (A > 0) → (ω > 0) → (0 < φ ∧ φ < π) →
  f (x: ℝ) := A * sin (ω * x + φ) →
  (∀ (x: ℝ), f x = 2 → x = π/3) →
  (∀ (x: ℝ), sin (ω * x + φ) = 0 → ∃ k ∈ ℤ, x = k * π) →
  f x = 2 * sin (x + π/6) ∧ 
  (∀ k ∈ ℤ, -π/3 + k*π ≤ x ∧ x ≤ π/6 + k*π) ∧
  g(x) = f(x) * cos(x) - 1 → (0 < x ∧ x < π/2) →
  g(x) ∈ (-1, 1/2] := 
sorry

end prove_f_properties_l788_788998


namespace parabola_circle_properties_l788_788637

section ParabolaCircleTangent

variables {A1 A2 A3 P Q M : Point} 
variables {parabola : Parabola} 
variables {circle : Circle} 
variables {line_l : Line}

-- Definitions of points
def O := Point.mk 0 0
def M := Point.mk 2 0
def P := Point.mk 1 (Real.sqrt (2 * (1 / 2)))
def Q := Point.mk 1 (-Real.sqrt (2 * (1 / 2)))

-- Definition of geometrical constructs
def parabola := {p : Point // p.y^2 = p.x}
def circle := {c : Point // (c.x - 2)^2 + c.y^2 = 1}
def line_l := {l : Line // l.slope = ⊤ ∧ l.x_intercept = 1 }

-- Tangent properties for lines A1A2 and A1A3
def is_tangent {A B : Point} (l : Line) (circle : Circle) : Prop :=
  ∃ r: Real, (∥circle.center - A∥ = r) ∧ (∥circle.center - B∥ = r) ∧ (∥circle.center - (line.foot circle.center)∥ = r)

-- Theorem/Statement to prove:
theorem parabola_circle_properties :
  (parabola = {p : Point // p.y^2 = p.x}) →
  (circle = {c : Point // (c.x - 2)^2 + c.y^2 = 1}) →
  (∀ A1 A2 A3 : Point, A1 ∈ parabola → A2 ∈ parabola → A3 ∈ parabola → 
    (is_tangent (line_through A1 A2) circle) → (is_tangent (line_through A1 A3) circle) → 
    ⊥ ≤ distance_from_point_to_line (line_through A2 A3) circle.center = 1 ) :=
sorry

end ParabolaCircleTangent

end parabola_circle_properties_l788_788637


namespace rectangle_other_side_length_l788_788740

theorem rectangle_other_side_length (P L : ℝ) (W : ℝ) (h1 : P = 40) (h2 : L = 8) : W = 12 :=
by
  -- Using the perimeter formula for a rectangle P = 2 * (L + W)
  have h3 : P = 2 * (L + W), sorry
  -- Substituting the given values
  rw [h1, h2] at h3,
  -- Solve for W
  -- Simplification steps, shown in the narrative above, will ultimately lead to W = 12
  sorry

end rectangle_other_side_length_l788_788740


namespace find_ratio_l788_788862

variables (a b c d : ℝ)

def condition1 : Prop := a / b = 5
def condition2 : Prop := b / c = 1 / 4
def condition3 : Prop := c^2 / d = 16

theorem find_ratio (h1 : condition1 a b) (h2 : condition2 b c) (h3 : condition3 c d) :
  d / a = 1 / 25 :=
sorry

end find_ratio_l788_788862


namespace g_sum_even_function_l788_788539

def g (a b c d x : ℝ) : ℝ := a * x ^ 8 + b * x ^ 6 - c * x ^ 4 + d * x ^ 2 + 5

theorem g_sum_even_function 
  (a b c d : ℝ) 
  (h : g a b c d 2 = 4)
  : g a b c d 2 + g a b c d (-2) = 8 :=
by
  sorry

end g_sum_even_function_l788_788539


namespace find_f_at_6_l788_788730

noncomputable def example_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (4 * x - 2) = x^2 - x + 2

theorem find_f_at_6 (f : ℝ → ℝ) (h : example_function f) : f 6 = 4 := 
by
  sorry

end find_f_at_6_l788_788730


namespace tan_ratio_difference_l788_788458

variable {x y : ℝ}

theorem tan_ratio_difference (h1 : (sin x / cos y) - (sin y / cos x) = 2)
                             (h2 : (cos x / sin y) - (cos y / sin x) = 3) :
  (tan x / tan y) - (tan y / tan x) = 5 := sorry

end tan_ratio_difference_l788_788458


namespace problem_solution_l788_788009

variable (f : ℝ → ℝ)

noncomputable def solution_set (x : ℝ) : Prop :=
  (0 < x ∧ x < 1/2) ∨ (2 < x)

theorem problem_solution
  (hf_even : ∀ x, f x = f (-x))
  (hf_decreasing : ∀ x y, x < y ∧ y ≤ 0 → f x > f y)
  (hf_at_1 : f 1 = 2) :
  ∀ x, f (Real.log x / Real.log 2) > 2 ↔ solution_set x :=
by
  sorry

end problem_solution_l788_788009


namespace parabola_and_circle_tangency_relationship_l788_788645

-- Definitions for points and their tangency
def is_tangent (line : ℝ → ℝ) (circle_center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∃ x, (x - circle_center.1)^2 + (line x - circle_center.2)^2 = radius^2

theorem parabola_and_circle_tangency_relationship :
  (∀ x y: ℝ, y^2 = x → ∃ x, (x - 2)^2 + y^2 = 1) ∧
  (∀ (a1 a2 a3 : ℝ × ℝ),
    (a1.2) ^ 2 = a1.1 → 
    (a2.2) ^ 2 = a2.1 → 
    (a3.2) ^ 2 = a3.1 →
    (is_tangent (λ x, (a1.2 / (a1.1 - x))) (2, 0) 1) →
    (is_tangent (λ x, (a1.2 / (a1.1 - x))) (2, 0) 1) →
    is_tangent (λ x, (a2.2 / (a2.1 - x))) (2, 0) 1 ∧
    is_tangent (λ x, (a3.2 / (a3.1 - x))) (2, 0) 1)
  := 
sorry

end parabola_and_circle_tangency_relationship_l788_788645


namespace arithmetic_sequence_sum_l788_788484

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a (n + 1) = a n + d)
  (h_a7 : a 7 = 12) :
  a 3 + a 11 = 24 :=
by
  sorry

end arithmetic_sequence_sum_l788_788484


namespace circles_ACD_and_BCD_orthogonal_l788_788098

-- Define mathematical objects and conditions
variables (A B C D : Point) -- Points in general position on the plane
variables (circle : Point → Point → Point → Circle)

-- Circles intersect orthogonally property
def orthogonal_intersection (c1 c2 : Circle) : Prop :=
  -- Definition of orthogonal intersection of circles goes here (omitted for brevity)
  sorry

-- Given conditions
def circles_ABC_and_ABD_orthogonal : Prop :=
  orthogonal_intersection (circle A B C) (circle A B D)

-- Theorem statement
theorem circles_ACD_and_BCD_orthogonal (h : circles_ABC_and_ABD_orthogonal A B C D circle) :
  orthogonal_intersection (circle A C D) (circle B C D) :=
sorry

end circles_ACD_and_BCD_orthogonal_l788_788098


namespace passes_through_orthocenter_l788_788872

theorem passes_through_orthocenter (A B C J E F X Y : Point)
    (circle_J_passing_through_BC : Circle J ∧ point_on_circle B J ∧ point_on_circle C J)
    (E_on_AC : ∃ p : Point, point_on_line p A C ∧ p = E)
    (F_on_AB : ∃ p : Point, point_on_line p A B ∧ p = F)
    (FXB_similar_EJC : similar_triangles (Triangle F X B) (Triangle E J C))
    (X_and_C_same_side_AB : same_side X C (Line A B))
    (EYC_similar_FJB : similar_triangles (Triangle E Y C) (Triangle F J B))
    (Y_and_B_same_side_AC : same_side Y B (Line A C))
    (H : Point) -- Orthocenter of triangle ABC
    (is_orthocenter_ABC : orthocenter H (Triangle A B C)) :
    collinear X Y H :=
sorry

end passes_through_orthocenter_l788_788872


namespace prove_the_equation_l788_788497

noncomputable theory
open_locale classical

variables {A B C A1 B1 C1 : Type*} [normed_group A] [normed_group B] [normed_group C]
variables {ABC : triangle A B C}
variables {r r1 r2 r3 r4 : ℝ}
variables {λ : ℝ}

/-- Given conditions for the problem -/
def conditions (ABC : triangle A B C)
  (r r1 r2 r3 r4 : ℝ)
  (A1 B1 C1 : ABC.points_on_sides AB AC BC)
  (λ : ℝ) : Prop :=
  ∃ (λ : ℝ),
    (λ ≠ 0 ∧
    (AC_len (ABC.side AC) / (C_len (ABC.point C1))) = λ ∧
    (BA_len (ABC.side BA) / (A_len (ABC.point A1))) = λ ∧
    (CB_len (ABC.side CB) / (B_len (ABC.point B1))) = λ)

/-
Prove that if the given conditions hold, then the value of ⟨λ⟩ that satisfies the equation:
-/
theorem prove_the_equation (ABC : triangle A B C) 
  (r r1 r2 r3 r4 : ℝ)
  (A1 B1 C1 : ABC.points_on_sides AB AC BC)
  (h : conditions ABC r r1 r2 r3 r4 A1 B1 C1 λ) 
  (Heqn : (1 / r1) + (1 / r2) + (1 / r3) = (1 / r4) + (4 / r)) :
  λ = 1 :=
sorry

end prove_the_equation_l788_788497


namespace interval_for_decreasing_function_l788_788806

def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)

def g (x : ℝ) : ℝ := Real.sin (2 * x + π / 3)

theorem interval_for_decreasing_function :
  let I : Set ℝ := (Set.Ioo (π / 4) (π / 3))
  ∀ x ∈ I, ∃ ω > 0, f ω x = g x := by
  sorry

end interval_for_decreasing_function_l788_788806


namespace length_of_other_parallel_side_l788_788296

-- Definitions based on conditions
def trapezium_area (a b h : ℝ) : ℝ := 0.5 * (a + b) * h

-- Given conditions
def side1 : ℝ := 22
def height : ℝ := 15
def area : ℝ := 300

-- Theorem statement: Prove that given these conditions, the length of the other side is 18
theorem length_of_other_parallel_side (b : ℝ) (h : ℝ) (a : ℝ) : 
  (trapezium_area side1 b height = area) → b = 18 := 
by 
  -- This is a placeholder proof. You need to provide the actual proof.
  intros h_area hb
  sorry

end length_of_other_parallel_side_l788_788296


namespace projection_onto_3_4_matrix_l788_788337

def projection_matrix := λ (u : ℝ) (v : ℝ), (3 * u + 4 * v) / 25

theorem projection_onto_3_4_matrix :
  ∀ (x y : ℝ),
  (λ (u v : ℝ), (3 * x + 4 * y) / 25) = (λ (u v : ℝ), (\(u * 9 / 25) + (v * 12 / 25), (u * 12 / 25) + (v * 16 / 25))) :=
by
  sorry

end projection_onto_3_4_matrix_l788_788337


namespace simple_interest_correct_l788_788673

def principal : ℝ := 400
def rate : ℝ := 0.20
def time : ℝ := 2

def simple_interest (P R T : ℝ) : ℝ := P * R * T

theorem simple_interest_correct :
  simple_interest principal rate time = 160 :=
by
  sorry

end simple_interest_correct_l788_788673


namespace lines_perpendicular_l788_788852

-- Given the conditions
variables {l1 l2 : Type} [has_slope l1 real] [has_slope l2 real] 
variable (b : ℝ)
variable (root1 root2 : ℝ)
variable (h_roots : α^2 + b * x - 1 = 0)

-- Define the positional relationship between the lines with slopes as roots of the equation
def slopes_are_roots (l1 l2 : Type) :=
has_slope.slope l1 = root1 ∧ has_slope.slope l2 = root2

-- Formulate the theorem statement
theorem lines_perpendicular (h : slopes_are_roots l1 l2 ∧ root1 * root2 = -1) :
  ∀ (l1 l2 : Type), (has_slope.slope l1 = root1 ∧ has_slope.slope l2 = root2) →
    perpendicular l1 l2 := 
begin
  sorry
end

end lines_perpendicular_l788_788852


namespace complement_intersection_l788_788444

open Set

noncomputable def U := ℝ
noncomputable def A : Set ℝ := { x : ℝ | |x| ≥ 1 }
noncomputable def B : Set ℝ := { x : ℝ | x^2 - 2 * x - 3 > 0 }

theorem complement_intersection :
  (U \ A) ∩ (U \ B) = { x : ℝ | -1 < x ∧ x < 1 } :=
by
  sorry

end complement_intersection_l788_788444


namespace sum_of_squares_representation_prime_as_sum_of_squares_l788_788134

theorem sum_of_squares_representation 
  (x y z : ℕ)
  (h1 : x > 0 ∧ y > 0 ∧ z > 0)
  (h2 : x * y - z ^ 2 = 1) :
  ∃ (a b c d : ℕ), x = a^2 + b^2 ∧ y = c^2 + d^2 ∧ z = a * c + b * d :=
sorry

theorem prime_as_sum_of_squares (q : ℕ) (h : prime (4*q + 1)) :
  ∃ (a b : ℤ), (4*q + 1) = a^2 + b^2 :=
let z := (2*q)! in
have z_pos : (2*q)! > 0, from factorial_pos (2*q),
have z_eq : (2*q)! = z, from rfl,
begin
  -- Use the sum_of_squares_representation theorem with specific values
  have exists_squares_rep : ∃ (a b c d : ℕ), (4*q + 1) = a^2 + b^2 ∧ y = c^2 + d^2 ∧ z = a * c + b * d,
  from sum_of_squares_representation (4*q + 1) y z (by split; exact nat.succ_pos') (by sorry),
  cases exists_squares_rep with a exists_c_d,
  cases exists_c_d with b exists_d,
  cases exists_d with c exists_d,
  cases exists_d with d h_rep,
  -- Conclusion
  use [a, b],
  exact h_rep.1,
end

end sum_of_squares_representation_prime_as_sum_of_squares_l788_788134


namespace find_m_l788_788486

noncomputable def line_equation (ρ θ m : ℝ) := 
  ρ * sin(θ + π / 3) = (sqrt 3 / 2) * m

noncomputable def parametric_curve_x (θ : ℝ) := 1 + sqrt 3 * cos θ
noncomputable def parametric_curve_y (θ : ℝ) := sqrt 3 * sin θ

noncomputable def rectangular_line_equation (x y m : ℝ) := 
  sqrt 3 * x + y = sqrt 3 * m 

noncomputable def ordinary_curve_equation (x y : ℝ) := 
  (x - 1)^2 + y^2 = 3

noncomputable def chord_length (x y m : ℝ) := 
  2 * sqrt(3 - (sqrt 3 / 2 * abs (1 - m))^2)

theorem find_m (m : ℝ) : 
  0 ≤ m ∧ m ≤ 2 ↔ ∀ (x y : ℝ), 
    (rectangular_line_equation x y m) →
    (ordinary_curve_equation x y) →
    (chord_length x y m) ≥ 3 := sorry

end find_m_l788_788486


namespace parabola_and_circle_tangency_relationship_l788_788641

-- Definitions for points and their tangency
def is_tangent (line : ℝ → ℝ) (circle_center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∃ x, (x - circle_center.1)^2 + (line x - circle_center.2)^2 = radius^2

theorem parabola_and_circle_tangency_relationship :
  (∀ x y: ℝ, y^2 = x → ∃ x, (x - 2)^2 + y^2 = 1) ∧
  (∀ (a1 a2 a3 : ℝ × ℝ),
    (a1.2) ^ 2 = a1.1 → 
    (a2.2) ^ 2 = a2.1 → 
    (a3.2) ^ 2 = a3.1 →
    (is_tangent (λ x, (a1.2 / (a1.1 - x))) (2, 0) 1) →
    (is_tangent (λ x, (a1.2 / (a1.1 - x))) (2, 0) 1) →
    is_tangent (λ x, (a2.2 / (a2.1 - x))) (2, 0) 1 ∧
    is_tangent (λ x, (a3.2 / (a3.1 - x))) (2, 0) 1)
  := 
sorry

end parabola_and_circle_tangency_relationship_l788_788641


namespace find_speed_way_home_l788_788858

theorem find_speed_way_home
  (speed_to_mother : ℝ)
  (average_speed : ℝ)
  (speed_to_mother_val : speed_to_mother = 130)
  (average_speed_val : average_speed = 109) :
  ∃ v : ℝ, v = 109 * 130 / 151 := by
  sorry

end find_speed_way_home_l788_788858


namespace problem_statement_l788_788870

noncomputable def x : ℕ := (finset.range (30 - 20 + 1)).sum (λ n, 20 + n)
def y : ℕ := (finset.range (30 - 20 + 1)).filter (λ n, ((20 + n) % 2 = 0)).card

theorem problem_statement : x + y = 281 :=
by 
  have hx : x = 275 := sorry,
  have hy : y = 6 := sorry,
  rw [hx, hy],
  norm_num

end problem_statement_l788_788870


namespace fraction_of_selected_color_films_equals_five_twenty_sixths_l788_788231

noncomputable def fraction_of_selected_color_films (x y : ℕ) : ℚ :=
  let bw_films := 40 * x
  let color_films := 10 * y
  let selected_bw_films := (y / x * 1 / 100) * bw_films
  let selected_color_films := color_films
  let total_selected_films := selected_bw_films + selected_color_films
  selected_color_films / total_selected_films

theorem fraction_of_selected_color_films_equals_five_twenty_sixths (x y : ℕ) (h1 : x > 0) (h2 : y > 0) :
  fraction_of_selected_color_films x y = 5 / 26 := by
  sorry

end fraction_of_selected_color_films_equals_five_twenty_sixths_l788_788231


namespace equilateral_triangle_area_sum_l788_788944

theorem equilateral_triangle_area_sum (A B C : Point)
  (hABC : right_triangle A B C)
  (equilateral_P1_on_AB : equilateral_triangle_on_side A B)
  (equilateral_P2_on_BC : equilateral_triangle_on_side B C)
  (equilateral_P3_on_CA : equilateral_triangle_on_side C A) :
  area (P1_on_AB A B)
  = area (P2_on_BC B C) + area (P3_on_CA C A) := 
by 
  sorry

end equilateral_triangle_area_sum_l788_788944


namespace length_segment_C_C_l788_788185

theorem length_segment_C_C' :
  let C := (4, 3)
  let C' := (4, -3)
  (C.2 - C'.2).abs = 6 :=
by
  sorry

end length_segment_C_C_l788_788185


namespace geometric_sequence_sum_l788_788025

noncomputable theory

-- Variable declarations as per the problem conditions
variables {x y : ℝ} (z : ℕ → ℂ)

-- Defining the sequence based on the conditions given
def geometric_sequence (z : ℕ → ℂ) : Prop :=
  z 1 = 1 ∧
  z 2 = x + y * complex.I ∧
  z 3 = -x + y * complex.I ∧
  0 < y

-- Stating the final sum to be proved
def sum_of_sequence (z : ℕ → ℂ) : ℂ :=
  ∑ i in finset.range 2019, z i

-- Proof goal
theorem geometric_sequence_sum (hz : geometric_sequence z) :
  sum_of_sequence z = 1 + real.sqrt 3 * complex.I :=
sorry

end geometric_sequence_sum_l788_788025


namespace part1_part2_l788_788767

-- Define the conditions
def P_condition (a x : ℝ) : Prop := 1 - a / x < 0
def Q_condition (x : ℝ) : Prop := abs (x + 2) < 3

-- First part: Given a = 3, prove the solution set P
theorem part1 (x : ℝ) : P_condition 3 x ↔ 0 < x ∧ x < 3 := by 
  sorry

-- Second part: Prove the range of values for the positive number a
theorem part2 (a : ℝ) (ha : 0 < a) : 
  (∀ x, (P_condition a x → Q_condition x)) → 0 < a ∧ a ≤ 1 := by 
  sorry

end part1_part2_l788_788767


namespace distance_to_midpoint_l788_788060

-- Define the basic setup for the right triangle XYZ with the given side lengths.
structure Triangle :=
  (X Y Z : Type)
  [dist : MetricSpace (X × Y)]
  (XY : ℝ)
  (XZ : ℝ)
  (YZ : ℝ)
  (right_triangle : (XZ ^ 2 + YZ ^ 2 = XY ^ 2))

-- Define the specific triangle with the given side lengths.
def XYZ : Triangle := {
  X := ℝ,
  Y := ℝ,
  Z := ℝ,
  dist := by apply_instance,
  XY := 15,
  XZ := 9,
  YZ := 12,
  right_triangle := by {
    simp,
    linarith,
  }
}

-- Main theorem stating that the distance from Z to the midpoint of XY is 7.5 units for the given right triangle.
theorem distance_to_midpoint (T : Triangle) : 
  T.XY = 15 ∧ T.XZ = 9 ∧ T.YZ = 12 ∧ T.right_triangle 
  → (dist (T.Z, midpoint ((T.X, T.Y)) = 7.5) :=
begin
  intro h,
  sorry -- Proof will be added here.
end

end distance_to_midpoint_l788_788060


namespace minimum_value_of_f_l788_788104

noncomputable def f (x : ℝ) : ℝ := |2*x - 1| + |3*x - 2| + |4*x - 3| + |5*x - 4|

theorem minimum_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f y ≥ 1) ∧ f x = 1 :=
by
  sorry

end minimum_value_of_f_l788_788104


namespace neg_p_sufficient_not_necessary_q_l788_788093

theorem neg_p_sufficient_not_necessary_q (p q : Prop) 
  (h₁ : p → ¬q) 
  (h₂ : ¬(¬q → p)) : (q → ¬p) ∧ ¬(¬p → q) :=
sorry

end neg_p_sufficient_not_necessary_q_l788_788093


namespace divide_660_stones_into_30_piles_l788_788982

theorem divide_660_stones_into_30_piles :
  ∃ (heaps : Fin 30 → ℕ),
    (∑ i, heaps i = 660) ∧
    ∀ i j, heaps i ≤ 2 * heaps j :=
sorry

end divide_660_stones_into_30_piles_l788_788982


namespace additional_people_needed_l788_788286

theorem additional_people_needed
  (initial_people : ℕ) (initial_time : ℕ) (new_time : ℕ)
  (h_initial : initial_people * initial_time = 24)
  (h_time : new_time = 2)
  (h_initial_people : initial_people = 8)
  (h_initial_time : initial_time = 3) :
  (24 / new_time) - initial_people = 4 :=
by
  sorry

end additional_people_needed_l788_788286


namespace length_AB_parallelogram_l788_788485

variables {V : Type*} [inner_product_space ℝ V]

/-- Given a parallelogram ABCD with specific properties, prove the length of AB is 1/2. -/
theorem length_AB_parallelogram
  (A B C D E : V)
  (h_parallelogram : ∃ (u v : V), D - A = v ∧ B - A = u ∧ C - B = v ∧ A - C = -u)
  (h_AD : ∥D - A∥ = 1)
  (h_angle_BAD : real.angle (B - A) (D - A) = real.pi / 3)
  (h_E_midpoint_CD : E = (C + D) / 2)
  (h_dot_product : inner (C - A) (E - B) = 1) :
  ∥B - A∥ = 1 / 2 :=
sorry

end length_AB_parallelogram_l788_788485


namespace gcf_lcm_60_72_l788_788194

def gcf_lcm_problem (a b : ℕ) : Prop :=
  gcd a b = 12 ∧ lcm a b = 360

theorem gcf_lcm_60_72 : gcf_lcm_problem 60 72 :=
by {
  sorry
}

end gcf_lcm_60_72_l788_788194


namespace a_7_minus_a_2_l788_788030

theorem a_7_minus_a_2 (S : ℕ → ℕ) (a : ℕ → ℕ) : 
  (∀ n, n ∈ Nat → S n = 2 * n^2 - 3 * n) →
  (∀ n, n ≥ 2 → a n = S n - S (n-1)) →
  a 7 - a 2 = 20 :=
by
  intro hS ha
  sorry

end a_7_minus_a_2_l788_788030


namespace find_y_l788_788048

variable {L B y : ℝ}

theorem find_y (h1 : 2 * ((L + y) + (B + y)) - 2 * (L + B) = 16) : y = 4 :=
by
  sorry

end find_y_l788_788048


namespace eccentricity_of_hyperbola_l788_788526

variable (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
variable (P : ℝ × ℝ)
variable (PF1 PF2 : ℝ)
variable (h1 : |PF1| + |PF2| = 3 * b)
variable (h2 : |PF1| * |PF2| = (9 / 4) * a * b)

theorem eccentricity_of_hyperbola (h3 : PF1 - PF2 = 2 * a) :
  let e := (c : ℝ) / a
  ∃ (c : ℝ), c = sqrt (a ^ 2 + b ^ 2) ∧ e = 5 / 3 :=
  sorry

end eccentricity_of_hyperbola_l788_788526


namespace max_ratio_xy_l788_788095

def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem max_ratio_xy (x y : ℕ) (hx : two_digit x) (hy : two_digit y) (hmean : (x + y) / 2 = 60) : x / y ≤ 33 / 7 :=
by
  sorry

end max_ratio_xy_l788_788095


namespace divide_660_stones_into_30_piles_l788_788987

theorem divide_660_stones_into_30_piles :
  ∃ (heaps : Fin 30 → ℕ),
    (∑ i, heaps i = 660) ∧
    (∀ i j, heaps i < 2 * heaps j ∨ heaps j < 2 * heaps i) :=
sorry

end divide_660_stones_into_30_piles_l788_788987


namespace projection_of_a_in_direction_of_b_l788_788036

variables (λ : ℝ)
def a := (1 : ℝ, λ)
def b := (2 : ℝ, 1)
def c := (1 : ℝ, -2)
def collinear (v₁ v₂ : ℝ × ℝ) : Prop := ∃ k : ℝ, v₁ = (k * v₂.1, k * v₂.2)

theorem projection_of_a_in_direction_of_b :
  (collinear (2 * a.1 + b.1, 2 * a.2 + b.2) c) →
     λ = -9/2 →
     (1 / sqrt ((1:ℝ)^2 + (λ^2))) * ((1 * 2 + λ * 1) / sqrt ((2:ℝ)^2 + (1:ℝ)^2)) = -sqrt 5 / 2 :=
by
  sorry

end projection_of_a_in_direction_of_b_l788_788036


namespace parabola_and_circle_tangency_l788_788617

open Real

noncomputable def parabola_eq : Prop :=
  (parabola : {x : ℝ → ℝ | ∃ y: ℝ, y^2 = x})

noncomputable def circle_eq : Prop :=
  (circle : {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2)^2 = 1})

theorem parabola_and_circle_tangency:
  (∀ x y : ℝ, ∃ p, y^2 = x ↔ p ∈ parabola_eq) →
  ((x - 2)^2 + y^2 = 1) →
  (∀ A1 A2 A3 : ℝ × ℝ,
    A1 ∈ parabola_eq ∧ A2 ∈ parabola_eq ∧ A3 ∈ parabola_eq →
    (tangential A1 A2 circle ∧ tangential A1 A3 circle →
    tangential A2 A3 circle
  )) := sorry

end parabola_and_circle_tangency_l788_788617


namespace misha_initial_dollars_needed_l788_788120

-- Definitions based on conditions
def ore_per_day := 1
def ore_cost := 3
def wheat_bundle_cost := 12
def wheat_bundle_size := 3
def wheat_to_ore_rate := 1

structure CityRequirements :=
(ore_needed : ℕ)
(wheat_needed : ℕ)

def city_requirements : CityRequirements := { ore_needed := 3, wheat_needed := 2 }

-- The desired property to prove
theorem misha_initial_dollars_needed : 
  ∀ (days_worked ore_per_day ore_cost wheat_bundle_cost wheat_bundle_size wheat_to_ore_rate : ℕ) 
    (city_requirements : CityRequirements),
  days_worked = 3 ∧ ore_per_day = 1 ∧ ore_cost = 3 ∧ 
  wheat_bundle_cost = 12 ∧ wheat_bundle_size = 3 ∧ 
  wheat_to_ore_rate = 1 ∧ city_requirements.ore_needed = 3 ∧ 
  city_requirements.wheat_needed = 2 → 
  9 = 9 :=
begin 
  sorry
end

end misha_initial_dollars_needed_l788_788120


namespace selection_count_l788_788751

def students := {'A', 'B', 'C', 'D', 'E'}
def selected_students (s : Finset char) (n : Nat) := s.card = n

theorem selection_count :
  let s := Finset.insert 'A' (Finset.insert 'B' Finset.empty) in
  ∃ n, selected_students students n → ∃ s₁ s₂ s₃, 
    (s₁ ∈ students ∧ s₂ ∈ students ∧ s₃ ∈ students) ∧
    s₁ ≠ s₂ ∧ s₂ ≠ s₃ ∧ s₃ ≠ s₁ ∧
    (('A' ∈ {s₁, s₂, s₃}) ∧ ('B' ∉ {s₁, s₂, s₃}) ∨
     ('B' ∈ {s₁, s₂, s₃}) ∧ ('A' ∉ {s₁, s₂, s₃}) ∨
     ('A' ∉ {s₁, s₂, s₃}) ∧ ('B' ∉ {s₁, s₂, s₃})) ∧
    42 := sorry

end selection_count_l788_788751


namespace lines_intersect_at_common_point_iff_l788_788548

theorem lines_intersect_at_common_point_iff (a b : ℝ) :
  (∃ x y : ℝ, a * x + 2 * b * y + 3 * (a + b + 1) = 0 ∧ 
               b * x + 2 * (a + b + 1) * y + 3 * a = 0 ∧ 
               (a + b + 1) * x + 2 * a * y + 3 * b = 0) ↔ 
  a + b = -1/2 :=
by
  sorry

end lines_intersect_at_common_point_iff_l788_788548


namespace divide_660_stones_into_30_heaps_l788_788958

theorem divide_660_stones_into_30_heaps :
    ∃ (heaps : Fin 30 → ℕ), (∑ i, heaps i = 660) ∧ (∀ i j, heaps i < 2 * heaps j) ∨ (heaps j < 2 * heaps i) := 
sorry

end divide_660_stones_into_30_heaps_l788_788958


namespace geometric_seq_range_Sn_exists_a_arith_seq_l788_788393

def is_geometric_seq (a : ℝ) : Prop :=
  a ≠ 0 ∧ a ≠ 1

def S (a : ℝ) (n : ℕ) : ℝ :=
  if a = 2 then 2 * n else (a * ((a - 1) ^ n - 1)) / (a - 2)

def S_arith_seq (a : ℝ) : Prop :=
  S a 1 + S a 2 = 2 * S a 3

theorem geometric_seq_range_Sn (a : ℝ) (n : ℕ) (h : is_geometric_seq a) :
  S a n = (if a = 2 then 2 * n else (a * ((a - 1) ^ n - 1)) / (a - 2)) :=
sorry

theorem exists_a_arith_seq :
  ∃ a : ℝ, is_geometric_seq a ∧ S_arith_seq a ∧ a = 1/2 :=
sorry

end geometric_seq_range_Sn_exists_a_arith_seq_l788_788393


namespace key_lime_yield_l788_788260

def audrey_key_lime_juice_yield (cup_to_key_lime_juice_ratio: ℚ) (lime_juice_doubling_factor: ℚ) (tablespoons_per_cup: ℕ) (num_key_limes: ℕ) : ℚ :=
  let total_lime_juice_cups := cup_to_key_lime_juice_ratio * lime_juice_doubling_factor
  let total_lime_juice_tablespoons := total_lime_juice_cups * tablespoons_per_cup
  total_lime_juice_tablespoons / num_key_limes

-- Statement of the problem
theorem key_lime_yield :
  audrey_key_lime_juice_yield (1/4) 2 16 8 = 1 := 
by 
  sorry

end key_lime_yield_l788_788260


namespace divide_660_stones_into_30_heaps_l788_788959

theorem divide_660_stones_into_30_heaps :
    ∃ (heaps : Fin 30 → ℕ), (∑ i, heaps i = 660) ∧ (∀ i j, heaps i < 2 * heaps j) ∨ (heaps j < 2 * heaps i) := 
sorry

end divide_660_stones_into_30_heaps_l788_788959


namespace sqrt_D_sometimes_rational_sometimes_not_l788_788943

noncomputable def a (x : ℤ) : ℤ := 2 * x + 1
noncomputable def b (x : ℤ) : ℤ := 2 * x + 3
noncomputable def c (x : ℤ) : ℤ := (a x) * (b x) + 5
noncomputable def D (x : ℤ) : ℤ := (a x)^2 + (b x)^2 + (c x)^2

theorem sqrt_D_sometimes_rational_sometimes_not (x : ℤ) : 
  ∃ y : ℝ, y^2 = D x ∧ real.irreducible (D x) := 
sorry

end sqrt_D_sometimes_rational_sometimes_not_l788_788943


namespace minimum_draws_pigeonhole_principle_draws_l788_788479

theorem minimum_draws (colors: ℕ) (outcomes: ℕ) (min_outcome_repeats: ℕ): ℕ :=
  if colors = 3 ∧ outcomes = 6 ∧ min_outcome_repeats = 5 then 25 else 0

theorem pigeonhole_principle_draws 
  (colors: ℕ) (outcomes: ℕ) (min_outcome_repeats: ℕ) 
  (hc: colors = 3) (ho: outcomes = 6) (hm: min_outcome_repeats = 5):
  minimum_draws colors outcomes min_outcome_repeats = 25 :=
begin
  rw minimum_draws,
  split_ifs,
  exact h_1,
  contradiction,
end

end minimum_draws_pigeonhole_principle_draws_l788_788479


namespace solve_equation_l788_788374

noncomputable def smallest_solution : Rat :=
  (8 - Real.sqrt 145) / 3

theorem solve_equation : 
  ∃ x : ℝ, (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ x = smallest_solution := sorry

end solve_equation_l788_788374


namespace algebraic_expression_value_l788_788029

-- Define the conditions and the target expression
theorem algebraic_expression_value (a x y : ℝ) 
  (h1 : x * Real.sqrt(a * (x - a)) + y * Real.sqrt(a * (y - a)) = Real.sqrt(Real.abs(Real.log (x - a) - Real.log (a - y)))) :
  (3 * x^2 + x * y - y^2) / (x^2 - x * y + y^2) = 1/3 := by
  sorry

end algebraic_expression_value_l788_788029


namespace prob_neq_zero_l788_788681

noncomputable def probability_no_one (a b c d : ℕ) : ℚ :=
  if 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ 1 ≤ d ∧ d ≤ 6 
  then (5/6)^4 
  else 0

theorem prob_neq_zero (a b c d : ℕ) :
  (1 ≤ a) ∧ (a ≤ 6) ∧ (1 ≤ b) ∧ (b ≤ 6) ∧ (1 ≤ c) ∧ (c ≤ 6) ∧ (1 ≤ d) ∧ (d ≤ 6) →
  (a - 1) * (b - 1) * (c - 1) * (d - 1) ≠ 0 ↔ 
  probability_no_one a b c d = 625/1296 :=
by
  sorry

end prob_neq_zero_l788_788681


namespace sum_of_x_coordinates_l788_788460

noncomputable def midpoint (p1 p2 : (ℝ × ℝ)) : (ℝ × ℝ) := 
((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem sum_of_x_coordinates (x1 y1 x2 y2 : ℝ) :
  let p1 := (x1, y1),
      p2 := (x2, y2),
      (xm, ym) := midpoint p1 p2 in 
  p1 = (2, 15) ∧ p2 = (10, -6) → xm * 2 = 12 :=
by sorry

end sum_of_x_coordinates_l788_788460


namespace projection_onto_vector_is_expected_l788_788304

def projection_matrix (u: ℝ × ℝ) : matrix (fin 2) (fin 2) ℝ :=
  let ⟨x, y⟩ := u in 
  (1 / (x^2 + y^2)) • (matrix.col_vec u ⬝ (matrix.transpose (matrix.row_vec u)))

def expected_matrix : matrix (fin 2) (fin 2) ℝ :=
  ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]]

theorem projection_onto_vector_is_expected :
  projection_matrix (3, 4) = expected_matrix := by
  sorry

end projection_onto_vector_is_expected_l788_788304


namespace parabola_and_circle_eq_line_A2A3_tangent_l788_788623

-- Define the conditions of the problem
-- Vertex of the parabola at the origin and focus on the x-axis
def parabola_eq : Prop := ∃ p > 0, ∀ x y : ℝ, (y^2 = 2 * p * x ↔ (x, y) ∈ C)

-- Define line l: x = 1
def line_l (x y : ℝ) : Prop := x = 1

-- Define the parabola C and the points of intersection P and Q
def intersection_points (y : ℝ) : Prop := (1, y) ∈ C

-- Define the perpendicularity condition OP ⊥ OQ
def perpendicular_condition (P Q : ℝ × ℝ) : Prop := (∃ p > 0, P = (1, sqrt p) ∧ Q = (1, -sqrt p))

-- Define the point M and its associated circle M tangent to line l
def point_M : ℝ × ℝ := (2, 0)

def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the points A1, A2, A3 on parabola C
def on_parabola (A : ℝ × ℝ) : Prop := (∃ p > 0, A.2^2 = 2 * p * A.1)

-- Define that lines A1A2 and A1A3 are tangent to circle M
def tangent_to_circle (A₁ A₂ : ℝ × ℝ) : Prop := sorry

-- Prove the equation of parabola C and circle M
theorem parabola_and_circle_eq : (∀ x y : ℝ, y^2 = x ∧ (x - 2)^2 + y^2 = 1) :=
by
  sorry

-- Prove the position relationship between line A2A3 and circle M
theorem line_A2A3_tangent (A₁ A₂ A₃ : ℝ × ℝ) :
    on_parabola A₁ ∧ on_parabola A₂ ∧ on_parabola A₃ ∧ tangent_to_circle A₁ A₂ ∧ tangent_to_circle A₁ A₃ →
    (∃ l_tangent : ℝ, tangent_to_circle A₂ A₃) :=
by
  sorry

end parabola_and_circle_eq_line_A2A3_tangent_l788_788623


namespace intersection_P_Q_l788_788547

noncomputable theory

def P : Set ℝ := {x | |x - 1| < 1}
def Q : Set ℝ := {x | -1 < x ∧ x < 2}

theorem intersection_P_Q :
  (P ∩ Q) = {x | 0 < x ∧ x < 2} :=
sorry

end intersection_P_Q_l788_788547


namespace trapezoid_larger_base_l788_788880

variable (x y : ℝ) 
variable (midline_len diff : ℝ)
hypothesis (h1 : midline_len = 10)
hypothesis (h2 : diff = 3)
hypothesis (h3 : (x + y) / 2 = midline_len)
hypothesis (h4 : x - y = diff)

theorem trapezoid_larger_base : x = 13 := by
  sorry

end trapezoid_larger_base_l788_788880


namespace freight_train_speed_l788_788549

theorem freight_train_speed
  (v_passenger_train : ℝ) -- Speed of the passenger train in km/h
  (time_passing : ℕ) -- Time in seconds
  (car_length : ℝ) -- Length of each freight car in meters
  (gap_length : ℝ) -- Gap between freight cars in meters
  (engine_length : ℝ) -- Length of the head of the freight train in meters
  (num_cars : ℕ) -- Number of freight train cars
  (freight_train_speed_expected : ℝ) -- Expected speed of the freight train in km/h
  : v_passenger_train = 60 → 
    time_passing = 18 →
    car_length = 15.8 →
    gap_length = 1.2 →
    engine_length = 10 →
    num_cars = 30 →
    freight_train_speed_expected = 44 →
    let v_passenger_train_mph : ℝ := v_passenger_train * 1000 / 3600 in
    let distance_passenger : ℝ := v_passenger_train_mph * time_passing in
    let total_length_freight : ℝ := (car_length + gap_length) * num_cars + engine_length in
    let relative_distance : ℝ := total_length_freight - distance_passenger in
    let speed_freight_mps : ℝ := relative_distance / time_passing in
    let speed_freight_kph : ℝ := speed_freight_mps * 3.6 in
    speed_freight_kph = freight_train_speed_expected :=
by
  intros 
  sorry

end freight_train_speed_l788_788549


namespace combined_meows_l788_788182

theorem combined_meows (first_cat_freq second_cat_freq third_cat_freq : ℕ) 
  (time : ℕ) 
  (h1 : first_cat_freq = 3)
  (h2 : second_cat_freq = 2 * first_cat_freq)
  (h3 : third_cat_freq = second_cat_freq / 3)
  (h4 : time = 5) : 
  first_cat_freq * time + second_cat_freq * time + third_cat_freq * time = 55 := 
by
  sorry

end combined_meows_l788_788182


namespace total_questions_needed_l788_788118

def m_total : ℕ := 35
def p_total : ℕ := 15
def t_total : ℕ := 20

def m_written : ℕ := (3 * m_total) / 7
def p_written : ℕ := p_total / 5
def t_written : ℕ := t_total / 4

def m_remaining : ℕ := m_total - m_written
def p_remaining : ℕ := p_total - p_written
def t_remaining : ℕ := t_total - t_written

def total_remaining : ℕ := m_remaining + p_remaining + t_remaining

theorem total_questions_needed : total_remaining = 47 := by
  sorry

end total_questions_needed_l788_788118


namespace min_value_PA_PF_l788_788394

def parabola_condition (x y : ℝ) : Prop := x^2 = 4 * y
def point_A := (-1, 8 : ℝ)
def focus_F := (0, 1 : ℝ)
def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem min_value_PA_PF (P : ℝ × ℝ)
  (hP : parabola_condition P.1 P.2) :
  distance P point_A + distance P focus_F ≥ 9 := 
sorry

end min_value_PA_PF_l788_788394


namespace projection_matrix_l788_788345

theorem projection_matrix
  (x y : ℝ) :
  let v := ![3, 4]
  let proj_v := (v ⬝ ![x, y]) / (v ⬝ v) • v
  let proj_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
    ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]] in
  proj_v = proj_matrix.mul_vec ![x, y] :=
by
  let v := ![3, 4]
  let proj_v := (v ⬝ ![x, y]) / (v ⬝ v) • v
  let proj_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]]
  sorry

end projection_matrix_l788_788345


namespace intersection_complement_l788_788851

def U : Set ℤ := Set.univ
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x^2 ≠ x}
def C_U_B : Set ℤ := {x | x ≠ 0 ∧ x ≠ 1}

theorem intersection_complement :
  A ∩ C_U_B = {-1, 2} :=
by
  sorry

end intersection_complement_l788_788851


namespace coefficient_of_x_l788_788589

theorem coefficient_of_x (x : ℝ) : 
  (let n := 8;
       k := 4;
       t := binomial n k * (x ^ (1/2)) ^ (n-k) * (x ^ (-1/4)) ^ k
   in t ) = 70 :=
sorry

end coefficient_of_x_l788_788589


namespace travel_distance_increase_l788_788721

-- First, define the conditions
def miles_per_gallon := 32
def fuel_efficiency_factor := 0.80
def tank_capacity := 12

-- Then, define the new fuel efficiency after modification
def new_miles_per_gallon := miles_per_gallon / fuel_efficiency_factor

-- Assert the current distance the car can travel on a full tank
def current_distance := miles_per_gallon * tank_capacity

-- Assert the new distance the car can travel on a full tank after modification
def new_distance := new_miles_per_gallon * tank_capacity

-- Finally, state the theorem to prove the difference in travel distance after modification
theorem travel_distance_increase : new_distance - current_distance = 76.8 := by
  -- Here comes the actual proof, but it's omitted and replaced with sorry
  sorry

end travel_distance_increase_l788_788721


namespace divide_stones_l788_788967

/-- A pile of 660 stones can be divided into 30 piles where the sizes of the piles differ by less than a factor of 2. -/
theorem divide_stones (n : ℕ) (p : ℕ) (stones : ℕ) :
  stones = 660 → p = 30 →
  ∃ (heaps : Fin p → ℕ),
    (∑ i, heaps i = stones) ∧ (∀ i j, heaps i ≤ 2 * heaps j ∧ heaps j ≤ 2 * heaps i) :=
by
  intros h1 h2
  sorry

end divide_stones_l788_788967


namespace part_I_part_II_l788_788835

-- Part (Ⅰ)
def f1 (x : ℝ) : ℝ := (1 / 3) * x^3 - 2 * x^2 + 3 * x

theorem part_I : set.range (λ x, f1 x) (set.Icc 0 3) = set.Icc 0 (4 / 3) := 
sorry

-- Part (Ⅱ)
def f2 (a b x : ℝ) : ℝ := (1 / 3) * x^3 - a * x^2 + 3 * x + b
def g (f2 : ℝ → ℝ → ℝ → ℝ) (b : ℝ) (x : ℝ) : ℝ := abs (f2 a b x) - (2 / 3)

theorem part_II : (∀ b : ℝ, ∃ (n : ℕ), ∀ (f2 a b x : ℝ) (g : ℝ → ℝ → ℝ → ℝ) (x : ℝ), g f2 b x = 0 → n ≤ 4) → -2 ≤ a ∧ a ≤ 2 :=
sorry

end part_I_part_II_l788_788835


namespace largest_valid_d_l788_788782

-- Define a function that checks if a given number leads to a two-digit quotient
def is_two_digit_quotient (d : ℕ) : Prop :=
  let num := d * 100 + 72 -- (□72) where □ represented by d
  let quotient := num / 6
  10 ≤ quotient ∧ quotient < 100

-- Define the statement to prove
theorem largest_valid_d : ∃ (d : ℕ), is_two_digit_quotient(d) ∧ d = 5 := by
  -- Proof omitted as it's specified not to include proof steps
  sorry

end largest_valid_d_l788_788782


namespace imaginary_part_of_z_l788_788422

-- Define the imaginary unit i where i^2 = -1
def imaginary_unit : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := (2 + imaginary_unit) * (1 - imaginary_unit)

-- State the theorem to prove the imaginary part of z
theorem imaginary_part_of_z : Complex.im z = -1 := by
  sorry

end imaginary_part_of_z_l788_788422


namespace max_value_of_f_l788_788023

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * sin x - 3 / 2

theorem max_value_of_f (a : ℝ) (h : 2 = (countZeros fun x => f a x = 0) (0, π) 0 π) :
  ∃ c ∈ Icc 0 (π / 2), f a c = a - 3 / 2 :=
sorry

end max_value_of_f_l788_788023


namespace symmetrical_placement_exists_l788_788403

-- Definitions based on the given conditions
def Pencil := ℝ × ℝ × ℝ -- A pencil's position in 3D space

-- A condition to define the pencils being identical straight circular cylinders can be represented
noncomputable def identical_cylindrical_pencils (p1 p2 : Pencil) : Prop :=
  p1 = p2

-- Now create a higher-level predicate indicating mutual touching
def mutual_touching (pencils : List Pencil) : Prop :=
  ∀ (p1 p2 : Pencil), p1 ∈ pencils → p2 ∈ pencils → p1 ≠ p2 → (∃ c : ℝ × ℝ × ℝ, common_boundary_point p1 p2 c)

-- Define the condition of common boundary point (example, this part can be highly simplified)
def common_boundary_point (p1 p2 : Pencil) (c : ℝ × ℝ × ℝ) : Prop :=
  -- Implement geometrically appropriate condition
  sorry

-- Now the main statement for proof
theorem symmetrical_placement_exists :
  ∃ (arrangement : List Pencil), arrangement.length = 6 ∧ mutual_touching arrangement :=
by
  sorry

end symmetrical_placement_exists_l788_788403


namespace stones_partition_l788_788997

theorem stones_partition (total_stones : ℕ) (piles : ℕ) (heaps : ℕ → ℕ) 
  (h_total : total_stones = 660)
  (h_piles : piles = 30)
  (h_sum_heaps : ∑ i in range piles, heaps i = 660)
  (h_factor : ∀ i j, i < piles → j < piles → heaps i ≤ 2 * heaps j) :
  ∃ heaps : Π i : ℕ, i < piles → ℕ,
    (∑ i in range piles, heaps i = 660) ∧
    (∀ i j, i < piles → j < piles → heaps i ≤ 2 * heaps j) :=
  sorry

end stones_partition_l788_788997


namespace flyers_left_to_hand_out_l788_788511

-- Definitions for given conditions
def total_flyers : Nat := 1236
def jack_handout : Nat := 120
def rose_handout : Nat := 320

-- Statement of the problem
theorem flyers_left_to_hand_out : total_flyers - (jack_handout + rose_handout) = 796 :=
by
  -- proof goes here
  sorry

end flyers_left_to_hand_out_l788_788511


namespace largest_prime_factor_of_6889_l788_788301

-- Defining 6889 in Lean
def n : ℕ := 6889

-- Property of being a prime number
def is_prime (p : ℕ) : Prop := prime p

-- Defining prime factors of a number
def prime_factors (n : ℕ) : set ℕ := {p | p ∣ n ∧ is_prime p}

-- Defining the largest element of a set
noncomputable def largest_element (s : set ℕ) [inhabited s] : ℕ :=
  Classical.some (exists_max (λ x y, x ≤ y) s)

-- Assert that the largest prime factor of 6889 is 71
theorem largest_prime_factor_of_6889 : largest_element (prime_factors n) = 71 := 
sorry

end largest_prime_factor_of_6889_l788_788301


namespace proof_problem_l788_788435

theorem proof_problem
  (a b: ℝ) (h1: a > 0)
  (h2: ∀ x: ℝ, ax^2 - 3x + 2 > 0 ↔ x < 1 ∨ x > b) :
  a = 1 ∧ b = 2 ∧ 
  (∀ c: ℝ, c > 1 → (∀ x: ℝ, x^2 - 2(c+1)x + 4c > 0 ↔ x < 2 ∨ x > 2c)) ∧
  (∀ c: ℝ, c = 1 → (∀ x: ℝ, x^2 - 2(c+1)x + 4c > 0 ↔ x ≠ 2)) ∧
  (∀ c: ℝ, c < 1 → (∀ x: ℝ, x^2 - 2(c+1)x + 4c > 0 ↔ x > 2 ∨ x < 2c)) :=
by sorry

end proof_problem_l788_788435


namespace limit_is_eight_l788_788865

theorem limit_is_eight (f : ℝ → ℝ) (a b x₀ : ℝ) (h : ℝ) 
  (h1 : DifferentiableOn ℝ f (Set.Ioo a b)) 
  (h2 : x₀ ∈ Set.Ioo a b)
  (h3 : fderiv ℝ f x₀ = 4) :
  tendsto (λ h, (f x₀ - f (x₀ - 2 * h)) / h) (𝓝 0) (𝓝 8) :=
sorry

end limit_is_eight_l788_788865


namespace parallel_line_point_l788_788551

-- Define the conditions and the final proof
theorem parallel_line_point (a : ℝ) (h : a = 2) :
  ∃ P2 : ℝ × ℝ, P2 = (1, 3) ∧
  (∀ x : ℝ, l.{0} x = x + a) ∧
  (∀ x : ℝ, ll.{0} x = 2/3 * x - 2) :=
begin
  -- Conditions for line l and line ll
  let P₀ : ℝ × ℝ := (0, a),
  let P₁ : ℝ × ℝ := (4, 0),
  let P₂ : ℝ × ℝ := (6, 2),
  -- Line ll passes through points (4, 0) and (6, 2)
  have ll_eq : ∀ x, ll x = 2 / (6 - 4) * (x - 4),
  have ll_eq_simplified : ∀ x, ll x = x - 4,

  -- line l with slope1 passing through (0,2)
  have l_eq : ∀ x, l x = a + x,

  -- Proof that a = 2 makes line l pass through (1, 3)
  existsi (1, 3),
  split,
  simp,
  split,
  intros x, rw l_eq, simp [h, add_comm],
  intros x, rw ll_eq_simplified, simp,
end

end parallel_line_point_l788_788551


namespace flyers_left_to_hand_out_l788_788510

-- Definitions for given conditions
def total_flyers : Nat := 1236
def jack_handout : Nat := 120
def rose_handout : Nat := 320

-- Statement of the problem
theorem flyers_left_to_hand_out : total_flyers - (jack_handout + rose_handout) = 796 :=
by
  -- proof goes here
  sorry

end flyers_left_to_hand_out_l788_788510


namespace parity_of_f_periodic_of_f_min_max_of_f_l788_788428

-- Definitions and Conditions
def domain (x : ℝ) : Prop := ∀ k : ℤ, x ≠ k * Real.pi
def f (x : ℝ) : ℝ := -- definition is unknown, assumed to be defined correctly.

axiom h1 : ∀ (x y : ℝ), domain x → domain y → domain (x - y) → f (x - y) = f x - f y
axiom h2 : ∃ (a : ℝ), a > 0 ∧ f a = 1
axiom h3 : ∀ (x : ℝ), 0 < x → x < 2 * a → f x > 0

-- Problem Statements
theorem parity_of_f : ∀ x : ℝ, domain x → f (-x) = -f x :=
sorry

theorem periodic_of_f : ∃ p : ℝ, p = 4 * a ∧ ∀ x : ℝ, domain x → domain (x + p) → f (x + p) = f x :=
sorry

theorem min_max_of_f : ∀ x : ℝ, 2 * a ≤ x → x ≤ 3 * a → (f 2 * a = 0) ∧ (f 3 * a = -1) :=
sorry

end parity_of_f_periodic_of_f_min_max_of_f_l788_788428


namespace greatest_x_for_factorial_l788_788196

def factorial (n : ℕ) : ℕ := if h : n = 0 then 1 else n * factorial (n - 1)

theorem greatest_x_for_factorial (x : ℕ) :
  greatestValue x (λ n, 4^x ∣ factorial n) = 16 :=
sorry

end greatest_x_for_factorial_l788_788196


namespace roots_operation_zero_l788_788272

def operation (a b : ℝ) : ℝ := a * b - a - b

theorem roots_operation_zero {x1 x2 : ℝ}
  (h1 : x1 + x2 = -1)
  (h2 : x1 * x2 = -1) :
  operation x1 x2 = 0 :=
by
  sorry

end roots_operation_zero_l788_788272


namespace k_eq_3_l788_788294

-- Definition of the problem conditions
def isReverse := ∀ (m n : ℕ), m = Nat.reverseDigits n

-- Statement of the problem
theorem k_eq_3 (k : ℕ) (h1 : k > 1) (a b : ℕ) (h_a : a > 0) (h_b : b > 0) (h_ab : a ≠ b)
  (h_rev : isReverse (k ^ a + 1) (k ^ b + 1)) : k = 3 :=
by
  sorry

end k_eq_3_l788_788294


namespace centroid_coincide_l788_788896

theorem centroid_coincide (A B C D E F G H I : Type) [Point A B C D E F G H I] 
  (h1 : Segment AB D)
  (h2 : Segment BC E)
  (h3 : Segment CA F)
  -- The condition that the ratio of the segments are equal and not equal to 1
  (h4 : Ratio AD DB BE EC CF FA)
  (h5 : Ratio AD DB ≠ 1)
  -- The intersections definition
  (h6 : Intersect AE BF G)
  (h7 : Intersect AE CD H)
  (h8 : Intersect BF CD I) :
  Centroid (Triangle A B C) = Centroid (Triangle G H I) := sorry

end centroid_coincide_l788_788896


namespace dihedral_angle_ABC_to_G_is_pi_minus_arctan_sqrt2_l788_788809

def regular_tetrahedron (A B C D E F G: Point) : Prop :=
  -- Definitions needed to define a regular tetrahedron with midpoints E, F, G on edges AB, BC, CD
  midpoint E A B ∧ midpoint F B C ∧ midpoint G C D ∧ 
  (∀ X Y Z : Point, regular_tetrahedron_property X Y Z A B C D)

theorem dihedral_angle_ABC_to_G_is_pi_minus_arctan_sqrt2 (A B C D E F G : Point)
  (h1 : regular_tetrahedron A B C D E F G)
  :
  dihedral_angle C F G E = π - arctan (sqrt 2) :=
sorry

end dihedral_angle_ABC_to_G_is_pi_minus_arctan_sqrt2_l788_788809


namespace complement_of_supplement_of_35_degree_l788_788193

def angle : ℝ := 35
def supplement (x : ℝ) : ℝ := 180 - x
def complement (x : ℝ) : ℝ := 90 - x

theorem complement_of_supplement_of_35_degree :
  complement (supplement angle) = -55 := by
  sorry

end complement_of_supplement_of_35_degree_l788_788193


namespace lines_intersect_or_parallel_l788_788588

variables (A B C A1 A2 B1 B2 C1 C2 P Q R : Type)
variables (circle : set A)
variables (BC CA AB line_AB line_BC line_CA l_a l_b l_c : set A)
variables (A1 A2 B1 B2 C1 C2 P Q R : A)

-- Conditions
axiom circle_intersects_BC : circle ∩ BC = {A1, A2}
axiom circle_intersects_CA : circle ∩ CA = {B1, B2}
axiom circle_intersects_AB : circle ∩ AB = {C1, C2}
axiom l_a_defined : l_a = (line_of_intersection (line_from_points B B1) (line_from_points C C2) ∩ 
                           line_of_intersection (line_from_points B B2) (line_from_points C C1))
axiom l_b_defined : l_b = -- similarly defined using permutation of points
axiom l_c_defined : l_c = -- similarly defined using permutation of points

theorem lines_intersect_or_parallel : 
  ∃ P : A, P ∈ l_a ∧ P ∈ l_b ∧ P ∈ l_c ∨ 
  (l_a = l_b ∧ l_b = l_c ∧ collinear l_a l_b l_c) :=
sorry

end lines_intersect_or_parallel_l788_788588


namespace problem_pairs_of_integers_satisfying_equation_l788_788168

theorem problem_pairs_of_integers_satisfying_equation :
  ({(a : ℤ) × (b : ℤ) | a^b = 64}).toFinset.card = 6 := by
  sorry

end problem_pairs_of_integers_satisfying_equation_l788_788168


namespace arithmetic_sequence_cubes_l788_788378

-- Given an arithmetic sequence of five integers represented as c-2d, c-d, c, c+d, c+2d
-- Prove that these sequences satisfy the given conditions.

theorem arithmetic_sequence_cubes (c d : ℤ) :
  let a1 := c - 2 * d
  let a2 := c - d
  let a3 := c
  let a4 := c + d
  let a5 := c + 2 * d
  (a1^3 + a2^3 + a3^3 + a4^3 = 16 * (a1 + a2 + a3 + a4)^2) ∧
  (a2^3 + a3^3 + a4^3 + a5^3 = 16 * (a2 + a3 + a4 + a5)^2) →
  ∃ (c d : ℤ), (c = 32) ∧ (d = 16) :=
begin
  -- Proof is omitted
  sorry
end

end arithmetic_sequence_cubes_l788_788378


namespace number_satisfies_conditions_l788_788206

def digits_match (x n : ℕ) : Prop :=
  let product := x * n in
  let first_digit := product / 10^(product.digits.length - 1) in
  let last_digit := product % 10 in
  first_digit = n / 10 ∧ last_digit = n % 10

theorem number_satisfies_conditions :
  let x := 987654321 in
  digits_match x 18 ∧
  digits_match x 27 ∧
  digits_match x 36 ∧
  digits_match x 45 ∧
  digits_match x 54 ∧
  digits_match x 63 ∧
  digits_match x 72 ∧
  digits_match x 81 ∧
  digits_match x 99 ∧
  (x * 90) % 100 = 90 :=
by
  sorry

end number_satisfies_conditions_l788_788206


namespace company_total_payment_correct_l788_788475

def totalEmployees : Nat := 450
def firstGroup : Nat := 150
def secondGroup : Nat := 200
def thirdGroup : Nat := 100

def firstBaseSalary : Nat := 2000
def secondBaseSalary : Nat := 2500
def thirdBaseSalary : Nat := 3000

def firstInitialBonus : Nat := 500
def secondInitialBenefit : Nat := 400
def thirdInitialBenefit : Nat := 600

def firstLayoffRound1 : Nat := (20 * firstGroup) / 100
def secondLayoffRound1 : Nat := (25 * secondGroup) / 100
def thirdLayoffRound1 : Nat := (15 * thirdGroup) / 100

def remainingFirstGroupRound1 : Nat := firstGroup - firstLayoffRound1
def remainingSecondGroupRound1 : Nat := secondGroup - secondLayoffRound1
def remainingThirdGroupRound1 : Nat := thirdGroup - thirdLayoffRound1

def firstAdjustedBonusRound1 : Nat := 400
def secondAdjustedBenefitRound1 : Nat := 300

def firstLayoffRound2 : Nat := (10 * remainingFirstGroupRound1) / 100
def secondLayoffRound2 : Nat := (15 * remainingSecondGroupRound1) / 100
def thirdLayoffRound2 : Nat := (5 * remainingThirdGroupRound1) / 100

def remainingFirstGroupRound2 : Nat := remainingFirstGroupRound1 - firstLayoffRound2
def remainingSecondGroupRound2 : Nat := remainingSecondGroupRound1 - secondLayoffRound2
def remainingThirdGroupRound2 : Nat := remainingThirdGroupRound1 - thirdLayoffRound2

def thirdAdjustedBenefitRound2 : Nat := (80 * thirdInitialBenefit) / 100

def totalBaseSalary : Nat :=
  (remainingFirstGroupRound2 * firstBaseSalary)
  + (remainingSecondGroupRound2 * secondBaseSalary)
  + (remainingThirdGroupRound2 * thirdBaseSalary)

def totalBonusesAndBenefits : Nat :=
  (remainingFirstGroupRound2 * firstAdjustedBonusRound1)
  + (remainingSecondGroupRound2 * secondAdjustedBenefitRound1)
  + (remainingThirdGroupRound2 * thirdAdjustedBenefitRound2)

def totalPayment : Nat :=
  totalBaseSalary + totalBonusesAndBenefits

theorem company_total_payment_correct :
  totalPayment = 893200 :=
by
  -- proof steps
  sorry

end company_total_payment_correct_l788_788475


namespace theta_in_third_quadrant_l788_788007

theorem theta_in_third_quadrant (θ : ℝ) (h1 : Real.tan θ > 0) (h2 : Real.sin θ < 0) : 
  ∃ q : ℕ, q = 3 := 
sorry

end theta_in_third_quadrant_l788_788007


namespace trajectory_eqn_of_point_Q_l788_788658

theorem trajectory_eqn_of_point_Q 
  (P : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (A : ℝ × ℝ := (-2, 0))
  (B : ℝ × ℝ := (2, 0))
  (l : ℝ := 10 / 3) 
  (hP_on_l : P.1 = l)
  (hQ_on_AP : (Q.2 * -4) = Q.1 * (P.2 - 0) - (P.2 * -4))
  (hBP_perp_BQ : (Q.2 * 4) = -Q.1 * ((3 * P.2) / 4 - 2))
: (Q.1^2 / 4) + Q.2^2 = 1 :=
sorry

end trajectory_eqn_of_point_Q_l788_788658


namespace proof_equivalence_l788_788850

open Set

noncomputable theory
def U := {x : ℤ | -3 < x ∧ x < 3}
def A := {1, 2} : Set ℤ
def B := {-2, -1, 2} : Set ℤ

theorem proof_equivalence :
  A ∪ (U \ B) = {0, 1, 2} :=
by
  sorry

end proof_equivalence_l788_788850


namespace alternating_sum_modulo_4020_l788_788775

theorem alternating_sum_modulo_4020 : 
  let T := (Finset.range 2010).sum (λ n => (-1)^n * 2 * (n + 1))
  in T % 4020 = 2010 :=
by
  let T := (Finset.range 2010).sum (λ n => (-1)^n * 2 * (n + 1))
  have ht : T = 2010 := sorry
  show T % 4020 = 2010
  rw ht
  exact mod_same_mod 4020

end alternating_sum_modulo_4020_l788_788775


namespace origin_inside_ellipse_l788_788013

theorem origin_inside_ellipse (k : ℝ) (h : k^2 * 0^2 + 0^2 - 4*k*0 + 2*k*0 + k^2 - 1 < 0) : 0 < |k| ∧ |k| < 1 :=
by
  sorry

end origin_inside_ellipse_l788_788013


namespace triangle_construction_feasible_l788_788447

theorem triangle_construction_feasible (a b s : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (a - b) / 2 < s) (h4 : s < (a + b) / 2) :
  ∃ c, (a + b > c ∧ b + c > a ∧ c + a > b) :=
sorry

end triangle_construction_feasible_l788_788447


namespace probability_ge_sqrt2_l788_788445

noncomputable def probability_length_chord_ge_sqrt2
  (a : ℝ)
  (h : a ≠ 0)
  (intersect_cond : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ (x - a)^2 + (y - a)^2 = 1)
  : ℝ :=
  if -1 ≤ a ∧ a ≤ 1 then (1 / Real.sqrt (1^2 + 1^2)) else 0

theorem probability_ge_sqrt2 
  (a : ℝ) 
  (h : a ≠ 0) 
  (intersect_cond : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ (x - a)^2 + (y - a)^2 = 1)
  (length_cond : (Real.sqrt (4 - 2*a^2) ≥ Real.sqrt 2)) : 
  probability_length_chord_ge_sqrt2 a h intersect_cond = (Real.sqrt 2 / 2) :=
by
  sorry

end probability_ge_sqrt2_l788_788445


namespace boat_distance_downstream_is_68_l788_788719

variable (boat_speed : ℕ) (stream_speed : ℕ) (time_hours : ℕ)

-- Given conditions
def effective_speed_downstream (boat_speed stream_speed : ℕ) : ℕ := boat_speed + stream_speed
def distance_downstream (speed time : ℕ) : ℕ := speed * time

theorem boat_distance_downstream_is_68 
  (h1 : boat_speed = 13) 
  (h2 : stream_speed = 4) 
  (h3 : time_hours = 4) : 
  distance_downstream (effective_speed_downstream boat_speed stream_speed) time_hours = 68 := 
by 
  sorry

end boat_distance_downstream_is_68_l788_788719


namespace boat_distance_downstream_is_68_l788_788718

variable (boat_speed : ℕ) (stream_speed : ℕ) (time_hours : ℕ)

-- Given conditions
def effective_speed_downstream (boat_speed stream_speed : ℕ) : ℕ := boat_speed + stream_speed
def distance_downstream (speed time : ℕ) : ℕ := speed * time

theorem boat_distance_downstream_is_68 
  (h1 : boat_speed = 13) 
  (h2 : stream_speed = 4) 
  (h3 : time_hours = 4) : 
  distance_downstream (effective_speed_downstream boat_speed stream_speed) time_hours = 68 := 
by 
  sorry

end boat_distance_downstream_is_68_l788_788718


namespace pile_division_660_stones_l788_788976

theorem pile_division_660_stones (n : ℕ) (heaps : List ℕ) :
  n = 660 ∧ heaps.length = 30 ∧ ∀ x ∈ heaps, ∀ y ∈ heaps, (x ≤ 2 * y ∧ y ≤ 2 * x) →
  ∃ heaps : List ℕ, n = 660 ∧ heaps.length = 30 ∧ (∀ x y ∈ heaps, x ≤ 2 * y ∧ y ≤ 2 * x) :=
by
  sorry

end pile_division_660_stones_l788_788976


namespace perfect_squares_difference_l788_788085

theorem perfect_squares_difference :
  ∀ (a b : ℕ),
  (∃ x y : ℕ, a = x^2 ∧ b = y^2 ∧ a * b = a + b + 4844) →
  (((sqrt a + 1) * (sqrt b + 1) * (sqrt a - 1) * (sqrt b - 1)) - 
   ((sqrt 68 + 1) * (sqrt 63 + 1) * (sqrt 68 - 1) * (sqrt 63 - 1)) =  691) :=
by
  intro a b
  intro h
  sorry

end perfect_squares_difference_l788_788085


namespace divisors_of_m2_l788_788101

def m := 2^15 * 5^21

theorem divisors_of_m2 (m := 2^15 * 5^21) :
  let m_square := m * m in
  let total_divisors_m2 := (30 + 1) * (42 + 1) in
  let total_divisors_m := (15 + 1) * (21 + 1) in
  let divisors_lt_m := (total_divisors_m2 - 1) / 2 in
  let result := divisors_lt_m - total_divisors_m in
  result = 314 :=
by
  sorry

end divisors_of_m2_l788_788101


namespace probability_two_yellow_out_of_three_draws_l788_788226

noncomputable def probability_of_exactly_n_yellow_balls (num_red num_yellow total_draws succ_draws : ℕ) : ℚ :=
  let p_yellow := (num_yellow : ℚ) / (num_red + num_yellow)
  let binom_coeff := (nat.choose total_draws succ_draws : ℚ)
  binom_coeff * p_yellow^succ_draws * (1 - p_yellow)^(total_draws - succ_draws)

theorem probability_two_yellow_out_of_three_draws :
  probability_of_exactly_n_yellow_balls 2 3 3 2 = 54 / 125 := 
sorry

end probability_two_yellow_out_of_three_draws_l788_788226


namespace james_sold_percentage_for_80_percent_l788_788073

noncomputable def sold_percentage (P : ℝ) : Prop :=
  let old_car_value : ℝ := 20000
  let new_car_sticker_price : ℝ := 30000
  let new_car_discounted_price : ℝ := new_car_sticker_price * 0.9
  let out_of_pocket : ℝ := 11000
  new_car_discounted_price - old_car_value * (P / 100) = out_of_pocket

theorem james_sold_percentage_for_80_percent :
  sold_percentage 80 :=
by
  simp [sold_percentage]
  norm_num
  sorry

end james_sold_percentage_for_80_percent_l788_788073


namespace sin_neg_p_l788_788438

theorem sin_neg_p (a : ℝ) : (¬ ∃ x : ℝ, Real.sin x > a) → (a ≥ 1) := 
by
  sorry

end sin_neg_p_l788_788438


namespace sequence_formula_l788_788031

theorem sequence_formula (a : ℕ → ℝ) 
  (h₁ : a 1 = 1) 
  (h₂ : ∀ n : ℕ, a (n + 1) = a n / (3 * a n + 1)) :
  ∀ n : ℕ, a n = 1 / (3 * n - 2) :=
begin
  sorry
end

end sequence_formula_l788_788031


namespace power_function_at_100_l788_788437

-- Given a power function f(x) = x^α that passes through the point (9, 3),
-- show that f(100) = 10.

theorem power_function_at_100 (α : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x ^ α)
  (h2 : f 9 = 3) : f 100 = 10 :=
sorry

end power_function_at_100_l788_788437


namespace parity_of_f_minimum_value_of_f_l788_788533

noncomputable def f (x a : ℝ) : ℝ := x^2 + |x - a| - 1

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f (x)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

theorem parity_of_f (a : ℝ) :
  (a = 0 → is_even_function (f a)) ∧
  (a ≠ 0 → ¬is_even_function (f a) ∧ ¬is_odd_function (f a)) := 
by sorry

theorem minimum_value_of_f (a : ℝ) :
  (a ≤ -1/2 → ∀ x : ℝ, f x a ≥ -a - 5 / 4) ∧
  (-1/2 < a ∧ a ≤ 1/2 → ∀ x : ℝ, f x a ≥ a^2 - 1) ∧
  (a > 1/2 → ∀ x : ℝ, f x a ≥ a - 5 / 4) :=
by sorry

end parity_of_f_minimum_value_of_f_l788_788533


namespace smallest_solution_is_39_over_8_l788_788365

noncomputable def smallest_solution (x : ℝ) : Prop :=
  (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ (x ≠ 0) ∧ (x ≠ 3)

theorem smallest_solution_is_39_over_8 : ∃ x > 0, smallest_solution x ∧ x = 39 / 8 :=
by
  sorry

end smallest_solution_is_39_over_8_l788_788365


namespace sum_of_pyramid_edges_l788_788741

-- Declare initial conditions as given in the problem statement
structure right_pyramid :=
(square_side : ℝ)
(peak_height_above_center : ℝ)
(sum_of_edges : ℝ)

-- Define the given conditions
def pyramid_conditions : right_pyramid :=
{ square_side := 8,
  peak_height_above_center := 10,
  sum_of_edges := 78 }

-- The theorem that needs to be proved
theorem sum_of_pyramid_edges (p : right_pyramid) :
  p.sum_of_edges = 4 * p.square_side + 4 * Real.sqrt (p.peak_height_above_center^2 + (p.square_side*Real.sqrt 2 / 2)^2) :=
by
  -- Assume the given conditions
  let p := pyramid_conditions,
  sorry

end sum_of_pyramid_edges_l788_788741


namespace projection_matrix_correct_l788_788316

variables {R : Type*} [field R] [decidable_eq R]
variables (x y : R)
def vector_v : matrix (fin 2) (fin 1) R := ![![3], ![4]]
def vector_u : matrix (fin 2) (fin 1) R := ![![x], ![y]]
def projection_matrix : matrix (fin 2) (fin 2) R := ![![9/25, 12/25], ![12/25, 16/25]]

theorem projection_matrix_correct :
  (projection_matrix R) ⬝ (vector_u x y) = (25 : R)⁻¹ • (transpose (vector_v 3 4) ⬝ (vector_u x y)) ⬝ (vector_v 3 4) := 
sorry

end projection_matrix_correct_l788_788316


namespace parabola_and_circle_tangency_relationship_l788_788644

-- Definitions for points and their tangency
def is_tangent (line : ℝ → ℝ) (circle_center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∃ x, (x - circle_center.1)^2 + (line x - circle_center.2)^2 = radius^2

theorem parabola_and_circle_tangency_relationship :
  (∀ x y: ℝ, y^2 = x → ∃ x, (x - 2)^2 + y^2 = 1) ∧
  (∀ (a1 a2 a3 : ℝ × ℝ),
    (a1.2) ^ 2 = a1.1 → 
    (a2.2) ^ 2 = a2.1 → 
    (a3.2) ^ 2 = a3.1 →
    (is_tangent (λ x, (a1.2 / (a1.1 - x))) (2, 0) 1) →
    (is_tangent (λ x, (a1.2 / (a1.1 - x))) (2, 0) 1) →
    is_tangent (λ x, (a2.2 / (a2.1 - x))) (2, 0) 1 ∧
    is_tangent (λ x, (a3.2 / (a3.1 - x))) (2, 0) 1)
  := 
sorry

end parabola_and_circle_tangency_relationship_l788_788644


namespace parabola_and_circle_eq_line_A2A3_tangent_l788_788626

-- Define the conditions of the problem
-- Vertex of the parabola at the origin and focus on the x-axis
def parabola_eq : Prop := ∃ p > 0, ∀ x y : ℝ, (y^2 = 2 * p * x ↔ (x, y) ∈ C)

-- Define line l: x = 1
def line_l (x y : ℝ) : Prop := x = 1

-- Define the parabola C and the points of intersection P and Q
def intersection_points (y : ℝ) : Prop := (1, y) ∈ C

-- Define the perpendicularity condition OP ⊥ OQ
def perpendicular_condition (P Q : ℝ × ℝ) : Prop := (∃ p > 0, P = (1, sqrt p) ∧ Q = (1, -sqrt p))

-- Define the point M and its associated circle M tangent to line l
def point_M : ℝ × ℝ := (2, 0)

def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the points A1, A2, A3 on parabola C
def on_parabola (A : ℝ × ℝ) : Prop := (∃ p > 0, A.2^2 = 2 * p * A.1)

-- Define that lines A1A2 and A1A3 are tangent to circle M
def tangent_to_circle (A₁ A₂ : ℝ × ℝ) : Prop := sorry

-- Prove the equation of parabola C and circle M
theorem parabola_and_circle_eq : (∀ x y : ℝ, y^2 = x ∧ (x - 2)^2 + y^2 = 1) :=
by
  sorry

-- Prove the position relationship between line A2A3 and circle M
theorem line_A2A3_tangent (A₁ A₂ A₃ : ℝ × ℝ) :
    on_parabola A₁ ∧ on_parabola A₂ ∧ on_parabola A₃ ∧ tangent_to_circle A₁ A₂ ∧ tangent_to_circle A₁ A₃ →
    (∃ l_tangent : ℝ, tangent_to_circle A₂ A₃) :=
by
  sorry

end parabola_and_circle_eq_line_A2A3_tangent_l788_788626


namespace number_of_students_before_new_year_l788_788698

variables (M N k ℓ : ℕ)
hypotheses (h1 : 100 * M = k * N)
             (h2 : 100 * (M + 1) = ℓ * (N + 3))
             (h3 : ℓ < 100)

theorem number_of_students_before_new_year (h1 : 100 * M = k * N)
                                             (h2 : 100 * (M + 1) = ℓ * (N + 3))
                                             (h3 : ℓ < 100) :
  N ≤ 197 :=
sorry

end number_of_students_before_new_year_l788_788698


namespace son_working_alone_l788_788689

theorem son_working_alone (M S : ℝ) (h1: M = 1 / 5) (h2: M + S = 1 / 3) : 1 / S = 7.5 :=
  by
  sorry

end son_working_alone_l788_788689


namespace neg_09_not_in_integers_l788_788709

def negative_numbers : Set ℝ := {x | x < 0}
def fractions : Set ℝ := {x | ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b}
def integers : Set ℝ := {x | ∃ (n : ℤ), x = n}
def rational_numbers : Set ℝ := {x | ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b}

theorem neg_09_not_in_integers : -0.9 ∉ integers :=
by {
  sorry
}

end neg_09_not_in_integers_l788_788709


namespace mother_hen_heavier_l788_788150

-- Define the weights in kilograms
def weight_mother_hen : ℝ := 2.3
def weight_baby_chick : ℝ := 0.4

-- State the theorem with the final correct answer
theorem mother_hen_heavier :
  weight_mother_hen - weight_baby_chick = 1.9 :=
by
  sorry

end mother_hen_heavier_l788_788150


namespace monotonicity_f_on_0_2_l788_788020

-- Define the function f
def f (x : ℝ) : ℝ := x + (4 / x)

-- First condition: f(1) = 5
def condition1 := f 1 = 5

-- Second condition: f(2) = 4
def condition2 := f 2 = 4

-- Prove that function f is decreasing on (0, 2)
theorem monotonicity_f_on_0_2 :
  (∀ (x : ℝ), 0 < x ∧ x < 2 → f x = x + 4 / x) →
  (∀ (x1 x2 : ℝ), 0 < x1 ∧ x1 < 2 ∧ 0 < x2 ∧ x2 < 2 ∧ x1 < x2 → (f x1 > f x2)) :=
by
  intros h x1 x2 hx1 hx2 hlt
  -- The proof of this theorem is omitted
  sorry

end monotonicity_f_on_0_2_l788_788020


namespace arithmetic_sequence_general_term_l788_788026

theorem arithmetic_sequence_general_term (a : ℕ → ℤ) (d : ℤ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_increasing : d > 0)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = a 2 ^ 2 - 4) :
  ∀ n, a n = 2 * n - 1 :=
by
  sorry

end arithmetic_sequence_general_term_l788_788026


namespace oh_squared_l788_788924

theorem oh_squared (O H : ℝ) (a b c R : ℝ) (h1 : R = 5) (h2 : a^2 + b^2 + c^2 = 50) :
  let OH := H - O in
  OH ^ 2 = 175 :=
by
  sorry

end oh_squared_l788_788924


namespace remaining_distance_l788_788161

-- Definitions of the given conditions
def D : ℕ := 500
def daily_alpha : ℕ := 30
def daily_beta : ℕ := 50
def effective_beta : ℕ := daily_beta / 2

-- Proving the theorem with given conditions
theorem remaining_distance (n : ℕ) (h : n = 25) :
  D - daily_alpha * n = 2 * (D - effective_beta * n) :=
by
  sorry

end remaining_distance_l788_788161


namespace smallest_solution_to_equation_l788_788369

noncomputable def smallest_solution := (11 - Real.sqrt 445) / 6

theorem smallest_solution_to_equation:
  ∃ x : ℝ, (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ (x = smallest_solution) :=
sorry

end smallest_solution_to_equation_l788_788369


namespace projection_onto_vector_is_expected_l788_788303

def projection_matrix (u: ℝ × ℝ) : matrix (fin 2) (fin 2) ℝ :=
  let ⟨x, y⟩ := u in 
  (1 / (x^2 + y^2)) • (matrix.col_vec u ⬝ (matrix.transpose (matrix.row_vec u)))

def expected_matrix : matrix (fin 2) (fin 2) ℝ :=
  ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]]

theorem projection_onto_vector_is_expected :
  projection_matrix (3, 4) = expected_matrix := by
  sorry

end projection_onto_vector_is_expected_l788_788303


namespace solve_equation_l788_788372

noncomputable def smallest_solution : Rat :=
  (8 - Real.sqrt 145) / 3

theorem solve_equation : 
  ∃ x : ℝ, (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ x = smallest_solution := sorry

end solve_equation_l788_788372


namespace OH_squared_l788_788913

variables {O H A B C : Type} [inner_product_space ℝ O]

def circumcenter (a b c : ℝ) : Type := -- Definition of circumcenter (e.g., type class for properties)
 sorry -- shared space with orthocenter and triangle sides

def orthocenter (a b c : ℝ) : Type := -- Definition of orthocenter (e.g., type class for properties)
 sorry -- shared space with circumcenter and triangle sides

variables (a b c R : ℝ) (triangle : circumcenter a b c) -- Defining triangle properties
variables (orthotriangle : orthocenter a b c) -- Defining orthotriangle within the triangle properties

theorem OH_squared 
  (hR : R = 5)
  (h_side_sum : a^2 + b^2 + c^2 = 50) : 
  let OH_squared := 
    (3 * R^2 + 2 * (R^2 - (a^2 + b^2 + c^2) / 2)) in
  OH_squared = 75 :=
by
  sorry

end OH_squared_l788_788913


namespace vector_bisector_l788_788930

noncomputable def a : ℝ^3 := ![8, -3, 5]
noncomputable def c : ℝ^3 := ![-3, 4, -6]
noncomputable def b : ℝ^3 := (1/3 : ℝ) • (a + c)

theorem vector_bisector 
  (h : ∀ b : ℝ^3, (b = (1/3 : ℝ) • (a + c)) → (‖ (a - b) ‖ = ‖ (c - b) ‖)) : 
  b = ![5/3, 1/3, -1/3] :=
by
  sorry

end vector_bisector_l788_788930


namespace range_of_m_l788_788823

theorem range_of_m
  (m : ℝ) 
  (α : ℂ)
  (hα_root : α ^ 2 - (2 * m - 1) * α + (m ^ 2 + 1) = 0)
  (hα_imaginary : α.im ≠ 0)
  (hα_abs : abs α ≤ 2) :
  - (3 / 4) < m ∧ m ≤ real.sqrt 3 := 
sorry

end range_of_m_l788_788823


namespace parabola_and_circle_tangency_relationship_l788_788642

-- Definitions for points and their tangency
def is_tangent (line : ℝ → ℝ) (circle_center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∃ x, (x - circle_center.1)^2 + (line x - circle_center.2)^2 = radius^2

theorem parabola_and_circle_tangency_relationship :
  (∀ x y: ℝ, y^2 = x → ∃ x, (x - 2)^2 + y^2 = 1) ∧
  (∀ (a1 a2 a3 : ℝ × ℝ),
    (a1.2) ^ 2 = a1.1 → 
    (a2.2) ^ 2 = a2.1 → 
    (a3.2) ^ 2 = a3.1 →
    (is_tangent (λ x, (a1.2 / (a1.1 - x))) (2, 0) 1) →
    (is_tangent (λ x, (a1.2 / (a1.1 - x))) (2, 0) 1) →
    is_tangent (λ x, (a2.2 / (a2.1 - x))) (2, 0) 1 ∧
    is_tangent (λ x, (a3.2 / (a3.1 - x))) (2, 0) 1)
  := 
sorry

end parabola_and_circle_tangency_relationship_l788_788642


namespace exists_trinomial_with_exponents_three_l788_788211

theorem exists_trinomial_with_exponents_three (x y : ℝ) :
  ∃ (a b c : ℝ) (t1 t2 t3 : ℕ × ℕ), 
  t1.1 + t1.2 = 3 ∧ t2.1 + t2.2 = 3 ∧ t3.1 + t3.2 = 3 ∧
  (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧
  (a * x ^ t1.1 * y ^ t1.2 + b * x ^ t2.1 * y ^ t2.2 + c * x ^ t3.1 * y ^ t3.2 ≠ 0) := sorry

end exists_trinomial_with_exponents_three_l788_788211


namespace perimeter_of_polygon_l788_788798

-- Define the dimensions of the strips and their arrangement
def strip_width : ℕ := 4
def strip_length : ℕ := 16
def num_vertical_strips : ℕ := 2
def num_horizontal_strips : ℕ := 2

-- State the problem condition and the expected perimeter
theorem perimeter_of_polygon : 
  let vertical_perimeter := num_vertical_strips * strip_length
  let horizontal_perimeter := num_horizontal_strips * strip_length
  let corner_segments_perimeter := (num_vertical_strips + num_horizontal_strips) * strip_width
  vertical_perimeter + horizontal_perimeter + corner_segments_perimeter = 80 :=
by
  sorry

end perimeter_of_polygon_l788_788798


namespace CG_eq_CD_l788_788755

theorem CG_eq_CD 
  {A B C D E F : Type}
  [circumcircle : ∀ (ABC  ∈ Type), D ≠ E]
  [midpoint_D : ∀ (arcBAC  ∈ Type) ∧ midpoint D]
  [midpoint_E : ∀ (arcBC  ∈ Type) ∧ midpoint E]
  (perp_CF_AB : CF ⊥ AB) :
  CG = CD := 
sorry

end CG_eq_CD_l788_788755


namespace det_of_cross_product_matrix_l788_788088

variables (a b c : ℝ^3)
def D := 2 * (a • (b × c))

theorem det_of_cross_product_matrix (a b c : ℝ^3) :
  let D' := Matrix.det (![![a × b, 2 * (b × c), c × a]] : Matrix 3 3 ℝ)
  in D' = 4 * D^2 := 
sorry

end det_of_cross_product_matrix_l788_788088


namespace smallest_n_l788_788885

noncomputable def a_n (n : ℕ) : ℝ :=
  let a₁ : ℝ := -1 + 2008 * d in  -- initial term from \(a_{2009}\)
  a₁ + (n - 1) * d

def S_n (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem smallest_n (d : ℝ) (h1 : d > 0)
  (h2 : ∃ a₁ a₂ : ℝ, a₁ < 0 ∧ a₂ > 0 ∧ a₁ + a₂ = 3 ∧ a₁ * a₂ = -5) :
  ∃ n : ℕ, n = 4018 ∧ S_n n a_n > 0 :=
by
  sorry

end smallest_n_l788_788885


namespace stones_partition_l788_788992

theorem stones_partition (total_stones : ℕ) (piles : ℕ) (heaps : ℕ → ℕ) 
  (h_total : total_stones = 660)
  (h_piles : piles = 30)
  (h_sum_heaps : ∑ i in range piles, heaps i = 660)
  (h_factor : ∀ i j, i < piles → j < piles → heaps i ≤ 2 * heaps j) :
  ∃ heaps : Π i : ℕ, i < piles → ℕ,
    (∑ i in range piles, heaps i = 660) ∧
    (∀ i j, i < piles → j < piles → heaps i ≤ 2 * heaps j) :=
  sorry

end stones_partition_l788_788992


namespace coplanar_tangency_points_l788_788242

-- Definitions used directly appear in the conditions.
def spatial_quadrilateral_circumscribed_around_sphere (Q : Set Point) (S : Sphere) :=
  ∀ p ∈ Q, ∃ q ∈ S, tangent_point p q

def tangency_points (Q : Set Point) (S : Sphere) (A B C D : Point) :=
  tangent_point A ⟨Q, S⟩ ∧ tangent_point B ⟨Q, S⟩ ∧ tangent_point C ⟨Q, S⟩ ∧ tangent_point D ⟨Q, S⟩

-- Mathematical equivalent proof problem.
theorem coplanar_tangency_points {Q : Set Point} {S : Sphere} {A B C D : Point} 
  (hsq : spatial_quadrilateral_circumscribed_around_sphere Q S)
  (htp : tangency_points Q S A B C D) : coplanar A B C D :=
by
  sorry

end coplanar_tangency_points_l788_788242


namespace sequence_sum_correct_l788_788811

-- Definitions and conditions of the sequence
def a (n : ℕ) : ℕ
| 0       := 0  -- This is added for function definition completeness in Lean
| 1       := 1
| (n + 1) := a n + 1 + n

-- Define the summation function
def sequence_sum (n : ℕ) : ℚ :=
  ∑ i in Finset.range n, (1 : ℚ) / a (i + 1)

-- Correct answer calculation
theorem sequence_sum_correct : sequence_sum 2017 = 2017 / 1009 := 
  sorry

end sequence_sum_correct_l788_788811


namespace no_solution_log_eq_l788_788581

theorem no_solution_log_eq (x : ℝ) : 
  ¬ ((log 2 (x - 1) = log 2 (2 * x + 1)) ∧ (x - 1 > 0) ∧ (2 * x + 1 > 0)) := 
by
  sorry

end no_solution_log_eq_l788_788581


namespace smallest_positive_integer_x_l788_788201

theorem smallest_positive_integer_x (x : ℕ) (h900 : ∃ a b c : ℕ, 900 = (2^a) * (3^b) * (5^c) ∧ a = 2 ∧ b = 2 ∧ c = 2) (h1152 : ∃ a b : ℕ, 1152 = (2^a) * (3^b) ∧ a = 7 ∧ b = 2) : x = 32 :=
by
  sorry

end smallest_positive_integer_x_l788_788201


namespace log_inequality_solution_l788_788385

theorem log_inequality_solution {a : ℝ} (h : log a (3 / 5) < 1) : a ∈ (Set.Ioo 0 (3 / 5) ∪ Set.Ioi 1) :=
sorry

end log_inequality_solution_l788_788385


namespace largest_non_representable_correct_largest_non_representable_not_provable_l788_788888

noncomputable def largest_non_representable (n : ℕ) : ℕ :=
  3^(n + 1) - 2^(n + 2)

theorem largest_non_representable_correct (n : ℕ) : 
  ∀ (s : ℕ), (s > 3^(n + 1) - 2^(n+2)) -> (∃ a b : ℕ, s = 2^n * a + b * 2^(n-1) * 3 ∨
  s = 2^(n-2) * (3^2 * b) ∨ s = 2^(n-3) * 3^3 ∨ ... ∨ s = 2 * 3^(n-1) ∨ s = 3^n) :=
    sorry

theorem largest_non_representable_not_provable (n : ℕ) :
  ¬ ∃ (s ≥ 0), s = 3^(n + 1) - 2^(n + 2) :=
    sorry

end largest_non_representable_correct_largest_non_representable_not_provable_l788_788888


namespace extra_people_needed_l788_788284

theorem extra_people_needed 
  (initial_people : ℕ) 
  (initial_time : ℕ) 
  (final_time : ℕ) 
  (work_done : ℕ) 
  (all_paint_same_rate : initial_people * initial_time = work_done) :
  initial_people = 8 →
  initial_time = 3 →
  final_time = 2 →
  work_done = 24 →
  ∃ extra_people : ℕ, extra_people = 4 :=
by
  sorry

end extra_people_needed_l788_788284


namespace cos_two_alpha_find_beta_l788_788037

open Real

-- Definitions for the given conditions
variables (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)

-- Definitions for vectors and orthogonality
def m : ℝ × ℝ := (cos α, -1)
def n : ℝ × ℝ := (2, sin α)
def orthogonal (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 = 0

-- Additional condition for part 2
def sin_diff_condition := sin (α - β) = sqrt 10 / 10

-- Statements to be proved
theorem cos_two_alpha (h_orth : orthogonal m n) : cos (2 * α) = -3 / 5 :=
sorry

theorem find_beta (h_orth : orthogonal m n) (h_sin_diff : sin_diff_condition α β) : β = π / 4 :=
sorry

end cos_two_alpha_find_beta_l788_788037


namespace hexahedron_volume_l788_788192

-- Definitions of the fundamental parameters.
variable (a : ℝ)

-- Conditions based on the problem statement.
def isosceles_trapezoid_waist_length : ℝ := a
def top_base_length : ℝ := a
def bottom_base_length : ℝ := 2 * a
def rectangle_length : ℝ := 2 * a
def rectangle_width : ℝ := a

-- The proposition to be proven, stating the volume of the hexahedron.
theorem hexahedron_volume :
  (volume_of_hexahedron isosceles_trapezoid_waist_length top_base_length bottom_base_length rectangle_length rectangle_width) = (13 / 12) * sqrt 2 * a^3 := 
sorry

end hexahedron_volume_l788_788192


namespace number_of_primes_with_digits_1234_l788_788455

def is_permutation_of_digits_1234 (n : Nat) : Prop :=
  let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10]
  digits ~ (multiset.mk [1, 2, 3, 4])

def ends_in_1_or_3 (n : Nat) : Prop :=
  (n % 10 = 1) ∨ (n % 10 = 3)

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

noncomputable def four_digit_primes_with_digits_1234_ending_in_1_or_3 : Nat :=
  (List.filter (λ n, is_prime n ∧ is_permutation_of_digits_1234 n ∧ ends_in_1_or_3 n) [1234, 1243, 1324,
    1342, 1423, 1432, 2134, 2143, 2314, 2341, 2413, 2431, 3124, 3142, 3214, 3241, 3412, 3421, 
    4123, 4132, 4213, 4231, 4312, 4321]).length

theorem number_of_primes_with_digits_1234 : four_digit_primes_with_digits_1234_ending_in_1_or_3 = 4 :=
sorry

end number_of_primes_with_digits_1234_l788_788455


namespace rotate_triangle_forms_two_cones_l788_788141

theorem rotate_triangle_forms_two_cones (T : Triangle) (h1 : T.equilateral) :
  rotate_around_base T = two_cones :=
sorry

end rotate_triangle_forms_two_cones_l788_788141


namespace probability_of_perfect_square_l788_788738

theorem probability_of_perfect_square (p : ℝ) (h₁ : ∀ n, 1 ≤ n ∧ n ≤ 120 → (n ≤ 60 → prob n = p) ∧ (n > 60 → prob n = 2p)) 
  (h₂ : 60 * p + 60 * 2 * p = 1) : (∑ n in finset.filter (λ n, ∃ k, n = k * k) (finset.range 121), prob n) = 13 / 180 := 
by {
  sorry
}

noncomputable def prob : ℕ → ℝ

end probability_of_perfect_square_l788_788738


namespace maxN_l788_788275

noncomputable def max_columns {α : Type*} [Fintype α] (rows : ℕ) (cols : ℕ) (arrangement : Array (Array α)) : Prop :=
(arrangement.size = rows) ∧
(∀ i, (arrangement[i]).size = cols) ∧
(∀ j, ∃ i₁ i₂, i₁ ≠ i₂ ∧ arrangement[i₁][j] = arrangement[i₂][j]) ∧
(∀ j₁ j₂, j₁ ≠ j₂ → ∃ i, arrangement[i][j₁] ≠ arrangement[i][j₂])

theorem maxN: ∃ (T : Array (Array (Fin 6))) (N : ℕ), N = 120 ∧ max_columns 6 N T :=
sorry

end maxN_l788_788275


namespace ratio_odd_even_divisors_l788_788527

def sum_of_divisors (n : ℕ) : ℕ := sorry -- This should be implemented as a function that calculates sum of divisors

def sum_of_odd_divisors (n : ℕ) : ℕ := sorry -- This should be implemented as a function that calculates sum of odd divisors

def sum_of_even_divisors (n : ℕ) : ℕ := sorry -- This should be implemented as a function that calculates sum of even divisors

theorem ratio_odd_even_divisors (M : ℕ) (h : M = 36 * 36 * 98 * 210) :
  sum_of_odd_divisors M / sum_of_even_divisors M = 1 / 60 :=
by {
  sorry
}

end ratio_odd_even_divisors_l788_788527


namespace books_sold_l788_788552

theorem books_sold (original_books : ℕ) (remaining_books : ℕ) (sold_books : ℕ) 
  (h1 : original_books = 51) 
  (h2 : remaining_books = 6) 
  (h3 : sold_books = original_books - remaining_books) : 
  sold_books = 45 :=
by 
  sorry

end books_sold_l788_788552


namespace odd_function_increasing_on_Icc_find_range_m_l788_788387

noncomputable def odd_function_increasing (f : ℝ → ℝ) := 
  ∀ x : ℝ, f (-x) = -f x

theorem odd_function_increasing_on_Icc (f : ℝ → ℝ)
  (h_odd : odd_function_increasing f) 
  (h1 : f 1 = 1) 
  (h_pos : ∀ a b : ℝ, a ∈ Icc (-1) 1 → b ∈ Icc (-1) 1 → a + b ≠ 0 → (f a + f b) / (a + b) > 0) :
  ∀ x1 x2 : ℝ, x1 ∈ Icc (-1) 1 → x2 ∈ Icc (-1) 1 → x1 < x2 → f x1 < f x2 := 
sorry

theorem find_range_m (f : ℝ → ℝ)
  (h_odd : odd_function_increasing f)
  (h1 : f 1 = 1)
  (h_pos : ∀ a b : ℝ, a ∈ Icc (-1) 1 → b ∈ Icc (-1) 1 → a + b ≠ 0 → (f a + f b) / (a + b) > 0)
  (hx : ∀ a : ℝ, a ∈ Icc (-1) 1 → f x ≥ m^2 - 2 * a * m - 2) :
  m ∈ Icc (-1) 1 := 
sorry

end odd_function_increasing_on_Icc_find_range_m_l788_788387


namespace g_symmetry_l788_788021

-- Define the original function f and its properties
def f (x : ℝ) : ℝ := Real.sin (1 / 2 * x + Real.pi / 6)

-- Define the transformed function g
def g (x : ℝ) : ℝ := f (x - Real.pi / 3)

-- State the theorem to prove the symmetry of g about the origin
theorem g_symmetry : ∀ x : ℝ, g(-x) = -g(x) :=
by
  -- Proof skipped
  sorry

end g_symmetry_l788_788021


namespace projection_onto_3_4_matrix_l788_788338

def projection_matrix := λ (u : ℝ) (v : ℝ), (3 * u + 4 * v) / 25

theorem projection_onto_3_4_matrix :
  ∀ (x y : ℝ),
  (λ (u v : ℝ), (3 * x + 4 * y) / 25) = (λ (u v : ℝ), (\(u * 9 / 25) + (v * 12 / 25), (u * 12 / 25) + (v * 16 / 25))) :=
by
  sorry

end projection_onto_3_4_matrix_l788_788338


namespace sampling_probabilities_equal_l788_788396

def population_size : ℕ := 50
def sample_size : ℕ := 10
def p1 : ℚ := sample_size / population_size
def p2 : ℚ := sample_size / population_size
def p3 : ℚ := sample_size / population_size

theorem sampling_probabilities_equal :
  p1 = p2 ∧ p2 = p3 :=
by
  -- Rely on calculations shown in the solution steps
  unfold p1 p2 p3
  -- Automatic simplification
  simp
  -- The equal probabilities were derived by assuming assumptions of equal chance.
  -- So we can conclude that p1 = p2 and p2 = p3
  sorry

end sampling_probabilities_equal_l788_788396


namespace woman_away_time_l788_788746

noncomputable def angle_hour_hand (n : ℝ) : ℝ := 150 + n / 2
noncomputable def angle_minute_hand (n : ℝ) : ℝ := 6 * n

theorem woman_away_time : 
  (∀ n : ℝ, abs (angle_hour_hand n - angle_minute_hand n) = 120) → 
  abs ((540 / 11 : ℝ) - (60 / 11 : ℝ)) = 43.636 :=
by sorry

end woman_away_time_l788_788746


namespace find_y_l788_788042

variables (x y : ℝ)

theorem find_y (h1 : x = 103) (h2 : x^3 * y - 4 * x^2 * y + 4 * x * y = 515400) : y = 1 / 2 :=
sorry

end find_y_l788_788042


namespace flyers_left_l788_788506

theorem flyers_left (total_flyers : ℕ) (jack_flyers : ℕ) (rose_flyers : ℕ) (h1 : total_flyers = 1236) (h2 : jack_flyers = 120) (h3 : rose_flyers = 320) : (total_flyers - (jack_flyers + rose_flyers) = 796) := 
by
  sorry

end flyers_left_l788_788506


namespace problem_I_problem_II_problem_III_l788_788110

namespace RelatedNumber

def is_related_number (A : Set ℕ) (n m : ℕ) : Prop :=
  ∀ P ⊆ A, card P = m →
  ∃ (a b c d : ℕ), {a, b, c, d} ⊆ P ∧ a + b + c + d = 4 * n + 1

def A_2n (n : ℕ) : Set ℕ := {k | 0 < k ∧ k ≤ 2 * n}

theorem problem_I (n : ℕ) :
  (n = 3) → (¬ is_related_number (A_2n n) n 5) ∧ is_related_number (A_2n n) n 6 :=
  sorry

theorem problem_II (n m : ℕ) :
  is_related_number (A_2n n) n m → m - n - 3 ≥ 0 :=
  sorry

theorem problem_III (n : ℕ) :
  ∃ m, is_related_number (A_2n n) n m ∧ ∀ m', is_related_number (A_2n n) n m' → m ≤ m' :=
  sorry

end RelatedNumber

end problem_I_problem_II_problem_III_l788_788110


namespace cos_beta_cos_2alpha_plus_beta_l788_788000

variables {α β : ℝ}

-- Conditions given in the problem statement.
axiom alpha_pos : 0 < α
axiom alpha_lt_pi_div_2 : α < π / 2
axiom beta_gt_pi_div_2 : π / 2 < β
axiom beta_lt_pi : β < π
axiom cos_eq_one_third : cos (α + π / 4) = 1 / 3
axiom cos_eq_sqrt3_div3 : cos (π / 4 - β / 2) = sqrt 3 / 3

-- Proof statement for question 1.
theorem cos_beta : cos β = -4 * sqrt 2 / 9 := sorry

-- Proof statement for question 2.
theorem cos_2alpha_plus_beta : cos (2 * α + β) = -1 := sorry

end cos_beta_cos_2alpha_plus_beta_l788_788000


namespace smallest_third_altitude_l788_788496

theorem smallest_third_altitude (h₁ h₂ : ℕ) (h₁eq : h₁ = 6) (h₂eq : h₂ = 18) 
  (h3int : ∃ (h₃ : ℕ), True) : ∃ (h₃ : ℕ), h₃ = 9 := 
by
  use 9
  sorry

end smallest_third_altitude_l788_788496


namespace john_ultramarathon_distance_l788_788077

theorem john_ultramarathon_distance :
  let initial_time := 8
  let time_increase_percentage := 0.75
  let speed_increase := 4
  let initial_speed := 8
  initial_time * (1 + time_increase_percentage) * (initial_speed + speed_increase) = 168 :=
by
  let initial_time := 8
  let time_increase_percentage := 0.75
  let speed_increase := 4
  let initial_speed := 8
  sorry

end john_ultramarathon_distance_l788_788077


namespace smallest_solution_is_39_over_8_l788_788364

noncomputable def smallest_solution (x : ℝ) : Prop :=
  (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ (x ≠ 0) ∧ (x ≠ 3)

theorem smallest_solution_is_39_over_8 : ∃ x > 0, smallest_solution x ∧ x = 39 / 8 :=
by
  sorry

end smallest_solution_is_39_over_8_l788_788364


namespace cost_price_of_computer_table_l788_788695

theorem cost_price_of_computer_table (sp : ℝ) (cp : ℝ) (markup : ℝ) (h1 : markup = 0.32) (h2 : sp = cp * (1 + markup)) (h3 : sp = 5400) :
  cp ≈ 4090.91 :=
by
  sorry

end cost_price_of_computer_table_l788_788695


namespace farmer_seeds_l788_788757

theorem farmer_seeds (h1 : 2 * 22.34 = 44.68) (h2 : 22.34 ≈ 134.04 / (6 : ℝ)) : 
  2 * 134.04 = (6 : ℝ) * 44.68 :=
by
  sorry

end farmer_seeds_l788_788757


namespace flyers_left_l788_788501

theorem flyers_left (initial_flyers : ℕ) (jack_flyers : ℕ) (rose_flyers : ℕ) (left_flyers : ℕ) :
  initial_flyers = 1236 →
  jack_flyers = 120 →
  rose_flyers = 320 →
  left_flyers = 796 →
  initial_flyers - (jack_flyers + rose_flyers) = left_flyers := 
by
  intros h_initial h_jack h_rose h_left
  rw [h_initial, h_jack, h_rose, h_left]
  simp
  sorry

end flyers_left_l788_788501


namespace largest_sphere_surface_area_in_cone_l788_788392

theorem largest_sphere_surface_area_in_cone :
  (∀ (r : ℝ), (∃ (r : ℝ), r > 0 ∧ (1^2 + (3^2 - r^2) = 3^2)) →
    4 * π * r^2 ≤ 2 * π) :=
by
  sorry

end largest_sphere_surface_area_in_cone_l788_788392


namespace magnitude_of_vector_addition_value_of_k_for_parallel_vectors_range_of_k_for_acute_angle_l788_788449

noncomputable def a : ℝ × ℝ := (-1, 3)
noncomputable def b : ℝ × ℝ := (1, -2)

-- Proof problem 1: The magnitude of vector addition
theorem magnitude_of_vector_addition :
  |a.fst + 2 * b.fst, a.snd + 2 * b.snd| = Real.sqrt 2 := by
  sorry

-- Proof problem 2: Finding the value of k given parallel vectors
theorem value_of_k_for_parallel_vectors (k : ℝ) :
  ((a.fst - b.fst, a.snd - b.snd) ∥ (a.fst + k * b.fst, a.snd + k * b.snd)) → k = -1 := by
  sorry

-- Proof problem 3: Range of k for acute angle
theorem range_of_k_for_acute_angle (k : ℝ) :
  dot (a.fst - b.fst, a.snd - b.snd) (a.fst + k * b.fst, a.snd + k * b.snd) > 0  →
  k ∈ Iio (-1) ∪ Ioc (-1, 17 / 12) := by
  sorry

end magnitude_of_vector_addition_value_of_k_for_parallel_vectors_range_of_k_for_acute_angle_l788_788449


namespace product_in_third_quadrant_l788_788813

-- Define the complex numbers z1 and z2
def z1 : ℂ := 1 - I
def z2 : ℂ := 3 - 4 * I

-- Prove that their product is in the third quadrant
theorem product_in_third_quadrant : 
  let product := z1 * z2
  in (product.re < 0 ∧ product.im < 0) := 
by
  sorry

end product_in_third_quadrant_l788_788813


namespace find_m_integer_l788_788173

noncomputable def a_sequence (n : ℕ) : ℝ :=
  if n = 0 then 0
  else if n = 1 then Real.pi / 6
  else Real.arctan (Real.sec (a_sequence (n - 1)))

theorem find_m_integer : ∃ m : ℕ, (∀ n, a_sequence n = a_sequence n ∧ 
  ∀ k, (∏ i in Finset.range k, Real.sin (a_sequence (i + 1))) = 1 / 100) → (m = 3333) :=
begin
  sorry
end

end find_m_integer_l788_788173


namespace smallest_possible_area_of_right_triangle_l788_788674

-- Definitions of given conditions
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

def area_of_right_triangle (a b : ℕ) : ℝ :=
  (a.to_real * b.to_real) / 2

-- Proof statement
theorem smallest_possible_area_of_right_triangle (a b : ℕ) (h1 : a = 5) (h2 : b = 6) :
  let A₁ := area_of_right_triangle a b in
  let A₂ := area_of_right_triangle b 5 in -- When assuming 6 is the hypotenuse
  A₂ < A₁ ∧ A₂ ≈ 8.29 :=
by
  sorry

end smallest_possible_area_of_right_triangle_l788_788674


namespace projection_matrix_3_4_l788_788327

theorem projection_matrix_3_4 :
  let v := λ α : Type, @vector α 2 := ![3, 4]
  let proj := λ x : vector ℝ 2, (v ℝ ⬝ x) / (v ℝ ⬝ v ℝ) • v ℝ
  proj = (λ x : vector ℝ 2, matrix.mul_vec ![
     ![9 / 25, 12 / 25],
     ![12 / 25, 16 / 25]
  ] x) :=
by sorry

end projection_matrix_3_4_l788_788327


namespace minimum_figures_of_type1_needed_l788_788256

-- Definitions of figures types (Type 1, Type 2, Type 3, Type 4)
-- Each figure is composed of 4 equilateral unit triangles 
def figure_type1 := sorry -- Placeholder for the actual figure type definition
def figure_type2 := sorry
def figure_type3 := sorry
def figure_type4 := sorry

-- Condition: Triangle T with side length 2022 divided into unit triangles
def triangle_T (n : ℕ) := sorry -- Placeholder for the equilateral triangle definition

-- Condition a): The grid is covered with figures which can be rotated
def covering_condition (T : Type) (figures : List (figure_type1 ∨ figure_type2 ∨ figure_type3 ∨ figure_type4)) := 
    sorry -- Placeholder for the condition that ensures full coverage without overlap

-- To be proved: the minimum number of figures of type 1 needed
theorem minimum_figures_of_type1_needed : 
    ∃ (figures : List (figure_type1 ∨ figure_type2 ∨ figure_type3 ∨ figure_type4)),
    covering_condition (triangle_T 2022) figures ∧
    (figures.filter (λ f, f.is_type1)).length = 1011 :=
sorry -- Proof is omitted

end minimum_figures_of_type1_needed_l788_788256


namespace number_of_paths_from_a_to_b_l788_788737

noncomputable def count_paths (start end : ℕ × ℕ) (blocked_cells : set (ℕ × ℕ)) (grid_size : ℕ × ℕ) : ℕ :=
sorry

theorem number_of_paths_from_a_to_b (a b : ℕ × ℕ) (blocked_cells : set (ℕ × ℕ)) (grid_size: ℕ × ℕ) :
  count_paths a b blocked_cells grid_size = 16 :=
sorry

end number_of_paths_from_a_to_b_l788_788737


namespace simplify_and_evaluate_expression_l788_788145

theorem simplify_and_evaluate_expression (x y : ℝ)
  (h : (x - 1)^2 + |y + 2| = 0) :
  (3 / 2 * x^2 * y - (x^2 * y - 3 * (2 * x * y - x^2 * y) - x * y)) = -9 :=
by {
  -- To be filled in with the actual proof steps.
  sorry,
}

end simplify_and_evaluate_expression_l788_788145


namespace sum_of_integers_85_to_100_l788_788761

theorem sum_of_integers_85_to_100 : ∑ k in finset.range (100 - 85 + 1) + 85 = 1480 :=
by
  sorry

end sum_of_integers_85_to_100_l788_788761


namespace det_rotation_75_degrees_l788_788530

def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]

theorem det_rotation_75_degrees :
  Matrix.det (rotation_matrix (Real.pi / 180 * 75)) = 1 :=
by
  sorry

end det_rotation_75_degrees_l788_788530


namespace f_of_5_eq_1_l788_788018

def f : ℝ → ℝ
| x := if x < 2 then abs (x^2 - 2) else f (x - 2)

theorem f_of_5_eq_1 : f 5 = 1 :=
by
  sorry

end f_of_5_eq_1_l788_788018


namespace min_Sn_l788_788398

noncomputable def a1_d (a1 d : ℤ) : Prop := 
  (a1 + 2 * d) ^ 2 = (a1 + d) * (a1 + 5 * d) ∧
  a1 + 9 * d = -17 ∧
  d ≠ 0

def Sn (n a1 d : ℤ) := n * (a1 + (a1 + (n - 1) * d)) / 2

noncomputable def Sn_min_value_helper : ℕ → ℤ
| 0     := 0
| (n + 1) := Sn_min_value_helper n + (a1 + (a1 + n * d))

noncomputable def Sn_min (a1 d : ℤ) : ℕ → ℚ
| 0     := 0
| (n + 1) := (Sn_min_value_helper (n + 1) : ℚ) / 2^(n + 1)

theorem min_Sn {a1 d : ℤ} (h : a1_d a1 d) : Sn_min a1 d 4 = -1/2 :=
sorry

end min_Sn_l788_788398


namespace sequence_general_formula_and_sum_l788_788446

theorem sequence_general_formula_and_sum (a b : ℕ → ℕ) :
  (∃ (S : ℕ → ℕ), a 1 = 1 ∧ b 1 = 1 ∧ a 2 = 3 ∧
    (∀ n : ℕ, 2 ≤ n → S (n + 1) + S (n - 1) = 2 * (S n + 1)) ∧
    (∀ n : ℕ, b 1 + 2 * b 2 + 2^2 * b 3 + ∃ S = n (\sum i in finset.range (n - 2), 2^i * b (i + 1)) + 2^(n-1) * b n = S n)) →
  (∀ n : ℕ, a n = 2 n - 1 ∧
    b n = if n = 1 then 1 else 2^(2 - n) ∧
    (∃ T : ℕ → ℕ, T 1 = 1 ∧
      (∀ n : ℕ, 2 ≤ n → T (n + 1) = T n + a (n + 1) * b (n + 1)) →
      T n = 11 - (2 * n + 3) * 2^(2 - n))) :=
begin
  sorry
end

end sequence_general_formula_and_sum_l788_788446


namespace min_distance_circle_to_q_l788_788734

-- Defining the mathematical entities and proof goal
theorem min_distance_circle_to_q :
  let circle_center : ℝ × ℝ := (0, 2)
  let radius : ℝ := 2
  let circle (x y : ℝ) := x^2 + (y - 2)^2 = radius^2
  let Q (k : ℝ) : ℝ × ℝ := (k / 2, k - 3)
  let distance (p1 p2 : ℝ × ℝ) := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  ∀ p ∈ {p : ℝ × ℝ | circle p.1 p.2},
    ∃ k, distance p (Q k) >= sqrt 5 - 2 :=
sorry

end min_distance_circle_to_q_l788_788734


namespace smallest_sum_of_two_3_digit_numbers_l788_788676

theorem smallest_sum_of_two_3_digit_numbers : 
  ∃ (a b c d e f : ℕ), {a, b, c, d, e, f} = {1, 2, 3, 7, 8, 9} ∧ 
  100 * a + 10 * b + c < 1000 ∧ 100 * d + 10 * e + f < 1000 ∧ 
  ∀ (a' b' c' d' e' f' : ℕ), {a', b', c', d', e', f'} = {1, 2, 3, 7, 8, 9} ∧ 
  100 * a' + 10 * b' + c' < 1000 ∧ 100 * d' + 10 * e' + f' < 1000 → 
  100 * a + 10 * b + c + 100 * d + 10 * e + f ≤ 100 * a' + 10 * b' + c' + 100 * d' + 10 * e' + f' ∧
  100 * a + 10 * b + c + 100 * d + 10 * e + f = 417 := sorry

end smallest_sum_of_two_3_digit_numbers_l788_788676


namespace sequence_general_term_l788_788599

theorem sequence_general_term (n : ℕ) : 
  let a_n := (2 * n - 1) / (2 * n) in 
  ∃ a, a = a_n :=
by
  sorry

end sequence_general_term_l788_788599


namespace median_of_set_with_mean_90_l788_788609

theorem median_of_set_with_mean_90 : 
  ∃ x : ℝ, (List.sum [91, 89, 88, 90, 87, x]) / 6 = 90 ∧ 
           (let sorted_list := List.sort (compare) [91, 89, 88, 90, 87, x]
            in (sorted_list.nth 2 + sorted_list.nth 3) / 2) = 89.5 :=
by
  sorry

end median_of_set_with_mean_90_l788_788609


namespace no_such_function_l788_788570

theorem no_such_function :
  ¬ ∃ f : ℝ → ℝ, (∀ y x : ℝ, 0 < x → x < y → f y > (y - x) * (f x)^2) :=
by
  sorry

end no_such_function_l788_788570


namespace minimum_value_l788_788421

theorem minimum_value (a b : ℝ) (h1 : ∀ x : ℝ, ax^2 + 2 * x + b > 0 ↔ x ≠ -1 / a) (h2 : a > b) : 
  ∃ a b, (ab = 1) → ∀ (a b), a > b → ∃ m, ((m = a - b + 9 / (a - b)) → m >= 6) :=
sorry

end minimum_value_l788_788421


namespace unpaintedRegionArea_l788_788187

def boardWidth1 : ℝ := 5
def boardWidth2 : ℝ := 7
def angle : ℝ := 45

theorem unpaintedRegionArea
  (bw1 bw2 angle : ℝ)
  (h1 : bw1 = boardWidth1)
  (h2 : bw2 = boardWidth2)
  (h3 : angle = 45) :
  let base := bw2 * Real.sqrt 2
  let height := bw1
  let area := base * height
  area = 35 * Real.sqrt 2 :=
by
  sorry

end unpaintedRegionArea_l788_788187


namespace min_possible_value_l788_788544

theorem min_possible_value
  (a b c d e f g h : Int)
  (h_distinct : List.Nodup [a, b, c, d, e, f, g, h])
  (h_set_a : a ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_b : b ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_c : c ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_d : d ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_e : e ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_f : f ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_g : g ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_h : h ∈ [-9, -6, -3, 0, 1, 3, 6, 10]) :
  ∃ a b c d e f g h : Int,
  ((a + b + c + d)^2 + (e + f + g + h)^2) = 2
  :=
  sorry

end min_possible_value_l788_788544


namespace plumber_spent_correct_amount_l788_788239

-- Define the given conditions
def copper_meters : ℕ := 10
def plastic_meters : ℕ := 15
def copper_cost_per_meter : ℕ := 5
def plastic_cost_per_meter : ℕ := 3
def discount_rate : ℚ := 0.10

-- Define the hypothesis
theorem plumber_spent_correct_amount :
  let total_cost_before_discount : ℚ := copper_meters * copper_cost_per_meter + plastic_meters * plastic_cost_per_meter,
      discount_amount : ℚ := discount_rate * total_cost_before_discount,
      total_cost_after_discount : ℚ := total_cost_before_discount - discount_amount
  in total_cost_after_discount = 85.50 := by
  sorry

end plumber_spent_correct_amount_l788_788239


namespace John_distance_proof_l788_788082

def initial_running_time : ℝ := 8
def increase_percentage : ℝ := 0.75
def initial_speed : ℝ := 8
def speed_increase : ℝ := 4

theorem John_distance_proof : 
  (initial_running_time + initial_running_time * increase_percentage) * (initial_speed + speed_increase) = 168 := 
by
  -- Proof can be completed here
  sorry

end John_distance_proof_l788_788082


namespace calculate_f_g2_l788_788033

def f (x : ℝ) : ℝ := x^2 + 1
def g (x : ℝ) : ℝ := 2 * x^3 - 1

theorem calculate_f_g2 : f (g 2) = 226 := by
  sorry

end calculate_f_g2_l788_788033


namespace number_of_solutions_in_range_l788_788774

def cubicEquationSolutionsInsideRange : ℝ → ℝ := 
  fun x => 3 * (Real.sin x)^4 - 7 * (Real.sin x)^3 + 4 * (Real.sin x)^2 - Real.sin x

theorem number_of_solutions_in_range : 
  (finset.card ((finset.filter (fun x => cubicEquationSolutionsInsideRange x = 0) (finset.Icc 0 (2 * Real.pi)))).val) = 3 := 
sorry

end number_of_solutions_in_range_l788_788774


namespace evaluate_expression_l788_788380

theorem evaluate_expression (x : ℝ) :
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 12 * x + 2 :=
by
  sorry

end evaluate_expression_l788_788380


namespace coefficient_x2_in_binomial_expansion_l788_788066

theorem coefficient_x2_in_binomial_expansion : 
  let term := Binomial.coeff 5 3 * (2^2 * (-1)^3) 
  in term = -40 :=
by 
  -- Define (2x - 1)^5 expansion terms and their evaluation.
  let term := Binomial.coeff 5 3 * (2^2 * (-1)^3)
  sorry

end coefficient_x2_in_binomial_expansion_l788_788066


namespace flyers_left_l788_788515

theorem flyers_left (total_flyers : ℕ) (jack_flyers : ℕ) (rose_flyers : ℕ) :
  total_flyers = 1236 → jack_flyers = 120 → rose_flyers = 320 → total_flyers - (jack_flyers + rose_flyers) = 796 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact eq.refl _

end flyers_left_l788_788515


namespace temp_difference_l788_788124

theorem temp_difference
  (temp_beijing : ℤ) 
  (temp_hangzhou : ℤ) 
  (h_beijing : temp_beijing = -10) 
  (h_hangzhou : temp_hangzhou = -1) : 
  temp_beijing - temp_hangzhou = -9 := 
by 
  rw [h_beijing, h_hangzhou] 
  sorry

end temp_difference_l788_788124


namespace expand_product_l788_788291

theorem expand_product (x : ℤ) : 
  (3 * x + 4) * (2 * x - 6) = 6 * x^2 - 10 * x - 24 :=
by
  sorry

end expand_product_l788_788291


namespace log_mul_l788_788413

theorem log_mul (a M N : ℝ) (ha_pos : 0 < a) (hM_pos : 0 < M) (hN_pos : 0 < N) (ha_ne_one : a ≠ 1) :
    Real.log (M * N) / Real.log a = Real.log M / Real.log a + Real.log N / Real.log a := by
  sorry

end log_mul_l788_788413


namespace divide_stones_l788_788963

/-- A pile of 660 stones can be divided into 30 piles where the sizes of the piles differ by less than a factor of 2. -/
theorem divide_stones (n : ℕ) (p : ℕ) (stones : ℕ) :
  stones = 660 → p = 30 →
  ∃ (heaps : Fin p → ℕ),
    (∑ i, heaps i = stones) ∧ (∀ i j, heaps i ≤ 2 * heaps j ∧ heaps j ≤ 2 * heaps i) :=
by
  intros h1 h2
  sorry

end divide_stones_l788_788963


namespace quadratic_function_passing_through_origin_l788_788847

-- Define the quadratic function y
def quadratic_function (m x : ℝ) : ℝ :=
  (m - 2) * x^2 - 4 * x + m^2 + 2 * m - 8

-- State the problem as a theorem
theorem quadratic_function_passing_through_origin (m : ℝ) (h: quadratic_function m 0 = 0) : m = -4 :=
by
  -- Since we only need the statement, we put sorry here
  sorry

end quadratic_function_passing_through_origin_l788_788847


namespace distances_equal_l788_788106

-- Define an acute-angled triangle
def AcuteAngledTriangle (A B C : Type) := ∀ (a b c : A), True

variables {A B C E F G H : Type} 

-- Define the base points of the heights through B and C
def BasePointsOfHeights (A B C E F : Type) [AcuteAngledTriangle A B C] := 
  ∃ (e : E) (f : F), True

-- Define the projections of B and C onto the line EF
def Projections (B C E F G H : Type) :=
  ∃ (g : G) (h : H), True

-- State the final theorem to prove that |HE| = |FG|
theorem distances_equal 
  (A B C E F G H : Type) 
  [AcuteAngledTriangle A B C] 
  [BasePointsOfHeights A B C E F] 
  [Projections B C E F G H] :
  True :=
  -- Start the proof (insert the actual proof here)
  sorry

end distances_equal_l788_788106


namespace sum_of_numbers_l788_788051

theorem sum_of_numbers (a b c : ℝ) (h1 : 2 * a + b = 46) (h2 : b + 2 * c = 53) (h3 : 2 * c + a = 29) :
  a + b + c = 48.8333 :=
by
  sorry

end sum_of_numbers_l788_788051


namespace pile_division_660_stones_l788_788973

theorem pile_division_660_stones (n : ℕ) (heaps : List ℕ) :
  n = 660 ∧ heaps.length = 30 ∧ ∀ x ∈ heaps, ∀ y ∈ heaps, (x ≤ 2 * y ∧ y ≤ 2 * x) →
  ∃ heaps : List ℕ, n = 660 ∧ heaps.length = 30 ∧ (∀ x y ∈ heaps, x ≤ 2 * y ∧ y ≤ 2 * x) :=
by
  sorry

end pile_division_660_stones_l788_788973


namespace proof_problem_l788_788427

-- Define the function f
def f (x : ℝ) : ℝ := |x| + |x - (1 / 2)|

-- Define the set A which is the solution set of the inequality f(x) < x + 1/2
def A : set ℝ := {x | f(x) < x + (1 / 2)}

-- The proof problem in Lean 4 statement
theorem proof_problem : (A = {x | 0 < x ∧ x < 1}) ∧ ∀ a ∈ A, |Real.log2 (1 - a)| > |Real.log2 (1 + a)| :=
by
  -- Defer the proof using sorry
  sorry

end proof_problem_l788_788427


namespace geometric_progression_first_term_l788_788615

theorem geometric_progression_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 8) 
  (h2 : a + a * r = 5) : 
  a = 2 * (4 - Real.sqrt 6) ∨ a = 2 * (4 + Real.sqrt 6) := 
by sorry

end geometric_progression_first_term_l788_788615


namespace no_of_knight_placements_l788_788466

-- Define a 5x5 chess board
def board : Type := fin 5 × fin 5

-- Define knight move function
def knight_moves (pos : board) : set board :=
  {p | (abs (p.1.val - pos.1.val) = 2 ∧ abs (p.2.val - pos.2.val) = 1) ∨ (abs (p.1.val - pos.1.val) = 1 ∧ abs (p.2.val - pos.2.val) = 2)}

-- Condition for knights not threatening each other
def non_threatening (knights : list board) : Prop :=
  ∀ k1 ∈ knights, ∀ k2 ∈ knights, k1 ≠ k2 → k2 ∉ knight_moves k1

-- We must place 5 knights on the board
def correct_knight_placement (knights : list board) : Prop :=
  knights.length = 5 ∧ non_threatening knights

-- The problem statement
theorem no_of_knight_placements : ∃ k : list board, correct_knight_placement k ∧ (list.length (filter correct_knight_placement (list.permutations (finset.to_list (finset.univ : finset board)))) = 8) := sorry

end no_of_knight_placements_l788_788466


namespace parabola_circle_properties_l788_788636

section ParabolaCircleTangent

variables {A1 A2 A3 P Q M : Point} 
variables {parabola : Parabola} 
variables {circle : Circle} 
variables {line_l : Line}

-- Definitions of points
def O := Point.mk 0 0
def M := Point.mk 2 0
def P := Point.mk 1 (Real.sqrt (2 * (1 / 2)))
def Q := Point.mk 1 (-Real.sqrt (2 * (1 / 2)))

-- Definition of geometrical constructs
def parabola := {p : Point // p.y^2 = p.x}
def circle := {c : Point // (c.x - 2)^2 + c.y^2 = 1}
def line_l := {l : Line // l.slope = ⊤ ∧ l.x_intercept = 1 }

-- Tangent properties for lines A1A2 and A1A3
def is_tangent {A B : Point} (l : Line) (circle : Circle) : Prop :=
  ∃ r: Real, (∥circle.center - A∥ = r) ∧ (∥circle.center - B∥ = r) ∧ (∥circle.center - (line.foot circle.center)∥ = r)

-- Theorem/Statement to prove:
theorem parabola_circle_properties :
  (parabola = {p : Point // p.y^2 = p.x}) →
  (circle = {c : Point // (c.x - 2)^2 + c.y^2 = 1}) →
  (∀ A1 A2 A3 : Point, A1 ∈ parabola → A2 ∈ parabola → A3 ∈ parabola → 
    (is_tangent (line_through A1 A2) circle) → (is_tangent (line_through A1 A3) circle) → 
    ⊥ ≤ distance_from_point_to_line (line_through A2 A3) circle.center = 1 ) :=
sorry

end ParabolaCircleTangent

end parabola_circle_properties_l788_788636


namespace leonardo_needs_more_money_l788_788908

-- Defining the problem
def cost_of_chocolate : ℕ := 500 -- 5 dollars in cents
def leonardo_own_money : ℕ := 400 -- 4 dollars in cents
def borrowed_money : ℕ := 59 -- borrowed cents

-- Prove that Leonardo needs 41 more cents
theorem leonardo_needs_more_money : (cost_of_chocolate - (leonardo_own_money + borrowed_money) = 41) :=
by
  sorry

end leonardo_needs_more_money_l788_788908


namespace divide_660_stones_into_30_piles_l788_788988

theorem divide_660_stones_into_30_piles :
  ∃ (heaps : Fin 30 → ℕ),
    (∑ i, heaps i = 660) ∧
    (∀ i j, heaps i < 2 * heaps j ∨ heaps j < 2 * heaps i) :=
sorry

end divide_660_stones_into_30_piles_l788_788988


namespace arctan_tan75_minus_3_tan30_l788_788266

noncomputable def tan75 : ℝ := 1 / (2 - real.sqrt 3)
noncomputable def tan30 : ℝ := 1 / real.sqrt 3

#eval real.arctan (tan75 - 3 * tan30) * (180 / real.pi) -- Converts radians to degrees

theorem arctan_tan75_minus_3_tan30 :
  real.arctan (tan75 - 3 * tan30) * (180 / real.pi) = 63.4349488 := by
  sorry

end arctan_tan75_minus_3_tan30_l788_788266


namespace find_a_0_l788_788390

noncomputable def is_prime_power (n : ℕ) : Prop :=
  ∃ p k : ℕ, prime p ∧ k > 0 ∧ n = p^k

def sequence_condition (a : ℕ → ℕ) : Prop :=
∀ k > 0, a k = Nat.find (λ n, n > a (k - 1) ∧ ∀ i < k, Nat.coprime n (a i))

def sequence_prime_or_prime_power (a_0 : ℕ) (a : ℕ → ℕ) : Prop :=
  a 0 = a_0 ∧ sequence_condition a ∧ ∀ n, prime (a n) ∨ is_prime_power (a n)

theorem find_a_0 :
  ∀ a_0 : ℕ, a_0 > 1 → (∀ a : ℕ → ℕ, sequence_prime_or_prime_power a_0 a) ↔ a_0 ∈ {2, 3, 4, 7, 8} :=
sorry

end find_a_0_l788_788390


namespace projection_onto_3_4_matrix_l788_788341

def projection_matrix := λ (u : ℝ) (v : ℝ), (3 * u + 4 * v) / 25

theorem projection_onto_3_4_matrix :
  ∀ (x y : ℝ),
  (λ (u v : ℝ), (3 * x + 4 * y) / 25) = (λ (u v : ℝ), (\(u * 9 / 25) + (v * 12 / 25), (u * 12 / 25) + (v * 16 / 25))) :=
by
  sorry

end projection_onto_3_4_matrix_l788_788341


namespace midpoint_of_AB_is_2_l788_788034

def midpoint (x₁ x₂ : ℝ) : ℝ := (x₁ + x₂) / 2

theorem midpoint_of_AB_is_2 : midpoint (-3) (7) = 2 :=
by
  sorry

end midpoint_of_AB_is_2_l788_788034


namespace smallest_three_digit_in_pascals_triangle_l788_788202

theorem smallest_three_digit_in_pascals_triangle : ∃ k n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ ∀ m, ((m <= n) ∧ (m >= 100)) → m ≥ n :=
by
  sorry

end smallest_three_digit_in_pascals_triangle_l788_788202


namespace sphere_visibility_area_l788_788911

noncomputable def S (n : ℕ) := sorry -- since it depends on visibility conditions

theorem sphere_visibility_area (n : ℕ) (r : ℕ → ℝ) :
  (∑ i in Finset.range n, S i / (r i)^2) = 4 * Real.pi :=
by
  sorry

end sphere_visibility_area_l788_788911


namespace fraction_inhabitable_l788_788183

-- Define the constants based on the given conditions
def fraction_water : ℚ := 3 / 5
def fraction_inhabitable_land : ℚ := 3 / 4

-- Define the theorem to prove that the fraction of Earth's surface that is inhabitable is 3/10
theorem fraction_inhabitable (w h : ℚ) (hw : w = fraction_water) (hh : h = fraction_inhabitable_land) : 
  (1 - w) * h = 3 / 10 :=
by
  sorry

end fraction_inhabitable_l788_788183


namespace parabola_and_circle_eq_line_A2A3_tangent_l788_788627

-- Define the conditions of the problem
-- Vertex of the parabola at the origin and focus on the x-axis
def parabola_eq : Prop := ∃ p > 0, ∀ x y : ℝ, (y^2 = 2 * p * x ↔ (x, y) ∈ C)

-- Define line l: x = 1
def line_l (x y : ℝ) : Prop := x = 1

-- Define the parabola C and the points of intersection P and Q
def intersection_points (y : ℝ) : Prop := (1, y) ∈ C

-- Define the perpendicularity condition OP ⊥ OQ
def perpendicular_condition (P Q : ℝ × ℝ) : Prop := (∃ p > 0, P = (1, sqrt p) ∧ Q = (1, -sqrt p))

-- Define the point M and its associated circle M tangent to line l
def point_M : ℝ × ℝ := (2, 0)

def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the points A1, A2, A3 on parabola C
def on_parabola (A : ℝ × ℝ) : Prop := (∃ p > 0, A.2^2 = 2 * p * A.1)

-- Define that lines A1A2 and A1A3 are tangent to circle M
def tangent_to_circle (A₁ A₂ : ℝ × ℝ) : Prop := sorry

-- Prove the equation of parabola C and circle M
theorem parabola_and_circle_eq : (∀ x y : ℝ, y^2 = x ∧ (x - 2)^2 + y^2 = 1) :=
by
  sorry

-- Prove the position relationship between line A2A3 and circle M
theorem line_A2A3_tangent (A₁ A₂ A₃ : ℝ × ℝ) :
    on_parabola A₁ ∧ on_parabola A₂ ∧ on_parabola A₃ ∧ tangent_to_circle A₁ A₂ ∧ tangent_to_circle A₁ A₃ →
    (∃ l_tangent : ℝ, tangent_to_circle A₂ A₃) :=
by
  sorry

end parabola_and_circle_eq_line_A2A3_tangent_l788_788627


namespace find_x_l788_788436

-- Define the planar vectors
def vec_a : ℝ × ℝ := (2, 3)
def vec_b (x : ℝ) : ℝ × ℝ := (x, 4)

-- Define the perpendicular condition and dot product operation
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Problem statement: Prove that given the condition, x must be 1/2
theorem find_x (x : ℝ)
  (h : dot_product vec_a (vec_a.1 - vec_b(x).1, vec_a.2 - vec_b(x).2) = 0) :
  x = 1/2 :=
sorry

end find_x_l788_788436


namespace carbon_atoms_in_compound_l788_788724

theorem carbon_atoms_in_compound 
    (molecular_weight : ℕ := 65)
    (carbon_weight : ℕ := 12)
    (hydrogen_weight : ℕ := 1)
    (oxygen_weight : ℕ := 16)
    (hydrogen_atoms : ℕ := 1)
    (oxygen_atoms : ℕ := 1) :
    ∃ (carbon_atoms : ℕ), molecular_weight = (carbon_atoms * carbon_weight) + (hydrogen_atoms * hydrogen_weight) + (oxygen_atoms * oxygen_weight) ∧ carbon_atoms = 4 :=
by
  sorry

end carbon_atoms_in_compound_l788_788724


namespace divide_stones_into_heaps_l788_788953

-- Definitions based on the identified conditions
variable (Heaps : List ℕ) -- list of heap sizes
variable (n_stones : ℕ) -- total number of stones
variable (n_heaps : ℕ) -- number of heaps

-- Conditions
axiom total_stones : n_stones = 660
axiom total_heaps : n_heaps = 30
axiom heap_size_condition : ∀ (h₁ h₂ : ℕ), h₁ ∈ Heaps → h₂ ∈ Heaps → h₁ ≤ 2 * h₂

-- Theorem statement
theorem divide_stones_into_heaps : 
  ∃ Heaps, Heaps.length = n_heaps ∧ Heaps.sum = n_stones ∧
  ∀ (h₁ h₂ : ℕ), h₁ ∈ Heaps → h₂ ∈ Heaps → h₁ ≤ 2 * h₂ :=
by
  sorry

end divide_stones_into_heaps_l788_788953


namespace weights_sum_ordering_l788_788236

-- Define the variables and their conditions
variables {a b c d : ℤ}
axiom h1 : a < b ∧ b < c ∧ c < d
axiom h2 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

-- Theorem statement for part (a)
theorem weights_sum_ordering (h1 : a < b ∧ b < c ∧ c < d) (h2 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  ∃ s : list ℤ, s = [a + b, a + c, a + d, b + c, b + d, c + d] ∧ s.nth 0 = some (a + b) ∧ s.nth 1 = some (a + c) ∧ s.nth 4 = some (b + d) ∧ s.nth 5 = some (c + d) :=
by {
  -- The proof will be written here
  sorry
}

end weights_sum_ordering_l788_788236


namespace remainder_of_acb_mod_n_l788_788940

open Nat

theorem remainder_of_acb_mod_n 
  (n : ℕ) (a b c : ℤ) 
  (hn_pos : 0 < n)
  (ha_inv : IsUnit (Units.mkOfNat n a))
  (hb_inv : IsUnit (Units.mkOfNat n b))
  (hc_inv : IsUnit (Units.mkOfNat n c))
  (h_ab : a ≡ b⁻¹ [ZMOD n]) :
  a * c * b ≡ c [ZMOD n] :=
sorry

end remainder_of_acb_mod_n_l788_788940


namespace car_maintenance_expense_l788_788519

-- Define constants and conditions
def miles_per_year : ℕ := 12000
def oil_change_interval : ℕ := 3000
def oil_change_price (quarter : ℕ) : ℕ := 
  if quarter = 1 then 55 
  else if quarter = 2 then 45 
  else if quarter = 3 then 50 
  else 40
def free_oil_changes_per_year : ℕ := 1

def tire_rotation_interval : ℕ := 6000
def tire_rotation_cost : ℕ := 40
def tire_rotation_discount : ℕ := 10 -- In percent

def brake_pad_interval : ℕ := 24000
def brake_pad_cost : ℕ := 200
def brake_pad_discount : ℕ := 20 -- In percent
def brake_pad_membership_cost : ℕ := 60
def membership_duration : ℕ := 2 -- In years

def total_annual_expense : ℕ :=
  let oil_changes := (miles_per_year / oil_change_interval) - free_oil_changes_per_year
  let oil_cost := (oil_change_price 2 + oil_change_price 3 + oil_change_price 4) -- Free oil change in Q1
  let tire_rotations := miles_per_year / tire_rotation_interval
  let tire_cost := (tire_rotation_cost * (100 - tire_rotation_discount) / 100) * tire_rotations
  let brake_pad_cost_per_year := (brake_pad_cost * (100 - brake_pad_discount) / 100) / membership_duration
  let membership_cost_per_year := brake_pad_membership_cost / membership_duration
  oil_cost + tire_cost + (brake_pad_cost_per_year + membership_cost_per_year)

-- Assert the proof problem
theorem car_maintenance_expense : total_annual_expense = 317 := by
  sorry

end car_maintenance_expense_l788_788519


namespace find_a6_l788_788493

-- Define the geometric sequence and the given terms
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

variables {a : ℕ → ℝ} (r : ℝ)

-- Given conditions
axiom a_2 : a 2 = 2
axiom a_10 : a 10 = 8
axiom geo_seq : geometric_sequence a

-- Statement to prove
theorem find_a6 : a 6 = 4 :=
sorry

end find_a6_l788_788493


namespace solve_equation_l788_788371

noncomputable def smallest_solution : Rat :=
  (8 - Real.sqrt 145) / 3

theorem solve_equation : 
  ∃ x : ℝ, (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ x = smallest_solution := sorry

end solve_equation_l788_788371


namespace totalCandlesInHouse_l788_788906

-- Definitions for the problem's conditions
def bedroomCandles : ℕ := 20
def livingRoomCandles : ℕ := bedroomCandles / 2
def donovanCandles : ℕ := 20

-- Problem to prove
theorem totalCandlesInHouse : bedroomCandles + livingRoomCandles + donovanCandles = 50 := by
  sorry

end totalCandlesInHouse_l788_788906


namespace projection_matrix_l788_788349

theorem projection_matrix
  (x y : ℝ) :
  let v := ![3, 4]
  let proj_v := (v ⬝ ![x, y]) / (v ⬝ v) • v
  let proj_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
    ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]] in
  proj_v = proj_matrix.mul_vec ![x, y] :=
by
  let v := ![3, 4]
  let proj_v := (v ⬝ ![x, y]) / (v ⬝ v) • v
  let proj_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]]
  sorry

end projection_matrix_l788_788349


namespace find_x_l788_788739

theorem find_x (x : ℝ) (h : 0 < x) (hx : 0.01 * x * x^2 = 16) : x = 12 :=
sorry

end find_x_l788_788739


namespace oh_squared_l788_788921

theorem oh_squared (O H : ℝ) (a b c R : ℝ) (h1 : R = 5) (h2 : a^2 + b^2 + c^2 = 50) :
  let OH := H - O in
  OH ^ 2 = 175 :=
by
  sorry

end oh_squared_l788_788921


namespace projection_matrix_l788_788348

theorem projection_matrix
  (x y : ℝ) :
  let v := ![3, 4]
  let proj_v := (v ⬝ ![x, y]) / (v ⬝ v) • v
  let proj_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
    ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]] in
  proj_v = proj_matrix.mul_vec ![x, y] :=
by
  let v := ![3, 4]
  let proj_v := (v ⬝ ![x, y]) / (v ⬝ v) • v
  let proj_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![![9 / 25, 12 / 25], ![12 / 25, 16 / 25]]
  sorry

end projection_matrix_l788_788348


namespace probability_Y_eq_neg2_l788_788831

noncomputable def two_point_distribution (p : ℝ) : ℕ → ℝ
| 0 => 1 - p
| 1 => p
| _ => 0

theorem probability_Y_eq_neg2 :
  let p := 0.6 in
  let X_dist := two_point_distribution p in
  let X := λ ω, if ω = 0 then 0 else 1 in
  let Y := λ ω, 3 * X ω - 2 in
  ∑ ω in {0, 1}, if Y ω = -2 then X_dist ω else 0 = 0.4 :=
by
  sorry

end probability_Y_eq_neg2_l788_788831


namespace smallest_six_digit_number_divisible_by_25_35_45_15_l788_788690

theorem smallest_six_digit_number_divisible_by_25_35_45_15 :
  ∃ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ 
           (25 ∣ n) ∧ 
           (35 ∣ n) ∧ 
           (45 ∣ n) ∧ 
           (15 ∣ n) ∧ 
           (∀ m : ℕ, 100000 ≤ m ∧ m < 1000000 ∧ 
                     (25 ∣ m) ∧ 
                     (35 ∣ m) ∧ 
                     (45 ∣ m) ∧ 
                     (15 ∣ m) → n ≤ m) :=
by
  use 100800
  sorry

end smallest_six_digit_number_divisible_by_25_35_45_15_l788_788690


namespace number_of_cars_in_trains_l788_788650

theorem number_of_cars_in_trains
  (s1 s2 s3 : ℕ)
  (h1 : s1 = 462)
  (h2 : s2 = 546)
  (h3 : s3 = 630)
  (g : ℕ := Nat.gcd (Nat.gcd s1 s2) s3)
  (h_g : g = 42) :
  (s1 / g = 11) ∧ (s2 / g = 13) ∧ (s3 / g = 15) :=
by
  rw [h1, h2, h3, h_g]
  norm_num
  exact dec_trivial

end number_of_cars_in_trains_l788_788650


namespace angle_DME_90_l788_788498

-- Given point E inside parallelogram ABCD such that AE = DE and ∠ABE = 90°
namespace Parallelogram
variables {A B C D E M N : Type*}
variables (AB CD AD BC AE DE BE ME BN DM : Prop)
variables [ADD_COMM_GROUP A] [ADD_COMM_GROUP B] [ADD_COMM_GROUP C]
variables [ADD_COMM_GROUP D] [ADD_COMM_GROUP E] [ADD_COMM_GROUP M] [ADD_COMM_GROUP N]

-- Condition 1: ABCD is a parallelogram
def is_parallelogram (A B C D : Type*) : Prop :=
  ∃ x1 x2 y1 y2 : Type*, 
  AD = y1 /\ BC = y2 /\ AB = x1 /\ CD = x2 
  ∧ (AD ∥ BC) ∧ (AB ∥ CD)

-- Condition 2: AE = DE
def is_isosceles (A E D : Type*) : Prop := 
  AE = DE

-- Condition 3: ∠ABE = 90°
def right_angle (A B E : Type*) : Prop :=
  ∠ABE = 90

-- Condition 4: M is the midpoint of BC
def is_midpoint (B C M : Type*) : Prop := 
  M ∈ B C 
  ∧ dist B M = dist M C

-- Definition: angle DME
def angle_DME (D M E : Type*) : Prop :=
  ∠DME = 90

theorem angle_DME_90 
  (par: is_parallelogram A B C D)
  (eqa: is_isosceles A E D)
  (ang: right_angle A B E)
  (mid: is_midpoint B C M) : 
  angle_DME D M E :=
sorry
end Parallelogram

end angle_DME_90_l788_788498


namespace proof_problem_l788_788461

-- Define the imaginary unit
def i : ℂ := complex.I

-- Given conditions
variable (b : ℝ)
variable (h : (2 - i) * (4 * i) = 4 - b * i)

-- Equivalence proof statement
theorem proof_problem : b = -8 :=
by
  -- Insert the proof steps here if desired.
  sorry

end proof_problem_l788_788461


namespace find_m_n_l788_788137

theorem find_m_n (m n : ℤ) (h : m^2 - 2 * m * n + 2 * n^2 - 8 * n + 16 = 0) : m = 4 ∧ n = 4 := 
by {
  sorry
}

end find_m_n_l788_788137


namespace part_one_part_two_l788_788886

theorem part_one (g : ℝ → ℝ) (h : ∀ x, g x = |x - 1| + 2) : {x : ℝ | |g x| < 5} = {x : ℝ | -2 < x ∧ x < 4} :=
sorry

theorem part_two (f g : ℝ → ℝ) (h1 : ∀ x, f x = |2 * x - a| + |2 * x + 3|) (h2 : ∀ x, g x = |x - 1| + 2) 
(h3 : ∀ x1 : ℝ, ∃ x2 : ℝ, f x1 = g x2) : {a : ℝ | a ≥ -1 ∨ a ≤ -5} :=
sorry

end part_one_part_two_l788_788886


namespace rain_probability_tel_aviv_l788_788693

open scoped Classical

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k) * (p^k) * ((1 - p)^(n - k))

theorem rain_probability_tel_aviv :
  binomial_probability 6 4 0.5 = 0.234375 :=
by 
  sorry

end rain_probability_tel_aviv_l788_788693


namespace min_q_difference_l788_788941

theorem min_q_difference (p q : ℕ) (hpq : 0 < p ∧ 0 < q) (ineq1 : (7:ℚ)/12 < p/q) (ineq2 : p/q < (5:ℚ)/8) (hmin : ∀ r s : ℕ, 0 < r ∧ 0 < s ∧ (7:ℚ)/12 < r/s ∧ r/s < (5:ℚ)/8 → q ≤ s) : q - p = 2 :=
sorry

end min_q_difference_l788_788941


namespace divide_stones_into_heaps_l788_788954

-- Definitions based on the identified conditions
variable (Heaps : List ℕ) -- list of heap sizes
variable (n_stones : ℕ) -- total number of stones
variable (n_heaps : ℕ) -- number of heaps

-- Conditions
axiom total_stones : n_stones = 660
axiom total_heaps : n_heaps = 30
axiom heap_size_condition : ∀ (h₁ h₂ : ℕ), h₁ ∈ Heaps → h₂ ∈ Heaps → h₁ ≤ 2 * h₂

-- Theorem statement
theorem divide_stones_into_heaps : 
  ∃ Heaps, Heaps.length = n_heaps ∧ Heaps.sum = n_stones ∧
  ∀ (h₁ h₂ : ℕ), h₁ ∈ Heaps → h₂ ∈ Heaps → h₁ ≤ 2 * h₂ :=
by
  sorry

end divide_stones_into_heaps_l788_788954


namespace isosceles_triangle_angle_bisectors_equal_isosceles_triangle_medians_equal_l788_788567

-- Definitions for the isosceles triangle and its properties
variables {A B C M A1 B1 : Point}
variables (AB AC BC : ℝ)
variable (isosceles : AB = AC)

-- Part (a): proving the angle bisectors are equal
theorem isosceles_triangle_angle_bisectors_equal 
  (angle_bisector_A : Line)
  (angle_bisector_B : Line)
  (bisects_A : bisects A angle_bisector_A BC) 
  (bisects_B : bisects B angle_bisector_B BC) :
  length angle_bisector_A = length angle_bisector_B :=
sorry

-- Part (b): proving the medians are equal
theorem isosceles_triangle_medians_equal 
  (median_A : Line)
  (median_B : Line)
  (midpoint_M : M = midpoint B C)
  (median_A_bisects: bisects A median_A M)
  (median_B_bisects: bisects B median_B M) :
  length median_A = length median_B :=
sorry

end isosceles_triangle_angle_bisectors_equal_isosceles_triangle_medians_equal_l788_788567


namespace general_term_and_max_n_l788_788397

noncomputable def a_n (n : ℕ) : ℕ := n + 1

def S_n (n : ℕ) : ℕ := n * (n + 3) / 2

theorem general_term_and_max_n :
  (∀ n : ℕ, n > 0 → a_n n = n + 1) ∧
  (let n_max := 8 in ∀ n : ℕ, S_n n < 5 * a_n n → n ≤ n_max) :=
begin
  sorry  
end

end general_term_and_max_n_l788_788397


namespace length_PC_l788_788873

-- Define lengths of the sides of triangle ABC.
def AB := 10
def BC := 8
def CA := 7

-- Define the similarity condition
def similar_triangles (PA PC : ℝ) : Prop :=
  PA / PC = AB / CA

-- Define the extension of side BC to point P
def extension_condition (PA PC : ℝ) : Prop :=
  PA = PC + BC

theorem length_PC (PC : ℝ) (PA : ℝ) :
  similar_triangles PA PC → extension_condition PA PC → PC = 56 / 3 :=
by
  intro h_sim h_ext
  sorry

end length_PC_l788_788873


namespace cupcakes_leftover_l788_788121

theorem cupcakes_leftover {total_cupcakes nutty_cupcakes gluten_free_cupcakes children children_no_nuts child_only_gf leftover_nutty leftover_regular : Nat} :
  total_cupcakes = 84 →
  children = 7 →
  nutty_cupcakes = 18 →
  gluten_free_cupcakes = 25 →
  children_no_nuts = 2 →
  child_only_gf = 1 →
  leftover_nutty = 3 →
  leftover_regular = 2 →
  leftover_nutty + leftover_regular = 5 :=
by
  sorry

end cupcakes_leftover_l788_788121


namespace boat_distance_downstream_l788_788717

-- Definitions of the given conditions
def boat_speed_still_water : ℝ := 13
def stream_speed : ℝ := 4
def travel_time_downstream : ℝ := 4

-- Mathematical statement to be proved
theorem boat_distance_downstream : 
  let effective_speed_downstream := boat_speed_still_water + stream_speed
  in effective_speed_downstream * travel_time_downstream = 68 :=
by
  sorry

end boat_distance_downstream_l788_788717


namespace general_admission_tickets_l788_788151

variable (x y : ℕ)

theorem general_admission_tickets (h1 : x + y = 525) (h2 : 4 * x + 6 * y = 2876) : y = 388 := by
  sorry

end general_admission_tickets_l788_788151


namespace find_a_and_b_l788_788388

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b

theorem find_a_and_b (a b : ℝ) (h_a : a < 0) (h_max : a + b = 3) (h_min : -a + b = -1) : a = -2 ∧ b = 1 :=
by
  sorry

end find_a_and_b_l788_788388


namespace perimeter_with_new_tiles_l788_788555

theorem perimeter_with_new_tiles (p_original : ℕ) (num_original_tiles : ℕ) (num_new_tiles : ℕ)
  (h1 : p_original = 16)
  (h2 : num_original_tiles = 9)
  (h3 : num_new_tiles = 3) :
  ∃ p_new : ℕ, p_new = 17 :=
by
  sorry

end perimeter_with_new_tiles_l788_788555


namespace midpoints_on_nine_point_circle_l788_788697

theorem midpoints_on_nine_point_circle
  (A B C O O1 O2 O3 : Point)
  (circumcenter_ABC : IsCircumcenter O A B C)
  (reflection_O1 : O1 = reflection O (line_through A B))
  (reflection_O2 : O2 = reflection O (line_through B C))
  (reflection_O3 : O3 = reflection O (line_through A C)) :
  let mid_O1O2 := midpoint O1 O2
  let mid_O2O3 := midpoint O2 O3
  let mid_O3O1 := midpoint O3 O1
  in lies_on_nine_point_circle mid_O1O2 A B C ∧
     lies_on_nine_point_circle mid_O2O3 A B C ∧
     lies_on_nine_point_circle mid_O3O1 A B C :=
sorry

end midpoints_on_nine_point_circle_l788_788697


namespace farmer_plough_l788_788729

theorem farmer_plough (x : ℝ) : 
  (∃ D : ℝ, D = 448 / x ∧ (D + 2) * 85 = 408) ∧ 
  448 - ( (448 / x + 2) * 85 - 40) = 448 - 408 :=
  x = 160 :=
begin
  sorry
end

end farmer_plough_l788_788729


namespace find_a_and_b_l788_788838

-- Define the function
def f (a b : ℝ) (x : ℝ) : ℝ := a * (Real.sin (2 * x - Real.pi / 3)) + b

-- Axis of symmetry condition
def axis_of_symmetry (f : ℝ → ℝ) : Prop :=
  ∃ k : ℤ, ∀ x : ℝ, f x = f (x + k * Real.pi / 2 + 5 * Real.pi / 12)

-- Main theorem statement
theorem find_a_and_b (a b : ℝ) (h_a_pos : a > 0) 
  (min_cond : ∃ x ∈ Icc 0 (Real.pi / 2), f a b x = -2) 
  (max_cond : ∃ x ∈ Icc 0 (Real.pi / 2), f a b x = Real.sqrt 3) :
  axis_of_symmetry (f a b) ∧ a = 2 ∧ b = Real.sqrt 3 - 2 := 
begin
  sorry
end

end find_a_and_b_l788_788838


namespace OH_squared_l788_788920

variables {A B C O H : Type}
variables (a b c R : ℝ)

-- Define the conditions
def IsCircumcenter (O : Type) := true -- placeholder, requires precise definition
def IsOrthocenter (H : Type) := true -- placeholder, requires precise definition
def sideLengths (a b c : ℝ) := true -- placeholder, requires precise definition
def circumradius (R : ℝ) := R = 5
def sumOfSquareSides (a b c : ℝ) := a^2 + b^2 + c^2 = 50

-- The main statement to be proven
theorem OH_squared (h1 : IsCircumcenter O)
                   (h2 : IsOrthocenter H)
                   (h3 : sideLengths a b c)
                   (h4 : circumradius R)
                   (h5 : sumOfSquareSides a b c) :
    let OH_squared := 9 * R^2 - (a^2 + b^2 + c^2)
    in OH_squared = 175 := sorry

end OH_squared_l788_788920


namespace oh_squared_l788_788922

theorem oh_squared (O H : ℝ) (a b c R : ℝ) (h1 : R = 5) (h2 : a^2 + b^2 + c^2 = 50) :
  let OH := H - O in
  OH ^ 2 = 175 :=
by
  sorry

end oh_squared_l788_788922


namespace divide_660_stones_into_30_piles_l788_788990

theorem divide_660_stones_into_30_piles :
  ∃ (heaps : Fin 30 → ℕ),
    (∑ i, heaps i = 660) ∧
    (∀ i j, heaps i < 2 * heaps j ∨ heaps j < 2 * heaps i) :=
sorry

end divide_660_stones_into_30_piles_l788_788990


namespace parabola_equation_circle_m_equation_line_a2a3_tangent_to_circle_m_l788_788633

noncomputable theory
open_locale classical

-- Definitions and conditions
def parabola_vertex_origin (x y : ℝ) : Prop := ∃ p : ℝ, p > 0 ∧ y^2 = 2 * p * x
def line_intersects_parabola_perpendicularly : Prop :=
  ∃ p : ℝ, p = 1 / 2 ∧ parabola_vertex_origin 1 p

def circle_m_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1
def line_tangent_to_circle_m (l : ℝ → ℝ) : Prop := ∀ x y : ℝ, circle_m_eq x y → l x = y

def points_on_parabola_and_tangent (A1 A2 A3 : ℝ × ℝ) : Prop :=
  parabola_vertex_origin A1.1 A1.2 ∧
  parabola_vertex_origin A2.1 A2.2 ∧
  parabola_vertex_origin A3.1 A3.2 ∧
  line_tangent_to_circle_m (λ y, A1.1 * y + A1.2) ∧
  line_tangent_to_circle_m (λ y, A1.1 * y + A3.2)

-- Statements to prove
theorem parabola_equation : ∃ C : ℝ → ℝ → Prop, (C = parabola_vertex_origin) := sorry
theorem circle_m_equation : ∃ M : ℝ → ℝ → Prop, (M = circle_m_eq) := sorry
theorem line_a2a3_tangent_to_circle_m :
  ∀ A1 A2 A3 : ℝ × ℝ, 
  (points_on_parabola_and_tangent A1 A2 A3) →
  ∃ l : ℝ → ℝ, line_tangent_to_circle_m l := sorry

end parabola_equation_circle_m_equation_line_a2a3_tangent_to_circle_m_l788_788633


namespace vertex_on_line_intersection_range_b_intersection_points_with_axes_l788_788028

-- Define the quadratic function
def quadratic_function (x m : ℝ) : ℝ := x^2 - 2*m*x + m^2 + m - 1

-- Prove that the vertex of the quadratic function lies on y=x-1
theorem vertex_on_line (m : ℝ) : quadratic_function m m = m - 1 :=
by
  sorry

-- Prove the range of b such that quadratic_function intersects y=x+b at two points
theorem intersection_range_b (x m b: ℝ) (H : quadratic_function x m = x + b) : b > -5 / 4 :=
by
  sorry

-- Prove the number of intersection points with coordinate axes for various m
theorem intersection_points_with_axes (m : ℝ) : 
  (m < 1 ∧ m ≠ (1 - Real.sqrt 5) / 2 ∧ m ≠ (1 + Real.sqrt 5) / 2) ∨ 
  (m = (1 - Real.sqrt 5) / 2) ∨ 
  (m = (1 + Real.sqrt 5) / 2) ∨ 
  (m = 1) ∨ 
  (m > 1) ∧ 
  (quadratic_function 0 m = 0 ∨ quadratic_function 1 m = 1 ∨ quadratic_function 1 m = quadratic_function 0 m) :=
by
  sorry

end vertex_on_line_intersection_range_b_intersection_points_with_axes_l788_788028


namespace identify_cauchy_functions_l788_788050

def is_cauchy_function (f : ℝ → ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    x1 ≠ x2 ∧
    f x1 = y1 ∧ f x2 = y2 ∧
    (|x1 * x2 + y1 * y2| - (Real.sqrt (x1^2 + y1^2)) * (Real.sqrt (x2^2 + y2^2)) = 0)

def cauchy_functions (f : ℝ → ℝ) : Prop :=
  f = (λ x, Real.log x) ∨ f = (λ x, Real.sqrt (2 * x^2 - 8))

theorem identify_cauchy_functions (f : ℝ → ℝ) :
  is_cauchy_function f ↔ cauchy_functions f :=
sorry

end identify_cauchy_functions_l788_788050


namespace perimeter_ABCDHG_eq_27_l788_788186

variables {A B C D E H G : Type} [ordered_field A]

def is_equilateral_triangle (a b c : A) := (a = b) ∧ (b = c) ∧ (c = a)
def is_square (a b c d : A) := (a = b) ∧ (b = c) ∧ (c = d) ∧ (d = a)

-- Given ABC is an equilateral triangle, AB = 6, D is the midpoint of AC, 
-- ADEH is a square, EHG is an equilateral triangle, prove the perimeter is 27
theorem perimeter_ABCDHG_eq_27 (AB BC CA AD DC DE EH HA EH HG GE : A) 
  (AB_eq_6 : AB = 6)
  (is_eq_triangle_ABC : is_equilateral_triangle AB BC CA)
  (midpoint_D : AD = DC)
  (AD_eq_half_CA : AD = CA / 2)
  (is_square_ADEH : is_square AD DE EH HA)
  (is_eq_triangle_EHG : is_equilateral_triangle EH HG GE)
  : AB + BC + CD + DE + EH + HG + GA = 27 :=
sorry

end perimeter_ABCDHG_eq_27_l788_788186


namespace joe_paint_usage_l788_788214

theorem joe_paint_usage :
  ∀ (total_paint initial_remaining_paint final_remaining_paint paint_first_week paint_second_week total_used : ℕ),
  total_paint = 360 →
  initial_remaining_paint = total_paint - paint_first_week →
  final_remaining_paint = initial_remaining_paint - paint_second_week →
  paint_first_week = (2 * total_paint) / 3 →
  paint_second_week = (1 * initial_remaining_paint) / 5 →
  total_used = paint_first_week + paint_second_week →
  total_used = 264 :=
by
  sorry

end joe_paint_usage_l788_788214


namespace minimum_odd_integers_l788_788188

theorem minimum_odd_integers (a b c d e f : ℤ) (h1 : a + b = 27) (h2 : a + b + c + d = 46) (h3 : a + b + c + d + e + f = 65) : 
  ∃ odd_count, odd_count = 3 ∧ (∃ l, {a, b, c, d, e, f} = l.to_finset ∧ l.filter (λ x, x % 2 ≠ 0) = odd_count) :=
by sorry

end minimum_odd_integers_l788_788188


namespace rate_of_descent_l788_788074

-- Define the conditions
def initial_elevation : ℝ := 400
def time_traveled_minutes : ℝ := 5
def final_elevation : ℝ := 350

-- Define the rate of downward travel that needs to be proved
def rate_of_downward_travel := (initial_elevation - final_elevation) / time_traveled_minutes

-- The theorem statement
theorem rate_of_descent
  (h1 : initial_elevation = 400)
  (h2 : time_traveled_minutes = 5)
  (h3 : final_elevation = 350) :
  rate_of_downward_travel = 10 := by
  sorry

end rate_of_descent_l788_788074


namespace maximize_viewing_angle_l788_788605

noncomputable def point := ℝ × ℝ

def line_c (p : point) : Prop := p.2 = p.1 + 1

def point_A : point := (1, 0)
def point_B : point := (3, 0)
def point_C : point := (1, 2)

theorem maximize_viewing_angle :
  (∀ (p : point), line_c p → ∠ point_A p point_B ≤ ∠ point_A point_C point_B) :=
sorry

end maximize_viewing_angle_l788_788605


namespace largest_digit_div_by_6_l788_788197

/-- M is the largest digit such that 3190M is divisible by 6 -/
theorem largest_digit_div_by_6 (M : ℕ) : (M ≤ 9) → (3190 * 10 + M) % 2 = 0 → (3190 * 10 + M) % 3 = 0 → M = 8 := 
by
  intro hM9 hDiv2 hDiv3
  sorry

end largest_digit_div_by_6_l788_788197


namespace range_of_k_l788_788035

noncomputable def vector_a : ℝ × ℝ := (2, 1)
noncomputable def vector_b (k : ℝ) : ℝ × ℝ := (1, k)

def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2
def norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

-- Angle is acute when cosine > 0
def is_acute_angle (k : ℝ) : Prop :=
  let a := vector_a
  let b := vector_b k
  let cos_theta := dot_product a b / (norm a * norm b)
  cos_theta > 0

-- Our task is to prove the range of k
theorem range_of_k (k : ℝ) : is_acute_angle k ↔ (k > -2 ∧ k ≠ 1 / 2) :=
  sorry

end range_of_k_l788_788035


namespace area_is_five_times_l788_788249

open EuclideanGeometry

-- Assume we have a type for points and a function for calculating areas of quadrilaterals.
variable {Point : Type} [Inhabited Point] [AffineSpace ℝ Point]

-- First, define the conditions based on the problem statement.
variables (A B C D A' B' C' D' : Point)
variable (midpoint : Point → Point → Point)

-- Define the conditions that the given points are midpoints of specified segments.
def is_midpoint (M X Y : Point) : Prop := M = midpoint X Y
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry -- Assume this captures the convex property.

-- The proof goal: show that the area of A'B'C'D' is five times the area of ABCD.
theorem area_is_five_times (
  h_convex: is_convex_quadrilateral A B C D)
  (h1 : is_midpoint A D A')
  (h2 : is_midpoint B A B')
  (h3 : is_midpoint C B C')
  (h4 : is_midpoint D C D') :
  area A' B' C' D' = 5 * area A B C D :=
sorry -- Proof not required

end area_is_five_times_l788_788249


namespace parabola_and_circle_tangency_relationship_l788_788643

-- Definitions for points and their tangency
def is_tangent (line : ℝ → ℝ) (circle_center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∃ x, (x - circle_center.1)^2 + (line x - circle_center.2)^2 = radius^2

theorem parabola_and_circle_tangency_relationship :
  (∀ x y: ℝ, y^2 = x → ∃ x, (x - 2)^2 + y^2 = 1) ∧
  (∀ (a1 a2 a3 : ℝ × ℝ),
    (a1.2) ^ 2 = a1.1 → 
    (a2.2) ^ 2 = a2.1 → 
    (a3.2) ^ 2 = a3.1 →
    (is_tangent (λ x, (a1.2 / (a1.1 - x))) (2, 0) 1) →
    (is_tangent (λ x, (a1.2 / (a1.1 - x))) (2, 0) 1) →
    is_tangent (λ x, (a2.2 / (a2.1 - x))) (2, 0) 1 ∧
    is_tangent (λ x, (a3.2 / (a3.1 - x))) (2, 0) 1)
  := 
sorry

end parabola_and_circle_tangency_relationship_l788_788643


namespace secret_sharing_day_l788_788517

theorem secret_sharing_day (students : ℕ) (initial_friends : ℕ) (new_friends_each_day : ℕ) (days : ℕ) (total_people : ℕ) :
  students = 1 ∧ initial_friends = 3 ∧ new_friends_each_day = 3 ∧ days = 7 →
  total_people = (3^(days + 1) - 1)/2 →
  total_people = 3280 →
  days_of_week days = "Sunday"
:= by
  sorry

end secret_sharing_day_l788_788517


namespace sin_alpha_second_quadrant_l788_788409

theorem sin_alpha_second_quadrant (α : ℝ) (h_α_quad_2 : π / 2 < α ∧ α < π) (h_cos_α : Real.cos α = -1 / 3) : Real.sin α = 2 * Real.sqrt 2 / 3 := 
sorry

end sin_alpha_second_quadrant_l788_788409


namespace jimin_initial_candies_l788_788899

def starting_candies (given : ℕ) (leftover : ℕ) : ℕ :=
  given + leftover

theorem jimin_initial_candies (candies_given : ℕ) (candies_left : ℕ) : starting_candies candies_given candies_left = 38 :=
by
  have given : ℕ := 25
  have leftover : ℕ := 13
  show starting_candies given leftover = 38
  sorry

end jimin_initial_candies_l788_788899


namespace num_eight_digit_numbers_with_product_4900_l788_788361

-- Define what it means to be an eight-digit number
def is_eight_digit_number (n : ℕ) : Prop :=
  10000000 ≤ n ∧ n < 100000000

-- Define the condition that the product of the digits equals 4900
def digits_product_eq_4900 (n : ℕ) : Prop :=
  (n.to_digits : List ℕ).prod = 4900

-- Define the main theorem to prove the number of eight-digit numbers whose digits' product equals 4900
theorem num_eight_digit_numbers_with_product_4900 : 
  { n : ℕ // is_eight_digit_number n ∧ digits_product_eq_4900 n }.card = 4200 :=
begin
  sorry
end

end num_eight_digit_numbers_with_product_4900_l788_788361


namespace OH_squared_l788_788916

variables {O H A B C : Type} [inner_product_space ℝ O]

def circumcenter (a b c : ℝ) : Type := -- Definition of circumcenter (e.g., type class for properties)
 sorry -- shared space with orthocenter and triangle sides

def orthocenter (a b c : ℝ) : Type := -- Definition of orthocenter (e.g., type class for properties)
 sorry -- shared space with circumcenter and triangle sides

variables (a b c R : ℝ) (triangle : circumcenter a b c) -- Defining triangle properties
variables (orthotriangle : orthocenter a b c) -- Defining orthotriangle within the triangle properties

theorem OH_squared 
  (hR : R = 5)
  (h_side_sum : a^2 + b^2 + c^2 = 50) : 
  let OH_squared := 
    (3 * R^2 + 2 * (R^2 - (a^2 + b^2 + c^2) / 2)) in
  OH_squared = 75 :=
by
  sorry

end OH_squared_l788_788916


namespace geometric_sequence_l788_788895

noncomputable def sequence : ℕ → ℝ
| 1     := 2
| n + 1 := (sequence n) * (1 / 2)

theorem geometric_sequence (m : ℕ) (h1: sequence 1 = 2) (h4: sequence 4 = 1/4) (hm: m = 15) : 
  sequence m = 2 * (1 / 2)^(m - 1) := by
  sorry

end geometric_sequence_l788_788895


namespace area_of_sector_l788_788008

theorem area_of_sector (θ : ℝ) (r : ℝ) (hθ : θ = 72) (hr : r = 20) : 
  1 / 5 * real.pi * r^2 = 80 * real.pi := 
by 
  -- Given that the central angle of the sector is 72 degrees and the radius is 20 cm
  have h1 : θ / 360 = 1 / 5 := by sorry
  calc
    1 / 5 * real.pi * r^2 
        = (θ / 360) * real.pi * r^2 : by sorry
    ... = 80 * real.pi : by sorry

end area_of_sector_l788_788008


namespace tan_product_identity_l788_788860

theorem tan_product_identity : 
  (∏ k in finset.range 89, 1 + real.tan (k + 1) * real.pi / 180) = 2^45 :=
sorry

end tan_product_identity_l788_788860


namespace cost_price_correct_l788_788611

open Real

-- Define the cost price of the table
def cost_price (C : ℝ) : ℝ := C

-- Define the marked price
def marked_price (C : ℝ) : ℝ := 1.30 * C

-- Define the discounted price
def discounted_price (C : ℝ) : ℝ := 0.85 * (marked_price C)

-- Define the final price after sales tax
def final_price (C : ℝ) : ℝ := 1.12 * (discounted_price C)

-- Given that the final price is 9522.84
axiom final_price_value : final_price 9522.84 = 1.2376 * 7695

-- Main theorem stating the problem to prove
theorem cost_price_correct (C : ℝ) : final_price C = 9522.84 -> C = 7695 := by
  sorry

end cost_price_correct_l788_788611


namespace equal_distribution_arithmetic_sequence_l788_788222

theorem equal_distribution_arithmetic_sequence :
  ∃ a d : ℚ, (a - 2 * d) + (a - d) = (a + (a + d) + (a + 2 * d)) ∧
  5 * a = 5 ∧
  a + 2 * d = 2 / 3 :=
by
  sorry

end equal_distribution_arithmetic_sequence_l788_788222


namespace extra_people_needed_l788_788285

theorem extra_people_needed 
  (initial_people : ℕ) 
  (initial_time : ℕ) 
  (final_time : ℕ) 
  (work_done : ℕ) 
  (all_paint_same_rate : initial_people * initial_time = work_done) :
  initial_people = 8 →
  initial_time = 3 →
  final_time = 2 →
  work_done = 24 →
  ∃ extra_people : ℕ, extra_people = 4 :=
by
  sorry

end extra_people_needed_l788_788285


namespace OH_squared_l788_788928

/-- 
Given:
  O is the circumcenter of triangle ABC.
  H is the orthocenter of triangle ABC.
  a, b, and c are the side lengths of triangle ABC.
  R is the circumradius of triangle ABC.
  R = 5.
  a^2 + b^2 + c^2 = 50.

Prove:
  OH^2 = 175.
-/
theorem OH_squared (a b c R : ℝ) (hR : R = 5) (habc : a^2 + b^2 + c^2 = 50) :
  let OH_squared := 9 * R^2 - (a^2 + b^2 + c^2) in
  OH_squared = 175 :=
by
  sorry

end OH_squared_l788_788928


namespace problem_statement_l788_788424

-- Define the function f and the conditions
variables (a : ℝ) (f : ℝ → ℝ) (xi : ℝ → ℝ)
variable (σ : ℝ)

noncomputable def integral_condition := ∫ x in -a..a, x^2 + real.sin x = 18
constant normal_dist : xi = measure_theory.measure.normal 1 σ^2
constant prob_dist1 : measure_theory.probability_measure xi ≤ 4 = 0.79

-- Define the properties needed for problem 3
axiom odd_func (f : ℝ → ℝ) : ∀ x, f(-x) = -f(x)
axiom periodic_func (f : ℝ → ℝ) : ∀ x, f(x + 2) = -f(x)

-- Lean 4 statement for the problem
theorem problem_statement :
  (
    (integral_condition → a = 3) ∧ 
    (¬(larger_R2_worse : ∀ R2 : ℝ, regression_effect R2)) ∧
    (odd_func f → periodic_func f → symmetric_about_x1: ∀ x, f(2 + x) = f(-x)) ∧
    (normal_dist xi ∧ prob_dist1 xi σ → measure_theory.probability_measure xi <-2> = 0.21)
  ) :=
sorry

end problem_statement_l788_788424


namespace sum_of_squares_l788_788464

theorem sum_of_squares :
  ∃ p q r s t u : ℤ, (∀ x : ℤ, 729 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) ∧ 
    (p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 8210) :=
sorry

end sum_of_squares_l788_788464


namespace minimum_n_value_l788_788824

open Real

variables {a : ℕ → ℝ} {r : ℝ}

-- Conditions
axiom geometric_sequence_pos : ∀ n, a n > 0
axiom common_ratio_gt_one : r > 1
axiom geometric_sequence : ∀ n, a (n + 1) = a n * r
axiom product_of_terms : ∀ n, ∏ i in finset.range n, a i = T n
axiom inequality_condition : 2 * a 4 > a 3
axiom specific_relation : a 2 * a 4 = a 3 ^ 2

-- Lean 4 statement
theorem minimum_n_value : ∃ n, n > 1 ∧ 2 * a 4 > a 3 ∧ a 2 * a 4 = a 3 ^ 2 ∧ n = 6 := sorry

end minimum_n_value_l788_788824


namespace john_ultramarathon_distance_l788_788075

theorem john_ultramarathon_distance :
  let initial_time := 8
  let time_increase_percentage := 0.75
  let speed_increase := 4
  let initial_speed := 8
  initial_time * (1 + time_increase_percentage) * (initial_speed + speed_increase) = 168 :=
by
  let initial_time := 8
  let time_increase_percentage := 0.75
  let speed_increase := 4
  let initial_speed := 8
  sorry

end john_ultramarathon_distance_l788_788075


namespace inductive_reasoning_proof_l788_788280

-- Definitions based on problem's conditions
def inductive_reasoning (n : ℕ → Prop) : Prop :=
  ∀ (n : ℕ), n = 1 ∨ n = 2 ∨ n = 4

def inference1 := true -- Placeholder for "Inferring properties of a ball by analogy with properties of a circle" being inductive reasoning
def inference2 := true -- Placeholder for "Inferring that the sum of the internal angles of all triangles is 180º ..."
def inference3 := false -- Placeholder for "Inferring that all students in the class scored 100 points because Zhang Jun scored 100 points ..."
def inference4 := true -- Placeholder for "Inferring the formula of each term of the sequence 1, 0, 1, 0, ..."

-- Main theorem
theorem inductive_reasoning_proof 
  (h1 : inference1 = true) 
  (h2 : inference2 = true) 
  (h3 : inference3 = false) 
  (h4 : inference4 = true) :
  inductive_reasoning (λ n, n = 1 ∨ n = 2 ∨ n = 4) :=
by sorry

end inductive_reasoning_proof_l788_788280


namespace max_cake_pieces_l788_788671

theorem max_cake_pieces : 
  ∀ (cake : ℕ), cake = 20^2 → 
  ∀ (sizes : list ℕ), sizes = [2^2, 4^2, 6^2] → 
  ∃ (num_pieces : ℕ), num_pieces = 18 :=
by
  assume cake h_cake sizes h_sizes,
  sorry

end max_cake_pieces_l788_788671


namespace line_intersects_circle_always_l788_788391

def circle (x y : ℝ) := (x - 2) ^ 2 + (y - 3) ^ 2 = 4
def line (m x y : ℝ) := (m + 2) * x + (2 * m + 1) * y = 7 * m + 8
def point_in_circle (p : ℝ × ℝ) := circle p.1 p.2

theorem line_intersects_circle_always (m : ℝ) :
  ∃ x y : ℝ, line m x y ∧ circle x y :=
sorry

end line_intersects_circle_always_l788_788391


namespace number_of_mixed_vegetable_plates_l788_788250

def cost_of_chapati := 6
def cost_of_rice := 45
def cost_of_mixed_vegetable := 70
def chapatis_ordered := 16
def rice_ordered := 5
def ice_cream_cups := 6 -- though not used, included for completeness
def total_amount_paid := 1111

def total_cost_of_known_items := (chapatis_ordered * cost_of_chapati) + (rice_ordered * cost_of_rice)
def amount_spent_on_mixed_vegetable := total_amount_paid - total_cost_of_known_items

theorem number_of_mixed_vegetable_plates : 
  amount_spent_on_mixed_vegetable / cost_of_mixed_vegetable = 11 := 
by sorry

end number_of_mixed_vegetable_plates_l788_788250


namespace part1_part2_l788_788103

-- Definitions for problem conditions and questions

/-- 
Let p and q be two distinct prime numbers greater than 5. 
Show that if p divides 5^q - 2^q then q divides p - 1.
-/
theorem part1 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hp_gt_5 : 5 < p) (hq_gt_5 : 5 < q) (h_distinct : p ≠ q) 
  (h_div : p ∣ 5^q - 2^q) : q ∣ p - 1 :=
by sorry

/-- 
Let p and q be two distinct prime numbers greater than 5.
Deduce that pq does not divide (5^p - 2^p)(5^q - 2^q).
-/
theorem part2 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hp_gt_5 : 5 < p) (hq_gt_5 : 5 < q) (h_distinct : p ≠ q) 
  (h_div_q_p1 : q ∣ p - 1)
  (h_div_p_q1 : p ∣ q - 1) : ¬(pq : ℕ) ∣ (5^p - 2^p) * (5^q - 2^q) :=
by sorry

end part1_part2_l788_788103


namespace students_before_new_year_le_197_l788_788706

variable (N M k ℓ : ℕ)

-- Conditions
axiom condition_1 : M = (k * N) / 100
axiom condition_2 : 100 * M = k * N
axiom condition_3 : 100 * (M + 1) = ℓ * (N + 3)
axiom condition_4 : ℓ < 100

-- The theorem to prove
theorem students_before_new_year_le_197 :
  N ≤ 197 :=
by
  sorry

end students_before_new_year_le_197_l788_788706


namespace projection_onto_vector_l788_788322

noncomputable def projection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  ![\[9 / 25, 12 / 25\], \[12 / 25, 16 / 25\]]

theorem projection_onto_vector:
    ∀ (x y : ℚ), (Matrix.mul_vec projection_matrix ![\x, \y]) = ![(9 * x + 12 * y) / 25, (12 * x + 16 * y) / 25] := by
  sorry

end projection_onto_vector_l788_788322


namespace f_7_eq_minus_1_l788_788935

-- Define the odd function f with the given properties
def is_odd_function (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f x

def period_2 (f : ℝ → ℝ) :=
  ∀ x, f (x + 2) = -f x

def f_restricted (f : ℝ → ℝ) :=
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 -> f x = x

-- The main statement: Under the given conditions, f(7) = -1
theorem f_7_eq_minus_1 (f : ℝ → ℝ)
  (H1 : is_odd_function f)
  (H2 : period_2 f)
  (H3 : f_restricted f) :
  f 7 = -1 :=
by
  sorry

end f_7_eq_minus_1_l788_788935


namespace find_eccentricity_l788_788015

variables (a b : ℝ)
variables (h1 : a > b) (h2 : b > 0)
variables (h3 : ∃ x y : ℝ, bx - ay + 2 * a * b = 0 ∧ (x^2)/(a^2) + (y^2)/(b^2) = 1)

def eccentricity (a b : ℝ) : ℝ := 
  Real.sqrt (1 - (b^2) / (a^2))

theorem find_eccentricity : 
  ∀ (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : ∃ x y : ℝ, bx - ay + 2 * a * b = 0 ∧ (x^2)/(a^2) + (y^2)/(b^2) = 1),
  (a^2 = 3 * b^2) → eccentricity a b = Real.sqrt (2 / 3) :=
by 
  sorry

end find_eccentricity_l788_788015


namespace emily_quiz_score_l788_788778

theorem emily_quiz_score :
  ∃ x : ℕ, 94 + 88 + 92 + 85 + 97 + x = 6 * 90 :=
by
  sorry

end emily_quiz_score_l788_788778


namespace find_value_of_alpha_beta_plus_alpha_plus_beta_l788_788804

variable (α β : ℝ)

theorem find_value_of_alpha_beta_plus_alpha_plus_beta
  (hα : α^2 + α - 1 = 0)
  (hβ : β^2 + β - 1 = 0)
  (hαβ : α ≠ β) :
  α * β + α + β = -2 := 
by
  sorry

end find_value_of_alpha_beta_plus_alpha_plus_beta_l788_788804


namespace farmer_initial_tomatoes_l788_788728

theorem farmer_initial_tomatoes 
  (T : ℕ) -- The initial number of tomatoes
  (picked : ℕ)   -- The number of tomatoes picked
  (diff : ℕ) -- The difference between initial number of tomatoes and picked
  (h1 : picked = 9) -- The farmer picked 9 tomatoes
  (h2 : diff = 8) -- The difference is 8
  (h3 : T - picked = diff) -- T - 9 = 8
  :
  T = 17 := sorry

end farmer_initial_tomatoes_l788_788728


namespace find_a_l788_788016

def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 2 else x^2

theorem find_a (a : ℝ) : (f 4 = 2 * f a) → (a = 2 ∨ a = -1) :=
by
  sorry

end find_a_l788_788016


namespace trig_identity_proof_l788_788411

/-- Theorem to prove that given sin(θ) + cos(θ) = 4/3 and π/4 < θ < π/2, 
    then cos(θ) - sin(θ) = - √(2)/3 -/
theorem trig_identity_proof (θ : ℝ) 
  (h1 : sin θ + cos θ = 4 / 3) 
  (h2 : π / 4 < θ ∧ θ < π / 2) :
  cos θ - sin θ = - √ 2 / 3 := 
sorry

end trig_identity_proof_l788_788411


namespace parabola_and_circle_tangency_l788_788618

open Real

noncomputable def parabola_eq : Prop :=
  (parabola : {x : ℝ → ℝ | ∃ y: ℝ, y^2 = x})

noncomputable def circle_eq : Prop :=
  (circle : {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2)^2 = 1})

theorem parabola_and_circle_tangency:
  (∀ x y : ℝ, ∃ p, y^2 = x ↔ p ∈ parabola_eq) →
  ((x - 2)^2 + y^2 = 1) →
  (∀ A1 A2 A3 : ℝ × ℝ,
    A1 ∈ parabola_eq ∧ A2 ∈ parabola_eq ∧ A3 ∈ parabola_eq →
    (tangential A1 A2 circle ∧ tangential A1 A3 circle →
    tangential A2 A3 circle
  )) := sorry

end parabola_and_circle_tangency_l788_788618


namespace parabola_and_circle_eq_line_A2A3_tangent_l788_788628

-- Define the conditions of the problem
-- Vertex of the parabola at the origin and focus on the x-axis
def parabola_eq : Prop := ∃ p > 0, ∀ x y : ℝ, (y^2 = 2 * p * x ↔ (x, y) ∈ C)

-- Define line l: x = 1
def line_l (x y : ℝ) : Prop := x = 1

-- Define the parabola C and the points of intersection P and Q
def intersection_points (y : ℝ) : Prop := (1, y) ∈ C

-- Define the perpendicularity condition OP ⊥ OQ
def perpendicular_condition (P Q : ℝ × ℝ) : Prop := (∃ p > 0, P = (1, sqrt p) ∧ Q = (1, -sqrt p))

-- Define the point M and its associated circle M tangent to line l
def point_M : ℝ × ℝ := (2, 0)

def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the points A1, A2, A3 on parabola C
def on_parabola (A : ℝ × ℝ) : Prop := (∃ p > 0, A.2^2 = 2 * p * A.1)

-- Define that lines A1A2 and A1A3 are tangent to circle M
def tangent_to_circle (A₁ A₂ : ℝ × ℝ) : Prop := sorry

-- Prove the equation of parabola C and circle M
theorem parabola_and_circle_eq : (∀ x y : ℝ, y^2 = x ∧ (x - 2)^2 + y^2 = 1) :=
by
  sorry

-- Prove the position relationship between line A2A3 and circle M
theorem line_A2A3_tangent (A₁ A₂ A₃ : ℝ × ℝ) :
    on_parabola A₁ ∧ on_parabola A₂ ∧ on_parabola A₃ ∧ tangent_to_circle A₁ A₂ ∧ tangent_to_circle A₁ A₃ →
    (∃ l_tangent : ℝ, tangent_to_circle A₂ A₃) :=
by
  sorry

end parabola_and_circle_eq_line_A2A3_tangent_l788_788628


namespace problem_1_problem_2_l788_788840

def f (x : ℝ) : ℝ := abs (3 * x + 2)

theorem problem_1 (x : ℝ) : f x < 4 - abs (x - 1) → x ∈ set.Ioo (-5/4) (1/2) := sorry

theorem problem_2 (m n a : ℝ) (hmn : m + n = 1) (hm_pos : m > 0) (hn_pos : n > 0) (ha_pos : a > 0) :
  (∀ x : ℝ, abs (x - a) - f x ≤ 1/m + 1/n) → a ∈ set.Ioo 0 (10/3) := sorry

end problem_1_problem_2_l788_788840


namespace calculate_a_minus_b_l788_788866

noncomputable def a := (20 - 7) / (9 - 5 : ℝ)
noncomputable def b := 7 - a * 5

theorem calculate_a_minus_b :
  a - b = 12.5 :=
by
  rw [a, b]
  have : a = 13 / 4, from by norm_num
  have : b = -9.25, from by norm_num
  norm_num
  sorry

end calculate_a_minus_b_l788_788866


namespace correct_statement_l788_788684

noncomputable theory

structure Point where
  x y z : ℝ

structure Plane where
  points : set Point
  hgc : ∀ (p1 p2 : Point), p1 ∈ points → p2 ∈ points → p1 ≠ p2 → ∃! line : set Point, ∀ p : Point, p ∈ line ↔ collinear p1 p2

def trapezoid : list Point := [Point.mk 0 0 0, Point.mk 1 0 0, Point.mk 0 1 0, Point.mk 1 1 0]

def isPlaneFigure (figure : list Point) : Prop :=
  ∃ (pl : Plane), ∀ p ∈ figure, p ∈ pl.points

axiom determinesPlane (points : list Point) : points.length = 3 → ∃ pl : Plane, ∀ p ∈ points, p ∈ pl.points

axiom dividesIntoFourParts (P1 P2 : Plane) : ¬ parallel P1 P2 → ∀ (space : set Point), dividesSpaceIntoParts P1 P2 space = 4

axiom intersectsAtThreeNonCollinearPoints (P1 P2 : Plane) : P1 ≠ P2 → ∃ p1 p2 p3 : Point, p1 ∈ P1.points ∧ p1 ∈ P2.points ∧ p2 ∈ P1.points ∧ p2 ∈ P2.points ∧ p3 ∈ P1.points ∧ p3 ∈ P2.points ∧ ¬ collinear p1 p2 p3

theorem correct_statement : isPlaneFigure(trapezoid) := 
  sorry

end correct_statement_l788_788684


namespace number_of_whole_numbers_between_sqrt50_and_sqrt200_l788_788038

theorem number_of_whole_numbers_between_sqrt50_and_sqrt200 :
  (finset.Ico 8 15).card = 7 := by
  sorry

end number_of_whole_numbers_between_sqrt50_and_sqrt200_l788_788038


namespace additional_people_needed_l788_788287

theorem additional_people_needed
  (initial_people : ℕ) (initial_time : ℕ) (new_time : ℕ)
  (h_initial : initial_people * initial_time = 24)
  (h_time : new_time = 2)
  (h_initial_people : initial_people = 8)
  (h_initial_time : initial_time = 3) :
  (24 / new_time) - initial_people = 4 :=
by
  sorry

end additional_people_needed_l788_788287


namespace ball_reaches_below_40_cm_at_bounce_8_l788_788714

-- Definitions to set up the initial conditions
def initial_height : ℝ := 360
def bounce_fraction : ℝ := 3 / 4
def threshold_height : ℝ := 40

-- The actual theorem to be proven
theorem ball_reaches_below_40_cm_at_bounce_8 :
  ∃ b : ℕ, 
  initial_height * (bounce_fraction ^ b) < threshold_height ∧ 
  ∀ b' : ℕ, b' < b → initial_height * (bounce_fraction ^ b') ≥ threshold_height :=
sorry

end ball_reaches_below_40_cm_at_bounce_8_l788_788714


namespace sum_logs_geometric_sequence_l788_788011

theorem sum_logs_geometric_sequence (a : ℕ → ℝ) (n : ℕ) (h_pos : ∀ m, a m > 0) 
  (h_condition : ∀ k, k ≥ 3 → a 5 * a (2*k - 5) = 2^(2 * k)) :
  ∑ i in finset.range n, real.logb 2 (a (2 * i + 1)) = n^2 := 
begin
  -- the proof would go here
  sorry
end

end sum_logs_geometric_sequence_l788_788011


namespace smallest_solution_is_39_over_8_l788_788363

noncomputable def smallest_solution (x : ℝ) : Prop :=
  (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ (x ≠ 0) ∧ (x ≠ 3)

theorem smallest_solution_is_39_over_8 : ∃ x > 0, smallest_solution x ∧ x = 39 / 8 :=
by
  sorry

end smallest_solution_is_39_over_8_l788_788363


namespace equation_represents_hyperbola_l788_788770

theorem equation_represents_hyperbola (x y : ℝ) :
  x^2 - 4*y^2 - 2*x + 8*y - 8 = 0 → ∃ a b h k : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ (a * (x - h)^2 - b * (y - k)^2 = 1) := 
sorry

end equation_represents_hyperbola_l788_788770


namespace at_least_half_girls_l788_788902

/--
John has five children. Each child is equally likely to be a boy or a girl.
Prove that the probability that at least half of them are girls is 1/2.
-/
theorem at_least_half_girls (h : true) :
  let total_combinations := 2^5,
      three_girls_ways := Nat.choose 5 3,
      four_girls_ways := Nat.choose 5 4,
      five_girls_ways := Nat.choose 5 5,
      favorable_combinations := three_girls_ways + four_girls_ways + five_girls_ways
  in (favorable_combinations : ℚ) / (total_combinations : ℚ) = 1 / 2 :=
by
  unfold total_combinations three_girls_ways four_girls_ways five_girls_ways favorable_combinations
  sorry

end at_least_half_girls_l788_788902


namespace unique_element_of_connected_space_l788_788084

open Topology Metric

variables {X : Type*} [MetricSpace X]

theorem unique_element_of_connected_space (hX1 : Nonempty X) (hX2 : IsConnected X) (hX3 : ∀ (x : X) (u : ℕ → X) (h : ∀ n, d (u n) x < 1 / (n + 1)), ∃ n, u n = x) : 
  ∃! (x : X), ∀ x', x' = x :=
sorry

end unique_element_of_connected_space_l788_788084


namespace trigonometric_expression_value_l788_788820

theorem trigonometric_expression_value (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - 2 * Real.cos α) = 5 := by
  sorry

end trigonometric_expression_value_l788_788820


namespace sum_powers_l788_788089

open Complex

theorem sum_powers (ω : ℂ) (h₁ : ω^5 = 1) (h₂ : ω ≠ 1) : 
  ω^10 + ω^12 + ω^14 + ω^16 + ω^18 + ω^20 + ω^22 + ω^24 + ω^26 + ω^28 + ω^30 = 1 := sorry

end sum_powers_l788_788089


namespace parabola_and_circle_tangency_relationship_l788_788646

-- Definitions for points and their tangency
def is_tangent (line : ℝ → ℝ) (circle_center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∃ x, (x - circle_center.1)^2 + (line x - circle_center.2)^2 = radius^2

theorem parabola_and_circle_tangency_relationship :
  (∀ x y: ℝ, y^2 = x → ∃ x, (x - 2)^2 + y^2 = 1) ∧
  (∀ (a1 a2 a3 : ℝ × ℝ),
    (a1.2) ^ 2 = a1.1 → 
    (a2.2) ^ 2 = a2.1 → 
    (a3.2) ^ 2 = a3.1 →
    (is_tangent (λ x, (a1.2 / (a1.1 - x))) (2, 0) 1) →
    (is_tangent (λ x, (a1.2 / (a1.1 - x))) (2, 0) 1) →
    is_tangent (λ x, (a2.2 / (a2.1 - x))) (2, 0) 1 ∧
    is_tangent (λ x, (a3.2 / (a3.1 - x))) (2, 0) 1)
  := 
sorry

end parabola_and_circle_tangency_relationship_l788_788646


namespace perimeter_convex_polygon_lt_pi_d_l788_788565

theorem perimeter_convex_polygon_lt_pi_d (n : ℕ) (d : ℝ) (h : d > 0) 
  (lengths : fin n → ℝ) 
  (convex : ∀ (i j : fin n), lengths i = lengths j) 
  (side_cond : ∀ (i : fin n), lengths i < d) 
  (diagonal_cond : ∀ (i j : fin n), i ≠ j → (lengths i + lengths j) < d) :
  ∑ i, lengths i < π * d :=
by
  sorry

end perimeter_convex_polygon_lt_pi_d_l788_788565


namespace range_of_a_l788_788434

-- Define the function f
def f (a x : ℝ) : ℝ := log a ((x - 2 * a) / (x + 2 * a))

-- Define the conditions and the theorem to be proved
theorem range_of_a (s t a : ℝ) (cond1 : a > 0) (cond2 : a ≠ 1) (cond3 : ∀ x ∈ Set.Icc s t, f a x ∈ Set.Icc (log a (t - a)) (log a (s - a))) :
  0 < a ∧ a < 1/5 :=
begin
  sorry
end

end range_of_a_l788_788434


namespace divide_660_stones_into_30_piles_l788_788979

theorem divide_660_stones_into_30_piles :
  ∃ (heaps : Fin 30 → ℕ),
    (∑ i, heaps i = 660) ∧
    ∀ i j, heaps i ≤ 2 * heaps j :=
sorry

end divide_660_stones_into_30_piles_l788_788979


namespace complex_parts_l788_788469

theorem complex_parts (z : ℂ) (hz : z = 2 - 3 * complex.i) : (z.re = 2 ∧ z.im = -3) :=
by
  sorry

end complex_parts_l788_788469


namespace cylindrical_pencils_common_point_l788_788406

theorem cylindrical_pencils_common_point :
  ∃ P : fin 6 → ℝ × ℝ × ℝ, ∀ i j : fin 6, i ≠ j → ∃ p : ℝ × ℝ × ℝ, on_boundary (P i) (d) p ∧ on_boundary (P j) (d) p :=
sorry

-- Definitions for "on_boundary" must be provided, assuming the standard definition of touching the boundary of the cylindrical pencil.

end cylindrical_pencils_common_point_l788_788406


namespace sum_of_remainders_3_digit_numbers_l788_788217

theorem sum_of_remainders_3_digit_numbers :
  let a_1 := 102
  let a_n := 998
  let d := 3
  let n := (a_n - a_1) / d + 1
  let S_n := n / 2 * (a_1 + a_n)
  S_n = 164450 :=
by
  -- Definitions based on conditions
  let a_1 := 102
  let a_n := 998
  let d := 3
  let n := (a_n - a_1) / d + 1
  let S_n := n / 2 * (a_1 + a_n)
  -- Skip the proof
  sorry

end sum_of_remainders_3_digit_numbers_l788_788217


namespace solution_set_of_abs_2x_minus_1_ge_3_l788_788375

theorem solution_set_of_abs_2x_minus_1_ge_3 :
  { x : ℝ | |2 * x - 1| ≥ 3 } = { x : ℝ | x ≤ -1 } ∪ { x : ℝ | x ≥ 2 } := 
sorry

end solution_set_of_abs_2x_minus_1_ge_3_l788_788375


namespace calculate_percentage_l788_788877

theorem calculate_percentage :
  let total_students := 40
  let A_on_both := 4
  let B_on_both := 6
  let C_on_both := 3
  let D_on_Test1_C_on_Test2 := 2
  let valid_students := A_on_both + B_on_both + C_on_both + D_on_Test1_C_on_Test2
  (valid_students / total_students) * 100 = 37.5 :=
by
  sorry

end calculate_percentage_l788_788877
