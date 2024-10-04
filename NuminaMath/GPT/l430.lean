import Mathlib
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Quadratic.Discr
import Mathlib.Analysis.Calculus.Fderiv.Basic
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Advanced
import Mathlib.Data.Finset
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.LinearAlgebra.Basic
import Mathlib.NumberTheory.ArithmeticFunction
import Mathlib.Prob.Basic
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.Independence
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic

namespace student_groups_l430_430543

theorem student_groups (N : ℕ) :
  (∃ (n : ℕ), n = 13 ∧ ∃ (m : ℕ), m ∈ {12, 14} ∧ N = 4 * 13 + 2 * m) → (N = 76 ∨ N = 80) :=
by
  intro h
  obtain ⟨n, hn, m, hm, hN⟩ := h
  rw [hn, hN]
  cases hm with h12 h14
  case inl =>
    simp [h12]
  case inr =>
    simp [h14]
  sorry

end student_groups_l430_430543


namespace line_to_slope_intercept_l430_430715

noncomputable def line_equation (v p q : ℝ × ℝ) : Prop :=
  (v.1 * (p.1 - q.1) + v.2 * (p.2 - q.2)) = 0

theorem line_to_slope_intercept (x y m b : ℝ) :
  line_equation (3, -4) (x, y) (2, 8) → (m, b) = (3 / 4, 6.5) :=
  by
    sorry

end line_to_slope_intercept_l430_430715


namespace sqrt_mul_sqrt_eq_six_l430_430293

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l430_430293


namespace geometric_seq_min_value_problem_l430_430803

theorem geometric_seq_min_value_problem 
  (a : ℕ → ℝ) 
  (h_geom : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q)
  (h_pos : ∀ n : ℕ, 0 < a n)
  (h_cond : a 8 = a 6 + 2 * a 4)
  (h_terms : ∃ m n : ℕ, sqrt (a m * a n) = sqrt 2 * a 1) :
  ∃ m n : ℕ, m + n = 4 ∧ (1 / m + 9 / n) = 4 :=
sorry

end geometric_seq_min_value_problem_l430_430803


namespace length_segment_cutoff_by_curve_l430_430841

theorem length_segment_cutoff_by_curve (t : ℝ) : 
  let C := {p : ℝ × ℝ | p.1 ^ 2 + (p.2 - 2) ^ 2 = 4},
      l := {p : ℝ × ℝ | ∃ t : ℝ, p = (2 * t, sqrt 3 * t + 2)} in
  ∃ segment_length : ℝ, segment_length = 4 :=
by
  sorry

end length_segment_cutoff_by_curve_l430_430841


namespace area_of_square_l430_430459

variable (x y : ℝ)
variable (d := 5 * x + 3 * y)
variable (s : ℝ)

theorem area_of_square : (d ^ 2 = 2 * s ^ 2) → s = (d / real.sqrt 2)
  → s ^ 2 = 1 / 2 * (d ^ 2) :=
by
  intros h1 h2
  rw [h1, ←h2]
  ring

end area_of_square_l430_430459


namespace sqrt_mul_simp_l430_430217

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l430_430217


namespace total_students_in_groups_l430_430531

theorem total_students_in_groups {N : ℕ} (h1 : ∃ g : ℕ → ℕ, (∀ i j, g i = 13 ∨ g j = 12 ∨ g j = 14) ∧ (∑ i in finset.range 6, g i) = N) : 
  N = 76 ∨ N = 80 :=
sorry

end total_students_in_groups_l430_430531


namespace decode_digit_l430_430887

-- Given conditions
def identical_shapes_encode_identical_digits (a b : ℕ) : Prop := true
def different_shapes_encode_different_digits (a b : ℕ) : Prop := a ≠ b
def raising_base_to_triangle_results_in_three_digit_number (base triangle : ℕ) : Prop := 
  let result := base ^ triangle
  100 ≤ result ∧ result < 1000

-- Statement of the problem
theorem decode_digit 
  (Δ : ℕ) (□ : ℕ) (△ : ℕ) (○ : ℕ) 
  (h_identical : identical_shapes_encode_identical_digits Δ Δ)
  (h_different1 : different_shapes_encode_different_digits Δ □)
  (h_different2 : different_shapes_encode_different_digits Δ △)
  (h_different3 : different_shapes_encode_different_digits Δ ○)
  (h_base_triangle : raising_base_to_triangle_results_in_three_digit_number (Δ * 10 + □) △) :
  □ = 6 :=
by
  sorry


end decode_digit_l430_430887


namespace dihedral_sum_lt_2pi_l430_430685

variable (T: Type) [TopologicalSpace T]

theorem dihedral_sum_lt_2pi (α β γ δ : Real) (AB BC CD DA : T)
  (h₁ : ∀ (A B C D : T), ∑ α + ∑ β + ∑ γ + ∑ δ = 4 * Real.pi) :
  α + β + γ + δ < 2 * Real.pi :=
sorry

end dihedral_sum_lt_2pi_l430_430685


namespace pyramid_layers_total_l430_430039

-- Since we are dealing with natural number calculations, noncomputable is generally not needed.

-- Definition of the pyramid layers and the number of balls in each layer
def number_of_balls (n : ℕ) : ℕ := n ^ 2

-- Given conditions for the layers
def third_layer_balls : ℕ := number_of_balls 3
def fifth_layer_balls : ℕ := number_of_balls 5

-- Statement of the problem proving that their sum is 34
theorem pyramid_layers_total : third_layer_balls + fifth_layer_balls = 34 := by
  sorry -- proof to be provided

end pyramid_layers_total_l430_430039


namespace sqrt_mult_simplify_l430_430249

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l430_430249


namespace total_students_76_or_80_l430_430578

theorem total_students_76_or_80 
  (N : ℕ)
  (h1 : ∃ g : ℕ → ℕ, (∑ i in finset.range 6, g i = N) ∧
                     (∃ a b : ℕ, finset.card {i | g i = a} = 4 ∧ 
                                 finset.card {i | g i = b} = 2 ∧ 
                                 (a = 13 ∧ (b = 12 ∨ b = 14))))
  : N = 76 ∨ N = 80 :=
sorry

end total_students_76_or_80_l430_430578


namespace value_of_n_l430_430020

theorem value_of_n 
  (m n : ℝ) 
  (h1 : 0 < m) 
  (h2 : 0 < n) 
  (h3 : m * n = 7) 
  (h4 : -35 * m^8 * n^3 = 35 * m^6 * n^4) : 
    n = real.cbrt 49 :=
  sorry

end value_of_n_l430_430020


namespace sum_of_distances_l430_430794

structure Point := (x : ℝ) (y : ℝ)

def distance (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 10, y := 2 }
def C : Point := { x := 5, y := 4 }
def P : Point := { x := 3, y := 1 }

theorem sum_of_distances :
  let AP := distance A P,
      BP := distance B P,
      CP := distance C P
  in AP + BP + CP = 15 + 0 * real.sqrt 1 →
  15 + 0 + 1 = 16 :=
by
  intros AP BP CP h
  calc
    AP + BP + CP = 15 + 0 * real.sqrt 1 : h
    ... = 15 + 0 + 1 : by simp
    ... = 16 : by simp

end sum_of_distances_l430_430794


namespace find_x_l430_430857

theorem find_x (x y : ℚ) (h1 : 3 * x - 2 * y = 7) (h2 : x + 3 * y = 8) : x = 37 / 11 := by
  sorry

end find_x_l430_430857


namespace count_triangles_in_figure_l430_430854

theorem count_triangles_in_figure : 
  (number_of_triangles_in_figure figure = 16) :=
  sorry

-- Definitions for the figure and number_of_triangles_in_figure
def figure : Type := Sorry -- Definition of the specific figure
def number_of_triangles_in_figure (f : Type) : Nat := Sorry -- Function to calculate number of triangles


end count_triangles_in_figure_l430_430854


namespace unique_B_l430_430897

-- Define bounded subsets of ℝ
variable {A H : Set ℝ} [Bounded A] [Bounded H]

-- Define the condition of pairwise distinct sums
def pairwise_distinct_sums (A B : Set ℝ) : Prop :=
∀ a1 a2 ∈ A, ∀ b1 b2 ∈ B, (a1 + b1 = a2 + b2) → (a1 = a2 ∧ b1 = b2)

-- Define the condition H = A + B
def add_sets_eq (H A B : Set ℝ) : Prop :=
∀ h ∈ H, ∃ (a ∈ A) (b ∈ B), h = a + b

-- Lean statement to prove that there exists at most one such set B
theorem unique_B (A H : Set ℝ) [Bounded A] [Bounded H] :
  (∃! B : Set ℝ, pairwise_distinct_sums A B ∧ add_sets_eq H A B) :=
sorry

end unique_B_l430_430897


namespace isosceles_triangle_perimeter_l430_430400

/-- Given an isosceles triangle with one side length of 3 cm and another side length of 5 cm,
    its perimeter is either 11 cm or 13 cm. -/
theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 5) : 
  (∃ c : ℝ, (c = 3 ∨ c = 5) ∧ (2 * a + b = 11 ∨ 2 * b + a = 13)) :=
by
  sorry

end isosceles_triangle_perimeter_l430_430400


namespace distance_to_focus_l430_430919

noncomputable def parabola_focus_distance (x y : ℝ) : Prop :=
  x = (-3 + 5) ∧ y^2 = 4 * x

theorem distance_to_focus (x y : ℝ) (h : parabola_focus_distance x y) :
  ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist (x, y) F = 6 :=
by
  obtain ⟨hx, hy⟩ := h
  use (1, 0)
  simp only [hx, Real.dist]
  have : y^2 = 4 * (2 : ℝ), by { exact hy }
  sorry

end distance_to_focus_l430_430919


namespace sqrt_mul_sqrt_eq_six_l430_430284

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l430_430284


namespace inverse_ratio_l430_430428

noncomputable def g (x : ℝ) : ℝ := (3 * x - 2) / (x + 4)
noncomputable def g_inv (x : ℝ) : ℝ := (4 * x + 2) / (3 - x)

theorem inverse_ratio : (a : ℝ) * (c : ℝ) * (g_inv a) = (g_inv a, g_inv c) 
by
  sorry

end inverse_ratio_l430_430428


namespace student_groups_l430_430540

theorem student_groups (N : ℕ) :
  (∃ (n : ℕ), n = 13 ∧ ∃ (m : ℕ), m ∈ {12, 14} ∧ N = 4 * 13 + 2 * m) → (N = 76 ∨ N = 80) :=
by
  intro h
  obtain ⟨n, hn, m, hm, hN⟩ := h
  rw [hn, hN]
  cases hm with h12 h14
  case inl =>
    simp [h12]
  case inr =>
    simp [h14]
  sorry

end student_groups_l430_430540


namespace find_rings_l430_430908

-- Define n as an integer with n ≥ 1
def n : ℕ := n
axiom h_n : n ≥ 1

-- Define the rings F_2 and F_4
def F_2 : Type := ℤ/2
def F_4 : Type := add_monoid_with_one.one.set.univ
-- This may need further precise definitions, the imports should be extended accordingly

-- Hypothesis: For every non-zero element x in A, x^(2^n + 1) = 1
noncomputable def satisfies_condition (A : Type) [ring A] [nontrivial A] : Prop :=
∀ x : A, x ≠ 0 → x^(2^n + 1) = (1 : A)

-- Statement of the main theorem
theorem find_rings (A : Type) [ring A] [nontrivial A] (h : satisfies_condition A) : 
  A ≅ F_2 ∨ A ≅ F_4 :=
sorry

end find_rings_l430_430908


namespace total_students_l430_430569

noncomputable def totalStudentsOptions (groups totalGroups specificGroupCount specificGroupSize otherGroupSizes : ℕ) : Set ℕ :=
  if totalGroups = 6 ∧ specificGroupCount = 4 ∧ specificGroupSize = 13 ∧ (otherGroupSizes = 12 ∨ otherGroupSizes = 14) then
    {52 + 2 * otherGroupSizes}
  else
    ∅

theorem total_students :
  totalStudentsOptions 6 4 13 12 = {76} ∧ totalStudentsOptions 6 4 13 14 = {80} :=
by
  -- This is where the proof would go, but we're skipping it as per instructions
  sorry

end total_students_l430_430569


namespace factory_workers_l430_430960

-- Define parameters based on given conditions
def sewing_factory_x : ℤ := 1995
def shoe_factory_y : ℤ := 1575

-- Conditions based on the problem setup
def shoe_factory_of_sewing_factory := (15 * sewing_factory_x) / 19 = shoe_factory_y
def shoe_factory_plan_exceed := (3 * shoe_factory_y) / 7 < 1000
def sewing_factory_plan_exceed := (3 * sewing_factory_x) / 5 > 1000

-- Theorem stating the problem's assertion
theorem factory_workers (x y : ℤ) 
  (h1 : (15 * x) / 19 = y)
  (h2 : (4 * y) / 7 < 1000)
  (h3 : (3 * x) / 5 > 1000) : 
  x = 1995 ∧ y = 1575 :=
sorry

end factory_workers_l430_430960


namespace total_students_possible_l430_430557

theorem total_students_possible (A B : ℕ) :
  (4 * 13) + 2 * A = 76 ∨ (4 * 13) + 2 * B = 80 :=
by
  -- Let N be the total number of students
  let N := (4 * 13)
  -- Given that the number of students in the remaining 2 groups differs by no more than 1
  have h : A = 12 ∨ B = 14 := sorry
  -- Prove the possible values
  exact or.inl (N + 2 * 12 = 76) <|> or.inr (N + 2 * 14 = 80)

end total_students_possible_l430_430557


namespace base_5_representation_l430_430480

theorem base_5_representation (n : ℕ) (h : n = 84) : 
  ∃ (a b c : ℕ), 
  a < 5 ∧ b < 5 ∧ c < 5 ∧ 
  n = a * 5^2 + b * 5^1 + c * 5^0 ∧ 
  a = 3 ∧ b = 1 ∧ c = 4 :=
by 
  -- Placeholder for the proof
  sorry

end base_5_representation_l430_430480


namespace coefficient_of_x2_in_expansion_l430_430757

theorem coefficient_of_x2_in_expansion :
  ∃ (c : ℕ), ( (∑ n in finset.range 7, (nat.choose 6 n) * x^n) *
               (1 + x ^ (-2)) ).coeff 2 = 30 :=
sorry

end coefficient_of_x2_in_expansion_l430_430757


namespace number_of_same_family_functions_l430_430456

-- Define the relevant sets and functions
def f (x : ℤ) : ℤ := x * x

-- Define the specific range we are interested in
def range_set : Set ℤ := {1, 9}

-- Set of possible domain elements
def domain_elements : Set ℤ := {-1, 1, -3, 3}

-- Helper function to check if a function has the desired range
def has_range (f : ℤ → ℤ) (ran : Set ℤ) : Prop :=
  ∀ y ∈ ran, ∃ x, f x = y

-- Definition of a function from the same family with a given domain
def same_family (dom : Set ℤ) (ran : Set ℤ) : Prop :=
  ∀ x ∈ dom, f x ∈ ran ∧ has_range f ran

-- The main theorem stating the number of such functions
theorem number_of_same_family_functions : 
  ∃ n : ℕ, n = 9 ∧ 
  ∃ dom_sets : Set (Set ℤ), 
    (∀ dom ∈ dom_sets, same_family dom range_set) ∧
    dom_sets.nonempty ∧
    dom_sets.card = 9 :=
by
  sorry

end number_of_same_family_functions_l430_430456


namespace total_students_l430_430565

noncomputable def totalStudentsOptions (groups totalGroups specificGroupCount specificGroupSize otherGroupSizes : ℕ) : Set ℕ :=
  if totalGroups = 6 ∧ specificGroupCount = 4 ∧ specificGroupSize = 13 ∧ (otherGroupSizes = 12 ∨ otherGroupSizes = 14) then
    {52 + 2 * otherGroupSizes}
  else
    ∅

theorem total_students :
  totalStudentsOptions 6 4 13 12 = {76} ∧ totalStudentsOptions 6 4 13 14 = {80} :=
by
  -- This is where the proof would go, but we're skipping it as per instructions
  sorry

end total_students_l430_430565


namespace sqrt_mul_eq_6_l430_430165

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l430_430165


namespace total_students_l430_430594

theorem total_students (n_groups : ℕ) (students_in_group : ℕ → ℕ)
    (h1 : n_groups = 6)
    (h2 : ∃ n : ℕ, (students_in_group n = 13) ∧ (finset.filter (λ g, students_in_group g = 13) (finset.range n_groups)).card = 4)
    (h3 : ∀ i j, i < n_groups → j < n_groups → abs (students_in_group i - students_in_group j) ≤ 1) :
    (∃ N, N = 76 ∨ N = 80) :=
begin
    sorry
end

end total_students_l430_430594


namespace probability_one_side_triangle_is_decagon_side_l430_430793

theorem probability_one_side_triangle_is_decagon_side (V : Fin 10 → Type) (regular_decagon : Finset (Triangle Vertex)) :
  (probability (λ T, count_decagon_sides T = 1) regular_decagon) = 1 / 2 :=
sorry

end probability_one_side_triangle_is_decagon_side_l430_430793


namespace profit_percentage_correct_l430_430105

variable (wholesalePrice : ℝ) (retailPrice : ℝ) (discountRate : ℝ)

def percentageProfit (wholesalePrice retailPrice discountRate : ℝ) : ℝ :=
  let discountAmount := discountRate * retailPrice
  let sellingPrice := retailPrice - discountAmount
  let profit := sellingPrice - wholesalePrice
  (profit / wholesalePrice) * 100

theorem profit_percentage_correct :
  wholesalePrice = 90 → retailPrice = 120 → discountRate = 0.10 → 
  percentageProfit wholesalePrice retailPrice discountRate = 20 := by
  intros
  unfold percentageProfit
  rw [H, H_1, H_2]
  norm_num
  sorry

end profit_percentage_correct_l430_430105


namespace sqrt_multiplication_l430_430138

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l430_430138


namespace sqrt_mult_simplify_l430_430235

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l430_430235


namespace max_value_interval_eq_l430_430835

theorem max_value_interval_eq :
  ∀ (a : ℝ), (∀ x ∈ set.Icc a 2, (-x^2 - 2*x + 3) ≤ 15/4) → a = -1/2 :=
begin
  sorry
end

end max_value_interval_eq_l430_430835


namespace exam_total_items_l430_430977

variables (X E : ℕ)

-- Definitions based on conditions
def incorrect_answers_ella : ℕ := 4
def marion_score : ℕ := 24
def marion_more_than_half_ella : Prop := marion_score = E / 2 + 6

-- Main statement to prove
theorem exam_total_items (h1 : marion_score = 24)
                         (h2 : marion_more_than_half_ella) :
  X = E + incorrect_answers_ella :=
sorry

end exam_total_items_l430_430977


namespace remainder_when_four_times_number_minus_nine_divided_by_eight_l430_430675

theorem remainder_when_four_times_number_minus_nine_divided_by_eight
  (n : ℤ) (h : n % 8 = 3) : (4 * n - 9) % 8 = 3 := by
  sorry

end remainder_when_four_times_number_minus_nine_divided_by_eight_l430_430675


namespace sqrt_mul_eq_6_l430_430159

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l430_430159


namespace regression_line_estimate_l430_430412

theorem regression_line_estimate:
  (∀ (x y : ℝ), y = 1.23 * x + a ↔ a = 5 - 1.23 * 4) →
  ∃ (y : ℝ), y = 1.23 * 2 + 0.08 :=
by
  intro h
  use 2.54
  simp
  sorry

end regression_line_estimate_l430_430412


namespace sqrt_mult_l430_430190

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l430_430190


namespace percentage_difference_eq_l430_430043

theorem percentage_difference_eq (x : ℝ) : 
  (x / 100 * 150) - (20 / 100 * 250) = 43 ↔ x = 62 :=
by
  -- We'll set up the conditions described in the problem context
  have h1 : (20 / 100 * 250) = 50 := by sorry
  have h2 : (x / 100 * 150) - 50 = 43 := by
    rw [h1]
    assumption
  -- Now solve for x based on this simplified condition
  calc 
    (x / 100 * 150) - 50 = 43   : by assumption
    150x / 100 - 50 = 43       : by sorry
    150x / 100 = 93            : add 50 to both sides
    150x = 93 * 100            : multiply both sides by 100
    150x = 9300                : calculate
    x = 9300 / 150             : divide both sides by 150
    x = 62                     : calculate 

  -- The theorem holds
  exact eq_rfl

end percentage_difference_eq_l430_430043


namespace solve_eq1_solve_eq2_l430_430375

-- Define the condition and the statement to be proved for the first equation
theorem solve_eq1 (x : ℝ) : 9 * x^2 - 25 = 0 → x = 5 / 3 ∨ x = -5 / 3 :=
by sorry

-- Define the condition and the statement to be proved for the second equation
theorem solve_eq2 (x : ℝ) : (x + 1)^3 - 27 = 0 → x = 2 :=
by sorry

end solve_eq1_solve_eq2_l430_430375


namespace solution_correct_l430_430970

theorem solution_correct :
  (∃ x y z w : ℕ, x = 3 ∧ y = 5 ∧ z = 7 ∧ w = 9 ∧ (x + (y + 1) * z - w = 36)) :=
by
  -- Providing definitions for clarity
  let x := 3
  let y := 5
  let z := 7
  let w := 9
  use x, y, z, w
  split; sorry

end solution_correct_l430_430970


namespace carolyn_lace_costs_l430_430748

/-- Problem statement translated to Lean 4. --/
theorem carolyn_lace_costs:
  (∀ (cuffs : List ℕ) (hems : List ℕ) (neckline_ruffles : List (ℕ × ℕ))
       (price_A price_B price_C : ℕ) (discount_A discount_B tax_rate: ℕ),
       cuffs = [50, 60, 70] →
       hems = [300, 350, 400] →
       neckline_ruffles = [(5, 20), (6, 25), (7, 30)] →
       price_A = 6 →
       price_B = 8 →
       price_C = 12 →
       discount_A = 10 →
       discount_B = 10 →
       tax_rate = 5 →
       let lace_A_length := List.sum cuffs / 100 + List.sum hems / 10 in
       let lace_B_length := (List.sum hems) / 300 in
       let lace_C_length := List.sum (neckline_ruffles.map (fun r => r.1 * r.2)) / 100 in
       let cost_A := lace_A_length * price_A in
       let cost_B := lace_B_length * price_B in
       let cost_C := lace_C_length * price_C in
       let discounted_cost_A := cost_A - cost_A * discount_A / 100 in
       let discounted_cost_B := cost_B - cost_B * discount_B / 100 in
       let total_cost_before_tax := discounted_cost_A + discounted_cost_B + cost_C in
       let tax := total_cost_before_tax * tax_rate / 100 in
       let total_cost_incl_tax := total_cost_before_tax + tax in
       discounted_cost_A = 66.42 ∧
       discounted_cost_B = 25.20 ∧
       cost_C = 55.20 ∧
       total_cost_incl_tax.round = 154.16) → 
     True :=
by { sorry }

end carolyn_lace_costs_l430_430748


namespace cos_shift_l430_430650

theorem cos_shift {x : ℝ} :
  (cos (2 * (x + π / 6))) = (cos (2 * x + π / 3)) :=
by
  sorry

end cos_shift_l430_430650


namespace total_investment_sum_l430_430659

theorem total_investment_sum :
  let R : ℝ := 2200
  let T : ℝ := R - 0.1 * R
  let V : ℝ := T + 0.1 * T
  R + T + V = 6358 := by
  sorry

end total_investment_sum_l430_430659


namespace tan_double_angle_l430_430818

-- Given conditions
variables (α : ℝ)
hypothesis1 : 0 < α ∧ α < π
hypothesis2 : cos α + sin α = -1 / 5

-- The statement to prove
theorem tan_double_angle (h₁ : 0 < α ∧ α < π) (h₂ : cos α + sin α = -1 / 5) : 
  Real.tan (2 * α) = -24 / 7 :=
sorry

end tan_double_angle_l430_430818


namespace median_line_eq_l430_430465

theorem median_line_eq {A B C : Point} (A_coord : A = (2, 1)) (B_coord : B = (-2, 3)) (C_coord : C = (0, 1)) :
  ∃ k b, line_eq k b A_coord ∧ line_eq k b (midpoint B_coord C_coord) ∧ k = -1/3 ∧ b = 1/3 := 
sorry

end median_line_eq_l430_430465


namespace floor_a_l430_430391

def a : ℕ → ℝ
| 0       := 1994
| (n + 1) := a n ^ 2 / (a n + 1)

theorem floor_a (n : ℕ) (h : 0 ≤ n ∧ n ≤ 998) : 1994 - (n : ℝ) = ∥a n∥ :=
by
  sorry

end floor_a_l430_430391


namespace find_a_l430_430620

theorem find_a (a : ℝ) :
  (∀ b, b = (ax + b * x ^ (1/2))^3 → coeff b x 3 = 20) →
  a = real.cbrt 20 :=
sorry

end find_a_l430_430620


namespace magic_8_ball_probability_l430_430898

noncomputable def probability_exactly_4_positive (p q : ℚ) (n k : ℕ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * (q ^ (n - k))

open Probability

theorem magic_8_ball_probability :
  probability_exactly_4_positive (3 / 7) (4 / 7) 7 4 = 181440 / 823543 :=
by
  sorry

end magic_8_ball_probability_l430_430898


namespace largest_intersection_x_value_l430_430627

-- Define the polynomial and the line equation
def poly (x : ℝ) (a b : ℝ) : ℝ := x^6 - 8*x^5 + 24*x^4 - 37*x^3 + a*x^2 + b*x - 6
def line (x d : ℝ) : ℝ := d*x + 2

-- Define the condition for intersection
def intersect_condition (a b d x : ℝ) : Prop := 
  poly x a b = line x d

-- Given conditions
variables (a b d : ℝ)
hypothesis h1 : a = d
hypothesis h2 : b - a = 28

-- Define the proof problem
theorem largest_intersection_x_value : ∃ x, intersect_condition a b d x ∧ 
  (∀ y, intersect_condition a b d y → y ≤ x) ∧ x = 5 :=
by
  sorry

end largest_intersection_x_value_l430_430627


namespace birds_on_fence_l430_430647

theorem birds_on_fence :
  let i := 12           -- initial birds
  let added1 := 8       -- birds that land first
  let T := i + added1   -- total first stage birds
  
  let fly_away1 := 5
  let join1 := 3
  let W := T - fly_away1 + join1   -- birds after some fly away, others join
  
  let D := W * 2       -- birds doubles
  
  let fly_away2 := D * 0.25  -- 25% fly away
  let D_after_fly_away := D - fly_away2
  
  let return_birds := 2        -- 2.5 birds return, rounded down to 2
  let final_birds := D_after_fly_away + return_birds
  
  final_birds = 29 := 
by {
  sorry
}

end birds_on_fence_l430_430647


namespace trig_identity_l430_430060

variable (α : ℝ)

theorem trig_identity :
  cos (2 * α) - cos (3 * α) - cos (4 * α) + cos (5 * α) = 
  -4 * sin (α / 2) * sin α * cos (7 * α / 2) := 
sorry

end trig_identity_l430_430060


namespace count_integers_between_sqrt_10_and_sqrt_50_l430_430443

theorem count_integers_between_sqrt_10_and_sqrt_50 :
  ∃ n : ℕ, n = 4 ∧ (∀ (k : ℕ), 4 ≤ k ∧ k ≤ 7 → (sqrt 10 : ℝ) < k ∧ k < (sqrt 50 : ℝ)) :=
by
  sorry

end count_integers_between_sqrt_10_and_sqrt_50_l430_430443


namespace function_even_function_monotonic_intervals_smallest_period_l430_430389

noncomputable def my_function (a : ℝ) (h : a ≠ 0) (x : ℝ) : ℝ :=
  2 + a * Real.cos x

theorem function_even (a : ℝ) (h : a ≠ 0) :
  ∀ x, my_function a h (-x) = my_function a h x :=
by
  intro x
  -- sorry: Proof will be provided here later
  sorry

theorem function_monotonic_intervals (a : ℝ) (h : a ≠ 0) :
  (a > 0 → ∀ k : ℤ, (∀ x ∈ (Set.Icc (2 * k * Real.pi - Real.pi) (2 * k * Real.pi)), my_function a h x) ∧
    (∀ x ∈ (Set.Icc (2 * k * Real.pi) (2 * k * Real.pi + Real.pi)), my_function a h x)) ∧
  (a < 0 → ∀ k : ℤ, (∀ x ∈ (Set.Icc (2 * k * Real.pi) (2 * k * Real.pi + Real.pi)), my_function a h x) ∧
    (∀ x ∈ (Set.Icc (2 * k * Real.pi - Real.pi) (2 * k * Real.pi)), my_function a h x)) :=
by
  intro x
  -- sorry: Proof will be provided here later
  sorry

theorem smallest_period (a : ℝ) (h : a ≠ 0) :
  is_periodic (my_function a h) (2 * Real.pi) ∧ ¬(∃ T, T > 0 ∧ T < 2 * Real.pi ∧ is_periodic (my_function a h) T) :=
by
  -- sorry: Proof will be provided here later
  sorry

end function_even_function_monotonic_intervals_smallest_period_l430_430389


namespace num_possible_bases_l430_430003

theorem num_possible_bases (b : ℕ) (h1 : b ≥ 2) (h2 : b^3 ≤ 256) (h3 : 256 < b^4) : ∃ n : ℕ, n = 2 :=
by
  sorry

end num_possible_bases_l430_430003


namespace difference_between_extremes_l430_430062

/-- Define the structure of a 3-digit integer and its digits. -/
structure ThreeDigitInteger where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  val : ℕ := 100 * hundreds + 10 * tens + units

/-- Define the problem conditions. -/
def satisfiesConditions (x : ThreeDigitInteger) : Prop :=
  x.hundreds > 0 ∧
  4 * x.hundreds = 2 * x.tens ∧
  2 * x.tens = x.units

/-- Given conditions prove the difference between the two greatest possible values of x is 124. -/
theorem difference_between_extremes :
  ∃ (x₁ x₂ : ThreeDigitInteger), 
    satisfiesConditions x₁ ∧ satisfiesConditions x₂ ∧
    (x₁.val = 248 ∧ x₂.val = 124 ∧ (x₁.val - x₂.val = 124)) :=
sorry

end difference_between_extremes_l430_430062


namespace geometric_sequence_sum_l430_430470

variable {a r : ℝ}

-- Conditions
def condition1 : a * (1 + r) = 10 := sorry
def condition2 : a * (1 - r^6) / (1 - r) = 250 := sorry

-- Question to be proved
def sum_of_first_five_terms : a * (1 + r + r^2 + r^3 + r^4) = 436 := sorry

theorem geometric_sequence_sum :
  condition1 ∧ condition2 → sum_of_first_five_terms := sorry

end geometric_sequence_sum_l430_430470


namespace num_even_multiples_of_four_perfect_squares_lt_5000_l430_430447

theorem num_even_multiples_of_four_perfect_squares_lt_5000 : 
  ∃ (k : ℕ), k = 17 ∧ ∀ (n : ℕ), (0 < n ∧ 16 * n^2 < 5000) ↔ (1 ≤ n ∧ n ≤ 17) :=
by
  sorry

end num_even_multiples_of_four_perfect_squares_lt_5000_l430_430447


namespace find_PQ_l430_430477

variable (P Q R : Type) [right_triangle P Q R] (PR : ℝ) (tanP : ℝ)

noncomputable def PQ_length (tanP : ℝ) (PR : ℝ) : ℝ := 
  let QR := tanP * PR
  real.sqrt (PR^2 + QR^2)

theorem find_PQ (h1 : ∠ PQR = 90) (h2 : pr = 12) (h3 : tanP = 3/4) : PQ_length 3/4 12 = 15 :=
by
  let QR := 3/4 * 12 
  have hQR : QR = 9 := by sorry
  have hPQ : PQ_length 3/4 12 = real.sqrt (12^2 + 9^2) := by sorry
  rw [hPQ]
  norm_num
  exact rfl

end find_PQ_l430_430477


namespace sqrt_mul_l430_430170

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l430_430170


namespace sqrt_mul_eq_l430_430305

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l430_430305


namespace part1_part2_l430_430392

-- Conditions and the equation of the circle
def circleCenterLine (a : ℝ) : Prop := ∃ y, y = a + 2
def circleRadius : ℝ := 2
def pointOnCircle (A : ℝ × ℝ) (a : ℝ) : Prop := (A.1 - a)^2 + (A.2 - (a + 2))^2 = circleRadius^2
def tangentToYAxis (a : ℝ) : Prop := abs a = circleRadius

-- Problem 1: Proving the equation of the circle C
def circleEq (x y a : ℝ) : Prop := (x - a)^2 + (y - (a + 2))^2 = circleRadius^2

theorem part1 (a : ℝ) (h : abs a = circleRadius) (h1 : pointOnCircle (2, 2) a) 
    (h2 : circleCenterLine a) : circleEq 2 0 2 := 
sorry

-- Conditions and the properties for Problem 2
def distSquared (P Q : ℝ × ℝ) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2
def QCondition (Q : ℝ × ℝ) : Prop := 
  distSquared Q (1, 3) - distSquared Q (1, 1) = 32
def onCircle (Q : ℝ × ℝ) (a : ℝ) : Prop := (Q.1 - a)^2 + (Q.2 - (a + 2))^2 = circleRadius^2

-- Problem 2: Proving the range of the abscissa a
theorem part2 (Q : ℝ × ℝ) (a : ℝ) 
    (hQ : QCondition Q) (hCircle : onCircle Q a) : 
    -3 ≤ a ∧ a ≤ 1 := 
sorry

end part1_part2_l430_430392


namespace distance_between_stations_l430_430037

theorem distance_between_stations :
  ∀ (x b: ℕ), 
    (x / 20 = (x + 65) / 25) → 
    (b = x + (x + 65)) → 
    b = 585 :=
by {
  assume x b h1 h2,
  sorry
}

end distance_between_stations_l430_430037


namespace coefficient_of_x10_in_expansion_l430_430758

theorem coefficient_of_x10_in_expansion :
  coeff (expand_polynomial ((x + 2)^10 * (x^2 - 1))) 10 = 179 := sorry

end coefficient_of_x10_in_expansion_l430_430758


namespace find_triangle_area_l430_430342

def area_of_triangle_perimeter_inradius (c : ℝ) (r : ℝ) (a b : ℝ) (p : ℝ) : Prop :=
  c = 12.5 ∧ r = 3.5 ∧ 40 ≤ p ∧ p ≤ 45 ∧ p = a + b + c ∧ a ≠ ⌊a⌋ ∧ b ≠ ⌊b⌋

theorem find_triangle_area :
  ∃ (a b p : ℝ),
    area_of_triangle_perimeter_inradius 12.5 3.5 a b p ∧
    (1 / 2 * 3.5 * p = 70) := 
by 
    use [14.2, 13.3, 40]
    split
    { -- prove the conditions are satisfied
      split; try {assumption}
      split; try {norm_num}
      split; try {linarith}
      all_goals {norm_num}
      split
      { -- prove a ≠ ⌊a⌋
        norm_num
      }
      { -- prove b ≠ ⌊b⌋
        norm_num
      }
    }
    { -- prove the area is correct
      sorry 
    }

end find_triangle_area_l430_430342


namespace pete_should_leave_by_0730_l430_430933

def walking_time : ℕ := 10
def train_time : ℕ := 80
def latest_arrival_time : String := "0900"
def departure_time : String := "0730"

theorem pete_should_leave_by_0730 :
  (latest_arrival_time = "0900" → walking_time = 10 ∧ train_time = 80 → departure_time = "0730") := by
  sorry

end pete_should_leave_by_0730_l430_430933


namespace sqrt_mul_simplify_l430_430270

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l430_430270


namespace total_students_possible_l430_430562

theorem total_students_possible (A B : ℕ) :
  (4 * 13) + 2 * A = 76 ∨ (4 * 13) + 2 * B = 80 :=
by
  -- Let N be the total number of students
  let N := (4 * 13)
  -- Given that the number of students in the remaining 2 groups differs by no more than 1
  have h : A = 12 ∨ B = 14 := sorry
  -- Prove the possible values
  exact or.inl (N + 2 * 12 = 76) <|> or.inr (N + 2 * 14 = 80)

end total_students_possible_l430_430562


namespace ellipse_properties_l430_430410

structure Point where
  x : ℝ
  y : ℝ

def passes_through (p : Point) (m n : ℝ) : Prop :=
  m * p.x^2 + n * p.y^2 = 1

def standard_equation (m n : ℝ) : Prop :=
  m = 1 / 9 ∧ n = 1 / 16

def vertices : Set Point :=
  { Point.mk 3 0, Point.mk (-3) 0, Point.mk 0 4, Point.mk 0 (-4) }

def eccentricity (m n : ℝ) : ℝ :=
  let a := sqrt (1 / n)
  let b := sqrt (1 / m)
  (sqrt (a^2 - b^2)) / a

theorem ellipse_properties (A B : Point) (m n : ℝ) :
  A = Point.mk 2 (- (4 * sqrt 5) / 3) →
  B = Point.mk (-1) ((8 * sqrt 2) / 3) →
  passes_through A m n →
  passes_through B m n →
  standard_equation m n ∧
  vertices = { Point.mk 3 0, Point.mk (-3) 0, Point.mk 0 4, Point.mk 0 (-4) } ∧
  eccentricity m n = sqrt 7 / 4 :=
by
  intros
  sorry

end ellipse_properties_l430_430410


namespace euler_characteristic_convex_polyhedron_l430_430070

-- Define the context of convex polyhedron with vertices (V), edges (E), and faces (F)
structure ConvexPolyhedron :=
  (V : ℕ) -- number of vertices
  (E : ℕ) -- number of edges
  (F : ℕ) -- number of faces
  (convex : Prop) -- property stating the polyhedron is convex

-- Euler characteristic theorem for convex polyhedra
theorem euler_characteristic_convex_polyhedron (P : ConvexPolyhedron) (h : P.convex) : P.V - P.E + P.F = 2 :=
sorry

end euler_characteristic_convex_polyhedron_l430_430070


namespace equation1_solution_equation2_solution_equation3_solution_l430_430601

theorem equation1_solution :
  ∀ x : ℝ, x^2 - 2 * x - 99 = 0 ↔ x = 11 ∨ x = -9 :=
by
  sorry

theorem equation2_solution :
  ∀ x : ℝ, x^2 + 5 * x = 7 ↔ x = (-5 - Real.sqrt 53) / 2 ∨ x = (-5 + Real.sqrt 53) / 2 :=
by
  sorry

theorem equation3_solution :
  ∀ x : ℝ, 4 * x * (2 * x + 1) = 3 * (2 * x + 1) ↔ x = -1/2 ∨ x = 3/4 :=
by
  sorry

end equation1_solution_equation2_solution_equation3_solution_l430_430601


namespace find_a_and_d_l430_430743

noncomputable def amplitude_and_shift (a b c d : ℝ) : Prop :=
  (∀ x : ℝ, (a * cos (b * x + c) + d ≤ 5 ∧ a * cos (b * x + c) + d ≥ 1))

theorem find_a_and_d (a b c d : ℝ) (h : amplitude_and_shift a b c d) : 
  a = 2 ∧ d = 3 :=
by
  sorry

end find_a_and_d_l430_430743


namespace correct_option_D_l430_430678

theorem correct_option_D (y : ℝ): 
  3 * y^2 - 2 * y^2 = y^2 :=
by
  sorry

end correct_option_D_l430_430678


namespace find_a_plus_b_l430_430027

def cubic_function (a b : ℝ) (x : ℝ) := x^3 - x^2 - a * x + b

def tangent_line (x : ℝ) := 2 * x + 1

theorem find_a_plus_b (a b : ℝ) 
  (h1 : tangent_line 0 = 1)
  (h2 : cubic_function a b 0 = 1)
  (h3 : deriv (cubic_function a b) 0 = 2) :
  a + b = -1 :=
by
  sorry

end find_a_plus_b_l430_430027


namespace larger_square_side_length_l430_430110

theorem larger_square_side_length (shaded_area unshaded_area : ℝ) (h_shaded : shaded_area = 18) (h_unshaded : unshaded_area = 18) : 
  let total_area := shaded_area + unshaded_area in 
  let side_length := Real.sqrt total_area in
  side_length = 6 :=
by
  sorry

end larger_square_side_length_l430_430110


namespace abs_ineq_l430_430608

theorem abs_ineq (x : ℝ) (h : |x + 1| > 3) : x < -4 ∨ x > 2 :=
  sorry

end abs_ineq_l430_430608


namespace BR_parallel_AC_l430_430399

open EuclideanGeometry

variables {A B C M P Q R O : Point}

-- Given conditions
axiom (h1 : acute_triangle A B C)
axiom (h2 : angle_bisector_intersects_circumcircle_of_ABC A B C O M)
axiom (h3 : AB > BC)
axiom (h4 : circle_with_diameter BM)
axiom (h5 : angle_bisectors_of_angles_intersects_gamma A O B C P Q)
axiom (h6 : BR = MR)
axiom (h7 : R_on_extension_of_QP Q P R)

-- Conclusion
theorem BR_parallel_AC : BR ∥ AC := 
sorry

end BR_parallel_AC_l430_430399


namespace total_students_possible_l430_430556

theorem total_students_possible (A B : ℕ) :
  (4 * 13) + 2 * A = 76 ∨ (4 * 13) + 2 * B = 80 :=
by
  -- Let N be the total number of students
  let N := (4 * 13)
  -- Given that the number of students in the remaining 2 groups differs by no more than 1
  have h : A = 12 ∨ B = 14 := sorry
  -- Prove the possible values
  exact or.inl (N + 2 * 12 = 76) <|> or.inr (N + 2 * 14 = 80)

end total_students_possible_l430_430556


namespace infinite_series_sum_l430_430349

theorem infinite_series_sum :
  (∑' k : ℕ, k / 2^k) = 2 :=
begin
  sorry
end

end infinite_series_sum_l430_430349


namespace rhombus_independent_variable_l430_430413

theorem rhombus_independent_variable (a C : ℝ) (h : C = 4 * a) : "a" is_independent_variable :=
sorry

end rhombus_independent_variable_l430_430413


namespace num_correct_propositions_l430_430395

theorem num_correct_propositions 
  (l α β : Type) 
  (is_parallel : l → α → Prop) 
  (is_perpendicular : l → α → Prop) 
  (is_plane_parallel : α → β → Prop)
  (is_plane_perpendicular : α → β → Prop) :
  (¬ ∀ (l_parallel_α l_parallel_β : is_parallel l α ∧ is_parallel l β), is_plane_parallel α β) ∧
  (∀ (l_perp_α l_parallel_β : is_perpendicular l α ∧ is_parallel l β), is_plane_perpendicular α β) ∧
  (∀ (l_perp_α l_perp_β : is_perpendicular l α ∧ is_perpendicular l β), is_plane_parallel α β) ∧
  (¬ ∀ (l_perp_α α_perp_β : is_perpendicular l α ∧ is_plane_perpendicular α β), is_parallel l β) :=
  sorry

end num_correct_propositions_l430_430395


namespace find_A_omega_range_of_f_l430_430834

-- Definition for question (1)
def max_value_f (A ω : ℝ) : Prop :=
  ∀ x : ℝ, A > 0 ∧ ω > 0 ∧ A * Math.sin (ω * x + Real.pi / 3) <= 2

-- Definition for question (1)
def period_f (ω : ℝ) : Prop :=
  Real.pi = 2 * Real.pi / ω

-- Proof for question (1)
theorem find_A_omega (A ω : ℝ) (h_max : max_value_f A ω) (h_period : period_f ω) :
  A = 2 ∧ ω = 2 := by
  sorry

-- Definition for question (2)
def deriv_interval (x : ℝ) : ℝ := 2 * x + Real.pi / 3

-- Proof for question (2)
theorem range_of_f (x : ℝ) (H : 0 <= x ∧ x <= Real.pi / 2) :
  let f := 2 * Math.sin (deriv_interval x)
  -Real.sqrt 3 <= f ∧ f <= 2 := by
  sorry

end find_A_omega_range_of_f_l430_430834


namespace angle_in_triangle_l430_430892

theorem angle_in_triangle
  (A B C : Type)
  (a b c : ℝ)
  (angle_ABC : ℝ)
  (h1 : a = 15)
  (h2 : angle_ABC = π/3 ∨ angle_ABC = 2 * π / 3)
  : angle_ABC = π/3 ∨ angle_ABC = 2 * π / 3 := 
  sorry

end angle_in_triangle_l430_430892


namespace mb_product_l430_430623

theorem mb_product :
  ∀ (m b : ℝ),
    (∀ (y x : ℝ), (y = m * x + b) → 
                  ((x = 0 ∧ y = -2) ∨ (x = 1 ∧ y = 1)) →
                  (b = -2) →
                  (m = 3)) →
    m * b = -6 :=
by
  assume m b h
  have b_eq : b = -2 := by sorry
  have m_eq : m = 3 := by sorry
  rw [m_eq, b_eq]
  exact mul_neg_eq_neg_mul_symm 3 2

end mb_product_l430_430623


namespace sqrt_mult_simplify_l430_430240

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l430_430240


namespace range_of_k_l430_430832

def quadratic (a b c x : ℝ) : ℝ := a * x ^ 2 + b * x + c

def f (k x : ℝ) : ℝ := (x ^ 2 + x + 1) / (quadratic k k 1 x)

theorem range_of_k (k : ℝ) : (∀ x : ℝ, quadratic k k 1 x ≠ 0) ↔ (0 ≤ k ∧ k < 4) :=
by
  sorry

end range_of_k_l430_430832


namespace quadratic_inequality_always_holds_l430_430054

theorem quadratic_inequality_always_holds (k : ℝ) (h : ∀ x : ℝ, (x^2 - k*x + 1) > 0) : -2 < k ∧ k < 2 :=
  sorry

end quadratic_inequality_always_holds_l430_430054


namespace length_of_MN_l430_430462

variables {A B C M N Q : Point}

-- Given conditions
def midpoint (M : Point) (B C : segment) : Prop :=
  dist M B = dist M C

def bisects_angle (A N B C : Point) : Prop :=
  ∠BAC = 2 * ∠BAN

def perp (B N A : Point) : Prop :=
  ∠BNA = π / 2

def on_line (Q : Point) (A C : segment) : Prop :=
  ∃ k : ℝ, Q = A + k * (C - A)

noncomputable def length (MN : segment) : ℝ :=
  dist M N

-- Triangle ABC with the setup and conditions
theorem length_of_MN 
  (h_midpoint : midpoint M B C)
  (h_bisect : bisects_angle A N B C)
  (h_perp : perp B N A)
  (h_on_line : on_line Q A C)
  (h_AB : dist A B = 20)
  (h_AC : dist A C = 25)
  (h_CQ : dist C Q = 5) :
  length (M, N) = 5 / 2 :=
sorry

end length_of_MN_l430_430462


namespace sqrt_mul_eq_6_l430_430168

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l430_430168


namespace sqrt_mult_l430_430197

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l430_430197


namespace sqrt_mul_simplify_l430_430269

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l430_430269


namespace inclination_angle_of_max_area_circle_l430_430823

-- Define the equation of the circle and the conditions
def circle_equation : ℝ → ℝ → ℝ → ℝ :=
  λ x y k, x^2 + y^2 + k*x + 2*y + k^2

-- Define the equation of the line
def line_equation : ℝ → ℝ → ℝ :=
  λ k x, (k + 1) * x + 2

-- State the main theorem/proof problem
theorem inclination_angle_of_max_area_circle :
  (∀ k : ℝ, circle_equation 0 0 k = 0 → ∃ α : ℝ, 
    line_equation k 0 = 2 → α = Real.pi / 4) := 
by
  intro k h_circle_eqn h_line_eqn
  sorry

end inclination_angle_of_max_area_circle_l430_430823


namespace original_price_of_shoes_l430_430753

theorem original_price_of_shoes (P : ℝ) (h1 : 2 * 0.60 * P + 0.80 * 100 = 140) : P = 50 :=
by
  sorry

end original_price_of_shoes_l430_430753


namespace probability_two_speak_french_distribution_of_X_l430_430085

theorem probability_two_speak_french (h1 : 7 = 2 + 2 + 3)
  (total_selections : finset (fin 7) → finset (fin 3))
  (speaks_french : finset (fin 7) := {0, 1, 2, 3, 4}) -- student IDs for those who can speak French
  (combinations : finset (fin 7) := {0, 1, 2, 3, 4, 5, 6})
  (h_combinations : finset.card combinations = 7)
  (events : finset (fin 3) → ℚ)
  (h_events : events (total_selections speaks_french) = 4 / 7)
  : events {x : fin 3 | x ∈ total_selections speaks_french} = 4 / 7 :=
by sorry

theorem distribution_of_X
  (h1 : 7 = 2 + 2 + 3)
  (total_selections : finset (fin 7) → finset (fin 3))
  (speaks_both : finset (fin 7) := {2, 3, 4}) -- student IDs for those who can speak both French and English
  (combinations : finset (fin 7) := {0, 1, 2, 3, 4, 5, 6})
  (h_combinations : finset.card combinations = 7)
  (X : ℕ → ℚ)
  (h_dist0 : X 0 = 4 / 35)
  (h_dist1 : X 1 = 18 / 35)
  (h_dist2 : X 2 = 12 / 35)
  (h_dist3 : X 3 = 1 / 35)
  : (X 0 = 4 / 35) ∧ (X 1 = 18 / 35) ∧ (X 2 = 12 / 35) ∧ (X 3 = 1 / 35) :=
by sorry

end probability_two_speak_french_distribution_of_X_l430_430085


namespace A_winning_strategy_n0_ge_8_B_winning_strategy_n0_le_5_no_winning_strategy_n0_6_or_7_l430_430610

-- Define the game state and the rules of the game
structure GameState where
  n0 : ℕ    -- positive integer that starts the game
  n : ℕ     -- current number in the sequence
  turn : Bool  -- true for A's turn, false for B's turn

-- A picking rule
def A_pick (state : GameState) (n_next : ℕ) : Prop :=
  state.n ≤ n_next ∧ n_next ≤ state.n^2

-- B picking rule
def B_pick (state : GameState) (n_next : ℕ) : Prop :=
  ∃ k : ℕ, (∃ p : ℕ, Prime p) ∧ state.n / n_next = p^k

-- Winning conditions
def A_wins (state : GameState) : Prop :=
  state.n = 1990

def B_wins (state : GameState) : Prop :=
  state.n = 1

-- Proving A has a winning strategy for n0 >= 8
theorem A_winning_strategy_n0_ge_8 (n0 : ℕ) (h : n0 ≥ 8) : ∃ seq : ℕ → ℕ, A_wins {| n0 := n0, n := 1990, turn := true |} :=
sorry

-- Proving B has a winning strategy for n0 <= 5
theorem B_winning_strategy_n0_le_5 (n0 : ℕ) (h : n0 ≤ 5) : ∃ seq : ℕ → ℕ, B_wins {| n0 := n0, n := 1, turn := true |} :=
sorry

-- Proving no one has a winning strategy for n0 = 6 or n0 = 7
theorem no_winning_strategy_n0_6_or_7 (n0 : ℕ) (h : n0 = 6 ∨ n0 = 7) :
  ¬ (∃ seq : ℕ → ℕ, A_wins {| n0 := n0, n := 1990, turn := true |}) ∧ ¬ (∃ seq : ℕ → ℕ, B_wins {| n0 := n0, n := 1, turn := true |}) :=
sorry

end A_winning_strategy_n0_ge_8_B_winning_strategy_n0_le_5_no_winning_strategy_n0_6_or_7_l430_430610


namespace difference_of_squares_l430_430865

theorem difference_of_squares (a b c : ℤ) (h₁ : a < b) (h₂ : b < c) (h₃ : a % 2 = 0) (h₄ : b % 2 = 0) (h₅ : c % 2 = 0) (h₆ : a + b + c = 1992) :
  c^2 - a^2 = 5312 :=
by
  sorry

end difference_of_squares_l430_430865


namespace bouquet_combinations_l430_430108

theorem bouquet_combinations : ∃ n : ℕ, n = 6 ∧ (∀ r c : ℕ, 4 * r + 3 * c = 60 → {r | r ≤ 15 ∧ (60 - 4 * r) % 3 = 0}.finite ∧ {r | r ≤ 15 ∧ (60 - 4 * r) % 3 = 0}.card = n) :=
by
  have : (∃ r c : ℕ, 4 * r + 3 * c = 60) := sorry,
  have r_bound : ∀ r, ∃ c, 4 * r + 3 * c = 60 → r ≤ 15 := sorry,
  have r_div_3 : ∀ r, 4 * r ≤ 60 ∧ (60 - 4 * r) % 3 = 0 := sorry,
  have r_values : ∀ r, r ∈ {0, 3, 6, 9, 12, 15} := sorry,
  have valid_count : {r | r ≤ 15 ∧ (60 - 4 * r) % 3 = 0}.card = 6 := sorry,
  use 6,
  split,
    refl,
  intros r c h,
  split,
    apply finite_set_of_int,
  exact valid_count

end bouquet_combinations_l430_430108


namespace internship_choices_l430_430788

theorem internship_choices :
  let choices := 3
  let students := 4
  (choices ^ students) = 81 := 
by
  intros
  calc
    3 ^ 4 = 81 : by norm_num

end internship_choices_l430_430788


namespace sqrt_mult_eq_six_l430_430326

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l430_430326


namespace find_ab_l430_430408

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 35) : a * b = 6 :=
by
  sorry

end find_ab_l430_430408


namespace radio_lowest_price_rank_l430_430131

-- Definitions based on the conditions
def total_items : ℕ := 38
def radio_highest_rank : ℕ := 16

-- The theorem statement
theorem radio_lowest_price_rank : (total_items - (radio_highest_rank - 1)) = 24 := by
  sorry

end radio_lowest_price_rank_l430_430131


namespace all_integers_of_arith_or_geom_mean_l430_430694

variable {n : ℕ} (S : Finset ℕ) (hS : ∀ (i j : ℕ) (hi : i ∈ S) (hj : j ∈ S), i ≠ j → (i + j) % 2 = 0 ∨ ∃ k : ℕ, k*k = i*j)

theorem all_integers_of_arith_or_geom_mean (hS_nonempty : S.nonempty) (hm : ∀ a ∈ S, 0 < a) : ∀ a ∈ S, a ∈ ℕ := 
by
  sorry

end all_integers_of_arith_or_geom_mean_l430_430694


namespace slope_of_tangent_line_l430_430638

theorem slope_of_tangent_line : 
  let y := λ x : ℝ, (1/3) * x^3 - 2 in
  let dydx := λ x : ℝ, x^2 in
  let point := (1 : ℝ, - (5 / 3) : ℝ) in
  dydx (point.fst) = 1 :=
by sorry

end slope_of_tangent_line_l430_430638


namespace smallest_number_divisible_by_11_and_conditional_modulus_l430_430050

theorem smallest_number_divisible_by_11_and_conditional_modulus :
  ∃ n : ℕ, (n % 11 = 0) ∧ (n % 3 = 2) ∧ (n % 4 = 2) ∧ (n % 5 = 2) ∧ (n % 6 = 2) ∧ (n % 7 = 2) ∧ n = 2102 :=
by
  sorry

end smallest_number_divisible_by_11_and_conditional_modulus_l430_430050


namespace convex_polyhedron_Euler_formula_l430_430073

theorem convex_polyhedron_Euler_formula (V E F : ℕ) (h : ¬∃ (P : Polyhedron), 
  convex P ∧ V = P.vertices ∧ E = P.edges ∧ F = P.faces) : 
  V - E + F = 2 :=
sorry

end convex_polyhedron_Euler_formula_l430_430073


namespace how_many_trucks_l430_430489

-- Define the conditions given in the problem
def people_to_lift_car : ℕ := 5
def people_to_lift_truck : ℕ := 2 * people_to_lift_car

-- Set up the problem conditions
def total_people_needed (cars : ℕ) (trucks : ℕ) : ℕ :=
  cars * people_to_lift_car + trucks * people_to_lift_truck

-- Now state the precise theorem we need to prove
theorem how_many_trucks (cars trucks total_people : ℕ) 
  (h1 : cars = 6)
  (h2 : trucks = 3)
  (h3 : total_people = total_people_needed cars trucks) :
  trucks = 3 :=
by
  sorry

end how_many_trucks_l430_430489


namespace total_students_l430_430568

noncomputable def totalStudentsOptions (groups totalGroups specificGroupCount specificGroupSize otherGroupSizes : ℕ) : Set ℕ :=
  if totalGroups = 6 ∧ specificGroupCount = 4 ∧ specificGroupSize = 13 ∧ (otherGroupSizes = 12 ∨ otherGroupSizes = 14) then
    {52 + 2 * otherGroupSizes}
  else
    ∅

theorem total_students :
  totalStudentsOptions 6 4 13 12 = {76} ∧ totalStudentsOptions 6 4 13 14 = {80} :=
by
  -- This is where the proof would go, but we're skipping it as per instructions
  sorry

end total_students_l430_430568


namespace sqrt3_mul_sqrt12_eq_6_l430_430254

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l430_430254


namespace sqrt_mul_eq_l430_430312

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l430_430312


namespace least_possible_cost_l430_430714

-- Definitions of the conditions
def cost_5_pound_bag := 13.82
def cost_10_pound_bag := 20.43
def cost_25_pound_bag := 32.25

def weight_5_pound_bag := 5
def weight_10_pound_bag := 10
def weight_25_pound_bag := 25

def min_weight := 65
def max_weight := 80

-- Theorem statement
theorem least_possible_cost :
  ∃ (num_5lb num_10lb num_25lb : ℕ),
    num_5lb * weight_5_pound_bag + num_10lb * weight_10_pound_bag + num_25lb * weight_25_pound_bag ≥ min_weight ∧
    num_5lb * weight_5_pound_bag + num_10lb * weight_10_pound_bag + num_25lb * weight_25_pound_bag ≤ max_weight ∧
    num_5lb * cost_5_pound_bag + num_10lb * cost_10_pound_bag + num_25lb * cost_25_pound_bag = 98.75 := sorry

end least_possible_cost_l430_430714


namespace count_integers_between_sqrt_10_and_sqrt_50_l430_430444

theorem count_integers_between_sqrt_10_and_sqrt_50 :
  ∃ n : ℕ, n = 4 ∧ (∀ (k : ℕ), 4 ≤ k ∧ k ≤ 7 → (sqrt 10 : ℝ) < k ∧ k < (sqrt 50 : ℝ)) :=
by
  sorry

end count_integers_between_sqrt_10_and_sqrt_50_l430_430444


namespace non_exclusive_events_l430_430677

-- Definition of events based on the given conditions
def total_balls : ℕ := 5
def red_balls : ℕ := 2
def black_balls : ℕ := 3

def at_least_one_black (selected : list ℕ) : Prop := 
  (selected.count 1 ≥ 1)

def both_black (selected : list ℕ) : Prop := 
  (selected.count 1 = 2)

def at_least_one_red (selected : list ℕ) : Prop := 
  (selected.count 0 ≥ 1)

def exactly_one_black (selected : list ℕ) : Prop := 
  (selected.count 1 = 1)

def exactly_two_black (selected : list ℕ) : Prop := 
  (selected.count 1 = 2)

def both_red (selected : list ℕ) : Prop := 
  (selected.count 0 = 2)

-- Main theorem stating the non-exclusive nature of the events A and B
theorem non_exclusive_events :
  (∃ selected : list ℕ, at_least_one_black selected ∧ both_black selected) ∧ 
  (∃ selected : list ℕ, at_least_one_black selected ∧ at_least_one_red selected) :=
by
  sorry

end non_exclusive_events_l430_430677


namespace nines_square_zeros_l430_430130

theorem nines_square_zeros (n : ℕ) : 
  let k := (10^n - 1) in 
  ∃ zeros : ℕ, (zeros = n - 1) ∧ (k^2).digits.count 0 = zeros := 
by
  sorry

end nines_square_zeros_l430_430130


namespace cone_CSA_change_rate_l430_430637

noncomputable def dCSA_dt (r l dr_dt dl_dt : ℝ) : ℝ :=
  π * (dr_dt * l + r * dl_dt)

theorem cone_CSA_change_rate :
  let r := 12
  let l := 14
  let dr_dt := -1
  let dl_dt := 2
  dCSA_dt r l dr_dt dl_dt = 10 * π :=
by
  -- We assume that the derivative function and values are correctly used here
  sorry

end cone_CSA_change_rate_l430_430637


namespace total_students_in_groups_l430_430537

theorem total_students_in_groups {N : ℕ} (h1 : ∃ g : ℕ → ℕ, (∀ i j, g i = 13 ∨ g j = 12 ∨ g j = 14) ∧ (∑ i in finset.range 6, g i) = N) : 
  N = 76 ∨ N = 80 :=
sorry

end total_students_in_groups_l430_430537


namespace sqrt3_mul_sqrt12_eq_6_l430_430264

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l430_430264


namespace max_zeros_consecutive_two_digit_product_l430_430662

theorem max_zeros_consecutive_two_digit_product :
  ∃ a b : ℕ, 10 ≤ a ∧ a < 100 ∧ b = a + 1 ∧ 10 ≤ b ∧ b < 100 ∧
  (∀ c, (c * 10) ∣ a * b → c ≤ 2) := 
  by
    sorry

end max_zeros_consecutive_two_digit_product_l430_430662


namespace sqrt_mul_l430_430180

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l430_430180


namespace Angela_insect_count_l430_430734

variables (Angela Jacob Dean : ℕ)
-- Conditions
def condition1 : Prop := Angela = Jacob / 2
def condition2 : Prop := Jacob = 5 * Dean
def condition3 : Prop := Dean = 30

-- Theorem statement proving Angela's insect count
theorem Angela_insect_count (h1 : condition1 Angela Jacob) (h2 : condition2 Jacob Dean) (h3 : condition3 Dean) : Angela = 75 :=
by
  sorry

end Angela_insect_count_l430_430734


namespace find_k_value_l430_430987

theorem find_k_value (P : ℝ × ℝ) (k : ℝ) (QR : ℝ) (hP : P = (8, 6)) (hQR : QR = 3) :
  S = (0, k) ∧ S = (0, 7) := 
by
  let OP := (P.1^2 + P.2^2).sqrt
  have hOP : OP = 10 := 
  sorry
  let OR := 10
  let OQ := OR - QR
  have hOQ : OQ = 7 := 
  sorry
  have hS : S = (0, 7) := 
  sorry
  exact ⟨rfl, hS⟩

end find_k_value_l430_430987


namespace geoffrey_remaining_money_l430_430381

theorem geoffrey_remaining_money (gma aunt uncle wallet cost_per_game num_games : ℕ)
  (h_gma : gma = 20)
  (h_aunt : aunt = 25)
  (h_uncle : uncle = 30)
  (h_wallet : wallet = 125)
  (h_cost_per_game : cost_per_game = 35)
  (h_num_games : num_games = 3) :
  wallet - num_games * cost_per_game = 20 :=
by
  -- Given conditions
  have h_gifts := h_gma + h_aunt + h_uncle
  have h_initial_money := h_wallet - h_gifts
  -- Calculate the total cost of games
  have h_total_cost := h_num_games * h_cost_per_game
  -- Calculate the remaining money
  have h_remaining := h_wallet - h_total_cost
  -- Simplification steps (omitted)
  sorry

end geoffrey_remaining_money_l430_430381


namespace total_students_l430_430582

theorem total_students (N : ℕ) (h1 : ∃ g1 g2 g3 g4 g5 g6 : ℕ, 
  g1 = 13 ∧ g2 = 13 ∧ g3 = 13 ∧ g4 = 13 ∧ 
  ((g5 = 12 ∧ g6 = 12) ∨ (g5 = 14 ∧ g6 = 14)) ∧ 
  N = g1 + g2 + g3 + g4 + g5 + g6) : 
  N = 76 ∨ N = 80 :=
by
  sorry

end total_students_l430_430582


namespace diagonals_of_angle_bisectors_l430_430636

theorem diagonals_of_angle_bisectors (a b : ℝ) (BAD ABC : ℝ) (hBAD : BAD = ABC) :
  ∃ d : ℝ, d = |a - b| :=
by
  sorry

end diagonals_of_angle_bisectors_l430_430636


namespace frequency_number_correct_l430_430397

-- Define the sample capacity and the group frequency as constants
def sample_capacity : ℕ := 100
def group_frequency : ℝ := 0.3

-- State the theorem
theorem frequency_number_correct : sample_capacity * group_frequency = 30 := by
  -- Immediate calculation
  sorry

end frequency_number_correct_l430_430397


namespace sqrt_mul_sqrt_eq_six_l430_430282

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l430_430282


namespace hyperbola_trajectory_of_C_l430_430478

noncomputable def is_hyperbola_trajectory (C : ℝ × ℝ) : Prop :=
∃ x y : ℝ, C = (x, y) ∧ (x^2 / 4 - y^2 / 5 = 1) ∧ (x ≥ 2)

theorem hyperbola_trajectory_of_C :
  ∀ C : ℝ × ℝ, (∃ A B : ℝ × ℝ, A = (-3, 0) ∧ B = (3, 0) ∧
  (dist C A - dist C B).abs = 4) → is_hyperbola_trajectory C :=
begin
  sorry
end

end hyperbola_trajectory_of_C_l430_430478


namespace total_students_in_groups_l430_430535

theorem total_students_in_groups {N : ℕ} (h1 : ∃ g : ℕ → ℕ, (∀ i j, g i = 13 ∨ g j = 12 ∨ g j = 14) ∧ (∑ i in finset.range 6, g i) = N) : 
  N = 76 ∨ N = 80 :=
sorry

end total_students_in_groups_l430_430535


namespace find_largest_cos_x_l430_430508

theorem find_largest_cos_x (x y z : ℝ) 
  (h1 : Real.sin x = Real.cot y)
  (h2 : Real.sin y = Real.cot z)
  (h3 : Real.sin z = Real.cot x) :
  Real.cos x ≤ Real.sqrt ((3 - Real.sqrt 5) / 2) := sorry

end find_largest_cos_x_l430_430508


namespace line_passes_through_fixed_point_correct_k_for_shortest_chord_l430_430393

-- Define the circle and the line
def C := { p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 169 }
def l (k : ℝ) := { p : ℝ × ℝ | k * p.1 - p.2 - 4 * k + 5 = 0 }

-- Define the fixed point (4, 5)
def fixed_point := (4, 5) : ℝ × ℝ

-- Statements to prove
theorem line_passes_through_fixed_point (k : ℝ) : fixed_point ∈ l k := by
  sorry

theorem correct_k_for_shortest_chord : ∀ k : ℝ, (k = -3/4) → 
  (let distance := sqrt ((4 - 1)^2 + (5 - 1)^2) in
  let radius := sqrt 169 in
  shortest_chord := 2 * sqrt (radius^2 - distance^2) =
  if l k contains shortest_chord) := by
  sorry

end line_passes_through_fixed_point_correct_k_for_shortest_chord_l430_430393


namespace sqrt_mult_simplify_l430_430246

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l430_430246


namespace overall_loss_percentage_l430_430979

def cost_price_radio : ℝ := 1500
def cost_price_speaker : ℝ := 2500
def cost_price_headphones : ℝ := 800

def sale_price_radio : ℝ := 1275
def sale_price_speaker : ℝ := 2300
def sale_price_headphones : ℝ := 700

def total_cost_price : ℝ := cost_price_radio + cost_price_speaker + cost_price_headphones
def total_sale_price : ℝ := sale_price_radio + sale_price_speaker + sale_price_headphones

def loss : ℝ := total_cost_price - total_sale_price
def loss_percentage : ℝ := (loss / total_cost_price) * 100

theorem overall_loss_percentage :
  loss_percentage = 10.94 := by
  sorry

end overall_loss_percentage_l430_430979


namespace sqrt_multiplication_l430_430143

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l430_430143


namespace max_small_change_l430_430042

    theorem max_small_change : ∃ max_amount : ℕ,
      (max_amount = 119) ∧ 
      ∀ target_amount : ℕ, target_amount ∈ {100, 50, 25, 10, 5} → ¬ ∃ coindist : list ℕ, 
        list_sum coindist = target_amount ∧ 
        (∀ c ∈ coindist, c ∈ {1, 5, 10, 25, 50, 100}) :=
    begin
      sorry
    end
    
end max_small_change_l430_430042


namespace max_red_sweaters_l430_430082

def max_girls_in_red_sweaters (n : Nat) (C : Fin n → Prop) (G : Fin n → Prop) (B : Fin n → Prop) :=
  ∃ (max_girls_red : Nat), max_girls_red = 24 ∧ ∀ i, C i → G i → ∃ j, (B j ∧ (j = i + 1 ∨ j = i - 1))

theorem max_red_sweaters (n : Nat) (h_n : n = 36) :
  ∃ (max_girls_red : Nat), max_girls_red = 24 := 
by
  have h_conditions : ∀ (C G B : Fin n → Prop), max_girls_in_red_sweaters n C G B → True, sorry
  sorry

end max_red_sweaters_l430_430082


namespace solve_eq1_solve_eq2_l430_430374

-- Define the condition and the statement to be proved for the first equation
theorem solve_eq1 (x : ℝ) : 9 * x^2 - 25 = 0 → x = 5 / 3 ∨ x = -5 / 3 :=
by sorry

-- Define the condition and the statement to be proved for the second equation
theorem solve_eq2 (x : ℝ) : (x + 1)^3 - 27 = 0 → x = 2 :=
by sorry

end solve_eq1_solve_eq2_l430_430374


namespace intersection_complement_l430_430437

def U := {1, 2, 3, 4, 5, 6}
def A := {2, 3, 4}
def B := {4, 5, 6}
def C_U (B : Set ℕ) (U : Set ℕ) := U \ B

theorem intersection_complement (U A B : Set ℕ) :
  A = {2, 3, 4} →
  B = {4, 5, 6} →
  U = {1, 2, 3, 4, 5, 6} →
  A ∩ C_U B U = {2, 3} :=
begin
  sorry
end

end intersection_complement_l430_430437


namespace price_difference_is_300_cents_l430_430341

noncomputable def list_price : ℝ := 59.99
noncomputable def tech_bargains_price : ℝ := list_price - 15
noncomputable def digital_deal_price : ℝ := 0.7 * list_price
noncomputable def price_difference : ℝ := tech_bargains_price - digital_deal_price
noncomputable def price_difference_in_cents : ℝ := price_difference * 100

theorem price_difference_is_300_cents :
  price_difference_in_cents = 300 := by
  sorry

end price_difference_is_300_cents_l430_430341


namespace total_students_76_or_80_l430_430572

theorem total_students_76_or_80 
  (N : ℕ)
  (h1 : ∃ g : ℕ → ℕ, (∑ i in finset.range 6, g i = N) ∧
                     (∃ a b : ℕ, finset.card {i | g i = a} = 4 ∧ 
                                 finset.card {i | g i = b} = 2 ∧ 
                                 (a = 13 ∧ (b = 12 ∨ b = 14))))
  : N = 76 ∨ N = 80 :=
sorry

end total_students_76_or_80_l430_430572


namespace closest_integer_to_7_mul_3_div_4_l430_430999

theorem closest_integer_to_7_mul_3_div_4 : 
  (argmin (λ n : ℤ, |(7 * (3 / 4 : ℝ)) - n) [5]) = 5 :=
sorry

end closest_integer_to_7_mul_3_div_4_l430_430999


namespace trig_expression_value_l430_430418

theorem trig_expression_value (a : ℝ) (h : a < 0) :
    let x := 4 * a,
        y := 3 * a,
        r := -5 * a,
        sin_alpha := y / r,
        tan_alpha := y / x,
        tan_2alpha := 2 * tan_alpha / (1 - tan_alpha^2) in
    25 * sin_alpha - 7 * tan_2alpha = -1497 / 55 :=
by
    let x := 4 * a
    let y := 3 * a
    let r := -5 * a
    let sin_alpha := y / r
    let tan_alpha := y / x
    let tan_2alpha := 2 * tan_alpha / (1 - tan_alpha^2)
    calc 25 * sin_alpha - 7 * tan_2alpha = sorry

end trig_expression_value_l430_430418


namespace correct_projection_matrix_l430_430916

open Matrix

noncomputable def proj_matrix_onto (u : Vector ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let n := dot_product u u
  scalar (1 / n) ⬝ (outer u u)

def matrix_projection_result : Matrix (Fin 2) (Fin 2) ℝ :=
  let u := ![4, 2]
  let v := ![2, 1]
  let P1 := proj_matrix_onto u
  let P2 := proj_matrix_onto v
  P1 ⬝ P2

theorem correct_projection_matrix : matrix_projection_result = ![
  ![4 / 5, 2 / 5], 
  ![2 / 5, 1 / 5]] := sorry

end correct_projection_matrix_l430_430916


namespace find_tan_alpha_plus_pi_div_12_l430_430404

theorem find_tan_alpha_plus_pi_div_12 (α : ℝ) (h : Real.sin α = 3 * Real.sin (α + Real.pi / 6)) :
  Real.tan (α + Real.pi / 12) = 2 * Real.sqrt 3 - 4 :=
by
  sorry

end find_tan_alpha_plus_pi_div_12_l430_430404


namespace eccentricity_is_half_l430_430829

noncomputable def eccentricity_ellipse (a b : ℝ) (h : a > b ∧ b > 0)
  (ha : ∃ l : ℝ → ℝ, (∀ x y, l x = y → y = (Real.tan (π / 3)) * (x - a)) ∧
                      (∀ x y, (x^2 + y^2 = b^2) → ∃ y1, l x = y1 → y1 = y)) : ℝ :=
let c := Real.sqrt (a^2 - b^2) in c / a

-- Statement of the theorem
theorem eccentricity_is_half (a b : ℝ) (h : a > b ∧ b > 0)
  (ha : ∃ l : ℝ → ℝ, (∀ x y, l x = y → y = (Real.tan (π / 3)) * (x - a)) ∧
                      (∀ x y, (x^2 + y^2 = b^2) → ∃ y1, l x = y1 → y1 = y)) :
  eccentricity_ellipse a b h ha = 1 / 2 :=
sorry

end eccentricity_is_half_l430_430829


namespace winning_strategy_first_player_l430_430990

theorem winning_strategy_first_player (m n : ℕ) (pile : ℕ) :
  (∀ remaining_pile : ℕ, 
    (remaining_pile = 0 → False) →
    (remaining_pile ≥ 1 → (remaining_pile - 1 ≥ m ∨ remaining_pile - 1 ≥ n)) ∧
    (remaining_pile ≥ 10 → (remaining_pile - 10 ≥ m ∨ remaining_pile - 10 ≥ n))) →
  (m ≥ 9 ∧ n ≥ 9 ∧ |m - n| ≤ 9) :=
by sorry

end winning_strategy_first_player_l430_430990


namespace describe_graph_l430_430338

noncomputable def points_satisfying_equation (x y : ℝ) : Prop :=
  (x - y) ^ 2 = x ^ 2 + y ^ 2

theorem describe_graph : {p : ℝ × ℝ | points_satisfying_equation p.1 p.2} = {p : ℝ × ℝ | p.1 = 0} ∪ {p : ℝ × ℝ | p.2 = 0} :=
by
  sorry

end describe_graph_l430_430338


namespace total_students_possible_l430_430555

theorem total_students_possible (A B : ℕ) :
  (4 * 13) + 2 * A = 76 ∨ (4 * 13) + 2 * B = 80 :=
by
  -- Let N be the total number of students
  let N := (4 * 13)
  -- Given that the number of students in the remaining 2 groups differs by no more than 1
  have h : A = 12 ∨ B = 14 := sorry
  -- Prove the possible values
  exact or.inl (N + 2 * 12 = 76) <|> or.inr (N + 2 * 14 = 80)

end total_students_possible_l430_430555


namespace max_ab_bc_cd_l430_430915

theorem max_ab_bc_cd (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) (h_sum : a + b + c + d = 200) : 
    ab + bc + cd ≤ 10000 := by
  sorry

end max_ab_bc_cd_l430_430915


namespace sqrt_mul_eq_6_l430_430161

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l430_430161


namespace find_number_l430_430717

variable (N : ℕ)

theorem find_number (h : 6 * ((N / 8) + 8 - 30) = 12) : N = 192 := 
by
  sorry

end find_number_l430_430717


namespace largest_sum_of_two_largest_angles_of_EFGH_l430_430476

theorem largest_sum_of_two_largest_angles_of_EFGH (x d : ℝ) (y z : ℝ) :
  (∃ a b : ℝ, a + 2 * b = x + 70 ∧ a + b = 70 ∧ 2 * a + 3 * b = 180) ∧
  (2 * x + 3 * d = 180) ∧ (x = 30) ∧ (y = 70) ∧ (z = 100) ∧ (z + 70 = x + d) ∧
  x + d + x + 2 * d + x + 3 * d + x = 360 →
  max (70 + y) (70 + z) + max (y + 70) (z + 70) = 210 := 
sorry

end largest_sum_of_two_largest_angles_of_EFGH_l430_430476


namespace tea_set_costs_l430_430005
noncomputable section

-- Definition for the conditions of part 1
def cost_condition1 (x y : ℝ) : Prop := x + 2 * y = 250
def cost_condition2 (x y : ℝ) : Prop := 3 * x + 4 * y = 600

-- Definition for the conditions of part 2
def cost_condition3 (a : ℝ) : ℝ := 108 * a + 60 * (80 - a)

-- Definition for the conditions of part 3
def profit (a b : ℝ) : ℝ := 30 * a + 20 * b

theorem tea_set_costs (x y : ℝ) (a : ℕ) :
  cost_condition1 x y →
  cost_condition2 x y →
  x = 100 ∧ y = 75 ∧ a ≤ 30 ∧ profit 30 50 = 1900 := by
  sorry

end tea_set_costs_l430_430005


namespace sqrt3_mul_sqrt12_eq_6_l430_430229

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l430_430229


namespace probability_stopping_same_type_l430_430995

-- Defining the types of socks
inductive SockType
| complex
| synthetic
| trigonometric

-- Victor's drawer contains exactly 2 of each type of sock
def drawer : List SockType := 
  [SockType.complex, SockType.complex, SockType.synthetic, SockType.synthetic, SockType.trigonometric, SockType.trigonometric]

-- Function to count the number of ways to draw 2 socks of the same type from the drawer
def same_type_pairs (drawer : List SockType) : Nat :=
  List.length (List.filter (λ (p : SockType × SockType), p.fst = p.snd) 
    (List.product drawer drawer).filter (λ (p : SockType × SockType), p.fst ≠ p.snd))

-- Total number of ways to draw 2 different socks from the drawer
def total_pairs (drawer : List SockType) : Nat :=
  List.length ((List.product drawer drawer).filter (λ (p : SockType × SockType), p.fst ≠ p.snd))

-- Probability that Victor stops with 2 socks of the same type
def stopping_probability_same_type (drawer : List SockType) : ℚ :=
  same_type_pairs drawer / total_pairs drawer

-- The theorem to state the probability that Victor stops with 2 socks of the same type
theorem probability_stopping_same_type : 
  stopping_probability_same_type drawer = 1/5 :=
by 
  -- (proof is to be filled in)
  sorry

end probability_stopping_same_type_l430_430995


namespace sqrt_mul_sqrt_eq_six_l430_430291

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l430_430291


namespace polynomial_expansion_correct_l430_430813

theorem polynomial_expansion_correct :
  let p := (x - 2)^8 in
  p.coeff 0 = 256 ∧
  p.coeff 8 = 1 ∧
  (∑ i in Finset.range 8, p.coeff (i + 1)) = -255 ∧
  (Finset.sum (Finset.range 9) (λ i, p.coeff i * if i % 2 = 0 then 1 else -1)) = 6561 :=
by
  sorry

end polynomial_expansion_correct_l430_430813


namespace simplify_log_expression_l430_430598

theorem simplify_log_expression :
  (1 / (Real.logBase 15 2 + 1) + 1 / (Real.logBase 10 3 + 1) + 1 / (Real.logBase 6 5 + 1)) = 2 :=
by
  sorry

end simplify_log_expression_l430_430598


namespace circle_eq_l430_430827

theorem circle_eq (x y : ℝ) (h_center : (-1, 2)) (h_pass : (2, -2)) :
  ∃ r : ℝ, (x + 1)^2 + (y - 2)^2 = r^2 ∧ r^2 = 25 := 
sorry

end circle_eq_l430_430827


namespace sqrt_mult_l430_430191

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l430_430191


namespace total_students_l430_430583

theorem total_students (N : ℕ) (h1 : ∃ g1 g2 g3 g4 g5 g6 : ℕ, 
  g1 = 13 ∧ g2 = 13 ∧ g3 = 13 ∧ g4 = 13 ∧ 
  ((g5 = 12 ∧ g6 = 12) ∨ (g5 = 14 ∧ g6 = 14)) ∧ 
  N = g1 + g2 + g3 + g4 + g5 + g6) : 
  N = 76 ∨ N = 80 :=
by
  sorry

end total_students_l430_430583


namespace sqrt_mult_simplify_l430_430244

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l430_430244


namespace find_BE_l430_430981

variables (A B C D E F G : Type) 
variables [parallelogram A B C D] [point E F G] (BD : line) (CD : line) (BC : line)

def intersect_at_vertex_A (E F G : point) : Prop :=
  intersects BD E ∧ intersects CD F ∧ intersects BC G ∧ 
  distance(B, E) = sqrt(10) ∧ ratio(FG, FE) = 9 ∧ distance(E, D) = 1

theorem find_BE : intersect_at_vertex_A E F G → distance(B, E) = sqrt(10) :=
by 
  sorry

end find_BE_l430_430981


namespace const_angle_sum_l430_430736

/-- Given a regular tetrahedron ABCD, and points E on AB and F on CD such that
    the ratios of AE to EB and CF to FD are equal to λ, prove that the sum of
    the angles αλ (formed by EF with AC) and βλ (formed by EF with BD) is
    constantly equal to π/2 for any λ in (0, +∞). -/
theorem const_angle_sum (A B C D E F : Type) 
    [regular_tetrahedron A B C D]
    (λ : ℝ) (hλ : 0 < λ) :
    (∃ E ∈ segment A B, ∃ F ∈ segment C D, (AE / EB = λ ∧ CF / FD = λ)) →
    let α := ∠ (line_through E F) (line_through A C) in
    let β := ∠ (line_through E F) (line_through B D) in
    (α + β = π / 2) :=
sorry

end const_angle_sum_l430_430736


namespace initial_eggs_correct_l430_430087

def initial_eggs (E : ℕ) : Prop :=
  let used_eggs := 5 in
  let laid_eggs := 2 * 3 in
  let current_eggs := 11 in
  E - used_eggs + laid_eggs = current_eggs

theorem initial_eggs_correct : ∃ (E : ℕ), initial_eggs E ∧ E = 10 :=
by
  use 10
  unfold initial_eggs
  simp
  sorry

end initial_eggs_correct_l430_430087


namespace plane_coloring_no_monochromatic_segment_l430_430941

theorem plane_coloring_no_monochromatic_segment :
  ∀ (color : ℝ × ℝ → ℕ), 
    (∀ (p : ℝ × ℝ), 
      (∃ (q : ℚ), dist (0, 0) p = abs q → color p = 0) ∨ 
      (∀ (irr : ℝ), dist (0, 0) p = irr ∧ irrational irr → color p = 1)) →
       (∀ (A B : ℝ × ℝ), A ≠ B → ¬ (∀ P, P ∈ segment ℝ A B → color P = color A)) :=
by
  intro color hcolor A B hAB
  sorry

end plane_coloring_no_monochromatic_segment_l430_430941


namespace jeffrey_unanswered_questions_l430_430899

theorem jeffrey_unanswered_questions(
    c w u : ℕ,
    h1 : 40 + 4 * c - w = 100,
    h2 : 6 * c + 3 * u = 120,
    h3 : c + w + u = 35
  ) : u = 5 := by
  sorry

end jeffrey_unanswered_questions_l430_430899


namespace minimize_std_deviation_l430_430997

theorem minimize_std_deviation (m n : ℝ) (h1 : m + n = 32) 
    (h2 : 11 ≤ 12 ∧ 12 ≤ m ∧ m ≤ n ∧ n ≤ 20 ∧ 20 ≤ 27) : 
    m = 16 :=
by {
  -- No proof required, only the theorem statement as per instructions
  sorry
}

end minimize_std_deviation_l430_430997


namespace compare_a_b_c_l430_430797

theorem compare_a_b_c (a b c : ℝ) (h1 : a = Real.log 0.9 / Real.log 0.8)
    (h2 : b = Real.log 0.9 / Real.log 1.1)
    (h3 : c = 1.1 ^ 0.9) :
    c > a ∧ a > b := 
sorry

end compare_a_b_c_l430_430797


namespace find_y_l430_430056

theorem find_y (y : ℝ) (h : (3 * y) / 7 = 12) : y = 28 :=
by
  -- The proof would go here
  sorry

end find_y_l430_430056


namespace S_formula_l430_430635

noncomputable def A (n : ℕ) : List ℕ := 
  List.range n |>.map (fun i => 2^i - 1)

noncomputable def T (A : List ℕ) (k : ℕ) : ℕ :=
  (A.powerset.filter (fun s => s.length = k)).sum (fun s => s.prod)

noncomputable def S (n : ℕ) : ℕ :=
  (List.range (n + 1)).tail.sum (T (A n))

theorem S_formula (n : ℕ) : S n = 2^((n * (n + 1)) / 2) - 1 :=
  sorry

end S_formula_l430_430635


namespace sqrt3_mul_sqrt12_eq_6_l430_430231

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l430_430231


namespace inscribed_circle_radius_square_l430_430703

theorem inscribed_circle_radius_square (ER RF GS SH : ℕ) (r : ℕ)
  (hER : ER = 24) (hRF : RF = 31) (hGS : GS = 40) (hSH : SH = 29)
  (htangent_eq: 
    arctan (24 / r) + arctan (31 / r) + arctan (40 / r) + arctan (29 / r) = 180) :
  r^2 = 945 :=
by { sorry }

end inscribed_circle_radius_square_l430_430703


namespace a_n_div_3_sum_two_cubes_a_n_div_3_not_sum_two_squares_l430_430379

def a_n (n : ℕ) : ℕ := 10^(3*n+2) + 2 * 10^(2*n+1) + 2 * 10^(n+1) + 1

theorem a_n_div_3_sum_two_cubes (n : ℕ) : ∃ x y : ℤ, (x > 0) ∧ (y > 0) ∧ (a_n n / 3 = x^3 + y^3) := sorry

theorem a_n_div_3_not_sum_two_squares (n : ℕ) : ¬ (∃ x y : ℤ, a_n n / 3 = x^2 + y^2) := sorry

end a_n_div_3_sum_two_cubes_a_n_div_3_not_sum_two_squares_l430_430379


namespace sqrt3_mul_sqrt12_eq_6_l430_430219

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l430_430219


namespace trigonometric_expression_value_l430_430409

theorem trigonometric_expression_value 
  (θ : ℝ) 
  (hθ1 : 0 < θ) 
  (hθ2 : θ < π / 4) 
  (h : sin θ - cos θ = - √14 / 4) : 
  (2 * cos θ ^ 2 - 1) / cos (π / 4 + θ) = 3 / 2 := 
sorry

end trigonometric_expression_value_l430_430409


namespace recurring_subtraction_l430_430767

theorem recurring_subtraction (x y : ℚ) (h1 : x = 35 / 99) (h2 : y = 7 / 9) : x - y = -14 / 33 := by
  sorry

end recurring_subtraction_l430_430767


namespace evaluate_expression_l430_430765

noncomputable def problem_expression : ℝ := (sqrt 8) * (2 ^ (3 / 2)) + (18 / 3) * 3 - (6 ^ (5 / 2))

theorem evaluate_expression :
  problem_expression = 26 - 36 * sqrt 6 :=
by
  sorry

end evaluate_expression_l430_430765


namespace sqrt_mul_simplify_l430_430266

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l430_430266


namespace maximum_median_soda_shop_l430_430739

noncomputable def soda_shop_median (total_cans : ℕ) (total_customers : ℕ) (min_cans_per_customer : ℕ) : ℝ :=
  if total_cans = 300 ∧ total_customers = 120 ∧ min_cans_per_customer = 1 then 3.5 else sorry

theorem maximum_median_soda_shop : soda_shop_median 300 120 1 = 3.5 :=
by
  sorry

end maximum_median_soda_shop_l430_430739


namespace range_of_f_range_of_a_l430_430080

section Problem1

-- Define the function
def f (x : ℝ) : ℝ := 2 * |x - 1| - |x - 4|

-- Statement of the first problem
theorem range_of_f : Set.Icc (-3) (Real.sup (Set.univ : Set ℝ)) = {y : ℝ | y >= -3}  := sorry
-- Explanation:
-- The statement "Set.Icc (-3) (Real.sup (Set.univ : Set ℝ)) = {y : ℝ | y >= -3}" translates to 
-- the range of the function f being [-3, +∞)

end Problem1

section Problem2

-- Define the function with variable parameter a
def g (a : ℝ) (x : ℝ) : ℝ := 2 * |x - 1| - |x - a|

-- Statement of the second problem
theorem range_of_a (a : ℝ) : (∀ x : ℝ, g a x ≥ -1) → 0 ≤ a ∧ a ≤ 2 := sorry
-- Explanation:
-- The statement translates to "if 2|x-1| - |x-a| ≥ -1 for all x ∈ ℝ, then 0 ≤ a ≤ 2"

end Problem2

end range_of_f_range_of_a_l430_430080


namespace quadratic_discriminant_l430_430371

theorem quadratic_discriminant :
  let a := 3
  let b := -7
  let c := -6 in
  b^2 - 4 * a * c = 121 := 
by 
  let a := 3
  let b := -7
  let c := -6
  sorry

end quadratic_discriminant_l430_430371


namespace count_7s_minus_2s_74_l430_430346

def count_digit (d : Nat) (n : Nat) : Nat :=
  n.digits.count d

def total_digit_count (d : Nat) (start : Nat) (end_ : Nat) : Nat :=
  (List.range' start (end_ - start + 1)).sum (count_digit d)

def diff_digit_count (d1 d2 : Nat) (start end_ : Nat) : Nat :=
  total_digit_count d1 start end_ - total_digit_count d2 start end_

theorem count_7s_minus_2s_74 : diff_digit_count 7 2 1 792 = 74 :=
by 
  sorry

end count_7s_minus_2s_74_l430_430346


namespace a_works_less_than_b_l430_430726

theorem a_works_less_than_b (A B : ℝ) (x y : ℝ)
  (h1 : A = 3 * B)
  (h2 : (A + B) * 22.5 = A * x)
  (h3 : y = 3 * x) :
  y - x = 60 :=
by sorry

end a_works_less_than_b_l430_430726


namespace total_students_l430_430580

theorem total_students (N : ℕ) (h1 : ∃ g1 g2 g3 g4 g5 g6 : ℕ, 
  g1 = 13 ∧ g2 = 13 ∧ g3 = 13 ∧ g4 = 13 ∧ 
  ((g5 = 12 ∧ g6 = 12) ∨ (g5 = 14 ∧ g6 = 14)) ∧ 
  N = g1 + g2 + g3 + g4 + g5 + g6) : 
  N = 76 ∨ N = 80 :=
by
  sorry

end total_students_l430_430580


namespace sqrt_mul_eq_6_l430_430163

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l430_430163


namespace average_speed_is_20_mph_l430_430923

-- Defining the conditions
def distance1 := 40 -- miles
def speed1 := 20 -- miles per hour
def distance2 := 20 -- miles
def speed2 := 40 -- miles per hour
def distance3 := 30 -- miles
def speed3 := 15 -- miles per hour

-- Calculating total distance and total time
def total_distance := distance1 + distance2 + distance3
def time1 := distance1 / speed1 -- hours
def time2 := distance2 / speed2 -- hours
def time3 := distance3 / speed3 -- hours
def total_time := time1 + time2 + time3

-- Theorem statement
theorem average_speed_is_20_mph : (total_distance / total_time) = 20 := by
  sorry

end average_speed_is_20_mph_l430_430923


namespace num_integers_between_sqrt10_sqrt50_l430_430445

theorem num_integers_between_sqrt10_sqrt50 : 
  ∃ n : ℕ, n = 4 ∧ 
    ∀ x : ℤ, (⌈real.sqrt 10⌉ ≤ x ∧ x ≤ ⌊real.sqrt 50⌋) → x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 :=
by
  sorry

end num_integers_between_sqrt10_sqrt50_l430_430445


namespace function_increasing_on_interval_l430_430429

theorem function_increasing_on_interval {x : ℝ} (hx : x < 1) : 
  (-1/2) * x^2 + x + 4 < -1/2 * (x + 1)^2 + (x + 1) + 4 :=
sorry

end function_increasing_on_interval_l430_430429


namespace final_amounts_calculation_l430_430903

noncomputable def article_A_original_cost : ℚ := 200
noncomputable def article_B_original_cost : ℚ := 300
noncomputable def article_C_original_cost : ℚ := 400
noncomputable def exchange_rate_euro_to_usd : ℚ := 1.10
noncomputable def exchange_rate_gbp_to_usd : ℚ := 1.30
noncomputable def discount_A : ℚ := 0.50
noncomputable def discount_B : ℚ := 0.30
noncomputable def discount_C : ℚ := 0.40
noncomputable def sales_tax_rate : ℚ := 0.05
noncomputable def reward_points : ℚ := 100
noncomputable def reward_point_value : ℚ := 0.05

theorem final_amounts_calculation :
  let discounted_A := article_A_original_cost * discount_A
  let final_A := (article_A_original_cost - discounted_A) * exchange_rate_euro_to_usd
  let discounted_B := article_B_original_cost * discount_B
  let final_B := (article_B_original_cost - discounted_B) * exchange_rate_gbp_to_usd
  let discounted_C := article_C_original_cost * discount_C
  let final_C := article_C_original_cost - discounted_C
  let total_discounted_cost_usd := final_A + final_B + final_C
  let sales_tax := total_discounted_cost_usd * sales_tax_rate
  let reward := reward_points * reward_point_value
  let final_amount_usd := total_discounted_cost_usd + sales_tax - reward
  let final_amount_euro := final_amount_usd / exchange_rate_euro_to_usd
  final_amount_usd = 649.15 ∧ final_amount_euro = 590.14 :=
by
  sorry

end final_amounts_calculation_l430_430903


namespace perimeter_triangle_PF1F2_l430_430812

-- Definition of given points and conditions
def F1 := (-1 : ℝ, 0 : ℝ)
def F2 := (1 : ℝ, 0 : ℝ)

-- Definition of the condition 
def condition (x y : ℝ) : Prop := 
  (Real.sqrt ((x - 1)^2 + y^2)) / (Real.abs (x - 3)) = Real.sqrt (3) / 3

-- Definition of the equation of the ellipse
def ellipse (x y : ℝ) : Prop := 
  (x^2 / 3) + (y^2 / 2) = 1

-- Theorem statement
theorem perimeter_triangle_PF1F2 (x y : ℝ) (h1 : condition x y) : 
  ellipse x y → 
  -- Given the ellipse condition, the perimeter is 2\sqrt{3} + 2
  let a := Real.sqrt 3
  let c := 1 in 
  2 * a + 2 * c = (2 * Real.sqrt 3 + 2) :=
begin
  sorry
end

end perimeter_triangle_PF1F2_l430_430812


namespace angle_bisector_length_l430_430483

theorem angle_bisector_length
  (α β a l : ℝ)
  (triangle_ABC : Type)
  (angle_BAC : angle triangle_ABC = α)
  (side_BC : side triangle_ABC = a)
  (angle_bisector_AD : angle_bisector triangle_ABC = AD)
  (height_AE_perp_BC : height_perp AD AE)
  (angle_between_AD_AE : angle_between AD AE = β)
  (angle_DAB_eq_DAC : ∠DAB = ∠DAC) :
  l = (a * (cos (β - α / 2)) * (cos (β + α / 2))) / (sin α * cos β) :=
begin
  sorry   -- Proof is not required, just the statement
end

end angle_bisector_length_l430_430483


namespace sqrt_mul_eq_l430_430308

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l430_430308


namespace area_of_triangle_given_conditions_l430_430484

def area_of_triangle (a c B : ℝ) : ℝ := 0.5 * a * c * Real.sin B

theorem area_of_triangle_given_conditions :
    area_of_triangle 1 2 (Real.pi / 3) = Real.sqrt 3 / 2 :=
by 
  sorry

end area_of_triangle_given_conditions_l430_430484


namespace solve_for_pairs_l430_430660
-- Import necessary libraries

-- Define the operation
def diamond (a b c d : ℤ) : ℤ × ℤ :=
  (a * c - b * d, a * d + b * c)

theorem solve_for_pairs : ∃! (x y : ℤ), diamond x 3 x y = (6, 0) ∧ (x, y) = (0, -2) := by
  sorry

end solve_for_pairs_l430_430660


namespace geom_seq_sum_first_4_l430_430888

variable {a : ℕ → ℝ}
variable {r a1 : ℝ}

-- Conditions of the problem
def geom_seq (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) = a n * r
def a2_eq_9 : Prop := a 2 = 9
def a5_eq_243 : Prop := a 5 = 243

-- Statement to prove
theorem geom_seq_sum_first_4 :
  geom_seq a → a2_eq_9 → a5_eq_243 → (a 0 + a 1 + a 2 + a 3) = 120 :=
by
  sorry

end geom_seq_sum_first_4_l430_430888


namespace sqrt_mul_eq_6_l430_430162

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l430_430162


namespace base2_to_base4_l430_430661

theorem base2_to_base4 (n : ℕ) (h : n = 0b101101100) : nat.to_digits 4 n = [2, 3, 1, 1, 0] :=
by {
  sorry
}

end base2_to_base4_l430_430661


namespace probability_one_instrument_l430_430686

theorem probability_one_instrument (total_people : ℕ)
  (at_least_one_instrument_fraction : ℚ)
  (two_or_more_instruments : ℕ) 
  (h_total_people : total_people = 800)
  (h_fraction : at_least_one_instrument_fraction = 3/5)
  (h_two_or_more : two_or_more_instruments = 96) :
  (let at_least_one_instrument := at_least_one_instrument_fraction * total_people
   in let exactly_one_instrument := at_least_one_instrument - two_or_more_instruments
   in exactly_one_instrument / total_people = 0.48) := 
by 
  sorry

end probability_one_instrument_l430_430686


namespace Rachelle_GPA_probability_l430_430868

noncomputable def probGPA := 
  let englishA := 1 / 4
  let englishB := 1 / 3
  let englishC := 5 / 12
  let historyA := 1 / 3
  let historyB := 1 / 4
  let historyC := 5 / 12
  
  let totalProbability := 
    (englishA * historyA) +
    (englishA * historyB) +
    (englishB * historyA) +
    (englishB * historyB)
    
  let totalProbability := 
    (1 / 12) + 
    (1 / 16) + 
    (1 / 9) + 
    (1 / 12)
    
  totalProbability = (49 / 144)

theorem Rachelle_GPA_probability :
  (4 * 3 + sorry + sorry) / 5) >= 3.6 := 
begin
  simp,
  sorry
end

end Rachelle_GPA_probability_l430_430868


namespace max_n_for_positive_sn_l430_430415
open Nat

variable {a : ℕ → ℤ}          -- variable defining the arithmetic sequence

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, ∃ d, a (n + 1) = a n + d

def sn (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n + 1) * a 0 + (n * (n + 1)) / 2 * (a 1 - a 0)

theorem max_n_for_positive_sn
  (h1 : is_arithmetic_sequence a)
  (h2 : (a 11 : ℚ) / (a 10) < -1)
  (h3 : ∀ n, n > 0 → sn a n ≤ sn a (n - 1)) :
  21 = max (set_of (λ n, sn a n > 0)).to_finset.max' sorry :=
sorry

end max_n_for_positive_sn_l430_430415


namespace total_students_76_or_80_l430_430576

theorem total_students_76_or_80 
  (N : ℕ)
  (h1 : ∃ g : ℕ → ℕ, (∑ i in finset.range 6, g i = N) ∧
                     (∃ a b : ℕ, finset.card {i | g i = a} = 4 ∧ 
                                 finset.card {i | g i = b} = 2 ∧ 
                                 (a = 13 ∧ (b = 12 ∨ b = 14))))
  : N = 76 ∨ N = 80 :=
sorry

end total_students_76_or_80_l430_430576


namespace sqrt_mult_l430_430200

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l430_430200


namespace total_students_l430_430563

noncomputable def totalStudentsOptions (groups totalGroups specificGroupCount specificGroupSize otherGroupSizes : ℕ) : Set ℕ :=
  if totalGroups = 6 ∧ specificGroupCount = 4 ∧ specificGroupSize = 13 ∧ (otherGroupSizes = 12 ∨ otherGroupSizes = 14) then
    {52 + 2 * otherGroupSizes}
  else
    ∅

theorem total_students :
  totalStudentsOptions 6 4 13 12 = {76} ∧ totalStudentsOptions 6 4 13 14 = {80} :=
by
  -- This is where the proof would go, but we're skipping it as per instructions
  sorry

end total_students_l430_430563


namespace triangle_area_bounds_l430_430119

theorem triangle_area_bounds (r : ℝ) : 
  let y := λ x : ℝ, x^2 - 1,
      vertex := (0, -1 : ℝ),
      intersections : set ℝ := { x : ℝ | y x = r },
      area := (r + 1) * real.sqrt (r + 1)
  in 8 ≤ area ∧ area ≤ 64 → 3 ≤ r ∧ r ≤ 15 :=
by
  sorry

end triangle_area_bounds_l430_430119


namespace sqrt_of_4_l430_430639

theorem sqrt_of_4 :
  ∃ x : ℝ, x^2 = 4 ∧ (x = 2 ∨ x = -2) :=
sorry

end sqrt_of_4_l430_430639


namespace integral_result_l430_430068

open Real
open Set
open Topology

noncomputable def integral_tanlncos : ℝ :=
  ∫ x in 0..π/4, tan x * log (cos x)

theorem integral_result : integral_tanlncos = -1 / 2 * (log (sqrt 2 / 2))^2 := by
  sorry

end integral_result_l430_430068


namespace johns_monthly_pool_expenses_l430_430491

def cost_per_cleaning := 150
def tip_percentage := 0.10
def number_of_cleanings_per_month := 30 / 3
def cleaning_cost_with_tip := cost_per_cleaning * (1 + tip_percentage)
def total_cleaning_cost_per_month := number_of_cleanings_per_month * cleaning_cost_with_tip
def chemical_cost_per_month := 2 * 200
def equipment_rental_per_month := 100
def electricity_cost_per_month := 75
def total_expenses_per_month := total_cleaning_cost_per_month + chemical_cost_per_month + equipment_rental_per_month + electricity_cost_per_month

theorem johns_monthly_pool_expenses :
  total_expenses_per_month = 2225 :=
by
  sorry

end johns_monthly_pool_expenses_l430_430491


namespace euler_characteristic_convex_polyhedron_l430_430071

-- Define the context of convex polyhedron with vertices (V), edges (E), and faces (F)
structure ConvexPolyhedron :=
  (V : ℕ) -- number of vertices
  (E : ℕ) -- number of edges
  (F : ℕ) -- number of faces
  (convex : Prop) -- property stating the polyhedron is convex

-- Euler characteristic theorem for convex polyhedra
theorem euler_characteristic_convex_polyhedron (P : ConvexPolyhedron) (h : P.convex) : P.V - P.E + P.F = 2 :=
sorry

end euler_characteristic_convex_polyhedron_l430_430071


namespace find_extrema_l430_430625

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x + 6

theorem find_extrema :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → f x ≤ 6) ∧
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 4 ∧ f x = 6) ∧
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → 2 ≤ f x) ∧
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 4 ∧ f x = 2) :=
by sorry

end find_extrema_l430_430625


namespace sqrt_mul_eq_l430_430313

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l430_430313


namespace area_of_annulus_equals_pi_s_squared_l430_430730

-- Definitions from the problem conditions
variables (r s t : ℝ)
variable (h : r > s)
variable (hc : t = 2 * s)

-- Define the area of the annulus
noncomputable def area_of_annulus (r s : ℝ) : ℝ :=
  π * (r^2 - s^2)

-- The proof statement
theorem area_of_annulus_equals_pi_s_squared (h₁ : r = real.sqrt(s^2 + (t/2)^2)) (h₂ : t = 2 * s) : 
  area_of_annulus r s = π * s^2 := by 
  sorry

end area_of_annulus_equals_pi_s_squared_l430_430730


namespace sqrt_mul_eq_6_l430_430166

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l430_430166


namespace hounds_score_points_l430_430467

theorem hounds_score_points (x y : ℕ) (h_total : x + y = 82) (h_margin : x - y = 18) : y = 32 :=
sorry

end hounds_score_points_l430_430467


namespace sqrt_mul_simp_l430_430214

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l430_430214


namespace sqrt_mul_l430_430179

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l430_430179


namespace tangent_of_angle_EAF_l430_430890

theorem tangent_of_angle_EAF 
  (A B C D E F : Type) 
  [rect : Rectangle A B C D] 
  [triangle : Triangle A E F] 
  (h1 : E ∈ Segment B C) 
  (h2 : F ∈ Segment C D) 
  (k : ℝ)
  (h_ratio_AB_BC : AB / BC = k)
  (h_ratio_BE_EC : BE / EC = k)
  (h_ratio_CF_FD : CF / FD = k) :
  tan (angle A E F) = (k^2 + k + 1) / ((1 + k)^2) :=
sorry

end tangent_of_angle_EAF_l430_430890


namespace sqrt_mul_simplify_l430_430273

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l430_430273


namespace distance_between_foci_l430_430362

theorem distance_between_foci (x y : ℝ) (h : 9 * x ^ 2 + y ^ 2 = 900) : 
  ∃ c : ℝ, c = 40 * real.sqrt 2 :=
by
  sorry

end distance_between_foci_l430_430362


namespace total_students_l430_430590

theorem total_students (n_groups : ℕ) (students_in_group : ℕ → ℕ)
    (h1 : n_groups = 6)
    (h2 : ∃ n : ℕ, (students_in_group n = 13) ∧ (finset.filter (λ g, students_in_group g = 13) (finset.range n_groups)).card = 4)
    (h3 : ∀ i j, i < n_groups → j < n_groups → abs (students_in_group i - students_in_group j) ≤ 1) :
    (∃ N, N = 76 ∨ N = 80) :=
begin
    sorry
end

end total_students_l430_430590


namespace distance_between_trees_l430_430466

theorem distance_between_trees (length_yard : ℕ) (num_trees : ℕ) (dist : ℕ) :
  length_yard = 275 → num_trees = 26 → dist = length_yard / (num_trees - 1) → dist = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  assumption

end distance_between_trees_l430_430466


namespace factorize_expression_l430_430352

variable {R : Type*} [CommRing R] (a b : R)

theorem factorize_expression : 2 * a^2 * b - 4 * a * b + 2 * b = 2 * b * (a - 1)^2 :=
by
  sorry

end factorize_expression_l430_430352


namespace common_difference_is_three_l430_430473

-- Define the arithmetic sequence and the conditions
def arith_seq (a : ℕ → ℤ) (d : ℤ) := ∀ n, a (n + 1) = a n + d

variables {a : ℕ → ℤ} {d : ℤ}

-- Define the specific conditions given in the problem
def a1 := (a 0 = 2)
def a3 := (a 2 = 8)

-- State the theorem
theorem common_difference_is_three (h_seq : arith_seq a d) (h_a1 : a1) (h_a3 : a3) : d = 3 := 
sorry

end common_difference_is_three_l430_430473


namespace sqrt_mul_sqrt_eq_six_l430_430286

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l430_430286


namespace fraction_simplification_l430_430053

theorem fraction_simplification (b : ℝ) (hb : b = 3) : 
  (3 * b⁻¹ + b⁻¹ / 3) / b^2 = 10 / 81 :=
by
  rw [hb]
  sorry

end fraction_simplification_l430_430053


namespace range_of_x_l430_430798

noncomputable def prop_p (x : ℝ) := |x - 4| ≤ 6
noncomputable def prop_q (x : ℝ) := x^2 + 3 * x ≥ 0

theorem range_of_x (x : ℝ) : ¬(prop_p x ∧ prop_q x) ∧ ¬(¬prop_p x) → x ∈ set.Icc (-2 : ℝ) 0 :=
by sorry

end range_of_x_l430_430798


namespace number_of_solutions_l430_430756

theorem number_of_solutions : 
  {x : ℤ // -5 * x ≥ 3 * x + 12} ∩ 
  {x : ℤ // -3 * x ≤ 15} ∩ 
  {x : ℤ // -6 * x ≥ x + 26} = {x : ℤ | x = -5 ∨ x = -4} :=
by
  sorry

end number_of_solutions_l430_430756


namespace volume_cone_l430_430985

open Real

/--
Given the angle between two generators of a cone (α), 
the angle between a plane through two generators and the base of the cone (β), 
and the height of the cone (h), 
the volume of the cone is given by:
V = (π * h^3 * (cos^2(β) + tan^2(α / 2))) / (3 * sin^2(β)).
-/
theorem volume_cone (α β h : ℝ) : 
  let V := (π * h^3 * (cos β ^ 2 + tan (α / 2) ^ 2)) / (3 * (sin β) ^ 2) in
  V = (π * h^3 * (cos β ^ 2 + tan (α / 2) ^ 2)) / (3 * (sin β) ^ 2) :=
by
  -- Proof is not provided, as it is skipped with sorry.
  sorry

end volume_cone_l430_430985


namespace expected_value_finite_l430_430512

section
variables {X : ℝ → ℝ} {P : Set ℝ → ℝ}

-- Let \(X\) be a random variable with \(P\) as its probability distribution
-- Assume the condition \( \varlimsup_{n} \frac{\mathrm{P}(|X|>2n)}{\mathrm{P}(|X|>n)}<\frac{1}{2} \)
def condition (n : ℕ) : Prop := 
  (limsup (λ n : ℕ, P{ x : ℝ | |X x| > 2*n } / P{ x : ℝ | |X x| > n })) < 1 / 2

-- Prove that \( \mathbf{E}|X|<\infty \)
theorem expected_value_finite (h : ∀ n, condition n) : ∫ x, P {|X x|} x < ∞ :=
by
  sorry
end

end expected_value_finite_l430_430512


namespace smallest_perfect_square_divisible_by_3_and_5_l430_430045

theorem smallest_perfect_square_divisible_by_3_and_5 : ∃ (n : ℕ), n > 0 ∧ (∃ (m : ℕ), n = m * m) ∧ (n % 3 = 0) ∧ (n % 5 = 0) ∧ n = 225 :=
by
  sorry

end smallest_perfect_square_divisible_by_3_and_5_l430_430045


namespace major_premise_wrong_l430_430975

variables {α : Type*} {a b : α} {plane : set α}

-- Definitions of the conditions
def line_not_in_plane (b : α) (plane : set α) : Prop := ¬(b ∈ plane)
def line_in_plane (a : α) (plane : set α) : Prop := a ∈ plane
def line_parallel_to_plane (b : α) (plane : set α) : Prop := sorry -- Define appropriately for parallelism

-- The theorem statement based on the problem
theorem major_premise_wrong (h1 : line_not_in_plane b plane) (h2 : line_in_plane a plane) (h3 : line_parallel_to_plane b plane) :
  false := sorry

end major_premise_wrong_l430_430975


namespace devin_iff_l430_430075
open scoped Topology

-- Define the Devin property for sequences
def isDevin (x : ℕ → ℝ) : Prop := 
  ∀ f : ℝ → ℝ, ContinuousOn f (set.Icc 0 1) →
  Tendsto (λ n, (1 / n) * (∑ i in Finset.range n, f (x i)))
  atTop (𝓝 (interval_integral (λ x, f x) (0 : ℝ) 1))

-- Main theorem
theorem devin_iff (x : ℕ → ℝ) (h_bound : ∀ n, 0 ≤ x n ∧ x n ≤ 1) :
  isDevin x ↔ (∀ k : ℕ, Tendsto (λ n, (1 / n) * (∑ i in Finset.range n, (x i) ^ k)) 
  atTop (𝓝 (1 / (k + 1)))) :=
sorry

end devin_iff_l430_430075


namespace known_number_l430_430023

theorem known_number (A B : ℕ) (h_hcf : 1 / (Nat.gcd A B) = 1 / 15) (h_lcm : 1 / Nat.lcm A B = 1 / 312) (h_B : B = 195) : A = 24 :=
by
  -- Skipping proof
  sorry

end known_number_l430_430023


namespace retailer_percentage_profit_l430_430098

-- Definitions
def wholesale_price : ℝ := 90
def retail_price : ℝ := 120
def discount_rate : ℝ := 0.10

-- The selling price after discount
def selling_price : ℝ := retail_price * (1 - discount_rate)

-- The profit made by the retailer
def profit : ℝ := selling_price - wholesale_price

-- The percentage profit
def percentage_profit : ℝ := (profit / wholesale_price) * 100

-- The proof statement
theorem retailer_percentage_profit : percentage_profit = 20 := by
  sorry

end retailer_percentage_profit_l430_430098


namespace smallest_n_partition_condition_l430_430501

theorem smallest_n_partition_condition (n : ℕ) (T : Set ℕ) (hT : T = {5, 6, 7, ..., n}) :
  ∃ n, ∀ A B : Set ℕ, A ∪ B = T → A ∩ B = ∅ → (∃ a b c ∈ A, a + b = c) ∨ (∃ a b c ∈ B, a + b = c) ↔ n = 625 := 
begin
  sorry,
end

end smallest_n_partition_condition_l430_430501


namespace sort_volumes_eventually_correct_order_maximum_sorting_steps_l430_430698

variables (n : ℕ)
variables (f : Fin n → Fin n) -- This maps from positions to positions

-- Part (a): Proof that all volumes will eventually be in the correct order
theorem sort_volumes_eventually_correct_order :
  ∃ (steps : ℕ → list (Fin n)), 
    (∀ k, steps k ≠ list.range n → steps (k + 1)) ∧
    (∀ k, steps k = list.range n → (∃ m, steps m = list.range n)) :=
sorry

-- Part (b): Proof for the maximum number of steps to sort the volumes
theorem maximum_sorting_steps :
  (∃ (steps : ℕ → list (Fin n)), 
    ∀ k, 
    steps k ≠ list.range n → 
    steps (k + 1) ≠ list.range n → 
    ∑ k, (steps k ≠ list.range n) ≤ 2^(n-1) - 1) :=
sorry

end sort_volumes_eventually_correct_order_maximum_sorting_steps_l430_430698


namespace sqrt3_mul_sqrt12_eq_6_l430_430220

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l430_430220


namespace max_borrowed_l430_430468

noncomputable def max_books_borrowed 
    (total_students : ℕ)
    (borrowed_0 : ℕ)
    (borrowed_1 : ℕ)
    (borrowed_2 : ℕ)
    (at_least_3_books : ℕ → ℕ)
    (average_books_per_student : ℕ)
    : ℕ :=
let total_books_borrowed := total_students * average_books_per_student in
let already_borrowed := borrowed_0 * 0 + borrowed_1 * 1 + borrowed_2 * 2 in
let remaining_students := total_students - (borrowed_0 + borrowed_1 + borrowed_2) in
let remaining_books := total_books_borrowed - already_borrowed in
remaining_books - (remaining_students - 1) * 3

theorem max_borrowed 
    (total_students : ℕ := 20)
    (borrowed_0 : ℕ := 3)
    (borrowed_1 : ℕ := 9)
    (borrowed_2 : ℕ := 4)
    (average_books_per_student : ℕ := 2)
    : max_books_borrowed total_students borrowed_0 borrowed_1 borrowed_2 (λ _, 3) average_books_per_student = 14 := 
by 
    rw [max_books_borrowed, ← add_assoc],
    sorry

end max_borrowed_l430_430468


namespace limit_exp_one_over_n_equals_e_l430_430487

theorem limit_exp_one_over_n_equals_e :
  tendsto (λ n : ℕ, (1 + 1 / (n : ℝ)) ^ n) atTop (𝓝 (2.71828 : ℝ)) :=
sorry

end limit_exp_one_over_n_equals_e_l430_430487


namespace number_of_five_digit_palindromes_l430_430091

-- Define what constitutes a five-digit palindrome
def is_five_digit_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  n ≥ 10000 ∧ n < 100000 ∧ digits = digits.reverse

-- Statement of the proof problem
theorem number_of_five_digit_palindromes : (finset.filter is_five_digit_palindrome (finset.range 100000)).card = 900 :=
sorry

end number_of_five_digit_palindromes_l430_430091


namespace sqrt_mul_simp_l430_430202

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l430_430202


namespace total_students_l430_430566

noncomputable def totalStudentsOptions (groups totalGroups specificGroupCount specificGroupSize otherGroupSizes : ℕ) : Set ℕ :=
  if totalGroups = 6 ∧ specificGroupCount = 4 ∧ specificGroupSize = 13 ∧ (otherGroupSizes = 12 ∨ otherGroupSizes = 14) then
    {52 + 2 * otherGroupSizes}
  else
    ∅

theorem total_students :
  totalStudentsOptions 6 4 13 12 = {76} ∧ totalStudentsOptions 6 4 13 14 = {80} :=
by
  -- This is where the proof would go, but we're skipping it as per instructions
  sorry

end total_students_l430_430566


namespace total_students_appeared_l430_430873

/--
In a practice paper at 2iim.com, questions were given from 5 topics. Out of the appearing students,
10% passed in all topics, 10% did not pass in any, 20% passed in one topic only, 25% in two topics only,
24% of the total students passed 4 topics only, and 500 students passed in 3 topics only.
Prove that the total number of students who appeared in the examination is approximately 4545.
-/
theorem total_students_appeared (T : ℝ) (h1 : 0.10 * T) (h2 : 0.10 * T) (h3 : 0.20 * T) (h4 : 0.25 * T)
  (h5 : 0.24 * T) (h6 : T - (0.10 * T + 0.10 * T + 0.20 * T + 0.25 * T + 0.24 * T) = 500) :
  T ≈ 4545 :=
begin
  sorry
end

end total_students_appeared_l430_430873


namespace time_to_cross_platform_l430_430697

/-- Definitions of the conditions in the problem. -/
def train_length : ℕ := 1500
def platform_length : ℕ := 1800
def time_to_cross_tree : ℕ := 100
def train_speed : ℕ := train_length / time_to_cross_tree
def total_distance : ℕ := train_length + platform_length

/-- Proof statement: The time for the train to pass the platform. -/
theorem time_to_cross_platform : (total_distance / train_speed) = 220 := by
  sorry

end time_to_cross_platform_l430_430697


namespace calculate_product_l430_430134

theorem calculate_product : 3^6 * 4^3 = 46656 := by
  sorry

end calculate_product_l430_430134


namespace sqrt_mult_eq_six_l430_430322

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l430_430322


namespace sqrt_mul_simp_l430_430216

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l430_430216


namespace sqrt_mult_simplify_l430_430238

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l430_430238


namespace sqrt_mul_eq_l430_430303

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l430_430303


namespace reversed_last_two_digits_l430_430957

theorem reversed_last_two_digits (n : ℕ) : 
  (n = 13) → 
  (let n_squared := n^2,
       n_squared_last_two_digits := n_squared % 100,
       n_plus_one_squared := (n + 1)^2,
       n_plus_one_squared_last_two_digits := n_plus_one_squared % 100 in
  n_squared_last_two_digits = (n_plus_one_squared_last_two_digits % 10) * 10 + (n_plus_one_squared_last_two_digits / 10)) :=
by {
  intro hn13,
  rw hn13,
  sorry
}

end reversed_last_two_digits_l430_430957


namespace solve_for_r_l430_430942

theorem solve_for_r : ∃ r : ℚ, 23 - 5 = 3 * r + 2 ∧ r = 16 / 3 :=
by 
  use 16 / 3
  split
  { sorry }
  { refl }

end solve_for_r_l430_430942


namespace candy_distribution_l430_430792

/-- Frank had 42 pieces of candy. He put them equally into 2 bags. -/
theorem candy_distribution (candies : ℕ) (bags : ℕ) (h₁ : candies = 42) (h₂ : bags = 2)
  : candies / bags = 21 :=
by
  intro candies
  intro bags
  intro h₁
  intro h₂
  rw [h₁, h₂]
  exact sorry

end candy_distribution_l430_430792


namespace sqrt_mul_eq_l430_430299

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l430_430299


namespace man_upstream_rate_l430_430716

theorem man_upstream_rate (rate_downstream : ℝ) (rate_still_water : ℝ) (rate_current : ℝ) 
    (h1 : rate_downstream = 32) (h2 : rate_still_water = 24.5) (h3 : rate_current = 7.5) : 
    rate_still_water - rate_current = 17 := 
by 
  sorry

end man_upstream_rate_l430_430716


namespace perpendicular_medians_iff_l430_430936

variables (A B C A1 B1 : Type) [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C]
  [EuclideanGeometry A1] [EuclideanGeometry B1]

-- Definitions and conditions
definition is_median (A B1 : EuclideanGeometry) (C: EuclideanGeometry) : Prop := sorry -- definition of a median

axiom midpoint_def (A B1 : EuclideanGeometry) (C: EuclideanGeometry) : Prop := sorry -- definition of a midpoint

theorem perpendicular_medians_iff (A B C A1 B1 : EuclideanGeometry)
  (hA1_midpoint : midpoint_def A1 B C)
  (hB1_midpoint : midpoint_def B1 A C)
  (h_median_AA1 : is_median A A1 C)
  (h_median_BB1 : is_median B B1 C):
  (perpendicular A A1 B B1) ↔ (a^2 + b^2 = 5 * (c^2)) :=
sorry

end perpendicular_medians_iff_l430_430936


namespace angle_between_vectors_l430_430435

open Real

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1^2 + v.2^2)

def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2

theorem angle_between_vectors 
  (x y : ℝ)
  (h_parallel : 3 * x - 4 * 9 = 0)
  (h_perpendicular : 9 * 4 + 4 * y = 0) :
  x = 12 ∧ y = -3 ∧
  let a : ℝ × ℝ := (3, 4)
      b : ℝ × ℝ := (9, 12)
      c : ℝ × ℝ := (4, -3)
      m : ℝ × ℝ := (2 * 3 - 9, 2 * 4 - 12)
      n : ℝ × ℝ := (3 + 4, 4 + (-3))
  in let cos_angle := (dot_product m n) / ((vector_magnitude m) * (vector_magnitude n))
  in cos_angle = -√2/2 ∧ acos cos_angle = 3 * π / 4 :=
by sorry

end angle_between_vectors_l430_430435


namespace inequality_relation_l430_430799

open Real

theorem inequality_relation (x : ℝ) :
  ¬ ((∀ x, (x - 1) * (x + 3) < 0 → (x + 1) * (x - 3) < 0) ∧
     (∀ x, (x + 1) * (x - 3) < 0 → (x - 1) * (x + 3) < 0)) := 
by
  sorry

end inequality_relation_l430_430799


namespace ab_c_value_l430_430947

noncomputable def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem ab_c_value (a b c : ℝ) :
  (∀ x : ℝ, f (x + 2) = 2 * x^2 + 6 * x + 5) ∧  (∀ x : ℝ, f x = a * x^2 + b * x + c) → a + b + c = 1 :=
by
  intro h
  sorry

end ab_c_value_l430_430947


namespace horizon_distance_ratio_l430_430700

def R : ℝ := 6000000
def h1 : ℝ := 1
def h2 : ℝ := 2

noncomputable def distance_to_horizon (R h : ℝ) : ℝ :=
  Real.sqrt (2 * R * h)

noncomputable def d1 : ℝ := distance_to_horizon R h1
noncomputable def d2 : ℝ := distance_to_horizon R h2

theorem horizon_distance_ratio : d2 / d1 = Real.sqrt 2 :=
  sorry

end horizon_distance_ratio_l430_430700


namespace count_integer_triangles_with_perimeter_17_l430_430850

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def is_integer_triangle_with_odd_side (a b c : ℕ) : Prop :=
  a + b + c = 17 ∧ (a % 2 = 1 ∨ b % 2 = 1 ∨ c % 2 = 1) ∧ is_triangle a b c

theorem count_integer_triangles_with_perimeter_17 : { (a, b, c) : ℕ × ℕ × ℕ // is_integer_triangle_with_odd_side a b c }.card = 5 :=
by
  sorry

end count_integer_triangles_with_perimeter_17_l430_430850


namespace degree_of_monic_poly_l430_430861

variable (p : ℝ → ℝ)
variable (is_monic : ∀ x: ℝ, leadingCoeff (polynomial.mk x) = 1)
variable (integer_n_lemma : ∀ n : ℕ, n > 0 → ∃ m : ℕ, m > 0 ∧ p m = 2^n)

theorem degree_of_monic_poly (hp : polynomial.monic p ∧ (integer_n_lemma p)) : polynomial.degree p = 1 := 
sorry

end degree_of_monic_poly_l430_430861


namespace A_and_C_complete_remaining_work_in_2_point_4_days_l430_430701

def work_rate_A : ℚ := 1 / 12
def work_rate_B : ℚ := 1 / 15
def work_rate_C : ℚ := 1 / 18
def work_completed_B_in_10_days : ℚ := (10 : ℚ) * work_rate_B
def remaining_work : ℚ := 1 - work_completed_B_in_10_days
def combined_work_rate_AC : ℚ := work_rate_A + work_rate_C
def time_to_complete_remaining_work : ℚ := remaining_work / combined_work_rate_AC

theorem A_and_C_complete_remaining_work_in_2_point_4_days :
  time_to_complete_remaining_work = 2.4 := 
sorry

end A_and_C_complete_remaining_work_in_2_point_4_days_l430_430701


namespace five_digit_palindromes_count_l430_430093

def is_palindrome (n : ℕ) : Prop :=
  let str := n.toString
  str = str.reverse

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

theorem five_digit_palindromes_count : { n : ℕ // is_five_digit n ∧ is_palindrome n }.card = 900 := 
  sorry

end five_digit_palindromes_count_l430_430093


namespace find_N_l430_430076

noncomputable def N : ℕ := 1156

-- Condition 1: N is a perfect square
axiom N_perfect_square : ∃ n : ℕ, N = n^2

-- Condition 2: All digits of N are less than 7
axiom N_digits_less_than_7 : ∀ d, d ∈ [1, 1, 5, 6] → d < 7

-- Condition 3: Adding 3 to each digit yields another perfect square
axiom N_plus_3_perfect_square : ∃ m : ℕ, (m^2 = 1156 + 3333)

theorem find_N : N = 1156 :=
by
  -- Proof goes here
  sorry

end find_N_l430_430076


namespace graph_passes_fixed_point_l430_430016

noncomputable def satisfies_properties (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : Prop :=
  let x : ℝ := 3 / 2 in
  f x = a^(2 * x - 3) - 5 ∧ f x = -4

theorem graph_passes_fixed_point : ∀ (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1), satisfies_properties a h1 h2 :=
by
  intro a
  intro h1
  intro h2
  let x := 3 / 2
  have h3 : 2 * x - 3 = 0 := by
    unfold x
    linarith
  refine ⟨_, _⟩
  show f x = a^(2 * x - 3) - 5, by
    sorry  -- The actual detailed proof step would go here
  show f x = -4, by
    rw [h3]
    sorry  -- The complete step of proving f is -4 at x = 3/2

end graph_passes_fixed_point_l430_430016


namespace tile_one_corner_removed_tile_opposite_corners_removed_tile_non_opposite_corners_removed_l430_430485

def can_tile (board : List (List (Option Bool))) (dominoes : Nat) : Prop :=
  List.join board |>.filterMap id |>.length = dominoes * 2

def is_checkerboard (board : List (List (Option Bool))) : Prop :=
  -- Check if the board is properly colored as a checkerboard
  True -- Assume its always true

theorem tile_one_corner_removed : ¬ can_tile (List.replicate 7 (List.replicate 8 (some tt) ++ List.replicate 1 none)) 31 :=
by
  sorry

theorem tile_opposite_corners_removed : ¬ can_tile (List.replicate 8 (List.replicate 2 (some tt) ++ List.replicate 6 (some tt) ++ List.replicate 2 none ++ List.replicate 6 (some tt))) 31 :=
by
  sorry

theorem tile_non_opposite_corners_removed : can_tile (List.replicate 8 (List.replicate 2 (some tt) ++ List.replicate 6 (some tt) ++ List.replicate 2 (some ff) ++ List.replicate 6 (some tt))) 31 :=
by
  sorry

#eval is_checkerboard (List.replicate 8 (List.replicate 8 (some tt))) -- just an example to check board setup


end tile_one_corner_removed_tile_opposite_corners_removed_tile_non_opposite_corners_removed_l430_430485


namespace sqrt3_mul_sqrt12_eq_6_l430_430218

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l430_430218


namespace quadratic_function_conditions_l430_430386

noncomputable def quadratic_function_example (x : ℝ) : ℝ :=
  -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_conditions :
  quadratic_function_example 1 = 0 ∧
  quadratic_function_example 5 = 0 ∧
  quadratic_function_example 3 = 10 :=
by
  sorry

end quadratic_function_conditions_l430_430386


namespace domain_of_f_l430_430343

noncomputable def f (x : ℝ) : ℝ := (1 / (x - 5)) + (1 / (x^2 - 4)) + (1 / (x^3 - 27))

theorem domain_of_f :
  ∀ x : ℝ, x ≠ 5 ∧ x ≠ 2 ∧ x ≠ -2 ∧ x ≠ 3 ↔
          ∃ y : ℝ, f y = f x :=
by
  sorry

end domain_of_f_l430_430343


namespace common_difference_is_three_l430_430472

-- Define the arithmetic sequence and the conditions
def arith_seq (a : ℕ → ℤ) (d : ℤ) := ∀ n, a (n + 1) = a n + d

variables {a : ℕ → ℤ} {d : ℤ}

-- Define the specific conditions given in the problem
def a1 := (a 0 = 2)
def a3 := (a 2 = 8)

-- State the theorem
theorem common_difference_is_three (h_seq : arith_seq a d) (h_a1 : a1) (h_a3 : a3) : d = 3 := 
sorry

end common_difference_is_three_l430_430472


namespace total_students_possible_l430_430560

theorem total_students_possible (A B : ℕ) :
  (4 * 13) + 2 * A = 76 ∨ (4 * 13) + 2 * B = 80 :=
by
  -- Let N be the total number of students
  let N := (4 * 13)
  -- Given that the number of students in the remaining 2 groups differs by no more than 1
  have h : A = 12 ∨ B = 14 := sorry
  -- Prove the possible values
  exact or.inl (N + 2 * 12 = 76) <|> or.inr (N + 2 * 14 = 80)

end total_students_possible_l430_430560


namespace measure_of_x_is_4_l430_430613

theorem measure_of_x_is_4
  (rectangle_length : ℕ := 12)
  (rectangle_width : ℕ := 12)
  (polygon_count : ℕ := 2)
  (side_of_square : ℕ := 12)
  (x : ℕ) :
  -- conditions
  rectangle_length * rectangle_width = 144 →
  2 * (side_of_square ^ 2) / polygon_count = 144 →
  -- question, i.e., the proof that x == 4
  x = side_of_square / 3 → x = 4 :=
begin
  sorry
end

end measure_of_x_is_4_l430_430613


namespace solve_system_and_find_6a_plus_b_l430_430844

theorem solve_system_and_find_6a_plus_b (x y a b : ℝ)
  (h1 : 3 * x - 2 * y + 20 = 0)
  (h2 : 2 * x + 15 * y - 3 = 0)
  (h3 : a * x - b * y = 3) :
  6 * a + b = -3 := by
  sorry

end solve_system_and_find_6a_plus_b_l430_430844


namespace monotonic_increasing_on_interval_min_value_on_interval_max_value_on_interval_l430_430425

noncomputable def f (x : ℝ) : ℝ := 1 - (3 / (x + 2))

theorem monotonic_increasing_on_interval :
  ∀ (x₁ x₂ : ℝ), 3 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 5 → f x₁ < f x₂ := sorry

theorem min_value_on_interval :
  ∃ (x : ℝ), x = 3 ∧ f x = 2 / 5 := sorry

theorem max_value_on_interval :
  ∃ (x : ℝ), x = 5 ∧ f x = 4 / 7 := sorry

end monotonic_increasing_on_interval_min_value_on_interval_max_value_on_interval_l430_430425


namespace inequality_solution_l430_430602

theorem inequality_solution (x : ℝ) : (|x - 1| + |x - 2| > 5) ↔ (x ∈ (-∞, -1) ∪ (4, ∞)) :=
by
  sorry

end inequality_solution_l430_430602


namespace sqrt_mul_simp_l430_430206

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l430_430206


namespace sqrt_mul_simplify_l430_430278

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l430_430278


namespace geoffrey_remaining_money_l430_430380

theorem geoffrey_remaining_money (gma aunt uncle wallet cost_per_game num_games : ℕ)
  (h_gma : gma = 20)
  (h_aunt : aunt = 25)
  (h_uncle : uncle = 30)
  (h_wallet : wallet = 125)
  (h_cost_per_game : cost_per_game = 35)
  (h_num_games : num_games = 3) :
  wallet - num_games * cost_per_game = 20 :=
by
  -- Given conditions
  have h_gifts := h_gma + h_aunt + h_uncle
  have h_initial_money := h_wallet - h_gifts
  -- Calculate the total cost of games
  have h_total_cost := h_num_games * h_cost_per_game
  -- Calculate the remaining money
  have h_remaining := h_wallet - h_total_cost
  -- Simplification steps (omitted)
  sorry

end geoffrey_remaining_money_l430_430380


namespace sqrt_mult_eq_six_l430_430316

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l430_430316


namespace length_of_median_CN_l430_430894

-- Definitions
universe u
variable {α : Type u}

structure Triangle (α : Type u) :=
(A B C : α)

structure Median (α : Type u) :=
(triangle : Triangle α)
(point : α)

structure Centroid (α : Type u) :=
(triangle : Triangle α)
(point : α)

-- Triangle with given properties.
variables {ABC : Triangle α}
variables {AL BM CN : Median α}
variables {K : Centroid α}
variables {a : ℝ}

-- Conditions
def triangle_with_medians_and_centroid (ABC : Triangle α) (AL BM CN : Median α) (K : Centroid α) : Prop :=
  K.triangle = ABC ∧ K.point = intersection_of_medians AL BM ∧ -- K is the intersection of AL and BM
  median_point AL.point = mid_point_of (ABC.B) (ABC.C) ∧ -- AL's midpoint L
  median_point BM.point = mid_point_of (ABC.A) (ABC.C) ∧ -- BM's midpoint M
  distance (ABC.A) (ABC.B) = a

-- Equivalent proof problem in Lean
theorem length_of_median_CN (ABC : Triangle α) (AL BM CN : Median α) (K : Centroid α)
  (h : triangle_with_medians_and_centroid ABC AL BM CN K) :
  distance CN.point (ABC.C) = a * (sqrt 3) / 2 :=
by sorry

end length_of_median_CN_l430_430894


namespace sqrt_mul_l430_430175

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l430_430175


namespace quadrilateral_area_l430_430078

def Point := (ℝ × ℝ)
def Triangle := (Point × Point × Point)

noncomputable def midpoint (A B : Point) : Point :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def length (A B : Point) : ℝ :=
  sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

noncomputable def area_of_triangle (A B C : Point) : ℝ :=
  abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2

noncomputable def intersection (A B C D : Point) : Point :=
  sorry  -- Intersection of line segments AB and CD (detailed calculation omitted)

noncomputable def area_of_BEFC (B A C D E F : Point) : ℝ :=
  let area_BEF := area_of_triangle B E F
  let area_CEF := area_of_triangle C E F
  area_BEF + area_CEF

theorem quadrilateral_area {A B C D E : Point}
  (h1 : is_right_triangle A B C)
  (h2 : length A B = 3)
  (h3 : length B C = 4)
  (h4 : D = (B.1, B.2 + 8))
  (h5 : E = midpoint A B)
  (F : Point := intersection E D A C) :
  area_of_BEFC B A C D E F = -- expected area here :=
sorry

end quadrilateral_area_l430_430078


namespace partition_theorem_l430_430358

-- Definitions for the conditions from the problem

def is_partition_possible (a b : ℝ) : Prop :=
  ∃ r : ℝ, ∃ a' b' : ℤ, 
    a = r * a' ∧ b = r * b' ∧
    (a' ≠ b') ∧ 
    a' * b' ≡ 2 [MOD 3] ∧
    (∀ n : ℤ, ∃ (A1 A2 A3 : set ℤ), 
      (n ∈ A1 ∧ (n + a) ∈ A2 ∧ (n + b) ∈ A3))

-- The theorem statement in Lean

theorem partition_theorem (a b : ℝ) (h1 : ¬ int.Cast a) (h2 : ¬ int.Cast b) (h3 : a ≠ b) :
  is_partition_possible a b :=
sorry

end partition_theorem_l430_430358


namespace dot_product_eq_neg6_l430_430825

noncomputable theory

variables {a b : ℝ × ℝ}

-- Condition 1: The projection of vector a in the direction of vector b is -2
def projection_condition (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.1 + a.2 * b.2) / (real.sqrt (b.1 ^ 2 + b.2 ^ 2)) = -2

-- Condition 2: The magnitude of vector b is 3
def magnitude_b_condition (b : ℝ × ℝ) : Prop :=
  real.sqrt (b.1 ^ 2 + b.2 ^ 2) = 3

-- Proving the dot product a ⬝ b equals -6
theorem dot_product_eq_neg6 (h1 : projection_condition a b) (h2 : magnitude_b_condition b) :
  a.1 * b.1 + a.2 * b.2 = -6 :=
by
  sorry

end dot_product_eq_neg6_l430_430825


namespace hyperbola_inscribed_circle_properties_l430_430814

theorem hyperbola_inscribed_circle_properties :
  let F1 := (3, 0),
      F2 := (-3, 0),
      P : ℝ × ℝ := sorry, -- point on the hyperbola right branch distinct from vertex
      O := (0, 0),
      hyperbola_eq : ∀ x y, (x^2 / 9) - (y^2 / 4) = 1 → 
                      (x^2 / 9 = 1 → y = 0) ∧ (let a: ℝ := 3 in |P.fst - F1.fst| - |P.fst - F2.fst| = 2 * a),
      inscribed_circle_center (P F1 F2 : ℝ × ℝ) : ℝ × ℝ := sorry, -- center of inscribed circle
      inscribed_circle_radius (P F1 F2 : ℝ × ℝ) : ℝ := sorry, -- radius of inscribed circle
      M := (inscribed_circle_center P F1 F2) in
  M.1 = 3 ∧ 
  let center_inscribed_circle := inscribed_circle_center P F1 F2 in
  let inscribed_circle_eq := λ (x y: ℝ), 
      (x - center_inscribed_circle.1)^2 + 
      (y - center_inscribed_circle.2)^2 = (inscribed_circle_radius P F1 F2)^2 in 
  inscribed_circle_eq 3 0 := sorry

end hyperbola_inscribed_circle_properties_l430_430814


namespace sqrt_mul_l430_430182

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l430_430182


namespace production_today_l430_430787

def average_production (P : ℕ) (n : ℕ) := P / n

theorem production_today :
  ∀ (T P n : ℕ), n = 9 → average_production P n = 50 → average_production (P + T) (n + 1) = 54 → T = 90 :=
by
  intros T P n h1 h2 h3
  sorry

end production_today_l430_430787


namespace student_groups_l430_430542

theorem student_groups (N : ℕ) :
  (∃ (n : ℕ), n = 13 ∧ ∃ (m : ℕ), m ∈ {12, 14} ∧ N = 4 * 13 + 2 * m) → (N = 76 ∨ N = 80) :=
by
  intro h
  obtain ⟨n, hn, m, hm, hN⟩ := h
  rw [hn, hN]
  cases hm with h12 h14
  case inl =>
    simp [h12]
  case inr =>
    simp [h14]
  sorry

end student_groups_l430_430542


namespace profit_percentage_correct_l430_430103

variable (wholesalePrice : ℝ) (retailPrice : ℝ) (discountRate : ℝ)

def percentageProfit (wholesalePrice retailPrice discountRate : ℝ) : ℝ :=
  let discountAmount := discountRate * retailPrice
  let sellingPrice := retailPrice - discountAmount
  let profit := sellingPrice - wholesalePrice
  (profit / wholesalePrice) * 100

theorem profit_percentage_correct :
  wholesalePrice = 90 → retailPrice = 120 → discountRate = 0.10 → 
  percentageProfit wholesalePrice retailPrice discountRate = 20 := by
  intros
  unfold percentageProfit
  rw [H, H_1, H_2]
  norm_num
  sorry

end profit_percentage_correct_l430_430103


namespace largest_integer_less_than_log_l430_430366

theorem largest_integer_less_than_log :
  (⌊log 3 2023⌋ = 6) :=
sorry

end largest_integer_less_than_log_l430_430366


namespace price_relation_l430_430455

-- Defining the conditions
variable (TotalPrice : ℕ) (NumberOfPens : ℕ)
variable (total_price_val : TotalPrice = 24) (number_of_pens_val : NumberOfPens = 16)

-- Statement of the problem
theorem price_relation (y x : ℕ) (h_y : y = 3 / 2) : y = 3 / 2 * x := 
  sorry

end price_relation_l430_430455


namespace quarterly_to_annual_rate_l430_430345

theorem quarterly_to_annual_rate (annual_rate : ℝ) (quarterly_rate : ℝ) (n : ℕ) (effective_annual_rate : ℝ) : 
  annual_rate = 4.5 →
  quarterly_rate = annual_rate / 4 →
  n = 4 →
  effective_annual_rate = (1 + quarterly_rate / 100)^n →
  effective_annual_rate * 100 = 4.56 :=
by
  intros h1 h2 h3 h4
  sorry

end quarterly_to_annual_rate_l430_430345


namespace first_scenario_second_scenario_l430_430978

variables {A B C : Prop}
variables {involved : Prop} -- "No one besides A, B, and C was involved."
variables {C_with_A : Prop} -- "C never goes on a job without A."
variables {B_cannot_drive : Prop} -- "B does not know how to drive."

-- Definitions based on problem conditions
def theft_involved : Prop := involved ∧ (A ∨ B ∨ C)

def condition_C_with_A : Prop := C → A 

def condition_B_cannot_drive : Prop := B → ¬B_cannot_drive

-- Proof statement to prove A is guilty
theorem first_scenario (h1 : theft_involved) (h2 : condition_C_with_A) (h3 : condition_B_cannot_drive) : A := 
sorry

-- Definitions based on second problem conditions
variables {A_with_accomplice : Prop} -- "A never goes on a job without at least one accomplice."
variables {C_innocent : Prop} -- "C is not guilty."

def condition_A_with_accomplice : Prop := A → (B ∨ C)

def condition_C_innocent : Prop := ¬C

-- Proof statement to prove B is guilty
theorem second_scenario (h1: theft_involved) (h2: condition_A_with_accomplice) (h3: condition_C_innocent) : B := 
sorry

end first_scenario_second_scenario_l430_430978


namespace factors_of_144_that_are_multiples_of_6_l430_430853

theorem factors_of_144_that_are_multiples_of_6:
  let n := 144
  let is_a_factor (m : ℕ) : Prop := n % m = 0
  let is_a_multiple_of_6 (m : ℕ) : Prop := m % 6 = 0
  let factors := { m : ℕ | is_a_factor m }
  let multiples_of_6 := { m ∈ factors | is_a_multiple_of_6 m }
  multiples_of_6.card = 8 :=
begin
  sorry
end

end factors_of_144_that_are_multiples_of_6_l430_430853


namespace correct_propositions_l430_430830

-- Conditions as given in the problem
variable (k x y : ℝ)
axiom prop1 : k > 0 → ∃ r1 r2 : ℝ, r1^2 - 2*r1 - k = 0 ∧ r2^2 - 2*r2 - k = 0
axiom prop2 : (x + y ≠ 8) → (x ≠ 2 ∨ y ≠ 6)

-- The proposition that "If xy = 0, then at least one of x and y is 0" is always true, so there is no false proposition
-- We formally state that there is no proposition that contradicts this
axiom prop3 : ¬(∃ (x y : ℝ), xy = 0 ∧ (x ≠ 0 ∧ y ≠ 0))

-- We need to prove that the correct propositions are 1 and 2
theorem correct_propositions : [1, 2] = [1, 2] := 
by {
  -- Skipping the exhaustive proof details
  sorry
}

end correct_propositions_l430_430830


namespace PQ_perpendicular_to_KM_l430_430035

-- Definitions for the circles, points, and lines described in the conditions
variables {circle1 circle2 : Circle}
variables {A B K P Q M F : Point}
variable KM : Line

-- Assumptions based on the problem conditions
axiom intersect_at_A_B : Circle.Intersect circle1 circle2 A ∧ Circle.Intersect circle1 circle2 B
axiom K_on_circle1 : circle1.Contains K
axiom P_on_circle2 : circle2.Contains P
axiom Q_on_circle2 : circle2.Contains Q
axiom KA_intersects_circle2_at_P : Line.Intersect (Line.mk K A) circle2 P
axiom KB_intersects_circle2_at_Q : Line.Intersect (Line.mk K B) circle2 Q
axiom KM_diameter : line_segment K M = 2 * (Circle.radius circle1)
axiom KF_tangent : TangentToCircleAtPoint circle1 K F
axiom KF_perp_KM : Perpendicular KF KM

-- Goal to prove
theorem PQ_perpendicular_to_KM : Perpendicular (Line.mk P Q) KM :=
by
  sorry

end PQ_perpendicular_to_KM_l430_430035


namespace cos_A_cos_B_l430_430461

variable {A B C a b c : ℝ}
axiom angles_arithmetic_seq : 2 * B = A + C
axiom sides_geometric_seq : (2 * b) ^ 2 = (2 * a) * (2 * c)
axiom triangle_angle_sum : A + B + C = Real.pi

theorem cos_A_cos_B : ℝ :=
by
  have B_eq : B = Real.pi / 3 := sorry
  have cos_A : Real.cos A = 1 / 2 := sorry
  have cos_B : Real.cos B = 1 / 2 := sorry
  show Real.cos A * Real.cos B = 1 / 4 by sorry

end cos_A_cos_B_l430_430461


namespace sqrt_mul_eq_6_l430_430169

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l430_430169


namespace sqrt3_mul_sqrt12_eq_6_l430_430223

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l430_430223


namespace alice_ate_more_l430_430348

theorem alice_ate_more (cookies : Fin 8 → ℕ) (h_alice : cookies 0 = 8) (h_tom : cookies 7 = 1) :
  cookies 0 - cookies 7 = 7 :=
by
  -- Placeholder for the actual proof, which is not required here
  sorry

end alice_ate_more_l430_430348


namespace boxes_of_nuts_purchased_l430_430785

theorem boxes_of_nuts_purchased (b : ℕ) (n : ℕ) (bolts_used : ℕ := 7 * 11 - 3) 
    (nuts_used : ℕ := 113 - bolts_used) (total_nuts : ℕ := nuts_used + 6) 
    (nuts_per_box : ℕ := 15) (h_bolts_boxes : b = 7) 
    (h_bolts_per_box : ∀ x, b * x = 77) 
    (h_nuts_boxes : ∃ x, n = x * nuts_per_box)
    : ∃ k, n = k * 15 ∧ k = 3 :=
by
  sorry

end boxes_of_nuts_purchased_l430_430785


namespace sum_of_powers_mod_p_square_l430_430810

theorem sum_of_powers_mod_p_square (p : ℕ) [hp : Fact (Nat.Prime p)] (odd_p : p % 2 = 1) : 
  (∑ k in Finset.range (p - 1), k^(2*p - 1)) % (p^2) = ((p * (p + 1)) / 2) % (p^2) :=
by 
  sorry

end sum_of_powers_mod_p_square_l430_430810


namespace sqrt_mul_sqrt_eq_six_l430_430283

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l430_430283


namespace vector_dot_product_l430_430438

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (2, -2)

theorem vector_dot_product : (a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2)) = -1 := by
  -- skipping the proof
  sorry

end vector_dot_product_l430_430438


namespace A_inter_complement_B_eq_set_minus_one_to_zero_l430_430846

open Set

theorem A_inter_complement_B_eq_set_minus_one_to_zero :
  let U := @univ ℝ
  let A := {x : ℝ | x < 0}
  let B := {x : ℝ | x ≤ -1}
  A ∩ (U \ B) = {x : ℝ | -1 < x ∧ x < 0} := 
by
  sorry

end A_inter_complement_B_eq_set_minus_one_to_zero_l430_430846


namespace rotation_by_120_degrees_l430_430612

def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    [Real.cos θ, -Real.sin θ],
    [Real.sin θ, Real.cos θ]
  ]

theorem rotation_by_120_degrees :
  ∀ θ : ℝ, rotation_matrix θ = rotation_matrix (2 * Real.pi / 3) → θ = 2 * Real.pi / 3 :=
by
  intro θ h
  sorry

end rotation_by_120_degrees_l430_430612


namespace smallest_sum_of_squares_l430_430621

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 187) : x^2 + y^2 ≥ 205 := sorry

end smallest_sum_of_squares_l430_430621


namespace total_students_in_groups_l430_430533

theorem total_students_in_groups {N : ℕ} (h1 : ∃ g : ℕ → ℕ, (∀ i j, g i = 13 ∨ g j = 12 ∨ g j = 14) ∧ (∑ i in finset.range 6, g i) = N) : 
  N = 76 ∨ N = 80 :=
sorry

end total_students_in_groups_l430_430533


namespace sqrt_mul_sqrt_eq_six_l430_430297

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l430_430297


namespace fencing_required_l430_430063

theorem fencing_required (L W : ℕ) (hL : L = 40) (hA : 40 * W = 680) : 2 * W + L = 74 :=
by sorry

end fencing_required_l430_430063


namespace arithmetic_sequence_max_lambda_l430_430805

theorem arithmetic_sequence {a : ℕ → ℚ} (h1 : a 1 = 2) (h2 : ∀ n, a (n + 1) * a n = 2 * a n - 1) :
  ∀ n, (1 / (a n - 1)) = n := 
  sorry

theorem max_lambda {a : ℕ → ℚ} {b : ℕ → ℚ} {T : ℕ → ℚ} 
  (h1 : a 1 = 2)
  (h2 : ∀ n, a (n + 1) * a n = 2 * a n - 1)
  (h3 : ∃ a4 : ℚ, b 1 = 20 * a4)
  (h4 : ∀ n, b (n + 1) = a n * b n)
  (h5 : ∀ n, T n = (∑ i in Finset.range n, b (i + 1)))
  (h6 : ∀ n, 2 * T n + 400 ≥ λ * n) :
  ∃ λ, λ ≤ 225 := 
  sorry

end arithmetic_sequence_max_lambda_l430_430805


namespace equal_acutes_l430_430600

open Real

theorem equal_acutes (a b c : ℝ) (ha : 0 < a ∧ a < π / 2) (hb : 0 < b ∧ b < π / 2) (hc : 0 < c ∧ c < π / 2)
  (h1 : sin b = (sin a + sin c) / 2) (h2 : cos b ^ 2 = cos a * cos c) : a = b ∧ b = c := 
by
  -- We have to fill the proof steps here.
  sorry

end equal_acutes_l430_430600


namespace total_students_total_students_alt_l430_430549

def number_of_students (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 
    a + b = 6 ∧ 
    a = 4 ∧ 
    (∀ g, g = 13) ∧ 
    (b = 2 ∧ ((g = 12 ∨ g = 14) ∧ ∀ h, g - h ≤ 1)) ∧ 
    (n = 52 + 2 * 12 ∨ n = 52 + 2 * 14)

theorem total_students : ∃ n : ℕ, number_of_students n :=
by
  use 76
  sorry

theorem total_students_alt : ∃ n : ℕ, number_of_students n :=
by
  use 80
  sorry

end total_students_total_students_alt_l430_430549


namespace max_parallelograms_in_hexagon_l430_430852

theorem max_parallelograms_in_hexagon (side_hexagon side_parallelogram1 side_parallelogram2 : ℝ)
                                        (angle_parallelogram : ℝ) :
  side_hexagon = 3 ∧ side_parallelogram1 = 1 ∧ side_parallelogram2 = 2 ∧ angle_parallelogram = (π / 3) →
  ∃ n : ℕ, n = 12 :=
by 
  sorry

end max_parallelograms_in_hexagon_l430_430852


namespace Lisa_total_spoons_l430_430516

theorem Lisa_total_spoons:
  (∀ n_children n_spoons_per_child n_decorative_spoons n_large_spoons n_teaspoons : ℕ,
    n_children = 4 →
    n_spoons_per_child = 3 →
    n_decorative_spoons = 2 →
    n_large_spoons = 10 →
    n_teaspoons = 15 →
    n_children * n_spoons_per_child + n_decorative_spoons + n_large_spoons + n_teaspoons = 39) :=
by intros;
   contradiction

end Lisa_total_spoons_l430_430516


namespace geometric_sequence_condition_l430_430401

variables (b : ℝ)

def p := (1, b, 9) -- 1, b, 9 form a geometric sequence means b^2 = 9
def q := b = 3

theorem geometric_sequence_condition :
  (p b) → (q b) → false := sorry

end geometric_sequence_condition_l430_430401


namespace count_four_digit_even_numbers_l430_430994

theorem count_four_digit_even_numbers : 
  let digits := {0, 1, 2, 3, 4, 5}
  in ∃ n : ℕ, (∃ l : List ℕ, (l.all (λ d, d ∈ digits)) ∧ l.length = 4 ∧ (l.head! ≠ 0) ∧ (l.nth 3 % 2 = 0) ∧ (nat_of_list l ≤ 4310)) ∧ n = 110 :=
sorry

end count_four_digit_even_numbers_l430_430994


namespace sqrt3_mul_sqrt12_eq_6_l430_430253

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l430_430253


namespace sqrt_mult_eq_six_l430_430315

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l430_430315


namespace sqrt_mul_simplify_l430_430277

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l430_430277


namespace monomial_coeff_degree_product_l430_430822

theorem monomial_coeff_degree_product (m n : ℚ) (h₁ : m = -3/4) (h₂ : n = 4) : m * n = -3 := 
by
  sorry

end monomial_coeff_degree_product_l430_430822


namespace factorize_2x2_minus_4x_factorize_xy2_minus_2xy_plus_x_l430_430351

-- Problem 1
theorem factorize_2x2_minus_4x (x : ℝ) : 
  2 * x^2 - 4 * x = 2 * x * (x - 2) := 
by 
  sorry

-- Problem 2
theorem factorize_xy2_minus_2xy_plus_x (x y : ℝ) :
  x * y^2 - 2 * x * y + x = x * (y - 1)^2 :=
by 
  sorry

end factorize_2x2_minus_4x_factorize_xy2_minus_2xy_plus_x_l430_430351


namespace sqrt3_mul_sqrt12_eq_6_l430_430232

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l430_430232


namespace find_g_at_3_l430_430955

noncomputable def g (x : ℝ) := sorry

theorem find_g_at_3 (h : ∀ x : ℝ, g (3^x) + 2 * x * g (3^(-x)) = 2) : g 3 = 2 / 7 :=
sorry

end find_g_at_3_l430_430955


namespace function_relationship_value_of_x_l430_430488

variable {x y : ℝ}

-- Given conditions:
-- Condition 1: y is inversely proportional to x
def inversely_proportional (p : ℝ) (q : ℝ) (k : ℝ) : Prop := p = k / q

-- Condition 2: y(2) = -3
def specific_value (x_val y_val : ℝ) : Prop := y_val = -3 ∧ x_val = 2

-- Questions rephrased as Lean theorems:

-- The function relationship between y and x is y = -6 / x
theorem function_relationship (k : ℝ) (hx : x ≠ 0) 
  (h_inv_prop: inversely_proportional y x k) (h_spec : specific_value 2 (-3)) : k = -6 :=
by
  sorry

-- When y = 2, x = -3
theorem value_of_x (hx : x ≠ 0) (hy : y = 2)
  (h_inv_prop : inversely_proportional y x (-6)) : x = -3 :=
by
  sorry

end function_relationship_value_of_x_l430_430488


namespace sqrt_mult_eq_six_l430_430325

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l430_430325


namespace father_l430_430965

variable (S F : ℕ)

theorem father's_age (h1 : F = 3 * S + 3) (h2 : F + 3 = 2 * (S + 3) + 10) : F = 33 := by
  sorry

end father_l430_430965


namespace winning_pair_probability_l430_430385

-- Define the six cards
inductive Color
| Blue | Orange

inductive Label
| X | Y | Z

structure Card where
  color : Color
  label : Label

-- all the cards
def cards : Finset Card := 
  { ⟨Color.Blue, Label.X⟩, ⟨Color.Blue, Label.Y⟩, ⟨Color.Blue, Label.Z⟩,
    ⟨Color.Orange, Label.X⟩, ⟨Color.Orange, Label.Y⟩, ⟨Color.Orange, Label.Z⟩ }

-- Define winning pair predicate
def winning_pair (c1 c2 : Card) : Prop :=
  c1.label = c2.label ∨ c1.color = c2.color

open_locale classical

noncomputable def probability_winning_pair : ℚ :=
let total_pairs := (cards.card.choose 2) in
let winning_pairs := (Finset.filter (λ p, winning_pair p.1 p.2) (cards.pairs)).card in
winning_pairs / total_pairs

theorem winning_pair_probability :
  probability_winning_pair = 3 / 5 := 
sorry

end winning_pair_probability_l430_430385


namespace total_students_76_or_80_l430_430571

theorem total_students_76_or_80 
  (N : ℕ)
  (h1 : ∃ g : ℕ → ℕ, (∑ i in finset.range 6, g i = N) ∧
                     (∃ a b : ℕ, finset.card {i | g i = a} = 4 ∧ 
                                 finset.card {i | g i = b} = 2 ∧ 
                                 (a = 13 ∧ (b = 12 ∨ b = 14))))
  : N = 76 ∨ N = 80 :=
sorry

end total_students_76_or_80_l430_430571


namespace chord_identity_l430_430876

variable (O : Type) [MetricSpace O] [NormedGroup O] [NormedSpace ℝ O]
variable (A B C D R : O)
variable (t e : ℝ)
variable (unit_circle : MetricSpace.Sphere (0 : O) 1)

-- Define conditions
def is_unit_circle (O : Type) [MetricSpace O] [NormedGroup O] [NormedSpace ℝ O] : Prop :=
  (MetricSpace.is_sphere unit_circle)

def chord_length_eq (A B : O) (l : ℝ) : Prop :=
  dist A B = l

def parallel_to_radius (chord : Set O) (radius : O) : Prop :=
  dist radius chord <= 1

def perpendicular_to (A B O : O) : Prop :=
  dist A B <= dist A O * dist B O

-- Assumes: definition to enforce parallel and perpendicular conditions
def unit_circle_conditions :=
  is_unit_circle O
  ∧ chord_length_eq A C t
  ∧ chord_length_eq A B t
  ∧ chord_length_eq D R t
  ∧ parallel_to_radius {A, B} R
  ∧ chord_length_eq C D e

-- Problem statement: Prove the given equations are true
theorem chord_identity (O A B C D R : O) (unit_circle : MetricSpace.Sphere (0 : O) 1)
  (t e : ℝ) :
  unit_circle_conditions →
  (e - t = 2) ∧ (et = 4) ∧ (e^2 - t^2 = 2 * real.sqrt 5) :=
sorry

end chord_identity_l430_430876


namespace largest_value_among_options_l430_430759

theorem largest_value_among_options :
  let A := 6 * Real.sqrt (Real.cbrt 6)
  let B := Real.cbrt 6
  let C := Real.sqrt (6)^(1 / 4)
  let D := 2 * Real.cbrt 6
  let E := 3 * Real.cbrt (4)
  max A (max B (max C (max D E))) = 2 * Real.cbrt 6 := by
  sorry

end largest_value_among_options_l430_430759


namespace sqrt_mul_l430_430173

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l430_430173


namespace find_value_of_a_l430_430430

theorem find_value_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 3, x^2 + 2*a*x + a^2 - 1 ≤ 24) ∧
  (∃ x ∈ Set.Icc 0 3, x^2 + 2*a*x + a^2 - 1 = 24) ∧
  (∀ x ∈ Set.Icc 0 3, x^2 + 2*a*x + a^2 - 1 ≥ 3) ∧
  (∃ x ∈ Set.Icc 0 3, x^2 + 2*a*x + a^2 - 1 = 3) → 
  a = 2 ∨ a = -5 :=
by
  sorry

end find_value_of_a_l430_430430


namespace apples_found_l430_430900

theorem apples_found (start_apples : ℕ) (end_apples : ℕ) (h_start : start_apples = 7) (h_end : end_apples = 81) : 
  end_apples - start_apples = 74 := 
by 
  sorry

end apples_found_l430_430900


namespace loop_sum_correct_l430_430666

def loop_sum (n : ℕ) : ℕ :=
  ∑ i in Finset.range (n+1), 2 * i

theorem loop_sum_correct : loop_sum 10 = 110 := by
  sorry

end loop_sum_correct_l430_430666


namespace functional_equation_solution_l430_430357

theorem functional_equation_solution (f : ℚ → ℚ) :
  (∀ x y z t : ℚ, x < y → y < z → z < t → 2 * y = x + t ∧ 2 * z = y + t →
    f(y) + f(z) = f(x) + f(t)) →
  ∃ C : ℚ, ∀ x : ℚ, f(x) = C * x :=
begin
  sorry
end

end functional_equation_solution_l430_430357


namespace percent_difference_l430_430855

theorem percent_difference (a b : ℝ) : 
  a = 67.5 * 250 / 100 → 
  b = 52.3 * 180 / 100 → 
  (a - b) = 74.61 :=
by
  intros ha hb
  rw [ha, hb]
  -- omitted proof
  sorry

end percent_difference_l430_430855


namespace total_students_l430_430567

noncomputable def totalStudentsOptions (groups totalGroups specificGroupCount specificGroupSize otherGroupSizes : ℕ) : Set ℕ :=
  if totalGroups = 6 ∧ specificGroupCount = 4 ∧ specificGroupSize = 13 ∧ (otherGroupSizes = 12 ∨ otherGroupSizes = 14) then
    {52 + 2 * otherGroupSizes}
  else
    ∅

theorem total_students :
  totalStudentsOptions 6 4 13 12 = {76} ∧ totalStudentsOptions 6 4 13 14 = {80} :=
by
  -- This is where the proof would go, but we're skipping it as per instructions
  sorry

end total_students_l430_430567


namespace remainder_product_mod_five_l430_430044

-- Define the conditions as congruences
def num1 : ℕ := 14452
def num2 : ℕ := 15652
def num3 : ℕ := 16781

-- State the main theorem using the conditions and the given problem
theorem remainder_product_mod_five : 
  (num1 % 5 = 2) → 
  (num2 % 5 = 2) → 
  (num3 % 5 = 1) → 
  ((num1 * num2 * num3) % 5 = 4) :=
by
  intros
  sorry

end remainder_product_mod_five_l430_430044


namespace sum_of_roots_eq_5_l430_430370

theorem sum_of_roots_eq_5 : 
  let f := (3 : ℚ) * X^3 - 15 * X^2 - 36 * X + 7 in
  (roots f).sum = 5 :=
by 
  sorry

end sum_of_roots_eq_5_l430_430370


namespace solve_and_verify_equation_l430_430691

theorem solve_and_verify_equation :
  ∀ x : ℝ, (x - 2) * (x - 3) = 0 ↔ x = 2 ∨ x = 3 :=
by
  intro x
  split
  case mp =>
    intro h
    have : x - 2 = 0 ∨ x - 3 = 0, from or_of_eq_zero_mul h
    cases this
    case inl => left; exact eq_of_sub_eq_zero this
    case inr => right; exact eq_of_sub_eq_zero this
  case mpr =>
    intro h
    cases h
    case inl => rw [h]; ring
    case inr => rw [h]; ring

end solve_and_verify_equation_l430_430691


namespace sqrt_mult_simplify_l430_430239

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l430_430239


namespace candies_in_packet_l430_430744

theorem candies_in_packet (candy_per_day_weekday : ℕ) (weekday_days : ℕ) 
(day_per_weekend : ℕ) (weekend_days : ℕ) (weeks : ℕ) (packs : ℕ) (total_candies : ℕ) : 
(candy_per_day_weekday = 2) →
(weekday_days = 5) →
(day_per_weekend = 1) →
(weekend_days = 2) →
(weeks = 3) →
(packs = 2) →
(total_candies = (candy_per_day_weekday * weekday_days + day_per_weekend * weekend_days) * weeks) →
(total_candies / packs = 18) :=
begin
  sorry
end

end candies_in_packet_l430_430744


namespace coefficient_of_x3_in_expansion_correct_l430_430772

noncomputable def coefficient_of_x3_in_expansion : ℕ :=
  let r := 6
  let binomial_coeff := Nat.choose 9 r
  (-1 : ℤ)^r * binomial_coeff

theorem coefficient_of_x3_in_expansion_correct :
  coefficient_of_x3_in_expansion = 84 :=
by
  sorry

end coefficient_of_x3_in_expansion_correct_l430_430772


namespace total_students_total_students_alt_l430_430550

def number_of_students (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 
    a + b = 6 ∧ 
    a = 4 ∧ 
    (∀ g, g = 13) ∧ 
    (b = 2 ∧ ((g = 12 ∨ g = 14) ∧ ∀ h, g - h ≤ 1)) ∧ 
    (n = 52 + 2 * 12 ∨ n = 52 + 2 * 14)

theorem total_students : ∃ n : ℕ, number_of_students n :=
by
  use 76
  sorry

theorem total_students_alt : ∃ n : ℕ, number_of_students n :=
by
  use 80
  sorry

end total_students_total_students_alt_l430_430550


namespace find_a1_l430_430067

variable {a : ℕ → ℝ}  -- The sequence terms a_n
variable {S : ℕ → ℝ}  -- The sum of the first n terms S_n
variable {q : ℝ}      -- The common ratio

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n+1) = a n * q
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := ∀ n, S n = a 0 * (1 - q^n) / (1 - q)

-- Given hypothesis
def given_condition (a : ℕ → ℝ) (S : ℕ → ℝ) := ∀ n, (finset.range n).sum a = 3 * S n

-- Objective to prove
theorem find_a1 (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)
    (hg : is_geometric_sequence a q)
    (hs : sum_of_first_n_terms a S)
    (hc : given_condition a S) :
    a 0 = S 1 :=
sorry

end find_a1_l430_430067


namespace f_2015_l430_430801

noncomputable def f : ℝ → ℝ := sorry

axiom f_symmetry (x : ℝ) : f(-x) = -f(x)
axiom f_periodicity (x : ℝ) (k : ℤ) : f(x + 4 * k) = f(x)
axiom f_definition (x : ℝ) (hx : 0 < x ∧ x < 2) : f(x) = Real.log (1 + 3 * x) / Real.log 2

theorem f_2015 : f 2015 = -2 := sorry

end f_2015_l430_430801


namespace total_students_l430_430593

theorem total_students (n_groups : ℕ) (students_in_group : ℕ → ℕ)
    (h1 : n_groups = 6)
    (h2 : ∃ n : ℕ, (students_in_group n = 13) ∧ (finset.filter (λ g, students_in_group g = 13) (finset.range n_groups)).card = 4)
    (h3 : ∀ i j, i < n_groups → j < n_groups → abs (students_in_group i - students_in_group j) ≤ 1) :
    (∃ N, N = 76 ∨ N = 80) :=
begin
    sorry
end

end total_students_l430_430593


namespace range_of_s_l430_430402

theorem range_of_s (x y : ℝ) (h : 4^x + 4^y = 2^(x+1) + 2^(y+1)) : 2 < 2^x + 2^y ∧ 2^x + 2^y ≤ 4 :=
by
  sorry

end range_of_s_l430_430402


namespace sqrt_mul_simp_l430_430205

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l430_430205


namespace symmetric_points_count_l430_430836

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2 * x^2 + 4 * x + 1 else 2 / Real.exp x

theorem symmetric_points_count : 
  ∃ x₁ x₂ ∈ (Set.Icc (0 : ℝ) 2), f x₁ = -f (-x₁) ∧ f x₂ = -f (-x₂) ∧ x₁ ≠ x₂ := by
  sorry

end symmetric_points_count_l430_430836


namespace sqrt_mul_eq_l430_430302

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l430_430302


namespace valid_configurations_count_l430_430519

-- Definitions for the problem conditions
def distinctMarbles : List String := ["Aggie", "Bumblebee", "Steelie", "Tiger", "Earth"]

def isValidConfig (config : List String) : Bool :=
  ¬("Earth" :: "Steelie" :: t) ∈ (List.tails config) ∧
  ¬("Steelie" :: "Earth" :: t) ∈ (List.tails config)
  
-- Statement of the problem
theorem valid_configurations_count : ∃ configs : Finset (List String), 
  configs.card = 72 ∧ 
  ∀ config ∈ configs, 
    config.perm distinctMarbles ∧ 
    isValidConfig config :=
by
  sorry

end valid_configurations_count_l430_430519


namespace sin_cos_45_eq_sqrt2_l430_430028

theorem sin_cos_45_eq_sqrt2 :
  sin (π / 4) + cos (π / 4) = sqrt 2 :=
by
  -- Considering π / 4 radians equivalent to 45 degrees.
  have h_sin : sin (π / 4) = sqrt 2 / 2 := by sorry
  have h_cos : cos (π / 4) = sqrt 2 / 2 := by sorry
  calc
    sin (π / 4) + cos (π / 4)
        = sqrt 2 / 2 + sqrt 2 / 2 : by rw [h_sin, h_cos]
    ... = (sqrt 2 + sqrt 2) / 2 : by sorry
    ... = 2 * sqrt 2 / 2 : by sorry
    ... = sqrt 2 : by sorry

end sin_cos_45_eq_sqrt2_l430_430028


namespace students_got_on_second_stop_l430_430486

-- Given conditions translated into definitions and hypotheses
def students_after_first_stop := 39
def students_after_second_stop := 68

-- The proof statement we aim to prove
theorem students_got_on_second_stop : (students_after_second_stop - students_after_first_stop) = 29 := by
  -- Proof goes here
  sorry

end students_got_on_second_stop_l430_430486


namespace pizza_topping_combinations_l430_430725

/-- 
A university cafeteria offers 4 flavors of pizza: pepperoni, chicken, Hawaiian, and vegetarian. 
A customer has the option to add extra cheese, mushrooms, or both to any kind of pizza. 
Given these conditions, prove that the combinations of toppings a customer can add to any kind of pizza are:
1. No extra toppings
2. Extra cheese
3. Mushrooms
4. Extra cheese and mushrooms.
-/
theorem pizza_topping_combinations :
  ∀ (pizza_flavors : Fin 4), 
  ∃ (topping_combinations: Fin 4), 
    topping_combinations = (0, 1, 2, 3) := sorry

end pizza_topping_combinations_l430_430725


namespace inequality_solution_l430_430603

theorem inequality_solution (x : ℝ) : (|x - 1| + |x - 2| > 5) ↔ (x ∈ (-∞, -1) ∪ (4, ∞)) :=
by
  sorry

end inequality_solution_l430_430603


namespace find_large_circle_radius_from_chord_l430_430784

-- Definitions of the conditions
def small_circle_radius_from_chord (chord_length : ℝ) (n : ℕ) : ℝ :=
  chord_length / (2 * (n - 1))

def large_circle_radius (small_radius : ℝ) : ℝ :=
  4 * small_radius

theorem find_large_circle_radius_from_chord :
  large_circle_radius (small_circle_radius_from_chord 16 5) = 8 :=
by
  sorry

end find_large_circle_radius_from_chord_l430_430784


namespace total_students_76_or_80_l430_430577

theorem total_students_76_or_80 
  (N : ℕ)
  (h1 : ∃ g : ℕ → ℕ, (∑ i in finset.range 6, g i = N) ∧
                     (∃ a b : ℕ, finset.card {i | g i = a} = 4 ∧ 
                                 finset.card {i | g i = b} = 2 ∧ 
                                 (a = 13 ∧ (b = 12 ∨ b = 14))))
  : N = 76 ∨ N = 80 :=
sorry

end total_students_76_or_80_l430_430577


namespace sqrt3_mul_sqrt12_eq_6_l430_430256

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l430_430256


namespace gcd_16016_20020_l430_430773

theorem gcd_16016_20020 : Int.gcd 16016 20020 = 4004 :=
by
  sorry

end gcd_16016_20020_l430_430773


namespace find_ratio_l430_430377

-- Definitions for experience years
def Roger := 42
def Peter := 12
def Robert := 8
def Mike := Robert - 2
def Tom := Roger - Peter - Robert - Mike

-- Problem Condition Assertions
axiom cond1 : Roger = Peter + Tom + Robert + Mike
axiom cond2 : Roger + 8 = 50
axiom cond3 : Peter = 12
axiom cond4 : Robert = Peter - 4
axiom cond5 : Robert = Mike + 2

-- Goal Statement
theorem find_ratio : Tom / Robert = 1 := 
by 
  have h1 : Tom = Robert := by
    rw [Roger, Peter, Robert, Mike] at cond1
    linarith
  rw h1
  exact div_self (ne_of_gt (lt_add_one' 0))

end find_ratio_l430_430377


namespace absolute_value_inequality_l430_430606

theorem absolute_value_inequality (x : ℝ) : (|x + 1| > 3) ↔ (x > 2 ∨ x < -4) :=
by
  sorry

end absolute_value_inequality_l430_430606


namespace sum_f_1_to_2017_l430_430920

noncomputable def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x < -1 then -(x + 2)^2
  else if -1 ≤ x ∧ x < 3 then x
  else f (x - 6)

theorem sum_f_1_to_2017 : 
  ∑ i in (finset.range 2017).map (function.embedding.mk (λ i, i + 1) (λ i j h, nat.succ_injective h)), f i = 337 :=
sorry

end sum_f_1_to_2017_l430_430920


namespace rectangle_proof_l430_430910

noncomputable def length_of_EF (AB CD BC DA : ℝ) [rectangle : AB = CD ∧ BC = DA]
  (A B E : Point) (EA EB : ℝ) [dist_EA : dist E A = EA] [dist_EB : dist E B = EB] : ℝ :=
  let x_E := (5 / 3 : ℝ)
  let y_E := (5 * (sqrt 35) / 3 : ℝ)
  y_E

theorem rectangle_proof :
  ∀ (A B C D E : Point)
    (AB CD BC DA : ℝ)
    (EA EB EF : ℝ)
    (h₁ : AB = 30 ∧ CD = 30)
    (h₂ : BC = 40 ∧ DA = 40)
    (h₃ : dist E A = 10)
    (h₄ : dist E B = 30)
    (h5 : perpendicular E D A)
    (length_EF : length_of_EF AB CD BC DA A B E 10 30 = (5 * sqrt 35 / 3 : ℝ)),
    EF = (5 * sqrt 35 / 3 : ℝ) :=
begin
  sorry
end

end rectangle_proof_l430_430910


namespace smallest_n_l430_430347

theorem smallest_n (r g b : ℕ) (h1 : 10 * r = 25 * n)
                              (h2 : 18 * g = 25 * n)
                              (h3 : 20 * b = 25 * n) 
                              (h4 : y = 25) : 
                              ∃ n, n = 15 := 
begin
  sorry
end

end smallest_n_l430_430347


namespace geometric_sum_eight_terms_l430_430889

theorem geometric_sum_eight_terms :
  ∀ (a : ℕ → ℝ) (q : ℝ), 
  a 8 = 1 ∧ q = 1 / 2 → (∑ i in finset.range 8, a i) = 255 :=
by
  sorry

end geometric_sum_eight_terms_l430_430889


namespace number_of_divisors_g_2023_l430_430382

theorem number_of_divisors_g_2023 :
  let g (n : ℕ) := 2^n * 3^n in
  nat.num_divisors (g 2023) = 4096576 :=
sorry

end number_of_divisors_g_2023_l430_430382


namespace trigonometric_identity_solution_trigonometric_solutions_l430_430059

theorem trigonometric_identity_solution:
  ∀ (z : ℝ), cos z ≠ 0 → 1 + 2 * (cos (2 * z) * (tan z) - sin (2 * z)) * (cos z) ^ 2 = cos (2 * z) :=
by {
  intro z,
  intro h,
  sorry
}

theorem trigonometric_solutions:
  ∀ (z : ℝ), cos z ≠ 0 → (∃ n : ℤ, z = n * π) ∨ (∃ k : ℤ, z = (2 * k + 1) * π / 4) :=
by {
  intro z,
  intro h,
  sorry
}

end trigonometric_identity_solution_trigonometric_solutions_l430_430059


namespace sqrt_mul_eq_l430_430307

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l430_430307


namespace isosceles_trapezoid_base_angles_equal_l430_430528

theorem isosceles_trapezoid_base_angles_equal {A B C D : Type*}
  [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
  (AB CD : ℝ) (AD BC : ℝ)
  (AB_parallel_CD : AB = CD) (AD_eq_BC : AD = BC) :
  ∀ (α β : ℝ), ∠ D A B = α ∧ ∠ B C D = β → α = β := by
sorry

end isosceles_trapezoid_base_angles_equal_l430_430528


namespace sqrt_multiplication_l430_430144

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l430_430144


namespace sqrt3_mul_sqrt12_eq_6_l430_430252

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l430_430252


namespace isosceles_triangle_base_length_l430_430630

-- Define the isosceles triangle problem
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  perimeter : ℝ
  isIsosceles : (side1 = side2 ∨ side1 = base ∨ side2 = base)
  sideLengthCondition : (side1 = 3 ∨ side2 = 3 ∨ base = 3)
  perimeterCondition : side1 + side2 + base = 13
  triangleInequality1 : side1 + side2 > base
  triangleInequality2 : side1 + base > side2
  triangleInequality3 : side2 + base > side1

-- Define the theorem to prove
theorem isosceles_triangle_base_length (T : IsoscelesTriangle) :
  T.base = 3 := by
  sorry

end isosceles_triangle_base_length_l430_430630


namespace eccentricity_of_hyperbola_l430_430394

noncomputable def hyperbola_eccentricity 
  (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
  (F₁ F₂ P : ℝ × ℝ)
  (h_foci : dist P F₁ = 3 * dist P F₂)
  (h_hyperbola : P.1 ^ 2 / a ^ 2 - P.2 ^ 2 / b ^ 2 = 1)
  (h_angle : angle F₁ P F₂ = π / 3) : ℝ :=
  let c := dist F₁ F₂ / 2 in
  real.sqrt (7 / 4)

theorem eccentricity_of_hyperbola 
  (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
  (F₁ F₂ P : ℝ × ℝ)
  (h_foci : dist P F₁ = 3 * dist P F₂)
  (h_hyperbola : P.1 ^ 2 / a ^ 2 - P.2 ^ 2 / b ^ 2 = 1)
  (h_angle : angle F₁ P F₂ = π / 3) :
  hyperbola_eccentricity a b h_a h_b F₁ F₂ P h_foci h_hyperbola h_angle = real.sqrt 7 / 2 := 
sorry

end eccentricity_of_hyperbola_l430_430394


namespace locus_of_intersection_of_internal_common_tangents_l430_430724

-- Definitions of the geometric constructs
variables {ABC : Triangle} {ω : Circle} (A B C : Point)
variables (D E K L : Point) (ℓ : Line)
variables (γ₁ γ₂ : Circle)

-- Conditions as given in the problem statement
variables (h₀ : is_inscribed_in (ABC) (ω))
variables (h₁ : parallel ℓ (BC))
variables (h₂ : intersects_at_line ℓ [AB] D)
variables (h₃ : intersects_at_line ℓ [AC] E)
variables (h₄ : intersects_ω_at_points ℓ ω K L)
variables (h₅ : between D K E)
variables (h₆ : is_tangent_to_segments γ₁ [KD] [BD])
variables (h₇ : is_tangent_to_segments γ₂ [LE] [CE])
variables (h₈ : is_tangent_to_circle γ₁ ω)
variables (h₉ : is_tangent_to_circle γ₂ ω)

-- Prove the locus of intersections
theorem locus_of_intersection_of_internal_common_tangents :
  locus_of_intersection_of_tangents_at_arc γ₁ γ₂ ω = midpoint_of_arc_not_containing_point B C A :=
sorry

end locus_of_intersection_of_internal_common_tangents_l430_430724


namespace part_a_part_b_part_c_l430_430427

section 
variable {a : ℝ} {f : ℝ → ℝ} 

-- Condition for part (I)
def f_def (a : ℝ) (x : ℝ) : ℝ := a * x + x * log x

theorem part_a (h : f_derivative = 3) : a = 2 :=
sorry

-- Condition for part (II)
def inequality_holds (k x : ℝ) : Prop := (x - 1) * k < f_def 2 x
def maximum_k (k_main : Int) : Prop := ∃ x > 1, inequality_holds (k_main + 1) x → False

theorem part_b (k_main : Int) (h1 : ∀ x, x > 1 → inequality_holds k_main x) : k_main = 3 :=
sorry

-- Condition for part (III)
def m_n_exponent (m n : ℕ) : ℝ := (m * n ^ n) ^ m
def n_m_exponent (m n : ℕ) : ℝ := (n * m ^ m) ^ n

theorem part_c (n m : ℕ) (h : n > m ∧ m ≥ 4) : m_n_exponent m n > n_m_exponent m n :=
sorry

end

end part_a_part_b_part_c_l430_430427


namespace partition_space_l430_430896

-- Define a subset of ℝ by flooring x and taking modulo 1979.
def Ai (i : ℕ) : set ℝ := { x : ℝ | ⌊x⌋ % 1979 = i }

-- State the main theorem that we aim to prove
theorem partition_space :
  (∀ x : ℝ, ∃ i : ℕ, 1 ≤ i ∧ i ≤ 1979 ∧ x ∈ Ai i) ∧
  (∀ i1 i2 : ℕ, 1 ≤ i1 ∧ i1 ≤ 1979 ∧ 1 ≤ i2 ∧ i2 ≤ 1979 ∧ i1 ≠ i2 → 
    ∀ x : ℝ, x ∈ Ai i1 → x ∉ Ai i2) :=
by 
  sorry -- Proof omitted

end partition_space_l430_430896


namespace sue_answer_l430_430115

theorem sue_answer (x : ℕ) (h : x = 8) : 4 * (3 * (x + 3) - 2) = 124 :=
by
  rw h
  sorry

end sue_answer_l430_430115


namespace multiple_of_2009_l430_430354

theorem multiple_of_2009 (a : ℕ) (g : ℕ → ℕ) (f : ℕ → ℕ) [bijective g] :
  (∀ x : ℕ, (Function.iterate f 2009 x) = g x + a) → 2009 ∣ a :=
sorry

end multiple_of_2009_l430_430354


namespace total_students_l430_430579

theorem total_students (N : ℕ) (h1 : ∃ g1 g2 g3 g4 g5 g6 : ℕ, 
  g1 = 13 ∧ g2 = 13 ∧ g3 = 13 ∧ g4 = 13 ∧ 
  ((g5 = 12 ∧ g6 = 12) ∨ (g5 = 14 ∧ g6 = 14)) ∧ 
  N = g1 + g2 + g3 + g4 + g5 + g6) : 
  N = 76 ∨ N = 80 :=
by
  sorry

end total_students_l430_430579


namespace forgotten_angle_measure_l430_430475

theorem forgotten_angle_measure 
  (total_sum : ℕ) 
  (measured_sum : ℕ) 
  (sides : ℕ) 
  (n_minus_2 : ℕ)
  (polygon_has_18_sides : sides = 18)
  (interior_angle_sum : total_sum = n_minus_2 * 180)
  (n_minus : n_minus_2 = (sides - 2))
  (measured : measured_sum = 2754) :
  ∃ forgotten_angle, forgotten_angle = total_sum - measured_sum ∧ forgotten_angle = 126 :=
by
  sorry

end forgotten_angle_measure_l430_430475


namespace find_s_and_x_l430_430780

theorem find_s_and_x (s x t : ℝ) (h1 : t = 15 * s^2) (h2 : t = 3.75) :
  s = 0.5 ∧ x = s / 2 → x = 0.25 :=
by
  sorry

end find_s_and_x_l430_430780


namespace sqrt_multiplication_l430_430151

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l430_430151


namespace sqrt_mul_simp_l430_430207

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l430_430207


namespace hyperbola_eccentricity_l430_430002

noncomputable def eccentricity_of_hyperbola (AC AB BC : ℝ) (k : ℝ) 
  (arithmetic_sequence : AC + AB = 2 * BC) 
  (angle_ACB : ∠BCA = 120) : ℝ :=
  AB / (BC - AC)

theorem hyperbola_eccentricity {A B C : Point}
  (foci_hyperbola : Hyperbola A B)
  (point_on_hyperbola : OnHyperbola C)
  (arithmetic_sequence : AC + AB = 2 * BC)
  (angle_ACB : ∠BCA = 120) :
  eccentricity_of_hyperbola AC AB BC = 7/2 :=
sorry

end hyperbola_eccentricity_l430_430002


namespace minimum_a_condition_l430_430863

-- Define the quadratic function
def f (a x : ℝ) := x^2 + a * x + 1

-- Define the condition that the function remains non-negative in the open interval (0, 1/2)
def f_non_negative_in_interval (a : ℝ) : Prop :=
  ∀ (x : ℝ), 0 < x ∧ x < 1 / 2 → f a x ≥ 0

-- State the theorem that the minimum value for a with the given condition is -5/2
theorem minimum_a_condition : ∀ (a : ℝ), f_non_negative_in_interval a → a ≥ -5 / 2 :=
by sorry

end minimum_a_condition_l430_430863


namespace cyclic_quadrilateral_opposite_angles_sum_l430_430643

theorem cyclic_quadrilateral_opposite_angles_sum (A B C D : Point) (c : Circle) (hA : A ∈ c) (hB : B ∈ c) (hC : C ∈ c) (hD : D ∈ c) :
  let α := ∠A B C
  let β := ∠B C D
  let γ := ∠C D A
  let δ := ∠D A B
  α + γ = 180 ∧ β + δ = 180 := 
sorry

end cyclic_quadrilateral_opposite_angles_sum_l430_430643


namespace shaded_angle_of_isosceles_triangles_fitting_in_square_l430_430980

theorem shaded_angle_of_isosceles_triangles_fitting_in_square :
  (∃ (θ : ℝ), 
    3 * 30 = 90 ∧ 
    ∀ (x y z : ℝ), x + y = z →  
    isoscelesθ : ∀ (n : ℕ), n = 3 → 
    θ = 75 → 
    3 * θ + 2 * θ = 180 ∧
    shaded_angle = 90 - θ → 
    shaded_angle = 15) :=
sorry

end shaded_angle_of_isosceles_triangles_fitting_in_square_l430_430980


namespace find_b1_l430_430634

theorem find_b1 (b : ℕ → ℕ) 
  (h : ∀ n, 2 ≤ n → ∑ i in Finset.range n.succ, b i = (n^2 + n) * b n)
  (h32 : b 32 = 1) :
  b 1 = 528 :=
sorry

end find_b1_l430_430634


namespace part_I_a_part_I_b_part_I_c_part_II_l430_430918

def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

theorem part_I_a : f 0 + f 1 = Real.sqrt 3 / 3 := sorry

theorem part_I_b : f (-1) + f 2 = Real.sqrt 3 / 3 := sorry

theorem part_I_c : f (-2) + f 3 = Real.sqrt 3 / 3 := sorry

theorem part_II (x1 x2 : ℝ) (h : x1 + x2 = 1) : f x1 + f x2 = Real.sqrt 3 / 3 := sorry

end part_I_a_part_I_b_part_I_c_part_II_l430_430918


namespace larger_volume_equal_surface_area_smaller_surface_area_equal_volume_ratio_of_surface_areas_l430_430079

noncomputable def surface_area_cube (a : ℝ) : ℝ := 6 * a^2
noncomputable def surface_area_sphere (r : ℝ) : ℝ := 4 * π * r^2

noncomputable def volume_cube (a : ℝ) : ℝ := a^3
noncomputable def volume_sphere (r : ℝ) : ℝ := (4 / 3) * π * r^3

theorem larger_volume_equal_surface_area (a r : ℝ) (h : surface_area_cube a = surface_area_sphere r) : volume_sphere r > volume_cube a :=
sorry

theorem smaller_surface_area_equal_volume (a r : ℝ) (h : volume_cube a = volume_sphere r) : surface_area_sphere r < surface_area_cube a :=
sorry

noncomputable def radius_inscribed_sphere (a : ℝ) : ℝ := a / 2
noncomputable def radius_edge_touching_sphere (a : ℝ) : ℝ := (sqrt 2 * a) / 2
noncomputable def radius_vertex_passing_sphere (a : ℝ) : ℝ := (sqrt 3 * a) / 2

noncomputable def surface_area (r : ℝ) : ℝ := 4 * π * r^2

theorem ratio_of_surface_areas (a : ℝ) :
  surface_area (radius_inscribed_sphere a) / surface_area (radius_inscribed_sphere a) = 1 ∧ 
  surface_area (radius_edge_touching_sphere a) / surface_area (radius_inscribed_sphere a) = 2 ∧ 
  surface_area (radius_vertex_passing_sphere a) / surface_area (radius_inscribed_sphere a) = 3 :=
sorry

end larger_volume_equal_surface_area_smaller_surface_area_equal_volume_ratio_of_surface_areas_l430_430079


namespace min_value_expression_l430_430817

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ (m : ℝ), m = 3 / 2 ∧ ∀ t > 0, (2 * x / (x + 2 * y) + y / x) ≥ m :=
by
  use 3 / 2
  sorry

end min_value_expression_l430_430817


namespace actual_number_of_sides_l430_430096

theorem actual_number_of_sides (apparent_angle : ℝ) (distortion_factor : ℝ)
  (sum_exterior_angles : ℝ) (actual_sides : ℕ) :
  apparent_angle = 18 ∧ distortion_factor = 1.5 ∧ sum_exterior_angles = 360 ∧ 
  apparent_angle / distortion_factor = sum_exterior_angles / actual_sides →
  actual_sides = 30 :=
by
  sorry

end actual_number_of_sides_l430_430096


namespace calc_3_op_2_op_4_op_1_l430_430751

def op (a b : ℕ) : ℕ :=
match a, b with
| 1, 1 => 2 | 1, 2 => 3 | 1, 3 => 4 | 1, 4 => 1
| 2, 1 => 3 | 2, 2 => 1 | 2, 3 => 2 | 2, 4 => 4
| 3, 1 => 4 | 3, 2 => 2 | 3, 3 => 1 | 3, 4 => 3
| 4, 1 => 1 | 4, 2 => 4 | 4, 3 => 3 | 4, 4 => 2
| _, _  => 0 -- default case, though won't be used

theorem calc_3_op_2_op_4_op_1 : op (op 3 2) (op 4 1) = 3 :=
by
  sorry

end calc_3_op_2_op_4_op_1_l430_430751


namespace max_tan_alpha_l430_430819

theorem max_tan_alpha (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
    (h : Real.tan (α + β) = 9 * Real.tan β) : Real.tan α ≤ 4 / 3 :=
by
  sorry

end max_tan_alpha_l430_430819


namespace find_delightful_set_l430_430029

def delightful_set (G : Graph V) (S : finset V) : Prop :=
  ∀ (v1 v2 : V), v1 ∈ S → v2 ∈ S → G.adj v1 v2 → false ∧ 
  ∀ v ∉ S, ∃ u ∈ S, G.adj v u
  
theorem find_delightful_set (V : Type) (G : Graph V) [Fintype V] [DecidableEq V]
  (h : fintype.card V = 10000) :
  ∃ (S : finset V), delightful_set G S ∧ S.card ≤ 9802 :=
sorry

end find_delightful_set_l430_430029


namespace is_linear_equation_D_l430_430680

theorem is_linear_equation_D :
  (∀ (x y : ℝ), 2 * x + 3 * y = 7 → false) ∧
  (∀ (x : ℝ), 3 * x ^ 2 = 3 → false) ∧
  (∀ (x : ℝ), 6 = 2 / x - 1 → false) ∧
  (∀ (x : ℝ), 2 * x - 1 = 20 → true) 
:= by {
  sorry
}

end is_linear_equation_D_l430_430680


namespace five_digit_palindromes_count_l430_430094

def is_palindrome (n : ℕ) : Prop :=
  let str := n.toString
  str = str.reverse

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

theorem five_digit_palindromes_count : { n : ℕ // is_five_digit n ∧ is_palindrome n }.card = 900 := 
  sorry

end five_digit_palindromes_count_l430_430094


namespace tangency_points_l430_430526

variables (A B C A1 B1 C1 : Point)
variables (BC CA AB : Line)
variables (incircle : Circle)

def on_side (P : Point) (L : Line) := ∃ (x : ℝ), x ∈ Icc (0:ℝ) 1 ∧ P = (L.start :+ x • L.direction)

def is_tangent (circle : Circle) (P : Point) (L : Line) := ∃ t, t ∈ Ioo (0:ℝ) 1 ∧ dist P (circle.center) = circle.radius ∧ P ∈ L

theorem tangency_points {A B C A1 B1 C1 : Point}
    (h1 : on_side A1 BC)
    (h2 : on_side B1 CA)
    (h3 : on_side C1 AB)
    (h4 : dist A C1 = dist A B1) 
    (h5 : dist B A1 = dist B C1) 
    (h6 : dist C A1 = dist C B1) 
    : 
    is_tangent incircle A1 BC ∧ is_tangent incircle B1 CA ∧ is_tangent incircle C1 AB := 
by 
  sorry

end tangency_points_l430_430526


namespace sqrt_mul_simplify_l430_430280

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l430_430280


namespace roots_equation_l430_430036

theorem roots_equation (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : a * 4^3 + b * 4^2 + c * 4 + d = 0) (h₃ : a * (-3)^3 + b * (-3)^2 + c * (-3) + d = 0) :
  (b + c) / a = -13 :=
by
  sorry

end roots_equation_l430_430036


namespace sqrt_mul_l430_430181

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l430_430181


namespace choose_4_out_of_10_l430_430879

theorem choose_4_out_of_10 : nat.choose 10 4 = 210 := by
  sorry

end choose_4_out_of_10_l430_430879


namespace sqrt_mul_eq_6_l430_430160

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l430_430160


namespace electrical_bill_undetermined_l430_430945

theorem electrical_bill_undetermined
    (gas_bill : ℝ)
    (gas_paid_fraction : ℝ)
    (additional_gas_payment : ℝ)
    (water_bill : ℝ)
    (water_paid_fraction : ℝ)
    (internet_bill : ℝ)
    (internet_payments : ℝ)
    (payment_amounts: ℝ)
    (total_remaining : ℝ) :
    gas_bill = 40 →
    gas_paid_fraction = 3 / 4 →
    additional_gas_payment = 5 →
    water_bill = 40 →
    water_paid_fraction = 1 / 2 →
    internet_bill = 25 →
    internet_payments = 4 * 5 →
    total_remaining = 30 →
    (∃ electricity_bill : ℝ, true) -> 
    false := by
  intro gas_bill_eq gas_paid_fraction_eq additional_gas_payment_eq
  intro water_bill_eq water_paid_fraction_eq
  intro internet_bill_eq internet_payments_eq 
  intro total_remaining_eq 
  intro exists_electricity_bill 
  sorry -- Proof that the electricity bill cannot be determined

end electrical_bill_undetermined_l430_430945


namespace sqrt_mult_eq_six_l430_430317

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l430_430317


namespace convex_polyhedron_Euler_formula_l430_430072

theorem convex_polyhedron_Euler_formula (V E F : ℕ) (h : ¬∃ (P : Polyhedron), 
  convex P ∧ V = P.vertices ∧ E = P.edges ∧ F = P.faces) : 
  V - E + F = 2 :=
sorry

end convex_polyhedron_Euler_formula_l430_430072


namespace intersects_line_l430_430122

theorem intersects_line (x y : ℝ) : 
  (3 * x + 2 * y = 5) ∧ ((x / 3) + (y / 2) = 1) → ∃ x y : ℝ, (3 * x + 2 * y = 5) ∧ ((x / 3) + (y / 2) = 1) :=
by
  intro h
  sorry

end intersects_line_l430_430122


namespace solve_quadratic_eq_solve_cubic_eq_l430_430372

-- Statement for the first equation
theorem solve_quadratic_eq (x : ℝ) : 9 * x^2 - 25 = 0 ↔ x = 5 / 3 ∨ x = -5 / 3 :=
by sorry

-- Statement for the second equation
theorem solve_cubic_eq (x : ℝ) : (x + 1)^3 - 27 = 0 ↔ x = 2 :=
by sorry

end solve_quadratic_eq_solve_cubic_eq_l430_430372


namespace find_slope_l430_430090

theorem find_slope 
  (k : ℝ)
  (h₀ : k ≠ 0)
  (ellipse_eq : ∀ x y : ℝ, (x^2 / 2 + y^2 = 1) ↔ ((2k^2 + 1) * x^2 + 4k^2 * x + 2k^2 - 2 = 0))
  (passes_through_focus : ∃ x y : ℝ, (x = -1) ∧ (y = 0) ∧ y = k * (x + 1))
  (midpoint_line : ∃ x₀ y₀ : ℝ, (x₀ = (-2k^2) / (2k^2 + 1)) ∧ (y₀ = k / (2k^2 + 1)) ∧ (x₀ + 2 * y₀ = 0)) :
  k = 1 :=
sorry

end find_slope_l430_430090


namespace cost_of_780_candies_l430_430699

def cost_of_candies (box_size: ℕ) (box_cost: ℝ) (num_candies: ℕ) (discount_threshold: ℕ) (discount_rate: ℝ): ℝ :=
  let num_boxes := (num_candies / box_size) in
  let total_cost := (num_boxes * box_cost) in
  if num_candies > discount_threshold then
    total_cost - (discount_rate * total_cost)
  else
    total_cost

theorem cost_of_780_candies :
  cost_of_candies 30 8 780 500 0.1 = 187.2 :=
by
  -- placeholder for the proof
  sorry

end cost_of_780_candies_l430_430699


namespace num_permutations_multiple_of_13_between_200_and_999_l430_430848

/-- 
Theorem: There are 122 integers between 200 and 999 inclusive, such that some permutation of their digits
is a multiple of 13 and also falls within the range 200 to 999 inclusive.
-/
theorem num_permutations_multiple_of_13_between_200_and_999 : 
  (finset.filter
    (λ n : ℕ, ∃ (m : ℕ) (hm : 200 ≤ m ∧ m ≤ 999), 
      m % 13 = 0 ∧ m.digits 10 ~ n.digits 10 ∧ 200 ≤ n ∧ n ≤ 999)
    (finset.range 800)).card = 122 :=
sorry

end num_permutations_multiple_of_13_between_200_and_999_l430_430848


namespace find_a7_over_b7_l430_430968

-- Definitions of the sequences and the arithmetic properties
variable {a b: ℕ → ℕ}  -- sequences a_n and b_n
variable {S T: ℕ → ℕ}  -- sums of the first n terms

-- Problem conditions
def is_arithmetic_sequence (seq: ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, seq (n + 1) - seq n = d

def sum_of_first_n_terms (seq: ℕ → ℕ) (sum_fn: ℕ → ℕ) : Prop :=
  ∀ n, sum_fn n = n * (seq 1 + seq n) / 2

-- Given conditions
axiom h1: is_arithmetic_sequence a
axiom h2: is_arithmetic_sequence b
axiom h3: sum_of_first_n_terms a S
axiom h4: sum_of_first_n_terms b T
axiom h5: ∀ n, S n / T n = (3 * n + 2) / (2 * n)

-- Main theorem to prove
theorem find_a7_over_b7 : (a 7) / (b 7) = (41 / 26) :=
sorry

end find_a7_over_b7_l430_430968


namespace retailer_percentage_profit_l430_430102

def wholesale_price : ℝ := 90
def retail_price : ℝ := 120
def discount_rate : ℝ := 0.10

def selling_price : ℝ := retail_price * (1 - discount_rate)
def profit : ℝ := selling_price - wholesale_price
def percentage_profit : ℝ := (profit / wholesale_price) * 100

theorem retailer_percentage_profit : percentage_profit = 20 :=
by
  sorry

end retailer_percentage_profit_l430_430102


namespace log_conditions_implies_relation_l430_430451

theorem log_conditions_implies_relation (x y z : ℝ) 
  (h1 : log 2 (log (1 / 2) (log 2 x)) = 0) 
  (h2 : log 3 (log (1 / 3) (log 3 y)) = 0) 
  (h3 : log 5 (log (1 / 5) (log 5 z)) = 0) : 
  z < x ∧ x < y := 
sorry

end log_conditions_implies_relation_l430_430451


namespace sqrt_multiplication_l430_430149

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l430_430149


namespace sqrt_mul_simplify_l430_430268

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l430_430268


namespace minimum_equilateral_triangles_l430_430664

theorem minimum_equilateral_triangles (s1 s2 : ℕ) (hs1 : s1 = 1) (hs2 : s2 = 15) :
  let A := fun s => (√3 / 4) * s^2 in
  (A s2) / (A s1) = 225 :=
by
  -- We start by acknowledging the given side lengths of the triangles.
  have hA1 : A s1 = (√3 / 4) * s1^2, by sorry
  have hA2 : A s2 = (√3 / 4) * s2^2, by sorry

  -- Calculate the areas of the given triangles.
  have hAreaSmall : A s1 = √3 / 4, by rw [hs1]; rw [hA1]; ring
  have hAreaLarge : A s2 = (√3 / 4) * 225, by rw [hs2]; rw [hA2]; ring

  -- Determine the ratio of areas, which gives the number of small triangles needed to cover the large one.
  have hRatio : (A s2) / (A s1) = 225, by calc
    (A s2) / (A s1) = ((√3 / 4) * 225) / (√3 / 4) : by rw [hAreaLarge, hAreaSmall]
    ... = 225 : by simp

  -- Conclude the proof with the desired result.
  exact hRatio

end minimum_equilateral_triangles_l430_430664


namespace sqrt_mul_eq_l430_430301

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l430_430301


namespace EthanHourlyWage_l430_430764

theorem EthanHourlyWage
  (hours_per_day : ℕ) (days_per_week : ℕ) (earned_in_5_weeks : ℕ)
  (hours_per_day_eq : hours_per_day = 8)
  (days_per_week_eq : days_per_week = 5)
  (earned_in_5_weeks_eq : earned_in_5_weeks = 3600) :
  ∃ (hourly_wage : ℕ), hourly_wage = 18 :=
by
  let hours_per_week := hours_per_day * days_per_week
  have hours_per_week_eq : hours_per_week = 40 := by
    rw [hours_per_day_eq, days_per_week_eq]
    norm_num

  let total_hours := hours_per_week * 5
  have total_hours_eq : total_hours = 200 := by
    rw [hours_per_week_eq]
    norm_num

  let hourly_wage := earned_in_5_weeks / total_hours
  have hourly_wage_eq : hourly_wage = 18 := by
    rw [earned_in_5_weeks_eq, total_hours_eq]
    norm_num

  use hourly_wage
  exact hourly_wage_eq

end EthanHourlyWage_l430_430764


namespace value_of_expression_l430_430450

theorem value_of_expression (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 3 = 21 :=
by
sorry

end value_of_expression_l430_430450


namespace weight_of_computer_rounded_l430_430013

noncomputable def weight_of_elephant : ℝ := 5.85
noncomputable def number_of_elephants : ℕ := 6
noncomputable def calculate_weight_computer : ℝ := weight_of_elephant * number_of_elephants

theorem weight_of_computer_rounded :
  Int.round (calculate_weight_computer) = 35 :=
sorry

end weight_of_computer_rounded_l430_430013


namespace retailer_percentage_profit_l430_430100

def wholesale_price : ℝ := 90
def retail_price : ℝ := 120
def discount_rate : ℝ := 0.10

def selling_price : ℝ := retail_price * (1 - discount_rate)
def profit : ℝ := selling_price - wholesale_price
def percentage_profit : ℝ := (profit / wholesale_price) * 100

theorem retailer_percentage_profit : percentage_profit = 20 :=
by
  sorry

end retailer_percentage_profit_l430_430100


namespace solve_for_x_l430_430095

theorem solve_for_x (x : ℝ) (h1 : 0 < x) (h2 : sqrt (7 * x / 3) = x) : x = 7 / 3 :=
by 
  sorry

end solve_for_x_l430_430095


namespace number_of_valid_groupings_l430_430656

-- Definitions based on conditions
def num_guides : ℕ := 2
def num_tourists : ℕ := 6
def total_groupings : ℕ := 2 ^ num_tourists
def invalid_groupings : ℕ := 2  -- All tourists go to one guide either a or b

-- The theorem to prove
theorem number_of_valid_groupings : total_groupings - invalid_groupings = 62 :=
by sorry

end number_of_valid_groupings_l430_430656


namespace sqrt_mul_eq_6_l430_430155

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l430_430155


namespace sqrt_multiplication_l430_430153

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l430_430153


namespace proof_problem_l430_430433

-- Define the sequences a_n and b_n
noncomputable def a : ℕ+ → ℕ
| 1 := 1
| (nat.succ n) := 2 * (a n)

noncomputable def b (n : ℕ+) : ℝ :=
  n / (4 * (a n))

-- Define the sum T_n
noncomputable def T (n : ℕ+) : ℝ :=
  ∑ k in finset.range n, (b (k + 1))

-- Proof statement
theorem proof_problem (n : ℕ+) : 
  (a n = 2^(n-1)) ∧ 
  (T n = 1 - (n + 2) / 2^(n+1)) := 
by {
  sorry
}

end proof_problem_l430_430433


namespace solve_zero_point_increasing_interval_l430_430824

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.sin x + a * Real.cos x
noncomputable def g (x : ℝ) : ℝ := (Real.sin x + Real.cos x) ^ 2 - 2 * (Real.sin x) ^ 2

theorem solve_zero_point : 
  (∃ a : ℝ, f (3 * Real.pi / 4) a = 0) → (f (3 * Real.pi / 4) 1 = 0) :=
sorry

theorem increasing_interval (k : ℤ) : 
  [k * Real.pi - 3 * Real.pi / 8, k * Real.pi + Real.pi / 8] → 
  MonoIncreasingOn g (Set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8)) :=
sorry

end solve_zero_point_increasing_interval_l430_430824


namespace probability_x_gt_3y_in_rectangle_l430_430525

theorem probability_x_gt_3y_in_rectangle :
  let region_vertices := [(0, 0), (3036, 0), (3036, 3037), (0, 3037)]
  let probability := 506 / 3037
  (∃ x y, (0 ≤ x ∧ x ≤ 3036) ∧ (0 ≤ y ∧ y ≤ 3037) ∧ x > 3 * y) →
  (∃ x y, (0 ≤ x ∧ x ≤ 3036) ∧ (0 ≤ y ∧ y ≤ 3037)) →
  (∑ i in region_vertices, probability) = 506 / 3037 :=
by
  sorry

end probability_x_gt_3y_in_rectangle_l430_430525


namespace distinct_k_mutations_of_permutations_l430_430498

/-- n-tuple (a1, a2, ..., an) is a permutation if every number from the set {1, 2, ..., n} occurs exactly once. --/
def is_permutation (n : ℕ) (a : Fin n → ℕ) : Prop :=
  ∀ i : Fin n, ∃ j : Fin n, a j = i + 1

/-- Define k-mutation of an n-tuple permutation by considering indices modulo n. --/
def k_mutation (n k : ℕ) (p : Fin n → ℕ) : Fin n → ℕ :=
  λ i, p i + p ((i + k) % n)

/-- Prove that for all pairs (n, k) with gcd(n, k) = d, the condition 2d ∣ ¬ n is equivalent to saying 
that every two distinct permutations will have distinct k-mutations. --/
theorem distinct_k_mutations_of_permutations (n k d : ℕ) (h_gcd : Nat.gcd n k = d) :
  2 * d ∣ n → False :=
sorry

end distinct_k_mutations_of_permutations_l430_430498


namespace solve_system_of_equations_l430_430944

theorem solve_system_of_equations (x y : Real) : 
  (3 * x^2 + 3 * y^2 - 2 * x^2 * y^2 = 3) ∧ 
  (x^4 + y^4 + (2/3) * x^2 * y^2 = 17) ↔
  ( (x = Real.sqrt 2 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3 )) ∨ 
    (x = -Real.sqrt 2 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3)) ∨ 
    (x = Real.sqrt 3 ∧ (y = Real.sqrt 2 ∨ y = -Real.sqrt 2 )) ∨ 
    (x = -Real.sqrt 3 ∧ (y = Real.sqrt 2 ∨ y = -Real.sqrt 2 )) ) := 
  by
    sorry

end solve_system_of_equations_l430_430944


namespace propositions_correctness_l430_430387

noncomputable def lines_and_planes (α β : Plane) (a b : Line) : Prop :=
(∀ (h₁ : b ⊆ α) (h₂ : ¬(a ⊆ α)), (a ∥ b → a ∥ α)) ∧ (α ∥ β ↔ (α ∥ β ∧ b ∥ β)) = false

theorem propositions_correctness {α β : Plane} {a b : Line} :
  lines_and_planes α β a b ↔ (Prop ① is true ∧ Prop ② is false) :=
sorry

end propositions_correctness_l430_430387


namespace euler_diff_eq_solution_l430_430364

theorem euler_diff_eq_solution (C1 C2 : ℝ) (x : ℝ)
  (h : ∀ (y : ℝ → ℝ), (∀ x, y x = C1 * (x⁻³) + C2 * (x²)) → (∀ x, x^2 * (deriv^[2] y x) + 2 * x * (deriv y x) - 6 * (y x) = 0)) :
  ∀ y : ℝ → ℝ, y = λ x, C1 * (x⁻³) + C2 * (x²) :=
sorry

end euler_diff_eq_solution_l430_430364


namespace total_students_l430_430581

theorem total_students (N : ℕ) (h1 : ∃ g1 g2 g3 g4 g5 g6 : ℕ, 
  g1 = 13 ∧ g2 = 13 ∧ g3 = 13 ∧ g4 = 13 ∧ 
  ((g5 = 12 ∧ g6 = 12) ∨ (g5 = 14 ∧ g6 = 14)) ∧ 
  N = g1 + g2 + g3 + g4 + g5 + g6) : 
  N = 76 ∨ N = 80 :=
by
  sorry

end total_students_l430_430581


namespace quadratic_solution_l430_430779

theorem quadratic_solution (x : ℝ) : -x^2 + 4 * x + 5 < 0 ↔ x > 5 ∨ x < -1 :=
sorry

end quadratic_solution_l430_430779


namespace sqrt3_mul_sqrt12_eq_6_l430_430226

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l430_430226


namespace train_crossing_time_l430_430065

-- Definitions for conditions
def train_length : ℝ := 100 -- train length in meters
def train_speed_kmh : ℝ := 90 -- train speed in km/hr
def train_speed_mps : ℝ := 25 -- train speed in m/s after conversion

-- Lean 4 statement to prove the time taken for the train to cross the electric pole is 4 seconds
theorem train_crossing_time : (train_length / train_speed_mps) = 4 := by
  sorry

end train_crossing_time_l430_430065


namespace sufficient_condition_implies_range_l430_430390

def setA : Set ℝ := {x | 1 ≤ x ∧ x < 3}

def setB (a : ℝ) : Set ℝ := {x | x^2 - a * x ≤ x - a}

theorem sufficient_condition_implies_range (a : ℝ) :
  (∀ x, x ∉ setA → x ∉ setB a) → (1 ≤ a ∧ a < 3) :=
by
  sorry

end sufficient_condition_implies_range_l430_430390


namespace first_term_geometric_sequence_l430_430782

theorem first_term_geometric_sequence :
  ∀ (a b c : ℕ), 
    let r := 243 / 81 in 
    b = a * r ∧ 
    c = b * r ∧ 
    81 = c * r ∧ 
    243 = 81 * r → 
    a = 3 :=
by
  intros
  let r : ℕ := 243 / 81
  sorry

end first_term_geometric_sequence_l430_430782


namespace number_of_correct_propositions_l430_430646

-- Definitions reflecting the conditions from the problem
def proposition1 := ∀ x : ℝ, x > 0 → (2 / x) decreases as x increases -- Placeholder for the actual definition
def proposition2 := ∀ a b c : ℝ, b^2 - 4 * a * c < 0 → (y = a * x^2 + b * x + c) does not intersect the x-axis
def proposition3 := ∀ (circle congruent_circles : Type) (chord1 chord2 : chord circle), -- Placeholder for the definition of equal arcs intercepted by equal chords
  equal_chords_intercept_equal_arcs circle congruent_circles chord1 chord2
def proposition4 := ∀ (triangle1 triangle2 : isosceles_triangle),  -- Placeholder for definition that two isosceles triangles with one equal angle are similar
  
-- The problem statement
theorem number_of_correct_propositions : (proposition1 → False) ∧ proposition2 ∧ (proposition3 → False) ∧ (proposition4 → False) → 1 :=
by
  sorry

end number_of_correct_propositions_l430_430646


namespace distance_not_all_odd_l430_430940

theorem distance_not_all_odd (A B C D : ℝ × ℝ) : 
  ∃ (P Q : ℝ × ℝ), dist P Q % 2 = 0 := by sorry

end distance_not_all_odd_l430_430940


namespace abs_ineq_l430_430609

theorem abs_ineq (x : ℝ) (h : |x + 1| > 3) : x < -4 ∨ x > 2 :=
  sorry

end abs_ineq_l430_430609


namespace sqrt_multiplication_l430_430152

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l430_430152


namespace maximum_value_of_f_l430_430663

noncomputable def f (t : ℝ) : ℝ := ((3^t - 4 * t) * t) / (9^t)

theorem maximum_value_of_f : ∃ t : ℝ, f t = 1/16 :=
sorry

end maximum_value_of_f_l430_430663


namespace sqrt_mult_eq_six_l430_430321

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l430_430321


namespace gwens_bonus_is_900_l430_430441

noncomputable def GwenBonus : ℝ :=
  let B := 900 in
  let A := (2 * (B / 3)) in
  let B' := (2 * (B / 3)) in
  let C := ((B / 3) / 2) in
  let total_value := (A + B' + C) in
  total_value

theorem gwens_bonus_is_900 :
  ∃ (B : ℝ), (B / 3 * 2 + B / 3 * 2 + (B / 3 / 2) = 1350) ∧ B = 900 :=
by {
  use 900,
  split,
  calc
    900 / 3 * 2 + 900 / 3 * 2 + (900 / 3 / 2) = 600 + 600 + 150 : by norm_num
    ... = 1350 : by norm_num,
  rfl
}

end gwens_bonus_is_900_l430_430441


namespace smallest_divisible_by_primes_squared_l430_430049

theorem smallest_divisible_by_primes_squared : 
  ∃ n : ℕ+, (∀ p : ℕ, p = 2^2 ∨ p = 3^2 ∨ p = 5^2 → p ∣ n) ∧ n = 900 :=
begin
  sorry
end

end smallest_divisible_by_primes_squared_l430_430049


namespace trig_identity_proof_l430_430077

def sin_63_cos_18_plus_cos_63_cos_108 : ℝ :=
  Real.sin (63 * Real.pi / 180) * Real.cos (18 * Real.pi / 180) +
  Real.cos (63 * Real.pi / 180) * Real.cos (108 * Real.pi / 180)

theorem trig_identity_proof :
  sin_63_cos_18_plus_cos_63_cos_108 = Real.sqrt 2 / 2 := by
  sorry

end trig_identity_proof_l430_430077


namespace solve_phi_l430_430831

variable (φ : ℝ)
def f (x : ℝ) := Math.sin x * Math.cos φ + Math.cos x * Math.sin φ
def y (x : ℝ) := f (2 * x + (Real.pi / 6))

axiom is_on_graph : (2 * (Real.pi / 6) + (Real.pi / 6) + φ) = @Real.sin_inv 1 2

theorem solve_phi
  (h : 0 < φ) (h1 : φ < Real.pi) (hg : y (Real.pi / 6) = 1 / 2):
  φ = Real.pi / 3 :=
by sorry

end solve_phi_l430_430831


namespace score_for_5_hours_is_120_l430_430967

-- Definitions based on conditions
def score_variation (t : ℝ) : ℝ := 10 * t^2

-- Conditions
axiom score_for_3_hours : score_variation 3 = 90
axiom max_score : ℝ := 120

-- Problem statement
theorem score_for_5_hours_is_120 (t : ℝ) (h : t = 5) : min (score_variation t) max_score = 120 :=
by
  sorry

end score_for_5_hours_is_120_l430_430967


namespace cone_height_l430_430026

theorem cone_height (l : ℝ) (A : ℝ) (π : ℝ) (h : ℝ) : 
  l = 10 ∧ A = 60 ∧ π ≠ 0 ∧ (π^2 ≠ 0) → 
  h = real.sqrt (100 - (36 / π^2)) :=
by 
  intros h1 h2 h3 h4
  sorry

end cone_height_l430_430026


namespace athletes_and_probability_l430_430921

-- Given conditions and parameters
def total_athletes_a := 27
def total_athletes_b := 9
def total_athletes_c := 18
def total_selected := 6
def athletes := ["A1", "A2", "A3", "A4", "A5", "A6"]

-- Definitions based on given conditions and solution steps
def selection_ratio := total_selected / (total_athletes_a + total_athletes_b + total_athletes_c)

def selected_from_a := total_athletes_a * selection_ratio
def selected_from_b := total_athletes_b * selection_ratio
def selected_from_c := total_athletes_c * selection_ratio

def pairs (l : List String) : List (String × String) :=
  (List.bind l (λ x => List.map (λ y => (x, y)) l)).filter (λ (x,y) => x < y)

def all_pairs := pairs athletes

def event_A (pair : String × String) : Bool :=
  pair.fst = "A5" ∨ pair.snd = "A5" ∨ pair.fst = "A6" ∨ pair.snd = "A6"

def favorable_event_A := all_pairs.filter event_A

noncomputable def probability_event_A := favorable_event_A.length / all_pairs.length

-- The main theorem: Number of athletes selected from each association and probability of event A
theorem athletes_and_probability : selected_from_a = 3 ∧ selected_from_b = 1 ∧ selected_from_c = 2 ∧ probability_event_A = 3/5 := by
  sorry

end athletes_and_probability_l430_430921


namespace max_value_of_t_l430_430881

noncomputable def f (x : ℝ) : ℝ := Real.log x

noncomputable def t (m : ℝ) : ℝ := (2 * m + Real.log m / m - m * Real.log m) / 2

theorem max_value_of_t : ∃ (m : ℝ), m > 1 ∧ (∀ x : ℝ, x > 1 → t x ≤ (Real.exp 2 + 1) / (2 * Real.exp)) :=
by
  use Real.exp
  split
  · exact Real.exp_pos
  sorry

end max_value_of_t_l430_430881


namespace total_students_in_groups_l430_430536

theorem total_students_in_groups {N : ℕ} (h1 : ∃ g : ℕ → ℕ, (∀ i j, g i = 13 ∨ g j = 12 ∨ g j = 14) ∧ (∑ i in finset.range 6, g i) = N) : 
  N = 76 ∨ N = 80 :=
sorry

end total_students_in_groups_l430_430536


namespace range_of_lambda_l430_430398

noncomputable def sequence_a (n : ℕ) (hn : n > 0) : ℝ := 1 / (2 ^ n - 1)

def sequence_b (λ : ℝ) (n : ℕ) (hn : n > 0) : ℝ := (λ * n + 1) * (2 ^ n - 1) * sequence_a n hn

def Sn (λ : ℝ) (n : ℕ) (hn : n > 0) : ℝ := ∑ i in Finset.range n, sequence_b λ (i + 1) (Nat.succ_pos i)

theorem range_of_lambda (λ : ℝ) :
  (∀ n > 0, Sn λ n n > Sn λ (n + 1) (n + 1)) ∧ Sn λ 8 8 = max (λ n > 0, Sn λ n n) →
  λ ∈ Set.Ioo (-(1 / 8)) (-(1 / 9)) :=
sorry

end range_of_lambda_l430_430398


namespace sqrt_mul_eq_6_l430_430154

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l430_430154


namespace floor_log_sum_eq_twelve_l430_430786

theorem floor_log_sum_eq_twelve :
  (∑ n in Finset.range 12, Int.floor (Real.log n / Real.log 3)) = 12 :=
by
  sorry

end floor_log_sum_eq_twelve_l430_430786


namespace probability_slope_ge_1_l430_430500

noncomputable def point_in_unit_square (P : ℝ × ℝ) : Prop :=
  0 ≤ P.1 ∧ P.1 ≤ 1 ∧ 0 ≤ P.2 ∧ P.2 ≤ 1

def fixed_point_Q := (1/2 : ℝ, 1/2 : ℝ)

def slope_condition (P : ℝ × ℝ) : Prop :=
  (P.2 - fixed_point_Q.2) / (P.1 - fixed_point_Q.1) ≥ 1

theorem probability_slope_ge_1
  (P : ℝ × ℝ) (h : point_in_unit_square P):
  let prob_event := {p : ℝ × ℝ | point_in_unit_square p ∧ slope_condition p}.card / {p : ℝ × ℝ | point_in_unit_square p}.card in
  prob_event = 1/8 ∧ ∃ (m n : ℕ), Nat.coprime m n ∧ m ≠ 0 ∧ n ≠ 0 ∧ m/n = 1/8 ∧ m + n = 9 :=
by
  sorry

end probability_slope_ge_1_l430_430500


namespace line_perpendicular_l430_430693

variables {a b : Line} {α β : Plane}

-- Assuming the conditions
axiom line_parallel_plane (a : Line) (α : Plane) : Prop
axiom line_perpendicular_plane (b : Line) (α : Plane) : Prop

-- Definitions for the problem conditions
def line_parallel_to_plane (a : Line) (α : Plane) : Prop := line_parallel_plane a α
def line_perpendicular_to_plane (b : Line) (α : Plane) : Prop := line_perpendicular_plane b α

-- Main theorem corresponding to the problem statement
theorem line_perpendicular (a : Line) (b : Line) (α : Plane) (h1 : line_parallel_to_plane a α) (h2 : line_perpendicular_to_plane b α) : line_perpendicular a b :=
sorry  -- Proof placeholder

end line_perpendicular_l430_430693


namespace sqrt3_mul_sqrt12_eq_6_l430_430260

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l430_430260


namespace sqrt_mul_l430_430178

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l430_430178


namespace num_permutations_multiple_of_13_between_200_and_999_l430_430849

/-- 
Theorem: There are 122 integers between 200 and 999 inclusive, such that some permutation of their digits
is a multiple of 13 and also falls within the range 200 to 999 inclusive.
-/
theorem num_permutations_multiple_of_13_between_200_and_999 : 
  (finset.filter
    (λ n : ℕ, ∃ (m : ℕ) (hm : 200 ≤ m ∧ m ≤ 999), 
      m % 13 = 0 ∧ m.digits 10 ~ n.digits 10 ∧ 200 ≤ n ∧ n ≤ 999)
    (finset.range 800)).card = 122 :=
sorry

end num_permutations_multiple_of_13_between_200_and_999_l430_430849


namespace high_school_ten_total_games_l430_430006

theorem high_school_ten_total_games (n : ℕ) (h₁ : n = 10)
  (h₂ : ∀ team, ∃ games, games = 6):
  let conference_games := (n * (n - 1)) / 2 * 2 in
  let non_conference_games := n * 6 in
  let total_games := conference_games + non_conference_games in
  total_games = 150 :=
sorry

end high_school_ten_total_games_l430_430006


namespace total_students_76_or_80_l430_430573

theorem total_students_76_or_80 
  (N : ℕ)
  (h1 : ∃ g : ℕ → ℕ, (∑ i in finset.range 6, g i = N) ∧
                     (∃ a b : ℕ, finset.card {i | g i = a} = 4 ∧ 
                                 finset.card {i | g i = b} = 2 ∧ 
                                 (a = 13 ∧ (b = 12 ∨ b = 14))))
  : N = 76 ∨ N = 80 :=
sorry

end total_students_76_or_80_l430_430573


namespace geometric_sequence_formula_l430_430469

def geom_seq (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ r : ℝ, a (n + 1) = r * a n

theorem geometric_sequence_formula (a : ℕ → ℝ) (h_geom : geom_seq a)
  (h1 : a 3 = 2) (h2 : a 6 = 16) :
  ∀ n : ℕ, a n = 2 ^ (n - 2) :=
by
  sorry

end geometric_sequence_formula_l430_430469


namespace min_value_of_y_plus_dist_l430_430840

def parabola (x y : ℝ) : Prop := x^2 = -4 * y
def point_Q : ℝ × ℝ := (-2 * Real.sqrt 2, 0)
def dist (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem min_value_of_y_plus_dist (x y : ℝ) (h : parabola x y) :
  let P := (x, y),
      Q := point_Q,
      F := (0, -1),
      dist_directrix := Real.abs (y - 1),
      dist_PF := dist P F,
      dist_PQ := dist P Q,
      dist_FQ := dist F Q in
  Real.abs y + dist_PQ = 2 :=
by
  sorry

end min_value_of_y_plus_dist_l430_430840


namespace altered_prism_edges_l430_430761

theorem altered_prism_edges :
  let original_edges := 12
  let vertices := 8
  let edges_per_vertex := 3
  let faces := 6
  let edges_per_face := 1
  let total_edges := original_edges + edges_per_vertex * vertices + edges_per_face * faces
  total_edges = 42 :=
by
  let original_edges := 12
  let vertices := 8
  let edges_per_vertex := 3
  let faces := 6
  let edges_per_face := 1
  let total_edges := original_edges + edges_per_vertex * vertices + edges_per_face * faces
  show total_edges = 42
  sorry

end altered_prism_edges_l430_430761


namespace log_equation_l430_430353

theorem log_equation : ∃ (y : ℝ), log y 8 = log 64 4 ∧ y = 512 := 
by
  exist 512
  simp only [log]
  -- The logarithmic properties and transformations can be assumed to hold true
  sorry

end log_equation_l430_430353


namespace sqrt_mul_simp_l430_430209

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l430_430209


namespace triangle_sin_a_triangle_area_l430_430820

theorem triangle_sin_a (B : ℝ) (a b c : ℝ) (hB : B = π / 4)
  (h_bc : b = Real.sqrt 5 ∧ c = Real.sqrt 2 ∨ a = 3 ∧ c = Real.sqrt 2) :
  Real.sin A = (3 * Real.sqrt 10) / 10 :=
sorry

theorem triangle_area (B a b c : ℝ) (hB : B = π / 4) (hb : b = Real.sqrt 5)
  (h_ac : a + c = 3) : 1 / 2 * a * c * Real.sin B = Real.sqrt 2 - 1 :=
sorry

end triangle_sin_a_triangle_area_l430_430820


namespace roots_inverse_sum_eq_two_thirds_l430_430860

theorem roots_inverse_sum_eq_two_thirds {x₁ x₂ : ℝ} (h1 : x₁ ^ 2 + 2 * x₁ - 3 = 0) (h2 : x₂ ^ 2 + 2 * x₂ - 3 = 0) : 
  (1 / x₁) + (1 / x₂) = 2 / 3 :=
sorry

end roots_inverse_sum_eq_two_thirds_l430_430860


namespace correct_judgments_l430_430728

-- Define the directions
inductive Direction
| A | B | C | D

open Direction

-- Teams
structure Team := (direction : Direction)

-- Conditions
def condition1 (A : Team) : Prop := A.direction ≠ A ∧ A.direction ≠ D
def condition2 (B : Team) : Prop := B.direction ≠ A ∧ B.direction ≠ B
def condition3 (C : Team) : Prop := C.direction ≠ A ∧ C.direction ≠ B
def condition4 (D : Team) : Prop := D.direction ≠ C ∧ D.direction ≠ D
def condition5 (C D : Team) : Prop := (C.direction ≠ D) → (D.direction ≠ A)

-- Judgments
def judgment1 (A : Team) : Prop := A.direction = B
def judgment2 (B : Team) : Prop := B.direction = D
def judgment3 (C : Team) : Prop := C.direction = D
def judgment4 (D : Team) : Prop := D.direction = C

-- Theorem statement
theorem correct_judgments (A B C D : Team) :
  condition1 A → condition2 B → condition3 C → condition4 D → condition5 C D →
  (judgment1 A) ∧ (judgment3 C) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end correct_judgments_l430_430728


namespace student_groups_l430_430545

theorem student_groups (N : ℕ) :
  (∃ (n : ℕ), n = 13 ∧ ∃ (m : ℕ), m ∈ {12, 14} ∧ N = 4 * 13 + 2 * m) → (N = 76 ∨ N = 80) :=
by
  intro h
  obtain ⟨n, hn, m, hm, hN⟩ := h
  rw [hn, hN]
  cases hm with h12 h14
  case inl =>
    simp [h12]
  case inr =>
    simp [h14]
  sorry

end student_groups_l430_430545


namespace sin_segment_ratio_is_rel_prime_l430_430008

noncomputable def sin_segment_ratio : ℕ × ℕ :=
  let p := 1
  let q := 8
  (p, q)
  
theorem sin_segment_ratio_is_rel_prime :
  1 < 8 ∧ gcd 1 8 = 1 ∧ sin_segment_ratio = (1, 8) :=
by
  -- gcd 1 8 = 1
  have h1 : gcd 1 8 = 1 := by exact gcd_one_right 8
  -- 1 < 8
  have h2 : 1 < 8 := by decide
  -- final tuple
  have h3 : sin_segment_ratio = (1, 8) := by rfl
  exact ⟨h2, h1, h3⟩

end sin_segment_ratio_is_rel_prime_l430_430008


namespace total_distinct_sets_l430_430403

-- Define the sets A, B, and C
def A : set ℕ := {1, 2, 3, 4}
def B : set ℕ := {5, 6, 7}
def C : set ℕ := {8, 9}

-- Define the total number of distinct sets that can be formed
theorem total_distinct_sets : 
  let num_sets_formed := (A.card * B.card) + (A.card * C.card) + (B.card * C.card) 
  in num_sets_formed = 26 :=
by {
  -- Add the proof steps here
  sorry
}

end total_distinct_sets_l430_430403


namespace simple_interest_rate_l430_430457

theorem simple_interest_rate (P : ℝ) (T : ℝ) (hT : T = 15)
  (doubles_in_15_years : ∃ R : ℝ, (P * 2 = P + (P * R * T) / 100)) :
  ∃ R : ℝ, R = 6.67 := 
by
  sorry

end simple_interest_rate_l430_430457


namespace inscribed_circle_radius_eq_l430_430617

-- Define the geometric conditions
structure GeometryProblem where
  (A B M : Point)
  (r AB : ℝ)
  (inscribed_circle : Circle)
  (arc_AM : Arc)
  (arc_BM : Arc)
    (arc_AM_radius : arc_AM.radius = AB)
    (arc_BM_radius : arc_BM.radius = AB)
    (intersect_at_M : arc_AM ∩ arc_BM = {M})
    (inscribed_radius : inscribed_circle.radius = r)
    (inscribed_in_triangle : inscribed_circle.isInscribedInCurvilinearTriangle AMB)

-- Statement of the theorem to prove
theorem inscribed_circle_radius_eq (g : GeometryProblem) : 
  g.inscribed_radius = (3 / 8) * g.AB := 
sorry

end inscribed_circle_radius_eq_l430_430617


namespace value_of_a_l430_430411

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, x > 0.5 → 1 - a / 2^x > 0) → a = Real.sqrt 2 :=
by
  sorry

end value_of_a_l430_430411


namespace AD_squared_eq_2AC_squared_l430_430993

-- Definitions based on the above conditions
variables (O A B D C : Point)
variables (r : ℝ) (circle_O : Circle O r) (circle_O' : Circle O' r)
variables (h1 : B ∈ line_segment AO)
variables (h2 : ⊥ BD AO) -- BD perpendicular to AO
variables (h3 : D ∈ circle_O) (h4 : C ∈ circle_O')

-- The proof problem
theorem AD_squared_eq_2AC_squared
  (h1 : B ∈ line_segment AO) 
  (h2 : perpendicular BD AO) 
  (h3 : D ∈ circle_O) 
  (h4 : C ∈ circle_O') : 
  distance_squared A D = 2 * distance_squared A C := 
sorry

end AD_squared_eq_2AC_squared_l430_430993


namespace problem_C1_polar_eq_problem_C2_polar_eq_find_distance_AB_l430_430479

noncomputable def C1_parametric (α : ℝ) : ℝ × ℝ :=
  ⟨1 + cos α, sin α⟩

def C2_cartesian (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 = 1

def C1_polar_eq (θ : ℝ) : ℝ :=
  2 * cos θ

def C2_polar_eq (ρ θ : ℝ) : Prop :=
  ρ^2 * (1 + 2 * sin θ^2) = 3

def intersection_distance (ρ1 ρ2 : ℝ) : ℝ :=
  abs (ρ1 - ρ2)

theorem problem_C1_polar_eq : ∀ θ, C1_polar_eq θ = 2 * cos θ :=
  sorry

theorem problem_C2_polar_eq : ∀ (ρ θ), C2_polar_eq ρ θ ↔ (ρ^2 * (1 + 2 * sin θ^2) = 3) :=
  sorry

theorem find_distance_AB : intersection_distance 1 (sqrt 30 / 5) = sqrt 30 / 5 - 1 :=
  sorry

end problem_C1_polar_eq_problem_C2_polar_eq_find_distance_AB_l430_430479


namespace sin_double_angle_l430_430419

theorem sin_double_angle : 
  (∃ θ : ℝ, θ ≠ 0 ∧ θ ≠ π ∧ θ ≠ -π ∧ (∀x y : ℝ, y = 2 * x → tan θ = 2)) →
  sin (2 * θ + π / 4) = √2 / 10 :=
by
  intro h
  obtain ⟨θ, hθ1, hθ2, hθ3, h_tan⟩ := h
  sorry

end sin_double_angle_l430_430419


namespace prime_factorization_uniqueness_l430_430596

theorem prime_factorization_uniqueness (m n : ℕ) (p : ℕ → ℕ) :
  p(m) = p(n) → m = n :=
by
  sorry

end prime_factorization_uniqueness_l430_430596


namespace complex_integer_series_sum_l430_430335

-- Define the conditions
def i_powers_sum : ℤ :=
  ((-100 : ℤ) to 100).sum (λ k, (Complex.i ^ k))

def int_sum : ℤ :=
  (List.range' 1 201).sum

-- Define the final sum to be proved
theorem complex_integer_series_sum :
  ((List.range' 1 201).sum + ((Complex.i ^ (-100)) + (Complex.i ^ (-99)) + (Complex.i ^ (-98)) + ⋯ + (Complex.i ^ 0) + ⋯ + (Complex.i ^ 100))) = 20302 :=
by sorry

end complex_integer_series_sum_l430_430335


namespace count_integers_satisfying_conditions_l430_430378

def is_repeating_decimal (n : ℕ) : Prop :=
  let k := n + 1
  (k % 3 ≠ 0) ∧ ∃ p : ℕ, p ≠ 2 ∧ p ≠ 5 ∧ k % p = 0

theorem count_integers_satisfying_conditions :
  (finset.range (151 - 51 + 1)).filter (λ n, 50 ≤ n + 51 ∧ is_repeating_decimal (n + 50)).card = 67 := 
sorry

end count_integers_satisfying_conditions_l430_430378


namespace sqrt_mul_l430_430185

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l430_430185


namespace total_students_l430_430570

noncomputable def totalStudentsOptions (groups totalGroups specificGroupCount specificGroupSize otherGroupSizes : ℕ) : Set ℕ :=
  if totalGroups = 6 ∧ specificGroupCount = 4 ∧ specificGroupSize = 13 ∧ (otherGroupSizes = 12 ∨ otherGroupSizes = 14) then
    {52 + 2 * otherGroupSizes}
  else
    ∅

theorem total_students :
  totalStudentsOptions 6 4 13 12 = {76} ∧ totalStudentsOptions 6 4 13 14 = {80} :=
by
  -- This is where the proof would go, but we're skipping it as per instructions
  sorry

end total_students_l430_430570


namespace tangent_line_at_pi_over_two_l430_430383

/-- Define the parametric equations for the curve -/
def x (t : ℝ) : ℝ := 2 * (t - Real.sin t)
def y (t : ℝ) : ℝ := 2 * (1 - Real.cos t)

/-- The tangent line equation at t = π/2 should be X - Y - π + 4 = 0. -/
theorem tangent_line_at_pi_over_two:
  ∃ (X Y : ℝ), (t = π / 2) → (X - Y - π + 4 = 0) :=
  by sorry

end tangent_line_at_pi_over_two_l430_430383


namespace monotonic_intervals_max_min_values_l430_430837

def f (x : ℝ) := x^3 - 3*x
def f_prime (x : ℝ) := 3*(x-1)*(x+1)

theorem monotonic_intervals :
  (∀ x : ℝ, x < -1 → 0 < f_prime x) ∧ (∀ x : ℝ, -1 < x ∧ x < 1 → f_prime x < 0) ∧ (∀ x : ℝ, x > 1 → 0 < f_prime x) :=
  by
  sorry

theorem max_min_values :
  ∀ x ∈ Set.Icc (-1 : ℝ) 3, f x ≤ 18 ∧ f x ≥ -2 ∧ 
  (f 1 = -2) ∧
  (f 3 = 18) :=
  by
  sorry

end monotonic_intervals_max_min_values_l430_430837


namespace prst_coplanar_l430_430963

theorem prst_coplanar (A B C D P R S T O : Point) 
  (h_AB_noncoplanar : ¬ Collinear A B D)
  (h_on_lines : P ∈ Line A B ∧ R ∈ Line B C ∧ S ∈ Line C D ∧ T ∈ Line D A)
  (h_O_condition : (dist O P)^2 - (dist A P) * (dist B P) = 
                   (dist O R)^2 - (dist B R) * (dist C R) ∧
                   (dist O R)^2 - (dist B R) * (dist C R) = 
                   (dist O S)^2 - (dist C S) * (dist D S) ∧
                   (dist O S)^2 - (dist C S) * (dist D S) = 
                   (dist O T)^2 - (dist D T) * (dist A T)) : 
  Coplanar P R S T :=
sorry

end prst_coplanar_l430_430963


namespace certain_number_pow_divides_factorial_l430_430862
 

theorem certain_number_pow_divides_factorial (k : ℕ) (fact : ℕ) (a : ℕ) (h1 : fact = 15!) (h2 : k = 6): 
  k = 6 → a^k ∣ fact → a = 3 :=
by
  -- skip the proof
  sorry

end certain_number_pow_divides_factorial_l430_430862


namespace points_concyclic_l430_430688

theorem points_concyclic
  (A B C H D E F : Point)
  (h_abc_acute : AcuteTriangle A B C)
  (h_semicircle_AB : Semicircle (Segment A B) (outside := true))
  (h_semicircle_AC : Semicircle (Segment A C) (outside := true))
  (h_AH_perp_BC : Perpendicular (Line A H) (Line B C) intersecting_at H)
  (h_D_on_BC : OnSegment D B C ∧ D ≠ B ∧ D ≠ C)
  (h_DE_parallel_AC : Parallel (Line D E) (Line A C))
  (h_DF_parallel_AB : Parallel (Line D F) (Line A B))
  : Concyclic {D, E, F, H} :=
sorry

end points_concyclic_l430_430688


namespace spiritual_connection_probability_l430_430655

-- Definitions for conditions
def num_set := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def spiritual_connection (a b : ℕ) : Prop := (a ∈ num_set ∧ b ∈ num_set ∧ |a - b| ≤ 1)

-- Main theorem
theorem spiritual_connection_probability :
  let favorable_outcomes := { (a, b) | a ∈ num_set ∧ b ∈ num_set ∧ spiritual_connection a b }
  let total_outcomes := { (a, b) | a ∈ num_set ∧ b ∈ num_set }
  (favorable_outcomes.to_finset.card : ℚ) / (total_outcomes.to_finset.card : ℚ) = 1 / 4 := by
  sorry

end spiritual_connection_probability_l430_430655


namespace sqrt_mult_eq_six_l430_430328

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l430_430328


namespace probability_no_self_draws_l430_430129

theorem probability_no_self_draws :
  let total_outcomes := 6
  let favorable_outcomes := 2
  let probability := favorable_outcomes / total_outcomes
  probability = 1 / 3 :=
by
  sorry

end probability_no_self_draws_l430_430129


namespace sqrt3_mul_sqrt12_eq_6_l430_430233

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l430_430233


namespace sqrt_mul_l430_430171

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l430_430171


namespace angle_in_second_quadrant_l430_430458

-- Definitions of conditions
def sin2_pos : Prop := Real.sin 2 > 0
def cos2_neg : Prop := Real.cos 2 < 0

-- Statement of the problem
theorem angle_in_second_quadrant (h1 : sin2_pos) (h2 : cos2_neg) : 
    (∃ α, 0 < α ∧ α < π ∧ P = (Real.sin α, Real.cos α)) :=
by
  sorry

end angle_in_second_quadrant_l430_430458


namespace Prove_BC_half_AD_l430_430930

structure Trapezoid (A B C D E : Type) :=
  (A B C D E : Point)
  (AD : Segment A D)
  (BC : Segment B C)
  (ABE : Triangle A B E)
  (BCE : Triangle B C E)
  (CDE : Triangle C D E)
  (perimeter_abc : ∀ (AB : Segment A B) (BE : Segment B E) (AE : Segment A E), perimeter ABE = perimeter BCE)
  (perimeter_bce : ∀ (BC : Segment B C) (CE : Segment C E) (BE : Segment B E), perimeter BCE = perimeter CDE)

theorem Prove_BC_half_AD (A B C D E : Point) (trapezoid : Trapezoid A B C D E) :
  BC = AD / 2 :=
by
  -- Proof to be completed
  sorry

end Prove_BC_half_AD_l430_430930


namespace sqrt_mult_simplify_l430_430243

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l430_430243


namespace jacket_price_increase_l430_430966

theorem jacket_price_increase :
  ∀ (original_price reduced_price1 reduced_price2 : ℝ),
  reduced_price1 = original_price * 0.75 →
  reduced_price2 = reduced_price1 * 0.75 →
  approximately_equal ((original_price - reduced_price2) / reduced_price2 * 100) 77.78 := by
  intros original_price reduced_price1 reduced_price2 h1 h2
  have h3 : reduced_price1 = original_price * 0.75 := h1
  have h4 : reduced_price2 = reduced_price1 * 0.75 := h2
  sorry

end jacket_price_increase_l430_430966


namespace max_projections_permutations_l430_430914

theorem max_projections_permutations (A : Fin 8 → Point) :
  N(A) = 56 := 
sorry

end max_projections_permutations_l430_430914


namespace power_function_through_point_l430_430838

theorem power_function_through_point (f : ℝ → ℝ) (a : ℝ) (h : ∀ x, f x = x ^ a) :
  f 2 = sqrt 2 → f x = sqrt x :=
by
  intro h1
  have h_eq : 2 ^ a = sqrt 2 := h 2 ▸ h1
  sorry

end power_function_through_point_l430_430838


namespace sqrt_mul_eq_l430_430298

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l430_430298


namespace sqrt_mult_eq_six_l430_430314

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l430_430314


namespace sqrt3_mul_sqrt12_eq_6_l430_430263

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l430_430263


namespace number_of_ways_to_choose_bases_l430_430791

-- Definitions of the conditions
def num_students : Nat := 4
def num_bases : Nat := 3

-- The main statement that we need to prove
theorem number_of_ways_to_choose_bases : (num_bases ^ num_students) = 81 := by
  sorry

end number_of_ways_to_choose_bases_l430_430791


namespace remainder_53_factorial_mod_59_l430_430775

theorem remainder_53_factorial_mod_59 :
  (factorial 53) % 59 = 30 := 
sorry

end remainder_53_factorial_mod_59_l430_430775


namespace expression_value_l430_430673

theorem expression_value : (36 + 9) ^ 2 - (9 ^ 2 + 36 ^ 2) = -1894224 :=
by
  sorry

end expression_value_l430_430673


namespace sqrt_multiplication_l430_430146

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l430_430146


namespace sqrt_mult_simplify_l430_430236

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l430_430236


namespace joan_blue_balloons_l430_430902

theorem joan_blue_balloons : 
  (original_balloons : ℕ) (lost_balloons : ℕ) (remaining_balloons : ℕ) 
  (h1 : original_balloons = 9) (h2 : lost_balloons = 2) 
  (h3 : remaining_balloons = original_balloons - lost_balloons) : remaining_balloons = 7 := by
  sorry

end joan_blue_balloons_l430_430902


namespace volume_of_box_l430_430754

-- Define the dimensions of the box
variables (L W H : ℝ)

-- Define the conditions as hypotheses
def side_face_area : Prop := H * W = 288
def top_face_area : Prop := L * W = 1.5 * 288
def front_face_area : Prop := L * H = 0.5 * (L * W)

-- Define the volume of the box
def box_volume : ℝ := L * W * H

-- The proof statement
theorem volume_of_box (h1 : side_face_area H W) (h2 : top_face_area L W) (h3 : front_face_area L H W) : box_volume L W H = 5184 :=
by
  sorry

end volume_of_box_l430_430754


namespace positive_integer_solutions_l430_430768

theorem positive_integer_solutions {a b c : ℕ} :
  a^3 - b^3 - c^3 = 3 * a * b * c ∧ a^2 = 2 * (a + b + c) →
  (a = 4 ∧ ((b = 1 ∧ c = 3) ∨ (b = 2 ∧ c = 2) ∨ (b = 3 ∧ c = 1))) :=
begin
  sorry
end

end positive_integer_solutions_l430_430768


namespace sum_from_neg_50_to_75_l430_430672

def sum_of_integers (a b : ℤ) : ℤ :=
  (b * (b + 1)) / 2 - (a * (a - 1)) / 2

theorem sum_from_neg_50_to_75 : sum_of_integers (-50) 75 = 1575 := by
  sorry

end sum_from_neg_50_to_75_l430_430672


namespace crayons_after_finding_additional_l430_430513

theorem crayons_after_finding_additional (x y : ℕ) (hx : x = 7) (hy : y = 87) : y - x + 15 = 95 :=
by
  rw [hx, hy]
  sorry

end crayons_after_finding_additional_l430_430513


namespace total_students_76_or_80_l430_430574

theorem total_students_76_or_80 
  (N : ℕ)
  (h1 : ∃ g : ℕ → ℕ, (∑ i in finset.range 6, g i = N) ∧
                     (∃ a b : ℕ, finset.card {i | g i = a} = 4 ∧ 
                                 finset.card {i | g i = b} = 2 ∧ 
                                 (a = 13 ∧ (b = 12 ∨ b = 14))))
  : N = 76 ∨ N = 80 :=
sorry

end total_students_76_or_80_l430_430574


namespace multiply_powers_l430_430681

theorem multiply_powers (x : ℝ) : x^3 * x^3 = x^6 :=
by sorry

end multiply_powers_l430_430681


namespace area_of_region_l430_430333

noncomputable def calculateArea : ℝ :=
  let circleC_radius : ℝ := 3
  let circleC_center_x : ℝ := 3
  let circleC_center_y : ℝ := 5
  let circleD_radius : ℝ := 2
  let circleD_center_x : ℝ := 10
  let circleD_center_y : ℝ := 5

  let rectangle_width : ℝ := abs (circleD_center_x - circleC_center_x)
  let rectangle_height : ℝ := circleC_center_y

  let rectangle_area : ℝ := rectangle_width * rectangle_height

  let sectorC_area : ℝ := (1 / 2) * π * circleC_radius^2
  let sectorD_area : ℝ := (1 / 2) * π * circleD_radius^2

  rectangle_area - (sectorC_area + sectorD_area)

theorem area_of_region : calculateArea = 35 - 6.5 * π :=
by
  sorry

end area_of_region_l430_430333


namespace proposition_3_proposition_4_l430_430123

variables {Plane : Type} {Line : Type} 
variables {α β : Plane} {a b : Line}

-- Assuming necessary properties of parallel planes and lines being subsets of planes
axiom plane_parallel (α β : Plane) : Prop
axiom line_in_plane (l : Line) (p : Plane) : Prop
axiom line_parallel (l m : Line) : Prop
axiom lines_skew (l m : Line) : Prop
axiom lines_coplanar (l m : Line) : Prop
axiom lines_do_not_intersect (l m : Line) : Prop

-- Assume the given conditions
variables (h1 : plane_parallel α β) 
variables (h2 : line_in_plane a α)
variables (h3 : line_in_plane b β)

-- State the equivalent proof problem as propositions to be proved in Lean
theorem proposition_3 (h1 : plane_parallel α β) 
                     (h2 : line_in_plane a α) 
                     (h3 : line_in_plane b β) : 
                     lines_do_not_intersect a b :=
sorry

theorem proposition_4 (h1 : plane_parallel α β) 
                     (h2 : line_in_plane a α) 
                     (h3 : line_in_plane b β) : 
                     lines_coplanar a b ∨ lines_skew a b :=
sorry

end proposition_3_proposition_4_l430_430123


namespace find_ellipse_equation_and_fixed_point_l430_430808

noncomputable def ellipse_equation (a b : ℝ) (p : ℝ × ℝ) (e : ℝ) : Prop :=
  let (x, y) := p in
  (a > b ∧ b > 0 ∧ a = 2 * sqrt (a^2 - b^2) ∧ (e = (sqrt (a^2 - b^2)) / a) ∧ (e = 1 / 2) ∧ 
  (x, y) = (1, 3 / 2) ∧ (x^2 / a^2 + y^2 / b^2 = 1))

noncomputable def line_intersects_ellipse_and_passes_through_fixed_point 
      (a b k m x y : ℝ) (A B D : ℝ × ℝ) : Prop := 
  let (x_A, y_A) := A in
  let (x_B, y_B) := B in
  let (x_D, y_D) := D in
  (a = 2 * b ∧ b = sqrt(3) ∧ D = (2, 0) ∧ 
  (y = k * x + m) ∧ (y^2 / 3 + x^2 / 4 = 1) ∧ 
  (x_A, y_A) ≠ (2, 0) ∧ (x_B, y_B) ≠ (2, 0) ∧ 
  ((x_D - x_A) * (x_D - x_B) + (y_D - y_A) * (y_D - y_B) = 0) ∧ 
  ∃! m, m = -2 * k ∨ m = - (2 * k / 7) ∧ 
  (if m = - (2 * k / 7) then (k * (x - 2 / 7) = 0)) ∧ 
  (if m = -2 * k then (k * (x - 2) = 0)) ∧ 
  ((2/7, 0) ∈ A ∨ B))

-- Defining the final theorem statement
theorem find_ellipse_equation_and_fixed_point :
  ∃ (a b : ℝ) (C : ℝ × ℝ → Prop),
    ellipse_equation a b (1, 3 / 2) (1 / 2) ∧ 
    (let l : ℝ → ℝ := λ x, - ((2:ℝ)/(7:ℝ))*x in ∀ k : ℝ, ∀ m : ℝ,
    line_intersects_ellipse_and_passes_through_fixed_point a b k m 2 0 D (1 / 7) 0) :=
begin
  sorry
end

end find_ellipse_equation_and_fixed_point_l430_430808


namespace find_x_for_g_l430_430859

noncomputable def g (x : ℝ) : ℝ := (↑((x + 5) / 3) : ℝ)^(1/3 : ℝ)

theorem find_x_for_g :
  ∃ x : ℝ, g (3 * x) = 3 * g x ↔ x = -65 / 12 :=
by
  sorry

end find_x_for_g_l430_430859


namespace projection_same_and_collinear_l430_430676

noncomputable def a : ℝ × ℝ × ℝ := (1, -1, 2)
noncomputable def b : ℝ × ℝ × ℝ := (0, 3, 0)
noncomputable def expected_p : ℝ × ℝ × ℝ := (4/7, 5/7, 8/7)

def are_collinear (u v w : ℝ × ℝ × ℝ) : Prop :=
  ∃ t₁ t₂, u = (t₁ • v) ∧  u = (t₂ • w) 

theorem projection_same_and_collinear :
  ∃ v : ℝ × ℝ × ℝ, (∃ p : ℝ × ℝ × ℝ, 
    p = vector_projection v a ∧ 
    p = vector_projection v b ∧ 
    are_collinear a b p ) → 
    p = expected_p :=
sorry

end projection_same_and_collinear_l430_430676


namespace total_students_total_students_alt_l430_430547

def number_of_students (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 
    a + b = 6 ∧ 
    a = 4 ∧ 
    (∀ g, g = 13) ∧ 
    (b = 2 ∧ ((g = 12 ∨ g = 14) ∧ ∀ h, g - h ≤ 1)) ∧ 
    (n = 52 + 2 * 12 ∨ n = 52 + 2 * 14)

theorem total_students : ∃ n : ℕ, number_of_students n :=
by
  use 76
  sorry

theorem total_students_alt : ∃ n : ℕ, number_of_students n :=
by
  use 80
  sorry

end total_students_total_students_alt_l430_430547


namespace lines_identical_pairs_count_l430_430502

theorem lines_identical_pairs_count :
  (∃ a d : ℝ, (4 * x + a * y + d = 0 ∧ d * x - 3 * y + 15 = 0)) →
  (∃! n : ℕ, n = 2) := 
sorry

end lines_identical_pairs_count_l430_430502


namespace length_of_side_edges_is_correct_l430_430949

noncomputable def edge_length_of_regular_tetrahedron (base_edge : ℝ) (side_face_angle : ℝ) : ℝ :=
  if base_edge = 1 ∧ side_face_angle = 120 then
    (Real.sqrt 6) / 4
  else
    0

theorem length_of_side_edges_is_correct :
  ∀ (base_edge : ℝ) (side_face_angle : ℝ),
    base_edge = 1 ∧ side_face_angle = 120 →
    edge_length_of_regular_tetrahedron base_edge side_face_angle = (Real.sqrt 6) / 4 :=
by
  intros base_edge side_face_angle h
  rw [edge_length_of_regular_tetrahedron]
  simp [h]
  sorry

end length_of_side_edges_is_correct_l430_430949


namespace sqrt_mul_simplify_l430_430267

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l430_430267


namespace sqrt_mul_sqrt_eq_six_l430_430290

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l430_430290


namespace polynomial_equality_l430_430906

noncomputable def polynomials_equal (P Q : ℂ[X]) : Prop :=
  (∃ k : ℂ, P = Q * C k) ∧ P.degree ≥ 1 ∧ Q.degree ≥ 1

theorem polynomial_equality (P Q : ℂ[X])
  (hP_degree : P.degree ≥ 1)
  (hQ_degree : Q.degree ≥ 1)
  (hP0Q0 : ∀ z : ℂ, P.eval z = 0 ↔ Q.eval z = 0)
  (hP1Q1 : ∀ z : ℂ, P.eval z = 1 ↔ Q.eval z = 1) :
  P = Q :=
sorry

end polynomial_equality_l430_430906


namespace sum_integers_neg50_to_75_l430_430670

-- Definitions representing the conditions
def symmetric_sum_to_zero (a b : ℤ) : Prop :=
  (a = -b) → (a + b = 0)

def arithmetic_series_sum (first last terms : ℤ) : ℤ :=
  let average := (first + last) / 2
  average * terms

-- The theorem we need to state
theorem sum_integers_neg50_to_75 : 
  (symmetric_sum_to_zero (-50) 50) →
  arithmetic_series_sum 51 75 25 = 1575 →
  sum (range (76 + 50 - 1)) = 1575 := 
sorry

end sum_integers_neg50_to_75_l430_430670


namespace sqrt_mul_eq_l430_430300

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l430_430300


namespace batsman_average_runs_l430_430619

/-- A batsman played 15 matches with an average of 30 runs per match,
    and 20 matches with an average of 15 runs per match. -/
theorem batsman_average_runs (matches1 matches2 avg1 avg2 total_matches : ℕ) 
  (h1 : matches1 = 15) (h2 : matches2 = 20) (h3 : avg1 = 30) (h4 : avg2 = 15) 
  (h_total : total_matches = 35) : 
  (matches1 * avg1 + matches2 * avg2) / total_matches = 21.43 :=
by
  sorry

end batsman_average_runs_l430_430619


namespace dilation_rotation_l430_430010

noncomputable def center : ℂ := 2 + 3 * Complex.I
noncomputable def scale_factor : ℂ := 3
noncomputable def initial_point : ℂ := -1 + Complex.I
noncomputable def final_image : ℂ := -4 + 12 * Complex.I

theorem dilation_rotation (z : ℂ) :
  z = (-1 + Complex.I) →
  let z' := center + scale_factor * (initial_point - center)
  let rotated_z := center + Complex.I * (z' - center)
  rotated_z = final_image := sorry

end dilation_rotation_l430_430010


namespace sqrt_mult_l430_430196

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l430_430196


namespace limit_arctg_lhs_and_rhs_l430_430069

open Real

noncomputable def limit_arctg_positive : Prop :=
  tendsto (λ (x : ℝ), arctan (1 / (2 - x))) (nhdsWithin 2 (Iio 2)) (nhds (π / 2))

noncomputable def limit_arctg_negative : Prop :=
  tendsto (λ (x : ℝ), arctan (1 / (2 - x))) (nhdsWithin 2 (Ioi 2)) (nhds (-π / 2))

-- We state the theorem without proof.
theorem limit_arctg_lhs_and_rhs :
  limit_arctg_positive ∧ limit_arctg_negative :=
by 
  split; sorry

end limit_arctg_lhs_and_rhs_l430_430069


namespace max_probability_first_black_ace_l430_430040

def probability_first_black_ace(k : ℕ) : ℚ :=
  if 1 ≤ k ∧ k ≤ 51 then (52 - k) / 1326 else 0

theorem max_probability_first_black_ace : 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 51 → probability_first_black_ace k ≤ probability_first_black_ace 1 :=
by
  sorry

end max_probability_first_black_ace_l430_430040


namespace sum_X_Y_Z_W_eq_156_l430_430737

theorem sum_X_Y_Z_W_eq_156 
  (X Y Z W : ℕ) 
  (h_arith_seq : Y - X = Z - Y)
  (h_geom_seq : Z / Y = 9 / 5)
  (h_W : W = Z^2 / Y) 
  (h_pos : 0 < X ∧ 0 < Y ∧ 0 < Z ∧ 0 < W) :
  X + Y + Z + W = 156 :=
sorry

end sum_X_Y_Z_W_eq_156_l430_430737


namespace ellipse_standard_equation_line_through_fixed_point_l430_430807

-- Proof Problem 1
theorem ellipse_standard_equation
  (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0)
  (perimeter_eq : 4 * a = 8)
  (delta_S_eq : c * b = sqrt 3)
  (eccentricity_lt : c / a < sqrt 3 / 2) :
  ∃ (a b : ℝ), ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ∧ (a = 2) ∧ (b = sqrt 3) := 
begin
  sorry
end

-- Proof Problem 2
theorem line_through_fixed_point
  (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (ellipse_eq : ∀ x y, (x^2 / a^2 + y^2 / b^2 = 1))
  (tangent_line_at : ∀ x0 y0 x y, (x0 * x / a^2 + y0 * y / b^2 = 1))
  (P_moves_along : ∀ x0 y0, (x0 + y0 = 3)) :
  ∃ (fixed_point : ℝ × ℝ), (fixed_point = (4 / 3, 1)) :=
begin
  sorry
end

end ellipse_standard_equation_line_through_fixed_point_l430_430807


namespace simplify_expression_l430_430599

variables (a : ℂ)
hypothesis (h1 : a ≠ 0)
hypothesis (h2 : a ≠ 1)
hypothesis (h3 : a ≠ -1)

theorem simplify_expression : (a - 1 / a) / ((a - 1) / a) = a + 1 :=
by
  sorry

end simplify_expression_l430_430599


namespace sqrt_mul_simp_l430_430210

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l430_430210


namespace circle_tangent_to_line_l430_430011

-- Defining the center of the circle
def center : ℝ × ℝ := (-1, 1)

-- Defining the line equation as a function
def line (x y : ℝ) : Prop := x - y = 0

-- Distance from a point to a line
def distance_point_to_line (p : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  (abs (A * p.1 + B * p.2 + C)) / (sqrt (A^2 + B^2))

-- Radius of the circle
def radius : ℝ := distance_point_to_line center 1 (-1) 0

-- Equation of the circle
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = radius^2

theorem circle_tangent_to_line :
  ∀ x y : ℝ, line x y → circle_equation x y :=
by
  sorry

end circle_tangent_to_line_l430_430011


namespace area_OMVK_l430_430771

theorem area_OMVK :
  ∀ (S_OKCL S_ONAM S_ONBM S_ABCD S_OMVK : ℝ),
    S_OKCL = 6 →
    S_ONAM = 12 →
    S_ONBM = 24 →
    S_ABCD = 4 * (S_OKCL + S_ONAM) →
    S_OMVK = S_ABCD - S_OKCL - S_ONAM - S_ONBM →
    S_OMVK = 30 :=
by
  intros S_OKCL S_ONAM S_ONBM S_ABCD S_OMVK h_OKCL h_ONAM h_ONBM h_ABCD h_OMVK
  rw [h_OKCL, h_ONAM, h_ONBM] at *
  sorry

end area_OMVK_l430_430771


namespace solve_quadratic_eq_solve_cubic_eq_l430_430373

-- Statement for the first equation
theorem solve_quadratic_eq (x : ℝ) : 9 * x^2 - 25 = 0 ↔ x = 5 / 3 ∨ x = -5 / 3 :=
by sorry

-- Statement for the second equation
theorem solve_cubic_eq (x : ℝ) : (x + 1)^3 - 27 = 0 ↔ x = 2 :=
by sorry

end solve_quadratic_eq_solve_cubic_eq_l430_430373


namespace panacea_arrangement_l430_430880

theorem panacea_arrangement : 
  let total_permutations := Nat.factorial 7 / (Nat.factorial 3 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1) in
  let aaa_together := Nat.factorial 5 in
  total_permutations - aaa_together = 720 :=
by
  let total_permutations := Nat.factorial 7 / (Nat.factorial 3 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1)
  let aaa_together := Nat.factorial 5
  show total_permutations - aaa_together = 720
  sorry

end panacea_arrangement_l430_430880


namespace sum_integers_neg50_to_75_l430_430669

-- Definitions representing the conditions
def symmetric_sum_to_zero (a b : ℤ) : Prop :=
  (a = -b) → (a + b = 0)

def arithmetic_series_sum (first last terms : ℤ) : ℤ :=
  let average := (first + last) / 2
  average * terms

-- The theorem we need to state
theorem sum_integers_neg50_to_75 : 
  (symmetric_sum_to_zero (-50) 50) →
  arithmetic_series_sum 51 75 25 = 1575 →
  sum (range (76 + 50 - 1)) = 1575 := 
sorry

end sum_integers_neg50_to_75_l430_430669


namespace sales_tax_difference_l430_430616

def item_price : ℝ := 20
def sales_tax_rate1 : ℝ := 0.065
def sales_tax_rate2 : ℝ := 0.06

theorem sales_tax_difference :
  (item_price * sales_tax_rate1) - (item_price * sales_tax_rate2) = 0.1 := 
by
  sorry

end sales_tax_difference_l430_430616


namespace sqrt3_mul_sqrt12_eq_6_l430_430221

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l430_430221


namespace train_B_start_time_l430_430117

noncomputable def train_from_A_start : ℕ := 8  -- 8 a.m. in hours
noncomputable def train_from_B_meet : ℕ := 12  -- 12 p.m. in hours
noncomputable def distance_AB : ℕ := 465       -- Distance between city A and city B in km
noncomputable def speed_A : ℕ := 60            -- Speed of train from city A in km/hr
noncomputable def speed_B : ℕ := 75            -- Speed of train from city B in km/hr

theorem train_B_start_time :
  let time_A := train_from_B_meet - train_from_A_start,            -- Time traveled by train A
      distance_covered_by_A := speed_A * time_A,                    -- Distance covered by train A
      remaining_distance := distance_AB - distance_covered_by_A,    -- Remaining distance covered by train B
      time_B := remaining_distance / speed_B,                       -- Time taken by train B to cover the remaining distance
      start_time_B := train_from_B_meet - time_B                    -- Starting time of train B
  in start_time_B = 9 :=
by
  sorry

end train_B_start_time_l430_430117


namespace sqrt_mul_simp_l430_430213

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l430_430213


namespace profit_percentage_correct_l430_430104

variable (wholesalePrice : ℝ) (retailPrice : ℝ) (discountRate : ℝ)

def percentageProfit (wholesalePrice retailPrice discountRate : ℝ) : ℝ :=
  let discountAmount := discountRate * retailPrice
  let sellingPrice := retailPrice - discountAmount
  let profit := sellingPrice - wholesalePrice
  (profit / wholesalePrice) * 100

theorem profit_percentage_correct :
  wholesalePrice = 90 → retailPrice = 120 → discountRate = 0.10 → 
  percentageProfit wholesalePrice retailPrice discountRate = 20 := by
  intros
  unfold percentageProfit
  rw [H, H_1, H_2]
  norm_num
  sorry

end profit_percentage_correct_l430_430104


namespace system_of_equations_abs_diff_l430_430845

theorem system_of_equations_abs_diff 
  (x y m n : ℝ) 
  (h₁ : 2 * x - y = m)
  (h₂ : x + m * y = n)
  (hx : x = 2)
  (hy : y = 1) : 
  |m - n| = 2 :=
by
  sorry

end system_of_equations_abs_diff_l430_430845


namespace percentage_square_area_in_rectangle_l430_430721

variable (s : ℝ)
variable (W : ℝ) (L : ℝ)
variable (hW : W = 3 * s) -- Width is 3 times the side of the square
variable (hL : L = (3 / 2) * W) -- Length is 3/2 times the width

theorem percentage_square_area_in_rectangle :
  (s^2 / ((27 * s^2) / 2)) * 100 = 7.41 :=
by 
  sorry

end percentage_square_area_in_rectangle_l430_430721


namespace ant_reaches_end_after_7_minutes_l430_430124

-- Define the rubber band length at the i-th minute
def band_length (i : ℕ) : ℕ := 4 + i

-- Define the fraction of the band covered by the ant in the i-th minute
def ant_fraction (i : ℕ) : ℚ := 1 / band_length i

-- Define the total distance covered by the ant after t minutes
def total_distance (t : ℕ) : ℚ := (Finset.range t).sum (λ i, ant_fraction i)

-- Prove that the total distance covered by the ant is greater than or equal to 1 after 7 minutes
theorem ant_reaches_end_after_7_minutes : total_distance 7 ≥ 1 := 
sorry

end ant_reaches_end_after_7_minutes_l430_430124


namespace sqrt_multiplication_l430_430141

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l430_430141


namespace hexagon_equilateral_triangles_l430_430128

theorem hexagon_equilateral_triangles (hexagon_area: ℝ) (num_hexagons : ℕ) (tri_area: ℝ) 
    (h1 : hexagon_area = 6) (h2 : num_hexagons = 4) (h3 : tri_area = 4) : 
    ∃ (num_triangles : ℕ), num_triangles = 8 := 
by
  sorry

end hexagon_equilateral_triangles_l430_430128


namespace circle_diameter_from_area_l430_430618

theorem circle_diameter_from_area (A r d : ℝ) (hA : A = 64 * Real.pi) (h_area : A = Real.pi * r^2) : d = 16 :=
by
  sorry

end circle_diameter_from_area_l430_430618


namespace coins_in_fourth_hour_l430_430986

def coins_first_hour : ℕ := 20
def coins_second_hour : ℕ := 30
def coins_third_hour : ℕ := 30
def coins_taken_out_fifth_hour : ℕ := 20
def coins_left_fifth_hour : ℕ := 100

theorem coins_in_fourth_hour (X : ℕ) :
  let total_before_fourth_hour := coins_first_hour + coins_second_hour + coins_third_hour in
  let total_after_fifth_hour := (total_before_fourth_hour + X) - coins_taken_out_fifth_hour in
  total_after_fifth_hour = coins_left_fifth_hour →
  X = 40 :=
by
  sorry

end coins_in_fourth_hour_l430_430986


namespace total_students_l430_430586

theorem total_students (N : ℕ) (h1 : ∃ g1 g2 g3 g4 g5 g6 : ℕ, 
  g1 = 13 ∧ g2 = 13 ∧ g3 = 13 ∧ g4 = 13 ∧ 
  ((g5 = 12 ∧ g6 = 12) ∨ (g5 = 14 ∧ g6 = 14)) ∧ 
  N = g1 + g2 + g3 + g4 + g5 + g6) : 
  N = 76 ∨ N = 80 :=
by
  sorry

end total_students_l430_430586


namespace rebate_percentage_l430_430904

theorem rebate_percentage (r : ℝ) (h1 : 0 ≤ r) (h2 : r ≤ 1) 
(h3 : (6650 - 6650 * r) * 1.10 = 6876.1) : r = 0.06 :=
sorry

end rebate_percentage_l430_430904


namespace rectangle_area_l430_430707

theorem rectangle_area (radius : ℝ) (ratio : ℝ) (width length : ℝ) 
  (h_radius : radius = 8)
  (h_ratio : ratio = 3)
  (h_width_diameter : width = 2 * radius)
  (h_length_ratio : length = ratio * width) :
  length * width = 768 :=
by
  rw [h_radius, h_ratio, h_width_diameter, h_length_ratio]
  simp
  norm_num
  sorry

end rectangle_area_l430_430707


namespace intersection_points_form_rectangle_l430_430964

theorem intersection_points_form_rectangle :
  let points := [(4, 3), (-4, -3), (3, 4), (-3, -4)] in
  ∃ p1 p2 p3 p4, 
  (p1, p2, p3, p4 ∈ points) ∧ 
  ((dist p1 p2 = dist p3 p4) ∧ 
   (dist p2 p3 = dist p4 p1) ∧ 
   (slope p1 p2 * slope p2 p3 = -1)) ∧
   ((dist p1 p4 = dist p2 p3) ∧ 
   (dist p3 p2 = dist p1 p4) ∧ 
   (slope p3 p2 * slope p2 p4 = -1))→ 
   (shape (p1, p2, p3, p4) = rectangle) :=
by
  sorry

end intersection_points_form_rectangle_l430_430964


namespace bisection_method_correctness_l430_430658

noncomputable def initial_interval_length : ℝ := 1
noncomputable def required_precision : ℝ := 0.01
noncomputable def minimum_bisections : ℕ := 7

theorem bisection_method_correctness :
  ∃ n : ℕ, (n ≥ minimum_bisections) ∧ (initial_interval_length / 2^n ≤ required_precision) :=
by
  sorry

end bisection_method_correctness_l430_430658


namespace find_k_l430_430750

def f (x : ℝ) : ℝ := 7 * x^2 + (3 / x) + 4

def g (x k : ℝ) : ℝ := x^3 - k * x

theorem find_k (k : ℝ) (h : f 3 - g 3 k = 5) : k = -12 :=
by
  sorry

end find_k_l430_430750


namespace trig_identity_unit_circle_point_l430_430417

theorem trig_identity (a : ℝ) :
  let P := (4, -3 : ℝ × ℝ)
  let r := Real.sqrt (P.1^2 + P.2^2)
  let sin_a := P.2 / r
  let cos_a := P.1 / r
  2 * sin_a - cos_a = -2 := by
    let P := (4, -3 : ℝ × ℝ)
    let r := Real.sqrt (P.1^2 + P.2^2)
    let sin_a := P.2 / r
    let cos_a := P.1 / r
    sorry

theorem unit_circle_point (a : ℝ) :
  let P := (4, -3 : ℝ × ℝ)
  let r := Real.sqrt (P.1^2 + P.2^2)
  let sin_a := P.2 / r
  let cos_a := P.1 / r
  (cos_a, sin_a) = (4 / 5, -3 / 5) := by
    let P := (4, -3 : ℝ × ℝ)
    let r := Real.sqrt (P.1^2 + P.2^2)
    let sin_a := P.2 / r
    let cos_a := P.1 / r
    sorry

end trig_identity_unit_circle_point_l430_430417


namespace sum_first_10_terms_l430_430432

def a_n (n : ℕ) : ℕ := 2^(n-1) + n

def S_10 : ℕ := (List.ofFn a_n [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).sum

theorem sum_first_10_terms : S_10 = 1078 :=
  by
    unfold S_10
    unfold a_n
    sorry

end sum_first_10_terms_l430_430432


namespace roots_opposite_signs_l430_430689

theorem roots_opposite_signs (m n k : ℝ) (hm : m ≠ 0): 
  g(k, m, n, k) * g(1/m, m, n, k) < 0 → (let r1 r2 := quadratic_roots m n k in r1 * r2 < 0) := 
by
  intros h
  have : (k / m * (m * k + n + 1)^2 < 0) := by sorry
  exact this
sorry

def g (x m n k : ℕ) : ℕ := m * x ^ 2 + n * x + k
def quadratic_roots (a b c : ℝ) : (ℝ, ℝ) := ((-b + sqrt(b^2 - 4*a*c)) / (2*a), (-b - sqrt(b^2 - 4*a*c)) / (2*a))

end roots_opposite_signs_l430_430689


namespace odd_function_property_l430_430439

-- Defining function f with the given properties.
def f : ℝ → ℝ := 
  λ x => if x > 0 then x^3 + 1 else -x^3 - 1

theorem odd_function_property (x : ℝ) (h_odd : ∀ x : ℝ, f(-x) = -f(x)) (h_pos : x > 0 → f(x) = x^3 + 1) :
  x < 0 → f(x) = -x^3 - 1 :=
by
  intro hx
  have h_neg := h_odd (-x)
  simp at h_neg
  exact sorry

end odd_function_property_l430_430439


namespace net_effect_sale_value_net_effect_sale_value_percentage_increase_l430_430864

def sale_value (P Q : ℝ) : ℝ := P * Q

theorem net_effect_sale_value (P Q : ℝ) :
  sale_value (0.8 * P) (1.8 * Q) = 1.44 * sale_value P Q :=
by
  sorry

theorem net_effect_sale_value_percentage_increase (P Q : ℝ) :
  (sale_value (0.8 * P) (1.8 * Q) - sale_value P Q) / sale_value P Q = 0.44 :=
by
  sorry

end net_effect_sale_value_net_effect_sale_value_percentage_increase_l430_430864


namespace reasoning_definition_l430_430682

-- Definitions of reasoning types
def EmotionalReasoning : Prop := 
  "infers certain characteristics based on feelings"

def DeductiveReasoning : Prop := 
  "infers certain characteristics based on logical deducing from general principles"

def AnalogicalReasoning : Prop := 
  "infers certain characteristics based on analogy with known similar cases"

def InductiveReasoning : Prop :=
  "infers that all objects of a certain class have certain characteristics based on the observation that some objects of that class have these characteristics"

-- Theorem statement
theorem reasoning_definition :
  ("Which type of reasoning infers that all objects of a certain class have certain characteristics based on the observation that some objects of that class have these characteristics?" = InductiveReasoning) :=
sorry

end reasoning_definition_l430_430682


namespace find_ab_l430_430407

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 35) : a * b = 6 :=
by
  sorry

end find_ab_l430_430407


namespace trigonometric_identity_l430_430866

open Real

noncomputable def sin_alpha (x y : ℝ) : ℝ :=
  y / sqrt (x^2 + y^2)

noncomputable def tan_alpha (x y : ℝ) : ℝ :=
  y / x

theorem trigonometric_identity (x y : ℝ) (h_x : x = 3/5) (h_y : y = -4/5) :
  sin_alpha x y * tan_alpha x y = 16/15 :=
by {
  -- math proof to be provided here
  sorry
}

end trigonometric_identity_l430_430866


namespace num_integers_between_sqrt10_sqrt50_l430_430446

theorem num_integers_between_sqrt10_sqrt50 : 
  ∃ n : ℕ, n = 4 ∧ 
    ∀ x : ℤ, (⌈real.sqrt 10⌉ ≤ x ∧ x ≤ ⌊real.sqrt 50⌋) → x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 :=
by
  sorry

end num_integers_between_sqrt10_sqrt50_l430_430446


namespace sqrt_mult_l430_430187

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l430_430187


namespace determine_a_l430_430422

theorem determine_a (a : ℝ) : 
  let f := λ x : ℝ, (x^2 + a) / (x + 1) in
  ∃ a : ℝ, (∃ x : ℝ, x = 1 ∧ (deriv f x = 1)) → a = -1 :=
by
  sorry

end determine_a_l430_430422


namespace angle_ABC_is_60_l430_430009

noncomputable def degree (angle : ℝ) := angle

structure Rectangle where
  all_angles_90 : ∀ {A B C D : Type}, 
    IsRectangle A B C D → ∀ a b c d : ℝ, 
    Angle a = degree 90 ∧ 
    Angle b = degree 90 ∧ 
    Angle c = degree 90 ∧ 
    Angle d = degree 90

structure EquilateralTriangle where
  all_angles_60 : ∀ {A F G : Type}, 
    IsEquilateralTriangle A F G → ∀ a b c : ℝ, 
    Angle a = degree 60 ∧ 
    Angle b = degree 60 ∧ 
    Angle c = degree 60

structure Parallelogram where
  supplementary_angles : ∀ {A B C D : Type}, 
    IsParallelogram A B C D → ∀ a b : ℝ, 
    Angle a + Angle b = degree 180

variables
  (AGHB : Type) [IsRectangle AGHB]
  (AFG : Type) [IsEquilateralTriangle AFG]
  (ADEF : Type) [IsRectangle ADEF]
  (ABCD : Type) [IsParallelogram ABCD]

theorem angle_ABC_is_60 
  (h1 : ∀ a b c d : ℝ, Angle a = degree 90 ∧ Angle b = degree 90 ∧ Angle c = degree 90 ∧ Angle d = degree 90)
  (h2 : ∀ a b c : ℝ, Angle a = degree 60 ∧ Angle b = degree 60 ∧ Angle c = degree 60)
  (h3 : ∀ a b : ℝ, Angle a + Angle b = degree 180)
  : ∃ abc : ℝ, Angle abc = degree 60 := sorry

end angle_ABC_is_60_l430_430009


namespace sin_value_tan_value_l430_430416

theorem sin_value (α : ℝ) :
  let cos_alpha := (3 * real.sqrt 10 / 10)
  let sin_alpha := (real.sqrt 10 / 10)
  sin (2 * α + real.pi / 6) = (3 * real.sqrt 3 - 4) / 10 :=
by
  aα_eq : cos (α + real.pi / 6) = cos_alpha := sorry
  aα_eq' : sin (α + real.pi / 6) = sin_alpha := sorry
  have hα_cos_sq : cos_alpha ^ 2 + sin_alpha ^ 2 = 1, by
    calc
      cos_alpha ^ 2 + sin_alpha ^ 2
          = (3 * real.sqrt 10 / 10) ^ 2 + (real.sqrt 10 / 10) ^ 2 : by sorry
      _ = 1 : by sorry
  have two_alpha := 2 * α + real.pi / 6
  exact sorry

theorem tan_value (α β : ℝ) :
  tan (α + β) = 2 / 5 →
  tan (2 * β - real.pi / 3) = 17 / 144 :=
by
  have h_tan := real.tan_add 2 * α 2 * β
  exact sorry

end sin_value_tan_value_l430_430416


namespace committee_members_count_l430_430878

theorem committee_members_count :
  (∀ (members : Set (Set ℕ)), 
    (∀ m ∈ members, ∃ (c₁ c₂ : ℕ), c₁ ≠ c₂ ∧ (c₁, c₂) ∈ m) ∧
    (∀ (c₁ c₂ : ℕ), Set.card {m | {c₁, c₂} ⊆ m ∧ m ∈ members} = 1) ∧
    Set.card (Set.of (λ c, c < 5)) = 5) →
  Set.card members = 10 := by
  sorry

end committee_members_count_l430_430878


namespace missing_number_is_twelve_l430_430776

theorem missing_number_is_twelve
  (x : ℤ)
  (h : 10010 - x * 3 * 2 = 9938) :
  x = 12 :=
sorry

end missing_number_is_twelve_l430_430776


namespace first_term_geometric_sequence_l430_430781

theorem first_term_geometric_sequence :
  ∀ (a b c : ℕ), 
    let r := 243 / 81 in 
    b = a * r ∧ 
    c = b * r ∧ 
    81 = c * r ∧ 
    243 = 81 * r → 
    a = 3 :=
by
  intros
  let r : ℕ := 243 / 81
  sorry

end first_term_geometric_sequence_l430_430781


namespace zhen_takes_4th_l430_430331

theorem zhen_takes_4th 
    (different_candies : ∀ i j, i ≠ j → candies_taken i ≠ candies_taken j)
    (tian_takes_half_remaining : ∀ s, remaining_candies_after_tian (s - tian_takes_half_remaining s) = s / 2)
    (zhen_takes_two_thirds_remaining : ∀ s, remaining_candies_after_zhen (s - zhen_takes_two_thirds_remaining s) = s / 3)
    (mei_takes_all_remaining : remaining_candies_after_mei 0)
    (li_takes_half_remaining : ∀ s, remaining_candies_after_li (s - li_takes_half_remaining s) = s / 2) 
    : zhen_takes_4th ∧ min_candies = 16 :=
begin
  sorry
end

end zhen_takes_4th_l430_430331


namespace penny_canoe_l430_430524

theorem penny_canoe (P : ℕ)
  (h1 : 140 * (2/3 : ℚ) * P + 35 = 595) : P = 6 :=
sorry

end penny_canoe_l430_430524


namespace exponential_sum_l430_430640

-- It is necessary to define complex exponential function in Lean to clarify the conditions
open complex

-- Define the parameters and statement of the problem
noncomputable def problem_statement : Prop :=
  8 * exp ((2 * real.pi * I) / 13) + 8 * exp ((15 * real.pi * I) / 26)
  = 8 * real.sqrt 3 * exp ((19 * real.pi * I) / 52)

-- The theorem statement that needs to be proved
theorem exponential_sum : problem_statement :=
  sorry

end exponential_sum_l430_430640


namespace equation_of_parabola_l430_430015

def parabola_passes_through_point (a h : ℝ) : Prop :=
  2 = a * (8^2) + h

def focus_x_coordinate (a h : ℝ) : Prop :=
  h + (1 / (4 * a)) = 3

theorem equation_of_parabola :
  ∃ (a h : ℝ), parabola_passes_through_point a h ∧ focus_x_coordinate a h ∧
    (∀ x y : ℝ, x = (15 / 256) * y^2 - (381 / 128)) :=
sorry

end equation_of_parabola_l430_430015


namespace arctan_sum_in_triangle_l430_430482

theorem arctan_sum_in_triangle (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_triangle : ∠C > 0) :
  arctan (a / (b + c - a)) + arctan (b / (a + c - b)) = π / 4 :=
by
  sorry

end arctan_sum_in_triangle_l430_430482


namespace inscribed_circle_radius_square_l430_430704

theorem inscribed_circle_radius_square (ER RF GS SH : ℕ) (r : ℕ)
  (hER : ER = 24) (hRF : RF = 31) (hGS : GS = 40) (hSH : SH = 29)
  (htangent_eq: 
    arctan (24 / r) + arctan (31 / r) + arctan (40 / r) + arctan (29 / r) = 180) :
  r^2 = 945 :=
by { sorry }

end inscribed_circle_radius_square_l430_430704


namespace locus_is_line_segment_l430_430018

theorem locus_is_line_segment :
  let P1 := (2 : ℝ, 1 : ℝ)
      P2 := (-2 : ℝ, -2 : ℝ)
      distance (x y : ℝ × ℝ) : ℝ := real.sqrt ((x.1 - y.1)^2 + (x.2 - y.2)^2) 
  in ∀ (P : ℝ × ℝ), distance P P1 + distance P P2 = 5 → (∃ λ, P = (λ * P1.1 + (1 - λ) * P2.1, λ * P1.2 + (1 - λ) * P2.2)) :=
begin
  sorry
end

end locus_is_line_segment_l430_430018


namespace norm_squared_eq_l430_430614

def matrix_N (x y z w : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  ![
    ![0, 2*y, z, 0],
    ![x, y, -z, w],
    ![x, -y, z, -w],
    ![0, 2*w, y, -x]
  ]

theorem norm_squared_eq (x y z w : ℝ) (h : (matrix_N x y z w)ᵀ ⬝ (matrix_N x y z w) = 1) :
  x^2 + y^2 + z^2 + w^2 = 3 / 2 :=
sorry

end norm_squared_eq_l430_430614


namespace sqrt_mult_l430_430189

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l430_430189


namespace points_of_intersection_is_2_l430_430001

noncomputable def points_of_intersection {f : ℝ → ℝ} (h_inv : Function.Injective f) : ℕ :=
  {x : ℝ | f (x^3) = f (x^6)}.count

theorem points_of_intersection_is_2
  {f : ℝ → ℝ} (h_inv : Function.Injective f) :
  points_of_intersection h_inv = 2 :=
sorry

end points_of_intersection_is_2_l430_430001


namespace range_sum_distances_l430_430423

-- Definitions for the ellipse and related concepts
def is_on_ellipse (x y : ℝ) : Prop :=
    (x^2 / 2) + y^2 = 1

def is_inside_ellipse (x y : ℝ) : Prop :=
    0 < (x^2 / 2) + y^2 ∧ (x^2 / 2) + y^2 ≤ 1

-- Foci of the ellipse
def F1 : ℝ × ℝ := (1, 0)
def F2 : ℝ × ℝ := (-1, 0)

-- Distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
    Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Main theorem statement
theorem range_sum_distances (x0 y0 : ℝ) (h : is_inside_ellipse x0 y0) : 
    2 ≤ distance (x0, y0) F1 + distance (x0, y0) F2 ∧ 
    distance (x0, y0) F1 + distance (x0, y0) F2 ≤ 2 * Real.sqrt 2 :=
sorry

end range_sum_distances_l430_430423


namespace sqrt3_mul_sqrt12_eq_6_l430_430250

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l430_430250


namespace aviana_brought_pieces_l430_430870

variable (total_people : ℕ) (fraction_eat_pizza : ℚ) (pieces_per_person : ℕ) (remaining_pieces : ℕ)

theorem aviana_brought_pieces (h1 : total_people = 15) 
                             (h2 : fraction_eat_pizza = 3 / 5) 
                             (h3 : pieces_per_person = 4) 
                             (h4 : remaining_pieces = 14) :
                             ∃ (brought_pieces : ℕ), brought_pieces = 50 :=
by sorry

end aviana_brought_pieces_l430_430870


namespace total_students_l430_430589

theorem total_students (n_groups : ℕ) (students_in_group : ℕ → ℕ)
    (h1 : n_groups = 6)
    (h2 : ∃ n : ℕ, (students_in_group n = 13) ∧ (finset.filter (λ g, students_in_group g = 13) (finset.range n_groups)).card = 4)
    (h3 : ∀ i j, i < n_groups → j < n_groups → abs (students_in_group i - students_in_group j) ≤ 1) :
    (∃ N, N = 76 ∨ N = 80) :=
begin
    sorry
end

end total_students_l430_430589


namespace BE_is_sqrt_10_l430_430983

-- Defining the context of the parallelogram and points
variables (A B C D E F G : Point)
variables (parallelogram_ABCD : Parallelogram A B C D)
variables (intersect_E : E ∈ Line(B, D))
variables (intersect_F : F ∈ Line(C, D))
variables (intersect_G : G ∈ Line(B, C))
variables (ratio_FG_FE : Ratio FG FE = 9)
variables (length_ED : Distance(E, D) = 1)

-- The goal is to find the length of BE
def find_BE : Real :=
  let BE := sqrt(10) in BE

theorem BE_is_sqrt_10
  (parallelogram_ABCD : Parallelogram A B C D)
  (intersect_E : E ∈ Line(B, D))
  (intersect_F : F ∈ Line(C, D))
  (intersect_G : G ∈ Line(B, C))
  (ratio_FG_FE : Ratio FG FE = 9)
  (length_ED : Distance(E, D) = 1) : 
  Distance(B, E) = sqrt(10) :=
by
  sorry

end BE_is_sqrt_10_l430_430983


namespace sqrt_multiplication_l430_430140

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l430_430140


namespace compute_b_c_over_a_l430_430420

variable (a b c d : ℂ) -- coefficients of the cubic equation
variable (a_ne_zero : a ≠ 0) -- condition: a ≠ 0
variable (r : ℂ) -- the remaining root

-- The cubic equation's given roots are 1 + i and 1 - i
def polynomial_with_roots (x : ℂ) := a * (x - (1 + complex.I)) * (x - (1 - complex.I)) * (x - r)

-- Sum of the roots is 3 (given)
def sum_of_roots := (1 + complex.I) + (1 - complex.I) + r = 3

noncomputable def b_c_sum_div_a := (b + c) / a

-- Final statement to prove
theorem compute_b_c_over_a : (a * (1 + complex.I) * (1 - complex.I) * r + b * (1 + complex.I) * (1 - complex.I) + c * (1 + complex.I) + d ≠ 0)
→ sum_of_roots
→ b_c_sum_div_a = 2 := sorry

end compute_b_c_over_a_l430_430420


namespace triangle_angle_bisector_property_l430_430998

theorem triangle_angle_bisector_property (a b c f_c : ℝ) (γ: ℝ) 
(h1 : ∀ {a b f_c : ℝ}, (1 / a + 1 / b = 1 / f_c)) : 
γ = 120 :=
begin
  sorry
end

end triangle_angle_bisector_property_l430_430998


namespace tan_ratio_sum_l430_430448

theorem tan_ratio_sum (x y : ℝ) 
  (h1 : (sin x) / (cos y) + (sin y) / (cos x) = 2) 
  (h2 : (cos x) / (sin y) + (cos y) / (sin x) = 4) :
  (tan x) / (tan y) + (tan y) / (tan x) = 4 :=
by
  sorry

end tan_ratio_sum_l430_430448


namespace student_average_less_than_actual_average_l430_430114

variable {a b c : ℝ}

theorem student_average_less_than_actual_average (h : a < b) (h2 : b < c) :
  (a + (b + c) / 2) / 2 < (a + b + c) / 3 :=
by
  sorry

end student_average_less_than_actual_average_l430_430114


namespace internship_choices_l430_430789

theorem internship_choices :
  let choices := 3
  let students := 4
  (choices ^ students) = 81 := 
by
  intros
  calc
    3 ^ 4 = 81 : by norm_num

end internship_choices_l430_430789


namespace sqrt_mul_eq_l430_430310

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l430_430310


namespace area_of_quadrilateral_centroids_l430_430000

noncomputable def square_side_length : ℝ := 40
noncomputable def point_Q_XQ : ℝ := 15
noncomputable def point_Q_YQ : ℝ := 35

theorem area_of_quadrilateral_centroids (h1 : square_side_length = 40)
    (h2 : point_Q_XQ = 15)
    (h3 : point_Q_YQ = 35) :
    ∃ (area : ℝ), area = 800 / 9 :=
by
  sorry

end area_of_quadrilateral_centroids_l430_430000


namespace fraction_of_work_left_l430_430061

theorem fraction_of_work_left 
  (A_days : ℝ) (B_days : ℝ) (work_days : ℝ) 
  (A_work_rate : A_days = 15) 
  (B_work_rate : B_days = 30) 
  (work_duration : work_days = 4)
  : (1 - (work_days * ((1 / A_days) + (1 / B_days)))) = 3 / 5 := 
by
  sorry

end fraction_of_work_left_l430_430061


namespace find_range_of_a_l430_430833

-- Define the function f and its derivative f'
def f (a x : ℝ) : ℝ := (x - 1) * Real.exp x - (1/2) * a * x^2

def f_prime (a x : ℝ) : ℝ := x * (Real.exp x - a)

-- Define the condition to be satisfied: f' should be above ax^3 + x^2 - (a - 1)x
def condition (a x : ℝ) : Prop := f_prime a x > a * x^3 + x^2 - (a - 1) * x

-- Prove the range of a that satisfies the condition for all x in (0, +∞)
theorem find_range_of_a :
  ∀ (a : ℝ), (∀ x > 0, condition a x) ↔ a ≤ 1/2 :=
begin
  sorry
end

end find_range_of_a_l430_430833


namespace orchard_trees_l430_430626

theorem orchard_trees (n : ℕ) (hn : n^2 + 146 = 7890) : 
    n^2 + 146 + 31 = 89^2 := by
  sorry

end orchard_trees_l430_430626


namespace part_1_part_2_decreasing_part_3_m_value_l430_430511

noncomputable def f : ℝ → ℝ := sorry

variable (f_pos_reals : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x * f y)
variable (f_range_constraint : ∀ x : ℝ, 1 < x → 0 < f x ∧ f x < 1) 
variable (f_at_2 : f 2 = 1 / 9)

-- 1. Prove f(x)f(1/x) = 1 for x > 0
theorem part_1 (x : ℝ) (h : 0 < x) : f x * f (1 / x) = 1 := sorry

-- 2. Prove that f(x) is strictly decreasing on (0, +∞)
theorem part_2_decreasing {x1 x2 : ℝ} (hx1 : 0 < x1) (hx2 : 0 < x2) (h : x1 < x2) : f x1 > f x2 := sorry

-- 3. If f(m) = 3, determine the value of m
theorem part_3_m_value (m : ℝ) (h_pos : 0 < m) (h_f_m : f m = 3) : m = Real.sqrt(2) / 2 := sorry

end part_1_part_2_decreasing_part_3_m_value_l430_430511


namespace sqrt_mult_l430_430199

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l430_430199


namespace remaining_cube_edges_l430_430111

theorem remaining_cube_edges (s : ℕ) (t : ℕ) (n : ℕ) :
  s = 4 → t = 1 → n = 8 → 48 = 12 * (s / t - 2) :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end remaining_cube_edges_l430_430111


namespace mersenne_primes_less_than_1000_l430_430355

open Nat

-- Definitions and Conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_mersenne_prime (p : ℕ) : Prop := ∃ n : ℕ, is_prime n ∧ p = 2^n - 1

-- Theorem Statement
theorem mersenne_primes_less_than_1000 : {p : ℕ | is_mersenne_prime p ∧ p < 1000} = {3, 7, 31, 127} :=
by
  sorry

end mersenne_primes_less_than_1000_l430_430355


namespace derivative_sine_composite_l430_430136

theorem derivative_sine_composite (x : ℝ) : 
  deriv (λ x, Real.sin(2 * x ^ 2 + x)) x = (4 * x + 1) * Real.cos(2 * x ^ 2 + x) :=
  sorry

end derivative_sine_composite_l430_430136


namespace min_sum_4410_l430_430021

def min_sum (a b c d : ℕ) : ℕ := a + b + c + d

theorem min_sum_4410 :
  ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a * b * c * d = 4410 ∧ min_sum a b c d = 69 :=
sorry

end min_sum_4410_l430_430021


namespace question_one_question_two_l430_430973

noncomputable def arrangements_one_box_two_balls : Nat :=
  let balls := {1, 2, 3, 4}
  let boxes := {A, B, C, D}
  -- Proof that given these conditions, the number of arrangements where one box contains 2 balls is 144.
  144

theorem question_one :
  arrangements_one_box_two_balls = 144 :=
by
  sorry

noncomputable def arrangements_two_boxes_empty : Nat :=
  let balls := {1, 2, 3, 4}
  let boxes := {A, B, C, D}
  -- Proof that given these conditions, the number of arrangements where exactly two boxes are empty is 84.
  84

theorem question_two :
  arrangements_two_boxes_empty = 84 :=
by
  sorry

end question_one_question_two_l430_430973


namespace find_f_for_neg6_neg3_l430_430624

def f (x : ℝ) : ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f(x)
axiom symmetric_about_3 (x : ℝ) : f (3 + x) = f (3 - x)
axiom defined_on_open_interval (x : ℝ) (h0 : 0 < x) (h1 : x < 3) : f (x) = 2^x

theorem find_f_for_neg6_neg3 (x : ℝ) (h0 : -6 < x) (h1 : x < -3) : f (x) = -2^(x + 6) :=
sorry

end find_f_for_neg6_neg3_l430_430624


namespace sqrt_mul_sqrt_eq_six_l430_430288

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l430_430288


namespace solve_inequality_l430_430604

theorem solve_inequality (x : ℝ) : |x - 1| + |x - 2| > 5 ↔ (x < -1 ∨ x > 4) :=
by
  sorry

end solve_inequality_l430_430604


namespace harmonic_mean_closest_to_six_l430_430628

def harmonic_mean (a b : ℕ) : ℚ := (2 * a * b) / (a + b)

theorem harmonic_mean_closest_to_six : 
     |harmonic_mean 3 2023 - 6| < 1 :=
sorry

end harmonic_mean_closest_to_six_l430_430628


namespace inscribed_circle_center_locus_l430_430802

open EuclideanGeometry

variables {A B : Point -- Define points A and B
           (AB : LineSegment A B)} -- Define the segment AB

-- Define the locus problem as a theorem in Lean 4
theorem inscribed_circle_center_locus :
  ∀ (C : Point), 
  (center_of_inscribed_circle (triangle A B C)) ∈ interior_circle_with_diameter AB := 
sorry

end inscribed_circle_center_locus_l430_430802


namespace sqrt_mult_eq_six_l430_430324

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l430_430324


namespace sqrt_mul_simplify_l430_430271

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l430_430271


namespace election_votes_l430_430474

noncomputable def total_votes (V: ℕ) (VB: ℕ) : ℕ :=
    V

theorem election_votes (V: ℕ) (H0: 0.2 * V = 0.2 * (VB + (VB + 0.15 * V))) (H1: VB = 2834) 
    : total_votes V VB = 8720 :=
    sorry

end election_votes_l430_430474


namespace tan_theta_eq_one_half_l430_430917

theorem tan_theta_eq_one_half
  (θ : ℝ)
  (h1 : 0 < θ ∧ θ < real.pi / 2)
  (a : ℝ × ℝ)
  (ha : a = (real.cos θ, 2))
  (b : ℝ × ℝ)
  (hb : b = (-1, real.sin θ))
  (h2 : a.1 * b.1 + a.2 * b.2 = 0) :
  real.tan θ = 1 / 2 := 
sorry

end tan_theta_eq_one_half_l430_430917


namespace range_of_k_l430_430431

open Real

def f (x : ℝ) : ℝ := x^2 + 4/x^2 - 3

def g (k x : ℝ) : ℝ := k * x + 2

theorem range_of_k (k : ℝ) :
  (∀ x1 ∈ Icc (-1 : ℝ) 2, ∃ x2 ∈ Icc (1 : ℝ) (sqrt 3), g k x1 > f x2) ↔ (k ∈ Ioo (-1/2 : ℝ) 1) :=
by
  sorry

end range_of_k_l430_430431


namespace correct_calculation_l430_430057

theorem correct_calculation (x : ℝ) : x * x^2 = x^3 :=
by sorry

end correct_calculation_l430_430057


namespace total_students_total_students_alt_l430_430552

def number_of_students (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 
    a + b = 6 ∧ 
    a = 4 ∧ 
    (∀ g, g = 13) ∧ 
    (b = 2 ∧ ((g = 12 ∨ g = 14) ∧ ∀ h, g - h ≤ 1)) ∧ 
    (n = 52 + 2 * 12 ∨ n = 52 + 2 * 14)

theorem total_students : ∃ n : ℕ, number_of_students n :=
by
  use 76
  sorry

theorem total_students_alt : ∃ n : ℕ, number_of_students n :=
by
  use 80
  sorry

end total_students_total_students_alt_l430_430552


namespace student_groups_l430_430546

theorem student_groups (N : ℕ) :
  (∃ (n : ℕ), n = 13 ∧ ∃ (m : ℕ), m ∈ {12, 14} ∧ N = 4 * 13 + 2 * m) → (N = 76 ∨ N = 80) :=
by
  intro h
  obtain ⟨n, hn, m, hm, hN⟩ := h
  rw [hn, hN]
  cases hm with h12 h14
  case inl =>
    simp [h12]
  case inr =>
    simp [h14]
  sorry

end student_groups_l430_430546


namespace sqrt_multiplication_l430_430147

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l430_430147


namespace cargo_arrival_day_l430_430702

-- Definitions based on conditions
def navigation_days : Nat := 21
def customs_days : Nat := 4
def warehouse_days_from_today : Nat := 2
def departure_days_ago : Nat := 30

-- Definition represents the total transit time
def total_transit_days : Nat := navigation_days + customs_days + warehouse_days_from_today

-- Theorem to prove the cargo always arrives at the rural warehouse 1 day after leaving the port in Vancouver
theorem cargo_arrival_day : 
  (departure_days_ago - total_transit_days + warehouse_days_from_today = 1) :=
by
  -- Placeholder for the proof
  sorry

end cargo_arrival_day_l430_430702


namespace sum_and_ratio_l430_430641

theorem sum_and_ratio (x y : ℝ) (h1 : x + y = 480) (h2 : x / y = 0.8) : y - x = 53.34 :=
by
  sorry

end sum_and_ratio_l430_430641


namespace sqrt_mul_eq_6_l430_430156

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l430_430156


namespace smallest_positive_perfect_square_divisible_by_3_and_5_is_225_l430_430047

def smallest_perf_square_divisible_by_3_and_5 : ℕ :=
  let n := 15 in n * n

theorem smallest_positive_perfect_square_divisible_by_3_and_5_is_225 :
  smallest_perf_square_divisible_by_3_and_5 = 225 :=
by
  sorry

end smallest_positive_perfect_square_divisible_by_3_and_5_is_225_l430_430047


namespace sqrt_mul_eq_6_l430_430157

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l430_430157


namespace sqrt3_mul_sqrt12_eq_6_l430_430251

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l430_430251


namespace sqrt3_mul_sqrt12_eq_6_l430_430228

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l430_430228


namespace marbles_initial_l430_430924

-- Definitions of initial conditions:
def marbles_currently : ℕ := 30
def marbles_given_to_brother : ℕ := 60
def marbles_given_to_sister := 2 * marbles_given_to_brother
def marbles_given_to_friend := 3 * marbles_currently

-- Theorem statement corresponding to the problem:
theorem marbles_initial : 
  let total_given := marbles_given_to_brother + marbles_given_to_sister + marbles_given_to_friend
  in (total_given + marbles_currently = 300) :=
by
  let total_given := marbles_given_to_brother + marbles_given_to_sister + marbles_given_to_friend
  have h : total_given + marbles_currently = 300 := sorry
  exact h

end marbles_initial_l430_430924


namespace sqrt_mult_l430_430198

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l430_430198


namespace line_through_intersection_points_of_circles_l430_430954

theorem line_through_intersection_points_of_circles :
  (∀ x y : ℝ, x^2 + y^2 = 9 ∧ (x + 4)^2 + (y + 3)^2 = 8 → 4 * x + 3 * y + 13 = 0) :=
by
  sorry

end line_through_intersection_points_of_circles_l430_430954


namespace astronaut_revolutions_l430_430084

variable (R : ℝ) -- Radius of the smallest circle C3
variable (n : ℕ) (h : n > 2) -- n is an integer greater than 2

noncomputable def C1_radius := n * R -- Radius of the fixed circle C1
noncomputable def C2_radius := 2 * R -- Radius of the rolling circle C2
noncomputable def C3_radius := R      -- Radius of the rolling circle C3

theorem astronaut_revolutions (Rpos : 0 < R) 
: let num_revolutions := n - 1 in 
  True := sorry

end astronaut_revolutions_l430_430084


namespace hyperbola_eccentricity_is_sqrt2_plus_1_l430_430494

-- Define the parabola and hyperbola digaram and coordinates
-- The hyperbola to have the same focus as the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def hyperbola (x y a b : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

def focus (x y : ℝ) : Prop := x = 1 ∧ y = 0

def intersects (x y : ℝ) : Prop := parabola x y ∧ ∃ a b, hyperbola x y a b

def AF_perpendicular (x y : ℝ) : Prop := y ≠ 0

noncomputable def eccentricity_hyperbola (x y a b : ℝ) : ℝ :=
  let c := 1 in
  let e := c / a in
  e

theorem hyperbola_eccentricity_is_sqrt2_plus_1 
  {a b : ℝ} (ha : a > 0) (hb : b > 0)
  (h_focus : focus 1 0) 
  (h_intersection_A : intersects 1 2)
  (h_perpendicular_AF : AF_perpendicular 1 2) : 
  eccentricity_hyperbola 1 2 a b = sqrt 2 + 1 := 
sorry

end hyperbola_eccentricity_is_sqrt2_plus_1_l430_430494


namespace ed_more_marbles_l430_430762

-- Define variables for initial number of marbles
variables {E D : ℕ}

-- Ed had some more marbles than Doug initially.
-- Doug lost 8 of his marbles at the playground.
-- Now Ed has 30 more marbles than Doug.
theorem ed_more_marbles (h : E = (D - 8) + 30) : E - D = 22 :=
by
  sorry

end ed_more_marbles_l430_430762


namespace a_1_correct_l430_430436

open Real

-- Definition of the sequence.
def a_n (n : ℕ) : ℝ := (sqrt 2)^(n - 2)

-- Statement to prove that a_1 = sqrt(2)/2
theorem a_1_correct : a_n 1 = sqrt 2 / 2 := by
  -- Skip proof for now
  sorry

end a_1_correct_l430_430436


namespace solve_inequality_l430_430605

theorem solve_inequality (x : ℝ) : |x - 1| + |x - 2| > 5 ↔ (x < -1 ∨ x > 4) :=
by
  sorry

end solve_inequality_l430_430605


namespace limit_of_sum_squared_series_l430_430336

variables (a r : ℝ) -- variables a and r are real numbers
variable h : -2 < r ∧ r < 2 -- constraint on r

-- Define the first term and common ratio
def first_term : ℝ := 3 * a
def common_ratio : ℝ := r / 2

-- Define the series with added constant and squared terms.
def modified_series (n : ℕ) : ℝ := ((first_term a) * (common_ratio r) ^ n + 2) ^ 2

-- Define the sum of squared terms as a geometric series.
def sum_of_squared_series (n : ℕ) : ℝ :=
  (first_term a + 2) ^ 2 * (1 - (common_ratio r) ^ (2 * (n + 1))) / (1 - (common_ratio r) ^ 2)

-- The limit of the sum of the squared series is given by
theorem limit_of_sum_squared_series :
  (lim_n (sum_of_squared_series a r n)) = (first_term a + 2)^2 / (1 - (common_ratio r)^2) :=
by
  sorry

end limit_of_sum_squared_series_l430_430336


namespace points_on_angle_bisector_are_equidistant_l430_430632

theorem points_on_angle_bisector_are_equidistant 
  (P : Point) (A B C : Point) (h1 : angle A B C) (h2 : equidistant P A B) : 
    on_angle_bisector P A B :=
sorry

end points_on_angle_bisector_are_equidistant_l430_430632


namespace rational_points_two_coloring_l430_430597

-- Define the problem within the Lean framework
open_locale classical

def rational_point := ℚ × ℚ

def distance (p q : rational_point) : ℚ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem rational_points_two_coloring :
  ∃ (f : rational_point → Prop), 
  (∀ p q : rational_point, distance p q = 1 → f p ≠ f q) :=
sorry

end rational_points_two_coloring_l430_430597


namespace part_a_part_b_l430_430926

-- Define the initial set of numbers
def initialSet := {i | 1 ≤ i ∧ i ≤ 2021}

-- Define the operation of selecting two numbers and replacing them with their absolute difference
def operation (s : Set ℕ) (a b : ℕ) : Set ℕ :=
  if a ∈ s ∧ b ∈ s then (s.erase a).erase b ∪ {abs (a - b)} else s

-- Prove that 2021 can be the last remaining number on the board
theorem part_a : ∃ s : Set ℕ, ((s = initialSet) ∧ ∃ a b, operation s a b = {2021}) :=
  sorry

-- Prove that 2020 cannot be the last remaining number on the board
theorem part_b : ¬ ∃ s : Set ℕ, ((s = initialSet) ∧ ∃ a b, operation s a b = {2020}) :=
  sorry

end part_a_part_b_l430_430926


namespace sqrt_mul_eq_6_l430_430164

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l430_430164


namespace chickadees_on_one_tree_after_finite_shots_l430_430030

theorem chickadees_on_one_tree_after_finite_shots :
  ∃ n : ℕ, ∀ (shot_count : ℕ), shot_count > n → 
    ∀ (trees : Fin 120 → ℕ), (∑ i, trees i = 2022) → 
    (∀ i j : Fin 120, trees i ≤ trees j + 1) → 
    (∃ k : Fin 120, ∀ m : Fin 120, trees m = 0 ∨ m = k) :=
by 
  sorry

end chickadees_on_one_tree_after_finite_shots_l430_430030


namespace incorrect_observation_l430_430958

theorem incorrect_observation (n : ℕ) (mean_original mean_corrected correct_obs incorrect_obs : ℝ)
  (h1 : n = 40) 
  (h2 : mean_original = 36) 
  (h3 : mean_corrected = 36.45) 
  (h4 : correct_obs = 34) 
  (h5 : n * mean_original = 1440) 
  (h6 : n * mean_corrected = 1458) 
  (h_diff : 1458 - 1440 = 18) :
  incorrect_obs = 52 :=
by
  sorry

end incorrect_observation_l430_430958


namespace exactly_one_correct_l430_430934

noncomputable def Proposition1 : Prop := 
  ∀ (P Q R : Plane), (P ⊥ Q) → (R ⊥ Q) → (P = R ∨ P ∩ R = ∅)

noncomputable def Proposition2 : Prop := 
  ∀ (l m : Line) (α : Plane), l ≠ m → (l ⊥ α) → (l ∥ m) → (m ⊥ α)

noncomputable def Proposition3 : Prop := 
  ∀ (α β : Plane) (m : Line), α ≠ β → (m ∈ α) → ((α ⊥ β) ↔ (m ⊥ β))

noncomputable def Proposition4 : Prop := 
  ∀ (a b : Line) (P : Point), ¬(a ∥ b) → (∃ γ : Plane, (P ∈ γ) ∧ ((γ ⊥ a ∧ γ ∥ b) ∨ (γ ⊥ b ∧ γ ∥ a)))

theorem exactly_one_correct : (Proposition1 ∨ Proposition2 ∨ Proposition3 ∨ Proposition4) ∧
                              (¬Proposition1 ∧ ¬Proposition3 ∧ ¬Proposition4) ∧ 
                              Proposition2 := by
  sorry

end exactly_one_correct_l430_430934


namespace arithmetic_seq_sin_identity_l430_430884

theorem arithmetic_seq_sin_identity:
  ∀ (a : ℕ → ℝ), (a 2 + a 6 = (3/2) * Real.pi) → (Real.sin (2 * a 4 - Real.pi / 3) = -1 / 2) :=
by
  sorry

end arithmetic_seq_sin_identity_l430_430884


namespace functional_equation_solution_l430_430356

noncomputable def func_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * x + f x * f y) = x * f (x + y)

theorem functional_equation_solution (f : ℝ → ℝ) :
  func_equation f →
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) :=
sorry

end functional_equation_solution_l430_430356


namespace probability_more_wins_than_losses_l430_430334

theorem probability_more_wins_than_losses
  (n_matches : ℕ)
  (win_prob lose_prob tie_prob : ℚ)
  (h_sum_probs : win_prob + lose_prob + tie_prob = 1)
  (h_win_prob : win_prob = 1/3)
  (h_lose_prob : lose_prob = 1/3)
  (h_tie_prob : tie_prob = 1/3)
  (h_n_matches : n_matches = 8) :
  ∃ (m n : ℕ), Nat.gcd m n = 1 ∧ m / n = 5483 / 13122 ∧ (m + n) = 18605 :=
by
  sorry

end probability_more_wins_than_losses_l430_430334


namespace total_students_l430_430584

theorem total_students (N : ℕ) (h1 : ∃ g1 g2 g3 g4 g5 g6 : ℕ, 
  g1 = 13 ∧ g2 = 13 ∧ g3 = 13 ∧ g4 = 13 ∧ 
  ((g5 = 12 ∧ g6 = 12) ∨ (g5 = 14 ∧ g6 = 14)) ∧ 
  N = g1 + g2 + g3 + g4 + g5 + g6) : 
  N = 76 ∨ N = 80 :=
by
  sorry

end total_students_l430_430584


namespace sqrt_mul_sqrt_eq_six_l430_430287

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l430_430287


namespace sandwich_combinations_l430_430107

-- Define the conditions
constant breads : ℕ := 5
constant meats : ℕ := 7
constant cheeses : ℕ := 6

constant turkey : ℕ := 1
constant mozzarella : ℕ := 1

constant rye : ℕ := 1
constant salami : ℕ := 1

constant white : ℕ := 1
constant chicken : ℕ := 1

-- Define the problem statement
theorem sandwich_combinations : 
  let total := (breads * meats * cheeses)
  let exclude_turkey_mozzarella := breads
  let exclude_rye_salami := cheeses
  let exclude_white_chicken := cheeses
  total - (exclude_turkey_mozzarella + exclude_rye_salami + exclude_white_chicken) = 193 :=
by
  -- the proof would go here, but is omitted
  sorry

end sandwich_combinations_l430_430107


namespace area_BCD_l430_430886

open Real EuclideanGeometry

noncomputable def point := (ℝ × ℝ)
noncomputable def A : point := (0, 0)
noncomputable def B : point := (10, 24)
noncomputable def C : point := (30, 0)
noncomputable def D : point := (40, 0)

def area_triangle (p1 p2 p3 : point) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  0.5 * |x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)|

theorem area_BCD : area_triangle B C D = 12 := sorry

end area_BCD_l430_430886


namespace right_triangle_circles_touch_l430_430471

theorem right_triangle_circles_touch
  (a b r : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : r > 0) 
  (h4 : ∃ c, c = sqrt (a^2 + b^2)) 
  (h5 : ∀ h, h = r → ∃ x y, ∃ d (h = d),  x = r ∧ y = r ∧ d = sqrt ((r - x)^2 + (r - y)^2)) : 
  1 / a + 1 / b = 1 / r := 
by
  -- Proof steps will go here
  sorry

end right_triangle_circles_touch_l430_430471


namespace sqrt_mul_sqrt_eq_six_l430_430296

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l430_430296


namespace odd_function_property_l430_430440

-- Defining function f with the given properties.
def f : ℝ → ℝ := 
  λ x => if x > 0 then x^3 + 1 else -x^3 - 1

theorem odd_function_property (x : ℝ) (h_odd : ∀ x : ℝ, f(-x) = -f(x)) (h_pos : x > 0 → f(x) = x^3 + 1) :
  x < 0 → f(x) = -x^3 - 1 :=
by
  intro hx
  have h_neg := h_odd (-x)
  simp at h_neg
  exact sorry

end odd_function_property_l430_430440


namespace sqrt_mul_l430_430184

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l430_430184


namespace segments_after_cuts_l430_430956

-- Definitions from the conditions
def cuts : ℕ := 10

-- Mathematically equivalent proof statement
theorem segments_after_cuts : (cuts + 1 = 11) :=
by sorry

end segments_after_cuts_l430_430956


namespace min_colors_correct_l430_430033

def min_colors (n : Nat) : Nat :=
  if n = 1 then 1
  else if n = 2 then 2
  else 3

theorem min_colors_correct (n : Nat) : min_colors n = 
  if n = 1 then 1
  else if n = 2 then 2
  else 3 := by
  sorry

end min_colors_correct_l430_430033


namespace sqrt_mult_simplify_l430_430248

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l430_430248


namespace problem_solved_by_all_students_l430_430946

-- Definitions and conditions
def students (n : ℕ) := fin n
def problems (n : ℕ) := fin (2^(n-1))

variable (student_solved : students n → problems n → Prop)
variable (solved_by_all : problems n → Prop)

-- Conditions (axioms)
axiom pairwise_intersection_non_empty :
  ∀ (p1 p2 : problems n), p1 ≠ p2 → ∃ s : students n, student_solved s p1 ∧ student_solved s p2

axiom pairwise_symmetric_difference_non_empty :
  ∀ (p1 p2 : problems n), p1 ≠ p2 → ∃ s : students n, (student_solved s p1 ∧ ¬ student_solved s p2) ∨ (¬ student_solved s p1 ∧ student_solved s p2)

-- Statement to be proved
theorem problem_solved_by_all_students (n : ℕ) :
  ∃ p : problems n, ∀ s : students n, student_solved s p :=
sorry

end problem_solved_by_all_students_l430_430946


namespace sqrt_mul_simp_l430_430212

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l430_430212


namespace sqrt_mult_l430_430195

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l430_430195


namespace right_triangle_min_perimeter_multiple_13_l430_430106

theorem right_triangle_min_perimeter_multiple_13 :
  ∃ (a b c : ℕ), 
    (a^2 + b^2 = c^2) ∧ 
    (a % 13 = 0 ∨ b % 13 = 0) ∧
    (a < b) ∧ 
    (a + b > c) ∧ 
    (a + b + c = 24) :=
sorry

end right_triangle_min_perimeter_multiple_13_l430_430106


namespace sqrt3_mul_sqrt12_eq_6_l430_430257

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l430_430257


namespace range_of_f_l430_430633

def f (x : ℤ) : ℤ := x^2 - 1

theorem range_of_f : 
  {y : ℤ | ∃ x ∈ ({-1, 0, 1, 2} : set ℤ), f x = y} = {0, -1, 3} :=
by
  sorry

end range_of_f_l430_430633


namespace sqrt3_mul_sqrt12_eq_6_l430_430262

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l430_430262


namespace square_of_radius_l430_430706

-- Definitions based on conditions
def ER := 24
def RF := 31
def GS := 40
def SH := 29

-- The goal is to find square of radius r such that r^2 = 841
theorem square_of_radius (r : ℝ) :
  let R := ER
  let F := RF
  let G := GS
  let S := SH
  (∀ r : ℝ, (R + F) * (G + S) = r^2) → r^2 = 841 :=
sorry

end square_of_radius_l430_430706


namespace find_triangle_area_l430_430463

variable (a b c A B C : ℝ)

axioms
  (sideA : b = Real.sqrt 2 * a)
  (sideB : Real.sqrt 3 * Real.cos B = Real.sqrt 2 * Real.cos A)
  (sideC : c = Real.sqrt 3 + 1)
  (angleA : A = (π / 6))
  (angleB : B = (π / 4))
  (angleC : C = (7 * π / 12))

noncomputable def triangle_area (a b c A B : ℝ) : ℝ :=
  1 / 2 * a * c * Real.sin B

theorem find_triangle_area :
  triangle_area a b c A B = (Real.sqrt 3 + 1) / 2 := by 
  sorry

end find_triangle_area_l430_430463


namespace minimum_distance_sasha_dania_l430_430976

theorem minimum_distance_sasha_dania :
  let v_Sasha := 2700 -- cm/min
  let v_Dania := 3575 -- cm/min
  let d_initial_Sasha := 29000 -- cm
  let d_initial_Dania := 31000 -- cm
  let steps_Sasha := 396
  let steps_Dania := 484
  let t := 396 / 45 -- minutes
  let d_min := |d_initial_Sasha - v_Sasha * t - (d_initial_Dania - v_Dania * t)| / 100 -- convert to meters
  d_min = 57 ∧ steps_Sasha = 396 ∧ steps_Dania = 484 := by
  sorry

end minimum_distance_sasha_dania_l430_430976


namespace chef_meals_prepared_l430_430874

theorem chef_meals_prepared (S D_added D_total L R : ℕ)
  (hS : S = 12)
  (hD_added : D_added = 5)
  (hD_total : D_total = 10)
  (hR : R + D_added = D_total)
  (hL : L = S + R) : L = 17 :=
by
  sorry

end chef_meals_prepared_l430_430874


namespace irregular_parallelogram_is_none_of_these_l430_430711

structure Parallelogram (P : Type) :=
(opposite_sides_parallel : ∀ (s₁ s₂ : P), s₁.parallel s₂)
(adjacent_sides_not_equal : ∀ (s₁ s₂ : P), s₁ ≠ s₂ → ¬(s₁.length = s₂.length))
(not_all_angles_ninety_degrees : ∀ (a : P), a.angle ≠ 90)
(sides_not_all_equidistant : ∀ (s₁ s₂ : P), ¬(∀ (p q : s₁), p.dist q = s₂.dist q))

def is_irregular_parallelogram (P : Type) [Parallelogram P] : Prop :=
∀ (R : Type), ¬(is_rectangle R ∨ is_square R ∨ is_rhombus R)

theorem irregular_parallelogram_is_none_of_these
    (P : Type) [Parallelogram P] : is_irregular_parallelogram P :=
by
  sorry

end irregular_parallelogram_is_none_of_these_l430_430711


namespace find_stadium_width_l430_430368

-- Conditions
def stadium_length : ℝ := 24
def stadium_height : ℝ := 16
def longest_pole : ℝ := 34

-- Width to be solved
def stadium_width : ℝ := 18

-- Theorem stating that given the conditions, the width must be 18
theorem find_stadium_width :
  stadium_length^2 + stadium_width^2 + stadium_height^2 = longest_pole^2 :=
by
  sorry

end find_stadium_width_l430_430368


namespace total_students_l430_430588

theorem total_students (n_groups : ℕ) (students_in_group : ℕ → ℕ)
    (h1 : n_groups = 6)
    (h2 : ∃ n : ℕ, (students_in_group n = 13) ∧ (finset.filter (λ g, students_in_group g = 13) (finset.range n_groups)).card = 4)
    (h3 : ∀ i j, i < n_groups → j < n_groups → abs (students_in_group i - students_in_group j) ≤ 1) :
    (∃ N, N = 76 ∨ N = 80) :=
begin
    sorry
end

end total_students_l430_430588


namespace total_students_l430_430591

theorem total_students (n_groups : ℕ) (students_in_group : ℕ → ℕ)
    (h1 : n_groups = 6)
    (h2 : ∃ n : ℕ, (students_in_group n = 13) ∧ (finset.filter (λ g, students_in_group g = 13) (finset.range n_groups)).card = 4)
    (h3 : ∀ i j, i < n_groups → j < n_groups → abs (students_in_group i - students_in_group j) ≤ 1) :
    (∃ N, N = 76 ∨ N = 80) :=
begin
    sorry
end

end total_students_l430_430591


namespace find_square_digit_l430_430007

def is_even (n : ℕ) : Prop := n % 2 = 0

def sum_digits_31_42_7s (s : ℕ) : ℕ :=
  3 + 1 + 4 + 2 + 7 + s

-- The main theorem to prove
theorem find_square_digit (d : ℕ) (h0 : is_even d) (h1 : (sum_digits_31_42_7s d) % 3 = 0) : d = 4 :=
by
  sorry

end find_square_digit_l430_430007


namespace erika_sum_prob_l430_430763

-- Define the problem conditions and required types.
def age := 16
def coin_outcome := {10, 25}
def die_outcome := {1, 2, 3, 4, 5, 6}
def fair_coin_prob := (1 : ℚ) / 2
def die_prob := (1 : ℚ) / 6

-- The main theorem to prove the stated probability.
theorem erika_sum_prob : (∑ (coin : ℕ) in coin_outcome, 
                          if coin = 10 then fair_coin_prob * die_prob else 0) = (1 : ℚ) / 12 := 
by sorry

end erika_sum_prob_l430_430763


namespace sqrt_mul_sqrt_eq_six_l430_430285

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l430_430285


namespace flight_time_l430_430125

def eagle_speed : ℕ := 15
def falcon_speed : ℕ := 46
def pelican_speed : ℕ := 33
def hummingbird_speed : ℕ := 30
def total_distance : ℕ := 248

theorem flight_time : (eagle_speed + falcon_speed + pelican_speed + hummingbird_speed) > 0 → 
                      total_distance / (eagle_speed + falcon_speed + pelican_speed + hummingbird_speed) = 2 :=
by
  -- Proof is skipped
  sorry

end flight_time_l430_430125


namespace sqrt_mult_simplify_l430_430241

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l430_430241


namespace cardinality_intersection_eq_three_l430_430843

def A : Set ℝ := {x : ℝ | x^2 ≤ 2}
def Z : Set ℤ := {x : ℤ | True} -- Representing the set of all integers

theorem cardinality_intersection_eq_three :
  (Set.card (A ∩ ↑Z) = 3) :=
by
  sorry

end cardinality_intersection_eq_three_l430_430843


namespace sum_of_divisors_of_231_l430_430051

theorem sum_of_divisors_of_231 :
  let n := 231 in 
  let divisors := [1, 3, 7, 11, 21, 33, 77, 231] in 
  ∑ d in divisors, d = 384 :=
by
  let n := 231
  have h : n = 3 * 7 * 11 := rfl
  sorry

end sum_of_divisors_of_231_l430_430051


namespace Joey_age_next_multiple_sum_l430_430490

-- Definitions for conditions
variables (Chloe Joey Zoe : ℕ) (k : ℕ)
def Joey_older_by_five := Joey = Chloe + 5
def Zoe_age := Zoe = 3
def Chloe_age_multiple := Chloe = 3 * k
def first_of_six := ∀ n < 6, ∃ m, (Chloe + n) = m * (Zoe + n)

-- Definition to prove the sum of the digits of Joey's age the next time his age is a multiple of Zoe's age
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem Joey_age_next_multiple_sum (h1 : Joey_older_by_five Joey Chloe) (h2 : Zoe_age Zoe) 
  (h3 : Chloe_age_multiple Chloe k) (h4 : first_of_six Chloe k Zoe) :
  sum_of_digits (Nat.find (λ n, ∃ m, n + Joey = m * (Zoe + n))) = 9 := sorry

end Joey_age_next_multiple_sum_l430_430490


namespace expansion_coefficient_x_l430_430361

/-- The coefficient of x in the expansion of x * (sqrt(x) - 1/x)^9 is -84. -/
theorem expansion_coefficient_x (x : ℝ) : 
  let expr := x * (Real.sqrt x - 1 / x) ^ 9
  coeff_of_x expr = -84 := 
sorry

end expansion_coefficient_x_l430_430361


namespace necessary_and_sufficient_condition_l430_430839

def line1 (a : ℝ) (x y : ℝ) := 2 * x - a * y + 1 = 0
def line2 (a : ℝ) (x y : ℝ) := (a - 1) * x - y + a = 0
def parallel (a : ℝ) : Prop := ∀ x y : ℝ, line1 a x y = line2 a x y

theorem necessary_and_sufficient_condition (a : ℝ) : 
  (a = 2 ↔ parallel a) :=
sorry

end necessary_and_sufficient_condition_l430_430839


namespace tunnel_length_is_4_miles_l430_430118

noncomputable def tunnel_length
  (train_length : ℝ)
  (time_to_exit : ℝ)
  (train_speed : ℝ) : ℝ :=
let front_distance := (train_speed / 60) * time_to_exit in
front_distance - train_length

theorem tunnel_length_is_4_miles :
  tunnel_length 2 4 90 = 4 :=
by
  sorry

end tunnel_length_is_4_miles_l430_430118


namespace sqrt_mul_l430_430176

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l430_430176


namespace sqrt_mul_l430_430174

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l430_430174


namespace angelas_insects_l430_430732

variable (DeanInsects : ℕ) (JacobInsects : ℕ) (AngelaInsects : ℕ)

theorem angelas_insects
  (h1 : DeanInsects = 30)
  (h2 : JacobInsects = 5 * DeanInsects)
  (h3 : AngelaInsects = JacobInsects / 2):
  AngelaInsects = 75 := 
by
  sorry

end angelas_insects_l430_430732


namespace degree_f_plus_g_at_1_l430_430749

-- Defining the polynomials f(z) and g(z)
def f (z : ℂ) : ℂ := a₃ * z^3 + a₂ * z^2 + a₁ * z + a₀
def g (z : ℂ) : ℂ := b₂ * z^2 + b₁ * z + b₀

-- Assuming a₃ = 0 when z = 1
axiom h_a₃ : a₃ = 0

-- Lean statement to prove the degree of f(z) + g(z) when z = 1 is 2
theorem degree_f_plus_g_at_1 (a₃ a₂ a₁ a₀ b₂ b₁ b₀ : ℂ) (hz : z = 1) (h_a₃ : a₃ = 0) : 
  degree (f(1) + g(1)) = 2 :=
sorry

end degree_f_plus_g_at_1_l430_430749


namespace max_combined_weight_l430_430615

theorem max_combined_weight (E A : ℕ) (h1 : A = 2 * E) (h2 : A + E = 90) (w_A : ℕ := 5) (w_E : ℕ := 2 * w_A) :
  E * w_E + A * w_A = 600 :=
by
  sorry

end max_combined_weight_l430_430615


namespace margo_total_distance_walked_l430_430517

/-- Given conditions -/
def walk_to_friend_time : ℚ := 15 / 60  -- time in hours
def return_time : ℚ := 10 / 60  -- time in hours
def average_walking_rate : ℚ := 3.6  -- miles per hour

/-- Total time Margo spent walking -/
def total_time : ℚ := walk_to_friend_time + return_time

/-- The proof statement -/
theorem margo_total_distance_walked : 
  (average_walking_rate * total_time) = 1.5 := by
simm sorry

end margo_total_distance_walked_l430_430517


namespace teacher_scheduling_l430_430025

def teacher_scheduling_count {T : ℕ} : Prop :=
  -- We define the conditions for the scheduling problem:
  -- 4 teachers (T=4)
  let t := 4 in
  -- 6 days
  let d := 6 in
  -- Each teacher is on duty for 1 or 2 consecutive days
  let consecutive_days := true in
  -- Prove that the number of ways to schedule the teachers is 144
  (t = 4 ∧ d = 6 ∧ consecutive_days) → 
  -- The number of ways to schedule is 144
  (144)

theorem teacher_scheduling : ∀ {T}, teacher_scheduling_count ↔ T = 144 :=
by
  intro T
  -- Proof skipped for now
  sorry

end teacher_scheduling_l430_430025


namespace tangent_line_at_origin_l430_430014

-- Given a function f(x) = x^3 + ax with an extremum at x = 1,
-- prove that the equation of the tangent line to the curve y = f(x) at the origin is 3x + y = 0.

theorem tangent_line_at_origin (a : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x = x^3 + a * x) 
  (extremum_at_one : ∀ x : ℝ, f'(1) = 0) : 
  ∃ (k : ℝ), (k = f'(0)) ∧ (∀ x y : ℝ, y = f x → 3 * x + y = 0) :=
sorry

end tangent_line_at_origin_l430_430014


namespace total_students_l430_430587

theorem total_students (n_groups : ℕ) (students_in_group : ℕ → ℕ)
    (h1 : n_groups = 6)
    (h2 : ∃ n : ℕ, (students_in_group n = 13) ∧ (finset.filter (λ g, students_in_group g = 13) (finset.range n_groups)).card = 4)
    (h3 : ∀ i j, i < n_groups → j < n_groups → abs (students_in_group i - students_in_group j) ≤ 1) :
    (∃ N, N = 76 ∨ N = 80) :=
begin
    sorry
end

end total_students_l430_430587


namespace speed_of_first_train_l430_430657

/-
Problem:
Two trains, with lengths 150 meters and 165 meters respectively, are running in opposite directions. One train is moving at 65 kmph, and they take 7.82006405004841 seconds to completely clear each other from the moment they meet. Prove that the speed of the first train is 79.99 kmph.
-/

theorem speed_of_first_train :
  ∀ (length1 length2 : ℝ) (speed2 : ℝ) (time : ℝ) (speed1 : ℝ),
  length1 = 150 → length2 = 165 → speed2 = 65 → time = 7.82006405004841 →
  ( 3.6 * (length1 + length2) / time = speed1 + speed2 ) →
  speed1 = 79.99 :=
by
  intros length1 length2 speed2 time speed1 h_length1 h_length2 h_speed2 h_time h_formula
  rw [h_length1, h_length2, h_speed2, h_time] at h_formula
  sorry

end speed_of_first_train_l430_430657


namespace orange_juice_amount_l430_430712

theorem orange_juice_amount (total_drink : ℝ) (percent_grapefruit : ℝ) (percent_lemon : ℝ) (percent_orange : ℝ) (amount_grapefruit : ℝ) (amount_lemon : ℝ) (amount_orange : ℝ) :
  total_drink = 50 →
  percent_grapefruit = 0.25 →
  percent_lemon = 0.35 →
  amount_grapefruit = percent_grapefruit * total_drink →
  amount_lemon = percent_lemon * total_drink →
  amount_orange = total_drink - (amount_grapefruit + amount_lemon) →
  amount_orange = 20 :=
by
  intros h_total h_percent_grapefruit h_percent_lemon h_amount_grapefruit h_amount_lemon h_amount_orange
  rw [h_total, h_percent_grapefruit, h_percent_lemon, h_amount_grapefruit, h_amount_lemon, h_amount_orange]
  sorry

end orange_juice_amount_l430_430712


namespace molecular_weight_CO_l430_430135

theorem molecular_weight_CO : 
  let molecular_weight_C := 12.01
  let molecular_weight_O := 16.00
  molecular_weight_C + molecular_weight_O = 28.01 :=
by
  sorry

end molecular_weight_CO_l430_430135


namespace phi_value_l430_430452

theorem phi_value (φ : ℝ) (h₁ : φ > 0) (h₂ : φ < 90) (h₃ : √2 * Real.cos (20 * Real.pi / 180) = Real.sin (φ * Real.pi / 180) - Real.cos (φ * Real.pi / 180)) :
    φ = 65 :=
by
  sorry

end phi_value_l430_430452


namespace barbara_spent_total_l430_430132

variables (cost_steaks cost_chicken total_spent per_pound_steak per_pound_chicken : ℝ)
variables (weight_steaks weight_chicken : ℝ)

-- Defining the given conditions
def conditions :=
  per_pound_steak = 15 ∧
  weight_steaks = 4.5 ∧
  cost_steaks = per_pound_steak * weight_steaks ∧

  per_pound_chicken = 8 ∧
  weight_chicken = 1.5 ∧
  cost_chicken = per_pound_chicken * weight_chicken

-- Proving the total spent by Barbara is $79.50
theorem barbara_spent_total 
  (h : conditions per_pound_steak weight_steaks cost_steaks per_pound_chicken weight_chicken cost_chicken) : 
  total_spent = 79.5 :=
sorry

end barbara_spent_total_l430_430132


namespace smallest_angle_cosine_l430_430777

theorem smallest_angle_cosine :
  (∃ θ : ℝ, 0 < θ ∧ θ < 360 ∧ cos θ = cos 45 + cos 48 - cos 75 - cos 30) →
  θ = 3 :=
by
  sorry

# Helper Definitions for Conditions
def cos_45 : ℝ := cos (Real.pi / 4)
def cos_48 : ℝ := cos (48 * Real.pi / 180)
def cos_75 : ℝ := cos (75 * Real.pi / 180)
def cos_30 : ℝ := cos (30 * Real.pi / 180)

end smallest_angle_cosine_l430_430777


namespace sqrt_multiplication_l430_430150

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l430_430150


namespace lcm_not_consecutive_in_arithmetic_progression_l430_430996

open Nat

theorem lcm_not_consecutive_in_arithmetic_progression (n : ℕ) (hn : n > 100) :
  ¬ (∃ (a : ℕ) (d : ℕ) (L : Finset ℕ),
    (∀ x ∈ L, x = a + d * Finset.card (Finset.filter (λ y, a + d * y ∈ L) L)) ∧ 
    Finset.card L = n * (n - 1) / 2 ∧ 
    (∀ i j ∈ L, i ≠ j → Nat.lcm i j ∈ L)) :=
by
  sorry

end lcm_not_consecutive_in_arithmetic_progression_l430_430996


namespace jon_april_spending_l430_430493

theorem jon_april_spending : 
  let c := 2 in       -- coffees per day
  let p := 2 in       -- price per coffee
  let d := 30 in      -- days in April
  let s := c * p in   -- daily spending
  let m := s * d in   -- monthly spending
  m = 120 := by 
  sorry

end jon_april_spending_l430_430493


namespace find_n_l430_430690

theorem find_n (n : ℕ) (h : ∃ a : ℤ, 4 * a^2 - (4 * real.sqrt 3 + 4) * a + real.sqrt 3 * n - 24 = 0) : n = 12 :=
sorry

end find_n_l430_430690


namespace system_of_equations_solutions_l430_430359

theorem system_of_equations_solutions (x1 x2 x3 : ℝ) :
  (2 * x1^2 / (1 + x1^2) = x2) ∧ (2 * x2^2 / (1 + x2^2) = x3) ∧ (2 * x3^2 / (1 + x3^2) = x1)
  → (x1 = 0 ∧ x2 = 0 ∧ x3 = 0) ∨ (x1 = 1 ∧ x2 = 1 ∧ x3 = 1) :=
by
  sorry

end system_of_equations_solutions_l430_430359


namespace ratio_of_radii_l430_430988

-- Define the conditions
variables {R r : ℝ} -- R and r are the radii of the larger and smaller circles, respectively
variable  (O O1 A B C M N : Point) -- O and O1 are centers of the circles, A is the touching point, others are relevant points for geometric construction

-- Hypotheses based on the problem statement
variables (h1 : 0 < r) (h2 : R > r)
variables (condition1 : is_tangent (circle O R) (circle O1 r) A)
variables (condition2 : is_tangent_to_circle (O, A, B, M) R)
variables (condition3 : is_tangent_to_circle (O, A, C, N) R)
variables (angle_ABC : ∠ B O C = 60)

theorem ratio_of_radii : R / r = 3 :=
sorry

end ratio_of_radii_l430_430988


namespace total_students_possible_l430_430559

theorem total_students_possible (A B : ℕ) :
  (4 * 13) + 2 * A = 76 ∨ (4 * 13) + 2 * B = 80 :=
by
  -- Let N be the total number of students
  let N := (4 * 13)
  -- Given that the number of students in the remaining 2 groups differs by no more than 1
  have h : A = 12 ∨ B = 14 := sorry
  -- Prove the possible values
  exact or.inl (N + 2 * 12 = 76) <|> or.inr (N + 2 * 14 = 80)

end total_students_possible_l430_430559


namespace decreasing_function_on_interval_l430_430674

theorem decreasing_function_on_interval (k : ℝ) :
  (∀ x ∈ set.Icc 0 2, deriv (λ x, x^3 + k * x^2) x ≤ 0) ↔ k ≤ -3 :=
sorry

end decreasing_function_on_interval_l430_430674


namespace pythagoras_schools_l430_430350

theorem pythagoras_schools (a b c : ℕ) (n : ℕ)
  (H1 : 3 * n = b) 
  (H2 : b + 1 = 2 * a)
  (H3 : b = 2 * a - 1)
  (H4 : a ≥ 24)
  (H5 : a < 31)
  (H6 : a % 3 = 2)
  (H7 : b % 3 = 0)
  (H8 : ∀ x, is_prime x → x = a)
  : n = 19 :=
sorry

end pythagoras_schools_l430_430350


namespace ratio_of_distances_l430_430991

/-- 
  Given two points A and B moving along intersecting lines with constant,
  but different velocities v_A and v_B respectively, prove that there exists a 
  point P such that at any moment in time, the ratio of distances AP to BP equals 
  the ratio of their velocities.
-/
theorem ratio_of_distances (A B : ℝ → ℝ × ℝ) (v_A v_B : ℝ)
  (intersecting_lines : ∃ t, A t = B t)
  (diff_velocities : v_A ≠ v_B) :
  ∃ P : ℝ × ℝ, ∀ t, (dist P (A t) / dist P (B t)) = v_A / v_B := 
sorry

end ratio_of_distances_l430_430991


namespace find_n_from_t_l430_430454

theorem find_n_from_t (n t : ℕ) (h1 : t = n * (n - 1) * (n + 1) + n) (h2 : t = 64) : n = 4 := by
  sorry

end find_n_from_t_l430_430454


namespace total_students_possible_l430_430558

theorem total_students_possible (A B : ℕ) :
  (4 * 13) + 2 * A = 76 ∨ (4 * 13) + 2 * B = 80 :=
by
  -- Let N be the total number of students
  let N := (4 * 13)
  -- Given that the number of students in the remaining 2 groups differs by no more than 1
  have h : A = 12 ∨ B = 14 := sorry
  -- Prove the possible values
  exact or.inl (N + 2 * 12 = 76) <|> or.inr (N + 2 * 14 = 80)

end total_students_possible_l430_430558


namespace translate_point_A_l430_430883

theorem translate_point_A :
  let A : ℝ × ℝ := (-1, 2)
  let x_translation : ℝ := 4
  let y_translation : ℝ := -2
  let A1 : ℝ × ℝ := (A.1 + x_translation, A.2 + y_translation)
  A1 = (3, 0) :=
by
  let A : ℝ × ℝ := (-1, 2)
  let x_translation : ℝ := 4
  let y_translation : ℝ := -2
  let A1 : ℝ × ℝ := (A.1 + x_translation, A.2 + y_translation)
  show A1 = (3, 0)
  sorry

end translate_point_A_l430_430883


namespace Angela_insect_count_l430_430735

variables (Angela Jacob Dean : ℕ)
-- Conditions
def condition1 : Prop := Angela = Jacob / 2
def condition2 : Prop := Jacob = 5 * Dean
def condition3 : Prop := Dean = 30

-- Theorem statement proving Angela's insect count
theorem Angela_insect_count (h1 : condition1 Angela Jacob) (h2 : condition2 Jacob Dean) (h3 : condition3 Dean) : Angela = 75 :=
by
  sorry

end Angela_insect_count_l430_430735


namespace minimum_value_l430_430019

noncomputable def f (x : ℝ) : ℝ := x^3 + x^2 - x + 1

theorem minimum_value (a b : ℝ) (h₀ : a = -2) (h₁ : b = 1) :
  ∃ c ∈ set.Icc a b, ∀ x ∈ set.Icc a b, f c ≤ f x ∧ f c = -1 :=
by
  sorry

end minimum_value_l430_430019


namespace average_selections_correct_l430_430720

noncomputable def cars := 18
noncomputable def selections_per_client := 3
noncomputable def clients := 18
noncomputable def total_selections := clients * selections_per_client
noncomputable def average_selections_per_car := total_selections / cars

theorem average_selections_correct :
  average_selections_per_car = 3 :=
by
  sorry

end average_selections_correct_l430_430720


namespace proof_opposite_abs_reciprocal_neg_half_l430_430961

theorem proof_opposite_abs_reciprocal_neg_half : 
  let x := -0.5 in 
  let reciprocal_x := 1 / x in 
  let abs_reciprocal_x := abs reciprocal_x in 
  let opposite_abs_reciprocal_x := -abs_reciprocal_x in 
  opposite_abs_reciprocal_x = -2 := 
by 
  sorry

end proof_opposite_abs_reciprocal_neg_half_l430_430961


namespace no_mark_on_2x_l430_430631

noncomputable def f (a b : ℕ) : ℕ := a ^ 2 + b ^ 2

def transform1 (a b : ℕ) : set (ℕ × ℕ) := { (b, a), (a - b, a + b) }

def transform2 (a b c d : ℕ) : set (ℕ × ℕ) := { (a * d + b * c, 4 * a * c - 4 * b * d) }

def initial_points : set (ℕ × ℕ) := { (1, 1), (2, 3), (4, 5), (999, 111) }

def can_mark (p : ℕ × ℕ) : Prop :=
  ∃ S, S ⊆ initial_points ∪ { q | ∃ a b, q ∈ transform1 a b ∧ (a, b) ∈ S }
      ∪ { q | ∃ a b c d, q ∈ transform2 a b c d ∧ (a, b) ∈ S ∧ (c, d) ∈ S } ∧ p ∈ S

theorem no_mark_on_2x (x : ℕ) : ¬ can_mark (x, 2 * x) :=
sorry

end no_mark_on_2x_l430_430631


namespace sum_segment_arc_length_l430_430521

theorem sum_segment_arc_length :
  let circle (x y : ℝ) : Prop := x^2 + y^2 = 16
  let line (x y : ℝ) : Prop := y = 4 - (2 - real.sqrt 3) * x
  ∀ A B : ℝ × ℝ,
    circle A.1 A.2 ∧ line A.1 A.2 ∧
    circle B.1 B.2 ∧ line B.1 B.2 ∧
    A ≠ B →
    dist A B + (2 * real.pi * 4 * (real.acos (real.sqrt 3 / 2) / (2 * real.pi))) = 
    4 * real.sqrt (2 - real.sqrt 3) + 2 * real.pi / 3
:= by
  sorry

end sum_segment_arc_length_l430_430521


namespace number_of_ways_to_choose_bases_l430_430790

-- Definitions of the conditions
def num_students : Nat := 4
def num_bases : Nat := 3

-- The main statement that we need to prove
theorem number_of_ways_to_choose_bases : (num_bases ^ num_students) = 81 := by
  sorry

end number_of_ways_to_choose_bases_l430_430790


namespace solve_ordered_pair_l430_430774

theorem solve_ordered_pair (x y : ℝ) (h1 : x + y = (5 - x) + (5 - y)) (h2 : x - y = (x - 1) + (y - 1)) : (x, y) = (4, 1) :=
by
  sorry

end solve_ordered_pair_l430_430774


namespace problem_statement_l430_430713

noncomputable def f : ℝ → ℝ
| x := if -3 ≤ x ∧ x < -1 then -(x + 2)^2 else if -1 ≤ x ∧ x < 3 then x else f (x - 6)

theorem problem_statement : f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + sum (list.range' 7 2010).map f = 337 := by
  sorry

end problem_statement_l430_430713


namespace find_ab_l430_430405

theorem find_ab (a b : ℝ) 
  (h1 : a + b = 5) 
  (h2 : a^3 + b^3 = 35) : a * b = 6 := 
by
  sorry

end find_ab_l430_430405


namespace sum_of_numbers_l430_430959

theorem sum_of_numbers (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b = 8)
  (h4 : (a + b + c) / 3 = a + 12) (h5 : (a + b + c) / 3 = c - 20) :
  a + b + c = 48 :=
sorry

end sum_of_numbers_l430_430959


namespace mod_remainder_one_l430_430912

variable {n : ℕ} (hn : 0 < n)
variable {a b : ℤ} 
variable (ha : a % n != 0)
variable (hb : b % n != 0)
variable (h_inv : (a % n) = ((b % n)⁻¹ : ℤ))

theorem mod_remainder_one (hn : 0 < n) (ha : a % n != 0) (hb : b % n != 0) (h_inv : (a % n) = (b % n)⁻¹) :
  (a * b) % n = 1 :=
by sorry

end mod_remainder_one_l430_430912


namespace tan_x_eq_1_x_eq_5π_over_12_l430_430882

-- Define the vectors m and n where x is in (0, π/2)
def m : ℝ × ℝ := (sqrt 2 / 2, - sqrt 2 / 2)
def n (x : ℝ) : ℝ × ℝ := (sin x, cos x)

-- First problem: If m is perpendicular to n, then tan(x) = 1 
theorem tan_x_eq_1 (x : ℝ) (h : x > 0 ∧ x < π / 2) (orthogonal : m.1 * (n x).1 + m.2 * (n x).2 = 0) : tan x = 1 := 
  sorry

-- Second problem: If the angle between m and n is π/3, then x = 5π/12 
theorem x_eq_5π_over_12 (x : ℝ) (h : x > 0 ∧ x < π / 2) (angle_condition : m.1 * (n x).1 + m.2 * (n x).2 = 1 / 2) : x = 5 * π / 12 := 
  sorry

end tan_x_eq_1_x_eq_5π_over_12_l430_430882


namespace polynomial_sequence_property_l430_430497

open Int

noncomputable def sequence (f : ℤ → ℤ) : ℕ → ℤ 
| 0     := 0
| (n+1) := f (sequence n)

theorem polynomial_sequence_property (f : ℤ → ℤ) (hf : ∀ n, polynomial.eval (sequence f n) (polynomial.C f)) :
  (∃ m : ℕ, m > 0 ∧ sequence f m = 0) → (sequence f 1 = 0 ∨ sequence f 2 = 0) :=
begin
  sorry
end

end polynomial_sequence_property_l430_430497


namespace integral_identity_l430_430529

noncomputable def T_n (f : ℝ → ℝ) (a_0 : ℝ) (a_k b_k : ℕ → ℝ) (n : ℕ) : ℝ → ℝ :=
  λ x, a_0 / 2 + ∑ k in Finset.range (n + 1), (a_k k * Real.cos (k * x) + b_k k * Real.sin (k * x))

noncomputable def a_k (f : ℝ → ℝ) (k : ℕ) : ℝ :=
  1 / Real.pi * ∫ x in -Real.pi..Real.pi, f x * Real.cos (k * x)

noncomputable def b_k (f : ℝ → ℝ) (k : ℕ) : ℝ :=
  1 / Real.pi * ∫ x in -Real.pi..Real.pi, f x * Real.sin (k * x)

theorem integral_identity (f : ℝ → ℝ) (n : ℕ) :
  let a_0 := a_k f 0 in
  ∫ x in -Real.pi..Real.pi, (f x - T_n f a_0 (a_k f) (b_k f) n x) ^ 2 = 
  ∫ x in -Real.pi..Real.pi, (f x) ^ 2 - (Real.pi * a_0 ^ 2) / 2 - Real.pi * ∑ k in Finset.range (n + 1), (a_k f k) ^ 2 + ∑ k in Finset.range n, (b_k f (k + 1)) ^ 2 := 
sorry

end integral_identity_l430_430529


namespace parallel_lines_m_l430_430811

theorem parallel_lines_m (m : ℝ) : (∀ x y : ℝ, (x + m * y + 7 = 0) → ( (m - 2) * x + 3 * y + 2 * m = 0 )) ↔ (m = 3 ∨ m = -1) :=
by 
  sorry -- The proof of the theorem

end parallel_lines_m_l430_430811


namespace find_percentage_l430_430695

theorem find_percentage (P : ℝ) (h: (20 / 100) * 580 = (P / 100) * 120 + 80) : P = 30 := 
by
  sorry

end find_percentage_l430_430695


namespace standard_dice_pips_total_l430_430929

theorem standard_dice_pips_total :
  ∀ (a b : ℕ), (a + b = 5) → (a < 7) → (b < 7) → (7 - a + 7 - b = 9) :=
by
  intros a b h_ab h_a_lt_7 h_b_lt_7
  have h1 : 7 - a + 7 - b = 14 - (a + b), by ring
  rw h_ab at h1
  exact h1

end standard_dice_pips_total_l430_430929


namespace min_value_of_quadratic_l430_430665

theorem min_value_of_quadratic : ∃ x : ℝ, 7 * x^2 - 28 * x + 1702 = 1674 ∧ ∀ y : ℝ, 7 * y^2 - 28 * y + 1702 ≥ 1674 :=
by
  sorry

end min_value_of_quadratic_l430_430665


namespace closest_fraction_to_team_japan_awards_l430_430738

def fractional_part_team_japan : ℚ := 13 / 150
def option1 : ℚ := 1 / 10
def option2 : ℚ := 1 / 11
def option3 : ℚ := 1 / 12
def option4 : ℚ := 1 / 13
def option5 : ℚ := 1 / 14

theorem closest_fraction_to_team_japan_awards:
  (∀ x ∈ {option1, option2, option3, option4, option5}, abs (fractional_part_team_japan - x) >= abs (fractional_part_team_japan - option2)) :=
sorry

end closest_fraction_to_team_japan_awards_l430_430738


namespace train_cross_time_is_9_seconds_l430_430723

-- Define the problem conditions
def train_speed_kmph : ℝ := 50
def train_length_m : ℝ := 125
def km_to_m : ℝ := 1000
def hr_to_s : ℝ := 3600

-- Conversion from km/hr to m/s
def train_speed_mps : ℝ := (train_speed_kmph * km_to_m) / hr_to_s

-- Desired result for the time
def time_to_cross : ℝ := train_length_m / train_speed_mps

-- Proof statement
theorem train_cross_time_is_9_seconds :
  |time_to_cross - 9| < 0.1 :=
by
  sorry

end train_cross_time_is_9_seconds_l430_430723


namespace find_m_plus_n_l430_430654

noncomputable def right_triangle (A B C : Point) : Prop :=
  ∠A = 90

def incenter (A B C I : Point) : Prop :=
  is_incenter(I, A, B, C)

def circle_centered_at_incenter_passing_through_A (A B C I E F : Point) : Prop :=
  is_circle(I, A) ∧ is_on_circle(E, I) ∧ is_on_circle(F, I) ∧ is_on_line_bc(E, F) ∧ E < F

def ratio_BE_EF (E F : Point) (rBE rEF : Real) : Prop :=
  rBE / rEF = 2 / 3

theorem find_m_plus_n (A B C I E F : Point) (rBE rEF : Real) (m n : Nat) :
  right_triangle(A, B, C) ∧
  incenter(A, B, C, I) ∧
  circle_centered_at_incenter_passing_through_A(A, B, C, I, E, F) ∧
  ratio_BE_EF(E, F, rBE, rEF) ∧
  gcd(m, n) = 1 ∧ 
  m + n = 7 := sorry

end find_m_plus_n_l430_430654


namespace smallest_a_l430_430506

theorem smallest_a (a b : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : ∀ x : ℤ, Real.sin (a * x + b) = Real.sin (17 * x)) :
  a = 17 :=
by
  sorry

end smallest_a_l430_430506


namespace face_opposite_one_is_three_l430_430086

def faces : List ℕ := [1, 2, 3, 4, 5, 6]

theorem face_opposite_one_is_three (x : ℕ) (h1 : x ∈ faces) (h2 : x ≠ 1) : x = 3 :=
by
  sorry

end face_opposite_one_is_three_l430_430086


namespace velocity_at_3s_l430_430434

-- Define the motion equation
def s (t : ℝ) : ℝ := (t^3 / 9) + t

-- Define the derivative of the motion equation
noncomputable def s' (t : ℝ) : ℝ := (1/3) * t^2 + 1

-- Define the specific time instance we are interested in
def t_val : ℝ := 3

-- Define the expected instantaneous velocity at t = 3 s
def expected_velocity : ℝ := 4

-- State the theorem to prove
theorem velocity_at_3s : s' t_val = expected_velocity :=
by
  -- The proof will go here
  sorry

end velocity_at_3s_l430_430434


namespace find_smallest_x_l430_430778

-- Definition of the conditions
def cong1 (x : ℤ) : Prop := x % 5 = 4
def cong2 (x : ℤ) : Prop := x % 7 = 6
def cong3 (x : ℤ) : Prop := x % 8 = 7

-- Statement of the problem
theorem find_smallest_x :
  ∃ (x : ℕ), x > 0 ∧ cong1 x ∧ cong2 x ∧ cong3 x ∧ x = 279 :=
by
  sorry

end find_smallest_x_l430_430778


namespace regression_lines_intersect_at_averages_l430_430034

variables (α β : Type) [LinearOrder α] [Field β]

def average (values : List α) : α :=
  values.sum / values.length

noncomputable def regression_line (data : List (α × β)) : β × β :=
  sorry -- This represents the parameters (m, b) for y = mx + b (omitted for simplicity)

noncomputable def average_x (data : List (α × β)) : α :=
  average (data.map Prod.fst)

noncomputable def average_y (data : List (α × β)) : β :=
  average (data.map Prod.snd)

noncomputable def intersects_at (line1 line2 : β × β) (point : α × β) : Prop :=
  let (m1, b1) := line1
  let (m2, b2) := line2
  let (x, y) := point
  (y = m1 * x + b1) ∧ (y = m2 * x + b2)

theorem regression_lines_intersect_at_averages
  (data1 data2 : List (α × β))
  (t1 t2 : β × β)
  (s : α) (t : β)
  (h1 : average_x data1 = s) (h2 : average_y data1 = t)
  (h3 : average_x data2 = s) (h4 : average_y data2 = t)
  (h5 : t1 = regression_line data1) (h6 : t2 = regression_line data2) :
  intersects_at t1 t2 (s, t) :=
sorry

end regression_lines_intersect_at_averages_l430_430034


namespace exists_universal_town_l430_430648

-- Define the town and road structure as a finite directed graph
structure TownGraph :=
  (T : Type) -- A type representing towns
  [fintype_T : fintype T] -- Finite number of towns
  (roads : T → T → Prop) -- One-direction roads between towns
  [decidable_rel_roads : decidable_rel roads] -- Decidability of the roads relation

-- Assume the graph is strongly connected
def strongly_connected (G : TownGraph) : Prop :=
  ∀ (a b : G.T), ∃ (p : list G.T), p.head = some a ∧ p.last = some b ∧ ∀ (i : ℕ), i < p.length - 1 → G.roads (list.nth_le p i sorry) (list.nth_le p (i + 1) sorry)

-- Lean 4 statement for the math proof problem
theorem exists_universal_town (G : TownGraph) (H_con: strongly_connected G) : ∃ (u : G.T), ∀ (v : G.T), ∃ (p : list G.T), p.head = some u ∧ p.last = some v ∧ ∀ (i : ℕ), i < p.length - 1 → G.roads (list.nth_le p i sorry) (list.nth_le p (i + 1) sorry) :=
sorry

end exists_universal_town_l430_430648


namespace smallest_positive_perfect_square_divisible_by_3_and_5_is_225_l430_430048

def smallest_perf_square_divisible_by_3_and_5 : ℕ :=
  let n := 15 in n * n

theorem smallest_positive_perfect_square_divisible_by_3_and_5_is_225 :
  smallest_perf_square_divisible_by_3_and_5 = 225 :=
by
  sorry

end smallest_positive_perfect_square_divisible_by_3_and_5_is_225_l430_430048


namespace difference_is_four_l430_430928

noncomputable def mean (scores : List (ℕ × ℚ)) : ℚ :=
  let total_students := scores.map (λ p => p.snd).sum
  let total_points := scores.map (λ p => p.fst * p.snd).sum
  total_points / total_students

-- conditions
def scores : List (ℕ × ℚ) :=
  [(70, 0.15 * 100), (80, 0.35 * 100), (85, 0.10 * 100), (90, 0.25 * 100), (95, 0.15 * 100)]

def median : ℚ :=
  let sorted_scores := scores.sortBy (λ p => p.fst)
  (sorted_scores.part (λ p => p.snd) 49 + sorted_scores.part (λ p => p.snd) 50) / 2

def difference_mean_median : ℚ :=
  let mean_score := mean scores
  let median_score := median
  mean_score - median_score

theorem difference_is_four : difference_mean_median == 4 := by
  -- Proof is omitted
  sorry

end difference_is_four_l430_430928


namespace sqrt_mult_simplify_l430_430247

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l430_430247


namespace sqrt_mul_sqrt_eq_six_l430_430295

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l430_430295


namespace largest_negative_integer_solution_l430_430943

noncomputable def inequality (x : ℝ) : Prop :=
  log (abs (x - 1)) ((x - 2) / x) > 1

theorem largest_negative_integer_solution :
  ∀ (x : ℝ), inequality x → (-sqrt 2 < x) ∧ (x < 0) → ∃ k : ℤ, k = -1 :=
by
  sorry

end largest_negative_integer_solution_l430_430943


namespace sqrt_mult_simplify_l430_430237

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l430_430237


namespace sqrt_mult_simplify_l430_430242

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l430_430242


namespace sqrt_mult_l430_430193

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l430_430193


namespace total_students_total_students_alt_l430_430554

def number_of_students (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 
    a + b = 6 ∧ 
    a = 4 ∧ 
    (∀ g, g = 13) ∧ 
    (b = 2 ∧ ((g = 12 ∨ g = 14) ∧ ∀ h, g - h ≤ 1)) ∧ 
    (n = 52 + 2 * 12 ∨ n = 52 + 2 * 14)

theorem total_students : ∃ n : ℕ, number_of_students n :=
by
  use 76
  sorry

theorem total_students_alt : ∃ n : ℕ, number_of_students n :=
by
  use 80
  sorry

end total_students_total_students_alt_l430_430554


namespace total_students_l430_430592

theorem total_students (n_groups : ℕ) (students_in_group : ℕ → ℕ)
    (h1 : n_groups = 6)
    (h2 : ∃ n : ℕ, (students_in_group n = 13) ∧ (finset.filter (λ g, students_in_group g = 13) (finset.range n_groups)).card = 4)
    (h3 : ∀ i j, i < n_groups → j < n_groups → abs (students_in_group i - students_in_group j) ≤ 1) :
    (∃ N, N = 76 ∨ N = 80) :=
begin
    sorry
end

end total_students_l430_430592


namespace pandas_bamboo_consumption_l430_430384

def small_pandas : ℕ := 4
def big_pandas : ℕ := 5
def daily_bamboo_small : ℕ := 25
def daily_bamboo_big : ℕ := 40
def days_in_week : ℕ := 7

theorem pandas_bamboo_consumption : 
  (small_pandas * daily_bamboo_small + big_pandas * daily_bamboo_big) * days_in_week = 2100 := by
  sorry

end pandas_bamboo_consumption_l430_430384


namespace sqrt_mult_eq_six_l430_430319

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l430_430319


namespace total_students_total_students_alt_l430_430548

def number_of_students (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 
    a + b = 6 ∧ 
    a = 4 ∧ 
    (∀ g, g = 13) ∧ 
    (b = 2 ∧ ((g = 12 ∨ g = 14) ∧ ∀ h, g - h ≤ 1)) ∧ 
    (n = 52 + 2 * 12 ∨ n = 52 + 2 * 14)

theorem total_students : ∃ n : ℕ, number_of_students n :=
by
  use 76
  sorry

theorem total_students_alt : ∃ n : ℕ, number_of_students n :=
by
  use 80
  sorry

end total_students_total_students_alt_l430_430548


namespace sum_mul_contents_subsets_l430_430684

theorem sum_mul_contents_subsets {α : Type} [fintype α] [decidable_eq α] (s : finset ℤ) (h : s = {1, 2, 3, 4}) : 
  ∑ t in s.powerset, t.prod id = 120 := 
by 
  rw h
  sorry

end sum_mul_contents_subsets_l430_430684


namespace sqrt_mult_l430_430194

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l430_430194


namespace sqrt_mul_l430_430172

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l430_430172


namespace stratified_sampling_l430_430120

theorem stratified_sampling (total : ℕ) (elderly middle_aged young sample : ℕ)
  (h1 : total = elderly + middle_aged + young)
  (h2 : sample = 36)
  (h3 : elderly = 28)
  (h4 : middle_aged = 56)
  (h5 : young = 84) :
  let ratio := (sample : ℚ) / total,
      num_elderly := (ratio * elderly.to_rat).to_nat,
      num_middle_aged := (ratio * middle_aged.to_rat).to_nat,
      num_young := (ratio * young.to_rat).to_nat
  in num_elderly = 6 ∧ num_middle_aged = 12 ∧ num_young = 18 :=
by
  sorry

end stratified_sampling_l430_430120


namespace convex_polygon_point_set_l430_430938

theorem convex_polygon_point_set
  (n : ℕ)
  (P : set (ℝ × ℝ))
  (is_convex_polygon : convex P)
  (has_n_sides : P.card = n)
  (n_geq_3 : n ≥ 3) :
  ∃ S : set (ℝ × ℝ), S.card = n - 2 ∧
  (∀ (A B C : ℝ × ℝ), A ∈ P ∧ B ∈ P ∧ C ∈ P ∧ A ≠ B ∧ B ≠ C ∧ C ≠ A →
   (A, B, C form_triangle ∧ ∃! p ∈ S, p inside_or_on_boundary_of_triangle A B C)) := sorry

end convex_polygon_point_set_l430_430938


namespace swimming_time_back_against_current_l430_430718

theorem swimming_time_back_against_current (swimming_speed : ℕ) (current_speed : ℕ) (time_taken : ℕ)
    (h1 : swimming_speed = 4) (h2 : current_speed = 2) (h3 : time_taken = 8) :
    time_taken = 8 :=
begin
  -- Proof goes here
  sorry
end

end swimming_time_back_against_current_l430_430718


namespace sqrt_mul_eq_l430_430304

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l430_430304


namespace sqrt3_mul_sqrt12_eq_6_l430_430224

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l430_430224


namespace length_of_BC_l430_430795

theorem length_of_BC (x : ℝ) (h1 : (20 * x^2) / 3 - (400 * x) / 3 = 140) :
  ∃ (BC : ℝ), BC = 29 := 
by
  sorry

end length_of_BC_l430_430795


namespace number_of_five_digit_palindromes_l430_430092

-- Define what constitutes a five-digit palindrome
def is_five_digit_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  n ≥ 10000 ∧ n < 100000 ∧ digits = digits.reverse

-- Statement of the proof problem
theorem number_of_five_digit_palindromes : (finset.filter is_five_digit_palindrome (finset.range 100000)).card = 900 :=
sorry

end number_of_five_digit_palindromes_l430_430092


namespace collinear_A_P_Q_l430_430891

open EuclideanGeometry

theorem collinear_A_P_Q
  (A B C X Y P Q : Point)
  (hAB_AC : dist A B > dist A C)
  (hX_ON_BIS : OnLine X (angleBisector A B C))
  (hY_ON_BIS : OnLine Y (angleBisector A B C))
  (hAngle_ABX_ACY : ∠ABX = ∠ACY)
  (hBX_int_CY_at_P : ∃ line t, OnLine P (extendLine line B X) ∧ OnLine P (segment C Y))
  (hCircBPY_CPQ_int_P_Q : (Circumcircle (Triangle.mk B P Y)).intersects (Circumcircle (Triangle.mk C P X)) = (P, Q)) :
  Collinear {A, P, Q} :=
by
  sorry

end collinear_A_P_Q_l430_430891


namespace sqrt3_mul_sqrt12_eq_6_l430_430259

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l430_430259


namespace apple_sharing_l430_430729

theorem apple_sharing (a b c : ℕ) (h : a + b + c = 30) (h1 : a ≥ 3) (h2 : b ≥ 3) (h3 : c ≥ 3) : 
    ∃ n, n = 253 :=
by 
  -- Using the conditions, we need to count the ways to distribute the remaining 21 apples
  let a' := a - 3
  let b' := b - 3
  let c' := c - 3
  have h' : a' + b' + c' = 21 := by
    rw [← nat.add_sub_of_le h1, ← nat.add_sub_of_le h2, ← nat.add_sub_of_le h3] at h
    exact nat.sub_add_cancel h

  -- Applying the stars and bars theorem
  use nat.choose 23 2
  have choose_eq : nat.choose 23 2 = 253 := by
    calc
      nat.choose 23 2 = 23 * 22 / 2 : rfl
               ...      = 253       : by norm_num
  exact choose_eq

end apple_sharing_l430_430729


namespace min_colors_needed_l430_430927

def circle_points_coloring (points_on_circle : ℕ) (k_colors : ℕ) : Prop :=
  ∀ (points : set ℕ) (seg : set (ℕ × ℕ)), 
    points_on_circle = 1000 ∧ 
    points.card = 10 ∧ 
    (∀ p1 p2, p1 ∈ points → p2 ∈ points → p1 ≠ p2 → (p1, p2) ∈ seg ∧ (p2, p1) ∈ seg)→ 
    (∃ seg_diff, seg_diff.card ≥ 3 ∧ ∀ (s ∈ seg_diff), (fst s) ≠ (snd s))

theorem min_colors_needed : ∃ k_colors, circle_points_coloring 1000 k_colors ∧ k_colors = 143 := by
  sorry

end min_colors_needed_l430_430927


namespace paul_initial_crayons_l430_430932

-- Define the variables for the crayons given away, lost, and left
def crayons_given_away : ℕ := 563
def crayons_lost : ℕ := 558
def crayons_left : ℕ := 332

-- Define the total number of crayons Paul got for his birthday
def initial_crayons : ℕ := 1453

-- The proof statement
theorem paul_initial_crayons :
  initial_crayons = crayons_given_away + crayons_lost + crayons_left :=
sorry

end paul_initial_crayons_l430_430932


namespace sqrt_mult_l430_430201

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l430_430201


namespace seedling_count_correct_l430_430041

def seedlings_count (x : ℕ) : ℕ := x^2 + 39

theorem seedling_count_correct :
  ∃ x : ℕ, seedlings_count x = 1975 ∧
  (seedlings_count x = x^2 + 39) ∧
  (seedlings_count (x + 1) = (x + 1)^2 - 50) :=
begin
  sorry
end

end seedling_count_correct_l430_430041


namespace sqrt_mul_simp_l430_430211

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l430_430211


namespace area_of_circle_outside_square_l430_430113

theorem area_of_circle_outside_square (r : ℝ) (s : ℝ) (π : ℝ) 
  (h_r : r = 2 * (2 : ℝ).sqrt / 2) 
  (h_s : s = 2 * 2) :
  (π * r^2) - s = 2 * π - 4 :=
by
  -- Assertion of given conditions
  have h_r_eq : r = (2 : ℝ).sqrt := by sorry
  have h_s_eq : s = 4 := by sorry
  -- Assertion of the correct answer
  calc (π * r^2) - s 
      = π * ((2 : ℝ).sqrt)^2 - 4 : by rw [h_r_eq, h_s_eq]
  ... = 2 * π - 4 : by field_simp; linarith

end area_of_circle_outside_square_l430_430113


namespace sqrt_mult_simplify_l430_430234

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l430_430234


namespace student_groups_l430_430539

theorem student_groups (N : ℕ) :
  (∃ (n : ℕ), n = 13 ∧ ∃ (m : ℕ), m ∈ {12, 14} ∧ N = 4 * 13 + 2 * m) → (N = 76 ∨ N = 80) :=
by
  intro h
  obtain ⟨n, hn, m, hm, hN⟩ := h
  rw [hn, hN]
  cases hm with h12 h14
  case inl =>
    simp [h12]
  case inr =>
    simp [h14]
  sorry

end student_groups_l430_430539


namespace projections_equal_l430_430499

open Real

theorem projections_equal :
  let A (k : ℕ) (n : ℕ) : ℝ := cos (2 * π * k / n)
  in A 3 42 - A 6 42 = A 7 42 - A 9 42 :=
by {
  intros,
  -- detailed proofs will follow here
  sorry
}

end projections_equal_l430_430499


namespace robert_salary_loss_l430_430066

theorem robert_salary_loss (S : ℝ) (h₀ : S > 0) :
  let decreased_salary := S * 0.80,
      final_salary := decreased_salary * 1.20 in
  (S - final_salary) / S = 0.04 :=
by
  let decreased_salary := S * 0.80,
      final_salary := decreased_salary * 1.20
  sorry

end robert_salary_loss_l430_430066


namespace sqrt_multiplication_l430_430142

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l430_430142


namespace fraction_inequality_l430_430507

theorem fraction_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (1 / a) + (1 / b) ≥ (4 / (a + b)) :=
by 
-- Skipping the proof using 'sorry'
sorry

end fraction_inequality_l430_430507


namespace sqrt_mul_sqrt_eq_six_l430_430289

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l430_430289


namespace instantaneous_velocity_at_4_l430_430731

noncomputable def motion_equation (t : ℝ) : ℝ := 1 - t + t^2

theorem instantaneous_velocity_at_4 : (derivative motion_equation 4 = 7) :=
by
  -- The proof would follow here, we assume it as a correct statement.
  sorry

end instantaneous_velocity_at_4_l430_430731


namespace problem1_problem2_l430_430510

-- Definitions for the sets and conditions
def setA : Set ℝ := {x | -1 < x ∧ x < 2}
def setB (a : ℝ) : Set ℝ := if a > 0 then {x | x ≤ -2 ∨ x ≥ (1 / a)} else ∅

-- Problem 1: Prove the intersection for a == 1
theorem problem1 : (setB 1) ∩ setA = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

-- Problem 2: Prove the range of a
theorem problem2 (a : ℝ) (h : setB a ⊆ setAᶜ) : 0 < a ∧ a ≤ 1/2 :=
by
  sorry

end problem1_problem2_l430_430510


namespace sqrt_multiplication_l430_430139

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l430_430139


namespace difference_between_fastest_and_slowest_l430_430481

def fastest_time (times : List ℝ) : ℝ := times.minimum'
def slowest_time (times : List ℝ) : ℝ := times.maximum'

theorem difference_between_fastest_and_slowest :
  let times := [11.4, 15.553, 22.6, 124, 166]
  slowest_time times - fastest_time times = 154.6 := by
  sorry

end difference_between_fastest_and_slowest_l430_430481


namespace square_of_radius_l430_430705

-- Definitions based on conditions
def ER := 24
def RF := 31
def GS := 40
def SH := 29

-- The goal is to find square of radius r such that r^2 = 841
theorem square_of_radius (r : ℝ) :
  let R := ER
  let F := RF
  let G := GS
  let S := SH
  (∀ r : ℝ, (R + F) * (G + S) = r^2) → r^2 = 841 :=
sorry

end square_of_radius_l430_430705


namespace B_spends_85_percent_l430_430083

def combined_salary (S_A S_B : ℝ) : Prop := S_A + S_B = 4000
def A_savings_percentage : ℝ := 0.05
def A_salary : ℝ := 3000
def B_salary : ℝ := 4000 - A_salary
def equal_savings (S_A S_B : ℝ) : Prop := A_savings_percentage * S_A = (1 - S_B / 100) * B_salary

theorem B_spends_85_percent (S_A S_B : ℝ) (B_spending_percentage : ℝ) :
  combined_salary S_A S_B ∧ S_A = A_salary ∧ equal_savings S_A B_spending_percentage → B_spending_percentage = 0.85 := by
  sorry

end B_spends_85_percent_l430_430083


namespace total_toothpicks_l430_430652

theorem total_toothpicks (length width internal_spacing : ℕ) (h_length: length = 50) (h_width: width = 40) (h_internal_spacing: internal_spacing = 10) : 
  let vertical_segments := (length / internal_spacing) + 1,
      horizontal_segments := (width / internal_spacing) + 1,
      total_vertical_lines := length + 1 + (vertical_segments - 1),
      total_horizontal_lines := width + 1 + (horizontal_segments - 1),
      total_vertical_toothpicks := total_vertical_lines * width,
      total_horizontal_toothpicks := total_horizontal_lines * length in
  total_vertical_toothpicks + total_horizontal_toothpicks = 4490 :=
by
  sorry

end total_toothpicks_l430_430652


namespace sqrt_mult_simplify_l430_430245

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l430_430245


namespace sum_of_decimals_l430_430972

theorem sum_of_decimals : 1.000 + 0.101 + 0.011 + 0.001 = 1.113 :=
by
  sorry

end sum_of_decimals_l430_430972


namespace nonagon_diagonals_l430_430709

theorem nonagon_diagonals : 
  let n := 9
  convex_polygon n 9 -> 
  ∃ (d : ℕ), d = n * (n - 3) / 2 ∧ d = 27 := 
by
  sorry

end nonagon_diagonals_l430_430709


namespace matches_for_ladder_l430_430992

theorem matches_for_ladder (n : ℕ) (h : n = 25) : 
  (6 + 6 * (n - 1) = 150) :=
by
  sorry

end matches_for_ladder_l430_430992


namespace solve_for_x_l430_430642

theorem solve_for_x : ∀ x : ℝ, (3 * x + 15 = (1 / 3) * (6 * x + 45)) → x = 0 := by
  intros x h
  sorry

end solve_for_x_l430_430642


namespace tan_ratio_l430_430509

theorem tan_ratio (x y : ℝ) (h1 : Real.sin (x + y) = 5 / 8) (h2 : Real.sin (x - y) = 1 / 4) : 
  (Real.tan x / Real.tan y) = 7 / 3 :=
sorry

end tan_ratio_l430_430509


namespace lateral_surface_area_of_cylinder_l430_430826

theorem lateral_surface_area_of_cylinder :
  let r := 1
  let h := 2
  2 * Real.pi * r * h = 4 * Real.pi :=
by
  sorry

end lateral_surface_area_of_cylinder_l430_430826


namespace billy_finished_before_margaret_l430_430133

-- Define the conditions
def billy_first_laps_time : ℕ := 2 * 60
def billy_next_three_laps_time : ℕ := 4 * 60
def billy_ninth_lap_time : ℕ := 1 * 60
def billy_tenth_lap_time : ℕ := 150
def margaret_total_time : ℕ := 10 * 60

-- The main statement to prove that Billy finished 30 seconds before Margaret
theorem billy_finished_before_margaret :
  (billy_first_laps_time + billy_next_three_laps_time + billy_ninth_lap_time + billy_tenth_lap_time) + 30 = margaret_total_time :=
by
  sorry

end billy_finished_before_margaret_l430_430133


namespace find_BE_l430_430982

variables (A B C D E F G : Type) 
variables [parallelogram A B C D] [point E F G] (BD : line) (CD : line) (BC : line)

def intersect_at_vertex_A (E F G : point) : Prop :=
  intersects BD E ∧ intersects CD F ∧ intersects BC G ∧ 
  distance(B, E) = sqrt(10) ∧ ratio(FG, FE) = 9 ∧ distance(E, D) = 1

theorem find_BE : intersect_at_vertex_A E F G → distance(B, E) = sqrt(10) :=
by 
  sorry

end find_BE_l430_430982


namespace sqrt_mul_simplify_l430_430281

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l430_430281


namespace smallest_perimeter_of_consecutive_even_triangle_l430_430667

theorem smallest_perimeter_of_consecutive_even_triangle : ∃ (n : ℕ), (n > 1) ∧ 
  (let a := 2 * n in 
   let b := 2 * n + 2 in 
   let c := 2 * n + 4 in 
   a + b > c ∧ 
   a + c > b ∧ 
   b + c > a ∧ 
   a + b + c = 18) := sorry

end smallest_perimeter_of_consecutive_even_triangle_l430_430667


namespace most_likely_number_of_cars_l430_430653

theorem most_likely_number_of_cars 
  (total_time_seconds : ℕ)
  (rate_cars_per_second : ℚ)
  (h1 : total_time_seconds = 180)
  (h2 : rate_cars_per_second = 8 / 15) : 
  ∃ (n : ℕ), n = 100 :=
by
  sorry

end most_likely_number_of_cars_l430_430653


namespace coefficient_and_degree_l430_430951

theorem coefficient_and_degree (a b : ℤ) (m : ℕ) (n : ℕ) :
  a = -4 → b = m + n → m = 1 → n = 5 → 
  ((a, b) = (-4, 6)) :=
by
  intros h1 h2 h3 h4
  rw [h1, h3, h4] at h2
  assumption

end coefficient_and_degree_l430_430951


namespace sqrt_mul_simplify_l430_430274

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l430_430274


namespace john_calories_burned_per_day_l430_430492

variables (calories_per_day_eaten : ℕ) (calories_needed_per_pound : ℕ) (days : ℕ) (pounds : ℕ)

def total_calories_needed : ℕ := pounds * calories_needed_per_pound
def daily_calorie_deficit_needed : ℕ := total_calories_needed / days
def calories_burned_per_day : ℕ := calories_per_day_eaten + daily_calorie_deficit_needed

theorem john_calories_burned_per_day
  (h1 : calories_per_day_eaten = 1800)
  (h2 : calories_needed_per_pound = 4000)
  (h3 : days = 80)
  (h4 : pounds = 10) :
  calories_burned_per_day calories_per_day_eaten calories_needed_per_pound days pounds = 2300 :=
by {
  -- Proof would go here
  sorry
}

end john_calories_burned_per_day_l430_430492


namespace sum_of_perpendiculars_eq_height_l430_430514

theorem sum_of_perpendiculars_eq_height
  (a m m₁ m₂ m₃ : ℝ)
  (h_eq_triangle : ∃ (ABC : EquilateralTriangle), Triangle.height ABC = m)
  (h_perpendiculars : ∀ (P : Point), ∃ (perp₁ perp₂ perp₃ : ℝ), perp₁ + perp₂ + perp₃ = m) :
  m = m₁ + m₂ + m₃ := 
  sorry

end sum_of_perpendiculars_eq_height_l430_430514


namespace total_students_l430_430585

theorem total_students (N : ℕ) (h1 : ∃ g1 g2 g3 g4 g5 g6 : ℕ, 
  g1 = 13 ∧ g2 = 13 ∧ g3 = 13 ∧ g4 = 13 ∧ 
  ((g5 = 12 ∧ g6 = 12) ∨ (g5 = 14 ∧ g6 = 14)) ∧ 
  N = g1 + g2 + g3 + g4 + g5 + g6) : 
  N = 76 ∨ N = 80 :=
by
  sorry

end total_students_l430_430585


namespace smallest_perfect_square_divisible_by_3_and_5_l430_430046

theorem smallest_perfect_square_divisible_by_3_and_5 : ∃ (n : ℕ), n > 0 ∧ (∃ (m : ℕ), n = m * m) ∧ (n % 3 = 0) ∧ (n % 5 = 0) ∧ n = 225 :=
by
  sorry

end smallest_perfect_square_divisible_by_3_and_5_l430_430046


namespace friendly_snakes_not_blue_l430_430939

variable (Snakes : Type)
variable (sally_snakes : Finset Snakes)
variable (blue : Snakes → Prop)
variable (friendly : Snakes → Prop)
variable (can_swim : Snakes → Prop)
variable (can_climb : Snakes → Prop)

variable [DecidablePred blue] [DecidablePred friendly] [DecidablePred can_swim] [DecidablePred can_climb]

-- The number of snakes in Sally's collection
axiom h_snakes_count : sally_snakes.card = 20
-- There are 7 blue snakes
axiom h_blue : (sally_snakes.filter blue).card = 7
-- There are 10 friendly snakes
axiom h_friendly : (sally_snakes.filter friendly).card = 10
-- All friendly snakes can swim
axiom h1 : ∀ s ∈ sally_snakes, friendly s → can_swim s
-- No blue snakes can climb
axiom h2 : ∀ s ∈ sally_snakes, blue s → ¬ can_climb s
-- Snakes that can't climb also can't swim
axiom h3 : ∀ s ∈ sally_snakes, ¬ can_climb s → ¬ can_swim s

theorem friendly_snakes_not_blue :
  ∀ s ∈ sally_snakes, friendly s → ¬ blue s :=
by
  sorry

end friendly_snakes_not_blue_l430_430939


namespace part1_l430_430064

theorem part1 (P Q R : Polynomial ℝ) : 
  ¬ ∃ (P Q R : Polynomial ℝ), (∀ x y z : ℝ, (x - y + 1)^3 * P.eval x + (y - z - 1)^3 * Q.eval y + (z - 2 * x + 1)^3 * R.eval z = 1) := sorry

end part1_l430_430064


namespace smallest_number_divisible_5_13_7_l430_430523

theorem smallest_number_divisible_5_13_7 : ∃ n, n > 0 ∧ n % 5 = 0 ∧ n % 13 = 0 ∧ n % 7 = 0 ∧ ∀ m, (m > 0 ∧ m % 5 = 0 ∧ m % 13 = 0 ∧ m % 7 = 0) → m ≥ n :=
begin
  use 455,
  split,
  { exact nat.zero_lt_succ 454 },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd.intro 91 rfl) },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd.intro 35 rfl) },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd.intro 65 rfl) },
  intro m,
  intro h,
  cases h with hm_pos hdiv,
  cases hdiv with h1 hdiv2,
  cases hdiv2 with h2 h3,
  have hmult := nat.mul_le_mul_left 5 (nat.mul_le_mul_left 13 (nat.mul_le_mul_left 7 hm_pos)),
  simp at *,
  exact hmult,
  sorry
end

end smallest_number_divisible_5_13_7_l430_430523


namespace area_of_triangle_BCD_l430_430871

theorem area_of_triangle_BCD
  (h₁ : ∀ (A B C : Type) {area_ABC : ℝ} (a₁ a₂ a₃ : ℝ → Prop), a₁ = area_ABC → area_ABC = 36 → AC = 8)
  (h₂ : ∀ {AC : ℝ} (length_AC : ℝ), length_AC = 8)
  (h₃ : ∀ {CD : ℝ} {AC : ℝ} (length_CD : ℝ), length_CD = 2 * AC)
  : (1/2) * CD * 9 = 72 
:= by
sorry

end area_of_triangle_BCD_l430_430871


namespace train_speed_in_km_per_hr_l430_430116

/-- Given the length of a train and a bridge, and the time taken for the train to cross the bridge, prove the speed of the train in km/hr -/
theorem train_speed_in_km_per_hr
  (train_length : ℕ)  -- 100 meters
  (bridge_length : ℕ) -- 275 meters
  (crossing_time : ℕ) -- 30 seconds
  (conversion_factor : ℝ) -- 1 m/s = 3.6 km/hr
  (h_train_length : train_length = 100)
  (h_bridge_length : bridge_length = 275)
  (h_crossing_time : crossing_time = 30)
  (h_conversion_factor : conversion_factor = 3.6) : 
  (train_length + bridge_length) / crossing_time * conversion_factor = 45 := 
sorry

end train_speed_in_km_per_hr_l430_430116


namespace douglas_vote_percent_in_county_X_l430_430877

theorem douglas_vote_percent_in_county_X (V : ℝ) :
  let P := (0.74 : ℝ)
  let PX := P * 100
  let total_votes := 3 * V
  let votes_in_Y := V
  let votes_in_X := 2 * V
  let percent_votes_Y := 0.5000000000000002
  let percent_total := 0.66
((percent_total * total_votes) =
    ((percent_votes_Y * votes_in_Y) + ((PX / 100) * votes_in_X))) →
  PX = 74
:=
by {
  intro h,

  -- Statement to prove: P = 74
  sorry,
}

end douglas_vote_percent_in_county_X_l430_430877


namespace decomposition_l430_430692

def vector3 := (ℝ × ℝ × ℝ)

def x : vector3 := (-2, 4, 7)
def p : vector3 := (0, 1, 2)
def q : vector3 := (1, 0, 1)
def r : vector3 := (-1, 2, 4)

def scalar_mul (a : ℝ) (v : vector3) : vector3 :=
  (a * v.1, a * v.2, a * v.3)

def vector_add (v1 v2 : vector3) : vector3 :=
  (v1.1 + v2.1, v1.2 + v2.2, v1.3 + v2.3)

def vector_sub (v1 v2 : vector3) : vector3 :=
  (v1.1 - v2.1, v1.2 - v2.2, v1.3 - v2.3)

theorem decomposition :
  x = vector_add (vector_add (scalar_mul 2 p) (vector_sub (scalar_mul (-1) q) r)) sorry :=
by sorry

end decomposition_l430_430692


namespace fence_length_l430_430869

theorem fence_length (r : ℝ) (θ : ℝ) 
  (h1 : r = 30) 
  (h2 : θ = (120 * Real.pi) / 180) : 
  (r * θ + 2 * r = 20 * Real.pi + 60) := 
by {
  rw [h1, h2],
  norm_num,
  ring,
}

end fence_length_l430_430869


namespace car_y_speed_l430_430330

noncomputable def carY_average_speed (vX : ℝ) (tY : ℝ) (d : ℝ) : ℝ :=
  d / tY

theorem car_y_speed (vX : ℝ := 35) (tY_min : ℝ := 72) (dX_after_Y : ℝ := 245) :
  carY_average_speed vX (dX_after_Y / vX) dX_after_Y = 35 := 
by
  sorry

end car_y_speed_l430_430330


namespace fibonacci_identity_l430_430337

noncomputable def Φ : ℝ := (1 + Real.sqrt 5) / 2

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

theorem fibonacci_identity (n : ℕ) : 
  fibonacci (n + 1) - Φ * fibonacci n = (-1/Φ)^n := by
  sorry

end fibonacci_identity_l430_430337


namespace sqrt_mul_simplify_l430_430279

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l430_430279


namespace total_students_total_students_alt_l430_430551

def number_of_students (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 
    a + b = 6 ∧ 
    a = 4 ∧ 
    (∀ g, g = 13) ∧ 
    (b = 2 ∧ ((g = 12 ∨ g = 14) ∧ ∀ h, g - h ≤ 1)) ∧ 
    (n = 52 + 2 * 12 ∨ n = 52 + 2 * 14)

theorem total_students : ∃ n : ℕ, number_of_students n :=
by
  use 76
  sorry

theorem total_students_alt : ∃ n : ℕ, number_of_students n :=
by
  use 80
  sorry

end total_students_total_students_alt_l430_430551


namespace no_solution_l430_430769

theorem no_solution (x : ℝ) : ¬ (3 * x - 2 < (x + 2)^2 ∧ (x + 2)^2 < 9 * x - 5) :=
by
  sorry

end no_solution_l430_430769


namespace neg_existential_proposition_l430_430842

open Nat

theorem neg_existential_proposition :
  (¬ (∃ n : ℕ, n + 10 / n < 4)) ↔ (∀ n : ℕ, n + 10 / n ≥ 4) :=
by
  sorry

end neg_existential_proposition_l430_430842


namespace dealership_suv_sales_l430_430710

theorem dealership_suv_sales
  (trucks_to_suvs_ratio : ℚ := 3/5)
  (expected_trucks : ℕ := 30)
  (suvs_to_vans_ratio : ℚ := 2/1)
  (expected_vans : ℕ := 15)
  : (expected_trucks / trucks_to_suvs_ratio) = 30 := 
by 
  have suvs_from_trucks := expected_trucks / trucks_to_suvs_ratio
  have suvs_from_vans := expected_vans * suvs_to_vans_ratio
  have suvs := min suvs_from_trucks suvs_from_vans
  exact eq.symm (min_eq_right (by norm_num : 30 = min 50 30))

-- Proof is omitted and marked with sorry when more complex, only for representation purposes.

end dealership_suv_sales_l430_430710


namespace jimmy_climb_l430_430901

/-- 
Jimmy takes 30 seconds to climb the first flight of stairs. 
Each following flight takes 10 seconds more than the preceding one. 
He climbs a total of 6 flights. 
Each flight of stairs has 12 steps, and each step is 20 cm high.
How long does it take Jimmy to climb these flights, and what is the total distance he climbs in meters? 
-/
theorem jimmy_climb:
  ∑ i in Finset.range 6, (30 + 10 * i) = 330 ∧ 6 * (12 * 20 / 100) = 14.4 :=
by
  sorry

end jimmy_climb_l430_430901


namespace ellipse_constant_product_l430_430809

theorem ellipse_constant_product 
  (a b : ℝ) (h_cond : a > b ∧ b > 0) (eccentricity : a^2 = b^2 + (a / 2)^2) 
  (area_triangle : a * b = 2 * sqrt 3)
  (line_l : ∀ (x : ℝ), x = 2 * sqrt 2) :
  let A1 := (-2 : ℝ, 0),
      A2 := (2 : ℝ, 0),
      ellipse_eq := (x y : ℝ) -> (x^2 / 4) + (y^2 / 3) = 1 in
  ∀ P (hP : (P.1^2 / 4) + (P.2^2 / 3) = 1 ∧ P ≠ A1 ∧ P ≠ A2),
  let E := (2 * sqrt 2, (2 * sqrt 2 + 2) * P.2 / (P.1 + 2)),
      F := (2 * sqrt 2, (2 * sqrt 2 - 2) * P.2 / (P.1 - 2)) in
  abs (E.1 - D.1) * abs (F.1 - D.1) = 3 :=
sorry

end ellipse_constant_product_l430_430809


namespace tammy_trees_l430_430004

-- Define the conditions as Lean definitions and the final statement to prove
theorem tammy_trees :
  (∀ (days : ℕ) (earnings : ℕ) (pricePerPack : ℕ) (orangesPerPack : ℕ) (orangesPerTree : ℕ),
    days = 21 →
    earnings = 840 →
    pricePerPack = 2 →
    orangesPerPack = 6 →
    orangesPerTree = 12 →
    (earnings / days) / (pricePerPack / orangesPerPack) / orangesPerTree = 10) :=
by
  intros days earnings pricePerPack orangesPerPack orangesPerTree
  sorry

end tammy_trees_l430_430004


namespace trigonometric_identity_l430_430796

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 4) :
  (Real.sin θ + Real.cos θ) / (17 * Real.sin θ) + (Real.sin θ ^ 2) / 4 = 21 / 68 := 
sorry

end trigonometric_identity_l430_430796


namespace find_ab_l430_430406

theorem find_ab (a b : ℝ) 
  (h1 : a + b = 5) 
  (h2 : a^3 + b^3 = 35) : a * b = 6 := 
by
  sorry

end find_ab_l430_430406


namespace amount_of_H2O_formed_l430_430770

-- Define the balanced chemical equation as a relation
def balanced_equation : Prop :=
  ∀ (naoh hcl nacl h2o : ℕ), 
    (naoh + hcl = nacl + h2o)

-- Define the reaction of 2 moles of NaOH and 2 moles of HCl
def reaction (naoh hcl : ℕ) : ℕ :=
  if (naoh = 2) ∧ (hcl = 2) then 2 else 0

theorem amount_of_H2O_formed :
  balanced_equation →
  reaction 2 2 = 2 :=
by
  sorry

end amount_of_H2O_formed_l430_430770


namespace max_n_for_a_n_eq_2008_l430_430909

def sum_of_digits (a : ℕ) : ℕ :=
  (Nat.digits 10 a).sum

def sequence_a (a_1 : ℕ) (n : ℕ) : ℕ :=
  Nat.recOn n a_1 (λ n a_n, a_n + sum_of_digits a_n)

theorem max_n_for_a_n_eq_2008 : ∃ n, n = 6 ∧ sequence_a 1919 n = 2008 :=
by
  sorry

end max_n_for_a_n_eq_2008_l430_430909


namespace sqrt_mul_eq_l430_430309

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l430_430309


namespace true_proposition_count_l430_430022

theorem true_proposition_count : ∃ n : ℕ, n = 2 ∧ 
  (∀ x : ℝ, (x > -3 → x > -6) ∧  -- original proposition
  (¬ (x > -6 → x > -3)) ∧        -- converse proposition
  (¬ (x ≤ -3 → x ≤ -6)) ∧        -- inverse proposition
  (x ≤ -6 → x ≤ -3))             -- contrapositive proposition
  :=
begin
  use 2,
  refine ⟨rfl, _⟩,
  intro x,
  split,
  {
    intro h,
    linarith,
  },
  split,
  {
    intro h,
    linarith,
  },
  split,
  {
    intro h,
    linarith,
  },
  {
    intro h,
    linarith,
  },
end

end true_proposition_count_l430_430022


namespace minimum_value_sqrt_7_l430_430913

noncomputable def minimum_value_complex_expression (p q r : ℕ) (ζ : ℂ) : ℂ :=
  abs (p + q * ζ + r * ζ^3)

theorem minimum_value_sqrt_7 (p q r : ℕ) (ζ : ℂ) 
  (hpqrdistinct : p ≠ q ∧ q ≠ r ∧ r ≠ p ∧ p > 0 ∧ q > 0 ∧ r > 0)
  (hζ : ζ^4 = 1 ∧ ζ ≠ 1) :
  minimum_value_complex_expression p q r ζ = √7 := sorry

end minimum_value_sqrt_7_l430_430913


namespace nicky_pace_l430_430520

theorem nicky_pace :
  ∃ v : ℝ, v = 3 ∧ (
    ∀ (head_start : ℝ) (cristina_pace : ℝ) (time : ℝ) (distance_encounter : ℝ), 
      head_start = 36 ∧ cristina_pace = 4 ∧ time = 36 ∧ distance_encounter = cristina_pace * time - head_start →
      distance_encounter / time = v
  ) :=
sorry

end nicky_pace_l430_430520


namespace triangle_ratios_l430_430752

theorem triangle_ratios (A B C P Q K L M N R : Point)
  (hP_on_BC : P ∈ lineSegment B C) (hQ_on_BC : Q ∈ lineSegment B C)
  (hK_perpendicular : perp P K (lineSegment A C)) (hL_perpendicular : perp Q L (lineSegment A C))
  (hM_on_AB : M ∈ lineSegment A B) (hN_on_AB : N ∈ lineSegment A B)
  (hPM_eq_PA : dist P M = dist P A) (hQN_eq_QA : dist Q N = dist Q A)
  (h_circ_AKM : R ∈ circumcircle (triangle A K M))
  (h_circ_ALN : R ∈ circumcircle (triangle A L N))
  (hR_midpoint_BC : R = midpoint B C) :
  (dist B C / dist A C = Real.sqrt 2) ∧ (dist A C / dist A B = Real.sqrt 3) :=
sorry

end triangle_ratios_l430_430752


namespace find_c_d_l430_430611

theorem find_c_d (c d : ℝ) (h1 : c ≠ 0) (h2 : d ≠ 0)
  (h3 : ∀ x : ℝ, x^2 + c * x + d = 0 → (x = c ∧ x = d)) :
  c = 1 ∧ d = -2 :=
by
  sorry

end find_c_d_l430_430611


namespace regular_n_gon_has_largest_area_regular_n_gon_has_largest_perimeter_l430_430527

-- Circle S and inscribed n-gon definitions
def Circle (r : ℝ) := {p : ℝ × ℝ // p.1 ^ 2 + p.2 ^ 2 = r ^ 2}
def n_gon (n : ℕ) (C : ℝ × ℝ → Prop) := {vertices : Fin n → ℝ × ℝ // ∀ i, C (vertices i)}

-- Regular n-gon definition
def regular_n_gon (n : ℕ) (C : ℝ × ℝ → Prop) := n_gon n C

-- Definitions to compare areas and perimeters of the inscribed n-gons
def Area (n : ℕ) (G : n_gon n _) := sorry -- Define the function for the area of an n-gon
def Perimeter (n : ℕ) (G : n_gon n _) := sorry -- Define the function for the perimeter of an n-gon

-- Proving largest area
theorem regular_n_gon_has_largest_area (r : ℝ) (n : ℕ) (C : ℝ × ℝ → Prop) (S : C = Circle r) (G : n_gon n C) 
  : Area n (regular_n_gon n C) ≥ Area n G := sorry

-- Proving largest perimeter
theorem regular_n_gon_has_largest_perimeter (r : ℝ) (n : ℕ) (C : ℝ × ℝ → Prop) (S : C = Circle r) (G : n_gon n C) 
  : Perimeter n (regular_n_gon n C) ≥ Perimeter n G := sorry

end regular_n_gon_has_largest_area_regular_n_gon_has_largest_perimeter_l430_430527


namespace equilateral_triangle_area_l430_430962

theorem equilateral_triangle_area (p : ℝ) : 
  let s := p in 
  let area := (sqrt 3 / 4) * s ^ 2 in
  (3 * s = 3 * p) → area = (sqrt 3 / 4) * p ^ 2 :=
by
  intro h
  let s := p
  let area := (sqrt 3 / 4) * s ^ 2
  have h1 : s = p := rfl
  rw h1
  use area
  sorry

end equilateral_triangle_area_l430_430962


namespace constant_term_in_expansion_l430_430952

noncomputable def binom_expansion (x : ℝ) : ℝ :=
  (sqrt x - 2 / sqrt x) ^ 6

theorem constant_term_in_expansion :
  ∀ x : ℝ, x ≠ 0 →  binom_expansion x = -160 :=
by
  sorry

end constant_term_in_expansion_l430_430952


namespace eccentricity_trajectory_P_l430_430885

def point3D := (ℝ × ℝ × ℝ)

def midpoint (p1 p2 : point3D) : point3D := 
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

def distance (p1 p2 : point3D) : ℝ :=
  (real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2))

noncomputable def cube_vertices : List point3D :=
  [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0), (0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2)]

def midpoint_BB1 := midpoint (2, 0, 0) (2, 0, 2)
def midpoint_B1C1 := midpoint (2, 0, 2) (2, 2, 2)

def plane_DMN_equation := (x y z : ℝ) → 3 * x - y + z + 2 = 0

axiom distance_condition (P : point3D) (hP : plane_DMN_equation P.1 P.2 P.3) :
  let dist_P_plane := abs (P.1 - 2) 
  let dist_PD := distance P (0, 2, 0)
  dist_P_plane = dist_PD

theorem eccentricity_trajectory_P : 
  ∃ (ecc : ℝ), ecc = 2 * real.sqrt 34 / 17 :=
  sorry

end eccentricity_trajectory_P_l430_430885


namespace polynomial_not_divisible_iff_divisible_by_3_l430_430414

noncomputable def polynomial_non_divisible (n : ℕ) : Prop :=
  ¬ (x^2 + x + 1 ∣ x^(2*n) + 1 + (x + 1)^(2*n))

theorem polynomial_not_divisible_iff_divisible_by_3 (n : ℕ) :
  polynomial_non_divisible n ↔ (3 ∣ n) :=
sorry

end polynomial_not_divisible_iff_divisible_by_3_l430_430414


namespace sqrt_multiplication_l430_430145

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l430_430145


namespace sqrt_mul_simplify_l430_430276

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l430_430276


namespace lucas_sequence_100_mod_10_l430_430515

def lucas_sequence (n : ℕ) : ℕ → ℕ
| 0     := 2
| 1     := 5
| (m+2) := (lucas_sequence m + lucas_sequence (m + 1)) % 10

theorem lucas_sequence_100_mod_10 : lucas_sequence 100 % 10 = 2 :=
by
  sorry

end lucas_sequence_100_mod_10_l430_430515


namespace smallest_sum_of_squares_l430_430622

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 187) : x^2 + y^2 ≥ 205 := sorry

end smallest_sum_of_squares_l430_430622


namespace total_students_possible_l430_430561

theorem total_students_possible (A B : ℕ) :
  (4 * 13) + 2 * A = 76 ∨ (4 * 13) + 2 * B = 80 :=
by
  -- Let N be the total number of students
  let N := (4 * 13)
  -- Given that the number of students in the remaining 2 groups differs by no more than 1
  have h : A = 12 ∨ B = 14 := sorry
  -- Prove the possible values
  exact or.inl (N + 2 * 12 = 76) <|> or.inr (N + 2 * 14 = 80)

end total_students_possible_l430_430561


namespace num_valid_hex_numbers_sum_digits_of_count_l430_430442

def hexadecimal_valid_numbers (n : ℕ) : Prop :=
  n < 500 ∧ ∀ d ∈ (nat.digits 16 n), d < 10

theorem num_valid_hex_numbers :
  let count := (finset.range 500).filter hexadecimal_valid_numbers).card
  count = 199 := by
  sorry

theorem sum_digits_of_count :
  let count := (finset.range 500).filter hexadecimal_valid_numbers).card
  (nat.digits 10 count).sum = 19 := by
  sorry

end num_valid_hex_numbers_sum_digits_of_count_l430_430442


namespace circle_area_k_l430_430948

theorem circle_area_k (C : ℝ) (hC : C = 36 * Real.pi) : ∃ k : ℝ, (∀ r : ℝ, C = 2 * Real.pi * r → k * Real.pi = Real.pi * r^2) ∧ k = 324 :=
begin
  sorry
end

end circle_area_k_l430_430948


namespace mutual_exclusion_not_opposite_l430_430055

theorem mutual_exclusion_not_opposite :
  let pocket := [{color := "red", id := 1}, {color := "red", id := 2},
                 {color := "black", id := 1}, {color := "black", id := 2}]
  let event_exactly_one_black (chosen : List {color : String, id : Nat}) :=
    chosen.length = 2 ∧ chosen.countp (λ b => b.color = "black") = 1
  let event_exactly_two_black (chosen : List {color : String, id : Nat}) :=
    chosen.length = 2 ∧ chosen.all (λ b => b.color = "black")
  ∀ (chosen : List {color : String, id : Nat}),
    chosen ⊆ pocket → (event_exactly_one_black chosen ↔ ¬ event_exactly_two_black chosen) := 
begin
  -- proof goes here
  sorry
end

end mutual_exclusion_not_opposite_l430_430055


namespace total_students_l430_430564

noncomputable def totalStudentsOptions (groups totalGroups specificGroupCount specificGroupSize otherGroupSizes : ℕ) : Set ℕ :=
  if totalGroups = 6 ∧ specificGroupCount = 4 ∧ specificGroupSize = 13 ∧ (otherGroupSizes = 12 ∨ otherGroupSizes = 14) then
    {52 + 2 * otherGroupSizes}
  else
    ∅

theorem total_students :
  totalStudentsOptions 6 4 13 12 = {76} ∧ totalStudentsOptions 6 4 13 14 = {80} :=
by
  -- This is where the proof would go, but we're skipping it as per instructions
  sorry

end total_students_l430_430564


namespace sqrt_multiplication_l430_430148

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l430_430148


namespace seq_is_triangular_in_base_9_l430_430595

noncomputable def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

noncomputable def seq_in_base_9 (n : ℕ) : ℕ
| 0 => 1
| (n + 1) => 9 * seq_in_base_9 n + 1

theorem seq_is_triangular_in_base_9 (n : ℕ) :
  ∃ m : ℕ, seq_in_base_9 n = triangular_number m :=
by
  sorry

end seq_is_triangular_in_base_9_l430_430595


namespace blue_marble_difference_l430_430989

theorem blue_marble_difference :
  ∃ a b : ℕ, (10 * a = 10 * b) ∧ (3 * a + b = 80) ∧ (7 * a - 9 * b = 40) := by
  sorry

end blue_marble_difference_l430_430989


namespace range_of_m_l430_430449

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → x^2 - x + 1 / 2 ≥ m) ↔ m ≤ 1 / 4 :=
by
  intro h
  sorry

end range_of_m_l430_430449


namespace ellipse_major_axis_length_l430_430126

theorem ellipse_major_axis_length (F1 F2 : ℝ × ℝ) (tangent_point : ℝ × ℝ) 
  (hF1 : F1 = (15, 30)) 
  (hF2 : F2 = (15, 90)) 
  (htangent : tangent_point.1 = 0) 
  (htangent_distance : tangent_point.2 = 60) :
  let distance (p1 p2 : ℝ × ℝ) : ℝ := ( (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2 ) ^ (1/2)
  in
  distance tangent_point F1 + distance tangent_point F2 = 30 * Real.sqrt 5 :=
sorry

end ellipse_major_axis_length_l430_430126


namespace sum_of_powers_of_i_l430_430816

theorem sum_of_powers_of_i : 
  ∀ (i : ℂ), i^2 = -1 → 1 + i + i^2 + i^3 + i^4 + i^5 + i^6 = i :=
by
  intro i h
  sorry

end sum_of_powers_of_i_l430_430816


namespace domain_of_f_l430_430363

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 7*x + 10)

def is_root (x : ℝ) : Prop := x^2 + 7 * x + 10 = 0

theorem domain_of_f :
  ∀ x : ℝ, x ∈ (-∞, -5) ∪ (-5, -2) ∪ (-2, ∞) ↔ ¬ is_root x :=
by sorry

end domain_of_f_l430_430363


namespace certain_positive_integer_value_l430_430867

-- Define factorial
def fact : ℕ → ℕ 
| 0     => 1
| (n+1) => (n+1) * fact n

-- Statement of the problem
theorem certain_positive_integer_value (i k m a : ℕ) :
  (fact 8 = 2^i * 3^k * 5^m * 7^a) ∧ (i + k + m + a = 11) → a = 1 :=
by 
  sorry

end certain_positive_integer_value_l430_430867


namespace beatrix_reads_704_pages_l430_430340

theorem beatrix_reads_704_pages : 
  ∀ (B C : ℕ), 
  C = 3 * B + 15 ∧ C = B + 1423 → B = 704 :=
by
  intro B C
  intro h
  sorry

end beatrix_reads_704_pages_l430_430340


namespace max_cos_half_sin_eq_1_l430_430369

noncomputable def max_value_expression (θ : ℝ) : ℝ :=
  Real.cos (θ / 2) * (1 - Real.sin θ)

theorem max_cos_half_sin_eq_1 : 
  ∀ θ : ℝ, 0 < θ ∧ θ < π → max_value_expression θ ≤ 1 :=
by
  intros θ h
  sorry

end max_cos_half_sin_eq_1_l430_430369


namespace problem1_problem2_l430_430746

-- Equivalent proof for Problem 1
theorem problem1 :
  sqrt (2 / 3) - (1 / 6 * sqrt 24 - 3 / 2 * sqrt 12) = 3 * sqrt 3 :=
by
  sorry

-- Equivalent proof for Problem 2
theorem problem2 :
  sqrt 12 + (1 / 3) ^ (-1 : ℤ) + abs (sqrt 3 - 1) - (Real.pi - 2) ^ 0 = 3 * sqrt 3 + 1 :=
by
  sorry

end problem1_problem2_l430_430746


namespace average_speed_whole_journey_l430_430683

-- Define the conditions given in the problem
def distance_to_place := 150 -- km
def speed_to_place := 50 -- km/hr
def speed_return := 30 -- km/hr

-- Define the correct answer from the solution
def correct_avg_speed := 37.5 -- km/hr

theorem average_speed_whole_journey : 
  let distance := distance_to_place + distance_to_place in
  let time_to_place := distance_to_place / speed_to_place in
  let time_return := distance_to_place / speed_return in
  let total_time := time_to_place + time_return in
  let avg_speed := distance / total_time in
  avg_speed = correct_avg_speed :=
by
  sorry

end average_speed_whole_journey_l430_430683


namespace max_spheres_in_cone_l430_430895

-- Define the parameters of the truncated cone and the spheres
variables {h: ℝ} (h_pos: h = 8)
variables {r1: ℝ} (r1_pos: r1 = 2)
variables {r2: ℝ} (r2_pos: r2 = 3)

-- Define the conditions about the positions and tangency of the spheres
variables (O1: ℝ × ℝ × ℝ) (O1_cond1: O1.2 = r1)
variables (O2: ℝ × ℝ × ℝ) (O2_cond1: O2.2 = r2)
variables (O1_O2_distance: ℝ) (distance_cond: O1_O2_distance = r1 + r2)

-- Define the maximum number of additional spheres with radius 3 that can be added
def max_additional_spheres (h r1 r2 O1 O2: ℝ): ℝ := 
  2

theorem max_spheres_in_cone: 
  ∀ (h r1 r2 O1 O2 : ℝ), 
  h = 8 → r1 = 2 → r2 = 3 → r1 + r2 = O1.2 + O2.2 → max_additional_spheres h r1 r2 O1 O2 = 2 := sorry

end max_spheres_in_cone_l430_430895


namespace tan_diff_angle_neg7_l430_430858

-- Define the main constants based on the conditions given
variables (α : ℝ)
axiom sin_alpha : Real.sin α = -3/5
axiom alpha_in_fourth_quadrant : 0 < α ∧ α < 2 * Real.pi ∧ α > 3 * Real.pi / 2

-- Define the statement that needs to be proven based on the question and the correct answer
theorem tan_diff_angle_neg7 : 
  Real.tan (α - Real.pi / 4) = -7 :=
sorry

end tan_diff_angle_neg7_l430_430858


namespace sqrt3_mul_sqrt12_eq_6_l430_430255

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l430_430255


namespace expressions_are_rational_l430_430760

-- Define each of the given expressions
def expr1 := Real.sqrt ((9 / 16) ^ 2)
def expr2 := Real.cbrt 0.125
def expr3 := Real.root 4 0.004096
def expr4 := Real.cbrt (-8) * Real.sqrt ((0.25)⁻¹)

-- Define the rationality condition for each expression
def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- Define the main theorem that all expressions are rational
theorem expressions_are_rational : 
  is_rational expr1 ∧ 
  is_rational expr2 ∧ 
  is_rational expr3 ∧ 
  is_rational expr4 :=
by
  sorry

end expressions_are_rational_l430_430760


namespace dan_total_dimes_l430_430741

-- Definitions of the conditions
def worth_of_dime : ℝ := 0.10
def worth_of_dimes_barry : ℝ := 10.00
def worth_of_dimes_dan_initial : ℝ := worth_of_dimes_barry / 2
def extra_dimes_dan_found : ℝ := 2 * worth_of_dime

-- Main proof statement
theorem dan_total_dimes : 
  let total_dimes_barry := worth_of_dimes_barry / worth_of_dime in 
  let initial_dimes_dan := total_dimes_barry / 2 in 
  let total_dimes_dan := initial_dimes_dan + (extra_dimes_dan_found / worth_of_dime) in
  total_dimes_dan = 52 := 
by
  sorry

end dan_total_dimes_l430_430741


namespace function_satisfies_equation_l430_430755

noncomputable def f (x : ℝ) : ℝ := x + 1 / x + 1 / (x - 1)

theorem function_satisfies_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) :
  f ((x - 1) / x) + f (1 / (1 - x)) = 2 - 2 * x := by
  sorry

end function_satisfies_equation_l430_430755


namespace number_of_subsets_satisfying_conditions_l430_430851

theorem number_of_subsets_satisfying_conditions : 
  let T := { T : set ℕ | (∀ i ∈ T, i ≤ 20 ∧ i ≥ 1 ∧ (i + 1 ∉ T) ∧ (i - 1 ∉ T)) ∧ ∃ k, T.card = k ∧ (T ≠ ∅ → ∀ i ∈ T, i ≥ k) } in 
  (∑ k in finset.range 11 \ {0}, nat.choose (20 - 2*(k - 1)) k) = 2812 :=
by sorry

end number_of_subsets_satisfying_conditions_l430_430851


namespace sqrt_mul_simplify_l430_430272

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l430_430272


namespace inequality_holds_equality_cases_l430_430935

noncomputable def posReal : Type := { x : ℝ // 0 < x }

variables (a b c d : posReal)

theorem inequality_holds (a b c d : posReal) :
  (a.1 - b.1) * (a.1 - c.1) / (a.1 + b.1 + c.1) +
  (b.1 - c.1) * (b.1 - d.1) / (b.1 + c.1 + d.1) +
  (c.1 - d.1) * (c.1 - a.1) / (c.1 + d.1 + a.1) +
  (d.1 - a.1) * (d.1 - b.1) / (d.1 + a.1 + b.1) ≥ 0 :=
sorry

theorem equality_cases (a b c d : posReal) :
  (a.1 - b.1) * (a.1 - c.1) / (a.1 + b.1 + c.1) +
  (b.1 - c.1) * (b.1 - d.1) / (b.1 + c.1 + d.1) +
  (c.1 - d.1) * (c.1 - a.1) / (c.1 + d.1 + a.1) +
  (d.1 - a.1) * (d.1 - b.1) / (d.1 + a.1 + b.1) = 0 ↔
  (a.1 = c.1 ∧ b.1 = d.1) :=
sorry

end inequality_holds_equality_cases_l430_430935


namespace combined_tickets_l430_430740

-- Definitions from the conditions
def dave_spent : Nat := 43
def dave_left : Nat := 55
def alex_spent : Nat := 65
def alex_left : Nat := 42

-- Theorem to prove that the combined starting tickets of Dave and Alex is 205
theorem combined_tickets : dave_spent + dave_left + alex_spent + alex_left = 205 := 
by
  sorry

end combined_tickets_l430_430740


namespace simplify_trig_expression_l430_430074

theorem simplify_trig_expression (α : ℝ) :
  (cos (α + (3 / 2) * real.pi) + 2 * cos ((11 / 6) * real.pi - α)) / 
  (2 * sin ((real.pi / 3) + α) + sqrt 3 * sin ((3 / 2) * real.pi - α)) = 
  sqrt 3 * real.cot α :=
sorry

end simplify_trig_expression_l430_430074


namespace sqrt_mult_eq_six_l430_430318

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l430_430318


namespace probability_at_least_one_not_in_use_l430_430127

theorem probability_at_least_one_not_in_use :
    ∀ (n : ℕ) (p : ℝ), n = 20 → p = 0.8 → (1 - p ^ n) = 1 - 0.8 ^ 20 :=
by
  assume n p hn hp
  rw [hn, hp]
  sorry

end probability_at_least_one_not_in_use_l430_430127


namespace find_other_number_l430_430687

theorem find_other_number (a b : ℕ) (h_lcm: Nat.lcm a b = 2310) (h_hcf: Nat.gcd a b = 55) (h_a: a = 210) : b = 605 := by
  sorry

end find_other_number_l430_430687


namespace total_students_in_groups_l430_430532

theorem total_students_in_groups {N : ℕ} (h1 : ∃ g : ℕ → ℕ, (∀ i j, g i = 13 ∨ g j = 12 ∨ g j = 14) ∧ (∑ i in finset.range 6, g i) = N) : 
  N = 76 ∨ N = 80 :=
sorry

end total_students_in_groups_l430_430532


namespace total_students_total_students_alt_l430_430553

def number_of_students (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 
    a + b = 6 ∧ 
    a = 4 ∧ 
    (∀ g, g = 13) ∧ 
    (b = 2 ∧ ((g = 12 ∨ g = 14) ∧ ∀ h, g - h ≤ 1)) ∧ 
    (n = 52 + 2 * 12 ∨ n = 52 + 2 * 14)

theorem total_students : ∃ n : ℕ, number_of_students n :=
by
  use 76
  sorry

theorem total_students_alt : ∃ n : ℕ, number_of_students n :=
by
  use 80
  sorry

end total_students_total_students_alt_l430_430553


namespace sqrt3_mul_sqrt12_eq_6_l430_430230

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l430_430230


namespace arithmetic_sequence_general_formula_and_min_value_l430_430806

variable (a_n S_n : ℕ → ℤ)
variable (n : ℕ)

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ a1 d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = ((n + 1) * a 0 + n * (n + 1) / 2 * (a 1 - a 0)) / 2

theorem arithmetic_sequence_general_formula_and_min_value :
  (is_arithmetic_sequence a_n)
  ∧ (sum_of_first_n_terms S_n a_n)
  ∧ (S_n = λ n, n^2 - 10 * n)
  → (a_n = λ n, 2 * n - 11)
  ∧ (∃ n_min : ℕ, ∀ n : ℕ, S_n n ≥ S_n n_min ∧ S_n n_min = -25) :=
by
  sorry

end arithmetic_sequence_general_formula_and_min_value_l430_430806


namespace maximize_cone_volume_l430_430649

-- Define the volume of the cone as a function of the height x
def cone_volume (x : ℝ) : ℝ := (1 / 3) * π * x * (20^2 - x^2)

-- Define the statement we need to prove
theorem maximize_cone_volume : ∃ x, 0 < x ∧ x < 20 ∧ x = (20 * real.sqrt 3) / 3 ∧
  (∀ y, 0 < y ∧ y < 20 → cone_volume y ≤ cone_volume x) :=
sorry

end maximize_cone_volume_l430_430649


namespace sum_of_extrema_of_f_l430_430971

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / ((Real.sin x) + 3)

theorem sum_of_extrema_of_f :
  (let max_val := max (f (Real.pi / 2)) (f (-Real.pi / 2)), min_val := min (f (Real.pi / 2)) (f (-Real.pi / 2)))
  (max_val + min_val) = -1 / 4 :=
by
  sorry

end sum_of_extrema_of_f_l430_430971


namespace sqrt_mul_l430_430177

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l430_430177


namespace series_sum_correct_l430_430137

-- Define the term for nth element in the series
def term (n : ℕ) : ℚ := 1 / ((2 * n - 1) * (2 * n + 1))

-- Define the summation of the series from n = 1 to n = 200
def series_sum (n : ℕ) : ℚ := ∑ k in Finset.range n, term (k + 1)

-- Theorem: Sum of the series from n=1 to n=200 equals 200/401
theorem series_sum_correct : series_sum 200 = 200 / 401 := by
  sorry

end series_sum_correct_l430_430137


namespace correct_calculation_l430_430679

-- Definitions for conditions
def cond_A (x y : ℝ) : Prop := 3 * x + 4 * y = 7 * x * y
def cond_B (x : ℝ) : Prop := 5 * x - 2 * x = 3 * x ^ 2
def cond_C (y : ℝ) : Prop := 7 * y ^ 2 - 5 * y ^ 2 = 2
def cond_D (a b : ℝ) : Prop := 6 * a ^ 2 * b - b * a ^ 2 = 5 * a ^ 2 * b

-- Proof statement using conditions
theorem correct_calculation (a b : ℝ) : cond_D a b :=
by
  unfold cond_D
  sorry

end correct_calculation_l430_430679


namespace roots_difference_is_one_l430_430953

noncomputable def quadratic_eq (p : ℝ) :=
  ∃ (α β : ℝ), (α ≠ β) ∧ (α - β = 1) ∧ (α ^ 2 - p * α + (p ^ 2 - 1) / 4 = 0) ∧ (β ^ 2 - p * β + (p ^ 2 - 1) / 4 = 0)

theorem roots_difference_is_one (p : ℝ) : quadratic_eq p :=
  sorry

end roots_difference_is_one_l430_430953


namespace find_b_l430_430742

theorem find_b
  (a b c d : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_period : ∀ x, a * sin (b * x + c) + d = a * sin (b * (x - 2 * π / 3) + c) + d)
  (h_cycles : (5 * π / 2) - (π / 2) = 2 * π) :
  b = 3 :=
sorry

end find_b_l430_430742


namespace plane_intersection_properties_l430_430821

open_locale classical

noncomputable theory

variables (α β : Type) [plane α] [plane β] (l m : Type) [line l] [line m] (P : point)
variables [h1 : line_intersects_plane α β l]
variables [h2 : line_in_plane α m]
variables [h3 : line_intersects l m P]

theorem plane_intersection_properties :
  (∃ k : Type, [line k] ∧ line_in_plane β k ∧ line_perpendicular_to_line m k)
  ∧ ¬(∃ k : Type, [line k] ∧ line_in_plane β k ∧ line_parallel_to_line m k) :=
sorry

end plane_intersection_properties_l430_430821


namespace leif_apples_l430_430495

-- Definitions based on conditions
def oranges : ℕ := 24
def apples (oranges apples_diff : ℕ) := oranges - apples_diff

-- Theorem stating the problem to prove
theorem leif_apples (oranges apples_diff : ℕ) (h1 : oranges = 24) (h2 : apples_diff = 10) : apples oranges apples_diff = 14 :=
by
  -- Using the definition of apples and given conditions, prove the number of apples
  rw [h1, h2]
  -- Calculating the number of apples
  show 24 - 10 = 14
  rfl

end leif_apples_l430_430495


namespace student_groups_l430_430544

theorem student_groups (N : ℕ) :
  (∃ (n : ℕ), n = 13 ∧ ∃ (m : ℕ), m ∈ {12, 14} ∧ N = 4 * 13 + 2 * m) → (N = 76 ∨ N = 80) :=
by
  intro h
  obtain ⟨n, hn, m, hm, hN⟩ := h
  rw [hn, hN]
  cases hm with h12 h14
  case inl =>
    simp [h12]
  case inr =>
    simp [h14]
  sorry

end student_groups_l430_430544


namespace sqrt_mul_eq_6_l430_430158

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l430_430158


namespace proof_problem_l430_430376

def number := 432

theorem proof_problem (y : ℕ) (n : ℕ) (h1 : y = 36) (h2 : 6^5 * 2 / n = y) : n = number :=
by 
  -- proof steps would go here
  sorry

end proof_problem_l430_430376


namespace square_grid_nodes_count_l430_430872

theorem square_grid_nodes_count :
  let vertices := {(0, 0), (0, 63), (63, 63), (63, 0)}
  let grid_nodes := {(x, y) | x ∈ (1..62) ∧ y ∈ (1..62)}
  let lines := {(x, x) | x ∈ (1..62)} ∪ {(x, 63 - x) | x ∈ (1..62)}
in
  (∃ p1 p2 ∈ grid_nodes, (p1 ∈ lines ∨ p2 ∈ lines) ∧
    p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2) = 453902 :=
begin
  sorry
end

end square_grid_nodes_count_l430_430872


namespace complex_number_first_quadrant_l430_430421

theorem complex_number_first_quadrant (z : ℂ) (h : z = complex.mk (sqrt 3) 2) : 
  ∃ x y : ℝ, z = complex.mk x y ∧ 0 < x ∧ 0 < y :=
by sorry

end complex_number_first_quadrant_l430_430421


namespace sum_of_odd_terms_l430_430109

theorem sum_of_odd_terms (a : ℕ → ℕ) (h1 : ∀ n, a (n + 1) = a n + 1) (h2 : (Finset.range 2500).sum (λ n, a n) = 7000) : 
  (Finset.range 1250).sum (λ n, a (2 * n)) = 2875 := 
sorry

end sum_of_odd_terms_l430_430109


namespace read_6005_l430_430696

theorem read_6005 : (read_number 6005 = "six thousand zero zero five") = False :=
sorry

end read_6005_l430_430696


namespace sqrt_mul_simp_l430_430208

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l430_430208


namespace cos_minus_sin_eq_l430_430388

open Real

theorem cos_minus_sin_eq : 
  ∀ (θ : ℝ), sin θ + cos θ = 4 / 3 ∧ (π / 4 < θ ∧ θ < π / 2) → cos θ - sin θ = -√2 / 3 :=
by
  sorry

end cos_minus_sin_eq_l430_430388


namespace sqrt_mul_eq_6_l430_430167

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l430_430167


namespace price_of_bread_l430_430518

-- Define the initial conditions
def cash_register_cost : ℕ := 1040
def loaves_per_day : ℕ := 40
def cakes_per_day : ℕ := 6
def price_per_cake : ℕ := 12
def daily_rent : ℕ := 20
def daily_electricity : ℕ := 2
def days_to_payoff : ℕ := 8

-- Main theorem to prove
theorem price_of_bread (P : ℝ) 
  (H : 8 * ((40 * P + 6 * 12) - (20 + 2)) = 1040) : 
  P = 2 :=
begin
  sorry
end

end price_of_bread_l430_430518


namespace sqrt_mult_l430_430188

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l430_430188


namespace Abigail_typing_speed_l430_430727

variable (W : ℕ) -- The number of words Abigail can type in half an hour

-- Conditions
variable (writtenWords : ℕ) (totalWords : ℕ)
variable (remainingTimeMinutes : ℕ) (remainingHalfHours : ℕ)

-- Values given in the problem
axiom writtenWords_value : writtenWords = 200
axiom totalWords_value : totalWords = 1000
axiom remainingTimeMinutes_value : remainingTimeMinutes = 80
axiom remainingHalfHours_value : remainingHalfHours = 8 / 3

-- Derived condition (words she needs to finish)
axiom wordsToFinish : totalWords - writtenWords = 800

-- Main statement to prove
theorem Abigail_typing_speed : W = 300 :=
by
  have h1 : (remainingHalfHours_value : ℚ) * (W : ℚ) = (wordsToFinish : ℚ),
  {
    rw [remainingHalfHours_value, wordsToFinish],
    norm_num,
  }
  have h2 : (8 / 3 : ℚ) * (W : ℚ) = 800 := h1
  have h3 : W = 800 * (3 / 8 : ℚ), from (div_eq_iff_mul_eq (by norm_num : (8 : ℚ) ≠ 0)).mpr (by norm_num : 8 / 3 * 300 = 800)
  exact congr_arg _ (eq.symm h3),
sorry

end Abigail_typing_speed_l430_430727


namespace angle_EBC_cyclic_quadrilateral_l430_430804

theorem angle_EBC_cyclic_quadrilateral 
  (ABCD_cyclic : CyclicQuadrilateral A B C D)
  (AC_diagonal : Diagonal A C)
  (BE_extension : ExtendedBeyond B E' A B)
  (angle_BAD : ∠BAD = 85)
  (angle_ADC : ∠ADC = 72) :
  ∠E'BC = 72 :=
by
  sorry

end angle_EBC_cyclic_quadrilateral_l430_430804


namespace albert_earnings_increase_percent_l430_430453

theorem albert_earnings_increase_percent (
  E : ℝ, 
  P : ℝ, 
  h1 : 1.27 * E = 567, 
  h2 : E + P * E = 562.54
) : P = 0.26 :=
by
  sorry

end albert_earnings_increase_percent_l430_430453


namespace ten_faucets_fill_50_gallons_l430_430783

-- Define the conditions
def five_faucets_fill_rate : ℝ := 200 / 15  -- gallons per minute

-- The theorem to prove
theorem ten_faucets_fill_50_gallons :
  (2 * five_faucets_fill_rate) * (1 / 4) = 50 / (112.5 / 60) :=
by
  sorry

end ten_faucets_fill_50_gallons_l430_430783


namespace sqrt3_mul_sqrt12_eq_6_l430_430225

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l430_430225


namespace triangle_area_specific_l430_430360

noncomputable def vector2_area_formula (u v : ℝ × ℝ) : ℝ :=
|u.1 * v.2 - u.2 * v.1|

noncomputable def triangle_area (u v : ℝ × ℝ) : ℝ :=
(vector2_area_formula u v) / 2

theorem triangle_area_specific :
  let A := (1, 3)
  let B := (5, -1)
  let C := (9, 4)
  let u := (1 - 9, 3 - 4)
  let v := (5 - 9, -1 - 4)
  triangle_area u v = 18 := 
by sorry

end triangle_area_specific_l430_430360


namespace student_groups_l430_430541

theorem student_groups (N : ℕ) :
  (∃ (n : ℕ), n = 13 ∧ ∃ (m : ℕ), m ∈ {12, 14} ∧ N = 4 * 13 + 2 * m) → (N = 76 ∨ N = 80) :=
by
  intro h
  obtain ⟨n, hn, m, hm, hN⟩ := h
  rw [hn, hN]
  cases hm with h12 h14
  case inl =>
    simp [h12]
  case inr =>
    simp [h14]
  sorry

end student_groups_l430_430541


namespace forty_percent_of_number_is_240_l430_430931

-- Define the conditions as assumptions in Lean
variable (N : ℝ)
variable (h1 : (1/4) * (1/3) * (2/5) * N = 20)

-- Prove that 40% of the number N is 240
theorem forty_percent_of_number_is_240 (h1: (1/4) * (1/3) * (2/5) * N = 20) : 0.40 * N = 240 :=
  sorry

end forty_percent_of_number_is_240_l430_430931


namespace sqrt_mult_eq_six_l430_430327

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l430_430327


namespace part_a_part_b_l430_430907

variable (f : ℝ → ℝ) [LipschitzWith 1 (λ x, max (f x) 0) id] -- Ensure f is (1-)Lipschitz and non-negative

-- Part (a)
theorem part_a (hf : ∀ x, filter.tendsto (λ n, f (x + n)) filter.at_top filter.at_top) :
  filter.tendsto f filter.at_top filter.at_top :=
by 
  sorry

-- Part (b)
theorem part_b (α : ℝ) (hα : 0 ≤ α) (hf : ∀ x, filter.tendsto (λ n, f (x + n)) filter.at_top (nhds α)) :
  filter.tendsto f filter.at_top (nhds α) :=
by 
  sorry

end part_a_part_b_l430_430907


namespace monotonic_decreasing_interval_of_f_l430_430629

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - Real.log x

theorem monotonic_decreasing_interval_of_f :
  ∀ x : ℝ, 0 < x ∧ x ≤ 1 → (f '(x)) < 0 :=
sorry

end monotonic_decreasing_interval_of_f_l430_430629


namespace triangle_inequality_l430_430017

theorem triangle_inequality (a : ℝ) :
  (3/2 < a) ∧ (a < 5) ↔ ((4 * a + 1 - (3 * a - 1) < 12 - a) ∧ (4 * a + 1 + (3 * a - 1) > 12 - a)) := 
by 
  sorry

end triangle_inequality_l430_430017


namespace sqrt3_mul_sqrt12_eq_6_l430_430227

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l430_430227


namespace sqrt_mult_eq_six_l430_430320

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l430_430320


namespace sqrt3_mul_sqrt12_eq_6_l430_430222

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l430_430222


namespace angelas_insects_l430_430733

variable (DeanInsects : ℕ) (JacobInsects : ℕ) (AngelaInsects : ℕ)

theorem angelas_insects
  (h1 : DeanInsects = 30)
  (h2 : JacobInsects = 5 * DeanInsects)
  (h3 : AngelaInsects = JacobInsects / 2):
  AngelaInsects = 75 := 
by
  sorry

end angelas_insects_l430_430733


namespace remaining_water_l430_430905

theorem remaining_water (initial_water : ℚ) (used_water : ℚ) (remaining_water : ℚ) 
  (h1 : initial_water = 3) (h2 : used_water = 5/4) : remaining_water = 7/4 :=
by
  -- The proof would go here, but we are skipping it as per the instructions.
  sorry

end remaining_water_l430_430905


namespace find_symmetric_point_l430_430969

def slope_angle (l : ℝ → ℝ → Prop) (θ : ℝ) := ∃ m, m = Real.tan θ ∧ ∀ x y, l x y ↔ y = m * (x - 1) + 1
def passes_through (l : ℝ → ℝ → Prop) (P : ℝ × ℝ) := l P.fst P.snd
def symmetric_point (A A' : ℝ × ℝ) (l : ℝ → ℝ → Prop) := 
  (A'.snd - A.snd = A'.fst - A.fst) ∧ 
  ((A'.fst + A.fst) / 2 + (A'.snd + A.snd) / 2 - 2 = 0)

theorem find_symmetric_point :
  ∃ l : ℝ → ℝ → Prop, 
    slope_angle l (135 : ℝ) ∧ 
    passes_through l (1, 1) ∧ 
    (∀ x y, l x y ↔ x + y = 2) ∧ 
    symmetric_point (3, 4) (-2, -1) l :=
by sorry

end find_symmetric_point_l430_430969


namespace radiusInscribedCircle_l430_430893

variable (A B C D : Type)
variable [InnerProductSpace ℝ A]
variable [InnerProductSpace ℝ B]
variable [InnerProductSpace ℝ C]

/-- Given a right triangle ABC with a right angle at C -/
def rightTriangle (A B C : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] : Prop :=
  ∃ (h : A), angle B C A = π / 2

/-- Given the radii of the inscribed circles of subtriangles ACD and BCD -/
def radiiInscribed (r_ACD : ℝ) (r_BCD : ℝ) : Prop :=
  r_ACD = 6 ∧ r_BCD = 8

/-- Prove the radius of the inscribed circle in triangle ABC is 14 -/
theorem radiusInscribedCircle (ABC : Type) [InnerProductSpace ℝ ABC]
  (right_triangle : rightTriangle A B C)
  (radii : radiiInscribed 6 8) :
  let r_ABC := 14 in
  r_ABC = 14 := sorry

end radiusInscribedCircle_l430_430893


namespace sqrt_mul_simp_l430_430204

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l430_430204


namespace rearrange_ways_l430_430530

universe u

variables {α : Type u} [DecidableEq α]

def seats := Fin₇ → α

def valid_arrangements (seat_change : seats → Prop) :=
  ∑ᵢ, (seat_change i : seats)

def initial_seats : Fin₇ := sorry

lemma no_adjacent_or_same_seat (seat_change : seats → Prop) : valid_arrangements seat_change = 38 :=
sorry

-- Formalization in Lean
theorem rearrange_ways : ∃ seat_change, valid_arrangements seat_change = 38 :=
sorry

end rearrange_ways_l430_430530


namespace spider_total_distance_l430_430112

theorem spider_total_distance : 
  ∀ (pos1 pos2 pos3 : ℝ), pos1 = 3 → pos2 = -1 → pos3 = 8.5 → 
  |pos2 - pos1| + |pos3 - pos2| = 13.5 := 
by 
  intros pos1 pos2 pos3 hpos1 hpos2 hpos3 
  sorry

end spider_total_distance_l430_430112


namespace sqrt_mult_eq_six_l430_430323

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l430_430323


namespace triangle_is_obtuse_l430_430464

-- Defining the problem and the conditions
variable (A B C : ℝ)
variable (h : A + B + C = π)
variable (h1 : cot A * cot B > 1)

-- The theorem stating the answer
theorem triangle_is_obtuse (A B C : ℝ) (h : A + B + C = π) (h1 : cot A * cot B > 1) : 
  π / 2 < C :=
sorry

end triangle_is_obtuse_l430_430464


namespace complex_expression_evaluation_l430_430747

theorem complex_expression_evaluation :
  (-1: ℤ) ^ 2019 + (π - 3.14) ^ 0 - real.sqrt 16 + 2 * real.sin (real.pi / 6) = -3 :=
by
  sorry

end complex_expression_evaluation_l430_430747


namespace sqrt_mult_l430_430186

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l430_430186


namespace identify_incorrect_props_l430_430424

def prop1 (p q : Prop) : Prop := (p ∨ q) → (p ∧ q)
def prop2 (a b : ℝ) : Prop := ¬((a > b) → (2^a > 2^b - 1)) = (a ≤ b → 2^a ≤ 2^b - 1)
def prop3 : Prop := ¬(∀ x : ℝ, x^2 + x ≥ 1) = ∃ x : ℝ, x^2 + x ≤ 1
def prop4 (x : ℝ) : Prop := (x > 1) → (x > 0) ∧ ¬((x > 0) → (x > 1))

theorem identify_incorrect_props : prop1 = false ∧ prop3 = false :=
by
  sorry

end identify_incorrect_props_l430_430424


namespace carnival_ring_toss_l430_430024

theorem carnival_ring_toss (total_amount : ℕ) (days : ℕ) (amount_per_day : ℕ) 
  (h1 : total_amount = 420) 
  (h2 : days = 3) 
  (h3 : total_amount = days * amount_per_day) : amount_per_day = 140 :=
by
  sorry

end carnival_ring_toss_l430_430024


namespace construct_polygon_odd_sides_l430_430038

theorem construct_polygon_odd_sides (n : ℕ) (h : n > 0) 
  (B : Fin (2 * n - 1) → ℝ × ℝ) :
  ∃ A : Fin (2 * n - 1) → ℝ × ℝ,
  ∀ i : Fin (2 * n - 1),
  B i = ((A i).1 + (A (⟨i.val + 1 % (2 * n - 1), sorry⟩)).1) / 2,
  (A i).2 + (A (⟨i.val + 1 % (2 * n - 1), sorry⟩)).2 / 2 :=
begin
  sorry
end

end construct_polygon_odd_sides_l430_430038


namespace complex_expression_l430_430800

theorem complex_expression (z : ℂ) (h : z = (i + 1) / (i - 1)) : z^2 + z + 1 = -i := 
by 
  sorry

end complex_expression_l430_430800


namespace number_of_students_l430_430974

variable S : ℕ

-- One half of the students are boys
def boys := S / 2

-- One fifth of the boys are under 6 feet tall
def boys_under_6_feet := boys / 5

-- There are 10 boys who are under 6 feet tall
def condition := boys_under_6_feet = 10

theorem number_of_students (h : condition) : S = 100 := by
  sorry

end number_of_students_l430_430974


namespace tailor_time_calculation_l430_430722

-- Define the basic quantities and their relationships
def time_ratio_shirt : ℕ := 1
def time_ratio_pants : ℕ := 2
def time_ratio_jacket : ℕ := 3

-- Given conditions
def shirts_made := 2
def pants_made := 3
def jackets_made := 4
def total_time_initial : ℝ := 10

-- Unknown time per shirt
noncomputable def time_per_shirt := total_time_initial / (shirts_made * time_ratio_shirt 
  + pants_made * time_ratio_pants 
  + jackets_made * time_ratio_jacket)

-- Future quantities
def future_shirts := 14
def future_pants := 10
def future_jackets := 2

-- Calculate the future total time required
noncomputable def future_time_required := (future_shirts * time_ratio_shirt 
  + future_pants * time_ratio_pants 
  + future_jackets * time_ratio_jacket) * time_per_shirt

-- State the theorem to prove
theorem tailor_time_calculation : future_time_required = 20 := by
  sorry

end tailor_time_calculation_l430_430722


namespace minimum_value_l430_430815

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

-- Condition: a, b, c are unit vectors and a · b = 0
def unit_vector (v : V) : Prop := ∥v∥ = 1
def orthogonal (u v : V) : Prop := ⟪u, v⟫ = 0

theorem minimum_value (ha : unit_vector a) (hb : unit_vector b) (hc : unit_vector c) (h_ab : orthogonal a b) : 
  (inner (a - c) (b - c)) = 1 - real.sqrt 2 :=
sorry

end minimum_value_l430_430815


namespace sqrt_mul_simplify_l430_430275

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l430_430275


namespace sum_of_solutions_equation_l430_430668

theorem sum_of_solutions_equation : 
  (∑ x in {x : ℝ | (6 * x) / 18 = 3 / x}.to_finset, x) = 0 := 
by
  sorry

end sum_of_solutions_equation_l430_430668


namespace find_x_l430_430856

theorem find_x (x y : ℚ) (h1 : 3 * x - 2 * y = 7) (h2 : x + 3 * y = 8) : x = 37 / 11 := by
  sorry

end find_x_l430_430856


namespace math_problem_l430_430365

theorem math_problem (n : ℤ) : (37 ∣ 2 * 6^(4 * n + 3) + 2^n) → n ≡ 10 [MOD 36] := by
  sorry

end math_problem_l430_430365


namespace retailer_percentage_profit_l430_430097

-- Definitions
def wholesale_price : ℝ := 90
def retail_price : ℝ := 120
def discount_rate : ℝ := 0.10

-- The selling price after discount
def selling_price : ℝ := retail_price * (1 - discount_rate)

-- The profit made by the retailer
def profit : ℝ := selling_price - wholesale_price

-- The percentage profit
def percentage_profit : ℝ := (profit / wholesale_price) * 100

-- The proof statement
theorem retailer_percentage_profit : percentage_profit = 20 := by
  sorry

end retailer_percentage_profit_l430_430097


namespace correct_statements_l430_430828

/-- Given the curve C: y = sqrt(4 - 2x^2 / 3), points A (-sqrt(6), 0) and B (sqrt(6), 0).
Let P be a point on C different from A and B. Line AP intersects the line x = 2sqrt(6) 
at point M, and line BP intersects the line x = 2sqrt(6) at point N. Then, the statements 
A, B, and D are correct. --/
theorem correct_statements (x y : ℝ) :
  let C := ∀ (x y : ℝ), y = sqrt(4 - 2 * x^2 / 3) in
  let A := (-sqrt(6), 0) in
  let B := (sqrt(6), 0) in
  let P := ∀ (x y : ℝ), y = sqrt(4 - 2 * x^2 / 3) ∧ (x ≠ -sqrt(6) ∧ x ≠ sqrt(6)) in
  ∀ (M N : ℝ × ℝ),
  (let Mx := 2 * sqrt(6) in 
  ∃ k : ℝ, M = (Mx, 3 * sqrt(6) * k) ∧ N = (Mx, -2 * sqrt(6) / (3 * k)) 
  ∧ k = sqrt(6) 
  → ((∀ P : ℝ × ℝ, (x ∈ Icc (sqrt(6) - sqrt(2)) (sqrt(6) + sqrt(2))) ∧ ((∃ (dx : ℝ), dx = sqrt(2)) 
  	→ true)
    ∧ (∃ F1 F2 : ℝ × ℝ, (P : ℝ × ℝ) ∧ (F1 := (sqrt(2), 0)) ∧ (F2 := (-sqrt(2), 0)) 
    ∧ (dist P F1 + dist P F2 = 2*sqrt(2)))
    ∧ (|dist M N| = 56 / 3)))) :=
by sorry

end correct_statements_l430_430828


namespace sqrt_mul_sqrt_eq_six_l430_430292

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l430_430292


namespace absolute_value_inequality_l430_430607

theorem absolute_value_inequality (x : ℝ) : (|x + 1| > 3) ↔ (x > 2 ∨ x < -4) :=
by
  sorry

end absolute_value_inequality_l430_430607


namespace sqrt_mult_l430_430192

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l430_430192


namespace complex_division_example_l430_430012

theorem complex_division_example : (3 + Complex.i) / (1 + Complex.i) = 2 - Complex.i := by
  sorry

end complex_division_example_l430_430012


namespace sqrt_mul_eq_l430_430306

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l430_430306


namespace card_drawing_problem_solution_l430_430645

open Finset

-- Define some constants based on the problem conditions
def total_cards := 16
def each_color_cards := 4
def cards_to_draw := 3

-- Define the primary condition of the problem
def conditions_met (selected_cards : Finset (Fin total_cards)) : Prop :=
  selected_cards.card = cards_to_draw ∧
  (∀ color_set : Finset (Fin total_cards), color_set.card = each_color_cards → selected_cards ∩ color_set ≠ selected_cards) ∧
  (selected_cards.filter (λ c, c < each_color_cards)).card ≤ 1

-- We aim to prove that the number of ways to draw the cards given the conditions is 472
theorem card_drawing_problem_solution :
  ∑ (s : Finset (Fin total_cards)) in (powerset (range total_cards)), if conditions_met s then 1 else 0 = 472 := 
by sorry

end card_drawing_problem_solution_l430_430645


namespace area_of_ATHEM_l430_430496

-- Define the problem conditions
structure Pentagon (A T H E M : Type) where
  AT : ℝ
  TH : ℝ
  MA : ℝ
  HE : ℝ
  EM : ℝ
  angle_THE : ℝ
  angle_EMA : ℝ

-- Instantiate the given problem
def ATHEM : Pentagon Point := {
  AT := 14,
  TH := 20,
  MA := 20,
  HE := 15,
  EM := 15,
  angle_THE := 90,
  angle_EMA := 90
}

-- Lean theorem to prove the area equivalence
theorem area_of_ATHEM : (area (ATHEM)) = 570.625 := by
  sorry

end area_of_ATHEM_l430_430496


namespace equilateral_hyperbola_real_axis_length_l430_430950

theorem equilateral_hyperbola_real_axis_length 
  (λ : ℝ)
  (y : ℝ)
  (h_hyperbola : ∀ x y : ℝ, x^2 - y^2 = λ ↔ (x^2 - y^2 = 4))
  (h_parabola : ∀ y : ℝ, y^2 = 16 * (-4))
  (h_directrix : ∀ (A B : ℝ), A = -4 ∧ B = -4 ∧ y > 0 ∧ |A - B| = 4 * real.sqrt 3) :
  2*real.sqrt(λ) = 4 :=
by {
  have hyp_obs := λ (x y : ℝ), h_hyperbola x y,
  sorry
}

end equilateral_hyperbola_real_axis_length_l430_430950


namespace at_least_one_good_l430_430644

theorem at_least_one_good (total_products : ℕ) (defective_products : ℕ) (selected_products : ℕ) :
  total_products = 12 → defective_products = 2 → selected_products = 3 →
  ∀ (s : finset ℕ), s.card = selected_products → ∀ (f : ℕ → Prop), 
  (∀ i, i ∈ s → (i ≤ total_products ∧ (i > defective_products → ¬f i))) → 
  (∃ i ∈ s, ¬f i) :=
by
  intros h_total h_defective h_selected hs_card hf H,
  sorry

end at_least_one_good_l430_430644


namespace sqrt3_mul_sqrt12_eq_6_l430_430258

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l430_430258


namespace can_reach_silence_l430_430925

inductive State
| ZZ : State  -- Both singing
| ZH : State  -- Singing and quiet
| HZ : State  -- Quiet and singing
| HH : State  -- Both quiet

def action (s : State) (play_organ : Bool) (window_open : Bool) : State :=
  match s with
  | State.ZZ => if window_open then State.ZH else State.ZZ
  | State.ZH => if play_organ then (if window_open then State.HH else State.HH) else State.ZH
  | State.HZ => if window_open then State.HH else State.HH
  | State.HH => State.HH

theorem can_reach_silence (s : State) : ∃ seq : List (Bool × Bool), foldl (λ st (p : Bool × Bool), action st p.1 p.2) s seq = State.HH :=
  sorry

end can_reach_silence_l430_430925


namespace total_students_in_groups_l430_430538

theorem total_students_in_groups {N : ℕ} (h1 : ∃ g : ℕ → ℕ, (∀ i j, g i = 13 ∨ g j = 12 ∨ g j = 14) ∧ (∑ i in finset.range 6, g i) = N) : 
  N = 76 ∨ N = 80 :=
sorry

end total_students_in_groups_l430_430538


namespace total_amount_paid_l430_430032

theorem total_amount_paid (cost_of_manicure : ℝ) (tip_percentage : ℝ) (total : ℝ) 
  (h1 : cost_of_manicure = 30) (h2 : tip_percentage = 0.3) (h3 : total = cost_of_manicure + cost_of_manicure * tip_percentage) : 
  total = 39 :=
by
  sorry

end total_amount_paid_l430_430032


namespace no_all_same_color_l430_430522

def chameleons_initial_counts (c b m : ℕ) : Prop :=
  c = 13 ∧ b = 15 ∧ m = 17

def chameleon_interaction (c b m : ℕ) : Prop :=
  (∃ c' b' m', c' + b' + m' = c + b + m ∧ 
  ((c' = c - 1 ∧ b' = b - 1 ∧ m' = m + 2) ∨
   (c' = c - 1 ∧ b' = b + 2 ∧ m' = m - 1) ∨
   (c' = c + 2 ∧ b' = b - 1 ∧ m' = m - 1)))

theorem no_all_same_color (c b m : ℕ) (h1 : chameleons_initial_counts c b m) : 
  ¬ (∃ x, c = x ∧ b = x ∧ m = x) := 
sorry

end no_all_same_color_l430_430522


namespace calc_expr1_calc_expr2_l430_430745

-- Definitions of the complex expressions for the questions
noncomputable def expr1 := (- (1/2) + (real.sqrt 3 / 2) * complex.I) * (real.sqrt 3 / 2 + (1/2) * complex.I)
noncomputable def expr2 := ((1 + 2 * complex.I) ^ 2 + 3 * (1 - complex.I)) / (2 + complex.I)

-- Theorem stating the expected results for the calculations
theorem calc_expr1 : expr1 = - real.sqrt 3 / 2 + (1/2) * complex.I := 
  sorry

theorem calc_expr2 : expr2 = 1 / 5 + 2 / 5 * complex.I := 
  sorry

end calc_expr1_calc_expr2_l430_430745


namespace value_of_expression_l430_430052

theorem value_of_expression : 1 + 2 / (1 + 2 / (2 * 2)) = 7 / 3 := 
by 
  -- proof to be filled in
  sorry

end value_of_expression_l430_430052


namespace roots_reciprocal_l430_430937

theorem roots_reciprocal (a b c r s : ℝ) (h_eqn : a ≠ 0) (h_roots : a * r^2 + b * r + c = 0 ∧ a * s^2 + b * s + c = 0) (h_cond : b^2 = 4 * a * c) : r * s = 1 :=
by
  -- Proof goes here
  sorry

end roots_reciprocal_l430_430937


namespace line_through_points_eq_l430_430031

def Point := (ℝ × ℝ)

def line_equation (m: ℝ) (x1 y1: ℝ): string :=
  "y - " ++ toString y1 ++ " = " ++ toString m ++ " * (x - " ++ toString x1 ++ ")"

theorem line_through_points_eq :
  ∃ (m: ℝ), m = (2 - 0.5) / (1.5 - 1) ∧
  ∀ (x1 y1 x2 y2: ℝ), x1 = 1 ∧ y1 = 0.5 ∧ x2 = 1.5 ∧ y2 = 2 →
  "6x - 2y = 5" = line_equation m x1 y1 :=
by
  sorry

end line_through_points_eq_l430_430031


namespace Cindy_correct_answer_l430_430332

theorem Cindy_correct_answer :
  (∃ x : ℕ, (x - 12) / 4 = 72 ∧ (x - 5) / 9 = 33) :=
begin
  sorry
end

end Cindy_correct_answer_l430_430332


namespace BE_is_sqrt_10_l430_430984

-- Defining the context of the parallelogram and points
variables (A B C D E F G : Point)
variables (parallelogram_ABCD : Parallelogram A B C D)
variables (intersect_E : E ∈ Line(B, D))
variables (intersect_F : F ∈ Line(C, D))
variables (intersect_G : G ∈ Line(B, C))
variables (ratio_FG_FE : Ratio FG FE = 9)
variables (length_ED : Distance(E, D) = 1)

-- The goal is to find the length of BE
def find_BE : Real :=
  let BE := sqrt(10) in BE

theorem BE_is_sqrt_10
  (parallelogram_ABCD : Parallelogram A B C D)
  (intersect_E : E ∈ Line(B, D))
  (intersect_F : F ∈ Line(C, D))
  (intersect_G : G ∈ Line(B, C))
  (ratio_FG_FE : Ratio FG FE = 9)
  (length_ED : Distance(E, D) = 1) : 
  Distance(B, E) = sqrt(10) :=
by
  sorry

end BE_is_sqrt_10_l430_430984


namespace first_expression_evaluation_second_expression_evaluation_l430_430766

noncomputable def first_expression : ℝ :=
  64 ^ (1 / 3) - (-2 / 3) ^ 0 + 125 ^ (1 / 3) + log 10 2 + log 10 50 + 2 ^ (1 + log 2 3)

theorem first_expression_evaluation : first_expression = 16 :=
by
  sorry

noncomputable def second_expression : ℝ :=
  log 10 14 - 2 * log 10 (7 / 3) + log 10 7 - log 10 18

theorem second_expression_evaluation : second_expression = 0 :=
by
  sorry

end first_expression_evaluation_second_expression_evaluation_l430_430766


namespace bulbs_problem_l430_430875

theorem bulbs_problem (n : ℕ) (switch : ℕ → bool) (bulb : ℕ → bool) :
  (∀ i, i ≤ 10 → switch i = false) ∧ (switch 11 = true) → n = 1024 := 
sorry

end bulbs_problem_l430_430875


namespace sqrt_mult_eq_six_l430_430329

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l430_430329


namespace sqrt3_mul_sqrt12_eq_6_l430_430261

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l430_430261


namespace transform_tetrahedron_part_l430_430719

structure Tetrahedron (V : Type) :=
(vertices : V)
(dividedInto24Parts : Prop)
(symmetryPlanes : Prop)

theorem transform_tetrahedron_part (T : Tetrahedron) :
  (∀ part1 part2 : Fin 24, ∃ (r1 r2 r3 : symmetryPlanes), 
  reflect reflect(r1, part1) ∘ reflect(r2, part2) = part2) := sorry

end transform_tetrahedron_part_l430_430719


namespace sqrt3_mul_sqrt12_eq_6_l430_430265

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l430_430265


namespace sqrt_mul_sqrt_eq_six_l430_430294

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l430_430294


namespace triangle_AB_equals_DE_l430_430504

theorem triangle_AB_equals_DE (A B C D E : Type) [is_triangle A B C]
  (angle_ACB : ∠ C = 60)
  (AC_lt_BC : |AC| < |BC|)
  (BD_eq_AC : |BD| = |AC|)
  (AC_eq_CE : |AC| = |CE|):
  |AB| = |DE| := 
begin
  sorry
end

end triangle_AB_equals_DE_l430_430504


namespace geometric_figures_l430_430708

def max_abs (a b : ℝ) := max (|a|) (|b|)

theorem geometric_figures (x y r : ℝ) : 
  (|x| + |y| ≤ r ∧ r ≤ 3 * max_abs x y) ↔ 
  ("diamond within circle within hexagon") := 
by 
  sorry

end geometric_figures_l430_430708


namespace number_of_people_l430_430088

theorem number_of_people (total_dining_bill : ℝ) (tip_percent : ℝ) (shared_amount_per_person : ℝ)
  (h1 : total_dining_bill = 139)
  (h2 : tip_percent = 0.10)
  (h3 : shared_amount_per_person = 21.842857142857145) :
  let tip := tip_percent * total_dining_bill in
  let total_paid := total_dining_bill + tip in
  let number_of_people := total_paid / shared_amount_per_person in
  number_of_people ≈ 7 :=
by
  sorry

end number_of_people_l430_430088


namespace sqrt_mul_simp_l430_430215

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l430_430215


namespace total_students_in_groups_l430_430534

theorem total_students_in_groups {N : ℕ} (h1 : ∃ g : ℕ → ℕ, (∀ i j, g i = 13 ∨ g j = 12 ∨ g j = 14) ∧ (∑ i in finset.range 6, g i) = N) : 
  N = 76 ∨ N = 80 :=
sorry

end total_students_in_groups_l430_430534


namespace sqrt_mul_l430_430183

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l430_430183


namespace sqrt_mul_eq_l430_430311

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l430_430311


namespace sup_E_s_eq_E_xi_l430_430911

-- Define non-negative random variable ξ
variable (Ω : Type) [MeasurableSpace Ω] (ξ : Ω → ℝ)
variable [IsProbabilityMeasure (measure_theory.MeasureSpace.measure Ω)]

-- Assume ξ is non-negative
axiom ξ_nonneg : ∀ ω, 0 ≤ ξ ω

-- Define the set S of simple, non-negative functions s such that s ≤ ξ
def S (ξ : Ω → ℝ) : set (Ω → ℝ) :=
  {s | simple_func s ∧ (∀ ω, 0 ≤ s ω) ∧ (∀ ω, s ω ≤ ξ ω)}

-- Define the expected value function E
noncomputable def E (f : Ω → ℝ) : ℝ := measure_theory.Lintegral Ω ℝ (λ x, ennreal.of_real (f x))

-- Lean statement for the proof problem
theorem sup_E_s_eq_E_xi :
  ∀ (ξ : Ω → ℝ), ∀ s ∈ S ξ, \sup (λ s, E s) = E ξ :=
by
  sorry

end sup_E_s_eq_E_xi_l430_430911


namespace lowest_score_is_C_l430_430460

variable (Score : Type) [LinearOrder Score]
variable (A B C : Score)

-- Translate conditions into Lean
variable (h1 : B ≠ max A (max B C) → A = min A (min B C))
variable (h2 : C ≠ min A (min B C) → A = max A (max B C))

-- Define the proof goal
theorem lowest_score_is_C : min A (min B C) =C :=
by
  sorry

end lowest_score_is_C_l430_430460


namespace toms_age_l430_430651

theorem toms_age (T S : ℕ) (h1 : T = 2 * S - 1) (h2 : T + S = 14) : T = 9 :=
sorry

end toms_age_l430_430651


namespace expected_value_xi_l430_430081

-- Defining the probabilistic conditions
def fairCoin : Prob := 0.5

def xi (A1 A2 : Bool) : ℝ :=
  if A1 || A2 then 1 else 0

-- Theorem stating the expected value of random variable xi
theorem expected_value_xi :
  ∃ (ξ : Boolean → Boolean → ℝ), ξ = xi ∧
    (⟨ A1, pA1 ⟩ = ⟨ true, fairCoin ⟩ ∨ ⟨ false, 1 - fairCoin ⟩) ∧
    (⟨ A2, pA2 ⟩ = ⟨ true, fairCoin ⟩ ∨ ⟨ false, 1 - fairCoin ⟩) →
      (E[ξ] = 0.75) := 
sorry

end expected_value_xi_l430_430081


namespace triangle_ABC_problem_l430_430339

-- Defining the function f
def f (x : ℝ) : ℝ := 1 / (1 - x)

theorem triangle_ABC_problem (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) :
  ((x - 1) * f x + f (1 / x) = 1 / (x - 1)) ∧
  (((1 - x) / x) * f (1 / x) + f x = x / (1 - x)) :=
by
  -- Proof is skipped
  sorry

end triangle_ABC_problem_l430_430339


namespace smallest_positive_period_of_f_max_min_value_of_f_in_interval_range_of_m_for_inequality_l430_430426

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x - sqrt 3 * cos (2 * x) + 1

theorem smallest_positive_period_of_f : smallest_period f = π := 
  sorry

theorem max_min_value_of_f_in_interval :
  ∀ x ∈ set.Icc (π / 4) (π / 2), 2 ≤ f x ∧ f x ≤ 3 :=
  sorry

theorem range_of_m_for_inequality {m : ℝ} :
  (∀ x ∈ set.Icc (π / 4) (π / 2), (f x - m) ^ 2 < 4) ↔ 1 < m ∧ m < 4 :=
  sorry

end smallest_positive_period_of_f_max_min_value_of_f_in_interval_range_of_m_for_inequality_l430_430426


namespace smallest_a_l430_430505

theorem smallest_a (a b : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : ∀ x : ℤ, Real.sin (a * x + b) = Real.sin (17 * x)) :
  a = 17 :=
by
  sorry

end smallest_a_l430_430505


namespace correct_equation_l430_430058

theorem correct_equation :
  let A := (sqrt (4 + 1/9) = 2 + 1/3) in
  let B := (sqrt 16 = ± sqrt 4) in
  let C := (sqrt ((-5) ^ 2) = -5) in
  let D := (cbrt ((-2) ^ 3) = -2) in
  ¬A ∧ ¬B ∧ ¬C ∧ D :=
by
  let A := (sqrt (4 + 1/9) = 2 + 1/3)
  let B := (sqrt 16 = ± sqrt 4)
  let C := (sqrt ((-5) ^ 2) = -5)
  let D := (cbrt ((-2) ^ 3) = -2)
  sorry

end correct_equation_l430_430058


namespace sum_from_neg_50_to_75_l430_430671

def sum_of_integers (a b : ℤ) : ℤ :=
  (b * (b + 1)) / 2 - (a * (a - 1)) / 2

theorem sum_from_neg_50_to_75 : sum_of_integers (-50) 75 = 1575 := by
  sorry

end sum_from_neg_50_to_75_l430_430671


namespace count_four_digit_integers_with_product_16_l430_430847

def four_digit_product_eq_16 (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000 ∧ (nat.digits 10 n).prod = 16

theorem count_four_digit_integers_with_product_16 :
  {n : ℕ | four_digit_product_eq_16 n}.to_finset.card = 11 :=
sorry

end count_four_digit_integers_with_product_16_l430_430847


namespace retailer_percentage_profit_l430_430101

def wholesale_price : ℝ := 90
def retail_price : ℝ := 120
def discount_rate : ℝ := 0.10

def selling_price : ℝ := retail_price * (1 - discount_rate)
def profit : ℝ := selling_price - wholesale_price
def percentage_profit : ℝ := (profit / wholesale_price) * 100

theorem retailer_percentage_profit : percentage_profit = 20 :=
by
  sorry

end retailer_percentage_profit_l430_430101


namespace find_line_equation_l430_430089
noncomputable def line_equation (l : ℝ → ℝ → Prop) : Prop :=
    (∀ x y : ℝ, l x y ↔ (2 * x + y - 4 = 0) ∨ (x + y - 3 = 0))

theorem find_line_equation (l : ℝ → ℝ → Prop) :
  (l 1 2) →
  (∃ x1 : ℝ, x1 > 0 ∧ ∃ y1 : ℝ, y1 > 0 ∧ l x1 0 ∧ l 0 y1) ∧
  (∃ x2 : ℝ, x2 < 0 ∧ ∃ y2 : ℝ, y2 > 0 ∧ l x2 0 ∧ l 0 y2) ∧
  (∃ x4 : ℝ, x4 > 0 ∧ ∃ y4 : ℝ, y4 < 0 ∧ l x4 0 ∧ l 0 y4) ∧
  (∃ x_int y_int : ℝ, l x_int 0 ∧ l 0 y_int ∧ x_int + y_int = 6) →
  (line_equation l) :=
by
  sorry

end find_line_equation_l430_430089


namespace find_dc_l430_430503

variables (O A B C H D : Type)
variables [Ring O] [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup H] [AddGroup D]

noncomputable def is_equilateral (A B C : Type) : Prop :=
  A = B ∧ B = C

noncomputable def height_eq_half_base (B H : Type) : Prop :=
  ∃ (A C : Type), H = (A + C) / 2

noncomputable def circumcircle_center (O B : Type) : Prop :=
  BO = 3

noncomputable def height_condition (BH : Type) : Prop :=
  BH = 1.5 ∧ BH = HO

noncomputable def angle_condition (BAC : Type) : Prop :=
  BAC = 30

theorem find_dc (DC : Type) (sqrt : Type) :
  circumcircle_center O B →
  height_condition BH →
  height_eq_half_base B H →
  is_equilateral A B C →
  (angle_condition BAC ∧ BAC = BDC) →
  (DC = 3 - sqrt 6 ∨ DC = 3 + sqrt 6) :=
by sorry

end find_dc_l430_430503


namespace least_positive_integer_modulo_l430_430367

theorem least_positive_integer_modulo :
  ∃ x : ℕ, (x + 7351) % 17 = 3071 % 17 ∧ x > 0 ∧ ∀ y : ℕ, (y + 7351) % 17 = 3071 % 17 → y > 0 → x ≤ y :=
begin
  sorry
end

end least_positive_integer_modulo_l430_430367


namespace average_annual_growth_rate_l430_430922

theorem average_annual_growth_rate (x : ℝ) (h1 : 6.4 * (1 + x)^2 = 8.1) : x = 0.125 :=
by
  -- proof goes here
  sorry

end average_annual_growth_rate_l430_430922


namespace rectangular_prism_diagonal_inequality_l430_430396

variable (a b c l : ℝ)

theorem rectangular_prism_diagonal_inequality (h : l^2 = a^2 + b^2 + c^2) : 
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := sorry

end rectangular_prism_diagonal_inequality_l430_430396


namespace sqrt_mul_simp_l430_430203

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l430_430203


namespace cost_of_adult_ticket_l430_430121

/-- Problem Statement: Given the conditions that adult tickets cost some cents, children's tickets cost 25 cents, 280 persons attended, total receipts were 140 dollars, and 80 children attended. Prove that the cost of an adult ticket is 60 cents. -/
theorem cost_of_adult_ticket (A : ℕ) 
  (children_ticket_cost : ℕ := 25)
  (total_persons : ℕ := 280)
  (total_receipts : ℕ := 140 * 100) -- in cents
  (children_attended : ℕ := 80) :
  ∃ A : ℕ, 
    (let adults_attended := total_persons - children_attended in
     let children_revenue := children_attended * children_ticket_cost in
     let total_revenue := total_receipts in
     let adult_revenue := total_revenue - children_revenue in
     adults_attended * A = adult_revenue) ∧
    A = 60 :=
begin
  sorry,
end

end cost_of_adult_ticket_l430_430121


namespace total_students_76_or_80_l430_430575

theorem total_students_76_or_80 
  (N : ℕ)
  (h1 : ∃ g : ℕ → ℕ, (∑ i in finset.range 6, g i = N) ∧
                     (∃ a b : ℕ, finset.card {i | g i = a} = 4 ∧ 
                                 finset.card {i | g i = b} = 2 ∧ 
                                 (a = 13 ∧ (b = 12 ∨ b = 14))))
  : N = 76 ∨ N = 80 :=
sorry

end total_students_76_or_80_l430_430575


namespace slope_of_line_l430_430344

theorem slope_of_line (x : ℝ) : (2 * x + 1) = 2 :=
by sorry

end slope_of_line_l430_430344


namespace retailer_percentage_profit_l430_430099

-- Definitions
def wholesale_price : ℝ := 90
def retail_price : ℝ := 120
def discount_rate : ℝ := 0.10

-- The selling price after discount
def selling_price : ℝ := retail_price * (1 - discount_rate)

-- The profit made by the retailer
def profit : ℝ := selling_price - wholesale_price

-- The percentage profit
def percentage_profit : ℝ := (profit / wholesale_price) * 100

-- The proof statement
theorem retailer_percentage_profit : percentage_profit = 20 := by
  sorry

end retailer_percentage_profit_l430_430099
