import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_students_solve_all_l476_47636

/-- Given a class of students and a set of problems, we define a solution matrix
    where each entry represents whether a student solved a particular problem. -/
def SolutionMatrix (n m : ℕ) := Fin n → Fin m → Bool

theorem two_students_solve_all
  (n m : ℕ)
  (h_n : n = 15)
  (h_m : m = 6)
  (solutions : SolutionMatrix n m)
  (h_more_than_half : ∀ j : Fin m, (Finset.filter (λ i => solutions i j) Finset.univ).card > n / 2) :
  ∃ i₁ i₂ : Fin n, ∀ j : Fin m, solutions i₁ j = true ∨ solutions i₂ j = true :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_students_solve_all_l476_47636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_loss_challenge_l476_47670

theorem weight_loss_challenge (W : ℝ) (h : W > 0) : 
  let initial_weight := W
  let weight_after_loss := W * (1 - 0.13)
  let final_weight := weight_after_loss * (1 + 0.02)
  let measured_loss_percentage := (initial_weight - final_weight) / initial_weight * 100
  ∃ ε > 0, |measured_loss_percentage - 11.26| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_loss_challenge_l476_47670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_members_in_organization_l476_47641

/-- Represents the number of committees in the organization -/
def num_committees : ℕ := 5

/-- Represents the number of committees each member belongs to -/
def committees_per_member : ℕ := 2

/-- Predicate indicating that a member belongs to two specific committees -/
def member_belongs_to (c₁ c₂ : Fin num_committees) (member : ℕ) : Prop :=
  sorry

/-- Theorem stating the total number of members in the organization -/
theorem total_members_in_organization :
  (∀ (member : ℕ), member < (num_committees.choose committees_per_member) →
    ∃! (c₁ c₂ : Fin num_committees), c₁ ≠ c₂ ∧ member_belongs_to c₁ c₂ member) →
  (∀ (c₁ c₂ : Fin num_committees), c₁ ≠ c₂ →
    ∃! (member : ℕ), member < (num_committees.choose committees_per_member) ∧
      member_belongs_to c₁ c₂ member) →
  (num_committees.choose committees_per_member) = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_members_in_organization_l476_47641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_number_puzzle_l476_47601

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_digit_2 (n : ℕ) : Prop := ∃ d : ℕ, d < 10 ∧ (n / 10 = 2 ∨ n % 10 = 2)

def exactly_three_of_four (p q r s : Prop) : Prop :=
  (p ∧ q ∧ r ∧ ¬s) ∨ (p ∧ q ∧ ¬r ∧ s) ∨ (p ∧ ¬q ∧ r ∧ s) ∨ (¬p ∧ q ∧ r ∧ s)

theorem apartment_number_puzzle :
  ∃! n : ℕ, 
    10 ≤ n ∧ n < 100 ∧
    (exactly_three_of_four
      (is_square n)
      (Odd n)
      (n % 3 = 0)
      (has_digit_2 n)) ∧
    n % 10 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_number_puzzle_l476_47601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_piece_arrangements_l476_47600

/-- Represents a valid arrangement of square pieces -/
structure SquareArrangement where
  layers : ℕ
  pieces_per_layer : List ℕ

/-- Counts the number of valid arrangements for n pieces -/
def count_arrangements (n : ℕ) : ℕ := sorry

/-- Predicate to check if an arrangement is valid -/
def is_valid_arrangement (arr : SquareArrangement) : Prop :=
  arr.layers ≥ 2 ∧
  arr.pieces_per_layer.sum = 9 ∧
  arr.pieces_per_layer.length = arr.layers ∧
  ∀ i, i + 1 < arr.layers → arr.pieces_per_layer[i]! > arr.pieces_per_layer[i+1]!

theorem nine_piece_arrangements :
  count_arrangements 9 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_piece_arrangements_l476_47600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rural_andhra_fdi_relation_l476_47688

/-- Represents the total Foreign Direct Investment (FDI) -/
def X : ℝ → ℝ := id

/-- Represents the FDI in urban Gujarat -/
def U : ℝ → ℝ := id

/-- The proportion of total FDI that goes to Gujarat -/
def gujarat_fdi_ratio : ℝ := 0.30

/-- The proportion of Gujarat's FDI that goes to rural areas -/
def gujarat_rural_ratio : ℝ := 0.20

/-- The proportion of total FDI that goes to Andhra Pradesh -/
def andhra_fdi_ratio : ℝ := 0.20

/-- The proportion of Andhra Pradesh's FDI that goes to rural areas -/
def andhra_rural_ratio : ℝ := 0.50

/-- Theorem stating the relationship between rural Andhra Pradesh FDI and urban Gujarat FDI -/
theorem rural_andhra_fdi_relation (hX : X 0 > 0) (hU : U 0 = (1 - gujarat_rural_ratio) * gujarat_fdi_ratio * X 0) :
  andhra_rural_ratio * andhra_fdi_ratio * X 0 = (5 / 12) * U 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rural_andhra_fdi_relation_l476_47688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_l476_47689

-- Define the ellipse and parabola
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the conditions
theorem ellipse_parabola_intersection (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ (x y : ℝ), 
    ellipse x y a b ∧ 
    parabola x y ∧ 
    x > 0 ∧ y > 0 ∧
    (x + 1)^2 + y^2 = (5/3)^2 →
  (ellipse x y 2 (Real.sqrt 3) ∧
   ∀ (m t : ℝ), 
     (∃ (x₁ y₁ x₂ y₂ : ℝ),
       ellipse x₁ y₁ 2 (Real.sqrt 3) ∧
       ellipse x₂ y₂ 2 (Real.sqrt 3) ∧
       x₁ = m * y₁ + 1 ∧
       x₂ = m * y₂ + 1 ∧
       t * (x₁ - 1) = (1 - t) * (x₂ - 1) ∧
       t * y₁ = (1 - t) * y₂) →
     0 < t ∧ t < 1/4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_l476_47689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_curve_tangent_lines_parabola_l476_47648

noncomputable section

-- Define the curve and parabola
def curve (x : ℝ) : ℝ := x / (2 * x - 1)
def parabola (x : ℝ) : ℝ := x^2

-- Define the point on the curve and the point the tangent line passes through
def point_on_curve : ℝ × ℝ := (1, 1)
def point_through : ℝ × ℝ := (2, 3)

-- Theorem for the tangent line to the curve
theorem tangent_line_curve :
  ∃ (m b : ℝ), ∀ x y : ℝ,
    y = curve x →
    (x = point_on_curve.1 ∧ y = point_on_curve.2) →
    m * x + y + b = 0 ∧
    m = -1 ∧ b = -1 :=
by sorry

-- Theorem for the tangent lines to the parabola
theorem tangent_lines_parabola :
  ∃ (m₁ b₁ m₂ b₂ : ℝ), ∀ x y : ℝ,
    y = parabola x →
    (m₁ * point_through.1 + point_through.2 + b₁ = 0) ∧
    (m₂ * point_through.1 + point_through.2 + b₂ = 0) ∧
    ((m₁ * x + y + b₁ = 0) ∨ (m₂ * x + y + b₂ = 0)) ∧
    m₁ = 2 ∧ b₁ = 1 ∧ m₂ = 6 ∧ b₂ = 9 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_curve_tangent_lines_parabola_l476_47648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_value_l476_47606

def sample : List ℝ := [9, 10, 11]

theorem xy_value (x y : ℝ) : 
  (let full_sample := sample ++ [x, y]
   let average : ℝ := (List.sum full_sample) / 5
   let variance : ℝ := (List.sum (List.map (λ z => (z - average)^2) full_sample)) / 5
   let std_dev : ℝ := Real.sqrt variance
   average = 10 ∧ 
   std_dev = Real.sqrt 2) →
  x * y = 96 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_value_l476_47606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biased_die_expected_value_l476_47642

def biased_die_probabilities : Fin 6 → ℚ
  | 0 => 1/10
  | 1 => 1/10
  | 2 => 1/10
  | 3 => 1/10
  | 4 => 1/10
  | 5 => 1/2

def biased_die_outcomes : Fin 6 → ℚ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | 4 => 5
  | 5 => 6

def expected_value (probs : Fin 6 → ℚ) (outcomes : Fin 6 → ℚ) : ℚ :=
  (Finset.sum Finset.univ fun i => probs i * outcomes i)

theorem biased_die_expected_value :
  expected_value biased_die_probabilities biased_die_outcomes = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_biased_die_expected_value_l476_47642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_reciprocal_l476_47625

/-- The function f(x) = (3 - x^2) / (1 + x^2) -/
noncomputable def f (x : ℝ) : ℝ := (3 - x^2) / (1 + x^2)

/-- For all non-zero real x, f(x) + f(1/x) = 2 -/
theorem f_sum_reciprocal (x : ℝ) (hx : x ≠ 0) : f x + f (1/x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_reciprocal_l476_47625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_to_geometric_l476_47639

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) :=
  ∃ r, ∀ n, b (n + 1) = r * b n

theorem arithmetic_to_geometric
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_arithmetic : arithmetic_sequence a d)
  (h_log_arithmetic : 2 * Real.log (a 2) = Real.log (a 1) + Real.log (a 4))
  (b : ℕ → ℝ)
  (h_b_def : ∀ n, b n = 1 / a (2^n))
  (h_sum : b 1 + b 2 + b 3 = 7/24) :
  geometric_sequence b ∧
  ((a 1 = 72/7 ∧ d = 0) ∨ (a 1 = 3 ∧ d = 3)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_to_geometric_l476_47639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_increase_l476_47617

theorem average_weight_increase (initial_count : ℕ) (initial_weight : ℝ) (old_weight new_weight : ℝ) :
  initial_count = 8 →
  old_weight = 45 →
  new_weight = 65 →
  (new_weight - old_weight) / (initial_count : ℝ) = 2.5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_increase_l476_47617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transforms_curve_l476_47620

-- Define the initial curve
noncomputable def initial_curve (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 36

-- Define the final curve
noncomputable def final_curve (x' y' : ℝ) : Prop := x'^2 + y'^2 = 1

-- Define the scaling transformation
noncomputable def scaling_transformation (x y : ℝ) : ℝ × ℝ := (x / 3, y / 2)

-- Theorem statement
theorem scaling_transforms_curve :
  ∀ x y : ℝ, initial_curve x y →
  (let (x', y') := scaling_transformation x y
   final_curve x' y') := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transforms_curve_l476_47620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_of_P_l476_47690

-- Define the function representing the graph
noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x * Real.pi / 180)

-- Define the point P
def P : ℝ × ℝ := (60, 0)

-- Theorem statement
theorem x_coordinate_of_P : 
  P.1 = 60 ∧ f P.1 = 0 := by
  sorry

#eval P.1 -- This will output 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_of_P_l476_47690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_condition_l476_47660

theorem triangle_angle_condition :
  (∀ A : Real, 0 < A → A < π → (Real.sin A > 1/2 → A > π/6)) ∧ 
  (∃ A : Real, 0 < A ∧ A < π ∧ A > π/6 ∧ Real.sin A ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_condition_l476_47660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pedal_triangle_similarity_124_no_pedal_triangle_similarity_118_l476_47627

/-- A triangle with angles in ratio a:b:c -/
structure RatioTriangle (a b c : ℕ) where
  vertices : Fin 3 → ℝ × ℝ
  angle_ratio : ∀ i : Fin 3, ∃ (θ : ℝ), θ = (a + b + c) / [a, b, c][i] * π / 180

/-- The nth pedal triangle of a given triangle -/
noncomputable def nthPedalTriangle (n : ℕ) {a b c : ℕ} (T : RatioTriangle a b c) : RatioTriangle a b c :=
  sorry

/-- Two triangles are similar -/
def similar {a b c a' b' c' : ℕ} (T1 : RatioTriangle a b c) (T2 : RatioTriangle a' b' c') : Prop :=
  sorry

theorem pedal_triangle_similarity_124 (n : ℕ) :
  ∀ (H : RatioTriangle 1 2 4), similar (nthPedalTriangle n H) H := by
  sorry

theorem no_pedal_triangle_similarity_118 (n : ℕ) :
  ∀ (G : RatioTriangle 1 1 8), ¬ similar (nthPedalTriangle n G) G := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pedal_triangle_similarity_124_no_pedal_triangle_similarity_118_l476_47627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_power_consumption_l476_47650

-- Define the power consumption function
noncomputable def Q (x : ℝ) : ℝ := (1/40) * x^3 - 2 * x^2 + 100 * x

-- Define the total power consumption function
noncomputable def f (v : ℝ) : ℝ := (30 / v) * Q v

-- Theorem statement
theorem min_power_consumption :
  ∀ v : ℝ, 0 < v → v ≤ 80 →
  f v ≥ 1800 ∧
  (f v = 1800 ↔ v = 40) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_power_consumption_l476_47650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_graph_symmetry_l476_47654

/-- An even function f(x) with parameters A, ω, and φ. -/
noncomputable def f (A ω φ x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

/-- Condition for f to be an even function. -/
def is_even (A ω φ : ℝ) : Prop := ∀ x, f A ω φ x = f A ω φ (-x)

/-- The shifted function g(x) obtained by shifting f(x) to the right by π/4 units. -/
noncomputable def g (A ω φ x : ℝ) : ℝ := f A ω φ (x - Real.pi/4)

/-- Condition for g to be symmetric about the origin. -/
def is_symmetric_about_origin (A ω φ : ℝ) : Prop := ∀ x, g A ω φ x = -g A ω φ (-x)

/-- Main theorem stating the conditions for the shifted graph to be symmetric about the origin. -/
theorem shifted_graph_symmetry (A ω φ : ℝ) (hA : A ≠ 0) (hω : ω > 0) (hφ : 0 ≤ φ ∧ φ ≤ Real.pi) :
  is_even A ω φ → (is_symmetric_about_origin A ω φ ↔ ω = 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_graph_symmetry_l476_47654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olympiad_problem_existence_l476_47673

/-- 
Given a set of students and problems where:
- The number of students (m) is greater than 1
- The number of problems (n) is greater than 1
- Each student solved a different number of problems
- Each problem was solved by a different number of students

Prove that there exists at least one student who solved exactly one problem.
-/

-- Define the number of problems solved by each student
def problems_solved : ℕ → ℕ := sorry

-- Define the number of students who solved each problem
def students_solved : ℕ → ℕ := sorry

theorem olympiad_problem_existence (m n : ℕ) 
  (hm : m > 1) 
  (hn : n > 1) 
  (different_student_counts : ∀ i j : ℕ, i ≠ j → problems_solved i ≠ problems_solved j)
  (different_problem_counts : ∀ i j : ℕ, i ≠ j → students_solved i ≠ students_solved j) :
  ∃ s : ℕ, problems_solved s = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_olympiad_problem_existence_l476_47673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypersphere_measure_l476_47614

noncomputable section

/-- The circumference of a circle with radius r -/
def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

/-- The area of a circle with radius r -/
def area (r : ℝ) : ℝ := Real.pi * r^2

/-- The surface area of a sphere with radius r -/
def surfaceArea (r : ℝ) : ℝ := 4 * Real.pi * r^2

/-- The volume of a sphere with radius r -/
def volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

/-- The 3D measure of a hypersphere with radius r -/
def threeDMeasure (r : ℝ) : ℝ := 8 * Real.pi * r^3

/-- The conjectured 4D measure of a hypersphere with radius r -/
def fourDMeasure (r : ℝ) : ℝ := 2 * Real.pi * r^4

theorem hypersphere_measure (r : ℝ) :
  (deriv area r) = circumference r ∧
  (deriv volume r) = surfaceArea r →
  (deriv fourDMeasure r) = threeDMeasure r :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypersphere_measure_l476_47614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_distance_l476_47685

/-- Regular truncated quadrilateral pyramid -/
structure TruncatedPyramid where
  a : ℝ  -- Lower base side length
  b : ℝ  -- Upper base side length
  α : ℝ  -- Angle between lateral face and base plane
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_b_lt_a : b < a
  h_α_pos : 0 < α
  h_α_lt_pi_half : α < π / 2

/-- The distance from the intersection line to the lower base -/
noncomputable def intersectionLineDistance (p : TruncatedPyramid) : ℝ :=
  (p.a * (p.a - p.b) * Real.tan p.α) / (3 * p.a - p.b)

/-- Theorem stating the distance from the intersection line to the lower base -/
theorem intersection_line_distance (p : TruncatedPyramid) :
  ∃ (d : ℝ), d = intersectionLineDistance p ∧ 
  d = (p.a * (p.a - p.b) * Real.tan p.α) / (3 * p.a - p.b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_distance_l476_47685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_sum_of_powers_l476_47684

theorem min_max_sum_of_powers (p q r s : ℝ) 
  (sum_condition : p + q + r + s = 8)
  (sum_squares_condition : p^2 + q^2 + r^2 + s^2 = 20) :
  let f := λ (x y z w : ℝ) ↦ 5 * (x^3 + y^3 + z^3 + w^3) - (x^4 + y^4 + z^4 + w^4)
  ∃ (min max : ℝ), 
    (∀ a b c d, a + b + c + d = 8 → a^2 + b^2 + c^2 + d^2 = 20 → 
      f a b c d ≥ min ∧ f a b c d ≤ max) ∧
    (f p q r s = min ∨ f p q r s = max) ∧
    min = -32 ∧ max = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_sum_of_powers_l476_47684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiling_iff_integer_dimension_l476_47657

/-- Represents a board with dimensions a × b -/
structure Board where
  a : ℝ
  b : ℝ

/-- Represents a small rectangle with dimensions 1 × t -/
structure SmallRectangle where
  t : ℝ
  t_pos : t > 0

/-- Predicate indicating whether a board can be tiled with small rectangles -/
def can_be_tiled (board : Board) (small_rect : SmallRectangle) : Prop :=
  ∃ (tiling : Set (ℝ × ℝ)),
    (∀ (x y : ℝ), (x, y) ∈ tiling → 
      x ≥ 0 ∧ x ≤ board.a ∧ y ≥ 0 ∧ y ≤ board.b) ∧
    (∀ (x y : ℝ), 0 ≤ x ∧ x ≤ board.a ∧ 0 ≤ y ∧ y ≤ board.b →
      ∃ (x' y' : ℝ), (x', y') ∈ tiling ∧ 
        x' ≤ x ∧ x < x' + 1 ∧ y' ≤ y ∧ y < y' + small_rect.t)

/-- Theorem stating that a board can be tiled if and only if one of its dimensions is an integer -/
theorem tiling_iff_integer_dimension (board : Board) (small_rect : SmallRectangle) :
  can_be_tiled board small_rect ↔ (∃ n : ℤ, board.a = n) ∨ (∃ m : ℤ, board.b = m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiling_iff_integer_dimension_l476_47657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_team_size_l476_47645

def is_valid_selection (S : Finset ℕ) : Prop :=
  ∀ x, x ∈ S →
    x ≤ 100 ∧
    (∀ y z, y ∈ S → z ∈ S → x ≠ y + z) ∧
    (∀ y, y ∈ S → x ≠ 2 * y)

theorem max_team_size :
  ∃ (S : Finset ℕ), is_valid_selection S ∧ S.card = 50 ∧
  ∀ (T : Finset ℕ), is_valid_selection T → T.card ≤ 50 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_team_size_l476_47645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_complex_product_l476_47696

theorem real_part_complex_product (α β : ℝ) :
  (Complex.cos α + Complex.I * Complex.sin α) * (Complex.cos β + Complex.I * Complex.sin β) =
  Complex.cos (α + β) + Complex.I * Complex.sin (α + β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_complex_product_l476_47696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_problem_l476_47674

/-- The quadratic function y = ax^2 + 2ax + 3a^2 + 3 -/
def f (a x : ℝ) : ℝ := a * x^2 + 2 * a * x + 3 * a^2 + 3

/-- The derivative of f with respect to x -/
def f' (a x : ℝ) : ℝ := 2 * a * x + 2 * a

theorem quadratic_function_problem (a : ℝ) :
  (∀ x ≤ -2, f' a x < 0) ∧
  (∃ x ∈ Set.Icc (-2) 1, ∀ y ∈ Set.Icc (-2) 1, f a x ≥ f a y) ∧
  (∃ x ∈ Set.Icc (-2) 1, f a x = 9) →
  a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_problem_l476_47674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_inequality_l476_47687

theorem sine_sum_inequality (α β : Real) (h1 : 0 < α) (h2 : α < π/2) (h3 : 0 < β) (h4 : β < π/2) :
  Real.sin (α + β) < Real.sin α + Real.sin β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_inequality_l476_47687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_refund_calculation_l476_47678

/-- The annual maximum amount eligible for tax deduction in rubles -/
def annual_max_deduction : ℝ := 200000

/-- The tax rate as a decimal -/
def tax_rate : ℝ := 0.13

/-- The number of years for the tax refund -/
def num_years : ℕ := 3

/-- Calculates the total tax refund for a given number of years -/
def total_tax_refund (max_deduction : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  max_deduction * rate * (years : ℝ)

/-- Theorem stating that the total tax refund for three years is 78000 rubles -/
theorem tax_refund_calculation :
  total_tax_refund annual_max_deduction tax_rate num_years = 78000 := by
  -- Unfold the definition of total_tax_refund
  unfold total_tax_refund
  -- Simplify the expression
  simp [annual_max_deduction, tax_rate, num_years]
  -- Perform the calculation
  norm_num

#eval total_tax_refund annual_max_deduction tax_rate num_years

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_refund_calculation_l476_47678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l476_47610

/-- The line from which the tangent is drawn -/
def line (x : ℝ) : ℝ := x + 1

/-- The circle to which the tangent is drawn -/
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

/-- The length of the tangent from a point (a, line a) to the circle -/
noncomputable def tangent_length (a : ℝ) : ℝ := 
  Real.sqrt (2 * (a - 1)^2 + 1)

/-- The theorem stating that the minimum length of the tangent is 1 -/
theorem min_tangent_length : 
  ∃ (a : ℝ), ∀ (b : ℝ), tangent_length a ≤ tangent_length b ∧ tangent_length a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l476_47610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_given_properties_l476_47629

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def divisors (n : ℕ) : Finset ℕ := Finset.filter (· ∣ n) (Finset.range (n + 1))

def sum_of_divisors (n : ℕ) : ℕ := (divisors n).sum id

theorem unique_number_with_given_properties :
  ∃! n : ℕ, 
    ((divisors n).card = 6) ∧ 
    (∃ p q : ℕ, p ∈ divisors n ∧ q ∈ divisors n ∧ p ≠ q ∧ is_prime p ∧ is_prime q) ∧
    (sum_of_divisors n = 78) ∧
    n = 45 :=
by
  sorry

#eval divisors 45
#eval sum_of_divisors 45

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_given_properties_l476_47629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_is_negative_two_l476_47619

-- Define the angle α
variable (α : Real)

-- Define the condition that the terminal side of α lies on y = -2x
def terminal_side_on_line (α : Real) : Prop :=
  ∃ x y : Real, x ≠ 0 ∧ y = -2 * x ∧ 
    (Real.cos α = x / Real.sqrt (x^2 + y^2)) ∧ 
    (Real.sin α = y / Real.sqrt (x^2 + y^2))

-- State the theorem
theorem tan_alpha_is_negative_two (α : Real) (h : terminal_side_on_line α) : Real.tan α = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_is_negative_two_l476_47619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tri_connected_squares_count_l476_47667

theorem tri_connected_squares_count : 
  let lower_bound := 2018
  let upper_bound := 3018
  (Finset.filter (λ n : ℕ ↦ n % 2 = 0 ∧ lower_bound ≤ n ∧ n ≤ upper_bound) (Finset.range (upper_bound + 1))).card = 501 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tri_connected_squares_count_l476_47667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_r_for_each_k_l476_47664

def sequence_f (c : ℕ) : ℕ → ℕ
  | 0 => 1  -- Add this case for n = 0
  | 1 => 1
  | 2 => c
  | (n + 3) => 2 * sequence_f c (n + 2) - sequence_f c (n + 1) + 2

theorem exists_r_for_each_k (c : ℕ) (hc : c > 0) :
  ∀ k : ℕ, ∃ r : ℕ, r = k^2 + (c - 3)*k + (c - 4) ∧
    sequence_f c k * sequence_f c (k + 1) = sequence_f c r :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_r_for_each_k_l476_47664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_c_over_a_l476_47692

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def is_acute_triangle (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2

def satisfies_condition (t : Triangle) : Prop :=
  (Real.sqrt 3 * t.c - 2 * Real.sin t.B * Real.sin t.C) = 
  Real.sqrt 3 * (t.b * Real.sin t.B - t.a * Real.sin t.A)

-- Theorem statement
theorem range_of_c_over_a (t : Triangle) 
  (h_acute : is_acute_triangle t) 
  (h_condition : satisfies_condition t) : 
  ∃ (x : ℝ), 1/2 < x ∧ x < 2 ∧ x = t.c / t.a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_c_over_a_l476_47692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_functions_example_l476_47677

-- Define the "H function" property
def is_h_function (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ D → x₂ ∈ D → x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

-- Define the functions
noncomputable def f₁ (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 + (1/2) * x
noncomputable def f₂ (x : ℝ) : ℝ := 3*x + Real.cos x - Real.sin x

-- Define the intervals
def D₁ : Set ℝ := Set.univ
def D₂ : Set ℝ := Set.Ioo 0 (Real.pi/2)

-- State the theorem
theorem h_functions_example : 
  is_h_function f₁ D₁ ∧ is_h_function f₂ D₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_functions_example_l476_47677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_power_sum_l476_47631

theorem log_power_sum (c d : ℝ) (hc : c = Real.log 4) (hd : d = Real.log 25) :
  (5 : ℝ) ^ (c / d) + (2 : ℝ) ^ (d / c) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_power_sum_l476_47631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2n_is_perfect_square_a_2n_specific_form_l476_47680

/-- Sequence a_n represents the number of natural numbers with digit sum n using only digits 1, 3, and 4 -/
def a : ℕ → ℕ := sorry

/-- Sequence c_n is defined as a_n + a_{n-2} -/
def c : ℕ → ℕ
| 0 => 0  -- Arbitrary value for n = 0
| 1 => 1
| 2 => 2
| (n + 3) => c (n + 2) + c (n + 1)

/-- The main theorem stating that a_{2n} is a perfect square for n ≥ 3 -/
theorem a_2n_is_perfect_square (n : ℕ) (h : n ≥ 3) : 
  ∃ k : ℕ, a (2 * n) = k^2 := by
  sorry

/-- The specific form of a_{2n} as (a_n + a_{n-2})^2 for n ≥ 3 -/
theorem a_2n_specific_form (n : ℕ) (h : n ≥ 3) :
  a (2 * n) = (c n)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2n_is_perfect_square_a_2n_specific_form_l476_47680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_points_l476_47603

/-- Calculates the area of a triangle given three points -/
noncomputable def area_triangle (P Q R : ℝ × ℝ) : ℝ :=
  (1/2) * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

/-- Given points A and B, proves the existence of two points C₁ and C₂ on the y-axis
    forming triangles ABC₁ and ABC₂ with area 5 -/
theorem triangle_area_points (A B : ℝ × ℝ) (h_A : A = (3, 4)) (h_B : B = (6, 6)) :
  ∃ (C₁ C₂ : ℝ × ℝ),
    (C₁.1 = 0 ∧ C₂.1 = 0) ∧
    (area_triangle A B C₁ = 5 ∧ area_triangle A B C₂ = 5) ∧
    (C₁ = (0, 16/3) ∧ C₂ = (0, -4/3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_points_l476_47603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_division_game_theorem_l476_47630

/-- Represents a player in the segment division game -/
inductive Player : Type
| kolya : Player
| leva : Player

/-- Determines the winner of the game based on the segment lengths -/
noncomputable def winner (k l : ℝ) : Player :=
  if k > l then Player.kolya else Player.leva

/-- Represents the strategy for dividing a segment into three parts -/
structure DivisionStrategy :=
  (part1 part2 part3 : ℝ)
  (sum_eq_whole : part1 + part2 + part3 = 1)
  (all_positive : part1 > 0 ∧ part2 > 0 ∧ part3 > 0)

/-- Checks if it's possible to form a triangle from given side lengths -/
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem segment_division_game_theorem (k l : ℝ) (hk : k > 0) (hl : l > 0) :
  (∃ (sk : DivisionStrategy), ∀ (sl : DivisionStrategy),
    ¬(∃ (i j m n o p : ℕ) (hij : i ≠ j) (hmn : m ≠ n) (hmo : m ≠ o) (hno : n ≠ o),
      canFormTriangle (k * sk.part1) (k * sk.part2) (l * sl.part3) ∧
      canFormTriangle (k * sk.part3) (l * sl.part1) (l * sl.part2))) ↔
  winner k l = Player.kolya :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_division_game_theorem_l476_47630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l476_47644

-- Define the function f(x) = xe^x
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := Real.exp x + x * Real.exp x

-- Define the point of tangency
noncomputable def point_of_tangency : ℝ × ℝ := (1, Real.exp 1)

-- Theorem: The equation of the tangent line is y = 2ex - e
theorem tangent_line_equation :
  let (x₀, y₀) := point_of_tangency
  let m := f_derivative x₀
  let tangent_line (x : ℝ) := m * (x - x₀) + y₀
  tangent_line = λ x => 2 * Real.exp 1 * x - Real.exp 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l476_47644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_properties_l476_47621

-- Define the logarithm function
noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem log_properties (b : ℝ) (hb : b > 0 ∧ b ≠ 1) :
  (log b b = 1) ∧
  (log b 1 = 0) ∧
  (log b (b^2) = 2) ∧
  (∀ x : ℝ, x > 0 → log b x ∈ Set.univ) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_properties_l476_47621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_l476_47672

/-- The circumference of the base of a right circular cone formed from a sector of a circle -/
theorem cone_base_circumference (r : ℝ) (angle : ℝ) :
  r > 0 →
  angle > 0 →
  angle < 2 * Real.pi →
  (r = 6 ∧ angle = 4 * Real.pi / 3) →
  2 * Real.pi * r * (angle / (2 * Real.pi)) = 8 * Real.pi :=
by
  intros hr hangle_pos hangle_bound hconditions
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_l476_47672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_equation_iff_on_surface_l476_47622

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Defines a cone with vertex at the origin, axis along positive z-axis, and angle φ with z-axis -/
structure Cone where
  φ : ℝ

/-- Predicate to check if a point lies on the surface of a given cone -/
def onConeSurface (p : Point3D) (c : Cone) : Prop :=
  p.x^2 + p.y^2 = p.z^2 * (Real.tan c.φ)^2

/-- Theorem stating that any point satisfying the cone equation lies on the cone surface -/
theorem cone_equation_iff_on_surface (p : Point3D) (c : Cone) :
  p.x^2 + p.y^2 - p.z^2 * (Real.tan c.φ)^2 = 0 ↔ onConeSurface p c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_equation_iff_on_surface_l476_47622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l476_47679

/-- A complex number z satisfying the given conditions -/
noncomputable def z (t : ℂ) : ℂ := t + 3 + Complex.I * Real.sqrt 3

/-- The condition that (t + 3)/(t - 3) is purely imaginary -/
def is_purely_imaginary (t : ℂ) : Prop :=
  ∃ (k : ℝ), (t + 3) / (t - 3) = Complex.I * k

theorem complex_number_properties 
  (t : ℂ) 
  (h : is_purely_imaginary t) :
  let z := z t
  -- 1. Locus equation
  ∃ (x y : ℝ), z = Complex.mk x y ∧ (x - 3)^2 + (y - Real.sqrt 3)^2 = 9 ∧
  -- 2. Range of argument
  ((0 ≤ Complex.arg z ∧ Complex.arg z < Real.pi / 2) ∨ 
  (11 * Real.pi / 6 ≤ Complex.arg z ∧ Complex.arg z < 2 * Real.pi)) ∧
  -- 3. Maximum and minimum values
  (∀ (w : ℂ), is_purely_imaginary ((w - 3) / Real.sqrt 3) →
    Complex.abs (z - 1)^2 + Complex.abs (z + 1)^2 ≤ 4 * (11 + 6 * Real.sqrt 3)) ∧
  (∀ (w : ℂ), is_purely_imaginary ((w - 3) / Real.sqrt 3) →
    4 * (11 - 6 * Real.sqrt 3) ≤ Complex.abs (z - 1)^2 + Complex.abs (z + 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l476_47679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_at_two_points_slope_when_distance_is_root_17_l476_47605

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

-- Define the line
def my_line (m x y : ℝ) : Prop := m * x - y + 1 - m = 0

-- Define the intersection points
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ my_circle x y ∧ my_line m x y}

-- Statement 1: The line always intersects the circle at two distinct points
theorem line_intersects_circle_at_two_points (m : ℝ) :
  ∃ (p1 p2 : ℝ × ℝ), p1 ∈ intersection_points m ∧ p2 ∈ intersection_points m ∧ p1 ≠ p2 :=
sorry

-- Statement 2: When the distance between intersection points is √17, the slope is ±√3
theorem slope_when_distance_is_root_17 :
  ∃ (m : ℝ), (∃ (p1 p2 : ℝ × ℝ), p1 ∈ intersection_points m ∧ p2 ∈ intersection_points m ∧
    (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 = 17) → (m = Real.sqrt 3 ∨ m = -Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_at_two_points_slope_when_distance_is_root_17_l476_47605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_set_eq_sum_probs_l476_47661

/-- A discrete random variable -/
structure DiscreteRandomVariable (Ω : Type*) (α : Type*) where
  pmf : α → ℝ
  nonneg : ∀ x, 0 ≤ pmf x
  sum_one : ∑' x, pmf x = 1

/-- The probability of a discrete random variable taking values in a set -/
noncomputable def prob_of_set {Ω α : Type*} [MeasurableSpace α] (X : DiscreteRandomVariable Ω α) (S : Set α) : ℝ :=
  ∑' x, (Set.indicator S (X.pmf)) x

/-- Theorem: The probability of a discrete random variable taking values in a set
    is equal to the sum of probabilities of taking each value in that set -/
theorem prob_set_eq_sum_probs {Ω α : Type*} [MeasurableSpace α] (X : DiscreteRandomVariable Ω α) (S : Set α) :
  prob_of_set X S = ∑' x, (Set.indicator S (X.pmf)) x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_set_eq_sum_probs_l476_47661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_monotonic_implies_a_equals_three_l476_47626

-- Define the power function
noncomputable def power_function (a : ℝ) (x : ℝ) : ℝ := (a^2 - 2*a - 2) * x^a

-- Define monotonicity for a function on ℝ
def is_monotonic (f : ℝ → ℝ) : Prop := 
  ∀ x y : ℝ, x < y → f x < f y ∨ (∀ z : ℝ, f z = f x)

-- Theorem statement
theorem power_function_monotonic_implies_a_equals_three :
  (is_monotonic (power_function a)) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_monotonic_implies_a_equals_three_l476_47626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_internal_sphere_radius_l476_47675

/-- The radius of a sphere internally tangent to the circumsphere of a regular tetrahedron -/
theorem internal_sphere_radius (a α : ℝ) (h_a : a > 0) (h_α : 0 < α ∧ α < π) : 
  ∃ ρ : ℝ, ρ = (a * (1 - Real.cos α)) / (2 * Real.sqrt (1 + Real.cos α) * (1 + Real.sqrt (-Real.cos α))) ∧
  ρ > 0 ∧ 
  ρ < a * Real.sqrt 6 / 4 := by
  sorry

#check internal_sphere_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_internal_sphere_radius_l476_47675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sine_linear_function_l476_47607

/-- A function y = sin x + ax is increasing on ℝ if and only if a ≥ 1 -/
theorem increasing_sine_linear_function (a : ℝ) :
  (∀ x : ℝ, StrictMono (fun x ↦ Real.sin x + a * x)) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sine_linear_function_l476_47607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_common_divisor_l476_47682

theorem largest_common_divisor :
  ∃ (k : ℕ), k = 8 ∧ 
  (∀ (m : ℕ), m > k → ¬(∀ (n : ℤ), (m : ℤ) ∣ ((n+2)*(n+4)*(n+6)*(n+8)*(n+10)))) ∧
  (∀ (n : ℤ), (k : ℤ) ∣ ((n+2)*(n+4)*(n+6)*(n+8)*(n+10))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_common_divisor_l476_47682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_theorem_l476_47612

-- Define the equation and its roots
noncomputable def equation (x : ℝ) : ℝ := Real.log x ^ 2 - Real.log x - 2

-- Define α and β as the roots of the equation
axiom α : ℝ
axiom β : ℝ
axiom α_root : equation α = 0
axiom β_root : equation β = 0
axiom α_pos : α > 0
axiom β_pos : β > 0

-- Define the expression we want to prove
noncomputable def expression : ℝ := (Real.log β) / (Real.log α) + (Real.log α) / (Real.log β)

-- The theorem to prove
theorem root_sum_theorem : expression = -5/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_theorem_l476_47612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_proof_l476_47663

theorem sine_function_proof (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 2 * Real.sin (3 * x - π / 4)
  (2 > 0) ∧ 
  (-π/2 < -π/4 ∧ -π/4 < 0) ∧ 
  (3 > 0) ∧ 
  (∀ x, f x ≥ -2) ∧ 
  (∀ x, f (x + 2*π/3) = f x) ∧ 
  (f 0 = -Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_proof_l476_47663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_eq_8R_squared_l476_47686

/-- A quadrilateral inscribed in a circle with perpendicular diagonals -/
structure InscribedQuadrilateral (R : ℝ) where
  /-- The four sides of the quadrilateral -/
  sides : Fin 4 → ℝ
  /-- The quadrilateral is inscribed in a circle of radius R -/
  inscribed : True
  /-- The diagonals of the quadrilateral are perpendicular -/
  perp_diagonals : True

/-- The sum of squares of sides of an inscribed quadrilateral with perpendicular diagonals -/
def sum_of_squares (R : ℝ) (q : InscribedQuadrilateral R) : ℝ :=
  Finset.sum Finset.univ (fun i => (q.sides i) ^ 2)

/-- Theorem: The sum of squares of sides equals 8R^2 -/
theorem sum_of_squares_eq_8R_squared (R : ℝ) (q : InscribedQuadrilateral R) :
  sum_of_squares R q = 8 * R^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_eq_8R_squared_l476_47686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l476_47697

/-- The function f(a,b,c) as defined in the problem -/
noncomputable def f (a b c : ℝ) : ℝ :=
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*a*c) + c / Real.sqrt (c^2 + 8*a*b)

/-- Theorem stating that the minimum value of f(a,b,c) is 1 for positive real numbers -/
theorem f_min_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  f a b c ≥ 1 := by
  sorry

#check f_min_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l476_47697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l476_47632

noncomputable def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

def tangent_line (x y : ℝ) : Prop := 2*x - Real.sqrt 2*y + 6 = 0

def intersecting_line (x y k : ℝ) : Prop := y = k*(x - 2)

noncomputable def distance_squared (x1 y1 x2 y2 : ℝ) : ℝ := (x2 - x1)^2 + (y2 - y1)^2

def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1*x2 + y1*y2

theorem ellipse_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (h_ecc : eccentricity a b = Real.sqrt 6 / 3)
  (h_tangent : ∃ (x y : ℝ), ellipse x y a a ∧ tangent_line x y) :
  (∀ (x y : ℝ), ellipse x y (Real.sqrt 6) (Real.sqrt 2) ↔ ellipse x y a b) ∧
  (∃ (m : ℝ), m = 7/3 ∧
    ∀ (k : ℝ) (hk : k ≠ 0) (x1 y1 x2 y2 : ℝ),
      ellipse x1 y1 a b → ellipse x2 y2 a b →
      intersecting_line x1 y1 k → intersecting_line x2 y2 k →
      distance_squared m 0 x1 y1 + dot_product (x1-m) y1 (x2-x1) (y2-y1) = -5/9) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l476_47632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_monotonicity_l476_47658

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 4 / (Real.exp (2 * x) + 1) + a

theorem odd_function_and_monotonicity (a : ℝ) :
  (∀ x, f (-x) a = -f x a) →
  (a = -2 ∧ ∀ x y, x < y → f x (-2) > f y (-2)) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_monotonicity_l476_47658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_y_value_l476_47662

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The first three terms of our sequence, extended to ℕ -/
def sequence_terms (y : ℚ) : ℕ → ℚ
  | 0 => -2
  | 1 => y - 4
  | 2 => -6 * y + 8
  | _ => 0  -- Default value for n ≥ 3

theorem arithmetic_sequence_y_value :
  ∀ y : ℚ, (is_arithmetic_sequence (sequence_terms y)) → y = 7/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_y_value_l476_47662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_win_probability_l476_47640

structure GameState where
  alicePos : Int
  billyPos : Int
  aliceTurn : Bool

def winningPosition : Int := 2
def losingPosition : Int := -2

noncomputable def aliceMove : ℝ := 1/2

noncomputable def billyMove : ℝ := 2/3

def isGameOver (state : GameState) : Bool :=
  state.alicePos = winningPosition ∨ state.alicePos = losingPosition ∨
  state.billyPos = winningPosition ∨ state.billyPos = losingPosition

def initialState : GameState :=
  { alicePos := 0, billyPos := 0, aliceTurn := true }

noncomputable def winProbability (state : GameState) : ℝ :=
  sorry

theorem billy_win_probability :
  winProbability initialState = 14/19 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_win_probability_l476_47640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l476_47623

/-- The function f(x) = x³ -/
def f (x : ℝ) : ℝ := x^3

/-- The function g(x) = ln x -/
noncomputable def g (x : ℝ) : ℝ := Real.log x

/-- The distance between points (m, f(m)) and (m, g(m)) -/
noncomputable def distance (m : ℝ) : ℝ := |f m - g m|

/-- Theorem: The minimum distance between the points where x = m intersects f(x) and g(x) is (1/3)(1 + ln 3) -/
theorem min_distance_theorem :
  ∃ (min_m : ℝ), ∀ (m : ℝ), m > 0 → distance m ≥ (1/3) * (1 + Real.log 3) ∧
  distance min_m = (1/3) * (1 + Real.log 3) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l476_47623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l476_47647

-- Define the vector product
def vector_product (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 * b.1, a.2 * b.2)

-- Define the vectors m and n
noncomputable def m : ℝ × ℝ := (1, 1/2)
noncomputable def n : ℝ × ℝ := (0, 1)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  let p : ℝ × ℝ := (x, Real.sin (x/2))
  (vector_product m p + n).2

-- State the theorem
theorem f_properties :
  (∀ x, f x ≤ 3/2) ∧
  (∀ x, f (x + 4*Real.pi) = f x) ∧
  (∀ t, t > 0 → (∀ x, f (x + t) = f x) → t ≥ 4*Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l476_47647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bottles_is_eight_l476_47666

/-- The minimum number of bottles needed to buy at least 60 fluid ounces of oil -/
def min_bottles : ℕ :=
  let fl_oz_needed : ℚ := 60
  let bottle_size_ml : ℚ := 250
  let fl_oz_per_liter : ℚ := 33.8
  let ml_per_liter : ℚ := 1000
  (((fl_oz_needed / fl_oz_per_liter * ml_per_liter) / bottle_size_ml).ceil).toNat

/-- Theorem stating that the minimum number of bottles needed is 8 -/
theorem min_bottles_is_eight : min_bottles = 8 := by
  sorry

#eval min_bottles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bottles_is_eight_l476_47666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_one_two_three_l476_47643

-- Define the Δ operation
noncomputable def delta (x y : ℝ) : ℝ := (x + y) / (x * y)

-- State the theorem
theorem delta_one_two_three : delta (delta 1 2) 3 = 1 := by
  -- Unfold the definition of delta
  unfold delta
  -- Simplify the expression
  simp
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_one_two_three_l476_47643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_not_div_by_6_or_8_l476_47651

theorem three_digit_not_div_by_6_or_8 : 
  Finset.card (Finset.filter (fun n => 100 ≤ n ∧ n ≤ 999 ∧ n % 6 ≠ 0 ∧ n % 8 ≠ 0) (Finset.range 1000)) = 675 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_not_div_by_6_or_8_l476_47651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l476_47653

open Set Real

noncomputable def f (x m : ℝ) := 2 * x + m + 8 / (x - 1)

theorem range_of_m (h : ∀ x > 1, f x m > 0) : 
  m ∈ Ioi (-10) :=
sorry

-- Note: Ioi (-10) represents the open interval (-10, +∞)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l476_47653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l476_47633

theorem vector_equation_solution (a b c : ℝ × ℝ) 
  (ha : a = (1, 2)) 
  (hb : b = (2, 3)) 
  (hc : c = (3, 4)) 
  (heq : ∃ (lambda mu : ℝ), c = lambda • a + mu • b) : 
  ∃ (lambda mu : ℝ), c = lambda • a + mu • b ∧ lambda + mu = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l476_47633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l476_47624

noncomputable def f (x : ℝ) := Real.cos (2 * x + 2 * Real.pi / 3) + 2 * (Real.cos x) ^ 2

noncomputable def g (x : ℝ) := f (x - Real.pi / 3)

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ 
   ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (k : ℤ), ∀ (x : ℝ), 
    (k : ℝ) * Real.pi - Real.pi / 6 ≤ x ∧ x ≤ (k : ℝ) * Real.pi + Real.pi / 3 → 
    ∀ (y : ℝ), x < y → f y < f x) ∧
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 → g x ≥ 1 / 2) ∧
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ g x = 1 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l476_47624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l476_47659

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 4)

theorem f_decreasing_on_interval :
  ∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc (Real.pi / 4) (5 * Real.pi / 4) → 
  x₂ ∈ Set.Icc (Real.pi / 4) (5 * Real.pi / 4) → 
  x₁ ≠ x₂ → 
  (f x₁ - f x₂) / (x₁ - x₂) < 0 := by
  sorry

#check f_decreasing_on_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l476_47659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_ratio_l476_47665

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > b, 
    if the angle between the asymptotes is 45°, then a/b = √2 + 1 -/
theorem hyperbola_asymptote_ratio (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (let θ := Real.arctan (b / a) - Real.arctan (-b / a); θ = π / 4) →
  a / b = Real.sqrt 2 + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_ratio_l476_47665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l476_47608

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 - 3 / x

-- State the theorem
theorem f_monotone_increasing : 
  ∀ (x₁ x₂ : ℝ), 0 < x₂ → x₂ < x₁ → f x₁ > f x₂ := by
  -- The proof will be skipped using 'sorry'
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l476_47608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_for_specific_trapezoid_l476_47669

/-- Represents a trapezoid ABCD with point E where the legs are extended to meet -/
structure ExtendedTrapezoid where
  -- Length of base AB
  ab : ℝ
  -- Length of base CD
  cd : ℝ
  -- Ensure ab and cd are positive
  ab_pos : 0 < ab
  cd_pos : 0 < cd

/-- The ratio of the area of triangle EAB to the area of trapezoid ABCD -/
noncomputable def area_ratio (t : ExtendedTrapezoid) : ℝ :=
  81 / 175

/-- Theorem stating the area ratio for the given trapezoid -/
theorem area_ratio_for_specific_trapezoid :
  ∃ t : ExtendedTrapezoid, t.ab = 9 ∧ t.cd = 16 ∧ area_ratio t = 81 / 175 := by
  -- Construct the specific trapezoid
  let t : ExtendedTrapezoid := {
    ab := 9
    cd := 16
    ab_pos := by norm_num
    cd_pos := by norm_num
  }
  -- Prove the existence
  use t
  -- Prove the conditions
  constructor
  · rfl  -- t.ab = 9
  constructor
  · rfl  -- t.cd = 16
  · rfl  -- area_ratio t = 81 / 175

-- Add this line to make the theorem opaque
attribute [simp] area_ratio_for_specific_trapezoid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_for_specific_trapezoid_l476_47669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_root_determinable_l476_47676

-- Define a cubic polynomial
def CubicPolynomial (a b c d : ℝ) : ℝ → ℝ := λ x ↦ a*x^3 + b*x^2 + c*x + d

-- Define the property of having three distinct roots between 0 and 1
def HasThreeDistinctRootsBetweenZeroAndOne (p : ℝ → ℝ) : Prop :=
  ∃ (r₁ r₂ r₃ : ℝ), 
    0 < r₁ ∧ r₁ < 1 ∧
    0 < r₂ ∧ r₂ < 1 ∧
    0 < r₃ ∧ r₃ < 1 ∧
    r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    p r₁ = 0 ∧ p r₂ = 0 ∧ p r₃ = 0

-- Theorem statement
theorem third_root_determinable 
  (a b c d : ℝ) 
  (r₁ r₂ : ℝ) 
  (h_distinct_roots : HasThreeDistinctRootsBetweenZeroAndOne (CubicPolynomial a b c d))
  (h_r₁ : CubicPolynomial a b c d r₁ = 0)
  (h_r₂ : CubicPolynomial a b c d r₂ = 0)
  : ∃ (r₃ : ℝ), 
    0 < r₃ ∧ r₃ < 1 ∧
    r₃ ≠ r₁ ∧ r₃ ≠ r₂ ∧
    CubicPolynomial a b c d r₃ = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_root_determinable_l476_47676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_2023_value_l476_47668

-- Define the function h recursively
def h : ℕ → ℤ
  | 0 => 3  -- We need to define h for 0 to cover all natural numbers
  | 1 => 3
  | 2 => 2
  | n+3 => h (n+2) - h (n+1) + 2*(n+3)

-- State the theorem
theorem h_2023_value : h 2023 = 4052 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_2023_value_l476_47668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_increase_theorem_l476_47699

/-- The percentage of rent spent relative to last year's rent -/
noncomputable def rent_percentage_increase (last_year_rent_percent : ℝ) 
                              (this_year_income_increase : ℝ) 
                              (this_year_rent_percent : ℝ) : ℝ :=
  (this_year_rent_percent * (1 + this_year_income_increase)) / last_year_rent_percent * 100

/-- Theorem: Given the conditions, the rent percentage increase is 187.5% -/
theorem rent_increase_theorem :
  rent_percentage_increase 0.20 0.25 0.30 = 187.5 := by
  -- Unfold the definition of rent_percentage_increase
  unfold rent_percentage_increase
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_increase_theorem_l476_47699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l476_47613

def sequence_a : ℕ → ℕ
  | 0 => 2  -- We define a_0 = 2 to match a_1 = 2 in the original problem
  | (n + 1) => (sequence_a n)^2 - n * (sequence_a n) + 1

theorem sequence_a_formula (n : ℕ) : sequence_a n = n + 2 := by
  induction n with
  | zero => rfl
  | succ n ih => 
    simp [sequence_a]
    sorry  -- The actual proof would go here

#eval sequence_a 0  -- Should output 2
#eval sequence_a 1  -- Should output 3
#eval sequence_a 2  -- Should output 4
#eval sequence_a 3  -- Should output 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l476_47613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_bounds_l476_47604

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 2  -- Define for 0 to cover all natural numbers
  | 1 => 2
  | (n + 2) => (sequence_a (n + 1) ^ 4 + 1) / (5 * sequence_a (n + 1))

theorem sequence_a_bounds (n : ℕ) (h : n > 1) : 1/5 < sequence_a n ∧ sequence_a n < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_bounds_l476_47604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floss_packet_size_l476_47616

/-- Represents the floss distribution problem --/
def floss_problem (students : ℕ) (floss_per_student : ℚ) (leftover : ℕ) : Prop :=
  let total_needed : ℚ := students * floss_per_student
  let total_bought : ℕ := (Int.toNat (Rat.ceil total_needed)) + leftover
  ∃ (packet_size : ℕ), 
    packet_size > 0 ∧ 
    (total_bought : ℚ) / packet_size = (total_bought / packet_size : ℕ) ∧
    packet_size * (total_bought / packet_size : ℕ) = total_bought ∧
    packet_size = 35

theorem floss_packet_size :
  floss_problem 20 (3/2) 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floss_packet_size_l476_47616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_food_ratio_l476_47609

/-- The amount of food the first dog eats per day -/
noncomputable def first_dog_food : ℝ := 1.5

/-- The amount of food the third dog eats more than the second dog -/
noncomputable def third_dog_extra : ℝ := 2.5

/-- The total amount of food Hannah prepares for all three dogs -/
noncomputable def total_food : ℝ := 10

/-- The amount of food the second dog eats -/
noncomputable def second_dog_food : ℝ := (total_food - first_dog_food - third_dog_extra) / 2

/-- Theorem stating that the ratio of the second dog's food to the first dog's food is 2:1 -/
theorem dog_food_ratio : second_dog_food / first_dog_food = 2 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_food_ratio_l476_47609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l476_47602

noncomputable def f (x : ℝ) : ℝ := (1/3) ^ (-x^2 + 2*x)

theorem f_monotone_increasing : 
  MonotoneOn f (Set.Ioi 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l476_47602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_one_more_than_zero_l476_47628

/-- The sequence of digits formed by concatenating decimal representations of natural numbers from 1 to n -/
def digitSequence (n : ℕ) : List ℕ :=
  (List.range n).bind (fun i => (i + 1).repr.toList.map (fun c => c.toNat - '0'.toNat))

/-- Count the occurrences of a digit in the sequence -/
def countDigit (d : ℕ) (seq : List ℕ) : ℕ :=
  seq.filter (· = d) |>.length

theorem digit_one_more_than_zero (n : ℕ) :
  countDigit 1 (digitSequence n) > countDigit 0 (digitSequence n) := by
  sorry

#check digit_one_more_than_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_one_more_than_zero_l476_47628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l476_47694

theorem sin_beta_value (α β : Real) (h1 : 0 < α ∧ α < Real.pi/2) (h2 : 0 < β ∧ β < Real.pi/2)
  (h3 : Real.cos α = 4/5) (h4 : Real.cos (α + β) = 5/13) : Real.sin β = 33/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l476_47694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_l476_47691

theorem largest_number (a b c d e : ℝ) : 
  a = 24680 + 1 / 1357 ∧ 
  b = 24680 - 1 / 1357 ∧ 
  c = 24680 * (1 / 1357) ∧ 
  d = 24680 / (1 / 1357) ∧ 
  e = Real.rpow 24680 1.3 → 
  e > a ∧ e > b ∧ e > c ∧ e > d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_l476_47691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l476_47683

theorem system_solution (a b c : ℝ) : 
  ∃ (x y z : ℝ),
    (z + a * y + a^2 * x + a^3 = 0) ∧
    (z + b * y + b^2 * x + b^3 = 0) ∧
    (z + c * y + c^2 * x + c^3 = 0) ∧
    x = -(a + b + c) ∧
    y = a * b + b * c + c * a ∧
    z = -(a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l476_47683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_primes_l476_47656

theorem least_multiple_of_primes (n : ℕ) : n = 46149 ↔ 
  (∀ p : ℕ, p ∈ ({11, 13, 17, 19} : Set ℕ) → Nat.Prime p ∧ p ∣ n) ∧
  (∀ m : ℕ, m < n → ∃ p : ℕ, p ∈ ({11, 13, 17, 19} : Set ℕ) ∧ Nat.Prime p ∧ ¬(p ∣ m)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_primes_l476_47656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l476_47634

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + Real.sqrt (5 - x)

-- State the theorem
theorem f_properties :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → f x ≤ 5) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → (f x = 5 ↔ x = 4)) ∧
  (∀ m : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → f x ≤ |m - 2|) → (m ≥ 7 ∨ m ≤ -3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l476_47634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distinct_prime_factors_of_5_5_minus_5_3_l476_47615

theorem sum_of_distinct_prime_factors_of_5_5_minus_5_3 :
  (Finset.sum (Finset.filter Nat.Prime (Finset.range ((5^5 - 5^3).factors.toFinset.card + 1)))
    (fun p => if p ∣ (5^5 - 5^3) then p else 0)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distinct_prime_factors_of_5_5_minus_5_3_l476_47615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l476_47635

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x) / Real.sqrt (x - x^2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x | 0 < x ∧ x < 1 ∧ x ≠ π/4} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l476_47635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_half_equals_pi_plus_one_l476_47655

-- Define the function f
noncomputable def f (a : ℝ) : ℝ := ∫ x in (0:ℝ)..a, (2 + Real.sin x)

-- State the theorem
theorem f_pi_half_equals_pi_plus_one : f (π / 2) = π + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_half_equals_pi_plus_one_l476_47655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_border_length_l476_47693

-- Define the circle's area
noncomputable def circle_area : ℝ := 176

-- Define the estimation of π
noncomputable def π_estimate : ℝ := 22 / 7

-- Define the additional length
noncomputable def extra_length : ℝ := 5

-- Theorem statement
theorem border_length :
  let r := Real.sqrt (circle_area / π_estimate)
  (2 * π_estimate * r + extra_length) = (88 * Real.sqrt 14 + 35) / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_border_length_l476_47693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_relationship_l476_47681

theorem condition_relationship : 
  {x : ℝ | (1/3:ℝ)^x < 1} ⊃ {x : ℝ | 1/x > 1} ∧ 
  {x : ℝ | (1/3:ℝ)^x < 1} ≠ {x : ℝ | 1/x > 1} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_relationship_l476_47681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_5_l476_47671

/-- A function g satisfying the given condition for all non-zero x -/
noncomputable def g : ℝ → ℝ := sorry

/-- The condition that g satisfies for all non-zero x -/
axiom g_condition (x : ℝ) (hx : x ≠ 0) : 4 * g x - 3 * g (1 / x) = 2 * x

/-- Theorem stating that g(5) equals 402/70 -/
theorem g_of_5 : g 5 = 402 / 70 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_5_l476_47671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_25_5_l476_47638

/-- A power function that passes through the point (25,5) -/
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x ^ α

/-- The theorem stating that the power function passing through (25,5) is the square root function -/
theorem power_function_through_25_5 :
  ∃ α : ℝ, (f α 25 = 5) ∧ ∀ x : ℝ, f α x = Real.sqrt x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_25_5_l476_47638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_B_l476_47695

def A : Finset Nat := {1, 2, 3, 4, 5, 6, 7}

def B : Finset Nat := A.product A |>.filter (λ (a, b) => (a * b) % 2 == 0)
  |>.image (λ (a, b) => a * b)

theorem cardinality_of_B : Finset.card B = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_B_l476_47695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_loss_approximately_1_7_percent_l476_47646

noncomputable def initial_investment : ℝ := 200

noncomputable def first_year_loss_rate : ℝ := 0.10
noncomputable def second_year_gain_rate : ℝ := 0.15
noncomputable def third_year_loss_rate : ℝ := 0.05

noncomputable def investment_after_three_years : ℝ :=
  initial_investment * (1 - first_year_loss_rate) * (1 + second_year_gain_rate) * (1 - third_year_loss_rate)

noncomputable def percentage_change : ℝ :=
  (investment_after_three_years - initial_investment) / initial_investment * 100

theorem investment_loss_approximately_1_7_percent :
  abs (percentage_change + 1.7) < 0.1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_loss_approximately_1_7_percent_l476_47646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_implies_a_equals_four_l476_47611

/-- A function f : ℝ → ℝ is exponential if there exist constants c ≠ 0 and b > 1 such that f(x) = c * b^x for all x -/
def IsExponentialFunction (f : ℝ → ℝ) : Prop :=
  ∃ (c b : ℝ), c ≠ 0 ∧ b > 1 ∧ ∀ x, f x = c * b^x

/-- The function under consideration -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a^2 - 5*a + 5) * a^x

theorem exponential_function_implies_a_equals_four :
  (∃ a : ℝ, IsExponentialFunction (f a)) → (∃ a : ℝ, a = 4 ∧ IsExponentialFunction (f a)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_implies_a_equals_four_l476_47611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_l476_47618

-- Define the basic geometric objects
structure Line : Type := (dummy : Unit)
structure Plane : Type := (dummy : Unit)

-- Define the relationships between lines and planes
def parallel_lines (l1 l2 : Line) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def angle_line_plane (l : Line) (p : Plane) : ℝ := sorry

-- State the propositions
def proposition1 (l1 l2 : Line) (p : Plane) : Prop :=
  parallel_lines l1 l2 → angle_line_plane l1 p = angle_line_plane l2 p

def proposition2 (l1 l2 : Line) (p : Plane) : Prop :=
  angle_line_plane l1 p = angle_line_plane l2 p → parallel_lines l1 l2

def proposition3 (l : Line) (p1 p2 : Plane) : Prop :=
  parallel_planes p1 p2 → angle_line_plane l p1 = angle_line_plane l p2

def proposition4 (l : Line) (p1 p2 : Plane) : Prop :=
  angle_line_plane l p1 = angle_line_plane l p2 → parallel_planes p1 p2

-- Theorem stating which propositions are correct
theorem correct_propositions :
  (∀ l1 l2 : Line, ∀ p : Plane, proposition1 l1 l2 p) ∧
  (∃ l1 l2 : Line, ∃ p : Plane, ¬proposition2 l1 l2 p) ∧
  (∀ l : Line, ∀ p1 p2 : Plane, proposition3 l p1 p2) ∧
  (∃ l : Line, ∃ p1 p2 : Plane, ¬proposition4 l p1 p2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_l476_47618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_profit_percent_l476_47637

noncomputable def purchase_price : ℝ := 42000
noncomputable def repair_cost : ℝ := 10000
noncomputable def selling_price : ℝ := 64900

noncomputable def total_cost : ℝ := purchase_price + repair_cost
noncomputable def profit : ℝ := selling_price - total_cost
noncomputable def profit_percent : ℝ := (profit / total_cost) * 100

theorem car_profit_percent : 
  abs (profit_percent - 24.81) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_profit_percent_l476_47637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_distribution_proof_l476_47649

/-- The number of ways to distribute 4 distinct books to 3 students -/
def distribute_books : ℕ := 36

/-- The number of ways to choose 2 books out of 4 -/
def choose_two_books : ℕ := Nat.choose 4 2

/-- The number of ways to permute 3 groups -/
def permute_groups : ℕ := Nat.factorial 3

theorem book_distribution_proof :
  distribute_books = choose_two_books * permute_groups :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_distribution_proof_l476_47649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_situps_l476_47698

/-- The ratio of sit-ups Peter does compared to Greg -/
noncomputable def peter_to_greg_ratio : ℚ := 25 / 18

/-- The ratio of sit-ups Peter does compared to Susan -/
noncomputable def peter_to_susan_ratio : ℚ := 25 / 15

/-- The number of sit-ups Peter did -/
def peter_situps : ℚ := 75 / 2

/-- The number of sit-ups Greg did -/
noncomputable def greg_situps : ℚ := peter_situps / peter_to_greg_ratio

/-- The number of sit-ups Susan did -/
noncomputable def susan_situps : ℚ := peter_situps / peter_to_susan_ratio

/-- Theorem stating the combined number of sit-ups Greg and Susan did -/
theorem combined_situps : greg_situps + susan_situps = 99 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_situps_l476_47698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_integer_roots_l476_47652

/-- Represents a quartic polynomial with rational coefficients -/
structure QuarticPolynomial where
  d : ℚ
  e : ℚ
  f : ℚ

/-- Checks if a number is a root of the quartic polynomial -/
def is_root (p : QuarticPolynomial) (x : ℝ) : Prop :=
  x^4 + p.d * x^2 + p.e * x + p.f = 0

theorem quartic_integer_roots (p : QuarticPolynomial) :
  is_root p (2 - Real.sqrt 5) →
  (∃ (r₁ r₂ : ℤ), is_root p (r₁ : ℝ) ∧ is_root p (r₂ : ℝ) ∧ r₁ ≠ r₂) →
  (∃ (r₁ r₂ : ℤ), is_root p (r₁ : ℝ) ∧ is_root p (r₂ : ℝ) ∧ r₁ ≠ r₂ ∧ ({r₁, r₂} : Set ℤ) = {-1, -3}) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_integer_roots_l476_47652
