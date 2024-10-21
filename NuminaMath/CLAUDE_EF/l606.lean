import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_alone_time_l606_60607

/-- The number of days it takes for worker c to finish the job alone -/
noncomputable def days_for_c : ℚ := 165 / 4

theorem c_alone_time :
  (∃ (rate_a rate_b rate_c : ℚ),
    rate_a + rate_b = 1 / 15 ∧
    rate_a + rate_b + rate_c = 1 / 11) →
  (∃ (rate_c : ℚ), 1 / rate_c = days_for_c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_alone_time_l606_60607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mariam_neighborhood_habitable_homes_l606_60658

def houses_side_a : ℕ := 40

def houses_side_b (x : ℕ) : ℕ := x^2 + 3*x

def uninhabitable_percentage : ℚ := 1/10

def environmental_constraint_percentage : ℚ := 1/2

theorem mariam_neighborhood_habitable_homes :
  let total_side_b := houses_side_b houses_side_a
  let habitable_side_b := total_side_b - (uninhabitable_percentage * ↑total_side_b).floor
  houses_side_a + habitable_side_b = 1588 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mariam_neighborhood_habitable_homes_l606_60658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_odd_condition_l606_60605

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (x + φ)

theorem sin_odd_condition (φ : ℝ) :
  (φ = Real.pi → (∀ x, f φ (-x) = -(f φ x))) ∧
  (∃ ψ, ψ ≠ Real.pi ∧ (∀ x, f ψ (-x) = -(f ψ x))) := by
  sorry

#check sin_odd_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_odd_condition_l606_60605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_theorem_l606_60644

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  DG : ℝ
  is_trapezoid : AB ≠ CD
  base_lengths : AB = 15 ∧ CD = 19
  altitude_length : DG = 17

/-- Represents the quadrilateral KLMN formed by midpoints -/
def Quadrilateral (t : Trapezoid) : Type :=
  { KLMN : Set (ℝ × ℝ) // 
    ∃ (K L M N : ℝ × ℝ),
    K ∈ KLMN ∧ L ∈ KLMN ∧ M ∈ KLMN ∧ N ∈ KLMN ∧
    K.1 = t.AB / 2 ∧ M.1 = t.CD / 2 }

/-- Area of the trapezoid -/
def area_trapezoid (t : Trapezoid) : ℝ := sorry

/-- Area of the quadrilateral -/
def area_quadrilateral (t : Trapezoid) (q : Quadrilateral t) : ℝ := sorry

/-- The ratio of areas is either 2 or 2/3 -/
theorem area_ratio_theorem (t : Trapezoid) (q : Quadrilateral t) :
  let r := (area_trapezoid t) / (area_quadrilateral t q)
  r = 2 ∨ r = 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_theorem_l606_60644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_g_evaluation_l606_60654

-- Define the function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≥ 3 then -x^2 + 1 else x + 10

-- State the theorem
theorem nested_g_evaluation : g (g (g (g (g 4)))) = -14 :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_g_evaluation_l606_60654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_parabola_l606_60686

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 15 = 0

-- Define the parabola
def parabola_eq (x y : ℝ) : Prop := y^2 = 4*x

-- Define the distance function between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem min_distance_circle_parabola :
  ∃ (d : ℝ), d = 3 ∧
  ∀ (x1 y1 x2 y2 : ℝ),
    circle_eq x1 y1 → parabola_eq x2 y2 →
    distance x1 y1 x2 y2 ≥ d :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_parabola_l606_60686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_holds_iff_l606_60603

def sum_series (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) (λ i => i * (i + 1))

theorem equation_holds_iff (n : ℕ) : 
  sum_series n = 3 * n^2 - 3 * n + 2 ↔ n = 1 ∨ n = 2 ∨ n = 3 := by
  sorry

#check equation_holds_iff

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_holds_iff_l606_60603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l606_60681

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (A^2 + B^2)

/-- The first line: x - y + 2 = 0 -/
def line1 (x y : ℝ) : Prop := x - y + 2 = 0

/-- The second line: x - y = 0 -/
def line2 (x y : ℝ) : Prop := x - y = 0

theorem distance_between_lines : 
  distance_between_parallel_lines 1 (-1) 2 0 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l606_60681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_infinite_series_sum_l606_60640

/-- The sum of the double infinite series ∑_{j = 0}^∞ ∑_{k = 0}^∞ 2^{-4k - 2j - (k + j)^2} is equal to 4/3. -/
theorem double_infinite_series_sum :
  (∑' j : ℝ, ∑' k : ℝ, (2 : ℝ) ^ (-4 * k - 2 * j - (k + j)^2)) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_infinite_series_sum_l606_60640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_interval_l606_60646

noncomputable def g : ℕ → ℝ
  | 0 => 0  -- Base case for 0
  | 1 => 0  -- Base case for 1
  | 2 => 0  -- Base case for 2
  | 3 => 0  -- Base case for 3
  | 4 => 0  -- Base case for 4
  | 5 => 0  -- Base case for 5
  | 6 => 0  -- Base case for 6
  | 7 => 0  -- Base case for 7
  | 8 => 0  -- Base case for 8
  | 9 => 0  -- Base case for 9
  | 10 => 0 -- Base case for 10
  | 11 => 0 -- Base case for 11
  | 12 => Real.log 12
  | (n+1) => Real.log (n + 1 + g n)

noncomputable def B : ℝ := g 2025

theorem B_interval : Real.log 2028 < B ∧ B < Real.log 2029 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_interval_l606_60646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construction_condition_l606_60689

structure RightTriangle where
  O : ℝ × ℝ
  A : ℝ × ℝ
  P : ℝ × ℝ
  is_right_angle : (O.1 - A.1) * (O.1 - P.1) + (O.2 - A.2) * (O.2 - P.2) = 0

noncomputable def angle (p q r : ℝ × ℝ) : ℝ := sorry

theorem construction_condition (t : RightTriangle) (B : ℝ × ℝ) :
  (∃ (P : ℝ × ℝ), t.P = P ∧ angle t.O t.A P = 2 * angle P B t.A) ↔
  (‖t.A - t.O‖ < ‖B - t.O‖ ∧ ‖t.A - P‖ = ‖t.A - B‖) :=
by
  sorry

#check construction_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_construction_condition_l606_60689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_and_b_l606_60647

noncomputable def A (a b : ℝ) : Set ℝ := {x | -2 < x ∧ x < -1 ∨ x > 1}
noncomputable def B (a b : ℝ) : Set ℝ := {x | a ≤ x ∧ x < b}

theorem find_a_and_b (a b : ℝ) :
  A a b ∪ B a b = {x | x > -2} ∧
  A a b ∩ B a b = {x | 1 < x ∧ x < 3} →
  a = -1 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_and_b_l606_60647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_f_when_a_is_1_range_of_a_when_f_increasing_l606_60657

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1/2 then x - 2/x else x^2 + 2*x + a - 1

-- Theorem for the zeros of f(x) when a = 1
theorem zeros_of_f_when_a_is_1 :
  ∀ x : ℝ, f 1 x = 0 ↔ x = Real.sqrt 2 ∨ x = 0 ∨ x = -2 :=
by sorry

-- Theorem for the range of a when f(x) is increasing on [-1, +∞)
theorem range_of_a_when_f_increasing :
  ∀ a : ℝ, (∀ x y : ℝ, -1 ≤ x ∧ x < y → f a x < f a y) ↔ a ≤ -15/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_f_when_a_is_1_range_of_a_when_f_increasing_l606_60657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_matrix_central_sum_l606_60601

/-- Represents a 4x4 matrix where each row and column forms an arithmetic sequence -/
def ArithmeticMatrix := Matrix (Fin 4) (Fin 4) ℚ

/-- Check if a sequence is arithmetic -/
def is_arithmetic_seq {α : Type*} [LinearOrder α] [Sub α] (s : ℕ → α) : Prop :=
  ∀ i j k : ℕ, j - i = k - j → s j - s i = s k - s j

/-- The property that each row and column of the matrix forms an arithmetic sequence -/
def is_arithmetic_matrix (M : ArithmeticMatrix) : Prop :=
  (∀ i : Fin 4, is_arithmetic_seq (λ j ↦ M i j)) ∧
  (∀ j : Fin 4, is_arithmetic_seq (λ i ↦ M i j))

/-- The sum of the two middle terms in the central 2x2 submatrix -/
def central_sum (M : ArithmeticMatrix) : ℚ := M 1 1 + M 2 2

theorem arithmetic_matrix_central_sum :
  ∀ M : ArithmeticMatrix,
  is_arithmetic_matrix M →
  M 0 0 = 3 →
  M 0 3 = 18 →
  M 3 0 = 4 →
  M 3 3 = 28 →
  central_sum M = 98 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_matrix_central_sum_l606_60601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l606_60628

/-- The sum of the infinite series Σ(3n + 2) / (n(n+1)(n+3)) for n from 1 to infinity -/
noncomputable def infiniteSeries : ℝ := ∑' n, (3 * n + 2) / (n * (n + 1) * (n + 3))

/-- Theorem stating that the infinite series sum is equal to 29/36 -/
theorem infiniteSeriesSum : infiniteSeries = 29 / 36 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l606_60628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l606_60636

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length
  (speed : ℝ)
  (time : ℝ)
  (bridge_length : ℝ)
  (h1 : speed = 36) -- Speed in km/h
  (h2 : time = 31.99744020478362) -- Time in seconds
  (h3 : bridge_length = 200) -- Bridge length in meters
  : ∃ (train_length : ℝ), abs (train_length - 119.9744020478362) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l606_60636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_points_in_region_l606_60635

-- Define the set of points satisfying the conditions
def S : Set (ℚ × ℚ) :=
  {p : ℚ × ℚ | p.1 > 0 ∧ p.2 > 0 ∧ p.1 + 2 * p.2 ≤ 6}

-- Theorem statement
theorem infinite_points_in_region : Set.Infinite S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_points_in_region_l606_60635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_increasing_interval_l606_60604

open Real

theorem sin_increasing_interval :
  ∃ (a b : ℝ), a < b ∧
  (∀ x y, a < x ∧ x < y ∧ y < b → sin (x - Real.pi/3) < sin (y - Real.pi/3)) ∧
  a = -Real.pi/6 ∧ b = 5*Real.pi/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_increasing_interval_l606_60604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l606_60624

noncomputable section

-- Define the vectors
def a (x : Real) : Real × Real := (Real.sin x, Real.sqrt 3 * Real.cos x)
def b : Real × Real := (-1, 1)
def c : Real × Real := (1, 1)

-- Define the parallel condition
def parallel (v w : Real × Real) : Prop :=
  ∃ (k : Real), v.1 * w.2 = k * v.2 * w.1

theorem vector_problem (x : Real) (h : x ∈ Set.Icc 0 Real.pi) :
  (parallel (a x + b) c → x = 5 * Real.pi / 6) ∧
  (a x • b = 1 / 2 → Real.sin (x + Real.pi / 6) = Real.sqrt 15 / 4) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l606_60624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_nondecreasing_functions_l606_60630

theorem count_nondecreasing_functions (n : ℕ) :
  let A := Finset.range n
  (Finset.filter
    (fun f : A → A =>
      (∀ x y : A, (f x).val - (f y).val ≤ x - y) ∧
      (∀ x y : A, x ≤ y → f x ≤ f y))
    (Finset.univ : Finset (A → A))).card =
  n * 2^(n-1) - (n-1) * 2^(n-2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_nondecreasing_functions_l606_60630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_sqrt_gt_sqrt_l606_60661

theorem sum_reciprocal_sqrt_gt_sqrt (n : ℕ) (h : n ≥ 2) :
  (Finset.range n).sum (λ i => 1 / Real.sqrt (i + 1 : ℝ)) > Real.sqrt n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_sqrt_gt_sqrt_l606_60661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l606_60676

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition for the triangle -/
def triangle_condition (t : Triangle) : Prop :=
  (Real.sin t.A / t.a = Real.sqrt 3 * Real.cos t.B / t.b) ∧ (t.b = 2)

/-- Area of a triangle -/
noncomputable def area (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.c * Real.sin t.B

/-- The theorem stating the properties of the triangle -/
theorem triangle_properties (t : Triangle) (h : triangle_condition t) :
  t.B = π / 3 ∧
  (∀ s : Triangle, triangle_condition s → area s ≤ Real.sqrt 3) ∧
  (∃ s : Triangle, triangle_condition s ∧ area s = Real.sqrt 3 ∧ s.a = s.b ∧ s.b = s.c) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l606_60676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_rectangle_area_is_11_l606_60656

noncomputable def original_rectangle_side_A : ℝ := 3
noncomputable def original_rectangle_side_B : ℝ := 4

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

def new_rectangle_length (a c : ℝ) : ℝ := c + 2 * a

def new_rectangle_width (a c : ℝ) : ℝ := |c - 2 * a|

def new_rectangle_area (l w : ℝ) : ℝ := l * w

theorem new_rectangle_area_is_11 :
  let c := hypotenuse original_rectangle_side_A original_rectangle_side_B
  let l := new_rectangle_length original_rectangle_side_A c
  let w := new_rectangle_width original_rectangle_side_A c
  new_rectangle_area l w = 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_rectangle_area_is_11_l606_60656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l606_60684

noncomputable def f (x : ℝ) : ℝ := x / (1 + x^2)

theorem problem_statement :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < 1 ∧ 0 < x₂ ∧ x₂ < 1 →
    (x₁ - x₂) * (f x₁ - f x₂) ≥ 0) ∧
  (∃ a : ℝ, a = 9/10 ∧
    ∀ x : ℝ, 0 < x ∧ x < 1 →
      (3*x^2 - x) / (1 + x^2) ≥ a * (x - 1/3)) ∧
  (∀ x₁ x₂ x₃ : ℝ, 
    x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₁ + x₂ + x₃ = 1 →
    ∃ y : ℝ, y = 0 ∧
      ∀ z : ℝ, z ≥ 0 ∧
        z = (3*x₁^2 - x₁) / (1 + x₁^2) + 
            (3*x₂^2 - x₂) / (1 + x₂^2) + 
            (3*x₃^2 - x₃) / (1 + x₃^2) →
        y ≤ z) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l606_60684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_row_10_equals_2_pow_10_l606_60618

def pascal_triangle : ℕ → ℕ → ℕ
| 0, _ => 1
| n+1, 0 => 1
| n+1, k+1 => pascal_triangle n k + pascal_triangle n (k+1)

def sum_row (n : ℕ) : ℕ :=
  (List.range (n+1)).map (pascal_triangle n) |> List.sum

theorem sum_row_10_equals_2_pow_10 :
  sum_row 10 = 2^10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_row_10_equals_2_pow_10_l606_60618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l606_60687

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := y^2 / 8 - x^2 / 6 = 1

-- Define the asymptote equation
def asymptote (x y : ℝ) : Prop := 2 * x - Real.sqrt 3 * y = 0

-- Theorem statement
theorem hyperbola_asymptote :
  ∀ ε > 0, ∃ x y : ℝ, hyperbola x y ∧ 
    abs (2 * x - Real.sqrt 3 * y) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l606_60687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_and_shape_l606_60639

/-- Represents a rhombus with given diagonal lengths -/
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ

/-- Calculates the area of a rhombus -/
noncomputable def area (r : Rhombus) : ℝ := (r.diagonal1 * r.diagonal2) / 2

/-- Checks if a rhombus is a square -/
def isSquare (r : Rhombus) : Prop := r.diagonal1 = r.diagonal2

theorem rhombus_area_and_shape (r : Rhombus) 
  (h1 : r.diagonal1 = 30) 
  (h2 : r.diagonal2 = 18) : 
  area r = 270 ∧ ¬isSquare r := by
  sorry

#check rhombus_area_and_shape

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_and_shape_l606_60639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_remainder_problem_l606_60698

theorem division_remainder_problem (n j : ℕ) 
  (hn : n > 0)
  (hj : j > 0)
  (h1 : (n : ℝ) / (j : ℝ) = 142.07)
  (h2 : (j : ℝ) = 400.000000000039) : 
  n % j = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_remainder_problem_l606_60698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_implies_a_eq_one_l606_60613

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 * x + a)

def is_tangent (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, (f a x₀ = 2 * x₀) ∧ 
            (deriv (f a) x₀ = 2)

theorem tangent_implies_a_eq_one :
  ∀ a : ℝ, is_tangent a → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_implies_a_eq_one_l606_60613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_coloring_exists_l606_60611

/-- Represents the color of a stone -/
inductive Color
| Black
| White

/-- Represents a regular 13-gon with stones at each vertex -/
structure RegularPolygon where
  stones : Fin 13 → Color

/-- Checks if a coloring is symmetric with respect to some axis -/
def is_symmetric (p : RegularPolygon) : Prop :=
  ∃ (axis : Fin 13), ∀ (i : Fin 13),
    p.stones i = p.stones ((2 * axis.val - i.val + 13) % 13)

/-- Represents the operation of exchanging two stones -/
def exchange (p : RegularPolygon) (i j : Fin 13) : RegularPolygon where
  stones := fun k => if k = i then p.stones j
                     else if k = j then p.stones i
                     else p.stones k

/-- The main theorem -/
theorem symmetric_coloring_exists (p : RegularPolygon) :
  ∃ (i j : Fin 13), is_symmetric (exchange p i j) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_coloring_exists_l606_60611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_ellipse_with_distance_l606_60670

noncomputable def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

def line_equation (k : ℝ) (x y : ℝ) : Prop := y = k*x + 2

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem line_through_ellipse_with_distance (k : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ,
    line_equation k x1 y1 ∧
    line_equation k x2 y2 ∧
    ellipse x1 y1 ∧
    ellipse x2 y2 ∧
    distance x1 y1 x2 y2 = 4) →
  k = Real.sqrt 2 / 4 ∨ k = -(Real.sqrt 2 / 4) :=
by
  sorry

#check line_through_ellipse_with_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_ellipse_with_distance_l606_60670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_top_block_exists_smallest_top_block_l606_60641

/-- Represents a pyramid of cubical blocks -/
structure BlockPyramid where
  bottom_layer : Finset ℕ
  second_layer : Finset ℕ
  third_layer : Finset ℕ
  top_block : ℕ

/-- Checks if a BlockPyramid is valid according to the problem conditions -/
def is_valid_pyramid (p : BlockPyramid) : Prop :=
  p.bottom_layer.card = 15 ∧
  p.second_layer.card = 9 ∧
  p.third_layer.card = 6 ∧
  (∀ n, n ∈ p.bottom_layer → 1 ≤ n ∧ n ≤ 15) ∧
  (∀ n, n ∈ p.second_layer → ∃ a b c, a ∈ p.bottom_layer ∧ b ∈ p.bottom_layer ∧ c ∈ p.bottom_layer ∧ n = a + b + c) ∧
  (∀ n, n ∈ p.third_layer → ∃ a b c, a ∈ p.second_layer ∧ b ∈ p.second_layer ∧ c ∈ p.second_layer ∧ n = a + b + c) ∧
  (∃ a b c, a ∈ p.third_layer ∧ b ∈ p.third_layer ∧ c ∈ p.third_layer ∧ p.top_block = a + b + c)

/-- The main theorem to prove -/
theorem smallest_top_block (p : BlockPyramid) :
  is_valid_pyramid p → p.top_block ≥ 119 := by
  sorry

/-- The smallest possible number for the top block is 119 -/
theorem exists_smallest_top_block :
  ∃ p : BlockPyramid, is_valid_pyramid p ∧ p.top_block = 119 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_top_block_exists_smallest_top_block_l606_60641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_arrangement_l606_60620

/-- A type representing a 600x600 table of 1 and -1 --/
def Table := Fin 600 → Fin 600 → Int

/-- Predicate that checks if a given value is either 1 or -1 --/
def IsValidEntry (n : Int) : Prop := n = 1 ∨ n = -1

/-- Predicate that checks if all entries in the table are valid --/
def IsValidTable (t : Table) : Prop :=
  ∀ i j, IsValidEntry (t i j)

/-- The sum of all entries in the table --/
def TableSum (t : Table) : Int :=
  Finset.sum (Finset.univ : Finset (Fin 600)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 600)) fun j =>
      t i j

/-- The sum of entries in a 4x6 or 6x4 rectangle starting at (i, j) --/
def RectangleSum (t : Table) (i j : Fin 600) (h w : Nat) : Int :=
  Finset.sum (Finset.range h) fun x =>
    Finset.sum (Finset.range w) fun y =>
      t ⟨i + x, sorry⟩ ⟨j + y, sorry⟩

/-- Predicate that checks if the table satisfies the rectangle sum condition --/
def SatisfiesRectangleCondition (t : Table) : Prop :=
  ∀ i j, (|RectangleSum t i j 4 6| > 4 ∧ |RectangleSum t i j 6 4| > 4)

theorem impossible_arrangement :
  ¬ ∃ t : Table, IsValidTable t ∧ 
    SatisfiesRectangleCondition t ∧ 
    |TableSum t| < 90000 := by
  sorry

#check impossible_arrangement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_arrangement_l606_60620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l606_60610

-- Define the hyperbola parameters
variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)

-- Define the parabola parameter
variable (p : ℝ)
variable (hp : p > 0)

-- Define the shared focus
variable (c : ℝ)

-- Define the intersection point of asymptote and parabola axis
noncomputable def intersection_point : ℝ × ℝ := (-5, -15/4)

-- State that the hyperbola and parabola share the same focus
axiom shared_focus : c = 5

-- State that the intersection point lies on the asymptote
axiom on_asymptote : intersection_point.2 = (3/4) * intersection_point.1

-- Define the eccentricity of the hyperbola
noncomputable def eccentricity : ℝ := c / a

-- Theorem to prove
theorem hyperbola_eccentricity : eccentricity = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l606_60610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inverse_point_l606_60627

def is_exponential (f : ℝ → ℝ) : Prop := ∃ a : ℝ, a > 0 ∧ ∀ x, f x = a^x

def inverse_passes_through (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  f p.snd = p.fst

theorem exponential_inverse_point (f : ℝ → ℝ) :
  is_exponential f → inverse_passes_through f (2, -1) → f = fun x ↦ (1/2)^x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inverse_point_l606_60627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_equality_l606_60608

open Real

theorem trig_expression_equality (c : ℝ) (h : c = π / 7) :
  (Real.sin (4 * c) * Real.sin (5 * c) * Real.cos (6 * c) * Real.sin (7 * c) * Real.sin (8 * c)) /
  (Real.sin (2 * c) * Real.sin (3 * c) * Real.sin (5 * c) * Real.sin (6 * c) * Real.sin (7 * c)) = -Real.cos c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_equality_l606_60608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_initial_choices_l606_60697

/-- Represents the friendship structure among sheep and the wolf -/
structure SheepGraph (n : ℕ) where
  edges : List (Fin n × Fin n)  -- list of friendships between sheep

/-- Represents the state of friendships between the wolf and sheep -/
def WolfFriends (n : ℕ) := Fin n → Bool

/-- Predicate to check if the wolf can eat all sheep given an initial friend choice -/
def canEatAllSheep {n : ℕ} (g : SheepGraph n) (initial : WolfFriends n) : Prop :=
  sorry -- Definition omitted for brevity

/-- The main theorem to prove -/
theorem max_initial_choices (n : ℕ) :
  ∃ (g : SheepGraph n),
    (∃ (m : ℕ), m = 2^(n-1) ∧
      (∀ (k : ℕ), k > m →
        ¬∃ (choices : Fin k → WolfFriends n),
          ∀ (i : Fin k), canEatAllSheep g (choices i))) ∧
    (∃ (choices : Fin (2^(n-1)) → WolfFriends n),
      ∀ (i : Fin (2^(n-1))), canEatAllSheep g (choices i)) :=
by
  sorry

#check max_initial_choices

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_initial_choices_l606_60697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_of_regular_quad_pyramid_l606_60667

/-- Regular quadrilateral pyramid -/
structure RegularQuadPyramid where
  /-- Side length of the square base -/
  a : ℝ
  /-- Angle between a lateral face and the base plane -/
  lateral_angle : ℝ

/-- The height of a regular quadrilateral pyramid -/
noncomputable def pyramid_height (p : RegularQuadPyramid) : ℝ :=
  p.a / 2

/-- Theorem: The height of a regular quadrilateral pyramid with base side length a
    and lateral face angle 45° with the base plane is a/2 -/
theorem height_of_regular_quad_pyramid (p : RegularQuadPyramid)
    (h_positive : p.a > 0)
    (h_angle : p.lateral_angle = π/4) :
  pyramid_height p = p.a / 2 := by
  -- Unfold the definition of pyramid_height
  unfold pyramid_height
  -- The definition directly gives us what we want to prove
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_of_regular_quad_pyramid_l606_60667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_c_inequality_l606_60694

theorem max_c_inequality (c : ℝ) : 
  (∀ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 → 
    x^3 + y^3 + z^3 - 3*x*y*z ≥ c * abs ((x-y)*(y-z)*(z-x))) ↔ 
  c ≤ (Real.sqrt 6 + 3 * Real.sqrt 2) / 2 * Real.rpow 3 (1/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_c_inequality_l606_60694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_prime_properties_l606_60662

noncomputable def f (x : ℝ) : ℝ := (1/2) * x + Real.sin x

noncomputable def f_prime (x : ℝ) : ℝ := (1/2) + Real.cos x

theorem f_prime_properties :
  (∀ x ∈ Set.Icc (-π/2) (π/2), f_prime (-x) = f_prime x) ∧
  (∃ x₀ ∈ Set.Icc (-π/2) (π/2), ∀ x ∈ Set.Icc (-π/2) (π/2), f_prime x ≤ f_prime x₀) ∧
  (¬∃ x₀ ∈ Set.Icc (-π/2) (π/2), ∀ x ∈ Set.Icc (-π/2) (π/2), f_prime x ≥ f_prime x₀) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_prime_properties_l606_60662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleTransformation_gcd_preserved_l606_60648

-- Define the type for our pair of natural numbers
def NumberPair := ℕ × ℕ

-- Define the operations
def swap : NumberPair → NumberPair
  | (m, n) => (n, m)

def sumFirst : NumberPair → NumberPair
  | (m, n) => (m + n, n)

def absDiff : NumberPair → NumberPair
  | (m, n) => (m, Int.natAbs (m - n))

-- Define a function that represents a sequence of operations
def applyOperations : List (NumberPair → NumberPair) → NumberPair → NumberPair
  | [], pair => pair
  | (op :: ops), pair => applyOperations ops (op pair)

-- The theorem to prove
theorem impossibleTransformation :
  ∀ (ops : List (NumberPair → NumberPair)),
    (∀ op ∈ ops, op = swap ∨ op = sumFirst ∨ op = absDiff) →
    applyOperations ops (901, 1219) ≠ (871, 1273) := by
  sorry

-- Helper theorem: GCD is preserved by the operations
theorem gcd_preserved (m n : ℕ) :
  Nat.gcd m n = Nat.gcd (swap (m, n)).1 (swap (m, n)).2 ∧
  Nat.gcd m n = Nat.gcd (sumFirst (m, n)).1 (sumFirst (m, n)).2 ∧
  Nat.gcd m n = Nat.gcd (absDiff (m, n)).1 (absDiff (m, n)).2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleTransformation_gcd_preserved_l606_60648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_m_bounds_l606_60688

theorem quadratic_roots_m_bounds (z₁ z₂ m : ℂ) :
  z₁^2 - 4 * z₂ = 16 + 20 * Complex.I →
  ∃ α β : ℂ, Complex.abs (α - β) = 2 * Real.sqrt 7 ∧
            α^2 + z₁ * α + z₂ + m = 0 ∧
            β^2 + z₁ * β + z₂ + m = 0 →
  (Complex.abs m ≤ Real.sqrt 41 + 7 ∧ 7 - Real.sqrt 41 ≤ Complex.abs m) ∧
  (∃ m₁ m₂ : ℂ, Complex.abs m₁ = Real.sqrt 41 + 7 ∧ Complex.abs m₂ = 7 - Real.sqrt 41) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_m_bounds_l606_60688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_fraction_l606_60674

open BigOperators

def Q (n : ℕ) : ℚ :=
  ∏ k in Finset.range (n - 2), (1 - 1 / (k + 3 : ℚ))

theorem product_equals_fraction :
  Q 2023 = 2 / 2023 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_fraction_l606_60674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fishing_company_profits_l606_60617

/-- Represents the fishing company's financial model --/
structure FishingCompany where
  initialCost : ℕ
  firstYearCost : ℕ
  annualCostIncrease : ℕ
  annualIncome : ℕ

/-- Calculate the total profit after n years --/
def totalProfit (company : FishingCompany) (n : ℕ) : ℤ :=
  n * company.annualIncome - (company.initialCost + company.firstYearCost * n + (n * (n - 1) * company.annualCostIncrease) / 2)

/-- Calculate the average annual profit after n years --/
def averageProfit (company : FishingCompany) (n : ℕ) : ℚ :=
  (totalProfit company n : ℚ) / n

/-- The main theorem about the fishing company's profits --/
theorem fishing_company_profits (company : FishingCompany) 
  (h_initial : company.initialCost = 980000)
  (h_first : company.firstYearCost = 120000)
  (h_increase : company.annualCostIncrease = 40000)
  (h_income : company.annualIncome = 500000) :
  (∃ n : ℕ, n = 10 ∧ totalProfit company n = 1020000) ∧
  (∃ n : ℕ, n = 9 ∧ averageProfit company n > 22222 ∧ averageProfit company n < 22223) := by
  sorry

#eval totalProfit 
  { initialCost := 980000
    firstYearCost := 120000
    annualCostIncrease := 40000
    annualIncome := 500000 }
  10

#eval averageProfit
  { initialCost := 980000
    firstYearCost := 120000
    annualCostIncrease := 40000
    annualIncome := 500000 }
  9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fishing_company_profits_l606_60617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_perpendicular_lines_l606_60653

noncomputable section

-- Define necessary structures and operations
variable (Point Line Circle : Type)
variable (tangent_at : Circle → Point → Line)
variable (intersect : Circle → Circle → Set Point)
variable (on_circle : Point → Circle → Prop)
variable (perpendicular : Line → Line → Prop)

theorem circle_intersection_perpendicular_lines 
  (P : Point) 
  (L M : Line) 
  (c₁ c₂ c₃ c₄ : Circle) 
  (A B C D : Point) :
  (∀ i j : Circle, i ≠ j → ¬(∀ p, on_circle p i → on_circle p j)) →
  (∀ i : Circle, on_circle P i) →
  (tangent_at c₁ P = L ∧ tangent_at c₂ P = L) →
  (tangent_at c₃ P = M ∧ tangent_at c₄ P = M) →
  (A ∈ intersect c₁ c₂ ∩ intersect c₃ c₄ ∧
   B ∈ intersect c₁ c₂ ∩ intersect c₃ c₄ ∧
   C ∈ intersect c₁ c₂ ∩ intersect c₃ c₄ ∧
   D ∈ intersect c₁ c₂ ∩ intersect c₃ c₄) →
  (∃ (c : Circle), on_circle A c ∧ on_circle B c ∧ on_circle C c ∧ on_circle D c) ↔ perpendicular L M :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_perpendicular_lines_l606_60653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l606_60612

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Calculates the distance from a point to a line -/
noncomputable def Line.distanceToPoint (l : Line) (x y : ℝ) : ℝ :=
  abs (l.a * x + l.b * y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Theorem statement for the given problem -/
theorem line_equation (l₁ l₂ l : Line) (A : ℝ × ℝ) :
  (∀ x y, l₁.contains x y ↔ 2*x + y - 8 = 0) →
  (∀ x y, l₂.contains x y ↔ x - 2*y + 1 = 0) →
  (∃ x y, l₁.contains x y ∧ l₂.contains x y ∧ l.contains x y) →
  (∃ a, ∀ x y, (l.contains x 0 ∧ l.contains 0 y) → x = a ∧ y = a) →
  l.contains A.fst A.snd →
  l.distanceToPoint 0 0 = 3 →
  ((∀ x y, l.contains x y ↔ 2*x - 3*y = 0) ∨
   (∀ x y, l.contains x y ↔ x + y - 5 = 0)) ∧
  ((∀ x y, l.contains x y ↔ x = -3) ∨
   (∀ x y, l.contains x y ↔ 5*x - 12*y + 39 = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l606_60612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_dividing_distances_l606_60621

/-- Triangle with sides 13, 14, and 15 divided into three equal parts -/
structure SpecialTriangle where
  -- Define the side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Ensure the triangle inequality holds
  tri_ineq : a + b > c ∧ b + c > a ∧ c + a > b
  -- Set the specific side lengths
  side_a : a = 13
  side_b : b = 14
  side_c : c = 15
  -- Ensure c is the longest side
  c_longest : c > a ∧ c > b

/-- The distances from the nearest vertices to the dividing lines -/
noncomputable def dividing_distances (t : SpecialTriangle) : ℝ × ℝ :=
  (Real.sqrt 33, Real.sqrt 42)

/-- The theorem stating the distances from the nearest vertices to the dividing lines -/
theorem special_triangle_dividing_distances (t : SpecialTriangle) :
  dividing_distances t = (Real.sqrt 33, Real.sqrt 42) := by
  -- Unfold the definition of dividing_distances
  unfold dividing_distances
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_dividing_distances_l606_60621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_mn_slippery_iff_coprime_l606_60632

/-- A function f is (m,n)-slippery if it satisfies certain properties. -/
def is_mn_slippery (f : ℝ → ℝ) (m n : ℕ+) : Prop :=
  ContinuousOn f (Set.Icc 0 m.val) ∧
  f 0 = 0 ∧ f m.val = n.val ∧
  ∀ t₁ t₂ : ℝ, 0 ≤ t₁ ∧ t₁ < t₂ ∧ t₂ ≤ m.val →
    (∃ k : ℤ, t₂ - t₁ = k) → (∃ k : ℤ, f t₂ - f t₁ = k) →
    t₂ - t₁ = 0 ∨ t₂ - t₁ = m.val

/-- There exists an (m,n)-slippery function if and only if gcd(m,n) = 1 -/
theorem exists_mn_slippery_iff_coprime (m n : ℕ+) :
  (∃ f : ℝ → ℝ, is_mn_slippery f m n) ↔ Nat.gcd m.val n.val = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_mn_slippery_iff_coprime_l606_60632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l606_60629

noncomputable def a : ℝ := 3^(3/10)
noncomputable def b : ℝ := 2^(21/10)
noncomputable def c : ℝ := 2 * Real.log 2 / Real.log 5

theorem abc_inequality : c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l606_60629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_four_l606_60602

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℚ
  r : ℚ
  first_term : a 1 = -1
  fourth_term : a 4 = 27
  seq_def : ∀ n : ℕ, a n = a 1 * r^(n-1)

/-- The sum of the first n terms of a geometric sequence -/
def geometricSum (s : GeometricSequence) (n : ℕ) : ℚ :=
  s.a 1 * (1 - s.r^n) / (1 - s.r)

theorem geometric_sum_four (s : GeometricSequence) : 
  geometricSum s 4 = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_four_l606_60602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_condition_l606_60659

def nextTerm (a : ℕ) : ℕ := 
  if a % 2 = 0 then a / 2 else 3 * a + 1

def satisfiesCondition (a : ℕ) : Bool :=
  let a2 := nextTerm a
  let a3 := nextTerm a2
  let a4 := nextTerm a3
  a < a2 ∧ a < a3 ∧ a < a4

def countSatisfying : ℕ := 
  (Finset.range 2501).filter (fun x => satisfiesCondition x) |>.card

theorem count_satisfying_condition : countSatisfying = 625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_condition_l606_60659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_cubic_roots_l606_60649

-- Define the polynomial
noncomputable def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6/5

-- Define the roots
noncomputable def u : ℝ := Real.sqrt 3
noncomputable def v : ℝ := Real.sqrt 3
noncomputable def w : ℝ := Real.sqrt 3

-- State the theorem
theorem triangle_area_from_cubic_roots :
  (f u = 0) ∧ (f v = 0) ∧ (f w = 0) →
  ∃ (A : ℝ), A > 0 ∧ A^2 = (u + v + w)/2 * ((u + v + w)/2 - u) * ((u + v + w)/2 - v) * ((u + v + w)/2 - w) ∧ A = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_cubic_roots_l606_60649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_integral_value_l606_60668

open MeasureTheory

theorem min_integral_value (f : ℝ → ℝ) (hf : Continuous f) 
  (h : ∀ x, f (x - 1) + f (x + 1) ≥ x + f x) : 
  ∫ x in Set.Icc 1 2005, f x ≥ 2010012 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_integral_value_l606_60668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l606_60634

-- Define set A
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 2}

-- Define set B
def B : Set ℝ := {1, 2, 3, 4}

-- Theorem statement
theorem intersection_A_B :
  A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l606_60634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numerator_and_denominator_for_0_45_repeating_l606_60626

def repeating_decimal_to_fraction (a : ℕ) : ℚ :=
  (a : ℚ) / (99 : ℚ)

def is_lowest_terms (q : ℚ) : Prop :=
  ∀ (n d : ℕ), q = n / d → Nat.gcd n d = 1

theorem sum_of_numerator_and_denominator_for_0_45_repeating :
  ∃ (n d : ℕ), repeating_decimal_to_fraction 45 = n / d ∧
                is_lowest_terms (n / d) ∧
                n + d = 16 := by
  sorry

#eval repeating_decimal_to_fraction 45

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numerator_and_denominator_for_0_45_repeating_l606_60626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_g_l606_60660

noncomputable def f (x : Real) : Real := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)

noncomputable def g (x : Real) : Real := f (x - Real.pi / 6)

theorem center_of_symmetry_g :
  ∀ x : Real, g (Real.pi / 2 + x) = g (Real.pi / 2 - x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_g_l606_60660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l606_60616

theorem coefficient_x_cubed_in_expansion : 
  let f : Polynomial ℤ := X^2 * (X - 2)^6
  f.coeff 3 = -192 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l606_60616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_from_two_unit_cubes_l606_60652

theorem cube_edge_from_two_unit_cubes :
  ∀ (v : ℝ), v > 0 →
  ∃ (c : ℝ), c > 0 ∧ c^3 = 2 * v ∧ c = (2 : ℝ) ^ (1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_from_two_unit_cubes_l606_60652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_randy_tractors_l606_60675

theorem farmer_randy_tractors : 
  let total_acres : ℕ := 1700
  let total_days : ℕ := 5
  let first_crew_days : ℕ := 2
  let second_crew_tractors : ℕ := 7
  let second_crew_days : ℕ := 3
  let acres_per_tractor_per_day : ℕ := 68
  let first_crew_tractors : ℕ := 2
  (first_crew_tractors * first_crew_days + second_crew_tractors * second_crew_days) * acres_per_tractor_per_day = total_acres := by
  -- Proof goes here
  sorry

#check farmer_randy_tractors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_randy_tractors_l606_60675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_fraction_equality_l606_60678

theorem cube_root_fraction_equality :
  (8 / 20.25 : ℝ) ^ (1/3 : ℝ) = (2 * (2 : ℝ) ^ (1/3 : ℝ)^2) / (3 * (3 : ℝ) ^ (1/3 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_fraction_equality_l606_60678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_term_of_geometric_sequence_l606_60619

/-- A geometric sequence with first term a and common ratio r. -/
def geometric_sequence (a : ℝ) (r : ℝ) : ℕ → ℝ := λ n ↦ a * r^(n - 1)

/-- Theorem: In a geometric sequence where the first term is 1/2 and the fifth term is 16, the third term is 2. -/
theorem third_term_of_geometric_sequence :
  ∀ (r : ℝ),
  (geometric_sequence (1/2) r 1 = 1/2) →
  (geometric_sequence (1/2) r 5 = 16) →
  (geometric_sequence (1/2) r 3 = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_term_of_geometric_sequence_l606_60619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_angle_measure_l606_60669

-- Define the triangle ABC
structure Triangle (V : Type) where
  A : V
  B : V
  C : V

-- Define the property of being isosceles
def isIsosceles {V : Type} (t : Triangle V) (dist : V → V → ℝ) : Prop :=
  dist t.A t.B = dist t.B t.C

-- Define the angle measure in degrees
def angleMeasure {V : Type} (t : Triangle V) (angle : V → V → V → ℝ) : V → V → V → ℝ :=
  angle

-- Theorem statement
theorem isosceles_triangle_angle_measure {V : Type} 
  (t : Triangle V) (dist : V → V → ℝ) (angle : V → V → V → ℝ) (t_deg : ℝ) :
  isIsosceles t dist →
  angleMeasure t angle t.B t.A t.C = t_deg →
  angleMeasure t angle t.A t.B t.C = 180 - 2 * t_deg := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_angle_measure_l606_60669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_negative_values_l606_60614

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x > f y

theorem range_of_negative_values
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_decreasing : decreasing_on f (Set.Ici 0))
  (h_zero : f 2 = 0) :
  {x : ℝ | f x < 0} = Set.Ioo (-2) 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_negative_values_l606_60614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertically_opposite_angles_not_imply_parallel_corresponding_angles_imply_parallel_alternate_angles_imply_parallel_consecutive_interior_angles_imply_parallel_l606_60623

-- Define the concept of two lines
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define the concept of parallel lines
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define the concept of angles between lines
noncomputable def angle (l1 l2 : Line) : ℝ :=
  Real.arctan ((l2.slope - l1.slope) / (1 + l1.slope * l2.slope))

-- Define corresponding angles
def corresponding_angles_equal (l1 l2 : Line) : Prop :=
  ∃ (a : ℝ), angle l1 l2 = a ∧ angle l2 l1 = a

-- Define alternate angles
def alternate_angles_equal (l1 l2 : Line) : Prop :=
  ∃ (a : ℝ), angle l1 l2 = a ∧ angle l2 l1 = -a

-- Define consecutive interior angles
def consecutive_interior_angles_supplementary (l1 l2 : Line) : Prop :=
  ∃ (a b : ℝ), angle l1 l2 = a ∧ angle l2 l1 = b ∧ a + b = Real.pi

-- Define vertically opposite angles
def vertically_opposite_angles_equal (l1 l2 : Line) : Prop :=
  ∃ (a : ℝ), angle l1 l2 = a ∧ angle l2 l1 = a

-- Theorem stating that vertically opposite angles being equal does not imply parallel lines
theorem vertically_opposite_angles_not_imply_parallel :
  ¬(∀ (l1 l2 : Line), vertically_opposite_angles_equal l1 l2 → parallel l1 l2) := by
  sorry

-- Theorems stating that the other conditions imply parallel lines
theorem corresponding_angles_imply_parallel :
  ∀ (l1 l2 : Line), corresponding_angles_equal l1 l2 → parallel l1 l2 := by
  sorry

theorem alternate_angles_imply_parallel :
  ∀ (l1 l2 : Line), alternate_angles_equal l1 l2 → parallel l1 l2 := by
  sorry

theorem consecutive_interior_angles_imply_parallel :
  ∀ (l1 l2 : Line), consecutive_interior_angles_supplementary l1 l2 → parallel l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertically_opposite_angles_not_imply_parallel_corresponding_angles_imply_parallel_alternate_angles_imply_parallel_consecutive_interior_angles_imply_parallel_l606_60623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_from_sin_cos_sum_l606_60665

theorem tan_sum_from_sin_cos_sum (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 120 / 169)
  (h2 : Real.cos x + Real.cos y = 119 / 169) :
  Real.tan x + Real.tan y = -3406440 / 28441 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_from_sin_cos_sum_l606_60665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_gain_percentage_l606_60637

noncomputable def potato_weight : ℝ := 900 / 1000
noncomputable def onion_weight : ℝ := 850 / 1000
noncomputable def carrot_weight : ℝ := 950 / 1000

def potato_sold : ℝ := 10
def onion_sold : ℝ := 15
def carrot_sold : ℝ := 25

noncomputable def actual_potato_weight : ℝ := potato_weight * potato_sold
noncomputable def actual_onion_weight : ℝ := onion_weight * onion_sold
noncomputable def actual_carrot_weight : ℝ := carrot_weight * carrot_sold

noncomputable def total_claimed_weight : ℝ := potato_sold + onion_sold + carrot_sold
noncomputable def total_actual_weight : ℝ := actual_potato_weight + actual_onion_weight + actual_carrot_weight

noncomputable def gain_weight : ℝ := total_claimed_weight - total_actual_weight

theorem dealer_gain_percentage : 
  ∃ ε > 0, |((gain_weight / total_actual_weight) * 100) - 9.89| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_gain_percentage_l606_60637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_n_zero_iff_n_odd_l606_60696

/-- Sequence c_n defined as the sum of n-th powers of 8 real numbers -/
def c (a : Fin 8 → ℝ) (n : ℕ) : ℝ :=
  (Finset.univ.sum fun i => (a i) ^ n)

/-- Theorem stating that c_n is zero if and only if n is odd, given specific conditions on a_i -/
theorem c_n_zero_iff_n_odd (a : Fin 8 → ℝ) 
    (h1 : ∃ i, a i ≠ 0)
    (h2 : a 1 = -a 0)
    (h3 : a 3 = -a 2)
    (h4 : a 5 = -a 4)
    (h5 : a 7 = -a 6) :
    ∀ n : ℕ, n > 0 → (c a n = 0 ↔ n % 2 = 1) := by
  sorry

#check c_n_zero_iff_n_odd

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_n_zero_iff_n_odd_l606_60696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_outcomes_exam_outcomes_eq_number_of_outcomes_l606_60672

/-- The number of possible outcomes for n people taking an exam where each person can either pass or fail is 2^n. -/
theorem exam_outcomes (n : ℕ) : (2 : ℕ) ^ n = 2 ^ n :=
  by
  -- The proof is trivial as we're stating that 2^n equals itself
  rfl

/-- Helper function to represent the number of outcomes -/
def number_of_outcomes (n : ℕ) : ℕ := 2 ^ n

/-- The main theorem connecting the exam outcomes to the number of outcomes -/
theorem exam_outcomes_eq_number_of_outcomes (n : ℕ) : 
  (2 : ℕ) ^ n = number_of_outcomes n :=
  by
  -- Unfold the definition of number_of_outcomes
  unfold number_of_outcomes
  -- The equality is now trivial
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_outcomes_exam_outcomes_eq_number_of_outcomes_l606_60672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turn_over_four_best_l606_60631

-- Define the type for card sides
inductive CardSide
| Letter (c : Char)
| Number (n : Nat)

-- Define a card as a pair of sides
def Card := (CardSide × CardSide)

-- Define a function to check if a character is a consonant
def is_consonant (c : Char) : Bool :=
  c.isAlpha && !(['a', 'e', 'i', 'o', 'u'].contains c.toLower)

-- Define a function to check if a number is odd
def is_odd (n : Nat) : Bool :=
  n % 2 = 1

-- Tom's claim
def tom_claim (card : Card) : Prop :=
  match card with
  | (CardSide.Letter c, CardSide.Number n) =>
      is_consonant c → is_odd n
  | (CardSide.Number n, CardSide.Letter c) =>
      is_consonant c → is_odd n
  | _ => True

-- Define the set of visible card sides
def visible_sides : List CardSide :=
  [CardSide.Letter 'A', CardSide.Letter 'B', CardSide.Number 4,
   CardSide.Number 7, CardSide.Number 8, CardSide.Number 5]

-- Define a membership instance for CardSide in Card
instance : Membership CardSide Card where
  mem s c := s = c.1 ∨ s = c.2

-- Theorem stating that turning over the card with 4 is the best choice
theorem turn_over_four_best (cards : List Card)
  (h1 : ∀ c ∈ cards, tom_claim c)
  (h2 : ∃ c ∈ cards, (CardSide.Number 4) ∈ c)
  (h3 : ∀ s ∈ visible_sides, ∃ c ∈ cards, s ∈ c) :
  ∃ c ∈ cards, (CardSide.Number 4) ∈ c ∧
    (∃ letter, (CardSide.Letter letter) ∈ c ∧ is_consonant letter) →
    ¬(∀ c ∈ cards, tom_claim c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_turn_over_four_best_l606_60631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l606_60671

noncomputable section

/-- The function f as described in the problem -/
def f (A ω x : ℝ) : ℝ := A * Real.sin (ω * x + Real.pi / 6)

/-- The function g as described in the problem -/
def g (B : ℝ) : ℝ := Real.sqrt 3 * f 2 2 B + f 2 2 (B + Real.pi / 4)

theorem range_of_g (A B C : ℝ) (a b c : ℝ) :
  A > 0 → 
  f A 2 0 = 2 → 
  f A 2 (Real.pi / 2) = -2 → 
  2 * Real.sin A * Real.sin C + Real.cos (2 * B) = 1 → 
  0 ≤ g B ∧ g B ≤ 4 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l606_60671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deck_area_difference_l606_60679

/-- Represents the dimensions of a rectangular pool -/
structure PoolDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangular pool -/
noncomputable def poolPerimeter (d : PoolDimensions) : ℝ := 2 * (d.length + d.width)

/-- Calculates the area of a rectangular deck surrounding a pool -/
noncomputable def rectangularDeckArea (d : PoolDimensions) (deckWidth : ℝ) : ℝ :=
  (d.length + 2 * deckWidth) * (d.width + 2 * deckWidth) - d.length * d.width

/-- Calculates the area of a square deck with the same perimeter as a rectangular deck -/
noncomputable def squareDeckArea (d : PoolDimensions) (deckWidth : ℝ) : ℝ :=
  let sideLength := (poolPerimeter d + 8 * deckWidth) / 4
  sideLength * sideLength - d.length * d.width

/-- Theorem stating the difference in area between square and rectangular decks -/
theorem deck_area_difference (d : PoolDimensions) (deckWidth : ℝ) :
  d.length = 60 ∧ d.width = 20 →
  squareDeckArea d deckWidth - rectangularDeckArea d deckWidth = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_deck_area_difference_l606_60679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_with_conditions_l606_60633

theorem count_divisors_with_conditions : ∃! n : ℕ, 
  n = (Finset.filter (fun a => 2 ∣ a ∧ a ∣ 18 ∧ 0 < a ∧ a ≤ 10) (Finset.range 11)).card ∧ 
  n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_with_conditions_l606_60633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_range_l606_60685

theorem triangle_side_ratio_range (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- angles are positive
  A + B + C = Real.pi ∧  -- sum of angles in a triangle
  A < Real.pi/2 ∧ B < Real.pi/2 ∧ C < Real.pi/2 ∧  -- acute-angled triangle
  C = 2 * B ∧  -- given condition
  c / Real.sin C = b / Real.sin B  -- law of sines
  →
  Real.sqrt 2 < c/b ∧ c/b < Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_range_l606_60685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l606_60600

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- The distance from the center to a focus of the hyperbola -/
noncomputable def Hyperbola.c (h : Hyperbola) : ℝ := Real.sqrt (h.a^2 + h.b^2)

/-- The right focus of the hyperbola -/
noncomputable def Hyperbola.right_focus (h : Hyperbola) : ℝ × ℝ := (h.c, 0)

/-- The left vertex of the hyperbola -/
def Hyperbola.left_vertex (h : Hyperbola) : ℝ × ℝ := (-h.a, 0)

/-- The right vertex of the hyperbola -/
def Hyperbola.right_vertex (h : Hyperbola) : ℝ × ℝ := (h.a, 0)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem about the specific hyperbola -/
theorem hyperbola_theorem (h : Hyperbola)
    (h_left_focus : distance h.left_vertex h.right_focus = 3)
    (h_right_focus : distance h.right_vertex h.right_focus = 1) :
    (h.a = 1 ∧ h.b = Real.sqrt 3) ∧
    (∀ k : ℝ, k ≠ 0 → ∀ x y : ℝ,
      (y - 1 = k * (x - 1) ∧ x^2 - y^2 / 3 = 1) →
      (x + 1) / 2 ≠ 1 ∨ (y + 1) / 2 ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l606_60600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_crooks_in_cabinet_l606_60673

-- Define is_crook as a predicate
def is_crook : ℕ → Prop := sorry

theorem min_crooks_in_cabinet (total_ministers : ℕ) (h1 : total_ministers = 100) 
  (h2 : ∀ (s : Finset ℕ), s.card = 10 → ∃ i ∈ s, is_crook i) : 
  ∃ (crooks : Finset ℕ), crooks.card = 91 ∧ 
    (∀ (s : Finset ℕ), s.card = 10 → ∃ i ∈ s, i ∈ crooks) ∧
    (∀ (c : Finset ℕ), c.card < 91 → 
      ∃ (s : Finset ℕ), s.card = 10 ∧ ∀ i ∈ s, i ∉ c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_crooks_in_cabinet_l606_60673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_constructible_l606_60691

/-- Represents a triangle with one known angle, one known side, and either circumcircle or incircle radius --/
structure TriangleConstruction where
  α : Real  -- Known angle in radians
  c : Real  -- Known side length
  r : Option Real  -- Circumcircle radius (if given)
  ρ : Option Real  -- Incircle radius (if given)

/-- Checks if a triangle can be constructed given the parameters --/
def isConstructible (t : TriangleConstruction) : Prop :=
  match t.r, t.ρ with
  | some r, none =>
    t.c ≤ 2 * r ∧ 
    ∃ γ : Real, 0 < γ ∧ γ < Real.pi ∧ t.α + γ < Real.pi
  | none, some ρ =>
    ∃ B C : Real × Real, 
      let A := (0, 0)
      let O := (ρ * Real.cos (t.α / 2), ρ * Real.sin (t.α / 2))
      let AB := (t.c * Real.cos t.α, t.c * Real.sin t.α)
      (B.1 - O.1)^2 + (B.2 - O.2)^2 = ρ^2 ∧
      (C.1 - O.1)^2 + (C.2 - O.2)^2 = ρ^2 ∧
      (C.1 - A.1) * (B.2 - A.2) = (C.2 - A.2) * (B.1 - A.1)
  | _, _ => False

/-- Theorem stating that a triangle can be constructed under the given conditions --/
theorem triangle_constructible (t : TriangleConstruction) : 
  isConstructible t → ∃ A B C : Real × Real, 
    let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
    let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
    let CA := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
    AB = t.c ∧ 
    Real.arccos ((AB^2 + CA^2 - BC^2) / (2 * AB * CA)) = t.α ∧
    (t.r.isSome → ∀ r, t.r = some r → Real.sqrt ((AB * BC * CA) / (4 * (AB + BC + CA) * (BC + CA - AB) * (CA + AB - BC) * (AB + BC - CA))) = r) ∧
    (t.ρ.isSome → ∀ ρ, t.ρ = some ρ → (AB + BC + CA) / 2 * ρ = AB * BC * CA / (4 * (AB + BC + CA))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_constructible_l606_60691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_C_l606_60683

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real

-- State the theorem
theorem cosine_of_angle_C (t : Triangle) 
  (h1 : Real.cos t.A = 5/13) 
  (h2 : Real.sin t.B = 4/5) : 
  Real.cos t.C = 33/65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_C_l606_60683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_for_linear_inequality_l606_60690

/-- A function to represent the left-hand side of the inequality -/
def f (m : ℤ) (x : ℝ) : ℝ := (m + 1 : ℝ) * x^(m.natAbs) + 2

/-- Definition of a linear function in x -/
def is_linear_in_x (m : ℤ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f m x = a * x + b ∧ a ≠ 0

/-- Theorem stating that m = 1 is the only value satisfying the condition -/
theorem unique_m_for_linear_inequality :
  ∃! m : ℤ, is_linear_in_x m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_for_linear_inequality_l606_60690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pyramid_surface_area_l606_60692

/-- A square pyramid with base side length 3 and height 4 -/
structure SquarePyramid where
  /-- The side length of the square base -/
  baseSide : ℝ
  /-- The length of the edge from the apex to a base vertex -/
  apexToBase : ℝ
  /-- The base side length is 3 -/
  base_side_eq : baseSide = 3
  /-- The apex to base length is 4 -/
  apex_to_base_eq : apexToBase = 4
  /-- The lateral faces are perpendicular to the base -/
  lateral_faces_perpendicular : apexToBase ^ 2 = baseSide ^ 2 + (apexToBase ^ 2 - baseSide ^ 2)

/-- The total surface area of the square pyramid -/
noncomputable def totalSurfaceArea (p : SquarePyramid) : ℝ :=
  p.baseSide ^ 2 + 2 * p.baseSide * Real.sqrt (p.apexToBase ^ 2 - p.baseSide ^ 2)

theorem square_pyramid_surface_area (p : SquarePyramid) :
  totalSurfaceArea p = 9 + 6 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pyramid_surface_area_l606_60692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_of_three_l606_60625

-- Define the function f as noncomputable due to its dependency on Real.sqrt
noncomputable def f (x : ℝ) : ℝ :=
  if x > 5 then Real.sqrt (x + 1)
  else x^2 + 1

-- State the theorem
theorem f_composition_of_three : f (f (f 3)) = Real.sqrt (Real.sqrt 11 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_of_three_l606_60625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_2008_l606_60680

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def ArithmeticSequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def SumArithmeticSequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_2008 (d : ℝ) :
  let a₁ : ℝ := -2008
  let S := SumArithmeticSequence a₁ d
  (S 12 / 12 - S 10 / 10 = 2) →
  (S 2008 = -2008) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_2008_l606_60680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_theorem_l606_60622

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with vertex and focus -/
structure Parabola where
  vertex : Point
  focus : Point

/-- Check if a point is in the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is on the parabola -/
def isOnParabola (para : Parabola) (p : Point) : Prop :=
  p.x^2 = 4 * (p.y - para.vertex.y)

/-- The main theorem -/
theorem parabola_point_theorem (para : Parabola) (p : Point) : 
  para.vertex = ⟨0, 0⟩ →
  para.focus = ⟨0, 1⟩ →
  p = ⟨20, 100⟩ →
  isInFirstQuadrant p ∧
  isOnParabola para p ∧
  distance p para.focus = 101 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_theorem_l606_60622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_one_l606_60682

-- Define the functions f and g
def f (t : ℝ) : ℝ := t^2 + 1

noncomputable def g (t : ℝ) : ℝ := t + Real.log t

-- Define the distance function h
noncomputable def h (t : ℝ) : ℝ := |f t - g t|

-- State the theorem
theorem min_distance_is_one :
  ∃ (t₀ : ℝ), t₀ > 0 ∧ h t₀ = 1 ∧ ∀ (t : ℝ), t > 0 → h t ≥ 1 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_one_l606_60682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_negative_product_l606_60606

def A : Finset Int := {0, 1, -3, 6, -8, -10, 5, 12, -13}
def B : Finset Int := {-1, 2, -4, 7, 6, -9, 8, -11, 10}

def negative_product (a b : Int) : Bool :=
  a * b < 0

theorem probability_negative_product :
  let total_pairs := (A.filter (· ≠ 0)).card * B.card
  let negative_pairs := ((A.filter (· ≠ 0)).filter (λ a ↦ ∃ b ∈ B, negative_product a b)).card * B.card
  (negative_pairs : ℚ) / total_pairs = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_negative_product_l606_60606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_real_root_set_l606_60643

/-- The polynomial in question -/
def P (a x : ℝ) : ℝ := x^4 + a*x^3 + x^2 + a*x - 1

/-- The set of all real values of a for which the polynomial has at least one real root -/
def A : Set ℝ := {a : ℝ | ∃ x : ℝ, P a x = 0}

/-- Theorem stating that A is equal to the interval (-∞, -1.5] -/
theorem polynomial_real_root_set : A = Set.Iic (-3/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_real_root_set_l606_60643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_implies_product_l606_60664

theorem power_sum_implies_product (x : ℝ) : (2:ℝ)^x + (2:ℝ)^x + (2:ℝ)^x = 256 → x * (x + 1) = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_implies_product_l606_60664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_continuity_l606_60655

noncomputable def f (x : ℝ) : ℝ := (4 : ℝ) ^ (1 / (3 - x))

theorem f_continuity :
  ContinuousAt f 1 ∧ ¬ContinuousAt f 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_continuity_l606_60655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_class_females_l606_60677

/-- Represents the number of students in each class -/
structure ClassComposition where
  males : Nat
  females : Nat

/-- The dancing problem setup -/
structure DancingProblem where
  class1 : ClassComposition
  class2 : ClassComposition
  class3_males : Nat
  unmatched_students : Nat

def problem : DancingProblem :=
  { class1 := { males := 17, females := 13 }
  , class2 := { males := 14, females := 18 }
  , class3_males := 15
  , unmatched_students := 2
  }

theorem third_class_females :
  ∃ (class3_females : Nat),
    problem.class1.males + problem.class2.males + problem.class3_males -
    (problem.class1.females + problem.class2.females + class3_females) =
    problem.unmatched_students ∧
    class3_females = 13 := by
  sorry

#check third_class_females

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_class_females_l606_60677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_consecutive_integers_l606_60650

theorem sum_of_consecutive_integers : ∃ (a : ℕ), 
  (List.range 150).foldl (λ sum i => sum + (a + i)) 0 = 1627395075 ∧ 
  (∀ (b : ℕ), (List.range 150).foldl (λ sum i => sum + (b + i)) 0 ≠ 2345679325) ∧
  (∀ (c : ℕ), (List.range 150).foldl (λ sum i => sum + (c + i)) 0 ≠ 3579112475) ∧
  (∀ (d : ℕ), (List.range 150).foldl (λ sum i => sum + (d + i)) 0 ≠ 4692582625) ∧
  (∀ (e : ℕ), (List.range 150).foldl (λ sum i => sum + (e + i)) 0 ≠ 5815938775) :=
by
  -- The proof goes here
  sorry

#check sum_of_consecutive_integers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_consecutive_integers_l606_60650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_price_achieves_target_and_maximizes_sales_l606_60651

/-- Represents the pricing and sales information for a product --/
structure ProductInfo where
  purchasePrice : ℚ
  initialSellingPrice : ℚ
  initialDailySales : ℚ
  priceReductionStep : ℚ
  salesIncreasePerStep : ℚ
  targetDailyProfit : ℚ

/-- Calculates the optimal selling price to achieve the target daily profit while maximizing sales --/
noncomputable def optimalSellingPrice (info : ProductInfo) : ℚ :=
  let x := (info.initialSellingPrice - info.purchasePrice) / 2
  info.initialSellingPrice - 2 * x

/-- Theorem stating that the optimal selling price achieves the target daily profit and maximizes sales --/
theorem optimal_price_achieves_target_and_maximizes_sales (info : ProductInfo)
  (h_purchase : info.purchasePrice = 190)
  (h_initial_price : info.initialSellingPrice = 210)
  (h_initial_sales : info.initialDailySales = 8)
  (h_price_step : info.priceReductionStep = 2)
  (h_sales_increase : info.salesIncreasePerStep = 4)
  (h_target_profit : info.targetDailyProfit = 280) :
  let optimalPrice := optimalSellingPrice info
  (optimalPrice - info.purchasePrice) * (info.initialDailySales + (info.initialSellingPrice - optimalPrice) / info.priceReductionStep * info.salesIncreasePerStep) = info.targetDailyProfit ∧
  optimalPrice = 200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_price_achieves_target_and_maximizes_sales_l606_60651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_share_price_increase_l606_60645

/-- Calculates the percent increase between two quarters given their percent increases from the start of the year -/
noncomputable def percent_increase_between_quarters (first_quarter_increase : ℝ) (second_quarter_increase : ℝ) : ℝ :=
  ((1 + second_quarter_increase) - (1 + first_quarter_increase)) / (1 + first_quarter_increase) * 100

theorem share_price_increase (first_quarter_increase second_quarter_increase : ℝ) 
  (h1 : first_quarter_increase = 0.2) 
  (h2 : second_quarter_increase = 0.5) : 
  percent_increase_between_quarters first_quarter_increase second_quarter_increase = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_share_price_increase_l606_60645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sophie_cupcake_price_l606_60609

/-- The price of a cupcake given Sophie's purchase -/
noncomputable def cupcake_price : ℚ :=
  let doughnut_price : ℚ := 1
  let pie_slice_price : ℚ := 2
  let cookie_price : ℚ := 6/10
  let total_spend : ℚ := 33
  let num_cupcakes : ℕ := 5
  let num_doughnuts : ℕ := 6
  let num_pie_slices : ℕ := 4
  let num_cookies : ℕ := 15
  (total_spend - (num_doughnuts * doughnut_price + num_pie_slices * pie_slice_price + num_cookies * cookie_price)) / num_cupcakes

theorem sophie_cupcake_price : cupcake_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sophie_cupcake_price_l606_60609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_binomial_sum_six_satisfies_equation_largest_n_is_six_l606_60666

theorem largest_n_binomial_sum (n : ℕ) : 
  (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) → n ≤ 6 :=
by sorry

theorem six_satisfies_equation : 
  Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 6 :=
by sorry

theorem largest_n_is_six : 
  ∃ (n : ℕ), Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n ∧ 
  ∀ (m : ℕ), Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_binomial_sum_six_satisfies_equation_largest_n_is_six_l606_60666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_determine_l606_60693

def T : Set (ℤ × ℤ × ℤ) := {(x, y, z) | 0 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ 0 ≤ z ∧ z ≤ 9}

def response (x y z a b c : ℤ) : ℤ :=
  |x + y - a - b| + |y + z - b - c| + |z + x - c - a|

def canDetermine (n : ℕ) : Prop :=
  ∀ (x y z : ℤ), (x, y, z) ∈ T →
    ∃ (moves : Fin n → ℤ × ℤ × ℤ),
      (∀ i, moves i ∈ T) ∧
      (∀ (x' y' z' : ℤ), (x', y', z') ∈ T →
        (∀ i, response x y z (moves i).1 (moves i).2.1 (moves i).2.2 =
               response x' y' z' (moves i).1 (moves i).2.1 (moves i).2.2) →
        x = x' ∧ y = y' ∧ z = z')

theorem min_moves_to_determine : (canDetermine 3) ∧ ¬(canDetermine 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_determine_l606_60693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_is_53_l606_60663

/-- A convex quadrilateral with specific properties -/
structure SpecialQuadrilateral where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  PQ_length : ℝ
  angle_P : ℝ
  sides : List ℝ
  x : ℝ

/-- Properties of the special quadrilateral -/
def is_special_quadrilateral (quad : SpecialQuadrilateral) : Prop :=
  quad.PQ_length = 24 ∧
  quad.angle_P = 45 ∧
  -- We'll need to define these functions or use alternatives from Mathlib
  true ∧ -- Placeholder for: is_parallel (line_through quad.P quad.Q) (line_through quad.R quad.S)
  true ∧ -- Placeholder for: is_geometric_progression quad.sides
  quad.PQ_length = quad.sides.maximum ∧
  quad.x ∈ quad.sides

/-- The sum of all possible values of x -/
noncomputable def sum_of_x_values (quad : SpecialQuadrilateral) : ℝ :=
  sorry -- Definition of sum calculation

/-- Theorem stating the sum of all possible x values is 53 -/
theorem sum_of_x_is_53 (quad : SpecialQuadrilateral) 
  (h : is_special_quadrilateral quad) : 
  sum_of_x_values quad = 53 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_is_53_l606_60663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l606_60699

/-- A rectangle dissected into nine non-overlapping squares with specific properties -/
structure RectangleWithSquares where
  /-- The sides of the nine squares -/
  a : Fin 9 → ℕ
  /-- The width of the rectangle -/
  w : ℕ
  /-- The height of the rectangle -/
  h : ℕ
  /-- The width and height are relatively prime -/
  coprime : Nat.Coprime w h
  /-- Relationships between the squares -/
  rel₁ : a 0 + a 1 = a 2
  rel₂ : a 1 + a 2 = a 3
  rel₃ : a 3 + a 2 = a 4
  rel₄ : a 4 + a 3 = a 5
  rel₅ : a 5 + a 4 = a 6
  rel₆ : a 6 + a 5 = a 7
  rel₇ : a 0 + a 3 + a 6 = a 8
  /-- The width is the sum of the 6th and 9th squares -/
  width_def : w = a 5 + a 8
  /-- The height is the 8th square -/
  height_def : h = a 7
  /-- The first square has side length 1 -/
  first_square : a 0 = 1
  /-- The second square has side length 3 -/
  second_square : a 1 = 3

/-- The perimeter of the rectangle is 204 -/
theorem rectangle_perimeter (r : RectangleWithSquares) : 2 * (r.w + r.h) = 204 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l606_60699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_chord_length_l606_60615

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the distance from a point to a line
def distToLine (x y : ℝ) : ℝ := |x + 1|

-- Define the distance between two points
noncomputable def distToPoint (x y : ℝ) : ℝ := ((x - 1)^2 + y^2).sqrt

-- Define a point on the parabola
def onParabola (x y : ℝ) : Prop := parabola x y ∧ distToLine x y = distToPoint x y

-- Define a chord of the parabola
def isChord (x1 y1 x2 y2 : ℝ) : Prop := 
  parabola x1 y1 ∧ parabola x2 y2

-- Define the midpoint of a chord
def isMidpoint (x y x1 y1 x2 y2 : ℝ) : Prop :=
  x = (x1 + x2) / 2 ∧ y = (y1 + y2) / 2

theorem parabola_and_chord_length :
  (∀ x y, onParabola x y → parabola x y) ∧
  (∀ x1 y1 x2 y2, isChord x1 y1 x2 y2 → isMidpoint 2 1 x1 y1 x2 y2 → 
    ((x2 - x1)^2 + (y2 - y1)^2).sqrt = Real.sqrt 35) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_chord_length_l606_60615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_problem_l606_60695

/-- The chord length cut by a line from a circle -/
noncomputable def chord_length (a b c : ℝ) (x₀ y₀ r : ℝ) : ℝ :=
  2 * Real.sqrt (r^2 - (abs (a * x₀ + b * y₀ + c) / Real.sqrt (a^2 + b^2))^2)

/-- The problem statement -/
theorem chord_length_problem :
  chord_length 3 (-4) (-4) 3 0 3 = 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_problem_l606_60695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l606_60642

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/5) * Real.sin (x + Real.pi/3) + Real.cos (x - Real.pi/6)

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 6/5 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l606_60642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sphere_radius_correct_largest_inner_cylinder_radius_correct_l606_60638

/-- Given three identical cylindrical surfaces with radius R, mutually perpendicular axes, and touching each other pairwise -/
structure CylinderConfiguration (R : ℝ) :=
  (radius : ℝ := R)
  (perpendicular_axes : Prop)
  (pairwise_touching : Prop)

/-- The radius of the smallest sphere touching the three cylinders -/
noncomputable def smallest_sphere_radius (config : CylinderConfiguration R) : ℝ :=
  R * (Real.sqrt 2 - 1)

/-- The radius of the largest cylinder touching the three given cylinders, with its axis passing inside the triangle formed by the contact points -/
noncomputable def largest_inner_cylinder_radius (config : CylinderConfiguration R) : ℝ :=
  R * Real.sqrt 2

theorem smallest_sphere_radius_correct (config : CylinderConfiguration R) :
  smallest_sphere_radius config = R * (Real.sqrt 2 - 1) := by
  sorry

theorem largest_inner_cylinder_radius_correct (config : CylinderConfiguration R) :
  largest_inner_cylinder_radius config = R * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sphere_radius_correct_largest_inner_cylinder_radius_correct_l606_60638
