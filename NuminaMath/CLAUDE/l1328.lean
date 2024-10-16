import Mathlib

namespace NUMINAMATH_CALUDE_new_distance_segment_l1328_132897

/-- New distance function between two points -/
def new_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₂ - x₁| + |y₂ - y₁|

/-- Predicate to check if a point is on a line segment -/
def on_segment (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = x₁ + t * (x₂ - x₁) ∧ y = y₁ + t * (y₂ - y₁)

/-- Theorem: If C is on segment AB, then |AC| + |BC| = |AB| -/
theorem new_distance_segment (x₁ y₁ x₂ y₂ x y : ℝ) :
  on_segment x₁ y₁ x₂ y₂ x y →
  new_distance x₁ y₁ x y + new_distance x y x₂ y₂ = new_distance x₁ y₁ x₂ y₂ :=
by sorry

end NUMINAMATH_CALUDE_new_distance_segment_l1328_132897


namespace NUMINAMATH_CALUDE_percentage_difference_l1328_132843

theorem percentage_difference : 
  (0.80 * 40) - ((4 / 5) * 20) = 16 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1328_132843


namespace NUMINAMATH_CALUDE_set_problems_l1328_132878

def U : Set ℕ := {1, 2, 3, 4}

def B : Set ℕ := {1, 4}

def A (x : ℕ) : Set ℕ := {1, 2, x^2}

theorem set_problems (x : ℕ) (hx : x ∈ U) :
  (U \ B = {2, 3}) ∧ 
  (A x ∩ B = B → x = 1) ∧
  ¬∃ (y : ℕ), y ∈ U ∧ A y ∪ B = U :=
by sorry

end NUMINAMATH_CALUDE_set_problems_l1328_132878


namespace NUMINAMATH_CALUDE_rectangle_length_calculation_l1328_132890

theorem rectangle_length_calculation (w : ℝ) (l_increase : ℝ) (w_decrease : ℝ) :
  w = 40 →
  l_increase = 0.30 →
  w_decrease = 0.17692307692307693 →
  (1 + l_increase) * (1 - w_decrease) * w = w →
  ∃ l : ℝ, l = 40 / 1.3 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_length_calculation_l1328_132890


namespace NUMINAMATH_CALUDE_percent_fifteen_percent_l1328_132863

-- Define the operations
def percent (y : Int) : Int := 8 - y
def prepercent (y : Int) : Int := y - 8

-- Theorem statement
theorem percent_fifteen_percent : prepercent (percent 15) = -15 := by
  sorry

end NUMINAMATH_CALUDE_percent_fifteen_percent_l1328_132863


namespace NUMINAMATH_CALUDE_distance_between_polar_points_l1328_132859

/-- Given two points P and Q in polar coordinates, where the difference of their angles is π/3,
    prove that the distance between them is 8√10. -/
theorem distance_between_polar_points (α β : Real) :
  let P : Real × Real := (4, α)
  let Q : Real × Real := (12, β)
  α - β = π / 3 →
  let distance := Real.sqrt ((12 * Real.cos β - 4 * Real.cos α)^2 + (12 * Real.sin β - 4 * Real.sin α)^2)
  distance = 8 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_polar_points_l1328_132859


namespace NUMINAMATH_CALUDE_incenter_x_coordinate_is_one_l1328_132889

/-- The triangle formed by the x-axis, y-axis, and the line x + y = 2 -/
structure Triangle where
  A : ℝ × ℝ := (0, 2)  -- y-intercept
  B : ℝ × ℝ := (2, 0)  -- x-intercept
  O : ℝ × ℝ := (0, 0)  -- origin

/-- The incenter of a triangle -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- The distance between a point and a line -/
def distancePointToLine (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ := sorry

theorem incenter_x_coordinate_is_one (t : Triangle) :
  (incenter t).1 = 1 ∧
  distancePointToLine (incenter t) (fun x => 0) =
  distancePointToLine (incenter t) (fun x => x) ∧
  distancePointToLine (incenter t) (fun x => 0) =
  distancePointToLine (incenter t) (fun x => 2 - x) :=
sorry

end NUMINAMATH_CALUDE_incenter_x_coordinate_is_one_l1328_132889


namespace NUMINAMATH_CALUDE_real_and_equal_roots_l1328_132847

/-- The quadratic equation in the problem -/
def quadratic_equation (k x : ℝ) : ℝ := 3 * x^2 - k * x + 2 * x + 15

/-- The discriminant of the quadratic equation -/
def discriminant (k : ℝ) : ℝ := (k - 2)^2 - 4 * 3 * 15

theorem real_and_equal_roots (k : ℝ) :
  (∃ x : ℝ, quadratic_equation k x = 0 ∧
    ∀ y : ℝ, quadratic_equation k y = 0 → y = x) ↔
  (k = 6 * Real.sqrt 5 + 2 ∨ k = -6 * Real.sqrt 5 + 2) :=
sorry

end NUMINAMATH_CALUDE_real_and_equal_roots_l1328_132847


namespace NUMINAMATH_CALUDE_area_of_pentagon_l1328_132883

-- Define the square ABCD
def ABCD : Set (ℝ × ℝ) := sorry

-- Define that BD is a diagonal of ABCD
def BD_is_diagonal (ABCD : Set (ℝ × ℝ)) : Prop := sorry

-- Define the length of BD
def BD_length : ℝ := 20

-- Define the rectangle BDFE
def BDFE : Set (ℝ × ℝ) := sorry

-- Define the pentagon ABEFD
def ABEFD : Set (ℝ × ℝ) := sorry

-- Define the area function
def area : Set (ℝ × ℝ) → ℝ := sorry

-- Theorem statement
theorem area_of_pentagon (h1 : BD_is_diagonal ABCD) (h2 : BD_length = 20) : 
  area ABEFD = 300 := by sorry

end NUMINAMATH_CALUDE_area_of_pentagon_l1328_132883


namespace NUMINAMATH_CALUDE_expression_evaluation_l1328_132880

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -1
  (3 * x^2 * y - x * y^2) - 2 * (-2 * x * y^2 + x^2 * y) = 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1328_132880


namespace NUMINAMATH_CALUDE_no_adjacent_knights_probability_l1328_132864

/-- The number of knights seated in a circle -/
def total_knights : ℕ := 20

/-- The number of knights selected for the quest -/
def selected_knights : ℕ := 4

/-- The probability that no two of the selected knights are sitting next to each other -/
def probability : ℚ := 60 / 7

/-- Theorem stating that the probability of no two selected knights sitting next to each other is 60/7 -/
theorem no_adjacent_knights_probability :
  probability = 60 / 7 := by sorry

end NUMINAMATH_CALUDE_no_adjacent_knights_probability_l1328_132864


namespace NUMINAMATH_CALUDE_product_of_solutions_l1328_132888

theorem product_of_solutions (y : ℝ) : (|y| = 3 * (|y| - 2)) → ∃ z : ℝ, (|z| = 3 * (|z| - 2)) ∧ (y * z = -9) := by
  sorry

end NUMINAMATH_CALUDE_product_of_solutions_l1328_132888


namespace NUMINAMATH_CALUDE_triangle_centroid_incenter_ratio_l1328_132885

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the circumcenter
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the incenter
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the midpoints of major arcs
def majorArcMidpoints (t : Triangle) : Triangle := sorry

-- Define the centroid of a triangle
def centroid (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem triangle_centroid_incenter_ratio (t : Triangle) :
  let O := circumcenter t
  let I := incenter t
  let G := centroid (majorArcMidpoints t)
  distance O A = distance O B ∧ 
  distance O B = distance O C ∧
  distance A B = 13 ∧ 
  distance B C = 14 ∧ 
  distance C A = 15 →
  distance G O / distance G I = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_centroid_incenter_ratio_l1328_132885


namespace NUMINAMATH_CALUDE_function_analysis_l1328_132839

/-- Given a function f and some conditions, prove its analytical expression and range -/
theorem function_analysis (f : ℝ → ℝ) (ω φ : ℝ) :
  (ω > 0) →
  (φ > 0 ∧ φ < Real.pi / 2) →
  (Real.tan φ = 2 * Real.sqrt 3) →
  (∀ x, f x = Real.sqrt 13 * Real.cos (ω * x) * Real.cos (ω * x - φ) - Real.sin (ω * x) ^ 2) →
  (∀ x, f (x + Real.pi / ω) = f x) →
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ Real.pi / ω) →
  (∀ x, f x = 2 * Real.sin (2 * x + Real.pi / 6)) ∧
  (Set.Icc (1 / 13) 2 = { y | ∃ x ∈ Set.Icc (Real.pi / 12) φ, f x = y }) := by
  sorry

end NUMINAMATH_CALUDE_function_analysis_l1328_132839


namespace NUMINAMATH_CALUDE_smallest_determinant_and_minimal_pair_l1328_132803

def determinant (a b : ℤ) : ℤ := 36 * b - 81 * a

theorem smallest_determinant_and_minimal_pair :
  (∃ c : ℕ+, ∀ a b : ℤ, determinant a b ≠ 0 → c ≤ |determinant a b|) ∧
  (∃ a b : ℕ, determinant a b = 9 ∧
    ∀ a' b' : ℕ, determinant a' b' = 9 → a + b ≤ a' + b') :=
by sorry

end NUMINAMATH_CALUDE_smallest_determinant_and_minimal_pair_l1328_132803


namespace NUMINAMATH_CALUDE_max_value_interval_range_l1328_132858

/-- The function f(x) = x^3 - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The interval (a, 6-a^2) --/
def interval (a : ℝ) : Set ℝ := {x | a < x ∧ x < 6 - a^2}

/-- Theorem stating the range of a for which f has a maximum on the interval --/
theorem max_value_interval_range :
  ∀ a : ℝ, (∃ x_max ∈ interval a, ∀ x ∈ interval a, f x ≤ f x_max) →
    a > -Real.sqrt 7 ∧ a ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_max_value_interval_range_l1328_132858


namespace NUMINAMATH_CALUDE_triangle_interior_center_points_l1328_132865

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle ABC in the Cartesian plane -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Count of interior center points in a triangle -/
def interiorCenterPoints (t : Triangle) : ℕ :=
  sorry

/-- The main theorem -/
theorem triangle_interior_center_points :
  let t : Triangle := {
    A := { x := 0, y := 0 },
    B := { x := 200, y := 100 },
    C := { x := 30, y := 330 }
  }
  interiorCenterPoints t = 31480 := by sorry

end NUMINAMATH_CALUDE_triangle_interior_center_points_l1328_132865


namespace NUMINAMATH_CALUDE_reflection_composition_l1328_132834

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_y_eq_x_minus_1 (p : ℝ × ℝ) : ℝ × ℝ :=
  let q := (p.1, p.2 + 1)
  let r := (q.2, q.1)
  (r.1, r.2 - 1)

theorem reflection_composition (H : ℝ × ℝ) :
  H = (5, 0) →
  reflect_y_eq_x_minus_1 (reflect_x H) = (1, 4) := by
  sorry

end NUMINAMATH_CALUDE_reflection_composition_l1328_132834


namespace NUMINAMATH_CALUDE_total_supervisors_is_25_l1328_132826

/-- The total number of supervisors on 5 buses -/
def total_supervisors : ℕ := 4 + 5 + 3 + 6 + 7

/-- Theorem stating that the total number of supervisors is 25 -/
theorem total_supervisors_is_25 : total_supervisors = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_supervisors_is_25_l1328_132826


namespace NUMINAMATH_CALUDE_triangle_side_length_l1328_132898

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  -- No specific conditions needed here

-- Define a point on a line segment
def PointOnSegment (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (1 - t) • A + t • B

-- Define perpendicularity
def Perpendicular (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (D.1 - C.1) + (B.2 - A.2) * (D.2 - C.2) = 0

-- Define equality of distances
def EqualDistances (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (D.1 - C.1)^2 + (D.2 - C.2)^2

-- Main theorem
theorem triangle_side_length 
  (P Q R E G : ℝ × ℝ) 
  (triangle : Triangle P Q R) 
  (e_on_pq : PointOnSegment P Q E)
  (g_on_pr : PointOnSegment P R G)
  (pq_perp_pr : Perpendicular P Q P R)
  (pg_perp_pr : Perpendicular P G P R)
  (qe_eq_eg : EqualDistances Q E E G)
  (eg_eq_gr : EqualDistances E G G R)
  (gr_eq_3 : EqualDistances G R P (P.1 + 3, P.2)) :
  EqualDistances P R P (P.1 + 6, P.2) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1328_132898


namespace NUMINAMATH_CALUDE_square_triangles_area_bounds_l1328_132815

-- Define the unit square
def UnitSquare : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the right-angled triangles constructed outward
def OutwardTriangles (s : Set (ℝ × ℝ)) : Set (Set (ℝ × ℝ)) := sorry

-- Define the vertices A, B, C, D
def RightAngleVertices (triangles : Set (Set (ℝ × ℝ))) : Set (ℝ × ℝ) := sorry

-- Define the incircle centers O₁, O₂, O₃, O₄
def IncircleCenters (triangles : Set (Set (ℝ × ℝ))) : Set (ℝ × ℝ) := sorry

-- Define the area of a quadrilateral
def QuadrilateralArea (vertices : Set (ℝ × ℝ)) : ℝ := sorry

theorem square_triangles_area_bounds :
  let s := UnitSquare
  let triangles := OutwardTriangles s
  let abcd := RightAngleVertices triangles
  let o₁o₂o₃o₄ := IncircleCenters triangles
  (QuadrilateralArea abcd ≤ 2) ∧ (QuadrilateralArea o₁o₂o₃o₄ ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_square_triangles_area_bounds_l1328_132815


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1328_132808

theorem point_on_x_axis (m : ℚ) :
  (∃ x : ℚ, x = 2 - m ∧ 0 = 3 * m + 1) → m = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1328_132808


namespace NUMINAMATH_CALUDE_last_three_digits_of_2_to_15000_l1328_132809

theorem last_three_digits_of_2_to_15000 (h : 2^500 ≡ 1 [ZMOD 1250]) :
  2^15000 ≡ 1 [ZMOD 1000] := by
sorry

end NUMINAMATH_CALUDE_last_three_digits_of_2_to_15000_l1328_132809


namespace NUMINAMATH_CALUDE_problem_solution_l1328_132810

-- Define what it means for a number to be a factor of another
def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

-- Define what it means for a number to be a divisor of another
def is_divisor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem problem_solution :
  (is_factor 5 25) ∧
  (is_divisor 19 209 ∧ ¬ is_divisor 19 63) ∧
  (is_factor 9 180) := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l1328_132810


namespace NUMINAMATH_CALUDE_sum_of_squares_over_products_l1328_132884

theorem sum_of_squares_over_products (a b c : ℝ) (h : a + b + c = 0) :
  a^2 / (b * c) + b^2 / (a * c) + c^2 / (a * b) = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_over_products_l1328_132884


namespace NUMINAMATH_CALUDE_x_leq_y_neither_necessary_nor_sufficient_for_abs_x_leq_abs_y_l1328_132887

theorem x_leq_y_neither_necessary_nor_sufficient_for_abs_x_leq_abs_y :
  ¬(∀ (x y : ℝ), x ≤ y → |x| ≤ |y|) ∧ ¬(∀ (x y : ℝ), |x| ≤ |y| → x ≤ y) := by
  sorry

end NUMINAMATH_CALUDE_x_leq_y_neither_necessary_nor_sufficient_for_abs_x_leq_abs_y_l1328_132887


namespace NUMINAMATH_CALUDE_factorial_sum_square_solutions_l1328_132848

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

def sum_factorials (n : ℕ) : ℕ := (Finset.range n).sum (λ i => factorial (i + 1))

theorem factorial_sum_square_solutions :
  ∀ n m : ℕ, sum_factorials n = m^2 ↔ (n = 1 ∧ m = 1) ∨ (n = 3 ∧ m = 3) := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_square_solutions_l1328_132848


namespace NUMINAMATH_CALUDE_work_done_on_bullet_l1328_132821

theorem work_done_on_bullet (m : Real) (v1 v2 : Real) :
  m = 0.01 →
  v1 = 500 →
  v2 = 200 →
  let K1 := (1/2) * m * v1^2
  let K2 := (1/2) * m * v2^2
  K1 - K2 = 1050 := by sorry

end NUMINAMATH_CALUDE_work_done_on_bullet_l1328_132821


namespace NUMINAMATH_CALUDE_equation_solutions_l1328_132825

theorem equation_solutions : ∀ x : ℝ,
  (x^2 - 3*x = 4 ↔ x = 4 ∨ x = -1) ∧
  (x*(x-2) + x - 2 = 0 ↔ x = 2 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1328_132825


namespace NUMINAMATH_CALUDE_triangle_properties_l1328_132831

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Two vectors are parallel -/
def parallel (v w : Vector2D) : Prop :=
  ∃ (k : ℝ), v.x * w.y = k * v.y * w.x

variable (ABC : Triangle)
variable (m n : Vector2D)

/-- The given conditions -/
axiom cond1 : m = ⟨2 * Real.sin ABC.B, -Real.sqrt 3⟩
axiom cond2 : n = ⟨Real.cos (2 * ABC.B), 2 * (Real.cos ABC.B)^2 - 1⟩
axiom cond3 : parallel m n
axiom cond4 : ABC.b = 2

/-- The theorem to be proved -/
theorem triangle_properties :
  ABC.B = Real.pi / 3 ∧
  (∀ (S : ℝ), S = 1/2 * ABC.a * ABC.c * Real.sin ABC.B → S ≤ Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1328_132831


namespace NUMINAMATH_CALUDE_robin_candy_count_l1328_132893

/-- Given Robin's initial candy count, the number she ate, and the number her sister gave her, 
    her final candy count is equal to 37. -/
theorem robin_candy_count (initial : ℕ) (eaten : ℕ) (received : ℕ) 
    (h1 : initial = 23) 
    (h2 : eaten = 7) 
    (h3 : received = 21) : 
  initial - eaten + received = 37 := by
  sorry

end NUMINAMATH_CALUDE_robin_candy_count_l1328_132893


namespace NUMINAMATH_CALUDE_min_value_theorem_l1328_132862

/-- Given f(x) = a^x - b, where a > 0, a ≠ 1, and b is real,
    and g(x) = x + 1, if f(x) * g(x) ≤ 0 for all real x,
    then the minimum value of 1/a + 4/b is 4. -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (ha' : a ≠ 1) 
  (hf : ∀ x : ℝ, a^x - b ≤ 0 ∨ x + 1 ≤ 0) :
  ∀ ε > 0, ∃ a₀ b₀ : ℝ, 1/a₀ + 4/b₀ < 4 + ε :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1328_132862


namespace NUMINAMATH_CALUDE_log_properties_l1328_132856

/-- Properties of the natural logarithm function -/
theorem log_properties :
  let f : ℝ → ℝ := Real.log
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → f (x₁ * x₂) = f x₁ + f x₂) ∧
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_log_properties_l1328_132856


namespace NUMINAMATH_CALUDE_people_per_entrance_l1328_132838

theorem people_per_entrance 
  (total_entrances : ℕ) 
  (total_people : ℕ) 
  (h1 : total_entrances = 5) 
  (h2 : total_people = 1415) :
  total_people / total_entrances = 283 :=
by sorry

end NUMINAMATH_CALUDE_people_per_entrance_l1328_132838


namespace NUMINAMATH_CALUDE_two_lines_exist_l1328_132886

/-- A line in the 2D plane represented by the equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The transformation from P to P' -/
def transform (x y : ℝ) : ℝ × ℝ :=
  (3 * x + 2 * y + 1, x + 4 * y - 3)

/-- A point (x, y) is on a line if it satisfies the line equation -/
def on_line (l : Line) (x y : ℝ) : Prop :=
  l.A * x + l.B * y + l.C = 0

/-- The main theorem stating that exactly two lines exist satisfying the given conditions -/
theorem two_lines_exist : ∃! (l1 l2 : Line), 
  l1 ≠ l2 ∧ 
  (∀ x y : ℝ, on_line l1 x y ↔ on_line l1 (transform x y).1 (transform x y).2) ∧
  (∀ x y : ℝ, on_line l2 x y ↔ on_line l2 (transform x y).1 (transform x y).2) :=
by sorry

end NUMINAMATH_CALUDE_two_lines_exist_l1328_132886


namespace NUMINAMATH_CALUDE_volume_of_P₃_l1328_132868

/-- Represents a polyhedron in the sequence -/
structure Polyhedron where
  index : ℕ
  volume : ℚ

/-- Constructs the next polyhedron in the sequence -/
def next_polyhedron (P : Polyhedron) : Polyhedron :=
  { index := P.index + 1,
    volume := P.volume + (3/2)^P.index }

/-- The initial regular tetrahedron -/
def P₀ : Polyhedron :=
  { index := 0,
    volume := 1 }

/-- Generates the nth polyhedron in the sequence -/
def generate_polyhedron (n : ℕ) : Polyhedron :=
  match n with
  | 0 => P₀
  | n + 1 => next_polyhedron (generate_polyhedron n)

/-- The theorem to be proved -/
theorem volume_of_P₃ :
  (generate_polyhedron 3).volume = 23/4 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_P₃_l1328_132868


namespace NUMINAMATH_CALUDE_value_of_expression_l1328_132814

theorem value_of_expression (x : ℝ) (h : x = 4) : (3*x + 7)^2 = 361 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1328_132814


namespace NUMINAMATH_CALUDE_inequality_proof_l1328_132891

open Real BigOperators

theorem inequality_proof (n : ℕ) (r s t u v : Fin n → ℝ) 
  (hr : ∀ i, r i > 1) (hs : ∀ i, s i > 1) (ht : ∀ i, t i > 1) (hu : ∀ i, u i > 1) (hv : ∀ i, v i > 1) :
  let R := (∑ i, r i) / n
  let S := (∑ i, s i) / n
  let T := (∑ i, t i) / n
  let U := (∑ i, u i) / n
  let V := (∑ i, v i) / n
  ∑ i, ((r i * s i * t i * u i * v i + 1) / (r i * s i * t i * u i * v i - 1)) ≥ 
    ((R * S * T * U * V + 1) / (R * S * T * U * V - 1)) ^ n :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1328_132891


namespace NUMINAMATH_CALUDE_cube_painting_theorem_l1328_132842

/-- The number of faces on a cube -/
def num_faces : ℕ := 6

/-- The number of available colors -/
def num_colors : ℕ := 6

/-- The number of rotational symmetries of a cube -/
def cube_symmetries : ℕ := 24

/-- The number of ways to paint all faces of a cube the same color -/
def all_same_color : ℕ := num_colors

/-- The number of ways to paint 5 faces the same color and 1 face a different color -/
def five_same_one_different : ℕ := num_faces * (num_colors - 1)

/-- The number of ways to paint all faces of a cube different colors, considering rotational symmetry -/
def all_different_colors : ℕ := (Nat.factorial num_colors) / cube_symmetries

theorem cube_painting_theorem :
  all_same_color = 6 ∧
  five_same_one_different = 30 ∧
  all_different_colors = 30 := by
  sorry

end NUMINAMATH_CALUDE_cube_painting_theorem_l1328_132842


namespace NUMINAMATH_CALUDE_percentage_of_defective_meters_l1328_132867

theorem percentage_of_defective_meters
  (total_meters : ℕ)
  (rejected_meters : ℕ)
  (h1 : total_meters = 150)
  (h2 : rejected_meters = 15) :
  (rejected_meters : ℝ) / (total_meters : ℝ) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_defective_meters_l1328_132867


namespace NUMINAMATH_CALUDE_roden_fish_purchase_l1328_132866

/-- Represents the number of fish bought in a single visit -/
structure FishPurchase where
  goldfish : ℕ
  bluefish : ℕ
  greenfish : ℕ

/-- Calculates the total number of fish bought during three visits -/
def totalFish (visit1 visit2 visit3 : FishPurchase) : ℕ :=
  visit1.goldfish + visit1.bluefish + visit1.greenfish +
  visit2.goldfish + visit2.bluefish + visit2.greenfish +
  visit3.goldfish + visit3.bluefish + visit3.greenfish

theorem roden_fish_purchase :
  let visit1 : FishPurchase := { goldfish := 15, bluefish := 7, greenfish := 0 }
  let visit2 : FishPurchase := { goldfish := 10, bluefish := 12, greenfish := 5 }
  let visit3 : FishPurchase := { goldfish := 3, bluefish := 7, greenfish := 9 }
  totalFish visit1 visit2 visit3 = 68 := by
  sorry

end NUMINAMATH_CALUDE_roden_fish_purchase_l1328_132866


namespace NUMINAMATH_CALUDE_f_log2_32_equals_17_l1328_132841

noncomputable def f (x : ℝ) : ℝ :=
  if x < 4 then Real.log 4 / Real.log 2
  else 1 + 2^(x - 1)

theorem f_log2_32_equals_17 : f (Real.log 32 / Real.log 2) = 17 := by
  sorry

end NUMINAMATH_CALUDE_f_log2_32_equals_17_l1328_132841


namespace NUMINAMATH_CALUDE_sandy_change_proof_l1328_132830

/-- Calculates the change received from a purchase given the payment amount and the costs of individual items. -/
def calculate_change (payment : ℚ) (item1_cost : ℚ) (item2_cost : ℚ) : ℚ :=
  payment - (item1_cost + item2_cost)

/-- Proves that given a $20 bill payment and purchases of $9.24 and $8.25, the change received is $2.51. -/
theorem sandy_change_proof :
  calculate_change 20 9.24 8.25 = 2.51 := by
  sorry

end NUMINAMATH_CALUDE_sandy_change_proof_l1328_132830


namespace NUMINAMATH_CALUDE_hexagon_fencing_cost_l1328_132853

/-- The cost of fencing an irregular hexagonal field -/
theorem hexagon_fencing_cost (side1 side2 side3 side4 side5 side6 : ℝ)
  (cost_first_three : ℝ) (cost_last_three : ℝ) :
  side1 = 20 ∧ side2 = 15 ∧ side3 = 25 ∧ side4 = 30 ∧ side5 = 10 ∧ side6 = 35 ∧
  cost_first_three = 3.5 ∧ cost_last_three = 4 →
  (side1 + side2 + side3) * cost_first_three + (side4 + side5 + side6) * cost_last_three = 510 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_fencing_cost_l1328_132853


namespace NUMINAMATH_CALUDE_polynomial_properties_l1328_132846

def polynomial_expansion (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : Prop :=
  (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5

theorem polynomial_properties 
  (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : ∀ x, polynomial_expansion x a₀ a₁ a₂ a₃ a₄ a₅) : 
  (a₀ + a₁ + a₂ + a₃ + a₄ = -31) ∧ 
  ((a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = -243) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_properties_l1328_132846


namespace NUMINAMATH_CALUDE_calculate_expression_l1328_132874

theorem calculate_expression : 
  (-2)^2 + Real.sqrt 8 - abs (1 - Real.sqrt 2) + (2023 - Real.pi)^0 = 6 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1328_132874


namespace NUMINAMATH_CALUDE_coefficient_of_x_in_second_equation_l1328_132806

theorem coefficient_of_x_in_second_equation 
  (x y z : ℝ) 
  (eq1 : 6*x - 5*y + 3*z = 22)
  (eq2 : x + 8*y - 11*z = 7/4)
  (eq3 : 5*x - 6*y + 2*z = 12)
  (sum_xyz : x + y + z = 10) :
  1 = 1 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_in_second_equation_l1328_132806


namespace NUMINAMATH_CALUDE_cos_135_degrees_l1328_132881

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l1328_132881


namespace NUMINAMATH_CALUDE_ellipse_area_ratio_range_l1328_132801

/-- An ellipse with given properties --/
structure Ellipse where
  foci : (ℝ × ℝ) × (ℝ × ℝ)
  passesThrough : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- A line intersecting the ellipse --/
structure IntersectingLine where
  passingThrough : ℝ × ℝ
  intersectionPoints : (ℝ × ℝ) × (ℝ × ℝ)

/-- The ratio of triangle areas --/
def areaRatio (e : Ellipse) (l : IntersectingLine) : ℝ := sorry

theorem ellipse_area_ratio_range 
  (e : Ellipse) 
  (l : IntersectingLine) 
  (h1 : e.foci = ((-Real.sqrt 3, 0), (Real.sqrt 3, 0)))
  (h2 : e.passesThrough = (1, Real.sqrt 3 / 2))
  (h3 : e.equation = fun x y ↦ x^2 / 4 + y^2 = 1)
  (h4 : l.passingThrough = (0, 2))
  (h5 : ∃ (M N : ℝ × ℝ), l.intersectionPoints = (M, N) ∧ 
        e.equation M.1 M.2 ∧ e.equation N.1 N.2 ∧ 
        (∃ t : ℝ, 0 < t ∧ t < 1 ∧ M = (t * l.passingThrough.1 + (1 - t) * N.1, 
                                       t * l.passingThrough.2 + (1 - t) * N.2))) :
  1/3 < areaRatio e l ∧ areaRatio e l < 1 := by sorry

end NUMINAMATH_CALUDE_ellipse_area_ratio_range_l1328_132801


namespace NUMINAMATH_CALUDE_coefficient_of_linear_term_l1328_132812

theorem coefficient_of_linear_term (a b c : ℝ) : 
  (fun x : ℝ => a * x^2 + b * x + c) = (fun x : ℝ => x^2 - 2*x + 3) → 
  b = -2 := by
sorry

end NUMINAMATH_CALUDE_coefficient_of_linear_term_l1328_132812


namespace NUMINAMATH_CALUDE_knights_archery_skill_l1328_132861

theorem knights_archery_skill (total : ℕ) (total_pos : total > 0) : 
  let gold := (3 * total) / 8
  let silver := total - gold
  let skilled := total / 4
  ∃ (gold_skilled silver_skilled : ℕ),
    gold_skilled + silver_skilled = skilled ∧
    gold_skilled * silver = 3 * silver_skilled * gold ∧
    gold_skilled * 7 = gold * 3 := by
  sorry

end NUMINAMATH_CALUDE_knights_archery_skill_l1328_132861


namespace NUMINAMATH_CALUDE_direction_vectors_of_line_l1328_132872

/-- Given a line with equation 3x - 4y + 1 = 0, prove that (4, 3) and (1, 3/4) are valid direction vectors. -/
theorem direction_vectors_of_line (x y : ℝ) : 
  (3 * x - 4 * y + 1 = 0) →
  (∃ (k : ℝ), k ≠ 0 ∧ (k * 4, k * 3) = (3, -4)) ∧
  (∃ (k : ℝ), k ≠ 0 ∧ (k * 1, k * (3/4)) = (3, -4)) :=
by sorry

end NUMINAMATH_CALUDE_direction_vectors_of_line_l1328_132872


namespace NUMINAMATH_CALUDE_billboard_shorter_side_l1328_132879

theorem billboard_shorter_side (length width : ℝ) : 
  length * width = 91 →
  2 * (length + width) = 40 →
  length > 0 →
  width > 0 →
  min length width = 7 := by
sorry

end NUMINAMATH_CALUDE_billboard_shorter_side_l1328_132879


namespace NUMINAMATH_CALUDE_min_cost_closed_chain_l1328_132896

/-- Represents the cost in cents to separate one link -/
def separation_cost : ℕ := 1

/-- Represents the cost in cents to attach one link -/
def attachment_cost : ℕ := 2

/-- Represents the number of pieces in the gold chain -/
def num_pieces : ℕ := 13

/-- Represents the number of links in each piece of the chain -/
def links_per_piece : ℕ := 80

/-- Calculates the total cost to separate and reattach one link -/
def link_operation_cost : ℕ := separation_cost + attachment_cost

/-- Theorem stating the minimum cost to form a closed chain -/
theorem min_cost_closed_chain : 
  ∃ (cost : ℕ), cost = (num_pieces - 1) * link_operation_cost ∧ 
  ∀ (other_cost : ℕ), other_cost ≥ cost := by sorry

end NUMINAMATH_CALUDE_min_cost_closed_chain_l1328_132896


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1328_132852

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x : ℝ, f x ≤ 0) ↔ (∀ x : ℝ, f x > 0) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1328_132852


namespace NUMINAMATH_CALUDE_expression_evaluation_l1328_132850

theorem expression_evaluation (a b c d : ℝ) 
  (ha : a = 11) (hb : b = 13) (hc : c = 17) (hd : d = 19) :
  (a^2 * (1/b - 1/d) + b^2 * (1/d - 1/a) + c^2 * (1/a - 1/c) + d^2 * (1/c - 1/b)) /
  (a * (1/b - 1/d) + b * (1/d - 1/a) + c * (1/a - 1/c) + d * (1/c - 1/b)) = a + b + c + d :=
by sorry

#eval (11 : ℝ) + 13 + 17 + 19  -- To verify the result is indeed 60

end NUMINAMATH_CALUDE_expression_evaluation_l1328_132850


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1328_132807

theorem rationalize_denominator : (14 : ℝ) / Real.sqrt 14 = Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1328_132807


namespace NUMINAMATH_CALUDE_rounding_317500_equals_31_8_ten_thousand_l1328_132827

/-- Rounds a natural number to the nearest thousand -/
def round_to_nearest_thousand (n : ℕ) : ℕ :=
  ((n + 500) / 1000) * 1000

/-- Converts a natural number to ten thousands -/
def to_ten_thousands (n : ℕ) : ℚ :=
  (n : ℚ) / 10000

theorem rounding_317500_equals_31_8_ten_thousand :
  to_ten_thousands (round_to_nearest_thousand 317500) = 31.8 := by
  sorry

end NUMINAMATH_CALUDE_rounding_317500_equals_31_8_ten_thousand_l1328_132827


namespace NUMINAMATH_CALUDE_andrea_pony_cost_l1328_132836

/-- The total annual cost for Andrea's pony -/
def annual_pony_cost (monthly_pasture_rent : ℕ) (daily_food_cost : ℕ) (lesson_cost : ℕ) 
  (lessons_per_week : ℕ) (months_per_year : ℕ) (days_per_year : ℕ) (weeks_per_year : ℕ) : ℕ :=
  monthly_pasture_rent * months_per_year +
  daily_food_cost * days_per_year +
  lesson_cost * lessons_per_week * weeks_per_year

theorem andrea_pony_cost :
  annual_pony_cost 500 10 60 2 12 365 52 = 15890 := by
  sorry

end NUMINAMATH_CALUDE_andrea_pony_cost_l1328_132836


namespace NUMINAMATH_CALUDE_log_sum_equality_l1328_132824

theorem log_sum_equality (p q : ℝ) (hp : p > 0) (hq : q > 0) (hq1 : q ≠ 1) :
  Real.log p + Real.log q = Real.log (p + q) ↔ p = q / (q - 1) :=
sorry

end NUMINAMATH_CALUDE_log_sum_equality_l1328_132824


namespace NUMINAMATH_CALUDE_x_1971_approximation_l1328_132873

/-- A sequence satisfying the given recurrence relation -/
def recurrence_sequence (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → 3 * x n - x (n - 1) = n

theorem x_1971_approximation
  (x : ℕ → ℝ)
  (h_recurrence : recurrence_sequence x)
  (h_x1_bound : |x 1| < 1971) :
  |x 1971 - 985.250000| < 0.000001 := by
  sorry

end NUMINAMATH_CALUDE_x_1971_approximation_l1328_132873


namespace NUMINAMATH_CALUDE_student_marks_theorem_l1328_132820

/-- Calculates the total marks secured by a student in an examination with the given conditions. -/
def total_marks (total_questions : ℕ) (correct_answers : ℕ) (marks_per_correct : ℕ) (marks_per_wrong : ℕ) : ℤ :=
  (correct_answers * marks_per_correct : ℤ) - ((total_questions - correct_answers) * marks_per_wrong)

/-- Theorem stating that under the given conditions, the student secures 150 marks. -/
theorem student_marks_theorem :
  total_marks 60 42 4 1 = 150 := by
  sorry

end NUMINAMATH_CALUDE_student_marks_theorem_l1328_132820


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l1328_132828

/-- A quadratic function f(x) = ax^2 + bx satisfying certain conditions -/
def QuadraticFunction (a b : ℝ) (f : ℝ → ℝ) : Prop :=
  a ≠ 0 ∧
  (∀ x, f x = a * x^2 + b * x) ∧
  (∀ x, f (-x + 5) = f (x - 3)) ∧
  (∃! x, f x = x)

/-- The domain and range conditions for the quadratic function -/
def DomainRangeCondition (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  m < n ∧
  (∀ x, f x ∈ Set.Icc (3*m) (3*n) ↔ x ∈ Set.Icc m n)

theorem quadratic_function_theorem :
  ∀ a b : ℝ, ∀ f : ℝ → ℝ,
  QuadraticFunction a b f →
  ∃ m n : ℝ,
    (∀ x, f x = -1/2 * x^2 + x) ∧
    m = -4 ∧ n = 0 ∧
    DomainRangeCondition f m n :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l1328_132828


namespace NUMINAMATH_CALUDE_sine_identity_l1328_132822

theorem sine_identity (α : Real) (h : α = π / 7) :
  1 / Real.sin α = 1 / Real.sin (2 * α) + 1 / Real.sin (3 * α) := by
  sorry

end NUMINAMATH_CALUDE_sine_identity_l1328_132822


namespace NUMINAMATH_CALUDE_initial_birds_count_l1328_132894

theorem initial_birds_count (total : ℕ) (additional : ℕ) (initial : ℕ) : 
  total = 42 → additional = 13 → initial + additional = total → initial = 29 := by
  sorry

end NUMINAMATH_CALUDE_initial_birds_count_l1328_132894


namespace NUMINAMATH_CALUDE_total_spent_proof_l1328_132844

-- Define the original prices and discount rates
def tshirt_price : ℚ := 20
def tshirt_discount : ℚ := 0.4
def hat_price : ℚ := 15
def hat_discount : ℚ := 0.6
def accessory_price : ℚ := 10
def bracelet_discount : ℚ := 0.3
def belt_discount : ℚ := 0.5
def sales_tax : ℚ := 0.05

-- Define the number of friends and their purchases
def total_friends : ℕ := 4
def bracelet_buyers : ℕ := 1
def belt_buyers : ℕ := 3

-- Define the function to calculate discounted price
def discounted_price (original_price : ℚ) (discount : ℚ) : ℚ :=
  original_price * (1 - discount)

-- Define the theorem
theorem total_spent_proof :
  let tshirt_discounted := discounted_price tshirt_price tshirt_discount
  let hat_discounted := discounted_price hat_price hat_discount
  let bracelet_discounted := discounted_price accessory_price bracelet_discount
  let belt_discounted := discounted_price accessory_price belt_discount
  let bracelet_total := tshirt_discounted + hat_discounted + bracelet_discounted
  let belt_total := tshirt_discounted + hat_discounted + belt_discounted
  let subtotal := bracelet_total * bracelet_buyers + belt_total * belt_buyers
  let total := subtotal * (1 + sales_tax)
  total = 98.7 := by
    sorry

end NUMINAMATH_CALUDE_total_spent_proof_l1328_132844


namespace NUMINAMATH_CALUDE_marble_problem_l1328_132875

theorem marble_problem :
  ∃ n : ℕ,
    (∀ m : ℕ, m > 0 ∧ m % 8 = 5 ∧ m % 7 = 2 → n ≤ m) ∧
    n % 8 = 5 ∧
    n % 7 = 2 ∧
    n % 9 = 1 ∧
    n = 37 := by
  sorry

end NUMINAMATH_CALUDE_marble_problem_l1328_132875


namespace NUMINAMATH_CALUDE_jessica_remaining_money_l1328_132876

/-- The remaining money after a purchase --/
def remaining_money (initial : ℚ) (spent : ℚ) : ℚ :=
  initial - spent

/-- Proof that Jessica's remaining money is $1.51 --/
theorem jessica_remaining_money :
  remaining_money 11.73 10.22 = 1.51 := by
  sorry

end NUMINAMATH_CALUDE_jessica_remaining_money_l1328_132876


namespace NUMINAMATH_CALUDE_final_symbol_invariant_l1328_132892

/-- Represents the state of the blackboard -/
structure BlackboardState where
  minus_count : Nat
  total_count : Nat

/-- Represents a single operation on the blackboard -/
inductive Operation
  | erase_same_plus
  | erase_same_minus
  | erase_different

/-- Applies an operation to the blackboard state -/
def apply_operation (state : BlackboardState) (op : Operation) : BlackboardState :=
  match op with
  | Operation.erase_same_plus => ⟨state.minus_count, state.total_count - 1⟩
  | Operation.erase_same_minus => ⟨state.minus_count - 2, state.total_count - 1⟩
  | Operation.erase_different => ⟨state.minus_count, state.total_count - 1⟩

/-- The main theorem stating that the final symbol is determined by the initial parity of minus signs -/
theorem final_symbol_invariant (initial_state : BlackboardState)
  (h_initial : initial_state.total_count = 1967)
  (h_valid : initial_state.minus_count ≤ initial_state.total_count) :
  ∃ (final_symbol : Bool),
    ∀ (ops : List Operation),
      (ops.foldl apply_operation initial_state).total_count = 1 →
      final_symbol = ((ops.foldl apply_operation initial_state).minus_count % 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_final_symbol_invariant_l1328_132892


namespace NUMINAMATH_CALUDE_art_club_theorem_l1328_132857

/-- Represents the distribution of students in a school's clubs -/
structure SchoolClubs where
  total_students : ℕ
  music_students : ℕ
  recitation_offset : ℕ
  dance_offset : ℤ

/-- Calculates the number of students in the art club -/
def art_club_students (sc : SchoolClubs) : ℤ :=
  sc.total_students - sc.music_students - (sc.music_students / 2 + sc.recitation_offset) - 
  (sc.music_students + 2 * sc.recitation_offset + sc.dance_offset)

/-- Theorem stating the number of students in the art club -/
theorem art_club_theorem (sc : SchoolClubs) 
  (h1 : sc.total_students = 220)
  (h2 : sc.dance_offset = -40) :
  art_club_students sc = 260 - (5/2 : ℚ) * sc.music_students - 3 * sc.recitation_offset :=
by sorry

end NUMINAMATH_CALUDE_art_club_theorem_l1328_132857


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l1328_132811

theorem triangle_side_lengths (a b c : ℝ) (angleC : ℝ) (area : ℝ) :
  a = 3 →
  angleC = 2 * Real.pi / 3 →
  area = 3 * Real.sqrt 3 / 4 →
  1/2 * a * b * Real.sin angleC = area →
  Real.cos angleC = (a^2 + b^2 - c^2) / (2 * a * b) →
  b = 1 ∧ c = Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_lengths_l1328_132811


namespace NUMINAMATH_CALUDE_two_numbers_difference_l1328_132840

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (square_diff : x^2 - y^2 = 24) : 
  |x - y| = 2.4 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l1328_132840


namespace NUMINAMATH_CALUDE_gabriel_capsule_days_l1328_132819

/-- The number of days in July -/
def days_in_july : ℕ := 31

/-- The number of days Gabriel forgot to take his capsules -/
def days_forgot : ℕ := 3

/-- The number of days Gabriel took his capsules in July -/
def days_took_capsules : ℕ := days_in_july - days_forgot

theorem gabriel_capsule_days : days_took_capsules = 28 := by
  sorry

end NUMINAMATH_CALUDE_gabriel_capsule_days_l1328_132819


namespace NUMINAMATH_CALUDE_hydrogen_chloride_production_l1328_132802

/-- Represents the balanced chemical equation for the reaction between methane and chlorine -/
structure BalancedEquation where
  methane : ℕ
  chlorine : ℕ
  tetrachloromethane : ℕ
  hydrogen_chloride : ℕ
  balanced : methane = 1 ∧ chlorine = 4 ∧ tetrachloromethane = 1 ∧ hydrogen_chloride = 4

/-- Represents the given reaction conditions -/
structure ReactionConditions where
  methane : ℕ
  chlorine : ℕ
  tetrachloromethane : ℕ
  methane_eq : methane = 3
  chlorine_eq : chlorine = 12
  tetrachloromethane_eq : tetrachloromethane = 3

/-- Theorem stating that given the reaction conditions, 12 moles of hydrogen chloride are produced -/
theorem hydrogen_chloride_production 
  (balanced : BalancedEquation) 
  (conditions : ReactionConditions) : 
  conditions.methane * balanced.hydrogen_chloride = 12 := by
  sorry

end NUMINAMATH_CALUDE_hydrogen_chloride_production_l1328_132802


namespace NUMINAMATH_CALUDE_min_f_value_l1328_132816

theorem min_f_value (d e f : ℕ+) (h1 : d < e) (h2 : e < f)
  (h3 : ∃! x y : ℝ, 3 * x + y = 3005 ∧ y = |x - d| + |x - e| + |x - f|) :
  1504 ≤ f :=
sorry

end NUMINAMATH_CALUDE_min_f_value_l1328_132816


namespace NUMINAMATH_CALUDE_max_value_xy_8x_y_l1328_132869

theorem max_value_xy_8x_y (x y : ℝ) (h : x^2 + y^2 = 20) :
  x * y + 8 * x + y ≤ 42 := by
  sorry

end NUMINAMATH_CALUDE_max_value_xy_8x_y_l1328_132869


namespace NUMINAMATH_CALUDE_stuffed_animals_problem_l1328_132849

theorem stuffed_animals_problem (num_dogs : ℕ) : 
  (∃ (group_size : ℕ), group_size > 0 ∧ (14 + num_dogs) = 7 * group_size) →
  num_dogs = 7 := by
sorry

end NUMINAMATH_CALUDE_stuffed_animals_problem_l1328_132849


namespace NUMINAMATH_CALUDE_inverse_sum_product_identity_l1328_132845

theorem inverse_sum_product_identity (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (y*z + x*z + x*y) * x⁻¹ * y⁻¹ * z⁻¹ * (x + y + z)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_product_identity_l1328_132845


namespace NUMINAMATH_CALUDE_area_of_overlap_l1328_132800

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shortLeg : ℝ
  longLeg : ℝ
  hypotenuse_eq : hypotenuse = 10
  shortLeg_eq : shortLeg = 5
  longLeg_eq : longLeg = 5 * Real.sqrt 3

/-- Represents the configuration of two overlapping 30-60-90 triangles -/
structure OverlappingTriangles where
  triangle1 : Triangle30_60_90
  triangle2 : Triangle30_60_90
  overlap_angle : ℝ
  overlap_angle_eq : overlap_angle = 60

/-- The theorem to be proved -/
theorem area_of_overlap (ot : OverlappingTriangles) :
  let base := 2 * ot.triangle1.shortLeg
  let height := ot.triangle1.longLeg
  base * height = 50 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_area_of_overlap_l1328_132800


namespace NUMINAMATH_CALUDE_residue_of_11_pow_2016_mod_19_l1328_132829

theorem residue_of_11_pow_2016_mod_19 : 11^2016 % 19 = 17 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_11_pow_2016_mod_19_l1328_132829


namespace NUMINAMATH_CALUDE_jordans_rectangle_width_l1328_132835

theorem jordans_rectangle_width (carol_length carol_width jordan_length : ℕ) 
  (jordan_width : ℕ) : 
  carol_length = 12 → 
  carol_width = 15 → 
  jordan_length = 9 → 
  carol_length * carol_width = jordan_length * jordan_width → 
  jordan_width = 20 := by
sorry

end NUMINAMATH_CALUDE_jordans_rectangle_width_l1328_132835


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l1328_132877

theorem binomial_expansion_sum : 
  let f : ℕ → ℕ → ℕ := λ m n => (Nat.choose 6 m) * (Nat.choose 4 n)
  f 3 0 + f 2 1 + f 1 2 + f 0 3 = 120 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l1328_132877


namespace NUMINAMATH_CALUDE_wednesday_water_intake_l1328_132855

/-- Calculates the water intake for Wednesday given the total weekly intake and intake for other days -/
theorem wednesday_water_intake 
  (total_weekly_intake : ℕ)
  (high_intake_days : ℕ)
  (low_intake_days : ℕ)
  (high_intake_amount : ℕ)
  (low_intake_amount : ℕ)
  (h1 : total_weekly_intake = 60)
  (h2 : high_intake_days = 3)
  (h3 : low_intake_days = 3)
  (h4 : high_intake_amount = 9)
  (h5 : low_intake_amount = 8) :
  total_weekly_intake - (high_intake_days * high_intake_amount + low_intake_days * low_intake_amount) = 9 := by
  sorry

#check wednesday_water_intake

end NUMINAMATH_CALUDE_wednesday_water_intake_l1328_132855


namespace NUMINAMATH_CALUDE_point_three_units_from_negative_two_l1328_132854

theorem point_three_units_from_negative_two (x : ℝ) : 
  (|x - (-2)| = 3) ↔ (x = -5 ∨ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_point_three_units_from_negative_two_l1328_132854


namespace NUMINAMATH_CALUDE_dress_price_after_discounts_l1328_132899

theorem dress_price_after_discounts (d : ℝ) : 
  let initial_discount_rate : ℝ := 0.65
  let staff_discount_rate : ℝ := 0.60
  let price_after_initial_discount : ℝ := d * (1 - initial_discount_rate)
  let final_price : ℝ := price_after_initial_discount * (1 - staff_discount_rate)
  final_price = d * 0.14 :=
by sorry

end NUMINAMATH_CALUDE_dress_price_after_discounts_l1328_132899


namespace NUMINAMATH_CALUDE_money_sharing_problem_l1328_132817

theorem money_sharing_problem (amanda_share ben_share carlos_share total : ℕ) : 
  amanda_share = 30 ∧ 
  ben_share = 2 * amanda_share + 10 ∧
  amanda_share + ben_share + carlos_share = total ∧
  3 * ben_share = 4 * amanda_share ∧
  3 * carlos_share = 9 * amanda_share →
  total = 190 := by sorry

end NUMINAMATH_CALUDE_money_sharing_problem_l1328_132817


namespace NUMINAMATH_CALUDE_monotonic_function_implies_a_le_e_l1328_132882

/-- Given f(x) = 2xe^x - ax^2 - 2ax is monotonically increasing on [1, +∞), prove that a ≤ e -/
theorem monotonic_function_implies_a_le_e (a : ℝ) :
  (∀ x ≥ 1, Monotone (fun x => 2 * x * Real.exp x - a * x^2 - 2 * a * x)) →
  a ≤ Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_function_implies_a_le_e_l1328_132882


namespace NUMINAMATH_CALUDE_imaginary_unit_cube_l1328_132805

theorem imaginary_unit_cube (i : ℂ) (h : i^2 = -1) : 1 + i^3 = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_cube_l1328_132805


namespace NUMINAMATH_CALUDE_polynomial_divisibility_and_divisor_l1328_132871

theorem polynomial_divisibility_and_divisor : ∃ m : ℤ,
  (∀ x : ℝ, (4 * x^2 - 6 * x + m) % (x - 3) = 0) ∧
  m = -18 ∧
  36 % m = 0 := by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_and_divisor_l1328_132871


namespace NUMINAMATH_CALUDE_remainder_problem_l1328_132851

theorem remainder_problem (N : ℕ) (R : ℕ) :
  (∃ q : ℕ, N = 34 * q + 2) →
  (N = 44 * 432 + R) →
  R = 0 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1328_132851


namespace NUMINAMATH_CALUDE_winning_probability_is_five_eighths_l1328_132818

/-- Represents the color of a ball in the lottery bag -/
inductive BallColor
  | Red
  | Yellow
  | White
  | Black

/-- Represents the lottery bag -/
structure LotteryBag where
  total_balls : ℕ
  red_balls : ℕ
  yellow_balls : ℕ
  black_balls : ℕ
  white_balls : ℕ
  h_total : total_balls = red_balls + yellow_balls + black_balls + white_balls

/-- Calculates the probability of winning in the lottery -/
def winning_probability (bag : LotteryBag) : ℚ :=
  (bag.red_balls + bag.yellow_balls + bag.white_balls : ℚ) / bag.total_balls

/-- The lottery bag configuration -/
def lottery_bag : LotteryBag := {
  total_balls := 24
  red_balls := 3
  yellow_balls := 6
  black_balls := 9
  white_balls := 6
  h_total := by rfl
}

/-- Theorem: The probability of winning in the given lottery bag is 5/8 -/
theorem winning_probability_is_five_eighths :
  winning_probability lottery_bag = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_winning_probability_is_five_eighths_l1328_132818


namespace NUMINAMATH_CALUDE_partnership_profit_l1328_132870

/-- A partnership problem with four partners A, B, C, and D -/
theorem partnership_profit (total_capital : ℝ) (total_profit : ℝ) : 
  (1 / 3 : ℝ) * total_capital / total_capital = 810 / total_profit →
  (1 / 3 : ℝ) + (1 / 4 : ℝ) + (1 / 5 : ℝ) + 
    (1 - ((1 / 3 : ℝ) + (1 / 4 : ℝ) + (1 / 5 : ℝ))) = 1 →
  total_profit = 2430 := by
  sorry

#check partnership_profit

end NUMINAMATH_CALUDE_partnership_profit_l1328_132870


namespace NUMINAMATH_CALUDE_dress_discount_price_l1328_132833

/-- The final price of a dress after applying a discount -/
def final_price (original_price discount_percentage : ℚ) : ℚ :=
  original_price * (1 - discount_percentage / 100)

/-- Theorem stating that a dress originally priced at $350 with a 60% discount costs $140 -/
theorem dress_discount_price : final_price 350 60 = 140 := by
  sorry

end NUMINAMATH_CALUDE_dress_discount_price_l1328_132833


namespace NUMINAMATH_CALUDE_parabola_hyperbola_equations_l1328_132832

/-- Given a parabola and a hyperbola with specific properties, prove their equations -/
theorem parabola_hyperbola_equations :
  ∀ (a b : ℝ) (P : ℝ × ℝ),
    a > 0 → b > 0 →
    P = (3/2, Real.sqrt 6) →
    -- Parabola vertex at origin
    -- Directrix of parabola passes through a focus of hyperbola
    -- Directrix perpendicular to line connecting foci of hyperbola
    -- Parabola and hyperbola intersect at P
    ∃ (p : ℝ),
      -- Parabola equation
      (λ (x y : ℝ) => y^2 = 2*p*x) P.1 P.2 ∧
      -- Hyperbola equation
      (λ (x y : ℝ) => x^2/a^2 - y^2/b^2 = 1) P.1 P.2 →
      -- Prove the specific equations
      (λ (x y : ℝ) => y^2 = 4*x) = (λ (x y : ℝ) => y^2 = 2*p*x) ∧
      (λ (x y : ℝ) => 4*x^2 - 4/3*y^2 = 1) = (λ (x y : ℝ) => x^2/a^2 - y^2/b^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_equations_l1328_132832


namespace NUMINAMATH_CALUDE_binomial_coefficient_n_choose_n_minus_one_l1328_132860

theorem binomial_coefficient_n_choose_n_minus_one (n : ℕ+) : 
  Nat.choose n.val (n.val - 1) = n.val := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_choose_n_minus_one_l1328_132860


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l1328_132804

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- The angles are supplementary
  a / b = 5 / 4 →  -- The ratio of the angles is 5:4
  min a b = 80 :=  -- The smaller angle is 80°
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l1328_132804


namespace NUMINAMATH_CALUDE_ellipse_vertex_distance_l1328_132837

/-- The distance between vertices of an ellipse with equation x²/121 + y²/49 = 1 is 22 -/
theorem ellipse_vertex_distance :
  let a : ℝ := Real.sqrt 121
  let b : ℝ := Real.sqrt 49
  let ellipse_equation := fun (x y : ℝ) => x^2 / 121 + y^2 / 49 = 1
  2 * a = 22 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_vertex_distance_l1328_132837


namespace NUMINAMATH_CALUDE_function_characterization_l1328_132813

theorem function_characterization (f : ℕ+ → ℕ+ → ℕ+) :
  (∀ x : ℕ+, f x x = x) →
  (∀ x y : ℕ+, f x y = f y x) →
  (∀ x y : ℕ+, (x + y) * (f x y) = y * (f x (x + y))) →
  (∀ x y : ℕ+, f x y = Nat.lcm x y) :=
by sorry

end NUMINAMATH_CALUDE_function_characterization_l1328_132813


namespace NUMINAMATH_CALUDE_rectangle_partition_l1328_132823

theorem rectangle_partition (n : ℕ) : 
  (∃ (partition : List (ℕ × ℕ)), 
    (∀ (strip : ℕ × ℕ), strip ∈ partition → strip.1 = 1) ∧ 
    (∀ (strip1 strip2 : ℕ × ℕ), strip1 ∈ partition → strip2 ∈ partition → strip1 ≠ strip2 → strip1.2 ≠ strip2.2) ∧
    (List.sum (List.map (λ strip => strip.2) partition) = 1995 * n)) ↔ 
  (1 ≤ n ∧ n ≤ 998) ∨ (n ≥ 3990) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_partition_l1328_132823


namespace NUMINAMATH_CALUDE_math_problems_l1328_132895

theorem math_problems :
  (∀ x : ℝ, x^2 + 2*x + 2 ≥ 0) ∧
  (∃ x y : ℝ, |x| > |y| ∧ x ≤ y) ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 2 ∧ x < 3 → 3*x - a < 0) → a ≥ 9) ∧
  (∀ m : ℝ, (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 - 2*x + m = 0 ∧ y^2 - 2*y + m = 0) ↔ m < 0) :=
by sorry

end NUMINAMATH_CALUDE_math_problems_l1328_132895
