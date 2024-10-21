import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_and_base_area_l989_98917

/-- A triangular pyramid with mutually perpendicular lateral edges -/
structure TriangularPyramid where
  edge1 : ℝ
  edge2 : ℝ
  edge3 : ℝ
  perpendicular : edge1 * edge2 = 0 ∧ edge2 * edge3 = 0 ∧ edge1 * edge3 = 0

/-- The volume of a triangular pyramid -/
noncomputable def volume (p : TriangularPyramid) : ℝ :=
  (1 / 6) * Real.sqrt (p.edge1 * p.edge2 * p.edge3)

/-- The area of the base of a triangular pyramid -/
noncomputable def baseArea (p : TriangularPyramid) : ℝ :=
  let a := Real.sqrt (p.edge1^2 + p.edge2^2)
  let b := Real.sqrt (p.edge2^2 + p.edge3^2)
  let c := Real.sqrt (p.edge1^2 + p.edge3^2)
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem pyramid_volume_and_base_area 
  (p : TriangularPyramid) 
  (h1 : p.edge1 = Real.sqrt 70)
  (h2 : p.edge2 = Real.sqrt 99)
  (h3 : p.edge3 = Real.sqrt 126) :
  volume p = 21 * Real.sqrt 55 ∧ baseArea p = 84 := by
  sorry

-- Remove the #eval statements as they're not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_and_base_area_l989_98917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_two_l989_98907

theorem cube_root_sum_equals_two (x : ℝ) (h_pos : x > 0) 
  (h_eq : (2 - x^3)^(1/3) + (2 + x^3)^(1/3) = 2) : 
  x^6 = 100/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_two_l989_98907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_milk_percentage_l989_98965

/-- Proves that given a 60-liter solution of milk and water, if adding 26.9 liters of water
    results in a new solution where milk forms 58%, then the initial percentage of milk
    in the original solution was approximately 83.99%. -/
theorem initial_milk_percentage (initial_volume : ℝ) (added_water : ℝ) (final_milk_percentage : ℝ) :
  initial_volume = 60 →
  added_water = 26.9 →
  final_milk_percentage = 58 →
  let final_volume := initial_volume + added_water
  let initial_milk_volume := (final_milk_percentage / 100) * final_volume
  let initial_milk_percentage := (initial_milk_volume / initial_volume) * 100
  ∃ ε > 0, |initial_milk_percentage - 83.99| < ε := by
  sorry

#check initial_milk_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_milk_percentage_l989_98965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l989_98905

-- Define the expression as noncomputable
noncomputable def p (a b c x : ℝ) : ℝ :=
  ((x - a)^3 + a*x) / ((a - b)*(a - c)) +
  ((x - b)^3 + b*x) / ((b - a)*(b - c)) +
  ((x - c)^3 + c*x) / ((c - a)*(c - b))

-- State the theorem
theorem simplify_expression (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  ∀ x : ℝ, p a b c x = a + b + c + 3*x + 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l989_98905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_2016_l989_98927

def a (n : ℕ) : ℚ := sorry

def S (n : ℕ) : ℚ := sorry

theorem sequence_sum_2016 :
  (a 1 = 1) →
  (∀ n : ℕ, n ≥ 2 → a n + 2 * S (n - 1) = n) →
  S 2016 = 1008 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_2016_l989_98927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_range_a_l989_98932

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a / x else (2 - 3*a) * x + 1

-- State the theorem
theorem f_decreasing_range_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ 2/3 < a ∧ a ≤ 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_range_a_l989_98932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_determination_l989_98938

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  ma : ℝ
  mb : ℝ
  mc : ℝ

def uniquely_determines_shape (data : Triangle → ℝ × ℝ × ℝ) : Prop :=
  ∀ t1 t2 : Triangle, data t1 = data t2 → t1 = t2

theorem triangle_shape_determination :
  (∀ t : Triangle, uniquely_determines_shape (λ t' => (t'.a, t'.b, t'.c))) ∧
  (∀ t : Triangle, uniquely_determines_shape (λ t' => (t'.α / t'.β, t'.c, 0))) ∧
  (∀ t : Triangle, uniquely_determines_shape (λ t' => (t'.a * t'.b * Real.cos t'.γ, 0, 0))) ∧
  (∃ t1 t2 : Triangle, t1.ma / t1.mb = t2.ma / t2.mb ∧ t1 ≠ t2) ∧
  (∀ t : Triangle, uniquely_determines_shape (λ t' => (t'.α, t'.β, t'.γ))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_determination_l989_98938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_AOB_is_right_angle_l989_98970

noncomputable section

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 8/3

-- Define the eccentricity of the ellipse
def eccentricity (a b : ℝ) : ℝ := Real.sqrt ((a^2 - b^2) / a^2)

-- Define the point M
def point_M : ℝ × ℝ := (Real.sqrt 6, 1)

-- Define the angle between two vectors
def angle (v w : ℝ × ℝ) : ℝ := Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

-- Define the theorem
theorem angle_AOB_is_right_angle 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : eccentricity a b = Real.sqrt 2 / 2) 
  (h4 : ellipse a b point_M.1 point_M.2) 
  (A B : ℝ × ℝ) 
  (h5 : A ≠ B) 
  (h6 : ellipse a b A.1 A.2) 
  (h7 : ellipse a b B.1 B.2) 
  (l : ℝ → ℝ) 
  (h8 : ∃ t : ℝ, circle_O t (l t)) : 
  angle (A.1, A.2) (B.1, B.2) = π / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_AOB_is_right_angle_l989_98970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_preserved_l989_98992

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def move_segment (n : ℕ) (k : ℕ) : ℕ :=
  let digits := Nat.digits 10 n
  let len := digits.length
  (digits.drop (len - k) ++ digits.take (len - k)).foldl (λ acc x => 10 * acc + x) 0

theorem divisibility_preserved (n : ℕ) (h1 : is_five_digit n) (h2 : n % 271 = 0) :
  ∀ k, 0 < k → k < 5 → (move_segment n k) % 271 = 0 :=
by
  sorry

#eval move_segment 98102 2  -- Should output 10298
#eval move_segment 98102 3  -- Should output 2981

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_preserved_l989_98992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_to_focus_l989_98956

/-- Parabola with equation x^2 = 4y, focus F(0,1), and directrix y = -1 -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → ℝ

/-- A point on the parabola -/
structure PointOnParabola (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : p.equation point.1 point.2

/-- Angle between two vectors -/
noncomputable def angle (v1 v2 : ℝ × ℝ) : ℝ := sorry

theorem parabola_distance_to_focus 
  (p : Parabola)
  (point : PointOnParabola p)
  (h_equation : p.equation = fun x y => x^2 = 4*y)
  (h_focus : p.focus = (0, 1))
  (h_directrix : p.directrix = fun x => -1)
  (h_angle : angle (point.point.1 - 0, point.point.2 - (-1)) (0, 2) = π/6) :
  Real.sqrt ((point.point.1 - p.focus.1)^2 + (point.point.2 - p.focus.2)^2) = 4/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_to_focus_l989_98956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l989_98984

-- Define the vectors and function
noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sin x - Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.sin x + Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- Define the triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- State the theorem
theorem triangle_area (t : Triangle) :
  t.a + t.b = 2 * Real.sqrt 3 →
  t.c = Real.sqrt 6 →
  f t.C = 2 →
  (1/2) * t.a * t.b * Real.sin t.C = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l989_98984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l989_98908

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 + x + 6) + |x| / (x - 1)

theorem domain_of_f :
  {x : ℝ | -x^2 + x + 6 ≥ 0 ∧ x ≠ 1} = {x : ℝ | x ∈ Set.Icc (-2) 1 ∪ Set.Ioc 1 3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l989_98908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l989_98922

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ -1 then 2*x + 4 else -x + 1

theorem solution_set_of_inequality (x : ℝ) :
  f x < 4 ↔ x ∈ Set.Ioo (-3 : ℝ) 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l989_98922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l989_98929

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 5 * Real.log x + a * x^2 - 6 * x

def tangent_parallel_to_x_axis (a : ℝ) : Prop :=
  (deriv (f a)) 1 = 0

theorem function_properties (a : ℝ) (h : tangent_parallel_to_x_axis a) :
  a = 1/2 ∧
  (∀ x y, x ∈ Set.Ioo 0 1 → y ∈ Set.Ioo 0 1 → x ≤ y → f (1/2) x ≤ f (1/2) y) ∧
  (∀ x y, x ∈ Set.Ioi 5 → y ∈ Set.Ioi 5 → x ≤ y → f (1/2) x ≤ f (1/2) y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l989_98929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_half_km_l989_98983

/-- The length of a bridge crossed by a man walking at a given speed in a given time -/
noncomputable def bridge_length (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time / 60

/-- Theorem stating that the length of the bridge is 1/2 km -/
theorem bridge_length_is_half_km (speed : ℝ) (time : ℝ) 
  (h_speed : speed = 10) (h_time : time = 3) :
  bridge_length speed time = 1/2 := by
  sorry

#check bridge_length_is_half_km

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_half_km_l989_98983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_year_undeclared_major_l989_98925

/-- Represents the fraction of students in each year --/
structure StudentDistribution where
  firstYear : ℚ
  secondYear : ℚ
  thirdYear : ℚ

/-- Represents the fraction of students who have declared a major in each year --/
structure DeclaredMajorDistribution where
  firstYear : ℚ
  secondYear : ℚ
  thirdYear : ℚ

/-- The main theorem about second-year students without a declared major --/
theorem second_year_undeclared_major
  (dist : StudentDistribution)
  (declared : DeclaredMajorDistribution)
  (h1 : dist.firstYear = 1/2)
  (h2 : dist.secondYear = 2/5)
  (h3 : dist.thirdYear = 1/10)
  (h4 : dist.firstYear + dist.secondYear + dist.thirdYear = 1)
  (h5 : declared.firstYear = 1/5)
  (h6 : declared.secondYear = 1/3 * declared.firstYear)
  (h7 : declared.thirdYear = 3/4 * declared.secondYear) :
  dist.secondYear - declared.secondYear = 1/3 := by
  sorry

#check second_year_undeclared_major

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_year_undeclared_major_l989_98925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_values_from_point_on_terminal_side_l989_98945

theorem trig_values_from_point_on_terminal_side 
  (x : ℝ) (θ : ℝ) (h1 : x ≠ 0) (h2 : Real.cos θ = x / 3) :
  Real.sin θ = -2/3 ∧ |Real.tan θ| = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_values_from_point_on_terminal_side_l989_98945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_equations_of_exp_l989_98937

noncomputable def f (x : ℝ) := Real.exp x

theorem tangent_equations_of_exp :
  (∃ x₀ : ℝ, f x₀ = x₀ + 1 ∧ deriv f x₀ = 1) ∧
  (∃ x₁ : ℝ, f x₁ = x₁ * Real.exp 1 ∧ deriv f x₁ = Real.exp 1) ∧
  (∀ x₂ : ℝ, f x₂ ≠ x₂ - 1 ∨ deriv f x₂ ≠ 1) ∧
  (∀ x₃ : ℝ, f x₃ ≠ x₃ / Real.exp 1 ∨ deriv f x₃ ≠ 1 / Real.exp 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_equations_of_exp_l989_98937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l989_98979

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x
noncomputable def g (x : ℝ) : ℝ := -(1/2) * x^(3/2)

-- State the theorem
theorem function_inequality (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 1, f a x < g x) → a < -3/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l989_98979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_l989_98934

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/36 + y^2/16 = 1

-- Define the right focus F₂
def right_focus (F₂ : ℝ × ℝ) : Prop :=
  ∃ (x : ℝ), F₂ = (x, 0) ∧ x > 0

-- Define points A and B on the ellipse and vertical line through F₂
def points_AB (A B F₂ : ℝ × ℝ) : Prop :=
  ∃ (x y₁ y₂ : ℝ),
    F₂ = (x, 0) ∧
    A = (x, y₁) ∧ B = (x, y₂) ∧
    is_on_ellipse x y₁ ∧ is_on_ellipse x y₂ ∧
    y₁ > 0 ∧ y₂ < 0

-- Define the left focus F₁
def left_focus (F₁ : ℝ × ℝ) : Prop :=
  ∃ (x : ℝ), F₁ = (-x, 0) ∧ x > 0

-- Theorem statement
theorem perimeter_of_triangle (A B F₁ F₂ : ℝ × ℝ) :
  right_focus F₂ →
  points_AB A B F₂ →
  left_focus F₁ →
  abs (A.1 - F₁.1) + abs (A.2 - F₁.2) +
  abs (B.1 - F₁.1) + abs (B.2 - F₁.2) +
  abs (A.1 - B.1) + abs (A.2 - B.2) = 24 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_l989_98934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_determinant_l989_98912

theorem triangle_determinant (A B C : Real) : 
  (A + B + C = π) →  -- Sum of angles in a triangle is π radians
  (A ≠ π/2 ∧ B ≠ π/2 ∧ C ≠ π/2) →  -- Non-right triangle condition
  let M : Matrix (Fin 3) (Fin 3) Real := ![![Real.sin (2*A), 1, 1],
                                           ![1, Real.sin (2*B), 1],
                                           ![1, 1, Real.sin (2*C)]]
  Matrix.det M = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_determinant_l989_98912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l989_98954

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with semi-major axis 2 and semi-minor axis √3 -/
def Ellipse (P : Point) : Prop :=
  P.x^2 / 4 + P.y^2 / 3 = 1

/-- Represents the foci of the ellipse -/
structure Foci where
  F₁ : Point
  F₂ : Point

/-- The angle between PF₁ and PF₂ is π/3 -/
def AngleCondition (P : Point) (foci : Foci) : Prop :=
  sorry -- We'll need to define this properly, but for now we'll use sorry

/-- The area of triangle F₁PF₂ -/
noncomputable def TriangleArea (P : Point) (foci : Foci) : ℝ :=
  sorry -- We'll need to define this properly, but for now we'll use sorry

theorem ellipse_triangle_area 
  (P : Point) (foci : Foci) 
  (h₁ : Ellipse P) 
  (h₂ : AngleCondition P foci) : 
  TriangleArea P foci = Real.sqrt 3 := by
  sorry -- The proof goes here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l989_98954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_transformation_l989_98923

def initial_sum : ℕ := (2018 * 2019) / 2

def final_sum : ℕ := 1009 * (3 + 6054)

def operation (a b : ℤ) : ℤ × ℤ := (5*a - 2*b, 3*a - 4*b)

theorem impossible_transformation : 
  ∀ (grid : List ℤ),
    grid.length = 2018 ∧ 
    (∀ n, n ∈ grid → 1 ≤ n ∧ n ≤ 2018) ∧
    grid.sum = initial_sum →
    ¬∃ (new_grid : List ℤ),
      new_grid.length = 2018 ∧
      new_grid.sum = final_sum ∧
      (∀ k, 1 ≤ k ∧ k ≤ 2018 → 3*k ∈ new_grid) ∧
      (∃ (steps : ℕ), ∃ (operations : List (ℕ × ℕ)),
        operations.length = steps ∧
        (List.foldl (λ g (i, j) => 
          let a := g[i]!
          let b := g[j]!
          g.set i (5*a - 2*b) |>.set j (3*a - 4*b)) grid operations) = new_grid) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_transformation_l989_98923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_is_four_l989_98994

/-- An arithmetic sequence with given conditions -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  first_sum : a 1 + a 2 = 10
  second_sum : a 3 + a 4 = 26
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The slope of the line passing through points P(n, a_n) and Q(n+1, a_{n+1}) -/
def lineSlope (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a (n + 1) - seq.a n

theorem slope_is_four (seq : ArithmeticSequence) : ∀ n : ℕ, lineSlope seq n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_is_four_l989_98994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_value_l989_98962

theorem z_value (z : ℕ) 
  (h1 : (Nat.factors z).length = 18)
  (h2 : 16 ∣ z)
  (h3 : 18 ∣ z) :
  z = 288 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_value_l989_98962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_circle_rotation_max_value_l989_98926

/-- Given a point A on the unit circle and B obtained by rotating OA counterclockwise by π/6,
    if m > 0 and the maximum value of my_A - 2y_B is 2, then m = 2√3 -/
theorem unit_circle_rotation_max_value (x_A y_A x_B y_B m : ℝ) :
  x_A^2 + y_A^2 = 1 →  -- A is on the unit circle
  x_B = x_A * Real.cos (π/6) - y_A * Real.sin (π/6) →  -- B is obtained by rotating OA by π/6
  y_B = x_A * Real.sin (π/6) + y_A * Real.cos (π/6) →
  m > 0 →
  (∀ α : ℝ, m * Real.sin α - 2 * Real.sin (α + π/6) ≤ 2) →  -- maximum value condition
  m = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_circle_rotation_max_value_l989_98926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_weight_probability_l989_98993

noncomputable section

variable {Ω : Type} [MeasurableSpace Ω]
variable (P : Measure Ω)

axiom weight_distribution (h₁ : P (Set.Iio 10) = 0.3) (h₂ : P (Set.Icc 10 30) = 0.4) :
  P (Set.Ioi 30) = 0.3

theorem letter_weight_probability (h₁ : P (Set.Iio 10) = 0.3) (h₂ : P (Set.Icc 10 30) = 0.4) :
  P (Set.Ioi 30) = 0.3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_weight_probability_l989_98993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_franking_bijection_l989_98998

/-- Represents a valid franking in Oddland or Squareland -/
structure Franking where
  stamps : ℕ → ℕ
  total_value : ℕ

/-- Checks if a franking is valid in Oddland -/
def is_valid_oddland (f : Franking) : Prop :=
  (∀ i j, i < j → f.stamps (2*i+1) ≥ f.stamps (2*j+1)) ∧
  (f.total_value = ∑' i, (2*i+1) * f.stamps (2*i+1))

/-- Checks if a franking is valid in Squareland -/
def is_valid_squareland (f : Franking) : Prop :=
  f.total_value = ∑' i, (i+1)^2 * f.stamps ((i+1)^2)

/-- The set of valid frankings in Oddland for a given total value -/
def oddland_frankings (n : ℕ) : Set Franking :=
  {f | f.total_value = n ∧ is_valid_oddland f}

/-- The set of valid frankings in Squareland for a given total value -/
def squareland_frankings (n : ℕ) : Set Franking :=
  {f | f.total_value = n ∧ is_valid_squareland f}

/-- The main theorem: there's a bijection between valid frankings in Oddland and Squareland -/
theorem franking_bijection (n : ℕ) :
  ∃ (φ : oddland_frankings n → squareland_frankings n), Function.Bijective φ := by
  sorry

#check franking_bijection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_franking_bijection_l989_98998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skin_initial_area_area_reduction_formula_holds_l989_98969

/-- The initial area of a magical skin that fulfills wishes --/
def initial_area : ℕ := 42

/-- The area of the skin after 10 wishes --/
def area_after_10_wishes : ℕ := initial_area / 3

/-- The number of regular wishes (reducing area by 1 dm²) before the cherished wish --/
def regular_wishes_before : ℕ := 4

/-- The cherished wish reduces the area by half --/
def cherished_wish_reduction (area : ℕ) : ℕ := area / 2

/-- The area reduction formula after 10 wishes --/
def area_reduction_formula : Prop :=
  cherished_wish_reduction (initial_area - regular_wishes_before) - (9 - regular_wishes_before) = area_after_10_wishes

/-- The theorem stating the initial area of the skin --/
theorem skin_initial_area : initial_area = 42 := by
  rfl

/-- Proof that the area reduction formula holds --/
theorem area_reduction_formula_holds : area_reduction_formula := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_skin_initial_area_area_reduction_formula_holds_l989_98969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_C_and_D_l989_98931

/-- Given points A, B, C, and D on a number line, prove that the distance between C and D is 6 units -/
theorem distance_between_C_and_D (A B C D : ℝ) :
  A = 1 →
  B = 3 →
  C = A - 2 →
  D = B + 2 →
  |D - C| = 6 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_C_and_D_l989_98931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l989_98941

noncomputable def f (x : ℝ) : ℝ := 4 * (Real.sin x)^3 * (Real.cos x) - 2 * (Real.sin x) * (Real.cos x) - 1/2 * Real.cos (4*x)

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ T = π/2 ∧ ∀ x : ℝ, f (x + T) = f x) ∧
  (∀ k : ℤ, ∀ x y : ℝ, k*π/2 + π/16 ≤ x ∧ x < y ∧ y ≤ k*π/2 + 5*π/16 → f x < f y) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ π/4 → f x ≤ 1/2) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ π/4 → f x ≥ -Real.sqrt 2 / 2) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ π/4 ∧ f x = 1/2) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ π/4 ∧ f x = -Real.sqrt 2 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l989_98941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_division_integer_side_l989_98974

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a division of a rectangle into smaller rectangles -/
structure RectangleDivision where
  original : Rectangle
  parts : List Rectangle

/-- Predicate to check if a rectangle has at least one integer side length -/
def hasIntegerSide (r : Rectangle) : Prop :=
  ∃ n : ℤ, (r.width = n) ∨ (r.height = n)

/-- Theorem statement -/
theorem rectangle_division_integer_side 
  (div : RectangleDivision) 
  (h : ∀ r ∈ div.parts, hasIntegerSide r) : 
  hasIntegerSide div.original := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_division_integer_side_l989_98974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l989_98944

theorem similar_triangles_side_length 
  (A₁ A₂ : ℝ) 
  (s₂ : ℝ) 
  (h_diff : A₁ - A₂ = 32) 
  (h_ratio : A₁ / A₂ = 9) 
  (h_integer : ∃ n : ℤ, A₂ = n) 
  (h_side : s₂ = 4) : 
  ∃ (s₁ : ℝ), s₁ = 12 ∧ (s₁ / s₂)^2 = A₁ / A₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l989_98944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cotangent_product_minimum_l989_98999

theorem triangle_cotangent_product_minimum (A B C : ℝ) :
  A > 0 → B > 0 → C > 0 → A + B + C = π →
  |((Real.tan A)⁻¹ + (Real.tan B)⁻¹) * ((Real.tan B)⁻¹ + (Real.tan C)⁻¹) * ((Real.tan C)⁻¹ + (Real.tan A)⁻¹)| ≥ 8 * Real.sqrt 3 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cotangent_product_minimum_l989_98999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_2x_plus_exp_l989_98935

theorem definite_integral_2x_plus_exp : ∫ (x : ℝ) in Set.Icc 0 1, (2 * x + Real.exp x) = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_2x_plus_exp_l989_98935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_implies_a_inequality_holds_condition_implies_a_range_l989_98916

noncomputable section

variable (a : ℝ)
variable (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x

theorem tangent_slope_implies_a (h1 : a > 0) (h2 : deriv (f a) 2 = 2) : a = 4 := by
  sorry

theorem inequality_holds (h : a > 0) (hx : x > 0) : f a x ≥ a * (1 - 1 / x) := by
  sorry

theorem condition_implies_a_range (h : a > 0) 
  (h_ineq : ∀ x, 1 < x → x < Real.exp 1 → f a x / (x - 1) > 1) : 
  a ≥ Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_implies_a_inequality_holds_condition_implies_a_range_l989_98916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ababi_ululu_equivalence_l989_98949

-- Define the alphabet
inductive Letter : Type
| A : Letter
| B : Letter

-- Define a word as a list of letters
def Word := List Letter

-- Define the complement of a letter
def complement : Letter → Letter
| Letter.A => Letter.B
| Letter.B => Letter.A

-- Define the complement of a word
def wordComplement : Word → Word
| [] => []
| (h :: t) => complement h :: wordComplement t

-- Define concatenation for Ababi language
def ababiConcat : Word → Word → Word := List.append

-- Define concatenation for Ululu language
def ululuConcat : Word → Word → Word
| [], [] => []
| (h1 :: t1), (h2 :: t2) => h1 :: h2 :: ululuConcat t1 t2
| _, _ => []  -- This case should not occur if words have equal length

-- Define the Ababi language
inductive Ababi : Word → Prop
| base : Ababi [Letter.A]
| concat_same (w : Word) : Ababi w → Ababi (ababiConcat w w)
| concat_complement (w : Word) : Ababi w → Ababi (ababiConcat w (wordComplement w))

-- Define the Ululu language
inductive Ululu : Word → Prop
| base : Ululu [Letter.A]
| concat_same (w : Word) : Ululu w → Ululu (ululuConcat w w)
| concat_complement (w : Word) : Ululu w → Ululu (ululuConcat w (wordComplement w))

-- Theorem statement
theorem ababi_ululu_equivalence : ∀ w : Word, Ababi w ↔ Ululu w := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ababi_ululu_equivalence_l989_98949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_closest_l989_98996

noncomputable section

/-- The line is defined by the equation y = (x - 3) / 3 -/
def line (x : ℝ) : ℝ := (x - 3) / 3

/-- The point we're measuring distance from -/
def point : ℝ × ℝ := (4, -2)

/-- The claimed closest point on the line -/
def closest_point : ℝ × ℝ := (33/10, 1/10)

/-- Theorem stating that closest_point is on the line and is the closest to point -/
theorem closest_point_is_closest :
  (line closest_point.fst = closest_point.snd) ∧
  (∀ x : ℝ, (x - closest_point.fst)^2 + (line x - closest_point.snd)^2 ≤ 
            (x - point.fst)^2 + (line x - point.snd)^2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_closest_l989_98996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_paint_intensity_l989_98988

/-- 
Given an initial paint with 50% intensity, when 80% of it is replaced with a 25% intensity paint, 
the resulting mixture has 30% intensity.
-/
theorem new_paint_intensity 
  (initial_intensity : ℝ) 
  (replacement_intensity : ℝ) 
  (replaced_fraction : ℝ) 
  (h1 : initial_intensity = 0.5)
  (h2 : replacement_intensity = 0.25)
  (h3 : replaced_fraction = 0.8) : 
  (1 - replaced_fraction) * initial_intensity + replaced_fraction * replacement_intensity = 0.3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_paint_intensity_l989_98988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_39_36_15_l989_98942

/-- The area of a triangle given its side lengths using Heron's formula -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: The area of a triangle with sides 39, 36, and 15 is 270 -/
theorem triangle_area_39_36_15 :
  triangle_area 39 36 15 = 270 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_39_36_15_l989_98942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_probability_l989_98946

-- Define the coordinates as a pair of natural numbers
def Coord := ℕ × ℕ

-- Define the possible directions for each object
inductive Direction
| Right
| Up
| Left
| Down

-- Define the objects
structure Object where
  start : Coord
  moveDirection : Direction → Bool

-- Define the problem setup
def setupProblem : Object × Object := 
  ({ start := (0, 0), 
     moveDirection := fun d => match d with
       | Direction.Right => true
       | Direction.Up => true
       | _ => false },
   { start := (5, 7), 
     moveDirection := fun d => match d with
       | Direction.Left => true
       | Direction.Down => true
       | _ => false })

-- Define the number of steps
def numSteps : ℕ := 6

-- Define the function to calculate the number of ways to reach a point
def waysToReach (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- Define the theorem
theorem meeting_probability : 
  let (objA, objB) := setupProblem
  let totalPaths := (2^numSteps : ℚ) * (2^numSteps : ℚ)
  let meetingWays := (List.range (numSteps + 1)).map (fun i => 
    (waysToReach numSteps i : ℚ) * (waysToReach numSteps (numSteps - i) : ℚ)) |>.sum
  meetingWays / totalPaths = 792 / 4096 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_probability_l989_98946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l989_98920

/-- The eccentricity of an ellipse with equation (x²/a² + y²/b² = 1) is √(1 - b²/a²) -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

/-- Given an ellipse with equation x²/25 + y²/9 = 1, its eccentricity is 4/5 -/
theorem ellipse_eccentricity : eccentricity 5 3 = 4/5 := by
  -- Unfold the definition of eccentricity
  unfold eccentricity
  -- Simplify the expression
  simp [Real.sqrt_eq_rpow]
  -- The rest of the proof
  sorry

#check ellipse_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l989_98920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_4500_simplification_l989_98906

theorem cube_root_4500_simplification (a b : ℕ) :
  (a > 0) → (b > 0) → (∀ k : ℕ, k > 0 → k^3 ∣ b → k = 1) → 
  (4500 : ℝ)^(1/3) = a * b^(1/3) → a + b = 31 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_4500_simplification_l989_98906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_transformation_theorem_minimum_distance_theorem_l989_98924

-- Part I
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

def scaling_transformation (lambda : ℝ) (x y x' y' : ℝ) : Prop :=
  x' = lambda * x ∧ y' = 3 * y ∧ lambda > 0

def transformed_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (e : ℝ) (a b : ℝ) : Prop :=
  e = Real.sqrt (1 - b^2 / a^2)

theorem circle_transformation_theorem :
  ∃ lambda : ℝ, ∀ x y x' y' a b : ℝ,
    circle_eq x y →
    scaling_transformation lambda x y x' y' →
    transformed_ellipse (2*lambda) 6 x' y' →
    eccentricity (4/5) (2*lambda) 6 →
    lambda = 5 := by sorry

-- Part II
def point_A : ℝ × ℝ := (2, 0)

noncomputable def curve_C (ρ θ : ℝ) : Prop :=
  ρ = (2 + 2 * Real.cos θ) / Real.sin θ^2

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem minimum_distance_theorem :
  ∃ min_dist : ℝ, 
    (∀ ρ θ : ℝ, curve_C ρ θ → 
      distance (ρ * Real.cos θ) (ρ * Real.sin θ) (point_A.1) (point_A.2) ≥ min_dist) ∧
    min_dist = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_transformation_theorem_minimum_distance_theorem_l989_98924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_range_of_c_l989_98990

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * x * Real.log x + 2

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    y = m * x + b ↔ y - f 1 = (deriv f 1) * (x - 1) :=
sorry

-- Theorem for the range of c
theorem range_of_c :
  ∀ (c : ℝ), (∀ (x : ℝ), x > 1 → f x ≤ x^2 - c*x) →
    c ≤ 1 - 3 * Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_range_of_c_l989_98990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_printer_ink_problem_l989_98958

theorem printer_ink_problem (initial_money : ℕ) 
  (black_ink_cost red_ink_cost yellow_ink_cost : ℕ) 
  (black_ink_count red_ink_count yellow_ink_count : ℕ) : ℕ :=
  let initial_money := 50
  let black_ink_cost := 11
  let red_ink_cost := 15
  let yellow_ink_cost := 13
  let black_ink_count := 2
  let red_ink_count := 3
  let yellow_ink_count := 2

  let total_cost := black_ink_cost * black_ink_count + 
                    red_ink_cost * red_ink_count + 
                    yellow_ink_cost * yellow_ink_count

  let additional_money_needed := total_cost - initial_money

  have : additional_money_needed = 43 := by sorry

  additional_money_needed


end NUMINAMATH_CALUDE_ERRORFEEDBACK_printer_ink_problem_l989_98958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partner_investment_time_l989_98960

/-- Given two partners p and q with investments and profits, calculate q's investment time --/
theorem partner_investment_time 
  (investment_ratio : ℚ) -- Ratio of p's investment to q's investment
  (profit_ratio : ℚ) -- Ratio of p's profit to q's profit
  (p_time : ℕ) -- Time p invested in months
  (h1 : investment_ratio = 7/5) -- Given investment ratio
  (h2 : profit_ratio = 7/11) -- Given profit ratio
  (h3 : p_time = 5) -- Given p's investment time
  : ℕ := -- q's investment time
by
  -- The actual proof would go here
  sorry

-- Example usage (commented out to avoid evaluation errors)
/- 
#eval partner_investment_time (7/5) (7/11) 5
-/

-- Alternative: use a separate function for computation
def compute_partner_investment_time : ℕ := 11

#eval compute_partner_investment_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partner_investment_time_l989_98960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_MM_proportion_l989_98957

/-- The proportion of red M&Ms -/
def R : ℚ := sorry

/-- The proportion of blue M&Ms -/
def B : ℚ := sorry

/-- The probability of a blue M&M following a red M&M -/
def p_red_blue : ℚ := 4/5

/-- The probability of a red M&M following a blue M&M -/
def p_blue_red : ℚ := 1/6

theorem MM_proportion :
  R + B = 1 →
  R * p_red_blue = B * p_blue_red →
  R = 5/29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_MM_proportion_l989_98957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_union_B_equals_open_negative_one_to_infinity_l989_98914

-- Define set A
def A : Set ℝ := {y | ∃ x, y = Real.exp (x * Real.log 3)}

-- Define set B
def B : Set ℝ := Set.Ioo (-1 : ℝ) 1

-- Theorem statement
theorem A_union_B_equals_open_negative_one_to_infinity :
  A ∪ B = Set.Ioi (-1 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_union_B_equals_open_negative_one_to_infinity_l989_98914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_time_theorem_l989_98991

/-- Represents a time between 1:00 and 2:00 --/
structure ClockTime where
  minutes : ℚ
  h : 0 < minutes ∧ minutes < 60

/-- The angle of the hour hand relative to 12 o'clock --/
def hourHandAngle (t : ClockTime) : ℚ :=
  30 + t.minutes / 2

/-- The angle of the minute hand relative to 12 o'clock --/
def minuteHandAngle (t : ClockTime) : ℚ :=
  6 * t.minutes

/-- The condition that the angle bisector points to 12 o'clock --/
def angleBisectorAt12 (t : ClockTime) : Prop :=
  hourHandAngle t = 360 - minuteHandAngle t

theorem coffee_time_theorem :
  ∃ (t : ClockTime), angleBisectorAt12 t ∧ 
    (abs (t.minutes - 50.7692307692) < 1/10000) := by
  sorry

#check coffee_time_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_time_theorem_l989_98991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_entree_cost_is_twenty_l989_98900

/-- Represents the cost of a meal with two identical entrees -/
structure MealCost where
  appetizer : ℚ
  entree : ℚ
  dessert : ℚ
  tipPercent : ℚ

/-- Calculates the total cost of the meal including tip -/
def totalCost (meal : MealCost) : ℚ :=
  let subtotal := meal.appetizer + 2 * meal.entree + meal.dessert
  subtotal * (1 + meal.tipPercent)

/-- Proves that the entree cost is $20.00 given the specified conditions -/
theorem entree_cost_is_twenty (meal : MealCost)
  (h_appetizer : meal.appetizer = 9)
  (h_dessert : meal.dessert = 11)
  (h_tip : meal.tipPercent = (3 : ℚ) / 10)
  (h_total : totalCost meal = 78) :
  meal.entree = 20 := by
  sorry

/-- Example calculation -/
def example_meal : MealCost :=
  { appetizer := 9, entree := 20, dessert := 11, tipPercent := (3 : ℚ) / 10 }

#eval totalCost example_meal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_entree_cost_is_twenty_l989_98900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_remainder_euler_theorem_sum_remainder_l989_98919

theorem geometric_sum_remainder (n : ℕ) (a r : ℤ) (m : ℕ) (h : m > 0) :
  (r^(n+1) - 1) / (r - 1) % m = ((r^(n+1) - 1) / (r - 1)) % m :=
sorry

theorem euler_theorem (a m : ℕ) (h : Nat.Coprime a m) :
  a ^ (Nat.totient m) % m = 1 :=
sorry

def S : ℤ := (9^1001 - 1) / 8

theorem sum_remainder :
  S % 1000 = 96 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_remainder_euler_theorem_sum_remainder_l989_98919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_circle_passes_origin_max_area_triangle_l989_98985

/-- Ellipse C with equation x²/12 + y²/3 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 12 + y^2 / 3 = 1

/-- Line l with equation x = my + 3 -/
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m * y + 3

/-- The eccentricity of ellipse C is √3/2 -/
axiom eccentricity : (Real.sqrt 3) / 2 = (Real.sqrt (12 - 3)) / (2 * Real.sqrt 3)

/-- The left vertex of ellipse C is on the circle x² + y² = 12 -/
axiom left_vertex : ∃ (x y : ℝ), ellipse_C x y ∧ x^2 + y^2 = 12 ∧ x < 0

theorem intersection_circle_passes_origin (m : ℝ) :
  m ≠ 0 →
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
    line_l m x₁ y₁ ∧ line_l m x₂ y₂ ∧
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    x₁ * x₂ + y₁ * y₂ = 0) →
  m = Real.sqrt (11/4) ∨ m = -Real.sqrt (11/4) := by
  sorry

theorem max_area_triangle (m : ℝ) :
  m ≠ 0 →
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
    line_l m x₁ y₁ ∧ line_l m x₂ y₂ ∧
    x₁ ≠ x₂ ∧ y₁ ≠ y₂) →
  (∀ (A : ℝ), 
    A = (1/2) * |((4 - x₁) * y₂ - (4 - x₂) * y₁)| →
    A ≤ 1) ∧
  (∃ (m₀ : ℝ), 
    m₀ ≠ 0 ∧
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
      line_l m₀ x₁ y₁ ∧ line_l m₀ x₂ y₂ ∧
      x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
      (1/2) * |((4 - x₁) * y₂ - (4 - x₂) * y₁)| = 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_circle_passes_origin_max_area_triangle_l989_98985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_position_after_541_transformations_l989_98913

/-- Represents the possible orientations of the rectangle --/
inductive RectangleOrientation
  | WXYZ
  | YZXW

/-- Represents a transformation of the rectangle --/
def transform (o : RectangleOrientation) : RectangleOrientation :=
  match o with
  | RectangleOrientation.WXYZ => RectangleOrientation.YZXW
  | RectangleOrientation.YZXW => RectangleOrientation.WXYZ

/-- Applies n transformations to the initial orientation --/
def applyTransformations (n : Nat) : RectangleOrientation :=
  if n % 2 = 0 then RectangleOrientation.WXYZ else RectangleOrientation.YZXW

theorem rectangle_position_after_541_transformations :
  applyTransformations 541 = RectangleOrientation.YZXW :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_position_after_541_transformations_l989_98913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_for_three_intersections_vector_relationship_l989_98977

/-- The curve M consisting of two parabolas -/
def M (x y : ℝ) : Prop :=
  x^2 = -y ∨ x^2 = 4*y

/-- The line l with slope k and y-intercept -3 -/
def l (k x y : ℝ) : Prop :=
  y = k*x - 3

/-- The number of intersection points between M and l -/
noncomputable def intersection_count (k : ℝ) : ℕ :=
  sorry

/-- Point F -/
def F : ℝ × ℝ := (0, 1)

/-- Theorem stating the minimum value of k for at least 3 intersections -/
theorem min_k_for_three_intersections :
  ∀ k > 0, (∀ k' > 0, intersection_count k' ≥ 3 → k ≤ k') ↔ k = Real.sqrt 3 :=
by sorry

/-- Theorem proving the relationship between vectors when there are exactly 3 intersections -/
theorem vector_relationship (k : ℝ) (A B C : ℝ × ℝ) :
  k = Real.sqrt 3 →
  intersection_count k = 3 →
  M A.1 A.2 →
  M B.1 B.2 →
  M C.1 C.2 →
  l k A.1 A.2 →
  l k B.1 B.2 →
  l k C.1 C.2 →
  A.1 > 0 →
  A.2 > 0 →
  (B.1 - F.1) * (C.1 - F.1) + (B.2 - F.2) * (C.2 - F.2) =
  (A.1 - F.1)^2 + (A.2 - F.2)^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_for_three_intersections_vector_relationship_l989_98977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_100_value_l989_98980

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

noncomputable def sequence_x (n : ℕ) : ℝ :=
  let a := geometric_sequence 1 2
  let b := arithmetic_sequence 2 5
  sorry

theorem x_100_value :
  sequence_x 100 = 2^397 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_100_value_l989_98980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_warehouse_optimization_l989_98987

-- Define the constants
noncomputable def total_investment : ℝ := 3200
noncomputable def iron_cost_per_meter : ℝ := 40
noncomputable def brick_cost_per_meter : ℝ := 45
noncomputable def roof_cost_per_square_meter : ℝ := 20

-- Define the function for the side wall length
noncomputable def f (x : ℝ) : ℝ := (320 - 4*x) / (9 + 2*x)

-- Define the surface area function
noncomputable def surface_area (x : ℝ) : ℝ := x * f x

-- Theorem statement
theorem warehouse_optimization :
  ∃ (x : ℝ), 
    0 < x ∧ 
    x < 80 ∧ 
    surface_area x ≤ 100 ∧
    (∀ (y : ℝ), 0 < y ∧ y < 80 → surface_area y ≤ surface_area x) ∧
    x = 15 ∧
    surface_area x = 100 :=
by
  sorry

#check warehouse_optimization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_warehouse_optimization_l989_98987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l989_98939

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

-- State the theorem
theorem f_properties :
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f (-x) = -f x) ∧  -- f is odd on [-1, 1]
  f 1 = 1/2 ∧  -- f(1) = 1/2
  (∀ x y, x ∈ Set.Ioo (-1 : ℝ) 1 → y ∈ Set.Ioo (-1 : ℝ) 1 → x < y → f x < f y)  -- f is monotonically increasing on (-1, 1)
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l989_98939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_divisibility_l989_98953

def c (n i : ℕ) : ℕ := (Nat.choose n i) % 2

def f (n q : ℕ) : ℕ := Finset.sum (Finset.range (n + 1)) (λ i => (c n i) * q^i)

theorem f_divisibility (m n q r : ℕ) (hq : ∀ α : ℕ, q + 1 ≠ 2^α) (h : f m q ∣ f n q) :
  f m r ∣ f n r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_divisibility_l989_98953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_sec_equation_l989_98997

theorem smallest_positive_solution_tan_sec_equation :
  ∃ (x : ℝ), x > 0 ∧ x = π / 26 ∧
  (∀ y : ℝ, y > 0 ∧ Real.tan (4 * y) + Real.tan (5 * y) = 1 / Real.cos (5 * y) → x ≤ y) ∧
  Real.tan (4 * (π / 26)) + Real.tan (5 * (π / 26)) = 1 / Real.cos (5 * (π / 26)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_sec_equation_l989_98997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l989_98973

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := x / Real.log (x + 2)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > -2 ∧ x ≠ -1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l989_98973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_and_area_l989_98904

-- Define the line l
def line_l (x y : ℝ) : Prop := 4 * x + 3 * y - 8 = 0

-- Define the circle C
def circle_C (x y a : ℝ) : Prop := x^2 + y^2 - a * x = 0

-- Define that line l passes through the center of circle C
def line_passes_through_center (a : ℝ) : Prop :=
  ∃ (x y : ℝ), line_l x y ∧ circle_C x y a ∧ x = a / 2 ∧ y = 0

-- Define point P
noncomputable def point_P : ℝ × ℝ := (1, Real.sqrt 3)

-- Define that P is on circle C
def P_on_circle_C (a : ℝ) : Prop :=
  let (x, y) := point_P
  circle_C x y a

-- Main theorem
theorem circle_tangent_and_area :
  ∃ (a : ℝ),
    line_passes_through_center a ∧
    P_on_circle_C a ∧
    (∀ (x y : ℝ), circle_C x y a ↔ x^2 + y^2 - 4*x = 0) ∧
    (∀ (x y : ℝ), x - Real.sqrt 3 * y + 2 = 0 ↔
      (∃ (t : ℝ), x = 1 + t ∧ y = Real.sqrt 3 + (Real.sqrt 3 / 3) * t)) ∧
    (∃ (A B : ℝ × ℝ),
      (let (xa, ya) := A; line_l xa ya ∧ circle_C xa ya a) ∧
      (let (xb, yb) := B; line_l xb yb ∧ circle_C xb yb a) ∧
      (16 : ℝ) / 5 = (1 / 2) * Real.sqrt ((xa - xb)^2 + (ya - yb)^2) * (8 / 5)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_and_area_l989_98904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_norm_given_condition_l989_98964

open Real NNReal

/-- Given a vector v in ℝ², prove that if ‖v + (4, 2)‖ = 10, then the minimum value of ‖v‖ is 10 - 2√5 -/
theorem min_norm_given_condition (v : ℝ × ℝ) :
  ‖v + (4, 2)‖ = 10 → ∃ (min_norm : ℝ), min_norm = 10 - 2 * Real.sqrt 5 ∧ ∀ w : ℝ × ℝ, ‖w + (4, 2)‖ = 10 → ‖w‖ ≥ min_norm := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_norm_given_condition_l989_98964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l989_98995

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given triangle satisfies the specified conditions -/
noncomputable def given_triangle : Triangle where
  a := 2 * Real.sqrt 2
  b := 5
  c := Real.sqrt 13
  A := Real.arcsin ((2 * Real.sqrt 13) / 13)
  B := Real.arcsin ((5 * Real.sin (Real.pi/4)) / (Real.sqrt 13))
  C := Real.pi/4

theorem triangle_properties (t : Triangle) (h : t = given_triangle) :
  t.C = Real.pi/4 ∧ 
  Real.sin t.A = (2 * Real.sqrt 13) / 13 ∧
  Real.sin (2 * t.A + Real.pi/4) = (17 * Real.sqrt 2) / 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l989_98995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_score_reversal_difference_only_198_possible_l989_98981

theorem score_reversal_difference (x y t : ℕ) 
  (h1 : x < 10) (h2 : y < 10) (h3 : t < 10) : 
  ∃ k : ℤ, (100 * x + 10 * t + y) - (100 * y + 10 * t + x) = 99 * k ∧ 
  (100 * x + 10 * t + y) - (100 * y + 10 * t + x) = 198 := by
  sorry

theorem only_198_possible : 
  ∀ n : ℕ, n ∈ ({198, 200, 202, 204, 206} : Set ℕ) → 
  (∃ k : ℤ, n = 99 * k) ↔ n = 198 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_score_reversal_difference_only_198_possible_l989_98981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_vertex_to_asymptote_is_correct_l989_98948

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 / 4 = 1

/-- The vertex of the hyperbola -/
noncomputable def vertex : ℝ × ℝ := (Real.sqrt 2, 0)

/-- The asymptote equation -/
def asymptote (x y : ℝ) : Prop := Real.sqrt 2 * x + y = 0

/-- The distance from the vertex to the asymptote -/
noncomputable def distance_vertex_to_asymptote : ℝ := 2 * Real.sqrt 3 / 3

theorem distance_vertex_to_asymptote_is_correct :
  distance_vertex_to_asymptote = 2 * Real.sqrt 3 / 3 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_vertex_to_asymptote_is_correct_l989_98948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l989_98903

noncomputable def hyperbola (m : ℝ) (x y : ℝ) : Prop := x^2 - y^2 / m = 1

def line_l (x y : ℝ) : Prop := 2 * x - 5 * y = 0

def point_A : ℝ × ℝ := (-2, 0)

noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |2 * x - 5 * y| / Real.sqrt (2^2 + (-5)^2)

noncomputable def distance_to_point (x y : ℝ) : ℝ :=
  Real.sqrt ((x - (-2))^2 + y^2)

theorem min_distance_sum (m : ℝ) (x y : ℝ) :
  m > 0 →
  hyperbola m x y →
  x > 0 →
  Real.sqrt (1 + m) = 2 →
  distance_to_line x y + distance_to_point x y ≥ 50 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l989_98903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_n_values_is_three_l989_98911

/-- Piecewise function f(x) --/
noncomputable def f (n k : ℝ) (x : ℝ) : ℝ :=
  if x < n then x^2 + 3 else 3*x + k

/-- Continuity condition for f(x) at x = n --/
def continuous_at_n (n k : ℝ) : Prop :=
  n^2 + 3 = 3*n + k

/-- Theorem stating that the sum of all possible values of n that make f continuous is 3 when k = 8 --/
theorem sum_of_n_values_is_three :
  ∃ (n₁ n₂ : ℝ), continuous_at_n n₁ 8 ∧ continuous_at_n n₂ 8 ∧ n₁ ≠ n₂ ∧ n₁ + n₂ = 3 :=
by
  sorry

#check sum_of_n_values_is_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_n_values_is_three_l989_98911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_matrix_correct_l989_98940

def projection_vector : Fin 2 → ℝ := ![2, -3]

theorem projection_matrix_correct :
  let v := projection_vector
  let P : Matrix (Fin 2) (Fin 2) ℝ := !![4/13, -6/13; -6/13, 9/13]
  ∀ (x : Fin 2 → ℝ),
    P.mulVec x = (((v • x) / (v • v)) • v) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_matrix_correct_l989_98940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_study_group_composition_l989_98921

/-- The number of ways to select participants for a competition --/
def selection_ways (boys girls : ℕ) : ℕ :=
  Nat.choose boys 2 * Nat.choose girls 1 * 6

/-- Theorem stating the possible number of boys in the group --/
theorem study_group_composition :
  ∃ (boys : ℕ), boys ∈ ({5, 6} : Set ℕ) ∧
  selection_ways boys (8 - boys) = 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_study_group_composition_l989_98921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_effectiveness_ranking_l989_98902

/-- Represents the size of a detergent box -/
inductive Size
  | S  -- Small
  | M  -- Medium
  | L  -- Large
  | XL -- Extra Large

/-- Cost of a detergent box -/
noncomputable def cost : Size → ℝ
| Size.S  => 1
| Size.M  => 1.6
| Size.L  => 1.25
| Size.XL => 1.5625

/-- Quantity of detergent in a box -/
noncomputable def quantity : Size → ℝ
| Size.S  => 10
| Size.M  => 13.5
| Size.L  => 15
| Size.XL => 18

/-- Cost-effectiveness of a detergent box -/
noncomputable def costEffectiveness (s : Size) : ℝ := cost s / quantity s

/-- Ranking of sizes by cost-effectiveness -/
def ranking : List Size := [Size.L, Size.XL, Size.S, Size.M]

theorem cost_effectiveness_ranking :
  ∀ (i j : Fin ranking.length), i < j →
  costEffectiveness (ranking.get i) < costEffectiveness (ranking.get j) :=
by
  sorry

#check cost_effectiveness_ranking

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_effectiveness_ranking_l989_98902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_l989_98930

/-- A line with slope k and y-intercept -3√2 -/
def line (k x y : ℝ) : Prop := k * x - y - 3 * Real.sqrt 2 = 0

/-- A circle with radius 3 centered at the origin -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9

/-- The line is tangent to the circle -/
def is_tangent (k : ℝ) : Prop := ∃ x y : ℝ, line k x y ∧ circle_eq x y ∧ 
  ∀ x' y' : ℝ, line k x' y' ∧ circle_eq x' y' → (x' = x ∧ y' = y)

/-- k = 1 is a sufficient but not necessary condition for tangency -/
theorem tangent_condition : 
  (∀ k : ℝ, k = 1 → is_tangent k) ∧ 
  ¬(∀ k : ℝ, is_tangent k → k = 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_l989_98930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l989_98955

noncomputable section

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (m1 : ℝ) (m2 : ℝ) : Prop :=
  m1 = m2

/-- Two lines are coincident if and only if their slopes and y-intercepts are equal -/
def coincident (m1 : ℝ) (b1 : ℝ) (m2 : ℝ) (b2 : ℝ) : Prop :=
  m1 = m2 ∧ b1 = b2

/-- The slope of line l1: mx + 2y - 2 = 0 -/
def slope_l1 (m : ℝ) : ℝ := -m / 2

/-- The slope of line l2: 5x + (m+3)y - 5 = 0 -/
def slope_l2 (m : ℝ) : ℝ := -5 / (m + 3)

/-- The y-intercept of line l1: mx + 2y - 2 = 0 -/
def intercept_l1 : ℝ := 1

/-- The y-intercept of line l2: 5x + (m+3)y - 5 = 0 -/
def intercept_l2 (m : ℝ) : ℝ := 5 / (m + 3)

theorem parallel_lines_m_value :
  ∀ m : ℝ, 
    parallel (slope_l1 m) (slope_l2 m) ∧
    ¬coincident (slope_l1 m) intercept_l1 (slope_l2 m) (intercept_l2 m) →
    m = -5 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l989_98955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_range_l989_98909

noncomputable def f (x a : ℝ) : ℝ := |x + 4/x - a| + a

theorem max_value_implies_range (a : ℝ) :
  (∀ x ∈ Set.Icc 1 4, f x a ≤ 5) ∧ (∃ x ∈ Set.Icc 1 4, f x a = 5) →
  a ∈ Set.Iio (9/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_range_l989_98909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_on_x_axis_l989_98910

/-- Represents a quadratic function of the form y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of the vertex of a quadratic function -/
noncomputable def vertex_x (f : QuadraticFunction) : ℝ := -f.b / (2 * f.a)

/-- The y-coordinate of the vertex of a quadratic function -/
noncomputable def vertex_y (f : QuadraticFunction) : ℝ := f.c - f.b^2 / (4 * f.a)

/-- A quadratic function has its vertex on the x-axis if and only if c = 3 -/
theorem vertex_on_x_axis (f : QuadraticFunction) (h1 : f.a = 3) (h2 : f.b = 6) : 
  vertex_y f = 0 ↔ f.c = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_on_x_axis_l989_98910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_region_formation_l989_98947

theorem region_formation (m n : ℕ) : 
  n * (n + 1) / 2 + 1 + m * (n + 1) = 1992 ↔ 
  (m = 995 ∧ n = 1) ∨ (m = 176 ∧ n = 10) ∨ (m = 80 ∧ n = 21) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_region_formation_l989_98947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_k_l989_98982

def k : ℕ := 10^45 - 46

theorem sum_of_digits_k : (Nat.digits 10 k).sum = 423 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_k_l989_98982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l989_98966

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_problem (a₁ d : ℝ) :
  d < 0 →
  arithmetic_sequence a₁ d 2 * arithmetic_sequence a₁ d 4 = 12 →
  arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 4 = 8 →
  (a₁ = 8 ∧ d = -2) ∧
  (Finset.sum (Finset.range 10) (λ n => arithmetic_sequence a₁ d (n + 1)) = -10) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l989_98966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mia_travel_time_l989_98901

/-- Represents the travel details of Mia's weekend getaway -/
structure TravelDetails where
  interstate_distance : ℕ
  rural_distance : ℕ
  rural_time : ℕ
  speed_ratio : ℕ

/-- Calculates the total travel time given the travel details -/
def total_travel_time (td : TravelDetails) : ℚ :=
  td.rural_time + td.interstate_distance * td.rural_time / (td.speed_ratio * td.rural_distance)

/-- Theorem stating that Mia's total travel time is 64 minutes -/
theorem mia_travel_time :
  let td : TravelDetails := {
    interstate_distance := 80,
    rural_distance := 12,
    rural_time := 24,
    speed_ratio := 4
  }
  total_travel_time td = 64 := by
  -- The proof goes here
  sorry

#eval total_travel_time {
  interstate_distance := 80,
  rural_distance := 12,
  rural_time := 24,
  speed_ratio := 4
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mia_travel_time_l989_98901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_dot_product_l989_98933

/-- Golden ratio -/
noncomputable def φ : ℝ := (Real.sqrt 5 - 1) / 2

theorem golden_section_dot_product (A B C D : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let BD := φ * BC
  let angle_BAC := Real.arccos ((AB^2 + AC^2 - BC^2) / (2 * AB * AC))
  AB = 2 ∧ AC = 3 ∧ angle_BAC = π/3 ∧ BD > BC - BD →
  let AD := (A.1 - D.1, A.2 - D.2)
  (AD.1 * (C.1 - B.1) + AD.2 * (C.2 - B.2)) = (7 * Real.sqrt 5 - 9) / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_dot_product_l989_98933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_output_for_15_l989_98951

noncomputable def function_machine (input : ℝ) : ℝ :=
  let step1 := input * 3
  if step1 ≤ 30 then
    step1 + 10
  else
    step1 / 2

theorem output_for_15 : function_machine 15 = 22.5 := by
  -- Unfold the definition of function_machine
  unfold function_machine
  -- Evaluate the first step
  have step1 : 15 * 3 = 45 := by norm_num
  -- Check the condition
  have condition : 45 > 30 := by norm_num
  -- Simplify based on the condition
  simp [step1, condition]
  -- Prove the final equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_output_for_15_l989_98951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l989_98915

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 3

-- Define the hyperbola
def my_hyperbola (x y a : ℝ) : Prop := x^2 / a^2 - y^2 / 2 = 1

-- Define the line passing through (2,1)
def my_line (x y : ℝ) : Prop := 2*x + y = 3

-- Define the right focus of the hyperbola
noncomputable def right_focus (a : ℝ) : ℝ × ℝ := (Real.sqrt (2 + a^2), 0)

-- Main theorem
theorem hyperbola_asymptotes (a : ℝ) (h1 : a > 0) :
  (∃ (x1 y1 x2 y2 : ℝ), 
    my_circle x1 y1 ∧ my_circle x2 y2 ∧  -- A and B are on the circle
    my_line x1 y1 ∧ my_line x2 y2 ∧      -- A and B are on the line
    my_line (right_focus a).1 (right_focus a).2  -- line passes through right focus
  ) →
  (∃ (k : ℝ), k = 2 * Real.sqrt 2 ∧ 
    (∀ (x y : ℝ), my_hyperbola x y a → (y = k*x ∨ y = -k*x))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l989_98915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_division_theorem_l989_98975

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A line segment with endpoints B and E -/
structure Segment where
  B : ℝ × ℝ
  E : ℝ × ℝ

/-- Represents that a segment divides a triangle into two similar triangles -/
def divides_into_similar_triangles (t : Triangle) (s : Segment) : Prop :=
  sorry

/-- The similarity ratio between two triangles -/
noncomputable def similarity_ratio (t1 t2 : Triangle) : ℝ :=
  sorry

/-- The angles of a triangle -/
noncomputable def triangle_angles (t : Triangle) : ℝ × ℝ × ℝ :=
  sorry

/-- Main theorem: If a segment BE divides triangle ABC into two similar triangles
    with similarity ratio √3, then the angles of ABC are 30°, 60°, and 90° -/
theorem triangle_division_theorem (t : Triangle) (s : Segment) :
  divides_into_similar_triangles t s →
  (∃ t1 t2 : Triangle, similarity_ratio t1 t2 = Real.sqrt 3) →
  triangle_angles t = (π/6, π/3, π/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_division_theorem_l989_98975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_area_theorem_l989_98952

/-- Rectangle ABCD with right triangle ADE -/
structure RectangleWithTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  y : ℝ  -- Changed from ℤ to ℝ

/-- The combined area of rectangle ABCD and triangle ADE -/
noncomputable def combinedArea (r : RectangleWithTriangle) : ℝ :=
  sorry

/-- The theorem stating the combined area -/
theorem combined_area_theorem (r : RectangleWithTriangle) :
  r.A = (10, -30) →
  r.B = (2010, 170) →
  r.D = (12, r.y) →
  r.E = (12, -30) →
  combinedArea r = 40400 + 20 * Real.sqrt 101 := by
  sorry

#check combined_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_area_theorem_l989_98952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dust_concentration_minimum_l989_98976

/-- Dust concentration function -/
noncomputable def dust_concentration (k : ℝ) (x : ℝ) : ℝ :=
  (8 * k) / (x ^ 2) + k / ((3 - x) ^ 2)

/-- Derivative of dust concentration function -/
noncomputable def dust_concentration_derivative (k : ℝ) (x : ℝ) : ℝ :=
  (-16 * k) / (x ^ 3) + (2 * k) / ((3 - x) ^ 3)

theorem dust_concentration_minimum (k : ℝ) (h : k > 0) :
  ∃ (x : ℝ), x > 0 ∧ x < 3 ∧
  ∀ (y : ℝ), y > 0 → y < 3 → dust_concentration k x ≤ dust_concentration k y :=
by
  -- We claim that x = 2 is the point of minimum concentration
  use 2
  constructor
  · -- Prove 2 > 0
    linarith
  constructor
  · -- Prove 2 < 3
    linarith
  · -- Prove that dust_concentration k 2 ≤ dust_concentration k y for all y ∈ (0, 3)
    intro y hy1 hy2
    -- Here we would need to prove the inequality, which requires more advanced techniques
    -- For now, we'll use sorry to skip the proof
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dust_concentration_minimum_l989_98976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l989_98967

theorem log_equation_solution (b : ℝ) (h : Real.log 15625 / Real.log b = -4/3) : 
  b = 1 / Real.sqrt (5^9) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l989_98967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_in_expansion_l989_98961

-- Define the expression
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt x + 3 / x) ^ 6

-- Define the constant term
def constant_term : ℝ := 135

-- Theorem statement
theorem constant_term_in_expansion :
  ∃ (g : ℝ → ℝ), (∀ x, x > 0 → f x = constant_term + x * g x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_in_expansion_l989_98961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l989_98978

-- Define a power function
noncomputable def power_function (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- State the theorem
theorem power_function_value (f : ℝ → ℝ) (α : ℝ) 
  (h1 : f = power_function α) 
  (h2 : f 2 = Real.sqrt 2) : 
  f (1/9) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l989_98978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC₁_l989_98950

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := (x + Real.sqrt 3)^2 + y^2 = 4

-- Define the curve C₂
def C₂ (x y θ : ℝ) : Prop := x = 2 + 2 * Real.cos θ ∧ y = 2 * Real.sin θ

-- Define the line C₃
def C₃ (ρ θ : ℝ) : Prop := θ = Real.pi / 3

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ θ₁ θ₂ : ℝ, C₂ A.1 A.2 θ₁ ∧ C₂ B.1 B.2 θ₂ ∧ C₃ A.1 (Real.pi / 3) ∧ C₃ B.1 (Real.pi / 3)

-- Define the center of circle C₁
noncomputable def C₁_center : ℝ × ℝ := (-Real.sqrt 3, 0)

-- Theorem statement
theorem area_of_triangle_ABC₁ (A B : ℝ × ℝ) (area_triangle : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → ℝ) :
  C₁ C₁_center.1 C₁_center.2 →
  intersection_points A B →
  area_triangle A B C₁_center = 3 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC₁_l989_98950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l989_98971

/-- Given complex numbers satisfying a specific ratio condition, 
    prove that the angle between two vectors is π/2 radians. -/
theorem angle_between_vectors (z₁ z₂ z₃ : ℂ) (a : ℝ) 
  (h₁ : (z₃ - z₁) / (z₂ - z₁) = a * Complex.I)
  (h₂ : a ≠ 0) :
  Complex.arg ((z₃ - z₁) / (z₂ - z₁)) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l989_98971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triplet_sums_l989_98989

theorem triplet_sums (a b c d e : ℚ × ℚ × ℚ) : 
  a = (3/2, 4/3, 13/6) → 
  b = (4, -3, 6) → 
  c = (5/2, 31/10, 7/5) → 
  d = (37/5, -47/5, 9) → 
  e = (-3/4, -9/4, 8) → 
  (a.1 + a.2.1 + a.2.2 ≠ 7 ∧ 
   b.1 + b.2.1 + b.2.2 = 7 ∧ 
   c.1 + c.2.1 + c.2.2 = 7 ∧ 
   d.1 + d.2.1 + d.2.2 = 7 ∧ 
   e.1 + e.2.1 + e.2.2 ≠ 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triplet_sums_l989_98989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_calculation_l989_98928

/-- Represents a segment of the car's journey -/
structure Segment where
  speed : ℝ  -- Speed in kph
  distance : Option ℝ  -- Distance in km, if given
  time : Option ℝ  -- Time in hours, if given

/-- Calculates the total distance traveled -/
noncomputable def totalDistance (segments : List Segment) : ℝ :=
  segments.foldl (fun acc s => acc +
    match s.distance with
    | some d => d
    | none => s.speed * Option.getD s.time 0
  ) 0

/-- Calculates the total time taken -/
noncomputable def totalTime (segments : List Segment) : ℝ :=
  segments.foldl (fun acc s => acc +
    match s.time with
    | some t => t
    | none => Option.getD s.distance 0 / s.speed
  ) 0

/-- The main theorem stating the average speed of the car -/
theorem average_speed_calculation (ε : ℝ) (hε : ε > 0) :
  let segments : List Segment := [
    ⟨45, some 30, none⟩,
    ⟨55, some 35, none⟩,
    ⟨70, none, some 0.5⟩,
    ⟨52, none, some (1/3)⟩,
    ⟨65, some 15, none⟩
  ]
  let avgSpeed := totalDistance segments / totalTime segments
  ‖avgSpeed - 64.82‖ < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_calculation_l989_98928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l989_98943

theorem tan_alpha_value (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.sin α ^ 2 + Real.cos (π / 2 + 2 * α) = 3 / 10) : 
  Real.tan α = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l989_98943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l989_98972

noncomputable def f (x : ℝ) := Real.sin x - (1/2) * Real.cos x

theorem max_value_of_f :
  ∃ (M : ℝ), M = Real.sqrt 5 / 2 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l989_98972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_equality_l989_98963

-- Define the arithmetic sequence
def arithmetic_seq (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

-- Define the geometric sequence
noncomputable def geometric_seq (b₁ : ℝ) (q : ℝ) (m : ℕ) : ℝ := b₁ * q ^ ((m - 1 : ℕ) : ℝ)

theorem sequence_equality (a₁ b₁ d q : ℝ) (h_d : d ≠ 0) :
  a₁ = b₁ →
  arithmetic_seq a₁ d 3 = geometric_seq b₁ q 3 →
  arithmetic_seq a₁ d 7 = geometric_seq b₁ q 5 →
  ∀ (n m : ℕ), arithmetic_seq a₁ d n = geometric_seq b₁ q m ↔ 
    (n : ℝ) = 2 ^ (((m : ℝ) + 1) / 2) - 1 ∧ m % 2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_equality_l989_98963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_florent_winning_strategy_l989_98936

/-- Represents a player in the game -/
inductive Player : Type where
  | Alicia : Player
  | Florent : Player

/-- Represents a move in the game -/
inductive Move : Type where
  | Add4 : Move
  | Div2 : Move
  | Sub2 : Move
  | Mul2 : Move

/-- Represents the state of the game -/
structure GameState where
  current : ℕ
  turn : ℕ
  player : Player

/-- Defines a valid move for a player -/
def validMove (state : GameState) (move : Move) : Prop :=
  match state.player, move with
  | Player.Alicia, Move.Add4 => True
  | Player.Alicia, Move.Div2 => state.current % 2 = 0
  | Player.Florent, Move.Sub2 => True
  | Player.Florent, Move.Mul2 => state.current % 2 = 1
  | _, _ => False

/-- Defines the winning condition for a player -/
def isWinning (state : GameState) : Prop :=
  match state.player with
  | Player.Alicia => state.current ≤ 1
  | Player.Florent => state.current > 1000 ∨ state.turn ≥ 100

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Theorem: Florent has a winning strategy if and only if n is odd or n = 998 -/
theorem florent_winning_strategy (n : ℕ) (h1 : 1 < n) (h2 : n ≤ 1000) :
  (n % 2 = 1 ∨ n = 998) ↔ ∃ (strategy : GameState → Move),
    ∀ (game : GameState),
      game.current = n →
      game.turn = 0 →
      game.player = Player.Alicia →
      ∃ (k : ℕ), isWinning (Nat.iterate (λ g ↦ applyMove g (strategy g)) k game) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_florent_winning_strategy_l989_98936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_63100_l989_98968

/-- The principal amount that satisfies the given interest conditions -/
noncomputable def find_principal (rate : ℝ) (time : ℝ) (interest_difference : ℝ) : ℝ :=
  let simple_interest := rate * time / 100
  let compound_interest := (1 + rate / 100) ^ time - 1
  interest_difference / (compound_interest - simple_interest)

/-- Theorem stating that the principal is $63,100 given the conditions -/
theorem principal_is_63100 :
  find_principal 10 2 631 = 63100 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_63100_l989_98968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_relation_l989_98959

noncomputable section

open Real

theorem triangle_side_relation (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  3 * Real.sin (A/2) * Real.sin (B/2) * Real.cos (C/2) + 
    Real.sin (3*A/2) * Real.sin (3*B/2) * Real.cos (3*C/2) = 0 →
  a^3 + b^3 = c^3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_relation_l989_98959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_C_equation_and_fixed_point_l989_98918

noncomputable section

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := y^2/2 + x^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the point S
def S : ℝ × ℝ := (-1/3, 0)

-- Define the point T
def T : ℝ × ℝ := (1, 0)

-- Define a line passing through S
def line_through_S (k : ℝ) (x y : ℝ) : Prop := y = k*(x + 1/3)

-- Define a circle with diameter AB
def circle_AB (a b : ℝ × ℝ) (x y : ℝ) : Prop :=
  let midpoint := ((a.1 + b.1)/2, (a.2 + b.2)/2)
  let radius := ((a.1 - b.1)^2 + (a.2 - b.2)^2) / 4
  (x - midpoint.1)^2 + (y - midpoint.2)^2 = radius

theorem ellipse_C_equation_and_fixed_point :
  -- The ellipse C has its center at the origin
  -- The eccentricity of ellipse C is √2/2
  -- One focus of ellipse C coincides with the focus of the parabola x² = 4y
  ∃ (c : ℝ × ℝ), c.2 = 1 ∧ c.1 = 0 ∧
  (∀ x y, parabola x y → (x = c.1 ∧ y = c.2)) →
  -- 1. The equation of ellipse C is y²/2 + x² = 1
  (∀ x y, ellipse_C x y ↔ y^2/2 + x^2 = 1) ∧
  -- 2. There exists a fixed point T(1,0) such that for any line l passing through S(-1/3, 0)
  --    and intersecting C at points A and B, the circle with diameter AB always passes through T(1,0)
  (∀ k a b, 
    line_through_S k a.1 a.2 ∧ 
    line_through_S k b.1 b.2 ∧ 
    ellipse_C a.1 a.2 ∧ 
    ellipse_C b.1 b.2 ∧ 
    a ≠ b →
    circle_AB a b T.1 T.2) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_C_equation_and_fixed_point_l989_98918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_freddy_travel_time_l989_98986

/-- Represents the travel information for a person --/
structure TravelInfo where
  distance : ℝ
  time : ℝ

/-- Calculates the average speed given travel information --/
noncomputable def average_speed (info : TravelInfo) : ℝ :=
  info.distance / info.time

theorem freddy_travel_time 
  (eddy_info : TravelInfo)
  (freddy_distance : ℝ)
  (speed_ratio : ℝ)
  (freddy_time : ℝ)
  (h1 : eddy_info.distance = 480)
  (h2 : eddy_info.time = 3)
  (h3 : freddy_distance = 300)
  (h4 : speed_ratio = 2.1333333333333333)
  (h5 : average_speed eddy_info / (freddy_distance / freddy_time) = speed_ratio) :
  freddy_time = 4 := by
  sorry

#check freddy_travel_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_freddy_travel_time_l989_98986
