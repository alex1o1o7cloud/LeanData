import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_implies_x_value_l260_26060

/-- Given two vectors a and b in ℝ², where a = (2, 1) and b = (x, -1),
    if a - b is collinear with b, then x = -2. -/
theorem vector_collinearity_implies_x_value :
  ∀ (x : ℝ),
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![x, -1]
  (∃ (k : ℝ), a - b = k • b) →
  x = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_implies_x_value_l260_26060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_capacity_is_80_percent_l260_26097

/-- Represents the dimensions and draining properties of a pool -/
structure Pool where
  width : ℚ
  length : ℚ
  depth : ℚ
  drain_rate : ℚ
  drain_time : ℚ

/-- Calculates the current capacity of the pool as a percentage -/
def current_capacity_percentage (p : Pool) : ℚ :=
  (p.drain_rate * p.drain_time) / (p.width * p.length * p.depth) * 100

/-- Theorem stating that the current capacity of the given pool is 80% -/
theorem pool_capacity_is_80_percent :
  let p : Pool := {
    width := 40,
    length := 150,
    depth := 10,
    drain_rate := 60,
    drain_time := 800
  }
  current_capacity_percentage p = 80 := by
  sorry

#eval current_capacity_percentage {
  width := 40,
  length := 150,
  depth := 10,
  drain_rate := 60,
  drain_time := 800
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_capacity_is_80_percent_l260_26097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_smallest_angle_sum_l260_26076

/-- Represents a quadrilateral with angles in arithmetic progression -/
structure ArithmeticQuadrilateral where
  θ : ℝ
  d : ℝ
  angle_sum : θ + (θ + d) + (θ + 2*d) + (θ + 3*d) = 360

/-- Represents two similar triangles within the quadrilateral -/
structure SimilarTriangles where
  x : ℝ
  y : ℝ
  p : ℝ
  triangle_sum : x + (x + p) + (x + 2*p) = 180
  angle_equality : x + p = 60

/-- Main theorem stating the largest possible sum of largest and smallest angles -/
theorem largest_smallest_angle_sum
  (quad : ArithmeticQuadrilateral)
  (triangles : SimilarTriangles)
  (h1 : quad.θ = triangles.x + triangles.p)
  (h2 : quad.θ = 90 - (3/2) * quad.d) :
  ∃ (d : ℝ), 
    (60 - (3/2) * d) + (150 - (3/2) * d) = 150 ∧ 
    ∀ (d' : ℝ), (60 - (3/2) * d') + (150 - (3/2) * d') ≤ 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_smallest_angle_sum_l260_26076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_domain_characterization_l260_26022

-- Define the function as noncomputable
noncomputable def f (x : ℝ) := Real.log (x^2 - 4*x + 3) / Real.log (1/3)

-- Define the domain of the function
def domain := {x : ℝ | x^2 - 4*x + 3 > 0}

-- State the theorem
theorem monotonic_increasing_interval :
  ∀ x ∈ domain, ∀ y ∈ domain,
    x < y ∧ y < 1 → f x < f y :=
by
  -- The proof is omitted for now
  sorry

-- Additional theorem to show that the domain is (-∞, 1) ∪ (3, +∞)
theorem domain_characterization :
  ∀ x : ℝ, x ∈ domain ↔ x < 1 ∨ x > 3 :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_domain_characterization_l260_26022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_inequality_l260_26062

theorem line_through_point_inequality (a b θ : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (Real.cos θ) / a + (Real.sin θ) / b = 1) : 
  1 / a^2 + 1 / b^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_inequality_l260_26062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abs_coeff_eq_125_512_l260_26013

/-- The polynomial P(x) -/
noncomputable def P (x : ℝ) : ℝ := 1 + (1/4) * x - (1/8) * x^2

/-- The polynomial Q(x) -/
noncomputable def Q (x : ℝ) : ℝ := P x * P (x^2) * P (x^4)

/-- The sum of absolute values of coefficients of Q(x) -/
noncomputable def sum_abs_coeff : ℝ := Q (-1)

/-- Theorem stating that the sum of absolute values of coefficients of Q(x) equals 125/512 -/
theorem sum_abs_coeff_eq_125_512 : sum_abs_coeff = 125/512 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abs_coeff_eq_125_512_l260_26013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sin_equivalence_l260_26044

theorem triangle_angle_sin_equivalence :
  (∀ (A B : Real), Real.sin A = Real.sin B → A = B) ∧
  (∀ (A B : Real), A = B → Real.sin A = Real.sin B) ∧
  (∀ (A B : Real), Real.sin A ≠ Real.sin B → A ≠ B) ∧
  (∀ (A B : Real), A ≠ B → Real.sin A ≠ Real.sin B) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sin_equivalence_l260_26044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l260_26052

open Real

-- Define the ellipse parameters
noncomputable def a : ℝ := 2
noncomputable def b : ℝ := Real.sqrt 3
noncomputable def c : ℝ := 1

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the moving line
def moving_line (k m x y : ℝ) : Prop := y = k * x + m

-- Define the fixed lines
def fixed_line_1 (x y : ℝ) : Prop := Real.sqrt 3 * x - y = 0
def fixed_line_2 (x y : ℝ) : Prop := Real.sqrt 3 * x + y = 0

-- Define the condition for the moving line to be tangent to the ellipse
def tangent_condition (k m : ℝ) : Prop := m^2 = 3 * k^2 + 4

-- Define the area of triangle OPQ
noncomputable def triangle_area (k m : ℝ) : ℝ := (3 * Real.sqrt 3 * k^2 + 4 * Real.sqrt 3) / (3 - k^2)

-- Main theorem
theorem min_triangle_area :
  ∀ k m : ℝ, -Real.sqrt 3 < k → k < Real.sqrt 3 →
  tangent_condition k m →
  ∃ (min_area : ℝ), min_area = 4 * Real.sqrt 3 / 3 ∧
  ∀ k' m' : ℝ, -Real.sqrt 3 < k' → k' < Real.sqrt 3 →
  tangent_condition k' m' →
  triangle_area k' m' ≥ min_area :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l260_26052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_given_cos_l260_26029

theorem tan_value_given_cos (α : Real) (h1 : Real.cos α = -3/5) (h2 : α ∈ Set.Ioo 0 π) :
  Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_given_cos_l260_26029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_side_sum_l260_26002

/-- A parallelogram with sides measuring 5, 11, 10y-3, and 4x+1 has x + y = 3.3 -/
theorem parallelogram_side_sum (x y : ℝ) : 
  (5 : ℝ) ∈ ({5, 11, 10*y-3, 4*x+1} : Set ℝ) →
  (11 : ℝ) ∈ ({5, 11, 10*y-3, 4*x+1} : Set ℝ) →
  (10*y - 3 : ℝ) ∈ ({5, 11, 10*y-3, 4*x+1} : Set ℝ) →
  (4*x + 1 : ℝ) ∈ ({5, 11, 10*y-3, 4*x+1} : Set ℝ) →
  x + y = 3.3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_side_sum_l260_26002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_transformation_l260_26003

theorem q_transformation (e x z : ℝ) (q : ℝ) (h : q = 5 * e / (4 * x * (z ^ 2))) :
  (5 * (4 * e) / (4 * (2 * x) * ((3 * z) ^ 2))) / q = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_transformation_l260_26003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_from_difference_l260_26083

theorem sin_double_angle_from_difference (θ : ℝ) :
  Real.cos θ - Real.sin θ = 3/5 → Real.sin (2*θ) = 16/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_from_difference_l260_26083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l260_26091

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

-- Define the point P
def point_P : ℝ × ℝ := (-1, 5)

-- Define the tangent lines
def tangent_line_1 (x y : ℝ) : Prop := 5*x + 12*y - 55 = 0
def tangent_line_2 (x : ℝ) : Prop := x = -1

-- Theorem statement
theorem tangent_lines_to_circle :
  (∀ x y, tangent_line_1 x y → my_circle x y → (x, y) = (x, y)) ∧
  (∀ x y, tangent_line_2 x → my_circle x y → (x, y) = (x, y)) ∧
  tangent_line_1 (point_P.1) (point_P.2) ∧
  tangent_line_2 (point_P.1) := by
  sorry

#check tangent_lines_to_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l260_26091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_a_b_range_k_obtuse_angle_l260_26065

-- Define points A, B, C in ℝ³
def A : Fin 3 → ℝ := ![-2, 0, 2]
def B : Fin 3 → ℝ := ![-1, 1, 2]
def C : Fin 3 → ℝ := ![-3, 0, 4]

-- Define vectors a and b
def a : Fin 3 → ℝ := λ i => B i - A i
def b : Fin 3 → ℝ := λ i => C i - A i

-- Part 1: Cosine of angle between a and b
theorem cosine_angle_a_b :
  (a 0 * b 0 + a 1 * b 1 + a 2 * b 2) / 
  (Real.sqrt ((a 0)^2 + (a 1)^2 + (a 2)^2) * Real.sqrt ((b 0)^2 + (b 1)^2 + (b 2)^2)) = -Real.sqrt 10 / 10 := by
  sorry

-- Part 2: Range of k for obtuse angle
theorem range_k_obtuse_angle (k : ℝ) :
  let v1 := λ (i : Fin 3) => k * a i + b i
  let v2 := λ (i : Fin 3) => k * a i - 2 * b i
  (v1 0 * v2 0 + v1 1 * v2 1 + v1 2 * v2 2 < 0 ∧ k ≠ 0) ↔ 
  (k > -5/2 ∧ k < 0) ∨ (k > 0 ∧ k < 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_a_b_range_k_obtuse_angle_l260_26065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_problem_l260_26089

theorem train_distance_problem (speed1 speed2 distance_diff : ℝ) 
  (h1 : speed1 = 20)
  (h2 : speed2 = 25)
  (h3 : distance_diff = 75) : 
  (distance_diff / (speed2 - speed1)) * (speed1 + speed2) = 675 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_problem_l260_26089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daily_sales_extrema_l260_26084

noncomputable def f (t : ℝ) : ℝ := 100 * (1 + 1 / t)

noncomputable def g (t : ℝ) : ℝ := 125 - |t - 25|

noncomputable def w (t : ℝ) : ℝ := f t * g t

theorem daily_sales_extrema :
  ∀ t : ℝ, 1 ≤ t ∧ t ≤ 30 →
  (∀ s : ℝ, 1 ≤ s ∧ s ≤ 30 → w s ≤ 20200) ∧
  (∃ s : ℝ, 1 ≤ s ∧ s ≤ 30 ∧ w s = 20200) ∧
  (∀ s : ℝ, 1 ≤ s ∧ s ≤ 30 → w s ≥ 12100) ∧
  (∃ s : ℝ, 1 ≤ s ∧ s ≤ 30 ∧ w s = 12100) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_daily_sales_extrema_l260_26084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_reflection_l260_26014

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is isosceles right with AB = BC -/
def isIsoscelesRight (t : Triangle) : Prop :=
  (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 = (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2).sqrt

/-- Theorem: In an isosceles right triangle ABC with AB = BC, where a laser is shot from A,
    reflects off mirrors BC and CA, and lands at X on AB, if AB/AX = 64,
    then CA/AY = 2/1, where Y is the point where the laser hits AC -/
theorem laser_reflection (t : Triangle) (X Y : Point) :
  isIsoscelesRight t →
  X.y = t.A.y →
  Y.x = t.A.x →
  distance t.A t.B / distance t.A X = 64 →
  distance t.C t.A / distance t.A Y = 2 / 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_reflection_l260_26014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wangHuaWins_l260_26032

/-- Represents the state of the game -/
structure GameState where
  marbles : ℕ

/-- Represents a player's move -/
inductive Move where
  | take : (n : ℕ) → n > 0 ∧ n ≤ 3 → Move

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.take n _ => ⟨state.marbles - n⟩

/-- Checks if the game is over -/
def isGameOver (state : GameState) : Prop :=
  state.marbles = 0

/-- Represents a winning strategy for the first player -/
def winningStrategy (initialState : GameState) : Prop :=
  ∃ (strategy : GameState → Move),
    ∀ (opponentStrategy : GameState → Move),
      ∃ (gameSequence : ℕ → GameState),
        (gameSequence 0 = initialState) ∧
        (∀ n : ℕ,
          gameSequence (n + 1) =
            if n % 2 = 0
            then applyMove (gameSequence n) (strategy (gameSequence n))
            else applyMove (gameSequence n) (opponentStrategy (gameSequence n))) ∧
        ∃ (k : ℕ), isGameOver (gameSequence k) ∧ k % 2 = 0

theorem wangHuaWins :
  winningStrategy ⟨2002⟩ := by
  sorry

#check wangHuaWins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wangHuaWins_l260_26032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l260_26039

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ¬(a = 3 ∧ b = 4 ∧ c = 7) → a + b > c ∧ b + c > a ∧ a + c > b := by
  sorry

#check triangle_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l260_26039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_perimeter_l260_26069

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  -- Base lengths
  ab : ℝ
  cd : ℝ
  -- Height
  h : ℝ
  -- Conditions
  ab_positive : 0 < ab
  cd_positive : 0 < cd
  h_positive : 0 < h
  cd_greater : ab < cd

/-- The perimeter of an isosceles trapezoid -/
noncomputable def perimeter (t : IsoscelesTrapezoid) : ℝ :=
  t.ab + t.cd + 2 * Real.sqrt ((t.cd - t.ab)^2 / 4 + t.h^2)

/-- Theorem: The perimeter of the specific isosceles trapezoid is 110 + 2√801 -/
theorem specific_trapezoid_perimeter :
  let t : IsoscelesTrapezoid := {
    ab := 40,
    cd := 70,
    h := 24,
    ab_positive := by norm_num,
    cd_positive := by norm_num,
    h_positive := by norm_num,
    cd_greater := by norm_num
  }
  perimeter t = 110 + 2 * Real.sqrt 801 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_perimeter_l260_26069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_is_pi_l260_26047

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (Real.tan x + (Real.tan x)⁻¹)^2

-- State the theorem
theorem period_of_f_is_pi :
  ∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), 0 < q ∧ q < p → ∃ (y : ℝ), f (y + q) ≠ f y :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_is_pi_l260_26047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eddy_travel_time_l260_26072

/-- Represents the travel time and distance for a journey between two cities -/
structure Journey where
  time : ℚ
  distance : ℚ

/-- Calculates the average speed given a journey -/
def averageSpeed (j : Journey) : ℚ := j.distance / j.time

theorem eddy_travel_time (freddy_journey : Journey) (eddy_journey : Journey) 
  (h1 : freddy_journey.time = 4)
  (h2 : freddy_journey.distance = 300)
  (h3 : eddy_journey.distance = 900)
  (h4 : averageSpeed eddy_journey = 4 * averageSpeed freddy_journey) :
  eddy_journey.time = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eddy_travel_time_l260_26072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_point_line_equation_intercept_sum_l260_26037

-- Define a line type
structure Line where
  slope : ℚ
  yIntercept : ℚ

-- Define a point type
structure Point where
  x : ℚ
  y : ℚ

-- Define a function to check if a point is on a line
def isPointOnLine (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.yIntercept

-- Define a function to get the x-intercept of a line
noncomputable def xIntercept (l : Line) : ℚ :=
  -l.yIntercept / l.slope

-- Theorem for the first part of the problem
theorem line_equation_through_point (l : Line) (p : Point) :
  l.slope = 2 ∧ p = Point.mk (-2) 1 ∧ isPointOnLine l p →
  ∀ x y : ℚ, y = 2*x + 5 ↔ isPointOnLine l (Point.mk x y) :=
by sorry

-- Theorem for the second part of the problem
theorem line_equation_intercept_sum (l : Line) :
  l.slope = 2 ∧ xIntercept l + l.yIntercept = 3 →
  ∀ x y : ℚ, y = 2*x + 6 ↔ isPointOnLine l (Point.mk x y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_point_line_equation_intercept_sum_l260_26037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_tangent_properties_l260_26058

-- Define the circle C₂
def C₂ (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 9

-- Define the property for curve C₁
def C₁_property (x y : ℝ) : Prop := |x + 2| = Real.sqrt ((x - 5)^2 + y^2) - 3

-- Define the equation of C₁
def C₁_equation (x y : ℝ) : Prop := y^2 = 20 * x

-- Define the point P
def P (x₀ : ℝ) : ℝ × ℝ := (x₀, -4)

-- Theorem statement
theorem curve_and_tangent_properties 
  (x₀ : ℝ) 
  (h₁ : x₀ ≠ 3 ∧ x₀ ≠ -3) 
  (h₂ : ∀ x y, C₁_property x y ↔ C₁_equation x y) 
  (h₃ : ∃ A B C D : ℝ × ℝ, 
    (C₁_equation A.1 A.2 ∧ C₁_equation B.1 B.2 ∧ 
     C₁_equation C.1 C.2 ∧ C₁_equation D.1 D.2) ∧
    (∃ k₁ k₂ : ℝ, 
      (A.2 + 4 = k₁ * (A.1 - x₀) ∧ C₂ A.1 A.2) ∧
      (B.2 + 4 = k₁ * (B.1 - x₀) ∧ C₂ B.1 B.2) ∧
      (C.2 + 4 = k₂ * (C.1 - x₀) ∧ C₂ C.1 C.2) ∧
      (D.2 + 4 = k₂ * (D.1 - x₀) ∧ C₂ D.1 D.2))) :
  ∃ A B C D : ℝ × ℝ, A.1 * B.1 * C.1 * D.1 = 6400 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_tangent_properties_l260_26058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_slope_angle_theorem_l260_26070

-- Define the points
noncomputable def A : ℝ × ℝ := (5, 2)
noncomputable def B : ℝ × ℝ := (0, -Real.sqrt 3 - 4)
noncomputable def C : ℝ × ℝ := (-1, -4)

-- Define the equations of the lines
def line1 (x y : ℝ) : Prop := 2 * x - 5 * y = 0
def line2 (x y : ℝ) : Prop := 2 * x + y - 12 = 0

-- Define the slope angle range
def slope_angle_range (θ : ℝ) : Prop :=
  (0 ≤ θ ∧ θ ≤ Real.pi / 4) ∨ (2 * Real.pi / 3 ≤ θ ∧ θ < Real.pi)

-- Theorem statement
theorem line_and_slope_angle_theorem :
  (∀ x y : ℝ, line1 x y → (x = A.1 ∧ y = A.2) → y = 2 * x) ∧
  (∀ x y : ℝ, line2 x y → (x = A.1 ∧ y = A.2) → y = 2 * x) ∧
  (∀ θ : ℝ, slope_angle_range θ ↔
    ∃ x y : ℝ, (x - C.1) * (B.2 - A.2) = (y - C.2) * (B.1 - A.1) ∧
               (x - C.1) * (A.2 - C.2) = (y - C.2) * (A.1 - C.1) ∧
               Real.tan θ = (y - C.2) / (x - C.1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_slope_angle_theorem_l260_26070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_worked_18_hours_l260_26068

/-- Represents Jason's work schedule and earnings --/
structure WorkSchedule where
  afterSchoolRate : ℚ
  saturdayRate : ℚ
  totalEarnings : ℚ
  saturdayHours : ℚ

/-- Calculates the total hours worked given a work schedule --/
noncomputable def totalHoursWorked (schedule : WorkSchedule) : ℚ :=
  let saturdayEarnings := schedule.saturdayRate * schedule.saturdayHours
  let afterSchoolEarnings := schedule.totalEarnings - saturdayEarnings
  let afterSchoolHours := afterSchoolEarnings / schedule.afterSchoolRate
  afterSchoolHours + schedule.saturdayHours

/-- Theorem stating that Jason worked 18 hours in total --/
theorem jason_worked_18_hours (schedule : WorkSchedule)
    (h1 : schedule.afterSchoolRate = 4)
    (h2 : schedule.saturdayRate = 6)
    (h3 : schedule.totalEarnings = 88)
    (h4 : schedule.saturdayHours = 8) :
    totalHoursWorked schedule = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_worked_18_hours_l260_26068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fred_payment_l260_26040

/-- The amount Fred paid with given his movie ticket and rental expenses, and the change he received. -/
theorem fred_payment (ticket_price : ℝ) (num_tickets : ℕ) (movie_rental : ℝ) (change : ℝ) :
  ticket_price = 5.92 →
  num_tickets = 2 →
  movie_rental = 6.79 →
  change = 1.37 →
  (ticket_price * (num_tickets : ℝ) + movie_rental + change) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fred_payment_l260_26040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_polygon_sides_max_polygon_sides_proof_l260_26038

/-- The maximum number of sides a polygon can have if each angle is either 172° or 173° --/
theorem max_polygon_sides : ℕ := 54

/-- Proof of the maximum number of sides for a polygon with angles of 172° or 173° --/
theorem max_polygon_sides_proof : 
  ∃ (angles : List ℝ), 
    (∀ a ∈ angles, a = 172 ∨ a = 173) ∧ 
    (angles.sum = 180 * (angles.length - 2)) ∧ 
    angles.length = 54 ∧
    (∀ (other_angles : List ℝ), 
      (∀ a ∈ other_angles, a = 172 ∨ a = 173) → 
      (other_angles.sum = 180 * (other_angles.length - 2)) → 
      other_angles.length ≤ 54) := by
  sorry

/-- The set of possible angle measures --/
def angle_measures : Set ℝ := {172, 173}

/-- The sum of interior angles for an n-sided convex polygon --/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- The condition that all angles must be either 172° or 173° --/
def all_angles_valid (angles : List ℝ) : Prop :=
  ∀ a ∈ angles, a ∈ angle_measures

/-- The condition that the sum of angles must equal the sum of interior angles --/
def sum_angles_valid (angles : List ℝ) : Prop :=
  angles.sum = sum_interior_angles angles.length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_polygon_sides_max_polygon_sides_proof_l260_26038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_prime_factor_of_5_polynomial_five_is_least_prime_factor_least_prime_factor_is_five_l260_26030

theorem least_prime_factor_of_5_polynomial (p : ℕ) :
  (Nat.Prime p ∧ p ∣ (5^6 - 5^4 + 5^2)) → p ≥ 5 := by
  sorry

theorem five_is_least_prime_factor :
  Nat.Prime 5 ∧ 5 ∣ (5^6 - 5^4 + 5^2) := by
  sorry

theorem least_prime_factor_is_five :
  (Nat.minFac (5^6 - 5^4 + 5^2)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_prime_factor_of_5_polynomial_five_is_least_prime_factor_least_prime_factor_is_five_l260_26030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_is_real_l260_26019

-- Define the functions f and g
noncomputable def f (m : ℝ) (x : ℝ) := m^x
noncomputable def g (m : ℝ) (x : ℝ) := x^m

-- Define the propositions p and q
def p (m : ℝ) := ∀ x y, 1 < x ∧ x < y → f m x < f m y
def q (m : ℝ) := ∀ x y, 1 < x ∧ x < y → g m x < g m y

-- State the theorem
theorem range_of_m_is_real :
  (∀ m : ℝ, (p m ∨ q m) ≠ ¬(p m)) →
  ∀ m : ℝ, True := by
  intro h
  intro m
  trivial


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_is_real_l260_26019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_minus_pi_over_four_l260_26000

theorem tan_theta_minus_pi_over_four (θ : Real) 
  (h1 : Real.sin θ = 3/5) 
  (h2 : θ > 0) 
  (h3 : θ < Real.pi/2) : 
  Real.tan (θ - Real.pi/4) = -1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_minus_pi_over_four_l260_26000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_c_value_l260_26042

noncomputable def project (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot := v.1 * w.1 + v.2 * w.2
  let norm_squared := w.1 * w.1 + w.2 * w.2
  (dot / norm_squared * w.1, dot / norm_squared * w.2)

theorem projection_c_value :
  ∃ c : ℝ, project (-5, c) (3, -1) = (1/10, -1/10) ∧ c = -16 := by
  use -16
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_c_value_l260_26042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_inequality_l260_26061

theorem root_product_inequality (a b : ℝ) (t₁ t₂ t₃ t₄ : ℝ) :
  (∀ t : ℝ, t^4 + a*t^3 + b*t^2 = (a+b)*(2*t-1) ↔ t = t₁ ∨ t = t₂ ∨ t = t₃ ∨ t = t₄) →
  0 < t₁ ∧ t₁ < t₂ ∧ t₂ < t₃ ∧ t₃ < t₄ →
  t₁ * t₄ > t₂ * t₃ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_inequality_l260_26061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_in_special_triangle_l260_26017

/-- Given a triangle ABC where c^2 = a^2 + b^2 + ab, prove that the measure of angle C is 120° -/
theorem angle_measure_in_special_triangle (a b c : ℝ) (h : c^2 = a^2 + b^2 + a*b) :
  (a^2 + b^2 - c^2) / (2*a*b) = -1/2 ∧ 0 < a ∧ 0 < b ∧ 0 < c → 
  ∃ (C : ℝ), C = 120 * π / 180 ∧ Real.cos C = (a^2 + b^2 - c^2) / (2*a*b) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_in_special_triangle_l260_26017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_plus_pi_half_implies_reciprocal_sin_cos_l260_26018

theorem tan_plus_pi_half_implies_reciprocal_sin_cos (x : ℝ) :
  Real.tan (x + π/2) = 5 → 1 / (Real.sin x * Real.cos x) = -26/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_plus_pi_half_implies_reciprocal_sin_cos_l260_26018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l260_26086

/-- The ellipse defined by (x²/16) + (y²/4) = 1 -/
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

/-- The line defined by x + 2y - √2 = 0 -/
def line (x y : ℝ) : Prop := x + 2*y - Real.sqrt 2 = 0

/-- The distance function from a point (x, y) to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (x + 2*y - Real.sqrt 2) / Real.sqrt 5

/-- Theorem stating that the maximum distance from the ellipse to the line is √10 -/
theorem max_distance_ellipse_to_line :
  ∃ (x y : ℝ), ellipse x y ∧
    (∀ (x' y' : ℝ), ellipse x' y' → distance_to_line x y ≥ distance_to_line x' y') ∧
    distance_to_line x y = Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l260_26086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_l260_26063

theorem hypotenuse_length (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi / 2) : 
  ∃ (AC AB BC : ℝ),
    AC > 0 ∧ AB > 0 ∧ BC > 0 ∧
    AB^2 + BC^2 = AC^2 ∧
    (Real.sin x)^2 * AC = AB^2 * (AC / 4) + BC^2 * (3 * AC / 4) - 3 * AC^3 / 16 ∧
    (Real.cos x)^2 * AC = AB^2 * (AC / 2) + BC^2 * (AC / 2) - AC^3 / 16 ∧
    AC = 2 * Real.sqrt 5 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_l260_26063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_common_tangent_length_l260_26023

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus point
def focus : ℝ × ℝ := (1, 0)

-- Define the line passing through the focus with slope k
def line (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the condition that k is positive
def k_positive (k : ℝ) : Prop := k > 0

-- Define the condition that A and B are on the parabola and the line
def points_on_curve_and_line (k : ℝ) (A B : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  parabola x₁ y₁ ∧ parabola x₂ y₂ ∧ line k x₁ y₁ ∧ line k x₂ y₂

-- Define the distance between A and B
noncomputable def distance_AB (A B : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Theorem statement
theorem external_common_tangent_length
  (k : ℝ) (A B : ℝ × ℝ)
  (h_k : k_positive k)
  (h_points : points_on_curve_and_line k A B)
  (h_distance : distance_AB A B = 5) :
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  2 * Real.sqrt ((x₁ + 1) * (x₂ + 1)) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_common_tangent_length_l260_26023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_intersects_asymptote_l260_26016

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := (3 * x^2 - 8 * x + 4) / (x^2 - 5 * x + 6)

-- Define the horizontal asymptote
def horizontal_asymptote : ℝ := 3

-- Theorem statement
theorem g_intersects_asymptote :
  ∃ x : ℝ, g x = horizontal_asymptote ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_intersects_asymptote_l260_26016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_implicit_function_derivatives_l260_26071

-- Define the implicit functions
noncomputable def f1 (x y : ℝ) : ℝ := x^2 + y^2 + 2*x - 6*y + 2
noncomputable def f2 (x y : ℝ) : ℝ := x^y - y^x

-- State the theorem
theorem implicit_function_derivatives :
  ∃ (y1 y2 : ℝ),
    f1 1 y1 = 0 ∧ 
    f2 1 y2 = 0 ∧
    ((∃ (dy1 : ℝ), dy1 = 1 ∨ dy1 = -1) ∧
     (∀ ε > 0, ∃ δ > 0, ∀ h, -δ < h ∧ h < δ →
       |f1 (1 + h) (y1 + dy1 * h) - f1 1 y1| < ε * |h|)) ∧
    (∃ (dy2 : ℝ), dy2 = 1 ∧
     (∀ ε > 0, ∃ δ > 0, ∀ h, -δ < h ∧ h < δ →
       |f2 (1 + h) (y2 + dy2 * h) - f2 1 y2| < ε * |h|)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_implicit_function_derivatives_l260_26071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_measure_l260_26073

structure IsoscelesTrapezoid where
  angles : Fin 4 → ℝ
  is_isosceles : True
  is_arithmetic_sequence : ∃ (a d : ℝ), ∀ i : Fin 4, angles i = a + i.val * d
  largest_angle : angles (Fin.last 3) = 150
  opposite_supplementary : ∃ (i j : Fin 4), i ≠ j ∧ angles i + angles j = 180

theorem smallest_angle_measure (T : IsoscelesTrapezoid) : T.angles 0 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_measure_l260_26073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_conditions_l260_26041

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ
  distance : ℝ
  time : ℝ

/-- Represents a race between two runners -/
structure Race where
  length : ℝ
  runnerA : Runner
  runnerB : Runner

/-- The conditions of our specific race -/
noncomputable def specificRace : Race where
  length := 200
  runnerA := { speed := 200 / 18, distance := 200, time := 18 }
  runnerB := { speed := 144 / 18, distance := 144, time := 25 }

theorem race_conditions (race : Race) : 
  race.length = 200 ∧ 
  race.runnerA.distance - race.runnerB.distance = 56 ∧
  race.runnerB.time - race.runnerA.time = 7 →
  race.runnerA.time = 18 := by
  sorry

#check race_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_conditions_l260_26041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_16_equals_x_minus_1_over_x_l260_26027

noncomputable def f (x : ℝ) : ℝ := (1 + x) / (2 - x)

noncomputable def f_n : ℕ → (ℝ → ℝ)
  | 0 => id
  | 1 => f
  | n + 1 => λ x => f (f_n n x)

theorem f_16_equals_x_minus_1_over_x :
  ∀ x : ℝ, x ≠ 0 → f_n 16 x = (x - 1) / x :=
by
  sorry

#check f_16_equals_x_minus_1_over_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_16_equals_x_minus_1_over_x_l260_26027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l260_26080

/-- Calculate the average speed of a car given distance and time -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- The problem statement -/
theorem car_average_speed :
  let distance : ℝ := 250
  let time : ℝ := 6
  let calculated_speed := average_speed distance time
  abs (calculated_speed - 41.67) < 0.01 := by
  -- Unfold definitions
  unfold average_speed
  -- Simplify the expression
  simp
  -- Prove the inequality
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l260_26080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_time_l260_26059

/-- The time (in seconds) it takes for a train to pass a bridge -/
noncomputable def time_to_pass_bridge (train_length bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating that the time for the train to pass the bridge is 50 seconds -/
theorem train_bridge_time :
  time_to_pass_bridge 360 140 36 = 50 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval time_to_pass_bridge 360 140 36

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_time_l260_26059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_ratio_theorem_l260_26095

/-- Sum of first n terms of a geometric sequence -/
noncomputable def geometric_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

/-- Theorem: If S3:S2 = 3:2 for a geometric sequence, then q = 1 or q = -1/2 -/
theorem geometric_ratio_theorem (a : ℝ) (q : ℝ) :
  (geometric_sum a q 3) / (geometric_sum a q 2) = 3/2 →
  q = 1 ∨ q = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_ratio_theorem_l260_26095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_condition_l260_26012

theorem necessary_not_sufficient_condition (m a : ℝ) (h : a ≠ 0) :
  (∀ m, |m| = a → m = a ∨ m = -a) ∧ (∃ m, (m = a ∨ m = -a) ∧ |m| ≠ a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_condition_l260_26012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l260_26066

/-- A cone with lateral surface area 16√2π and an axial section that is an isosceles right triangle has volume 64π/3 -/
theorem cone_volume (r l h : ℝ) : 
  r > 0 → l > 0 → h > 0 →
  l = Real.sqrt 2 * r →
  π * r * l = 16 * Real.sqrt 2 * π →
  h^2 + r^2 = l^2 →
  (1/3) * π * r^2 * h = 64*π/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l260_26066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brown_to_blonde_ratio_l260_26075

/-- The number of dolls with blonde hair -/
def blonde_dolls : ℕ := 4

/-- The number of dolls with brown hair -/
noncomputable def brown_dolls : ℕ := 16

/-- The number of dolls with black hair -/
noncomputable def black_dolls : ℕ := brown_dolls - 2

/-- The total number of dolls with black and brown hair -/
noncomputable def total_black_brown : ℕ := brown_dolls + black_dolls

/-- The statement that the total number of black and brown haired dolls
    is 26 more than the number of blonde-haired dolls -/
axiom total_black_brown_eq : total_black_brown = blonde_dolls + 26

/-- The theorem stating that the ratio of brown-haired dolls to blonde-haired dolls is 4:1 -/
theorem brown_to_blonde_ratio : 
  (brown_dolls : ℚ) / blonde_dolls = 4 / 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brown_to_blonde_ratio_l260_26075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_five_problem_solver_l260_26020

/-- Represents the number of problems solved by a student -/
def ProblemsSolved := ℕ

/-- Represents a group of students and their problem-solving statistics -/
structure StudentGroup where
  total_students : ℕ
  total_problems : ℕ
  one_problem_solvers : ℕ
  two_problem_solvers : ℕ
  three_problem_solvers : ℕ

/-- The theorem to be proved -/
theorem exists_five_problem_solver (group : StudentGroup) 
  (h1 : group.total_students = 10)
  (h2 : group.total_problems = 35)
  (h3 : group.one_problem_solvers > 0)
  (h4 : group.two_problem_solvers > 0)
  (h5 : group.three_problem_solvers > 0)
  (h6 : group.one_problem_solvers + group.two_problem_solvers + group.three_problem_solvers ≤ group.total_students) :
  ∃ (student : ℕ), student ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_five_problem_solver_l260_26020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_e_when_f_prime_is_two_l260_26096

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem
theorem x_equals_e_when_f_prime_is_two :
  ∀ x : ℝ, x > 0 → (deriv f x = 2) → x = Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_e_when_f_prime_is_two_l260_26096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_l260_26033

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  (([3, 4, 5, 9, 10, 15, 18, 30].filter (λ m ↦ n % m = 0)).length = 4)

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n → (n = 36 ∨ n = 45 ∨ n = 72) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_l260_26033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_max_area_l260_26004

/-- Represents the length of the garden -/
def length (l : ℝ) : ℝ := l

/-- Represents the width of the garden -/
def width (l : ℝ) : ℝ := 400 - 2 * l

/-- The total fencing available -/
def total_fencing : ℝ := 400

/-- The fencing constraint: two lengths and one width must equal the total fencing -/
def fencing_constraint (l : ℝ) : Prop := 2 * length l + width l = total_fencing

/-- The area of the garden as a function of length -/
def area (l : ℝ) : ℝ := length l * width l

/-- The maximum area of the garden -/
def max_area : ℝ := 20000

theorem garden_max_area :
  ∃ l : ℝ, fencing_constraint l ∧ 
  (∀ l' : ℝ, fencing_constraint l' → area l' ≤ max_area) ∧
  area l = max_area := by
  sorry

#check garden_max_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_max_area_l260_26004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_distance_to_line_l260_26051

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem conditions
def circleThroughPoint (c : Circle) : Prop :=
  let (x, y) := c.center
  (2 - x)^2 + (1 - y)^2 = c.radius^2

def circleTangentToAxes (c : Circle) : Prop :=
  let (x, y) := c.center
  x = c.radius ∧ y = c.radius

-- Define the distance formula from a point to a line
noncomputable def distanceToLine (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  let (x, y) := p
  |a * x + b * y + c| / Real.sqrt (a^2 + b^2)

-- The main theorem
theorem circle_distance_to_line :
  ∀ c : Circle,
    circleThroughPoint c →
    circleTangentToAxes c →
    distanceToLine c.center 2 (-1) (-3) = 2 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_distance_to_line_l260_26051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_three_l260_26001

/-- A linear function satisfying specific properties -/
noncomputable def f : ℝ → ℝ := sorry

/-- f is a linear function -/
axiom f_linear : ∃ (a b : ℝ), ∀ x, f x = a * x + b

/-- The first functional equation -/
axiom f_eq1 : ∀ x, f x = 3 * (Function.invFun f x) + 9

/-- The second functional equation -/
axiom f_eq2 : ∀ x, f (f x) = 4 * x - 2

/-- The value of f at 0 -/
axiom f_at_zero : f 0 = 3

/-- The main theorem to prove -/
theorem f_at_three : f 3 = 3 * Real.sqrt 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_three_l260_26001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_56_l260_26077

/-- Represents the boat journey problem -/
structure BoatJourney where
  river_speed : ℝ
  boat_speed : ℝ
  total_time : ℝ

/-- Calculates the upstream distance given a boat journey -/
noncomputable def upstream_distance (j : BoatJourney) : ℝ :=
  (j.total_time * (j.boat_speed - j.river_speed) * (j.boat_speed + j.river_speed)) / (3 * j.boat_speed)

/-- Theorem stating that for the given conditions, the upstream distance is 56 km -/
theorem upstream_distance_is_56 :
  let j : BoatJourney := {
    river_speed := 2,
    boat_speed := 6,
    total_time := 21
  }
  upstream_distance j = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_56_l260_26077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l260_26026

-- Define a geometric sequence
noncomputable def geometric_sequence (a₁ q : ℝ) : ℕ → ℝ := λ n => a₁ * q^(n-1)

-- Define the sum of the first n terms of a geometric sequence
noncomputable def S (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * (1 - q^n) / (1 - q)

-- Theorem statement
theorem geometric_sequence_ratio 
  (a₁ q : ℝ) (hq : q ≠ 1) (h : S a₁ q 4 / S a₁ q 2 = 3) : 
  S a₁ q 6 / S a₁ q 4 = 7/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l260_26026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_line_equations_l260_26088

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 36

-- Define the fixed point Q
def Q : ℝ × ℝ := (4, 0)

-- Define the moving point P on the circle
def P (x y : ℝ) : Prop := my_circle x y

-- Define M as the midpoint of PQ
def M (x y : ℝ) : Prop := ∃ (px py : ℝ), P px py ∧ x = (px + 4) / 2 ∧ y = py / 2

-- Define the trajectory of M
def trajectory_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 9

-- Define the line l passing through (0, -3)
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x - 3

-- Define the condition for the intersection points
def intersection_condition (x₁ x₂ : ℝ) : Prop := x₁ / x₂ + x₂ / x₁ = 21 / 2

-- Theorem statement
theorem trajectory_and_line_equations :
  ∀ (x y : ℝ), M x y → trajectory_M x y ∧
  ∃ (k₁ k₂ : ℝ), 
    (∀ (x y : ℝ), line_l k₁ x y ↔ x - y - 3 = 0) ∧
    (∀ (x y : ℝ), line_l k₂ x y ↔ 17 * x - 7 * y - 21 = 0) ∧
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      trajectory_M x₁ y₁ ∧ trajectory_M x₂ y₂ ∧
      line_l k₁ x₁ y₁ ∧ line_l k₁ x₂ y₂ ∧
      x₁ ≠ x₂ ∧ intersection_condition x₁ x₂) ∧
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      trajectory_M x₁ y₁ ∧ trajectory_M x₂ y₂ ∧
      line_l k₂ x₁ y₁ ∧ line_l k₂ x₂ y₂ ∧
      x₁ ≠ x₂ ∧ intersection_condition x₁ x₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_line_equations_l260_26088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ratio_on_unit_circle_l260_26028

theorem angle_ratio_on_unit_circle (α : ℝ) (m : ℝ) :
  (m < 0) →  -- α is in the second quadrant
  (m^2 + (3/5)^2 = 1) →  -- P(m, 3/5) is on the unit circle
  ((Real.sin α - 2*Real.cos α) / (Real.sin α + Real.cos α) = -11) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ratio_on_unit_circle_l260_26028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_3000_l260_26011

/-- Represents a geometric sequence -/
structure GeometricSequence where
  first_term : ℝ
  common_ratio : ℝ

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def sum_n_terms (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.first_term * (1 - seq.common_ratio^n) / (1 - seq.common_ratio)

theorem geometric_sequence_sum_3000 (seq : GeometricSequence) :
  sum_n_terms seq 1000 = 300 →
  sum_n_terms seq 2000 = 570 →
  sum_n_terms seq 3000 = 813 := by
  sorry

#check geometric_sequence_sum_3000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_3000_l260_26011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_expansion_coefficients_l260_26045

theorem polynomial_expansion_coefficients :
  ∃ (a₁ a₂ a₃ : ℝ), (fun x : ℝ ↦ (x + 1)^3 * (x + 2)^2) =
  (fun x : ℝ ↦ x^5 + a₁*x^4 + a₂*x^3 + a₃*x^2 + 16*x + 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_expansion_coefficients_l260_26045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_CBG_l260_26036

-- Define the circle and points
def Circle : Type := ℝ × ℝ × ℝ  -- center (x, y) and radius r
def Point : Type := ℝ × ℝ

-- Define the given circle and points
def circleO : Circle := (0, 0, 3)  -- Assuming center at origin for simplicity
noncomputable def A : Point := sorry
noncomputable def B : Point := sorry
noncomputable def C : Point := sorry
noncomputable def D : Point := sorry
noncomputable def E : Point := sorry
noncomputable def F : Point := sorry
noncomputable def G : Point := sorry

-- Define the properties of the configuration
def is_equilateral (A B C : Point) : Prop := sorry
def is_inscribed (A B C : Point) (circle : Circle) : Prop := sorry
def extend_line (A B D : Point) (length : ℝ) : Prop := sorry
def parallel_lines (l1 l2 : Point → Point → Prop) : Prop := sorry
def collinear (A F G : Point) : Prop := sorry
def on_circle (G : Point) (circle : Circle) : Prop := sorry
def intersection (F : Point) (l1 l2 : Point → Point → Prop) : Prop := sorry

-- Define the area calculation function
noncomputable def area (A B C : Point) : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_CBG :
  is_equilateral A B C →
  is_inscribed A B C circleO →
  extend_line A B D 15 →
  extend_line A C E 13 →
  parallel_lines (λ x y => collinear D x y) (λ x y => collinear A E y) →
  parallel_lines (λ x y => collinear E x y) (λ x y => collinear A D y) →
  intersection F (λ x y => collinear D x y) (λ x y => collinear E x y) →
  collinear A F G →
  on_circle G circleO →
  G ≠ A →
  area C B G = (195 * Real.sqrt 3) / 98 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_CBG_l260_26036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_printer_task_pages_l260_26024

/-- 
Proves that given the conditions of two printers working together and separately,
the total number of pages in the task is 360.
-/
theorem printer_task_pages : ℕ := by
  let printer_a_rate : ℚ := 6  -- pages per minute
  let printer_b_rate : ℚ := printer_a_rate + 3  -- pages per minute
  let combined_time : ℚ := 24  -- minutes
  let printer_a_time : ℚ := 60  -- minutes

  have h1 : (printer_a_rate + printer_b_rate) * combined_time = printer_a_rate * printer_a_time := by
    sorry

  have h2 : printer_a_rate * printer_a_time = 360 := by
    sorry

  exact 360


end NUMINAMATH_CALUDE_ERRORFEEDBACK_printer_task_pages_l260_26024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotonically_decreasing_l260_26006

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (2 * x - Real.pi / 4)

noncomputable def g (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (x / 2 - Real.pi / 12)

def monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem g_monotonically_decreasing :
  monotonically_decreasing g (7 * Real.pi / 6) (19 * Real.pi / 6) := by
  sorry

#check g_monotonically_decreasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotonically_decreasing_l260_26006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_tangent_lines_intersect_l260_26005

/-- A line in a 2D plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane --/
def Point := ℝ × ℝ

/-- Homothety center for two circles --/
noncomputable def homothetyCenter (c1 c2 : Circle) : Point := sorry

/-- Tangent point between a circle and a line --/
noncomputable def tangentPoint (c : Circle) (l : Line) : Point := sorry

/-- Tangent point between two circles --/
noncomputable def circleTangentPoint (c1 c2 : Circle) : Point := sorry

/-- Line passing through two points --/
noncomputable def lineThrough (p1 p2 : Point) : Line := sorry

/-- Check if a point lies on a line --/
def pointOnLine (p : Point) (l : Line) : Prop := sorry

/-- Main theorem --/
theorem all_tangent_lines_intersect (L : Line) (C : Circle) :
  ∃ H : Point, ∀ ω : Circle,
    (tangentPoint ω L ≠ circleTangentPoint ω C) →
    pointOnLine H (lineThrough (tangentPoint ω L) (circleTangentPoint ω C)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_tangent_lines_intersect_l260_26005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_x_l260_26031

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 5) + Real.sin (x / 7)

theorem smallest_max_x : 
  ∃ x : ℝ, x = 12690 ∧
  (∀ y : ℝ, 0 < y → y < x → f y ≤ f x) ∧ 
  (Real.sin (x / 5) = 1) ∧ 
  (Real.sin (x / 7) = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_x_l260_26031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_analysis_l260_26085

-- Define the expression
noncomputable def f (x : ℝ) : ℝ := (3*x^3 - 3*x + 6) / ((x+1)*(x-2)) - (11*x - 2) / ((x+1)*(x-2))

-- Define the simplified expression
noncomputable def g (x : ℝ) : ℝ := (3*x^3 - 14*x + 8) / ((x+1)*(x-2))

-- Theorem statement
theorem expression_analysis :
  (∀ x : ℝ, x ≠ -1 ∧ x ≠ 2 → f x = g x) ∧
  (∀ x : ℝ, x = -1 ∨ x = 2 → ¬ ∃ y : ℝ, f x = y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_analysis_l260_26085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_d_value_l260_26074

/-- An ellipse in the first quadrant tangent to both x-axis and y-axis -/
structure Ellipse where
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ
  tangent_x : Bool
  tangent_y : Bool
  first_quadrant : Bool

/-- The specific ellipse from the problem -/
def problem_ellipse (d : ℝ) : Ellipse where
  focus1 := (3, 8)
  focus2 := (d, 8)
  tangent_x := true
  tangent_y := true
  first_quadrant := true

theorem ellipse_focus_d_value :
  ∃ d : ℝ, problem_ellipse d = problem_ellipse d ∧ d = 247 / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_d_value_l260_26074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_problem_l260_26092

/-- The probability that the last passenger sits in their assigned seat -/
noncomputable def last_passenger_probability (n : ℕ) : ℝ :=
  if n ≥ 2 then 1/2 else 0

/-- The seating problem theorem -/
theorem seating_problem (n : ℕ) (h : n ≥ 2) :
  last_passenger_probability n = 1/2 := by
  unfold last_passenger_probability
  simp [h]

#check seating_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_problem_l260_26092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_tiling_probability_l260_26007

/-- Represents the number of valid colorings for a 2 × n grid -/
def a : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => 2 * a (n + 1) + 2 * a n

/-- The probability of a valid tiling for a 2 × n grid -/
def p (n : ℕ) : ℚ := (1 / 4) ^ n * (a n : ℚ)

/-- The generating function for the sequence a_n -/
noncomputable def A (x : ℝ) : ℝ := (2 * x + 2 * x^2) / (1 - 2 * x - 2 * x^2)

/-- The main theorem stating the probability of a valid tiling -/
theorem valid_tiling_probability : ∑' (n : ℕ), (1 / 2) ^ n * p n = 9 / 23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_tiling_probability_l260_26007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotated_figure_l260_26067

/-- The volume of the solid formed by rotating a specific plane figure about the x-axis -/
theorem volume_of_rotated_figure : 
  let f (x : ℝ) := Real.sqrt x
  let a : ℝ := 0
  let b : ℝ := 1
  let V := π * ∫ x in a..b, f x ^ 2
  V = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotated_figure_l260_26067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l260_26082

theorem solve_exponential_equation :
  ∃ y : ℝ, (64 : ℝ) ^ (3 * y) = (16 : ℝ) ^ (2 * y + 3) ↔ y = 6/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l260_26082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uncle_vanya_journey_l260_26081

/-- Represents the time taken for a journey with given distances -/
def journey_time (walk : ℝ) (cycle : ℝ) (drive : ℝ) (x y z : ℝ) : ℝ :=
  walk * x + cycle * y + drive * z

/-- The conditions given in the problem -/
axiom condition1 : journey_time 2 3 20 1 1 1 = 66
axiom condition2 : journey_time 5 8 30 1 1 1 = 144

/-- The theorem to be proved -/
theorem uncle_vanya_journey : journey_time 4 5 80 1 1 1 = 174 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_uncle_vanya_journey_l260_26081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_x_minus_two_l260_26079

-- Define the function as noncomputable
noncomputable def f (x : ℝ) := Real.sqrt (x - 2)

-- State the theorem
theorem domain_of_sqrt_x_minus_two :
  {x : ℝ | ∃ y : ℝ, f x = y} = {x : ℝ | x ≥ 2} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_x_minus_two_l260_26079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taller_cycle_height_l260_26057

/-- Represents the height and shadow length of a cycle -/
structure MyCycle where
  height : ℝ
  shadow : ℝ

/-- Proves that the height of the taller cycle is 2.5 feet -/
theorem taller_cycle_height 
  (c1 : MyCycle) -- First cycle
  (c2 : MyCycle) -- Second cycle
  (h1 : c1.shadow = 5) -- Shadow length of first cycle
  (h2 : c2.height = 2) -- Height of second cycle
  (h3 : c2.shadow = 4) -- Shadow length of second cycle
  (h4 : c1.height / c1.shadow = c2.height / c2.shadow) -- Similar triangles
  : c1.height = 2.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_taller_cycle_height_l260_26057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_difference_representation_l260_26034

theorem unique_difference_representation :
  ∃ (S : Set ℕ), 
    (∀ n : ℕ, n > 0 → ∃ (x y : ℕ), x ∈ S ∧ y ∈ S ∧ n = x - y) ∧ 
    (∀ n : ℕ, n > 0 → ∀ (x₁ y₁ x₂ y₂ : ℕ), 
      x₁ ∈ S → y₁ ∈ S → x₂ ∈ S → y₂ ∈ S →
      n = x₁ - y₁ → n = x₂ - y₂ → x₁ = x₂ ∧ y₁ = y₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_difference_representation_l260_26034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_closest_l260_26056

noncomputable section

/-- The line y = x/3 -/
def line (x : ℝ) : ℝ := x / 3

/-- The point we're finding the closest point to -/
def target : ℝ × ℝ := (9, 2)

/-- The claimed closest point on the line -/
def closest_point : ℝ × ℝ := (87/10, 29/10)

/-- Theorem stating that the closest_point is indeed the closest point on the line to the target -/
theorem closest_point_is_closest :
  ∀ x : ℝ, (x - 9)^2 + (line x - 2)^2 ≥ (87/10 - 9)^2 + (29/10 - 2)^2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_closest_l260_26056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_circumcenter_common_point_collinear_l260_26053

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A triangle in a plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The incenter of a triangle --/
noncomputable def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- The circumcenter of a triangle --/
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- Predicate to check if three points are collinear --/
def collinear (p q r : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if a circle is inside a triangle --/
def circle_inside_triangle (c : Circle) (t : Triangle) : Prop := sorry

/-- Predicate to check if a circle is tangent to two sides of a triangle --/
def circle_tangent_to_two_sides (c : Circle) (t : Triangle) : Prop := sorry

/-- Predicate to check if a point is on a circle --/
def point_on_circle (p : ℝ × ℝ) (c : Circle) : Prop := sorry

/-- Main theorem --/
theorem incenter_circumcenter_common_point_collinear 
  (t : Triangle) (c₁ c₂ c₃ : Circle) (K : ℝ × ℝ) :
  circle_inside_triangle c₁ t →
  circle_inside_triangle c₂ t →
  circle_inside_triangle c₃ t →
  circle_tangent_to_two_sides c₁ t →
  circle_tangent_to_two_sides c₂ t →
  circle_tangent_to_two_sides c₃ t →
  c₁.radius = c₂.radius →
  c₂.radius = c₃.radius →
  point_on_circle K c₁ →
  point_on_circle K c₂ →
  point_on_circle K c₃ →
  collinear (incenter t) (circumcenter t) K :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_circumcenter_common_point_collinear_l260_26053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_sum_l260_26015

theorem determinant_sum (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  Matrix.det !![2, 5, 12; 4, a, b; 4, c, a] = 0 →
  a + b + c = 68.4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_sum_l260_26015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_increase_l260_26049

theorem square_area_increase (s : ℝ) (h : s > 0) : 
  (((1.4 * s)^2 - s^2) / s^2) = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_increase_l260_26049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_percentage_calculation_l260_26054

/-- Calculates the percentage of a stock given the investment amount, stock price, and annual income. -/
noncomputable def stock_percentage (investment_amount : ℝ) (stock_price : ℝ) (annual_income : ℝ) : ℝ :=
  (annual_income / (investment_amount / stock_price)) / stock_price * 100

/-- Theorem stating that given the specific investment scenario, the stock percentage is approximately 14.71%. -/
theorem stock_percentage_calculation :
  let investment_amount : ℝ := 6800
  let stock_price : ℝ := 136
  let annual_income : ℝ := 1000
  |stock_percentage investment_amount stock_price annual_income - 14.71| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_percentage_calculation_l260_26054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_zero_l260_26078

theorem function_identity_zero (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^666 + y) = f (x^2023 + 2*y) + f (x^42)) →
  (∀ x : ℝ, f x = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_zero_l260_26078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_matrix_Q_l260_26090

def v : Fin 3 → ℚ
  | 0 => 3
  | 1 => -1
  | 2 => 2

def Q : Matrix (Fin 3) (Fin 3) ℚ :=
  !![9/14, -3/14, 6/14;
    -3/14, 1/14, -2/14;
    6/14, -2/14, 4/14]

theorem projection_matrix_Q (u : Fin 3 → ℚ) :
  Q.mulVec u = (Matrix.dotProduct u v / (Matrix.dotProduct v v)) • v := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_matrix_Q_l260_26090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l260_26099

theorem constant_term_expansion : 
  ∃ (f : ℝ → ℝ), (∀ x : ℝ, f x = (x^(1/2) - 2/x)^6) ∧ 
  (∃ c : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ y : ℝ, 0 < |y - 1| ∧ |y - 1| < δ → |f y - f 1 - c * (y - 1)| ≤ ε * |y - 1|) ∧
  c = 60 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l260_26099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_N_subset_M_l260_26009

-- Define the sets M and N
def M : Set ℝ := {x | x > 0 ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- State the theorem
theorem N_subset_M : N ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_N_subset_M_l260_26009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_ratio_approx_l260_26043

-- Define income variables
variable (J : ℝ) -- Juan's income

-- Define income relationships
def T (J : ℝ) : ℝ := 0.5 * J -- Tim's income
def M (J : ℝ) : ℝ := 1.6 * T J -- Mart's income
def K (J : ℝ) : ℝ := 1.2 * M J -- Katie's income
def X (J : ℝ) : ℝ := 0.7 * K J -- Jason's income

-- Define tax rate
def tax_rate : ℝ := 0.15

-- Define expense rates
def mart_expense_rate : ℝ := 0.25
def katie_expense_rate : ℝ := 0.35
def jason_expense_rate : ℝ := 0.2

-- Define after-tax incomes
def juan_after_tax (J : ℝ) : ℝ := J * (1 - tax_rate)
def jason_after_tax (J : ℝ) : ℝ := X J * (1 - tax_rate)

-- Define expenses
def jason_expenses (J : ℝ) : ℝ := K J * jason_expense_rate

-- Define final incomes
def juan_final (J : ℝ) : ℝ := juan_after_tax J
def jason_final (J : ℝ) : ℝ := jason_after_tax J - jason_expenses J

-- Theorem to prove
theorem income_ratio_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  ∀ J, J > 0 → |juan_final J / jason_final J - 2.2415| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_ratio_approx_l260_26043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l260_26050

-- Define the function f(x) = x - 1/x
noncomputable def f (x : ℝ) : ℝ := x - 1/x

-- Theorem stating the properties of the function f
theorem f_properties :
  -- 1. Domain is (-∞, 0) ∪ (0, +∞)
  (∀ x, f x ≠ 0 → x ≠ 0) ∧
  -- 2. Function is odd
  (∀ x, f (-x) = -f x) ∧
  -- 3. Function is increasing on (0, +∞)
  (∀ x y, 0 < x → 0 < y → x < y → f x < f y) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l260_26050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_inverse_l260_26021

/-- Given functions f, g, and h with specific properties, prove that a - b = 7/12 -/
theorem function_composition_inverse (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x + b
  let g : ℝ → ℝ := λ x => x^2 + 2*x + 1
  let h : ℝ → ℝ := f ∘ g
  (∀ x, (λ x => 2*x + 3) (h x) = x) →
  a - b = 7/12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_inverse_l260_26021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l260_26087

noncomputable def f (x : ℝ) := -Real.cos (2 * x)

theorem f_properties :
  (∀ x, f x = f (-x)) ∧                   -- f is even
  (∀ y, f (y + π) = f y) ∧                -- π is a period of f
  (∀ p, p > 0 ∧ (∀ y, f (y + p) = f y) → p ≥ π) -- π is the smallest positive period
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l260_26087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_l260_26035

theorem second_train_speed 
  (length1 : ℝ) (length2 : ℝ) (speed1 : ℝ) (clear_time : ℝ) :
  length1 = 100 →
  length2 = 220 →
  speed1 = 42 →
  clear_time = 15.99872010239181 →
  (let total_length := length1 + length2
   let total_length_km := total_length / 1000
   let clear_time_hours := clear_time / 3600
   let relative_speed := total_length_km / clear_time_hours
   let speed2 := relative_speed - speed1
   speed2) = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_l260_26035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l260_26064

def vector_a : Fin 2 → ℝ := ![1, 1]
def vector_b : Fin 2 → ℝ := ![2, 0]

theorem angle_between_vectors : 
  Real.arccos ((vector_a 0 * vector_b 0 + vector_a 1 * vector_b 1) / 
    (Real.sqrt ((vector_a 0)^2 + (vector_a 1)^2) * 
     Real.sqrt ((vector_b 0)^2 + (vector_b 1)^2))) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l260_26064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l260_26046

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi ∧
  2 * t.b - t.c = 2 * t.a * Real.cos t.C ∧
  t.a = Real.sqrt 3 ∧
  Real.sin t.B + Real.sin t.C = 6 * Real.sqrt 2 * Real.sin t.B * Real.sin t.C

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) : 
  t.A = Real.pi / 3 ∧ (1 / 2 * t.b * t.c * Real.sin t.A) = Real.sqrt 3 / 8 := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l260_26046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_similar_piles_l260_26010

theorem no_three_similar_piles (x : ℝ) (hx : x > 0) :
  ¬ ∃ (a b c : ℝ), (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (a + b + c = x) ∧
    (a ≤ Real.sqrt 2 * b ∧ b ≤ Real.sqrt 2 * a) ∧
    (a ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * a) ∧
    (b ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_similar_piles_l260_26010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_cycle_l260_26094

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Reflection of a point across a line segment -/
noncomputable def reflect (p : Point) (p1 p2 : Point) : Point :=
  sorry

/-- Transformation step for the quadrilateral -/
noncomputable def transform (q : Quadrilateral) : Quadrilateral :=
  let A' := reflect q.A q.B q.D
  let D' := reflect q.D A' q.C
  { A := A', B := q.B, C := q.C, D := D' }

/-- Theorem stating that the quadrilateral returns to its original state after 6 steps -/
theorem quadrilateral_cycle (q : Quadrilateral) 
  (h1 : distance q.A q.B = 1)
  (h2 : distance q.B q.C = 1)
  (h3 : distance q.C q.D = 1)
  (h4 : distance q.A q.D ≠ 1) :
  (transform^[6] q) = q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_cycle_l260_26094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_divisors_count_l260_26093

def has_n_divisors (n m : ℕ) : Prop := (Nat.divisors m).card = n

theorem square_divisors_count (m : ℕ) (h : has_n_divisors 4 m) : has_n_divisors 7 (m^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_divisors_count_l260_26093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l260_26008

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem f_properties :
  (∃ f' : ℝ → ℝ, DifferentiableAt ℝ f 1 ∧ deriv f 1 = 0) ∧
  (∃ M : ℝ, f (-1) = M ∧ ∀ x : ℝ, f x ≤ M) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l260_26008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_uncertain_photos_l260_26055

/-- Represents a time on the magical clock -/
structure MagicalTime where
  hour : Nat
  minute : Nat
  deriving Repr

/-- Represents a photograph of the magical clock -/
structure Photograph where
  id : Nat
  time : MagicalTime
  deriving Repr

/-- Checks if a given time is forbidden (0:00, 6:00, 12:00, 18:00, or 24:00) -/
def isForbiddenTime (t : MagicalTime) : Bool :=
  (t.hour = 0 ∨ t.hour = 6 ∨ t.hour = 12 ∨ t.hour = 18 ∨ t.hour = 24) ∧ t.minute = 0

/-- Checks if two photographs look identical -/
def looksIdentical (p1 p2 : Photograph) : Bool :=
  sorry

/-- The set of all possible times a photograph could represent -/
def possibleTimes (p : Photograph) : List MagicalTime :=
  sorry

/-- Checks if a photograph is uncertain -/
def isUncertain (p : Photograph) (photos : List Photograph) : Bool :=
  sorry

/-- Theorem: The minimum number of uncertain photographs is 3 -/
theorem min_uncertain_photos
  (photos : List Photograph)
  (h1 : photos.length = 100)
  (h2 : ∀ p, p ∈ photos → ¬ isForbiddenTime p.time)
  (h3 : ∀ p1 p2, p1 ∈ photos → p2 ∈ photos → p1 ≠ p2 → ¬ looksIdentical p1 p2)
  : (photos.filter (λ p => isUncertain p photos)).length ≥ 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_uncertain_photos_l260_26055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_40_l260_26098

/-- The angle of the minute hand at 3:40 p.m. -/
def minute_hand_angle : ℝ := 240

/-- The angle of the hour hand at 3:40 p.m. -/
def hour_hand_angle : ℝ := 110

/-- The maximum angle between two clock hands -/
def max_angle : ℝ := 180

/-- The smaller angle between the hour hand and the minute hand at 3:40 p.m. -/
noncomputable def clock_angle (m h : ℝ) : ℝ :=
  min (abs (m - h)) (360 - abs (m - h))

theorem clock_angle_at_3_40 :
  clock_angle minute_hand_angle hour_hand_angle = 130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_40_l260_26098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_to_cone_volume_l260_26048

noncomputable def sector_radius : ℝ := 3
noncomputable def sector_angle : ℝ := 120 * (Real.pi / 180)  -- Convert degrees to radians

noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

theorem sector_to_cone_volume :
  let arc_length := sector_angle * sector_radius
  let base_radius := arc_length / (2 * Real.pi)
  let height := Real.sqrt (sector_radius^2 - base_radius^2)
  cone_volume base_radius height = (2 * Real.sqrt 2 / 3) * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_to_cone_volume_l260_26048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaporation_rate_theorem_l260_26025

/-- The percentage of substance remaining after one year of evaporation -/
noncomputable def remaining_after_one_year (evaporation_rate : ℝ) : ℝ :=
  100 - evaporation_rate

/-- The percentage of substance remaining after two years of evaporation -/
noncomputable def remaining_after_two_years (evaporation_rate : ℝ) : ℝ :=
  (remaining_after_one_year evaporation_rate) * (1 - evaporation_rate / 100)

/-- Theorem stating that if 64% remains after two years, the yearly evaporation rate is 20% -/
theorem evaporation_rate_theorem :
  ∃ (rate : ℝ), remaining_after_two_years rate = 64 ∧ rate = 20 := by
  use 20
  apply And.intro
  · -- Proof that remaining_after_two_years 20 = 64
    sorry
  · -- Proof that rate = 20
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaporation_rate_theorem_l260_26025
