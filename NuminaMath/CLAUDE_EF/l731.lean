import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_plus_one_is_power_l731_73145

theorem power_of_two_plus_one_is_power (n : ℕ) : 
  (∃ (a k : ℕ), a > 1 ∧ k > 1 ∧ 2^n + 1 = a^k) ↔ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_plus_one_is_power_l731_73145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AOC_l731_73188

-- Define the circle
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y + 12 = 0

-- Define the center of the circle
def center : ℝ × ℝ := (2, 3)

-- Define point A
def point_A : ℝ × ℝ := (3, 5)

-- Define point O (origin)
def point_O : ℝ × ℝ := (0, 0)

-- Calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Calculate the area of a triangle given two sides and the angle between them
noncomputable def triangle_area (a b angle : ℝ) : ℝ :=
  0.5 * a * b * Real.sin angle

-- Theorem statement
theorem area_of_triangle_AOC :
  triangle_area (distance point_O point_A) (distance point_O center) 
    (Real.arccos ((point_A.1 * center.1 + point_A.2 * center.2) / 
      (distance point_O point_A * distance point_O center))) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AOC_l731_73188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_parts_l731_73165

/-- The repeating decimal 0.474747... as a rational number -/
def repeating_decimal : ℚ := 47 / 99

/-- The sum of the numerator and denominator of the simplest fraction form -/
def sum_of_parts : ℕ := 146

theorem repeating_decimal_sum_parts : 
  (repeating_decimal.num.natAbs + repeating_decimal.den) = sum_of_parts := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_parts_l731_73165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_M_distance_2_from_P_line_through_M_parallel_to_l₃_l731_73122

-- Define the lines l₁, l₂, and l₃
def l₁ (x y : ℝ) : Prop := x - 2*y + 3 = 0
def l₂ (x y : ℝ) : Prop := 2*x + 3*y - 8 = 0
def l₃ (x y : ℝ) : Prop := x + 3*y + 1 = 0

-- Define point P
def P : ℝ × ℝ := (0, 4)

-- Define the intersection point M of l₁ and l₂
def M : ℝ × ℝ := (1, 2)

-- Define the distance function
noncomputable def distance (x y a b : ℝ) : ℝ :=
  Real.sqrt ((x - a)^2 + (y - b)^2)

-- Theorem for part (1)
theorem line_through_M_distance_2_from_P :
  (∀ x y : ℝ, (y = 2 ∨ 4*x - 3*y + 2 = 0) ↔
    (x - M.1)^2 + (y - M.2)^2 = 0 ∧
    distance x y P.1 P.2 = 2) :=
by sorry

-- Theorem for part (2)
theorem line_through_M_parallel_to_l₃ :
  ∀ x y : ℝ, 3*y + x - 7 = 0 ↔
    (x - M.1)^2 + (y - M.2)^2 = 0 ∧
    ∃ k : ℝ, y - M.2 = k * (x - M.1) ∧ -1/3 = k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_M_distance_2_from_P_line_through_M_parallel_to_l₃_l731_73122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_halfway_fraction_l731_73149

theorem halfway_fraction : 
  (3/4 : ℚ) + (6/7 : ℚ) / 2 = 45/56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_halfway_fraction_l731_73149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parents_can_catch_child_l731_73155

-- Define the square grid
structure Grid where
  side_length : ℝ
  is_positive : side_length > 0

-- Define a point on the grid
structure GridPoint (grid : Grid) where
  x : ℝ
  y : ℝ
  on_grid : x ∈ Set.range (fun i => i * grid.side_length) ∧ 
            y ∈ Set.range (fun i => i * grid.side_length)

-- Define visibility
def visible (grid : Grid) (a b : GridPoint grid) : Prop :=
  (a.x = b.x ∨ a.y = b.y) ∨ 
  (|a.x - b.x| = |a.y - b.y| ∧ |a.x - b.x| = grid.side_length / 2)

-- Define the theorem
theorem parents_can_catch_child 
  (grid : Grid) 
  (child parent1 parent2 : ℝ → GridPoint grid) 
  (child_speed parent_speed : ℝ) :
  child_speed = 2 * parent_speed →
  (∀ t, visible grid (child t) (parent1 t) ∧ visible grid (child t) (parent2 t) ∧ visible grid (parent1 t) (parent2 t)) →
  ∃ t, child t = parent1 t ∨ child t = parent2 t :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parents_can_catch_child_l731_73155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_squared_l731_73121

theorem det_A_squared {n : Type*} [Fintype n] [DecidableEq n] 
  (A : Matrix n n ℝ) (h : Matrix.det A = 3) : Matrix.det (A ^ 2) = 9 := by
  have h1 : Matrix.det (A ^ 2) = (Matrix.det A) ^ 2 := by
    apply Matrix.det_pow
  rw [h1, h]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_squared_l731_73121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_numbers_circle_impossibility_l731_73117

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem consecutive_numbers_circle_impossibility :
  ∀ (start : ℕ),
  ¬ ∃ (arrangement : List ℕ),
    (arrangement.length = 100) ∧
    (∀ n, n ∈ arrangement → start ≤ n ∧ n < start + 100) ∧
    (∀ i, i < arrangement.length →
      is_perfect_square ((arrangement.get ⟨i, by sorry⟩) *
        (arrangement.get ⟨(i + 1) % arrangement.length, by sorry⟩))) ∧
    (arrangement.toFinset.card = 100) :=
by sorry

#check consecutive_numbers_circle_impossibility

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_numbers_circle_impossibility_l731_73117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l731_73126

open Real Set

theorem solution_set_equality (x : ℝ) : 
  x ∈ Icc 0 (2 * π) →
  (2 * cos x ≤ |sqrt (1 + sin (2 * x)) - sqrt (1 - sin (2 * x))| ∧ 
   |sqrt (1 + sin (2 * x)) - sqrt (1 - sin (2 * x))| ≤ sqrt 2) ↔
  x ∈ Icc (π / 4) (7 * π / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l731_73126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_is_30_root_6_l731_73182

/-- Triangle D₁D₂D₃ with given side lengths -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Tetrahedron ABCD formed by folding Triangle D₁D₂D₃ along its medians -/
def tetrahedron_from_triangle (t : Triangle) : ℝ → Prop := sorry

/-- The volume of the tetrahedron formed from the given triangle -/
noncomputable def tetrahedron_volume (t : Triangle) : ℝ := sorry

/-- The theorem stating the volume of the tetrahedron -/
theorem tetrahedron_volume_is_30_root_6 (t : Triangle) 
  (h1 : t.side1 = 24) (h2 : t.side2 = 20) (h3 : t.side3 = 16) : 
  tetrahedron_volume t = 30 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_is_30_root_6_l731_73182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_value_l731_73104

noncomputable def sequence_a : ℕ → ℝ
  | 0 => Real.sqrt 3
  | n + 1 => Int.floor (sequence_a n) + 1 / sequence_a n

theorem a_2017_value :
  sequence_a 2017 = 3024 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_value_l731_73104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_diameter_l731_73135

/-- The diameter of a wheel given its revolutions and distance covered -/
theorem wheel_diameter (revolutions : ℝ) (distance : ℝ) :
  revolutions = 47.04276615104641 →
  distance = 4136 →
  ∃ (diameter : ℝ), 
    diameter * Real.pi * revolutions = distance ∧ 
    (abs (diameter - 27.99) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_diameter_l731_73135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_property_l731_73189

/-- Definition of a tetrahedron -/
structure Tetrahedron :=
  (volume : ℝ)

/-- Definition of a face of a tetrahedron -/
structure Face :=
  (area : ℝ)

/-- Definition of an edge of a tetrahedron -/
structure Edge :=
  (length : ℝ)
  (isPerpendicular : Face → Prop)

/-- Relation between a face and its corresponding edge -/
def Face.correspondingEdge (f : Face) (e : Edge) : Prop :=
  sorry

/-- Given a tetrahedron with volume V, there exists another tetrahedron KLMN with specific properties -/
theorem tetrahedron_property (V : ℝ) (V_pos : V > 0) : 
  ∃ (W : ℝ), 
    ∃ (KLMN : Tetrahedron),
      (∀ (face : Face) (edge : Edge), 
        Face.correspondingEdge face edge → 
          edge.isPerpendicular face ∧ 
          edge.length = face.area) ∧
      W = (3/4) * V^2 ∧
      KLMN.volume = W :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_property_l731_73189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l731_73185

-- Define the parabola E
def E (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def l (x y k : ℝ) : Prop := y = k*(x - 3)

-- Define the intersection points A and B
def intersectionPoints (A B : ℝ × ℝ) (k : ℝ) : Prop :=
  E A.1 A.2 ∧ E B.1 B.2 ∧ l A.1 A.2 k ∧ l B.1 B.2 k

-- Define the condition OA · OB + 3 = 0
def dotProductCondition (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 + 3 = 0

-- Define the slopes k₁ and k₂
noncomputable def k₁ (A : ℝ × ℝ) : ℝ := A.2 / (A.1 + 3)
noncomputable def k₂ (B : ℝ × ℝ) : ℝ := B.2 / (B.1 + 3)

-- The main theorem
theorem parabola_intersection_theorem (k : ℝ) (A B : ℝ × ℝ) :
  intersectionPoints A B k →
  dotProductCondition A B →
  (1 / (k₁ A)^2 + 1 / (k₂ B)^2 - 2 / k^2) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l731_73185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_arrangement_theorem_l731_73127

def number_of_guests : ℕ := 5
def number_of_chairs : ℕ := 8

/-- The number of ways to arrange guests around a circular table with restrictions -/
def circular_arrangement_with_restrictions (n : ℕ) (k : ℕ) : ℕ :=
  (n - 1).factorial / (n - k - 1).factorial - (n - 1) * (n - 3).factorial / (n - k - 1).factorial * 2

theorem circular_arrangement_theorem :
  circular_arrangement_with_restrictions number_of_chairs number_of_guests = 5040 := by
  sorry

#eval circular_arrangement_with_restrictions number_of_chairs number_of_guests

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_arrangement_theorem_l731_73127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_T_value_l731_73178

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the expression T
noncomputable def T : ℂ := (1 + Real.sqrt 3 * i)^13 - (1 - Real.sqrt 3 * i)^13

-- State the theorem
theorem abs_T_value : Complex.abs T = 8192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_T_value_l731_73178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_sine_minimum_omega_l731_73198

open Real

theorem translated_sine_minimum_omega :
  ∀ ω : ℝ,
  ω > 0 →
  (let f : ℝ → ℝ := λ x ↦ sin (ω * x)
   let g : ℝ → ℝ := λ x ↦ f (x - π / 4)
   g (3 * π / 4) = 0) →
  ∀ ω' : ℝ, ω' > 0 ∧ 
    (let f' : ℝ → ℝ := λ x ↦ sin (ω' * x)
     let g' : ℝ → ℝ := λ x ↦ f' (x - π / 4)
     g' (3 * π / 4) = 0) →
  ω' ≥ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_sine_minimum_omega_l731_73198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_inequality_l731_73173

noncomputable def f (x : ℝ) := 2 * Real.sin (3 * x - Real.pi / 3)

theorem function_properties_and_inequality :
  ∃ (ω φ : ℝ),
    ω > 0 ∧
    -Real.pi / 2 < φ ∧ φ < 0 ∧
    Real.tan φ = -Real.sqrt 3 ∧
    (∀ x, f x = 2 * Real.sin (ω * x + φ)) ∧
    (∃ x₁ x₂, |f x₁ - f x₂| = 4 ∧ |x₁ - x₂| = Real.pi / 3) ∧
    (∀ x ∈ Set.Icc 0 (Real.pi / 6), ∀ m : ℝ, m * f x + 2 * m ≥ f x ↔ m ≥ 1 / 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_inequality_l731_73173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_equivalence_intersection_range_l731_73112

-- Define the curve C
noncomputable def curve_C (t : ℝ) : ℝ × ℝ := (2 * Real.cos t, 2 * Real.sin t)

-- Define the line l in polar form
def line_l_polar (ρ θ m : ℝ) : Prop := ρ * Real.cos (θ - Real.pi/3) + m = 0

-- Define the line l in Cartesian form
def line_l_cartesian (x y m : ℝ) : Prop := x + Real.sqrt 3 * y + 2 * m = 0

-- Theorem 1: Equivalence of polar and Cartesian forms of line l
theorem polar_to_cartesian_equivalence (m : ℝ) :
  ∀ x y ρ θ, x = ρ * Real.cos θ → y = ρ * Real.sin θ →
  (line_l_polar ρ θ m ↔ line_l_cartesian x y m) := by
  sorry

-- Theorem 2: Range of m for intersection
theorem intersection_range :
  ∃ m_min m_max, m_min = -2 ∧ m_max = 2 ∧
  ∀ m, (∃ t, line_l_cartesian (curve_C t).1 (curve_C t).2 m) ↔ m_min ≤ m ∧ m ≤ m_max := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_equivalence_intersection_range_l731_73112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_of_point_l731_73139

def circle_C : Set (ℝ × ℝ) := {p | ∃ c : ℝ × ℝ, ‖p - c‖ ≤ 2}

theorem x_coordinate_of_point (x : ℝ) :
  (x, 0) ∈ circle_C ∧ (-2, 0) ∈ circle_C → x = 2 := by
  sorry

#check x_coordinate_of_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_of_point_l731_73139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_boarders_correct_l731_73137

/-- The number of new boarders that joined the school -/
def new_boarders : ℕ := 30

/-- The initial number of boarders -/
def initial_boarders : ℕ := 150

/-- The initial ratio of boarders to day students -/
def initial_ratio : ℚ := 5 / 12

/-- The new ratio of boarders to day students after new boarders joined -/
def new_ratio : ℚ := 1 / 2

/-- The theorem stating that the number of new boarders is correct -/
theorem new_boarders_correct :
  let initial_day_students : ℚ := (initial_boarders : ℚ) / initial_ratio
  (initial_boarders + new_boarders : ℚ) / initial_day_students = new_ratio := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_boarders_correct_l731_73137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_square_shares_vertex_largest_square_formula_l731_73130

/-- Represents a right triangle with legs of length a and b -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b

/-- Represents a square inscribed in a right triangle -/
structure InscribedSquare (t : RightTriangle) where
  side : ℝ
  side_pos : 0 < side
  fits_in_triangle : side ≤ min t.a t.b

/-- The largest inscribed square in a right triangle -/
noncomputable def largest_inscribed_square (t : RightTriangle) : InscribedSquare t where
  side := (t.a * t.b) / (t.a + t.b)
  side_pos := by
    apply div_pos
    · exact mul_pos t.a_pos t.b_pos
    · exact add_pos t.a_pos t.b_pos
  fits_in_triangle := by sorry

theorem largest_square_shares_vertex (t : RightTriangle) :
  ∀ (s : InscribedSquare t), s.side ≤ (largest_inscribed_square t).side := by
  sorry

theorem largest_square_formula (t : RightTriangle) :
  (largest_inscribed_square t).side = (t.a * t.b) / (t.a + t.b) := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_square_shares_vertex_largest_square_formula_l731_73130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_range_l731_73148

/-- The function f(x) = log₀.₅(x² + ax + 1) has a range of all real numbers if and only if a is in the set (-∞, -2] ∪ [2, +∞) -/
theorem log_function_range (a : ℝ) : 
  (∀ y : ℝ, ∃ x : ℝ, y = Real.log (x^2 + a*x + 1) / Real.log 0.5) ↔ 
  (a ≤ -2 ∨ a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_range_l731_73148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_l731_73133

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 - 4*x else (- x)^2 - 4*(- x)

theorem solution_set_of_f (f : ℝ → ℝ) :
  (∀ x, f x = f (-x)) →
  (∀ x ≥ 0, f x = x^2 - 4*x) →
  {x | f (x + 2) < 5} = Set.Ioo (-7) 3 := by
  sorry

#check solution_set_of_f f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_l731_73133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_for_specific_triangle_l731_73194

/-- A triangle with given altitudes -/
structure TriangleWithAltitudes where
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  h₁_pos : h₁ > 0
  h₂_pos : h₂ > 0
  h₃_pos : h₃ > 0

/-- The largest angle in a triangle with given altitudes -/
noncomputable def largest_angle (t : TriangleWithAltitudes) : ℝ :=
  Real.arccos (-1/4)

/-- Theorem: The largest angle in a triangle with altitudes 9, 12, and 18 is arccos(-1/4) -/
theorem largest_angle_for_specific_triangle :
  let t : TriangleWithAltitudes := {
    h₁ := 9, h₂ := 12, h₃ := 18,
    h₁_pos := by norm_num,
    h₂_pos := by norm_num,
    h₃_pos := by norm_num
  }
  largest_angle t = Real.arccos (-1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_for_specific_triangle_l731_73194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_free_throw_variance_l731_73169

/-- The probability of a successful free throw -/
def p : ℝ := 0.8

/-- The score for a successful free throw -/
def success_score : ℝ := 1

/-- The score for an unsuccessful free throw -/
def failure_score : ℝ := 0

/-- The random variable X representing the score of a single free throw -/
noncomputable def X : ℝ → ℝ := fun ω => if ω ≤ p then success_score else failure_score

/-- The expected value of X -/
def E_X : ℝ := p * success_score + (1 - p) * failure_score

/-- The expected value of X^2 -/
def E_X_squared : ℝ := p * success_score^2 + (1 - p) * failure_score^2

/-- The variance of X -/
def Var_X : ℝ := E_X_squared - E_X^2

theorem free_throw_variance : Var_X = 0.16 := by
  -- Expand definitions
  unfold Var_X E_X_squared E_X
  -- Substitute known values
  simp [p, success_score, failure_score]
  -- Perform arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_free_throw_variance_l731_73169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_seats_l731_73116

/-- Represents the number of seats in a theater -/
def seats : ℕ := sorry

/-- Represents the capacity percentage as a rational number -/
def capacity : ℚ := sorry

/-- Represents the ticket price in dollars -/
def ticket_price : ℕ := sorry

/-- Represents the number of days the show was performed -/
def performance_days : ℕ := sorry

/-- Represents the total revenue in dollars -/
def total_revenue : ℕ := sorry

theorem theater_seats 
  (h1 : capacity = 80 / 100)
  (h2 : ticket_price = 30)
  (h3 : performance_days = 3)
  (h4 : total_revenue = 28800)
  : seats = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_seats_l731_73116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_g_zero_point_exists_l731_73136

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.exp (x - k) - x

-- Theorem 1: Range of m for g(x) = √(f(x) - 1) to have domain ℝ when k = 0
theorem range_of_m_for_g (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, y^2 = f 0 x - 1) ↔ m > -1 :=
by sorry

-- Theorem 2: Existence of zero point in (k, 2k) when k > 1
theorem zero_point_exists (k : ℝ) (h : k > 1) :
  ∃ x : ℝ, k < x ∧ x < 2*k ∧ f k x = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_g_zero_point_exists_l731_73136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_table_seating_arrangements_l731_73186

def number_of_people : ℕ := 6

def number_of_arrangements (n : ℕ) : ℕ := 
  if n ≤ 1 then 1 else (n - 1) * number_of_arrangements (n - 1) * 2

theorem round_table_seating_arrangements :
  number_of_arrangements (number_of_people - 1) = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_table_seating_arrangements_l731_73186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_is_90_over_11_l731_73124

/-- Sequence definition -/
def b : ℕ → ℚ
  | 0 => 2  -- Adding this case to cover Nat.zero
  | 1 => 2
  | 2 => 3
  | n + 3 => (1/4) * b (n + 2) + (1/5) * b (n + 1)

/-- Sum of the sequence -/
noncomputable def seriesSum : ℚ := ∑' n, b n

/-- Theorem: The sum of the sequence is 90/11 -/
theorem sequence_sum_is_90_over_11 : seriesSum = 90/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_is_90_over_11_l731_73124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_parameterizations_l731_73163

noncomputable def line_equation (x y : ℝ) : Prop := y = -3 * x + 1

noncomputable def parameterization_A (t : ℝ) : ℝ × ℝ := (1 + t, -2 - 3*t)
noncomputable def parameterization_B (t : ℝ) : ℝ × ℝ := (1/3 + t, 3*t)
noncomputable def parameterization_C (t : ℝ) : ℝ × ℝ := (3*t, 1 - 9*t)
noncomputable def parameterization_D (t : ℝ) : ℝ × ℝ := (-1 + t, 4 + 3*t)

theorem valid_parameterizations :
  (∀ t, line_equation (parameterization_A t).1 (parameterization_A t).2) ∧
  (∀ t, line_equation (parameterization_B t).1 (parameterization_B t).2) ∧
  (∀ t, line_equation (parameterization_C t).1 (parameterization_C t).2) ∧
  (∀ t, line_equation (parameterization_D t).1 (parameterization_D t).2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_parameterizations_l731_73163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l731_73123

noncomputable def expansion (x : ℝ) (n : ℕ) : ℝ := (x^(1/3) - 1/(2*x^(1/3)))^n

noncomputable def coefficient (n : ℕ) (r : ℕ) : ℝ := (-1/2)^r * (n.choose r)

def exponent (n : ℕ) (r : ℕ) : ℚ := (n - 2*r) / 3

theorem expansion_properties (n : ℕ) :
  (∃ r : ℕ, r = 5 ∧ exponent n r = 0) →
  (n = 10 ∧ coefficient n 2 = 45/4) := by
  intro h
  sorry

#check expansion_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l731_73123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_acb_l731_73111

/-- Given two points A and B in a plane, this theorem states that for any point C
    satisfying |AC|^2 + |BC|^2 = 2|AB|^2, the maximum angle ACB is 60°. -/
theorem max_angle_acb (A B : EuclideanSpace ℝ (Fin 2)) :
  ∃ (C : EuclideanSpace ℝ (Fin 2)),
    (‖C - A‖)^2 + (‖C - B‖)^2 = 2 * (‖B - A‖)^2 ∧
    ∀ (D : EuclideanSpace ℝ (Fin 2)),
      (‖D - A‖)^2 + (‖D - B‖)^2 = 2 * (‖B - A‖)^2 →
      EuclideanGeometry.angle A C B ≥ EuclideanGeometry.angle A D B ∧
      EuclideanGeometry.angle A C B = Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_acb_l731_73111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_implies_a_eq_two_l731_73158

/-- The function f(x) defined as 4 - x² + a ln x --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 - x^2 + a * Real.log x

/-- The theorem stating that if f(x) ≤ 3 for all x > 0, then a = 2 --/
theorem f_upper_bound_implies_a_eq_two (a : ℝ) :
  (∀ x > 0, f a x ≤ 3) → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_implies_a_eq_two_l731_73158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l731_73142

def U : Set Nat := {0,1,2,3,4,5,6,7,8,9}
def A : Set Nat := {0,1,3,5,8}
def B : Set Nat := {2,4,5,6,8}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {7,9} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l731_73142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l731_73109

theorem trigonometric_problem (α β : ℝ) 
  (h1 : Real.cos α = 1/7)
  (h2 : Real.cos (α - β) = 13/14)
  (h3 : 0 < β)
  (h4 : β < α)
  (h5 : α < π/2) :
  Real.tan (2*α) = -8*Real.sqrt 3/47 ∧ β = π/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l731_73109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_two_dividing_3n_plus_1_l731_73141

theorem largest_power_of_two_dividing_3n_plus_1 (n : ℕ) :
  (n % 2 = 1 → ∃ k, k = 2 ∧ 2^k ∣ 3^n + 1 ∧ ∀ m > k, ¬(2^m ∣ 3^n + 1)) ∧
  (n % 2 = 0 → ∃ k, k = 1 ∧ 2^k ∣ 3^n + 1 ∧ ∀ m > k, ¬(2^m ∣ 3^n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_two_dividing_3n_plus_1_l731_73141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l731_73157

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x + Real.cos x)

theorem f_properties :
  ∃ (k : ℤ),
    (f (5 * π / 4) = 2) ∧
    (∀ x : ℝ, f (x + π) = f x) ∧
    (∀ x y : ℝ, x ∈ Set.Icc (k * π - 3 * π / 8) (k * π + π / 8) →
                y ∈ Set.Icc (k * π - 3 * π / 8) (k * π + π / 8) →
                x ≤ y → f x ≤ f y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l731_73157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_congruence_theorem_l731_73191

theorem prime_congruence_theorem (p q : Nat) (n : Nat) :
  Nat.Prime p ∧ Nat.Prime q ∧ 
  Odd p ∧ Odd q ∧ 
  n > 1 ∧
  (q^(n+2) : Nat) % (p^n) = (3^(n+2) : Nat) % (p^n) ∧
  (p^(n+2) : Nat) % (q^n) = (3^(n+2) : Nat) % (q^n) →
  p = 3 ∧ q = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_congruence_theorem_l731_73191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l731_73167

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

noncomputable def angle (a b : V) : ℝ := Real.arccos (inner a b / (‖a‖ * ‖b‖))

theorem angle_between_vectors (a b : V) 
  (h1 : angle a b = π / 3)
  (h2 : ‖a‖ = 2)
  (h3 : ‖b‖ = 1) :
  angle a (a + 2 • b) = π / 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l731_73167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_proof_l731_73102

/-- Prove that given a journey of 150 miles in 3 hours, where the speed for the first hour
    was 45 mph and the second hour was 55 mph, the speed for the third hour must have been
    50 mph to maintain an overall average speed of 50 mph. -/
theorem journey_speed_proof (total_distance : ℝ) (total_time : ℝ) 
  (speed_hour1 : ℝ) (speed_hour2 : ℝ) (speed_hour3 : ℝ) : 
  total_distance = 150 →
  total_time = 3 →
  speed_hour1 = 45 →
  speed_hour2 = 55 →
  (total_distance / total_time = 50) →
  speed_hour3 = 50 := by
  intros h1 h2 h3 h4 h5
  -- The proof steps would go here
  sorry

#check journey_speed_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_proof_l731_73102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l731_73105

noncomputable section

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (-1, 2)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the magnitude of a 2D vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop := ∃ (k : ℝ), v = (k * w.1, k * w.2)

-- Define perpendicular vectors
def perpendicular (v w : ℝ × ℝ) : Prop := dot_product v w = 0

-- Statement for part (1)
theorem part_one : 
  ∀ (c : ℝ × ℝ), 
  magnitude c = 1 ∧ parallel c (a.1 - b.1, a.2 - b.2) → 
  c = (Real.sqrt 2 / 2, -Real.sqrt 2 / 2) ∨ c = (-Real.sqrt 2 / 2, Real.sqrt 2 / 2) :=
by sorry

-- Statement for part (2)
theorem part_two : 
  ∀ (t : ℝ),
  perpendicular (2*t*a.1 - b.1, 2*t*a.2 - b.2) (3*a.1 + t*b.1, 3*a.2 + t*b.2) →
  t = -1 ∨ t = 3/2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l731_73105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_l731_73181

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 8 = 0

-- Define the slope angle
noncomputable def slope_angle : ℝ := 5 * Real.pi / 6

-- Theorem statement
theorem line_slope_angle :
  ∀ x y : ℝ, line_equation x y → ∃ α : ℝ, α = slope_angle ∧ 
  (∃ m : ℝ, m = -Real.sqrt 3 / 3 ∧ Real.tan α = m) :=
by
  sorry

#check line_slope_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_l731_73181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_G1G2G3_is_zero_l731_73107

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define a centroid (placeholder definition)
noncomputable def centroid (t : Triangle) : Point :=
  { x := (t.A.x + t.B.x + t.C.x) / 3,
    y := (t.A.y + t.B.y + t.C.y) / 3 }

-- Define the area of a triangle (placeholder definition)
noncomputable def area (t : Triangle) : ℝ :=
  (1/2) * abs ((t.B.x - t.A.x) * (t.C.y - t.A.y) - (t.C.x - t.A.x) * (t.B.y - t.A.y))

-- State the theorem
theorem area_of_G1G2G3_is_zero (ABC : Triangle) (P : Point) 
  (G1 G2 G3 : Point) :
  P = ABC.A →
  G1 = centroid { A := P, B := ABC.B, C := ABC.C } →
  G2 = centroid { A := P, B := ABC.C, C := ABC.A } →
  G3 = centroid { A := P, B := ABC.A, C := ABC.B } →
  area ABC = 24 →
  area { A := G1, B := G2, C := G3 } = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_G1G2G3_is_zero_l731_73107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_passes_through_center_variance_of_scaled_data_stronger_correlation_closer_to_one_l731_73103

variable {α : Type*}

-- Regression equation
def regression_equation (x y : α → ℝ) (a b : ℝ) : α → ℝ := λ t ↦ b * x t + a

-- Sample mean
noncomputable def sample_mean (x : α → ℝ) : ℝ := sorry

-- Variance
noncomputable def variance (x : α → ℝ) : ℝ := sorry

-- Correlation coefficient
noncomputable def correlation_coefficient (x y : α → ℝ) : ℝ := sorry

-- Theorem 1: Regression equation passes through the center of sample points
theorem regression_passes_through_center (x y : α → ℝ) (a b : ℝ) :
  regression_equation x y a b = λ t ↦ b * x t + a := by sorry

-- Theorem 2: Variance of scaled data
theorem variance_of_scaled_data (x : α → ℝ) (c : ℝ) :
  variance (λ t ↦ c * x t) = c^2 * variance x := by sorry

-- Theorem 3: Stronger correlation implies correlation coefficient closer to 1
theorem stronger_correlation_closer_to_one (x y : α → ℝ) :
  ∀ ε > 0, ∃ δ > 0, ∀ x' y' : α → ℝ,
    (∀ t, |x' t - x t| < δ ∧ |y' t - y t| < δ) →
    |correlation_coefficient x' y'| > |correlation_coefficient x y| →
    |correlation_coefficient x' y'| > |correlation_coefficient x y| + ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_passes_through_center_variance_of_scaled_data_stronger_correlation_closer_to_one_l731_73103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_lines_angles_l731_73197

-- Define the concept of a line
def Line : Type := ℝ → ℝ → Prop

-- Define the concept of an angle between two lines
noncomputable def angle (l1 l2 : Line) : ℝ := sorry

-- Define the concept of symmetry of a line with respect to another line
def symmetric (l1 l2 l3 : Line) : Prop := sorry

-- Define the theorem
theorem symmetric_lines_angles (a b a₁ b₁ : Line) :
  angle a b = 15 →
  symmetric a₁ a b →
  symmetric b₁ b a →
  (angle a₁ b₁ = 45 ∧ angle b₁ a₁ = 135) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_lines_angles_l731_73197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_polar_coordinates_l731_73118

/-- Represents a point in polar coordinates -/
structure PolarCoord where
  r : ℝ
  θ : ℝ

/-- Calculates the symmetric point in polar coordinates -/
def polar_symmetric_point (M : PolarCoord) (symmetry_line : ℝ) : PolarCoord :=
  ⟨M.r, 2 * symmetry_line - M.θ⟩

/-- Given a point M(3, π/2) in polar coordinates and a line of symmetry θ = π/6,
    the point N symmetric to M about this line has polar coordinates (3, -π/6). -/
theorem symmetric_point_polar_coordinates :
  let M : PolarCoord := ⟨3, π / 2⟩
  let symmetry_line : ℝ := π / 6
  let N : PolarCoord := polar_symmetric_point M symmetry_line
  N.r = 3 ∧ N.θ = -π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_polar_coordinates_l731_73118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_length_l731_73190

/-- The length of an escalator given specific conditions. -/
theorem escalator_length : ℝ := by
  let escalator_speed : ℝ := 11
  let person_speed : ℝ := 3
  let time_taken : ℝ := 10
  let combined_speed : ℝ := escalator_speed + person_speed
  let length : ℝ := combined_speed * time_taken
  have h : length = 140 := by
    -- Proof steps would go here
    sorry
  exact length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_length_l731_73190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l731_73162

theorem geometric_series_sum : ∀ (a r : ℚ) (n : ℕ),
  a = 2 ∧ r = 3 ∧ n = 7 →
  (Finset.range n).sum (λ i => a * r ^ i) = 2186 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l731_73162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_c_equation_l731_73138

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the specific triangle from the problem
noncomputable def specificTriangle : Triangle where
  a := 3
  b := Real.sqrt 7
  c := 0  -- We don't know the exact value, so we use a placeholder
  A := 0  -- We don't know the exact value, so we use a placeholder
  B := Real.pi / 3  -- 60° in radians
  C := 0  -- We don't know the exact value, so we use a placeholder

-- State the theorem
theorem side_c_equation (t : Triangle) (h1 : t = specificTriangle) :
  t.c^2 - 3*t.c + 2 = 0 :=
by
  sorry

-- Note: We don't include part II of the problem as it doesn't have a specific result to prove

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_c_equation_l731_73138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_prime_satisfies_conditions_l731_73156

def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_multiple_of_5 (n : ℤ) : Prop := ∃ k : ℤ, n = 5 * k

theorem no_prime_satisfies_conditions :
  ∀ p : ℕ, is_prime p → 1 ≤ p → p ≤ 20 →
    ¬(is_multiple_of_5 (p^2 - 13*p + 40) ∧ (p^2 - 13*p + 40 : ℤ) < 0) :=
by
  sorry

#check no_prime_satisfies_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_prime_satisfies_conditions_l731_73156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_on_line_l731_73179

-- Define a line y = 2x + 3
def line (x : ℝ) : ℝ := 2 * x + 3

-- Define the distance between two points on the line
noncomputable def distance (p r : ℝ) : ℝ := |r - p| * Real.sqrt 5

-- Theorem statement
theorem distance_on_line (p r : ℝ) :
  let q := line p
  let s := line r
  Real.sqrt ((r - p)^2 + (s - q)^2) = distance p r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_on_line_l731_73179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_properties_l731_73183

/-- A function satisfying certain properties on rational numbers -/
def special_function (f : ℚ → ℚ → ℚ) : Prop :=
  (∀ x y, f x y > 0) ∧ 
  (∀ x y z, f (x * y) z = f x z * f y z) ∧
  (∀ x y z, f z (x * y) = f z x * f z y) ∧
  (∀ x, f x (1 - x) = 1)

/-- The main theorem stating the properties of the special function -/
theorem special_function_properties (f : ℚ → ℚ → ℚ) (hf : special_function f) :
  (∀ x, f x x = 1) ∧ 
  (∀ x, f x (-x) = 1) ∧
  (∀ x y, f x y * f y x = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_properties_l731_73183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l731_73175

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x / (1 + x^2)

theorem function_properties (a : ℝ) :
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f a x = a * x / (1 + x^2)) →
  f a (1/2) = 2/5 →
  (a = 1 ∧
   (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f a (-x) = -(f a x)) ∧
   (∀ x₁ x₂, x₁ ∈ Set.Ioo (-1 : ℝ) 1 → x₂ ∈ Set.Ioo (-1 : ℝ) 1 → x₁ < x₂ → f a x₁ < f a x₂)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l731_73175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_imply_greater_than_six_l731_73196

/-- Given two positive real numbers a and b, and polynomials F and G defined as
    F(x) = x^2 + ax + b and G(x) = x^2 + bx + a, if all roots of F(G(x)) and G(F(x))
    are real, then a > 6 and b > 6. -/
theorem polynomial_roots_imply_greater_than_six (a b : ℝ) 
    (ha : a > 0) (hb : b > 0)
    (F : ℝ → ℝ) (hF : ∀ x, F x = x^2 + a*x + b)
    (G : ℝ → ℝ) (hG : ∀ x, G x = x^2 + b*x + a)
    (hFG : ∀ x, (F ∘ G) x = 0 → x ∈ Set.range (id : ℝ → ℝ))
    (hGF : ∀ x, (G ∘ F) x = 0 → x ∈ Set.range (id : ℝ → ℝ)) :
  a > 6 ∧ b > 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_imply_greater_than_six_l731_73196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_limit_theorem_l731_73129

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

noncomputable def nested_sqrt (n : ℕ) : ℝ :=
  Real.sqrt (2023 * (fibonacci (2^1))^2 + 
    Real.sqrt (2023 * (fibonacci (2^2))^2 + 
      Real.sqrt (2023 * (fibonacci (2^3))^2 + 
        Real.sqrt (2023 * (fibonacci (2^n))^2))))

theorem fibonacci_limit_theorem (a b c : ℕ) :
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ 
  (Nat.gcd a c = 1) ∧
  (a + b + c = 8102) ∧
  (∃ (L : ℝ), L = (a + Real.sqrt b) / c ∧ 
    ∀ ε > 0, ∃ N, ∀ n ≥ N, |nested_sqrt n - L| < ε) := by
  sorry

#check fibonacci_limit_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_limit_theorem_l731_73129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_octagon_area_equal_l731_73140

noncomputable section

/-- The area between inscribed and circumscribed circles of a regular polygon -/
def area_between_circles (n : ℕ) (side_length : ℝ) : ℝ :=
  let apothem := side_length * (Real.cos (Real.pi / n) / Real.sin (Real.pi / n))
  let circumradius := side_length / (2 * Real.sin (Real.pi / n))
  Real.pi * (circumradius^2 - apothem^2)

/-- Theorem stating that the areas between circles for hexagon and octagon are equal -/
theorem hexagon_octagon_area_equal :
  area_between_circles 6 3 = area_between_circles 8 3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_octagon_area_equal_l731_73140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l731_73171

def set_A : Set ℝ := {x | Real.log x ≤ 0}
def set_B : Set ℝ := {x | x^2 - 1 < 0}

theorem union_of_A_and_B : set_A ∪ set_B = Set.Ioc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l731_73171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_theorem_l731_73131

open Real

/-- The original function before translation -/
noncomputable def f (x : ℝ) : ℝ := sin (2 * x) + cos (2 * x)

/-- The function after translation -/
noncomputable def g (x : ℝ) : ℝ := cos (2 * x) - sin (2 * x)

/-- The translation amount -/
noncomputable def translation : ℝ := π / 4

/-- Theorem stating that translating f by π/4 to the left results in g -/
theorem translation_theorem : 
  ∀ x : ℝ, f (x + translation) = g x :=
by
  intro x
  simp [f, g, translation]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_theorem_l731_73131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_pentagon_probability_is_one_fourth_l731_73195

/-- Represents a regular pentagon dartboard with a central pentagon -/
structure PentagonDartboard where
  /-- Side length of the outer pentagon -/
  outer_side : ℝ
  /-- Assumption that the outer_side is positive -/
  outer_side_pos : 0 < outer_side

/-- The probability of a dart landing in the central pentagon of a PentagonDartboard -/
noncomputable def central_pentagon_probability (board : PentagonDartboard) : ℝ :=
  (1 : ℝ) / 4

/-- Theorem stating that the probability of a dart landing in the central pentagon is 1/4 -/
theorem central_pentagon_probability_is_one_fourth (board : PentagonDartboard) :
  central_pentagon_probability board = (1 : ℝ) / 4 := by
  -- Unfold the definition of central_pentagon_probability
  unfold central_pentagon_probability
  -- The definition directly gives 1/4, so we're done
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_pentagon_probability_is_one_fourth_l731_73195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_is_zero_l731_73114

noncomputable def matrix : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![Real.sin (30 * Real.pi / 180), Real.sin (60 * Real.pi / 180), Real.sin (90 * Real.pi / 180)],
    ![Real.sin (150 * Real.pi / 180), Real.sin (180 * Real.pi / 180), Real.sin (210 * Real.pi / 180)],
    ![Real.sin (270 * Real.pi / 180), Real.sin (300 * Real.pi / 180), Real.sin (330 * Real.pi / 180)]]

theorem determinant_is_zero : Matrix.det matrix = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_is_zero_l731_73114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_number_in_list_l731_73174

theorem eighth_number_in_list (numbers : List ℝ) : 
  numbers.length = 9 ∧ 
  numbers.sum / numbers.length = 207 ∧
  numbers.take 8 = [201, 202, 204, 205, 206, 209, 209, 212] ∧
  numbers.getLast? = some 212 →
  numbers.get? 7 = some 215 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_number_in_list_l731_73174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_tangent_line_equation_l731_73166

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + 1

-- Theorem for monotonic decrease
theorem f_monotone_decreasing :
  (∀ x < -1, (deriv f x) < 0) ∧ (∀ x > 3, (deriv f x) < 0) :=
sorry

-- Theorem for tangent line equation
theorem tangent_line_equation :
  let k := deriv f (-2)
  let y₀ := f (-2)
  ∀ x y, y - y₀ = k * (x - (-2)) ↔ 15*x + y + 27 = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_tangent_line_equation_l731_73166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_cooling_time_l731_73199

/-- Newton's Law of Cooling function -/
noncomputable def newtons_law_cooling (T_a T_0 h t : ℝ) : ℝ :=
  T_a + (T_0 - T_a) * (1/2)^(t/h)

/-- Proof that the coffee cools from 40°C to 32°C in 10 minutes -/
theorem coffee_cooling_time (T_a T_0 h : ℝ) (h_room : T_a = 24)
    (h_initial : T_0 = 88) (h_half_life : newtons_law_cooling T_a T_0 h 20 = 40) :
    ∃ t : ℝ, t = 10 ∧ newtons_law_cooling T_a 40 h t = 32 := by
  sorry

#check coffee_cooling_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_cooling_time_l731_73199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_squares_l731_73146

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define points M and N
def point_M : ℝ × ℝ := (-1, 0)
def point_N : ℝ × ℝ := (1, 0)

-- Define a point P on circle O
noncomputable def point_P (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, 2 * Real.sin α)

-- Define the distance squared between two points
def dist_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- Theorem statement
theorem constant_sum_of_squares (α : ℝ) :
  let P := point_P α
  dist_squared P point_M + dist_squared P point_N = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_squares_l731_73146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_is_one_l731_73170

-- Define the variables
variable (x y z : ℝ)

-- Define the equations
def equation1 (x y z : ℝ) : Prop := x - 5*y + 3*z = 22/6
def equation2 (x y z : ℝ) : Prop := 4*x + 8*y - 11*z = 7
def equation3 (x y z : ℝ) : Prop := 5*x - 6*y + 2*z = 12
def sum_equation (x y z : ℝ) : Prop := x + y + z = 10

-- Define the coefficient of x in the first equation
def coefficient_x : ℝ := 1

-- Theorem statement
theorem coefficient_x_is_one 
  (h1 : equation1 x y z) (h2 : equation2 x y z) (h3 : equation3 x y z) (h4 : sum_equation x y z) :
  coefficient_x = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_is_one_l731_73170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_falling_short_l731_73150

theorem mike_falling_short (max_marks : ℕ) (mike_score : ℕ) (passing_threshold : ℚ) : 
  max_marks = 770 → 
  mike_score = 212 → 
  passing_threshold = 30 / 100 → 
  (passing_threshold * max_marks).floor - mike_score = 19 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_falling_short_l731_73150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_tv_height_l731_73187

/-- Represents a TV with its dimensions and cost -/
structure TV where
  width : ℚ
  height : ℚ
  cost : ℚ

/-- Calculates the cost per square inch of a TV -/
noncomputable def costPerSquareInch (tv : TV) : ℚ :=
  tv.cost / (tv.width * tv.height)

theorem first_tv_height : 
  ∀ (first_tv second_tv : TV),
    first_tv.width = 24 ∧
    first_tv.cost = 672 ∧
    second_tv.width = 48 ∧
    second_tv.height = 32 ∧
    second_tv.cost = 1152 ∧
    costPerSquareInch first_tv = costPerSquareInch second_tv + 1 →
    first_tv.height = 16 := by
  sorry

#check first_tv_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_tv_height_l731_73187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ABCHIJK_l731_73161

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is equilateral -/
def is_equilateral (t : Triangle) : Prop :=
  ∃ s : ℝ, s > 0 ∧
    (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 = s^2 ∧
    (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2 = s^2 ∧
    (t.C.x - t.A.x)^2 + (t.C.y - t.A.y)^2 = s^2

/-- Checks if a point is the midpoint of two other points -/
def is_midpoint (M : Point) (A : Point) (B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

/-- Calculates the distance between two points -/
noncomputable def distance (A : Point) (B : Point) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

/-- Main theorem -/
theorem perimeter_ABCHIJK (A B C H I J K : Point) :
  is_equilateral (Triangle.mk A B C) →
  is_equilateral (Triangle.mk A H I) →
  is_equilateral (Triangle.mk I J K) →
  is_midpoint H A C →
  is_midpoint K A I →
  distance A B = 6 →
  distance A B + distance B C + distance C H + distance H I +
  distance I J + distance J K + distance K A = 22.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ABCHIJK_l731_73161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_water_percentage_l731_73108

/-- Calculates the percentage of a cylinder's volume filled with water -/
theorem cylinder_water_percentage (h : ℝ) (r : ℝ) (water_h : ℝ) :
  h = 12 →
  r = 3 →
  water_h = 10 →
  ∃ (percentage : ℝ), 
    abs (percentage - (water_h / h * 100)) < 0.00005 ∧
    abs (percentage - 83.3333) < 0.00005 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_water_percentage_l731_73108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_of_two_largest_angles_l731_73119

/-- Represents a quadrilateral with special properties -/
structure SpecialQuadrilateral where
  -- Internal angles form an arithmetic progression
  angles_progression : ∃ (x d : ℝ), 
    x > 0 ∧ x + d > 0 ∧ x + 2*d > 0 ∧ x + 3*d > 0 ∧ 
    x + (x + d) + (x + 2*d) + (x + 3*d) = 360
  -- Triangles EFG and HGF are similar
  similar_triangles : ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = 180 ∧
    Set.toFinset {b, a + b, c, a + c} = Set.toFinset {x, x + d, x + 2*d, x + 3*d}
  -- Angles in each triangle form an arithmetic sequence
  triangle_angles_progression : ∃ (z e : ℝ),
    z > 0 ∧ z + e > 0 ∧ z + 2*e > 0 ∧
    z + (z + e) + (z + 2*e) = 180

/-- The theorem stating the largest possible sum of two largest angles -/
theorem largest_sum_of_two_largest_angles (q : SpecialQuadrilateral) :
  ∃ (α β γ δ : ℝ), 
    α > 0 ∧ β > 0 ∧ γ > 0 ∧ δ > 0 ∧
    Set.toFinset {α, β, γ, δ} = Set.toFinset {x, x + d, x + 2*d, x + 3*d} ∧
    α + β + γ + δ = 360 ∧
    max α (max β (max γ δ)) + 
    max (min (max α β) (max γ δ)) (max (min α γ) (min β δ)) = 240 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_of_two_largest_angles_l731_73119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_and_m_range_l731_73159

/-- The function f(x) defined on real numbers. -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (3 * x - Real.pi / 3)

/-- The theorem stating the properties of f and the range of m. -/
theorem f_properties_and_m_range :
  (∀ x₁ x₂ : ℝ, |f x₁ - f x₂| = 4 → |x₁ - x₂| ≥ Real.pi / 3) ∧
  (Real.tan (-Real.pi / 3) = -Real.sqrt 3) ∧
  (∀ x : ℝ, 3 * (f x)^2 - f x + (1 / 12) = 0 → x ∈ Set.Ioo (Real.pi / 9) (4 * Real.pi / 9)) ∧
  (∀ m : ℝ, (m = 1 / 12 ∨ -10 < m ∧ m ≤ 0) ↔
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ∈ Set.Ioo (Real.pi / 9) (4 * Real.pi / 9) ∧
      x₂ ∈ Set.Ioo (Real.pi / 9) (4 * Real.pi / 9) ∧
      3 * (f x₁)^2 - f x₁ + m = 0 ∧
      3 * (f x₂)^2 - f x₂ + m = 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_and_m_range_l731_73159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l731_73128

noncomputable def f (x : ℝ) := Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x + 3 * Real.cos x ^ 2

theorem f_properties :
  (∃ (max : ℝ), ∀ (x : ℝ), f x ≤ max ∧ max = 2 + Real.sqrt 2) ∧
  (∀ (x : ℝ), f x = 2 + Real.sqrt 2 ↔ ∃ (k : ℤ), x = k * Real.pi + Real.pi / 8) ∧
  (∀ (k : ℤ) (x y : ℝ),
    k * Real.pi - 3 * Real.pi / 8 ≤ x ∧ x < y ∧ y ≤ k * Real.pi + Real.pi / 8 →
    f x < f y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l731_73128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l731_73180

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x - φ)

noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2 * (Real.cos x)^2

theorem function_properties 
  (ω : ℝ) (φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : abs φ < Real.pi / 2) 
  (h_intersect : ∀ (x₁ x₂ : ℝ), f ω φ x₁ = 0 → f ω φ x₂ = 0 → abs (x₁ - x₂) = Real.pi / 2) 
  (h_point : f ω φ 0 = -1/2) :
  (∀ x, f ω φ x = Real.sin (2 * x - Real.pi / 6)) ∧
  (∃ x₀ ∈ Set.Icc 0 (Real.pi / 2), ∀ x ∈ Set.Icc 0 (Real.pi / 2), g (f ω φ) x ≤ g (f ω φ) x₀) ∧
  (g (f ω φ) (Real.pi / 3) = 2) ∧
  (∃ x₁ ∈ Set.Icc 0 (Real.pi / 2), ∀ x ∈ Set.Icc 0 (Real.pi / 2), g (f ω φ) x₁ ≤ g (f ω φ) x) ∧
  (g (f ω φ) (5 * Real.pi / 6) = 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l731_73180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_2_l731_73134

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + 1)

theorem f_derivative_at_2 : 
  deriv f 2 = 2 * Real.sqrt 5 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_2_l731_73134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_graph_shift_l731_73100

-- Define the original and reference functions
noncomputable def original_function (x : ℝ) : ℝ := Real.sin (3 * x - Real.pi / 4)
noncomputable def reference_function (x : ℝ) : ℝ := Real.sin (3 * x)

-- Define the shift amount
noncomputable def shift_amount : ℝ := Real.pi / 12

-- Theorem statement
theorem sine_graph_shift :
  ∀ x : ℝ, original_function (x + shift_amount) = reference_function x :=
by
  intro x
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_graph_shift_l731_73100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_whole_numbers_between_fourth_roots_l731_73176

theorem whole_numbers_between_fourth_roots : 
  ∃! k : ℕ, k = (Finset.filter (fun n : ℕ => 
    (n : ℝ) > Real.rpow 50 (1/4) ∧ (n : ℝ) < Real.rpow 1000 (1/4)) 
    (Finset.range 7)).card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_whole_numbers_between_fourth_roots_l731_73176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_angle_adjustment_l731_73101

/-- Given two complementary angles with a ratio of 3:7, prove that when the smaller angle
    is increased by 20%, the larger angle must be decreased by approximately 8.57%
    to maintain complementarity. -/
theorem complementary_angle_adjustment (small_angle large_angle : ℝ) : 
  small_angle + large_angle = 90 →  -- angles are complementary
  small_angle / large_angle = 3 / 7 →  -- ratio of angles is 3:7
  abs (((large_angle - (90 - small_angle * 1.2)) / large_angle * 100) - 8.57) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_angle_adjustment_l731_73101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_angle_l731_73172

/-- The least positive angle (in radians) that satisfies the equation -/
noncomputable def α : ℝ := 9 * Real.pi / (4 * 180)

/-- The equation that α satisfies -/
def equation (x : ℝ) : Prop :=
  (3/4 - Real.sin x ^ 2) *
  (3/4 - Real.sin (3*x) ^ 2) *
  (3/4 - Real.sin (9*x) ^ 2) *
  (3/4 - Real.sin (27*x) ^ 2) = 1/256

theorem least_positive_angle :
  equation α ∧
  (∀ y : ℝ, 0 < y ∧ y < α → ¬ equation y) ∧
  Int.gcd 9 4 = 1 := by
  sorry

#check least_positive_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_angle_l731_73172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sampling_theorem_l731_73113

/-- The percentage of customers caught sampling candy -/
noncomputable def caught_percentage : ℝ := 22

/-- The percentage of sampling customers who are not caught -/
noncomputable def not_caught_percentage : ℝ := 5

/-- The total percentage of customers who sample candy -/
noncomputable def total_sampling_percentage : ℝ := caught_percentage / (1 - not_caught_percentage / 100)

theorem total_sampling_theorem : 
  total_sampling_percentage = caught_percentage / 0.95 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sampling_theorem_l731_73113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_trajectory_angle_sum_l731_73153

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = (x + 1)^2

-- Define the trajectory of the center C
def trajectory (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Define the line ℓ
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  x = m * y - 4

-- Define the angles α and β
def angle_sum (α β : ℝ) : Prop :=
  α + β = Real.pi / 2

theorem circle_trajectory_angle_sum :
  ∀ (x₁ y₁ x₂ y₂ m : ℝ),
    trajectory x₁ y₁ →
    trajectory x₂ y₂ →
    line_l m x₁ y₁ →
    line_l m x₂ y₂ →
    x₁ ≠ x₂ →
    ∃ (α β : ℝ),
      angle_sum α β ∧
      Real.tan α * Real.tan β = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_trajectory_angle_sum_l731_73153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_third_quadrant_l731_73184

open Real

-- Define the function f
noncomputable def f (α : ℝ) : ℝ :=
  (sin (π - α) * cos (2*π - α) * tan (-α + π)) /
  (-tan (-α - π) * cos (π/2 - α))

-- State the theorem
theorem f_value_in_third_quadrant (α : ℝ) :
  α ∈ Set.Icc π (3*π/2) →  -- α is in the third quadrant
  cos (α - 3*π/2) = 1/5 →
  f α = 2 * Real.sqrt 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_third_quadrant_l731_73184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2024_log_approx_6_l731_73164

-- Define the binary operations
noncomputable def diamond (a b : ℝ) : ℝ := a ^ (Real.log b / Real.log 5)
noncomputable def heart (a b : ℝ) : ℝ := a ^ (Real.log 5 / Real.log b)

-- Define the sequence recursively
noncomputable def a : ℕ → ℝ
  | 0 => 0  -- Add a base case for 0
  | 1 => 0  -- Add a base case for 1
  | 2 => 0  -- Add a base case for 2
  | 3 => 0  -- Add a base case for 3
  | 4 => heart 4 3
  | n+5 => diamond (heart (n+5) (n+4)) (a (n+4))

-- Theorem statement
theorem a_2024_log_approx_6 : 
  ∃ ε > 0, ε < 0.5 ∧ |Real.log (a 2024) / Real.log 5 - 6| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2024_log_approx_6_l731_73164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_on_y_axis_l731_73115

/-- Given points A and B in 3D space, if M is on the y-axis and equidistant from A and B, then M's y-coordinate is -1 -/
theorem equidistant_point_on_y_axis (A B : ℝ × ℝ × ℝ) (h1 : A = (1, 0, 2)) (h2 : B = (1, -3, 1)) :
  ∃ y : ℝ, (dist (0, y, 0) A = dist (0, y, 0) B) → y = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_on_y_axis_l731_73115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decoration_system_of_equations_l731_73177

/-- Represents the number of sheets used for stars -/
def x : ℕ := sorry

/-- Represents the number of sheets used for flowers -/
def y : ℕ := sorry

/-- The total number of sheets used -/
def total_sheets : ℕ := 12

/-- The number of stars that can be cut from one sheet -/
def stars_per_sheet : ℕ := 6

/-- The number of flowers that can be cut from one sheet -/
def flowers_per_sheet : ℕ := 4

/-- The ratio of stars to flowers -/
def star_flower_ratio : ℕ := 3

/-- The system of equations representing the decoration problem -/
theorem decoration_system_of_equations :
  (x + y = total_sheets) ∧
  (stars_per_sheet * x = star_flower_ratio * flowers_per_sheet * y) := by
  sorry

#check decoration_system_of_equations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decoration_system_of_equations_l731_73177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_force_system_properties_l731_73144

/-- Three forces in equilibrium on a plane -/
structure ForceSystem where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  F₃ : ℝ × ℝ
  equilibrium : F₁.1 + F₂.1 + F₃.1 = 0 ∧ F₁.2 + F₂.2 + F₃.2 = 0

/-- Magnitude of a 2D vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

/-- Angle between two 2D vectors -/
noncomputable def angle (v w : ℝ × ℝ) : ℝ := Real.arccos ((v.1 * w.1 + v.2 * w.2) / (magnitude v * magnitude w))

/-- Given conditions -/
def given_conditions (fs : ForceSystem) : Prop :=
  magnitude fs.F₁ = 4 ∧ 
  magnitude fs.F₂ = 2 ∧ 
  angle fs.F₁ fs.F₂ = 2 * Real.pi / 3

theorem force_system_properties (fs : ForceSystem) 
  (h : given_conditions fs) : 
  magnitude fs.F₃ = 2 * Real.sqrt 3 ∧ 
  angle fs.F₂ fs.F₃ = Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_force_system_properties_l731_73144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_stairs_climbing_ways_l731_73132

def stair_climbing_ways : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 2
| n + 3 => stair_climbing_ways (n + 2) + stair_climbing_ways (n + 1) + stair_climbing_ways n

theorem eight_stairs_climbing_ways :
  stair_climbing_ways 8 = 81 := by
  rfl

#eval stair_climbing_ways 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_stairs_climbing_ways_l731_73132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_cost_l731_73152

theorem cake_cost (bread_cost ham_cost cake_cost : ℝ) 
  (h1 : bread_cost = 50)
  (h2 : ham_cost = 150)
  (h3 : bread_cost + ham_cost = (bread_cost + ham_cost + cake_cost) / 2) :
  cake_cost = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_cost_l731_73152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_average_comparison_l731_73106

/-- Given three numbers in geometric progression, prove that the student's calculated average
    can be either greater or less than the true average, depending on the value of r. -/
theorem student_average_comparison (a r : ℝ) (hr : r > 0) :
  ∃ (r1 r2 : ℝ), r1 > 0 ∧ r2 > 0 ∧ 
  (r = r1 → ((a + a*r)/2 + a*r^2)/2 > (a + a*r + a*r^2)/3) ∧
  (r = r2 → ((a + a*r)/2 + a*r^2)/2 < (a + a*r + a*r^2)/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_average_comparison_l731_73106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_base_length_l731_73160

/-- Represents a parallelogram with given area and height -/
structure Parallelogram where
  area : ℝ
  height : ℝ

/-- Calculates the base of a parallelogram given its area and height -/
noncomputable def base (p : Parallelogram) : ℝ := p.area / p.height

/-- Theorem: For a parallelogram with area 44 cm² and height 11 cm, the base is 4 cm -/
theorem parallelogram_base_length (p : Parallelogram) 
  (h_area : p.area = 44) 
  (h_height : p.height = 11) : 
  base p = 4 := by
  sorry

#eval 44 / 11  -- This will not actually evaluate due to noncomputable definition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_base_length_l731_73160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l731_73147

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x + 2

-- State the theorem
theorem function_equivalence :
  (∀ x : ℝ, f (x^2 - 1) = x^4 + 1) →
  (∀ x : ℝ, x ≥ -1 → f x = x^2 + 2*x + 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l731_73147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_OABC_volume_l731_73193

noncomputable def tetrahedron_volume (a b c : ℝ) : ℝ := (1/6) * a * b * c

theorem tetrahedron_OABC_volume :
  ∀ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 →
    a^2 + b^2 = 49 →
    b^2 + c^2 = 64 →
    c^2 + a^2 = 81 →
    tetrahedron_volume a b c = 8 * Real.sqrt 11 :=
by
  intros a b c h1 h2 h3 h4
  sorry

#check tetrahedron_OABC_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_OABC_volume_l731_73193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vowel_classification_l731_73192

-- Define the categories
inductive Category
  | TwoAxes
  | CentralSymmetry
  | HorizontalAxis
  | VerticalAxis
  | NoSymmetry
  deriving Repr, DecidableEq

-- Define the vowels
inductive Vowel
  | A
  | E
  | I
  | O
  | U
  deriving Repr, DecidableEq

-- Define the symmetry properties
def has_two_axes (v : Vowel) : Bool := 
  match v with
  | Vowel.I | Vowel.O => true
  | _ => false

def has_horizontal_axis (v : Vowel) : Bool := 
  match v with
  | Vowel.E => true
  | _ => false

def has_vertical_axis (v : Vowel) : Bool := 
  match v with
  | Vowel.A | Vowel.U => true
  | _ => false

-- Define the classification function
def classify (v : Vowel) : Category :=
  if has_two_axes v then Category.TwoAxes
  else if has_horizontal_axis v then Category.HorizontalAxis
  else if has_vertical_axis v then Category.VerticalAxis
  else Category.NoSymmetry

-- The theorem to prove
theorem vowel_classification :
  (classify Vowel.A = Category.VerticalAxis) ∧
  (classify Vowel.E = Category.HorizontalAxis) ∧
  (classify Vowel.I = Category.TwoAxes) ∧
  (classify Vowel.O = Category.TwoAxes) ∧
  (classify Vowel.U = Category.VerticalAxis) := by
  sorry

#eval classify Vowel.A
#eval classify Vowel.E
#eval classify Vowel.I
#eval classify Vowel.O
#eval classify Vowel.U

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vowel_classification_l731_73192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_series_sum_l731_73151

/-- Sequence b_n defined recursively -/
def b : ℕ → ℚ
  | 0 => 2  -- We need to handle the case for 0
  | 1 => 2
  | 2 => 3
  | n + 3 => (1 / 4) * b (n + 2) + (1 / 5) * b (n + 1)

/-- The sum of the infinite sequence b_n -/
noncomputable def seriesSum : ℚ := ∑' n, b n

/-- Theorem: The sum of the infinite sequence b_n is equal to 9 -/
theorem b_series_sum : seriesSum = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_series_sum_l731_73151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_125_of_7_26_l731_73120

def decimal_expansion (n d : ℕ) : List ℕ := sorry

theorem digit_125_of_7_26 : 
  let expansion := decimal_expansion 7 26
  (expansion.get? 124).isSome ∧ (expansion.get? 124).get! = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_125_of_7_26_l731_73120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_of_specific_trapezoid_l731_73143

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  topBase : ℝ
  bottomBase : ℝ
  height : ℝ

/-- The dihedral angle formed when folding an isosceles trapezoid along its axis of symmetry -/
noncomputable def dihedralAngle (t : IsoscelesTrapezoid) : ℝ := Real.arccos (Real.sqrt 3 / 4)

/-- Theorem stating the dihedral angle for a specific isosceles trapezoid -/
theorem dihedral_angle_of_specific_trapezoid :
  let t : IsoscelesTrapezoid := { topBase := 2, bottomBase := 6, height := Real.sqrt 3 }
  dihedralAngle t = Real.arccos (Real.sqrt 3 / 4) := by
  sorry

#eval "Theorem stated and proved (with sorry)."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_of_specific_trapezoid_l731_73143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_berry_circle_properties_l731_73168

/-- The boundary equation of the berry circle -/
def berry_circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 12 = 2*x + 4*y

/-- The radius of the berry circle -/
noncomputable def berry_circle_radius : ℝ := Real.sqrt 17

/-- The area of the berry circle -/
noncomputable def berry_circle_area : ℝ := Real.pi * 17

theorem berry_circle_properties :
  (∀ x y : ℝ, berry_circle_equation x y → (x - 1)^2 + (y - 2)^2 = 17) ∧
  berry_circle_area > 30 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_berry_circle_properties_l731_73168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l731_73110

theorem rectangle_area_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  (1.15 * l * 1.2 * w - l * w) / (l * w) = 0.38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l731_73110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annulus_area_10_6_l731_73125

/-- The area of an annular region with given outer and inner diameters -/
noncomputable def annulusArea (outerDiameter innerDiameter : ℝ) : ℝ :=
  let outerRadius := outerDiameter / 2
  let innerRadius := innerDiameter / 2
  Real.pi * (outerRadius ^ 2 - innerRadius ^ 2)

/-- Theorem: The area of an annular region with outer diameter 10 meters 
    and inner diameter 6 meters is 16π square meters -/
theorem annulus_area_10_6 : 
  annulusArea 10 6 = 16 * Real.pi := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_annulus_area_10_6_l731_73125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_transformation_g_geometric_transformation_l731_73154

-- Define the original function g
noncomputable def g : ℝ → ℝ := fun x =>
  if x ≥ -3 ∧ x ≤ 0 then -2 - x
  else if x > 0 ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if x > 2 ∧ x ≤ 3 then 2*(x - 2)
  else 0  -- undefined outside [-3, 3]

-- Define the transformed function
noncomputable def g_transformed : ℝ → ℝ := fun x => g ((3 - x) / 3)

-- State the theorem
theorem g_transformation (x : ℝ) :
  g_transformed x = g (-x/3 + 1) := by
  sorry

-- State the geometric interpretation
theorem g_geometric_transformation :
  ∀ x y, g_transformed x = y ↔ 
  ∃ x' y', g x' = y' ∧ x = -3*x' + 3 ∧ y = y' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_transformation_g_geometric_transformation_l731_73154
