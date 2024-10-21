import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l581_58102

/-- Definition of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ :=
  2 * (r.width + r.height)

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ :=
  r.width * r.height

/-- Predicate to check if a rectangle is subdivided into n identical squares -/
def Rectangle.isSubdividedIntoSquares (r : Rectangle) (n : ℕ) : Prop :=
  ∃ (sideLength : ℝ), r.width = n * sideLength ∧ r.height = sideLength

/-- A rectangle ABCD divided into five identical squares with a perimeter of 200 cm has an area of 12500/9 cm². -/
theorem rectangle_area (ABCD : Rectangle) (p : ℝ) (n : ℕ) : 
  ABCD.perimeter = p → 
  p = 200 →
  n = 5 →
  ABCD.isSubdividedIntoSquares n →
  ABCD.area = 12500 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l581_58102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l581_58179

/-- The speed of a train crossing a bridge -/
noncomputable def train_speed (train_length bridge_length crossing_time : ℝ) : ℝ :=
  (train_length + bridge_length) / crossing_time

/-- Theorem: The speed of the train is approximately 11.11 m/s -/
theorem train_speed_approx :
  let train_length : ℝ := 100
  let bridge_length : ℝ := 300
  let crossing_time : ℝ := 36
  abs (train_speed train_length bridge_length crossing_time - 11.11) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l581_58179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relations_l581_58128

theorem angle_relations (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α = 1/7) (h4 : Real.sin β = Real.sqrt 10 / 10) :
  Real.sin (α + β) = Real.sqrt 5 / 5 ∧ α + 2*β = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relations_l581_58128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_parabola_chord_length_range_s_nonnegative_condition_l581_58168

-- Define the parabola structure
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

-- Define the chord length function
noncomputable def chord_length (p : Parabola) : ℝ :=
  let x₁ := (-p.b + Real.sqrt (p.b^2 - 4*p.a*p.c)) / (2*p.a)
  let x₂ := (-p.b - Real.sqrt (p.b^2 - 4*p.a*p.c)) / (2*p.a)
  abs (x₁ - x₂)

-- Theorem for part (1)
theorem chord_length_specific_parabola :
  chord_length ⟨1, -5, -6, by norm_num⟩ = 7 := by sorry

-- Theorem for part (2)
theorem chord_length_range (n : ℝ) (h : 1 ≤ n ∧ n < 3) :
  2 * Real.sqrt 2 ≤ chord_length ⟨1, n+1, -1, by norm_num⟩ ∧
  chord_length ⟨1, n+1, -1, by norm_num⟩ < 2 * Real.sqrt 5 := by sorry

-- Define the s function for part (3)
def s (m n t : ℝ) : ℝ :=
  (m*t + 4)^2 - (n + t)^2

-- Theorem for part (3)
theorem s_nonnegative_condition (m n : ℕ) (hm : m ≠ 1) :
  (∀ t, s m n t ≥ 0) ↔ ((m = 2 ∧ n = 2) ∨ (m = 4 ∧ n = 1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_parabola_chord_length_range_s_nonnegative_condition_l581_58168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_show_attendance_l581_58188

/-- The number of adults at a show given ticket prices and total receipts -/
def num_adults (adult_price child_price : ℚ) (total_receipts : ℚ) : ℕ :=
  let num_children := (total_receipts / (2 * adult_price + child_price)).floor
  (2 * num_children).toNat

theorem show_attendance (adult_price child_price total_receipts : ℚ) 
  (h1 : adult_price = 5.5)
  (h2 : child_price = 2.5)
  (h3 : total_receipts = 1026) :
  num_adults adult_price child_price total_receipts = 152 := by
  sorry

#eval num_adults 5.5 2.5 1026

end NUMINAMATH_CALUDE_ERRORFEEDBACK_show_attendance_l581_58188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l581_58177

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := x / Real.log x

-- State the theorem
theorem inequality_proof (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  f x < f (x^2) ∧ f (x^2) < (f x)^2 :=
by
  -- Proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l581_58177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cos_sin_l581_58162

theorem min_value_cos_sin (α β : Real) (h1 : 0 ≤ α ∧ α ≤ Real.pi/2) (h2 : 0 < β ∧ β ≤ Real.pi/2) :
  (Real.cos α) ^ 2 * Real.sin β + 1 / Real.sin β ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cos_sin_l581_58162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem1_proof_problem2_proof_l581_58164

-- Problem 1
theorem problem1_proof : (1) - 1^2023 + Real.sqrt 9 - (Real.pi - 3)^0 + abs (Real.sqrt 2 - 1) = Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem2_proof : 
  {x : ℝ | (3*x + 1)/2 ≥ (4*x + 3)/3 ∧ 2*x + 7 ≥ 5*x - 17} = {x : ℝ | 3 ≤ x ∧ x ≤ 8} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem1_proof_problem2_proof_l581_58164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_combination_probability_l581_58180

/-- Function to calculate the probability of two people picking the same color combination. -/
noncomputable def probability_same_combination (red blue green total : ℕ) : ℚ := 
  sorry

/-- The probability of two people picking the same color combination from a jar of candies. -/
theorem same_color_combination_probability : 
  let total_candies : ℕ := 12 + 12 + 6
  let red_candies : ℕ := 12
  let blue_candies : ℕ := 12
  let green_candies : ℕ := 6
  ∃ (prob : ℚ), 
    (prob = 251 / 3045) ∧ 
    (prob = probability_same_combination red_candies blue_candies green_candies total_candies)
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_combination_probability_l581_58180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l581_58127

/-- A parabola passing through specific points -/
structure Parabola where
  -- The evaluation function of the parabola
  eval : ℝ → ℝ
  -- The parabola passes through (2,0) and (-1,0)
  passes_through_A : eval 2 = 0
  passes_through_B : eval (-1) = 0
  -- The parabola intersects the y-axis at point C
  intersects_y_axis : ∃ y, eval 0 = y
  -- OC = 2
  OC_distance : |eval 0| = 2

/-- The theorem stating the possible equations of the parabola -/
theorem parabola_equation (p : Parabola) : 
  (∀ x, p.eval x = -x^2 + x + 2) ∨ (∀ x, p.eval x = x^2 - x - 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l581_58127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_point_on_line_parabola_real_points_quad_func_unique_real_point_l581_58132

noncomputable def is_real_point (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f a = a + 2

noncomputable def line (x : ℝ) : ℝ := (1/3) * x + 4

noncomputable def parabola (k : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + 2 - k

noncomputable def quad_func (m t n : ℝ) (x : ℝ) : ℝ := (1/8) * x^2 + (m - t + 1) * x + 2*n + 2*t - 2

theorem real_point_on_line :
  ∃ (a : ℝ), is_real_point line a ∧ a = 3 ∧ line a = 5 := by sorry

theorem parabola_real_points (k : ℝ) :
  (∃ (a b : ℝ), a ≠ b ∧ 
    is_real_point (parabola k) a ∧ 
    is_real_point (parabola k) b ∧ 
    (a - b)^2 = 8) →
  k = 0 := by sorry

theorem quad_func_unique_real_point (t : ℝ) :
  (∃! (a : ℝ), ∀ (m n : ℝ), -2 ≤ m ∧ m ≤ 3 →
    is_real_point (quad_func m t n) a ∧
    (∀ (n' : ℝ), is_real_point (quad_func m t n') a → n' ≥ t + 4)) →
  t = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_point_on_line_parabola_real_points_quad_func_unique_real_point_l581_58132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l581_58190

-- Define the right triangle ABC
def triangle_ABC (A B C : ℝ) : Prop :=
  (0 < A) ∧ (0 < B) ∧ (0 < C) ∧ (A + B + C = Real.pi) ∧ (C = Real.pi/2)

-- Define the conditions
def conditions (A B C : ℝ) : Prop :=
  triangle_ABC A B C ∧ (Real.sin A = 5/13) ∧ (B = 10)

-- State the theorem
theorem area_of_triangle_ABC (A B C : ℝ) (h : conditions A B C) : 
  (1/2) * B * Real.sqrt (((5/13)/Real.sin A)^2 - B^2) = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l581_58190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_line_MN_passes_through_E_l581_58176

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
def conditions (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ a^2 = 4 ∧ b = 1

-- Theorem for part I
theorem ellipse_equation (a b : ℝ) (h : conditions a b) :
  ∀ x y : ℝ, ellipse_C x y a b ↔ x^2 / 4 + y^2 = 1 :=
sorry

-- Define a point on the line x = 4
noncomputable def point_on_line (y : ℝ) : ℝ × ℝ := (4, y)

-- Define the left and right vertices of the ellipse
def left_vertex : ℝ × ℝ := (-2, 0)
def right_vertex : ℝ × ℝ := (2, 0)

-- Define a line passing through two points
noncomputable def line_through_points (p1 p2 : ℝ × ℝ) (x : ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  y1 + (y2 - y1) * (x - x1) / (x2 - x1)

-- Define the point E
def point_E : ℝ × ℝ := (1, 0)

-- Theorem for part II
theorem line_MN_passes_through_E (a b : ℝ) (h : conditions a b) (y : ℝ) :
  let P := point_on_line y
  let A := left_vertex
  let B := right_vertex
  ∃ (x1 y1 x2 y2 : ℝ),
    ellipse_C x1 y1 a b ∧
    ellipse_C x2 y2 a b ∧
    y1 = line_through_points A P x1 ∧
    y2 = line_through_points B P x2 ∧
    point_E.2 = line_through_points (x1, y1) (x2, y2) point_E.1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_line_MN_passes_through_E_l581_58176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_roll_distribution_l581_58149

/-- The number of guests at the conference --/
def num_guests : ℕ := 4

/-- The number of roll types --/
def num_roll_types : ℕ := 4

/-- The total number of rolls --/
def total_rolls : ℕ := 16

/-- The number of rolls each guest receives --/
def rolls_per_guest : ℕ := 4

/-- The probability that each guest gets one roll of each type --/
def probability : ℚ := 16 / 2925

theorem conference_roll_distribution :
  (num_guests = 4) →
  (num_roll_types = 4) →
  (total_rolls = 16) →
  (rolls_per_guest = 4) →
  (probability = 16 / 2925) →
  (∀ (m n : ℕ), Nat.Coprime m n → probability = m / n → m + n = 2941) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_roll_distribution_l581_58149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clearance_sale_savings_l581_58152

/-- The percentage saved on a sweater during a clearance sale -/
noncomputable def percentage_saved (amount_saved : ℝ) (amount_paid : ℝ) : ℝ :=
  (amount_saved / (amount_saved + amount_paid)) * 100

/-- Proof that the percentage saved is approximately 10% -/
theorem clearance_sale_savings : 
  let amount_saved : ℝ := 5
  let amount_paid : ℝ := 45
  abs (percentage_saved amount_saved amount_paid - 10) < 0.5 := by
  sorry

/-- Evaluate the percentage saved -/
def evaluate_savings : ℚ :=
  (5 : ℚ) / ((5 : ℚ) + (45 : ℚ)) * 100

#eval evaluate_savings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clearance_sale_savings_l581_58152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_sine_to_cosine_l581_58194

theorem min_shift_sine_to_cosine :
  let f (x : ℝ) := Real.sin (x / 2 + π / 3)
  let g (x : ℝ) := Real.cos (x / 2)
  ∃ m : ℝ, m > 0 ∧ (∀ x : ℝ, f (x + m) = g x) ∧
    (∀ m' : ℝ, m' > 0 → (∀ x : ℝ, f (x + m') = g x) → m ≤ m')
  ∧ m = π / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_sine_to_cosine_l581_58194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_root_of_shifted_difference_l581_58134

/-- A monic polynomial of degree 2014 -/
def MonicPoly2014 := {p : Polynomial ℝ // p.Monic ∧ p.degree = 2014}

/-- The theorem statement -/
theorem exists_root_of_shifted_difference (P Q : MonicPoly2014) 
  (h : ∀ x : ℝ, P.val.eval x ≠ Q.val.eval x) :
  ∃ x : ℝ, P.val.eval (x - 1) = Q.val.eval (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_root_of_shifted_difference_l581_58134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_PB_l581_58198

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

-- Define the upper vertex B
def B : ℝ × ℝ := (0, 1)

-- Define a point P on the ellipse
noncomputable def P : ℝ → ℝ × ℝ
  | θ => (Real.sqrt 5 * Real.cos θ, Real.sin θ)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem max_distance_PB :
  ∀ θ : ℝ, distance (P θ) B ≤ 5/2 ∧ 
  ∃ θ₀ : ℝ, distance (P θ₀) B = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_PB_l581_58198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l581_58145

/-- An ellipse with given properties has a major axis of length 6 -/
theorem ellipse_major_axis_length (a b c : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_a_gt_b : a > b)
  (h_ellipse : Set (ℝ × ℝ))
  (h_ellipse_eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) ∈ h_ellipse)
  (h_foci : c^2 = a^2 - b^2)
  (h_line : Set (ℝ × ℝ))
  (h_line_eq : ∃ m k : ℝ, ∀ x y : ℝ, y = m * x + k ↔ (x, y) ∈ h_line)
  (h_line_focus : (-c, 0) ∈ h_line)
  (h_y_intercept : (0, 1) ∈ h_line)
  (h_intersection : ∃ A B : ℝ × ℝ, A ∈ h_ellipse ∧ A ∈ h_line ∧ B ∈ h_ellipse ∧ B ∈ h_line)
  (h_distance_ratio : ∃ A B : ℝ × ℝ, A ∈ h_ellipse ∧ A ∈ h_line ∧ B ∈ h_ellipse ∧ B ∈ h_line ∧
    dist A (-c, 0) = 3 * dist B (-c, 0))
  (h_perpendicular : ∃ A : ℝ × ℝ, A ∈ h_ellipse ∧ A ∈ h_line ∧ (A.1 - c) * (A.2 - 0) = 0) :
  2 * a = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l581_58145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_coverage_l581_58123

-- Define a non-self-intersecting polygon in the plane
def Polygon : Type := Set (ℝ × ℝ)

-- Define a circle in the plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the interior of a set in the plane
noncomputable def Interior (S : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

-- Define a function to check if a set covers another set
def Covers (A B : Set (ℝ × ℝ)) : Prop := B ⊆ A

-- Main theorem
theorem circle_coverage (P : Polygon) (C : List Circle) 
  (h_cover : Covers (⋃ c ∈ C, Interior {x | ‖x - c.center‖ < c.radius}) (Interior P)) :
  ∃ (big_circle : Circle), 
    big_circle.radius = (C.map Circle.radius).sum ∧ 
    Covers (Interior {x | ‖x - big_circle.center‖ < big_circle.radius}) (Interior P) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_coverage_l581_58123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l581_58124

-- Define the line equation
def line (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 2 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line A.1 A.2 ∧ circle_eq A.1 A.2 ∧ line B.1 B.2 ∧ circle_eq B.1 B.2

-- Theorem statement
theorem chord_length (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l581_58124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_props_true_l581_58120

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (parallel_line_line : Line → Line → Prop)
variable (perpendicular_line_line : Line → Line → Prop)

-- Define the propositions
def prop1 (Line Plane : Type)
    (parallel_line_plane : Line → Plane → Prop)
    (parallel_plane_plane : Plane → Plane → Prop)
    (parallel_line_line : Line → Line → Prop) : Prop :=
  ∀ (m n : Line) (α β : Plane),
    parallel_line_plane m α → parallel_line_plane n β → parallel_plane_plane α β →
    parallel_line_line m n

def prop2 (Line Plane : Type)
    (perpendicular_line_plane : Line → Plane → Prop)
    (parallel_line_plane : Line → Plane → Prop)
    (parallel_plane_plane : Plane → Plane → Prop)
    (perpendicular_line_line : Line → Line → Prop) : Prop :=
  ∀ (m n : Line) (α β : Plane),
    perpendicular_line_plane m α → parallel_line_plane n β → parallel_plane_plane α β →
    perpendicular_line_line m n

def prop3 (Line Plane : Type)
    (parallel_line_plane : Line → Plane → Prop)
    (perpendicular_line_plane : Line → Plane → Prop)
    (perpendicular_plane_plane : Plane → Plane → Prop)
    (parallel_line_line : Line → Line → Prop) : Prop :=
  ∀ (m n : Line) (α β : Plane),
    parallel_line_plane m α → perpendicular_line_plane n β → perpendicular_plane_plane α β →
    parallel_line_line m n

def prop4 (Line Plane : Type)
    (perpendicular_line_plane : Line → Plane → Prop)
    (perpendicular_plane_plane : Plane → Plane → Prop)
    (perpendicular_line_line : Line → Line → Prop) : Prop :=
  ∀ (m n : Line) (α β : Plane),
    perpendicular_line_plane m α → perpendicular_line_plane n β → perpendicular_plane_plane α β →
    perpendicular_line_line m n

-- Theorem statement
theorem exactly_two_props_true :
  ∃ (Line Plane : Type)
     (parallel_line_plane perpendicular_line_plane : Line → Plane → Prop)
     (parallel_plane_plane perpendicular_plane_plane : Plane → Plane → Prop)
     (parallel_line_line perpendicular_line_line : Line → Line → Prop),
    ¬(prop1 Line Plane parallel_line_plane parallel_plane_plane parallel_line_line) ∧
    (prop2 Line Plane perpendicular_line_plane parallel_line_plane parallel_plane_plane perpendicular_line_line) ∧
    ¬(prop3 Line Plane parallel_line_plane perpendicular_line_plane perpendicular_plane_plane parallel_line_line) ∧
    (prop4 Line Plane perpendicular_line_plane perpendicular_plane_plane perpendicular_line_line) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_props_true_l581_58120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_staircase_partitions_l581_58157

/-- A staircase of height n consists of all cells (i,j) of an n × n square where 1 ≤ j ≤ i ≤ n -/
def Staircase (n : ℕ) : Set (ℕ × ℕ) :=
  {p | p.1 ≤ n ∧ p.2 ≤ n ∧ p.2 ≤ p.1}

/-- A partition of a staircase is a subset of its cells -/
def StaircasePartition (n : ℕ) : Set (Set (ℕ × ℕ)) :=
  {s | s ⊆ Staircase n}

/-- The number of ways to partition a staircase of height n -/
noncomputable def NumPartitions (n : ℕ) : ℕ :=
  Finset.card (Finset.powerset (Finset.filter (fun p => p.1 ≤ n ∧ p.2 ≤ n ∧ p.2 ≤ p.1) (Finset.product (Finset.range n) (Finset.range n))))

theorem staircase_partitions (n : ℕ) (h : n > 0) : 
  NumPartitions n = 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_staircase_partitions_l581_58157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_proof_l581_58129

open Real

theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) :
  c = 5 →
  b * (2 * sin B + sin A) + (2 * a + b) * sin A = 2 * c * sin C →
  (∃ (A B C : ℝ), A ∈ Set.Ioo 0 π ∧ B ∈ Set.Ioo 0 π ∧ C ∈ Set.Ioo 0 π ∧ A + B + C = π) →
  C = 2 * π / 3 ∧
  (cos A = 4 / 5 → b = 4 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_proof_l581_58129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_polynomials_with_property_l581_58113

theorem no_four_polynomials_with_property : 
  ¬ ∃ (p₁ p₂ p₃ p₄ : ℝ → ℝ), 
    (∀ (x : ℝ), ∀ (i j : Fin 4), i ≠ j → 
      (p₁ x + p₂ x + p₃ x + p₄ x - (match i with
        | 0 => p₁ x
        | 1 => p₂ x
        | 2 => p₃ x
        | 3 => p₄ x) - 
      (match j with
        | 0 => p₁ x
        | 1 => p₂ x
        | 2 => p₃ x
        | 3 => p₄ x)) ≠ 0) ∧ 
    (∀ (i j k : Fin 4), i ≠ j → j ≠ k → k ≠ i → 
      ∃ (x : ℝ), (match i with
        | 0 => p₁ x
        | 1 => p₂ x
        | 2 => p₃ x
        | 3 => p₄ x) + 
      (match j with
        | 0 => p₁ x
        | 1 => p₂ x
        | 2 => p₃ x
        | 3 => p₄ x) + 
      (match k with
        | 0 => p₁ x
        | 1 => p₂ x
        | 2 => p₃ x
        | 3 => p₄ x) = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_polynomials_with_property_l581_58113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l581_58121

/-- The function f(x) = e^x - ax^2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x^2

/-- The derivative of f(x) -/
noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 2 * a * x

/-- The equation f(x) + f'(x) = 2 - ax^2 -/
def equation (a : ℝ) (x : ℝ) : Prop :=
  f a x + f_prime a x = 2 - a * x^2

theorem range_of_a (a : ℝ) :
  (∃ x, x ∈ Set.Ioo 0 1 ∧ equation a x) →
  a ∈ Set.Ioo 1 (Real.exp 1 - 1) := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l581_58121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_interchanged_numbers_l581_58138

def M : ℕ := 9876543210

/-- Number of ways to choose k disjoint pairs from n adjacent positions -/
def choose_pairs (n k : ℕ) : ℕ := Nat.choose (n - k) k

/-- Sum of binomial coefficients for choosing disjoint pairs -/
def sum_disjoint_pairs (n : ℕ) : ℕ :=
  (List.range 5).map (choose_pairs (n - 1) ∘ (· + 1)) |>.sum

theorem count_interchanged_numbers :
  sum_disjoint_pairs 10 = 88 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_interchanged_numbers_l581_58138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billion_product_without_zeros_l581_58178

theorem billion_product_without_zeros :
  ∃ (a b : ℕ), 
    a > 0 ∧ b > 0 ∧
    (∀ d : ℕ, d > 0 → d < 10 → (a / (10 ^ (Nat.log a d))) % 10 ≠ 0) ∧
    (∀ d : ℕ, d > 0 → d < 10 → (b / (10 ^ (Nat.log b d))) % 10 ≠ 0) ∧
    a * b = 1000000000 :=
by
  -- Provide the two numbers we found
  use 512, 1953125
  
  -- Split the goal into separate parts
  apply And.intro
  · norm_num -- Proves 512 > 0
  apply And.intro
  · norm_num -- Proves 1953125 > 0
  apply And.intro
  · -- Prove that 512 has no zeros
    intro d hd1 hd2
    sorry -- This part requires more detailed proof
  apply And.intro
  · -- Prove that 1953125 has no zeros
    intro d hd1 hd2
    sorry -- This part requires more detailed proof
  · -- Prove the product equals 1000000000
    norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_billion_product_without_zeros_l581_58178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_gain_percentage_l581_58105

noncomputable def cost_price : ℝ := 1200
noncomputable def initial_loss_percentage : ℝ := 10
noncomputable def price_increase : ℝ := 180

noncomputable def initial_selling_price : ℝ := cost_price * (1 - initial_loss_percentage / 100)
noncomputable def new_selling_price : ℝ := initial_selling_price + price_increase

theorem watch_gain_percentage :
  (new_selling_price - cost_price) / cost_price * 100 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_gain_percentage_l581_58105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interest_after_trebling_l581_58146

/-- Simple interest calculation function -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

theorem total_interest_after_trebling (P R : ℝ) :
  simpleInterest P R 10 = 600 →
  simpleInterest (3 * P) R 5 + 600 = 1500 := by
  intro h
  sorry

#check total_interest_after_trebling

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interest_after_trebling_l581_58146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_parabola_to_line_l581_58173

/-- The parabola defined by x²=4y -/
def parabola (x y : ℝ) : Prop := x^2 = 4*y

/-- The line y=x-3 -/
def line (x y : ℝ) : Prop := y = x - 3

/-- The distance from a point (x,y) to the line y=x-3 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x - 3 - y| / Real.sqrt 2

theorem minimum_distance_parabola_to_line :
  ∀ M : ℝ × ℝ, parabola M.1 M.2 → 
  (∃ (m : ℝ × ℝ), parabola m.1 m.2 ∧ distance_to_line m.1 m.2 ≤ distance_to_line M.1 M.2) →
  ∃ (m : ℝ × ℝ), parabola m.1 m.2 ∧ distance_to_line m.1 m.2 = Real.sqrt 2 :=
by
  sorry

#check minimum_distance_parabola_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_parabola_to_line_l581_58173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equation_proof_l581_58104

theorem trig_equation_proof (a b : ℝ) (θ : ℝ) (h : 0 < a) (k : 0 < b) 
  (eq : (Real.sin θ)^6 / a^2 + (Real.cos θ)^6 / b^2 = 1 / (a + b)) : 
  (Real.sin θ)^12 / a^5 + (Real.cos θ)^12 / b^5 = 1 / (a + b)^5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equation_proof_l581_58104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_total_cost_l581_58191

-- Define the total cost function
noncomputable def total_cost (V : ℝ) : ℝ := 1000 * V + 16000 / V - 500

-- Theorem statement
theorem minimum_total_cost :
  ∀ V : ℝ, V > 0.5 → total_cost V ≥ 7500 ∧ 
  ∃ V₀ : ℝ, V₀ > 0.5 ∧ total_cost V₀ = 7500 := by
  sorry

#check minimum_total_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_total_cost_l581_58191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_n_characterization_l581_58139

/-- A function that checks if a permutation of [1..n] satisfies the divisibility condition -/
def is_valid_arrangement (n : ℕ) (perm : List ℕ) : Prop :=
  perm.length = n ∧
  perm.toFinset = Finset.range n.succ \ {0} ∧
  ∀ i, i < n - 1 → (perm.get? i).isSome ∧ (perm.get? (i + 1)).isSome ∧
    (((perm.get? i).get! ∣ (perm.get? (i + 1)).get!) ∨ ((perm.get? (i + 1)).get! ∣ (perm.get? i).get!))

/-- The set of valid n -/
def valid_n : Set ℕ := {n | n > 1 ∧ ∃ perm : List ℕ, is_valid_arrangement n perm}

/-- The theorem to be proved -/
theorem valid_n_characterization : valid_n = {2, 3, 4, 6} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_n_characterization_l581_58139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_code_permutations_l581_58112

theorem area_code_permutations : 
  Finset.card (Finset.univ.filter (λ p : Fin 4 → Fin 4 ↦ Function.Injective p)) = 24 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_code_permutations_l581_58112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l581_58116

/-- Triangle ABC with vertices A(0,2), B(0,0), C(10,0) -/
structure Triangle where
  A : ℝ × ℝ := (0, 2)
  B : ℝ × ℝ := (0, 0)
  C : ℝ × ℝ := (10, 0)

/-- Area of a triangle given base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

/-- Area of the right part of the triangle divided by line x = a -/
noncomputable def rightArea (a : ℝ) : ℝ := triangleArea (10 - a) 2

/-- Theorem: The vertical line x = a divides the triangle into two equal areas iff a = 5 -/
theorem equal_area_division (t : Triangle) (a : ℝ) : 
  rightArea a = triangleArea 10 2 / 2 ↔ a = 5 := by
  sorry

#check equal_area_division

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l581_58116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_properties_l581_58171

noncomputable section

def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + Real.pi / 3)

theorem cosine_function_properties (ω : ℝ) (h1 : ω > 0) :
  (∀ x y : ℝ, x - y = Real.pi / (2 * ω) → f ω x = f ω y) →
  ω = 2 ∧
  (∀ k : ℤ, ∃ x : ℝ, x = k * Real.pi / 2 - Real.pi / 6 ∧ ∀ y : ℝ, f ω x = f ω y → x = y) ∧
  (∀ k : ℤ, ∀ x : ℝ, k * Real.pi - 2 * Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi - Real.pi / 6 →
    ∀ y : ℝ, k * Real.pi - 2 * Real.pi / 3 ≤ y ∧ y ≤ x → f ω x ≥ f ω y) ∧
  (∀ k : ℤ, f ω (k * Real.pi - Real.pi / 6) = 1 ∧ ∀ x : ℝ, f ω x ≤ 1) ∧
  (∀ k : ℤ, f ω (k * Real.pi + Real.pi / 3) = -1 ∧ ∀ x : ℝ, f ω x ≥ -1) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_properties_l581_58171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l581_58108

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.cos (ω * x) * Real.sin (ω * x - Real.pi/3) + Real.sqrt 3 * (Real.cos (ω * x))^2 - Real.sqrt 3 / 4

theorem problem_solution (ω : ℝ) (A B : ℝ) (a b : ℝ) :
  ω > 0 →
  (∃ (center axis : ℝ), |center - axis| = Real.pi/4) →
  f ω A = 0 →
  Real.sin B = 4/5 →
  a = Real.sqrt 3 →
  ω = 1 ∧
  (∃ (k : ℤ), ∃ (axis : ℝ), axis = (1/2) * ↑k * Real.pi + Real.pi/12) ∧
  b = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l581_58108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l581_58106

/-- The circle equation -/
def circleEq (x y : ℝ) : Prop := x^2 - 8*x + y^2 = 84

/-- The line equation -/
def lineEq (x y : ℝ) : Prop := y = 6 - 2*x

/-- The region of interest -/
def region (x y : ℝ) : Prop := circleEq x y ∧ lineEq x y ∧ y ≥ 0

/-- The area of the region -/
noncomputable def area : ℝ := 2.34 * Real.pi

/-- Theorem stating that the area of the region is 2.34π -/
theorem area_of_region : area = 2.34 * Real.pi := by
  -- The proof is omitted for now
  sorry

#check area_of_region

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l581_58106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_angle_properties_l581_58110

-- Define the inclination angles
variable (α β : Real)

-- Define the equations of the lines
noncomputable def line_l₁ (x y : Real) : Prop := x - Real.sqrt 2 * y + 1 = 0
noncomputable def line_l (x y : Real) : Prop := 2 * Real.sqrt 2 * x - y + 6 = 0

-- Define the point P
noncomputable def point_P : Real × Real := (-Real.sqrt 2, 2)

-- State the theorem
theorem line_and_angle_properties :
  -- Given conditions
  (∀ x y, line_l₁ x y → Real.tan α = Real.sqrt 2 / 2) →
  β = 2 * α →
  (∀ x y, line_l x y → (y - point_P.2) = Real.tan β * (x - point_P.1)) →
  -- Conclusions
  (∀ x y, line_l x y ↔ 2 * Real.sqrt 2 * x - y + 6 = 0) ∧
  (Real.cos (2 * β) / (1 + Real.cos (2 * β) - Real.sin (2 * β)) = 1 / 2 + Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_angle_properties_l581_58110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_probability_comparison_l581_58183

/-- The probability of a correct answer when guessing -/
noncomputable def p : ℝ := 1 / 4

/-- The probability of an incorrect answer when guessing -/
noncomputable def q : ℝ := 1 - p

/-- The number of questions in the 2011 exam -/
def n₁ : ℕ := 20

/-- The number of correct answers required to pass in 2011 -/
def k₁ : ℕ := 3

/-- The number of questions in the 2012 exam -/
def n₂ : ℕ := 40

/-- The number of correct answers required to pass in 2012 -/
def k₂ : ℕ := 6

/-- The probability of passing the exam in 2011 -/
noncomputable def prob_2011 : ℝ := 1 - (Finset.sum (Finset.range k₁) (λ i => (n₁.choose i : ℝ) * p ^ i * q ^ (n₁ - i)))

/-- The probability of passing the exam in 2012 -/
noncomputable def prob_2012 : ℝ := 1 - (Finset.sum (Finset.range k₂) (λ i => (n₂.choose i : ℝ) * p ^ i * q ^ (n₂ - i)))

theorem exam_probability_comparison : prob_2011 > prob_2012 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_probability_comparison_l581_58183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l581_58140

/-- Calculates the interest rate given the principal, time, and total interest for a simple interest loan. -/
noncomputable def calculate_interest_rate (principal : ℝ) (time : ℝ) (total_interest : ℝ) : ℝ :=
  (total_interest * 100) / (principal * time)

/-- Theorem stating that for a loan with $15000 principal and $5400 interest after 3 years, the interest rate is 12% -/
theorem interest_rate_calculation : 
  let principal : ℝ := 15000
  let time : ℝ := 3
  let total_interest : ℝ := 5400
  calculate_interest_rate principal time total_interest = 12 := by
  sorry

-- Remove the #eval statement as it's not compatible with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l581_58140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_last_term_sum_l581_58135

/-- Represents the sequence where the nth term is n * 9 + (n+1) = 111...1 (with n ones) -/
def seq (n : ℕ) : ℕ := n * 9 + (n + 1)

/-- The result of the nth term in the sequence is 111...1 (with n ones) -/
def seq_result (n : ℕ) : ℕ := (10^n) - 1

/-- The last term in the sequence has 6 ones -/
def last_term_ones : ℕ := 6

/-- Triangle (△) is the multiplicand in the last term of the sequence -/
def triangle : ℕ := 12345

/-- O is the addend in the last term of the sequence -/
def O : ℕ := 6

theorem sequence_last_term_sum :
  seq last_term_ones = seq_result last_term_ones ∧
  triangle * 9 + O = seq_result last_term_ones ∧
  triangle + O = 12351 := by
  sorry

#eval triangle + O

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_last_term_sum_l581_58135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_1_to_3_eq_0_6_l581_58159

/-- Distribution of random variable X -/
def X : ℕ → ℝ
| 0 => 0.1
| 1 => 0.1
| 2 => 0.2
| 3 => 0.3
| 4 => 0.2
| 5 => 0.1
| _ => 0

/-- The sum of all probabilities equals 1 -/
axiom sum_prob_one : ∑' n, X n = 1

/-- Probability of X being between 1 and 3, inclusive -/
def P_1_to_3 : ℝ := X 1 + X 2 + X 3

/-- Theorem: P(1 ≤ X ≤ 3) = 0.6 -/
theorem prob_1_to_3_eq_0_6 : P_1_to_3 = 0.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_1_to_3_eq_0_6_l581_58159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_l581_58133

/-- A cone with three mutually perpendicular generatrices has a central angle
    of 2√6π/3 when its lateral surface is unfolded. -/
theorem cone_central_angle (l : ℝ) (h : l > 0) :
  let R := (Real.sqrt 6 * l) / 3
  2 * Real.pi * R / l = 2 * Real.sqrt 6 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_l581_58133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_three_half_pi_plus_two_theta_l581_58196

theorem sin_three_half_pi_plus_two_theta (θ : Real) (h : Real.tan θ = 1/3) :
  Real.sin (3/2 * Real.pi + 2 * θ) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_three_half_pi_plus_two_theta_l581_58196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rajesh_savings_percentage_l581_58170

def monthly_salary : ℚ := 15000
def food_percentage : ℚ := 40 / 100
def medicine_percentage : ℚ := 20 / 100
def savings : ℚ := 4320

def remaining_amount : ℚ := monthly_salary - (monthly_salary * (food_percentage + medicine_percentage))

theorem rajesh_savings_percentage :
  savings / remaining_amount * 100 = 72 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rajesh_savings_percentage_l581_58170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_function_l581_58101

theorem max_value_of_function (x : ℝ) : 
  Real.cos (2 * x) + 6 * Real.cos (π / 2 - x) ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_function_l581_58101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_calculation_l581_58197

/-- Calculates a person's savings given their income sources, tax rates, and expenditure ratio. -/
noncomputable def calculate_savings (total_income : ℝ) (income_ratio : Fin 3 → ℝ) (tax_rates : Fin 2 → ℝ) (income_expenditure_ratio : Fin 2 → ℝ) : ℝ :=
  let income_sum := (income_ratio 0) + (income_ratio 1) + (income_ratio 2)
  let salary := total_income * (income_ratio 0) / income_sum
  let side_business := total_income * (income_ratio 1) / income_sum
  let interest := total_income * (income_ratio 2) / income_sum
  let tax_salary := salary * (tax_rates 0)
  let tax_side_business := side_business * (tax_rates 1)
  let net_income := salary + side_business + interest - tax_salary - tax_side_business
  let expenditure := total_income * (income_expenditure_ratio 1) / ((income_expenditure_ratio 0) + (income_expenditure_ratio 1))
  net_income - expenditure

/-- Theorem stating that given the problem conditions, the person's savings are 950. -/
theorem savings_calculation :
  let total_income : ℝ := 10000
  let income_ratio : Fin 3 → ℝ := ![5, 3, 2]
  let tax_rates : Fin 2 → ℝ := ![0.15, 0.10]
  let income_expenditure_ratio : Fin 2 → ℝ := ![10, 8]
  calculate_savings total_income income_ratio tax_rates income_expenditure_ratio = 950 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_calculation_l581_58197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l581_58154

theorem problem_solution :
  (∀ x : ℝ, 3 * (9 : ℝ)^x * 81 = (3 : ℝ)^21 → x = 8) ∧
  (∀ a m n : ℝ, a^m = 2 ∧ a^n = 5 → 
    (a^(m+n) = 10 ∧ a^(3*m-4*n) = 8/625)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l581_58154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l581_58189

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x) / ((x-3)^2)

theorem inequality_solution :
  {x : ℝ | f x ≥ 0 ∧ x ≠ 3} = {x : ℝ | x ≤ -5 ∨ (0 ≤ x ∧ x < 3) ∨ x > 3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l581_58189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_condition_a_value_t_comparison_l581_58182

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 4 * x + a
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define t1, t2, and t3
noncomputable def t1 (a : ℝ) (x : ℝ) : ℝ := (1/2) * f a x
noncomputable def t2 (a : ℝ) (x : ℝ) : ℝ := g a x
noncomputable def t3 (x : ℝ) : ℝ := 2^x

-- State the theorems
theorem monotonicity_condition (a : ℝ) :
  ∀ m : ℝ, (∃ x y : ℝ, x < y ∧ x ≥ -1 ∧ y ≤ 3*m ∧ f a x > f a y) ↔ m > (1/3) :=
by sorry

theorem a_value (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 1 = g a 1 → a = 2 :=
by sorry

theorem t_comparison (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x : ℝ, 0 < x ∧ x < 1 → t2 a x < t1 a x ∧ t1 a x < t3 x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_condition_a_value_t_comparison_l581_58182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friends_tshirt_purchase_l581_58137

/-- The total amount spent by a group of friends buying discounted t-shirts -/
noncomputable def total_spent (num_friends : ℕ) (original_price : ℝ) (discount_percent : ℝ) : ℝ :=
  num_friends * (original_price * (1 - discount_percent / 100))

/-- Theorem: 4 friends buying t-shirts at 50% off from $20 spend $40 in total -/
theorem friends_tshirt_purchase :
  total_spent 4 20 50 = 40 := by
  -- Unfold the definition of total_spent
  unfold total_spent
  -- Simplify the arithmetic
  simp [mul_assoc, mul_comm, mul_left_comm]
  -- Prove the equality
  norm_num

#check friends_tshirt_purchase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friends_tshirt_purchase_l581_58137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixing_alcohol_solutions_l581_58126

/-- Represents an alcohol solution with a given volume and percentage --/
structure Solution where
  volume : ℝ
  percentage : ℝ

/-- Calculates the total amount of alcohol in a solution --/
noncomputable def alcohol_amount (s : Solution) : ℝ :=
  s.volume * s.percentage / 100

/-- Theorem: Mixing 40L of 20% solution with 60L of unknown percentage to get 50% solution --/
theorem mixing_alcohol_solutions (x : ℝ) :
  let solution1 : Solution := ⟨40, 20⟩
  let solution2 : Solution := ⟨60, x⟩
  let mixed_solution : Solution := ⟨solution1.volume + solution2.volume, 50⟩
  alcohol_amount solution1 + alcohol_amount solution2 = alcohol_amount mixed_solution →
  x = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixing_alcohol_solutions_l581_58126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_problem_l581_58117

theorem sin_cos_problem (α : ℝ) 
  (h1 : Real.sin α = 1/3) 
  (h2 : π/2 < α ∧ α < π) : 
  Real.sin (α - π/6) = (Real.sqrt 3 + 2 * Real.sqrt 2) / 6 ∧ 
  Real.cos (2 * α) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_problem_l581_58117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_inequality_l581_58192

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def S_n (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_inequality (a₁ : ℝ) (d : ℝ) (n : ℕ) 
  (h : d < 0) (h_n : n ≥ 1) :
  n * (arithmetic_sequence a₁ d n) ≤ S_n a₁ d n ∧ S_n a₁ d n ≤ n * a₁ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_inequality_l581_58192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_l581_58184

/-- Helper function to calculate surface area of a cube -/
noncomputable def surfaceArea (cube : Set (ℝ × ℝ × ℝ)) : ℝ :=
  sorry -- Implementation details omitted

/-- Given three vertices of a cube, prove that its surface area is 294 -/
theorem cube_surface_area (A B C : ℝ × ℝ × ℝ) : 
  A = (2, 4, 6) → B = (3, 0, -3) → C = (6, -5, 5) → 
  ∃ (cube : Set (ℝ × ℝ × ℝ)), 
    A ∈ cube ∧ B ∈ cube ∧ C ∈ cube ∧ 
    (∃ (S : ℝ), S = surfaceArea cube ∧ S = 294) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_l581_58184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l581_58158

noncomputable def f (x : ℝ) := Real.log (x^2 - x - 2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -1 ∨ x > 2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l581_58158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_circumcenter_l581_58141

noncomputable def Length (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

noncomputable def Slope (P Q : ℝ × ℝ) : ℝ :=
  (Q.2 - P.2) / (Q.1 - P.1)

def RightTriangle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

def Equidistant (D A B C : ℝ × ℝ) : Prop :=
  Length D A = Length D B ∧ Length D B = Length D C

theorem right_triangle_circumcenter (A B C : ℝ × ℝ) (D : ℝ × ℝ) :
  RightTriangle A B C →
  Length A C = 100 →
  Slope A C = 4/3 →
  Equidistant D A B C →
  D = (30, 40) ∧ Length A B = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_circumcenter_l581_58141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_islander_theorem_l581_58136

/-- Represents the type of islander: Knight (truth-teller) or Liar --/
inductive IslanderType
  | Knight
  | Liar
deriving BEq, Repr

/-- Represents an arrangement of islanders around a table --/
def Arrangement := List IslanderType

/-- Checks if the given statement is consistent with the islander types --/
def is_consistent (arr : Arrangement) : Bool :=
  arr.zipWith (λ a b => match (a, b) with
    | (IslanderType.Knight, IslanderType.Liar) => true
    | (IslanderType.Liar, IslanderType.Knight) => false
    | _ => false
  ) (arr.rotate 1) |>.all id

/-- The main theorem about the number of liars in the arrangement --/
theorem islander_theorem (arr : Arrangement) (h1 : arr.length = 450) 
  (h2 : is_consistent arr) :
  (arr.count IslanderType.Liar = 150) ∨ (arr.count IslanderType.Liar = 450) :=
by sorry

#eval is_consistent [IslanderType.Knight, IslanderType.Liar, IslanderType.Knight]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_islander_theorem_l581_58136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_side_length_l581_58114

/-- An equilateral triangle inscribed in an ellipse -/
structure InscribedTriangle where
  /-- The ellipse equation: x^2 + 9y^2 = 9 -/
  ellipse : ℝ → ℝ → Prop := fun x y ↦ x^2 + 9*y^2 = 9

  /-- One vertex of the triangle is at (0,1) -/
  vertex : ℝ × ℝ := (0, 1)

  /-- The altitude passes through (0,1) and is aligned with the y-axis -/
  altitude_aligned : Prop := True  -- This is always true given the setup

  /-- The triangle is equilateral -/
  is_equilateral : Prop

/-- The square of the side length of the inscribed equilateral triangle -/
noncomputable def side_length_squared (t : InscribedTriangle) : ℝ := 972 / 196

/-- Theorem: The square of the side length of the inscribed equilateral triangle is 972/196 -/
theorem inscribed_triangle_side_length 
  (t : InscribedTriangle) : side_length_squared t = 972 / 196 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_side_length_l581_58114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_height_l581_58151

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Calculate the intersection point of two lines -/
noncomputable def intersection (l1 l2 : Line) : Point :=
  { x := (l2.intercept - l1.intercept) / (l1.slope - l2.slope),
    y := l1.slope * (l2.intercept - l1.intercept) / (l1.slope - l2.slope) + l1.intercept }

theorem intersection_height : 
  let pole1 : Point := { x := 0, y := 30 }
  let pole2 : Point := { x := 150, y := 90 }
  let line1 : Line := { slope := (pole2.y - 0) / (pole2.x - 0), intercept := pole1.y }
  let line2 : Line := { slope := (0 - pole2.y) / (0 - pole2.x), intercept := 0 }
  let intersectionPoint := intersection line1 line2
  intersectionPoint.y = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_height_l581_58151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_side_length_l581_58174

noncomputable def triangle_in_circle (r : ℝ) (a b c : ℝ) (angle_B : ℝ) : Prop :=
  r > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧
  angle_B > 0 ∧ angle_B < Real.pi ∧
  a ≤ 2 * r ∧ b ≤ 2 * r ∧ c ≤ 2 * r ∧
  a / (Real.sin angle_B) = 2 * r

theorem third_side_length 
  (r : ℝ) (a b : ℝ) (angle_B : ℝ) :
  triangle_in_circle r a b 300 angle_B →
  r = 300 ∧ a = 300 ∧ b = 450 ∧ angle_B = Real.pi / 3 →
  300 = 300 := by
  sorry

#check third_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_side_length_l581_58174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l581_58195

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x : ℝ | x < 0 ∨ x > 3}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ioc 3 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l581_58195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_side_ratio_l581_58175

theorem acute_triangle_side_ratio (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle conditions
  A + B + C = π ∧  -- Sum of angles in a triangle
  A = 2 * B ∧  -- Given condition
  a / Real.sin A = b / Real.sin B  -- Sine law
  →
  Real.sqrt 2 < a / b ∧ a / b < Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_side_ratio_l581_58175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_8m_is_integer_l581_58186

theorem sqrt_8m_is_integer (m : ℕ) : m = 2 → ∃ k : ℕ, (8 * m : ℝ).sqrt = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_8m_is_integer_l581_58186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l581_58147

def a : ℕ → ℤ
  | 0 => 1  -- Adding the case for 0
  | 1 => 1
  | 2 => -3
  | 3 => 5
  | 4 => -7
  | 5 => 9
  | n + 6 => a n  -- Adjusting the recursive case

theorem sequence_formula (n : ℕ) : a n = (-1)^(n+1) * (2*n - 1) := by
  sorry

#eval a 1  -- Testing the function
#eval a 2
#eval a 3
#eval a 4
#eval a 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l581_58147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_100_zero_count_l581_58153

-- Define g₀
def g₀ (x : ℝ) : ℝ := x + |x - 150| - |x + 150|

-- Define gₙ recursively
def g (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 2

-- Theorem statement
theorem g_100_zero_count :
  ∃! (s : Set ℝ), (∀ x ∈ s, g 100 x = 0) ∧ (Finite s ∧ Nat.card s = 299) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_100_zero_count_l581_58153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_disjoint_in_lambda_system_l581_58148

/-- A λ-system on a set Ω is a collection of subsets of Ω satisfying certain properties. -/
class LambdaSystem (Ω : Type) (L : Set (Set Ω)) where
  contains_omega : Set.univ ∈ L
  complement_closed : ∀ A, A ∈ L → Aᶜ ∈ L
  union_disjoint_closed : ∀ A B, A ∈ L → B ∈ L → A ∩ B = ∅ → A ∪ B ∈ L

/-- If A and B are disjoint elements of a λ-system L, then their union is also in L. -/
theorem union_of_disjoint_in_lambda_system {Ω : Type} {L : Set (Set Ω)} [LambdaSystem Ω L]
  {A B : Set Ω} (hA : A ∈ L) (hB : B ∈ L) (hDisjoint : A ∩ B = ∅) :
  A ∪ B ∈ L := by
  apply LambdaSystem.union_disjoint_closed
  exact hA
  exact hB
  exact hDisjoint


end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_disjoint_in_lambda_system_l581_58148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l581_58118

-- Define set A
def A : Set ℝ := {x : ℝ | (1/2 : ℝ) ≤ (2 : ℝ)^(x+1) ∧ (2 : ℝ)^(x+1) ≤ 16}

-- Define set B
def B : Set ℝ := Set.Ioo (-1 : ℝ) 4

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioc (-1 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l581_58118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_floor_system_l581_58172

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- State the theorem
theorem unique_solution_floor_system :
  ∃! (x y : ℝ), 
    (floor (x + y - 3) = 2 - Int.floor x) ∧
    (floor (x + 1) + floor (y - 7) + Int.floor x = Int.floor y) ∧
    x = 3 ∧ y = -1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_floor_system_l581_58172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_k_representation_l581_58103

theorem base_k_representation (k : ℕ) (h1 : k > 0) : 
  (8 : ℚ) / 65 = (3 : ℚ) / k + (4 : ℚ) / k^2 + (3 : ℚ) / k^3 + (4 : ℚ) / k^4 + 
    (((3 : ℚ) / k + (4 : ℚ) / k^2) / (1 - 1 / k^2)) ↔ k = 17 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_k_representation_l581_58103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_x1_plus_x2_l581_58155

/-- The function f(x) = a*sin(x) - √3*cos(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x - Real.sqrt 3 * Real.cos x

/-- Theorem stating the minimum value of |x₁ + x₂| given the conditions -/
theorem min_value_x1_plus_x2 (a : ℝ) :
  (∃ x, f a x = f a (-π/6 - (x + π/6))) →  -- Axis of symmetry at x = -π/6
  (∃ x₁ x₂, f a x₁ * f a x₂ = -4) →        -- f(x₁) * f(x₂) = -4
  (∃ x₁ x₂, f a x₁ * f a x₂ = -4 ∧ |x₁ + x₂| = 2*π/3 ∧ 
    ∀ y₁ y₂, f a y₁ * f a y₂ = -4 → |y₁ + y₂| ≥ 2*π/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_x1_plus_x2_l581_58155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_in_third_quadrant_l581_58199

noncomputable def point_P : ℝ × ℝ := (Real.sin (2018 * Real.pi / 180), Real.cos (2018 * Real.pi / 180))

theorem point_P_in_third_quadrant :
  point_P.1 < 0 ∧ point_P.2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_in_third_quadrant_l581_58199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jog_walk_distance_ratio_l581_58131

-- Define the given conditions
noncomputable def total_time : ℝ := 21 / 60  -- Convert minutes to hours
noncomputable def walk_time : ℝ := 9 / 60    -- Convert minutes to hours
def walk_speed : ℝ := 4
def jog_speed : ℝ := 8

-- Define the distances
noncomputable def walk_distance : ℝ := walk_time * walk_speed
noncomputable def jog_distance : ℝ := (total_time - walk_time) * jog_speed

-- Theorem statement
theorem jog_walk_distance_ratio :
  jog_distance / walk_distance = 8 / 3 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jog_walk_distance_ratio_l581_58131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_64_over_11_l581_58125

/-- The area of the triangle bounded by the y-axis and two lines -/
noncomputable def triangle_area : ℝ :=
  let line1 : ℝ → ℝ := fun x ↦ 5 * x - 4
  let line2 : ℝ → ℝ := fun x ↦ (16 - 2 * x) / 4
  let intersection_x : ℝ := 16 / 11
  let base : ℝ := line2 0 - line1 0
  let height : ℝ := intersection_x
  (1 / 2) * base * height

/-- The theorem stating that the area of the triangle is 64/11 -/
theorem triangle_area_is_64_over_11 : triangle_area = 64 / 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_64_over_11_l581_58125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l581_58160

open BigOperators
open Polynomial

theorem coefficient_x_cubed_in_expansion : 
  let X : Polynomial ℤ := X
  let expansion := (X + 1) * (X - 2)^5
  coeff expansion 3 = -40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l581_58160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_greater_than_negative_one_l581_58181

-- Define the standard normal distribution
noncomputable def standard_normal (X : ℝ → ℝ) : Prop :=
  ∀ x, X x = (1 / Real.sqrt (2 * Real.pi)) * Real.exp (-(x^2) / 2)

-- Define the probability measure
noncomputable def prob (X : ℝ → ℝ) (event : ℝ → Prop) : ℝ :=
  ∫ x in {x | event x}, X x

-- Theorem statement
theorem prob_greater_than_negative_one 
  (X : ℝ → ℝ) 
  (h1 : standard_normal X) 
  (h2 : prob X (λ x => x > 1) = p) :
  prob X (λ x => x > -1) = 1 - p :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_greater_than_negative_one_l581_58181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_a_b_l581_58107

noncomputable def f (a b x : ℝ) : ℝ := -1/b * Real.exp (a * x)

theorem max_sum_a_b (a b : ℝ) : 
  a > 0 → 
  b > 0 → 
  (∃ (t : ℝ), (f a b t + 1/b = -a/b * t) ∧ 
              (t^2 + (f a b t)^2 = 1) ∧ 
              (∀ (x : ℝ), x ≠ t → x^2 + (f a b x)^2 > 1)) → 
  a + b ≤ Real.sqrt 2 :=
by
  sorry

#check max_sum_a_b

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_a_b_l581_58107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_race_problem_l581_58115

/-- The minimum distance between two points on opposite sides of a wall, passing through the wall --/
noncomputable def min_distance_through_wall (wall_length a_dist b_dist : ℝ) : ℝ :=
  Real.sqrt ((wall_length ^ 2) + ((a_dist + b_dist) ^ 2))

/-- Theorem stating the minimum distance for the given problem --/
theorem min_distance_race_problem :
  let wall_length := 1500
  let a_dist := 400
  let b_dist := 600
  let exact_distance := min_distance_through_wall wall_length a_dist b_dist
  ⌊exact_distance⌋₊ = 1803 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_race_problem_l581_58115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dichromate_molecular_weight_l581_58169

/-- Represents the molecular formula of a compound -/
structure MolecularFormula where
  chromium : Float
  oxygen : Float

/-- Calculates the molecular weight of a compound given its formula and atomic weights -/
def molecularWeight (formula : MolecularFormula) (crWeight oWeight : Float) : Float :=
  formula.chromium * crWeight + formula.oxygen * oWeight

/-- Theorem: The molecular weight of Dichromate is 216.00 g/mol -/
theorem dichromate_molecular_weight :
  let dichromate : MolecularFormula := { chromium := 2, oxygen := 7 }
  let crWeight : Float := 52.00
  let oWeight : Float := 16.00
  molecularWeight dichromate crWeight oWeight = 216.00 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dichromate_molecular_weight_l581_58169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_between_one_and_two_l581_58167

/-- A hyperbola with foci F and E, and points A and B on it. -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  F : ℝ × ℝ
  E : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_equation : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → (x, y) = A ∨ (x, y) = B
  h_F_left : F.1 < 0
  h_E_right : 0 < E.1
  h_F_on_axis : F.2 = 0
  h_E_on_axis : E.2 = 0
  h_A_on_F_perp : A.1 = F.1
  h_B_on_F_perp : B.1 = F.1
  h_EA_EB_pos : (E.1 - A.1) * (E.1 - B.1) + (E.2 - A.2) * (E.2 - B.2) > 0

/-- The eccentricity of a hyperbola. -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Theorem: The eccentricity of the hyperbola is between 1 and 2. -/
theorem eccentricity_between_one_and_two (h : Hyperbola) : 
  1 < eccentricity h ∧ eccentricity h < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_between_one_and_two_l581_58167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_point_l581_58166

/-- The distance from the origin (0, 0) to the point (-15, 8) in a rectangular coordinate system is 17 units. -/
theorem distance_to_point : ∃ d : ℝ, d = 17 ∧ d = Real.sqrt ((-15)^2 + 8^2) := by
  -- Define the point
  let x : ℝ := -15
  let y : ℝ := 8
  
  -- Define the distance
  let d : ℝ := Real.sqrt (x^2 + y^2)
  
  -- Prove that d exists and equals 17
  use d
  constructor
  · -- Prove d = 17
    sorry
  · -- Prove d = Real.sqrt ((-15)^2 + 8^2)
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_point_l581_58166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_views_l581_58161

/-- A frustum is a portion of a solid (usually a cone or pyramid) lying between two parallel planes cutting the solid. -/
structure Frustum where
  /-- The base radius of the frustum -/
  base_radius : ℝ
  /-- The top radius of the frustum -/
  top_radius : ℝ
  /-- The height of the frustum -/
  height : ℝ
  /-- Assertion that the base radius is greater than the top radius -/
  h_radius : base_radius > top_radius
  /-- Assertion that all dimensions are positive -/
  h_positive : base_radius > 0 ∧ top_radius > 0 ∧ height > 0

/-- The front view of a frustum is an isosceles trapezoid -/
def front_view_is_isosceles_trapezoid (f : Frustum) : Prop :=
  sorry

/-- The side view of a frustum is an isosceles trapezoid -/
def side_view_is_isosceles_trapezoid (f : Frustum) : Prop :=
  sorry

/-- The top view of a frustum is two concentric circles -/
def top_view_is_concentric_circles (f : Frustum) : Prop :=
  sorry

/-- The front and side views of a frustum are congruent -/
def front_side_views_congruent (f : Frustum) : Prop :=
  sorry

theorem frustum_views (f : Frustum) :
    front_view_is_isosceles_trapezoid f ∧
    side_view_is_isosceles_trapezoid f ∧
    front_side_views_congruent f ∧
    top_view_is_concentric_circles f := by
  sorry

#check frustum_views

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_views_l581_58161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2009_equals_negative_6_l581_58187

def sequence_a : ℕ → ℤ
  | 0 => 3  -- We define a_0 as 3 to handle the base case
  | 1 => 6
  | (n + 2) => sequence_a (n + 1) - sequence_a n

theorem a_2009_equals_negative_6 : sequence_a 2009 = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2009_equals_negative_6_l581_58187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_stops_in_quarter_x_l581_58109

/-- Represents the quarters of the circular track -/
inductive Quarter
  | X
  | Y
  | Z
  | W

/-- Represents a point on the circular track -/
structure TrackPoint where
  distance : ℚ  -- Distance from the starting point T (using rationals instead of reals)
  quarter : Quarter

/-- The circular track -/
structure Track where
  circumference : ℚ
  start : TrackPoint

/-- Calculates the final position after running a given distance -/
def final_position (track : Track) (run_distance : ℚ) : TrackPoint :=
  { distance := run_distance % track.circumference,
    quarter := Quarter.X }  -- Placeholder, actual calculation would be more complex

theorem runner_stops_in_quarter_x (track : Track) (run_distance : ℚ) :
  track.circumference = 60 →
  track.start.distance = 0 →
  track.start.quarter = Quarter.X →
  run_distance = 7920 →
  (final_position track run_distance).quarter = Quarter.X :=
by
  intro h_circ h_start_dist h_start_quarter h_run_dist
  sorry  -- Placeholder for the actual proof

#eval 7920 % 60  -- Should evaluate to 0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_stops_in_quarter_x_l581_58109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_side_lengths_l581_58185

/-- A convex hexagon with specific side length properties -/
structure ConvexHexagon where
  sides : Fin 6 → ℕ
  convex : Bool
  distinct_lengths : Finset.card (Finset.image sides Finset.univ) = 3
  two_seven_sides : (sides 0 = 7 ∧ sides 1 = 7) ∨ (sides 0 = 7 ∧ sides 2 = 7) ∨ (sides 1 = 7 ∧ sides 2 = 7)
  one_eight_side : ∃ i, sides i = 8
  perimeter : (sides 0) + (sides 1) + (sides 2) + (sides 3) + (sides 4) + (sides 5) = 45

/-- The theorem stating the properties of the hexagon's side lengths -/
theorem hexagon_side_lengths (h : ConvexHexagon) :
  ∃ (i j k l m n : Fin 6), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ m ∧ m ≠ n ∧ n ≠ i ∧
    Multiset.ofList [h.sides i, h.sides j, h.sides k, h.sides l, h.sides m, h.sides n] =
    Multiset.ofList [6, 7, 7, 8, 8, 9] :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_side_lengths_l581_58185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_logs_equals_two_l581_58111

def f (x : ℝ) : ℝ := 4 * x^5 + 3 * x^3 + 2 * x + 1

theorem f_sum_logs_equals_two :
  f (Real.log 3 / Real.log 2) + f (Real.log 3 / Real.log (1/2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_logs_equals_two_l581_58111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_study_time_difference_l581_58144

/-- The daily differences in study time (in minutes) between Sasha and Asha over a week. -/
def daily_differences : List Int := [15, -5, 25, -10, 15, 5, 20]

/-- The number of days in the week. -/
def num_days : Nat := daily_differences.length

/-- The average difference in study time per day. -/
def average_difference : ℚ := (daily_differences.sum : ℚ) / num_days

theorem average_study_time_difference :
  Int.natAbs (Int.floor average_difference) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_study_time_difference_l581_58144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_resistors_theorem_l581_58119

/-- The combined resistance of three parallel resistors -/
noncomputable def combined_resistance (r1 r2 r3 : ℝ) : ℝ :=
  1 / (1/r1 + 1/r2 + 1/r3)

/-- Theorem stating that given the resistances of two resistors and the combined resistance of three parallel resistors, the resistance of the third resistor is 2 ohms -/
theorem parallel_resistors_theorem (r2 r3 r_total : ℝ) 
  (h1 : r2 = 5)
  (h2 : r3 = 6)
  (h3 : r_total = 0.8666666666666666)
  : ∃ r1 : ℝ, r1 = 2 ∧ combined_resistance r1 r2 r3 = r_total :=
by
  sorry

#check parallel_resistors_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_resistors_theorem_l581_58119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_product_of_distances_l581_58143

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents an angle -/
structure Angle where
  vertex : Point
  side1 : Line
  side2 : Line

/-- Represents a ray -/
structure Ray where
  origin : Point
  direction : Point

/-- Function to calculate the distance from a point to a line -/
noncomputable def distancePointToLine (p : Point) (l : Line) : ℝ := 
  sorry

/-- Function to check if two angles are equal -/
def anglesEqual (a1 : Angle) (a2 : Angle) : Prop := 
  sorry

/-- Function to check if a point lies on a ray -/
def pointOnRay (p : Point) (r : Ray) : Prop := 
  sorry

/-- Function to construct an angle from a vertex and two rays -/
def angleFromRays (v : Point) (r1 r2 : Ray) : Angle :=
  sorry

/-- Theorem statement -/
theorem equal_product_of_distances
  (A O B : Point)
  (angleAOB : Angle)
  (M N : Point)
  (rayOM rayON : Ray)
  (P Q : Point)
  (h1 : angleAOB.vertex = O)
  (h2 : anglesEqual (angleFromRays O rayOM rayON) angleAOB)
  (h3 : pointOnRay P rayOM)
  (h4 : pointOnRay Q rayON) :
  (distancePointToLine P angleAOB.side1) * (distancePointToLine Q angleAOB.side2) =
  (distancePointToLine P angleAOB.side2) * (distancePointToLine Q angleAOB.side1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_product_of_distances_l581_58143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_correct_l581_58100

/-- The function f(x) = x^2 + 5x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 5*x + c

/-- 1 is in the range of f -/
def one_in_range (c : ℝ) : Prop := ∃ x, f c x = 1

/-- The smallest value of c such that 1 is in the range of f -/
noncomputable def smallest_c : ℝ := 29/4

theorem smallest_c_correct :
  (∀ c < smallest_c, ¬(one_in_range c)) ∧ (one_in_range smallest_c) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_correct_l581_58100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_quadrilateral_type_l581_58163

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  a : Point
  b : Point
  c : Point
  d : Point

/-- Defines the parallelogram EFGH -/
def parallelogram : Quadrilateral where
  a := ⟨0, 0⟩  -- E
  b := ⟨0, 5⟩  -- F
  c := ⟨7, 5⟩  -- G
  d := ⟨7, 0⟩  -- H

/-- Creates a line from a point and an angle -/
noncomputable def createLine (p : Point) (angle : ℝ) : Line :=
  ⟨Real.tan angle, p.y - p.x * Real.tan angle⟩

/-- Finds the intersection point of two lines -/
noncomputable def intersectionPoint (l1 l2 : Line) : Point :=
  let x := (l2.intercept - l1.intercept) / (l1.slope - l2.slope)
  let y := l1.slope * x + l1.intercept
  ⟨x, y⟩

/-- Determines the type of quadrilateral based on its properties -/
def determineQuadrilateralType (q : Quadrilateral) : String :=
  sorry  -- The actual determination logic would go here

/-- Main theorem: The quadrilateral formed by intersecting lines has specific properties -/
theorem intersection_quadrilateral_type :
  let e_line_45 := createLine parallelogram.a (45 * π / 180)
  let f_line_75 := createLine parallelogram.b (75 * π / 180)
  let g_line_neg45 := createLine parallelogram.c (-45 * π / 180)
  let h_line_neg75 := createLine parallelogram.d (-75 * π / 180)
  let intersection_quad := Quadrilateral.mk
    (intersectionPoint e_line_45 g_line_neg45)
    (intersectionPoint e_line_45 h_line_neg75)
    (intersectionPoint f_line_75 g_line_neg45)
    (intersectionPoint f_line_75 h_line_neg75)
  determineQuadrilateralType intersection_quad = "Square" ∨
  determineQuadrilateralType intersection_quad = "Rectangle" ∨
  determineQuadrilateralType intersection_quad = "Rhombus" ∨
  determineQuadrilateralType intersection_quad = "Parallelogram" :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_quadrilateral_type_l581_58163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_solution_l581_58150

/-- Represents the compound interest problem -/
structure CompoundInterest where
  P : ℝ  -- Principal amount
  r : ℝ  -- Interest rate (in decimal form)
  A₂ : ℝ := P * (1 + r)^2  -- Amount after 2 years
  A₃ : ℝ := P * (1 + r)^3  -- Amount after 3 years

/-- The compound interest problem with given conditions -/
def problem : CompoundInterest → Prop := fun ci =>
  ci.A₂ = 2442 ∧ ci.A₃ = 2926

/-- The theorem stating that the interest rate is approximately 19.82% -/
theorem interest_rate_solution (ci : CompoundInterest) (h : problem ci) :
  abs (ci.r - 0.1982) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_solution_l581_58150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalized_form_of_7_over_3_plus_sqrt_8_l581_58156

noncomputable def rationalize_denominator (x : ℚ) (y : ℝ) : ℝ := x / (3 + y)

def is_not_divisible_by_square_of_prime (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ n)

theorem rationalized_form_of_7_over_3_plus_sqrt_8 :
  ∃ (A C D : ℤ) (B : ℕ),
    rationalize_denominator 7 (Real.sqrt 8) = (A * Real.sqrt (B : ℝ) + C) / D ∧
    D > 0 ∧
    is_not_divisible_by_square_of_prime B ∧
    Int.gcd A (Int.gcd C D) = 1 ∧
    A = -14 ∧
    B = 2 ∧
    C = 21 ∧
    D = 1 :=
by
  sorry

#eval (-14 : ℤ) + (2 : ℤ) + (21 : ℤ) + (1 : ℤ)  -- Should output 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalized_form_of_7_over_3_plus_sqrt_8_l581_58156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l581_58165

/-- Helper function to represent the area of a triangle given its side lengths. -/
def is_area_of_triangle (a b c S : ℝ) : Prop :=
  ∃ (s : ℝ), s = (a + b + c)/2 ∧ S = Real.sqrt (s*(s-a)*(s-b)*(s-c))

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and the condition a^2 + 2(b^2 + c^2) = 2√2, the maximum area of triangle ABC is 1/4. -/
theorem max_triangle_area (a b c : ℝ) (h : a^2 + 2*(b^2 + c^2) = 2*Real.sqrt 2) :
  ∃ (S : ℝ), S ≤ 1/4 ∧ ∀ (S' : ℝ), is_area_of_triangle a b c S' → S' ≤ S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l581_58165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_and_b_is_45_degrees_l581_58142

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![3, 1]

-- Define the dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Define the magnitude of a vector
noncomputable def magnitude (v : Fin 2 → ℝ) : ℝ := Real.sqrt ((v 0)^2 + (v 1)^2)

-- Define the angle between two vectors
noncomputable def angle_between (v w : Fin 2 → ℝ) : ℝ := 
  Real.arccos ((dot_product v w) / (magnitude v * magnitude w))

-- Theorem statement
theorem angle_between_a_and_b_is_45_degrees :
  angle_between a b = π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_and_b_is_45_degrees_l581_58142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_height_l581_58122

/-- Given a rectangle with length 12 dm and width 6 dm, and a parallelogram with the same area and base 12 dm, the height of the parallelogram is 6 dm. -/
theorem parallelogram_height (rect_length rect_width parallelogram_base parallelogram_height : ℝ) 
  (h1 : rect_length = 12)
  (h2 : rect_width = 6)
  (h3 : parallelogram_base = 12)
  (h4 : rect_length * rect_width = parallelogram_base * parallelogram_height) :
  parallelogram_height = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_height_l581_58122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l581_58193

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 - Real.sqrt 3 * Real.sin x * Real.cos x + 1/2

structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : 0 < A ∧ A < π
  h2 : 0 < B ∧ B < π
  h3 : 0 < C ∧ C < π
  h4 : A + B + C = π

theorem triangle_area (t : Triangle) 
  (h1 : f (t.B + t.C) = 3/2)
  (h2 : t.a = Real.sqrt 3)
  (h3 : t.b + t.c = 3) :
  (1/2 : ℝ) * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l581_58193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l581_58130

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 12))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 15/2 ∨ x > 15/2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l581_58130
