import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l641_64181

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 4*x else -x^2 - 4*x

-- Theorem statement
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧ 
  (Set.Ioo (-5 : ℝ) 0 ∪ Set.Ioi 5 = {x | f x > x}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l641_64181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l641_64111

-- Define the max function
noncomputable def max' (a b : ℝ) : ℝ := if a ≥ b then a else b

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := max' (2 * x + 1) (5 - x)

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ (m = 11/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l641_64111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_assistant_age_is_65_l641_64162

/-- Represents the class with students and teachers -/
structure MyClass where
  initial_students : Nat
  initial_avg_age : Nat
  leaving_students_ages : List Nat
  joining_students_ages : List Nat
  final_avg_age : Nat

/-- Calculates the combined age of the teacher and assistant -/
def teacher_assistant_age (c : MyClass) : Nat :=
  let initial_total_age := c.initial_students * c.initial_avg_age
  let leaving_total_age := c.leaving_students_ages.sum
  let joining_total_age := c.joining_students_ages.sum
  let new_student_total_age := initial_total_age - leaving_total_age + joining_total_age
  let final_total_age := (c.initial_students - c.leaving_students_ages.length + c.joining_students_ages.length + 2) * c.final_avg_age
  final_total_age - new_student_total_age

/-- Theorem stating the combined age of the teacher and assistant -/
theorem teacher_assistant_age_is_65 (c : MyClass) 
  (h1 : c.initial_students = 30)
  (h2 : c.initial_avg_age = 10)
  (h3 : c.leaving_students_ages = [11, 12, 13])
  (h4 : c.joining_students_ages = [9, 14])
  (h5 : c.final_avg_age = 11) :
  teacher_assistant_age c = 65 := by
  sorry

#eval teacher_assistant_age {
  initial_students := 30,
  initial_avg_age := 10,
  leaving_students_ages := [11, 12, 13],
  joining_students_ages := [9, 14],
  final_avg_age := 11
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_assistant_age_is_65_l641_64162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_zeros_and_symmetry_l641_64186

open Real

theorem cos_zeros_and_symmetry (ω φ : ℝ) : 
  ω > 0 ∧ 
  (cos (ω * (-π/6) + φ) = 0) ∧ 
  (cos (ω * (5*π/6) + φ) = 0) ∧
  (∃! (a b : ℝ), -π/6 < a ∧ a < b ∧ b < 5*π/6 ∧ 
    (∀ x, -π/6 ≤ x ∧ x ≤ 5*π/6 → 
      cos (ω * x + φ) = cos (ω * (a + b - x) + φ)) ∧
    (∀ x, -π/6 ≤ x ∧ x ≤ 5*π/6 → 
      cos (ω * x + φ) = cos (ω * (a - x + b) + φ)))
  → ω * φ = 5*π/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_zeros_and_symmetry_l641_64186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_proof_l641_64173

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 + 8*x + 9*y^2 - 18*y + 49 = 0

/-- The area of the ellipse -/
noncomputable def ellipse_area : ℝ := 25 * Real.pi / 3

/-- Theorem: The area of the ellipse defined by the equation
    x^2 + 8x + 9y^2 - 18y + 49 = 0 is equal to 25π/3 -/
theorem ellipse_area_proof :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), ellipse_equation x y ↔ (x+4)^2/a^2 + (y-1)^2/b^2 = 1) ∧
  ellipse_area = Real.pi * a * b := by
  sorry

#check ellipse_area_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_proof_l641_64173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l641_64174

noncomputable def f (x : ℝ) := Real.log (1 + 1/x) + Real.sqrt (1 - x^2)

theorem domain_of_f :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ (0 < x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l641_64174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l641_64133

/-- Triangle with special properties --/
structure SpecialTriangle where
  A : ℝ × ℝ  -- Vertex A of the triangle
  B : ℝ × ℝ  -- Vertex B of the triangle
  C : ℝ × ℝ  -- Vertex C of the triangle
  C1 : ℝ × ℝ     -- Projection of C on AB
  mc : ℝ         -- Radius of the circular sector
  sector_area_half : ℝ -- Area of the circular sector is half the triangle area
  right_angle : Bool   -- Indicates if there's a right angle
  two_equal_angles : Bool -- Indicates if two angles are equal

/-- Angles of the triangle --/
noncomputable def triangle_angles (t : SpecialTriangle) : ℝ × ℝ × ℝ :=
  sorry

/-- Length of the area-bisecting circular arc --/
noncomputable def arc_length (t : SpecialTriangle) : ℝ :=
  sorry

/-- Length of the area-bisecting straight line segment parallel to AB --/
noncomputable def parallel_segment_length (t : SpecialTriangle) : ℝ :=
  sorry

/-- Main theorem --/
theorem special_triangle_properties (t : SpecialTriangle) :
  (triangle_angles t = (19.7667, 70.2333, 90) ∨
   triangle_angles t = (90, 23.2167, 66.7833) ∨
   triangle_angles t = (23.2167, 23.2167, 133.5667)) ∧
  arc_length t < parallel_segment_length t :=
by
  sorry

#check special_triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l641_64133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_escalator_ascent_time_l641_64127

/-- Represents the time taken for a person to ascend an escalator -/
noncomputable def escalator_ascent_time (escalator_speed : ℝ) (escalator_length : ℝ) 
  (initial_walk_speed : ℝ) (final_walk_speed : ℝ) : ℝ :=
  escalator_length / (escalator_speed + (initial_walk_speed + final_walk_speed) / 2)

/-- Theorem stating the time taken for a specific escalator ascent scenario -/
theorem specific_escalator_ascent_time :
  escalator_ascent_time 20 300 3 5 = 12.5 := by
  -- Unfold the definition of escalator_ascent_time
  unfold escalator_ascent_time
  -- Simplify the expression
  simp
  -- Perform the numerical calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_escalator_ascent_time_l641_64127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l641_64152

/-- Given a curve y = x^4 and a line x + 2y - 8 = 0, 
    this theorem states that any tangent line to the curve 
    that is parallel to the given line has the equation 8x + 16y + 3 = 0 -/
theorem tangent_line_equation : 
  let curve := fun (x : ℝ) => x^4
  let given_line := fun (x y : ℝ) => x + 2*y - 8 = 0
  let tangent_line := fun (x y : ℝ) => 8*x + 16*y + 3 = 0
  ∀ x₀ y₀ : ℝ, y₀ = curve x₀ → 
    (∃ k : ℝ, k ≠ 0 ∧ (deriv curve) x₀ = k * (-1/2)) →
    tangent_line x₀ y₀ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l641_64152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l641_64104

-- Define the parabola
def is_on_parabola (x y : ℝ) : Prop := x^2 = 16*y

-- Define the focus
def focus : ℝ × ℝ := (0, 4)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the area of a triangle given two points
noncomputable def triangle_area (p1 p2 : ℝ × ℝ) : ℝ :=
  (1/2) * abs (p1.1 * p2.2 - p2.1 * p1.2)

-- The main theorem
theorem parabola_triangle_area (x y : ℝ) :
  is_on_parabola x y →
  distance (x, y) focus = 8 →
  triangle_area (x, y) focus = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l641_64104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_of_parallel_sides_l641_64183

/-- Represents a polygon with n sides inscribed in a circle -/
structure InscribedPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Check if two line segments are parallel -/
def are_parallel (a b c d : ℝ × ℝ) : Prop :=
  (b.1 - a.1) * (d.2 - c.2) = (b.2 - a.2) * (d.1 - c.1)

/-- The ratio of lengths of two line segments -/
noncomputable def length_ratio (a b c d : ℝ × ℝ) : ℝ :=
  Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2) / Real.sqrt ((d.1 - c.1)^2 + (d.2 - c.2)^2)

theorem constant_ratio_of_parallel_sides 
  (A B : InscribedPolygon 93) 
  (h : ∀ i : Fin 93, are_parallel 
    (A.vertices i) (A.vertices (i.succ)) 
    (B.vertices i) (B.vertices (i.succ))) :
  ∃ c : ℝ, ∀ i : Fin 93, 
    length_ratio 
      (A.vertices i) (A.vertices (i.succ)) 
      (B.vertices i) (B.vertices (i.succ)) = c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_of_parallel_sides_l641_64183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_values_theorem_l641_64140

-- Define the given condition
def x_condition (x : ℝ) : Prop :=
  x > Real.pi / 2 ∧ x < 3 * Real.pi / 4

-- State the theorem
theorem sin_values_theorem (x : ℝ) 
  (h1 : x_condition x) 
  (h2 : Real.cos (x - Real.pi / 4) = Real.sqrt 2 / 10) : 
  Real.sin x = 4 / 5 ∧ 
  Real.sin (2 * x + Real.pi / 6) = -(7 + 24 * Real.sqrt 3) / 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_values_theorem_l641_64140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l641_64146

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 - Real.cos x ^ 2) / (Real.sin x * Real.tan x)

-- State the theorem
theorem f_properties :
  -- Part 1: Domain of f
  (∀ x : ℝ, f x ≠ 0 ↔ ∀ k : ℤ, x ≠ Real.pi / 2 * (k : ℝ)) ∧
  -- Part 2: Specific value
  (∀ θ : ℝ, f θ = -Real.sqrt 5 / 5 → θ > Real.pi → θ < 3 * Real.pi / 2 → Real.tan (Real.pi - θ) = -2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l641_64146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_length_l641_64135

theorem right_triangle_side_length (Q P R : ℝ×ℝ) (cosQ : ℝ) :
  cosQ = 3 / 5 →
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) = 15 →
  (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0 →
  Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_length_l641_64135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_properties_l641_64103

-- Define the set S of all non-zero real numbers
def S : Set ℝ := {x : ℝ | x ≠ 0}

-- Define the binary operation *
def star (a b : ℝ) : ℝ := 2 * a * b

-- Theorem stating the properties of the operation
theorem star_properties :
  (∀ (a b : ℝ), a ∈ S → b ∈ S → star a b = star b a) ∧ 
  (∃ (a b c : ℝ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ star (star a b) c ≠ star a (star b c)) ∧
  (∀ (a : ℝ), a ∈ S → star a (1/2) = a ∧ star (1/2) a = a) ∧
  (∃ (a : ℝ), a ∈ S ∧ ∀ (b : ℝ), b ∈ S → star a b ≠ 1/2 ∨ star b a ≠ 1/2) ∧
  (∀ (a : ℝ), a ∈ S → star a (1/(2*a)) = 1/2 ∧ star (1/(2*a)) a = 1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_properties_l641_64103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_to_vertices_l641_64150

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A square in a 2D plane -/
structure Square where
  center : Point
  side_length : ℝ

/-- Check if a point is on the boundary of a square -/
def on_boundary (p : Point) (s : Square) : Prop :=
  let half_side := s.side_length / 2
  (abs (p.x - s.center.x) = half_side ∧ abs (p.y - s.center.y) ≤ half_side) ∨
  (abs (p.y - s.center.y) = half_side ∧ abs (p.x - s.center.x) ≤ half_side)

/-- Sum of distances from a point to the vertices of a square -/
noncomputable def sum_distances_to_vertices (p : Point) (s : Square) : ℝ :=
  let half_side := s.side_length / 2
  let v1 := Point.mk (s.center.x - half_side) (s.center.y - half_side)
  let v2 := Point.mk (s.center.x + half_side) (s.center.y - half_side)
  let v3 := Point.mk (s.center.x + half_side) (s.center.y + half_side)
  let v4 := Point.mk (s.center.x - half_side) (s.center.y + half_side)
  distance p v1 + distance p v2 + distance p v3 + distance p v4

theorem min_sum_distances_to_vertices
  (A B : Point)
  (d : ℝ)
  (h_distance : distance A B = d)
  (h_positive : d > 0) :
  ∀ s : Square,
    on_boundary A s → on_boundary B s →
    sum_distances_to_vertices A s ≥ (1 + Real.sqrt 2) * d :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_to_vertices_l641_64150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_func_increasing_implies_a_negative_l641_64143

-- Define the exponential function as noncomputable
noncomputable def exp_func (a : ℝ) (x : ℝ) : ℝ := (1 - a) ^ x

-- State the theorem
theorem exp_func_increasing_implies_a_negative (a : ℝ) :
  (∀ x y : ℝ, x < y → exp_func a x < exp_func a y) → a < 0 :=
by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_func_increasing_implies_a_negative_l641_64143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_m_l641_64113

/-- A power function of the form (a * x^b) where a and b are constants -/
def PowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x^b

/-- A function is increasing on an interval -/
def IncreasingOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x < f y

/-- The open interval (0, +∞) -/
def OpenPosReals : Set ℝ := {x : ℝ | x > 0}

theorem power_function_increasing_m (m : ℝ) :
  let f := fun x : ℝ ↦ (m^2 - m - 5) * x^(m-1)
  PowerFunction f ∧ IncreasingOn f OpenPosReals → m = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_m_l641_64113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_special_expansion_l641_64108

/-- 
Given a binomial expansion (√x + 2/x²)¹⁰ where the sixth term's 
binomial coefficient is the maximum, the constant term is 180.
-/
theorem constant_term_of_special_expansion : 
  let n : ℕ := 10
  let expansion := fun (x : ℝ) => (Real.sqrt x + 2 / x^2)^n
  let sixth_term_max := ∀ k, 0 ≤ k ∧ k ≤ n → Nat.choose n 5 ≥ Nat.choose n k
  let constant_term : ℝ := (Nat.choose n 2) * 2^2
  sixth_term_max → constant_term = 180 := by
  intro h
  -- The proof goes here
  sorry

#check constant_term_of_special_expansion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_special_expansion_l641_64108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_proof_l641_64142

/-- Simple interest calculation -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_rate_proof (principal time interest : ℝ) 
  (h1 : principal = 800)
  (h2 : time = 5)
  (h3 : interest = 160)
  (h4 : simple_interest principal (4 : ℝ) time = interest) : 
  simple_interest principal (4 : ℝ) time = interest := by
  sorry

#check interest_rate_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_proof_l641_64142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_necessary_condition_for_parallel_lines_l641_64109

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V]

-- Define lines and planes
variable (m n : Submodule ℝ V) (α β : Subspace ℝ V)

-- Define the conditions
variable (h_m_in_α : m ≤ α)

-- Theorem statement
theorem not_necessary_condition_for_parallel_lines :
  ¬(∀ (n : Submodule ℝ V), m = n → n ≤ α) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_necessary_condition_for_parallel_lines_l641_64109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_work_time_l641_64117

/-- Work rates and total work -/
def W : ℝ := 1 -- Total work (normalized to 1)
variable (Wp Wq Wr : ℝ) -- Work rates of p, q, and r respectively

/-- p can do the work in the same time as q and r together -/
axiom h1 : Wp = Wq + Wr

/-- p and q together can complete the work in 10 days -/
axiom h2 : Wp + Wq = W / 10

/-- r alone needs 50 days to complete the work -/
axiom h3 : Wr = W / 50

/-- Theorem: q alone can complete the work in 25 days -/
theorem q_work_time : Wq = W / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_work_time_l641_64117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_male_students_25_or_older_l641_64168

theorem percentage_of_male_students_25_or_older 
  (total_students : ℝ) 
  (male_percentage : ℝ) 
  (female_25_or_older_percentage : ℝ) 
  (prob_less_than_25 : ℝ) 
  (h1 : male_percentage = 0.4)
  (h2 : female_25_or_older_percentage = 0.3)
  (h3 : prob_less_than_25 = 0.66) :
  let female_percentage := 1 - male_percentage
  let male_25_or_older_percentage := 1 - (prob_less_than_25 - (1 - female_25_or_older_percentage) * female_percentage) / male_percentage
  male_25_or_older_percentage = 0.4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_male_students_25_or_older_l641_64168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_seven_is_half_l641_64121

open Real

/-- The decimal representation of 8/11 -/
noncomputable def decimal_rep : ℝ := 8 / 11

/-- The length of the repeating part in the decimal representation -/
def period_length : ℕ := 2

/-- The count of 7's in one period of the decimal representation -/
def count_of_sevens : ℕ := 1

/-- The probability of selecting a 7 from the decimal representation of 8/11 -/
def probability_of_seven : ℚ := count_of_sevens / period_length

theorem probability_of_seven_is_half : probability_of_seven = 1/2 := by
  rfl

#eval probability_of_seven

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_seven_is_half_l641_64121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l641_64114

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 + t, 1 + t)

-- Define the circle C in polar coordinates
noncomputable def circle_C (θ : ℝ) : ℝ := 4 * Real.sqrt 2 * Real.sin (θ + Real.pi/4)

-- Define the circle C in Cartesian coordinates
def circle_C_cartesian (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y = 0

-- Define point P
def point_P : ℝ × ℝ := (2, 1)

-- Theorem statement
theorem intersection_distance_difference :
  ∃ (A B : ℝ × ℝ),
    (∃ t : ℝ, line_l t = A) ∧
    (∃ t : ℝ, line_l t = B) ∧
    circle_C_cartesian A.1 A.2 ∧
    circle_C_cartesian B.1 B.2 ∧
    |Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) -
     Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2)| = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l641_64114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_superinvariant_sets_l641_64129

-- Define a superinvariant set
def IsSuperinvariant (S : Set ℝ) : Prop :=
  ∀ (x₀ a b : ℝ) (ha : a > 0),
    (∀ x ∈ S, ∃ y ∈ S, x₀ + a * (x - x₀) = y + b) ∧
    (∀ t ∈ S, ∃ u ∈ S, t + b = x₀ + a * (u - x₀))

-- Define the set of all superinvariant sets
def SuperinvariantSets : Set (Set ℝ) :=
  {S | IsSuperinvariant S}

-- Theorem stating the characterization of superinvariant sets
theorem characterization_of_superinvariant_sets :
  SuperinvariantSets = {Set.univ} ∪ 
    {S | ∃ x₀ : ℝ, S = {x₀} ∨ S = Set.univ \ {x₀} ∨ 
      S = Set.Iic x₀ ∨ S = Set.Ici x₀ ∨
      S = Set.Iio x₀ ∨ S = Set.Ioi x₀} :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_superinvariant_sets_l641_64129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_album_cost_is_20_l641_64151

-- Define the costs of the book, CD, and album
noncomputable def book_cost : ℝ := 18
noncomputable def cd_cost : ℝ := book_cost - 4
noncomputable def album_cost : ℝ := cd_cost / 0.7

-- Theorem statement
theorem album_cost_is_20 :
  album_cost = 20 :=
by
  -- Unfold the definitions
  unfold album_cost cd_cost book_cost
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_album_cost_is_20_l641_64151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_is_fifteen_l641_64138

def a : ℕ → ℕ
| 0 => 1  -- a₁ = 1 (we use 0-based indexing here)
| n + 1 => 2 * a n + 1  -- aₙ₊₁ = 2aₙ + 1

theorem fourth_term_is_fifteen : a 3 = 15 := by
  -- Unfold the definition of a for the first few terms
  have a0 : a 0 = 1 := rfl
  have a1 : a 1 = 3 := by simp [a]
  have a2 : a 2 = 7 := by simp [a, a1]
  have a3 : a 3 = 15 := by simp [a, a2]
  -- The goal follows directly from a3
  exact a3

#eval a 3  -- This will evaluate a 3 and print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_is_fifteen_l641_64138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_for_linear_function_l641_64194

/-- A function f(x) is linear if it can be written as f(x) = ax + b for some constants a and b, where a ≠ 0 --/
def IsLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The given function --/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * (x ^ (m^2 - 3)) + m

/-- The theorem statement --/
theorem unique_m_for_linear_function :
  ∃! m : ℝ, IsLinearFunction (f m) ∧ m - 2 ≠ 0 ∧ m = -2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_for_linear_function_l641_64194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_grocery_spending_l641_64156

/-- Represents the cost of groceries --/
structure GroceryCost where
  apples : ℚ
  bread : ℚ
  cereal : ℚ
  cheese : ℚ

/-- Calculates the total cost before discounts --/
def totalCost (g : GroceryCost) : ℚ :=
  g.apples + g.bread + g.cereal + g.cheese

/-- Applies bakery discount to bread and cheese --/
def applyBakeryDiscount (g : GroceryCost) (discount : ℚ) : GroceryCost :=
  { g with
    bread := g.bread * (1 - discount)
    cheese := g.cheese * (1 - discount)
  }

/-- Applies coupon if total is above the threshold --/
def applyCoupon (total : ℚ) (couponValue : ℚ) (threshold : ℚ) : ℚ :=
  if total ≥ threshold then total - couponValue else total

theorem tom_grocery_spending :
  let initialCost : GroceryCost := {
    apples := 4 * 2
    bread := 2 * 3
    cereal := 3 * 5
    cheese := 1 * 6
  }
  let bakeryDiscountRate := 1/4
  let couponValue := 10
  let couponThreshold := 30

  let discountedCost := applyBakeryDiscount initialCost bakeryDiscountRate
  let totalAfterDiscount := totalCost discountedCost
  let finalCost := applyCoupon totalAfterDiscount couponValue couponThreshold

  finalCost = 22 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_grocery_spending_l641_64156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_simplification_l641_64148

noncomputable def complex_expression (a b : ℝ) : ℝ :=
  4 * a * Real.sqrt ((a - Real.sqrt b) * (a^2 - b^2) / (a^2 - b)) *
  ((Real.sqrt ((a + Real.sqrt (a^2 - b)) / 2) + Real.sqrt ((a - Real.sqrt (a^2 - b)) / 2)) /
  ((a + Real.sqrt ((a - b) / (a - b)))^2 - (a - Real.sqrt ((a - b) / (a + b)))^2))

theorem complex_expression_simplification (a b : ℝ) (h : a ≠ b ∧ a > 0 ∧ b > 0) :
  complex_expression a b = a + b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_simplification_l641_64148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_larger_x_l641_64165

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  (x - 3)^2 / 5^2 - (y - 17)^2 / 12^2 = 1

/-- The center of the hyperbola -/
def center : ℝ × ℝ := (3, 17)

/-- The distance from the center to each focus -/
noncomputable def c : ℝ := Real.sqrt (5^2 + 12^2)

/-- The focus with the larger x-coordinate -/
noncomputable def focus_larger_x : ℝ × ℝ := (center.1 + c, center.2)

theorem hyperbola_focus_larger_x :
  focus_larger_x = (16, 17) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_larger_x_l641_64165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_traffic_flow_l641_64166

/-- Traffic speed as a function of traffic density -/
noncomputable def v (x : ℝ) : ℝ :=
  if x ≤ 20 then 60
  else if x ≤ 200 then -1/3 * x + 200/3
  else 0

/-- Traffic flow as a function of traffic density -/
noncomputable def f (x : ℝ) : ℝ := x * v x

/-- The maximum traffic flow occurs at density 100 and is approximately 3333 -/
theorem max_traffic_flow :
  ∃ (x_max : ℝ), x_max = 100 ∧
  ∀ (x : ℝ), 0 ≤ x → x ≤ 200 → f x ≤ f x_max ∧
  ⌊f x_max⌋ = 3333 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_traffic_flow_l641_64166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangement_count_l641_64139

/-- The number of ways to arrange 6 students in a line for a photo, 
    where two specific students must stand next to each other. -/
def photo_arrangements : ℕ := 60

/-- The total number of students -/
def total_students : ℕ := 6

/-- The number of students that must stand next to each other -/
def adjacent_students : ℕ := 2

/-- The number of remaining students after considering the adjacent pair as one unit -/
def remaining_students : ℕ := total_students - adjacent_students + 1

theorem photo_arrangement_count : 
  photo_arrangements = (Nat.factorial (remaining_students - 1)) * remaining_students := by
  sorry

#eval photo_arrangements
#eval (Nat.factorial (remaining_students - 1)) * remaining_students

end NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangement_count_l641_64139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l641_64182

/-- The angle of inclination of a line given its equation in the form ax + by + c = 0 -/
noncomputable def angle_of_inclination (a b : ℝ) : ℝ :=
  Real.pi - Real.arctan (|a / b|)

/-- The line equation 2x + y - 1 = 0 has an angle of inclination π - arctan 2 -/
theorem line_inclination : angle_of_inclination 2 (-1) = Real.pi - Real.arctan 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l641_64182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_replant_prob_is_half_X_is_binomial_expectation_X_is_two_l641_64158

-- Define the probability of a seed germinating
noncomputable def seed_germination_prob : ℝ := 1 / 2

-- Define the number of seeds per pit
def seeds_per_pit : ℕ := 3

-- Define the probability of no replanting needed for a single pit
noncomputable def no_replant_prob : ℝ := 1 - (seed_germination_prob ^ seeds_per_pit + 
  seeds_per_pit * seed_germination_prob * (1 - seed_germination_prob) ^ 2)

-- Define the number of pits
def n : ℕ := 4

-- Define the random variable X as the number of pits needing replanting
noncomputable def X : ℕ → ℝ := fun k => Nat.choose n k * (1 - no_replant_prob) ^ k * no_replant_prob ^ (n - k)

-- Theorem 1: Probability of no replanting needed for each pit
theorem no_replant_prob_is_half : no_replant_prob = 1 / 2 := by sorry

-- Theorem 2: X follows a Binomial(4, 1/2) distribution
theorem X_is_binomial : ∀ k, 0 ≤ k ∧ k ≤ n → X k = Nat.choose n k * (1/2)^k * (1/2)^(n-k) := by sorry

-- Theorem 3: Expectation of X is 2
theorem expectation_X_is_two : n * (1 - no_replant_prob) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_replant_prob_is_half_X_is_binomial_expectation_X_is_two_l641_64158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_inclination_angle_relationship_l641_64107

-- Define the line equation
def line_equation (x y α : ℝ) : Prop := x * Real.cos α - y + 1 = 0

-- Define the inclination angle
noncomputable def inclination_angle (α : ℝ) : ℝ := Real.arctan (Real.cos α)

-- Theorem statement
theorem inclination_angle_range :
  ∀ α : ℝ, ∃ θ : ℝ, 
    (line_equation (Real.cos θ) (Real.sin θ) α) ∧ 
    ((0 ≤ θ ∧ θ ≤ Real.pi/4) ∨ (3*Real.pi/4 ≤ θ ∧ θ < Real.pi)) :=
by
  sorry

-- Additional helper theorem to establish the relationship between α and θ
theorem inclination_angle_relationship (α : ℝ) :
  Real.tan (inclination_angle α) = Real.cos α :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_inclination_angle_relationship_l641_64107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_between_18_and_19_l641_64110

-- Define the points
def A : ℝ × ℝ := (16, 0)
def B : ℝ × ℝ := (0, 0)

-- Define D implicitly
def D : ℝ × ℝ := (0, 2)

-- Distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_sum_between_18_and_19 :
  18 < distance A D + distance B D ∧ distance A D + distance B D < 19 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_between_18_and_19_l641_64110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_two_eq_eight_l641_64119

/-- A power function passing through (3, 27) -/
noncomputable def f : ℝ → ℝ :=
  fun x => x ^ (Real.log 27 / Real.log 3)

/-- Theorem stating that f(2) = 8 -/
theorem f_of_two_eq_eight : f 2 = 8 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp [Real.rpow_def]
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_two_eq_eight_l641_64119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_implies_k_value_l641_64144

theorem root_difference_implies_k_value (k : ℝ) :
  (∃ α β : ℝ, α^2 + k*α + 15 = 0 ∧ β^2 + k*β + 15 = 0) ∧
  (∃ γ δ : ℝ, γ^2 - k*γ + 15 = 0 ∧ δ^2 - k*δ + 15 = 0) ∧
  (∃ f : ℝ → ℝ, Function.Bijective f ∧
    (∀ x, x^2 + k*x + 15 = 0 → (f x)^2 - k*(f x) + 15 = 0) ∧
    (∀ x, x^2 + k*x + 15 = 0 → f x = x + 3)) →
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_implies_k_value_l641_64144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necklace_sales_total_l641_64195

/-- Calculate the total amount earned from necklace sales with discount and tax --/
theorem necklace_sales_total (bead_count gemstone_count crystal_count wooden_count : ℕ)
  (bead_price gemstone_price crystal_price wooden_price : ℚ)
  (discount_rate tax_rate : ℚ) :
  let total_before_discount := bead_count * bead_price + gemstone_count * gemstone_price +
                               crystal_count * crystal_price + wooden_count * wooden_price
  let gemstone_discount := if gemstone_count ≥ 2 then discount_rate * (gemstone_count * gemstone_price) else 0
  let total_after_discount := total_before_discount - gemstone_discount
  let total_with_tax := total_after_discount * (1 + tax_rate)
  bead_count = 4 ∧ gemstone_count = 3 ∧ crystal_count = 2 ∧ wooden_count = 5 ∧
  bead_price = 3 ∧ gemstone_price = 7 ∧ crystal_price = 5 ∧ wooden_price = 2 ∧
  discount_rate = 1/10 ∧ tax_rate = 2/25 →
  total_with_tax = 5497/100 := by
  intro h
  sorry

#eval (4 * 3 + 3 * 7 + 2 * 5 + 5 * 2 : ℚ)
#eval ((4 * 3 + 3 * 7 + 2 * 5 + 5 * 2 : ℚ) - (1/10 * (3 * 7)))
#eval ((4 * 3 + 3 * 7 + 2 * 5 + 5 * 2 : ℚ) - (1/10 * (3 * 7))) * (1 + 2/25)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necklace_sales_total_l641_64195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l641_64141

def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}

theorem set_operations :
  (Set.univ \ (A ∪ B) = {x : ℝ | x ≤ 2 ∨ x ≥ 10}) ∧
  (Set.univ \ (A ∩ B) = {x : ℝ | x < 3 ∨ x ≥ 7}) ∧
  ((Set.univ \ A) ∩ B = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l641_64141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_eccentricity_l641_64179

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ b ≤ a

/-- A parabola with focal distance f -/
structure Parabola where
  f : ℝ
  h_positive : 0 < f

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Theorem: Eccentricity of an ellipse intersected by a specific parabola -/
theorem ellipse_parabola_intersection_eccentricity (e : Ellipse) (p : Parabola) :
  (p.f = e.a) →  -- Parabola focus at ellipse center
  (∃ x y : ℝ, x^2 / e.a^2 + y^2 / e.b^2 = 1 ∧ y = x^2 / (4 * p.f)) →  -- Parabola passes through ellipse foci
  (∃ x₁ x₂ x₃ y₁ y₂ y₃ : ℝ, 
    (x₁^2 / e.a^2 + y₁^2 / e.b^2 = 1 ∧ y₁ = x₁^2 / (4 * p.f)) ∧
    (x₂^2 / e.a^2 + y₂^2 / e.b^2 = 1 ∧ y₂ = x₂^2 / (4 * p.f)) ∧
    (x₃^2 / e.a^2 + y₃^2 / e.b^2 = 1 ∧ y₃ = x₃^2 / (4 * p.f)) ∧
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃)) →  -- Exactly three intersection points
  eccentricity e = 2 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_eccentricity_l641_64179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l641_64131

/-- Definition of f₁(x) -/
noncomputable def f₁ (x : ℝ) : ℝ := Real.sqrt (x^2 + 48)

/-- Definition of fₙ(x) for n ≥ 1 -/
noncomputable def f : ℕ → ℝ → ℝ
  | 0, x => f₁ x  -- Define f₀ as f₁ for completeness
  | n + 1, x => Real.sqrt (x^2 + 6 * f n x)

/-- Theorem stating that 4 is the unique real solution to fₙ(x) = 2x -/
theorem unique_solution (n : ℕ) (hn : n ≥ 1) :
  ∀ x : ℝ, f n x = 2 * x ↔ x = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l641_64131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_k_for_one_asymptote_l641_64134

/-- The rational function g(x) with parameter k -/
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x + k) / (x^2 - 2*x - 35)

/-- Theorem stating that there is no k for which g(x) has exactly one vertical asymptote -/
theorem no_k_for_one_asymptote :
  ¬ ∃ k : ℝ, ∃! x : ℝ, (x^2 - 2*x - 35 = 0 ∧ x^2 - 2*x + k ≠ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_k_for_one_asymptote_l641_64134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_1001_eq_1752_l641_64137

/-- The number of positive integer multiples of 1001 that can be expressed as 10^j - 10^i -/
def count_multiples_1001 : ℕ :=
  Finset.sum (Finset.range 25) (fun k ↦ (150 - 6 * k))

/-- The theorem stating that the count of multiples is 1752 -/
theorem count_multiples_1001_eq_1752 : count_multiples_1001 = 1752 := by
  sorry

#eval count_multiples_1001

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_1001_eq_1752_l641_64137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_cosine_l641_64180

noncomputable section

open Real

structure Line where
  slope : ℝ
  intercept : ℝ

def Line.angle (l : Line) : ℝ := Real.arctan l.slope

def Line.isPerpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

theorem perpendicular_lines_cosine (α : ℝ) :
  (∃ (l : Line), l.angle = α ∧ l.isPerpendicular ⟨-1/2, 2⟩) →
  cos ((2017 / 2) * π - 2 * α) = 4 / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_cosine_l641_64180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_camp_wonka_marshmallows_l641_64116

def camp_wonka (total_campers : ℕ) (boy_percentage : ℚ) (girl_percentage : ℚ)
                (boy_toast_percentage : ℚ) (girl_toast_percentage : ℚ) : ℕ :=
  let num_boys := (total_campers : ℚ) * boy_percentage
  let num_girls := (total_campers : ℚ) * girl_percentage
  let boys_toasting := num_boys * boy_toast_percentage
  let girls_toasting := num_girls * girl_toast_percentage
  (boys_toasting + girls_toasting).floor.toNat

theorem camp_wonka_marshmallows :
  camp_wonka 180 (60/100) (40/100) (55/100) (80/100) = 117 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_camp_wonka_marshmallows_l641_64116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_ratio_l641_64167

/-- Represents the tank filling problem -/
structure TankProblem where
  capacity : ℝ
  fillRate : ℝ
  drainRate1 : ℝ
  drainRate2 : ℝ
  fillTime : ℝ

/-- Calculates the net flow rate into the tank -/
def netFlowRate (p : TankProblem) : ℝ :=
  p.fillRate - (p.drainRate1 + p.drainRate2)

/-- Calculates the total water added to the tank during the fill time -/
def totalWaterAdded (p : TankProblem) : ℝ :=
  (netFlowRate p) * p.fillTime

/-- Calculates the initial amount of water in the tank -/
def initialWater (p : TankProblem) : ℝ :=
  p.capacity - totalWaterAdded p

/-- Theorem stating the ratio of initial water to capacity is 1:2 -/
theorem initial_water_ratio (p : TankProblem) 
  (h1 : p.capacity = 2)
  (h2 : p.fillRate = 0.5)
  (h3 : p.drainRate1 = 0.25)
  (h4 : p.drainRate2 = 1/6)
  (h5 : p.fillTime = 12) :
  initialWater p / p.capacity = 1/2 := by
  sorry

-- Remove the #eval statement as it's causing issues
-- #eval initial_water_ratio
--   { capacity := 2
--   , fillRate := 0.5
--   , drainRate1 := 0.25
--   , drainRate2 := 1/6
--   , fillTime := 12 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_ratio_l641_64167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_50_mod_7_t_50_equiv_t_49_mod_7_t_50_mod_7_final_l641_64124

-- Define the sequence T
def T : ℕ → ℕ
  | 0 => 3  -- Define for 0 to handle all natural numbers
  | n+1 => 3^(T n)

-- State the theorem
theorem t_50_mod_7 : T 49 % 7 = 6 := by
  sorry

-- Prove that T 50 ≡ T 49 (mod 7)
theorem t_50_equiv_t_49_mod_7 : T 50 % 7 = T 49 % 7 := by
  sorry

-- Combine the two theorems to get the final result
theorem t_50_mod_7_final : T 50 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_50_mod_7_t_50_equiv_t_49_mod_7_t_50_mod_7_final_l641_64124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_price_l641_64175

/-- The price of a chair in dollars -/
def chair_price : ℝ := sorry

/-- The price of a table in dollars -/
def table_price : ℝ := 84

/-- The relationship between chair and table prices -/
axiom price_relationship : 2 * chair_price + table_price = 0.6 * (chair_price + 2 * table_price)

theorem total_price : chair_price + table_price = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_price_l641_64175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pony_bridge_crossing_time_l641_64149

-- Define the cycle times for monsters A and B
def cycleTimeA : ℕ := 3
def cycleTimeB : ℕ := 5

-- Theorem statement
theorem pony_bridge_crossing_time :
  Nat.lcm cycleTimeA cycleTimeB = 15 := by
  -- Proof goes here
  sorry

#eval Nat.lcm cycleTimeA cycleTimeB

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pony_bridge_crossing_time_l641_64149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_candidate_percentage_l641_64184

def votes : List Nat := [4571, 9892, 17653, 3217, 15135, 11338, 8629]

theorem winning_candidate_percentage :
  let total_votes := votes.sum
  let max_votes := votes.maximum?
  let percentage := (max_votes.getD 0 : ℚ) / total_votes * 100
  ∃ (ε : ℚ), abs (percentage - 24.03) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_candidate_percentage_l641_64184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l641_64193

noncomputable section

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x = y

-- Define a point on the curve
def point_on_curve (M : ℝ × ℝ) : Prop := curve_C M.1 M.2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  curve_C A.1 A.2 ∧ line_l A.1 A.2 ∧
  curve_C B.1 B.2 ∧ line_l B.1 B.2 ∧
  A ≠ B

-- Define the area of triangle ABM
noncomputable def triangle_area (A B M : ℝ × ℝ) : ℝ :=
  abs ((A.1 - M.1) * (B.2 - M.2) - (B.1 - M.1) * (A.2 - M.2)) / 2

-- Theorem statement
theorem max_triangle_area :
  ∃ (A B : ℝ × ℝ),
    intersection_points A B →
    ∀ (M : ℝ × ℝ),
      point_on_curve M →
      triangle_area A B M ≤ (Real.sqrt 2 + 1) / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l641_64193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l641_64153

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
theorem triangle_problem (a b c A B C : ℝ) 
    (h1 : (Real.sin A + Real.sin C) / (c - b) = Real.sin B / (c - a))
    (h2 : a = 2 * Real.sqrt 3) 
    (h3 : (1/2) * b * c * Real.sin A = 2 * Real.sqrt 3) :
  A = π/3 ∧ a + b + c = 6 + 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l641_64153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_condition_l641_64147

-- Define the function f(x)
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := -1/2 * x^2 + b * Real.log (x + 2)

-- Define the derivative of f(x)
noncomputable def f_deriv (b : ℝ) (x : ℝ) : ℝ := -x + b / (x + 2)

-- State the theorem
theorem monotonic_decreasing_condition (b : ℝ) : 
  (∀ x ∈ Set.Ioo (-1 : ℝ) (1 : ℝ), f_deriv b x < 0) ↔ b ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_condition_l641_64147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platform_l641_64100

theorem train_crossing_platform (train_length platform_length signal_pole_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 500)
  (h3 : signal_pole_time = 18) : 
  let train_speed := train_length / signal_pole_time
  let total_distance := train_length + platform_length
  (total_distance / train_speed) = 48 := by
  sorry

#check train_crossing_platform

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platform_l641_64100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l641_64169

/-- The parabola y = x^2 - 4x + 20 -/
noncomputable def parabola (x : ℝ) : ℝ := x^2 - 4*x + 20

/-- The line y = x - 6 -/
noncomputable def line (x : ℝ) : ℝ := x - 6

/-- The distance between a point (x, y) and the line y = x - 6 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x - y - 6| / Real.sqrt 2

/-- The shortest distance between a point on the parabola and a point on the line -/
theorem shortest_distance : 
  ∃ (x : ℝ), ∀ (z : ℝ), distance_to_line z (parabola z) ≥ 103 * Real.sqrt 2 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l641_64169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_female_nonbinary_difference_l641_64123

/-- Represents the number of female students -/
def F : ℕ := sorry

/-- Represents the number of non-binary students -/
def N : ℕ := sorry

/-- The average score of all students -/
def total_average : ℝ := 90

/-- The number of male students -/
def male_count : ℕ := 8

/-- The average score of male students -/
def male_average : ℝ := 85

/-- The average score of female students -/
def female_average : ℝ := 92

/-- The average score of non-binary students -/
def nonbinary_average : ℝ := 88

/-- Theorem stating the difference between female and non-binary students -/
theorem female_nonbinary_difference : 
  total_average * (male_count + F + N) = 
    (male_average * male_count) + (female_average * F) + (nonbinary_average * N) →
  F - N = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_female_nonbinary_difference_l641_64123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l641_64155

/-- The function f for which we want to find the increasing interval -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) - Real.cos (ω * x)

/-- The tangent function used in the problem conditions -/
noncomputable def tan_func (ω : ℝ) (x : ℝ) : ℝ := Real.tan (ω * x)

/-- Theorem stating the interval where f is increasing -/
theorem f_increasing_interval (ω : ℝ) (a : ℝ) :
  (ω > 0) →
  (∃ A B : ℝ, tan_func ω A = a ∧ tan_func ω B = a ∧ |B - A| ≥ π) →
  (∀ k : ℤ, ∀ x ∈ Set.Icc (2 * π * k - π / 3) (2 * π * k + 2 * π / 3),
    HasDerivAt (f ω) (Real.sqrt 3 * ω * Real.cos (ω * x) + ω * Real.sin (ω * x)) x ∧
    Real.sqrt 3 * ω * Real.cos (ω * x) + ω * Real.sin (ω * x) > 0) :=
by
  sorry

#check f_increasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l641_64155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l641_64101

theorem product_remainder (a b c : ℕ) : 
  a % 7 = 2 → b % 7 = 3 → c % 7 = 4 → (a * b * c) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l641_64101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_max_min_p_max_min_l641_64190

-- Function u(x) = x^3 - 3x^2 - 9x + 35
def u (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 35

-- Function p(x) = x^2 ln(x)
noncomputable def p (x : ℝ) : ℝ := x^2 * Real.log x

-- Theorem for the maximum and minimum of u(x) on [-4, 4]
theorem u_max_min : 
  (∃ x ∈ Set.Icc (-4 : ℝ) 4, u x = 40) ∧ 
  (∀ x ∈ Set.Icc (-4 : ℝ) 4, u x ≤ 40) ∧
  (∃ x ∈ Set.Icc (-4 : ℝ) 4, u x = -41) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) 4, u x ≥ -41) := by
  sorry

-- Theorem for the maximum and minimum of p(x) on [1, e]
theorem p_max_min : 
  (∃ x ∈ Set.Icc (1 : ℝ) (Real.exp 1), p x = (Real.exp 1)^2) ∧ 
  (∀ x ∈ Set.Icc (1 : ℝ) (Real.exp 1), p x ≤ (Real.exp 1)^2) ∧
  (∃ x ∈ Set.Icc (1 : ℝ) (Real.exp 1), p x = 0) ∧
  (∀ x ∈ Set.Icc (1 : ℝ) (Real.exp 1), p x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_max_min_p_max_min_l641_64190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angles_implies_equilateral_l641_64132

-- Define a triangle
structure Triangle where
  a : ℝ  -- side length a
  b : ℝ  -- side length b
  c : ℝ  -- side length c
  α : ℝ  -- angle α
  β : ℝ  -- angle β
  γ : ℝ  -- angle γ

-- Define an equilateral triangle
def IsEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- Define a triangle with equal angles
def HasEqualAngles (t : Triangle) : Prop :=
  t.α = t.β ∧ t.β = t.γ

-- Theorem statement
theorem equal_angles_implies_equilateral (t : Triangle) :
  HasEqualAngles t → IsEquilateral t :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angles_implies_equilateral_l641_64132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisibility_pairs_l641_64177

theorem prime_divisibility_pairs (n p : ℕ) : 
  Nat.Prime p ∧ 
  n ≤ 2 * p ∧ 
  (p - 1)^n + 1 ∣ n^(p - 1) →
  ((n = 1 ∧ Nat.Prime p) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3)) := by
  sorry

#check prime_divisibility_pairs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisibility_pairs_l641_64177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_count_l641_64122

/-- Represents the number of coins of each type in the bag -/
def num_coins : ℕ := sorry

/-- The total value of all coins in rupees -/
def total_value : ℚ := 35

/-- The value of one rupee in rupees -/
def one_rupee_value : ℚ := 1

/-- The value of 50 paise in rupees -/
def fifty_paise_value : ℚ := 1/2

/-- The value of 25 paise in rupees -/
def twenty_five_paise_value : ℚ := 1/4

/-- Theorem stating that there are 20 coins of each type -/
theorem coin_count : 
  (num_coins * one_rupee_value + 
   num_coins * fifty_paise_value + 
   num_coins * twenty_five_paise_value = total_value) → 
  num_coins = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_count_l641_64122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_length_l641_64125

/-- The maximum projection length of OP on the x-axis given the specified conditions -/
theorem max_projection_length (A : ℝ × ℝ) (P : ℝ × ℝ) :
  (A.1^2 / 25 + A.2^2 / 9 = 1) →  -- A is on the ellipse
  (∃ l : ℝ, P = ((l - 1) * A.1, (l - 1) * A.2)) →  -- P satisfies AP⃗ = (l-1)OA⃗
  (A.1 * P.1 + A.2 * P.2 = 72) →  -- OA⃗ ⋅ OP⃗ = 72
  (∃ proj : ℝ, proj ≤ 15 ∧ 
    ∀ θ : ℝ, |P.1 * Real.cos θ + P.2 * Real.sin θ| ≤ proj ∧
    ∃ θ₀ : ℝ, |P.1 * Real.cos θ₀ + P.2 * Real.sin θ₀| = 15) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_length_l641_64125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_theorem_l641_64178

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Represents a hyperbola -/
def Hyperbola := {p : Point | p.x^2 / 4 - p.y^2 = 1}

/-- The right focus of the hyperbola -/
def rightFocus : Point := { x := 2, y := 0 }

/-- Check if a line passes through the right focus -/
def passesThroughRightFocus (l : Line) : Prop :=
  l.p1 = rightFocus ∨ l.p2 = rightFocus

/-- Check if a line intersects the hyperbola at two points -/
def intersectsHyperbola (l : Line) : Prop :=
  ∃ (p1 p2 : Point), p1 ∈ Hyperbola ∧ p2 ∈ Hyperbola ∧ 
    (l.p1 = p1 ∧ l.p2 = p2) ∨ (l.p1 = p2 ∧ l.p2 = p1)

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

/-- The main theorem -/
theorem hyperbola_intersection_theorem :
  ∃! (a : ℝ), 
    (∃! (l1 l2 l3 : Line), 
      passesThroughRightFocus l1 ∧ passesThroughRightFocus l2 ∧ passesThroughRightFocus l3 ∧
      intersectsHyperbola l1 ∧ intersectsHyperbola l2 ∧ intersectsHyperbola l3 ∧
      distance l1.p1 l1.p2 = a ∧ distance l2.p1 l2.p2 = a ∧ distance l3.p1 l3.p2 = a) ∧
    a = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_theorem_l641_64178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_theorem_l641_64118

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_valid_partition (s : Set ℕ) (n : ℕ) (k : ℕ) : Prop :=
  ∃ (partition : List (ℕ × ℕ)),
    partition.length = k ∧
    (∀ (pair : ℕ × ℕ), pair ∈ partition → pair.1 ∈ s ∧ pair.2 ∈ s) ∧
    (∀ x ∈ s, ∃ (pair : ℕ × ℕ), pair ∈ partition ∧ (x = pair.1 ∨ x = pair.2)) ∧
    (∀ (pair1 pair2 : ℕ × ℕ), pair1 ∈ partition → pair2 ∈ partition → pair1 ≠ pair2 →
      is_prime (pair1.1 + pair1.2) ∧ is_prime (pair2.1 + pair2.2) ∧
      pair1.1 + pair1.2 ≠ pair2.1 + pair2.2)

theorem partition_theorem :
  (is_valid_partition (Finset.range 12).toSet 12 6) ∧
  ¬(is_valid_partition (Finset.range 22).toSet 22 11) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_theorem_l641_64118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elberta_money_rounded_l641_64170

noncomputable def granny_smith_money : ℝ := 75

noncomputable def anjou_money : ℝ := granny_smith_money / 4

noncomputable def elberta_money : ℝ := anjou_money * 1.1

theorem elberta_money_rounded : Int.floor (elberta_money + 0.5) = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elberta_money_rounded_l641_64170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liters_to_pints_conversion_l641_64172

/-- Approximation relation for real numbers -/
def approx (x y : ℝ) : Prop := abs (x - y) < 0.05

notation:50 a " ≈ " b => approx a b

/-- Given that 0.75 liters is approximately 1.575 pints, prove that one liter is approximately 2.1 pints. -/
theorem liters_to_pints_conversion (h : approx ((0.75 : ℝ) * 1) 1.575) :
  approx ((1 : ℝ) * 1) 2.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_liters_to_pints_conversion_l641_64172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_theorem_l641_64128

/-- Represents the investment scenario described in the problem -/
structure Investment where
  total : ℚ
  ratio : ℚ
  growth_rate : ℚ
  years : ℕ

/-- Calculates the amount invested in tech companies -/
def tech_investment (i : Investment) : ℚ :=
  i.total * i.ratio / (1 + i.ratio)

/-- Calculates the future value of the investment -/
def future_value (i : Investment) : ℚ :=
  i.total * (1 + i.growth_rate) ^ i.years

/-- Theorem stating the investment amounts and future value -/
theorem investment_theorem (i : Investment) 
  (h1 : i.total = 250000)
  (h2 : i.ratio = 6)
  (h3 : i.growth_rate = 1/20)
  (h4 : i.years = 3) :
  ⌊tech_investment i⌋ = 214286 ∧ 
  ⌊future_value i⌋ = 289406 := by
  sorry

#eval ⌊(250000 : ℚ) * 6 / 7⌋
#eval ⌊(250000 : ℚ) * (1 + 1/20)^3⌋

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_theorem_l641_64128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_with_tangent_circles_l641_64112

-- Define the triangle ABC
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the circles
structure Circle where
  center : Point
  radius : ℝ

-- Define the configuration
structure Configuration where
  triangle : Triangle
  circles : Fin 6 → Circle

-- Helper functions (declared but not implemented)
def are_tangent_to_triangle : (Fin 6 → Circle) → Triangle → Prop := sorry
def are_tangent_to_each_other : (Fin 6 → Circle) → Prop := sorry
def circles_on_side : Point → Point → List Point → Prop := sorry
def perimeter : Triangle → ℝ := sorry

-- State the theorem
theorem perimeter_of_triangle_with_tangent_circles 
  (config : Configuration) 
  (h1 : ∀ i, (config.circles i).radius = 2)
  (h2 : are_tangent_to_triangle config.circles config.triangle)
  (h3 : are_tangent_to_each_other config.circles)
  (h4 : circles_on_side config.triangle.A config.triangle.B [config.circles 0, config.circles 1, config.circles 2])
  (h5 : circles_on_side config.triangle.B config.triangle.C [config.circles 3])
  (h6 : circles_on_side config.triangle.C config.triangle.A [config.circles 4, config.circles 5]) :
  perimeter config.triangle = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_with_tangent_circles_l641_64112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_range_l641_64159

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ) + 1

theorem phi_range (ω φ : ℝ) (h_ω : ω > 0) (h_φ : |φ| ≤ π/2)
  (h_period : ∀ x₁ x₂, f ω φ x₁ = 3 → f ω φ x₂ = 3 → x₁ ≠ x₂ → |x₁ - x₂| = π)
  (h_greater : ∀ x ∈ Set.Ioo (π/24) (π/3), f ω φ x > 2) :
  φ ∈ Set.Icc (π/12) (π/6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_range_l641_64159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_is_850_l641_64163

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculate the area of intersection between a plane and a cube -/
noncomputable def intersectionArea (cube : Cube) (plane : Plane) : ℝ := sorry

/-- The main theorem stating the area of intersection -/
theorem intersection_area_is_850 (cube : Cube) (P Q R : Point3D) : 
  cube.A = Point3D.mk 0 0 0 →
  cube.B = Point3D.mk 24 0 0 →
  cube.C = Point3D.mk 24 0 24 →
  cube.D = Point3D.mk 24 24 24 →
  P.x - cube.A.x = 4 →
  cube.B.x - P.x = 16 →
  Q.y - cube.B.y = 12 →
  cube.C.z - R.z = 8 →
  P.y = 0 ∧ P.z = 0 →
  Q.x = 24 ∧ Q.z = 12 →
  R.x = 24 ∧ R.y = 8 →
  let plane := Plane.mk 2 (-1) 1 24
  intersectionArea cube plane = 850 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_is_850_l641_64163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_expected_value_l641_64120

-- Define the random variable X and its probability mass function
variable (X : ℝ → ℝ)
variable (P : ℝ → ℝ)

-- Define the expected value of X as a function of t
def E (t : ℝ) : ℝ := 0.3 * t + 0.2 * (2 - t) + 0.2 * t^2 + 0.3 * 6

-- State the theorem
theorem minimize_expected_value :
  ∃ (t : ℝ), t ∈ Set.Icc (-1 : ℝ) 2 ∧
  ∀ (s : ℝ), s ∈ Set.Icc (-1 : ℝ) 2 → E t ≤ E s ∧
  t = -0.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_expected_value_l641_64120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_is_68_l641_64196

/-- Represents a type of rock with its weight and value -/
structure Rock where
  weight : ℕ
  value : ℕ
deriving BEq, Repr

/-- The problem setup -/
def cave_problem :=
  let rock6 : Rock := ⟨6, 16⟩
  let rock3 : Rock := ⟨3, 9⟩
  let rock2 : Rock := ⟨2, 3⟩
  let max_weight : ℕ := 24
  let max_rocks_per_type : ℕ := 4
  let min_rocks_available : ℕ := 30
  (rock6, rock3, rock2, max_weight, max_rocks_per_type, min_rocks_available)

/-- A valid collection of rocks -/
def valid_collection (rocks : List Rock) : Prop :=
  (rocks.foldl (λ acc r => acc + r.weight) 0 ≤ 24) ∧
  (∀ r : Rock, (rocks.filter (λ x => x == r)).length ≤ 4)

/-- The total value of a collection of rocks -/
def total_value (rocks : List Rock) : ℕ :=
  rocks.foldl (λ acc r => acc + r.value) 0

/-- The main theorem -/
theorem max_value_is_68 :
  ∃ (rocks : List Rock),
    valid_collection rocks ∧
    total_value rocks = 68 ∧
    (∀ (other_rocks : List Rock),
      valid_collection other_rocks →
      total_value other_rocks ≤ 68) := by
  sorry

#eval cave_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_is_68_l641_64196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_oil_price_theorem_l641_64192

/-- Calculates the reduced price of oil per kg given a price reduction percentage and additional quantity obtained for a fixed price. -/
noncomputable def reduced_price_per_kg (reduction_percent : ℝ) (additional_kg : ℝ) (fixed_price : ℝ) : ℝ :=
  let original_price := (fixed_price / (1 - reduction_percent / 100) - fixed_price) / additional_kg
  original_price * (1 - reduction_percent / 100)

/-- Theorem stating that given a 30% reduction in oil price and the ability to buy 9 kg more for Rs. 1800 after the reduction, the reduced price is approximately Rs. 60 per kg. -/
theorem reduced_oil_price_theorem :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |reduced_price_per_kg 30 9 1800 - 60| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_oil_price_theorem_l641_64192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_C_to_origin_l641_64164

noncomputable def A : ℝ × ℝ := (1, 1)
noncomputable def B : ℝ × ℝ := (4, 11/2)

theorem distance_C_to_origin (C : ℝ × ℝ) 
  (h : C - A = 2 • (B - C)) : 
  Real.sqrt ((C.1)^2 + (C.2)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_C_to_origin_l641_64164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_30_l641_64189

noncomputable def full_rotation : ℝ := 360

def clock_hours : ℕ := 12

def minutes_per_hour : ℕ := 60

noncomputable def hour_hand_degrees_per_hour : ℝ := full_rotation / clock_hours

noncomputable def minute_hand_degrees_per_minute : ℝ := full_rotation / minutes_per_hour

noncomputable def hour_hand_position : ℝ := 3 * hour_hand_degrees_per_hour + (30 / minutes_per_hour) * hour_hand_degrees_per_hour

noncomputable def minute_hand_position : ℝ := 30 * minute_hand_degrees_per_minute

noncomputable def clock_angle : ℝ := min (abs (minute_hand_position - hour_hand_position)) (full_rotation - abs (minute_hand_position - hour_hand_position))

theorem clock_angle_at_3_30 : clock_angle = 75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_30_l641_64189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_30_l641_64199

theorem sum_remainder_mod_30 (p q r s t : ℕ) 
  (hp : p % 30 = 7)
  (hq : q % 30 = 11)
  (hr : r % 30 = 18)
  (hs : s % 30 = 5)
  (ht : t % 30 = 9) :
  (p + q + r + s + t) % 30 = 20 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_30_l641_64199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conical_frustum_properties_l641_64197

/-- Represents a conical frustum -/
structure ConicalFrustum where
  top_radius : ℝ
  bottom_radius : ℝ
  height : ℝ

/-- Calculates the lateral surface area of a conical frustum -/
noncomputable def lateral_surface_area (f : ConicalFrustum) : ℝ :=
  Real.pi * (f.top_radius + f.bottom_radius) * Real.sqrt (f.height^2 + (f.bottom_radius - f.top_radius)^2)

/-- Calculates the original height of the whole cone -/
noncomputable def original_cone_height (f : ConicalFrustum) : ℝ :=
  (f.bottom_radius * f.height) / (f.bottom_radius - f.top_radius)

/-- Theorem stating the properties of a specific conical frustum -/
theorem conical_frustum_properties :
  let f : ConicalFrustum := ⟨4, 6, 9⟩
  lateral_surface_area f = 10 * Real.pi * Real.sqrt 85 ∧
  original_cone_height f = 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conical_frustum_properties_l641_64197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_length_is_five_l641_64102

-- Define the set of digits
def Digits : Finset ℕ := {1, 2, 3, 4, 5}

-- Define a function to check if two numbers are adjacent in a list
def are_adjacent (a b : ℕ) (l : List ℕ) : Prop :=
  ∃ i, (l.get? i = some a ∧ l.get? (i + 1) = some b) ∨
       (l.get? i = some b ∧ l.get? (i + 1) = some a)

-- Define the property of a valid integer
def valid_integer (l : List ℕ) : Prop :=
  l.Nodup ∧ 
  l.toFinset = Digits ∧ 
  ¬(are_adjacent 3 4 l)

-- State the theorem
theorem integer_length_is_five :
  (∃ S : Finset (List ℕ), (∀ l ∈ S, valid_integer l) ∧ S.card = 72) →
  (∀ l, valid_integer l → l.length = 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_length_is_five_l641_64102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x_squared_of_f_l641_64145

noncomputable def f (x : ℝ) : ℝ := (-2 * Real.sqrt x - 1 / Real.sqrt x) ^ 6

noncomputable def coeff_x_squared (f : ℝ → ℝ) : ℝ :=
  (1 / 2) * ((deriv (deriv f)) 0)

theorem coeff_x_squared_of_f : coeff_x_squared f = 192 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x_squared_of_f_l641_64145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_angle_theorem_l641_64126

/-- Two identical cones with common vertex D touch plane α along generatrices DE and DF.
    φ is the angle EDF, and β is the angle between the line of intersection of
    the base planes of the cones and plane α. -/
def cone_configuration (φ β : ℝ) : Prop :=
  φ > 0 ∧ φ < Real.pi ∧ β > 0 ∧ β < Real.pi

/-- The angle between the height and the generatrix of each cone -/
noncomputable def cone_angle (φ β : ℝ) : ℝ :=
  Real.arctan (Real.sin (φ / 2) / Real.cos (β / 9))

/-- Theorem stating the relationship between the cone configuration and the resulting angle -/
theorem cone_angle_theorem (φ β : ℝ) :
  cone_configuration φ β →
  cone_angle φ β = Real.arctan (Real.sin (φ / 2) / Real.cos (β / 9)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_angle_theorem_l641_64126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_monotonicity_l641_64188

theorem exponential_monotonicity (a b : ℝ) (h : a > b) : (2 : ℝ)^a > (2 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_monotonicity_l641_64188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_for_smallest_central_angle_l641_64115

/-- The circle with center (2, 0) and radius 2√2 -/
def myCircle : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + p.2^2 = 8}

/-- The point P -/
noncomputable def P : ℝ × ℝ := (1, Real.sqrt 2)

/-- The center of the circle -/
def C : ℝ × ℝ := (2, 0)

/-- A line passing through point P -/
structure Line where
  slope : ℝ
  passesThrough : (P.1, P.2) ∈ {p : ℝ × ℝ | p.2 - P.2 = slope * (p.1 - P.1)}

/-- Central angle function (definition omitted) -/
def centralAngle (c : Set (ℝ × ℝ)) (center : ℝ × ℝ) (l : Line) : ℝ := sorry

/-- Theorem stating that the slope for the smallest central angle is √2/2 -/
theorem slope_for_smallest_central_angle (l : Line) :
  (∀ l' : Line, l.slope = Real.sqrt 2 / 2 → 
    centralAngle myCircle C l ≤ centralAngle myCircle C l') →
  l.slope = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_for_smallest_central_angle_l641_64115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptote_l641_64187

/-- A point is a focus of the hyperbola mx² + ny² = 1 -/
def is_focus_of_hyperbola (f : ℝ × ℝ) (m n : ℝ) : Prop :=
sorry

/-- A point is a focus of the parabola y = (1/8)x² -/
def is_focus_of_parabola (f : ℝ × ℝ) : Prop :=
sorry

/-- The eccentricity of the hyperbola mx² + ny² = 1 -/
noncomputable def eccentricity_of_hyperbola (m n : ℝ) : ℝ :=
sorry

/-- A real number is the distance from the focus to an asymptote of the hyperbola mx² + ny² = 1 -/
def is_distance_focus_to_asymptote (d m n : ℝ) : Prop :=
sorry

/-- The distance from the focus to an asymptote of a hyperbola sharing a focus with a parabola -/
theorem distance_focus_to_asymptote 
  (m n : ℝ) 
  (h_hyperbola : ∀ x y : ℝ, m * x^2 + n * y^2 = 1) 
  (h_parabola : ∀ x y : ℝ, y = (1/8) * x^2) 
  (h_shared_focus : ∃ f : ℝ × ℝ, is_focus_of_hyperbola f m n ∧ is_focus_of_parabola f) 
  (h_eccentricity : eccentricity_of_hyperbola m n = 2) : 
  ∃ d : ℝ, d = Real.sqrt 3 ∧ is_distance_focus_to_asymptote d m n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptote_l641_64187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l641_64157

/-- The focus of a parabola given by y = ax^2 + bx + c -/
noncomputable def parabola_focus (a b c : ℝ) : ℝ × ℝ :=
  let h := -b / (2 * a)
  let k := c - b^2 / (4 * a)
  (h, k + 1 / (4 * a))

/-- Theorem: The focus of the parabola y = 4x^2 + 8x - 5 is at (-1, -8.9375) -/
theorem focus_of_specific_parabola :
  parabola_focus 4 8 (-5) = (-1, -8.9375) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l641_64157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_body_motion_indeterminate_l641_64136

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A rigid body in 3D space -/
structure RigidBody where
  points : Set Point3D
  motion : Motion

/-- The motion of a rigid body -/
inductive Motion
  | Determinate
  | Indeterminate

/-- A function that checks if a point is stationary -/
def isStationary (p : Point3D) (m : Motion) : Prop := sorry

/-- A function that checks if a point describes a circle -/
def describesCircle (p : Point3D) (m : Motion) : Prop := sorry

/-- The main theorem -/
theorem body_motion_indeterminate 
  (body : RigidBody) 
  (p1 p2 : Point3D) 
  (h1 : p1 ∈ body.points) 
  (h2 : p2 ∈ body.points) 
  (h3 : isStationary p1 body.motion) 
  (h4 : describesCircle p2 body.motion) : 
  body.motion = Motion.Indeterminate := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_body_motion_indeterminate_l641_64136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_implies_a_range_l641_64171

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then (1/3) * x + 1 else Real.log x

theorem two_roots_implies_a_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f x - a * x = 0 ∧ f y - a * y = 0) →
  (∀ z : ℝ, z ≠ x ∧ z ≠ y → f z - a * z ≠ 0) →
  a > 1/3 ∧ a < 1/Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_implies_a_range_l641_64171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_farther_from_origin_B_farther_from_C_l641_64160

-- Define the points
def A : ℝ × ℝ := (0, -7)
def B : ℝ × ℝ := (-4, 0)
def C : ℝ × ℝ := (-4, -7)
def Origin : ℝ × ℝ := (0, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem for part a
theorem A_farther_from_origin :
  distance A Origin > distance B Origin := by sorry

-- Theorem for part b
theorem B_farther_from_C :
  distance B C > distance A C := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_farther_from_origin_B_farther_from_C_l641_64160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l641_64198

-- Define the function f(x) = 1 / (x - 1)
noncomputable def f (x : ℝ) : ℝ := 1 / (x - 1)

-- Theorem stating that f satisfies the given conditions
theorem f_satisfies_conditions :
  (f 0 * f 2 < 0) ∧ (∀ x : ℝ, x ≠ 1 → f x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l641_64198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_l641_64185

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (2 + x) / x
  else if x > 0 then Real.log (1 / x) / Real.log 2
  else 0  -- We define f(0) as 0, since it's not specified in the original problem

-- Define the solution set
def solution_set : Set ℝ :=
  Set.Icc (-2/3) 0 ∪ Set.Ici 4

-- State the theorem
theorem f_inequality_solution :
  {x : ℝ | f x + 2 ≤ 0} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_l641_64185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_l641_64106

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

theorem compound_interest_rate : 
  let principal_simple := 3500.000000000004
  let rate_simple := 6
  let time := 2
  let principal_compound := 4000
  simple_interest principal_simple rate_simple time = 
    0.5 * compound_interest principal_compound (10 : ℝ) time := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_l641_64106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_different_sets_l641_64191

def S : Set ℝ := {x | x^2 - 5 * |x| + 6 = 0}

def T (a : ℝ) : Set ℝ := {x | (a - 2) * x = 2}

theorem count_different_sets : 
  ∃ (A : Set ℝ), (∀ a, a ∈ A ↔ T a ≠ S) ∧ Finite A ∧ Nat.card A = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_different_sets_l641_64191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_in_college_proof_l641_64105

/-- The number of students studying at least one language in a college given the following conditions:
  * 325 students study Hindi
  * 385 students study Sanskrit
  * 480 students study English
  * 240 students study French
  * 115 students study Hindi and Physics
  * 175 students study Sanskrit and Chemistry
  * 210 students study English and History
  * 95 students study French and Mathematics
  * 140 students study Hindi and Sanskrit
  * 195 students study Sanskrit and English
  * 165 students study English and French
  * 110 students study Hindi and French
  * 75 students study all four languages
  * 35 students study Hindi, Sanskrit, and English
  * 45 students study Sanskrit, English, and French
  * 30 students study Hindi, English, and French
-/
def total_students_in_college : ℕ := 525

theorem total_students_in_college_proof :
  total_students_in_college = 325 + 385 + 480 + 240
  - 140 - 195 - 165 - 110 - 175 - 210 - 95
  + 35 + 45 + 30 + 75 - 75 := by
  sorry

#eval total_students_in_college -- Should output 525

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_in_college_proof_l641_64105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_open_interval_f_deriv_positive_l641_64176

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 + x - sin x

-- State the theorem
theorem f_increasing_on_open_interval :
  ∀ x y, 0 < x ∧ x < y ∧ y < 2 * π → f x < f y := by
  -- We'll prove this later
  sorry

-- Helper lemma for the derivative of f
lemma f_deriv (x : ℝ) : deriv f x = 1 - cos x := by
  -- We'll prove this later
  sorry

-- Theorem stating that f' is always positive on the open interval (0, 2π)
theorem f_deriv_positive :
  ∀ x, 0 < x ∧ x < 2 * π → 0 < deriv f x := by
  -- We'll prove this later
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_open_interval_f_deriv_positive_l641_64176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_case_l641_64154

/-- The chord length cut by a line on a circle -/
noncomputable def chord_length (a b c : ℝ) (x₀ y₀ r : ℝ) : ℝ :=
  2 * Real.sqrt (r^2 - ((a*x₀ + b*y₀ + c)^2 / (a^2 + b^2)))

theorem chord_length_specific_case : 
  chord_length 2 (-1) 2 2 2 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_case_l641_64154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_has_real_root_l641_64161

theorem polynomial_has_real_root
  (P : ℝ → ℝ)
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ)
  (h_nonzero : a₁ * a₂ * a₃ ≠ 0)
  (h_eq : ∀ x : ℝ, P (a₁ * x + b₁) + P (a₂ * x + b₂) = P (a₃ * x + b₃)) :
  ∃ r : ℝ, P r = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_has_real_root_l641_64161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_y_l641_64130

-- Define the function y as noncomputable
noncomputable def y (x : ℝ) : ℝ :=
  Real.tan (x + 2 * Real.pi / 3) - Real.tan (x + Real.pi / 6) + 
  Real.cos (x + Real.pi / 6) + Real.sin (x + Real.pi / 6)

-- State the theorem
theorem min_value_of_y :
  ∃ (min_y : ℝ), min_y = Real.sqrt 2 ∧
  ∀ (x : ℝ), -Real.pi/4 ≤ x ∧ x ≤ -Real.pi/6 → y x ≥ min_y :=
by
  -- The proof is omitted and replaced with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_y_l641_64130
