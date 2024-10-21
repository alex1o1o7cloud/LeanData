import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hope_encryption_l116_11634

/-- Represents the 26 letters of the English alphabet -/
inductive Letter : Type where
| A | B | C | D | E | F | G | H | I | J | K | L | M
| N | O | P | Q | R | S | T | U | V | W | X | Y | Z
deriving Inhabited

/-- Converts a character to a Letter -/
def charToLetter (c : Char) : Letter := 
  match c with
  | 'a' => Letter.A | 'b' => Letter.B | 'c' => Letter.C | 'd' => Letter.D
  | 'e' => Letter.E | 'f' => Letter.F | 'g' => Letter.G | 'h' => Letter.H
  | 'i' => Letter.I | 'j' => Letter.J | 'k' => Letter.K | 'l' => Letter.L
  | 'm' => Letter.M | 'n' => Letter.N | 'o' => Letter.O | 'p' => Letter.P
  | 'q' => Letter.Q | 'r' => Letter.R | 's' => Letter.S | 't' => Letter.T
  | 'u' => Letter.U | 'v' => Letter.V | 'w' => Letter.W | 'x' => Letter.X
  | 'y' => Letter.Y | 'z' => Letter.Z
  | _ => Letter.A  -- Default case

/-- Shifts a letter forward by n positions, wrapping around if necessary -/
def shiftLetter (l : Letter) (n : Nat) : Letter :=
  match l with
  | Letter.A => [Letter.A, Letter.B, Letter.C, Letter.D, Letter.E].get! n
  | Letter.B => [Letter.B, Letter.C, Letter.D, Letter.E, Letter.F].get! n
  | Letter.C => [Letter.C, Letter.D, Letter.E, Letter.F, Letter.G].get! n
  | Letter.D => [Letter.D, Letter.E, Letter.F, Letter.G, Letter.H].get! n
  | Letter.E => [Letter.E, Letter.F, Letter.G, Letter.H, Letter.I].get! n
  | Letter.F => [Letter.F, Letter.G, Letter.H, Letter.I, Letter.J].get! n
  | Letter.G => [Letter.G, Letter.H, Letter.I, Letter.J, Letter.K].get! n
  | Letter.H => [Letter.H, Letter.I, Letter.J, Letter.K, Letter.L].get! n
  | Letter.I => [Letter.I, Letter.J, Letter.K, Letter.L, Letter.M].get! n
  | Letter.J => [Letter.J, Letter.K, Letter.L, Letter.M, Letter.N].get! n
  | Letter.K => [Letter.K, Letter.L, Letter.M, Letter.N, Letter.O].get! n
  | Letter.L => [Letter.L, Letter.M, Letter.N, Letter.O, Letter.P].get! n
  | Letter.M => [Letter.M, Letter.N, Letter.O, Letter.P, Letter.Q].get! n
  | Letter.N => [Letter.N, Letter.O, Letter.P, Letter.Q, Letter.R].get! n
  | Letter.O => [Letter.O, Letter.P, Letter.Q, Letter.R, Letter.S].get! n
  | Letter.P => [Letter.P, Letter.Q, Letter.R, Letter.S, Letter.T].get! n
  | Letter.Q => [Letter.Q, Letter.R, Letter.S, Letter.T, Letter.U].get! n
  | Letter.R => [Letter.R, Letter.S, Letter.T, Letter.U, Letter.V].get! n
  | Letter.S => [Letter.S, Letter.T, Letter.U, Letter.V, Letter.W].get! n
  | Letter.T => [Letter.T, Letter.U, Letter.V, Letter.W, Letter.X].get! n
  | Letter.U => [Letter.U, Letter.V, Letter.W, Letter.X, Letter.Y].get! n
  | Letter.V => [Letter.V, Letter.W, Letter.X, Letter.Y, Letter.Z].get! n
  | Letter.W => [Letter.W, Letter.X, Letter.Y, Letter.Z, Letter.A].get! n
  | Letter.X => [Letter.X, Letter.Y, Letter.Z, Letter.A, Letter.B].get! n
  | Letter.Y => [Letter.Y, Letter.Z, Letter.A, Letter.B, Letter.C].get! n
  | Letter.Z => [Letter.Z, Letter.A, Letter.B, Letter.C, Letter.D].get! n

/-- Encrypts a single character by shifting it 4 positions forward -/
def encryptChar (c : Char) : Char :=
  match shiftLetter (charToLetter c) 4 with
  | Letter.A => 'a' | Letter.B => 'b' | Letter.C => 'c' | Letter.D => 'd'
  | Letter.E => 'e' | Letter.F => 'f' | Letter.G => 'g' | Letter.H => 'h'
  | Letter.I => 'i' | Letter.J => 'j' | Letter.K => 'k' | Letter.L => 'l'
  | Letter.M => 'm' | Letter.N => 'n' | Letter.O => 'o' | Letter.P => 'p'
  | Letter.Q => 'q' | Letter.R => 'r' | Letter.S => 's' | Letter.T => 't'
  | Letter.U => 'u' | Letter.V => 'v' | Letter.W => 'w' | Letter.X => 'x'
  | Letter.Y => 'y' | Letter.Z => 'z'

/-- Encrypts a string by encrypting each character -/
def encrypt (s : String) : String :=
  s.map encryptChar

theorem hope_encryption :
  encrypt "hope" = "lsti" := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hope_encryption_l116_11634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_line_circle_intersection_l116_11668

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := (a - 1) * x + 2 * y + 3 * a = 0
def l₂ (a x y : ℝ) : Prop := x + a * y + 6 = 0

-- Define the circle
def myCircle (x y : ℝ) : Prop := (x + 6)^2 + y^2 = 25

-- Theorem 1: l₁ ⊥ l₂ when a = 1/3
theorem perpendicular_lines :
  ∀ x y : ℝ, l₁ (1/3) x y ∧ l₂ (1/3) x y → ((-1/3 + 1) / 2) * (-3) = -1 :=
by
  sorry

-- Theorem 2: l₂ intersects the circle at two distinct points for all a
theorem line_circle_intersection :
  ∀ a : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ, 
    l₂ a x₁ y₁ ∧ myCircle x₁ y₁ ∧
    l₂ a x₂ y₂ ∧ myCircle x₂ y₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_line_circle_intersection_l116_11668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_theorem_l116_11657

theorem class_size_theorem (total_candies total_cookies total_apples : ℕ)
  (h_candies : total_candies = 300)
  (h_cookies : total_cookies = 210)
  (h_apples : total_apples = 163)
  (h_ratio : ∃ (r : ℕ), r > 0 ∧
    (total_candies % 23) = r ∧
    (total_cookies % 23) = 3 * r ∧
    (total_apples % 23) = 2 * r) :
  23 = 23 :=
by
  sorry

#check class_size_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_theorem_l116_11657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_shift_equals_h_l116_11688

noncomputable def g (x : ℝ) : ℝ :=
  if x < -1 then x + 3
  else if x ≤ 2 then -x^2 + 4
  else x - 2

noncomputable def h (x : ℝ) : ℝ :=
  if x < -3 then x + 5
  else if x ≤ 0 then -x^2 - 4*x
  else x

theorem g_shift_equals_h : ∀ x : ℝ, g (x + 2) = h x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_shift_equals_h_l116_11688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l116_11627

-- Define the bounds of the region
noncomputable def lower_bound : ℝ := 1/2
noncomputable def upper_bound : ℝ := 2

-- Define the function representing the curve
noncomputable def f (x : ℝ) : ℝ := 1/x

-- State the theorem
theorem area_of_region : 
  (∫ x in lower_bound..upper_bound, f x) = 2 * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l116_11627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pablo_stack_difference_l116_11630

/-- The height of Pablo's toy block stacks -/
structure BlockStacks where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- The conditions of Pablo's toy block stacks -/
def pablo_stacks (third : ℕ) : BlockStacks where
  first := 5
  second := 7  -- 5 + 2
  third := third
  fourth := third + 5

theorem pablo_stack_difference :
  ∀ (stacks : BlockStacks),
  stacks.first = 5 →
  stacks.second = stacks.first + 2 →
  stacks.fourth = stacks.third + 5 →
  stacks.first + stacks.second + stacks.third + stacks.fourth = 21 →
  stacks.second - stacks.third = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pablo_stack_difference_l116_11630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_hanoi_is_100_l116_11638

/-- Represents the distance from the starting point to Hanoi -/
def distance_to_hanoi : ℝ := sorry

/-- Represents the total travel time for all students -/
def total_travel_time : ℝ := sorry

/-- The speed of the motorcycle in km/h -/
def motorcycle_speed : ℝ := 50

/-- The speed of the bicycle in km/h -/
def bicycle_speed : ℝ := 10

/-- The time after which B switches to a bicycle -/
def switch_time : ℝ := 1.5

/-- The theorem stating the distance to Hanoi is 100 km -/
theorem distance_to_hanoi_is_100 :
  (bicycle_speed * (total_travel_time - switch_time) = distance_to_hanoi - motorcycle_speed * switch_time) ∧
  (motorcycle_speed * (total_travel_time - (switch_time + 1)) = distance_to_hanoi - (bicycle_speed * switch_time + bicycle_speed)) ∧
  (total_travel_time = 4) →
  distance_to_hanoi = 100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_hanoi_is_100_l116_11638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_survey_not_accurate_data_l116_11632

/-- Represents advantages of sampling surveys -/
inductive SamplingSurveyAdvantage
  | smallerScope
  | timeSaving
  | resourceSaving

/-- Represents characteristics of data quality -/
inductive DataQuality
  | accurate
  | approximate

/-- Properties of sampling surveys -/
structure SamplingSurvey where
  scope : SamplingSurveyAdvantage
  time : SamplingSurveyAdvantage
  resources : SamplingSurveyAdvantage
  dataQuality : DataQuality

/-- Theorem stating that obtaining accurate data is not an advantage of sampling surveys -/
theorem sampling_survey_not_accurate_data 
  (survey : SamplingSurvey) 
  (h1 : survey.scope = SamplingSurveyAdvantage.smallerScope)
  (h2 : survey.time = SamplingSurveyAdvantage.timeSaving)
  (h3 : survey.resources = SamplingSurveyAdvantage.resourceSaving) :
  survey.dataQuality = DataQuality.approximate :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_survey_not_accurate_data_l116_11632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l116_11609

/-- An arithmetic sequence -/
structure ArithmeticSequence (α : Type*) [AddCommGroup α] where
  a : ℕ → α
  d : α
  h : ∀ n, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence -/
def partialSum {α : Type*} [AddCommGroup α] [Field α] (seq : ArithmeticSequence α) (n : ℕ) : α :=
  (n : α) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_sum 
  {α : Type*} [AddCommGroup α] [Field α] (seq : ArithmeticSequence α) :
  seq.a 17 - 2 + seq.a 2000 = 1 → partialSum seq 2016 = 3024 :=
by
  sorry

#check arithmetic_sequence_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l116_11609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_case_l116_11600

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter and centroid
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ := sorry

noncomputable def centroid (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle at vertex A
noncomputable def angle_A (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_special_case (t : Triangle) :
  distance t.A (circumcenter t) = distance t.A (centroid t) →
  angle_A t = π / 3 ∨ angle_A t = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_case_l116_11600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_ratio_l116_11643

theorem cylinder_surface_ratio (a : ℝ) (h : a > 0) : 
  (4 * a^2) / (Real.pi * (a / Real.pi)^2) = 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_ratio_l116_11643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inradii_sum_equals_altitude_l116_11649

/-- Type representing a point in a plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate indicating that a point forms a right angle with two other points. -/
def RightAngle (A B C : Point) : Prop := sorry

/-- Predicate indicating that a point is the foot of the altitude from another point in a triangle. -/
def IsAltitude (A H B C : Point) : Prop := sorry

/-- Predicate indicating that a real number is the inradius of a triangle formed by three points. -/
def IsInradius (r : ℝ) (A B C : Point) : Prop := sorry

/-- Function to calculate the distance between two points. -/
def dist (A B : Point) : ℝ := sorry

/-- Given a right triangle ABC with right angle at A and H as the foot of the altitude from A,
    the sum of the inradii of triangles ABC, ABH, and AHC is equal to AH. -/
theorem inradii_sum_equals_altitude (A B C H : Point) (r_ABC r_ABH r_AHC : ℝ) :
  RightAngle A B C →
  IsAltitude A H B C →
  IsInradius r_ABC A B C →
  IsInradius r_ABH A B H →
  IsInradius r_AHC A H C →
  r_ABC + r_ABH + r_AHC = dist A H := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inradii_sum_equals_altitude_l116_11649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l116_11673

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ) - Real.sqrt 3 * Real.cos (ω * x + φ)

theorem function_properties
  (ω φ : ℝ)
  (h_ω : ω > 0)
  (h_φ : |φ| < π / 2)
  (h_symmetry : ∀ x, f ω φ x = f ω φ (π / 2 - x)) :
  (∃ T > 0, ∀ x, f ω φ (x + T) = f ω φ x) ∧
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < π / 2 → f ω φ x₁ < f ω φ x₂) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l116_11673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l116_11641

/-- A function f: ℝ → ℝ that is periodic with period 2 and equals x³ on [-1, 1) -/
noncomputable def f : ℝ → ℝ :=
  sorry

/-- The property that f is periodic with period 2 -/
axiom f_periodic (x : ℝ) : f (x + 2) = f x

/-- The property that f equals x³ on [-1, 1) -/
axiom f_cube (x : ℝ) (h : -1 ≤ x ∧ x < 1) : f x = x^3

/-- The function g(x) = f(x) - log_a|x| -/
noncomputable def g (a : ℝ) : ℝ → ℝ := 
  λ x ↦ f x - Real.log (abs x) / Real.log a

/-- The property that g has at least 6 zero points -/
def g_zero_points (a : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ), 
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧ x₁ ≠ x₆ ∧
    x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧ x₂ ≠ x₆ ∧
    x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧ x₃ ≠ x₆ ∧
    x₄ ≠ x₅ ∧ x₄ ≠ x₆ ∧
    x₅ ≠ x₆ ∧
    g a x₁ = 0 ∧ g a x₂ = 0 ∧ g a x₃ = 0 ∧ g a x₄ = 0 ∧ g a x₅ = 0 ∧ g a x₆ = 0

/-- The theorem stating the range of a -/
theorem range_of_a : 
  {a : ℝ | g_zero_points a} = {a : ℝ | 0 < a ∧ a ≤ 1/5 ∨ 5 < a} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l116_11641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_classroom_seating_theorem_l116_11698

/-- Represents the seating arrangement in a classroom --/
structure SeatingArrangement where
  total_students : Nat
  row_sizes : List Nat
  rows_filled_to_capacity : Bool

/-- Checks if a seating arrangement is valid --/
def is_valid_arrangement (arr : SeatingArrangement) : Prop :=
  arr.total_students = 55 ∧
  arr.row_sizes.all (λ size => size = 5 ∨ size = 8 ∨ size = 9) ∧
  arr.rows_filled_to_capacity

/-- Calculates the number of rows with 9 students --/
def count_nine_student_rows (arr : SeatingArrangement) : Nat :=
  arr.row_sizes.filter (· = 9) |>.length

/-- The main theorem to prove --/
theorem classroom_seating_theorem (arr : SeatingArrangement) :
  is_valid_arrangement arr →
  count_nine_student_rows arr = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_classroom_seating_theorem_l116_11698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l116_11670

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := y^2 - 4*x^2 = 16

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the foci of the hyperbola
noncomputable def focus1 : ℝ × ℝ := (-2*Real.sqrt 5, 0)
noncomputable def focus2 : ℝ × ℝ := (2*Real.sqrt 5, 0)

theorem hyperbola_focus_distance (x y : ℝ) :
  is_on_hyperbola x y →
  (distance x y focus1.1 focus1.2 = 2 ∨ distance x y focus2.1 focus2.2 = 2) →
  (distance x y focus1.1 focus1.2 = 10 ∨ distance x y focus2.1 focus2.2 = 10) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l116_11670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l116_11679

def triangle_proof (A B C : ℝ) (a b c : ℝ) : Prop :=
  let S := Real.sqrt 3 / 2  -- Area of the triangle
  (Real.sqrt 3 * b * Real.sin A - a * Real.cos B - 2 * a = 0) ∧ 
  (b = Real.sqrt 7) ∧ 
  (1 / 2 * a * c * Real.sin B = S) →
  (B = 2 * Real.pi / 3) ∧ 
  ((a = 1 ∧ c = 2) ∨ (a = 2 ∧ c = 1))

theorem triangle_theorem : ∀ A B C a b c, triangle_proof A B C a b c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l116_11679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_tetrahedron_volume_l116_11610

structure TrihedralAngle where
  -- Define properties of a trihedral angle
  mk :: -- placeholder

structure Point where
  -- Define properties of a point in 3D space
  mk :: -- placeholder

structure Plane where
  -- Define properties of a plane
  mk :: -- placeholder

def insideTrihedralAngle (p : Point) (angle : TrihedralAngle) : Prop :=
  sorry -- Define what it means for a point to be inside a trihedral angle

def intersectionTriangle (angle : TrihedralAngle) (plane : Plane) : Set Point :=
  sorry -- Define the triangle formed by the intersection of the trihedral angle and the plane

def centroid (triangle : Set Point) : Point :=
  sorry -- Define the centroid of a triangle

def tetrahedronVolume (angle : TrihedralAngle) (plane : Plane) : ℝ :=
  sorry -- Define the volume of the tetrahedron formed by the trihedral angle and the plane

instance : Membership Point Plane where
  mem := sorry -- Define membership of a point in a plane

theorem minimal_tetrahedron_volume 
  (angle : TrihedralAngle) 
  (p : Point) 
  (h : insideTrihedralAngle p angle) :
  ∀ (plane : Plane),
    p ∈ plane →
    (p = centroid (intersectionTriangle angle plane)) ↔
    (∀ (otherPlane : Plane),
      p ∈ otherPlane →
      tetrahedronVolume angle plane ≤ tetrahedronVolume angle otherPlane) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_tetrahedron_volume_l116_11610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_OAB_and_symmetric_circle_l116_11669

-- Define the Cartesian coordinate system
structure Point where
  x : ℝ
  y : ℝ

def O : Point := ⟨0, 0⟩
def A : Point := ⟨4, -3⟩

-- Define vector (removing the duplicate definition)
-- structure Vector where
--   x : ℝ
--   y : ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the dot product of two vectors
def dotProduct (v1 v2 : Point) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

-- Define the theorem
theorem triangle_OAB_and_symmetric_circle :
  ∃ (B : Point) (AB : Point),
    -- Conditions
    (distance O A)^2 = A.x^2 + A.y^2 ∧
    (distance A B)^2 = 4 * (distance O A)^2 ∧
    B.y > 0 ∧
    dotProduct A AB = 0 ∧
    -- Conclusions
    AB = ⟨6, 8⟩ ∧
    ∀ (x y : ℝ), (x - 1)^2 + (y - 3)^2 = 10 ↔
      ∃ (x' y' : ℝ), x'^2 - 6*x' + y'^2 + 2*y' = 0 ∧
        (x + x')/2 = (y + y')/4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_OAB_and_symmetric_circle_l116_11669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_second_quadrant_l116_11650

/-- Given acute angles A and B in an acute triangle ABC, 
    the point P(cos B - sin A, sin B - cos A) is in the second quadrant. -/
theorem point_in_second_quadrant (A B : ℝ) : 
  0 < A → A < π/2 → 
  0 < B → B < π/2 → 
  0 < A + B → A + B < π → 
  (Real.cos B - Real.sin A < 0) ∧ (Real.sin B - Real.cos A > 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_second_quadrant_l116_11650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_right_handed_players_l116_11692

theorem cricket_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (left_handed_non_thrower_percentage : ℚ) :
  total_players = 100 →
  throwers = 60 →
  left_handed_non_thrower_percentage = 40 / 100 →
  (∀ t : ℕ, t ≤ throwers → t ∈ Set.range (λ x : ℕ => x)) →
  ∃ right_handed : ℕ, right_handed = 84 ∧ right_handed = throwers + (total_players - throwers) * (1 - left_handed_non_thrower_percentage) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_right_handed_players_l116_11692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_tetrahedron_ABCD_l116_11672

/-- Tetrahedron ABCD with given properties -/
structure Tetrahedron where
  /-- Angle between faces ABC and BCD in radians -/
  angle_ABC_BCD : ℝ
  /-- Area of face ABC -/
  area_ABC : ℝ
  /-- Area of face BCD -/
  area_BCD : ℝ
  /-- Length of edge BC -/
  length_BC : ℝ

/-- The volume of a tetrahedron with the given properties -/
noncomputable def tetrahedron_volume (t : Tetrahedron) : ℝ := 375 * Real.sqrt 2

/-- Theorem stating that the volume of the tetrahedron ABCD with the given properties is 375√2 -/
theorem volume_of_tetrahedron_ABCD (t : Tetrahedron) 
  (h1 : t.angle_ABC_BCD = π / 4)  -- 45° in radians
  (h2 : t.area_ABC = 150)
  (h3 : t.area_BCD = 90)
  (h4 : t.length_BC = 12) :
  tetrahedron_volume t = 375 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_tetrahedron_ABCD_l116_11672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_4_or_6_l116_11612

/-- The set of numbers from 1 to 100 -/
def NumberSet : Finset Nat := Finset.range 100

/-- A number is divisible by 4 or 6 -/
def DivisibleBy4Or6 (n : Nat) : Prop := n % 4 = 0 ∨ n % 6 = 0

/-- DecidablePred instance for DivisibleBy4Or6 -/
instance : DecidablePred DivisibleBy4Or6 :=
  fun n => show Decidable (n % 4 = 0 ∨ n % 6 = 0) from inferInstance

theorem probability_divisible_by_4_or_6 : 
  (Finset.filter DivisibleBy4Or6 NumberSet).card / NumberSet.card = 33 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_4_or_6_l116_11612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cross_section_area_l116_11637

-- Define the prism
noncomputable def prism_side_length : ℝ := 8

-- Define the cutting plane
def cutting_plane (x y z : ℝ) : Prop := 3 * x - 5 * y + 2 * z = 20

-- Define the vertices of the base square
noncomputable def base_vertices : List (ℝ × ℝ × ℝ) :=
  [(4 * Real.sqrt 2, 0, 0),
   (0, 4 * Real.sqrt 2, 0),
   (-4 * Real.sqrt 2, 0, 0),
   (0, -4 * Real.sqrt 2, 0)]

-- Define the function to calculate the area of the cross-section
noncomputable def cross_section_area : ℝ := sorry

-- Theorem statement
theorem max_cross_section_area :
  ∃ (area : ℝ), 
    (∀ (x y z : ℝ), cutting_plane x y z → 
      cross_section_area ≤ area) ∧
    (∃ (x y z : ℝ), cutting_plane x y z ∧ 
      cross_section_area = area) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cross_section_area_l116_11637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l116_11655

theorem power_equation_solution (n : ℝ) : 
  (1/5 : ℝ)^(35 : ℕ) * (1/4 : ℝ)^n = 1/(2*(10 : ℝ)^(35 : ℕ)) → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l116_11655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_function_properties_l116_11652

/-- Correlation function of a stationary random function -/
noncomputable def correlation_function (D α : ℝ) (τ : ℝ) : ℝ :=
  D * Real.exp (-α * |τ|) * (1 + α * |τ|)

/-- Correlation function of the derivative of a stationary random function -/
noncomputable def correlation_function_derivative (D α : ℝ) (τ : ℝ) : ℝ :=
  D * α^2 * Real.exp (-α * |τ|) * (1 - α * |τ|)

/-- Theorem stating the properties of the correlation function and its derivative -/
theorem correlation_function_properties (D α : ℝ) (h_α : α > 0) (h_D : D > 0) :
  (∀ τ, correlation_function_derivative D α τ =
    D * α^2 * Real.exp (-α * |τ|) * (1 - α * |τ|)) ∧
  correlation_function_derivative D α 0 = D * α^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_function_properties_l116_11652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rahul_work_time_l116_11690

/-- Represents the number of days it takes Rahul to complete the work alone -/
def rahul_days : ℝ := 3

/-- Represents the total payment for the completed work -/
def total_payment : ℝ := 105

/-- Represents Rahul's share of the payment -/
def rahul_share : ℝ := 42

/-- Represents the number of days it takes Rajesh to complete the work alone -/
def rajesh_days : ℝ := 2

theorem rahul_work_time : rahul_days = 3 := by
  -- The proof goes here
  sorry

#check rahul_work_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rahul_work_time_l116_11690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l116_11699

/-- Represents a conic section -/
structure ConicSection where
  F₁ : ℝ × ℝ  -- First focus
  F₂ : ℝ × ℝ  -- Second focus

/-- Calculates the distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: Eccentricity of a conic section given specific distance ratios -/
theorem conic_section_eccentricity (C : ConicSection) (P : ℝ × ℝ) 
  (h : ∃ (m : ℝ), distance P C.F₁ = 4*m ∧ distance C.F₁ C.F₂ = 3*m ∧ distance P C.F₂ = 2*m) :
  ∃ (e : ℝ), (e = 1/2 ∨ e = 3/2) ∧ 
  (∀ (Q : ℝ × ℝ), distance Q C.F₁ + distance Q C.F₂ = (1/e) * distance C.F₁ C.F₂ ∨
                  |distance Q C.F₁ - distance Q C.F₂| = e * distance C.F₁ C.F₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l116_11699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_non_prime_sequence_l116_11642

theorem largest_non_prime_sequence :
  ∃ (a : ℕ),
    (a < 40) ∧
    (a > 9) ∧
    (∀ i : ℕ, i ∈ Finset.range 5 → ¬ Nat.Prime (a - i)) ∧
    (∀ b : ℕ, b > a → 
      ¬(∀ i : ℕ, i ∈ Finset.range 5 → (b - i < 40) ∧ (b - i > 9) ∧ ¬ Nat.Prime (b - i))) →
    a = 36 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_non_prime_sequence_l116_11642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_line_theorem_l116_11674

/-- A line passing through a point with a specific distance from a circle's center -/
structure SpecialLine where
  -- Point P
  p : ℝ × ℝ
  -- Circle C
  c : ℝ → ℝ → ℝ
  -- Center of circle C
  center : ℝ × ℝ
  -- Line L
  l : ℝ → ℝ → Prop
  -- Conditions
  point_on_line : l p.1 p.2
  distance_from_center : ∀ (x y : ℝ), l x y → (x - center.1)^2 + (y - center.2)^2 = 1

/-- The equations of the special line -/
def special_line_equations (sl : SpecialLine) : Prop :=
  ∀ x y, sl.l x y ↔ (3*x + 4*y = 6 ∨ x = 2)

/-- Theorem stating that the special line has the specified equations -/
theorem special_line_theorem (sl : SpecialLine) 
  (h1 : sl.p = (2, 0))
  (h2 : ∀ x y, sl.c x y = x^2 + y^2 - 6*x + 4*y + 4)
  (h3 : sl.center = (3, -2)) :
  special_line_equations sl :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_line_theorem_l116_11674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_l116_11683

theorem problem_1 : 9 + Real.rpow (-8) (1/3) - (-1)^3 = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_l116_11683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_w_range_l116_11686

noncomputable section

-- Define the function f
def f (w : ℝ) (x : ℝ) : ℝ := Real.sin (w * x + Real.pi / 4)

-- State the theorem
theorem w_range (w : ℝ) : 
  w > 0 → 
  (∀ x₁ x₂, Real.pi/2 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.pi → f w x₁ > f w x₂) → 
  w ∈ Set.Icc (1/2 : ℝ) (5/4 : ℝ) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_w_range_l116_11686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log2_domain_count_l116_11682

def B : Finset ℕ := Finset.filter (λ x => x ≤ 10) (Finset.range 11)

theorem log2_domain_count : 
  (Finset.filter (λ A : Finset ℕ => A.Nonempty ∧ A ⊆ B ∧ ∀ x ∈ A, x > 0) (Finset.powerset B)).card = 1023 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log2_domain_count_l116_11682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l116_11658

theorem binomial_expansion_coefficient (n : ℕ) : 
  (4^n / 2^n = 32) → 
  (∃ k : ℕ, (n.choose k) * 3^k * (5 - 4/3 * ↑k : ℚ) = 1) →
  (n.choose 3) * 3^3 = 270 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l116_11658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_value_l116_11661

noncomputable section

/-- A function f is symmetric about the y-axis if f(x) = f(-x) for all x -/
def SymmetricAboutYAxis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The given function after transformation -/
def f (φ : ℝ) (x : ℝ) : ℝ :=
  Real.sin (2 * x - 2 * Real.pi / 3 + φ)

/-- Theorem stating that if f is symmetric about the y-axis, then φ = π/6 -/
theorem symmetry_implies_phi_value (φ : ℝ) :
  SymmetricAboutYAxis (f φ) → φ = Real.pi / 6 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_value_l116_11661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_properties_l116_11624

-- Define a triangle with angles A, B, and C
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = Real.pi

-- Theorem statement
theorem triangle_trig_properties (t : Triangle) : 
  Real.sin (t.B + t.C) = Real.sin t.A ∧ Real.cos (t.B + t.C) ≠ Real.cos t.A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_properties_l116_11624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_of_sequence_l116_11697

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def geometric_mean (x y : ℝ) : ℝ :=
  Real.sqrt (x * y)

theorem geometric_mean_of_sequence :
  ∀ a : ℕ → ℝ,
  a 1 = 1 →
  (∀ n : ℕ, a (n + 1) = a n + 2) →
  (geometric_mean (a 1) (a 5))^2 = 9 :=
by
  sorry

#check geometric_mean_of_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_of_sequence_l116_11697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_sqrt_l116_11620

-- Define the function f(x) = √x
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- State the theorem
theorem derivative_of_sqrt (x : ℝ) (h : x > 0) :
  deriv f x = 1 / (2 * Real.sqrt x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_sqrt_l116_11620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_pqr_properties_l116_11693

/-- Triangle PQR with point S on the angle bisector of ∠QPR -/
structure TrianglePQR where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  pq_length : dist P Q = 8
  qr_length : dist Q R = 15
  pr_length : dist P R = 17
  right_angle : (P.1 - Q.1) * (P.1 - R.1) + (P.2 - Q.2) * (P.2 - R.2) = 0
  s_on_bisector : (S.1 - P.1) * (Q.2 - P.2) = (S.2 - P.2) * (Q.1 - P.1)

/-- The length of QS is 4.8 and the altitude from P to QS is 25 -/
theorem triangle_pqr_properties (t : TrianglePQR) : 
  dist t.Q t.S = 4.8 ∧ 
  abs ((t.P.1 - t.Q.1) * (t.S.2 - t.Q.2) - (t.P.2 - t.Q.2) * (t.S.1 - t.Q.1)) / 
    Real.sqrt ((t.S.1 - t.Q.1)^2 + (t.S.2 - t.Q.2)^2) = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_pqr_properties_l116_11693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l116_11691

/-- Given two functions that intersect at two points, prove their sum of x-coordinates of vertices -/
theorem intersection_sum (a b c d : ℝ) : 
  (∀ x, -(abs (x - a)) + b = 2 * (abs (x - c)) + d → x = 1 ∨ x = 7) →
  -(abs (1 - a)) + b = 4 →
  -(abs (7 - a)) + b = 2 →
  2 * (abs (1 - c)) + d = 4 →
  2 * (abs (7 - c)) + d = 2 →
  a + c = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l116_11691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_value_function_m_range_l116_11615

noncomputable section

/-- A function is a mean value function on [a, b] if there exist x₁, x₂ in (a, b) such that
    f'(x₁) = f'(x₂) = (f(b) - f(a)) / (b - a) -/
def IsMeanValueFunction (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b ∧
    (deriv f x₁ = (f b - f a) / (b - a)) ∧
    (deriv f x₂ = (f b - f a) / (b - a))

/-- The function f(x) = (1/3)x³ - (1/2)x² + m -/
def f (m : ℝ) : ℝ → ℝ := λ x ↦ (1/3) * x^3 - (1/2) * x^2 + m

theorem mean_value_function_m_range :
  ∀ m : ℝ, IsMeanValueFunction (f m) 0 m ↔ 3/4 < m ∧ m < 3/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_value_function_m_range_l116_11615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_properties_l116_11681

-- Define the correlation coefficient
def correlation_coefficient : ℝ → ℝ := λ r ↦ r

-- Define the absolute value of the correlation coefficient
def abs_correlation_coefficient : ℝ → ℝ := λ r ↦ |r|

-- Define the degree of correlation
noncomputable def degree_of_correlation : ℝ → ℝ := sorry

-- State the theorem
theorem correlation_coefficient_properties :
  ∀ r : ℝ,
  (abs_correlation_coefficient r ≤ 1) ∧
  (∀ ε > 0, ε < 1 →
    degree_of_correlation (1 - ε) > degree_of_correlation (1 - 2*ε)) ∧
  (∀ ε > 0, ε < 1 →
    degree_of_correlation ε < degree_of_correlation (2*ε)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_properties_l116_11681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_wickets_one_over_l116_11604

/-- Represents the maximum number of wickets a bowler can take in a given number of overs -/
def max_wickets (overs : ℕ) : ℕ := sorry

/-- Axiom: The maximum number of wickets in 6 overs is 10 -/
axiom max_wickets_6_overs : max_wickets 6 = 10

/-- Axiom: The number of wickets in any number of overs is non-negative -/
axiom non_negative_wickets (n : ℕ) : max_wickets n ≥ 0

/-- Axiom: The maximum number of wickets in more overs is at least as much as in fewer overs -/
axiom monotone_wickets (m n : ℕ) : m ≤ n → max_wickets m ≤ max_wickets n

/-- Theorem: The maximum number of wickets a bowler can take in one over is 6 -/
theorem max_wickets_one_over : max_wickets 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_wickets_one_over_l116_11604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_ratio_equality_l116_11662

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the point structure
structure Point where
  x : ℝ
  y : ℝ

-- Define the line structure
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the distance function between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define membership for Point in Circle
def Point.inCircle (p : Point) (c : Circle) : Prop :=
  distance p (Point.mk c.center.1 c.center.2) = c.radius

-- Define membership for Point on Line
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define the IsTangentLine predicate
def IsTangentLine (l : Line) (c : Circle) (p : Point) : Prop :=
  p.onLine l ∧ p.inCircle c ∧ ∀ q : Point, q.onLine l → q.inCircle c → q = p

-- Define the theorem
theorem tangent_line_ratio_equality 
  (circle1 circle2 : Circle) 
  (B C A₁ A₂ : Point) 
  (l : Line) 
  (h1 : B.inCircle circle1 ∧ B.inCircle circle2)
  (h2 : C.inCircle circle1 ∧ C.inCircle circle2)
  (h3 : A₁.onLine l ∧ A₁.inCircle circle1)
  (h4 : A₂.onLine l ∧ A₂.inCircle circle2)
  (h5 : IsTangentLine l circle1 A₁)
  (h6 : IsTangentLine l circle2 A₂) :
  distance A₁ B / distance A₁ C = distance A₂ B / distance A₂ C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_ratio_equality_l116_11662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soup_weight_on_fourth_day_l116_11639

noncomputable def soup_weight (initial_weight : ℝ) (days : ℕ) : ℝ :=
  initial_weight / (2 ^ days)

theorem soup_weight_on_fourth_day :
  soup_weight 80 4 = 5 := by
  -- Unfold the definition of soup_weight
  unfold soup_weight
  -- Simplify the expression
  simp [pow_succ]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_soup_weight_on_fourth_day_l116_11639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_intersection_l116_11695

noncomputable def f (x : ℝ) : ℝ := (x^2 - 6*x + 8) / (x^2 - 6*x + 9)

theorem asymptotes_intersection :
  ∃ (x y : ℝ), x = 3 ∧ y = 1 ∧
  (∀ ε > 0, ∃ δ > 0, ∀ t : ℝ, |t - x| < δ → |f t - y| < ε) ∧
  (∀ M > 0, ∃ N : ℝ, ∀ t : ℝ, |t| > N → |f t - y| < M) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_intersection_l116_11695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cassie_brian_meeting_time_meeting_at_10_30_l116_11656

/-- Represents the time in hours after 8:15 AM -/
def time (t : ℝ) : ℝ := t

/-- Cassie's departure time (8:15 AM) -/
def cassie_start : ℝ := 0

/-- Brian's departure time (9:00 AM, which is 0.75 hours after 8:15 AM) -/
def brian_start : ℝ := 0.75

/-- Cassie's speed in miles per hour -/
def cassie_speed : ℝ := 14

/-- Brian's speed in miles per hour -/
def brian_speed : ℝ := 18

/-- Total distance between Escanaba and Marquette in miles -/
def total_distance : ℝ := 62

/-- The time when Cassie and Brian meet -/
noncomputable def meeting_time : ℝ := (total_distance + brian_speed * brian_start) / (cassie_speed + brian_speed)

theorem cassie_brian_meeting_time :
  meeting_time = 2.25 := by sorry

/-- 2.25 hours after 8:15 AM is 10:30 AM -/
theorem meeting_at_10_30 :
  time meeting_time = time 2.25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cassie_brian_meeting_time_meeting_at_10_30_l116_11656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_equals_negative_one_l116_11678

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Define for 0 to cover all natural numbers
  | (n + 1) => 1 / (1 - sequence_a n)

theorem a_2018_equals_negative_one : sequence_a 2018 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_equals_negative_one_l116_11678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_cube_l116_11603

theorem perfect_cube (x y z : ℝ) (h : x + y + z = 0) :
  (x^2*y^2 + y^2*z^2 + z^2*x^2) * (x^3*y*z + x*y^3*z + x*y*z^3) * 
  (x^3*y^2*z + x^3*y*z^2 + x^2*y^3*z + x*y^3*z^2 + x^2*y*z^3 + x*y^2*z^3) = 
  (6 : ℝ)^(1/3) * (x * y * z * (x * y + y * z + z * x))^3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_cube_l116_11603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l116_11647

-- Define the points A and B
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (3, 3)

-- Define a point P on the x-axis
def P (x : ℝ) : ℝ × ℝ := (x, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem min_sum_distances :
  ∃ (x : ℝ), ∀ (y : ℝ), 
    distance (P x) A + distance (P x) B ≤ distance (P y) A + distance (P y) B ∧
    distance (P x) A + distance (P x) B = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l116_11647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equals_two_thirds_l116_11687

/-- The repeating decimal 0.666... as a real number -/
noncomputable def repeating_decimal : ℝ := ∑' n, 6 / 10^(n + 1)

/-- Theorem: The repeating decimal 0.666... is equal to 2/3 -/
theorem repeating_decimal_equals_two_thirds : repeating_decimal = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equals_two_thirds_l116_11687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_weight_is_14_l116_11626

/-- The weight of one kayak in pounds -/
noncomputable def kayak_weight : ℚ := 35

/-- The number of kayaks that weigh the same as the bowling balls -/
def num_kayaks : ℕ := 4

/-- The number of bowling balls that weigh the same as the kayaks -/
def num_bowling_balls : ℕ := 10

/-- The weight of one bowling ball in pounds -/
noncomputable def bowling_ball_weight : ℚ := (kayak_weight * num_kayaks) / num_bowling_balls

theorem bowling_ball_weight_is_14 :
  bowling_ball_weight = 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_weight_is_14_l116_11626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_red_two_blue_is_four_sevenths_l116_11671

/-- The number of red marbles in the bag -/
def num_red : ℕ := 15

/-- The number of blue marbles in the bag -/
def num_blue : ℕ := 9

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := num_red + num_blue

/-- The number of marbles drawn -/
def num_drawn : ℕ := 4

/-- The probability of drawing exactly two red and two blue marbles -/
noncomputable def prob_two_red_two_blue : ℚ := 
  (Nat.choose num_red 2 * Nat.choose num_blue 2) / Nat.choose total_marbles num_drawn

theorem prob_two_red_two_blue_is_four_sevenths : prob_two_red_two_blue = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_red_two_blue_is_four_sevenths_l116_11671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_cows_count_l116_11685

/-- The number of cows after one year given the initial number of cows -/
noncomputable def cows_after_one_year (initial_cows : ℝ) : ℝ := 
  initial_cows + (1/2) * initial_cows

/-- The number of cows after two years given the initial number of cows -/
noncomputable def cows_after_two_years (initial_cows : ℝ) : ℝ := 
  cows_after_one_year (cows_after_one_year initial_cows)

/-- Theorem stating that if there are 450 cows after two years, 
    then there were initially 200 cows -/
theorem initial_cows_count : 
  cows_after_two_years 200 = 450 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_cows_count_l116_11685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l116_11684

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  semiMajorAxis : ℝ
  semiMinorAxis : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the distance between foci of an ellipse -/
noncomputable def distanceBetweenFoci (e : Ellipse) : ℝ :=
  2 * Real.sqrt (e.semiMajorAxis^2 - e.semiMinorAxis^2)

/-- Theorem: The distance between foci of the given ellipse is 2√33 -/
theorem ellipse_foci_distance :
  ∀ (e : Ellipse),
    (∃ (p1 p2 p3 : Point),
      p1 = ⟨1, 5⟩ ∧ p2 = ⟨6, 1⟩ ∧ p3 = ⟨9, 5⟩ ∧
      (distance e.center p1 = e.semiMajorAxis ∨ distance e.center p2 = e.semiMajorAxis ∨ distance e.center p3 = e.semiMajorAxis) ∧
      (distance e.center p1 = e.semiMinorAxis ∨ distance e.center p2 = e.semiMinorAxis ∨ distance e.center p3 = e.semiMinorAxis)) →
    (∃ (p : Point), distance e.center p = e.semiMinorAxis ∧ e.semiMinorAxis = e.semiMajorAxis + 3) →
    distanceBetweenFoci e = 2 * Real.sqrt 33 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l116_11684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_is_36_l116_11616

/-- A rectangle with midpoints forming a rhombus -/
structure MidpointRhombusRectangle where
  length : ℚ
  width : ℚ
  area : ℚ
  ab_twice_bc : length = 2 * width
  rectangle_area : area = length * width

/-- The area of the rhombus formed by the midpoints of the rectangle -/
def rhombus_area (r : MidpointRhombusRectangle) : ℚ :=
  r.length * r.width / 2

/-- Theorem: The area of the rhombus is 36 given the conditions -/
theorem rhombus_area_is_36 (r : MidpointRhombusRectangle) 
  (h : r.area = 72) : rhombus_area r = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_is_36_l116_11616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_and_eccentricity_l116_11677

/-- Hyperbola C with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0

/-- Line l with slope m and y-intercept c -/
structure Line where
  m : ℝ
  c : ℝ

/-- The line l passes through the right focus F of hyperbola C and intersects C at only one point -/
def line_tangent_to_hyperbola (C : Hyperbola) (l : Line) : Prop :=
  l.m = Real.sqrt 3 ∧ l.c = -4 * Real.sqrt 3 ∧ C.b / C.a = Real.sqrt 3

/-- Focal length of a hyperbola -/
noncomputable def focal_length (C : Hyperbola) : ℝ := 2 * Real.sqrt (C.a^2 + C.b^2)

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (C : Hyperbola) : ℝ := Real.sqrt (C.a^2 + C.b^2) / C.a

theorem hyperbola_focal_length_and_eccentricity (C : Hyperbola) (l : Line) :
  line_tangent_to_hyperbola C l → focal_length C = 6 ∧ eccentricity C = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_and_eccentricity_l116_11677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l116_11625

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- Ensure positive side lengths
  (A > 0 ∧ B > 0 ∧ C > 0) →  -- Ensure positive angles
  (A + B + C = Real.pi) →    -- Sum of angles in a triangle is π
  (Real.tan B = (Real.sqrt 3 * a * c) / (a^2 + c^2 - b^2)) →
  (B = Real.pi / 3 ∨ B = 2 * Real.pi / 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l116_11625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l116_11696

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos (x + Real.pi/3)

theorem f_properties :
  let f := f
  ∀ k : ℤ,
    (∀ x ∈ Set.Icc (Real.pi/12 + k*Real.pi) (7*Real.pi/12 + k*Real.pi), 
      ∀ y ∈ Set.Icc (Real.pi/12 + k*Real.pi) (7*Real.pi/12 + k*Real.pi), 
      x ≤ y → f x ≥ f y) ∧
    (∀ x ∈ Set.Icc (-5*Real.pi/12 + k*Real.pi) (Real.pi/12 + k*Real.pi), 
      ∀ y ∈ Set.Icc (-5*Real.pi/12 + k*Real.pi) (Real.pi/12 + k*Real.pi), 
      x ≤ y → f x ≤ f y) ∧
    (∀ x ∈ Set.Icc 0 (Real.pi/2), f x ≥ -Real.sqrt 3) ∧
    (f (Real.pi/2) = -Real.sqrt 3) ∧
    (∀ x ∈ Set.Icc 0 (Real.pi/2), f x ≤ 1 - Real.sqrt 3 / 2) ∧
    (f (Real.pi/12) = 1 - Real.sqrt 3 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l116_11696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ellipse_trajectory_l116_11653

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the constant perimeter
def constantPerimeter (t : Triangle) : Prop :=
  ∃ p : ℝ, dist t.A t.B + dist t.B t.C + dist t.C t.A = p

-- Define AB = 6
def sideAB_is_6 (t : Triangle) : Prop :=
  dist t.A t.B = 6

-- Define the minimum cos C
def min_cos_C (t : Triangle) : Prop :=
  ∃ P : ℝ × ℝ, t.C = P ∧ (t.A.1 - t.C.1) * (t.B.1 - t.C.1) + (t.A.2 - t.C.2) * (t.B.2 - t.C.2) = 7/25 * dist t.A t.C * dist t.B t.C

-- Define the ellipse equation
def on_ellipse (p : ℝ × ℝ) : Prop :=
  p.1^2 / 25 + p.2^2 / 16 = 1 ∧ p.2 ≠ 0

-- Define the line through A intersecting the ellipse
def line_through_A (A M N : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, M.2 = k * (M.1 - A.1) ∧ N.2 = k * (N.1 - A.1)

-- Main theorem
theorem triangle_ellipse_trajectory (t : Triangle) :
  constantPerimeter t →
  sideAB_is_6 t →
  min_cos_C t →
  (∀ p : ℝ × ℝ, p = t.C → on_ellipse p) ∧
  (∀ M N : ℝ × ℝ, on_ellipse M → on_ellipse N → line_through_A t.A M N →
    ¬∃ min : ℝ, ∀ M' N' : ℝ × ℝ, on_ellipse M' → on_ellipse N' → line_through_A t.A M' N' →
      dist t.B M * dist t.B N ≥ min) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ellipse_trajectory_l116_11653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_proof_l116_11665

-- Define the inequalities p and q
def p (x : ℝ) : Prop := (x^3 - 4*x) / (2*x) ≤ 0
def q (x m : ℝ) : Prop := x^2 - (2*m + 1)*x + m^2 + m ≤ 0

-- Define the range of m
def m_range (m : ℝ) : Prop := (m ∈ Set.Icc (-2) (-1) ∪ Set.Ioo 0 1)

-- Theorem statement
theorem m_range_proof (m : ℝ) :
  (∀ x, q x m → p x) ∧ 
  (∃ x, p x ∧ ¬q x m) →
  m_range m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_proof_l116_11665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l116_11606

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- The left focus of an ellipse -/
noncomputable def leftFocus (e : Ellipse) : Point := ⟨-Real.sqrt (e.a^2 - e.b^2), 0⟩

/-- The top vertex of an ellipse -/
def topVertex (e : Ellipse) : Point := ⟨0, e.b⟩

/-- The right directrix of an ellipse -/
noncomputable def rightDirectrix (e : Ellipse) : Line := ⟨1, 0, e.a^2 / Real.sqrt (e.a^2 - e.b^2)⟩

/-- Checks if a ray reflects off a line parallel to another line -/
def reflectsParallel (incident : Line) (surface : Line) (reflected : Line) : Prop := sorry

/-- Checks if a circle is tangent to a line -/
def isTangent (center : Point) (radius : ℝ) (line : Line) : Prop := sorry

/-- Main theorem -/
theorem ellipse_properties (e : Ellipse) 
  (h_reflect : reflectsParallel 
    (Line.mk 0 1 (-e.b)) -- Incident ray through top vertex perpendicular to AF
    (rightDirectrix e) 
    (Line.mk (leftFocus e).x (topVertex e).y 0)) -- Line AF
  (h_tangent : isTangent 
    (Point.mk (e.b / 2) (-e.b / 2)) -- Center of circle through A, B, F
    (Real.sqrt 10 * e.b / 2) -- Radius of circle
    (Line.mk 3 (-1) 3)) -- Line 3x - y + 3 = 0
  : eccentricity e = Real.sqrt 2 / 2 ∧ e.a^2 = 2 ∧ e.b^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l116_11606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_is_60_l116_11611

noncomputable def distance_point_to_line (x y : ℝ) : ℝ := |y - 10|

noncomputable def distance_between_points (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

def is_valid_point (x y : ℝ) : Prop :=
  distance_point_to_line x y = 3 ∧
  distance_between_points x y 5 10 = 10

theorem sum_of_coordinates_is_60 :
  ∃ (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ),
    is_valid_point x1 y1 ∧
    is_valid_point x2 y2 ∧
    is_valid_point x3 y3 ∧
    is_valid_point x4 y4 ∧
    x1 + y1 + x2 + y2 + x3 + y3 + x4 + y4 = 60 := by
  sorry

#check sum_of_coordinates_is_60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_is_60_l116_11611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_rectangle_area_l116_11618

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + 4*x + y^2 - 6*y = 28

/-- The radius of the circle -/
noncomputable def circle_radius : ℝ := Real.sqrt 41

/-- The area of the rectangle circumscribing the circle -/
def rectangle_area : ℝ := 164

/-- Theorem stating that a rectangle circumscribing the given circle has an area of 164 square units -/
theorem circumscribed_rectangle_area :
  ∀ (x y : ℝ), circle_equation x y →
  ∃ (rect_width rect_height : ℝ),
    rect_width = 2 * circle_radius ∧
    rect_height = 2 * circle_radius ∧
    rect_width * rect_height = rectangle_area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_rectangle_area_l116_11618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l116_11644

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, Real.cos x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, 2 * Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - 1

def is_period (T : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def is_smallest_positive_period (T : ℝ) (f : ℝ → ℝ) : Prop :=
  T > 0 ∧ is_period T f ∧ ∀ S, 0 < S ∧ S < T → ¬ is_period S f

def monotone_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f y < f x

theorem f_properties :
  (is_smallest_positive_period Real.pi f) ∧
  (∀ k : ℤ, monotone_decreasing_on f (Set.Icc (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l116_11644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_conditions_l116_11602

/-- Profit function for the company -/
noncomputable def profit_function (x : ℝ) : ℝ := 19 - 24 / (x + 2) - (3/2) * x

/-- Theorem stating the conditions for maximum profit -/
theorem max_profit_conditions (a : ℝ) (h_a : a > 0) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ a → profit_function x ≤ profit_function (min a 2)) ∧
  (a ≥ 2 → profit_function 2 = profit_function (min a 2)) ∧
  (0 < a ∧ a < 2 → profit_function a = profit_function (min a 2)) := by
  sorry

#check max_profit_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_conditions_l116_11602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_OC_coordinates_l116_11676

noncomputable def A : ℝ × ℝ := (0, 1)
noncomputable def B : ℝ × ℝ := (-3, 4)
noncomputable def O : ℝ × ℝ := (0, 0)

def angle_bisector (O A B C : ℝ × ℝ) : Prop := 
  ∃ (t : ℝ), 0 < t ∧ t * ‖(A.1 - O.1, A.2 - O.2)‖ = ‖(B.1 - O.1, B.2 - O.2)‖

theorem OC_coordinates :
  ∀ (C : ℝ × ℝ),
    angle_bisector O A B C →
    ‖(C.1 - O.1, C.2 - O.2)‖ = 2 →
    C = (-Real.sqrt 10 / 5, 3 * Real.sqrt 10 / 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_OC_coordinates_l116_11676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_is_945_l116_11607

noncomputable section

-- Define the complex number on the right side of the equation
def rightSide : ℂ := -1 / Real.sqrt 2 + Complex.I / Real.sqrt 2

-- Define the property that a complex number is a solution to the equation
def isSolution (z : ℂ) : Prop := z^7 = rightSide

-- Define the angle of a complex number in radians
def angle (z : ℂ) : ℝ := Complex.arg z

-- Convert radians to degrees
def toDegrees (θ : ℝ) : ℝ := θ * 180 / Real.pi

-- Define the property that an angle is between 0° and 360°
def isValidAngle (θ : ℝ) : Prop := 0 ≤ toDegrees θ ∧ toDegrees θ < 360

-- Theorem statement
theorem sum_of_angles_is_945 :
  ∃ (z₁ z₂ z₃ z₄ z₅ z₆ z₇ : ℂ),
    (∀ i, i ∈ [z₁, z₂, z₃, z₄, z₅, z₆, z₇] → isSolution i ∧ isValidAngle (angle i)) ∧
    (∀ i j, i ∈ [z₁, z₂, z₃, z₄, z₅, z₆, z₇] → j ∈ [z₁, z₂, z₃, z₄, z₅, z₆, z₇] → i ≠ j → angle i ≠ angle j) ∧
    toDegrees (angle z₁ + angle z₂ + angle z₃ + angle z₄ + angle z₅ + angle z₆ + angle z₇) = 945 :=
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_is_945_l116_11607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_grains_approx_l116_11680

/-- Represents a type of rice with its properties -/
structure Rice where
  grains_per_cup : ℝ
  tablespoons_per_half_cup : ℝ
  teaspoons_per_tablespoon : ℝ

/-- Calculate the number of grains per teaspoon for a given type of rice -/
noncomputable def grains_per_teaspoon (r : Rice) : ℝ :=
  r.grains_per_cup / (2 * r.tablespoons_per_half_cup * r.teaspoons_per_tablespoon)

/-- The three types of rice with their properties -/
def basmati : Rice := ⟨490, 8, 3⟩
def jasmine : Rice := ⟨470, 7.5, 3.5⟩
def arborio : Rice := ⟨450, 9, 2.5⟩

/-- The average number of grains per teaspoon for the three types of rice -/
noncomputable def average_grains_per_teaspoon : ℝ :=
  (grains_per_teaspoon basmati + grains_per_teaspoon jasmine + grains_per_teaspoon arborio) / 3

/-- Theorem stating that the average number of grains per teaspoon is approximately 9.7333 -/
theorem average_grains_approx :
  |average_grains_per_teaspoon - 9.7333| < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_grains_approx_l116_11680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snail_return_time_l116_11645

/-- Represents a 90-degree turn direction -/
inductive Turn
  | Left
  | Right

/-- Represents a single segment of the snail's path -/
structure PathSegment where
  duration : ℕ  -- duration in 15-minute intervals
  turn : Turn

/-- Represents the entire path of the snail -/
def SnailPath := List PathSegment

/-- Checks if a path returns to the starting point -/
def returns_to_start (p : SnailPath) : Prop := sorry

/-- Calculates the total time of a path in 15-minute intervals -/
def total_time (p : SnailPath) : ℕ := sorry

theorem snail_return_time (p : SnailPath) :
  returns_to_start p → ∃ n : ℕ, total_time p = 4 * n := by
  sorry

#check snail_return_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snail_return_time_l116_11645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l116_11636

-- Define set A
def A : Set ℝ := {x | (x - 2) / (x + 1) ≥ 0}

-- Define set B
def B : Set ℝ := {y | 0 ≤ y ∧ y < 4}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Icc 2 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l116_11636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_at_latest_time_l116_11613

/-- The temperature function in degrees Celsius -/
def T (t : ℝ) : ℝ := -t^2 + 10*t + 60

/-- The latest time when the temperature is 80 degrees Celsius -/
noncomputable def latest_time : ℝ := 5 + Real.sqrt 5

/-- Theorem stating that the temperature at the latest time is 80°C and
    for all times after that, the temperature is less than 80°C -/
theorem temperature_at_latest_time : 
  T latest_time = 80 ∧ 
  ∀ t > latest_time, T t < 80 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_at_latest_time_l116_11613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_point_x_value_l116_11660

/-- A circle in the xy-plane with diameter endpoints (-5,0) and (25,0) -/
def my_circle : Set (ℝ × ℝ) :=
  {p | (p.1 - 10)^2 + p.2^2 = 225}

/-- The point (x,15) is on the circle -/
def point_on_circle (x : ℝ) : Prop :=
  (x, 15) ∈ my_circle

theorem circle_point_x_value :
  ∀ x : ℝ, point_on_circle x → x = 10 := by
  intro x h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_point_x_value_l116_11660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_B_H_Q_l116_11640

-- Define the basic geometric objects
variable (A B C H I P K Q : EuclideanSpace ℝ 2)

-- Define the conditions
variable (acute_triangle : IsAcute A B C)
variable (H_orthocenter : IsOrthocenter H A B C)
variable (I_incenter : IsIncenter I A B C)
variable (P_on_circumcircle : OnCircle P (circumcenter B C I) (dist (circumcenter B C I) B))
variable (P_on_AB : SegmentND A B P)
variable (P_not_B : P ≠ B)
variable (K_projection : IsProjection K H (Line.through A I))
variable (Q_reflection : IsReflection Q P K)

-- State the theorem
theorem collinear_B_H_Q : Collinear B H Q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_B_H_Q_l116_11640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_wave_travel_time_constant_l116_11664

-- Definitions
def wave_speed (L : ℝ) : ℝ := sorry
def wave_travel_time (L : ℝ) : ℝ := sorry

theorem spring_wave_travel_time_constant 
  (L₀ : ℝ) -- Initial unstretched length
  (L₁ : ℝ) -- Initial stretched length
  (t₁ : ℝ) -- Initial travel time
  (L₂ : ℝ) -- New stretched length
  (h₁ : L₁ > L₀) -- Initial stretch condition
  (h₂ : L₂ = 2 * L₁) -- New stretch is twice the initial
  (h₃ : ∀ L, L > L₀ → ∃ C, wave_speed L = C * L) -- Wave speed proportional to stretched length
  : wave_travel_time L₂ = t₁ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_wave_travel_time_constant_l116_11664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_hemisphere_radius_l116_11629

noncomputable def hemisphereVolume (r : ℝ) : ℝ := (2/3) * Real.pi * r^3

theorem smaller_hemisphere_radius :
  let R : ℝ := 3  -- radius of the original hemisphere
  let n : ℕ := 64  -- number of smaller hemispheres
  let V : ℝ := hemisphereVolume R  -- volume of the original hemisphere
  ∃ r : ℝ, 
    r > 0 ∧  -- radius is positive
    n * hemisphereVolume r = V ∧  -- total volume is conserved
    r = 3/4  -- the radius of each smaller hemisphere
  := by
    sorry

#check smaller_hemisphere_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_hemisphere_radius_l116_11629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_min_coefficient_x_squared_l116_11628

-- Part 1
def f (m n : ℕ) (x : ℝ) : ℝ := (1 + x)^m + (1 + x)^n

theorem sum_of_coefficients (m n : ℕ) (h : m = 7 ∧ n = 7) :
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ),
    f m n x = a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀ ∧
    a₀ + a₂ + a₄ + a₆ = 128 := by sorry

-- Part 2
theorem min_coefficient_x_squared (m n : ℕ) (h : m + n = 19) :
  ∃ (c : ℝ), c = (Nat.choose m 2 + Nat.choose n 2 : ℝ) ∧
    ∀ (m' n' : ℕ), m' + n' = 19 →
      (Nat.choose m' 2 + Nat.choose n' 2 : ℝ) ≥ c ∧
      c = 81 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_min_coefficient_x_squared_l116_11628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_two_heads_four_coins_l116_11654

theorem probability_at_least_two_heads_four_coins :
  (1 : ℚ) - (Nat.choose 4 0) * (1/2 : ℚ)^0 * (1/2 : ℚ)^4 - (Nat.choose 4 1) * (1/2 : ℚ)^1 * (1/2 : ℚ)^3 = 11/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_two_heads_four_coins_l116_11654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_seven_to_sixth_power_l116_11667

theorem cube_root_of_seven_to_sixth_power : (7 : ℝ) ^ ((1/3 : ℝ) * 6) = 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_seven_to_sixth_power_l116_11667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_imag_part_l116_11659

theorem complex_equation_imag_part :
  ∀ z : ℂ, (z + 1) * (2 - I) = 5 * I → z.im = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_imag_part_l116_11659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_team_win_rate_proof_l116_11694

theorem team_win_rate_proof (total_games : ℚ) (first_games : ℚ) (first_win_rate : ℚ) (overall_win_rate : ℚ) :
  total_games = 120 →
  first_games = 30 →
  first_win_rate = 2/5 →
  overall_win_rate = 7/10 →
  let remaining_games := total_games - first_games
  let first_wins := first_win_rate * first_games
  let total_wins := overall_win_rate * total_games
  let remaining_wins := total_wins - first_wins
  (remaining_wins / remaining_games) = 4/5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_team_win_rate_proof_l116_11694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_triangle_apothem_comparison_l116_11614

theorem rectangle_triangle_apothem_comparison :
  ∀ (l w t : ℝ),
  l > 0 ∧ w > 0 ∧ t > 0 →
  l * w = 2 * (2 * (l + w)) →
  (Real.sqrt 3 / 4) * t^2 = 2 * (3 * t) →
  w / 2 < (Real.sqrt 3 / 2 * t) / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_triangle_apothem_comparison_l116_11614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_cutting_theorem_l116_11608

/-- Represents the number of good operations in a rope-cutting process -/
def GoodOperations : ℕ → ℕ := sorry

/-- Represents the set of different rope lengths recorded during the process -/
def DifferentLengths : ℕ → Finset ℕ := sorry

/-- The initial length of the rope -/
def initialLength : ℕ := 2018

/-- The binary digit sum of a natural number -/
def binaryDigitSum : ℕ → ℕ := sorry

theorem rope_cutting_theorem :
  (∀ n : ℕ, n ≤ GoodOperations initialLength → n ≤ 2016) ∧
  (GoodOperations initialLength ≥ 6) ∧
  (∀ process1 process2 : ℕ → ℕ → ℕ,
    GoodOperations (process1 initialLength 0) = 6 ∧
    GoodOperations (process2 initialLength 0) = 6 →
    (DifferentLengths (process1 initialLength 0)).card =
    (DifferentLengths (process2 initialLength 0)).card) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_cutting_theorem_l116_11608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_minimized_at_80_optimal_batch_size_l116_11646

/-- Represents the cost function for product production --/
noncomputable def cost_function (x : ℝ) : ℝ := 800 / x + x / 8

/-- Theorem stating that the cost function is minimized at x = 80 --/
theorem cost_minimized_at_80 :
  ∀ x : ℝ, x > 0 → cost_function x ≥ cost_function 80 := by
  sorry

/-- Corollary stating that 80 is the optimal number of products per batch --/
theorem optimal_batch_size :
  ∃ x : ℕ, x > 0 ∧ ∀ y : ℕ, y > 0 → cost_function (x : ℝ) ≤ cost_function (y : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_minimized_at_80_optimal_batch_size_l116_11646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_luisa_mall_distance_l116_11689

/-- Represents Luisa's trip details and gas information -/
structure TripInfo where
  grocery_miles : ℕ
  pet_store_miles : ℕ
  home_miles : ℕ
  miles_per_gallon : ℕ
  cost_per_gallon : ℚ
  total_gas_cost : ℚ

/-- Calculates the miles driven to the mall based on the given trip information -/
def miles_to_mall (trip : TripInfo) : ℚ :=
  let total_gallons := trip.total_gas_cost / trip.cost_per_gallon
  let total_miles := total_gallons * trip.miles_per_gallon
  total_miles - (trip.grocery_miles + trip.pet_store_miles + trip.home_miles)

/-- Theorem stating that Luisa drove 6 miles to the mall -/
theorem luisa_mall_distance :
  let trip : TripInfo := {
    grocery_miles := 10,
    pet_store_miles := 5,
    home_miles := 9,
    miles_per_gallon := 15,
    cost_per_gallon := 7/2,
    total_gas_cost := 7
  }
  miles_to_mall trip = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_luisa_mall_distance_l116_11689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speaker_max_discount_l116_11633

/-- Represents the maximum discount percentage that can be offered on a product. -/
noncomputable def max_discount (cost_price marked_price : ℝ) (min_profit_margin : ℝ) : ℝ :=
  let profit_amount := min_profit_margin * cost_price
  let max_discount_amount := marked_price - cost_price - profit_amount
  (max_discount_amount / marked_price) * 100

/-- Theorem stating the maximum discount for the given scenario. -/
theorem speaker_max_discount :
  max_discount 600 900 0.05 = 30 := by
  sorry

-- Remove the #eval line as it's not computable
-- #eval max_discount 600 900 0.05

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speaker_max_discount_l116_11633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l116_11605

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (3 * a * x)
def g (k b : ℝ) (x : ℝ) : ℝ := k * x + b

-- State the theorem
theorem function_properties 
  (a : ℝ) 
  (k b : ℝ) 
  (h_k_nonzero : k ≠ 0) 
  (h_tangent : (deriv (f a)) 1 = Real.exp 1)
  (h_odd : ∀ x, g k b (-x) = -(g k b x))
  (h_above : ∀ x ∈ Set.Ioo (-2) 2, f a x > g k b x) :
  a = 1/3 ∧ 
  b = 0 ∧ 
  k ∈ Set.Icc (-(1/(2 * Real.exp 2))) (Real.exp 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l116_11605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_of_distances_l116_11663

-- Define the necessary structures
structure Point where
  x : ℝ
  y : ℝ

def Line (A B : Point) : Set Point :=
  {P : Point | ∃ t : ℝ, P.x = A.x + t * (B.x - A.x) ∧ P.y = A.y + t * (B.y - A.y)}

def SameSide (C D : Point) (l : Set Point) : Prop :=
  ∃ f : Point → ℝ, (∀ P ∈ l, f P = 0) ∧ f C * f D > 0

noncomputable def Distance (A B : Point) : ℝ :=
  Real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2)

def AngleBisector (A B C : Point) : Set Point :=
  {P : Point | Distance A P / Distance B P = Distance A C / Distance B C}

-- State the theorem
theorem equality_of_distances
  (A B C D E : Point)
  (h1 : SameSide C D (Line A B))
  (h2 : Distance A B = Distance A D + Distance B C)
  (h3 : E ∈ AngleBisector A B C ∩ AngleBisector B A D) :
  Distance C E = Distance D E := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_of_distances_l116_11663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_log_solutions_l116_11651

open Real

theorem sin_eq_log_solutions :
  ∃! (S : Set ℝ), (∀ x ∈ S, x > 0 ∧ sin x = log x) ∧ Finite S ∧ Nat.card S = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_log_solutions_l116_11651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_sine_curve_l116_11675

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x)

-- Define the theorem
theorem area_enclosed_by_sine_curve (x₁ x₂ : ℝ) :
  f x₁ = 1 →
  f x₂ = 1 →
  |x₁ - x₂| = 2 * π / 3 →
  (∫ (x : ℝ) in x₁..x₂, 1 - f x) = 2 * π / 3 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_sine_curve_l116_11675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lakshmi_share_calculation_l116_11621

/-- Represents the investment and time details of a partner -/
structure Partner where
  investment : ℚ
  monthsInvested : ℚ

/-- Calculates a partner's share of the gain based on their investment-months and total investment-months -/
def calculateShare (partner : Partner) (totalInvestmentMonths : ℚ) (totalGain : ℚ) : ℚ :=
  (partner.investment * partner.monthsInvested / totalInvestmentMonths) * totalGain

theorem lakshmi_share_calculation (x : ℚ) (annualGain : ℚ) : annualGain = 58000 → 
  let raman : Partner := { investment := x, monthsInvested := 12 }
  let lakshmi : Partner := { investment := 2 * x, monthsInvested := 6 }
  let muthu : Partner := { investment := 3 * x, monthsInvested := 4 }
  let gowtham : Partner := { investment := 4 * x, monthsInvested := 9 }
  let pradeep : Partner := { investment := 5 * x, monthsInvested := 1 }
  let totalInvestmentMonths := raman.investment * raman.monthsInvested + 
                               lakshmi.investment * lakshmi.monthsInvested + 
                               muthu.investment * muthu.monthsInvested + 
                               gowtham.investment * gowtham.monthsInvested + 
                               pradeep.investment * pradeep.monthsInvested
  calculateShare lakshmi totalInvestmentMonths annualGain = (12 / 77) * 58000 := by
  sorry

#eval (12 : ℚ) / 77 * 58000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lakshmi_share_calculation_l116_11621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_container_height_who_water_container_height_l116_11631

/-- Proves that for a cuboid-shaped container with a given bottom area and volume, 
    the height can be calculated. -/
theorem container_height 
  (bottom_area : ℝ) 
  (volume : ℝ) 
  (h_area_positive : bottom_area > 0) 
  (h_volume_positive : volume > 0) :
  volume / bottom_area = 40 :=
by
  sorry

/-- Calculates the height of a cuboid-shaped water container -/
noncomputable def calculate_container_height 
  (bottom_area : ℝ) 
  (volume : ℝ) 
  (h_area_positive : bottom_area > 0) 
  (h_volume_positive : volume > 0) : ℝ :=
volume / bottom_area

/-- Proves that the calculated height for the given specifications is 40 cm -/
theorem who_water_container_height :
  calculate_container_height 50 2000 (by norm_num) (by norm_num) = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_container_height_who_water_container_height_l116_11631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_C₂_C₃_max_distance_AB_l116_11623

-- Define the curves
noncomputable def C₁ (t α : ℝ) : ℝ × ℝ := (t * Real.cos α, t * Real.sin α)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := 
  (2 * Real.sin θ * Real.cos θ, 2 * Real.sin θ * Real.sin θ)

noncomputable def C₃ (θ : ℝ) : ℝ × ℝ := 
  (2 * Real.sqrt 3 * Real.cos θ * Real.cos θ, 2 * Real.sqrt 3 * Real.cos θ * Real.sin θ)

-- Theorem for the intersection points of C₂ and C₃
theorem intersection_C₂_C₃ : 
  ∃ θ₁ θ₂ : ℝ, C₂ θ₁ = (0, 0) ∧ C₃ θ₂ = (0, 0) ∧ 
  ∃ θ₃ θ₄ : ℝ, C₂ θ₃ = (Real.sqrt 3 / 2, 3 / 2) ∧ C₃ θ₄ = (Real.sqrt 3 / 2, 3 / 2) :=
by sorry

-- Function to calculate the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem for the maximum value of |AB|
theorem max_distance_AB : 
  ∃ t₁ t₂ α : ℝ, t₁ ≠ 0 ∧ t₂ ≠ 0 ∧ 0 ≤ α ∧ α < Real.pi ∧
  (∃ θ₁ θ₂ : ℝ, C₁ t₁ α = C₂ θ₁ ∧ C₁ t₂ α = C₃ θ₂) ∧
  (∀ s₁ s₂ β θ₃ θ₄ : ℝ, s₁ ≠ 0 → s₂ ≠ 0 → 0 ≤ β → β < Real.pi →
    C₁ s₁ β = C₂ θ₃ → C₁ s₂ β = C₃ θ₄ → 
    distance (C₁ t₁ α) (C₁ t₂ α) ≥ distance (C₁ s₁ β) (C₁ s₂ β)) ∧
  distance (C₁ t₁ α) (C₁ t₂ α) = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_C₂_C₃_max_distance_AB_l116_11623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interview_probability_l116_11619

/-- The probability of selecting two students from a group of 30,
    where 22 take German and 26 take Italian,
    such that at least one student takes German and at least one takes Italian. -/
theorem interview_probability (total : ℕ) (german : ℕ) (italian : ℕ) 
    (h_total : total = 30)
    (h_german : german = 22)
    (h_italian : italian = 26)
    (h_subset : german + italian ≥ total) : 
    (Nat.choose total 2 - (Nat.choose (total - german) 2 + Nat.choose (total - italian) 2)) / Nat.choose total 2 = 401 / 435 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interview_probability_l116_11619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_daughter_probability_l116_11601

-- Define the genotypes
inductive Allele | A | a
inductive XChromosome | XB | Xb
inductive YChromosome | Y

-- Define the parent genotypes
structure MotherGenotype where
  autosomal : Allele × Allele
  sexChromosome : XChromosome × XChromosome

structure FatherGenotype where
  autosomal : Allele × Allele
  sexChromosome : XChromosome × YChromosome

-- Define the probability of a normal daughter
def probabilityNormalDaughter (mother : MotherGenotype) (father : FatherGenotype) : ℚ :=
  3 / 4

-- Theorem statement
theorem normal_daughter_probability 
  (mother : MotherGenotype) 
  (father : FatherGenotype) 
  (h1 : mother.autosomal = (Allele.A, Allele.a))
  (h2 : mother.sexChromosome = (XChromosome.XB, XChromosome.Xb))
  (h3 : father.autosomal = (Allele.A, Allele.a))
  (h4 : father.sexChromosome = (XChromosome.XB, YChromosome.Y)) :
  probabilityNormalDaughter mother father = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_daughter_probability_l116_11601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l116_11648

open Real

theorem trigonometric_identities :
  (∀ θ : ℝ, θ = 75 * π / 180 → sin θ * cos θ = 1/4) ∧
  (∀ α β γ : ℝ, α = 10 * π / 180 ∧ β = 20 * π / 180 ∧ γ = 40 * π / 180 →
    sin α * cos β * cos γ = 1/8) ∧
  (∀ φ : ℝ, φ = 105 * π / 180 → tan φ = -2 - sqrt 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l116_11648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_sides_l116_11622

/-- Given vectors m and n, and function f as their dot product -/
noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (2 * x) + 2, Real.cos x)
noncomputable def n (x : ℝ) : ℝ × ℝ := (1, 2 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem to be proved -/
theorem max_sum_sides (t : Triangle) (h1 : t.a = Real.sqrt 3) (h2 : f t.A = 4) :
  t.b + t.c ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_sides_l116_11622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_quadratic_l116_11666

/-- Given a quadratic equation ax^2 + 30x + c = 0 with exactly one solution,
    where a + c = 41 and a < c, prove that a ≈ 6.525 and c ≈ 34.475 -/
theorem unique_solution_quadratic (a c : ℝ) : 
  (∃! x, a * x^2 + 30 * x + c = 0) →
  a + c = 41 →
  a < c →
  (abs (a - 6.525) < 0.001 ∧ abs (c - 34.475) < 0.001) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_quadratic_l116_11666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radii_ratio_theorem_l116_11635

/-- A triangular pyramid with specific properties -/
structure TriangularPyramid where
  /-- The angle between the first lateral face and the base -/
  α : Real
  /-- The angle between the second lateral face and the base -/
  β : Real
  /-- The angle between the third lateral face and the base -/
  γ : Real
  /-- The lateral faces have equal areas -/
  lateral_faces_equal : True

/-- The ratio of the radii of the inscribed and escribed spheres -/
noncomputable def sphere_radii_ratio (p : TriangularPyramid) : Real :=
  (3 - Real.cos p.α - Real.cos p.β - Real.cos p.γ) / (3 + Real.cos p.α + Real.cos p.β + Real.cos p.γ)

/-- Theorem stating the ratio of the radii of the inscribed and escribed spheres -/
theorem sphere_radii_ratio_theorem (p : TriangularPyramid) :
  let r := sphere_radii_ratio p
  ∃ (inscribed_radius escribed_radius : Real),
    inscribed_radius / escribed_radius = r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radii_ratio_theorem_l116_11635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l116_11617

/-- An isosceles triangle with two sides of lengths 3 and 7 has a perimeter of 17. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  ((a = 3 ∧ b = 7 ∧ c = 7) ∨ (a = 7 ∧ b = 3 ∧ c = 7) ∨ (a = 7 ∧ b = 7 ∧ c = 3)) →
  a + b + c = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l116_11617
