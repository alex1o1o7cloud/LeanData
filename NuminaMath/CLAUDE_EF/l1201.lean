import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volunteer_assignment_count_l1201_120111

/-- Represents a volunteer --/
inductive Volunteer : Type
  | A | B | C | D | E
  deriving BEq, Repr

/-- Represents a group assignment --/
def GroupAssignment := List (List Volunteer)

/-- Check if a group assignment is valid --/
def isValidAssignment (assignment : GroupAssignment) : Bool :=
  (assignment.length = 3) &&
  (assignment.all (λ group => group.length > 0)) &&
  !(assignment.any (λ group => group.contains Volunteer.A && group.contains Volunteer.B)) &&
  !(assignment.any (λ group => group.contains Volunteer.C && group.contains Volunteer.D))

/-- All possible group assignments --/
def allAssignments : List GroupAssignment := sorry

/-- Count of valid assignments --/
def validAssignmentCount : Nat :=
  (allAssignments.filter isValidAssignment).length

theorem volunteer_assignment_count : validAssignmentCount = 288 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volunteer_assignment_count_l1201_120111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_base_length_for_specific_trapezoid_l1201_120173

/-- Represents a trapezoid ABCD with point E on BC -/
structure Trapezoid where
  /-- Length of upper base AD -/
  ad : ℝ
  /-- Ratio of BE to DE -/
  be_de_ratio : ℝ
  /-- BE = be_de_ratio * DE -/
  be_de_prop : be_de_ratio > 0

/-- The length of the lower base BC in the trapezoid -/
noncomputable def lower_base_length (t : Trapezoid) : ℝ :=
  t.ad * (1 + t.be_de_ratio) / (t.be_de_ratio / (1 + t.be_de_ratio))

theorem lower_base_length_for_specific_trapezoid :
  ∃ t : Trapezoid, t.ad = 12 ∧ t.be_de_ratio = 2 ∧ lower_base_length t = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_base_length_for_specific_trapezoid_l1201_120173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_has_extremum_l1201_120177

noncomputable def f (x : ℝ) : ℝ := x + 2 / x

-- Statement to prove
theorem f_is_odd_and_has_extremum :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∃ x₀ : ℝ, ∀ x : ℝ, f x₀ ≤ f x ∨ f x₀ ≥ f x) :=
by
  constructor
  · -- Proof that f is odd
    intro x
    simp [f]
    ring
  · -- Proof that f has an extremum
    use (2 : ℝ).sqrt
    intro x
    -- The minimum occurs at x = √2
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_has_extremum_l1201_120177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_n_sided_angle_angles_l1201_120193

/-- 
Given a regular n-sided angle where edges form angles of φ with its axis of symmetry,
this theorem proves the relationships for the plane angle α and dihedral angle γ.
-/
theorem regular_n_sided_angle_angles 
  (n : ℕ) 
  (φ : ℝ) 
  (h_n : n ≥ 3) 
  (h_φ : 0 < φ ∧ φ < π/2) :
  ∃ (α γ : ℝ),
    (0 < α ∧ α < π) ∧
    (0 < γ ∧ γ < π) ∧
    Real.sin (α/2) = Real.sin (π/n) * Real.sin φ ∧
    Real.sin (γ/2) = Real.sin (2*π/n) * Real.sin φ / Real.sin α :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_n_sided_angle_angles_l1201_120193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_fourth_blue_faces_iff_s_eq_four_l1201_120127

/-- Represents a regular tetrahedron with edge length s -/
structure Tetrahedron where
  s : ℝ
  s_pos : s > 0

/-- The number of unit tetrahedra obtained by cutting a tetrahedron with edge length s -/
noncomputable def num_unit_tetrahedra (t : Tetrahedron) : ℝ :=
  t.s^3 / 6

/-- The total number of faces of all unit tetrahedra -/
noncomputable def total_faces (t : Tetrahedron) : ℝ :=
  4 * num_unit_tetrahedra t

/-- The number of blue faces (faces on the surface of the original tetrahedron) -/
noncomputable def blue_faces (t : Tetrahedron) : ℝ :=
  4 * (t.s * (t.s - 1)) / 2

/-- The theorem stating that one-fourth of the total faces are blue iff s = 4 -/
theorem one_fourth_blue_faces_iff_s_eq_four (t : Tetrahedron) :
  blue_faces t / total_faces t = 1/4 ↔ t.s = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_fourth_blue_faces_iff_s_eq_four_l1201_120127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_monotonous_interval_l1201_120147

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ) + 1

noncomputable def g (ω φ : ℝ) (x : ℝ) : ℝ := f ω φ x + Real.cos (2 * x) - 1

noncomputable def h (ω φ : ℝ) (x : ℝ) : ℝ := g ω φ (x - Real.pi/4)

theorem max_monotonous_interval 
  (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : -Real.pi/2 < φ ∧ φ < Real.pi/2) 
  (h_period : ∀ x, f ω φ (x + Real.pi) = f ω φ x) 
  (h_point : f ω φ 0 = 1) :
  (∀ m : ℝ, (∀ x y, 0 < x ∧ x < y ∧ y < m → h ω φ x < h ω φ y) → m ≤ 3*Real.pi/8) ∧
  (∃ δ > 0, ∀ x y, 0 < x ∧ x < y ∧ y < 3*Real.pi/8 - δ → h ω φ x < h ω φ y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_monotonous_interval_l1201_120147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_geometric_sequence_sin_l1201_120114

theorem no_geometric_sequence_sin (a : Real) : 
  0 < a ∧ a < 2 * Real.pi → 
  ¬(∃r : Real, Real.sin (2 * a) = r * Real.sin a ∧ Real.sin (3 * a) = r * Real.sin (2 * a)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_geometric_sequence_sin_l1201_120114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_slope_l1201_120181

-- Define the points on the original line
def point1 : ℚ × ℚ := (3, -4)
def point2 : ℚ × ℚ := (-6, 2)

-- Define the slope of the original line
noncomputable def original_slope : ℚ := (point2.2 - point1.2) / (point2.1 - point1.1)

-- Define the slope of the perpendicular line
noncomputable def perpendicular_slope : ℚ := -1 / original_slope

-- Theorem statement
theorem perpendicular_line_slope :
  perpendicular_slope = 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_slope_l1201_120181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_heads_probability_l1201_120119

/-- Probability of getting heads for an unfair coin -/
noncomputable def p_heads : ℝ := 3/4

/-- Number of coin tosses -/
def n_tosses : ℕ := 40

/-- Probability of getting an odd number of heads after n tosses -/
noncomputable def P_odd (n : ℕ) : ℝ := 1/2 * (1 - 1/(4^n))

/-- Theorem: The probability of getting an odd number of heads
    after 40 tosses of an unfair coin with 3/4 probability of heads
    is 1/2(1 - 1/4^40) -/
theorem odd_heads_probability :
  P_odd n_tosses = 1/2 * (1 - 1/(4^n_tosses)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_heads_probability_l1201_120119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_tangent_line_from_origin_l1201_120150

noncomputable section

/-- The parabola defined by y = -x^2 + 9/2 * x - 4 -/
def parabola (x : ℝ) : ℝ := -x^2 + 9/2 * x - 4

/-- A point is in the first quadrant if both its x and y coordinates are positive -/
def in_first_quadrant (p : ℝ × ℝ) : Prop := 0 < p.1 ∧ 0 < p.2

/-- A line y = kx is tangent to the parabola at point (x, y) -/
def is_tangent_line (k : ℝ) (x y : ℝ) : Prop :=
  y = parabola x ∧ y = k * x ∧ (k - 9/2)^2 = 16

/-- There exists a unique tangent line from the origin with slope 1/2 -/
theorem unique_tangent_line_from_origin :
  ∃! k : ℝ, ∃ x y : ℝ,
    is_tangent_line k x y ∧
    in_first_quadrant (x, y) ∧
    k = 1/2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_tangent_line_from_origin_l1201_120150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_from_distances_l1201_120148

open Real

/-- An equilateral triangle with its inscribed circle and a point inside the circle -/
structure TriangleWithPoint where
  -- The side length of the equilateral triangle
  side : ℝ
  -- The radius of the inscribed circle
  r : ℝ
  -- The coordinates of the center of the inscribed circle
  I : ℝ × ℝ
  -- The coordinates of a point inside the inscribed circle
  P : ℝ × ℝ
  -- The distances from P to the sides of the triangle
  d₁ : ℝ
  d₂ : ℝ
  d₃ : ℝ
  -- Ensure P is inside the inscribed circle
  h_inside : (P.1 - I.1)^2 + (P.2 - I.2)^2 < r^2
  -- Ensure the distances are positive
  h_positive : d₁ > 0 ∧ d₂ > 0 ∧ d₃ > 0
  -- Ensure the sum of distances equals the semiperimeter
  h_sum : d₁ + d₂ + d₃ = side * Real.sqrt 3 / 2

/-- The theorem to be proved -/
theorem triangle_from_distances (t : TriangleWithPoint) :
  Real.sqrt t.d₁ + Real.sqrt t.d₂ > Real.sqrt t.d₃ ∧ 
  Real.sqrt t.d₁ + Real.sqrt t.d₃ > Real.sqrt t.d₂ ∧ 
  Real.sqrt t.d₂ + Real.sqrt t.d₃ > Real.sqrt t.d₁ ∧
  (Real.sqrt t.d₁ + Real.sqrt t.d₂ + Real.sqrt t.d₃) * (Real.sqrt t.d₂ + Real.sqrt t.d₃ - Real.sqrt t.d₁) * 
  (Real.sqrt t.d₁ + Real.sqrt t.d₃ - Real.sqrt t.d₂) * (Real.sqrt t.d₁ + Real.sqrt t.d₂ - Real.sqrt t.d₃) / 16 = 
  3 * Real.sqrt 3 / 16 * Real.sqrt (t.r^2 - ((t.P.1 - t.I.1)^2 + (t.P.2 - t.I.2)^2)) := by
  sorry

#check triangle_from_distances

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_from_distances_l1201_120148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_270_complex_l1201_120128

/-- Represents a 270° counter-clockwise rotation of a complex number -/
def rotate_270 (z : ℂ) : ℂ := -Complex.I * z

/-- The original complex number -/
def original : ℂ := -8 + 2*Complex.I

/-- The expected result after rotation -/
def expected_result : ℂ := -2 + 8*Complex.I

theorem rotation_270_complex :
  rotate_270 original = expected_result := by
  -- Expand the definitions and simplify
  simp [rotate_270, original, expected_result]
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_270_complex_l1201_120128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1201_120102

noncomputable section

variable (f : ℝ → ℝ)

axiom f_domain : ∀ x : ℝ, x > 0 → ∃ y : ℝ, f x = y

axiom f_equation : ∀ x : ℝ, x > 0 → f x - 2 * x * f (1 / x) + 3 * x^2 = 0

theorem f_minimum_value :
  ∃ m : ℝ, (∀ x : ℝ, x > 0 → f x ≥ m) ∧ (∃ x : ℝ, x > 0 ∧ f x = m) ∧ m = 3 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1201_120102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_sin_2theta_l1201_120143

noncomputable def f (x : ℝ) := Real.sqrt 2 * Real.cos (x + Real.pi/4)

theorem f_range_and_sin_2theta :
  (∀ x ∈ Set.Icc (-Real.pi/2) (Real.pi/2), f x ∈ Set.Icc (-1) (Real.sqrt 2)) ∧
  (∀ θ ∈ Set.Ioo 0 (Real.pi/2), f θ = 1/2 → Real.sin (2*θ) = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_sin_2theta_l1201_120143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_career_preference_theorem_l1201_120117

/-- Represents a class with male and female students -/
structure MyClass where
  maleCount : ℕ
  femaleCount : ℕ
  maleToFemaleRatio : (maleCount : ℚ) / femaleCount = 2 / 3

/-- Represents a career preference in the class -/
structure CareerPreference (c : MyClass) where
  angleDegrees : ℕ
  maleRatio : ℚ
  femaleRatio : ℚ
  angleProportionality : (angleDegrees : ℚ) / 360 = (maleRatio * c.maleCount + femaleRatio * c.femaleCount) / (c.maleCount + c.femaleCount)

/-- Main theorem: If a career preference occupies 144 degrees and is preferred by 1/4 of males, then 1/2 of females prefer it -/
theorem career_preference_theorem (c : MyClass) (p : CareerPreference c) 
    (h1 : p.angleDegrees = 144) 
    (h2 : p.maleRatio = 1 / 4) : 
  p.femaleRatio = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_career_preference_theorem_l1201_120117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l1201_120184

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  16 * x^2 - 64 * x - 4 * y^2 + 8 * y - 11 = 0

/-- The distance between vertices of the hyperbola -/
noncomputable def vertex_distance : ℝ := Real.sqrt 71 / 2

/-- Theorem stating that the distance between vertices of the given hyperbola is √71/2 -/
theorem hyperbola_vertex_distance :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola_eq x₁ y₁ ∧ 
    hyperbola_eq x₂ y₂ ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = vertex_distance^2 ∧
    ∀ (x y : ℝ), hyperbola_eq x y → 
      (x - x₁)^2 + (y - y₁)^2 ≤ vertex_distance^2 ∧
      (x - x₂)^2 + (y - y₂)^2 ≤ vertex_distance^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l1201_120184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_year_increase_l1201_120157

/-- Represents the price of a stock over three years -/
structure StockPrice where
  initial : ℚ
  year1 : ℚ
  year2 : ℚ
  year3 : ℚ

/-- Calculates the percentage change between two values -/
def percentChange (old : ℚ) (new : ℚ) : ℚ :=
  (new - old) / old * 100

/-- Theorem stating the percentage increase in the third year -/
theorem third_year_increase (sp : StockPrice) 
  (h1 : sp.year1 = sp.initial * (1 + 1/5))
  (h2 : sp.year2 = sp.year1 * (1 - 1/4))
  (h3 : sp.year3 = sp.initial * (1 + 27/25))
  : percentChange sp.year2 sp.year3 = 20 := by
  sorry

#eval percentChange 90 108

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_year_increase_l1201_120157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_two_same_color_proof_l1201_120113

def red_marbles : ℕ := 6
def white_marbles : ℕ := 7
def blue_marbles : ℕ := 8
def green_marbles : ℕ := 4

def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles + green_marbles

def probability_at_least_two_same_color : ℚ :=
  69 / 115

theorem probability_at_least_two_same_color_proof :
  (Nat.choose total_marbles 3 : ℚ) * probability_at_least_two_same_color =
    (Nat.choose red_marbles 2 * (total_marbles - red_marbles) +
     Nat.choose white_marbles 2 * (total_marbles - white_marbles) +
     Nat.choose blue_marbles 2 * (total_marbles - blue_marbles) +
     Nat.choose green_marbles 2 * (total_marbles - green_marbles) +
     Nat.choose red_marbles 3 + Nat.choose white_marbles 3 +
     Nat.choose blue_marbles 3 + Nat.choose green_marbles 3 : ℚ) :=
by sorry

#eval probability_at_least_two_same_color

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_two_same_color_proof_l1201_120113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l1201_120107

noncomputable def z : ℂ := 15 * Complex.I / (3 + 4 * Complex.I)

theorem imaginary_part_of_z : z.im = 9 / 5 := by
  -- Convert complex division to multiplication by conjugate
  have h1 : z = 15 * Complex.I * (3 - 4 * Complex.I) / ((3 + 4 * Complex.I) * (3 - 4 * Complex.I)) := by sorry
  
  -- Simplify numerator and denominator
  have h2 : z = (45 * Complex.I - 60) / 25 := by sorry
  
  -- Extract imaginary part
  have h3 : z.im = 45 / 25 := by sorry
  
  -- Simplify fraction
  calc
    z.im = 45 / 25 := h3
    _    = 9 / 5   := by norm_num

  -- The proof is complete

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l1201_120107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octahedron_volume_l1201_120194

/-- A regular octahedron described around a sphere -/
structure RegularOctahedron (R : ℝ) where
  /-- The octahedron is regular -/
  is_regular : True
  /-- The octahedron is described around a sphere of radius R -/
  sphere_radius : ℝ := R
  /-- The octahedron has three mutually perpendicular diagonals -/
  has_perpendicular_diagonals : True
  /-- The diagonals intersect at the center of the sphere -/
  diagonals_intersect_at_center : True

/-- The volume of a regular octahedron described around a sphere of radius R -/
noncomputable def volume (o : RegularOctahedron R) : ℝ := 4 * R^3 * Real.sqrt 3

/-- Theorem: The volume of a regular octahedron described around a sphere of radius R is 4R³√3 -/
theorem regular_octahedron_volume (R : ℝ) (o : RegularOctahedron R) :
  volume o = 4 * R^3 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octahedron_volume_l1201_120194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AOB_l1201_120195

noncomputable section

open Real

-- Define the curves
def C₁ (t : ℝ) : ℝ × ℝ := (sqrt 3 * t, sqrt (1 - t^2))

def C₂ (θ : ℝ) : ℝ := sqrt 3 / sin (θ + π/6)

-- Define the points
def A : ℝ × ℝ := C₁ (1/sqrt 2)

def B : ℝ × ℝ := (C₂ (π/3) * cos (π/3), C₂ (π/3) * sin (π/3))

def O : ℝ × ℝ := (0, 0)

-- State the theorem
theorem area_of_triangle_AOB :
  let S := (1/2) * abs (A.1 * B.2 - A.2 * B.1)
  S = sqrt 6 / 4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AOB_l1201_120195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_with_remainder_l1201_120103

theorem division_with_remainder (dividend : ℝ) (divisor : ℝ) (quotient : ℕ) (remainder : ℝ) : 
  dividend = divisor * (quotient : ℝ) + remainder → 
  0 ≤ remainder ∧ remainder < divisor →
  dividend = 76.6 →
  divisor = 1.8 →
  quotient = 42 →
  remainder = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_with_remainder_l1201_120103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_height_l1201_120165

/-- Represents a cone with its vertex and base circumference on a sphere of radius 1 -/
structure ConeOnSphere where
  height : ℝ
  baseRadius : ℝ
  sphereConstraint : height^2 + baseRadius^2 = 2 * height

/-- The volume of the cone -/
noncomputable def coneVolume (c : ConeOnSphere) : ℝ :=
  (1/3) * Real.pi * c.baseRadius^2 * c.height

/-- Theorem stating that the height maximizing the cone's volume is 4/3 -/
theorem max_volume_height :
  ∃ (c : ConeOnSphere), ∀ (c' : ConeOnSphere), coneVolume c ≥ coneVolume c' ∧ c.height = 4/3 := by
  sorry

#check max_volume_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_height_l1201_120165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumference_linear_constants_in_circumference_r_is_variable_l1201_120179

/-- The circumference of a circle as a function of its radius -/
noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

/-- Theorem stating that the circumference is a linear function of the radius -/
theorem circumference_linear (r₁ r₂ : ℝ) :
  circumference (r₁ + r₂) = circumference r₁ + circumference r₂ := by
  sorry

/-- Theorem stating that 2 and π are constants in the circumference formula -/
theorem constants_in_circumference :
  ∃ (k : ℝ), ∀ (r : ℝ), circumference r = k * r := by
  sorry

/-- Theorem stating that r is a variable in the circumference formula -/
theorem r_is_variable :
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ circumference r₁ ≠ circumference r₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumference_linear_constants_in_circumference_r_is_variable_l1201_120179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeros_15_factorial_base_12_l1201_120132

theorem trailing_zeros_15_factorial_base_12 :
  ∃ k : ℕ, (12 ^ k : ℕ) ∣ Nat.factorial 15 ∧ 
    ¬((12 ^ (k + 1) : ℕ) ∣ Nat.factorial 15) ∧ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeros_15_factorial_base_12_l1201_120132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_integer_sum_of_powers_l1201_120159

theorem not_integer_sum_of_powers (k l : ℕ+) :
  ∃ M : ℕ+, ∀ n : ℕ+, n > M → ¬ ∃ m : ℤ, ((k : ℝ) + 1/2)^(n : ℝ) + ((l : ℝ) + 1/2)^(n : ℝ) = m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_integer_sum_of_powers_l1201_120159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_constant_term_l1201_120133

noncomputable def binomial_expansion (x : ℝ) (n : ℕ) : ℕ → ℝ := 
  fun r => (Nat.choose n r) * (3 * x) ^ (n - r) * (1 / (x * Real.sqrt x)) ^ r

theorem smallest_n_with_constant_term :
  ∀ n : ℕ, n > 0 → (
    (∃ r : ℕ, r ≤ n ∧ binomial_expansion x n r = 1) ↔ n ≥ 5
  ) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_constant_term_l1201_120133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_not_symmetric_about_one_l1201_120136

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (2 * x) / (1 + abs x)

-- Theorem stating the properties of f(x)
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ y, y ∈ Set.range f ↔ -2 < y ∧ y < 2) ∧  -- range of f is (-2, 2)
  (∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂) ∧  -- f is strictly increasing
  (∀ x, abs (f x) = abs (f (-x)))  -- |f(x)| is symmetric about y-axis
:= by sorry

-- Additional theorem to show that |f(x+1)| is not symmetric about x = 1
theorem f_not_symmetric_about_one :
  ¬(∀ x, abs (f (x + 1)) = abs (f (2 - x))) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_not_symmetric_about_one_l1201_120136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_leq_one_l1201_120172

theorem negation_of_sin_leq_one :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x₀ : ℝ, Real.sin x₀ ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_leq_one_l1201_120172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volleyball_team_selection_l1201_120112

/-- The number of ways to choose 7 starters from a volleyball team -/
theorem volleyball_team_selection (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) (quad_starters : ℕ) : 
  total_players = 18 →
  quadruplets = 4 →
  starters = 7 →
  quad_starters = 2 →
  (Nat.choose quadruplets quad_starters) * (Nat.choose (total_players - quadruplets) (starters - quad_starters)) = 12012 := by
  intro h1 h2 h3 h4
  sorry

#check volleyball_team_selection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volleyball_team_selection_l1201_120112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l1201_120125

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 2*x - 1/x^2

-- State the theorem
theorem tangent_line_at_one :
  let slope := f' 1
  let y_intercept := f 1 - slope * 1
  ∀ x y : ℝ, y = slope * x + y_intercept ↔ x - y + 1 = 0 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l1201_120125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_length_162_to_30_l1201_120168

/-- The number of terms in an arithmetic sequence -/
def arithmeticSequenceLength (a₁ aₙ d : ℤ) : ℕ :=
  (((aₙ - a₁) / d).toNat + 1)

/-- Theorem: The arithmetic sequence starting with 162, ending with 30,
    and having a common difference of -3 contains exactly 45 terms. -/
theorem arithmetic_sequence_length_162_to_30 :
  arithmeticSequenceLength 162 30 (-3) = 45 := by
  rw [arithmeticSequenceLength]
  norm_num
  rfl

#eval arithmeticSequenceLength 162 30 (-3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_length_162_to_30_l1201_120168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_l1201_120166

-- Define the points and vectors
def A : ℝ × ℝ := (0, -1)
def B : ℝ → ℝ × ℝ := λ x => (x, -3)
def M : ℝ × ℝ → Prop := λ p => ∃ x : ℝ, p = (x, (1/4) * x^2 - 2)

-- Define vector operations
def vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)
def dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
def parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, v = (k * w.1, k * w.2)

-- State the theorem
theorem trajectory_of_M (x y : ℝ) : 
  M (x, y) ↔ 
    (parallel (vec (x, y) (B x)) (vec (0, 0) A) ∧ 
     dot (vec (x, y) A) (vec A (B x)) = dot (vec (x, y) (B x)) (vec (B x) A) ∧
     y = (1/4) * x^2 - 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_l1201_120166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_journey_distance_l1201_120144

/-- A hiker's journey over three days -/
structure HikerJourney where
  day1_distance : ℚ
  day1_speed : ℚ
  day2_speed_increase : ℚ
  day3_speed : ℚ
  day3_time : ℚ

/-- Calculate the total distance of the hiker's journey -/
def total_distance (journey : HikerJourney) : ℚ :=
  let day1_time := journey.day1_distance / journey.day1_speed
  let day2_time := day1_time - 1
  let day2_speed := journey.day1_speed + journey.day2_speed_increase
  let day2_distance := day2_speed * day2_time
  let day3_distance := journey.day3_speed * journey.day3_time
  journey.day1_distance + day2_distance + day3_distance

/-- Theorem stating the total distance of the hiker's journey -/
theorem hiker_journey_distance :
  ∀ (journey : HikerJourney),
    journey.day1_distance = 18 ∧
    journey.day1_speed = 3 ∧
    journey.day2_speed_increase = 1 ∧
    journey.day3_speed = 5 ∧
    journey.day3_time = 6 →
    total_distance journey = 68 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_journey_distance_l1201_120144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_10_equals_3_pow_89_l1201_120126

def c : ℕ → ℕ
  | 0 => 3  -- Adding this case for Nat.zero
  | 1 => 3
  | 2 => 6
  | (n + 3) => c (n + 2) * c (n + 1)

theorem c_10_equals_3_pow_89 : c 10 = 3^89 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_10_equals_3_pow_89_l1201_120126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ab_value_l1201_120100

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -a * Real.log x + (a + 1) * x - (1/2) * x^2

/-- The theorem stating the maximum value of ab -/
theorem max_ab_value (a b : ℝ) (h_a : a > 0) 
  (h_f : ∀ x > 0, f a x ≥ -(1/2) * x^2 + a * x + b) : 
  a * b ≤ Real.exp 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ab_value_l1201_120100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_iff_l1201_120161

open Real

/-- The function g(x) defined on the positive real numbers -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (2*x - 1) * exp x - a * x^2 + a

/-- The statement that g is monotonically increasing on (0, +∞) -/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → g a x < g a y

/-- The main theorem: the range of a for which g is monotonically increasing -/
theorem g_monotone_increasing_iff :
  ∀ a : ℝ, is_monotone_increasing a ↔ a ≤ 2 * sqrt (exp 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_iff_l1201_120161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_properties_l1201_120170

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x
noncomputable def g (x : ℝ) : ℝ := (Real.log x) / x

-- State the theorem
theorem f_g_properties :
  -- Part 1: When a=1, f(x) has a minimum value of 1 at x=1
  (∀ x ∈ Set.Ioo 0 (Real.exp 1), f 1 x ≥ f 1 1) ∧ (f 1 1 = 1) ∧
  -- Part 2: When a=1, for all x in (0,e], f(x) > g(x) + 1/2
  (∀ x ∈ Set.Ioo 0 (Real.exp 1), f 1 x > g x + 1/2) ∧
  -- Part 3: The minimum value of f(x) is 3 when a = e^2
  (∀ x ∈ Set.Ioo 0 (Real.exp 1), f (Real.exp 2) x ≥ 3) ∧
  (∃ x ∈ Set.Ioo 0 (Real.exp 1), f (Real.exp 2) x = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_properties_l1201_120170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_when_a_zero_f_prime_geq_g_iff_a_leq_zero_l1201_120121

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * (a + 2) * x^2 + x

-- Define the derivative of f
noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := x^2 - (a + 2) * x + 1

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.exp 1 - Real.exp x / x

-- Theorem for part (1)
theorem min_slope_when_a_zero :
  ∀ x : ℝ, f_prime 0 x ≥ 0 ∧ ∃ y : ℝ, f_prime 0 y = 0 :=
by sorry

-- Theorem for part (2)
theorem f_prime_geq_g_iff_a_leq_zero :
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → f_prime a x ≥ g x) ↔ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_when_a_zero_f_prime_geq_g_iff_a_leq_zero_l1201_120121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_squared_in_cones_l1201_120109

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones -/
structure ConeConfiguration where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ

noncomputable def maxSphereRadiusSquared (config : ConeConfiguration) : ℝ :=
  let coneLength := Real.sqrt (config.cone1.baseRadius ^ 2 + config.cone1.height ^ 2)
  let sphereCenter := config.cone1.height - config.intersectionDistance
  (config.cone1.baseRadius * sphereCenter / coneLength) ^ 2

theorem max_sphere_radius_squared_in_cones :
  let config := ConeConfiguration.mk
    (Cone.mk 4 10)
    (Cone.mk 4 10)
    4
  maxSphereRadiusSquared config = 144 / 29 := by
  sorry

#eval (144 : Nat) + 29

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_squared_in_cones_l1201_120109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_correct_l1201_120189

-- Define the original function
noncomputable def original_function (x : ℝ) : ℝ := (1/2)^x

-- Define the inverse function
noncomputable def inverse_function (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- State the theorem
theorem inverse_function_correct (x : ℝ) (h : x > 0) :
  original_function (inverse_function x) = x ∧
  inverse_function (original_function x) = x :=
by
  sorry

#check inverse_function_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_correct_l1201_120189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_ratio_l1201_120108

/-- An ellipse with center at the origin and foci on the x-axis -/
structure Ellipse where
  a : ℝ
  c : ℝ
  h_a_pos : 0 < a
  h_c_lt_a : c < a

/-- A parabola with vertex at the origin and focus on the x-axis -/
structure Parabola where
  p : ℝ
  h_p_pos : 0 < p

/-- A line not perpendicular to the x-axis -/
structure Line where
  k : ℝ
  h_k_ne_zero : k ≠ 0

/-- The main theorem -/
theorem ellipse_parabola_intersection_ratio
  (C₁ : Ellipse) (C₂ : Parabola) (l : Line)
  (h_F₂ : C₁.c = C₂.p)
  (h_AF₁ : |Real.sqrt ((C₁.a - C₁.c)^2 + C₁.a^2 * (1 - (C₁.a - C₁.c)^2 / C₁.a^2)) + C₁.c| = 7/2)
  (h_AF₂ : |Real.sqrt ((C₁.a - C₁.c)^2 + C₁.a^2 * (1 - (C₁.a - C₁.c)^2 / C₁.a^2)) - C₁.c| = 5/2)
  (h_obtuse : C₁.a - C₁.c > 0) :
  let B := (C₁.a * Real.sqrt (1 + l.k^2), C₁.a * l.k * Real.sqrt (1 + l.k^2))
  let E := (-C₁.a * Real.sqrt (1 + l.k^2), -C₁.a * l.k * Real.sqrt (1 + l.k^2))
  let C := (C₂.p * (1 + l.k^2), 2 * C₂.p * l.k)
  let D := (C₂.p * (1 + l.k^2), -2 * C₂.p * l.k)
  let G := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  let H := ((B.1 + E.1) / 2, (B.2 + E.2) / 2)
  |B.1 - E.1| * |G.1 - C₁.c| / (|C.1 - D.1| * |H.1 - C₁.c|) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_ratio_l1201_120108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drone_altitude_l1201_120120

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- The park setup with points A, B, C, D, O, and H -/
structure ParkSetup where
  O : Point3D
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  H : Point3D

/-- The conditions of the problem -/
def validSetup (setup : ParkSetup) : Prop :=
  -- A is directly north of O
  setup.A.x = setup.O.x ∧ setup.A.y > setup.O.y ∧ setup.A.z = setup.O.z ∧
  -- B is directly west of O
  setup.B.y = setup.O.y ∧ setup.B.x < setup.O.x ∧ setup.B.z = setup.O.z ∧
  -- C is southeast of O at a 45-degree angle
  setup.C.x - setup.O.x = setup.O.y - setup.C.y ∧ setup.C.x > setup.O.x ∧ setup.C.y < setup.O.y ∧ setup.C.z = setup.O.z ∧
  -- D is directly east of O
  setup.D.y = setup.O.y ∧ setup.D.x > setup.O.x ∧ setup.D.z = setup.O.z ∧
  -- H is directly above O
  setup.H.x = setup.O.x ∧ setup.H.y = setup.O.y ∧ setup.H.z > setup.O.z ∧
  -- Distance CD = 190 m
  distance setup.C setup.D = 190 ∧
  -- Length of HC = 200 m
  distance setup.H setup.C = 200 ∧
  -- Length of HD = 160 m
  distance setup.H setup.D = 160

theorem drone_altitude (setup : ParkSetup) (h : validSetup setup) :
  distance setup.O setup.H = 30 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_drone_altitude_l1201_120120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_postulate_l1201_120123

-- Define the basic geometric objects
variable (Point Line : Type)

-- Define the notion of a point being on a line
variable (isOn : Point → Line → Prop)

-- Define the notion of two lines being parallel
variable (isParallel : Line → Line → Prop)

-- State the parallel postulate
theorem parallel_postulate 
  (L : Line) (P : Point) (h : ¬ isOn P L) :
  ∃! M : Line, isOn P M ∧ isParallel M L :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_postulate_l1201_120123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l1201_120130

/-- The curve equation -/
def curve (x y : ℝ) : Prop := x^2 - y - 2 * Real.log (Real.sqrt x) = 0

/-- The line equation -/
def line (x y : ℝ) : Prop := 4*x + 4*y + 1 = 0

/-- The distance function from a point to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (4*x + 4*y + 1) / Real.sqrt (4^2 + 4^2)

/-- The theorem stating the minimum distance -/
theorem min_distance_theorem :
  ∃ (x y : ℝ), curve x y ∧ 
  (∀ (x' y' : ℝ), curve x' y' → distance_to_line x y ≤ distance_to_line x' y') ∧
  distance_to_line x y = Real.sqrt 2 / 2 * (1 + Real.log 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l1201_120130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_is_reals_implies_m_range_range_is_reals_implies_m_range_l1201_120160

open Real

-- Define the function f as noncomputable due to the use of log
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := log (x^2 - 2*m*x + m + 2)

-- Statement for question 1
theorem domain_is_reals_implies_m_range (m : ℝ) :
  (∀ x, ∃ y, f m x = y) → -1 < m ∧ m < 2 := by
  sorry

-- Statement for question 2
theorem range_is_reals_implies_m_range (m : ℝ) :
  (∀ y, ∃ x, f m x = y) → m ≤ -1 ∨ m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_is_reals_implies_m_range_range_is_reals_implies_m_range_l1201_120160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_min_dot_product_achievable_l1201_120124

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + y = 2

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Define the dot product of two vectors from origin
def dot_product (x₁ y₁ x₂ y₂ : ℝ) : ℝ := x₁ * x₂ + y₁ * y₂

theorem min_dot_product :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    line_equation x₁ y₁ →
    line_equation x₂ y₂ →
    distance x₁ y₁ x₂ y₂ = Real.sqrt 2 →
    dot_product x₁ y₁ x₂ y₂ ≥ 3/2 :=
by sorry

theorem min_dot_product_achievable :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_equation x₁ y₁ ∧
    line_equation x₂ y₂ ∧
    distance x₁ y₁ x₂ y₂ = Real.sqrt 2 ∧
    dot_product x₁ y₁ x₂ y₂ = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_min_dot_product_achievable_l1201_120124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1201_120163

theorem constant_term_expansion : ∃ (f : ℝ → ℝ), 
  (∀ x : ℝ, x ≠ 0 → f x = (x - 1/x) * (2*x + 1/x)^5) ∧ 
  (∃ c : ℝ, c = -40 ∧ 
    ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → |f x - c| < ε) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1201_120163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_p_value_l1201_120149

noncomputable section

-- Define the ellipse and parabola
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1
def parabola (x y p : ℝ) : Prop := x^2 = 2*p*y

-- Define the intersection points A and B
def intersection_points (p : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    ellipse x1 y1 ∧ parabola x1 y1 p ∧
    ellipse x2 y2 ∧ parabola x2 y2 p ∧
    (x1 ≠ x2 ∨ y1 ≠ y2)

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the point N
def N : ℝ × ℝ := (0, 13/2)

-- Define the circumcenter condition
def circumcenter_on_ellipse (p : ℝ) : Prop :=
  ∃ (x y : ℝ), ellipse x y ∧ 
    (∀ (a b : ℝ × ℝ), (ellipse a.1 a.2 ∧ parabola a.1 a.2 p) →
                       (ellipse b.1 b.2 ∧ parabola b.1 b.2 p) →
                       (a ≠ b) →
                       (x - a.1)^2 + (y - a.2)^2 = (x - b.1)^2 + (y - b.2)^2)

-- Define the circumcircle condition
def circumcircle_through_N (p : ℝ) : Prop :=
  ∃ (x y : ℝ), 
    (∀ (a b : ℝ × ℝ), (ellipse a.1 a.2 ∧ parabola a.1 a.2 p) →
                       (ellipse b.1 b.2 ∧ parabola b.1 b.2 p) →
                       (a ≠ b) →
                       (x - a.1)^2 + (y - a.2)^2 = (x - b.1)^2 + (y - b.2)^2) ∧
    (x - N.1)^2 + (y - N.2)^2 = (x - origin.1)^2 + (y - origin.2)^2

-- Theorem statement
theorem unique_p_value :
  ∀ p : ℝ, p > 0 →
    intersection_points p →
    (circumcenter_on_ellipse p ∨ circumcircle_through_N p) →
    p = 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_p_value_l1201_120149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1201_120191

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := x + x / (x^2 + 1) + (x * (x + 5)) / (x^2 + 3) + (3 * (x + 2)) / (x * (x^2 + 3))

/-- Theorem stating that f(x) ≥ 1 for all x > 0 -/
theorem f_minimum_value (x : ℝ) (hx : x > 0) : f x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1201_120191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1201_120198

noncomputable section

open Real

theorem problem_1 (α β : ℝ) 
  (h1 : Real.sin α = 3/5) 
  (h2 : Real.cos β = 4/5) 
  (h3 : α ∈ Set.Ioo (π/2) π) 
  (h4 : β ∈ Set.Ioo 0 (π/2)) : 
  Real.cos (α + β) = -1 := by sorry

theorem problem_2 (α β : ℝ) 
  (h1 : Real.cos α = 1/7) 
  (h2 : Real.cos (α - β) = 13/14) 
  (h3 : 0 < β) (h4 : β < α) (h5 : α < π/2) : 
  β = π/3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1201_120198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_primes_product_five_times_sum_l1201_120129

theorem three_primes_product_five_times_sum : 
  ∃! (a b c : ℕ), 
    Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧
    a * b * c = 5 * (a + b + c) ∧
    Finset.toSet {a, b, c} = {2, 5, 7} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_primes_product_five_times_sum_l1201_120129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_diagonals_validity_invalid_external_diagonals_l1201_120162

/-- Represents a right regular prism (box) --/
structure Box where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- Calculates the lengths of external diagonals of a box --/
noncomputable def external_diagonals (box : Box) : (ℝ × ℝ × ℝ) :=
  (Real.sqrt (box.a^2 + box.b^2), Real.sqrt (box.b^2 + box.c^2), Real.sqrt (box.a^2 + box.c^2))

/-- Checks if a triple of numbers satisfies the external diagonal conditions --/
def is_valid_external_diagonals (d1 d2 d3 : ℝ) : Prop :=
  d1^2 + d2^2 > d3^2 ∧ d1^2 + d3^2 > d2^2 ∧ d2^2 + d3^2 > d1^2

/-- Theorem: The external diagonals of a box must satisfy the validity conditions --/
theorem external_diagonals_validity (box : Box) :
  let (d1, d2, d3) := external_diagonals box
  is_valid_external_diagonals d1 d2 d3 := by
  sorry

/-- Checks if a given triple could be the lengths of external diagonals --/
def could_be_external_diagonals (d1 d2 d3 : ℝ) : Prop :=
  d1 > 0 ∧ d2 > 0 ∧ d3 > 0 ∧ is_valid_external_diagonals d1 d2 d3

/-- Theorem: The sets {3,4,6} and {4,5,6} cannot be the lengths of external diagonals --/
theorem invalid_external_diagonals :
  ¬ could_be_external_diagonals 3 4 6 ∧ ¬ could_be_external_diagonals 4 5 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_diagonals_validity_invalid_external_diagonals_l1201_120162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1201_120145

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + 1/x

-- State the theorem
theorem f_monotone_increasing :
  ∀ x y : ℝ, 1 < x → x < y → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1201_120145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_inscribed_circle_l1201_120105

structure ConvexPolygon where
  sides : List ℝ
  convex : Bool
  red_sides : List ℝ
  blue_sides : List ℝ
  red_sum_less_than_half_perimeter : (red_sides.sum) < ((sides.sum) / 2)
  no_adjacent_blue_sides : Bool

/-- Represents an inscribed circle in a polygon -/
def InscribedCircle (p : ConvexPolygon) (r : ℝ) : Prop := sorry

theorem no_inscribed_circle (p : ConvexPolygon) : 
  ¬∃ (r : ℝ), r > 0 ∧ InscribedCircle p r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_inscribed_circle_l1201_120105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1201_120174

noncomputable def f (x : ℝ) : ℝ := 3 * x + 1 + 12 / (x^2)

theorem f_minimum_value :
  (∀ x > 0, f x ≥ 10) ∧ (∃ x > 0, f x = 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1201_120174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_x_alcohol_percentage_l1201_120131

/-- Represents a solution with a volume and alcohol percentage -/
structure Solution where
  volume : ℝ
  alcoholPercentage : ℝ

/-- Calculates the amount of alcohol in a solution -/
noncomputable def alcoholAmount (s : Solution) : ℝ :=
  s.volume * s.alcoholPercentage

/-- Represents the mixing of two solutions -/
noncomputable def mixSolutions (s1 s2 : Solution) : Solution where
  volume := s1.volume + s2.volume
  alcoholPercentage := (alcoholAmount s1 + alcoholAmount s2) / (s1.volume + s2.volume)

theorem solution_x_alcohol_percentage :
  ∀ (x : Solution),
    x.volume = 300 →
    let y : Solution := ⟨100, 0.30⟩
    let mixed := mixSolutions x y
    mixed.alcoholPercentage = 0.15 →
    x.alcoholPercentage = 0.10 := by
  sorry

#check solution_x_alcohol_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_x_alcohol_percentage_l1201_120131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_is_zero_l1201_120137

/-- Represents a three-digit number in a given base -/
structure ThreeDigitNumber (base : ℕ) where
  hundreds : ℕ
  tens : ℕ
  ones : ℕ
  valid : hundreds < base ∧ tens < base ∧ ones < base

/-- Converts a ThreeDigitNumber to its numerical value -/
def to_nat {base : ℕ} (n : ThreeDigitNumber base) : ℕ :=
  n.hundreds * base^2 + n.tens * base + n.ones

/-- Reverses the digits of a ThreeDigitNumber -/
def reverse {base : ℕ} (n : ThreeDigitNumber base) : ThreeDigitNumber base where
  hundreds := n.ones
  tens := n.tens
  ones := n.hundreds
  valid := by
    constructor
    · exact n.valid.right.right
    · constructor
      · exact n.valid.right.left
      · exact n.valid.left

theorem last_digit_is_zero :
  ∀ (n : ThreeDigitNumber 5),
    (∃ (m : ThreeDigitNumber 8), to_nat n = to_nat (reverse m)) →
    n.ones = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_is_zero_l1201_120137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_of_derivative_lower_bound_when_a_is_one_l1201_120196

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - Real.log (2 * x + a)

-- Theorem for part (1)
theorem unique_zero_of_derivative (a : ℝ) (h : 0 < a ∧ a < 2) :
  ∃! x : ℝ, x > -a/2 ∧ (deriv (f a)) x = 0 := by
  sorry

-- Theorem for part (2)
theorem lower_bound_when_a_is_one (x : ℝ) (h : x > -1/2) :
  f 1 x > 3/2 - Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_of_derivative_lower_bound_when_a_is_one_l1201_120196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l1201_120182

noncomputable def intersection_point : ℝ × ℝ := (3, 3/4)

def line1 (p : ℝ × ℝ) : Prop := 3 * p.1 + 4 * p.2 - 12 = 0

def line2 (p : ℝ × ℝ) : Prop := 6 * p.1 - 4 * p.2 - 12 = 0

def line3 (p : ℝ × ℝ) : Prop := p.1 - 3 = 0

def line4 (p : ℝ × ℝ) : Prop := p.2 - 3/4 = 0

theorem unique_intersection :
  (∀ p : ℝ × ℝ, line1 p ∧ line2 p ∧ line3 p ∧ line4 p → p = intersection_point) ∧
  (line1 intersection_point ∧ line2 intersection_point ∧ line3 intersection_point ∧ line4 intersection_point) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l1201_120182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isabella_babysitting_hours_l1201_120115

/-- Represents the number of hours Isabella babysits each day -/
def daily_hours : ℕ := sorry

/-- Isabella's hourly rate in dollars -/
def hourly_rate : ℕ := 5

/-- Number of days Isabella works per week -/
def days_per_week : ℕ := 6

/-- Number of weeks Isabella has worked -/
def total_weeks : ℕ := 7

/-- Total amount earned by Isabella in dollars -/
def total_earned : ℕ := 1050

theorem isabella_babysitting_hours :
  daily_hours * hourly_rate * days_per_week * total_weeks = total_earned ∧
  daily_hours = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isabella_babysitting_hours_l1201_120115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1201_120183

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := -5 * Real.exp x + 3

-- Define the point of tangency
def point : ℝ × ℝ := (0, -2)

-- Define the slope of the tangent line at x = 0
def tangent_slope : ℝ := -5

-- Theorem statement
theorem tangent_line_equation :
  let x₀ := point.fst
  let y₀ := point.snd
  (λ x y => tangent_slope * (x - x₀) - (y - y₀) = 0) =
  (λ x y => 5 * x + y + 2 = 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1201_120183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1201_120187

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  (Real.cos (ω * x))^2 - (Real.sin (ω * x))^2 + 2 * Real.sqrt 3 * Real.cos (ω * x) * Real.sin (ω * x)

theorem triangle_problem (ω : ℝ) (A B C : ℝ) (a b c : ℝ) :
  ω > 0 →
  (∀ x : ℝ, f ω (x + π / (2 * ω)) = f ω x) →
  a = Real.sqrt 3 →
  b + c = 3 →
  (∀ x : ℝ, f 1 x ≤ 1) →
  f 1 A = 1 →
  Real.sin B * Real.sin C = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1201_120187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_A_from_tan_A_l1201_120164

theorem cos_A_from_tan_A (A : ℝ) (h : Real.tan A = -2) : Real.cos A = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_A_from_tan_A_l1201_120164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l1201_120190

noncomputable def f (x : ℝ) : ℝ := (2*x^2 + 3*x - 7) / (x - 3)

def is_slant_asymptote (m b : ℝ) : Prop :=
  ∀ ε > 0, ∃ N : ℝ, ∀ x > N, |f x - (m*x + b)| < ε

theorem slant_asymptote_sum :
  ∃ m b : ℝ, is_slant_asymptote m b ∧ m + b = 11 := by
  -- We know m = 2 and b = 9 from our manual calculation
  use 2, 9
  constructor
  · sorry  -- Proof of is_slant_asymptote 2 9
  · -- Proof that 2 + 9 = 11
    norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l1201_120190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_3AB_l1201_120186

open Matrix

variable {n : ℕ}
variable (A B : Matrix (Fin n) (Fin n) ℝ)

theorem det_3AB (h1 : det A = -3) (h2 : det B = 5) :
  det ((3 : ℝ) • (A * B)) = -15 * 3^n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_3AB_l1201_120186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_circumscribed_sphere_radius_l1201_120153

/-- Given a tetrahedron P-ABC with edge lengths PA = BC = √6, PB = AC = √8, and PC = AB = √10,
    the radius of its circumscribed sphere is √3. -/
theorem tetrahedron_circumscribed_sphere_radius 
  (P A B C : EuclideanSpace ℝ (Fin 3))
  (h_PA : ‖P - A‖ = Real.sqrt 6)
  (h_BC : ‖B - C‖ = Real.sqrt 6)
  (h_PB : ‖P - B‖ = Real.sqrt 8)
  (h_AC : ‖A - C‖ = Real.sqrt 8)
  (h_PC : ‖P - C‖ = Real.sqrt 10)
  (h_AB : ‖A - B‖ = Real.sqrt 10) :
  ∃ (center : EuclideanSpace ℝ (Fin 3)), 
    ‖center - P‖ = Real.sqrt 3 ∧ 
    ‖center - A‖ = Real.sqrt 3 ∧ 
    ‖center - B‖ = Real.sqrt 3 ∧ 
    ‖center - C‖ = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_circumscribed_sphere_radius_l1201_120153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_representation_235_l1201_120154

theorem binary_representation_235 : ∃ (binary : List Bool), 
  (binary.foldl (λ acc b => 2 * acc + if b then 1 else 0) 0 = 235) ∧ 
  (binary.filter id).length - (binary.filter not).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_representation_235_l1201_120154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kishore_miscellaneous_expenses_l1201_120116

def monthly_salary (savings : ℕ) (savings_percentage : ℚ) : ℚ :=
  (savings : ℚ) / savings_percentage

def total_known_expenses (rent milk groceries education petrol : ℕ) : ℕ :=
  rent + milk + groceries + education + petrol

def miscellaneous_expenses (salary : ℚ) (known_expenses savings : ℕ) : ℚ :=
  salary - (known_expenses : ℚ) - (savings : ℚ)

theorem kishore_miscellaneous_expenses 
  (rent milk groceries education petrol savings : ℕ)
  (savings_percentage : ℚ)
  (h1 : rent = 5000)
  (h2 : milk = 1500)
  (h3 : groceries = 4500)
  (h4 : education = 2500)
  (h5 : petrol = 2000)
  (h6 : savings = 2400)
  (h7 : savings_percentage = 1/10) :
  miscellaneous_expenses 
    (monthly_salary savings savings_percentage)
    (total_known_expenses rent milk groceries education petrol)
    savings = 6100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kishore_miscellaneous_expenses_l1201_120116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_to_cryptarithm_l1201_120199

-- Define the type for digits (1 to 9)
def Digit := Fin 9

-- Define the cryptarithm function
def cryptarithm (B O C b M U K J A : Digit) : Prop :=
  let numerator := B.val + O.val + C.val + b.val + M.val + O.val + U.val
  let denominator := K.val + J.val + A.val + C.val + C.val
  numerator * 29 = denominator * 22

-- Define the condition for all digits being different
def all_different (B O C b M U K J A : Digit) : Prop :=
  B ≠ O ∧ B ≠ C ∧ B ≠ b ∧ B ≠ M ∧ B ≠ U ∧ B ≠ K ∧ B ≠ J ∧ B ≠ A ∧
  O ≠ C ∧ O ≠ b ∧ O ≠ M ∧ O ≠ U ∧ O ≠ K ∧ O ≠ J ∧ O ≠ A ∧
  C ≠ b ∧ C ≠ M ∧ C ≠ U ∧ C ≠ K ∧ C ≠ J ∧ C ≠ A ∧
  b ≠ M ∧ b ≠ U ∧ b ≠ K ∧ b ≠ J ∧ b ≠ A ∧
  M ≠ U ∧ M ≠ K ∧ M ≠ J ∧ M ≠ A ∧
  U ≠ K ∧ U ≠ J ∧ U ≠ A ∧
  K ≠ J ∧ K ≠ A ∧
  J ≠ A

-- Theorem statement
theorem no_solutions_to_cryptarithm :
  ¬∃ (B O C b M U K J A : Digit),
    cryptarithm B O C b M U K J A ∧ all_different B O C b M U K J A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_to_cryptarithm_l1201_120199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yujin_ribbon_tape_length_l1201_120122

/-- The length of ribbon tape needed for one ribbon in meters -/
def ribbon_length : ℝ := 0.84

/-- The number of ribbons made -/
def ribbons_made : ℕ := 10

/-- The remaining length of ribbon tape in meters -/
def remaining_length : ℝ := 0.5

/-- The original length of Yujin's ribbon tape in meters -/
def original_length : ℝ := ribbon_length * (ribbons_made : ℝ) + remaining_length

theorem yujin_ribbon_tape_length :
  original_length = 8.9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yujin_ribbon_tape_length_l1201_120122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_coordinates_l1201_120101

/-- A point on a parabola with given properties -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : x^2 = 4*y
  distance_to_focus : Real.sqrt (x^2 + (y - 1)^2) = 2

/-- Theorem stating the coordinates of the point on the parabola -/
theorem parabola_point_coordinates (M : ParabolaPoint) : (M.x = 2 ∨ M.x = -2) ∧ M.y = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_coordinates_l1201_120101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1201_120158

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem equation_solutions : 
  {x : ℝ | 4 * x^2 - 40 * (floor x) + 51 = 0} = 
  {Real.sqrt 29 / 2, Real.sqrt 189 / 2, Real.sqrt 229 / 2, Real.sqrt 269 / 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1201_120158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_distance_formula_l1201_120141

/-- Two airplanes traveling in opposite directions -/
structure Airplanes where
  speed1 : ℝ
  speed2 : ℝ
  initial_time : ℝ
  initial_distance : ℝ

/-- The combined speed of the airplanes -/
def combined_speed (a : Airplanes) : ℝ := a.speed1 + a.speed2

/-- The time it takes for the airplanes to reach a certain distance -/
noncomputable def time_to_distance (a : Airplanes) (distance : ℝ) : ℝ :=
  distance / combined_speed a

/-- Theorem: The time to reach a certain distance is given by distance / combined_speed -/
theorem time_to_distance_formula (a : Airplanes) (distance : ℝ) :
  a.speed1 = 400 ∧ a.speed2 = 250 ∧ a.initial_time = 2.5 ∧ a.initial_distance = 1625 →
  time_to_distance a distance = distance / 650 := by
  intro h
  unfold time_to_distance
  unfold combined_speed
  simp [h.1, h.2]
  ring

#check time_to_distance_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_distance_formula_l1201_120141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l1201_120142

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem omega_value 
  (ω : ℝ) 
  (φ : ℝ) 
  (h1 : ω > 0)
  (h2 : ∀ x y, 0 < x ∧ x < y ∧ y < π/3 → f ω φ x < f ω φ y)
  (h3 : f ω φ (π/6) + f ω φ (π/3) = 0)
  (h4 : f ω φ 0 = -1) :
  ω = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l1201_120142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unshaded_probability_is_36_55_l1201_120104

/-- Represents a 2 by 10 rectangle with shaded ends -/
structure ShadedRectangle where
  width : Nat
  height : Nat
  shaded_ends : Bool

/-- Calculates the total number of rectangles in the grid -/
def total_rectangles (r : ShadedRectangle) : Nat :=
  3 * Nat.choose (r.width + 1) 2

/-- Calculates the number of rectangles not including shaded squares -/
def unshaded_rectangles (r : ShadedRectangle) : Nat :=
  3 * Nat.choose (r.width - 1) 2

/-- The probability of choosing an unshaded rectangle -/
def unshaded_probability (r : ShadedRectangle) : ℚ :=
  ↑(unshaded_rectangles r) / ↑(total_rectangles r)

theorem unshaded_probability_is_36_55 (r : ShadedRectangle) :
  r.width = 10 ∧ r.height = 2 ∧ r.shaded_ends = true →
  unshaded_probability r = 36 / 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unshaded_probability_is_36_55_l1201_120104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janes_correct_answers_percentage_l1201_120188

theorem janes_correct_answers_percentage 
  (t : ℝ)  -- total number of problems
  (h1 : t > 0)  -- ensure total problems is positive
  (y : ℝ)  -- number of problems solved correctly together
  (h2 : y ≥ 0 ∧ y ≤ t/3)  -- ensure y is non-negative and not more than 1/3 of total problems
  (h3 : 0.7 * (t/3) + y = 0.82 * t)  -- Sarah's correct answers equation
  (h4 : y + 0.85 * (t/3) = 0.66 * t)  -- Jane's correct answers equation
  : (y + 0.85 * (t/3)) / t = 0.66 := by
  sorry

#check janes_correct_answers_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_janes_correct_answers_percentage_l1201_120188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l1201_120178

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4*x + 3) / (2*x - 6)
noncomputable def g (x a b c : ℝ) : ℝ := (a*x^2 + b*x + c) / (x - 3)

theorem intersection_point 
  (a b c : ℝ) 
  (h1 : ∀ x, x ≠ 3 → (2*x - 6 = 0 ↔ x - 3 = 0))
  (h2 : ∃ m₁ m₂, (m₁ * m₂ = -1 ∧ 
                  (∀ ε > 0, ∃ M, ∀ x > M, |f x - (m₁*x + 1)| < ε) ∧ 
                  (∀ ε > 0, ∃ M, ∀ x > M, |g x a b c - (m₂*x + 1)| < ε)))
  (h3 : f (-1) = g (-1) a b c)
  (h4 : ∃ x₀, x₀ ≠ -1 ∧ f x₀ = g x₀ a b c)
  : ∃ x₀, x₀ = 1 ∧ f x₀ = g x₀ a b c ∧ f x₀ = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l1201_120178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_divisibility_by_13_l1201_120135

theorem six_digit_divisibility_by_13 : 
  (Finset.filter 
    (fun n : ℕ => 
      ((n / 100) + (n % 100)) % 13 = 0) 
    (Finset.range 900000)).card = 72000 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_divisibility_by_13_l1201_120135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circlesun_properties_l1201_120167

/-- Custom operation ☉ between plane vectors -/
def circlesun (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.2 - a.2 * b.1

theorem circlesun_properties :
  ∀ (a b : ℝ × ℝ) (l : ℝ),
  -- 1. If a and b are collinear, then a☉b = 0
  (∃ (k : ℝ), a = (k * b.1, k * b.2) → circlesun a b = 0) ∧
  -- 2. a☉b ≠ b☉a (negation of the incorrect statement)
  (circlesun a b ≠ circlesun b a) ∧
  -- 3. For any l ∈ ℝ, (la)☉b = l(a☉b)
  (circlesun (l * a.1, l * a.2) b = l * circlesun a b) ∧
  -- 4. (a☉b)² + (a⋅b)² = |a|²|b|²
  ((circlesun a b)^2 + (a.1 * b.1 + a.2 * b.2)^2 = (a.1^2 + a.2^2) * (b.1^2 + b.2^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circlesun_properties_l1201_120167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_framed_painting_ratio_l1201_120156

theorem framed_painting_ratio :
  ∀ (y : ℝ),
  y > 0 →
  (15 + 2*y) * (20 + 6*y) = 2 * 15 * 20 →
  (min (15 + 2*y) (20 + 6*y)) / (max (15 + 2*y) (20 + 6*y)) = 4 / 7 :=
by
  intro y hy heq
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_framed_painting_ratio_l1201_120156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_repair_percentage_l1201_120146

/-- Calculates the percentage of the original cost spent on repairs for a scooter sale -/
theorem scooter_repair_percentage (repair_cost profit : ℚ) (profit_percentage : ℚ) :
  repair_cost = 500 →
  profit = 1100 →
  profit_percentage = 20 →
  let original_cost := profit / (profit_percentage / 100)
  let repair_percentage := (repair_cost / original_cost) * 100
  ∃ (ε : ℚ), abs (repair_percentage - 9.09) < ε ∧ ε > 0 := by
  sorry

#eval (500 / (1100 / 0.2)) * 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_repair_percentage_l1201_120146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l1201_120138

-- Define the equations
def eq1 (x y : ℝ) : Prop := (x - 2*y + 3)*(3*x + 2*y - 5) = 0
def eq2 (x y : ℝ) : Prop := (x + y - 3)*(x^2 - 5*y + 6) = 0

-- Define a point as a pair of real numbers
def Point := ℝ × ℝ

-- Define what it means for a point to satisfy both equations
def satisfies_both (p : Point) : Prop :=
  let (x, y) := p
  eq1 x y ∧ eq2 x y

-- State the theorem
theorem intersection_points_count :
  ∃ (S : Finset Point), (∀ p ∈ S, satisfies_both p) ∧ (S.card = 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l1201_120138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_condition_l1201_120155

/-- Line represented by ax + (a+2)y + 2 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + (a + 2) * y + 2 = 0

/-- Line represented by x + ay + 1 = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop := x + a * y + 1 = 0

/-- The slope of line1 -/
noncomputable def slope1 (a : ℝ) : ℝ := -a / (a + 2)

/-- The slope of line2 -/
noncomputable def slope2 (a : ℝ) : ℝ := -1 / a

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (a : ℝ) : Prop := slope1 a = slope2 a

theorem parallel_lines_condition (a : ℝ) :
  parallel a ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_condition_l1201_120155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exchange_theorem_l1201_120140

/-- Given an exchange rate between Japanese yen and Chinese yuan, 
    calculate the amount of Chinese yuan that can be exchanged for a given amount of Japanese yen. -/
noncomputable def exchange_rate (rate : ℝ) (yen : ℝ) : ℝ :=
  (yen / 100) * rate

/-- Theorem: Given the exchange rate of 7.2 Chinese yuan for 100 Japanese yen,
    60,000 Japanese yen can be exchanged for 4320 Chinese yuan. -/
theorem exchange_theorem : exchange_rate 7.2 60000 = 4320 := by
  -- Unfold the definition of exchange_rate
  unfold exchange_rate
  -- Simplify the arithmetic
  simp [mul_div_assoc]
  -- Check that the result is equal to 4320
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exchange_theorem_l1201_120140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_theorem_l1201_120192

/-- A pyramid with a rectangular base and specific properties -/
structure Pyramid where
  b : ℝ  -- length of the diagonal of the base
  angle_between_diagonals : ℝ
  lateral_edge_angle : ℝ
  angle_between_diagonals_is_60 : angle_between_diagonals = Real.pi / 3
  lateral_edge_angle_is_45 : lateral_edge_angle = Real.pi / 4

/-- The volume of the pyramid -/
noncomputable def pyramid_volume (p : Pyramid) : ℝ :=
  (p.b ^ 3 * Real.sqrt 3) / 24

/-- Theorem stating the volume of the pyramid with given properties -/
theorem pyramid_volume_theorem (p : Pyramid) : 
  pyramid_volume p = (p.b ^ 3 * Real.sqrt 3) / 24 := by
  -- Unfold the definition of pyramid_volume
  unfold pyramid_volume
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_theorem_l1201_120192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crayon_cost_conversion_l1201_120175

/-- Calculates the total cost of crayons in EUR and GBP given the specified conditions --/
theorem crayon_cost_conversion 
  (half_dozens : ℕ) 
  (price_per_crayon : ℝ) 
  (first_discount_percent : ℝ) 
  (second_discount_percent : ℝ) 
  (usd_to_eur_rate : ℝ) 
  (usd_to_gbp_rate : ℝ) :
  let total_crayons := half_dozens * 6
  let initial_cost := total_crayons * price_per_crayon
  let after_first_discount := initial_cost * (1 - first_discount_percent / 100)
  let final_cost_usd := after_first_discount * (1 - second_discount_percent / 100)
  let final_cost_eur := final_cost_usd * usd_to_eur_rate
  let final_cost_gbp := final_cost_usd * usd_to_gbp_rate
  half_dozens = 4 ∧ 
  price_per_crayon = 2 ∧ 
  first_discount_percent = 10 ∧ 
  second_discount_percent = 5 ∧ 
  usd_to_eur_rate = 0.85 ∧ 
  usd_to_gbp_rate = 0.75 →
  final_cost_eur = 34.884 ∧ final_cost_gbp = 30.78 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crayon_cost_conversion_l1201_120175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_placements_l1201_120180

-- Define the size of the grid
def gridSize : Nat := 10

-- Define the width of the rectangle
def rectWidth : Nat := 3

-- Function to calculate the number of placements for a given length
def placementsForLength (l : Nat) : Nat :=
  if l ≥ rectWidth ∧ l ≤ gridSize then
    (gridSize - rectWidth + 1) * (gridSize - l + 1)
  else
    0

-- Function to calculate the total number of placements
def totalPlacements : Nat :=
  2 * (List.range (gridSize - rectWidth + 1)).foldr (fun i acc => acc + placementsForLength (i + rectWidth)) 0 -
  placementsForLength rectWidth

-- Theorem statement
theorem rectangle_placements :
  totalPlacements = 384 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_placements_l1201_120180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_men_to_women_ratio_l1201_120176

/-- Represents a group of people with men, women, and children -/
structure PeopleGroup where
  men : ℕ
  women : ℕ
  children : ℕ

/-- The properties of the specific group in the problem -/
def problem_group : PeopleGroup → Prop
  | g => g.children = 30 ∧
         g.women = 3 * g.children ∧
         g.men + g.women + g.children = 300

/-- The theorem stating the ratio of men to women is 2:1 -/
theorem men_to_women_ratio (g : PeopleGroup) (h : problem_group g) : 
  g.men * 1 = g.women * 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_men_to_women_ratio_l1201_120176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_from_container_A_l1201_120169

/-- The amount of mixture taken from container A -/
def a : ℝ := sorry

/-- The amount of mixture taken from container B -/
def b : ℝ := sorry

/-- The percentage of acid in container A -/
def acid_percentage_A : ℝ := 0.40

/-- The percentage of acid in container B -/
def acid_percentage_B : ℝ := 0.60

/-- The amount of pure water added -/
def water_added : ℝ := 70

/-- The total volume of the final mixture -/
def total_volume : ℝ := 100

/-- The percentage of acid in the final mixture -/
def final_acid_percentage : ℝ := 0.17

theorem amount_from_container_A : 
  (a + b + water_added = total_volume) →
  (acid_percentage_A * a + acid_percentage_B * b = final_acid_percentage * total_volume) →
  a = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_from_container_A_l1201_120169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_CBG_l1201_120197

-- Define the circle and points
def Circle : Type := ℝ × ℝ × ℝ  -- center (x, y) and radius r
def Point : Type := ℝ × ℝ

-- Define the given circle and points
noncomputable def circleO : Circle := (0, 0, 3)  -- Assuming center at origin for simplicity
noncomputable def A : Point := sorry
noncomputable def B : Point := sorry
noncomputable def C : Point := sorry
noncomputable def D : Point := sorry
noncomputable def E : Point := sorry
noncomputable def F : Point := sorry
noncomputable def G : Point := sorry

-- Define helper functions
def dist (p q : Point) : ℝ := sorry
def Line (p q : Point) : Type := sorry

-- Define the properties of the triangle and points
axiom equilateral_ABC : sorry -- IsEquilateral A B C
axiom inscribed_ABC : sorry -- Inscribed A B C circleO
axiom AD_length : dist A D = 15
axiom AE_length : dist A E = 17
axiom l1_parallel_AE : sorry -- IsParallel (Line D F) (Line A E)
axiom l2_parallel_AD : sorry -- IsParallel (Line E F) (Line A D)
axiom G_on_circle : sorry -- OnCircle G circleO
axiom G_collinear : sorry -- Collinear A F G
axiom G_distinct : G ≠ A

-- Define the area function
def TriangleArea (p q r : Point) : ℝ := sorry

-- Define the theorem
theorem area_CBG : 
  ∃ (area : ℝ), TriangleArea C B G = area ∧ area = (765 * Real.sqrt 3) / 1024 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_CBG_l1201_120197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_truth_l1201_120106

-- Define the propositions
def p : Prop := ∀ (A B : Set ℝ), (A = Bᶜ) ↔ (A ∩ B = ∅)

def q : Prop := ∀ (f : ℝ → ℝ), (∀ x, f x = f (-x)) → (∀ x y, f x = y ↔ f (-x) = y)

-- State the theorem
theorem proposition_truth : ¬p ∧ q := by
  constructor
  · -- Prove ¬p
    sorry
  · -- Prove q
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_truth_l1201_120106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triples_characterization_l1201_120134

/-- A triple of positive integers (a, b, p) where p is prime and a^p = b! + p -/
structure SpecialTriple where
  a : ℕ
  b : ℕ
  p : ℕ
  a_pos : 0 < a
  b_pos : 0 < b
  p_pos : 0 < p
  p_prime : Nat.Prime p
  equation : a^p = Nat.factorial b + p

/-- The set of all SpecialTriples -/
def AllSpecialTriples : Set SpecialTriple := {t : SpecialTriple | True}

/-- The set containing only the triples (2, 2, 2) and (3, 4, 3) -/
def CorrectTriples : Set SpecialTriple :=
  {t : SpecialTriple | (t.a = 2 ∧ t.b = 2 ∧ t.p = 2) ∨ (t.a = 3 ∧ t.b = 4 ∧ t.p = 3)}

theorem special_triples_characterization :
  AllSpecialTriples = CorrectTriples := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triples_characterization_l1201_120134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_triangle_area_zero_l1201_120110

/-- Area of a triangle given coordinates of its vertices -/
noncomputable def area_triangle (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
  (1/2) * abs ((x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂)))

/-- Two perpendicular lines intersecting at (4,12) with y-intercepts summing to 16 form a triangle with area 0 -/
theorem perpendicular_lines_triangle_area_zero (m₁ m₂ b₁ b₂ : ℝ) : 
  m₁ * m₂ = -1 → -- perpendicular lines
  12 = 4 * m₁ + b₁ → -- first line passes through (4,12)
  12 = 4 * m₂ + b₂ → -- second line passes through (4,12)
  b₁ + b₂ = 16 → -- sum of y-intercepts is 16
  area_triangle 0 b₁ 0 b₂ 4 12 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_triangle_area_zero_l1201_120110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1201_120118

noncomputable def f (x : ℝ) : ℝ := 1 / Real.sin x - (Real.cos x / Real.sin x) - Real.sin x

theorem f_properties :
  (f 0 = 0) ∧
  (deriv f 0 = -1/2) ∧
  (deriv f (Real.pi/2) = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1201_120118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eggs_next_month_l1201_120171

/-- Calculates the number of eggs after a month, given the initial amount, 
    the amount bought, and the amount eaten. -/
def eggs_after_month (initial : ℕ) (bought : ℕ) (eaten : ℕ) : ℕ :=
  initial + bought - eaten

/-- Proves that given the conditions in the problem, 
    the number of eggs after mother buys eggs for the next month will be 41. -/
theorem eggs_next_month : 
  let initial_last_month : ℕ := 27
  let total_after_buying : ℕ := 58
  let eaten_this_month : ℕ := 48
  let bought : ℕ := total_after_buying - initial_last_month
  eggs_after_month total_after_buying eaten_this_month bought = 41 := by
  sorry

#eval eggs_after_month 58 31 48  -- Should evaluate to 41

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eggs_next_month_l1201_120171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_cosine_in_special_triangle_l1201_120151

theorem largest_angle_cosine_in_special_triangle (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_ratio : ∃ k : ℝ, k > 0 ∧ a = 2*k ∧ b = 4*k ∧ c = 3*k) :
  let max_cos := min (min ((a^2 + b^2 - c^2) / (2*a*b)) ((b^2 + c^2 - a^2) / (2*b*c))) ((c^2 + a^2 - b^2) / (2*c*a))
  max_cos = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_cosine_in_special_triangle_l1201_120151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_64_l1201_120139

theorem divisibility_by_64 (n : ℕ) : 64 ∣ (3^(2*n) - 8*n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_64_l1201_120139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_problem_l1201_120185

theorem complement_intersection_problem (U A B : Set ℕ) : 
  U = {0, 1, 2, 3} →
  A = {0, 1, 2} →
  B = {2, 3} →
  (U \ A) ∩ B = {3} := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_problem_l1201_120185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_average_age_l1201_120152

/-- Represents a cricket team with given properties -/
structure CricketTeam where
  totalMembers : ℕ
  captainAge : ℕ
  wicketKeeperAgeDiff : ℕ
  averageAgeDiff : ℕ

/-- The average age of the cricket team satisfies the given conditions -/
theorem cricket_team_average_age (team : CricketTeam)
  (h1 : team.totalMembers = 11)
  (h2 : team.captainAge = 26)
  (h3 : team.wicketKeeperAgeDiff = 5)
  (h4 : team.averageAgeDiff = 1) :
  let averageAge := (team.totalMembers * team.captainAge +
    team.totalMembers * team.wicketKeeperAgeDiff +
    (team.totalMembers - 2) * team.averageAgeDiff) / (2 * team.totalMembers)
  averageAge = 24 := by
  sorry

#check cricket_team_average_age

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_average_age_l1201_120152
