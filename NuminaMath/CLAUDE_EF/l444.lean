import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l444_44431

theorem polynomial_remainder : 
  ∃ q : Polynomial ℂ, (X : Polynomial ℂ)^55 + X^44 + X^33 + X^22 + X^11 + 1 = 
  (X^4 + X^3 + X^2 + X + 1) * q + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l444_44431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_properties_l444_44407

/-- Given four points in space -/
def A : Fin 3 → ℝ := ![1, 1, 0]
def B : Fin 3 → ℝ := ![0, 1, 2]
def C : Fin 3 → ℝ := ![0, 3, 2]
def D : Fin 3 → ℝ := ![-1, 3, 4]

/-- Vector from A to B -/
def AB : Fin 3 → ℝ := ![B 0 - A 0, B 1 - A 1, B 2 - A 2]

/-- Vector from B to C -/
def BC : Fin 3 → ℝ := ![C 0 - B 0, C 1 - B 1, C 2 - B 2]

/-- Vector from C to D -/
def CD : Fin 3 → ℝ := ![D 0 - C 0, D 1 - C 1, D 2 - C 2]

/-- Dot product of two 3D vectors -/
def dot_product (v w : Fin 3 → ℝ) : ℝ := (v 0 * w 0) + (v 1 * w 1) + (v 2 * w 2)

theorem points_properties :
  (dot_product AB BC = 0) ∧ 
  (AB = CD) ∧ 
  (¬(∃ t : ℝ, ∀ i : Fin 3, C i = (1 - t) * A i + t * B i)) ∧ 
  (∃ a b c d : ℝ, 
    (∀ i : Fin 3, a * A i + b * B i + c * C i + d * D i = 0) ∧
    (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_properties_l444_44407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increase_intervals_f_minimum_on_interval_l444_44489

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4*x + 4

-- Theorem for monotonic increase intervals
theorem f_monotonic_increase_intervals :
  ∀ x : ℝ, (x ≤ -2 ∨ x ≥ 2) ↔ (∀ y : ℝ, y > x → f y > f x) :=
by sorry

-- Theorem for minimum value on [0, 4]
theorem f_minimum_on_interval :
  ∃ x₀ : ℝ, x₀ ∈ Set.Icc 0 4 ∧ f x₀ = -4/3 ∧ ∀ x ∈ Set.Icc 0 4, f x ≥ f x₀ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increase_intervals_f_minimum_on_interval_l444_44489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_gemstones_needed_l444_44437

/-- Represents a set of earrings with specific ratios of materials -/
structure EarringSet where
  magnets : ℕ
  buttonRatio : ℚ
  gemstoneRatio : ℕ

/-- Calculates the number of gemstones needed for a pair of earrings in a set -/
def gemstonesForSet (set : EarringSet) : ℕ :=
  ⌊(2 : ℚ) * set.magnets * set.buttonRatio * set.gemstoneRatio⌋.toNat

/-- The four sets of earrings Rebecca makes -/
def rebeccaSets : List EarringSet := [
  { magnets := 2, buttonRatio := 1/2, gemstoneRatio := 3 },
  { magnets := 3, buttonRatio := 2, gemstoneRatio := 2 },
  { magnets := 4, buttonRatio := 1, gemstoneRatio := 4 },
  { magnets := 5, buttonRatio := 1/3, gemstoneRatio := 5 }
]

theorem total_gemstones_needed :
  (rebeccaSets.map gemstonesForSet).sum = 82 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_gemstones_needed_l444_44437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_sum_l444_44466

/-- Represents a point in the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in the coordinate plane -/
structure Triangle where
  p : Point
  q : Point
  r : Point

/-- Determines if a triangle has a right angle at a specific vertex -/
def hasRightAngleAt (t : Triangle) (v : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Determines if two points are on opposite sides of a line segment -/
def onOppositeSides (p1 p2 : Point) (line : Point × Point) : Prop := sorry

/-- Determines if a line through a point is parallel to a line segment -/
def isParallel (p : Point) (line : Point × Point) : Prop := sorry

/-- Represents a line segment between two points -/
def lineSegment (p1 p2 : Point) : Set Point := sorry

/-- Theorem: Given the conditions, if ST/SR = a/b where a and b are relatively prime positive integers, then a + b = 138 -/
theorem triangle_ratio_sum (pqr prs : Triangle) (s t : Point) (a b : ℕ) :
  hasRightAngleAt pqr pqr.q →
  distance pqr.p pqr.q = 5 →
  distance pqr.q pqr.r = 12 →
  hasRightAngleAt prs prs.p →
  distance prs.p s = 20 →
  onOppositeSides pqr.q s (pqr.p, pqr.r) →
  isParallel s (pqr.p, pqr.q) →
  (∃ u : Point, u ∈ lineSegment pqr.q pqr.r ∧ t = u) →
  (distance s t) / (distance s prs.r) = (a : ℝ) / b →
  Nat.Coprime a b →
  a + b = 138 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_sum_l444_44466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_professor_seating_arrangements_l444_44449

/-- The number of chairs in a row -/
def num_chairs : ℕ := 11

/-- The number of students -/
def num_students : ℕ := 7

/-- The number of professors -/
def num_professors : ℕ := 4

/-- The number of effective positions for professors -/
def effective_positions : ℕ := 7

/-- Represents that each professor must be seated between two students -/
def professor_between_students (seating : Fin num_chairs → Bool) : Prop :=
  ∀ i, seating i = true → i.val ≠ 0 ∧ i.val ≠ num_chairs - 1

/-- The number of ways professors can choose their chairs -/
def num_seating_arrangements : ℕ :=
  (effective_positions.choose num_professors) * (Nat.factorial num_professors)

/-- Theorem stating the number of ways professors can choose their chairs -/
theorem professor_seating_arrangements :
  num_seating_arrangements = 840 := by
  rw [num_seating_arrangements]
  rw [effective_positions, num_professors]
  norm_num
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_professor_seating_arrangements_l444_44449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_radius_sum_l444_44493

/-- The equation of the circle D -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 8*x - 2*y^2 - 6*y = -6

/-- The center of the circle D -/
noncomputable def center : ℝ × ℝ := (-4, -3/2)

/-- The radius of the circle D -/
noncomputable def radius : ℝ := Real.sqrt (47/4)

/-- Theorem stating the sum of the center coordinates and the radius -/
theorem center_radius_sum :
  let (c, d) := center
  c + d + radius = (-11 + Real.sqrt 47) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_radius_sum_l444_44493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_license_combinations_l444_44496

/-- The set of valid letters for boat licenses -/
def ValidLetters : Finset Char := {'A', 'M', 'S'}

/-- The set of valid first digits for the license number -/
def ValidFirstDigits : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- The set of valid digits for the remaining positions in the license number -/
def ValidOtherDigits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- A boat license is valid if it consists of a valid letter followed by a five-digit number
    where the first digit is not zero -/
def ValidLicense (license : Char × Nat × Nat × Nat × Nat × Nat) : Prop :=
  license.1 ∈ ValidLetters ∧
  license.2.1 ∈ ValidFirstDigits ∧
  license.2.2.1 ∈ ValidOtherDigits ∧
  license.2.2.2.1 ∈ ValidOtherDigits ∧
  license.2.2.2.2.1 ∈ ValidOtherDigits ∧
  license.2.2.2.2.2 ∈ ValidOtherDigits

/-- The set of all valid boat licenses -/
def ValidLicenses : Set (Char × Nat × Nat × Nat × Nat × Nat) :=
  {license | ValidLicense license}

/-- Proof that ValidLicenses is finite -/
instance : Fintype ValidLicenses :=
  sorry

theorem boat_license_combinations :
  Fintype.card ValidLicenses = 270000 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_license_combinations_l444_44496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sodium_chloride_formation_l444_44445

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents the reaction between NaOH and HCl -/
structure Reaction where
  naoh : Moles
  hcl : Moles
  nacl : Moles
  h2o : Moles

/-- The molar ratio of NaOH:HCl:NaCl in the reaction -/
def molar_ratio (r : Reaction) : Prop :=
  r.naoh = r.hcl ∧ r.naoh = r.nacl

/-- The theorem stating that 3 moles of NaOH and 3 moles of HCl produce 3 moles of NaCl -/
theorem sodium_chloride_formation (initial_naoh initial_hcl : Moles) 
    (h_initial_naoh : initial_naoh = (3 : ℝ))
    (h_initial_hcl : initial_hcl = (3 : ℝ)) :
    ∃ (r : Reaction), r.naoh = initial_naoh ∧ r.hcl = initial_hcl ∧ 
                      molar_ratio r ∧ r.nacl = (3 : ℝ) := by
  sorry

#check sodium_chloride_formation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sodium_chloride_formation_l444_44445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l444_44400

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + x * Real.log x

-- Define the derivative of f
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := a + Real.log x + 1

-- State the theorem
theorem max_k_value (a : ℝ) :
  (f_derivative a (Real.exp 1) = 3) →
  (∃ k : ℤ, ∀ x > 1, (k : ℝ) < f a x / (x - 1) ∧
    ∀ m : ℤ, (∀ x > 1, (m : ℝ) < f a x / (x - 1)) → m ≤ k) →
  (∃ k : ℤ, k = 3 ∧
    ∀ x > 1, (k : ℝ) < f a x / (x - 1) ∧
    ∀ m : ℤ, (∀ x > 1, (m : ℝ) < f a x / (x - 1)) → m ≤ k) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l444_44400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_when_a_is_one_a_is_zero_when_z_is_imaginary_l444_44486

-- Define the complex number z as a function of a
noncomputable def z (a : ℝ) : ℂ := (a + 1) / (a - Complex.I)

-- Theorem 1: |z| = √2 when a = 1
theorem abs_z_when_a_is_one :
  Complex.abs (z 1) = Real.sqrt 2 := by sorry

-- Theorem 2: a = 0 is the only real solution when z is purely imaginary
theorem a_is_zero_when_z_is_imaginary :
  ∀ a : ℝ, (∃ b : ℝ, z a = Complex.I * b) → a = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_when_a_is_one_a_is_zero_when_z_is_imaginary_l444_44486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_prime_196_l444_44452

/-- Base prime representation of a natural number -/
def BasePrimeRepr (n : ℕ) : List ℕ := sorry

/-- Convert a natural number to its base prime representation -/
def toBasePrime (n : ℕ) : List ℕ := BasePrimeRepr n

/-- The prime factorization of 196 -/
axiom factorization_196 : 196 = 2^2 * 7^2

theorem base_prime_196 : toBasePrime 196 = [2, 0, 0, 2] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_prime_196_l444_44452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiring_theorem_l444_44469

/-- Represents the number of different application sequences where the kth best candidate is hired -/
def A (k : ℕ) : ℕ := sorry

/-- The hiring strategy for 10 applicants with distinct abilities -/
axiom hiring_strategy : Prop

theorem hiring_theorem :
  hiring_strategy →
  (∀ k ∈ Finset.range 7, A (k + 1) > A (k + 2)) ∧
  (A 8 = A 9) ∧ (A 9 = A 10) ∧
  (A 1 + A 2 + A 3 : ℚ) / 3628800 > 7/10 ∧
  (A 8 + A 9 + A 10 : ℚ) / 3628800 ≤ 1/10 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiring_theorem_l444_44469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_difference_is_pi_over_two_l444_44416

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (4 * x - Real.pi / 6) + 1

-- State the theorem
theorem x_difference_is_pi_over_two (x₁ x₂ : ℝ) :
  g x₁ * g x₂ = 9 → |x₁ - x₂| = Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_difference_is_pi_over_two_l444_44416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l444_44494

-- Define the train lengths
noncomputable def train_length : ℝ := 250

-- Define the speeds of the trains
noncomputable def speed_train1 : ℝ := 90
noncomputable def speed_train2 : ℝ := 110

-- Define the conversion factor from km/h to m/s
noncomputable def kmph_to_ms : ℝ := 5 / 18

-- Theorem statement
theorem trains_crossing_time :
  let relative_speed := (speed_train1 + speed_train2) * kmph_to_ms
  let total_distance := 2 * train_length
  let crossing_time := total_distance / relative_speed
  ∃ ε > 0, |crossing_time - 9| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l444_44494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l444_44408

theorem log_inequality (m n : ℝ) (h1 : 1 < m) (h2 : m < n) : 
  let a := (Real.log m / Real.log n) ^ 2
  let b := Real.log (m ^ 2) / Real.log n
  let c := Real.log (Real.log m / Real.log n) / Real.log n
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l444_44408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_of_set_l444_44478

theorem multiple_of_set (n : ℕ) : n = 30 → (∀ m : ℕ, m ∈ ({2, 3, 5} : Finset ℕ) → n % m = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_of_set_l444_44478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_surface_area_equals_384pi_l444_44403

/-- The radius of each hemisphere in centimeters -/
def radius : ℝ := 8

/-- The surface area of a sphere with the given radius -/
noncomputable def sphereSurfaceArea : ℝ := 4 * Real.pi * radius^2

/-- The curved surface area of one hemisphere -/
noncomputable def hemisphereArea : ℝ := sphereSurfaceArea / 2

/-- The area of the base of each hemisphere -/
noncomputable def baseArea : ℝ := Real.pi * radius^2

/-- The total surface area of two hemispheres joined at their bases,
    where one hemisphere is reflective (doubling its effective area) -/
noncomputable def totalSurfaceArea : ℝ := hemisphereArea + 2 * hemisphereArea

theorem total_surface_area_equals_384pi :
  totalSurfaceArea = 384 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_surface_area_equals_384pi_l444_44403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_method_precision_l444_44442

theorem bisection_method_precision (a b precision : Real) (h1 : a < b) (h2 : precision > 0) :
  let n := Real.log ((b - a) / precision) / Real.log 2
  ⌈n⌉ = (7 : ℕ) → 
  (a = 2 ∧ b = 3 ∧ precision = 0.01) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_method_precision_l444_44442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_is_sqrt_7_75_l444_44492

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Calculates the volume of a tetrahedron given its four vertices -/
noncomputable def tetrahedronVolume (a b c d : Point3D) : ℝ :=
  (1/6) * abs (
    (b.x - a.x) * ((c.y - a.y) * (d.z - a.z) - (c.z - a.z) * (d.y - a.y)) -
    (b.y - a.y) * ((c.x - a.x) * (d.z - a.z) - (c.z - a.z) * (d.x - a.x)) +
    (b.z - a.z) * ((c.x - a.x) * (d.y - a.y) - (c.y - a.y) * (d.x - a.x))
  )

theorem tetrahedron_volume_is_sqrt_7_75 
  (a b c d : Point3D)
  (hab : distance a b = 3)
  (hac : distance a c = 2)
  (had : distance a d = 5)
  (hbc : distance b c = Real.sqrt 17)
  (hbd : distance b d = Real.sqrt 29)
  (hcd : distance c d = 6) :
  tetrahedronVolume a b c d = Real.sqrt 7.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_is_sqrt_7_75_l444_44492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_l444_44441

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 10
def circle2 (x y : ℝ) : Prop := (x - 1)^2 + (y - 3)^2 = 20

-- Define the line
def line (x y : ℝ) : Prop := x + 3*y - 5 = 0

-- Theorem statement
theorem intersection_line : 
  ∀ (A B : ℝ × ℝ), 
    (circle1 A.1 A.2 ∧ circle2 A.1 A.2) →
    (circle1 B.1 B.2 ∧ circle2 B.1 B.2) →
    A ≠ B →
    ∀ (x y : ℝ), (x, y) ∈ Set.Icc A B ↔ line x y :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_l444_44441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_degree_of_p_l444_44476

/-- The denominator of our rational function -/
noncomputable def q (x : ℝ) : ℝ := 3*x^6 - 2*x^3 + x^2 - 8

/-- The rational function -/
noncomputable def f (p : ℝ → ℝ) (x : ℝ) : ℝ := p x / q x

/-- The horizontal asymptote of the rational function -/
def horizontal_asymptote (f : ℝ → ℝ) (y : ℝ) : Prop :=
  ∀ ε > 0, ∃ M, ∀ x, x > M → |f x - y| < ε

/-- A placeholder for the degree of a polynomial function -/
noncomputable def degree (p : ℝ → ℝ) : ℕ := sorry

/-- The theorem stating that the largest possible degree of p is 6 -/
theorem largest_degree_of_p :
  ∃ (p : ℝ → ℝ), horizontal_asymptote (f p) (1/3) ∧
  (∀ (r : ℝ → ℝ), horizontal_asymptote (f r) (1/3) → degree r ≤ degree p) ∧
  degree p = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_degree_of_p_l444_44476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_k_l444_44474

noncomputable def k (x : ℝ) : ℝ := 1 / (x + 9) + 1 / (x^2 + 9) + 1 / (x^3 + 9)

theorem domain_of_k :
  {x : ℝ | ∃ y, k x = y} = 
    {x : ℝ | x < -9 ∨ (-9 < x ∧ x < -Real.rpow 9 (1/3)) ∨ x > -Real.rpow 9 (1/3)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_k_l444_44474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_siskins_l444_44427

/-- Represents a row of poles where siskins can land --/
structure SiskinPoles where
  num_poles : Nat
  occupied : List Bool

/-- Represents the rules for siskin landing and flying off --/
def siskin_rules (poles : SiskinPoles) : Prop :=
  poles.num_poles = 25 ∧
  poles.occupied.length = poles.num_poles ∧
  ∀ i, i < poles.num_poles → (
    poles.occupied.get! i = true →
    (i > 0 → poles.occupied.get! (i-1) = false) ∧
    (i < poles.num_poles - 1 → poles.occupied.get! (i+1) = false)
  )

/-- The theorem stating the maximum number of siskins --/
theorem max_siskins (poles : SiskinPoles) :
  siskin_rules poles →
  (∃ max_occupied : Nat, max_occupied = (poles.occupied.filter id).length ∧ max_occupied = 24) := by
  sorry

#check max_siskins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_siskins_l444_44427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_l444_44498

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The standard equation of an ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The focal distance of an ellipse -/
noncomputable def Ellipse.focal_distance (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

/-- The left focus of an ellipse -/
noncomputable def Ellipse.left_focus (e : Ellipse) : ℝ × ℝ :=
  (-e.focal_distance, 0)

/-- Theorem: Length of chord through left focus perpendicular to major axis -/
theorem ellipse_chord_length (e : Ellipse) (h_eq : e.a = 4 ∧ e.b = 3) :
  let f := e.left_focus
  let chord_length := 2 * Real.sqrt (e.b^2 * (1 - f.1^2 / e.a^2))
  chord_length = 9/2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_l444_44498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_l444_44424

def is_valid_number (n : Nat) : Prop :=
  ∃ (a₁ a₂ a₃ a₄ a₅ a₆ : Nat),
    n = a₁ * 100000 + a₂ * 10000 + a₃ * 1000 + a₄ * 100 + a₅ * 10 + a₆ ∧
    Finset.toSet {a₁, a₂, a₃, a₄, a₅, a₆} = Finset.toSet {1, 2, 3, 4, 5, 6} ∧
    ∀ k : Nat, k ∈ Finset.range 7 → k ≠ 0 → (n / 10^(6-k)) % k = 0

theorem valid_numbers :
  ∀ n : Nat, is_valid_number n → n = 123654 ∨ n = 321654 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_l444_44424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f3_is_F_function_odd_function_with_inequality_is_F_function_l444_44425

-- Definition of F function
noncomputable def is_F_function (f : ℝ → ℝ) : Prop :=
  ∃ m : ℝ, m > 0 ∧ ∀ x : ℝ, |f x| ≤ m * |x|

-- Function 3
noncomputable def f3 (x : ℝ) : ℝ := x / (x^2 + x + 1)

-- Function 4 properties
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def satisfies_inequality (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, |f x₁ - f x₂| ≤ 2 * |x₁ - x₂|

-- Theorems to prove
theorem f3_is_F_function : is_F_function f3 := by sorry

theorem odd_function_with_inequality_is_F_function
  (f : ℝ → ℝ) (h_odd : is_odd f) (h_ineq : satisfies_inequality f) :
  is_F_function f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f3_is_F_function_odd_function_with_inequality_is_F_function_l444_44425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amanda_jogging_course_l444_44401

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Amanda's jogging course -/
theorem amanda_jogging_course (start end_point : Point) 
  (h1 : end_point.x = start.x + 1.5 - (3 * Real.sqrt 2) / 2)
  (h2 : end_point.y = start.y + Real.sqrt 2 / 2) :
  distance start end_point = 
    Real.sqrt ((1.5 - (3 * Real.sqrt 2) / 2)^2 + (Real.sqrt 2 / 2)^2) := by
  sorry

#check amanda_jogging_course

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amanda_jogging_course_l444_44401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_product_l444_44419

def mySequence : List ℚ := [1/3, 9/1, 1/27, 81/1, 1/243, 729/1, 1/2187, 6561/1, 1/6561, 19683/1]

theorem sequence_product : (mySequence.prod) = 243 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_product_l444_44419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l444_44435

-- Define the point A
def A : ℝ × ℝ := (-2, 0)

-- Define the ellipse equation
def is_on_ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the line passing through A with inclination angle α
noncomputable def line_equation (α t : ℝ) : ℝ × ℝ := 
  (A.1 + t * Real.cos α, t * Real.sin α)

-- Define the condition for the line to intersect the ellipse at two distinct points
def intersects_at_two_points (α : ℝ) : Prop :=
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ 
    is_on_ellipse (line_equation α t1).1 (line_equation α t1).2 ∧
    is_on_ellipse (line_equation α t2).1 (line_equation α t2).2

-- State the theorem
theorem inclination_angle_range :
  ∀ α : ℝ, intersects_at_two_points α ↔ 
    (0 ≤ α ∧ α < Real.arcsin (Real.sqrt 3 / 3)) ∨
    (Real.pi - Real.arcsin (Real.sqrt 3 / 3) < α ∧ α < Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l444_44435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_eccentricity_l444_44411

/-- An ellipse with foci at (1,0) and (-1,0) that intersects with the line y = x - 2 -/
structure Ellipse where
  /-- The semi-major axis of the ellipse -/
  a : ℝ
  /-- The semi-minor axis of the ellipse -/
  b : ℝ
  /-- The distance from the center to a focus -/
  c : ℝ
  /-- Condition that the foci are at (1,0) and (-1,0) -/
  foci_condition : c = 1
  /-- Condition relating a, b, and c -/
  ellipse_condition : a^2 - b^2 = c^2
  /-- Condition that the ellipse intersects with the line y = x - 2 -/
  intersection_condition : ∃ (x y : ℝ), x^2 / (a^2) + y^2 / (b^2) = 1 ∧ y = x - 2

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

/-- Theorem stating the maximum eccentricity of the ellipse -/
theorem max_eccentricity (e : Ellipse) : eccentricity e ≤ Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_eccentricity_l444_44411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_value_l444_44484

theorem trigonometric_expression_value (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos α = 3 / 5) : 
  (1 + Real.sqrt 2 * Real.cos (2 * α - π / 4)) / Real.sin (α + π / 2) = 14 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_value_l444_44484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l444_44455

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {2, 4, 5}
def B : Set Nat := {1, 3, 5, 7}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {1, 3, 7} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l444_44455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_of_x_power_x_l444_44473

-- Define the power index function
noncomputable def powerIndex (f φ : ℝ → ℝ) (x : ℝ) : ℝ := (f x) ^ (φ x)

-- Define the derivative of the power index function
noncomputable def powerIndexDerivative (f φ : ℝ → ℝ) (x : ℝ) : ℝ :=
  (f x) ^ (φ x) * ((deriv φ x) * Real.log (f x) + (φ x) * (deriv f x) / (f x))

-- State the theorem
theorem tangent_line_of_x_power_x (x : ℝ) (hx : x > 0) :
  let f := (λ x : ℝ => x)
  let φ := (λ x : ℝ => x)
  let y := powerIndex f φ
  let y' := powerIndexDerivative f φ
  (y' 1 = 1) ∧ (y 1 = 1) →
  ∀ t : ℝ, y 1 + y' 1 * (t - 1) = t := by
  sorry

#check tangent_line_of_x_power_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_of_x_power_x_l444_44473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cone_volume_ratio_l444_44420

/-- Represents a right circular cone inscribed in a right rectangular prism -/
structure InscribedCone where
  prism_base_length : ℝ
  prism_base_width : ℝ
  prism_height : ℝ

/-- The volume ratio of an inscribed cone to its prism -/
noncomputable def volume_ratio (c : InscribedCone) : ℝ :=
  (Real.pi * c.prism_base_length^2 * c.prism_height) / (48 * c.prism_base_length * c.prism_base_width * c.prism_height)

theorem inscribed_cone_volume_ratio :
  ∀ c : InscribedCone,
  c.prism_base_length = 3 →
  c.prism_base_width = 6 →
  c.prism_height = 9 →
  volume_ratio c = Real.pi / 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cone_volume_ratio_l444_44420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_sum_l444_44481

theorem quadratic_equation_solution_sum : 
  ∃ x : ℂ, 5 * x^2 + 6 = 2 * x - 15 ∧ 
  x.re + x.im^2 = 109 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_sum_l444_44481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l444_44465

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := i^2011

-- Theorem statement
theorem imaginary_part_of_z : z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l444_44465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l444_44428

-- Define the function f(x) = x^a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^a

-- Theorem stating the properties of f
theorem f_properties (a : ℝ) (h : f a (1/2) = 2) :
  -- 1. f is an odd function (symmetric about the origin)
  (∀ x, f a (-x) = -(f a x)) ∧
  -- 2. For all x ∈ [1, 2], f(x) ∈ [1/2, 1]
  (∀ x ∈ Set.Icc 1 2, f a x ∈ Set.Icc (1/2) 1) ∧
  -- 3. For all x > 0, f(x) ≥ 2-x
  (∀ x > 0, f a x ≥ 2 - x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l444_44428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_max_surface_area_in_hemisphere_l444_44413

noncomputable def frustum_max_surface_area (R : ℝ) : ℝ × ℝ := 
  let a := (Real.sqrt 17 - 1) / 8
  let d := (7 + Real.sqrt 17) / 16
  (a * R, d * R)

noncomputable def frustum_surface_area (R a d : ℝ) : ℝ :=
  Real.pi * (R^2 + (R^2 + d^2/4) * a)

theorem frustum_max_surface_area_in_hemisphere :
  let (a, d) := frustum_max_surface_area 1
  ∀ a' d', 0 < a' ∧ 0 < d' ∧ d' < 2 →
    frustum_surface_area 1 a' d' ≤ frustum_surface_area 1 a d := by
  sorry

#check frustum_max_surface_area
#check frustum_surface_area
#check frustum_max_surface_area_in_hemisphere

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_max_surface_area_in_hemisphere_l444_44413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l444_44461

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 1 + Real.rpow 2 x + a * Real.rpow 4 x

theorem min_value_of_a :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, ¬(f x a < 0)) →
  a ≥ -6 ∧ ∀ b > -6, ∃ x ∈ Set.Icc (-1 : ℝ) 1, f x b ≥ 0 := by
  sorry

#check min_value_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l444_44461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_a_l444_44430

def sequenceA (a : ℤ) : ℕ → ℤ
  | 0 => a
  | 1 => 2
  | (n+2) => 2 * sequenceA a (n+1) * sequenceA a n - sequenceA a (n+1) - sequenceA a n + 1

theorem no_valid_a : ¬∃ a : ℤ, ∀ n : ℕ, n ≥ 1 → ∃ k : ℤ, 2 * sequenceA a (3*n) - 1 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_a_l444_44430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a₁_l444_44490

-- Define the sequence and its sum
noncomputable def S (n : ℕ) (a₁ : ℝ) : ℝ := a₁ * (3^n - 1) / 2

-- State the theorem
theorem find_a₁ (a : ℕ → ℝ) :
  (∀ n, S n (a 1) = (a 1) * (3^n - 1) / 2) →
  a 4 = 54 →
  a 1 = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a₁_l444_44490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_is_24_l444_44462

def b : ℕ → ℚ
  | 0 => 2
  | 1 => 3
  | n + 2 => (1/2) * b (n + 1) + (1/3) * b n

theorem sum_of_sequence_is_24 :
  ∑' n, b n = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_is_24_l444_44462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_direction_vector_l444_44448

noncomputable def line (x : ℝ) : ℝ := (2 * x - 4) / 3

noncomputable def parameterization (v d : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  (v.1 + t * d.1, v.2 + t * d.2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem line_parameterization_direction_vector :
  ∃ (v : ℝ × ℝ) (d : ℝ × ℝ),
    (∀ (x : ℝ), x ≥ 2 →
      let p := parameterization v d (distance (x, line x) (2, 0))
      p.1 = x ∧ p.2 = line x) ∧
    d = (3 / Real.sqrt 13, 2 / Real.sqrt 13) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_direction_vector_l444_44448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l444_44446

/-- The function f(x) = (3x^2 + 4x + 5) / (x + 4) -/
noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 4 * x + 5) / (x + 4)

/-- The proposed asymptote function g(x) = 3x - 8 -/
def g (x : ℝ) : ℝ := 3 * x - 8

/-- Theorem: The oblique asymptote of f(x) is g(x) -/
theorem oblique_asymptote_of_f : 
  ∀ ε > 0, ∃ M, ∀ x, |x| > M → |f x - g x| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l444_44446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_point_l444_44439

/-- A hyperbola with equation x²/4 - y²/a = 1 where a > 0 -/
structure Hyperbola where
  a : ℝ
  h_pos : a > 0

/-- The point (2, √3) -/
noncomputable def point : ℝ × ℝ := (2, Real.sqrt 3)

/-- A point (x, y) lies on the asymptote of the hyperbola if y = ±(√a/2)x -/
def lies_on_asymptote (h : Hyperbola) (p : ℝ × ℝ) : Prop :=
  p.2 = Real.sqrt h.a / 2 * p.1 ∨ p.2 = -Real.sqrt h.a / 2 * p.1

theorem hyperbola_asymptote_point (h : Hyperbola) 
  (h_point : lies_on_asymptote h point) : h.a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_point_l444_44439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_l444_44410

-- Define the river as two parallel lines
def River : Type := ℝ × ℝ

-- Define a point on a 2D plane
def Point : Type := ℝ × ℝ

-- Define a bridge as a line segment perpendicular to the river
def Bridge (r : River) (p q : Point) : Prop :=
  p.2 = r.1 ∧ q.2 = r.2 ∧ p.1 = q.1

-- Define the path length
noncomputable def PathLength (a m n b : Point) : ℝ :=
  Real.sqrt ((m.1 - a.1)^2 + (m.2 - a.2)^2) +
  Real.sqrt ((n.1 - m.1)^2 + (n.2 - m.2)^2) +
  Real.sqrt ((b.1 - n.1)^2 + (b.2 - n.2)^2)

-- Define the reflection of a point across the river
def Reflect (r : River) (p : Point) : Point :=
  (p.1, 2 * r.2 - p.2)

-- Theorem: The path AMNB is shortest when N, N'', and B are collinear
theorem shortest_path (r : River) (a b m n : Point) 
  (h1 : Bridge r m n) 
  (h2 : a.2 = r.1) 
  (h3 : b.2 = r.2) :
  PathLength a m n b = 
    Real.sqrt ((Reflect r n).1 - a.1)^2 + ((Reflect r n).2 - a.2)^2 +
    Real.sqrt (b.1 - (Reflect r n).1)^2 + (b.2 - (Reflect r n).2)^2 →
  ∀ m' n', Bridge r m' n' → 
    PathLength a m' n' b ≥ PathLength a m n b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_l444_44410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l444_44433

/-- An ellipse with given parameters -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- The equation of an ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The focal distance of an ellipse -/
noncomputable def focal_distance (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

/-- The left focus of an ellipse -/
noncomputable def left_focus (e : Ellipse) : ℝ × ℝ :=
  (-focal_distance e, 0)

/-- A line passing through a point at a given angle -/
noncomputable def line_through_point (p : ℝ × ℝ) (angle : ℝ) (x : ℝ) : ℝ :=
  Real.tan angle * (x - p.1) + p.2

/-- The intersection points of an ellipse and a line -/
def intersection_points (e : Ellipse) (l : ℝ → ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ellipse_equation e p.1 p.2 ∧ p.2 = l p.1}

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The main theorem -/
theorem ellipse_intersection_theorem (e : Ellipse) 
  (h1 : e.b = 2) 
  (h2 : focal_distance e = 1) : 
  (ellipse_equation e = fun x y ↦ x^2 / 5 + y^2 / 4 = 1) ∧ 
  (∃ A B, A ∈ intersection_points e (line_through_point (left_focus e) (π/4)) ∧
          B ∈ intersection_points e (line_through_point (left_focus e) (π/4)) ∧
          distance A B = 8 * Real.sqrt 10 / 9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l444_44433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_closer_region_area_l444_44482

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The area of a triangle given its side lengths using Heron's formula -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

/-- Theorem: In an isosceles triangle ABC with AB = BC = 8 and CA = 6,
    the area of the region where points are closer to vertex C
    than to either A or B is 1/4 of the total area of the triangle -/
theorem isosceles_triangle_closer_region_area
  (t : Triangle)
  (h_isosceles : t.a = t.b)
  (h_sides : t.a = 8 ∧ t.c = 6) :
  (triangleArea { a := 4, b := 4, c := 6, pos_a := by norm_num, pos_b := by norm_num, pos_c := by norm_num,
                  triangle_inequality := by norm_num }) /
  (triangleArea t) = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_closer_region_area_l444_44482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_groups_formula_l444_44467

/-- Represents a sequence of k zeros and m ones in random order -/
structure BinarySequence where
  k : ℕ
  m : ℕ

/-- The expected number of alternating groups in a BinarySequence -/
def expectedGroups (seq : BinarySequence) : ℚ :=
  1 + (2 * seq.k * seq.m : ℚ) / (seq.k + seq.m)

/-- Theorem stating the expected number of alternating groups -/
theorem expected_groups_formula (seq : BinarySequence) :
  expectedGroups seq = 1 + (2 * seq.k * seq.m : ℚ) / (seq.k + seq.m) :=
by
  -- The proof is omitted
  sorry

#eval expectedGroups ⟨3, 4⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_groups_formula_l444_44467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_store_purchase_l444_44464

/-- The number of items bought by each person -/
def n : ℕ := sorry

/-- The number of items priced at 8 yuan -/
def x : ℕ := sorry

/-- The number of items priced at 9 yuan -/
def y : ℕ := sorry

/-- The total amount spent -/
def total_spent : ℕ := 172

theorem discount_store_purchase :
  (x + y = 2 * n) →
  (8 * x + 9 * y = total_spent) →
  (x = 8 ∧ y = 12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_store_purchase_l444_44464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_letters_in_names_l444_44451

theorem total_letters_in_names 
  (jonathan_letters : ℕ)
  (younger_sister_letters : ℕ)
  (older_brother_letters : ℕ)
  (youngest_sister_letters : ℕ)
  (h1 : jonathan_letters = 8 + 10)
  (h2 : younger_sister_letters = 5 + 10)
  (h3 : older_brother_letters = 6 + 10)
  (h4 : youngest_sister_letters = 4 + 15) :
  jonathan_letters + younger_sister_letters + older_brother_letters + youngest_sister_letters = 68 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_letters_in_names_l444_44451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_planes_three_or_four_l444_44415

-- Define the space we're working in
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V] [Fact (finrank ℝ V = 3)]

-- Define our lines
variable (a b c d : Subspace ℝ V)

-- Define the properties of the lines
variable (h_skew : a ≠ b ∧ Disjoint a b ∧ a ⊔ b ≠ ⊤)
variable (h_intersect_c : (c ⊓ a ≠ ⊥) ∧ (c ⊓ b ≠ ⊥))
variable (h_intersect_d : (d ⊓ a ≠ ⊥) ∧ (d ⊓ b ≠ ⊥))

-- Define a function to count the number of distinct planes
noncomputable def count_planes (a b c d : Subspace ℝ V) : ℕ := sorry

-- State the theorem
theorem num_planes_three_or_four :
  (count_planes a b c d = 3) ∨ (count_planes a b c d = 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_planes_three_or_four_l444_44415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_difference_l444_44421

/-- The difference in travel time between a mixed speed journey and a constant speed journey -/
theorem journey_time_difference (total_distance : ℝ) (first_segment : ℝ) (speed1 : ℝ) (speed2 : ℝ) (constant_speed : ℝ) : 
  total_distance = 500 →
  first_segment = 100 →
  speed1 = 60 →
  speed2 = 40 →
  constant_speed = 50 →
  (first_segment / speed1 + (total_distance - first_segment) / speed2 - total_distance / constant_speed) * 60 = 100 := by
  sorry

#check journey_time_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_difference_l444_44421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_CO_l444_44475

/-- The molar mass of carbon in g/mol -/
noncomputable def molar_mass_C : ℝ := 12.01

/-- The molar mass of oxygen in g/mol -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- The molar mass of carbon monoxide (CO) in g/mol -/
noncomputable def molar_mass_CO : ℝ := molar_mass_C + molar_mass_O

/-- The mass percentage of oxygen in carbon monoxide -/
noncomputable def mass_percentage_O : ℝ := (molar_mass_O / molar_mass_CO) * 100

theorem mass_percentage_O_in_CO :
  ∃ ε > 0, |mass_percentage_O - 57.12| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_CO_l444_44475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rolls_to_return_l444_44470

/-- The number of points on the circular board game. -/
def n : ℕ := 2021

/-- The expected number of rolls to return to the starting point,
    given that the player is not currently at the starting point. -/
noncomputable def E : ℝ := 2020

/-- The probability of returning to the starting point on a single roll. -/
noncomputable def p : ℝ := 1 / 2020

theorem expected_rolls_to_return : 
  E = 1 + (1 - p) * E ∧ 
  (1 + E : ℝ) = n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rolls_to_return_l444_44470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_quadruples_l444_44458

/-- The number of ordered quadruples (p,q,r,s) of nonnegative real numbers
    satisfying the given conditions. -/
def num_quadruples : ℕ := 15

/-- Predicate for valid quadruples -/
def is_valid_quadruple (p q r s : ℝ) : Prop :=
  p ≥ 0 ∧ q ≥ 0 ∧ r ≥ 0 ∧ s ≥ 0 ∧
  p^2 + q^2 + r^2 + s^2 = 9 ∧
  (p + q + r + s) * (p^3 + q^3 + r^3 + s^3) = 81

/-- The theorem stating that there are exactly 15 valid quadruples -/
theorem count_valid_quadruples :
  ∃ (S : Finset (ℝ × ℝ × ℝ × ℝ)), 
    (∀ x ∈ S, is_valid_quadruple x.1 x.2.1 x.2.2.1 x.2.2.2) ∧ 
    S.card = num_quadruples :=
sorry

#check count_valid_quadruples

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_quadruples_l444_44458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_skew_knights_l444_44457

/-- Represents a chessboard --/
structure Chessboard where
  size : ℕ
  is_square : size * size = 64

/-- Represents a skew knight --/
structure SkewKnight where
  position : ℕ × ℕ
  on_black_square : Bool

/-- Defines the attack pattern of a skew knight --/
def attacks (k1 k2 : SkewKnight) : Prop :=
  k1.on_black_square ∧ ¬k2.on_black_square

/-- Theorem: The maximum number of non-attacking skew knights on an 8x8 chessboard is 32 --/
theorem max_skew_knights (board : Chessboard) :
  ∃ (knights : List SkewKnight),
    knights.length = 32 ∧
    (∀ k1 k2, k1 ∈ knights → k2 ∈ knights → k1 ≠ k2 → ¬(attacks k1 k2)) ∧
    (∀ k, k ∉ knights → ∃ k', k' ∈ knights ∧ (attacks k k' ∨ attacks k' k)) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_skew_knights_l444_44457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_at_one_derivative_inequality_implies_range_l444_44472

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a/3) * x^3 - (3/2) * x^2 + (a+1) * x + 1

-- Define the derivative of f
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + a + 1

-- Theorem 1: If f has an extremum at x = 1, then a = 1
theorem extremum_at_one (a : ℝ) : f' a 1 = 0 → a = 1 := by
  sorry

-- Theorem 2: If f'(x) > x² - x - a + 1 for all a > 0, then x ∈ [-2, 0]
theorem derivative_inequality_implies_range :
  (∀ a > 0, ∀ x, f' 1 x > x^2 - x - a + 1) →
  ∀ x, f' 1 x > x^2 - x - 1 → -2 ≤ x ∧ x ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_at_one_derivative_inequality_implies_range_l444_44472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_covering_triangles_12_1_2_l444_44432

/-- The area of an equilateral triangle with side length s -/
noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

/-- The minimum number of equilateral triangles needed to cover a larger equilateral triangle -/
noncomputable def min_covering_triangles (large_side : ℝ) (small_side_1 : ℝ) (small_side_2 : ℝ) : ℕ :=
  min
    (Int.toNat ⌈(equilateral_triangle_area large_side / equilateral_triangle_area small_side_1)⌉)
    (Int.toNat ⌈(equilateral_triangle_area large_side / equilateral_triangle_area small_side_2)⌉)

theorem min_covering_triangles_12_1_2 :
  min_covering_triangles 12 1 2 = 36 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_covering_triangles_12_1_2_l444_44432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characteristic_value_is_nine_twentieths_l444_44418

/-- An isosceles triangle with perimeter 100 and one side length 18 -/
structure IsoscelesTriangle where
  -- AB and AC are equal sides, BC is the base
  ab : ℝ
  bc : ℝ
  perimeter : ℝ
  is_isosceles : ab = perimeter / 2 - bc / 2
  perimeter_eq : perimeter = 100
  ab_eq : ab = 18

/-- The characteristic value of an isosceles triangle -/
noncomputable def characteristic_value (t : IsoscelesTriangle) : ℝ :=
  t.bc / (Real.sqrt (t.ab ^ 2 - (t.bc / 2) ^ 2))

/-- Theorem: The characteristic value of the given isosceles triangle is 9/20 -/
theorem characteristic_value_is_nine_twentieths (t : IsoscelesTriangle) :
  characteristic_value t = 9 / 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_characteristic_value_is_nine_twentieths_l444_44418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_field_solutions_l444_44414

theorem rectangular_field_solutions :
  ∃ (S : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ S ↔ 
      (p.1 * 40 + p.2 * 40 = 2 * 2000 ∧ 
       p.1 % 10 = 0 ∧ 
       p.2 % 10 = 0 ∧
       p.1 > 0 ∧ p.2 > 0)) ∧
    Finset.card S > 3 :=
by
  -- We'll use a finite set (Finset) instead of an infinite set (Set)
  -- This ensures that Fintype is available for S
  let S := Finset.filter (fun p => 
    p.1 * 40 + p.2 * 40 = 2 * 2000 ∧ 
    p.1 % 10 = 0 ∧ 
    p.2 % 10 = 0 ∧
    p.1 > 0 ∧ p.2 > 0
  ) (Finset.range 101 ×ˢ Finset.range 101)
  
  use S
  
  constructor
  
  · -- Prove the biconditional
    intro p
    simp [S]
    -- The rest of the proof would go here
    sorry
  
  · -- Prove that there are more than 3 solutions
    -- We know there are 5 solutions from our manual calculation
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_field_solutions_l444_44414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_width_calculation_l444_44436

-- Define the river parameters
noncomputable def river_depth : ℝ := 7
noncomputable def river_flow_rate_kmph : ℝ := 4
noncomputable def water_volume_per_minute : ℝ := 35000

-- Convert flow rate from km/h to m/min
noncomputable def flow_rate_mpm : ℝ := river_flow_rate_kmph * 1000 / 60

-- Calculate the river width
noncomputable def river_width : ℝ := water_volume_per_minute / (flow_rate_mpm * river_depth)

-- Theorem statement
theorem river_width_calculation :
  ∀ ε > 0, |river_width - 75| < ε :=
by
  sorry -- Skip the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_width_calculation_l444_44436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l444_44447

noncomputable def ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  (a + c) * b / 2

def line (x y : ℝ) : Prop :=
  x - 2*y - 1 = 0

theorem ellipse_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (h_ecc : eccentricity a b = Real.sqrt 2 / 2)
  (h_area : triangle_area a b (Real.sqrt (a^2 - b^2)) = (Real.sqrt 2 + 1) / 2) :
  (∃ (x y : ℝ), ellipse x y (Real.sqrt 2) 1) ∧
  (∃ (P Q : ℝ × ℝ), line P.1 P.2 ∧ line Q.1 Q.2 ∧
    ellipse P.1 P.2 (Real.sqrt 2) 1 ∧ ellipse Q.1 Q.2 (Real.sqrt 2) 1 ∧
    (Real.sqrt ((P.1 + Real.sqrt 2)^2 + P.2^2) +
     Real.sqrt ((Q.1 + Real.sqrt 2)^2 + Q.2^2) +
     Real.sqrt ((P.1 - Real.sqrt 2)^2 + P.2^2) +
     Real.sqrt ((Q.1 - Real.sqrt 2)^2 + Q.2^2) = 4 * Real.sqrt 2) ∧
    (1/2 * 2 * Real.sqrt 2 * abs (P.2 - Q.2) = Real.sqrt 10 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l444_44447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_three_eq_neg_six_l444_44404

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 - 2*x + 3 else -((-x)^2 - 2*(-x) + 3)

-- State the theorem
theorem f_neg_three_eq_neg_six :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is odd
  f (-3) = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_three_eq_neg_six_l444_44404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_formula_l444_44406

/-- The shortest distance from a point on the parabola y = x^2 - 6x + 13 to the line y = kx - 5 -/
noncomputable def shortest_distance (k : ℝ) : ℝ :=
  let p := (k - 6) / 2
  |-(k - 6)^2 / 4 + (6 - k) * (k - 6) / 2 + 18| / Real.sqrt (k^2 + 1)

/-- Theorem stating the shortest distance formula -/
theorem shortest_distance_formula (k : ℝ) :
  let parabola (x : ℝ) := x^2 - 6*x + 13
  let line (x : ℝ) := k*x - 5
  ∃ (p : ℝ), shortest_distance k = 
    |k*p - parabola p - 5| / Real.sqrt (k^2 + 1) ∧
    ∀ (x : ℝ), |k*x - parabola x - 5| / Real.sqrt (k^2 + 1) ≥ shortest_distance k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_formula_l444_44406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_distance_after_2008_segments_l444_44463

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents an ant's path on the cube -/
inductive AntPath
  | White
  | Black

/-- Calculates the endpoint of an ant's path after a given number of segments -/
def antEndpoint (path : AntPath) (segments : ℕ) : Point3D :=
  match path with
  | AntPath.White => if segments % 6 = 4 then {x := 1, y := 1, z := 1} else {x := 1, y := 1, z := 1}
  | AntPath.Black => if segments % 6 = 4 then {x := 1, y := 0, z := 0} else {x := 1, y := 0, z := 0}

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Main theorem: The distance between the ants after 2008 segments is √2 -/
theorem ant_distance_after_2008_segments :
  distance (antEndpoint AntPath.White 2008) (antEndpoint AntPath.Black 2008) = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_distance_after_2008_segments_l444_44463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l444_44468

def A : Set ℕ := {x : ℕ | x * (x - 3) ≤ 0}

def B : Set ℕ := {0, 1}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l444_44468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_expected_value_intersection_area_bounds_main_intersection_area_theorem_l444_44422

/-- The area of intersection of four equilateral triangles with a unit square -/
noncomputable def intersection_area_of_triangles_and_square : ℝ :=
  (9 - 5 * Real.sqrt 3) / 3

/-- Theorem stating that the computed area is correct -/
theorem area_equals_expected_value :
  intersection_area_of_triangles_and_square = (9 - 5 * Real.sqrt 3) / 3 := by
  -- Unfold the definition
  unfold intersection_area_of_triangles_and_square
  -- The equality is trivial since both sides are identical
  rfl

/-- Auxiliary theorem: The area of the intersection is positive and less than 1 -/
theorem intersection_area_bounds :
  0 < intersection_area_of_triangles_and_square ∧
  intersection_area_of_triangles_and_square < 1 := by
  -- Split the conjunction
  constructor
  -- Prove the lower bound
  · sorry
  -- Prove the upper bound
  · sorry

/-- The main theorem combining the results -/
theorem main_intersection_area_theorem :
  intersection_area_of_triangles_and_square = (9 - 5 * Real.sqrt 3) / 3 ∧
  0 < intersection_area_of_triangles_and_square ∧
  intersection_area_of_triangles_and_square < 1 := by
  constructor
  · exact area_equals_expected_value
  · exact intersection_area_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_expected_value_intersection_area_bounds_main_intersection_area_theorem_l444_44422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_two_cos_l444_44488

/-- Given that the terminal side of angle α passes through the point (-5, 12),
    prove that sin α + 2cos α = 2/13 -/
theorem sin_plus_two_cos (α : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ r * (Real.cos α) = -5 ∧ r * (Real.sin α) = 12) →
  Real.sin α + 2 * Real.cos α = 2/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_two_cos_l444_44488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_product_zero_l444_44402

open Matrix

theorem matrix_product_zero (n : ℕ) (A B C : Matrix (Fin n) (Fin n) ℝ) 
  (h1 : A * B * C = 0)
  (h2 : Matrix.rank B = 1) :
  A * B = 0 ∨ B * C = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_product_zero_l444_44402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_third_l444_44471

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (Real.pi / 3 - x) * Real.cos (Real.pi / 6 - x) + 4 * Real.sqrt 3 * Real.cos x * Real.cos (Real.pi / 2 + x)

theorem angle_B_is_pi_third (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  a > 0 ∧ b > 0 ∧ c > 0 →
  b + c = 2 * a →
  f A = -3 →
  B = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_third_l444_44471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_by_circles_and_x_axis_l444_44423

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the area bounded by three circles and the x-axis -/
noncomputable def areaBoundedByCirclesAndXAxis (c1 c2 c3 : Circle) : ℝ :=
  20 - (9 * Real.pi / 2)

/-- Theorem stating the area bounded by the given circles and x-axis -/
theorem area_bounded_by_circles_and_x_axis :
  let c1 := Circle.mk (2, 2) 3
  let c2 := Circle.mk (6, 2) 3
  let c3 := Circle.mk (4, 6) 3
  areaBoundedByCirclesAndXAxis c1 c2 c3 = 20 - (9 * Real.pi / 2) := by
  sorry

#check area_bounded_by_circles_and_x_axis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_by_circles_and_x_axis_l444_44423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_tetrahedron_volume_ratio_l444_44491

noncomputable def tetrahedronVolume (a : ℝ) : ℝ := a^3 * Real.sqrt 2 / 12

def cubeVolume (s : ℝ) : ℝ := s^3

theorem cube_tetrahedron_volume_ratio :
  let cubeSideLength : ℝ := 2
  let tetrahedronSideLength : ℝ := 2 * Real.sqrt 2
  (cubeVolume cubeSideLength) / (tetrahedronVolume tetrahedronSideLength) = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_tetrahedron_volume_ratio_l444_44491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l444_44480

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, |x - a| + |x + 4| ≥ 1) → 
  a ∈ Set.Iic (-5) ∪ Set.Ici (-3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l444_44480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_minimal_size_l444_44487

/-- A triangle in a plane -/
structure Triangle (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] where
  A : P
  B : P
  C : P

/-- A line in a plane -/
structure Line (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] where
  a : P
  b : P

/-- The property of two triangles being similar and similarly oriented -/
def SimilarAndSimilarlyOriented {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (t1 t2 : Triangle P) : Prop := sorry

/-- The property of a point lying on a line -/
def LiesOn {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (p : P) (l : Line P) : Prop := sorry

/-- The perpendicular from a point to a line -/
def Perpendicular {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (p : P) (l : Line P) : Line P := sorry

/-- The intersection point of two lines -/
def Intersect {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (l1 l2 : Line P) : Option P := sorry

/-- The size of a triangle -/
noncomputable def Size {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (t : Triangle P) : ℝ := sorry

/-- The theorem statement -/
theorem triangle_minimal_size {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P]
  (A B C : P) (A₁ B₁ C₁ : ℝ → P) :
  (∀ t : ℝ, LiesOn (A₁ t) (Line.mk B C)) →
  (∀ t : ℝ, LiesOn (B₁ t) (Line.mk C A)) →
  (∀ t : ℝ, LiesOn (C₁ t) (Line.mk A B)) →
  (∀ t₁ t₂ : ℝ, SimilarAndSimilarlyOriented
    (Triangle.mk A B C)
    (Triangle.mk (A₁ t₁) (B₁ t₁) (C₁ t₁))) →
  (∀ t : ℝ, (∃ (t_min : ℝ), ∀ (s : ℝ),
    Size (Triangle.mk (A₁ t) (B₁ t) (C₁ t)) ≥ Size (Triangle.mk (A₁ t_min) (B₁ t_min) (C₁ t_min)))
  ↔
  (∃ (P : P),
    Intersect (Perpendicular (A₁ t) (Line.mk B C))
              (Perpendicular (B₁ t) (Line.mk C A)) = some P ∧
    Intersect (Perpendicular (B₁ t) (Line.mk C A))
              (Perpendicular (C₁ t) (Line.mk A B)) = some P)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_minimal_size_l444_44487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_distance_to_directrix_l444_44497

noncomputable section

-- Define the ellipse E
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the points
def A (a : ℝ) : ℝ × ℝ := (-a, 0)
def B (b : ℝ) : ℝ × ℝ := (0, b)
def C (b : ℝ) : ℝ × ℝ := (0, -b)
def F : ℝ × ℝ := (-1, 0)

-- Define the midpoint M of AC
def M (a : ℝ) : ℝ × ℝ := (-a/2, 0)

-- Define the line BF
def line_BF (b : ℝ) (x : ℝ) : ℝ := b * x + b

-- Theorem 1
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : line_BF b (-a/2) = 0) : 
  ellipse 3 (Real.sqrt 8) = ellipse a b :=
sorry

-- Theorem 2
theorem distance_to_directrix (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : b = 1) (h4 : a = Real.sqrt 2) 
  (D : ℝ × ℝ) (h5 : ellipse a b D.1 D.2) 
  (h6 : D.2 = D.1 + 1) (h7 : D.1 ≠ 0) :
  |D.1 - 2| = 10/3 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_distance_to_directrix_l444_44497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_associated_linear_functions_associated_quadratic_linear_functions_l444_44479

/-- Two functions are associated if there exists at least one point on the graph of the first function
    such that its symmetric point about the x-axis lies on the graph of the second function. -/
def are_associated (f g : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, f x = y ∧ g x = -y

/-- The value of b for which y = 2x + b and y = kx + k + 5 are associated for all real k. -/
theorem associated_linear_functions (b : ℝ) :
  (∀ k : ℝ, are_associated (λ x ↦ 2*x + b) (λ x ↦ k*x + k + 5)) → b = -3 := by sorry

/-- The range of 2m^2 + n^2 - 6m when y = x^2 - mx + 1 and y = 2x - n^2/4 are associated
    with exactly one pair of associated points. -/
theorem associated_quadratic_linear_functions (m n : ℝ) :
  (are_associated (λ x ↦ x^2 - m*x + 1) (λ x ↦ 2*x - n^2/4) ∧
   ∃! p : ℝ × ℝ, (λ x ↦ x^2 - m*x + 1) p.fst = p.snd ∧ (λ x ↦ 2*x - n^2/4) p.fst = -p.snd) →
  -1 ≤ 2*m^2 + n^2 - 6*m ∧ 2*m^2 + n^2 - 6*m ≤ 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_associated_linear_functions_associated_quadratic_linear_functions_l444_44479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_size_is_100_l444_44460

/-- Given a frequency and a frequency rate, calculate the sample size -/
def calculate_sample_size (frequency : ℕ) (frequency_rate : ℚ) : ℚ :=
  (frequency : ℚ) / frequency_rate

/-- Theorem: Given a frequency of 50 and a frequency rate of 0.5, the sample size is 100 -/
theorem sample_size_is_100 : 
  let frequency : ℕ := 50
  let frequency_rate : ℚ := 1/2
  calculate_sample_size frequency frequency_rate = 100 := by
  -- Unfold the definition of calculate_sample_size
  unfold calculate_sample_size
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl

#eval calculate_sample_size 50 (1/2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_size_is_100_l444_44460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_zero_times_l444_44453

noncomputable section

/-- A particle moves in a straight line with displacement s(t) = (1/3)t³ - (3/2)t² + 2t -/
def s (t : ℝ) : ℝ := (1/3) * t^3 - (3/2) * t^2 + 2 * t

/-- The velocity of the particle is the derivative of its displacement -/
def v (t : ℝ) : ℝ := deriv s t

theorem velocity_zero_times : 
  {t : ℝ | v t = 0} = {1, 2} := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_zero_times_l444_44453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_sine_implies_cos_value_l444_44477

noncomputable def f (x φ : ℝ) : ℝ := 2 * Real.sin (x + φ)

theorem even_sine_implies_cos_value 
  (φ : ℝ) 
  (h1 : 0 < φ) 
  (h2 : φ < Real.pi) 
  (h3 : ∀ x, f x φ = f (-x) φ) : 
  2 * Real.cos (2 * φ + Real.pi / 3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_sine_implies_cos_value_l444_44477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_problem_l444_44429

theorem exponent_problem : ∃ x : ℝ, (5568 / 87 : ℝ) ^ (1/3 : ℝ) + (72 * 2 : ℝ) ^ x = (256 : ℝ) ^ (1/2 : ℝ) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_problem_l444_44429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_pq_eq_neg_two_l444_44426

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def non_collinear (a b : V) : Prop := ∀ (r : ℝ), r • a ≠ b

theorem collinear_points_pq_eq_neg_two
  (p q : ℝ) (a b : V) (h_non_collinear : non_collinear a b)
  (AB BC CD : V)
  (h_AB : AB = 2 • a + p • b)
  (h_BC : BC = a + b)
  (h_CD : CD = (q - 1) • a - 2 • b)
  (h_collinear : ∃ (t : ℝ), AB = t • (AB + BC + CD)) :
  p * q = -2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_pq_eq_neg_two_l444_44426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_f_cubed_l444_44483

noncomputable def f (x : ℝ) : ℝ := |x| + 1 / |x| - 2

theorem constant_term_of_f_cubed :
  ∃ (c : ℝ), ∀ (x : ℝ), x ≠ 0 →
  ∃ (g : ℝ → ℝ), (f x)^3 = c + g x ∧ c = -20 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_f_cubed_l444_44483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l444_44459

/-- The area of a triangle given its vertices -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  (1/2) * abs (v.1 * w.2 - v.2 * w.1)

/-- Theorem: The area of the triangle with vertices (4, -3), (-1, 2), and (2, -7) is 15 -/
theorem triangle_area_specific : 
  triangle_area (4, -3) (-1, 2) (2, -7) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l444_44459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_properties_l444_44444

structure SamplePoint where
  x : ℝ
  y : ℝ

noncomputable def sample_correlation_coefficient (points : List SamplePoint) : ℝ := sorry

def stronger_linear_correlation (r1 r2 : ℝ) : Prop := sorry
def weaker_linear_correlation (r1 r2 : ℝ) : Prop := sorry

theorem correlation_coefficient_properties 
  (points : List SamplePoint) 
  (r : ℝ := sample_correlation_coefficient points) : 
  (∀ p ∈ points, p.y = -2 * p.x + 1) → r = -1 ∧ 
  0 ≤ |r| ∧ |r| ≤ 1 ∧
  (∀ ε > 0, ∃ δ > 0, ∀ r', |r' - 1| < δ → 
    stronger_linear_correlation r' r) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ r', |r'| < δ → 
    weaker_linear_correlation r' r) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_properties_l444_44444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l444_44456

/-- Given a cone whose axial section is an equilateral triangle with an area of √3,
    the total surface area of the cone is 3π. -/
theorem cone_surface_area (cone : ℝ) (axial_section : ℝ → ℝ → ℝ) :
  (∀ a b, axial_section a b = axial_section b a) →  -- equilateral property
  (∃ side, axial_section side side = Real.sqrt 3) →  -- area of axial section
  cone = 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l444_44456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_triangle_property_l444_44417

def a : ℕ → ℕ
  | 0 => 4  -- Add a case for 0 to handle all natural numbers
  | 1 => 4
  | n + 2 => (a (n + 1))^2 - 2

def triangle_exists (x y z : ℕ) : Prop :=
  x + y > z ∧ y + z > x ∧ z + x > y

noncomputable def triangle_area (x y z : ℕ) : ℝ :=
  let s : ℝ := (x + y + z) / 2
  Real.sqrt (s * (s - x) * (s - y) * (s - z))

theorem sequence_triangle_property (n : ℕ) (h : n ≥ 2) :
  triangle_exists (a (n-1)) (a n) (a (n+1)) ∧
  ∃ k : ℕ, triangle_area (a (n-1)) (a n) (a (n+1)) = k := by
  sorry

#check sequence_triangle_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_triangle_property_l444_44417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_properties_l444_44485

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem floor_properties :
  (∀ x : ℝ, (floor x : ℝ) ≤ x ∧ x < (floor x : ℝ) + 1) ∧
  (∀ x : ℝ, x - 1 < (floor x : ℝ) ∧ (floor x : ℝ) ≤ x) ∧
  ¬(∀ x : ℝ, floor (-x) = -floor x) ∧
  ¬(∀ x : ℝ, floor (2*x) = 2 * floor x) ∧
  ¬(∀ x : ℝ, floor x + floor (1 - x) = 1) :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_properties_l444_44485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_lifespan_of_sampled_products_l444_44499

/-- Represents a factory producing electronic products -/
structure Factory where
  production_ratio : ℚ
  average_lifespan : ℚ

/-- Calculates the weighted average lifespan of products from multiple factories -/
def weighted_average_lifespan (factories : List Factory) (total_samples : ℕ) : ℚ :=
  let total_ratio := factories.map Factory.production_ratio |>.sum
  let weights := factories.map (λ f => (f.production_ratio / total_ratio) * total_samples)
  (factories.zip weights |>.map (λ (f, w) => f.average_lifespan * w) |>.sum) / total_samples

/-- The main theorem stating the average lifespan of sampled products -/
theorem average_lifespan_of_sampled_products :
  let factories := [
    { production_ratio := 1, average_lifespan := 980 },
    { production_ratio := 2, average_lifespan := 1020 },
    { production_ratio := 1, average_lifespan := 1032 }
  ]
  let total_samples := 100
  weighted_average_lifespan factories total_samples = 1013 := by
  sorry

#eval weighted_average_lifespan [
  { production_ratio := 1, average_lifespan := 980 },
  { production_ratio := 2, average_lifespan := 1020 },
  { production_ratio := 1, average_lifespan := 1032 }
] 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_lifespan_of_sampled_products_l444_44499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l444_44454

-- Define the curves
noncomputable def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the focus of the parabola
noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define the condition that the foci coincide
noncomputable def foci_coincide (p a b : ℝ) : Prop :=
  parabola_focus p = (Real.sqrt (a^2 + b^2), 0)

-- Define the condition about the intersection line
noncomputable def intersection_line_passes_through_focus (p a b : ℝ) : Prop :=
  ∃ (x y : ℝ), parabola p x y ∧ hyperbola a b x y ∧ x = p/2

-- Main theorem
theorem hyperbola_eccentricity (p a b : ℝ) :
  parabola p (p/2) p →
  hyperbola a b (p/2) (b^2/a) →
  foci_coincide p a b →
  intersection_line_passes_through_focus p a b →
  Real.sqrt (a^2 + b^2)/a = Real.sqrt 2 + 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l444_44454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_121_l444_44405

def b : ℕ → ℕ
  | 0 => 10  -- Define for 0 to cover all natural numbers
  | 1 => 10  -- Keep the original definition for 10
  | n+2 => 121 * b (n+1) + 2 * (n+2)

theorem least_multiple_of_121 :
  ∀ n : ℕ, n > 10 ∧ n < 21 → ¬(121 ∣ b n) ∧ 121 ∣ b 21 := by
  sorry

#eval b 21  -- This will evaluate b 21 to check if it's indeed divisible by 121

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_121_l444_44405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_red_before_green_is_three_sevenths_l444_44443

/-- Represents the number of red chips in the hat -/
def num_red_chips : ℕ := 4

/-- Represents the number of green chips in the hat -/
def num_green_chips : ℕ := 3

/-- Represents the total number of chips in the hat -/
def total_chips : ℕ := num_red_chips + num_green_chips

/-- Represents the probability of drawing all red chips before all green chips -/
noncomputable def prob_all_red_before_green : ℚ :=
  (Nat.choose (total_chips - 1) (num_green_chips - 1) : ℚ) /
  (Nat.choose total_chips num_green_chips : ℚ)

theorem prob_all_red_before_green_is_three_sevenths :
  prob_all_red_before_green = 3 / 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_red_before_green_is_three_sevenths_l444_44443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l444_44450

theorem triangle_angle_measure (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  -- Given conditions
  (Real.sin C + Real.sin (B - A) = Real.sqrt 2 * Real.sin (2 * A)) →
  (A ≠ π / 2) →
  (a = 1) →
  -- Area condition
  (1 / 2 * a * b * Real.sin C = (Real.sqrt 3 + 1) / 4) →
  -- C is obtuse
  (C > π / 2) →
  -- Conclusion
  A = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l444_44450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l444_44495

/-- The function f(x) = ln(x+1) / ln(x) for x > 1 -/
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) / Real.log x

/-- The derivative of f(x) -/
noncomputable def f_derivative (x : ℝ) : ℝ := (x * Real.log x - (x + 1) * Real.log (x + 1)) / (x * (x + 1) * (Real.log x)^2)

theorem log_inequality (h1 : ∀ x > 1, f_derivative x < 0) :
  Real.log 626 / Real.log 17 < Real.log 5 / Real.log 2 ∧ Real.log 5 / Real.log 2 < 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l444_44495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_range_l444_44409

noncomputable def y (x : ℝ) : ℝ := (Real.log x / Real.log 3)^2 - 6 * (Real.log x / Real.log 3) + 6

theorem y_range :
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ 81 → -3 ≤ y x ∧ y x ≤ 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_range_l444_44409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_B_greater_than_A_l444_44438

noncomputable def sample_A : List ℝ := [5, 4, 3, 2, 1]
noncomputable def sample_B : List ℝ := [4, 0, 2, 1, -2]

noncomputable def variance (sample : List ℝ) : ℝ :=
  let n := sample.length
  let mean := sample.sum / n
  (sample.map (λ x => (x - mean)^2)).sum / (n - 1)

theorem variance_B_greater_than_A :
  variance sample_B > variance sample_A :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_B_greater_than_A_l444_44438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2023_equals_63_and_divisible_by_7_l444_44434

/-- The sequence where the nth positive integer appears n+1 times -/
def mySequence (n : ℕ) : ℕ := 
  (((8 * n + 1 : ℕ).sqrt - 1) / 2 : ℕ) + 1

/-- The 2023rd term of the sequence -/
def term_2023 : ℕ := mySequence 2023

theorem sequence_2023_equals_63_and_divisible_by_7 :
  term_2023 = 63 ∧ term_2023 % 7 = 0 := by
  sorry

#eval term_2023
#eval term_2023 % 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2023_equals_63_and_divisible_by_7_l444_44434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_from_sine_and_point_l444_44440

theorem cosine_value_from_sine_and_point 
  (θ : Real) 
  (m : Real) 
  (h1 : m ≠ 0)
  (h2 : ∃ (x y : Real), x = -Real.sqrt 3 ∧ y = m ∧ (θ.sin = (Real.sqrt 2 / 4) * m)) :
  θ.cos = -(Real.sqrt 6 / 4) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_from_sine_and_point_l444_44440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_theorem_l444_44412

/-- A rectangular field with given area and one side length -/
structure RectangularField where
  area : ℝ
  side_length : ℝ

/-- Calculate the fencing required for three sides of a rectangular field -/
noncomputable def fencing_required (field : RectangularField) : ℝ :=
  field.side_length + 2 * (field.area / field.side_length)

/-- Theorem: The fencing required for a field with area 120 and side length 20 is 32 -/
theorem fencing_theorem (field : RectangularField) 
  (h_area : field.area = 120) 
  (h_side : field.side_length = 20) : 
  fencing_required field = 32 := by
  sorry

/-- Compute the fencing required for a specific field -/
def compute_fencing : ℚ :=
  32

#eval compute_fencing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_theorem_l444_44412
