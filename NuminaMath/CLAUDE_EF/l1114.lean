import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_minimum_l1114_111430

theorem perfect_square_minimum (m n : ℕ+) :
  (∀ x : ℝ, ∃ y : ℝ, x^2 + m.val * x + (10 + n.val) = y^2) →
  (n.val = 6 ∧ (m.val = 8 ∨ m.val = 8)) ∧
  (∀ k : ℕ+, k.val < n.val → ¬∃ l : ℕ+, ∀ x : ℝ, ∃ y : ℝ, x^2 + l.val * x + (10 + k.val) = y^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_minimum_l1114_111430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1114_111454

noncomputable section

-- Define the ellipse (C)
def Ellipse (x y : ℝ) : Prop :=
  x^2/16 + y^2/12 = 1

-- Define the parabola
def Parabola (x y : ℝ) : Prop :=
  x^2 = 8 * Real.sqrt 3 * y

-- Define the line x = -2
def Line (x : ℝ) : Prop :=
  x = -2

-- Define the slope of line AB
def Slope (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (y₂ - y₁) / (x₂ - x₁) = 1/2

-- Define the area of quadrilateral APBQ
noncomputable def Area (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  3 * Real.sqrt (48 - 3 * ((y₂ - y₁) / (x₂ - x₁))^2)

theorem ellipse_properties :
  -- The ellipse has center at origin, foci on x-axis, and eccentricity 1/2
  ∃ (a b c : ℝ), a > b ∧ b > 0 ∧ c/a = 1/2 ∧ a^2 - b^2 = c^2 ∧
  -- One vertex of the ellipse coincides with the focus of the parabola
  ∃ (x₀ y₀ : ℝ), Ellipse x₀ y₀ ∧ Parabola x₀ y₀ ∧
  -- The line x = -2 intersects the ellipse at two points
  ∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧ Ellipse (-2) y₁ ∧ Ellipse (-2) y₂ ∧
  -- A and B are on opposite sides of x = -2
  ∀ (x₁ y₁ x₂ y₂ : ℝ), Ellipse x₁ y₁ ∧ Ellipse x₂ y₂ ∧ Slope x₁ y₁ x₂ y₂ →
    (x₁ + 2) * (x₂ + 2) < 0 →
    -- The maximum area of APBQ is 12√3
    Area x₁ y₁ x₂ y₂ ≤ 12 * Real.sqrt 3 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1114_111454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l1114_111473

/-- A rhombus that circumscribes a circle of radius 1 with smaller angles of 60° has an area of 8√3/3 -/
theorem rhombus_area (r : ℝ) (θ : ℝ) :
  r = 1 →
  θ = π / 3 →
  (4 * r^2 * Real.tan θ) = (8 * Real.sqrt 3) / 3 := by
  intros hr hθ
  rw [hr, hθ]
  norm_num
  ring
  sorry  -- Placeholder for the detailed proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l1114_111473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1114_111431

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the equation
def equation (x : ℝ) : Prop := 3 * x^3 - (floor x : ℝ) = 3

-- State the theorem
theorem unique_solution :
  ∃! x : ℝ, equation x ∧ x = (4/3 : ℝ)^(1/3 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1114_111431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yan_distance_ratio_l1114_111463

/-- Yan's walking speed -/
noncomputable def w : ℝ := sorry

/-- Distance from Yan to his home -/
noncomputable def x : ℝ := sorry

/-- Distance from Yan to the library -/
noncomputable def y : ℝ := sorry

/-- Yan's bicycle speed -/
noncomputable def bicycle_speed : ℝ := 5 * w

/-- Time taken to walk directly to the library -/
noncomputable def time_walk_library : ℝ := y / w

/-- Time taken to walk home and then ride bicycle to library -/
noncomputable def time_walk_home_ride_library : ℝ := x / w + (x + y) / bicycle_speed

theorem yan_distance_ratio :
  x > 0 ∧ y > 0 ∧ w > 0 ∧ time_walk_library = time_walk_home_ride_library →
  x / y = 2 / 3 := by
  sorry

#check yan_distance_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yan_distance_ratio_l1114_111463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_domain_M_l1114_111499

-- Define the function f (marked as noncomputable due to Real.sqrt)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 - 4)

-- Define the domain M
def M : Set ℝ := {x : ℝ | x ≤ -2 ∨ x ≥ 2}

-- Theorem statement
theorem complement_of_domain_M :
  Set.compl M = Set.Ioo (-2 : ℝ) (2 : ℝ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_domain_M_l1114_111499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_digit_is_nine_l1114_111421

def square_sequence (n : ℕ) : List ℕ :=
  (List.range n).map (λ x => (x + 1) ^ 2)

def digits_of_nat (n : ℕ) : List ℕ :=
  n.repr.data.map (λ c => c.toNat - '0'.toNat)

def digit_at_position (seq : List ℕ) (pos : ℕ) : Option ℕ := do
  let digits := seq.bind digits_of_nat
  digits[pos - 1]?

theorem hundredth_digit_is_nine :
  digit_at_position (square_sequence 99) 100 = some 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_digit_is_nine_l1114_111421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1114_111490

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := Real.exp (2 * x) - a * x^2 + b * x - 1

-- Define the second derivative of f
noncomputable def f'' (a : ℝ) (x : ℝ) : ℝ := 4 * Real.exp (2 * x) - 2 * a

theorem a_range (a b : ℝ) : 
  (∃ b, f a b 1 = 0) →
  (∃ x y, 0 < x ∧ x < y ∧ y < 1 ∧ f'' a x = 0 ∧ f'' a y = 0) →
  a > Real.exp 2 - 3 ∧ a < Real.exp 2 + 1 :=
by
  sorry

#check a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1114_111490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_theorem_l1114_111433

/-- Represents a sampling scenario --/
structure Scenario where
  numbers : List Nat
  deriving Repr

/-- Checks if a scenario is a valid stratified sampling --/
def isStratifiedSampling (s : Scenario) (totalStudents firstGrade secondGrade thirdGrade sampleSize : Nat) : Prop :=
  s.numbers.length = sampleSize ∧
  (s.numbers.filter (λ x => x ≤ firstGrade)).length = 4 ∧
  (s.numbers.filter (λ x => firstGrade < x ∧ x ≤ firstGrade + secondGrade)).length = 3 ∧
  (s.numbers.filter (λ x => firstGrade + secondGrade < x ∧ x ≤ totalStudents)).length = 3

/-- Checks if a scenario is a valid systematic sampling --/
def isSystematicSampling (s : Scenario) (totalStudents sampleSize : Nat) : Prop :=
  s.numbers.length = sampleSize ∧
  ∃ k, (List.range sampleSize).map (λ i => k * (totalStudents / sampleSize) + i + 1) = s.numbers

/-- The main theorem --/
theorem sampling_theorem (totalStudents firstGrade secondGrade thirdGrade sampleSize : Nat)
  (scenario1 scenario4 : Scenario) :
  totalStudents = 270 →
  firstGrade = 108 →
  secondGrade = 81 →
  thirdGrade = 81 →
  sampleSize = 10 →
  scenario1.numbers = [5, 9, 100, 107, 111, 121, 180, 195, 200, 265] →
  scenario4.numbers = [11, 38, 60, 90, 119, 146, 173, 200, 227, 254] →
  (isStratifiedSampling scenario1 totalStudents firstGrade secondGrade thirdGrade sampleSize ∧
   ¬isSystematicSampling scenario1 totalStudents sampleSize) ∧
  (isStratifiedSampling scenario4 totalStudents firstGrade secondGrade thirdGrade sampleSize ∧
   ¬isSystematicSampling scenario4 totalStudents sampleSize) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_theorem_l1114_111433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_duty_roster_arrangements_l1114_111444

/-- The number of students -/
def num_students : ℕ := 5

/-- The number of days in the duty roster -/
def num_days : ℕ := 5

/-- The number of possible days for student A to be on duty -/
def student_a_options : ℕ := 2

/-- Calculate the total number of possible arrangements for the duty roster -/
def total_arrangements : ℕ := student_a_options * Nat.factorial (num_students - 1)

/-- Theorem stating that the total number of arrangements is 48 -/
theorem duty_roster_arrangements :
  total_arrangements = 48 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_duty_roster_arrangements_l1114_111444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1114_111492

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (a * Real.sin B = b * Real.sin A) →
  (b * Real.sin C = c * Real.sin B) →
  (c * Real.sin A = a * Real.sin C) →
  ((2 * a - b) * Real.cos C = c * Real.cos B) →
  (c = 2) →
  (1 / 2 * a * b * Real.sin C = Real.sqrt 3) →
  (C = π / 3 ∧ a + b + c = 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1114_111492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l1114_111417

/-- A structure representing a triplet of real numbers satisfying the given equations -/
structure SolutionTriplet where
  a : ℝ
  b : ℝ
  c : ℝ
  eq1 : a + b + c = 1/a + 1/b + 1/c
  eq2 : a^2 + b^2 + c^2 = 1/a^2 + 1/b^2 + 1/c^2

/-- The set of all valid solution triplets -/
def SolutionSet : Set SolutionTriplet := {t | t.a ≠ 0 ∧ t.b ≠ 0 ∧ t.c ≠ 0}

/-- Function to generate solution triplets for s₃ = 1 case -/
noncomputable def solutionForS3Pos (k : ℝ) : SolutionTriplet :=
  { a := 1
    b := (k - 1 + Real.sqrt ((k - 1)^2 - 4)) / 2
    c := (k - 1 - Real.sqrt ((k - 1)^2 - 4)) / 2
    eq1 := by sorry
    eq2 := by sorry }

/-- Function to generate solution triplets for s₃ = -1 case -/
noncomputable def solutionForS3Neg (k : ℝ) : SolutionTriplet :=
  { a := -1
    b := (k + 1 + Real.sqrt ((k + 1)^2 - 4)) / 2
    c := (k + 1 - Real.sqrt ((k + 1)^2 - 4)) / 2
    eq1 := by sorry
    eq2 := by sorry }

/-- Theorem stating that all solutions are of the specified form -/
theorem solution_characterization :
  ∀ t ∈ SolutionSet,
    (∃ k : ℝ, |k - 1| ≥ 2 ∧
      (t = solutionForS3Pos k ∨
       t = { a := (solutionForS3Pos k).b, b := (solutionForS3Pos k).c, c := (solutionForS3Pos k).a, eq1 := by sorry, eq2 := by sorry } ∨
       t = { a := (solutionForS3Pos k).c, b := (solutionForS3Pos k).a, c := (solutionForS3Pos k).b, eq1 := by sorry, eq2 := by sorry })) ∨
    (∃ k : ℝ, |k + 1| ≥ 2 ∧
      (t = solutionForS3Neg k ∨
       t = { a := (solutionForS3Neg k).b, b := (solutionForS3Neg k).c, c := (solutionForS3Neg k).a, eq1 := by sorry, eq2 := by sorry } ∨
       t = { a := (solutionForS3Neg k).c, b := (solutionForS3Neg k).a, c := (solutionForS3Neg k).b, eq1 := by sorry, eq2 := by sorry })) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l1114_111417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_QF_distance_l1114_111407

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 16*x

-- Define the focus F
def focus : ℝ × ℝ := (4, 0)

-- Define the directrix l
def directrix (x : ℝ) : ℝ := -4

-- Define a point on the directrix
variable (P : ℝ × ℝ)

-- Define Q as a point on the parabola
variable (Q : ℝ × ℝ)

-- Assumption that P is on the directrix
axiom P_on_directrix : P.2 = directrix P.1

-- Assumption that Q is on the parabola
axiom Q_on_parabola : parabola Q.1 Q.2

-- Assumption that PF = 4FQ
axiom PF_eq_4FQ : 
  (P.1 - focus.1)^2 + (P.2 - focus.2)^2 = 
  16 * ((Q.1 - focus.1)^2 + (Q.2 - focus.2)^2)

-- Theorem to prove
theorem QF_distance : 
  (Q.1 - focus.1)^2 + (Q.2 - focus.2)^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_QF_distance_l1114_111407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1114_111410

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 / (1 - x) + Real.log (1 + x)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -1 < x ∧ x ≠ 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1114_111410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frame_area_theorem_l1114_111424

theorem frame_area_theorem (outer_side_length : ℝ) (frame_width : ℝ) (num_frames : ℕ) :
  outer_side_length = 6 →
  frame_width = 1 →
  num_frames = 4 →
  let inner_side_length := outer_side_length - 2 * frame_width
  let frame_area := outer_side_length^2 - inner_side_length^2
  let total_frame_area := (num_frames : ℝ) * frame_area
  let overlap_area := ((num_frames : ℝ) * ((num_frames : ℝ) - 1) / 2) * frame_width^2
  total_frame_area - overlap_area = 74 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frame_area_theorem_l1114_111424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_centers_distance_l1114_111418

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is internally tangent to another circle -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c2.radius - c1.radius)^2

/-- Checks if three points are collinear -/
def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem circle_centers_distance (c1 c2 c3 : Circle) :
  are_externally_tangent c1 c2 →
  is_internally_tangent c1 c3 →
  is_internally_tangent c2 c3 →
  are_collinear c1.center c2.center c3.center →
  c1.radius = 6 →
  c2.radius = 14 →
  distance c1.center c3.center = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_centers_distance_l1114_111418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_ratio_l1114_111400

def projection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![1/18, -2/9],
    ![-2/9,  8/9]]

theorem projection_ratio :
  ∃ (a b : ℚ), a ≠ 0 ∧
  (∀ (x : Fin 2 → ℚ), Matrix.mulVec projection_matrix (![a, b]) = ![a, b]) →
  b / a = -17/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_ratio_l1114_111400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l1114_111467

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  2 * x^2 - 3 * y^2 + 8 * x - 12 * y - 8 = 0

/-- Definition of a focus for a hyperbola -/
def is_focus (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (X Y : ℝ), hyperbola_eq X Y ↔ 
    ((X - (x - Real.sqrt (a^2 + b^2)))^2 / a^2) - ((Y - y)^2 / b^2) = 1 ∨
    ((X - (x + Real.sqrt (a^2 + b^2)))^2 / a^2) - ((Y - y)^2 / b^2) = 1

/-- Theorem: One focus of the given hyperbola -/
theorem hyperbola_focus : is_focus (-2 + Real.sqrt 30 / 3) (-2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l1114_111467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinates_l1114_111405

/-- Curve C defined by parametric equations -/
noncomputable def C (α : Real) : Real × Real :=
  (3 + 2 * Real.cos α, 2 * Real.sqrt 3 + 2 * Real.sin α)

/-- Line l passing through origin -/
noncomputable def l (θ : Real) : Real → Real × Real := fun ρ ↦ (ρ * Real.cos θ, ρ * Real.sin θ)

/-- Theorem stating the rectangular coordinates of point D -/
theorem midpoint_coordinates (A B : Real × Real) (ρ₀ : Real) :
  let D := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ∃ α₁ α₂ : Real, C α₁ = A ∧ C α₂ = B ∧
  l (Real.pi / 3) ρ₀ = D ∧
  D = (9 / 4, 9 * Real.sqrt 3 / 4) := by
  sorry

#check midpoint_coordinates

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinates_l1114_111405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_select_two_divisible_by_100_l1114_111477

theorem select_two_divisible_by_100 (S : Finset ℤ) (h : S.card = 52) :
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (100 ∣ (a + b) ∨ 100 ∣ (a - b)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_select_two_divisible_by_100_l1114_111477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_phase_shift_odd_condition_l1114_111449

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The sine function with a phase shift -/
noncomputable def SineWithPhase (φ : ℝ) : ℝ → ℝ := λ x ↦ Real.sin (2 * x + φ)

theorem sine_phase_shift_odd_condition (φ : ℝ) :
  (φ = π → IsOdd (SineWithPhase φ)) ∧
  (IsOdd (SineWithPhase φ) → ∃ k : ℤ, φ = k * π) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_phase_shift_odd_condition_l1114_111449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_15_plus_cos_15_l1114_111429

theorem sin_15_plus_cos_15 : Real.sin (15 * π / 180) + Real.cos (15 * π / 180) = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_15_plus_cos_15_l1114_111429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_culture_medium_greatest_impact_l1114_111416

/- Define the factors that can impact the experiment -/
inductive ExperimentalFactor
  | SoilSampling
  | SpreaderSterilization
  | CultureMediumComposition
  | HandDisinfection

/- Define the impact level of a factor -/
inductive ImpactLevel
  | Low
  | Medium
  | High

/- Define a function to assess the impact of a factor -/
noncomputable def assessImpact : ExperimentalFactor → ImpactLevel :=
  fun factor => match factor with
    | ExperimentalFactor.CultureMediumComposition => ImpactLevel.High
    | _ => ImpactLevel.Low

/- Define what it means for a factor to have the greatest impact -/
def hasGreatestImpact (factor : ExperimentalFactor) : Prop :=
  ∀ other : ExperimentalFactor, assessImpact factor = ImpactLevel.High ∧ 
    (other ≠ factor → assessImpact other ≠ ImpactLevel.High)

/- State the theorem -/
theorem culture_medium_greatest_impact :
  hasGreatestImpact ExperimentalFactor.CultureMediumComposition := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_culture_medium_greatest_impact_l1114_111416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_surface_area_l1114_111486

/-- Represents the lengths of available sticks in centimeters -/
def sticks : List ℝ := [2, 3, 5, 6, 9]

/-- Calculates the surface area of a rectangular box given three edge lengths -/
def surfaceArea (a b c : ℝ) : ℝ := 2 * (a * b + b * c + a * c)

/-- Checks if three lengths can form valid edges of a rectangular box -/
def isValidBox (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  (∃ x y z, x ∈ sticks ∧ y ∈ sticks ∧ z ∈ sticks ∧ (a = x ∨ a = x + y)) ∧ 
  (∃ x y z, x ∈ sticks ∧ y ∈ sticks ∧ z ∈ sticks ∧ (b = x ∨ b = x + y)) ∧ 
  (∃ x y z, x ∈ sticks ∧ y ∈ sticks ∧ z ∈ sticks ∧ (c = x ∨ c = x + y))

/-- Theorem stating the maximum surface area of the rectangular box -/
theorem max_surface_area : 
  (∃ a b c, isValidBox a b c ∧ 
   ∀ x y z, isValidBox x y z → surfaceArea x y z ≤ surfaceArea a b c) ∧
  (∃ a b c, isValidBox a b c ∧ surfaceArea a b c = 416) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_surface_area_l1114_111486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_donald_duck_downstream_time_l1114_111448

/-- Represents the swimming scenario for Donald Duck -/
structure SwimmingScenario where
  pool_length : ℝ
  upstream_time : ℝ
  still_water_time : ℝ

/-- Calculates the downstream swimming time given a swimming scenario -/
noncomputable def downstream_time (s : SwimmingScenario) : ℝ :=
  let still_water_speed := s.pool_length / s.still_water_time
  let current_speed := (s.pool_length / s.upstream_time - still_water_speed) / -2
  s.pool_length / (still_water_speed + current_speed)

/-- Theorem stating that the downstream time is 40 seconds for the given scenario -/
theorem donald_duck_downstream_time :
  let scenario : SwimmingScenario := {
    pool_length := 2000,  -- 2 kilometers in meters
    upstream_time := 60,
    still_water_time := 48  -- Rounded for simplicity
  }
  downstream_time scenario = 40 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_donald_duck_downstream_time_l1114_111448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tennis_ball_ratio_l1114_111434

theorem tennis_ball_ratio 
  (total : ℕ) 
  (extra_yellow : ℕ) 
  (new_ratio : ℚ) 
  (h1 : total = 64) 
  (h2 : extra_yellow = 20) 
  (h3 : new_ratio = 8 / 13) : 
  (total / 2 : ℚ) / (total / 2 : ℚ) = 1 := by
  have original_white := total / 2
  have original_yellow := total / 2
  have new_white := original_white
  have new_yellow := original_yellow + extra_yellow
  have h4 : new_ratio = (new_white : ℚ) / new_yellow := by sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tennis_ball_ratio_l1114_111434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_neznaika_error_is_0_16_percent_l1114_111488

/-- Conversion factor from kilolunas to kilograms -/
noncomputable def kiloluna_to_kg : ℝ := 0.24

/-- Neznaika's method to convert kg to kilolunas -/
noncomputable def neznaika_kg_to_kiloluna (kg : ℝ) : ℝ := kg * 4 * 1.04

/-- Correct method to convert kg to kilolunas -/
noncomputable def correct_kg_to_kiloluna (kg : ℝ) : ℝ := kg / kiloluna_to_kg

/-- Percentage error of Neznaika's method -/
noncomputable def neznaika_error : ℝ := 
  (1 - neznaika_kg_to_kiloluna 1 / correct_kg_to_kiloluna 1) * 100

theorem neznaika_error_is_0_16_percent :
  abs (neznaika_error - 0.16) < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_neznaika_error_is_0_16_percent_l1114_111488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_c_is_hyperbola_l1114_111401

/-- Given an ellipse and a curve C, prove that C is a hyperbola with a specific equation -/
theorem curve_c_is_hyperbola (x y : ℝ) (C : Set (ℝ × ℝ)) (dist : ℝ × ℝ → ℝ × ℝ → ℝ) :
  (∃ a b : ℝ, a = 13 ∧ b = 12 ∧ x^2 / a^2 + y^2 / b^2 = 1) →
  (∃ f₁ f₂ : ℝ × ℝ, ∀ p : ℝ × ℝ, p ∈ C → |dist p f₁ - dist p f₂| = 8) →
  (∃ a' b' : ℝ, a' = 4 ∧ b' = 3 ∧ x^2 / a'^2 - y^2 / b'^2 = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_c_is_hyperbola_l1114_111401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_even_ratio_l1114_111458

/-- The sum of pairwise products of a sequence of 2n binary digits -/
def S (n : ℕ+) (x : Fin (2 * n) → Fin 2) : ℕ :=
  (Finset.range n).sum (λ i => x (2 * i) * x (2 * i + 1))

/-- The number of sequences of 2n binary digits where S is odd -/
def O (n : ℕ+) : ℕ := Finset.card (Finset.filter (λ x => S n x % 2 = 1) (Finset.univ))

/-- The number of sequences of 2n binary digits where S is even -/
def E (n : ℕ+) : ℕ := Finset.card (Finset.filter (λ x => S n x % 2 = 0) (Finset.univ))

/-- The main theorem stating the ratio of odd to even sequences -/
theorem odd_even_ratio (n : ℕ+) : (O n : ℚ) / (E n : ℚ) = (2^(n:ℕ) - 1) / (2^(n:ℕ) + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_even_ratio_l1114_111458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_nested_sqrt_l1114_111432

noncomputable def nested_sqrt (x y : ℝ) : ℕ → ℝ
| 0 => y
| n + 1 => x + Real.sqrt (nested_sqrt x y n)

theorem unique_solution_nested_sqrt :
  ∃! (x y : ℝ), x + y = 6 ∧ y = nested_sqrt x y 1974 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_nested_sqrt_l1114_111432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_condition_l1114_111483

/-- Given vectors a and b in ℝ², if (11a - 2018b) ∥ (10a + 2017b), then lambda = -17/3 -/
theorem vector_parallel_condition (a b : ℝ × ℝ) (lambda : ℝ) 
  (h1 : a = (3, 2))
  (h2 : b = (-7, lambda + 1))
  (h3 : ∃ (t : ℝ), 11 • a - 2018 • b = t • (10 • a + 2017 • b)) :
  lambda = -17/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_condition_l1114_111483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_and_range_l1114_111451

def S (n : ℕ) (a : ℝ) : ℝ :=
  if n ≤ 4 then 2^n - 1 else -n^2 + (a-1)*n

def a_n (n : ℕ) (a : ℝ) : ℝ :=
  if n ≤ 4 then 2^(n-1)
  else if n = 5 then 5*a - 45
  else -2*n + a

theorem sequence_and_range (a : ℝ) :
  (∀ n : ℕ, a_n n a = S n a - S (n-1) a) ∧
  (∀ n : ℕ, a_n 5 a ≥ a_n n a) →
  a ≥ 53/5 := by
  sorry

#check sequence_and_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_and_range_l1114_111451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_specific_angles_l1114_111446

theorem tan_sum_specific_angles (α β : Real) 
  (h1 : Real.tan α = 5) 
  (h2 : Real.tan β = 3) : 
  Real.tan (α + β) = -4/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_specific_angles_l1114_111446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1114_111487

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - Real.log x)

-- State the theorem
theorem f_domain : ∀ x : ℝ, (0 < x ∧ x ≤ Real.exp 1) ↔ (∃ y : ℝ, f x = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1114_111487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_lighthouse_height_ellas_model_height_l1114_111426

/-- Represents a lighthouse with its height and base volume -/
structure Lighthouse where
  height : ℝ
  baseVolume : ℝ

/-- Calculates the scale factor between two lighthouses based on their base volumes -/
noncomputable def scaleFactor (original : Lighthouse) (model : Lighthouse) : ℝ :=
  (original.baseVolume / model.baseVolume) ^ (1/3)

/-- Theorem stating that a scaled model of the lighthouse will have the correct height -/
theorem scaled_lighthouse_height 
  (original : Lighthouse)
  (model : Lighthouse)
  (h1 : original.height = 60)
  (h2 : original.baseVolume = 18850)
  (h3 : model.baseVolume = 0.01885)
  : model.height * 100 = 60 := by
  sorry

/-- The main theorem proving the height of Ella's model lighthouse -/
theorem ellas_model_height : ∃ (original model : Lighthouse),
  original.height = 60 ∧
  original.baseVolume = 18850 ∧
  model.baseVolume = 0.01885 ∧
  model.height * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_lighthouse_height_ellas_model_height_l1114_111426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thousand_chime_date_l1114_111409

-- Define the clock chime pattern
def chimes_per_hour (hour : ℕ) : ℕ :=
  if hour % 12 = 0 then 14 else hour % 12 + 2

-- Define the total chimes in a day
def chimes_per_day : ℕ := 126

-- Define the starting date and time
def start_date : ℕ := 15  -- March 15
def start_hour : ℕ := 15  -- 3 PM
def start_minute : ℕ := 45

-- Define the target number of chimes
def target_chimes : ℕ := 1000

-- Theorem to prove
theorem thousand_chime_date :
  ∃ (days : ℕ),
    (92 : ℕ) + -- Chimes from 3:45 PM to midnight on start date
    days * chimes_per_day +
    (26 : ℕ) = target_chimes ∧  -- Chimes on the final day
    start_date + days + 1 = 23 -- March 23
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thousand_chime_date_l1114_111409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factor_l1114_111456

theorem polynomial_factor : ∃ q : Polynomial ℝ, (X^4 - 4*X^2 + 4 : Polynomial ℝ) = (X^2 - 2) * q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factor_l1114_111456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_arrangement_theorem_l1114_111439

/-- The number of students -/
def total_students : ℕ := 7

/-- The number of girls -/
def num_girls : ℕ := 3

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of ways to arrange students with all boys together and all girls together -/
def arrange_together : ℕ := 288

/-- The number of ways to arrange students alternating boys and girls -/
def arrange_alternate : ℕ := 144

/-- The number of ways to arrange students with a specific boy and girl in a certain order -/
def arrange_specific : ℕ := 2520

theorem student_arrangement_theorem :
  (total_students = num_girls + num_boys) ∧
  (arrange_together = 2 * Nat.factorial num_boys * Nat.factorial num_girls) ∧
  (arrange_alternate = Nat.factorial num_boys * Nat.factorial num_girls) ∧
  (arrange_specific = Nat.choose total_students 2 * Nat.factorial (total_students - 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_arrangement_theorem_l1114_111439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AE_l1114_111485

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The intersection point of two line segments -/
def intersectionPoint : Point :=
  { x := 3/2, y := 15/8 }

theorem length_of_AE : 
  let A : Point := { x := 0, y := 3 }
  let B : Point := { x := 4, y := 0 }
  let C : Point := { x := 3, y := 3 }
  let D : Point := { x := 1, y := 0 }
  let E : Point := intersectionPoint
  distance A E = 15 * Real.sqrt 13 / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AE_l1114_111485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_formula_l1114_111420

/-- The area of a triangle formed by three lines -/
noncomputable def triangleArea (a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) : ℝ :=
  let Δ := Matrix.det !![a₁, b₁, c₁; a₂, b₂, c₂; a₃, b₃, c₃]
  let C₁ := a₂ * b₃ - a₃ * b₂
  let C₂ := a₃ * b₁ - a₁ * b₃
  let C₃ := a₁ * b₂ - a₂ * b₁
  Δ^2 / (2 * abs (C₁ * C₂ * C₃))

/-- Theorem: The area of the triangle bounded by three lines is given by the formula -/
theorem triangle_area_formula (a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) :
  let line₁ : ℝ → ℝ → ℝ := λ x y => a₁ * x + b₁ * y + c₁
  let line₂ : ℝ → ℝ → ℝ := λ x y => a₂ * x + b₂ * y + c₂
  let line₃ : ℝ → ℝ → ℝ := λ x y => a₃ * x + b₃ * y + c₃
  triangleArea a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ = 
    (Matrix.det !![a₁, b₁, c₁; a₂, b₂, c₂; a₃, b₃, c₃])^2 / 
    (2 * abs ((a₂ * b₃ - a₃ * b₂) * (a₃ * b₁ - a₁ * b₃) * (a₁ * b₂ - a₂ * b₁))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_formula_l1114_111420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l1114_111428

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : 
  square_area = 1225 →
  rectangle_breadth = 10 →
  ((2 / 5) * Real.sqrt square_area) * rectangle_breadth = 140 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l1114_111428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_real_roots_m_range_l1114_111447

theorem equation_real_roots_m_range :
  ∀ m : ℝ, (∃ x : ℝ, 4^x + m * 2^x + m^2 - 1 = 0) →
  m ∈ Set.Ioc (-2 * Real.sqrt 3 / 3) 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_real_roots_m_range_l1114_111447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_property_l1114_111493

/-- A rational function with specific properties -/
noncomputable def f (x : ℝ) : ℝ := 
  let p (x : ℝ) := 2 * x  -- Linear function
  let q (x : ℝ) := (x + 3) * (x - 2)  -- Quadratic function with roots at -3 and 2
  p x / q x

/-- Theorem stating the properties of the rational function and its value at x = -1 -/
theorem rational_function_property :
  (f 0 = 0) ∧ 
  (f 3 = 1) ∧ 
  (f (-1) = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_property_l1114_111493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1114_111462

theorem tan_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (π / 2 + α) = -3 / 5) : Real.tan α = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1114_111462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamonds_10th_design_l1114_111497

/-- Represents the number of diamonds in the nth design of the geometric sequence. -/
def diamonds (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | n + 2 => diamonds (n + 1) + (n + 2)^2

/-- The theorem stating that the total number of diamonds in the 10th design is 385. -/
theorem diamonds_10th_design : diamonds 10 = 385 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamonds_10th_design_l1114_111497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_calculation_l1114_111442

noncomputable section

/-- The probability that a randomly selected point from a square with vertices at (±2, ±2) 
    is within one unit of the origin -/
def probability_within_unit_circle : ℝ := Real.pi / 16

/-- The side length of the square -/
def square_side : ℝ := 4

/-- The area of the square -/
def square_area : ℝ := square_side ^ 2

/-- The radius of the circle -/
def circle_radius : ℝ := 1

/-- The area of the circle -/
def circle_area : ℝ := Real.pi * circle_radius ^ 2

theorem probability_calculation : 
  probability_within_unit_circle = circle_area / square_area :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_calculation_l1114_111442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_exterior_angles_convex_ngon_l1114_111403

/-- The sum of the exterior angles of a convex n-gon is 360 degrees. -/
theorem sum_exterior_angles_convex_ngon (n : ℕ) (h : n ≥ 3) :
  (n : ℝ) * 360 / n = 360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_exterior_angles_convex_ngon_l1114_111403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_ratio_minimizes_cost_l1114_111481

/-- Represents a cylindrical boiler -/
structure CylindricalBoiler where
  volume : ℝ
  base_cost : ℝ
  lateral_cost : ℝ

/-- The optimal ratio of diameter to height for minimum construction cost -/
noncomputable def optimal_ratio (boiler : CylindricalBoiler) : ℝ :=
  boiler.lateral_cost / boiler.base_cost

/-- Theorem: The optimal ratio of diameter to height that minimizes 
    the construction cost of a cylindrical boiler is the ratio of 
    lateral surface cost to base cost -/
theorem optimal_ratio_minimizes_cost (boiler : CylindricalBoiler) 
    (h_volume : boiler.volume > 0)
    (h_base_cost : boiler.base_cost > 0)
    (h_lateral_cost : boiler.lateral_cost > 0) :
    ∀ (d h : ℝ), d > 0 → h > 0 → 
    π * (d / 2)^2 * h = boiler.volume →
    (boiler.base_cost * π * d^2 + boiler.lateral_cost * π * d * h) ≥ 
    (boiler.base_cost * π * (optimal_ratio boiler * h)^2 + 
     boiler.lateral_cost * π * (optimal_ratio boiler * h) * h) :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_ratio_minimizes_cost_l1114_111481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1114_111459

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - Real.sqrt 3 * Real.cos (2 * x)

theorem f_properties :
  ∃ (period : ℝ) (max_value : ℝ) (max_set : Set ℝ),
    (∀ (t : ℝ), t > 0 ∧ (∀ (x : ℝ), f (x + t) = f x) → period ≤ t) ∧
    period = Real.pi ∧
    (∀ (x : ℝ), f x ≤ max_value) ∧
    max_value = 2 ∧
    max_set = {x | ∃ (k : ℤ), x = k * Real.pi + 5 * Real.pi / 12} ∧
    (∀ (x : ℝ), x ∈ max_set ↔ f x = max_value) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1114_111459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_triangle_area_value_l1114_111408

open Real

/-- Triangle ABC with given properties -/
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
  h4 : b * sin A = Real.sqrt 3 * a * cos B
  h5 : b = 3
  h6 : a = 2

/-- The measure of angle B is π/3 -/
theorem angle_B_measure (t : Triangle) : t.B = π / 3 := by sorry

/-- The area of triangle ABC -/
noncomputable def triangle_area (t : Triangle) : ℝ := 1 / 2 * t.a * t.c * sin t.B

/-- The area of triangle ABC is (√3 + 3√2) / 2 -/
theorem triangle_area_value (t : Triangle) : 
  triangle_area t = (Real.sqrt 3 + 3 * Real.sqrt 2) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_triangle_area_value_l1114_111408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_M_less_than_threshold_l1114_111482

def Q (M : ℕ) : ℚ :=
  (Int.floor (M / 3 : ℚ) + Int.ceil ((2 * M : ℕ) / 3 : ℚ)) / (M + 1 : ℕ)

theorem first_M_less_than_threshold : ∀ k : ℕ, 
  k > 0 ∧ 6 ∣ k ∧ k < 390 → Q k ≥ 320/450 ∧ 
  Q 390 < 320/450 ∧ 
  390 % 6 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_M_less_than_threshold_l1114_111482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_product_l1114_111406

theorem fraction_sum_product (a b c d : ℕ+) : 
  (1 : ℚ) / 5 < a / b ∧ a / b < (1 : ℚ) / 4 ∧
  (1 : ℚ) / 5 < c / d ∧ c / d < (1 : ℚ) / 4 ∧
  b ≤ 19 ∧ d ≤ 19 ∧
  Nat.Coprime a.val b.val ∧ Nat.Coprime c.val d.val ∧
  ∀ (x y : ℕ+), (1 : ℚ) / 5 < x / y ∧ x / y < (1 : ℚ) / 4 ∧ y ≤ 19 ∧ Nat.Coprime x.val y.val →
    (a + b ≥ x + y ∧ c + d ≤ x + y) →
  (a + b) * (c + d) = 253 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_product_l1114_111406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_min_surface_area_l1114_111425

/-- Represents a cylinder with given radius and height -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ :=
  Real.pi * c.radius^2 * c.height

/-- Calculates the surface area of a cylinder (including base) -/
noncomputable def cylinderSurfaceArea (c : Cylinder) : ℝ :=
  2 * Real.pi * c.radius * c.height + Real.pi * c.radius^2

/-- Theorem: For a cylinder with volume 27π, the surface area is minimized when the radius is 3 -/
theorem cylinder_min_surface_area (c : Cylinder) :
  cylinderVolume c = 27 * Real.pi →
  ∀ c' : Cylinder, cylinderVolume c' = 27 * Real.pi →
    cylinderSurfaceArea c' ≥ cylinderSurfaceArea { radius := 3, height := 3 } :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_min_surface_area_l1114_111425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_200_by_25_percent_l1114_111452

noncomputable def increase_by_percentage (initial : ℝ) (percentage : ℝ) : ℝ :=
  initial * (1 + percentage / 100)

theorem increase_200_by_25_percent :
  increase_by_percentage 200 25 = 250 := by
  -- Unfold the definition of increase_by_percentage
  unfold increase_by_percentage
  -- Simplify the expression
  simp [mul_add, mul_div_assoc, mul_comm]
  -- Perform the numerical calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_200_by_25_percent_l1114_111452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_theorem_l1114_111443

-- Define set A
def A : Set ℝ := {x | (2 : ℝ)^(4*x+6) ≥ (64 : ℝ)^x}

-- Define set B
def B : Set ℝ := {x | 2*x^2 + x - 15 ≤ 0}

-- Define set C
def C (k : ℝ) : Set ℝ := {x | -2 ≤ x - k ∧ x - k ≤ 1/2}

theorem sets_theorem :
  (A = {x | x ≤ 3}) ∧
  (B = {x | -3 ≤ x ∧ x ≤ 5/2}) ∧
  ((Aᶜ ∪ B) = {x | -3 ≤ x ∧ x ≤ 5/2 ∨ x > 3}) ∧
  (∀ k, (C k ⊆ B) ↔ -1 ≤ k ∧ k ≤ 2) :=
by
  sorry

#check sets_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_theorem_l1114_111443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_box_cost_l1114_111413

/-- Calculates the minimum amount to spend on boxes for packaging a collection --/
theorem min_box_cost (collection_volume box_length box_width box_height box_cost : ℝ) : 
  collection_volume = 3060000 →
  box_length = 20 →
  box_width = 20 →
  box_height = 15 →
  box_cost = 1.2 →
  ⌈collection_volume / (box_length * box_width * box_height)⌉ * box_cost = 612 := by
  sorry

#check min_box_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_box_cost_l1114_111413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_theorem_l1114_111414

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  p : Point
  q : Point
  r : Point
  s : Point

/-- Represents the folding of the rectangle -/
def fold (rect : Rectangle) (t : Point) (u : Point) : Rectangle :=
  sorry

/-- The area of a rectangle -/
def area (rect : Rectangle) : ℝ :=
  sorry

/-- Check if a number is not divisible by the square of any prime -/
def notDivisibleBySquareOfPrime (n : ℕ) : Prop :=
  sorry

theorem rectangle_area_theorem (pqrs : Rectangle) (t u : Point) (r' s' : Point) :
  t.x - pqrs.p.x > pqrs.s.x - u.x →  -- QT > PU
  fold pqrs t u = Rectangle.mk pqrs.p pqrs.q r' s' →
  ((r'.x - pqrs.p.x) ^ 2 + (r'.y - pqrs.p.y) ^ 2).sqrt = 7 →
  ((t.x - pqrs.q.x) ^ 2 + (t.y - pqrs.q.y) ^ 2).sqrt = 29 →
  ∃ (x y z : ℕ), 
    area pqrs = x + y * Real.sqrt z ∧
    notDivisibleBySquareOfPrime z ∧
    x + y + z = 601 ∧
    area pqrs = 182 * Real.sqrt 13 + 406 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_theorem_l1114_111414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_l1114_111468

theorem acute_triangle (α β γ : Real) 
  (h1 : Real.sin α > Real.cos β) 
  (h2 : Real.sin β > Real.cos γ) 
  (h3 : Real.sin γ > Real.cos α) 
  (h4 : α + β + γ = Real.pi) : 
  α < Real.pi/2 ∧ β < Real.pi/2 ∧ γ < Real.pi/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_l1114_111468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_non_congruent_triangles_l1114_111460

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Predicate for triangles with perimeter 12 -/
def perimeter_12 (t : IntTriangle) : Prop := t.a + t.b + t.c = 12

/-- Predicate for non-congruent triangles -/
def non_congruent (t1 t2 : IntTriangle) : Prop :=
  ¬(t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∧
  ¬(t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∧
  ¬(t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

/-- The main theorem stating there are exactly 5 non-congruent triangles -/
theorem five_non_congruent_triangles :
  ∃ (s : Finset IntTriangle),
    (∀ t ∈ s, perimeter_12 t) ∧
    (∀ t1 ∈ s, ∀ t2 ∈ s, t1 ≠ t2 → non_congruent t1 t2) ∧
    s.card = 5 ∧
    (∀ t : IntTriangle, perimeter_12 t → ∃ t' ∈ s, ¬(non_congruent t t')) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_non_congruent_triangles_l1114_111460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_arrangement_probability_l1114_111412

def num_red_lamps : ℕ := 4
def num_blue_lamps : ℕ := 4
def total_lamps : ℕ := num_red_lamps + num_blue_lamps
def num_lamps_on : ℕ := 5

def probability_specific_arrangement : ℚ :=
  9 / 1960

theorem specific_arrangement_probability :
  probability_specific_arrangement =
    (Nat.choose (num_blue_lamps - 1) 2 * Nat.choose (num_red_lamps - 1) 2 : ℚ) /
    (Nat.choose total_lamps num_red_lamps * Nat.choose total_lamps num_lamps_on : ℚ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_arrangement_probability_l1114_111412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_7_equals_2_l1114_111480

/-- Given a real number x such that x + 1/x = 2, Sₘ is defined as xᵐ + 1/xᵐ -/
noncomputable def S (x : ℝ) (m : ℕ) : ℝ := x^m + 1/(x^m)

/-- Theorem: For a real number x satisfying x + 1/x = 2, S₇ = 2 -/
theorem S_7_equals_2 (x : ℝ) (h : x + 1/x = 2) : S x 7 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_7_equals_2_l1114_111480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_figure_l1114_111484

/-- The area of the figure bounded by r = 6 sin(3φ) and r = 3 (for r ≥ 3) -/
noncomputable def bounded_area : ℝ := 3 * Real.pi + (9 * Real.sqrt 3) / 2

/-- The polar equation r = 6 sin(3φ) -/
noncomputable def curve_equation (φ : ℝ) : ℝ := 6 * Real.sin (3 * φ)

/-- The constant boundary r = 3 -/
def boundary_equation : ℝ := 3

/-- Theorem stating the existence of an area function satisfying the given conditions -/
theorem area_of_bounded_figure :
  ∃ (a : ℝ → ℝ → ℝ), 
    (∀ φ r, r = curve_equation φ ∨ r = boundary_equation → r ≥ boundary_equation → 
      a φ r = bounded_area) ∧
    (∀ φ r, r < curve_equation φ ∧ r < boundary_equation → a φ r = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_figure_l1114_111484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_midpoint_relation_l1114_111402

/-- A tetrahedron with edge lengths and midpoint distances -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  m₁ : ℝ
  m₂ : ℝ
  m₃ : ℝ
  m₄ : ℝ
  m₅ : ℝ
  m₆ : ℝ

/-- The sum of squares of edge lengths equals four times the sum of squares of midpoint distances -/
theorem tetrahedron_edge_midpoint_relation (t : Tetrahedron) :
  t.a^2 + t.b^2 + t.c^2 + t.d^2 + t.e^2 + t.f^2 =
  4 * (t.m₁^2 + t.m₂^2 + t.m₃^2 + t.m₄^2 + t.m₅^2 + t.m₆^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_midpoint_relation_l1114_111402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_two_monochromatic_triangles_no_two_monochromatic_triangles_for_n_less_than_8_l1114_111470

/-- A point on a plane with a color --/
structure ColoredPoint where
  x : ℝ
  y : ℝ
  color : Bool -- True for red, False for blue

/-- The set of all triangles formed by the given points --/
def TriangleSet (points : Finset ColoredPoint) : Finset (Finset ColoredPoint) :=
  sorry

/-- Predicate to check if three points are collinear --/
def AreCollinear (p1 p2 p3 : ColoredPoint) : Prop :=
  sorry

/-- Predicate to check if a triangle is monochromatic --/
def Monochromatic (triangle : Finset ColoredPoint) : Prop :=
  sorry

/-- The number of triangles that have a given line segment as an edge --/
def TrianglesWithEdge (points : Finset ColoredPoint) (p1 p2 : ColoredPoint) : ℕ :=
  sorry

theorem smallest_n_for_two_monochromatic_triangles :
  ∀ n : ℕ,
  n ≥ 8 →
  ∀ points : Finset ColoredPoint,
  points.card = n →
  (∀ p1 p2 p3, p1 ∈ points → p2 ∈ points → p3 ∈ points → p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬AreCollinear p1 p2 p3) →
  (∀ p1 p2 p3 p4, p1 ∈ points → p2 ∈ points → p3 ∈ points → p4 ∈ points → p1 ≠ p2 → p3 ≠ p4 → TrianglesWithEdge points p1 p2 = TrianglesWithEdge points p3 p4) →
  ∃ t1 t2 : Finset ColoredPoint,
    t1 ∈ TriangleSet points ∧
    t2 ∈ TriangleSet points ∧
    t1 ≠ t2 ∧
    Monochromatic t1 ∧
    Monochromatic t2 :=
  sorry

theorem no_two_monochromatic_triangles_for_n_less_than_8 :
  ∀ n : ℕ,
  n < 8 →
  ∃ points : Finset ColoredPoint,
  points.card = n ∧
  (∀ p1 p2 p3, p1 ∈ points → p2 ∈ points → p3 ∈ points → p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬AreCollinear p1 p2 p3) ∧
  (∀ p1 p2 p3 p4, p1 ∈ points → p2 ∈ points → p3 ∈ points → p4 ∈ points → p1 ≠ p2 → p3 ≠ p4 → TrianglesWithEdge points p1 p2 = TrianglesWithEdge points p3 p4) ∧
  ¬∃ t1 t2 : Finset ColoredPoint,
    t1 ∈ TriangleSet points ∧
    t2 ∈ TriangleSet points ∧
    t1 ≠ t2 ∧
    Monochromatic t1 ∧
    Monochromatic t2 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_two_monochromatic_triangles_no_two_monochromatic_triangles_for_n_less_than_8_l1114_111470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1114_111489

/-- Given a hyperbola C: x²/a² - y²/b² = 1 (a > 0, b > 0) whose asymptotes are tangent to the circle (x-2)² + y² = 1, 
    the equation of the asymptotes of C is y = ±(√3/3)x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ k : ℝ, k = Real.sqrt 3 / 3 ∧ 
    ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1 ∧ (y = k*x ∨ y = -k*x)) → 
      ((x - 2)^2 + y^2 = 1) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1114_111489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l1114_111427

-- Define the distances and times for the two parts of the journey
noncomputable def distance1 : ℝ := 125
noncomputable def time1 : ℝ := 2.5
noncomputable def distance2 : ℝ := 270
noncomputable def time2 : ℝ := 3

-- Define the total distance and total time
noncomputable def totalDistance : ℝ := distance1 + distance2
noncomputable def totalTime : ℝ := time1 + time2

-- Define the average speed
noncomputable def averageSpeed : ℝ := totalDistance / totalTime

-- Theorem statement
theorem train_average_speed :
  |averageSpeed - 71.82| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l1114_111427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_determine_longer_wire_l1114_111441

-- Define the length of the wires
variable (L : ℝ)

-- Define the remaining lengths of the wires after cutting
noncomputable def remaining_length1 (L : ℝ) : ℝ := L * (3/4)
noncomputable def remaining_length2 (L : ℝ) : ℝ := L - (1/4)

-- Theorem stating that we cannot determine which remaining wire is longer
theorem cannot_determine_longer_wire (h : L > 0) :
  ¬(∀ L, remaining_length1 L > remaining_length2 L) ∧
  ¬(∀ L, remaining_length1 L < remaining_length2 L) ∧
  ¬(∀ L, remaining_length1 L = remaining_length2 L) := by
  sorry

-- Examples demonstrating different scenarios
example (h : L = 1) : remaining_length1 L = remaining_length2 L := by
  simp [remaining_length1, remaining_length2, h]
  norm_num

example (h : L > 1) : remaining_length1 L < remaining_length2 L := by
  simp [remaining_length1, remaining_length2]
  linarith

example (h : 0 < L ∧ L < 1) : remaining_length1 L > remaining_length2 L := by
  simp [remaining_length1, remaining_length2]
  linarith

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_determine_longer_wire_l1114_111441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_solutions_l1114_111476

/-- The differential equation F(x, y, dy/dx) = (dy/dx)^2 - (6x + y)(dy/dx) + 6xy = 0 -/
def F (x y dydx : ℝ) : ℝ := dydx^2 - (6*x + y)*dydx + 6*x*y

/-- Special solution y₁ = C₁ e^x -/
noncomputable def y₁ (x C₁ : ℝ) : ℝ := C₁ * Real.exp x

/-- Special solution y₂ = 3x^2 + C₂ -/
def y₂ (x C₂ : ℝ) : ℝ := 3 * x^2 + C₂

/-- Theorem stating that y₁ and y₂ are special solutions of the differential equation -/
theorem special_solutions (x C₁ C₂ : ℝ) :
  (∃ (dydx : ℝ), F x (y₁ x C₁) dydx = 0) ∧
  (∃ (dydx : ℝ), F x (y₂ x C₂) dydx = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_solutions_l1114_111476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1114_111469

noncomputable def f (x : ℝ) := Real.cos (2 * x) + 2 * Real.sin x * Real.sin (Real.pi / 4)

theorem f_properties :
  (∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x ∧ 
    ∀ q : ℝ, q > 0 ∧ (∀ x : ℝ, f (x + q) = f x) → p ≤ q) ∧
  (∃ M : ℝ, ∀ x : ℝ, f x ≤ M ∧ ∃ x : ℝ, f x = M) ∧
  (∃ S : Set ℝ, ∀ x : ℝ, x ∈ S ↔ f x = 2) ∧
  (∀ A : ℝ, 0 < A ∧ A < Real.pi / 2 → 
    f A = 0 → 
    ∃ c : ℝ, c > 0 ∧ 
    7^2 = 5^2 + c^2 - 2 * 5 * c * Real.cos A ∧
    5 * c * Real.sin A / 2 = 10) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1114_111469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_identity_l1114_111465

theorem binomial_sum_identity (n m : ℕ) :
  (Finset.range (n - m + 1)).sum (fun k => (n + 1 - m - k) * Nat.choose (n - k) m) = Nat.choose (n + 2) (m + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_identity_l1114_111465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_range_l1114_111419

noncomputable def f (x φ : ℝ) : ℝ := -2 * Real.sin (2 * x + φ)

theorem phi_range (φ : ℝ) :
  (|φ| < π) →
  (∀ x₁ x₂, π/5 < x₁ ∧ x₁ < x₂ ∧ x₂ < 5*π/8 → f x₁ φ < f x₂ φ) →
  φ ∈ Set.Icc (π/10) (π/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_range_l1114_111419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1114_111475

noncomputable def f (x : ℝ) : ℝ := x / (1 + x^2)

theorem f_properties :
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f (-x) = -f x) ∧
  f (1/3) = 3/10 ∧
  (∀ x y, x ∈ Set.Icc (-1 : ℝ) 1 → y ∈ Set.Icc (-1 : ℝ) 1 → x < y → f x < f y) ∧
  (∀ m : ℝ, (∃ x, x ∈ Set.Icc (1/2 : ℝ) 1 ∧ f (m*x - x) + f (x^2 - 1) > 0) ↔ 1 < m ∧ m ≤ 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1114_111475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_mixture_l1114_111478

/-- Represents a salt solution with its concentration and cost per kg -/
structure SaltSolution where
  concentration : ℝ
  cost_per_kg : ℝ

/-- Represents a mixture of two salt solutions -/
structure Mixture where
  solution1 : SaltSolution
  amount1 : ℝ
  solution2 : SaltSolution
  amount2 : ℝ

/-- Calculate the total weight of a mixture -/
noncomputable def total_weight (m : Mixture) : ℝ :=
  m.amount1 + m.amount2

/-- Calculate the salt concentration of a mixture -/
noncomputable def mixture_concentration (m : Mixture) : ℝ :=
  (m.solution1.concentration * m.amount1 + m.solution2.concentration * m.amount2) / (m.amount1 + m.amount2)

/-- Calculate the total cost of a mixture -/
noncomputable def total_cost (m : Mixture) : ℝ :=
  m.solution1.cost_per_kg * m.amount1 + m.solution2.cost_per_kg * m.amount2

/-- The optimal mixture theorem -/
theorem optimal_mixture 
  (solution1 : SaltSolution)
  (solution2 : SaltSolution)
  (total_amount : ℝ)
  (target_concentration : ℝ)
  (h1 : solution1.concentration = 0.1)
  (h2 : solution2.concentration = 0.3)
  (h3 : solution1.cost_per_kg = 3)
  (h4 : solution2.cost_per_kg = 5)
  (h5 : total_amount = 400)
  (h6 : target_concentration = 0.25) :
  ∃ (m : Mixture),
    m.solution1 = solution1 ∧
    m.solution2 = solution2 ∧
    m.amount1 = 100 ∧
    m.amount2 = 300 ∧
    total_weight m = total_amount ∧
    mixture_concentration m = target_concentration ∧
    (∀ (m' : Mixture),
      m'.solution1 = solution1 →
      m'.solution2 = solution2 →
      total_weight m' = total_amount →
      mixture_concentration m' = target_concentration →
      total_cost m ≤ total_cost m') :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_mixture_l1114_111478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_remediation_proof_l1114_111471

/-- The daily remediation rate of Team A in meters per day -/
noncomputable def team_a_rate : ℝ := 900

/-- The daily remediation rate of Team B in meters per day -/
noncomputable def team_b_rate : ℝ := 1500 - team_a_rate

/-- The total daily remediation rate of both teams in meters per day -/
noncomputable def total_rate : ℝ := 1500

/-- The time it takes for Team A to remediate 3600 meters in days -/
noncomputable def team_a_time : ℝ := 3600 / team_a_rate

/-- The time it takes for Team B to remediate 2400 meters in days -/
noncomputable def team_b_time : ℝ := 2400 / team_b_rate

theorem river_remediation_proof :
  team_a_rate = 900 ∧
  team_a_rate + team_b_rate = total_rate ∧
  team_a_time = team_b_time :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_remediation_proof_l1114_111471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_angle_l1114_111494

noncomputable def a (α : Real) : Fin 2 → Real := ![3/2, Real.sin α]
noncomputable def b (α : Real) : Fin 2 → Real := ![Real.cos α, 1/3]

theorem parallel_vectors_angle (α : Real) 
  (h1 : 0 < α ∧ α < π/2)  -- α is acute
  (h2 : ∃ k : Real, a α = k • b α)  -- a is parallel to b
  : α = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_angle_l1114_111494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_totient_bound_l1114_111474

/-- Euler's totient function -/
def phi : ℕ → ℕ := sorry

/-- Iterated application of phi function k times -/
def iterated_phi : ℕ → ℕ → ℕ
  | 0, n => n
  | k + 1, n => iterated_phi k (phi n)

theorem euler_totient_bound (n k : ℕ) (hn : n > 0) (hk : k > 0) 
  (h_phi : phi 1 = 1) (h_iterated : iterated_phi k n = 1) : 
  n ≤ 3^k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_totient_bound_l1114_111474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1114_111455

-- Define the triangle
structure Triangle :=
  (A B C : ℝ)  -- angles
  (a b c : ℝ)  -- sides

-- Define properties of the triangle
def is_acute (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2

def sine_product (t : Triangle) : Prop :=
  Real.sin t.A * Real.sin t.C = 3/4

def side_product (t : Triangle) : Prop :=
  t.b^2 = t.a * t.c

-- Main theorem
theorem triangle_properties (t : Triangle) 
  (h_acute : is_acute t)
  (h_sine : sine_product t)
  (h_side : side_product t) :
  t.B = Real.pi/3 ∧ 
  (t.b = Real.sqrt 3 → t.a + t.b + t.c = 3 * Real.sqrt 3) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1114_111455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_count_l1114_111491

mutual
  def a : ℕ → ℕ
    | 0 => 0
    | 1 => 0
    | 2 => 1
    | n + 3 => a (n + 1) + b (n + 1)

  def b : ℕ → ℕ
    | 0 => 0
    | 1 => 1
    | 2 => 0
    | n + 3 => a (n + 2) + b (n + 1)
end

theorem valid_sequences_count : a 20 + b 20 = 1874 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_count_l1114_111491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1114_111436

def sequenceF : ℕ → ℕ
| 0 => 2
| 1 => 4
| n + 2 => 
  let prev := sequenceF (n + 1)
  let prev_prev := sequenceF n
  let sum := (prev % 10) + (prev_prev % 10)
  if sum < 10 then sum else (sum / 10 + sum % 10)

theorem sequence_properties :
  (∀ n : ℕ, sequenceF n ≠ 9) ∧
  (sequenceF 99 = 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1114_111436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1114_111464

noncomputable def f (x : ℝ) : ℝ := 2 / Real.sqrt (3 - 2*x - x^2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -3 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1114_111464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_consecutive_fib_sum_divisible_by_12_l1114_111450

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

def sum_consecutive_fib (start : ℕ) (length : ℕ) : ℕ :=
  (List.range length).map (λ i => fib (start + i)) |>.sum

def is_divisible_by_12 (n : ℕ) : Prop :=
  n % 12 = 0

theorem smallest_consecutive_fib_sum_divisible_by_12 :
  (∀ start, is_divisible_by_12 (sum_consecutive_fib start 24)) ∧
  (∀ N < 24, ∃ start, ¬is_divisible_by_12 (sum_consecutive_fib start N)) :=
sorry

#check smallest_consecutive_fib_sum_divisible_by_12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_consecutive_fib_sum_divisible_by_12_l1114_111450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coincides_with_origin_lies_on_line_lies_in_first_or_third_quadrant_l1114_111479

/-- Complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 5*m + 6) (m^2 - 3*m + 2)

/-- Condition 1: z coincides with the origin -/
theorem coincides_with_origin (m : ℝ) :
  z m = 0 ↔ m = 2 := by sorry

/-- Condition 2: z lies on the line y = 2x -/
theorem lies_on_line (m : ℝ) :
  (z m).im = 2 * (z m).re ↔ m = 2 ∨ m = 5 := by sorry

/-- Condition 3: z lies in the first or third quadrant -/
theorem lies_in_first_or_third_quadrant (m : ℝ) :
  (z m).re * (z m).im > 0 ↔ m < 1 ∨ m > 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coincides_with_origin_lies_on_line_lies_in_first_or_third_quadrant_l1114_111479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_camping_cost_difference_l1114_111496

noncomputable section

-- Define the amounts paid by each person
def alice_paid : ℝ := 130
def bob_paid : ℝ := 150
def carlos_paid : ℝ := 200

-- Define the total cost
def total_cost : ℝ := alice_paid + bob_paid + carlos_paid

-- Define the equal share each person should pay
noncomputable def equal_share : ℝ := total_cost / 3

-- Define the amounts Alice and Bob gave to Carlos
noncomputable def a : ℝ := equal_share - alice_paid
noncomputable def b : ℝ := equal_share - bob_paid

-- Theorem to prove
theorem camping_cost_difference : a - b = 20 := by
  -- Expand definitions
  unfold a b equal_share total_cost alice_paid bob_paid carlos_paid
  -- Perform algebraic simplification
  simp [sub_eq_add_neg, add_comm, add_left_comm, add_assoc]
  -- The proof is completed by normalization
  norm_num

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_camping_cost_difference_l1114_111496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_equivalence_l1114_111423

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x)

-- Define the shifted function
noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin (x + Real.pi / 4)

-- Theorem stating that g is equivalent to f shifted left by π/8
theorem shift_equivalence : ∀ x : ℝ, g x = f (x + Real.pi / 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_equivalence_l1114_111423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_polynomial_division_l1114_111415

theorem remainder_of_polynomial_division (X : Type) [CommRing X] : 
  let x : Polynomial X := Polynomial.X
  let f := x^2021 + 1
  let g := x^8 - x^6 + x^4 - x^2 + 1
  (x^2 + 1) * g = x^10 + 1 →
  ∃ q : Polynomial X, f = q * g + (x - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_polynomial_division_l1114_111415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_midpoints_on_circle_l1114_111435

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- We'll leave this empty for now, as the specific implementation isn't crucial for the theorem
  mk :: -- Add constructor

/-- The set of midpoints of sides and diagonals of a regular polygon -/
def midpoints (n : ℕ) (p : RegularPolygon n) : Set (Real × Real) :=
  sorry -- Implementation details omitted

/-- A circle in 2D Euclidean space -/
structure Circle where
  center : Real × Real
  radius : Real

/-- The number of points from a set that lie on a circle -/
def pointsOnCircle (s : Set (Real × Real)) (c : Circle) : ℕ :=
  sorry -- Implementation details omitted

/-- The main theorem -/
theorem max_midpoints_on_circle :
  ∀ (p : RegularPolygon 1976) (c : Circle),
    pointsOnCircle (midpoints 1976 p) c ≤ 1976 ∧
    ∃ (c' : Circle), pointsOnCircle (midpoints 1976 p) c' = 1976 := by
  sorry -- Proof omitted

#check max_midpoints_on_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_midpoints_on_circle_l1114_111435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_present_value_l1114_111453

/-- Represents the value depletion rate per annum as a real number between 0 and 1. -/
noncomputable def depletionRate : ℝ := 0.10

/-- Represents the number of years passed. -/
noncomputable def years : ℝ := 2

/-- Represents the value of the machine after 2 years in dollars. -/
noncomputable def valueAfterTwoYears : ℝ := 729

/-- Calculates the present value of the machine given its value after a certain number of years
    and the annual depletion rate. -/
noncomputable def calculatePresentValue (futureValue : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  futureValue / ((1 - rate) ^ time)

theorem machine_present_value :
  calculatePresentValue valueAfterTwoYears depletionRate years = 900 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_present_value_l1114_111453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_height_greater_than_103_3_l1114_111404

noncomputable def tower_height (distance : ℝ) (angle : ℝ) : ℝ := distance * Real.tan angle

theorem tower_height_greater_than_103_3 (distance : ℝ) (angle : ℝ) 
  (h1 : distance = 100)
  (h2 : angle = 46 * π / 180) :
  tower_height distance angle > 103.3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_height_greater_than_103_3_l1114_111404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f1_is_odd_f2_is_odd_and_even_f3_is_odd_l1114_111461

-- Function 1
noncomputable def f1 (x : ℝ) : ℝ := x^3 - 1/x

theorem f1_is_odd : ∀ x ≠ 0, f1 (-x) = -f1 x := by sorry

-- Function 2
noncomputable def f2 (x : ℝ) : ℝ := Real.sqrt (x^2 - 1) + Real.sqrt (1 - x^2)

theorem f2_is_odd_and_even : 
  (∀ x ∈ ({-1, 1} : Set ℝ), f2 (-x) = -f2 x) ∧ 
  (∀ x ∈ ({-1, 1} : Set ℝ), f2 (-x) = f2 x) := by sorry

-- Function 3
noncomputable def f3 (x : ℝ) : ℝ := 
  if x > 0 then x^2 + 2
  else if x = 0 then 0
  else -x^2 - 2

theorem f3_is_odd : ∀ x, f3 (-x) = -f3 x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f1_is_odd_f2_is_odd_and_even_f3_is_odd_l1114_111461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_equal_distribution_l1114_111440

/-- Represents the distribution of items to a friend --/
structure Distribution where
  small_blocks : Nat
  medium_blocks : Nat
  large_blocks : Nat
  cards : Nat
  figurines : Nat

/-- Represents Laura's items and constraints --/
def laura_items : Nat × Nat × Nat × Nat × Nat :=
  (12, 10, 6, 8, 4)

/-- The number of friends --/
def num_friends : Nat := 4

/-- Checks if a distribution satisfies the constraints --/
def valid_distribution (d : Distribution) : Prop :=
  d.small_blocks ≤ 3 ∧ d.figurines ≤ 2

/-- Checks if a list of distributions is equal --/
def equal_distributions (ds : List Distribution) : Prop :=
  ∀ d₁ d₂, d₁ ∈ ds → d₂ ∈ ds → d₁ = d₂

/-- Theorem stating the impossibility of equal distribution --/
theorem impossibility_of_equal_distribution :
  ¬∃ (ds : List Distribution),
    ds.length = num_friends ∧
    (∀ d, d ∈ ds → valid_distribution d) ∧
    equal_distributions ds ∧
    (ds.map (λ d => d.small_blocks)).sum = laura_items.1 ∧
    (ds.map (λ d => d.medium_blocks)).sum = laura_items.2.1 ∧
    (ds.map (λ d => d.large_blocks)).sum = laura_items.2.2.1 ∧
    (ds.map (λ d => d.cards)).sum = laura_items.2.2.2.1 ∧
    (ds.map (λ d => d.figurines)).sum = laura_items.2.2.2.2 :=
by sorry

#check impossibility_of_equal_distribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_equal_distribution_l1114_111440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1114_111422

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def f (x : ℝ) : ℝ := x - (floor x : ℝ)

theorem f_properties :
  (∀ x : ℝ, f (x + 1) = f x) ∧
  (∃ S : Set ℝ, (Set.Infinite S) ∧ (∀ x ∈ S, f x = 1/2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1114_111422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l1114_111411

theorem intersection_point_sum (a b : ℝ) : 
  (b = a - 1) ∧ (b = 3 / a) → (1 / a + 1 / b = Real.sqrt 13 / 3 ∨ 1 / a + 1 / b = -Real.sqrt 13 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l1114_111411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_equal_distances_l1114_111498

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance from a point to a line -/
noncomputable def distancePointToLine (x y : ℝ) (l : Line) : ℝ :=
  (abs (l.a * x + l.b * y + l.c)) / Real.sqrt (l.a^2 + l.b^2)

/-- Check if a point is on a line -/
def isPointOnLine (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem line_through_point_with_equal_distances :
  ∃ l : Line,
    (isPointOnLine 1 2 l) ∧
    (distancePointToLine 2 3 l = distancePointToLine 4 (-5) l) ∧
    ((l.a = 3 ∧ l.b = 2 ∧ l.c = -7) ∨ (l.a = 4 ∧ l.b = 1 ∧ l.c = -6)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_equal_distances_l1114_111498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1114_111495

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * Real.sin x + 4

theorem problem_solution (a b : ℝ) :
  (f a b (Real.log (Real.log 10 / Real.log 2)) = 5) →
  (Real.log (Real.log 10 / Real.log 2) + Real.log (Real.log 2 / Real.log 2) = 0) →
  (f a b (Real.log (Real.log 2 / Real.log 2)) = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1114_111495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1114_111457

/-- The function f(t) = (t^2 + 0.5t) / (t^2 + 1) -/
noncomputable def f (t : ℝ) : ℝ := (t^2 + 0.5*t) / (t^2 + 1)

/-- The range of f is [-1/4, 1/4] -/
theorem range_of_f :
  ∀ y : ℝ, (∃ t : ℝ, f t = y) ↔ -1/4 ≤ y ∧ y ≤ 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1114_111457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_zero_l1114_111445

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (Real.exp (3 * x) / (1 + Real.exp (3 * x))))

theorem f_derivative_at_zero : 
  deriv f 0 = 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_zero_l1114_111445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_equation_solution_l1114_111437

theorem square_root_equation_solution :
  ∀ x : ℝ, x + 2 ≥ 0 → x - 1 ≥ 0 → 4 * x - 7 ≥ 0 →
  (∃ s₁ s₂ s₃ : Bool, (if s₁ then 1 else -1) * Real.sqrt (x + 2) + 
                      (if s₂ then 1 else -1) * Real.sqrt (x - 1) = 
                      (if s₃ then 1 else -1) * Real.sqrt (4 * x - 7)) →
  x = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_equation_solution_l1114_111437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l1114_111438

theorem expression_equality : Real.sqrt (9/16) - (-9.9)^(0 : ℝ) + 8^(-2/3 : ℝ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l1114_111438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_double_capital_l1114_111472

/-- Time for capital to double with compound interest -/
theorem time_to_double_capital (P r : ℝ) (n : ℕ+) (h_P : P > 0) (h_r : r > 0) :
  let t := Real.log 2 / (↑n * Real.log (1 + r / ↑n))
  (2 : ℝ) * P = P * (1 + r / ↑n) ^ (n * t) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_double_capital_l1114_111472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_gain_percentage_approx_l1114_111466

noncomputable section

-- Define the selling prices and profits for each item
def selling_price1 : ℝ := 195
def profit1 : ℝ := 45
def selling_price2 : ℝ := 330
def profit2 : ℝ := 80
def selling_price3 : ℝ := 120
def profit3 : ℝ := 30

-- Calculate the cost prices
def cost_price1 : ℝ := selling_price1 - profit1
def cost_price2 : ℝ := selling_price2 - profit2
def cost_price3 : ℝ := selling_price3 - profit3

-- Calculate total cost price and total selling price
def total_cost_price : ℝ := cost_price1 + cost_price2 + cost_price3
def total_selling_price : ℝ := selling_price1 + selling_price2 + selling_price3

-- Calculate total profit
def total_profit : ℝ := total_selling_price - total_cost_price

-- Calculate overall gain percentage
def overall_gain_percentage : ℝ := (total_profit / total_cost_price) * 100

-- Theorem statement
theorem overall_gain_percentage_approx :
  |overall_gain_percentage - 31.63| < 0.01 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_gain_percentage_approx_l1114_111466
