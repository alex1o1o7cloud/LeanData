import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l183_18336

/-- A hyperbola with center at the origin and one focus at (-√5, 0) -/
structure Hyperbola where
  /-- The focus of the hyperbola -/
  focus : ℝ × ℝ
  /-- A point on the hyperbola -/
  point : ℝ × ℝ
  /-- The focus is at (-√5, 0) -/
  focus_def : focus = (-Real.sqrt 5, 0)
  /-- The midpoint of the line segment from the focus to the point is at (0, 2) -/
  midpoint_def : ((focus.1 + point.1) / 2, (focus.2 + point.2) / 2) = (0, 2)

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 - y^2/4 = 1

/-- The eccentricity of the hyperbola -/
noncomputable def eccentricity : ℝ := Real.sqrt 5

/-- Theorem stating that the given hyperbola satisfies the equation x² - y²/4 = 1
    and has eccentricity √5 -/
theorem hyperbola_properties (h : Hyperbola) :
  (∀ x y, hyperbola_equation x y ↔ (x, y) = h.point) ∧
  eccentricity = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l183_18336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continued_fraction_reciprocal_l183_18391

/-- The continued fraction representation of x --/
noncomputable def x : ℝ := Real.sqrt 3 + 1

/-- The theorem stating the value of 1/((x+2)(x-3)) --/
theorem continued_fraction_reciprocal :
  1 / ((x + 2) * (x - 3)) = ((Real.sqrt 3) + 6) / -33 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continued_fraction_reciprocal_l183_18391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_triple_solutions_l183_18384

theorem integer_triple_solutions :
  ∀ x y z : ℕ,
  x > y ∧ y > z ∧ z > 0 ∧ x^2 = y * 2^z + 1 →
  ((∃ z' : ℕ, z' ≥ 4 ∧ z = z' ∧ x = 2^(z'-1) + 1 ∧ y = 2^(z'-2) + 1) ∨
   (∃ z' : ℕ, z' ≥ 5 ∧ z = z' ∧ x = 2^(z'-1) - 1 ∧ y = 2^(z'-2) - 1) ∨
   (∃ z' : ℕ, z' ≥ 3 ∧ z = z' ∧ x = 2^z' - 1 ∧ y = 2^z' - 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_triple_solutions_l183_18384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_fraction_condition_l183_18309

theorem integer_fraction_condition (n : ℕ) :
  (∃ a b : ℕ, a > b ∧ a > 0 ∧ b > 0 ∧ n = (4 * a * b) / (a - b)) →
  (n > 4 ∧ (¬(n % 4 = 3) ∨ ¬(Nat.Prime n ∧ n % 4 = 3))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_fraction_condition_l183_18309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_length_l183_18337

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def Circle.onCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

theorem circle_chord_length (c : Circle) (A B : ℝ × ℝ) :
  c.radius = 3 →
  c.onCircle A →
  c.onCircle B →
  A ≠ B →
  ‖(A.1 - c.center.1, A.2 - c.center.2) + (B.1 - c.center.1, B.2 - c.center.2)‖ = 2 * Real.sqrt 5 →
  ‖(A.1 - B.1, A.2 - B.2)‖ = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_length_l183_18337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_census_suitable_scenarios_l183_18350

/-- Represents a survey scenario --/
inductive SurveyScenario
| TVLifespan
| ManuscriptReview
| RiverPollution
| SchoolUniformSizes

/-- Determines if a survey scenario is suitable for a census --/
def isSuitableForCensus (scenario : SurveyScenario) : Bool :=
  match scenario with
  | .TVLifespan => false
  | .ManuscriptReview => true
  | .RiverPollution => false
  | .SchoolUniformSizes => true

/-- The list of all survey scenarios --/
def allScenarios : List SurveyScenario :=
  [.TVLifespan, .ManuscriptReview, .RiverPollution, .SchoolUniformSizes]

theorem census_suitable_scenarios :
  (allScenarios.filter isSuitableForCensus) =
  [SurveyScenario.ManuscriptReview, SurveyScenario.SchoolUniformSizes] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_census_suitable_scenarios_l183_18350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_properties_l183_18333

/-- A sequence defined recursively with the first two terms being coprime positive integers -/
def SpecialSequence (x₁ x₂ : ℕ) : ℕ → ℕ
  | 0 => x₁
  | 1 => x₂
  | (n + 2) => SpecialSequence x₁ x₂ (n + 1) * SpecialSequence x₁ x₂ n + 1

/-- The main theorem about the special sequence -/
theorem special_sequence_properties (x₁ x₂ : ℕ) (h : Nat.Coprime x₁ x₂) (h₁ : x₁ > 0) (h₂ : x₂ > 0) :
  (∀ i > 1, ∃ j > i, (SpecialSequence x₁ x₂ i) ^ i ∣ (SpecialSequence x₁ x₂ j) ^ j) ∧
  ¬(∃ j > 1, x₁ ∣ (SpecialSequence x₁ x₂ j) ^ j) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_properties_l183_18333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_difference_l183_18395

noncomputable def arithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = d

noncomputable def sumOfArithmeticSequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

theorem arithmetic_sequence_difference (a : ℕ → ℝ) (d : ℝ) (n : ℕ) :
  arithmeticSequence a d →
  let b := λ m => sumOfArithmeticSequence a m / m
  arithmeticSequence (λ m => a m - b m) (d / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_difference_l183_18395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_max_sin_B_plus_sin_C_l183_18377

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  angle_sum : A + B + C = Real.pi
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

-- Define the given equation
def given_equation (t : Triangle) : Prop :=
  2 * t.a * Real.sin t.A = (2 * t.b + t.c) * Real.sin t.B + (2 * t.c + t.b) * Real.sin t.C

-- Theorem 1: Measure of angle A
theorem angle_A_measure (t : Triangle) (h : given_equation t) : t.A = 2 * Real.pi / 3 := by
  sorry

-- Theorem 2: Maximum value of sin(B) + sin(C)
theorem max_sin_B_plus_sin_C (t : Triangle) (h : given_equation t) : 
  ∃ (B' C' : Real), B' + C' = Real.pi / 3 ∧ 
    (∀ (B C : Real), B + C = Real.pi / 3 → Real.sin B + Real.sin C ≤ Real.sin B' + Real.sin C') ∧
    Real.sin B' + Real.sin C' = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_max_sin_B_plus_sin_C_l183_18377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equals_fraction_l183_18368

/-- The repeating decimal 0.56̄ -/
def repeating_decimal : ℚ := 56/99

/-- The fraction 56/99 -/
def fraction : ℚ := 56/99

/-- Theorem stating that the repeating decimal 0.56̄ is equal to the fraction 56/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equals_fraction_l183_18368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_red_shoes_l183_18357

theorem probability_two_red_shoes (total_shoes : ℕ) (red_shoes : ℕ) (green_shoes : ℕ) :
  total_shoes = red_shoes + green_shoes →
  total_shoes = 10 →
  red_shoes = 4 →
  green_shoes = 6 →
  (red_shoes : ℚ) / total_shoes * ((red_shoes - 1) : ℚ) / (total_shoes - 1) = 333 / 2500 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_red_shoes_l183_18357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_condition_l183_18343

/-- Two vectors form an acute angle if their dot product is positive and not equal to the product of their magnitudes. -/
def acute_angle (a b : ℝ × ℝ) : Prop :=
  0 < (a.1 * b.1 + a.2 * b.2) ∧ (a.1 * b.1 + a.2 * b.2)^2 < (a.1^2 + a.2^2) * (b.1^2 + b.2^2)

/-- The main theorem stating the condition for λ given the vectors form an acute angle. -/
theorem acute_angle_condition (l : ℝ) :
  acute_angle (1, 3) (2 + l, 1) ↔ l > -5 ∧ l ≠ -5/3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_condition_l183_18343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_concurrent_l183_18338

-- Define the ellipse
def Γ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1}

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, -1)

-- Define lines l₁ and l₂
def l₁ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -2}
def l₂ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = -1}

-- Define point P
def P (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀, y₀)

-- Define line l₃
def l₃ (x₀ y₀ : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ t : ℝ, p = P x₀ y₀ + ⟨t * y₀, -t * x₀⟩}

-- Define points C, D, and E
def C : ℝ × ℝ := (-2, -1)
noncomputable def D (x₀ y₀ : ℝ) : ℝ × ℝ := sorry
noncomputable def E (x₀ y₀ : ℝ) : ℝ × ℝ := sorry

-- Define a function to check if three points are collinear
def collinear (p q r : ℝ × ℝ) : Prop :=
  (q.2 - p.2) * (r.1 - p.1) = (r.2 - p.2) * (q.1 - p.1)

-- Define a function to check if three lines are concurrent
def concurrent (l m n : (ℝ × ℝ) → (ℝ × ℝ) → Prop) : Prop :=
  ∃ p : ℝ × ℝ, l p p ∧ m p p ∧ n p p

-- Theorem statement
theorem lines_concurrent (x₀ y₀ : ℝ) (h₁ : x₀ > 0) (h₂ : y₀ > 0) (h₃ : P x₀ y₀ ∈ Γ) :
  concurrent
    (λ p q => collinear A (D x₀ y₀) p)
    (λ p q => collinear B (E x₀ y₀) p)
    (λ p q => collinear C (P x₀ y₀) p) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_concurrent_l183_18338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_lines_subset_line_plane_l183_18303

-- Define the set of angles formed by two skew lines
def skew_line_angles : Set ℝ := {θ | 0 < θ ∧ θ ≤ Real.pi / 2}

-- Define the set of angles formed by a line and a plane
def line_plane_angles : Set ℝ := {θ | 0 ≤ θ ∧ θ ≤ Real.pi / 2}

-- Theorem statement
theorem skew_lines_subset_line_plane :
  skew_line_angles ⊂ line_plane_angles := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_lines_subset_line_plane_l183_18303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_citizens_can_be_informed_and_gathered_prove_all_citizens_can_be_informed_and_gathered_l183_18361

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the kingdom -/
structure Kingdom where
  side_length : ℝ
  citizens : Set Point

/-- Represents the capabilities of citizens -/
structure Citizen where
  speed : ℝ
  start_position : Point

/-- The main theorem to be proved -/
theorem all_citizens_can_be_informed_and_gathered 
  (k : Kingdom) 
  (c : Citizen) 
  (total_time : ℝ) : Prop :=
  k.side_length = 2 ∧ 
  c.speed = 3 ∧ 
  total_time = 7 →
  ∃ (central_point : Point), 
    ∀ (p : Point), 
      p ∈ k.citizens → 
      ∃ (t : ℝ), 
        t ≤ total_time ∧ 
        (∃ (path : ℝ → Point), 
          path 0 = p ∧ 
          path t = central_point ∧ 
          ∀ (s : ℝ), 0 ≤ s ∧ s ≤ t → 
            (path s).x ≥ 0 ∧ (path s).x ≤ k.side_length ∧ 
            (path s).y ≥ 0 ∧ (path s).y ≤ k.side_length ∧
            ∃ (v : ℝ), v ≤ c.speed ∧ 
              ∀ (ε : ℝ), ε > 0 →
                (path (s + ε)).x - (path s).x ≤ v * ε ∧
                (path (s + ε)).y - (path s).y ≤ v * ε)

/-- Proof of the theorem -/
theorem prove_all_citizens_can_be_informed_and_gathered :
  all_citizens_can_be_informed_and_gathered 
    { side_length := 2, citizens := Set.univ } 
    { speed := 3, start_position := ⟨0, 0⟩ } 
    7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_citizens_can_be_informed_and_gathered_prove_all_citizens_can_be_informed_and_gathered_l183_18361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_axial_cross_section_area_l183_18319

/-- Definition of a cone -/
structure Cone where
  slant_height : ℝ
  net_is_semicircle : Prop
  axial_cross_section_area : ℝ

/-- A cone with a semicircular net and slant height 2 has an axial cross-section area of √3 -/
theorem cone_axial_cross_section_area (cone : Cone) : 
  cone.slant_height = 2 → 
  cone.net_is_semicircle → 
  cone.axial_cross_section_area = Real.sqrt 3 := by
  intro h1 h2
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_axial_cross_section_area_l183_18319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decrease_interval_l183_18397

noncomputable def f (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ)

theorem monotonic_decrease_interval 
  (φ : ℝ) 
  (h1 : 0 < φ) 
  (h2 : φ < π/2) 
  (h3 : f 0 φ = Real.sqrt 3) :
  ∃ (a b : ℝ), 
    a = π/12 ∧ 
    b = 7*π/12 ∧ 
    ∀ x ∈ Set.Icc 0 π, 
      (∀ y ∈ Set.Icc a b, x < y → f x φ > f y φ) ∧
      (∀ y ∈ Set.Ioo 0 a, x < y → f x φ < f y φ) ∧
      (∀ y ∈ Set.Ioo b π, x < y → f x φ < f y φ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decrease_interval_l183_18397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_extra_men_needed_l183_18398

/-- Represents the road construction project --/
structure RoadProject where
  totalLength : ℚ
  totalDays : ℚ
  initialMen : ℚ
  daysPassed : ℚ
  lengthCompleted : ℚ

/-- Calculates the number of extra men needed to complete the project on time --/
noncomputable def extraMenNeeded (project : RoadProject) : ℚ :=
  let remainingLength := project.totalLength - project.lengthCompleted
  let remainingDays := project.totalDays - project.daysPassed
  let newWorkRate := remainingLength / remainingDays
  let workRatePerMan := (project.totalLength / project.totalDays) / project.initialMen
  let totalMenNeeded := newWorkRate / workRatePerMan
  totalMenNeeded - project.initialMen

/-- Theorem stating that for the given project, 6 extra men are needed --/
theorem six_extra_men_needed (project : RoadProject) 
  (h1 : project.totalLength = 10)
  (h2 : project.totalDays = 300)
  (h3 : project.initialMen = 30)
  (h4 : project.daysPassed = 100)
  (h5 : project.lengthCompleted = 2) :
  extraMenNeeded project = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_extra_men_needed_l183_18398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_rearrangements_l183_18354

def base : ℕ := 3

def expression : ℕ → ℕ → ℕ
| 0, _ => base
| n+1, h => base ^ (expression n h)

def rearrangements : List (ℕ → ℕ) := [
  (λ _ => expression 3 3),
  (λ _ => base^((base^base)^base)),
  (λ _ => ((base^base)^base)^base),
  (λ _ => (base^(base^base))^base),
  (λ _ => (base^base)^(base^base))
]

theorem distinct_rearrangements :
  (rearrangements.map (λ f => f 0)).toFinset.card = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_rearrangements_l183_18354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l183_18313

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + Real.log (x - 2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Ioi 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l183_18313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_island_puzzle_l183_18393

-- Define the inhabitants
inductive Inhabitant : Type
| K : Inhabitant
| M : Inhabitant
| P : Inhabitant

-- Define a function to represent whether an inhabitant is a truth-teller or a liar
def is_truth_teller : Inhabitant → Prop := fun _ => True

-- Define K's statement
def K_statement (h : Inhabitant → Prop) : Prop :=
  ∀ i : Inhabitant, ¬(h i)

-- Define M's statement
def M_statement (h : Inhabitant → Prop) : Prop :=
  ∃ i : Inhabitant, h i

-- The main theorem
theorem island_puzzle :
  ∃! h : Inhabitant → Prop,
    (h Inhabitant.K = false) ∧
    (h Inhabitant.M = true) ∧
    (h Inhabitant.P = false) ∧
    (h Inhabitant.K ↔ K_statement h) ∧
    (h Inhabitant.M ↔ M_statement h) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_island_puzzle_l183_18393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_removable_rooks_l183_18346

/-- Represents a chessboard configuration -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- Checks if a rook at position (i, j) attacks an odd number of other rooks -/
def attacks_odd (board : Chessboard) (i j : Fin 8) : Bool :=
  sorry

/-- Removes a rook from the board at position (i, j) -/
def remove_rook (board : Chessboard) (i j : Fin 8) : Chessboard :=
  sorry

/-- Checks if a move is valid (i.e., removes a rook that attacks an odd number of rooks) -/
def is_valid_move (board : Chessboard) (i j : Fin 8) : Bool :=
  sorry

/-- Returns the number of rooks on the board -/
def count_rooks (board : Chessboard) : Nat :=
  sorry

/-- Initial board configuration with all squares occupied -/
def initial_board : Chessboard :=
  sorry

/-- Theorem: The maximum number of rooks that can be removed is 59 -/
theorem max_removable_rooks :
  ∀ (final_board : Chessboard),
    (∃ (moves : List (Fin 8 × Fin 8)), 
      (∀ (move : Fin 8 × Fin 8), move ∈ moves → is_valid_move initial_board move.fst move.snd) ∧
      (final_board = moves.foldl (λ b m => remove_rook b m.fst m.snd) initial_board)) →
    count_rooks final_board ≥ 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_removable_rooks_l183_18346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_equals_4_85_l183_18304

/-- Circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Check if a point is on a circle -/
def isOnCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  distance p c.center = c.radius

/-- Check if a point is outside a circle -/
def isOutsideCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  distance p c.center > c.radius

theorem intersection_distance_equals_4_85 
  (A B C : Circle)
  (h1 : A.center = (0, 0))
  (h2 : B.center = (3, 0))
  (h3 : C.center = (-3, 0))
  (h4 : A.radius = 2.5)
  (h5 : B.radius = 3)
  (h6 : C.radius = 2)
  (B' C' : ℝ × ℝ)
  (hB' : isOnCircle B' A ∧ isOnCircle B' C ∧ isOutsideCircle B' B)
  (hC' : isOnCircle C' A ∧ isOnCircle C' B ∧ isOutsideCircle C' C) :
  distance B' C' = 4.85 := by
  sorry

#check intersection_distance_equals_4_85

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_equals_4_85_l183_18304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_integers_l183_18351

theorem existence_of_special_integers :
  ∃ (S : Finset ℕ), 
    Finset.card S = 2009 ∧ 
    (∀ a b, a ∈ S → b ∈ S → a ≠ b → a ≠ b) ∧
    (∀ x, x ∈ S → (Finset.sum S id) % x = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_integers_l183_18351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l183_18379

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Define the line
def line_eq (a b x y : ℝ) : Prop := 2*a*x - b*y + 2 = 0

-- Define the chord length
def chord_length : ℝ := 4

-- Theorem statement
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : line_eq a b (-1) 2) 
  (h_chord : ∃ x y, circle_eq x y ∧ line_eq a b x y ∧ 
    (x + 1)^2 + (y - 2)^2 = chord_length^2 / 4) : 
  (∀ x y, x > 0 ∧ y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l183_18379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetric_about_y_axis_l183_18369

noncomputable def g (x : ℝ) : ℝ := |⌈x⌉| - |⌈x + 1⌉|

theorem g_symmetric_about_y_axis : ∀ x : ℝ, g (-x) = g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetric_about_y_axis_l183_18369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_6_l183_18328

-- Define the geometric sequence and its properties
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Define the sum of the first n terms of a geometric sequence
noncomputable def geometric_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  (a 1) * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_6 (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  (∀ n, a n > 0) →
  q > 1 →
  a 3 + a 5 = 20 →
  a 2 * a 6 = 64 →
  geometric_sum a q 6 = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_6_l183_18328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_y_relationship_l183_18364

theorem inverse_proportion_y_relationship :
  ∀ (y₁ y₂ : ℝ),
  ((-2 : ℝ), y₁) ∈ {p : ℝ × ℝ | p.1 ≠ 0 ∧ p.2 = 2 / p.1} →
  ((-1 : ℝ), y₂) ∈ {p : ℝ × ℝ | p.1 ≠ 0 ∧ p.2 = 2 / p.1} →
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_y_relationship_l183_18364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_l183_18330

def is_valid_function (f : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, (f^[n]) n = n) ∧
  (∀ m n : ℕ, Int.natAbs (f (m * n) - f m * f n) < 2017)

theorem unique_function :
  ∀ f : ℕ → ℕ, is_valid_function f → ∀ n : ℕ, f n = n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_l183_18330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sides_intersection_plane_cube_l183_18362

/-- A cube is a three-dimensional solid object with six square faces. -/
structure Cube where

/-- A plane is a flat, two-dimensional surface that extends infinitely far. -/
structure Plane where

/-- A polygon is a plane figure with straight sides. -/
structure Polygon where
  sides : ℕ

/-- The intersection of a plane and a cube results in a polygon. -/
def intersection (c : Cube) (p : Plane) : Polygon :=
  sorry

/-- The maximum number of sides a polygon can have when formed by the intersection of a plane and a cube. -/
def max_sides : ℕ := 6

/-- Theorem stating that the maximum number of sides a polygon can have when formed by 
    the intersection of a plane and a cube is 6. -/
theorem max_sides_intersection_plane_cube :
  ∀ (c : Cube) (p : Plane), (intersection c p).sides ≤ max_sides :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sides_intersection_plane_cube_l183_18362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_dc_length_l183_18329

noncomputable section

open Real EuclideanGeometry

theorem right_triangle_dc_length 
  (A B C D : EuclideanSpace ℝ (Fin 2))
  (h_right_angle : angle A D B = π / 2)
  (h_ab_length : dist A B = 30)
  (h_sin_a : sin (angle B A C) = 4 / 5)
  (h_sin_c : sin (angle A C B) = 1 / 4) :
  dist D C = 24 * Real.sqrt 15 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_dc_length_l183_18329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_circumscribed_circle_radius_l183_18327

/-- The radius of the circumscribed circle of a rectangle -/
def circumscribed_circle_radius (shorter_side diagonal_angle : ℝ) : ℝ :=
  sorry

/-- Given a rectangle with shorter side 1 and acute angle between diagonals 60°,
    prove that the radius of its circumscribed circle is 1. -/
theorem rectangle_circumscribed_circle_radius
  (shorter_side : ℝ)
  (diagonal_angle : ℝ)
  (h1 : shorter_side = 1)
  (h2 : diagonal_angle = Real.pi / 3) :
  circumscribed_circle_radius shorter_side diagonal_angle = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_circumscribed_circle_radius_l183_18327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_invariant_l183_18360

theorem rotation_invariant (a b c φ : ℝ) : 
  let a₁ := a * (Real.cos φ)^2 - 2*b * Real.cos φ * Real.sin φ + c * (Real.sin φ)^2
  let b₁ := a * Real.cos φ * Real.sin φ + b * ((Real.cos φ)^2 - (Real.sin φ)^2) - c * Real.cos φ * Real.sin φ
  let c₁ := a * (Real.sin φ)^2 + 2*b * Real.cos φ * Real.sin φ + c * (Real.cos φ)^2
  a₁ * c₁ - b₁^2 = a * c - b^2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_invariant_l183_18360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l183_18399

/-- Calculates the length of a train given its speed, the time to cross a platform, and the platform length. -/
noncomputable def trainLength (speed : ℝ) (crossTime : ℝ) (platformLength : ℝ) : ℝ :=
  speed * (1000 / 3600) * crossTime - platformLength

/-- Theorem stating that a train running at 72 kmph and crossing a 290 m platform in 26 seconds has a length of 230 m. -/
theorem train_length_calculation :
  trainLength 72 26 290 = 230 := by
  -- Unfold the definition of trainLength
  unfold trainLength
  -- Simplify the arithmetic expression
  simp [mul_assoc, mul_comm, mul_left_comm]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l183_18399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l183_18371

def Coin := Bool

structure CoinFlip :=
  (penny : Coin)
  (nickel : Coin)
  (dime : Coin)
  (quarter : Coin)
  (half_dollar : Coin)

def all_same (c : CoinFlip) : Prop :=
  c.penny = c.dime ∧ c.penny = c.half_dollar

def total_outcomes : Nat := 32

def favorable_outcomes : Nat := 8

theorem coin_flip_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 4 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l183_18371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_half_value_max_expected_cost_budget_not_exceeded_l183_18326

-- Define the probability of failing a test
variable (p : ℝ) 

-- Define the conditions
axiom p_range : 0 < p ∧ p < 1

-- Define the probability function f
def f (p : ℝ) : ℝ := -3 * p^5 + 12 * p^4 - 17 * p^3 + 9 * p^2

-- Define the expected cost function
def expected_cost (p : ℝ) : ℝ := 120 + 240 * p * (1 - p)^2

-- Define the total cost function
def total_cost (p : ℝ) : ℝ := 300 * expected_cost p + 10000

-- Theorem statements
theorem f_half_value : f (1/2) = 25/32 := by sorry

theorem max_expected_cost : ∀ p, 0 < p ∧ p < 1 → expected_cost p ≤ 1400/9 := by sorry

theorem budget_not_exceeded : ∀ p, 0 < p ∧ p < 1 → total_cost p < 60000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_half_value_max_expected_cost_budget_not_exceeded_l183_18326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stan_average_speed_l183_18318

/-- Represents a driving segment with distance in miles and time in hours -/
structure DrivingSegment where
  distance : ℝ
  time : ℝ

/-- Calculates the average speed given a list of driving segments -/
noncomputable def averageSpeed (segments : List DrivingSegment) : ℝ :=
  let totalDistance := segments.map (λ s => s.distance) |>.sum
  let totalTime := segments.map (λ s => s.time) |>.sum
  totalDistance / totalTime

theorem stan_average_speed :
  let segments : List DrivingSegment := [
    { distance := 450, time := 8 },
    { distance := 480, time := 8 },
    { distance := 270, time := 5 }
  ]
  averageSpeed segments = 1200 / 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stan_average_speed_l183_18318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_l183_18339

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 3) - 1

theorem symmetry_of_f :
  ∀ (x : ℝ), f (Real.pi / 3 + (Real.pi / 3 - x)) = f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_l183_18339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_relationship_l183_18352

noncomputable def a : ℝ := (6 : ℝ) ^ (0.7 : ℝ)
noncomputable def b : ℝ := (0.7 : ℝ) ^ (6 : ℝ)
noncomputable def c : ℝ := Real.log 6 / Real.log 0.7

theorem magnitude_relationship : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_relationship_l183_18352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cafeteria_satisfaction_survey_l183_18389

/-- Represents the vote counts for a food option -/
structure VoteCounts where
  rating5 : Nat
  rating4 : Nat
  rating3 : Nat
  rating2 : Nat
  rating1 : Nat

/-- Calculates the mean satisfaction rating for a food option -/
noncomputable def meanSatisfactionRating (votes : VoteCounts) : Real :=
  let totalVotes := votes.rating5 + votes.rating4 + votes.rating3 + votes.rating2 + votes.rating1
  let weightedSum := 5 * votes.rating5 + 4 * votes.rating4 + 3 * votes.rating3 + 2 * votes.rating2 + votes.rating1
  (weightedSum : Real) / (totalVotes : Real)

/-- Calculates the total number of votes for a food option -/
def totalVotes (votes : VoteCounts) : Nat :=
  votes.rating5 + votes.rating4 + votes.rating3 + votes.rating2 + votes.rating1

/-- The main theorem to prove -/
theorem cafeteria_satisfaction_survey (option1 option2 option3 : VoteCounts)
    (h1 : option1 = { rating5 := 130, rating4 := 105, rating3 := 61, rating2 := 54, rating1 := 33 })
    (h2 : option2 = { rating5 := 78, rating4 := 174, rating3 := 115, rating2 := 81, rating1 := 27 })
    (h3 : option3 = { rating5 := 95, rating4 := 134, rating3 := 102, rating2 := 51, rating1 := 31 }) :
    (abs (meanSatisfactionRating option1 - 3.64) < 0.01) ∧
    (abs (meanSatisfactionRating option2 - 3.41) < 0.01) ∧
    (abs (meanSatisfactionRating option3 - 3.51) < 0.01) ∧
    (totalVotes option1 + totalVotes option2 + totalVotes option3 = 1271) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cafeteria_satisfaction_survey_l183_18389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l183_18372

/-- The equation of the asymptote of a hyperbola -/
def asymptote_equation (x y : ℝ) : Prop := ∃ k : ℝ, k * x = y ∨ k * x = -y

/-- Definition of a hyperbola -/
def is_hyperbola (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

/-- Definition of the distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- Theorem about the asymptote of a hyperbola -/
theorem hyperbola_asymptote 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hP : ∃ (x y : ℝ), is_hyperbola x y a b ∧ 
                     ∃ (x₁ y₁ x₂ y₂ : ℝ), 
                       distance x y x₁ y₁ + distance x y x₂ y₂ - distance x₁ y₁ x₂ y₂ = 2 * a ∧
                       Real.cos (60 * π / 180) = (distance x₁ y₁ x₂ y₂)^2 / (2 * distance x y x₁ y₁ * distance x y x₂ y₂) ∧
                       distance x y 0 0 = Real.sqrt 7 * a) :
  asymptote_equation x y ∧ ∃ k : ℝ, k = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l183_18372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_satisfying_condition_l183_18307

theorem infinitely_many_n_satisfying_condition (k : ℕ) :
  ∃ f : ℕ → ℕ, Function.Injective f ∧ ∀ m : ℕ, ∀ r : ℕ, r ≤ k → r ∣ (f m - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_satisfying_condition_l183_18307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_f_l183_18335

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := -3 * (Real.cos x)^2 + 4 * Real.sin x + 5

-- Theorem statement
theorem max_min_f :
  (∃ (x_max : ℝ), 0 < x_max ∧ x_max < Real.pi ∧ 
    (∀ (x : ℝ), 0 < x ∧ x < Real.pi → f x ≤ f x_max) ∧
    f x_max = 9) ∧
  (¬∃ (x_min : ℝ), 0 < x_min ∧ x_min < Real.pi ∧ 
    (∀ (x : ℝ), 0 < x ∧ x < Real.pi → f x_min ≤ f x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_f_l183_18335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_1997_if_1990_four_times_l183_18353

-- Define a polynomial with integer coefficients
def IntPolynomial := Polynomial ℤ

-- Define the property of a polynomial taking value 1990 at four distinct integers
def Takes1990AtFourInts (p : IntPolynomial) : Prop :=
  ∃ (a b c d : ℤ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    p.eval a = 1990 ∧ p.eval b = 1990 ∧ p.eval c = 1990 ∧ p.eval d = 1990

-- State the theorem
theorem no_1997_if_1990_four_times (p : IntPolynomial) :
  Takes1990AtFourInts p → ¬∃ (k : ℤ), p.eval k = 1997 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_1997_if_1990_four_times_l183_18353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_percentage_approximately_71_33_l183_18388

-- Define the properties of each liquid
noncomputable def liquid_A_volume : ℝ := 100
noncomputable def liquid_A_solution_percentage : ℝ := 0.25

noncomputable def liquid_B_volume : ℝ := 90
noncomputable def liquid_B_solution_percentage : ℝ := 0.30

noncomputable def liquid_C_volume : ℝ := 60
noncomputable def liquid_C_solution_percentage : ℝ := 0.40

noncomputable def liquid_D_volume : ℝ := 50
noncomputable def liquid_D_solution_percentage : ℝ := 0.20

-- Define the total volume of the mixture
noncomputable def total_volume : ℝ := liquid_A_volume + liquid_B_volume + liquid_C_volume + liquid_D_volume

-- Define the total amount of solution in the mixture
noncomputable def total_solution : ℝ := 
  liquid_A_volume * liquid_A_solution_percentage +
  liquid_B_volume * liquid_B_solution_percentage +
  liquid_C_volume * liquid_C_solution_percentage +
  liquid_D_volume * liquid_D_solution_percentage

-- Define the amount of water in the mixture
noncomputable def total_water : ℝ := total_volume - total_solution

-- Define the percentage of water in the mixture
noncomputable def water_percentage : ℝ := (total_water / total_volume) * 100

-- Theorem statement
theorem water_percentage_approximately_71_33 : 
  (water_percentage ≥ 71.32) ∧ (water_percentage ≤ 71.34) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_percentage_approximately_71_33_l183_18388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_max_l183_18358

/-- Tetrahedron with one variable edge length -/
structure Tetrahedron where
  x : ℝ
  h1 : 0 < x
  h2 : x < Real.sqrt 3

/-- Volume of the tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ := 
  (t.x / 12) * Real.sqrt (3 - t.x^2)

/-- The point where the volume is maximized -/
noncomputable def max_point : ℝ := Real.sqrt 6 / 2

/-- The maximum volume of the tetrahedron -/
noncomputable def max_volume : ℝ := 1 / 8

theorem tetrahedron_volume_max (t : Tetrahedron) : 
  (∀ y, 0 < y → y < Real.sqrt 3 → volume ⟨y, by sorry, by sorry⟩ ≤ max_volume) ∧ 
  volume ⟨max_point, by sorry, by sorry⟩ = max_volume := by
  sorry

#check tetrahedron_volume_max

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_max_l183_18358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_mixture_theorem_l183_18308

/-- Represents a salt solution with a given volume and concentration -/
structure SaltSolution where
  volume : ℝ
  concentration : ℝ
  volume_nonneg : 0 ≤ volume
  conc_between_0_and_1 : 0 ≤ concentration ∧ concentration ≤ 1

/-- Represents the mixture of two salt solutions -/
noncomputable def mix (s1 s2 : SaltSolution) : SaltSolution where
  volume := s1.volume + s2.volume
  concentration := (s1.volume * s1.concentration + s2.volume * s2.concentration) / (s1.volume + s2.volume)
  volume_nonneg := by
    apply add_nonneg
    exact s1.volume_nonneg
    exact s2.volume_nonneg
  conc_between_0_and_1 := by sorry

/-- Theorem: Mixing 70 ounces of 60% salt solution with 70 ounces of 20% salt solution results in a 40% salt solution -/
theorem salt_mixture_theorem :
  let s1 : SaltSolution := ⟨70, 0.6, by norm_num, by norm_num⟩
  let s2 : SaltSolution := ⟨70, 0.2, by norm_num, by norm_num⟩
  let result := mix s1 s2
  result.concentration = 0.4 := by
  simp [mix]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_mixture_theorem_l183_18308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blood_donation_expiration_l183_18332

-- Define the start date (January 1st at noon)
def start_date : Nat := 0  -- Represent as seconds since epoch

-- Define the effective period in seconds
def effective_period : Nat := Nat.factorial 10

-- Define the end date (expiration date)
def end_date : Nat := start_date + effective_period

-- Function to convert seconds to date components
def seconds_to_date (seconds : Nat) : (Nat × Nat × Nat) :=
  let days := seconds / (24 * 60 * 60)
  let year := 2023  -- Assuming the start year is 2023
  let month := if days < 31 then 1 else 2
  let day := if days < 31 then days + 1 else days - 31 + 1
  (year, month, day)

-- Theorem to prove
theorem blood_donation_expiration :
  let (year, month, day) := seconds_to_date end_date
  year = 2023 ∧ month = 2 ∧ day = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blood_donation_expiration_l183_18332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l183_18340

-- Define the two functions
noncomputable def f (x : ℝ) : ℝ := 3 * Real.log x
def g (x : ℝ) : ℝ := 3 * x - 9

-- Theorem statement
theorem intersection_count : ∃! x : ℝ, x > 0 ∧ f x = g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l183_18340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_theorem_l183_18356

/-- A bi-infinite sequence of elements of type α -/
def BiInfiniteSeq (α : Type*) := ℤ → α

/-- A sequence is periodic with period p if it repeats every p elements -/
def IsPeriodic {α : Type*} (s : BiInfiniteSeq α) (p : ℕ) : Prop :=
  ∀ i : ℤ, s (i + p) = s i

/-- A segment of length n starting at index i in a bi-infinite sequence -/
def Segment {α : Type*} (s : BiInfiniteSeq α) (i : ℤ) (n : ℕ) : Fin n → α :=
  λ j ↦ s (i + j)

/-- A segment s₁ is contained in a sequence s₂ if there exists an index where they match -/
def SegmentContained {α : Type*} {n : ℕ} (s₁ : Fin n → α) (s₂ : BiInfiniteSeq α) : Prop :=
  ∃ i : ℤ, ∀ j : Fin n, s₁ j = s₂ (i + j)

theorem largest_n_theorem {α : Type*} :
  ∀ n : ℕ,
  (∃ (A B : BiInfiniteSeq α),
    IsPeriodic A 1995 ∧
    ¬ IsPeriodic B 1995 ∧
    ∀ i : ℤ, SegmentContained (Segment B i n) A) →
  n ≤ 1995 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_theorem_l183_18356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l183_18396

/-- Given that a, b, and a+b form an arithmetic progression,
    a, b, and ab form a geometric progression,
    and 0 < log_m(ab) < 1, prove that m > 8 -/
theorem range_of_m (a b m : ℝ) 
  (h1 : 2 * b = a + (a + b))  -- arithmetic progression condition
  (h2 : b^2 = a * (a * b))    -- geometric progression condition
  (h3 : 0 < Real.log (a * b) / Real.log m)
  (h4 : Real.log (a * b) / Real.log m < 1) : 
  m > 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l183_18396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_two_planes_implies_planes_parallel_l183_18349

-- Define the types for lines and planes
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]

def Line (V : Type*) [NormedAddCommGroup V] := Set V
def Plane (V : Type*) [NormedAddCommGroup V] := Set V

-- Define perpendicularity between a line and a plane
def perpendicular {V : Type*} [NormedAddCommGroup V] (l : Line V) (p : Plane V) : Prop := sorry

-- Define parallelism between planes
def parallel {V : Type*} [NormedAddCommGroup V] (p1 p2 : Plane V) : Prop := sorry

-- State the theorem
theorem line_perp_two_planes_implies_planes_parallel 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]
  (a : Line V) (α β : Plane V) :
  perpendicular a α → perpendicular a β → parallel α β :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_two_planes_implies_planes_parallel_l183_18349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_exponential_increasing_l183_18344

-- Define the exponential function with base 1/2
noncomputable def f (x : ℝ) : ℝ := (1/2)^x

-- State that f is an exponential function
axiom f_is_exponential : ∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧ ∀ x, f x = a^x

-- State that f is not an increasing function
axiom f_not_increasing : ¬(∀ x y : ℝ, x < y → f x < f y)

-- Theorem to prove
theorem not_all_exponential_increasing :
  ¬(∀ g : ℝ → ℝ, (∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧ ∀ x, g x = a^x) →
    (∀ x y : ℝ, x < y → g x < g y)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_exponential_increasing_l183_18344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_5x_over_cos_x_l183_18320

theorem cos_5x_over_cos_x (x : ℝ) (h : Real.sin (3 * x) / Real.sin x = 5 / 3) : 
  Real.cos (5 * x) / Real.cos x = -11 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_5x_over_cos_x_l183_18320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_problem_l183_18334

theorem inequality_problem :
  (∀ t : ℝ, |t + 3| - |t - 2| ≤ 6 * m - m^2) →
  (∃ lambda : ℝ, lambda = 5 ∧ 
    (∀ m : ℝ, (∀ t : ℝ, |t + 3| - |t - 2| ≤ 6 * m - m^2) → m ≤ lambda) ∧
    (∀ x y z : ℝ, 3 * x + 4 * y + 5 * z = lambda → 
      x^2 + y^2 + z^2 ≥ (1/2) ∧ 
      ∃ x₀ y₀ z₀ : ℝ, 3 * x₀ + 4 * y₀ + 5 * z₀ = lambda ∧ x₀^2 + y₀^2 + z₀^2 = 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_problem_l183_18334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l183_18374

noncomputable def f (m n t x : ℝ) : ℝ := (m * x^2 + t) / (x + n)

theorem function_properties (t : ℝ) (h_t : t > 0) :
  -- Part 1
  (∀ x, f 1 0 t (-x) = -f 1 0 t x) ∧ 
  f 1 0 t 1 = t + 1 →
  -- Part 2
  (∀ x y, 0 < x ∧ x < y ∧ y < Real.sqrt t → f 1 0 t x > f 1 0 t y) ∧
  (∀ x y, Real.sqrt t ≤ x ∧ x < y → f 1 0 t x < f 1 0 t y) ∧
  -- Part 3
  (∃ x_max x_min, x_max ∈ Set.Icc 2 4 ∧ x_min ∈ Set.Icc 2 4 ∧
    ∀ x ∈ Set.Icc 2 4, f 1 0 t x ≤ f 1 0 t x_max ∧ f 1 0 t x_min ≤ f 1 0 t x ∧
    f 1 0 t x_max = f 1 0 t x_min + 2) →
  t = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l183_18374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_triangle_is_right_angle_l183_18302

noncomputable section

variable (a : ℝ)

/-- The intersection points of the hyperbola and parabola -/
def A (a : ℝ) : ℝ × ℝ := (1, a)
def B (a : ℝ) : ℝ × ℝ := (-1, -a)
def C (a : ℝ) : ℝ × ℝ := (-a, -1)

/-- The slopes of line segments AC and BC -/
def k_AC (a : ℝ) : ℝ := (C a).2 - (A a).2 / ((C a).1 - (A a).1)
def k_BC (a : ℝ) : ℝ := (C a).2 - (B a).2 / ((C a).1 - (B a).1)

/-- Theorem: The triangle formed by the intersection points is a right triangle -/
theorem intersection_triangle_is_right_angle (h : a ≠ 1 ∧ a ≠ -1) :
  k_AC a * k_BC a = -1 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_triangle_is_right_angle_l183_18302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_problem_l183_18310

/-- A quadratic function with specific properties -/
def f (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- The conditions for the quadratic function -/
structure QuadraticConditions (a b c : ℝ) : Prop where
  min_at_neg_one : f a b c (-1) = -3
  min_value : ∀ x, f a b c x ≥ -3
  value_at_two : f a b c 2 = 15/4

/-- The theorem to be proved -/
theorem quadratic_problem {a b c m : ℝ} 
  (h : QuadraticConditions a b c) (h_m : m > 1) :
  (∀ x ∈ Set.Icc (-2*m + 3) (-m + 2), f a b c x ≥ -9/4) ∧
  (∃ x ∈ Set.Icc (-2*m + 3) (-m + 2), f a b c x = -9/4) →
  m = 2 - Real.sqrt 7 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_problem_l183_18310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_y_saves_two_minutes_l183_18311

-- Define the routes and their characteristics
noncomputable def route_x_distance : ℝ := 8
noncomputable def route_x_speed : ℝ := 40

noncomputable def route_y_segment1_distance : ℝ := 5
noncomputable def route_y_segment1_speed : ℝ := 50
noncomputable def route_y_segment2_distance : ℝ := 1
noncomputable def route_y_segment2_speed : ℝ := 20
noncomputable def route_y_segment3_distance : ℝ := 1
noncomputable def route_y_segment3_speed : ℝ := 60

-- Define the function to calculate time given distance and speed
noncomputable def calculate_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

-- Theorem statement
theorem route_y_saves_two_minutes : 
  let time_x := calculate_time route_x_distance route_x_speed
  let time_y := calculate_time route_y_segment1_distance route_y_segment1_speed +
                calculate_time route_y_segment2_distance route_y_segment2_speed +
                calculate_time route_y_segment3_distance route_y_segment3_speed
  (time_x - time_y) * 60 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_y_saves_two_minutes_l183_18311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_arithmetic_sequence_l183_18363

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  sum : ℕ → ℝ
  sum_def : ∀ n, sum n = n / 2 * (2 * a 1 + (n - 1) * d)
  arithmetic : ∀ n, a (n + 1) = a n + d

/-- The direction vector of a line passing through two points -/
def directionVector (P Q : ℕ × ℝ) : ℝ × ℝ :=
  (Q.1 - P.1, Q.2 - P.2)

theorem direction_vector_arithmetic_sequence 
  (seq : ArithmeticSequence) 
  (h1 : seq.sum 2 = 10) 
  (h2 : seq.sum 5 = 55) :
  ∀ n : ℕ, directionVector (n, seq.a n) (n + 2, seq.a (n + 2)) = (2, 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_arithmetic_sequence_l183_18363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eleventh_number_proof_l183_18305

theorem eleventh_number_proof (numbers : List ℚ) 
  (h_count : numbers.length = 21)
  (h_avg_all : numbers.sum / 21 = 44)
  (h_avg_first : (numbers.take 11).sum / 11 = 48)
  (h_avg_last : (numbers.drop 10).sum / 11 = 41) :
  numbers[10] = 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eleventh_number_proof_l183_18305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_inverse_is_zero_matrix_l183_18306

def A : Matrix (Fin 2) (Fin 2) ℝ := !![6, -4; -3, 2]

theorem matrix_inverse_is_zero_matrix :
  A⁻¹ = !![0, 0; 0, 0] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_inverse_is_zero_matrix_l183_18306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_15_15_l183_18365

/-- Represents the time on a clock -/
structure ClockTime where
  hours : ℕ
  minutes : ℕ

/-- Calculates the angle of the hour hand from 12 o'clock position -/
noncomputable def hourHandAngle (t : ClockTime) : ℝ :=
  (t.hours % 12 : ℝ) * 30 + (t.minutes : ℝ) * 0.5

/-- Calculates the angle of the minute hand from 12 o'clock position -/
noncomputable def minuteHandAngle (t : ClockTime) : ℝ :=
  (t.minutes : ℝ) * 6

/-- Calculates the angle between hour and minute hands -/
noncomputable def angleBetweenHands (t : ClockTime) : ℝ :=
  |hourHandAngle t - minuteHandAngle t|

/-- Theorem: At 15:15, the angle between the minute and hour hands is 7.5 degrees -/
theorem angle_at_15_15 :
  angleBetweenHands ⟨15, 15⟩ = 7.5 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval angleBetweenHands ⟨15, 15⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_15_15_l183_18365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_element_in_set_representation_l183_18381

theorem element_in_set_representation (A : Set α) (a : α) :
  a ∈ A → (a ∈ A ∧ ¬(({a} : Set α) ⊆ A)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_element_in_set_representation_l183_18381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_second_equilibrium_ball_at_equilibrium_at_second_time_l183_18376

/-- The motion equation of a ball in simple harmonic motion --/
noncomputable def motion_equation (t : ℝ) : ℝ := 15 * Real.sin (2 * t + Real.pi / 6)

/-- The velocity equation derived from the motion equation --/
noncomputable def velocity_equation (t : ℝ) : ℝ := 30 * Real.cos (2 * t + Real.pi / 6)

/-- The time when the ball returns to the equilibrium position for the second time --/
noncomputable def second_equilibrium_time : ℝ := 11 * Real.pi / 12

/-- Theorem stating that the velocity at the second equilibrium position is 30 cm/s --/
theorem velocity_at_second_equilibrium :
  velocity_equation second_equilibrium_time = 30 := by
  sorry

/-- Theorem stating that the ball is at the equilibrium position at the calculated time --/
theorem ball_at_equilibrium_at_second_time :
  motion_equation second_equilibrium_time = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_second_equilibrium_ball_at_equilibrium_at_second_time_l183_18376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_benzene_molecular_weight_l183_18301

/-- The atomic weight of carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The number of carbon atoms in a benzene molecule -/
def carbon_count : ℕ := 6

/-- The number of hydrogen atoms in a benzene molecule -/
def hydrogen_count : ℕ := 6

/-- The molecular weight of benzene in g/mol -/
def benzene_weight : ℝ := carbon_weight * (carbon_count : ℝ) + hydrogen_weight * (hydrogen_count : ℝ)

/-- Theorem stating that the molecular weight of benzene is approximately 78.108 g/mol -/
theorem benzene_molecular_weight : ∃ ε > 0, |benzene_weight - 78.108| < ε := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_benzene_molecular_weight_l183_18301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_even_before_first_odd_l183_18325

/-- A fair 6-sided die is modeled as a function from ℕ to {1,2,3,4,5,6} --/
def FairDie := ℕ → Fin 6

/-- An infinite sequence of die rolls --/
def RollSequence := ℕ → Fin 6

/-- Predicate to check if a number is even --/
def isEven (n : Fin 6) : Prop := n % 2 = 0

/-- Predicate to check if a number is odd --/
def isOdd (n : Fin 6) : Prop := n % 2 ≠ 0

/-- Predicate to check if all even numbers appear in a sequence before the first odd number --/
def allEvenBeforeFirstOdd (s : RollSequence) : Prop :=
  ∃ n : ℕ, (∀ m < n, isEven (s m)) ∧ 
            (∀ k : Fin 6, isEven k → ∃ m < n, s m = k) ∧
            isOdd (s n)

/-- The probability measure on the space of roll sequences --/
noncomputable def P : (RollSequence → Prop) → ℝ := sorry

/-- The theorem to be proved --/
theorem probability_all_even_before_first_odd :
  P allEvenBeforeFirstOdd = 1/20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_even_before_first_odd_l183_18325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_ratio_l183_18345

/-- Represents the speed of a cyclist -/
structure CyclistSpeed where
  distance : ℝ  -- distance in meters
  time : ℝ      -- time in minutes
  speed : ℝ     -- speed in meters per minute

/-- Calculate the speed given distance and time -/
noncomputable def calculateSpeed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- Calculate the ratio of two speeds -/
def speedRatio (speed1 : ℝ) (speed2 : ℝ) : ℝ × ℝ :=
  (speed1, speed2)

theorem cyclist_speed_ratio :
  let cyclistA : CyclistSpeed := {
    distance := 3 * Real.pi * 1000,  -- 3 laps of a circular track with 1km diameter
    time := 10,
    speed := calculateSpeed (3 * Real.pi * 1000) 10
  }
  let cyclistB : CyclistSpeed := {
    distance := 4 * 5000,  -- 2 round trips on a 5km straight track
    time := 5,
    speed := calculateSpeed (4 * 5000) 5
  }
  speedRatio cyclistA.speed cyclistB.speed = (3 * Real.pi, 40) := by
  sorry

#check cyclist_speed_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_ratio_l183_18345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_condition_l183_18316

theorem necessary_not_sufficient_condition :
  (∀ x : ℝ, |x - 1| ≤ 1 → 2 - x ≥ 0) ∧
  (∃ x : ℝ, 2 - x ≥ 0 ∧ |x - 1| > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_condition_l183_18316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_increase_total_percentage_increase_l183_18385

theorem price_increase (P : ℝ) (h : P > 0) : 
  P * (1 + 0.15) * (1 + 0.40) * (1 + 0.20) * (1 - 0.10) * (1 + 0.25) = P * 2.1735 :=
by sorry

theorem total_percentage_increase : 
  (2.1735 - 1) * 100 = 117.35 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_increase_total_percentage_increase_l183_18385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_cans_for_soda_l183_18366

theorem minimum_cans_for_soda (can_capacity : ℕ) (total_needed : ℕ) (h1 : can_capacity = 15) (h2 : total_needed = 150) : 
  (Nat.ceil (total_needed / can_capacity : ℚ)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_cans_for_soda_l183_18366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_probability_l183_18387

theorem divisibility_probability : 
  ∃ (count : ℕ), count = (Finset.filter (λ n : ℕ ↦ 1 ≤ n ∧ n ≤ 1000 ∧ (13 ∣ n * (n + 1)) ∧ (17 ∣ n * (n + 1))) (Finset.range 1001)).card ∧ count = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_probability_l183_18387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l183_18331

/-- An arithmetic sequence with its first term and common difference -/
structure ArithmeticSequence where
  a1 : ℝ
  d : ℝ

/-- The nth term of an arithmetic sequence -/
noncomputable def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a1 + seq.d * (n - 1 : ℝ)

/-- The sum of the first n terms of an arithmetic sequence -/
noncomputable def ArithmeticSequence.sum (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  n * (2 * seq.a1 + (n - 1 : ℝ) * seq.d) / 2

theorem arithmetic_sequence_max_sum
  (seq : ArithmeticSequence)
  (h1 : seq.a1 > 0)
  (h2 : seq.sum 3 = seq.sum 16) :
  ∃ (n : ℕ), (n = 9 ∨ n = 10) ∧
    ∀ (m : ℕ), seq.sum m ≤ seq.sum n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l183_18331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_rotation_theorem_l183_18390

noncomputable def dilation_matrix (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![k, 0; 0, k]

noncomputable def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![(Real.sqrt 2) / 2, (Real.sqrt 2) / 2; -(Real.sqrt 2) / 2, (Real.sqrt 2) / 2]

noncomputable def result_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![5, 5; -5, 5]

theorem dilation_rotation_theorem (k : ℝ) (h₁ : k > 0) :
  rotation_matrix * dilation_matrix k = result_matrix → k = 5 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_rotation_theorem_l183_18390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_pour_function_l183_18383

/-- Represents the function for the amount of pure alcohol poured out -/
noncomputable def f (x : ℝ) : ℝ := 1 + (19/20) * x

/-- The total volume of the container in liters -/
def totalVolume : ℝ := 20

/-- The volume poured out in each iteration in liters -/
def pourVolume : ℝ := 1

/-- Theorem stating the correct form of the function f -/
theorem alcohol_pour_function (k : ℕ) (x : ℝ) 
  (h1 : k ≥ 1) 
  (h2 : x ≥ 0) 
  (h3 : x ≤ totalVolume) :
  f x = 1 + (19/20) * x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_pour_function_l183_18383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rook_in_rectangle_l183_18375

/-- Represents a chessboard with rooks -/
structure Chessboard where
  rooks : Finset (Nat × Nat)
  valid : ∀ r1 r2, r1 ∈ rooks → r2 ∈ rooks → r1 ≠ r2 → r1.1 ≠ r2.1 ∧ r1.2 ≠ r2.2

/-- Represents a rectangle on the chessboard -/
structure Rectangle where
  top_left : Nat × Nat
  width : Nat
  height : Nat

/-- Checks if a position is within a rectangle -/
def in_rectangle (pos : Nat × Nat) (rect : Rectangle) : Prop :=
  pos.1 ≥ rect.top_left.1 ∧ pos.1 < rect.top_left.1 + rect.width ∧
  pos.2 ≥ rect.top_left.2 ∧ pos.2 < rect.top_left.2 + rect.height

theorem rook_in_rectangle (board : Chessboard) :
  board.rooks.card = 8 →
  ∀ rect : Rectangle, rect.width = 4 ∧ rect.height = 5 →
  ∃ rook ∈ board.rooks, in_rectangle rook rect := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rook_in_rectangle_l183_18375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_diff_l183_18317

theorem min_abs_diff (a b : ℕ) (h : a * b - 6 * a + 5 * b = 373) : 
  (∀ x y : ℕ, x * y - 6 * x + 5 * y = 373 → |Int.ofNat x - Int.ofNat y| ≥ |Int.ofNat a - Int.ofNat b|) →
  |Int.ofNat a - Int.ofNat b| = 31 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_diff_l183_18317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_correct_l183_18359

def sandwich_price : ℚ := 245 / 100
def soda_price : ℚ := 87 / 100
def chips_price : ℚ := 129 / 100
def sandwich_quantity : ℕ := 2
def soda_quantity : ℕ := 4
def chips_quantity : ℕ := 3
def sandwich_discount : ℚ := 1 / 10
def sales_tax : ℚ := 2 / 25

def total_cost : ℚ :=
  let sandwich_cost := sandwich_quantity * sandwich_price
  let soda_cost := soda_quantity * soda_price
  let chips_cost := chips_quantity * chips_price
  let discounted_sandwich_cost := sandwich_cost * (1 - sandwich_discount)
  let subtotal := discounted_sandwich_cost + soda_cost + chips_cost
  let tax_amount := subtotal * sales_tax
  let total := subtotal + tax_amount
  (total * 100).floor / 100

theorem total_cost_is_correct : total_cost = 127 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_correct_l183_18359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_PQRS_value_l183_18378

-- Define the points P, Q, R, S as noncomputable
noncomputable def P : ℝ := 3 * Real.sqrt 2010 + 2 * Real.sqrt 2011
noncomputable def Q : ℝ := -3 * Real.sqrt 2010 - 2 * Real.sqrt 2011
noncomputable def R : ℝ := 3 * Real.sqrt 2010 - 2 * Real.sqrt 2011
noncomputable def S : ℝ := 2 * Real.sqrt 2011 - 3 * Real.sqrt 2010

-- State the theorem
theorem PQRS_value : P * Q * R * S = -40434584 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_PQRS_value_l183_18378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l183_18322

def α_set : Set ℝ := {-2, -1, -1/2, 1/3, 1/2, 1, 2, 3}

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def is_decreasing_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f y < f x

theorem power_function_properties :
  ∃! α, α ∈ α_set ∧ is_even_function (fun x ↦ x^α) ∧ 
  is_decreasing_on_positive (fun x ↦ x^α) ∧ 
  α = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l183_18322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l183_18315

noncomputable def f (a b x : ℝ) : ℝ := a * (2 * (Real.cos (x / 2))^2 + Real.sin x) + b

theorem f_properties :
  ∀ (a b : ℝ),
  (∀ (k : ℤ), (∀ (x : ℝ), x ∈ Set.Icc (2 * π * (k : ℝ) + π/4) (2 * π * (k : ℝ) + 5*π/4) → 
    Monotone (fun x => f (-1) b x))) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 π → f a b x ∈ Set.Icc 5 8) →
  ((a = 3 * Real.sqrt 2 - 3 ∧ b = 5) ∨ (a = 3 - 3 * Real.sqrt 2 ∧ b = 8)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l183_18315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_a_plus_2b_l183_18394

theorem max_value_of_a_plus_2b :
  ∃ (M : ℝ), M = Real.sqrt 3 ∧ 
  ∀ (a b c : ℝ), a^2 + 2*b^2 + 3*c^2 = 1 → a + 2*b ≤ M :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_a_plus_2b_l183_18394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l183_18321

-- Define the constants
noncomputable def a : ℝ := (3 : ℝ) ^ (0.4 : ℝ)
noncomputable def b : ℝ := (0.4 : ℝ) ^ (3 : ℝ)
noncomputable def c : ℝ := Real.log 3 / Real.log 0.4

-- State the theorem
theorem order_of_abc : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l183_18321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_l183_18392

theorem min_value_a (a b : ℕ) (h1 : b - a = 2013) 
  (h2 : ∃ x : ℕ, x^2 - a*x + b = 0) : 
  (∀ a' : ℕ, (∃ b' : ℕ, b' - a' = 2013 ∧ ∃ x : ℕ, x^2 - a'*x + b' = 0) → a' ≥ a) ∧ a = 93 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_l183_18392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l183_18367

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- The equation of the ellipse -/
def Ellipse.equation (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- The eccentricity of the ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- Theorem statement -/
theorem ellipse_properties (e : Ellipse) 
  (h_foci : e.equation ⟨-1, 0⟩ ∧ e.equation ⟨1, 0⟩)
  (m : Point) 
  (h_m_on_ellipse : e.equation m)
  (h_m_perpendicular : m.x = 1)
  (n : Point)
  (h_n_on_ellipse : e.equation n)
  (h_mn_slope : (n.y - m.y) / (n.x - m.x) = 3/4)
  (h_mn_f1n_ratio : (m.x - n.x)^2 + (m.y - n.y)^2 = 25 * ((n.x + 1)^2 + n.y^2)) :
  (e.a^2 = 4 ∧ e.b^2 = 3) ∧ 
  e.eccentricity = Real.sqrt 21 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l183_18367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l183_18382

-- Define the constants as noncomputable
noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := Real.log 10 / Real.log 15
noncomputable def c : ℝ := Real.sin (1/2)

-- State the theorem
theorem ordering_abc : b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l183_18382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_and_perpendicular_distance_l183_18324

noncomputable def point : ℝ × ℝ := (12, -5)
def line (x : ℝ) : ℝ := 3 * x + 4

noncomputable def distance_from_origin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 ^ 2) + (p.2 ^ 2))

noncomputable def perpendicular_distance (p : ℝ × ℝ) (m a : ℝ) : ℝ :=
  abs (m * p.1 - p.2 + a) / Real.sqrt (m ^ 2 + 1)

theorem distance_and_perpendicular_distance :
  distance_from_origin point = 13 ∧
  perpendicular_distance point 3 4 = 4.5 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_and_perpendicular_distance_l183_18324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uncle_vanya_journey_l183_18342

/-- Represents the time in minutes for Uncle Vanya's journey -/
noncomputable def journey_time (walk_speed : ℝ) (cycle_speed : ℝ) (drive_speed : ℝ) 
  (walk_dist : ℝ) (cycle_dist : ℝ) (drive_dist : ℝ) : ℝ :=
  walk_dist / walk_speed + cycle_dist / cycle_speed + drive_dist / drive_speed

theorem uncle_vanya_journey : 
  ∀ (walk_speed cycle_speed drive_speed : ℝ),
  walk_speed > 0 ∧ cycle_speed > 0 ∧ drive_speed > 0 →
  journey_time walk_speed cycle_speed drive_speed 4 6 40 = 132 →
  journey_time walk_speed cycle_speed drive_speed 5 8 30 = 144 →
  journey_time walk_speed cycle_speed drive_speed 8 10 160 = 348 :=
by
  sorry

#eval "Uncle Vanya's journey theorem is defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_uncle_vanya_journey_l183_18342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_area_theorem_l183_18347

/-- Represents a cone with given height and volume -/
structure Cone where
  height : ℝ
  volume : ℝ

/-- The area of the base of a cone -/
noncomputable def baseArea (c : Cone) : ℝ :=
  (3 * c.volume) / c.height

/-- Theorem: The area of the base of a cone with height 6 cm and volume 60 cm³ is 30 cm² -/
theorem cone_base_area_theorem (c : Cone) (h1 : c.height = 6) (h2 : c.volume = 60) : 
  baseArea c = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_area_theorem_l183_18347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l183_18386

theorem chord_length (r d : ℝ) (hr : r = 4) (hd : d = 3) :
  2 * Real.sqrt (r^2 - d^2) = 2 * Real.sqrt 7 :=
by
  -- Replace the proof with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l183_18386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l183_18312

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.c * Real.sin t.A - 2 * t.b * Real.sin t.C = 0 ∧
  t.a^2 - t.b^2 - t.c^2 = (Real.sqrt 5 / 5) * t.a * t.c

-- Helper function for area calculation
noncomputable def Area (t : Triangle) : ℝ :=
  1/2 * t.b * t.c * Real.sin t.A

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : satisfies_conditions t) : 
  Real.cos t.A = -(Real.sqrt 5 / 5) ∧
  (t.b = Real.sqrt 5 → Area t = 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l183_18312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_interval_l183_18348

-- Define sets A and B
def A : Set ℝ := {x | (1/9 : ℝ) ≤ Real.exp (Real.log 3 * x) ∧ Real.exp (Real.log 3 * x) ≤ 1}
def B : Set ℝ := {x | x^2 < 1}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- State the theorem
theorem intersection_equals_interval : A_intersect_B = Set.Ioc (-1 : ℝ) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_interval_l183_18348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_subgrid_exists_l183_18355

-- Define the grid type
def Grid := Fin 3 → Fin 7 → Bool

-- Define a subgrid
def Subgrid (g : Grid) (r c h w : Nat) : Fin h → Fin w → Bool :=
  fun i j => g ⟨r + i.val, by sorry⟩ ⟨c + j.val, by sorry⟩

-- Define the property of having uniform corners
def UniformCorners {h w : Nat} (sg : Fin h → Fin w → Bool) : Prop :=
  sg ⟨0, by sorry⟩ ⟨0, by sorry⟩ = sg ⟨0, by sorry⟩ ⟨w - 1, by sorry⟩ ∧
  sg ⟨0, by sorry⟩ ⟨0, by sorry⟩ = sg ⟨h - 1, by sorry⟩ ⟨0, by sorry⟩ ∧
  sg ⟨0, by sorry⟩ ⟨0, by sorry⟩ = sg ⟨h - 1, by sorry⟩ ⟨w - 1, by sorry⟩

-- The main theorem
theorem uniform_subgrid_exists (g : Grid) :
  ∃ m n r c, 2 ≤ m ∧ m ≤ 3 ∧ 2 ≤ n ∧ n ≤ 7 ∧
    UniformCorners (Subgrid g r c m n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_subgrid_exists_l183_18355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_is_negative_1_1_l183_18314

-- Define the points as functions of a
def point1 (a : ℝ) : ℝ × ℝ := (3 * a, a - 5)
def point2 : ℝ × ℝ := (5, -2)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem product_of_roots_is_negative_1_1 :
  ∃ a₁ a₂ : ℝ, a₁ ≠ a₂ ∧ 
  (distance (point1 a₁) point2 = 3 * Real.sqrt 5) ∧
  (distance (point1 a₂) point2 = 3 * Real.sqrt 5) ∧
  (a₁ * a₂ = -1.1) := by
  sorry

#check product_of_roots_is_negative_1_1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_is_negative_1_1_l183_18314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_x_is_correct_l183_18341

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 5) + Real.sin (x / 7)

noncomputable def deg_to_rad (x : ℝ) : ℝ := x * (Real.pi / 180)

def smallest_max_x : ℝ := 5850

theorem smallest_max_x_is_correct :
  -- f(x) reaches its maximum at smallest_max_x
  f (deg_to_rad smallest_max_x) = 2 ∧
  -- smallest_max_x is the smallest positive value
  ∀ y : ℝ, 0 < y ∧ y < smallest_max_x → f (deg_to_rad y) < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_x_is_correct_l183_18341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_matches_count_l183_18373

theorem game_matches_count (points_per_win : ℕ) (krishna_win_ratio : ℚ) (callum_total_points : ℕ) : ℕ := by
  let total_matches : ℕ := 8
  have h1 : points_per_win = 10 := by sorry
  have h2 : krishna_win_ratio = 3/4 := by sorry
  have h3 : callum_total_points = 20 := by sorry
  have h4 : total_matches * (1 - krishna_win_ratio) * points_per_win = callum_total_points := by sorry
  exact total_matches

#check game_matches_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_matches_count_l183_18373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_moving_point_l183_18300

/-- The trajectory of a point M(x,y) satisfying the given conditions -/
def trajectory (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1 ∧ x > 0

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The theorem stating the trajectory of point M -/
theorem trajectory_of_moving_point (x y : ℝ) :
  distance x y (-5) 0 - distance x y 5 0 = 8 →
  trajectory x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_moving_point_l183_18300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_figure_l183_18323

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | (floor p.1)^2 + (floor p.2)^2 = 13}

theorem area_of_figure : MeasureTheory.volume S = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_figure_l183_18323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l183_18380

noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + 3 * x + b

noncomputable def g (a b x : ℝ) : ℝ := |f a b x| - 2/3

theorem problem_solution :
  (∀ x ∈ Set.Icc (0 : ℝ) 3, f 2 0 x ∈ Set.Icc 0 (4/3)) ∧
  (∀ b : ℝ, (∃ s : Finset ℝ, s.card ≤ 4 ∧ ∀ x : ℝ, g a b x = 0 → x ∈ s) →
    a ∈ Set.Icc (-2 : ℝ) 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l183_18380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l183_18370

theorem equation_solutions (x : ℝ) (h : x ≠ 0) :
  (8 : ℝ)^(2/x) - (2 : ℝ)^((3*x+3)/x) + 12 = 0 ↔ x = (3 * Real.log 2) / Real.log 6 ∨ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l183_18370
