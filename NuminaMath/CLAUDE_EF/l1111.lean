import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_fraction_l1111_111157

/-- Represents a right circular cone containing two spheres -/
structure ConeTwoSpheres where
  r : ℝ  -- radius of the smaller sphere
  h : ℝ  -- height of the cone
  R : ℝ  -- radius of the cone's base

/-- Conditions for the cone and spheres -/
def valid_cone_spheres (c : ConeTwoSpheres) : Prop :=
  c.r > 0 ∧ c.h > 0 ∧ c.R > 0 ∧
  c.h = 8 * c.r ∧ c.R = 2 * Real.sqrt 2 * c.r

/-- Volume of the cone -/
noncomputable def cone_volume (c : ConeTwoSpheres) : ℝ :=
  (1 / 3) * Real.pi * c.R^2 * c.h

/-- Volume of the two spheres -/
noncomputable def spheres_volume (c : ConeTwoSpheres) : ℝ :=
  (4 / 3) * Real.pi * c.r^3 + (4 / 3) * Real.pi * (2 * c.r)^3

/-- Theorem: The fraction of the cone's volume not occupied by the two spheres is 7/16 -/
theorem unoccupied_volume_fraction (c : ConeTwoSpheres) 
  (h : valid_cone_spheres c) : 
  (cone_volume c - spheres_volume c) / cone_volume c = 7 / 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_fraction_l1111_111157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighteen_min_correct_answers_for_excellent_l1111_111188

/-- The minimum number of correct answers needed for an excellent rating -/
def min_correct_answers_for_excellent (total_questions : ℕ) 
  (correct_points incorrect_points excellent_threshold : ℤ) : ℕ :=
  let min_correct := 
    (excellent_threshold + total_questions * incorrect_points) / 
    (correct_points - incorrect_points)
  Int.ceil min_correct |>.toNat

/-- Proof that 18 is the minimum number of correct answers needed for an excellent rating -/
theorem eighteen_min_correct_answers_for_excellent :
  min_correct_answers_for_excellent 20 5 (-1) 85 = 18 := by
  sorry

#eval min_correct_answers_for_excellent 20 5 (-1) 85

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighteen_min_correct_answers_for_excellent_l1111_111188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_shift_l1111_111109

/-- Given a parabola y = 3x^2, shifting it up by 2 units and right by 3 units results in y = 3(x-3)^2 + 2 -/
theorem parabola_shift (x y : ℝ) : 
  (y = 3 * x^2) → 
  (fun x' y' => y' = 3 * x'^2 + 2) (x - 3) y ↔ 
  y = 3 * (x - 3)^2 + 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_shift_l1111_111109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bracelet_prices_and_max_purchase_l1111_111110

-- Define the prices of bracelets A and B
def price_A : ℕ → Prop := λ x ↦ 3 * x + 20 = 128 ∧ x + 40 = 76
def price_B : ℕ → Prop := λ y ↦ 108 + y = 128 ∧ 36 + 2 * y = 76

-- Define the constraint for the kindergarten purchase
def kindergarten_constraint : ℕ → Prop := λ m ↦ 
  36 * m + 20 * (100 - m) ≤ 2500 ∧ m ≤ 100

-- Theorem statement
theorem bracelet_prices_and_max_purchase :
  (∃ (a b : ℕ), price_A a ∧ price_B b) ∧
  (∃ (m : ℕ), kindergarten_constraint m ∧
    ∀ (n : ℕ), kindergarten_constraint n → n ≤ m) ∧
  (∀ (a : ℕ), price_A a → a = 36) ∧
  (∀ (b : ℕ), price_B b → b = 20) ∧
  (∃ (m : ℕ), kindergarten_constraint m ∧ m = 31) :=
by sorry

#check bracelet_prices_and_max_purchase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bracelet_prices_and_max_purchase_l1111_111110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_unique_property_l1111_111143

open Real

-- Define the interval (0, π/2)
def openInterval : Set ℝ := Set.Ioo 0 (π / 2)

-- Define the property of having a minimum positive period of π
def hasMinPeriodPi (f : ℝ → ℝ) : Prop :=
  ∃ (p : ℝ), p > 0 ∧ p = π ∧ ∀ (x : ℝ), f (x + p) = f x

-- Define the property of being increasing on the interval (0, π/2)
def isIncreasingOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ (x y : ℝ), x ∈ s → y ∈ s → x < y → f x < f y

-- State the theorem
theorem tan_unique_property :
  (hasMinPeriodPi sin ∧ isIncreasingOn sin openInterval) = False ∧
  (hasMinPeriodPi cos ∧ isIncreasingOn cos openInterval) = False ∧
  (hasMinPeriodPi tan ∧ isIncreasingOn tan openInterval) = True ∧
  (hasMinPeriodPi (fun x ↦ sin (x / 2)) ∧ isIncreasingOn (fun x ↦ sin (x / 2)) openInterval) = False :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_unique_property_l1111_111143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l1111_111141

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l1111_111141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_right_focus_to_line_l1111_111108

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := y = Real.sqrt 3 * x

/-- The right focus of the ellipse -/
def right_focus : ℝ × ℝ := (1, 0)

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ :=
  let (x₀, y₀) := p
  let A := -Real.sqrt 3
  let B := 1
  let C := 0
  (|A * x₀ + B * y₀ + C|) / Real.sqrt (A^2 + B^2)

theorem distance_right_focus_to_line :
  distance_point_to_line right_focus (λ x => Real.sqrt 3 * x) = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_right_focus_to_line_l1111_111108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteenth_term_is_63_l1111_111135

/-- A geometric sequence is defined by its common ratio and any term. -/
structure GeometricSequence where
  ratio : ℝ
  term : ℕ → ℝ
  seq_def : ∀ n : ℕ, term (n + 1) = term n * ratio

/-- The geometric sequence with 7th term = 7 and 10th term = 21 -/
noncomputable def specialSequence : GeometricSequence :=
  { ratio := Real.rpow (21 / 7) (1 / 3)
    term := λ n => 7 * Real.rpow (Real.rpow (21 / 7) (1 / 3)) (n - 7 : ℝ)
    seq_def := by sorry }

theorem thirteenth_term_is_63 :
  specialSequence.term 13 = 63 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteenth_term_is_63_l1111_111135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_count_l1111_111114

theorem infinite_solutions_count :
  ∃ (S : Finset ℤ), (Finset.card S = 3) ∧
    (∀ n : ℤ, (∃ (f : ℤ → ℤ × ℤ), Function.Injective f ∧
      (∀ k : ℤ, let (x, y) := f k; x * y^2 + y^2 - x - y = n)) ↔ n ∈ S) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_count_l1111_111114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_cells_outcome_l1111_111120

/-- Represents the possible outcomes of orange cells on the board -/
inductive OrangeOutcome
  | outcome1
  | outcome2

/-- The size of the board -/
def boardSize : Nat := 2022

/-- Function to count orange cells based on a coloring strategy -/
def count_orange (coloring : Fin boardSize → Fin boardSize → Bool) : Nat :=
  sorry

/-- Theorem stating the possible outcomes of orange cells -/
theorem orange_cells_outcome :
  ∀ (orange_count : Nat),
  (∃ (coloring : Fin boardSize → Fin boardSize → Bool),
   orange_count = count_orange coloring) →
  (orange_count = boardSize * (boardSize - 2) ∨
   orange_count = (boardSize - 1) * (boardSize - 2)) :=
by
  sorry

#check orange_cells_outcome

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_cells_outcome_l1111_111120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_F_zeros_l1111_111139

noncomputable def f (x : ℝ) := (1 + Real.tan x) * Real.sin (2 * x)

noncomputable def F (x : ℝ) := f x - 2

def domain_f : Set ℝ := {x | ∀ k : ℤ, x ≠ Real.pi / 2 + k * Real.pi}

theorem f_domain : domain_f = {x | f x ≠ 0} := by sorry

theorem F_zeros : ∀ x ∈ Set.Ioo 0 Real.pi, F x = 0 ↔ x = Real.pi / 4 ∨ x = Real.pi / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_F_zeros_l1111_111139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_swimmer_speed_third_swimmer_speed_with_conditions_l1111_111190

/-- Represents a swimmer in the race --/
structure Swimmer where
  speed : ℝ
  startDelay : ℝ

/-- Represents the swimming race scenario --/
structure SwimmingRace where
  laneLength : ℝ
  swimmers : Fin 3 → Swimmer
  equidistantTime : ℝ
  meetingDistance2 : ℝ
  meetingDistance1 : ℝ

/-- The main theorem stating the speed of the third swimmer --/
theorem third_swimmer_speed (race : SwimmingRace) : 
  (race.swimmers 2).speed = 22 / 15 := by
  sorry

/-- Conditions of the race --/
def race_conditions (race : SwimmingRace) : Prop :=
  race.laneLength = 50 ∧
  (race.swimmers 0).startDelay = 0 ∧
  (race.swimmers 1).startDelay = 5 ∧
  (race.swimmers 2).startDelay = 10 ∧
  race.meetingDistance2 = 4 ∧
  race.meetingDistance1 = 7 ∧
  ∃ t : ℝ, t < race.equidistantTime ∧ 
    (race.swimmers 0).speed * (t + 10) = 
    (race.swimmers 1).speed * (t + 5) ∧
    (race.swimmers 1).speed * (t + 5) = 
    (race.swimmers 2).speed * t

/-- The main theorem with conditions --/
theorem third_swimmer_speed_with_conditions (race : SwimmingRace) 
  (h : race_conditions race) : 
  (race.swimmers 2).speed = 22 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_swimmer_speed_third_swimmer_speed_with_conditions_l1111_111190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l1111_111103

/-- Predicate indicating that the terminal side of an angle is in the third quadrant -/
def AngleInThirdQuadrant (θ : Real) : Prop :=
  Real.sin θ < 0 ∧ Real.cos θ < 0

/-- 
Given an angle θ, if sin θ · cos θ > 0 and cos θ · tan θ < 0, 
then the terminal side of angle θ is in the third quadrant.
-/
theorem angle_in_third_quadrant (θ : Real) 
  (h1 : Real.sin θ * Real.cos θ > 0) 
  (h2 : Real.cos θ * Real.tan θ < 0) : 
  AngleInThirdQuadrant θ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l1111_111103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_location_point_P_location_zero_l1111_111182

-- Define the points A and P
noncomputable def A : ℝ × ℝ := (2, 0)
noncomputable def P (m : ℝ) : ℝ × ℝ := (m, 2*m - 1)

-- Define the angle POA
noncomputable def angle_POA (m : ℝ) : ℝ := Real.arctan ((2*m - 1) / m) - Real.arctan 0

-- Theorem statement
theorem point_P_location (m : ℝ) (h : m ≠ 0) :
  angle_POA m = π/4 → m = 1 ∨ m = 1/3 := by
  sorry

-- Additional lemma to handle the case when m = 0
theorem point_P_location_zero :
  angle_POA 0 ≠ π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_location_point_P_location_zero_l1111_111182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stick_puzzle_solvable_l1111_111116

/-- Represents an end of the stick -/
inductive StickEnd
| Left
| Right

/-- Represents the stick with a rope attached to one end -/
structure Stick :=
  (rope_end : StickEnd)

/-- Represents the state of the puzzle -/
structure PuzzleState :=
  (stick : Stick)
  (connected_to_jacket : Bool)

/-- Represents actions that can be taken -/
inductive PuzzleAction
| ConnectToJacket
| DisconnectFromJacket

/-- Applies an action to the current state -/
def apply_action (state : PuzzleState) (action : PuzzleAction) : PuzzleState :=
  match action with
  | PuzzleAction.ConnectToJacket => { state with connected_to_jacket := true }
  | PuzzleAction.DisconnectFromJacket => { state with connected_to_jacket := false }

/-- Theorem: It's possible to connect the stick to the jacket and then disconnect it -/
theorem stick_puzzle_solvable (initial_state : PuzzleState) :
  ∃ (intermediate_state final_state : PuzzleState),
    apply_action initial_state PuzzleAction.ConnectToJacket = intermediate_state ∧
    apply_action intermediate_state PuzzleAction.DisconnectFromJacket = final_state ∧
    final_state.connected_to_jacket = false := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stick_puzzle_solvable_l1111_111116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l1111_111118

noncomputable def a : ℕ → ℝ
  | 0 => Real.sqrt 2 / 2
  | n + 1 => Real.sqrt 2 / 2 * Real.sqrt (1 - Real.sqrt (1 - (a n)^2))

noncomputable def b : ℕ → ℝ
  | 0 => 1
  | n + 1 => (Real.sqrt (1 + (b n)^2) - 1) / (b n)

theorem sequence_inequality (n : ℕ) : 2^(n+2) * a n < Real.pi ∧ Real.pi < 2^(n+2) * b n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l1111_111118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_august_volume_l1111_111176

def honey_volumes : List Nat := [13, 15, 16, 17, 19, 21]

theorem largest_august_volume :
  let total_volume := honey_volumes.sum
  let august_sales := 2 * (total_volume / 3)
  let unsold_volume := honey_volumes.find? (λ v => v % 3 = total_volume % 3)
  let sold_volumes := honey_volumes.filter (λ v => v ≠ unsold_volume.getD 0)
  let august_volumes := sold_volumes.filter (λ v => v ∉ [13, 15])
  august_volumes.maximum? = some 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_august_volume_l1111_111176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_dots_is_75_l1111_111194

/-- Represents a cube with faces marked with dots -/
structure Cube where
  faces : Fin 6 → Nat
  sum_opposite_faces : ∀ i : Fin 3, faces i + faces (i + 3) = 7

/-- Represents the assembled figure of cubes -/
structure AssembledFigure where
  cubes : Fin 7 → Cube
  glued_faces_equal : ∀ i j : Fin 9, ∃ (f₁ f₂ : Fin 6), (cubes i).faces f₁ = (cubes j).faces f₂

/-- The theorem to be proved -/
def total_dots_on_figure (fig : AssembledFigure) : Nat :=
  75

/-- The main theorem stating that the total number of dots on the assembled figure is 75 -/
theorem total_dots_is_75 (fig : AssembledFigure) : 
  total_dots_on_figure fig = 75 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_dots_is_75_l1111_111194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_vertex_asymptote_l1111_111106

-- Define the parabola
def parabola (a : ℝ) (x y : ℝ) : Prop := x = a * y^2 ∧ a ≠ 0

-- Define the focus of a parabola
noncomputable def focus (a : ℝ) : ℝ × ℝ := (1 / (4 * a), 0)

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 12 - y^2 / 4 = 1

-- Define the vertex of the hyperbola
noncomputable def vertex : ℝ × ℝ := (Real.sqrt 12, 0)

-- Define the asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := y = -1 / Real.sqrt 3 * x

-- Theorem for the focus of the parabola
theorem parabola_focus (a : ℝ) (h : a ≠ 0) :
  parabola a (focus a).1 (focus a).2 := by
  sorry

-- Theorem for the distance from vertex to asymptote
theorem distance_vertex_asymptote :
  let d := Real.sqrt ((vertex.1 + Real.sqrt 3 * vertex.2)^2 / (1 + 3))
  d = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_vertex_asymptote_l1111_111106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lune_area_l1111_111198

/-- The area of a lune formed by two semicircles -/
theorem lune_area (d₁ d₂ : ℝ) (h₁ : d₁ = 3) (h₂ : d₂ = 4) : 
  (π * (d₁/2)^2 / 2) - (π * (d₂/2)^2 / 3) = (11/24) * π :=
by
  -- Substitute the values of d₁ and d₂
  rw [h₁, h₂]
  -- Simplify the expression
  ring_nf
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lune_area_l1111_111198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_degree_lower_bound_l1111_111105

theorem polynomial_degree_lower_bound
  (t : ℝ) (n : ℕ) (f : Polynomial ℝ)
  (h_t : t ≥ 3)
  (h_f : ∀ k : ℕ, k ≤ n → |f.eval (k : ℝ) - t^k| < 1) :
  Polynomial.degree f ≥ n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_degree_lower_bound_l1111_111105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l1111_111126

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2) / e.a

/-- A point on an ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The foci of an ellipse -/
noncomputable def foci (e : Ellipse) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := Real.sqrt (e.a^2 - e.b^2)
  ((-c, 0), (c, 0))

/-- Predicate for perpendicular lines from a point to the foci -/
def hasPerpFociLines (e : Ellipse) (p : PointOnEllipse e) : Prop :=
  let (f1, f2) := foci e
  (p.x - f1.1) * (p.x - f2.1) + (p.y - f1.2) * (p.y - f2.2) = 0

theorem ellipse_eccentricity_range (e : Ellipse) :
  (∃ p : PointOnEllipse e, hasPerpFociLines e p) →
  Real.sqrt 2 / 2 ≤ eccentricity e ∧ eccentricity e < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l1111_111126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conjugate_in_fourth_quadrant_l1111_111195

/-- Given a complex number z satisfying (z-3)(2-i) = 5i, 
    prove that its conjugate has a positive real part and a negative imaginary part -/
theorem conjugate_in_fourth_quadrant (z : ℂ) (h : (z - 3) * (2 - Complex.I) = 5 * Complex.I) :
  0 < z.re ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conjugate_in_fourth_quadrant_l1111_111195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equals_one_l1111_111192

-- Define the angle 30 degrees in radians
noncomputable def angle_30 : Real := 30 * (Real.pi / 180)

-- State the theorem
theorem trigonometric_expression_equals_one :
  let tan_30 := Real.tan angle_30
  let sin_30 := Real.sin angle_30
  (tan_30^2 - sin_30^2) / (tan_30^2 * sin_30^2) = 1 :=
by
  -- Assume the given conditions
  have h1 : Real.tan angle_30 = Real.sin angle_30 / Real.cos angle_30 := by sorry
  have h2 : Real.sin angle_30 = 1/2 := by sorry
  
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equals_one_l1111_111192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_dissection_l1111_111187

/-- Two sets of points in ℝ² are similar. -/
def Similar (P Q : Set (ℝ × ℝ)) : Prop :=
  ∃ (r : ℝ) (t : ℝ × ℝ), r > 0 ∧
    ∀ (p : ℝ × ℝ), p ∈ P ↔ (r • p + t) ∈ Q

/-- Two sets of points in ℝ² are congruent. -/
def Congruent (P Q : Set (ℝ × ℝ)) : Prop :=
  ∃ (t : ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ P ↔ (p + t) ∈ Q

/-- A rectangle with width 1 and height k can be dissected into two similar but incongruent polygons if and only if k > 0 and k ≠ 1. -/
theorem rectangle_dissection (k : ℝ) : 
  (∃ (P Q : Set (ℝ × ℝ)), 
    P ∪ Q = {(x, y) | 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ k} ∧
    P ∩ Q = ∅ ∧
    Similar P Q ∧
    ¬ Congruent P Q) ↔ 
  (k > 0 ∧ k ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_dissection_l1111_111187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_speed_is_20_l1111_111147

-- Define the problem parameters
noncomputable def distance : ℝ := 15
noncomputable def bus_delay : ℝ := 1/4
noncomputable def bus_speed_ratio : ℝ := 1.5

-- Define the bicycle speed
noncomputable def bicycle_speed : ℝ := 20

-- Theorem statement
theorem bicycle_speed_is_20 :
  let bus_speed := bus_speed_ratio * bicycle_speed
  let bicycle_time := distance / bicycle_speed
  let bus_time := distance / bus_speed
  bicycle_time - bus_time = bus_delay := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_speed_is_20_l1111_111147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_areas_sum_l1111_111111

-- Define the parabola
def Parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the focus of the parabola
def Focus : ℝ × ℝ := (1, 0)

-- Define the vector sum condition
def VectorSumZero (A B C : ℝ × ℝ) : Prop :=
  (A.1 - Focus.1, A.2 - Focus.2) + (B.1 - Focus.1, B.2 - Focus.2) + (C.1 - Focus.1, C.2 - Focus.2) = (0, 0)

-- Define the area of a triangle formed by the origin, focus, and a point
noncomputable def TriangleArea (p : ℝ × ℝ) : ℝ := |p.2| / 2

-- Main theorem
theorem parabola_triangle_areas_sum
  (A B C : ℝ × ℝ)
  (hA : Parabola A)
  (hB : Parabola B)
  (hC : Parabola C)
  (hABC : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (hSum : VectorSumZero A B C) :
  (TriangleArea A)^2 + (TriangleArea B)^2 + (TriangleArea C)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_areas_sum_l1111_111111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decision_box_is_diamond_l1111_111146

/-- Represents the shape of a flowchart element -/
inductive FlowchartShape
| Diamond
| Rectangle
| Oval
| Parallelogram

/-- Represents a decision box in a flowchart -/
def decision_box : FlowchartShape := FlowchartShape.Diamond

/-- Theorem stating that a decision box is diamond-shaped -/
theorem decision_box_is_diamond : decision_box = FlowchartShape.Diamond := by
  rfl

#check decision_box_is_diamond

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decision_box_is_diamond_l1111_111146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_line_l1111_111148

/-- Given a curve y = x^3 + 2x + 1 with a tangent at x = 1 perpendicular to ax - 2y - 3 = 0, prove a = -2/5 -/
theorem tangent_perpendicular_line (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x^3 + 2*x + 1) →
  (∃ k : ℝ, k * (((fun x ↦ 3*x^2 + 2) 1) * 1 + (-1)) = -1 ∧ k * (a * 1 + (-2)) = 1) →
  a = -2/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_line_l1111_111148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_theorem_l1111_111197

open Real MeasureTheory

/-- The length of the arc of the curve y = e^x + 13 from x = ln √15 to x = ln √24 -/
noncomputable def arcLength : ℝ := 1 + (1/2) * log (10/9)

/-- The function representing the curve -/
noncomputable def f (x : ℝ) : ℝ := exp x + 13

/-- The lower bound of the interval -/
noncomputable def a : ℝ := log (sqrt 15)

/-- The upper bound of the interval -/
noncomputable def b : ℝ := log (sqrt 24)

/-- Theorem stating that the arc length of the curve y = e^x + 13 
    from x = ln √15 to x = ln √24 is equal to 1 + (1/2) * ln(10/9) -/
theorem arc_length_theorem : 
  ∫ x in a..b, sqrt (1 + (deriv f x)^2) = arcLength := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_theorem_l1111_111197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_f_upper_bound_achievable_l1111_111173

/-- The function f as defined in the problem -/
noncomputable def f (A B x y : ℝ) : ℝ := min x (min (A / y) (y + B / x))

/-- The theorem stating that f(x,y) is always less than or equal to √(A + B) -/
theorem f_upper_bound (A B x y : ℝ) (hA : 0 < A) (hB : 0 < B) (hx : 0 < x) (hy : 0 < y) :
  f A B x y ≤ Real.sqrt (A + B) := by
  sorry

/-- The theorem stating that the upper bound √(A + B) is achievable -/
theorem f_upper_bound_achievable (A B : ℝ) (hA : 0 < A) (hB : 0 < B) :
  ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ f A B x y = Real.sqrt (A + B) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_f_upper_bound_achievable_l1111_111173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_functional_equation_l1111_111144

theorem polynomial_functional_equation (p : Polynomial ℝ) :
  (∀ x : ℝ, x ≠ 0 → p.eval x ^ 2 + p.eval (1 / x) ^ 2 = p.eval (x ^ 2) * p.eval (1 / x ^ 2)) →
  p = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_functional_equation_l1111_111144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_school_l1111_111171

-- Define the travel times in hours
noncomputable def rush_hour_time : ℝ := 15 / 60
noncomputable def minimal_traffic_time : ℝ := 10 / 60

-- Define the speed difference
def speed_difference : ℝ := 20

-- Define the distance function
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- Theorem statement
theorem distance_to_school : 
  ∃ (rush_hour_speed : ℝ),
    distance rush_hour_speed rush_hour_time = 
    distance (rush_hour_speed + speed_difference) minimal_traffic_time ∧
    distance rush_hour_speed rush_hour_time = 10 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_school_l1111_111171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_ratio_l1111_111161

def num_balls : Nat := 24
def num_bins : Nat := 6

def distribution_A : Fin num_bins → Nat
  | ⟨0, _⟩ => 3
  | ⟨1, _⟩ => 5
  | _ => 4

def distribution_B : Fin num_bins → Nat
  | _ => 4

noncomputable def count_distributions (d : Fin num_bins → Nat) : Nat :=
  sorry

theorem distribution_ratio :
  (count_distributions distribution_A) / (count_distributions distribution_B) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_ratio_l1111_111161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_slope_theorem_l1111_111136

-- We'll use noncomputable here as we're dealing with real numbers
noncomputable section

-- Define midpoint function
def my_midpoint (x₁ y₁ x₂ y₂ : ℝ) : ℝ × ℝ :=
  ((x₁ + x₂) / 2, (y₁ + y₂) / 2)

-- Define slope function
def my_slope (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  (y₂ - y₁) / (x₂ - x₁)

-- Theorem statement
theorem midpoint_slope_theorem :
  let m₁ := my_midpoint 3 7 7 8
  let m₂ := my_midpoint 10 3 15 13
  my_slope m₁.1 m₁.2 m₂.1 m₂.2 = 1 / 15 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_slope_theorem_l1111_111136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_in_equilateral_cone_l1111_111162

/-- A sphere inscribed in an equilateral cone such that it touches the cone's surface at the circumference of the base -/
structure SphereInCone where
  r : ℝ  -- radius of the sphere
  h : ℝ  -- height of the cone
  base_radius : ℝ  -- radius of the base of the cone

/-- The surface area of a sphere -/
noncomputable def sphere_surface_area (s : SphereInCone) : ℝ := 4 * Real.pi * s.r^2

/-- The surface area of the spherical cap inside the cone -/
noncomputable def spherical_cap_area (s : SphereInCone) : ℝ := Real.pi * s.r^2

/-- The fraction of the sphere's surface inside the cone -/
noncomputable def fraction_inside (s : SphereInCone) : ℝ :=
  spherical_cap_area s / sphere_surface_area s

/-- The fraction of the sphere's surface outside the cone -/
noncomputable def fraction_outside (s : SphereInCone) : ℝ :=
  1 - fraction_inside s

theorem sphere_in_equilateral_cone (s : SphereInCone) :
  fraction_inside s = 1/4 ∧ fraction_outside s = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_in_equilateral_cone_l1111_111162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l1111_111179

theorem cos_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos (α + π / 6) = 4 / 5) :
  Real.cos α = (4 * Real.sqrt 3 + 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l1111_111179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jacks_walking_speed_l1111_111158

theorem jacks_walking_speed 
  (initial_distance : ℝ) 
  (christina_speed : ℝ) 
  (lindy_speed : ℝ) 
  (lindy_distance : ℝ) 
  (h1 : initial_distance = 150)
  (h2 : christina_speed = 8)
  (h3 : lindy_speed = 10)
  (h4 : lindy_distance = 100) : 
  ∃ (jacks_speed : ℝ), jacks_speed = 7 := by
  sorry

#check jacks_walking_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jacks_walking_speed_l1111_111158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_condition_l1111_111115

theorem circle_intersection_condition (R r : ℝ) (h : R > r) :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (1, 0)
  let C : ℝ × ℝ := (1, 1)
  let D : ℝ × ℝ := (0, 1)
  let square_ABCD : Set (ℝ × ℝ) := {A, B, C, D}
  let circle1 : Set (ℝ × ℝ) := {(x, y) | (x - A.1)^2 + (y - A.2)^2 = R^2}
  let circle2 : Set (ℝ × ℝ) := {(x, y) | (x - C.1)^2 + (y - C.2)^2 = r^2}
  (∃ (P Q : ℝ × ℝ), P ≠ Q ∧ P ∈ circle1 ∩ circle2 ∧ Q ∈ circle1 ∩ circle2) ↔
  (R - r < Real.sqrt 2 ∧ Real.sqrt 2 < R + r) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_condition_l1111_111115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_type_b_books_cost_of_type_b_books_proof_l1111_111183

/-- Given a class purchasing books with the following conditions:
  - Total number of books to be purchased is 100
  - There are two types of books: A and B
  - The unit price of type A book is $10
  - The unit price of type B book is $6
  - The number of type A books purchased is represented by a
This theorem proves that the cost of purchasing type B books is $6(100-a)$ dollars. -/
theorem cost_of_type_b_books (a : ℕ) : ℕ → ℚ :=
  fun _ => 6 * (100 - a)

/-- The total number of books is 100 -/
def total_books : ℕ := 100

/-- The unit price of type A book is $10 -/
def price_a : ℚ := 10

/-- The unit price of type B book is $6 -/
def price_b : ℚ := 6

/-- The number of type A books purchased -/
def a : ℕ := 0  -- We define 'a' as a def instead of a variable

#check cost_of_type_b_books a total_books

/-- Proof that the cost of type B books is equal to the price of B multiplied by (total books - a) -/
theorem cost_of_type_b_books_proof : 
  cost_of_type_b_books a total_books = price_b * (total_books - a) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_type_b_books_cost_of_type_b_books_proof_l1111_111183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_theorem_l1111_111184

-- Define the exponential function
noncomputable def exp (x : ℝ) : ℝ := Real.exp x

-- Define the concept of a function being symmetric to another with respect to y-axis
def symmetric_to_y_axis (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (-x)

-- Define the translation of a function
def translate (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ :=
  λ x ↦ f (x - a)

-- Theorem statement
theorem function_symmetry_theorem (f : ℝ → ℝ) :
  symmetric_to_y_axis (translate f 1) exp →
  f = λ x ↦ exp (-x - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_theorem_l1111_111184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_team_exists_l1111_111186

/-- Represents a team assignment for a student over three classes -/
structure TeamAssignment where
  assign : Fin 3 → Fin 3

/-- The number of students -/
def numStudents : Nat := 30

/-- The number of teams -/
def numTeams : Nat := 3

/-- The number of students per team -/
def studentsPerTeam : Nat := 10

/-- A function that assigns teams to students for three consecutive classes -/
def assignTeams : Fin numStudents → TeamAssignment :=
  sorry

theorem same_team_exists :
  ∃ (s₁ s₂ : Fin numStudents), s₁ ≠ s₂ ∧ assignTeams s₁ = assignTeams s₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_team_exists_l1111_111186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_between_curves_l1111_111127

-- Define the ellipse C₁
def C₁ (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the line C₂
def C₂ (x y : ℝ) : Prop := x + y - 4 = 0

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem minimum_distance_between_curves :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    C₁ x₁ y₁ ∧ C₂ x₂ y₂ ∧
    (∀ (x₃ y₃ x₄ y₄ : ℝ), C₁ x₃ y₃ → C₂ x₄ y₄ →
      distance x₁ y₁ x₂ y₂ ≤ distance x₃ y₃ x₄ y₄) ∧
    distance x₁ y₁ x₂ y₂ = Real.sqrt 2 ∧
    x₁ = 3/2 ∧ y₁ = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_between_curves_l1111_111127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_divisibility_and_cube_property_l1111_111164

def a (n : ℕ) : ℕ := 4^(6^n) + 1943

theorem a_divisibility_and_cube_property :
  (∀ n : ℕ, n ≥ 1 → 2013 ∣ a n) ∧
  (∀ n : ℕ, n ≥ 1 → (∃ k : ℕ, k > 0 ∧ a n - 207 = k^3) ↔ n = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_divisibility_and_cube_property_l1111_111164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_union_is_29_l1111_111124

-- Define the vertices of the original triangle
def A : ℚ × ℚ := (2, 6)
def B : ℚ × ℚ := (5, -2)
def C : ℚ × ℚ := (7, 3)

-- Define the reflection line
def reflectionLine : ℚ := 2

-- Function to reflect a point about y = reflectionLine
def reflect (p : ℚ × ℚ) : ℚ × ℚ :=
  (p.1, 2 * reflectionLine - p.2)

-- Define the reflected vertices
def A' : ℚ × ℚ := reflect A
def B' : ℚ × ℚ := reflect B
def C' : ℚ × ℚ := reflect C

-- Function to calculate the area of a triangle given its vertices
noncomputable def triangleArea (p q r : ℚ × ℚ) : ℚ :=
  (1/2) * abs (p.1 * (q.2 - r.2) + q.1 * (r.2 - p.2) + r.1 * (p.2 - q.2))

-- Theorem statement
theorem area_of_union_is_29 :
  triangleArea A B C + triangleArea A' B' C' = 29 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_union_is_29_l1111_111124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_l1111_111152

/-- A tetrahedron with right angles at vertex P -/
structure RightTetrahedron where
  S : ℝ  -- Sum of all edge lengths
  h : S > 0

/-- The volume of a right tetrahedron -/
noncomputable def volume (t : RightTetrahedron) : ℝ :=
  t.S^3 / (162 * (1 + Real.sqrt 2)^3)

/-- The theorem stating that the volume function gives the maximum volume -/
theorem max_volume (t : RightTetrahedron) :
  ∀ V : ℝ, V ≤ volume t := by
  sorry

#check max_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_l1111_111152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_redesigned_survey_theorem_l1111_111131

/-- Calculates the number of customers sent the redesigned survey -/
def redesigned_survey_recipients (original_recipients : ℕ) (original_responses : ℕ) (redesigned_responses : ℕ) (rate_increase : ℚ) : ℕ :=
  let x := (redesigned_responses * original_recipients : ℚ) / (original_responses + rate_increase * original_recipients)
  Int.natAbs (Int.ceil x)

/-- Theorem stating that given the conditions, the number of customers sent the redesigned survey is approximately 66 -/
theorem redesigned_survey_theorem (original_recipients : ℕ) (original_responses : ℕ) (redesigned_responses : ℕ) (rate_increase : ℚ) :
  original_recipients = 60 →
  original_responses = 7 →
  redesigned_responses = 9 →
  rate_increase = 0.02 →
  redesigned_survey_recipients original_recipients original_responses redesigned_responses rate_increase = 66 :=
by
  sorry

#eval redesigned_survey_recipients 60 7 9 (2/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_redesigned_survey_theorem_l1111_111131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_N_l1111_111130

def is_valid_N (N : ℕ) : Prop :=
  (N * 375 % 1000 = 0) ∧
  (N * 125 % 1000 = 0) ∧
  (N * 250 % 1000 = 0)

theorem smallest_valid_N :
  ∃ N : ℕ, N > 0 ∧ is_valid_N N ∧ ∀ m : ℕ, m > 0 ∧ m < N → ¬is_valid_N m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_N_l1111_111130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tripleSum_value_l1111_111142

/-- The sum of 1 / (2^a * 5^b * 7^c) over all positive integer triples (a, b, c) where 1 ≤ a < b < c -/
noncomputable def tripleSum : ℝ :=
  ∑' (a : ℕ), ∑' (b : ℕ), ∑' (c : ℕ),
    if 1 ≤ a ∧ a < b ∧ b < c then (1 : ℝ) / (2^a * 5^b * 7^c) else 0

theorem tripleSum_value : tripleSum = 1 / 14244 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tripleSum_value_l1111_111142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_amount_approximation_l1111_111181

/-- The monthly interest payment in dollars -/
def monthly_payment : ℝ := 234

/-- The annual interest rate as a decimal -/
def annual_rate : ℝ := 0.09

/-- The number of times interest is compounded per year -/
def compounds_per_year : ℝ := 12

/-- The time period in years -/
def time_period : ℝ := 1

/-- The principal amount of the investment -/
noncomputable def principal : ℝ :=
  monthly_payment * (1 - (1 + annual_rate / compounds_per_year)^(-compounds_per_year * time_period)) / (annual_rate / compounds_per_year)

/-- Theorem stating that the calculated principal is approximately $2607.47 -/
theorem investment_amount_approximation (ε : ℝ) (h : ε > 0) :
  |principal - 2607.47| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_amount_approximation_l1111_111181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_museum_entry_ratio_l1111_111191

theorem museum_entry_ratio (adult_fee child_fee total_fee : ℚ) 
  (min_adults min_children : ℕ) : 
  adult_fee = 30 →
  child_fee = 15 →
  total_fee = 2250 →
  min_adults = 50 →
  min_children = 20 →
  ∃ (adults children : ℕ),
    adults ≥ min_adults ∧
    children ≥ min_children ∧
    adult_fee * (adults : ℚ) + child_fee * (children : ℚ) = total_fee ∧
    adults = children ∧
    ∀ (a c : ℕ),
      a ≥ min_adults →
      c ≥ min_children →
      adult_fee * (a : ℚ) + child_fee * (c : ℚ) = total_fee →
      |((a : ℚ) / (c : ℚ)) - 1| ≥ |((adults : ℚ) / (children : ℚ)) - 1| :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_museum_entry_ratio_l1111_111191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cadence_work_duration_l1111_111128

/-- Represents the number of years Cadence worked for her old company -/
def old_company_years : ℝ := sorry

/-- Represents Cadence's monthly salary at her old company -/
def old_salary : ℝ := 5000

/-- Represents the percentage increase in salary at her new company -/
def salary_increase : ℝ := 0.20

/-- Represents the total earnings from both companies -/
def total_earnings : ℝ := 426000

/-- Represents the number of months Cadence worked longer at her new company -/
def extra_months : ℝ := 5

theorem cadence_work_duration :
  old_company_years * 12 * old_salary +
  (old_company_years + extra_months / 12) * 12 * (old_salary * (1 + salary_increase)) =
  total_earnings →
  old_company_years = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cadence_work_duration_l1111_111128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_10_probability_l1111_111121

-- Define a segment of natural numbers
def Segment := List ℕ

-- Define a function to check if a number is divisible by 10
def isDivisibleBy10 (n : ℕ) : Bool := n % 10 = 0

-- Define the probability of choosing a number divisible by 10 from a segment
noncomputable def probabilityDivisibleBy10 (s : Segment) : ℚ :=
  (s.filter isDivisibleBy10).length / s.length

-- Theorem statement
theorem divisibility_by_10_probability :
  (∃ s : Segment, probabilityDivisibleBy10 s = 1) ∧
  (∀ s : Segment, s.length > 0 → probabilityDivisibleBy10 s ≠ 0 → probabilityDivisibleBy10 s ≥ 1/19) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_10_probability_l1111_111121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_black_count_with_white_rectangle_l1111_111104

/-- Represents a 9x9 grid where each cell can be either white or black -/
def Grid := Fin 9 → Fin 9 → Bool

/-- Checks if there exists a 1x4 white rectangle in the grid -/
def has_white_rectangle (g : Grid) : Prop :=
  ∃ (i j : Fin 9), 
    (∀ k : Fin 4, g i (j + k) = false) ∨ 
    (∀ k : Fin 4, g (i + k) j = false)

/-- Counts the number of black squares in the grid -/
def black_count (g : Grid) : ℕ :=
  (Finset.sum Finset.univ fun i => Finset.sum Finset.univ fun j => if g i j then 1 else 0)

/-- The main theorem stating that 19 is the largest number of black squares
    that always guarantees a 1x4 white rectangle -/
theorem largest_black_count_with_white_rectangle : 
  (∀ g : Grid, black_count g ≤ 19 → has_white_rectangle g) ∧
  (∃ g : Grid, black_count g = 20 ∧ ¬has_white_rectangle g) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_black_count_with_white_rectangle_l1111_111104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_plus_b_positive_iff_f_a_plus_f_b_positive_l1111_111112

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 * (Real.exp x - Real.exp (-x))

-- State the theorem
theorem a_plus_b_positive_iff_f_a_plus_f_b_positive (a b : ℝ) :
  (a + b > 0) ↔ (f a + f b > 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_plus_b_positive_iff_f_a_plus_f_b_positive_l1111_111112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_equation_l1111_111154

theorem count_integers_satisfying_equation : 
  ∃ (S : Finset ℤ), (∀ n : ℤ, n ∈ S ↔ 2 + ⌊(200 * n : ℚ) / 201⌋ = ⌈(198 * n : ℚ) / 200⌉) ∧ Finset.card S = 40200 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_equation_l1111_111154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l1111_111137

theorem repeating_decimal_sum (a b : ℕ) : 
  (5 : ℚ) / 13 = 0.1 * (10 * a + b) * (∑' k, (1 / 100) ^ k) → a + b = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l1111_111137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_decrease_percentage_l1111_111125

/-- Calculates the percentage decrease between two values -/
noncomputable def percentageDecrease (oldValue newValue : ℝ) : ℝ :=
  ((oldValue - newValue) / oldValue) * 100

/-- The old revenue in billions of dollars -/
def oldRevenue : ℝ := 69.0

/-- The new revenue in billions of dollars -/
def newRevenue : ℝ := 52.0

/-- Theorem stating that the percentage decrease in revenue is approximately 24.64% -/
theorem revenue_decrease_percentage :
  abs (percentageDecrease oldRevenue newRevenue - 24.64) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_decrease_percentage_l1111_111125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l1111_111149

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := 
  (Real.sin x ^ 3 + 7 * Real.sin x ^ 2 + Real.cos x * Real.sin x + 3 * Real.cos x ^ 2 - 11) / (Real.sin x - 1)

-- State the theorem
theorem g_range : 
  Set.range (fun x => g x) = Set.Ici (Real.sqrt 2 - 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l1111_111149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1111_111163

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (3 - x) / (x + 2)

-- Define a predicate for the domain of f
def IsValidArg (f : ℝ → ℝ) (x : ℝ) : Prop :=
  (3 - x ≥ 0) ∧ (x + 2 ≠ 0)

-- State the theorem about the domain of f
theorem domain_of_f : 
  {x : ℝ | IsValidArg f x} = {x : ℝ | x ≤ 3 ∧ x ≠ -2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1111_111163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2013_deg_l1111_111140

theorem sin_2013_deg : Real.sin (2013 * π / 180) = -Real.sin (33 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2013_deg_l1111_111140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_negative_30_degrees_l1111_111159

theorem sin_cos_negative_30_degrees :
  Real.sin (-(30 * Real.pi / 180)) = -1/2 ∧ 
  Real.cos (-(30 * Real.pi / 180)) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_negative_30_degrees_l1111_111159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_difference_l1111_111132

noncomputable def f (x : ℝ) : ℝ := 5 * x - 7

noncomputable def g (x : ℝ) : ℝ := x / 5 + 3

theorem function_composition_difference :
  ∀ x : ℝ, (f (g x)) - (g (f x)) = 32 / 5 := by
  intro x
  -- Expand the definitions of f and g
  unfold f g
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_difference_l1111_111132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_division_point_l1111_111102

noncomputable def divide_segment (z₁ z₂ : ℂ) (m n : ℝ) : ℂ :=
  (m * z₂ + n * z₁) / (m + n)

theorem segment_division_point :
  let z₁ : ℂ := 2 - 5*I
  let z₂ : ℂ := -6 + 2*I
  let m : ℝ := 1
  let n : ℝ := 3
  divide_segment z₁ z₂ m n = -1 - 13/4*I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_division_point_l1111_111102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_polynomial_identity_l1111_111119

variable (x y z : ℝ)

-- Define elementary symmetric polynomials
def σ₁ (x y z : ℝ) : ℝ := x + y + z
def σ₂ (x y z : ℝ) : ℝ := x*y + y*z + z*x
def σ₃ (x y z : ℝ) : ℝ := x*y*z

-- Theorem statement
theorem symmetric_polynomial_identity (x y z : ℝ) :
  x^3 + y^3 + z^3 - 3*x*y*z = σ₁ x y z * (σ₁ x y z ^2 - 3*σ₂ x y z) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_polynomial_identity_l1111_111119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifthTermSimplification_l1111_111123

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := 
  Nat.choose n k

-- Define the fifth term of the expansion
noncomputable def fifthTerm (a x : ℝ) : ℝ :=
  (binomial 7 4 : ℝ) * (a / x^2)^(7 - 4) * (-x / a^3)^4

-- Theorem statement
theorem fifthTermSimplification (a x : ℝ) (ha : a ≠ 0) (hx : x ≠ 0) :
  fifthTerm a x = 35 / (x^2 * a^9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifthTermSimplification_l1111_111123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_functions_with_pi_period_l1111_111151

noncomputable def f₁ (x : ℝ) := Real.cos (abs (2 * x))
noncomputable def f₂ (x : ℝ) := abs (Real.cos x)
noncomputable def f₃ (x : ℝ) := abs (Real.sin (2 * x + Real.pi / 2))
noncomputable def f₄ (x : ℝ) := Real.tan (abs x)

def is_even (f : ℝ → ℝ) := ∀ x, f (-x) = f x

def has_period (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) :=
  has_period f p ∧ p > 0 ∧ ∀ q, 0 < q ∧ q < p → ¬(has_period f q)

theorem even_functions_with_pi_period :
  (is_even f₁ ∧ smallest_positive_period f₁ Real.pi) ∧
  (is_even f₂ ∧ smallest_positive_period f₂ Real.pi) ∧
  ¬(is_even f₃ ∧ smallest_positive_period f₃ Real.pi) ∧
  ¬(is_even f₄ ∧ smallest_positive_period f₄ Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_functions_with_pi_period_l1111_111151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_1_expression_2_l1111_111138

-- Define α as a real number
variable (α : ℝ)

-- Define the conditions
noncomputable def tan_alpha : ℝ := 3
def second_quadrant (α : ℝ) : Prop := 0 < α ∧ α < Real.pi ∧ Real.sin α > 0 ∧ Real.cos α < 0

-- Theorem 1
theorem expression_1 (h1 : Real.tan α = tan_alpha) (h2 : second_quadrant α) :
  (Real.sin α - 2 * Real.cos α) / (Real.sin α + Real.cos α) = 1 / 4 := by sorry

-- Theorem 2
theorem expression_2 (h1 : Real.tan α = tan_alpha) (h2 : second_quadrant α) :
  Real.cos α * Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) + 
  Real.sin α * Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) = Real.sin α - Real.cos α := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_1_expression_2_l1111_111138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_condition_l1111_111180

/-- The line equation -/
def line (x y : ℝ) : Prop := 2 * x - y = 1

/-- The circle equation -/
def circle_eq (x y a : ℝ) : Prop := x^2 + y^2 + 2 * a * x - 2 * y + 3 = 0

/-- The condition on parameter a -/
def condition (a : ℝ) : Prop := 4 - Real.sqrt 30 < a ∧ a < 4 + Real.sqrt 30

/-- The tangency property -/
def is_tangent (a : ℝ) : Prop := ∃ x y : ℝ, line x y ∧ circle_eq x y a ∧
  ∀ x' y' : ℝ, line x' y' ∧ circle_eq x' y' a → (x', y') = (x, y)

/-- The main theorem -/
theorem tangency_condition (a : ℝ) :
  (is_tangent a → condition a) ∧ ¬(condition a → is_tangent a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_condition_l1111_111180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_region_l1111_111193

-- Define the region
def Region : Set (Fin 3 → ℝ) := 
  {p | |p 0 + 2*p 1 + p 2| + |p 0 + 2*p 1 - p 2| ≤ 12 ∧ 
       p 0 ≥ 0 ∧ p 1 ≥ 0 ∧ p 2 ≥ 0}

-- State the theorem
theorem volume_of_region : MeasureTheory.volume Region = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_region_l1111_111193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_divisible_by_11_l1111_111145

/-- Sequence a_n defined recursively -/
def a : ℕ → ℤ
  | 0 => 1  -- Adding this case to handle n = 0
  | 1 => 1
  | 2 => 3
  | (n + 3) => (n + 4) * a (n + 2) - (n + 3) * a (n + 1)

/-- Predicate for a_n being divisible by 11 -/
def is_divisible_by_11 (n : ℕ) : Prop := 11 ∣ a n

/-- Theorem stating the conditions for a_n to be divisible by 11 -/
theorem a_divisible_by_11 (n : ℕ) : 
  is_divisible_by_11 n ↔ n = 4 ∨ n = 8 ∨ n ≥ 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_divisible_by_11_l1111_111145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_vertical_asymptote_l1111_111178

/-- A rational function with a quadratic numerator and denominator -/
noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (x^2 + 2*x + c) / (x^2 - 5*x + 6)

/-- The property of having exactly one vertical asymptote -/
def has_exactly_one_vertical_asymptote (c : ℝ) : Prop :=
  (∃! x : ℝ, (x^2 - 5*x + 6 = 0) ∧ (x^2 + 2*x + c ≠ 0))

/-- Theorem stating the conditions for f to have exactly one vertical asymptote -/
theorem f_has_one_vertical_asymptote :
  ∀ c : ℝ, has_exactly_one_vertical_asymptote c ↔ (c = -8 ∨ c = -15) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_vertical_asymptote_l1111_111178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_participants_2005_l1111_111175

/-- Calculates the number of participants in a health awareness marathon over years --/
def marathon_participants (initial : ℕ) (growth_rate : ℚ) (additional : ℕ) (years : ℕ) : ℕ :=
  if years ≤ 2 then
    ((initial : ℚ) * (1 + growth_rate) ^ (years - 1)).floor.toNat
  else if years ≤ 3 then
    ((initial : ℚ) * (1 + growth_rate) ^ (years - 1)).floor.toNat
  else
    (((initial : ℚ) * (1 + growth_rate) ^ 3 * (1 + growth_rate) ^ (years - 3)).floor.toNat + additional * (years - 3))

/-- The number of participants in 2005 is 17500 --/
theorem participants_2005 :
  marathon_participants 1000 1 500 5 = 17500 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_participants_2005_l1111_111175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_naive_number_with_divisibility_l1111_111150

/-- Represents a four-digit natural number -/
structure FourDigitNumber where
  value : Nat
  is_four_digit : value ≥ 1000 ∧ value < 10000

/-- Extracts the thousands digit from a four-digit number -/
def thousands (n : FourDigitNumber) : Nat :=
  n.value / 1000

/-- Extracts the hundreds digit from a four-digit number -/
def hundreds (n : FourDigitNumber) : Nat :=
  (n.value / 100) % 10

/-- Extracts the tens digit from a four-digit number -/
def tens (n : FourDigitNumber) : Nat :=
  (n.value / 10) % 10

/-- Extracts the units digit from a four-digit number -/
def units (n : FourDigitNumber) : Nat :=
  n.value % 10

/-- Defines a "naive number" -/
def is_naive (n : FourDigitNumber) : Prop :=
  thousands n = units n + 6 ∧ hundreds n = tens n + 2

/-- Defines P(M) -/
def P (n : FourDigitNumber) : Nat :=
  3 * (thousands n + hundreds n) + tens n + units n

/-- Defines Q(M) -/
def Q (n : FourDigitNumber) : Int :=
  thousands n - 5

/-- Theorem statement -/
theorem max_naive_number_with_divisibility : 
  ∃ (M : FourDigitNumber), 
    is_naive M ∧ 
    (P M : Int) % (Q M) = 0 ∧ 
    (∀ (N : FourDigitNumber), is_naive N ∧ (P N : Int) % (Q N) = 0 → N.value ≤ M.value) ∧
    M.value = 9313 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_naive_number_with_divisibility_l1111_111150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_three_equals_eleven_l1111_111167

-- Define the function f
def f (t : ℝ) : ℝ := t^2 + 2

-- State the theorem
theorem f_of_three_equals_eleven :
  (∀ x : ℝ, x ≠ 0 → f (x - 1/x) = x^2 + 1/x^2) →
  f 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_three_equals_eleven_l1111_111167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l1111_111101

theorem log_inequality : 
  let a := Real.log π / Real.log 3
  let b := Real.log (Real.sqrt 3) / Real.log 2
  let c := Real.log (Real.sqrt 2) / Real.log 3
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l1111_111101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_interior_angle_l1111_111134

/-- The number of sides in a hexagon -/
def n : ℕ := 6

/-- The sum of interior angles of a polygon with n sides -/
noncomputable def sumInteriorAngles (n : ℕ) : ℝ := (n - 2) * 180

/-- The measure of each interior angle in a regular polygon with n sides -/
noncomputable def interiorAngle (n : ℕ) : ℝ := sumInteriorAngles n / n

/-- Theorem: The measure of an interior angle of a regular hexagon is 120 degrees -/
theorem regular_hexagon_interior_angle : interiorAngle n = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_interior_angle_l1111_111134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l1111_111165

/-- Given ω > 0 and the graph of y = sin(ωx + π/3) coincides with the original
    graph after shifting to the right by 4π/3 units, the minimum value of ω is 3/2. -/
theorem min_omega_value (ω : ℝ) (h1 : ω > 0)
  (h2 : ∀ x : ℝ, Real.sin (ω * x + π/3) = Real.sin (ω * (x - 4*π/3) + π/3)) :
  ∀ ω' : ℝ, ω' > 0 → ω' ≥ 3/2 → ω = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l1111_111165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_B4E1_to_dec_l1111_111153

/-- Converts a single hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | '0' => 0 | '1' => 1 | '2' => 2 | '3' => 3 | '4' => 4
  | '5' => 5 | '6' => 6 | '7' => 7 | '8' => 8 | '9' => 9
  | 'A' => 10 | 'B' => 11 | 'C' => 12 | 'D' => 13 | 'E' => 14 | 'F' => 15
  | _ => 0

/-- Converts a hexadecimal string to its decimal value -/
def hex_string_to_dec (s : String) : ℕ :=
  s.data.reverse.enum.foldl (fun acc (i, c) => acc + (hex_to_dec c) * (16 ^ i)) 0

/-- Theorem stating that B4E1 in hexadecimal is equal to 46305 in decimal -/
theorem hex_B4E1_to_dec :
  hex_string_to_dec "B4E1" = 46305 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_B4E1_to_dec_l1111_111153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_theorem_l1111_111174

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x + 4 / x
noncomputable def g (x a : ℝ) : ℝ := 2^x + a

-- Define the domains for x1 and x2
def x1_domain : Set ℝ := Set.Icc (1/2) 3
def x2_domain : Set ℝ := Set.Icc 2 3

-- State the theorem
theorem a_range_theorem (a : ℝ) :
  (∀ x1 ∈ x1_domain, ∃ x2 ∈ x2_domain, f x1 ≥ g x2 a) ↔ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_theorem_l1111_111174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_cosine_sum_l1111_111156

theorem min_value_of_cosine_sum (x y z : ℝ) (hx : x ∈ Set.Icc 0 Real.pi) (hy : y ∈ Set.Icc 0 Real.pi) (hz : z ∈ Set.Icc 0 Real.pi) :
  Real.cos (x - y) + Real.cos (y - z) + Real.cos (z - x) ≥ -1 ∧
  ∃ (a b c : ℝ), a ∈ Set.Icc 0 Real.pi ∧ b ∈ Set.Icc 0 Real.pi ∧ c ∈ Set.Icc 0 Real.pi ∧
    Real.cos (a - b) + Real.cos (b - c) + Real.cos (c - a) = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_cosine_sum_l1111_111156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_population_in_seven_years_l1111_111100

/-- The number of years it takes for the populations of Village X and Village Y to be equal -/
def years_to_equal_population : ℕ :=
  let village_x_initial : ℕ := 68000
  let village_x_natural_decrease : ℕ := 1200
  let village_x_birth_rate : ℚ := 18 / 1000
  let village_x_death_rate : ℚ := 22 / 1000
  let village_x_migration_rate : ℚ := -3 / 1000

  let village_y_initial : ℕ := 42000
  let village_y_natural_increase : ℕ := 800
  let village_y_birth_rate : ℚ := 16 / 1000
  let village_y_death_rate : ℚ := 13 / 1000
  let village_y_migration_rate : ℚ := 6 / 1000

  let village_x_total_change : ℚ := 
    village_x_birth_rate * village_x_initial - 
    village_x_death_rate * village_x_initial - 
    village_x_natural_decrease + 
    village_x_migration_rate * village_x_initial

  let village_y_total_change : ℚ := 
    village_y_birth_rate * village_y_initial - 
    village_y_death_rate * village_y_initial + 
    village_y_natural_increase + 
    village_y_migration_rate * village_y_initial

  let years : ℚ := (village_x_initial - village_y_initial) / 
                   (village_y_total_change - village_x_total_change)

  (years.ceil.toNat)

theorem equal_population_in_seven_years : 
  years_to_equal_population = 7 :=
by sorry

#eval years_to_equal_population

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_population_in_seven_years_l1111_111100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_diagonal_ratio_proof_l1111_111129

/-- The ratio of the shorter diagonal to the longer diagonal in a regular octagon -/
noncomputable def regular_octagon_diagonal_ratio : ℝ := Real.sqrt (2 - Real.sqrt 2) / 2

/-- Theorem stating that the ratio of the shorter diagonal to the longer diagonal
    in a regular octagon is equal to √(2 - √2) / 2 -/
theorem regular_octagon_diagonal_ratio_proof :
  ∃ (short_diagonal long_diagonal : ℝ),
    short_diagonal > 0 ∧
    long_diagonal > 0 ∧
    short_diagonal < long_diagonal ∧
    short_diagonal / long_diagonal = regular_octagon_diagonal_ratio :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_diagonal_ratio_proof_l1111_111129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_m_inclination_angle_l1111_111122

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (l1 l2 : Line) : ℝ :=
  abs (l1.c - l2.c) / Real.sqrt (l1.a^2 + l1.b^2)

/-- The inclination angle of a line -/
noncomputable def inclination_angle (l : Line) : ℝ :=
  Real.arctan (l.a / l.b)

theorem line_m_inclination_angle 
  (m : Line)
  (l1 : Line)
  (l2 : Line)
  (h1 : l1.a = 1 ∧ l1.b = -1 ∧ l1.c = 1)
  (h2 : l2.a = 1 ∧ l2.b = -1 ∧ l2.c = 3)
  (h3 : distance_between_parallel_lines l1 l2 = Real.sqrt 2)
  (h4 : ∃ (p q : ℝ × ℝ), 
        (m.a * p.1 + m.b * p.2 + m.c = 0) ∧
        (m.a * q.1 + m.b * q.2 + m.c = 0) ∧
        p.1 - p.2 + 1 = 0 ∧ q.1 - q.2 + 3 = 0 ∧
        Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 2 * Real.sqrt 2) :
  inclination_angle m = π/12 ∨ inclination_angle m = 5*π/12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_m_inclination_angle_l1111_111122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_one_l1111_111107

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x + 2 * x + 1

-- Define the derivative of the curve
noncomputable def f' (x : ℝ) : ℝ := Real.exp x + x * Real.exp x + 2

-- Theorem statement
theorem tangent_line_at_zero_one :
  let x₀ : ℝ := 0
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  (λ x => m * (x - x₀) + y₀) = (λ x => 3 * x + 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_one_l1111_111107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangent_line_slope_l1111_111199

/-- The parabola y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The circle (x-4)^2 + y^2 = 4 -/
def myCircle (x y : ℝ) : Prop := (x-4)^2 + y^2 = 4

/-- The focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- A line passing through a point (x₀, y₀) with slope k -/
def line (x₀ y₀ k x y : ℝ) : Prop := y - y₀ = k * (x - x₀)

/-- A line is tangent to a circle if the distance from the center of the circle to the line equals the radius -/
def is_tangent (x₀ y₀ k : ℝ) : Prop := 
  let center := (4, 0)
  let radius := 2
  |k * (center.1 - x₀) - (center.2 - y₀)| / Real.sqrt (k^2 + 1) = radius

theorem parabola_circle_tangent_line_slope :
  ∀ k : ℝ, 
  (∃ x y : ℝ, parabola x y ∧ myCircle x y ∧ 
    line focus.1 focus.2 k x y ∧
    is_tangent focus.1 focus.2 k) →
  k = 2 * Real.sqrt 5 / 5 ∨ k = -2 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangent_line_slope_l1111_111199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_expansion_l1111_111172

noncomputable def volume_cylinder (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem cylinder_volume_expansion (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  volume_cylinder (2 * r) (2 * h) = 8 * volume_cylinder r h :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_expansion_l1111_111172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l1111_111168

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of f being invertible
def IsInvertible (f : ℝ → ℝ) : Prop :=
  ∀ y, ∃! x, f x = y

-- Theorem statement
theorem intersection_points_count (h : IsInvertible f) :
  (∃! x₁ x₂ x₃, (f (x₁^2) = f (x₁^6)) ∧ (f (x₂^2) = f (x₂^6)) ∧ (f (x₃^2) = f (x₃^6)) ∧
    (∀ x, f (x^2) = f (x^6) → x = x₁ ∨ x = x₂ ∨ x = x₃)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l1111_111168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_tangent_line_equation_l1111_111185

-- Define the curve
def f (x : ℝ) : ℝ := -x^3 + 3*x^2

-- Define the point of tangency
def point : ℝ × ℝ := (1, 2)

-- Define the slope of the tangent line
def tangent_slope : ℝ := 3

-- Define the equation of the tangent line
def tangent_line (x : ℝ) : ℝ := tangent_slope * x - 1

-- Theorem statement
theorem tangent_line_at_point :
  (∀ x, tangent_line x = tangent_slope * (x - point.1) + point.2) ∧
  (tangent_line point.1 = point.2) ∧
  (∃ ε > 0, ∀ h ≠ 0, |h| < ε →
    |(f (point.1 + h) - f point.1) / h - tangent_slope| < |h|) :=
sorry

-- Proof that the tangent line equation matches the given answer
theorem tangent_line_equation :
  ∀ x, tangent_line x = 3*x - 1 :=
by
  intro x
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_tangent_line_equation_l1111_111185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l1111_111155

-- Define the ellipse G
def G (x y : ℝ) : Prop := x^2 / 12 + y^2 / 4 = 1

-- Define point M
noncomputable def M : ℝ × ℝ := (Real.sqrt 6, Real.sqrt 2)

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 6 / 3

-- Define line l
def l (x y : ℝ) : Prop := y = x + 2

-- Define point P
def P : ℝ × ℝ := (-3, 2)

-- State the theorem
theorem ellipse_triangle_area :
  G M.1 M.2 →  -- M is on ellipse G
  (∃ A B : ℝ × ℝ, 
    G A.1 A.2 ∧ G B.1 B.2 ∧  -- A and B are on ellipse G
    l A.1 A.2 ∧ l B.1 B.2 ∧  -- A and B are on line l
    (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2) →  -- PAB is isosceles
  ∃ S : ℝ, S = 9/2 ∧ 
    (∀ A B : ℝ × ℝ, G A.1 A.2 → G B.1 B.2 → l A.1 A.2 → l B.1 B.2 →
      (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2 →
      S = 1/2 * ((B.1 - A.1)^2 + (B.2 - A.2)^2)^(1/2) * 
          |((A.2 - P.2) * (B.1 - A.1) - (A.1 - P.1) * (B.2 - A.2)) / 
           ((B.1 - A.1)^2 + (B.2 - A.2)^2)^(1/2)|) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l1111_111155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equal_cost_transfer_l1111_111169

/-- Represents a city in the network -/
structure City where
  id : Nat

/-- Represents an airline in the network -/
structure Airline where
  id : Nat

/-- Represents the network of cities and airlines -/
structure Network where
  cities : Finset City
  airlines : Finset Airline
  connections : City → City → Finset Airline
  num_cities : cities.card = 50
  num_airlines : airlines.card = 71
  all_connected : ∀ c1 c2 : City, c1 ∈ cities → c2 ∈ cities → c1 ≠ c2 → (connections c1 c2).Nonempty
  symmetric : ∀ c1 c2 : City, connections c1 c2 = connections c2 c1

/-- The cost of a flight between two cities -/
def flight_cost (n : Network) (c1 c2 : City) : ℚ :=
  1 / (n.connections c1 c2).card

/-- No indirect route is cheaper than a direct flight -/
axiom no_cheaper_indirect (n : Network) :
  ∀ c1 c2 c3 : City, c1 ∈ n.cities → c2 ∈ n.cities → c3 ∈ n.cities →
    c1 ≠ c2 → c2 ≠ c3 → c1 ≠ c3 →
    flight_cost n c1 c3 ≤ flight_cost n c1 c2 + flight_cost n c2 c3

/-- The main theorem to be proved -/
theorem exists_equal_cost_transfer (n : Network) :
  ∃ c1 c2 c3 : City, c1 ∈ n.cities ∧ c2 ∈ n.cities ∧ c3 ∈ n.cities ∧
    c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
    flight_cost n c1 c2 = flight_cost n c2 c3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equal_cost_transfer_l1111_111169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l1111_111160

theorem constant_term_binomial_expansion (n : ℕ) :
  let P := (4 : ℝ) ^ n  -- Sum of coefficients when x = 1
  let Q := (2 : ℝ) ^ n  -- Sum of binomial coefficients
  P + Q = 272 →
  (∃ k : ℕ, (n.choose k) * 3^(n-k) * ((-1 : ℤ)^k : ℝ) = 108) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l1111_111160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_c_value_l1111_111196

-- Define the system of equations
def system (a b c x y : ℤ) : Prop :=
  (2 * x + y = 2037) ∧ (y = |x - a| + |x - b| + |x - c|)

-- Define the condition for exactly one solution
def unique_solution (a b c : ℤ) : Prop :=
  ∃! (x y : ℤ), system a b c x y

-- Main theorem
theorem min_c_value (a b c : ℤ) 
  (h1 : a < b) (h2 : b < c) (h3 : unique_solution a b c) :
  c ≥ 1019 ∧ ∃ (a' b' : ℤ), a' < b' ∧ b' < 1019 ∧ unique_solution a' b' 1019 :=
by
  sorry

#check min_c_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_c_value_l1111_111196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_for_given_resistance_l1111_111166

-- Define the voltage as a constant
def voltage : ℚ := 48

-- Define the relationship between current and resistance
noncomputable def current (resistance : ℚ) : ℚ := voltage / resistance

-- State the theorem
theorem current_for_given_resistance :
  current 12 = 4 := by
  -- Unfold the definition of current
  unfold current
  -- Simplify the fraction
  simp [voltage]
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_for_given_resistance_l1111_111166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l1111_111133

/-- Given a train of length 300 m that crosses a platform in 48 seconds and a signal pole in 18 seconds, the length of the platform is 500.16 m. -/
theorem platform_length (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) 
  (h1 : train_length = 300)
  (h2 : time_platform = 48)
  (h3 : time_pole = 18) :
  let speed := train_length / time_pole
  let platform_length := speed * time_platform - train_length
  platform_length = 500.16 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l1111_111133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_f_inequality_l1111_111170

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Theorem stating the maximum and minimum values of f(x) on [1/3, 3]
theorem f_extrema :
  (∀ x ∈ Set.Icc (1/3 : ℝ) 3, f x ≤ f 3) ∧
  (∀ x ∈ Set.Icc (1/3 : ℝ) 3, f (Real.exp (-1)) ≤ f x) ∧
  f 3 = 3 * Real.log 3 ∧
  f (Real.exp (-1)) = -1 / Real.exp 1 := by
  sorry

-- Theorem for the inequality
theorem f_inequality (x : ℝ) (h : x > 0) :
  f x - (x + 1)^2 ≤ -3*x - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_f_inequality_l1111_111170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_quadratics_with_one_solution_l1111_111117

-- Define the form of numbers as 2^m * 5^n
def powerForm (x : ℕ) : Prop :=
  ∃ (m n : ℕ), x = 2^m * 5^n

-- Define the condition for a quadratic equation to have exactly one real solution
def hasOneRealSolution (a b c : ℕ) : Prop :=
  4 * b^2 = 4 * a * c

-- Theorem statement
theorem infinitely_many_quadratics_with_one_solution :
  ∀ N : ℕ, ∃ (S : Finset (ℕ × ℕ × ℕ)),
    S.card > N ∧
    (∀ (abc : ℕ × ℕ × ℕ), abc ∈ S →
      let (a, b, c) := abc
      a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
      powerForm a ∧ powerForm b ∧ powerForm c ∧
      hasOneRealSolution a b c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_quadratics_with_one_solution_l1111_111117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_labeling_l1111_111189

/-- A line in a plane --/
structure Line where
  -- Add necessary fields

/-- A region formed by the intersection of lines --/
structure Region where
  -- Add necessary fields

/-- A labeling function that assigns integers to regions --/
def LabelingFunction := Region → Int

/-- Two lines are parallel --/
def are_parallel (l1 l2 : Line) : Prop :=
  sorry

/-- Three lines are concurrent --/
def are_concurrent (l1 l2 l3 : Line) : Prop :=
  sorry

/-- Divides regions into two sets based on which side of the line they're on --/
def regions_on_sides (l : Line) (regions : Finset Region) : (Finset Region) × (Finset Region) :=
  sorry

/-- Theorem: Existence of a valid labeling for m lines --/
theorem exists_valid_labeling (m : ℕ) (lines : Finset Line) (regions : Finset Region) :
  (lines.card = m) →
  (∀ l1 l2, l1 ∈ lines → l2 ∈ lines → l1 ≠ l2 → ¬ are_parallel l1 l2) →
  (∀ l1 l2 l3, l1 ∈ lines → l2 ∈ lines → l3 ∈ lines → l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 → ¬ are_concurrent l1 l2 l3) →
  ∃ f : LabelingFunction,
    (∀ r, r ∈ regions → f r ≠ 0 ∧ |f r| ≤ m) ∧
    (∀ l, l ∈ lines →
      let (left, right) := regions_on_sides l regions
      (left.sum f) = 0 ∧ (right.sum f) = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_labeling_l1111_111189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_one_two_l1111_111177

-- Define the function f(x) = lg(2x) + x - 2
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x) / Real.log 2 + x - 2

-- State the theorem
theorem solution_in_interval_one_two :
  ∃ x : ℝ, x > 1 ∧ x < 2 ∧ f x = 0 :=
by
  sorry

#eval "The theorem has been stated."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_one_two_l1111_111177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l1111_111113

theorem division_problem (x y : ℕ) (h1 : x % y = 9) (h2 : (x : ℚ) / (y : ℚ) = 96.25) : y = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l1111_111113
