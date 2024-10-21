import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_in_payments_l1308_130889

noncomputable def harry_paid : ℝ := 150
noncomputable def ron_paid : ℝ := 180
noncomputable def hermione_paid : ℝ := 210
noncomputable def total_paid : ℝ := harry_paid + ron_paid + hermione_paid
noncomputable def share_per_person : ℝ := total_paid / 3

noncomputable def harry_owes : ℝ := share_per_person - harry_paid
noncomputable def ron_owes : ℝ := share_per_person - ron_paid

theorem difference_in_payments : harry_owes - ron_owes = 30 := by
  -- Unfold definitions
  unfold harry_owes ron_owes share_per_person total_paid harry_paid ron_paid hermione_paid
  -- Simplify the expression
  simp [sub_sub, add_div]
  -- The proof steps would go here, but we'll use sorry to skip them
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_in_payments_l1308_130889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_result_l1308_130826

theorem complex_division_result : 
  (1 - Complex.I) / (2 + Complex.I) = 1/5 - 3/5 * Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_result_l1308_130826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_line_through_C_parallel_AB_l1308_130816

-- Define the polar coordinate system
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

-- Define the curves C1 and C2
def C1 (p : PolarPoint) : Prop := p.ρ = 2
def C2 (p : PolarPoint) : Prop := p.ρ * Real.sin (p.θ - Real.pi/4) = Real.sqrt 2

-- Define points A and B as intersection points of C1 and C2
noncomputable def A : PolarPoint := sorry
noncomputable def B : PolarPoint := sorry
axiom A_on_C1 : C1 A
axiom A_on_C2 : C2 A
axiom B_on_C1 : C1 B
axiom B_on_C2 : C2 B
axiom A_ne_B : A ≠ B

-- Define point C
def C : PolarPoint := ⟨1, 0⟩

-- Define the distance between two polar points
noncomputable def distance (p1 p2 : PolarPoint) : ℝ := sorry

-- Define the equation of a line in polar form
def polarLine (a b c : ℝ) (p : PolarPoint) : Prop :=
  a * p.ρ * Real.cos p.θ + b * p.ρ * Real.sin p.θ = c

-- Theorem statements
theorem distance_AB : distance A B = 2 * Real.sqrt 2 := by sorry

theorem line_through_C_parallel_AB :
  ∃ (k : ℝ), polarLine (Real.sqrt 2) (-Real.sqrt 2) k C ∧
             ∀ (p : PolarPoint), polarLine (Real.sqrt 2) (-Real.sqrt 2) k p ↔
                                 Real.sqrt 2 * p.ρ * Real.sin (p.θ - Real.pi/4) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_line_through_C_parallel_AB_l1308_130816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_range_on_interval_l1308_130801

noncomputable def f (x : ℝ) : ℝ := (2*x + 1) / (x + 1)

theorem f_monotone_increasing :
  ∀ x₁ x₂ : ℝ, -1 < x₁ → x₁ < x₂ → f x₁ < f x₂ :=
by
  sorry

theorem f_range_on_interval :
  ∀ y : ℝ, 1 ≤ y ∧ y ≤ 5/3 → ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ f x = y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_range_on_interval_l1308_130801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twisted_star_angle_sum_l1308_130836

/-- Represents an n-pointed twisted star formed from a convex n-gon. -/
structure TwistedStar (n : ℕ) where
  sides : ℕ
  h_sides : sides = n
  h_n_ge_6 : n ≥ 6
  h_non_parallel : ∀ k, k < n → True  -- Placeholder for non-parallel condition

/-- The sum of interior angles at the n points of an n-pointed twisted star. -/
noncomputable def interiorAngleSum (star : TwistedStar n) : ℝ := 
  180 * (n - 4)

/-- Theorem stating that the sum of interior angles of an n-pointed twisted star is 180°(n - 4). -/
theorem twisted_star_angle_sum (n : ℕ) (star : TwistedStar n) :
  interiorAngleSum star = 180 * (n - 4) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_twisted_star_angle_sum_l1308_130836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1308_130860

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x) + Real.log (1 - 3 * x) / Real.log 10

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Iio (1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1308_130860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_origin_to_line_l1308_130890

/-- The distance from the origin (0, 0) to the line x + 2y - 5 = 0 is √5. -/
theorem distance_origin_to_line : 
  let line := {(x, y) : ℝ × ℝ | x + 2 * y - 5 = 0}
  ∃ d : ℝ, d = Real.sqrt 5 ∧ ∀ (p : ℝ × ℝ), p ∈ line → dist (0, 0) p ≥ d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_origin_to_line_l1308_130890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_theta_from_cos_minus_sin_l1308_130822

theorem cos_two_theta_from_cos_minus_sin (θ : ℝ) (h : Real.cos θ - Real.sin θ = 3/5) : 
  Real.cos (2 * θ) = 9/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_theta_from_cos_minus_sin_l1308_130822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_folding_hole_punching_l1308_130838

-- Define the paper structure
structure Paper where
  length : ℝ
  width : ℝ
  holes : Set (ℝ × ℝ)

-- Define the folding operations
noncomputable def foldTopToBottom (p : Paper) : Paper := 
  { length := p.length / 2, width := p.width, holes := p.holes }

noncomputable def foldLeftToRight (p : Paper) : Paper := 
  { length := p.length, width := p.width / 2, holes := p.holes }

-- Define rotation
def rotate90Clockwise (p : Paper) : Paper := 
  { length := p.width, width := p.length, holes := p.holes }

-- Define hole punching
def punchHole (p : Paper) (x y : ℝ) : Paper :=
  { length := p.length, width := p.width, holes := insert (x, y) p.holes }

-- Define unfolding operation
noncomputable def unfold (p : Paper) : Paper :=
  { length := p.length * 2, width := p.width * 2,
    holes := {(x, y) | (x, y) ∈ p.holes ∨ 
                       (p.length - x, y) ∈ p.holes ∨ 
                       (x, p.width - y) ∈ p.holes ∨ 
                       (p.length - x, p.width - y) ∈ p.holes} }

-- Theorem statement
theorem paper_folding_hole_punching 
  (p : Paper) (x y : ℝ) 
  (h_x : 0 < x ∧ x < p.length / 2) 
  (h_y : 0 < y ∧ y < p.width / 2) : 
  let folded := foldLeftToRight (foldTopToBottom p)
  let rotated := rotate90Clockwise folded
  let punched := punchHole rotated x y
  let unfolded := unfold (unfold punched)
  (∃ (a b c d : ℝ × ℝ), 
    a ∈ unfolded.holes ∧ 
    b ∈ unfolded.holes ∧ 
    c ∈ unfolded.holes ∧ 
    d ∈ unfolded.holes ∧
    a.1 < unfolded.length / 2 ∧ a.2 < unfolded.width / 2 ∧
    b.1 > unfolded.length / 2 ∧ b.2 < unfolded.width / 2 ∧
    c.1 < unfolded.length / 2 ∧ c.2 > unfolded.width / 2 ∧
    d.1 > unfolded.length / 2 ∧ d.2 > unfolded.width / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_folding_hole_punching_l1308_130838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_equation_l1308_130805

theorem solution_to_equation (x : ℝ) :
  x = 1 + (Real.log 3) / (Real.log 2) →
  (2 : ℝ)^(2*x) - 8 * (2 : ℝ)^x + 12 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_equation_l1308_130805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1308_130880

/-- The area of a triangle given its side lengths -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  Real.sqrt (1/4 * (a^2 * c^2 - ((a^2 + c^2 - b^2)/2)^2))

/-- The maximum area of a triangle with given constraints -/
theorem max_triangle_area :
  ∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    a + b = 10 ∧
    ∀ (a' b' : ℝ),
      a' > 0 → b' > 0 → a' + b' = 10 →
      triangle_area a' b' 6 ≤ triangle_area a b 6 ∧
      triangle_area a b 6 = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1308_130880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_l1308_130848

/-- A right circular cone inscribed in a right prism -/
structure InscribedCone where
  /-- Width of the rectangular base of the prism -/
  w : ℝ
  /-- Height of both the cone and the prism -/
  h : ℝ
  /-- Assumption that w and h are positive -/
  w_pos : w > 0
  h_pos : h > 0

/-- Volume of the inscribed cone -/
noncomputable def cone_volume (c : InscribedCone) : ℝ := (1 / 3) * Real.pi * c.w^2 * c.h

/-- Volume of the prism -/
def prism_volume (c : InscribedCone) : ℝ := 2 * c.w^2 * c.h

/-- Theorem stating the ratio of volumes -/
theorem volume_ratio (c : InscribedCone) : 
  cone_volume c / prism_volume c = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_l1308_130848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_value_l1308_130886

theorem triangle_angle_value (A B C : ℝ) (a b c : ℝ) :
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  ((2 * a + c) * Real.cos B + b * Real.cos C = 0) →
  B = 2 * π / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_value_l1308_130886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_general_equation_l1308_130832

/-- A curve C in the Cartesian coordinate system -/
structure ParametricCurve where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The specific curve C described in the problem -/
noncomputable def C : ParametricCurve :=
  { x := λ t => 2 + (Real.sqrt 2 / 2) * t,
    y := λ t => 1 + (Real.sqrt 2 / 2) * t }

/-- The general equation of a line -/
def GeneralEquation (a b c : ℝ) : ℝ → ℝ → Prop :=
  λ x y => a * x + b * y + c = 0

/-- Theorem stating that the curve C satisfies the general equation x - y - 1 = 0 -/
theorem curve_C_general_equation :
  ∀ t, GeneralEquation 1 (-1) (-1) (C.x t) (C.y t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_general_equation_l1308_130832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_paper_thickness_l1308_130862

noncomputable def paper_thickness (n : ℕ) : ℝ :=
  1 / (2 ^ n)

theorem initial_paper_thickness :
  paper_thickness 50 = 1 → paper_thickness 0 = 1 / (2^50) :=
by
  intro h
  unfold paper_thickness
  simp
  -- The rest of the proof would go here
  sorry

#eval Float.toString (1 / (2^50 : Float))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_paper_thickness_l1308_130862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_circle_l1308_130843

/-- The length of the chord cut by a circle on a line --/
noncomputable def chord_length (center : ℝ × ℝ) (radius : ℝ) (line_x : ℝ) : ℝ :=
  2 * Real.sqrt (radius^2 - (center.1 - line_x)^2)

/-- Theorem: The length of the chord cut by the circle (x-2)^2 + (y-2)^2 = 4 on the line x = 0 is 2√2 --/
theorem chord_length_specific_circle :
  chord_length (2, 2) 2 0 = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_circle_l1308_130843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_count_l1308_130841

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a triangle is right-angled -/
def isRightTriangle (a b c : Point) : Prop :=
  (b.x - a.x) * (c.x - a.x) + (b.y - a.y) * (c.y - a.y) = 0 ∨
  (a.x - b.x) * (c.x - b.x) + (a.y - b.y) * (c.y - b.y) = 0 ∨
  (a.x - c.x) * (b.x - c.x) + (a.y - c.y) * (b.y - c.y) = 0

/-- The set of points C on the x-axis that form a right triangle with A and B -/
def rightTrianglePoints (a b : Point) : Set Point :=
  {c : Point | c.y = 0 ∧ isRightTriangle a b c}

theorem right_triangle_count :
  let a : Point := ⟨-2, 3⟩
  let b : Point := ⟨4, 3⟩
  ∃ (s : Finset Point), s.card = 3 ∧ ∀ c, c ∈ s ↔ c ∈ rightTrianglePoints a b := by
  sorry

#check right_triangle_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_count_l1308_130841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l1308_130885

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 3 * x n + 2) / (x n + 4)

theorem sequence_convergence :
  ∃ m : ℕ, m ∈ Set.Icc 19 54 ∧ 
    x m ≤ 3 + 1 / 2^15 ∧ 
    ∀ k : ℕ, k > 0 ∧ k < m → x k > 3 + 1 / 2^15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l1308_130885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_root_sum_k_l1308_130864

theorem common_root_sum_k : ∃ (S : Finset ℝ), 
  (∀ k ∈ S, ∃ x : ℝ, (x^3 - 3*x^2 + 2*x = 0 ∧ x^2 + 3*x + k = 0)) ∧
  (∀ k : ℝ, (∃ x : ℝ, x^3 - 3*x^2 + 2*x = 0 ∧ x^2 + 3*x + k = 0) → k ∈ S) ∧
  (S.sum id = -14) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_root_sum_k_l1308_130864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_not_always_four_fifths_l1308_130863

theorem sine_not_always_four_fifths (k : ℝ) (h : k ≠ 0) :
  ∃ α : ℝ, (∃ t : ℝ, t > 0 ∧ 3 * k * t = Real.cos α ∧ 4 * k * t = Real.sin α) ∧ Real.sin α ≠ 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_not_always_four_fifths_l1308_130863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l1308_130837

theorem repeating_decimal_sum : 
  let x : ℚ := 1/9  -- 0.1̅
  let y : ℚ := 2/99  -- 0.02̅
  let z : ℚ := 3/9999  -- 0.0003̅
  x + y + z = 13151314 / 99999999 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l1308_130837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_after_increment_l1308_130853

def S (n : ℕ) : ℕ := sorry

theorem sum_of_digits_after_increment (n : ℕ) (h : S n = 1274) :
  S (n + 1) ∈ ({1, 3, 12, 1239, 1265} : Set ℕ) → S (n + 1) = 1239 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_after_increment_l1308_130853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_parabola_l1308_130815

/-- Predicate to check if a given equation represents a parabola -/
def IsParabola (f : ℝ → ℝ → Prop) : Prop := 
  ∃ a b c d e : ℝ, a ≠ 0 ∧ ∀ x y, f x y ↔ a*x^2 + b*x*y + c*y^2 + d*x + e*y = 0

/-- The equation x^2 + ky^2 = 1 does not represent a parabola for any real k -/
theorem not_parabola (k : ℝ) : ¬ IsParabola (fun x y => x^2 + k*y^2 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_parabola_l1308_130815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_e_l1308_130872

noncomputable def f (x : ℝ) : ℝ := |Real.log x - 1/2|

theorem product_equals_e (a b : ℝ) (ha : a > 0) (hb : b > 0) (hne : a ≠ b) (heq : f a = f b) : a * b = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_e_l1308_130872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_trip_average_speed_l1308_130893

/-- Represents a segment of the car's journey -/
structure Segment where
  speed : ℝ  -- Speed in kph
  time : ℝ   -- Time in hours
  distance : ℝ -- Distance in km

/-- Calculates the average speed given a list of segments -/
noncomputable def averageSpeed (segments : List Segment) : ℝ :=
  let totalDistance := segments.foldr (fun s acc => s.distance + acc) 0
  let totalTime := segments.foldr (fun s acc => s.time + acc) 0
  totalDistance / totalTime

/-- The main theorem stating the average speed of the car trip -/
theorem car_trip_average_speed :
  let s1 : Segment := { speed := 30, time := 1, distance := 30 }
  let s2 : Segment := { speed := 45, time := 35 / 45, distance := 35 }
  let s3 : Segment := { speed := 70, time := 0.5, distance := 35 }
  let s4 : Segment := { speed := 55, time := 1 / 3, distance := 55 * (1 / 3) }
  let s5 : Segment := { speed := 80, time := 2 / 3, distance := 80 * (2 / 3) }
  let trip := [s1, s2, s3, s4, s5]
  abs (averageSpeed trip - 52.67) < 0.01 := by
  sorry

#eval "Car trip average speed theorem is defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_trip_average_speed_l1308_130893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l1308_130877

noncomputable def f (x : ℝ) := 2 * x + Real.log x

theorem tangent_line_slope :
  ∃ (x₀ : ℝ), x₀ > 0 ∧
  (f x₀ - (-1)) / x₀ = (2 + 1 / x₀) ∧
  (2 + 1 / x₀) = 3 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l1308_130877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_difference_l1308_130842

def arithmeticSequence (a d n : ℤ) : ℤ := a + (n - 1) * d

theorem arithmetic_sequence_difference : 
  let a := -12
  let d := 5
  let term1500 := arithmeticSequence a d 1500
  let term1520 := arithmeticSequence a d 1520
  (term1520 - term1500).natAbs = 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_difference_l1308_130842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_point_closer_to_vertex_l1308_130819

def Triangle (A B C : ℝ × ℝ) : Prop := True

def isIsosceles (A B C : ℝ × ℝ) : Prop := 
  Dist.dist A B = Dist.dist A C

def insideTriangle (P A B C : ℝ × ℝ) : Prop := True

theorem isosceles_triangle_point_closer_to_vertex (A B C P : ℝ × ℝ) :
  Triangle A B C →
  isIsosceles A B C →
  Dist.dist A B = 7 →
  Dist.dist B C = 10 →
  insideTriangle P A B C →
  Dist.dist P B = 2 * Dist.dist P C →
  Dist.dist P A < Dist.dist P B ∧ Dist.dist P A < Dist.dist P C :=
by
  sorry

#check isosceles_triangle_point_closer_to_vertex

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_point_closer_to_vertex_l1308_130819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_OQ_OD_l1308_130878

/-- The minimum distance between OQ and OD given specific conditions -/
theorem min_distance_OQ_OD :
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (0, 3)
  let C : ℝ × ℝ := (3, 0)
  let O : ℝ × ℝ := (0, 0)
  let OQ (m : ℝ) : ℝ × ℝ := (m * A.1 + (1 - m) * B.1, m * A.2 + (1 - m) * B.2)
  ∃ (D : ℝ × ℝ), 
    (D.1 - C.1)^2 + (D.2 - C.2)^2 = 1 ∧
    (∀ (m : ℝ), ∀ (D' : ℝ × ℝ), (D'.1 - C.1)^2 + (D'.2 - C.2)^2 = 1 →
      ((OQ m).1 - D.1)^2 + ((OQ m).2 - D.2)^2 ≤ ((OQ m).1 - D'.1)^2 + ((OQ m).2 - D'.2)^2) ∧
    ((OQ 0).1 - D.1)^2 + ((OQ 0).2 - D.2)^2 = (3 * Real.sqrt 2 - 1)^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_OQ_OD_l1308_130878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_cosine_sum_l1308_130839

theorem smallest_angle_cosine_sum (φ : ℝ) : 
  (φ > 0) →
  (φ < 360) →
  (Real.cos (φ * π / 180) = Real.sin (30 * π / 180) + Real.cos (24 * π / 180) - Real.sin (18 * π / 180) - Real.cos (12 * π / 180)) →
  (∀ θ : ℝ, (θ > 0) → (θ < φ) → 
    (Real.cos (θ * π / 180) ≠ Real.sin (30 * π / 180) + Real.cos (24 * π / 180) - Real.sin (18 * π / 180) - Real.cos (12 * π / 180))) →
  φ = 168 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_cosine_sum_l1308_130839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_quadrant_I_l1308_130884

/-- The range of k for which the system of equations has a solution in Quadrant I -/
theorem solution_in_quadrant_I (k : ℝ) : 
  (∃ x y : ℝ, 3 * x - 2 * y = 7 ∧ k * x + 4 * y = 10 ∧ x > 0 ∧ y > 0) ↔ 
  (-6 < k ∧ k < 30 / 7) := by
  sorry

#check solution_in_quadrant_I

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_quadrant_I_l1308_130884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_one_zeros_of_h_l1308_130824

noncomputable section

-- Define the function g(x)
def g (x : ℝ) : ℝ := Real.exp x * (x + 1)

-- Define the function h(x, a)
def h (x a : ℝ) : ℝ := g x - a * (x^3 + x^2)

-- Theorem for the tangent line
theorem tangent_line_at_zero_one :
  ∃ (m b : ℝ), ∀ x, m * x + b = g x + (deriv g 0) * (x - 0) :=
sorry

-- Theorem for the number of zeros of h(x)
theorem zeros_of_h (x a : ℝ) (hx : x > 0) (ha : a > 0) :
  (∃! x, h x a = 0 ∧ x > 0) ∧ a = Real.exp 2 / 4 ∧
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ h x₁ a = 0 ∧ h x₂ a = 0 ∧ x₁ > 0 ∧ x₂ > 0) ∧ a > Real.exp 2 / 4 ∧
  (∀ x, x > 0 → h x a ≠ 0) ∧ 0 < a ∧ a < Real.exp 2 / 4 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_one_zeros_of_h_l1308_130824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l1308_130881

-- Define the function f on the interval [a, b]
variable (f : ℝ → ℝ) (a b : ℝ)

-- Define the property of being an intersection point
def is_intersection_point (f : ℝ → ℝ) (a b x : ℝ) : Prop :=
  x = 2 ∧ x ∈ Set.Icc a b

-- Theorem statement
theorem intersection_count (f : ℝ → ℝ) (a b : ℝ) :
  (∃! x, is_intersection_point f a b x) ∨ (¬ ∃ x, is_intersection_point f a b x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l1308_130881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_of_curve_and_circle_l1308_130830

/-- The intersection point of a curve and a circle -/
theorem intersection_point_of_curve_and_circle :
  ∃ (t : ℝ), 
    t > 0 ∧
    Real.sqrt t = Real.sqrt 3 ∧ 
    (Real.sqrt (3*t))/3 = 1 ∧ 
    (Real.sqrt t)^2 + ((Real.sqrt (3*t))/3)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_of_curve_and_circle_l1308_130830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_equation_l1308_130845

/-- Given a curve C with polar equation ρ = 2√2 cos(θ - π/4), 
    its rectangular coordinate equation is x² + y² - 2x - 2y = 0 -/
theorem polar_to_rectangular_equation :
  ∀ (x y ρ θ : ℝ), 
    ρ = 2 * Real.sqrt 2 * Real.cos (θ - π/4) →
    ρ * Real.cos θ = x →
    ρ * Real.sin θ = y →
    ρ^2 = x^2 + y^2 →
    x^2 + y^2 - 2*x - 2*y = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_equation_l1308_130845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_tangents_range_l1308_130834

/-- The curve function -/
noncomputable def curve (a x : ℝ) : ℝ := (x - a) * Real.log x

/-- Condition for tangent line passing through origin -/
def tangent_through_origin (a m : ℝ) : Prop :=
  a * Real.log m + m - a = 0

/-- The theorem stating the range of a for which two tangent lines exist -/
theorem two_tangents_range (a : ℝ) :
  (∃ m₁ m₂ : ℝ, m₁ ≠ m₂ ∧ m₁ > 0 ∧ m₂ > 0 ∧
    tangent_through_origin a m₁ ∧ tangent_through_origin a m₂) ↔
  a < -Real.exp 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_tangents_range_l1308_130834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_distance_between_cars_l1308_130829

/-- The distance between two cars on a main road after specific movements -/
def distance_between_cars (initial_distance : ℝ) 
  (first_car_forward : ℝ) (second_car_forward : ℝ) : ℝ :=
  initial_distance - first_car_forward - second_car_forward

/-- The final distance between two cars given specific conditions -/
theorem final_distance_between_cars : distance_between_cars 105 25 35 = 45 := by
  unfold distance_between_cars
  norm_num
  
#eval distance_between_cars 105 25 35

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_distance_between_cars_l1308_130829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_equation_line_l_equation_when_perpendicular_max_area_APQ_l1308_130866

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the distance ratio condition
def distance_ratio (x y : ℝ) : Prop :=
  ((x - 1)^2 + y^2) / ((x - 4)^2 + y^2) = 1/4

-- Define line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 4)

-- Define the perpendicular condition
def perpendicular_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- Theorem 1: Equation of curve C
theorem curve_C_equation (x y : ℝ) :
  distance_ratio x y → curve_C x y := by sorry

-- Theorem 2: Equation of line l when OP⊥OQ
theorem line_l_equation_when_perpendicular (k x y : ℝ) :
  curve_C x y ∧ line_l k x y ∧
  perpendicular_condition x y (-y) x →
  k = Real.sqrt 7 / 7 ∨ k = -(Real.sqrt 7 / 7) := by sorry

-- Theorem 3: Maximum area of triangle APQ
theorem max_area_APQ :
  ∃ (max_area : ℝ), (∀ k : ℝ, 6 * Real.sqrt ((1 - 3*k^2) / (1 + k^2)) ≤ max_area) ∧ max_area = 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_equation_line_l_equation_when_perpendicular_max_area_APQ_l1308_130866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_min_sum_perpendicular_chords_l1308_130846

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/8 + y^2/4 = 1

-- Define the foci
def left_focus : ℝ × ℝ := (-2, 0)
def right_focus : ℝ × ℝ := (2, 0)

-- Define a line passing through the left focus with inclination angle θ
def line_through_focus (θ : ℝ) (x y : ℝ) : Prop :=
  y = (x + 2) * Real.tan θ

-- Define the length of a line segment
noncomputable def length (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem ellipse_chord_length (θ : ℝ) (A B : ℝ × ℝ) :
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧
  line_through_focus θ A.1 A.2 ∧ line_through_focus θ B.1 B.2 →
  length A B = 4 * Real.sqrt 2 / (2 - Real.cos θ ^ 2) := by
  sorry

-- Minimum value theorem
theorem min_sum_perpendicular_chords :
  ∃ (A B D E : ℝ × ℝ),
    ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ ellipse D.1 D.2 ∧ ellipse E.1 E.2 ∧
    (∃ θ₁, line_through_focus θ₁ A.1 A.2 ∧ line_through_focus θ₁ B.1 B.2) ∧
    (∃ θ₂, line_through_focus θ₂ D.1 D.2 ∧ line_through_focus θ₂ E.1 E.2) ∧
    (θ₁ - θ₂ = π / 2 ∨ θ₂ - θ₁ = π / 2) ∧
    length A B + length D E = 16 * Real.sqrt 2 / 3 ∧
    ∀ (P Q R S : ℝ × ℝ),
      ellipse P.1 P.2 ∧ ellipse Q.1 Q.2 ∧ ellipse R.1 R.2 ∧ ellipse S.1 S.2 →
      (∃ φ₁, line_through_focus φ₁ P.1 P.2 ∧ line_through_focus φ₁ Q.1 Q.2) →
      (∃ φ₂, line_through_focus φ₂ R.1 R.2 ∧ line_through_focus φ₂ S.1 S.2) →
      (φ₁ - φ₂ = π / 2 ∨ φ₂ - φ₁ = π / 2) →
      length P Q + length R S ≥ 16 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_min_sum_perpendicular_chords_l1308_130846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandys_speed_l1308_130814

/-- Calculates speed in km/hr given distance in meters and time in seconds -/
noncomputable def speed_km_hr (distance : ℝ) (time : ℝ) : ℝ :=
  (distance / time) * 3.6

/-- Sandy's run -/
theorem sandys_speed :
  let distance := (500 : ℝ)
  let time := (99.9920006399488 : ℝ)
  abs (speed_km_hr distance time - 18.000288) < 0.000001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandys_speed_l1308_130814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l1308_130835

-- Define the line l: x + λy + 2 - 3λ = 0
def line_l (x y : ℝ) (lambda : ℝ) : Prop := x + lambda * y + 2 - 3 * lambda = 0

-- Define the point P(1,1)
def point_P : ℝ × ℝ := (1, 1)

-- State the theorem
theorem max_distance_to_line :
  ∃ (d : ℝ), d = Real.sqrt 13 ∧
  ∀ (x y : ℝ) (lambda : ℝ), line_l x y lambda →
    (x - point_P.1) ^ 2 + (y - point_P.2) ^ 2 ≤ d ^ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l1308_130835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_justin_reads_enough_pages_l1308_130892

-- Define the minimum number of pages Justin needs to read
def minimum_pages : ℕ := 100

-- Define the number of pages Justin read on the first day
def first_day_pages : ℕ := 10

-- Define the number of remaining days
def remaining_days : ℕ := 6

-- Define the function to calculate the total pages read
def total_pages_read (first_day : ℕ) (remaining : ℕ) : ℕ :=
  first_day + (List.range remaining).foldl (λ acc i => acc + first_day * 2^(i + 1)) 0

-- Theorem statement
theorem justin_reads_enough_pages : 
  total_pages_read first_day_pages remaining_days ≥ minimum_pages :=
by sorry

#eval total_pages_read first_day_pages remaining_days

end NUMINAMATH_CALUDE_ERRORFEEDBACK_justin_reads_enough_pages_l1308_130892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt3_f_pi_6_lt_f_pi_3_l1308_130882

-- Define the function f on the open interval (0, π/2)
noncomputable def f : ℝ → ℝ := sorry

-- Define the derivative of f
noncomputable def f' : ℝ → ℝ := sorry

-- Condition that f(x) < f'(x) * tan(x) for all x in (0, π/2)
axiom f_lt_f'_tan_x : ∀ x : ℝ, 0 < x → x < Real.pi / 2 → 
  f x < f' x * Real.tan x

-- Theorem to prove
theorem sqrt3_f_pi_6_lt_f_pi_3 : 
  Real.sqrt 3 * f (Real.pi / 6) < f (Real.pi / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt3_f_pi_6_lt_f_pi_3_l1308_130882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_R_l1308_130803

/-- Represents a square in 2D space -/
structure Square where
  sideLength : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an angle in radians -/
def Angle := ℝ

/-- Represents a region in 2D space -/
structure Region where
  points : Set Point

/-- Function to calculate the area of a region -/
noncomputable def areaOfRegion (r : Region) : ℝ := sorry

/-- Function to determine if a point is closer to one vertex than others -/
def isCloserToVertex (p : Point) (v : Point) (otherVertices : List Point) : Prop := sorry

/-- Theorem stating the area of region R in the given problem -/
theorem area_of_region_R (squareABCD : Square) (squareBEFG : Square) (angleEBF : Angle) : 
  squareABCD.sideLength = 4 →
  angleEBF = Real.pi * 2 / 3 →
  let vertexB : Point := ⟨0, 0⟩
  let otherVertices : List Point := [⟨4, 0⟩, ⟨4, 4⟩, ⟨0, 4⟩]
  let R : Region := { points := { p | isCloserToVertex p vertexB otherVertices } }
  areaOfRegion R = 4 * Real.sqrt 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_R_l1308_130803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_soldiers_same_schedule_l1308_130849

/-- Represents a point in 2D plane (soldier's height and age) -/
structure Soldier where
  height : ℝ
  age : ℝ

/-- Represents an L-shaped division of the plane -/
structure LShape where
  pivot : Soldier

/-- Checks if a soldier is in the same region as the pivot of an L-shape -/
def sameRegion (s : Soldier) (l : LShape) : Prop :=
  (s.height > l.pivot.height ∧ s.age > l.pivot.age) ∨
  (s.height < l.pivot.height ∧ s.age < l.pivot.age)

/-- The main theorem -/
theorem two_soldiers_same_schedule
  (soldiers : Finset Soldier)
  (h_count : soldiers.card = 85)
  (h_distinct : ∀ s1 s2 : Soldier, s1 ∈ soldiers → s2 ∈ soldiers → s1 ≠ s2 → s1.height ≠ s2.height ∨ s1.age ≠ s2.age)
  (lshapes : Finset LShape)
  (h_lshapes : lshapes.card = 10) :
  ∃ s1 s2 : Soldier, s1 ∈ soldiers ∧ s2 ∈ soldiers ∧ s1 ≠ s2 ∧
    ∀ l ∈ lshapes, sameRegion s1 l ↔ sameRegion s2 l :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_soldiers_same_schedule_l1308_130849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lloyds_earnings_l1308_130847

/-- Lloyd's working conditions and earnings calculation --/
theorem lloyds_earnings 
  (regular_hours : ℝ) 
  (total_hours : ℝ) 
  (regular_rate : ℝ) 
  (overtime_multiplier : ℝ) :
  regular_hours = 7.5 →
  total_hours = 10.5 →
  regular_rate = 4 →
  overtime_multiplier = 1.5 →
  (regular_hours * regular_rate + 
   (total_hours - regular_hours) * (regular_rate * overtime_multiplier)) = 48 := by
  intro h1 h2 h3 h4
  -- The proof steps would go here
  sorry

#check lloyds_earnings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lloyds_earnings_l1308_130847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_one_f_greater_than_two_iff_l1308_130851

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -x + 3 else 4 * x

theorem f_composition_negative_one : f (f (-1)) = 16 := by sorry

theorem f_greater_than_two_iff (x₀ : ℝ) : f x₀ > 2 ↔ x₀ ∈ Set.Iic 0 ∪ Set.Ioi (1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_one_f_greater_than_two_iff_l1308_130851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_zeros_l1308_130871

noncomputable def f (x : ℝ) : ℝ := 3 * Real.log x + (1/2) * x^2 - 4*x + 1

noncomputable def g (x m : ℝ) : ℝ := f x - m

theorem tangent_line_and_zeros (x : ℝ) :
  (∃ y : ℝ, x + 2*y - 6*Real.log 2 + 8 = 0 ↔ 
    y = f 2 ∧ (x - 2) * (deriv f) 2 = y - f 2) ∧
  (∀ m : ℝ, 
    (m > -5/2 ∨ m < 3*Real.log 3 - 13/2 → 
      (∃! x, g x m = 0)) ∧
    (m = -5/2 ∨ m = 3*Real.log 3 - 13/2 → 
      (∃ x₁ x₂, x₁ ≠ x₂ ∧ g x₁ m = 0 ∧ g x₂ m = 0 ∧ ∀ x, g x m = 0 → x = x₁ ∨ x = x₂)) ∧
    (3*Real.log 3 - 13/2 < m ∧ m < -5/2 → 
      (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
        g x₁ m = 0 ∧ g x₂ m = 0 ∧ g x₃ m = 0 ∧
        ∀ x, g x m = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_zeros_l1308_130871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mia_bus_encounter_probability_l1308_130807

/-- The duration of the arrival window in minutes -/
noncomputable def arrivalWindow : ℝ := 60

/-- The duration the bus waits at the stop in minutes -/
noncomputable def busWaitTime : ℝ := 15

/-- The probability that Mia arrives while the bus is at the stop -/
noncomputable def arrivalProbability : ℝ := 25 / 128

theorem mia_bus_encounter_probability :
  let totalArea := arrivalWindow * arrivalWindow
  let triangleArea := (arrivalWindow - busWaitTime) * busWaitTime / 2
  let rectangleArea := busWaitTime * busWaitTime
  (triangleArea + rectangleArea) / totalArea = arrivalProbability := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mia_bus_encounter_probability_l1308_130807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_is_2016_l1308_130875

/-- A partition of a set into 63 subsets -/
def Partition (α : Type*) := 
  Fin 63 → Set α

/-- The property that a partition is valid for a set A -/
def ValidPartition {α : Type*} (A : Set α) (P : Partition α) : Prop :=
  (∀ i j, i ≠ j → Disjoint (P i) (P j)) ∧ 
  (∀ i, (P i).Nonempty) ∧
  (⋃ i, P i) = A

/-- The property that a set A satisfies the condition -/
def SatisfiesCondition (A : Set ℕ) : Prop :=
  ∀ P : Partition ℕ, ValidPartition A P →
    ∃ i : Fin 63, ∃ x y, x ∈ P i ∧ y ∈ P i ∧ x > y ∧ 31 * x ≤ 32 * y

theorem smallest_n_is_2016 :
  (∀ n < 2016, ¬SatisfiesCondition (Finset.range (n + 1)).toSet) ∧
  SatisfiesCondition (Finset.range 2016).toSet :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_is_2016_l1308_130875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_forest_analysis_l1308_130876

/-- Represents the data for a single tree sample -/
structure TreeSample where
  rootArea : ℝ
  volume : ℝ

/-- Represents the forest data -/
structure ForestData where
  samples : List TreeSample
  sumRootArea : ℝ
  sumVolume : ℝ
  sumRootAreaSquared : ℝ
  sumVolumeSquared : ℝ
  sumRootAreaVolume : ℝ
  totalRootArea : ℝ

/-- Calculates the average of a list of real numbers -/
noncomputable def average (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

/-- Calculates the sample correlation coefficient -/
noncomputable def sampleCorrelationCoefficient (data : ForestData) : ℝ :=
  let n := data.samples.length
  let xMean := average (data.samples.map (·.rootArea))
  let yMean := average (data.samples.map (·.volume))
  let numerator := data.sumRootAreaVolume - n * xMean * yMean
  let denominatorX := data.sumRootAreaSquared - n * xMean * xMean
  let denominatorY := data.sumVolumeSquared - n * yMean * yMean
  numerator / Real.sqrt (denominatorX * denominatorY)

/-- Estimates the total volume based on the total root area -/
noncomputable def estimateTotalVolume (data : ForestData) : ℝ :=
  let avgRootArea := average (data.samples.map (·.rootArea))
  let avgVolume := average (data.samples.map (·.volume))
  (avgVolume / avgRootArea) * data.totalRootArea

theorem forest_analysis (data : ForestData)
  (h1 : data.samples.length = 10)
  (h2 : data.sumRootArea = 0.6)
  (h3 : data.sumVolume = 3.9)
  (h4 : data.sumRootAreaSquared = 0.038)
  (h5 : data.sumVolumeSquared = 1.6158)
  (h6 : data.sumRootAreaVolume = 0.2474)
  (h7 : data.totalRootArea = 186) :
  let avgRootArea := average (data.samples.map (·.rootArea))
  let avgVolume := average (data.samples.map (·.volume))
  let corrCoeff := sampleCorrelationCoefficient data
  let totalVolume := estimateTotalVolume data
  avgRootArea = 0.06 ∧
  avgVolume = 0.39 ∧
  abs (corrCoeff - 0.97) < 0.01 ∧
  abs (totalVolume - 1209) < 1 := by
  sorry

#eval "The Lean 4 statement has been generated successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_forest_analysis_l1308_130876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l1308_130840

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (px py a b c : ℝ) : ℝ :=
  (|a * px + b * py + c|) / (Real.sqrt (a^2 + b^2))

theorem distance_circle_center_to_line :
  let circle_center_x : ℝ := 0
  let circle_center_y : ℝ := 2
  let line_a : ℝ := 1
  let line_b : ℝ := 1
  let line_c : ℝ := -6
  distance_point_to_line circle_center_x circle_center_y line_a line_b line_c = 2 * Real.sqrt 2 := by
  sorry

#check distance_circle_center_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l1308_130840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chance_of_regaining_control_l1308_130831

-- Define constants
noncomputable def normalTemp : ℝ := 32
noncomputable def currentTemp : ℝ := 8
noncomputable def tempDropPerIncrement : ℝ := 3
noncomputable def skidChanceIncreasePerIncrement : ℝ := 0.05
noncomputable def seriousAccidentChance : ℝ := 0.24

-- Define functions
noncomputable def tempDrop : ℝ := normalTemp - currentTemp

noncomputable def skidChanceIncrease : ℝ := 
  (tempDrop / tempDropPerIncrement) * skidChanceIncreasePerIncrement

-- Theorem to prove
theorem chance_of_regaining_control : 
  ∃ (chanceOfRegainingControl : ℝ),
    chanceOfRegainingControl = 0.4 ∧
    skidChanceIncrease * (1 - chanceOfRegainingControl) = seriousAccidentChance :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chance_of_regaining_control_l1308_130831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_extension_l1308_130833

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 2*x else -x^2 - 2*x

-- State the theorem
theorem odd_function_extension :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x ≥ 0, f x = x^2 - 2*x) →
  ∀ x < 0, f x = -x^2 - 2*x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_extension_l1308_130833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_objective_function_range_l1308_130823

/-- Given constraints on x and y, prove the range of z = 3x - y -/
theorem objective_function_range (x y : ℝ) 
  (h1 : x + 2*y ≥ 2) 
  (h2 : 2*x + y ≤ 4) 
  (h3 : 4*x - y ≥ -1) : 
  -3/2 ≤ 3*x - y ∧ 3*x - y ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_objective_function_range_l1308_130823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l1308_130894

/-- Given two plane vectors a and b, where the angle between them is 60°,
    a = (2,0), and |b| = 1, prove that |a + 2b| = 2√3 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) : 
  (a.1 = 2 ∧ a.2 = 0) →                  -- a = (2,0)
  Real.sqrt (b.1^2 + b.2^2) = 1 →        -- |b| = 1
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π/3 →  -- angle is 60°
  Real.sqrt ((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2) = 2 * Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l1308_130894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_apex_angle_l1308_130861

structure Sphere where
  radius : ℝ
  center : ℝ × ℝ × ℝ

structure Cone where
  apex : ℝ × ℝ × ℝ
  axis : ℝ × ℝ × ℝ
  apexAngle : ℝ

def spheresOnTable (s1 s2 s3 : Sphere) : Prop :=
  s1.radius = 2 ∧ s2.radius = 2 ∧ s3.radius = 1 ∧
  s1.center.2.2 = s1.radius ∧ s2.center.2.2 = s2.radius ∧ s3.center.2.2 = s3.radius

def spheresTouchExternally (s1 s2 s3 : Sphere) : Prop :=
  (s1.center.1 - s2.center.1)^2 + (s1.center.2.1 - s2.center.2.1)^2 = (s1.radius + s2.radius)^2 ∧
  (s2.center.1 - s3.center.1)^2 + (s2.center.2.1 - s3.center.2.1)^2 = (s2.radius + s3.radius)^2 ∧
  (s3.center.1 - s1.center.1)^2 + (s3.center.2.1 - s1.center.2.1)^2 = (s3.radius + s1.radius)^2

def coneTouchesSpheres (c : Cone) (s1 s2 s3 : Sphere) : Prop :=
  ∃ (p1 p2 p3 : ℝ × ℝ × ℝ),
    (p1.1 - c.apex.1)^2 + (p1.2.1 - c.apex.2.1)^2 + (p1.2.2 - c.apex.2.2)^2 = 
      ((p1.1 - s1.center.1)^2 + (p1.2.1 - s1.center.2.1)^2 + (p1.2.2 - s1.center.2.2)^2) * (Real.tan (c.apexAngle / 2))^2 ∧
    (p2.1 - c.apex.1)^2 + (p2.2.1 - c.apex.2.1)^2 + (p2.2.2 - c.apex.2.2)^2 = 
      ((p2.1 - s2.center.1)^2 + (p2.2.1 - s2.center.2.1)^2 + (p2.2.2 - s2.center.2.2)^2) * (Real.tan (c.apexAngle / 2))^2 ∧
    (p3.1 - c.apex.1)^2 + (p3.2.1 - c.apex.2.1)^2 + (p3.2.2 - c.apex.2.2)^2 = 
      ((p3.1 - s3.center.1)^2 + (p3.2.1 - s3.center.2.1)^2 + (p3.2.2 - s3.center.2.2)^2) * (Real.tan (c.apexAngle / 2))^2

def coneApexEquidistant (c : Cone) (s1 s2 : Sphere) : Prop :=
  (c.apex.1 - s1.center.1)^2 + (c.apex.2.1 - s1.center.2.1)^2 + (c.apex.2.2 - s1.center.2.2)^2 =
  (c.apex.1 - s2.center.1)^2 + (c.apex.2.1 - s2.center.2.1)^2 + (c.apex.2.2 - s2.center.2.2)^2

def conePerpendicularToThirdSphere (c : Cone) (s3 : Sphere) : Prop :=
  c.axis.1 = 0 ∧ c.axis.2.1 = 0 ∧ c.axis.2.2 = 1

theorem cone_apex_angle (s1 s2 s3 : Sphere) (c : Cone) :
  spheresOnTable s1 s2 s3 →
  spheresTouchExternally s1 s2 s3 →
  coneTouchesSpheres c s1 s2 s3 →
  coneApexEquidistant c s1 s2 →
  conePerpendicularToThirdSphere c s3 →
  c.apexAngle = 2 * Real.arctan ((Real.sqrt 5 - 2) / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_apex_angle_l1308_130861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_good_point_on_circle_l1308_130854

/-- A type representing a point on the circle with its label -/
structure Point where
  label : Int
  deriving Repr

/-- Definition of a good point -/
def is_good_point (circle : List Point) (start : Nat) : Prop :=
  ∀ (direction : Bool) (steps : Nat), 
    let rotated := if direction then circle.rotate start else circle.rotate' start
    List.sum (List.map Point.label (rotated.take steps)) > 0

/-- The main theorem -/
theorem exists_good_point_on_circle 
  (circle : List Point) 
  (h_length : circle.length = 2000)
  (h_labels : ∀ p ∈ circle, p.label = 1 ∨ p.label = -1)
  (h_negative_count : (circle.filter (λ p => p.label = -1)).length < 667) :
  ∃ i, is_good_point circle i := by
  sorry

#check exists_good_point_on_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_good_point_on_circle_l1308_130854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_interior_angle_sum_l1308_130888

theorem regular_polygon_interior_angle_sum 
  (n : ℕ) 
  (h_exterior_angle : 360 / n = 45) : 
  180 * (n - 2) = 1080 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_interior_angle_sum_l1308_130888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l1308_130895

/-- The time required to fill a tank with given capacity and flow rates -/
noncomputable def time_to_fill (tank_capacity : ℝ) (initial_fill : ℝ) (inflow_rate : ℝ) (outflow_rate1 : ℝ) (outflow_rate2 : ℝ) : ℝ :=
  let remaining_volume := tank_capacity - initial_fill
  let net_flow_rate := inflow_rate - (outflow_rate1 + outflow_rate2)
  remaining_volume / net_flow_rate

/-- Theorem stating that the time to fill the tank is 60 minutes -/
theorem tank_fill_time :
  time_to_fill 10000 5000 (1000 / 2) (1000 / 4) (1000 / 6) = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l1308_130895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_after_15th_innings_l1308_130883

noncomputable def batsman_average (total_innings : ℕ) (runs_in_last_innings : ℕ) (average_increase : ℝ) : ℝ :=
  let previous_average := (total_innings - 1 : ℝ) * (average_increase + (runs_in_last_innings : ℝ) / total_innings)
  (previous_average + runs_in_last_innings) / total_innings

theorem batsman_average_after_15th_innings :
  batsman_average 15 85 3 = 43 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_after_15th_innings_l1308_130883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_shaded_area_l1308_130821

-- Define the side length of the square
def side_length : ℝ := 4

-- Define pi (using Real.pi)
noncomputable def π : ℝ := Real.pi

-- Define the shaded area of Figure A
noncomputable def shaded_area_A : ℝ := side_length^2 - π * (side_length / 2)^2

-- Define the shaded area of Figure B
noncomputable def shaded_area_B : ℝ := side_length^2 - π * (side_length / 2)^2

-- Define the shaded area of Figure C
noncomputable def shaded_area_C : ℝ := π * (side_length * Real.sqrt 2 / 2)^2 - side_length^2

-- Theorem stating that the shaded area of Figure C is the largest
theorem largest_shaded_area :
  shaded_area_C > shaded_area_A ∧ shaded_area_C > shaded_area_B := by
  sorry

#eval side_length -- This will compile and output 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_shaded_area_l1308_130821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_15_l1308_130857

theorem remainder_sum_mod_15 (a b c d : ℕ) 
  (ha : a % 15 = 11)
  (hb : b % 15 = 12)
  (hc : c % 15 = 13)
  (hd : d % 15 = 14) : 
  (a + b + c + d) % 15 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_15_l1308_130857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1308_130868

noncomputable def f (x : ℝ) : ℝ := Real.sqrt ((5 * x - 4) / (x + 3)) + Real.sqrt ((x - 16) / (x + 3))

def domain : Set ℝ := {x | x < -3 ∨ x ≥ 16}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | (y ≥ 2 ∧ y < 1 + Real.sqrt 5) ∨ y > 1 + Real.sqrt 5} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1308_130868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_m_with_k_diff_primes_l1308_130802

/-- σ(n) denotes the sum of all positive divisors of a positive integer n -/
def sigma (n : ℕ+) : ℕ+ :=
  sorry

/-- D(n) is the set of all distinct prime factors of n -/
def D (n : ℕ+) : Finset ℕ :=
  sorry

/-- For any natural number k, there exists a positive integer m such that
    the cardinality of the set difference between D(σ(m)) and D(m) is equal to k -/
theorem exists_m_with_k_diff_primes (k : ℕ) :
  ∃ m : ℕ+, (D (sigma m) \ D m).card = k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_m_with_k_diff_primes_l1308_130802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_not_divisible_by_three_not_in_sequence_l1308_130850

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The sequence defined in the problem -/
def problem_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => problem_sequence n + sum_of_digits (problem_sequence n)

/-- Theorem stating that no term in the sequence is divisible by 3 -/
theorem sequence_not_divisible_by_three (n : ℕ) : 
  problem_sequence n % 3 ≠ 0 := by sorry

/-- Corollary: 793210041 is not in the sequence -/
theorem not_in_sequence : ¬ ∃ n, problem_sequence n = 793210041 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_not_divisible_by_three_not_in_sequence_l1308_130850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_zero_h_even_iff_m_neg_two_m_range_for_g_f_equality_l1308_130873

-- Define the functions
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + m*x + 1
noncomputable def h (m : ℝ) (x : ℝ) : ℝ := f m x + 2*x
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi/6)

-- Theorem statements
theorem max_value_f_zero (x : ℝ) : 
  f 0 x ≤ 1 := by sorry

theorem h_even_iff_m_neg_two (m : ℝ) : 
  (∀ x : ℝ, h m x = h m (-x)) ↔ m = -2 := by sorry

theorem m_range_for_g_f_equality : 
  (∀ x₁ ∈ Set.Icc 1 2, ∃ x₂ ∈ Set.Icc 0 Real.pi, g x₂ = f m x₁) ↔ 
  m ∈ Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_zero_h_even_iff_m_neg_two_m_range_for_g_f_equality_l1308_130873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_exists_l1308_130806

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 + 4*x + 4

-- Define the line
def line (n c x : ℝ) : ℝ := n*x + c

-- Define the condition for intersection points being 4 units apart
def intersection_condition (k n c : ℝ) : Prop :=
  abs (parabola k - line n c k) = 4

-- Define the condition that the line passes through (2, 5)
def point_condition (n c : ℝ) : Prop :=
  line n c 2 = 5

-- Main theorem
theorem unique_line_exists :
  ∃! (n c : ℝ), c ≠ 0 ∧
  point_condition n c ∧
  (∃! k, intersection_condition k n c) ∧
  n = 2 ∧ c = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_exists_l1308_130806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_imply_n_equals_one_l1308_130855

/-- Given vectors a, b, and c in ℝ², prove that if a + b is parallel to c, then n = 1 -/
theorem parallel_vectors_imply_n_equals_one (n : ℝ) :
  let a : Fin 2 → ℝ := ![n, -1]
  let b : Fin 2 → ℝ := ![-1, 1]
  let c : Fin 2 → ℝ := ![-1, 2]
  (∃ (k : ℝ), a + b = k • c) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_imply_n_equals_one_l1308_130855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1308_130811

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  A = π / 3 →  -- 60° in radians
  b = 2 →
  c = 1 →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1308_130811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_probability_l1308_130809

structure Point where
  x : ℝ
  y : ℝ

def Parallelogram (P Q R S : Point) : Prop :=
  (Q.x - P.x = R.x - S.x) ∧ (Q.y - P.y = R.y - S.y)

def RightOfYAxis (p : Point) : Prop := p.x > 0

-- We need to define a probability measure on the parallelogram
-- This is a simplification and may not fully capture the concept of probability in this context
noncomputable def Probability (f : Point → Prop) : ℝ := sorry

theorem parallelogram_probability (P Q R S : Point)
  (h_parallelogram : Parallelogram P Q R S)
  (h_P : P = ⟨-4, 4⟩)
  (h_Q : Q = ⟨2, -2⟩)
  (h_R : R = ⟨4, -2⟩)
  (h_S : S = ⟨-2, 4⟩) :
  Probability RightOfYAxis = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_probability_l1308_130809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_and_skew_projections_l1308_130899

-- Define perpendicularity of lines
def perpendicular (m1 : ℝ) (m2 : ℝ) : Prop := m1 * m2 = -1

-- Define the concept of skew lines
def skew_lines (l1 l2 : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define the concept of projection
def projection (l : Set (ℝ × ℝ × ℝ)) (plane : Set (ℝ × ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

-- Define parallel lines
def parallel_lines (l1 l2 : Set (ℝ × ℝ)) : Prop := sorry

-- Statement of the theorem
theorem perpendicular_and_skew_projections :
  (perpendicular 2 (-1/2)) ∧
  (∃ (l1 l2 : Set (ℝ × ℝ × ℝ)) (plane : Set (ℝ × ℝ × ℝ)),
    skew_lines l1 l2 ∧ parallel_lines (projection l1 plane) (projection l2 plane)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_and_skew_projections_l1308_130899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l1308_130870

-- Define the ellipse
def Ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 11 = 1

-- Define the foci
structure Foci :=
  (F₁ : ℝ × ℝ)
  (F₂ : ℝ × ℝ)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem ellipse_foci_distance 
  (P : ℝ × ℝ) 
  (h_ellipse : Ellipse P.1 P.2)
  (foci : Foci)
  (h_distance : distance P foci.F₁ = 3) :
  distance P foci.F₂ = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l1308_130870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distinct_integers_for_distinct_ratios_l1308_130897

theorem min_distinct_integers_for_distinct_ratios :
  ∃ (n : ℕ) (a : Fin 2006 → ℕ+),
    (∀ i j : Fin 2005, i ≠ j → (a i : ℚ) / (a (i + 1) : ℚ) ≠ (a j : ℚ) / (a (j + 1) : ℚ)) →
    (∃ (s : Finset ℕ+), s.card = n ∧ (∀ i : Fin 2006, a i ∈ s)) →
    (∀ m : ℕ, m < n →
      ¬∃ (b : Fin 2006 → ℕ+) (t : Finset ℕ+),
        t.card = m ∧
        (∀ i : Fin 2006, b i ∈ t) ∧
        (∀ i j : Fin 2005, i ≠ j → (b i : ℚ) / (b (i + 1) : ℚ) ≠ (b j : ℚ) / (b (j + 1) : ℚ))) →
    n = 46 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distinct_integers_for_distinct_ratios_l1308_130897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_price_calculation_l1308_130898

theorem dress_price_calculation (original_price discount_rate tax_rate : ℝ) 
  (h1 : original_price = 100)
  (h2 : discount_rate = 0.30)
  (h3 : tax_rate = 0.15) :
  original_price * (1 - discount_rate) * (1 + tax_rate) = 80.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_price_calculation_l1308_130898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_implies_lambda_l1308_130810

/-- Prove that for vectors a = (1, l) and b = (-2, 1), if a is parallel to b, then l = -1/2. -/
theorem vector_parallel_implies_lambda (l : ℝ) :
  let a : ℝ × ℝ := (1, l)
  let b : ℝ × ℝ := (-2, 1)
  (∃ (k : ℝ), a = k • b) → l = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_implies_lambda_l1308_130810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_range_l1308_130859

/-- Given two functions f and g, prove that if there exist two points where a line is tangent to both functions, then the parameter a is in the open interval (-1, 1/8) -/
theorem tangent_line_range (a : ℝ) : 
  let f := λ (x : ℝ) => x^2 + x + 2*a
  let g := λ (x : ℝ) => -1/x
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ > 0 ∧ 
    (∃ m b : ℝ, 
      (∀ x : ℝ, m*x + b = f x ∨ m*x + b = g x → 
        (x = x₁ ∨ x = x₂) ∧ 
        (deriv f x₁ = m ∨ deriv g x₂ = m)))) → 
  a > -1 ∧ a < 1/8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_range_l1308_130859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l1308_130813

-- Define the parametric equation of line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 + (1/2) * t, (Real.sqrt 3 / 2) * t)

-- Define the polar equation of curve C
noncomputable def curve_C (θ : ℝ) : ℝ := 8 * Real.cos θ / (Real.sin θ)^2

-- State the theorem
theorem intersection_chord_length :
  -- The Cartesian equation of curve C is y² = 8x
  (∀ x y, (x, y) ∈ Set.range (fun θ => (curve_C θ * Real.cos θ, curve_C θ * Real.sin θ)) ↔ y^2 = 8*x) ∧
  -- The length of chord AB is 32/3
  (∃ A B : ℝ × ℝ,
    A ∈ Set.range line_l ∧
    B ∈ Set.range line_l ∧
    A ∈ Set.range (fun θ => (curve_C θ * Real.cos θ, curve_C θ * Real.sin θ)) ∧
    B ∈ Set.range (fun θ => (curve_C θ * Real.cos θ, curve_C θ * Real.sin θ)) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 32/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l1308_130813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_m_value_l1308_130808

theorem angle_terminal_side_m_value (α : ℝ) (m : ℝ) :
  let P : ℝ × ℝ := (-8 * m, -3)
  Real.cos α = -4/5 →
  m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_m_value_l1308_130808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_C_is_quadratic_l1308_130858

-- Define the structure of a quadratic function
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero : a ≠ 0

-- Define the given functions
noncomputable def func_A : ℝ → ℝ := λ x => 1 / x
noncomputable def func_B : ℝ → ℝ := λ x => x + 1
noncomputable def func_C : ℝ → ℝ := λ x => 2 * x^2 - 1
noncomputable def func_D : ℝ → ℝ := λ x => (2/3) * x

-- Theorem statement
theorem only_C_is_quadratic :
  (∃ q : QuadraticFunction, ∀ x, func_C x = q.a * x^2 + q.b * x + q.c) ∧
  (¬ ∃ q : QuadraticFunction, ∀ x, func_A x = q.a * x^2 + q.b * x + q.c) ∧
  (¬ ∃ q : QuadraticFunction, ∀ x, func_B x = q.a * x^2 + q.b * x + q.c) ∧
  (¬ ∃ q : QuadraticFunction, ∀ x, func_D x = q.a * x^2 + q.b * x + q.c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_C_is_quadratic_l1308_130858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solutions_and_properties_l1308_130865

noncomputable def a : ℝ × ℝ := (1, Real.sqrt 3)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def f (x : ℝ) : ℝ := a.1 * (b x).1 + a.2 * (b x).2 - 1

theorem f_solutions_and_properties :
  (∀ x : ℝ, f x = 0 ↔ ∃ k : ℤ, x = 2 * k * Real.pi ∨ x = 2 * Real.pi / 3 + 2 * k * Real.pi) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) →
    (∀ y : ℝ, y ∈ Set.Icc 0 (Real.pi / 3) → x ≤ y → f x ≤ f y) ∧
    (∀ y : ℝ, y ∈ Set.Icc (Real.pi / 3) (Real.pi / 2) → x ≤ y → f y ≤ f x)) ∧
  (f 0 = 0 ∧ f (Real.pi / 3) = 1 ∧
    ∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → 0 ≤ f x ∧ f x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solutions_and_properties_l1308_130865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_function_min_value_achieved_l1308_130869

theorem min_value_trig_function (x : ℝ) (h1 : Real.sin x ≠ 0) (h2 : Real.cos x ≠ 0) : 
  1 / (Real.cos x)^2 + 1 / (Real.sin x)^2 ≥ 4 :=
by sorry

theorem min_value_achieved : ∃ x : ℝ, Real.sin x ≠ 0 ∧ Real.cos x ≠ 0 ∧ 
  1 / (Real.cos x)^2 + 1 / (Real.sin x)^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_function_min_value_achieved_l1308_130869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composition_l1308_130887

-- Define the function f with domain [-2, 2]
def f : Set ℝ := Set.Icc (-2) 2

-- Define the composition g(x) = f(x^2 - 1)
def g : Set ℝ := { x | x^2 - 1 ∈ f }

-- Theorem statement
theorem domain_of_composition :
  g = Set.Icc (-Real.sqrt 3) (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composition_l1308_130887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l1308_130856

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 9 = 1

-- Define the foci
noncomputable def F1 : ℝ × ℝ := (-Real.sqrt 7, 0)
noncomputable def F2 : ℝ × ℝ := (Real.sqrt 7, 0)

-- Define a point on the ellipse
noncomputable def P : ℝ × ℝ := sorry

-- Helper functions (not to be proved)
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem ellipse_triangle_area :
  ellipse P.1 P.2 →
  angle F1 P F2 = π / 6 →
  triangle_area F1 P F2 = 18 - 9 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l1308_130856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_in_set_with_average_27_l1308_130825

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬(n % m = 0)

theorem largest_prime_in_set_with_average_27 
  (S : Finset Nat) 
  (distinct_primes : ∀ p ∈ S, isPrime p ∧ ∀ q ∈ S, p ≠ q → p ≠ q)
  (average_27 : (S.sum id) / S.card = 27) :
  ∃ p ∈ S, p = 139 ∧ ∀ q ∈ S, q ≤ p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_in_set_with_average_27_l1308_130825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snail_kite_problem_l1308_130828

theorem snail_kite_problem (sequence : Fin 5 → ℕ) 
  (first_day : sequence 0 = 3)
  (arithmetic : ∀ i : Fin 4, sequence (i.succ) = sequence i + (sequence 1 - sequence 0))
  (total : Finset.sum (Finset.range 5) (λ i => sequence i) = 35) :
  sequence 1 - sequence 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snail_kite_problem_l1308_130828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_dataset_l1308_130818

noncomputable def data_set : List ℝ := [2, 4, 5, 6, 8]

noncomputable def mean (xs : List ℝ) : ℝ :=
  (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  (xs.map (fun x => (x - mean xs) ^ 2)).sum / xs.length

theorem variance_of_dataset : variance data_set = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_dataset_l1308_130818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_constant_l1308_130879

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the point C
def C (c : ℝ) : ℝ × ℝ := (0, c)

-- Define a chord AB passing through C
def chord (c : ℝ) (A B : ℝ × ℝ) : Prop :=
  A.2 = parabola A.1 ∧ B.2 = parabola B.1 ∧ c - 0 = (B.2 - A.2) / (B.1 - A.1) * (0 - A.1) + A.2

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define the expression t
noncomputable def t (c : ℝ) (A B : ℝ × ℝ) : ℝ :=
  1 / distance (C c) A + 1 / distance (C c) B

-- State the theorem
theorem parabola_chord_constant (c : ℝ) (h : c > 0) :
  ∀ A B : ℝ × ℝ, chord c A B → t c A B = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_constant_l1308_130879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABM_is_30_degrees_l1308_130820

/-- A rectangle with sides of length 1 and 2 -/
structure Rectangle :=
  (AB : ℝ) (BC : ℝ)
  (h1 : AB = 1)
  (h2 : BC = 2)

/-- The angle ABM formed when the rectangle is folded -/
noncomputable def angle_ABM (rect : Rectangle) : ℝ := 
  Real.arctan (3 / (2 * Real.sqrt 2)) * (180 / Real.pi)

/-- Theorem stating that angle ABM is 30 degrees -/
theorem angle_ABM_is_30_degrees (rect : Rectangle) : 
  angle_ABM rect = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABM_is_30_degrees_l1308_130820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_result_l1308_130874

structure Competitor where
  speed : ℝ → ℝ
  distance : ℝ → ℝ

def Tortoise : Competitor where
  speed := λ _ => 1  -- Constant speed
  distance := λ t => t  -- Linear distance function

noncomputable def Hare : Competitor where
  speed := λ t => if t < 1 then 2 else if 1 ≤ t ∧ t < 2 then 0 else 2
  distance := λ t => if t < 1 then 2*t else if 1 ≤ t ∧ t < 2 then 2 else 2 + 2*(t-2)

theorem race_result (t : ℝ) (h : t > 2) : Tortoise.distance t > Hare.distance t := by
  sorry

#check race_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_result_l1308_130874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_overhead_expenses_l1308_130827

/-- Given the cost price, selling price, and profit percent of a radio,
    calculate the overhead expenses of the retailer. -/
theorem retailer_overhead_expenses 
  (cost_price : ℝ) 
  (selling_price : ℝ) 
  (profit_percent : ℝ) 
  (h1 : cost_price = 225)
  (h2 : selling_price = 300)
  (h3 : profit_percent = 17.64705882352942) :
  ∃ (overhead_expenses : ℝ), 
    (selling_price - (cost_price + overhead_expenses) = (profit_percent / 100) * (cost_price + overhead_expenses)) ∧
    (abs (overhead_expenses - 30) < 0.01) := by
  sorry

#check retailer_overhead_expenses

end NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_overhead_expenses_l1308_130827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_existence_l1308_130896

theorem unique_m_existence : ∃! m : ℝ, ∃ a b : ℝ, 
  (2 : ℝ) ^ a = (5 : ℝ) ^ b ∧ 
  (2 : ℝ) ^ a = m ∧
  (5 : ℝ) ^ b = m ∧
  a + b = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_existence_l1308_130896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_at_distance_3_count_l1308_130812

/-- A lattice point in 3D space -/
structure LatticePoint3D where
  x : ℤ
  y : ℤ
  z : ℤ

/-- The squared distance of a point from the origin -/
def squaredDistance (p : LatticePoint3D) : ℤ :=
  p.x^2 + p.y^2 + p.z^2

/-- The set of lattice points with distance 3 from the origin -/
def latticePointsAtDistance3 : Set LatticePoint3D :=
  {p : LatticePoint3D | squaredDistance p = 9}

/-- Finite type instance for latticePointsAtDistance3 -/
instance : Fintype latticePointsAtDistance3 :=
  sorry

theorem lattice_points_at_distance_3_count :
  Fintype.card latticePointsAtDistance3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_at_distance_3_count_l1308_130812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_first_second_draw_l1308_130817

/-- A box containing products -/
structure Box where
  total : ℕ
  firstClass : ℕ
  secondClass : ℕ

/-- Event A: "the first draw is a first-class product" -/
noncomputable def eventA (box : Box) : ℝ :=
  box.firstClass / box.total

/-- Event B: "the second draw is a first-class product" -/
noncomputable def eventB (box : Box) : ℝ :=
  (box.firstClass - 1) / (box.total - 1)

/-- The probability of both events A and B occurring -/
noncomputable def eventAB (box : Box) : ℝ :=
  (box.firstClass / box.total) * ((box.firstClass - 1) / (box.total - 1))

/-- The conditional probability of event B given event A -/
noncomputable def conditionalProbability (box : Box) : ℝ :=
  eventAB box / eventA box

theorem conditional_probability_first_second_draw (box : Box) 
  (h1 : box.total = 4)
  (h2 : box.firstClass = 3)
  (h3 : box.secondClass = 1)
  (h4 : box.total = box.firstClass + box.secondClass) :
  conditionalProbability box = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_first_second_draw_l1308_130817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_approximation_l1308_130852

structure RandomSimulation where
  n : ℕ
  m : ℕ
  x : Fin n → ℝ
  y : Fin n → ℝ
  x_in_range : ∀ i, 0 ≤ x i ∧ x i ≤ 1
  y_in_range : ∀ i, 0 ≤ y i ∧ y i ≤ 1
  m_def : m = (Finset.filter (fun i => (x i)^2 + (y i)^2 < 1) (Finset.univ)).card

/-- Approximation of π using random simulation -/
theorem pi_approximation (sim : RandomSimulation) :
  ∃ ε > 0, |Real.pi - 4 * (sim.m : ℝ) / (sim.n : ℝ)| < ε := by
  sorry

#check pi_approximation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_approximation_l1308_130852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l1308_130891

noncomputable section

/-- The angle of inclination of a line given its equation -/
def angle_of_inclination (a b c : ℝ) : ℝ := Real.arctan (-a / b)

/-- The range of angles of inclination for the given family of lines -/
def angle_range : Set ℝ := Set.union (Set.Icc 0 (Real.pi / 6)) (Set.Icc (5 * Real.pi / 6) Real.pi)

/-- Theorem stating the range of angles of inclination for the given family of lines -/
theorem angle_of_inclination_range (θ : ℝ) :
  angle_of_inclination (Real.cos θ) (Real.sqrt 3) 1 ∈ angle_range := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l1308_130891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_covered_area_larger_l1308_130867

/-- Represents a calendar page -/
structure CalendarPage where
  area : ℝ

/-- Represents the overlap of two calendar pages -/
structure Overlap where
  covered_area : ℝ
  uncovered_area : ℝ

/-- The theorem stating that the covered area is larger than the uncovered area -/
theorem covered_area_larger
  (lower_page upper_page : CalendarPage)
  (overlap : Overlap)
  (h1 : lower_page.area = upper_page.area)
  (h2 : overlap.covered_area + overlap.uncovered_area = lower_page.area)
  (h3 : overlap.covered_area > 0)
  (h4 : overlap.uncovered_area > 0) :
  overlap.covered_area > overlap.uncovered_area := by
  sorry

#check covered_area_larger

end NUMINAMATH_CALUDE_ERRORFEEDBACK_covered_area_larger_l1308_130867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1308_130804

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 3

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the point P
def point_P : ℝ × ℝ := (2, 1)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem intersection_distance_sum :
  ∃ (A B : ℝ × ℝ),
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    distance point_P A + distance point_P B = Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1308_130804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_next_number_is_eight_l1308_130800

def sequenceList : List ℕ := [2, 16, 4, 14, 6, 12]

def increasing_by_two (list : List ℕ) : Prop :=
  ∀ i : ℕ, i < (list.length / 2 - 1) → list[2*i]! + 2 = list[2*i + 2]!

def decreasing_by_two (list : List ℕ) : Prop :=
  ∀ i : ℕ, i < (list.length / 2 - 1) → list[2*i + 1]! - 2 = list[2*i + 3]!

theorem next_number_is_eight (h1 : increasing_by_two sequenceList) (h2 : decreasing_by_two sequenceList) :
  8 = sequenceList[4]! + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_next_number_is_eight_l1308_130800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_products_l1308_130844

/-- A type representing a permutation of the numbers 1 through 9 -/
def Permutation9 := { p : Fin 9 → Fin 9 // Function.Bijective p }

/-- The product of three elements from a permutation -/
def tripleProduct (p : Permutation9) (i j k : Fin 3) : ℕ :=
  (p.val (3 * i) + 1) * (p.val (3 * j + 1) + 1) * (p.val (3 * k + 2) + 1)

/-- The sum of products for a given permutation -/
def sumOfProducts (p : Permutation9) : ℕ :=
  tripleProduct p 0 0 0 + tripleProduct p 1 1 1 + tripleProduct p 2 2 2

/-- The main theorem: the minimum sum of products is 214 -/
theorem min_sum_of_products :
  (∀ p : Permutation9, sumOfProducts p ≥ 214) ∧ (∃ p : Permutation9, sumOfProducts p = 214) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_products_l1308_130844
