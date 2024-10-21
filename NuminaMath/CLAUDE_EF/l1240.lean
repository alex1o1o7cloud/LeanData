import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_store_cost_l1240_124005

/-- Calculates the total cost of candies in a store with specific ratios of candy types. -/
theorem candy_store_cost (grape_count : ℕ) (candy_price : ℚ) : 
  grape_count = 24 →
  candy_price = 5/2 →
  (grape_count + grape_count/3 + 2*grape_count) * candy_price = 200 := by
  intros h_grape h_price
  -- Convert grape_count to ℚ for rational arithmetic
  have grape_count_q : ℚ := grape_count
  -- Substitute known values
  rw [h_grape, h_price]
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

-- We can use #eval to check our theorem, but it's not necessary for the proof
-- #eval candy_store_cost 24 (5/2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_store_cost_l1240_124005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_ratio_l1240_124049

/-- The sum of the first n terms of a geometric sequence with first term a and common ratio q -/
noncomputable def geometricSum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

/-- Theorem: For a geometric sequence with common ratio -5, the ratio of consecutive sums is -4 -/
theorem geometric_sequence_sum_ratio :
  ∀ (a : ℝ) (n : ℕ), 
  let q : ℝ := -5
  let S : ℕ → ℝ := λ k => geometricSum a q k
  S (n + 1) / S n = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_ratio_l1240_124049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_palindrome_count_l1240_124068

def is_valid_digit (d : ℕ) : Prop := d = 5 ∨ d = 6 ∨ d = 7

def is_odd (d : ℕ) : Bool := d % 2 = 1

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def count_palindromes : ℕ :=
  let valid_digits := [5, 6, 7]
  let odd_digits := valid_digits.filter is_odd
  (valid_digits.length ^ 3) * odd_digits.length

theorem palindrome_count :
  count_palindromes = 54 := by
  sorry

#eval count_palindromes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_palindrome_count_l1240_124068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_b_l1240_124044

def a : ℕ → ℝ
  | 0 => 1  -- Adding this case to cover Nat.zero
  | 1 => 1
  | 2 => 2
  | (n + 3) => 2 * a (n + 2) - a (n + 1) + 2

def b (n : ℕ) : ℝ := a (n + 1) - a n

theorem arithmetic_sequence_b :
  ∃ (d : ℝ), (∀ n : ℕ, b (n + 1) - b n = d) ∧ b 1 = 1 := by
  sorry

#eval a 0
#eval a 1
#eval a 2
#eval a 3
#eval b 1
#eval b 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_b_l1240_124044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l1240_124021

-- Define the line l
def line_l (t m : ℝ) : ℝ × ℝ := (3 * t, 4 * t + m)

-- Define the circle C
noncomputable def circle_C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ * Real.cos θ, 2 * Real.cos θ * Real.sin θ)

-- Define the chord length
noncomputable def chord_length (m : ℝ) : ℝ := Real.sqrt 3

-- Theorem statement
theorem intersection_chord_length (m : ℝ) :
  (∃ t₁ t₂ θ₁ θ₂,
    t₁ ≠ t₂ ∧
    line_l t₁ m = circle_C θ₁ ∧
    line_l t₂ m = circle_C θ₂ ∧
    chord_length m = Real.sqrt ((line_l t₁ m).1 - (line_l t₂ m).1)^2 + ((line_l t₁ m).2 - (line_l t₂ m).2)^2) →
  m = -13/6 ∨ m = -1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l1240_124021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_not_satisfied_count_l1240_124096

theorem inequality_not_satisfied_count : ∃ (S : Finset ℤ), 
  (∀ x ∈ S, ¬(4*x^2 + 16*x + 15 > 23)) ∧ 
  (∀ x : ℤ, x ∉ S → (4*x^2 + 16*x + 15 > 23)) ∧ 
  Finset.card S = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_not_satisfied_count_l1240_124096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_perpendicular_lines_l1240_124001

/-- A dihedral angle -/
structure DihedralAngle where
  -- Add necessary fields
  angle : Real

/-- A point inside a dihedral angle -/
structure PointInDihedralAngle (d : DihedralAngle) where
  -- Add necessary fields
  coords : Fin 3 → Real

/-- The plane angle of a dihedral angle -/
def planeAngle (d : DihedralAngle) : Real :=
  d.angle

/-- The angle formed by perpendicular lines from a point to the faces of a dihedral angle -/
def perpendicularLinesAngle (d : DihedralAngle) (p : PointInDihedralAngle d) : Real :=
  sorry

/-- Two angles are complementary if they sum to π/2 -/
def isComplementary (α β : Real) : Prop :=
  α + β = Real.pi / 2

theorem dihedral_angle_perpendicular_lines 
  (d : DihedralAngle) (p : PointInDihedralAngle d) : 
  (perpendicularLinesAngle d p = planeAngle d) ∨ 
  (isComplementary (perpendicularLinesAngle d p) (planeAngle d)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_perpendicular_lines_l1240_124001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_hands_at_4pm_hour_hand_rotation_minute_hand_rotation_l1240_124036

-- Define the clock and time-related constants
def hours_on_clock : ℕ := 12
def minutes_per_hour : ℕ := 60
def degrees_in_circle : ℕ := 360
def minutes_from_4pm_to_630pm : ℕ := 150

-- Define the rotation rates
def hour_hand_rotation_rate : ℚ := (degrees_in_circle : ℚ) / ((hours_on_clock * minutes_per_hour) : ℚ)
def minute_hand_rotation_rate : ℚ := (degrees_in_circle : ℚ) / (minutes_per_hour : ℚ)

-- Theorem for the angle between hour and minute hands at 4:00 PM
theorem angle_between_hands_at_4pm :
  (4 * (degrees_in_circle / hours_on_clock : ℚ)).floor = 120 := by sorry

-- Theorem for hour hand rotation from 4:00 PM to 6:30 PM
theorem hour_hand_rotation :
  ((minutes_from_4pm_to_630pm : ℚ) * hour_hand_rotation_rate).floor = 75 := by sorry

-- Theorem for minute hand rotation from 4:00 PM to 6:30 PM
theorem minute_hand_rotation :
  ((minutes_from_4pm_to_630pm : ℚ) * minute_hand_rotation_rate).floor = 900 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_hands_at_4pm_hour_hand_rotation_minute_hand_rotation_l1240_124036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nonagon_diagonal_intersection_probability_l1240_124041

/-- A regular nonagon is a 9-sided polygon with all sides and angles equal -/
structure RegularNonagon where

/-- A diagonal of a polygon is a line segment that connects two non-adjacent vertices -/
structure Diagonal (n : RegularNonagon) where

/-- Two diagonals intersect if they cross each other inside the polygon -/
def Intersect (n : RegularNonagon) (d1 d2 : Diagonal n) : Prop := sorry

/-- The probability of an event is the number of favorable outcomes divided by the total number of possible outcomes -/
noncomputable def Probability {α : Type} (event : Set α) (space : Set α) : ℚ := sorry

theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) :
  Probability {p : Diagonal n × Diagonal n | p.1 ≠ p.2 ∧ Intersect n p.1 p.2}
              {p : Diagonal n × Diagonal n | p.1 ≠ p.2} = 14/39 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nonagon_diagonal_intersection_probability_l1240_124041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_two_is_valid_input_l1240_124023

theorem expression_simplification (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 1) :
  (x^2 - 1) / (x + 2) / (1 - 3 / (x + 2)) = x + 1 := by
  sorry

-- Define the original expression
noncomputable def original_expression (x : ℝ) : ℝ :=
  (x^2 - 1) / (x + 2) / (1 - 3 / (x + 2))

-- Theorem stating that 2 is a valid input for the expression
theorem two_is_valid_input : original_expression 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_two_is_valid_input_l1240_124023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_is_pi_over_two_l1240_124019

theorem angle_sum_is_pi_over_two (θ₁ θ₂ : Real) 
  (h_acute₁ : 0 < θ₁ ∧ θ₁ < π / 2)
  (h_acute₂ : 0 < θ₂ ∧ θ₂ < π / 2)
  (h_eq : (Real.sin θ₁)^2020 / (Real.cos θ₂)^2018 + (Real.cos θ₁)^2020 / (Real.sin θ₂)^2018 = 1) :
  θ₁ + θ₂ = π / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_is_pi_over_two_l1240_124019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_empty_fraction_greater_than_one_l1240_124054

-- Define the function f
def f (x : ℝ) : ℝ := abs (2*x - 1) + 1

-- Define the set P (solution set of f(x) < 2)
def P : Set ℝ := {x | f x < 2}

-- Define the set Q (solution set of ||x|-2| < 1)
def Q : Set ℝ := {x | abs (abs x - 2) < 1}

-- Theorem 1: P ∩ Q = ∅
theorem intersection_empty : P ∩ Q = ∅ := by sorry

-- Theorem 2: For m > 1 and n ∈ P, (m+n)/(1+mn) > 1
theorem fraction_greater_than_one (m n : ℝ) (hm : m > 1) (hn : n ∈ P) : 
  (m + n) / (1 + m * n) > 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_empty_fraction_greater_than_one_l1240_124054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_factors_count_l1240_124071

theorem even_factors_count (n : ℕ) (h : n = 2^3 * 5^1 * 7^2) :
  (Finset.filter (λ x => x ∣ n ∧ Even x) (Finset.range (n + 1))).card = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_factors_count_l1240_124071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l1240_124083

/-- Definition of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Definition of perimeter for a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Definition of area for a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

/-- A rectangle ABCD divided into four identical squares with a perimeter of 160 cm has an area of 1024 square centimeters. -/
theorem rectangle_area (ABCD : Rectangle) 
  (p : Rectangle.perimeter ABCD = 160) 
  (h : ∃ (s : ℝ), ABCD.length = 4 * s ∧ ABCD.width = s) : 
  Rectangle.area ABCD = 1024 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l1240_124083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l1240_124002

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ (a₁ d : ℚ), ∀ n, a n = a₁ + (n - 1) * d

/-- Sum of first n terms of an arithmetic sequence -/
def S (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_ratio (a : ℕ → ℚ) :
  arithmetic_sequence a →
  (a 5) / (a 3) = 2 →
  (S a 9) / (S a 5) = 18 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l1240_124002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_star_intersections_star_2018_25_l1240_124076

/-- Definition of a regular (n; k)-star -/
def regular_star (n k : ℕ) : Prop :=
  Nat.Coprime n k ∧ n ≥ 5 ∧ k < n / 2

/-- Number of self-intersections in a regular (n; k)-star -/
def num_intersections (n k : ℕ) : ℕ := n * (k - 1)

/-- Theorem stating the number of self-intersections in a regular (n; k)-star -/
theorem regular_star_intersections (n k : ℕ) (h : regular_star n k) :
  num_intersections n k = n * (k - 1) := by
  sorry

/-- Theorem for the specific case of (2018; 25)-star -/
theorem star_2018_25 :
  regular_star 2018 25 ∧ num_intersections 2018 25 = 48432 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_star_intersections_star_2018_25_l1240_124076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_m_in_range_l1240_124025

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - 2*m*x + 3*m)

theorem f_increasing_iff_m_in_range :
  ∀ m : ℝ, (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂) ↔ -1 < m ∧ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_m_in_range_l1240_124025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1240_124039

-- Sequence A
def seq_a : ℕ → ℚ
  | 0 => 2
  | n + 1 => seq_a n + n + 1

-- Sequence B
def seq_b : ℕ → ℚ
  | 0 => 1
  | n + 1 => 3 * seq_b n + 2

-- Sequence C
def S : ℕ → ℚ
  | n => 3^n + 1/2

def seq_c : ℕ → ℚ
  | 0 => S 0
  | n + 1 => S (n + 1) - S n

-- Sequence D
def seq_d : ℕ → ℚ
  | 0 => 1
  | n + 1 => (2 * seq_d n) / (2 + seq_d n)

theorem sequence_properties :
  (seq_a 2 = 7) ∧
  (seq_b 3 = 53) ∧
  (¬ ∃ r : ℚ, ∀ n : ℕ, n > 0 → seq_c (n + 1) = r * seq_c n) ∧
  (seq_d 4 ≠ 1/5) := by
  sorry

#eval seq_a 2
#eval seq_b 3
#eval seq_d 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1240_124039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_calculation_l1240_124090

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def isValidTriangle (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi ∧ t.A > 0 ∧ t.B > 0 ∧ t.C > 0

noncomputable def satisfiesCondition (t : Triangle) : Prop :=
  Real.tan t.B * (Real.cos t.A - Real.cos t.C) = Real.sin t.C - Real.sin t.A

-- Define the theorem
theorem isosceles_triangle (t : Triangle) 
  (h1 : isValidTriangle t) 
  (h2 : satisfiesCondition t) : 
  t.A = t.C := by sorry

-- Define additional conditions for perimeter calculation
def additionalConditions (t : Triangle) : Prop :=
  t.B = Real.pi/6 ∧ (1/2 * t.a * t.c * Real.sin t.B) = 4

-- Define the perimeter calculation
def perimeter (t : Triangle) : Real :=
  t.a + t.b + t.c

-- Define the theorem for perimeter
theorem perimeter_calculation (t : Triangle) 
  (h1 : isValidTriangle t) 
  (h2 : satisfiesCondition t) 
  (h3 : additionalConditions t) : 
  perimeter t = 8 + 2 * Real.sqrt 6 - 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_calculation_l1240_124090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inscribed_sphere_half_volume_l1240_124050

-- Define the truncated cone
structure TruncatedCone where
  R₁ : ℝ  -- radius of the larger base
  R₂ : ℝ  -- radius of the smaller base
  H : ℝ   -- height of the cone
  h : R₁ > R₂ -- condition that R₁ is greater than R₂

-- Define the inscribed sphere
structure InscribedSphere where
  R : ℝ  -- radius of the sphere

-- Define the relationship between the cone and the sphere
def sphereInscribedInCone (c : TruncatedCone) (s : InscribedSphere) : Prop :=
  c.H = 2 * s.R

-- Define the volumes
noncomputable def volumeCone (c : TruncatedCone) : ℝ :=
  (1/3) * Real.pi * c.H * (c.R₁^2 + c.R₁*c.R₂ + c.R₂^2)

noncomputable def volumeSphere (s : InscribedSphere) : ℝ :=
  (4/3) * Real.pi * s.R^3

-- Theorem statement
theorem angle_of_inscribed_sphere_half_volume 
  (c : TruncatedCone) (s : InscribedSphere) 
  (h₁ : sphereInscribedInCone c s) 
  (h₂ : volumeSphere s = (1/2) * volumeCone c) :
  Real.arctan 2 = Real.arctan ((c.R₁ - c.R₂) / c.H) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inscribed_sphere_half_volume_l1240_124050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_2014_l1240_124017

theorem function_value_2014 (f : ℕ → ℕ) 
  (h1 : ∀ n, f (f n) + f n = 2 * n + 3)
  (h2 : f 0 = 1) : 
  f 2014 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_2014_l1240_124017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_with_late_fees_l1240_124077

/-- Calculates the final amount due on a loan with compound interest --/
def final_amount (initial_amount : ℝ) (interest_rate : ℝ) (num_periods : ℕ) : ℝ :=
  initial_amount * (1 + interest_rate) ^ num_periods

/-- Theorem: The final amount due on a $500 loan with 2% interest compounded every 30 days for 90 days is approximately $530.60 --/
theorem loan_with_late_fees (ε : ℝ) (ε_pos : ε > 0) : 
  ∃ (result : ℝ), abs (final_amount 500 0.02 3 - result) < ε ∧ abs (result - 530.60) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_with_late_fees_l1240_124077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l1240_124024

/-- The curve function f(x) = x^3 + 11 -/
def f (x : ℝ) : ℝ := x^3 + 11

/-- The point P on the curve -/
def P : ℝ × ℝ := (1, 12)

/-- The slope of the tangent line at point P -/
noncomputable def m : ℝ := 3 * P.1^2

/-- The y-intercept of the tangent line -/
noncomputable def b : ℝ := P.2 - m * P.1

/-- The x-intercept of the tangent line -/
noncomputable def x_intercept : ℝ := -b / m

/-- The y-intercept of the tangent line -/
noncomputable def y_intercept : ℝ := b

/-- The area of the triangle formed by the tangent line and the coordinate axes -/
noncomputable def triangle_area : ℝ := (1/2) * x_intercept * y_intercept

theorem tangent_triangle_area : triangle_area = 27/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l1240_124024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_inequality_l1240_124008

def f (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => 1 / (i + 1 : ℚ))

theorem harmonic_sum_inequality (n : ℕ) :
  f (2^n) > (n + 2 : ℚ) / 2 :=
by
  have h1 : f 2 = 3/2 := by sorry
  have h2 : f 4 > 2 := by sorry
  have h3 : f 8 > 5/2 := by sorry
  have h4 : f 16 > 3 := by sorry
  have h5 : f 32 > 7/2 := by sorry
  sorry

#eval f 2
#eval f 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_inequality_l1240_124008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_foci_l1240_124078

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 5)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 24

/-- The first focus of the ellipse -/
def focus1 : ℝ × ℝ := (4, 5)

/-- The second focus of the ellipse -/
def focus2 : ℝ × ℝ := (-6, 9)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_between_foci :
  distance focus1 focus2 = 2 * Real.sqrt 29 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_foci_l1240_124078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_force_ratio_on_spinning_planet_l1240_124013

/-- The ratio of normal forces experienced by people at the equator and North Pole -/
noncomputable def normal_force_ratio (R ω g : ℝ) : ℝ :=
  1 - R * ω^2 / g

/-- Theorem stating the ratio of normal forces on a spinning planet -/
theorem normal_force_ratio_on_spinning_planet
  (R ω g : ℝ)
  (h_R : R > 0)
  (h_ω : ω ≥ 0)
  (h_g : g > 0) :
  normal_force_ratio R ω g = 1 - R * ω^2 / g :=
by
  -- Unfold the definition of normal_force_ratio
  unfold normal_force_ratio
  -- The equation is now trivially true by definition
  rfl

#check normal_force_ratio_on_spinning_planet

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_force_ratio_on_spinning_planet_l1240_124013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_shift_l1240_124092

theorem sin_graph_shift (x : ℝ) : 
  Real.sin (2 * (x + π / 6)) = Real.sin (2 * x + π / 3) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_shift_l1240_124092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_robot_packages_100_boxes_per_hour_l1240_124010

/-- The number of boxes a worker can package per hour -/
noncomputable def worker_boxes_per_hour : ℝ := 20

/-- The number of boxes a robot can package per hour -/
noncomputable def robot_boxes_per_hour : ℝ := 5 * worker_boxes_per_hour

/-- The time it takes for 4 workers to package 1600 boxes -/
noncomputable def worker_time : ℝ := 1600 / (4 * worker_boxes_per_hour)

/-- The time it takes for 1 robot to package 1600 boxes -/
noncomputable def robot_time : ℝ := 1600 / robot_boxes_per_hour

/-- The theorem stating that a robot can package 100 boxes per hour -/
theorem robot_packages_100_boxes_per_hour :
  robot_boxes_per_hour = 100 ∧
  worker_time - robot_time = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_robot_packages_100_boxes_per_hour_l1240_124010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_l1240_124070

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles in radians
  (a b c : ℝ)  -- Side lengths

-- Define our specific triangle
noncomputable def our_triangle : Triangle :=
{ A := Real.pi/6,  -- 30° in radians
  B := 0,    -- We don't know B yet, so we leave it as 0
  C := 0,    -- We don't know C yet, so we leave it as 0
  a := 6,
  b := 6 * Real.sqrt 3,
  c := 0     -- We don't need c for this problem
}

-- State the theorem
theorem angle_B_measure (t : Triangle) (h : t = our_triangle) :
  t.B = Real.pi/3 ∨ t.B = 2*Real.pi/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_l1240_124070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1240_124035

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi / 2 + 2 * x) - 5 * Real.sin x

-- State the theorem
theorem f_max_value : 
  ∀ x : ℝ, f x ≤ 17/8 ∧ ∃ y : ℝ, f y = 17/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1240_124035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_parameters_l1240_124056

/-- The probability density function of a normal distribution -/
noncomputable def normal_pdf (μ σ : ℝ) (x : ℝ) : ℝ :=
  1 / (σ * Real.sqrt (2 * Real.pi)) * Real.exp (-(x - μ)^2 / (2 * σ^2))

/-- The given probability density function -/
noncomputable def given_pdf (x : ℝ) : ℝ :=
  1 / Real.sqrt (8 * Real.pi) * Real.exp (x^2 / 8)

/-- Theorem stating the existence of μ and σ that satisfy the given conditions -/
theorem normal_distribution_parameters :
  ∃ (μ σ : ℝ), σ > 0 ∧ ∀ x, given_pdf x = normal_pdf μ σ x ∧ μ = 0 ∧ σ = 2 := by
  -- We'll use μ = 0 and σ = 2 as our witnesses
  use 0, 2
  -- Now we prove the three conjuncts
  constructor
  · -- Prove σ > 0
    norm_num
  · -- Prove the equality of PDFs and the values of μ and σ
    intro x
    constructor
    · -- Prove equality of PDFs
      simp [given_pdf, normal_pdf]
      -- The rest of the proof would involve algebraic manipulation
      sorry
    · -- Prove μ = 0 and σ = 2
      simp

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_parameters_l1240_124056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1240_124009

-- Define the variables and conditions
variable (a b : ℝ)
variable (h1 : b > a)
variable (h2 : a > 0)
variable (h3 : a * b = 2)

-- Define the expression
noncomputable def f (a b : ℝ) : ℝ := (a^2 + b^2) / (a - b)

-- State the theorem
theorem f_range :
  ∀ x : ℝ, f a b ≤ x → x ≤ -4 :=
by
  sorry

#check f_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1240_124009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_international_society_theorem_l1240_124046

theorem international_society_theorem (S : Finset ℕ) (f : ℕ → Fin 6) 
  (h1 : S.card = 1978) (h2 : ∀ n, n ∈ S → n ≥ 1 ∧ n ≤ 1978) :
  ∃ x ∈ S, (∃ y z, y ∈ S ∧ z ∈ S ∧ f y = f z ∧ f x = f y ∧ x = y + z) ∨ 
           (∃ y, y ∈ S ∧ f x = f y ∧ x = 2 * y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_international_society_theorem_l1240_124046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_and_intersecting_l1240_124079

noncomputable def circle_C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

noncomputable def point_P : ℝ × ℝ := (Real.sqrt 3, 1)

theorem line_tangent_and_intersecting :
  ∃ (l : Set (ℝ × ℝ)), 
    (point_P ∈ l) ∧ 
    (point_P ∈ circle_C) ∧
    (∃ (p : ℝ × ℝ), p ∈ l ∧ p ∈ circle_C) ∧
    (∀ (p : ℝ × ℝ), p ∈ l ∧ p ∈ circle_C → p = point_P) :=
by
  sorry

#check line_tangent_and_intersecting

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_and_intersecting_l1240_124079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_zero_exactly_one_zero_l1240_124063

-- Define the function f(x) for a given 'a'
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

-- Theorem 1: Maximum value of f(x) when a = 0
theorem max_value_when_a_zero :
  ∃ (M : ℝ), M = -1 ∧ ∀ (x : ℝ), x > 0 → f 0 x ≤ M := by
  sorry

-- Theorem 2: Condition for f(x) to have exactly one zero
theorem exactly_one_zero (a : ℝ) :
  (∃! (x : ℝ), x > 0 ∧ f a x = 0) ↔ a > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_zero_exactly_one_zero_l1240_124063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_allowance_l1240_124097

/-- John's weekly allowance in dollars -/
def weekly_allowance : ℚ := 2.8125

/-- Amount spent at the arcade as a fraction of the weekly allowance -/
def arcade_spend : ℚ := 3/5

/-- Amount spent at the toy store as a fraction of the remaining allowance after arcade -/
def toy_store_spend : ℚ := 1/3

/-- Amount spent at the candy store in dollars -/
def candy_store_spend : ℚ := 3/4

theorem johns_allowance :
  weekly_allowance * (1 - arcade_spend) * (1 - toy_store_spend) = candy_store_spend ∧
  weekly_allowance = 2.8125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_allowance_l1240_124097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_focal_length_l1240_124088

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0

/-- The area of the triangle formed by the origin and the intersection points
    of the line x=a with the hyperbola's asymptotes -/
def triangle_area (h : Hyperbola) : ℝ := h.a * h.b

/-- The focal length of a hyperbola -/
noncomputable def focal_length (h : Hyperbola) : ℝ := 2 * (h.a^2 + h.b^2).sqrt

/-- Theorem: If the area of the triangle is 8, then the minimum focal length is 8 -/
theorem min_focal_length (h : Hyperbola) (h_area : triangle_area h = 8) :
  ∃ (min_focal : ℝ), min_focal = 8 ∧ ∀ (h' : Hyperbola), triangle_area h' = 8 → focal_length h' ≥ min_focal :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_focal_length_l1240_124088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_lattice_points_on_curve_l1240_124012

/-- A lattice point is a point on the plane with integer coordinates. -/
def LatticePoint (p : ℝ × ℝ) : Prop :=
  ∃ (x y : ℤ), p = (↑x, ↑y)

/-- The curve is defined by the equation y = (1/5)(x^2 - x + 1). -/
def OnCurve (p : ℝ × ℝ) : Prop :=
  p.2 = (1/5) * (p.1^2 - p.1 + 1)

/-- Theorem: There are no lattice points on the curve y = (1/5)(x^2 - x + 1). -/
theorem no_lattice_points_on_curve :
  ¬∃ (p : ℝ × ℝ), LatticePoint p ∧ OnCurve p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_lattice_points_on_curve_l1240_124012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_at_one_l1240_124061

noncomputable def P (x : ℝ) : ℝ := 3 * x^3 - 5 * x^2 + 2 * x - 6

noncomputable def coeffMean (P : ℝ → ℝ) : ℝ :=
  (3 + (-5) + 2 + (-6)) / 4

noncomputable def Q (x : ℝ) : ℝ :=
  let m := coeffMean P
  m * x^3 + m * x^2 + m * x + m

theorem Q_at_one : Q 1 = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_at_one_l1240_124061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l1240_124006

theorem cos_alpha_value (α : ℝ) 
  (h1 : 0 < α) (h2 : α < π/2) 
  (h3 : Real.cos (π/3 + α) = 1/3) : 
  Real.cos α = (2 * Real.sqrt 6 + 1) / 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l1240_124006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_union_B_eq_B_l1240_124059

-- Define set A
def A : Set ℕ := {n | ∃ k, 2^n + 3^n = 5 * k}

-- Define set B
def B : Set ℕ := {m | ∃ u v : ℕ, m = u^2 - v^2}

-- Theorem statement
theorem A_union_B_eq_B : A ∪ B = B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_union_B_eq_B_l1240_124059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1240_124034

open Real

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := Real.sqrt 3 * sin (2 * x) + 2 * (cos x)^2 + m

-- State the theorem
theorem function_properties :
  ∃ m : ℝ, (∀ x ∈ Set.Icc 0 (π / 2), f x m ≥ 3) ∧
           (∃ x₀ ∈ Set.Icc 0 (π / 2), f x₀ m = 3) ∧
           m = 3 ∧
           (∀ a : ℝ, ∃ x_max ∈ Set.Icc a (a + π), 
             ∀ x ∈ Set.Icc a (a + π), f x m ≤ f x_max m) ∧
           (∀ a : ℝ, ∃ x_max ∈ Set.Icc a (a + π), f x_max m = 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1240_124034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_pi_sixths_l1240_124007

theorem cos_seven_pi_sixths : Real.cos (7 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_pi_sixths_l1240_124007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_ratio_l1240_124072

theorem race_time_ratio : 
  ∀ (total_time walking_time : ℕ),
  total_time = 21 →
  walking_time = 9 →
  let jogging_time := total_time - walking_time
  (jogging_time : ℚ) / (walking_time : ℚ) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_ratio_l1240_124072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_initial_number_l1240_124047

theorem largest_initial_number :
  ∃ (a₁ a₂ a₃ a₄ a₅ : ℕ),
    89 + a₁ + a₂ + a₃ + a₄ + a₅ = 100 ∧
    ¬(a₁ ∣ 89) ∧
    ¬(a₂ ∣ (89 + a₁)) ∧
    ¬(a₃ ∣ (89 + a₁ + a₂)) ∧
    ¬(a₄ ∣ (89 + a₁ + a₂ + a₃)) ∧
    ¬(a₅ ∣ (89 + a₁ + a₂ + a₃ + a₄)) ∧
    ∀ n > 89, ¬∃ (b₁ b₂ b₃ b₄ b₅ : ℕ),
      n + b₁ + b₂ + b₃ + b₄ + b₅ = 100 ∧
      ¬(b₁ ∣ n) ∧
      ¬(b₂ ∣ (n + b₁)) ∧
      ¬(b₃ ∣ (n + b₁ + b₂)) ∧
      ¬(b₄ ∣ (n + b₁ + b₂ + b₃)) ∧
      ¬(b₅ ∣ (n + b₁ + b₂ + b₃ + b₄)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_initial_number_l1240_124047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trihedral_angle_right_dihedral_implies_right_face_l1240_124051

/-- A trihedral angle --/
structure TrihedralAngle where
  /-- Face angles of the trihedral angle --/
  face_angles : Fin 3 → ℝ
  /-- Dihedral angles of the trihedral angle --/
  dihedral_angles : Fin 3 → ℝ

/-- A right angle in radians --/
noncomputable def right_angle : ℝ := Real.pi / 2

/-- Theorem: If all dihedral angles of a trihedral angle are right angles, 
    then all its face angles are right angles --/
theorem trihedral_angle_right_dihedral_implies_right_face 
  (t : TrihedralAngle) 
  (h : ∀ i : Fin 3, t.dihedral_angles i = right_angle) : 
  ∀ i : Fin 3, t.face_angles i = right_angle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trihedral_angle_right_dihedral_implies_right_face_l1240_124051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1240_124027

open Real

theorem trigonometric_equation_solution (α β : ℝ) 
  (h1 : Real.sin (π - α) = Real.sqrt 2 * Real.cos ((3 * π) / 2 + β))
  (h2 : Real.sqrt 3 * Real.cos (-α) = - Real.sqrt 2 * Real.cos (π - β))
  (h3 : 0 < α) (h4 : α < π)
  (h5 : 0 < β) (h6 : β < π) :
  ((α = π / 4 ∧ β = π / 6) ∨ (α = 3 * π / 4 ∧ β = 5 * π / 6)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1240_124027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_and_die_probability_l1240_124066

-- Define the probability of heads for the biased coin
noncomputable def p_heads : ℝ := 0.3

-- Define the number of coin flips
def num_flips : ℕ := 2

-- Define the number of sides on the die
def die_sides : ℕ := 6

-- Define the probability of rolling a 6 on the die
noncomputable def p_roll_6 : ℝ := 1 / die_sides

-- Theorem statement
theorem coin_and_die_probability :
  let p_at_least_one_head := 1 - (1 - p_heads) ^ num_flips
  p_at_least_one_head * p_roll_6 = 0.51 / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_and_die_probability_l1240_124066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l1240_124067

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of line l1: (a - 1)x + y - 1 = 0 -/
noncomputable def slope_l1 (a : ℝ) : ℝ := -(a - 1)

/-- The slope of line l2: 3x + ay + 2 = 0 -/
noncomputable def slope_l2 (a : ℝ) : ℝ := -3 / a

theorem perpendicular_lines (a : ℝ) : 
  perpendicular (slope_l1 a) (slope_l2 a) → a = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l1240_124067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_weight_loss_l1240_124089

/-- The weight the student needs to lose to weigh twice as much as his sister -/
noncomputable def weight_to_lose (total_weight student_weight : ℝ) : ℝ :=
  student_weight - 2 * (total_weight - student_weight) / 3

theorem student_weight_loss (total_weight student_weight : ℝ) 
  (h1 : total_weight = 116)
  (h2 : student_weight = 79) :
  weight_to_lose total_weight student_weight = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_weight_loss_l1240_124089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_hemisphere_equal_volume_l1240_124073

/-- The radius of a sphere that forms a hemisphere of radius 3∛2 cm with equal volume --/
noncomputable def sphere_radius : ℝ := 3

/-- The radius of the hemisphere --/
noncomputable def hemisphere_radius : ℝ := 3 * Real.rpow 2 (1/3)

/-- The volume of a sphere with radius r --/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

/-- The volume of a hemisphere with radius r --/
noncomputable def hemisphere_volume (r : ℝ) : ℝ := (2/3) * Real.pi * r^3

theorem sphere_hemisphere_equal_volume :
  sphere_volume sphere_radius = hemisphere_volume hemisphere_radius :=
by sorry

#check sphere_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_hemisphere_equal_volume_l1240_124073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1240_124074

/-- Sum of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n / 2 * (2 * a₁ + (n - 1) * d)

/-- Theorem: The sum of the first 12 terms of an arithmetic sequence 
    starting at 4 with common difference 5 is 378 -/
theorem arithmetic_sequence_sum : 
  arithmetic_sum 4 5 12 = 378 := by
  -- Unfold the definition of arithmetic_sum
  unfold arithmetic_sum
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1240_124074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_axis_of_sin_function_l1240_124040

theorem symmetric_axis_of_sin_function (x : ℝ) :
  x = -π / 12 → Real.sin (3 * x + 3 * π / 4) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_axis_of_sin_function_l1240_124040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l1240_124062

theorem vector_equation_solution :
  ∃! (t s : ℝ), (⟨3, 1⟩ : ℝ × ℝ) + t • (⟨4, -6⟩ : ℝ × ℝ) = (⟨0, 2⟩ : ℝ × ℝ) + s • (⟨-3, 5⟩ : ℝ × ℝ) ∧
  t = 6 ∧ s = -9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l1240_124062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_transformation_l1240_124095

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(√(x+1))
def domain_f_sqrt (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 3

-- Define the domain of f(1-x)
def domain_f_minus (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 0

-- Theorem statement
theorem domain_transformation :
  (∀ x, f (Real.sqrt (x + 1)) ≠ 0 ↔ domain_f_sqrt x) →
  (∀ x, f (1 - x) ≠ 0 ↔ domain_f_minus x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_transformation_l1240_124095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_exist_l1240_124037

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions for the specific triangle
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 14 ∧ t.c = 16 ∧ t.A = 45 * Real.pi / 180

-- Theorem statement
theorem two_solutions_exist (t : Triangle) 
  (h : triangle_conditions t) : 
  ∃ (t1 t2 : Triangle), 
    t1 ≠ t2 ∧ 
    triangle_conditions t1 ∧ 
    triangle_conditions t2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_exist_l1240_124037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l1240_124058

/-- Pyramid with square base and vertex equidistant from base vertices -/
structure SquareBasePyramid where
  -- Side length of the square base
  baseSide : ℝ
  -- Angle between any two adjacent base vertices and the apex
  apexAngle : ℝ

/-- Volume of the square base pyramid -/
noncomputable def pyramidVolume (p : SquareBasePyramid) : ℝ :=
  (4 / 3) * Real.sqrt (Real.tan (p.apexAngle / 2) ^ 2 + 1)

/-- Theorem: Volume of the specific pyramid -/
theorem specific_pyramid_volume :
  ∀ (p : SquareBasePyramid),
  p.baseSide = 2 →
  pyramidVolume p = (4 / 3) * Real.sqrt (Real.tan (p.apexAngle / 2) ^ 2 + 1) :=
by
  intro p h
  unfold pyramidVolume
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l1240_124058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_groups_school_l1240_124064

theorem max_groups_school (total_boys total_girls num_classes : ℕ) 
  (min_boys_per_class min_girls_per_class : ℕ) :
  total_boys = 300 →
  total_girls = 300 →
  num_classes = 5 →
  min_boys_per_class = 33 →
  min_girls_per_class = 33 →
  (∀ i : Fin num_classes, 
    (total_boys / num_classes : ℕ) + (total_girls / num_classes : ℕ) = 
    (total_boys + total_girls) / num_classes) →
  (∀ i : Fin num_classes, 
    (total_boys / num_classes : ℕ) ≥ min_boys_per_class ∧
    (total_girls / num_classes : ℕ) ≥ min_girls_per_class) →
  ∃ max_groups : ℕ, 
    (∀ arrangement : Fin num_classes → ℕ × ℕ,
      (∀ i : Fin num_classes, 
        (arrangement i).1 + (arrangement i).2 = (total_boys + total_girls) / num_classes ∧
        (arrangement i).1 ≥ min_boys_per_class ∧
        (arrangement i).2 ≥ min_girls_per_class) →
      (Finset.sum (Finset.univ : Finset (Fin num_classes)) 
        (λ i => min (arrangement i).1 (arrangement i).2)) ≥ max_groups) ∧
    max_groups = 192 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_groups_school_l1240_124064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_metrics_correctness_l1240_124053

/-- Represents Elaine's financial situation over two years --/
structure FinancialSituation where
  last_year_job_income : ℝ
  last_year_side_job_income : ℝ
  last_year_rent_percentage : ℝ
  last_year_groceries_utilities_percentage : ℝ
  last_year_healthcare_percentage : ℝ
  this_year_job_income_raise : ℝ
  this_year_side_job_income_multiplier : ℝ
  this_year_rent_percentage : ℝ
  this_year_groceries_utilities_percentage : ℝ
  this_year_healthcare_percentage : ℝ
  this_year_healthcare_cost_increase : ℝ

/-- Calculates financial metrics based on Elaine's situation --/
def calculate_metrics (s : FinancialSituation) : 
  ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ := 
  sorry

/-- Theorem stating the correctness of the calculated metrics --/
theorem metrics_correctness (s : FinancialSituation) 
  (h1 : s.last_year_job_income = 20000)
  (h2 : s.last_year_side_job_income = 5000)
  (h3 : s.last_year_rent_percentage = 0.1)
  (h4 : s.last_year_groceries_utilities_percentage = 0.2)
  (h5 : s.last_year_healthcare_percentage = 0.15)
  (h6 : s.this_year_job_income_raise = 0.15)
  (h7 : s.this_year_side_job_income_multiplier = 2)
  (h8 : s.this_year_rent_percentage = 0.3)
  (h9 : s.this_year_groceries_utilities_percentage = 0.25)
  (h10 : s.this_year_healthcare_percentage = 0.15)
  (h11 : s.this_year_healthcare_cost_increase = 0.1) :
  let (rent_increase, groceries_increase, healthcare_increase, 
       last_year_savings_percentage, this_year_savings_percentage,
       last_year_expenses_ratio, this_year_expenses_ratio,
       savings_decrease, savings_decrease_percentage) := calculate_metrics s
  rent_increase = 2.96 ∧ 
  groceries_increase = 0.65 ∧ 
  healthcare_increase = 0.452 ∧
  last_year_savings_percentage = 0.55 ∧
  this_year_savings_percentage = 0.285 ∧
  last_year_expenses_ratio = 0.45 ∧
  this_year_expenses_ratio = 0.715 ∧
  savings_decrease = 4345 ∧
  savings_decrease_percentage = 0.265 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_metrics_correctness_l1240_124053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_boys_or_girls_at_rink_l1240_124030

structure Student where
  name : String

structure Time where
  value : Nat

def MetAtRink (b g : Student) : Prop := sorry
def AtRink (s : Student) (t : Time) : Prop := sorry

structure ClassRink where
  students : Set Student
  boys : Set Student
  girls : Set Student
  visited_rink : ∀ s : Student, s ∈ students → s ∈ boys ∪ girls
  boys_girls_partition : boys ∪ girls = students ∧ boys ∩ girls = ∅
  all_met : ∀ b g, b ∈ boys → g ∈ girls → MetAtRink b g

theorem all_boys_or_girls_at_rink (c : ClassRink) :
  ∃ t : Time, (∀ b : Student, b ∈ c.boys → AtRink b t) ∨ 
                (∀ g : Student, g ∈ c.girls → AtRink g t) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_boys_or_girls_at_rink_l1240_124030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l1240_124022

def S : ℚ :=
  (2014 : ℚ) / 3 * ((1 : ℚ) / 2 - 1 / 2015)

theorem sum_remainder : 
  (Int.floor (S + 1/2) + 1) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l1240_124022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_a_sequence_l1240_124081

noncomputable def a (n : ℕ+) : ℝ := (n - Real.sqrt 97) / (n - Real.sqrt 98)

theorem max_min_a_sequence :
  let first_30 := Finset.range 30
  (∀ i ∈ first_30, a ⟨i + 1, Nat.succ_pos i⟩ ≤ a 10) ∧
  (∀ i ∈ first_30, a ⟨i + 1, Nat.succ_pos i⟩ ≥ a 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_a_sequence_l1240_124081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slant_angle_for_area_ratio_2pi_l1240_124014

/-- For a cone with radius r, height h, and slant height l -/
structure Cone where
  r : ℝ
  h : ℝ
  l : ℝ

/-- The angle between the slant height and the axis of the cone -/
noncomputable def slant_angle (c : Cone) : ℝ := Real.arccos (c.h / c.l)

/-- The ratio of lateral surface area to cross-sectional area -/
noncomputable def area_ratio (c : Cone) : ℝ := (Real.pi * c.r * c.l) / (c.r * c.h)

theorem cone_slant_angle_for_area_ratio_2pi (c : Cone) 
  (h_ratio : area_ratio c = 2 * Real.pi) : 
  slant_angle c = Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slant_angle_for_area_ratio_2pi_l1240_124014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_circle_center_l1240_124048

/-- The equation of the circle in polar coordinates -/
def circle_equation (ρ θ : Real) : Prop := ρ = 2 * Real.cos (θ + Real.pi/4)

/-- The center of the circle in polar coordinates -/
noncomputable def circle_center : Real × Real := (1, -Real.pi/4)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center :
  let (r, θ) := circle_center
  circle_equation r θ ∧ 
  ∀ (ρ φ : Real), circle_equation ρ φ → 
    (ρ * Real.cos φ - r * Real.cos θ)^2 + (ρ * Real.sin φ - r * Real.sin θ)^2 ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_circle_center_l1240_124048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_B_l1240_124003

-- Define the inverse proportion function
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

-- Define the linear function
noncomputable def linear_function (a : ℝ) (x : ℝ) : ℝ := a * x

-- Define the theorem
theorem intersection_point_B 
  (k a : ℝ) 
  (hk : k ≠ 0) 
  (ha : a ≠ 0) 
  (h_intersect_A : inverse_proportion k (-1) = linear_function a (-1) ∧ 
                   inverse_proportion k (-1) = 2) 
  (h_intersect_B : ∃ x y : ℝ, inverse_proportion k x = linear_function a x ∧ 
                               inverse_proportion k x = y ∧ 
                               (x, y) ≠ (-1, 2)) :
  ∃ x y : ℝ, inverse_proportion k x = linear_function a x ∧ 
             inverse_proportion k x = y ∧ 
             (x, y) = (1, -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_B_l1240_124003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_diameter_calculation_l1240_124026

/-- The volume of the lead wire in cubic decimeters -/
def volume : ℝ := 2.2

/-- The length of the wire in meters -/
def length : ℝ := 112.04507993669432

/-- The diameter of the wire in centimeters -/
def diameter : ℝ := 0.50016

/-- Theorem stating that the given volume and length result in the specified diameter -/
theorem wire_diameter_calculation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ 
  |diameter - 2 * Real.sqrt ((volume * 1000) / (Real.pi * length * 100))| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_diameter_calculation_l1240_124026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nguyen_fabric_needed_l1240_124038

-- Define the constants
def fabric_per_pair : ℝ := 8.5
def pairs_needed : ℕ := 7
def fabric_available_yards : ℝ := 3.5
def yards_to_feet : ℝ := 3

-- Theorem statement
theorem nguyen_fabric_needed : 
  fabric_per_pair * (pairs_needed : ℝ) - fabric_available_yards * yards_to_feet = 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nguyen_fabric_needed_l1240_124038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fudge_distribution_l1240_124082

theorem fudge_distribution (y : ℕ) (total_pounds : ℚ) : 
  total_pounds = 15.5 →
  (∃ (k : ℕ), 
    9 * k + 7 * k + y * k = (total_pounds * 16).floor ∧
    9 * k = 72 ∧
    7 * k = 56 ∧
    y * k = 120 ∧
    9 * k - 7 * k = 16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fudge_distribution_l1240_124082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_condition_condition_implies_right_triangle_l1240_124093

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_A : ℝ
  angle_B : ℝ
  angle_C : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_angle_A : 0 < angle_A
  pos_angle_B : 0 < angle_B
  pos_angle_C : 0 < angle_C
  sum_angles : angle_A + angle_B + angle_C = π

-- Define the theorem
theorem right_triangle_condition (t : Triangle) :
  t.a * Real.cos t.angle_B + t.a * Real.cos t.angle_C = t.b + t.c →
  t.a^2 = t.b^2 + t.c^2 :=
by
  sorry

-- Define what it means for a triangle to be right-angled
def is_right_triangle (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2 ∨ t.b^2 = t.a^2 + t.c^2 ∨ t.c^2 = t.a^2 + t.b^2

-- State the main theorem
theorem condition_implies_right_triangle (t : Triangle) :
  t.a * Real.cos t.angle_B + t.a * Real.cos t.angle_C = t.b + t.c →
  is_right_triangle t :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_condition_condition_implies_right_triangle_l1240_124093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_one_sixth_l1240_124004

/-- The area between the curves y = √x and y = x from x = 0 to x = 1 -/
noncomputable def area_between_curves : ℝ :=
  ∫ x in (0:ℝ)..1, (Real.sqrt x - x)

/-- Theorem stating that the area between the curves y = √x and y = x is 1/6 -/
theorem area_is_one_sixth : area_between_curves = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_one_sixth_l1240_124004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_and_decreasing_l1240_124052

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := 2 * Real.sin x - Real.pi * Real.log x

-- State the theorem
theorem zero_point_and_decreasing :
  (∃ x₀ : ℝ, x₀ ∈ Set.Ioo 1 (Real.exp 1) ∧ f x₀ = 0) ∧
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Ioo 0 Real.pi → x₂ ∈ Set.Ioo 0 Real.pi → x₁ < x₂ → f x₁ > f x₂) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_and_decreasing_l1240_124052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_expression_l1240_124042

theorem simplify_sqrt_expression : Real.sqrt (28 - 10 * Real.sqrt 7) = 5 - Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_expression_l1240_124042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1240_124075

theorem hyperbola_eccentricity 
  (m n : ℝ) 
  (h1 : n > m) 
  (h2 : m > 0) 
  (h3 : ∀ x y : ℝ, m * x^2 + n * y^2 = 1 → 
       ∃ a b c : ℝ, a^2 = 1/m ∧ b^2 = 1/n ∧ c^2/a^2 = 1/2) :
  ∃ e : ℝ, (∀ x y : ℝ, m * x^2 - n * y^2 = 1 → 
    e = Real.sqrt ((1/m + 1/n)/(1/m))) ∧ e = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1240_124075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_bounds_l1240_124098

-- Rename the circle definition to avoid conflict with existing definitions
def circleC (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1

theorem circle_bounds (x y : ℝ) (h : circleC x y) :
  ((3 - Real.sqrt 3) / 4 ≤ (y - 2) / (x - 1) ∧ (y - 2) / (x - 1) ≤ (3 + Real.sqrt 3) / 4) ∧
  (-Real.sqrt 5 - 2 ≤ x - 2*y ∧ x - 2*y ≤ Real.sqrt 5 - 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_bounds_l1240_124098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1240_124020

-- Define the sets A and B
def A : Set ℝ := {x | (2 : ℝ)^x - 4 > 0}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

-- State the theorem
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1240_124020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_direction_crossing_time_l1240_124016

/-- Represents a train with its length and speed -/
structure Train where
  length : ℝ
  speed : ℝ

/-- Calculates the time taken for two trains to cross each other -/
noncomputable def timeToCross (t1 t2 : Train) (relativeSpeed : ℝ) : ℝ :=
  (t1.length + t2.length) / relativeSpeed

/-- Theorem stating the time taken for trains to cross in opposite directions -/
theorem opposite_direction_crossing_time
  (t1 t2 : Train)
  (h1 : t1.length = 120)
  (h2 : t2.length = 150)
  (h3 : t1.speed = 80 * 1000 / 3600)
  (h4 : t2.speed = 100 * 1000 / 3600)
  (h5 : timeToCross t1 t2 (t2.speed - t1.speed) = 60) :
  timeToCross t1 t2 (t1.speed + t2.speed) = 5.4 := by
  sorry

#check opposite_direction_crossing_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_direction_crossing_time_l1240_124016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_consecutive_positive_l1240_124099

def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n = a (n - 1) + a (n + 2)

theorem max_consecutive_positive (a : ℕ → ℝ) (h : sequence_property a) :
  (∃ k : ℕ, ∀ i : ℕ, i < 5 → a (k + i) > 0) ∧
  ¬(∃ k : ℕ, ∀ i : ℕ, i < 6 → a (k + i) > 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_consecutive_positive_l1240_124099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_m_n_equals_three_g_2x_correct_l1240_124084

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 1/2 then 1
  else if 1/2 ≤ x ∧ x < 1 then -1
  else 0

-- Define the function g
noncomputable def g (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 1 then 1
  else 0

-- State the theorem
theorem sum_m_n_equals_three (m n : ℤ) 
  (h : ∀ x : ℝ, m * g (n * x) - g x = f x) : 
  m + n = 3 := by
  sorry

-- Define g(2x)
noncomputable def g_2x (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 1/2 then 1
  else 0

-- State the theorem for g(2x)
theorem g_2x_correct (x : ℝ) : g (2 * x) = g_2x x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_m_n_equals_three_g_2x_correct_l1240_124084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_area_of_semicircles_l1240_124086

/-- The area of the figure remaining after cutting two smaller semicircles from a larger semicircle -/
theorem remaining_area_of_semicircles (AB AC CB : ℝ) (h_chord : ℝ) : 
  AB = AC + CB →  -- C is on the diameter AB
  h_chord = 8 →   -- The chord through C perpendicular to AB has length 8
  AC = CB →       -- C bisects AB (derived from the chord length)
  let r := AB / 2
  let area_large := π * r^2 / 2
  let area_small := π * (AC / 2)^2 / 2 + π * (CB / 2)^2 / 2
  area_large - area_small = 4 * π := by
  sorry

#check remaining_area_of_semicircles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_area_of_semicircles_l1240_124086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l1240_124087

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 - 1 / x

-- State the theorem
theorem tangent_slope_at_one :
  (deriv f) 1 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l1240_124087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prize_selection_count_l1240_124080

theorem prize_selection_count : (
  let total_essays : Nat := 20
  let first_prize : Nat := 1
  let second_prizes : Nat := 2
  let third_prizes : Nat := 4
  
  (Nat.choose total_essays first_prize) *
  (Nat.choose (total_essays - first_prize) second_prizes) *
  (Nat.choose (total_essays - first_prize - second_prizes) third_prizes)
) = 8145600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prize_selection_count_l1240_124080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_c_value_angle_C_value_l1240_124000

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ

-- Define the given conditions
noncomputable def givenTriangle : Triangle where
  a := 2 * Real.sqrt 3
  B := 45 * (Real.pi / 180)  -- Convert to radians
  S := 3 + Real.sqrt 3
  b := 0  -- Placeholder, will be calculated
  c := 0  -- To be proven
  A := 0  -- Placeholder, will be calculated
  C := 0  -- To be proven

-- Theorem for side c
theorem side_c_value (t : Triangle) (h1 : t = givenTriangle) :
  t.c = Real.sqrt 2 + Real.sqrt 6 := by sorry

-- Theorem for angle C
theorem angle_C_value (t : Triangle) (h1 : t = givenTriangle) :
  t.C = 75 * (Real.pi / 180) := by sorry  -- Convert to radians

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_c_value_angle_C_value_l1240_124000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_commutative_add_distrib_avg_avg_not_associative_avg_not_distrib_add_avg_no_identity_l1240_124057

/-- The average operation -/
noncomputable def avg (a b : ℝ) : ℝ := (a + b) / 2

/-- Commutativity of average operation -/
theorem avg_commutative (a b : ℝ) : avg a b = avg b a := by sorry

/-- Distributivity of addition over average -/
theorem add_distrib_avg (a b c : ℝ) : a + avg b c = avg (a + b) (a + c) := by sorry

/-- Non-associativity of average operation -/
theorem avg_not_associative : ¬ ∀ a b c : ℝ, avg a (avg b c) = avg (avg a b) c := by sorry

/-- Non-distributivity of average over addition -/
theorem avg_not_distrib_add : ¬ ∀ a b c : ℝ, avg a (b + c) = avg a b + avg a c := by sorry

/-- Non-existence of identity element for average operation -/
theorem avg_no_identity : ¬ ∃ e : ℝ, ∀ a : ℝ, avg a e = a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_commutative_add_distrib_avg_avg_not_associative_avg_not_distrib_add_avg_no_identity_l1240_124057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_earnings_proof_l1240_124033

/-- Mary's weekly earnings based on her work schedule and hourly rate -/
def mary_weekly_earnings
  (hours_long_day : ℕ)
  (num_long_days : ℕ)
  (hours_short_day : ℕ)
  (num_short_days : ℕ)
  (hourly_rate : ℕ) : ℕ :=
  (hours_long_day * num_long_days + hours_short_day * num_short_days) * hourly_rate

theorem mary_earnings_proof :
  mary_weekly_earnings 9 3 5 2 11 = 407 := by
  -- Unfold the definition of mary_weekly_earnings
  unfold mary_weekly_earnings
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_earnings_proof_l1240_124033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_bead_necklace_colorings_l1240_124018

/-- The number of ways to color a necklace of n beads with 4 colors,
    where no two adjacent beads have the same color. -/
def necklace_colorings : ℕ → ℕ
| 0 => 4  -- Base case for n = 0 (technically not a necklace, but needed for recursion)
| 1 => 4  -- Base case for n = 1
| n + 2 => 4 * 3^(n+1) - necklace_colorings (n+1)

theorem seven_bead_necklace_colorings :
  necklace_colorings 7 = 2188 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_bead_necklace_colorings_l1240_124018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_divisibility_satisfying_polynomials_l1240_124029

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- The property that f satisfies the divisibility condition -/
def SatisfiesDivisibilityCondition (f : IntPolynomial) : Prop :=
  ∀ (p : ℕ) (u v : ℕ), Nat.Prime p → (p ∣ u * v - 1) → (p ∣ Int.toNat (f u * f v - 1))

/-- The characterization of polynomials satisfying the divisibility condition -/
theorem characterization_of_divisibility_satisfying_polynomials
  (f : IntPolynomial) (h : SatisfiesDivisibilityCondition f) :
  ∃ (n : ℕ) (a : ℤ), (a = 1 ∨ a = -1) ∧ (∀ x, f x = a * (x : ℤ)^n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_divisibility_satisfying_polynomials_l1240_124029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_squares_triangle_area_l1240_124043

-- Define a right triangle with legs a and b
structure RightTriangle where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b

-- Define the area of the triangle formed by the centers of squares
noncomputable def centerSquaresTriangleArea (t : RightTriangle) : ℝ := (t.a + t.b)^2 / 4

-- Theorem statement
theorem center_squares_triangle_area (t : RightTriangle) :
  centerSquaresTriangleArea t = (t.a + t.b)^2 / 4 := by
  -- Unfold the definition of centerSquaresTriangleArea
  unfold centerSquaresTriangleArea
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_squares_triangle_area_l1240_124043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_women_in_business_class_l1240_124094

theorem women_in_business_class 
  (total_passengers : ℕ) 
  (women_percentage : ℚ) 
  (business_class_percentage : ℚ) 
  (h1 : total_passengers = 300) 
  (h2 : women_percentage = 70 / 100) 
  (h3 : business_class_percentage = 8 / 100) : 
  Int.floor ((total_passengers : ℚ) * women_percentage * business_class_percentage) = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_women_in_business_class_l1240_124094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_quadrilateral_theorem_l1240_124055

-- Define the points
variable (A B C D E F G H I P Q R S : EuclideanSpace ℝ (Fin 2))

-- Define the convex quadrilateral ABCD
def is_convex_quadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define E as an interior point of ABCD
def is_interior_point (E A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the similarity of triangles
def similar_triangles (A B C D E F : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the projection of a point onto a line
def projection (E P A B : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define a cyclic quadrilateral
def is_cyclic_quadrilateral (P Q R S : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

theorem convex_quadrilateral_theorem 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_interior_point E A B C D)
  (h3 : similar_triangles A B F D C E)
  (h4 : similar_triangles B C G A D E)
  (h5 : similar_triangles C D H B A E)
  (h6 : similar_triangles D A I C B E)
  (h7 : projection E P A B)
  (h8 : projection E Q B C)
  (h9 : projection E R C D)
  (h10 : projection E S D A)
  (h11 : is_cyclic_quadrilateral P Q R S) :
  ∃ (k : ℝ), k > 0 ∧ 
    ‖E - F‖ * ‖C - D‖ = k ∧
    ‖E - G‖ * ‖D - A‖ = k ∧
    ‖E - H‖ * ‖A - B‖ = k ∧
    ‖E - I‖ * ‖B - C‖ = k :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_quadrilateral_theorem_l1240_124055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_implies_c_value_l1240_124069

noncomputable def vector_projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_squared := v.1^2 + v.2^2
  (dot_product / norm_squared) • v

theorem projection_implies_c_value (c : ℝ) :
  let u : ℝ × ℝ := (-3, c)
  let v : ℝ × ℝ := (1, 2)
  vector_projection u v = (-7/5) • v →
  c = -2 :=
by
  intros
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_implies_c_value_l1240_124069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_two_roots_condition_l1240_124065

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x - Real.pi / 4)

-- State the theorem
theorem exact_two_roots_condition (ω : ℝ) :
  (ω > 0) →
  (∃! (r₁ r₂ : ℝ), r₁ ∈ Set.Icc 0 (2 * Real.pi) ∧
                    r₂ ∈ Set.Icc 0 (2 * Real.pi) ∧
                    r₁ ≠ r₂ ∧
                    |f ω r₁| = 1 ∧
                    |f ω r₂| = 1 ∧
                    ∀ x ∈ Set.Icc 0 (2 * Real.pi), |f ω x| = 1 → (x = r₁ ∨ x = r₂)) ↔
  (7 / 8 ≤ ω ∧ ω < 11 / 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_two_roots_condition_l1240_124065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_finish_third_l1240_124032

/-- Represents the runners in the race -/
inductive Runner : Type
  | P | Q | R | S | T | U
deriving BEq, Repr

/-- Represents the finishing order of the race -/
def FinishingOrder := List Runner

/-- Checks if runner1 beats runner2 in the given finishing order -/
def beats (order : FinishingOrder) (runner1 runner2 : Runner) : Prop :=
  order.indexOf runner1 < order.indexOf runner2

/-- Checks if the given finishing order satisfies all race conditions -/
def validOrder (order : FinishingOrder) : Prop :=
  beats order Runner.P Runner.Q ∧
  beats order Runner.P Runner.R ∧
  beats order Runner.Q Runner.S ∧
  beats order Runner.P Runner.T ∧
  beats order Runner.T Runner.Q ∧
  beats order Runner.P Runner.U ∧
  beats order Runner.U Runner.T

/-- Checks if a runner finished third in the given order -/
def finishedThird (order : FinishingOrder) (runner : Runner) : Prop :=
  order.indexOf runner = 2

/-- Theorem: P, S, and T cannot finish third in any valid race order -/
theorem cannot_finish_third : 
  ∀ (order : FinishingOrder), validOrder order → 
    ¬(finishedThird order Runner.P ∨ 
      finishedThird order Runner.S ∨ 
      finishedThird order Runner.T) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_finish_third_l1240_124032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_value_in_third_quadrant_l1240_124060

theorem sine_value_in_third_quadrant (α : ℝ) 
  (h1 : Real.cos α = -3/5) 
  (h2 : π < α ∧ α < 3*π/2) : 
  Real.sin α = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_value_in_third_quadrant_l1240_124060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l1240_124011

-- Define the function f
noncomputable def f (A : ℝ) (φ : ℝ) (x : ℝ) : ℝ := 
  A * (Real.sin (x/2) * Real.cos φ + Real.cos (x/2) * Real.sin φ)

-- State the theorem
theorem triangle_area_proof 
  (A : ℝ) (φ : ℝ) 
  (h1 : A > 0) 
  (h2 : 0 < φ) (h3 : φ < Real.pi/2)
  (h4 : ∀ x, f A φ x ≤ 2)
  (h5 : f A φ 0 = 1)
  (a b c : ℝ) (A' B C : ℝ)
  (h6 : a = 2)
  (h7 : f A φ (2*A') = 2)
  (h8 : 2*b*Real.sin C = Real.sqrt 2 * c)
  (h9 : 0 < A') (h10 : A' < Real.pi/2)
  (h11 : 0 < B) (h12 : B < Real.pi/2)
  (h13 : 0 < C) (h14 : C < Real.pi/2)
  (h15 : A' + B + C = Real.pi)
  : φ = Real.pi/6 ∧ 
    (1/2) * a * b * Real.sin C = 1 + Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l1240_124011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_and_point_problem_l1240_124028

/-- Given that the terminal side of angle α passes through point P(m, 2√2),
    sin(α) = 2√2/3, and α is in the second quadrant, prove that m = -1 and
    the given expression evaluates to -5/3. -/
theorem angle_and_point_problem (α : ℝ) (m : ℝ) :
  (∃ (P : ℝ × ℝ), P = (m, 2 * Real.sqrt 2)) →
  Real.sin α = 2 * Real.sqrt 2 / 3 →
  π / 2 < α ∧ α < π →
  m = -1 ∧
  (2 * Real.sqrt 2 * Real.sin α + 3 * Real.sin (π / 2 + α)) /
  (Real.cos (π + α) + Real.sqrt 2 * Real.cos (5 * π / 2 + α)) = -5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_and_point_problem_l1240_124028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_cyclic_perpendicular_chords_perpendicular_chords_cyclic_tangents_l1240_124085

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with a center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a quadrilateral with four vertices -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a quadrilateral is circumscribed around a circle -/
def is_circumscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

/-- Checks if a quadrilateral is cyclic (can be inscribed in a circle) -/
def is_cyclic (q : Quadrilateral) (c : Circle) : Prop := sorry

/-- Creates a chord (line segment) between two points -/
def chord (p1 p2 : Point) : Line := sorry

/-- Finds the point of tangency on a circle for a given external point -/
def tangent_point (c : Circle) (p : Point) : Point := sorry

/-- Checks if two lines are perpendicular -/
def is_perpendicular (l1 l2 : Line) : Prop := sorry

/-- Theorem: If a quadrilateral is both circumscribed and cyclic, 
    then the chords connecting opposite tangent points are perpendicular -/
theorem circumscribed_cyclic_perpendicular_chords 
  (c : Circle) (q : Quadrilateral) : 
  is_circumscribed q c → is_cyclic q c → 
  is_perpendicular 
    (chord (tangent_point c q.A) (tangent_point c q.C)) 
    (chord (tangent_point c q.B) (tangent_point c q.D)) :=
by sorry

/-- Theorem: If two chords in a circle are perpendicular, 
    then the tangents at their endpoints form a cyclic quadrilateral -/
theorem perpendicular_chords_cyclic_tangents 
  (c : Circle) (p1 p2 p3 p4 : Point) :
  is_perpendicular (chord p1 p3) (chord p2 p4) → 
  is_cyclic (Quadrilateral.mk 
    (tangent_point c p1) 
    (tangent_point c p2) 
    (tangent_point c p3) 
    (tangent_point c p4)) c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_cyclic_perpendicular_chords_perpendicular_chords_cyclic_tangents_l1240_124085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diet_cola_price_proof_l1240_124015

/-- The cost of a diet cola at Taco Palace --/
def diet_cola_cost : ℝ := 2

/-- The cost of a Taco Grande Plate --/
def T : ℝ := Real.mk 0  -- We define T as a real number, initialized to 0

/-- Theorem: Given the conditions of Mike and John's lunch at Taco Palace, 
    the cost of the diet cola is $2 --/
theorem diet_cola_price_proof (h1 : T + 2 + 4 + diet_cola_cost = 2 * T) 
                               (h2 : (T + 2 + 4 + diet_cola_cost) + T = 24) : 
  diet_cola_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diet_cola_price_proof_l1240_124015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1240_124045

noncomputable def f (x : ℝ) : ℝ := 1 / Real.exp (abs x) - x^2

theorem range_of_a (a : ℝ) : 
  (f (3^(a-1)) > f (-1/9)) ↔ (a ∈ Set.Ioi (-1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1240_124045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_relationship_l1240_124091

theorem condition_relationship : 
  ¬(∀ a b : ℝ, (1/2:ℝ)^a < (1/2:ℝ)^b → a^2 > b^2) ∧ 
  ¬(∀ a b : ℝ, a^2 > b^2 → (1/2:ℝ)^a < (1/2:ℝ)^b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_relationship_l1240_124091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_pairs_l1240_124031

theorem count_integer_pairs : ∃! n : ℕ, 
  n = (Finset.filter (fun p : ℕ × ℕ => 
    let a := p.1
    let b := p.2
    a > 0 ∧ b > 0 ∧ a + b ≤ 100 ∧ 
    (a : ℚ) + (1 : ℚ) / b = 7 * ((1 : ℚ) / a + (b : ℚ))
  ) (Finset.product (Finset.range 101) (Finset.range 101))).card ∧
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_pairs_l1240_124031
