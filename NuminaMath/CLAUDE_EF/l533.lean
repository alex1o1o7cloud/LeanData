import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_2_range_of_a_given_conditions_l533_53370

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 2 * x

-- Part 1
theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x ≥ 2 * x + 1} = Set.Iic 1 ∪ Set.Ici 3 :=
sorry

-- Part 2
theorem range_of_a_given_conditions :
  ∀ a : ℝ, a > 0 → (∀ x : ℝ, x > -2 → f a x > 0) → a ≥ 2 :=
sorry

#check solution_set_when_a_is_2
#check range_of_a_given_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_2_range_of_a_given_conditions_l533_53370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l533_53354

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 - 4*x + 3

/-- Point A -/
def A : ℝ × ℝ := (0, 3)

/-- Point B -/
def B : ℝ × ℝ := (3, 0)

/-- Point C -/
def C (p q : ℝ) : ℝ × ℝ := (p, q)

/-- The area of triangle ABC given p and q -/
noncomputable def triangle_area (p q : ℝ) : ℝ :=
  (1/2) * abs (A.1 * B.2 + B.1 * q + p * A.2 - 
               B.1 * A.2 - p * B.2 - A.1 * q)

/-- The main theorem -/
theorem max_triangle_area :
  ∃ (p q : ℝ), 
    0 ≤ p ∧ p ≤ 3 ∧
    parabola p q ∧
    (∀ (p' q' : ℝ), 0 ≤ p' ∧ p' ≤ 3 ∧ parabola p' q' → 
      triangle_area p q ≥ triangle_area p' q') ∧
    triangle_area p q = 22.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l533_53354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_between_50_and_500_l533_53337

theorem perfect_squares_between_50_and_500 : 
  let n : ℕ := 8  -- smallest integer such that n^2 ≥ 50
  let m : ℕ := 22 -- largest integer such that m^2 ≤ 500
  ∀ k ∈ Finset.range (m - n + 1), (k + n)^2 ∈ Set.Ioo 50 500 →
  (Finset.range (m - n + 1)).card = 15 :=
by
  intro n m k hk hsquare
  simp only [Finset.card_range]
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_between_50_and_500_l533_53337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheapest_trip_cost_l533_53342

-- Define the cities and distances
def distance_DF : ℚ := 4000
def distance_DE : ℚ := 4500

-- Define travel costs
def bus_cost_per_km : ℚ := 1/5
def plane_cost_per_km : ℚ := 3/20
def plane_booking_fee : ℚ := 150

-- Function to calculate bus cost
def bus_cost (distance : ℚ) : ℚ :=
  distance * bus_cost_per_km

-- Function to calculate plane cost
def plane_cost (distance : ℚ) : ℚ :=
  distance * plane_cost_per_km + plane_booking_fee

-- Function to get the cheaper option between bus and plane
noncomputable def cheaper_option (distance : ℚ) : ℚ :=
  min (bus_cost distance) (plane_cost distance)

-- Theorem statement
theorem cheapest_trip_cost :
  ∃ (distance_EF : ℚ),
    distance_EF^2 = distance_DE^2 - distance_DF^2 ∧
    cheaper_option distance_DE +
    bus_cost distance_EF +
    cheaper_option distance_DF = 1987 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheapest_trip_cost_l533_53342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carls_marbles_l533_53300

/-- Represents the problem of calculating additional marbles Carl took out --/
def additional_marbles_problem (initial_marbles : ℕ) (lost_fraction : ℚ) 
  (new_marbles : ℕ) (final_marbles : ℕ) : Prop :=
  let lost_marbles : ℕ := (initial_marbles * lost_fraction.num / lost_fraction.den).toNat
  let remaining_initial : ℕ := initial_marbles - lost_marbles
  let original_to_return : ℕ := final_marbles - new_marbles
  let total_needed : ℕ := original_to_return + lost_marbles
  total_needed - initial_marbles = 10

/-- The theorem stating the solution to Carl's marble problem --/
theorem carls_marbles : 
  additional_marbles_problem 12 (1/2) 25 41 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carls_marbles_l533_53300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_remaining_area_right_triangle_l533_53360

-- Define necessary structures and functions
structure Point where
  x : ℝ
  y : ℝ

def RightTriangle (A B C : Point) : Prop := sorry

def SegmentLength (P Q : Point) : ℝ := sorry

def CircleTangent (A : Point) (r : ℝ) (B C : Point) : Prop := sorry

def MaxRemainingArea (A B C : Point) (r : ℝ) : ℝ := sorry

-- Main theorem
theorem max_remaining_area_right_triangle (A B C : Point) (r : ℝ) :
  RightTriangle A B C →
  SegmentLength B C = 2 * Real.pi →
  CircleTangent A r B C →
  (∃ (S : ℝ), S = MaxRemainingArea A B C r ∧ S = Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_remaining_area_right_triangle_l533_53360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_circle_radius_l533_53357

-- Define the angle
def angle : ℝ := 60

-- Define the radius of the smaller circle
variable (r : ℝ)

-- Define the radius of the larger circle as a function of r
def R (r : ℝ) : ℝ := 3 * r

-- Theorem statement
theorem larger_circle_radius (r : ℝ) (h1 : angle = 60) (h2 : r > 0) :
  R r = 3 * r :=
by
  -- Unfold the definition of R
  unfold R
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_circle_radius_l533_53357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_given_tan_cot_sum_l533_53365

theorem tan_sum_given_tan_cot_sum (x y : ℝ) 
  (h1 : Real.tan x + Real.tan y = 10)
  (h2 : (Real.tan x)⁻¹ + (Real.tan y)⁻¹ = 15) : 
  Real.tan (x + y) = 30 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_given_tan_cot_sum_l533_53365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_negative_11pi_12_g_max_min_in_interval_l533_53332

noncomputable def f (x : Real) : Real :=
  (4 * (Real.cos x)^4 - 2 * Real.cos (2 * x) - 1) / (Real.sin (Real.pi / 4 + x) * Real.sin (Real.pi / 4 - x))

noncomputable def g (x : Real) : Real :=
  (1 / 2) * f x + Real.sin (2 * x)

theorem f_value_at_negative_11pi_12 :
  f (-11 * Real.pi / 12) = Real.sqrt 3 := by sorry

theorem g_max_min_in_interval :
  ∀ x ∈ Set.Icc 0 (Real.pi / 4),
    g x ≤ Real.sqrt 2 ∧
    1 ≤ g x ∧
    (∃ x₁ ∈ Set.Icc 0 (Real.pi / 4), g x₁ = Real.sqrt 2) ∧
    (∃ x₂ ∈ Set.Icc 0 (Real.pi / 4), g x₂ = 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_negative_11pi_12_g_max_min_in_interval_l533_53332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_conditions_l533_53327

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define properties of a triangle
def Triangle.isEquilateral (t : Triangle) : Prop := sorry
def Triangle.allSidesEqual (t : Triangle) : Prop := sorry
def Triangle.allAnglesEqual (t : Triangle) : Prop := sorry
def Triangle.twoAngles60Deg (t : Triangle) : Prop := sorry
def Triangle.isoscelesOneAngle60Deg (t : Triangle) : Prop := sorry

-- State the theorem
theorem equilateral_triangle_conditions (t : Triangle) :
  (Triangle.allSidesEqual t → Triangle.isEquilateral t) ∧
  (Triangle.allAnglesEqual t → Triangle.isEquilateral t) ∧
  (Triangle.twoAngles60Deg t → Triangle.isEquilateral t) ∧
  (Triangle.isoscelesOneAngle60Deg t → Triangle.isEquilateral t) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_conditions_l533_53327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_l533_53348

/-- The angle between the asymptotes of a hyperbola -/
noncomputable def asymptote_angle (a b : ℝ) : ℝ := 
  2 * Real.arctan (b / a)

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (a b : ℝ) : ℝ := 
  Real.sqrt (1 + b^2 / a^2)

theorem hyperbola_asymptote_angle 
  (a b : ℝ) 
  (h1 : b > 0) 
  (h2 : eccentricity a b = Real.sqrt 2) : 
  asymptote_angle a b = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_l533_53348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_first_four_terms_sequence_general_term_l533_53310

/-- The sequence a_n defined by the given first four terms and the general formula. -/
noncomputable def sequence_a (n : ℕ+) : ℝ :=
  Real.sqrt (3 * n.val - 1)

/-- The first four terms of the sequence match the given values. -/
theorem sequence_first_four_terms :
  sequence_a 1 = Real.sqrt 2 ∧
  sequence_a 2 = Real.sqrt 5 ∧
  sequence_a 3 = 2 * Real.sqrt 2 ∧
  sequence_a 4 = Real.sqrt 11 := by
  sorry

/-- The general term formula for the sequence is correct for all positive integers. -/
theorem sequence_general_term (n : ℕ+) :
  sequence_a n = Real.sqrt (3 * n.val - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_first_four_terms_sequence_general_term_l533_53310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l533_53315

-- Define the complex number z
def z (b : ℝ) : ℂ := 3 + Complex.I * b

-- Define the condition that (1+3i) · z is purely imaginary
def isPurelyImaginary (z : ℂ) : Prop := (Complex.re ((1 + 3*Complex.I) * z) = 0) ∧ (Complex.im ((1 + 3*Complex.I) * z) ≠ 0)

-- Define omega
noncomputable def ω (b : ℝ) : ℂ := z b / (2 + Complex.I)

-- The theorem to prove
theorem complex_number_problem (b : ℝ) : 
  isPurelyImaginary (z b) → z b = 3 + Complex.I ∧ Complex.abs (ω b) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l533_53315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_true_propositions_l533_53341

-- Define the propositions as axioms
axiom prop1 : Prop
axiom prop2 : Prop
axiom prop3 : Prop
axiom prop4 : Prop
axiom prop5 : Prop

-- Define a function to count true propositions
def countTrueProps (p1 p2 p3 p4 p5 : Bool) : Nat :=
  (if p1 then 1 else 0) + (if p2 then 1 else 0) + (if p3 then 1 else 0) + (if p4 then 1 else 0) + (if p5 then 1 else 0)

-- Theorem statement
theorem four_true_propositions :
  ∃ (b1 b2 b3 b4 b5 : Bool),
    (b1 ↔ prop1) ∧
    (b2 ↔ prop2) ∧
    (b3 ↔ prop3) ∧
    (b4 ↔ prop4) ∧
    (b5 ↔ prop5) ∧
    countTrueProps b1 b2 b3 b4 b5 = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_true_propositions_l533_53341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_to_widescreen_area_ratio_l533_53335

/-- Represents a TV model with given aspect ratio and diagonal length -/
structure TVModel where
  aspectRatioWidth : ℕ
  aspectRatioHeight : ℕ
  diagonalLength : ℝ

/-- Calculates the area of a TV model -/
noncomputable def calculateArea (tv : TVModel) : ℝ := 
  let totalRatio := (tv.aspectRatioWidth^2 + tv.aspectRatioHeight^2 : ℝ)
  let scaleFactor := tv.diagonalLength^2 / totalRatio
  (tv.aspectRatioWidth * tv.aspectRatioHeight : ℝ) * scaleFactor

/-- The main theorem stating the area ratio of standard to widescreen TV -/
theorem standard_to_widescreen_area_ratio : 
  let standardTV := TVModel.mk 4 3 20
  let widescreenTV := TVModel.mk 16 9 20
  (calculateArea standardTV) / (calculateArea widescreenTV) = 337 / 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_to_widescreen_area_ratio_l533_53335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_of_tan_fourth_quadrant_l533_53338

theorem sin_of_tan_fourth_quadrant (α : ℝ) :
  (α > -π / 2 ∧ α < 0) →  -- α is in the fourth quadrant
  Real.tan α = -5 / 12 →
  Real.sin α = -5 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_of_tan_fourth_quadrant_l533_53338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendiculars_concurrent_iff_equation_holds_l533_53369

/-- A triangle with points on its sides and perpendiculars drawn from these points -/
structure TriangleWithPerps where
  A : ℝ × ℝ  -- Vertex A of the triangle
  B : ℝ × ℝ  -- Vertex B of the triangle
  C : ℝ × ℝ  -- Vertex C of the triangle
  U : ℝ × ℝ  -- Point on BC
  V : ℝ × ℝ  -- Point on CA
  W : ℝ × ℝ  -- Point on AB
  on_BC : (U.1 - B.1) * (C.2 - B.2) = (U.2 - B.2) * (C.1 - B.1)  -- U is on BC
  on_CA : (V.1 - C.1) * (A.2 - C.2) = (V.2 - C.2) * (A.1 - C.1)  -- V is on CA
  on_AB : (W.1 - A.1) * (B.2 - A.2) = (W.2 - A.2) * (B.1 - A.1)  -- W is on AB

/-- The perpendiculars are concurrent -/
def are_concurrent (t : TriangleWithPerps) : Prop :=
  ∃ M : ℝ × ℝ,
    (M.1 - t.U.1) * (t.C.1 - t.B.1) + (M.2 - t.U.2) * (t.C.2 - t.B.2) = 0 ∧
    (M.1 - t.V.1) * (t.A.1 - t.C.1) + (M.2 - t.V.2) * (t.A.2 - t.C.2) = 0 ∧
    (M.1 - t.W.1) * (t.B.1 - t.A.1) + (M.2 - t.W.2) * (t.B.2 - t.A.2) = 0

/-- The equation of squared distances -/
def equation_holds (t : TriangleWithPerps) : Prop :=
  let dist_sq (p q : ℝ × ℝ) := (p.1 - q.1)^2 + (p.2 - q.2)^2
  dist_sq t.A t.W + dist_sq t.B t.U + dist_sq t.C t.V =
  dist_sq t.A t.V + dist_sq t.C t.U + dist_sq t.B t.W

/-- The main theorem -/
theorem perpendiculars_concurrent_iff_equation_holds (t : TriangleWithPerps) :
  are_concurrent t ↔ equation_holds t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendiculars_concurrent_iff_equation_holds_l533_53369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_rep_845th_digit_l533_53356

/-- The decimal representation of 7/29 -/
def decimal_rep : ℚ := 7/29

/-- The length of the repeating sequence in the decimal representation of 7/29 -/
def cycle_length : ℕ := 28

/-- The repeating sequence in the decimal representation of 7/29 -/
def repeating_sequence : List ℕ := [2, 4, 1, 3, 7, 9, 3, 1, 0, 3, 4, 4, 8, 2, 7, 5, 8, 6, 2, 0, 6, 8, 9, 6, 5, 5, 1, 7]

/-- The position we're interested in -/
def target_position : ℕ := 845

theorem decimal_rep_845th_digit :
  (repeating_sequence.get! ((target_position - 1) % cycle_length)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_rep_845th_digit_l533_53356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_of_sequence_l533_53392

def mySequence (n : ℕ) : ℚ :=
  (-1)^(n+1 : ℤ) * (((2^n : ℕ) + 1 : ℚ) / ((2*n - 1 : ℕ) : ℚ))

theorem tenth_term_of_sequence :
  mySequence 10 = -1025 / 19 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_of_sequence_l533_53392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_l533_53328

/-- Cube with vertices at (±a, ±a, ±a) -/
def Cube (a : ℝ) : Set (Fin 3 → ℝ) :=
  {v | ∀ i, v i = a ∨ v i = -a}

/-- Tetrahedron with vertices at (2a,2a,2a), (2a,-2a,-2a), (-2a,2a,-2a), (-2a,-2a,2a) -/
def Tetrahedron (a : ℝ) : Set (Fin 3 → ℝ) :=
  {v | v ∈ ({(λ _ ↦ 2*a), (λ i ↦ if i = 0 then 2*a else -2*a),
            (λ i ↦ if i = 1 then 2*a else -2*a), (λ i ↦ if i = 2 then 2*a else -2*a)} : Set (Fin 3 → ℝ))}

/-- Volume of a set in ℝ³ -/
noncomputable def volume (S : Set (Fin 3 → ℝ)) : ℝ := sorry

theorem intersection_volume (a : ℝ) (h : a > 0) :
  volume (Cube a ∩ Tetrahedron a) = (4/3) * a^3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_l533_53328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x_axis_intersection_l533_53377

noncomputable def f (x : ℝ) : ℝ := (x^5 - 1) / 5

theorem tangent_line_at_x_axis_intersection :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := deriv f x₀
  (∀ x, f x = 0 → x = x₀) →
  y₀ = 0 →
  m = 1 →
  ∀ x y, y = m * (x - x₀) + y₀ ↔ y = x - 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x_axis_intersection_l533_53377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_AE_length_l533_53363

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The theorem to be proved -/
theorem segment_AE_length 
  (A B C D E : Point)
  (h_A : A = ⟨0, 4⟩)
  (h_B : B = ⟨6, 0⟩)
  (h_C : C = ⟨5, 3⟩)
  (h_D : D = ⟨3, 0⟩)
  (h_E : E.x ∈ Set.Icc 0 6 ∧ E.y ∈ Set.Icc 0 4) -- E is within the grid
  (h_AB_CD : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
    E = ⟨A.x + t * (B.x - A.x), A.y + t * (B.y - A.y)⟩ ∧
    ∃ s : ℝ, 0 < s ∧ s < 1 ∧ 
    E = ⟨C.x + s * (D.x - C.x), C.y + s * (D.y - C.y)⟩) 
  : distance A E = (5 * Real.sqrt 13) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_AE_length_l533_53363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_one_equals_11_25_l533_53389

-- Define the functions t and s
def t (x : ℝ) : ℝ := 4 * x - 9

noncomputable def s (y : ℝ) : ℝ := 
  let x := (y + 9) / 4  -- Inverse of t
  x^2 + 4 * x - 5

-- State the theorem
theorem s_of_one_equals_11_25 : s 1 = 11.25 := by
  -- Expand the definition of s
  unfold s
  -- Simplify the expression
  simp [add_div, mul_div, pow_two]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_one_equals_11_25_l533_53389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_111010110101_div_4_l533_53303

def binary_to_decimal (b : List Bool) : Nat :=
  b.foldr (λ bit acc => 2 * acc + if bit then 1 else 0) 0

def last_two_digits (b : List Bool) : List Bool :=
  b.reverse.take 2

theorem remainder_of_111010110101_div_4 :
  binary_to_decimal [true, true, true, false, true, false, true, true, false, true, false, true] % 4 =
  binary_to_decimal (last_two_digits [true, true, true, false, true, false, true, true, false, true, false, true]) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_111010110101_div_4_l533_53303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l533_53387

theorem sin_plus_cos_value (α : Real) 
  (h1 : Real.sin (2 * α) = 2/3) 
  (h2 : 0 < α ∧ α < Real.pi) : 
  Real.sin α + Real.cos α = Real.sqrt 15 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l533_53387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l533_53326

theorem sin_minus_cos_value (θ : Real) (h1 : Real.sin θ + Real.cos θ = 3/4) 
  (h2 : 0 < θ ∧ θ < Real.pi) : Real.sin θ - Real.cos θ = Real.sqrt 23 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l533_53326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_paths_independent_of_n_l533_53333

/-- A board with 1000 columns and n rows -/
structure Board (n : ℕ) where
  rows : ℕ
  cols : ℕ := 1000
  n_odd : Odd n
  n_large : n > 2020

/-- A path on the board -/
def BoardPath (n : ℕ) := List (ℕ × ℕ)

/-- The number of paths from (1,1) to (1000,n) -/
def num_paths (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of paths is independent of n -/
theorem num_paths_independent_of_n (n m : ℕ) 
  (hn : Board n) (hm : Board m) : 
  num_paths n = num_paths m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_paths_independent_of_n_l533_53333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_distance_to_focus_l533_53367

/-- Parabola with equation y² = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Focus of the parabola y² = 4x -/
def Focus : ℝ × ℝ := (1, 0)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_min_distance_to_focus :
  ∀ P ∈ Parabola, 1 ≤ distance P Focus ∧
  ∃ Q ∈ Parabola, distance Q Focus = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_distance_to_focus_l533_53367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_S_value_l533_53352

/-- The trajectory of the center of circle M -/
def trajectory (x y : ℝ) : Prop := y^2 = 4*x

/-- Point E -/
def E : ℝ × ℝ := (2, 0)

/-- Point F -/
def F : ℝ × ℝ := (1, 0)

/-- Length of chord PQ -/
def chord_length : ℝ := 4

/-- Circle M passes through point E and cuts chord PQ on y-axis -/
axiom circle_property (x y : ℝ) : 
  trajectory x y → ((x - E.1)^2 + y^2 = x^2 + (chord_length/2)^2)

/-- Dot product condition for points A and B -/
def dot_product_condition (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = -4

/-- Area of triangle OFA -/
noncomputable def area_OFA (A : ℝ × ℝ) : ℝ := (1/2) * A.2

/-- Area of triangle OAB -/
noncomputable def area_OAB (A B : ℝ × ℝ) : ℝ := (1/2) * |A.1 * B.2 - A.2 * B.1|

/-- Total area S -/
noncomputable def S (A B : ℝ × ℝ) : ℝ := area_OFA A + area_OAB A B

/-- Theorem: Minimum value of S is 4√3 -/
theorem min_S_value : 
  ∃ (A B : ℝ × ℝ), 
    trajectory A.1 A.2 ∧ 
    trajectory B.1 B.2 ∧ 
    dot_product_condition A B ∧ 
    (∀ (X Y : ℝ × ℝ), 
      trajectory X.1 X.2 → 
      trajectory Y.1 Y.2 → 
      dot_product_condition X Y → 
      S A B ≤ S X Y) ∧
    S A B = 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_S_value_l533_53352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_thirds_pi_rad_to_deg_l533_53381

/-- Conversion factor from radians to degrees -/
noncomputable def rad_to_deg : ℝ := 180 / Real.pi

/-- Converts radians to degrees -/
noncomputable def radians_to_degrees (x : ℝ) : ℝ := x * rad_to_deg

theorem two_thirds_pi_rad_to_deg :
  radians_to_degrees ((2 / 3) * Real.pi) = -120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_thirds_pi_rad_to_deg_l533_53381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l533_53301

-- Define the function f(x) = 2cos(x)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x

-- State the theorem
theorem f_satisfies_conditions :
  (∀ x, f (-x) = f x) ∧  -- f is an even function
  (∀ x, f x ≤ 2) ∧  -- maximum value of f is 2
  (∃ x, f x = 2) ∧  -- f attains the maximum value 2
  (¬ ∃ a b c, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c)  -- f is not a quadratic function
  := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l533_53301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_comparison_l533_53305

theorem sin_comparison :
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ π / 2 → Real.sin x < Real.sin y) →
  (0 < π - 3 ∧ π - 3 < 1 ∧ 1 < π - 2 ∧ π - 2 < π / 2) →
  Real.sin 3 < Real.sin 1 ∧ Real.sin 1 < Real.sin 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_comparison_l533_53305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_equilateral_triangle_in_rectangle_l533_53368

/-- The maximum area of an equilateral triangle inscribed in a 12x13 rectangle -/
theorem max_area_equilateral_triangle_in_rectangle : 
  ∃ (A : ℝ), 
    (∀ (a : ℝ), 
      (∃ (x y : ℝ), 
        0 ≤ x ∧ x ≤ 12 ∧ 
        0 ≤ y ∧ y ≤ 13 ∧ 
        a = (Real.sqrt 3 / 4) * (min x y)^2) → 
      a ≤ A) ∧ 
    A = 48 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_equilateral_triangle_in_rectangle_l533_53368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_neg_pi_third_l533_53353

theorem cos_neg_pi_third : Real.cos (-π/3) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_neg_pi_third_l533_53353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_jumps_l533_53306

noncomputable def jump1 (x : ℝ) : ℝ := x / Real.sqrt 3

noncomputable def jump2 (x : ℝ) : ℝ := x / Real.sqrt 3 + (1 - 1 / Real.sqrt 3)

def is_valid_jump (x y : ℝ) : Prop :=
  y = jump1 x ∨ y = jump2 x

def is_reachable (start target : ℝ) (ε : ℝ) : Prop :=
  ∃ (n : ℕ) (jumps : Fin (n + 1) → ℝ),
    jumps 0 = start ∧
    (∀ i : Fin n, is_valid_jump (jumps i) (jumps i.succ)) ∧
    |jumps n - target| < ε

theorem grasshopper_jumps (a : ℝ) (h₁ : 0 ≤ a) (h₂ : a ≤ 1) :
  ∀ start : ℝ, 0 ≤ start → start ≤ 1 →
    is_reachable start a (1/100) := by
  sorry

#check grasshopper_jumps

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_jumps_l533_53306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_iff_a_eq_one_g_two_zeros_iff_a_in_range_l533_53349

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (4^x + a) / 2^x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - (a + 1)

-- Theorem for part 1
theorem f_even_iff_a_eq_one (a : ℝ) :
  (∀ x, f a x = f a (-x)) ↔ a = 1 := by sorry

-- Theorem for part 2
theorem g_two_zeros_iff_a_in_range (a : ℝ) :
  (∃ x y, x ≠ y ∧ x ∈ Set.Icc (-1) 1 ∧ y ∈ Set.Icc (-1) 1 ∧ g a x = 0 ∧ g a y = 0) ↔
  (a ∈ Set.Icc (1/2) 1 ∪ Set.Ioc 1 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_iff_a_eq_one_g_two_zeros_iff_a_in_range_l533_53349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_range_l533_53302

/-- Given functions f and g, prove that if there exists a point where the tangent line of f
    is also tangent to g, then the parameter a of g is in the range (0, √(2e)]. -/
theorem tangent_line_range (a : ℝ) (h_a : a ≠ 0) :
  ∃ (x₀ : ℝ), 
    (∃ (t : ℝ), (Real.exp x₀ - a * Real.sqrt t) / (x₀ - t) = Real.exp x₀ ∧ 
                Real.exp x₀ = a / (2 * Real.sqrt t) ∧ 
                t > 0) 
    → 0 < a ∧ a ≤ Real.sqrt (2 * Real.exp 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_range_l533_53302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_satisfies_conditions_l533_53394

/-- A plane in 3D space --/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The distance from a point to a plane --/
noncomputable def distance_point_to_plane (p : Plane) (x y z : ℝ) : ℝ :=
  (abs (p.a * x + p.b * y + p.c * z + p.d)) / 
  (Real.sqrt (p.a^2 + p.b^2 + p.c^2))

/-- Check if a plane contains a line defined by the intersection of two other planes --/
def plane_contains_line (p : Plane) (p1 : Plane) (p2 : Plane) : Prop :=
  ∃ (a b : ℝ), p.a = a * p1.a + b * p2.a ∧
                p.b = a * p1.b + b * p2.b ∧
                p.c = a * p1.c + b * p2.c ∧
                p.d = a * p1.d + b * p2.d

theorem plane_equation_satisfies_conditions :
  let p1 : Plane := ⟨1, 2, 3, -2⟩
  let p2 : Plane := ⟨1, -1, 1, -3⟩
  let p : Plane := ⟨5, -11, 1, -17⟩
  (plane_contains_line p p1 p2) ∧
  (p ≠ p1) ∧ (p ≠ p2) ∧
  (distance_point_to_plane p 3 1 (-1) = 2 / Real.sqrt 3) ∧
  (p.a > 0) ∧
  (Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs (Int.floor p.a)) (Int.natAbs (Int.floor p.b))) 
                    (Int.natAbs (Int.floor p.c))) 
           (Int.natAbs (Int.floor p.d)) = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_satisfies_conditions_l533_53394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_360_l533_53383

noncomputable def geometricSequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

noncomputable def geometricSum (a₁ : ℝ) (q : ℝ) (m n : ℕ) : ℝ :=
  a₁ * q^(m-1) * (1 - q^(n-m+1)) / (1 - q)

theorem geometric_sequence_sum_360 :
  ∀ m n : ℕ, m < n →
  geometricSum 3 2 m n = 360 →
  n = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_360_l533_53383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_value_from_cosine_l533_53371

theorem sine_value_from_cosine (α : ℝ) 
  (h1 : Real.cos (α + π/6) = 1/3) 
  (h2 : 0 < α) 
  (h3 : α < π) : 
  Real.sin α = (2 * Real.sqrt 6 - 1) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_value_from_cosine_l533_53371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_days_worked_l533_53319

/-- The factory produces this many toys per week -/
def weekly_production : ℕ := 8000

/-- The workers produce this many toys per day -/
def daily_production : ℕ := 2000

/-- The number of days worked per week -/
def days_worked : ℕ := weekly_production / daily_production

theorem correct_days_worked :
  days_worked = 4 := by
  unfold days_worked weekly_production daily_production
  norm_num

#eval days_worked -- This will output 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_days_worked_l533_53319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_value_l533_53376

theorem angle_value (α : Real) 
  (h1 : Real.sin α = -Real.sqrt 2 / 2)
  (h2 : π / 2 < α)
  (h3 : α < 3 * π / 2) :
  α = 5 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_value_l533_53376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_empty_time_l533_53390

-- Define the fill time without leak
def fill_time_no_leak : ℚ := 5

-- Define the fill time with leak
def fill_time_with_leak : ℚ := 6

-- Define the function to calculate the time to empty the cistern
def time_to_empty (fill_time_no_leak : ℚ) (fill_time_with_leak : ℚ) : ℚ :=
  (fill_time_no_leak * fill_time_with_leak) / (fill_time_with_leak - fill_time_no_leak)

-- Theorem statement
theorem leak_empty_time :
  time_to_empty fill_time_no_leak fill_time_with_leak = 30 := by
  -- Unfold the definition of time_to_empty
  unfold time_to_empty
  -- Perform the calculation
  norm_num
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_empty_time_l533_53390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l533_53345

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the function y = 2⌊x⌋ + 1
noncomputable def f (x : ℝ) : ℤ :=
  2 * (floor x) + 1

-- Define the domain
def domain : Set ℝ :=
  { x : ℝ | -1 ≤ x ∧ x < 3 }

-- Define the range
def range : Set ℤ :=
  { y : ℤ | ∃ x ∈ domain, f x = y }

-- Theorem statement
theorem range_of_f : range = {-1, 1, 3, 5} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l533_53345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_values_l533_53304

-- Define the lines l1 and l2
def l1 (a : ℝ) : ℝ → ℝ → Prop := λ x y ↦ a * x + (a + 2) * y + 1 = 0
def l2 (a : ℝ) : ℝ → ℝ → Prop := λ x y ↦ x + a * y - 2 = 0

-- Define what it means for two lines to be parallel
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (m b₁ b₂ : ℝ), ∀ x y, (f x y ↔ y = m * x + b₁) ∧ (g x y ↔ y = m * x + b₂)

-- State the theorem
theorem parallel_lines_a_values :
  ∀ a : ℝ, parallel (l1 a) (l2 a) → a = -1 ∨ a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_values_l533_53304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l533_53397

noncomputable section

-- Define the hyperbola C
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the point P
def P : ℝ × ℝ := (1, Real.sqrt 3)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the right focus F
def F (c : ℝ) : ℝ × ℝ := (c, 0)

-- State the theorem
theorem hyperbola_equation (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧
  (∃ k : ℝ, P.1 / k = P.2 / (b / a)) ∧  -- P is on the asymptote
  c = 4 ∧  -- Distance from center to focus
  (F c).1^2 + (F c).2^2 + P.1^2 + P.2^2 = ((F c).1 - P.1)^2 + ((F c).2 - P.2)^2  -- ∠FPO = 90°
  →
  hyperbola 2 (2 * Real.sqrt 3) = hyperbola a b :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l533_53397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_consecutive_good_numbers_l533_53385

-- Define what a good number is
def is_good_number (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (∃ k : ℕ, k > 1 ∧ p^k ∣ n)

-- Define the sequence
def a : ℕ → ℕ
  | 0 => 8
  | n + 1 => 4 * a n * (a n + 1)

-- State the theorem
theorem infinitely_many_consecutive_good_numbers :
  ∃ (a : ℕ → ℕ), ∀ n : ℕ, n ≥ 1 →
    is_good_number (a n) ∧
    is_good_number (a n + 1) ∧
    a (n + 1) = 4 * a n * (a n + 1) :=
by
  -- Use the sequence defined above
  use a
  -- Introduce variables
  intro n hn
  -- Split the goal into three parts
  constructor
  · sorry -- Proof that a n is a good number
  constructor
  · sorry -- Proof that a n + 1 is a good number
  · -- Proof of the recurrence relation
    cases n
    · -- Base case: n = 0
      simp [a]
    · -- Inductive case: n = m + 1
      simp [a]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_consecutive_good_numbers_l533_53385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_quadrilateral_is_rectangle_l533_53321

/-- A rhombus with diagonals of length 6 and 10 -/
structure Rhombus :=
  (diagonal1 : ℝ)
  (diagonal2 : ℝ)
  (is_positive : diagonal1 > 0 ∧ diagonal2 > 0)
  (diag_lengths : diagonal1 = 6 ∧ diagonal2 = 10)

/-- A quadrilateral represented by its four vertices -/
structure Quadrilateral (α : Type*) :=
  (v1 v2 v3 v4 : α)

/-- A quadrilateral formed by connecting the midpoints of a rhombus's sides -/
def midpoint_quadrilateral (r : Rhombus) : Quadrilateral ℝ :=
  sorry

/-- Predicate to check if a quadrilateral is a rectangle -/
def IsRectangle (q : Quadrilateral ℝ) : Prop :=
  sorry

/-- Function to get the side length of a quadrilateral -/
def QuadrilateralSide (q : Quadrilateral ℝ) (i : Fin 4) : ℝ :=
  sorry

/-- Theorem stating that the midpoint quadrilateral is a rectangle with sides 3 and 5 -/
theorem midpoint_quadrilateral_is_rectangle (r : Rhombus) :
  let q := midpoint_quadrilateral r
  IsRectangle q ∧ 
  (∃ (s1 s2 : ℝ), s1 = 3 ∧ s2 = 5 ∧ 
    (QuadrilateralSide q 0 = s1 ∧ QuadrilateralSide q 2 = s1) ∧
    (QuadrilateralSide q 1 = s2 ∧ QuadrilateralSide q 3 = s2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_quadrilateral_is_rectangle_l533_53321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_increase_ratio_l533_53309

/-- Represents a right circular cone filled with water -/
structure WaterCone where
  baseRadius : ℝ
  waterHeight : ℝ

/-- Represents a spherical marble -/
structure Marble where
  radius : ℝ

/-- Calculates the volume of water in a cone -/
noncomputable def coneVolume (cone : WaterCone) : ℝ :=
  (1/3) * Real.pi * cone.baseRadius^2 * cone.waterHeight

/-- Calculates the volume of a marble -/
noncomputable def marbleVolume (marble : Marble) : ℝ :=
  (4/3) * Real.pi * marble.radius^3

/-- Theorem: The ratio of water level increase is 32:1 -/
theorem water_level_increase_ratio :
  ∀ (narrowCone wideCone : WaterCone) (narrowMarble wideMarble : Marble),
    narrowCone.baseRadius = 4 →
    wideCone.baseRadius = 8 →
    coneVolume narrowCone = coneVolume wideCone →
    narrowMarble.radius = 2 →
    wideMarble.radius = 1 →
    let narrowNewHeight := narrowCone.waterHeight + (marbleVolume narrowMarble) / ((1/3) * Real.pi * narrowCone.baseRadius^2)
    let wideNewHeight := wideCone.waterHeight + (marbleVolume wideMarble) / ((1/3) * Real.pi * wideCone.baseRadius^2)
    (narrowNewHeight - narrowCone.waterHeight) / (wideNewHeight - wideCone.waterHeight) = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_increase_ratio_l533_53309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_l533_53398

/-- Predicate stating that triangle ABC is equilateral -/
def IsEquilateral (A B C : ℝ × ℝ) : Prop := sorry

/-- Predicate stating that O is the center of the inscribed circle of triangle ABC -/
def IsInscribedCircle (O A B C : ℝ × ℝ) : Prop := sorry

/-- Function calculating the area of a circle with center O -/
def CircleArea (O : ℝ × ℝ) : ℝ := sorry

/-- Function calculating the area of triangle ABC -/
def TriangleArea (A B C : ℝ × ℝ) : ℝ := sorry

/-- Given an equilateral triangle ABC with inscribed circle of area 4π, prove its area is 12√3 -/
theorem equilateral_triangle_area (A B C O : ℝ × ℝ) : 
  IsEquilateral A B C →
  IsInscribedCircle O A B C →
  CircleArea O = 4 * Real.pi →
  TriangleArea A B C = 12 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_l533_53398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_problem_l533_53384

noncomputable def S : ℝ := (4/3)^(1/3) - 1

theorem sum_problem (m n : ℕ) (h1 : (S + 1)^3 = m/n) (h2 : Nat.Coprime m n) :
  10 * m + n = 43 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_problem_l533_53384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l533_53396

noncomputable def f (x : ℝ) (q : ℝ → ℝ) : ℝ := (x^3 + x^2 - 4*x - 4) / (q x)

noncomputable def q (x : ℝ) : ℝ := (4/3) * x^2 - 4/3

theorem q_satisfies_conditions :
  (∀ x, x ≠ 1 ∧ x ≠ -1 → q x ≠ 0) ∧
  (q 1 = 0) ∧
  (q (-1) = 0) ∧
  (q 4 = 20) ∧
  (∃ C > 0, ∀ x, abs x > C → abs (f x q) > 1) :=
by
  sorry

#check q_satisfies_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l533_53396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sum_l533_53320

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 9 = 1

-- Define a point on the ellipse
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse x y

-- Define symmetry with respect to the origin
def symmetric (p q : PointOnEllipse) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

-- Define the distance between two points
noncomputable def distance (p1 p2 : PointOnEllipse) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem max_distance_sum (p q r : PointOnEllipse) (h : symmetric p q) :
  distance r p + distance r q ≤ 10 ∧ ∃ (p' q' r' : PointOnEllipse), symmetric p' q' ∧ distance r' p' + distance r' q' = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sum_l533_53320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_sequence_l533_53347

theorem parabola_sequence (n : ℕ) : 
  2 * ((3^n - 1) / 2) - 3 * (3^(n-1)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_sequence_l533_53347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l533_53355

noncomputable def f (x : Real) : Real := Real.sin x * Real.cos x + Real.cos x ^ 2

noncomputable def g (x : Real) : Real := f (x - Real.pi/8)

theorem f_properties :
  (∃ (T : Real), T > 0 ∧ T = Real.pi ∧ ∀ (x : Real), f (x + T) = f x) ∧
  (∀ (k : Int), ∀ (x : Real), f (Real.pi/8 + k * Real.pi/2 + x) = f (Real.pi/8 + k * Real.pi/2 - x)) ∧
  (∀ (x₀ : Real), g x₀ ≥ 1 ↔ ∃ (k : Int), Real.pi/8 + k * Real.pi ≤ x₀ ∧ x₀ ≤ 3*Real.pi/8 + k * Real.pi) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l533_53355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_incircle_areas_formula_l533_53317

/-- Given a triangle ABC with sides a, b, c, this function calculates the sum of the areas of the
    incircle of ABC and the incircles of three smaller triangles formed by tangents parallel to
    the sides of ABC. -/
noncomputable def sumOfIncircleAreas (a b c : ℝ) : ℝ :=
  (b + c - a) * (c + a - b) * (a + b - c) * (a^2 + b^2 + c^2) * Real.pi / (a + b + c)^3

/-- Theorem stating that the sum of the areas of all four incircles in the described configuration
    is equal to the formula given by sumOfIncircleAreas. -/
theorem sum_of_incircle_areas_formula (a b c : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) :
    ∃ (actualSumOfAreas : ℝ → ℝ → ℝ → ℝ), actualSumOfAreas a b c = sumOfIncircleAreas a b c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_incircle_areas_formula_l533_53317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_value_l533_53393

theorem integral_value (a : ℝ) (h1 : a > 0) 
  (h2 : (Nat.choose 6 2) * a^4 = 15) : 
  ∫ x in (-a)..a, (x^2 + x + Real.sqrt (4 - x^2)) = 2/3 + 2*Real.pi/3 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_value_l533_53393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smoking_probability_l533_53346

theorem smoking_probability (p5 p10 : ℝ) (h1 : p5 = 0.02) (h2 : p10 = 0.16) :
  (1 - p10) / (1 - p5) = 6 / 7 := by
  have pA := 1 - p5
  have pB := 1 - p10
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smoking_probability_l533_53346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_B_and_complement_of_A_l533_53373

def U : Set Int := {x | -1 ≤ x ∧ x ≤ 5}
def A : Set Int := {1, 2, 5}
def B : Set Int := {x | 0 ≤ x ∧ x < 4}

theorem intersection_of_B_and_complement_of_A :
  B ∩ (U \ A) = {0, 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_B_and_complement_of_A_l533_53373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_C₁_and_C₂_l533_53313

-- Define the curves C₁ and C₂
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (4 + 5 * Real.cos t, 5 + 5 * Real.sin t)

noncomputable def C₂ (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Define the intersection points in polar coordinates
def intersection_points : Set (ℝ × ℝ) := {(2, Real.pi/2), (Real.sqrt 2, Real.pi/4)}

-- Theorem statement
theorem intersection_of_C₁_and_C₂ :
  ∀ (ρ θ : ℝ), ρ > 0 → 0 ≤ θ → θ < 2*Real.pi →
  (∃ t : ℝ, C₁ t = (ρ * Real.cos θ, ρ * Real.sin θ)) ∧ (C₂ θ = ρ) →
  (ρ, θ) ∈ intersection_points := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_C₁_and_C₂_l533_53313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_three_l533_53340

-- Define the function f(x) = a^x - 1
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x - 1

-- Theorem statement
theorem inverse_f_at_three (a : ℝ) (h : f a 1 = 1) : 
  ∃ (finv : ℝ → ℝ), Function.LeftInverse finv (f a) ∧ Function.RightInverse finv (f a) ∧ finv 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_three_l533_53340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_one_solution_sum_sum_of_b_values_l533_53336

theorem quadratic_one_solution_sum (b : ℝ) : 
  (∃! x, 3 * x^2 + b * x + 12 * x + 27 = 0) →
  (b = 6 ∨ b = -30) :=
by sorry

theorem sum_of_b_values : 
  Finset.sum {6, -30} id = -24 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_one_solution_sum_sum_of_b_values_l533_53336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_existence_l533_53380

-- Define a sequence as a function from ℕ to ℝ
def Sequence := ℕ → ℝ

-- Define what it means for an interval to be a feeding trough
def IsFeedingTrough (s : Sequence) (a b : ℝ) : Prop :=
  ∀ l : ℝ, l > 0 → (∃ n : ℕ, (Set.Finite {k : ℕ | a ≤ s k ∧ s k ≤ b} → Finset.card (Finset.filter (fun k => a ≤ s k ∧ s k ≤ b) (Finset.range n)) > ↑n * l))

theorem sequence_existence :
  (∃ s : Sequence, ∀ a b : ℝ, a < b → ¬IsFeedingTrough s a b) ∧
  (∃ s : Sequence, ∀ a b : ℝ, a < b → IsFeedingTrough s a b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_existence_l533_53380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_equals_two_sqrt_five_fifths_l533_53364

-- Define the angle α and point P
noncomputable def α : Real := Real.arccos (1 / 2)
noncomputable def P : (Real × Real) := (Real.cos (Real.pi / 3), 1)

-- State the theorem
theorem sin_alpha_equals_two_sqrt_five_fifths :
  P.1 = Real.cos (Real.pi / 3) →
  P.2 = 1 →
  Real.sin α = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_equals_two_sqrt_five_fifths_l533_53364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_value_l533_53375

theorem ab_value (a b : ℝ) (h1 : (3 : ℝ)^a = (81 : ℝ)^(b+2)) (h2 : (125 : ℝ)^b = (5 : ℝ)^(a-3)) : a * b = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_value_l533_53375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt3_over_3_l533_53316

/-- An equilateral triangle with vertices A, B, and C -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_equilateral : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) =
                   Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) ∧
                   Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) =
                   Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)

/-- An ellipse with foci F1 and F2, passing through points P and Q -/
structure Ellipse where
  F1 : ℝ × ℝ
  F2 : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  passes_through : Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) +
                   Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) =
                   Real.sqrt ((Q.1 - F1.1)^2 + (Q.2 - F1.2)^2) +
                   Real.sqrt ((Q.1 - F2.1)^2 + (Q.2 - F2.2)^2)

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  let c := Real.sqrt ((e.F1.1 - e.F2.1)^2 + (e.F1.2 - e.F2.2)^2) / 2
  let a := (Real.sqrt ((e.P.1 - e.F1.1)^2 + (e.P.2 - e.F1.2)^2) +
            Real.sqrt ((e.P.1 - e.F2.1)^2 + (e.P.2 - e.F2.2)^2)) / 2
  c / a

theorem ellipse_eccentricity_sqrt3_over_3
  (t : EquilateralTriangle)
  (e : Ellipse)
  (h1 : e.F1 = t.A)
  (h2 : e.F2.1 = (t.B.1 + t.C.1) / 2 ∧ e.F2.2 = (t.B.2 + t.C.2) / 2)
  (h3 : e.P = t.B ∧ e.Q = t.C) :
  eccentricity e = Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt3_over_3_l533_53316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_odd_dice_l533_53391

/-- The number of sides on a standard die -/
def numSides : ℕ := 6

/-- The set of outcomes for a single die roll -/
def dieOutcomes : Finset ℕ := Finset.range numSides

/-- Predicate for odd numbers -/
def isOdd (n : ℕ) : Bool := n % 2 = 1

/-- The set of odd outcomes for a single die roll -/
def oddOutcomes : Finset ℕ := dieOutcomes.filter (fun n => isOdd n)

/-- The probability of rolling two odd numbers with two fair dice -/
theorem prob_two_odd_dice :
  (oddOutcomes.card : ℚ) * oddOutcomes.card / (numSides * numSides) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_odd_dice_l533_53391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_components_for_circuit_reliability_l533_53386

theorem min_components_for_circuit_reliability (n : ℕ) : 
  (∀ m : ℕ, m < n → 1 - (1/2 : ℝ)^m < 0.95) ∧ 
  (1 - (1/2 : ℝ)^n ≥ 0.95) → 
  n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_components_for_circuit_reliability_l533_53386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boy_girl_ratio_l533_53339

/-- Represents the number of students in the class -/
def total_students : ℕ := 30

/-- Represents the difference between the number of boys and girls -/
def boy_girl_difference : ℕ := 3

/-- Theorem stating that the ratio of boys to girls is 16/13 -/
theorem boy_girl_ratio :
  ∃ (boys girls : ℕ),
    boys + girls = total_students ∧
    boys = girls + boy_girl_difference ∧
    (boys : ℚ) / (girls : ℚ) = 16 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boy_girl_ratio_l533_53339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baron_munchausen_claim_l533_53351

def is_ten_digit (n : ℕ) : Prop := 10^9 ≤ n ∧ n < 10^10

def sum_of_squares_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).map (λ d ↦ d^2) |>.sum

theorem baron_munchausen_claim :
  ∃ (a b : ℕ), 
    is_ten_digit a ∧ 
    is_ten_digit b ∧
    a ≠ b ∧
    a % 10 ≠ 0 ∧
    b % 10 ≠ 0 ∧
    a - sum_of_squares_of_digits a = b - sum_of_squares_of_digits b :=
by
  sorry

#check baron_munchausen_claim

end NUMINAMATH_CALUDE_ERRORFEEDBACK_baron_munchausen_claim_l533_53351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l533_53331

def geometric_sequence (a₁ : ℕ) (r : ℕ) : ℕ → ℕ
  | 0 => a₁
  | n + 1 => r * geometric_sequence a₁ r n

def sum_of_geometric_sequence (a₁ : ℕ) (r : ℕ) : ℕ → ℕ
  | 0 => 0
  | n + 1 => geometric_sequence a₁ r n + sum_of_geometric_sequence a₁ r n

theorem sequence_properties :
  (geometric_sequence 1 2 4 = 16) ∧ (sum_of_geometric_sequence 1 2 8 = 255) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l533_53331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_three_four_l533_53378

-- Define the nabla operation
noncomputable def nabla (a b : ℝ) : ℝ := (a^2 + b^2) / (1 + a^2 * b^2)

-- Theorem statement
theorem nabla_three_four :
  nabla 3 4 = 25 / 145 := by
  -- Unfold the definition of nabla
  unfold nabla
  -- Simplify the expression
  simp [pow_two]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_three_four_l533_53378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_theorem_l533_53382

theorem divisibility_theorem (a b m n : ℕ) 
  (ha : a > 1) 
  (hcoprime : Nat.Coprime a b) 
  (hdiv : (a^m + b^m) % (a^n + b^n) = 0) : 
  n ∣ m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_theorem_l533_53382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_solution_set_part2_a_range_l533_53388

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 3|

-- Part 1
theorem part1_solution_set :
  {x : ℝ | f 2 x ≥ 2 * x} = Set.Iic (5/2) :=
sorry

-- Part 2
theorem part2_a_range :
  ∀ a : ℝ, (∃ x : ℝ, f a x ≤ (1/2) * a + 5) → a ∈ Set.Icc (-16/3) 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_solution_set_part2_a_range_l533_53388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_incorrect_expression_l533_53311

def incorrect_expression : ℤ → Prop :=
  fun n => (n = 6 - (-6)) → n ≠ 0

def correct_expressions : ℤ → Prop :=
  fun n => (n = 1 - 5 → n = -4) ∧
           (n = 0 - 3 → n = -3) ∧
           (n = -15 - (-5) → n = -10)

theorem identify_incorrect_expression :
  ∃ n, incorrect_expression n ∧
       ∀ m, correct_expressions m :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_incorrect_expression_l533_53311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_decreasing_l533_53395

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2)

theorem f_is_even_and_decreasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_decreasing_l533_53395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_range_l533_53379

theorem y_range (m n k : ℝ) (hm : m ≥ 0) (hn : n ≥ 0) (hk : k ≥ 0)
  (h1 : m - k + 1 = 2*k + n) (h2 : 2*k + n = 1) :
  ∀ y₀, 2*k^2 - 8*k + 6 = y₀ → (5/2 ≤ y₀ ∧ y₀ ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_range_l533_53379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_l533_53334

/-- The height of a right pyramid with an equilateral triangular base -/
theorem pyramid_height (perimeter : ℝ) (apex_distance : ℝ) (h1 : perimeter = 24) (h2 : apex_distance = 10) :
  let side_length := perimeter / 3
  let triangle_height := (Real.sqrt 3 / 2) * side_length
  let centroid_distance := (2 / 3) * triangle_height
  Real.sqrt (apex_distance^2 - centroid_distance^2) = Real.sqrt 94.6667 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_l533_53334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_xy_bounds_l533_53308

theorem sum_xy_bounds (x y : ℝ) (h1 : 1 - Real.sqrt (x - 1) = Real.sqrt (y - 1)) 
  (h2 : x ≥ 1) (h3 : y ≥ 1) : 
  (∃ (a b : ℝ), a + b = 3 ∧ x + y ≤ a + b) ∧ 
  (∃ (c d : ℝ), c + d = 5/2 ∧ x + y ≥ c + d) := by
  sorry

#check sum_xy_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_xy_bounds_l533_53308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discontinuities_l533_53318

noncomputable section

-- Function 1
def f1 (x : ℝ) : ℝ :=
  if x ≠ 2 then 3 * x else 1

-- Function 2
noncomputable def f2 (x : ℝ) : ℝ :=
  (x^2 - 16) / (x - 4)

-- Function 3
def f3 (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 else x + 1

-- Function 4
noncomputable def f4 (x : ℝ) : ℝ :=
  abs (x - 1) / (x - 1)

theorem discontinuities :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (δ : ℝ), δ > 0 → ∃ (x : ℝ), abs (x - 2) < δ ∧ abs (f1 x - f1 2) ≥ ε) ∧
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (δ : ℝ), δ > 0 → ∃ (x : ℝ), abs (x - 4) < δ ∧ abs (f2 x - 8) ≥ ε) ∧
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (δ : ℝ), δ > 0 → ∃ (x : ℝ), abs x < δ ∧ abs (f3 x - f3 0) ≥ ε) ∧
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (δ : ℝ), δ > 0 → ∃ (x : ℝ), abs (x - 1) < δ ∧ abs (f4 x - f4 1) ≥ ε) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discontinuities_l533_53318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l533_53322

noncomputable def f (x : ℝ) := Real.cos x * Real.sin (x + Real.pi/3) - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 4

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧ T = Real.pi) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → f x ≤ 1/4) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → f x ≥ -1/2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) ∧ f x = 1/4) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) ∧ f x = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l533_53322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_sum_theorem_l533_53330

/-- Represents a rational number as a repeating decimal in a given base. -/
def RepeatingDecimal (numerator : ℕ) (base : ℕ) : ℚ :=
  numerator / (base^2 - 1)

theorem base_sum_theorem (R₁ R₃ : ℕ) (F₁ F₂ : ℚ) :
  R₁ > 1 ∧ R₃ > 1 ∧
  F₁ = RepeatingDecimal 45 R₁ ∧
  F₂ = RepeatingDecimal 54 R₁ ∧
  F₁ = RepeatingDecimal 36 R₃ ∧
  F₂ = RepeatingDecimal 63 R₃ →
  R₁ + R₃ = 20 := by
  sorry

#check base_sum_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_sum_theorem_l533_53330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_angle_l533_53362

/-- The circle equation: x^2 + y^2 - 4x + 3 = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

/-- A line passing through the origin -/
def line_through_origin (k : ℝ) (x y : ℝ) : Prop := y = k * x

/-- The line is tangent to the circle -/
def is_tangent (k : ℝ) : Prop :=
  ∃ x y, circle_equation x y ∧ line_through_origin k x y ∧
  ∀ x' y', circle_equation x' y' ∧ line_through_origin k x' y' → (x', y') = (x, y)

/-- The angle of inclination of the line -/
noncomputable def angle_of_inclination (k : ℝ) : ℝ := Real.arctan k

theorem tangent_line_angle :
  ∃ k, is_tangent k ∧
  (angle_of_inclination k = π/6 ∨ angle_of_inclination k = 5*π/6) := by
  sorry

#check tangent_line_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_angle_l533_53362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_distance_points_l533_53325

/-- The number of points on the parabola y² = 4x that are at a distance of √2/2 from the line y = x -/
theorem parabola_line_distance_points : 
  ∃! (points : Finset (ℝ × ℝ)), 
    Finset.card points = 3 ∧ 
    (∀ p ∈ points, let (x, y) := p; y^2 = 4*x ∧ |x - y| = 1) ∧
    (∀ p : ℝ × ℝ, (let (x, y) := p; y^2 = 4*x ∧ |x - y| = 1) → p ∈ points) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_distance_points_l533_53325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_two_segment_trip_l533_53366

/-- Calculate the average speed of a two-segment trip -/
theorem average_speed_two_segment_trip 
  (local_distance : ℝ) 
  (local_speed : ℝ) 
  (highway_distance : ℝ) 
  (highway_speed : ℝ) 
  (h1 : local_distance = 60) 
  (h2 : local_speed = 20) 
  (h3 : highway_distance = 120) 
  (h4 : highway_speed = 60) : 
  (local_distance + highway_distance) / ((local_distance / local_speed) + (highway_distance / highway_speed)) = 36 := by
  sorry

#check average_speed_two_segment_trip

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_two_segment_trip_l533_53366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_notes_count_l533_53343

theorem total_notes_count 
  (red_rows : ℕ) (notes_per_row : ℕ) (blue_per_red : ℕ) (additional_blue : ℕ)
  (h1 : red_rows = 5)
  (h2 : notes_per_row = 6)
  (h3 : blue_per_red = 2)
  (h4 : additional_blue = 10) :
  red_rows * notes_per_row + red_rows * notes_per_row * blue_per_red + additional_blue = 100 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_notes_count_l533_53343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_sequences_l533_53358

def word : String := "FREQUENCY"

def is_vowel (c : Char) : Bool :=
  c = 'A' || c = 'E' || c = 'I' || c = 'O' || c = 'U'

def valid_sequence (s : List Char) : Bool :=
  s.length = 5 &&
  s.head? = some 'F' &&
  s.getLast? = some 'Y' &&
  s.toFinset.card = 5 &&
  (s.get? 1).map is_vowel = some true &&
  s.toFinset ⊆ word.toList.toFinset

theorem count_valid_sequences :
  (List.filter valid_sequence (List.permutations word.toList)).length = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_sequences_l533_53358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carson_speed_l533_53312

/-- Represents the running speed of a person in miles per hour -/
def RunningSpeed : Type := ℝ

/-- Represents the time taken for a trip in minutes -/
def TripTime : Type := ℝ

/-- Represents the distance of a trip in miles -/
def TripDistance : Type := ℝ

/-- Given:
  * Jerry can run to school and back in the time it takes Carson to run to school
  * Jerry's one-way trip to school takes 15 minutes
  * The distance from Jerry's house to school is 4 miles
  Prove that Carson's running speed is 8 miles per hour -/
theorem carson_speed (jerry_oneway : TripTime) (distance : TripDistance) :
  jerry_oneway = (15 : ℝ) →
  distance = (4 : ℝ) →
  ∃ (carson_speed : RunningSpeed), carson_speed = (8 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carson_speed_l533_53312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_dot_product_l533_53399

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (2*x + 2) / (2*x - 4)

-- Define the point P
def P : ℝ × ℝ := (2, 1)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem intersection_dot_product :
  ∀ (l : ℝ → ℝ × ℝ) (A B : ℝ × ℝ),
    (∃ (t : ℝ), l t = P) →  -- l passes through P
    (∃ (t₁ t₂ : ℝ), l t₁ = A ∧ l t₂ = B ∧ 
      A.2 = f A.1 ∧ B.2 = f B.1) →  -- A and B are intersections of l and f
    (A.1 + B.1, A.2 + B.2) • P = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_dot_product_l533_53399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_intersection_l533_53314

def A : Finset ℕ := {1, 2, 3, 5}
def B : Finset ℕ := {2, 3, 5, 6, 7}

theorem number_of_subsets_of_intersection : Finset.card (Finset.powerset (A ∩ B)) = 8 := by
  -- Calculate A ∩ B
  have h_intersection : A ∩ B = {2, 3, 5} := by rfl
  
  -- Count the elements in the powerset of A ∩ B
  have h_count : Finset.card (Finset.powerset {2, 3, 5}) = 8 := by rfl
  
  -- Use the above facts to prove the theorem
  rw [h_intersection, h_count]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_intersection_l533_53314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l533_53324

noncomputable def f (x : ℝ) : ℝ := Real.sin (x^2)

theorem f_derivative :
  deriv f = fun x => 2 * x * Real.cos (x^2) := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l533_53324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_vectors_magnitude_l533_53323

/-- Given three points in 3D space and the origin, prove that the magnitude of the sum of the vectors from the origin to these points is 2√3. -/
theorem sum_of_vectors_magnitude (P₁ P₂ P₃ O : ℝ × ℝ × ℝ) : 
  P₁ = (1, 1, 0) → P₂ = (0, 1, 1) → P₃ = (1, 0, 1) → O = (0, 0, 0) →
  ‖(P₁.1 - O.1, P₁.2.1 - O.2.1, P₁.2.2 - O.2.2) + 
    (P₂.1 - O.1, P₂.2.1 - O.2.1, P₂.2.2 - O.2.2) + 
    (P₃.1 - O.1, P₃.2.1 - O.2.1, P₃.2.2 - O.2.2)‖ = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_vectors_magnitude_l533_53323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_perimeter_sum_l533_53361

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The pentagon vertices -/
def pentagon : List Point := [
  ⟨0, 0⟩, ⟨1, 3⟩, ⟨3, 3⟩, ⟨4, 0⟩, ⟨2, -1⟩
]

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the perimeter of the pentagon -/
noncomputable def perimeter : ℝ :=
  let edges := List.zip pentagon (pentagon.rotateLeft 1)
  edges.foldl (fun acc (p1, p2) => acc + distance p1 p2) 0

/-- The theorem to be proved -/
theorem pentagon_perimeter_sum :
  ∃ (p q r : ℕ), perimeter = p + q * Real.sqrt 10 + r * Real.sqrt 13 ∧ p + q + r = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_perimeter_sum_l533_53361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinearity_condition_l533_53344

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Two vectors are non-collinear if they are linearly independent -/
def NonCollinear (a b : V) : Prop := LinearIndependent ℝ ![a, b]

/-- Three points are collinear if the vector from the first to the third 
    is a scalar multiple of the vector from the first to the second -/
def AreCollinear (A B C : V) : Prop := ∃ k : ℝ, C - A = k • (B - A)

theorem collinearity_condition 
  (a b : V) (hab : NonCollinear a b) 
  (A B C : V) (l m : ℝ) 
  (hAB : B - A = l • a + b) 
  (hAC : C - A = a + m • b) :
  AreCollinear A B C ↔ l * m = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinearity_condition_l533_53344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_configuration_theorem_l533_53350

structure CircleConfiguration where
  A : ℝ → ℝ → Prop  -- Circle A
  B : ℝ → ℝ → Prop  -- Circle B
  C : ℝ → ℝ → Prop  -- Circle C
  D : ℝ → ℝ → Prop  -- Circle D
  E : ℝ → ℝ → Prop  -- Circle E
  F : ℝ → ℝ → Prop  -- Circle F
  T : ℝ → ℝ → Prop  -- Equilateral triangle T

-- Define these as axioms or assumptions since we don't have their implementations
axiom IsEquilateral : (ℝ → ℝ → Prop) → Prop
axiom IsInscribedIn : (ℝ → ℝ → Prop) → (ℝ → ℝ → Prop) → Prop
axiom IsInternallyTangentTo : (ℝ → ℝ → Prop) → (ℝ → ℝ → Prop) → Prop
axiom IsExternallyTangentTo : (ℝ → ℝ → Prop) → (ℝ → ℝ → Prop) → Prop

def is_valid_configuration (config : CircleConfiguration) : Prop :=
  ∃ (m n : ℕ), 
    (∀ x y, config.A x y ↔ (x^2 + y^2 = 12^2)) ∧
    (∀ x y, config.B x y ↔ (x^2 + y^2 = 5^2)) ∧
    (∀ x y, config.C x y ↔ (x^2 + y^2 = 3^2)) ∧
    (∀ x y, config.D x y ↔ (x^2 + y^2 = 2^2)) ∧
    (∀ x y, config.E x y ↔ (x^2 + y^2 = (m/n : ℝ)^2)) ∧
    (∀ x y, config.F x y ↔ (x^2 + y^2 = 1^2)) ∧
    (IsEquilateral config.T) ∧
    (IsInscribedIn config.T config.A) ∧
    (IsInternallyTangentTo config.B config.A) ∧
    (IsInternallyTangentTo config.C config.A) ∧
    (IsInternallyTangentTo config.D config.A) ∧
    (IsExternallyTangentTo config.B config.E) ∧
    (IsExternallyTangentTo config.C config.E) ∧
    (IsExternallyTangentTo config.D config.E) ∧
    (IsInternallyTangentTo config.F config.A) ∧
    (IsExternallyTangentTo config.F config.E) ∧
    (Nat.Coprime m n)

theorem circle_configuration_theorem (config : CircleConfiguration) :
  is_valid_configuration config → ∃ (m n : ℕ), m + n = 23 ∧ (m : ℝ) / n = 21 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_configuration_theorem_l533_53350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_l533_53307

noncomputable def a : ℝ := 8.05678
noncomputable def b : ℝ := 8.056 + 7 / 9 * (1 / 1000)
noncomputable def c : ℝ := 8.05 + 67 / 99 * (1 / 100)
noncomputable def d : ℝ := 8.0 + 567 / 999 * (1 / 10)
noncomputable def e : ℝ := 8 + 567 / 9999

theorem largest_number : a > b ∧ a > c ∧ a > d ∧ a > e :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_l533_53307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_104th_group_is_2072_l533_53374

def sequenceNum (n : ℕ) : ℕ := 2 * n + 1

def group_size (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 4
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | _ => 0

def sum_of_104th_group : ℕ → Prop
| sum => ∃ start : ℕ,
    (start + 1) * 4 = 104 ∧
    sum = sequenceNum (start * 10 + 256) + sequenceNum (start * 10 + 257) +
          sequenceNum (start * 10 + 258) + sequenceNum (start * 10 + 259)

theorem sum_of_104th_group_is_2072 : sum_of_104th_group 2072 := by
  sorry

#eval sequenceNum 257 + sequenceNum 258 + sequenceNum 259 + sequenceNum 260

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_104th_group_is_2072_l533_53374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_odd_if_sigma_eq_2n_plus_1_l533_53372

/-- Sum of divisors function -/
def sigma (n : ℕ) : ℕ := (Finset.sum (Nat.divisors n) id)

/-- Main theorem -/
theorem square_of_odd_if_sigma_eq_2n_plus_1 (n : ℕ) (h : n > 0) :
  sigma n = 2 * n + 1 → ∃ k : ℕ, k % 2 = 1 ∧ n = k^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_odd_if_sigma_eq_2n_plus_1_l533_53372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solution_l533_53329

theorem diophantine_equation_solution (a b n : ℕ) (p : ℕ) (h_prime : Nat.Prime p) :
  a ^ 2013 + b ^ 2013 = p ^ n →
  ∃ k : ℕ, a = 2 ^ k ∧ b = 2 ^ k ∧ n = 2013 * k + 1 ∧ p = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solution_l533_53329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_of_the_year_eligibility_l533_53359

/-- The number of members in the Cinematic Academy -/
def academy_members : ℕ := 765

/-- The fraction of lists a film must appear in to be considered for "movie of the year" -/
def required_fraction : ℚ := 1/4

/-- The smallest number of top-10 lists a film can appear on to be considered for "movie of the year" -/
def min_lists : ℕ := 192

theorem movie_of_the_year_eligibility :
  min_lists = (Nat.ceil ((academy_members : ℚ) * required_fraction) : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_of_the_year_eligibility_l533_53359
