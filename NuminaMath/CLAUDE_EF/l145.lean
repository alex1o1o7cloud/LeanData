import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_circle_radius_l145_14512

def circle_radii (n : ℕ) (a d : ℝ) : ℕ → ℝ := λ i ↦ a + i * d

theorem middle_circle_radius 
  (n : ℕ) 
  (h_n : n = 5) 
  (a d : ℝ) 
  (h_a : a = 6) 
  (h_largest : circle_radii n a d (n - 1) = 20) :
  circle_radii n a d ((n - 1) / 2) = 13 :=
by
  sorry

#eval circle_radii 5 6 3.5 2  -- This should evaluate to 13

end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_circle_radius_l145_14512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_k_equals_13_l145_14597

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_k_equals_13 :
  ∀ k : ℕ,
  k > 0 →
  let a := arithmetic_sequence (-3) ((3/2 + 3) / k)
  let S := arithmetic_sum (-3) ((3/2 + 3) / k)
  (a (k + 1) = 3/2) → (S k = -12) → (k = 13) := by
  sorry

#check arithmetic_sequence_k_equals_13

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_k_equals_13_l145_14597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_l145_14546

/-- The fixed point through which M1M2 passes for a parabola y^2 = 2px -/
noncomputable def fixed_point (p a b : ℝ) : ℝ × ℝ := (a, 2 * p * a / b)

/-- Parabola equation -/
def on_parabola (p : ℝ) (point : ℝ × ℝ) : Prop :=
  point.2^2 = 2 * p * point.1

/-- Check if three points are collinear -/
def collinear (A B C : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (B.1 - A.1)

/-- Theorem stating that M1M2 passes through a fixed point -/
theorem line_passes_through_fixed_point 
  (p a b : ℝ) 
  (h_ab : a * b ≠ 0) 
  (h_b : b^2 ≠ 2 * p * a) 
  (M M1 M2 : ℝ × ℝ) 
  (h_M : on_parabola p M) 
  (h_M1 : on_parabola p M1) 
  (h_M2 : on_parabola p M2) 
  (h_AM : collinear (a, b) M M1) 
  (h_BM : collinear (-a, 0) M M2) 
  (h_M1M2 : M1 ≠ M2) :
  ∃ (t : ℝ), (1 - t) • M1 + t • M2 = fixed_point p a b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_l145_14546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_l145_14521

def num_balls : ℕ := 4
def num_boxes : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k
def arrange (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Define the function for the number of ways to distribute balls
def number_of_ways_to_distribute_balls (balls boxes : ℕ) : ℕ :=
  choose balls 2 * arrange boxes boxes

theorem ball_distribution :
  number_of_ways_to_distribute_balls num_balls num_boxes =
  choose num_balls 2 * arrange num_boxes num_boxes :=
by
  -- Unfold the definition of number_of_ways_to_distribute_balls
  unfold number_of_ways_to_distribute_balls
  -- The equality now holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_l145_14521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_below_zero_notation_l145_14524

/-- Represents temperature in Celsius -/
structure Temperature where
  value : ℝ

/-- The zero point on the Celsius scale -/
def zero : Temperature := ⟨0⟩

/-- Represents 1°C above zero -/
def one_above_zero : Temperature := ⟨1⟩

/-- Represents 1°C below zero -/
def one_below_zero : Temperature := ⟨-1⟩

/-- States that 1°C above zero is denoted as +1°C -/
axiom above_zero_notation : one_above_zero.value = 1

/-- Theorem: If 1°C above zero is denoted as +1°C, then 1°C below zero should be denoted as -1°C -/
theorem below_zero_notation : one_below_zero.value = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_below_zero_notation_l145_14524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_die_three_given_sum_seven_l145_14528

/-- The probability of one die showing 3, given that the sum of two fair dice is 7 -/
theorem prob_one_die_three_given_sum_seven : ∃ (p : Real), p = 1 / 3 := by
  -- Define the sample space of all possible outcomes when rolling two dice
  let sample_space : Finset (Nat × Nat) := sorry

  -- Define the event where the sum of the dice is 7
  let sum_seven : Finset (Nat × Nat) := sorry

  -- Define the event where one die shows 3 and the sum is 7
  let one_three_and_sum_seven : Finset (Nat × Nat) := sorry

  -- Define the probability function
  let prob : Finset (Nat × Nat) → Real := sorry

  -- The probability of one die showing 3 given the sum is 7
  -- is equal to the probability of (one die showing 3 and sum is 7) 
  -- divided by the probability of (sum is 7)
  let prob_one_three_given_sum_seven : Real :=
    prob one_three_and_sum_seven / prob sum_seven

  -- The theorem statement
  have h : prob_one_three_given_sum_seven = 1 / 3 := by sorry
  
  -- Conclude the theorem
  exact ⟨prob_one_three_given_sum_seven, h⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_die_three_given_sum_seven_l145_14528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_strictly_increasing_implies_a_range_l145_14544

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (4 - a/2)*x + 2

-- State the theorem
theorem function_strictly_increasing_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) →
  a ∈ Set.Icc 4 8 ∧ a ≠ 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_strictly_increasing_implies_a_range_l145_14544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l145_14519

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h : 2 * t.a * Real.cos t.C + t.c = 2 * t.b) :
  t.A = π / 3 ∧ 
  (t.a = 1 → 2 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l145_14519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_face_prob_is_five_sixths_l145_14508

/-- A dodecahedron with blue and red faces -/
structure Dodecahedron where
  blue_faces : ℕ
  red_faces : ℕ
  total_faces_eq_12 : blue_faces + red_faces = 12

/-- The probability of rolling a blue face on a dodecahedron -/
def blue_face_probability (d : Dodecahedron) : ℚ :=
  d.blue_faces / 12

/-- Theorem: The probability of rolling a blue face on a dodecahedron
    with 10 blue faces and 2 red faces is 5/6 -/
theorem blue_face_prob_is_five_sixths :
  ∃ d : Dodecahedron, d.blue_faces = 10 ∧ d.red_faces = 2 ∧ blue_face_probability d = 5/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_face_prob_is_five_sixths_l145_14508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_theorem_l145_14573

def basketball_problem (team_scores : List ℕ) (num_losses : ℕ) : Prop :=
  let total_games := team_scores.length
  let loss_diff := 2
  let win_ratio := 3
  ∃ (opponent_scores : List ℕ),
    (total_games = 8) ∧
    (team_scores = [2, 4, 6, 8, 10, 12, 14, 16]) ∧
    (num_losses = 3) ∧
    (opponent_scores.length = total_games) ∧
    (opponent_scores.sum = 36) ∧
    (∃ (loss_indices : List ℕ),
      (loss_indices.length = num_losses) ∧
      (∀ i ∈ loss_indices, i < opponent_scores.length ∧ i < team_scores.length ∧ 
        opponent_scores[i]! = team_scores[i]! + loss_diff)) ∧
    (∃ (win_indices : List ℕ),
      (win_indices.length = total_games - num_losses) ∧
      (∀ i ∈ win_indices, i < team_scores.length ∧ i < opponent_scores.length ∧ 
        team_scores[i]! = win_ratio * opponent_scores[i]!))

theorem basketball_theorem : 
  basketball_problem [2, 4, 6, 8, 10, 12, 14, 16] 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_theorem_l145_14573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l145_14562

/-- Given a curve and a line with specific properties, prove that the difference of reciprocals of parameters equals 1/2 -/
theorem curve_line_intersection (a b : ℝ) (P Q : ℝ × ℝ) : 
  a ≠ 0 → b ≠ 0 → a ≠ b → -- a and b are non-zero and different
  (∀ x y, y^2 / b - x^2 / a = 1 ↔ x + y = 2) → -- curve and line equations are equivalent
  y^2 / b - x^2 / a = 1 → -- P and Q satisfy the curve equation
  P.1 + P.2 = 2 → Q.1 + Q.2 = 2 → -- P and Q satisfy the line equation
  P.1 * Q.1 + P.2 * Q.2 = 0 → -- dot product of OP and OQ is zero
  1 / b - 1 / a = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l145_14562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_l145_14523

noncomputable def f (x a : ℝ) := Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2 + a

theorem min_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x a ≥ -1) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x a = -1) →
  a = -1 := by
  sorry

#check min_value_implies_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_l145_14523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_physics_score_l145_14588

-- Define the scores for each subject
variable (P C M : ℝ)

-- Define the conditions
def total_average (P C M : ℝ) : Prop := (P + C + M) / 3 = 75
def physics_math_average (P M : ℝ) : Prop := (P + M) / 2 = 90
def physics_chem_average (P C : ℝ) : Prop := (P + C) / 2 = 70

-- Theorem to prove
theorem physics_score : 
  total_average P C M → physics_math_average P M → physics_chem_average P C → P = 95 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_physics_score_l145_14588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_monotonicity_max_difference_l145_14541

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - 2*a*x
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := x^2 + 2*b*x

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a
noncomputable def g' (b : ℝ) (x : ℝ) : ℝ := 2*x + 2*b

theorem opposite_monotonicity_max_difference (a b : ℝ) :
  a > 0 →
  (∀ x ∈ Set.Ioo a b, f' a x * g' b x ≤ 0) →
  (b - a ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_monotonicity_max_difference_l145_14541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_length_is_80_l145_14531

/-- Represents a rectangular lawn with intersecting roads -/
structure LawnWithRoads where
  width : ℕ
  roadWidth : ℕ
  roadArea : ℕ

/-- Calculates the length of the lawn given its properties -/
def calculateLawnLength (lawn : LawnWithRoads) : ℕ :=
  (lawn.roadArea + lawn.roadWidth * lawn.roadWidth - lawn.roadWidth * lawn.width) / lawn.roadWidth

/-- Theorem stating that the length of the lawn is 80 meters -/
theorem lawn_length_is_80 (lawn : LawnWithRoads) 
    (h1 : lawn.width = 60)
    (h2 : lawn.roadWidth = 10)
    (h3 : lawn.roadArea = 1300) : 
  calculateLawnLength lawn = 80 := by
  sorry

#eval calculateLawnLength { width := 60, roadWidth := 10, roadArea := 1300 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_length_is_80_l145_14531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_2_4_neg1_eq_neg1_l145_14594

/-- The function g as defined in the problem -/
noncomputable def g (a b c : ℝ) : ℝ := (c^2 + a) / (c^2 - b)

/-- Theorem stating that g(2, 4, -1) = -1 -/
theorem g_2_4_neg1_eq_neg1 : g 2 4 (-1) = -1 := by
  -- Unfold the definition of g
  unfold g
  -- Simplify the expression
  simp [pow_two]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_2_4_neg1_eq_neg1_l145_14594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_number_l145_14501

theorem fraction_of_number (x : ℝ) : 
  (3 / 4 : ℝ) * (1 / 2 : ℝ) * (2 / 5 : ℝ) * x = 759.0000000000001 ↔ x = 5060.000000000001 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_number_l145_14501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_in_cube_is_4_sqrt_3_l145_14558

/-- The length of a line segment within a cube -/
noncomputable def segment_length_in_cube (x y : ℝ × ℝ × ℝ) (cube_edge : ℝ) : ℝ :=
  let (x1, y1, z1) := x
  let (x2, y2, z2) := y
  let cube_start := (0, 0, 4)
  let cube_end := (4, 4, 8)
  let (ex1, ey1, ez1) := (max x1 0, max y1 0, max z1 4)
  let (ex2, ey2, ez2) := (min x2 4, min y2 4, min z2 8)
  Real.sqrt ((ex2 - ex1)^2 + (ey2 - ey1)^2 + (ez2 - ez1)^2)

theorem segment_length_in_cube_is_4_sqrt_3 :
  segment_length_in_cube (0, 0, 0) (5, 5, 13) 4 = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_in_cube_is_4_sqrt_3_l145_14558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_with_avg_distance_pi_half_unique_constant_avg_distance_l145_14581

/-- Arc distance between two points on a unit circle -/
noncomputable def arc_distance (x y : ℝ) : ℝ :=
  min (abs (x - y)) (2 * Real.pi - abs (x - y))

/-- Average arc distance from a point to a set of points on a unit circle -/
noncomputable def avg_arc_distance (x : ℝ) (points : Finset ℝ) : ℝ :=
  (points.sum (λ y => arc_distance x y)) / points.card

/-- Existence of a point with average arc distance π/2 for any finite set of points -/
theorem exists_point_with_avg_distance_pi_half (points : Finset ℝ) :
  ∃ x : ℝ, avg_arc_distance x points = Real.pi / 2 := by
  sorry

/-- Uniqueness of π/2 as the constant average arc distance -/
theorem unique_constant_avg_distance :
  ∃! α : ℝ, ∀ (points : Finset ℝ), ∃ x : ℝ, avg_arc_distance x points = α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_with_avg_distance_pi_half_unique_constant_avg_distance_l145_14581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_subset_complement_implies_m_range_l145_14580

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 ≤ 0}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*m*x + m^2 - 4 ≤ 0}

-- Define the complement of B
def complement_B (m : ℝ) : Set ℝ := {x : ℝ | ¬(x ∈ B m)}

-- Theorem 1
theorem intersection_implies_m_value :
  ∀ m : ℝ, (A ∩ B m = Set.Icc 0 3) → m = 2 := by sorry

-- Theorem 2
theorem subset_complement_implies_m_range :
  ∀ m : ℝ, (A ⊆ complement_B m) → (m > 5 ∨ m < -3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_subset_complement_implies_m_range_l145_14580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheet_metal_not_volume_l145_14556

/-- Represents a cylindrical ventilation duct -/
structure CylindricalDuct where
  radius : ℝ
  height : ℝ

/-- Calculates the lateral surface area of a cylindrical duct -/
noncomputable def lateralSurfaceArea (duct : CylindricalDuct) : ℝ :=
  2 * Real.pi * duct.radius * duct.height

/-- Calculates the volume of a cylindrical duct -/
noncomputable def volume (duct : CylindricalDuct) : ℝ :=
  Real.pi * duct.radius^2 * duct.height

/-- Theorem stating that the amount of sheet metal required for an open-ended
    cylindrical ventilation duct is not equal to its volume -/
theorem sheet_metal_not_volume (duct : CylindricalDuct) :
  lateralSurfaceArea duct ≠ volume duct := by
  sorry

#check sheet_metal_not_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheet_metal_not_volume_l145_14556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighteen_mn_equals_p_2n_q_m_l145_14582

theorem eighteen_mn_equals_p_2n_q_m (m n : ℤ) (P Q : ℕ) 
  (h1 : P = (3 : ℕ)^(m.toNat)) (h2 : Q = (2 : ℕ)^(n.toNat)) : 
  (18 : ℕ)^((m*n).toNat) = P^((2*n).toNat) * Q^(m.toNat) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighteen_mn_equals_p_2n_q_m_l145_14582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_separated_l145_14548

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Determines if two circles are separated -/
def are_separated (c1 c2 : Circle) : Prop :=
  distance c1.center c2.center > c1.radius + c2.radius

/-- The first circle from the problem -/
def circle1 : Circle :=
  { center := (-1, -3), radius := 1 }

/-- The second circle from the problem -/
def circle2 : Circle :=
  { center := (3, -1), radius := 3 }

/-- Theorem stating that the two circles are separated -/
theorem circles_are_separated : are_separated circle1 circle2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_separated_l145_14548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_14_division_fifty_six_is_smallest_l145_14560

def is_n_division (n : ℕ) (A : Set ℕ) (division : Fin n → Set ℕ) : Prop :=
  (⋂ i, division i) = A ∧ ∀ i j, (division i ∩ division j).Nonempty

def has_required_pair (S : Set ℕ) : Prop :=
  ∃ a b, a ∈ S ∧ b ∈ S ∧ b < a ∧ a ≤ 4/3 * (b : ℚ)

theorem smallest_m_for_14_division : 
  ∀ m : ℕ, m ≥ 56 →
    ∀ division : Fin 14 → Set ℕ, 
      is_n_division 14 (Finset.range m).toSet division →
        ∃ i : Fin 14, has_required_pair (division i) :=
sorry

theorem fifty_six_is_smallest : 
  ∀ m : ℕ, m < 56 →
    ∃ division : Fin 14 → Set ℕ, 
      is_n_division 14 (Finset.range m).toSet division ∧
        ∀ i : Fin 14, ¬(has_required_pair (division i)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_14_division_fifty_six_is_smallest_l145_14560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_heads_before_three_tails_sum_of_fraction_parts_l145_14579

/-- The probability of getting heads in a fair coin flip -/
noncomputable def p_heads : ℝ := 1 / 2

/-- The probability of getting tails in a fair coin flip -/
noncomputable def p_tails : ℝ := 1 / 2

/-- The probability of encountering 4 heads before 3 tails in repeated fair coin flips -/
noncomputable def q : ℝ := 2 / 11

theorem four_heads_before_three_tails : q = 2 / 11 := by
  -- The proof is omitted for now
  sorry

/-- The sum of numerator and denominator when q is expressed as a fraction m/n -/
def m_plus_n : ℕ := 13

theorem sum_of_fraction_parts : m_plus_n = 13 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_heads_before_three_tails_sum_of_fraction_parts_l145_14579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l145_14564

noncomputable section

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := x^3 + 1/x

-- Define the derivative of the curve
noncomputable def f' (x : ℝ) : ℝ := 3*x^2 - 1/x^2

-- Define the point of tangency
def x₀ : ℝ := -1
noncomputable def y₀ : ℝ := f x₀

-- Theorem statement
theorem tangent_line_equation :
  ∀ x y : ℝ, (y - y₀ = f' x₀ * (x - x₀)) ↔ (2*x - y = 0) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l145_14564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adriatic_tyrrhenian_bijection_l145_14506

def is_adriatic (s : List Nat) : Prop :=
  s.head? = some 1 ∧ 
  ∀ i, i + 1 < s.length → s[i+1]! ≥ 2 * s[i]!

def is_tyrrhenian (s : List Nat) (n : Nat) : Prop :=
  s.getLast? = some n ∧
  ∀ i, i + 1 < s.length → s[i+1]! > (s.take (i+1)).sum

def adriatic_sequences (n : Nat) : Set (List Nat) :=
  {s | is_adriatic s ∧ ∀ x ∈ s, x ≤ n}

def tyrrhenian_sequences (n : Nat) : Set (List Nat) :=
  {s | is_tyrrhenian s n ∧ ∀ x ∈ s, x ≤ n}

theorem adriatic_tyrrhenian_bijection (n : Nat) :
  ∃ f : adriatic_sequences n → tyrrhenian_sequences n, Function.Bijective f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_adriatic_tyrrhenian_bijection_l145_14506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AC_through_origin_l145_14509

noncomputable section

variable (p : ℝ) (h : p > 0)

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the focus
def focus : ℝ × ℝ := (p/2, 0)

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -p/2

-- Define points A, B, and C
variable (A B C : ℝ × ℝ)

-- A and B lie on the parabola
variable (hA : parabola p A.1 A.2)
variable (hB : parabola p B.1 B.2)

-- F, A, and B are collinear
variable (hFAB : ∃ (t : ℝ), A = focus p + t • (B - focus p))

-- C lies on the directrix
variable (hC : directrix p C.1)

-- BC is parallel to the x-axis
variable (hBC : B.2 = C.2)

-- Theorem statement
theorem line_AC_through_origin :
  ∃ (k : ℝ), A = k • C :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AC_through_origin_l145_14509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_at_one_l145_14572

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 / exp x) + exp x

-- State the theorem
theorem second_derivative_at_one :
  (deriv (deriv f)) 1 = exp 1 - (1 / exp 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_at_one_l145_14572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l145_14510

-- Define the system of inequalities
def system (x y : ℝ) : Prop :=
  x + 2*y ≤ 6 ∧ 3*x + y ≥ 3 ∧ x ≤ 4 ∧ y ≥ 0

-- Define the polygonal region
def polygonal_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | system p.1 p.2}

-- Define the length of a line segment
noncomputable def segment_length (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

-- Statement to prove
theorem longest_side_length :
  ∃ (a b : ℝ × ℝ), a ∈ polygonal_region ∧ b ∈ polygonal_region ∧
    (∀ (c d : ℝ × ℝ), c ∈ polygonal_region → d ∈ polygonal_region →
      segment_length c d ≤ segment_length a b) ∧
    segment_length a b = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l145_14510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l145_14511

theorem range_of_a (x₁ x₂ m a : ℝ) : 
  (x₁^2 - m*x₁ - 2 = 0) →
  (x₂^2 - m*x₂ - 2 = 0) →
  (m ∈ Set.Icc (-1 : ℝ) 1) →
  (¬ ∀ m ∈ Set.Icc (-1 : ℝ) 1, a^2 - 5*a - 3 ≥ |x₁ - x₂|) →
  (∃! x : ℝ, x^2 + 2*Real.sqrt 2*a*x + 11*a ≤ 0) →
  (a = 0 ∨ a = 11/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l145_14511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_rate_calculation_l145_14570

/-- Given a rectangular plot with specified dimensions and fencing cost, 
    calculate the rate per meter for fencing. -/
theorem fencing_rate_calculation 
  (width : ℝ)
  (length : ℝ)
  (length_eq : length = width + 10)
  (perimeter_eq : 2 * (length + width) = 220)
  (total_cost : ℝ)
  (total_cost_eq : total_cost = 1430) :
  total_cost / (2 * (length + width)) = 6.5 :=
by
  sorry

#check fencing_rate_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_rate_calculation_l145_14570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_point_distance_sum_l145_14527

-- Define the triangle and point P
noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (12, 0)
noncomputable def C : ℝ × ℝ := (4, 6)
noncomputable def P : ℝ × ℝ := (5, 3)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem fermat_point_distance_sum :
  distance P A + distance P B + distance P C = Real.sqrt 34 + Real.sqrt 58 + Real.sqrt 10 ∧
  (1 : ℝ) + 1 + 1 = 3 := by
  sorry

#check fermat_point_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_point_distance_sum_l145_14527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_lent_is_10000_l145_14578

/-- Proves that the sum lent is 10000, given the specified conditions -/
theorem sum_lent_is_10000 
  (interest_rate : ℝ) 
  (loan_duration : ℕ)
  (interest_difference : ℝ) 
  (h1 : interest_rate = 0.075)
  (h2 : loan_duration = 7)
  (h3 : interest_difference = 4750) :
  (interest_difference / (1 - interest_rate * (loan_duration : ℝ))) = 10000 := by
  sorry

#check sum_lent_is_10000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_lent_is_10000_l145_14578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_a_in_range_l145_14561

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then Real.log x / Real.log a
  else if x > 1 then (3*a - 1)*x + a/2
  else 0  -- undefined for x ≤ 0

theorem f_decreasing_iff_a_in_range (a : ℝ) :
  (∀ x₁ x₂, x₁ < x₂ → f a x₁ > f a x₂) ↔ (0 < a ∧ a ≤ 2/7) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_a_in_range_l145_14561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_104th_bracket_l145_14534

/-- The sequence term for a given index -/
def sequenceTerm (n : ℕ) : ℕ := 2 * n + 1

/-- The number of terms in a bracket given its position -/
def termsInBracket (position : ℕ) : ℕ :=
  match position % 4 with
  | 0 => 4
  | n => n

/-- The starting index for a given bracket position -/
def bracketStartIndex (position : ℕ) : ℕ :=
  (position - 1) * (position - 1) / 2 + 1

/-- Sum of terms in a bracket given its position -/
def bracketSum (position : ℕ) : ℕ :=
  let start := bracketStartIndex position
  let terms := termsInBracket position
  (List.range terms).map (fun i => sequenceTerm (start + i)) |>.sum

theorem sum_of_104th_bracket :
  bracketSum 104 = 2072 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_104th_bracket_l145_14534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_mold_radius_l145_14586

/-- The volume of a hemisphere with radius r -/
noncomputable def hemisphereVolume (r : ℝ) : ℝ := (2/3) * Real.pi * r^3

/-- The radius of the large hemisphere-shaped bowl -/
def largeRadius : ℝ := 2

/-- The number of smaller hemisphere-shaped molds -/
def numSmallMolds : ℕ := 64

theorem small_mold_radius :
  ∃ (r : ℝ), 
    r > 0 ∧
    numSmallMolds * hemisphereVolume r = hemisphereVolume largeRadius ∧
    r = 1/2 := by
  -- Introduce the small radius
  let r : ℝ := 1/2
  
  -- Prove existence
  use r
  
  -- Prove the three conditions
  constructor
  · -- r > 0
    norm_num
  
  constructor
  · -- numSmallMolds * hemisphereVolume r = hemisphereVolume largeRadius
    -- This step requires computation, which we'll skip for now
    sorry
  
  · -- r = 1/2
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_mold_radius_l145_14586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_properties_l145_14518

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2

-- Define the inverse function of f
noncomputable def f_inverse : ℝ → ℝ := Function.invFun f

-- Theorem statement
theorem inverse_f_properties :
  (∀ x, f_inverse (-x) = -f_inverse x) ∧ 
  (∀ x y, 0 < x ∧ x < y → f_inverse x < f_inverse y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_properties_l145_14518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l145_14545

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b * x + 1

-- State the theorem
theorem function_properties (a b : ℝ) 
  (h : (deriv (f a b)) 1 = 2) :
  a * b ≤ 1 ∧ a^2 + b^2 ≥ 2 ∧ (3 : ℝ)^a + (3 : ℝ)^b ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l145_14545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_l145_14577

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - (1 / 2) * a * x^2

-- State the theorem
theorem f_has_two_zeros (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ ∀ z : ℝ, f a z = 0 → z = x ∨ z = y) ↔ a ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_l145_14577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l145_14592

noncomputable def ε : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

def M : Set ℂ := {-3, -3 * ε, -3 * ε^2}

def P (m : ℂ) (x y z : ℂ) : ℂ := x^3 + y^3 + z^3 + m*x*y*z

def LinearTrinomial (a b c : ℂ) (x y z : ℂ) : ℂ := a*x + b*y + c*z

theorem polynomial_factorization (m : ℂ) : 
  (∃ a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ : ℂ, 
    ∀ x y z : ℂ, P m x y z = 
      LinearTrinomial a₁ b₁ c₁ x y z * 
      LinearTrinomial a₂ b₂ c₂ x y z * 
      LinearTrinomial a₃ b₃ c₃ x y z) ↔ 
  m ∈ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l145_14592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l145_14569

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x)

-- State the theorem
theorem omega_range (ω : ℝ) (h1 : ω > 0) :
  (∀ x ∈ Set.Icc (-π/3) (π/4), f ω x ≥ -2) ∧
  (∃ x ∈ Set.Icc (-π/3) (π/4), f ω x = -2) →
  ω ∈ Set.Ioo 0 (3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l145_14569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l145_14549

/-- The center of a circle with diameter endpoints (-1,-4) and (-7,6) is located at (-4, 1) -/
theorem circle_center (M : Set (ℝ × ℝ)) : 
  (∃ (a b c d : ℝ), (a, b) ∈ M ∧ (c, d) ∈ M ∧ 
   (a, b) = (-1, -4) ∧ (c, d) = (-7, 6) ∧
   (∀ (x y : ℝ), (x, y) ∈ M → (x - a)^2 + (y - b)^2 = (c - a)^2 + (d - b)^2)) →
  (∃ (x y : ℝ), (x, y) ∈ M ∧ 
   (∀ (p q : ℝ), (p, q) ∈ M → (p - x)^2 + (q - y)^2 ≤ (c - a)^2 + (d - b)^2) ∧
   x = -4 ∧ y = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l145_14549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l145_14505

/-- Given a function g such that g(3x) = 3 / (3 + 2x) for all x > 0,
    prove that 3g(x) = 27 / (9 + 2x) -/
theorem function_equality (g : ℝ → ℝ) (h : ∀ x > 0, g (3 * x) = 3 / (3 + 2 * x)) :
  ∀ x > 0, 3 * g x = 27 / (9 + 2 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l145_14505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_ratio_l145_14507

/-- Given a rectangle with sides s and 3s, the ratio of its longer diagonal to the diagonal of a square with side s is √5. -/
theorem rectangle_diagonal_ratio (s : ℝ) (h : s > 0) : 
  (Real.sqrt (s^2 + (3*s)^2)) / (Real.sqrt (2 * s^2)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_ratio_l145_14507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_curve_min_distance_achievable_l145_14566

/-- The curve equation -/
def curve_equation (x y : ℝ) : Prop :=
  x * y - (5/2) * x - 2 * y + 3 = 0

/-- The distance from the origin to a point (x, y) -/
noncomputable def distance_from_origin (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

/-- The theorem stating the minimum distance from the origin to any point on the curve -/
theorem min_distance_to_curve :
  ∀ x y : ℝ, curve_equation x y → distance_from_origin x y ≥ Real.sqrt (5/4) := by
  sorry

/-- The theorem stating that the minimum distance is achievable -/
theorem min_distance_achievable :
  ∃ x y : ℝ, curve_equation x y ∧ distance_from_origin x y = Real.sqrt (5/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_curve_min_distance_achievable_l145_14566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_sum_exists_l145_14584

theorem power_of_two_sum_exists (H : Finset ℕ) : 
  H ⊆ Finset.range 1999 → H.card = 1000 → 
  ∃ a b : ℕ, a ∈ H ∧ b ∈ H ∧ ∃ k : ℕ, a + b = 2^k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_sum_exists_l145_14584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_product_is_729_l145_14543

def mySequence : List ℚ := [1/3, 9/1, 1/27, 81/1, 1/243, 729/1, 1/2187, 6561/1, 1/19683, 59049/1, 1/177147, 531441/1]

theorem sequence_product_is_729 : 
  mySequence.prod = 729 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_product_is_729_l145_14543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l145_14563

theorem expression_evaluation (a : ℝ) (h : a ≠ 0) :
  (1 / 9) * a^0 + (1 / (9 * a))^0 - 81^(-(1/2 : ℝ)) - (-27)^(-(1/3 : ℝ)) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l145_14563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colony_ratio_rounded_l145_14585

/-- The ratio of colonies needed to ratify to total colonies -/
def colony_ratio : ℚ := 10 / 15

/-- Rounding a rational number to the nearest tenth -/
def round_to_tenth (q : ℚ) : ℚ := 
  (q * 10).floor / 10 + if (q * 10 - (q * 10).floor ≥ 1/2) then 1/10 else 0

/-- Theorem stating that the colony ratio rounded to the nearest tenth is 0.7 -/
theorem colony_ratio_rounded : round_to_tenth colony_ratio = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colony_ratio_rounded_l145_14585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flowerbed_fraction_l145_14522

/-- Represents the dimensions of a trapezoidal area in a rectangular yard --/
structure YardTrapezoid where
  shortSide : ℝ
  longSide : ℝ
  height : ℝ

/-- Calculates the area of an isosceles right triangle --/
noncomputable def isoscelesRightTriangleArea (legLength : ℝ) : ℝ :=
  (1 / 2) * legLength ^ 2

/-- Calculates the area of a rectangle --/
noncomputable def rectangleArea (length width : ℝ) : ℝ :=
  length * width

/-- Theorem: The fraction of a rectangular yard occupied by two congruent isosceles right triangular flower beds --/
theorem flowerbed_fraction (yard : YardTrapezoid) (h1 : yard.shortSide = 18) (h2 : yard.longSide = 30) (h3 : yard.height = 10) :
  let triangleLeg := (yard.longSide - yard.shortSide) / 2
  let flowerbedsArea := 2 * isoscelesRightTriangleArea triangleLeg
  let yardArea := rectangleArea ((yard.shortSide + yard.longSide) / 2) yard.height
  flowerbedsArea / yardArea = 3 / 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flowerbed_fraction_l145_14522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_l145_14587

/-- The speed of an object is its distance traveled divided by the time taken. -/
noncomputable def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

/-- Theorem: A bus traveling 201 meters in 3 seconds has a speed of 67 meters per second. -/
theorem bus_speed : speed 201 3 = 67 := by
  -- Unfold the definition of speed
  unfold speed
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_l145_14587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closed_disk_no_congruent_partition_l145_14515

open Set Real

structure ClosedDisk where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

def is_congruent (S T : Set (ℝ × ℝ)) : Prop :=
  ∃ f : ℝ × ℝ → ℝ × ℝ, Function.Bijective f ∧ ∀ x y : ℝ × ℝ, x ∈ S ∧ y ∈ S → dist (f x) (f y) = dist x y

theorem closed_disk_no_congruent_partition (D : ClosedDisk) :
  ¬∃ (H₁ H₂ : Set (ℝ × ℝ)), 
    (H₁ ∪ H₂ = {p : ℝ × ℝ | dist p D.center ≤ D.radius}) ∧ 
    (H₁ ∩ H₂ = ∅) ∧
    (is_congruent H₁ H₂) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closed_disk_no_congruent_partition_l145_14515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_exists_l145_14596

-- Define the function f(x) = (1/10)^x - x
noncomputable def f (x : ℝ) : ℝ := (1/10)^x - x

-- State the theorem
theorem intersection_point_exists : ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < 1/2 ∧ f x₀ = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_exists_l145_14596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_egg_supply_calculation_l145_14595

/-- Represents the number of dozen eggs supplied to the first store each day -/
def daily_supply_first_store : ℕ → ℕ := sorry

/-- Represents the daily supply to the second store in eggs -/
def daily_supply_second_store : ℕ := 30

/-- Represents the total weekly supply to both stores in eggs -/
def total_weekly_supply : ℕ := 630

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

theorem egg_supply_calculation :
  ∃ (d : ℕ), daily_supply_first_store d = 5 ∧
  d * 12 * days_in_week + daily_supply_second_store * days_in_week = total_weekly_supply :=
by
  use 5
  constructor
  · sorry  -- Proof that daily_supply_first_store 5 = 5
  · sorry  -- Proof that 5 * 12 * 7 + 30 * 7 = 630


end NUMINAMATH_CALUDE_ERRORFEEDBACK_egg_supply_calculation_l145_14595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_imply_equal_exponents_l145_14532

/-- Two terms are like terms if they have the same variables with the same exponents -/
def like_terms (term1 term2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ x y, ∃ c₁ c₂ : ℚ, term1 x y = c₁ * (x^(term1 x y).num * y^(term1 x y).den) ∧
                      term2 x y = c₂ * (x^(term2 x y).num * y^(term2 x y).den) ∧
                      (term1 x y).num = (term2 x y).num ∧
                      (term1 x y).den = (term2 x y).den

theorem like_terms_imply_equal_exponents (m : ℕ) :
  like_terms (λ x y ↦ 3 * x^m * y^3) (λ x y ↦ x^2 * y^3) → m = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_imply_equal_exponents_l145_14532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crystal_profit_maximum_l145_14533

/-- Profit function for Crystal Products Factory -/
noncomputable def f (n : ℕ) : ℝ := (10 + n) * (100 - 80 / Real.sqrt (n + 1 : ℝ)) - 100 * n

/-- The profit is maximized at n = 8 with a value of 520 million yuan -/
theorem crystal_profit_maximum :
  (∀ n : ℕ, f n ≤ 520) ∧ f 8 = 520 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crystal_profit_maximum_l145_14533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_snow_equation_initial_snow_is_six_l145_14559

/-- The amount of snow in inches on a given day -/
def snow_amount : ℕ → ℝ := sorry

/-- The initial amount of snow in inches -/
def initial_snow : ℝ := sorry

/-- The conditions of the snow problem -/
axiom snow_conditions :
  -- Day 2: 8 inches added
  snow_amount 2 = initial_snow + 8 ∧
  -- Day 4: 2 inches melted over 2 days
  snow_amount 4 = snow_amount 2 - 2 ∧
  -- Day 5: 2 times initial snow added
  snow_amount 5 = snow_amount 4 + 2 * initial_snow ∧
  -- Final amount is 24 inches (2 feet)
  snow_amount 5 = 24

/-- The theorem stating that the initial snow amount satisfies the equation -/
theorem initial_snow_equation : 3 * initial_snow + 6 = 24 := by
  sorry

/-- The theorem stating that the initial snow amount is 6 inches -/
theorem initial_snow_is_six : initial_snow = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_snow_equation_initial_snow_is_six_l145_14559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_theorem_l145_14575

theorem square_division_theorem (a p q : ℝ) (h_pos : a > 0) (h_p_pos : p > 0) (h_q_pos : q > 0) :
  let inner_square_area := (a^2 * (p^2 + q^2)) / ((p + q)^2)
  ∃ (P Q R S : ℝ × ℝ),
    P.1 = (p * a) / (p + q) ∧
    P.2 = 0 ∧
    Q.1 = a ∧
    Q.2 = (p * a) / (p + q) ∧
    R.1 = (q * a) / (p + q) ∧
    R.2 = a ∧
    S.1 = 0 ∧
    S.2 = (q * a) / (p + q) ∧
    (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (Q.1 - R.1)^2 + (Q.2 - R.2)^2 ∧
    (Q.1 - R.1)^2 + (Q.2 - R.2)^2 = (R.1 - S.1)^2 + (R.2 - S.2)^2 ∧
    (R.1 - S.1)^2 + (R.2 - S.2)^2 = (S.1 - P.1)^2 + (S.2 - P.2)^2 ∧
    (P.1 - R.1)^2 + (P.2 - R.2)^2 = (Q.1 - S.1)^2 + (Q.2 - S.2)^2 ∧
    inner_square_area = ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) := by
  sorry

#check square_division_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_theorem_l145_14575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_set_cardinality_l145_14535

-- Define a set A with cardinality α
variable (A : Type*) [Fintype A]
variable (α : ℕ)

-- Define the hypothesis that |A| = α
axiom h_card : Fintype.card A = α

-- State the theorem
theorem power_set_cardinality :
  Fintype.card (Set A) = 2^α ∧ 2^α > α := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_set_cardinality_l145_14535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_97_l145_14554

theorem square_of_97 : 97 * 97 = 9409 := by
  -- Define 97 as 100 - 3
  have h1 : 97 = 100 - 3 := by norm_num
  
  -- Use the square of a difference formula
  calc
    97 * 97 = (100 - 3) * (100 - 3) := by rw [h1]
    _       = 100 * 100 - 2 * 3 * 100 + 3 * 3 := by ring
    _       = 10000 - 600 + 9 := by norm_num
    _       = 9409 := by norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_97_l145_14554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l145_14517

-- Define the sign function
noncomputable def sign (α : ℝ) : ℤ :=
  if α > 0 then 1
  else if α < 0 then -1
  else 0

-- Define the system of equations
def satisfies_system (x y z : ℝ) : Prop :=
  x = 2018 - 2019 * (sign (y + z)) ∧
  y = 2018 - 2019 * (sign (z + x)) ∧
  z = 2018 - 2019 * (sign (x + y))

-- Theorem statement
theorem exactly_three_solutions :
  ∃! (s : Finset (ℝ × ℝ × ℝ)), s.card = 3 ∧ ∀ (x y z : ℝ), (x, y, z) ∈ s ↔ satisfies_system x y z :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l145_14517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l145_14550

/-- The set of triples satisfying the given conditions -/
def S : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p;
                    x = (y - z)^2 ∧ 
                    y = (x - z)^2 ∧ 
                    z = (x - y)^2}

/-- The set of solution triples -/
def T : Set (ℝ × ℝ × ℝ) :=
  {(0, 0, 0), (1, 1, 1), (1, 0, 1), (1, 1, 0), (0, 1, 1)}

theorem solution_set_equality : S = T := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l145_14550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_difference_in_S_l145_14530

def S : Set ℤ := {-20, -10, 0, 5, 15, 25}

theorem largest_difference_in_S : 
  (∀ x y : ℤ, x ∈ S → y ∈ S → x - y ≤ 45) ∧ (∃ a b : ℤ, a ∈ S ∧ b ∈ S ∧ a - b = 45) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_difference_in_S_l145_14530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_without_discount_is_30_percent_l145_14598

-- Define the discount rate
def discount_rate : ℚ := 5 / 100

-- Define the profit rate with discount
def profit_rate_with_discount : ℚ := 235 / 1000

-- Define the function to calculate the selling price with discount
noncomputable def selling_price_with_discount (cost_price : ℚ) : ℚ :=
  cost_price * (1 + profit_rate_with_discount)

-- Define the function to calculate the marked price before discount
noncomputable def marked_price (selling_price_with_discount : ℚ) : ℚ :=
  selling_price_with_discount / (1 - discount_rate)

-- Define the function to calculate the profit rate without discount
noncomputable def profit_rate_without_discount (cost_price : ℚ) : ℚ :=
  (marked_price (selling_price_with_discount cost_price) - cost_price) / cost_price

-- Theorem statement
theorem profit_without_discount_is_30_percent :
  ∀ cost_price : ℚ, cost_price > 0 →
    profit_rate_without_discount cost_price = 30 / 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_without_discount_is_30_percent_l145_14598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_transformed_cosine_l145_14555

noncomputable section

-- Define the original function
def f (x : ℝ) : ℝ := Real.cos (2 * x)

-- Define the shifted function
def g (x : ℝ) : ℝ := f (x - Real.pi / 20)

-- Define the final function after halving horizontal coordinates
def h (x : ℝ) : ℝ := g (2 * x)

-- Theorem statement
theorem symmetry_axis_of_transformed_cosine :
  ∃ (k : ℤ), h (Real.pi / 40 + k * Real.pi / 2) = h (Real.pi / 40 - k * Real.pi / 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_transformed_cosine_l145_14555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_min_perimeter_l145_14599

/-- Given a sector with constant area S, prove that the perimeter C is minimized
    when the central angle α is 2 radians, and the minimum perimeter is 4√S. -/
theorem sector_min_perimeter (S : ℝ) (hS : S > 0) :
  ∃ (α : ℝ), α > 0 ∧
  (∀ θ > 0, Real.sqrt (2*S*θ) + 2*Real.sqrt (2*S/θ) ≥ Real.sqrt (2*S*α) + 2*Real.sqrt (2*S/α)) ∧
  α = 2 ∧
  Real.sqrt (2*S*α) + 2*Real.sqrt (2*S/α) = 4*Real.sqrt S := by
  sorry

#check sector_min_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_min_perimeter_l145_14599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l145_14538

/-- Given a hyperbola with equation x²/a² - y²/3 = 1 (where a > 0) and eccentricity 2, prove that a = 1 -/
theorem hyperbola_eccentricity (a : ℝ) (h1 : a > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 3 = 1) → 
  (Real.sqrt (a^2 + 3) / a = 2) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l145_14538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l145_14516

/-- Given two lines l₁ and l₂ in the form of linear equations,
    prove that the value of 'a' is 1 when the lines are perpendicular. -/
theorem perpendicular_lines_a_value (a : ℝ) :
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    a * x₁ + y₁ + 3 = 0 → 
    x₂ + (2 * a - 3) * y₂ = 4 → 
    a + (2 * a - 3) = 0) →
  a = 1 :=
by
  intro h
  have : a + (2 * a - 3) = 0 := by
    apply h 0 (-3) 4 0
    · simp [mul_zero, add_zero]
    · simp [mul_zero, sub_zero]
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l145_14516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_2_l145_14537

/-- A power function passing through (2, √2/2) -/
noncomputable def f (x : ℝ) : ℝ := x^(-(1/2 : ℝ))

theorem inverse_f_at_2 :
  f 2 = Real.sqrt 2 / 2 → f⁻¹ 2 = (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_2_l145_14537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_price_for_livestream_sales_l145_14552

/-- Represents the sales data for a product -/
structure SalesData where
  cost_price : ℝ
  price_sales_relation : ℝ → ℝ
  profit_equation : ℝ → ℝ
  desired_profit : ℝ

/-- Checks if a given price maximizes profit while minimizing inventory -/
def is_optimal_price (data : SalesData) (price : ℝ) : Prop :=
  data.profit_equation price = data.desired_profit ∧
  ∀ p, p > price → data.profit_equation p ≤ data.desired_profit

/-- The main theorem stating the optimal price for the given sales data -/
theorem optimal_price_for_livestream_sales : 
  let data : SalesData := {
    cost_price := 10,
    price_sales_relation := λ x ↦ -10 * x + 400,
    profit_equation := λ x ↦ (x - 10) * (-10 * x + 400),
    desired_profit := 2160
  }
  is_optimal_price data 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_price_for_livestream_sales_l145_14552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_not_on_line_in_plane_l145_14557

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the subset and element relations
variable (subset : Line → Plane → Prop)
variable (belongs : Point → Plane → Prop)
variable (lies_on : Point → Line → Prop)

-- State the theorem
theorem point_not_on_line_in_plane 
  (A : Point) (a : Line) (α : Plane) :
  ¬(belongs A α) → subset a α → ¬(lies_on A a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_not_on_line_in_plane_l145_14557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_three_digit_integer_non_divisor_l145_14513

theorem greatest_three_digit_integer_non_divisor : ∃ (n : ℕ), n = 996 ∧ 
  (∀ m : ℕ, m > n → m < 1000 → (m * (2 * m + 1)) ∣ Nat.factorial m) ∧
  ¬((n * (2 * n + 1)) ∣ Nat.factorial n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_three_digit_integer_non_divisor_l145_14513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_equation_l145_14574

/-- Hyperbola C with foci F₁ and F₂, and points A and B satisfying specific conditions -/
structure HyperbolaC where
  a : ℝ
  b : ℝ
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_equation : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ↔ (x, y) = A ∨ (x, y) = B
  h_F₁_left : F₁.1 < 0
  h_F₂_right : 0 < F₂.1
  h_A_right : 0 < A.1
  h_B_left : B.1 < 0
  h_AF₁B_collinear : ∃ (t : ℝ), B = F₁ + t • (A - F₁)
  h_F₁A_3F₁B : A - F₁ = 3 • (B - F₁)
  h_OF₁_OA : F₁.1^2 + F₁.2^2 = A.1^2 + A.2^2

/-- The asymptotes of hyperbola C have the equation y = ±2x -/
theorem asymptotes_equation (h : HyperbolaC) : 
  ∃ (k : ℝ), k = 2 ∧ 
    (∀ (x y : ℝ), (y = k * x ∨ y = -k * x) ↔ 
      (∀ ε > 0, ∃ (x' y' : ℝ), 
        x'^2 / h.a^2 - y'^2 / h.b^2 = 1 ∧ 
        |x' - x| < ε ∧ 
        |y' - y| < ε)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_equation_l145_14574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_candidates_count_l145_14590

theorem exam_candidates_count (total : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : girls = 900)
  (h2 : total = boys + girls)
  (h3 : (34 : ℚ) / 100 * boys + (32 : ℚ) / 100 * girls = (1 - 669 / 1000) * total) :
  total = 2000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_candidates_count_l145_14590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l145_14536

-- Define the parameter a
variable (a : ℝ)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x + Real.exp 3 * a

-- Define the function g
noncomputable def g (x : ℝ) (x₀ : ℝ) : ℝ :=
  if x ≤ x₀ then x + a - (x - a) / Real.exp x
  else (1 - x) * Real.log x - a * (x + 1)

-- State the theorem
theorem problem_statement :
  (-6/5 ≤ a) → (a < 3/Real.exp 3 - 1) →
  ∃ x₀ : ℝ, (x₀ > 0) ∧ (f a x₀ = 0) ∧
  (3 < x₀) ∧ (x₀ < 4) ∧
  ∃ x₁ x₂ : ℝ, (x₁ < x₂) ∧ (g x₁ x₀ = 0) ∧ (g x₂ x₀ = 0) ∧
  (∀ x, (g x x₀ = 0) → (x = x₁ ∨ x = x₂)) ∧
  (Real.exp x₂ - x₂) / (Real.exp x₁ - x₁) > Real.exp ((x₁ + x₂) / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l145_14536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_semi_major_axis_l145_14571

/-- Given an ellipse with equation x^2 + my^2 = 1 and eccentricity √3/2, 
    its semi-major axis length is either 1 or 2. -/
theorem ellipse_semi_major_axis (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + m*y^2 = 1) →  -- ellipse equation
  (∃ (c a : ℝ), c/a = Real.sqrt 3/2 ∧ c^2 + (m*a^2) = 1) →  -- eccentricity condition
  (∃ (a : ℝ), a = 1 ∨ a = 2) := 
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_semi_major_axis_l145_14571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_proof_l145_14514

/-- The curve C in polar coordinates -/
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.sin θ * Real.cos θ, 2 * Real.sin θ * Real.sin θ)

/-- The line l parameterized by t -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (Real.sqrt 3 * t + Real.sqrt 3, -3 * t + 2)

/-- The shortest distance from curve C to line l -/
def shortest_distance : ℝ := 1

theorem shortest_distance_proof :
  ∀ θ t : ℝ,
  let (x, y) := curve_C θ
  let (lx, ly) := line_l t
  (x - lx)^2 + (y - ly)^2 ≥ shortest_distance^2 :=
by
  sorry

#check shortest_distance_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_proof_l145_14514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_side_count_l145_14591

def ConvexHexagon := Unit

theorem hexagon_side_count (ABCDEF : ConvexHexagon)
  (distinct_lengths : Nat)
  (side_AB : ℝ)
  (side_BC : ℝ)
  (perimeter : ℝ)
  (h1 : distinct_lengths = 2)
  (h2 : side_AB = 7)
  (h3 : side_BC = 5)
  (h4 : perimeter = 38) :
  ∃ (count : Nat), count = 2 ∧ 
  (∃ (sides : Finset ℝ), sides.card = 6 ∧ 
   (Finset.sum sides id) = perimeter ∧
   (∀ s ∈ sides, s = side_AB ∨ s = side_BC) ∧
   (Finset.filter (λ s => s = side_BC) sides).card = count) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_side_count_l145_14591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_result_l145_14525

-- Define f and g as noncomputable functions
noncomputable def f (x : ℝ) : ℝ := x + 3
noncomputable def g (x : ℝ) : ℝ := 2 * x

-- Define their inverses as noncomputable functions
noncomputable def f_inv (x : ℝ) : ℝ := x - 3
noncomputable def g_inv (x : ℝ) : ℝ := x / 2

-- State the theorem
theorem composition_result : f (g_inv (f_inv (f_inv (g (f 15))))) = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_result_l145_14525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_isosceles_right_triangle_inradius_equals_area_l145_14504

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle where
  /-- The length of the equal legs of the triangle -/
  leg : ℝ
  /-- The leg must be positive -/
  leg_pos : leg > 0

/-- The area of an isosceles right triangle -/
noncomputable def area (t : IsoscelesRightTriangle) : ℝ := t.leg^2 / 2

/-- The inradius of an isosceles right triangle -/
noncomputable def inradius (t : IsoscelesRightTriangle) : ℝ := t.leg * (2 - Real.sqrt 2) / 4

/-- Theorem stating that there is exactly one isosceles right triangle
    where the inradius equals the area -/
theorem unique_isosceles_right_triangle_inradius_equals_area :
  ∃! t : IsoscelesRightTriangle, inradius t = area t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_isosceles_right_triangle_inradius_equals_area_l145_14504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_computation_l145_14502

theorem matrix_N_computation (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : N.mulVec (![1, -2] : Fin 2 → ℝ) = ![2, 1])
  (h2 : N.mulVec (![-2, 3] : Fin 2 → ℝ) = ![0, -2]) :
  N.mulVec (![4, -1] : Fin 2 → ℝ) = ![-20, 4] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_computation_l145_14502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_value_l145_14589

open Real

noncomputable def f (x : ℝ) : ℝ := x / log x
noncomputable def g (x : ℝ) : ℝ := x / (x^2 - Real.exp 1 * x + Real.exp 2)

-- State the theorem
theorem min_t_value (t : ℝ) :
  (∀ x > 1, (t + 1) * g x ≤ t * f x) →
  t > 0 →
  t ≥ 1 / (Real.exp 2 - 1) :=
by
  sorry

-- You can add more lemmas or theorems here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_value_l145_14589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_product_equals_eight_l145_14553

theorem power_product_equals_eight (m n : ℝ) (h : m + n - 3 = 0) : 
  (2 : ℝ)^m * (2 : ℝ)^n = 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_product_equals_eight_l145_14553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_not_nonpositive_l145_14568

-- Define the function f(x) = lg(x^2 - 2x + a)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - 2*x + a) / Real.log 10

-- Theorem statement
theorem range_not_nonpositive (a : ℝ) : 
  ¬(∀ y : ℝ, y ≤ 0 → ∃ x : ℝ, f a x = y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_not_nonpositive_l145_14568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_ratio_condition_solution_l145_14542

/-- The divisors of a positive integer n satisfy the ratio condition if
    (d₂-d₁) : (d₃-d₂) : ⋯ : (dₖ-dₖ₋₁) = 1 : 2 : ⋯ : (k-1),
    where 1 = d₁ < d₂ < ... < dₖ = n are all positive divisors of n. -/
def satisfies_ratio_condition (n : ℕ) : Prop :=
  ∃ (k : ℕ) (d : Fin (k + 1) → ℕ),
    d 0 = 1 ∧ d (Fin.last k) = n ∧
    (∀ i : Fin k, d i < d i.succ) ∧
    (∀ i : Fin (k + 1), n % d i = 0) ∧
    (∀ i : ℕ, i > 0 → i ∣ n → ∃ j : Fin (k + 1), d j = i) ∧
    (∀ i : Fin k,
      (d i.succ - d i) * i.val.succ = (d i.succ.succ - d i.succ) * i.val)

/-- The only composite positive integer satisfying the ratio condition is 4. -/
theorem unique_ratio_condition_solution :
  ∀ n : ℕ, n > 1 → ¬Nat.Prime n → satisfies_ratio_condition n → n = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_ratio_condition_solution_l145_14542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_three_eq_neg_one_l145_14526

-- Define the function f as noncomputable due to Real.log
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log (3 * x + 1) else -Real.log (-3 * x + 1)

-- State the theorem
theorem f_neg_three_eq_neg_one :
  (∀ x, f (-x) = -f x) →  -- f is odd
  f (-3) = -1 :=
by
  -- The proof is skipped using 'sorry'
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_three_eq_neg_one_l145_14526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l145_14593

open Real

-- Define the original function
noncomputable def f (x φ : ℝ) : ℝ := cos (2 * x + φ)

-- Define the shifted function
noncomputable def g (x φ : ℝ) : ℝ := f (x - π / 2) φ

-- Define the overlapping function
noncomputable def h (x : ℝ) : ℝ := sin (2 * x + π / 3)

theorem phi_value (φ : ℝ) :
  (- π ≤ φ ∧ φ < π) →
  (∀ x, g x φ = h x) →
  |φ| = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l145_14593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheat_expenditure_l145_14583

-- Define the price reduction percentage
def price_reduction : ℚ := 15 / 100

-- Define the additional amount of wheat that can be purchased
def additional_wheat : ℚ := 3

-- Define the reduced price per kg
def reduced_price : ℚ := 25

-- Define the function to calculate the original price
noncomputable def original_price (reduced_price : ℚ) (price_reduction : ℚ) : ℚ :=
  reduced_price / (1 - price_reduction)

-- Define the function to calculate the amount spent
noncomputable def amount_spent (reduced_price : ℚ) (original_price : ℚ) (additional_wheat : ℚ) : ℚ :=
  (additional_wheat * reduced_price * original_price) / (original_price - reduced_price)

-- Theorem statement
theorem wheat_expenditure :
  let orig_price := original_price reduced_price price_reduction
  ∃ ε > 0, |amount_spent reduced_price orig_price additional_wheat - 450| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheat_expenditure_l145_14583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l145_14529

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 / (x + 1) - 1

theorem f_properties :
  (∀ x, f x = 0 ↔ x = 1 ∨ x = -1/2) ∧
  (∀ x y, 0 < x → 0 < y → x < y → f x < f y) ∧
  (∀ a, (∀ x, 0 < x → 0 < f (a * x^2 + 2 * a)) → 1/2 ≤ a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l145_14529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_proof_l145_14565

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (1/2) * (a + b) * h

/-- Theorem: For a trapezium with parallel sides of 20 cm and 18 cm, and an area of 228 square centimeters,
    the distance between the parallel sides is 12 cm. -/
theorem trapezium_height_proof :
  ∃ (h : ℝ), trapezium_area 20 18 h = 228 ∧ h = 12 := by
  use 12
  constructor
  · simp [trapezium_area]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_proof_l145_14565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_327_l145_14500

def sequenceC (n : ℕ) : ℕ :=
  match n with
  | 0 => 17
  | m + 1 => sequenceC m + 2^m * 10

theorem sixth_term_is_327 : sequenceC 5 = 327 := by
  rfl

#eval sequenceC 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_327_l145_14500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersections_third_degree_polynomials_l145_14539

-- Define the type of third-degree polynomials with leading coefficient 1
def ThirdDegreePolynomial : Type :=
  {p : ℝ → ℝ // ∃ a b c : ℝ, ∀ x, p x = x^3 + a*x^2 + b*x + c}

-- State the theorem
theorem max_intersections_third_degree_polynomials :
  ∀ p q : ThirdDegreePolynomial, p ≠ q →
  (∃ S : Finset ℝ, (∀ x ∈ S, p.val x = q.val x) ∧ S.card ≤ 2) ∧
  (¬∃ S : Finset ℝ, (∀ x ∈ S, p.val x = q.val x) ∧ S.card > 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersections_third_degree_polynomials_l145_14539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l145_14567

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x + 5 else -2 * x + 8

theorem f_properties :
  (f 2 = 4) ∧
  (f (f (-1)) = 0) ∧
  (∀ x, f x ≥ 4 ↔ -1 ≤ x ∧ x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l145_14567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_base_four_l145_14551

/-- Converts a base-k number represented as a list of digits to a natural number. -/
def toNatBase (digits : List Nat) (k : Nat) : Nat :=
  digits.foldl (fun acc d => acc * k + d) 0

/-- The theorem stating that 4 is the unique base k ≥ 4 for which 132 in base k equals 30 in decimal. -/
theorem unique_base_four :
  ∃! k : Nat, k ≥ 4 ∧ toNatBase [1, 3, 2] k = 30 := by
  sorry

#check unique_base_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_base_four_l145_14551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_trig_functions_l145_14540

open Real

theorem order_of_trig_functions (x : ℝ) (h : x > -1/2 ∧ x < 0) :
  cos ((x + 1) * π) < sin (cos (x * π)) ∧ sin (cos (x * π)) < cos (sin (x * π)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_trig_functions_l145_14540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_l145_14503

theorem inscribed_sphere_volume (cube_surface_area : Real) (sphere_volume : Real) : 
  cube_surface_area = 24 → 
  sphere_volume = (4 / 3) * Real.pi := by
  intro h_surface_area
  
  -- Calculate the side length of the cube
  have cube_side : Real := Real.sqrt (24 / 6)
  
  -- Calculate the radius of the inscribed sphere
  have sphere_radius : Real := cube_side / 2
  
  -- Calculate the volume of the sphere
  have sphere_vol : Real := (4 / 3) * Real.pi * (sphere_radius ^ 3)
  
  -- Show that this volume equals the given sphere_volume
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_l145_14503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bound_implies_m_bound_l145_14576

noncomputable def sequenceA (a m : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => (1/4) * (sequenceA a m n)^2 + m

theorem sequence_bound_implies_m_bound (a m : ℝ) (h1 : 0 < a) (h2 : a < 1) 
  (h3 : ∀ n : ℕ, sequenceA a m n < 3) : m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bound_implies_m_bound_l145_14576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_properties_l145_14520

-- Define the units digit function
def G : ℕ → ℕ := sorry

-- Properties of G
axiom G_range (n : ℕ) : G n < 10

-- Statements to prove
theorem units_digit_properties :
  (∃ a b : ℕ, G (a - b) ≠ G a - G b) ∧
  (∀ a b c : ℕ, a > 0 → b > 0 → c > 0 → a - b = 10 * c → G a = G b) ∧
  (∀ a b c : ℕ, G (a * b * c) = G (G a * G b * G c)) ∧
  G (3^2015) ≠ 9 := by
  sorry

#check units_digit_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_properties_l145_14520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_permutation_l145_14547

theorem grid_permutation (N k m : ℕ) 
  (h1 : k < N) 
  (h2 : m < N) 
  (h3 : k + m < N) 
  (h4 : Nat.Coprime (N - k) (N - m)) : 
  ∀ col : Fin N, Function.Surjective (λ row : Fin N => 
    let initial_pos := (row.val * (N - k) + col.val) % N
    if initial_pos < m then
      (initial_pos + k) % N
    else if initial_pos < N - k then
      initial_pos
    else
      (initial_pos - (N - k - m)) % N
  ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_permutation_l145_14547
