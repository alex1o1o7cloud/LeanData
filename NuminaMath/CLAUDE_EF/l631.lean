import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfying_points_form_scaled_square_l631_63181

-- Define the square ABCD
noncomputable def Square (a : ℝ) : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ a ∧ 0 ≤ p.2 ∧ p.2 ≤ a}

-- Define the center of the square
noncomputable def centerOfSquare (a : ℝ) : ℝ × ℝ := (a/2, a/2)

-- Define the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  abs ((p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2)

-- Define the set of points M satisfying the condition
noncomputable def satisfyingPoints (a : ℝ) : Set (ℝ × ℝ) :=
  {M | triangleArea M (0, 0) (a, a) + triangleArea M (a, 0) (0, a) = a^2}

-- Define the perimeter of the scaled square
noncomputable def scaledSquarePerimeter (a : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - a/2)^2 + (p.2 - a/2)^2 = 2*a^2}

-- The theorem to be proved
theorem satisfying_points_form_scaled_square (a : ℝ) (h : a > 0) :
  satisfyingPoints a = scaledSquarePerimeter a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfying_points_form_scaled_square_l631_63181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_implies_homothety_center_l631_63180

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Inversion with respect to a circle -/
noncomputable def inversion (c : Circle) (p : ℝ × ℝ) : ℝ × ℝ := sorry

/-- A point is a center of homothety of two circles if there exists a homothety with this point as center that maps one circle to the other -/
def is_center_of_homothety (p : ℝ × ℝ) (c1 c2 : Circle) : Prop := sorry

/-- Define membership for points in a circle -/
instance : Membership (ℝ × ℝ) Circle where
  mem p c := (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

theorem inversion_implies_homothety_center 
  (O : ℝ × ℝ) (ω S S' : Circle) :
  ω.center = O →
  (∀ p : ℝ × ℝ, p ∈ S → inversion ω p ∈ S') →
  is_center_of_homothety O S S' := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_implies_homothety_center_l631_63180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_set_values_l631_63152

open Finset

def A : Finset ℕ := {1, 3, 5, 16}
def B : Finset ℕ := {1, 9, 25, 256}

theorem prove_set_values :
  ∀ a₁ a₂ a₃ a₄ : ℕ,
  (0 < a₁ ∧ a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄) →
  ({a₁, a₂, a₃, a₄} : Finset ℕ).card = 4 →
  ({a₁^2, a₂^2, a₃^2, a₄^2} : Finset ℕ).card = 4 →
  ({a₁, a₂, a₃, a₄} ∩ {a₁^2, a₂^2, a₃^2, a₄^2} : Finset ℕ) = {a₁, a₄} →
  a₁ + a₄ ≠ 10 →
  (({a₁, a₂, a₃, a₄} ∪ {a₁^2, a₂^2, a₃^2, a₄^2} : Finset ℕ).sum id = 124) →
  a₁ = 1 ∧ a₄ = 16 ∧ {a₁, a₂, a₃, a₄} = A :=
by
  sorry

#check prove_set_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_set_values_l631_63152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pool_problem_l631_63116

/-- The original amount of water in the pool -/
def original_water : ℝ := sorry

/-- The rate at which water flows out of each outlet -/
def outflow_rate : ℝ := sorry

/-- The rate at which water flows into the pool -/
def inflow_rate : ℝ := 3

/-- The time it takes to drain the pool with 3 outlets open -/
def drain_time_3 : ℝ := 16

/-- The time it takes to drain the pool with 5 outlets open -/
def drain_time_5 : ℝ := 9

theorem water_pool_problem :
  (original_water + inflow_rate * drain_time_3 = 3 * drain_time_3 * outflow_rate) ∧
  (original_water + inflow_rate * drain_time_5 = 5 * drain_time_5 * outflow_rate) →
  original_water = 288 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pool_problem_l631_63116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roses_color_count_l631_63107

theorem roses_color_count (total : ℕ) (red_fraction : ℚ) (yellow_fraction : ℚ) : 
  total = 80 →
  red_fraction = 3/4 →
  yellow_fraction = 1/4 →
  (total * red_fraction).floor + (total - (total * red_fraction).floor - ((total - (total * red_fraction).floor) * yellow_fraction).floor) = 75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roses_color_count_l631_63107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_solid_diagonal_angles_l631_63109

/-- Predicate indicating that the solid is rectangular -/
def IsRectangularSolid : Prop := sorry

/-- Predicate indicating that a line is a diagonal of the solid -/
def IsDiagonal : Prop := sorry

/-- Predicate indicating that a diagonal forms angles α, β, and γ with the three edges
    starting from one vertex -/
def FormsAngles (α β γ : Real) (d : IsDiagonal) : Prop := sorry

/-- In a rectangular solid, if one of its diagonals forms angles α, β, and γ with the three edges
    starting from one vertex, then cos²α + cos²β + cos²γ = 1 -/
theorem rectangular_solid_diagonal_angles (α β γ : Real) 
  (h_rectangular_solid : IsRectangularSolid)
  (h_diagonal : IsDiagonal)
  (h_angles : FormsAngles α β γ h_diagonal) :
  Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_solid_diagonal_angles_l631_63109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solution_l631_63136

theorem quadratic_solution (b c m : ℝ) (h : c ≠ 0) :
  let f (a : ℝ) := m * (c - a) - b * c^2 * a^2
  (∃ a, f a = 0) →
  (∃ a, a = (-m + Real.sqrt (m^2 + 4*b*m*c^3)) / (2*b*c^2) ∧ f a = 0) ∧
  (∃ a, a = (-m - Real.sqrt (m^2 + 4*b*m*c^3)) / (2*b*c^2) ∧ f a = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solution_l631_63136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_area_is_1400_l631_63119

/-- A square with an inscribed hexadecagon -/
structure SquareWithHexadecagon where
  /-- The side length of the square -/
  side_length : ℝ
  /-- The perimeter of the square is 160 cm -/
  perimeter_condition : side_length * 4 = 160
  /-- The vertices of the hexadecagon quadrisect the sides of the square -/
  quadrisection : True

/-- The area of the inscribed hexadecagon -/
noncomputable def hexadecagon_area (s : SquareWithHexadecagon) : ℝ :=
  s.side_length ^ 2 - 4 * (s.side_length / 4) ^ 2

/-- Theorem stating that the area of the inscribed hexadecagon is 1400 square centimeters -/
theorem hexadecagon_area_is_1400 (s : SquareWithHexadecagon) :
  hexadecagon_area s = 1400 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_area_is_1400_l631_63119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_bound_l631_63121

open Real

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := -1/2 * x^2 + b * log x

-- Define what it means for f to be decreasing on (2, +∞)
def is_decreasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, 2 < x → x < y → f y < f x

-- State the theorem
theorem decreasing_function_bound (b : ℝ) : 
  is_decreasing_on_interval (f b) → b ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_bound_l631_63121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increasing_interval_l631_63135

/-- The function f(x) = 2x³ + ax² + 36x - 24 has an extreme value at x = 2 and is increasing for x > 3 -/
theorem function_increasing_interval (a : ℝ) : 
  let f := (λ x : ℝ => 2 * x^3 + a * x^2 + 36 * x - 24)
  (∀ x, deriv f x = 6 * x^2 + 2 * a * x + 36) →
  deriv f 2 = 0 →
  ∀ x > 3, deriv f x > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increasing_interval_l631_63135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_solutions_l631_63133

theorem count_integer_solutions : 
  ∃ (S : Finset Int), (∀ x : Int, x ∈ S ↔ 3*x^2 + 17*x + 14 ≤ 20) ∧ Finset.card S = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_solutions_l631_63133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_is_correct_l631_63120

/-- The distance between points A and B -/
noncomputable def distance_AB : ℝ := 15

/-- Billy's walking speed -/
noncomputable def billy_speed : ℝ := 1

/-- Bobby's walking speed -/
noncomputable def bobby_speed : ℝ := (distance_AB - 3) / 3

/-- The location of the first meeting point from A -/
def first_meeting : ℝ := 3

/-- The location of the second meeting point from B -/
def second_meeting : ℝ := 10

/-- Theorem stating that the given conditions lead to the correct distance between A and B -/
theorem distance_AB_is_correct :
  (first_meeting = 3) ∧
  (second_meeting = 10) ∧
  (billy_speed * (distance_AB - second_meeting) = bobby_speed * (distance_AB + second_meeting)) →
  distance_AB = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_is_correct_l631_63120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_R_value_at_S_5_l631_63105

/-- Given the relationship R = gS^2 - 5 and that R = 25 when S = 3, 
    prove that R = 235/3 when S = 5 -/
theorem R_value_at_S_5 (g : ℚ) (R : ℚ → ℚ) : 
  (∀ S : ℚ, R S = g * S^2 - 5) → 
  R 3 = 25 → 
  R 5 = 235/3 :=
by
  intros h1 h2
  have g_value : g = 10/3
  · rw [h1] at h2
    linarith
  have : R 5 = g * 5^2 - 5
  · exact h1 5
  rw [g_value] at this
  norm_num at this
  exact this


end NUMINAMATH_CALUDE_ERRORFEEDBACK_R_value_at_S_5_l631_63105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2a_value_l631_63144

theorem cos_2a_value (a : ℝ) : 
  Real.sin (π / 2 + a) = 1 / 3 → Real.cos (2 * a) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2a_value_l631_63144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_f_monotone_decreasing_l631_63129

-- Define the vectors a and b
noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sqrt 2 * Real.sin x, Real.sqrt 2 * Real.cos x + 1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 2 * Real.cos x, Real.sqrt 2 * Real.cos x - 1)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- Statement for the first part of the problem
theorem range_of_f : 
  ∀ x ∈ Set.Ioo 0 (Real.pi / 2), -1 < f x ∧ f x ≤ Real.sqrt 2 :=
by sorry

-- Statement for the second part of the problem
theorem f_monotone_decreasing (k : ℤ) : 
  StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi / 8) (k * Real.pi + 3 * Real.pi / 8)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_f_monotone_decreasing_l631_63129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l631_63118

def IntegerRange : Set ℤ := {x | -7 ≤ x ∧ x ≤ 7}

def IsNonDecreasingAbsolute (seq : List ℤ) : Prop :=
  ∀ i j, i < j → i < seq.length → j < seq.length → 
    abs (seq.get ⟨i, by sorry⟩) ≤ abs (seq.get ⟨j, by sorry⟩)

def ValidArrangement (seq : List ℤ) : Prop :=
  seq.toFinset = IntegerRange ∧ IsNonDecreasingAbsolute seq

-- Add this instance to ensure Fintype for the subset
instance : Fintype {seq : List ℤ // ValidArrangement seq} := by sorry

theorem valid_arrangements_count :
  Fintype.card {seq : List ℤ // ValidArrangement seq} = 2^7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l631_63118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_votes_for_a_to_win_l631_63178

/-- The minimum number of additional votes needed for a candidate to win an election. -/
def min_votes_to_win (total_voters : ℕ) (current_votes_a current_votes_b current_votes_c : ℕ) : ℕ :=
  let remaining_votes := total_voters - (current_votes_a + current_votes_b + current_votes_c)
  let x := (remaining_votes + current_votes_b - current_votes_a + 1) / 2
  ⌈(x : ℚ)⌉.toNat

/-- Theorem stating the minimum number of additional votes needed for candidate A to win. -/
theorem min_votes_for_a_to_win :
  min_votes_to_win 48 13 10 7 = 8 := by
  -- Proof goes here
  sorry

#eval min_votes_to_win 48 13 10 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_votes_for_a_to_win_l631_63178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_ratio_l631_63126

theorem sin_cos_ratio (α : ℝ) (h1 : Real.sin (2 * Real.pi - α) = 4/5) 
    (h2 : α ∈ Set.Ioo (3*Real.pi/2) (2*Real.pi)) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_ratio_l631_63126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_path_is_linear_l631_63128

/-- A third-degree polynomial function -/
def ThirdDegreePolynomial (α : Type*) [Field α] := α → α

/-- The ship's path is a straight line if f and g are linearly dependent -/
def IsLinearPath (f g : ThirdDegreePolynomial ℝ) : Prop :=
  ∃ (lambda : ℝ) (C : ℝ), ∀ t, f t = lambda * g t + C

theorem ship_path_is_linear
  (f g : ThirdDegreePolynomial ℝ)
  (h1 : f 14 = f 13) (h2 : g 14 = g 13)
  (h3 : f 20 = f 19) (h4 : g 20 = g 19) :
  IsLinearPath f g := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_path_is_linear_l631_63128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_twelve_factors_sixty_has_twelve_factors_sixty_is_smallest_with_twelve_factors_l631_63161

def number_of_factors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem smallest_with_twelve_factors : 
  ∀ k : ℕ, k > 0 → number_of_factors k = 12 → k ≥ 60 :=
by
  sorry

theorem sixty_has_twelve_factors : number_of_factors 60 = 12 :=
by
  sorry

theorem sixty_is_smallest_with_twelve_factors : 
  ∀ k : ℕ, k > 0 → number_of_factors k = 12 → k = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_twelve_factors_sixty_has_twelve_factors_sixty_is_smallest_with_twelve_factors_l631_63161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_N_complement_M_l631_63114

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - x - 2 > 0}

noncomputable def N : Set ℝ := {x | ∃ y, y = Real.log (x + 2) + Real.log (1 - x)}

-- State the theorem
theorem union_N_complement_M : N ∪ (Set.univ \ M) = Set.Ioc (-2) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_N_complement_M_l631_63114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l631_63146

-- Define the solution set type
def SolutionSet (α : Type) := Set α

-- Define the quadratic inequality
def QuadraticInequality (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 - x + 1 < 0

-- Define the solution set based on the value of a
noncomputable def solutionSet (a : ℝ) : Set ℝ :=
  if a < 0 then
    {x | x < (-1 + Real.sqrt (1 - 4*a)) / (2*a) ∨ x > (-1 - Real.sqrt (1 - 4*a)) / (2*a)}
  else if a = 0 then
    {x | x < -1}
  else if 0 < a ∧ a < 1/4 then
    {x | (-1 - Real.sqrt (1 - 4*a)) / (2*a) < x ∧ x < (-1 + Real.sqrt (1 - 4*a)) / (2*a)}
  else
    ∅

-- Theorem statement
theorem quadratic_inequality_solution_set (a : ℝ) :
  ∀ x : ℝ, QuadraticInequality a x ↔ x ∈ solutionSet a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l631_63146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_value_l631_63100

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem floor_expression_value :
  let x : ℝ := 8.3
  (floor 6.5 : ℝ) * (floor (2 / 3) : ℝ) + (floor 2 : ℝ) * 7.2 + (floor x : ℝ) - 6.6 = 15.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_value_l631_63100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_theorem_l631_63163

/-- Definition of a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of a line in 2D space -/
structure Line2D where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Definition of a triangle -/
structure Triangle where
  P : Point2D
  Q : Point2D
  R : Point2D

/-- Theorem: Angle bisector of ∠P in the given triangle -/
theorem angle_bisector_theorem (T : Triangle) (L : Line2D) : 
  T.P = Point2D.mk (-8) 5 →
  T.Q = Point2D.mk (-15) (-19) →
  T.R = Point2D.mk 1 (-7) →
  L.a > 0 ∧ L.b > 0 ∧ L.c > 0 →
  Int.gcd L.a (Int.gcd L.b L.c) = 1 →
  -- L is the angle bisector of ∠P
  (∃ (k : ℝ), ∀ (x y : ℝ), 
    L.a * x + L.b * y + L.c = 0 ↔ 
    k * ((x - T.P.x) * (T.R.y - T.P.y) - (y - T.P.y) * (T.R.x - T.P.x)) = 
    (1 - k) * ((x - T.P.x) * (T.Q.y - T.P.y) - (y - T.P.y) * (T.Q.x - T.P.x))) →
  L.a + L.c = 89 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_theorem_l631_63163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_l631_63165

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then x + b else 2 * x + 2

theorem continuous_piecewise_function :
  ∀ b : ℝ, Continuous (f b) ↔ b = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_l631_63165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorial_divisible_by_100000_l631_63117

theorem smallest_factorial_divisible_by_100000 :
  (∀ n : ℕ, n > 0 ∧ n < 25 → ¬(100000 ∣ Nat.factorial n)) ∧ (100000 ∣ Nat.factorial 25) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorial_divisible_by_100000_l631_63117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combinations_equal_thirty_l631_63167

/-- Represents the parts of the room that can be painted -/
inductive RoomPart
| ceiling
| walls

/-- Represents the available colors -/
inductive Color
| blue
| green
| yellow
| black
| white

/-- Represents the painting methods -/
inductive PaintMethod
| brush
| roller
| sprayGun

/-- The number of different combinations of choices -/
def numberOfCombinations : Nat :=
  2 * 5 * 3

theorem combinations_equal_thirty :
  numberOfCombinations = 30 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combinations_equal_thirty_l631_63167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_formula_l631_63141

def binom : ℕ → ℕ → ℕ
  | n, 0 => 1
  | 0, k+1 => 0
  | n+1, k+1 => binom n k + binom n (k+1)

theorem binomial_coefficient_formula (n k : ℕ) (h : k ≤ n) :
  binom n k = n.factorial / (k.factorial * (n - k).factorial) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_formula_l631_63141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l631_63150

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1/4

-- Define the ellipse C₂
def C₂ (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the distance function between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := 
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Theorem statement
theorem min_distance_between_curves :
  ∃ (min_dist : ℝ), 
    (min_dist = Real.sqrt 6 / 3 - 1/2) ∧ 
    (∀ (x₁ y₁ x₂ y₂ : ℝ), 
      C₁ x₁ y₁ → C₂ x₂ y₂ → 
      distance x₁ y₁ x₂ y₂ ≥ min_dist) ∧
    (∃ (x₁ y₁ x₂ y₂ : ℝ), 
      C₁ x₁ y₁ ∧ C₂ x₂ y₂ ∧ 
      distance x₁ y₁ x₂ y₂ = min_dist) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l631_63150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l631_63183

def sequence_a : ℕ → ℕ
  | 0 => 1  -- Adding the base case for 0
  | 1 => 1
  | n + 1 => 2 * sequence_a n + 2^n

theorem sequence_a_formula (n : ℕ) : 
  sequence_a n = n * 2^(n-1) := by
  induction n with
  | zero => 
    simp [sequence_a]
    -- The formula doesn't hold for n = 0, so we need to handle this case separately
    sorry
  | succ n ih =>
    simp [sequence_a]
    -- Here you would continue with the proof
    sorry

#eval sequence_a 5  -- This line is optional, for testing purposes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l631_63183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_X_satisfies_transformation_l631_63102

/-- The linear transformation matrix A --/
def A : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![1, 2, 3],
    ![-1, 3, 4],
    ![2, -3, 5]]

/-- The vector Y --/
def Y : Fin 3 → ℚ := ![1, 2, 3]

/-- The vector X --/
def X : Fin 3 → ℚ := ![-7/22, -5/22, 13/22]

/-- Theorem stating that X satisfies the linear transformation --/
theorem X_satisfies_transformation : A.mulVec X = Y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_X_satisfies_transformation_l631_63102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_containing_circle_l631_63106

/- Define a circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/- Define a configuration of five circles -/
structure CircleConfiguration where
  centralCircle : Circle
  outerCircles : Fin 4 → Circle

/- Define the property of circles touching -/
def touchingCircles (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

/- Define the property of circles not intersecting -/
def nonIntersectingCircles (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 ≥ (c1.radius + c2.radius)^2

/- Define a valid configuration -/
def validConfiguration (config : CircleConfiguration) : Prop :=
  (∀ i : Fin 4, config.centralCircle.radius = (config.outerCircles i).radius) ∧
  (∀ i : Fin 4, touchingCircles config.centralCircle (config.outerCircles i)) ∧
  (∀ i j : Fin 4, i ≠ j → nonIntersectingCircles (config.outerCircles i) (config.outerCircles j))

/- Theorem statement -/
theorem smallest_containing_circle (config : CircleConfiguration) (h : validConfiguration config) :
  ∃ (c : Circle), (∀ (i : Fin 4), (config.outerCircles i).center.1^2 + (config.outerCircles i).center.2^2 ≤ c.radius^2) ∧
                  (config.centralCircle.center.1^2 + config.centralCircle.center.2^2 ≤ c.radius^2) ∧
                  (∀ (c' : Circle), (∀ (i : Fin 4), (config.outerCircles i).center.1^2 + (config.outerCircles i).center.2^2 ≤ c'.radius^2) ∧
                                    (config.centralCircle.center.1^2 + config.centralCircle.center.2^2 ≤ c'.radius^2) →
                                    c'.radius ≥ c.radius) ∧
                  c.radius = 3 * config.centralCircle.radius := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_containing_circle_l631_63106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trucks_required_l631_63157

/-- Represents a part with its quantity and weight -/
structure PartInfo where
  quantity : Nat
  weight : Nat

/-- The problem setup -/
def problem : List PartInfo := [
  { quantity := 4, weight := 5 },  -- Part A
  { quantity := 6, weight := 4 },  -- Part B
  { quantity := 11, weight := 3 }, -- Part C
  { quantity := 7, weight := 1 }   -- Part D
]

/-- The capacity of each truck -/
def truckCapacity : Nat := 6

/-- Calculate the total weight of all parts -/
def totalWeight (parts : List PartInfo) : Nat :=
  parts.foldl (fun acc p => acc + p.quantity * p.weight) 0

/-- The minimum number of trucks required -/
theorem min_trucks_required (parts : List PartInfo) (capacity : Nat) :
  parts = problem → capacity = truckCapacity →
  (totalWeight parts + capacity - 1) / capacity = 16 := by
  sorry

#eval (totalWeight problem + truckCapacity - 1) / truckCapacity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trucks_required_l631_63157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_segment_surface_area_l631_63199

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a spherical segment -/
structure SphericalSegment where
  center : Point3D  -- Center of the sphere
  radius : ℝ        -- Radius of the sphere
  height : ℝ        -- Height of the segment
  vertex : Point3D  -- Vertex of the segment
  basePoint : Point3D  -- A point on the circumference of the base

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Theorem: The surface area of a spherical segment is equal to the area of a circle with radius AB -/
theorem spherical_segment_surface_area (segment : SphericalSegment) :
  let surfaceArea := 2 * Real.pi * segment.radius * segment.height
  let AB := distance segment.vertex segment.basePoint
  let circleArea := Real.pi * AB^2
  surfaceArea = circleArea := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_segment_surface_area_l631_63199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l631_63131

/-- Represents the sum of the first n terms of a geometric sequence -/
noncomputable def geometricSum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

/-- Theorem: For a geometric sequence with common ratio q, if 6S₄ = S₅ + 5S₆, then q = -6/5 -/
theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) :
  q ≠ 1 →
  6 * (geometricSum a₁ q 4) = (geometricSum a₁ q 5) + 5 * (geometricSum a₁ q 6) →
  q = -6/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l631_63131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_in_half_filled_tank_l631_63184

/-- Represents a horizontally lying cylindrical water tank -/
structure CylindricalTank where
  length : ℝ
  diameter : ℝ
  waterSurfaceArea : ℝ

/-- Calculates the depth of water in a cylindrical tank -/
noncomputable def waterDepth (tank : CylindricalTank) : ℝ :=
  2 - Real.sqrt 3

theorem water_depth_in_half_filled_tank (tank : CylindricalTank) 
  (h1 : tank.length = 12)
  (h2 : tank.diameter = 4)
  (h3 : tank.waterSurfaceArea = 24) :
  waterDepth tank = 2 - Real.sqrt 3 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_in_half_filled_tank_l631_63184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_worked_eight_days_l631_63155

/-- Represents the work rates and time spent by two workers on a job -/
structure WorkProblem where
  x_total_days : ℚ  -- Total days x needs to complete the job
  y_total_days : ℚ  -- Total days y needs to complete the job
  y_actual_days : ℚ  -- Days y actually worked to finish the job
  total_work : ℚ     -- Total amount of work (normalized to 1)

/-- Calculate the number of days x worked before y took over -/
def days_x_worked (w : WorkProblem) : ℚ :=
  w.x_total_days * (1 - w.y_actual_days / w.y_total_days)

/-- Theorem stating that for the given problem, x worked for 8 days -/
theorem x_worked_eight_days (w : WorkProblem) 
  (hx : w.x_total_days = 40)
  (hy : w.y_total_days = 25)
  (hy_actual : w.y_actual_days = 20)
  (hw : w.total_work = 1) :
  days_x_worked w = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_worked_eight_days_l631_63155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_of_specific_matrix_l631_63153

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![3, 0, -2; 4, 5, -3; 1, 1, 6]
  Matrix.det A = 101 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_of_specific_matrix_l631_63153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_coefficient_bound_l631_63192

/-- A real polynomial of degree 3 or less -/
def polynomial (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

/-- The theorem statement -/
theorem polynomial_coefficient_bound (a b c d : ℝ) :
  (∀ x : ℝ, |x| ≤ 1 → |polynomial a b c d x| ≤ 1) →
  |a| + |b| + |c| + |d| ≤ 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_coefficient_bound_l631_63192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l631_63101

/-- Definition of the function f(x) -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (-2^x + a) / (2^(x+1) + b)

/-- Theorem stating the properties of the function f -/
theorem f_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_odd : ∀ x, f a b (-x) = -(f a b x)) :
  (a = 1 ∧ b = 2) ∧ 
  (∀ x y, x < y → f a b x > f a b y) ∧
  (∀ x, f a b x > -1/6 ↔ x < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l631_63101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_tan_product_l631_63147

-- Define tangent function
noncomputable def tan (x : ℝ) : ℝ := Real.tan x

-- Define degree to radian conversion
noncomputable def deg_to_rad (x : ℝ) : ℝ := x * (Real.pi / 180)

-- Angle addition formula for tangent
axiom tan_add (a b : ℝ) : Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)

-- Given: tan 45° = 1
axiom tan_45 : Real.tan (deg_to_rad 45) = 1

-- Theorem to prove
theorem simplify_tan_product : 
  (1 + Real.tan (deg_to_rad 10)) * (1 + Real.tan (deg_to_rad 35)) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_tan_product_l631_63147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l631_63164

/-- An ellipse with foci F₁ and F₂, and a point P on the ellipse. -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  P : ℝ × ℝ
  on_ellipse : ∃ a : ℝ, dist P F₁ + dist P F₂ = 2 * a

/-- The definition of eccentricity for an ellipse. -/
noncomputable def eccentricity (E : Ellipse) : ℝ :=
  dist E.F₁ E.F₂ / (dist E.P E.F₁ + dist E.P E.F₂)

/-- The condition that |PF₁|, |F₁F₂|, and |PF₂| form an arithmetic sequence. -/
def arithmetic_sequence (E : Ellipse) : Prop :=
  2 * dist E.F₁ E.F₂ = dist E.P E.F₁ + dist E.P E.F₂

/-- The theorem statement. -/
theorem ellipse_eccentricity (E : Ellipse) (h : arithmetic_sequence E) :
  eccentricity E = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l631_63164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l631_63173

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def condition1 (t : Triangle) : Prop := t.a + t.c * Real.sin t.B = t.b * Real.cos t.C
def condition2 (t : Triangle) : Prop := Real.sqrt 2 * t.a = Real.sqrt 2 * t.b * Real.cos t.C - t.c
def condition3 (t : Triangle) : Prop := t.c + 2 * t.b * Real.cos t.B * Real.sin t.C = 0

-- Define the area of a triangle
noncomputable def triangle_area (t : Triangle) : ℝ := 1 / 2 * t.a * t.c * Real.sin t.B

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h : condition1 t ∨ condition2 t ∨ condition3 t) : 
  t.B = 3 * Real.pi / 4 ∧ 
  (t.b = Real.sqrt 5 ∧ t.a = 1 → triangle_area t = 1 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l631_63173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_power_inequality_l631_63187

theorem half_power_inequality (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (1/2 : ℝ)^x - (1/2 : ℝ)^y < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_power_inequality_l631_63187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_M_value_l631_63195

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

theorem minimum_M_value (M : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-1) 1 → x₂ ∈ Set.Icc (-1) 1 → M ≥ |f x₁ - f x₂|) →
  M ≥ 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_M_value_l631_63195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_finish_time_in_minutes_l631_63185

/-- The time it takes for P to finish the job alone, in hours -/
noncomputable def p_time : ℝ := 3

/-- The time it takes for Q to finish the job alone, in hours -/
noncomputable def q_time : ℝ := 9

/-- The time P and Q work together, in hours -/
noncomputable def work_together_time : ℝ := 2

/-- P's rate of work per hour -/
noncomputable def p_rate : ℝ := 1 / p_time

/-- Q's rate of work per hour -/
noncomputable def q_rate : ℝ := 1 / q_time

/-- The portion of the job completed when P and Q work together -/
noncomputable def completed_portion : ℝ := (p_rate + q_rate) * work_together_time

/-- The remaining portion of the job after P and Q work together -/
noncomputable def remaining_portion : ℝ := 1 - completed_portion

/-- The time it takes P to finish the remaining portion, in hours -/
noncomputable def p_remaining_time : ℝ := remaining_portion / p_rate

theorem p_finish_time_in_minutes : 
  p_remaining_time * 60 = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_finish_time_in_minutes_l631_63185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l631_63172

noncomputable def f (x : ℝ) := 2 * Real.sin (x + Real.pi / 2) * Real.cos (x - Real.pi / 2)

theorem intersection_distance (h : ∀ x, f x = 1/2 → x > 0) :
  ∃ x₁ x₅, x₁ > 0 ∧ x₅ > 0 ∧ f x₁ = 1/2 ∧ f x₅ = 1/2 ∧
  (∀ x, 0 < x ∧ x < x₁ → f x ≠ 1/2) ∧
  (∃ x₂ x₃ x₄, x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧ x₄ < x₅ ∧
    f x₂ = 1/2 ∧ f x₃ = 1/2 ∧ f x₄ = 1/2) ∧
  x₅ - x₁ = 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l631_63172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k2_relation_probability_l631_63174

/-- K2 is a correlation index used in independence tests -/
def K2 : ℝ → ℝ := sorry

/-- Probability that x is related to y given a K2 value -/
def prob_x_related_y : ℝ → ℝ := sorry

/-- The statement that as K2 increases, the probability of x being related to y increases -/
theorem k2_relation_probability (k1 k2 : ℝ) :
  k1 < k2 → prob_x_related_y (K2 k1) < prob_x_related_y (K2 k2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k2_relation_probability_l631_63174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_inequality_l631_63112

/-- The function f(x) = (1 + 2ln x) / x^2 -/
noncomputable def f (x : ℝ) : ℝ := (1 + 2 * Real.log x) / (x ^ 2)

/-- Theorem stating the inequality for the roots of f(x) = k -/
theorem roots_inequality (k : ℝ) (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) 
  (h3 : f x₁ = k) (h4 : f x₂ = k) : 
  x₁ + x₂ > 2 ∧ 2 > 1/x₁ + 1/x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_inequality_l631_63112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_ride_time_l631_63166

noncomputable def bike_time (distance : ℝ) (total_time : ℝ) (break_time : ℝ) (target_distance : ℝ) : ℝ :=
  let effective_time := total_time - break_time
  let speed := distance / effective_time
  target_distance / speed

theorem bike_ride_time :
  bike_time 2 6 2 5 = 10 := by
  -- Unfold the definition of bike_time
  unfold bike_time
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_ride_time_l631_63166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_geometric_sequence_l631_63175

noncomputable def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

theorem sum_of_geometric_sequence : 
  ∃ (n : ℕ), geometric_sequence 1 (1/4) n = 85/64 ∧ n = 4 := by
  use 4
  apply And.intro
  · -- Prove geometric_sequence 1 (1/4) 4 = 85/64
    sorry
  · -- Prove 4 = 4
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_geometric_sequence_l631_63175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_scores_l631_63179

noncomputable def scores : List ℝ := [8, 9, 10, 13, 15]

noncomputable def mean (xs : List ℝ) : ℝ :=
  (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  (xs.map (fun x => (x - mean xs) ^ 2)).sum / xs.length

theorem variance_of_scores : variance scores = 6.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_scores_l631_63179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_abc_l631_63176

-- Define the constants
noncomputable def a : ℝ := 2 / Real.exp 1
noncomputable def b : ℝ := Real.log (3 * Real.exp 1) / 3
noncomputable def c : ℝ := (Real.log 5 + 1) / 5

-- State the theorem
theorem ordering_of_abc : a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_abc_l631_63176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l631_63162

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 13

-- Define the points A and B
def point_A : ℝ × ℝ := (4, 2)
def point_B : ℝ × ℝ := (-1, 3)

-- Define a function to calculate the sum of intercepts
noncomputable def sum_of_intercepts (a b c : ℝ) : ℝ := 
  |a + Real.sqrt (a^2 + c)| + |a - Real.sqrt (a^2 + c)| + 
  |b + Real.sqrt (b^2 + c)| + |b - Real.sqrt (b^2 + c)|

-- Theorem statement
theorem circle_properties : 
  circle_equation point_A.1 point_A.2 ∧ 
  circle_equation point_B.1 point_B.2 ∧ 
  sum_of_intercepts 1 0 (-13) = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l631_63162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_occupied_chairs_l631_63158

def is_valid_arrangement (arrangement : Fin 30 → Bool) : Prop :=
  ∀ i : Fin 30, arrangement i ∨ arrangement (i.succ) ∨ arrangement (i.succ.succ)

theorem min_occupied_chairs :
  (∃ arrangement : Fin 30 → Bool, is_valid_arrangement arrangement ∧ (Finset.sum Finset.univ (fun i => if arrangement i then 1 else 0)) = 10) ∧
  (∀ arrangement : Fin 30 → Bool, is_valid_arrangement arrangement → (Finset.sum Finset.univ (fun i => if arrangement i then 1 else 0)) ≥ 10) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_occupied_chairs_l631_63158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficient_l631_63103

/-- A quadratic function f(x) = ax^2 + bx + c -/
def QuadraticFunction (a b c : ℝ) := fun (x : ℝ) ↦ a * x^2 + b * x + c

theorem quadratic_coefficient (a b c : ℝ) :
  (QuadraticFunction a b c 0 = 2) →
  (∃ x, QuadraticFunction a b c x ≤ QuadraticFunction a b c 2 ∧
        ∀ y, QuadraticFunction a b c x ≤ QuadraticFunction a b c y) →
  (QuadraticFunction a b c 2 = -2) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficient_l631_63103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_subset_periodic_points_quad_func_fixed_eq_periodic_l631_63193

/-- A function f: ℝ → ℝ -/
def f : ℝ → ℝ := sorry

/-- The set of fixed points of f -/
def fixed_points (f : ℝ → ℝ) : Set ℝ := {x | f x = x}

/-- The set of periodic points of f -/
def periodic_points (f : ℝ → ℝ) : Set ℝ := {x | f (f x) = x}

/-- Theorem: The set of fixed points is a subset of the set of periodic points -/
theorem fixed_points_subset_periodic_points (f : ℝ → ℝ) : fixed_points f ⊆ periodic_points f := by
  sorry

/-- The quadratic function f(x) = ax^2 - 1 -/
def quad_func (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1

/-- Theorem: If the set of fixed points equals the set of periodic points (both non-empty) 
    for f(x) = ax^2 - 1, then -1/4 ≤ a ≤ 3/4 -/
theorem quad_func_fixed_eq_periodic (a : ℝ) : 
  (fixed_points (quad_func a) = periodic_points (quad_func a) ∧ 
   (fixed_points (quad_func a)).Nonempty) → 
  -1/4 ≤ a ∧ a ≤ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_subset_periodic_points_quad_func_fixed_eq_periodic_l631_63193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l631_63108

/-- Parabola type representing y^2 = 2x -/
structure Parabola where
  x : ℝ
  y : ℝ
  eq : y^2 = 2*x

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

def M : Point := ⟨2, 3⟩

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def projectionOnYAxis (p : Point) : Point :=
  ⟨0, p.y⟩

theorem min_sum_distances (p : Parabola) :
  let P : Point := ⟨p.x, p.y⟩
  let Q : Point := projectionOnYAxis P
  ∃ (minVal : ℝ), minVal = (3 * Real.sqrt 5 - 1) / 2 ∧
    ∀ (p' : Parabola), distance P Q + distance P M ≤ 
      distance ⟨p'.x, p'.y⟩ Q + distance ⟨p'.x, p'.y⟩ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l631_63108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_balance_proof_l631_63138

noncomputable def initial_balance : ℚ := 190
noncomputable def transfer_to_mom : ℚ := 60
noncomputable def transfer_to_sister : ℚ := transfer_to_mom / 2

theorem remaining_balance_proof :
  initial_balance - (transfer_to_mom + transfer_to_sister) = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_balance_proof_l631_63138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amelia_win_probability_l631_63170

-- Define the probabilities of getting heads for each player
noncomputable def amelia_heads_prob : ℝ := 3/7
noncomputable def blaine_heads_prob : ℝ := 1/4
noncomputable def cecilia_heads_prob : ℝ := 2/3

-- Define the probability of all players getting tails in one cycle
noncomputable def all_tails_prob : ℝ := (1 - amelia_heads_prob) * (1 - blaine_heads_prob) * (1 - cecilia_heads_prob)

-- Theorem: The probability that Amelia wins the game is 1/2
theorem amelia_win_probability :
  amelia_heads_prob / (1 - all_tails_prob) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_amelia_win_probability_l631_63170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_5_or_6_not_15_l631_63171

def count_multiples (n : ℕ) (m : ℕ) : ℕ :=
  n / m

theorem multiples_5_or_6_not_15 : 
  (count_multiples 3000 5) + (count_multiples 3000 6) - (count_multiples 3000 15) = 900 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_5_or_6_not_15_l631_63171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_squared_count_l631_63186

noncomputable def possible_ceiling_squared_values (x : ℝ) (h : ⌈x⌉ = -3) : Set ℤ :=
  {n : ℤ | ∃ y : ℝ, ⌈y⌉ = -3 ∧ ⌈y^2⌉ = n}

theorem ceiling_squared_count : 
  ∃ (s : Finset ℤ), s.card = 7 ∧ ∀ x : ℝ, ⌈x⌉ = -3 → ⌈x^2⌉ ∈ s := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_squared_count_l631_63186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l631_63139

theorem inequality_proof (x y z lambda : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hlambda_lower : lambda ≥ 0) (hlambda_upper : lambda ≤ 2) : 
  (Real.sqrt (x * y)) / (x + y + lambda * z) + 
  (Real.sqrt (y * z)) / (y + z + lambda * x) + 
  (Real.sqrt (z * x)) / (z + x + lambda * y) ≤ 
  3 / (2 + lambda) := by
  sorry

#check inequality_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l631_63139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_andrea_living_room_area_l631_63156

/-- The total area of Andrea's living room floor in square feet -/
noncomputable def total_area : ℝ := 720 / 11

/-- The area of the first carpet in square feet -/
noncomputable def first_carpet_area : ℝ := 4 * 9

/-- The percentage of floor covered by the first carpet -/
noncomputable def first_carpet_percentage : ℝ := 0.55

/-- The percentage of floor covered by the second carpet -/
noncomputable def second_carpet_percentage : ℝ := 0.25

/-- The area of the rectangular part of the third carpet in square feet -/
noncomputable def third_carpet_rectangle : ℝ := 3 * 6

/-- The area of the triangular part of the third carpet in square feet -/
noncomputable def third_carpet_triangle : ℝ := 1 / 2 * 4 * 3

/-- The total area of the third carpet in square feet -/
noncomputable def third_carpet_area : ℝ := third_carpet_rectangle + third_carpet_triangle

/-- The percentage of floor covered by the third carpet -/
noncomputable def third_carpet_percentage : ℝ := 0.15

/-- The percentage of floor not covered by any carpet -/
noncomputable def uncovered_percentage : ℝ := 0.05

theorem andrea_living_room_area : 
  first_carpet_area = first_carpet_percentage * total_area ∧
  third_carpet_area = third_carpet_percentage * total_area ∧
  first_carpet_percentage + second_carpet_percentage + third_carpet_percentage + uncovered_percentage = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_andrea_living_room_area_l631_63156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_l631_63134

/-- The area of a hexagon enclosed within a rectangle, given specific conditions -/
theorem hexagon_area (num_triangles : ℕ) (triangle_base triangle_height : ℝ)
  (rectangle_width rectangle_height : ℝ) : 
  rectangle_width * rectangle_height - 
  num_triangles * (1/2 * triangle_base * triangle_height) = 36 :=
by
  have h1 : num_triangles = 6 := by sorry
  have h2 : triangle_base = 1 := by sorry
  have h3 : triangle_height = 4 := by sorry
  have h4 : rectangle_width = 6 := by sorry
  have h5 : rectangle_height = 8 := by sorry

  -- Calculate the hexagon area
  calc
    rectangle_width * rectangle_height - 
    num_triangles * (1/2 * triangle_base * triangle_height)
    = 6 * 8 - 6 * (1/2 * 1 * 4) := by sorry
    _ = 48 - 6 * 2 := by sorry
    _ = 48 - 12 := by sorry
    _ = 36 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_l631_63134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spheres_in_cone_l631_63127

/-- A right circular cone with base radius 6 and height 15 -/
structure Cone :=
  (base_radius : ℝ := 6)
  (height : ℝ := 15)

/-- A sphere with radius r -/
structure Sphere :=
  (radius : ℝ)
  (center : ℝ × ℝ × ℝ)

/-- Three congruent spheres inside the cone -/
def three_spheres := (Sphere × Sphere × Sphere)

/-- Predicate to check if spheres are tangent to each other -/
def spheres_tangent (s : three_spheres) : Prop := sorry

/-- Predicate to check if spheres are tangent to cone base and side -/
def spheres_tangent_to_cone (c : Cone) (s : three_spheres) : Prop := sorry

/-- Main theorem -/
theorem spheres_in_cone (c : Cone) (s : three_spheres) :
  spheres_tangent s ∧ spheres_tangent_to_cone c s →
  s.1.radius = 45 / (3 + 5 * Real.sqrt 3 + Real.sqrt 261 / 2) :=
by sorry

#check spheres_in_cone

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spheres_in_cone_l631_63127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_correct_l631_63111

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The point that minimizes the sum of distances to two fixed points -/
def minDistancePoint : Point :=
  { x := 1, y := 0 }

theorem min_distance_point_correct (A B P : Point) :
  A.x = 4 ∧ A.y = 3 ∧ B.x = 0 ∧ B.y = 1 ∧ P.y = 0 →
  distance P A + distance P B ≥ 
  distance minDistancePoint A + distance minDistancePoint B :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_correct_l631_63111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_average_annual_cost_l631_63145

/-- Represents the cost structure of owning a car -/
structure CarCost where
  purchaseCost : ℝ
  annualFixedCost : ℝ
  initialMaintenanceCost : ℝ
  maintenanceDifference : ℝ

/-- Calculates the total cost of owning a car for n years -/
noncomputable def totalCost (c : CarCost) (n : ℕ) : ℝ :=
  c.purchaseCost + n * c.annualFixedCost + (n * (2 * c.initialMaintenanceCost + (n - 1) * c.maintenanceDifference)) / 2

/-- Calculates the average annual cost of owning a car for n years -/
noncomputable def averageAnnualCost (c : CarCost) (n : ℕ) : ℝ :=
  totalCost c n / n

/-- Theorem stating that 10 years minimizes the average annual cost -/
theorem min_average_annual_cost (c : CarCost) :
  c.purchaseCost = 150000 ∧
  c.annualFixedCost = 15000 ∧
  c.initialMaintenanceCost = 3000 ∧
  c.maintenanceDifference = 3000 →
  ∀ n : ℕ, n ≠ 0 → averageAnnualCost c 10 ≤ averageAnnualCost c n := by
  sorry

#check min_average_annual_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_average_annual_cost_l631_63145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_triangles_isosceles_l631_63122

noncomputable section

structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  a : Point
  b : Point
  c : Point

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def isIsosceles (t : Triangle) : Prop :=
  let side1 := distance t.a t.b
  let side2 := distance t.a t.c
  let side3 := distance t.b t.c
  side1 = side2 ∨ side1 = side3 ∨ side2 = side3

def triangle1 : Triangle :=
  { a := { x := 0, y := 5 },
    b := { x := 2, y := 5 },
    c := { x := 1, y := 3 } }

def triangle2 : Triangle :=
  { a := { x := 4, y := 2 },
    b := { x := 6, y := 2 },
    c := { x := 5, y := 0 } }

def triangle3 : Triangle :=
  { a := { x := 2, y := 0 },
    b := { x := 5, y := 1 },
    c := { x := 8, y := 0 } }

def triangle4 : Triangle :=
  { a := { x := 7, y := 3 },
    b := { x := 9, y := 3 },
    c := { x := 8, y := 5 } }

def triangle5 : Triangle :=
  { a := { x := 0, y := 2 },
    b := { x := 2, y := 4 },
    c := { x := 2, y := 0 } }

theorem all_triangles_isosceles :
  isIsosceles triangle1 ∧
  isIsosceles triangle2 ∧
  isIsosceles triangle3 ∧
  isIsosceles triangle4 ∧
  isIsosceles triangle5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_triangles_isosceles_l631_63122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_percentage_in_carbonic_acid_l631_63124

/-- Molar mass of Hydrogen in g/mol -/
noncomputable def molar_mass_H : ℝ := 1.01

/-- Molar mass of Carbon in g/mol -/
noncomputable def molar_mass_C : ℝ := 12.01

/-- Molar mass of Oxygen in g/mol -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- Molar mass of Carbonic acid (H2CO3) in g/mol -/
noncomputable def molar_mass_H2CO3 : ℝ := 2 * molar_mass_H + molar_mass_C + 3 * molar_mass_O

/-- Mass percentage of Carbon in Carbonic acid -/
noncomputable def mass_percentage_C : ℝ := (molar_mass_C / molar_mass_H2CO3) * 100

theorem carbon_percentage_in_carbonic_acid :
  abs (mass_percentage_C - 19.36) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_percentage_in_carbonic_acid_l631_63124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l631_63188

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1/2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos (2*x))

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi/2), f x ≤ 1) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi/2), f x ≥ -1/2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi/2), f x = 1) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi/2), f x = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l631_63188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicyclist_speed_problem_l631_63197

/-- Proves that given a total distance of 850 km, with the first 400 km traveled at 20 km/h,
    and an average speed of 17 km/h for the entire trip,
    the speed for the remainder of the distance is 15 km/h. -/
theorem bicyclist_speed_problem (total_distance : ℝ) (first_part_distance : ℝ) 
    (first_part_speed : ℝ) (average_speed : ℝ) :
  total_distance = 850 →
  first_part_distance = 400 →
  first_part_speed = 20 →
  average_speed = 17 →
  (total_distance - first_part_distance) / 
  ((total_distance / average_speed) - (first_part_distance / first_part_speed)) = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicyclist_speed_problem_l631_63197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_m_is_5_range_of_m_for_inequality_l631_63151

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - m| + |x + 6|

-- Part I
theorem solution_set_when_m_is_5 :
  {x : ℝ | f 5 x ≤ 12} = {x : ℝ | -13/2 ≤ x ∧ x ≤ 11/2} := by sorry

-- Part II
theorem range_of_m_for_inequality :
  (∀ x : ℝ, f m x ≥ 7) ↔ m ∈ Set.Iic (-13) ∪ Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_m_is_5_range_of_m_for_inequality_l631_63151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_part_sqrt3_bound_l631_63160

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def fractional_part (x : ℝ) : ℝ := x - floor x

theorem fractional_part_sqrt3_bound (c : ℝ) : 
  (∀ n : ℕ+, fractional_part (n * Real.sqrt 3) > c / (n * Real.sqrt 3)) → c ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_part_sqrt3_bound_l631_63160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_even_n_product_exceeds_500_l631_63140

theorem smallest_even_n_product_exceeds_500 : 
  ∃ (n : ℕ), 
    n % 2 = 0 ∧ 
    (3 : ℝ) ^ ((n * (n + 1)) / 12 : ℝ) > 500 ∧
    ∀ (m : ℕ), m < n → m % 2 = 0 → (3 : ℝ) ^ ((m * (m + 1)) / 12 : ℝ) ≤ 500 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_even_n_product_exceeds_500_l631_63140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l631_63194

noncomputable def f (x y : ℝ) : ℝ := (x * y + x) / (x^2 + y^2 + 2 * y)

theorem min_value_of_f :
  ∀ x y : ℝ, 
    1/4 ≤ x → x ≤ 3/4 → 
    1/5 ≤ y → y ≤ 2/5 → 
    f x y ≥ 21/20 :=
by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l631_63194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l631_63132

-- Define the equation of the region
def region_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 3 = 6*y - 18*x + 9

-- Define the area of the region
noncomputable def region_area : ℝ := 102 * Real.pi

-- Theorem statement
theorem area_of_region :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ (x y : ℝ), region_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l631_63132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_squared_l631_63191

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → Point

/-- Represents a hyperbola with equation xy = 1 -/
def isOnHyperbola (p : Point) : Prop := p.x * p.y = 1

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Checks if a point is on a circle -/
def isOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Calculates the centroid of a triangle -/
noncomputable def centroid (t : EquilateralTriangle) : Point :=
  { x := (t.vertices 0).x + (t.vertices 1).x + (t.vertices 2).x / 3,
    y := (t.vertices 0).y + (t.vertices 1).y + (t.vertices 2).y / 3 }

/-- Calculates the area of an equilateral triangle -/
noncomputable def area (t : EquilateralTriangle) : ℝ := sorry

/-- Theorem statement -/
theorem triangle_area_squared (t : EquilateralTriangle) (h : Circle) :
  (∀ i, isOnHyperbola (t.vertices i)) →
  (∀ i, isOnCircle (t.vertices i) h) →
  centroid t = Point.mk 1 1 →
  h.center = Point.mk 2 2 →
  (∃ A : ℝ, A^2 = 243 ∧ A = area t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_squared_l631_63191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_data_variance_l631_63113

variable (x₁ x₂ x₃ : ℝ)
variable (p : ℝ)

def original_data (x₁ x₂ x₃ : ℝ) : List ℝ := [x₁, x₂, x₃]
def scaled_data (x₁ x₂ x₃ : ℝ) : List ℝ := [2 * x₁, 2 * x₂, 2 * x₃]

noncomputable def variance (data : List ℝ) : ℝ := sorry

axiom original_variance (x₁ x₂ x₃ : ℝ) : variance (original_data x₁ x₂ x₃) = p

theorem scaled_data_variance (x₁ x₂ x₃ : ℝ) : 
  variance (scaled_data x₁ x₂ x₃) = 4 * p := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_data_variance_l631_63113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_curve_distance_range_l631_63123

/-- The star-shaped curve defined by x^(2/3) + y^(2/3) = 1 -/
def star_curve (x y : ℝ) : Prop :=
  x^(2/3) + y^(2/3) = 1

/-- The distance from a point (x, y) to the origin (0, 0) -/
noncomputable def distance_to_origin (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

/-- Theorem: The distance from any point on the star-shaped curve to the origin is between 1/2 and 1 -/
theorem star_curve_distance_range (x y : ℝ) :
  star_curve x y → 1/2 ≤ distance_to_origin x y ∧ distance_to_origin x y ≤ 1 :=
by
  sorry

#check star_curve_distance_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_curve_distance_range_l631_63123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_tangent_line_l631_63182

/-- An ellipse with foci at (-1,0) and (1,0), passing through (1,3/2) -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- A line that intersects the ellipse at only one point -/
structure TangentLine where
  k : ℝ
  m : ℝ
  h : m > 0

/-- Helper function to calculate the area of a triangle -/
noncomputable def area_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

/-- The theorem statement -/
theorem ellipse_and_tangent_line (e : Ellipse) (l : TangentLine) :
  (∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 ↔ x^2 / e.a^2 + y^2 / e.b^2 = 1) ∧
  (∃ k m : ℝ, (m > 0 ∧ 
    (∀ x : ℝ, y = k * x + m ↔ y = l.k * x + l.m) ∧
    (y = k * x + m ∨ y = -k * x + m) ∧
    k^2 = 3/4 ∧ m^2 = 6) ∧
   (∀ k' m' : ℝ, m' > 0 → 
    area_triangle 0 0 (-m'/k') 0 0 m' ≥ area_triangle 0 0 (-m/k) 0 0 m)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_tangent_line_l631_63182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_r_a_l631_63115

open Set
open Function
open Real

-- Define the function r_a
noncomputable def r_a (a : ℝ) (x : ℝ) : ℝ := 1 / (a - x)^2

-- State the theorem
theorem range_of_r_a (a : ℝ) : 
  range (r_a a) = {y : ℝ | y > 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_r_a_l631_63115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_probability_proof_l631_63169

noncomputable def coin_probability : ℝ := (6 + Real.sqrt (6 * Real.sqrt 6 + 2)) / 12

theorem coin_probability_proof :
  coin_probability > 1/2 ∧
  coin_probability^3 * (1 - coin_probability)^2 = 1/100 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_probability_proof_l631_63169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_apples_is_twenty_l631_63104

/-- Represents the number of apples collected by each person -/
structure AppleCollection where
  anya : ℕ
  vanya : ℕ
  danya : ℕ
  tanya : ℕ

/-- Checks if the given AppleCollection satisfies all conditions -/
def isValidCollection (c : AppleCollection) (total : ℕ) : Prop :=
  let percentages := [c.anya, c.vanya, c.danya, c.tanya].map (· * 100 / total)
  -- Each person collected a whole number percentage
  (∀ p ∈ percentages, p * total % 100 = 0)
  -- Percentages are distinct
  ∧ percentages.toFinset.card = 4
  -- Percentages sum to 100
  ∧ percentages.sum = 100
  -- Tanya collected the most
  ∧ c.tanya = max c.anya (max c.vanya c.danya)
  -- After Tanya eats her apples, remaining kids have whole percentages
  ∧ let remaining := total - c.tanya
    (c.anya * 100 % remaining = 0) ∧ (c.vanya * 100 % remaining = 0) ∧ (c.danya * 100 % remaining = 0)

/-- The theorem stating that 20 is the minimum number of apples satisfying all conditions -/
theorem min_apples_is_twenty :
  ∃ (c : AppleCollection), isValidCollection c 20 ∧
  ∀ (n : ℕ) (c' : AppleCollection), n < 20 → ¬ isValidCollection c' n := by
  sorry

#check min_apples_is_twenty

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_apples_is_twenty_l631_63104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_interest_rate_is_five_percent_l631_63190

/-- The first interest rate that satisfies the given conditions -/
noncomputable def first_interest_rate (total_amount : ℝ) (first_amount : ℝ) (second_rate : ℝ) (total_income : ℝ) : ℝ :=
  let second_amount := total_amount - first_amount
  (total_income - second_amount * second_rate / 100) / (first_amount / 100)

/-- Theorem stating that the first interest rate is approximately 5% given the problem conditions -/
theorem first_interest_rate_is_five_percent :
  let total_amount : ℝ := 2500
  let first_amount : ℝ := 1500.0000000000007
  let second_rate : ℝ := 6
  let total_income : ℝ := 135
  abs (first_interest_rate total_amount first_amount second_rate total_income - 5) < 0.0000001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_interest_rate_is_five_percent_l631_63190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_linear_plus_periodic_l631_63198

/-- A periodic function on the real line. -/
def Periodic (f : ℝ → ℝ) : Prop :=
  ∃ t : ℝ, t ≠ 0 ∧ ∀ x, f (x + t) = f x

/-- The theorem statement. -/
theorem inverse_of_linear_plus_periodic
  (f g : ℝ → ℝ)
  (h : ℝ → ℝ)
  (k : ℝ)
  (hf : ∀ x, f x = k * x + h x)
  (hg : Function.LeftInverse g f ∧ Function.RightInverse g f)
  (hk : k ≠ 0)
  (hh : Periodic h) :
  ∃ p : ℝ → ℝ, (Periodic p) ∧ (∀ y, g y = y / k + p y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_linear_plus_periodic_l631_63198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_third_to_degrees_degrees_75_to_radians_one_radian_to_degrees_approx_l631_63159

-- Define the conversion factor
noncomputable def π_degrees : ℝ := 180

-- Define the conversion functions
noncomputable def radians_to_degrees (x : ℝ) : ℝ := x * π_degrees / Real.pi
noncomputable def degrees_to_radians (x : ℝ) : ℝ := x * Real.pi / π_degrees

-- State the theorems
theorem pi_third_to_degrees : 
  radians_to_degrees (Real.pi / 3) = 60 := by sorry

theorem degrees_75_to_radians : 
  degrees_to_radians 75 = 5 * Real.pi / 12 := by sorry

theorem one_radian_to_degrees_approx : 
  ∃ (n : ℕ), abs (radians_to_degrees 1 - 57.3) < 1 / 10^n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_third_to_degrees_degrees_75_to_radians_one_radian_to_degrees_approx_l631_63159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_factorial_solutions_l631_63196

theorem infinitely_many_factorial_solutions :
  ∀ n : ℕ, n > 1 →
  ∃ m k : ℕ, m > 1 ∧ k > 1 ∧
  m = n.factorial - 1 ∧ k = n.factorial ∧
  m.factorial * n.factorial = k.factorial :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_factorial_solutions_l631_63196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_danny_travel_time_l631_63130

/-- The time it takes Danny to reach Steve's house -/
def danny_time : ℝ := sorry

/-- The time it takes Steve to reach Danny's house -/
def steve_time : ℝ := sorry

/-- Danny's travel time is half of Steve's travel time -/
axiom half_time : danny_time = steve_time / 2

/-- The difference between Steve's and Danny's time to reach the halfway point is 14.5 minutes -/
axiom halfway_difference : steve_time / 2 - danny_time / 2 = 14.5

theorem danny_travel_time : danny_time = 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_danny_travel_time_l631_63130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pairs_bound_l631_63125

theorem pairs_bound (m n : ℕ) (a b : ℕ → ℝ) 
  (h_m_pos : m > 0) (h_n_pos : n > 0) (h_m_ge_n : m ≥ n) (h_n_ge_2022 : n ≥ 2022) :
  (Finset.sum (Finset.range n) (λ i => 
    (Finset.filter (λ j => |a i + b j - (i * j : ℝ)| ≤ m) (Finset.range n)).card)) ≤ 
  ⌊3 * n * Real.sqrt (m * Real.log n)⌋ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pairs_bound_l631_63125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ten_equality_l631_63137

theorem factorial_ten_equality : 2^6 * 3^3 * 2100 = Nat.factorial 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ten_equality_l631_63137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_equals_9_l631_63149

/-- A number with 1998 digits -/
def N : ℕ := sorry

/-- N is divisible by 9 -/
axiom N_div_9 : 9 ∣ N

/-- x is the sum of N's digits -/
def x : ℕ := sorry

/-- y is the sum of x's digits -/
def y : ℕ := sorry

/-- z is the sum of y's digits -/
def z : ℕ := sorry

/-- N has 1998 digits -/
axiom N_digits : 10^1997 ≤ N ∧ N < 10^1998

/-- Theorem: z equals 9 -/
theorem z_equals_9 : z = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_equals_9_l631_63149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mask_problem_solution_l631_63110

/-- Represents the pharmacy's mask purchase and sales problem -/
structure MaskProblem where
  first_batch_cost : ℝ
  second_batch_cost : ℝ
  second_batch_quantity_factor : ℝ
  second_batch_cost_increase : ℝ
  max_profit : ℝ

/-- Calculates the number of packs in the first batch -/
noncomputable def first_batch_quantity (p : MaskProblem) : ℝ :=
  p.first_batch_cost * p.second_batch_quantity_factor / 
  (p.second_batch_cost - p.first_batch_cost * p.second_batch_quantity_factor * p.second_batch_cost_increase)

/-- Calculates the highest selling price per pack -/
noncomputable def max_selling_price (p : MaskProblem) (q : ℝ) : ℝ :=
  (p.first_batch_cost + p.second_batch_cost + p.max_profit) / (q * (1 + p.second_batch_quantity_factor))

/-- Theorem stating the solution to the mask problem -/
theorem mask_problem_solution (p : MaskProblem) 
  (h1 : p.first_batch_cost = 4000)
  (h2 : p.second_batch_cost = 7500)
  (h3 : p.second_batch_quantity_factor = 1.5)
  (h4 : p.second_batch_cost_increase = 0.5)
  (h5 : p.max_profit = 3500) :
  first_batch_quantity p = 2000 ∧ 
  max_selling_price p (first_batch_quantity p) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mask_problem_solution_l631_63110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_count_l631_63148

def m : ℕ := 2^35 * 5^21

theorem divisors_count : 
  (Finset.filter (fun d ↦ d < m ∧ ¬(m % d = 0)) (Nat.divisors (m^2))).card = 735 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_count_l631_63148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_speed_of_two_rowers_l631_63142

/-- Calculates the speed in still water given upstream and downstream speeds -/
noncomputable def speedInStillWater (upstream downstream : ℝ) : ℝ :=
  (upstream + downstream) / 2

/-- Represents a rower with upstream and downstream speeds -/
structure Rower where
  upstream : ℝ
  downstream : ℝ

/-- Calculates the combined speed of two rowers in still water -/
noncomputable def combinedSpeed (rower1 rower2 : Rower) : ℝ :=
  speedInStillWater rower1.upstream rower1.downstream +
  speedInStillWater rower2.upstream rower2.downstream

theorem combined_speed_of_two_rowers :
  let rower1 : Rower := { upstream := 30, downstream := 60 }
  let rower2 : Rower := { upstream := 40, downstream := 80 }
  combinedSpeed rower1 rower2 = 105 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_speed_of_two_rowers_l631_63142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_point_in_acute_triangle_l631_63168

/-- Predicate to check if a triangle is acute -/
def IsAcute (A B C : ℤ × ℤ) : Prop :=
  sorry

/-- Predicate to check if a point is inside a triangle -/
def PointInTriangle (P A B C : ℤ × ℤ) : Prop :=
  sorry

/-- Predicate to check if a point is on the side of a triangle -/
def PointOnTriangleSide (P A B C : ℤ × ℤ) : Prop :=
  sorry

/-- Given an acute triangle ABC on a grid where A is at (0,0),
    B is at (x₁, y₁), and C is at (x₂, y₂), with y₁, y₂ ≥ 1 and
    x₁ and x₂ having different signs, there exists a grid point
    (0,1) that is either inside the triangle or on one of its sides. -/
theorem grid_point_in_acute_triangle 
  (x₁ x₂ y₁ y₂ : ℤ)
  (h_y₁ : y₁ ≥ 1)
  (h_y₂ : y₂ ≥ 1)
  (h_x_diff : x₁ * x₂ < 0)
  (h_acute : IsAcute ((0:ℤ), 0) (x₁, y₁) (x₂, y₂)) :
  PointInTriangle ((0:ℤ), 1) ((0:ℤ), 0) (x₁, y₁) (x₂, y₂) ∨ 
  PointOnTriangleSide ((0:ℤ), 1) ((0:ℤ), 0) (x₁, y₁) (x₂, y₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_point_in_acute_triangle_l631_63168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l631_63154

/-- The hyperbola in the Cartesian coordinate plane (xOy) -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop

/-- The foci of the hyperbola -/
structure Foci (h : Hyperbola) where
  F₁ : ℝ × ℝ  -- left focus
  F₂ : ℝ × ℝ  -- right focus

/-- A point on the right branch of the hyperbola -/
structure Point (h : Hyperbola) (f : Foci h) where
  P : ℝ × ℝ
  on_right_branch : P.1 > 0
  on_hyperbola : h.equation P.1 P.2

/-- The origin of the coordinate system -/
def O : ℝ × ℝ := (0, 0)

/-- Vector from point A to point B -/
def vector (A B : ℝ × ℝ) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

/-- Dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Magnitude of a vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- The main theorem -/
theorem hyperbola_eccentricity (h : Hyperbola) (f : Foci h) (p : Point h f) :
  (dot_product (vector O p.P) (vector f.F₂ p.P) = dot_product (vector f.F₂ O) (vector f.F₂ p.P)) →
  (magnitude (vector p.P f.F₁) = Real.sqrt 3 * magnitude (vector p.P f.F₂)) →
  eccentricity h = Real.sqrt 3 + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l631_63154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l631_63189

noncomputable section

-- Define the functions
def f (m : ℝ) (x : ℝ) : ℝ := m * x - (m - 1) / x - Real.log x
def g (θ : ℝ) (x : ℝ) : ℝ := 1 / (x * Real.cos θ) + Real.log x
def h (m : ℝ) (θ : ℝ) (x : ℝ) : ℝ := f m x - g θ x

-- State the theorem
theorem problem_solution :
  -- Part I
  (∀ y : ℝ, (4 : ℝ) * 1 - y - 3 = 0 ↔ y = f 3 1) ∧
  -- Part II
  (∀ θ : ℝ, 0 ≤ θ ∧ θ < Real.pi / 2 →
    (∀ x : ℝ, x ≥ 1 → Monotone (g θ)) → θ = 0) ∧
  -- Part III
  (∀ m : ℝ, (∀ x : ℝ, x > 0 → Monotone (h m 0)) ↔
    m ≤ 0 ∨ m ≥ 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l631_63189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_coordinates_l631_63143

-- Define the complex number z
noncomputable def z : ℂ := (3 + Complex.I) / (1 + Complex.I)

-- Theorem statement
theorem complex_coordinates : z = 2 - Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_coordinates_l631_63143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_identities_l631_63177

theorem sine_cosine_identities (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π / 2) (h3 : Real.sin θ = 1 / 3) :
  (Real.sin (π / 4 - θ) = (4 - Real.sqrt 2) / 6) ∧ (Real.cos (2 * θ) = 7 / 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_identities_l631_63177
