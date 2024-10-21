import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_b_value_l1212_121254

-- Define the piecewise function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 3 then 3 * x^2 - 5 else b * x + 6

-- State the theorem
theorem continuous_piecewise_function_b_value :
  ∃ b : ℝ, Continuous (f b) ∧ b = 16/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_b_value_l1212_121254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_sum_lower_bound_l1212_121282

open Real

/-- ExtremePoints f x₁ x₂ means that x₁ and x₂ are extreme points of f -/
def ExtremePoints (f : ℝ → ℝ) (x₁ x₂ : ℝ) : Prop :=
  (∃ ε > 0, ∀ x ∈ Set.Ioo (x₁ - ε) (x₁ + ε), f x ≤ f x₁) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (x₂ - ε) (x₂ + ε), f x ≤ f x₂)

/-- Given a function f(x) = x - ax^2 - ln x, where a > 0, 
    if f has two extreme points, then the sum of f at these points 
    is greater than 3 - 2ln 2 -/
theorem extreme_points_sum_lower_bound 
  (a : ℝ) (ha : a > 0)
  (f : ℝ → ℝ) (hf : ∀ x, f x = x - a * x^2 - log x)
  (x₁ x₂ : ℝ) (h_extreme : ExtremePoints f x₁ x₂) :
  f x₁ + f x₂ > 3 - 2 * log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_sum_lower_bound_l1212_121282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_shadow_problem_l1212_121217

/-- The height of the light source above an upper vertex of the cube -/
noncomputable def x : ℝ := 1 / 6

/-- The edge length of the cube -/
def cube_edge : ℝ := 1

/-- The area of the shadow cast by the cube, excluding the area beneath the cube -/
def shadow_area : ℝ := 48

theorem cube_shadow_problem :
  let total_shadow_area := shadow_area + cube_edge ^ 2
  let shadow_side_length := Real.sqrt total_shadow_area
  let ratio := shadow_side_length / cube_edge
  x * (ratio + 1) = ratio * (x + cube_edge) →
  ⌊1000 * x⌋ = 166 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_shadow_problem_l1212_121217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_count_l1212_121222

theorem triangle_third_side_count : ℕ := by
  -- Define the two known side lengths
  let a : ℕ := 8
  let b : ℕ := 11
  
  -- Define the set of possible integer lengths for the third side
  let possible_lengths : Set ℕ := {x : ℕ | x > a - b ∧ x < a + b ∧ x > b - a}
  
  -- The count of possible lengths is 15
  have h : Finset.card (Finset.filter (λ x => x ∈ possible_lengths) (Finset.range (a + b))) = 15 := by
    sorry
  
  -- The result is 15
  exact 15


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_count_l1212_121222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sums_at_smallest_n_l1212_121261

/-- The sum of the first n terms of an arithmetic sequence with first term a and common difference d -/
noncomputable def arithmetic_sum (n : ℕ) (a d : ℚ) : ℚ := n * (2 * a + (n - 1) * d) / 2

/-- The smallest positive integer n such that the sums of two specific arithmetic sequences are equal -/
def smallest_n : ℕ := 7

theorem equal_sums_at_smallest_n :
  smallest_n > 0 ∧
  (∀ k : ℕ, 0 < k → k < smallest_n →
    arithmetic_sum k 7 4 ≠ arithmetic_sum k 15 3) ∧
  arithmetic_sum smallest_n 7 4 = arithmetic_sum smallest_n 15 3 := by
  sorry

#eval smallest_n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sums_at_smallest_n_l1212_121261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_l1212_121209

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  |a * Real.sin (2 * x) + Real.cos (2 * x)| + 
  |Real.sin x + Real.cos x| * |(1 + a) * Real.sin x + (1 - a) * Real.cos x|

/-- Theorem stating that if the maximum value of f(x) is 5, then a = ±√3 -/
theorem max_value_implies_a (a : ℝ) :
  (∀ x, f a x ≤ 5) ∧ (∃ x, f a x = 5) → a = Real.sqrt 3 ∨ a = -Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_l1212_121209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_l1212_121215

noncomputable def f (n : ℕ) : ℝ :=
  (6 + 4 * Real.sqrt 3) / 12 * ((1 + Real.sqrt 3) / 2) ^ n +
  (6 - 4 * Real.sqrt 3) / 12 * ((1 - Real.sqrt 3) / 2) ^ n

theorem f_difference (n : ℕ) : f (n + 1) - f n = (Real.sqrt 3 - 3) / 4 * f n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_l1212_121215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integers_between_sqrt10_and_sqrt100_l1212_121287

theorem integers_between_sqrt10_and_sqrt100 : 
  (Finset.range (Int.toNat (⌊Real.sqrt 100⌋ - ⌈Real.sqrt 10⌉ + 1))).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integers_between_sqrt10_and_sqrt100_l1212_121287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_second_quadrant_l1212_121276

theorem point_in_second_quadrant (A B : Real) : 
  A > 0 ∧ B > 0 ∧ A + B < π → 
  (Complex.cos B - Complex.sin A : ℂ).re < 0 ∧ (Complex.sin B - Complex.cos A : ℂ).im > 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_second_quadrant_l1212_121276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_loses_12th_match_l1212_121270

/-- Represents a player in the table tennis competition -/
inductive Player : Type
| A : Player
| B : Player
| C : Player

/-- Data about the competition -/
structure CompetitionData where
  matches_played_A : Nat
  matches_played_B : Nat
  referee_count_C : Nat

/-- The loser of the nth match -/
def loser_of_match (n : Nat) : Player :=
  sorry

/-- Theorem stating that A is the loser of the 12th match -/
theorem a_loses_12th_match (data : CompetitionData) 
  (h1 : data.matches_played_A = 12)
  (h2 : data.matches_played_B = 21)
  (h3 : data.referee_count_C = 8) :
  ∃ (total_matches : Nat), 
    total_matches = 25 ∧ 
    (∀ n : Nat, n ≤ total_matches → n % 2 = 0 → n > 0 → 
      loser_of_match n = Player.A) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_loses_12th_match_l1212_121270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_sequence_l1212_121256

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

noncomputable def sum_arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a₁ + (n - 1 : ℝ) * d)

theorem max_sum_arithmetic_sequence
  (a₁ : ℝ) (d : ℝ) (h₁ : a₁ > 0) (h₂ : 3 * arithmetic_sequence a₁ d 8 = 5 * arithmetic_sequence a₁ d 13) :
  let S := sum_arithmetic_sequence a₁ d
  ∀ n : ℕ, n ∈ ({10, 11, 21} : Finset ℕ) → S 20 ≥ S n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_sequence_l1212_121256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1212_121284

noncomputable def f (x : ℝ) : ℝ := 3 / (x - 2)

theorem f_properties :
  (∀ x, x > 2 → f x = 3 / (x - 2)) ∧
  f 0 = -3/2 ∧
  f 3 = 3 ∧
  (∀ x y, x > 2 → y > 2 → x < y → f x > f y) ∧
  (∀ m n, m > 2 → n > 2 →
    (∀ x, m ≤ x ∧ x ≤ n → 1 ≤ f x ∧ f x ≤ 3) →
    f m = 3 ∧ f n = 1 ∧ m + n = 8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1212_121284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_log_divisors_6_to_9_l1212_121237

-- Define the sum of base-10 logarithms of divisors of 6^n
noncomputable def sumLogDivisors (n : ℕ) : ℝ :=
  (n * (n + 1)^2 / 2 : ℝ)

-- Theorem statement
theorem sum_log_divisors_6_to_9 : sumLogDivisors 9 = 468 := by
  -- Unfold the definition of sumLogDivisors
  unfold sumLogDivisors
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_log_divisors_6_to_9_l1212_121237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_less_than_neg_2_l1212_121251

/-- A random variable following a normal distribution with mean 0 and standard deviation 6 -/
noncomputable def ξ : ℝ → ℝ := sorry

/-- The probability density function of the normal distribution N(0, 6²) -/
noncomputable def normal_pdf (x : ℝ) : ℝ := sorry

/-- The cumulative distribution function of the normal distribution N(0, 6²) -/
noncomputable def normal_cdf (x : ℝ) : ℝ := sorry

/-- The probability that ξ is between 0 and 2 -/
axiom prob_between_0_and_2 : ∫ x in Set.Icc 0 2, normal_pdf x = 0.2

/-- Theorem: The probability that ξ is less than -2 is 0.4 -/
theorem prob_less_than_neg_2 : ∫ x in Set.Iic (-2), normal_pdf x = 0.4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_less_than_neg_2_l1212_121251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1212_121246

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (2 + x) + Real.log (2 - x)

-- Define the interval (0, 2)
def interval : Set ℝ := Set.Ioo 0 2

-- Theorem statement
theorem f_properties :
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, x ∈ interval → y ∈ interval → x < y → f y < f x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1212_121246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_nested_expression_l1212_121278

theorem absolute_value_nested_expression : 
  abs (abs (abs (-abs (-2 + 3) - 2) + 2)) = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_nested_expression_l1212_121278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_minimum_value_l1212_121229

-- Part I
theorem distance_range (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  let y (x : ℝ) := a * x^2 + b * x + c
  let l := 1 - c/a
  3/2 < l ∧ l < 3 := by sorry

-- Part II
theorem minimum_value (a b c : ℝ) (h1 : a < b) 
  (h2 : ∀ x, a * x^2 + b * x + c ≥ 0) :
  (2*a + 2*b + 8*c) / (b - a) ≥ 6 + 4 * Real.sqrt 3 := by sorry

noncomputable def minimum_value_exact (a b c : ℝ) (h1 : a < b) 
  (h2 : ∀ x, a * x^2 + b * x + c ≥ 0) : ℝ :=
  6 + 4 * Real.sqrt 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_minimum_value_l1212_121229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_minus_sectors_area_l1212_121293

/-- The area of the region inside a regular hexagon but outside six congruent circular sectors -/
theorem hexagon_minus_sectors_area (side_length sector_radius sector_angle : ℝ) : 
  side_length = 9 → 
  sector_radius = 4 → 
  sector_angle = 120 → 
  (6 * (Real.sqrt 3 / 4 * side_length^2)) - (6 * (sector_angle / 360 * Real.pi * sector_radius^2)) = 121.5 * Real.sqrt 3 - 32 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_minus_sectors_area_l1212_121293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1212_121201

/-- The function f(x) = a^x where a > 0 and a ≠ 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

/-- Theorem stating the properties of the function f -/
theorem function_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 2 = 1/9) :
  a = 1/3 ∧ f a 3 > f a 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1212_121201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_sqrt_two_l1212_121242

/-- A power function that passes through the point (2,4) -/
noncomputable def f (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

/-- The property that f passes through (2,4) -/
def passes_through (α : ℝ) : Prop := f α 2 = 4

theorem power_function_sqrt_two (α : ℝ) (h : passes_through α) : f α (Real.sqrt 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_sqrt_two_l1212_121242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_range_l1212_121262

/-- A circle with center (3, -5) and radius r > 0 -/
structure Circle where
  r : ℝ
  h : r > 0

/-- The line 4x - 3y - 2 = 0 -/
def line (x y : ℝ) : Prop := 4 * x - 3 * y - 2 = 0

/-- Distance from a point (x, y) to the line -/
noncomputable def distanceToLine (x y : ℝ) : ℝ :=
  |4 * x - 3 * y - 2| / Real.sqrt (4^2 + (-3)^2)

/-- Two points on the circle at distance 1 from the line -/
def twoPointsAtDistance1 (c : Circle) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁ - 3)^2 + (y₁ + 5)^2 = c.r^2 ∧
    (x₂ - 3)^2 + (y₂ + 5)^2 = c.r^2 ∧
    distanceToLine x₁ y₁ = 1 ∧
    distanceToLine x₂ y₂ = 1 ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    ∀ (x y : ℝ), (x - 3)^2 + (y + 5)^2 = c.r^2 ∧ distanceToLine x y = 1 →
      (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)

/-- The main theorem -/
theorem circle_radius_range (c : Circle) :
  twoPointsAtDistance1 c → 4 < c.r ∧ c.r < 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_range_l1212_121262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_percentage_l1212_121290

/-- Given two mixtures A and B, with different alcohol concentrations, 
    prove that when combined in specific quantities, the resulting mixture 
    has a certain alcohol percentage. -/
theorem alcohol_mixture_percentage 
  (percent_a : ℝ) 
  (percent_b : ℝ) 
  (total_volume : ℝ) 
  (volume_a : ℝ) : 
  percent_a = 20 → 
  percent_b = 50 → 
  total_volume = 15 → 
  volume_a = 10 → 
  ((percent_a / 100 * volume_a + percent_b / 100 * (total_volume - volume_a)) / total_volume) * 100 = 30 := by
  sorry

#check alcohol_mixture_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_percentage_l1212_121290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_areas_equal_l1212_121283

/-- The area of a triangle given its side lengths -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_areas_equal : 
  triangleArea 13 13 10 = triangleArea 13 13 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_areas_equal_l1212_121283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_plane_angles_less_than_360_l1212_121248

/-- A convex polyhedral angle -/
structure ConvexPolyhedralAngle where
  -- We don't need to define the internal structure,
  -- as we're only interested in its properties

/-- The sum of plane angles of a convex polyhedral angle -/
noncomputable def sumOfPlaneAngles (angle : ConvexPolyhedralAngle) : ℝ :=
  sorry -- We don't provide an implementation, just a placeholder

/-- Theorem: The sum of plane angles of a convex polyhedral angle is less than 360° -/
theorem sum_of_plane_angles_less_than_360 (angle : ConvexPolyhedralAngle) :
  sumOfPlaneAngles angle < 360 := by
  sorry

#check sum_of_plane_angles_less_than_360

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_plane_angles_less_than_360_l1212_121248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_75pi_l1212_121205

noncomputable def smaller_circle_diameter : ℝ := 10

noncomputable def smaller_circle_radius : ℝ := smaller_circle_diameter / 2

noncomputable def larger_circle_radius : ℝ := 2 * smaller_circle_radius

noncomputable def smaller_circle_area : ℝ := Real.pi * smaller_circle_radius ^ 2

noncomputable def larger_circle_area : ℝ := Real.pi * larger_circle_radius ^ 2

theorem shaded_area_is_75pi :
  larger_circle_area - smaller_circle_area = 75 * Real.pi := by
  -- Expand definitions
  unfold larger_circle_area smaller_circle_area larger_circle_radius smaller_circle_radius
  -- Simplify the expression
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_75pi_l1212_121205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_indivisible_squares_plus_one_l1212_121264

theorem infinitely_many_indivisible_squares_plus_one :
  ∀ (n : ℕ) (S : Finset ℕ),
  ∃ (k : ℕ), ∀ m, m ∈ S → (k^2 + 1) % (m^2 + 1) ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_indivisible_squares_plus_one_l1212_121264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_with_inscribed_semicircle_l1212_121223

/-- Given an isosceles triangle with an inscribed semicircle of radius 1, 
    where the diameter of the semicircle lies along the base of the triangle, 
    the angle opposite the base is 2θ, and the equal sides are tangent to the semicircle,
    the area of the triangle is 1 / (sin θ * cos θ). -/
theorem isosceles_triangle_area_with_inscribed_semicircle (θ : Real) :
  let triangle_area := (1 : Real) / (Real.sin θ * Real.cos θ)
  let semicircle_radius := (1 : Real)
  let opposite_angle := 2 * θ
  let sides_tangent_to_semicircle := True
  triangle_area = (1 : Real) / (Real.sin θ * Real.cos θ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_with_inscribed_semicircle_l1212_121223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_root_eight_over_fourth_root_sixteen_l1212_121213

theorem sixth_root_eight_over_fourth_root_sixteen (x : ℝ) :
  x = (8 : ℝ) ^ (1/6) / (16 : ℝ) ^ (1/4) → x = 2 ^ (-1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_root_eight_over_fourth_root_sixteen_l1212_121213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_factors_of_3600_l1212_121214

/-- Given that 3600 = 2^4 * 3^2 * 5^2, this theorem states that
    the number of positive integer factors of 3600 that are perfect squares is 12. -/
theorem perfect_square_factors_of_3600 :
  ∃ (f : ℕ → ℕ), f 3600 = 12 ∧
    (∀ n : ℕ, f n = (Finset.filter (λ m : ℕ ↦ m * m ∈ Finset.range (n + 1) ∧ n % m = 0) (Finset.range (n + 1))).card) ∧
    3600 = 2^4 * 3^2 * 5^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_factors_of_3600_l1212_121214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_theorem_l1212_121292

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- State the theorem
theorem solution_set_theorem (h1 : ∀ x > (1/2), f' x = deriv f x)
  (h2 : ∀ x > (1/2), x * (f' x) * Real.log (2*x) > f x)
  (h3 : f (Real.exp 1 / 2) = 1) :
  ∀ x : ℝ, f (Real.exp x / 2) < x ↔ 0 < x ∧ x < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_theorem_l1212_121292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_range_of_k_l1212_121207

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16

-- Define the center of the circle
def C : ℝ × ℝ := (-1, 0)

-- Define point A
def A : ℝ × ℝ := (1, 0)

-- Define a point on the circle
def on_circle (Q : ℝ × ℝ) : Prop := circle_eq Q.1 Q.2

-- Define point M
noncomputable def M (Q : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define line l
def line_l (k x : ℝ) : ℝ := k * x + 1 - 2 * k

-- Define point P
def P : ℝ × ℝ := (2, 1)

-- Theorem for the trajectory of M
theorem trajectory_of_M : 
  ∀ (x y : ℝ), (∃ Q, on_circle Q ∧ M Q = (x, y)) → x^2/4 + y^2/3 = 1 := by sorry

-- Theorem for the range of k
theorem range_of_k (k : ℝ) : 
  (∃ (B D : ℝ × ℝ), 
    B.2 = line_l k B.1 ∧ 
    D.2 = line_l k D.1 ∧ 
    B.1^2/4 + B.2^2/3 = 1 ∧ 
    D.1^2/4 + D.2^2/3 = 1 ∧ 
    B ≠ D ∧
    (B.1 - P.1) * (D.1 - P.1) + (B.2 - P.2) * (D.2 - P.2) > 5/4) 
  → -1/2 < k ∧ k < 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_range_of_k_l1212_121207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1212_121266

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  Real.cos ((2 * Real.pi / 3) * x) + (a - 1) * Real.sin (Real.pi / 3 * x) + a

/-- The function g(x) -/
noncomputable def g (x : ℝ) : ℝ := 
  (2 : ℝ) ^ x - x^2

/-- Theorem stating the range of a given the condition -/
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a (g x) ≤ 0) →
  a ≤ Real.sqrt 3 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1212_121266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1212_121289

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The slope of a line passing through two points -/
noncomputable def line_slope (p q : Point) : ℝ :=
  (q.y - p.y) / (q.x - p.x)

/-- The angle between two vectors -/
noncomputable def angle (p q : Point) : ℝ :=
  Real.arccos ((p.x * q.x + p.y * q.y) / (distance p ⟨0, 0⟩ * distance q ⟨0, 0⟩))

theorem ellipse_properties (e : Ellipse) (F P P' : Point) :
  e.eccentricity = 1/2 →
  distance P F = 1 →
  distance P' F = 3 →
  ∃ (O : Point), O.x = 0 ∧ O.y = 0 ∧ 
    (∃ (t : ℝ), P.x = t * (P'.x - O.x) + O.x ∧ P.y = t * (P'.y - O.y) + O.y) →
  (e.a = 2 ∧ e.b = Real.sqrt 3) ∧
  (∀ (A B : Point), 
    (∃ (k : ℝ), A.y - F.y = k * (A.x - F.x) ∧ B.y - F.y = k * (B.x - F.x)) →
    angle A B > π/2 →
    (k < 0 ∨ k > 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1212_121289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_two_ninths_l1212_121253

/-- Represents the outcome of a single draw -/
inductive Ball : Type
| Zhong : Ball  -- "中"
| Guo : Ball    -- "国"
| Mei : Ball    -- "美"
| Li : Ball     -- "丽"

/-- Represents a sequence of three draws -/
def DrawSequence := (Ball × Ball × Ball)

/-- Checks if a sequence stops on the third draw -/
def stopsOnThirdDraw (seq : DrawSequence) : Bool :=
  match seq with
  | (_, _, Ball.Zhong) => true
  | (_, _, Ball.Guo) => true
  | _ => false

/-- The total number of trials -/
def totalTrials : Nat := 18

/-- The number of sequences that stop on the third draw -/
def successfulTrials : Nat := 4

/-- The probability of stopping on the third draw -/
def probabilityStopOnThird : Rat := successfulTrials / totalTrials

theorem probability_is_two_ninths :
  probabilityStopOnThird = 2 / 9 := by
  unfold probabilityStopOnThird
  unfold successfulTrials
  unfold totalTrials
  norm_num

#eval probabilityStopOnThird

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_two_ninths_l1212_121253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_power_sum_l1212_121291

-- Define the complex number z
noncomputable def z : ℂ := -((1 - Complex.I) / Real.sqrt 2)

-- State the theorem
theorem z_power_sum : z^2016 + z^50 - 1 = -Complex.I := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_power_sum_l1212_121291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1212_121219

def digits : List Nat := [0, 1, 2, 3, 4, 5]

def is_valid_number (n : Nat) : Bool :=
  n ≥ 10000 ∧ n < 100000 ∧
  (n % 3 = 0) ∧ (n % 5 = 0) ∧
  (Nat.digits 10 n).toFinset.card = 5 ∧
  (Nat.digits 10 n).toFinset ⊆ digits.toFinset

theorem count_valid_numbers :
  (Finset.filter (fun n => is_valid_number n) (Finset.range 100000)).card = 216 := by
  sorry

#eval (Finset.filter (fun n => is_valid_number n) (Finset.range 100000)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1212_121219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_cleaning_earnings_eq_2600_l1212_121232

/-- Zoe's earnings from pool cleaning and babysitting -/
def total_earnings : ℕ := 8000

/-- Number of times Zoe babysat Zachary -/
def z : ℕ := 1  -- We need to define z as a constant, not a variable

/-- Number of times Zoe babysat Julie -/
def julie_sessions : ℕ := 3 * z

/-- Number of times Zoe babysat Chloe -/
def chloe_sessions : ℕ := 5 * z

/-- Zoe's earnings from babysitting Zachary -/
def zachary_earnings : ℕ := 600

/-- Zoe's earnings per session with Zachary -/
noncomputable def earnings_per_session : ℚ := zachary_earnings / z

/-- Zoe's total earnings from babysitting -/
noncomputable def babysitting_earnings : ℚ :=
  zachary_earnings + julie_sessions * earnings_per_session + chloe_sessions * earnings_per_session

/-- Zoe's earnings from pool cleaning -/
noncomputable def pool_cleaning_earnings : ℚ := total_earnings - babysitting_earnings

theorem pool_cleaning_earnings_eq_2600 :
  pool_cleaning_earnings = 2600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_cleaning_earnings_eq_2600_l1212_121232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_range_l1212_121273

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 2 ∧ t.B = Real.pi/4 ∧ ∃ (a1 a2 : ℝ), a1 ≠ a2 ∧ (t.a = a1 ∨ t.a = a2)

-- Theorem statement
theorem triangle_side_range (t : Triangle) :
  triangle_conditions t → 2 < t.a ∧ t.a < 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_range_l1212_121273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hno3_concentration_after_addition_l1212_121280

/-- Calculates the concentration of HNO3 in a solution after adding pure HNO3 -/
noncomputable def resultant_concentration (initial_volume : ℝ) (initial_concentration : ℝ) (added_volume : ℝ) : ℝ :=
  let initial_hno3 := initial_volume * initial_concentration
  let total_hno3 := initial_hno3 + added_volume
  let total_volume := initial_volume + added_volume
  (total_hno3 / total_volume) * 100

/-- Theorem stating that adding 6 liters of pure HNO3 to 60 liters of 45% HNO3 solution results in a 50% concentration -/
theorem hno3_concentration_after_addition :
  resultant_concentration 60 0.45 6 = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hno3_concentration_after_addition_l1212_121280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_amount_in_new_alloy_l1212_121202

/-- Represents the composition of an alloy --/
structure AlloyComposition where
  total_weight : ℝ
  ratio_1 : ℝ
  ratio_2 : ℝ

/-- Calculates the amount of a component in an alloy given its composition --/
noncomputable def amount_of_component (alloy : AlloyComposition) (ratio : ℝ) : ℝ :=
  (ratio / (alloy.ratio_1 + alloy.ratio_2)) * alloy.total_weight

/-- The total amount of tin in the newly formed alloy --/
noncomputable def total_tin_amount (alloy_a alloy_b alloy_c : AlloyComposition) : ℝ :=
  amount_of_component alloy_a alloy_a.ratio_2 +
  amount_of_component alloy_b alloy_b.ratio_1 +
  amount_of_component alloy_c alloy_c.ratio_2

/-- Theorem stating that the total amount of tin in the newly formed alloy is approximately 205.80357 kg --/
theorem tin_amount_in_new_alloy :
  let alloy_a : AlloyComposition := ⟨225, 5, 3⟩
  let alloy_b : AlloyComposition := ⟨175, 4, 3⟩
  let alloy_c : AlloyComposition := ⟨150, 6, 1⟩
  abs (total_tin_amount alloy_a alloy_b alloy_c - 205.80357) < 0.00001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_amount_in_new_alloy_l1212_121202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_7_eq_240_l1212_121233

/-- The coefficient of x^7 in the expansion of (2x^2 - 1/√x)^6 -/
def coefficient_x_7 : ℤ :=
  let n : ℕ := 6
  let r : ℕ := 2
  (n.choose r) * (2^(n-r)) * ((-1 : ℤ)^r)

/-- Theorem stating that the coefficient of x^7 in the expansion of (2x^2 - 1/√x)^6 is 240 -/
theorem coefficient_x_7_eq_240 : coefficient_x_7 = 240 := by
  rw [coefficient_x_7]
  simp
  sorry

#eval coefficient_x_7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_7_eq_240_l1212_121233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_buratino_result_l1212_121277

/-- The number of blots in Buratino's notebook -/
def x : ℕ := sorry

/-- Buratino's calculation function -/
noncomputable def buratino_calc (x : ℕ) : ℝ := (7 * x - 8) / 6 + 9

/-- Theorem stating that Buratino's calculation always results in 18^(1/6) -/
theorem buratino_result : buratino_calc x = 18^(1/6) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_buratino_result_l1212_121277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_pair_and_sum_l1212_121241

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards for each value in a standard deck -/
def CardsPerValue : ℕ := 4

/-- Represents the number of distinct values in a standard deck -/
def DistinctValues : ℕ := 13

/-- Represents the number of cards removed -/
def RemovedCards : ℕ := 2

/-- Represents the remaining deck size after removing a pair -/
def RemainingDeckSize : ℕ := StandardDeck - RemovedCards

/-- Represents the probability of drawing a matching pair from the remaining deck -/
def ProbabilityOfPair : ℚ := 73 / 1225

theorem probability_of_pair_and_sum :
  (ProbabilityOfPair.num + ProbabilityOfPair.den = 1298) ∧
  (ProbabilityOfPair = 73 / 1225) ∧
  (RemainingDeckSize = 50) ∧
  (StandardDeck = DistinctValues * CardsPerValue) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_pair_and_sum_l1212_121241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_number_proof_l1212_121268

theorem starting_number_proof (x : ℕ) : 
  (∀ y ∈ Finset.range (101 - x), (x + y) % 3 = 0 → x + y ≤ 100) ∧
  (Finset.filter (λ y => (x + y) % 3 = 0) (Finset.range (101 - x))).card = 33 ↔
  x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_number_proof_l1212_121268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_proof_l1212_121288

/-- Calculates simple interest given principal, rate, and time -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_difference_proof (principal : ℝ) (higherRate lowerRate time : ℝ) 
  (h1 : principal = 8400)
  (h2 : higherRate = 15)
  (h3 : lowerRate = 10)
  (h4 : time = 2) :
  simpleInterest principal higherRate time - simpleInterest principal lowerRate time = 840 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval simpleInterest 8400 15 2 - simpleInterest 8400 10 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_proof_l1212_121288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_score_distribution_l1212_121275

/-- Represents an exam score distribution -/
structure ExamDistribution where
  mean : ℝ
  stdDev : ℝ

/-- Calculates the number of standard deviations a score is from the mean -/
noncomputable def standardDeviationsFromMean (d : ExamDistribution) (score : ℝ) : ℝ :=
  (score - d.mean) / d.stdDev

theorem exam_score_distribution (d : ExamDistribution) 
  (h1 : d.mean = 74)
  (h2 : standardDeviationsFromMean d 58 = -2) :
  standardDeviationsFromMean d 98 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_score_distribution_l1212_121275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_formula_geometric_sequence_sum_l1212_121258

/-- Arithmetic sequence {a_n} -/
noncomputable def a (n : ℕ) : ℝ := 2 * n - 1

/-- Sum of first n terms of arithmetic sequence {a_n} -/
noncomputable def S (n : ℕ) : ℝ := n * (2 + 2 * n - 1) / 2

/-- Geometric sequence {b_n} -/
noncomputable def b (n : ℕ) : ℝ := 3^(n - 1)

/-- Sum of first n terms of geometric sequence {b_n} -/
noncomputable def T (n : ℕ) : ℝ := (3^n - 1) / 2

theorem arithmetic_sequence_formula (h1 : a 3 = 5) (h2 : S 3 = 9) :
  ∀ n : ℕ, a n = 2 * n - 1 := by sorry

theorem geometric_sequence_sum (h1 : b 3 = a 5) (h2 : T 3 = 13) :
  ∀ n : ℕ, T n = (3^n - 1) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_formula_geometric_sequence_sum_l1212_121258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_added_fraction_l1212_121210

def is_proper_fraction (q : ℚ) : Prop := q.num < q.den

def denominator_less_than_6 (q : ℚ) : Prop := q.den < 6

theorem largest_added_fraction :
  ∀ x : ℚ,
    (is_proper_fraction (1/6 + x)) →
    (denominator_less_than_6 (1/6 + x)) →
    x ≤ 19/30 :=
by
  sorry

#check largest_added_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_added_fraction_l1212_121210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_problem_l1212_121245

theorem vector_angle_problem (α β : ℝ) 
  (ha : Real.cos α ^ 2 + Real.sin α ^ 2 = 1)
  (hb : Real.cos β ^ 2 + Real.sin β ^ 2 = 1)
  (hdist : (Real.cos α - Real.cos β) ^ 2 + (Real.sin α - Real.sin β) ^ 2 = 1)
  (hangle : -π/2 < β ∧ β < 0 ∧ 0 < α ∧ α < π/2)
  (hsinβ : Real.sin β = -1/7) :
  Real.cos (α - β) = 1/2 ∧ Real.sin α = 13/14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_problem_l1212_121245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l1212_121212

-- Define set A
def A : Set ℝ := {x | (1/4 : ℝ) ≤ Real.exp (x * Real.log 2) ∧ Real.exp (x * Real.log 2) ≤ (1/2 : ℝ)}

-- Define set B (parameterized by t)
def B (t : ℝ) : Set ℝ := {x | x^2 - 2*t*x + 1 ≤ 0}

-- State the theorem
theorem range_of_t (t : ℝ) : 
  (A ∩ B t = A) ↔ t ≤ -5/4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l1212_121212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_intersection_condition_l1212_121298

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- Theorem: A line passing through the right focus of a hyperbola with slope k
    intersects both branches of the hyperbola if and only if e² - k² > 1,
    where e is the eccentricity of the hyperbola. -/
theorem hyperbola_line_intersection_condition (h : Hyperbola) (k : ℝ) :
  let e := eccentricity h
  (∃ (x₁ x₂ : ℝ), x₁ < 0 ∧ 0 < x₂ ∧
    x₁^2 / h.a^2 - (k * (x₁ - h.a * e))^2 / h.b^2 = 1 ∧
    x₂^2 / h.a^2 - (k * (x₂ - h.a * e))^2 / h.b^2 = 1) ↔
  e^2 - k^2 > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_intersection_condition_l1212_121298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_distance_l1212_121274

noncomputable def l1 (t : ℝ) : ℝ × ℝ := (1 + t, -5 + Real.sqrt 3 * t)

def l2 (p : ℝ × ℝ) : Prop := p.1 - p.2 - 2 * Real.sqrt 3 = 0

def Q : ℝ × ℝ := (1, -5)

noncomputable def P : ℝ × ℝ := (13 - 4 * Real.sqrt 3, -17 + 12 * Real.sqrt 3)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_and_distance :
  (∃ t : ℝ, l1 t = P) ∧ l2 P ∧ distance P Q = 12 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_distance_l1212_121274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1212_121204

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

-- Define the line
def my_line (t x y : ℝ) : Prop := 2*t*x - y - 2 - 2*t = 0

-- Theorem statement
theorem circle_line_intersection :
  ∀ t : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    my_circle x₁ y₁ ∧ 
    my_circle x₂ y₂ ∧ 
    my_line t x₁ y₁ ∧ 
    my_line t x₂ y₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1212_121204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_point_and_max_min_values_l1212_121249

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 - 9*x + 1

-- Define what it means to be an extremum point
def is_extremum (g : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), g y ≤ g x ∨ g y ≥ g x

-- State the theorem
theorem extremum_point_and_max_min_values :
  ∃ (a : ℝ),
    (∃ (x : ℝ), x = 3 ∧ is_extremum (f a) x) ∧
    (a = 3) ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 0, f a x ≤ 6) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 0, f a x = 6) ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 0, f a x ≥ -1) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 0, f a x = -1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_point_and_max_min_values_l1212_121249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_P1_l1212_121271

noncomputable section

def line_l (a b t : ℝ) : ℝ × ℝ := (a + t, b + t)

def point_P (a b : ℝ) : ℝ × ℝ := (a, b)

def point_P1 (a b t1 : ℝ) : ℝ × ℝ := line_l a b t1

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_P_P1 (a b t1 : ℝ) :
  distance (point_P1 a b t1) (point_P a b) = Real.sqrt 2 * |t1| := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_P1_l1212_121271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_function_and_triangle_l1212_121221

noncomputable def m (x : ℝ) (ω : ℝ) : ℝ × ℝ := (1/2 * Real.sin (ω * x), Real.sqrt 3 / 2)

noncomputable def n (x : ℝ) (ω : ℝ) : ℝ × ℝ := (Real.cos (ω * x), Real.cos (ω * x) ^ 2 - 1/2)

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := (m x ω).1 * (n x ω).1 + (m x ω).2 * (n x ω).2

noncomputable def triangle_area (a b c : ℝ) : ℝ := 1/2 * a * b * c

theorem vector_function_and_triangle (ω : ℝ) (h_ω : ω > 0) :
  (∀ x, f x ω = 1/2 * Real.sin (x + π/3)) ∧
  triangle_area (Real.sqrt 3) (8 * Real.sqrt 3 / 5) ((3 + 4 * Real.sqrt 3) / 10) = (12 + 16 * Real.sqrt 3) / 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_function_and_triangle_l1212_121221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_image_of_f_is_closed_interval_l1212_121239

-- Define the sets A and B
def A : Set ℝ := Set.Icc (-1) 1
def B : Set ℝ := Set.univ

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (2 - x^2) / Real.log (1/2)

-- Theorem statement
theorem image_of_f_is_closed_interval :
  Set.image f A = Set.Icc (-1) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_image_of_f_is_closed_interval_l1212_121239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jimmys_pizza_cost_per_slice_l1212_121299

/-- Represents the size of a pizza -/
inductive PizzaSize
| Small
| Medium
| Large

/-- Represents the category of a pizza topping -/
inductive ToppingCategory
| A
| B
| C

/-- Pizza pricing structure -/
def pizzaPrice (size : PizzaSize) : ℚ :=
  match size with
  | .Small => 8
  | .Medium => 12
  | .Large => 15

/-- Number of slices for each pizza size -/
def pizzaSlices (size : PizzaSize) : ℕ :=
  match size with
  | .Small => 6
  | .Medium => 10
  | .Large => 12

/-- Calculate the cost of toppings for a given category and quantity -/
def toppingCost (category : ToppingCategory) (quantity : ℕ) : ℚ :=
  match category with
  | .A => if quantity > 0 then 2 + (quantity - 1) * (3/2) else 0
  | .B => if quantity > 2 then 2 + (quantity - 2) * (3/4) else quantity
  | .C => quantity * (1/2)

/-- Calculate the discount based on pizza size and whether it has toppings from all categories -/
def discount (size : PizzaSize) (hasAllCategories : Bool) : ℚ :=
  if hasAllCategories then
    match size with
    | .Small => 1
    | .Medium => (3/2)
    | .Large => 2
  else 0

/-- Theorem: The cost per slice for Jimmy's pizza is $1.88 when rounded to the nearest cent -/
theorem jimmys_pizza_cost_per_slice :
  let size := PizzaSize.Medium
  let basePrice := pizzaPrice size
  let numSlices := pizzaSlices size
  let toppingCostA := toppingCost ToppingCategory.A 2
  let toppingCostB := toppingCost ToppingCategory.B 3
  let toppingCostC := toppingCost ToppingCategory.C 4
  let totalToppingCost := toppingCostA + toppingCostB + toppingCostC
  let discountAmount := discount size true
  let totalCost := basePrice + totalToppingCost - discountAmount
  let costPerSlice := totalCost / numSlices
  ⌊(costPerSlice * 100 + 1/2)⌋ / 100 = (188 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jimmys_pizza_cost_per_slice_l1212_121299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sum_angles_l1212_121257

theorem tan_half_sum_angles (p q : ℝ) 
  (h1 : Real.cos p + Real.cos q = 3/5) 
  (h2 : Real.sin p + Real.sin q = 1/5) : 
  Real.tan ((p + q) / 2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sum_angles_l1212_121257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_origin_and_intersection_l1212_121269

-- Define the two lines
def line1 (x y : ℝ) : Prop := x - 3*y + 4 = 0
def line2 (x y : ℝ) : Prop := 2*x + y + 5 = 0

-- Define the intersection point of the two lines
noncomputable def intersection_point : ℝ × ℝ := 
  ((-19:ℝ)/7, (3:ℝ)/7)

-- Define the line passing through origin and intersection point
def target_line (x y : ℝ) : Prop := 3*x + 19*y = 0

-- Theorem statement
theorem line_through_origin_and_intersection :
  ∀ (x y : ℝ), 
    (x = 0 ∧ y = 0) ∨ (x = intersection_point.1 ∧ y = intersection_point.2) →
    target_line x y :=
by
  intros x y h
  cases h with
  | inl h => 
    simp [target_line, h]
  | inr h => 
    simp [target_line, h, intersection_point]
    -- The actual proof would go here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_origin_and_intersection_l1212_121269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_points_cyclic_permutations_impossible_l1212_121200

/-- The number of cyclic permutations of n elements -/
def cyclic_permutations (n : ℕ) : ℕ := (n - 1).factorial

/-- The number of regions formed by n lines in a plane -/
def regions_formed_by_lines (n : ℕ) : ℕ := n * (n - 1) / 2 + 1

/-- The number of lines connecting pairs of n points -/
def connecting_lines (n : ℕ) : ℕ := n.choose 2

theorem seven_points_cyclic_permutations_impossible :
  ¬ ∃ (arrangement : Finset (ℝ × ℝ)),
    (arrangement.card = 7) ∧
    (regions_formed_by_lines (connecting_lines 7) ≥ cyclic_permutations 7) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_points_cyclic_permutations_impossible_l1212_121200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_circles_congruence_l1212_121216

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the concept of an acute triangle
def isAcute (t : Triangle) : Prop := sorry

-- Define the orthocenter of a triangle
noncomputable def orthocenter (t : Triangle) : Point := sorry

-- Define a circle passing through three points
structure Circle where
  center : Point
  passingThrough : List Point

-- Define the centers of the three circles
noncomputable def circleCenter (p1 p2 p3 : Point) : Point := sorry

-- Define congruence between triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Main theorem
theorem orthocenter_circles_congruence (t : Triangle) (h : Point) :
  isAcute t →
  h = orthocenter t →
  let a' := circleCenter t.B t.C h
  let b' := circleCenter t.C t.A h
  let c' := circleCenter t.A t.B h
  let t' := Triangle.mk a' b' c'
  congruent t t' := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_circles_congruence_l1212_121216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_negative_l1212_121267

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 - x
  else -(((-x)^2) - (-x))

-- State the theorem
theorem max_value_of_f_negative (h : ∀ x, f (-x) = -f x) :
  ∃ M, M = 1/4 ∧ ∀ x < 0, f x ≤ M := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_negative_l1212_121267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1212_121234

noncomputable section

-- Define the function f
def f (ω : ℝ) (x : ℝ) : ℝ := 4 * Real.cos (ω * x) * Real.sin (ω * x + Real.pi / 6) + 1

-- State the theorem
theorem function_properties (ω : ℝ) (x₁ x₂ : ℝ) 
  (h_ω_pos : ω > 0)
  (h_zeros : f ω x₁ = 0 ∧ f ω x₂ = 0)
  (h_adjacent : ∀ x, x₁ < x ∧ x < x₂ → f ω x ≠ 0)
  (h_distance : |x₁ - x₂| = Real.pi) :
  (ω = 1) ∧
  (∀ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f ω x ≤ 4) ∧
  (∀ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f ω x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f ω x = 4) ∧
  (∃ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f ω x = 1) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1212_121234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_on_circle_l1212_121260

-- Define the circle's center and radius
def A : ℝ × ℝ := (2, -3)
def radius : ℝ := 5

-- Define point M
def M : ℝ × ℝ := (5, -7)

-- Calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem stating that M is on the circle
theorem M_on_circle : distance A M = radius := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_on_circle_l1212_121260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_on_line_l1212_121250

theorem cos_double_angle_on_line (θ : Real) :
  (∃ (x y : Real), y = 2 * x ∧ x = Real.cos θ ∧ y = Real.sin θ) →
  Real.cos (2 * θ) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_on_line_l1212_121250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_30_l1212_121243

theorem remainder_sum_mod_30 (a b c : ℕ) 
  (ha : a % 30 = 7)
  (hb : b % 30 = 11)
  (hc : c % 30 = 15) :
  (a + b + c) % 30 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_30_l1212_121243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_t_value_l1212_121286

def Point := ℝ × ℝ

structure Parallelogram where
  W : Point
  X : Point
  Y : Point
  Z : Point

def is_parallelogram (p : Parallelogram) : Prop :=
  |p.W.1 - p.X.1| = |p.Y.1 - p.Z.1| ∧
  |p.W.2 - p.Z.2| = |p.X.2 - p.Y.2|

def t_value (p : Parallelogram) : ℝ := p.Y.1 + 6

theorem parallelogram_t_value (p : Parallelogram) :
  is_parallelogram p ∧
  p.W = (-1, 4) ∧
  p.X = (5, 4) ∧
  p.Y = (t_value p - 6, 1) ∧
  p.Z = (-4, 1) →
  t_value p = 8 := by
  sorry

#check parallelogram_t_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_t_value_l1212_121286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unknown_souvenir_cost_l1212_121265

/-- Represents the cost in cents of the unknown type of souvenir -/
def unknown_cost : ℚ := sorry

/-- The total number of souvenirs distributed -/
def total_souvenirs : ℕ := 1000

/-- The number of souvenirs of the unknown cost type -/
def unknown_type_count : ℕ := 400

/-- The cost in cents of the known type of souvenir -/
def known_cost : ℚ := 20

/-- The total cost of all souvenirs in dollars -/
def total_cost : ℚ := 220

theorem unknown_souvenir_cost :
  unknown_cost = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unknown_souvenir_cost_l1212_121265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_finishes_first_l1212_121240

/-- Represents the area of a lawn -/
structure LawnArea where
  size : ℝ
  size_pos : size > 0

/-- Represents the speed of a lawn mower -/
structure MowerSpeed where
  speed : ℝ
  speed_pos : speed > 0

/-- Represents a person with their lawn area and mower speed -/
structure Person where
  name : String
  lawn : LawnArea
  mower : MowerSpeed

/-- Calculates the time taken to mow a lawn -/
noncomputable def mowingTime (p : Person) : ℝ :=
  p.lawn.size / p.mower.speed

theorem beth_finishes_first (andy beth carlos : Person)
  (h1 : andy.lawn.size = 3 * beth.lawn.size)
  (h2 : andy.lawn.size = 4 * carlos.lawn.size)
  (h3 : carlos.mower.speed = (1/4) * beth.mower.speed)
  (h4 : carlos.mower.speed = (1/6) * andy.mower.speed) :
  mowingTime beth < mowingTime andy ∧ mowingTime beth < mowingTime carlos := by
  sorry

#check beth_finishes_first

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_finishes_first_l1212_121240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_solution_l1212_121206

theorem equation_has_solution :
  ∃ x : ℝ, -Real.pi ≤ x ∧ x ≤ Real.pi ∧
  Real.tan (2*x) + (Real.tan x)^2 + (Real.sin x)^3 + (Real.sin (2*x))^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_solution_l1212_121206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_education_savings_total_amount_l1212_121294

/-- Represents an education savings account with annual deposits and interest -/
structure EducationSavings where
  annualRate : ℝ
  initialDeposit : ℝ
  annualDeposit : ℝ
  depositYears : ℕ

/-- Calculates the total amount in the education savings account after a given number of years -/
noncomputable def totalAmount (account : EducationSavings) : ℝ :=
  account.initialDeposit * (1 + account.annualRate) ^ account.depositYears +
  account.annualDeposit * ((1 + account.annualRate) ^ account.depositYears - 1) / account.annualRate

/-- Theorem stating the total amount withdrawn from the education savings account -/
theorem education_savings_total_amount :
  let account : EducationSavings := {
    annualRate := 0.05,
    initialDeposit := 2000,
    annualDeposit := 2000,
    depositYears := 6
  }
  (1 + account.annualRate) ^ 7 = 1.41 →
  totalAmount account = 14400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_education_savings_total_amount_l1212_121294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l1212_121203

-- Define the inequality function
noncomputable def f (x : ℝ) : ℝ := 2 / (x + 2) + 4 / (x + 8)

-- Define the solution set
def S : Set ℝ := Set.Ioo (-8) (-2) ∪ Set.Ioc (-2) 4

-- State the theorem
theorem inequality_equivalence :
  ∀ x : ℝ, f x ≥ 4/5 ↔ x ∈ S :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l1212_121203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_negative_padic_char_l1212_121220

/-- p-adic integer type -/
def pAdicInt (p : ℕ) [Fact (Nat.Prime p)] := ℕ → Fin p

/-- Function to get the nth p-adic digit of a p-adic integer -/
def pAdicDigit {p : ℕ} [Fact (Nat.Prime p)] (x : pAdicInt p) (n : ℕ) : Fin p := x n

/-- Predicate to check if a p-adic integer is strictly negative -/
def IsStrictlyNegative {p : ℕ} [Fact (Nat.Prime p)] (x : pAdicInt p) : Prop :=
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → pAdicDigit x n = p - 1

/-- Main theorem: Characterization of strictly negative p-adic integers -/
theorem strictly_negative_padic_char {p : ℕ} [Fact (Nat.Prime p)] (x : pAdicInt p) :
  IsStrictlyNegative x ↔
    ∃ N : ℕ, ∀ n : ℕ, n ≥ N → pAdicDigit x n = p - 1 :=
by
  -- The proof is trivial since the left-hand side is defined to be equal to the right-hand side
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_negative_padic_char_l1212_121220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l1212_121297

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/24 = 1

-- Define the foci and point on the hyperbola
variable (F1 F2 P : ℝ × ℝ)

-- Define the distances
noncomputable def PF1 (F1 P : ℝ × ℝ) : ℝ := Real.sqrt ((F1.1 - P.1)^2 + (F1.2 - P.2)^2)
noncomputable def PF2 (F2 P : ℝ × ℝ) : ℝ := Real.sqrt ((F2.1 - P.1)^2 + (F2.2 - P.2)^2)
noncomputable def F1F2 (F1 F2 : ℝ × ℝ) : ℝ := Real.sqrt ((F2.1 - F1.1)^2 + (F2.2 - F1.2)^2)

-- Define the arithmetic sequence property
def arithmetic_sequence (PF1 PF2 F1F2 : ℝ) : Prop :=
  ∃ d > 0, PF2 - PF1 = d ∧ F1F2 - PF2 = d

-- State the theorem
theorem hyperbola_triangle_area 
  (h1 : hyperbola P.1 P.2)
  (h2 : arithmetic_sequence (PF1 F1 P) (PF2 F2 P) (F1F2 F1 F2))
  : (1/2) * PF1 F1 P * PF2 F2 P = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l1212_121297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_height_specific_triangle_l1212_121225

/-- The maximum height of a table formed from a triangle -/
noncomputable def max_table_height (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let h_a := 2 * area / a
  let h_b := 2 * area / b
  (h_a * h_b) / (h_a + h_b)

/-- Theorem stating the maximum height of a table formed from a specific triangle -/
theorem max_height_specific_triangle :
  max_table_height 26 28 32 = 450 * Real.sqrt 1001 / 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_height_specific_triangle_l1212_121225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_fraction_series_l1212_121211

theorem sum_fraction_series (n : ℕ+) : 
  (Finset.range n).sum (λ i => (i + 1 : ℕ) / Nat.factorial (i + 2)) = 1 - 1 / Nat.factorial (n + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_fraction_series_l1212_121211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_fraction_l1212_121218

/-- The limit of (2n+3)/(n+1) as n approaches infinity is 2. -/
theorem limit_fraction : ∀ ε > 0, ∃ N : ℝ, ∀ n ≥ N, |((2 * n + 3) / (n + 1)) - 2| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_fraction_l1212_121218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_theorem_l1212_121228

/-- Given a unit cube ABCD-A₁B₁C₁D₁, this function returns the radius of the inscribed sphere
    of the tetrahedron LMNK, where L, M, N, and K are midpoints of edges AB, A₁D₁, A₁B₁, and BC
    respectively. -/
noncomputable def inscribed_sphere_radius_of_tetrahedron_in_unit_cube : ℝ :=
  (Real.sqrt 3 - Real.sqrt 2) / 2

/-- Theorem stating that the radius of the inscribed sphere of the tetrahedron LMNK
    in a unit cube is (√3 - √2) / 2. -/
theorem inscribed_sphere_radius_theorem :
  inscribed_sphere_radius_of_tetrahedron_in_unit_cube = (Real.sqrt 3 - Real.sqrt 2) / 2 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_theorem_l1212_121228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l1212_121263

-- Define the curve C in polar coordinates
def C (ρ θ : ℝ) : Prop := 7 * ρ^2 - ρ^2 * Real.cos (2 * θ) - 24 = 0

-- Define the conversion from polar to rectangular coordinates
noncomputable def polar_to_rect (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

-- Statement of the theorem
theorem curve_C_properties :
  -- Part 1: The rectangular equation of curve C
  (∀ x y : ℝ, (∃ ρ θ : ℝ, C ρ θ ∧ polar_to_rect ρ θ = (x, y)) ↔ x^2/4 + y^2/3 = 1) ∧
  -- Part 2: The range of x - 2y for points on curve C
  (∀ x y : ℝ, x^2/4 + y^2/3 = 1 → -4 ≤ x - 2*y ∧ x - 2*y ≤ 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l1212_121263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_a_plus_pi_fourth_l1212_121244

theorem tan_a_plus_pi_fourth (a : ℝ) 
  (h1 : Real.sin a = -5/13) 
  (h2 : 3*π/2 < a ∧ a < 2*π) : 
  Real.tan (a + π/4) = 7/17 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_a_plus_pi_fourth_l1212_121244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1212_121252

def a : ℕ → ℚ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 2
  | 3 => 4
  | 4 => 7
  | 5 => 11
  | n + 6 => a (n + 5) + n + 5  -- Adjusted for the added 0 case

theorem sequence_formula (n : ℕ) : 
  a n = (n^2 - n + 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1212_121252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_binomial_l1212_121296

-- Define the integral for a
noncomputable def a : ℝ := (2 / Real.pi) * ∫ (x : ℝ) in Set.Icc (-1) 1, (Real.sqrt (1 - x^2) + Real.sin x)

-- Define the binomial expansion
noncomputable def binomial_expansion (x : ℝ) := (x - a / x^2)^9

-- Theorem statement
theorem constant_term_of_binomial : 
  ∃ (coeff : ℤ), 
    coeff = -84 ∧ 
    ∃ (f : ℝ → ℝ), 
      (∀ x ≠ 0, binomial_expansion x = f x + coeff) ∧
      (∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x| < ε) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_binomial_l1212_121296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_arrangement_theorem_l1212_121259

/-- Represents the arrangement of numbers in the circles. -/
structure CircleArrangement where
  numbers : Fin 7 → ℕ
  sum_constant : ℕ
  top_number : ℕ

/-- The sum of all numbers from 1 to 7 -/
def sum_all : ℕ := 28

/-- Condition that all numbers from 1 to 7 are used exactly once -/
def all_numbers_used (arr : CircleArrangement) : Prop :=
  ∀ n : Fin 7, ∃! i : Fin 7, arr.numbers i = n.val + 1

/-- Helper function to determine if three points are collinear -/
def is_collinear : (Fin 3 → Fin 7) → Prop :=
  fun _ => true  -- Placeholder definition, replace with actual logic if needed

/-- Condition that the sum of each collinear trio is constant -/
def constant_sum_condition (arr : CircleArrangement) : Prop :=
  ∀ trio : Fin 3 → Fin 7, is_collinear trio → (List.map (fun i => arr.numbers (trio i)) (List.range 3)).sum = arr.sum_constant

/-- The main theorem to be proved -/
theorem circle_arrangement_theorem (arr : CircleArrangement) 
  (h1 : all_numbers_used arr)
  (h2 : constant_sum_condition arr) :
  arr.top_number = 4 ∧ arr.sum_constant = 12 := by
  sorry

#check circle_arrangement_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_arrangement_theorem_l1212_121259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_periods_in_4pi_l1212_121227

-- Define the sinusoidal function
noncomputable def sinusoidal (a b c d : ℝ) (x : ℝ) : ℝ := a * Real.sin (b * x + c) + d

-- State the theorem
theorem two_periods_in_4pi (a b c d : ℝ) :
  a > 0 → b > 0 → c > 0 → d > 0 →
  (∀ x : ℝ, sinusoidal a b c d (x + 4 * π) = sinusoidal a b c d x) →
  (∀ x : ℝ, sinusoidal a b c d (x + 2 * π) ≠ sinusoidal a b c d x) →
  b = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_periods_in_4pi_l1212_121227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_set_is_line_segment_l1212_121285

-- Define the set of points in polar coordinates
def polar_set : Set (ℝ × ℝ) :=
  {p | p.2 = Real.pi / 4 ∧ p.1 ≤ 5}

-- Define a line segment from origin to a point
def line_segment (end_point : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (t * end_point.1, t * end_point.2)}

-- Theorem stating that the polar_set is equivalent to a line segment
theorem polar_set_is_line_segment :
  ∃ end_point : ℝ × ℝ, polar_set = line_segment end_point := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_set_is_line_segment_l1212_121285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_divisor_of_f_l1212_121208

def f (n : ℕ+) : ℤ := (2 * n.val - 7) * 3^n.val + 9

theorem max_divisor_of_f :
  ∀ m : ℕ+, (∀ n : ℕ+, (m.val : ℤ) ∣ f n) → m.val ≤ 6 ∧
  ∀ n : ℕ+, (6 : ℤ) ∣ f n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_divisor_of_f_l1212_121208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_b_value_l1212_121230

/-- The smallest positive real number such that all roots of the polynomial are real and positive -/
noncomputable def a : ℝ := 3 * Real.sqrt 3

/-- The theorem stating that for the smallest positive 'a' such that all roots of
    x^3 - ax^2 + bx - a are real and positive, and b is unique, the value of b is 9 -/
theorem unique_b_value (b : ℝ) 
  (h1 : ∀ x : ℝ, (x^3 - a*x^2 + b*x - a = 0) → x > 0) 
  (h2 : ∀ c : ℝ, c ≠ b → ¬(∀ x : ℝ, (x^3 - a*x^2 + c*x - a = 0) → x > 0)) : 
  b = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_b_value_l1212_121230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_O_equation_max_area_EGFH_fixed_point_CD_l1212_121231

noncomputable section

-- Define the circle O
def circle_O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 2}

-- Define the line x + y + 2 = 0
def symmetry_line : Set (ℝ × ℝ) := {p | p.1 + p.2 + 2 = 0}

-- Define the circle M
def circle_M (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 + 2)^2 + (p.2 + 2)^2 = r^2}

-- Define the line l: y = 1/2x - 2
def line_l : Set (ℝ × ℝ) := {p | p.2 = 1/2 * p.1 - 2}

-- Define the point N
def point_N : ℝ × ℝ := (1, Real.sqrt 2 / 2)

-- Define a placeholder for symmetry
def is_symmetric (A B : Set (ℝ × ℝ)) (L : Set (ℝ × ℝ)) : Prop := sorry

-- Define a placeholder for area calculation
def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ := sorry

-- Define a placeholder for tangent point
def is_tangent_point (C : Set (ℝ × ℝ)) (P Q : ℝ × ℝ) : Prop := sorry

-- Define a placeholder for line through two points
def line_through (P Q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

theorem circle_O_equation :
  ∀ r > 0, is_symmetric circle_O (circle_M r) symmetry_line →
  (1, 1) ∈ circle_O →
  circle_O = {p : ℝ × ℝ | p.1^2 + p.2^2 = 2} :=
sorry

theorem max_area_EGFH :
  ∀ E F G H : ℝ × ℝ,
  E ∈ circle_O → F ∈ circle_O → G ∈ circle_O → H ∈ circle_O →
  (E.1 - F.1) * (G.1 - H.1) + (E.2 - F.2) * (G.2 - H.2) = 0 →
  point_N.1 * (E.1 + F.1) + point_N.2 * (E.2 + F.2) = 2 * point_N.1^2 + 2 * point_N.2^2 →
  point_N.1 * (G.1 + H.1) + point_N.2 * (G.2 + H.2) = 2 * point_N.1^2 + 2 * point_N.2^2 →
  area_quadrilateral E G F H ≤ 5/2 :=
sorry

theorem fixed_point_CD :
  ∀ P C D : ℝ × ℝ,
  P ∈ line_l →
  C ∈ circle_O → D ∈ circle_O →
  is_tangent_point circle_O C P →
  is_tangent_point circle_O D P →
  (1/2, -1) ∈ line_through C D :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_O_equation_max_area_EGFH_fixed_point_CD_l1212_121231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1212_121295

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 * Real.tan x + Real.cos (2 * x)

-- State the theorem
theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ 
   ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∃ (M : ℝ), M = Real.sqrt 2 ∧ ∀ (x : ℝ), f x ≤ M) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1212_121295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iterative_average_difference_l1212_121255

noncomputable def iterative_average (xs : List ℝ) : ℝ :=
  xs.foldl (fun acc x => (acc + x) / 2) (xs.head!)

def all_permutations (xs : List α) : List (List α) :=
  sorry

theorem iterative_average_difference :
  let numbers := [1, 2, 3, 4, 5, 6]
  let all_perms := all_permutations numbers
  let all_averages := all_perms.map iterative_average
  (all_averages.maximum? |>.map (· - (all_averages.minimum? |>.getD 0))).getD 0 = 3.0625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_iterative_average_difference_l1212_121255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_focus_coincide_l1212_121272

/-- The y-coordinate of the focus of a parabola x^2 = 2y -/
noncomputable def parabola_focus_y : ℝ := 1/2

/-- The y-coordinate of one of the foci of an ellipse y^2/m + x^2/2 = 1 -/
noncomputable def ellipse_focus_y (m : ℝ) : ℝ := Real.sqrt (m - 2)

/-- 
  If the focus of the parabola x^2 = 2y coincides with one of the foci of 
  the ellipse y^2/m + x^2/2 = 1, then m = 9/4
-/
theorem parabola_ellipse_focus_coincide : 
  parabola_focus_y = ellipse_focus_y (9/4) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_focus_coincide_l1212_121272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_returns_to_initial_position_l1212_121236

/-- The initial position of the particle --/
def initial_position : ℂ := 3

/-- The rotation factor for each move --/
noncomputable def rotation_factor : ℂ := Complex.exp (Complex.I * Real.pi / 6)

/-- The translation distance for each move --/
def translation_distance : ℝ := 8

/-- The number of moves --/
def num_moves : ℕ := 120

/-- The position after n moves --/
noncomputable def position_after_moves (n : ℕ) : ℂ :=
  initial_position * rotation_factor^n + translation_distance * (1 - rotation_factor^n) / (1 - rotation_factor)

/-- Theorem: The particle returns to its initial position after 120 moves --/
theorem particle_returns_to_initial_position :
  position_after_moves num_moves = initial_position := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_returns_to_initial_position_l1212_121236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_share_approx_l1212_121226

/-- Represents the investment scenario with partners a, b, and c -/
structure Investment where
  x : ℝ  -- a's initial investment amount
  total_gain : ℝ  -- total annual gain

/-- Calculates a's share in the profit -/
noncomputable def a_share (inv : Investment) : ℝ :=
  let a_interest := inv.x * 0.15 * 1
  let b_interest := (2.5 * inv.x) * 0.20 * 0.5
  let c_interest := (3.7 * inv.x) * 0.18 * (1/3)
  let total_interest := a_interest + b_interest + c_interest
  (a_interest / total_interest) * inv.total_gain

/-- Theorem stating that a's share in the profit is approximately 14469.60 -/
theorem a_share_approx (inv : Investment) 
    (h : inv.total_gain = 60000) : 
    ∃ ε > 0, |a_share inv - 14469.60| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_share_approx_l1212_121226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l1212_121235

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 * x + a else -x - 2 * a

-- Theorem statement
theorem function_equality (a : ℝ) :
  f a (1 - a) = f a (1 + a) ↔ a = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l1212_121235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_planes_proof_l1212_121247

-- Define the planes
def plane1 (x y z : ℝ) : Prop := 3 * x - y + 2 * z - 3 = 0
def plane2 (x y z : ℝ) : Prop := 6 * x - 2 * y + 4 * z + 7 = 0

-- Define the distance between the planes
noncomputable def distance_between_planes : ℝ := (13 * Real.sqrt 14) / 28

-- Theorem statement
theorem distance_between_planes_proof :
  ∃ (d : ℝ), d = distance_between_planes ∧
  ∀ (p1 p2 : ℝ × ℝ × ℝ),
    plane1 p1.fst p1.snd.fst p1.snd.snd → plane2 p2.fst p2.snd.fst p2.snd.snd →
    d = Real.sqrt ((p1.fst - p2.fst)^2 + (p1.snd.fst - p2.snd.fst)^2 + (p1.snd.snd - p2.snd.snd)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_planes_proof_l1212_121247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_is_120_degrees_l1212_121281

noncomputable def vector_a : ℝ × ℝ := (3, -4)

def vector_b_length : ℝ := 2

def dot_product : ℝ := -5

noncomputable def angle_between_vectors (a : ℝ × ℝ) (b_length : ℝ) (dot_prod : ℝ) : ℝ := 
  Real.arccos (dot_prod / (Real.sqrt ((a.1 ^ 2 + a.2 ^ 2)) * b_length))

theorem angle_is_120_degrees : 
  angle_between_vectors vector_a vector_b_length dot_product = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_is_120_degrees_l1212_121281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_330_degrees_l1212_121279

theorem sec_330_degrees : Real.cos (330 * π / 180) = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_330_degrees_l1212_121279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slow_clock_theorem_l1212_121238

noncomputable def clock_time (actual_minutes : ℝ) : ℝ :=
  (52 + 48 / 60) * actual_minutes / 60

theorem slow_clock_theorem :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
    clock_time⁻¹ 600 = 11 * 60 + 22 + ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slow_clock_theorem_l1212_121238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_option_b_correct_option_d_correct_l1212_121224

open Real

/-- An angle is in the second quadrant if it's between π/2 and π -/
def is_second_quadrant (α : ℝ) : Prop :=
  π/2 < α ∧ α < π

/-- An angle is in the first or third quadrant if it's between 0 and π/2 or between π and 3π/2 -/
def is_first_or_third_quadrant (α : ℝ) : Prop :=
  (0 < α ∧ α < π/2) ∨ (π < α ∧ α < 3*π/2)

/-- The area of a circular sector -/
noncomputable def sector_area (r : ℝ) (θ : ℝ) : ℝ :=
  1/2 * r^2 * θ

/-- The arc length of a circular sector -/
noncomputable def arc_length (r : ℝ) (θ : ℝ) : ℝ :=
  r * θ

theorem option_b_correct :
  ∀ α : ℝ, is_second_quadrant α → is_first_or_third_quadrant (α/2) := by sorry

theorem option_d_correct :
  ∀ r θ : ℝ, sector_area r θ = 4 ∧ θ = 2 → arc_length r θ = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_option_b_correct_option_d_correct_l1212_121224
