import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_lines_l232_23254

/-- The distance between two parallel lines -/
noncomputable def distance_between_lines (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : ℝ :=
  |a₂ * (-c₁/a₁) + b₂ * 0 + c₂| / Real.sqrt (a₂^2 + b₂^2)

/-- Theorem: The distance between the lines 5x + 12y + 3 = 0 and 10x + 24y + 5 = 0 is 1/26 -/
theorem distance_specific_lines :
  distance_between_lines 5 12 3 10 24 5 = 1/26 := by
  -- Expand the definition of distance_between_lines
  unfold distance_between_lines
  -- Simplify the expression
  simp [Real.sqrt_eq_rpow]
  -- The rest of the proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_lines_l232_23254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_le_g_on_interval_l232_23227

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 1 / x
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := b * x

-- State the theorem
theorem f_le_g_on_interval (a b : ℝ) :
  a ∈ Set.Icc 0 1 →
  b ≥ Real.log 2 / 2 + 1 / 4 →
  ∀ x ∈ Set.Icc 2 (Real.exp 1), f a x ≤ g b x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_le_g_on_interval_l232_23227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_product_from_cosine_distances_l232_23261

noncomputable def cosine_similarity (x1 y1 x2 y2 : ℝ) : ℝ :=
  (x1 * x2 + y1 * y2) / (Real.sqrt (x1^2 + y1^2) * Real.sqrt (x2^2 + y2^2))

noncomputable def cosine_distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  1 - cosine_similarity x1 y1 x2 y2

theorem tan_product_from_cosine_distances (α β : ℝ) :
  cosine_distance (Real.cos α) (Real.sin α) (Real.cos β) (Real.sin β) = 1/3 →
  cosine_distance (Real.cos β) (Real.sin β) (Real.cos α) (-Real.sin α) = 1/2 →
  Real.tan α * Real.tan β = 1/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_product_from_cosine_distances_l232_23261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_proof_l232_23264

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x - 1)

-- Define the derivative of f
noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := (a * x - 1)⁻¹ * a

theorem a_value_proof (a : ℝ) :
  f_prime a 2 = 2 → a = 2/3 := by
  intro h
  -- The proof steps would go here
  sorry

#check a_value_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_proof_l232_23264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_2_4_l232_23239

-- Define a power function
noncomputable def power_function (α : ℝ) : ℝ → ℝ := λ x ↦ x ^ α

-- State the theorem
theorem power_function_through_point_2_4 :
  ∃ α : ℝ, power_function α 2 = 4 ∧ α = 2 := by
  -- Provide the value of α
  use 2
  
  constructor
  · -- Prove power_function 2 2 = 4
    simp [power_function]
    norm_num
  
  · -- Prove α = 2
    rfl

-- The proof is complete, so we don't need 'sorry'

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_2_4_l232_23239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_eq_1_iff_tan_x_eq_1_l232_23232

theorem sin_2x_eq_1_iff_tan_x_eq_1 : ∀ x : ℝ, Real.sin (2 * x) = 1 ↔ Real.tan x = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_eq_1_iff_tan_x_eq_1_l232_23232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_abs_x_minus_one_l232_23211

-- Define the integrand function
def f (x : ℝ) : ℝ := |x - 1|

-- State the theorem
theorem integral_abs_x_minus_one : ∫ x in Set.Icc 0 1, f x = (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_abs_x_minus_one_l232_23211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f3_f4_equal_l232_23206

-- Define the functions
noncomputable def f1 (x : ℝ) : ℝ := Real.sqrt (x^2)
def f2 (x : ℝ) : ℝ := x
def f3 (x : ℝ) : ℝ := 4 * x^4
noncomputable def f4 (x : ℝ) : ℝ := |x|
def f5 (x : ℝ) : ℝ := 3 * x^3
noncomputable def f6 (x : ℝ) : ℝ := x^2 / x

-- State the theorem
theorem only_f3_f4_equal :
  (∀ x, f1 x = f2 x) = False ∧
  (∀ x, f3 x = f4 x) = True ∧
  (∀ x, f4 x = f5 x) = False ∧
  (∀ x, f1 x = f6 x) = False :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f3_f4_equal_l232_23206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_difference_is_125_l232_23222

def roundToNearestMultipleOf5 (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

def joSum (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def kateSum (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (λ i => roundToNearestMultipleOf5 (i + 1))

theorem sum_difference_is_125 :
  kateSum 50 - joSum 50 = 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_difference_is_125_l232_23222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_rolling_distance_l232_23203

/-- Represents the horizontal distance traveled by a larger cylinder rolling over a smaller one -/
noncomputable def rolling_distance (r₁ r₂ : ℝ) : ℝ :=
  2 * Real.pi * r₁ - 2 * Real.pi * ((r₁ - r₂) / r₁) * r₁ + 2 * (r₁ + r₂) * Real.sqrt 3 / 2

/-- Theorem stating the distance traveled by a cylinder of radius 72 rolling over a cylinder of radius 24 -/
theorem cylinder_rolling_distance :
  rolling_distance 72 24 = 96 * Real.pi + 96 * Real.sqrt 3 := by
  sorry

#eval (96 + 96 + 3 : Nat)  -- Expected result: 195

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_rolling_distance_l232_23203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cos_relation_l232_23271

theorem tan_cos_relation (θ : Real) (r : Real) :
  Real.tan θ = -7/24 →
  π/2 < θ ∧ θ < π →
  100 * Real.cos θ = r →
  r = -96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cos_relation_l232_23271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_lines_with_integer_chord_l232_23255

/-- A line of the form mx - y - 4m + 1 = 0 -/
structure Line where
  m : ℝ

/-- The circle x^2 + y^2 = 25 -/
def Circle : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = 25}

/-- The chord length of the intersection between a line and the circle -/
noncomputable def chordLength (l : Line) : ℝ :=
  2 * Real.sqrt (25 - (|4 * l.m - 1| / Real.sqrt (1 + l.m^2))^2)

/-- Predicate for integer chord length -/
def hasIntegerChordLength (l : Line) : Prop :=
  ∃ n : ℕ, chordLength l = n

/-- The main theorem -/
theorem count_lines_with_integer_chord : 
  (∃ S : Finset Line, S.card = 9 ∧ (∀ l ∈ S, hasIntegerChordLength l) ∧
    (∀ l : Line, hasIntegerChordLength l → l ∈ S)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_lines_with_integer_chord_l232_23255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l232_23288

/-- The function f(x) defined in the problem -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (4^x - k * 2^(x+1) + 1) / (4^x + 2^x + 1)

/-- The theorem stating the range of k -/
theorem range_of_k : 
  (∀ x₁ x₂ x₃ : ℝ, ∃ a b c : ℝ, 
    a = f k x₁ ∧ 
    b = f k x₂ ∧ 
    c = f k x₃ ∧
    a + b > c ∧ b + c > a ∧ c + a > b) → 
  k ∈ Set.Icc (-2 : ℝ) (1/4 : ℝ) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l232_23288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_pi_third_l232_23286

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sin x - (1/2) * x

-- State the theorem
theorem f_max_at_pi_third :
  ∀ x ∈ Set.Icc 0 π, f x ≤ f (π/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_pi_third_l232_23286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotonicity_intervals_l232_23250

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + (1 - a) / x - 1

-- Define the derivative of f
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 1 / x - a - (1 - a) / (x^2)

-- Theorem for part I
theorem tangent_line_at_one (x : ℝ) :
  (f 1 1 = -2) ∧ (f_derivative 1 1 = 0) → 
  ∃ y, y = -2 ∧ (∀ h : ℝ, h ≠ 0 → (f 1 (1 + h) - f 1 1) / h - y = 0) := by
  sorry

-- Theorem for part II
theorem monotonicity_intervals :
  (∀ x ∈ Set.Ioo 1 2, f_derivative (1/3) x > 0) ∧
  (∀ x ∈ Set.Ioo 0 1, f_derivative (1/3) x < 0) ∧
  (∀ x ∈ Set.Ioi 2, f_derivative (1/3) x < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotonicity_intervals_l232_23250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_power_problem_l232_23246

theorem rational_power_problem (x y : ℚ) (h : (x - 3)^2 + |y + 4| = 0) : y^(3 : ℤ) = -64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_power_problem_l232_23246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_equal_perimeters_implies_equilateral_l232_23257

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The centroid of a triangle -/
noncomputable def centroid (t : Triangle) : Point :=
  { x := (t.A.x + t.B.x + t.C.x) / 3,
    y := (t.A.y + t.B.y + t.C.y) / 3 }

/-- The perimeter of a triangle -/
noncomputable def perimeter (t : Triangle) : ℝ :=
  sorry

/-- Check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop :=
  sorry

/-- The main theorem -/
theorem centroid_equal_perimeters_implies_equilateral (t : Triangle) :
  let M := centroid t
  let ABM := Triangle.mk t.A t.B M
  let BCM := Triangle.mk t.B t.C M
  let ACM := Triangle.mk t.A t.C M
  perimeter ABM = perimeter BCM ∧ perimeter BCM = perimeter ACM →
  isEquilateral t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_equal_perimeters_implies_equilateral_l232_23257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_increase_first_to_fifth_square_l232_23223

/-- Calculates the side length of the nth square in a sequence where each square's side is 120% longer than the previous -/
noncomputable def nthSquareSide (n : ℕ) : ℝ :=
  3 * (1.2 ^ (n - 1))

/-- Calculates the perimeter of a square given its side length -/
noncomputable def squarePerimeter (side : ℝ) : ℝ :=
  4 * side

/-- Calculates the percent increase between two values -/
noncomputable def percentIncrease (initial : ℝ) (final : ℝ) : ℝ :=
  (final - initial) / initial * 100

theorem perimeter_increase_first_to_fifth_square :
  let firstPerimeter := squarePerimeter (nthSquareSide 1)
  let fifthPerimeter := squarePerimeter (nthSquareSide 5)
  abs (percentIncrease firstPerimeter fifthPerimeter - 107.4) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_increase_first_to_fifth_square_l232_23223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l232_23204

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (|x| - x^2)

theorem domain_of_f : ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ -1 ≤ x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l232_23204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_angle_theorem_l232_23251

/-- Predicate to assert that P is an external point forming external angles u, v, w with ABC -/
def IsExternal (A B C P : EuclideanPlane) (u v w : ℝ) : Prop := sorry

/-- Predicate to assert that x is the angle inside ABC opposite to w -/
def AngleOpposite (A B C : EuclideanPlane) (w x : ℝ) : Prop := sorry

/-- Given a triangle ABC and an external point P, prove that the internal angle
    opposite to one external angle is equal to the sum of the other two external angles. -/
theorem external_angle_theorem (A B C P : EuclideanPlane) (u v w x : ℝ) : 
  IsExternal A B C P u v w → AngleOpposite A B C w x → x = u + v := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_angle_theorem_l232_23251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plan_c_most_cost_effective_l232_23219

/-- Represents a phone plan with its pricing details -/
structure PhonePlan where
  monthlyFee : ℚ
  costPerMinute : ℚ
  freeMinutes : ℕ
  discountThreshold : ℚ
  discountRate : ℚ

/-- Calculates the cost of a phone plan given the usage in minutes -/
def calculateCost (plan : PhonePlan) (minutes : ℕ) : ℚ :=
  let baseCost := plan.monthlyFee + (max 0 (minutes - plan.freeMinutes) : ℚ) * plan.costPerMinute
  if baseCost > plan.discountThreshold then
    baseCost * (1 - plan.discountRate)
  else
    baseCost

/-- Theorem stating that Plan C is the most cost-effective for John's usage -/
theorem plan_c_most_cost_effective (planA planB planC : PhonePlan) (john_usage : ℕ) : 
  planA.monthlyFee = 3 ∧ 
  planA.costPerMinute = 3/10 ∧ 
  planA.freeMinutes = 0 ∧ 
  planA.discountThreshold = 0 ∧ 
  planA.discountRate = 0 ∧
  planB.monthlyFee = 6 ∧ 
  planB.costPerMinute = 1/5 ∧ 
  planB.freeMinutes = 0 ∧ 
  planB.discountThreshold = 10 ∧ 
  planB.discountRate = 1/10 ∧
  planC.monthlyFee = 5 ∧ 
  planC.costPerMinute = 7/20 ∧ 
  planC.freeMinutes = 100 ∧ 
  planC.discountThreshold = 0 ∧ 
  planC.discountRate = 0 ∧
  john_usage = 45 →
  calculateCost planC john_usage ≤ calculateCost planA john_usage ∧
  calculateCost planC john_usage ≤ calculateCost planB john_usage := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plan_c_most_cost_effective_l232_23219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_circle_center_to_line_l232_23270

/-- The distance from a point (x₀, y₀) to a line ax + by + c = 0 -/
noncomputable def point_to_line_distance (x₀ y₀ a b c : ℝ) : ℝ :=
  (|a * x₀ + b * y₀ + c|) / Real.sqrt (a^2 + b^2)

/-- The line y = x + 1 in slope-intercept form -/
def line_equation (x y : ℝ) : Prop := y = x + 1

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, 0)

theorem distance_from_circle_center_to_line :
  point_to_line_distance (circle_center.1) (circle_center.2) (-1) 1 (-1) = 3 * Real.sqrt 2 / 2 := by
  sorry

#eval circle_center

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_circle_center_to_line_l232_23270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_monotone_increasing_l232_23208

theorem tan_monotone_increasing (a b : ℝ) (h1 : -π/2 < a) (h2 : a < b) (h3 : b < π/2) :
  Real.tan a < Real.tan b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_monotone_increasing_l232_23208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l232_23218

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the first curve
def curve1 (x y : ℝ) : Prop :=
  (x - floor x)^2 + y^2 = x - floor x

-- Define the second curve
def curve2 (x y : ℝ) : Prop :=
  y = 1/4 * x

-- Define an intersection point
def is_intersection (x y : ℝ) : Prop :=
  curve1 x y ∧ curve2 x y

-- State the theorem
theorem intersection_count :
  ∃ (S : Finset (ℝ × ℝ)), (∀ p ∈ S, is_intersection p.1 p.2) ∧ S.card = 10 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l232_23218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_find_heaviest_and_lightest_l232_23268

-- Define a type for coins
structure Coin where
  weight : ℕ

-- Define a type for the balance scale
inductive Comparison
  | LessThan : Coin → Coin → Comparison
  | GreaterThan : Coin → Coin → Comparison
  | Equal : Coin → Coin → Comparison

-- Define a function to compare two coins
def compareCoin (c1 c2 : Coin) : Comparison :=
  if c1.weight < c2.weight then Comparison.LessThan c1 c2
  else if c1.weight > c2.weight then Comparison.GreaterThan c1 c2
  else Comparison.Equal c1 c2

-- Define a function to find the heaviest and lightest coins
def findHeaviestAndLightest (coins : List Coin) : Option (Coin × Coin) :=
  sorry

-- Theorem statement
theorem can_find_heaviest_and_lightest :
  ∀ (coins : List Coin),
    coins.length = 10 →
    (∀ c1 c2, c1 ∈ coins → c2 ∈ coins → c1 ≠ c2 → c1.weight ≠ c2.weight) →
    ∃ (heaviest lightest : Coin) (comparisons : List (Coin × Coin)),
      comparisons.length ≤ 13 ∧
      heaviest ∈ coins ∧
      lightest ∈ coins ∧
      (∀ c, c ∈ coins → c.weight ≤ heaviest.weight ∧ c.weight ≥ lightest.weight) :=
by
  sorry

#check can_find_heaviest_and_lightest

end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_find_heaviest_and_lightest_l232_23268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_divisibility_l232_23294

theorem five_digit_divisibility (P Q R S T : Fin 5) : 
  P.val + 1 ≠ Q.val + 1 →
  P.val + 1 ≠ R.val + 1 →
  P.val + 1 ≠ S.val + 1 →
  P.val + 1 ≠ T.val + 1 →
  Q.val + 1 ≠ R.val + 1 →
  Q.val + 1 ≠ S.val + 1 →
  Q.val + 1 ≠ T.val + 1 →
  R.val + 1 ≠ S.val + 1 →
  R.val + 1 ≠ T.val + 1 →
  S.val + 1 ≠ T.val + 1 →
  (100 * (P.val + 1) + 10 * (Q.val + 1) + (R.val + 1)) % 5 = 0 →
  (100 * (Q.val + 1) + 10 * (R.val + 1) + (S.val + 1)) % 3 = 0 →
  (100 * (R.val + 1) + 10 * (S.val + 1) + (T.val + 1)) % 4 = 0 →
  P.val + 1 = 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_divisibility_l232_23294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l232_23237

-- Define the line l passing through (-1, 0) with slope k
def line (k : ℝ) (x : ℝ) : ℝ := k * (x + 1)

-- Define the circle x^2 + y^2 = 2x
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 2*x

-- Define the condition that the line intersects the circle at two points
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  circle_eq x₁ (line k x₁) ∧ 
  circle_eq x₂ (line k x₂)

-- State the theorem
theorem slope_range :
  ∀ k : ℝ, intersects_at_two_points k ↔ -Real.sqrt 3 / 3 < k ∧ k < Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l232_23237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_expansion_l232_23274

def trailingZeros (n : ℕ) : ℕ :=
  (Nat.digits 10 n).reverse.takeWhile (· = 0) |>.length

theorem zeros_in_expansion : 
  trailingZeros ((10^15 - 3)^2) = 15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_expansion_l232_23274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_reduction_approx_thirteen_percent_l232_23241

/-- The percentage reduction in the length of a square's side when it loses one quarter of its area -/
noncomputable def side_reduction_percentage : ℝ :=
  let initial_side := 1
  let initial_area := initial_side ^ 2
  let new_area := initial_area * 3 / 4
  let new_side := Real.sqrt new_area
  (initial_side - new_side) / initial_side * 100

/-- Theorem stating that the side reduction percentage is approximately 13% -/
theorem side_reduction_approx_thirteen_percent :
  13 < side_reduction_percentage ∧ side_reduction_percentage < 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_reduction_approx_thirteen_percent_l232_23241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_when_inequality_holds_l232_23233

-- Define g as an odd function
noncomputable def g : ℝ → ℝ := sorry

-- Define f as a piecewise function
noncomputable def f : ℝ → ℝ :=
  fun x => if x ≤ 0 then x^3 else g x

-- State the theorem
theorem range_of_x_when_inequality_holds :
  (∀ x < 0, g x = -Real.log (1 - x)) →  -- Condition for g when x < 0
  (∀ x, g (-x) = -g x) →                -- g is an odd function
  (∀ x, f (2 - x^2) > f x) →            -- Given inequality
  (∀ x, f (2 - x^2) > f x → -2 < x ∧ x < 1) -- Conclusion
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_when_inequality_holds_l232_23233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l232_23205

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The point A -/
def A : ℝ × ℝ := (-3, 0)

/-- The point B -/
def B : ℝ × ℝ := (-2, 5)

/-- The point on the y-axis -/
def P : ℝ × ℝ := (0, 2)

theorem equidistant_point : 
  distance A.1 A.2 P.1 P.2 = distance B.1 B.2 P.1 P.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l232_23205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l232_23260

/-- A function that determines the quadrant of an angle θ based on its sine and tangent values -/
noncomputable def determine_quadrant (θ : Real) : Nat :=
  if Real.sin θ > 0 ∧ Real.tan θ < 0 then 2 else 0

/-- Theorem stating that if sin θ > 0 and tan θ < 0, then θ is in the second quadrant -/
theorem angle_in_second_quadrant (θ : Real) (h1 : Real.sin θ > 0) (h2 : Real.tan θ < 0) :
  determine_quadrant θ = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l232_23260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_bounds_l232_23278

/-- The function g defined for positive real numbers x, y, and z -/
noncomputable def g (x y z : ℝ) : ℝ := x^2 / (x^2 + y^2) + y^2 / (y^2 + z^2) + z^2 / (z^2 + x^2)

/-- Theorem stating the bounds of g for positive real numbers x, y, and z -/
theorem g_bounds (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (3/2 : ℝ) ≤ g x y z ∧ g x y z ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_bounds_l232_23278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l232_23230

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin (x / 2) * Real.cos (x / 2) - 2 * (Real.cos (x / 2))^2

theorem f_properties :
  (f (π / 3) = 0) ∧
  (∀ k : ℤ, ∀ x : ℝ, x ∈ Set.Icc (2 * π / 3 + 2 * ↑k * π) (5 * π / 3 + 2 * ↑k * π) → 
    (∀ y : ℝ, y ∈ Set.Icc (2 * π / 3 + 2 * ↑k * π) (5 * π / 3 + 2 * ↑k * π) → x ≤ y → f x ≥ f y)) ∧
  (∀ k : ℤ, f (2 * π / 3 + ↑k * π) = f (2 * π / 3 - ↑k * π)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l232_23230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_volume_in_specific_prism_l232_23265

/-- Represents a right prism with a right-angled triangular base -/
structure RightPrism where
  ab : ℝ
  ac : ℝ
  aa1 : ℝ
  ab_perp_bc : ab ≤ ac -- Ensures AB is perpendicular to BC (AB ≤ AC in a right triangle)

/-- Calculates the maximum volume of a sphere that can fit inside the given right prism -/
noncomputable def maxSphereVolume (p : RightPrism) : ℝ :=
  (4 / 3) * Real.pi * (min ((p.ab + Real.sqrt (p.ac^2 - p.ab^2) - p.ac) / 2) (p.aa1 / 2))^3

/-- Theorem stating the maximum volume of a sphere in the specific prism -/
theorem max_sphere_volume_in_specific_prism :
  ∃ p : RightPrism, p.ab = 6 ∧ p.ac = 10 ∧ p.aa1 = 3 ∧ maxSphereVolume p = 9 * Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_volume_in_specific_prism_l232_23265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_is_two_sevenths_l232_23297

-- Define the amounts received by A, B, and C
variable (A B C : ℚ)

-- Define the conditions
def condition1 (A B C : ℚ) : Prop := A = (1/3) * (B + C)
def condition2 (A B C : ℚ) : Prop := A = B + 20
def condition3 (A B C : ℚ) : Prop := A + B + C = 720

-- Define the fraction we want to prove
def fraction (A B C : ℚ) : ℚ := B / (A + C)

-- Theorem statement
theorem fraction_is_two_sevenths 
  (h1 : condition1 A B C) 
  (h2 : condition2 A B C) 
  (h3 : condition3 A B C) : 
  fraction A B C = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_is_two_sevenths_l232_23297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_and_l1_properties_l232_23217

/-- Definition of the line l with parameter m -/
def line_l (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (2 + m) * x + (1 - 2 * m) * y + 4 - 3 * m = 0

/-- The fixed point M -/
def point_M : ℝ × ℝ := (-1, -2)

/-- Definition of line l₁ -/
def line_l1 : ℝ → ℝ → Prop :=
  λ x y ↦ 2 * x + y + 4 = 0

/-- Main theorem -/
theorem line_l_and_l1_properties :
  (∀ m : ℝ, line_l m (point_M.1) (point_M.2)) ∧
  line_l1 (point_M.1) (point_M.2) ∧
  ∃ (a b : ℝ), 
    line_l1 a 0 ∧ 
    line_l1 0 b ∧ 
    point_M = ((a + 0) / 2, (0 + b) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_and_l1_properties_l232_23217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_year_markup_is_25_percent_l232_23210

/-- Calculates the New Year markup percentage given initial markup, February discount, and February profit -/
noncomputable def newYearMarkup (initialMarkup februaryDiscount februaryProfit : ℝ) : ℝ :=
  let initialPrice := 1 + initialMarkup
  let x := (1 + februaryProfit - initialPrice * (1 - februaryDiscount)) / (initialPrice * (1 - februaryDiscount))
  100 * x

/-- Theorem stating that given the specified markups and discount, the New Year markup is 25% -/
theorem new_year_markup_is_25_percent :
  newYearMarkup 0.20 0.25 0.125 = 25 := by
  sorry

-- Remove the #eval line as it's not computable
-- #eval newYearMarkup 0.20 0.25 0.125

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_year_markup_is_25_percent_l232_23210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_species_minimum_l232_23258

/-- Given a total number of birds and a condition on their arrangement,
    calculate the minimum number of species required. -/
def minimum_species (total_birds : ℕ) (even_between : Prop) : ℕ :=
  (total_birds + 1) / 2

/-- Helper function to represent the number of birds between two birds -/
def number_between {species : Type} (bird1 bird2 : species) : ℕ := 
  sorry

/-- The problem statement translated to a theorem -/
theorem bird_species_minimum :
  let total_birds : ℕ := 2021
  let even_between : Prop := ∀ (species : Type) (bird1 bird2 : species),
    ∃ (n : ℕ), number_between bird1 bird2 = 2 * n
  minimum_species total_birds even_between = 1011 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_species_minimum_l232_23258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rabbits_first_month_l232_23256

/-- Growth pattern for mice population -/
def mice_growth (n : ℕ) (initial : ℕ) : ℕ :=
  initial * 2^(n - 1)

/-- Growth pattern for rabbits population -/
def rabbits_growth (n : ℕ) (initial : ℕ) : ℕ :=
  match n with
  | 0 => initial  -- Adding case for 0
  | 1 => initial
  | 2 => initial
  | k + 3 => rabbits_growth (k + 2) initial + rabbits_growth (k + 1) initial

/-- The theorem stating the minimum number of rabbits in the first month -/
theorem min_rabbits_first_month :
  ∃ (initial : ℕ), initial > 0 ∧
  rabbits_growth 7 initial = mice_growth 7 2 + 1 ∧
  ∀ (k : ℕ), k > 0 ∧ k < initial →
    rabbits_growth 7 k ≠ mice_growth 7 2 + 1 :=
by
  -- The proof goes here
  sorry

#eval rabbits_growth 7 5  -- This should output 13
#eval mice_growth 7 2     -- This should output 128

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rabbits_first_month_l232_23256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_digit_divisibility_l232_23289

theorem nine_digit_divisibility (n : Nat) : n < 10 → (91 ∣ 12345*10000 + n*1000 + 789) ↔ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_digit_divisibility_l232_23289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l232_23269

theorem min_abs_difference (a b : ℕ) (h : a * b - 4 * a + 3 * b = 475) :
  ∃ (a' b' : ℕ), a' * b' - 4 * a' + 3 * b' = 475 ∧
  ∀ (x y : ℕ), x * y - 4 * x + 3 * y = 475 →
  (a' - b' : ℤ).natAbs ≤ (x - y : ℤ).natAbs ∧
  (a' - b' : ℤ).natAbs = 455 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l232_23269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_proof_l232_23245

theorem age_difference_proof (total_age : ℕ) (ratio_halima : ℕ) (ratio_beckham : ℕ) (ratio_michelle : ℕ) 
  (h1 : total_age = 126)
  (h2 : ratio_halima = 4)
  (h3 : ratio_beckham = 3)
  (h4 : ratio_michelle = 7) :
  (ratio_halima : ℚ) * (total_age : ℚ) / ((ratio_halima : ℚ) + (ratio_beckham : ℚ) + (ratio_michelle : ℚ)) -
  (ratio_beckham : ℚ) * (total_age : ℚ) / ((ratio_halima : ℚ) + (ratio_beckham : ℚ) + (ratio_michelle : ℚ)) = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_proof_l232_23245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2023_pi_6_l232_23272

noncomputable def f : ℕ → ℝ → ℝ
  | 0, x => 3 * Real.sin x
  | n + 1, x => 9 / (3 - f n x)

theorem f_2023_pi_6 : f 2023 (Real.pi / 6) = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2023_pi_6_l232_23272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_orthocenter_collinearity_l232_23234

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define an ellipse
structure Ellipse where
  foci : Point × Point
  constant_sum : ℝ

-- Define the orthocenter of a triangle
noncomputable def orthocenter (t : Triangle) : Point :=
  sorry

-- Define the points of intersection of two ellipses
noncomputable def intersection_points (e1 e2 : Ellipse) : Set Point :=
  sorry

-- Define a predicate for points being collinear
def collinear (points : Set Point) : Prop :=
  sorry

-- Define a predicate for foci being aligned with triangle altitudes
def foci_aligned_with_altitudes (t : Triangle) (e1 e2 : Ellipse) : Prop :=
  sorry

theorem ellipse_intersection_orthocenter_collinearity 
  (t : Triangle) (e1 e2 : Ellipse) 
  (h_align : foci_aligned_with_altitudes t e1 e2) :
  collinear (insert (orthocenter t) (intersection_points e1 e2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_orthocenter_collinearity_l232_23234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_third_side_length_l232_23275

/-- Given a triangle XYZ satisfying certain conditions, prove that the maximum length of the third side is √925 --/
theorem max_third_side_length (X Y Z : ℝ) (a b : ℝ) : 
  Real.cos (3 * X) + Real.cos (3 * Y) + Real.cos (3 * Z) = 1 →
  Real.sin (2 * X) + Real.sin (2 * Y) + Real.sin (2 * Z) = 1 →
  a = 15 ∧ b = 20 →
  ∃ c : ℝ, c ≤ Real.sqrt 925 ∧ 
  (c = Real.sqrt 925 → c^2 = a^2 + b^2 - 2*a*b*Real.cos X ∨ 
                       c^2 = a^2 + b^2 - 2*a*b*Real.cos Y ∨ 
                       c^2 = a^2 + b^2 - 2*a*b*Real.cos Z) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_third_side_length_l232_23275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l232_23280

/-- The eccentricity of a hyperbola given the conditions of the problem -/
noncomputable def hyperbola_eccentricity (a : ℝ) : ℝ :=
  Real.sqrt ((16 / a^2) + 1)

/-- The statement of the problem -/
theorem hyperbola_eccentricity_range :
  ∀ a : ℝ, a > 0 →
  (∀ x y : ℝ, y^2 = 8*x ∧ x^2/a^2 - y^2/16 = 1 →
    (4/a * Real.sqrt (4 - a^2) > 4)) →
  hyperbola_eccentricity a > 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l232_23280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_class_size_l232_23220

/-- Represents the number of students in each of the first four rows -/
def x : ℕ := 10

/-- Total number of students in the class -/
def total_students : ℕ := 4 * x + (x + 2)

/-- The exhibition requirement -/
def exhibition_requirement : Prop := total_students ≥ 50

/-- The statement to prove -/
theorem smallest_class_size :
  (∀ y : ℕ, y < x → ¬(4 * y + (y + 2) ≥ 50)) →
  exhibition_requirement →
  total_students = 52 := by
  intro h1 h2
  -- The proof goes here
  sorry

#eval total_students

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_class_size_l232_23220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_to_icosahedron_l232_23207

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents an octahedron -/
structure Octahedron where
  vertices : Finset Point3D
  edges : Finset (Point3D × Point3D)

/-- Represents an icosahedron -/
structure Icosahedron where
  vertices : Finset Point3D

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Checks if an octahedron is regular -/
def Octahedron.isRegular (o : Octahedron) : Prop := sorry

/-- Selects division points on the edges of an octahedron -/
noncomputable def selectDivisionPoints (o : Octahedron) : Finset Point3D := sorry

/-- Checks if the division points are selected alternately closer and farther from vertices -/
def isAlternateSelection (o : Octahedron) (points : Finset Point3D) : Prop := sorry

/-- Checks if a set of points forms a regular icosahedron -/
def isRegularIcosahedron (points : Finset Point3D) : Prop := sorry

/-- Main theorem: The selected division points on a regular octahedron form a regular icosahedron -/
theorem octahedron_to_icosahedron (o : Octahedron) :
  o.isRegular →
  let points := selectDivisionPoints o
  isAlternateSelection o points →
  isRegularIcosahedron points := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_to_icosahedron_l232_23207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptote_l232_23221

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 4 - x^2 / 8 = 1

/-- The upper vertex of the hyperbola -/
def upper_vertex : ℝ × ℝ := (0, 2)

/-- One of the asymptotes of the hyperbola -/
def asymptote (x y : ℝ) : Prop :=
  x + Real.sqrt 2 * y = 0

/-- The distance formula from a point to a line -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

theorem distance_to_asymptote :
  let (x₀, y₀) := upper_vertex
  distance_point_to_line x₀ y₀ 1 (Real.sqrt 2) 0 = 2 * Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptote_l232_23221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plate_coverage_fraction_l232_23247

/-- The fraction of a larger circular plate's surface not covered by a smaller circular plate -/
theorem plate_coverage_fraction (d_small d_big : ℝ) (h_small : d_small = 10) (h_big : d_big = 12) :
  (π * (d_big / 2)^2 - π * (d_small / 2)^2) / (π * (d_big / 2)^2) = 11 / 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plate_coverage_fraction_l232_23247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l232_23213

/-- Circle C in Cartesian coordinates -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

/-- Line l in Cartesian coordinates -/
def line_l (x y : ℝ) : Prop := 4*x + 3*y - 11 = 0

/-- Point P -/
def point_P : ℝ × ℝ := (2, 1)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_distance_product :
  ∃ (A B : ℝ × ℝ),
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    A ≠ B ∧
    (distance point_P A) * (distance point_P B) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l232_23213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersections_l232_23273

-- Define the curves
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := ((2 + t) / 6, Real.sqrt t)
noncomputable def C₂ (s : ℝ) : ℝ × ℝ := (-(2 + s) / 6, -Real.sqrt s)
noncomputable def C₃ (θ : ℝ) : ℝ := 2 * Real.cos θ - Real.sin θ

-- State the theorem
theorem curve_intersections :
  -- 1. Cartesian equation of C₁
  (∀ x y, y ≥ 0 → (∃ t, C₁ t = (x, y)) ↔ y^2 = 6*x - 2) ∧
  -- 2. Intersection points of C₃ with C₁
  (∃ t θ, C₁ t = (1/2, 1) ∧ C₃ θ = 0) ∧
  (∃ t θ, C₁ t = (1, 2) ∧ C₃ θ = 0) ∧
  -- 3. Intersection points of C₃ with C₂
  (∃ s θ, C₂ s = (-1/2, -1) ∧ C₃ θ = 0) ∧
  (∃ s θ, C₂ s = (-1, -2) ∧ C₃ θ = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersections_l232_23273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_properties_l232_23242

-- Define the types for points and circles
variable {Point Circle : Type*}

-- Define the properties and relations
variable (is_convex_quadrilateral : Point → Point → Point → Point → Prop)
variable (has_incircle : Point → Point → Point → Point → Point → Prop)
variable (is_outside : Point → Point → Point → Point → Point → Prop)
variable (angle_eq : Point → Point → Point → Point → Point → Point → Prop)
variable (ray_within_angle : Point → Point → Point → Point → Point → Prop)
variable (is_incircle_of_triangle : Point → Point → Point → Point → Prop)
variable (share_common_tangent : Point → Point → Point → Point → Prop)
variable (are_concyclic : Point → Point → Point → Point → Prop)

-- State the theorem
theorem incircle_properties
  {A B C D O P : Point}
  {I₁ I₂ I₃ I₄ : Point}
  (h_convex : is_convex_quadrilateral A B C D)
  (h_incircle : has_incircle A B C D O)
  (h_outside : is_outside P A B C D)
  (h_angle_eq : angle_eq A P B C P D)
  (h_ray_within1 : ray_within_angle P B A P C)
  (h_ray_within2 : ray_within_angle P D A P C)
  (h_incircle1 : is_incircle_of_triangle I₁ A B P)
  (h_incircle2 : is_incircle_of_triangle I₂ B C P)
  (h_incircle3 : is_incircle_of_triangle I₃ C D P)
  (h_incircle4 : is_incircle_of_triangle I₄ D A P) :
  share_common_tangent I₁ I₂ I₃ I₄ ∧
  are_concyclic I₁ I₂ I₃ I₄ :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_properties_l232_23242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anya_walk_time_l232_23292

/-- Time spent on a one-way trip by bus -/
noncomputable def bus_time : ℝ := 0.5 / 2

/-- Time spent on a one-way trip by walking -/
noncomputable def walk_time : ℝ := 1.5 - bus_time

/-- Total time spent when walking both ways -/
noncomputable def total_walk_time : ℝ := 2 * walk_time

theorem anya_walk_time :
  (walk_time + bus_time = 1.5) ∧
  (bus_time + bus_time = 0.5) →
  total_walk_time = 2.5 := by
  sorry

#eval (2.5 : Float)  -- This will evaluate to 2.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anya_walk_time_l232_23292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pages_read_difference_l232_23240

theorem pages_read_difference (total_pages : ℕ) (fraction_read : ℚ) : 
  total_pages = 60 → 
  fraction_read = 2/3 → 
  (fraction_read * total_pages).floor - (total_pages - (fraction_read * total_pages).floor) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pages_read_difference_l232_23240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_proposition1_is_correct_l232_23216

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the area of a triangle
noncomputable def area (t : Triangle) : ℝ := (1/2) * t.b * t.c * Real.sin t.A

-- Define an arithmetic sequence
def arithmeticSequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Define a geometric sequence
def geometricSequence (a₁ r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

-- Proposition 1
def proposition1 (t : Triangle) : Prop :=
  area t = Real.sqrt 3 / 2 ∧ t.c = 2 ∧ t.A = Real.pi/3 → t.a = Real.sqrt 3

-- Proposition 2
def proposition2 (a₁ d : ℝ) : Prop :=
  a₁ = 2 ∧
  ∃ r, geometricSequence a₁ r 1 = arithmeticSequence a₁ d 1 ∧
       geometricSequence a₁ r 2 = arithmeticSequence a₁ d 3 ∧
       geometricSequence a₁ r 3 = arithmeticSequence a₁ d 4
  → d = -1/2

-- Proposition 3
def proposition3 (t : Triangle) : Prop :=
  Real.sin t.A ^ 2 < Real.sin t.B ^ 2 + Real.sin t.C ^ 2 →
  t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2

theorem only_proposition1_is_correct :
  (∃ t : Triangle, proposition1 t) ∧
  (¬∃ a₁ d : ℝ, proposition2 a₁ d) ∧
  (¬∃ t : Triangle, proposition3 t) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_proposition1_is_correct_l232_23216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_A_l232_23282

-- Define points in R²
def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (0, 15)
def C : ℝ × ℝ := (3, 9)

-- Define the line y = x
def line_y_eq_x (p : ℝ × ℝ) : Prop := p.2 = p.1

-- Define A' and B' as points on y = x
noncomputable def A' : ℝ × ℝ := (7.5, 7.5)
noncomputable def B' : ℝ × ℝ := (5, 5)

-- Define the property that AA' and BB' intersect at C
def intersect_at_C : Prop :=
  ∃ t₁ t₂ : ℝ, 0 < t₁ ∧ 0 < t₂ ∧
    C = (t₁ • A + (1 - t₁) • A') ∧
    C = (t₂ • B + (1 - t₂) • B')

-- Main theorem
theorem length_of_A'B'_is_5_sqrt_2 :
  line_y_eq_x A' ∧ line_y_eq_x B' ∧ intersect_at_C →
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 5 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_A_l232_23282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_positive_integers_l232_23236

def is_valid_set (M : Set ℕ) : Prop :=
  2018 ∈ M ∧
  (∀ m ∈ M, ∀ d : ℕ, d > 0 → d ∣ m → d ∈ M) ∧
  (∀ (k m : ℕ), k ∈ M → m ∈ M → 1 < k → k < m → k * m + 1 ∈ M)

theorem all_positive_integers (M : Set ℕ) :
  is_valid_set M → M = {n : ℕ | n > 0} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_positive_integers_l232_23236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expr_l232_23291

open Real

/-- The quadratic function f(x) = x^2 - x + k -/
def f (k : ℤ) (x : ℝ) : ℝ := x^2 - x + k

/-- The function g(x) = f(x) - 2 -/
def g (k : ℤ) (x : ℝ) : ℝ := f k x - 2

/-- Condition that g(x) has two distinct zeros in the interval (-1, 3/2) -/
def has_two_zeros (k : ℤ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ -1 < x₁ ∧ x₁ < 3/2 ∧ -1 < x₂ ∧ x₂ < 3/2 ∧ g k x₁ = 0 ∧ g k x₂ = 0

/-- The expression to be minimized -/
noncomputable def expr (k : ℤ) (x : ℝ) : ℝ := (f k x)^2 / (f k x) + 2 / (f k x)

theorem min_value_of_expr (k : ℤ) (h : has_two_zeros k) :
  ∃ x : ℝ, ∀ y : ℝ, expr k x ≤ expr k y ∧ expr k x = 81/28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expr_l232_23291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_y_pay_l232_23212

/-- The weekly pay of two employees -/
noncomputable def total_pay : ℚ := 560

/-- The ratio of X's pay to Y's pay -/
noncomputable def pay_ratio : ℚ := 12/10

/-- The weekly pay of employee Y -/
noncomputable def y_pay : ℚ := total_pay / (1 + pay_ratio)

theorem employee_y_pay : ⌊y_pay⌋ = 255 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_y_pay_l232_23212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cards_proof_l232_23209

/-- The number of possible face colors for the cards -/
def num_colors : ℕ := 2017

/-- A strategy for the helper to choose which card to leave face up -/
def helper_strategy (n : ℕ) := (Fin n → Fin num_colors) → Fin n

/-- A strategy for the magician to guess the color of a face-down card -/
def magician_strategy (n : ℕ) := Fin n → Fin num_colors

/-- Predicate to check if a given number of cards allows for a successful strategy -/
def successful_strategy (n : ℕ) : Prop :=
  ∃ (h : helper_strategy n) (m : magician_strategy n),
    ∀ (cards : Fin n → Fin num_colors),
      m (h cards) = cards (h cards)

/-- The minimum number of cards needed for a successful strategy -/
def min_cards : ℕ := 2018

theorem min_cards_proof :
  (successful_strategy min_cards) ∧
  (∀ k < min_cards, ¬(successful_strategy k)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cards_proof_l232_23209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_2_implies_a_geq_e_l232_23200

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x

-- State the theorem
theorem f_geq_2_implies_a_geq_e (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 (Real.exp 2), f a x ≥ 2) → a ≥ Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_2_implies_a_geq_e_l232_23200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_problem_l232_23287

noncomputable def conic_curve (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

noncomputable def point_A : ℝ × ℝ := (0, Real.sqrt 3)

def F₁ : ℝ × ℝ := (-1, 0)

def F₂ : ℝ × ℝ := (1, 0)

noncomputable def line_L (t : ℝ) : ℝ × ℝ := (-1 + (Real.sqrt 3 / 2) * t, (1 / 2) * t)

noncomputable def polar_equation_AF₂ (ρ θ : ℝ) : Prop :=
  Real.sqrt 3 * ρ * Real.cos θ + ρ * Real.sin θ - Real.sqrt 3 = 0

theorem conic_section_problem :
  (∀ t, ∃ θ, conic_curve θ = line_L t) ∧
  (∀ ρ θ, polar_equation_AF₂ ρ θ ↔ 
    ∃ k, k * (point_A.1 - F₂.1) = point_A.2 - F₂.2 ∧
         k * ρ * Real.cos θ = point_A.1 - F₂.1 ∧
         k * ρ * Real.sin θ = point_A.2 - F₂.2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_problem_l232_23287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l232_23290

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition relating sides and angles -/
def condition (t : Triangle) : Prop :=
  t.a / (t.b * t.c) + t.c / (t.a * t.b) - t.b / (t.a * t.c) = 1 / (t.a * Real.cos t.C + t.c * Real.cos t.A)

/-- The area of the triangle -/
noncomputable def area (t : Triangle) : ℝ := 3 * Real.sqrt 3 / 2

/-- The radius of the circumcircle -/
noncomputable def circumradius (t : Triangle) : ℝ := Real.sqrt 3

theorem triangle_properties (t : Triangle) 
  (h1 : condition t)
  (h2 : area t = 3 * Real.sqrt 3 / 2)
  (h3 : circumradius t = Real.sqrt 3)
  (h4 : t.c > t.a) :
  t.B = π / 3 ∧ t.c = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l232_23290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_to_nine_point_circle_l232_23235

-- Define a triangle
structure Triangle where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)

-- Define the orthocenter, circumcenter, and centroid of a triangle
noncomputable def orthocenter (t : Triangle) : EuclideanSpace ℝ (Fin 2) := sorry
noncomputable def circumcenter (t : Triangle) : EuclideanSpace ℝ (Fin 2) := sorry
noncomputable def centroid (t : Triangle) : EuclideanSpace ℝ (Fin 2) := sorry

-- Define a circle
structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

-- Define the circumcircle of a triangle
noncomputable def circumcircle (t : Triangle) : Circle := sorry

-- Define the nine-point circle of a triangle
noncomputable def ninePointCircle (t : Triangle) : Circle := sorry

-- Define a homothety
def homothety (center : EuclideanSpace ℝ (Fin 2)) (k : ℝ) (p : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := sorry

-- Define the image of a circle under a homothety
noncomputable def homotheticImage (h : EuclideanSpace ℝ (Fin 2) → EuclideanSpace ℝ (Fin 2)) (c : Circle) : Circle := sorry

-- The main theorem
theorem circumcircle_to_nine_point_circle (t : Triangle) :
  let H := orthocenter t
  let O := circumcenter t
  let M := centroid t
  let h := homothety H (1/2)
  homotheticImage h (circumcircle t) = ninePointCircle t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_to_nine_point_circle_l232_23235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_dice_distinct_numbers_probability_l232_23215

theorem seven_dice_distinct_numbers_probability (n : ℕ) (p : ℚ) : 
  n = 7 →  -- number of dice
  p = (n.factorial / 6^n : ℚ) →  -- probability of n distinct numbers from n dice rolls
  p = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_dice_distinct_numbers_probability_l232_23215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_108_l232_23293

theorem sum_of_factors_108 : (Finset.sum (Nat.divisors 108) id) = 280 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_108_l232_23293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_plus_pi_fourth_l232_23244

/-- An angle with vertex at the origin, initial side along the positive x-axis, 
    and terminal side passing through a point. -/
structure AngleAtOrigin where
  terminal_x : ℝ
  terminal_y : ℝ

/-- The tangent of an angle defined by an AngleAtOrigin -/
noncomputable def tan_angle (α : AngleAtOrigin) : ℝ :=
  α.terminal_y / α.terminal_x

theorem tan_double_plus_pi_fourth (α : AngleAtOrigin) 
  (h : α.terminal_x = 2 ∧ α.terminal_y = 1) : 
  Real.tan (2 * Real.arctan (tan_angle α) + π/4) = -7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_plus_pi_fourth_l232_23244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_85_l232_23281

theorem square_of_85 : 85^2 = 7225 := by
  -- Decomposition of 85 as 80 + 5
  have h1 : 85 = 80 + 5 := by norm_num
  
  -- The identity (a + b)^2 = a^2 + 2ab + b^2
  have h2 : ∀ a b : ℕ, (a + b)^2 = a^2 + 2*a*b + b^2 := by
    intros a b
    ring

  -- Apply the identity to 85^2
  calc 85^2 = (80 + 5)^2 := by rw [h1]
       _ = 80^2 + 2*80*5 + 5^2 := by rw [h2]
       _ = 6400 + 800 + 25 := by norm_num
       _ = 7225 := by norm_num

  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_85_l232_23281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l232_23229

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ
  directrix : ℝ → ℝ

/-- Point on a parabola -/
def on_parabola (C : Parabola) (point : ℝ × ℝ) : Prop :=
  point.2^2 = 2 * C.p * point.1

/-- Point on a line -/
def on_line (l : ℝ → ℝ) (point : ℝ × ℝ) : Prop :=
  point.2 = l point.1

/-- Equilateral triangle -/
def is_equilateral_triangle (A B F : ℝ × ℝ) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 
  (B.1 - F.1)^2 + (B.2 - F.2)^2 ∧
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = 
  (B.1 - F.1)^2 + (B.2 - F.2)^2

/-- Distance between two points -/
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

/-- Main theorem -/
theorem parabola_theorem (C : Parabola) (A B : ℝ × ℝ) :
  C.p > 0 →
  on_parabola C A →
  on_line C.directrix B →
  is_equilateral_triangle A B C.focus →
  distance A B = 4 →
  C.p = 2 ∧
  ∃ N : ℝ × ℝ, N.1 = 2 ∧ N.2 = 0 ∧
    ∀ Q R : ℝ × ℝ, on_parabola C Q → on_parabola C R →
    (∃ m : ℝ, Q.1 = m * Q.2 + N.1 ∧ R.1 = m * R.2 + N.1) →
    1 / (distance N Q)^2 + 1 / (distance N R)^2 = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l232_23229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sqrt7_343sqrt7_eq_4_l232_23225

-- Define the logarithm
noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem log_sqrt7_343sqrt7_eq_4 : 
  log_base (Real.sqrt 7) (343 * Real.sqrt 7) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sqrt7_343sqrt7_eq_4_l232_23225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_dots_is_75_l232_23252

/-- Represents a single die -/
structure Die where
  faces : Fin 6 → Nat
  opposite_sum : ∀ i : Fin 6, faces i + faces (5 - i) = 7

/-- Represents the figure formed by combining 7 dice -/
structure Figure where
  dice : Fin 7 → Die
  glued_faces : Fin 9 → Nat
  glued_faces_match : ∀ i : Fin 9, ∃ j k : Fin 7, ∃ f₁ f₂ : Fin 6, 
    (dice j).faces f₁ = (dice k).faces f₂ ∧ (dice j).faces f₁ = glued_faces i

/-- The total number of dots on the surface of the figure -/
def total_dots (fig : Figure) : Nat :=
  2 * (fig.glued_faces 0 + fig.glued_faces 1 + fig.glued_faces 2 + 
       fig.glued_faces 3 + fig.glued_faces 4 + fig.glued_faces 5 + 
       fig.glued_faces 6 + fig.glued_faces 7 + fig.glued_faces 8) + 
  3 * 7

theorem total_dots_is_75 (fig : Figure) : total_dots fig = 75 := by
  sorry

#check total_dots_is_75

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_dots_is_75_l232_23252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_almost_perfect_iff_t_eq_l232_23266

def s (n : ℕ) : ℕ := (Finset.sum (Nat.divisors n) id)

def mod (n k : ℕ) : ℕ := n % k

def t (n : ℕ) : ℕ := Finset.sum (Finset.range n) (λ k => mod n (k + 1))

theorem almost_perfect_iff_t_eq (n : ℕ+) :
  s n.val = 2 * n.val - 1 ↔ t n.val = t (n.val - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_almost_perfect_iff_t_eq_l232_23266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_condition_l232_23214

-- Define IsCircle as a predicate on sets of points in ℝ²
def IsCircle (S : Set (ℝ × ℝ)) : Prop := 
  ∃ (c : ℝ × ℝ) (r : ℝ), r > 0 ∧ S = {p : ℝ × ℝ | (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2}

theorem circle_condition (k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*x + 6*y + k = 0 → 
    IsCircle {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 + 6*p.2 + k = 0}) 
  → k < 10 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_condition_l232_23214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sharpened_off_length_is_54_l232_23259

/-- The total length of sharpened-off parts from three pencils -/
def total_sharpened_off_length (original_lengths sharpened_lengths : Fin 3 → ℕ) : ℕ :=
  (Finset.univ.sum fun i => original_lengths i - sharpened_lengths i)

/-- Theorem stating that the total sharpened-off length for the given pencils is 54 inches -/
theorem sharpened_off_length_is_54 :
  let original_lengths : Fin 3 → ℕ := ![31, 42, 25]
  let sharpened_lengths : Fin 3 → ℕ := ![14, 19, 11]
  total_sharpened_off_length original_lengths sharpened_lengths = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sharpened_off_length_is_54_l232_23259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pe_tangent_to_omega_l232_23298

-- Define the structure for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the structure for a circle
structure Circle where
  center : Point2D
  radius : ℝ

-- Define the isosceles trapezoid
def IsoscelesTrapezoid (A B C D : Point2D) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ D.x - C.x = k * (B.x - A.x) ∧ D.y - C.y = k * (B.y - A.y)

-- Define the midpoint
noncomputable def Midpoint (A C : Point2D) : Point2D :=
  ⟨(A.x + C.x) / 2, (A.y + C.y) / 2⟩

-- Define a point being on a circle
def PointOnCircle (P : Point2D) (C : Circle) : Prop :=
  (P.x - C.center.x)^2 + (P.y - C.center.y)^2 = C.radius^2

-- Define a line being tangent to a circle at a point
def TangentToCircle (P : Point2D) (C : Circle) : Prop :=
  PointOnCircle P C ∧
  ∀ R : Point2D, R ≠ P → (R.x - P.x) * (P.x - C.center.x) + (R.y - P.y) * (P.y - C.center.y) = 0 →
    ¬PointOnCircle R C

-- Main theorem
theorem pe_tangent_to_omega (A B C D E : Point2D) (ω Ω : Circle) (P : Point2D) :
  IsoscelesTrapezoid A B C D →
  E = Midpoint A C →
  PointOnCircle A ω ∧ PointOnCircle B ω ∧ PointOnCircle E ω →
  PointOnCircle C Ω ∧ PointOnCircle D Ω ∧ PointOnCircle E Ω →
  TangentToCircle P ω →
  TangentToCircle P Ω →
  TangentToCircle P Ω := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pe_tangent_to_omega_l232_23298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_is_25_l232_23285

/-- Represents the train problem with given conditions -/
structure TrainProblem where
  distance : ℝ  -- Distance between stations P and Q
  speed1 : ℝ    -- Speed of the first train
  time1 : ℝ     -- Travel time of the first train
  time2 : ℝ     -- Travel time of the second train

/-- The speed of the second train given the problem conditions -/
noncomputable def second_train_speed (p : TrainProblem) : ℝ :=
  (p.distance - p.speed1 * p.time1) / p.time2

/-- Theorem stating that the speed of the second train is 25 km/h -/
theorem second_train_speed_is_25 (p : TrainProblem) 
  (h1 : p.distance = 110)
  (h2 : p.speed1 = 20)
  (h3 : p.time1 = 3)
  (h4 : p.time2 = 2) : 
  second_train_speed p = 25 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_is_25_l232_23285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cell_phone_length_theorem_l232_23226

/-- Represents the dimensions of a rectangular cell phone. -/
structure CellPhone where
  width : ℝ
  circumference : ℝ

/-- Calculates the length of a rectangular cell phone given its width and circumference. -/
noncomputable def calculateLength (phone : CellPhone) : ℝ :=
  (phone.circumference - 2 * phone.width) / 2

/-- Theorem stating that a rectangular cell phone with width 9 cm and circumference 46 cm has a length of 14 cm. -/
theorem cell_phone_length_theorem (phone : CellPhone) 
  (h1 : phone.width = 9)
  (h2 : phone.circumference = 46) : 
  calculateLength phone = 14 := by
  sorry

#check cell_phone_length_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cell_phone_length_theorem_l232_23226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_3pi_minus_x_eq_2_implies_expression_eq_neg_3_l232_23201

theorem tan_3pi_minus_x_eq_2_implies_expression_eq_neg_3 (x : ℝ) :
  Real.tan (3 * Real.pi - x) = 2 →
  (2 * (Real.cos (x / 2))^2 - Real.sin x - 1) / (Real.sin x + Real.cos x) = -3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_3pi_minus_x_eq_2_implies_expression_eq_neg_3_l232_23201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_acute_angles_l232_23238

structure RightTriangle where
  angle1 : ℝ
  angle2 : ℝ
  is_right_angled : angle1 + angle2 = 90

theorem right_triangle_acute_angles (triangle : RightTriangle) 
  (h : triangle.angle1 = 37) : 
  triangle.angle2 = 53 := by
  have sum_eq_90 : triangle.angle1 + triangle.angle2 = 90 := triangle.is_right_angled
  rw [h] at sum_eq_90
  linarith

#check right_triangle_acute_angles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_acute_angles_l232_23238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_rate_calculation_l232_23224

/-- The rate per kg of grapes -/
def grapeRate : ℕ → Prop := sorry

/-- The total amount paid -/
def totalPaid : ℕ := sorry

/-- The amount of grapes purchased in kg -/
def grapesKg : ℕ := sorry

/-- The amount of mangoes purchased in kg -/
def mangoesKg : ℕ := sorry

/-- The rate per kg of mangoes -/
def mangoRate : ℕ := sorry

theorem grape_rate_calculation (h1 : grapesKg = 8)
                               (h2 : mangoesKg = 9)
                               (h3 : mangoRate = 55)
                               (h4 : totalPaid = 1135)
                               (h5 : totalPaid = grapesKg * g + mangoesKg * mangoRate)
                               (g : ℕ) :
  grapeRate g ↔ g = 80 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_rate_calculation_l232_23224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_problem_stream_speed_l232_23231

/-- Represents the speed of a boat in still water -/
def boat_speed (downstream_distance upstream_distance downstream_time upstream_time : ℝ) : ℝ := 
  sorry

/-- Represents the speed of the stream -/
def stream_speed (downstream_distance upstream_distance downstream_time upstream_time : ℝ) : ℝ := 
  sorry

/-- Theorem stating that given the conditions of the rowing problem, the stream speed is 3 km/h -/
theorem rowing_problem_stream_speed 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (h1 : downstream_distance = 90) 
  (h2 : upstream_distance = 72) 
  (h3 : downstream_time = 3) 
  (h4 : upstream_time = 3) 
  (h5 : downstream_distance = (boat_speed downstream_distance upstream_distance downstream_time upstream_time + 
                               stream_speed downstream_distance upstream_distance downstream_time upstream_time) * downstream_time)
  (h6 : upstream_distance = (boat_speed downstream_distance upstream_distance downstream_time upstream_time - 
                             stream_speed downstream_distance upstream_distance downstream_time upstream_time) * upstream_time) :
  stream_speed downstream_distance upstream_distance downstream_time upstream_time = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_problem_stream_speed_l232_23231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_l232_23202

/-- Given a rectangle composed of two rectangles R1 and R2, and three squares S1, S2, and S3,
    where the overall rectangle has a width of 4050 units and a height of 2550 units,
    the side length of S2 is 1500 units. -/
theorem square_side_length (R1 R2 S1 S2 S3 : ℕ → ℕ → Prop) 
  (h_width : ∃ w1 w2 s1 s2 s3, R1 w1 0 ∧ R2 w2 0 ∧ S1 s1 s1 ∧ S2 s2 s2 ∧ S3 s3 s3 ∧ w1 + w2 + s1 + s2 + s3 = 4050)
  (h_height : ∃ h1 h2 s2, R1 0 h1 ∧ R2 0 h2 ∧ S2 s2 s2 ∧ h1 + h2 + s2 = 2550)
  (h_square1 : ∀ x y, S1 x y → x = y)
  (h_square2 : ∀ x y, S2 x y → x = y)
  (h_square3 : ∀ x y, S3 x y → x = y)
  (h_equal_squares : ∃ s, (∀ x, S1 s x → S1 x s) ∧ (∀ x, S3 s x → S3 x s))
  (h_rect_equal_height : ∃ h, (∀ w, R1 w h → R1 h w) ∧ (∀ w, R2 w h → R2 h w)) :
  ∃ s, S2 s s ∧ s = 1500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_l232_23202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_k_and_t_range_l232_23262

/-- The function f(x) = k * 3^x + 3^(-x) -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * (3^x) + 3^(-x)

/-- f is an odd function -/
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_implies_k_and_t_range :
  ∀ k : ℝ, is_odd_function (f k) →
    (k = -1) ∧
    (∀ t : ℝ, (∀ x : ℝ, f k (t * x^2 + (t - 1) * x) + f k (x + t - 3) > 0) ↔ t ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_k_and_t_range_l232_23262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_flights_time_l232_23243

def stairClimbingTime (n : ℕ) : ℕ := 25 + (n - 1) * 8

def totalClimbingTime (n : ℕ) : ℕ := 
  (List.range n).map stairClimbingTime |>.sum

theorem seven_flights_time : totalClimbingTime 7 = 343 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_flights_time_l232_23243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l232_23295

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (x - Real.pi / 2) + Real.cos (Real.pi - x)

-- Define the interval bounds
noncomputable def lower_bound (k : ℤ) : ℝ := -Real.pi / 3 + 2 * ↑k * Real.pi
noncomputable def upper_bound (k : ℤ) : ℝ := 2 * Real.pi / 3 + 2 * ↑k * Real.pi

-- Theorem statement
theorem f_strictly_increasing (k : ℤ) :
  StrictMonoOn f (Set.Icc (lower_bound k) (upper_bound k)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l232_23295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l232_23249

/-- Represents an isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  longBase : ℝ
  baseAngle : ℝ
  area : ℝ

/-- The area of the isosceles trapezoid with given parameters -/
def trapezoidArea (t : IsoscelesTrapezoid) : ℝ :=
  t.area

/-- Theorem stating the area of the specific isosceles trapezoid -/
theorem specific_trapezoid_area :
  ∃ t : IsoscelesTrapezoid, 
    t.longBase = 20 ∧ 
    t.baseAngle = Real.arccos 0.6 ∧ 
    abs (trapezoidArea t - 74.12) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l232_23249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sphere_in_torus_l232_23253

/-- The center of the generating circle of the torus -/
def P : Fin 3 → ℝ := ![5, 0, 1]

/-- The radius of the generating circle of the torus -/
def torus_radius : ℝ := 1

/-- The radius of the largest spherical ball -/
noncomputable def sphere_radius : ℝ := 13/2

theorem largest_sphere_in_torus :
  let O : Fin 3 → ℝ := ![0, 0, sphere_radius]
  (P 0 ^ 2 + (sphere_radius - P 2) ^ 2 = sphere_radius ^ 2) ∧
  (∀ r : ℝ, r > sphere_radius →
    P 0 ^ 2 + (r - P 2) ^ 2 > r ^ 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sphere_in_torus_l232_23253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_battery_life_is_8_hours_l232_23263

/-- Represents the battery life of a cell phone -/
structure CellPhoneBattery where
  idle_life : ℚ  -- Battery life in hours when idle
  use_life : ℚ   -- Battery life in hours when in constant use
  total_time : ℚ -- Total time since last recharge in hours
  use_time : ℚ   -- Time the phone has been in use in hours

/-- Calculates the remaining battery life in hours -/
noncomputable def remaining_battery_life (battery : CellPhoneBattery) : ℚ :=
  let idle_rate := 1 / battery.idle_life
  let use_rate := 1 / battery.use_life
  let idle_time := battery.total_time - battery.use_time
  let battery_used := idle_time * idle_rate + battery.use_time * use_rate
  let battery_remaining := 1 - battery_used
  battery_remaining / idle_rate

/-- Theorem stating that given the conditions, the remaining battery life is 8 hours -/
theorem remaining_battery_life_is_8_hours (battery : CellPhoneBattery) 
  (h1 : battery.idle_life = 24)
  (h2 : battery.use_life = 3)
  (h3 : battery.total_time = 9)
  (h4 : battery.use_time = 1) : 
  remaining_battery_life battery = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_battery_life_is_8_hours_l232_23263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_bhd_l232_23277

/-- Represents a point in 2D space -/
structure MyPoint where
  x : ℝ
  y : ℝ

/-- Represents a vector in 2D space -/
structure MyVector where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure MyTriangle where
  a : MyPoint
  b : MyPoint
  c : MyPoint

/-- Checks if a triangle is equilateral -/
def is_equilateral (t : MyTriangle) : Prop := sorry

/-- Checks if two vectors are equal -/
def vector_eq (v1 v2 : MyVector) : Prop := v1.x = v2.x ∧ v1.y = v2.y

/-- The main theorem -/
theorem equilateral_bhd 
  (A B C D E H K : MyPoint)
  (ABC : MyTriangle)
  (CDE : MyTriangle)
  (EHK : MyTriangle)
  (BHD : MyTriangle)
  (h1 : is_equilateral ABC)
  (h2 : is_equilateral CDE)
  (h3 : is_equilateral EHK)
  (h4 : vector_eq (MyVector.mk (D.x - A.x) (D.y - A.y)) (MyVector.mk (K.x - D.x) (K.y - D.y)))
  : is_equilateral BHD := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_bhd_l232_23277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_parallelepiped_l232_23279

-- Define the rectangular parallelepiped
structure RectParallelepiped where
  a : ℝ  -- length of the square base
  b : ℝ  -- height of the parallelepiped
  h_pos : a > 0 ∧ b > 0  -- positive dimensions

-- Define the angle between BD₁ and plane BDC₁
noncomputable def angle (p : RectParallelepiped) : ℝ :=
  Real.arcsin (p.a * p.b / Real.sqrt ((2 * p.a^2 + p.b^2) * (p.a^2 + 2 * p.b^2)))

-- Theorem statement
theorem max_angle_parallelepiped (p : RectParallelepiped) : 
  angle p ≤ Real.arcsin (1/3) := by
  sorry

#check max_angle_parallelepiped

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_parallelepiped_l232_23279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_assembly_l232_23299

/-- Represents a rhombus-shaped figurine with a specific color pattern -/
structure Rhombus :=
  (pattern : Bool → Bool → Bool)  -- True represents one color, False the other

/-- Represents a larger shape composed of rhombuses -/
structure LargeShape :=
  (pattern : Fin 4 → Fin 4 → Bool)  -- True represents one color, False the other

/-- Rotates a rhombus by 90 degrees clockwise -/
def rotate (r : Rhombus) : Rhombus :=
  { pattern := fun x y => r.pattern y (not x) }

/-- Checks if a large shape can be assembled from a given rhombus -/
def canAssemble (r : Rhombus) (s : LargeShape) : Prop :=
  ∃ (arrangement : Fin 4 → Fin 4 → Rhombus),
    ∀ i j, s.pattern i j = (arrangement i j).pattern (i.val % 2 = 0) (j.val % 2 = 0)

/-- The four larger shapes to consider -/
def shapeA : LargeShape := sorry
def shapeB : LargeShape := sorry
def shapeC : LargeShape := sorry
def shapeD : LargeShape := sorry

/-- The main theorem stating that shape C cannot be assembled while others can -/
theorem rhombus_assembly (r : Rhombus) :
  (canAssemble r shapeA) ∧
  (canAssemble r shapeB) ∧
  ¬(canAssemble r shapeC) ∧
  (canAssemble r shapeD) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_assembly_l232_23299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_exists_l232_23276

-- Define the point P
noncomputable def P : ℝ × ℝ := (4/3, 2)

-- Define the line type
structure Line where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0

-- Define the condition that the line passes through P
def passesThrough (l : Line) : Prop :=
  4/(3*l.a) + 2/l.b = 1

-- Define the perimeter condition
def perimeterCondition (l : Line) : Prop :=
  l.a + l.b + Real.sqrt (l.a^2 + l.b^2) = 12

-- Define the area condition
def areaCondition (l : Line) : Prop :=
  l.a * l.b = 12

-- The main theorem
theorem unique_line_exists :
  ∃! l : Line, passesThrough l ∧ perimeterCondition l ∧ areaCondition l ∧
    3 * l.a = 4 ∧ 4 * l.b = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_exists_l232_23276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l232_23284

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 3 * Real.cos x ^ 2

-- State the theorem
theorem f_properties :
  -- 1. Smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S)) ∧
  -- 2. Monotonically increasing on [kπ - π/3, kπ + π/6] for all k ∈ ℤ
  (∀ k : ℤ, ∀ x y : ℝ, 
    x ∈ Set.Icc (k * Real.pi - Real.pi/3) (k * Real.pi + Real.pi/6) ∧ 
    y ∈ Set.Icc (k * Real.pi - Real.pi/3) (k * Real.pi + Real.pi/6) ∧ 
    x ≤ y → 
    f x ≤ f y) ∧
  -- 3. Range of f(x) on [-π/6, π/3] is [1, 4]
  Set.range (f ∘ (fun x => x : Set.Icc (-Real.pi/6) (Real.pi/3) → ℝ)) = Set.Icc 1 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l232_23284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l232_23267

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Calculate the distance between two parallel lines -/
noncomputable def distance (l1 l2 : Line) : ℝ :=
  |l1.c - l2.c| / Real.sqrt (l1.a^2 + l1.b^2)

/-- The main theorem -/
theorem line_equation (l1 : Line) :
  parallel l1 { a := 1, b := 1, c := -1 } →
  distance l1 { a := 1, b := 1, c := -1 } = Real.sqrt 2 →
  (l1 = { a := 1, b := 1, c := -1 } ∨ l1 = { a := 1, b := 1, c := 3 }) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l232_23267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_third_angle_l232_23248

theorem cosine_third_angle (A B C : ℝ) (h1 : 0 < A) (h2 : A < Real.pi) 
  (h3 : 0 < B) (h4 : B < Real.pi) (h5 : Real.cos A = 4/5) (h6 : Real.cos B = 5/13) : 
  Real.cos C = 16/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_third_angle_l232_23248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_arithmetic_nor_geometric_l232_23283

-- Define the sequence S_n
def S (n : ℕ) : ℕ := 2 * n * (n - 1) + 1

-- Define a_n explicitly
def a : ℕ → ℕ
  | 0 => 0  -- Adding this case to cover Nat.zero
  | 1 => 1
  | (n + 2) => S (n + 2) - S (n + 1)

-- Theorem statement
theorem not_arithmetic_nor_geometric : 
  ¬(∃ d : ℕ, ∀ n > 1, a (n + 1) = a n + d) ∧ 
  ¬(∃ r : ℚ, r ≠ 1 ∧ ∀ n > 1, a (n + 1) = a n * r) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_arithmetic_nor_geometric_l232_23283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_spending_equals_16X_l232_23296

/-- Emily's daily spending from Friday to Sunday -/
def X : ℝ := sorry

/-- Combined total spending of Emily and John over four days -/
def Y : ℝ := sorry

/-- Emily's spending on Monday -/
def emily_monday : ℝ := 4 * X

/-- John's daily spending from Friday to Sunday -/
def john_fri_to_sun : ℝ := 2 * X

/-- John's spending on Monday -/
def john_monday : ℝ := 3 * X

/-- Theorem: The combined total spending Y is equal to 16X -/
theorem combined_spending_equals_16X : Y = 16 * X := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_spending_equals_16X_l232_23296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_reflected_arcs_area_l232_23228

/-- The area of the region bounded by 8 reflected arcs of a regular octagon with side length 1 inscribed in a circle --/
noncomputable def boundedArea : ℝ :=
  2 * (1 + Real.sqrt 2) - 8 * (Real.pi / (8 * (2 - Real.sqrt 2)) - Real.sqrt 2 / 4)

/-- Theorem stating the area of the bounded region --/
theorem octagon_reflected_arcs_area :
  let r : ℝ := 1 / Real.sqrt (2 - Real.sqrt 2)
  let octagonArea : ℝ := 2 * (1 + Real.sqrt 2)
  let reflectedArcArea : ℝ := Real.pi / (8 * (2 - Real.sqrt 2)) - Real.sqrt 2 / 4
  octagonArea - 8 * reflectedArcArea = boundedArea := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_reflected_arcs_area_l232_23228
