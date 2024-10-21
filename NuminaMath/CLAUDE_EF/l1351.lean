import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_exists_l1351_135140

/-- The number of positive integer divisors of n -/
def d (n : ℕ+) : ℕ := sorry

/-- g₁(n) is thrice the number of positive integer divisors of n -/
def g₁ (n : ℕ+) : ℕ := 3 * d n

/-- For j ≥ 2, gⱼ(n) = g₁(gⱼ₋₁(n)) -/
def g : ℕ → ℕ+ → ℕ
  | 0, n => n
  | 1, n => g₁ n
  | j+2, n => 
    let prev := g (j+1) n
    if h : prev > 0 then g₁ ⟨prev, h⟩ else 0

/-- There exists exactly one positive integer n ≤ 100 such that g₃₀(n) = 18 -/
theorem unique_n_exists : ∃! (n : ℕ+), n ≤ 100 ∧ g 30 n = 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_exists_l1351_135140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_inexpressible_l1351_135127

def expressible (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, a > b ∧ c > d ∧ n = (2^a - 2^b) / (2^c - 2^d)

theorem smallest_inexpressible : 
  (∀ m : ℕ, 0 < m ∧ m < 11 → expressible m) ∧ 
  ¬expressible 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_inexpressible_l1351_135127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_X_l1351_135181

/-- The variance of a uniformly distributed random variable on [α, β] is (β - α)^2 / 12 -/
noncomputable def variance_uniform (α β : ℝ) : ℝ := (β - α)^2 / 12

/-- The random variable X is uniformly distributed on [4, 6] -/
noncomputable def X : ℝ → ℝ := sorry

theorem variance_of_X :
  variance_uniform 4 6 = 1/3 := by
  -- Unfold the definition of variance_uniform
  unfold variance_uniform
  -- Simplify the expression
  simp [sub_eq_add_neg, pow_two]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_X_l1351_135181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_auto_credit_percentage_approx_l1351_135126

/-- Represents the total consumer installment credit in billions of dollars -/
noncomputable def total_consumer_credit : ℝ := 465.1162790697675

/-- Represents the credit extended by automobile finance companies in billions of dollars -/
noncomputable def auto_finance_credit : ℝ := 50

/-- Represents the fraction of total automobile installment credit accounted for by auto finance companies -/
noncomputable def auto_finance_fraction : ℝ := 1 / 4

/-- Calculates the percentage of consumer installment credit accounted for by automobile installment credit -/
noncomputable def auto_credit_percentage : ℝ :=
  (auto_finance_credit / auto_finance_fraction) / total_consumer_credit * 100

theorem auto_credit_percentage_approx :
  abs (auto_credit_percentage - 42.99) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_auto_credit_percentage_approx_l1351_135126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_positive_integers_l1351_135184

theorem range_of_positive_integers (x : ℝ) (K : List ℤ) : 
  x > -2 →
  K.length = 20 →
  K.Sorted (·<·) →
  K.Nodup →
  K.head! = -3*Int.ceil x + 7 →
  (∀ i ∈ K, ∃ j : ℕ, i = K.head! + j ∧ j < 20) →
  ((K.maximum?.getD 0) - (K.minimum?.getD 0) : ℤ) = 19 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_positive_integers_l1351_135184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1351_135136

-- Define the triangle ABC
def triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

-- Define the conditions
def triangle_conditions (a b c : ℝ) : Prop :=
  triangle a b c ∧
  b^2 - a^2 = (1/2) * c^2 ∧
  (1/2) * a * b * Real.sin (Real.pi/4) = 3

-- State the theorem
theorem triangle_properties (a b c : ℝ) 
  (h : triangle_conditions a b c) : 
  Real.tan (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))) = 2 ∧
  b = 3 ∧
  2 * Real.pi * (a / (2 * Real.sin (Real.pi/4))) = Real.pi * Real.sqrt 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1351_135136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1351_135151

theorem equation_solution :
  ∃ y : ℝ, 5 * (4 : ℝ)^y = 1280 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1351_135151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_value_l1351_135111

/-- The area of a regular octagon formed by removing isosceles right triangles 
    from the corners of a square, where the legs of these triangles are 2 units long. -/
noncomputable def octagon_area : ℝ :=
  let leg_length : ℝ := 2
  let triangle_hypotenuse : ℝ := leg_length * Real.sqrt 2
  let square_side : ℝ := 2 * leg_length + triangle_hypotenuse
  let triangle_area : ℝ := (1 / 2) * leg_length * leg_length
  let square_area : ℝ := square_side * square_side
  let total_triangle_area : ℝ := 4 * triangle_area
  square_area - total_triangle_area

/-- The area of the octagon is equal to 16 + 8√2 square units. -/
theorem octagon_area_value : octagon_area = 16 + 8 * Real.sqrt 2 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem and may cause issues
-- #eval octagon_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_value_l1351_135111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1351_135198

/-- The parabola defined by the equation 4x³ - 8xy + 4y² + 8x - 16y + 3 = 0 -/
def Parabola (x y : ℝ) : Prop :=
  4 * x^3 - 8 * x * y + 4 * y^2 + 8 * x - 16 * y + 3 = 0

/-- The vertex of the parabola -/
noncomputable def Vertex : ℝ × ℝ := (-3/2, 0)

/-- The focus of the parabola -/
noncomputable def Focus : ℝ × ℝ := (-11/8, 1/8)

/-- The directrix of the parabola -/
def Directrix (x y : ℝ) : Prop :=
  x + y = -7/4

theorem parabola_properties :
  ∃ (x y : ℝ), Parabola x y ∧
    ((x, y) = Vertex ∨ 
    (x, y) = Focus ∨ 
    Directrix x y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1351_135198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_cos_a_l1351_135189

theorem largest_cos_a (a b c : ℝ) 
  (h1 : Real.sin a = Real.tan b⁻¹) 
  (h2 : Real.sin b = Real.tan c⁻¹) 
  (h3 : Real.sin c = Real.tan a⁻¹) : 
  ∃ (max_cos_a : ℝ), max_cos_a = Real.sqrt ((3 - Real.sqrt 5) / 2) ∧ 
  ∀ (cos_a : ℝ), cos_a = Real.cos a → cos_a ≤ max_cos_a := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_cos_a_l1351_135189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_with_special_angle_is_nonagon_l1351_135155

/-- A convex polygon where an interior angle is 180° more than three times its exterior angle has 9 sides. -/
theorem polygon_with_special_angle_is_nonagon (n : ℕ) :
  (∃ (i e : ℝ), i = 3 * e + 180 ∧ i + e = (n - 2 : ℝ) * 180 / n) →
  n = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_with_special_angle_is_nonagon_l1351_135155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_cube_arrangement_l1351_135117

/-- A type representing the vertices of a cube --/
inductive CubeVertex
| v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8

/-- A function type representing an arrangement of numbers on a cube --/
def CubeArrangement := CubeVertex → Fin 8

/-- Predicate to check if two vertices share an edge --/
def sharesEdge (v1 v2 : CubeVertex) : Prop := sorry

/-- The sum of numbers on adjacent vertices --/
def sumOfAdjacent (arr : CubeArrangement) (v : CubeVertex) : ℕ :=
  match v with
  | CubeVertex.v1 => sorry
  | CubeVertex.v2 => sorry
  | CubeVertex.v3 => sorry
  | CubeVertex.v4 => sorry
  | CubeVertex.v5 => sorry
  | CubeVertex.v6 => sorry
  | CubeVertex.v7 => sorry
  | CubeVertex.v8 => sorry

/-- The main theorem stating the impossibility of the arrangement --/
theorem no_valid_cube_arrangement :
  ¬∃ (arr : CubeArrangement), 
    (∀ v : CubeVertex, (arr v).val + 1 ∣ sumOfAdjacent arr v) ∧ 
    (∀ v : CubeVertex, ∀ w : CubeVertex, v ≠ w → arr v ≠ arr w) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_cube_arrangement_l1351_135117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_after_11th_inning_l1351_135177

/-- Represents a batsman's performance over a series of innings -/
structure BatsmanPerformance where
  innings : ℕ
  totalRuns : ℕ
  averageIncrease : ℚ
  lastInningScore : ℕ

/-- Calculates the new average after an additional inning -/
noncomputable def newAverage (bp : BatsmanPerformance) : ℚ :=
  (bp.totalRuns + bp.lastInningScore : ℚ) / (bp.innings + 1 : ℚ)

/-- Theorem stating that under given conditions, the new average is 45 -/
theorem batsman_average_after_11th_inning
  (bp : BatsmanPerformance)
  (h1 : bp.innings = 10)
  (h2 : bp.lastInningScore = 95)
  (h3 : bp.averageIncrease = 5)
  (h4 : newAverage bp = (bp.totalRuns : ℚ) / (bp.innings : ℚ) + bp.averageIncrease) :
  newAverage bp = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_after_11th_inning_l1351_135177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_p_range_l1351_135122

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * (x - 1) * Real.exp x
noncomputable def g (x p : ℝ) : ℝ := Real.exp x - x + p

-- Statement 1
theorem f_range (a : ℝ) (h : ∀ x y, a < x → x < y → f x < f y) :
  f a ∈ Set.Ici (-2) := by
  sorry

-- Statement 2
theorem p_range (p : ℝ) (h : ∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 (Real.exp 1) ∧ g x₀ p ≥ f x₀ - x₀) :
  p ∈ Set.Ici (-Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_p_range_l1351_135122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_l1351_135190

noncomputable def point_B : ℝ × ℝ := (0, 0)
noncomputable def point_D : ℝ × ℝ := (8, 6)
noncomputable def point_A : ℝ × ℝ := (20, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem sum_of_distances :
  distance point_A point_D + distance point_B point_D = 10 + 6 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_l1351_135190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_heads_probability_even_heads_probability_50_l1351_135197

/-- Probability of getting heads for an unfair coin -/
noncomputable def p_heads : ℝ := 2/3

/-- Probability of getting an even number of heads after n tosses of an unfair coin -/
noncomputable def p_even_heads (n : ℕ) : ℝ := 1/2 * (1 + 1/3^n)

/-- Theorem stating the probability of getting an even number of heads after n tosses -/
theorem even_heads_probability (n : ℕ) : 
  p_even_heads n = 1/2 * (1 + 1/3^n) := by sorry

/-- Theorem for the specific case of 50 tosses -/
theorem even_heads_probability_50 : 
  p_even_heads 50 = 1/2 * (1 + 1/3^50) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_heads_probability_even_heads_probability_50_l1351_135197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_l1351_135186

theorem smallest_number : 
  let a : ℝ := -3
  let b : ℝ := 0
  let c : ℝ := -(-1)
  let d : ℝ := (-1)^2
  min a (min b (min c d)) = a := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_l1351_135186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_income_calculation_l1351_135188

/-- Calculates the annual income from a stock investment -/
noncomputable def annual_income (investment : ℝ) (dividend_rate : ℝ) (stock_price : ℝ) : ℝ :=
  let num_shares := investment / stock_price
  let face_value := 100  -- Assuming standard face value of $100 per share
  let income_per_share := face_value * dividend_rate
  num_shares * income_per_share

/-- Theorem stating that investing $6800 in a 20% stock at $136 yields $1000 annual income -/
theorem investment_income_calculation :
  annual_income 6800 0.20 136 = 1000 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval annual_income 6800 0.20 136

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_income_calculation_l1351_135188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_lower_bound_l1351_135138

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- Theorem: If the distance from any point on the right branch of a hyperbola
    to the line bx + ay - 2ab = 0 is always greater than a,
    then the eccentricity is greater than or equal to 2√3/3 -/
theorem hyperbola_eccentricity_lower_bound (h : Hyperbola)
  (h_distance : ∀ (x y : ℝ), x^2 / h.a^2 - y^2 / h.b^2 = 1 → x > 0 →
    Real.sqrt ((h.b * x + h.a * y - 2 * h.a * h.b)^2 / (h.a^2 + h.b^2)) > h.a) :
  eccentricity h ≥ 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_lower_bound_l1351_135138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1351_135137

-- Define the function (noncomputable due to sqrt)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - x) - Real.sqrt (x - 3)

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 1 ∧ ∀ x, x ∈ Set.Icc 3 4 → f x ≤ M := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1351_135137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_common_factors_with_45_l1351_135121

/-- The number of positive common factors of two positive integers -/
def numCommonFactors (a b : ℕ) : ℕ :=
  (Finset.filter (fun d => d ∣ a ∧ d ∣ b) (Finset.range (min a b + 1))).card

/-- Theorem stating that if a positive integer n has exactly 3 positive common factors with 45, then n = 15 -/
theorem three_common_factors_with_45 (n : ℕ) (hn : 0 < n) :
  numCommonFactors n 45 = 3 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_common_factors_with_45_l1351_135121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_62_5_l1351_135187

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  altitude : ℚ
  shorter_base : ℚ
  longer_base : ℚ
  shorter_base_eq : shorter_base = 2 * altitude
  longer_base_eq : longer_base = 3 * altitude
  shorter_base_length : shorter_base = 10

/-- Calculates the area of a trapezoid -/
def trapezoid_area (t : Trapezoid) : ℚ :=
  (t.shorter_base + t.longer_base) * t.altitude / 2

/-- Theorem stating that the area of the given trapezoid is 62.5 square units -/
theorem trapezoid_area_is_62_5 (t : Trapezoid) :
  trapezoid_area t = 125/2 := by
  sorry

#check trapezoid_area_is_62_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_62_5_l1351_135187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_average_gain_l1351_135103

/-- Calculate the percentage gain for a given false weight -/
noncomputable def percentage_gain (false_weight : ℝ) : ℝ :=
  (1000 - false_weight) / false_weight * 100

/-- Calculate the average percentage gain for three false weights -/
noncomputable def average_percentage_gain (w1 w2 w3 : ℝ) : ℝ :=
  (percentage_gain w1 + percentage_gain w2 + percentage_gain w3) / 3

/-- Theorem stating that the average percentage gain for the given false weights is approximately 4.20% -/
theorem shopkeeper_average_gain :
  let w1 := 940
  let w2 := 960
  let w3 := 980
  abs (average_percentage_gain w1 w2 w3 - 4.20) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_average_gain_l1351_135103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_tan_shifted_l1351_135196

open Set Real

noncomputable def tan_shifted (x : ℝ) : ℝ := tan (x + π/3)

theorem domain_of_tan_shifted :
  {x : ℝ | ∃ y, tan_shifted x = y} = {x : ℝ | ∀ k : ℤ, x ≠ k * π + π/6} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_tan_shifted_l1351_135196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_0_to_2_instantaneous_speed_at_2_l1351_135149

-- Define the displacement function
noncomputable def S (t : ℝ) : ℝ := 3 * t - t^2

-- Define average speed
noncomputable def average_speed (t₁ t₂ : ℝ) : ℝ := (S t₂ - S t₁) / (t₂ - t₁)

-- Define instantaneous speed as the absolute value of the derivative
noncomputable def instantaneous_speed (t : ℝ) : ℝ := 
  abs (deriv S t)

-- Theorem for average speed
theorem average_speed_0_to_2 : average_speed 0 2 = 1 := by sorry

-- Theorem for instantaneous speed
theorem instantaneous_speed_at_2 : instantaneous_speed 2 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_0_to_2_instantaneous_speed_at_2_l1351_135149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_maximum_point_exists_and_bounded_l1351_135168

noncomputable def f (x : ℝ) : ℝ := x * (x - 1 - Real.log x)

theorem f_maximum_point_exists_and_bounded :
  (∀ x > 0, f x ≥ 0) →
  ∃ x₀ > 0, (∀ x > 0, f x ≤ f x₀) ∧ 
    (∀ x > 0, x ≠ x₀ → f x < f x₀) ∧
    (Real.exp (-2) < f x₀ ∧ f x₀ < (1/2)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_maximum_point_exists_and_bounded_l1351_135168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_is_right_l1351_135163

-- Define the shapes
structure Triangle where

structure Pentagon where

-- Define the properties of the shapes
def total_area (t1 t2 : Triangle) (p : Pentagon) : ℝ := 24

-- Define the rectangles
def rectangle_4x6 (t1 t2 : Triangle) (p : Pentagon) : Prop :=
  sorry -- Placeholder definition

def rectangle_3x8 (t1 t2 : Triangle) (p : Pentagon) : Prop :=
  sorry -- Placeholder definition

-- Theorem statement
theorem vasya_is_right (t1 t2 : Triangle) (p : Pentagon) :
  total_area t1 t2 p = 24 →
  rectangle_4x6 t1 t2 p →
  rectangle_3x8 t1 t2 p →
  ∃ (t1' t2' : Triangle) (p' : Pentagon),
    total_area t1' t2' p' = 24 ∧
    rectangle_4x6 t1' t2' p' ∧
    rectangle_3x8 t1' t2' p' :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_is_right_l1351_135163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_both_lines_l1351_135128

/-- Two lines in 2D space -/
structure Line2D where
  a : ℝ × ℝ  -- Point on the line
  v : ℝ × ℝ  -- Direction vector

/-- The first line -/
noncomputable def line1 : Line2D := { a := (3, 2), v := (3, -4) }

/-- The second line -/
noncomputable def line2 : Line2D := { a := (4, -6), v := (6, 3) }

/-- A point lies on a line if it satisfies the parametric equation -/
def lies_on (p : ℝ × ℝ) (l : Line2D) : Prop :=
  ∃ t : ℝ, p.1 = l.a.1 + t * l.v.1 ∧ p.2 = l.a.2 + t * l.v.2

/-- The intersection point of the two lines -/
noncomputable def intersection_point : ℝ × ℝ := (84/11, -46/11)

/-- Theorem: The intersection_point lies on both lines -/
theorem intersection_point_on_both_lines :
  lies_on intersection_point line1 ∧ lies_on intersection_point line2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_both_lines_l1351_135128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_newer_bus_distance_l1351_135156

/-- 
Given an older bus model that travels a certain distance in a specific duration,
and a newer model that can travel a percentage farther in the same time,
this function calculates the distance the newer model can travel.
-/
noncomputable def newer_model_distance (old_distance : ℝ) (percentage_increase : ℝ) : ℝ :=
  old_distance * (1 + percentage_increase / 100)

/-- 
Theorem stating that given an older bus model that travels 300 miles in a specific duration,
and a newer model that can travel 30% farther in the same time,
the newer model will travel 390 miles in that duration.
-/
theorem newer_bus_distance : 
  newer_model_distance 300 30 = 390 := by
  -- Unfold the definition of newer_model_distance
  unfold newer_model_distance
  -- Simplify the arithmetic
  simp [mul_add, mul_one]
  -- Check that 300 * (1 + 30 / 100) = 390
  norm_num

-- We can't use #eval with noncomputable functions, so we'll use #check instead
#check newer_model_distance 300 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_newer_bus_distance_l1351_135156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_intersect_N_l1351_135160

def M : Set ℤ := {-1, 1}

def N : Set ℤ := {x | (1/2 : ℝ) < (2 : ℝ)^(x+1) ∧ (2 : ℝ)^(x+1) < 4}

theorem M_intersect_N : M ∩ N = {-1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_intersect_N_l1351_135160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_wallets_one_coin_possible_l1351_135191

/-- Represents a wallet that can contain coins and other wallets. -/
inductive Wallet
| empty : Wallet
| with_coin : Wallet
| with_wallet : Wallet → Wallet

/-- Represents the scenario with two wallets and one coin. -/
def TwoWalletsOneCoin : Prop :=
  ∃ (wallet_a wallet_b : Wallet),
    (wallet_a = Wallet.with_coin ∨ 
     (∃ inner, wallet_a = Wallet.with_wallet inner ∧ inner = Wallet.with_coin)) ∧
    (wallet_b = Wallet.with_coin ∨ 
     (∃ inner, wallet_b = Wallet.with_wallet inner ∧ inner = Wallet.with_coin)) ∧
    (wallet_a = Wallet.with_coin ∨ wallet_b = Wallet.with_coin) ∧
    ¬(wallet_a = Wallet.with_coin ∧ wallet_b = Wallet.with_coin)

/-- Theorem stating that the TwoWalletsOneCoin scenario is possible. -/
theorem two_wallets_one_coin_possible : TwoWalletsOneCoin := by
  sorry

#check two_wallets_one_coin_possible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_wallets_one_coin_possible_l1351_135191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_is_correct_l1351_135183

/-- The length of the major axis of an ellipse with given foci and tangent to y-axis -/
def ellipse_major_axis_length : ℝ := 20

/-- Definition of the ellipse -/
def ellipse : Set (ℝ × ℝ) :=
  let f1 : ℝ × ℝ := (-15, 10)
  let f2 : ℝ × ℝ := (15, 30)
  {p : ℝ × ℝ | ∃ k, dist p f1 + dist p f2 = k ∧ 
               ∃ y, p.1 = 0 ∧ dist (0, y) f1 + dist (0, y) f2 = k}

/-- The ellipse is tangent to the y-axis -/
axiom ellipse_tangent_y_axis : 
  ∃ p : ℝ × ℝ, p.1 = 0 ∧ p ∈ ellipse

/-- The calculated length of the major axis is correct -/
theorem ellipse_major_axis_length_is_correct : 
  ellipse_major_axis_length = 20 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_is_correct_l1351_135183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_proof_l1351_135145

def line_direction : ℝ × ℝ := (5, 4)

def perpendicular_to_line_direction : ℝ × ℝ := (4, -5)

def v : ℝ → ℝ → ℝ × ℝ := λ v₁ v₂ => (v₁, v₂)

theorem projection_vector_proof (v₁ v₂ : ℝ) (h1 : 3 * v₁ + v₂ = 4) 
  (h2 : ∃ (k : ℝ), v v₁ v₂ = k • perpendicular_to_line_direction) : 
  v v₁ v₂ = (16/7, -20/7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_proof_l1351_135145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_and_sequence_theorem_l1351_135102

/-- A quadratic function satisfying certain conditions -/
def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧
  (∀ x, f x = a * x^2 + b * x + c) ∧
  (∀ x, 4 * x ≤ f x ∧ f x ≤ (1/2) * (x + 2)^2) ∧
  f (-4 + 2 * Real.sqrt 3) = 0

/-- The sequence of functions defined recursively -/
noncomputable def f_seq : ℕ → (ℝ → ℝ)
  | 0 => λ x => 3 / (2 + x)
  | n + 1 => λ x => f_seq 0 (f_seq n x)

/-- The main theorem combining both parts of the problem -/
theorem quadratic_and_sequence_theorem :
  (∃! f : ℝ → ℝ, quadratic_function f ∧ 
    ∀ x, f x = (1/3) * x^2 + (8/3) * x + (4/3)) ∧
  f_seq 2008 0 = (3^2010 + 3) / (3^2010 - 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_and_sequence_theorem_l1351_135102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_specific_l1351_135161

theorem tan_double_angle_specific (x : ℝ) 
  (h1 : Real.cos x = 4/5) 
  (h2 : x ∈ Set.Ioo (-π/2) 0) : 
  Real.tan (2*x) = -24/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_specific_l1351_135161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l1351_135101

/-- Represents a tap that can fill or empty a cistern -/
structure Tap where
  fillTime : ℚ
  isFilling : Bool

/-- Calculates the rate at which a tap fills or empties a cistern -/
noncomputable def tapRate (tap : Tap) : ℚ :=
  if tap.isFilling then 1 / tap.fillTime else -1 / tap.fillTime

/-- Calculates the time to fill the cistern when multiple taps are opened simultaneously -/
noncomputable def fillTime (taps : List Tap) : ℚ :=
  1 / (taps.map tapRate).sum

theorem cistern_fill_time (tapA tapB tapC : Tap)
  (hA : tapA = { fillTime := 7, isFilling := true })
  (hB : tapB = { fillTime := 9, isFilling := false })
  (hC : tapC = { fillTime := 12, isFilling := true }) :
  fillTime [tapA, tapB, tapC] = 252 / 29 := by
  sorry

#eval (252 : ℚ) / 29

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l1351_135101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_uniqueness_l1351_135195

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a circle in 2D space
structure Circle where
  center : Point
  radius : ℝ

-- Define a parabola
structure Parabola where
  focus : Point
  directrix : Line

-- Define a tangent to a parabola
def Tangent (p : Parabola) := Line

-- Define a point that lies on a circle
def LiesOnCircle (p : Point) (c : Circle) : Prop := 
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- Define a circumcircle of three points
noncomputable def Circumcircle (p1 p2 p3 : Point) : Circle := sorry

-- Define intersection of two lines
noncomputable def Intersection (l1 l2 : Line) : Point := sorry

-- Theorem statement
theorem parabola_focus_uniqueness 
  (p : Parabola) 
  (t1 t2 t3 t4 : Tangent p) 
  (distinct : t1 ≠ t2 ∧ t1 ≠ t3 ∧ t1 ≠ t4 ∧ t2 ≠ t3 ∧ t2 ≠ t4 ∧ t3 ≠ t4) :
  ∃! f : Point, 
    (LiesOnCircle f (Circumcircle (Intersection t1 t2) (Intersection t2 t3) (Intersection t3 t1))) ∧
    (LiesOnCircle f (Circumcircle (Intersection t1 t2) (Intersection t2 t4) (Intersection t4 t1))) ∧
    (LiesOnCircle f (Circumcircle (Intersection t1 t3) (Intersection t3 t4) (Intersection t4 t1))) ∧
    (LiesOnCircle f (Circumcircle (Intersection t2 t3) (Intersection t3 t4) (Intersection t4 t2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_uniqueness_l1351_135195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_sum_of_squares_power_of_five_l1351_135132

theorem existence_of_sum_of_squares_power_of_five (n : ℕ) :
  ∃ x y : ℤ, x^2 + y^2 = (5 : ℤ)^n ∧ Nat.gcd x.natAbs 5 = 1 ∧ Nat.gcd y.natAbs 5 = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_sum_of_squares_power_of_five_l1351_135132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_similar_to_almost_diagonal_l1351_135144

open Matrix Complex

theorem matrix_similar_to_almost_diagonal {n : ℕ} (A : Matrix (Fin n) (Fin n) ℂ) 
  (h : ∀ (c : ℂ), A ≠ c • 1) : 
  ∃ (P : Matrix (Fin n) (Fin n) ℂ), IsUnit P ∧ 
    (∃ (i : Fin n), ∀ (j : Fin n), i ≠ j → (P⁻¹ * A * P) j j = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_similar_to_almost_diagonal_l1351_135144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_if_arbitrarily_long_runs_l1351_135139

/-- A function that represents the decimal expansion of a real number -/
def DecimalExpansion := ℕ → Fin 10

/-- A property that states a decimal expansion has arbitrarily long runs of a single digit -/
def HasArbitrarilyLongRuns (d : DecimalExpansion) : Prop :=
  ∀ k : ℕ, ∃ n m : ℕ, ∀ i : ℕ, i < k → d (n + i) = d (n + k + m + i)

/-- Convert a real number to its decimal expansion -/
noncomputable def realToDecimalExpansion (x : ℝ) : DecimalExpansion :=
  fun n => ⟨(⌊x * 10^n⌋ % 10).toNat, by sorry⟩

/-- Theorem stating that a number with arbitrarily long runs of a single digit in its decimal expansion is irrational -/
theorem irrational_if_arbitrarily_long_runs (x : ℝ) (d : DecimalExpansion)
    (h₁ : d = realToDecimalExpansion x)
    (h₂ : HasArbitrarilyLongRuns d) :
    Irrational x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_if_arbitrarily_long_runs_l1351_135139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_zero_line_equation_l1351_135162

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := y^2 = -x
def line (k x y : ℝ) : Prop := y = k * (x + 1)

-- Define the intersection points A and B
axiom intersection_points (k : ℝ) : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), parabola x₁ y₁ ∧ line k x₁ y₁ ∧ parabola x₂ y₂ ∧ line k x₂ y₂ ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the dot product of two vectors
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ := v₁.1 * v₂.1 + v₁.2 * v₂.2

-- Define the area of a triangle given three points
noncomputable def triangle_area (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := 
  (1/2) * abs ((p₂.1 - p₁.1) * (p₃.2 - p₁.2) - (p₃.1 - p₁.1) * (p₂.2 - p₁.2))

-- Theorem 1: OA · OB = 0
theorem dot_product_zero (k : ℝ) : 
  ∀ (x₁ y₁ x₂ y₂ : ℝ), parabola x₁ y₁ ∧ line k x₁ y₁ ∧ parabola x₂ y₂ ∧ line k x₂ y₂ ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
  dot_product (x₁, y₁) (x₂, y₂) = 0 := by sorry

-- Theorem 2: If area of OAB is 5/4, then the line equation is 2x + 3y + 2 = 0 or 2x - 3y + 2 = 0
theorem line_equation (k : ℝ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), parabola x₁ y₁ ∧ line k x₁ y₁ ∧ parabola x₂ y₂ ∧ line k x₂ y₂ ∧ 
   triangle_area origin (x₁, y₁) (x₂, y₂) = 5/4) →
  (k = 2/3 ∨ k = -2/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_zero_line_equation_l1351_135162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_solar_panel_area_minimum_total_cost_in_yuan_l1351_135193

/-- Represents the annual electricity consumption function -/
noncomputable def C (k : ℝ) (x : ℝ) : ℝ := k / (20 * x + 100)

/-- Represents the total cost function over 15 years -/
noncomputable def F (k : ℝ) (x : ℝ) : ℝ := 15 * C k x + 0.5 * x

/-- Theorem stating the optimal solar panel area and minimum total cost -/
theorem optimal_solar_panel_area (k : ℝ) :
  k = 2400 →
  (∀ x : ℝ, x ≥ 0 → F k x ≥ F k 55) ∧
  F k 55 = 57.5 := by
  sorry

/-- Corollary confirming the minimum total cost in yuan -/
theorem minimum_total_cost_in_yuan (k : ℝ) :
  k = 2400 →
  F k 55 * 10000 = 575000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_solar_panel_area_minimum_total_cost_in_yuan_l1351_135193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l1351_135157

/-- Calculates the time taken for the slower train to pass the driver of the faster train -/
noncomputable def time_to_pass (train_length : ℝ) (speed_fast : ℝ) (speed_slow : ℝ) : ℝ :=
  let relative_speed := speed_fast + speed_slow
  let relative_speed_mps := relative_speed * (1000 / 3600)
  train_length / relative_speed_mps

/-- Theorem stating the time taken for the slower train to pass the driver of the faster train -/
theorem train_passing_time :
  let train_length : ℝ := 500
  let speed_fast : ℝ := 45
  let speed_slow : ℝ := 15
  time_to_pass train_length speed_fast speed_slow = 30 := by
  -- Unfold the definition of time_to_pass
  unfold time_to_pass
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l1351_135157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_side_length_theorem_l1351_135115

/-- Represents an equiangular hexagon with an inscribed square -/
structure InscribedSquareHexagon where
  -- The side length of AB
  ab : ℝ
  -- The side length of EF
  ef : ℝ
  -- Assertion that the hexagon is equiangular
  is_equiangular : True
  -- Assertion that MNPQ is a square
  is_square : True
  -- Assertion that M is on AB, N is on CD, P is on EF
  points_on_sides : True

/-- The side length of the inscribed square in the hexagon -/
noncomputable def inscribed_square_side_length (h : InscribedSquareHexagon) : ℝ :=
  24 * Real.sqrt 3 - 22

/-- Theorem stating the side length of the inscribed square -/
theorem inscribed_square_side_length_theorem (h : InscribedSquareHexagon)
  (h_ab : h.ab = 50)
  (h_ef : h.ef = 45 * (Real.sqrt 3 - 2)) :
  inscribed_square_side_length h = 24 * Real.sqrt 3 - 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_side_length_theorem_l1351_135115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_specific_l1351_135185

theorem sin_double_angle_specific (α : ℝ) 
  (h1 : Real.sin (π - α) = 1/3) 
  (h2 : π/2 ≤ α) 
  (h3 : α ≤ π) : 
  Real.sin (2*α) = -4*Real.sqrt 2/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_specific_l1351_135185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_conversion_l1351_135106

/-- Given a principal P and time T, the simple interest at 5% for T years is 840. 
    Prove that the new interest rate R2 for the same principal over 8 years, 
    yielding the same interest, is (5 * T) / 8. -/
theorem simple_interest_rate_conversion 
  (P T : ℝ) 
  (h1 : P * 5 * T / 100 = 840) 
  (h2 : P > 0) 
  (h3 : T > 0) : 
  P * ((5 * T) / 8) * 8 / 100 = 840 := by
  sorry

#check simple_interest_rate_conversion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_conversion_l1351_135106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_light_probability_l1351_135143

/-- The probability of encountering a red light at an intersection -/
noncomputable def p : ℝ := 1/3

/-- The number of intersections -/
def n : ℕ := 4

/-- The number of red lights needed to have exactly 4 minutes of waiting time -/
def k : ℕ := 2

/-- The probability of encountering exactly k red lights out of n intersections -/
noncomputable def prob_exact_k_red_lights : ℝ := Nat.choose n k * p^k * (1-p)^(n-k)

theorem red_light_probability : prob_exact_k_red_lights = 8/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_light_probability_l1351_135143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_g_zeros_l1351_135107

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 6) * Real.cos (ω * x) + 1 / 2

noncomputable def g (x : ℝ) : ℝ := Real.sin (x + Real.pi / 6)

def has_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem f_period_and_g_zeros (ω : ℝ) (h_ω : ω > 0) :
  (has_period (f ω) Real.pi ∧ ∀ T > 0, has_period (f ω) T → T ≥ Real.pi) ↔ ω = 1 ∧
  {x : ℝ | g x = 0 ∧ x ∈ Set.Icc (-Real.pi) Real.pi} = {-Real.pi/6, 5*Real.pi/6} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_g_zeros_l1351_135107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_reduction_l1351_135174

/-- Represents a pyramid with a square base -/
structure Pyramid where
  baseLength : ℝ
  baseWidth : ℝ
  height : ℝ

/-- Calculates the volume of a pyramid -/
noncomputable def pyramidVolume (p : Pyramid) : ℝ :=
  (1 / 3) * p.baseLength * p.baseWidth * p.height

theorem pyramid_volume_reduction (originalPyramid : Pyramid)
  (h_volume : pyramidVolume originalPyramid = 72)
  (h_base : originalPyramid.baseLength = 2 ∧ originalPyramid.baseWidth = 2)
  (h_height : originalPyramid.height = 9) :
  let newPyramid : Pyramid := {
    baseLength := originalPyramid.baseLength,
    baseWidth := 0.9 * originalPyramid.baseWidth,
    height := 0.7 * originalPyramid.height
  }
  pyramidVolume newPyramid = 7.56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_reduction_l1351_135174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1351_135194

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  sum_angles : A + B + C = Real.pi
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- Define the conditions
def ArithmeticProgression (a b c : Real) : Prop :=
  b - a = c - b

theorem triangle_properties (t : Triangle) 
  (h1 : t.C = 2 * t.A) 
  (h2 : ArithmeticProgression t.a t.b t.c) : 
  (Real.cos t.A = 3/4) ∧ 
  (t.a = 2 → t.a * t.b * Real.sin t.C / 4 = 15 * Real.sqrt 7 / 16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1351_135194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_range_l1351_135172

theorem triangle_side_range (a : ℝ) : 
  (3 < 2*a+1 ∧ 2*a+1 < 11 ∧ 3 + (2*a+1) > 8) ↔ (2 < a ∧ a < 5) :=
by
  constructor
  · intro h
    have h1 : 2*a+1 > 3 := h.1
    have h2 : 2*a+1 < 11 := h.2.1
    have h3 : 3 + (2*a+1) > 8 := h.2.2
    constructor
    · linarith
    · linarith
  · intro h
    constructor
    · linarith
    · constructor
      · linarith
      · linarith
  done

#check triangle_side_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_range_l1351_135172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_condition_l1351_135108

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) - x - a * x

theorem extreme_value_condition (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x - 1| ∧ |x - 1| < ε → f a x ≤ f a 1) →
  a = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_condition_l1351_135108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_is_120_degrees_l1351_135104

-- Define a triangle with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions given in the problem
def satisfiesConditions (t : Triangle) : Prop :=
  t.a + 3 * t.b + 3 * t.c = t.a^2 ∧ t.a + 3 * t.b - 3 * t.c = -4

-- Define the angle C using the law of cosines
noncomputable def cosC (t : Triangle) : ℝ :=
  (t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b)

-- Theorem statement
theorem largest_angle_is_120_degrees (t : Triangle) 
  (h : satisfiesConditions t) : cosC t = -1/2 := by
  sorry

#check largest_angle_is_120_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_is_120_degrees_l1351_135104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_license_plate_increase_l1351_135153

theorem license_plate_increase : 
  (26^4 * 10^2 : ℚ) / (26^2 * 10^3) = 26^2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_license_plate_increase_l1351_135153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_8_factorial_l1351_135131

theorem divisors_of_8_factorial : 
  (Finset.card (Finset.filter (λ d => d > 0 ∧ d ∣ Nat.factorial 8) (Finset.range (Nat.factorial 8 + 1)))) = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_8_factorial_l1351_135131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_below_line_l1351_135170

/-- The circle with equation (x-3)^2 + (y-10)^2 = 64 -/
def myCircle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 10)^2 = 64}

/-- The line with equation y = 8 -/
def myLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 8}

/-- The area of the circle below the line -/
noncomputable def areaBelow : ℝ :=
  64 * Real.arccos (1/4) - 2 * Real.sqrt 60

theorem circle_area_below_line :
  ∃ ε > 0, |areaBelow - 68.86| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_below_line_l1351_135170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_positive_implies_positive_slope_l1351_135199

open Set
open Function

theorem derivative_positive_implies_positive_slope 
  (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h : ∀ x, HasDerivAt f (f' x) x) 
  (h_pos : ∀ x, f' x > 0) 
  (x₁ x₂ : ℝ) (h_neq : x₁ ≠ x₂) : 
  (f x₁ - f x₂) / (x₁ - x₂) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_positive_implies_positive_slope_l1351_135199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1351_135167

/-- Given a hyperbola C and a line l, prove that C has the equation x² - y²/3 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (C : Set (ℝ × ℝ)),
    (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 / a^2 - y^2 / b^2 = 1) ∧
    (∃ (l : Set (ℝ × ℝ)),
      (∀ (x y : ℝ), (x, y) ∈ l ↔ x + Real.sqrt 3 * y = 0) ∧
      (∃ (asymptote : Set (ℝ × ℝ)), asymptote ⊆ C ∧ 
        (∀ (p q : ℝ × ℝ), p ∈ asymptote ∧ q ∈ asymptote → 
          (p.1 - q.1) * Real.sqrt 3 + (p.2 - q.2) = 0)) ∧
      (∃ (focus : ℝ × ℝ), focus ∈ C ∧ 
        abs (focus.1 + Real.sqrt 3 * focus.2) / Real.sqrt (1 + 3) = 1))) →
  a = 1 ∧ b = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1351_135167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_is_four_ninths_l1351_135154

/-- The sum of the infinite series ∑(k/4^k) from k=1 to ∞ -/
noncomputable def series_sum : ℝ := ∑' k : ℕ+, (k : ℝ) / (4 : ℝ) ^ (k : ℕ)

/-- The theorem stating that the sum of the series is 4/9 -/
theorem series_sum_is_four_ninths : series_sum = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_is_four_ninths_l1351_135154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l1351_135119

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the sets P and Q
def P (t : ℝ) : Set ℝ := {x | |f (x + t) - 1| < 2}
def Q : Set ℝ := {x | f x < -1}

-- State the theorem
theorem range_of_t (h_decreasing : ∀ x y : ℝ, x < y → f y < f x)
                   (h_f0 : f 0 = 3)
                   (h_f3 : f 3 = -1)
                   (h_subset : ∀ t : ℝ, P t ⊆ Q)
                   (h_proper : ∀ t : ℝ, ∃ x : ℝ, x ∈ Q ∧ x ∉ P t) :
  ∀ t : ℝ, t ≤ -3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l1351_135119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_sections_proof_l1351_135152

-- Define the ellipse equation
noncomputable def ellipse_eq (x y : ℝ) : Prop := x^2 / 3 + y^2 / 12 = 1

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop := x^2 - y^2 / 4 = 1

-- Define the points
noncomputable def point_A : ℝ × ℝ := (3/2, -Real.sqrt 3)
noncomputable def point_B : ℝ × ℝ := (-Real.sqrt 2, 2)
noncomputable def point_P : ℝ × ℝ := (-Real.sqrt 2, 2)

-- Define the foci of the reference ellipse
noncomputable def focus_distance : ℝ := Real.sqrt 5

theorem conic_sections_proof :
  -- The ellipse passes through points A and B
  (ellipse_eq point_A.1 point_A.2 ∧ ellipse_eq point_B.1 point_B.2) ∧
  -- The hyperbola passes through point P
  hyperbola_eq point_P.1 point_P.2 ∧
  -- The hyperbola has foci at (±√5, 0)
  ∃ (a b : ℝ), a^2 - b^2 = focus_distance^2 ∧
    ∀ (x y : ℝ), hyperbola_eq x y ↔ (x^2 / a^2 - y^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_sections_proof_l1351_135152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_squares_constant_l1351_135130

theorem cosine_sum_squares_constant (A B C : ℝ) 
  (h1 : Real.cos A + Real.cos B + Real.cos C = 0) 
  (h2 : Real.sin A + Real.sin B + Real.sin C = 0) : 
  Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2 = 3/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_squares_constant_l1351_135130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solutions_l1351_135176

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 1 else 2*x

theorem f_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ = -2 ∧ x₂ = 5/2 ∧ f x₁ = 5 ∧ f x₂ = 5 ∧
  ∀ (x : ℝ), f x = 5 → x = x₁ ∨ x = x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solutions_l1351_135176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_minus_2alpha_l1351_135116

theorem tan_pi_minus_2alpha (α : ℝ) 
  (h1 : Real.sin (2 * α) = -Real.sin α) 
  (h2 : α ∈ Set.Ioo (π / 2) π) : 
  Real.tan (π - 2 * α) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_minus_2alpha_l1351_135116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_length_increases_l1351_135150

/-- Represents the distance of a person from a streetlight -/
def distance_from_streetlight : ℝ → ℝ := sorry

/-- Represents the length of a person's shadow -/
def shadow_length : ℝ → ℝ := sorry

/-- Theorem stating that as the distance from a streetlight increases, 
    the length of a person's shadow increases -/
theorem shadow_length_increases (d₁ d₂ : ℝ) :
  d₁ < d₂ → shadow_length (distance_from_streetlight d₁) < shadow_length (distance_from_streetlight d₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_length_increases_l1351_135150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relay_race_permutations_l1351_135173

theorem relay_race_permutations (n : ℕ) : n = 5 → (n - 1).factorial = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relay_race_permutations_l1351_135173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_k_value_l1351_135159

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define vectors a and b
variable (a b : V)

-- Define that a and b are not collinear
variable (h_not_collinear : ∀ (r : ℝ), a ≠ r • b)

-- Define vector m
def m (a b : V) : V := 2 • a - 3 • b

-- Define vector n
def n (a b : V) (k : ℝ) : V := 3 • a + k • b

-- Define that m is parallel to n
def parallel (m n : V) : Prop := ∃ (t : ℝ), n = t • m

-- Theorem statement
theorem parallel_vectors_k_value
  (h_parallel : parallel (m a b) (n a b k)) :
  k = -9/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_k_value_l1351_135159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_minus_2_over_a_minus_1_l1351_135182

open Real

-- Define the function f as noncomputable
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * a * x^2 + 2 * b * x + c

-- State the theorem
theorem range_of_b_minus_2_over_a_minus_1 
  (a b c : ℝ) 
  (h_diff : Differentiable ℝ (f a b c))
  (h_max : ∃ x₁ ∈ Set.Ioo 0 1, IsLocalMax (f a b c) x₁)
  (h_min : ∃ x₂ ∈ Set.Ioo 1 2, IsLocalMin (f a b c) x₂) :
  (1/4 : ℝ) < (b - 2) / (a - 1) ∧ (b - 2) / (a - 1) < 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_minus_2_over_a_minus_1_l1351_135182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equality_l1351_135129

def a : ℂ := 3 - 2*Complex.I
def b : ℂ := -2 + 3*Complex.I

theorem complex_expression_equality : 3*a + 4*b = 1 + 6*Complex.I := by
  -- Expand the definitions of a and b
  simp [a, b]
  -- Perform the arithmetic
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equality_l1351_135129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_any_triangle_l1351_135113

-- Define the types
def Triangle : Type := Unit  -- Placeholder definition
def Area : Triangle → ℝ := λ _ => 0  -- Placeholder definition
def Base : Triangle → ℝ := λ _ => 0  -- Placeholder definition
def Height : Triangle → ℝ := λ _ => 0  -- Placeholder definition

-- Define the triangle classifications
def IsAcute : Triangle → Prop := λ _ => True  -- Placeholder definition
def IsRight : Triangle → Prop := λ _ => True  -- Placeholder definition
def IsObtuse : Triangle → Prop := λ _ => True  -- Placeholder definition

-- State the theorem
theorem area_of_any_triangle 
  (t : Triangle) 
  (h_acute : ∀ t, IsAcute t → Area t = (1/2) * Base t * Height t)
  (h_right : ∀ t, IsRight t → Area t = (1/2) * Base t * Height t)
  (h_obtuse : ∀ t, IsObtuse t → Area t = (1/2) * Base t * Height t)
  (h_complete : ∀ t, IsAcute t ∨ IsRight t ∨ IsObtuse t) :
  Area t = (1/2) * Base t * Height t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_any_triangle_l1351_135113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_extrema_l1351_135179

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

-- Define the derivative of f
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

-- Define the function g
noncomputable def g (a b x : ℝ) : ℝ := (f' a b x) * Real.exp (-x)

-- State the theorem
theorem tangent_line_and_extrema (a b : ℝ) 
  (h1 : f' a b 1 = 2*a) 
  (h2 : f' a b 2 = -b) : 
  (∃ (m c : ℝ), m = 6 ∧ c = -1 ∧ ∀ (x y : ℝ), y = f a b x → m*x + 2*y + c = 0) ∧ 
  (∀ (x : ℝ), g a b x ≥ -3) ∧
  (∀ (x : ℝ), g a b x ≤ 15 * Real.exp (-3)) ∧
  (g a b 0 = -3) ∧
  (g a b 3 = 15 * Real.exp (-3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_extrema_l1351_135179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_annual_profit_l1351_135112

/-- Additional cost function -/
noncomputable def C (x : ℝ) : ℝ :=
  if x < 80 then (1/3) * x^2 + 10 * x
  else 51 * x + 10000 / x - 1450

/-- Annual profit function -/
noncomputable def L (x : ℝ) : ℝ :=
  50 * x - C x - 250

/-- Theorem stating the maximum annual profit and optimal production -/
theorem max_annual_profit :
  ∃ (max_profit : ℝ) (optimal_production : ℝ),
    max_profit = 1000 ∧
    optimal_production = 100 ∧
    ∀ x > 0, L x ≤ max_profit ∧
    L optimal_production = max_profit :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_annual_profit_l1351_135112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_with_reciprocal_eccentricity_l1351_135171

/-- Given a hyperbola with equation 3x^2 - y^2 = 3, prove that an ellipse with the same foci
    and reciprocal eccentricity has the equation x^2/16 + y^2/12 = 1 -/
theorem ellipse_with_reciprocal_eccentricity 
  (hyperbola : ℝ → ℝ → Prop) 
  (hyperbola_eq : ∀ x y, hyperbola x y ↔ 3 * x^2 - y^2 = 3) 
  (ellipse : ℝ → ℝ → Prop) 
  (same_foci : ∀ x y, hyperbola x y → ∃ f₁ f₂ : ℝ × ℝ, 
               (f₁.1 = -f₂.1 ∧ f₁.2 = 0 ∧ f₂.2 = 0) ∧
               ∀ x' y', ellipse x' y' → (x' - f₁.1)^2 + (y' - f₁.2)^2 = (x' - f₂.1)^2 + (y' - f₂.2)^2)
  (reciprocal_eccentricity : ∀ e_h e_e : ℝ, (∀ x y, hyperbola x y → e_h = Real.sqrt (1 + 3)) →
                             (∀ x y, ellipse x y → e_e * e_h = 1)) :
  ∀ x y, ellipse x y ↔ x^2/16 + y^2/12 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_with_reciprocal_eccentricity_l1351_135171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_truncated_pyramid_l1351_135105

/-- The volume of a regular truncated quadrilateral pyramid with base sides a and b,
    where the lateral surface area is half of the total surface area. -/
theorem volume_truncated_pyramid (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let lateral_area := 2 * (a + b) * ((a^2 + b^2) / (2 * (a + b)))
  let total_area := a^2 + b^2 + lateral_area
  lateral_area = total_area / 2 →
  (a * b * (a^2 + a*b + b^2)) / (3 * (a + b)) =
    (1/3) * (a*b / (a+b)) * (a^2 + b^2 + a*b) := by
  sorry

#check volume_truncated_pyramid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_truncated_pyramid_l1351_135105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l1351_135169

-- Define the curves C₁ and C₂
noncomputable def C₁ (θ : Real) : Real × Real :=
  (Real.sqrt 5 * Real.cos θ, Real.sqrt 5 * Real.sin θ)

noncomputable def C₂ (t : Real) : Real × Real :=
  (1 - (Real.sqrt 2 / 2) * t, -(Real.sqrt 2 / 2) * t)

-- Define the domain for θ
def θ_domain (θ : Real) : Prop :=
  0 ≤ θ ∧ θ ≤ Real.pi / 2

-- State the theorem
theorem intersection_point :
  ∃! p : Real × Real, 
    (∃ θ, θ_domain θ ∧ C₁ θ = p) ∧ 
    (∃ t, C₂ t = p) ∧
    p = (2, 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l1351_135169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_patio_ratio_is_four_to_one_l1351_135134

/-- Represents a rectangular patio with given dimensions -/
structure RectangularPatio where
  perimeter : ℝ
  length : ℝ

/-- Calculates the width of a rectangular patio given its perimeter and length -/
noncomputable def calculateWidth (patio : RectangularPatio) : ℝ :=
  (patio.perimeter - 2 * patio.length) / 2

/-- Calculates the ratio of length to width for a rectangular patio -/
noncomputable def lengthToWidthRatio (patio : RectangularPatio) : ℝ :=
  patio.length / (calculateWidth patio)

/-- Theorem: The ratio of length to width for a rectangular patio with 
    perimeter 100 feet and length 40 feet is 4:1 -/
theorem patio_ratio_is_four_to_one :
  let patio : RectangularPatio := { perimeter := 100, length := 40 }
  lengthToWidthRatio patio = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_patio_ratio_is_four_to_one_l1351_135134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_asymptote_intersection_l1351_135175

/-- The function f(x) = (x^2 + 4x - 5) / (x^2 + 4x + 5) -/
noncomputable def f (x : ℝ) : ℝ := (x^2 + 4*x - 5) / (x^2 + 4*x + 5)

/-- The horizontal asymptote of f(x) as x approaches infinity -/
def horizontal_asymptote : ℝ := 1

/-- Theorem: There is no real point of intersection of the asymptotes of f(x) -/
theorem no_real_asymptote_intersection :
  ¬ ∃ (x y : ℝ), (∀ ε > 0, ∃ N, ∀ t > N, |f t - y| < ε) ∧
                 (∃ c, ∀ ε > 0, ∃ δ > 0, ∀ t, |t - c| < δ → |f t| > 1/ε) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_asymptote_intersection_l1351_135175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_tangent_and_g_monotone_l1351_135192

noncomputable section

/-- A function f with parameters a, b, and c. -/
def f (a b c : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (a/2) * x^2 + b * x + c

/-- The derivative of f with respect to x. -/
def f_derivative (a b : ℝ) (x : ℝ) : ℝ := x^2 - a * x + b

/-- A function g defined in terms of f. -/
def g (a b c : ℝ) (x : ℝ) : ℝ := f a b c x + 2 * x

/-- The derivative of g with respect to x. -/
def g_derivative (a : ℝ) (x : ℝ) : ℝ := x^2 - a * x + 2

theorem f_tangent_and_g_monotone (a b c : ℝ) :
  (f_derivative a b 0 = 1) ∧
  (f a b c 0 = 1) ∧
  (∀ x : ℝ, g_derivative a x ≥ 0) →
  (b = 0 ∧ c = 1 ∧ a ≤ 2 * Real.sqrt 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_tangent_and_g_monotone_l1351_135192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nonzero_integer_solutions_l1351_135123

theorem nonzero_integer_solutions :
  {(a, b) : ℤ × ℤ | (a ≠ 0 ∨ b ≠ 0) ∧ (a^2 + b) * (a + b^2) = (a - b)^2} =
  {(0, 1), (1, 0), (-1, -1), (2, -1), (-1, 2)} := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nonzero_integer_solutions_l1351_135123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_range_l1351_135158

/-- The ellipse (C) with equation x^2/4 + y^2/3 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The left vertex of the ellipse -/
def A₁ : ℝ × ℝ := (-2, 0)

/-- The right vertex of the ellipse -/
def A₂ : ℝ × ℝ := (2, 0)

/-- The slope of line PA₂ is in the range [-2, -1] -/
def slope_PA₂_range (a b : ℝ) : Prop := -2 ≤ b/(a-2) ∧ b/(a-2) ≤ -1

/-- The theorem to be proved -/
theorem ellipse_slope_range {a b : ℝ} :
  ellipse_C a b →
  a ≠ 2 →
  a ≠ -2 →
  slope_PA₂_range a b →
  3/8 ≤ b/(a+2) ∧ b/(a+2) ≤ 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_range_l1351_135158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diana_rate_is_four_l1351_135146

/-- Diana's rate of doing situps per minute -/
def diana_rate : ℝ := sorry

/-- Hani's rate of doing situps per minute -/
def hani_rate : ℝ := sorry

/-- The time Diana spent doing situps -/
def diana_time : ℝ := sorry

/-- The relation between Hani's and Diana's rates -/
axiom hani_rate_def : hani_rate = diana_rate + 3

/-- Diana's total situps -/
axiom diana_total : diana_rate * diana_time = 40

/-- Total situps done by both -/
axiom total_situps : diana_rate * diana_time + hani_rate * diana_time = 110

theorem diana_rate_is_four : diana_rate = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diana_rate_is_four_l1351_135146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_and_g_min_l1351_135142

open Real

/-- The function f(x) -/
noncomputable def f (θ : ℝ) (x : ℝ) : ℝ := cos x * cos (x - θ) - 1/2 * cos θ

/-- The function g(x) -/
noncomputable def g (θ : ℝ) (x : ℝ) : ℝ := 2 * f θ (3/2 * x)

theorem f_max_and_g_min (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π) :
  (∀ x, f θ x ≤ f θ (π/3)) →
  θ = 2*π/3 ∧ ∀ x ∈ Set.Icc 0 (π/3), g θ x ≥ -1/2 ∧ ∃ y ∈ Set.Icc 0 (π/3), g θ y = -1/2 :=
by
  sorry

#check f_max_and_g_min

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_and_g_min_l1351_135142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_pi_equals_zero_l1351_135125

theorem sin_two_pi_equals_zero : 
  (∀ x : ℝ, Real.sin (x + 2*Real.pi) = Real.sin x) → Real.sin (2*Real.pi) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_pi_equals_zero_l1351_135125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_bound_implies_a_range_l1351_135135

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -Real.cos x - Real.sin x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := Real.sin x - Real.cos x

-- State the theorem
theorem derivative_bound_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc (Real.pi / 2) Real.pi, f' x < a) →
  a ∈ Set.Ioi (Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_bound_implies_a_range_l1351_135135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_two_l1351_135165

theorem least_number_with_remainder_two : ∃! n : ℕ, 
  n > 1 ∧
  (∀ d : ℕ, d ∈ ({3, 4, 5, 6, 7} : Finset ℕ) → n % d = 2) ∧
  (∀ m : ℕ, m > 1 ∧ (∀ d : ℕ, d ∈ ({3, 4, 5, 6, 7} : Finset ℕ) → m % d = 2) → m ≥ n) ∧
  n = 422 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_two_l1351_135165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_in_unit_disk_l1351_135180

variable (a : ℝ)
variable (z : ℂ)

def f (z : ℂ) : ℂ := z^2
noncomputable def φ (z : ℂ) : ℂ := -a * Complex.exp z

theorem two_roots_in_unit_disk (h : 0 < a ∧ a < Real.exp (-1)) :
  ∃! (S : Finset ℂ), S.card = 2 ∧ (∀ z ∈ S, Complex.abs z < 1 ∧ z^2 = a * Complex.exp z) := by
  sorry

#check two_roots_in_unit_disk

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_in_unit_disk_l1351_135180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_is_500_l1351_135114

/-- Calculates the overall profit from selling a refrigerator and a mobile phone -/
def overall_profit (refrigerator_cost mobile_cost : ℕ) 
  (refrigerator_loss_percent mobile_profit_percent : ℚ) : ℚ :=
  let refrigerator_loss := (refrigerator_loss_percent / 100) * refrigerator_cost
  let refrigerator_sell := refrigerator_cost - refrigerator_loss
  let mobile_profit := (mobile_profit_percent / 100) * mobile_cost
  let mobile_sell := mobile_cost + mobile_profit
  let total_cost := refrigerator_cost + mobile_cost
  let total_sell := refrigerator_sell + mobile_sell
  total_sell - total_cost

/-- Proves that the overall profit is 500 for the given scenario -/
theorem profit_is_500 : 
  overall_profit 15000 8000 2 10 = 500 := by
  -- Unfold the definition of overall_profit
  unfold overall_profit
  -- Simplify the arithmetic expressions
  simp [Nat.cast_add, Nat.cast_sub]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_is_500_l1351_135114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_wins_l1351_135178

/-- Represents the game board --/
structure GameBoard where
  rows : Nat
  cols : Nat

/-- Represents a player in the game --/
inductive Player where
  | Petya
  | Vasya

/-- Represents a move in the game --/
structure Move where
  row : Nat
  col : Nat
  direction : Bool  -- True for horizontal, False for vertical

/-- Defines the game state --/
structure GameState where
  board : GameBoard
  currentPlayer : Player
  movesPlayed : List Move

/-- Checks if a move is valid --/
def isValidMove (state : GameState) (move : Move) : Bool :=
  sorry

/-- Applies a move to the game state --/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if the game is over --/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Determines the winner of the game --/
def getWinner (state : GameState) : Option Player :=
  sorry

/-- Petya's winning strategy --/
def petyaStrategy (state : GameState) : Move :=
  sorry

/-- Simulates the game using the given strategies for both players --/
def playGame (initialState : GameState) (petyaStrategy : GameState → Move) (vasyaStrategy : GameState → Move) : GameState :=
  sorry

/-- Theorem stating that Petya has a winning strategy --/
theorem petya_wins (initialState : GameState) :
  initialState.board.rows = 5 ∧ initialState.board.cols = 9 ∧ initialState.currentPlayer = Player.Petya →
  ∃ (strategy : GameState → Move),
    ∀ (vasyaStrategy : GameState → Move),
      getWinner (playGame initialState strategy vasyaStrategy) = some Player.Petya := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_wins_l1351_135178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_inequality_l1351_135110

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the lengths of sides
noncomputable def side_length (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the vector from one point to another
def vector (p q : ℝ × ℝ) : ℝ × ℝ :=
  (q.1 - p.1, q.2 - p.2)

-- Define the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

-- Define the dot product of two vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Define the scalar multiplication of a vector
def scalar_mult (t : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (t * v.1, t * v.2)

-- Define the theorem
theorem triangle_vector_inequality (ABC : Triangle) (t : ℝ) :
  side_length ABC.A ABC.B = Real.sqrt 3 →
  side_length ABC.B ABC.C = 2 →
  side_length ABC.A ABC.C = 1 →
  (0 ≤ t ∧ t ≤ 3/2) ↔ 
  magnitude (vector ABC.B ABC.A - scalar_mult t (vector ABC.B ABC.C)) ≤ Real.sqrt 3 * magnitude (vector ABC.A ABC.C) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_inequality_l1351_135110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binet_formula_l1351_135133

noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def φ_hat : ℝ := (1 - Real.sqrt 5) / 2

def fib : ℕ → ℝ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem binet_formula (n : ℕ) : fib n = (φ^n - φ_hat^n) / Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binet_formula_l1351_135133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1351_135109

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) (m n : ℝ × ℝ) :
  -- Triangle conditions
  0 < A ∧ A < 2 * Real.pi / 3 →
  c^2 = a^2 + b^2 - a*b →
  -- Tangent condition
  Real.tan A - Real.tan B = (Real.sqrt 3 / 2) * (1 + Real.tan A * Real.tan B) →
  -- Vector definitions
  m = (Real.sin A, 1) →
  n = (3, Real.cos (2 * A)) →
  -- Conclusions
  B = Real.pi / 4 ∧
  (∀ x : ℝ, m.1 * n.1 + m.2 * n.2 ≤ 17/8) ∧
  (∃ x : ℝ, m.1 * n.1 + m.2 * n.2 = 17/8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1351_135109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_in_ratio_triangle_l1351_135148

theorem smallest_angle_in_ratio_triangle : 
  ∀ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 180 →
  (a : ℝ) / 4 = (b : ℝ) / 5 ∧ (b : ℝ) / 5 = (c : ℝ) / 9 →
  min a (min b c) = 40 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_in_ratio_triangle_l1351_135148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_distances_l1351_135141

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Sum of distances from one point to a list of points -/
noncomputable def sum_distances (p : Point) (points : List Point) : ℝ :=
  points.foldl (fun acc q => acc + distance p q) 0

theorem max_sum_distances
  (points : List Point)
  (petya : Point)
  (h_count : points.length = 99)
  (h_sum : sum_distances petya points = 1000) :
  ∀ vasya : Point, sum_distances vasya points ≤ 99000 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_distances_l1351_135141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_quadratics_common_point_l1351_135118

/-- Given three quadratic polynomials with pairwise distinct leading coefficients,
    if the graphs of any two of them have exactly one common point,
    then all three graphs have exactly one common point. -/
theorem three_quadratics_common_point
  (p₁ p₂ p₃ : ℝ → ℝ)
  (h₁ : ∃ a₁ b₁ c₁, ∀ x, p₁ x = a₁ * x^2 + b₁ * x + c₁)
  (h₂ : ∃ a₂ b₂ c₂, ∀ x, p₂ x = a₂ * x^2 + b₂ * x + c₂)
  (h₃ : ∃ a₃ b₃ c₃, ∀ x, p₃ x = a₃ * x^2 + b₃ * x + c₃)
  (h_distinct : ∀ a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃, 
                (∀ x, p₁ x = a₁ * x^2 + b₁ * x + c₁) →
                (∀ x, p₂ x = a₂ * x^2 + b₂ * x + c₂) →
                (∀ x, p₃ x = a₃ * x^2 + b₃ * x + c₃) →
                a₁ ≠ a₂ ∧ a₂ ≠ a₃ ∧ a₁ ≠ a₃)
  (h_common₁₂ : ∃! x, p₁ x = p₂ x)
  (h_common₂₃ : ∃! x, p₂ x = p₃ x)
  (h_common₁₃ : ∃! x, p₁ x = p₃ x) :
  ∃! x, p₁ x = p₂ x ∧ p₂ x = p₃ x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_quadratics_common_point_l1351_135118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_turn_brings_to_coincidence_l1351_135124

/-- A planar figure in 2D space -/
structure PlanarFigure where
  -- Add necessary fields to represent a planar figure
  coords : ℝ × ℝ

/-- Represents a symmetry axis in 2D space -/
structure SymmetryAxis where
  -- Add necessary fields to represent a symmetry axis
  direction : ℝ × ℝ

/-- Represents a rotation in 3D space -/
structure Rotation where
  axis : ℝ × ℝ × ℝ
  angle : ℝ

/-- Predicate to check if two figures are symmetrical -/
def are_symmetrical (f1 f2 : PlanarFigure) : Prop :=
  sorry

/-- Predicate to check if two figures are symmetrically positioned -/
def are_symmetrically_positioned (f1 f2 : PlanarFigure) : Prop :=
  sorry

/-- Function to apply a rotation to a planar figure -/
noncomputable def apply_rotation (f : PlanarFigure) (r : Rotation) : PlanarFigure :=
  sorry

/-- Predicate to check if two figures coincide -/
def coincide (f1 f2 : PlanarFigure) : Prop :=
  sorry

/-- Theorem stating that a half-turn rotation brings symmetrical figures to coincide -/
theorem half_turn_brings_to_coincidence (f1 f2 : PlanarFigure) 
  (h1 : are_symmetrical f1 f2) 
  (h2 : are_symmetrically_positioned f1 f2) :
  ∃ (r : Rotation), r.angle = Real.pi ∧ coincide (apply_rotation f1 r) f2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_turn_brings_to_coincidence_l1351_135124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l1351_135166

theorem election_votes (candidate1_percentage : ℚ) (candidate2_votes : ℕ) :
  candidate1_percentage = 70 →
  candidate2_votes = 240 →
  ∃ total_votes : ℕ,
    (candidate1_percentage / 100 * total_votes = (total_votes - candidate2_votes)) ∧
    (total_votes = 800) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l1351_135166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_walk_time_l1351_135164

/-- The time in minutes to walk to the bus stop at the usual speed -/
noncomputable def usual_time : ℝ := 36

/-- The ratio of the slower speed to the usual speed -/
noncomputable def speed_ratio : ℝ := 4/5

/-- The additional time taken when walking at the slower speed -/
noncomputable def additional_time : ℝ := 9

theorem bus_stop_walk_time : 
  speed_ratio * (usual_time + additional_time) = usual_time := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_walk_time_l1351_135164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bound_l1351_135120

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a square with side length 1
def UnitSquare : Set Point := {p : Point | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1}

-- Define a function to calculate the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

-- Theorem statement
theorem triangle_area_bound (points : Finset Point) :
  points.card = 9 → (∀ p ∈ points, p ∈ UnitSquare) →
  ∃ p1 p2 p3, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ triangleArea p1 p2 p3 ≤ 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bound_l1351_135120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_paying_customers_per_week_l1351_135100

/-- The number of people who pay taxes in a store every week, given the percentage of non-tax-paying customers and daily shoppers. -/
def taxPayingCustomersPerWeek (nonTaxPayingPercentage : ℚ) (dailyShoppers : ℕ) : ℕ :=
  ⌊((1 - nonTaxPayingPercentage / 100) * dailyShoppers * 7 : ℚ)⌋.toNat

/-- Theorem stating that given 6% of customers do not pay tax and 1000 people shop every day,
    the number of people who pay taxes in the store every week is 6580. -/
theorem tax_paying_customers_per_week :
  taxPayingCustomersPerWeek (6 : ℚ) 1000 = 6580 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_paying_customers_per_week_l1351_135100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_words_on_sunday_l1351_135147

def word_limit : ℕ := 1000
def words_on_saturday : ℕ := 450
def words_over_limit : ℕ := 100

theorem words_on_sunday : 
  word_limit + words_over_limit - words_on_saturday = 650 := by
  -- Proof goes here
  sorry

#eval word_limit + words_over_limit - words_on_saturday

end NUMINAMATH_CALUDE_ERRORFEEDBACK_words_on_sunday_l1351_135147
