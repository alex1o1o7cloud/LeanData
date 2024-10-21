import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l341_34141

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x / (1 + Real.sqrt x)

-- Define the area
noncomputable def area : ℝ := ∫ x in Set.Icc 0 1, f x

-- Theorem statement
theorem area_calculation : area = 5/3 - 2 * Real.log 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l341_34141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l341_34125

def a : ℕ → ℚ
  | 0 => 1  -- Add a case for 0
  | 1 => 1
  | n + 2 => 2^(n+1) * (a (n+1))^2 - a (n+1) + 1 / 2^(n+2)

theorem a_general_term (n : ℕ) (h : n ≥ 1) : 
  a n = 3^(2^(n-1)) / 2^n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l341_34125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l341_34187

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (Real.sin x - Real.cos x)^2 - 1

-- State the theorem
theorem smallest_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ x, f (x + p) = f x) ∧
  (∀ q, q > 0 ∧ (∀ x, f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l341_34187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eighth_term_l341_34152

/-- A sequence satisfying the given recurrence relation -/
def satisfiesRecurrence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → 2 * a n + a (n + 1) = 0

theorem sequence_eighth_term (a : ℕ → ℝ) (h : satisfiesRecurrence a) (h3 : a 3 = -2) :
  a 8 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eighth_term_l341_34152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_one_implies_x_eight_l341_34166

theorem cube_root_sum_one_implies_x_eight (x : ℝ) (hx : x > 0) 
  (h : (1 - x^4)^(1/3) + (1 + x^4)^(1/3) = 1) : 
  x^8 = 28/27 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_one_implies_x_eight_l341_34166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_simplification_g_monotonic_increase_l341_34134

open Real

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := (1/2) * sin (2*x + π/3)

-- Define the shifted function g
noncomputable def g (x : ℝ) : ℝ := (1/2) * sin (2*(x + π/3) + π/3)

-- Simplify g to its actual form
theorem g_simplification (x : ℝ) : g x = -(1/2) * sin (2*x) := by
  sorry

-- Theorem stating the interval of monotonic increase for g
theorem g_monotonic_increase (k : ℤ) :
  StrictMonoOn g (Set.Icc (k * π + π/4) (k * π + 3*π/4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_simplification_g_monotonic_increase_l341_34134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_tangent_condition_l341_34121

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1/2 * x^2 - a*x + Real.log x

/-- Predicate for having a vertical tangent at a point -/
def HasVerticalTangentAt (f : ℝ → ℝ) (x : ℝ) : Prop := sorry

/-- Theorem stating the condition for the existence of a vertical tangent line -/
theorem vertical_tangent_condition (a : ℝ) : 
  (∃ x > 0, HasVerticalTangentAt (f a) x) ↔ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_tangent_condition_l341_34121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_meters_percentage_l341_34165

/-- The percentage of defective meters, given the number of defective meters and the total number of meters examined. -/
noncomputable def percentage_defective (defective : ℕ) (total : ℕ) : ℝ :=
  (defective : ℝ) / (total : ℝ) * 100

/-- Theorem stating that the percentage of defective meters is 0.08% when 2 out of 2500 meters are rejected. -/
theorem defective_meters_percentage :
  percentage_defective 2 2500 = 0.08 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_meters_percentage_l341_34165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_l341_34170

def A : ℝ × ℝ := (12, 0)
def B : ℝ × ℝ := (0, 0)
def D : ℝ × ℝ := (6, 8)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem sum_of_distances : distance A D + distance B D = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_l341_34170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l341_34174

theorem problem_1 : (Real.sqrt 2 - 1)^0 - Real.sqrt 6 / Real.sqrt 3 - (-1/2)⁻¹ + |(-(Real.sqrt 2))| = 3 := by sorry

theorem problem_2 : 
  let f (x : ℝ) := 2*x^2 - 3*x - 5
  ∀ x, f x = 0 ↔ (x = 5/2 ∨ x = -1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l341_34174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shelby_driving_time_in_rain_l341_34107

/-- Represents Shelby's driving scenario -/
structure DrivingScenario where
  speed_sunny : ℚ  -- Speed when not raining (mph)
  speed_rainy : ℚ  -- Speed when raining (mph)
  total_distance : ℚ  -- Total distance driven (miles)
  total_time : ℚ  -- Total time driven (minutes)

/-- Calculates the time spent driving in the rain -/
def time_in_rain (scenario : DrivingScenario) : ℚ :=
  let speed_sunny_per_min := scenario.speed_sunny / 60
  let speed_rainy_per_min := scenario.speed_rainy / 60
  (scenario.total_distance - speed_sunny_per_min * scenario.total_time) / 
  (speed_rainy_per_min - speed_sunny_per_min)

/-- Theorem stating that the time spent driving in the rain is approximately 34 minutes -/
theorem shelby_driving_time_in_rain :
  let scenario := DrivingScenario.mk 40 25 25 50
  let rain_time := time_in_rain scenario
  (⌊rain_time + 1/2⌋ : ℤ) = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shelby_driving_time_in_rain_l341_34107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l341_34185

noncomputable def f (x : ℝ) : ℝ := (4^x - 1) / (4^x + 1)

theorem f_properties :
  (∀ x : ℝ, f x < 1/3 ↔ x < 1/2) ∧
  (Set.range f = Set.Ioo (-1) 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l341_34185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l341_34129

-- Define the solution set of the inequality
def solution_set (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < 2 ∧ x^2 - (a+1)*x + a < 0}

-- Define the condition for x and y
def condition (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ 2/x + 1/y = 1

-- Define the inequality that always holds
def inequality (x y k : ℝ) : Prop := x + 2*y ≥ Real.sqrt k - 9/Real.sqrt k

theorem problem_solution :
  (∀ x, x ∈ solution_set 2) ∧
  (∀ x y k, condition x y → inequality x y k) →
  (2 ∈ Set.Icc 0 81) ∧ (∀ k, k > 0 ∧ k ≤ 81 → (∀ x y, condition x y → inequality x y k)) :=
by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l341_34129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_teeth_count_l341_34169

/-- The number of teeth a dog has -/
def dog_teeth : ℕ := 42

/-- The number of teeth a cat has -/
def cat_teeth : ℕ := 30

/-- The number of teeth a pig has -/
def pig_teeth : ℕ := 28

/-- The number of dogs -/
def num_dogs : ℕ := 5

/-- The number of cats -/
def num_cats : ℕ := 10

/-- The number of pigs -/
def num_pigs : ℕ := 7

/-- The total number of teeth for all animals -/
def total_teeth : ℕ := 706

theorem dog_teeth_count : dog_teeth = 42 :=
by
  -- State the equation
  have h1 : num_dogs * dog_teeth + num_cats * cat_teeth + num_pigs * pig_teeth = total_teeth := by sorry
  
  -- Solve for dog_teeth
  have h2 : num_dogs * dog_teeth = total_teeth - (num_cats * cat_teeth + num_pigs * pig_teeth) := by sorry
  have h3 : dog_teeth = (total_teeth - (num_cats * cat_teeth + num_pigs * pig_teeth)) / num_dogs := by sorry
  
  -- Evaluate the right-hand side
  have h4 : (total_teeth - (num_cats * cat_teeth + num_pigs * pig_teeth)) / num_dogs = 42 := by sorry
  
  -- Conclude
  exact h4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_teeth_count_l341_34169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_constant_sum_l341_34106

noncomputable section

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the constant c
def c : ℝ := 1/8

-- Define a point on the parabola
def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, parabola x)

-- Define the point C
def C : ℝ × ℝ := (0, c)

-- Define the tangent slope at a point on the parabola
def tangent_slope (x : ℝ) : ℝ := 2 * x

-- Define a predicate for a chord being perpendicular to the tangent
def is_perpendicular_chord (A B : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  (y₂ - y₁) / (x₂ - x₁) * tangent_slope x₁ = -1 ∨
  (y₂ - y₁) / (x₂ - x₁) * tangent_slope x₂ = -1

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := P
  let (x₂, y₂) := Q
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Define the t value for a chord
noncomputable def t_value (A B : ℝ × ℝ) : ℝ :=
  1 / (distance A C)^2 + 1 / (distance B C)^2

end noncomputable section

-- State the theorem
theorem chord_constant_sum :
  ∀ (A B : ℝ × ℝ),
    A.2 = parabola A.1 →
    B.2 = parabola B.1 →
    (A.1 - C.1) * (B.1 - C.1) < 0 →
    is_perpendicular_chord A B →
    t_value A B = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_constant_sum_l341_34106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l341_34199

noncomputable section

open Real

theorem triangle_ABC_properties (A B C : ℝ) (a b c : ℝ) :
  -- Define the triangle
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given conditions
  sin A * sin B * cos B + sin B ^ 2 * cos A = 2 * sqrt 2 * sin C * cos B ∧
  b = 2 ∧
  (1 / 2) * a * c * sin B = sqrt 2 →
  -- Prove
  tan B = 2 * sqrt 2 ∧
  a + c = 2 * sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l341_34199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_y_payment_is_290_l341_34143

/-- The weekly payment for employee Y, given that:
    1. The total payment for X and Y is 638.
    2. X's payment is 120% of Y's payment. -/
noncomputable def employee_y_payment : ℚ :=
  let total_payment : ℚ := 638
  let x_payment_ratio : ℚ := 12/10
  total_payment / (1 + x_payment_ratio)

theorem employee_y_payment_is_290 : employee_y_payment = 290 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_y_payment_is_290_l341_34143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_points_at_sqrt2_distance_specific_points_on_line_l341_34147

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A line parameterized by t -/
def line (t : ℝ) : Point :=
  { x := 3 - t, y := 4 + t }

/-- The fixed point P -/
def P : Point :=
  { x := 3, y := 4 }

theorem line_points_at_sqrt2_distance :
  ∀ t : ℝ, distance (line t) P = Real.sqrt 2 ↔ t = 1 ∨ t = -1 := by
  sorry

theorem specific_points_on_line :
  { p : Point | ∃ t : ℝ, p = line t ∧ distance p P = Real.sqrt 2 } =
  { { x := 4, y := 3 }, { x := 2, y := 5 } } := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_points_at_sqrt2_distance_specific_points_on_line_l341_34147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_nine_terms_equals_511_512_l341_34167

/-- The sum of the first n terms of a geometric sequence with first term a and common ratio r -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

/-- The infinite geometric sequence {1, 1/2, 1/4, 1/8, ...} -/
noncomputable def our_sequence (n : ℕ) : ℝ := (1/2)^n

theorem sum_of_nine_terms_equals_511_512 :
  ∃ n : ℕ, geometric_sum 1 (1/2) n = 511/512 ∧ n = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_nine_terms_equals_511_512_l341_34167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_x_in_terms_of_a_and_b_l341_34161

theorem tan_x_in_terms_of_a_and_b 
  (a b x : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : 0 < x) 
  (h4 : x < π/2) 
  (h5 : Real.sin x = (2*a*b) / Real.sqrt (4*a^2*b^2 + (a^2 + b^2)^2)) : 
  Real.tan x = (2*a*b) / (a^2 + b^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_x_in_terms_of_a_and_b_l341_34161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l341_34150

-- Define a structure for a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  angleSum : A + B + C = π
  side_a : a > 0
  side_b : b > 0
  side_c : c > 0

-- Define the main theorem
theorem triangle_property (t : Triangle) 
  (h : (t.b / Real.cos t.B) + (t.c / Real.cos t.C) = (t.a / Real.cos t.A) + (3 * t.a / (Real.cos t.B * Real.cos t.C))) :
  (Real.tan t.B * Real.tan t.C = 1/2) ∧ 
  (∀ A' : Real, A' = t.A → Real.tan A' ≤ -2 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l341_34150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_inequality_l341_34193

theorem sine_sum_inequality (α₁ α₂ α₃ : ℝ) 
  (h₁ : 0 ≤ α₁ ∧ α₁ ≤ π) 
  (h₂ : 0 ≤ α₂ ∧ α₂ ≤ π) 
  (h₃ : 0 ≤ α₃ ∧ α₃ ≤ π) : 
  Real.sin α₁ + Real.sin α₂ + Real.sin α₃ ≤ 3 * Real.sin ((α₁ + α₂ + α₃) / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_inequality_l341_34193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_circumcenters_l341_34176

-- Define the points
variable (A B C D E F G H O₁ O₂ O₃ O₄ : EuclideanSpace ℝ (Fin 2))

-- Define the parallelogram ABCD
def is_parallelogram (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  (B - A) = (D - C) ∧ (C - B) = (A - D)

-- Define the points on the sides of the parallelogram
def on_side (P Q R : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ R = (1 - t) • P + t • Q

-- Define the circumcenter of a triangle
def is_circumcenter (O P Q R : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist O P = dist O Q ∧ dist O Q = dist O R

-- State the theorem
theorem parallelogram_circumcenters 
  (h_parallelogram : is_parallelogram A B C D)
  (h_E : on_side E A B)
  (h_F : on_side F B C)
  (h_G : on_side G C D)
  (h_H : on_side H D A)
  (h_O₁ : is_circumcenter O₁ A E H)
  (h_O₂ : is_circumcenter O₂ B E F)
  (h_O₃ : is_circumcenter O₃ C G F)
  (h_O₄ : is_circumcenter O₄ D G H) :
  is_parallelogram O₁ O₂ O₃ O₄ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_circumcenters_l341_34176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goat_rope_problem_l341_34190

theorem goat_rope_problem (rope_length tree_radius stake_distance : Real) : 
  rope_length = 4.7 ∧ 
  tree_radius = 0.5 ∧ 
  stake_distance = 1 →
  2 * tree_radius * Real.sqrt 8 + tree_radius * (Real.pi + 2 * Real.arcsin (1/3)) > rope_length := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goat_rope_problem_l341_34190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_weight_range_l341_34144

/-- Represents the boat's characteristics and water conditions -/
structure BoatConditions where
  length : ℝ
  breadth : ℝ
  waterDepth : ℝ
  sinkRate : ℝ
  minSink : ℝ
  maxSink : ℝ
  minWaterBelow : ℝ

/-- Defines the valid range for boat conditions -/
def validBoatConditions (bc : BoatConditions) : Prop :=
  3 ≤ bc.length ∧ bc.length ≤ 5 ∧
  2 ≤ bc.breadth ∧ bc.breadth ≤ 3 ∧
  1 ≤ bc.waterDepth ∧ bc.waterDepth ≤ 2 ∧
  bc.sinkRate = 0.1 ∧
  bc.minSink = 0.03 ∧
  bc.maxSink = 0.06 ∧
  bc.minWaterBelow = 0.5

/-- Calculates the weight of the man based on how much the boat sinks -/
noncomputable def calculateWeight (sinkDepth : ℝ) (sinkRate : ℝ) : ℝ :=
  sinkDepth / sinkRate

/-- Theorem stating the minimum and maximum weight range of the man -/
theorem man_weight_range (bc : BoatConditions) 
  (h : validBoatConditions bc) :
  calculateWeight bc.minSink bc.sinkRate = 30 ∧
  calculateWeight bc.maxSink bc.sinkRate = 60 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_weight_range_l341_34144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_hyperbola_l341_34103

/-- Given two conic sections C1 and C2 with common foci, prove the eccentricity of C2 -/
theorem eccentricity_of_hyperbola (a b a1 b1 : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a1 > 0) (h4 : b1 > 0) :
  let C1 := {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1}
  let C2 := {(x, y) : ℝ × ℝ | x^2 / a1^2 - y^2 / b1^2 = 1}
  let F1 := (0 : ℝ × ℝ) -- Placeholder for focus 1
  let F2 := (0 : ℝ × ℝ) -- Placeholder for focus 2
  let M := (0 : ℝ × ℝ) -- Placeholder for intersection point
  let e1 : ℝ := 3/4 -- Eccentricity of C1
  (∃ (x y : ℝ), (x, y) ∈ C1 ∩ C2 ∧ x > 0 ∧ y > 0) → -- C1 and C2 intersect in first quadrant
  ((F1.1 - M.1)^2 + (F1.2 - M.2)^2 + (F2.1 - M.1)^2 + (F2.2 - M.2)^2 = 
    ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2)) → -- ∠F1MF2 = 90°
  let e2 := Real.sqrt ((a1^2 - b1^2) / a1^2) -- Eccentricity of C2
  e2 = 3 * Real.sqrt 2 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_hyperbola_l341_34103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l341_34156

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = Real.sqrt 10 ∧
  t.a^2 + t.b^2 - t.c^2 = t.a * t.b * Real.sin t.C ∧
  t.a * Real.cos t.B + t.b * Real.sin t.A = t.c

-- Theorem to prove
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  Real.tan t.C = 2 ∧ (1/2 : Real) * t.a * t.b * Real.sin t.C = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l341_34156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_equation_l341_34153

/-- Given a parabola C: y^2 = 2px (p > 0) with focus F, and points P on C and Q on the x-axis such that:
    - O is the coordinate origin
    - PF is perpendicular to the x-axis
    - PQ is perpendicular to OP
    - |FQ| = 6
    Then the directrix equation of C is x = -3/2 -/
theorem parabola_directrix_equation (p : ℝ) (F P Q : ℝ × ℝ) :
  p > 0 →
  (∀ x y, y^2 = 2*p*x ↔ (x, y) ∈ Set.range (λ t : ℝ ↦ (t^2/(2*p), t))) →
  F = (p/2, 0) →
  P.2^2 = 2*p*P.1 →
  P.2 * (F.1 - P.1) = 0 →
  Q.2 = 0 →
  (P.2 - 0) * (Q.1 - 0) = -(P.1 - 0) * (Q.2 - P.2) →
  |F.1 - Q.1| = 6 →
  (λ x : ℝ ↦ x = -3/2) = (λ x : ℝ ↦ x = -p/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_equation_l341_34153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_park_fencing_cost_specific_park_fencing_cost_pence_l341_34124

/-- Represents a rectangular park with specific properties -/
structure RectangularPark where
  length : ℝ
  width : ℝ
  area : ℝ
  total_fencing_cost : ℝ
  side_ratio : ℝ
  (area_eq : area = length * width)
  (ratio_eq : length = side_ratio * width)
  (positive_dimensions : length > 0 ∧ width > 0)

/-- Calculates the fencing cost per meter for a given rectangular park -/
noncomputable def fencing_cost_per_meter (park : RectangularPark) : ℝ :=
  park.total_fencing_cost / (2 * (park.length + park.width))

/-- Theorem stating the fencing cost per meter for a specific rectangular park -/
theorem specific_park_fencing_cost :
  ∃ (park : RectangularPark),
    park.area = 3750 ∧
    park.side_ratio = 3/2 ∧
    park.total_fencing_cost = 200 ∧
    fencing_cost_per_meter park = 0.80 := by
  sorry

/-- Conversion from dollars to pence -/
def dollars_to_pence (dollars : ℝ) : ℝ :=
  dollars * 100

/-- Theorem stating the fencing cost per meter in pence for a specific rectangular park -/
theorem specific_park_fencing_cost_pence :
  ∃ (park : RectangularPark),
    park.area = 3750 ∧
    park.side_ratio = 3/2 ∧
    park.total_fencing_cost = 200 ∧
    dollars_to_pence (fencing_cost_per_meter park) = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_park_fencing_cost_specific_park_fencing_cost_pence_l341_34124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_variance_l341_34173

noncomputable def sample : List ℝ := [2, 3, 6, 6, 8]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m) ^ 2)).sum / xs.length

theorem sample_variance :
  mean sample = 5 → variance sample = 24/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_variance_l341_34173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_intersections_l341_34198

/-- Curve C₁ defined by its Cartesian equation -/
def C₁ (x y : ℝ) : Prop := y^2 - 8*x - 16 = 0

/-- Curve C₂ defined by its Cartesian equation -/
def C₂ (a x y : ℝ) : Prop := x = a*y

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem min_distance_between_intersections (a : ℝ) :
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    C₁ x₁ y₁ ∧ C₁ x₂ y₂ ∧
    C₂ a x₁ y₁ ∧ C₂ a x₂ y₂ ∧
    (∀ x₃ y₃ x₄ y₄ : ℝ,
      C₁ x₃ y₃ ∧ C₁ x₄ y₄ ∧ C₂ a x₃ y₃ ∧ C₂ a x₄ y₄ →
      distance x₁ y₁ x₂ y₂ ≤ distance x₃ y₃ x₄ y₄) ∧
    distance x₁ y₁ x₂ y₂ = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_intersections_l341_34198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_fourth_shiny_after_fifth_draw_l341_34160

def total_pennies : ℕ := 12
def shiny_pennies : ℕ := 5
def dull_pennies : ℕ := 7

def probability_more_than_five_draws : ℚ :=
  let favorable_outcomes := (Nat.choose shiny_pennies 3 * Nat.choose dull_pennies 2) +
                            (Nat.choose shiny_pennies 2 * Nat.choose dull_pennies 3) +
                            (Nat.choose shiny_pennies 1 * Nat.choose dull_pennies 4) +
                            (Nat.choose dull_pennies 4)
  let total_outcomes := Nat.choose total_pennies 5
  (favorable_outcomes : ℚ) / total_outcomes

theorem probability_fourth_shiny_after_fifth_draw :
  probability_more_than_five_draws = 35 / 36 := by
  sorry

#eval Nat.add (35 : ℕ) (36 : ℕ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_fourth_shiny_after_fifth_draw_l341_34160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cafe_meeting_probability_l341_34149

/-- The probability that two people meet at a café -/
theorem cafe_meeting_probability : ℝ := by
  let arrival_window : ℝ := 0.75
  let stay_duration : ℝ := 0.5
  let total_area : ℝ := arrival_window ^ 2
  let non_meeting_area : ℝ := (arrival_window - stay_duration) ^ 2 / 2
  let meeting_area : ℝ := total_area - non_meeting_area
  have h : meeting_area / total_area = 8 / 9 := by sorry
  exact 8 / 9


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cafe_meeting_probability_l341_34149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l341_34172

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def valid_subset (M : Finset ℕ) : Prop :=
  M ⊆ Finset.range 15 ∧
  ∀ a b c, a ∈ M → b ∈ M → c ∈ M → a ≠ b → b ≠ c → a ≠ c → ¬is_square (a * b * c)

theorem max_subset_size :
  ∃ M : Finset ℕ, valid_subset M ∧ M.card = 10 ∧
  ∀ N : Finset ℕ, valid_subset N → N.card ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l341_34172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_decreasing_l341_34123

-- Define the function f(x) = ln(1 - x^2)
noncomputable def f (x : ℝ) : ℝ := Real.log (1 - x^2)

-- Theorem stating that f is even and decreasing in (0, 1)
theorem f_even_and_decreasing :
  (∀ x, f (-x) = f x) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < 1 → f y < f x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_decreasing_l341_34123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_65_minus_12_plus_24_base7_l341_34114

/-- Represents a number in base 7 -/
structure Base7 where
  value : Nat
  valid : value < 7

/-- Calculates the units digit of a number in base 7 -/
def unitsDigitBase7 (n : Nat) : Nat :=
  n % 7

/-- Performs subtraction in base 7 -/
def subtractBase7 (a b : Nat) : Nat :=
  (a - b + 7 * ((a - b) / 7 + 1)) % 7

/-- Performs addition in base 7 -/
def addBase7 (a b : Nat) : Nat :=
  (a + b) % 7

/-- The main theorem to prove -/
theorem units_digit_65_minus_12_plus_24_base7 :
  unitsDigitBase7 (addBase7 (subtractBase7 65 12) 24) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_65_minus_12_plus_24_base7_l341_34114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_contents_with_one_seed_l341_34181

/-- Represents the possible contents of a bag -/
inductive BagContent
  | Poppy
  | Millet
  | Mixture
deriving Inhabited

/-- Represents a bag with a label and actual content -/
structure Bag where
  label : BagContent
  content : BagContent
deriving Inhabited

/-- Checks if all bags are incorrectly labeled -/
def all_incorrectly_labeled (bags : List Bag) : Prop :=
  ∀ bag ∈ bags, bag.label ≠ bag.content

/-- Checks if the contents of the bags are different -/
def different_contents (bags : List Bag) : Prop :=
  ∀ i j, i ≠ j → (bags.get! i).content ≠ (bags.get! j).content

/-- Represents the act of examining a single seed -/
def examine_seed (bag : Bag) : BagContent :=
  match bag.content with
  | BagContent.Mixture => BagContent.Mixture
  | content => content

/-- The main theorem -/
theorem identify_contents_with_one_seed 
  (bags : List Bag) 
  (h_count : bags.length = 3)
  (h_incorrect : all_incorrectly_labeled bags)
  (h_different : different_contents bags) :
  ∃ (examined_bag : Bag), 
    examined_bag ∈ bags ∧ 
    (∀ bag ∈ bags, ∃ content, bag.content = content) := by
  sorry

#check identify_contents_with_one_seed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_contents_with_one_seed_l341_34181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_fraction_l341_34131

-- Complex numbers are already defined in Mathlib, so we don't need to redefine them
-- The imaginary unit i is also already defined

-- Theorem statement
theorem imaginary_part_of_fraction :
  Complex.im ((3 - Complex.I) / (2 + Complex.I)) = -1 :=
by
  -- Rewrite the fraction
  have h1 : (3 - Complex.I) / (2 + Complex.I) = (3 - Complex.I) * (2 - Complex.I) / ((2 + Complex.I) * (2 - Complex.I)) := by sorry
  
  -- Simplify the numerator
  have h2 : (3 - Complex.I) * (2 - Complex.I) = 6 + 3*Complex.I - 2*Complex.I + Complex.I*Complex.I := by sorry
  have h3 : Complex.I*Complex.I = -1 := by sorry
  have h4 : (3 - Complex.I) * (2 - Complex.I) = 5 - 5*Complex.I := by sorry
  
  -- Simplify the denominator
  have h5 : (2 + Complex.I) * (2 - Complex.I) = 4 + 2*Complex.I - 2*Complex.I - Complex.I*Complex.I := by sorry
  have h6 : (2 + Complex.I) * (2 - Complex.I) = 5 := by sorry
  
  -- Combine the results
  have h7 : (3 - Complex.I) / (2 + Complex.I) = (5 - 5*Complex.I) / 5 := by sorry
  have h8 : (5 - 5*Complex.I) / 5 = 1 - Complex.I := by sorry
  
  -- Get the imaginary part
  have h9 : Complex.im (1 - Complex.I) = -1 := by sorry
  
  -- Conclude
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_fraction_l341_34131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_point_is_minimum_l341_34119

/-- The function representing the translated graph -/
def f (x : ℝ) : ℝ := 2 * abs (x - 4) - 2

/-- The minimum point of the translated graph -/
def min_point : ℝ × ℝ := (4, -2)

/-- Theorem stating that min_point is the minimum point of f -/
theorem min_point_is_minimum :
  ∀ x : ℝ, f (min_point.fst) ≤ f x ∧ 
  (f (min_point.fst) = f x → x = min_point.fst) := by
  sorry

#check min_point_is_minimum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_point_is_minimum_l341_34119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_l341_34175

/-- Definition of the line l -/
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  k * x - y + 1 + 2 * k = 0

/-- A point is in the fourth quadrant -/
def in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

/-- The line l does not pass through the fourth quadrant -/
def not_in_fourth_quadrant (k : ℝ) : Prop :=
  ∀ x y, line_l k x y → ¬in_fourth_quadrant x y

/-- Point A is on the negative x-axis -/
def point_A (k : ℝ) (x : ℝ) : Prop :=
  line_l k x 0 ∧ x < 0

/-- Point B is on the positive y-axis -/
def point_B (k : ℝ) (y : ℝ) : Prop :=
  line_l k 0 y ∧ y > 0

/-- Area of triangle AOB -/
noncomputable def area_AOB (k : ℝ) : ℝ :=
  let x_A := -(1 + 2 * k) / k
  let y_B := 1 + 2 * k
  (1 / 2) * abs x_A * y_B

theorem line_l_properties (k : ℝ) :
  not_in_fourth_quadrant k →
  (∃ x, point_A k x) →
  (∃ y, point_B k y) →
  (k ≥ 0 ∧
   (∀ k', area_AOB k' ≥ area_AOB k → area_AOB k ≥ 4) ∧
   (area_AOB k = 4 → k = 1/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_l341_34175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_greater_half_iff_S_n_bound_l341_34179

-- Define the sequence a_n
noncomputable def a : ℕ → ℝ → ℝ
  | 0, x => x
  | n + 1, x => (2 * (a n x)^2) / (4 * (a n x) - 1)

-- Define S_n as the sum of the first n terms
noncomputable def S (n : ℕ) (x : ℝ) : ℝ :=
  Finset.sum (Finset.range n) (λ i => a i x)

-- Theorem I
theorem a_n_greater_half_iff (x : ℝ) :
  (∀ n : ℕ, a (n + 1) x > 1/2) ↔ (x > 1/4 ∧ x ≠ 1/2) := by
  sorry

-- Theorem II
theorem S_n_bound (n : ℕ) :
  S n 1 < n^2 / 4 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_greater_half_iff_S_n_bound_l341_34179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_table_l341_34115

/-- A table with 4 rows and 6 columns --/
def Table := Fin 4 → Fin 6 → ℕ

/-- The sum of a row in the table --/
def row_sum (t : Table) (i : Fin 4) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin 6)) (λ j => t i j))

/-- The sum of a column in the table --/
def col_sum (t : Table) (j : Fin 6) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin 4)) (λ i => t i j))

/-- The total sum of all numbers in the table --/
def total_sum (t : Table) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin 4 × Fin 6)) (λ (i, j) => t i j)

/-- All row and column sums are different --/
def distinct_sums (t : Table) : Prop :=
  ∀ (i j : Fin 4) (k l : Fin 6), 
    (i ≠ j ∨ k ≠ l) → row_sum t i ≠ row_sum t j ∧ col_sum t k ≠ col_sum t l

theorem min_sum_table :
  ∃ (t : Table), distinct_sums t ∧ 
    (∀ (t' : Table), distinct_sums t' → total_sum t ≤ total_sum t') ∧
    total_sum t = 43 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_table_l341_34115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l341_34140

-- Define the function for n/m
noncomputable def f (a : ℝ) : ℝ := 4^(a + 18 / (2 * a + 1))

-- State the theorem
theorem min_value_of_f (a : ℝ) (h : a > 0) :
  ∀ x > 0, f x ≥ f (5/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l341_34140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_K_mass_percentage_in_KBrO3_l341_34104

-- Define the atomic masses
noncomputable def atomic_mass_K : ℝ := 39.10
noncomputable def atomic_mass_Br : ℝ := 79.90
noncomputable def atomic_mass_O : ℝ := 16.00

-- Define the composition of KBrO3
def K_count : ℕ := 1
def Br_count : ℕ := 1
def O_count : ℕ := 3

-- Define the molar mass of KBrO3
noncomputable def molar_mass_KBrO3 : ℝ :=
  K_count * atomic_mass_K + Br_count * atomic_mass_Br + O_count * atomic_mass_O

-- Define the mass percentage calculation
noncomputable def mass_percentage (element_mass : ℝ) (compound_mass : ℝ) : ℝ :=
  (element_mass / compound_mass) * 100

-- Theorem statement
theorem K_mass_percentage_in_KBrO3 :
  abs (mass_percentage atomic_mass_K molar_mass_KBrO3 - 23.41) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_K_mass_percentage_in_KBrO3_l341_34104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_iff_solution_set_l341_34128

-- Define the set of solutions
def solution_set : Set ℝ := Set.Ioo (-2) 2 ∪ Set.Ioi 4

-- Define the inequality
def inequality (x : ℝ) : Prop :=
  (x^2 + 1) / (x + 2) ≥ 3 / (x - 2) + 1

-- Theorem statement
theorem inequality_iff_solution_set :
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_iff_solution_set_l341_34128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_telegraph_current_intensity_approx_l341_34127

/-- Telegraph system parameters -/
structure TelegraphSystem where
  num_cells : ℕ
  distance : ℝ
  wire_diameter : ℝ
  unit_wire_resistance : ℝ
  cell_emf : ℝ
  cell_internal_resistance : ℝ
  morse_machine_equivalent_distance : ℝ

/-- Calculate the current intensity in the telegraph system -/
noncomputable def calculate_current_intensity (system : TelegraphSystem) : ℝ :=
  let total_distance := system.distance + system.morse_machine_equivalent_distance
  let wire_resistance := (total_distance * 1000 / (system.wire_diameter^2)) * system.unit_wire_resistance
  let total_internal_resistance := (system.num_cells : ℝ) * system.cell_internal_resistance
  let total_resistance := wire_resistance + total_internal_resistance
  let total_emf := (system.num_cells : ℝ) * system.cell_emf
  total_emf / total_resistance

/-- Theorem: The current intensity in the given telegraph system is approximately 0.572 Amperes -/
theorem telegraph_current_intensity_approx (ε : ℝ) (hε : ε > 0) :
  let system : TelegraphSystem := {
    num_cells := 60,
    distance := 79,
    wire_diameter := 5,
    unit_wire_resistance := 0.2,
    cell_emf := 1.079,
    cell_internal_resistance := 0.62,
    morse_machine_equivalent_distance := 16
  }
  |calculate_current_intensity system - 0.572| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_telegraph_current_intensity_approx_l341_34127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sin_cos_equation_solution_l341_34102

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the solution set
def solution_set : Set ℝ := {x | ∃ n : ℤ, 2 * Real.pi * n ≤ x ∧ x ≤ Real.pi / 2 + 2 * Real.pi * n}

-- State the theorem
theorem floor_sin_cos_equation_solution :
  {x : ℝ | floor (Real.sin x + Real.cos x) = 1} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sin_cos_equation_solution_l341_34102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_half_difference_l341_34183

theorem cosine_half_difference (α β : ℝ) : 
  Real.sin α + Real.sin β = -27/65 →
  Real.tan ((α + β)/2) = 7/9 →
  5*Real.pi/2 < α →
  α < 3*Real.pi →
  -Real.pi/2 < β →
  β < 0 →
  Real.cos ((α - β)/2) = 27/(7*Real.sqrt 130) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_half_difference_l341_34183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_digit_391_base8_is_6_l341_34122

/-- The first digit of a number in a given base is the coefficient of the highest power of the base that's less than or equal to the number. -/
def first_digit_base (n : ℕ) (base : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let k := Nat.log base n
    n / base ^ k

/-- Theorem: The first digit of 391 (base 10) in base 8 is 6. -/
theorem first_digit_391_base8_is_6 :
  first_digit_base 391 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_digit_391_base8_is_6_l341_34122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_alliances_required_l341_34155

/-- Represents an alliance of countries -/
structure Alliance where
  members : Finset Nat

/-- The set of all countries -/
def Countries : Finset Nat := Finset.range 100

/-- A set of alliances that satisfies the given conditions -/
structure AllianceSystem where
  alliances : Finset Alliance
  max_size : ∀ a, a ∈ alliances → a.members.card ≤ 50
  common_alliance : ∀ i j, i ∈ Countries → j ∈ Countries → i ≠ j → 
    ∃ a, a ∈ alliances ∧ i ∈ a.members ∧ j ∈ a.members
  max_union : ∀ a b, a ∈ alliances → b ∈ alliances → a ≠ b → 
    (a.members ∪ b.members).card ≤ 80

/-- The main theorem stating the minimum number of alliances required -/
theorem min_alliances_required : 
  ∃ (sys : AllianceSystem), sys.alliances.card = 6 ∧ 
  (∀ (sys' : AllianceSystem), sys'.alliances.card ≥ 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_alliances_required_l341_34155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_never_zero_l341_34164

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | n + 2 =>
    if (a n * a (n + 1)) % 2 = 0 then
      5 * a (n + 1) - 3 * a n
    else
      a (n + 1) - a n

theorem a_never_zero : ∀ n : ℕ, a n ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_never_zero_l341_34164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_third_quadrant_implies_angle_in_third_quadrant_l341_34188

-- Define what it means for a point to be in the third quadrant
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

-- Define what it means for an angle to be in the third quadrant
def angle_in_third_quadrant (θ : ℝ) : Prop := Real.pi < θ ∧ θ < 3*Real.pi/2

-- Theorem statement
theorem point_in_third_quadrant_implies_angle_in_third_quadrant (θ : ℝ) :
  in_third_quadrant (Real.sin θ) (Real.cos θ) → angle_in_third_quadrant θ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_third_quadrant_implies_angle_in_third_quadrant_l341_34188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_circle_area_sum_l341_34126

/-- Represents the grid and circle configuration --/
structure GridConfig where
  gridSize : Nat
  squareSize : Real
  smallCircleRadius : Real
  largeCircleRadius : Real
  smallCircleCount : Nat
  largeCircleCount : Nat

/-- Calculates the total area of the grid --/
def totalGridArea (config : GridConfig) : Real :=
  (config.gridSize ^ 2 : Nat) * config.squareSize ^ 2

/-- Calculates the total area of all circles --/
noncomputable def totalCircleArea (config : GridConfig) : Real :=
  (config.smallCircleCount * config.smallCircleRadius ^ 2 +
   config.largeCircleCount * config.largeCircleRadius ^ 2) * Real.pi

/-- Theorem stating the result for the specific configuration --/
theorem grid_circle_area_sum (config : GridConfig)
    (h1 : config.gridSize = 6)
    (h2 : config.squareSize = 1)
    (h3 : config.smallCircleRadius = 0.5)
    (h4 : config.largeCircleRadius = 1)
    (h5 : config.smallCircleCount = 4)
    (h6 : config.largeCircleCount = 1) :
    let A := totalGridArea config
    let B := totalCircleArea config
    A + B = 38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_circle_area_sum_l341_34126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_intersection_l341_34105

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := Real.log (x / 2) + 1 / 2

-- Define the distance function between intersection points
noncomputable def distance (m : ℝ) : ℝ := 2 * Real.exp (m - 1 / 2) - Real.log m

-- Theorem stating the minimum value of |AB|
theorem min_distance_intersection :
  ∃ (m₀ : ℝ), m₀ > 0 ∧ ∀ (m : ℝ), m > 0 → distance m ≥ 2 + Real.log 2 := by
  sorry

#check min_distance_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_intersection_l341_34105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_cosine_l341_34117

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the minimum value of |PF₂|²/(|PF₁| - |OA|) is 12a,
    then the minimum value of cos α (where α is the acute angle formed by
    the asymptotes in the first quadrant) is 1/5. -/
theorem hyperbola_asymptote_angle_cosine (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ P : ℝ × ℝ, P.1^2 / a^2 - P.2^2 / b^2 = 1 →
    (∃ (F₁ F₂ : ℝ × ℝ) (O A : ℝ × ℝ) (c : ℝ),
      (F₁.1 = -c ∧ F₁.2 = 0) ∧
      (F₂.1 = c ∧ F₂.2 = 0) ∧
      (O.1 = 0 ∧ O.2 = 0) ∧
      (A.1 = a ∧ A.2 = 0) ∧
      (c^2 = a^2 + b^2) →
      ∀ m : ℝ, (Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2))^2 /
        (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) - a) ≥ 12 * a)) →
  ∃ α : ℝ, α > 0 ∧ α < π/2 ∧ Real.tan α = b/a ∧
    ∀ β : ℝ, (β > 0 ∧ β < π/2 ∧ Real.tan β = b/a) → Real.cos β ≥ 1/5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_cosine_l341_34117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_inequality_l341_34136

theorem average_speed_inequality (a b : ℝ) (h : 0 < a ∧ a < b) : 
  let v := (2 * a * b) / (a + b)
  a < v ∧ v < Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_inequality_l341_34136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_properties_l341_34195

open Real Set

noncomputable def a : ℝ × ℝ := (1, Real.sqrt 3)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def f (x : ℝ) : ℝ := a.1 * (b x).1 + a.2 * (b x).2 - 1

theorem vector_dot_product_properties :
  (∀ x, f x = 0 ↔ ∃ k : ℤ, x = 2 * π * k ∨ x = 2 * π / 3 + 2 * π * k) ∧
  (∀ x ∈ Icc 0 (π / 2),
    (StrictMonoOn f (Icc 0 (π / 3)) ∧ StrictMonoOn f (Icc (π / 3) (π / 2))) ∧
    (∀ y ∈ Icc 0 (π / 2), 0 ≤ f y ∧ f y ≤ 1) ∧
    (∃ x₁ ∈ Icc 0 (π / 2), f x₁ = 0) ∧
    (∃ x₂ ∈ Icc 0 (π / 2), f x₂ = 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_properties_l341_34195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_odd_in_S_l341_34108

def S : Finset ℕ := {1, 2, 3, 4, 5}

def is_odd (n : ℕ) : Bool := n % 2 = 1

theorem probability_of_odd_in_S :
  (S.filter (fun n => is_odd n)).card / S.card = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_odd_in_S_l341_34108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_location_l341_34112

theorem complex_number_location (z : ℂ) (h : z * (1 + Complex.I) = Complex.abs (Real.sqrt 3 - Complex.I)) : 
  z = 1 - Complex.I :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_location_l341_34112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_equal_sides_not_necessarily_congruent_l341_34194

-- Define a quadrilateral as a structure with four points
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define a predicate for equal corresponding sides
def equal_sides (q1 q2 : Quadrilateral) : Prop :=
  distance q1.A q1.B = distance q2.A q2.B ∧
  distance q1.B q1.C = distance q2.B q2.C ∧
  distance q1.C q1.D = distance q2.C q2.D ∧
  distance q1.D q1.A = distance q2.D q2.A

-- Define congruence for quadrilaterals
def congruent (q1 q2 : Quadrilateral) : Prop :=
  ∃ (f : ℝ × ℝ → ℝ × ℝ), 
    (f q1.A = q2.A ∧ f q1.B = q2.B ∧ f q1.C = q2.C ∧ f q1.D = q2.D) ∧
    ∀ x y, distance x y = distance (f x) (f y)

-- Theorem statement
theorem quadrilateral_equal_sides_not_necessarily_congruent :
  ∃ (q1 q2 : Quadrilateral), equal_sides q1 q2 ∧ ¬congruent q1 q2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_equal_sides_not_necessarily_congruent_l341_34194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_sum_l341_34139

theorem trigonometric_identity_sum :
  ∃ (a b c d : ℕ+), 
    (∀ x : ℝ, (Real.cos x) + (Real.cos (5*x)) + (Real.cos (9*x)) + (Real.cos (13*x)) = 
      (a : ℝ) * (Real.cos ((b : ℝ)*x)) * (Real.cos ((c : ℝ)*x)) * (Real.cos ((d : ℝ)*x))) ∧
    (a + b + c + d = 16) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_sum_l341_34139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_solution_mixing_l341_34158

/-- Represents a salt solution with a given volume and concentration -/
structure SaltSolution where
  volume : ℝ
  concentration : ℝ
  volumeNonneg : 0 ≤ volume
  concentrationBounded : 0 ≤ concentration ∧ concentration ≤ 1

/-- Represents the mixing of two salt solutions -/
noncomputable def mixSolutions (s1 s2 : SaltSolution) : SaltSolution :=
  { volume := s1.volume + s2.volume,
    concentration := (s1.volume * s1.concentration + s2.volume * s2.concentration) / (s1.volume + s2.volume),
    volumeNonneg := by
      apply add_nonneg
      exact s1.volumeNonneg
      exact s2.volumeNonneg
    concentrationBounded := by sorry }

/-- The theorem stating that mixing 30 oz of 20% salt solution with 30 oz of 60% salt solution results in 60 oz of 40% salt solution -/
theorem salt_solution_mixing :
  let s1 : SaltSolution := ⟨30, 0.2, by norm_num, by norm_num⟩
  let s2 : SaltSolution := ⟨30, 0.6, by norm_num, by norm_num⟩
  let result := mixSolutions s1 s2
  result.volume = 60 ∧ result.concentration = 0.4 := by
  simp [mixSolutions]
  constructor
  · norm_num
  · field_simp
    norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_solution_mixing_l341_34158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_followers_count_l341_34148

/-- Represents the types of islanders -/
inductive IslanderType
  | Knight
  | Liar
  | Follower

/-- Represents an answer to the question -/
inductive Answer
  | Yes
  | No

/-- The total number of islanders -/
def totalIslanders : Nat := 2018

/-- The number of "Yes" answers -/
def yesAnswers : Nat := 1009

/-- Function to determine a knight's answer based on the actual situation -/
def knightAnswer (moreKnightsThanLiars : Bool) : Answer :=
  if moreKnightsThanLiars then Answer.Yes else Answer.No

/-- Function to determine a liar's answer based on the actual situation -/
def liarAnswer (moreKnightsThanLiars : Bool) : Answer :=
  if moreKnightsThanLiars then Answer.No else Answer.Yes

/-- Function to determine a follower's answer based on previous answers -/
def followerAnswer (previousYes previousNo : Nat) : Answer :=
  if previousYes > previousNo then Answer.Yes
  else if previousNo > previousYes then Answer.No
  else Answer.Yes  -- Simplified tie-breaking

/-- The main theorem to prove -/
theorem max_followers_count :
  ∃ (knights liars followers : Nat),
    knights + liars + followers = totalIslanders ∧
    followers = 1009 ∧
    (∀ maxFollowers, 
      maxFollowers > followers →
      ¬∃ (newKnights newLiars : Nat),
        newKnights + newLiars + maxFollowers = totalIslanders ∧
        newKnights + newLiars = yesAnswers) := by
  -- Proof sketch:
  -- 1. Show that followers = 1009 is possible
  -- 2. Show that any larger number of followers would violate the constraints
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_followers_count_l341_34148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_arithmetic_from_geometric_l341_34146

theorem log_arithmetic_from_geometric (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_geom : a * c = b^2) : 
  (Real.log a - Real.log b = Real.log b - Real.log c) := by
  sorry

#check log_arithmetic_from_geometric

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_arithmetic_from_geometric_l341_34146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_eq_pi_sufficient_not_necessary_l341_34178

noncomputable def curve (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

def passes_through_origin (φ : ℝ) : Prop :=
  ∃ x : ℝ, curve x φ = 0

theorem phi_eq_pi_sufficient_not_necessary :
  (passes_through_origin π) ∧
  ¬(∀ φ : ℝ, passes_through_origin φ → φ = π) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_eq_pi_sufficient_not_necessary_l341_34178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_parallel_line_in_plane_l341_34177

-- Define the necessary structures
structure Point where

structure Line where

structure Plane where

-- Define the relations
def ParallelLinePlane (l : Line) (p : Plane) : Prop := sorry

def PointInPlane (pt : Point) (p : Plane) : Prop := sorry

def LinePassesThrough (l : Line) (pt : Point) : Prop := sorry

def ParallelLines (l1 l2 : Line) : Prop := sorry

def LineInPlane (l : Line) (p : Plane) : Prop := sorry

-- Theorem statement
theorem unique_parallel_line_in_plane 
  (a : Line) (α : Plane) (A : Point)
  (h1 : ParallelLinePlane a α)
  (h2 : PointInPlane A α) :
  ∃! l : Line, 
    LinePassesThrough l A ∧ 
    ParallelLines l a ∧ 
    LineInPlane l α :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_parallel_line_in_plane_l341_34177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_side_length_l341_34189

/-- Given a right pyramid with a square base, prove that the side length of the base is 12 meters
    when the area of one lateral face is 144 square meters and the slant height is 24 meters. -/
theorem pyramid_base_side_length 
  (pyramid : Real) 
  (lateral_face_area : Real)
  (slant_height : Real)
  (h1 : lateral_face_area = 144)
  (h2 : slant_height = 24) : 
  Real.sqrt (2 * lateral_face_area / slant_height) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_side_length_l341_34189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_simplification_and_evaluation_l341_34118

theorem function_simplification_and_evaluation (α : ℝ) : 
  (Real.sin (α - 3 * Real.pi) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3/2 * Real.pi)) / 
    (Real.cos (-Real.pi - α) * Real.sin (-Real.pi - α)) = -Real.cos α ∧
  (Real.sin (α - 3/2 * Real.pi) = 1/5 → 
    (Real.sin (α - 3 * Real.pi) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3/2 * Real.pi)) / 
      (Real.cos (-Real.pi - α) * Real.sin (-Real.pi - α)) = -1/5) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_simplification_and_evaluation_l341_34118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_reciprocal_l341_34192

theorem tan_sum_reciprocal (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 4) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_reciprocal_l341_34192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l341_34133

theorem sum_remainder_mod_15 (a b c : ℕ) 
  (ha : a % 15 = 11) 
  (hb : b % 15 = 12) 
  (hc : c % 15 = 13) : 
  (a + b + c) % 15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l341_34133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_local_extrema_l341_34130

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 2) / (2*x - 2)

-- State the theorem
theorem f_local_extrema :
  (∃ δ₁ > 0, ∀ x ∈ Set.Ioo (-δ₁) δ₁, f x ≤ f 0) ∧
  (∃ δ₂ > 0, ∀ x ∈ Set.Ioo (2 - δ₂) (2 + δ₂), f x ≥ f 2) ∧
  f 0 = -1 ∧
  f 2 = 1 := by
  sorry

-- Additional lemmas to support the main theorem
lemma f_eq_alt (x : ℝ) (h : x ≠ 1) : f x = (x - 1) / 2 + 1 / (2 * (x - 1)) := by
  sorry

lemma f_zero_eq_neg_one : f 0 = -1 := by
  sorry

lemma f_two_eq_one : f 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_local_extrema_l341_34130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_neg_half_delta_satisfies_limit_l341_34168

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (6 * x^2 + x - 1) / (x + 1/2)

-- State the theorem
theorem limit_of_f_at_neg_half :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x + 1/2| ∧ |x + 1/2| < δ → |f x + 5| < ε :=
by
  sorry

-- Define delta as a function of epsilon
noncomputable def δ (ε : ℝ) : ℝ := ε / 6

-- State that this delta satisfies the limit definition
theorem delta_satisfies_limit :
  ∀ ε > 0, ∀ x : ℝ, 0 < |x + 1/2| ∧ |x + 1/2| < δ ε → |f x + 5| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_neg_half_delta_satisfies_limit_l341_34168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_theorem_l341_34157

/-- Two parallel lines in 2D space --/
structure ParallelLines where
  a : ℝ
  b : ℝ
  c₁ : ℝ
  c₂ : ℝ

/-- Calculate the distance between two parallel lines --/
noncomputable def distance_between_parallel_lines (l : ParallelLines) : ℝ :=
  abs (l.c₂ - l.c₁) / Real.sqrt (l.a^2 + l.b^2)

/-- Calculate the length of the line segment formed by intersections --/
noncomputable def intersection_length (l : ParallelLines) (y : ℝ) : ℝ :=
  abs ((-(l.c₂ + l.b * y)) / l.a - (-(l.c₁ + l.b * y)) / l.a)

/-- The main theorem --/
theorem parallel_lines_theorem (l : ParallelLines) :
  l.a = 3 ∧ l.b = 4 ∧ l.c₁ = -7 ∧ l.c₂ = 8 →
  (distance_between_parallel_lines l = 3) ∧
  (intersection_length l 3 = 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_theorem_l341_34157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_product_in_S_l341_34138

def S : Set Int := {-8, -4, -2, 1, 5}

theorem smallest_product_in_S :
  (∀ x y, x ∈ S → y ∈ S → x * y ≥ -40) ∧ (∃ x y, x ∈ S ∧ y ∈ S ∧ x * y = -40) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_product_in_S_l341_34138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_random_event_l341_34116

/-- Represents an idiom with its meaning --/
structure Idiom where
  chinese : String
  english : String

/-- Determines if an idiom describes a random event --/
def isRandomEvent (i : Idiom) : Prop :=
  i.chinese = "一箭双雕" ∧ i.english = "To kill two birds with one stone"

/-- The list of given idioms --/
def idioms : List Idiom := [
  { chinese := "日落西山", english := "The sun sets in the west" },
  { chinese := "揠苗助长", english := "To pull up seedlings to help them grow" },
  { chinese := "一箭双雕", english := "To kill two birds with one stone" },
  { chinese := "一步登天", english := "To ascend to heaven in one step" }
]

theorem only_one_random_event :
  ∃! i, i ∈ idioms ∧ isRandomEvent i :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_random_event_l341_34116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_pairings_l341_34171

/-- Represents a person in the circle -/
def Person := Fin 12

/-- Defines the relationship of knowing between two people -/
def knows (a b : Person) : Prop :=
  (a.val + 1 = b.val % 12) ∨ (b.val + 1 = a.val % 12) ∨ (a.val + 2 = b.val % 12)

/-- A valid pairing of people -/
def ValidPairing (pairs : List (Person × Person)) : Prop :=
  pairs.length = 6 ∧
  (∀ (p : Person × Person), p ∈ pairs → knows p.1 p.2) ∧
  (∀ (p : Person), ∃! (q : Person), (p, q) ∈ pairs ∨ (q, p) ∈ pairs)

/-- The main theorem stating there are exactly 3 valid pairings -/
theorem exactly_three_pairings :
  ∃! (pairings : List (List (Person × Person))),
    pairings.length = 3 ∧
    (∀ pairing ∈ pairings, ValidPairing pairing) ∧
    (∀ pairing, ValidPairing pairing → pairing ∈ pairings) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_pairings_l341_34171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_one_l341_34154

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x + 1/x

-- State the theorem
theorem derivative_at_one :
  deriv f 1 = 0 := by
  -- We'll use the sorry tactic to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_one_l341_34154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l341_34111

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  area : Real

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sqrt 2 * Real.cos t.A * (t.b * Real.cos t.C + t.c * Real.cos t.B) = t.a)
  (h2 : t.a = Real.sqrt 5)
  (h3 : t.area = Real.sqrt 2 - 1) :
  t.A = Real.pi / 4 ∧ t.a + t.b + t.c = 3 + Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l341_34111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_approx_l341_34142

/-- Represents the time (in hours) it takes to fill a cistern when two taps are opened simultaneously, 
    given the time it takes to fill and empty the cistern individually. -/
noncomputable def simultaneous_fill_time (fill_time empty_time : ℝ) : ℝ :=
  1 / (1 / fill_time - 1 / empty_time)

/-- Theorem stating that for a cistern that can be filled in 4 hours and emptied in 9 hours,
    the time to fill when both taps are opened simultaneously is approximately 7.2 hours. -/
theorem cistern_fill_time_approx :
  ∀ ε > 0, |simultaneous_fill_time 4 9 - 7.2| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_approx_l341_34142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_september_monday_yes_l341_34110

/-- Represents the hair color of a resident --/
inductive HairColor
| Blonde
| Brunette

/-- Represents a month --/
inductive Month
| September
| October

/-- Represents a day of the week --/
inductive Weekday
| Monday
| Friday

/-- Represents a resident of city N --/
structure Resident where
  birthMonth : Month
  mondayHairColor : HairColor

/-- The number of residents who answered "yes" on the given day --/
def yesAnswers (day : Weekday) (month : Month) : Nat := 
  sorry -- Implementation details omitted

/-- The total number of autumn-born residents --/
def autumnBornResidents : Nat := 
  sorry -- Implementation details omitted

/-- There are exactly four Mondays in October --/
axiom four_mondays_october : True

/-- No one in city N was born in November --/
axiom no_november_births : True

/-- On a Monday in October, 200 autumn-born residents answered "yes" --/
axiom october_monday_yes : yesAnswers Weekday.Monday Month.October = 200

/-- On Friday of the same week, only 50 of the same people answered "yes" --/
axiom october_friday_yes : yesAnswers Weekday.Friday Month.October = 50

/-- The number of "yes" answers on the last Monday of September is 0 --/
theorem last_september_monday_yes :
  yesAnswers Weekday.Monday Month.September = 0 := by
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_september_monday_yes_l341_34110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_shelves_optimal_l341_34182

/-- The number of books -/
def total_books : ℕ := 1300

/-- The maximum number of bookshelves -/
def max_shelves : ℕ := 18

/-- A function that determines if, for a given number of shelves,
    there will always be at least 5 books on the same shelf after any rearrangement -/
def always_five_books_together (shelves : ℕ) : Prop :=
  ∀ (arrangement1 arrangement2 : ℕ → Finset ℕ),
    (∀ i, i < shelves → arrangement1 i ⊆ Finset.range total_books) →
    (∀ i, i < shelves → arrangement2 i ⊆ Finset.range total_books) →
    (∀ i, i < shelves → (arrangement1 i).card + (arrangement2 i).card = total_books) →
    ∃ i j, i < shelves ∧ j < shelves ∧ (arrangement1 i ∩ arrangement2 j).card ≥ 5

/-- The theorem stating that max_shelves is the maximum number of shelves
    for which there will always be at least 5 books on the same shelf after any rearrangement -/
theorem max_shelves_optimal :
  always_five_books_together max_shelves ∧
  ¬always_five_books_together (max_shelves + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_shelves_optimal_l341_34182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_weight_problem_l341_34184

theorem tank_weight_problem (u v : ℝ) :
  let m := (9/5 * v - 4/5 * u) -- weight of empty tank
  let n := 12/5 * (u - v) -- weight of oil when full
  (3/4 * n + m = u) → -- when tank is 3/4 full
  (1/3 * n + m = v) → -- when tank is 1/3 full
  (n + m = 8/5 * u - 3/5 * v) -- when tank is full
:= by
  intros m n h1 h2
  have h3 : n = 12/5 * (u - v) := by
    linarith
  have h4 : m = 9/5 * v - 4/5 * u := by
    linarith
  calc
    n + m = 12/5 * (u - v) + (9/5 * v - 4/5 * u) := by rw [h3, h4]
    _ = 12/5 * u - 12/5 * v + 9/5 * v - 4/5 * u := by ring
    _ = 8/5 * u - 3/5 * v := by ring

#check tank_weight_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_weight_problem_l341_34184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l341_34151

noncomputable def f (a x : ℝ) : ℝ := 1 - 2*a - 2*a*Real.cos x - 2*(Real.sin x)^2

noncomputable def g (a : ℝ) : ℝ :=
  if a < -2 then 1
  else if -2 ≤ a ∧ a ≤ 2 then -a^2/2 - 2*a - 1
  else 1 - 4*a

theorem min_value_of_f (a : ℝ) : 
  IsGLB {f a x | x : ℝ} (g a) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l341_34151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_extrema_l341_34113

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

-- Define the interval
def interval : Set ℝ := Set.Icc 0 (Real.pi / 2)

theorem tangent_line_and_extrema :
  -- The equation of the tangent line at (0, f(0)) is y = 1
  (∀ y, Set.range (λ x ↦ (x, y)) ∩ {p : ℝ × ℝ | p.2 = f p.1} ∩ {(0, f 0)} ≠ ∅ → y = 1) ∧
  -- The maximum value in the interval is 1
  (∃ x ∈ interval, f x = 1 ∧ ∀ y ∈ interval, f y ≤ f x) ∧
  -- The minimum value in the interval is -π/2
  (∃ x ∈ interval, f x = -Real.pi / 2 ∧ ∀ y ∈ interval, f x ≤ f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_extrema_l341_34113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_equals_2F_l341_34109

/-- The function F defined as log((1+x)/(1-x)) -/
noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

/-- The function G defined by replacing x in F with (2x)/(1+x^2) -/
noncomputable def G (x : ℝ) : ℝ := F ((2 * x) / (1 + x^2))

/-- Theorem stating that G equals 2F for all real x where both are defined -/
theorem G_equals_2F (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) (h3 : x^2 ≠ -1) : 
  G x = 2 * (F x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_equals_2F_l341_34109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_tan_angle_MAN_slope_MN_constant_l341_34186

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 13

-- Define the line
def line_eq (x₀ y₀ x y : ℝ) : Prop := x₀ * x + y₀ * y = 13

-- Part 1
theorem line_intersects_circle (x₀ y₀ : ℝ) (h : x₀^2 + y₀^2 > 13) :
  ∃ x y, circle_eq x y ∧ line_eq x₀ y₀ x y := by sorry

-- Part 2
theorem tan_angle_MAN :
  circle_eq 2 3 →
  ∃ x y, circle_eq x y ∧ y - 3 = (3/2) * (x - 2) ∧ x^2 + y^2 = 13 →
  (12 : ℝ) / 5 = abs ((3/2 - (-3/2)) / (1 + 3/2 * (-3/2))) := by sorry

-- Part 3
theorem slope_MN_constant (k : ℝ) :
  circle_eq 2 3 →
  ∃ xM yM xN yN,
    circle_eq xM yM ∧ circle_eq xN yN ∧
    yM - 3 = k * (xM - 2) ∧
    yN - 3 = (-k) * (xN - 2) →
  (yM - yN) / (xM - xN) = 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_tan_angle_MAN_slope_MN_constant_l341_34186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_l341_34162

-- Define the side length of the larger equilateral triangle
noncomputable def large_triangle_side : ℝ := 2

-- Define the area of the larger equilateral triangle
noncomputable def large_triangle_area : ℝ := (Real.sqrt 3 / 4) * large_triangle_side^2

-- Define the area of the smaller equilateral triangle
noncomputable def small_triangle_area : ℝ := large_triangle_area / 2

-- Define the side length of the smaller equilateral triangle (circle radius)
noncomputable def circle_radius : ℝ := Real.sqrt (4 * small_triangle_area / Real.sqrt 3)

-- Theorem statement
theorem hexagon_perimeter :
  let hexagon_side := circle_radius
  let hexagon_perimeter := 6 * hexagon_side
  hexagon_perimeter = 6 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_l341_34162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_f_l341_34120

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x ≥ 0 then 2 * x^2 - 4 * x else -2 * x^2 - 4 * x

-- State the theorem
theorem odd_function_f :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x ≥ 0, f x = 2 * x^2 - 4 * x) ∧
  (∀ x < 0, f x = -2 * x^2 - 4 * x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_f_l341_34120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_side_length_is_two_l341_34135

/-- The length of a side of an isosceles triangle constructed inside a regular hexagon -/
noncomputable def isosceles_side_length (hexagon_side : ℝ) : ℝ :=
  let hexagon_area := 3 * Real.sqrt 3 / 2 * hexagon_side^2
  let triangle_area := hexagon_area / 6
  let triangle_height := 2 * triangle_area / hexagon_side
  Real.sqrt (triangle_height^2 + (hexagon_side / 2)^2)

/-- Theorem: The length of the congruent sides of the isosceles triangle is 2 -/
theorem isosceles_side_length_is_two :
  isosceles_side_length 2 = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_side_length_is_two_l341_34135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eigenvalues_of_specific_matrix_l341_34101

theorem eigenvalues_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 12; 4, 3]
  ∀ (k : ℝ) (v : Fin 2 → ℝ),
    v ≠ 0 →
    A.mulVec v = k • v →
    k = 3 + 4 * Real.sqrt 3 ∨ k = 3 - 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eigenvalues_of_specific_matrix_l341_34101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_range_l341_34180

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the directrix of the parabola
def directrix : ℝ := -2

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem parabola_point_range (x₀ y₀ : ℝ) :
  parabola x₀ y₀ →
  distance (x₀, y₀) focus > distance (directrix, 0) focus →
  x₀ > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_range_l341_34180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fall_spending_l341_34197

/-- Represents the expenditure of Grove Town in millions of dollars -/
structure Expenditure where
  amount : ℝ

/-- The total expenditure of Grove Town in 1995 -/
def total_expenditure : Expenditure := ⟨0⟩

/-- The expenditure by the beginning of March -/
def march_expenditure : Expenditure := ⟨0.7⟩

/-- The expenditure by the end of September -/
def september_expenditure : Expenditure := ⟨3.2⟩

/-- The expenditure during the fall months (September, October, November) -/
def fall_expenditure : Expenditure := ⟨2.5⟩

/-- Axiom: The expenditure by the beginning of March is 0.7 million -/
axiom march_spending : march_expenditure.amount = 0.7

/-- Axiom: The expenditure by the end of September is 3.2 million -/
axiom september_spending : september_expenditure.amount = 3.2

/-- The main theorem to prove -/
theorem fall_spending : fall_expenditure.amount = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fall_spending_l341_34197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_whole_number_above_sum_l341_34145

theorem smallest_whole_number_above_sum : 
  ⌈(3 + 1/3 : ℝ) + (5 + 1/4 : ℝ) + (7 + 1/6 : ℝ) + (9 + 1/8 : ℝ)⌉ = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_whole_number_above_sum_l341_34145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_degree_bound_l341_34196

theorem polynomial_degree_bound (p : ℕ) (f : Polynomial ℤ) (d : ℕ) :
  Prime p →
  f.degree = d →
  f.eval 0 = 0 →
  f.eval 1 = 1 →
  (∀ n : ℕ, n > 0 → (f.eval (n : ℤ)) % p = 0 ∨ (f.eval (n : ℤ)) % p = 1) →
  d ≥ p - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_degree_bound_l341_34196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_discount_percentage_l341_34163

theorem apple_discount_percentage 
  (original_price : ℝ) 
  (discounted_total : ℝ) 
  (quantity : ℝ) 
  (h1 : original_price = 5)
  (h2 : discounted_total = 30)
  (h3 : quantity = 10) :
  (((original_price * quantity - discounted_total) / quantity) / original_price) * 100 = 40 :=
by
  -- Introduce local definitions
  let original_total := original_price * quantity
  let discount_amount := original_total - discounted_total
  let discount_per_kg := discount_amount / quantity
  let discount_percentage := (discount_per_kg / original_price) * 100
  
  -- Proof steps would go here, but we're using sorry to skip the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_discount_percentage_l341_34163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_knights_tour_exists_l341_34191

/-- A knight's tour on a chessboard -/
def KnightsTour (n : ℕ) : Prop :=
  ∃ (tour : List (ℕ × ℕ)),
    (∀ (i j : ℕ), i < 4*n+1 ∧ j < 4*n+1 → (i, j) ∈ tour) ∧
    (∀ (pos : ℕ × ℕ), pos ∈ tour → List.count pos tour = 1) ∧
    (∀ (i : ℕ), i < tour.length - 1 →
      let (x₁, y₁) := tour[i]!
      let (x₂, y₂) := tour[i+1]!
      (Int.natAbs (x₁ - x₂) = 1 ∧ Int.natAbs (y₁ - y₂) = 2) ∨
      (Int.natAbs (x₁ - x₂) = 2 ∧ Int.natAbs (y₁ - y₂) = 1))

/-- Theorem: A knight can traverse any (4n+1) × (4n+1) chessboard -/
theorem knights_tour_exists (n : ℕ) : KnightsTour n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_knights_tour_exists_l341_34191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_non_dominating_partitions_l341_34137

/-- A partition of a natural number -/
def Partition := List ℕ

/-- Check if a list is in descending order -/
def isDescending (l : List ℕ) : Prop :=
  ∀ i j, i < j → j < l.length → l.get! i ≥ l.get! j

/-- Check if a list sums to n -/
def sumsTo (l : List ℕ) (n : ℕ) : Prop :=
  l.sum = n

/-- Check if all elements in a list are positive -/
def allPositive (l : List ℕ) : Prop :=
  ∀ x, x ∈ l → x > 0

/-- Define a valid partition -/
def isValidPartition (p : Partition) : Prop :=
  isDescending p ∧ sumsTo p 20 ∧ allPositive p

/-- Define dominance between two partitions -/
def dominates (p1 p2 : Partition) : Prop :=
  ∀ k, k ≤ min p1.length p2.length →
    (p1.take k).sum ≥ (p2.take k).sum

/-- The main theorem -/
theorem max_non_dominating_partitions :
  ∃ (partitions : List Partition),
    partitions.length = 20 ∧
    (∀ p, p ∈ partitions → isValidPartition p) ∧
    (∀ p1 p2, p1 ∈ partitions → p2 ∈ partitions → p1 ≠ p2 → ¬(dominates p1 p2) ∧ ¬(dominates p2 p1)) ∧
    (∀ p : Partition, isValidPartition p →
      p ∉ partitions →
      ∃ q, q ∈ partitions ∧ (dominates p q ∨ dominates q p)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_non_dominating_partitions_l341_34137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_intersection_right_angle_iff_opposite_sides_squares_equal_l341_34132

-- Define a convex quadrilateral
structure ConvexQuadrilateral (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (A B C D : V)
  (convex : Convex ℝ {A, B, C, D})

-- Define the intersection point of diagonals
noncomputable def DiagonalIntersection {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (quad : ConvexQuadrilateral V) : V :=
  sorry

-- Define the angle between two vectors
noncomputable def angle {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) : ℝ :=
  Real.arccos ((inner a b) / (norm a * norm b))

-- State the theorem
theorem diagonal_intersection_right_angle_iff_opposite_sides_squares_equal
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (quad : ConvexQuadrilateral V) :
  let E := DiagonalIntersection quad
  angle (quad.A - E) (quad.C - E) = π / 2 ↔ 
  ‖quad.A - quad.B‖^2 + ‖quad.C - quad.D‖^2 = ‖quad.A - quad.D‖^2 + ‖quad.B - quad.C‖^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_intersection_right_angle_iff_opposite_sides_squares_equal_l341_34132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_common_ratio_l341_34159

/-- The common ratio of the geometric series 2/3 + 4/9 + 8/27 + ... is 2/3 -/
theorem geometric_series_common_ratio : 
  let a : ℕ → ℚ := λ n => (2 / 3) * (2 / 3) ^ n
  ∀ n : ℕ, a (n + 1) / a n = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_common_ratio_l341_34159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_figure_l341_34100

/-- The perimeter of a square ABCD -/
noncomputable def square_perimeter : ℝ := 48

/-- The side length of the square ABCD -/
noncomputable def square_side : ℝ := square_perimeter / 4

/-- The number of sides in the figure ABFCDE -/
def sides_in_figure : ℕ := 6

/-- Theorem: The perimeter of figure ABFCDE is 72 inches -/
theorem perimeter_of_figure (h1 : square_perimeter = 48) 
  (h2 : square_side = square_perimeter / 4)
  (h3 : sides_in_figure = 6) : 
  (sides_in_figure : ℝ) * square_side = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_figure_l341_34100
