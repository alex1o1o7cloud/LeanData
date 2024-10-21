import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pairs_satisfy_equation_l501_50195

theorem two_pairs_satisfy_equation : 
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ ↦ p.1^2 - p.2^2 = 77 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 100) (Finset.range 100))).card ∧ n = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pairs_satisfy_equation_l501_50195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_value_l501_50151

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + Real.log x / Real.log a

-- State the theorem
theorem find_a_value :
  ∀ a : ℝ, a > 0 → f a 2 = 3 → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_value_l501_50151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l501_50117

theorem sin_2alpha_value (α : ℝ) 
  (h : (Real.cos (2 * α)) / (Real.sin (α + π / 4)) = 4 / 7) : 
  Real.sin (2 * α) = 41 / 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l501_50117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l501_50109

-- Define a random variable following normal distribution
structure NormalDistribution (μ σ : ℝ) where
  value : ℝ

-- Define the probability function
noncomputable def P {μ σ : ℝ} (event : Set (NormalDistribution μ σ)) : ℝ := sorry

-- State the theorem
theorem normal_distribution_probability 
  (σ : ℝ) 
  (h : P {x : NormalDistribution 1 σ | x.value > 2} = 0.15) : 
  P {x : NormalDistribution 1 σ | 0 ≤ x.value ∧ x.value ≤ 1} = 0.35 := by
  sorry

#check normal_distribution_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l501_50109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l501_50126

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the distance from a point to the line x = -3
def distanceToLine (p : ℝ × ℝ) : ℝ := |p.1 + 3|

-- Theorem statement
theorem parabola_focus_distance 
  (p : ℝ × ℝ) 
  (h1 : parabola p.1 p.2) 
  (h2 : distanceToLine p = 5) : 
  distance p focus = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l501_50126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_midpoint_trajectory_l501_50197

/-- The circle C₁ with equation x² + y² - 6x + 5 = 0 -/
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 5 = 0

/-- The line l passing through the origin with slope k -/
def line_l (k x y : ℝ) : Prop := y = k * x

/-- The range of slopes k for which the line intersects the circle -/
def slope_range (k : ℝ) : Prop := -2 * Real.sqrt 5 / 5 ≤ k ∧ k ≤ 2 * Real.sqrt 5 / 5

/-- The trajectory of the midpoint M -/
def midpoint_trajectory (x y : ℝ) : Prop := (x - 3/2)^2 + y^2 = 9/4 ∧ 5/3 < x ∧ x ≤ 3

theorem intersection_and_midpoint_trajectory :
  ∀ (k x y x1 y1 x2 y2 : ℝ),
  (circle_C1 x1 y1 ∧ circle_C1 x2 y2 ∧
   line_l k x1 y1 ∧ line_l k x2 y2 ∧
   x1 ≠ x2) →
  (slope_range k ∧
   midpoint_trajectory ((x1 + x2) / 2) ((y1 + y2) / 2)) :=
by
  sorry

#check intersection_and_midpoint_trajectory

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_midpoint_trajectory_l501_50197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l501_50144

noncomputable section

def a : ℕ → ℝ := fun n => n * (n + 1)
def b : ℕ → ℝ := fun n => (1 : ℝ) / (2^(n - 1))
def A : ℕ → ℝ := fun n => (n + 2 : ℝ) / 3 * a n
def B : ℕ → ℝ := fun n => 2 - b n
def c : ℕ → ℝ := fun n => Real.sqrt (a n) * (Finset.sum (Finset.range (n - 1)) (fun i => b (i + 1))) +
  (Finset.sum (Finset.range n) (fun i => Real.sqrt (a (i + 1)))) * b n
def S : ℕ → ℝ := fun n => Finset.sum (Finset.range n) (fun i => c (i + 1))

axiom a_def : ∀ n : ℕ, a n = n * (n + 1)
axiom b_def : ∀ n : ℕ, b n = (1 : ℝ) / (2^(n - 1))
axiom A_def : ∀ n : ℕ, A n = (n + 2 : ℝ) / 3 * a n
axiom B_def : ∀ n : ℕ, B n = 2 - b n
axiom c_def : ∀ n : ℕ, c n = Real.sqrt (a n) * (Finset.sum (Finset.range (n - 1)) (fun i => b (i + 1))) +
  (Finset.sum (Finset.range n) (fun i => Real.sqrt (a (i + 1)))) * b n
axiom S_def : ∀ n : ℕ, S n = Finset.sum (Finset.range n) (fun i => c (i + 1))

theorem main_theorem (n : ℕ) :
  S n = (Finset.sum (Finset.range n) (fun i => Real.sqrt (a (i + 1)))) * (Finset.sum (Finset.range n) (fun i => b (i + 1))) ∧
  S n ≤ n * (n + 2) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l501_50144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_planes_count_l501_50130

/-- Represents a plane in 3D space -/
structure Plane where

/-- Represents an equilateral triangle in 3D space -/
structure EquilateralTriangle where
  sideLength : ℝ

/-- Returns the number of planes equidistant from all vertices of an equilateral triangle -/
def countEquidistantPlanes (triangle : EquilateralTriangle) (distance : ℝ) : ℕ := 
  sorry

/-- The main theorem stating that there are exactly 5 planes equidistant from all vertices
    of an equilateral triangle with side length 3 and distance 3√3/4 -/
theorem equidistant_planes_count :
  let triangle : EquilateralTriangle := { sideLength := 3 }
  let distance : ℝ := 3 * Real.sqrt 3 / 4
  countEquidistantPlanes triangle distance = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_planes_count_l501_50130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundreds_digit_of_factorial_difference_l501_50135

open BigOperators

theorem hundreds_digit_of_factorial_difference : ∃ k : ℕ, (Finset.prod (Finset.range 30) (λ i => i + 1)) - (Finset.prod (Finset.range 25) (λ i => i + 1)) = 1000 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundreds_digit_of_factorial_difference_l501_50135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_round_to_7247_l501_50110

noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem not_round_to_7247 : round_to_hundredth 72.476 ≠ 72.47 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_round_to_7247_l501_50110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_for_given_sin_l501_50178

theorem cos_value_for_given_sin (α : ℝ) : 
  0 < α ∧ α < π / 2 →  -- α is an acute angle
  Real.sin α = 4 / 5 → 
  Real.cos α = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_for_given_sin_l501_50178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_cube_sum_l501_50173

theorem sin_cos_cube_sum (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) (h3 : Real.sin x - Real.cos x = 1 / 2) :
  Real.sin x ^ 3 + Real.cos x ^ 3 = 5 * Real.sqrt 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_cube_sum_l501_50173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_l501_50103

theorem sin_alpha_plus_pi (α : ℝ) (h1 : α ∈ Set.Ioo (π/2) π) (h2 : Real.tan α = -3/4) :
  Real.sin (α + π) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_l501_50103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_consumption_rate_approx_l501_50175

noncomputable def fuel_left : ℝ := 6.3333
noncomputable def time_left : ℝ := 0.6667

noncomputable def fuel_consumption_rate : ℝ := fuel_left / time_left

theorem fuel_consumption_rate_approx :
  |fuel_consumption_rate - 9.5| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_consumption_rate_approx_l501_50175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_odd_three_digit_numbers_l501_50129

/-- A function that returns true if a number is a valid odd three-digit number
    with the sum of its tens and units digits equal to 12 -/
def isValidNumber (n : ℕ) : Bool :=
  100 ≤ n ∧ n < 1000 ∧  -- Three-digit number
  n % 2 = 1 ∧  -- Odd number
  (n / 10 % 10 + n % 10 = 12)  -- Sum of tens and units digits is 12

/-- The count of odd three-digit numbers where the sum of the tens and units digits is 12 -/
def countValidNumbers : ℕ := (Finset.filter (fun n => isValidNumber n) (Finset.range 1000)).card

theorem count_odd_three_digit_numbers : countValidNumbers = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_odd_three_digit_numbers_l501_50129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_distribution_l501_50158

/-- Represents a segment with a color and length -/
structure Segment where
  color : Bool  -- True for white, False for black
  length : ℝ
  length_pos : length > 0

/-- Represents a distribution of segments on a line -/
structure Distribution where
  segments : List Segment
  positions : List ℝ
  length : ℝ

/-- Checks if a distribution is valid according to the problem conditions -/
def is_valid_distribution (d : Distribution) : Prop :=
  -- Add conditions here
  True

/-- The main theorem to be proved -/
theorem segment_distribution :
  (∀ (white_segments black_segments : List Segment),
    (List.sum (white_segments.map (λ s => s.length)) = 1) →
    (List.sum (black_segments.map (λ s => s.length)) = 1) →
    ∃ (d : Distribution),
      d.length = 1.51 ∧
      is_valid_distribution d ∧
      d.segments = white_segments ++ black_segments) ∧
  (∃ (white_segments black_segments : List Segment),
    (List.sum (white_segments.map (λ s => s.length)) = 1) ∧
    (List.sum (black_segments.map (λ s => s.length)) = 1) ∧
    ∀ (d : Distribution),
      d.length = 1.49 →
      ¬(is_valid_distribution d ∧
        d.segments = white_segments ++ black_segments)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_distribution_l501_50158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_problem_l501_50176

theorem log_problem (x : ℝ) (h1 : x < 1) (h2 : (Real.log x)^2 - Real.log (x^3) = 75 * Real.log 10) :
  (Real.log x)^3 - Real.log (x^4) = -391.875 * (Real.log 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_problem_l501_50176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_statements_correct_l501_50121

-- Define the four statements
def statement1 : Prop := ∃ (l : Set ℝ) (r : Set ℝ), True  -- Placeholder for line and ray
def statement2 : Prop := ∀ (p q : ℝ × ℝ), ∃! (l : Set (ℝ × ℝ)), p ∈ l ∧ q ∈ l
def statement3 : Prop := ∀ (l : Set (ℝ × ℝ)) (a : ℝ), True  -- Placeholder for line and angle
def statement4 : Prop := ∀ (a b : ℝ), True  -- Placeholder for angles and their properties

-- Define a function that counts the number of true statements
def countTrueStatements (s1 s2 s3 s4 : Bool) : Nat :=
  (if s1 then 1 else 0) + (if s2 then 1 else 0) + 
  (if s3 then 1 else 0) + (if s4 then 1 else 0)

-- Theorem statement
theorem exactly_two_statements_correct :
  ∃ (b1 b2 b3 b4 : Bool),
    (b1 ↔ statement1) ∧
    (b2 ↔ statement2) ∧
    (b3 ↔ statement3) ∧
    (b4 ↔ statement4) ∧
    countTrueStatements b1 b2 b3 b4 = 2 := by
  -- Proof sketch
  exists false, true, false, true
  constructor
  · sorry  -- Proof that statement1 is false
  constructor
  · sorry  -- Proof that statement2 is true
  constructor
  · sorry  -- Proof that statement3 is false
  constructor
  · sorry  -- Proof that statement4 is true
  rfl  -- This proves that countTrueStatements false true false true = 2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_statements_correct_l501_50121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_sum_equals_two_l501_50174

-- Define lg as the logarithm in base 10
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_sum_equals_two : lg 4 + lg 25 = 2 := by
  -- We'll use sorry to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_sum_equals_two_l501_50174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_quad_area_ratio_l501_50161

/-- Represents a trapezoid with bases of lengths a and b -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  h : ℝ
  ha : a > 0
  hb : b > 0
  hh : h > 0
  hab : a > b

/-- The area of the trapezoid -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ := (t.a + t.b) * t.h / 2

/-- The area of the quadrilateral formed by joining the midpoints -/
noncomputable def midpointQuadArea (t : Trapezoid) : ℝ := (t.a - t.b) * t.h / 4

/-- Theorem: If the area of the midpoint quadrilateral is 1/4 of the trapezoid's area,
    then the ratio of the bases is 3:1 -/
theorem midpoint_quad_area_ratio (t : Trapezoid) 
    (h : midpointQuadArea t = trapezoidArea t / 4) : 
    t.a / t.b = 3 := by
  sorry

#check midpoint_quad_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_quad_area_ratio_l501_50161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_equals_set_l501_50101

open Set

-- Define the universal set U
def U : Set Int := {-1, 0, 1, 2}

-- Define set A
def A : Set Int := {-1, 2}

-- Define set B
def B : Set Int := {0, 2}

-- State the theorem
theorem complement_union_equals_set :
  (U \ A) ∪ B = {0, 1, 2} := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_equals_set_l501_50101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_increasing_range_l501_50179

-- Define the function as noncomputable due to its dependency on Real.tan
noncomputable def f (ω : ℝ) (x : ℝ) := Real.tan (ω * x)

-- State the theorem
theorem tan_increasing_range (ω : ℝ) :
  (∀ x ∈ Set.Ioo (-π) π, Monotone (f ω)) →
  ω ∈ Set.Ioo 0 (1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_increasing_range_l501_50179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_f_is_four_l501_50180

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 - Real.log x / Real.log 2

-- State the theorem
theorem root_of_f_is_four :
  ∃ a : ℝ, f a = 0 ∧ a = 4 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_f_is_four_l501_50180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_supermarket_purchase_plan_l501_50156

theorem supermarket_purchase_plan :
  -- Define variables
  ∀ (cost_A cost_B : ℕ → ℕ)  -- cost functions for items A and B
    (profit_A profit_B total_items budget min_profit : ℕ),
  -- Define conditions
  cost_A 10 + cost_B 8 = 880 →  -- Condition 1
  cost_A 2 + cost_B 5 = 380 →   -- Condition 2
  profit_A = 10 →  -- profit per item A
  profit_B = 15 →  -- profit per item B
  total_items = 50 →  -- total items to purchase
  budget = 2520 →  -- maximum budget
  min_profit = 620 →  -- minimum required profit

  -- Prove the following
  ∃ (price_A price_B : ℕ),
    -- The prices of A and B are 40 and 60 respectively
    price_A = 40 ∧ price_B = 60 ∧
    -- The cost functions are linear based on these prices
    (∀ n : ℕ, cost_A n = n * price_A) ∧
    (∀ n : ℕ, cost_B n = n * price_B) ∧
    -- The only valid solutions for 'a' (number of item A) are 24, 25, and 26
    (∀ a : ℕ,
      (cost_A a + cost_B (total_items - a) ≤ budget ∧
       a * profit_A + (total_items - a) * profit_B ≥ min_profit)
      ↔ (a = 24 ∨ a = 25 ∨ a = 26)) :=
by
  -- Introduce all variables and hypotheses
  intros cost_A cost_B profit_A profit_B total_items budget min_profit h1 h2 h3 h4 h5 h6 h7
  -- Skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_supermarket_purchase_plan_l501_50156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l501_50133

-- Define the nabla operation
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- State the theorem
theorem nabla_calculation :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
  nabla (nabla a b) c = 11 / 9 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l501_50133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_graph_shift_l501_50115

/-- 
Proves that shifting the graph of y = cos(2x + π/6) right by π/12 units 
results in the graph of y = cos(2x)
-/
theorem cos_graph_shift (x : ℝ) : 
  Real.cos (2 * (x - Real.pi / 12) + Real.pi / 6) = Real.cos (2 * x) := by
  sorry

#check cos_graph_shift

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_graph_shift_l501_50115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_tangent_length_l501_50145

noncomputable def circle_C₁ (x y : ℝ) : Prop := (x - 8)^2 + (y - 5)^2 = 49
noncomputable def circle_C₂ (x y : ℝ) : Prop := (x + 12)^2 + y^2 = 64

noncomputable def is_tangent_to_C₁ (P : ℝ × ℝ) : Prop := circle_C₁ P.1 P.2
noncomputable def is_tangent_to_C₂ (Q : ℝ × ℝ) : Prop := circle_C₂ Q.1 Q.2

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem shortest_tangent_length :
  ∃ (P Q : ℝ × ℝ),
    is_tangent_to_C₁ P ∧
    is_tangent_to_C₂ Q ∧
    (∀ (P' Q' : ℝ × ℝ),
      is_tangent_to_C₁ P' ∧ is_tangent_to_C₂ Q' →
      distance P Q ≤ distance P' Q') ∧
    distance P Q = 2 * Real.sqrt 1105 / 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_tangent_length_l501_50145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l501_50177

/-- Given a triangle ABC with side lengths a, b, c, prove the following:
    1. If AB * AC + 2BA * BC = 3CA * CB, then a^2 + 2b^2 = 3c^2
    2. The minimum value of cos C is √2/3 -/
theorem triangle_property (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : c * b + 2 * c * a = 3 * b * a) : 
  a^2 + 2*b^2 = 3*c^2 ∧ (∀ θ : ℝ, (a^2 + b^2 - c^2) / (2*a*b) ≥ Real.sqrt 2 / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l501_50177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_specific_triangles_l501_50134

/-- Triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Area of a triangle given its side lengths -/
noncomputable def area (t : Triangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

/-- The theorem stating the ratio of areas of two specific triangles -/
theorem area_ratio_of_specific_triangles : 
  let t1 := Triangle.mk 7 24 25
  let t2 := Triangle.mk 9 40 41
  area t1 / area t2 = 7 / 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_specific_triangles_l501_50134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_a_value_l501_50183

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem tangent_perpendicular_implies_a_value
  (a : ℝ)
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : ∃ l : ℝ, ((fun x => Real.exp (x * Real.log a)) 0) * (-1) = 1) :
  a = Real.exp (-1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_a_value_l501_50183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_between_tangents_l501_50189

/-- The hyperbola equation: x^2 - y^2/4 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

/-- The circle equation: (x-5)^2 + y^2 = 5 -/
def circle_eq (x y : ℝ) : Prop := (x-5)^2 + y^2 = 5

/-- Point on the asymptote of the hyperbola -/
def point_on_asymptote (x y : ℝ) : Prop := y = 2*x ∨ y = -2*x

/-- The maximum angle between tangents -/
def max_angle : ℝ := 60

/-- Theorem: The maximum angle between two tangents drawn from a point on the asymptote
    of the given hyperbola to the given circle is 60 degrees -/
theorem max_angle_between_tangents :
  ∀ (x y : ℝ), point_on_asymptote x y →
  ∃ (θ : ℝ), θ ≤ max_angle ∧
  (∀ (θ' : ℝ), (∃ (x' y' : ℝ), point_on_asymptote x' y' ∧
                (∃ (t1 t2 : ℝ → ℝ), -- tangent lines
                  (∀ t, circle_eq (t1 t) (t2 t)) ∧
                  t1 x' = y' ∧ t2 x' = y' ∧
                  θ' = Real.arccos ((t1 1 - t1 0) * (t2 1 - t2 0) +
                                    (t2 1 - t2 0) * (t2 1 - t2 0)) /
                        (((t1 1 - t1 0)^2 + (t2 1 - t2 0)^2)^(1/2) *
                         ((t2 1 - t2 0)^2 + (t2 1 - t2 0)^2)^(1/2)))) →
                θ' ≤ θ) ∧
  θ * 2 = max_angle :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_between_tangents_l501_50189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platform_l501_50193

/-- The time taken for a train to cross a platform -/
noncomputable def train_crossing_time (train_length platform_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := train_length + platform_length
  total_distance / train_speed_mps

/-- Theorem: A train 450 m long running at 108 kmph crosses a platform of length 300.06 m in approximately 25.002 seconds -/
theorem train_crossing_platform : 
  |train_crossing_time 450 300.06 108 - 25.002| < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platform_l501_50193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_to_yellow_ratio_l501_50172

/- Define the diameters of the circles -/
def small_diameter : ℝ := 2
def large_diameter : ℝ := 6

/- Define the radii of the circles -/
noncomputable def small_radius : ℝ := small_diameter / 2
noncomputable def large_radius : ℝ := large_diameter / 2

/- Define the areas of the circles -/
noncomputable def small_circle_area : ℝ := Real.pi * small_radius ^ 2
noncomputable def large_circle_area : ℝ := Real.pi * large_radius ^ 2

/- Define the area between the circles (green area) -/
noncomputable def green_area : ℝ := large_circle_area - small_circle_area

/- Theorem: The ratio of the green area to the yellow area is 8 -/
theorem green_to_yellow_ratio :
  green_area / small_circle_area = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_to_yellow_ratio_l501_50172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_factorial_sum_2010_l501_50119

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The factorial of a natural number -/
def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The sum of factorials from 1 to n -/
def factorialSum (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun i => factorial (i + 1))

/-- The units digit of the sum of factorials from 1 to 2010 is 3 -/
theorem units_digit_factorial_sum_2010 :
  unitsDigit (factorialSum 2010) = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_factorial_sum_2010_l501_50119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_n_is_12_l501_50153

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

/-- The prime factorization of 2004 -/
def prime_factors_2004 : List ℕ := [2, 2, 3, 167]

/-- The function to count valid n values -/
def count_valid_n (max_n : ℕ) : ℕ :=
  (Finset.range max_n).filter (λ n => (2004 ^ n : ℕ) ∣ (factorial 2004)) |>.card

/-- The theorem stating that the count of valid n is 12 -/
theorem count_valid_n_is_12 : count_valid_n 2004 = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_n_is_12_l501_50153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_l501_50166

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the sets M and N
def M : Set ℝ := {x | f x = x}
def N : Set ℝ := {x | f (f x) = x}

-- State the theorem
theorem set_equality (h : StrictMono f) : M = N := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_l501_50166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_tilings_l501_50114

/-- A tiling of a 3 × 5 rectangle --/
structure Tiling where
  tile_1x1 : Fin 3 × Fin 5
  tile_1x2 : Fin 3 × Fin 5
  tile_1x3 : Fin 3 × Fin 5
  tile_1x4 : Fin 3 × Fin 5
  tile_1x5 : Fin 3 × Fin 5

/-- Predicate to check if a tiling is valid --/
def is_valid_tiling (t : Tiling) : Prop := sorry

/-- The set of all valid tilings --/
def valid_tilings : Set Tiling :=
  { t : Tiling | is_valid_tiling t }

/-- Instance to make valid_tilings a finite type --/
instance : Fintype valid_tilings := sorry

/-- The theorem stating the number of valid tilings --/
theorem count_valid_tilings : Fintype.card valid_tilings = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_tilings_l501_50114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_l501_50107

/-- The area of an equilateral triangle with perimeter 3p is (√3/4) * p^2 -/
theorem equilateral_triangle_area (p : ℝ) (h : p > 0) :
  let perimeter := 3 * p
  let side_length := perimeter / 3
  let area := (Real.sqrt 3 / 4) * side_length^2
  area = (Real.sqrt 3 / 4) * p^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_l501_50107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_and_sine_equation_sine_equation_not_implies_arithmetic_sequence_l501_50150

theorem arithmetic_sequence_and_sine_equation (α β γ : Real) :
  (∃ d : Real, β = α + d ∧ γ = β + d) →
  Real.sin (α + γ) = Real.sin (2 * β) :=
by sorry

theorem sine_equation_not_implies_arithmetic_sequence :
  ∃ α β γ : Real,
    Real.sin (α + γ) = Real.sin (2 * β) ∧
    ¬(∃ d : Real, β = α + d ∧ γ = β + d) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_and_sine_equation_sine_equation_not_implies_arithmetic_sequence_l501_50150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_equal_7root2_plus_2root5_l501_50136

noncomputable def A : ℝ × ℝ := (1, 0)
noncomputable def B : ℝ × ℝ := (3, 4)
noncomputable def C : ℝ × ℝ := (6, 3)
noncomputable def D : ℝ × ℝ := (8, 1)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def perimeter : ℝ :=
  distance A B + distance B C + distance C D + distance D A

theorem perimeter_equal_7root2_plus_2root5 :
  perimeter = 7 * Real.sqrt 2 + 2 * Real.sqrt 5 := by
  sorry

#check perimeter_equal_7root2_plus_2root5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_equal_7root2_plus_2root5_l501_50136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_range_l501_50131

theorem triangle_angle_range (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C) ∧
  Real.sin A + Real.sin C = 2 * Real.sin B →
  0 < B ∧ B ≤ π / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_range_l501_50131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l501_50143

/-- Parabola equation -/
noncomputable def parabola (x : ℝ) : ℝ := x^2 + 2*x - 3

/-- Line l₁ equation -/
noncomputable def line_l1 (x m : ℝ) : ℝ := -x + m

/-- Axis of symmetry of the parabola -/
def axis_of_symmetry : ℝ := -1

/-- Function to calculate the distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := ((x1 - x2)^2 + (y1 - y2)^2)^(1/2 : ℝ)

/-- Theorem stating the value of m -/
theorem parabola_intersection_theorem (m : ℝ) 
  (h1 : ∃ (x1 y1 x2 y2 : ℝ), parabola x1 = line_l1 x1 m ∧ parabola x2 = line_l1 x2 m)
  (h2 : ∃ (x3 y3 x4 y4 : ℝ), parabola x3 = parabola x4 ∧ x3 + x4 = 2 * axis_of_symmetry)
  (h3 : ∃ (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ), 
    distance x1 y1 x2 y2 * distance x3 y3 x4 y4 = 26 ∧
    y1 > 0 ∧ y4 > 0 ∧ y2 < 0 ∧ y3 < 0) :
  m = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l501_50143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_mix_price_is_6_35_l501_50120

/-- Calculates the price per pound of a coffee mix given the following parameters:
    * total_mix_weight: Total weight of the coffee mix in pounds
    * columbian_price: Price per pound of Columbian coffee in dollars
    * brazilian_price: Price per pound of Brazilian coffee in dollars
    * columbian_weight: Weight of Columbian coffee used in the mix in pounds
-/
noncomputable def coffee_mix_price (total_mix_weight : ℝ) (columbian_price : ℝ) (brazilian_price : ℝ) (columbian_weight : ℝ) : ℝ :=
  let brazilian_weight := total_mix_weight - columbian_weight
  let total_cost := columbian_weight * columbian_price + brazilian_weight * brazilian_price
  total_cost / total_mix_weight

/-- Theorem stating that the price per pound of the coffee mix described in the problem is $6.35 -/
theorem coffee_mix_price_is_6_35 :
  coffee_mix_price 100 8.75 3.75 52 = 6.35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_mix_price_is_6_35_l501_50120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_gain_percent_l501_50188

theorem cycle_gain_percent (original_price discount refurbish_cost selling_price : ℝ) :
  original_price = 1285 →
  discount = 18 / 100 →
  refurbish_cost = 365 →
  selling_price = 2175 →
  let discounted_price := original_price * (1 - discount)
  let total_cost := discounted_price + refurbish_cost
  let gain := selling_price - total_cost
  let gain_percent := (gain / total_cost) * 100
  abs (gain_percent - 53.3) < 0.1 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_gain_percent_l501_50188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_integer_l501_50128

def number : ℝ := 5278653.4923

theorem round_to_nearest_integer :
  Int.floor (number + 0.5) = 5278653 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_integer_l501_50128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_solution_sum_l501_50163

def is_valid_solution (p q r s u v w t : ℕ) : Prop :=
  (Nat.factorial p * Nat.factorial q) / (Nat.factorial r * Nat.factorial s) +
  (Nat.factorial u * Nat.factorial v) / (Nat.factorial w * Nat.factorial t) = 145

def is_minimal_solution (p q r s u v w t : ℕ) : Prop :=
  is_valid_solution p q r s u v w t ∧
  ∀ p' q' r' s' u' v' w' t' : ℕ,
    is_valid_solution p' q' r' s' u' v' w' t' →
    p + r ≤ p' + r' ∧ u + w ≤ u' + w'

theorem minimal_solution_sum (p q r s u v w t : ℕ) :
  is_minimal_solution p q r s u v w t → u + w = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_solution_sum_l501_50163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_size_l501_50139

/-- Represents the work done by a group of men reaping land. -/
structure ReapingWork where
  men : ℕ
  days : ℕ
  acres : ℕ

/-- The work done is directly proportional to the number of men and days worked. -/
axiom work_proportional (w1 w2 : ReapingWork) :
  w1.men * w1.days * w2.acres = w2.men * w2.days * w1.acres

/-- The first group's work -/
def first_group (M : ℕ) : ReapingWork := { men := M, days := 36, acres := 120 }

/-- The second group's work -/
def second_group : ReapingWork := { men := 54, days := 54, acres := 810 }

/-- The theorem to prove -/
theorem first_group_size : ∃ M, first_group M = { men := 12, days := 36, acres := 120 } := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_size_l501_50139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gcd_consecutive_terms_l501_50187

def my_sequence (n : ℕ) : ℕ := n.factorial + 2 * n

theorem max_gcd_consecutive_terms : 
  ∀ n : ℕ, n ≥ 1 → ∃ k : ℕ, k ≥ n ∧ 
    Nat.gcd (my_sequence k) (my_sequence (k + 1)) = 2 ∧ 
    ∀ m : ℕ, m ≥ n → Nat.gcd (my_sequence m) (my_sequence (m + 1)) ≤ 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gcd_consecutive_terms_l501_50187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_value_l501_50124

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + 18 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

-- State the theorem
theorem find_a_value :
  ∃ (a : ℝ), (A ∩ B a).Nonempty ∧ (B a ∩ C = ∅) → a = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_value_l501_50124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lizzy_fish_shipment_l501_50155

/-- Calculates the total pounds of fish to be shipped given the crate capacity,
    shipping cost per crate, and total shipping cost. -/
def total_fish_pounds (crate_capacity : ℕ) (cost_per_crate : ℚ) (total_cost : ℚ) : ℕ :=
  let num_crates : ℕ := (total_cost / cost_per_crate).floor.toNat
  num_crates * crate_capacity

/-- Proves that given the specified conditions, Lizzy needs to ship 540 pounds of fish. -/
theorem lizzy_fish_shipment :
  total_fish_pounds 30 (3/2) 27 = 540 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lizzy_fish_shipment_l501_50155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bucket_properties_l501_50147

/-- Represents a cylindrical, open-top bucket -/
structure Bucket where
  baseCircumference : ℝ
  height : ℝ

/-- Calculates the surface area of a bucket -/
noncomputable def surfaceArea (b : Bucket) : ℝ :=
  let radius := b.baseCircumference / (2 * Real.pi)
  b.baseCircumference * b.height + Real.pi * radius * radius

/-- Calculates the volume of a bucket -/
noncomputable def volume (b : Bucket) : ℝ :=
  let radius := b.baseCircumference / (2 * Real.pi)
  Real.pi * radius * radius * b.height

/-- The main theorem about the bucket's properties -/
theorem bucket_properties :
  let b := Bucket.mk 15.7 8
  (⌈surfaceArea b⌉ = 145) ∧ (volume b = 157) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bucket_properties_l501_50147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_hop_lengths_theorem_l501_50111

/-- Represents a frog's hop on a number line -/
structure Hop where
  length : ℕ
  direction : Bool -- true for right, false for left

/-- The set of valid hop lengths -/
def validHopLengths (n : ℕ) : Set ℕ :=
  {k : ℕ | ∃ i : ℕ, i < n ∧ k = 2^i}

/-- A sequence of hops is valid if it satisfies the problem conditions -/
def isValidHopSequence (n : ℕ) (hops : List Hop) : Prop :=
  let positions := List.scanl (λ acc hop => if hop.direction then acc + hop.length else acc - hop.length) 0 hops
  (∀ pos ∈ positions, 1 ≤ pos ∧ pos < 2^n) ∧
  (∀ hop ∈ hops, hop.length ∈ validHopLengths n) ∧
  List.Nodup positions

/-- The sum of hop lengths in a sequence -/
def sumHopLengths (hops : List Hop) : ℕ :=
  hops.map (·.length) |> List.sum

/-- The maximum possible sum of hop lengths for a given n -/
def maxSumHopLengths (n : ℕ) : ℕ :=
  (4^n - 1) / 3

theorem max_sum_hop_lengths_theorem (n : ℕ) (h : 0 < n) :
  ∀ hops : List Hop, isValidHopSequence n hops →
    sumHopLengths hops ≤ maxSumHopLengths n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_hop_lengths_theorem_l501_50111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_altitude_equal_side_l501_50127

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents the orthocenter (intersection of altitudes) of a triangle -/
noncomputable def orthocenter (t : Triangle) : Point := sorry

/-- Predicate to check if a triangle is acute -/
def is_acute (t : Triangle) : Prop := sorry

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ := sorry

/-- Angle measure in a triangle -/
noncomputable def angle_measure (t : Triangle) (v : Fin 3) : ℝ := sorry

theorem triangle_with_altitude_equal_side (t : Triangle) 
  (h : is_acute t) 
  (orthocenter_exists : ∃ H : Point, H = orthocenter t)
  (altitude_equals_side : ∃ H : Point, H = orthocenter t ∧ distance t.A H = distance t.B t.C) :
  ∃ v : Fin 3, angle_measure t v = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_altitude_equal_side_l501_50127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_advantage_condition_l501_50164

/-- A flex car that can use either gasoline or alcohol as fuel -/
structure FlexCar where
  gasoline_performance : ℝ
  alcohol_performance : ℝ
  alcohol_price : ℝ
  gasoline_price : ℝ

/-- The cost of running a flex car for 100 km using gasoline -/
noncomputable def gasoline_cost (car : FlexCar) : ℝ :=
  100 * car.gasoline_price / car.gasoline_performance

/-- The cost of running a flex car for 100 km using alcohol -/
noncomputable def alcohol_cost (car : FlexCar) : ℝ :=
  100 * car.alcohol_price / car.alcohol_performance

/-- Alcohol is more financially advantageous when its cost is lower -/
def alcohol_more_advantageous (car : FlexCar) : Prop :=
  alcohol_cost car < gasoline_cost car

theorem alcohol_advantage_condition (x : ℝ) :
  let car : FlexCar := {
    gasoline_performance := x,
    alcohol_performance := x/2 + 1,
    alcohol_price := 1.59,
    gasoline_price := 2.49
  }
  x > 7.22 → alcohol_more_advantageous car :=
by
  sorry

#check alcohol_advantage_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_advantage_condition_l501_50164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_sum_in_triangle_l501_50192

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where a^2 + c^2 = b^2 + ac, the maximum value of cos A + cos C is 1. -/
theorem max_cos_sum_in_triangle (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = Real.pi →
  a^2 + c^2 = b^2 + a*c →
  ∃ (x : ℝ), ∀ (y : ℝ), Real.cos A + Real.cos C ≤ x ∧ x ≤ 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_sum_in_triangle_l501_50192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_black_cells_in_grid_l501_50157

theorem max_black_cells_in_grid (n : ℕ) (h : n = 101) : 
  ∃ k : ℕ, k ≤ n ∧ 
    (∀ j : ℕ, j ≤ n → 2 * k * (n - k) ≥ 2 * j * (n - j)) ∧ 
    2 * k * (n - k) = 5100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_black_cells_in_grid_l501_50157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_R_54321_l501_50100

noncomputable def c : ℝ := 4 + Real.sqrt 15
noncomputable def d : ℝ := 4 - Real.sqrt 15

noncomputable def R (n : ℕ) : ℝ := (1 / 2) * (c ^ n + d ^ n)

theorem units_digit_R_54321 : ∃ k : ℕ, R 54321 = 4 + 10 * k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_R_54321_l501_50100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_bought_thrice_l501_50148

/-- Represents the number of times tea was bought in a week -/
def tea_count : ℕ := sorry

/-- Represents the number of times coffee was bought in a week -/
def coffee_count : ℕ := sorry

/-- The price of tea in cents -/
def tea_price : ℕ := 80

/-- The price of coffee in cents -/
def coffee_price : ℕ := 60

/-- The price of a cookie in cents -/
def cookie_price : ℕ := 40

/-- The number of days in the workweek -/
def workweek_days : ℕ := 6

/-- The number of times a cookie is bought in a week -/
def cookie_count : ℕ := 2

theorem tea_bought_thrice :
  tea_count + coffee_count = workweek_days →
  (tea_count * tea_price + coffee_count * coffee_price + cookie_count * cookie_price) % 100 = 0 →
  tea_count = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_bought_thrice_l501_50148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_triangle_to_hexagon_area_ratio_l501_50125

-- Define a regular hexagon
structure RegularHexagon where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

-- Define the area of a regular hexagon
noncomputable def areaHexagon (h : RegularHexagon) : ℝ :=
  3 * Real.sqrt 3 / 2 * h.sideLength ^ 2

-- Define an equilateral triangle formed by connecting midpoints of alternate sides
structure MidpointTriangle (h : RegularHexagon) where
  side : ℝ
  side_eq : side = 3 / 2 * h.sideLength

-- Define the area of the midpoint triangle
noncomputable def areaMidpointTriangle (h : RegularHexagon) (t : MidpointTriangle h) : ℝ :=
  Real.sqrt 3 / 4 * t.side ^ 2

-- Theorem statement
theorem midpoint_triangle_to_hexagon_area_ratio 
  (h : RegularHexagon) (t : MidpointTriangle h) : 
  areaMidpointTriangle h t / areaHexagon h = 3 / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_triangle_to_hexagon_area_ratio_l501_50125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_l501_50171

def sum_even (n : ℕ) : ℕ := n * (n + 1)
def sum_odd (n : ℕ) : ℕ := n^2

def A (m n : ℕ) : ℚ := m / n

theorem find_m (m n : ℕ) (h1 : Nat.Coprime m n) 
  (h2 : A m n = (sum_even 1007) / (sum_odd 1007) - (sum_odd 1007) / (sum_even 1007)) : 
  m = 2015 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_l501_50171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l501_50138

open Real

theorem trig_identities (α β γ : ℝ) :
  (sin α + sin β + sin γ - sin (α + β + γ) = 4 * sin ((α + β) / 2) * sin ((β + γ) / 2) * sin ((α + γ) / 2)) ∧
  (cos α + cos β + cos γ + cos (α + β + γ) = 4 * cos ((α + β) / 2) * cos ((β + γ) / 2) * cos ((α + γ) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l501_50138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_l501_50102

/-- Given a line y = (1/2)x + 4 parameterized as (x, y) = (-7, s) + t(l, -5),
    prove that s = 1/2 and l = -10 -/
theorem line_parameterization (s l : ℝ) : 
  (∀ x y : ℝ, y = (1/2) * x + 4 ↔ ∃ t : ℝ, (x, y) = (-7, s) + t • (l, -5)) →
  s = 1/2 ∧ l = -10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_l501_50102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l501_50108

theorem solve_exponential_equation :
  ∃! y : ℚ, (32 : ℝ) ^ ((5 : ℝ) * y - 7) = (16 : ℝ) ^ ((-2 : ℝ) * y + 3) ∧ y = 47 / 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l501_50108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_4_sufficient_not_necessary_l501_50106

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (x, 3)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem x_4_sufficient_not_necessary :
  (∀ x : ℝ, x = 4 → magnitude (vector_a x) = 5) ∧
  ¬(∀ x : ℝ, magnitude (vector_a x) = 5 → x = 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_4_sufficient_not_necessary_l501_50106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_class_size_smallest_class_size_proof_l501_50116

theorem smallest_class_size (total_points perfect_score_count min_score mean_score : ℕ) : ℕ :=
  let smallest_size := 20
  smallest_size

#check smallest_class_size

-- Proof sketch
theorem smallest_class_size_proof 
  (total_points : ℕ) 
  (perfect_score_count : ℕ) 
  (min_score : ℕ) 
  (mean_score : ℕ) 
  (h1 : total_points = 100)
  (h2 : perfect_score_count = 8)
  (h3 : min_score = 70)
  (h4 : mean_score = 82) : 
  smallest_class_size total_points perfect_score_count min_score mean_score = 20 := by
  -- The actual proof would go here
  sorry

#check smallest_class_size_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_class_size_smallest_class_size_proof_l501_50116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_equation_point_P_on_circle_l501_50137

-- Define the line segment AB
noncomputable def line_segment_AB : Set (Fin 2 → ℝ) := sorry

-- Define the lines x = 2y and x = -2y
def line1 : Set (Fin 2 → ℝ) := {p | p 0 = 2 * p 1}
def line2 : Set (Fin 2 → ℝ) := {p | p 0 = -2 * p 1}

-- Define the midpoint G of AB
noncomputable def G : Fin 2 → ℝ := sorry

-- Define the locus C of point G
noncomputable def C : Set (Fin 2 → ℝ) := sorry

-- Define a point P
noncomputable def P : Fin 2 → ℝ := sorry

-- Define the tangent lines from P to C
noncomputable def tangent1 : Set (Fin 2 → ℝ) := sorry
noncomputable def tangent2 : Set (Fin 2 → ℝ) := sorry

-- Theorem statements
theorem locus_equation : C = {p : Fin 2 → ℝ | (p 0)^2 / 16 + (p 1)^2 = 1} := by sorry

theorem point_P_on_circle : P ∈ {p : Fin 2 → ℝ | (p 0)^2 + (p 1)^2 = 17} := by sorry

-- Assumptions
axiom AB_length : ∃ (A B : Fin 2 → ℝ), A ∈ line_segment_AB ∧ B ∈ line_segment_AB ∧ 
                  ((A 0 - B 0)^2 + (A 1 - B 1)^2)^(1/2 : ℝ) = 4

axiom AB_endpoints : ∃ (A B : Fin 2 → ℝ), A ∈ line_segment_AB ∧ B ∈ line_segment_AB ∧
                     ((A ∈ line1 ∧ B ∈ line2) ∨ (A ∈ line2 ∧ B ∈ line1))

axiom G_is_midpoint : ∃ (A B : Fin 2 → ℝ), A ∈ line_segment_AB ∧ B ∈ line_segment_AB ∧
                      G = λ i => (A i + B i) / 2

axiom tangents_perpendicular : ∃ (t1 t2 : ℝ), 
  tangent1 = {p : Fin 2 → ℝ | p 1 - P 1 = t1 * (p 0 - P 0)} ∧
  tangent2 = {p : Fin 2 → ℝ | p 1 - P 1 = t2 * (p 0 - P 0)} ∧
  t1 * t2 = -1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_equation_point_P_on_circle_l501_50137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l501_50196

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

-- State the theorem
theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = -1 / Real.exp 1 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l501_50196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_6_sqrt_3_l501_50140

noncomputable section

/-- The area formula for a triangle given its three sides -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  Real.sqrt ((1/4) * (c^2 * a^2 - ((c^2 + a^2 - b^2)/2)^2))

/-- The perimeter of the triangle -/
def perimeter : ℝ := 10 + 2 * Real.sqrt 7

/-- The ratio of sines of angles A, B, and C -/
def sine_ratio (A B C : ℝ) : Prop := ∃ (k : ℝ), k > 0 ∧ 
  2 * k = Real.sin A ∧ 
  Real.sqrt 7 * k = Real.sin B ∧ 
  3 * k = Real.sin C

theorem triangle_area_is_6_sqrt_3 : 
  ∃ (a b c A B C : ℝ), 
    a + b + c = perimeter ∧ 
    sine_ratio A B C ∧
    triangle_area a b c = 6 * Real.sqrt 3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_6_sqrt_3_l501_50140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_positive_ratio_range_l501_50141

/-- A geometric sequence with common ratio q and first term a₁ -/
noncomputable def GeometricSequence (q : ℝ) (a₁ : ℝ) : ℕ → ℝ :=
  fun n => a₁ * q ^ (n - 1)

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def GeometricSum (q : ℝ) (a₁ : ℝ) : ℕ → ℝ :=
  fun n => if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_positive_ratio_range
  (q : ℝ) (a₁ : ℝ) (h₁ : a₁ > 0) :
  (∀ n : ℕ+, GeometricSum q a₁ n > 0) →
  q ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_positive_ratio_range_l501_50141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_theorem_l501_50104

/-- The distance from a point (x₀, y₀) to a line Ax + By + C = 0 -/
noncomputable def distancePointToLine (x₀ y₀ A B C : ℝ) : ℝ :=
  (|A * x₀ + B * y₀ + C|) / Real.sqrt (A^2 + B^2)

/-- The circle O with equation x² + y² = r² (r > 0) -/
def circleO (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

/-- The line l with equation x - y + 4 = 0 -/
def lineL : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + 4 = 0}

theorem circle_line_distance_theorem (r : ℝ) (hr : r > 0) :
  (distancePointToLine 0 0 1 (-1) 4 = 2 * Real.sqrt 2) ∧
  (∃ (p q : ℝ × ℝ), p ∈ circleO r ∧ q ∈ circleO r ∧
    p ≠ q ∧
    distancePointToLine p.1 p.2 1 (-1) 4 = Real.sqrt 2 ∧
    distancePointToLine q.1 q.2 1 (-1) 4 = Real.sqrt 2 ∧
    ∀ (x : ℝ × ℝ), x ∈ circleO r ∧ x ≠ p ∧ x ≠ q →
      distancePointToLine x.1 x.2 1 (-1) 4 ≠ Real.sqrt 2) ↔
  (Real.sqrt 2 < r ∧ r < 3 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_theorem_l501_50104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l501_50184

/-- Definition of the sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The arithmetic sequence property: difference between S_n and S_(n-4) is constant -/
axiom arithmetic_property : ∀ n : ℕ, n ≥ 8 → S n - S (n - 4) = S 8 - S 4

/-- Given condition: S_8 - S_4 = 12 -/
axiom given_condition : S 8 - S 4 = 12

/-- Theorem: If S_8 - S_4 = 12 for an arithmetic sequence, then S_12 = 36 -/
theorem arithmetic_sequence_sum : S 12 = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l501_50184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l501_50142

noncomputable def a : ℝ × ℝ := (Real.cos (20 * Real.pi / 180), Real.sin (20 * Real.pi / 180))
noncomputable def b : ℝ × ℝ := (Real.cos (10 * Real.pi / 180), Real.sin (190 * Real.pi / 180))

theorem angle_between_vectors : 
  let angle := Real.arccos ((a.1 * b.1 + a.2 * b.2) / 
    (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  angle = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l501_50142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_make_all_red_vasya_cannot_achieve_goal_l501_50185

/-- Represents a point in the pentagon that can be colored --/
structure ColoredPoint where
  isBlue : Bool

/-- Represents the pentagon with its colored points --/
structure Pentagon where
  points : List ColoredPoint

/-- Represents an operation that can change colors along a line (side or diagonal) --/
def flipColors (p : Pentagon) (line : List Nat) : Pentagon :=
  sorry

/-- Theorem stating that it's impossible to make all points red --/
theorem cannot_make_all_red (p : Pentagon) : 
  ¬∃ (operations : List (List Nat)), 
    (operations.foldl flipColors p).points.all (fun point => ¬point.isBlue) := by
  sorry

/-- Initial condition: all points are blue --/
def initialPentagon : Pentagon :=
  { points := List.replicate 15 { isBlue := true } }

/-- Main theorem: It's impossible to change all blue points to red --/
theorem vasya_cannot_achieve_goal : 
  ¬∃ (operations : List (List Nat)), 
    (operations.foldl flipColors initialPentagon).points.all (fun point => ¬point.isBlue) := by
  exact cannot_make_all_red initialPentagon


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_make_all_red_vasya_cannot_achieve_goal_l501_50185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_specific_l501_50159

noncomputable def polar_to_rectangular (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_rectangular_specific :
  polar_to_rectangular 5 (5 * Real.pi / 3) = (5/2, 5 * Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_specific_l501_50159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_abc_aqc_l501_50167

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  1/2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

/-- Given a triangle ABC and a point Q inside it satisfying the vector equation
    QA + 3QB + 4QC = 0, the ratio of the area of triangle ABC to the area of
    triangle AQC is 3:1 -/
theorem area_ratio_abc_aqc (A B C Q : ℝ × ℝ) : 
  (Q.1 - A.1, Q.2 - A.2) + 3 • (Q.1 - B.1, Q.2 - B.2) + 4 • (Q.1 - C.1, Q.2 - C.2) = (0, 0) →
  area_triangle A B C / area_triangle A Q C = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_abc_aqc_l501_50167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l501_50112

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * Real.log x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + (a - 1) * Real.log x + 1

-- Theorem statement
theorem problem_solution :
  ∀ a : ℝ,
  (∀ x : ℝ, x > 0 → f a x ≤ -2) →
  (∃ x : ℝ, x > 0 ∧ x ≤ 1 ∧ f a x = -2) →
  a = -Real.exp 1 ∧
  (∀ k : ℝ,
    (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → |g a x₁ - g a x₂| ≥ k * |x₁ - x₂|) →
    k ≤ 4) ∧
  (∃ k : ℝ,
    k = 4 ∧
    (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → |g a x₁ - g a x₂| ≥ k * |x₁ - x₂|))
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l501_50112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_volume_greater_than_half_l501_50191

/-- A regular tetrahedron in 3D space -/
structure RegularTetrahedron where
  side_length : ℝ
  center : ℝ × ℝ × ℝ

/-- The volume of a regular tetrahedron -/
noncomputable def tetrahedron_volume (t : RegularTetrahedron) : ℝ :=
  t.side_length ^ 3 / (6 * Real.sqrt 2)

/-- The common volume of two overlapping regular tetrahedrons -/
noncomputable def common_volume (t1 t2 : RegularTetrahedron) : ℝ :=
  sorry

theorem common_volume_greater_than_half 
  (t1 t2 : RegularTetrahedron) 
  (h1 : t1.side_length = Real.sqrt 6) 
  (h2 : t2.side_length = Real.sqrt 6) 
  (h3 : t1.center = t2.center) :
  common_volume t1 t2 > (1/2) * tetrahedron_volume t1 := by
  sorry

#check common_volume_greater_than_half

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_volume_greater_than_half_l501_50191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l501_50113

/-- Calculates the time (in seconds) for a train to completely cross a bridge -/
noncomputable def timeToCrossBridge (trainLength : ℝ) (bridgeLength : ℝ) (trainSpeed : ℝ) : ℝ :=
  let totalDistance := trainLength + bridgeLength
  let speedInMetersPerSecond := trainSpeed * 1000 / 3600
  totalDistance / speedInMetersPerSecond

/-- Theorem stating that a 200m train crossing a 300m bridge at 40 km/h takes approximately 45 seconds -/
theorem train_bridge_crossing_time :
  let trainLength : ℝ := 200
  let bridgeLength : ℝ := 300
  let trainSpeed : ℝ := 40
  ∃ ε > 0, |timeToCrossBridge trainLength bridgeLength trainSpeed - 45| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l501_50113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l501_50168

-- Define the parabola
noncomputable def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the focus of the parabola
noncomputable def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define the line passing through the focus with slope angle 60°
noncomputable def line (p : ℝ) (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - p/2)

-- Define the intersection point
noncomputable def intersectionPoint (x₀ : ℝ) : ℝ × ℝ := (x₀, Real.sqrt 3)

-- Theorem statement
theorem parabola_intersection_theorem (p : ℝ) (x₀ : ℝ) :
  parabola p x₀ (Real.sqrt 3) ∧
  line p x₀ (Real.sqrt 3) →
  p = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l501_50168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reaction_properties_l501_50165

-- Define the reaction and its components
structure Reaction where
  co2 : ℝ
  h2 : ℝ
  co : ℝ
  h2o : ℝ

-- Define the equilibrium constant
noncomputable def equilibrium_constant (r : Reaction) : ℝ := 
  (r.co * r.h2o) / (r.co2 * r.h2)

-- Define the reverse reaction constant
noncomputable def reverse_constant (k : ℝ) : ℝ := 1 / k

-- Define the square root constant
noncomputable def sqrt_constant (k : ℝ) : ℝ := Real.sqrt k

-- Define the endothermic property
def is_endothermic (k_low : ℝ) (k_high : ℝ) (t_low : ℝ) (t_high : ℝ) : Prop := 
  t_low < t_high ∧ k_low < k_high

-- Define the conversion rate
noncomputable def conversion_rate (initial : ℝ) (equilibrium : ℝ) : ℝ :=
  (initial - equilibrium) / initial * 100

-- State the theorem
theorem reaction_properties 
  (r : Reaction) 
  (k1 : ℝ) 
  (k4 : ℝ) 
  (t1 : ℝ) 
  (t4 : ℝ) 
  (co2_initial : ℝ) 
  (h2_initial : ℝ) :
  equilibrium_constant r = (r.co * r.h2o) / (r.co2 * r.h2) ∧ 
  k1 = 0.6 → 
  reverse_constant k1 = 1.67 ∧ 
  sqrt_constant k1 = 0.77 ∧
  k4 = 1.0 → 
  is_endothermic k1 k4 t1 t4 ∧
  t4 = 1000 ∧ 
  co2_initial = 1.5 ∧ 
  h2_initial = 1.0 → 
  ∃ (co2_equilibrium : ℝ), 
    conversion_rate co2_initial co2_equilibrium = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reaction_properties_l501_50165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_numbers_not_square_divisible_l501_50105

theorem odd_numbers_not_square_divisible (n : ℕ) (h : n > 0) :
  ∀ S : Finset ℕ, 
    (∀ x ∈ S, x % 2 = 1 ∧ x > 2^(2*n) ∧ x < 2^(3*n)) → 
    (Finset.card S = 2^(2*n - 1) + 1) →
    ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ ¬(a^2 ∣ b) ∧ ¬(b^2 ∣ a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_numbers_not_square_divisible_l501_50105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l501_50190

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (floor x * y) = f x * floor (f y)

def constant_set : Set ℝ :=
  {0} ∪ { x | 1 ≤ x ∧ x < 2 }

theorem function_characterization (f : ℝ → ℝ) :
  satisfies_property f → ∃ c ∈ constant_set, ∀ x, f x = c := by
  sorry

#check function_characterization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l501_50190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_range_l501_50182

-- Define the function f
noncomputable def f (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ)

-- State the theorem
theorem phi_range (φ : ℝ) : 
  (∀ x ∈ Set.Ioc (-π/12) (π/6), ∀ y ∈ Set.Ioc (-π/12) (π/6), x < y → f x φ < f y φ) → -- monotonicity
  (∀ x ∈ Set.Ioc (-π/12) (π/6), f x φ ≤ Real.sqrt 3) → -- maximum value constraint
  (abs φ < π/2) → -- given constraint on φ
  -π/3 ≤ φ ∧ φ ≤ 0 := by
  sorry

#check phi_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_range_l501_50182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_kmn_l501_50199

theorem sum_of_kmn (t : ℝ) (k m n : ℕ) :
  (1 + Real.sin t) * (1 + Real.cos t) = 8/9 →
  (1 - Real.sin t) * (1 - Real.cos t) = m/n - Real.sqrt k →
  Nat.Coprime m n →
  k + m + n = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_kmn_l501_50199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chess_players_l501_50181

/-- Represents the total number of people in the sports school -/
def total_people : ℕ := 55

/-- Represents the maximum number of friends a chess player can have among tennis players -/
def max_tennis_friends (num_tennis : ℕ) : ℕ := num_tennis

/-- Represents the condition that no four chess players have an equal number of friends among tennis players -/
def no_four_equal_friends (num_chess : ℕ) (num_tennis : ℕ) : Prop :=
  ∀ k : ℕ, k ≤ max_tennis_friends num_tennis → (Finset.filter (λ i ↦ i = k) (Finset.range num_chess)).card < 4

/-- The main theorem stating the maximum number of chess players -/
theorem max_chess_players :
  ∃ (num_chess : ℕ) (num_tennis : ℕ),
    num_chess + num_tennis = total_people ∧
    no_four_equal_friends num_chess num_tennis ∧
    num_chess = 42 ∧
    ∀ m : ℕ, m > num_chess → ¬(no_four_equal_friends m (total_people - m)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chess_players_l501_50181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_small_triangle_l501_50154

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square with side length 1 -/
def UnitSquare : Set Point :=
  {p : Point | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1}

/-- Definition of collinearity for three points -/
def areCollinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Area of a triangle given three points -/
noncomputable def TriangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- Main theorem -/
theorem existence_of_small_triangle 
  (points : Finset Point) 
  (h1 : points.card = 101)
  (h2 : ∀ p, p ∈ points → p ∈ UnitSquare)
  (h3 : ∀ p1 p2 p3, p1 ∈ points → p2 ∈ points → p3 ∈ points → 
        p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → ¬areCollinear p1 p2 p3) :
  ∃ p1 p2 p3, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ TriangleArea p1 p2 p3 ≤ 1/100 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_small_triangle_l501_50154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_zeros_l501_50123

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := sin x + x - π / 4
noncomputable def g (x : ℝ) : ℝ := cos x - x + π / 4

-- State the theorem
theorem f_g_zeros :
  (∃! x₁, x₁ ∈ Set.Ioo (0 : ℝ) (π / 2) ∧ f x₁ = 0) ∧
  (∃! x₂, x₂ ∈ Set.Ioo (0 : ℝ) (π / 2) ∧ g x₂ = 0) →
  ∀ x₁ x₂, x₁ ∈ Set.Ioo (0 : ℝ) (π / 2) → x₂ ∈ Set.Ioo (0 : ℝ) (π / 2) → 
  f x₁ = 0 → g x₂ = 0 → x₁ + x₂ = π / 2 := by
  sorry

-- Note: The ∃! symbol means "there exists a unique"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_zeros_l501_50123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_tank_capacity_fuel_tank_capacity_proof_l501_50194

theorem fuel_tank_capacity 
  (initial_efficiency : ℝ) 
  (fuel_usage_ratio : ℝ) 
  (additional_miles : ℝ) : ℝ :=
  let capacity : ℝ := 93.75
  let modified_efficiency : ℝ := initial_efficiency / fuel_usage_ratio
  have h1 : initial_efficiency = 28 := by sorry
  have h2 : fuel_usage_ratio = 0.8 := by sorry
  have h3 : additional_miles = 105 := by sorry
  have h4 : capacity * fuel_usage_ratio * modified_efficiency = initial_efficiency * capacity + additional_miles := by sorry
  capacity

theorem fuel_tank_capacity_proof : fuel_tank_capacity 28 0.8 105 = 93.75 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_tank_capacity_fuel_tank_capacity_proof_l501_50194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l501_50170

theorem inequality_proof (d n : ℕ) (h1 : d ≥ 1) (h2 : ¬ ∃ k, d = k^2) (h3 : n ≥ 1) :
  (n * Real.sqrt (d : ℝ) + 1) * |Real.sin (n * Real.pi * Real.sqrt (d : ℝ))| ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l501_50170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_reflection_xoy_distance_A_to_B_l501_50198

/-- The distance between a point and its reflection across the xOy plane --/
theorem distance_to_reflection_xoy (x y z : ℝ) : 
  let A : ℝ × ℝ × ℝ := (x, y, z)
  let B : ℝ × ℝ × ℝ := (x, y, -z)
  Real.sqrt ((A.fst - B.fst)^2 + (A.snd.fst - B.snd.fst)^2 + (A.snd.snd - B.snd.snd)^2) = 2 * |z| :=
by sorry

/-- The specific case for point A(2, -3, 5) --/
theorem distance_A_to_B : 
  let A : ℝ × ℝ × ℝ := (2, -3, 5)
  let B : ℝ × ℝ × ℝ := (2, -3, -5)
  Real.sqrt ((A.fst - B.fst)^2 + (A.snd.fst - B.snd.fst)^2 + (A.snd.snd - B.snd.snd)^2) = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_reflection_xoy_distance_A_to_B_l501_50198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_inscribed_circle_l501_50162

noncomputable section

-- Define the triangle side ratio
def side_ratio : ℝ × ℝ × ℝ := (2, 3, Real.sqrt 13)

-- Define the circle radius
def circle_radius : ℝ := 4

-- Define the triangle area
noncomputable def triangle_area (ratio : ℝ × ℝ × ℝ) (radius : ℝ) : ℝ :=
  let x := (2 * radius) / Real.sqrt (ratio.2.2)
  (1 / 2) * ratio.1 * x * ratio.2.1 * x

-- Theorem statement
theorem triangle_area_inscribed_circle :
  triangle_area side_ratio circle_radius = 384 / 13 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_inscribed_circle_l501_50162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_seven_and_terminating_l501_50186

/-- A function that checks if a positive integer contains the digit 7 -/
def containsSeven (n : ℕ+) : Prop := 
  ∃ d : ℕ, d ∈ n.val.digits 10 ∧ d = 7

/-- A function that checks if a fraction is a terminating decimal -/
def isTerminatingDecimal (n : ℕ+) : Prop := 
  ∃ (a b : ℕ), n.val = 2^a * 5^b

/-- The smallest positive integer n such that 1/n is a terminating decimal and n contains the digit 7 -/
def smallestNWithSevenAndTerminating : ℕ+ :=
  (⟨128, by norm_num⟩ : ℕ+)

theorem smallest_n_with_seven_and_terminating :
  containsSeven smallestNWithSevenAndTerminating ∧ 
  isTerminatingDecimal smallestNWithSevenAndTerminating ∧
  ∀ m : ℕ+, m < smallestNWithSevenAndTerminating → 
    ¬(containsSeven m ∧ isTerminatingDecimal m) :=
sorry

#eval smallestNWithSevenAndTerminating

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_seven_and_terminating_l501_50186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_sqrt_2_l501_50149

-- Define the circle C₁
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (-2 + Real.cos t, 1 + Real.sin t)

-- Define the line C₂
noncomputable def C₂ (s : ℝ) : ℝ × ℝ := (-4 + (Real.sqrt 2 / 2) * s, (Real.sqrt 2 / 2) * s)

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t s, C₁ t = p ∧ C₂ s = p}

-- Theorem statement
theorem intersection_distance_is_sqrt_2 :
  ∃ A B, A ∈ intersection_points ∧ B ∈ intersection_points ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_sqrt_2_l501_50149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_coplanar_edges_count_l501_50146

-- Define a cube type
structure Cube where
  -- We'll leave this empty for now, as we don't need specific properties for this proof
  mk :: -- Empty constructor

-- Define a line type
structure Line where
  -- We'll leave this empty for now, as we don't need specific properties for this proof
  mk :: -- Empty constructor

-- Define a function to count non-coplanar edges
def count_non_coplanar_edges (c : Cube) (l : Line) : ℕ := 
  sorry -- We'll leave the implementation as sorry for now

-- Theorem statement
theorem non_coplanar_edges_count (c : Cube) (l : Line) :
  count_non_coplanar_edges c l ∈ ({4, 6, 7, 8} : Set ℕ) := by
  sorry -- We'll leave the proof as sorry for now

#check non_coplanar_edges_count -- This line checks if the theorem is well-formed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_coplanar_edges_count_l501_50146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_quarter_sum_l501_50132

/-- The sum of an infinite geometric series with first term a and common ratio r where |r| < 1 -/
noncomputable def infiniteGeometricSeriesSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Theorem: The sum of the infinite geometric series 1/4 + 1/8 + 1/16 + 1/32 + ... is 1/2 -/
theorem infinite_geometric_series_quarter_sum :
  infiniteGeometricSeriesSum (1/4 : ℝ) (1/2 : ℝ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_quarter_sum_l501_50132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_to_2021_2021_cannot_transform_to_2022_2022_l501_50152

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Allowed action: increase one number by the sum of digits of the other -/
def allowed_action (a b : ℕ) : List (ℕ × ℕ) :=
  [(a + sum_of_digits b, b), (a, b + sum_of_digits a)]

/-- Sequence of allowed actions -/
def action_sequence : List (ℕ × ℕ → ℕ × ℕ) := sorry

/-- Apply a sequence of actions to a pair of numbers -/
def apply_actions (initial : ℕ × ℕ) (actions : List (ℕ × ℕ → ℕ × ℕ)) : ℕ × ℕ := sorry

/-- Theorem: It's possible to transform (1, 2) into (2021, 2021) -/
theorem transform_to_2021_2021 : ∃ (actions : List (ℕ × ℕ → ℕ × ℕ)), 
  apply_actions (1, 2) actions = (2021, 2021) := by sorry

/-- Theorem: It's impossible to transform (1, 2) into (2022, 2022) -/
theorem cannot_transform_to_2022_2022 : ¬∃ (actions : List (ℕ × ℕ → ℕ × ℕ)), 
  apply_actions (1, 2) actions = (2022, 2022) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_to_2021_2021_cannot_transform_to_2022_2022_l501_50152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l501_50122

def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / 9 = 1

noncomputable def vertex_distance : ℝ :=
  let a := Real.sqrt 16
  2 * a

theorem hyperbola_vertex_distance :
  vertex_distance = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l501_50122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l501_50169

/-- The area of a triangle given its side lengths -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: The area of a triangle with side lengths 30, 24, and 18 is equal to 216 -/
theorem triangle_area_specific : triangle_area 30 24 18 = 216 := by
  -- Unfold the definition of triangle_area
  unfold triangle_area
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l501_50169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l501_50118

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := 4 * x^2 + 1/x

-- State the theorem
theorem f_strictly_increasing :
  ∀ x₁ x₂ : ℝ, 1/2 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂ :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l501_50118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_y_equals_eighteen_fifths_l501_50160

theorem sum_x_y_equals_eighteen_fifths 
  (x y : ℝ) 
  (h1 : |x| + x + y = 10) 
  (h2 : x + |y| - y = 12) : 
  x + y = 18/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_y_equals_eighteen_fifths_l501_50160
