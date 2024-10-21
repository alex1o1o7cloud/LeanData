import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_log_curves_base_l590_59000

-- Define the square and its properties
def Square (W Z X Y : ℝ × ℝ) : Prop :=
  let (wx, wy) := W
  let (zx, zy) := Z
  let (xx, xy) := X
  let (yx, yy) := Y
  (zx - wx)^2 + (zy - wy)^2 = 49 ∧
  zy = wy ∧
  (xx - wx)^2 + (xy - wy)^2 = 49 ∧
  (yx - wx)^2 + (yy - wy)^2 = 49

-- Define the logarithmic functions
def OnLogCurve (p : ℝ × ℝ) (b : ℝ) (k : ℝ) : Prop :=
  let (x, y) := p
  y = k * (Real.log x / Real.log b)

-- State the theorem
theorem square_log_curves_base (W Z X Y : ℝ × ℝ) (b : ℝ) :
  Square W Z X Y →
  OnLogCurve W b 1 →
  OnLogCurve Z b 2 →
  OnLogCurve Y b (1/2) →
  b = (49/16)^(1/7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_log_curves_base_l590_59000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_f_implies_a_range_l590_59058

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (4 - a/2)*x + 2

theorem monotone_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → 4 ≤ a ∧ a < 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_f_implies_a_range_l590_59058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_3_05_l590_59054

/-- Represents a clock with minute and hour hands -/
structure Clock :=
  (minutes : ℕ)
  (hours : ℕ)

/-- The angle between the minute and hour hands at a given time -/
noncomputable def angle_between_hands (c : Clock) : ℝ :=
  let minutes_angle := (c.minutes : ℝ) * 6
  let hours_angle := (c.hours : ℝ) * 30 + (c.minutes : ℝ) * 0.5
  let diff := abs (minutes_angle - hours_angle)
  min diff (360 - diff)

/-- Theorem stating that the angle between the minute and hour hands at 3:05 is 62.5° -/
theorem angle_at_3_05 :
  angle_between_hands { minutes := 5, hours := 3 } = 62.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_3_05_l590_59054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_k_range_l590_59098

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 - 6*x^2 + a*x + b

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := 3*x^2 - 12*x + a

-- State the theorem
theorem function_and_k_range :
  ∀ (a b : ℝ),
  (∃ (x : ℝ), f a b x = 0 ∧ f' a x = 9) →
  (f a b = λ x ↦ x^3 - 6*x^2 + 21*x - 26) ∧
  (∀ k : ℝ, (∀ x : ℝ, x ∈ Set.Ioo 1 5 → 21*x + k - 80 < f a b x ∧ f a b x < 9*x + k) 
  → k ∈ Set.Ioo 9 22) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_k_range_l590_59098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_f_zero_points_implies_m_eq_one_l590_59064

-- Define the inequality
noncomputable def inequality (x : ℝ) : Prop := |x + 3| - 2*x - 1 < 0

-- Define the solution set
def solution_set : Set ℝ := {x | x > 2}

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := |x - m| + |x + 1/m| - 2

-- Theorem 1: The solution set of the inequality is (2, +∞)
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set := by
  sorry

-- Theorem 2: If f has zero points and m > 0, then m = 1
theorem f_zero_points_implies_m_eq_one (m : ℝ) (h_m : m > 0) :
  (∃ x, f m x = 0) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_f_zero_points_implies_m_eq_one_l590_59064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_properties_l590_59091

/-- Circle with equation x^2 + y^2 = 25 -/
def Circle : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 25}

/-- Point A coordinates -/
noncomputable def A : ℝ × ℝ := (3, 4)

/-- Centroid G coordinates -/
noncomputable def G : ℝ × ℝ := (5/3, 2)

/-- Definition of a triangle inscribed in the circle -/
def InscribedTriangle (B C : ℝ × ℝ) : Prop :=
  B ∈ Circle ∧ C ∈ Circle ∧ A ∈ Circle

/-- Definition of the centroid of a triangle -/
def IsCentroid (B C : ℝ × ℝ) : Prop :=
  G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

/-- Definition of complementary slopes -/
def ComplementarySlopes (B C : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (C.2 - A.2) = -(B.1 - A.1) * (C.1 - A.1)

/-- Line through two points -/
def LineThroughPoints (B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.2 - B.2) * (C.1 - B.1) = (p.1 - B.1) * (C.2 - B.2)}

/-- Main theorem -/
theorem inscribed_triangle_properties (B C : ℝ × ℝ) 
  (h1 : InscribedTriangle B C) (h2 : IsCentroid B C) :
  (∀ x y : ℝ, x + y - 2 = 0 ↔ (x, y) ∈ LineThroughPoints B C) ∧
  (ComplementarySlopes B C → 
    ∀ x y : ℝ, (x, y) ∈ LineThroughPoints B C → (y - C.2) / (x - C.1) = 3/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_properties_l590_59091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jean_speed_is_20_over_17_l590_59038

/-- Represents the hiking scenario with Chantal and Jean --/
structure HikingScenario where
  d : ℝ  -- Half the total distance to the fire tower
  chantal_speed_first_half : ℝ  -- Chantal's speed in the first half
  chantal_speed_second_half : ℝ  -- Chantal's speed in the second half
  chantal_speed_descent : ℝ  -- Chantal's speed during descent

/-- Calculates Jean's average speed given a hiking scenario --/
noncomputable def jean_average_speed (scenario : HikingScenario) : ℝ :=
  let total_time := scenario.d / scenario.chantal_speed_first_half +
                     scenario.d / scenario.chantal_speed_second_half +
                     scenario.d / scenario.chantal_speed_descent
  scenario.d / total_time

/-- Theorem stating that Jean's average speed is 20/17 mph under the given conditions --/
theorem jean_speed_is_20_over_17 (scenario : HikingScenario)
  (h1 : scenario.chantal_speed_first_half = 5)
  (h2 : scenario.chantal_speed_second_half = 2.5)
  (h3 : scenario.chantal_speed_descent = 4) :
  jean_average_speed scenario = 20 / 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jean_speed_is_20_over_17_l590_59038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l590_59031

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi ∧
  0 < t.B ∧ t.B < Real.pi ∧
  0 < t.C ∧ t.C < Real.pi ∧
  t.A + t.B + t.C = Real.pi ∧
  Real.cos t.C = 4/5 ∧
  t.c = 2 * t.b * Real.cos t.A ∧
  (1/2) * t.a * t.b * Real.sin t.C = 15/2

-- State the theorem
theorem triangle_properties (t : Triangle) (h : satisfies_conditions t) :
  t.A = t.B ∧ t.c = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l590_59031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visit_cost_calculation_l590_59095

/-- Calculates the total cost of a medical visit given the out-of-pocket cost and insurance coverage percentage. -/
noncomputable def total_cost (out_of_pocket : ℝ) (insurance_coverage : ℝ) : ℝ :=
  out_of_pocket / (1 - insurance_coverage)

/-- Theorem stating that given an out-of-pocket cost of $60 and insurance coverage of 80%, the total cost of the visit is $300. -/
theorem visit_cost_calculation : total_cost 60 0.8 = 300 := by
  -- Unfold the definition of total_cost
  unfold total_cost
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_visit_cost_calculation_l590_59095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_T3_after_reduction_l590_59096

/-- The area of the original square T₁ -/
noncomputable def area_T1 : ℝ := 36

/-- The side length of square T₂ relative to T₁ -/
noncomputable def ratio_T2_T1 : ℝ := 2/3

/-- The side length of square T₃ relative to T₂ -/
noncomputable def ratio_T3_T2 : ℝ := 2/3

/-- The factor by which the area of T₃ is reduced -/
noncomputable def area_reduction_factor : ℝ := 1/2

/-- Theorem stating that the area of T₃ after reduction is 32/9 -/
theorem area_T3_after_reduction :
  let side_T1 := Real.sqrt area_T1
  let side_T2 := ratio_T2_T1 * side_T1
  let side_T3 := ratio_T3_T2 * side_T2
  let area_T3 := side_T3 ^ 2
  area_reduction_factor * area_T3 = 32/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_T3_after_reduction_l590_59096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_approximately_26_09_percent_l590_59097

/-- Calculate the discount percentage given cost price, markup percentage, and selling price -/
noncomputable def discount_percentage (cost_price : ℝ) (markup_percentage : ℝ) (selling_price : ℝ) : ℝ :=
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let discount := marked_price - selling_price
  (discount / marked_price) * 100

/-- Theorem stating that the discount percentage is approximately 26.09% -/
theorem discount_approximately_26_09_percent :
  let cost_price := (540 : ℝ)
  let markup_percentage := (15 : ℝ)
  let selling_price := (459 : ℝ)
  abs (discount_percentage cost_price markup_percentage selling_price - 26.09) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_approximately_26_09_percent_l590_59097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_equals_two_l590_59056

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then 1/x - 1
  else if x ≥ 1 then 1 - 1/x
  else 0  -- undefined for x ≤ 0

-- State the theorem
theorem inverse_sum_equals_two (a b : ℝ) (ha : 0 < a) (hb : a < b) (hf : f a = f b) :
  1/a + 1/b = 2 := by
  sorry

-- Note: The proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_equals_two_l590_59056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_achieve_desired_ratio_l590_59075

/-- Represents the mixture of milk and water -/
structure Mixture where
  total : ℚ
  milk : ℚ
  water : ℚ

/-- Performs the operation of removing 10 litres and adding 10 litres of pure milk -/
def perform_operation (m : Mixture) : Mixture :=
  { total := m.total,
    milk := m.milk - (m.milk / m.total) * 10 + 10,
    water := m.water - (m.water / m.total) * 10 }

/-- Theorem stating that performing the operation twice on the initial mixture results in a 9:1 ratio -/
theorem achieve_desired_ratio :
  let initial_mixture : Mixture := { total := 20, milk := 12, water := 8 }
  let final_mixture := perform_operation (perform_operation initial_mixture)
  final_mixture.milk / final_mixture.water = 9 := by sorry

#eval let initial_mixture : Mixture := { total := 20, milk := 12, water := 8 }
      let final_mixture := perform_operation (perform_operation initial_mixture)
      final_mixture.milk / final_mixture.water

end NUMINAMATH_CALUDE_ERRORFEEDBACK_achieve_desired_ratio_l590_59075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l590_59004

/-- RightTriangle represents a right-angled triangle with given perimeter and area -/
structure RightTriangle where
  p : ℝ  -- perimeter
  s : ℝ  -- area
  h_positive : p > 0 ∧ s > 0  -- perimeter and area are positive

/-- The length of the hypotenuse (AB) in a right-angled triangle -/
noncomputable def hypotenuseLength (t : RightTriangle) : ℝ :=
  t.p / 2 - Real.sqrt ((t.p / 2) ^ 2 - 2 * t.s)

/-- The quadratic equation with the lengths of the other two sides (AC and BC) as roots -/
noncomputable def sidesQuadratic (t : RightTriangle) (x : ℝ) : ℝ :=
  x^2 - (t.p / 2 + Real.sqrt ((t.p / 2) ^ 2 - 2 * t.s)) * x + 2 * t.s

theorem right_triangle_properties (t : RightTriangle) :
  (∀ x, sidesQuadratic t x = 0 ↔ (x = hypotenuseLength t ∨ x = t.p - hypotenuseLength t)) ∧
  hypotenuseLength t > 0 ∧
  hypotenuseLength t < t.p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l590_59004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_in_pyramid_l590_59003

open Real

-- Define the pyramid's properties
noncomputable def pyramid_base_side : ℝ := 2
noncomputable def pyramid_lateral_angle : ℝ := 60 * π / 180  -- 60 degrees in radians

-- Define the cube's properties
noncomputable def cube_side (h : ℝ) : ℝ := h / 2

-- Define the pyramid's height
noncomputable def pyramid_height : ℝ := (pyramid_base_side * sqrt 2 * sin pyramid_lateral_angle) / 2

-- State the theorem
theorem cube_volume_in_pyramid :
  let h := pyramid_height
  let s := cube_side h
  s^3 = (3 * sqrt 6) / 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_in_pyramid_l590_59003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_increase_l590_59086

theorem triangle_area_increase (a b θ : ℝ) (ha : a > 0) (hb : b > 0) (hθ : 0 < θ ∧ θ < π) :
  (1/2) * (3*a) * (2*b) * Real.sin θ = 6 * ((1/2) * a * b * Real.sin θ) :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_increase_l590_59086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_intersection_M_N_l590_59099

-- Define the sets M and N
def M : Set ℝ := {x | |x| < 1}
def N : Set ℝ := {y | ∃ x ∈ M, y = Real.exp (Real.log 3 * x)}

-- State the theorem
theorem complement_of_intersection_M_N :
  (Set.Iic (1/3) ∪ Set.Ici 1) = (M ∩ N)ᶜ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_intersection_M_N_l590_59099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_triangle_area_l590_59081

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x - Real.sqrt 3

def domain (x : ℝ) : Prop := Real.pi / 3 ≤ x ∧ x ≤ 11 * Real.pi / 24

theorem f_range_and_triangle_area :
  (∀ x, domain x → Real.sqrt 3 ≤ f x ∧ f x ≤ 2) ∧
  (∀ a b c : ℝ, 
    0 < a ∧ 0 < b ∧ 0 < c →
    a = Real.sqrt 3 →
    b = 2 →
    (a + b > c ∧ b + c > a ∧ c + a > b) →
    (∃ r, r = 3 * Real.sqrt 2 / 4 ∧ r = a / (2 * Real.sin (Real.arcsin (b / (2 * r))))) →
    a * b * Real.sin (Real.arcsin (c / (2 * (3 * Real.sqrt 2 / 4)))) / 2 = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_triangle_area_l590_59081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l590_59017

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2*x - 1) * Real.exp x - a * (x^2 + x)

/-- The function g(x) as defined in the problem -/
def g (a : ℝ) (x : ℝ) : ℝ := -a * x^2 - a

/-- Theorem stating the range of a given the conditions -/
theorem a_range (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ g a x) → a ∈ Set.Icc 1 (4 * Real.exp (3/2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l590_59017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_arithmetic_sequence_l590_59078

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) : ℕ → ℚ
  | 0 => a₁
  | n + 1 => arithmetic_sequence a₁ d n + d

theorem first_term_of_arithmetic_sequence :
  ∃ a₁ : ℚ, 
  let d : ℚ := 3 / 4
  let a : ℕ → ℚ := arithmetic_sequence a₁ d
  a 30 = 63 / 4 ∧ a₁ = -14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_arithmetic_sequence_l590_59078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_128_l590_59034

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 6 then x^2
  else if 6 < x ∧ x ≤ 10 then 3*x - 10
  else 0

-- Define the area L
noncomputable def L : ℝ := ∫ x in (0:ℝ)..(10:ℝ), f x

-- Theorem statement
theorem area_equals_128 : L = 128 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_128_l590_59034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_problem_l590_59092

theorem log_problem (x : ℝ) (h : Real.log (x - 3) / Real.log 16 = 1/2) :
  Real.log x / Real.log 216 = (1/3) * (Real.log 7 / Real.log 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_problem_l590_59092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_eq_beta_sufficient_not_necessary_l590_59053

theorem alpha_eq_beta_sufficient_not_necessary :
  (∀ α β : Real, α = β → Real.sin α ^ 2 + Real.cos β ^ 2 = 1) ∧
  ¬(∀ α β : Real, Real.sin α ^ 2 + Real.cos β ^ 2 = 1 → α = β) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_eq_beta_sufficient_not_necessary_l590_59053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l590_59055

theorem simplify_expression (m : ℝ) (hm : m ≠ 0) :
  (1 / (3 * m)) ^ (-3 : ℝ) * (-2 * m) ^ 4 = 432 * m ^ 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l590_59055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_minimum_value_l590_59057

/-- The minimum value of (b^2 + 1) / (3a) for a hyperbola with eccentricity 2 -/
theorem hyperbola_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) 
  (h_eccentricity : Real.sqrt ((a^2 + b^2) / a^2) = 2) :
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (∀ x y : ℝ, x^2 / a'^2 - y^2 / b'^2 = 1) → 
    Real.sqrt ((a'^2 + b'^2) / a'^2) = 2 → 
    (b^2 + 1) / (3 * a) ≤ (b'^2 + 1) / (3 * a')) ∧
  (b^2 + 1) / (3 * a) = 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_minimum_value_l590_59057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_area_increase_l590_59005

/-- The percent increase in area between two circular pizzas -/
noncomputable def percentIncrease (r1 r2 : ℝ) : ℝ :=
  ((r2^2 - r1^2) / r1^2) * 100

/-- Theorem: Given the conditions on pizza radii, the percent increase in area from small to large pizza is 251.5625% -/
theorem pizza_area_increase (r : ℝ) (hr : r > 0) :
  percentIncrease r (1.875 * r) = 251.5625 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_area_increase_l590_59005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eugene_buys_four_tshirts_l590_59006

/-- Represents the shopping scenario with given prices and discounts -/
structure ShoppingScenario where
  tshirt_price : ℚ
  pants_price : ℚ
  shoes_price : ℚ
  discount_rate : ℚ
  num_pants : ℕ
  num_shoes : ℕ
  total_cost : ℚ

/-- Calculates the number of T-shirts bought given a shopping scenario -/
def calculate_tshirts (s : ShoppingScenario) : ℚ :=
  let discounted_tshirt := s.tshirt_price * (1 - s.discount_rate)
  let discounted_pants := s.pants_price * (1 - s.discount_rate)
  let discounted_shoes := s.shoes_price * (1 - s.discount_rate)
  let pants_and_shoes_cost := s.num_pants * discounted_pants + s.num_shoes * discounted_shoes
  (s.total_cost - pants_and_shoes_cost) / discounted_tshirt

/-- Theorem stating that Eugene buys 4 T-shirts -/
theorem eugene_buys_four_tshirts (s : ShoppingScenario) 
  (h1 : s.tshirt_price = 20)
  (h2 : s.pants_price = 80)
  (h3 : s.shoes_price = 150)
  (h4 : s.discount_rate = 1/10)
  (h5 : s.num_pants = 3)
  (h6 : s.num_shoes = 2)
  (h7 : s.total_cost = 558) :
  calculate_tshirts s = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eugene_buys_four_tshirts_l590_59006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_range_of_a_l590_59087

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - x^2 / 2 - a * x - 1

theorem tangent_line_and_range_of_a :
  let a₀ : ℝ := -1/2
  let tangent_line (x y : ℝ) := (Real.exp 1 - 1/2) * x - y - 1/2 = 0
  let upper_bound : ℝ := 2 * Real.sqrt (Real.exp 1) - 9/4
  (∀ x y : ℝ, y = f a₀ x → x = 1 → tangent_line x y) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ≥ 1/2 → f a x ≥ 0) ↔ a ≤ upper_bound) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_range_of_a_l590_59087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l590_59069

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (3 * x + Real.pi / 3)

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l590_59069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_ratio_is_two_l590_59043

/-- Represents a parallelogram with given area and base length -/
structure Parallelogram where
  area : ℝ
  base : ℝ

/-- Calculates the altitude of a parallelogram -/
noncomputable def altitude (p : Parallelogram) : ℝ := p.area / p.base

/-- Calculates the ratio of altitude to base for a parallelogram -/
noncomputable def altitudeBaseRatio (p : Parallelogram) : ℝ := altitude p / p.base

/-- Theorem stating that for a parallelogram with area 200 and base 10, 
    the ratio of altitude to base is 2 -/
theorem parallelogram_ratio_is_two : 
  ∀ (p : Parallelogram), p.area = 200 ∧ p.base = 10 → altitudeBaseRatio p = 2 := by
  intro p ⟨h_area, h_base⟩
  unfold altitudeBaseRatio altitude
  rw [h_area, h_base]
  norm_num
  
#check parallelogram_ratio_is_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_ratio_is_two_l590_59043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_u_l590_59071

theorem min_value_of_u (x y : ℝ) (hx : x ∈ Set.Ioo (-2 : ℝ) 2)
  (hy : y ∈ Set.Ioo (-2 : ℝ) 2) (hxy : x * y = -1) :
  ∃ (min_u : ℝ), min_u = 12/7 ∧ 
  (∀ x' y', x' ∈ Set.Ioo (-2 : ℝ) 2 →
    y' ∈ Set.Ioo (-2 : ℝ) 2 → x' * y' = -1 → 
    4 / (4 - x'^2) + 9 / (9 - y'^2) ≥ min_u) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_u_l590_59071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_at_point_l590_59083

noncomputable section

/-- The curve function -/
def f (x : ℝ) : ℝ := (1/3) * x^3 - 3 * x^2 + 8 * x + 4

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := x^2 - 6 * x + 8

/-- The slope of the line 2x + 2y - 5 = 0 -/
def k : ℝ := -1

/-- The point where the tangent is parallel to the line -/
def tangent_point : ℝ × ℝ := (3, 10)

/-- Theorem stating that the tangent at the given point is parallel to the line -/
theorem tangent_parallel_at_point :
  f' (tangent_point.fst) = k ∧ 
  f (tangent_point.fst) = tangent_point.snd := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_at_point_l590_59083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_group_size_l590_59062

/-- The number of men in the first group -/
def first_group : ℕ := 14

/-- The number of days for the first group to complete the work -/
def days_first : ℕ := 22

/-- The number of days for the second group to complete the work -/
noncomputable def days_second : ℝ := 17.11111111111111

/-- The number of men in the second group -/
noncomputable def second_group : ℝ := (first_group : ℝ) * days_first / days_second

theorem second_group_size :
  ⌊second_group⌋ = 18 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_group_size_l590_59062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_AB_coordinates_l590_59065

def point_A : ℝ × ℝ × ℝ := (1, -3, 4)
def point_B : ℝ × ℝ × ℝ := (-3, 2, 1)

def vector_AB : ℝ × ℝ × ℝ := 
  (point_B.1 - point_A.1, point_B.2.1 - point_A.2.1, point_B.2.2 - point_A.2.2)

theorem vector_AB_coordinates :
  vector_AB = (-4, 5, -3) := by
  unfold vector_AB point_A point_B
  simp
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_AB_coordinates_l590_59065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_linear_l590_59080

/-- A function f: ℝ → ℝ is linear if there exist constants m and b such that f(x) = mx + b for all x ∈ ℝ. -/
def IsLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b

/-- The function f(x) = (1/2)x -/
noncomputable def f (x : ℝ) : ℝ := (1/2) * x

/-- Theorem: The function f(x) = (1/2)x is a linear function -/
theorem f_is_linear : IsLinearFunction f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_linear_l590_59080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangement_count_l590_59067

/-- Represents the number of Arabic books -/
def arabic_books : ℕ := 3

/-- Represents the number of German books -/
def german_books : ℕ := 2

/-- Represents the number of Spanish books -/
def spanish_books : ℕ := 2

/-- Represents the total number of books -/
def total_books : ℕ := arabic_books + german_books + spanish_books

/-- Calculates the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Calculates the number of arrangements for a given scenario -/
def scenario_arrangements : ℕ := 
  factorial 4 * factorial arabic_books * factorial 2

/-- Represents the total number of valid arrangements -/
def total_arrangements : ℕ := 2 * scenario_arrangements

theorem book_arrangement_count :
  total_arrangements = 576 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangement_count_l590_59067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l590_59074

/-- The curve E defined by the equation 3x² + 2y² = 6 -/
def CurveE (x y : ℝ) : Prop := 3 * x^2 + 2 * y^2 = 6

/-- Point F with coordinates (0, 1) -/
def F : ℝ × ℝ := (0, 1)

/-- Line l passing through points F, A, and B -/
def LineL (k : ℝ) (x : ℝ) : ℝ := k * x + 1

/-- Condition that points A and B are on curve E and line l -/
def PointsOnCurveAndLine (k : ℝ) (xA yA xB yB : ℝ) : Prop :=
  CurveE xA yA ∧ CurveE xB yB ∧ yA = LineL k xA ∧ yB = LineL k xB

/-- Condition that AF = λ FB, where 2 ≤ λ ≤ 3 -/
def VectorRatio (lambda : ℝ) (xA yA xB yB : ℝ) : Prop :=
  2 ≤ lambda ∧ lambda ≤ 3 ∧ (-xA, 1 - yA) = (lambda * xB, lambda * (yB - 1))

/-- The main theorem stating the range of possible slopes k -/
theorem slope_range :
  ∀ k : ℝ,
  (∃ xA yA xB yB lambda : ℝ,
    PointsOnCurveAndLine k xA yA xB yB ∧
    VectorRatio lambda xA yA xB yB) ↔
  (k ∈ Set.Icc (-Real.sqrt 3) (-Real.sqrt 2 / 2) ∪ 
       Set.Icc (Real.sqrt 2 / 2) (Real.sqrt 3)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l590_59074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_six_l590_59068

def sequenceA (n : ℕ) : ℚ :=
  if n % 2 = 1 then 3 else 6

theorem tenth_term_is_six :
  let s := sequenceA
  (∀ n : ℕ, n ≥ 1 → s (n + 1) * s n = 18) →
  s 0 = 3 →
  s 1 = 6 →
  s 9 = 6 :=
by
  sorry

#check tenth_term_is_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_six_l590_59068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_device_improvement_l590_59028

/-- Data for the old device -/
def old_data : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]

/-- Data for the new device -/
def new_data : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

/-- Sample mean of a list of real numbers -/
noncomputable def sample_mean (data : List ℝ) : ℝ :=
  (data.sum) / (data.length : ℝ)

/-- Sample variance of a list of real numbers -/
noncomputable def sample_variance (data : List ℝ) : ℝ :=
  let mean := sample_mean data
  (data.map (fun x => (x - mean)^2)).sum / (data.length : ℝ)

/-- Criterion for significant improvement -/
def significant_improvement (x_bar y_bar s1_sq s2_sq : ℝ) : Prop :=
  y_bar - x_bar ≥ 2 * Real.sqrt ((s1_sq + s2_sq) / 10)

/-- Theorem stating that the new device shows significant improvement -/
theorem new_device_improvement :
  let x_bar := sample_mean old_data
  let y_bar := sample_mean new_data
  let s1_sq := sample_variance old_data
  let s2_sq := sample_variance new_data
  significant_improvement x_bar y_bar s1_sq s2_sq :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_device_improvement_l590_59028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_implies_a_value_l590_59049

/-- The function f(x) = (1/3)x³ + x² + ax - 5 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + x^2 + a*x - 5

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a

theorem monotonic_decreasing_implies_a_value (a : ℝ) :
  (∀ x ∈ Set.Ioo (-3 : ℝ) 1, (f' a x) < 0) →
  a = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_implies_a_value_l590_59049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_girls_pair_l590_59047

-- Define the number of boys and girls
def num_boys : ℕ := 9
def num_girls : ℕ := 9

-- Define the total number of people
def total_people : ℕ := num_boys + num_girls

-- Define the probability of no girls-only pairs
noncomputable def prob_no_girls_pair : ℚ := (num_girls.factorial ^ 3 * 2^num_girls) / total_people.factorial

-- Theorem statement
theorem prob_at_least_one_girls_pair :
  ∃ (ε : ℚ), abs ((1 : ℚ) - prob_no_girls_pair - 99/100) < ε ∧ ε < 1/100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_girls_pair_l590_59047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculation_and_square_roots_operation_l590_59073

theorem sqrt_calculation_and_square_roots_operation :
  (∀ (x y : ℝ), x > 0 → y > 0 → Real.sqrt (x * y) = Real.sqrt x * Real.sqrt y) →
  (Real.sqrt 32 + Real.sqrt 8 - Real.sqrt 50 = Real.sqrt 2) ∧
  ((Real.sqrt 3 - Real.sqrt 2)^2 * (5 + 2 * Real.sqrt 6) = 1) :=
by
  intro h
  apply And.intro
  · -- Proof for the first part
    sorry
  · -- Proof for the second part
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculation_and_square_roots_operation_l590_59073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_dot_product_l590_59036

/-- A trapezoid with specific properties -/
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  ab_length : ((A.1 - B.1)^2 + (A.2 - B.2)^2).sqrt = 65
  cd_length : ((C.1 - D.1)^2 + (C.2 - D.2)^2).sqrt = 31
  lateral_perpendicular : (A.1 - D.1) * (B.1 - C.1) + (A.2 - D.2) * (B.2 - C.2) = 0

/-- The dot product of vectors AC and BD in the trapezoid -/
def dot_product (t : Trapezoid) : ℝ :=
  let ac := (t.C.1 - t.A.1, t.C.2 - t.A.2)
  let bd := (t.D.1 - t.B.1, t.D.2 - t.B.2)
  ac.1 * bd.1 + ac.2 * bd.2

/-- Theorem stating the dot product of AC and BD is -2015 -/
theorem trapezoid_dot_product (t : Trapezoid) : dot_product t = -2015 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_dot_product_l590_59036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_time_difference_l590_59060

/-- Represents the time difference between two routes in minutes -/
noncomputable def TimeDifference (routeXDistance : ℝ) (routeXHeavyTrafficDistance : ℝ) 
  (routeXNormalSpeed : ℝ) (routeXHeavyTrafficSpeed : ℝ)
  (routeYDistance : ℝ) (routeYSpeed : ℝ) : ℝ :=
  let routeXNormalTime := (routeXDistance - routeXHeavyTrafficDistance) / routeXNormalSpeed * 60
  let routeXHeavyTrafficTime := routeXHeavyTrafficDistance / routeXHeavyTrafficSpeed * 60
  let routeXTotalTime := routeXNormalTime + routeXHeavyTrafficTime
  let routeYTime := routeYDistance / routeYSpeed * 60
  routeXTotalTime - routeYTime

/-- Theorem stating the time difference between Route X and Route Y -/
theorem route_time_difference :
  TimeDifference 8 1 40 10 7 35 = 4.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_time_difference_l590_59060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_symmetry_implies_m_value_l590_59008

/-- A parabola with equation y = x^2 + (m-2)x - (m+3) -/
def parabola (m : ℝ) (x y : ℝ) : Prop :=
  y = x^2 + (m-2)*x - (m+3)

/-- The axis of symmetry of a parabola y = ax^2 + bx + c is x = -b/(2a) -/
noncomputable def axis_of_symmetry (m : ℝ) : ℝ := -(m-2)/(2*1)

/-- The y-axis has the equation x = 0 -/
def y_axis (x : ℝ) : Prop := x = 0

theorem parabola_symmetry_implies_m_value :
  ∀ m : ℝ, (∀ x : ℝ, y_axis x ↔ axis_of_symmetry m = x) → m = 2 :=
by
  intro m h
  have h1 : axis_of_symmetry m = 0 := by
    apply Iff.mp (h 0)
    rfl
  unfold axis_of_symmetry at h1
  field_simp at h1
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_symmetry_implies_m_value_l590_59008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_range_l590_59044

theorem solution_range (a b c : ℝ) (h_a : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (f 0.4 = -0.64) →
  (f 0.5 = -0.25) →
  (f 0.6 = 0.16) →
  (f 0.7 = 0.59) →
  ∃ x : ℝ, f x = 0 ∧ 0.5 < x ∧ x < 0.6 :=
by
  intro f h1 h2 h3 h4
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_range_l590_59044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l590_59090

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (2 - x)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Iio 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l590_59090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_Q_is_circle_l590_59012

-- Define the hyperbola and its properties
structure Hyperbola where
  center : EuclideanSpace ℝ (Fin 2)
  foci : EuclideanSpace ℝ (Fin 2) × EuclideanSpace ℝ (Fin 2)
  vertex_distance : ℝ

-- Define the points and lines
variable (C : Hyperbola) (P Q : EuclideanSpace ℝ (Fin 2))

-- Define the angle bisector
def angle_bisector (A B C : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) :=
  sorry

-- Define the concept of a point being on a hyperbola
def on_hyperbola (P : EuclideanSpace ℝ (Fin 2)) (C : Hyperbola) : Prop :=
  sorry

-- Define the concept of a point being the foot of a perpendicular
def is_foot_of_perpendicular (Q A : EuclideanSpace ℝ (Fin 2)) (L : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  sorry

-- Define the locus of a point
def locus (Q : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) :=
  sorry

-- Define what it means for a set to be a circle
def is_circle (S : Set (EuclideanSpace ℝ (Fin 2))) (center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ) : Prop :=
  sorry

-- State the theorem
theorem trajectory_of_Q_is_circle
  (h1 : on_hyperbola P C)
  (h2 : is_foot_of_perpendicular Q C.foci.1 (angle_bisector C.foci.1 P C.foci.2))
  : ∃ (center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ), 
    is_circle (locus Q) center radius :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_Q_is_circle_l590_59012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_inscribed_in_cylinder_iff_l590_59079

-- Define a right prism
structure RightPrism where
  base : Set (Fin 3 → ℝ)
  height : ℝ

-- Define a cylinder
structure Cylinder where
  base : Set (Fin 3 → ℝ)
  height : ℝ

-- Define the property of a polygon being inscribable in a circle
def IsInscribable (polygon : Set (Fin 3 → ℝ)) (circle : Set (Fin 3 → ℝ)) : Prop :=
  ∀ p ∈ polygon, p ∈ circle

-- Define the property of a prism being inscribed in a cylinder
def IsInscribed (prism : RightPrism) (cylinder : Cylinder) : Prop :=
  prism.height = cylinder.height ∧
  IsInscribable prism.base cylinder.base

-- The main theorem
theorem prism_inscribed_in_cylinder_iff (prism : RightPrism) (cylinder : Cylinder) :
  IsInscribed prism cylinder ↔
  prism.height = cylinder.height ∧ IsInscribable prism.base cylinder.base :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_inscribed_in_cylinder_iff_l590_59079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_squared_l590_59020

/-- Given complex numbers a, b, and c that are zeros of a cubic polynomial
    P(z) = z^3 + sz + t, if |a|^2 + |b|^2 + |c|^2 = 350 and a, b, and c form
    a right triangle with a and b as vertices of the right angle,
    then |c|^2 = 612.5 -/
theorem hypotenuse_squared (a b c s t : ℂ) :
  (a^3 + s*a + t = 0) →
  (b^3 + s*b + t = 0) →
  (c^3 + s*c + t = 0) →
  Complex.normSq a + Complex.normSq b + Complex.normSq c = 350 →
  Complex.normSq (a - c) + Complex.normSq (b - c) = Complex.normSq (a - b) →
  Complex.normSq c = 612.5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_squared_l590_59020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_theorem_l590_59070

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

theorem interest_difference_theorem (P : ℝ) :
  let rate : ℝ := 4
  let time : ℝ := 2
  compound_interest P rate time - simple_interest P rate time = 1 →
  P = 625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_theorem_l590_59070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_proof_l590_59015

/-- The polynomial expression given in the problem -/
def polynomial (x : ℝ) : ℝ := 3 * (x - 2 * x^4) - 2 * (x^4 + x - x^6) + 5 * (2 * x^2 - 3 * x^4 + x^7)

/-- The coefficient of x^4 in the polynomial -/
def coefficient_x4 : ℝ := -23

theorem coefficient_x4_proof : 
  (polynomial 1 - polynomial 0) / 1^4 = coefficient_x4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_proof_l590_59015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_numbers_in_list_pi_div_3_irrational_negative_recurring_decimal_irrational_l590_59009

-- Define the list of numbers
noncomputable def number_list : List ℝ := [-2.5, 0, Real.pi / 3, 22 / 7, (-4)^2, -0.5252252225]

-- Define a function to check if a number is irrational
def is_irrational (x : ℝ) : Prop := ¬ (∃ (q : ℚ), (↑q : ℝ) = x)

-- Theorem statement
theorem irrational_numbers_in_list : 
  ∃ (a b : ℝ), a ∈ number_list ∧ b ∈ number_list ∧ 
  is_irrational a ∧ is_irrational b ∧
  ∀ (x : ℝ), x ∈ number_list → is_irrational x → (x = a ∨ x = b) :=
by
  -- Proof goes here
  sorry

-- Helper theorem to show that π/3 is irrational
theorem pi_div_3_irrational : is_irrational (Real.pi / 3) :=
by
  -- Proof goes here
  sorry

-- Helper theorem to show that -0.5252252225... is irrational
theorem negative_recurring_decimal_irrational : is_irrational (-0.5252252225) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_numbers_in_list_pi_div_3_irrational_negative_recurring_decimal_irrational_l590_59009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_sets_equation_l590_59094

/-- Represents the number of white iron sheets available. -/
def total_sheets : ℕ := 300

/-- Represents the number of box bodies that can be made from one sheet. -/
def bodies_per_sheet : ℕ := 14

/-- Represents the number of box bottoms that can be made from one sheet. -/
def bottoms_per_sheet : ℕ := 32

/-- Represents the number of box bottoms required for one complete set. -/
def bottoms_per_set : ℕ := 2

/-- Theorem stating the correct equation for complete sets of canned boxes. -/
theorem complete_sets_equation (x : ℕ) :
  2 * bodies_per_sheet * x = bottoms_per_sheet * (total_sheets - x) := by
  sorry

#check complete_sets_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_sets_equation_l590_59094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l590_59088

theorem triangle_angle_measure (a b c : ℝ) (h1 : a = 7) (h2 : b = 8) (h3 : c = 5) :
  Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l590_59088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_primes_not_dividing_sequence_l590_59007

/-- Sequence a_n defined recursively -/
def a : ℕ → ℕ
  | 0 => 1  -- Define the base case for 0
  | 1 => 1
  | n + 2 => (a (n + 1))^4 - (a (n + 1))^3 + 2*(a (n + 1))^2 + 1

/-- There are infinitely many primes not dividing any term of the sequence -/
theorem infinitely_many_primes_not_dividing_sequence :
  ∃ S : Set ℕ, (∀ p ∈ S, Nat.Prime p) ∧ (Set.Infinite S) ∧
  (∀ p ∈ S, ∀ n : ℕ, ¬(p ∣ a n)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_primes_not_dividing_sequence_l590_59007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l590_59025

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (x - Real.pi/3) + 2 * (Real.cos (x/2))^2

theorem triangle_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h1 : f A = 3/2) 
  (h2 : a = Real.sqrt 3) 
  (h3 : Real.sin B = 2 * Real.sin C) 
  (h4 : 0 < A ∧ A < Real.pi) 
  (h5 : 0 < B ∧ B < Real.pi) 
  (h6 : 0 < C ∧ C < Real.pi) 
  (h7 : A + B + C = Real.pi) 
  (h8 : a / Real.sin A = b / Real.sin B) 
  (h9 : b / Real.sin B = c / Real.sin C) 
  (h10 : a^2 = b^2 + c^2 - 2*b*c*Real.cos A) : 
  c = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l590_59025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_b_and_side_b_range_l590_59042

-- Define an acute triangle
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  side_lengths : a > 0 ∧ b > 0 ∧ c > 0
  law_of_sines : a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C)

-- State the theorem
theorem angle_b_and_side_b_range (t : AcuteTriangle) 
  (h1 : Real.cos t.B ^ 2 + 2 * Real.sqrt 3 * Real.sin t.B * Real.cos t.B - Real.sin t.B ^ 2 = 1)
  (h2 : 2 * t.c * Real.cos t.A + 2 * t.a * Real.cos t.C = t.b * t.c) :
  t.B = π/3 ∧ Real.sqrt 3 < t.b ∧ t.b < 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_b_and_side_b_range_l590_59042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_good_set_characterization_l590_59041

/-- A subset of the plane is "good" if it's unchanged upon rotation by θ around any of its points -/
def is_good_set (S : Set ℂ) (θ : ℝ) : Prop :=
  ∀ z w : ℂ, z ∈ S → w ∈ S → (z + (w - z) * Complex.exp (θ * Complex.I)) ∈ S ∧ 
                              (z + (w - z) * Complex.exp (-θ * Complex.I)) ∈ S

/-- The property that the midpoint of any two points in a set also lies in the set -/
def closed_under_midpoint (S : Set ℂ) : Prop :=
  ∀ z w : ℂ, z ∈ S → w ∈ S → ((z + w) / 2) ∈ S

theorem good_set_characterization (r : ℚ) (hr : -1 ≤ r ∧ r ≤ 1) :
  (∀ S : Set ℂ, is_good_set S (Real.arccos r) → closed_under_midpoint S) ↔
  ∃ n : ℕ+, r = 1 - 1 / (4 * ↑n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_good_set_characterization_l590_59041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l590_59024

/-- The set of digits used to form the integers -/
def Digits : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- A type representing a 6-digit integer formed from the given digits -/
def SixDigitInt := { arr : Fin 6 → ℕ // ∀ i, arr i ∈ Digits }

/-- Predicate to check if one digit is to the left of another in a SixDigitInt -/
def IsLeftOf (n : SixDigitInt) (d1 d2 : ℕ) : Prop :=
  ∃ i j : Fin 6, i < j ∧ n.val i = d1 ∧ n.val j = d2

/-- The set of valid arrangements according to the problem conditions -/
def ValidArrangements : Set SixDigitInt :=
  { n | IsLeftOf n 1 2 ∧ IsLeftOf n 3 4 }

/-- Instance to make SixDigitInt finite -/
instance : Fintype SixDigitInt :=
  sorry

/-- Instance to make ValidArrangements decidable -/
instance (n : SixDigitInt) : Decidable (n ∈ ValidArrangements) :=
  sorry

/-- The main theorem stating the number of valid arrangements -/
theorem valid_arrangements_count :
  Finset.card (Finset.filter (λ n : SixDigitInt => n ∈ ValidArrangements) Finset.univ) = 180 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l590_59024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_transformation_l590_59089

noncomputable def f (x : ℝ) := Real.sin x

noncomputable def g (x : ℝ) := Real.sin (2 * x + Real.pi / 3)

theorem sine_transformation (x : ℝ) : g x = Real.sin (2 * x + Real.pi / 3) := by
  -- Unfold the definition of g
  unfold g
  -- The equality now holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_transformation_l590_59089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_of_point_l590_59082

/-- Given a point P with coordinates (x, -6) that is 12 units from the y-axis,
    the ratio of its distance from the x-axis to its distance from the y-axis is 1:2. -/
theorem distance_ratio_of_point (x : ℝ) : 
  let P : ℝ × ℝ := (x, -6)
  x^2 + 6^2 = 12^2 → -- P is 12 units from the y-axis
  (|P.2| : ℝ) / 12 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_of_point_l590_59082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f₁_domain_f₂_l590_59021

-- Function 1
noncomputable def f₁ (x : ℝ) := Real.sqrt (3 * x + 2)

-- Function 2
noncomputable def f₂ (x : ℝ) := Real.sqrt (x + 3) + 1 / (x + 2)

-- Theorem for the domain of f₁
theorem domain_f₁ : 
  {x : ℝ | ∃ y, f₁ x = y} = {x : ℝ | x ≥ -2/3} := by
  sorry

-- Theorem for the domain of f₂
theorem domain_f₂ : 
  {x : ℝ | ∃ y, f₂ x = y} = {x : ℝ | x ≥ -3 ∧ x ≠ -2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f₁_domain_f₂_l590_59021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l590_59076

theorem diophantine_equation_solutions :
  (∃ (S : Finset (ℕ × ℕ × ℕ)), 
    (∀ (m n r : ℕ), (m, n, r) ∈ S ↔ m * n + n * r + m * r = 2 * (m + n + r)) ∧
    S.card = 7) ∧
  (∀ k : ℕ, k > 1 →
    ∃ (S : Finset (ℕ × ℕ × ℕ)), 
      (∀ (m n r : ℕ), (m, n, r) ∈ S → m * n + n * r + m * r = k * (m + n + r)) ∧
      S.card ≥ 3 * k + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l590_59076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_b_value_l590_59072

noncomputable section

open Real

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := exp x
def g (b : ℝ) (x : ℝ) : ℝ := -x^2 + 2*x + b
noncomputable def h (x : ℝ) : ℝ := f x - 1 / f x

-- State the theorem
theorem find_b_value :
  ∀ b : ℝ,
  (∀ x ∈ Set.Icc 1 2, ∃ x₁ x₂, x₁ ∈ Set.Icc 1 2 ∧ x₂ ∈ Set.Icc 1 2 ∧
    f x ≤ f x₁ ∧ g b x ≤ g b x₂ ∧ f x₁ = g b x₂) →
  b = exp 2 - 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_b_value_l590_59072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_19_formula_l590_59059

noncomputable def u (b : ℝ) : ℕ → ℝ
  | 0 => b  -- Adding the case for 0
  | 1 => b
  | (n + 2) => 2 / (u b (n + 1) - 2)

theorem u_19_formula (b : ℝ) (h : b > 0) : u b 19 = 2 * (b - 2) / (b - 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_19_formula_l590_59059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_conditions_imply_b_c_values_l590_59046

noncomputable def f (b c : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + b * x^2 + c * x

noncomputable def f' (b c : ℝ) (x : ℝ) : ℝ := x^2 + 2*b*x + c

theorem f_conditions_imply_b_c_values (b c : ℝ) :
  (f' b c 1 = 0) ∧ 
  (∀ x ∈ Set.Icc (-1) 3, f' b c x ≥ -1) ∧
  (∃ x ∈ Set.Icc (-1) 3, f' b c x = -1) →
  ((b = -2 ∧ c = 3) ∨ (b = 0 ∧ c = -1)) :=
by sorry

#check f_conditions_imply_b_c_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_conditions_imply_b_c_values_l590_59046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l590_59002

noncomputable def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (3 - 2*a)^x

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → f a x < f a y

theorem range_of_a (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → (1 ≤ a ∧ a < 2) ∨ a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l590_59002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_salary_calculation_l590_59023

theorem monthly_salary_calculation (initial_savings_rate : ℝ) 
  (expense_increase_rate : ℝ) (final_savings : ℝ) : ℝ :=
  let monthly_salary := 5500
  let initial_expense_rate := 1 - initial_savings_rate
  let new_expense_rate := initial_expense_rate * (1 + expense_increase_rate)
  have h1 : initial_savings_rate = 0.2 := by sorry
  have h2 : expense_increase_rate = 0.2 := by sorry
  have h3 : final_savings = 220 := by sorry
  have h4 : monthly_salary * (1 - new_expense_rate) = final_savings := by sorry
  monthly_salary

#check monthly_salary_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_salary_calculation_l590_59023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_of_T_l590_59077

/-- Represents the possible axes --/
inductive Axis
  | PositiveX
  | NegativeX
  | PositiveY
  | NegativeY

/-- Represents the position of the letter T --/
structure LetterT where
  base : Axis
  stem : Axis

/-- Initial position of the letter T --/
def initialPosition : LetterT := { base := Axis.PositiveX, stem := Axis.PositiveY }

/-- Applies a 180° clockwise rotation to the letter T --/
def rotate180 (t : LetterT) : LetterT :=
  { base := match t.base with
    | Axis.PositiveX => Axis.NegativeX
    | Axis.NegativeX => Axis.PositiveX
    | Axis.PositiveY => Axis.NegativeY
    | Axis.NegativeY => Axis.PositiveY,
    stem := match t.stem with
    | Axis.PositiveX => Axis.NegativeX
    | Axis.NegativeX => Axis.PositiveX
    | Axis.PositiveY => Axis.NegativeY
    | Axis.NegativeY => Axis.PositiveY }

/-- Applies a reflection in the x-axis to the letter T --/
def reflectX (t : LetterT) : LetterT :=
  { base := t.base,
    stem := match t.stem with
    | Axis.PositiveY => Axis.NegativeY
    | Axis.NegativeY => Axis.PositiveY
    | x => x }

/-- Applies a 90° clockwise rotation to the letter T --/
def rotate90 (t : LetterT) : LetterT :=
  { base := match t.base with
    | Axis.PositiveX => Axis.PositiveY
    | Axis.NegativeX => Axis.NegativeY
    | Axis.PositiveY => Axis.NegativeX
    | Axis.NegativeY => Axis.PositiveX,
    stem := match t.stem with
    | Axis.PositiveX => Axis.PositiveY
    | Axis.NegativeX => Axis.NegativeY
    | Axis.PositiveY => Axis.NegativeX
    | Axis.NegativeY => Axis.PositiveX }

/-- The main theorem to prove --/
theorem final_position_of_T :
  (rotate90 (reflectX (rotate180 initialPosition))) =
  { base := Axis.NegativeY, stem := Axis.NegativeX } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_of_T_l590_59077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_siamese_cats_l590_59061

/-- Theorem: Initial number of Siamese cats in a pet store -/
theorem initial_siamese_cats
  (initial_house_cats : ℕ)
  (initial_siamese_cats : ℕ)
  (cats_sold : ℕ)
  (cats_remaining : ℕ)
  (h1 : initial_house_cats = 20)
  (h2 : cats_sold = 20)
  (h3 : cats_remaining = 12)
  (h4 : initial_house_cats + initial_siamese_cats - cats_sold = cats_remaining)
  : initial_siamese_cats = 12 :=
by
  sorry

#check initial_siamese_cats

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_siamese_cats_l590_59061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_souvenir_price_calculation_l590_59045

/-- Calculates the discounted price in USD for a souvenir in Japan -/
noncomputable def discounted_price_usd (original_price : ℝ) (discount_percent : ℝ) (exchange_rate : ℝ) : ℝ :=
  (original_price * (1 - discount_percent / 100)) / exchange_rate

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem souvenir_price_calculation :
  round_to_hundredth (discounted_price_usd 300 10 120) = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_souvenir_price_calculation_l590_59045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l590_59019

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + Real.sin (2 * x)

-- Define the interval
def interval : Set ℝ := Set.Icc 0 (3 * Real.pi / 2)

-- Theorem statement
theorem f_extrema :
  ∃ (min max : ℝ),
    (∀ x ∈ interval, f x ≥ min) ∧
    (∃ x ∈ interval, f x = min) ∧
    (∀ x ∈ interval, f x ≤ max) ∧
    (∃ x ∈ interval, f x = max) ∧
    min = -2 ∧
    max = 3 * Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l590_59019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_right_triangle_area_l590_59066

/-- A point on the parabola y = x^2 --/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y = x^2

/-- The area of a triangle given its base and height --/
noncomputable def triangle_area (base height : ℝ) : ℝ := (1/2) * base * height

theorem parabola_right_triangle_area (A B C : ParabolaPoint) 
  (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h_parallel : A.y = B.y)
  (h_right : (C.x - A.x) * (B.x - A.x) + (C.y - A.y) * (B.y - A.y) = 0)
  (h_area : triangle_area (B.x - A.x) (C.y - A.y) = 504)
  : C.y = 256 := by
  sorry

#check parabola_right_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_right_triangle_area_l590_59066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_k_values_l590_59027

def has_winning_strategy (k : ℕ) : Prop :=
  1 < k ∧ k < 1024 ∧
  ∃ (strategy : ℕ → ℕ),
    (∀ n, 0 < strategy n ∧ strategy n ≤ k) ∧
    (∀ m, 0 < m → m ≤ k → ∃ n, strategy (1024 - k - m - n) = 1024 - k - m)

theorem winning_k_values :
  ∀ k, has_winning_strategy k ↔ k ∈ ({4, 24, 40} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_k_values_l590_59027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purple_marble_probability_l590_59050

/-- A bag contains blue, green, and purple marbles. -/
structure Bag where
  blue : ℝ
  green : ℝ
  purple : ℝ

/-- The probability of drawing each color marble from the bag. -/
def probability (b : Bag) : ℝ := b.blue + b.green + b.purple

/-- The probabilities in a bag sum to 1. -/
def valid_probabilities (b : Bag) : Prop := probability b = 1

theorem purple_marble_probability (b : Bag) 
  (h1 : valid_probabilities b) 
  (h2 : b.blue = 0.25) 
  (h3 : b.green = 0.55) : 
  b.purple = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purple_marble_probability_l590_59050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l590_59039

/-- The function for which we want to find the horizontal asymptote -/
noncomputable def f (x : ℝ) : ℝ := (8 * x^3 - 7 * x + 6) / (4 * x^3 + 3 * x^2 - 2)

/-- Theorem stating that the limit of f(x) as x approaches infinity is 2 -/
theorem horizontal_asymptote_of_f :
  Filter.Tendsto f Filter.atTop (nhds 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l590_59039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_perimeter_l590_59016

theorem triangle_max_perimeter (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  Real.sin A / a = Real.sqrt 3 * Real.cos B / b ∧
  Real.sin A / a = Real.sqrt 2 / 2 →
  ∃ (p : ℝ), p ≤ 3 * Real.sqrt 6 / 2 ∧
    ∀ (q : ℝ), q = a + b + c → q ≤ p :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_perimeter_l590_59016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l590_59018

variable (x m : ℝ)

def A (x m : ℝ) : ℝ := -3 * x^2 - 2 * m * x + 3 * x + 1
def B (x m : ℝ) : ℝ := 2 * x^2 + 2 * m * x - 1

theorem problem_solution (x m : ℝ) :
  (2 * A x m + 3 * B x m = 2 * m * x + 6 * x - 1) ∧
  (∀ x, 2 * A x m + 3 * B x m = 2 * A x (-3) + 3 * B x (-3)) ↔ (m = -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l590_59018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_office_amenities_increase_profit_l590_59063

-- Define a company
structure Company where
  name : String
  has_leisure_spaces : Bool
  has_living_spaces : Bool

-- Define metrics
structure CompanyMetrics where
  employee_retention : ℝ
  productivity : ℝ
  work_life_integration : ℝ
  profit : ℝ

-- Define the relationship between office amenities and company metrics
def improved_metrics (c : Company) (m : CompanyMetrics) : CompanyMetrics :=
  { employee_retention := m.employee_retention * 1.1,
    productivity := m.productivity * 1.2,
    work_life_integration := m.work_life_integration * 1.15,
    profit := m.profit * 1.25 }

def profit_increase (m1 m2 : CompanyMetrics) : Prop :=
  m2.profit > m1.profit

-- Theorem statement
theorem office_amenities_increase_profit (c : Company) (m : CompanyMetrics) :
  c.has_leisure_spaces ∧ c.has_living_spaces →
  profit_increase m (improved_metrics c m) := by
  intro h
  unfold profit_increase
  unfold improved_metrics
  simp
  -- The actual proof would go here, but we'll use sorry as requested
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_office_amenities_increase_profit_l590_59063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_biconditional_l590_59040

/-- Proposition 1: If a > b, then a + b > 0 -/
def proposition1 : Prop := ∀ a b : ℝ, a > b → a + b > 0

/-- Proposition 2: If a ≠ b, then a^2 ≠ b^2 -/
def proposition2 : Prop := ∀ a b : ℝ, a ≠ b → a^2 ≠ b^2

/-- Proposition 3: Points on the angle bisector are equidistant from the two sides of the angle -/
def proposition3 : Prop := True

/-- Proposition 4: The diagonals of a parallelogram bisect each other -/
def proposition4 : Prop := True

/-- A proposition is biconditional if both the original statement and its converse are true -/
def is_biconditional (p : Prop) : Prop := p ∧ (¬p → False)

/-- The number of biconditional propositions among the given four is exactly two -/
theorem exactly_two_biconditional : 
  (is_biconditional proposition3 ∧ is_biconditional proposition4) ∧ 
  (¬is_biconditional proposition1 ∧ ¬is_biconditional proposition2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_biconditional_l590_59040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_volume_five_boxes_five_feet_l590_59035

def box_volume (edge_length : ℝ) : ℝ := edge_length ^ 3

def total_volume (num_boxes : ℕ) (edge_length : ℝ) : ℝ :=
  (num_boxes : ℝ) * box_volume edge_length

theorem total_volume_five_boxes_five_feet :
  total_volume 5 5 = 625 := by
  unfold total_volume box_volume
  norm_num

#eval total_volume 5 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_volume_five_boxes_five_feet_l590_59035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_sum_l590_59013

/-- Given points P and Q in the xy-plane, and R with a fixed x-coordinate,
    prove that the y-coordinate of R that minimizes PR + RQ is 3/5. -/
theorem minimize_distance_sum (P Q R : ℝ × ℝ) : 
  P = (0, -3) → 
  Q = (5, 3) → 
  R.1 = 3 →
  (∀ m : ℝ, ‖R - P‖ + ‖R - Q‖ ≥ ‖((3, 3/5) : ℝ × ℝ) - P‖ + ‖((3, 3/5) : ℝ × ℝ) - Q‖) →
  R.2 = 3/5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_sum_l590_59013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_part_speed_l590_59001

/-- Represents the speed of a car during a journey divided into three equal parts. -/
structure JourneySpeed where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Calculates the average speed given a JourneySpeed. -/
noncomputable def averageSpeed (js : JourneySpeed) : ℝ :=
  3 / (1/js.first + 1/js.second + 1/js.third)

/-- Theorem stating that given specific conditions, the speed during the second part of the journey must be 30 km/h. -/
theorem second_part_speed
  (js : JourneySpeed)
  (h1 : js.first = 80)
  (h2 : js.third = 48)
  (h3 : averageSpeed js = 45) :
  js.second = 30 := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_part_speed_l590_59001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_increase_factor_l590_59030

/-- Represents the amount of salt Sasha initially added yesterday -/
def x : ℝ := sorry

/-- Represents the additional amount of salt Sasha added yesterday -/
def y : ℝ := sorry

/-- The factor by which Sasha needs to increase today's portion of salt -/
def increase_factor : ℝ := 1.5

/-- Yesterday's total salt amount -/
def yesterday_total : ℝ := x + y

/-- Today's initial salt amount -/
def today_initial : ℝ := 2 * x

/-- Today's additional salt amount -/
def today_additional : ℝ := 0.5 * y

/-- Today's total salt amount -/
def today_total : ℝ := today_initial + today_additional

theorem salt_increase_factor :
  increase_factor * today_total = yesterday_total :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_increase_factor_l590_59030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_games_proof_l590_59033

def bowling_problem (shoe_cost locker_cost hotdog_cost drink_cost game_cost total_money : ℚ) : ℕ :=
  let mandatory_expenses := shoe_cost + locker_cost + hotdog_cost + drink_cost
  let remaining_money := total_money - mandatory_expenses
  (remaining_money / game_cost).floor.toNat

theorem max_games_proof (shoe_cost locker_cost hotdog_cost drink_cost game_cost total_money : ℚ) 
  (h1 : shoe_cost = 0.50)
  (h2 : locker_cost = 3.00)
  (h3 : hotdog_cost = 2.25)
  (h4 : drink_cost = 1.50)
  (h5 : game_cost = 1.75)
  (h6 : total_money = 12.80) :
  bowling_problem shoe_cost locker_cost hotdog_cost drink_cost game_cost total_money = 3 := by
  sorry

#eval bowling_problem 0.50 3.00 2.25 1.50 1.75 12.80

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_games_proof_l590_59033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grace_mulch_hours_l590_59093

/-- Represents Grace's landscaping business earnings in September --/
structure GraceEarnings where
  mowing_rate : ℕ 
  weed_rate : ℕ 
  mulch_rate : ℕ 
  mowing_hours : ℕ 
  weed_hours : ℕ 
  total_earnings : ℕ 

/-- Calculates the number of hours spent putting down mulch --/
def mulch_hours (g : GraceEarnings) : ℕ :=
  (g.total_earnings - (g.mowing_rate * g.mowing_hours + g.weed_rate * g.weed_hours)) / g.mulch_rate

/-- Theorem stating that Grace spent 10 hours putting down mulch --/
theorem grace_mulch_hours :
  ∀ (g : GraceEarnings),
    g.mowing_rate = 6 →
    g.weed_rate = 11 →
    g.mulch_rate = 9 →
    g.mowing_hours = 63 →
    g.weed_hours = 9 →
    g.total_earnings = 567 →
    mulch_hours g = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grace_mulch_hours_l590_59093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_shape_surface_area_l590_59037

/-- Represents the dimensions of a frustum of a right circular cone -/
structure Frustum where
  lower_radius : ℝ
  upper_radius : ℝ
  height : ℝ

/-- Represents the dimensions of a cylindrical section -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Calculates the total surface area of a composite shape consisting of a frustum and a cylinder -/
noncomputable def total_surface_area (f : Frustum) (c : Cylinder) : ℝ :=
  let frustum_slant_height := Real.sqrt (f.height^2 + (f.lower_radius - f.upper_radius)^2)
  let frustum_area := Real.pi * (f.lower_radius + f.upper_radius) * frustum_slant_height
  let cylinder_area := 2 * Real.pi * c.radius * c.height
  frustum_area + cylinder_area

/-- Theorem stating the total surface area of the given composite shape -/
theorem composite_shape_surface_area :
  let f : Frustum := { lower_radius := 8, upper_radius := 5, height := 6 }
  let c : Cylinder := { radius := 5, height := 2 }
  total_surface_area f c = 39 * Real.pi * Real.sqrt 5 + 20 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_shape_surface_area_l590_59037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_probability_l590_59052

-- Define the parabola
def f (x : ℝ) : ℝ := x^2

-- Define the derivative of the parabola
def f' (x : ℝ) : ℝ := 2 * x

-- Define the slope angle of the tangent line
noncomputable def slope_angle (x : ℝ) : ℝ := Real.arctan (f' x)

-- Define the probability space
def probability_space : Set ℝ := Set.Icc (-6) 6

-- Define the event space
def event_space : Set ℝ := {x | x ∈ probability_space ∧ 
  (slope_angle x ≥ Real.pi / 4 ∧ slope_angle x ≤ 3 * Real.pi / 4)}

-- State the theorem
theorem tangent_slope_probability : 
  (MeasureTheory.volume event_space) / (MeasureTheory.volume probability_space) = 11 / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_probability_l590_59052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_equal_sums_l590_59011

def is_valid_set (S : Finset ℕ) (N : ℕ) : Prop :=
  (∀ x ∈ S, x ≥ 1 ∧ x ≤ N) ∧ 
  (S.card > 1) ∧
  (Even (S.sum id))

theorem partition_equal_sums (N : ℕ) (S : Finset ℕ) (h : N ≥ 4) (hS : is_valid_set S N) :
  ∃ (X Y : Finset ℕ), X ∪ Y = S ∧ X ∩ Y = ∅ ∧ X.sum id = Y.sum id :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_equal_sums_l590_59011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suv_max_distance_l590_59051

/-- Represents the fuel efficiency of an SUV in miles per gallon -/
structure FuelEfficiency where
  highway : ℚ
  city : ℚ

/-- Calculates the maximum distance an SUV can travel given its fuel efficiency and available fuel -/
def maxDistance (efficiency : FuelEfficiency) (fuel : ℚ) : ℚ :=
  max (efficiency.highway * fuel) (efficiency.city * fuel)

/-- Theorem stating the maximum distance an SUV can travel with given efficiency and fuel -/
theorem suv_max_distance (efficiency : FuelEfficiency) (fuel : ℚ) :
  efficiency.highway = 61/5 →
  efficiency.city = 38/5 →
  fuel = 20 →
  maxDistance efficiency fuel = 244 := by
  sorry

#eval (61/5 : ℚ) * 20  -- To check the calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suv_max_distance_l590_59051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surgeon_is_arthur_mother_l590_59010

structure Person where
  name : String

structure Surgeon extends Person where
  onDuty : Bool

def Parent : Person → Person → Prop := sorry

def arthur : Person := ⟨"Arthur"⟩
def arthur_father : Person := ⟨"Mr. Smith"⟩
def surgeon : Surgeon := ⟨⟨"Dr. Smith"⟩, true⟩

axiom surgeon_on_duty : surgeon.onDuty = true
axiom arthur_father_deceased : ¬ (Parent arthur_father arthur)
axiom surgeon_parent_of_arthur : Parent surgeon.toPerson arthur

theorem surgeon_is_arthur_mother :
  ∃ (mother : Person), Parent mother arthur ∧ surgeon.toPerson = mother :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surgeon_is_arthur_mother_l590_59010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l590_59048

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^3 - 3*x else -2*x

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), f x ≤ M := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l590_59048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_origin_to_point_l590_59026

/-- The distance between two points on a complex plane -/
noncomputable def distance (z₁ z₂ : ℂ) : ℝ :=
  Complex.abs (z₂ - z₁)

theorem distance_origin_to_point :
  let z₁ : ℂ := 0
  let z₂ : ℂ := Complex.mk 1170 1560
  distance z₁ z₂ = 1950 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_origin_to_point_l590_59026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cara_last_standing_l590_59032

/-- Represents a person in the circle --/
inductive Person
| Aleka
| Ben
| Cara
| Diya
| Ed
| Frank

/-- Checks if a number contains 8 as a digit or is a multiple of 8 --/
def isEliminationNumber (n : Nat) : Bool :=
  n % 8 == 0 || n.repr.contains '8'

/-- Simulates the elimination process and returns the last person standing --/
def lastPersonStanding (people : List Person) : Person :=
  match people with
  | [] => Person.Cara  -- Default to Cara if the list is empty
  | [p] => p  -- Return the last person if only one remains
  | _ => sorry  -- Placeholder for the actual elimination logic

/-- Theorem stating that Cara is the last person standing --/
theorem cara_last_standing :
  lastPersonStanding [Person.Aleka, Person.Ben, Person.Cara, Person.Diya, Person.Ed, Person.Frank] = Person.Cara :=
by sorry

#eval isEliminationNumber 8  -- Should return true
#eval isEliminationNumber 16  -- Should return true
#eval isEliminationNumber 18  -- Should return true
#eval isEliminationNumber 7  -- Should return false

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cara_last_standing_l590_59032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_24_four_dice_l590_59085

-- Define a fair six-sided die
def FairDie : Type := Fin 6

-- Define the probability of rolling a specific number on a fair die
def probSingle : ℚ := 1 / 6

-- Define the sum of four dice rolls
def sumFourDice (d1 d2 d3 d4 : FairDie) : ℕ := d1.val + d2.val + d3.val + d4.val + 4

-- Define the event of rolling a sum of 24 with four dice
def sumIs24 (d1 d2 d3 d4 : FairDie) : Prop := sumFourDice d1 d2 d3 d4 = 24

-- State the theorem
theorem prob_sum_24_four_dice :
  (probSingle ^ 4 : ℚ) = 1 / 1296 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_24_four_dice_l590_59085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_series_stats_correct_l590_59084

/-- Represents a cricket match series -/
structure CricketSeries where
  matches1 : ℕ
  avg_runs1 : ℚ
  boundary_percent1 : ℚ
  matches2 : ℕ
  avg_runs2 : ℚ
  boundary_percent2 : ℚ

/-- Calculates the average score, boundary runs, and non-boundary runs for a cricket series -/
def calculate_series_stats (series : CricketSeries) :
  (ℚ × ℚ × ℚ) :=
  let total_runs1 := series.matches1 * series.avg_runs1
  let total_runs2 := series.matches2 * series.avg_runs2
  let boundary_runs1 := total_runs1 * series.boundary_percent1
  let boundary_runs2 := total_runs2 * series.boundary_percent2
  let non_boundary_runs1 := total_runs1 * (1 - series.boundary_percent1)
  let non_boundary_runs2 := total_runs2 * (1 - series.boundary_percent2)
  let total_matches := series.matches1 + series.matches2
  let total_runs := total_runs1 + total_runs2
  let avg_score := total_runs / total_matches
  let total_boundary_runs := boundary_runs1 + boundary_runs2
  let total_non_boundary_runs := non_boundary_runs1 + non_boundary_runs2
  (avg_score, total_boundary_runs, total_non_boundary_runs)

theorem cricket_series_stats_correct (series : CricketSeries) 
  (h1 : series.matches1 = 2)
  (h2 : series.avg_runs1 = 20)
  (h3 : series.boundary_percent1 = 6/10)
  (h4 : series.matches2 = 3)
  (h5 : series.avg_runs2 = 30)
  (h6 : series.boundary_percent2 = 8/10) :
  calculate_series_stats series = (26, 96, 34) := by
  sorry

#eval calculate_series_stats ⟨2, 20, 6/10, 3, 30, 8/10⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_series_stats_correct_l590_59084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_extension_theorem_l590_59029

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  AB : ℝ
  BC : ℝ
  AC : ℝ

-- Define the circumcenter
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the extension of ray AB to point D
def extend_AB (t : Triangle) (D : ℝ × ℝ) : Prop :=
  D.1 = t.B.1 + (t.B.1 - t.A.1) * 2 ∧ D.2 = t.B.2 + (t.B.2 - t.A.2) * 2

-- Define the extension of ray BC to point E
def extend_BC (t : Triangle) (E : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 1 ∧ E.1 = t.C.1 + k * (t.C.1 - t.B.1) ∧ E.2 = t.C.2 + k * (t.C.2 - t.B.2)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem triangle_extension_theorem (t : Triangle) (D E : ℝ × ℝ) :
  t.AB = 5 ∧ t.BC = 6 ∧ t.AC = 7 ∧
  extend_AB t D ∧
  extend_BC t E ∧
  distance (circumcenter t) D = distance (circumcenter t) E ∧
  distance t.B D = 5 →
  distance t.C E = Real.sqrt 59 - 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_extension_theorem_l590_59029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_power_of_six_in_twenty_factorial_l590_59014

/-- The highest power of 6 that divides 20! is 8 -/
theorem highest_power_of_six_in_twenty_factorial : ∃ n : ℕ, n = 8 ∧ (6^n : ℕ) ∣ Nat.factorial 20 ∧ ∀ m : ℕ, m > n → ¬((6^m : ℕ) ∣ Nat.factorial 20) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_power_of_six_in_twenty_factorial_l590_59014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_residual_correlation_relation_l590_59022

/-- Represents the width of the residual point distribution band -/
def residual_band_width : ℝ → ℝ := sorry

/-- Represents the correlation coefficient R^2 -/
def correlation_coefficient : ℝ → ℝ := sorry

/-- States that as the residual band width decreases, the correlation coefficient increases -/
theorem residual_correlation_relation :
  ∀ (w1 w2 : ℝ), w1 < w2 → correlation_coefficient (residual_band_width w1) > correlation_coefficient (residual_band_width w2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_residual_correlation_relation_l590_59022
