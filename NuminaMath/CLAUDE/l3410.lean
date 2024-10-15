import Mathlib

namespace NUMINAMATH_CALUDE_johnson_family_reunion_l3410_341044

theorem johnson_family_reunion (num_children : ℕ) (num_adults : ℕ) (num_blue_adults : ℕ) : 
  num_children = 45 →
  num_adults = num_children / 3 →
  num_blue_adults = num_adults / 3 →
  num_adults - num_blue_adults = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_johnson_family_reunion_l3410_341044


namespace NUMINAMATH_CALUDE_workers_savings_l3410_341097

theorem workers_savings (P : ℝ) (f : ℝ) (h1 : P > 0) (h2 : 0 ≤ f ∧ f ≤ 1) 
  (h3 : 12 * f * P = 6 * (1 - f) * P) : f = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_workers_savings_l3410_341097


namespace NUMINAMATH_CALUDE_speaking_orders_count_l3410_341051

def total_students : ℕ := 6
def speakers_to_select : ℕ := 4
def specific_students : ℕ := 2

theorem speaking_orders_count : 
  (total_students.choose speakers_to_select * speakers_to_select.factorial) -
  ((total_students - specific_students).choose speakers_to_select * speakers_to_select.factorial) = 336 :=
by sorry

end NUMINAMATH_CALUDE_speaking_orders_count_l3410_341051


namespace NUMINAMATH_CALUDE_express_delivery_growth_rate_l3410_341094

theorem express_delivery_growth_rate (initial_packages : ℕ) (final_packages : ℕ) (x : ℝ) :
  initial_packages = 200 →
  final_packages = 242 →
  initial_packages * (1 + x)^2 = final_packages :=
by sorry

end NUMINAMATH_CALUDE_express_delivery_growth_rate_l3410_341094


namespace NUMINAMATH_CALUDE_total_annual_income_percentage_l3410_341035

def initial_investment : ℝ := 2800
def initial_rate : ℝ := 0.05
def additional_investment : ℝ := 1400
def additional_rate : ℝ := 0.08

theorem total_annual_income_percentage :
  let total_investment := initial_investment + additional_investment
  let total_income := initial_investment * initial_rate + additional_investment * additional_rate
  (total_income / total_investment) * 100 = 6 := by
sorry

end NUMINAMATH_CALUDE_total_annual_income_percentage_l3410_341035


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l3410_341070

theorem rectangular_to_polar_conversion :
  ∀ (x y : ℝ),
    x = 3 ∧ y = -3 →
    ∃ (r θ : ℝ),
      r > 0 ∧
      0 ≤ θ ∧ θ < 2 * Real.pi ∧
      r = 3 * Real.sqrt 2 ∧
      θ = 7 * Real.pi / 4 ∧
      x = r * Real.cos θ ∧
      y = r * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l3410_341070


namespace NUMINAMATH_CALUDE_levi_goal_difference_l3410_341096

/-- The number of baskets Levi wants to beat his brother by -/
def basketDifference (leviInitial : ℕ) (brotherInitial : ℕ) (brotherIncrease : ℕ) (leviIncrease : ℕ) : ℕ :=
  (leviInitial + leviIncrease) - (brotherInitial + brotherIncrease)

/-- Theorem stating that Levi wants to beat his brother by 5 baskets -/
theorem levi_goal_difference : basketDifference 8 12 3 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_levi_goal_difference_l3410_341096


namespace NUMINAMATH_CALUDE_strange_number_theorem_l3410_341017

theorem strange_number_theorem : ∃! x : ℝ, (x - 7) * 7 = (x - 11) * 11 := by
  sorry

end NUMINAMATH_CALUDE_strange_number_theorem_l3410_341017


namespace NUMINAMATH_CALUDE_chord_intersection_lengths_l3410_341031

/-- Given a circle with radius 7, perpendicular diameters EF and GH, and a chord EJ of length 12
    intersecting GH at M, prove that GM = 7 + √13 and MH = 7 - √13 -/
theorem chord_intersection_lengths (O : ℝ × ℝ) (E F G H J M : ℝ × ℝ) :
  let r : ℝ := 7
  let circle := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}
  (E ∈ circle) ∧ (F ∈ circle) ∧ (G ∈ circle) ∧ (H ∈ circle) ∧ (J ∈ circle) →
  (E.1 - F.1) * (G.2 - H.2) = 0 ∧ (E.2 - F.2) * (G.1 - H.1) = 0 →
  (E.1 - J.1)^2 + (E.2 - J.2)^2 = 12^2 →
  M.1 = (G.1 + H.1) / 2 ∧ M.2 = (G.2 + H.2) / 2 →
  (M.1 - G.1)^2 + (M.2 - G.2)^2 = (7 + Real.sqrt 13)^2 ∧
  (M.1 - H.1)^2 + (M.2 - H.2)^2 = (7 - Real.sqrt 13)^2 :=
by sorry

end NUMINAMATH_CALUDE_chord_intersection_lengths_l3410_341031


namespace NUMINAMATH_CALUDE_continuous_function_solution_l3410_341033

open Set
open Function
open Real

theorem continuous_function_solution {f : ℝ → ℝ} (hf : Continuous f) 
  (hdom : ∀ x, x ∈ Ioo (-1) 1 → f x ≠ 0) 
  (heq : ∀ x ∈ Ioo (-1) 1, (1 - x^2) * f ((2*x) / (1 + x^2)) = (1 + x^2)^2 * f x) :
  ∃ c : ℝ, ∀ x ∈ Ioo (-1) 1, f x = c / (1 - x^2) :=
sorry

end NUMINAMATH_CALUDE_continuous_function_solution_l3410_341033


namespace NUMINAMATH_CALUDE_birthday_stickers_l3410_341029

theorem birthday_stickers (initial_stickers final_stickers : ℕ) 
  (h1 : initial_stickers = 39)
  (h2 : final_stickers = 61) :
  final_stickers - initial_stickers = 22 := by
sorry

end NUMINAMATH_CALUDE_birthday_stickers_l3410_341029


namespace NUMINAMATH_CALUDE_special_hexagon_perimeter_l3410_341021

/-- An equilateral hexagon with specific properties -/
structure SpecialHexagon where
  -- Side length of the hexagon
  side : ℝ
  -- Assertion that the hexagon is equilateral
  is_equilateral : True
  -- Three nonadjacent acute interior angles measure 45°
  has_three_45deg_angles : True
  -- The enclosed area of the hexagon is 12√3
  area_eq_12root3 : side^2 * (3 * Real.sqrt 2 / 4 + Real.sqrt 3 / 2 - Real.sqrt 6 / 4) = 12 * Real.sqrt 3

/-- The perimeter of a SpecialHexagon is 24 -/
theorem special_hexagon_perimeter (h : SpecialHexagon) : h.side * 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_special_hexagon_perimeter_l3410_341021


namespace NUMINAMATH_CALUDE_inverse_composition_theorem_l3410_341091

-- Define the functions f and g
variables (f g : ℝ → ℝ)

-- Define the condition f⁻¹ ∘ g = 3x - 2
def condition (f g : ℝ → ℝ) : Prop :=
  ∀ x, (f⁻¹ ∘ g) x = 3 * x - 2

-- Theorem statement
theorem inverse_composition_theorem (hfg : condition f g) :
  g⁻¹ (f (-10)) = -8/3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_composition_theorem_l3410_341091


namespace NUMINAMATH_CALUDE_pear_juice_blend_percentage_l3410_341049

/-- Represents the amount of juice extracted from a fruit -/
structure JuiceYield where
  fruit : String
  amount : ℚ
  count : ℕ

/-- Calculates the percentage of pear juice in a blend -/
def pear_juice_percentage (pear_yield orange_yield : JuiceYield) : ℚ :=
  let pear_juice := pear_yield.amount / pear_yield.count
  let orange_juice := orange_yield.amount / orange_yield.count
  let total_juice := pear_juice + orange_juice
  (pear_juice / total_juice) * 100

/-- Theorem: The percentage of pear juice in the blend is 40% -/
theorem pear_juice_blend_percentage :
  let pear_yield := JuiceYield.mk "pear" 8 3
  let orange_yield := JuiceYield.mk "orange" 8 2
  pear_juice_percentage pear_yield orange_yield = 40 := by
  sorry

end NUMINAMATH_CALUDE_pear_juice_blend_percentage_l3410_341049


namespace NUMINAMATH_CALUDE_no_loops_in_process_flowchart_l3410_341060

-- Define the basic concepts
def ProcessFlowchart : Type := Unit
def AlgorithmFlowchart : Type := Unit
def Process : Type := Unit
def FlowLine : Type := Unit

-- Define the properties of process flowcharts
def is_similar_to (pf : ProcessFlowchart) (af : AlgorithmFlowchart) : Prop := sorry
def refine_step_by_step (p : Process) : Prop := sorry
def connect_adjacent_processes (fl : FlowLine) : Prop := sorry
def is_directional (fl : FlowLine) : Prop := sorry

-- Define the concept of a loop
def Loop : Type := Unit
def contains_loop (pf : ProcessFlowchart) (l : Loop) : Prop := sorry

-- State the theorem
theorem no_loops_in_process_flowchart (pf : ProcessFlowchart) (af : AlgorithmFlowchart) 
  (p : Process) (fl : FlowLine) :
  is_similar_to pf af →
  refine_step_by_step p →
  connect_adjacent_processes fl →
  is_directional fl →
  ∀ l : Loop, ¬ contains_loop pf l := by
  sorry

end NUMINAMATH_CALUDE_no_loops_in_process_flowchart_l3410_341060


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l3410_341039

theorem triangle_ABC_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧
  (A + B + C = π) →
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) ∧ (b / Real.sin B = c / Real.sin C) →
  -- Given equation
  (Real.sqrt 3 * Real.sin B + b * Real.cos A = c) →
  -- Prove angle B
  (B = π / 6) ∧
  -- Prove area when a = √3 * c and b = 2
  (a = Real.sqrt 3 * c ∧ b = 2 → 
   (1 / 2) * a * b * Real.sin C = Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l3410_341039


namespace NUMINAMATH_CALUDE_crispy_red_plum_pricing_l3410_341020

theorem crispy_red_plum_pricing (x : ℝ) 
  (h1 : x > 0) 
  (h2 : x > 5) 
  (first_batch_cost : ℝ := 12000)
  (second_batch_cost : ℝ := 11000)
  (price_difference : ℝ := 5)
  (quantity_difference : ℝ := 40) :
  first_batch_cost / x = second_batch_cost / (x - price_difference) - quantity_difference := by
sorry

end NUMINAMATH_CALUDE_crispy_red_plum_pricing_l3410_341020


namespace NUMINAMATH_CALUDE_c_profit_is_21000_l3410_341008

/-- Calculates the profit for a partner given the total profit, total parts, and the partner's parts. -/
def calculateProfit (totalProfit : ℕ) (totalParts : ℕ) (partnerParts : ℕ) : ℕ :=
  (totalProfit / totalParts) * partnerParts

/-- Proves that given the specified conditions, C's profit is $21000. -/
theorem c_profit_is_21000 (totalProfit : ℕ) (a_parts b_parts c_parts : ℕ) :
  totalProfit = 56700 →
  a_parts = 8 →
  b_parts = 9 →
  c_parts = 10 →
  calculateProfit totalProfit (a_parts + b_parts + c_parts) c_parts = 21000 := by
  sorry

#eval calculateProfit 56700 27 10

end NUMINAMATH_CALUDE_c_profit_is_21000_l3410_341008


namespace NUMINAMATH_CALUDE_transformed_line_y_intercept_l3410_341079

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point vertically -/
def translateVertical (p : Point) (dy : ℝ) : Point :=
  { x := p.x, y := p.y + dy }

/-- Translates a point horizontally -/
def translateHorizontal (p : Point) (dx : ℝ) : Point :=
  { x := p.x + dx, y := p.y }

/-- Reflects a point in the line y = x -/
def reflectInDiagonal (p : Point) : Point :=
  { x := p.y, y := p.x }

/-- Applies a series of transformations to a line -/
def transformLine (l : Line) : Line :=
  sorry  -- The actual transformation is implemented here

/-- The main theorem stating that the transformed line has a y-intercept of -7 -/
theorem transformed_line_y_intercept :
  let originalLine : Line := { slope := 3, intercept := 6 }
  let transformedLine := transformLine originalLine
  transformedLine.intercept = -7 := by
  sorry


end NUMINAMATH_CALUDE_transformed_line_y_intercept_l3410_341079


namespace NUMINAMATH_CALUDE_election_winner_percentage_l3410_341006

theorem election_winner_percentage (total_votes : ℕ) (winner_votes : ℕ) (margin : ℕ) : 
  winner_votes = 490 →
  margin = 280 →
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 7/10 :=
by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l3410_341006


namespace NUMINAMATH_CALUDE_cube_surface_area_ratio_l3410_341019

theorem cube_surface_area_ratio :
  let original_volume : ℝ := 1000
  let removed_volume : ℝ := 64
  let original_side : ℝ := original_volume ^ (1/3)
  let removed_side : ℝ := removed_volume ^ (1/3)
  let shaded_area : ℝ := removed_side ^ 2
  let total_surface_area : ℝ := 
    3 * original_side ^ 2 + 
    3 * removed_side ^ 2 + 
    3 * (original_side ^ 2 - removed_side ^ 2)
  shaded_area / total_surface_area = 2 / 75
  := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_ratio_l3410_341019


namespace NUMINAMATH_CALUDE_min_y_value_l3410_341086

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 18*x + 54*y) :
  ∃ (y_min : ℝ), y_min = 27 - Real.sqrt 810 ∧ ∀ (y' : ℝ), ∃ (x' : ℝ), x'^2 + y'^2 = 18*x' + 54*y' → y' ≥ y_min :=
sorry

end NUMINAMATH_CALUDE_min_y_value_l3410_341086


namespace NUMINAMATH_CALUDE_sum_of_exponents_15_factorial_l3410_341001

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largestPerfectSquareDivisor (n : ℕ) : ℕ :=
  sorry

def sumOfPrimeFactorExponents (n : ℕ) : ℕ :=
  sorry

theorem sum_of_exponents_15_factorial :
  sumOfPrimeFactorExponents (largestPerfectSquareDivisor (factorial 15).sqrt) = 9 :=
sorry

end NUMINAMATH_CALUDE_sum_of_exponents_15_factorial_l3410_341001


namespace NUMINAMATH_CALUDE_triangle_altitude_l3410_341002

/-- Given a triangle with area 720 square feet and base 36 feet, prove its altitude is 40 feet -/
theorem triangle_altitude (area : ℝ) (base : ℝ) (altitude : ℝ) :
  area = 720 →
  base = 36 →
  area = (1/2) * base * altitude →
  altitude = 40 := by
sorry

end NUMINAMATH_CALUDE_triangle_altitude_l3410_341002


namespace NUMINAMATH_CALUDE_divisibility_1001_l3410_341016

theorem divisibility_1001 (n : ℕ) : 1001 ∣ n → 7 ∣ n ∧ 11 ∣ n ∧ 13 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_1001_l3410_341016


namespace NUMINAMATH_CALUDE_less_likely_white_ball_l3410_341092

theorem less_likely_white_ball (red_balls white_balls : ℕ) 
  (h_red : red_balls = 8) (h_white : white_balls = 2) :
  (white_balls : ℚ) / (red_balls + white_balls) < (red_balls : ℚ) / (red_balls + white_balls) :=
by sorry

end NUMINAMATH_CALUDE_less_likely_white_ball_l3410_341092


namespace NUMINAMATH_CALUDE_area_of_right_triangle_with_inscribed_circle_l3410_341052

-- Define the right triangle with inscribed circle
def RightTriangleWithInscribedCircle (a b c r : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧
  a^2 + b^2 = c^2 ∧
  c = 13 ∧
  (r + 6) = a ∧
  (r + 7) = b

-- Theorem statement
theorem area_of_right_triangle_with_inscribed_circle 
  (a b c r : ℝ) 
  (h : RightTriangleWithInscribedCircle a b c r) :
  (1/2 : ℝ) * a * b = 42 :=
by
  sorry

#check area_of_right_triangle_with_inscribed_circle

end NUMINAMATH_CALUDE_area_of_right_triangle_with_inscribed_circle_l3410_341052


namespace NUMINAMATH_CALUDE_volume_of_rhombus_revolution_l3410_341054

/-- A rhombus with side length 1 and shorter diagonal equal to its side -/
structure Rhombus where
  side_length : ℝ
  side_length_is_one : side_length = 1
  shorter_diagonal_eq_side : ℝ
  shorter_diagonal_eq_side_prop : shorter_diagonal_eq_side = side_length

/-- The volume of the solid of revolution formed by rotating the rhombus -/
noncomputable def volume_of_revolution (r : Rhombus) : ℝ := 
  3 * Real.pi / 2

/-- Theorem stating that the volume of the solid of revolution is 3π/2 -/
theorem volume_of_rhombus_revolution (r : Rhombus) : 
  volume_of_revolution r = 3 * Real.pi / 2 := by sorry

end NUMINAMATH_CALUDE_volume_of_rhombus_revolution_l3410_341054


namespace NUMINAMATH_CALUDE_min_sum_abs_roots_irrational_quadratic_l3410_341068

theorem min_sum_abs_roots_irrational_quadratic (p q : ℤ) 
  (h_irrational : ∀ (α : ℝ), α^2 + p*α + q = 0 → ¬ IsAlgebraic ℚ α) :
  ∃ (α₁ α₂ : ℝ), 
    α₁^2 + p*α₁ + q = 0 ∧ 
    α₂^2 + p*α₂ + q = 0 ∧ 
    |α₁| + |α₂| ≥ Real.sqrt 5 ∧
    (∃ (p' q' : ℤ) (β₁ β₂ : ℝ), 
      β₁^2 + p'*β₁ + q' = 0 ∧ 
      β₂^2 + p'*β₂ + q' = 0 ∧ 
      |β₁| + |β₂| = Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_abs_roots_irrational_quadratic_l3410_341068


namespace NUMINAMATH_CALUDE_base_sum_theorem_l3410_341056

theorem base_sum_theorem : ∃ (R₁ R₂ : ℕ), 
  (R₁ > 1 ∧ R₂ > 1) ∧
  (4 * R₁ + 5) * (R₂^2 - 1) = (3 * R₂ + 4) * (R₁^2 - 1) ∧
  (5 * R₁ + 4) * (R₂^2 - 1) = (4 * R₂ + 3) * (R₁^2 - 1) ∧
  R₁ + R₂ = 23 := by
  sorry

end NUMINAMATH_CALUDE_base_sum_theorem_l3410_341056


namespace NUMINAMATH_CALUDE_class_test_results_l3410_341026

theorem class_test_results (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ) 
  (h1 : p_first = 0.75)
  (h2 : p_second = 0.65)
  (h3 : p_neither = 0.20) :
  p_first + p_second - (1 - p_neither) = 0.60 := by
  sorry

end NUMINAMATH_CALUDE_class_test_results_l3410_341026


namespace NUMINAMATH_CALUDE_red_light_is_random_event_l3410_341003

/-- Definition of a random event -/
def is_random_event (event : Type) : Prop :=
  ∃ (outcome : event → Prop) (probability : event → ℝ),
    (∀ e : event, 0 ≤ probability e ∧ probability e ≤ 1) ∧
    (∀ e : event, outcome e ↔ probability e > 0)

/-- Representation of passing through an intersection with a traffic signal -/
inductive TrafficSignalEvent
| RedLight
| GreenLight
| YellowLight

/-- Theorem stating that encountering a red light at an intersection is a random event -/
theorem red_light_is_random_event :
  is_random_event TrafficSignalEvent :=
sorry

end NUMINAMATH_CALUDE_red_light_is_random_event_l3410_341003


namespace NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l3410_341059

-- Define a function to get the nth prime number
def nthPrime (n : ℕ) : ℕ := sorry

-- Define a function to sum the first n prime numbers
def sumFirstNPrimes (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_first_six_primes_mod_seventh_prime :
  sumFirstNPrimes 6 % nthPrime 7 = 7 := by sorry

end NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l3410_341059


namespace NUMINAMATH_CALUDE_minsu_age_is_15_l3410_341083

/-- Minsu's age this year -/
def minsu_age : ℕ := 15

/-- Minsu's mother's age this year -/
def mother_age : ℕ := minsu_age + 28

/-- The age difference between Minsu and his mother is 28 years this year -/
axiom age_difference : mother_age = minsu_age + 28

/-- After 13 years, the mother's age will be twice Minsu's age -/
axiom future_age_relation : mother_age + 13 = 2 * (minsu_age + 13)

/-- Theorem: Minsu's age this year is 15 -/
theorem minsu_age_is_15 : minsu_age = 15 := by
  sorry

end NUMINAMATH_CALUDE_minsu_age_is_15_l3410_341083


namespace NUMINAMATH_CALUDE_total_amount_paid_l3410_341010

-- Define the purchased amounts, rates, and discounts
def grape_amount : ℝ := 3
def mango_amount : ℝ := 9
def orange_amount : ℝ := 5
def banana_amount : ℝ := 7

def grape_rate : ℝ := 70
def mango_rate : ℝ := 55
def orange_rate : ℝ := 40
def banana_rate : ℝ := 20

def grape_discount : ℝ := 0.05
def mango_discount : ℝ := 0.10
def orange_discount : ℝ := 0.08
def banana_discount : ℝ := 0

def sales_tax : ℝ := 0.05

-- Define the theorem
theorem total_amount_paid : 
  let grape_cost := grape_amount * grape_rate
  let mango_cost := mango_amount * mango_rate
  let orange_cost := orange_amount * orange_rate
  let banana_cost := banana_amount * banana_rate

  let grape_discounted := grape_cost * (1 - grape_discount)
  let mango_discounted := mango_cost * (1 - mango_discount)
  let orange_discounted := orange_cost * (1 - orange_discount)
  let banana_discounted := banana_cost * (1 - banana_discount)

  let total_discounted := grape_discounted + mango_discounted + orange_discounted + banana_discounted
  let total_with_tax := total_discounted * (1 + sales_tax)

  total_with_tax = 1017.45 := by
    sorry

end NUMINAMATH_CALUDE_total_amount_paid_l3410_341010


namespace NUMINAMATH_CALUDE_orange_savings_percentage_l3410_341027

-- Define the given conditions
def family_size : ℕ := 4
def orange_cost : ℚ := 3/2  -- $1.5 as a rational number
def planned_spending : ℚ := 15

-- Define the theorem
theorem orange_savings_percentage :
  let saved_amount := family_size * orange_cost
  let savings_ratio := saved_amount / planned_spending
  savings_ratio * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_orange_savings_percentage_l3410_341027


namespace NUMINAMATH_CALUDE_percent_calculation_l3410_341058

theorem percent_calculation (x : ℝ) (h : 0.6 * x = 42) : 0.5 * x = 35 := by
  sorry

end NUMINAMATH_CALUDE_percent_calculation_l3410_341058


namespace NUMINAMATH_CALUDE_no_prime_sum_53_l3410_341082

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- State the theorem
theorem no_prime_sum_53 : ¬∃ (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 53 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_53_l3410_341082


namespace NUMINAMATH_CALUDE_prop_a_prop_b_prop_c_prop_d_l3410_341028

-- Proposition A
theorem prop_a (a b : ℝ) (h : b > a ∧ a > 0) : 1 / a > 1 / b := by sorry

-- Proposition B
theorem prop_b : ∃ a b c : ℝ, a > b ∧ a * c ≤ b * c := by sorry

-- Proposition C
theorem prop_c (a b c : ℝ) (h1 : a * c^2 > b * c^2) (h2 : c ≠ 0) : a > b := by sorry

-- Proposition D
theorem prop_d : 
  (∃ x : ℝ, x > -3 ∧ x^2 ≤ 9) ↔ ¬(∀ x : ℝ, x > -3 → x^2 > 9) := by sorry

end NUMINAMATH_CALUDE_prop_a_prop_b_prop_c_prop_d_l3410_341028


namespace NUMINAMATH_CALUDE_tunnel_length_tunnel_length_specific_l3410_341098

/-- Calculates the length of a tunnel given train specifications and travel time -/
theorem tunnel_length (train_length : ℝ) (train_speed_kmh : ℝ) (travel_time_min : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let travel_time_s := travel_time_min * 60
  let total_distance := train_speed_ms * travel_time_s
  let tunnel_length_m := total_distance - train_length
  let tunnel_length_km := tunnel_length_m / 1000
  tunnel_length_km

/-- The length of the tunnel is 1.7 km given the specified conditions -/
theorem tunnel_length_specific : tunnel_length 100 72 1.5 = 1.7 := by
  sorry

end NUMINAMATH_CALUDE_tunnel_length_tunnel_length_specific_l3410_341098


namespace NUMINAMATH_CALUDE_candy_eaten_l3410_341089

theorem candy_eaten (katie_candy : ℕ) (sister_candy : ℕ) (remaining_candy : ℕ) :
  katie_candy = 8 →
  sister_candy = 23 →
  remaining_candy = 23 →
  katie_candy + sister_candy - remaining_candy = 8 :=
by sorry

end NUMINAMATH_CALUDE_candy_eaten_l3410_341089


namespace NUMINAMATH_CALUDE_art_club_students_l3410_341037

/-- The number of students in the art club -/
def num_students : ℕ := 15

/-- The number of artworks each student makes per quarter -/
def artworks_per_quarter : ℕ := 2

/-- The number of quarters in a school year -/
def quarters_per_year : ℕ := 4

/-- The total number of artworks collected in two school years -/
def total_artworks : ℕ := 240

/-- Theorem stating that the number of students in the art club is 15 -/
theorem art_club_students :
  num_students * artworks_per_quarter * quarters_per_year * 2 = total_artworks :=
by sorry

end NUMINAMATH_CALUDE_art_club_students_l3410_341037


namespace NUMINAMATH_CALUDE_combined_population_l3410_341040

/-- The combined population of New England and New York given their relative populations -/
theorem combined_population (new_england_pop : ℕ) (new_york_pop : ℕ) :
  new_england_pop = 2100000 →
  new_york_pop = (2 : ℕ) * new_england_pop / 3 →
  new_england_pop + new_york_pop = 3500000 :=
by sorry

end NUMINAMATH_CALUDE_combined_population_l3410_341040


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l3410_341065

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 64 ways to distribute 7 distinguishable balls into 3 indistinguishable boxes -/
theorem seven_balls_three_boxes : distribute 7 3 = 64 := by sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l3410_341065


namespace NUMINAMATH_CALUDE_arcsin_cos_arcsin_plus_arccos_sin_arccos_l3410_341015

theorem arcsin_cos_arcsin_plus_arccos_sin_arccos (x : ℝ) : 
  Real.arcsin (Real.cos (Real.arcsin x)) + Real.arccos (Real.sin (Real.arccos x)) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_cos_arcsin_plus_arccos_sin_arccos_l3410_341015


namespace NUMINAMATH_CALUDE_complex_fraction_real_l3410_341075

theorem complex_fraction_real (a : ℝ) : 
  (((1 : ℂ) + a * Complex.I) / (2 - Complex.I)).im = 0 → a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_l3410_341075


namespace NUMINAMATH_CALUDE_sqrt_eight_and_one_ninth_l3410_341036

theorem sqrt_eight_and_one_ninth (x : ℝ) : 
  x = Real.sqrt (8 + 1 / 9) → x = Real.sqrt 73 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_and_one_ninth_l3410_341036


namespace NUMINAMATH_CALUDE_certain_and_uncertain_digits_l3410_341043

def value : ℝ := 945.673
def absolute_error : ℝ := 0.03

def is_certain (digit : ℕ) (place_value : ℝ) : Prop :=
  place_value > absolute_error

def is_uncertain (digit : ℕ) (place_value : ℝ) : Prop :=
  place_value < absolute_error

theorem certain_and_uncertain_digits :
  (is_certain 9 100) ∧
  (is_certain 4 10) ∧
  (is_certain 5 1) ∧
  (is_certain 6 0.1) ∧
  (is_uncertain 7 0.01) ∧
  (is_uncertain 3 0.001) :=
by sorry

end NUMINAMATH_CALUDE_certain_and_uncertain_digits_l3410_341043


namespace NUMINAMATH_CALUDE_seashell_difference_l3410_341081

theorem seashell_difference (fred_shells tom_shells : ℕ) 
  (h1 : fred_shells = 43)
  (h2 : tom_shells = 15) :
  fred_shells - tom_shells = 28 := by
  sorry

end NUMINAMATH_CALUDE_seashell_difference_l3410_341081


namespace NUMINAMATH_CALUDE_geometric_progression_proof_l3410_341046

/-- Given a geometric progression with b₃ = -1 and b₆ = 27/8,
    prove that the first term b₁ = -4/9 and the common ratio q = -3/2 -/
theorem geometric_progression_proof (b : ℕ → ℚ) :
  b 3 = -1 ∧ b 6 = 27/8 →
  (∃ q : ℚ, ∀ n : ℕ, b (n + 1) = b n * q) →
  b 1 = -4/9 ∧ (∀ n : ℕ, b (n + 1) = b n * (-3/2)) := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_proof_l3410_341046


namespace NUMINAMATH_CALUDE_alcohol_solution_percentage_l3410_341025

theorem alcohol_solution_percentage 
  (initial_volume : ℝ) 
  (initial_percentage : ℝ) 
  (added_alcohol : ℝ) 
  (added_water : ℝ) 
  (h1 : initial_volume = 40)
  (h2 : initial_percentage = 5)
  (h3 : added_alcohol = 5.5)
  (h4 : added_water = 4.5) :
  let initial_alcohol := initial_volume * (initial_percentage / 100)
  let new_alcohol := initial_alcohol + added_alcohol
  let new_volume := initial_volume + added_alcohol + added_water
  let new_percentage := (new_alcohol / new_volume) * 100
  new_percentage = 15 := by
sorry

end NUMINAMATH_CALUDE_alcohol_solution_percentage_l3410_341025


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3410_341023

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x : ℝ, x^2 + (a+1)*x + a*b > 0 ↔ x < -1 ∨ x > 4) → a + b = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3410_341023


namespace NUMINAMATH_CALUDE_molecular_weight_correct_l3410_341064

/-- The molecular weight of C6H8O7 in g/mol -/
def molecular_weight : ℝ := 192

/-- The number of moles used in the given condition -/
def given_moles : ℝ := 7

/-- The total weight of the given moles in g -/
def given_total_weight : ℝ := 1344

/-- Theorem: The molecular weight of C6H8O7 is correct given the condition -/
theorem molecular_weight_correct : 
  molecular_weight * given_moles = given_total_weight := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_correct_l3410_341064


namespace NUMINAMATH_CALUDE_square_of_difference_of_roots_l3410_341088

theorem square_of_difference_of_roots (p q : ℝ) : 
  (2 * p^2 + 7 * p - 30 = 0) → 
  (2 * q^2 + 7 * q - 30 = 0) → 
  (p - q)^2 = 289 / 4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_of_roots_l3410_341088


namespace NUMINAMATH_CALUDE_grid_product_theorem_l3410_341014

def grid := Fin 3 → Fin 3 → ℕ

def is_valid_grid (g : grid) : Prop :=
  (∀ i j, g i j ∈ Finset.range 10) ∧
  (∀ i j k, i ≠ j → g i k ≠ g j k) ∧
  (∀ i j k, i ≠ j → g k i ≠ g k j)

def row_product (g : grid) (i : Fin 3) : ℕ :=
  (g i 0) * (g i 1) * (g i 2)

def col_product (g : grid) (j : Fin 3) : ℕ :=
  (g 0 j) * (g 1 j) * (g 2 j)

def all_products_equal (g : grid) (P : ℕ) : Prop :=
  (∀ i : Fin 3, row_product g i = P) ∧
  (∀ j : Fin 3, col_product g j = P)

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem grid_product_theorem :
  ∀ (g : grid) (P : ℕ),
    is_valid_grid g →
    all_products_equal g P →
    P = Nat.sqrt (factorial 9) ∧
    (P = 1998 ∨ P = 2000) :=
by sorry

end NUMINAMATH_CALUDE_grid_product_theorem_l3410_341014


namespace NUMINAMATH_CALUDE_disjunction_false_implies_both_false_l3410_341030

theorem disjunction_false_implies_both_false (p q : Prop) :
  (¬(p ∨ q)) → (¬p ∧ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_disjunction_false_implies_both_false_l3410_341030


namespace NUMINAMATH_CALUDE_parallel_planes_iff_parallel_lines_l3410_341048

/-- Two lines are parallel -/
def parallel_lines (m n : Line) : Prop := sorry

/-- Two planes are parallel -/
def parallel_planes (α β : Plane) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular_line_plane (l : Line) (π : Plane) : Prop := sorry

/-- Two objects are different -/
def different {α : Type*} (a b : α) : Prop := a ≠ b

theorem parallel_planes_iff_parallel_lines 
  (m n : Line) (α β : Plane) 
  (h1 : different m n)
  (h2 : different α β)
  (h3 : perpendicular_line_plane m β)
  (h4 : perpendicular_line_plane n β) :
  parallel_planes α β ↔ parallel_lines m n := by sorry

end NUMINAMATH_CALUDE_parallel_planes_iff_parallel_lines_l3410_341048


namespace NUMINAMATH_CALUDE_positive_rational_function_uniqueness_l3410_341011

/-- A function from positive rationals to positive rationals -/
def PositiveRationalFunction := {f : ℚ → ℚ // ∀ x, 0 < x → 0 < f x}

/-- The property that f(x+1) = f(x) + 1 for all positive rationals x -/
def HasUnitPeriod (f : PositiveRationalFunction) : Prop :=
  ∀ x : ℚ, 0 < x → f.val (x + 1) = f.val x + 1

/-- The property that f(1/x) = 1/f(x) for all positive rationals x -/
def HasInverseProperty (f : PositiveRationalFunction) : Prop :=
  ∀ x : ℚ, 0 < x → f.val (1 / x) = 1 / f.val x

/-- The main theorem: if a function satisfies both properties, it must be the identity function -/
theorem positive_rational_function_uniqueness (f : PositiveRationalFunction) 
    (h1 : HasUnitPeriod f) (h2 : HasInverseProperty f) : 
    ∀ x : ℚ, 0 < x → f.val x = x := by
  sorry

end NUMINAMATH_CALUDE_positive_rational_function_uniqueness_l3410_341011


namespace NUMINAMATH_CALUDE_boys_joining_group_l3410_341041

theorem boys_joining_group (total : ℕ) (initial_boys : ℕ) (initial_girls : ℕ) (boys_joining : ℕ) :
  total = 48 →
  initial_boys + initial_girls = total →
  initial_boys * 5 = initial_girls * 3 →
  (initial_boys + boys_joining) * 3 = initial_girls * 5 →
  boys_joining = 32 := by
sorry

end NUMINAMATH_CALUDE_boys_joining_group_l3410_341041


namespace NUMINAMATH_CALUDE_product_evaluation_l3410_341012

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l3410_341012


namespace NUMINAMATH_CALUDE_inequality_range_l3410_341093

theorem inequality_range (a b x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ a b, |2*a - b| + |a + b| ≥ |a| * (|x - 1| + |x + 1|)) →
  x ∈ Set.Icc (-3/2) (3/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l3410_341093


namespace NUMINAMATH_CALUDE_no_integer_solution_l3410_341095

theorem no_integer_solution :
  ¬ ∃ (x y : ℤ), 15 * x^2 - 7 * y^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3410_341095


namespace NUMINAMATH_CALUDE_triangle_angle_determinant_zero_l3410_341000

theorem triangle_angle_determinant_zero (A B C : ℝ) (h : A + B + C = π) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![![Real.cos A ^ 2, Real.tan A, 1],
                                        ![Real.cos B ^ 2, Real.tan B, 1],
                                        ![Real.cos C ^ 2, Real.tan C, 1]]
  Matrix.det M = 0 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_determinant_zero_l3410_341000


namespace NUMINAMATH_CALUDE_divisibility_for_odd_n_l3410_341099

theorem divisibility_for_odd_n (n : ℕ) (h : Odd n) :
  ∃ k : ℤ, (82 : ℤ)^n + 454 * (69 : ℤ)^n = 1963 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_for_odd_n_l3410_341099


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3410_341053

/-- The function representing the given inequality -/
def f (k x : ℝ) : ℝ := ((k^2 + 6*k + 14)*x - 9) * ((k^2 + 28)*x - 2*k^2 - 12*k)

/-- The solution set M for the inequality f(k, x) < 0 -/
def M (k : ℝ) : Set ℝ := {x | f k x < 0}

/-- The statement of the problem -/
theorem inequality_solution_set (k : ℝ) : 
  (M k ∩ Set.range (Int.cast : ℤ → ℝ) = {1}) ↔ (k < -14 ∨ (2 < k ∧ k ≤ 14/3)) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3410_341053


namespace NUMINAMATH_CALUDE_speed_ratio_and_distance_l3410_341009

/-- Represents a traveler with a constant speed -/
structure Traveler where
  speed : ℝ
  startPosition : ℝ

/-- Represents the problem setup -/
structure TravelProblem where
  A : Traveler
  B : Traveler
  C : Traveler
  distanceAB : ℝ
  timeToCMeetA : ℝ
  timeAMeetsB : ℝ
  BPastMidpoint : ℝ
  CFromA : ℝ

/-- The main theorem that proves the speed ratio and distance -/
theorem speed_ratio_and_distance 
  (p : TravelProblem)
  (h1 : p.A.startPosition = 0)
  (h2 : p.B.startPosition = 0)
  (h3 : p.C.startPosition = p.distanceAB)
  (h4 : p.timeToCMeetA = 20)
  (h5 : p.timeAMeetsB = 10)
  (h6 : p.BPastMidpoint = 105)
  (h7 : p.CFromA = 315)
  : p.A.speed / p.B.speed = 3 ∧ p.distanceAB = 1890 := by
  sorry

#check speed_ratio_and_distance

end NUMINAMATH_CALUDE_speed_ratio_and_distance_l3410_341009


namespace NUMINAMATH_CALUDE_metallic_sheet_length_l3410_341072

/-- Represents the dimensions and properties of a metallic sheet and the box formed from it. -/
structure MetallicSheet where
  width : ℝ
  length : ℝ
  cutSize : ℝ
  boxVolume : ℝ

/-- Theorem stating that given the conditions of the problem, the length of the sheet is 48 m. -/
theorem metallic_sheet_length
  (sheet : MetallicSheet)
  (h1 : sheet.width = 36)
  (h2 : sheet.cutSize = 8)
  (h3 : sheet.boxVolume = 5120)
  (h4 : sheet.boxVolume = (sheet.length - 2 * sheet.cutSize) * (sheet.width - 2 * sheet.cutSize) * sheet.cutSize) :
  sheet.length = 48 := by
  sorry

#check metallic_sheet_length

end NUMINAMATH_CALUDE_metallic_sheet_length_l3410_341072


namespace NUMINAMATH_CALUDE_point_inside_circle_l3410_341074

/-- A circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ

/-- A point P at distance d from the center of the circle -/
structure Point where
  P : ℝ × ℝ
  d : ℝ

/-- Definition of a point being inside a circle -/
def is_inside (c : Circle) (p : Point) : Prop :=
  p.d < c.r

/-- Theorem: If the distance from a point to the center of a circle
    is less than the radius, then the point is inside the circle -/
theorem point_inside_circle (c : Circle) (p : Point) 
    (h : p.d < c.r) : is_inside c p := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_l3410_341074


namespace NUMINAMATH_CALUDE_max_x_value_l3410_341018

theorem max_x_value (x y z : ℝ) (sum_eq : x + y + z = 10) (prod_eq : x*y + x*z + y*z = 20) :
  x ≤ 10/3 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ + y₀ + z₀ = 10 ∧ x₀*y₀ + x₀*z₀ + y₀*z₀ = 20 ∧ x₀ = 10/3 :=
by sorry

end NUMINAMATH_CALUDE_max_x_value_l3410_341018


namespace NUMINAMATH_CALUDE_chrysanthemum_arrangement_count_l3410_341077

/-- Represents the number of pots for each color of chrysanthemum -/
structure ChrysanthemumCounts where
  white : Nat
  yellow : Nat
  red : Nat

/-- Calculates the number of arrangements for chrysanthemums with given conditions -/
def chrysanthemumArrangements (counts : ChrysanthemumCounts) : Nat :=
  sorry

/-- The main theorem stating the number of arrangements for the given problem -/
theorem chrysanthemum_arrangement_count :
  let counts : ChrysanthemumCounts := { white := 2, yellow := 2, red := 1 }
  chrysanthemumArrangements counts = 16 := by
  sorry

end NUMINAMATH_CALUDE_chrysanthemum_arrangement_count_l3410_341077


namespace NUMINAMATH_CALUDE_no_prime_pair_divisibility_l3410_341050

theorem no_prime_pair_divisibility : ¬∃ (p q : ℕ), Prime p ∧ Prime q ∧ p > 5 ∧ q > 5 ∧ (p * q ∣ (5^p - 2^p) * (5^q - 2^q)) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_pair_divisibility_l3410_341050


namespace NUMINAMATH_CALUDE_square_areas_and_perimeters_l3410_341084

theorem square_areas_and_perimeters (x : ℝ) : 
  (x^2 + 4*x + 4) > 0 ∧ 
  (4*x^2 - 12*x + 9) > 0 ∧ 
  4 * (x + 2) + 4 * (2*x - 3) = 32 → 
  x = 3 := by sorry

end NUMINAMATH_CALUDE_square_areas_and_perimeters_l3410_341084


namespace NUMINAMATH_CALUDE_min_throws_for_repeated_sum_l3410_341071

/-- The minimum number of distinct sums possible when rolling three six-sided dice -/
def distinct_sums : ℕ := 16

/-- The minimum number of throws needed to guarantee a repeated sum -/
def min_throws : ℕ := distinct_sums + 1

/-- Theorem stating that the minimum number of throws to ensure a repeated sum is 17 -/
theorem min_throws_for_repeated_sum :
  min_throws = 17 := by sorry

end NUMINAMATH_CALUDE_min_throws_for_repeated_sum_l3410_341071


namespace NUMINAMATH_CALUDE_same_lunch_group_probability_l3410_341032

/-- The number of students in the school -/
def total_students : ℕ := 900

/-- The number of lunch groups -/
def num_groups : ℕ := 6

/-- The number of students in each lunch group -/
def students_per_group : ℕ := total_students / num_groups

/-- The probability of a single student being assigned to a specific group -/
def prob_single_student : ℚ := 1 / num_groups

/-- The number of specific students we're interested in -/
def num_specific_students : ℕ := 4

theorem same_lunch_group_probability :
  (prob_single_student ^ (num_specific_students - 1) : ℚ) = 1 / 216 :=
sorry

end NUMINAMATH_CALUDE_same_lunch_group_probability_l3410_341032


namespace NUMINAMATH_CALUDE_counterexample_21_l3410_341022

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem counterexample_21 :
  ¬(is_prime 21) ∧ ¬(is_prime (21 + 3)) :=
by sorry

end NUMINAMATH_CALUDE_counterexample_21_l3410_341022


namespace NUMINAMATH_CALUDE_trefoil_cases_l3410_341087

theorem trefoil_cases (total_boxes : ℕ) (boxes_per_case : ℕ) (h1 : total_boxes = 24) (h2 : boxes_per_case = 8) :
  total_boxes / boxes_per_case = 3 := by
  sorry

end NUMINAMATH_CALUDE_trefoil_cases_l3410_341087


namespace NUMINAMATH_CALUDE_cobys_road_trip_l3410_341073

/-- Coby's road trip problem -/
theorem cobys_road_trip 
  (distance_to_idaho : ℝ) 
  (distance_from_idaho : ℝ) 
  (speed_from_idaho : ℝ) 
  (total_time : ℝ) 
  (h1 : distance_to_idaho = 640)
  (h2 : distance_from_idaho = 550)
  (h3 : speed_from_idaho = 50)
  (h4 : total_time = 19) :
  let time_from_idaho := distance_from_idaho / speed_from_idaho
  let time_to_idaho := total_time - time_from_idaho
  distance_to_idaho / time_to_idaho = 80 := by
sorry


end NUMINAMATH_CALUDE_cobys_road_trip_l3410_341073


namespace NUMINAMATH_CALUDE_product_equals_120_l3410_341045

theorem product_equals_120 (n : ℕ) (h : n = 3) : 
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_120_l3410_341045


namespace NUMINAMATH_CALUDE_prime_divisors_condition_l3410_341007

theorem prime_divisors_condition (a n : ℕ) (ha : a > 2) :
  (∀ p : ℕ, Nat.Prime p → p ∣ (a^n - 1) → p ∣ (a^(3^2016) - 1)) →
  ∃ l : ℕ, l > 0 ∧ a = 2^l - 1 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_divisors_condition_l3410_341007


namespace NUMINAMATH_CALUDE_solve_equation_l3410_341067

theorem solve_equation (x : ℚ) : 
  (x - 30) / 2 = (5 - 3*x) / 6 + 2 → x = 167/6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3410_341067


namespace NUMINAMATH_CALUDE_min_stamps_for_50_cents_l3410_341047

/-- Represents the number of stamps needed to make a certain value -/
structure StampCombination :=
  (fives : Nat) -- number of 5-cent stamps
  (fours : Nat) -- number of 4-cent stamps

/-- Calculates the total value of stamps in cents -/
def totalValue (sc : StampCombination) : Nat :=
  5 * sc.fives + 4 * sc.fours

/-- Calculates the total number of stamps -/
def totalStamps (sc : StampCombination) : Nat :=
  sc.fives + sc.fours

/-- Checks if a StampCombination is valid (i.e., totals 50 cents) -/
def isValid (sc : StampCombination) : Prop :=
  totalValue sc = 50

/-- Theorem: The minimum number of stamps needed to make 50 cents 
    using only 5-cent and 4-cent stamps is 11 -/
theorem min_stamps_for_50_cents :
  ∃ (sc : StampCombination), 
    isValid sc ∧ 
    totalStamps sc = 11 ∧ 
    (∀ (sc' : StampCombination), isValid sc' → totalStamps sc' ≥ totalStamps sc) :=
  sorry

end NUMINAMATH_CALUDE_min_stamps_for_50_cents_l3410_341047


namespace NUMINAMATH_CALUDE_expression_equals_one_l3410_341024

theorem expression_equals_one (x z : ℝ) (h1 : x ≠ z) (h2 : x ≠ -z) :
  (x / (x - z) - z / (x + z)) / (z / (x - z) + x / (x + z)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l3410_341024


namespace NUMINAMATH_CALUDE_cos_squared_plus_sin_double_l3410_341090

/-- If the terminal side of angle α passes through point P(2, 1) in the Cartesian coordinate system, then cos²α + sin(2α) = 8/5 -/
theorem cos_squared_plus_sin_double (α : Real) :
  (∃ (x y : Real), x = 2 ∧ y = 1 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.cos α ^ 2 + Real.sin (2 * α) = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_plus_sin_double_l3410_341090


namespace NUMINAMATH_CALUDE_reach_11_from_1_l3410_341085

/-- Represents the set of operations allowed by the calculator -/
inductive CalcOp
  | mul3 : CalcOp  -- Multiply by 3
  | add3 : CalcOp  -- Add 3
  | div3 : CalcOp  -- Divide by 3 (when divisible)

/-- Applies a single calculator operation to a number -/
def applyOp (n : ℕ) (op : CalcOp) : ℕ :=
  match op with
  | CalcOp.mul3 => n * 3
  | CalcOp.add3 => n + 3
  | CalcOp.div3 => if n % 3 = 0 then n / 3 else n

/-- Checks if it's possible to reach the target number from the start number using the given operations -/
def canReach (start target : ℕ) : Prop :=
  ∃ (steps : List CalcOp), (steps.foldl applyOp start) = target

/-- Theorem stating that it's possible to reach 11 from 1 using the calculator operations -/
theorem reach_11_from_1 : canReach 1 11 := by
  sorry

end NUMINAMATH_CALUDE_reach_11_from_1_l3410_341085


namespace NUMINAMATH_CALUDE_tan_product_thirty_degrees_l3410_341004

theorem tan_product_thirty_degrees :
  let A : Real := 30 * π / 180
  let B : Real := 30 * π / 180
  (1 + Real.tan A) * (1 + Real.tan B) = (4 + 2 * Real.sqrt 3) / 3 := by
sorry

end NUMINAMATH_CALUDE_tan_product_thirty_degrees_l3410_341004


namespace NUMINAMATH_CALUDE_ellipse_condition_l3410_341042

def ellipse_equation (x y k : ℝ) : Prop :=
  3 * x^2 + 6 * y^2 - 12 * x + 18 * y = k

def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  ∃ (a b h c d : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (x y : ℝ), ellipse_equation x y k ↔ (x - h)^2 / a^2 + (y - d)^2 / b^2 = 1

theorem ellipse_condition (k : ℝ) :
  is_non_degenerate_ellipse k ↔ k > -51/2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l3410_341042


namespace NUMINAMATH_CALUDE_total_new_games_l3410_341063

/-- Given Katie's and her friends' game collections, prove the total number of new games they have together. -/
theorem total_new_games 
  (katie_new : ℕ) 
  (katie_percent : ℚ) 
  (friends_new : ℕ) 
  (friends_percent : ℚ) 
  (h1 : katie_new = 84) 
  (h2 : katie_percent = 75 / 100) 
  (h3 : friends_new = 8) 
  (h4 : friends_percent = 10 / 100) : 
  katie_new + friends_new = 92 := by
  sorry

end NUMINAMATH_CALUDE_total_new_games_l3410_341063


namespace NUMINAMATH_CALUDE_james_night_out_cost_l3410_341061

/-- Calculate the total amount James spent for a night out -/
theorem james_night_out_cost : 
  let entry_fee : ℚ := 25
  let friends_count : ℕ := 8
  let rounds_count : ℕ := 3
  let james_drinks_count : ℕ := 7
  let cocktail_price : ℚ := 8
  let non_alcoholic_price : ℚ := 4
  let james_cocktails_count : ℕ := 6
  let burger_price : ℚ := 18
  let tip_percentage : ℚ := 0.25

  let friends_drinks_cost := friends_count * rounds_count * cocktail_price
  let james_drinks_cost := james_cocktails_count * cocktail_price + 
                           (james_drinks_count - james_cocktails_count) * non_alcoholic_price
  let food_cost := burger_price
  let subtotal := entry_fee + friends_drinks_cost + james_drinks_cost + food_cost
  let tip := subtotal * tip_percentage
  let total_cost := subtotal + tip

  total_cost = 358.75 := by sorry

end NUMINAMATH_CALUDE_james_night_out_cost_l3410_341061


namespace NUMINAMATH_CALUDE_cat_litter_weight_l3410_341069

/-- Calculates the weight of cat litter in each container given the problem conditions. -/
theorem cat_litter_weight 
  (container_price : ℝ) 
  (litter_box_capacity : ℝ) 
  (total_cost : ℝ) 
  (total_days : ℝ) 
  (h1 : container_price = 21)
  (h2 : litter_box_capacity = 15)
  (h3 : total_cost = 210)
  (h4 : total_days = 210) :
  (total_cost * litter_box_capacity) / (container_price * total_days / 7) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cat_litter_weight_l3410_341069


namespace NUMINAMATH_CALUDE_product_xyz_l3410_341013

theorem product_xyz (x y z : ℚ) 
  (eq1 : 3 * x + 4 * y = 60)
  (eq2 : 6 * x - 4 * y = 12)
  (eq3 : 2 * x - 3 * z = 38) :
  x * y * z = -1584 / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_xyz_l3410_341013


namespace NUMINAMATH_CALUDE_red_non_honda_percentage_l3410_341005

/-- Calculates the percentage of red non-Honda cars in Chennai --/
theorem red_non_honda_percentage
  (total_cars : ℕ) 
  (honda_cars : ℕ) 
  (red_honda_ratio : ℚ) 
  (total_red_ratio : ℚ) 
  (h1 : total_cars = 900)
  (h2 : honda_cars = 500)
  (h3 : red_honda_ratio = 90 / 100)
  (h4 : total_red_ratio = 60 / 100) :
  (total_red_ratio * total_cars - red_honda_ratio * honda_cars) / (total_cars - honda_cars) = 9 / 40 :=
by
  sorry

#eval (9 : ℚ) / 40 -- This should evaluate to 0.225 or 22.5%

end NUMINAMATH_CALUDE_red_non_honda_percentage_l3410_341005


namespace NUMINAMATH_CALUDE_triangle_minimum_perimeter_l3410_341078

/-- Given a triangle with sides a, b, c, semiperimeter p, and area S,
    the perimeter is minimized when the triangle is equilateral. -/
theorem triangle_minimum_perimeter 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (p : ℝ) (hp : p = (a + b + c) / 2)
  (S : ℝ) (hS : S > 0)
  (harea : S^2 = p * (p - a) * (p - b) * (p - c)) :
  ∃ (min_a min_b min_c : ℝ),
    min_a = min_b ∧ min_b = min_c ∧
    min_a + min_b + min_c ≤ a + b + c ∧
    (min_a + min_b + min_c) / 2 * ((min_a + min_b + min_c) / 2 - min_a) * 
    ((min_a + min_b + min_c) / 2 - min_b) * ((min_a + min_b + min_c) / 2 - min_c) = S^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_minimum_perimeter_l3410_341078


namespace NUMINAMATH_CALUDE_inequality_proof_l3410_341076

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  a * b > a * c :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3410_341076


namespace NUMINAMATH_CALUDE_solution_of_system_l3410_341057

/-- The system of equations:
    1. xy + 5yz - 6xz = -2z
    2. 2xy + 9yz - 9xz = -12z
    3. yz - 2xz = 6z
-/
def system_of_equations (x y z : ℝ) : Prop :=
  x*y + 5*y*z - 6*x*z = -2*z ∧
  2*x*y + 9*y*z - 9*x*z = -12*z ∧
  y*z - 2*x*z = 6*z

theorem solution_of_system :
  (∃ (x y z : ℝ), system_of_equations x y z ∧ (x = -2 ∧ y = 2 ∧ z = 1/6)) ∧
  (∀ (x : ℝ), system_of_equations x 0 0) ∧
  (∀ (y : ℝ), system_of_equations 0 y 0) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_system_l3410_341057


namespace NUMINAMATH_CALUDE_inequality_implication_l3410_341038

theorem inequality_implication (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 2 * (3^(2*x)) - 3^x + a^2 - a - 3 > 0) → 
  a < -1 ∨ a > 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_implication_l3410_341038


namespace NUMINAMATH_CALUDE_prob_sum_less_than_10_l3410_341066

/-- A fair die with faces labeled 1 to 6 -/
def FairDie : Finset ℕ := Finset.range 6

/-- The sample space of rolling a fair die twice -/
def SampleSpace : Finset (ℕ × ℕ) :=
  FairDie.product FairDie

/-- The event where the sum of face values is less than 10 -/
def EventSumLessThan10 : Finset (ℕ × ℕ) :=
  SampleSpace.filter (fun (x, y) => x + y < 10)

/-- Theorem: The probability of the sum of face values being less than 10
    when rolling a fair six-sided die twice is 5/6 -/
theorem prob_sum_less_than_10 :
  (EventSumLessThan10.card : ℚ) / SampleSpace.card = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_less_than_10_l3410_341066


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3410_341080

/-- Given an ellipse and a hyperbola with common foci and specified eccentricity,
    prove the equation of the hyperbola. -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ a b : ℝ, a^2 = 12 ∧ b^2 = 3 ∧ x^2 / a^2 + y^2 / b^2 = 1) →  -- Ellipse equation
  (∃ c : ℝ, c^2 = 12 - 3) →                                    -- Foci distance
  (∃ e : ℝ, e = 3/2) →                                         -- Hyperbola eccentricity
  x^2 / 4 - y^2 / 5 = 1                                        -- Hyperbola equation
:= by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3410_341080


namespace NUMINAMATH_CALUDE_amber_guppies_problem_l3410_341034

/-- The number of guppies Amber initially bought -/
def initial_guppies : ℕ := 7

/-- The number of baby guppies Amber saw in the first sighting (3 dozen) -/
def first_sighting : ℕ := 36

/-- The total number of guppies Amber has after the second sighting -/
def total_guppies : ℕ := 52

/-- The number of additional baby guppies Amber saw two days after the first sighting -/
def additional_guppies : ℕ := total_guppies - (initial_guppies + first_sighting)

theorem amber_guppies_problem :
  additional_guppies = 9 := by sorry

end NUMINAMATH_CALUDE_amber_guppies_problem_l3410_341034


namespace NUMINAMATH_CALUDE_complement_not_always_smaller_than_supplement_l3410_341062

theorem complement_not_always_smaller_than_supplement :
  ¬ (∀ θ : Real, 0 < θ ∧ θ < π → (π / 2 - θ) < (π - θ)) := by
  sorry

end NUMINAMATH_CALUDE_complement_not_always_smaller_than_supplement_l3410_341062


namespace NUMINAMATH_CALUDE_sqrt_diff_approx_three_l3410_341055

theorem sqrt_diff_approx_three (k : ℕ) (h : k ≥ 7) :
  |Real.sqrt (9 * (k + 1)^2 + (k + 1)) - Real.sqrt (9 * k^2 + k) - 3| < (1 : ℝ) / 1000 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_diff_approx_three_l3410_341055
