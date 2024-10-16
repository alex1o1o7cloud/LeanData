import Mathlib

namespace NUMINAMATH_CALUDE_cookie_jar_remaining_l3471_347110

/-- The amount left in the cookie jar after Doris and Martha's spending -/
theorem cookie_jar_remaining (initial_amount : ℕ) (doris_spent : ℕ) (martha_spent : ℕ) : 
  initial_amount = 21 → 
  doris_spent = 6 → 
  martha_spent = doris_spent / 2 → 
  initial_amount - (doris_spent + martha_spent) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_jar_remaining_l3471_347110


namespace NUMINAMATH_CALUDE_exponent_simplification_l3471_347117

theorem exponent_simplification :
  (5^6 * 5^9 * 5) / 5^3 = 5^13 := by sorry

end NUMINAMATH_CALUDE_exponent_simplification_l3471_347117


namespace NUMINAMATH_CALUDE_log4_one_sixteenth_eq_neg_two_l3471_347175

-- Define the logarithm function for base 4
noncomputable def log4 (x : ℝ) : ℝ := Real.log x / Real.log 4

-- Theorem statement
theorem log4_one_sixteenth_eq_neg_two : log4 (1/16) = -2 := by
  sorry

end NUMINAMATH_CALUDE_log4_one_sixteenth_eq_neg_two_l3471_347175


namespace NUMINAMATH_CALUDE_min_area_ratio_l3471_347131

-- Define the triangles
structure EquilateralTriangle :=
  (A B C : ℝ × ℝ)

structure RightTriangle :=
  (D E F : ℝ × ℝ)

-- Define the conditions
def inscribed (rt : RightTriangle) (et : EquilateralTriangle) : Prop :=
  sorry

def right_angle (rt : RightTriangle) : Prop :=
  sorry

def angle_edf_30 (rt : RightTriangle) : Prop :=
  sorry

-- Define the area ratio
def area_ratio (rt : RightTriangle) (et : EquilateralTriangle) : ℝ :=
  sorry

-- Theorem statement
theorem min_area_ratio 
  (et : EquilateralTriangle) 
  (rt : RightTriangle) 
  (h1 : inscribed rt et) 
  (h2 : right_angle rt) 
  (h3 : angle_edf_30 rt) :
  ∃ (min_ratio : ℝ), 
    (∀ (rt' : RightTriangle), inscribed rt' et → right_angle rt' → angle_edf_30 rt' → 
      area_ratio rt' et ≥ min_ratio) ∧ 
    min_ratio = 3/14 :=
  sorry

end NUMINAMATH_CALUDE_min_area_ratio_l3471_347131


namespace NUMINAMATH_CALUDE_problem_statement_l3471_347191

/-- Given real numbers x, y, and z satisfying certain conditions, 
    prove that a specific expression equals 13.5 -/
theorem problem_statement (x y z : ℝ) 
  (h1 : x*z/(x+y) + y*x/(y+z) + z*y/(z+x) = -9)
  (h2 : y*z/(x+y) + z*x/(y+z) + x*y/(z+x) = 15) :
  y/(x+y) + z/(y+z) + x/(z+x) = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3471_347191


namespace NUMINAMATH_CALUDE_perpendicular_line_plane_condition_l3471_347100

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_line_plane_condition
  (m n : Line) (α β : Plane)
  (h1 : parallel m n)
  (h2 : perpendicular_line_plane n β)
  (h3 : parallel_plane α β) :
  perpendicular_line_plane m α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_plane_condition_l3471_347100


namespace NUMINAMATH_CALUDE_prob_at_least_one_one_value_l3471_347185

/-- The probability of rolling a specific number on a fair six-sided die -/
def prob_single : ℚ := 1 / 6

/-- The probability of not rolling a specific number on a fair six-sided die -/
def prob_not_single : ℚ := 1 - prob_single

/-- The probability of at least one die showing 1 when two fair six-sided dice are rolled once -/
def prob_at_least_one_one : ℚ := 
  prob_single * prob_not_single + 
  prob_not_single * prob_single + 
  prob_single * prob_single

theorem prob_at_least_one_one_value : prob_at_least_one_one = 11 / 36 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_one_value_l3471_347185


namespace NUMINAMATH_CALUDE_simplify_expression_l3471_347101

theorem simplify_expression : (2^8 + 4^5) * (2^3 - (-2)^3)^2 = 327680 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3471_347101


namespace NUMINAMATH_CALUDE_distance_propositions_l3471_347114

-- Define the distance measure
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := |x₂ - x₁| + |y₂ - y₁|

-- Define propositions
def proposition1 (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (x ∈ Set.Icc x₁ x₂ ∧ y ∈ Set.Icc y₁ y₂) →
  distance x₁ y₁ x y + distance x y x₂ y₂ = distance x₁ y₁ x₂ y₂

def proposition2 (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (x - x₁) * (x₂ - x) + (y - y₁) * (y₂ - y) = 0 →
  (distance x₁ y₁ x y)^2 + (distance x y x₂ y₂)^2 = (distance x₁ y₁ x₂ y₂)^2

def proposition3 (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  distance x₁ y₁ x y + distance x y x₂ y₂ > distance x₁ y₁ x₂ y₂

-- Theorem statement
theorem distance_propositions :
  (∀ x₁ y₁ x₂ y₂ x y, proposition1 x₁ y₁ x₂ y₂ x y) ∧
  (∃ x₁ y₁ x₂ y₂ x y, ¬proposition2 x₁ y₁ x₂ y₂ x y) ∧
  (∃ x₁ y₁ x₂ y₂ x y, ¬proposition3 x₁ y₁ x₂ y₂ x y) :=
sorry

end NUMINAMATH_CALUDE_distance_propositions_l3471_347114


namespace NUMINAMATH_CALUDE_inequality_multiplication_l3471_347102

theorem inequality_multiplication (m n : ℝ) (h : m > n) : 2 * m > 2 * n := by
  sorry

end NUMINAMATH_CALUDE_inequality_multiplication_l3471_347102


namespace NUMINAMATH_CALUDE_initial_speed_is_40_l3471_347145

/-- A person's journey with varying speeds -/
def Journey (D T : ℝ) (initial_speed final_speed : ℝ) : Prop :=
  initial_speed > 0 ∧ final_speed > 0 ∧ D > 0 ∧ T > 0 ∧
  (2/3 * D) / (1/3 * T) = initial_speed ∧
  (1/3 * D) / (1/3 * T) = final_speed

/-- Theorem: Given the conditions, the initial speed is 40 kmph -/
theorem initial_speed_is_40 (D T : ℝ) :
  Journey D T initial_speed 20 → initial_speed = 40 := by
  sorry

#check initial_speed_is_40

end NUMINAMATH_CALUDE_initial_speed_is_40_l3471_347145


namespace NUMINAMATH_CALUDE_exponential_inequality_l3471_347152

theorem exponential_inequality (m n : ℝ) : (0.2 : ℝ) ^ m < (0.2 : ℝ) ^ n → m > n := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l3471_347152


namespace NUMINAMATH_CALUDE_division_problem_l3471_347147

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2/5) : 
  c / a = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3471_347147


namespace NUMINAMATH_CALUDE_angle_problem_l3471_347188

theorem angle_problem (x : ℝ) : 
  x + (3 * x - 10) = 180 → x = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_problem_l3471_347188


namespace NUMINAMATH_CALUDE_money_division_l3471_347111

theorem money_division (p q r : ℕ) (total : ℚ) :
  p + q + r = 22 →
  3 * total / 22 = p →
  7 * total / 22 = q →
  12 * total / 22 = r →
  q - p = 4400 →
  r - q = 5500 := by
sorry

end NUMINAMATH_CALUDE_money_division_l3471_347111


namespace NUMINAMATH_CALUDE_sum_of_numbers_l3471_347119

theorem sum_of_numbers : 4321 + 3214 + 2143 - 1432 = 8246 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l3471_347119


namespace NUMINAMATH_CALUDE_baker_remaining_cakes_l3471_347107

/-- The number of cakes initially made by the baker -/
def initial_cakes : ℕ := 149

/-- The number of cakes sold by the baker -/
def sold_cakes : ℕ := 10

/-- The number of cakes the baker still has -/
def remaining_cakes : ℕ := initial_cakes - sold_cakes

theorem baker_remaining_cakes : remaining_cakes = 139 := by
  sorry

end NUMINAMATH_CALUDE_baker_remaining_cakes_l3471_347107


namespace NUMINAMATH_CALUDE_trousers_final_cost_l3471_347172

def calculate_final_cost (original_price : ℝ) (in_store_discount : ℝ) (additional_promotion : ℝ) (sales_tax : ℝ) (handling_fee : ℝ) : ℝ :=
  let price_after_in_store_discount := original_price * (1 - in_store_discount)
  let price_after_additional_promotion := price_after_in_store_discount * (1 - additional_promotion)
  let price_with_tax := price_after_additional_promotion * (1 + sales_tax)
  price_with_tax + handling_fee

theorem trousers_final_cost :
  calculate_final_cost 100 0.20 0.10 0.05 5 = 80.60 := by
  sorry

end NUMINAMATH_CALUDE_trousers_final_cost_l3471_347172


namespace NUMINAMATH_CALUDE_hydras_always_live_l3471_347134

/-- Represents the number of new heads a hydra can grow in a week -/
inductive NewHeads
  | five : NewHeads
  | seven : NewHeads

/-- The state of the hydras after a certain number of weeks -/
structure HydraState where
  weeks : ℕ
  totalHeads : ℕ

/-- The initial state of the hydras -/
def initialState : HydraState :=
  { weeks := 0, totalHeads := 2016 + 2017 }

/-- The change in total heads after one week -/
def weeklyChange (a b : NewHeads) : ℕ :=
  match a, b with
  | NewHeads.five, NewHeads.five => 6
  | NewHeads.five, NewHeads.seven => 8
  | NewHeads.seven, NewHeads.five => 8
  | NewHeads.seven, NewHeads.seven => 10

/-- The state transition function -/
def nextState (state : HydraState) (a b : NewHeads) : HydraState :=
  { weeks := state.weeks + 1
  , totalHeads := state.totalHeads + weeklyChange a b }

theorem hydras_always_live :
  ∀ (state : HydraState), state.totalHeads % 2 = 1 →
    ∀ (a b : NewHeads), (nextState state a b).totalHeads % 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_hydras_always_live_l3471_347134


namespace NUMINAMATH_CALUDE_square_sum_of_sqrt3_plus_minus_2_l3471_347112

theorem square_sum_of_sqrt3_plus_minus_2 :
  let a : ℝ := Real.sqrt 3 + 2
  let b : ℝ := Real.sqrt 3 - 2
  a^2 + b^2 = 14 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_sqrt3_plus_minus_2_l3471_347112


namespace NUMINAMATH_CALUDE_expression_value_at_1580_l3471_347159

theorem expression_value_at_1580 : 
  let a : ℝ := 1580
  let expr := 2*a - (((2*a - 3)/(a + 1)) - ((a + 1)/(2 - 2*a)) - ((a^2 + 3)/(2*a^(2-2)))) * ((a^3 + 1)/(a^2 - a)) + 2/a
  expr = 2 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_1580_l3471_347159


namespace NUMINAMATH_CALUDE_four_leaf_area_l3471_347130

/-- The area of a four-leaf shape formed by semicircles drawn on each side of a square -/
theorem four_leaf_area (a : ℝ) (h : a > 0) :
  let square_side := a
  let semicircle_radius := a / 2
  let leaf_area := π * semicircle_radius^2 / 2 - square_side^2 / 4
  4 * leaf_area = a^2 / 2 * (π - 2) :=
by sorry

end NUMINAMATH_CALUDE_four_leaf_area_l3471_347130


namespace NUMINAMATH_CALUDE_major_axis_endpoints_of_ellipse_l3471_347176

/-- An ellipse is defined by the equation 6x^2 + y^2 = 36 -/
def ellipse (x y : ℝ) : Prop := 6 * x^2 + y^2 = 36

/-- The endpoints of the major axis of the ellipse -/
def major_axis_endpoints : Set (ℝ × ℝ) := {(-6, 0), (6, 0)}

/-- Theorem: The coordinates of the endpoints of the major axis of the ellipse 6x^2 + y^2 = 36 are (0, -6) and (0, 6) -/
theorem major_axis_endpoints_of_ellipse :
  major_axis_endpoints = {(0, -6), (0, 6)} :=
sorry

end NUMINAMATH_CALUDE_major_axis_endpoints_of_ellipse_l3471_347176


namespace NUMINAMATH_CALUDE_expand_product_l3471_347103

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3471_347103


namespace NUMINAMATH_CALUDE_rental_cost_calculation_l3471_347123

def base_daily_rate : ℚ := 30
def per_mile_rate : ℚ := 0.25
def discount_rate : ℚ := 0.1
def discount_threshold : ℕ := 5
def rental_days : ℕ := 6
def miles_driven : ℕ := 500

def calculate_total_cost : ℚ :=
  let daily_cost := if rental_days > discount_threshold
                    then base_daily_rate * (1 - discount_rate) * rental_days
                    else base_daily_rate * rental_days
  let mileage_cost := per_mile_rate * miles_driven
  daily_cost + mileage_cost

theorem rental_cost_calculation :
  calculate_total_cost = 287 :=
by sorry

end NUMINAMATH_CALUDE_rental_cost_calculation_l3471_347123


namespace NUMINAMATH_CALUDE_alex_coin_distribution_l3471_347120

/-- The minimum number of additional coins needed -/
def min_additional_coins (friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (friends * (friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed for Alex -/
theorem alex_coin_distribution (friends : ℕ) (initial_coins : ℕ) 
  (h1 : friends = 15) (h2 : initial_coins = 100) : 
  min_additional_coins friends initial_coins = 20 := by
  sorry

end NUMINAMATH_CALUDE_alex_coin_distribution_l3471_347120


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l3471_347148

/-- Given two perpendicular lines with direction vectors (4, -5) and (b, 8), prove that b = 10 -/
theorem perpendicular_lines_b_value :
  let v1 : Fin 2 → ℝ := ![4, -5]
  let v2 : Fin 2 → ℝ := ![b, 8]
  (∀ (i j : Fin 2), i ≠ j → v1 i * v2 i + v1 j * v2 j = 0) →
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l3471_347148


namespace NUMINAMATH_CALUDE_power_of_power_three_l3471_347164

theorem power_of_power_three : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_l3471_347164


namespace NUMINAMATH_CALUDE_bike_race_distance_difference_l3471_347155

theorem bike_race_distance_difference 
  (race_duration : ℝ) 
  (clara_speed : ℝ) 
  (denise_speed : ℝ) 
  (h1 : race_duration = 5)
  (h2 : clara_speed = 18)
  (h3 : denise_speed = 16) :
  clara_speed * race_duration - denise_speed * race_duration = 10 := by
sorry

end NUMINAMATH_CALUDE_bike_race_distance_difference_l3471_347155


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3471_347167

theorem possible_values_of_a (x y a : ℝ) :
  (|3 * y - 18| + |a * x - y| = 0) →
  (x > 0) →
  (∃ n : ℕ, x = 2 * n) →
  (x ≤ y) →
  (a = 3 ∨ a = 3/2 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3471_347167


namespace NUMINAMATH_CALUDE_circle_equation_l3471_347182

/-- A circle with center (a, b) and radius r -/
structure Circle where
  a : ℝ
  b : ℝ
  r : ℝ

/-- The standard equation of a circle -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.a)^2 + (y - c.b)^2 = c.r^2

/-- A circle is tangent to the x-axis -/
def Circle.tangentToXAxis (c : Circle) : Prop :=
  c.b = c.r

/-- The center of the circle lies on the line y = 3x -/
def Circle.centerOnLine (c : Circle) : Prop :=
  c.b = 3 * c.a

/-- The chord intercepted by the circle on the line y = x has length 2√7 -/
def Circle.chordLength (c : Circle) : Prop :=
  2 * c.r^2 = (c.a - c.b)^2 + 14

/-- The main theorem -/
theorem circle_equation (c : Circle) 
  (h1 : c.tangentToXAxis)
  (h2 : c.centerOnLine)
  (h3 : c.chordLength) :
  (c.equation 1 3 ∧ c.r^2 = 9) ∨ (c.equation (-1) 3 ∧ c.r^2 = 9) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3471_347182


namespace NUMINAMATH_CALUDE_cube_with_tunnel_surface_area_l3471_347104

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with a drilled tunnel -/
structure CubeWithTunnel where
  sideLength : ℝ
  E : Point3D
  I : Point3D
  J : Point3D
  K : Point3D

/-- Calculates the total surface area of a cube with a drilled tunnel -/
def totalSurfaceArea (cube : CubeWithTunnel) : ℝ :=
  sorry

/-- Theorem stating the total surface area of the cube with tunnel -/
theorem cube_with_tunnel_surface_area :
  ∀ (cube : CubeWithTunnel),
    cube.sideLength = 10 ∧
    (cube.I.x - cube.E.x)^2 + (cube.I.y - cube.E.y)^2 + (cube.I.z - cube.E.z)^2 = 3^2 ∧
    (cube.J.x - cube.E.x)^2 + (cube.J.y - cube.E.y)^2 + (cube.J.z - cube.E.z)^2 = 3^2 ∧
    (cube.K.x - cube.E.x)^2 + (cube.K.y - cube.E.y)^2 + (cube.K.z - cube.E.z)^2 = 3^2 →
    totalSurfaceArea cube = 630 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_tunnel_surface_area_l3471_347104


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_equals_two_l3471_347133

theorem unique_solution_implies_a_equals_two (a : ℝ) :
  (∃! x : ℝ, 0 ≤ x^2 - a*x + a ∧ x^2 - a*x + a ≤ 1) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_equals_two_l3471_347133


namespace NUMINAMATH_CALUDE_range_of_a_l3471_347144

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then -x + 5 else a^x + 2*a + 2

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ y : ℝ, y ≥ 3 → ∃ x : ℝ, f a x = y) ∧ 
  (∀ x : ℝ, f a x ≥ 3) →
  a ∈ Set.Icc (1/2) 1 ∪ Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3471_347144


namespace NUMINAMATH_CALUDE_inequalities_not_equivalent_l3471_347158

theorem inequalities_not_equivalent : 
  ¬(∀ x : ℝ, (Real.sqrt (x - 1) < Real.sqrt (2 - x)) ↔ (x - 1 < 2 - x)) := by
sorry

end NUMINAMATH_CALUDE_inequalities_not_equivalent_l3471_347158


namespace NUMINAMATH_CALUDE_sheridan_cats_l3471_347170

theorem sheridan_cats (initial : ℝ) (gave_first : ℝ) (received : ℝ) (gave_second : ℝ) 
  (h1 : initial = 17.5)
  (h2 : gave_first = 6.2)
  (h3 : received = 2.8)
  (h4 : gave_second = 1.3) :
  initial - gave_first + received - gave_second = 12.8 := by
sorry

end NUMINAMATH_CALUDE_sheridan_cats_l3471_347170


namespace NUMINAMATH_CALUDE_polynomial_on_unit_circle_l3471_347186

/-- A polynomial p(z) with complex coefficients a and b -/
def p (a b z : ℂ) : ℂ := z^2 + a*z + b

/-- The property that |p(z)| = 1 on the unit circle -/
def unit_circle_property (a b : ℂ) : Prop :=
  ∀ z : ℂ, Complex.abs z = 1 → Complex.abs (p a b z) = 1

/-- Theorem: If |p(z)| = 1 on the unit circle, then a = 0 and b = 0 -/
theorem polynomial_on_unit_circle (a b : ℂ) 
  (h : unit_circle_property a b) : a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_on_unit_circle_l3471_347186


namespace NUMINAMATH_CALUDE_multiplication_equality_l3471_347181

theorem multiplication_equality (x : ℝ) : x * 240 = 173 * 240 ↔ x = 173 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_equality_l3471_347181


namespace NUMINAMATH_CALUDE_simplify_expression_l3471_347146

theorem simplify_expression : 18 * (8 / 15) * (1 / 12) - (1 / 4) = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3471_347146


namespace NUMINAMATH_CALUDE_jessie_current_weight_l3471_347163

def jessie_weight_problem (initial_weight lost_weight : ℕ) : Prop :=
  initial_weight = 69 ∧ lost_weight = 35 →
  initial_weight - lost_weight = 34

theorem jessie_current_weight : jessie_weight_problem 69 35 := by
  sorry

end NUMINAMATH_CALUDE_jessie_current_weight_l3471_347163


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3471_347157

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 1 + a 7 = -2)
  (h_a3 : a 3 = 2) :
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = -3 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3471_347157


namespace NUMINAMATH_CALUDE_service_center_location_l3471_347142

/-- Represents a highway with exits and a service center -/
structure Highway where
  third_exit : ℝ
  seventh_exit : ℝ
  twelfth_exit : ℝ
  service_center : ℝ

/-- Theorem stating the location of the service center on the highway -/
theorem service_center_location (h : Highway) 
  (h_third : h.third_exit = 30)
  (h_seventh : h.seventh_exit = 90)
  (h_twelfth : h.twelfth_exit = 195)
  (h_service : h.service_center = h.third_exit + 2/3 * (h.seventh_exit - h.third_exit)) :
  h.service_center = 70 := by
  sorry

#check service_center_location

end NUMINAMATH_CALUDE_service_center_location_l3471_347142


namespace NUMINAMATH_CALUDE_product_negative_five_sum_options_l3471_347178

theorem product_negative_five_sum_options (a b c : ℤ) : 
  a * b * c = -5 → (a + b + c = 5 ∨ a + b + c = -3 ∨ a + b + c = -7) := by
sorry

end NUMINAMATH_CALUDE_product_negative_five_sum_options_l3471_347178


namespace NUMINAMATH_CALUDE_circle_points_equality_l3471_347169

theorem circle_points_equality (z : ℂ) (h : Complex.abs z = 1) :
  let A : ℂ := Complex.I
  let B : ℂ := Complex.I * z
  let C : ℂ := z
  let D : ℂ := -Complex.I
  let E : ℂ := -1
  Complex.abs (z - 1) ^ 2 + Complex.abs (z + 1) ^ 2 = 
  Complex.abs (Complex.I * z + 1) ^ 2 + Complex.abs (z + Complex.I) ^ 2 := by
sorry

end NUMINAMATH_CALUDE_circle_points_equality_l3471_347169


namespace NUMINAMATH_CALUDE_larry_gave_candies_l3471_347166

/-- Given that Anna starts with 5 candies and ends up with 91 candies after receiving some from Larry,
    prove that Larry gave Anna 86 candies. -/
theorem larry_gave_candies (initial_candies final_candies : ℕ) 
  (h1 : initial_candies = 5)
  (h2 : final_candies = 91) :
  final_candies - initial_candies = 86 := by
  sorry

end NUMINAMATH_CALUDE_larry_gave_candies_l3471_347166


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l3471_347153

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → y = 0 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l3471_347153


namespace NUMINAMATH_CALUDE_money_ratio_problem_l3471_347141

/-- Given the ratio of money between Ravi and Giri, and the amounts of money
    Ravi and Kiran have, prove the ratio of money between Giri and Kiran. -/
theorem money_ratio_problem (ravi_giri_ratio : ℚ) (ravi_money kiran_money : ℕ) :
  ravi_giri_ratio = 6 / 7 →
  ravi_money = 36 →
  kiran_money = 105 →
  ∃ (giri_money : ℕ), 
    (ravi_money : ℚ) / giri_money = ravi_giri_ratio ∧
    (giri_money : ℚ) / kiran_money = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_money_ratio_problem_l3471_347141


namespace NUMINAMATH_CALUDE_faye_pencil_rows_l3471_347196

def total_pencils : ℕ := 35
def pencils_per_row : ℕ := 5

theorem faye_pencil_rows :
  total_pencils / pencils_per_row = 7 := by
  sorry

end NUMINAMATH_CALUDE_faye_pencil_rows_l3471_347196


namespace NUMINAMATH_CALUDE_genevieve_coffee_consumption_l3471_347138

/-- Proves that Genevieve drank 6 pints of coffee given the conditions -/
theorem genevieve_coffee_consumption 
  (total_coffee : ℚ) 
  (num_thermoses : ℕ) 
  (genevieve_thermoses : ℕ) 
  (h1 : total_coffee = 4.5) 
  (h2 : num_thermoses = 18) 
  (h3 : genevieve_thermoses = 3) 
  (h4 : ∀ g : ℚ, g * 8 = g * (8 : ℚ)) -- Conversion from gallons to pints
  : (total_coffee * 8 * genevieve_thermoses) / num_thermoses = 6 := by
  sorry

end NUMINAMATH_CALUDE_genevieve_coffee_consumption_l3471_347138


namespace NUMINAMATH_CALUDE_tank_capacity_l3471_347171

/-- 
Given a tank with an unknown capacity, prove that if it's initially filled to 3/4 of its capacity,
and adding 7 gallons fills it to 9/10 of its capacity, then the tank's total capacity is 140/3 gallons.
-/
theorem tank_capacity (tank_capacity : ℝ) : 
  (3 / 4 * tank_capacity + 7 = 9 / 10 * tank_capacity) ↔ 
  (tank_capacity = 140 / 3) := by
sorry

end NUMINAMATH_CALUDE_tank_capacity_l3471_347171


namespace NUMINAMATH_CALUDE_sanchez_problem_l3471_347184

theorem sanchez_problem (x y : ℕ+) 
  (h1 : x.val - y.val = 2)
  (h2 : x.val * y.val = 120) :
  x.val + y.val = 22 := by
  sorry

end NUMINAMATH_CALUDE_sanchez_problem_l3471_347184


namespace NUMINAMATH_CALUDE_intersection_range_l3471_347126

open Set Real

theorem intersection_range (m : ℝ) : 
  let A : Set ℝ := {x | |x - 1| + |x + 1| ≤ 3}
  let B : Set ℝ := {x | x^2 - (2*m + 1)*x + m^2 + m < 0}
  (A ∩ B).Nonempty → m > -3/2 ∧ m < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_range_l3471_347126


namespace NUMINAMATH_CALUDE_additional_toothpicks_for_extension_l3471_347150

/-- The number of toothpicks required for a staircase with n steps -/
def toothpicks (n : ℕ) : ℕ :=
  if n ≤ 2 then 2 * n + 2 else 2 * n + (n - 1) * (n - 2)

theorem additional_toothpicks_for_extension :
  toothpicks 4 = 26 →
  toothpicks 6 - toothpicks 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_additional_toothpicks_for_extension_l3471_347150


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l3471_347151

def inequality_system (x : ℝ) : Prop :=
  (1 - 3 * (x - 1) < 8 - x) ∧ ((x - 3) / 2 + 2 ≥ x)

def integer_solutions : Set ℤ :=
  {-1, 0, 1}

theorem inequality_system_integer_solutions :
  ∀ (n : ℤ), (n ∈ integer_solutions) ↔ (inequality_system (n : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l3471_347151


namespace NUMINAMATH_CALUDE_a_10_value_l3471_347125

def sequence_property (a : ℕ+ → ℤ) : Prop :=
  ∀ p q : ℕ+, a (p + q) = a p + a q

theorem a_10_value (a : ℕ+ → ℤ) (h1 : sequence_property a) (h2 : a 2 = -6) :
  a 10 = -30 := by
  sorry

end NUMINAMATH_CALUDE_a_10_value_l3471_347125


namespace NUMINAMATH_CALUDE_floor_div_p_equals_86422_l3471_347195

/-- A function that generates the sequence of 6-digit numbers with digits in non-increasing order -/
def nonIncreasingDigitSequence : ℕ → ℕ := sorry

/-- The 2010th number in the sequence -/
def p : ℕ := nonIncreasingDigitSequence 2010

/-- Theorem stating that the floor division of p by 10 equals 86422 -/
theorem floor_div_p_equals_86422 : p / 10 = 86422 := by sorry

end NUMINAMATH_CALUDE_floor_div_p_equals_86422_l3471_347195


namespace NUMINAMATH_CALUDE_towel_rate_problem_l3471_347183

/-- Proves that the rate of two towels is 250 given the conditions of the problem -/
theorem towel_rate_problem (price1 price2 avg_price : ℕ) 
  (h1 : price1 = 100)
  (h2 : price2 = 150)
  (h3 : avg_price = 155)
  : ((10 * avg_price) - (3 * price1 + 5 * price2)) / 2 = 250 := by
  sorry

end NUMINAMATH_CALUDE_towel_rate_problem_l3471_347183


namespace NUMINAMATH_CALUDE_distribute_six_balls_four_boxes_l3471_347174

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 65 ways to distribute 6 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_six_balls_four_boxes : distribute_balls 6 4 = 65 := by sorry

end NUMINAMATH_CALUDE_distribute_six_balls_four_boxes_l3471_347174


namespace NUMINAMATH_CALUDE_article_cost_price_l3471_347179

/-- The cost price of an article satisfying certain profit conditions -/
theorem article_cost_price (C : ℝ) (S : ℝ) : 
  S = 1.05 * C →  -- Condition 1: Selling price is 105% of cost price
  (1.05 * C - 1) = 1.1 * (0.95 * C) →  -- Condition 2: New selling price equals 110% of new cost price
  C = 200 := by
sorry

end NUMINAMATH_CALUDE_article_cost_price_l3471_347179


namespace NUMINAMATH_CALUDE_special_product_equality_l3471_347187

theorem special_product_equality (x y : ℝ) : 
  (2 * x^3 - 5 * y^2) * (4 * x^6 + 10 * x^3 * y^2 + 25 * y^4) = 8 * x^9 - 125 * y^6 := by
  sorry

end NUMINAMATH_CALUDE_special_product_equality_l3471_347187


namespace NUMINAMATH_CALUDE_f_of_one_equals_negative_two_l3471_347199

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_of_one_equals_negative_two
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_def : ∀ x, x < 0 → f x = x - x^4) :
  f 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_one_equals_negative_two_l3471_347199


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l3471_347161

theorem unique_solution_to_equation : ∃! x : ℝ, 
  (x ≠ 5 ∧ x ≠ 3) ∧ 
  (x - 2) * (x - 5) * (x - 3) * (x - 2) * (x - 4) * (x - 5) * (x - 3) = 
  (x - 5) * (x - 3) * (x - 5) ∧ 
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l3471_347161


namespace NUMINAMATH_CALUDE_philips_banana_groups_l3471_347154

theorem philips_banana_groups (total_bananas : ℕ) (group_size : ℕ) (h1 : total_bananas = 180) (h2 : group_size = 18) :
  total_bananas / group_size = 10 := by
  sorry

end NUMINAMATH_CALUDE_philips_banana_groups_l3471_347154


namespace NUMINAMATH_CALUDE_constant_term_of_expansion_l3471_347173

/-- The constant term in the expansion of (9x + 2/(3x))^8 -/
def constant_term : ℕ := 90720

/-- The binomial coefficient (8 choose 4) -/
def binomial_8_4 : ℕ := 70

theorem constant_term_of_expansion :
  constant_term = binomial_8_4 * 9^4 * 2^4 / 3^4 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_of_expansion_l3471_347173


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3471_347180

theorem complex_modulus_problem (z : ℂ) : (z - 3) * (1 - 3*I) = 10 → Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3471_347180


namespace NUMINAMATH_CALUDE_screenwriter_speed_l3471_347121

/-- Calculates the average words per minute for a given script and writing duration -/
def average_words_per_minute (total_words : ℕ) (total_hours : ℕ) : ℚ :=
  (total_words : ℚ) / (total_hours * 60 : ℚ)

/-- Theorem stating that a 30,000-word script written in 100 hours has an average writing speed of 5 words per minute -/
theorem screenwriter_speed : average_words_per_minute 30000 100 = 5 := by
  sorry

#eval average_words_per_minute 30000 100

end NUMINAMATH_CALUDE_screenwriter_speed_l3471_347121


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l3471_347165

theorem circle_center_radius_sum : ∃ (a b r : ℝ),
  (∀ (x y : ℝ), x^2 - 14*x + y^2 + 6*y = 25 ↔ (x - a)^2 + (y - b)^2 = r^2) ∧
  a + b + r = 4 + Real.sqrt 83 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l3471_347165


namespace NUMINAMATH_CALUDE_rain_probability_weekend_l3471_347193

/-- Probability of rain on at least one day during a weekend given specific conditions --/
theorem rain_probability_weekend (p_rain_sat : ℝ) (p_rain_sun_given_rain_sat : ℝ) (p_rain_sun_given_no_rain_sat : ℝ)
  (h1 : p_rain_sat = 0.3)
  (h2 : p_rain_sun_given_rain_sat = 0.7)
  (h3 : p_rain_sun_given_no_rain_sat = 0.6) :
  1 - (1 - p_rain_sat) * (1 - p_rain_sun_given_no_rain_sat) = 0.72 := by
  sorry

#check rain_probability_weekend

end NUMINAMATH_CALUDE_rain_probability_weekend_l3471_347193


namespace NUMINAMATH_CALUDE_third_roll_six_prob_l3471_347127

-- Define the probabilities for each die
def fair_die_prob : ℚ := 1/6
def biased_die_six_prob : ℚ := 2/3
def biased_die_other_prob : ℚ := 1/15

-- Define the probability of choosing each die
def die_choice_prob : ℚ := 1/2

-- Define the event of rolling two sixes
def two_sixes_prob (die_prob : ℚ) : ℚ := die_prob * die_prob

-- Define the total probability of rolling two sixes
def total_two_sixes_prob : ℚ := 
  die_choice_prob * two_sixes_prob fair_die_prob + 
  die_choice_prob * two_sixes_prob biased_die_six_prob

-- Define the conditional probability of choosing each die given two sixes
def fair_die_given_two_sixes : ℚ := 
  (two_sixes_prob fair_die_prob * die_choice_prob) / total_two_sixes_prob

def biased_die_given_two_sixes : ℚ := 
  (two_sixes_prob biased_die_six_prob * die_choice_prob) / total_two_sixes_prob

-- Theorem statement
theorem third_roll_six_prob : 
  fair_die_prob * fair_die_given_two_sixes + 
  biased_die_six_prob * biased_die_given_two_sixes = 65/102 := by
  sorry

end NUMINAMATH_CALUDE_third_roll_six_prob_l3471_347127


namespace NUMINAMATH_CALUDE_lucky_larry_problem_l3471_347108

theorem lucky_larry_problem (a b c d e : ℤ) : 
  a = 2 ∧ b = 3 ∧ c = 4 ∧ d = 5 →
  a - b - c - d + e = a - (b - (c - (d + e))) →
  e / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lucky_larry_problem_l3471_347108


namespace NUMINAMATH_CALUDE_sum_of_divisors_180_l3471_347115

def sumOfDivisors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_180 : sumOfDivisors 180 = 546 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_180_l3471_347115


namespace NUMINAMATH_CALUDE_line_intersection_range_l3471_347109

/-- Given a line y = 2x + (3-a) intersecting the x-axis between points (3,0) and (4,0) inclusive, 
    the range of values for a is 9 ≤ a ≤ 11. -/
theorem line_intersection_range (a : ℝ) : 
  (∃ x : ℝ, 3 ≤ x ∧ x ≤ 4 ∧ 0 = 2*x + (3-a)) → 
  (9 ≤ a ∧ a ≤ 11) := by
sorry

end NUMINAMATH_CALUDE_line_intersection_range_l3471_347109


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_slope_range_l3471_347149

/-- The slope range for a line intersecting an ellipse --/
theorem line_ellipse_intersection_slope_range :
  ∀ m : ℝ,
  (∃ x y : ℝ, y = m * x + 7 ∧ 4 * x^2 + 25 * y^2 = 100) →
  -Real.sqrt (9/5) ≤ m ∧ m ≤ Real.sqrt (9/5) := by
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_slope_range_l3471_347149


namespace NUMINAMATH_CALUDE_square_side_length_l3471_347129

theorem square_side_length (x : ℝ) (h : x > 0) (h_area : x^2 = 4 * 3) : x = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3471_347129


namespace NUMINAMATH_CALUDE_expression_equivalence_l3471_347168

theorem expression_equivalence :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * 
  (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l3471_347168


namespace NUMINAMATH_CALUDE_car_speed_problem_l3471_347190

/-- Given a car traveling for 2 hours with a speed of 40 km/h in the second hour
    and an average speed of 90 km/h over the entire journey,
    prove that the speed in the first hour must be 140 km/h. -/
theorem car_speed_problem (speed_second_hour : ℝ) (average_speed : ℝ) (speed_first_hour : ℝ) :
  speed_second_hour = 40 →
  average_speed = 90 →
  average_speed = (speed_first_hour + speed_second_hour) / 2 →
  speed_first_hour = 140 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3471_347190


namespace NUMINAMATH_CALUDE_locus_C_is_ellipse_l3471_347140

/-- Circle O₁ with equation (x-1)² + y² = 1 -/
def circle_O₁ : Set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + p.2^2 = 1}

/-- Circle O₂ with equation (x+1)² + y² = 16 -/
def circle_O₂ : Set (ℝ × ℝ) :=
  {p | (p.1 + 1)^2 + p.2^2 = 16}

/-- The set of points P(x, y) that represent the center of circle C -/
def locus_C : Set (ℝ × ℝ) :=
  {p | ∃ r > 0,
    (∀ q ∈ circle_O₁, (p.1 - q.1)^2 + (p.2 - q.2)^2 = (r + 1)^2) ∧
    (∀ q ∈ circle_O₂, (p.1 - q.1)^2 + (p.2 - q.2)^2 = (4 - r)^2)}

/-- Theorem stating that the locus of the center of circle C is an ellipse -/
theorem locus_C_is_ellipse : ∃ a b c d e f : ℝ,
  a > 0 ∧ b^2 < 4 * a * c ∧
  locus_C = {p | a * p.1^2 + b * p.1 * p.2 + c * p.2^2 + d * p.1 + e * p.2 + f = 0} :=
sorry

end NUMINAMATH_CALUDE_locus_C_is_ellipse_l3471_347140


namespace NUMINAMATH_CALUDE_dice_probability_l3471_347162

/-- The number of dice --/
def n : ℕ := 7

/-- The number of sides on each die --/
def sides : ℕ := 12

/-- The number of favorable outcomes on each die (numbers less than 6) --/
def favorable : ℕ := 5

/-- The number of dice we want to show a favorable outcome --/
def k : ℕ := 3

/-- The probability of exactly k out of n dice showing a number less than 6 --/
def probability : ℚ := (n.choose k) * (favorable / sides) ^ k * ((sides - favorable) / sides) ^ (n - k)

theorem dice_probability : probability = 10504375 / 373248 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l3471_347162


namespace NUMINAMATH_CALUDE_vendor_apples_thrown_away_l3471_347132

/-- Represents the percentage of apples remaining after each operation -/
def apples_remaining (initial_percentage : ℚ) (sell_percentage : ℚ) : ℚ :=
  initial_percentage * (1 - sell_percentage)

/-- Represents the percentage of apples thrown away -/
def apples_thrown (initial_percentage : ℚ) (throw_percentage : ℚ) : ℚ :=
  initial_percentage * throw_percentage

theorem vendor_apples_thrown_away :
  let initial_stock := 1
  let first_day_remaining := apples_remaining initial_stock (60 / 100)
  let first_day_thrown := apples_thrown first_day_remaining (40 / 100)
  let second_day_remaining := apples_remaining (first_day_remaining - first_day_thrown) (50 / 100)
  let second_day_thrown := second_day_remaining
  first_day_thrown + second_day_thrown = 28 / 100 := by
  sorry

end NUMINAMATH_CALUDE_vendor_apples_thrown_away_l3471_347132


namespace NUMINAMATH_CALUDE_erica_ice_cream_weeks_l3471_347113

/-- The number of weeks Erica buys ice cream -/
def ice_cream_weeks (orange_creamsicle_price : ℚ) 
                    (ice_cream_sandwich_price : ℚ)
                    (nutty_buddy_price : ℚ)
                    (total_spent : ℚ) : ℚ :=
  let weekly_spending := 3 * orange_creamsicle_price + 
                         2 * ice_cream_sandwich_price + 
                         2 * nutty_buddy_price
  total_spent / weekly_spending

/-- Theorem stating that Erica buys ice cream for 6 weeks -/
theorem erica_ice_cream_weeks : 
  ice_cream_weeks 2 1.5 3 90 = 6 := by
  sorry

end NUMINAMATH_CALUDE_erica_ice_cream_weeks_l3471_347113


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3471_347137

theorem cube_volume_from_surface_area : 
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 864 → s^3 = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3471_347137


namespace NUMINAMATH_CALUDE_same_gender_officers_l3471_347192

theorem same_gender_officers (total_members : Nat) (boys : Nat) (girls : Nat) :
  total_members = 24 →
  boys = 12 →
  girls = 12 →
  boys + girls = total_members →
  (boys * (boys - 1) + girls * (girls - 1) : Nat) = 264 := by
  sorry

end NUMINAMATH_CALUDE_same_gender_officers_l3471_347192


namespace NUMINAMATH_CALUDE_expand_expression_l3471_347156

theorem expand_expression (x : ℝ) : (17*x + 18 - 3*x^2) * (4*x) = -12*x^3 + 68*x^2 + 72*x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3471_347156


namespace NUMINAMATH_CALUDE_max_product_constrained_l3471_347136

theorem max_product_constrained (x y : ℕ+) (h : 7 * x + 4 * y = 140) :
  x * y ≤ 168 :=
sorry

end NUMINAMATH_CALUDE_max_product_constrained_l3471_347136


namespace NUMINAMATH_CALUDE_expression_simplification_l3471_347160

theorem expression_simplification (x y : ℝ) : 
  -(3*x*y - 2*x^2) - 2*(3*x^2 - x*y) = -4*x^2 - x*y := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3471_347160


namespace NUMINAMATH_CALUDE_inequality_solution_one_inequality_solution_two_l3471_347124

-- Part 1
theorem inequality_solution_one : 
  {x : ℝ | 1 < x^2 - 3*x + 1 ∧ x^2 - 3*x + 1 < 9 - x} = 
  {x : ℝ | (-2 < x ∧ x < 0) ∨ (3 < x ∧ x < 4)} := by sorry

-- Part 2
def solution_set (a : ℝ) : Set ℝ :=
  {x : ℝ | (x - a) / (x - a^2) < 0}

theorem inequality_solution_two (a : ℝ) : 
  (a = 0 ∨ a = 1 → solution_set a = ∅) ∧
  (0 < a ∧ a < 1 → solution_set a = {x : ℝ | a^2 < x ∧ x < a}) ∧
  ((a < 0 ∨ a > 1) → solution_set a = {x : ℝ | a < x ∧ x < a^2}) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_one_inequality_solution_two_l3471_347124


namespace NUMINAMATH_CALUDE_bills_age_l3471_347189

theorem bills_age (caroline_age bill_age : ℕ) : 
  bill_age = 2 * caroline_age - 1 →
  bill_age + caroline_age = 26 →
  bill_age = 17 := by
sorry

end NUMINAMATH_CALUDE_bills_age_l3471_347189


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l3471_347105

theorem modulus_of_complex_number : 
  let i : ℂ := Complex.I
  ∃ (z : ℂ), z = i^2017 / (1 + i) ∧ Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l3471_347105


namespace NUMINAMATH_CALUDE_composition_equation_solution_l3471_347106

theorem composition_equation_solution :
  let δ : ℝ → ℝ := λ x ↦ 4 * x + 5
  let φ : ℝ → ℝ := λ x ↦ 5 * x + 4
  ∀ x : ℝ, δ (φ x) = 9 → x = -3/5 :=
by sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l3471_347106


namespace NUMINAMATH_CALUDE_grants_yearly_expense_l3471_347116

/-- Grant's yearly newspaper delivery expense --/
def grants_expense : ℝ := 200

/-- Juanita's daily expense from Monday to Saturday --/
def juanita_weekday_expense : ℝ := 0.5

/-- Juanita's Sunday expense --/
def juanita_sunday_expense : ℝ := 2

/-- Number of weeks in a year --/
def weeks_per_year : ℕ := 52

/-- Difference between Juanita's and Grant's yearly expenses --/
def expense_difference : ℝ := 60

/-- Theorem stating Grant's yearly newspaper delivery expense --/
theorem grants_yearly_expense : 
  grants_expense = 
    weeks_per_year * (6 * juanita_weekday_expense + juanita_sunday_expense) - expense_difference :=
by sorry

end NUMINAMATH_CALUDE_grants_yearly_expense_l3471_347116


namespace NUMINAMATH_CALUDE_ellipse_m_value_l3471_347143

/-- Given an ellipse with equation x²/4 + y²/m = 1, foci on the x-axis, 
    and eccentricity 1/2, prove that m = 3 -/
theorem ellipse_m_value (m : ℝ) 
  (h1 : ∀ (x y : ℝ), x^2/4 + y^2/m = 1 → (∃ (a b c : ℝ), a^2 = 4 ∧ b^2 = m ∧ c^2 = a^2 - b^2))
  (h2 : ∃ (e : ℝ), e = 1/2 ∧ e^2 = (4 - m)/4) : 
  m = 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l3471_347143


namespace NUMINAMATH_CALUDE_product_line_size_l3471_347122

/-- Represents the product line of Company C -/
structure ProductLine where
  n : ℕ                  -- number of products
  prices : Fin n → ℝ     -- prices of products
  avg_price : ℝ          -- average price
  min_price : ℝ          -- minimum price
  max_price : ℝ          -- maximum price
  low_price_count : ℕ    -- count of products below $1000

/-- The product line satisfies the given conditions -/
def satisfies_conditions (pl : ProductLine) : Prop :=
  pl.avg_price = 1200 ∧
  (∀ i, pl.prices i ≥ 400) ∧
  pl.low_price_count = 10 ∧
  (∀ i, pl.prices i < 1000 ∨ pl.prices i ≥ 1000) ∧
  pl.max_price = 11000 ∧
  (∃ i, pl.prices i = pl.max_price)

/-- The theorem to be proved -/
theorem product_line_size (pl : ProductLine) 
  (h : satisfies_conditions pl) : pl.n = 20 := by
  sorry


end NUMINAMATH_CALUDE_product_line_size_l3471_347122


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l3471_347139

theorem complex_number_in_third_quadrant : 
  let z : ℂ := (1 - Complex.I)^2 / (1 + Complex.I)
  (z.re < 0) ∧ (z.im < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l3471_347139


namespace NUMINAMATH_CALUDE_center_value_15_l3471_347198

-- Define a 3x3 array type
def Array3x3 := Fin 3 → Fin 3 → ℝ

-- Define a predicate for arithmetic sequences in rows and columns
def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

def all_rows_cols_arithmetic (arr : Array3x3) : Prop :=
  (∀ i : Fin 3, is_arithmetic_sequence (arr i 0) (arr i 1) (arr i 2)) ∧
  (∀ j : Fin 3, is_arithmetic_sequence (arr 0 j) (arr 1 j) (arr 2 j))

-- State the theorem
theorem center_value_15 (arr : Array3x3) :
  all_rows_cols_arithmetic arr →
  arr 0 0 = 3 →
  arr 0 2 = 15 →
  arr 2 0 = 9 →
  arr 2 2 = 33 →
  arr 1 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_center_value_15_l3471_347198


namespace NUMINAMATH_CALUDE_geometric_sum_l3471_347177

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 + a 6 = 3 →
  a 6 + a 10 = 12 →
  a 8 + a 12 = 24 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_l3471_347177


namespace NUMINAMATH_CALUDE_rectangle_y_value_l3471_347197

/-- Given a rectangle with vertices at (-3, y), (5, y), (-3, -1), and (5, -1),
    with an area of 48 square units and y > 0, prove that y = 5. -/
theorem rectangle_y_value (y : ℝ) 
    (h1 : (5 - (-3)) * (y - (-1)) = 48)  -- Area of rectangle is 48
    (h2 : y > 0) :                       -- y is positive
  y = 5 := by sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l3471_347197


namespace NUMINAMATH_CALUDE_peter_class_size_l3471_347128

/-- The number of students in Peter's class with 2 hands each -/
def students_with_two_hands : ℕ := 10

/-- The number of students in Peter's class with 1 hand each -/
def students_with_one_hand : ℕ := 3

/-- The number of students in Peter's class with 3 hands each -/
def students_with_three_hands : ℕ := 1

/-- The total number of hands in the class excluding Peter's -/
def total_hands_excluding_peter : ℕ := 20

/-- The number of hands Peter has (assumed to be typical) -/
def peter_hands : ℕ := 2

/-- The total number of students in Peter's class, including Peter -/
def total_students : ℕ := 14

theorem peter_class_size :
  (students_with_two_hands * 2 + 
   students_with_one_hand * 1 + 
   students_with_three_hands * 3 + 
   peter_hands) / 2 = total_students := by sorry

end NUMINAMATH_CALUDE_peter_class_size_l3471_347128


namespace NUMINAMATH_CALUDE_library_visitors_l3471_347194

theorem library_visitors (month_days : ℕ) (non_sunday_visitors : ℕ) (avg_visitors : ℕ) :
  month_days = 30 →
  non_sunday_visitors = 700 →
  avg_visitors = 750 →
  ∃ (sunday_visitors : ℕ),
    (5 * sunday_visitors + 25 * non_sunday_visitors) / month_days = avg_visitors ∧
    sunday_visitors = 1000 :=
by sorry

end NUMINAMATH_CALUDE_library_visitors_l3471_347194


namespace NUMINAMATH_CALUDE_p_or_q_necessary_not_sufficient_l3471_347135

theorem p_or_q_necessary_not_sufficient (p q : Prop) :
  (¬¬p → (p ∨ q)) ∧ ¬((p ∨ q) → ¬¬p) := by
  sorry

end NUMINAMATH_CALUDE_p_or_q_necessary_not_sufficient_l3471_347135


namespace NUMINAMATH_CALUDE_library_average_MB_per_hour_l3471_347118

/-- Calculates the average megabytes per hour of music in a digital library -/
def averageMBPerHour (days : ℕ) (totalMB : ℕ) : ℕ :=
  let hoursPerDay : ℕ := 24
  let totalHours : ℕ := days * hoursPerDay
  let exactAverage : ℚ := totalMB / totalHours
  (exactAverage + 1/2).floor.toNat

/-- Proves that the average megabytes per hour for the given library is 67 MB -/
theorem library_average_MB_per_hour :
  averageMBPerHour 15 24000 = 67 := by
  sorry

end NUMINAMATH_CALUDE_library_average_MB_per_hour_l3471_347118
