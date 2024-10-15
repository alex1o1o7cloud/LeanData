import Mathlib

namespace NUMINAMATH_CALUDE_new_computer_cost_new_computer_cost_is_600_l1310_131070

theorem new_computer_cost (used_computers_cost : ℕ) (savings : ℕ) : ℕ :=
  let new_computer_cost := used_computers_cost + savings
  new_computer_cost

#check new_computer_cost 400 200

theorem new_computer_cost_is_600 :
  new_computer_cost 400 200 = 600 := by sorry

end NUMINAMATH_CALUDE_new_computer_cost_new_computer_cost_is_600_l1310_131070


namespace NUMINAMATH_CALUDE_parabola_equation_l1310_131021

/-- A parabola with vertex at the origin, directrix perpendicular to the x-axis, 
    and passing through the point (1, -√2) has the equation y² = 2x -/
theorem parabola_equation : ∃ (f : ℝ → ℝ),
  (∀ x y : ℝ, f x = y ↔ y^2 = 2*x) ∧ 
  (f 0 = 0) ∧ 
  (∃ a : ℝ, ∀ x : ℝ, (x < a ↔ f x < 0) ∧ (x > a ↔ f x > 0)) ∧
  (f 1 = -Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1310_131021


namespace NUMINAMATH_CALUDE_trap_speed_constant_and_eight_l1310_131066

/-- Representation of a 4-level staircase --/
structure Staircase :=
  (h : ℝ)  -- height of each step
  (b : ℝ)  -- width of each step
  (a : ℝ)  -- length of the staircase
  (v : ℝ)  -- speed of the mouse

/-- The speed of the mouse trap required to catch the mouse --/
def trap_speed (s : Staircase) : ℝ := 8

/-- Theorem stating that the trap speed is constant and equal to 8 cm/s --/
theorem trap_speed_constant_and_eight (s : Staircase) 
  (h_height : s.h = 3)
  (h_width : s.b = 1)
  (h_length : s.a = 8)
  (h_mouse_speed : s.v = 17) :
  trap_speed s = 8 ∧ 
  ∀ (placement : ℝ), 0 ≤ placement ∧ placement ≤ s.a → trap_speed s = 8 := by
  sorry

#check trap_speed_constant_and_eight

end NUMINAMATH_CALUDE_trap_speed_constant_and_eight_l1310_131066


namespace NUMINAMATH_CALUDE_pilot_miles_flown_l1310_131096

theorem pilot_miles_flown (tuesday_miles : ℕ) (thursday_miles : ℕ) (total_weeks : ℕ) (total_miles : ℕ) : 
  thursday_miles = 1475 → 
  total_weeks = 3 → 
  total_miles = 7827 → 
  total_miles = total_weeks * (tuesday_miles + thursday_miles) → 
  tuesday_miles = 1134 := by
sorry

end NUMINAMATH_CALUDE_pilot_miles_flown_l1310_131096


namespace NUMINAMATH_CALUDE_coffee_shop_optimal_price_l1310_131016

/-- Profit function for the coffee shop -/
def profit (p : ℝ) : ℝ := 150 * p - 4 * p^2 - 200

/-- The constraint on the price -/
def price_constraint (p : ℝ) : Prop := p ≤ 30

/-- The optimal price that maximizes profit -/
def optimal_price : ℝ := 19

theorem coffee_shop_optimal_price :
  ∃ (p : ℝ), price_constraint p ∧ 
  ∀ (q : ℝ), price_constraint q → profit p ≥ profit q ∧
  p = optimal_price :=
sorry

end NUMINAMATH_CALUDE_coffee_shop_optimal_price_l1310_131016


namespace NUMINAMATH_CALUDE_complex_number_properties_l1310_131069

def z : ℂ := 4 + 3 * Complex.I

theorem complex_number_properties :
  Complex.abs z = 5 ∧ (1 + Complex.I) / z = (7 + Complex.I) / 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l1310_131069


namespace NUMINAMATH_CALUDE_manufacturing_earnings_l1310_131026

/-- Calculates total earnings given hourly wage, bonus per widget, number of widgets, and work hours -/
def totalEarnings (hourlyWage : ℚ) (bonusPerWidget : ℚ) (numWidgets : ℕ) (workHours : ℕ) : ℚ :=
  hourlyWage * workHours + bonusPerWidget * numWidgets

/-- Proves that the total earnings for the given conditions is $700 -/
theorem manufacturing_earnings :
  totalEarnings (12.5) (0.16) 1250 40 = 700 := by
  sorry

end NUMINAMATH_CALUDE_manufacturing_earnings_l1310_131026


namespace NUMINAMATH_CALUDE_fruit_basket_problem_l1310_131000

theorem fruit_basket_problem (total_fruits : ℕ) 
  (mangoes pears pawpaws kiwis lemons : ℕ) : 
  total_fruits = 58 →
  mangoes = 18 →
  pears = 10 →
  pawpaws = 12 →
  kiwis = lemons →
  total_fruits = mangoes + pears + pawpaws + kiwis + lemons →
  lemons = 9 := by
sorry

end NUMINAMATH_CALUDE_fruit_basket_problem_l1310_131000


namespace NUMINAMATH_CALUDE_max_cards_purchasable_l1310_131030

theorem max_cards_purchasable (budget : ℚ) (card_cost : ℚ) (h1 : budget = 15/2) (h2 : card_cost = 17/20) :
  ⌊budget / card_cost⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_cards_purchasable_l1310_131030


namespace NUMINAMATH_CALUDE_sugar_amount_is_two_l1310_131028

-- Define the ratios and quantities
def sugar_to_cheese_ratio : ℚ := 1 / 4
def vanilla_to_cheese_ratio : ℚ := 1 / 2
def eggs_to_vanilla_ratio : ℚ := 2
def eggs_used : ℕ := 8

-- Define the function to calculate sugar used
def sugar_used (eggs : ℕ) : ℚ :=
  (eggs : ℚ) / eggs_to_vanilla_ratio / vanilla_to_cheese_ratio * sugar_to_cheese_ratio

-- Theorem statement
theorem sugar_amount_is_two : sugar_used eggs_used = 2 := by
  sorry

end NUMINAMATH_CALUDE_sugar_amount_is_two_l1310_131028


namespace NUMINAMATH_CALUDE_distribute_5_8_l1310_131091

/-- The number of ways to distribute n different items into m boxes with at most one item per box -/
def distribute (n m : ℕ) : ℕ :=
  (m - n + 1).factorial * (m.choose n)

/-- Theorem: The number of ways to distribute 5 different items into 8 boxes
    with at most one item per box is 6720 -/
theorem distribute_5_8 : distribute 5 8 = 6720 := by
  sorry

#eval distribute 5 8

end NUMINAMATH_CALUDE_distribute_5_8_l1310_131091


namespace NUMINAMATH_CALUDE_third_student_weight_l1310_131049

theorem third_student_weight (original_count : ℕ) (original_avg : ℝ) 
  (new_count : ℕ) (new_avg : ℝ) (first_weight : ℝ) (second_weight : ℝ) :
  original_count = 29 →
  original_avg = 28 →
  new_count = original_count + 3 →
  new_avg = 27.3 →
  first_weight = 20 →
  second_weight = 30 →
  ∃ (third_weight : ℝ),
    third_weight = new_count * new_avg - original_count * original_avg - first_weight - second_weight ∧
    third_weight = 11.6 := by
  sorry

end NUMINAMATH_CALUDE_third_student_weight_l1310_131049


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1310_131076

theorem absolute_value_inequality (k : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| > k) → k < 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1310_131076


namespace NUMINAMATH_CALUDE_sum_P_2_neg_2_l1310_131019

/-- A cubic polynomial with specific properties -/
structure CubicPolynomial (k : ℝ) where
  P : ℝ → ℝ
  is_cubic : ∃ (a b c : ℝ), ∀ x, P x = a * x^3 + b * x^2 + c * x + k
  P_0 : P 0 = k
  P_1 : P 1 = 3 * k
  P_neg_1 : P (-1) = 4 * k

/-- The sum of P(2) and P(-2) for a cubic polynomial with specific properties -/
theorem sum_P_2_neg_2 (k : ℝ) (P : CubicPolynomial k) :
  P.P 2 + P.P (-2) = 24 * k := by sorry

end NUMINAMATH_CALUDE_sum_P_2_neg_2_l1310_131019


namespace NUMINAMATH_CALUDE_right_triangle_sets_l1310_131013

-- Define a function to check if three numbers can form a right triangle
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

-- Theorem stating that the given sets of numbers satisfy or don't satisfy the right triangle condition
theorem right_triangle_sets :
  (is_right_triangle 6 8 10) ∧
  (is_right_triangle (6/5) 2 (8/5)) ∧
  (is_right_triangle 5 12 13) ∧
  ¬(is_right_triangle (Real.sqrt 8) 2 (Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l1310_131013


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l1310_131088

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

theorem fifteenth_term_of_sequence :
  let a₁ : ℤ := -3
  let d : ℤ := 4
  arithmetic_sequence a₁ d 15 = 53 := by
sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l1310_131088


namespace NUMINAMATH_CALUDE_fruits_given_to_jane_l1310_131087

def initial_plums : ℕ := 16
def initial_guavas : ℕ := 18
def initial_apples : ℕ := 21
def fruits_left : ℕ := 15

def total_initial_fruits : ℕ := initial_plums + initial_guavas + initial_apples

theorem fruits_given_to_jane :
  total_initial_fruits - fruits_left = 40 := by sorry

end NUMINAMATH_CALUDE_fruits_given_to_jane_l1310_131087


namespace NUMINAMATH_CALUDE_max_value_of_m_l1310_131015

theorem max_value_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hm : m > 0)
  (heq : 5 = m^2 * (a^2/b^2 + b^2/a^2) + m * (a/b + b/a)) :
  m ≤ (-1 + Real.sqrt 21) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_m_l1310_131015


namespace NUMINAMATH_CALUDE_hilton_marbles_l1310_131063

/-- Calculates the final number of marbles Hilton has -/
def final_marbles (initial : ℕ) (found : ℕ) (lost : ℕ) : ℕ :=
  initial + found - lost + 2 * lost

theorem hilton_marbles : final_marbles 26 6 10 = 42 := by
  sorry

end NUMINAMATH_CALUDE_hilton_marbles_l1310_131063


namespace NUMINAMATH_CALUDE_ads_on_first_page_l1310_131033

theorem ads_on_first_page (page1 page2 page3 page4 : ℕ) : 
  page2 = 2 * page1 →
  page3 = page2 + 24 →
  page4 = 3 * page2 / 4 →
  68 = 2 * (page1 + page2 + page3 + page4) / 3 →
  page1 = 12 := by
sorry

end NUMINAMATH_CALUDE_ads_on_first_page_l1310_131033


namespace NUMINAMATH_CALUDE_garden_area_difference_l1310_131085

-- Define the dimensions of the gardens
def karl_length : ℝ := 20
def karl_width : ℝ := 45
def makenna_length : ℝ := 25
def makenna_width : ℝ := 40

-- Define the areas of the gardens
def karl_area : ℝ := karl_length * karl_width
def makenna_area : ℝ := makenna_length * makenna_width

-- Theorem to prove
theorem garden_area_difference : makenna_area - karl_area = 100 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_difference_l1310_131085


namespace NUMINAMATH_CALUDE_students_without_A_l1310_131056

theorem students_without_A (total : ℕ) (history : ℕ) (math : ℕ) (both : ℕ) 
  (h_total : total = 40)
  (h_history : history = 12)
  (h_math : math = 18)
  (h_both : both = 6) :
  total - (history + math - both) = 16 := by
  sorry

end NUMINAMATH_CALUDE_students_without_A_l1310_131056


namespace NUMINAMATH_CALUDE_soccer_team_win_percentage_l1310_131064

/-- Calculate the percentage of games won by a soccer team -/
theorem soccer_team_win_percentage 
  (total_games : ℕ) 
  (games_won : ℕ) 
  (h1 : total_games = 130) 
  (h2 : games_won = 78) : 
  (games_won : ℚ) / total_games * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_win_percentage_l1310_131064


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l1310_131060

/-- Proves that the cost price of a bicycle for seller A is 150 given the selling conditions --/
theorem bicycle_cost_price
  (profit_A_to_B : ℝ) -- Profit percentage when A sells to B
  (profit_B_to_C : ℝ) -- Profit percentage when B sells to C
  (price_C : ℝ)       -- Price C pays for the bicycle
  (h1 : profit_A_to_B = 20)
  (h2 : profit_B_to_C = 25)
  (h3 : price_C = 225) :
  ∃ (cost_price_A : ℝ), cost_price_A = 150 ∧
    price_C = cost_price_A * (1 + profit_A_to_B / 100) * (1 + profit_B_to_C / 100) :=
by sorry

end NUMINAMATH_CALUDE_bicycle_cost_price_l1310_131060


namespace NUMINAMATH_CALUDE_ellipse_k_range_l1310_131004

theorem ellipse_k_range : 
  ∀ k : ℝ, (∃ x y : ℝ, (x^2 / (k - 2) + y^2 / (3 - k) = 1) ∧ 
  ((k - 2 > 0) ∧ (3 - k > 0) ∧ (k - 2 ≠ 3 - k))) → 
  (k > 2 ∧ k < 3 ∧ k ≠ 5/2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l1310_131004


namespace NUMINAMATH_CALUDE_initial_number_proof_l1310_131017

theorem initial_number_proof (x : ℕ) : x + 17 = 29 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l1310_131017


namespace NUMINAMATH_CALUDE_net_increase_theorem_l1310_131092

/-- Represents the different types of vehicles -/
inductive VehicleType
  | Car
  | Motorcycle
  | Van

/-- Represents the different phases of the play -/
inductive PlayPhase
  | BeforeIntermission
  | Intermission
  | AfterIntermission

/-- Initial number of vehicles in the back parking lot -/
def initialVehicles : VehicleType → ℕ
  | VehicleType.Car => 50
  | VehicleType.Motorcycle => 75
  | VehicleType.Van => 25

/-- Arrival rate per hour for each vehicle type during regular play time -/
def arrivalRate : VehicleType → ℕ
  | VehicleType.Car => 70
  | VehicleType.Motorcycle => 120
  | VehicleType.Van => 30

/-- Departure rate per hour for each vehicle type during regular play time -/
def departureRate : VehicleType → ℕ
  | VehicleType.Car => 40
  | VehicleType.Motorcycle => 60
  | VehicleType.Van => 20

/-- Duration of each phase in hours -/
def phaseDuration : PlayPhase → ℚ
  | PlayPhase.BeforeIntermission => 1
  | PlayPhase.Intermission => 1/2
  | PlayPhase.AfterIntermission => 3/2

/-- Net increase rate per hour for each vehicle type during a given phase -/
def netIncreaseRate (v : VehicleType) (p : PlayPhase) : ℚ :=
  match p with
  | PlayPhase.BeforeIntermission => (arrivalRate v - departureRate v : ℚ)
  | PlayPhase.Intermission => (arrivalRate v * 3/2 : ℚ)
  | PlayPhase.AfterIntermission => (arrivalRate v - departureRate v : ℚ)

/-- Total net increase for a given vehicle type -/
def totalNetIncrease (v : VehicleType) : ℚ :=
  (netIncreaseRate v PlayPhase.BeforeIntermission * phaseDuration PlayPhase.BeforeIntermission) +
  (netIncreaseRate v PlayPhase.Intermission * phaseDuration PlayPhase.Intermission) +
  (netIncreaseRate v PlayPhase.AfterIntermission * phaseDuration PlayPhase.AfterIntermission)

/-- Theorem stating the net increase for each vehicle type -/
theorem net_increase_theorem :
  ⌊totalNetIncrease VehicleType.Car⌋ = 127 ∧
  ⌊totalNetIncrease VehicleType.Motorcycle⌋ = 240 ∧
  ⌊totalNetIncrease VehicleType.Van⌋ = 47 := by
  sorry


end NUMINAMATH_CALUDE_net_increase_theorem_l1310_131092


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_l1310_131032

theorem least_n_satisfying_inequality : 
  ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, k > 0 → k < n → (1 : ℚ) / k - (1 : ℚ) / (k + 1) ≥ (1 : ℚ) / 15) ∧
  ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15) ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_l1310_131032


namespace NUMINAMATH_CALUDE_f_4_3_2_1_l1310_131098

/-- The mapping f from (a₁, a₂, a₃, a₄) to (b₁, b₂, b₃, b₄) based on the equation
    x^4 + a₁x³ + a₂x² + a₃x + a₄ = (x+1)^4 + b₁(x+1)³ + b₂(x+1)² + b₃(x+1) + b₄ -/
def f (a₁ a₂ a₃ a₄ : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  sorry

theorem f_4_3_2_1 : f 4 3 2 1 = (0, -3, 4, -1) := by
  sorry

end NUMINAMATH_CALUDE_f_4_3_2_1_l1310_131098


namespace NUMINAMATH_CALUDE_smallest_tangent_circle_l1310_131005

/-- The line to which the circle is tangent -/
def line (x y : ℝ) : ℝ := x - y - 4

/-- The circle to which the target circle is tangent -/
def given_circle (x y : ℝ) : ℝ := x^2 + y^2 + 2*x - 2*y

/-- The equation of the target circle -/
def target_circle (x y : ℝ) : ℝ := (x - 1)^2 + (y + 1)^2 - 2

/-- Theorem stating that the target circle is the smallest circle tangent to both the line and the given circle -/
theorem smallest_tangent_circle :
  ∀ r > 0, ∀ a b : ℝ,
    (∀ x y : ℝ, (x - a)^2 + (y - b)^2 = r^2 → line x y ≠ 0) ∧
    (∀ x y : ℝ, (x - a)^2 + (y - b)^2 = r^2 → given_circle x y ≠ 0) →
    r^2 ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_tangent_circle_l1310_131005


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1310_131044

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a d →
  d > 0 →
  a 1 + a 2 + a 3 = 15 →
  a 1 * a 2 * a 3 = 80 →
  a 11 + a 12 + a 13 = 105 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1310_131044


namespace NUMINAMATH_CALUDE_square_circle_union_area_l1310_131008

/-- The area of the union of a square and a circle with specific dimensions -/
theorem square_circle_union_area :
  let square_side : ℝ := 12
  let circle_radius : ℝ := 12
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let overlap_area : ℝ := (1 / 4) * circle_area
  square_area + circle_area - overlap_area = 144 + 108 * π :=
by sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l1310_131008


namespace NUMINAMATH_CALUDE_system_solutions_l1310_131083

theorem system_solutions :
  let S : Set (ℝ × ℝ × ℝ) := { (x, y, z) | x^5 = y^3 + 2*z ∧ y^5 = z^3 + 2*x ∧ z^5 = x^3 + 2*y }
  S = {(0, 0, 0), (Real.sqrt 2, Real.sqrt 2, Real.sqrt 2), (-Real.sqrt 2, -Real.sqrt 2, -Real.sqrt 2)} := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l1310_131083


namespace NUMINAMATH_CALUDE_product_without_linear_term_l1310_131040

theorem product_without_linear_term (m : ℝ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (x + m) * (x + 8) = a * x^2 + b) → m = -8 := by
  sorry

end NUMINAMATH_CALUDE_product_without_linear_term_l1310_131040


namespace NUMINAMATH_CALUDE_angle_problem_l1310_131065

/-- Represents an angle in degrees and minutes -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

/-- Addition of two angles -/
def Angle.add (a b : Angle) : Angle :=
  sorry

/-- Subtraction of two angles -/
def Angle.sub (a b : Angle) : Angle :=
  sorry

/-- Equality of two angles -/
def Angle.eq (a b : Angle) : Prop :=
  sorry

theorem angle_problem (x y : Angle) :
  Angle.add x y = Angle.mk 67 56 →
  Angle.sub x y = Angle.mk 12 40 →
  Angle.eq x (Angle.mk 40 18) ∧ Angle.eq y (Angle.mk 27 38) :=
by
  sorry

end NUMINAMATH_CALUDE_angle_problem_l1310_131065


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_attained_l1310_131078

theorem min_value_of_expression (x : ℝ) : 
  (15 - x) * (13 - x) * (15 + x) * (13 + x) ≥ -961 :=
by sorry

theorem min_value_attained (x : ℝ) : 
  (15 - x) * (13 - x) * (15 + x) * (13 + x) = -961 ↔ x = Real.sqrt 197 ∨ x = -Real.sqrt 197 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_attained_l1310_131078


namespace NUMINAMATH_CALUDE_fifteenth_digit_of_sum_one_eighth_one_sixth_l1310_131094

/-- The decimal representation of a rational number -/
def decimalRepresentation (q : ℚ) : ℕ → ℕ := sorry

/-- The sum of decimal representations of two rational numbers -/
def sumDecimalRepresentations (q₁ q₂ : ℚ) : ℕ → ℕ := sorry

/-- Theorem: The 15th digit after the decimal point in the sum of 1/8 and 1/6 is 6 -/
theorem fifteenth_digit_of_sum_one_eighth_one_sixth : 
  sumDecimalRepresentations (1/8) (1/6) 15 = 6 := by sorry

end NUMINAMATH_CALUDE_fifteenth_digit_of_sum_one_eighth_one_sixth_l1310_131094


namespace NUMINAMATH_CALUDE_inequality_proof_l1310_131074

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a * b) / (a^5 + b^5 + a * b) + (b * c) / (b^5 + c^5 + b * c) + (c * a) / (c^5 + a^5 + c * a) ≤ 1 ∧
  ((a * b) / (a^5 + b^5 + a * b) + (b * c) / (b^5 + c^5 + b * c) + (c * a) / (c^5 + a^5 + c * a) = 1 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1310_131074


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1310_131072

-- Define set A
def A : Set ℝ := {a : ℝ | ∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0}

-- Define set B
def B : Set ℝ := {a : ℝ | ∀ x : ℝ, ¬(|x - 4| + |x - 3| < a)}

-- State the theorem
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1310_131072


namespace NUMINAMATH_CALUDE_rational_roots_of_equation_l1310_131035

theorem rational_roots_of_equation (a b c d : ℝ) :
  ∃ x : ℚ, (a + b)^2 * (x + c^2) * (x + d^2) - (c + d)^2 * (x + a^2) * (x + b^2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_rational_roots_of_equation_l1310_131035


namespace NUMINAMATH_CALUDE_sequence_determination_l1310_131068

/-- A sequence is determined if its terms are uniquely defined by given conditions -/
def is_determined (a : ℕ → ℝ) : Prop := sorry

/-- Arithmetic sequence with given S₁ and S₂ -/
def arithmetic_sequence (a : ℕ → ℝ) (S₁ S₂ : ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a n = a 1 + (n - 1) * d ∧ S₁ = a 1 ∧ S₂ = a 1 + a 2

/-- Geometric sequence with given S₁ and S₂ -/
def geometric_sequence_S₁S₂ (a : ℕ → ℝ) (S₁ S₂ : ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a n = a 1 * q^(n - 1) ∧ S₁ = a 1 ∧ S₂ = a 1 + a 1 * q

/-- Geometric sequence with given S₁ and S₃ -/
def geometric_sequence_S₁S₃ (a : ℕ → ℝ) (S₁ S₃ : ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a n = a 1 * q^(n - 1) ∧ S₁ = a 1 ∧ S₃ = a 1 + a 1 * q + a 1 * q^2

/-- Sequence satisfying given recurrence relations -/
def recurrence_sequence (a : ℕ → ℝ) (x y c : ℝ) : Prop :=
  a 1 = c ∧ 
  (∀ n : ℕ, a (2*n + 2) = a (2*n) + x ∧ a (2*n + 1) = a (2*n - 1) + y)

theorem sequence_determination :
  ∀ a : ℕ → ℝ, ∀ S₁ S₂ S₃ x y c : ℝ,
  (is_determined a ↔ arithmetic_sequence a S₁ S₂) ∧
  (is_determined a ↔ geometric_sequence_S₁S₂ a S₁ S₂) ∧
  ¬(is_determined a ↔ geometric_sequence_S₁S₃ a S₁ S₃) ∧
  ¬(is_determined a ↔ recurrence_sequence a x y c) :=
sorry

end NUMINAMATH_CALUDE_sequence_determination_l1310_131068


namespace NUMINAMATH_CALUDE_tangent_line_parallel_increasing_intervals_decreasing_interval_extreme_values_l1310_131046

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - (2*a + 3)*x + a^2

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x - (2*a + 3)

-- Theorem for part 1
theorem tangent_line_parallel (a : ℝ) :
  f_derivative a (-1) = 2 → a = -1/2 := by sorry

-- Theorems for part 2
theorem increasing_intervals :
  let a := -2
  ∀ x, (x < 1/3 ∨ x > 1) → (f_derivative a x > 0) := by sorry

theorem decreasing_interval :
  let a := -2
  ∀ x, (1/3 < x ∧ x < 1) → (f_derivative a x < 0) := by sorry

theorem extreme_values :
  let a := -2
  (f a (1/3) = 112/27) ∧ (f a 1 = 4) := by sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_increasing_intervals_decreasing_interval_extreme_values_l1310_131046


namespace NUMINAMATH_CALUDE_max_prism_plane_intersections_l1310_131052

/-- A prism is a three-dimensional shape with two identical ends (bases) and flat sides. -/
structure Prism where
  base : Set (ℝ × ℝ)  -- Represents the base of the prism
  height : ℝ           -- Represents the height of the prism

/-- A plane in three-dimensional space. -/
structure Plane where
  normal : ℝ × ℝ × ℝ  -- Normal vector of the plane
  d : ℝ                -- Distance from the origin

/-- Represents the number of edges a plane intersects with a prism. -/
def intersectionCount (prism : Prism) (plane : Plane) : ℕ :=
  sorry  -- Implementation details omitted

/-- Theorem: The maximum number of edges a plane can intersect in a prism is 8. -/
theorem max_prism_plane_intersections (prism : Prism) :
  ∀ plane : Plane, intersectionCount prism plane ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_prism_plane_intersections_l1310_131052


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l1310_131099

theorem partial_fraction_decomposition_product (A B C : ℝ) : 
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -2 ∧ x ≠ 3 → 
    (x^2 - 19) / (x^3 - 2*x^2 - 5*x + 6) = A / (x - 1) + B / (x + 2) + C / (x - 3)) →
  A * B * C = 3 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l1310_131099


namespace NUMINAMATH_CALUDE_phone_call_duration_l1310_131048

/-- Calculates the duration of a phone call given the initial credit, cost per minute, and remaining credit -/
theorem phone_call_duration (initial_credit remaining_credit cost_per_minute : ℚ) : 
  initial_credit = 30 ∧ 
  cost_per_minute = 16/100 ∧ 
  remaining_credit = 264/10 →
  (initial_credit - remaining_credit) / cost_per_minute = 22 := by
  sorry

end NUMINAMATH_CALUDE_phone_call_duration_l1310_131048


namespace NUMINAMATH_CALUDE_estimate_viewers_l1310_131001

theorem estimate_viewers (total_population : ℕ) (sample_size : ℕ) (sample_viewers : ℕ) 
  (h1 : total_population = 3600)
  (h2 : sample_size = 200)
  (h3 : sample_viewers = 160) :
  (total_population : ℚ) * (sample_viewers : ℚ) / (sample_size : ℚ) = 2880 := by
  sorry

end NUMINAMATH_CALUDE_estimate_viewers_l1310_131001


namespace NUMINAMATH_CALUDE_sqrt_five_irrational_and_greater_than_two_l1310_131010

theorem sqrt_five_irrational_and_greater_than_two :
  ∃ x : ℝ, Irrational x ∧ x > 2 ∧ x ^ 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_irrational_and_greater_than_two_l1310_131010


namespace NUMINAMATH_CALUDE_binomial_square_derivation_l1310_131034

theorem binomial_square_derivation (x y : ℝ) :
  ∃ (a b : ℝ), (-1/2 * x + y) * (y + 1/2 * x) = a^2 - b^2 :=
sorry

end NUMINAMATH_CALUDE_binomial_square_derivation_l1310_131034


namespace NUMINAMATH_CALUDE_mr_green_potato_yield_l1310_131054

/-- Calculates the expected potato yield for a rectangular garden -/
def expected_potato_yield (length_steps : ℕ) (width_steps : ℕ) (step_length : ℚ) (yield_per_sqft : ℚ) : ℚ :=
  (length_steps : ℚ) * step_length * (width_steps : ℚ) * step_length * yield_per_sqft

/-- Theorem: The expected potato yield for Mr. Green's garden is 2109.375 pounds -/
theorem mr_green_potato_yield :
  expected_potato_yield 18 25 (5/2) (3/4) = 2109375/1000 := by
  sorry

end NUMINAMATH_CALUDE_mr_green_potato_yield_l1310_131054


namespace NUMINAMATH_CALUDE_sum_of_roots_when_product_is_24_l1310_131039

theorem sum_of_roots_when_product_is_24 (x₁ x₂ : ℝ) :
  (x₁ + 3) * (x₁ - 4) = 24 →
  (x₂ + 3) * (x₂ - 4) = 24 →
  x₁ + x₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_when_product_is_24_l1310_131039


namespace NUMINAMATH_CALUDE_work_completion_time_proof_l1310_131067

/-- Represents the time in days for a person to complete the work alone -/
structure WorkTime :=
  (days : ℚ)
  (days_pos : days > 0)

/-- Represents the combined work rate of multiple people -/
def combined_work_rate (work_times : List WorkTime) : ℚ :=
  work_times.map (λ wt => 1 / wt.days) |> List.sum

/-- The time required for the group to complete the work together -/
def group_work_time (work_times : List WorkTime) : ℚ :=
  1 / combined_work_rate work_times

theorem work_completion_time_proof 
  (david_time : WorkTime)
  (john_time : WorkTime)
  (mary_time : WorkTime)
  (h1 : david_time.days = 5)
  (h2 : john_time.days = 9)
  (h3 : mary_time.days = 7) :
  ⌈group_work_time [david_time, john_time, mary_time]⌉ = 3 := by
  sorry

#eval ⌈(315 : ℚ) / 143⌉

end NUMINAMATH_CALUDE_work_completion_time_proof_l1310_131067


namespace NUMINAMATH_CALUDE_right_triangle_leg_square_l1310_131090

theorem right_triangle_leg_square (a c : ℝ) (h1 : c = a + 2) : ∃ b : ℝ, b^2 = 2*a + 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_square_l1310_131090


namespace NUMINAMATH_CALUDE_right_triangle_identification_l1310_131022

theorem right_triangle_identification (a b c : ℝ) : 
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ 
  (a = 1 ∧ b = 2 ∧ c = Real.sqrt 3) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 4) ∨ 
  (a = 6 ∧ b = 8 ∧ c = 9) →
  (a^2 + b^2 = c^2 ↔ a = 1 ∧ b = 2 ∧ c = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_identification_l1310_131022


namespace NUMINAMATH_CALUDE_caleb_spent_correct_amount_l1310_131002

-- Define the given conditions
def total_burgers : ℕ := 50
def single_burger_cost : ℚ := 1
def double_burger_cost : ℚ := 1.5
def double_burgers_bought : ℕ := 37

-- Define the function to calculate the total cost
def total_cost : ℚ :=
  (double_burgers_bought * double_burger_cost) +
  ((total_burgers - double_burgers_bought) * single_burger_cost)

-- Theorem to prove
theorem caleb_spent_correct_amount :
  total_cost = 68.5 := by sorry

end NUMINAMATH_CALUDE_caleb_spent_correct_amount_l1310_131002


namespace NUMINAMATH_CALUDE_gcd_problem_l1310_131024

theorem gcd_problem (h : Nat.Prime 101) :
  Nat.gcd (101^6 + 1) (3 * 101^6 + 101^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1310_131024


namespace NUMINAMATH_CALUDE_remainder_problem_l1310_131059

theorem remainder_problem (m : ℤ) (h : m % 24 = 23) : m % 288 = 23 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1310_131059


namespace NUMINAMATH_CALUDE_min_ratio_T2_T1_l1310_131055

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle in 2D space -/
structure Triangle :=
  (A B C : Point)

/-- Checks if a triangle is acute -/
def is_acute (t : Triangle) : Prop := sorry

/-- Calculates the area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Represents the altitude of a triangle -/
structure Altitude :=
  (base : Point) (foot : Point)

/-- Calculates the projection of a point onto a line -/
def project (p : Point) (l : Point × Point) : Point := sorry

/-- Calculates the area of T_1 as defined in the problem -/
def area_T1 (t : Triangle) (AD BE CF : Altitude) : ℝ := sorry

/-- Calculates the area of T_2 as defined in the problem -/
def area_T2 (t : Triangle) (AD BE CF : Altitude) : ℝ := sorry

/-- The main theorem: The ratio T_2/T_1 is always greater than or equal to 25 for any acute triangle -/
theorem min_ratio_T2_T1 (t : Triangle) (AD BE CF : Altitude) :
  is_acute t →
  (area_T2 t AD BE CF) / (area_T1 t AD BE CF) ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_min_ratio_T2_T1_l1310_131055


namespace NUMINAMATH_CALUDE_quadratic_two_roots_condition_l1310_131031

/-- 
For a quadratic equation x^2 - 2x + k = 0 to have two real roots, 
k must satisfy k ≤ 1
-/
theorem quadratic_two_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + k = 0 ∧ y^2 - 2*y + k = 0) →
  k ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_condition_l1310_131031


namespace NUMINAMATH_CALUDE_M_intersect_N_empty_l1310_131089

-- Define set M
def M : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- Define set N
def N : Set (ℝ × ℝ) := {p | p.2 = Real.exp p.1}

-- Theorem statement
theorem M_intersect_N_empty : M ∩ (N.image Prod.fst) = ∅ := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_empty_l1310_131089


namespace NUMINAMATH_CALUDE_sam_long_sleeve_shirts_l1310_131036

/-- Given information about Sam's shirts to wash -/
structure ShirtWashing where
  short_sleeve : ℕ
  washed : ℕ
  unwashed : ℕ

/-- The number of long sleeve shirts Sam had to wash -/
def long_sleeve_shirts (s : ShirtWashing) : ℕ :=
  s.washed + s.unwashed - s.short_sleeve

/-- Theorem stating the number of long sleeve shirts Sam had to wash -/
theorem sam_long_sleeve_shirts :
  ∀ s : ShirtWashing,
  s.short_sleeve = 40 →
  s.washed = 29 →
  s.unwashed = 34 →
  long_sleeve_shirts s = 23 := by
  sorry

end NUMINAMATH_CALUDE_sam_long_sleeve_shirts_l1310_131036


namespace NUMINAMATH_CALUDE_vector_simplification_1_vector_simplification_2_l1310_131079

variable {V : Type*} [AddCommGroup V]

-- Define vectors
variable (A B C D E O : V)

-- Define the vector operations
def vec (X Y : V) := Y - X

-- Theorem statements
theorem vector_simplification_1 :
  (vec B A - vec B C) - (vec E D - vec E C) = vec D A := by sorry

theorem vector_simplification_2 :
  (vec A C + vec B O + vec O A) - (vec D C - vec D O - vec O B) = 0 := by sorry

end NUMINAMATH_CALUDE_vector_simplification_1_vector_simplification_2_l1310_131079


namespace NUMINAMATH_CALUDE_smallest_cube_multiplier_l1310_131025

theorem smallest_cube_multiplier (n : ℕ) (h : n = 1512) :
  (∃ (y : ℕ), 49 * n = y^3) ∧
  (∀ (x : ℕ), x > 0 → x < 49 → ¬∃ (y : ℕ), x * n = y^3) :=
sorry

end NUMINAMATH_CALUDE_smallest_cube_multiplier_l1310_131025


namespace NUMINAMATH_CALUDE_ice_cream_sales_l1310_131045

def tuesday_sales : ℕ := 12000

def wednesday_sales : ℕ := 2 * tuesday_sales

def total_sales : ℕ := tuesday_sales + wednesday_sales

theorem ice_cream_sales : total_sales = 36000 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sales_l1310_131045


namespace NUMINAMATH_CALUDE_total_spent_on_decks_l1310_131082

/-- The cost of a trick deck in dollars -/
def deck_cost : ℝ := 8

/-- The discount rate for buying 5 or more decks -/
def discount_rate : ℝ := 0.1

/-- The number of decks Victor bought -/
def victor_decks : ℕ := 6

/-- The number of decks Alice bought -/
def alice_decks : ℕ := 4

/-- The number of decks Bob bought -/
def bob_decks : ℕ := 3

/-- The minimum number of decks to qualify for a discount -/
def discount_threshold : ℕ := 5

/-- Function to calculate the cost of decks with potential discount -/
def calculate_cost (num_decks : ℕ) : ℝ :=
  let base_cost := (num_decks : ℝ) * deck_cost
  if num_decks ≥ discount_threshold then
    base_cost * (1 - discount_rate)
  else
    base_cost

/-- Theorem stating the total amount spent on trick decks -/
theorem total_spent_on_decks : 
  calculate_cost victor_decks + calculate_cost alice_decks + calculate_cost bob_decks = 99.20 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_on_decks_l1310_131082


namespace NUMINAMATH_CALUDE_opposite_abs_neg_five_l1310_131012

theorem opposite_abs_neg_five : -(abs (-5)) = -5 := by sorry

end NUMINAMATH_CALUDE_opposite_abs_neg_five_l1310_131012


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1310_131006

theorem inequality_equivalence (p : ℝ) (hp : p > 0) : 
  (∀ x : ℝ, 0 < x ∧ x < π / 2 → (1 / Real.sin x ^ 2) + (p / Real.cos x ^ 2) ≥ 9) ↔ 
  p ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1310_131006


namespace NUMINAMATH_CALUDE_total_carrots_is_nine_l1310_131086

-- Define the number of carrots grown by Sandy
def sandy_carrots : ℕ := 6

-- Define the number of carrots grown by Sam
def sam_carrots : ℕ := 3

-- Define the total number of carrots
def total_carrots : ℕ := sandy_carrots + sam_carrots

-- Theorem stating that the total number of carrots is 9
theorem total_carrots_is_nine : total_carrots = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_carrots_is_nine_l1310_131086


namespace NUMINAMATH_CALUDE_cone_height_l1310_131020

/-- The height of a cone given its lateral surface properties -/
theorem cone_height (r l : ℝ) (h : r > 0) (h' : l > 0) : 
  (l = 3) → (2 * Real.pi * r = 2 * Real.pi / 3 * 3) → 
  Real.sqrt (l^2 - r^2) = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_cone_height_l1310_131020


namespace NUMINAMATH_CALUDE_arun_remaining_work_days_arun_remaining_work_days_proof_l1310_131011

-- Define the work rates and time
def arun_tarun_rate : ℚ := 1 / 10
def arun_rate : ℚ := 1 / 60
def initial_work_days : ℕ := 4
def total_work : ℚ := 1

-- Theorem statement
theorem arun_remaining_work_days : ℕ :=
  let remaining_work : ℚ := total_work - (arun_tarun_rate * initial_work_days)
  let arun_remaining_days : ℚ := remaining_work / arun_rate
  36

-- Proof
theorem arun_remaining_work_days_proof :
  arun_remaining_work_days = 36 := by
  sorry

end NUMINAMATH_CALUDE_arun_remaining_work_days_arun_remaining_work_days_proof_l1310_131011


namespace NUMINAMATH_CALUDE_root_square_value_l1310_131047

theorem root_square_value (x₁ x₂ : ℂ) : 
  x₁ ≠ x₂ →
  (x₁ - 1)^2 = -3 →
  (x₂ - 1)^2 = -3 →
  x₁ = 1 - Complex.I * Real.sqrt 3 →
  x₂^2 = -2 + 2 * Complex.I * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_root_square_value_l1310_131047


namespace NUMINAMATH_CALUDE_credibility_is_97_5_percent_l1310_131081

/-- Critical values table -/
def critical_values : List (Float × Float) := [
  (0.15, 2.072),
  (0.10, 2.706),
  (0.05, 3.841),
  (0.025, 5.024),
  (0.010, 6.635),
  (0.001, 10.828)
]

/-- The calculated K^2 value -/
def K_squared : Float := 6.109

/-- Function to determine credibility based on K^2 value and critical values table -/
def determine_credibility (K_sq : Float) (crit_vals : List (Float × Float)) : Float :=
  let lower_bound := crit_vals.find? (fun (p, k) => K_sq > k)
  let upper_bound := crit_vals.find? (fun (p, k) => K_sq ≤ k)
  match lower_bound, upper_bound with
  | some (p_lower, _), some (p_upper, _) => 100 * (1 - p_lower)
  | _, _ => 0  -- Default case if bounds are not found

/-- Theorem stating the credibility of the relationship -/
theorem credibility_is_97_5_percent :
  determine_credibility K_squared critical_values = 97.5 :=
sorry

end NUMINAMATH_CALUDE_credibility_is_97_5_percent_l1310_131081


namespace NUMINAMATH_CALUDE_triangle_reciprocal_sum_l1310_131051

/-- Given a triangle with sides a, b, c, semiperimeter p, inradius r, and circumradius R,
    prove that 1/ab + 1/bc + 1/ac = 1/(2rR) -/
theorem triangle_reciprocal_sum (a b c p r R : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ p > 0 ∧ r > 0 ∧ R > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_semiperimeter : p = (a + b + c) / 2)
  (h_inradius : r = (a * b * c) / (4 * p))
  (h_circumradius : R = (a * b * c) / (4 * (p - a) * (p - b) * (p - c))) :
  1 / (a * b) + 1 / (b * c) + 1 / (a * c) = 1 / (2 * r * R) := by
  sorry

end NUMINAMATH_CALUDE_triangle_reciprocal_sum_l1310_131051


namespace NUMINAMATH_CALUDE_square_of_negative_two_x_squared_l1310_131053

theorem square_of_negative_two_x_squared (x : ℝ) : (-2 * x^2)^2 = 4 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_two_x_squared_l1310_131053


namespace NUMINAMATH_CALUDE_melody_reading_pages_l1310_131038

theorem melody_reading_pages (science civics chinese : ℕ) (total_tomorrow : ℕ) (english : ℕ) : 
  science = 16 → 
  civics = 8 → 
  chinese = 12 → 
  total_tomorrow = 14 → 
  (english / 4 + science / 4 + civics / 4 + chinese / 4 : ℚ) = total_tomorrow → 
  english = 20 := by
sorry

end NUMINAMATH_CALUDE_melody_reading_pages_l1310_131038


namespace NUMINAMATH_CALUDE_even_operations_l1310_131061

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem even_operations (n : ℤ) (h : is_even n) :
  (is_even (n + 4)) ∧ (is_even (n - 6)) ∧ (is_even (n * 8)) := by
  sorry

end NUMINAMATH_CALUDE_even_operations_l1310_131061


namespace NUMINAMATH_CALUDE_prob_third_batch_value_l1310_131007

/-- Represents a batch of parts -/
structure Batch :=
  (total : ℕ)
  (standard : ℕ)
  (h : standard ≤ total)

/-- Represents the experiment of selecting two standard parts from a batch -/
def select_two_standard (b : Batch) : ℚ :=
  (b.standard : ℚ) / b.total * ((b.standard - 1) : ℚ) / (b.total - 1)

/-- The probability of selecting the third batch given that two standard parts were selected -/
def prob_third_batch (b1 b2 b3 : Batch) : ℚ :=
  let p1 := select_two_standard b1
  let p2 := select_two_standard b2
  let p3 := select_two_standard b3
  p3 / (p1 + p2 + p3)

theorem prob_third_batch_value :
  let b1 : Batch := ⟨30, 20, by norm_num⟩
  let b2 : Batch := ⟨30, 15, by norm_num⟩
  let b3 : Batch := ⟨30, 10, by norm_num⟩
  prob_third_batch b1 b2 b3 = 3 / 68 := by
  sorry

end NUMINAMATH_CALUDE_prob_third_batch_value_l1310_131007


namespace NUMINAMATH_CALUDE_f_decreasing_interval_f_extremum_at_3_l1310_131058

/-- The function f(x) = 2x³ - 15x² + 36x - 24 -/
def f (x : ℝ) : ℝ := 2 * x^3 - 15 * x^2 + 36 * x - 24

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 6 * x^2 - 30 * x + 36

/-- Theorem stating that the decreasing interval of f is (2, 3) -/
theorem f_decreasing_interval :
  ∀ x : ℝ, (2 < x ∧ x < 3) ↔ (f' x < 0) :=
sorry

/-- Theorem stating that f has an extremum at x = 3 -/
theorem f_extremum_at_3 :
  f' 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_f_extremum_at_3_l1310_131058


namespace NUMINAMATH_CALUDE_even_function_max_value_l1310_131043

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_max_value
  (f : ℝ → ℝ)
  (h_even : IsEven f)
  (h_max : ∀ x ∈ Set.Icc (-2) (-1), f x ≤ -2)
  (h_attains : ∃ x ∈ Set.Icc (-2) (-1), f x = -2) :
  (∀ x ∈ Set.Icc 1 2, f x ≤ -2) ∧ (∃ x ∈ Set.Icc 1 2, f x = -2) :=
sorry

end NUMINAMATH_CALUDE_even_function_max_value_l1310_131043


namespace NUMINAMATH_CALUDE_area_of_triangle_FYG_l1310_131003

theorem area_of_triangle_FYG (EF GH : ℝ) (area_EFGH : ℝ) (angle_E : ℝ) :
  EF = 15 →
  GH = 25 →
  area_EFGH = 400 →
  angle_E = 30 * π / 180 →
  ∃ (area_FYG : ℝ), area_FYG = 240 - 45 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_area_of_triangle_FYG_l1310_131003


namespace NUMINAMATH_CALUDE_find_x_l1310_131023

theorem find_x (x : ℕ+) 
  (n : ℤ) (h_n : n = x.val^2 + 2*x.val + 17)
  (d : ℤ) (h_d : d = 2*x.val + 5)
  (h_div : n = d * x.val + 7) : 
  x.val = 2 := by
sorry

end NUMINAMATH_CALUDE_find_x_l1310_131023


namespace NUMINAMATH_CALUDE_f_of_5_eq_19_l1310_131075

/-- Given f(x) = (7x + 3) / (x - 3), prove that f(5) = 19 -/
theorem f_of_5_eq_19 : 
  let f : ℝ → ℝ := λ x ↦ (7 * x + 3) / (x - 3)
  f 5 = 19 := by
  sorry

end NUMINAMATH_CALUDE_f_of_5_eq_19_l1310_131075


namespace NUMINAMATH_CALUDE_angle_measure_proof_l1310_131093

theorem angle_measure_proof : Real.arccos (Real.sin (19 * π / 180)) = 71 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l1310_131093


namespace NUMINAMATH_CALUDE_weight_of_b_l1310_131018

/-- Given three weights a, b, and c, prove that b equals 60 when:
    1. The average of a, b, and c is 60.
    2. The average of a and b is 70.
    3. The average of b and c is 50. -/
theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 60)
  (h2 : (a + b) / 2 = 70)
  (h3 : (b + c) / 2 = 50) : 
  b = 60 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l1310_131018


namespace NUMINAMATH_CALUDE_inequality_proof_l1310_131050

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 3 / (1 + a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1310_131050


namespace NUMINAMATH_CALUDE_M_subset_N_l1310_131077

def M : Set ℝ := {x | ∃ k : ℤ, x = (k * Real.pi / 4) + (Real.pi / 4)}
def N : Set ℝ := {x | ∃ k : ℤ, x = (k * Real.pi / 8) - (Real.pi / 4)}

theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l1310_131077


namespace NUMINAMATH_CALUDE_circus_tent_capacity_l1310_131071

/-- The number of sections in the circus tent -/
def num_sections : ℕ := 4

/-- The capacity of each section in the circus tent -/
def section_capacity : ℕ := 246

/-- The total capacity of the circus tent -/
def total_capacity : ℕ := num_sections * section_capacity

theorem circus_tent_capacity : total_capacity = 984 := by
  sorry

end NUMINAMATH_CALUDE_circus_tent_capacity_l1310_131071


namespace NUMINAMATH_CALUDE_pet_store_total_l1310_131009

/-- The number of dogs for sale in the pet store -/
def num_dogs : ℕ := 12

/-- The number of cats for sale in the pet store -/
def num_cats : ℕ := num_dogs / 3

/-- The number of birds for sale in the pet store -/
def num_birds : ℕ := 4 * num_dogs

/-- The number of fish for sale in the pet store -/
def num_fish : ℕ := 5 * num_dogs

/-- The number of reptiles for sale in the pet store -/
def num_reptiles : ℕ := 2 * num_dogs

/-- The number of rodents for sale in the pet store -/
def num_rodents : ℕ := num_dogs

/-- The total number of animals for sale in the pet store -/
def total_animals : ℕ := num_dogs + num_cats + num_birds + num_fish + num_reptiles + num_rodents

theorem pet_store_total : total_animals = 160 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_total_l1310_131009


namespace NUMINAMATH_CALUDE_family_age_calculation_l1310_131062

theorem family_age_calculation (initial_members : ℕ) (initial_avg_age : ℝ) 
  (current_members : ℕ) (current_avg_age : ℝ) (baby_age : ℝ) : ℝ :=
by
  -- Define the conditions
  have h1 : initial_members = 5 := by sorry
  have h2 : initial_avg_age = 17 := by sorry
  have h3 : current_members = 6 := by sorry
  have h4 : current_avg_age = 17 := by sorry
  have h5 : baby_age = 2 := by sorry

  -- Define the function to calculate the time elapsed
  let time_elapsed := 
    (current_members * current_avg_age - initial_members * initial_avg_age - baby_age) / 
    (initial_members : ℝ)

  -- Prove that the time elapsed is 3 years
  have : time_elapsed = 3 := by sorry

  -- Return the result
  exact time_elapsed

end NUMINAMATH_CALUDE_family_age_calculation_l1310_131062


namespace NUMINAMATH_CALUDE_equation_roots_l1310_131042

theorem equation_roots (m : ℝ) : 
  (∃! x : ℝ, (m - 2) * x^2 - 2 * (m - 1) * x + m = 0) → 
  (∃ x : ℝ, ∀ y : ℝ, m * y^2 - (m + 2) * y + (4 - m) = 0 ↔ y = x) := by
  sorry

end NUMINAMATH_CALUDE_equation_roots_l1310_131042


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_deg_has_20_sides_l1310_131041

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides. -/
theorem regular_polygon_exterior_angle_18_deg_has_20_sides :
  ∀ n : ℕ, n > 0 →
  (360 : ℝ) / n = 18 →
  n = 20 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_deg_has_20_sides_l1310_131041


namespace NUMINAMATH_CALUDE_value_of_s_l1310_131095

theorem value_of_s (a b c w s p : ℕ) 
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ w ≠ 0 ∧ s ≠ 0 ∧ p ≠ 0)
  (h2 : a ≠ b ∧ a ≠ c ∧ a ≠ w ∧ a ≠ s ∧ a ≠ p)
  (h3 : b ≠ c ∧ b ≠ w ∧ b ≠ s ∧ b ≠ p)
  (h4 : c ≠ w ∧ c ≠ s ∧ c ≠ p)
  (h5 : w ≠ s ∧ w ≠ p)
  (h6 : s ≠ p)
  (eq1 : a + b = w)
  (eq2 : w + c = s)
  (eq3 : s + a = p)
  (eq4 : b + c + p = 16) : 
  s = 8 := by
sorry

end NUMINAMATH_CALUDE_value_of_s_l1310_131095


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l1310_131097

/-- The function f(x) = a^(2-x) + 2 always passes through the point (2, 3) for all a > 0 and a ≠ 1 -/
theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(2 - x) + 2
  f 2 = 3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l1310_131097


namespace NUMINAMATH_CALUDE_faye_pencils_l1310_131084

/-- The number of pencils Faye has in all sets -/
def total_pencils (rows_per_set : ℕ) (pencils_per_row : ℕ) (num_sets : ℕ) : ℕ :=
  rows_per_set * pencils_per_row * num_sets

/-- Theorem stating the total number of pencils Faye has -/
theorem faye_pencils :
  total_pencils 14 11 3 = 462 := by
  sorry

end NUMINAMATH_CALUDE_faye_pencils_l1310_131084


namespace NUMINAMATH_CALUDE_intersection_point_coordinates_l1310_131073

/-- Given a triangle ABC with points F on BC and G on AC, prove that the intersection Q of BG and AF
    can be expressed as a linear combination of A, B, and C. -/
theorem intersection_point_coordinates (A B C F G Q : ℝ × ℝ) : 
  (∃ t : ℝ, F = (1 - t) • B + t • C ∧ t = 1/3) →  -- F lies on BC with BF:FC = 2:1
  (∃ s : ℝ, G = (1 - s) • A + s • C ∧ s = 3/5) →  -- G lies on AC with AG:GC = 3:2
  (∃ u v : ℝ, Q = (1 - u) • B + u • G ∧ Q = (1 - v) • A + v • F) →  -- Q is intersection of BG and AF
  Q = (2/5) • A + (1/3) • B + (4/9) • C := by sorry

end NUMINAMATH_CALUDE_intersection_point_coordinates_l1310_131073


namespace NUMINAMATH_CALUDE_split_bill_example_l1310_131029

/-- Calculates the amount each person should pay when splitting a bill equally -/
def split_bill (num_people : ℕ) (num_bread : ℕ) (bread_price : ℕ) (num_hotteok : ℕ) (hotteok_price : ℕ) : ℕ :=
  ((num_bread * bread_price + num_hotteok * hotteok_price) / num_people)

/-- Theorem stating that given the conditions, each person should pay 1650 won -/
theorem split_bill_example : split_bill 4 5 200 7 800 = 1650 := by
  sorry

end NUMINAMATH_CALUDE_split_bill_example_l1310_131029


namespace NUMINAMATH_CALUDE_disk_color_difference_l1310_131037

theorem disk_color_difference (total : ℕ) (blue_ratio yellow_ratio green_ratio : ℕ) : 
  total = 126 →
  blue_ratio = 3 →
  yellow_ratio = 7 →
  green_ratio = 8 →
  let ratio_sum := blue_ratio + yellow_ratio + green_ratio
  let blue_count := (blue_ratio * total) / ratio_sum
  let green_count := (green_ratio * total) / ratio_sum
  green_count - blue_count = 35 := by
sorry

end NUMINAMATH_CALUDE_disk_color_difference_l1310_131037


namespace NUMINAMATH_CALUDE_max_gcd_11n_plus_3_6n_plus_1_l1310_131014

theorem max_gcd_11n_plus_3_6n_plus_1 :
  ∃ (k : ℕ), k > 0 ∧ gcd (11 * k + 3) (6 * k + 1) = 7 ∧
  ∀ (n : ℕ), n > 0 → gcd (11 * n + 3) (6 * n + 1) ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_11n_plus_3_6n_plus_1_l1310_131014


namespace NUMINAMATH_CALUDE_linear_function_intersection_l1310_131057

theorem linear_function_intersection (k : ℝ) : 
  (∃ x : ℝ, k * x + 3 = 0 ∧ x^2 = 36) → (k = 1/2 ∨ k = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_intersection_l1310_131057


namespace NUMINAMATH_CALUDE_train_length_calculation_l1310_131080

/-- The length of a train given its speed and time to cross a pole. -/
def train_length (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: A train with speed 53.99999999999999 m/s that crosses a pole in 20 seconds has a length of 1080 meters. -/
theorem train_length_calculation :
  let speed : ℝ := 53.99999999999999
  let time : ℝ := 20
  train_length speed time = 1080 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1310_131080


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l1310_131027

open Real

theorem triangle_ABC_properties (A B C a b c : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  2 * sin A * sin C * (1 / (tan A * tan C) - 1) = -1 ∧
  a + c = 3 * sqrt 3 / 2 ∧
  b = sqrt 3 →
  B = π / 3 ∧
  (1 / 2) * a * c * sin B = 5 * sqrt 3 / 16 := by
sorry


end NUMINAMATH_CALUDE_triangle_ABC_properties_l1310_131027
