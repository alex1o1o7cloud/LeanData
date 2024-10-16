import Mathlib

namespace NUMINAMATH_CALUDE_square_field_area_l116_11635

/-- The area of a square field with side length 5 meters is 25 square meters. -/
theorem square_field_area :
  let side_length : ℝ := 5
  let area : ℝ := side_length ^ 2
  area = 25 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l116_11635


namespace NUMINAMATH_CALUDE_girls_on_playground_l116_11678

theorem girls_on_playground (total_children boys : ℕ) 
  (h1 : total_children = 63) 
  (h2 : boys = 35) : 
  total_children - boys = 28 := by
sorry

end NUMINAMATH_CALUDE_girls_on_playground_l116_11678


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l116_11697

theorem quadratic_no_real_roots : 
  ∀ x : ℝ, 2 * x^2 - 5 * x + 6 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l116_11697


namespace NUMINAMATH_CALUDE_unique_square_multiple_of_five_in_range_l116_11669

theorem unique_square_multiple_of_five_in_range : 
  ∃! x : ℕ, 
    (∃ n : ℕ, x = n^2) ∧ 
    (x % 5 = 0) ∧ 
    (50 < x) ∧ 
    (x < 120) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_square_multiple_of_five_in_range_l116_11669


namespace NUMINAMATH_CALUDE_system_solution_l116_11653

theorem system_solution (x y k : ℝ) 
  (eq1 : x + 2*y = 6 + 3*k) 
  (eq2 : 2*x + y = 3*k) : 
  2*y - 2*x = 12 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l116_11653


namespace NUMINAMATH_CALUDE_function_maximum_value_l116_11683

theorem function_maximum_value (x : ℝ) (h : x < 0) : 
  ∃ (M : ℝ), M = -4 ∧ ∀ y, y < 0 → x + 4/x ≤ M :=
sorry

end NUMINAMATH_CALUDE_function_maximum_value_l116_11683


namespace NUMINAMATH_CALUDE_area_of_ω_l116_11658

-- Define the circle ω
def ω : Set (ℝ × ℝ) := sorry

-- Define points A and B
def A : ℝ × ℝ := (4, 15)
def B : ℝ × ℝ := (12, 9)

-- State that A and B lie on ω
axiom A_on_ω : A ∈ ω
axiom B_on_ω : B ∈ ω

-- Define the tangent lines at A and B
def tangent_A : Set (ℝ × ℝ) := sorry
def tangent_B : Set (ℝ × ℝ) := sorry

-- State that the tangent lines intersect at a point on the x-axis
axiom tangents_intersect_x_axis : ∃ x : ℝ, (x, 0) ∈ tangent_A ∩ tangent_B

-- Define the area of a circle
def circle_area (c : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_of_ω : circle_area ω = 306 * Real.pi := sorry

end NUMINAMATH_CALUDE_area_of_ω_l116_11658


namespace NUMINAMATH_CALUDE_alicia_tax_payment_l116_11648

/-- Calculates the local tax amount in cents given an hourly wage in dollars and a tax rate as a percentage. -/
def local_tax_cents (hourly_wage : ℚ) (tax_rate : ℚ) : ℚ :=
  hourly_wage * 100 * (tax_rate / 100)

/-- Proves that Alicia's local tax payment is 50 cents per hour. -/
theorem alicia_tax_payment :
  local_tax_cents 25 2 = 50 := by
  sorry

#eval local_tax_cents 25 2

end NUMINAMATH_CALUDE_alicia_tax_payment_l116_11648


namespace NUMINAMATH_CALUDE_fish_cost_is_80_l116_11645

/-- The cost of fish in pesos per kilogram -/
def fish_cost : ℕ := 80

/-- The cost of pork in pesos per kilogram -/
def pork_cost : ℕ := 105

/-- Theorem stating that the cost of fish is 80 pesos per kilogram -/
theorem fish_cost_is_80 :
  (530 = 4 * fish_cost + 2 * pork_cost) →
  (875 = 7 * fish_cost + 3 * pork_cost) →
  fish_cost = 80 := by
  sorry

end NUMINAMATH_CALUDE_fish_cost_is_80_l116_11645


namespace NUMINAMATH_CALUDE_complex_absolute_value_l116_11619

theorem complex_absolute_value (z : ℂ) : 
  (z + 1) / (z - 2) = 1 - 3*I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l116_11619


namespace NUMINAMATH_CALUDE_eliza_initial_rings_l116_11693

/-- The number of ornamental rings Eliza initially bought -/
def initial_rings : ℕ := 100

/-- The total stock after Eliza's purchase -/
def total_stock : ℕ := 3 * initial_rings

/-- The remaining stock after selling 3/4 of the total -/
def remaining_after_sale : ℕ := (total_stock * 1) / 4

/-- The stock after mother's purchase -/
def stock_after_mother_purchase : ℕ := remaining_after_sale + 300

/-- The final stock -/
def final_stock : ℕ := stock_after_mother_purchase - 150

theorem eliza_initial_rings :
  final_stock = 225 :=
by sorry

end NUMINAMATH_CALUDE_eliza_initial_rings_l116_11693


namespace NUMINAMATH_CALUDE_ball_drawing_probabilities_l116_11682

/-- Represents the ball drawing process with 3 red and 2 white balls initially -/
structure BallDrawing where
  redBalls : ℕ := 3
  whiteBalls : ℕ := 2

/-- Probability of an event in the ball drawing process -/
def probability (event : Bool) : ℚ := sorry

/-- Event of drawing a red ball on the first draw -/
def A₁ : Bool := sorry

/-- Event of drawing a red ball on the second draw -/
def A₂ : Bool := sorry

/-- Event of drawing a white ball on the first draw -/
def B₁ : Bool := sorry

/-- Event of drawing a white ball on the second draw -/
def B₂ : Bool := sorry

/-- Event of drawing balls of the same color on both draws -/
def C : Bool := sorry

/-- Conditional probability of B₂ given A₁ -/
def conditionalProbability (B₂ A₁ : Bool) : ℚ := sorry

theorem ball_drawing_probabilities (bd : BallDrawing) :
  conditionalProbability B₂ A₁ = 3/5 ∧
  probability (B₁ ∧ A₂) = 8/25 ∧
  probability C = 8/25 := by sorry

end NUMINAMATH_CALUDE_ball_drawing_probabilities_l116_11682


namespace NUMINAMATH_CALUDE_triangle_area_l116_11692

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the area of the triangle is (18 + 8√3) / 25 when a = √3, c = 8/5, and A = π/3 -/
theorem triangle_area (a b c A B C : ℝ) : 
  a = Real.sqrt 3 →
  c = 8 / 5 →
  A = π / 3 →
  (1 / 2) * a * c * Real.sin B = (18 + 8 * Real.sqrt 3) / 25 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l116_11692


namespace NUMINAMATH_CALUDE_fewer_buses_than_cars_l116_11617

theorem fewer_buses_than_cars (ratio_buses_to_cars : ℚ) (num_cars : ℕ) : 
  ratio_buses_to_cars = 1 / 13 → num_cars = 65 → num_cars - (num_cars / 13 : ℕ) = 60 := by
  sorry

end NUMINAMATH_CALUDE_fewer_buses_than_cars_l116_11617


namespace NUMINAMATH_CALUDE_total_batteries_used_l116_11650

theorem total_batteries_used (flashlight_batteries : ℕ) (toy_batteries : ℕ) (controller_batteries : ℕ)
  (h1 : flashlight_batteries = 2)
  (h2 : toy_batteries = 15)
  (h3 : controller_batteries = 2) :
  flashlight_batteries + toy_batteries + controller_batteries = 19 := by
  sorry

end NUMINAMATH_CALUDE_total_batteries_used_l116_11650


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l116_11681

theorem nested_fraction_equality : 1 + (1 / (1 + (1 / (1 + (1 / (1 + 2)))))) = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l116_11681


namespace NUMINAMATH_CALUDE_mike_earnings_l116_11632

/-- Calculates the earnings from selling working video games --/
def calculate_earnings (total_games : ℕ) (non_working_games : ℕ) (price_per_game : ℚ) (tax_rate : ℚ) : ℚ :=
  let working_games := total_games - non_working_games
  let total_revenue := working_games * price_per_game
  total_revenue

/-- Proves that Mike's earnings from selling working video games is $72 --/
theorem mike_earnings : 
  calculate_earnings 20 11 8 (12/100) = 72 := by
  sorry

end NUMINAMATH_CALUDE_mike_earnings_l116_11632


namespace NUMINAMATH_CALUDE_fifi_hangers_l116_11680

theorem fifi_hangers (total green blue yellow pink : ℕ) : 
  total = 16 →
  green = 4 →
  blue = green - 1 →
  yellow = blue - 1 →
  total = green + blue + yellow + pink →
  pink = 7 := by
sorry

end NUMINAMATH_CALUDE_fifi_hangers_l116_11680


namespace NUMINAMATH_CALUDE_unit_digit_7_2023_l116_11636

def unit_digit (n : ℕ) : ℕ := n % 10

def power_7_unit_digit (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | _ => 0  -- This case should never occur

theorem unit_digit_7_2023 : unit_digit (7^2023) = 3 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_7_2023_l116_11636


namespace NUMINAMATH_CALUDE_managers_salary_l116_11673

theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (salary_increase : ℝ) :
  num_employees = 20 →
  avg_salary = 1500 →
  salary_increase = 1000 →
  let total_salary := num_employees * avg_salary
  let new_avg_salary := avg_salary + salary_increase
  let new_total_salary := (num_employees + 1) * new_avg_salary
  new_total_salary - total_salary = 22500 := by
  sorry

end NUMINAMATH_CALUDE_managers_salary_l116_11673


namespace NUMINAMATH_CALUDE_amar_car_distance_l116_11637

/-- Given that Amar's speed to car's speed ratio is 18:48, prove that when Amar covers 675 meters,
    the car covers 1800 meters in the same time. -/
theorem amar_car_distance (amar_speed car_speed : ℝ) (amar_distance : ℝ) :
  amar_speed / car_speed = 18 / 48 →
  amar_distance = 675 →
  ∃ car_distance : ℝ, car_distance = 1800 ∧ amar_distance / car_distance = amar_speed / car_speed :=
by sorry

end NUMINAMATH_CALUDE_amar_car_distance_l116_11637


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l116_11605

/-- The total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) : ℝ :=
  let firstRebound := initialHeight * reboundFactor
  let secondRebound := firstRebound * reboundFactor
  initialHeight + firstRebound + (initialHeight - firstRebound) + 
  secondRebound + (firstRebound - secondRebound) + secondRebound

/-- Theorem: The total distance traveled by a ball dropped from 80 cm 
    with a 50% rebound factor is 200 cm when it touches the floor for the third time -/
theorem ball_bounce_distance :
  totalDistance 80 0.5 = 200 := by
  sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l116_11605


namespace NUMINAMATH_CALUDE_product_inequality_l116_11686

theorem product_inequality (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h₁ : a₁ > 1) (h₂ : a₂ > 1) (h₃ : a₃ > 1) (h₄ : a₄ > 1) (h₅ : a₅ > 1) : 
  16 * (a₁ * a₂ * a₃ * a₄ * a₅ + 1) ≥ (1 + a₁) * (1 + a₂) * (1 + a₃) * (1 + a₄) * (1 + a₅) := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l116_11686


namespace NUMINAMATH_CALUDE_units_digit_G_1000_l116_11694

def G (n : ℕ) : ℕ := 3^(2^n) + 2

theorem units_digit_G_1000 : G 1000 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_G_1000_l116_11694


namespace NUMINAMATH_CALUDE_prob_three_two_digit_dice_l116_11657

-- Define the number of dice
def num_dice : ℕ := 6

-- Define the number of sides on each die
def num_sides : ℕ := 12

-- Define the number of two-digit outcomes on a single die
def two_digit_outcomes : ℕ := 3

-- Define the probability of rolling a two-digit number on a single die
def prob_two_digit : ℚ := two_digit_outcomes / num_sides

-- Define the probability of rolling a one-digit number on a single die
def prob_one_digit : ℚ := 1 - prob_two_digit

-- Define the number of dice we want to show two-digit numbers
def target_two_digit : ℕ := 3

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- State the theorem
theorem prob_three_two_digit_dice :
  (binomial num_dice target_two_digit : ℚ) * prob_two_digit ^ target_two_digit * prob_one_digit ^ (num_dice - target_two_digit) = 135 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_two_digit_dice_l116_11657


namespace NUMINAMATH_CALUDE_simplify_fraction_l116_11612

theorem simplify_fraction (b : ℝ) (h : b ≠ 1) :
  (b - 1) / (b + b / (b - 1)) = (b - 1)^2 / b^2 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l116_11612


namespace NUMINAMATH_CALUDE_complement_of_union_equals_zero_five_l116_11623

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 4}

theorem complement_of_union_equals_zero_five :
  (U \ (A ∪ B)) = {0, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_zero_five_l116_11623


namespace NUMINAMATH_CALUDE_remainder_calculation_l116_11664

theorem remainder_calculation (L S R : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 1620)
  (h3 : L = 6 * S + R) : 
  R = 90 := by
sorry

end NUMINAMATH_CALUDE_remainder_calculation_l116_11664


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l116_11639

/-- An ellipse with semi-major axis 2 and semi-minor axis √b -/
def Ellipse (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 4) + (p.2^2 / b) = 1}

/-- A line with slope m passing through (0, 1) -/
def Line (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * p.1 + 1}

/-- The theorem statement -/
theorem ellipse_line_intersection (b : ℝ) :
  (∀ m : ℝ, (Ellipse b ∩ Line m).Nonempty) →
  b ∈ Set.Icc 1 4 ∪ Set.Ioi 4 := by
  sorry


end NUMINAMATH_CALUDE_ellipse_line_intersection_l116_11639


namespace NUMINAMATH_CALUDE_power_divisibility_implies_equality_l116_11604

theorem power_divisibility_implies_equality (m n : ℕ) : 
  m > 1 → n > 1 → (4^m - 1) % n = 0 → (n - 1) % (2^m) = 0 → n = 2^m + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_divisibility_implies_equality_l116_11604


namespace NUMINAMATH_CALUDE_inequality_implies_a_geq_two_l116_11602

theorem inequality_implies_a_geq_two (a : ℝ) :
  (∀ x y : ℝ, x^2 + 2*x + a ≥ -y^2 - 2*y) → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_a_geq_two_l116_11602


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l116_11667

def f (x : ℝ) := x^3 - x - 1

theorem root_exists_in_interval :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l116_11667


namespace NUMINAMATH_CALUDE_distance_to_origin_l116_11616

theorem distance_to_origin (x y : ℝ) (h1 : y = 14) 
  (h2 : Real.sqrt ((x - 1)^2 + (y - 8)^2) = 8) (h3 : x > 1) :
  Real.sqrt (x^2 + y^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l116_11616


namespace NUMINAMATH_CALUDE_min_pizzas_to_break_even_l116_11603

def car_cost : ℕ := 6000
def earning_per_pizza : ℕ := 12
def gas_cost_per_pizza : ℕ := 4

theorem min_pizzas_to_break_even :
  let net_earning_per_pizza := earning_per_pizza - gas_cost_per_pizza
  (∀ n : ℕ, n * net_earning_per_pizza < car_cost → n < 750) ∧
  750 * net_earning_per_pizza ≥ car_cost :=
by sorry

end NUMINAMATH_CALUDE_min_pizzas_to_break_even_l116_11603


namespace NUMINAMATH_CALUDE_number_calculation_l116_11655

theorem number_calculation (x : ℝ) (number : ℝ) : x = 4 ∧ number = 3 * x + 36 → number = 48 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l116_11655


namespace NUMINAMATH_CALUDE_unique_solution_l116_11691

theorem unique_solution : ∃! (n : ℕ), n > 0 ∧ 5^29 * 4^15 = 2 * n^29 :=
by
  use 10
  constructor
  · sorry -- Proof that 10 satisfies the equation
  · sorry -- Proof of uniqueness

#check unique_solution

end NUMINAMATH_CALUDE_unique_solution_l116_11691


namespace NUMINAMATH_CALUDE_parallelepiped_inequality_l116_11666

/-- Theorem: For any parallelepiped with sides a, b, c, and diagonal d,
    the sum of squares of the sides is greater than or equal to one-third of the square of the diagonal. -/
theorem parallelepiped_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_diagonal : d^2 = a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c) :
  a^2 + b^2 + c^2 ≥ (1/3) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_inequality_l116_11666


namespace NUMINAMATH_CALUDE_amount_with_r_l116_11649

theorem amount_with_r (total : ℝ) (p q r : ℝ) : 
  total = 9000 →
  p + q + r = total →
  r = (2/3) * (p + q) →
  r = 3600 := by
sorry

end NUMINAMATH_CALUDE_amount_with_r_l116_11649


namespace NUMINAMATH_CALUDE_triangle_minimum_product_l116_11679

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if 2c cos B = 2a + b and the area of the triangle is (√3/12)c, then ab ≥ 1/3 -/
theorem triangle_minimum_product (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  2 * c * Real.cos B = 2 * a + b →
  (1 / 2) * a * b * Real.sin C = (Real.sqrt 3 / 12) * c →
  a * b ≥ 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_minimum_product_l116_11679


namespace NUMINAMATH_CALUDE_school_population_l116_11615

theorem school_population (blind : ℕ) (deaf : ℕ) (other : ℕ) : 
  deaf = 3 * blind → 
  other = 2 * blind → 
  deaf = 180 → 
  blind + deaf + other = 360 :=
by
  sorry

end NUMINAMATH_CALUDE_school_population_l116_11615


namespace NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l116_11690

theorem sum_of_solutions_squared_equation (x : ℝ) :
  (∀ x, (x - 4)^2 = 16 → x = 0 ∨ x = 8) →
  (∃ a b, (a - 4)^2 = 16 ∧ (b - 4)^2 = 16 ∧ a + b = 8) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l116_11690


namespace NUMINAMATH_CALUDE_cow_inheritance_problem_l116_11674

theorem cow_inheritance_problem (x y : ℕ) (z : ℝ) 
  (h1 : x^2 = 10*y + z)
  (h2 : z < 10)
  (h3 : Odd y)
  (h4 : x^2 % 10 = 6) :
  (10 - z) / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cow_inheritance_problem_l116_11674


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l116_11624

theorem arithmetic_expression_equality : 8 + 15 / 3 - 4 * 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l116_11624


namespace NUMINAMATH_CALUDE_marble_probability_l116_11676

/-- Represents a box of marbles -/
structure MarbleBox where
  total : ℕ
  black : ℕ
  white : ℕ
  valid : black + white = total

/-- Represents two boxes of marbles -/
structure TwoBoxes where
  box1 : MarbleBox
  box2 : MarbleBox
  total36 : box1.total + box2.total = 36

/-- The probability of drawing a black marble from a box -/
def probBlack (box : MarbleBox) : ℚ :=
  box.black / box.total

/-- The probability of drawing a white marble from a box -/
def probWhite (box : MarbleBox) : ℚ :=
  box.white / box.total

theorem marble_probability (boxes : TwoBoxes)
    (h : probBlack boxes.box1 * probBlack boxes.box2 = 13/18) :
    probWhite boxes.box1 * probWhite boxes.box2 = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l116_11676


namespace NUMINAMATH_CALUDE_equation_implication_l116_11618

theorem equation_implication (x y z : ℝ) :
  1 / (y * z - x^2) + 1 / (z * x - y^2) + 1 / (x * y - z^2) = 0 →
  x / (y * z - x^2)^2 + y / (z * x - y^2)^2 + z / (x * y - z^2)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_implication_l116_11618


namespace NUMINAMATH_CALUDE_valid_plates_count_l116_11642

/-- Represents the set of valid characters for a license plate position -/
inductive PlateChar
| Letter
| Digit

/-- Represents a license plate -/
structure LicensePlate :=
  (first : PlateChar)
  (middle : PlateChar)
  (last : PlateChar)

/-- Checks if a license plate is valid according to the given conditions -/
def isValidPlate (plate : LicensePlate) : Prop :=
  plate.first = PlateChar.Letter ∧
  plate.last = PlateChar.Digit ∧
  plate.first ≠ plate.last

/-- The number of available letters -/
def numLetters : ℕ := 26

/-- The number of available digits -/
def numDigits : ℕ := 10

/-- Counts the number of valid license plates -/
def countValidPlates : ℕ := sorry

/-- Theorem stating that the number of valid license plates is 9360 -/
theorem valid_plates_count :
  countValidPlates = 9360 :=
sorry

end NUMINAMATH_CALUDE_valid_plates_count_l116_11642


namespace NUMINAMATH_CALUDE_compare_expressions_l116_11652

theorem compare_expressions (a : ℝ) : (a + 3) * (a - 5) < (a + 2) * (a - 4) := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_l116_11652


namespace NUMINAMATH_CALUDE_point_on_segment_vector_relation_l116_11651

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define points
variable (A B M O C D : V)

-- Define conditions
variable (h1 : M ∈ closedSegment A B)
variable (h2 : O ∉ line_through A B)
variable (h3 : C = 2 • O - A)  -- C is symmetric to A with respect to O
variable (h4 : D = 2 • C - B)  -- D is symmetric to B with respect to C
variable (x y : ℝ)
variable (h5 : O - M = x • (O - C) + y • (O - D))

-- Theorem statement
theorem point_on_segment_vector_relation :
  x + 3 * y = -1 :=
sorry

end NUMINAMATH_CALUDE_point_on_segment_vector_relation_l116_11651


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l116_11654

theorem fraction_to_decimal : (19 : ℚ) / (2^2 * 5^3) = (38 : ℚ) / 1000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l116_11654


namespace NUMINAMATH_CALUDE_product_of_common_ratios_l116_11614

/-- Given two nonconstant geometric sequences with different common ratios
    satisfying a specific equation, prove that the product of their common ratios is 9. -/
theorem product_of_common_ratios (x p r : ℝ) (hx : x ≠ 0) (hp : p ≠ 1) (hr : r ≠ 1) (hpr : p ≠ r) :
  3 * x * p^2 - 4 * x * r^2 = 5 * (3 * x * p - 4 * x * r) →
  p * r = 9 := by
sorry

end NUMINAMATH_CALUDE_product_of_common_ratios_l116_11614


namespace NUMINAMATH_CALUDE_red_yellow_peach_difference_l116_11628

theorem red_yellow_peach_difference (red_peaches yellow_peaches : ℕ) 
  (h1 : red_peaches = 19) 
  (h2 : yellow_peaches = 11) : 
  red_peaches - yellow_peaches = 8 := by
sorry

end NUMINAMATH_CALUDE_red_yellow_peach_difference_l116_11628


namespace NUMINAMATH_CALUDE_swimmer_speed_is_4_l116_11659

/-- The swimmer's speed in still water -/
def swimmer_speed : ℝ := 4

/-- The speed of the water current -/
def current_speed : ℝ := 1

/-- The time taken to swim against the current -/
def swim_time : ℝ := 2

/-- The distance swum against the current -/
def swim_distance : ℝ := 6

/-- Theorem stating that the swimmer's speed in still water is 4 km/h -/
theorem swimmer_speed_is_4 :
  swimmer_speed = 4 ∧
  current_speed = 1 ∧
  swim_time = 2 ∧
  swim_distance = 6 →
  swimmer_speed = swim_distance / swim_time + current_speed :=
by sorry

end NUMINAMATH_CALUDE_swimmer_speed_is_4_l116_11659


namespace NUMINAMATH_CALUDE_smallest_fraction_l116_11626

theorem smallest_fraction (x : ℝ) (h : x = 7) : 
  6 / (x + 1) < 6 / x ∧ 
  6 / (x + 1) < 6 / (x - 1) ∧ 
  6 / (x + 1) < x / 6 ∧ 
  6 / (x + 1) < (x + 1) / 6 := by
sorry

end NUMINAMATH_CALUDE_smallest_fraction_l116_11626


namespace NUMINAMATH_CALUDE_trig_identity_l116_11607

theorem trig_identity (α : Real) (h : Real.tan α = -1/2) :
  (Real.cos α - Real.sin α)^2 / Real.cos (2 * α) = 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l116_11607


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l116_11638

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Represents a binary number as a list of bits (least significant bit first) -/
def binary_1101 : List Bool := [true, false, true, true]
def binary_111 : List Bool := [true, true, true]
def binary_10001111 : List Bool := [true, true, true, true, false, false, false, true]

theorem binary_multiplication_theorem :
  (binary_to_decimal binary_1101) * (binary_to_decimal binary_111) =
  binary_to_decimal binary_10001111 ∧
  (binary_to_decimal binary_1101) * (binary_to_decimal binary_111) = 143 := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l116_11638


namespace NUMINAMATH_CALUDE_evaluate_expression_l116_11672

theorem evaluate_expression : 6 - 5 * (10 - (2 + 1)^2) * 3 = -9 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l116_11672


namespace NUMINAMATH_CALUDE_count_special_divisors_300_l116_11613

/-- The number of positive divisors of 300 not divisible by 5 or 3 -/
def count_special_divisors (n : ℕ) : ℕ :=
  (Finset.filter (fun d => d ∣ n ∧ ¬(5 ∣ d) ∧ ¬(3 ∣ d)) (Finset.range (n + 1))).card

/-- Theorem stating that the count of special divisors of 300 is 3 -/
theorem count_special_divisors_300 :
  count_special_divisors 300 = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_special_divisors_300_l116_11613


namespace NUMINAMATH_CALUDE_fuel_mixture_cost_l116_11608

/-- Represents the cost of the other liquid per gallon -/
def other_liquid_cost : ℝ := 3

/-- The total volume of the mixture in gallons -/
def total_volume : ℝ := 12

/-- The cost of the final fuel mixture per gallon -/
def final_fuel_cost : ℝ := 8

/-- The cost of oil per gallon -/
def oil_cost : ℝ := 15

/-- The volume of one of the liquids used in the mixture -/
def one_liquid_volume : ℝ := 7

theorem fuel_mixture_cost : 
  one_liquid_volume * other_liquid_cost + (total_volume - one_liquid_volume) * oil_cost = 
  total_volume * final_fuel_cost :=
sorry

end NUMINAMATH_CALUDE_fuel_mixture_cost_l116_11608


namespace NUMINAMATH_CALUDE_min_value_theorem_l116_11629

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_mn : m + n = 1) :
  (1 / m + 2 / n) ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l116_11629


namespace NUMINAMATH_CALUDE_jerry_video_games_l116_11689

theorem jerry_video_games (initial_games new_games : ℕ) : 
  initial_games = 7 → new_games = 2 → initial_games + new_games = 9 :=
by sorry

end NUMINAMATH_CALUDE_jerry_video_games_l116_11689


namespace NUMINAMATH_CALUDE_chess_games_ratio_l116_11627

theorem chess_games_ratio (total_games won_games : ℕ) 
  (h1 : total_games = 44)
  (h2 : won_games = 16) :
  let lost_games := total_games - won_games
  Nat.gcd lost_games won_games = 4 ∧ 
  lost_games / 4 = 7 ∧ 
  won_games / 4 = 4 := by
sorry

end NUMINAMATH_CALUDE_chess_games_ratio_l116_11627


namespace NUMINAMATH_CALUDE_b_profit_fraction_l116_11656

/-- The fraction of profit a partner receives in a business partnership -/
def profit_fraction (capital_fraction : ℚ) (months : ℕ) (total_capital_time : ℚ) : ℚ :=
  (capital_fraction * months) / total_capital_time

theorem b_profit_fraction :
  let a_capital_fraction : ℚ := 1/4
  let a_months : ℕ := 15
  let b_capital_fraction : ℚ := 3/4
  let b_months : ℕ := 10
  let total_capital_time : ℚ := a_capital_fraction * a_months + b_capital_fraction * b_months
  profit_fraction b_capital_fraction b_months total_capital_time = 2/3 := by
sorry

end NUMINAMATH_CALUDE_b_profit_fraction_l116_11656


namespace NUMINAMATH_CALUDE_farmers_wheat_cleaning_l116_11620

theorem farmers_wheat_cleaning (original_rate : ℕ) (new_rate : ℕ) (last_day_acres : ℕ) :
  original_rate = 80 →
  new_rate = original_rate + 10 →
  last_day_acres = 30 →
  ∃ (total_acres : ℕ) (planned_days : ℕ),
    total_acres = 480 ∧
    planned_days * original_rate = total_acres ∧
    (planned_days - 1) * new_rate + last_day_acres = total_acres :=
by
  sorry

end NUMINAMATH_CALUDE_farmers_wheat_cleaning_l116_11620


namespace NUMINAMATH_CALUDE_red_marbles_count_l116_11609

theorem red_marbles_count (green yellow red : ℕ) : 
  green + yellow + red > 0 →  -- Ensure the bag is not empty
  green = 3 * (red / 2) →     -- Ratio condition for green
  yellow = 4 * (red / 2) →    -- Ratio condition for yellow
  green + yellow = 63 →       -- Number of non-red marbles
  red = 18 := by
  sorry

end NUMINAMATH_CALUDE_red_marbles_count_l116_11609


namespace NUMINAMATH_CALUDE_base3_20121_equals_178_l116_11685

/-- Converts a base 3 number to base 10 -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base3_20121_equals_178 : 
  base3ToBase10 [2, 0, 1, 2, 1] = 178 := by
  sorry

end NUMINAMATH_CALUDE_base3_20121_equals_178_l116_11685


namespace NUMINAMATH_CALUDE_integral_sqrt_plus_xcosx_equals_pi_half_l116_11668

open Real MeasureTheory Interval

theorem integral_sqrt_plus_xcosx_equals_pi_half :
  ∫ x in (-1)..1, (Real.sqrt (1 - x^2) + x * Real.cos x) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_plus_xcosx_equals_pi_half_l116_11668


namespace NUMINAMATH_CALUDE_soccer_players_count_l116_11621

theorem soccer_players_count (total_socks : ℕ) (socks_per_player : ℕ) : 
  total_socks = 16 → socks_per_player = 2 → total_socks / socks_per_player = 8 := by
  sorry

end NUMINAMATH_CALUDE_soccer_players_count_l116_11621


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_hyperbola_l116_11647

theorem no_integer_solutions_for_hyperbola : 
  ¬∃ (x y : ℤ), x^2 - y^2 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_hyperbola_l116_11647


namespace NUMINAMATH_CALUDE_grid_intersection_sum_zero_l116_11688

/-- Represents a cell in the grid -/
inductive CellValue
  | Plus : CellValue
  | Minus : CellValue
  | Zero : CellValue

/-- Represents the grid -/
def Grid := Matrix (Fin 1980) (Fin 1981) CellValue

/-- The sum of all numbers in the grid is zero -/
def sumIsZero (g : Grid) : Prop := sorry

/-- The sum of four numbers at the intersections of two rows and two columns -/
def intersectionSum (g : Grid) (r1 r2 : Fin 1980) (c1 c2 : Fin 1981) : Int := sorry

theorem grid_intersection_sum_zero (g : Grid) (h : sumIsZero g) :
  ∃ (r1 r2 : Fin 1980) (c1 c2 : Fin 1981), intersectionSum g r1 r2 c1 c2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_grid_intersection_sum_zero_l116_11688


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l116_11644

/-- Given a geometric sequence with common ratio q ≠ 1, if the ratio of the sum of the first 10 terms
    to the sum of the first 5 terms is 1:2, then the ratio of the sum of the first 15 terms to the
    sum of the first 5 terms is 3:4. -/
theorem geometric_sequence_sum_ratio (q : ℝ) (a : ℕ → ℝ) (h1 : q ≠ 1) 
  (h2 : ∀ n, a (n + 1) = q * a n) 
  (h3 : (1 - q^10) / (1 - q^5) = 1 / 2) :
  (1 - q^15) / (1 - q^5) = 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l116_11644


namespace NUMINAMATH_CALUDE_card_area_theorem_l116_11662

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem card_area_theorem (original : Rectangle) 
  (h1 : original.length = 3 ∧ original.width = 7)
  (h2 : ∃ (shortened : Rectangle), 
    (shortened.length = original.length ∧ shortened.width = original.width - 2) ∨
    (shortened.length = original.length - 2 ∧ shortened.width = original.width) ∧
    area shortened = 15) :
  ∃ (other_shortened : Rectangle),
    (other_shortened.length = original.length - 2 ∧ other_shortened.width = original.width) ∨
    (other_shortened.length = original.length ∧ other_shortened.width = original.width - 2) ∧
    area other_shortened ≠ area shortened ∧
    area other_shortened = 7 := by
  sorry

end NUMINAMATH_CALUDE_card_area_theorem_l116_11662


namespace NUMINAMATH_CALUDE_paperboy_delivery_count_l116_11675

def delivery_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | 3 => 2
  | m + 3 => delivery_ways (m + 2) + delivery_ways (m + 1) + delivery_ways m

theorem paperboy_delivery_count :
  delivery_ways 12 = 504 :=
by sorry

end NUMINAMATH_CALUDE_paperboy_delivery_count_l116_11675


namespace NUMINAMATH_CALUDE_tan_alpha_values_l116_11687

theorem tan_alpha_values (α : Real) 
  (h : 2 * Real.sin α ^ 2 + Real.sin α * Real.cos α - 3 * Real.cos α ^ 2 = 7/5) : 
  Real.tan α = 2 ∨ Real.tan α = -11/3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_values_l116_11687


namespace NUMINAMATH_CALUDE_junior_count_in_club_l116_11677

theorem junior_count_in_club (total_students : ℕ) 
  (junior_selection_rate : ℚ) (senior_selection_rate : ℚ) : ℕ :=
by
  sorry

#check junior_count_in_club 30 (2/5) (1/4) = 11

end NUMINAMATH_CALUDE_junior_count_in_club_l116_11677


namespace NUMINAMATH_CALUDE_log_gt_x_squared_over_one_plus_x_l116_11646

theorem log_gt_x_squared_over_one_plus_x :
  ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, 0 < x → x < a → Real.log (1 + x) > x^2 / (1 + x) := by
  sorry

end NUMINAMATH_CALUDE_log_gt_x_squared_over_one_plus_x_l116_11646


namespace NUMINAMATH_CALUDE_radio_price_rank_l116_11633

theorem radio_price_rank (prices : Finset ℕ) (radio_price : ℕ) : 
  prices.card = 16 → 
  (∀ (p1 p2 : ℕ), p1 ∈ prices → p2 ∈ prices → p1 ≠ p2) →
  radio_price ∈ prices →
  (prices.filter (λ p => p > radio_price)).card = 3 →
  (prices.filter (λ p => p < radio_price)).card = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_radio_price_rank_l116_11633


namespace NUMINAMATH_CALUDE_investment_final_values_l116_11601

/-- Calculates the final value of an investment after two years --/
def final_value (initial : ℝ) (year1_change : ℝ) (year1_dividend : ℝ) (year2_change : ℝ) : ℝ :=
  (initial * (1 + year1_change) + initial * year1_dividend) * (1 + year2_change)

/-- Proves that the final values of investments D, E, and F are correct --/
theorem investment_final_values :
  let d := final_value 100 0 0.1 0.05
  let e := final_value 100 0.3 0 (-0.1)
  let f := final_value 100 (-0.1) 0 0.2
  d = 115.5 ∧ e = 117 ∧ f = 108 :=
by sorry

#eval final_value 100 0 0.1 0.05
#eval final_value 100 0.3 0 (-0.1)
#eval final_value 100 (-0.1) 0 0.2

end NUMINAMATH_CALUDE_investment_final_values_l116_11601


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l116_11631

theorem absolute_value_inequality (x y z : ℝ) : 
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l116_11631


namespace NUMINAMATH_CALUDE_smallest_acute_angle_in_right_triangle_l116_11661

theorem smallest_acute_angle_in_right_triangle (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 90 → (a / b) = (3 / 2) → min a b = 36 := by
  sorry

end NUMINAMATH_CALUDE_smallest_acute_angle_in_right_triangle_l116_11661


namespace NUMINAMATH_CALUDE_existence_of_number_with_prime_multiples_l116_11630

theorem existence_of_number_with_prime_multiples : ∃ x : ℝ, 
  (∃ p : ℕ, Nat.Prime p ∧ (10 : ℝ) * x = p) ∧ 
  (∃ q : ℕ, Nat.Prime q ∧ (15 : ℝ) * x = q) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_number_with_prime_multiples_l116_11630


namespace NUMINAMATH_CALUDE_complement_intersection_equals_singleton_l116_11634

def U : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
def A : Set ℤ := {x | -1 < x ∧ x < 3}
def B : Set ℤ := {x | x^2 - x - 2 ≤ 0}

theorem complement_intersection_equals_singleton :
  (U \ A) ∩ B = {-1} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_singleton_l116_11634


namespace NUMINAMATH_CALUDE_negation_of_proposition_l116_11643

theorem negation_of_proposition (x : ℝ) :
  ¬(2 * x + 1 ≤ 0) ↔ (2 * x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l116_11643


namespace NUMINAMATH_CALUDE_rahul_share_is_142_l116_11665

/-- Calculates the share of payment for a worker given the total payment and the work rates of two workers --/
def calculate_share (total_payment : ℚ) (rahul_days : ℚ) (rajesh_days : ℚ) : ℚ :=
  let rahul_rate := 1 / rahul_days
  let rajesh_rate := 1 / rajesh_days
  let combined_rate := rahul_rate + rajesh_rate
  let rahul_share_ratio := rahul_rate / combined_rate
  total_payment * rahul_share_ratio

/-- Theorem stating that Rahul's share is 142 given the problem conditions --/
theorem rahul_share_is_142 :
  calculate_share 355 3 2 = 142 := by
  sorry

end NUMINAMATH_CALUDE_rahul_share_is_142_l116_11665


namespace NUMINAMATH_CALUDE_thirteen_students_in_line_l116_11606

/-- The number of students in a line, given specific positions of Taehyung and Namjoon -/
def students_in_line (people_between_taehyung_and_namjoon : ℕ) (people_behind_namjoon : ℕ) : ℕ :=
  1 + people_between_taehyung_and_namjoon + 1 + people_behind_namjoon

/-- Theorem stating that there are 13 students in the line -/
theorem thirteen_students_in_line : 
  students_in_line 3 8 = 13 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_students_in_line_l116_11606


namespace NUMINAMATH_CALUDE_range_of_a_l116_11640

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 2*a| + |2*x - a| ≥ a^2) → -3/2 ≤ a ∧ a ≤ 3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l116_11640


namespace NUMINAMATH_CALUDE_welch_distance_before_pie_l116_11699

/-- The distance Mr. Welch drove before buying a pie -/
def distance_before_pie (total_distance : ℕ) (distance_after_pie : ℕ) : ℕ :=
  total_distance - distance_after_pie

/-- Theorem: Mr. Welch drove 35 miles before buying a pie -/
theorem welch_distance_before_pie :
  distance_before_pie 78 43 = 35 := by
  sorry

end NUMINAMATH_CALUDE_welch_distance_before_pie_l116_11699


namespace NUMINAMATH_CALUDE_money_distribution_l116_11622

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 400)
  (AC_sum : A + C = 300)
  (C_amount : C = 50) : 
  B + C = 150 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l116_11622


namespace NUMINAMATH_CALUDE_coupon1_best_at_209_95_l116_11611

def coupon1_discount (price : ℝ) : ℝ := 0.15 * price

def coupon2_discount (price : ℝ) : ℝ := 30

def coupon3_discount (price : ℝ) : ℝ := 0.25 * (price - 120)

def coupon4_discount (price : ℝ) : ℝ := 0.05 * price

def price_options : List ℝ := [169.95, 189.95, 209.95, 229.95, 249.95]

theorem coupon1_best_at_209_95 :
  let price := 209.95
  (∀ (other_price : ℝ), other_price ∈ price_options → other_price < price → 
    ¬(coupon1_discount other_price > coupon2_discount other_price ∧
      coupon1_discount other_price > coupon3_discount other_price ∧
      coupon1_discount other_price > coupon4_discount other_price)) ∧
  (coupon1_discount price > coupon2_discount price) ∧
  (coupon1_discount price > coupon3_discount price) ∧
  (coupon1_discount price > coupon4_discount price) :=
by sorry

end NUMINAMATH_CALUDE_coupon1_best_at_209_95_l116_11611


namespace NUMINAMATH_CALUDE_closest_reps_20_eq_12_or_13_l116_11670

def weight_25 : ℕ := 25
def weight_20 : ℕ := 20
def reps_25 : ℕ := 10

def total_weight : ℕ := 2 * weight_25 * reps_25

def closest_reps (w : ℕ) : Set ℕ :=
  {n : ℕ | n * 2 * w ≥ total_weight ∧ 
    ∀ m : ℕ, m * 2 * w ≥ total_weight → n ≤ m}

theorem closest_reps_20_eq_12_or_13 : 
  closest_reps weight_20 = {12, 13} :=
sorry

end NUMINAMATH_CALUDE_closest_reps_20_eq_12_or_13_l116_11670


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l116_11660

theorem smallest_number_divisible (n : ℕ) : n ≥ 62 →
  (∃ (k : ℕ), n - 8 = 18 * k ∧ n - 8 ≥ 44) →
  (∀ (m : ℕ), m < n →
    ¬(∃ (l : ℕ), m - 8 = 18 * l ∧ m - 8 ≥ 44)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l116_11660


namespace NUMINAMATH_CALUDE_max_value_of_b_l116_11663

theorem max_value_of_b (a b : ℤ) : 
  (a + b)^2 + a*(a + b) + b = 0 → b ≤ 9 := by sorry

end NUMINAMATH_CALUDE_max_value_of_b_l116_11663


namespace NUMINAMATH_CALUDE_inequality_system_solution_l116_11625

theorem inequality_system_solution (a b : ℝ) :
  (∀ x : ℝ, (0 < x ∧ x < 4) ↔ (x - a < 1 ∧ x + b > 2)) →
  b - a = -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l116_11625


namespace NUMINAMATH_CALUDE_representation_of_2020_as_sum_of_five_cubes_l116_11696

theorem representation_of_2020_as_sum_of_five_cubes :
  ∃ (n : ℤ), 2020 = (n + 2)^3 + n^3 + (-n - 1)^3 + (-n - 1)^3 + (-2)^3 :=
by
  use 337
  sorry

end NUMINAMATH_CALUDE_representation_of_2020_as_sum_of_five_cubes_l116_11696


namespace NUMINAMATH_CALUDE_triangle_rotation_l116_11641

/-- Triangle OAB with given properties and rotation of OA --/
theorem triangle_rotation (A : ℝ × ℝ) : 
  let O : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (5, 0)
  let angle_ABO : ℝ := π / 2  -- 90°
  let angle_AOB : ℝ := π / 6  -- 30°
  let rotation_angle : ℝ := 2 * π / 3  -- 120°
  A.1 > 0 ∧ A.2 > 0 →  -- A is in the first quadrant
  (A.1 - O.1) * (B.2 - O.2) = (B.1 - O.1) * (A.2 - O.2) →  -- ABO is a right angle
  (A.1 - O.1) * (B.1 - O.1) + (A.2 - O.2) * (B.2 - O.2) = 
    Real.cos angle_AOB * Real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2) * Real.sqrt ((B.1 - O.1)^2 + (B.2 - O.2)^2) →
  let rotated_A : ℝ × ℝ := (
    A.1 * Real.cos rotation_angle - A.2 * Real.sin rotation_angle,
    A.1 * Real.sin rotation_angle + A.2 * Real.cos rotation_angle
  )
  rotated_A = (-5/2 * (1 + Real.sqrt 3), 5/2 * (Real.sqrt 3 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_rotation_l116_11641


namespace NUMINAMATH_CALUDE_triangle_properties_l116_11698

noncomputable section

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The triangle satisfies the given condition -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.a + 2 * t.c = t.b * Real.cos t.C + Real.sqrt 3 * t.b * Real.sin t.C

theorem triangle_properties (t : Triangle) 
  (h : satisfiesCondition t) : 
  t.B = 2 * Real.pi / 3 ∧ 
  (t.b = 3 → 
    6 < t.a + t.b + t.c ∧ 
    t.a + t.b + t.c ≤ 3 + 2 * Real.sqrt 3) := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_properties_l116_11698


namespace NUMINAMATH_CALUDE_james_cd_count_l116_11610

theorem james_cd_count :
  let short_cd_length : ℝ := 1.5
  let long_cd_length : ℝ := 2 * short_cd_length
  let short_cd_count : ℕ := 2
  let long_cd_count : ℕ := 1
  let total_length : ℝ := 6
  (short_cd_count * short_cd_length + long_cd_count * long_cd_length = total_length) →
  (short_cd_count + long_cd_count = 3) :=
by
  sorry

#check james_cd_count

end NUMINAMATH_CALUDE_james_cd_count_l116_11610


namespace NUMINAMATH_CALUDE_certain_number_proof_l116_11684

theorem certain_number_proof (x : ℝ) : x * 2.13 = 0.3408 → x = 0.1600 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l116_11684


namespace NUMINAMATH_CALUDE_point_order_on_increasing_line_a_less_than_b_l116_11695

/-- A line in 2D space defined by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a given line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

theorem point_order_on_increasing_line
  (l : Line)
  (p1 p2 : Point)
  (h_slope : l.slope > 0)
  (h_x : p1.x < p2.x)
  (h_on1 : p1.liesOn l)
  (h_on2 : p2.liesOn l) :
  p1.y < p2.y :=
sorry

theorem a_less_than_b :
  let l : Line := { slope := 2/3, intercept := -3 }
  let p1 : Point := { x := -1, y := a }
  let p2 : Point := { x := 1/2, y := b }
  p1.liesOn l → p2.liesOn l → a < b :=
sorry

end NUMINAMATH_CALUDE_point_order_on_increasing_line_a_less_than_b_l116_11695


namespace NUMINAMATH_CALUDE_correct_solution_for_equation_l116_11600

theorem correct_solution_for_equation (x a : ℚ) :
  a = 2/3 →
  (2*x - 1)/3 = (x + a)/2 - 2 →
  x = -8 := by
sorry

end NUMINAMATH_CALUDE_correct_solution_for_equation_l116_11600


namespace NUMINAMATH_CALUDE_min_plane_spotlights_theorem_min_space_spotlights_theorem_l116_11671

/-- A spotlight that illuminates a 90° plane angle --/
structure PlaneSpotlight where
  angle : ℝ
  angle_eq : angle = 90

/-- A spotlight that illuminates a trihedral angle with all plane angles of 90° --/
structure SpaceSpotlight where
  angle : ℝ
  angle_eq : angle = 90

/-- The minimum number of spotlights required to illuminate the entire plane --/
def min_plane_spotlights : ℕ := 4

/-- The minimum number of spotlights required to illuminate the entire space --/
def min_space_spotlights : ℕ := 8

/-- Theorem stating the minimum number of spotlights required for full plane illumination --/
theorem min_plane_spotlights_theorem (s : PlaneSpotlight) :
  min_plane_spotlights = 4 := by sorry

/-- Theorem stating the minimum number of spotlights required for full space illumination --/
theorem min_space_spotlights_theorem (s : SpaceSpotlight) :
  min_space_spotlights = 8 := by sorry

end NUMINAMATH_CALUDE_min_plane_spotlights_theorem_min_space_spotlights_theorem_l116_11671
