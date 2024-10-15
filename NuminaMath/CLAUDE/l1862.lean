import Mathlib

namespace NUMINAMATH_CALUDE_unit_circle_sector_arc_length_l1862_186224

theorem unit_circle_sector_arc_length (θ : Real) :
  (1/2 * θ = 1) → (θ = 2) := by
  sorry

end NUMINAMATH_CALUDE_unit_circle_sector_arc_length_l1862_186224


namespace NUMINAMATH_CALUDE_sum_of_42_odd_numbers_l1862_186252

/-- The sum of the first n odd numbers -/
def sumOfOddNumbers (n : ℕ) : ℕ :=
  n * n

theorem sum_of_42_odd_numbers :
  sumOfOddNumbers 42 = 1764 := by
  sorry

#eval sumOfOddNumbers 42  -- This will output 1764

end NUMINAMATH_CALUDE_sum_of_42_odd_numbers_l1862_186252


namespace NUMINAMATH_CALUDE_blue_paint_cans_l1862_186272

theorem blue_paint_cans (total_cans : ℕ) (blue_ratio yellow_ratio : ℕ) 
  (h1 : total_cans = 42)
  (h2 : blue_ratio = 4)
  (h3 : yellow_ratio = 3) : 
  (blue_ratio * total_cans) / (blue_ratio + yellow_ratio) = 24 := by
  sorry

end NUMINAMATH_CALUDE_blue_paint_cans_l1862_186272


namespace NUMINAMATH_CALUDE_b_alone_time_l1862_186225

/-- The time (in days) it takes for A and B together to complete the work -/
def combined_time : ℚ := 12

/-- The time (in days) it takes for A alone to complete the work -/
def a_time : ℚ := 24

/-- The work rate of A (work per day) -/
def a_rate : ℚ := 1 / a_time

/-- The combined work rate of A and B (work per day) -/
def combined_rate : ℚ := 1 / combined_time

/-- The work rate of B (work per day) -/
def b_rate : ℚ := combined_rate - a_rate

/-- The time (in days) it takes for B alone to complete the work -/
def b_time : ℚ := 1 / b_rate

theorem b_alone_time : b_time = 24 := by
  sorry

end NUMINAMATH_CALUDE_b_alone_time_l1862_186225


namespace NUMINAMATH_CALUDE_product_congruence_l1862_186298

theorem product_congruence : ∃ m : ℕ, 0 ≤ m ∧ m < 25 ∧ (93 * 59 * 84) % 25 = m ∧ m = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_congruence_l1862_186298


namespace NUMINAMATH_CALUDE_optimal_strategy_is_valid_l1862_186229

/-- Represents a chain of links -/
structure Chain where
  links : ℕ

/-- Represents a cut in the chain -/
structure Cut where
  position : ℕ

/-- Represents a payment strategy for the hotel stay -/
structure PaymentStrategy where
  cut : Cut
  dailyPayments : List ℕ

/-- Checks if a payment strategy is valid for the given chain and number of days -/
def isValidPaymentStrategy (c : Chain) (days : ℕ) (s : PaymentStrategy) : Prop :=
  c.links = days ∧
  s.cut.position > 0 ∧
  s.cut.position < c.links ∧
  s.dailyPayments.length = days ∧
  s.dailyPayments.sum = c.links

/-- The optimal payment strategy for a 7-day stay with a 7-link chain -/
def optimalStrategy : PaymentStrategy :=
  { cut := { position := 3 },
    dailyPayments := [1, 1, 1, 1, 1, 1, 1] }

/-- Theorem stating that the optimal strategy is valid for a 7-day stay with a 7-link chain -/
theorem optimal_strategy_is_valid :
  isValidPaymentStrategy { links := 7 } 7 optimalStrategy := by sorry

end NUMINAMATH_CALUDE_optimal_strategy_is_valid_l1862_186229


namespace NUMINAMATH_CALUDE_sams_eatery_meal_cost_l1862_186241

/-- Calculates the cost of a meal at Sam's Eatery with a discount --/
def meal_cost (hamburger_price : ℚ) (fries_price : ℚ) (drink_price : ℚ) 
              (num_hamburgers : ℕ) (num_fries : ℕ) (num_drinks : ℕ) 
              (discount_percent : ℚ) : ℕ :=
  let total_before_discount := hamburger_price * num_hamburgers + 
                               fries_price * num_fries + 
                               drink_price * num_drinks
  let discount_amount := total_before_discount * (discount_percent / 100)
  let total_after_discount := total_before_discount - discount_amount
  (total_after_discount + 1/2).floor.toNat

/-- The cost of the meal at Sam's Eatery is 35 dollars --/
theorem sams_eatery_meal_cost : 
  meal_cost 5 3 2 3 4 6 10 = 35 := by
  sorry


end NUMINAMATH_CALUDE_sams_eatery_meal_cost_l1862_186241


namespace NUMINAMATH_CALUDE_nicks_chocolate_oranges_l1862_186294

/-- Proves the number of chocolate oranges Nick had initially -/
theorem nicks_chocolate_oranges 
  (candy_bar_price : ℕ) 
  (chocolate_orange_price : ℕ) 
  (fundraising_goal : ℕ) 
  (candy_bars_to_sell : ℕ) 
  (h1 : candy_bar_price = 5)
  (h2 : chocolate_orange_price = 10)
  (h3 : fundraising_goal = 1000)
  (h4 : candy_bars_to_sell = 160)
  (h5 : candy_bar_price * candy_bars_to_sell + chocolate_orange_price * chocolate_oranges = fundraising_goal) :
  chocolate_oranges = 20 := by
  sorry

end NUMINAMATH_CALUDE_nicks_chocolate_oranges_l1862_186294


namespace NUMINAMATH_CALUDE_janes_coins_l1862_186216

theorem janes_coins (q d : ℕ) : 
  q + d = 30 → 
  (10 * q + 25 * d) - (25 * q + 10 * d) = 150 →
  25 * q + 10 * d = 450 :=
by sorry

end NUMINAMATH_CALUDE_janes_coins_l1862_186216


namespace NUMINAMATH_CALUDE_cloth_woven_approx_15_meters_l1862_186207

/-- The rate at which the loom weaves cloth in meters per second -/
def weaving_rate : ℝ := 0.127

/-- The time taken by the loom to weave the cloth in seconds -/
def weaving_time : ℝ := 118.11

/-- The amount of cloth woven in meters -/
def cloth_woven : ℝ := weaving_rate * weaving_time

/-- Theorem stating that the amount of cloth woven is approximately 15 meters -/
theorem cloth_woven_approx_15_meters : 
  ∃ ε > 0, |cloth_woven - 15| < ε := by sorry

end NUMINAMATH_CALUDE_cloth_woven_approx_15_meters_l1862_186207


namespace NUMINAMATH_CALUDE_division_property_l1862_186236

theorem division_property (n : ℕ) (hn : n > 0) :
  (5^(n-1) + 3^(n-1)) ∣ (5^n + 3^n) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_property_l1862_186236


namespace NUMINAMATH_CALUDE_ava_watched_hours_l1862_186265

-- Define the number of minutes in an hour
def minutes_per_hour : ℕ := 60

-- Define the number of minutes Ava watched television
def ava_watched_minutes : ℕ := 240

-- Theorem to prove
theorem ava_watched_hours : ava_watched_minutes / minutes_per_hour = 4 := by
  sorry

end NUMINAMATH_CALUDE_ava_watched_hours_l1862_186265


namespace NUMINAMATH_CALUDE_tangent_line_at_point_one_four_l1862_186273

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 + x + 2

/-- The derivative of the parabola function -/
def f' (x : ℝ) : ℝ := 2*x + 1

theorem tangent_line_at_point_one_four :
  let x₀ : ℝ := 1
  let y₀ : ℝ := 4
  -- The point (1,4) lies on the parabola
  (f x₀ = y₀) →
  -- The slope of the tangent line at (1,4) is 3
  (f' x₀ = 3) ∧
  -- The equation of the tangent line is 3x - y + 1 = 0
  (∀ x y, y - y₀ = f' x₀ * (x - x₀) ↔ 3*x - y + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_one_four_l1862_186273


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l1862_186268

theorem solution_to_system_of_equations :
  let x : ℚ := -49/3
  let y : ℚ := -17/6
  (3 * x - 18 * y = 2) ∧ (4 * y - x = 5) := by
sorry

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l1862_186268


namespace NUMINAMATH_CALUDE_cards_taken_away_l1862_186285

theorem cards_taken_away (initial_cards final_cards : ℕ) 
  (h1 : initial_cards = 76)
  (h2 : final_cards = 17) :
  initial_cards - final_cards = 59 := by
  sorry

end NUMINAMATH_CALUDE_cards_taken_away_l1862_186285


namespace NUMINAMATH_CALUDE_car_cost_equation_l1862_186287

/-- Proves that the original cost of the car satisfies the given equation -/
theorem car_cost_equation (repair_cost selling_price profit_percent : ℝ) 
  (h1 : repair_cost = 15000)
  (h2 : selling_price = 64900)
  (h3 : profit_percent = 13.859649122807017) :
  ∃ C : ℝ, (1 + profit_percent / 100) * C = selling_price - repair_cost :=
sorry

end NUMINAMATH_CALUDE_car_cost_equation_l1862_186287


namespace NUMINAMATH_CALUDE_expression_equals_36_l1862_186230

theorem expression_equals_36 : ∃ (expr : ℝ), 
  (expr = 13 * (3 - 3 / 13)) ∧ (expr = 36) :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_36_l1862_186230


namespace NUMINAMATH_CALUDE_josette_bought_three_bottles_l1862_186233

/-- The number of bottles Josette bought for €1.50, given that 4 bottles cost €2 -/
def bottles_bought (cost_four_bottles : ℚ) (amount_spent : ℚ) : ℚ :=
  amount_spent / (cost_four_bottles / 4)

/-- Theorem stating that Josette bought 3 bottles -/
theorem josette_bought_three_bottles : 
  bottles_bought 2 (3/2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_josette_bought_three_bottles_l1862_186233


namespace NUMINAMATH_CALUDE_sum_of_legs_special_triangle_l1862_186292

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  a : ℕ  -- shorter leg
  b : ℕ  -- longer leg
  c : ℕ  -- hypotenuse
  right_angle : a^2 + b^2 = c^2
  consecutive_even : b = a + 2

/-- The sum of legs of a right triangle with hypotenuse 50 and consecutive even legs is 70 -/
theorem sum_of_legs_special_triangle :
  ∀ (t : RightTriangle), t.c = 50 → t.a + t.b = 70 := by
  sorry

#check sum_of_legs_special_triangle

end NUMINAMATH_CALUDE_sum_of_legs_special_triangle_l1862_186292


namespace NUMINAMATH_CALUDE_monkey_apple_problem_l1862_186209

/-- Given a number of monkeys and apples, this function checks if they satisfy the conditions:
    1. If each monkey gets 3 apples, there will be 6 left.
    2. If each monkey gets 4 apples, the last monkey will get less than 4 apples. -/
def satisfies_conditions (monkeys : ℕ) (apples : ℕ) : Prop :=
  (apples = 3 * monkeys + 6) ∧ 
  (apples < 4 * monkeys) ∧ 
  (apples > 4 * (monkeys - 1))

/-- Theorem stating that the only solutions satisfying the conditions are
    (7 monkeys, 27 apples), (8 monkeys, 30 apples), or (9 monkeys, 33 apples) -/
theorem monkey_apple_problem :
  ∀ monkeys apples : ℕ, 
    satisfies_conditions monkeys apples ↔ 
    ((monkeys = 7 ∧ apples = 27) ∨ 
     (monkeys = 8 ∧ apples = 30) ∨ 
     (monkeys = 9 ∧ apples = 33)) :=
by sorry

end NUMINAMATH_CALUDE_monkey_apple_problem_l1862_186209


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l1862_186203

theorem quadratic_solution_property (k : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 + 5 * x + k = 0 ∧ 3 * y^2 + 5 * y + k = 0 ∧ 
   |x + y| = x^2 + y^2) ↔ k = -10/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l1862_186203


namespace NUMINAMATH_CALUDE_direction_cosines_sum_of_squares_l1862_186247

/-- Direction cosines of a vector in 3D space -/
structure DirectionCosines where
  α : ℝ
  β : ℝ
  γ : ℝ

/-- Theorem: The sum of squares of direction cosines equals 1 -/
theorem direction_cosines_sum_of_squares (dc : DirectionCosines) : 
  dc.α^2 + dc.β^2 + dc.γ^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_direction_cosines_sum_of_squares_l1862_186247


namespace NUMINAMATH_CALUDE_choir_members_count_l1862_186277

theorem choir_members_count : ∃! n : ℕ, 
  150 < n ∧ n < 250 ∧ 
  n % 4 = 3 ∧ 
  n % 5 = 4 ∧ 
  n % 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_choir_members_count_l1862_186277


namespace NUMINAMATH_CALUDE_ab_positive_sufficient_not_necessary_l1862_186288

theorem ab_positive_sufficient_not_necessary :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * b > 0) ∧
  (∃ a b : ℝ, a * b > 0 ∧ ¬(a > 0 ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_ab_positive_sufficient_not_necessary_l1862_186288


namespace NUMINAMATH_CALUDE_exists_infinite_set_satisfying_equation_l1862_186212

/-- A function from positive integers to positive integers -/
def PositiveIntegerFunction : Type := ℕ+ → ℕ+

/-- The property that f(x) + f(x+2) ≤ 2f(x+1) for all x -/
def SatisfiesInequality (f : PositiveIntegerFunction) : Prop :=
  ∀ x : ℕ+, f x + f (x + 2) ≤ 2 * f (x + 1)

/-- The property that (i-j)f(k) + (j-k)f(i) + (k-i)f(j) = 0 for all i, j, k in a set M -/
def SatisfiesEquation (f : PositiveIntegerFunction) (M : Set ℕ+) : Prop :=
  ∀ i j k : ℕ+, i ∈ M → j ∈ M → k ∈ M →
    (i - j : ℤ) * (f k : ℤ) + (j - k : ℤ) * (f i : ℤ) + (k - i : ℤ) * (f j : ℤ) = 0

/-- The main theorem -/
theorem exists_infinite_set_satisfying_equation
  (f : PositiveIntegerFunction) (h : SatisfiesInequality f) :
  ∃ M : Set ℕ+, Set.Infinite M ∧ SatisfiesEquation f M := by
  sorry

end NUMINAMATH_CALUDE_exists_infinite_set_satisfying_equation_l1862_186212


namespace NUMINAMATH_CALUDE_mashas_measurements_impossible_l1862_186254

/-- A pentagon inscribed in a circle with given interior angles -/
structure InscribedPentagon where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ
  angle5 : ℝ

/-- The sum of interior angles of a pentagon is 540° -/
axiom pentagon_angle_sum (p : InscribedPentagon) :
  p.angle1 + p.angle2 + p.angle3 + p.angle4 + p.angle5 = 540

/-- Opposite angles in an inscribed quadrilateral sum to 180° -/
axiom inscribed_quadrilateral_opposite_angles (a b : ℝ) :
  a + b = 180 → ∃ (p : InscribedPentagon), p.angle1 = a ∧ p.angle3 = b

/-- Masha's measurements -/
def mashas_pentagon : InscribedPentagon := {
  angle1 := 80,
  angle2 := 90,
  angle3 := 100,
  angle4 := 130,
  angle5 := 140
}

/-- Theorem: Masha's measurements are impossible for a pentagon inscribed in a circle -/
theorem mashas_measurements_impossible : 
  ¬∃ (p : InscribedPentagon), p = mashas_pentagon :=
sorry

end NUMINAMATH_CALUDE_mashas_measurements_impossible_l1862_186254


namespace NUMINAMATH_CALUDE_complex_sum_power_l1862_186248

theorem complex_sum_power (i : ℂ) : i * i = -1 → (1 - i)^2016 + (1 + i)^2016 = 2^1009 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_power_l1862_186248


namespace NUMINAMATH_CALUDE_actual_distance_travelled_l1862_186295

/-- The actual distance travelled by a person under specific conditions -/
theorem actual_distance_travelled (normal_speed fast_speed additional_distance : ℝ) 
  (h1 : normal_speed = 10)
  (h2 : fast_speed = 14)
  (h3 : additional_distance = 20)
  (h4 : (actual_distance / normal_speed) = ((actual_distance + additional_distance) / fast_speed)) :
  actual_distance = 50 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_travelled_l1862_186295


namespace NUMINAMATH_CALUDE_cord_length_proof_l1862_186262

/-- Given a cord divided into 19 equal parts, which when cut results in 20 pieces
    with the longest piece being 8 meters and the shortest being 2 meters,
    prove that the original length of the cord is 114 meters. -/
theorem cord_length_proof (n : ℕ) (longest shortest : ℝ) :
  n = 19 ∧
  longest = 8 ∧
  shortest = 2 →
  n * ((longest + shortest) / 2 + 1) = 114 :=
by sorry

end NUMINAMATH_CALUDE_cord_length_proof_l1862_186262


namespace NUMINAMATH_CALUDE_ratio_fraction_l1862_186218

theorem ratio_fraction (x y : ℝ) (h : x / y = 2 / 3) : x / (x + y) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_fraction_l1862_186218


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_is_correct_l1862_186227

/-- The number of zeros after the decimal point and before the first non-zero digit
    in the terminating decimal representation of 1/(2^7 * 5^3) -/
def zeros_before_first_nonzero : ℕ :=
  4

/-- The fraction we're considering -/
def fraction : ℚ :=
  1 / (2^7 * 5^3)

/-- Theorem stating that the number of zeros before the first non-zero digit
    in the terminating decimal representation of our fraction is correct -/
theorem zeros_before_first_nonzero_is_correct :
  zeros_before_first_nonzero = 4 ∧
  ∃ (n : ℕ), fraction * 10^zeros_before_first_nonzero = n / 10^zeros_before_first_nonzero ∧
             n % 10 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_is_correct_l1862_186227


namespace NUMINAMATH_CALUDE_basketball_points_difference_basketball_game_theorem_l1862_186201

/-- The difference between the combined points of Tobee and Jay and Sean's points is 2 -/
theorem basketball_points_difference : ℕ → ℕ → ℕ → Prop :=
  fun tobee_points jay_points_diff total_team_points =>
    let jay_points := tobee_points + jay_points_diff
    let combined_points := tobee_points + jay_points
    let sean_points := total_team_points - combined_points
    combined_points - sean_points = 2

/-- Given the conditions of the basketball game -/
theorem basketball_game_theorem :
  basketball_points_difference 4 6 26 := by
  sorry

end NUMINAMATH_CALUDE_basketball_points_difference_basketball_game_theorem_l1862_186201


namespace NUMINAMATH_CALUDE_max_product_under_constraint_max_product_achievable_l1862_186281

theorem max_product_under_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 9 * a^2 + 16 * b^2 = 25) : a * b ≤ 25 / 24 := by
  sorry

theorem max_product_achievable (ε : ℝ) (hε : ε > 0) : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 9 * a^2 + 16 * b^2 = 25 ∧ a * b > 25 / 24 - ε := by
  sorry

end NUMINAMATH_CALUDE_max_product_under_constraint_max_product_achievable_l1862_186281


namespace NUMINAMATH_CALUDE_solve_linear_system_l1862_186245

/-- Given a system of linear equations with parameters m and n,
    prove that m + n = -2 when x = 2 and y = 1 is a solution. -/
theorem solve_linear_system (m n : ℚ) : 
  (2 * m + 1 = -3) → (2 - 2 * 1 = 2 * n) → m + n = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_system_l1862_186245


namespace NUMINAMATH_CALUDE_parabola_equation_l1862_186202

/-- A parabola that opens downward with focus at (0, -2) -/
structure DownwardParabola where
  focus : ℝ × ℝ
  opens_downward : focus.2 < 0
  focus_y : focus.1 = 0 ∧ focus.2 = -2

/-- The hyperbola y²/3 - x² = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 / 3 - p.1^2 = 1}

/-- The standard form of a downward-opening parabola -/
def ParabolaEquation (p : ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | q.1^2 = -2 * p * q.2}

theorem parabola_equation (C : DownwardParabola) 
    (h : C.focus ∈ Hyperbola) : 
    ParabolaEquation 4 = {q : ℝ × ℝ | q.1^2 = -8 * q.2} := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l1862_186202


namespace NUMINAMATH_CALUDE_x_greater_than_half_l1862_186276

theorem x_greater_than_half (x : ℝ) (h1 : 1 / x^2 < 4) (h2 : 1 / x > -2) (h3 : x ≠ 0) : x > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_x_greater_than_half_l1862_186276


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l1862_186270

theorem restaurant_bill_proof : 
  ∀ (total_friends : ℕ) (paying_friends : ℕ) (extra_payment : ℚ),
    total_friends = 12 →
    paying_friends = 10 →
    extra_payment = 3 →
    ∃ (bill : ℚ), 
      bill = paying_friends * (bill / total_friends + extra_payment) ∧
      bill = 180 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l1862_186270


namespace NUMINAMATH_CALUDE_abie_spent_64_dollars_l1862_186299

def initial_bags : ℕ := 20
def original_price : ℚ := 2
def shared_fraction : ℚ := 2/5
def half_price_bags : ℕ := 18
def coupon_bags : ℕ := 4
def coupon_price_fraction : ℚ := 3/4

def total_spent : ℚ :=
  initial_bags * original_price +
  half_price_bags * (original_price / 2) +
  coupon_bags * (original_price * coupon_price_fraction)

theorem abie_spent_64_dollars : total_spent = 64 := by
  sorry

end NUMINAMATH_CALUDE_abie_spent_64_dollars_l1862_186299


namespace NUMINAMATH_CALUDE_backpack_price_equation_l1862_186253

/-- Represents the price of a backpack after discounts -/
def discounted_price (x : ℝ) : ℝ := 0.8 * x - 10

/-- Theorem stating that the discounted price equals the final selling price -/
theorem backpack_price_equation (x : ℝ) : 
  discounted_price x = 90 ↔ 0.8 * x - 10 = 90 := by sorry

end NUMINAMATH_CALUDE_backpack_price_equation_l1862_186253


namespace NUMINAMATH_CALUDE_no_such_function_exists_l1862_186250

theorem no_such_function_exists : 
  ∀ f : ℝ → ℝ, ∃ x y : ℝ, (f x + f y) / 2 < f ((x + y) / 2) + |x - y| := by
  sorry

end NUMINAMATH_CALUDE_no_such_function_exists_l1862_186250


namespace NUMINAMATH_CALUDE_man_walking_problem_l1862_186231

theorem man_walking_problem (x : ℝ) :
  let final_x := x - 6 * Real.sin (2 * Real.pi / 3)
  let final_y := 6 * Real.cos (2 * Real.pi / 3)
  final_x ^ 2 + final_y ^ 2 = 12 →
  x = 3 * Real.sqrt 3 + Real.sqrt 3 ∨ x = 3 * Real.sqrt 3 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_man_walking_problem_l1862_186231


namespace NUMINAMATH_CALUDE_ellipse_and_fixed_point_l1862_186244

noncomputable section

/-- The ellipse C with given conditions -/
structure Ellipse :=
  (a b c : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : (c / b) = Real.sqrt 3 / 3)
  (h5 : b + c + 2*c = 3 + Real.sqrt 3)

/-- The equation of the ellipse is x²/4 + y²/3 = 1 -/
def ellipse_equation (C : Ellipse) : Prop :=
  C.a = 2 ∧ C.b = Real.sqrt 3

/-- The fixed point on x-axis -/
def fixed_point : ℝ × ℝ := (5/2, 0)

/-- The line QM passes through the fixed point -/
def line_passes_through_fixed_point (C : Ellipse) (P : ℝ × ℝ) : Prop :=
  let F := (C.c, 0)
  let M := (4, P.2)
  let Q := sorry -- Intersection of PF with the ellipse
  ∃ t : ℝ, fixed_point = (1 - t) • Q + t • M

/-- Main theorem -/
theorem ellipse_and_fixed_point (C : Ellipse) :
  ellipse_equation C ∧
  ∀ P, P.1^2 / 4 + P.2^2 / 3 = 1 → line_passes_through_fixed_point C P :=
sorry

end

end NUMINAMATH_CALUDE_ellipse_and_fixed_point_l1862_186244


namespace NUMINAMATH_CALUDE_exists_cubic_polynomial_with_positive_roots_and_negative_derivative_roots_l1862_186261

/-- A cubic polynomial -/
def CubicPolynomial (a b c d : ℝ) : ℝ → ℝ := fun x ↦ a*x^3 + b*x^2 + c*x + d

/-- The derivative of a cubic polynomial -/
def DerivativeCubicPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ 3*a*x^2 + 2*b*x + c

/-- All roots of a function are positive -/
def AllRootsPositive (f : ℝ → ℝ) : Prop := ∀ x, f x = 0 → x > 0

/-- All roots of a function are negative -/
def AllRootsNegative (f : ℝ → ℝ) : Prop := ∀ x, f x = 0 → x < 0

/-- A function has at least one unique root -/
def HasUniqueRoot (f : ℝ → ℝ) : Prop := ∃ x, f x = 0 ∧ ∀ y, f y = 0 → y = x

theorem exists_cubic_polynomial_with_positive_roots_and_negative_derivative_roots :
  ∃ (a b c d : ℝ), 
    let P := CubicPolynomial a b c d
    let P' := DerivativeCubicPolynomial (3*a) (2*b) c
    AllRootsPositive P ∧
    AllRootsNegative P' ∧
    HasUniqueRoot P ∧
    HasUniqueRoot P' :=
sorry

end NUMINAMATH_CALUDE_exists_cubic_polynomial_with_positive_roots_and_negative_derivative_roots_l1862_186261


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l1862_186275

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
def y_intercept (a b : ℝ) : ℝ := b

/-- Given a line with equation y = 2x - 1, prove that its y-intercept is -1. -/
theorem y_intercept_of_line (x y : ℝ) (h : y = 2 * x - 1) : y_intercept 2 (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l1862_186275


namespace NUMINAMATH_CALUDE_vectors_perpendicular_if_sum_norm_eq_diff_norm_l1862_186282

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem vectors_perpendicular_if_sum_norm_eq_diff_norm 
  (a b : ℝ × ℝ) (h : ‖a + b‖ = ‖a - b‖) : 
  angle_between_vectors a b = π / 2 := by sorry

end NUMINAMATH_CALUDE_vectors_perpendicular_if_sum_norm_eq_diff_norm_l1862_186282


namespace NUMINAMATH_CALUDE_exist_numbers_not_triangle_l1862_186274

/-- Theorem: There exist natural numbers a and b, both greater than 1000,
    such that for any perfect square c, the triple (a, b, c) does not
    satisfy the triangle inequality. -/
theorem exist_numbers_not_triangle : ∃ a b : ℕ,
  a > 1000 ∧ b > 1000 ∧
  ∀ c : ℕ, (∃ d : ℕ, c = d * d) →
    ¬(a + b > c ∧ b + c > a ∧ a + c > b) := by
  sorry

end NUMINAMATH_CALUDE_exist_numbers_not_triangle_l1862_186274


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l1862_186238

theorem boys_to_girls_ratio : 
  let num_boys : ℕ := 40
  let num_girls : ℕ := num_boys + 64
  (num_boys : ℚ) / (num_girls : ℚ) = 5 / 13 :=
by sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l1862_186238


namespace NUMINAMATH_CALUDE_books_per_box_is_fifteen_l1862_186217

/-- Represents the number of books in Henry's collection at different stages --/
structure BookCollection where
  initial : Nat
  room : Nat
  coffeeTable : Nat
  kitchen : Nat
  final : Nat
  pickedUp : Nat

/-- Calculates the number of books in each donation box --/
def booksPerBox (collection : BookCollection) : Nat :=
  let totalDonated := collection.initial - collection.final + collection.pickedUp
  let outsideBoxes := collection.room + collection.coffeeTable + collection.kitchen
  let inBoxes := totalDonated - outsideBoxes
  inBoxes / 3

/-- Theorem stating that the number of books in each box is 15 --/
theorem books_per_box_is_fifteen (collection : BookCollection)
  (h1 : collection.initial = 99)
  (h2 : collection.room = 21)
  (h3 : collection.coffeeTable = 4)
  (h4 : collection.kitchen = 18)
  (h5 : collection.final = 23)
  (h6 : collection.pickedUp = 12) :
  booksPerBox collection = 15 := by
  sorry

end NUMINAMATH_CALUDE_books_per_box_is_fifteen_l1862_186217


namespace NUMINAMATH_CALUDE_sqrt_sum_2014_l1862_186246

theorem sqrt_sum_2014 (a b c : ℕ) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c = Real.sqrt 2014 →
  ((a = 0 ∧ b = 0 ∧ c = 2014) ∨
   (a = 0 ∧ b = 2014 ∧ c = 0) ∨
   (a = 2014 ∧ b = 0 ∧ c = 0)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_2014_l1862_186246


namespace NUMINAMATH_CALUDE_solution_set_characterization_l1862_186266

def valid_digit (n : ℕ) : Prop := n ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)

def base_10_value (x y z : ℕ) : ℕ := 100 * x + 10 * y + z

def base_7_value (x y z : ℕ) : ℕ := 49 * x + 7 * y + z

def satisfies_equation (x y z : ℕ) : Prop :=
  base_10_value x y z = 2 * base_7_value x y z

def valid_triple (x y z : ℕ) : Prop :=
  valid_digit x ∧ valid_digit y ∧ valid_digit z ∧ satisfies_equation x y z

theorem solution_set_characterization :
  {t : ℕ × ℕ × ℕ | valid_triple t.1 t.2.1 t.2.2} =
  {(3,1,2), (5,2,2), (4,1,4), (6,2,4), (5,1,6)} := by sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l1862_186266


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l1862_186206

/-- Given an ellipse and a hyperbola with the same foci, prove that the semi-major axis of the ellipse is 4 -/
theorem ellipse_hyperbola_same_foci (a : ℝ) : a > 0 →
  (∀ x y : ℝ, x^2 / a^2 + y^2 / 9 = 1) →
  (∀ x y : ℝ, x^2 / 4 - y^2 / 3 = 1) →
  (∀ c : ℝ, c^2 = 7 → 
    (∀ x y : ℝ, x^2 / a^2 + y^2 / 9 = 1 → (x + c)^2 + y^2 = a^2 ∧ (x - c)^2 + y^2 = a^2) ∧
    (∀ x y : ℝ, x^2 / 4 - y^2 / 3 = 1 → (x + c)^2 - y^2 = 4 ∧ (x - c)^2 - y^2 = 4)) →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l1862_186206


namespace NUMINAMATH_CALUDE_six_digit_increase_characterization_l1862_186221

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def last_digit (n : ℕ) : ℕ := n % 10

def move_last_to_first (n : ℕ) : ℕ :=
  (n / 10) + (last_digit n * 100000)

def increases_by_integer_factor (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ move_last_to_first n = k * n

def S : Set ℕ := {111111, 222222, 333333, 444444, 555555, 666666, 777777, 888888, 999999, 
                  142857, 102564, 128205, 153846, 179487, 205128, 230769}

theorem six_digit_increase_characterization :
  ∀ n : ℕ, is_six_digit n ∧ increases_by_integer_factor n ↔ n ∈ S :=
sorry

end NUMINAMATH_CALUDE_six_digit_increase_characterization_l1862_186221


namespace NUMINAMATH_CALUDE_stock_sale_loss_l1862_186228

/-- Calculates the overall loss from selling a stock with given conditions -/
def calculate_overall_loss (stock_worth : ℝ) : ℝ :=
  let profit_portion := 0.2 * stock_worth
  let loss_portion := 0.8 * stock_worth
  let profit := 0.1 * profit_portion
  let loss := 0.05 * loss_portion
  loss - profit

/-- Theorem stating the overall loss for the given stock and conditions -/
theorem stock_sale_loss (stock_worth : ℝ) (h : stock_worth = 12500) :
  calculate_overall_loss stock_worth = 250 := by
  sorry

end NUMINAMATH_CALUDE_stock_sale_loss_l1862_186228


namespace NUMINAMATH_CALUDE_extreme_value_interval_equation_solution_range_l1862_186258

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x

theorem extreme_value_interval (a : ℝ) (h : a > 0) :
  (∃ x ∈ Set.Ioo a (a + 1/2), ∀ y ∈ Set.Ioo a (a + 1/2), f x ≥ f y) →
  1/2 < a ∧ a < 1 :=
sorry

theorem equation_solution_range (k : ℝ) :
  (∃ x ≥ 1, f x = k / (x + 1)) →
  k ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_extreme_value_interval_equation_solution_range_l1862_186258


namespace NUMINAMATH_CALUDE_anushas_share_multiple_l1862_186297

/-- Proves that the multiple of Anusha's share is 12 given the problem conditions -/
theorem anushas_share_multiple (anusha babu esha : ℕ) (m : ℕ) : 
  anusha = 84 →
  m * anusha = 8 * babu →
  8 * babu = 6 * esha →
  anusha + babu + esha = 378 →
  m = 12 := by
  sorry

end NUMINAMATH_CALUDE_anushas_share_multiple_l1862_186297


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1862_186296

/-- Proves that the repeating decimal 0.3̄03 is equal to the fraction 109/330 -/
theorem repeating_decimal_to_fraction : 
  (0.3 : ℚ) + (3 : ℚ) / 100 / (1 - 1 / 100) = 109 / 330 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1862_186296


namespace NUMINAMATH_CALUDE_sequence_sum_formula_l1862_186264

def sequence_sum (n : ℕ) : ℕ → ℕ
| 0 => 5
| m + 1 => 2 * sequence_sum n m + (m + 1) + 5

theorem sequence_sum_formula (n : ℕ) :
  sequence_sum n n = 6 * 2^n - (n + 6) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_formula_l1862_186264


namespace NUMINAMATH_CALUDE_impossible_closed_line_l1862_186286

/-- Represents a prism with a given number of lateral edges and total edges. -/
structure Prism where
  lateral_edges : ℕ
  total_edges : ℕ

/-- Represents the possibility of forming a closed broken line from translated edges of a prism. -/
def can_form_closed_line (p : Prism) : Prop :=
  ∃ (arrangement : List ℝ), 
    arrangement.length = p.total_edges ∧ 
    arrangement.sum = 0 ∧
    (∀ i ∈ arrangement, i = 0 ∨ i = 1 ∨ i = -1)

/-- Theorem stating that it's impossible to form a closed broken line from the given prism's edges. -/
theorem impossible_closed_line (p : Prism) 
  (h1 : p.lateral_edges = 373) 
  (h2 : p.total_edges = 1119) : 
  ¬ can_form_closed_line p := by
  sorry

end NUMINAMATH_CALUDE_impossible_closed_line_l1862_186286


namespace NUMINAMATH_CALUDE_factorial_equation_l1862_186267

theorem factorial_equation : (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_l1862_186267


namespace NUMINAMATH_CALUDE_floor_plus_x_eq_seventeen_fourths_l1862_186257

theorem floor_plus_x_eq_seventeen_fourths :
  ∃ x : ℚ, (⌊x⌋ : ℚ) + x = 17/4 ∧ x = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_x_eq_seventeen_fourths_l1862_186257


namespace NUMINAMATH_CALUDE_focus_of_given_parabola_l1862_186263

/-- A parabola is a set of points in a plane that are equidistant from a fixed point (focus) and a fixed line (directrix). -/
structure Parabola where
  /-- The equation of the parabola in the form y = a(x - h)^2 + k -/
  equation : ℝ → ℝ
  /-- The coefficient 'a' determines the direction and width of the parabola -/
  a : ℝ
  /-- The horizontal shift of the vertex -/
  h : ℝ
  /-- The vertical shift of the vertex -/
  k : ℝ

/-- The focus of a parabola is a point from which all points on the parabola are equidistant to the directrix. -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- Given parabola y = (x-3)^2 + 2 -/
def given_parabola : Parabola where
  equation := fun x ↦ (x - 3)^2 + 2
  a := 1
  h := 3
  k := 2

/-- Theorem: The focus of the parabola y = (x-3)^2 + 2 is at the point (3, 9/4) -/
theorem focus_of_given_parabola :
  focus given_parabola = (3, 9/4) := by sorry

end NUMINAMATH_CALUDE_focus_of_given_parabola_l1862_186263


namespace NUMINAMATH_CALUDE_triangular_number_gcd_bound_l1862_186279

/-- The nth triangular number -/
def T (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The statement to be proved -/
theorem triangular_number_gcd_bound :
  (∀ n : ℕ, n > 0 → Nat.gcd (8 * T n) (n + 1) ≤ 4) ∧
  (∃ n : ℕ, n > 0 ∧ Nat.gcd (8 * T n) (n + 1) = 4) := by
  sorry

end NUMINAMATH_CALUDE_triangular_number_gcd_bound_l1862_186279


namespace NUMINAMATH_CALUDE_units_digit_of_29_power_8_7_l1862_186220

theorem units_digit_of_29_power_8_7 : 29^(8^7) % 10 = 1 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_29_power_8_7_l1862_186220


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1862_186278

theorem trigonometric_identities (θ : Real) 
  (h : (2 - Real.tan θ) / (1 + Real.tan θ) = 1) : 
  Real.tan (2 * θ) = 4/3 ∧ 
  (Real.sin θ + Real.cos θ) / (Real.cos θ - 3 * Real.sin θ) = -3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1862_186278


namespace NUMINAMATH_CALUDE_ellipse_with_foci_on_y_axis_l1862_186223

theorem ellipse_with_foci_on_y_axis (m n : ℝ) (h1 : m > n) (h2 : n > 0) :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ 
  ∀ (x y : ℝ), m * x^2 + n * y^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_with_foci_on_y_axis_l1862_186223


namespace NUMINAMATH_CALUDE_fraction_invariance_l1862_186293

theorem fraction_invariance (x y : ℝ) (square : ℝ) :
  (2 * x * y) / (x^2 + square) = (2 * (3*x) * (3*y)) / ((3*x)^2 + square) →
  square = y^2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_invariance_l1862_186293


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l1862_186251

/-- Theorem: Sum of reciprocals squared --/
theorem sum_of_reciprocals_squared :
  let a := Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 15
  let b := -Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 15
  let c := Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 15
  let d := -Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 15
  (1/a + 1/b + 1/c + 1/d)^2 = 960/3481 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l1862_186251


namespace NUMINAMATH_CALUDE_cloth_sale_loss_per_metre_l1862_186249

/-- Calculates the loss per metre for a cloth sale -/
theorem cloth_sale_loss_per_metre
  (total_metres : ℕ)
  (total_selling_price : ℕ)
  (cost_price_per_metre : ℕ)
  (h1 : total_metres = 600)
  (h2 : total_selling_price = 18000)
  (h3 : cost_price_per_metre = 35) :
  (cost_price_per_metre * total_metres - total_selling_price) / total_metres = 5 := by
  sorry

#check cloth_sale_loss_per_metre

end NUMINAMATH_CALUDE_cloth_sale_loss_per_metre_l1862_186249


namespace NUMINAMATH_CALUDE_ducks_at_lake_michigan_l1862_186235

theorem ducks_at_lake_michigan (ducks_north_pond : ℕ) (ducks_lake_michigan : ℕ) : 
  ducks_north_pond = 2 * ducks_lake_michigan + 6 →
  ducks_north_pond = 206 →
  ducks_lake_michigan = 100 := by
sorry

end NUMINAMATH_CALUDE_ducks_at_lake_michigan_l1862_186235


namespace NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l1862_186260

/-- Given three positive integers with LCM 45600 and product 109183500000, their HCF is 2393750 -/
theorem hcf_from_lcm_and_product (a b c : ℕ+) 
  (h_lcm : Nat.lcm (a.val) (Nat.lcm (b.val) (c.val)) = 45600)
  (h_product : a * b * c = 109183500000) :
  Nat.gcd (a.val) (Nat.gcd (b.val) (c.val)) = 2393750 := by
  sorry

end NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l1862_186260


namespace NUMINAMATH_CALUDE_expenditure_ratio_l1862_186219

theorem expenditure_ratio (rajan_income balan_income rajan_expenditure balan_expenditure : ℚ) : 
  (rajan_income / balan_income = 7 / 6) →
  (rajan_income = 7000) →
  (rajan_income - rajan_expenditure = 1000) →
  (balan_income - balan_expenditure = 1000) →
  (rajan_expenditure / balan_expenditure = 6 / 5) :=
by
  sorry

end NUMINAMATH_CALUDE_expenditure_ratio_l1862_186219


namespace NUMINAMATH_CALUDE_bridge_length_proof_l1862_186205

/-- Given a train that crosses a bridge and passes a lamp post, prove the length of the bridge. -/
theorem bridge_length_proof (train_length : ℝ) (bridge_crossing_time : ℝ) (lamp_post_passing_time : ℝ)
  (h1 : train_length = 400)
  (h2 : bridge_crossing_time = 45)
  (h3 : lamp_post_passing_time = 15) :
  (bridge_crossing_time * train_length / lamp_post_passing_time) - train_length = 800 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_proof_l1862_186205


namespace NUMINAMATH_CALUDE_root_sum_ratio_l1862_186210

theorem root_sum_ratio (m₁ m₂ : ℝ) : 
  (∃ a b : ℝ, m₁ * (a^2 - 2*a) + 3*a + 7 = 0 ∧ 
              m₂ * (b^2 - 2*b) + 3*b + 7 = 0 ∧ 
              a/b + b/a = 9/10) →
  m₁/m₂ + m₂/m₁ = ((323/40)^2 * 4 - 18) / 9 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_ratio_l1862_186210


namespace NUMINAMATH_CALUDE_exp_three_has_property_M_g_property_M_iff_l1862_186239

-- Define property M
def has_property_M (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f (x₀ + 1) = f x₀ + f 1

-- Statement for f(x) = 3^x
theorem exp_three_has_property_M :
  ∃ x₀ : ℝ, (3 : ℝ)^(x₀ + 1) = (3 : ℝ)^x₀ + (3 : ℝ)^1 :=
sorry

-- Define g(x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (a / (2 * x^2 + 1))

-- Statement for g(x)
theorem g_property_M_iff (a : ℝ) :
  (a > 0) →
  (has_property_M (g a) ↔ 6 - 3 * Real.sqrt 3 ≤ a ∧ a ≤ 6 + 3 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_exp_three_has_property_M_g_property_M_iff_l1862_186239


namespace NUMINAMATH_CALUDE_surface_area_increase_percentage_l1862_186289

/-- The percentage increase in surface area when placing a hemispherical cap on a sphere -/
theorem surface_area_increase_percentage (R : ℝ) (R_pos : R > 0) : 
  let sphere_area := 4 * Real.pi * R^2
  let cap_radius := R * Real.sqrt 3 / 2
  let cap_area := 2 * Real.pi * cap_radius^2
  let covered_cap_height := R / 2
  let covered_cap_area := 2 * Real.pi * R * covered_cap_height
  let area_increase := cap_area - covered_cap_area
  area_increase / sphere_area * 100 = 12.5 :=
sorry

end NUMINAMATH_CALUDE_surface_area_increase_percentage_l1862_186289


namespace NUMINAMATH_CALUDE_loan_interest_rate_l1862_186213

theorem loan_interest_rate (principal : ℝ) (total_paid : ℝ) (time : ℝ) : 
  principal = 150 → 
  total_paid = 159 → 
  time = 1 → 
  (total_paid - principal) / (principal * time) = 0.06 := by
sorry

end NUMINAMATH_CALUDE_loan_interest_rate_l1862_186213


namespace NUMINAMATH_CALUDE_true_discount_proof_l1862_186208

/-- Calculates the true discount given the banker's discount and sum due -/
def true_discount (bankers_discount : ℚ) (sum_due : ℚ) : ℚ :=
  let a : ℚ := 1
  let b : ℚ := sum_due
  let c : ℚ := -sum_due * bankers_discount
  (-b + (b^2 - 4*a*c).sqrt) / (2*a)

/-- Proves that the true discount is 246 given the banker's discount of 288 and sum due of 1440 -/
theorem true_discount_proof (bankers_discount sum_due : ℚ) 
  (h1 : bankers_discount = 288)
  (h2 : sum_due = 1440) : 
  true_discount bankers_discount sum_due = 246 := by
  sorry

#eval true_discount 288 1440

end NUMINAMATH_CALUDE_true_discount_proof_l1862_186208


namespace NUMINAMATH_CALUDE_tan_theta_value_l1862_186290

theorem tan_theta_value (θ : Real) 
  (h1 : 0 < θ) (h2 : θ < π/2)
  (h3 : (Real.sin θ + Real.cos θ)^2 + Real.sqrt 3 * Real.cos (2*θ) = 3) :
  Real.tan θ = 2 - Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_value_l1862_186290


namespace NUMINAMATH_CALUDE_harrys_family_age_ratio_l1862_186237

/-- Given Harry's age, the age difference between Harry and his father, and his mother's age when she gave birth to him, 
    prove that the ratio of the age difference between Harry's parents to Harry's age is 1:25. -/
theorem harrys_family_age_ratio (harry_age : ℕ) (father_age_diff : ℕ) (mother_age_at_birth : ℕ)
  (h1 : harry_age = 50)
  (h2 : father_age_diff = 24)
  (h3 : mother_age_at_birth = 22) :
  (father_age_diff + harry_age - (mother_age_at_birth + harry_age)) / harry_age = 1 / 25 := by
  sorry

end NUMINAMATH_CALUDE_harrys_family_age_ratio_l1862_186237


namespace NUMINAMATH_CALUDE_shaded_area_is_three_point_five_l1862_186242

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  L : ℝ × ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ
  O : ℝ × ℝ
  Q : ℝ × ℝ
  P : ℝ × ℝ
  h_dimensions : M.1 - L.1 = 4 ∧ O.2 - M.2 = 5
  h_equal_segments : 
    (M.1 - L.1 = 1) ∧ 
    (Q.2 - M.2 = 1) ∧ 
    (P.1 - Q.1 = 1) ∧ 
    (O.2 - P.2 = 1)

/-- The area of the shaded region in the special rectangle -/
def shadedArea (r : SpecialRectangle) : ℝ := sorry

/-- Theorem stating that the shaded area is 3.5 -/
theorem shaded_area_is_three_point_five (r : SpecialRectangle) : 
  shadedArea r = 3.5 := by sorry

end NUMINAMATH_CALUDE_shaded_area_is_three_point_five_l1862_186242


namespace NUMINAMATH_CALUDE_smallest_prime_12_less_than_square_l1862_186234

theorem smallest_prime_12_less_than_square : ∃ (n : ℕ), 
  (∀ (m : ℕ), m < n → ¬(∃ (k : ℕ), Prime (k^2 - 12) ∧ k^2 - 12 > 0)) ∧ 
  Prime (n^2 - 12) ∧ 
  n^2 - 12 > 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_12_less_than_square_l1862_186234


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_tangent_l1862_186291

/-- The value of n for which the ellipse 2x^2 + 3y^2 = 6 and the hyperbola 3x^2 - n(y-1)^2 = 3 are tangent -/
def tangent_n : ℝ := -6

/-- The equation of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := 2 * x^2 + 3 * y^2 = 6

/-- The equation of the hyperbola -/
def is_on_hyperbola (x y n : ℝ) : Prop := 3 * x^2 - n * (y - 1)^2 = 3

/-- Two curves are tangent if they intersect at exactly one point -/
def are_tangent (f g : ℝ → ℝ → Prop) : Prop :=
  ∃! p : ℝ × ℝ, f p.1 p.2 ∧ g p.1 p.2

theorem ellipse_hyperbola_tangent :
  are_tangent (λ x y => is_on_ellipse x y) (λ x y => is_on_hyperbola x y tangent_n) :=
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_tangent_l1862_186291


namespace NUMINAMATH_CALUDE_katya_magic_pen_problem_l1862_186243

theorem katya_magic_pen_problem (p_katya : ℚ) (p_pen : ℚ) (total_problems : ℕ) (min_correct : ℚ) :
  p_katya = 4/5 →
  p_pen = 1/2 →
  total_problems = 20 →
  min_correct = 13 →
  ∃ x : ℕ, x ≥ 10 ∧
    x * p_katya + (total_problems - x) * p_pen ≥ min_correct ∧
    ∀ y : ℕ, y < 10 → y * p_katya + (total_problems - y) * p_pen < min_correct :=
by sorry

end NUMINAMATH_CALUDE_katya_magic_pen_problem_l1862_186243


namespace NUMINAMATH_CALUDE_f_properties_l1862_186214

noncomputable def f (x : ℝ) := 2 * Real.cos x * (Real.cos x + Real.sqrt 3 * Real.sin x)

theorem f_properties :
  ∃ (T : ℝ),
    (∀ x, f (x + T) = f x) ∧
    T = π ∧
    (∀ k : ℤ, StrictMonoOn f (Set.Ioo (↑k * π - π / 3) (↑k * π + π / 6))) ∧
    (∀ x ∈ Set.Icc 0 (π / 2), f x ≤ 3) ∧
    (∃ x ∈ Set.Icc 0 (π / 2), f x = 3) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1862_186214


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1862_186255

theorem closest_integer_to_cube_root : ∃ n : ℤ, 
  n = 10 ∧ ∀ m : ℤ, m ≠ n → |n - (7^3 + 9^3 + 3)^(1/3)| < |m - (7^3 + 9^3 + 3)^(1/3)| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1862_186255


namespace NUMINAMATH_CALUDE_cone_volume_l1862_186200

/-- Given a cone whose lateral surface is an arc of a sector with radius 2 and arc length 2π,
    prove that its volume is (√3 * π) / 3 -/
theorem cone_volume (r : Real) (h : Real) :
  (r = 1) →
  (h^2 + r^2 = 2^2) →
  (1/3 * π * r^2 * h = (Real.sqrt 3 * π) / 3) := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l1862_186200


namespace NUMINAMATH_CALUDE_marias_cupcakes_l1862_186226

/-- 
Given that Maria made some cupcakes, sold 5, made 10 more, and ended up with 24 cupcakes,
this theorem proves that she initially made 19 cupcakes.
-/
theorem marias_cupcakes (x : ℕ) 
  (h : x - 5 + 10 = 24) : x = 19 := by
  sorry

end NUMINAMATH_CALUDE_marias_cupcakes_l1862_186226


namespace NUMINAMATH_CALUDE_chad_dog_food_packages_l1862_186240

/-- Given Chad's purchase of cat and dog food, prove he bought 2 packages of dog food -/
theorem chad_dog_food_packages : 
  ∀ (cat_packages dog_packages : ℕ),
  cat_packages = 6 →
  ∀ (cat_cans_per_package dog_cans_per_package : ℕ),
  cat_cans_per_package = 9 →
  dog_cans_per_package = 3 →
  cat_packages * cat_cans_per_package = dog_packages * dog_cans_per_package + 48 →
  dog_packages = 2 :=
by sorry

end NUMINAMATH_CALUDE_chad_dog_food_packages_l1862_186240


namespace NUMINAMATH_CALUDE_library_books_count_l1862_186284

theorem library_books_count : ∃ (n : ℕ), 
  500 < n ∧ n < 650 ∧ 
  ∃ (r : ℕ), n = 12 * r + 7 ∧
  ∃ (l : ℕ), n = 25 * l - 5 ∧
  n = 595 := by
  sorry

end NUMINAMATH_CALUDE_library_books_count_l1862_186284


namespace NUMINAMATH_CALUDE_inequality_preservation_l1862_186280

theorem inequality_preservation (m n : ℝ) (h : m > n) : m - 6 > n - 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1862_186280


namespace NUMINAMATH_CALUDE_greatest_common_multiple_15_20_under_125_l1862_186215

theorem greatest_common_multiple_15_20_under_125 : 
  ∃ n : ℕ, n = 120 ∧ 
  (∀ m : ℕ, m < 125 ∧ 15 ∣ m ∧ 20 ∣ m → m ≤ n) ∧
  15 ∣ n ∧ 20 ∣ n ∧ n < 125 :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_15_20_under_125_l1862_186215


namespace NUMINAMATH_CALUDE_min_sum_squares_l1862_186222

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (min : ℝ), min = 16/3 ∧ x^2 + y^2 + z^2 ≥ min ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀^3 + y₀^3 + z₀^3 - 3*x₀*y₀*z₀ = 8 ∧ x₀^2 + y₀^2 + z₀^2 = min :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1862_186222


namespace NUMINAMATH_CALUDE_harry_book_pages_l1862_186204

/-- Given that Selena's book has x pages and Harry's book has y fewer pages than half of Selena's book,
    prove that the number of pages in Harry's book is (x/2) - y. -/
theorem harry_book_pages (x y : ℕ) (selena_pages : ℕ) (harry_pages : ℕ) 
    (h1 : selena_pages = x)
    (h2 : harry_pages = selena_pages / 2 - y) :
  harry_pages = x / 2 - y := by
  sorry

end NUMINAMATH_CALUDE_harry_book_pages_l1862_186204


namespace NUMINAMATH_CALUDE_forty_students_in_music_l1862_186211

/-- Represents the number of students in various categories in a high school. -/
structure SchoolData where
  total : ℕ
  art : ℕ
  both : ℕ
  neither : ℕ

/-- Calculates the number of students taking music based on the given school data. -/
def studentsInMusic (data : SchoolData) : ℕ :=
  data.total - data.neither - (data.art - data.both)

/-- Theorem stating that given the specific school data, 40 students are taking music. -/
theorem forty_students_in_music :
  let data : SchoolData := {
    total := 500,
    art := 20,
    both := 10,
    neither := 450
  }
  studentsInMusic data = 40 := by
  sorry


end NUMINAMATH_CALUDE_forty_students_in_music_l1862_186211


namespace NUMINAMATH_CALUDE_group_size_l1862_186283

theorem group_size (total : ℕ) 
  (h1 : (total : ℚ) / 5 = (0.12 * total + 64 : ℚ)) : total = 800 := by
  sorry

end NUMINAMATH_CALUDE_group_size_l1862_186283


namespace NUMINAMATH_CALUDE_triangle_problem_l1862_186259

theorem triangle_problem (a b c A B C : ℝ) : 
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  (2 * a - c) * Real.sin A + (2 * c - a) * Real.sin C = 2 * b * Real.sin B →
  b = 1 →
  B = π / 3 ∧ 
  ∃ (p : ℝ), p = a + b + c ∧ Real.sqrt 3 + 1 < p ∧ p ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1862_186259


namespace NUMINAMATH_CALUDE_exists_n_with_totient_inequality_l1862_186271

open Nat

theorem exists_n_with_totient_inequality : 
  ∃ (n : ℕ), n > 0 ∧ totient (2*n - 1) + totient (2*n + 1) < (1 : ℚ) / 1000 * totient (2*n) :=
by sorry

end NUMINAMATH_CALUDE_exists_n_with_totient_inequality_l1862_186271


namespace NUMINAMATH_CALUDE_two_numbers_sum_product_l1862_186256

theorem two_numbers_sum_product (S P : ℝ) (h : S^2 ≥ 4*P) :
  ∃! (x y : ℝ × ℝ), (x.1 + x.2 = S ∧ x.1 * x.2 = P) ∧ (y.1 + y.2 = S ∧ y.1 * y.2 = P) ∧ x ≠ y :=
by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_product_l1862_186256


namespace NUMINAMATH_CALUDE_ab_value_l1862_186232

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧
    Real.sqrt (2 * log a) = m ∧
    Real.sqrt (2 * log b) = n ∧
    log (Real.sqrt a) = (m^2 : ℝ) / 4 ∧
    log (Real.sqrt b) = (n^2 : ℝ) / 4 ∧
    m + n + (m^2 : ℝ) / 4 + (n^2 : ℝ) / 4 = 104) →
  a * b = 10^260 := by
sorry

end NUMINAMATH_CALUDE_ab_value_l1862_186232


namespace NUMINAMATH_CALUDE_lettuce_types_count_l1862_186269

/-- The number of lunch combo options given the number of lettuce types -/
def lunch_combos (lettuce_types : ℕ) : ℕ :=
  lettuce_types * 3 * 4 * 2

/-- Theorem stating that there are 2 types of lettuce -/
theorem lettuce_types_count : ∃ (n : ℕ), n = 2 ∧ lunch_combos n = 48 := by
  sorry

end NUMINAMATH_CALUDE_lettuce_types_count_l1862_186269
