import Mathlib

namespace donut_combinations_l522_52282

theorem donut_combinations : 
  let total_donuts : ℕ := 8
  let donut_types : ℕ := 5
  let remaining_donuts : ℕ := total_donuts - donut_types
  Nat.choose (remaining_donuts + donut_types - 1) (donut_types - 1) = 35 :=
by sorry

end donut_combinations_l522_52282


namespace simple_interest_rate_l522_52290

/-- Simple interest rate calculation -/
theorem simple_interest_rate 
  (principal : ℝ) 
  (final_amount : ℝ) 
  (time : ℝ) 
  (h1 : principal = 750) 
  (h2 : final_amount = 900) 
  (h3 : time = 8) :
  (final_amount - principal) * 100 / (principal * time) = 2.5 := by
sorry

end simple_interest_rate_l522_52290


namespace low_card_value_is_one_l522_52293

/-- A card type in the high-low game -/
inductive CardType
| High
| Low

/-- The high-low card game -/
structure HighLowGame where
  total_cards : Nat
  high_cards : Nat
  low_cards : Nat
  high_card_value : Nat
  low_card_value : Nat
  target_points : Nat
  target_low_cards : Nat
  ways_to_reach_target : Nat

/-- Conditions for the high-low game -/
def game_conditions (g : HighLowGame) : Prop :=
  g.total_cards = 52 ∧
  g.high_cards = g.low_cards ∧
  g.high_cards + g.low_cards = g.total_cards ∧
  g.high_card_value = 2 ∧
  g.target_points = 5 ∧
  g.target_low_cards = 3 ∧
  g.ways_to_reach_target = 4

/-- Theorem stating that under the given conditions, the low card value must be 1 -/
theorem low_card_value_is_one (g : HighLowGame) :
  game_conditions g → g.low_card_value = 1 := by
  sorry

end low_card_value_is_one_l522_52293


namespace probability_of_sum_15_is_correct_l522_52270

/-- Represents a standard 52-card deck -/
def standardDeck : Nat := 52

/-- Represents the number of cards for each value in a standard deck -/
def cardsPerValue : Nat := 4

/-- Represents the probability of drawing two number cards (2 through 10) 
    from a standard 52-card deck that total 15 -/
def probabilityOfSum15 : ℚ := 28 / 221

theorem probability_of_sum_15_is_correct : 
  probabilityOfSum15 = (
    -- Probability of drawing a 5, 6, 7, 8, or 9 first, then completing the pair
    (5 * cardsPerValue * 4 * cardsPerValue) / (standardDeck * (standardDeck - 1)) +
    -- Probability of drawing a 10 first, then a 5
    (cardsPerValue * cardsPerValue) / (standardDeck * (standardDeck - 1))
  ) := by sorry

end probability_of_sum_15_is_correct_l522_52270


namespace solve_average_salary_l522_52258

def average_salary_problem (num_employees : ℕ) (manager_salary : ℕ) (avg_increase : ℕ) : Prop :=
  let total_salary := num_employees * (manager_salary / (num_employees + 1) - avg_increase)
  let new_total_salary := total_salary + manager_salary
  let new_average := new_total_salary / (num_employees + 1)
  (manager_salary / (num_employees + 1) - avg_increase) = 2400 ∧
  new_average = (manager_salary / (num_employees + 1) - avg_increase) + avg_increase

theorem solve_average_salary :
  average_salary_problem 24 4900 100 := by
  sorry

end solve_average_salary_l522_52258


namespace max_min_product_l522_52278

theorem max_min_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (sum_eq : a + b + c = 12) (sum_prod_eq : a * b + b * c + c * a = 32) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 4 ∧
  ∀ (m' : ℝ), m' = min (a * b) (min (b * c) (c * a)) → m' ≤ 4 := by
sorry

end max_min_product_l522_52278


namespace g_symmetric_to_f_max_value_of_a_l522_52295

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 4*x + 3

-- Define the function g (to be proved)
def g (x : ℝ) : ℝ := x^2 - 8*x + 15

-- Theorem 1: Prove that g is symmetric to f about x=1
theorem g_symmetric_to_f : ∀ x : ℝ, g x = f (2 - x) := by sorry

-- Theorem 2: Prove the maximum value of a
theorem max_value_of_a : 
  (∀ x : ℝ, g x ≥ g 6 - 4) ∧ 
  (∀ a : ℝ, (∀ x : ℝ, g x ≥ g a - 4) → a ≤ 6) := by sorry

end g_symmetric_to_f_max_value_of_a_l522_52295


namespace min_value_fraction_l522_52266

theorem min_value_fraction (a b : ℝ) (h1 : a > b) (h2 : a * b = 1) :
  (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2 := by
  sorry

end min_value_fraction_l522_52266


namespace slices_in_large_pizza_l522_52223

/-- Given that Mary orders 2 large pizzas, eats 7 slices, and has 9 slices remaining,
    prove that there are 8 slices in a large pizza. -/
theorem slices_in_large_pizza :
  ∀ (total_pizzas : ℕ) (slices_eaten : ℕ) (slices_remaining : ℕ),
    total_pizzas = 2 →
    slices_eaten = 7 →
    slices_remaining = 9 →
    (slices_remaining + slices_eaten) / total_pizzas = 8 :=
by
  sorry

end slices_in_large_pizza_l522_52223


namespace unique_integer_sum_l522_52277

theorem unique_integer_sum (y : ℝ) : 
  y = Real.sqrt ((Real.sqrt 77) / 2 + 5 / 2) →
  ∃! (d e f : ℕ+), 
    y^100 = 2*y^98 + 18*y^96 + 15*y^94 - y^50 + (d:ℝ)*y^46 + (e:ℝ)*y^44 + (f:ℝ)*y^40 ∧
    d + e + f = 242 := by
  sorry

end unique_integer_sum_l522_52277


namespace unique_integer_divisible_by_14_with_sqrt_between_25_and_25_3_l522_52228

theorem unique_integer_divisible_by_14_with_sqrt_between_25_and_25_3 :
  ∃! n : ℕ+, 14 ∣ n ∧ 25 < (n : ℝ).sqrt ∧ (n : ℝ).sqrt < 25.3 :=
by
  -- The proof would go here
  sorry

end unique_integer_divisible_by_14_with_sqrt_between_25_and_25_3_l522_52228


namespace max_m_inequality_l522_52264

theorem max_m_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ m : ℝ, ∀ a b : ℝ, a > 0 → b > 0 → 4/a + 1/b ≥ m/(a+4*b)) ∧
  (∀ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → 4/a + 1/b ≥ m/(a+4*b)) → m ≤ 16) :=
sorry

end max_m_inequality_l522_52264


namespace min_value_function_min_value_attained_l522_52232

theorem min_value_function (x : ℝ) (h : x > 0) : 3 * x + 12 / x^2 ≥ 9 :=
sorry

theorem min_value_attained : ∃ x : ℝ, x > 0 ∧ 3 * x + 12 / x^2 = 9 :=
sorry

end min_value_function_min_value_attained_l522_52232


namespace coefficient_of_z_in_equation1_l522_52262

-- Define the system of equations
def equation1 (x y z : ℚ) : Prop := 6 * x - 5 * y + z = 22 / 3
def equation2 (x y z : ℚ) : Prop := 4 * x + 8 * y - 11 * z = 7
def equation3 (x y z : ℚ) : Prop := 5 * x - 6 * y + 2 * z = 12

-- Define the sum condition
def sum_condition (x y z : ℚ) : Prop := x + y + z = 10

-- Theorem statement
theorem coefficient_of_z_in_equation1 (x y z : ℚ) 
  (eq1 : equation1 x y z) (eq2 : equation2 x y z) (eq3 : equation3 x y z) 
  (sum : sum_condition x y z) : 
  ∃ (a b c : ℚ), equation1 x y z ↔ a * x + b * y + 1 * z = 22 / 3 :=
sorry

end coefficient_of_z_in_equation1_l522_52262


namespace nine_b_value_l522_52240

theorem nine_b_value (a b : ℚ) (h1 : 8 * a + 3 * b = 0) (h2 : b - 3 = a) : 9 * b = 216 / 11 := by
  sorry

end nine_b_value_l522_52240


namespace car_speed_problem_l522_52226

/-- Proves that the speed of Car A is 70 km/h given the conditions of the problem -/
theorem car_speed_problem (time : ℝ) (speed_B : ℝ) (ratio : ℝ) :
  time = 10 →
  speed_B = 35 →
  ratio = 2 →
  let distance_A := time * (ratio * speed_B)
  let distance_B := time * speed_B
  (distance_A / distance_B = ratio) →
  (ratio * speed_B = 70) :=
by sorry

end car_speed_problem_l522_52226


namespace original_apples_in_B_l522_52242

/-- Represents the number of apples in each basket -/
structure AppleBaskets where
  A : ℕ  -- Number of apples in basket A
  B : ℕ  -- Number of apples in basket B
  C : ℕ  -- Number of apples in basket C

/-- The conditions of the apple basket problem -/
def apple_basket_conditions (baskets : AppleBaskets) : Prop :=
  -- Condition 1: The number of apples in basket C is twice the number of apples in basket A
  baskets.C = 2 * baskets.A ∧
  -- Condition 2: After transferring 12 apples from B to A, A has 24 less than C
  baskets.A + 12 = baskets.C - 24 ∧
  -- Condition 3: After the transfer, B has 6 more than C
  baskets.B - 12 = baskets.C + 6

theorem original_apples_in_B (baskets : AppleBaskets) :
  apple_basket_conditions baskets → baskets.B = 90 := by
  sorry

end original_apples_in_B_l522_52242


namespace toms_fruit_purchase_cost_l522_52276

/-- Calculates the total cost of a fruit purchase with a quantity-based discount --/
def fruitPurchaseCost (lemonPrice papayaPrice mangoPrice : ℕ) 
                      (lemonQty papayaQty mangoQty : ℕ) 
                      (fruitPerDiscount : ℕ) (discountAmount : ℕ) : ℕ :=
  let totalCost := lemonPrice * lemonQty + papayaPrice * papayaQty + mangoPrice * mangoQty
  let totalFruits := lemonQty + papayaQty + mangoQty
  let discountQty := totalFruits / fruitPerDiscount
  totalCost - discountQty * discountAmount

/-- Theorem: Tom's fruit purchase costs $21 --/
theorem toms_fruit_purchase_cost : 
  fruitPurchaseCost 2 1 4 6 4 2 4 1 = 21 := by
  sorry

#eval fruitPurchaseCost 2 1 4 6 4 2 4 1

end toms_fruit_purchase_cost_l522_52276


namespace gwen_birthday_money_l522_52234

/-- The amount of money Gwen received from her mom -/
def mom_money : ℕ := 8

/-- The difference between the money Gwen received from her mom and dad -/
def difference : ℕ := 3

/-- The amount of money Gwen received from her dad -/
def dad_money : ℕ := mom_money - difference

theorem gwen_birthday_money : dad_money = 5 := by
  sorry

end gwen_birthday_money_l522_52234


namespace sqrt_x_minus_8_range_l522_52279

-- Define the condition for a meaningful square root
def meaningful_sqrt (x : ℝ) : Prop := x - 8 ≥ 0

-- Theorem stating the range of x for which √(x-8) is meaningful
theorem sqrt_x_minus_8_range (x : ℝ) : 
  meaningful_sqrt x ↔ x ≥ 8 := by
sorry

end sqrt_x_minus_8_range_l522_52279


namespace merchant_profit_percentage_l522_52281

/-- Calculates the profit percentage for a merchant who marks up goods by 50%
    and then offers a 10% discount on the marked price. -/
theorem merchant_profit_percentage (cost_price : ℝ) (cost_price_pos : 0 < cost_price) :
  let markup_percentage : ℝ := 0.5
  let discount_percentage : ℝ := 0.1
  let marked_price : ℝ := cost_price * (1 + markup_percentage)
  let selling_price : ℝ := marked_price * (1 - discount_percentage)
  let profit : ℝ := selling_price - cost_price
  let profit_percentage : ℝ := profit / cost_price * 100
  profit_percentage = 35 := by
  sorry


end merchant_profit_percentage_l522_52281


namespace triangle_properties_l522_52243

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  2 * a * Real.sin (C + π / 6) = b + c →
  B = π / 4 →
  b - a = Real.sqrt 2 - Real.sqrt 3 →
  A = π / 3 ∧
  (1 / 2) * a * b * Real.sin C = (3 + Real.sqrt 3) / 4 :=
by sorry

end triangle_properties_l522_52243


namespace total_amount_calculation_l522_52227

theorem total_amount_calculation (two_won_bills : ℕ) (one_won_bills : ℕ) : 
  two_won_bills = 8 → one_won_bills = 2 → two_won_bills * 2 + one_won_bills * 1 = 18 := by
  sorry

end total_amount_calculation_l522_52227


namespace rectangle_square_overlap_ratio_l522_52218

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with side length -/
structure Square where
  side : ℝ

/-- The area of overlap between a rectangle and a square -/
def overlap_area (r : Rectangle) (s : Square) : ℝ := sorry

/-- The theorem stating the ratio of rectangle's width to height -/
theorem rectangle_square_overlap_ratio 
  (r : Rectangle) 
  (s : Square) 
  (h1 : overlap_area r s = 0.6 * r.width * r.height) 
  (h2 : overlap_area r s = 0.3 * s.side * s.side) : 
  r.width / r.height = 12.5 := by sorry

end rectangle_square_overlap_ratio_l522_52218


namespace distance_between_points_distance_X_to_Y_l522_52251

/-- The distance between two points X and Y, given the walking speeds of two people
    and the distance one person has walked when they meet. -/
theorem distance_between_points (yolanda_speed bob_speed : ℝ) 
  (time_difference : ℝ) (bob_distance : ℝ) : ℝ :=
  let total_time := bob_distance / bob_speed
  let yolanda_time := total_time + time_difference
  let yolanda_distance := yolanda_time * yolanda_speed
  bob_distance + yolanda_distance

/-- The specific problem statement -/
theorem distance_X_to_Y : distance_between_points 1 2 1 20 = 31 := by
  sorry

end distance_between_points_distance_X_to_Y_l522_52251


namespace magnitude_of_complex_power_l522_52283

theorem magnitude_of_complex_power : 
  Complex.abs ((5 : ℂ) - (2 * Real.sqrt 3) * Complex.I) ^ 4 = 1369 := by sorry

end magnitude_of_complex_power_l522_52283


namespace perfect_square_condition_l522_52207

theorem perfect_square_condition (a b k : ℝ) : 
  (∃ (c : ℝ), a^2 + 2*(k-3)*a*b + 9*b^2 = c^2) → (k = 0 ∨ k = 6) := by
  sorry

end perfect_square_condition_l522_52207


namespace banana_arrangements_count_l522_52203

/-- The number of unique arrangements of the letters in BANANA -/
def banana_arrangements : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of unique arrangements of BANANA is 60 -/
theorem banana_arrangements_count : banana_arrangements = 60 := by
  sorry

end banana_arrangements_count_l522_52203


namespace intersection_of_A_and_B_l522_52217

def A : Set ℝ := {x | x ≥ -4}
def B : Set ℝ := {x | x ≤ 3}

theorem intersection_of_A_and_B : A ∩ B = {x | -4 ≤ x ∧ x ≤ 3} := by sorry

end intersection_of_A_and_B_l522_52217


namespace distribute_six_students_two_activities_l522_52248

/-- The number of ways to distribute n students between 2 activities,
    where each activity can have at most k students. -/
def distribute_students (n k : ℕ) : ℕ :=
  Nat.choose n k + Nat.choose n (n / 2)

/-- Theorem stating that the number of ways to distribute 6 students
    between 2 activities, where each activity can have at most 4 students,
    is equal to 35. -/
theorem distribute_six_students_two_activities :
  distribute_students 6 4 = 35 := by
  sorry

#eval distribute_students 6 4

end distribute_six_students_two_activities_l522_52248


namespace function_range_l522_52206

/-- The range of the function f(x) = (e^(3x) - 2) / (e^(3x) + 2) is (-1, 1) -/
theorem function_range (x : ℝ) : 
  -1 < (Real.exp (3 * x) - 2) / (Real.exp (3 * x) + 2) ∧ 
  (Real.exp (3 * x) - 2) / (Real.exp (3 * x) + 2) < 1 := by
  sorry

end function_range_l522_52206


namespace decimal_point_problem_l522_52237

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 / x) : x = Real.sqrt 30 / 100 := by
  sorry

end decimal_point_problem_l522_52237


namespace sum_of_cubes_special_case_l522_52285

theorem sum_of_cubes_special_case (x y : ℝ) (h1 : x + y = 1) (h2 : x * y = 1) : 
  x^3 + y^3 = -2 := by
sorry

end sum_of_cubes_special_case_l522_52285


namespace ellipse_equation_l522_52241

/-- The equation of an ellipse with given properties -/
theorem ellipse_equation (e : ℝ) (c : ℝ) (h1 : e = 1/2) (h2 : c = 1) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), (x^2 / (a^2) + y^2 / (b^2) = 1) ↔ (x^2 / 4 + y^2 / 3 = 1)) :=
sorry

end ellipse_equation_l522_52241


namespace batsman_average_theorem_l522_52204

/-- Represents a batsman's score data -/
structure BatsmanScore where
  initialAverage : ℝ
  inningsPlayed : ℕ
  newInningScore : ℝ
  averageIncrease : ℝ

/-- Theorem: If a batsman's average increases by 5 after scoring 100 in the 11th inning, 
    then his new average is 50 -/
theorem batsman_average_theorem (b : BatsmanScore) 
  (h1 : b.inningsPlayed = 10)
  (h2 : b.newInningScore = 100)
  (h3 : b.averageIncrease = 5)
  : b.initialAverage + b.averageIncrease = 50 := by
  sorry

#check batsman_average_theorem

end batsman_average_theorem_l522_52204


namespace neither_directly_nor_inversely_proportional_l522_52288

-- Define what it means for y to be directly proportional to x
def is_directly_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

-- Define what it means for y to be inversely proportional to x
def is_inversely_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x = k / x

-- Define the two equations
def eq_A (x y : ℝ) : Prop := x^2 + x*y = 0
def eq_D (x y : ℝ) : Prop := 4*x + y^2 = 7

-- Theorem statement
theorem neither_directly_nor_inversely_proportional :
  (¬ ∃ f : ℝ → ℝ, (∀ x y : ℝ, eq_A x y ↔ y = f x) ∧ 
    (is_directly_proportional f ∨ is_inversely_proportional f)) ∧
  (¬ ∃ g : ℝ → ℝ, (∀ x y : ℝ, eq_D x y ↔ y = g x) ∧ 
    (is_directly_proportional g ∨ is_inversely_proportional g)) :=
by sorry

end neither_directly_nor_inversely_proportional_l522_52288


namespace overlap_area_l522_52220

-- Define the points on a 2D grid
def Point := ℕ × ℕ

-- Define the rectangle
def rectangle : List Point := [(0, 0), (3, 0), (3, 2), (0, 2)]

-- Define the triangle
def triangle : List Point := [(2, 0), (2, 2), (4, 2)]

-- Function to calculate the area of a right triangle
def rightTriangleArea (base height : ℕ) : ℚ :=
  (base * height) / 2

-- Theorem stating that the overlapping area is 1 square unit
theorem overlap_area :
  let overlapBase := 1
  let overlapHeight := 2
  rightTriangleArea overlapBase overlapHeight = 1 := by
  sorry

end overlap_area_l522_52220


namespace dimes_in_jar_l522_52222

/-- Represents the number of coins of each type in the jar -/
structure CoinCount where
  dimes : ℕ
  quarters : ℕ

/-- Calculates the total value of coins in cents -/
def totalValue (coins : CoinCount) : ℕ :=
  coins.dimes * 10 + coins.quarters * 25

/-- Theorem stating that given the conditions, there are 15 dimes in the jar -/
theorem dimes_in_jar : ∃ (coins : CoinCount),
  coins.dimes = 3 * coins.quarters / 2 ∧
  totalValue coins = 400 ∧
  coins.dimes = 15 := by
  sorry

end dimes_in_jar_l522_52222


namespace dagger_example_l522_52214

-- Define the † operation
def dagger (m n p q : ℚ) : ℚ := m * p * (q / n)

-- Theorem statement
theorem dagger_example : dagger (7/12) (8/3) = 14 := by
  sorry

end dagger_example_l522_52214


namespace tiling_uniqueness_l522_52202

/-- A rectangular grid --/
structure RectangularGrid where
  rows : ℕ
  cols : ℕ

/-- A cell in the grid --/
structure Cell where
  row : ℕ
  col : ℕ

/-- A tiling of the grid --/
def Tiling (grid : RectangularGrid) := Set (Set Cell)

/-- The set of central cells for a given tiling --/
def CentralCells (grid : RectangularGrid) (tiling : Tiling grid) : Set Cell :=
  sorry

/-- Theorem: The set of central cells uniquely determines the tiling --/
theorem tiling_uniqueness (grid : RectangularGrid) 
  (tiling1 tiling2 : Tiling grid) :
  CentralCells grid tiling1 = CentralCells grid tiling2 → tiling1 = tiling2 :=
sorry

end tiling_uniqueness_l522_52202


namespace simplify_fraction_1_simplify_fraction_2_l522_52261

-- Part 1
theorem simplify_fraction_1 (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (3 * a^2 * b) / (6 * a * b^2 * c) = a / (2 * b * c) := by sorry

-- Part 2
theorem simplify_fraction_2 (x y : ℝ) (h : x ≠ y) :
  (2 * (x - y)^3) / (y - x) = -2 * (x - y)^2 := by sorry

end simplify_fraction_1_simplify_fraction_2_l522_52261


namespace polynomial_coefficients_sum_l522_52219

theorem polynomial_coefficients_sum (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, a₁ * (x - 1)^4 + a₂ * (x - 1)^3 + a₃ * (x - 1)^2 + a₄ * (x - 1) + a₅ = x^4) →
  a₂ + a₃ + a₄ = 14 := by
sorry

end polynomial_coefficients_sum_l522_52219


namespace gcf_of_26_and_16_l522_52229

theorem gcf_of_26_and_16 :
  let n : ℕ := 26
  let m : ℕ := 16
  let lcm_nm : ℕ := 52
  Nat.lcm n m = lcm_nm →
  Nat.gcd n m = 8 := by
sorry

end gcf_of_26_and_16_l522_52229


namespace extreme_value_condition_monotonicity_intervals_min_value_on_interval_l522_52255

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 1

-- Theorem 1: f(x) has an extreme value at x = 1 if and only if a = -1
theorem extreme_value_condition (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 1 ∧ |x - 1| < ε → f a x ≤ f a 1) ↔ a = -1 :=
sorry

-- Theorem 2: Monotonicity intervals depend on the value of a
theorem monotonicity_intervals (a : ℝ) :
  (a = 0 → ∀ (x y : ℝ), x < y → f a x < f a y) ∧
  (a > 0 → ∀ (x y : ℝ), (x < y ∧ y < -a) ∨ (x > 0 ∧ y > x) → f a x < f a y) ∧
  (a > 0 → ∀ (x y : ℝ), -a < x ∧ x < y ∧ y < 0 → f a x > f a y) ∧
  (a < 0 → ∀ (x y : ℝ), (x < y ∧ y < 0) ∨ (x > -a ∧ y > x) → f a x < f a y) ∧
  (a < 0 → ∀ (x y : ℝ), 0 < x ∧ x < y ∧ y < -a → f a x > f a y) :=
sorry

-- Theorem 3: Minimum value on [0, 2] depends on the value of a
theorem min_value_on_interval (a : ℝ) :
  (a ≥ 0 → ∀ (x : ℝ), x ∈ Set.Icc 0 2 → f a x ≥ f a 0) ∧
  (-2 < a ∧ a < 0 → ∀ (x : ℝ), x ∈ Set.Icc 0 2 → f a x ≥ f a (-a)) ∧
  (a ≤ -2 → ∀ (x : ℝ), x ∈ Set.Icc 0 2 → f a x ≥ f a 2) :=
sorry

end extreme_value_condition_monotonicity_intervals_min_value_on_interval_l522_52255


namespace complement_union_theorem_l522_52230

def U : Set Nat := {0, 1, 2, 3, 4}
def A : Set Nat := {0, 1, 2, 3}
def B : Set Nat := {2, 3, 4}

theorem complement_union_theorem :
  (U \ A) ∪ B = {2, 3, 4} := by sorry

end complement_union_theorem_l522_52230


namespace gary_book_multiple_l522_52213

/-- Proves that Gary's books are 5 times the combined number of Darla's and Katie's books -/
theorem gary_book_multiple (darla_books katie_books gary_books : ℕ) : 
  darla_books = 6 →
  katie_books = darla_books / 2 →
  gary_books = (darla_books + katie_books) * (gary_books / (darla_books + katie_books)) →
  darla_books + katie_books + gary_books = 54 →
  gary_books / (darla_books + katie_books) = 5 := by
sorry

end gary_book_multiple_l522_52213


namespace chord_length_concentric_circles_l522_52273

/-- Given two concentric circles with radii R and r, where the area of the ring between them is 18π,
    the length of a chord of the larger circle that is tangent to the smaller circle is 6√2. -/
theorem chord_length_concentric_circles (R r : ℝ) (h : R > r) :
  (π * R^2 - π * r^2 = 18 * π) →
  ∃ c : ℝ, c = 6 * Real.sqrt 2 ∧ c^2 = 4 * (R^2 - r^2) := by
  sorry

end chord_length_concentric_circles_l522_52273


namespace negative_a_fifth_squared_l522_52257

theorem negative_a_fifth_squared (a : ℝ) : (-a^5)^2 = a^10 := by
  sorry

end negative_a_fifth_squared_l522_52257


namespace min_value_expression_l522_52272

theorem min_value_expression (a b : ℝ) (h : a ≠ -1) :
  |a + b| + |1 / (a + 1) - b| ≥ 1 := by
sorry

end min_value_expression_l522_52272


namespace intersection_solution_set_l522_52287

def solution_set (f : ℝ → ℝ) : Set ℝ := {x | f x < 0}

def A : Set ℝ := solution_set (λ x => x^2 - 2*x - 3)
def B : Set ℝ := solution_set (λ x => x^2 + x - 6)

theorem intersection_solution_set (a b : ℝ) :
  solution_set (λ x => x^2 + a*x + b) = A ∩ B → a + b = -3 := by
  sorry

end intersection_solution_set_l522_52287


namespace soccer_games_played_l522_52224

theorem soccer_games_played (win_percentage : ℝ) (games_won : ℝ) (total_games : ℝ) : 
  win_percentage = 0.40 → games_won = 63.2 → win_percentage * total_games = games_won → total_games = 158 := by
  sorry

end soccer_games_played_l522_52224


namespace soap_survey_households_l522_52210

theorem soap_survey_households (total : ℕ) (neither : ℕ) (only_A : ℕ) (both : ℕ) :
  total = 160 ∧
  neither = 80 ∧
  only_A = 60 ∧
  both = 5 →
  total = neither + only_A + both + 3 * both :=
by sorry

end soap_survey_households_l522_52210


namespace binomial_expansion_problem_l522_52238

theorem binomial_expansion_problem (n : ℕ) (h : (2 : ℝ)^n = 256) :
  n = 8 ∧ (Nat.choose n (n / 2) : ℝ) = 70 := by
  sorry

end binomial_expansion_problem_l522_52238


namespace water_consumption_per_person_per_hour_l522_52209

theorem water_consumption_per_person_per_hour 
  (num_people : ℕ) 
  (total_hours : ℕ) 
  (total_bottles : ℕ) 
  (h1 : num_people = 4) 
  (h2 : total_hours = 16) 
  (h3 : total_bottles = 32) : 
  (total_bottles : ℚ) / total_hours / num_people = 1/2 := by
  sorry

end water_consumption_per_person_per_hour_l522_52209


namespace a1_value_l522_52297

theorem a1_value (x : ℝ) (a : Fin 8 → ℝ) :
  (x - 1)^7 = a 0 + a 1 * (x + 1) + a 2 * (x + 1)^2 + a 3 * (x + 1)^3 + 
              a 4 * (x + 1)^4 + a 5 * (x + 1)^5 + a 6 * (x + 1)^6 + a 7 * (x + 1)^7 →
  a 1 = 448 := by
sorry

end a1_value_l522_52297


namespace arithmetic_operations_equal_reciprocal_2016_l522_52274

theorem arithmetic_operations_equal_reciprocal_2016 :
  (1 / 8 * 1 / 9 * 1 / 28 : ℚ) = 1 / 2016 ∧ 
  ((1 / 8 - 1 / 9) * 1 / 28 : ℚ) = 1 / 2016 := by
  sorry

end arithmetic_operations_equal_reciprocal_2016_l522_52274


namespace apples_added_to_pile_l522_52254

/-- Given an initial pile of apples and a final pile of apples,
    calculate the number of apples added. -/
def applesAdded (initial final : ℕ) : ℕ := final - initial

/-- Theorem stating that 5 apples were added to the pile -/
theorem apples_added_to_pile :
  let initial := 8
  let final := 13
  applesAdded initial final = 5 := by sorry

end apples_added_to_pile_l522_52254


namespace profit_calculation_l522_52200

/-- Profit calculation for a product with variable price reduction --/
theorem profit_calculation 
  (price_tag : ℕ) 
  (discount : ℚ) 
  (initial_profit : ℕ) 
  (initial_sales : ℕ) 
  (sales_increase : ℕ) 
  (x : ℕ) 
  (h1 : price_tag = 80)
  (h2 : discount = 1/5)
  (h3 : initial_profit = 24)
  (h4 : initial_sales = 220)
  (h5 : sales_increase = 20) :
  ∃ y : ℤ, y = (24 - x) * (initial_sales + sales_increase * x) :=
by sorry

end profit_calculation_l522_52200


namespace min_value_of_y_l522_52271

theorem min_value_of_y (x : ℝ) (h : x > 3) : x + 1 / (x - 3) ≥ 5 ∧ ∃ x₀ > 3, x₀ + 1 / (x₀ - 3) = 5 := by
  sorry

end min_value_of_y_l522_52271


namespace marcia_pants_count_l522_52292

/-- Represents the number of items in Marcia's wardrobe -/
structure Wardrobe where
  skirts : Nat
  blouses : Nat
  pants : Nat

/-- Represents the prices of items and the total budget -/
structure Prices where
  skirt_price : ℕ
  blouse_price : ℕ
  pant_price : ℕ
  total_budget : ℕ

/-- Calculates the cost of pants with the sale applied -/
def pants_cost (n : ℕ) (price : ℕ) : ℕ :=
  if n % 2 = 0 then
    n / 2 * price + n / 2 * (price / 2)
  else
    (n / 2 + 1) * price + (n / 2) * (price / 2)

/-- Theorem stating that Marcia needs to add 2 pairs of pants -/
theorem marcia_pants_count (w : Wardrobe) (p : Prices) : w.pants = 2 :=
  by
    have h1 : w.skirts = 3 := by sorry
    have h2 : w.blouses = 5 := by sorry
    have h3 : p.skirt_price = 20 := by sorry
    have h4 : p.blouse_price = 15 := by sorry
    have h5 : p.pant_price = 30 := by sorry
    have h6 : p.total_budget = 180 := by sorry
    
    have skirt_cost : ℕ := w.skirts * p.skirt_price
    have blouse_cost : ℕ := w.blouses * p.blouse_price
    have remaining_budget : ℕ := p.total_budget - (skirt_cost + blouse_cost)
    
    have pants_fit_budget : pants_cost w.pants p.pant_price = remaining_budget := by sorry
    
    sorry -- Complete the proof here

end marcia_pants_count_l522_52292


namespace sheet_width_correct_l522_52245

/-- The width of a rectangular metallic sheet -/
def sheet_width : ℝ := 36

/-- The length of the rectangular metallic sheet -/
def sheet_length : ℝ := 48

/-- The side length of the square cut from each corner -/
def cut_square_side : ℝ := 8

/-- The volume of the resulting open box -/
def box_volume : ℝ := 5120

/-- Theorem stating that the given dimensions result in the correct volume -/
theorem sheet_width_correct : 
  (sheet_length - 2 * cut_square_side) * (sheet_width - 2 * cut_square_side) * cut_square_side = box_volume :=
by sorry

end sheet_width_correct_l522_52245


namespace sticker_problem_l522_52265

theorem sticker_problem (bob tom dan : ℕ) 
  (h1 : dan = 2 * tom) 
  (h2 : tom = 3 * bob) 
  (h3 : dan = 72) : 
  bob = 12 := by
  sorry

end sticker_problem_l522_52265


namespace polynomial_factorization_l522_52298

theorem polynomial_factorization (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end polynomial_factorization_l522_52298


namespace factor_implication_l522_52239

theorem factor_implication (m n : ℝ) : 
  (∃ a b : ℝ, 3 * X^3 - m * X + n = a * (X - 3) * (X + 1) * X) →
  |3 * m + n| = 81 :=
by
  sorry

end factor_implication_l522_52239


namespace backpack_pencilcase_combinations_l522_52244

/-- The number of combinations formed by selecting one item from each of two sets -/
def combinations (set1 : ℕ) (set2 : ℕ) : ℕ := set1 * set2

/-- Theorem: The number of combinations formed by selecting one backpack from 2 styles
    and one pencil case from 2 styles is equal to 4 -/
theorem backpack_pencilcase_combinations :
  let backpacks : ℕ := 2
  let pencilcases : ℕ := 2
  combinations backpacks pencilcases = 4 := by
  sorry

end backpack_pencilcase_combinations_l522_52244


namespace tangent_line_and_zeros_l522_52280

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - 6*x + 1

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x - 6

-- Define the function g
def g (a : ℝ) (m : ℝ) (x : ℝ) : ℝ := f a x - m

theorem tangent_line_and_zeros (a : ℝ) :
  f' a 1 = -6 →
  (∃ b c : ℝ, ∀ x y : ℝ, 12*x + 2*y - 1 = 0 ↔ y = (f a 1) + f' a 1 * (x - 1)) ∧
  (∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ∈ Set.Icc (-2) 4 ∧ x₂ ∈ Set.Icc (-2) 4 ∧ x₃ ∈ Set.Icc (-2) 4 ∧
    x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    g a m x₁ = 0 ∧ g a m x₂ = 0 ∧ g a m x₃ = 0) →
    m ∈ Set.Icc (-1) (9/2)) :=
by sorry

end tangent_line_and_zeros_l522_52280


namespace parabola_vertex_l522_52253

-- Define the parabola
def f (x : ℝ) : ℝ := -3 * (x + 1)^2 + 1

-- State the theorem
theorem parabola_vertex : 
  ∃ (x y : ℝ), (∀ t : ℝ, f t ≤ f x) ∧ y = f x ∧ x = -1 ∧ y = 1 :=
sorry

end parabola_vertex_l522_52253


namespace triangle_sine_comparison_l522_52252

/-- For a triangle ABC, compare the sum of reciprocals of sines of doubled angles
    with the sum of reciprocals of sines of angles. -/
theorem triangle_sine_comparison (A B C : ℝ) (h_triangle : A + B + C = π) :
  (A > π / 2 ∨ B > π / 2 ∨ C > π / 2) →
    1 / Real.sin (2 * A) + 1 / Real.sin (2 * B) + 1 / Real.sin (2 * C) <
    1 / Real.sin A + 1 / Real.sin B + 1 / Real.sin C ∧
  (A ≤ π / 2 ∧ B ≤ π / 2 ∧ C ≤ π / 2) →
    1 / Real.sin (2 * A) + 1 / Real.sin (2 * B) + 1 / Real.sin (2 * C) ≥
    1 / Real.sin A + 1 / Real.sin B + 1 / Real.sin C :=
by sorry

end triangle_sine_comparison_l522_52252


namespace set_operations_l522_52236

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2 * x - 4 ≥ 0}

-- Define the theorem
theorem set_operations :
  (Set.univ \ (A ∩ B) = {x | x < 2 ∨ x ≥ 3}) ∧
  ((Set.univ \ A) ∩ (Set.univ \ B) = {x | x < -1}) := by sorry

end set_operations_l522_52236


namespace thirty_percent_passed_l522_52211

/-- The swim club scenario -/
structure SwimClub where
  total_members : ℕ
  not_passed_with_course : ℕ
  not_passed_without_course : ℕ

/-- Calculate the percentage of members who passed the lifesaving test -/
def passed_percentage (club : SwimClub) : ℚ :=
  1 - (club.not_passed_with_course + club.not_passed_without_course : ℚ) / club.total_members

/-- The theorem stating that 30% of members passed the test -/
theorem thirty_percent_passed (club : SwimClub) 
  (h1 : club.total_members = 50)
  (h2 : club.not_passed_with_course = 5)
  (h3 : club.not_passed_without_course = 30) : 
  passed_percentage club = 30 / 100 := by
  sorry

#eval passed_percentage ⟨50, 5, 30⟩

end thirty_percent_passed_l522_52211


namespace kids_joined_in_l522_52208

theorem kids_joined_in (initial_kids final_kids : ℕ) (h : initial_kids = 14 ∧ final_kids = 36) :
  final_kids - initial_kids = 22 := by
  sorry

end kids_joined_in_l522_52208


namespace boat_speed_in_still_water_l522_52289

/-- Proves that the speed of a boat in still water is 12 mph given certain conditions -/
theorem boat_speed_in_still_water 
  (distance : ℝ) 
  (downstream_time : ℝ) 
  (downstream_speed : ℝ) 
  (h1 : distance = 45) 
  (h2 : downstream_time = 3)
  (h3 : downstream_speed = distance / downstream_time)
  (h4 : ∃ (current_speed : ℝ), downstream_speed = 12 + current_speed) :
  12 = (12 : ℝ) := by sorry

end boat_speed_in_still_water_l522_52289


namespace range_of_m_range_of_x_l522_52250

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x + 1) * (x - 5) ≤ 0
def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m ∧ m > 0

-- Part 1: Range of m when p is a sufficient condition for q
theorem range_of_m : 
  (∀ x, p x → ∀ m, q x m) → 
  ∀ m, m ≥ 4 := 
sorry

-- Part 2: Range of x when m=5, "p ∨ q" is true, and "p ∧ q" is false
theorem range_of_x : 
  ∀ x, (p x ∨ q x 5) ∧ ¬(p x ∧ q x 5) → 
  (x ∈ Set.Icc (-4) (-1) ∪ Set.Ioc 5 6) := 
sorry

end range_of_m_range_of_x_l522_52250


namespace jake_money_left_jake_final_amount_l522_52216

theorem jake_money_left (initial_amount : ℝ) (motorcycle_percent : ℝ) 
  (concert_percent : ℝ) (investment_percent : ℝ) (investment_loss_percent : ℝ) : ℝ :=
  let after_motorcycle := initial_amount * (1 - motorcycle_percent)
  let after_concert := after_motorcycle * (1 - concert_percent)
  let investment := after_concert * investment_percent
  let investment_loss := investment * investment_loss_percent
  let final_amount := after_concert - investment + (investment - investment_loss)
  final_amount

theorem jake_final_amount : 
  jake_money_left 5000 0.35 0.25 0.40 0.20 = 1462.50 := by
  sorry

end jake_money_left_jake_final_amount_l522_52216


namespace probability_of_cooking_sum_of_probabilities_l522_52225

/-- Represents the set of available courses. -/
inductive Course
| Planting
| Cooking
| Pottery
| Carpentry

/-- The probability of selecting a specific course from the available courses. -/
def probability_of_course (course : Course) : ℚ :=
  1 / 4

/-- Theorem stating that the probability of selecting "cooking" is 1/4. -/
theorem probability_of_cooking :
  probability_of_course Course.Cooking = 1 / 4 := by
  sorry

/-- Theorem stating that the sum of probabilities for all courses is 1. -/
theorem sum_of_probabilities :
  (probability_of_course Course.Planting) +
  (probability_of_course Course.Cooking) +
  (probability_of_course Course.Pottery) +
  (probability_of_course Course.Carpentry) = 1 := by
  sorry

end probability_of_cooking_sum_of_probabilities_l522_52225


namespace simplify_fraction_l522_52249

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) :
  (x^2 + x) / (x^2 - 2*x + 1) / ((x + 1) / (x - 1)) = x / (x - 1) := by
  sorry

end simplify_fraction_l522_52249


namespace problem_1_l522_52201

theorem problem_1 : (Real.sqrt 5 + Real.sqrt 3) * (Real.sqrt 5 - Real.sqrt 3) + (Real.sqrt 3 - 2)^2 = 9 - 4 * Real.sqrt 3 := by
  sorry

end problem_1_l522_52201


namespace amanda_camila_hike_ratio_l522_52233

/-- Proves that the ratio of Amanda's hikes to Camila's hikes is 8:1 --/
theorem amanda_camila_hike_ratio :
  let camila_hikes : ℕ := 7
  let steven_hikes : ℕ := camila_hikes + 4 * 16
  let amanda_hikes : ℕ := steven_hikes - 15
  amanda_hikes / camila_hikes = 8 := by
sorry

end amanda_camila_hike_ratio_l522_52233


namespace quadratic_rewrite_l522_52235

theorem quadratic_rewrite (b : ℝ) : 
  (∃ n : ℝ, ∀ x : ℝ, x^2 + b*x + 72 = (x + n)^2 + 20) → 
  (b = 4 * Real.sqrt 13 ∨ b = -4 * Real.sqrt 13) := by
sorry

end quadratic_rewrite_l522_52235


namespace candidate_vote_percentage_l522_52268

-- Define the total number of votes
def total_votes : ℕ := 7600

-- Define the difference in votes between the winner and loser
def vote_difference : ℕ := 2280

-- Define the percentage of votes received by the losing candidate
def losing_candidate_percentage : ℚ := 35

-- Theorem statement
theorem candidate_vote_percentage :
  (2 * losing_candidate_percentage * total_votes : ℚ) = 
  (100 * (total_votes - vote_difference) : ℚ) := by sorry

end candidate_vote_percentage_l522_52268


namespace olympic_medal_scenario_l522_52246

/-- The number of ways to award medals in the Olympic 100-meter sprint -/
def olympic_medal_ways (total_athletes : ℕ) (european_athletes : ℕ) (asian_athletes : ℕ) (max_european_medals : ℕ) : ℕ :=
  -- Define the function here
  sorry

/-- Theorem: The number of ways to award medals in the given Olympic scenario is 588 -/
theorem olympic_medal_scenario : olympic_medal_ways 10 4 6 2 = 588 := by
  sorry

end olympic_medal_scenario_l522_52246


namespace base_conversion_equality_l522_52260

theorem base_conversion_equality (b : ℝ) : b > 0 → (4 * 5 + 3 = b^2 + 2) → b = Real.sqrt 21 := by
  sorry

end base_conversion_equality_l522_52260


namespace tangent_point_coordinates_l522_52269

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 - 10*x + 3

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 3*x^2 - 10

-- Theorem statement
theorem tangent_point_coordinates :
  ∀ (x y : ℝ),
    x < 0 →
    y = curve x →
    curve_derivative x = 2 →
    (x = -2 ∧ y = 15) :=
by sorry

end tangent_point_coordinates_l522_52269


namespace intersecting_lines_regions_l522_52231

/-- The number of regions created by n intersecting lines in a plane -/
def total_regions (n : ℕ) : ℚ :=
  (n^2 + n + 2) / 2

/-- The number of bounded regions (polygons) created by n intersecting lines in a plane -/
def bounded_regions (n : ℕ) : ℚ :=
  (n^2 - 3*n + 2) / 2

/-- Theorem stating the number of regions and bounded regions created by n intersecting lines -/
theorem intersecting_lines_regions (n : ℕ) :
  (total_regions n = (n^2 + n + 2) / 2) ∧
  (bounded_regions n = (n^2 - 3*n + 2) / 2) := by
  sorry

end intersecting_lines_regions_l522_52231


namespace cos_2alpha_minus_3pi_over_5_l522_52284

theorem cos_2alpha_minus_3pi_over_5 (α : Real) 
  (h : Real.sin (α + π/5) = Real.sqrt 7 / 3) : 
  Real.cos (2*α - 3*π/5) = 5/9 := by
sorry

end cos_2alpha_minus_3pi_over_5_l522_52284


namespace kim_average_increase_l522_52299

def kim_scores : List ℝ := [92, 85, 90, 95]

theorem kim_average_increase :
  let initial_avg := (kim_scores.take 3).sum / 3
  let new_avg := kim_scores.sum / 4
  new_avg - initial_avg = 1.5 := by sorry

end kim_average_increase_l522_52299


namespace journey_distance_is_25_l522_52212

/-- Represents a segment of the journey with speed and duration -/
structure Segment where
  speed : ℝ
  duration : ℝ

/-- Calculates the distance covered in a segment -/
def distance_covered (s : Segment) : ℝ := s.speed * s.duration

/-- The journey segments -/
def journey_segments : List Segment := [
  ⟨4, 1⟩,
  ⟨5, 0.5⟩,
  ⟨3, 0.75⟩,
  ⟨2, 0.5⟩,
  ⟨6, 0.5⟩,
  ⟨7, 0.25⟩,
  ⟨4, 1.5⟩,
  ⟨6, 0.75⟩
]

/-- The total distance covered during the journey -/
def total_distance : ℝ := (journey_segments.map distance_covered).sum

theorem journey_distance_is_25 : total_distance = 25 := by sorry

end journey_distance_is_25_l522_52212


namespace complex_fraction_equality_l522_52215

theorem complex_fraction_equality (z : ℂ) (h : z = 2 + I) : 
  (2 * I) / (z - 1) = 1 + I := by
sorry

end complex_fraction_equality_l522_52215


namespace complex_exponential_sum_l522_52291

theorem complex_exponential_sum (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = (3/5 : ℂ) + (2/5 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = (3/5 : ℂ) - (2/5 : ℂ) * Complex.I :=
by sorry

end complex_exponential_sum_l522_52291


namespace largest_prime_factor_of_1023_l522_52259

theorem largest_prime_factor_of_1023 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1023 ∧ ∀ q, Nat.Prime q → q ∣ 1023 → q ≤ p :=
by sorry

end largest_prime_factor_of_1023_l522_52259


namespace square_difference_l522_52221

theorem square_difference (x : ℤ) (h : x^2 = 3136) : (x + 2) * (x - 2) = 3132 := by
  sorry

end square_difference_l522_52221


namespace factor_sum_l522_52263

theorem factor_sum (x y : ℝ) (a b c d e f g : ℤ) :
  16 * x^8 - 256 * y^4 = (a*x + b*y) * (c*x^2 + d*x*y + e*y^2) * (f*x^2 + g*y^2) →
  a + b + c + d + e + f + g = 7 := by
  sorry

end factor_sum_l522_52263


namespace min_value_product_quotient_l522_52286

theorem min_value_product_quotient (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ((x^2 + 4*x + 1) * (y^2 + 4*y + 1) * (z^2 + 4*z + 1)) / (x*y*z) ≥ 216 ∧
  ((1^2 + 4*1 + 1) * (1^2 + 4*1 + 1) * (1^2 + 4*1 + 1)) / (1*1*1) = 216 := by
  sorry

end min_value_product_quotient_l522_52286


namespace solution_to_equation_l522_52294

theorem solution_to_equation : ∃ x : ℤ, (2010 + x)^2 = x^2 ∧ x = -1005 := by
  sorry

end solution_to_equation_l522_52294


namespace max_true_statements_l522_52267

theorem max_true_statements (x : ℝ) : 
  let statements := [
    (0 < x^2 ∧ x^2 < 1),
    (x^2 > 1),
    (-1 < x ∧ x < 0),
    (0 < x ∧ x < 1),
    (0 < x - Real.sqrt x ∧ x - Real.sqrt x < 1)
  ]
  ¬∃ (s : Finset (Fin 5)), s.card > 3 ∧ (∀ i ∈ s, statements[i.val]) :=
by sorry

end max_true_statements_l522_52267


namespace solution_set_of_f_neg_x_l522_52275

-- Define the function f
def f (a b x : ℝ) : ℝ := (a * x - 1) * (x - b)

-- State the theorem
theorem solution_set_of_f_neg_x (a b : ℝ) :
  (∀ x : ℝ, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  (∀ x : ℝ, f a b (-x) < 0 ↔ x < -3 ∨ x > 1) :=
by sorry

end solution_set_of_f_neg_x_l522_52275


namespace gcd_binomial_divisibility_l522_52256

theorem gcd_binomial_divisibility (m n : ℕ) (h1 : 0 < m) (h2 : m ≤ n) : 
  ∃ k : ℤ, (Int.gcd m n : ℚ) / n * (n.choose m) = k := by
  sorry

end gcd_binomial_divisibility_l522_52256


namespace quadratic_inequality_solution_range_l522_52205

theorem quadratic_inequality_solution_range (c : ℝ) : 
  (c > 0 ∧ ∃ x : ℝ, 2 * x^2 - 8 * x + c < 0) ↔ (0 < c ∧ c < 8) := by
  sorry

end quadratic_inequality_solution_range_l522_52205


namespace base_three_20201_equals_181_l522_52247

def base_three_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3^i)) 0

theorem base_three_20201_equals_181 :
  base_three_to_ten [1, 0, 2, 0, 2] = 181 := by
  sorry

end base_three_20201_equals_181_l522_52247


namespace division_of_squares_l522_52296

theorem division_of_squares (a : ℝ) : 2 * a^2 / a^2 = 2 := by
  sorry

end division_of_squares_l522_52296
