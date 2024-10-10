import Mathlib

namespace molecular_weight_4_moles_BaI2_value_l1448_144807

/-- The molecular weight of 4 moles of Barium iodide (BaI2) -/
def molecular_weight_4_moles_BaI2 : ℝ :=
  let atomic_weight_Ba : ℝ := 137.33
  let atomic_weight_I : ℝ := 126.90
  let molecular_weight_BaI2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_I
  4 * molecular_weight_BaI2

/-- Theorem stating that the molecular weight of 4 moles of Barium iodide is 1564.52 grams -/
theorem molecular_weight_4_moles_BaI2_value : 
  molecular_weight_4_moles_BaI2 = 1564.52 := by
  sorry

end molecular_weight_4_moles_BaI2_value_l1448_144807


namespace game_show_probability_l1448_144829

/-- Represents the amount of money in each box -/
def box_values : Fin 3 → ℕ
  | 0 => 4
  | 1 => 400
  | 2 => 4000

/-- The total number of ways to assign 3 keys to 3 boxes -/
def total_assignments : ℕ := 6

/-- The number of assignments that result in winning more than $4000 -/
def winning_assignments : ℕ := 1

/-- The probability of winning more than $4000 -/
def win_probability : ℚ := winning_assignments / total_assignments

theorem game_show_probability :
  win_probability = 1 / 6 := by sorry

end game_show_probability_l1448_144829


namespace oula_deliveries_l1448_144806

/-- Proves that Oula made 96 deliveries given the problem conditions -/
theorem oula_deliveries :
  ∀ (oula_deliveries tona_deliveries : ℕ) 
    (pay_per_delivery : ℕ) 
    (pay_difference : ℕ),
  pay_per_delivery = 100 →
  tona_deliveries = 3 * oula_deliveries / 4 →
  pay_difference = 2400 →
  pay_per_delivery * oula_deliveries - pay_per_delivery * tona_deliveries = pay_difference →
  oula_deliveries = 96 := by
sorry

end oula_deliveries_l1448_144806


namespace range_of_m_l1448_144890

def p (x : ℝ) : Prop := 12 / (x + 2) ≥ 1

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x : ℝ, ¬(p x) → ¬(q x m)) →
  (∃ x : ℝ, ¬(p x) ∧ (q x m)) →
  (0 < m ∧ m < 3) :=
sorry

end range_of_m_l1448_144890


namespace intersection_M_N_l1448_144873

def M : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1, 2} := by sorry

end intersection_M_N_l1448_144873


namespace range_of_m_l1448_144877

/-- Proposition p: m + 2 < 0 -/
def p (m : ℝ) : Prop := m + 2 < 0

/-- Proposition q: the equation x^2 + mx + 1 = 0 has no real roots -/
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + 1 ≠ 0

/-- The range of real numbers for m given the conditions -/
theorem range_of_m (m : ℝ) (h1 : ¬¬p m) (h2 : ¬(p m ∧ q m)) : m < -2 := by
  sorry

end range_of_m_l1448_144877


namespace equation_solution_l1448_144827

theorem equation_solution :
  let f : ℝ → ℝ := λ x => (2*x + 1)*(3*x + 1)*(5*x + 1)*(30*x + 1)
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    f x₁ = 10 ∧ f x₂ = 10 ∧
    x₁ = (-4 + Real.sqrt 31) / 15 ∧
    x₂ = (-4 - Real.sqrt 31) / 15 ∧
    ∀ x : ℝ, f x = 10 → (x = x₁ ∨ x = x₂) :=
by sorry

end equation_solution_l1448_144827


namespace prob_three_tails_correct_l1448_144837

/-- Represents a coin with a given probability of heads -/
structure Coin where
  prob_heads : ℚ
  prob_heads_nonneg : 0 ≤ prob_heads
  prob_heads_le_one : prob_heads ≤ 1

/-- A fair coin with probability of heads = 1/2 -/
def fair_coin : Coin where
  prob_heads := 1/2
  prob_heads_nonneg := by norm_num
  prob_heads_le_one := by norm_num

/-- A biased coin with probability of heads = 2/3 -/
def biased_coin : Coin where
  prob_heads := 2/3
  prob_heads_nonneg := by norm_num
  prob_heads_le_one := by norm_num

/-- Sequence of coins: two fair coins, one biased coin, two fair coins -/
def coin_sequence : List Coin :=
  [fair_coin, fair_coin, biased_coin, fair_coin, fair_coin]

/-- Calculates the probability of getting at least 3 tails in a row -/
def prob_three_tails_in_row (coins : List Coin) : ℚ :=
  sorry

theorem prob_three_tails_correct :
  prob_three_tails_in_row coin_sequence = 13/48 := by
  sorry

end prob_three_tails_correct_l1448_144837


namespace tyler_age_l1448_144860

/-- Represents the ages of Tyler and Clay -/
structure Ages where
  tyler : ℕ
  clay : ℕ

/-- The conditions of the problem -/
def validAges (ages : Ages) : Prop :=
  ages.tyler = 3 * ages.clay + 1 ∧ ages.tyler + ages.clay = 21

/-- The theorem to prove -/
theorem tyler_age (ages : Ages) (h : validAges ages) : ages.tyler = 16 := by
  sorry

end tyler_age_l1448_144860


namespace f_min_value_l1448_144833

/-- The function f(x) = x^2 + 8x + 15 -/
def f (x : ℝ) : ℝ := x^2 + 8*x + 15

/-- Theorem: The minimum value of f(x) = x^2 + 8x + 15 is -1 -/
theorem f_min_value : ∃ (a : ℝ), f a = -1 ∧ ∀ (x : ℝ), f x ≥ -1 := by
  sorry

end f_min_value_l1448_144833


namespace logarithm_properties_l1448_144800

-- Define the theorem
theorem logarithm_properties (a b m n : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hm1 : m ≠ 1) (hn : 0 < n) : 
  (Real.log a / Real.log b) * (Real.log b / Real.log a) = 1 ∧ 
  (Real.log n / Real.log a) / (Real.log n / Real.log (m * a)) = 1 + (Real.log m / Real.log a) :=
by sorry

end logarithm_properties_l1448_144800


namespace correct_equation_transformation_l1448_144861

theorem correct_equation_transformation (x : ℝ) : 
  (x / 3 = 7) → (x = 21) :=
by sorry

end correct_equation_transformation_l1448_144861


namespace napkin_length_calculation_l1448_144891

/-- Given a tablecloth and napkins with specified dimensions, calculate the length of each napkin. -/
theorem napkin_length_calculation
  (tablecloth_length : ℕ)
  (tablecloth_width : ℕ)
  (num_napkins : ℕ)
  (napkin_width : ℕ)
  (total_material : ℕ)
  (h1 : tablecloth_length = 102)
  (h2 : tablecloth_width = 54)
  (h3 : num_napkins = 8)
  (h4 : napkin_width = 7)
  (h5 : total_material = 5844)
  (h6 : total_material = tablecloth_length * tablecloth_width + num_napkins * napkin_width * (total_material - tablecloth_length * tablecloth_width) / (napkin_width * num_napkins)) :
  (total_material - tablecloth_length * tablecloth_width) / (napkin_width * num_napkins) = 6 := by
  sorry

#check napkin_length_calculation

end napkin_length_calculation_l1448_144891


namespace find_a_and_b_l1448_144874

-- Define the system of inequalities
def inequality_system (a b x : ℝ) : Prop :=
  (3 * x - 2 < a + 1) ∧ (6 - 2 * x < b + 2)

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  -1 < x ∧ x < 2

-- Theorem statement
theorem find_a_and_b :
  ∀ a b : ℝ,
  (∀ x : ℝ, inequality_system a b x ↔ solution_set x) →
  a = 3 ∧ b = 6 := by
  sorry

end find_a_and_b_l1448_144874


namespace square_side_length_l1448_144850

/-- Given a rectangle with length 400 feet and width 300 feet, prove that a square with perimeter
    twice that of the rectangle has a side length of 700 feet. -/
theorem square_side_length (rectangle_length : ℝ) (rectangle_width : ℝ)
    (h1 : rectangle_length = 400)
    (h2 : rectangle_width = 300)
    (square_perimeter : ℝ)
    (h3 : square_perimeter = 2 * (2 * (rectangle_length + rectangle_width)))
    (square_side : ℝ)
    (h4 : square_perimeter = 4 * square_side) :
  square_side = 700 :=
by sorry

end square_side_length_l1448_144850


namespace magnitude_of_complex_power_l1448_144884

theorem magnitude_of_complex_power : 
  Complex.abs ((2 : ℂ) + (2 : ℂ) * Complex.I) ^ 8 = (4096 : ℝ) := by
  sorry

end magnitude_of_complex_power_l1448_144884


namespace totalLives_eq_110_l1448_144847

/-- The total number of lives for remaining players after some quit and bonus lives are added -/
def totalLives : ℕ :=
  let initialPlayers : ℕ := 16
  let quitPlayers : ℕ := 7
  let remainingPlayers : ℕ := initialPlayers - quitPlayers
  let playersWithTenLives : ℕ := 3
  let playersWithEightLives : ℕ := 4
  let playersWithSixLives : ℕ := 2
  let bonusLives : ℕ := 4
  
  let livesBeforeBonus : ℕ := 
    playersWithTenLives * 10 + 
    playersWithEightLives * 8 + 
    playersWithSixLives * 6
  
  let totalBonusLives : ℕ := remainingPlayers * bonusLives
  
  livesBeforeBonus + totalBonusLives

theorem totalLives_eq_110 : totalLives = 110 := by
  sorry

end totalLives_eq_110_l1448_144847


namespace total_rainfall_is_23_inches_l1448_144803

/-- Calculates the total rainfall over three days given specific conditions --/
def totalRainfall (mondayHours : ℝ) (mondayRate : ℝ) 
                  (tuesdayHours : ℝ) (tuesdayRate : ℝ)
                  (wednesdayHours : ℝ) : ℝ :=
  mondayHours * mondayRate + 
  tuesdayHours * tuesdayRate + 
  wednesdayHours * (2 * tuesdayRate)

/-- Proves that the total rainfall over the three days is 23 inches --/
theorem total_rainfall_is_23_inches : 
  totalRainfall 7 1 4 2 2 = 23 := by
  sorry


end total_rainfall_is_23_inches_l1448_144803


namespace four_variable_inequality_l1448_144899

theorem four_variable_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d)^2 ≤ 4 * (a^2 + b^2 + c^2 + d^2) := by
  sorry

end four_variable_inequality_l1448_144899


namespace purchase_problem_l1448_144867

/-- Represents the prices and quantities of small light bulbs and electric motors --/
structure PurchaseInfo where
  bulb_price : ℝ
  motor_price : ℝ
  bulb_quantity : ℕ
  motor_quantity : ℕ

/-- Calculates the total cost of a purchase --/
def total_cost (info : PurchaseInfo) : ℝ :=
  info.bulb_price * info.bulb_quantity + info.motor_price * info.motor_quantity

/-- Theorem stating the properties of the purchase problem --/
theorem purchase_problem :
  ∃ (info : PurchaseInfo),
    -- Conditions
    info.bulb_price + info.motor_price = 12 ∧
    info.bulb_price * info.bulb_quantity = 30 ∧
    info.motor_price * info.motor_quantity = 45 ∧
    info.bulb_quantity = 2 * info.motor_quantity ∧
    -- Results
    info.bulb_price = 3 ∧
    info.motor_price = 9 ∧
    -- Optimal purchase
    (∀ (alt_info : PurchaseInfo),
      alt_info.bulb_quantity + alt_info.motor_quantity = 90 ∧
      alt_info.bulb_quantity ≤ alt_info.motor_quantity / 2 →
      total_cost info ≤ total_cost alt_info) ∧
    info.bulb_quantity = 30 ∧
    info.motor_quantity = 60 ∧
    total_cost info = 630 :=
  sorry


end purchase_problem_l1448_144867


namespace historian_writing_speed_l1448_144857

/-- Given a historian who wrote 60,000 words in 150 hours,
    prove that the average number of words written per hour is 400. -/
theorem historian_writing_speed :
  let total_words : ℕ := 60000
  let total_hours : ℕ := 150
  let average_words_per_hour : ℚ := total_words / total_hours
  average_words_per_hour = 400 := by
  sorry

end historian_writing_speed_l1448_144857


namespace gcd_930_868_l1448_144888

theorem gcd_930_868 : Nat.gcd 930 868 = 62 := by
  sorry

end gcd_930_868_l1448_144888


namespace school_population_equality_l1448_144887

theorem school_population_equality (m d : ℕ) (M D : ℝ) :
  m > 0 → d > 0 →
  (M / m + D / d) / 2 = (M + D) / (m + d) →
  m = d :=
sorry

end school_population_equality_l1448_144887


namespace beef_to_steaks_l1448_144898

/-- Given 15 pounds of beef cut into 12-ounce steaks, prove that the number of steaks obtained is 20. -/
theorem beef_to_steaks :
  let pounds_of_beef : ℕ := 15
  let ounces_per_pound : ℕ := 16
  let ounces_per_steak : ℕ := 12
  let total_ounces : ℕ := pounds_of_beef * ounces_per_pound
  let number_of_steaks : ℕ := total_ounces / ounces_per_steak
  number_of_steaks = 20 :=
by sorry

end beef_to_steaks_l1448_144898


namespace min_value_expression_l1448_144823

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 3*c) / (a + 2*b + c) + 4*b / (a + b + 2*c) - 8*c / (a + b + 3*c) ≥ -17 + 12 * Real.sqrt 2 :=
sorry

end min_value_expression_l1448_144823


namespace shirt_cost_is_15_l1448_144839

/-- The cost of one pair of jeans -/
def jeans_cost : ℝ := sorry

/-- The cost of one shirt -/
def shirt_cost : ℝ := sorry

/-- The first condition: 3 pairs of jeans and 2 shirts cost $69 -/
axiom condition1 : 3 * jeans_cost + 2 * shirt_cost = 69

/-- The second condition: 2 pairs of jeans and 3 shirts cost $71 -/
axiom condition2 : 2 * jeans_cost + 3 * shirt_cost = 71

/-- Theorem: The cost of one shirt is $15 -/
theorem shirt_cost_is_15 : shirt_cost = 15 := by sorry

end shirt_cost_is_15_l1448_144839


namespace lego_sale_quadruple_pieces_l1448_144801

/-- Represents the number of Lego pieces sold for each type -/
structure LegoSale where
  single : ℕ
  double : ℕ
  triple : ℕ
  quadruple : ℕ

/-- Calculates the total number of circles from a LegoSale -/
def totalCircles (sale : LegoSale) : ℕ :=
  sale.single + 2 * sale.double + 3 * sale.triple + 4 * sale.quadruple

/-- The main theorem to prove -/
theorem lego_sale_quadruple_pieces (sale : LegoSale) :
  sale.single = 100 →
  sale.double = 45 →
  sale.triple = 50 →
  totalCircles sale = 1000 →
  sale.quadruple = 165 := by
  sorry

#check lego_sale_quadruple_pieces

end lego_sale_quadruple_pieces_l1448_144801


namespace dandelion_puff_distribution_l1448_144865

theorem dandelion_puff_distribution (total : ℕ) (given_away : ℕ) (friends : ℕ) 
  (h1 : total = 100) 
  (h2 : given_away = 42) 
  (h3 : friends = 7) :
  (total - given_away) / friends = 8 ∧ 
  (8 : ℚ) / (total - given_away) = 4 / 29 := by
  sorry

end dandelion_puff_distribution_l1448_144865


namespace midpoint_implies_xy_24_l1448_144872

-- Define the points
def A : ℝ × ℝ := (2, 10)
def C : ℝ × ℝ := (4, 7)

-- Define B as a function of x and y
def B (x y : ℝ) : ℝ × ℝ := (x, y)

-- Define the midpoint condition
def is_midpoint (m a b : ℝ × ℝ) : Prop :=
  m.1 = (a.1 + b.1) / 2 ∧ m.2 = (a.2 + b.2) / 2

-- Theorem statement
theorem midpoint_implies_xy_24 (x y : ℝ) :
  is_midpoint C A (B x y) → x * y = 24 := by
  sorry

end midpoint_implies_xy_24_l1448_144872


namespace second_group_size_l1448_144864

theorem second_group_size (total : ℕ) (group1 group3 group4 : ℕ) 
  (h1 : total = 24)
  (h2 : group1 = 5)
  (h3 : group3 = 7)
  (h4 : group4 = 4) :
  total - (group1 + group3 + group4) = 8 := by
sorry

end second_group_size_l1448_144864


namespace fraction_zero_implies_x_negative_one_l1448_144883

theorem fraction_zero_implies_x_negative_one (x : ℝ) :
  (abs x - 1) / (x - 1) = 0 → x = -1 := by
  sorry

end fraction_zero_implies_x_negative_one_l1448_144883


namespace greatest_power_of_two_congruence_l1448_144851

theorem greatest_power_of_two_congruence (m : ℕ) : 
  (∀ n : ℤ, Odd n → (n^2 * (1 + n^2 - n^4)) ≡ 1 [ZMOD 2^m]) ↔ m ≤ 7 :=
sorry

end greatest_power_of_two_congruence_l1448_144851


namespace decimal_to_fraction_l1448_144802

theorem decimal_to_fraction (x : ℚ) (h : x = 368/100) : x = 92/25 := by
  sorry

end decimal_to_fraction_l1448_144802


namespace josie_cart_wait_time_l1448_144871

/-- Represents the shopping trip details -/
structure ShoppingTrip where
  total_time : ℕ
  shopping_time : ℕ
  wait_cabinet : ℕ
  wait_restock : ℕ
  wait_checkout : ℕ

/-- Calculates the time waited for a cart given a shopping trip -/
def time_waited_for_cart (trip : ShoppingTrip) : ℕ :=
  trip.total_time - trip.shopping_time - (trip.wait_cabinet + trip.wait_restock + trip.wait_checkout)

/-- Theorem stating that Josie waited 3 minutes for a cart -/
theorem josie_cart_wait_time :
  ∃ (trip : ShoppingTrip),
    trip.total_time = 90 ∧
    trip.shopping_time = 42 ∧
    trip.wait_cabinet = 13 ∧
    trip.wait_restock = 14 ∧
    trip.wait_checkout = 18 ∧
    time_waited_for_cart trip = 3 := by
  sorry

end josie_cart_wait_time_l1448_144871


namespace fraction_equality_l1448_144856

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 15)
  (h2 : p / n = 3)
  (h3 : p / q = 1 / 10) :
  m / q = 1 / 2 := by
  sorry

end fraction_equality_l1448_144856


namespace least_integer_with_1323_divisors_l1448_144859

/-- The number of distinct positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- Checks if n can be expressed as m * 30^k where 30 is not a divisor of m -/
def is_valid_form (n m k : ℕ) : Prop :=
  n = m * (30 ^ k) ∧ ¬(30 ∣ m)

theorem least_integer_with_1323_divisors :
  ∃ (n m k : ℕ),
    (∀ i < n, num_divisors i ≠ 1323) ∧
    num_divisors n = 1323 ∧
    is_valid_form n m k ∧
    m + k = 83 :=
sorry

end least_integer_with_1323_divisors_l1448_144859


namespace imaginary_part_of_complex_fraction_l1448_144810

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := 1 / (3 + 4 * I)
  Complex.im z = -4 / 25 := by
  sorry

end imaginary_part_of_complex_fraction_l1448_144810


namespace relatively_prime_squares_l1448_144813

theorem relatively_prime_squares (a b c : ℤ) 
  (h_coprime : ∀ d : ℤ, d ∣ a ∧ d ∣ b ∧ d ∣ c → d = 1 ∨ d = -1)
  (h_eq : 1 / a + 1 / b = 1 / c) :
  ∃ (p q r : ℤ), (a + b = p^2) ∧ (a - c = q^2) ∧ (b - c = r^2) := by
  sorry

end relatively_prime_squares_l1448_144813


namespace hemisphere_surface_area_l1448_144826

theorem hemisphere_surface_area (r : ℝ) (h : r > 0) : 
  π * r^2 = 3 → 2 * π * r^2 + π * r^2 = 9 := by
  sorry

end hemisphere_surface_area_l1448_144826


namespace total_problems_solved_l1448_144808

def initial_problems : ℕ := 12
def additional_problems : ℕ := 7

theorem total_problems_solved :
  initial_problems + additional_problems = 19 := by
  sorry

end total_problems_solved_l1448_144808


namespace amanda_keeps_22_candy_bars_l1448_144825

/-- The number of candy bars Amanda keeps for herself given the initial amount, 
    the amount given to her sister initially, the amount bought later, 
    and the multiplier for the second giving. -/
def amanda_candy_bars (initial : ℕ) (first_given : ℕ) (bought : ℕ) (multiplier : ℕ) : ℕ :=
  initial - first_given + bought - (multiplier * first_given)

/-- Theorem stating that Amanda keeps 22 candy bars for herself 
    given the specific conditions in the problem. -/
theorem amanda_keeps_22_candy_bars : 
  amanda_candy_bars 7 3 30 4 = 22 := by sorry

end amanda_keeps_22_candy_bars_l1448_144825


namespace quadratic_root_and_m_l1448_144875

/-- Given a quadratic equation x^2 + 2x + m = 0 where 2 is a root,
    prove that the other root is -4 and m = -8 -/
theorem quadratic_root_and_m (m : ℝ) : 
  (2 : ℝ)^2 + 2*2 + m = 0 → 
  (∃ (other_root : ℝ), other_root = -4 ∧ 
   other_root^2 + 2*other_root + m = 0 ∧ 
   m = -8) :=
by sorry

end quadratic_root_and_m_l1448_144875


namespace third_difference_of_cubic_is_six_l1448_144896

/-- Finite difference operator -/
def finiteDifference (f : ℕ → ℝ) : ℕ → ℝ := fun n ↦ f (n + 1) - f n

/-- Third finite difference -/
def thirdFiniteDifference (f : ℕ → ℝ) : ℕ → ℝ :=
  finiteDifference (finiteDifference (finiteDifference f))

/-- Cubic function -/
def cubicFunction : ℕ → ℝ := fun n ↦ (n : ℝ) ^ 3

theorem third_difference_of_cubic_is_six :
  ∀ n, thirdFiniteDifference cubicFunction n = 6 := by sorry

end third_difference_of_cubic_is_six_l1448_144896


namespace vector_inequality_l1448_144854

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
variable (hd : FiniteDimensional.finrank ℝ V = 2)

/-- Given four vectors a, b, c, d in a 2D real vector space such that their sum is zero,
    prove that the sum of their norms is greater than or equal to the sum of the norms
    of their pairwise sums with d. -/
theorem vector_inequality (a b c d : V) (h : a + b + c + d = 0) :
  ‖a‖ + ‖b‖ + ‖c‖ + ‖d‖ ≥ ‖a + d‖ + ‖b + d‖ + ‖c + d‖ :=
sorry

end vector_inequality_l1448_144854


namespace least_subtraction_for_divisibility_l1448_144843

def original_number : ℕ := 42398
def divisor : ℕ := 15
def number_to_subtract : ℕ := 8

theorem least_subtraction_for_divisibility :
  (∀ k : ℕ, k < number_to_subtract → ¬(divisor ∣ (original_number - k))) ∧
  (divisor ∣ (original_number - number_to_subtract)) := by
  sorry

end least_subtraction_for_divisibility_l1448_144843


namespace class_presentation_periods_l1448_144821

/-- The number of periods required for all student presentations in a class --/
def periods_required (total_students : ℕ) (period_length : ℕ) (individual_presentation_length : ℕ) 
  (group_presentation_length : ℕ) (group_presentations : ℕ) : ℕ :=
  let individual_students := total_students - group_presentations
  let total_minutes := individual_students * individual_presentation_length + 
                       group_presentations * group_presentation_length
  (total_minutes + period_length - 1) / period_length

theorem class_presentation_periods :
  periods_required 32 40 8 12 4 = 7 := by
  sorry

end class_presentation_periods_l1448_144821


namespace monthly_income_A_l1448_144870

/-- Given the average monthly incomes of pairs of individuals, prove the monthly income of A. -/
theorem monthly_income_A (income_AB income_BC income_AC : ℚ) 
  (h1 : (income_A + income_B) / 2 = 5050)
  (h2 : (income_B + income_C) / 2 = 6250)
  (h3 : (income_A + income_C) / 2 = 5200)
  : income_A = 4000 := by
  sorry

where
  income_A : ℚ := sorry
  income_B : ℚ := sorry
  income_C : ℚ := sorry

end monthly_income_A_l1448_144870


namespace cos_alpha_plus_pi_sixth_l1448_144882

theorem cos_alpha_plus_pi_sixth (α : Real) (h : Real.sin (α - π/3) = 1/3) : 
  Real.cos (α + π/6) = -1/3 := by
sorry

end cos_alpha_plus_pi_sixth_l1448_144882


namespace quadratic_inequality_minimum_l1448_144838

theorem quadratic_inequality_minimum (a b c : ℝ) 
  (h1 : ∀ x, 3 < x ∧ x < 4 → a * x^2 + b * x + c > 0)
  (h2 : ∀ x, x ≤ 3 ∨ x ≥ 4 → a * x^2 + b * x + c ≤ 0) :
  ∃ m, m = (c^2 + 5) / (a + b) ∧ 
    (∀ k, k = (c^2 + 5) / (a + b) → m ≤ k) ∧
    m = 4 * Real.sqrt 5 :=
sorry

end quadratic_inequality_minimum_l1448_144838


namespace muffin_sale_total_l1448_144878

theorem muffin_sale_total (boys : ℕ) (girls : ℕ) (boys_muffins : ℕ) (girls_muffins : ℕ) : 
  boys = 3 → 
  girls = 2 → 
  boys_muffins = 12 → 
  girls_muffins = 20 → 
  boys * boys_muffins + girls * girls_muffins = 76 := by
sorry

end muffin_sale_total_l1448_144878


namespace binomial_fraction_zero_l1448_144840

theorem binomial_fraction_zero : (Nat.choose 2 5 * 3^5) / Nat.choose 10 5 = 0 := by
  sorry

end binomial_fraction_zero_l1448_144840


namespace solve_equation_l1448_144881

theorem solve_equation : ∃ x : ℝ, (10 - x = 15) ∧ (x = -5) := by sorry

end solve_equation_l1448_144881


namespace small_triangle_perimeter_l1448_144815

/-- Represents a triangle divided into smaller triangles -/
structure DividedTriangle where
  large_perimeter : ℝ
  num_small_triangles : ℕ
  small_perimeter : ℝ

/-- The property that the sum of 6 small triangle perimeters minus 3 small triangle perimeters
    equals the large triangle perimeter -/
def perimeter_property (dt : DividedTriangle) : Prop :=
  6 * dt.small_perimeter - 3 * dt.small_perimeter = dt.large_perimeter

/-- Theorem stating that for a triangle with perimeter 120 divided into 9 equal smaller triangles,
    each small triangle has a perimeter of 40 -/
theorem small_triangle_perimeter
  (dt : DividedTriangle)
  (h1 : dt.large_perimeter = 120)
  (h2 : dt.num_small_triangles = 9)
  (h3 : perimeter_property dt) :
  dt.small_perimeter = 40 := by
  sorry

end small_triangle_perimeter_l1448_144815


namespace parabola_tangent_to_line_l1448_144897

/-- A parabola y = ax^2 + bx - 4 is tangent to the line y = 2x + 3 if and only if
    a = -(b-2)^2 / 28 and b ≠ 2 -/
theorem parabola_tangent_to_line (a b : ℝ) :
  (∃ x y : ℝ, y = a * x^2 + b * x - 4 ∧ y = 2 * x + 3 ∧
    ∀ x' : ℝ, x' ≠ x → a * x'^2 + b * x' - 4 ≠ 2 * x' + 3) ↔
  (a = -(b-2)^2 / 28 ∧ b ≠ 2) :=
sorry

end parabola_tangent_to_line_l1448_144897


namespace gate_width_scientific_notation_l1448_144819

theorem gate_width_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.000000014 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.4 ∧ n = -8 := by
  sorry

end gate_width_scientific_notation_l1448_144819


namespace expression_simplification_l1448_144809

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  ((2 * x + 1) / x - 1) / ((x^2 - 1) / x) = Real.sqrt 3 / 3 := by
  sorry

end expression_simplification_l1448_144809


namespace multiplication_table_odd_fraction_l1448_144831

theorem multiplication_table_odd_fraction :
  let table_size : ℕ := 16
  let total_products : ℕ := table_size * table_size
  let odd_numbers : ℕ := (table_size + 1) / 2
  let odd_products : ℕ := odd_numbers * odd_numbers
  (odd_products : ℚ) / total_products = 1 / 4 := by
sorry

end multiplication_table_odd_fraction_l1448_144831


namespace a_2017_equals_16_l1448_144846

def sequence_with_property_P (a : ℕ → ℕ) : Prop :=
  ∀ p q : ℕ, a p = a q → a (p + 1) = a (q + 1)

theorem a_2017_equals_16 (a : ℕ → ℕ) 
  (h_prop : sequence_with_property_P a)
  (h1 : a 1 = 1)
  (h2 : a 2 = 2)
  (h3 : a 3 = 3)
  (h5 : a 5 = 2)
  (h678 : a 6 + a 7 + a 8 = 21) :
  a 2017 = 16 := by
  sorry

end a_2017_equals_16_l1448_144846


namespace street_paths_l1448_144868

theorem street_paths (P Q : ℕ) (h1 : P = 130) (h2 : Q = 65) : P - 2*Q + 2014 = 2014 := by
  sorry

end street_paths_l1448_144868


namespace sum_of_ages_l1448_144892

/-- Represents the ages of Alex, Chris, and Bella -/
structure Ages where
  alex : ℕ
  chris : ℕ
  bella : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.alex = ages.chris + 8 ∧
  ages.alex + 10 = 3 * (ages.chris - 6) ∧
  ages.bella = 2 * ages.chris

/-- The theorem to prove -/
theorem sum_of_ages (ages : Ages) :
  satisfiesConditions ages →
  ages.alex + ages.chris + ages.bella = 80 := by
  sorry

end sum_of_ages_l1448_144892


namespace f_minimum_at_negative_one_l1448_144879

-- Define the function f(x) = xe^x
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

-- State the theorem
theorem f_minimum_at_negative_one :
  IsLocalMin f (-1 : ℝ) := by sorry

end f_minimum_at_negative_one_l1448_144879


namespace exam_correct_answers_l1448_144848

/-- Proves the number of correct answers in an exam with given conditions -/
theorem exam_correct_answers 
  (total_questions : ℕ) 
  (correct_score : ℤ) 
  (wrong_score : ℤ) 
  (total_score : ℤ) 
  (h1 : total_questions = 70)
  (h2 : correct_score = 3)
  (h3 : wrong_score = -1)
  (h4 : total_score = 38) :
  ∃ (correct wrong : ℕ),
    correct + wrong = total_questions ∧
    correct_score * correct + wrong_score * wrong = total_score ∧
    correct = 27 := by
  sorry

end exam_correct_answers_l1448_144848


namespace dance_team_recruitment_l1448_144834

theorem dance_team_recruitment :
  ∀ (track_team choir dance_team : ℕ),
  track_team + choir + dance_team = 100 →
  choir = 2 * track_team →
  dance_team = choir + 10 →
  dance_team = 46 := by
sorry

end dance_team_recruitment_l1448_144834


namespace longest_boat_through_bend_l1448_144862

theorem longest_boat_through_bend (a : ℝ) (h : a > 0) :
  ∃ c : ℝ, c = 2 * a * Real.sqrt 2 ∧
  ∀ l : ℝ, l > c → ¬ (∃ θ : ℝ, 
    l * Real.cos θ ≤ a ∧ l * Real.sin θ ≤ a) := by
  sorry

end longest_boat_through_bend_l1448_144862


namespace potato_price_correct_l1448_144853

/-- The price of potatoes per kilo -/
def potato_price : ℝ := 2

theorem potato_price_correct (
  initial_amount : ℝ)
  (potato_kilos : ℝ)
  (tomato_kilos : ℝ)
  (cucumber_kilos : ℝ)
  (banana_kilos : ℝ)
  (tomato_price : ℝ)
  (cucumber_price : ℝ)
  (banana_price : ℝ)
  (remaining_amount : ℝ)
  (h1 : initial_amount = 500)
  (h2 : potato_kilos = 6)
  (h3 : tomato_kilos = 9)
  (h4 : cucumber_kilos = 5)
  (h5 : banana_kilos = 3)
  (h6 : tomato_price = 3)
  (h7 : cucumber_price = 4)
  (h8 : banana_price = 5)
  (h9 : remaining_amount = 426)
  (h10 : initial_amount - (potato_kilos * potato_price + tomato_kilos * tomato_price + 
         cucumber_kilos * cucumber_price + banana_kilos * banana_price) = remaining_amount) :
  potato_price = 2 := by
  sorry

end potato_price_correct_l1448_144853


namespace james_spent_six_l1448_144814

/-- The total amount James spent on milk, bananas, and sales tax -/
def total_spent (milk_price banana_price tax_rate : ℚ) : ℚ :=
  let subtotal := milk_price + banana_price
  let tax := subtotal * tax_rate
  subtotal + tax

/-- Theorem stating that James spent $6 given the problem conditions -/
theorem james_spent_six :
  total_spent 3 2 (1/5) = 6 := by
  sorry

end james_spent_six_l1448_144814


namespace complex_fraction_simplification_l1448_144842

def f (x : ℕ) : ℚ := (x^4 + 625 : ℚ)

theorem complex_fraction_simplification :
  (f 20 * f 40 * f 60 * f 80) / (f 10 * f 30 * f 50 * f 70) = 7 := by
  sorry

end complex_fraction_simplification_l1448_144842


namespace unique_line_pair_l1448_144824

/-- Two equations represent the same line if they have the same slope and y-intercept -/
def same_line (a b : ℝ) : Prop :=
  ∃ (m c : ℝ), ∀ (x y : ℝ),
    (2 * x + a * y + 10 = 0 ↔ y = m * x + c) ∧
    (b * x - 3 * y - 15 = 0 ↔ y = m * x + c)

/-- There exists exactly one pair (a, b) such that the given equations represent the same line -/
theorem unique_line_pair : ∃! (p : ℝ × ℝ), same_line p.1 p.2 := by sorry

end unique_line_pair_l1448_144824


namespace scientific_notation_equiv_l1448_144836

theorem scientific_notation_equiv : 
  0.0000006 = 6 * 10^(-7) := by sorry

end scientific_notation_equiv_l1448_144836


namespace volunteer_selection_l1448_144852

/-- The number of ways to select 5 people out of 9 (5 male and 4 female), 
    ensuring both genders are included. -/
theorem volunteer_selection (n m f : ℕ) 
  (h1 : n = 5) -- Total number to be selected
  (h2 : m = 5) -- Number of male students
  (h3 : f = 4) -- Number of female students
  : Nat.choose (m + f) n - Nat.choose m n = 125 := by
  sorry

end volunteer_selection_l1448_144852


namespace smallest_n_repeating_decimal_l1448_144863

/-- A number is a repeating decimal with period k if it can be expressed as m/(10^k - 1) for some integer m -/
def is_repeating_decimal (x : ℚ) (k : ℕ) : Prop :=
  ∃ m : ℤ, x = m / (10^k - 1)

/-- The smallest positive integer n < 1000 such that 1/n is a repeating decimal with period 3
    and 1/(n+6) is a repeating decimal with period 2 is 27 -/
theorem smallest_n_repeating_decimal : 
  ∃ n : ℕ, n < 1000 ∧ 
           is_repeating_decimal (1 / n) 3 ∧ 
           is_repeating_decimal (1 / (n + 6)) 2 ∧
           ∀ m : ℕ, m < n → ¬(is_repeating_decimal (1 / m) 3 ∧ is_repeating_decimal (1 / (m + 6)) 2) ∧
           n = 27 :=
sorry

end smallest_n_repeating_decimal_l1448_144863


namespace probability_no_adjacent_birch_is_two_forty_fifths_l1448_144869

def total_trees : ℕ := 15
def birch_trees : ℕ := 6
def non_birch_trees : ℕ := 9

def probability_no_adjacent_birch : ℚ :=
  (Nat.choose (non_birch_trees + 1) birch_trees) / (Nat.choose total_trees birch_trees)

theorem probability_no_adjacent_birch_is_two_forty_fifths :
  probability_no_adjacent_birch = 2 / 45 := by
  sorry

end probability_no_adjacent_birch_is_two_forty_fifths_l1448_144869


namespace tangent_segments_area_l1448_144849

/-- The area of the region formed by all line segments of length 6 that are tangent to a circle with radius 4 at their midpoints -/
theorem tangent_segments_area (r : ℝ) (l : ℝ) (h_r : r = 4) (h_l : l = 6) :
  let outer_radius := Real.sqrt (r^2 + (l/2)^2)
  (π * outer_radius^2 - π * r^2) = 9 * π :=
sorry

end tangent_segments_area_l1448_144849


namespace employee_hourly_rate_l1448_144828

/-- Proves that the hourly rate for the first 40 hours is $11.25 given the conditions -/
theorem employee_hourly_rate 
  (x : ℝ) -- hourly rate for the first 40 hours
  (overtime_hours : ℝ) -- number of overtime hours
  (overtime_rate : ℝ) -- overtime hourly rate
  (gross_pay : ℝ) -- total gross pay
  (h1 : overtime_hours = 10.75)
  (h2 : overtime_rate = 16)
  (h3 : gross_pay = 622)
  (h4 : 40 * x + overtime_hours * overtime_rate = gross_pay) :
  x = 11.25 :=
by sorry

end employee_hourly_rate_l1448_144828


namespace triangle_bisector_angle_tangent_l1448_144886

/-- Represents a triangle with given side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a line that bisects both perimeter and area of a triangle -/
structure BisectingLine where
  startPoint : ℝ × ℝ
  endPoint : ℝ × ℝ

/-- The acute angle between two bisecting lines -/
def angleBetweenBisectors (l1 l2 : BisectingLine) : ℝ := sorry

/-- Checks if a line bisects both perimeter and area of a triangle -/
def isBisectingLine (t : Triangle) (l : BisectingLine) : Prop := sorry

theorem triangle_bisector_angle_tangent (t : Triangle) 
  (h1 : t.a = 13 ∧ t.b = 14 ∧ t.c = 15) : 
  ∃ (l1 l2 : BisectingLine) (θ : ℝ),
    isBisectingLine t l1 ∧ 
    isBisectingLine t l2 ∧ 
    θ = angleBetweenBisectors l1 l2 ∧ 
    0 < θ ∧ θ < π/2 ∧
    Real.tan θ = sorry -- This should be replaced with the actual value or expression
    := by sorry

end triangle_bisector_angle_tangent_l1448_144886


namespace candidate_A_votes_l1448_144830

def total_votes : ℕ := 560000
def invalid_percentage : ℚ := 15 / 100
def candidate_A_percentage : ℚ := 85 / 100

theorem candidate_A_votes : 
  ⌊(1 - invalid_percentage) * candidate_A_percentage * total_votes⌋ = 404600 := by
  sorry

end candidate_A_votes_l1448_144830


namespace main_triangle_area_l1448_144885

/-- A triangle with a point inside it -/
structure TriangleWithInnerPoint where
  /-- The triangle -/
  triangle : Set (ℝ × ℝ)
  /-- The point inside the triangle -/
  inner_point : ℝ × ℝ
  /-- The point is inside the triangle -/
  point_inside : inner_point ∈ triangle

/-- The areas of smaller triangles formed by lines parallel to the sides -/
structure SmallerTriangleAreas where
  /-- The first smaller triangle area -/
  area1 : ℝ
  /-- The second smaller triangle area -/
  area2 : ℝ
  /-- The third smaller triangle area -/
  area3 : ℝ

/-- Calculate the area of the main triangle given the areas of smaller triangles -/
def calculateMainTriangleArea (smaller_areas : SmallerTriangleAreas) : ℝ :=
  sorry

/-- The theorem stating the relationship between smaller triangle areas and the main triangle area -/
theorem main_triangle_area 
  (t : TriangleWithInnerPoint) 
  (areas : SmallerTriangleAreas)
  (h1 : areas.area1 = 16)
  (h2 : areas.area2 = 25)
  (h3 : areas.area3 = 36) :
  calculateMainTriangleArea areas = 225 :=
sorry

end main_triangle_area_l1448_144885


namespace youngest_child_age_l1448_144866

def restaurant_problem (father_charge : ℝ) (child_charge_per_year : ℝ) (total_bill : ℝ) : Prop :=
  ∃ (twin_age youngest_age : ℕ),
    father_charge = 4.95 ∧
    child_charge_per_year = 0.45 ∧
    total_bill = 9.45 ∧
    twin_age > youngest_age ∧
    total_bill = father_charge + child_charge_per_year * (2 * twin_age + youngest_age) ∧
    youngest_age = 2

theorem youngest_child_age :
  restaurant_problem 4.95 0.45 9.45
  := by sorry

end youngest_child_age_l1448_144866


namespace mixed_number_division_equality_l1448_144880

theorem mixed_number_division_equality :
  (4 + 2/3 + 5 + 1/4) / (3 + 1/2 - (2 + 3/5)) = 11 + 1/54 := by sorry

end mixed_number_division_equality_l1448_144880


namespace smallest_gcd_bc_l1448_144822

theorem smallest_gcd_bc (a b c : ℕ+) (h1 : Nat.gcd a b = 255) (h2 : Nat.gcd a c = 855) :
  ∃ (b' c' : ℕ+), Nat.gcd a b' = 255 ∧ Nat.gcd a c' = 855 ∧ 
    Nat.gcd b' c' = 15 ∧ 
    ∀ (b'' c'' : ℕ+), Nat.gcd a b'' = 255 → Nat.gcd a c'' = 855 → 
      Nat.gcd b'' c'' ≥ 15 :=
by sorry

end smallest_gcd_bc_l1448_144822


namespace solution_set_equivalence_l1448_144832

theorem solution_set_equivalence (x : ℝ) : 
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by
sorry

end solution_set_equivalence_l1448_144832


namespace floor_a_equals_1994_minus_n_l1448_144876

def a : ℕ → ℚ
  | 0 => 1994
  | n + 1 => (a n)^2 / (a n + 1)

theorem floor_a_equals_1994_minus_n (n : ℕ) (h : n ≤ 998) :
  ⌊a n⌋ = 1994 - n :=
by sorry

end floor_a_equals_1994_minus_n_l1448_144876


namespace equivalence_condition_l1448_144845

theorem equivalence_condition (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ (a, b) ≠ (0, 0)) : 
  (1 / a < 1 / b) ↔ (a * b / (a^3 - b^3) > 0) :=
by sorry

end equivalence_condition_l1448_144845


namespace expo_arrangements_l1448_144817

/-- The number of ways to arrange n distinct objects. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k objects from n distinct objects. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of volunteers. -/
def num_volunteers : ℕ := 5

/-- The number of foreign friends. -/
def num_foreign_friends : ℕ := 2

/-- The total number of people. -/
def total_people : ℕ := num_volunteers + num_foreign_friends

/-- The number of positions where the foreign friends can be placed. -/
def foreign_friend_positions : ℕ := total_people - num_foreign_friends - 1

theorem expo_arrangements : 
  choose foreign_friend_positions 1 * arrangements num_volunteers * arrangements num_foreign_friends = 960 := by
  sorry

end expo_arrangements_l1448_144817


namespace a_2022_eq_674_l1448_144844

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 1
  | n+3 => (n+3) / (a n * a (n+1) * a (n+2))

theorem a_2022_eq_674 : a 2022 = 674 := by
  sorry

end a_2022_eq_674_l1448_144844


namespace cubic_root_sum_l1448_144894

theorem cubic_root_sum (a b c : ℝ) (p q r : ℕ+) : 
  a^3 - 3*a^2 - 7*a - 1 = 0 →
  b^3 - 3*b^2 - 7*b - 1 = 0 →
  c^3 - 3*c^2 - 7*c - 1 = 0 →
  a ≠ b →
  b ≠ c →
  c ≠ a →
  (1 / (a^(1/3) - b^(1/3)) + 1 / (b^(1/3) - c^(1/3)) + 1 / (c^(1/3) - a^(1/3)))^2 = p * q^(1/3) / r →
  Nat.gcd p.val r.val = 1 →
  ∀ (prime : ℕ), prime.Prime → ¬(∃ (k : ℕ), q = prime^3 * k) →
  100 * p + 10 * q + r = 1913 := by
  sorry

end cubic_root_sum_l1448_144894


namespace f_max_value_l1448_144818

/-- The function f(z) = -6z^2 + 24z - 12 -/
def f (z : ℝ) : ℝ := -6 * z^2 + 24 * z - 12

theorem f_max_value :
  (∀ z : ℝ, f z ≤ 12) ∧ (∃ z : ℝ, f z = 12) := by sorry

end f_max_value_l1448_144818


namespace elvins_phone_bill_l1448_144895

/-- Elvin's monthly telephone bill -/
def monthly_bill (call_charge : ℕ) (internet_charge : ℕ) : ℕ :=
  call_charge + internet_charge

theorem elvins_phone_bill 
  (internet_charge : ℕ) 
  (first_month_call_charge : ℕ) 
  (h1 : monthly_bill first_month_call_charge internet_charge = 50)
  (h2 : monthly_bill (2 * first_month_call_charge) internet_charge = 76) :
  monthly_bill (2 * first_month_call_charge) internet_charge = 76 :=
by
  sorry

#check elvins_phone_bill

end elvins_phone_bill_l1448_144895


namespace triangle_formation_l1448_144804

/-- Determines if three lengths can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The given groups of numbers --/
def group_A : (ℝ × ℝ × ℝ) := (5, 7, 12)
def group_B : (ℝ × ℝ × ℝ) := (7, 7, 15)
def group_C : (ℝ × ℝ × ℝ) := (6, 9, 16)
def group_D : (ℝ × ℝ × ℝ) := (6, 8, 12)

theorem triangle_formation :
  ¬(can_form_triangle group_A.1 group_A.2.1 group_A.2.2) ∧
  ¬(can_form_triangle group_B.1 group_B.2.1 group_B.2.2) ∧
  ¬(can_form_triangle group_C.1 group_C.2.1 group_C.2.2) ∧
  can_form_triangle group_D.1 group_D.2.1 group_D.2.2 :=
by sorry

end triangle_formation_l1448_144804


namespace cos_double_angle_special_case_l1448_144805

theorem cos_double_angle_special_case (α : Real) 
  (h : Real.sin (α + Real.pi / 2) = 1 / 2) : 
  Real.cos (2 * α) = -1 / 2 := by
  sorry

end cos_double_angle_special_case_l1448_144805


namespace sufficient_not_necessary_l1448_144841

def l1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y - 1 = 0
def l2 (a : ℝ) (x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

def parallel (a : ℝ) : Prop :=
  ∀ x1 y1 x2 y2 : ℝ, l1 a x1 y1 ∧ l2 a x2 y2 → (y1 - y2) * (a + 1) = (x1 - x2) * 2

theorem sufficient_not_necessary :
  (parallel (-2)) ∧ (∃ a : ℝ, a ≠ -2 ∧ parallel a) :=
sorry

end sufficient_not_necessary_l1448_144841


namespace park_area_l1448_144893

/-- The area of a rectangular park with perimeter 80 meters and length three times the width is 300 square meters. -/
theorem park_area (width length : ℝ) (h_perimeter : 2 * (width + length) = 80) (h_length : length = 3 * width) :
  width * length = 300 :=
sorry

end park_area_l1448_144893


namespace salad_dressing_vinegar_weight_l1448_144855

/-- Given a bowl of salad dressing with specified properties, prove the weight of vinegar. -/
theorem salad_dressing_vinegar_weight
  (bowl_capacity : ℝ)
  (oil_fraction : ℝ)
  (vinegar_fraction : ℝ)
  (oil_density : ℝ)
  (total_weight : ℝ)
  (h_bowl : bowl_capacity = 150)
  (h_oil_frac : oil_fraction = 2/3)
  (h_vinegar_frac : vinegar_fraction = 1/3)
  (h_oil_density : oil_density = 5)
  (h_total_weight : total_weight = 700)
  (h_fractions : oil_fraction + vinegar_fraction = 1) :
  (total_weight - oil_density * (oil_fraction * bowl_capacity)) / (vinegar_fraction * bowl_capacity) = 4 := by
  sorry


end salad_dressing_vinegar_weight_l1448_144855


namespace mean_problem_l1448_144835

theorem mean_problem (x : ℝ) : 
  (12 + x + 42 + 78 + 104) / 5 = 62 → 
  (128 + 255 + 511 + 1023 + x) / 5 = 398.2 := by
sorry

end mean_problem_l1448_144835


namespace tangerines_oranges_percentage_l1448_144820

/-- Represents the quantities of fruits in Tina's bag -/
structure FruitBag where
  apples : ℕ
  oranges : ℕ
  tangerines : ℕ
  grapes : ℕ
  kiwis : ℕ

/-- Calculates the total number of fruits in the bag -/
def totalFruits (bag : FruitBag) : ℕ :=
  bag.apples + bag.oranges + bag.tangerines + bag.grapes + bag.kiwis

/-- Calculates the number of tangerines and oranges in the bag -/
def tangerinesAndOranges (bag : FruitBag) : ℕ :=
  bag.tangerines + bag.oranges

/-- Theorem stating that the percentage of tangerines and oranges in the remaining fruits is 47.5% -/
theorem tangerines_oranges_percentage (initialBag : FruitBag)
    (h1 : initialBag.apples = 9)
    (h2 : initialBag.oranges = 5)
    (h3 : initialBag.tangerines = 17)
    (h4 : initialBag.grapes = 12)
    (h5 : initialBag.kiwis = 7) :
    let finalBag : FruitBag := {
      apples := initialBag.apples,
      oranges := initialBag.oranges - 2 + 3,
      tangerines := initialBag.tangerines - 10 + 6,
      grapes := initialBag.grapes - 4,
      kiwis := initialBag.kiwis - 3
    }
    (tangerinesAndOranges finalBag : ℚ) / (totalFruits finalBag : ℚ) * 100 = 47.5 := by
  sorry

end tangerines_oranges_percentage_l1448_144820


namespace new_to_original_student_ratio_l1448_144816

theorem new_to_original_student_ratio 
  (original_avg : ℝ) 
  (new_student_avg : ℝ) 
  (avg_decrease : ℝ) 
  (h1 : original_avg = 40)
  (h2 : new_student_avg = 34)
  (h3 : avg_decrease = 4)
  (h4 : original_avg = (original_avg - avg_decrease) + 6) :
  ∃ (O N : ℕ), N = 2 * O ∧ N > 0 ∧ O > 0 := by
  sorry

end new_to_original_student_ratio_l1448_144816


namespace missing_roots_theorem_l1448_144811

def p (x : ℝ) : ℝ := 12 * x^5 - 8 * x^4 - 45 * x^3 + 45 * x^2 + 8 * x - 12

theorem missing_roots_theorem (h1 : p 1 = 0) (h2 : p 1.5 = 0) (h3 : p (-2) = 0) :
  p (2/3) = 0 ∧ p (-1/2) = 0 := by
  sorry

end missing_roots_theorem_l1448_144811


namespace min_value_of_function_l1448_144889

theorem min_value_of_function (x : ℝ) (h : x > 2) : 
  (4 / (x - 2) + x) ≥ 6 := by
sorry

end min_value_of_function_l1448_144889


namespace herd_division_l1448_144812

theorem herd_division (total : ℕ) (fourth_son : ℕ) : 
  (total : ℚ) / 3 + total / 5 + total / 6 + fourth_son = total ∧ 
  fourth_son = 19 → 
  total = 63 := by
sorry

end herd_division_l1448_144812


namespace union_equal_iff_a_geq_one_l1448_144858

/-- The set A defined as {x | 2 ≤ x ≤ 6} -/
def A : Set ℝ := {x : ℝ | 2 ≤ x ∧ x ≤ 6}

/-- The set B defined as {x | 2a ≤ x ≤ a+3} -/
def B (a : ℝ) : Set ℝ := {x : ℝ | 2*a ≤ x ∧ x ≤ a+3}

/-- Theorem stating that A ∪ B = A if and only if a ≥ 1 -/
theorem union_equal_iff_a_geq_one (a : ℝ) : A ∪ B a = A ↔ a ≥ 1 := by sorry

end union_equal_iff_a_geq_one_l1448_144858
