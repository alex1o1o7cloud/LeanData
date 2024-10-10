import Mathlib

namespace original_ghee_quantity_l3029_302954

/-- Proves that the original quantity of ghee is 30 kg given the conditions of the problem. -/
theorem original_ghee_quantity (x : ℝ) : 
  (0.5 * x = 0.3 * (x + 20)) → x = 30 := by
  sorry

end original_ghee_quantity_l3029_302954


namespace AQ_length_l3029_302949

/-- Square ABCD with side length 10 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 0) ∧ B = (10, 0) ∧ C = (10, 10) ∧ D = (0, 10))

/-- Points P, Q, R, X, Y -/
structure SpecialPoints (ABCD : Square) :=
  (P : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (R : ℝ × ℝ)
  (X : ℝ × ℝ)
  (Y : ℝ × ℝ)
  (P_on_CD : P.2 = 10)
  (Q_on_AD : Q.1 = 0)
  (R_on_CD : R.2 = 10)
  (BQ_perp_AP : (Q.2 / 10) * ((10 - Q.2) / P.1) = -1)
  (RQ_parallel_PA : (Q.2 - 10) / (-P.1) = (10 - Q.2) / P.1)
  (X_on_BC_AP : X.1 = 10 ∧ X.2 = (10 - Q.2) * (X.1 / P.1) + Q.2)
  (Y_on_circumcircle : ∃ (center : ℝ × ℝ) (radius : ℝ),
    (Y.1 - center.1)^2 + (Y.2 - center.2)^2 = radius^2 ∧
    (P.1 - center.1)^2 + (P.2 - center.2)^2 = radius^2 ∧
    (Q.1 - center.1)^2 + (Q.2 - center.2)^2 = radius^2 ∧
    (0 - center.1)^2 + (10 - center.2)^2 = radius^2)
  (angle_PYR : Real.cos (105 * π / 180) = 
    ((Y.1 - P.1) * (R.1 - Y.1) + (Y.2 - P.2) * (R.2 - Y.2)) /
    (Real.sqrt ((Y.1 - P.1)^2 + (Y.2 - P.2)^2) * Real.sqrt ((R.1 - Y.1)^2 + (R.2 - Y.2)^2)))

/-- The main theorem -/
theorem AQ_length (ABCD : Square) (points : SpecialPoints ABCD) :
  Real.sqrt ((points.Q.1 - ABCD.A.1)^2 + (points.Q.2 - ABCD.A.2)^2) = 10 * Real.sqrt 3 - 10 := by
  sorry

end AQ_length_l3029_302949


namespace second_number_is_72_l3029_302910

theorem second_number_is_72 (a b c : ℚ) : 
  a + b + c = 264 ∧ 
  a = 2 * b ∧ 
  c = (1 / 3) * a → 
  b = 72 := by
  sorry

end second_number_is_72_l3029_302910


namespace concrete_amount_l3029_302998

/-- The amount of bricks ordered in tons -/
def bricks : ℝ := 0.17

/-- The amount of stone ordered in tons -/
def stone : ℝ := 0.5

/-- The total amount of material ordered in tons -/
def total_material : ℝ := 0.83

/-- The amount of concrete ordered in tons -/
def concrete : ℝ := total_material - (bricks + stone)

theorem concrete_amount : concrete = 0.16 := by
  sorry

end concrete_amount_l3029_302998


namespace ivan_commute_l3029_302934

theorem ivan_commute (T : ℝ) (D : ℝ) (h1 : T > 0) (h2 : D > 0) : 
  let v := D / T
  let new_time := T - 65
  (D / (1.6 * v) = new_time) → 
  (D / (1.3 * v) = T - 40) :=
by sorry

end ivan_commute_l3029_302934


namespace square_sum_equality_l3029_302944

theorem square_sum_equality : 12^2 + 2*(12*5) + 5^2 = 289 := by
  sorry

end square_sum_equality_l3029_302944


namespace tan_beta_value_l3029_302971

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.tan β = 1/7 := by
  sorry

end tan_beta_value_l3029_302971


namespace total_puzzle_time_l3029_302970

def puzzle_time (warm_up_time : ℕ) (additional_puzzles : ℕ) (time_factor : ℕ) : ℕ :=
  warm_up_time + additional_puzzles * (warm_up_time * time_factor)

theorem total_puzzle_time :
  puzzle_time 10 2 3 = 70 :=
by
  sorry

end total_puzzle_time_l3029_302970


namespace cd_equals_three_plus_b_l3029_302987

theorem cd_equals_three_plus_b 
  (a b c d : ℝ) 
  (h1 : a + b = 11) 
  (h2 : b + c = 9) 
  (h3 : a + d = 5) : 
  c + d = 3 + b := by
sorry

end cd_equals_three_plus_b_l3029_302987


namespace even_binomial_coefficients_iff_power_of_two_l3029_302920

def is_power_of_two (n : ℕ+) : Prop :=
  ∃ k : ℕ, n = 2^k

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem even_binomial_coefficients_iff_power_of_two (n : ℕ+) :
  (∀ k : ℕ, 1 ≤ k ∧ k < n → Even (binomial_coefficient n k)) ↔ is_power_of_two n :=
sorry

end even_binomial_coefficients_iff_power_of_two_l3029_302920


namespace last_digit_389_base4_l3029_302975

def decimal_to_base4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem last_digit_389_base4 :
  (decimal_to_base4 389).getLast? = some 1 := by
  sorry

end last_digit_389_base4_l3029_302975


namespace max_balls_in_cube_l3029_302932

theorem max_balls_in_cube (cube_volume ball_volume : ℝ) (h1 : cube_volume = 1000) (h2 : ball_volume = 36 * Real.pi) :
  ⌊cube_volume / ball_volume⌋ = 8 := by
  sorry

end max_balls_in_cube_l3029_302932


namespace no_formula_fits_all_data_l3029_302996

def data : List (ℕ × ℕ) := [(1, 2), (2, 6), (3, 12), (4, 20), (5, 30)]

def formula_a (x : ℕ) : ℕ := 4 * x - 2
def formula_b (x : ℕ) : ℕ := x^3 - x^2 + 2*x
def formula_c (x : ℕ) : ℕ := 2 * x^2
def formula_d (x : ℕ) : ℕ := x^2 + 2*x + 1

theorem no_formula_fits_all_data :
  ¬(∀ (x y : ℕ), (x, y) ∈ data → 
    (y = formula_a x ∨ y = formula_b x ∨ y = formula_c x ∨ y = formula_d x)) :=
by sorry

end no_formula_fits_all_data_l3029_302996


namespace insert_two_digits_into_five_digit_number_l3029_302988

/-- The number of ways to insert two indistinguishable digits into a 5-digit number to form a 7-digit number -/
def insert_two_digits (n : ℕ) : ℕ :=
  let total_positions := n + 1
  let total_arrangements := total_positions * total_positions
  let arrangements_together := total_positions
  total_arrangements - arrangements_together

/-- The theorem stating that inserting two indistinguishable digits into a 5-digit number results in 30 different 7-digit numbers -/
theorem insert_two_digits_into_five_digit_number :
  insert_two_digits 5 = 30 := by
  sorry

end insert_two_digits_into_five_digit_number_l3029_302988


namespace divisibility_properties_l3029_302918

theorem divisibility_properties (a m n : ℕ) (ha : a ≥ 2) (hm : m > 0) (hn : n > 0) (h_div : m ∣ n) :
  (∃ k, a^n - 1 = k * (a^m - 1)) ∧
  ((∃ k, a^n + 1 = k * (a^m + 1)) ↔ Odd (n / m)) :=
by sorry

end divisibility_properties_l3029_302918


namespace four_player_tournament_games_l3029_302904

/-- The number of games in a round-robin tournament with n players -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a tournament with 4 players, where each player plays against 
    every other player exactly once, the total number of games is 6 -/
theorem four_player_tournament_games : 
  num_games 4 = 6 := by
  sorry

end four_player_tournament_games_l3029_302904


namespace change_ways_50_cents_l3029_302938

/-- Represents the number of ways to make change for a given amount using pennies, nickels, and dimes. -/
def changeWays (amount : ℕ) : ℕ := sorry

/-- The value of a penny in cents -/
def pennyValue : ℕ := 1

/-- The value of a nickel in cents -/
def nickelValue : ℕ := 5

/-- The value of a dime in cents -/
def dimeValue : ℕ := 10

/-- The total amount we want to make change for, in cents -/
def totalAmount : ℕ := 50

theorem change_ways_50_cents :
  changeWays totalAmount = 35 := by sorry

end change_ways_50_cents_l3029_302938


namespace dino_hourly_rate_l3029_302956

/-- Dino's monthly income calculation -/
theorem dino_hourly_rate (hours1 hours2 hours3 : ℕ) (rate2 rate3 : ℚ) 
  (expenses leftover : ℚ) (total_income : ℚ) :
  hours1 = 20 →
  hours2 = 30 →
  hours3 = 5 →
  rate2 = 20 →
  rate3 = 40 →
  expenses = 500 →
  leftover = 500 →
  total_income = expenses + leftover →
  total_income = hours1 * (total_income - hours2 * rate2 - hours3 * rate3) / hours1 + hours2 * rate2 + hours3 * rate3 →
  (total_income - hours2 * rate2 - hours3 * rate3) / hours1 = 10 :=
by sorry

end dino_hourly_rate_l3029_302956


namespace intersection_implies_a_values_l3029_302953

def A : Set ℝ := {-1, 2, 3}
def B (a : ℝ) : Set ℝ := {a + 1, a^2 + 3}

theorem intersection_implies_a_values :
  ∀ a : ℝ, (A ∩ B a = {3}) → (a = 0 ∨ a = 2) :=
by sorry

end intersection_implies_a_values_l3029_302953


namespace slices_per_pizza_l3029_302966

theorem slices_per_pizza (total_pizzas : ℕ) (total_slices : ℕ) 
  (h1 : total_pizzas = 21) 
  (h2 : total_slices = 168) : 
  total_slices / total_pizzas = 8 := by
  sorry

end slices_per_pizza_l3029_302966


namespace fourth_day_temperature_l3029_302914

def temperature_problem (t1 t2 t3 avg : ℚ) : Prop :=
  let sum3 := t1 + t2 + t3
  let sum4 := 4 * avg
  sum4 - sum3 = -36

theorem fourth_day_temperature :
  temperature_problem 13 (-15) (-10) (-12) := by sorry

end fourth_day_temperature_l3029_302914


namespace min_sum_parallel_vectors_l3029_302958

theorem min_sum_parallel_vectors (x y : ℝ) : 
  x > 0 → y > 0 → 
  (∃ (k : ℝ), k ≠ 0 ∧ k • (1 - x, x) = (1, -y)) →
  (∀ a b : ℝ, a > 0 → b > 0 → (∃ (k : ℝ), k ≠ 0 ∧ k • (1 - a, a) = (1, -b)) → a + b ≥ 4) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (∃ (k : ℝ), k ≠ 0 ∧ k • (1 - a, a) = (1, -b)) ∧ a + b = 4) :=
by sorry


end min_sum_parallel_vectors_l3029_302958


namespace abc_sum_product_bounds_l3029_302931

theorem abc_sum_product_bounds (a b c : ℝ) (h : a + b + c = 1) :
  ∀ ε > 0, ∃ x : ℝ, x = a * b + a * c + b * c ∧ x ≤ 1/3 ∧ ∃ y : ℝ, y = a * b + a * c + b * c ∧ y < -ε :=
by sorry

end abc_sum_product_bounds_l3029_302931


namespace linear_function_property_l3029_302916

/-- A linear function is a function f such that f(x) = mx + b for some constants m and b. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

/-- Given a linear function g such that g(10) - g(4) = 24, prove that g(16) - g(4) = 48. -/
theorem linear_function_property (g : ℝ → ℝ) 
  (h_linear : LinearFunction g) 
  (h_condition : g 10 - g 4 = 24) : 
  g 16 - g 4 = 48 := by
  sorry

end linear_function_property_l3029_302916


namespace chord_length_when_k_2_single_intersection_point_l3029_302977

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := y^2 = 12 * x
def line (k x y : ℝ) : Prop := y = k * x - 1

-- Part 1: Chord length when k = 2
theorem chord_length_when_k_2 :
  ∀ x₁ y₁ x₂ y₂ : ℝ,
  parabola x₁ y₁ → parabola x₂ y₂ →
  line 2 x₁ y₁ → line 2 x₂ y₂ →
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 75 :=
sorry

-- Part 2: Conditions for single intersection point
theorem single_intersection_point :
  ∀ k : ℝ,
  (∃! x y : ℝ, parabola x y ∧ line k x y) ↔ (k = 0 ∨ k = -3) :=
sorry

end chord_length_when_k_2_single_intersection_point_l3029_302977


namespace powerjet_pump_l3029_302947

/-- The amount of water pumped in a given time -/
def water_pumped (rate : ℝ) (time : ℝ) : ℝ := rate * time

/-- Theorem: A pump operating at 500 gallons per hour will pump 250 gallons in 30 minutes -/
theorem powerjet_pump (rate : ℝ) (time : ℝ) (h1 : rate = 500) (h2 : time = 1/2) : 
  water_pumped rate time = 250 := by
  sorry

end powerjet_pump_l3029_302947


namespace fermat_last_digit_l3029_302940

/-- Fermat number -/
def F (n : ℕ) : ℕ := 2^(2^n) + 1

/-- The last digit of Fermat numbers for n ≥ 2 is always 7 -/
theorem fermat_last_digit (n : ℕ) (h : n ≥ 2) : F n % 10 = 7 := by
  sorry

end fermat_last_digit_l3029_302940


namespace dons_pizza_consumption_l3029_302900

/-- Don's pizza consumption problem -/
theorem dons_pizza_consumption (darias_consumption : ℝ) (total_consumption : ℝ) 
  (h1 : darias_consumption = 2.5 * (total_consumption - darias_consumption))
  (h2 : total_consumption = 280) : 
  total_consumption - darias_consumption = 80 := by
  sorry

end dons_pizza_consumption_l3029_302900


namespace equation_one_solutions_l3029_302967

theorem equation_one_solutions (x : ℝ) : x * (x - 2) = x - 2 ↔ x = 1 ∨ x = 2 := by
  sorry

end equation_one_solutions_l3029_302967


namespace total_solar_systems_and_planets_l3029_302976

/-- The number of planets in the galaxy -/
def num_planets : ℕ := 20

/-- The number of additional solar systems for each planet -/
def additional_solar_systems : ℕ := 8

/-- The total number of solar systems and planets in the galaxy -/
def total_count : ℕ := num_planets * (additional_solar_systems + 1) + num_planets

theorem total_solar_systems_and_planets :
  total_count = 200 :=
by sorry

end total_solar_systems_and_planets_l3029_302976


namespace train_length_l3029_302907

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 9 → ∃ length : ℝ, 
  (length ≥ 150 ∧ length < 151) ∧ 
  length = speed * (1000 / 3600) * time := by
  sorry

end train_length_l3029_302907


namespace min_value_expression_equality_achieved_l3029_302950

theorem min_value_expression (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((x - 2)^2 + (x + 1)^2) ≥ Real.sqrt 13 := by
  sorry

theorem equality_achieved : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((x - 2)^2 + (x + 1)^2) = Real.sqrt 13 := by
  sorry

end min_value_expression_equality_achieved_l3029_302950


namespace student_tickets_sold_l3029_302917

theorem student_tickets_sold (adult_price student_price total_tickets total_amount : ℚ)
  (h1 : adult_price = 4)
  (h2 : student_price = (5/2))
  (h3 : total_tickets = 59)
  (h4 : total_amount = (445/2))
  (h5 : ∃ (adult_tickets student_tickets : ℚ),
    adult_tickets + student_tickets = total_tickets ∧
    adult_price * adult_tickets + student_price * student_tickets = total_amount) :
  ∃ (student_tickets : ℚ), student_tickets = 9 := by
sorry

end student_tickets_sold_l3029_302917


namespace green_curlers_count_l3029_302951

def total_curlers : ℕ := 16

def pink_curlers : ℕ := total_curlers / 4

def blue_curlers : ℕ := 2 * pink_curlers

def green_curlers : ℕ := total_curlers - (pink_curlers + blue_curlers)

theorem green_curlers_count : green_curlers = 4 := by
  sorry

end green_curlers_count_l3029_302951


namespace middle_school_count_l3029_302959

structure School where
  total_students : ℕ
  sample_size : ℕ
  middle_school_in_sample : ℕ

def middle_school_students (s : School) : ℕ :=
  s.total_students * s.middle_school_in_sample / s.sample_size

theorem middle_school_count (s : School) 
  (h1 : s.total_students = 2000)
  (h2 : s.sample_size = 400)
  (h3 : s.middle_school_in_sample = 180) :
  middle_school_students s = 900 := by
  sorry

end middle_school_count_l3029_302959


namespace square_area_ratio_when_doubled_l3029_302989

theorem square_area_ratio_when_doubled (s : ℝ) (h : s > 0) :
  (s^2) / ((2*s)^2) = 1/4 := by
  sorry

end square_area_ratio_when_doubled_l3029_302989


namespace pony_price_calculation_l3029_302984

/-- The regular price of Fox jeans in dollars -/
def fox_price : ℝ := 15

/-- The discount rate for Pony jeans as a decimal -/
def pony_discount : ℝ := 0.10999999999999996

/-- The sum of discount rates for Fox and Pony jeans as a decimal -/
def total_discount : ℝ := 0.22

/-- The number of Fox jeans purchased -/
def fox_quantity : ℕ := 3

/-- The number of Pony jeans purchased -/
def pony_quantity : ℕ := 2

/-- The total savings from the purchase in dollars -/
def total_savings : ℝ := 8.91

/-- The regular price of Pony jeans in dollars -/
def pony_price : ℝ := 18

theorem pony_price_calculation :
  fox_price * fox_quantity * (total_discount - pony_discount) +
  pony_price * pony_quantity * pony_discount = total_savings :=
sorry

end pony_price_calculation_l3029_302984


namespace range_of_m_l3029_302933

-- Define the equations
def equation1 (m x : ℝ) := x^2 + m*x + 1 = 0
def equation2 (m x : ℝ) := 4*x^2 + 4*(m-2)*x + 1 = 0

-- Define the conditions
def condition_p (m : ℝ) := ∃ x y, x < 0 ∧ y < 0 ∧ x ≠ y ∧ equation1 m x ∧ equation1 m y
def condition_q (m : ℝ) := ∀ x, ¬(equation2 m x)

-- Theorem statement
theorem range_of_m (m : ℝ) : 
  ((condition_p m ∨ condition_q m) ∧ ¬(condition_p m ∧ condition_q m)) →
  (m ∈ Set.Ioo 1 2 ∪ Set.Ici 3) :=
sorry

end range_of_m_l3029_302933


namespace solve_for_a_l3029_302913

theorem solve_for_a : ∃ a : ℝ, (3 * 3 - 2 * a = 5) ∧ a = 2 := by sorry

end solve_for_a_l3029_302913


namespace subtraction_puzzle_l3029_302997

theorem subtraction_puzzle (X Y : ℕ) : 
  X ≤ 9 → Y ≤ 9 → 45 + 8 * Y = 100 + 10 * X + 2 → X + Y = 10 := by
  sorry

end subtraction_puzzle_l3029_302997


namespace sin_c_special_triangle_l3029_302929

/-- Given a right triangle ABC where A is the right angle, if the logarithms of 
    the side lengths form an arithmetic sequence with a negative common difference, 
    then sin C equals (√5 - 1)/2 -/
theorem sin_c_special_triangle (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_right_angle : a^2 = b^2 + c^2)
  (h_arithmetic_seq : ∃ d : ℝ, d < 0 ∧ Real.log a - Real.log b = d ∧ Real.log b - Real.log c = d) :
  Real.sin (Real.arccos (c / a)) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end sin_c_special_triangle_l3029_302929


namespace min_profit_is_266_l3029_302985

/-- Represents the production plan for the clothing factory -/
structure ProductionPlan where
  typeA : ℕ
  typeB : ℕ

/-- Calculates the total cost for a given production plan -/
def totalCost (plan : ProductionPlan) : ℕ :=
  34 * plan.typeA + 42 * plan.typeB

/-- Calculates the total revenue for a given production plan -/
def totalRevenue (plan : ProductionPlan) : ℕ :=
  39 * plan.typeA + 50 * plan.typeB

/-- Calculates the profit for a given production plan -/
def profit (plan : ProductionPlan) : ℤ :=
  totalRevenue plan - totalCost plan

/-- Theorem: The minimum profit is 266 yuan -/
theorem min_profit_is_266 :
  ∃ (minProfit : ℕ), minProfit = 266 ∧
  ∀ (plan : ProductionPlan),
    plan.typeA + plan.typeB = 40 →
    1536 ≤ totalCost plan →
    totalCost plan ≤ 1552 →
    minProfit ≤ profit plan := by
  sorry

#check min_profit_is_266

end min_profit_is_266_l3029_302985


namespace pigs_in_blanket_calculation_l3029_302972

/-- The number of appetizers per guest -/
def appetizers_per_guest : ℕ := 6

/-- The number of guests -/
def number_of_guests : ℕ := 30

/-- The number of dozen deviled eggs -/
def dozen_deviled_eggs : ℕ := 3

/-- The number of dozen kebabs -/
def dozen_kebabs : ℕ := 2

/-- The additional number of dozen appetizers to make -/
def additional_dozen_appetizers : ℕ := 8

/-- The number of items in a dozen -/
def items_per_dozen : ℕ := 12

theorem pigs_in_blanket_calculation : 
  let total_appetizers := appetizers_per_guest * number_of_guests
  let made_appetizers := dozen_deviled_eggs * items_per_dozen + dozen_kebabs * items_per_dozen
  let remaining_appetizers := total_appetizers - made_appetizers
  let planned_additional_appetizers := additional_dozen_appetizers * items_per_dozen
  let pigs_in_blanket := remaining_appetizers - planned_additional_appetizers
  (pigs_in_blanket / items_per_dozen : ℕ) = 2 := by
  sorry

end pigs_in_blanket_calculation_l3029_302972


namespace sufficient_not_necessary_condition_l3029_302943

/-- A sequence of 8 positive real numbers -/
structure Sequence :=
  (terms : Fin 8 → ℝ)
  (positive : ∀ i, terms i > 0)

/-- Predicate for a geometric sequence -/
def is_geometric (s : Sequence) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ i : Fin 7, s.terms i.succ = q * s.terms i

theorem sufficient_not_necessary_condition (s : Sequence) :
  (s.terms 0 + s.terms 7 < s.terms 3 + s.terms 4 → ¬is_geometric s) ∧
  ∃ s' : Sequence, ¬is_geometric s' ∧ s'.terms 0 + s'.terms 7 ≥ s'.terms 3 + s'.terms 4 :=
sorry

end sufficient_not_necessary_condition_l3029_302943


namespace C14_not_allotrope_C60_l3029_302942

/-- Represents an atom -/
structure Atom where
  name : String

/-- Represents a molecule -/
structure Molecule where
  name : String

/-- Defines the concept of allotrope -/
def is_allotrope (a b : Atom) : Prop :=
  ∃ (element : String), a.name = element ∧ b.name = element

/-- C14 is an atom -/
def C14 : Atom := ⟨"C14"⟩

/-- C60 is a molecule -/
def C60 : Molecule := ⟨"C60"⟩

/-- Theorem stating that C14 is not an allotrope of C60 -/
theorem C14_not_allotrope_C60 : ¬∃ (a : Atom), is_allotrope C14 a ∧ a.name = C60.name := by
  sorry

end C14_not_allotrope_C60_l3029_302942


namespace max_expected_expenditure_l3029_302994

/-- Linear regression model for fiscal revenue and expenditure -/
def fiscal_model (x y a b ε : ℝ) : Prop :=
  y = a + b * x + ε

/-- Theorem: Maximum expected expenditure given fiscal revenue -/
theorem max_expected_expenditure
  (a b x y ε : ℝ)
  (model : fiscal_model x y a b ε)
  (h_a : a = 2)
  (h_b : b = 0.8)
  (h_ε : |ε| ≤ 0.5)
  (h_x : x = 10) :
  y ≤ 10.5 := by
  sorry

#check max_expected_expenditure

end max_expected_expenditure_l3029_302994


namespace triangle_angle_relation_l3029_302905

theorem triangle_angle_relation (A B C : Real) : 
  A + B + C = Real.pi →
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  Real.sin A = Real.cos B →
  Real.sin A = Real.tan C →
  Real.cos A ^ 3 + Real.cos A ^ 2 - Real.cos A = 1 / 2 := by
  sorry

end triangle_angle_relation_l3029_302905


namespace equation_solution_l3029_302939

theorem equation_solution (x : ℝ) :
  (x / 5) / 3 = 9 / (x / 3) → x = 15 * Real.sqrt 1.8 ∨ x = -15 * Real.sqrt 1.8 := by
  sorry

end equation_solution_l3029_302939


namespace opposite_of_negative_two_l3029_302955

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℝ) : ℝ := -a

/-- Prove that the opposite of -2 is 2. -/
theorem opposite_of_negative_two : opposite (-2) = 2 := by
  sorry

end opposite_of_negative_two_l3029_302955


namespace potato_yield_increase_l3029_302928

theorem potato_yield_increase (initial_area initial_yield final_area : ℝ) 
  (h1 : initial_area = 27)
  (h2 : final_area = 24)
  (h3 : initial_area * initial_yield = final_area * (initial_yield * (1 + yield_increase_percentage / 100))) :
  yield_increase_percentage = 12.5 := by
  sorry

end potato_yield_increase_l3029_302928


namespace fraction_zero_l3029_302981

theorem fraction_zero (a : ℝ) : (a^2 - 1) / (a + 1) = 0 ↔ a = 1 :=
by
  sorry

end fraction_zero_l3029_302981


namespace second_number_is_72_l3029_302924

theorem second_number_is_72 (a b c : ℚ) : 
  a + b + c = 264 ∧ 
  a = 2 * b ∧ 
  c = (1/3) * a → 
  b = 72 := by
sorry

end second_number_is_72_l3029_302924


namespace no_solution_lcm_equation_l3029_302925

theorem no_solution_lcm_equation :
  ¬ ∃ (a b : ℕ), 2 * a + 3 * b = Nat.lcm a b := by
  sorry

end no_solution_lcm_equation_l3029_302925


namespace fraction_value_l3029_302982

theorem fraction_value (a b c d : ℝ) 
  (ha : a = 4 * b) 
  (hb : b = 3 * c) 
  (hc : c = 5 * d) : 
  (a * b) / (c * d) = 180 := by
sorry

end fraction_value_l3029_302982


namespace halloween_candy_problem_l3029_302936

/-- The number of candy pieces Robin's sister gave her -/
def candy_from_sister (initial : ℕ) (eaten : ℕ) (final : ℕ) : ℕ :=
  final - (initial - eaten)

theorem halloween_candy_problem :
  let initial := 23
  let eaten := 7
  let final := 37
  candy_from_sister initial eaten final = 21 := by
  sorry

end halloween_candy_problem_l3029_302936


namespace oil_redistribution_l3029_302964

theorem oil_redistribution (trucks_a : Nat) (boxes_a : Nat) (trucks_b : Nat) (boxes_b : Nat) 
  (containers_per_box : Nat) (new_trucks : Nat) :
  trucks_a = 7 →
  boxes_a = 20 →
  trucks_b = 5 →
  boxes_b = 12 →
  containers_per_box = 8 →
  new_trucks = 10 →
  (trucks_a * boxes_a + trucks_b * boxes_b) * containers_per_box / new_trucks = 160 := by
  sorry

end oil_redistribution_l3029_302964


namespace even_cubic_implies_odd_factor_l3029_302968

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function g: ℝ → ℝ is odd if g(-x) = -g(x) for all x ∈ ℝ -/
def IsOdd (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

/-- Given f(x) = x³ * g(x) is an even function, prove that g(x) is odd -/
theorem even_cubic_implies_odd_factor
    (g : ℝ → ℝ) (f : ℝ → ℝ)
    (h1 : ∀ x, f x = x^3 * g x)
    (h2 : IsEven f) :
  IsOdd g :=
by sorry

end even_cubic_implies_odd_factor_l3029_302968


namespace inequalities_hold_l3029_302902

theorem inequalities_hold (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a * b ≤ 1 / 4) ∧ (Real.sqrt a + Real.sqrt b ≤ Real.sqrt 2) ∧ (a^2 + b^2 ≥ 1 / 2) := by
  sorry

end inequalities_hold_l3029_302902


namespace tan_angle_equality_l3029_302945

theorem tan_angle_equality (n : ℤ) : 
  -150 < n ∧ n < 150 ∧ Real.tan (n * π / 180) = Real.tan (1600 * π / 180) → n = -20 :=
by sorry

end tan_angle_equality_l3029_302945


namespace value_of_a_l3029_302992

theorem value_of_a (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) 
  (h : ∀ x : ℝ, x^2 + 2*x^10 = a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
                 a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9 + a₁₀*(x+1)^10) : 
  a = 3 := by
sorry

end value_of_a_l3029_302992


namespace range_of_fraction_l3029_302919

theorem range_of_fraction (x y : ℝ) 
  (h1 : x - 2*y + 4 ≥ 0) 
  (h2 : x ≤ 2) 
  (h3 : x + y - 2 ≥ 0) : 
  1/4 ≤ (y + 1) / (x + 2) ∧ (y + 1) / (x + 2) ≤ 3/2 := by
  sorry

end range_of_fraction_l3029_302919


namespace equation_solution_l3029_302926

theorem equation_solution : 
  ∀ x : ℝ, x * (x - 1) = x ↔ x = 0 ∨ x = 2 := by sorry

end equation_solution_l3029_302926


namespace sufficient_but_not_necessary_l3029_302923

theorem sufficient_but_not_necessary :
  (∀ x : ℝ, x > 1 → x^2 + 2*x > 0) ∧
  (∃ x : ℝ, x^2 + 2*x > 0 ∧ ¬(x > 1)) :=
by sorry

end sufficient_but_not_necessary_l3029_302923


namespace unique_prime_pair_square_sum_l3029_302946

theorem unique_prime_pair_square_sum : 
  ∀ p q : ℕ, 
    Prime p → Prime q → p > 0 → q > 0 →
    (∃ n : ℕ, p^(q-1) + q^(p-1) = n^2) →
    p = 2 ∧ q = 2 :=
by sorry

end unique_prime_pair_square_sum_l3029_302946


namespace pyramid_volume_l3029_302927

theorem pyramid_volume (base_length : ℝ) (base_width : ℝ) (edge_length : ℝ) :
  base_length = 5 →
  base_width = 10 →
  edge_length = 15 →
  let base_area := base_length * base_width
  let diagonal := Real.sqrt (base_length^2 + base_width^2)
  let height := Real.sqrt (edge_length^2 - (diagonal / 2)^2)
  let volume := (1 / 3) * base_area * height
  volume = 232 :=
by sorry

end pyramid_volume_l3029_302927


namespace marker_cost_l3029_302915

theorem marker_cost (total_students : ℕ) (total_cost : ℕ) 
  (h_total_students : total_students = 40)
  (h_total_cost : total_cost = 3388) :
  ∃ (s n c : ℕ),
    s > total_students / 2 ∧
    s ≤ total_students ∧
    n > 1 ∧
    c > n ∧
    s * n * c = total_cost ∧
    c = 11 := by
  sorry

end marker_cost_l3029_302915


namespace equation_A_is_circle_l3029_302978

/-- A polar equation represents a circle if and only if it describes all points at a constant distance from the origin. -/
def is_circle (f : ℝ → ℝ → Prop) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ ρ θ : ℝ, f ρ θ ↔ ρ = r

/-- The polar equation ρ = 1 -/
def equation_A (ρ θ : ℝ) : Prop := ρ = 1

theorem equation_A_is_circle : is_circle equation_A :=
sorry

end equation_A_is_circle_l3029_302978


namespace multiply_polynomial_equals_difference_of_powers_l3029_302957

theorem multiply_polynomial_equals_difference_of_powers (x : ℝ) :
  (x^4 + 25*x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end multiply_polynomial_equals_difference_of_powers_l3029_302957


namespace average_difference_l3029_302906

/-- The number of students in the school -/
def num_students : ℕ := 120

/-- The number of teachers in the school -/
def num_teachers : ℕ := 4

/-- The list of class sizes -/
def class_sizes : List ℕ := [40, 30, 30, 20]

/-- Average number of students per class from a teacher's perspective -/
def t : ℚ := (num_students : ℚ) / num_teachers

/-- Average number of students per class from a student's perspective -/
def s : ℚ := (List.sum (List.map (λ x => x * x) class_sizes) : ℚ) / num_students

theorem average_difference : t - s = -167/100 := by sorry

end average_difference_l3029_302906


namespace white_balls_count_l3029_302941

theorem white_balls_count (red_balls : ℕ) (total_balls : ℕ) (white_balls : ℕ) : 
  red_balls = 3 →
  (red_balls : ℚ) / total_balls = 1 / 4 →
  total_balls = red_balls + white_balls →
  white_balls = 9 := by
sorry

end white_balls_count_l3029_302941


namespace geometric_sequence_sum_l3029_302960

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end geometric_sequence_sum_l3029_302960


namespace original_number_proof_l3029_302908

theorem original_number_proof (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 / x) :
  x = Real.sqrt 30 / 100 := by
sorry

end original_number_proof_l3029_302908


namespace square_perimeter_from_quadratic_root_l3029_302901

theorem square_perimeter_from_quadratic_root : ∃ (x₁ x₂ : ℝ), 
  (x₁ - 1) * (x₁ - 10) = 0 ∧ 
  (x₂ - 1) * (x₂ - 10) = 0 ∧ 
  x₁ ≠ x₂ ∧
  (max x₁ x₂)^2 = 100 ∧
  4 * (max x₁ x₂) = 40 :=
by sorry


end square_perimeter_from_quadratic_root_l3029_302901


namespace greatest_integer_inequality_l3029_302991

theorem greatest_integer_inequality (x : ℤ) :
  (∀ y : ℤ, 3 * y^2 - 5 * y - 2 < 4 - 2 * y → y ≤ 1) ∧
  (3 * 1^2 - 5 * 1 - 2 < 4 - 2 * 1) :=
by sorry

end greatest_integer_inequality_l3029_302991


namespace action_figures_per_shelf_l3029_302961

theorem action_figures_per_shelf 
  (total_figures : ℕ) 
  (num_shelves : ℕ) 
  (h1 : total_figures = 80) 
  (h2 : num_shelves = 8) : 
  total_figures / num_shelves = 10 := by
  sorry

end action_figures_per_shelf_l3029_302961


namespace min_distance_sum_l3029_302999

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Definition of circle C₂ -/
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + (y - 4)^2 = 1

/-- Definition of the locus of point P -/
def P_locus (a b : ℝ) : Prop := b = -(1/2) * a + 5/2

/-- The main theorem -/
theorem min_distance_sum (a b : ℝ) : 
  C₁ a b → C₂ a b → P_locus a b → 
  Real.sqrt (a^2 + b^2) + Real.sqrt ((a - 5)^2 + (b + 1)^2) ≥ Real.sqrt 34 :=
sorry

end min_distance_sum_l3029_302999


namespace can_identify_counterfeit_coins_l3029_302937

/-- Represents the result of checking a pair of coins -/
inductive CheckResult
  | Zero
  | One
  | Two

/-- Represents a coin -/
inductive Coin
  | One
  | Two
  | Three
  | Four
  | Five

/-- A function that checks a pair of coins and returns the number of counterfeit coins -/
def checkPair (c1 c2 : Coin) : CheckResult := sorry

/-- The set of all coins -/
def allCoins : Finset Coin := sorry

/-- The set of counterfeit coins -/
def counterfeitCoins : Finset Coin := sorry

/-- The four pairs of coins to be checked -/
def pairsToCheck : List (Coin × Coin) := sorry

theorem can_identify_counterfeit_coins :
  (Finset.card allCoins = 5) →
  (Finset.card counterfeitCoins = 2) →
  (List.length pairsToCheck = 4) →
  ∃ (f : List CheckResult → Finset Coin),
    ∀ (results : List CheckResult),
      List.length results = 4 →
      results = List.map (fun (p : Coin × Coin) => checkPair p.1 p.2) pairsToCheck →
      f results = counterfeitCoins :=
sorry

end can_identify_counterfeit_coins_l3029_302937


namespace square_of_trinomial_13_5_3_l3029_302983

theorem square_of_trinomial_13_5_3 : (13 + 5 + 3)^2 = 441 := by
  sorry

end square_of_trinomial_13_5_3_l3029_302983


namespace sqrt_sum_inequality_l3029_302990

theorem sqrt_sum_inequality (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) +
  Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) +
  Real.sqrt (e / (a + b + c + d)) > 2 ∧
  ∀ n : ℝ, (∀ a b c d e : ℝ, a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
    Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) +
    Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) +
    Real.sqrt (e / (a + b + c + d)) > n) →
  n ≤ 2 :=
by sorry

end sqrt_sum_inequality_l3029_302990


namespace medal_ratio_is_two_to_one_l3029_302948

/-- The ratio of swimming medals to track medals -/
def medal_ratio (total_medals track_medals badminton_medals : ℕ) : ℚ :=
  let swimming_medals := total_medals - track_medals - badminton_medals
  (swimming_medals : ℚ) / track_medals

/-- Theorem stating that the ratio of swimming medals to track medals is 2:1 -/
theorem medal_ratio_is_two_to_one :
  medal_ratio 20 5 5 = 2 / 1 := by
  sorry

end medal_ratio_is_two_to_one_l3029_302948


namespace polynomial_one_root_product_l3029_302969

theorem polynomial_one_root_product (d e : ℝ) : 
  (∃! x : ℝ, x^2 + d*x + e = 0) → 
  d = 2*e - 3 → 
  ∃ e₁ e₂ : ℝ, (∀ e' : ℝ, (∃ x : ℝ, x^2 + d*x + e' = 0) → (e' = e₁ ∨ e' = e₂)) ∧ 
              e₁ * e₂ = 9/4 :=
by sorry

end polynomial_one_root_product_l3029_302969


namespace six_balls_four_boxes_l3029_302930

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 84 ways to distribute 6 indistinguishable balls into 4 distinguishable boxes -/
theorem six_balls_four_boxes : distribute_balls 6 4 = 84 := by
  sorry

end six_balls_four_boxes_l3029_302930


namespace parabola_equation_l3029_302922

/-- A parabola is defined by its vertex and a point it passes through. -/
structure Parabola where
  vertex : ℝ × ℝ
  point : ℝ × ℝ

/-- The analytical expression of a parabola. -/
def parabola_expression (p : Parabola) : ℝ → ℝ :=
  fun x => -(x + 2)^2 + 3

theorem parabola_equation (p : Parabola) 
  (h1 : p.vertex = (-2, 3)) 
  (h2 : p.point = (1, -6)) : 
  ∀ x, parabola_expression p x = -(x + 2)^2 + 3 := by
  sorry

#check parabola_equation

end parabola_equation_l3029_302922


namespace sum_of_digits_of_expression_l3029_302963

/-- The sum of digits of a natural number in base 10 -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The expression (10^(4n^2 + 8) + 1)^2 -/
def expression (n : ℕ) : ℕ := (10^(4*n^2 + 8) + 1)^2

theorem sum_of_digits_of_expression (n : ℕ) (h : n > 0) : 
  sumOfDigits (expression n) = 4 := by sorry

end sum_of_digits_of_expression_l3029_302963


namespace arithmetic_expression_evaluation_l3029_302973

theorem arithmetic_expression_evaluation : 15 - 2 + 4 / 1 / 2 * 8 = 29 := by
  sorry

end arithmetic_expression_evaluation_l3029_302973


namespace min_marked_cells_13x13_board_l3029_302909

/-- Represents a rectangular board -/
structure Board :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a rectangle that can be placed on the board -/
structure Rectangle :=
  (length : Nat)
  (width : Nat)

/-- Function to calculate the minimum number of cells to mark -/
def minMarkedCells (b : Board) (r : Rectangle) : Nat :=
  sorry

/-- Theorem stating that 84 is the minimum number of cells to mark -/
theorem min_marked_cells_13x13_board (b : Board) (r : Rectangle) :
  b.rows = 13 ∧ b.cols = 13 ∧ r.length = 6 ∧ r.width = 1 →
  minMarkedCells b r = 84 :=
by sorry

end min_marked_cells_13x13_board_l3029_302909


namespace function_sum_l3029_302935

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_sum (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f 2)
  (h_def : ∀ x ∈ Set.Ioo 0 1, f x = Real.sin (Real.pi * x)) :
  f (-5/2) + f 1 + f 2 = -1 := by
sorry

end function_sum_l3029_302935


namespace average_weight_l3029_302912

theorem average_weight (a b c : ℝ) 
  (avg_ab : (a + b) / 2 = 70)
  (avg_bc : (b + c) / 2 = 50)
  (weight_b : b = 60) :
  (a + b + c) / 3 = 60 := by
  sorry

end average_weight_l3029_302912


namespace binomial_expansion_theorem_l3029_302993

theorem binomial_expansion_theorem (a b c k n : ℝ) :
  (n ≥ 2) →
  (a ≠ b) →
  (a * b ≠ 0) →
  (a = k * b + c) →
  (k > 0) →
  (c ≠ 0) →
  (c ≠ b * (k - 1)) →
  (∃ (x y : ℝ), (x + y)^n = (a - b)^n ∧ x + y = 0) →
  (n = -b * (k - 1) / c) := by
  sorry

end binomial_expansion_theorem_l3029_302993


namespace geometric_progression_problem_l3029_302921

theorem geometric_progression_problem (b₃ b₆ : ℚ) 
  (h₁ : b₃ = -1)
  (h₂ : b₆ = 27/8) :
  ∃ (b₁ q : ℚ), 
    b₁ = -4/9 ∧ 
    q = -3/2 ∧ 
    b₃ = b₁ * q^2 ∧ 
    b₆ = b₁ * q^5 := by
  sorry

end geometric_progression_problem_l3029_302921


namespace distribute_six_balls_three_boxes_l3029_302903

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 7 ways to distribute 6 indistinguishable balls into 3 indistinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute_balls 6 3 = 7 := by sorry

end distribute_six_balls_three_boxes_l3029_302903


namespace remainder_theorem_l3029_302911

def polynomial (x : ℝ) : ℝ := 8*x^4 - 6*x^3 + 17*x^2 - 27*x + 35

def divisor (x : ℝ) : ℝ := 2*x - 8

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ),
    polynomial x = (divisor x) * q x + 1863 :=
by sorry

end remainder_theorem_l3029_302911


namespace sequence_existence_iff_N_bound_l3029_302980

theorem sequence_existence_iff_N_bound (N : ℕ+) :
  (∃ s : ℕ → ℕ+, 
    (∀ n, s n < s (n + 1)) ∧ 
    (∃ p : ℕ+, ∀ n, s (n + 1) - s n = s (n + 1 + p) - s (n + p)) ∧
    (∀ n : ℕ+, s (s n) - s (s (n - 1)) ≤ N ∧ N < s (1 + s n) - s (s (n - 1))))
  ↔
  (∃ t : ℕ+, t^2 ≤ N ∧ N < t^2 + t) :=
by sorry

end sequence_existence_iff_N_bound_l3029_302980


namespace exists_two_sum_of_squares_representations_l3029_302995

theorem exists_two_sum_of_squares_representations : 
  ∃ (n : ℕ) (a b c d : ℕ), 
    n < 100 ∧ 
    a ≠ b ∧ 
    c ≠ d ∧ 
    (a, b) ≠ (c, d) ∧
    (a, b) ≠ (d, c) ∧
    n = a^2 + b^2 ∧ 
    n = c^2 + d^2 := by
  sorry

end exists_two_sum_of_squares_representations_l3029_302995


namespace distance_between_points_l3029_302962

def point1 : ℝ × ℝ := (2, -3)
def point2 : ℝ × ℝ := (13, 4)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = Real.sqrt 170 := by
  sorry

end distance_between_points_l3029_302962


namespace inequality_solution_set_l3029_302986

theorem inequality_solution_set (x : ℝ) :
  (x + 5) * (3 - 2*x) ≤ 6 ↔ x ≤ -9/2 ∨ x ≥ 1 := by
  sorry

end inequality_solution_set_l3029_302986


namespace exists_valid_coloring_l3029_302979

/-- A circle in a 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A point on a circle. -/
structure PointOnCircle (c : Circle) where
  point : ℝ × ℝ
  on_circle : (point.1 - c.center.1)^2 + (point.2 - c.center.2)^2 = c.radius^2

/-- A coloring function that assigns either red or blue to each point on the circle. -/
def Coloring (c : Circle) := PointOnCircle c → Bool

/-- Predicate to check if three points form a right-angled triangle. -/
def IsRightAngledTriangle (c : Circle) (p1 p2 p3 : PointOnCircle c) : Prop :=
  ∃ (i j k : Fin 3), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    let points := [p1, p2, p3]
    let a := points[i].point
    let b := points[j].point
    let c := points[k].point
    (b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2) = 0

/-- The main theorem: there exists a coloring such that no inscribed right-angled triangle
    has all vertices of the same color. -/
theorem exists_valid_coloring (c : Circle) :
  ∃ (coloring : Coloring c),
    ∀ (p1 p2 p3 : PointOnCircle c),
      IsRightAngledTriangle c p1 p2 p3 →
        coloring p1 ≠ coloring p2 ∨ coloring p2 ≠ coloring p3 ∨ coloring p1 ≠ coloring p3 :=
by sorry

end exists_valid_coloring_l3029_302979


namespace geometric_sequence_ratio_l3029_302952

/-- For a geometric sequence with common ratio 2, the ratio of the 4th term to the 2nd term is 4. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = 2 * a n) :
  a 4 / a 2 = 4 := by
  sorry

end geometric_sequence_ratio_l3029_302952


namespace rectangle_max_area_l3029_302965

/-- Given a rectangle with perimeter 60 meters and one side three times longer than the other,
    the maximum area is 168.75 square meters. -/
theorem rectangle_max_area (perimeter : ℝ) (ratio : ℝ) (area : ℝ) :
  perimeter = 60 →
  ratio = 3 →
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x = ratio * y ∧ 2 * (x + y) = perimeter ∧ x * y = area) →
  area = 168.75 := by
  sorry

end rectangle_max_area_l3029_302965


namespace proper_divisor_of_two_square_representations_l3029_302974

theorem proper_divisor_of_two_square_representations (n s t u v : ℕ) 
  (h1 : n = s^2 + t^2)
  (h2 : n = u^2 + v^2)
  (h3 : s ≥ t)
  (h4 : t ≥ 0)
  (h5 : u ≥ v)
  (h6 : v ≥ 0)
  (h7 : s > u) :
  1 < Nat.gcd (s * u - t * v) n ∧ Nat.gcd (s * u - t * v) n < n :=
by sorry

end proper_divisor_of_two_square_representations_l3029_302974
