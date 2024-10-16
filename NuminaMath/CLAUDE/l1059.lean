import Mathlib

namespace NUMINAMATH_CALUDE_complex_fourth_power_l1059_105963

theorem complex_fourth_power (i : ℂ) (h : i^2 = -1) : (1 - i)^4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_power_l1059_105963


namespace NUMINAMATH_CALUDE_division_problem_l1059_105901

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 22 →
  divisor = 3 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 7 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1059_105901


namespace NUMINAMATH_CALUDE_triangle_condition_implies_linear_l1059_105982

/-- A function satisfying the triangle condition -/
def TriangleCondition (f : ℝ → ℝ) : Prop :=
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a ≠ b → b ≠ c → a ≠ c →
    (a + b > c ∧ b + c > a ∧ c + a > b ↔ f a + f b > f c ∧ f b + f c > f a ∧ f c + f a > f b)

/-- The main theorem statement -/
theorem triangle_condition_implies_linear (f : ℝ → ℝ) (h : TriangleCondition f) :
  ∃ (c : ℝ), c > 0 ∧ ∀ (x : ℝ), x > 0 → f x = c * x :=
sorry

end NUMINAMATH_CALUDE_triangle_condition_implies_linear_l1059_105982


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1059_105964

theorem expression_simplification_and_evaluation :
  let m : ℝ := Real.sqrt 3
  (m - (m + 9) / (m + 1)) / ((m^2 + 3*m) / (m + 1)) = 1 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1059_105964


namespace NUMINAMATH_CALUDE_exactlyOneBlack_bothBlack_mutuallyExclusive_atLeastOneBlack_bothRed_complementary_l1059_105960

-- Define the sample space
def SampleSpace := Fin 4 × Fin 4

-- Define the events
def exactlyOneBlack (outcome : SampleSpace) : Prop :=
  (outcome.1 = 2 ∧ outcome.2 = 0) ∨ (outcome.1 = 2 ∧ outcome.2 = 1) ∨
  (outcome.1 = 3 ∧ outcome.2 = 0) ∨ (outcome.1 = 3 ∧ outcome.2 = 1)

def bothBlack (outcome : SampleSpace) : Prop :=
  outcome.1 = 2 ∧ outcome.2 = 3

def atLeastOneBlack (outcome : SampleSpace) : Prop :=
  outcome.1 = 2 ∨ outcome.1 = 3 ∨ outcome.2 = 2 ∨ outcome.2 = 3

def bothRed (outcome : SampleSpace) : Prop :=
  outcome.1 = 0 ∧ outcome.2 = 1

-- Theorem statements
theorem exactlyOneBlack_bothBlack_mutuallyExclusive :
  ∀ (outcome : SampleSpace), ¬(exactlyOneBlack outcome ∧ bothBlack outcome) :=
sorry

theorem atLeastOneBlack_bothRed_complementary :
  ∀ (outcome : SampleSpace), atLeastOneBlack outcome ↔ ¬(bothRed outcome) :=
sorry

end NUMINAMATH_CALUDE_exactlyOneBlack_bothBlack_mutuallyExclusive_atLeastOneBlack_bothRed_complementary_l1059_105960


namespace NUMINAMATH_CALUDE_max_profit_fruit_transport_l1059_105986

/-- Represents the fruit transportation problem --/
structure FruitTransport where
  totalCars : Nat
  totalCargo : Nat
  minCarsPerFruit : Nat
  cargoA : Nat
  cargoB : Nat
  cargoC : Nat
  profitA : Nat
  profitB : Nat
  profitC : Nat

/-- Calculates the profit for a given arrangement of cars --/
def calculateProfit (ft : FruitTransport) (x y : Nat) : Nat :=
  ft.profitA * ft.cargoA * x + ft.profitB * ft.cargoB * y + ft.profitC * ft.cargoC * (ft.totalCars - x - y)

/-- Theorem stating the maximum profit and optimal arrangement --/
theorem max_profit_fruit_transport (ft : FruitTransport)
  (h1 : ft.totalCars = 20)
  (h2 : ft.totalCargo = 120)
  (h3 : ft.minCarsPerFruit = 3)
  (h4 : ft.cargoA = 7)
  (h5 : ft.cargoB = 6)
  (h6 : ft.cargoC = 5)
  (h7 : ft.profitA = 1200)
  (h8 : ft.profitB = 1800)
  (h9 : ft.profitC = 1500)
  (h10 : ∀ x y, x + y ≤ ft.totalCars → x ≥ ft.minCarsPerFruit → y ≥ ft.minCarsPerFruit → 
    ft.totalCars - x - y ≥ ft.minCarsPerFruit → ft.cargoA * x + ft.cargoB * y + ft.cargoC * (ft.totalCars - x - y) = ft.totalCargo) :
  ∃ (x y : Nat), x = 3 ∧ y = 14 ∧ calculateProfit ft x y = 198900 ∧
    ∀ (a b : Nat), a + b ≤ ft.totalCars → a ≥ ft.minCarsPerFruit → b ≥ ft.minCarsPerFruit → 
      ft.totalCars - a - b ≥ ft.minCarsPerFruit → calculateProfit ft a b ≤ 198900 :=
by sorry


end NUMINAMATH_CALUDE_max_profit_fruit_transport_l1059_105986


namespace NUMINAMATH_CALUDE_luke_carries_four_trays_l1059_105900

/-- The number of trays Luke can carry at a time -/
def trays_per_trip (trays_table1 trays_table2 total_trips : ℕ) : ℕ :=
  (trays_table1 + trays_table2) / total_trips

/-- Theorem stating that Luke can carry 4 trays at a time -/
theorem luke_carries_four_trays :
  trays_per_trip 20 16 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_luke_carries_four_trays_l1059_105900


namespace NUMINAMATH_CALUDE_binomial_sum_condition_l1059_105977

theorem binomial_sum_condition (n : ℕ) (hn : n ≥ 2) :
  (∀ i j : ℕ, 0 ≤ i ∧ i ≤ n ∧ 0 ≤ j ∧ j ≤ n →
    (i + j) % 2 = (Nat.choose n i + Nat.choose n j) % 2) ↔
  ∃ k : ℕ, k ≥ 2 ∧ n = 2^k - 2 :=
by sorry

end NUMINAMATH_CALUDE_binomial_sum_condition_l1059_105977


namespace NUMINAMATH_CALUDE_farm_animals_l1059_105948

/-- Given a farm with chickens and pigs, prove the number of chickens. -/
theorem farm_animals (total_legs : ℕ) (num_pigs : ℕ) (num_chickens : ℕ) : 
  total_legs = 48 → num_pigs = 9 → 2 * num_chickens + 4 * num_pigs = total_legs → num_chickens = 6 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l1059_105948


namespace NUMINAMATH_CALUDE_sandwich_not_vegetable_percentage_l1059_105990

def sandwich_weight : ℝ := 180
def vegetable_weight : ℝ := 50

theorem sandwich_not_vegetable_percentage :
  let non_vegetable_weight := sandwich_weight - vegetable_weight
  let percentage := (non_vegetable_weight / sandwich_weight) * 100
  ∃ ε > 0, |percentage - 72.22| < ε :=
sorry

end NUMINAMATH_CALUDE_sandwich_not_vegetable_percentage_l1059_105990


namespace NUMINAMATH_CALUDE_lcm_gcd_product_9_12_l1059_105969

theorem lcm_gcd_product_9_12 : Nat.lcm 9 12 * Nat.gcd 9 12 = 108 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_9_12_l1059_105969


namespace NUMINAMATH_CALUDE_product_equality_l1059_105938

theorem product_equality : ∃ x : ℝ, 469138 * x = 4690910862 ∧ x = 10000.1 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l1059_105938


namespace NUMINAMATH_CALUDE_kitten_growth_l1059_105918

/-- Given an initial length of a kitten and two doubling events, calculate the final length. -/
theorem kitten_growth (initial_length : ℝ) : 
  initial_length = 4 → (initial_length * 2 * 2 = 16) := by
  sorry

end NUMINAMATH_CALUDE_kitten_growth_l1059_105918


namespace NUMINAMATH_CALUDE_special_quadrilateral_is_kite_l1059_105923

/-- A quadrilateral with perpendicular diagonals, two adjacent equal sides, and one pair of equal opposite angles -/
structure SpecialQuadrilateral where
  /-- The quadrilateral has perpendicular diagonals -/
  perp_diagonals : Bool
  /-- Two adjacent sides of the quadrilateral are equal -/
  adj_sides_equal : Bool
  /-- One pair of opposite angles are equal -/
  opp_angles_equal : Bool

/-- Definition of a kite -/
def is_kite (q : SpecialQuadrilateral) : Prop :=
  q.perp_diagonals ∧ q.adj_sides_equal

/-- Theorem stating that a quadrilateral with the given properties is a kite -/
theorem special_quadrilateral_is_kite (q : SpecialQuadrilateral) 
  (h1 : q.perp_diagonals = true) 
  (h2 : q.adj_sides_equal = true) 
  (h3 : q.opp_angles_equal = true) : 
  is_kite q :=
sorry

end NUMINAMATH_CALUDE_special_quadrilateral_is_kite_l1059_105923


namespace NUMINAMATH_CALUDE_square_less_than_triple_l1059_105962

theorem square_less_than_triple (x : ℤ) : x^2 < 3*x ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_less_than_triple_l1059_105962


namespace NUMINAMATH_CALUDE_pond_water_after_50_days_l1059_105979

/-- Calculates the remaining water in a pond after a given number of days, 
    given an initial amount and daily evaporation rate. -/
def remaining_water (initial_amount : ℝ) (evaporation_rate : ℝ) (days : ℕ) : ℝ :=
  initial_amount - evaporation_rate * days

/-- Theorem stating that given specific initial conditions, 
    the amount of water remaining after 50 days is 200 gallons. -/
theorem pond_water_after_50_days 
  (initial_amount : ℝ) 
  (evaporation_rate : ℝ) 
  (h1 : initial_amount = 250)
  (h2 : evaporation_rate = 1) :
  remaining_water initial_amount evaporation_rate 50 = 200 :=
by
  sorry

#eval remaining_water 250 1 50

end NUMINAMATH_CALUDE_pond_water_after_50_days_l1059_105979


namespace NUMINAMATH_CALUDE_arnel_shares_with_five_friends_l1059_105974

/-- The number of friends Arnel shares pencils with -/
def number_of_friends (num_boxes : ℕ) (pencils_per_box : ℕ) (kept_pencils : ℕ) (pencils_per_friend : ℕ) : ℕ :=
  ((num_boxes * pencils_per_box) - kept_pencils) / pencils_per_friend

/-- Theorem stating that Arnel shares pencils with 5 friends -/
theorem arnel_shares_with_five_friends :
  number_of_friends 10 5 10 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arnel_shares_with_five_friends_l1059_105974


namespace NUMINAMATH_CALUDE_parabola_vertex_l1059_105992

/-- The parabola defined by y = -3(x-1)^2 - 2 has its vertex at (1, -2) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -3 * (x - 1)^2 - 2 → (1, -2) = (x, y) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1059_105992


namespace NUMINAMATH_CALUDE_book_choice_theorem_l1059_105924

/-- The number of ways to choose 1 book from different sets of books -/
def choose_one_book (diff_lit : Nat) (diff_math : Nat) (id_lit : Nat) (id_math : Nat) : Nat :=
  if diff_lit > 0 ∧ diff_math > 0 then
    diff_lit + diff_math
  else if id_lit > 0 ∧ id_math > 0 then
    (if diff_math > 0 then diff_math + 1 else 2)
  else
    0

/-- Theorem stating the number of ways to choose 1 book in different scenarios -/
theorem book_choice_theorem :
  (choose_one_book 5 4 0 0 = 9) ∧
  (choose_one_book 0 0 5 4 = 5) ∧
  (choose_one_book 0 4 5 0 = 2) := by
  sorry

end NUMINAMATH_CALUDE_book_choice_theorem_l1059_105924


namespace NUMINAMATH_CALUDE_show_completion_time_l1059_105989

theorem show_completion_time (num_episodes : ℕ) (episode_length : ℕ) (daily_watch_time : ℕ) : 
  num_episodes = 20 → 
  episode_length = 30 → 
  daily_watch_time = 120 → 
  (num_episodes * episode_length) / daily_watch_time = 5 :=
by sorry

end NUMINAMATH_CALUDE_show_completion_time_l1059_105989


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1059_105905

theorem quadratic_equation_properties (x y : ℝ) 
  (h : (x - y)^2 - 2*(x + y) + 1 = 0) : 
  (x ≥ 0 ∧ y ≥ 0) ∧ 
  (x > 1 ∧ y < x → Real.sqrt x - Real.sqrt y = 1) ∧
  (x < 1 ∧ y < 1 → Real.sqrt x + Real.sqrt y = 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1059_105905


namespace NUMINAMATH_CALUDE_sqrt_four_equals_two_l1059_105920

theorem sqrt_four_equals_two : Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_equals_two_l1059_105920


namespace NUMINAMATH_CALUDE_largest_non_formable_is_correct_l1059_105927

/-- Represents the coin denominations in Limonia -/
def coin_denominations (n : ℕ) : Finset ℕ :=
  {3*n - 1, 6*n + 1, 6*n + 4, 6*n + 7}

/-- Predicate to check if an amount can be formed using the given coin denominations -/
def is_formable (n : ℕ) (amount : ℕ) : Prop :=
  ∃ (a b c d : ℕ), amount = a*(3*n - 1) + b*(6*n + 1) + c*(6*n + 4) + d*(6*n + 7)

/-- The largest non-formable amount in Limonia -/
def largest_non_formable (n : ℕ) : ℕ := 6*n^2 + 4*n - 5

/-- Theorem stating that the largest non-formable amount is correct -/
theorem largest_non_formable_is_correct (n : ℕ) :
  (∀ m : ℕ, m > largest_non_formable n → is_formable n m) ∧
  ¬is_formable n (largest_non_formable n) :=
sorry

end NUMINAMATH_CALUDE_largest_non_formable_is_correct_l1059_105927


namespace NUMINAMATH_CALUDE_max_x_minus_y_l1059_105922

theorem max_x_minus_y (x y z : ℝ) 
  (sum_eq : x + y + z = 2) 
  (prod_eq : x*y + y*z + z*x = 1) : 
  ∃ (max : ℝ), max = 2 / Real.sqrt 3 ∧ 
  ∀ (a b c : ℝ), a + b + c = 2 → a*b + b*c + c*a = 1 → 
  |a - b| ≤ max := by
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l1059_105922


namespace NUMINAMATH_CALUDE_boys_without_laptops_l1059_105998

theorem boys_without_laptops (total_boys : ℕ) (total_laptops : ℕ) (girls_with_laptops : ℕ) : 
  total_boys = 20 → 
  total_laptops = 25 → 
  girls_with_laptops = 16 → 
  total_boys - (total_laptops - girls_with_laptops) = 11 := by
sorry

end NUMINAMATH_CALUDE_boys_without_laptops_l1059_105998


namespace NUMINAMATH_CALUDE_travel_time_proof_l1059_105971

/-- Given a person traveling at 20 km/hr for a distance of 160 km,
    prove that the time taken is 8 hours. -/
theorem travel_time_proof (speed : ℝ) (distance : ℝ) (time : ℝ)
    (h1 : speed = 20)
    (h2 : distance = 160)
    (h3 : time * speed = distance) :
  time = 8 :=
by sorry

end NUMINAMATH_CALUDE_travel_time_proof_l1059_105971


namespace NUMINAMATH_CALUDE_shift_hours_is_eight_l1059_105955

/-- Calculates the number of hours in each person's shift given the following conditions:
  * 20 people are hired
  * Each person makes on average 20 shirts per day
  * Employees are paid $12 an hour plus $5 per shirt
  * Shirts are sold for $35 each
  * Nonemployee expenses are $1000 a day
  * The company makes $9080 in profits per day
-/
def calculateShiftHours (
  numEmployees : ℕ)
  (shirtsPerPerson : ℕ)
  (hourlyWage : ℕ)
  (perShirtBonus : ℕ)
  (shirtPrice : ℕ)
  (nonEmployeeExpenses : ℕ)
  (dailyProfit : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the shift hours calculated by the function is 8 -/
theorem shift_hours_is_eight :
  calculateShiftHours 20 20 12 5 35 1000 9080 = 8 := by
  sorry

end NUMINAMATH_CALUDE_shift_hours_is_eight_l1059_105955


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1059_105985

theorem polynomial_simplification (y : ℝ) :
  (3 * y - 2) * (6 * y^12 + 3 * y^11 + 6 * y^10 + 3 * y^9) =
  18 * y^13 - 3 * y^12 + 12 * y^11 - 3 * y^10 - 6 * y^9 := by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1059_105985


namespace NUMINAMATH_CALUDE_almost_perfect_is_odd_square_l1059_105931

/-- Sum of divisors function -/
def sigma (N : ℕ+) : ℕ := sorry

/-- Definition of almost perfect number -/
def is_almost_perfect (N : ℕ+) : Prop :=
  sigma N = 2 * N.val + 1

/-- Main theorem: Every almost perfect number is a square of an odd number -/
theorem almost_perfect_is_odd_square (N : ℕ+) (h : is_almost_perfect N) :
  ∃ M : ℕ, N.val = M^2 ∧ Odd M := by sorry

end NUMINAMATH_CALUDE_almost_perfect_is_odd_square_l1059_105931


namespace NUMINAMATH_CALUDE_students_suggesting_bacon_l1059_105934

theorem students_suggesting_bacon (total : ℕ) (mashed_potatoes : ℕ) (tomatoes : ℕ) 
  (h1 : total = 826)
  (h2 : mashed_potatoes = 324)
  (h3 : tomatoes = 128) :
  total - (mashed_potatoes + tomatoes) = 374 := by
  sorry

end NUMINAMATH_CALUDE_students_suggesting_bacon_l1059_105934


namespace NUMINAMATH_CALUDE_DE_DB_ratio_l1059_105949

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
variable (right_angle_ABC : (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0)
variable (right_angle_ABD : (D.1 - A.1) * (B.1 - A.1) + (D.2 - A.2) * (B.2 - A.2) = 0)
variable (AC_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 2)
variable (BC_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 3)
variable (AD_length : Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = 7)
variable (C_D_opposite : (B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1) * (B.1 - A.1) * (D.2 - A.2) - (B.2 - A.2) * (D.1 - A.1) < 0)
variable (D_parallel_AC : (D.2 - A.2) * (C.1 - A.1) = (D.1 - A.1) * (C.2 - A.2))
variable (E_on_CB_extended : ∃ t : ℝ, E = (B.1 + t * (C.1 - B.1), B.2 + t * (C.2 - B.2)) ∧ t > 1)

-- Theorem statement
theorem DE_DB_ratio :
  Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) / Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 5 / 9 :=
sorry

end NUMINAMATH_CALUDE_DE_DB_ratio_l1059_105949


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1059_105966

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 2 = 8 →
  (2 * a 4 - a 3 = a 3 - 4 * a 5) →
  (a 1 + a 2 + a 3 + a 4 + a 5 = 31) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1059_105966


namespace NUMINAMATH_CALUDE_min_square_sum_min_square_sum_achievable_l1059_105970

theorem min_square_sum (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : x₁ + 3 * x₂ + 2 * x₃ = 50) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 1250 / 7 := by
  sorry

theorem min_square_sum_achievable : 
  ∃ x₁ x₂ x₃ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ 
  x₁ + 3 * x₂ + 2 * x₃ = 50 ∧ 
  x₁^2 + x₂^2 + x₃^2 = 1250 / 7 := by
  sorry

end NUMINAMATH_CALUDE_min_square_sum_min_square_sum_achievable_l1059_105970


namespace NUMINAMATH_CALUDE_anthony_lunch_money_l1059_105991

theorem anthony_lunch_money (juice_cost cupcake_cost remaining_amount : ℕ) 
  (h1 : juice_cost = 27)
  (h2 : cupcake_cost = 40)
  (h3 : remaining_amount = 8) :
  juice_cost + cupcake_cost + remaining_amount = 75 := by
  sorry

end NUMINAMATH_CALUDE_anthony_lunch_money_l1059_105991


namespace NUMINAMATH_CALUDE_coffee_shop_revenue_l1059_105984

/-- The number of customers who ordered coffee -/
def coffee_customers : ℕ := 7

/-- The price of a cup of coffee in dollars -/
def coffee_price : ℕ := 5

/-- The number of customers who ordered tea -/
def tea_customers : ℕ := 8

/-- The price of a cup of tea in dollars -/
def tea_price : ℕ := 4

/-- The total revenue of the coffee shop in dollars -/
def total_revenue : ℕ := 67

theorem coffee_shop_revenue :
  coffee_customers * coffee_price + tea_customers * tea_price = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_coffee_shop_revenue_l1059_105984


namespace NUMINAMATH_CALUDE_square_of_digit_sum_sum_of_cube_digits_l1059_105957

-- Define the sum of digits function
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Part a
theorem square_of_digit_sum (N : ℕ) : N = (sumOfDigits N)^2 ↔ N = 1 ∨ N = 81 := by sorry

-- Part b
theorem sum_of_cube_digits (N : ℕ) : N = sumOfDigits (N^3) ↔ N = 1 ∨ N = 8 ∨ N = 17 ∨ N = 18 ∨ N = 26 ∨ N = 27 := by sorry

end NUMINAMATH_CALUDE_square_of_digit_sum_sum_of_cube_digits_l1059_105957


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l1059_105958

/-- Given plane vectors a and b, where the angle between them is 60°,
    a = (2,0), and |b| = 1, then |a+b| = √7 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  (a = (2, 0)) →
  (Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 1) →
  (Real.cos (60 * Real.pi / 180) = a.1 * b.1 + a.2 * b.2) →
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l1059_105958


namespace NUMINAMATH_CALUDE_doughnuts_theorem_l1059_105921

/-- The number of boxes of doughnuts -/
def num_boxes : ℕ := 4

/-- The number of doughnuts in each box -/
def doughnuts_per_box : ℕ := 12

/-- The total number of doughnuts -/
def total_doughnuts : ℕ := num_boxes * doughnuts_per_box

theorem doughnuts_theorem : total_doughnuts = 48 := by
  sorry

end NUMINAMATH_CALUDE_doughnuts_theorem_l1059_105921


namespace NUMINAMATH_CALUDE_female_employees_count_l1059_105913

/-- Represents the number of employees in a company -/
structure Company where
  total_employees : ℕ
  female_managers : ℕ
  male_employees : ℕ
  female_employees : ℕ

/-- Conditions for the company -/
def company_conditions (c : Company) : Prop :=
  c.female_managers = 200 ∧
  c.total_employees * 2 = (c.female_managers + (c.male_employees * 2 / 5)) * 5 ∧
  c.total_employees = c.male_employees + c.female_employees

/-- Theorem stating that under the given conditions, the number of female employees is 500 -/
theorem female_employees_count (c : Company) :
  company_conditions c → c.female_employees = 500 := by
  sorry

end NUMINAMATH_CALUDE_female_employees_count_l1059_105913


namespace NUMINAMATH_CALUDE_no_function_satisfies_equation_l1059_105980

theorem no_function_satisfies_equation :
  ¬ ∃ (f : ℝ → ℝ), ∀ (x y : ℝ), f (x + y) = x * f x + y := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_equation_l1059_105980


namespace NUMINAMATH_CALUDE_function_properties_l1059_105953

noncomputable def f (a b x : ℝ) := a * Real.sin (2 * x) + b * Real.cos (2 * x)

theorem function_properties (a b : ℝ) (h1 : a * b ≠ 0) 
  (h2 : ∀ x : ℝ, f a b x ≤ |f a b (π/6)|) : 
  (f a b (11*π/12) = 0) ∧ 
  (|f a b (7*π/12)| < |f a b (π/5)|) ∧ 
  (∀ x : ℝ, f a b (-x) ≠ f a b x ∧ f a b (-x) ≠ -f a b x) ∧
  (∀ k m : ℝ, ∃ x : ℝ, k * x + m = f a b x) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1059_105953


namespace NUMINAMATH_CALUDE_distinctNumbers_eq_2001_l1059_105994

/-- The number of distinct numbers in the list [⌊1²/500⌋, ⌊2²/500⌋, ⌊3²/500⌋, ..., ⌊1000²/500⌋] -/
def distinctNumbers : ℕ :=
  let list := List.range 1000
  let floorList := list.map (fun n => Int.floor ((n + 1)^2 / 500 : ℚ))
  floorList.eraseDups.length

/-- The theorem stating that the number of distinct numbers in the list is 2001 -/
theorem distinctNumbers_eq_2001 : distinctNumbers = 2001 := by
  sorry

end NUMINAMATH_CALUDE_distinctNumbers_eq_2001_l1059_105994


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l1059_105910

theorem scientific_notation_equivalence : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 3080000 = a * (10 : ℝ) ^ n :=
by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l1059_105910


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1059_105959

theorem quadratic_real_roots (a b c : ℝ) (ha : a ≠ 0) (hac : a * c < 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1059_105959


namespace NUMINAMATH_CALUDE_jerichos_remaining_money_l1059_105936

def jerichos_money_problem (jerichos_money : ℚ) (debt_to_annika : ℚ) : Prop :=
  2 * jerichos_money = 60 ∧
  debt_to_annika = 14 ∧
  let debt_to_manny := debt_to_annika / 2
  let remaining_money := jerichos_money - debt_to_annika - debt_to_manny
  remaining_money = 9

theorem jerichos_remaining_money :
  ∀ (jerichos_money : ℚ) (debt_to_annika : ℚ),
  jerichos_money_problem jerichos_money debt_to_annika :=
by
  sorry

end NUMINAMATH_CALUDE_jerichos_remaining_money_l1059_105936


namespace NUMINAMATH_CALUDE_decompose_power_l1059_105907

theorem decompose_power (a : ℝ) (h : a > 0) : 
  ∃ (x y z w : ℝ), 
    a^(3/4) = a^x * a^y * a^z * a^w ∧ 
    y = x + 1/6 ∧ 
    z = y + 1/6 ∧ 
    w = z + 1/6 ∧
    x = -1/16 ∧ 
    y = 5/48 ∧ 
    z = 13/48 ∧ 
    w = 7/16 := by
  sorry

end NUMINAMATH_CALUDE_decompose_power_l1059_105907


namespace NUMINAMATH_CALUDE_negation_existence_positive_real_l1059_105973

theorem negation_existence_positive_real (R_plus : Set ℝ) :
  (¬ ∃ x ∈ R_plus, x > x^2) ↔ (∀ x ∈ R_plus, x ≤ x^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_existence_positive_real_l1059_105973


namespace NUMINAMATH_CALUDE_larger_number_problem_l1059_105911

theorem larger_number_problem (L S : ℕ) 
  (h1 : L - S = 2415)
  (h2 : L = 21 * S + 15) : 
  L = 2535 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1059_105911


namespace NUMINAMATH_CALUDE_pentagonal_pyramid_edges_and_faces_l1059_105952

/-- A pentagonal pyramid is a polyhedron with a pentagonal base and triangular lateral faces. -/
structure PentagonalPyramid where
  base_edges : ℕ
  lateral_edges : ℕ
  lateral_faces : ℕ
  base_faces : ℕ

/-- The properties of a pentagonal pyramid. -/
def pentagonal_pyramid : PentagonalPyramid :=
  { base_edges := 5
  , lateral_edges := 5
  , lateral_faces := 5
  , base_faces := 1
  }

/-- The number of edges in a pentagonal pyramid. -/
def num_edges (p : PentagonalPyramid) : ℕ := p.base_edges + p.lateral_edges

/-- The number of faces in a pentagonal pyramid. -/
def num_faces (p : PentagonalPyramid) : ℕ := p.lateral_faces + p.base_faces

theorem pentagonal_pyramid_edges_and_faces :
  num_edges pentagonal_pyramid = 10 ∧ num_faces pentagonal_pyramid = 6 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_pyramid_edges_and_faces_l1059_105952


namespace NUMINAMATH_CALUDE_limit_cosine_fraction_l1059_105947

theorem limit_cosine_fraction :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x| ∧ |x| < δ → 
    |(1 - Real.cos (2*x)) / (Real.cos (7*x) - Real.cos (3*x)) + (1/10)| < ε := by
sorry

end NUMINAMATH_CALUDE_limit_cosine_fraction_l1059_105947


namespace NUMINAMATH_CALUDE_more_solutions_for_first_eq_l1059_105903

/-- The upper bound for x, y, z, and t -/
def upperBound : ℕ := 10^6

/-- The number of integral solutions of x² - y² = z³ - t³ -/
def N : ℕ := sorry

/-- The number of integral solutions of x² - y² = z³ - t³ + 1 -/
def M : ℕ := sorry

/-- Predicate for the first equation -/
def firstEq (x y z t : ℕ) : Prop :=
  x^2 - y^2 = z^3 - t^3 ∧ x ≤ upperBound ∧ y ≤ upperBound ∧ z ≤ upperBound ∧ t ≤ upperBound

/-- Predicate for the second equation -/
def secondEq (x y z t : ℕ) : Prop :=
  x^2 - y^2 = z^3 - t^3 + 1 ∧ x ≤ upperBound ∧ y ≤ upperBound ∧ z ≤ upperBound ∧ t ≤ upperBound

/-- Theorem stating that N > M -/
theorem more_solutions_for_first_eq : N > M := by
  sorry

end NUMINAMATH_CALUDE_more_solutions_for_first_eq_l1059_105903


namespace NUMINAMATH_CALUDE_rick_card_distribution_l1059_105909

theorem rick_card_distribution (total_cards : ℕ) (kept_cards : ℕ) (miguel_cards : ℕ) 
  (num_friends : ℕ) (num_sisters : ℕ) (sister_cards : ℕ) 
  (h1 : total_cards = 130) 
  (h2 : kept_cards = 15)
  (h3 : miguel_cards = 13)
  (h4 : num_friends = 8)
  (h5 : num_sisters = 2)
  (h6 : sister_cards = 3) :
  (total_cards - kept_cards - miguel_cards - num_sisters * sister_cards) / num_friends = 12 := by
  sorry

end NUMINAMATH_CALUDE_rick_card_distribution_l1059_105909


namespace NUMINAMATH_CALUDE_square_area_ratio_l1059_105965

/-- If the perimeter of one square is 4 times the perimeter of another square,
    then the area of the larger square is 16 times the area of the smaller square. -/
theorem square_area_ratio (s L : ℝ) (hs : s > 0) (hL : L > 0) 
    (h_perimeter : 4 * L = 4 * (4 * s)) : L^2 = 16 * s^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1059_105965


namespace NUMINAMATH_CALUDE_remainder_sum_l1059_105999

theorem remainder_sum (c d : ℤ) (hc : c % 60 = 53) (hd : d % 40 = 29) :
  (c + d) % 20 = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_sum_l1059_105999


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1059_105961

/-- Given a hyperbola and conditions on its asymptote and focus, prove its equation --/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ k : ℝ, (a / b = Real.sqrt 3) ∧ 
   (∃ x y : ℝ, x^2 = 24*y ∧ (y^2 / a^2 - x^2 / b^2 = 1))) →
  (∃ x y : ℝ, y^2 / 27 - x^2 / 9 = 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1059_105961


namespace NUMINAMATH_CALUDE_problem_solution_l1059_105914

theorem problem_solution (a b c d : ℝ) : 
  8 = (4 / 100) * a →
  4 = (d / 100) * a →
  8 = (d / 100) * b →
  c = b / a →
  c = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1059_105914


namespace NUMINAMATH_CALUDE_unique_prime_triple_l1059_105919

theorem unique_prime_triple : 
  ∀ p q r : ℕ,
  (Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r) →
  (Nat.Prime (4 * q - 1)) →
  ((p + q : ℚ) / (p + r) = r - p) →
  (p = 2 ∧ q = 3 ∧ r = 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_triple_l1059_105919


namespace NUMINAMATH_CALUDE_younger_brother_age_l1059_105995

theorem younger_brother_age 
  (older younger : ℕ) 
  (sum_condition : older + younger = 46)
  (age_relation : younger = older / 3 + 10) :
  younger = 19 := by
  sorry

end NUMINAMATH_CALUDE_younger_brother_age_l1059_105995


namespace NUMINAMATH_CALUDE_expression_equality_l1059_105902

theorem expression_equality : 
  Real.sqrt 27 / (Real.sqrt 3 / 2) * (2 * Real.sqrt 2) - 6 * Real.sqrt 2 = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1059_105902


namespace NUMINAMATH_CALUDE_line_equation_proof_l1059_105940

/-- A line in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y = l.c

/-- Check if a vector is normal to a line -/
def isNormalVector (l : Line2D) (v : Vector2D) : Prop :=
  l.a * v.x + l.b * v.y = 0

/-- The main theorem -/
theorem line_equation_proof (l : Line2D) (A : Point2D) (n : Vector2D) :
  l.a = 2 ∧ l.b = 1 ∧ l.c = 2 →
  A.x = 1 ∧ A.y = 0 →
  n.x = 2 ∧ n.y = -1 →
  pointOnLine l A ∧ isNormalVector l n := by
  sorry

#check line_equation_proof

end NUMINAMATH_CALUDE_line_equation_proof_l1059_105940


namespace NUMINAMATH_CALUDE_distance_between_foci_rectangular_hyperbola_l1059_105941

/-- The distance between the foci of a rectangular hyperbola -/
theorem distance_between_foci_rectangular_hyperbola (c : ℝ) :
  let hyperbola := {(x, y) : ℝ × ℝ | x * y = c^2}
  let foci := {(c, c), (-c, -c)}
  (Set.ncard foci = 2) →
  ∀ (f₁ f₂ : ℝ × ℝ), f₁ ∈ foci → f₂ ∈ foci → f₁ ≠ f₂ →
    Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = 2 * Real.sqrt 2 * c :=
by sorry

end NUMINAMATH_CALUDE_distance_between_foci_rectangular_hyperbola_l1059_105941


namespace NUMINAMATH_CALUDE_smaller_number_problem_l1059_105997

theorem smaller_number_problem (x y : ℝ) : 
  y = 2 * x - 3 →   -- One number is 3 less than twice the other
  x + y = 39 →      -- Their sum is 39
  x = 14            -- The smaller number is 14
  := by sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l1059_105997


namespace NUMINAMATH_CALUDE_g_equals_h_intersection_at_most_one_point_l1059_105906

-- Define the functions g and h
def g (x : ℝ) : ℝ := 2 * x - 1
def h (t : ℝ) : ℝ := 2 * t - 1

-- Theorem 1: g and h are the same function
theorem g_equals_h : g = h := by sorry

-- Theorem 2: For any function f, the intersection of y = f(x) and x = 2 has at most one point
theorem intersection_at_most_one_point (f : ℝ → ℝ) :
  ∃! y, f 2 = y := by sorry

end NUMINAMATH_CALUDE_g_equals_h_intersection_at_most_one_point_l1059_105906


namespace NUMINAMATH_CALUDE_parabola_properties_l1059_105972

/-- Definition of the parabola -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- Definition of the directrix -/
def directrix (x : ℝ) : Prop := x = -4

/-- Definition of a line passing through (-4, 0) with slope m -/
def line (m x y : ℝ) : Prop := y = m * (x + 4)

/-- Theorem stating the properties of the parabola and its intersecting lines -/
theorem parabola_properties :
  (∃ (x y : ℝ), directrix x ∧ y = 0) ∧ 
  (∀ (m : ℝ), (∃ (x y : ℝ), parabola x y ∧ line m x y) ↔ m ≤ 1/2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l1059_105972


namespace NUMINAMATH_CALUDE_inverse_function_derivative_l1059_105968

-- Define the function f and its inverse g
variable (f : ℝ → ℝ) (g : ℝ → ℝ)
variable (x₀ : ℝ) (y₀ : ℝ)

-- State the conditions
variable (hf : Differentiable ℝ f)
variable (hg : Differentiable ℝ g)
variable (hinverse : Function.LeftInverse g f ∧ Function.RightInverse g f)
variable (hderiv : (deriv f x₀) ≠ 0)
variable (hy : y₀ = f x₀)

-- State the theorem
theorem inverse_function_derivative :
  (deriv g y₀) = 1 / (deriv f x₀) := by sorry

end NUMINAMATH_CALUDE_inverse_function_derivative_l1059_105968


namespace NUMINAMATH_CALUDE_expression_equals_two_power_thirty_l1059_105983

theorem expression_equals_two_power_thirty :
  (((16^16 / 16^14)^3 * 8^6) / 2^12) = 2^30 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_two_power_thirty_l1059_105983


namespace NUMINAMATH_CALUDE_fruit_display_total_l1059_105908

/-- Proves that the total number of fruits on a display is 35, given specific conditions --/
theorem fruit_display_total (bananas oranges apples : ℕ) : 
  bananas = 5 →
  oranges = 2 * bananas →
  apples = 2 * oranges →
  bananas + oranges + apples = 35 := by
sorry

end NUMINAMATH_CALUDE_fruit_display_total_l1059_105908


namespace NUMINAMATH_CALUDE_oil_depth_theorem_l1059_105928

/-- Represents a horizontal cylindrical tank with oil -/
structure OilTank where
  length : ℝ
  diameter : ℝ
  oil_surface_area : ℝ

/-- Calculates the possible depths of oil in the tank -/
def oil_depths (tank : OilTank) : Set ℝ :=
  { h | h = 3 - Real.sqrt 5 ∨ h = 3 + Real.sqrt 5 }

theorem oil_depth_theorem (tank : OilTank) 
  (h_length : tank.length = 10)
  (h_diameter : tank.diameter = 6)
  (h_area : tank.oil_surface_area = 40) :
  ∀ h ∈ oil_depths tank, 
    ∃ c : ℝ, 
      c = tank.oil_surface_area / tank.length ∧ 
      c ^ 2 = 2 * (tank.diameter / 2) * h - h ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_oil_depth_theorem_l1059_105928


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1059_105967

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(1 + b²/a²) -/
theorem hyperbola_eccentricity (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let e := Real.sqrt (1 + b^2 / a^2)
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) → e = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1059_105967


namespace NUMINAMATH_CALUDE_units_digit_of_n_l1059_105916

/-- Given two natural numbers m and n, returns true if their product has a units digit of 1 -/
def product_has_units_digit_one (m n : ℕ) : Prop :=
  (m * n) % 10 = 1

/-- Given a natural number m, returns true if it has a units digit of 9 -/
def has_units_digit_nine (m : ℕ) : Prop :=
  m % 10 = 9

theorem units_digit_of_n (m n : ℕ) 
  (h1 : m * n = 11^4)
  (h2 : has_units_digit_nine m) :
  n % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l1059_105916


namespace NUMINAMATH_CALUDE_exists_removable_factorial_for_perfect_square_l1059_105978

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

def product_of_factorials (n : ℕ) : ℕ := (Finset.range n).prod (λ i => factorial (i + 1))

theorem exists_removable_factorial_for_perfect_square :
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ 100 ∧ 
  ∃ m : ℕ, product_of_factorials 100 / factorial k = m ^ 2 :=
sorry

end NUMINAMATH_CALUDE_exists_removable_factorial_for_perfect_square_l1059_105978


namespace NUMINAMATH_CALUDE_fraction_of_7000_l1059_105926

theorem fraction_of_7000 (x : ℝ) : x = 0.101 →
  x * 7000 - (1 / 1000) * 7000 = 700 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_7000_l1059_105926


namespace NUMINAMATH_CALUDE_room_width_calculation_l1059_105993

/-- Proves that given a rectangular room with length 5.5 meters, if the total cost of paving the floor at a rate of 1200 Rs per square meter is 24750 Rs, then the width of the room is 3.75 meters. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) (width : ℝ) : 
  length = 5.5 → 
  cost_per_sqm = 1200 → 
  total_cost = 24750 → 
  width = total_cost / cost_per_sqm / length → 
  width = 3.75 := by
sorry


end NUMINAMATH_CALUDE_room_width_calculation_l1059_105993


namespace NUMINAMATH_CALUDE_shoe_multiple_l1059_105942

theorem shoe_multiple (bonny_shoes becky_shoes bobby_shoes : ℕ) : 
  bonny_shoes = 13 →
  bonny_shoes = 2 * becky_shoes - 5 →
  bobby_shoes = 27 →
  ∃ m : ℕ, m * becky_shoes = bobby_shoes ∧ m = 3 := by
sorry

end NUMINAMATH_CALUDE_shoe_multiple_l1059_105942


namespace NUMINAMATH_CALUDE_white_square_area_l1059_105976

theorem white_square_area (cube_edge : ℝ) (blue_paint_area : ℝ) : 
  cube_edge = 12 → 
  blue_paint_area = 432 → 
  (cube_edge ^ 2 * 6 - blue_paint_area) / 6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_white_square_area_l1059_105976


namespace NUMINAMATH_CALUDE_x_is_soccer_ball_price_l1059_105944

-- Define the variables
variable (x : ℝ)  -- Unit price of soccer balls
variable (y : ℝ)  -- Unit price of basketballs
variable (s : ℕ)  -- Number of soccer balls
variable (b : ℕ)  -- Number of basketballs

-- Define the conditions
axiom soccer_twice_basketball : s = 2 * b
axiom total_soccer_cost : s * x = 5000
axiom total_basketball_cost : b * y = 4000
axiom price_difference : y = x + 30
axiom equation_holds : 5000 / x = 2 * (4000 / (30 + x))

-- Theorem to prove
theorem x_is_soccer_ball_price : x = (5000 : ℝ) / s :=
sorry

end NUMINAMATH_CALUDE_x_is_soccer_ball_price_l1059_105944


namespace NUMINAMATH_CALUDE_incompatible_inequalities_l1059_105912

theorem incompatible_inequalities (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ¬(a + b < c + d ∧ 
    (a + b) * (c + d) < a * b + c * d ∧ 
    (a + b) * c * d < (c + d) * a * b) :=
by sorry

end NUMINAMATH_CALUDE_incompatible_inequalities_l1059_105912


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l1059_105975

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  (∃! x, 2 * a * x^2 + 10 * x + c = 0) →
  a + c = 12 →
  a < c →
  a = 1.15 ∧ c = 10.85 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l1059_105975


namespace NUMINAMATH_CALUDE_perfect_square_factors_of_4410_l1059_105956

/-- Given that 4410 = 2 × 3² × 5 × 7², this function counts the number of positive integer factors of 4410 that are perfect squares. -/
def count_perfect_square_factors : ℕ :=
  let prime_factorization := [(2, 1), (3, 2), (5, 1), (7, 2)]
  sorry

/-- The theorem states that the number of positive integer factors of 4410 that are perfect squares is 4. -/
theorem perfect_square_factors_of_4410 : count_perfect_square_factors = 4 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_factors_of_4410_l1059_105956


namespace NUMINAMATH_CALUDE_total_peonies_count_l1059_105987

/-- Represents the number of peonies in each category -/
structure PeonyCount where
  single : ℕ
  double : ℕ
  thousand : ℕ

/-- The total number of peonies -/
def total_peonies (count : PeonyCount) : ℕ :=
  count.single + count.double + count.thousand

/-- The difference between thousand-petal and single-petal peonies -/
def thousand_single_difference (count : PeonyCount) : ℤ :=
  count.thousand - count.single

/-- The sample of peonies used for observation -/
def sample : PeonyCount :=
  { single := 4, double := 2, thousand := 6 }

/-- The theorem stating the total number of peonies -/
theorem total_peonies_count :
  ∃ (count : PeonyCount),
    total_peonies count = 180 ∧
    thousand_single_difference count = 30 ∧
    (∃ (k : ℕ),
      count.single = k * sample.single ∧
      count.double = k * sample.double ∧
      count.thousand = k * sample.thousand) :=
sorry

end NUMINAMATH_CALUDE_total_peonies_count_l1059_105987


namespace NUMINAMATH_CALUDE_three_circles_collinearity_l1059_105950

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the point structure
structure Point where
  x : ℝ
  y : ℝ

-- Define the line structure
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem three_circles_collinearity 
  (circleA circleB circleC : Circle)
  (A B C : Point)
  (B₁ C₁ C₂ A₂ A₃ B₃ : Point)
  (X Y Z : Point) :
  -- Conditions
  circleA.center = (A.x, A.y) →
  circleB.center = (B.x, B.y) →
  circleC.center = (C.x, C.y) →
  circleA.radius = circleB.radius →
  circleB.radius = circleC.radius →
  -- B₁ and C₁ are on circleA
  (B₁.x - A.x)^2 + (B₁.y - A.y)^2 = circleA.radius^2 →
  (C₁.x - A.x)^2 + (C₁.y - A.y)^2 = circleA.radius^2 →
  -- C₂ and A₂ are on circleB
  (C₂.x - B.x)^2 + (C₂.y - B.y)^2 = circleB.radius^2 →
  (A₂.x - B.x)^2 + (A₂.y - B.y)^2 = circleB.radius^2 →
  -- A₃ and B₃ are on circleC
  (A₃.x - C.x)^2 + (A₃.y - C.y)^2 = circleC.radius^2 →
  (B₃.x - C.x)^2 + (B₃.y - C.y)^2 = circleC.radius^2 →
  -- X is the intersection of B₁C₁ and BC
  (∃ (l₁ : Line), l₁.a * B₁.x + l₁.b * B₁.y + l₁.c = 0 ∧
                  l₁.a * C₁.x + l₁.b * C₁.y + l₁.c = 0 ∧
                  l₁.a * X.x + l₁.b * X.y + l₁.c = 0) →
  (∃ (l₂ : Line), l₂.a * B.x + l₂.b * B.y + l₂.c = 0 ∧
                  l₂.a * C.x + l₂.b * C.y + l₂.c = 0 ∧
                  l₂.a * X.x + l₂.b * X.y + l₂.c = 0) →
  -- Y is the intersection of C₂A₂ and CA
  (∃ (l₃ : Line), l₃.a * C₂.x + l₃.b * C₂.y + l₃.c = 0 ∧
                  l₃.a * A₂.x + l₃.b * A₂.y + l₃.c = 0 ∧
                  l₃.a * Y.x + l₃.b * Y.y + l₃.c = 0) →
  (∃ (l₄ : Line), l₄.a * C.x + l₄.b * C.y + l₄.c = 0 ∧
                  l₄.a * A.x + l₄.b * A.y + l₄.c = 0 ∧
                  l₄.a * Y.x + l₄.b * Y.y + l₄.c = 0) →
  -- Z is the intersection of A₃B₃ and AB
  (∃ (l₅ : Line), l₅.a * A₃.x + l₅.b * A₃.y + l₅.c = 0 ∧
                  l₅.a * B₃.x + l₅.b * B₃.y + l₅.c = 0 ∧
                  l₅.a * Z.x + l₅.b * Z.y + l₅.c = 0) →
  (∃ (l₆ : Line), l₆.a * A.x + l₆.b * A.y + l₆.c = 0 ∧
                  l₆.a * B.x + l₆.b * B.y + l₆.c = 0 ∧
                  l₆.a * Z.x + l₆.b * Z.y + l₆.c = 0) →
  -- Conclusion: X, Y, and Z are collinear
  ∃ (l : Line), l.a * X.x + l.b * X.y + l.c = 0 ∧
                l.a * Y.x + l.b * Y.y + l.c = 0 ∧
                l.a * Z.x + l.b * Z.y + l.c = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_three_circles_collinearity_l1059_105950


namespace NUMINAMATH_CALUDE_inclination_angle_range_l1059_105946

/-- The range of inclination angles for a line passing through (1, 1) and (2, m²) -/
theorem inclination_angle_range (m : ℝ) : 
  let α := Real.arctan (m^2 - 1)
  0 ≤ α ∧ α < π/2 ∨ 3*π/4 ≤ α ∧ α < π := by
  sorry

end NUMINAMATH_CALUDE_inclination_angle_range_l1059_105946


namespace NUMINAMATH_CALUDE_melies_initial_money_l1059_105904

/-- The amount of meat Méliès bought in kilograms -/
def meat_amount : ℝ := 2

/-- The cost of meat per kilogram in dollars -/
def meat_cost_per_kg : ℝ := 82

/-- The amount of money Méliès has left after paying for the meat in dollars -/
def money_left : ℝ := 16

/-- The initial amount of money in Méliès' wallet in dollars -/
def initial_money : ℝ := meat_amount * meat_cost_per_kg + money_left

theorem melies_initial_money :
  initial_money = 180 :=
sorry

end NUMINAMATH_CALUDE_melies_initial_money_l1059_105904


namespace NUMINAMATH_CALUDE_fraction_change_l1059_105981

theorem fraction_change (a b p q x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : b ≠ 0) 
  (h3 : (a^2 + x^2) / (b^2 + x^2) = p / q) 
  (h4 : q ≠ p) : 
  x^2 = (b^2 * p - a^2 * q) / (q - p) := by
  sorry

end NUMINAMATH_CALUDE_fraction_change_l1059_105981


namespace NUMINAMATH_CALUDE_largest_single_digit_divisor_of_9984_l1059_105929

theorem largest_single_digit_divisor_of_9984 : ∃ (d : ℕ), d < 10 ∧ d ∣ 9984 ∧ ∀ (n : ℕ), n < 10 ∧ n ∣ 9984 → n ≤ d :=
by sorry

end NUMINAMATH_CALUDE_largest_single_digit_divisor_of_9984_l1059_105929


namespace NUMINAMATH_CALUDE_cubic_sum_problem_l1059_105932

theorem cubic_sum_problem (a b c : ℂ) 
  (sum_condition : a + b + c = 2)
  (product_sum_condition : a * b + a * c + b * c = -1)
  (product_condition : a * b * c = -8) :
  a^3 + b^3 + c^3 = 69 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_problem_l1059_105932


namespace NUMINAMATH_CALUDE_coffee_cream_ratio_l1059_105988

/-- The ratio of cream in Joe's coffee to JoAnn's coffee -/
theorem coffee_cream_ratio :
  let initial_coffee : ℝ := 20
  let joe_drank : ℝ := 3
  let cream_added : ℝ := 4
  let joann_drank : ℝ := 3
  let joe_cream : ℝ := cream_added
  let joann_total : ℝ := initial_coffee + cream_added
  let joann_cream_ratio : ℝ := cream_added / joann_total
  let joann_cream : ℝ := cream_added - (joann_drank * joann_cream_ratio)
  (joe_cream / joann_cream) = 8 / 7 := by
sorry

end NUMINAMATH_CALUDE_coffee_cream_ratio_l1059_105988


namespace NUMINAMATH_CALUDE_month_days_l1059_105930

theorem month_days (days_taken : ℕ) (days_forgotten : ℕ) : 
  days_taken = 27 → days_forgotten = 4 → days_taken + days_forgotten = 31 := by
sorry

end NUMINAMATH_CALUDE_month_days_l1059_105930


namespace NUMINAMATH_CALUDE_prob_at_least_six_heads_in_eight_flips_prob_at_least_six_heads_in_eight_flips_proof_l1059_105954

/-- The probability of getting at least 6 heads in 8 fair coin flips -/
theorem prob_at_least_six_heads_in_eight_flips : ℚ :=
  37 / 256

/-- Proof that the probability of getting at least 6 heads in 8 fair coin flips is 37/256 -/
theorem prob_at_least_six_heads_in_eight_flips_proof :
  prob_at_least_six_heads_in_eight_flips = 37 / 256 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_six_heads_in_eight_flips_prob_at_least_six_heads_in_eight_flips_proof_l1059_105954


namespace NUMINAMATH_CALUDE_arithmetic_sequences_equal_sum_l1059_105996

/-- Sum of first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ d n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequences_equal_sum :
  ∃! (n : ℕ), n > 0 ∧ arithmetic_sum 5 5 n = arithmetic_sum 22 3 n :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_equal_sum_l1059_105996


namespace NUMINAMATH_CALUDE_max_carlson_jars_l1059_105937

/-- Represents the initial state of jam jars for Carlson and Baby -/
structure JamJars where
  carlson_weights : List ℕ
  baby_weights : List ℕ

/-- Checks if the given JamJars satisfies the initial condition -/
def initial_condition (jars : JamJars) : Prop :=
  (jars.carlson_weights.sum = 13 * jars.baby_weights.sum) ∧
  (∀ w ∈ jars.carlson_weights, w > 0) ∧
  (∀ w ∈ jars.baby_weights, w > 0)

/-- Checks if the given JamJars satisfies the final condition after transfer -/
def final_condition (jars : JamJars) : Prop :=
  let smallest := jars.carlson_weights.minimum?
  match smallest with
  | some min =>
    ((jars.carlson_weights.sum - min) = 8 * (jars.baby_weights.sum + min)) ∧
    (∀ w ∈ jars.carlson_weights, w ≥ min)
  | none => False

/-- The main theorem stating the maximum number of jars Carlson could have initially had -/
theorem max_carlson_jars (jars : JamJars) :
  initial_condition jars → final_condition jars →
  jars.carlson_weights.length ≤ 23 :=
by sorry

#check max_carlson_jars

end NUMINAMATH_CALUDE_max_carlson_jars_l1059_105937


namespace NUMINAMATH_CALUDE_sanchez_rope_theorem_l1059_105915

/-- Represents the number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- Represents the number of feet of rope bought last week -/
def last_week_feet : ℕ := 6

/-- Represents the difference in feet between last week's and this week's purchase -/
def difference_feet : ℕ := 4

/-- Calculates the total inches of rope bought by Mr. Sanchez -/
def total_inches : ℕ := 
  (last_week_feet * inches_per_foot) + ((last_week_feet - difference_feet) * inches_per_foot)

/-- Theorem stating that the total inches of rope bought is 96 -/
theorem sanchez_rope_theorem : total_inches = 96 := by
  sorry

end NUMINAMATH_CALUDE_sanchez_rope_theorem_l1059_105915


namespace NUMINAMATH_CALUDE_function_inequality_equivalence_l1059_105939

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x

theorem function_inequality_equivalence (a : ℝ) :
  (a > 0) →
  (∀ m n : ℝ, m > 0 → n > 0 → m ≠ n →
    Real.sqrt (m * n) + (m + n) / 2 > (m - n) / (f a m - f a n)) ↔
  a ≥ 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_equivalence_l1059_105939


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_35_l1059_105925

theorem modular_inverse_of_5_mod_35 : 
  ∃ x : ℕ, x < 35 ∧ (5 * x) % 35 = 1 :=
by
  use 29
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_35_l1059_105925


namespace NUMINAMATH_CALUDE_small_bottle_price_l1059_105933

/-- The price of a small bottle given the following conditions:
  * 1375 large bottles were purchased at $1.75 each
  * 690 small bottles were purchased
  * The average price of all bottles is approximately $1.6163438256658595
-/
theorem small_bottle_price :
  let large_bottles : ℕ := 1375
  let small_bottles : ℕ := 690
  let large_price : ℝ := 1.75
  let avg_price : ℝ := 1.6163438256658595
  let total_bottles : ℕ := large_bottles + small_bottles
  let small_price : ℝ := (avg_price * total_bottles - large_price * large_bottles) / small_bottles
  ∃ ε > 0, |small_price - 1.34988436247191| < ε := by
  sorry

end NUMINAMATH_CALUDE_small_bottle_price_l1059_105933


namespace NUMINAMATH_CALUDE_range_of_m_l1059_105951

-- Define the functions f and g
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + c
def g (c m : ℝ) (x : ℝ) : ℝ := x * (f c x + m*x - 5)

-- State the theorem
theorem range_of_m (c : ℝ) :
  (∃! x, f c x = 0) →
  (∃ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3 ∧ g c m x₁ < g c m x₂ ∧ g c m x₂ > g c m x₁) →
  -1/3 < m ∧ m < 5/4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1059_105951


namespace NUMINAMATH_CALUDE_problem_solution_l1059_105945

-- Define the line l: x + my + 2√3 = 0
def line_l (m : ℝ) (x y : ℝ) : Prop := x + m * y + 2 * Real.sqrt 3 = 0

-- Define the circle O: x² + y² = r² (r > 0)
def circle_O (r : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = r^2 ∧ r > 0

-- Define line l': x = 3
def line_l' (x : ℝ) : Prop := x = 3

theorem problem_solution :
  -- Part 1
  (∀ r : ℝ, (∀ m : ℝ, ∃ x y : ℝ, line_l m x y ∧ circle_O r x y) ↔ r ≥ 2 * Real.sqrt 3) ∧
  -- Part 2
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    circle_O 5 x₁ y₁ ∧ circle_O 5 x₂ y₂ ∧ 
    (∃ m : ℝ, line_l m x₁ y₁ ∧ line_l m x₂ y₂) →
    2 * Real.sqrt 13 ≤ Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≤ 10) ∧
  -- Part 3
  (∀ s t : ℝ, s^2 + t^2 = 1 →
    ∃ x y : ℝ, 
      (x - 3)^2 + (y - (1 - 3*s)/t)^2 = ((3 - s)/t)^2 ∧
      (x = 3 + 2 * Real.sqrt 2 ∧ y = 0 ∨ x = 3 - 2 * Real.sqrt 2 ∧ y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1059_105945


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1059_105917

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ -4) (h2 : x ≠ 12) :
  (7 * x - 3) / (x^2 - 8 * x - 48) = 11 / (x + 4) + 0 / (x - 12) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1059_105917


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1059_105943

-- Define the quadratic function
def f (a b x : ℝ) := x^2 + b*x + a

-- State the theorem
theorem quadratic_inequality_solution (a b : ℝ) 
  (h : ∀ x, f a b x > 0 ↔ x ∈ Set.Iio 1 ∪ Set.Ioi 5) : 
  a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1059_105943


namespace NUMINAMATH_CALUDE_area_comparison_l1059_105935

-- Define a polygon as a list of points in 2D space
def Polygon := List (Real × Real)

-- Function to calculate the area of a polygon
noncomputable def area (p : Polygon) : Real := sorry

-- Function to check if a polygon is convex
def isConvex (p : Polygon) : Prop := sorry

-- Function to check if two polygons have equal corresponding sides
def equalSides (p1 p2 : Polygon) : Prop := sorry

-- Function to check if a polygon is inscribed in a circle
def isInscribed (p : Polygon) : Prop := sorry

-- Theorem statement
theorem area_comparison 
  (A B : Polygon) 
  (h1 : isConvex A) 
  (h2 : isConvex B) 
  (h3 : equalSides A B) 
  (h4 : isInscribed B) : 
  area B ≥ area A := by sorry

end NUMINAMATH_CALUDE_area_comparison_l1059_105935
