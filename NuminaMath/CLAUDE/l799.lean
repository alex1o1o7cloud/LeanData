import Mathlib

namespace magnitude_of_complex_fraction_l799_79973

theorem magnitude_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.abs (2 * i / (1 + i)) = Real.sqrt 2 := by
  sorry

end magnitude_of_complex_fraction_l799_79973


namespace sum_remainder_mod_seven_l799_79927

theorem sum_remainder_mod_seven : 
  (102345 + 102346 + 102347 + 102348 + 102349 + 102350) % 7 = 5 := by
sorry

end sum_remainder_mod_seven_l799_79927


namespace oil_consumption_ranking_l799_79920

/-- Oil consumption per person for each region -/
structure OilConsumption where
  west : ℝ
  nonWest : ℝ
  russia : ℝ

/-- The ranking of oil consumption is correct if Russia > Non-West > West -/
def correctRanking (consumption : OilConsumption) : Prop :=
  consumption.russia > consumption.nonWest ∧ consumption.nonWest > consumption.west

/-- Theorem stating that the given oil consumption data results in the correct ranking -/
theorem oil_consumption_ranking (consumption : OilConsumption) 
  (h_west : consumption.west = 55.084)
  (h_nonWest : consumption.nonWest = 214.59)
  (h_russia : consumption.russia = 1038.33) :
  correctRanking consumption := by
  sorry

#check oil_consumption_ranking

end oil_consumption_ranking_l799_79920


namespace sin_period_scaled_l799_79924

/-- The period of the function y = sin(x/3) is 6π -/
theorem sin_period_scaled (x : ℝ) : 
  ∃ (p : ℝ), p > 0 ∧ ∀ (t : ℝ), Real.sin (t / 3) = Real.sin ((t + p) / 3) ∧ p = 6 * Real.pi :=
by sorry

end sin_period_scaled_l799_79924


namespace tire_purchase_l799_79960

theorem tire_purchase (cost_per_tire : ℚ) (total_cost : ℚ) (num_tires : ℕ) : 
  cost_per_tire = 1/2 →
  total_cost = 4 →
  num_tires = (total_cost / cost_per_tire).num →
  num_tires = 8 := by
sorry

end tire_purchase_l799_79960


namespace sum_of_solutions_quadratic_l799_79990

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 + 5*x - 24 = 4*x + 38) → 
  (∃ a b : ℝ, (a + b = -1) ∧ (x = a ∨ x = b)) :=
by
  sorry

end sum_of_solutions_quadratic_l799_79990


namespace natural_numbers_satisfying_conditions_l799_79926

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def sum_of_digits (n : ℕ) : ℕ := sorry

def num_positive_divisors (n : ℕ) : ℕ := sorry

def has_form_4k_plus_3 (p : ℕ) : Prop := ∃ k : ℕ, p = 4 * k + 3

def has_prime_divisor_with_4_or_more_digits (n : ℕ) : Prop := 
  ∃ p : ℕ, is_prime p ∧ p ∣ n ∧ p ≥ 1000

theorem natural_numbers_satisfying_conditions (n : ℕ) : 
  (∀ m : ℕ, m > 1 → is_square m → ¬(m ∣ n)) ∧
  (∃! p : ℕ, is_prime p ∧ p ∣ n ∧ has_form_4k_plus_3 p) ∧
  (sum_of_digits n + 2 = num_positive_divisors n) ∧
  (is_square (n + 3)) ∧
  (¬has_prime_divisor_with_4_or_more_digits n) ↔
  (n = 222 ∨ n = 2022) := by sorry

end natural_numbers_satisfying_conditions_l799_79926


namespace interest_rate_calculation_l799_79976

/-- Simple interest calculation -/
def simple_interest (principal time rate : ℝ) : ℝ := principal * time * rate

/-- Theorem: Given the conditions, prove the rate of interest is 0.06 -/
theorem interest_rate_calculation (principal time interest : ℝ) 
  (h_principal : principal = 15000)
  (h_time : time = 3)
  (h_interest : interest = 2700) :
  ∃ rate : ℝ, simple_interest principal time rate = interest ∧ rate = 0.06 := by
  sorry

#check interest_rate_calculation

end interest_rate_calculation_l799_79976


namespace volume_circumscribed_sphere_unit_cube_l799_79939

/-- The volume of a circumscribed sphere of a cube with edge length 1 -/
theorem volume_circumscribed_sphere_unit_cube :
  let edge_length : ℝ := 1
  let radius : ℝ := (Real.sqrt 3) / 2
  let volume : ℝ := (4/3) * Real.pi * radius^3
  volume = (Real.sqrt 3 / 2) * Real.pi := by
sorry

end volume_circumscribed_sphere_unit_cube_l799_79939


namespace sheila_hourly_wage_l799_79946

/-- Sheila's work schedule and earnings -/
structure WorkSchedule where
  hours_mwf : ℕ  -- Hours worked on Monday, Wednesday, Friday
  days_mwf : ℕ   -- Number of days worked with hours_mwf
  hours_tt : ℕ   -- Hours worked on Tuesday, Thursday
  days_tt : ℕ    -- Number of days worked with hours_tt
  weekly_earnings : ℕ  -- Weekly earnings in dollars

/-- Calculate Sheila's hourly wage -/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  let total_hours := schedule.hours_mwf * schedule.days_mwf + schedule.hours_tt * schedule.days_tt
  schedule.weekly_earnings / total_hours

/-- Theorem: Sheila's hourly wage is $12 -/
theorem sheila_hourly_wage :
  let sheila_schedule : WorkSchedule := {
    hours_mwf := 8,
    days_mwf := 3,
    hours_tt := 6,
    days_tt := 2,
    weekly_earnings := 432
  }
  hourly_wage sheila_schedule = 12 := by
  sorry


end sheila_hourly_wage_l799_79946


namespace conic_is_pair_of_lines_l799_79915

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop := 9 * x^2 - 36 * y^2 = 0

/-- The first line of the pair -/
def line1 (x y : ℝ) : Prop := x = 2 * y

/-- The second line of the pair -/
def line2 (x y : ℝ) : Prop := x = -2 * y

/-- Theorem stating that the conic equation represents a pair of straight lines -/
theorem conic_is_pair_of_lines :
  ∀ x y : ℝ, conic_equation x y ↔ (line1 x y ∨ line2 x y) :=
sorry

end conic_is_pair_of_lines_l799_79915


namespace sin_30_deg_value_l799_79998

theorem sin_30_deg_value (f : ℝ → ℝ) (h : ∀ x, f (Real.cos x) = Real.cos (3 * x)) :
  f (Real.sin (π / 6)) = -1 := by
  sorry

end sin_30_deg_value_l799_79998


namespace system_of_equations_solution_l799_79900

theorem system_of_equations_solution :
  ∃ (x y : ℚ), (4 * x - 6 * y = -14) ∧ (8 * x + 3 * y = -15) ∧ (x = -11/5) ∧ (y = 13/15) := by
sorry

end system_of_equations_solution_l799_79900


namespace balloon_ratio_l799_79909

theorem balloon_ratio (mary_balloons nancy_balloons : ℕ) 
  (h1 : mary_balloons = 28) (h2 : nancy_balloons = 7) :
  mary_balloons / nancy_balloons = 4 := by
  sorry

end balloon_ratio_l799_79909


namespace geometric_sequence_ratio_l799_79916

theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) : 
  let a₂ := a₁ * q
  let a₃ := a₁ * q^2
  let S₃ := a₁ + a₂ + a₃
  (S₃ = 13 ∧ 2 * (a₂ + 2) = a₁ + a₃) → (q = 3 ∨ q = 1/3) :=
by sorry

end geometric_sequence_ratio_l799_79916


namespace smallest_number_with_8_divisors_multiple_of_24_l799_79945

def is_multiple_of_24 (n : ℕ) : Prop := ∃ k : ℕ, n = 24 * k

def count_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem smallest_number_with_8_divisors_multiple_of_24 :
  ∀ n : ℕ, is_multiple_of_24 n ∧ count_divisors n = 8 → n ≥ 720 :=
by sorry

end smallest_number_with_8_divisors_multiple_of_24_l799_79945


namespace middle_five_sum_l799_79931

theorem middle_five_sum (total : ℕ) (avg_all : ℚ) (avg_first_ten : ℚ) (avg_last_ten : ℚ) (avg_middle_seven : ℚ) :
  total = 21 →
  avg_all = 44 →
  avg_first_ten = 48 →
  avg_last_ten = 41 →
  avg_middle_seven = 45 →
  (total * avg_all : ℚ) = (10 * avg_first_ten + 10 * avg_last_ten + (total - 20) * avg_middle_seven : ℚ) →
  (5 : ℕ) * ((7 : ℕ) * avg_middle_seven - avg_first_ten - avg_last_ten) = 226 :=
by sorry

end middle_five_sum_l799_79931


namespace no_primes_divisible_by_42_l799_79997

theorem no_primes_divisible_by_42 : 
  ∀ p : ℕ, Prime p → ¬(42 ∣ p) :=
by
  sorry

end no_primes_divisible_by_42_l799_79997


namespace min_marked_cells_for_unique_determination_l799_79980

/-- Represents a 9x9 board -/
def Board := Fin 9 → Fin 9 → Bool

/-- An L-shaped piece covering 3 cells -/
structure LPiece where
  x : Fin 9
  y : Fin 9
  orientation : Fin 4

/-- Checks if a given L-piece is uniquely determined by the marked cells -/
def isUniqueDetermination (board : Board) (piece : LPiece) : Bool :=
  sorry

/-- Checks if all possible L-piece placements are uniquely determined -/
def allPiecesUnique (board : Board) : Bool :=
  sorry

/-- Counts the number of marked cells on the board -/
def countMarkedCells (board : Board) : Nat :=
  sorry

/-- The main theorem: The minimum number of marked cells for unique determination is 63 -/
theorem min_marked_cells_for_unique_determination :
  ∃ (board : Board), allPiecesUnique board ∧ countMarkedCells board = 63 ∧
  ∀ (other_board : Board), allPiecesUnique other_board → countMarkedCells other_board ≥ 63 :=
sorry

end min_marked_cells_for_unique_determination_l799_79980


namespace negation_of_existential_real_exp_l799_79994

theorem negation_of_existential_real_exp (p : Prop) : 
  (p ↔ ∃ x : ℝ, Real.exp x < 0) → 
  (¬p ↔ ∀ x : ℝ, Real.exp x ≥ 0) := by
  sorry

end negation_of_existential_real_exp_l799_79994


namespace cake_price_is_twelve_l799_79925

/-- Represents the daily sales and expenses of Marie's bakery --/
structure BakeryFinances where
  cashRegisterCost : ℕ
  breadPrice : ℕ
  breadQuantity : ℕ
  cakeQuantity : ℕ
  rentCost : ℕ
  electricityCost : ℕ
  profitDays : ℕ

/-- Calculates the price of each cake based on the given finances --/
def calculateCakePrice (finances : BakeryFinances) : ℕ :=
  let dailyBreadIncome := finances.breadPrice * finances.breadQuantity
  let dailyExpenses := finances.rentCost + finances.electricityCost
  let dailyProfitWithoutCakes := dailyBreadIncome - dailyExpenses
  let totalProfit := finances.cashRegisterCost
  let profitFromCakes := totalProfit - (finances.profitDays * dailyProfitWithoutCakes)
  profitFromCakes / (finances.cakeQuantity * finances.profitDays)

/-- Theorem stating that the cake price is $12 given the specific conditions --/
theorem cake_price_is_twelve (finances : BakeryFinances)
  (h1 : finances.cashRegisterCost = 1040)
  (h2 : finances.breadPrice = 2)
  (h3 : finances.breadQuantity = 40)
  (h4 : finances.cakeQuantity = 6)
  (h5 : finances.rentCost = 20)
  (h6 : finances.electricityCost = 2)
  (h7 : finances.profitDays = 8) :
  calculateCakePrice finances = 12 := by
  sorry

end cake_price_is_twelve_l799_79925


namespace inequality_system_solutions_l799_79968

def inequality_system (x t : ℝ) : Prop :=
  6 - (2 * x + 5) > -15 ∧ (x + 3) / 2 - t < x

theorem inequality_system_solutions :
  (∀ x : ℤ, inequality_system x 2 → x ≥ 0) ∧
  (∃ x : ℤ, inequality_system x 2 ∧ x = 0) ∧
  (∀ x : ℝ, inequality_system x 4 ↔ -5 < x ∧ x < 8) ∧
  (∃! t : ℝ, ∀ x : ℝ, inequality_system x t ↔ -5 < x ∧ x < 8) ∧
  (∀ t : ℝ, (∃! (a b c : ℤ), 
    inequality_system (a : ℝ) t ∧ 
    inequality_system (b : ℝ) t ∧ 
    inequality_system (c : ℝ) t ∧ 
    a < b ∧ b < c ∧
    (∀ x : ℤ, inequality_system (x : ℝ) t → x = a ∨ x = b ∨ x = c)) 
    ↔ -1 < t ∧ t ≤ -1/2) :=
by sorry

end inequality_system_solutions_l799_79968


namespace sum_with_radical_conjugate_l799_79961

-- Define the radical conjugate
def radical_conjugate (a b : ℝ) : ℝ := a - b

-- Theorem statement
theorem sum_with_radical_conjugate :
  let x : ℝ := 15 - Real.sqrt 500
  let y : ℝ := radical_conjugate 15 (Real.sqrt 500)
  x + y = 30 := by
  sorry

end sum_with_radical_conjugate_l799_79961


namespace polynomial_factorization_l799_79901

theorem polynomial_factorization (a : ℝ) :
  (a^2 + 2*a)*(a^2 + 2*a + 2) + 1 = (a + 1)^4 := by
  sorry

end polynomial_factorization_l799_79901


namespace probability_of_black_ball_l799_79975

theorem probability_of_black_ball (prob_red prob_white : ℝ) 
  (h_red : prob_red = 0.42)
  (h_white : prob_white = 0.28)
  (h_sum : prob_red + prob_white + (1 - prob_red - prob_white) = 1) :
  1 - prob_red - prob_white = 0.30 := by
sorry

end probability_of_black_ball_l799_79975


namespace rectangle_cutting_l799_79917

/-- Represents a rectangle on a cartesian plane with sides parallel to coordinate axes -/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h1 : x_min < x_max
  h2 : y_min < y_max

/-- Predicate to check if a vertical line intersects a rectangle -/
def vertical_intersects (x : ℝ) (r : Rectangle) : Prop :=
  r.x_min < x ∧ x < r.x_max

/-- Predicate to check if a horizontal line intersects a rectangle -/
def horizontal_intersects (y : ℝ) (r : Rectangle) : Prop :=
  r.y_min < y ∧ y < r.y_max

/-- Any two rectangles can be cut by a vertical or a horizontal line -/
axiom rectangle_separation (r1 r2 : Rectangle) :
  (∃ x : ℝ, vertical_intersects x r1 ∧ vertical_intersects x r2) ∨
  (∃ y : ℝ, horizontal_intersects y r1 ∧ horizontal_intersects y r2)

/-- The main theorem -/
theorem rectangle_cutting (rectangles : Set Rectangle) :
  ∃ (x y : ℝ), ∀ r ∈ rectangles, vertical_intersects x r ∨ horizontal_intersects y r :=
sorry

end rectangle_cutting_l799_79917


namespace complex_expression_simplification_l799_79949

theorem complex_expression_simplification :
  (7 - 3*Complex.I) - 4*(2 + 5*Complex.I) + 3*(1 - 4*Complex.I) = 2 - 35*Complex.I :=
by sorry

end complex_expression_simplification_l799_79949


namespace eric_ben_difference_l799_79936

theorem eric_ben_difference (jack ben eric : ℕ) : 
  jack = 26 → 
  ben = jack - 9 → 
  eric + ben + jack = 50 → 
  ben - eric = 10 := by
sorry

end eric_ben_difference_l799_79936


namespace min_value_inequality_l799_79951

theorem min_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : b + c ≥ a) :
  b / c + c / (a + b) ≥ Real.sqrt 2 - 1 / 2 := by
  sorry

end min_value_inequality_l799_79951


namespace simplify_fraction_l799_79904

theorem simplify_fraction (a : ℝ) (h : a ≠ 2) :
  (a - 2) * ((a^2 - 4) / (a^2 - 4*a + 4)) = a + 2 := by
  sorry

end simplify_fraction_l799_79904


namespace modular_inverse_13_mod_64_l799_79986

theorem modular_inverse_13_mod_64 :
  ∃ x : ℕ, x < 64 ∧ (13 * x) % 64 = 1 :=
by
  use 5
  sorry

end modular_inverse_13_mod_64_l799_79986


namespace handshakes_at_gathering_l799_79983

theorem handshakes_at_gathering (n : ℕ) (h : n = 6) : 
  n * (2 * n - 1) = 60 := by
  sorry

#check handshakes_at_gathering

end handshakes_at_gathering_l799_79983


namespace trajectory_of_moving_circle_l799_79923

-- Define the two fixed circles
def C1 (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 2
def C2 (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 2

-- Define a predicate for a point being on the trajectory
def OnTrajectory (x y : ℝ) : Prop :=
  x^2 / 2 - y^2 / 14 = 1 ∨ x = 0

-- Theorem statement
theorem trajectory_of_moving_circle :
  ∀ (x y r : ℝ),
  (∃ (x1 y1 : ℝ), C1 x1 y1 ∧ (x - x1)^2 + (y - y1)^2 = r^2) →
  (∃ (x2 y2 : ℝ), C2 x2 y2 ∧ (x - x2)^2 + (y - y2)^2 = r^2) →
  OnTrajectory x y :=
sorry

end trajectory_of_moving_circle_l799_79923


namespace chaperones_count_l799_79958

/-- Calculates the number of volunteer chaperones given the number of children,
    additional lunches, cost per lunch, and total cost. -/
def calculate_chaperones (children : ℕ) (additional : ℕ) (cost_per_lunch : ℕ) (total_cost : ℕ) : ℕ :=
  (total_cost / cost_per_lunch) - children - additional - 1

/-- Theorem stating that the number of volunteer chaperones is 6 given the problem conditions. -/
theorem chaperones_count :
  let children : ℕ := 35
  let additional : ℕ := 3
  let cost_per_lunch : ℕ := 7
  let total_cost : ℕ := 308
  calculate_chaperones children additional cost_per_lunch total_cost = 6 := by
  sorry

#eval calculate_chaperones 35 3 7 308

end chaperones_count_l799_79958


namespace sum_product_inequality_l799_79929

theorem sum_product_inequality (a b c d : ℝ) (h : a + b + c + d = 0) :
  5 * (a * b + b * c + c * d) + 8 * (a * c + a * d + b * d) ≤ 0 := by
  sorry

end sum_product_inequality_l799_79929


namespace social_relationships_theorem_l799_79977

/-- Represents the relationship between two people -/
inductive Relationship
  | Knows
  | DoesNotKnow

/-- A function representing the relationship between people -/
def relationship (people : ℕ) : (Fin people → Fin people → Relationship) :=
  sorry

theorem social_relationships_theorem (n : ℕ) :
  ∃ (A B : Fin (2*n+2)), ∃ (S : Finset (Fin (2*n+2))),
    S.card ≥ n ∧
    (∀ C ∈ S, C ≠ A ∧ C ≠ B) ∧
    (∀ C ∈ S, (relationship (2*n+2) A C = relationship (2*n+2) B C)) :=
  sorry

end social_relationships_theorem_l799_79977


namespace shopkeeper_profit_percentage_l799_79957

/-- Calculates the profit percentage given the sale price including tax, tax rate, and cost price. -/
def profit_percentage (sale_price_with_tax : ℚ) (tax_rate : ℚ) (cost_price : ℚ) : ℚ :=
  let sale_price := sale_price_with_tax / (1 + tax_rate)
  let profit := sale_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that under the given conditions, the profit percentage is approximately 4.54%. -/
theorem shopkeeper_profit_percentage :
  let sale_price_with_tax : ℚ := 616
  let tax_rate : ℚ := 1/10
  let cost_price : ℚ := 535.65
  abs (profit_percentage sale_price_with_tax tax_rate cost_price - 454/100) < 1/100 := by
  sorry

#eval profit_percentage 616 (1/10) 535.65

end shopkeeper_profit_percentage_l799_79957


namespace coprime_n_minus_two_and_n_squared_minus_n_minus_one_part2_solutions_part3_solution_l799_79966

-- Part 1
theorem coprime_n_minus_two_and_n_squared_minus_n_minus_one (n : ℕ) :
  Nat.gcd (n - 2) (n^2 - n - 1) = 1 :=
sorry

-- Part 2
def is_valid_solution_part2 (n m : ℕ) : Prop :=
  n^3 - 3*n^2 + n + 2 = 5^m

theorem part2_solutions :
  ∀ n m : ℕ, is_valid_solution_part2 n m ↔ (n = 3 ∧ m = 1) ∨ (n = 1 ∧ m = 0) :=
sorry

-- Part 3
def is_valid_solution_part3 (n m : ℕ) : Prop :=
  2*n^3 - n^2 + 2*n + 1 = 3^m

theorem part3_solution :
  ∀ n m : ℕ, is_valid_solution_part3 n m ↔ (n = 0 ∧ m = 0) :=
sorry

end coprime_n_minus_two_and_n_squared_minus_n_minus_one_part2_solutions_part3_solution_l799_79966


namespace teacher_age_l799_79962

theorem teacher_age (num_students : Nat) (student_avg_age : ℝ) (new_avg_age : ℝ) : 
  num_students = 30 →
  student_avg_age = 15 →
  new_avg_age = 16 →
  (num_students * student_avg_age + (num_students + 1) * new_avg_age - num_students * student_avg_age) = 46 := by
  sorry

end teacher_age_l799_79962


namespace river_depth_problem_l799_79933

theorem river_depth_problem (depth_may : ℝ) (depth_june : ℝ) (depth_july : ℝ) 
  (h1 : depth_june = depth_may + 10)
  (h2 : depth_july = 3 * depth_june)
  (h3 : depth_july = 45) :
  depth_may = 5 := by
sorry

end river_depth_problem_l799_79933


namespace multiplication_with_fraction_l799_79934

theorem multiplication_with_fraction : 8 * (1 / 7) * 14 = 16 := by
  sorry

end multiplication_with_fraction_l799_79934


namespace divisibility_by_eleven_l799_79974

theorem divisibility_by_eleven (m : ℕ+) (k : ℕ) (h : 33 ∣ m ^ k) : 11 ∣ m := by
  sorry

end divisibility_by_eleven_l799_79974


namespace police_can_catch_gangster_police_can_reach_same_side_l799_79930

/-- Represents the setup of the police and gangster problem -/
structure PoliceGangsterSetup where
  a : ℝ  -- side length of the square
  police_speed : ℝ  -- speed of the police officer
  gangster_speed : ℝ  -- speed of the gangster
  h_positive_a : 0 < a  -- side length is positive
  h_positive_police_speed : 0 < police_speed  -- police speed is positive
  h_gangster_speed : gangster_speed = 2.9 * police_speed  -- gangster speed is 2.9 times police speed

/-- Theorem stating that the police officer can always reach a side of the square before the gangster moves more than one side length -/
theorem police_can_catch_gangster (setup : PoliceGangsterSetup) :
  setup.a / (2 * setup.police_speed) < 1.45 * setup.a := by
  sorry

/-- Corollary stating that the police officer can always end up on the same side as the gangster -/
theorem police_can_reach_same_side (setup : PoliceGangsterSetup) :
  ∃ (t : ℝ), t > 0 ∧ t * setup.police_speed ≥ setup.a / 2 ∧ t * setup.gangster_speed < setup.a := by
  sorry

end police_can_catch_gangster_police_can_reach_same_side_l799_79930


namespace parking_lot_problem_l799_79989

theorem parking_lot_problem :
  let total_cars : ℝ := 300
  let valid_ticket_ratio : ℝ := 0.75
  let permanent_pass_ratio : ℝ := 0.2
  let unpaid_cars : ℝ := 30
  valid_ticket_ratio * total_cars +
  permanent_pass_ratio * (valid_ticket_ratio * total_cars) +
  unpaid_cars = total_cars :=
by sorry

end parking_lot_problem_l799_79989


namespace probability_three_green_marbles_l799_79905

/-- The probability of picking exactly k successes in n trials with probability p for each trial. -/
def binomialProbability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The number of green marbles -/
def greenMarbles : ℕ := 8

/-- The number of purple marbles -/
def purpleMarbles : ℕ := 7

/-- The total number of marbles -/
def totalMarbles : ℕ := greenMarbles + purpleMarbles

/-- The number of trials -/
def numTrials : ℕ := 7

/-- The number of green marbles we want to pick -/
def targetGreen : ℕ := 3

/-- The probability of picking a green marble in one trial -/
def probGreen : ℚ := greenMarbles / totalMarbles

theorem probability_three_green_marbles :
  binomialProbability numTrials targetGreen probGreen = 34454336 / 136687500 := by
  sorry

end probability_three_green_marbles_l799_79905


namespace merchandise_profit_rate_l799_79981

/-- Given a merchandise with cost price x, prove that the profit rate is 5% -/
theorem merchandise_profit_rate (x : ℝ) (h : 1.1 * x - 10 = 210) : 
  (210 - x) / x * 100 = 5 := by
  sorry

end merchandise_profit_rate_l799_79981


namespace exactly_two_false_l799_79984

-- Define the basic concepts
def Line : Type := sorry
def perpendicular (l1 l2 : Line) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def skew (l1 l2 : Line) : Prop := sorry
def intersects (l1 l2 : Line) : Prop := sorry

-- Define the statements
def statement1 : Prop := ∀ l1 l2 l3 : Line, perpendicular l1 l3 → perpendicular l2 l3 → parallel l1 l2
def statement2 : Prop := ∀ l1 l2 l3 : Line, parallel l1 l3 → parallel l2 l3 → parallel l1 l2
def statement3 : Prop := ∀ a b c : Line, parallel a b → perpendicular b c → perpendicular a c
def statement4 : Prop := ∀ a b l1 l2 : Line, skew a b → intersects l1 a → intersects l1 b → intersects l2 a → intersects l2 b → skew l1 l2

-- The theorem to prove
theorem exactly_two_false : 
  (¬statement1 ∧ statement2 ∧ statement3 ∧ ¬statement4) ∨
  (¬statement1 ∧ statement2 ∧ ¬statement3 ∧ ¬statement4) ∨
  (¬statement1 ∧ ¬statement2 ∧ statement3 ∧ ¬statement4) ∨
  (statement1 ∧ ¬statement2 ∧ statement3 ∧ ¬statement4) ∨
  (statement1 ∧ statement2 ∧ ¬statement3 ∧ ¬statement4) ∨
  (¬statement1 ∧ ¬statement2 ∧ statement3 ∧ statement4) :=
sorry

end exactly_two_false_l799_79984


namespace perfect_squares_between_powers_of_three_l799_79953

theorem perfect_squares_between_powers_of_three : 
  (Finset.range (Nat.succ (Nat.sqrt (3^10 + 3))) 
    |>.filter (λ n => n^2 ≥ 3^5 + 3 ∧ n^2 ≤ 3^10 + 3)).card = 228 := by
  sorry

end perfect_squares_between_powers_of_three_l799_79953


namespace paulas_aunt_money_l799_79978

/-- The amount of money Paula's aunt gave her -/
def aunt_money (shirt_price : ℕ) (num_shirts : ℕ) (pants_price : ℕ) (money_left : ℕ) : ℕ :=
  shirt_price * num_shirts + pants_price + money_left

/-- Theorem stating the total amount of money Paula's aunt gave her -/
theorem paulas_aunt_money :
  aunt_money 11 2 13 74 = 109 := by
  sorry

end paulas_aunt_money_l799_79978


namespace quadratic_inequality_solution_l799_79941

theorem quadratic_inequality_solution (c : ℝ) : 
  (∀ x : ℝ, -x^2 + 6*x + c < 0 ↔ x < 2 ∨ x > 4) → c = -8 := by
  sorry

end quadratic_inequality_solution_l799_79941


namespace min_frames_for_18x15_grid_l799_79902

/-- Represents a grid with given dimensions -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a square frame with side length 1 -/
structure Frame :=
  (side_length : ℕ := 1)

/-- Calculates the minimum number of frames needed to cover the grid -/
def min_frames_needed (g : Grid) : ℕ :=
  g.rows * g.cols - ((g.rows - 2) / 2 * (g.cols - 2))

/-- The theorem stating the minimum number of frames needed for an 18x15 grid -/
theorem min_frames_for_18x15_grid :
  let g : Grid := ⟨18, 15⟩
  min_frames_needed g = 166 := by
  sorry

#eval min_frames_needed ⟨18, 15⟩

end min_frames_for_18x15_grid_l799_79902


namespace problem_solution_l799_79922

theorem problem_solution (x : ℝ) (h : x + 1/x = 3) : 
  (x - 3)^2 + 36/((x - 3)^2) = 12 := by
  sorry

end problem_solution_l799_79922


namespace worker_times_relationship_l799_79921

/-- The time it takes for two workers to load a truck together -/
def combined_time : ℝ := 3.428571428571429

/-- The time it takes for the first worker to load a truck alone -/
def worker1_time : ℝ := 6

/-- The time it takes for the second worker to load a truck alone -/
def worker2_time : ℝ := 8

/-- Theorem stating the relationship between the workers' times -/
theorem worker_times_relationship : 
  1 / combined_time = 1 / worker1_time + 1 / worker2_time :=
sorry

end worker_times_relationship_l799_79921


namespace cone_prism_volume_ratio_l799_79965

/-- The ratio of the volume of a right circular cone inscribed in a right rectangular prism to the volume of the prism -/
theorem cone_prism_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (1 / 3 * π * r^2 * h) / (6 * r^2 * h) = π / 18 := by
  sorry


end cone_prism_volume_ratio_l799_79965


namespace inequality_theorem_l799_79943

theorem inequality_theorem (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n ≥ 1) :
  x + (n^n : ℝ) / x^n ≥ n + 1 := by sorry

end inequality_theorem_l799_79943


namespace subtraction_problem_l799_79969

theorem subtraction_problem (x : ℤ) : x - 29 = 63 → x - 47 = 45 := by
  sorry

end subtraction_problem_l799_79969


namespace coloring_book_problem_l799_79908

theorem coloring_book_problem (book1 : ℕ) (book2 : ℕ) (colored : ℕ) : 
  book1 = 23 → book2 = 32 → colored = 44 → 
  (book1 + book2) - colored = 11 := by
sorry

end coloring_book_problem_l799_79908


namespace total_third_grade_students_l799_79991

theorem total_third_grade_students : 
  let class_a : ℕ := 48
  let class_b : ℕ := 65
  let class_c : ℕ := 57
  let class_d : ℕ := 72
  class_a + class_b + class_c + class_d = 242 := by
sorry

end total_third_grade_students_l799_79991


namespace right_angled_parallelopiped_l799_79988

structure Parallelopiped where
  AB : ℝ
  AA' : ℝ

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def is_right_angled (M N P : Point) : Prop :=
  (M.x - N.x) * (P.x - N.x) + (M.y - N.y) * (P.y - N.y) + (M.z - N.z) * (P.z - N.z) = 0

theorem right_angled_parallelopiped (p : Parallelopiped) (N : Point) :
  p.AB = 12 * Real.sqrt 3 →
  p.AA' = 18 →
  N.x = 9 * Real.sqrt 3 ∧ N.y = 0 ∧ N.z = 0 →
  ∃ P : Point, P.x = 0 ∧ P.y = 0 ∧ P.z = 27 / 2 ∧
    ∀ M : Point, M.x = 12 * Real.sqrt 3 → M.z = 18 →
      is_right_angled M N P := by
  sorry

#check right_angled_parallelopiped

end right_angled_parallelopiped_l799_79988


namespace no_divisible_by_five_append_l799_79942

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Appends a digit to the left of 864 to form a four-digit number -/
def appendDigit (d : Digit) : Nat := d.val * 1000 + 864

/-- Theorem: There are no digits that can be appended to the left of 864
    to create a four-digit number divisible by 5 -/
theorem no_divisible_by_five_append :
  ∀ d : Digit, ¬(appendDigit d % 5 = 0) := by
  sorry

end no_divisible_by_five_append_l799_79942


namespace range_of_a_l799_79907

def has_real_roots (m : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁^2 - m*x₁ - 1 = 0 ∧ x₂^2 - m*x₂ - 1 = 0

def roots_difference_bound (a : ℝ) : Prop :=
  ∀ (m : ℝ), has_real_roots m → 
    ∃ (x₁ x₂ : ℝ), x₁^2 - m*x₁ - 1 = 0 ∧ x₂^2 - m*x₂ - 1 = 0 ∧ a^2 + 4*a - 3 ≤ |x₁ - x₂|

def quadratic_has_solution (a : ℝ) : Prop :=
  ∃ (x : ℝ), x^2 + 2*x + a < 0

def proposition_p (a : ℝ) : Prop :=
  has_real_roots 0 ∧ roots_difference_bound a

def proposition_q (a : ℝ) : Prop :=
  quadratic_has_solution a

theorem range_of_a :
  ∀ (a : ℝ), (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a) →
    a = 1 ∨ a < -5 :=
by sorry

end range_of_a_l799_79907


namespace intersection_A_B_range_of_a_l799_79970

-- Define the sets A, B, and C
def A : Set ℝ := {x | (x + 3) / (x - 1) ≥ 0}
def B : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def C (a : ℝ) : Set ℝ := {x | x ≥ a^2 - 2}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

-- Theorem for the range of a when B ∪ C = C
theorem range_of_a (a : ℝ) (h : B ∪ C a = C a) : a ∈ Set.Icc (-1) 1 := by sorry

end intersection_A_B_range_of_a_l799_79970


namespace min_value_theorem_l799_79992

theorem min_value_theorem (x y : ℝ) (h1 : x + y = 4) (h2 : x > y) (h3 : y > 0) :
  (∀ a b : ℝ, a + b = 4 → a > b → b > 0 → (2 / (a - b) + 1 / b) ≥ 2) ∧ 
  (∃ a b : ℝ, a + b = 4 ∧ a > b ∧ b > 0 ∧ 2 / (a - b) + 1 / b = 2) := by
sorry

end min_value_theorem_l799_79992


namespace kaleb_toy_purchase_l799_79944

/-- Represents the problem of calculating how many toys Kaleb can buy -/
theorem kaleb_toy_purchase (saved : ℝ) (new_allowance : ℝ) (allowance_increase : ℝ) 
  (toy_cost : ℝ) : 
  saved = 21 → 
  new_allowance = 15 → 
  allowance_increase = 0.2 →
  toy_cost = 6 →
  (((saved + new_allowance) / 2) / toy_cost : ℝ) = 3 := by
  sorry

#check kaleb_toy_purchase

end kaleb_toy_purchase_l799_79944


namespace tangent_line_slope_l799_79937

/-- Given a curve y = x³ + ax + b and a line y = kx + 1 tangent to the curve at point (l, 3),
    prove that k = 2. -/
theorem tangent_line_slope (a b l : ℝ) : 
  (∃ k : ℝ, (3 = l^3 + a*l + b) ∧ (3 = k*l + 1) ∧ 
   (∀ x : ℝ, k*x + 1 ≤ x^3 + a*x + b) ∧
   (∃ x : ℝ, x ≠ l ∧ k*x + 1 < x^3 + a*x + b)) →
  (∃ k : ℝ, k = 2 ∧ (3 = k*l + 1)) := by
sorry

end tangent_line_slope_l799_79937


namespace radio_cost_price_l799_79903

/-- 
Given a radio sold for Rs. 1330 with a 30% loss, 
prove that the original cost price was Rs. 1900.
-/
theorem radio_cost_price (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1330)
  (h2 : loss_percentage = 30) : 
  (selling_price / (1 - loss_percentage / 100)) = 1900 := by
  sorry

end radio_cost_price_l799_79903


namespace euclidean_algorithm_steps_bound_l799_79955

/-- The number of steps in the Euclidean algorithm for (a, b) -/
def euclidean_steps (a b : ℕ) : ℕ := sorry

/-- The number of digits in the decimal representation of a natural number -/
def decimal_digits (n : ℕ) : ℕ := sorry

theorem euclidean_algorithm_steps_bound (a b : ℕ) (h : a > b) :
  euclidean_steps a b ≤ 5 * decimal_digits b := by sorry

end euclidean_algorithm_steps_bound_l799_79955


namespace chocolate_bars_in_large_box_l799_79999

theorem chocolate_bars_in_large_box :
  let small_boxes : ℕ := 17
  let bars_per_small_box : ℕ := 26
  let total_bars : ℕ := small_boxes * bars_per_small_box
  total_bars = 442 := by
sorry

end chocolate_bars_in_large_box_l799_79999


namespace marias_trip_distance_l799_79959

theorem marias_trip_distance :
  ∀ (D : ℝ),
  (D / 2) / 4 + 180 = D / 2 →
  D = 480 :=
by
  sorry

end marias_trip_distance_l799_79959


namespace f_2_eq_1_l799_79911

/-- The function f(x) = x^5 - 5x^4 + 10x^3 - 10x^2 + 5x - 1 -/
def f (x : ℝ) : ℝ := x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - 1

/-- Theorem: f(2) = 1 -/
theorem f_2_eq_1 : f 2 = 1 := by
  sorry

end f_2_eq_1_l799_79911


namespace quadratic_inequality_range_l799_79940

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := by
sorry

end quadratic_inequality_range_l799_79940


namespace trigonometric_expression_equality_trigonometric_fraction_simplification_l799_79928

-- Problem 1
theorem trigonometric_expression_equality : 
  Real.cos (2/3 * Real.pi) - Real.tan (-Real.pi/4) + 3/4 * Real.tan (Real.pi/6) - Real.sin (-31/6 * Real.pi) = Real.sqrt 3 / 4 := by
  sorry

-- Problem 2
theorem trigonometric_fraction_simplification (α : Real) : 
  (Real.sin (Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.cos (-α + 3/2 * Real.pi)) / 
  (Real.cos (Real.pi/2 - α) * Real.sin (-Real.pi - α)) = -Real.cos α := by
  sorry

end trigonometric_expression_equality_trigonometric_fraction_simplification_l799_79928


namespace sum_integer_part_l799_79993

theorem sum_integer_part : ⌊(2010 : ℝ) / 1000 + (1219 : ℝ) / 100 + (27 : ℝ) / 10⌋ = 16 := by
  sorry

end sum_integer_part_l799_79993


namespace arccos_negative_one_l799_79956

theorem arccos_negative_one : Real.arccos (-1) = Real.pi := by
  sorry

end arccos_negative_one_l799_79956


namespace quadratic_equation_solution_trigonometric_expression_value_l799_79912

-- Part 1: Quadratic equation
theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 4*x + 3
  ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 1 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

-- Part 2: Trigonometric expression
theorem trigonometric_expression_value :
  4 * Real.sin (π/6) - Real.sqrt 2 * Real.cos (π/4) + Real.sqrt 3 * Real.tan (π/3) = 4 :=
by sorry

end quadratic_equation_solution_trigonometric_expression_value_l799_79912


namespace pascal_contest_certificates_l799_79987

theorem pascal_contest_certificates 
  (num_boys : ℕ) 
  (num_girls : ℕ) 
  (percent_boys_cert : ℚ) 
  (percent_girls_cert : ℚ) 
  (h1 : num_boys = 30)
  (h2 : num_girls = 20)
  (h3 : percent_boys_cert = 30 / 100)
  (h4 : percent_girls_cert = 40 / 100) :
  (num_boys * percent_boys_cert + num_girls * percent_girls_cert) / (num_boys + num_girls) = 34 / 100 := by
sorry


end pascal_contest_certificates_l799_79987


namespace elise_comic_book_cost_l799_79995

/-- Calculates the amount spent on a comic book given initial money, saved money, puzzle cost, and final money --/
def comic_book_cost (initial_money saved_money puzzle_cost final_money : ℕ) : ℕ :=
  initial_money + saved_money - puzzle_cost - final_money

/-- Proves that Elise spent $2 on the comic book --/
theorem elise_comic_book_cost :
  comic_book_cost 8 13 18 1 = 2 := by
  sorry

end elise_comic_book_cost_l799_79995


namespace robin_bracelet_cost_l799_79985

/-- Represents the types of bracelets available --/
inductive BraceletType
| Plastic
| Metal
| Beaded

/-- Represents a friend and their bracelet preference --/
structure Friend where
  name : String
  preference : List BraceletType

/-- Calculates the cost of a single bracelet --/
def braceletCost (type : BraceletType) : ℚ :=
  match type with
  | BraceletType.Plastic => 2
  | BraceletType.Metal => 3
  | BraceletType.Beaded => 5

/-- Calculates the total cost for a friend's bracelets --/
def friendCost (friend : Friend) : ℚ :=
  let numBracelets := friend.name.length
  let preferredTypes := friend.preference
  let costs := preferredTypes.map braceletCost
  let totalCost := costs.sum * numBracelets / preferredTypes.length
  totalCost

/-- Applies discount if applicable --/
def applyDiscount (total : ℚ) (numBracelets : ℕ) : ℚ :=
  if numBracelets ≥ 10 then total * (1 - 0.1) else total

/-- Applies sales tax --/
def applySalesTax (total : ℚ) : ℚ :=
  total * (1 + 0.07)

/-- The main theorem to prove --/
theorem robin_bracelet_cost : 
  let friends : List Friend := [
    ⟨"Jessica", [BraceletType.Plastic]⟩,
    ⟨"Tori", [BraceletType.Metal]⟩,
    ⟨"Lily", [BraceletType.Beaded]⟩,
    ⟨"Patrice", [BraceletType.Metal, BraceletType.Beaded]⟩
  ]
  let totalCost := friends.map friendCost |>.sum
  let numBracelets := friends.map (fun f => f.name.length) |>.sum
  let discountedCost := applyDiscount totalCost numBracelets
  let finalCost := applySalesTax discountedCost
  finalCost = 7223/100 := by
  sorry

end robin_bracelet_cost_l799_79985


namespace complement_M_union_N_eq_nonneg_reals_l799_79948

-- Define the set of real numbers
variable (r : Set ℝ)

-- Define set M
def M : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (1 - 2/x)}

-- Define set N
def N : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x - 1)}

-- Statement to prove
theorem complement_M_union_N_eq_nonneg_reals :
  (Set.univ \ M) ∪ N = Set.Ici (0 : ℝ) := by sorry

end complement_M_union_N_eq_nonneg_reals_l799_79948


namespace first_day_over_200_paperclips_l799_79938

def paperclip_count (n : ℕ) : ℕ :=
  if n < 2 then 3 else 3 * 2^(n - 2)

theorem first_day_over_200_paperclips :
  ∀ n : ℕ, n < 9 → paperclip_count n ≤ 200 ∧
  paperclip_count 9 > 200 :=
sorry

end first_day_over_200_paperclips_l799_79938


namespace solution_to_inequality_system_l799_79932

theorem solution_to_inequality_system (x : ℝ) :
  (x + 3 > 2 ∧ 1 - 2*x < -3) → x = 3 :=
by sorry

end solution_to_inequality_system_l799_79932


namespace circle_equation_l799_79963

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangent line
def tangentLine (x : ℝ) : ℝ := 2 * x + 1

-- Define the properties of the circle
def circleProperties (c : Circle) : Prop :=
  -- The center is on the x-axis
  c.center.2 = 0 ∧
  -- The circle is tangent to the line y = 2x + 1 at point (0, 1)
  c.radius^2 = c.center.1^2 + 1 ∧
  -- The tangent line is perpendicular to the radius at the point of tangency
  2 * c.center.1 + c.center.2 - 1 = 0

-- Theorem statement
theorem circle_equation (c : Circle) (h : circleProperties c) :
  ∀ (x y : ℝ), (x - 2)^2 + y^2 = 5 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 :=
sorry

end circle_equation_l799_79963


namespace perpendicular_angles_counterexample_l799_79971

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents an angle in 3D space -/
structure Angle3D where
  vertex : Point3D
  side1 : Point3D
  side2 : Point3D

/-- Checks if two line segments are perpendicular in 3D space -/
def isPerpendicular (a b c d : Point3D) : Prop := sorry

/-- Calculates the measure of an angle in degrees -/
def angleMeasure (angle : Angle3D) : ℝ := sorry

/-- Theorem: There exist angles with perpendicular sides that are neither equal nor sum to 180° -/
theorem perpendicular_angles_counterexample :
  ∃ (α β : Angle3D),
    isPerpendicular α.vertex α.side1 β.vertex β.side1 ∧
    isPerpendicular α.vertex α.side2 β.vertex β.side2 ∧
    angleMeasure α ≠ angleMeasure β ∧
    angleMeasure α + angleMeasure β ≠ 180 := by
  sorry

end perpendicular_angles_counterexample_l799_79971


namespace joe_trading_cards_l799_79947

theorem joe_trading_cards (cards_per_box : ℕ) (num_boxes : ℕ) (h1 : cards_per_box = 8) (h2 : num_boxes = 11) :
  cards_per_box * num_boxes = 88 := by
sorry

end joe_trading_cards_l799_79947


namespace expression_evaluation_l799_79967

theorem expression_evaluation : -20 + 12 * (8 / 4) - 5 = -1 := by
  sorry

end expression_evaluation_l799_79967


namespace one_is_monomial_l799_79919

/-- A monomial is an algebraic expression with only one term. -/
def IsMonomial (expr : ℕ) : Prop :=
  expr = 1 ∨ ∃ (base : ℕ) (exponent : ℕ), expr = base ^ exponent

/-- Theorem stating that 1 is a monomial. -/
theorem one_is_monomial : IsMonomial 1 := by sorry

end one_is_monomial_l799_79919


namespace parallelogram_area_l799_79996

/-- The area of a parallelogram with base 30 cm and height 12 cm is 360 square centimeters. -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 30 → 
  height = 12 → 
  area = base * height →
  area = 360 :=
by sorry

end parallelogram_area_l799_79996


namespace coplanar_vectors_m_l799_79972

/-- Three vectors in ℝ³ are coplanar if and only if their scalar triple product is zero -/
def coplanar (a b c : ℝ × ℝ × ℝ) : Prop :=
  let (a₁, a₂, a₃) := a
  let (b₁, b₂, b₃) := b
  let (c₁, c₂, c₃) := c
  a₁ * (b₂ * c₃ - b₃ * c₂) - a₂ * (b₁ * c₃ - b₃ * c₁) + a₃ * (b₁ * c₂ - b₂ * c₁) = 0

theorem coplanar_vectors_m (m : ℝ) : 
  coplanar (1, -1, 0) (-1, 2, 1) (2, 1, m) → m = 3 := by
  sorry

end coplanar_vectors_m_l799_79972


namespace chessboard_covering_impossibility_l799_79914

/-- Represents a chessboard -/
structure Chessboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a domino -/
inductive Domino
  | TwoByTwo
  | OneByFour

/-- Represents a set of dominoes -/
def DominoSet := List Domino

/-- A function to check if a set of dominoes can cover a chessboard -/
def can_cover (board : Chessboard) (dominoes : DominoSet) : Prop :=
  sorry

/-- A function to replace one 2x2 domino with a 1x4 domino in a set -/
def replace_one_domino (dominoes : DominoSet) : DominoSet :=
  sorry

theorem chessboard_covering_impossibility (board : Chessboard) (original_dominoes : DominoSet) :
  board.rows = 2007 →
  board.cols = 2008 →
  can_cover board original_dominoes →
  ¬(can_cover board (replace_one_domino original_dominoes)) :=
  sorry

end chessboard_covering_impossibility_l799_79914


namespace problem_one_problem_two_l799_79982

-- Problem 1
theorem problem_one (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := by
  sorry

-- Problem 2
theorem problem_two (a b c x y z : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^x = b^y) (h5 : b^y = c^z)
  (h6 : 1/x + 1/y + 1/z = 0) : a * b * c = 1 := by
  sorry

end problem_one_problem_two_l799_79982


namespace johny_journey_distance_johny_journey_specific_distance_l799_79913

/-- Calculates the total distance of Johny's journey given his travel pattern. -/
theorem johny_journey_distance : ℕ → ℕ → ℕ
  | south_distance, east_extra_distance =>
    let east_distance := south_distance + east_extra_distance
    let north_distance := 2 * east_distance
    south_distance + east_distance + north_distance

/-- Proves that Johny's journey distance is 220 miles given the specific conditions. -/
theorem johny_journey_specific_distance :
  johny_journey_distance 40 20 = 220 := by
  sorry

end johny_journey_distance_johny_journey_specific_distance_l799_79913


namespace pizza_recipe_l799_79964

theorem pizza_recipe (water flour salt : ℚ) : 
  water = 10 ∧ 
  salt = (1/2) * flour ∧ 
  water + flour + salt = 34 →
  flour = 16 := by
sorry

end pizza_recipe_l799_79964


namespace region_characterization_l799_79918

def f (x : ℝ) : ℝ := x^2 - 6*x + 5

theorem region_characterization (x y : ℝ) :
  f x + f y ≤ 0 ∧ f x - f y ≥ 0 →
  (x - 3)^2 + (y - 3)^2 ≤ 8 ∧ (x - y)*(x + y - 6) ≥ 0 := by
  sorry

end region_characterization_l799_79918


namespace jessica_non_work_days_l799_79910

/-- Calculates the number of non-work days given the problem conditions -/
theorem jessica_non_work_days 
  (total_days : ℕ) 
  (full_day_earnings : ℚ) 
  (non_work_deduction : ℚ) 
  (half_days : ℕ) 
  (total_earnings : ℚ) 
  (h1 : total_days = 30)
  (h2 : full_day_earnings = 80)
  (h3 : non_work_deduction = 40)
  (h4 : half_days = 5)
  (h5 : total_earnings = 1600) :
  ∃ (non_work_days : ℕ), 
    non_work_days = 5 ∧ 
    (total_days : ℚ) = (non_work_days : ℚ) + (half_days : ℚ) + 
      ((total_earnings + non_work_deduction * (non_work_days : ℚ) - 
        (half_days : ℚ) * full_day_earnings / 2) / full_day_earnings) :=
by sorry

end jessica_non_work_days_l799_79910


namespace intersection_of_A_and_B_l799_79954

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end intersection_of_A_and_B_l799_79954


namespace exactly_four_sets_l799_79906

-- Define the set S1 as {-1, 0, 1}
def S1 : Set Int := {-1, 0, 1}

-- Define the set S2 as {-2, 0, 2}
def S2 : Set Int := {-2, 0, 2}

-- Define the set R as {-2, 0, 1, 2}
def R : Set Int := {-2, 0, 1, 2}

-- Define the conditions for set A
def satisfiesConditions (A : Set Int) : Prop :=
  (A ∩ S1 = {0, 1}) ∧ (A ∪ S2 = R)

-- Theorem stating that there are exactly 4 sets satisfying the conditions
theorem exactly_four_sets :
  ∃! (s : Finset (Set Int)), (∀ A ∈ s, satisfiesConditions A) ∧ s.card = 4 := by
  sorry

end exactly_four_sets_l799_79906


namespace range_of_m_l799_79952

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 6*x - 16

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ∈ Set.Icc (-25) (-16)) ∧
  (∀ y ∈ Set.Icc (-25) (-16), ∃ x ∈ Set.Icc 0 m, f x = y) →
  m ∈ Set.Icc 3 6 :=
sorry

end range_of_m_l799_79952


namespace negation_existence_lt_one_l799_79935

theorem negation_existence_lt_one :
  (¬ ∃ x : ℝ, x < 1) ↔ (∀ x : ℝ, x ≥ 1) := by sorry

end negation_existence_lt_one_l799_79935


namespace y_days_to_finish_work_l799_79950

/-- The number of days x needs to finish the work alone -/
def x_days : ℕ := 36

/-- The number of days y worked before leaving -/
def y_worked : ℕ := 12

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining : ℕ := 18

/-- The work rate of x (portion of work completed per day) -/
def x_rate : ℚ := 1 / x_days

/-- The total amount of work to be done -/
def total_work : ℚ := 1

/-- The amount of work completed by x after y left -/
def x_completed : ℚ := x_rate * x_remaining

theorem y_days_to_finish_work : ℕ := by
  sorry

end y_days_to_finish_work_l799_79950


namespace parabola_circle_tangent_radius_l799_79979

/-- The radius of a circle that is tangent to the parabola y = 1/4 * x^2 at a point where the tangent line to the parabola is also tangent to the circle. -/
theorem parabola_circle_tangent_radius : ∃ (r : ℝ) (P : ℝ × ℝ),
  r > 0 ∧
  (P.2 = (1/4) * P.1^2) ∧
  ((P.1 - 1)^2 + (P.2 - 2)^2 = r^2) ∧
  (∃ (m : ℝ), (∀ (x y : ℝ), y - P.2 = m * (x - P.1) → 
    y = (1/4) * x^2 ∨ (x - 1)^2 + (y - 2)^2 = r^2)) →
  r = Real.sqrt 2 := by
sorry

end parabola_circle_tangent_radius_l799_79979
