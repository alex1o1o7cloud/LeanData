import Mathlib

namespace exchange_probability_l2584_258490

/-- Represents the colors of balls -/
inductive Color
  | Red | Green | Yellow | Violet | Black | Orange

/-- Represents a bag of balls -/
def Bag := List Color

/-- Initial configuration of Arjun's bag -/
def arjunInitialBag : Bag :=
  [Color.Red, Color.Red, Color.Green, Color.Yellow, Color.Violet]

/-- Initial configuration of Becca's bag -/
def beccaInitialBag : Bag :=
  [Color.Black, Color.Black, Color.Orange]

/-- Represents the exchange process -/
def exchange (bag1 bag2 : Bag) : Bag × Bag :=
  sorry

/-- Checks if a bag has exactly 3 different colors -/
def hasThreeColors (bag : Bag) : Bool :=
  sorry

/-- Calculates the probability of the final configuration -/
def finalProbability (arjunBag beccaBag : Bag) : ℚ :=
  sorry

/-- The main theorem to be proved -/
theorem exchange_probability :
  finalProbability arjunInitialBag beccaInitialBag = 3/10 :=
sorry

end exchange_probability_l2584_258490


namespace hat_promotion_savings_l2584_258411

/-- Calculates the percentage saved when buying three hats under a promotional offer --/
theorem hat_promotion_savings : 
  let regular_price : ℝ := 60
  let discount_second : ℝ := 0.25
  let discount_third : ℝ := 0.35
  let total_regular : ℝ := 3 * regular_price
  let price_first : ℝ := regular_price
  let price_second : ℝ := regular_price * (1 - discount_second)
  let price_third : ℝ := regular_price * (1 - discount_third)
  let total_discounted : ℝ := price_first + price_second + price_third
  let savings : ℝ := total_regular - total_discounted
  let percentage_saved : ℝ := (savings / total_regular) * 100
  percentage_saved = 20 := by
  sorry


end hat_promotion_savings_l2584_258411


namespace longer_string_length_l2584_258472

theorem longer_string_length 
  (total_length : ℕ) 
  (difference : ℕ) 
  (h1 : total_length = 348) 
  (h2 : difference = 72) : 
  ∃ (longer shorter : ℕ), 
    longer + shorter = total_length ∧ 
    longer - shorter = difference ∧ 
    longer = 210 := by
  sorry

end longer_string_length_l2584_258472


namespace cubic_sum_zero_l2584_258407

theorem cubic_sum_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_sum : (a^2 / (b - c)^2) + (b^2 / (c - a)^2) + (c^2 / (a - b)^2) = 0) :
  (a^3 / (b - c)^3) + (b^3 / (c - a)^3) + (c^3 / (a - b)^3) = 0 := by
  sorry

end cubic_sum_zero_l2584_258407


namespace range_of_x2_plus_y2_l2584_258438

theorem range_of_x2_plus_y2 (x y : ℝ) (h : (x + 2)^2 + y^2/4 = 1) :
  ∃ (min max : ℝ), min = 1 ∧ max = 28/3 ∧
  (x^2 + y^2 ≥ min ∧ x^2 + y^2 ≤ max) ∧
  (∀ z, (∃ a b : ℝ, (a + 2)^2 + b^2/4 = 1 ∧ z = a^2 + b^2) → z ≥ min ∧ z ≤ max) :=
sorry

end range_of_x2_plus_y2_l2584_258438


namespace C_power_50_l2584_258423

def C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; -4, -1]

theorem C_power_50 : C^50 = !![101, 50; -200, -99] := by sorry

end C_power_50_l2584_258423


namespace min_tenuous_g7_l2584_258403

/-- A tenuous function is an integer-valued function g such that
    g(x) + g(y) > x^2 for all positive integers x and y. -/
def Tenuous (g : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, g x + g y > (x : ℤ)^2

/-- The sum of g(1) to g(10) for a function g. -/
def SumG (g : ℕ+ → ℤ) : ℤ :=
  (Finset.range 10).sum (fun i => g ⟨i + 1, Nat.succ_pos i⟩)

/-- A tenuous function g that minimizes the sum of g(1) to g(10). -/
def MinTenuous (g : ℕ+ → ℤ) : Prop :=
  Tenuous g ∧ ∀ h : ℕ+ → ℤ, Tenuous h → SumG g ≤ SumG h

theorem min_tenuous_g7 (g : ℕ+ → ℤ) (hg : MinTenuous g) : g ⟨7, by norm_num⟩ ≥ 49 := by
  sorry

end min_tenuous_g7_l2584_258403


namespace complex_subtraction_l2584_258426

theorem complex_subtraction (i : ℂ) (h : i * i = -1) :
  let z₁ : ℂ := 3 + 4 * i
  let z₂ : ℂ := 1 + 2 * i
  z₁ - z₂ = 2 + 2 * i :=
by sorry

end complex_subtraction_l2584_258426


namespace stone_bucket_probability_l2584_258443

/-- The probability of having exactly k stones in the bucket after n seconds -/
def f (n k : ℕ) : ℚ :=
  (↑(Nat.floor ((n - k : ℤ) / 2)) : ℚ) / 2^n

/-- The main theorem stating the probability of having 1337 stones after 2017 seconds -/
theorem stone_bucket_probability : f 2017 1337 = 340 / 2^2017 := by sorry

end stone_bucket_probability_l2584_258443


namespace arrangements_theorem_l2584_258421

-- Define the number of officers and intersections
def num_officers : ℕ := 5
def num_intersections : ℕ := 3

-- Define the function to calculate the number of arrangements
def arrangements_with_AB_together : ℕ := sorry

-- State the theorem
theorem arrangements_theorem : arrangements_with_AB_together = 36 := by sorry

end arrangements_theorem_l2584_258421


namespace kramers_packing_rate_l2584_258465

/-- Kramer's cigarette packing rate -/
theorem kramers_packing_rate 
  (boxes_per_case : ℕ) 
  (cases_packed : ℕ) 
  (packing_time_hours : ℕ) 
  (h1 : boxes_per_case = 5)
  (h2 : cases_packed = 240)
  (h3 : packing_time_hours = 2) :
  (boxes_per_case * cases_packed) / (packing_time_hours * 60) = 10 := by
  sorry

#check kramers_packing_rate

end kramers_packing_rate_l2584_258465


namespace line_through_point_with_opposite_intercepts_l2584_258422

theorem line_through_point_with_opposite_intercepts :
  ∃ (m c : ℝ), 
    (∀ x y : ℝ, y = m * x + c ↔ 
      (x = 1 ∧ y = 3) ∨ 
      (∃ a : ℝ, (x = a ∧ y = 0) ∨ (x = 0 ∧ y = -a))) →
    m = 1 ∧ c = 2 := by
  sorry

end line_through_point_with_opposite_intercepts_l2584_258422


namespace mike_changes_64_tires_l2584_258424

/-- The number of tires on a motorcycle -/
def motorcycle_tires : ℕ := 2

/-- The number of tires on a car -/
def car_tires : ℕ := 4

/-- The number of motorcycles Mike changes tires on -/
def num_motorcycles : ℕ := 12

/-- The number of cars Mike changes tires on -/
def num_cars : ℕ := 10

/-- The total number of tires Mike changes -/
def total_tires : ℕ := num_motorcycles * motorcycle_tires + num_cars * car_tires

theorem mike_changes_64_tires : total_tires = 64 := by
  sorry

end mike_changes_64_tires_l2584_258424


namespace set_union_problem_l2584_258488

theorem set_union_problem (M N : Set ℕ) (x : ℕ) :
  M = {0, x} →
  N = {1, 2} →
  M ∩ N = {2} →
  M ∪ N = {0, 1, 2} := by
sorry

end set_union_problem_l2584_258488


namespace gcd_polynomial_and_multiple_l2584_258432

theorem gcd_polynomial_and_multiple (y : ℤ) : 
  18090 ∣ y → 
  Int.gcd ((3*y + 5)*(6*y + 7)*(10*y + 3)*(5*y + 11)*(y + 7)) y = 8085 := by
  sorry

end gcd_polynomial_and_multiple_l2584_258432


namespace circle_diameter_ratio_l2584_258495

theorem circle_diameter_ratio (D C : ℝ → Prop) (r_D r_C : ℝ) : 
  (∀ x, C x → D x) →  -- C is inside D
  (2 * r_D = 20) →    -- Diameter of D is 20 cm
  (π * r_D^2 - π * r_C^2 = 2 * π * r_C^2) →  -- Ratio of shaded area to area of C is 2:1
  2 * r_C = 20 * Real.sqrt 3 / 3 :=
by sorry

end circle_diameter_ratio_l2584_258495


namespace expression_value_l2584_258444

theorem expression_value (p q : ℝ) : 
  (∃ x : ℝ, x = 3 ∧ p * x^3 + q * x - 1 = 13) → 
  (∃ y : ℝ, y = -3 ∧ p * y^3 + q * y - 1 = -15) :=
sorry

end expression_value_l2584_258444


namespace least_four_digit_multiple_l2584_258405

theorem least_four_digit_multiple : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m → n ≤ m) ∧
  n = 1050 :=
by sorry

end least_four_digit_multiple_l2584_258405


namespace total_marbles_l2584_258477

/-- The total number of marbles for three people given specific conditions -/
theorem total_marbles (my_marbles : ℕ) (brother_marbles : ℕ) (friend_marbles : ℕ) 
  (h1 : my_marbles = 16)
  (h2 : my_marbles - 2 = 2 * (brother_marbles + 2))
  (h3 : friend_marbles = 3 * (my_marbles - 2)) :
  my_marbles + brother_marbles + friend_marbles = 63 := by
  sorry

end total_marbles_l2584_258477


namespace lcm_of_308_and_275_l2584_258420

theorem lcm_of_308_and_275 :
  let a := 308
  let b := 275
  let hcf := 11
  let lcm := Nat.lcm a b
  (Nat.gcd a b = hcf) → (lcm = 7700) := by
sorry

end lcm_of_308_and_275_l2584_258420


namespace indefinite_integral_sin_3x_l2584_258494

theorem indefinite_integral_sin_3x (x : ℝ) :
  (deriv (fun x => -1/3 * (x + 5) * Real.cos (3 * x) + 1/9 * Real.sin (3 * x))) x
  = (x + 5) * Real.sin (3 * x) := by
  sorry

end indefinite_integral_sin_3x_l2584_258494


namespace negation_of_universal_proposition_negation_of_sqrt_proposition_l2584_258406

theorem negation_of_universal_proposition (p : ℝ → Prop) :
  (¬ ∀ x ≤ 0, p x) ↔ (∃ x₀ ≤ 0, ¬ p x₀) := by sorry

-- Define the specific proposition
def sqrt_prop (x : ℝ) : Prop := Real.sqrt (x^2) = -x

-- Main theorem
theorem negation_of_sqrt_proposition :
  (¬ ∀ x ≤ 0, sqrt_prop x) ↔ (∃ x₀ ≤ 0, ¬ sqrt_prop x₀) :=
negation_of_universal_proposition sqrt_prop

end negation_of_universal_proposition_negation_of_sqrt_proposition_l2584_258406


namespace work_completion_time_l2584_258445

/-- Represents the amount of work one man can do in one day -/
def man_work : ℝ := sorry

/-- Represents the amount of work one boy can do in one day -/
def boy_work : ℝ := sorry

/-- The number of days it takes 6 men and 8 boys to complete the work -/
def x : ℝ := sorry

theorem work_completion_time :
  (6 * man_work + 8 * boy_work) * x = (26 * man_work + 48 * boy_work) * 2 ∧
  (6 * man_work + 8 * boy_work) * x = (15 * man_work + 20 * boy_work) * 4 →
  x = 5 := by sorry

end work_completion_time_l2584_258445


namespace sculpture_cost_in_cny_l2584_258418

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℝ := 5

/-- Exchange rate from US dollars to Chinese yuan -/
def usd_to_cny : ℝ := 8

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℝ := 200

/-- Theorem stating the cost of the sculpture in Chinese yuan -/
theorem sculpture_cost_in_cny :
  (sculpture_cost_nad / usd_to_nad) * usd_to_cny = 320 := by
  sorry

end sculpture_cost_in_cny_l2584_258418


namespace claire_crafting_hours_l2584_258469

def total_hours : ℕ := 24
def cleaning_hours : ℕ := 4
def cooking_hours : ℕ := 2
def sleeping_hours : ℕ := 8

def remaining_hours : ℕ := total_hours - (cleaning_hours + cooking_hours + sleeping_hours)

theorem claire_crafting_hours : remaining_hours / 2 = 5 := by
  sorry

end claire_crafting_hours_l2584_258469


namespace number_exists_l2584_258476

theorem number_exists : ∃ x : ℝ, (2/3 * x)^3 - 10 = 14 := by
  sorry

end number_exists_l2584_258476


namespace smallest_prime_with_digit_sum_23_l2584_258489

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- Theorem: 599 is the smallest prime number whose digits sum to 23 -/
theorem smallest_prime_with_digit_sum_23 :
  (is_prime 599) ∧ 
  (digit_sum 599 = 23) ∧ 
  (∀ n : ℕ, n < 599 → ¬(is_prime n ∧ digit_sum n = 23)) := by sorry

end smallest_prime_with_digit_sum_23_l2584_258489


namespace percentage_of_b_l2584_258414

theorem percentage_of_b (a b c : ℝ) (h1 : 12 = 0.04 * a) (h2 : ∃ p, p * b = 4) (h3 : c = b / a) :
  ∃ p, p * b = 4 ∧ p = 4 / (c * 300) := by
  sorry

end percentage_of_b_l2584_258414


namespace gas_cost_equation_l2584_258458

/-- The total cost of gas for a trip satisfies the given equation based on the change in cost per person when additional friends join. -/
theorem gas_cost_equation (x : ℝ) : x > 0 → (x / 5) - (x / 8) = 15.50 := by
  sorry

end gas_cost_equation_l2584_258458


namespace a_values_theorem_l2584_258461

theorem a_values_theorem (a b x : ℝ) (h1 : a - b = x) (h2 : x ≠ 0) (h3 : a^3 - b^3 = 19*x^3) :
  a = 3*x ∨ a = -2*x :=
by sorry

end a_values_theorem_l2584_258461


namespace x_squared_plus_x_minus_one_zero_l2584_258473

theorem x_squared_plus_x_minus_one_zero (x : ℝ) :
  x^2 + x - 1 = 0 → x^4 + 2*x^3 - 3*x^2 - 4*x + 5 = 2 := by
  sorry

end x_squared_plus_x_minus_one_zero_l2584_258473


namespace papi_calot_plants_l2584_258455

/-- The number of plants Papi Calot needs to buy for his potato garden. -/
def total_plants (rows : ℕ) (plants_per_row : ℕ) (additional_plants : ℕ) : ℕ :=
  rows * plants_per_row + additional_plants

/-- Theorem stating the total number of plants Papi Calot needs to buy. -/
theorem papi_calot_plants : total_plants 7 18 15 = 141 := by
  sorry

end papi_calot_plants_l2584_258455


namespace garden_border_rocks_l2584_258441

theorem garden_border_rocks (rocks_placed : Float) (additional_rocks : Float) : 
  rocks_placed = 125.0 → additional_rocks = 64.0 → rocks_placed + additional_rocks = 189.0 := by
  sorry

end garden_border_rocks_l2584_258441


namespace box_weights_sum_l2584_258498

theorem box_weights_sum (heavy_box light_box sum : ℚ) : 
  heavy_box = 14/15 → 
  light_box = heavy_box - 1/10 → 
  sum = heavy_box + light_box → 
  sum = 53/30 := by sorry

end box_weights_sum_l2584_258498


namespace egg_laying_hens_l2584_258485

/-- Calculates the number of egg-laying hens on Mr. Curtis's farm -/
theorem egg_laying_hens (total_chickens roosters non_laying_hens : ℕ) 
  (h1 : total_chickens = 325)
  (h2 : roosters = 28)
  (h3 : non_laying_hens = 20) :
  total_chickens - roosters - non_laying_hens = 277 := by
  sorry

#check egg_laying_hens

end egg_laying_hens_l2584_258485


namespace bottle_production_l2584_258475

/-- Given that 6 identical machines produce 420 bottles per minute at a constant rate,
    prove that 10 such machines will produce 2800 bottles in 4 minutes. -/
theorem bottle_production
  (machines : ℕ → ℕ) -- Function mapping number of machines to bottles produced per minute
  (h1 : machines 6 = 420) -- 6 machines produce 420 bottles per minute
  (h2 : ∀ n : ℕ, machines n = n * (machines 1)) -- Constant rate production
  : machines 10 * 4 = 2800 := by
  sorry


end bottle_production_l2584_258475


namespace f_increasing_when_a_1_m_range_l2584_258470

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + a*x

-- Define monotonically increasing
def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- Theorem 1: f is monotonically increasing when a = 1
theorem f_increasing_when_a_1 :
  monotonically_increasing (f 1) := by sorry

-- Define the necessary but not sufficient condition
def necessary_not_sufficient (m : ℝ) : Prop :=
  (∀ a : ℝ, monotonically_increasing (f a) → |a - 1| ≤ m) ∧
  (∃ a : ℝ, |a - 1| ≤ m ∧ ¬monotonically_increasing (f a))

-- Theorem 2: The range of m is [0,1)
theorem m_range :
  ∀ m : ℝ, (m > 0 ∧ necessary_not_sufficient m) ↔ (0 ≤ m ∧ m < 1) := by sorry

end f_increasing_when_a_1_m_range_l2584_258470


namespace matrix_linear_combination_l2584_258425

theorem matrix_linear_combination : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, 1; 3, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![-3, -2; 2, -4]
  2 • A + 3 • B = !![-1, -4; 12, -2] := by
  sorry

end matrix_linear_combination_l2584_258425


namespace shooting_training_results_l2584_258479

/-- Represents the shooting scores and their frequencies -/
structure ShootingData :=
  (scores : List Nat)
  (frequencies : List Nat)
  (excellent_threshold : Nat)
  (total_freshmen : Nat)

/-- Calculates the mode of the shooting data -/
def mode (data : ShootingData) : Nat :=
  sorry

/-- Calculates the average score of the shooting data -/
def average_score (data : ShootingData) : Rat :=
  sorry

/-- Estimates the number of excellent shooters -/
def estimate_excellent_shooters (data : ShootingData) : Nat :=
  sorry

/-- The main theorem proving the results of the shooting training -/
theorem shooting_training_results (data : ShootingData) 
  (h1 : data.scores = [6, 7, 8, 9])
  (h2 : data.frequencies = [1, 6, 3, 2])
  (h3 : data.excellent_threshold = 8)
  (h4 : data.total_freshmen = 1500) :
  mode data = 7 ∧ 
  average_score data = 15/2 ∧ 
  estimate_excellent_shooters data = 625 :=
sorry

end shooting_training_results_l2584_258479


namespace point_movement_l2584_258460

/-- The possible final positions of a point that starts 3 units from the origin,
    moves 4 units right, and then 1 unit left. -/
def final_positions : Set ℤ :=
  {0, 6}

/-- The theorem stating the possible final positions of the point. -/
theorem point_movement (A : ℤ) : 
  (abs A = 3) → 
  ((A + 4 - 1) ∈ final_positions) :=
by sorry

end point_movement_l2584_258460


namespace geometric_series_common_ratio_l2584_258419

theorem geometric_series_common_ratio :
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := -8/3
  let a₃ : ℚ := 64/21
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 2 → a₂ = a₁ * r ∧ a₃ = a₂ * r) →
  r = -14/3 :=
by sorry

end geometric_series_common_ratio_l2584_258419


namespace allen_reading_speed_l2584_258416

/-- The number of pages in Allen's book -/
def total_pages : ℕ := 120

/-- The number of days Allen took to read the book -/
def days_to_read : ℕ := 12

/-- The number of pages Allen read per day -/
def pages_per_day : ℕ := total_pages / days_to_read

/-- Theorem stating that Allen read 10 pages per day -/
theorem allen_reading_speed : pages_per_day = 10 := by
  sorry

end allen_reading_speed_l2584_258416


namespace circle_tangent_slope_l2584_258427

/-- The circle with center (2,0) and radius √3 -/
def Circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 3

/-- The vector from origin to point M -/
def OM (x y : ℝ) : ℝ × ℝ := (x, y)

/-- The vector from center C to point M -/
def CM (x y : ℝ) : ℝ × ℝ := (x - 2, y)

/-- The dot product of OM and CM -/
def dotProduct (x y : ℝ) : ℝ := x * (x - 2) + y * y

theorem circle_tangent_slope (x y : ℝ) :
  Circle x y →
  dotProduct x y = 0 →
  (y / x = Real.sqrt 3 ∨ y / x = -Real.sqrt 3) :=
by sorry

end circle_tangent_slope_l2584_258427


namespace two_possible_values_for_k_l2584_258428

theorem two_possible_values_for_k (a b c k : ℝ) : 
  (a / (b + c) = k ∧ b / (c + a) = k ∧ c / (a + b) = k) → 
  (k = 1/2 ∨ k = -1) ∧ ∀ x : ℝ, (x = 1/2 ∨ x = -1) → ∃ a b c : ℝ, a / (b + c) = x ∧ b / (c + a) = x ∧ c / (a + b) = x :=
by sorry

end two_possible_values_for_k_l2584_258428


namespace intersection_point_l2584_258447

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The equation of the first line: 2x + y + 2 = 0 -/
def line1 (x y : ℝ) : Prop := 2 * x + y + 2 = 0

/-- The equation of the second line: ax + 4y - 2 = 0 -/
def line2 (a x y : ℝ) : Prop := a * x + 4 * y - 2 = 0

/-- The theorem stating that the intersection point of the two perpendicular lines is (-1, 0) -/
theorem intersection_point :
  ∃ (a : ℝ),
    (∀ x y : ℝ, perpendicular (-2) (-a/4)) →
    (∀ x y : ℝ, line1 x y ∧ line2 a x y → x = -1 ∧ y = 0) :=
by sorry


end intersection_point_l2584_258447


namespace inequality_proof_l2584_258450

theorem inequality_proof (x y z : ℝ) (h1 : x < 0) (h2 : x < y) (h3 : y < z) :
  x + y < y + z := by
  sorry

end inequality_proof_l2584_258450


namespace initial_pigs_count_l2584_258437

theorem initial_pigs_count (initial_cows initial_goats added_cows added_pigs added_goats total_after : ℕ) :
  initial_cows = 2 →
  initial_goats = 6 →
  added_cows = 3 →
  added_pigs = 5 →
  added_goats = 2 →
  total_after = 21 →
  ∃ initial_pigs : ℕ, 
    initial_cows + initial_pigs + initial_goats + added_cows + added_pigs + added_goats = total_after ∧
    initial_pigs = 3 :=
by sorry

end initial_pigs_count_l2584_258437


namespace william_wins_l2584_258440

theorem william_wins (total_rounds : ℕ) (williams_advantage : ℕ) (williams_wins : ℕ) : 
  total_rounds = 15 → williams_advantage = 5 → williams_wins = 10 → 
  williams_wins = (total_rounds + williams_advantage) / 2 := by
  sorry

end william_wins_l2584_258440


namespace like_terms_imply_m_and_n_l2584_258452

/-- Two algebraic expressions are like terms if their variables have the same base and exponents -/
def are_like_terms (expr1 expr2 : ℕ → ℕ → ℝ) : Prop :=
  ∀ x y, ∃ c1 c2 : ℝ, expr1 x y = c1 * (x^(expr1 1 0) * y^(expr1 0 1)) ∧
                      expr2 x y = c2 * (x^(expr2 1 0) * y^(expr2 0 1)) ∧
                      expr1 1 0 = expr2 1 0 ∧
                      expr1 0 1 = expr2 0 1

theorem like_terms_imply_m_and_n (m n : ℕ) :
  are_like_terms (λ x y => -3 * x^(m-1) * y^3) (λ x y => 4 * x * y^(m+n)) →
  m = 2 ∧ n = 1 :=
by sorry

end like_terms_imply_m_and_n_l2584_258452


namespace cubic_equation_with_double_root_l2584_258463

/-- The cubic equation coefficients -/
def a : ℝ := 3
def b : ℝ := 9
def c : ℝ := -135

/-- The cubic equation has a double root -/
def has_double_root (x y : ℝ) : Prop :=
  x = 2 * y ∨ y = 2 * x

/-- The value of k for which the statement holds -/
def k : ℝ := 525

/-- The main theorem -/
theorem cubic_equation_with_double_root :
  ∃ (x y : ℝ),
    a * x^3 + b * x^2 + c * x + k = 0 ∧
    a * y^3 + b * y^2 + c * y + k = 0 ∧
    has_double_root x y ∧
    k > 0 :=
sorry

end cubic_equation_with_double_root_l2584_258463


namespace point_c_transformation_l2584_258446

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def translate (p : ℝ × ℝ) (t : ℝ × ℝ) : ℝ × ℝ := (p.1 + t.1, p.2 + t.2)

theorem point_c_transformation :
  let c : ℝ × ℝ := (3, 3)
  let c' := translate (reflect_x (reflect_y c)) (3, -4)
  c' = (0, -7) := by sorry

end point_c_transformation_l2584_258446


namespace no_reciprocal_sum_equals_sum_reciprocals_l2584_258468

theorem no_reciprocal_sum_equals_sum_reciprocals :
  ¬∃ (x y : ℝ), x ≠ -y ∧ x ≠ 0 ∧ y ≠ 0 ∧ (1 / (x + y) = 1 / x + 1 / y) := by
  sorry

end no_reciprocal_sum_equals_sum_reciprocals_l2584_258468


namespace best_play_win_probability_best_play_win_probability_multi_l2584_258439

/-- The probability that the best play wins with a majority of votes in a two-play competition. -/
theorem best_play_win_probability (n : ℕ) : ℝ :=
  let total_mothers : ℕ := 2 * n
  let confident_mothers : ℕ := n
  let non_confident_mothers : ℕ := n
  let vote_for_best_prob : ℝ := 1 / 2
  let vote_for_child_prob : ℝ := 1 / 2
  1 - (1 / 2) ^ n

/-- The probability that the best play wins with a majority of votes in a multi-play competition. -/
theorem best_play_win_probability_multi (n s : ℕ) : ℝ :=
  let total_mothers : ℕ := s * n
  let confident_mothers : ℕ := n
  let non_confident_mothers : ℕ := (s - 1) * n
  let vote_for_best_prob : ℝ := 1 / 2
  let vote_for_child_prob : ℝ := 1 / 2
  1 - (1 / 2) ^ ((s - 1) * n)

#check best_play_win_probability
#check best_play_win_probability_multi

end best_play_win_probability_best_play_win_probability_multi_l2584_258439


namespace calculate_expression_l2584_258456

theorem calculate_expression : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end calculate_expression_l2584_258456


namespace circle_equation_constant_l2584_258457

theorem circle_equation_constant (F : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 4*x + 8*y + F = 0 → 
    ∃ h k : ℝ, ∀ x' y' : ℝ, (x' - h)^2 + (y' - k)^2 = 4^2 → 
      x'^2 + y'^2 - 4*x' + 8*y' + F = 0) → 
  F = 4 := by
sorry

end circle_equation_constant_l2584_258457


namespace range_of_a_l2584_258464

-- Define the sets A, B, and C
def A : Set ℝ := {x | -5 < x ∧ x < 1}
def B : Set ℝ := {x | -2 < x ∧ x < 8}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem range_of_a (a : ℝ) (h : A ∩ B ⊆ C a) : a ≥ 1 := by
  sorry

end range_of_a_l2584_258464


namespace quadratic_function_properties_l2584_258471

/-- A quadratic function satisfying certain conditions -/
def f (x : ℝ) : ℝ := 2*x^2 - 2*x + 1

/-- The main theorem about the quadratic function f -/
theorem quadratic_function_properties :
  (∀ x, f (x + 1) - f x = 2 * x) ∧
  f 0 = 1 ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, -1/2 ≤ f x ∧ f x ≤ 1) ∧
  (∀ a : ℝ,
    (a ≤ -1/2 → ∀ x ∈ Set.Icc a (a + 1), 2*a^2 + 2*a + 3 ≤ f x ∧ f x ≤ 2*a^2 - 2*a + 1) ∧
    (-1/2 < a ∧ a ≤ 0 → ∀ x ∈ Set.Icc a (a + 1), -1/2 ≤ f x ∧ f x ≤ 2*a^2 - 2*a + 1) ∧
    (0 ≤ a ∧ a < 1/2 → ∀ x ∈ Set.Icc a (a + 1), -1/2 ≤ f x ∧ f x ≤ 2*a^2 + 2*a + 3) ∧
    (1/2 ≤ a → ∀ x ∈ Set.Icc a (a + 1), 2*a^2 - 2*a + 1 ≤ f x ∧ f x ≤ 2*a^2 + 2*a + 3)) :=
by sorry

end quadratic_function_properties_l2584_258471


namespace shopkeeper_milk_packets_l2584_258474

/-- Proves that the shopkeeper bought 150 packets of milk given the conditions -/
theorem shopkeeper_milk_packets 
  (packet_volume : ℕ) 
  (ounce_to_ml : ℕ) 
  (total_ounces : ℕ) 
  (h1 : packet_volume = 250)
  (h2 : ounce_to_ml = 30)
  (h3 : total_ounces = 1250) :
  (total_ounces * ounce_to_ml) / packet_volume = 150 := by
  sorry

end shopkeeper_milk_packets_l2584_258474


namespace worker_y_defective_rate_l2584_258482

/-- Calculates the defective rate of worker y given the conditions of the problem -/
theorem worker_y_defective_rate 
  (x_rate : Real) 
  (y_fraction : Real) 
  (total_rate : Real) 
  (hx : x_rate = 0.005) 
  (hy : y_fraction = 0.8) 
  (ht : total_rate = 0.0074) : 
  Real :=
by
  sorry

#check worker_y_defective_rate

end worker_y_defective_rate_l2584_258482


namespace inequality_system_solution_l2584_258436

theorem inequality_system_solution (x : ℝ) :
  (1/3 * x - 1 ≤ 1/2 * x + 1) →
  (3 * x - (x - 2) ≥ 6) →
  (x + 1 > (4 * x - 1) / 3) →
  (2 ≤ x ∧ x < 4) := by
sorry

end inequality_system_solution_l2584_258436


namespace triangle_side_b_is_4_triangle_area_is_4_sqrt_3_l2584_258467

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  side_angle_relation : a = b * Real.cos C + c * Real.cos B

-- Theorem 1
theorem triangle_side_b_is_4 (abc : Triangle) (h : abc.a - 4 * Real.cos abc.C = abc.c * Real.cos abc.B) :
  abc.b = 4 := by sorry

-- Theorem 2
theorem triangle_area_is_4_sqrt_3 (abc : Triangle)
  (h1 : abc.a - 4 * Real.cos abc.C = abc.c * Real.cos abc.B)
  (h2 : abc.a^2 + abc.b^2 + abc.c^2 = 2 * Real.sqrt 3 * abc.a * abc.b * Real.sin abc.C) :
  abc.a * abc.b * Real.sin abc.C / 2 = 4 * Real.sqrt 3 := by sorry

end triangle_side_b_is_4_triangle_area_is_4_sqrt_3_l2584_258467


namespace research_institute_reward_allocation_l2584_258401

theorem research_institute_reward_allocation :
  let n : ℕ := 10
  let a₁ : ℚ := 2
  let r : ℚ := 2
  let S := (a₁ * (1 - r^n)) / (1 - r)
  S = 2046 := by
sorry

end research_institute_reward_allocation_l2584_258401


namespace chord_length_polar_l2584_258404

/-- The chord length cut by the line ρcos(θ) = 1/2 from the circle ρ = 2cos(θ) is √3 -/
theorem chord_length_polar (ρ θ : ℝ) : 
  (ρ * Real.cos θ = 1/2) →  -- Line equation
  (ρ = 2 * Real.cos θ) →    -- Circle equation
  ∃ (chord_length : ℝ), chord_length = Real.sqrt 3 ∧ 
    chord_length = 2 * Real.sqrt (1 - (1/2)^2) := by
  sorry

end chord_length_polar_l2584_258404


namespace spiral_similarity_composition_l2584_258400

open Real

/-- A spiral similarity (also known as a rotational homothety) -/
structure SpiralSimilarity where
  center : ℝ × ℝ
  angle : ℝ
  coefficient : ℝ

/-- Composition of two spiral similarities -/
def compose (P₁ P₂ : SpiralSimilarity) : SpiralSimilarity :=
  sorry

/-- Rotation -/
structure Rotation where
  center : ℝ × ℝ
  angle : ℝ

/-- Check if a spiral similarity is a rotation -/
def isRotation (P : SpiralSimilarity) : Prop :=
  sorry

/-- The angle between two vectors -/
def vectorAngle (v₁ v₂ : ℝ × ℝ) : ℝ :=
  sorry

theorem spiral_similarity_composition
  (P₁ P₂ : SpiralSimilarity)
  (h₁ : P₁.angle = P₂.angle)
  (h₂ : P₁.coefficient * P₂.coefficient = 1)
  (M : ℝ × ℝ)
  (N : ℝ × ℝ)
  (hN : N = sorry) -- N = P₁(M)
  : 
  let P := compose P₂ P₁
  ∃ (R : Rotation), 
    isRotation P ∧ 
    P.center = R.center ∧
    R.center.fst = P₁.center.fst ∧ R.center.snd = P₂.center.snd ∧
    R.angle = 2 * vectorAngle (M.fst - P₁.center.fst, M.snd - P₁.center.snd) (N.fst - M.fst, N.snd - M.snd) :=
sorry

end spiral_similarity_composition_l2584_258400


namespace logarithm_inequality_l2584_258449

theorem logarithm_inequality (x : ℝ) (h : 1 < x ∧ x < 10) :
  Real.log (Real.log x) < (Real.log x)^2 ∧ (Real.log x)^2 < Real.log (x^2) := by
  sorry

end logarithm_inequality_l2584_258449


namespace range_of_a_l2584_258431

-- Define the conditions
def p (x a : ℝ) : Prop := |x - a| < 4
def q (x : ℝ) : Prop := -x^2 + 5*x - 6 > 0

-- Define the theorem
theorem range_of_a :
  ∃ (a_min a_max : ℝ),
    (a_min = -1 ∧ a_max = 6) ∧
    (∀ a : ℝ, (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x) ↔ a_min ≤ a ∧ a ≤ a_max) :=
sorry

end range_of_a_l2584_258431


namespace solve_otimes_equation_l2584_258453

-- Define the custom operation ⊗
def otimes (a b : ℝ) : ℝ := (a - 2) * (b + 1)

-- Theorem statement
theorem solve_otimes_equation : 
  ∃! x : ℝ, otimes (-4) (x + 3) = 6 ∧ x = -5 := by sorry

end solve_otimes_equation_l2584_258453


namespace max_value_x_minus_2z_l2584_258415

theorem max_value_x_minus_2z (x y z : ℝ) :
  x^2 + y^2 + z^2 = 16 →
  ∃ (max : ℝ), max = 4 * Real.sqrt 5 ∧ ∀ (x' y' z' : ℝ), x'^2 + y'^2 + z'^2 = 16 → x' - 2*z' ≤ max :=
sorry

end max_value_x_minus_2z_l2584_258415


namespace calculation_proof_l2584_258499

theorem calculation_proof : (1000 : ℤ) * 7 / 10 * 17 * (5^2) = 297500 := by
  sorry

end calculation_proof_l2584_258499


namespace dogs_and_movies_percentage_l2584_258408

theorem dogs_and_movies_percentage
  (total_students : ℕ)
  (dogs_and_games_percentage : ℚ)
  (dogs_preference : ℕ)
  (h1 : total_students = 30)
  (h2 : dogs_and_games_percentage = 1/2)
  (h3 : dogs_preference = 18) :
  (dogs_preference - (dogs_and_games_percentage * total_students)) / total_students = 1/10 :=
sorry

end dogs_and_movies_percentage_l2584_258408


namespace afternoon_campers_count_l2584_258496

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 44

/-- The difference between morning and afternoon campers -/
def difference : ℕ := 5

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := morning_campers - difference

theorem afternoon_campers_count : afternoon_campers = 39 := by
  sorry

end afternoon_campers_count_l2584_258496


namespace triangle_on_bottom_l2584_258454

/-- Represents the positions of faces on a cube -/
inductive CubeFace
  | Top
  | Bottom
  | East
  | South
  | West
  | North

/-- Represents the flattened cube configuration -/
structure FlattenedCube where
  faces : List CubeFace
  triangle_position : CubeFace

/-- The specific flattened cube configuration from the problem -/
def problem_cube : FlattenedCube := sorry

/-- Theorem stating that the triangle is on the bottom face in the given configuration -/
theorem triangle_on_bottom (c : FlattenedCube) : c.triangle_position = CubeFace.Bottom := by
  sorry

end triangle_on_bottom_l2584_258454


namespace mehki_age_proof_l2584_258466

/-- Proves that Mehki's age is 16 years old given the specified conditions -/
theorem mehki_age_proof (zrinka_age jordyn_age mehki_age : ℕ) : 
  zrinka_age = 6 →
  jordyn_age = zrinka_age - 4 →
  mehki_age = 2 * (jordyn_age + zrinka_age) →
  mehki_age = 16 := by
sorry

end mehki_age_proof_l2584_258466


namespace johns_total_pay_johns_total_pay_this_year_l2584_258459

/-- Calculates the total pay (salary + bonus) given a salary and bonus percentage -/
def totalPay (salary : ℝ) (bonusPercentage : ℝ) : ℝ :=
  salary * (1 + bonusPercentage)

/-- Theorem: John's total pay is equal to his salary plus his bonus -/
theorem johns_total_pay (salary : ℝ) (bonusPercentage : ℝ) :
  totalPay salary bonusPercentage = salary + (salary * bonusPercentage) :=
by sorry

/-- Theorem: John's total pay this year is $220,000 -/
theorem johns_total_pay_this_year 
  (lastYearSalary lastYearBonus thisYearSalary : ℝ)
  (h1 : lastYearSalary = 100000)
  (h2 : lastYearBonus = 10000)
  (h3 : thisYearSalary = 200000)
  (h4 : lastYearBonus / lastYearSalary = thisYearSalary * bonusPercentage / thisYearSalary) :
  totalPay thisYearSalary (lastYearBonus / lastYearSalary) = 220000 :=
by sorry

end johns_total_pay_johns_total_pay_this_year_l2584_258459


namespace parallel_vectors_imply_x_eq_neg_one_l2584_258435

/-- Two vectors in R² -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Definition of parallel vectors in R² -/
def parallel (v w : Vector2D) : Prop :=
  ∃ (k : ℝ), v.x = k * w.x ∧ v.y = k * w.y

/-- The main theorem -/
theorem parallel_vectors_imply_x_eq_neg_one :
  ∀ (x : ℝ),
  let a : Vector2D := ⟨x, 1⟩
  let b : Vector2D := ⟨1, -1⟩
  parallel a b → x = -1 := by
  sorry

end parallel_vectors_imply_x_eq_neg_one_l2584_258435


namespace revenue_fall_percentage_l2584_258480

theorem revenue_fall_percentage (R R' P P' : ℝ) 
  (h1 : P = 0.1 * R)
  (h2 : P' = 0.14 * R')
  (h3 : P' = 0.98 * P) :
  R' = 0.7 * R := by
sorry

end revenue_fall_percentage_l2584_258480


namespace max_z3_value_max_z3_value_tight_l2584_258412

theorem max_z3_value (z₁ z₂ z₃ : ℂ) 
  (h₁ : Complex.abs z₁ ≤ 1)
  (h₂ : Complex.abs z₂ ≤ 2)
  (h₃ : Complex.abs (2 * z₃ - z₁ - z₂) ≤ Complex.abs (z₁ - z₂)) :
  Complex.abs z₃ ≤ Real.sqrt 5 :=
by
  sorry

theorem max_z3_value_tight : ∃ (z₁ z₂ z₃ : ℂ),
  Complex.abs z₁ ≤ 1 ∧
  Complex.abs z₂ ≤ 2 ∧
  Complex.abs (2 * z₃ - z₁ - z₂) ≤ Complex.abs (z₁ - z₂) ∧
  Complex.abs z₃ = Real.sqrt 5 :=
by
  sorry

end max_z3_value_max_z3_value_tight_l2584_258412


namespace difference_of_squares_divisible_by_eight_l2584_258481

theorem difference_of_squares_divisible_by_eight (a b : ℤ) (h : a > b) :
  ∃ k : ℤ, 4 * (a - b) * (a + b + 1) = 8 * k := by
  sorry

end difference_of_squares_divisible_by_eight_l2584_258481


namespace square_of_sum_l2584_258478

theorem square_of_sum (a b : ℝ) : a^2 + b^2 + 2*a*b = (a + b)^2 := by sorry

end square_of_sum_l2584_258478


namespace specific_figure_perimeter_l2584_258434

/-- A composite figure made of squares and triangles -/
structure CompositeFigure where
  squareSideLength : ℝ
  triangleSideLength : ℝ
  numSquares : ℕ
  numTriangles : ℕ

/-- Calculate the perimeter of the composite figure -/
def perimeter (figure : CompositeFigure) : ℝ :=
  let squareContribution := 2 * figure.squareSideLength * (figure.numSquares + 2)
  let triangleContribution := figure.triangleSideLength * figure.numTriangles
  squareContribution + triangleContribution

/-- Theorem: The perimeter of the specific composite figure is 17 -/
theorem specific_figure_perimeter :
  let figure : CompositeFigure :=
    { squareSideLength := 2
      triangleSideLength := 1
      numSquares := 4
      numTriangles := 3 }
  perimeter figure = 17 := by
  sorry

end specific_figure_perimeter_l2584_258434


namespace function_properties_l2584_258486

def f (a c x : ℝ) : ℝ := a * x^2 + 2 * x + c

def g (a c x : ℝ) : ℝ := f a c x - 2 * x - 3 + |x - 1|

theorem function_properties :
  ∀ a c : ℕ+,
  f a c 1 = 5 →
  6 < f a c 2 ∧ f a c 2 < 11 →
  (a = 1 ∧ c = 2) ∧
  (∀ x : ℝ, g a c x ≥ -1/4) ∧
  (∃ x : ℝ, g a c x = -1/4) :=
by sorry

end function_properties_l2584_258486


namespace min_value_expression_min_value_achievable_l2584_258430

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 48) :
  x^2 + 4*x*y + 4*y^2 + 3*z^2 ≥ 144 :=
by sorry

theorem min_value_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 48 ∧ x^2 + 4*x*y + 4*y^2 + 3*z^2 = 144 :=
by sorry

end min_value_expression_min_value_achievable_l2584_258430


namespace pyramid_angle_ratio_relationship_l2584_258462

/-- A pyramid with all lateral faces forming the same angle with the base -/
structure Pyramid where
  base_area : ℝ
  lateral_angle : ℝ
  total_to_base_ratio : ℝ

/-- The angle formed by the lateral faces with the base of the pyramid -/
def lateral_angle (p : Pyramid) : ℝ := p.lateral_angle

/-- The ratio of the total surface area to the base area of the pyramid -/
def total_to_base_ratio (p : Pyramid) : ℝ := p.total_to_base_ratio

/-- Theorem stating the relationship between the lateral angle and the total-to-base area ratio -/
theorem pyramid_angle_ratio_relationship (p : Pyramid) :
  lateral_angle p = Real.arccos (4 / (total_to_base_ratio p - 1)) ∧
  total_to_base_ratio p > 5 := by sorry

end pyramid_angle_ratio_relationship_l2584_258462


namespace mersenne_prime_definition_l2584_258413

def is_mersenne_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, n = 2^p - 1 ∧ Nat.Prime n

def largest_known_prime : ℕ := 2^82589933 - 1

axiom largest_known_prime_is_prime : Nat.Prime largest_known_prime

theorem mersenne_prime_definition :
  ∀ n : ℕ, is_mersenne_prime n → (∃ name : String, name = "Mersenne prime") :=
by sorry

end mersenne_prime_definition_l2584_258413


namespace store_revenue_l2584_258402

theorem store_revenue (december : ℝ) (november january : ℝ)
  (h1 : november = (3/5) * december)
  (h2 : january = (1/3) * november) :
  december = (5/2) * ((november + january) / 2) :=
by sorry

end store_revenue_l2584_258402


namespace expand_and_simplify_l2584_258451

theorem expand_and_simplify (x : ℝ) (h : x ≠ 0) :
  (3 / 4) * (8 / x^2 + 12 * x - 5) = 6 / x^2 + 9 * x - 15 / 4 := by
  sorry

end expand_and_simplify_l2584_258451


namespace warehouse_repacking_l2584_258491

/-- The number of books left over after repacking in the warehouse scenario -/
theorem warehouse_repacking (initial_boxes : Nat) (books_per_initial_box : Nat) 
  (damaged_books : Nat) (books_per_new_box : Nat) 
  (h1 : initial_boxes = 1200)
  (h2 : books_per_initial_box = 35)
  (h3 : damaged_books = 100)
  (h4 : books_per_new_box = 45) : 
  (initial_boxes * books_per_initial_box - damaged_books) % books_per_new_box = 5 := by
  sorry

end warehouse_repacking_l2584_258491


namespace six_hour_rental_cost_l2584_258410

/-- Represents the cost structure for kayak and paddle rental --/
structure RentalCost where
  paddleFee : ℕ
  kayakHourlyRate : ℕ

/-- Calculates the total cost for a given number of hours --/
def totalCost (rc : RentalCost) (hours : ℕ) : ℕ :=
  rc.paddleFee + rc.kayakHourlyRate * hours

theorem six_hour_rental_cost 
  (rc : RentalCost)
  (three_hour_cost : totalCost rc 3 = 30)
  (kayak_rate : rc.kayakHourlyRate = 5) :
  totalCost rc 6 = 45 := by
  sorry

#check six_hour_rental_cost

end six_hour_rental_cost_l2584_258410


namespace perpendicular_line_equation_l2584_258484

/-- Given a line L1 with equation 2x-y+3=0 and a point P(1,1), 
    the line L2 passing through P and perpendicular to L1 
    has the equation x+2y-3=0 -/
theorem perpendicular_line_equation : 
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 2*x - y + 3 = 0
  let P : ℝ × ℝ := (1, 1)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ x + 2*y - 3 = 0
  (∀ x y, L2 x y ↔ (y - P.2 = -(1/2) * (x - P.1))) ∧ 
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → L2 x₁ y₁ → L2 x₂ y₂ → 
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) = 0) ∧
  L2 P.1 P.2 := by
  sorry


end perpendicular_line_equation_l2584_258484


namespace cards_per_page_l2584_258497

theorem cards_per_page
  (num_packs : ℕ)
  (cards_per_pack : ℕ)
  (num_pages : ℕ)
  (h1 : num_packs = 60)
  (h2 : cards_per_pack = 7)
  (h3 : num_pages = 42)
  : (num_packs * cards_per_pack) / num_pages = 10 :=
by
  sorry

end cards_per_page_l2584_258497


namespace pizza_slices_l2584_258433

theorem pizza_slices (x : ℚ) 
  (half_eaten : x / 2 = x - x / 2)
  (third_of_remaining_eaten : x / 2 - (x / 2) / 3 = x / 2 - x / 6)
  (four_slices_left : x / 2 - x / 6 = 4) : x = 12 := by
  sorry

end pizza_slices_l2584_258433


namespace greatest_power_of_two_factor_l2584_258429

theorem greatest_power_of_two_factor (n : ℕ) : 
  (∃ k : ℕ, 2^k ∣ (10^1000 + 4^500) ∧ 
   ∀ m : ℕ, 2^m ∣ (10^1000 + 4^500) → m ≤ k) → 
  n = 1003 :=
sorry

end greatest_power_of_two_factor_l2584_258429


namespace complex_points_on_circle_l2584_258448

theorem complex_points_on_circle 
  (a₁ a₂ a₃ a₄ a₅ : ℂ) 
  (h_nonzero : a₁ ≠ 0 ∧ a₂ ≠ 0 ∧ a₃ ≠ 0 ∧ a₄ ≠ 0 ∧ a₅ ≠ 0)
  (h_ratio : a₂ / a₁ = a₃ / a₂ ∧ a₃ / a₂ = a₄ / a₃ ∧ a₄ / a₃ = a₅ / a₄)
  (S : ℝ)
  (h_sum : a₁ + a₂ + a₃ + a₄ + a₅ = 4 * (1 / a₁ + 1 / a₂ + 1 / a₃ + 1 / a₄ + 1 / a₅))
  (h_S_real : a₁ + a₂ + a₃ + a₄ + a₅ = S)
  (h_S_bound : abs S ≤ 2) :
  ∃ (r : ℝ), r > 0 ∧ Complex.abs a₁ = r ∧ Complex.abs a₂ = r ∧ 
             Complex.abs a₃ = r ∧ Complex.abs a₄ = r ∧ Complex.abs a₅ = r := by
  sorry

end complex_points_on_circle_l2584_258448


namespace bet_is_unfair_a_has_advantage_l2584_258442

/-- Represents the outcome of rolling two dice -/
def DiceRoll := Fin 6 × Fin 6

/-- The probability of A winning (sum < 8) -/
def probAWins : ℚ := 7/12

/-- The probability of B winning (sum ≥ 8) -/
def probBWins : ℚ := 5/12

/-- A's bet amount in forints -/
def aBet : ℚ := 10

/-- B's bet amount in forints -/
def bBet : ℚ := 8

/-- Expected gain for A in forints -/
def expectedGainA : ℚ := bBet * probAWins - aBet * probBWins

theorem bet_is_unfair : expectedGainA = 1/2 := by sorry

theorem a_has_advantage : expectedGainA > 0 := by sorry

end bet_is_unfair_a_has_advantage_l2584_258442


namespace evaluate_expression_l2584_258483

theorem evaluate_expression : (0.5 ^ 4) / (0.05 ^ 3) = 500 := by
  sorry

end evaluate_expression_l2584_258483


namespace gcd_153_68_l2584_258487

theorem gcd_153_68 : Nat.gcd 153 68 = 17 := by
  sorry

end gcd_153_68_l2584_258487


namespace fractional_equation_root_l2584_258409

theorem fractional_equation_root (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ x ≠ 2 ∧ m / (x - 2) + 2 * x / (x - 2) = 1) → m = -4 := by
  sorry

end fractional_equation_root_l2584_258409


namespace divisor_problem_l2584_258492

theorem divisor_problem (n : ℕ) (h : n = 1101) : 
  ∃ (d : ℕ), d > 1 ∧ (n + 3) % d = 0 ∧ d = 8 := by
  sorry

end divisor_problem_l2584_258492


namespace triangle_side_length_l2584_258417

theorem triangle_side_length (a b c : ℝ) (S : ℝ) (hA : a = 4) (hB : b = 5) (hS : S = 5 * Real.sqrt 3) :
  c = Real.sqrt 21 ∨ c = Real.sqrt 61 := by
  sorry

end triangle_side_length_l2584_258417


namespace roses_cut_l2584_258493

def initial_roses : ℕ := 6
def final_roses : ℕ := 16

theorem roses_cut (cut_roses : ℕ) : cut_roses = final_roses - initial_roses := by
  sorry

end roses_cut_l2584_258493
