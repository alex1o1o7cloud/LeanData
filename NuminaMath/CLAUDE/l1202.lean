import Mathlib

namespace NUMINAMATH_CALUDE_travel_options_count_l1202_120276

/-- The number of bus options from A to B -/
def bus_options : ℕ := 5

/-- The number of train options from A to B -/
def train_options : ℕ := 6

/-- The number of boat options from A to B -/
def boat_options : ℕ := 2

/-- The total number of travel options from A to B -/
def total_options : ℕ := bus_options + train_options + boat_options

theorem travel_options_count :
  total_options = 13 :=
by sorry

end NUMINAMATH_CALUDE_travel_options_count_l1202_120276


namespace NUMINAMATH_CALUDE_red_item_count_l1202_120283

/-- Represents the number of items of a specific color in the box -/
structure ColorCount where
  hats : ℕ
  gloves : ℕ

/-- Represents the contents of the box -/
structure Box where
  red : ColorCount
  green : ColorCount
  orange : ColorCount

/-- The maximum number of draws needed to guarantee a pair of each color -/
def max_draws (b : Box) : ℕ :=
  max (b.red.hats + b.red.gloves) (max (b.green.hats + b.green.gloves) (b.orange.hats + b.orange.gloves)) + 2

/-- The theorem stating that if it takes 66 draws to guarantee a pair of each color,
    given 23 green items and 11 orange items, then there must be 30 red items -/
theorem red_item_count (b : Box) 
  (h_green : b.green.hats + b.green.gloves = 23)
  (h_orange : b.orange.hats + b.orange.gloves = 11)
  (h_draws : max_draws b = 66) :
  b.red.hats + b.red.gloves = 30 := by
  sorry

end NUMINAMATH_CALUDE_red_item_count_l1202_120283


namespace NUMINAMATH_CALUDE_sophie_rearrangement_time_l1202_120246

def name_length : ℕ := 6
def rearrangements_per_minute : ℕ := 18

theorem sophie_rearrangement_time :
  (name_length.factorial / rearrangements_per_minute : ℚ) / 60 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sophie_rearrangement_time_l1202_120246


namespace NUMINAMATH_CALUDE_max_value_parabola_l1202_120248

/-- The maximum value of y = -3x^2 + 6, where x is a real number, is 6. -/
theorem max_value_parabola :
  ∃ (M : ℝ), M = 6 ∧ ∀ (x : ℝ), -3 * x^2 + 6 ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_parabola_l1202_120248


namespace NUMINAMATH_CALUDE_rebeccas_income_l1202_120223

/-- Rebecca's annual income problem -/
theorem rebeccas_income (R : ℚ) : 
  (∃ (J : ℚ), J = 18000 ∧ R + 3000 = 0.5 * (R + 3000 + J)) → R = 15000 := by
  sorry

end NUMINAMATH_CALUDE_rebeccas_income_l1202_120223


namespace NUMINAMATH_CALUDE_prob_two_out_of_three_germinate_l1202_120281

/-- The probability of exactly k successes in n independent Bernoulli trials 
    with probability p of success for each trial -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of exactly 2 successes out of 3 trials 
    with probability 4/5 of success for each trial -/
theorem prob_two_out_of_three_germinate : 
  binomial_probability 3 2 (4/5) = 48/125 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_out_of_three_germinate_l1202_120281


namespace NUMINAMATH_CALUDE_valid_square_root_expression_l1202_120294

theorem valid_square_root_expression (a b : ℝ) : 
  (Real.sqrt (-a^2 * b^2) = -a * b) ↔ (a * b = 0) := by sorry

end NUMINAMATH_CALUDE_valid_square_root_expression_l1202_120294


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1202_120217

theorem inequality_solution_set (x : ℝ) : x^2 + 3 < 4*x ↔ 1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1202_120217


namespace NUMINAMATH_CALUDE_monomial_sum_condition_l1202_120234

/-- If the sum of two monomials 3x^5y^m and -2x^ny^7 is still a monomial in terms of x and y, 
    then m - n = 2 -/
theorem monomial_sum_condition (m n : ℕ) : 
  (∃ (a : ℚ) (p q : ℕ), 3 * X^5 * Y^m + -2 * X^n * Y^7 = a * X^p * Y^q) → 
  m - n = 2 := by
  sorry

end NUMINAMATH_CALUDE_monomial_sum_condition_l1202_120234


namespace NUMINAMATH_CALUDE_fifth_number_21st_row_l1202_120273

/-- Represents the array of odd numbers -/
def oddNumberArray (row : ℕ) (position : ℕ) : ℕ :=
  2 * (row * (row - 1) / 2 + position) - 1

/-- The theorem to prove -/
theorem fifth_number_21st_row :
  oddNumberArray 21 5 = 809 :=
sorry

end NUMINAMATH_CALUDE_fifth_number_21st_row_l1202_120273


namespace NUMINAMATH_CALUDE_fraction_problem_l1202_120200

theorem fraction_problem (N : ℝ) (f : ℝ) 
  (h1 : (1 / 3) * f * N = 15) 
  (h2 : (3 / 10) * N = 54) : 
  f = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l1202_120200


namespace NUMINAMATH_CALUDE_sample_frequency_calculation_l1202_120229

theorem sample_frequency_calculation (total_volume : ℕ) (num_groups : ℕ) 
  (freq_3 freq_4 freq_5 freq_6 : ℕ) (ratio_group_1 : ℚ) :
  total_volume = 80 →
  num_groups = 6 →
  freq_3 = 10 →
  freq_4 = 12 →
  freq_5 = 14 →
  freq_6 = 20 →
  ratio_group_1 = 1/5 →
  ∃ (freq_1 freq_2 : ℕ),
    freq_1 = 16 ∧
    freq_2 = 8 ∧
    freq_1 + freq_2 + freq_3 + freq_4 + freq_5 + freq_6 = total_volume ∧
    freq_1 = (ratio_group_1 * total_volume).num := by
  sorry

end NUMINAMATH_CALUDE_sample_frequency_calculation_l1202_120229


namespace NUMINAMATH_CALUDE_cloth_sale_meters_l1202_120243

/-- Proves that the number of meters of cloth sold is 85, given the total selling price,
    profit per meter, and cost price per meter. -/
theorem cloth_sale_meters (total_selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ) :
  total_selling_price = 8500 →
  profit_per_meter = 15 →
  cost_price_per_meter = 85 →
  (total_selling_price / (cost_price_per_meter + profit_per_meter) : ℕ) = 85 := by
sorry

end NUMINAMATH_CALUDE_cloth_sale_meters_l1202_120243


namespace NUMINAMATH_CALUDE_parabola_coefficient_b_l1202_120263

/-- Given a parabola y = ax^2 + bx + c with vertex at (q, -q) and y-intercept at (0, q),
    where q ≠ 0, the coefficient b is equal to -4. -/
theorem parabola_coefficient_b (a b c q : ℝ) (hq : q ≠ 0) :
  (∀ x, a * x^2 + b * x + c = a * (x - q)^2 - q) →
  (a * 0^2 + b * 0 + c = q) →
  b = -4 := by
sorry

end NUMINAMATH_CALUDE_parabola_coefficient_b_l1202_120263


namespace NUMINAMATH_CALUDE_fraction_equality_l1202_120224

theorem fraction_equality (a b : ℝ) (h : a ≠ 0) : b / a = (a * b) / (a^2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1202_120224


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l1202_120226

/-- A random variable following a normal distribution with mean μ and standard deviation σ. -/
structure NormalRV (μ σ : ℝ) where
  (σ_pos : σ > 0)

/-- The probability that a normal random variable falls within a given interval. -/
def prob_interval (X : NormalRV μ σ) (a b : ℝ) : ℝ := sorry

/-- Theorem: For a normal distribution N(4, 1²), given specific probabilities for certain intervals,
    the probability P(5 < X < 6) is equal to 0.1359. -/
theorem normal_distribution_probability (X : NormalRV 4 1) :
  prob_interval X 2 6 = 0.9544 →
  prob_interval X 3 5 = 0.6826 →
  prob_interval X 5 6 = 0.1359 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l1202_120226


namespace NUMINAMATH_CALUDE_symmetric_center_phi_l1202_120203

theorem symmetric_center_phi (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = Real.sin (-2 * x + φ)) →
  0 < φ →
  φ < π →
  (∃ k : ℤ, -2 * (π / 3) + φ = k * π) →
  φ = 2 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_symmetric_center_phi_l1202_120203


namespace NUMINAMATH_CALUDE_berry_average_temperature_l1202_120202

def berry_temperatures : List (List Float) := [
  [37.3, 37.2, 36.9],  -- Sunday
  [36.6, 36.9, 37.1],  -- Monday
  [37.1, 37.3, 37.2],  -- Tuesday
  [36.8, 37.3, 37.5],  -- Wednesday
  [37.1, 37.7, 37.3],  -- Thursday
  [37.5, 37.4, 36.9],  -- Friday
  [36.9, 37.0, 37.1]   -- Saturday
]

def average_temperature (temperatures : List (List Float)) : Float :=
  let total_sum := temperatures.map (·.sum) |>.sum
  let total_count := temperatures.length * temperatures.head!.length
  total_sum / total_count.toFloat

theorem berry_average_temperature :
  (average_temperature berry_temperatures).floor = 37 ∧
  (average_temperature berry_temperatures - (average_temperature berry_temperatures).floor) * 100 ≥ 62 :=
by sorry

end NUMINAMATH_CALUDE_berry_average_temperature_l1202_120202


namespace NUMINAMATH_CALUDE_joes_steakhouse_wages_l1202_120236

/-- Proves that the hourly wage of a manager is $8.5 given the conditions from Joe's Steakhouse --/
theorem joes_steakhouse_wages (manager_wage dishwasher_wage chef_wage : ℝ) : 
  chef_wage = dishwasher_wage * 1.22 →
  dishwasher_wage = manager_wage / 2 →
  chef_wage = manager_wage - 3.315 →
  manager_wage = 8.5 := by
sorry

end NUMINAMATH_CALUDE_joes_steakhouse_wages_l1202_120236


namespace NUMINAMATH_CALUDE_intersection_value_l1202_120213

theorem intersection_value (m n : ℝ) (h1 : n = 2 / m) (h2 : n = m + 3) :
  1 / m - 1 / n = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_value_l1202_120213


namespace NUMINAMATH_CALUDE_cubic_function_minimum_l1202_120282

theorem cubic_function_minimum (a b c : ℝ) : 
  let f := fun x => a * x^3 + b * x^2 + c * x - 34
  let f' := fun x => 3 * a * x^2 + 2 * b * x + c
  (∀ x, f' x ≤ 0 ↔ -2 ≤ x ∧ x ≤ 3) →
  (∃ x₀, ∀ x, f x ≥ f x₀) →
  f 3 = -115 →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_minimum_l1202_120282


namespace NUMINAMATH_CALUDE_time_differences_not_constant_l1202_120288

/-- Represents the relationship between height and time for the sliding car experiment -/
def slide_data : List (ℝ × ℝ) :=
  [(10, 4.23), (20, 3.00), (30, 2.45), (40, 2.13), (50, 1.89), (60, 1.71), (70, 1.59)]

/-- Calculates the time difference between two consecutive measurements -/
def time_diff (data : List (ℝ × ℝ)) (i : ℕ) : ℝ :=
  match data.get? i, data.get? (i+1) with
  | some (_, t1), some (_, t2) => t1 - t2
  | _, _ => 0

/-- Theorem stating that time differences are not constant -/
theorem time_differences_not_constant : ∃ i j, i ≠ j ∧ i < slide_data.length - 1 ∧ j < slide_data.length - 1 ∧ time_diff slide_data i ≠ time_diff slide_data j :=
sorry

end NUMINAMATH_CALUDE_time_differences_not_constant_l1202_120288


namespace NUMINAMATH_CALUDE_triangle_problem_l1202_120215

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : Real.sqrt 3 * t.b * Real.sin t.A = t.a * Real.cos t.B)
  (h2 : t.b = 3)
  (h3 : Real.sin t.C = Real.sqrt 3 * Real.sin t.A) :
  t.B = π / 6 ∧ t.a = 3 ∧ t.c = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1202_120215


namespace NUMINAMATH_CALUDE_tim_travel_distance_l1202_120272

/-- Represents the distance traveled by Tim and Élan -/
structure TravelDistance where
  tim : ℝ
  elan : ℝ

/-- Calculates the distance traveled in one hour given initial speeds -/
def distanceInHour (timSpeed : ℝ) (elanSpeed : ℝ) : TravelDistance :=
  { tim := timSpeed, elan := elanSpeed }

/-- Theorem: Tim travels 60 miles before meeting Élan -/
theorem tim_travel_distance (initialDistance : ℝ) (timInitialSpeed : ℝ) (elanInitialSpeed : ℝ) :
  initialDistance = 90 ∧ timInitialSpeed = 10 ∧ elanInitialSpeed = 5 →
  (let d1 := distanceInHour timInitialSpeed elanInitialSpeed
   let d2 := distanceInHour (2 * timInitialSpeed) (2 * elanInitialSpeed)
   let d3 := distanceInHour (4 * timInitialSpeed) (4 * elanInitialSpeed)
   d1.tim + d2.tim + (initialDistance - d1.tim - d1.elan - d2.tim - d2.elan) * (4 * timInitialSpeed) / (4 * timInitialSpeed + 4 * elanInitialSpeed) = 60) :=
by
  sorry


end NUMINAMATH_CALUDE_tim_travel_distance_l1202_120272


namespace NUMINAMATH_CALUDE_animal_books_count_animal_books_proof_l1202_120261

def book_price : ℕ := 16
def space_books : ℕ := 1
def train_books : ℕ := 3
def total_spent : ℕ := 224

theorem animal_books_count : ℕ :=
  (total_spent - book_price * (space_books + train_books)) / book_price

#check animal_books_count

theorem animal_books_proof :
  animal_books_count = 10 :=
by sorry

end NUMINAMATH_CALUDE_animal_books_count_animal_books_proof_l1202_120261


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l1202_120285

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x < 0}

-- State the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : solution_set a b c = {x : ℝ | x < 1 ∨ x > 3}) :
  c < 0 ∧
  a + 2*b + 4*c < 0 ∧
  {x : ℝ | c*x + a < 0} = {x : ℝ | x > -1/3} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l1202_120285


namespace NUMINAMATH_CALUDE_no_solution_to_inequality_system_l1202_120232

theorem no_solution_to_inequality_system :
  ¬∃ (x y z t : ℝ), 
    (abs x > abs (y - z + t)) ∧
    (abs y > abs (x - z + t)) ∧
    (abs z > abs (x - y + t)) ∧
    (abs t > abs (x - y + z)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_inequality_system_l1202_120232


namespace NUMINAMATH_CALUDE_johns_money_in_euros_johns_money_in_euros_proof_l1202_120237

/-- Proves that John's money in Euros is 612 given the conditions of the problem -/
theorem johns_money_in_euros : ℝ → Prop :=
  fun conversion_rate =>
    ∀ (darwin mia laura john : ℝ),
      darwin = 45 →
      mia = 2 * darwin + 20 →
      laura = 3 * (mia + darwin) - 30 →
      john = 1.5 * (laura + darwin) →
      conversion_rate = 0.85 →
      john * conversion_rate = 612

/-- Proof of the theorem -/
theorem johns_money_in_euros_proof : johns_money_in_euros 0.85 := by
  sorry

end NUMINAMATH_CALUDE_johns_money_in_euros_johns_money_in_euros_proof_l1202_120237


namespace NUMINAMATH_CALUDE_geometric_sequence_min_value_l1202_120239

/-- A geometric sequence with positive terms where the 7th term is √2/2 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧ 
  (∀ n : ℕ, a n > 0) ∧
  (a 7 = Real.sqrt 2 / 2)

/-- The minimum value of 1/a_3 + 2/a_11 for the given geometric sequence is 4 -/
theorem geometric_sequence_min_value (a : ℕ → ℝ) (h : GeometricSequence a) :
  (1 / a 3 + 2 / a 11) ≥ 4 ∧ ∃ b : ℕ → ℝ, GeometricSequence b ∧ 1 / b 3 + 2 / b 11 = 4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_min_value_l1202_120239


namespace NUMINAMATH_CALUDE_singles_percentage_l1202_120250

def total_hits : ℕ := 40
def home_runs : ℕ := 2
def triples : ℕ := 3
def doubles : ℕ := 6

def singles : ℕ := total_hits - (home_runs + triples + doubles)

def percentage_singles : ℚ := singles / total_hits * 100

theorem singles_percentage : percentage_singles = 72.5 := by
  sorry

end NUMINAMATH_CALUDE_singles_percentage_l1202_120250


namespace NUMINAMATH_CALUDE_average_increase_is_eight_l1202_120284

/-- Represents a cricketer's batting statistics -/
structure CricketerStats where
  initialInnings : Nat
  initialTotalRuns : Nat
  newInningScore : Nat
  newAverage : Nat

/-- Calculates the increase in average for a cricketer -/
def averageIncrease (stats : CricketerStats) : Nat :=
  stats.newAverage - (stats.initialTotalRuns / stats.initialInnings)

/-- Theorem: Given the specific conditions, the average increase is 8 runs -/
theorem average_increase_is_eight (stats : CricketerStats) 
  (h1 : stats.initialInnings = 9)
  (h2 : stats.newInningScore = 200)
  (h3 : stats.newAverage = 128)
  (h4 : stats.initialTotalRuns + stats.newInningScore = (stats.initialInnings + 1) * stats.newAverage) :
  averageIncrease stats = 8 := by
  sorry

#eval averageIncrease { initialInnings := 9, initialTotalRuns := 1080, newInningScore := 200, newAverage := 128 }

end NUMINAMATH_CALUDE_average_increase_is_eight_l1202_120284


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1202_120242

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x : ℝ | a * x^2 + 5 * x + b > 0}) : 
  {x : ℝ | b * x^2 - 5 * x + a > 0} = Set.Ioo (-1/2) (-1/3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1202_120242


namespace NUMINAMATH_CALUDE_afternoon_more_than_morning_l1202_120290

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 6

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 8

/-- The difference in emails between afternoon and morning -/
def email_difference : ℕ := afternoon_emails - morning_emails

theorem afternoon_more_than_morning : email_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_more_than_morning_l1202_120290


namespace NUMINAMATH_CALUDE_star_six_three_l1202_120255

-- Define the * operation
def star (a b : ℤ) : ℤ := 4*a + 5*b - 2*a*b

-- Theorem statement
theorem star_six_three : star 6 3 = 3 := by sorry

end NUMINAMATH_CALUDE_star_six_three_l1202_120255


namespace NUMINAMATH_CALUDE_max_profit_l1202_120275

-- Define the linear relationship between price and quantity
def sales_quantity (x : ℝ) : ℝ := -2 * x + 180

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - 50) * (sales_quantity x)

-- Theorem statement
theorem max_profit :
  ∃ (x : ℝ), x = 70 ∧ profit x = 800 ∧ ∀ (y : ℝ), profit y ≤ profit x :=
sorry

end NUMINAMATH_CALUDE_max_profit_l1202_120275


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1202_120258

theorem quadratic_factorization :
  ∀ x : ℝ, x^2 - 2*x - 2 = 0 ↔ (x - 1)^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1202_120258


namespace NUMINAMATH_CALUDE_first_class_rate_total_l1202_120279

-- Define the pass rate
def pass_rate : ℝ := 0.95

-- Define the rate of first-class products among qualified products
def first_class_rate_qualified : ℝ := 0.20

-- Theorem statement
theorem first_class_rate_total (pass_rate : ℝ) (first_class_rate_qualified : ℝ) :
  pass_rate * first_class_rate_qualified = 0.19 := by
  sorry

end NUMINAMATH_CALUDE_first_class_rate_total_l1202_120279


namespace NUMINAMATH_CALUDE_panel_discussion_selection_l1202_120260

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

theorem panel_discussion_selection (num_boys num_girls : ℕ) 
  (hb : num_boys = 5) (hg : num_girls = 4) : 
  -- I. Number of ways to select 2 boys and 2 girls
  (choose num_boys 2) * (choose num_girls 2) = 60 ∧ 
  -- II. Number of ways to select 4 people including at least one of boy A or girl B
  (choose (num_boys + num_girls) 4) - (choose (num_boys + num_girls - 2) 4) = 91 ∧
  -- III. Number of ways to select 4 people containing both boys and girls
  (choose (num_boys + num_girls) 4) - (choose num_boys 4) - (choose num_girls 4) = 120 :=
by sorry

end NUMINAMATH_CALUDE_panel_discussion_selection_l1202_120260


namespace NUMINAMATH_CALUDE_master_bedroom_size_l1202_120269

theorem master_bedroom_size (total_area : ℝ) (common_area : ℝ) (guest_ratio : ℝ) :
  total_area = 2300 →
  common_area = 1000 →
  guest_ratio = 1/4 →
  ∃ (master_size : ℝ),
    master_size = 1040 ∧
    total_area = common_area + master_size + guest_ratio * master_size :=
by
  sorry

end NUMINAMATH_CALUDE_master_bedroom_size_l1202_120269


namespace NUMINAMATH_CALUDE_right_triangle_in_circle_l1202_120225

theorem right_triangle_in_circle (d : ℝ) (b : ℝ) (a : ℝ) : 
  d = 10 → b = 8 → a * a + b * b = d * d → a = 6 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_in_circle_l1202_120225


namespace NUMINAMATH_CALUDE_simplify_fraction_l1202_120287

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1202_120287


namespace NUMINAMATH_CALUDE_sqrt_twelve_is_quadratic_radical_l1202_120297

/-- Definition of a quadratic radical -/
def is_quadratic_radical (x : ℝ) : Prop :=
  ∃ (y : ℝ), y ≥ 0 ∧ x = Real.sqrt y

/-- Theorem stating that √12 is a quadratic radical -/
theorem sqrt_twelve_is_quadratic_radical :
  is_quadratic_radical (Real.sqrt 12) :=
by
  sorry


end NUMINAMATH_CALUDE_sqrt_twelve_is_quadratic_radical_l1202_120297


namespace NUMINAMATH_CALUDE_basketball_season_games_l1202_120265

/-- The total number of games played by a basketball team in a season -/
def total_games : ℕ := 93

/-- The number of games in the first segment -/
def first_segment : ℕ := 40

/-- The number of games in the second segment -/
def second_segment : ℕ := 30

/-- The win rate for the first segment -/
def first_rate : ℚ := 1/2

/-- The win rate for the second segment -/
def second_rate : ℚ := 3/5

/-- The win rate for the remaining games -/
def remaining_rate : ℚ := 17/20

/-- The overall win rate for the season -/
def overall_rate : ℚ := 31/50

theorem basketball_season_games :
  let remaining_games := total_games - first_segment - second_segment
  let total_wins := (first_rate * first_segment) + (second_rate * second_segment) + (remaining_rate * remaining_games)
  total_wins = overall_rate * total_games := by sorry

#eval total_games

end NUMINAMATH_CALUDE_basketball_season_games_l1202_120265


namespace NUMINAMATH_CALUDE_nonconsecutive_choose_18_5_l1202_120256

def nonconsecutive_choose (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - k + 1) k

theorem nonconsecutive_choose_18_5 :
  nonconsecutive_choose 18 5 = Nat.choose 14 5 :=
sorry

end NUMINAMATH_CALUDE_nonconsecutive_choose_18_5_l1202_120256


namespace NUMINAMATH_CALUDE_complex_number_modulus_l1202_120296

theorem complex_number_modulus : Complex.abs ((1 - Complex.I) / Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l1202_120296


namespace NUMINAMATH_CALUDE_sum_of_digits_An_l1202_120289

-- Define the product An
def An (n : ℕ) : ℕ :=
  (List.range (n + 1)).foldl (λ acc i => acc * (10^(2^i) - 1)) 9

-- Define the sum of digits function
def sumOfDigits (m : ℕ) : ℕ :=
  if m < 10 then m else m % 10 + sumOfDigits (m / 10)

-- State the theorem
theorem sum_of_digits_An (n : ℕ) :
  sumOfDigits (An n) = 9 * 2^n := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_An_l1202_120289


namespace NUMINAMATH_CALUDE_average_temp_bucyrus_l1202_120227

/-- The average temperature in Bucyrus, Ohio over three days -/
def average_temperature (temp1 temp2 temp3 : ℤ) : ℚ :=
  (temp1 + temp2 + temp3) / 3

/-- Theorem stating that the average of the given temperatures is -7 -/
theorem average_temp_bucyrus :
  average_temperature (-14) (-8) 1 = -7 := by
  sorry

end NUMINAMATH_CALUDE_average_temp_bucyrus_l1202_120227


namespace NUMINAMATH_CALUDE_complement_of_A_l1202_120249

def A : Set ℝ := {y | ∃ x, y = 2^x}

theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = Set.Iic 0 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1202_120249


namespace NUMINAMATH_CALUDE_sum_first_eight_primes_mod_tenth_prime_l1202_120259

def first_eight_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]
def tenth_prime : Nat := 29

theorem sum_first_eight_primes_mod_tenth_prime :
  (first_eight_primes.sum) % tenth_prime = 19 := by sorry

end NUMINAMATH_CALUDE_sum_first_eight_primes_mod_tenth_prime_l1202_120259


namespace NUMINAMATH_CALUDE_not_geometric_progression_l1202_120253

theorem not_geometric_progression : 
  ¬∃ (a r : ℝ) (p q k : ℤ), 
    p ≠ q ∧ q ≠ k ∧ p ≠ k ∧ 
    a * r^(p-1) = 10 ∧ 
    a * r^(q-1) = 11 ∧ 
    a * r^(k-1) = 12 := by
  sorry

end NUMINAMATH_CALUDE_not_geometric_progression_l1202_120253


namespace NUMINAMATH_CALUDE_speaking_orders_count_l1202_120292

/-- The number of students in the class --/
def totalStudents : ℕ := 7

/-- The number of students to be selected for speaking --/
def selectedSpeakers : ℕ := 4

/-- Function to calculate the number of speaking orders --/
def speakingOrders (total : ℕ) (selected : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of speaking orders under given conditions --/
theorem speaking_orders_count :
  speakingOrders totalStudents selectedSpeakers = 600 :=
sorry

end NUMINAMATH_CALUDE_speaking_orders_count_l1202_120292


namespace NUMINAMATH_CALUDE_cylinder_min_circumscribed_sphere_l1202_120210

/-- For a cylinder with surface area 16π and base radius r, 
    the surface area of its circumscribed sphere is minimized when r² = 8√5/5 -/
theorem cylinder_min_circumscribed_sphere (r : ℝ) : 
  (2 * π * r^2 + 2 * π * r * ((8 : ℝ) / r - r) = 16 * π) →
  (∃ (R : ℝ), R^2 = r^2 + ((8 : ℝ) / r - r)^2 / 4 ∧ 
    ∀ (R' : ℝ), R'^2 = r^2 + ((8 : ℝ) / r' - r')^2 / 4 → R'^2 ≥ R^2) →
  r^2 = 8 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_cylinder_min_circumscribed_sphere_l1202_120210


namespace NUMINAMATH_CALUDE_smallest_positive_term_l1202_120271

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

/-- Sum of first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ d : ℚ) (n : ℕ) : ℚ := n * (a₁ + arithmetic_sequence a₁ d n) / 2

theorem smallest_positive_term (d : ℚ) :
  let a := arithmetic_sequence (-12) d
  let S := arithmetic_sum (-12) d
  S 13 = 0 →
  (∀ k < 8, a k ≤ 0) ∧ a 8 > 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_term_l1202_120271


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1202_120212

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 99) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 19 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1202_120212


namespace NUMINAMATH_CALUDE_fourth_root_equation_solution_l1202_120211

theorem fourth_root_equation_solution (x : ℝ) : 
  (x * (x^4)^(1/2))^(1/4) = 4 → x = 2^(8/3) := by sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solution_l1202_120211


namespace NUMINAMATH_CALUDE_average_difference_l1202_120264

theorem average_difference : 
  let set1 : List ℝ := [10, 20, 60]
  let set2 : List ℝ := [10, 40, 25]
  (set1.sum / set1.length) - (set2.sum / set2.length) = 5 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l1202_120264


namespace NUMINAMATH_CALUDE_books_returned_thursday_l1202_120220

/-- The number of books returned on Thursday given the initial conditions and final count. -/
theorem books_returned_thursday 
  (initial_wednesday : ℕ) 
  (checkout_wednesday : ℕ) 
  (checkout_thursday : ℕ) 
  (returned_friday : ℕ) 
  (final_friday : ℕ) 
  (h1 : initial_wednesday = 98) 
  (h2 : checkout_wednesday = 43) 
  (h3 : checkout_thursday = 5) 
  (h4 : returned_friday = 7) 
  (h5 : final_friday = 80) : 
  final_friday = initial_wednesday - checkout_wednesday - checkout_thursday + returned_friday + 23 := by
  sorry

#check books_returned_thursday

end NUMINAMATH_CALUDE_books_returned_thursday_l1202_120220


namespace NUMINAMATH_CALUDE_san_francisco_super_bowl_probability_l1202_120293

theorem san_francisco_super_bowl_probability 
  (p_play : ℝ) 
  (p_not_play : ℝ) 
  (h1 : p_play = 9 * p_not_play) 
  (h2 : p_play + p_not_play = 1) : 
  p_play = 0.9 := by
sorry

end NUMINAMATH_CALUDE_san_francisco_super_bowl_probability_l1202_120293


namespace NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l1202_120221

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of rectangles in a square grid -/
def rectanglesInGrid (n : ℕ) : ℕ := (choose n 2) ^ 2

theorem rectangles_in_5x5_grid :
  rectanglesInGrid 5 = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l1202_120221


namespace NUMINAMATH_CALUDE_max_m_value_l1202_120291

theorem max_m_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = x + 2 * y) :
  ∀ m : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → x * y = x + 2 * y → x * y ≥ m - 2) → m ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l1202_120291


namespace NUMINAMATH_CALUDE_natasha_exercise_time_l1202_120241

/-- Proves that Natasha exercised for 30 minutes daily given the conditions of the problem -/
theorem natasha_exercise_time :
  ∀ (d : ℕ),
  let natasha_daily_time : ℕ := 30
  let natasha_total_time : ℕ := d * natasha_daily_time
  let esteban_daily_time : ℕ := 10
  let esteban_days : ℕ := 9
  let esteban_total_time : ℕ := esteban_daily_time * esteban_days
  let total_exercise_time : ℕ := 5 * 60
  natasha_total_time + esteban_total_time = total_exercise_time →
  natasha_daily_time = 30 :=
by
  sorry

#check natasha_exercise_time

end NUMINAMATH_CALUDE_natasha_exercise_time_l1202_120241


namespace NUMINAMATH_CALUDE_cos_90_degrees_l1202_120274

theorem cos_90_degrees : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_90_degrees_l1202_120274


namespace NUMINAMATH_CALUDE_intersecting_lines_k_value_l1202_120231

/-- Given three lines that intersect at the same point, prove the value of k -/
theorem intersecting_lines_k_value (x y k : ℚ) : 
  (y = 6 * x + 4) ∧ 
  (y = -3 * x - 30) ∧ 
  (y = 4 * x + k) → 
  k = -32/9 := by sorry

end NUMINAMATH_CALUDE_intersecting_lines_k_value_l1202_120231


namespace NUMINAMATH_CALUDE_unique_solution_implies_negative_a_l1202_120208

theorem unique_solution_implies_negative_a :
  ∀ a : ℝ,
  (∃! x : ℝ, |x^2 - 1| = a * |x - 1|) →
  a < 0 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_implies_negative_a_l1202_120208


namespace NUMINAMATH_CALUDE_sum_of_u_and_v_l1202_120233

theorem sum_of_u_and_v (u v : ℚ) 
  (eq1 : 3 * u - 4 * v = 17) 
  (eq2 : 5 * u + 3 * v = -1) : 
  u + v = -41 / 29 := by
sorry

end NUMINAMATH_CALUDE_sum_of_u_and_v_l1202_120233


namespace NUMINAMATH_CALUDE_intersection_x_value_l1202_120254

/-- The x-coordinate of the intersection point of two lines -/
def intersection_x (m1 b1 a2 b2 c2 : ℚ) : ℚ :=
  (c2 - b2 + b1) / (m1 + a2)

/-- Theorem: The x-coordinate of the intersection point of y = 4x - 29 and 3x + y = 105 is 134/7 -/
theorem intersection_x_value :
  intersection_x 4 (-29) 3 1 105 = 134 / 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_x_value_l1202_120254


namespace NUMINAMATH_CALUDE_water_depth_calculation_l1202_120201

/-- The depth of water given Dean's height and a multiplier -/
def water_depth (dean_height : ℝ) (depth_multiplier : ℝ) : ℝ :=
  dean_height * depth_multiplier

/-- Theorem: The water depth is 60 feet when Dean's height is 6 feet
    and the depth is 10 times his height -/
theorem water_depth_calculation :
  water_depth 6 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_calculation_l1202_120201


namespace NUMINAMATH_CALUDE_min_value_sum_of_distances_min_value_achievable_l1202_120244

theorem min_value_sum_of_distances (x : ℝ) :
  Real.sqrt (x^2 + (1 - x)^2) + Real.sqrt ((1 - x)^2 + (1 + x)^2) ≥ Real.sqrt 5 :=
by sorry

theorem min_value_achievable :
  ∃ x : ℝ, Real.sqrt (x^2 + (1 - x)^2) + Real.sqrt ((1 - x)^2 + (1 + x)^2) = Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_distances_min_value_achievable_l1202_120244


namespace NUMINAMATH_CALUDE_pear_juice_percentage_approx_19_23_l1202_120219

/-- Represents the juice yield from fruits -/
structure JuiceYield where
  pears : ℕ
  pearJuice : ℚ
  oranges : ℕ
  orangeJuice : ℚ

/-- Represents the blend composition -/
structure Blend where
  pears : ℕ
  oranges : ℕ

/-- Calculates the percentage of pear juice in a blend -/
def pear_juice_percentage (yield : JuiceYield) (blend : Blend) : ℚ :=
  let pear_juice := (blend.pears : ℚ) * yield.pearJuice / yield.pears
  let orange_juice := (blend.oranges : ℚ) * yield.orangeJuice / yield.oranges
  let total_juice := pear_juice + orange_juice
  pear_juice / total_juice * 100

theorem pear_juice_percentage_approx_19_23 (yield : JuiceYield) (blend : Blend) :
  yield.pears = 4 ∧ 
  yield.pearJuice = 10 ∧ 
  yield.oranges = 1 ∧ 
  yield.orangeJuice = 7 ∧
  blend.pears = 8 ∧
  blend.oranges = 12 →
  abs (pear_juice_percentage yield blend - 19.23) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_pear_juice_percentage_approx_19_23_l1202_120219


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l1202_120251

theorem min_value_squared_sum (a b c d : ℝ) (h1 : a * b = 2) (h2 : c * d = 18) :
  (a * c)^2 + (b * d)^2 ≥ 12 ∧ ∃ (a' b' c' d' : ℝ), a' * b' = 2 ∧ c' * d' = 18 ∧ (a' * c')^2 + (b' * d')^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l1202_120251


namespace NUMINAMATH_CALUDE_total_rounded_to_nearest_dollar_l1202_120216

def purchase1 : ℚ := 245/100
def purchase2 : ℚ := 358/100
def purchase3 : ℚ := 796/100

def round_to_nearest_dollar (x : ℚ) : ℤ :=
  if x - x.floor < 1/2 then x.floor else x.ceil

theorem total_rounded_to_nearest_dollar :
  round_to_nearest_dollar (purchase1 + purchase2 + purchase3) = 14 := by
  sorry

end NUMINAMATH_CALUDE_total_rounded_to_nearest_dollar_l1202_120216


namespace NUMINAMATH_CALUDE_average_speed_two_hours_car_average_speed_l1202_120278

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) : 
  speed1 > 0 → speed2 > 0 → (speed1 + speed2) / 2 = (speed1 * 1 + speed2 * 1) / (1 + 1) := by
  sorry

/-- The average speed of a car traveling 90 km in the first hour and 30 km in the second hour is 60 km/h -/
theorem car_average_speed : 
  let speed1 : ℝ := 90
  let speed2 : ℝ := 30
  (speed1 + speed2) / 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_car_average_speed_l1202_120278


namespace NUMINAMATH_CALUDE_max_d_value_l1202_120214

def a (n : ℕ+) : ℕ := 103 + n^2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  ∃ (N : ℕ+), d N = 13 ∧ ∀ (n : ℕ+), d n ≤ 13 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l1202_120214


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l1202_120247

/-- Calculates the length of a bridge given the parameters of an elephant train passing through it. -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_cm : ℝ) (time_to_pass : ℝ) : 
  train_length = 15 →
  train_speed_cm = 275 →
  time_to_pass = 48 →
  (train_speed_cm / 100 * time_to_pass) - train_length = 117 := by
  sorry

#check bridge_length_calculation

end NUMINAMATH_CALUDE_bridge_length_calculation_l1202_120247


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l1202_120286

theorem quadratic_unique_solution (b d : ℤ) : 
  (∃! x : ℝ, b * x^2 + 24 * x + d = 0) →
  b + d = 41 →
  b < d →
  b = 9 ∧ d = 32 := by
sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l1202_120286


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l1202_120268

theorem arithmetic_simplification : (4 + 6 + 2) / 3 - 2 / 3 = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l1202_120268


namespace NUMINAMATH_CALUDE_soft_drink_storage_l1202_120252

theorem soft_drink_storage (initial_small : ℕ) (initial_big : ℕ) 
  (percent_big_sold : ℚ) (total_remaining : ℕ) :
  initial_small = 6000 →
  initial_big = 15000 →
  percent_big_sold = 14 / 100 →
  total_remaining = 18180 →
  ∃ (percent_small_sold : ℚ),
    percent_small_sold = 12 / 100 ∧
    (initial_small - initial_small * percent_small_sold) +
    (initial_big - initial_big * percent_big_sold) = total_remaining :=
by sorry

end NUMINAMATH_CALUDE_soft_drink_storage_l1202_120252


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l1202_120270

/-- The equation of the asymptote of a hyperbola -/
def asymptote_equation (a b : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | y = (b / a) * x ∨ y = -(b / a) * x}

/-- The focal length of a hyperbola -/
def focal_length (c : ℝ) : ℝ := 2 * c

theorem hyperbola_asymptote (b : ℝ) (h1 : b > 0) :
  let C : Set (ℝ × ℝ) := {(x, y) | x^2 / 16 - y^2 / b^2 = 1}
  let f : ℝ := focal_length 5
  asymptote_equation 4 3 = {(x, y) | y = (3 / 4) * x ∨ y = -(3 / 4) * x} :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l1202_120270


namespace NUMINAMATH_CALUDE_solve_equation_l1202_120218

theorem solve_equation (x y : ℝ) :
  3 * x - 5 * y = 7 → y = (3 * x - 7) / 5 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l1202_120218


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l1202_120205

theorem reciprocal_of_negative_fraction (n : ℤ) (h : n ≠ 0) :
  (-(1 : ℚ) / n)⁻¹ = -n := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l1202_120205


namespace NUMINAMATH_CALUDE_complex_product_real_l1202_120267

theorem complex_product_real (x : ℝ) : 
  let z₁ : ℂ := 2 + I
  let z₂ : ℂ := x - 2 * I
  (z₁ * z₂).im = 0 → x = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_product_real_l1202_120267


namespace NUMINAMATH_CALUDE_function_properties_l1202_120206

-- Define the function f(x) = x^3 + ax^2 + b
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := 3*x + y - 3 = 0

-- Theorem statement
theorem function_properties (a b : ℝ) :
  (tangent_line 1 (f a b 1)) →
  (∃ (f' : ℝ → ℝ), 
    (∀ x, f' x = 3*x^2 - 6*x) ∧
    (∀ x, x < 0 → (f' x > 0)) ∧
    (∀ x, 0 < x ∧ x < 2 → (f' x < 0)) ∧
    (∀ x, x > 2 → (f' x > 0))) ∧
  (∀ t, t > 0 →
    (t ≤ 2 → 
      (∀ x, x ∈ Set.Icc 0 t → f (-3) 2 x ≤ 2 ∧ f (-3) 2 t ≤ f (-3) 2 x) ∧
      f (-3) 2 t = t^3 - 3*t^2 + 2) ∧
    (2 < t ∧ t ≤ 3 →
      (∀ x, x ∈ Set.Icc 0 t → -2 ≤ f (-3) 2 x ∧ f (-3) 2 x ≤ 2)) ∧
    (t > 3 →
      (∀ x, x ∈ Set.Icc 0 t → -2 ≤ f (-3) 2 x ∧ f (-3) 2 x ≤ f (-3) 2 t) ∧
      f (-3) 2 t = t^3 - 3*t^2 + 2)) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l1202_120206


namespace NUMINAMATH_CALUDE_house_distances_l1202_120277

-- Define the positions of houses on a straight line
variable (A B V G : ℝ)

-- Define the distances between houses
def AB := |A - B|
def VG := |V - G|
def AG := |A - G|
def BV := |B - V|

-- State the theorem
theorem house_distances (h1 : AB = 600) (h2 : VG = 600) (h3 : AG = 3 * BV) :
  AG = 900 ∨ AG = 1800 := by
  sorry

end NUMINAMATH_CALUDE_house_distances_l1202_120277


namespace NUMINAMATH_CALUDE_sum_of_fractions_bound_l1202_120245

theorem sum_of_fractions_bound (x y z : ℝ) (h : |x*y*z| = 1) :
  (1 / (x^2 + x + 1) + 1 / (x^2 - x + 1)) +
  (1 / (y^2 + y + 1) + 1 / (y^2 - y + 1)) +
  (1 / (z^2 + z + 1) + 1 / (z^2 - z + 1)) ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_bound_l1202_120245


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_sum_of_reciprocals_of_roots_l1202_120222

-- Define the coefficients of the first equation: 2x^2 - 5x + 1 = 0
def a₁ : ℚ := 2
def b₁ : ℚ := -5
def c₁ : ℚ := 1

-- Define the coefficients of the second equation: 2x^2 - 11x + 13 = 0
def a₂ : ℚ := 2
def b₂ : ℚ := -11
def c₂ : ℚ := 13

-- Theorem for the sum of cubes of roots
theorem sum_of_cubes_of_roots :
  let x₁ := (-b₁ + Real.sqrt (b₁^2 - 4*a₁*c₁)) / (2*a₁)
  let x₂ := (-b₁ - Real.sqrt (b₁^2 - 4*a₁*c₁)) / (2*a₁)
  x₁^3 + x₂^3 = 95/8 := by sorry

-- Theorem for the sum of reciprocals of roots
theorem sum_of_reciprocals_of_roots :
  let y₁ := (-b₂ + Real.sqrt (b₂^2 - 4*a₂*c₂)) / (2*a₂)
  let y₂ := (-b₂ - Real.sqrt (b₂^2 - 4*a₂*c₂)) / (2*a₂)
  y₁/y₂ + y₂/y₁ = 69/26 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_sum_of_reciprocals_of_roots_l1202_120222


namespace NUMINAMATH_CALUDE_david_investment_time_l1202_120257

/-- Simple interest calculation -/
def simpleInterest (principal time rate : ℝ) : ℝ :=
  principal * (1 + time * rate)

theorem david_investment_time :
  ∀ (rate : ℝ),
  rate > 0 →
  simpleInterest 710 3 rate = 815 →
  simpleInterest 710 4 rate = 850 :=
by
  sorry

end NUMINAMATH_CALUDE_david_investment_time_l1202_120257


namespace NUMINAMATH_CALUDE_candy_remaining_l1202_120228

theorem candy_remaining (initial : ℕ) (first_eaten : ℕ) (second_eaten : ℕ) 
  (h1 : initial = 21)
  (h2 : first_eaten = 5)
  (h3 : second_eaten = 9) : 
  initial - first_eaten - second_eaten = 7 := by
  sorry

end NUMINAMATH_CALUDE_candy_remaining_l1202_120228


namespace NUMINAMATH_CALUDE_vector_linear_combination_l1202_120240

/-- Given two planar vectors a and b, prove that their linear combination results in the specified vector. -/
theorem vector_linear_combination (a b : ℝ × ℝ) :
  a = (1, 1) → b = (1, -1) → (1/2 : ℝ) • a - (3/2 : ℝ) • b = (-1, 2) := by sorry

end NUMINAMATH_CALUDE_vector_linear_combination_l1202_120240


namespace NUMINAMATH_CALUDE_timePerPlayer_is_36_l1202_120238

/-- Represents a sports tournament with given parameters -/
structure Tournament where
  teamSize : ℕ
  playersOnField : ℕ
  matchDuration : ℕ
  hTeamSize : teamSize = 10
  hPlayersOnField : playersOnField = 8
  hMatchDuration : matchDuration = 45
  hPlayersOnFieldLessTeamSize : playersOnField < teamSize

/-- Calculates the time each player spends on the field -/
def timePerPlayer (t : Tournament) : ℕ :=
  t.playersOnField * t.matchDuration / t.teamSize

/-- Theorem stating that each player spends 36 minutes on the field -/
theorem timePerPlayer_is_36 (t : Tournament) : timePerPlayer t = 36 := by
  sorry

end NUMINAMATH_CALUDE_timePerPlayer_is_36_l1202_120238


namespace NUMINAMATH_CALUDE_intersection_point_translated_line_l1202_120262

/-- The intersection point of the line y = 3x + 6 with the x-axis is (-2, 0) -/
theorem intersection_point_translated_line (x y : ℝ) :
  y = 3 * x + 6 ∧ y = 0 → x = -2 ∧ y = 0 := by sorry

end NUMINAMATH_CALUDE_intersection_point_translated_line_l1202_120262


namespace NUMINAMATH_CALUDE_midpoint_vector_sum_l1202_120230

-- Define the triangle ABC and its midpoints
variable (A B C D E F : ℝ × ℝ)

-- Define the conditions
axiom D_midpoint : D = (A + B) / 2
axiom E_midpoint : E = (B + C) / 2
axiom F_midpoint : F = (C + A) / 2

-- Define vector operations
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

-- State the theorem
theorem midpoint_vector_sum :
  vec D B - vec D E + vec C F = vec C D :=
sorry

end NUMINAMATH_CALUDE_midpoint_vector_sum_l1202_120230


namespace NUMINAMATH_CALUDE_total_spent_is_64_l1202_120295

/-- The total amount spent by Victor and his friend on trick decks -/
def total_spent (deck_price : ℕ) (victor_decks : ℕ) (friend_decks : ℕ) : ℕ :=
  deck_price * (victor_decks + friend_decks)

/-- Proof that Victor and his friend spent $64 in total -/
theorem total_spent_is_64 :
  total_spent 8 6 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_64_l1202_120295


namespace NUMINAMATH_CALUDE_investment_change_investment_change_specific_l1202_120207

theorem investment_change (initial_investment : ℝ) 
                          (first_year_loss_percent : ℝ) 
                          (second_year_gain_percent : ℝ) : ℝ :=
  let first_year_amount := initial_investment * (1 - first_year_loss_percent / 100)
  let second_year_amount := first_year_amount * (1 + second_year_gain_percent / 100)
  let total_change_percent := (second_year_amount - initial_investment) / initial_investment * 100
  total_change_percent

theorem investment_change_specific : 
  investment_change 200 10 25 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_investment_change_investment_change_specific_l1202_120207


namespace NUMINAMATH_CALUDE_triangle_properties_l1202_120299

open Real

theorem triangle_properties (A B C a b c : ℝ) : 
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  a > 0 → b > 0 → c > 0 →
  2 * a * cos C + c = 2 * b →
  a = Real.sqrt 3 →
  (1 / 2) * b * c * sin A = Real.sqrt 3 / 2 →
  A = π / 3 ∧ a + b + c = 3 + Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l1202_120299


namespace NUMINAMATH_CALUDE_round_robin_tournament_sessions_l1202_120204

/-- The number of matches in a round-robin tournament with n players -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The minimum number of sessions required for a tournament -/
def min_sessions (total_matches : ℕ) (max_per_session : ℕ) : ℕ :=
  (total_matches + max_per_session - 1) / max_per_session

theorem round_robin_tournament_sessions :
  let n : ℕ := 10  -- number of players
  let max_per_session : ℕ := 15  -- maximum matches per session
  min_sessions (num_matches n) max_per_session = 3 := by
  sorry

end NUMINAMATH_CALUDE_round_robin_tournament_sessions_l1202_120204


namespace NUMINAMATH_CALUDE_card_area_problem_l1202_120266

theorem card_area_problem (length width : ℝ) 
  (h1 : length = 4 ∧ width = 6)
  (h2 : (length - 1) * width = 18 ∨ length * (width - 1) = 18) :
  (if (length - 1) * width = 18 
   then length * (width - 1) 
   else (length - 1) * width) = 20 := by
  sorry

end NUMINAMATH_CALUDE_card_area_problem_l1202_120266


namespace NUMINAMATH_CALUDE_inequality_range_l1202_120235

theorem inequality_range (a : ℝ) : (∀ x : ℝ, |x + 1| + |x - 3| ≥ a) ↔ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_inequality_range_l1202_120235


namespace NUMINAMATH_CALUDE_base_five_to_decimal_l1202_120298

/-- Converts a list of digits in a given base to its decimal (base 10) representation -/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ (digits.length - 1 - i)) 0

/-- The base 5 number 243₅ is equal to 73 in base 10 -/
theorem base_five_to_decimal : to_decimal [2, 4, 3] 5 = 73 := by
  sorry

end NUMINAMATH_CALUDE_base_five_to_decimal_l1202_120298


namespace NUMINAMATH_CALUDE_petya_win_probability_l1202_120209

/-- Represents the number of stones a player can take in one turn -/
inductive StonesPerTurn
  | one
  | two
  | three
  | four

/-- Represents a player in the game -/
inductive Player
  | petya
  | computer

/-- Represents the state of the game -/
structure GameState where
  stones : Nat
  turn : Player

/-- The initial state of the game -/
def initialState : GameState :=
  { stones := 16, turn := Player.petya }

/-- Represents the strategy of a player -/
def Strategy := GameState → StonesPerTurn

/-- Petya's random strategy -/
def petyaStrategy : Strategy :=
  fun _ => sorry -- Randomly choose between 1 and 4 stones

/-- Computer's optimal strategy -/
def computerStrategy : Strategy :=
  fun _ => sorry -- Always choose the optimal number of stones

/-- The probability of Petya winning the game -/
def petyaWinProbability : ℚ :=
  1 / 256

/-- Theorem stating that Petya's win probability is 1/256 -/
theorem petya_win_probability :
  petyaWinProbability = 1 / 256 := by sorry


end NUMINAMATH_CALUDE_petya_win_probability_l1202_120209


namespace NUMINAMATH_CALUDE_intersecting_segments_l1202_120280

/-- Given two intersecting line segments PQ and RS, prove that x + y = 145 -/
theorem intersecting_segments (x y : ℝ) : 
  (60 + (y + 5) = 180) →  -- Linear pair on PQ
  (4 * x = y + 5) →       -- Vertically opposite angles
  x + y = 145 := by sorry

end NUMINAMATH_CALUDE_intersecting_segments_l1202_120280
