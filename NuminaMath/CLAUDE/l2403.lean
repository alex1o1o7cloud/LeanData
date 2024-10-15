import Mathlib

namespace NUMINAMATH_CALUDE_division_438_by_4_result_l2403_240394

/-- Represents the place value of a digit in a number -/
inductive PlaceValue
  | Ones
  | Tens
  | Hundreds
  | Thousands

/-- Represents a division operation with its result -/
structure DivisionResult (dividend : ℕ) (divisor : ℕ) where
  quotient : ℕ
  remainder : ℕ
  highest_place_value : PlaceValue
  valid : dividend = quotient * divisor + remainder
  remainder_bound : remainder < divisor

/-- The division of 438 by 4 -/
def division_438_by_4 : DivisionResult 438 4 := sorry

theorem division_438_by_4_result :
  division_438_by_4.highest_place_value = PlaceValue.Hundreds ∧
  division_438_by_4.remainder = 2 := by sorry

end NUMINAMATH_CALUDE_division_438_by_4_result_l2403_240394


namespace NUMINAMATH_CALUDE_brownies_problem_l2403_240350

theorem brownies_problem (total_brownies : ℕ) (tina_per_day : ℕ) (husband_per_day : ℕ) 
  (shared : ℕ) (left : ℕ) :
  total_brownies = 24 →
  tina_per_day = 2 →
  husband_per_day = 1 →
  shared = 4 →
  left = 5 →
  ∃ (days : ℕ), days = 5 ∧ 
    total_brownies = days * (tina_per_day + husband_per_day) + shared + left :=
by sorry

end NUMINAMATH_CALUDE_brownies_problem_l2403_240350


namespace NUMINAMATH_CALUDE_bowling_team_weight_l2403_240353

theorem bowling_team_weight (initial_players : ℕ) (initial_avg : ℝ) 
  (new_player1_weight : ℝ) (new_avg : ℝ) :
  initial_players = 7 →
  initial_avg = 103 →
  new_player1_weight = 110 →
  new_avg = 99 →
  ∃ (new_player2_weight : ℝ),
    (initial_players * initial_avg + new_player1_weight + new_player2_weight) / 
    (initial_players + 2) = new_avg ∧
    new_player2_weight = 60 := by
  sorry

end NUMINAMATH_CALUDE_bowling_team_weight_l2403_240353


namespace NUMINAMATH_CALUDE_annual_turbans_count_l2403_240363

/-- Represents the annual salary structure and partial payment details --/
structure SalaryInfo where
  annual_cash : ℕ  -- Annual cash component in Rupees
  turban_price : ℕ  -- Price of one turban in Rupees
  partial_months : ℕ  -- Number of months worked
  partial_cash : ℕ  -- Cash received for partial work in Rupees
  partial_turbans : ℕ  -- Number of turbans received for partial work

/-- Calculates the number of turbans in the annual salary --/
def calculate_annual_turbans (info : SalaryInfo) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that the number of turbans in the annual salary is 1 --/
theorem annual_turbans_count (info : SalaryInfo) 
  (h1 : info.annual_cash = 90)
  (h2 : info.turban_price = 50)
  (h3 : info.partial_months = 9)
  (h4 : info.partial_cash = 55)
  (h5 : info.partial_turbans = 1) :
  calculate_annual_turbans info = 1 := by
  sorry

end NUMINAMATH_CALUDE_annual_turbans_count_l2403_240363


namespace NUMINAMATH_CALUDE_simplify_expression_l2403_240360

theorem simplify_expression : 4 * (15 / 7) * (21 / (-45)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2403_240360


namespace NUMINAMATH_CALUDE_regular_polygon_diagonals_l2403_240326

/-- A regular polygon with exterior angles measuring 60° has 9 diagonals -/
theorem regular_polygon_diagonals :
  ∀ (n : ℕ),
  (360 / n = 60) →  -- Each exterior angle measures 60°
  (n * (n - 3)) / 2 = 9  -- Number of diagonals
  := by sorry

end NUMINAMATH_CALUDE_regular_polygon_diagonals_l2403_240326


namespace NUMINAMATH_CALUDE_bus_meeting_time_l2403_240388

structure BusJourney where
  totalDistance : ℝ
  distanceToCountyTown : ℝ
  bus1DepartureTime : ℝ
  bus1ArrivalCountyTown : ℝ
  bus1StopTime : ℝ
  bus1ArrivalProvincialCapital : ℝ
  bus2DepartureTime : ℝ
  bus2Speed : ℝ

def meetingTime (j : BusJourney) : ℝ := sorry

theorem bus_meeting_time (j : BusJourney) 
  (h1 : j.totalDistance = 189)
  (h2 : j.distanceToCountyTown = 54)
  (h3 : j.bus1DepartureTime = 8.5)
  (h4 : j.bus1ArrivalCountyTown = 9.25)
  (h5 : j.bus1StopTime = 0.25)
  (h6 : j.bus1ArrivalProvincialCapital = 11)
  (h7 : j.bus2DepartureTime = 9)
  (h8 : j.bus2Speed = 60) :
  meetingTime j = 72 / 60 := by sorry

end NUMINAMATH_CALUDE_bus_meeting_time_l2403_240388


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2403_240302

theorem min_value_of_expression (x : ℝ) (h : x > 3) :
  x + 4 / (x - 3) ≥ 7 ∧ (x + 4 / (x - 3) = 7 ↔ x = 5) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2403_240302


namespace NUMINAMATH_CALUDE_rhombus_area_with_diagonals_6_and_8_l2403_240343

/-- The area of a rhombus with diagonals of lengths 6 and 8 is 24. -/
theorem rhombus_area_with_diagonals_6_and_8 : 
  ∀ (r : ℝ × ℝ → ℝ), 
  (∀ d₁ d₂, r (d₁, d₂) = (1/2) * d₁ * d₂) →
  r (6, 8) = 24 := by
sorry

end NUMINAMATH_CALUDE_rhombus_area_with_diagonals_6_and_8_l2403_240343


namespace NUMINAMATH_CALUDE_expression_evaluation_l2403_240339

theorem expression_evaluation (a b : ℤ) (h1 : a = 4) (h2 : b = -1) :
  -2*a - b^2 + 2*a*b = -17 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2403_240339


namespace NUMINAMATH_CALUDE_negation_equivalence_l2403_240354

-- Define the original proposition
def original_proposition : Prop := ∃ x : ℝ, Real.exp x - x - 2 ≤ 0

-- Define the negation of the proposition
def negation_proposition : Prop := ∀ x : ℝ, Real.exp x - x - 2 > 0

-- Theorem stating the equivalence between the negation of the original proposition
-- and the negation_proposition
theorem negation_equivalence : 
  (¬ original_proposition) ↔ negation_proposition :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2403_240354


namespace NUMINAMATH_CALUDE_equation_solution_l2403_240370

theorem equation_solution : ∃! x : ℝ, (128 : ℝ)^(x - 1) / (16 : ℝ)^(x - 1) = (64 : ℝ)^(3 * x) ∧ x = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2403_240370


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l2403_240364

/-- 
Given a geometric sequence {a_n} with positive terms and common ratio q > 1,
if a_5 + a_4 - a_3 - a_2 = 5, then a_6 + a_7 ≥ 20.
-/
theorem geometric_sequence_minimum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  q > 1 →
  (∀ n, a (n + 1) = q * a n) →
  a 5 + a 4 - a 3 - a 2 = 5 →
  a 6 + a 7 ≥ 20 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l2403_240364


namespace NUMINAMATH_CALUDE_white_to_brown_dog_weight_ratio_l2403_240304

def brown_dog_weight : ℝ := 4
def black_dog_weight : ℝ := brown_dog_weight + 1
def grey_dog_weight : ℝ := black_dog_weight - 2
def average_weight : ℝ := 5
def num_dogs : ℕ := 4

def white_dog_weight : ℝ := average_weight * num_dogs - (brown_dog_weight + black_dog_weight + grey_dog_weight)

theorem white_to_brown_dog_weight_ratio :
  white_dog_weight / brown_dog_weight = 2 := by
  sorry

end NUMINAMATH_CALUDE_white_to_brown_dog_weight_ratio_l2403_240304


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l2403_240374

theorem min_value_sum_of_reciprocals (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) :
  1 / (a^2 + 2*b^2) + 1 / (b^2 + 2*c^2) + 1 / (c^2 + 2*a^2) ≥ 9 ∧
  (1 / (a^2 + 2*b^2) + 1 / (b^2 + 2*c^2) + 1 / (c^2 + 2*a^2) = 9 ↔ a = 1/3 ∧ b = 1/3 ∧ c = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l2403_240374


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2403_240352

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x - 1
  ∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 5 ∧ x₂ = 2 - Real.sqrt 5 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2403_240352


namespace NUMINAMATH_CALUDE_chris_candy_distribution_l2403_240311

/-- The number of friends Chris has -/
def num_friends : ℕ := 35

/-- The number of candy pieces each friend receives -/
def candy_per_friend : ℕ := 12

/-- The total number of candy pieces Chris gave to his friends -/
def total_candy : ℕ := num_friends * candy_per_friend

theorem chris_candy_distribution :
  total_candy = 420 :=
by sorry

end NUMINAMATH_CALUDE_chris_candy_distribution_l2403_240311


namespace NUMINAMATH_CALUDE_quadratic_equation_transformation_l2403_240348

theorem quadratic_equation_transformation (x : ℝ) :
  x^2 + 6*x - 1 = 0 →
  ∃ (m n : ℝ), (x + m)^2 = n ∧ m - n = -7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_transformation_l2403_240348


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2403_240365

theorem imaginary_part_of_z (θ : ℝ) : 
  let z : ℂ := Complex.mk (Real.sin (2 * θ) - 1) (Real.sqrt 2 * Real.cos θ - 1)
  (z.re = 0 ∧ z.im ≠ 0) → z.im = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2403_240365


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l2403_240307

theorem matrix_equation_solution :
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -8; 9, 3]
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![46/7, -58/7; -39/14, 51/14]
  N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l2403_240307


namespace NUMINAMATH_CALUDE_child_ticket_cost_l2403_240305

/-- Proves that the cost of each child's ticket is $7 -/
theorem child_ticket_cost (num_adults num_children : ℕ) (concession_cost total_cost adult_ticket_cost : ℚ) :
  num_adults = 5 →
  num_children = 2 →
  concession_cost = 12 →
  total_cost = 76 →
  adult_ticket_cost = 10 →
  (total_cost - concession_cost - num_adults * adult_ticket_cost) / num_children = 7 :=
by sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l2403_240305


namespace NUMINAMATH_CALUDE_product_b_original_price_l2403_240335

theorem product_b_original_price 
  (price_a : ℝ) 
  (price_b : ℝ) 
  (initial_relation : price_a = 1.2 * price_b)
  (price_a_after : ℝ)
  (price_a_decrease : price_a_after = 0.9 * price_a)
  (price_a_final : price_a_after = 198)
  : price_b = 183.33 := by
  sorry

end NUMINAMATH_CALUDE_product_b_original_price_l2403_240335


namespace NUMINAMATH_CALUDE_min_vertical_distance_l2403_240380

/-- The minimum vertical distance between y = |x-1| and y = -x^2 - 5x - 6 is 4 -/
theorem min_vertical_distance : ∃ (d : ℝ), d = 4 ∧ 
  ∀ (x : ℝ), d ≤ |x - 1| - (-x^2 - 5*x - 6) :=
by sorry

end NUMINAMATH_CALUDE_min_vertical_distance_l2403_240380


namespace NUMINAMATH_CALUDE_lunchroom_tables_l2403_240372

theorem lunchroom_tables (total_students : ℕ) (students_per_table : ℕ) (h1 : total_students = 204) (h2 : students_per_table = 6) :
  total_students / students_per_table = 34 := by
  sorry

end NUMINAMATH_CALUDE_lunchroom_tables_l2403_240372


namespace NUMINAMATH_CALUDE_interest_rate_for_doubling_l2403_240393

/-- The time in years for the money to double --/
def doubling_time : ℝ := 4

/-- The interest rate as a decimal --/
def interest_rate : ℝ := 0.25

/-- Simple interest formula: Final amount = Principal * (1 + rate * time) --/
def simple_interest (principal rate time : ℝ) : ℝ := principal * (1 + rate * time)

theorem interest_rate_for_doubling :
  simple_interest 1 interest_rate doubling_time = 2 := by sorry

end NUMINAMATH_CALUDE_interest_rate_for_doubling_l2403_240393


namespace NUMINAMATH_CALUDE_library_books_checkout_l2403_240310

theorem library_books_checkout (fiction_books : ℕ) (nonfiction_ratio fiction_ratio : ℕ) : 
  fiction_books = 24 → 
  nonfiction_ratio = 7 →
  fiction_ratio = 6 →
  ∃ (total_books : ℕ), total_books = fiction_books + (fiction_books * nonfiction_ratio) / fiction_ratio ∧ total_books = 52 :=
by
  sorry

end NUMINAMATH_CALUDE_library_books_checkout_l2403_240310


namespace NUMINAMATH_CALUDE_chef_michel_pies_l2403_240332

/-- Represents the number of pieces a shepherd's pie is cut into -/
def shepherds_pie_pieces : ℕ := 4

/-- Represents the number of pieces a chicken pot pie is cut into -/
def chicken_pot_pie_pieces : ℕ := 5

/-- Represents the number of customers who ordered shepherd's pie slices -/
def shepherds_pie_customers : ℕ := 52

/-- Represents the number of customers who ordered chicken pot pie slices -/
def chicken_pot_pie_customers : ℕ := 80

/-- Calculates the total number of pies sold by Chef Michel -/
def total_pies_sold : ℕ := 
  (shepherds_pie_customers / shepherds_pie_pieces) + 
  (chicken_pot_pie_customers / chicken_pot_pie_pieces)

theorem chef_michel_pies : total_pies_sold = 29 := by
  sorry

end NUMINAMATH_CALUDE_chef_michel_pies_l2403_240332


namespace NUMINAMATH_CALUDE_percentage_spent_is_80_percent_l2403_240340

-- Define the costs and money amounts
def cheeseburger_cost : ℚ := 3
def milkshake_cost : ℚ := 5
def cheese_fries_cost : ℚ := 8
def jim_money : ℚ := 20
def cousin_money : ℚ := 10

-- Define the total cost of the meal
def total_cost : ℚ := 2 * cheeseburger_cost + 2 * milkshake_cost + cheese_fries_cost

-- Define the combined money
def combined_money : ℚ := jim_money + cousin_money

-- Theorem to prove
theorem percentage_spent_is_80_percent :
  (total_cost / combined_money) * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_spent_is_80_percent_l2403_240340


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_3136_l2403_240321

theorem largest_prime_factor_of_3136 (p : Nat) : 
  Nat.Prime p ∧ p ∣ 3136 → p ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_3136_l2403_240321


namespace NUMINAMATH_CALUDE_no_natural_solution_l2403_240384

theorem no_natural_solution :
  ¬ ∃ (x y z t : ℕ), 16^x + 21^y + 26^z = t^2 := by
sorry

end NUMINAMATH_CALUDE_no_natural_solution_l2403_240384


namespace NUMINAMATH_CALUDE_reflection_line_sum_l2403_240331

/-- Given a point and its image under reflection across a line, prove the sum of the line's slope and y-intercept. -/
theorem reflection_line_sum (x₁ y₁ x₂ y₂ : ℝ) (m b : ℝ) 
  (h₁ : (x₁, y₁) = (2, 3))  -- Original point
  (h₂ : (x₂, y₂) = (10, 7))  -- Image point
  (h₃ : ∀ x y, y = m * x + b →  -- Reflection line equation
              (x - x₁) * (x - x₂) + (y - y₁) * (y - y₂) = 0) :
  m + b = 15 := by sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l2403_240331


namespace NUMINAMATH_CALUDE_equation_solution_l2403_240377

theorem equation_solution : ∃ x₁ x₂ : ℝ,
  (1 / (x₁ + 10) + 1 / (x₁ + 8) = 1 / (x₁ + 11) + 1 / (x₁ + 7) + 1 / (2 * x₁ + 36)) ∧
  (1 / (x₂ + 10) + 1 / (x₂ + 8) = 1 / (x₂ + 11) + 1 / (x₂ + 7) + 1 / (2 * x₂ + 36)) ∧
  (5 * x₁^2 + 140 * x₁ + 707 = 0) ∧
  (5 * x₂^2 + 140 * x₂ + 707 = 0) ∧
  x₁ ≠ x₂ :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2403_240377


namespace NUMINAMATH_CALUDE_cos_a_minus_b_l2403_240309

theorem cos_a_minus_b (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3/2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5/8 := by
sorry

end NUMINAMATH_CALUDE_cos_a_minus_b_l2403_240309


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2403_240371

theorem negation_of_proposition (p : Prop) : 
  (¬ (∀ x : ℝ, x^3 - x^2 + 1 < 0)) ↔ (∃ x : ℝ, x^3 - x^2 + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2403_240371


namespace NUMINAMATH_CALUDE_gerald_expense_l2403_240369

/-- Represents Gerald's baseball supplies expense situation -/
structure BaseballExpenses where
  season_length : ℕ
  saving_months : ℕ
  chore_price : ℕ
  chores_per_month : ℕ

/-- Calculates the monthly expense for baseball supplies -/
def monthly_expense (e : BaseballExpenses) : ℕ :=
  (e.saving_months * e.chores_per_month * e.chore_price) / e.season_length

/-- Theorem: Given Gerald's specific situation, his monthly expense is $100 -/
theorem gerald_expense :
  let e : BaseballExpenses := {
    season_length := 4,
    saving_months := 8,
    chore_price := 10,
    chores_per_month := 5
  }
  monthly_expense e = 100 := by sorry

end NUMINAMATH_CALUDE_gerald_expense_l2403_240369


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_eighteen_l2403_240362

theorem sum_of_roots_equals_eighteen : 
  let f (x : ℝ) := (3 * x^3 + 2 * x^2 - 9 * x + 15) - (4 * x^3 - 16 * x^2 + 27)
  ∃ (r₁ r₂ r₃ : ℝ), (∀ x, f x = 0 ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃)) ∧ r₁ + r₂ + r₃ = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_eighteen_l2403_240362


namespace NUMINAMATH_CALUDE_job_completion_time_l2403_240342

theorem job_completion_time 
  (m d r : ℕ) 
  (h1 : m > 0) 
  (h2 : d > 0) 
  (h3 : m + r > 0) : 
  (m * d : ℚ) / (m + r) = (m * d : ℕ) / (m + r) := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2403_240342


namespace NUMINAMATH_CALUDE_pie_to_bar_representation_l2403_240330

-- Define the structure of a pie chart
structure PieChart :=
  (section1 : ℝ)
  (section2 : ℝ)
  (section3 : ℝ)

-- Define the structure of a bar graph
structure BarGraph :=
  (bar1 : ℝ)
  (bar2 : ℝ)
  (bar3 : ℝ)

-- Define the conditions of the pie chart
def validPieChart (p : PieChart) : Prop :=
  p.section1 = p.section2 ∧ p.section3 = p.section1 + p.section2

-- Define the correct bar graph representation
def correctBarGraph (p : PieChart) (b : BarGraph) : Prop :=
  b.bar1 = b.bar2 ∧ b.bar3 = b.bar1 + b.bar2

-- Theorem: For a valid pie chart, there exists a correct bar graph representation
theorem pie_to_bar_representation (p : PieChart) (h : validPieChart p) :
  ∃ b : BarGraph, correctBarGraph p b :=
sorry

end NUMINAMATH_CALUDE_pie_to_bar_representation_l2403_240330


namespace NUMINAMATH_CALUDE_polygon_diagonals_sides_l2403_240345

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A polygon has 33 more diagonals than sides if and only if it has 11 sides -/
theorem polygon_diagonals_sides (n : ℕ) : diagonals n = n + 33 ↔ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_sides_l2403_240345


namespace NUMINAMATH_CALUDE_alligator_growth_in_year_l2403_240391

def initial_population : ℝ := 4
def growth_factor : ℝ := 1.5
def months : ℕ := 12

def alligator_population (t : ℕ) : ℝ :=
  initial_population * growth_factor ^ t

theorem alligator_growth_in_year :
  alligator_population months = 518.9853515625 :=
sorry

end NUMINAMATH_CALUDE_alligator_growth_in_year_l2403_240391


namespace NUMINAMATH_CALUDE_last_red_ball_fourth_draw_probability_l2403_240334

def initial_white_balls : ℕ := 8
def initial_red_balls : ℕ := 2
def total_balls : ℕ := initial_white_balls + initial_red_balls
def draws : ℕ := 4

def favorable_outcomes : ℕ := (Nat.choose 3 1) * (Nat.choose initial_white_balls 2)
def total_outcomes : ℕ := Nat.choose total_balls draws

theorem last_red_ball_fourth_draw_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_last_red_ball_fourth_draw_probability_l2403_240334


namespace NUMINAMATH_CALUDE_number_of_scooters_l2403_240322

/-- Represents the number of wheels on a vehicle -/
def wheels (vehicle : String) : ℕ :=
  match vehicle with
  | "bicycle" => 2
  | "tricycle" => 3
  | "scooter" => 2
  | _ => 0

/-- The total number of vehicles -/
def total_vehicles : ℕ := 10

/-- The total number of wheels -/
def total_wheels : ℕ := 26

/-- Proves that the number of scooters is 2 -/
theorem number_of_scooters :
  ∃ (b t s : ℕ),
    b + t + s = total_vehicles ∧
    b * wheels "bicycle" + t * wheels "tricycle" + s * wheels "scooter" = total_wheels ∧
    s = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_of_scooters_l2403_240322


namespace NUMINAMATH_CALUDE_base6_divisibility_by_13_l2403_240328

/-- Converts a base-6 number of the form 3dd4₆ to base 10 --/
def base6ToBase10 (d : ℕ) : ℕ := 3 * 6^3 + d * 6^2 + d * 6^1 + 4 * 6^0

/-- Checks if a natural number is a valid base-6 digit --/
def isBase6Digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 5

theorem base6_divisibility_by_13 :
  ∀ d : ℕ, isBase6Digit d → (base6ToBase10 d % 13 = 0 ↔ d = 4) := by sorry

end NUMINAMATH_CALUDE_base6_divisibility_by_13_l2403_240328


namespace NUMINAMATH_CALUDE_honey_jar_problem_l2403_240347

/-- Represents the process of drawing out honey and replacing with sugar solution --/
def draw_and_replace (initial_honey : ℝ) (percent : ℝ) : ℝ :=
  initial_honey * (1 - percent)

/-- The amount of honey remaining after three iterations --/
def remaining_honey (initial_honey : ℝ) : ℝ :=
  draw_and_replace (draw_and_replace (draw_and_replace initial_honey 0.3) 0.4) 0.5

/-- Theorem stating that if 315 grams of honey remain after the process, 
    the initial amount was 1500 grams --/
theorem honey_jar_problem (initial_honey : ℝ) :
  remaining_honey initial_honey = 315 → initial_honey = 1500 := by
  sorry

end NUMINAMATH_CALUDE_honey_jar_problem_l2403_240347


namespace NUMINAMATH_CALUDE_tan_2x_geq_1_solution_set_l2403_240320

theorem tan_2x_geq_1_solution_set :
  {x : ℝ | Real.tan (2 * x) ≥ 1} = {x : ℝ | ∃ k : ℤ, k * Real.pi / 2 + Real.pi / 8 ≤ x ∧ x < k * Real.pi / 2 + Real.pi / 4} :=
by sorry

end NUMINAMATH_CALUDE_tan_2x_geq_1_solution_set_l2403_240320


namespace NUMINAMATH_CALUDE_cube_five_minus_thirteen_equals_square_six_plus_seventysix_l2403_240379

theorem cube_five_minus_thirteen_equals_square_six_plus_seventysix :
  5^3 - 13 = 6^2 + 76 := by
  sorry

end NUMINAMATH_CALUDE_cube_five_minus_thirteen_equals_square_six_plus_seventysix_l2403_240379


namespace NUMINAMATH_CALUDE_tree_height_after_two_years_l2403_240344

/-- Given a tree that triples its height every year and reaches 81 feet after 4 years,
    this function calculates its height after a given number of years. -/
def tree_height (years : ℕ) : ℚ :=
  81 / (3 ^ (4 - years))

/-- Theorem stating that the height of the tree after 2 years is 9 feet. -/
theorem tree_height_after_two_years :
  tree_height 2 = 9 := by sorry

end NUMINAMATH_CALUDE_tree_height_after_two_years_l2403_240344


namespace NUMINAMATH_CALUDE_max_area_rectangle_perimeter_24_l2403_240390

/-- The maximum area of a rectangle with perimeter 24 is 36 -/
theorem max_area_rectangle_perimeter_24 :
  ∀ (length width : ℝ), length > 0 → width > 0 →
  2 * (length + width) = 24 →
  length * width ≤ 36 := by
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_perimeter_24_l2403_240390


namespace NUMINAMATH_CALUDE_geometric_series_r_value_l2403_240337

/-- Given a geometric series with first term a and common ratio r,
    S is the sum of the entire series,
    S_odd is the sum of terms with odd powers of r -/
def geometric_series (a r : ℝ) (S S_odd : ℝ) : Prop :=
  ∃ (n : ℕ), S = a * (1 - r^n) / (1 - r) ∧
             S_odd = a * r * (1 - r^(2*n)) / (1 - r^2)

theorem geometric_series_r_value (a r : ℝ) :
  geometric_series a r 20 8 → r = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_r_value_l2403_240337


namespace NUMINAMATH_CALUDE_kindergarten_class_size_l2403_240317

theorem kindergarten_class_size 
  (num_groups : ℕ) 
  (time_per_student : ℕ) 
  (time_per_group : ℕ) 
  (h1 : num_groups = 3)
  (h2 : time_per_student = 4)
  (h3 : time_per_group = 24) :
  num_groups * (time_per_group / time_per_student) = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_kindergarten_class_size_l2403_240317


namespace NUMINAMATH_CALUDE_annual_interest_income_l2403_240386

/-- Calculates the annual interest income from two municipal bonds -/
theorem annual_interest_income
  (total_investment : ℝ)
  (rate1 rate2 : ℝ)
  (investment1 : ℝ)
  (h1 : total_investment = 32000)
  (h2 : rate1 = 0.0575)
  (h3 : rate2 = 0.0625)
  (h4 : investment1 = 20000)
  (h5 : investment1 < total_investment) :
  investment1 * rate1 + (total_investment - investment1) * rate2 = 1900 := by
  sorry

end NUMINAMATH_CALUDE_annual_interest_income_l2403_240386


namespace NUMINAMATH_CALUDE_pencils_in_drawer_l2403_240329

/-- The total number of pencils after adding more -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: The total number of pencils is 71 -/
theorem pencils_in_drawer : total_pencils 41 30 = 71 := by
  sorry

end NUMINAMATH_CALUDE_pencils_in_drawer_l2403_240329


namespace NUMINAMATH_CALUDE_rain_probability_theorem_l2403_240398

/-- Given probabilities for rain events in counties -/
theorem rain_probability_theorem 
  (p_monday : ℝ) 
  (p_neither : ℝ) 
  (p_both : ℝ) 
  (h1 : p_monday = 0.7) 
  (h2 : p_neither = 0.35) 
  (h3 : p_both = 0.6) :
  ∃ (p_tuesday : ℝ), p_tuesday = 0.55 := by
sorry


end NUMINAMATH_CALUDE_rain_probability_theorem_l2403_240398


namespace NUMINAMATH_CALUDE_square_partition_exists_l2403_240316

/-- A square is a four-sided polygon with all sides equal and all angles equal to 90 degrees. -/
structure Square where
  sides : Fin 4 → ℝ
  angles : Fin 4 → ℝ
  sides_equal : ∀ i j, sides i = sides j
  angles_right : ∀ i, angles i = 90

/-- A convex pentagon is a five-sided polygon with all interior angles less than 180 degrees. -/
structure ConvexPentagon where
  sides : Fin 5 → ℝ
  angles : Fin 5 → ℝ
  angles_convex : ∀ i, angles i < 180

/-- A partition of a square into convex pentagons -/
structure SquarePartition where
  square : Square
  pentagons : List ConvexPentagon
  is_partition : Square → List ConvexPentagon → Prop

/-- Theorem: There exists a partition of a square into a finite number of convex pentagons -/
theorem square_partition_exists : ∃ p : SquarePartition, p.pentagons.length > 0 := by
  sorry

end NUMINAMATH_CALUDE_square_partition_exists_l2403_240316


namespace NUMINAMATH_CALUDE_fall_semester_duration_l2403_240375

/-- The duration of the fall semester in weeks -/
def semester_length : ℕ := 15

/-- The number of hours Paris studies during weekdays -/
def weekday_hours : ℕ := 3

/-- The number of hours Paris studies on Saturday -/
def saturday_hours : ℕ := 4

/-- The number of hours Paris studies on Sunday -/
def sunday_hours : ℕ := 5

/-- The total number of hours Paris studies during the semester -/
def total_study_hours : ℕ := 360

theorem fall_semester_duration :
  semester_length * (5 * weekday_hours + saturday_hours + sunday_hours) = total_study_hours := by
  sorry

end NUMINAMATH_CALUDE_fall_semester_duration_l2403_240375


namespace NUMINAMATH_CALUDE_divisible_by_2000_arrangement_l2403_240312

theorem divisible_by_2000_arrangement (nums : List ℕ) (h : nums.length = 23) :
  ∃ (arrangement : List ℕ → ℕ), arrangement nums % 2000 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_2000_arrangement_l2403_240312


namespace NUMINAMATH_CALUDE_planes_and_perpendicular_lines_l2403_240327

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- State the theorem
theorem planes_and_perpendicular_lines 
  (α β : Plane) (m n : Line) :
  parallel α β → 
  perpendicular n α → 
  perpendicular m β → 
  line_parallel m n :=
by sorry

end NUMINAMATH_CALUDE_planes_and_perpendicular_lines_l2403_240327


namespace NUMINAMATH_CALUDE_fraction_undefined_at_two_l2403_240397

theorem fraction_undefined_at_two (x : ℝ) : 
  x / (2 - x) = x / (2 - x) → x ≠ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_undefined_at_two_l2403_240397


namespace NUMINAMATH_CALUDE_cos_sum_fifteenths_l2403_240300

theorem cos_sum_fifteenths : Real.cos (4 * π / 15) + Real.cos (10 * π / 15) + Real.cos (14 * π / 15) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_fifteenths_l2403_240300


namespace NUMINAMATH_CALUDE_tangent_slope_at_two_l2403_240376

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem tangent_slope_at_two
  (h_even : ∀ x, f x = f (-x))
  (h_neg : ∀ x, x < 0 → f x = x / (x - 1))
  : deriv f 2 = 1 / 9 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_two_l2403_240376


namespace NUMINAMATH_CALUDE_chocolate_cost_l2403_240392

theorem chocolate_cost (candies_per_box : ℕ) (cost_per_box : ℕ) (total_candies : ℕ) :
  candies_per_box = 25 →
  cost_per_box = 6 →
  total_candies = 600 →
  (total_candies / candies_per_box) * cost_per_box = 144 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_cost_l2403_240392


namespace NUMINAMATH_CALUDE_red_balls_count_l2403_240399

/-- Given a bag with 2400 balls of three colors (red, green, blue) distributed
    in the ratio 15:13:17, prove that the number of red balls is 795. -/
theorem red_balls_count (total : ℕ) (red green blue : ℕ) :
  total = 2400 →
  red + green + blue = total →
  red * 13 = green * 15 →
  red * 17 = blue * 15 →
  red = 795 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l2403_240399


namespace NUMINAMATH_CALUDE_sculpture_exposed_area_l2403_240318

/-- Represents the sculpture with its properties --/
structure Sculpture where
  cubeEdge : Real
  bottomLayerCubes : Nat
  middleLayerCubes : Nat
  topLayerCubes : Nat
  submersionRatio : Real

/-- Calculates the exposed surface area of the sculpture --/
def exposedSurfaceArea (s : Sculpture) : Real :=
  sorry

/-- Theorem stating that the exposed surface area of the given sculpture is 12.75 square meters --/
theorem sculpture_exposed_area :
  let s : Sculpture := {
    cubeEdge := 0.5,
    bottomLayerCubes := 16,
    middleLayerCubes := 9,
    topLayerCubes := 1,
    submersionRatio := 0.5
  }
  exposedSurfaceArea s = 12.75 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_exposed_area_l2403_240318


namespace NUMINAMATH_CALUDE_roots_sum_zero_l2403_240319

theorem roots_sum_zero (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : Real.log (abs x₁) = m) 
  (h₂ : Real.log (abs x₂) = m) 
  (h₃ : x₁ ≠ x₂) : 
  x₁ + x₂ = 0 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_zero_l2403_240319


namespace NUMINAMATH_CALUDE_circle_c_equation_l2403_240378

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  passes_through : ℝ × ℝ

-- Define the equation of a circle
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = 
    (c.passes_through.1 - c.center.1)^2 + (c.passes_through.2 - c.center.2)^2

-- Theorem statement
theorem circle_c_equation :
  let c : Circle := { center := (1, 1), passes_through := (0, 0) }
  ∀ x y : ℝ, circle_equation c x y ↔ (x - 1)^2 + (y - 1)^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_c_equation_l2403_240378


namespace NUMINAMATH_CALUDE_volume_complete_octagonal_pyramid_l2403_240351

/-- The volume of a complete pyramid with a regular octagonal base, given the dimensions of its truncated version. -/
theorem volume_complete_octagonal_pyramid 
  (lower_base_side : ℝ) 
  (upper_base_side : ℝ) 
  (truncated_height : ℝ) 
  (h_lower : lower_base_side = 0.4) 
  (h_upper : upper_base_side = 0.3) 
  (h_height : truncated_height = 0.5) : 
  ∃ (volume : ℝ), volume = (16/75) * (Real.sqrt 2 + 1) := by
  sorry

#check volume_complete_octagonal_pyramid

end NUMINAMATH_CALUDE_volume_complete_octagonal_pyramid_l2403_240351


namespace NUMINAMATH_CALUDE_jane_reading_period_l2403_240385

/-- Represents Jane's reading habits and total pages read --/
structure ReadingHabit where
  morning_pages : ℕ
  evening_pages : ℕ
  total_pages : ℕ

/-- Calculates the number of days Jane reads based on her reading habit --/
def calculate_reading_days (habit : ReadingHabit) : ℚ :=
  habit.total_pages / (habit.morning_pages + habit.evening_pages)

/-- Theorem stating that Jane reads for 7 days --/
theorem jane_reading_period (habit : ReadingHabit) 
  (h1 : habit.morning_pages = 5)
  (h2 : habit.evening_pages = 10)
  (h3 : habit.total_pages = 105) :
  calculate_reading_days habit = 7 := by
  sorry


end NUMINAMATH_CALUDE_jane_reading_period_l2403_240385


namespace NUMINAMATH_CALUDE_shopkeeper_red_cards_l2403_240395

/-- Calculates the total number of red cards in all decks --/
def total_red_cards (total_decks : ℕ) (standard_decks : ℕ) (special_decks : ℕ) 
  (red_cards_standard : ℕ) (additional_red_cards_special : ℕ) : ℕ :=
  (standard_decks * red_cards_standard) + 
  (special_decks * (red_cards_standard + additional_red_cards_special))

theorem shopkeeper_red_cards : 
  total_red_cards 15 5 10 26 4 = 430 := by
  sorry

#eval total_red_cards 15 5 10 26 4

end NUMINAMATH_CALUDE_shopkeeper_red_cards_l2403_240395


namespace NUMINAMATH_CALUDE_min_value_expression_l2403_240381

theorem min_value_expression (a b c : ℝ) 
  (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 4) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (4/c - 1)^2 ≥ 12 - 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2403_240381


namespace NUMINAMATH_CALUDE_log_inequality_implies_range_l2403_240323

theorem log_inequality_implies_range (x : ℝ) (hx : x > 0) :
  (Real.log x) ^ 2015 < (Real.log x) ^ 2014 ∧ 
  (Real.log x) ^ 2014 < (Real.log x) ^ 2016 →
  0 < x ∧ x < (1/10 : ℝ) := by sorry

end NUMINAMATH_CALUDE_log_inequality_implies_range_l2403_240323


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_a_value_l2403_240313

/-- Proves that for a hyperbola x²/a² - y² = 1 with a > 0, 
    if one of its asymptotes is y + 2x = 0, then a = 2 -/
theorem hyperbola_asymptote_a_value (a : ℝ) (h1 : a > 0) : 
  (∃ x y : ℝ, x^2 / a^2 - y^2 = 1 ∧ y + 2*x = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_a_value_l2403_240313


namespace NUMINAMATH_CALUDE_candy_car_cost_proof_l2403_240346

/-- The cost of a candy car given initial amount and change received -/
def candy_car_cost (initial_amount change : ℚ) : ℚ :=
  initial_amount - change

/-- Theorem stating the cost of the candy car is $0.45 -/
theorem candy_car_cost_proof (initial_amount change : ℚ) 
  (h1 : initial_amount = 1.80)
  (h2 : change = 1.35) : 
  candy_car_cost initial_amount change = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_candy_car_cost_proof_l2403_240346


namespace NUMINAMATH_CALUDE_zephyr_island_population_reaches_capacity_l2403_240355

/-- Represents the population growth on Zephyr Island -/
def zephyr_island_population (initial_year : ℕ) (initial_population : ℕ) (years_passed : ℕ) : ℕ :=
  initial_population * (4 ^ (years_passed / 20))

/-- Represents the maximum capacity of Zephyr Island -/
def zephyr_island_capacity (total_acres : ℕ) (acres_per_person : ℕ) : ℕ :=
  total_acres / acres_per_person

/-- Theorem stating that the population will reach or exceed the maximum capacity in 40 years -/
theorem zephyr_island_population_reaches_capacity :
  let initial_year := 2023
  let initial_population := 500
  let total_acres := 30000
  let acres_per_person := 2
  let years_to_capacity := 40
  zephyr_island_population initial_year initial_population years_to_capacity ≥ 
    zephyr_island_capacity total_acres acres_per_person ∧
  zephyr_island_population initial_year initial_population (years_to_capacity - 20) < 
    zephyr_island_capacity total_acres acres_per_person :=
by
  sorry


end NUMINAMATH_CALUDE_zephyr_island_population_reaches_capacity_l2403_240355


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2403_240306

/-- Two numbers are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x y : ℝ) 
  (h1 : InverselyProportional x y)
  (h2 : x + y = 28)
  (h3 : x - y = 8) :
  (∃ z : ℝ, z = 7 → y = 180 / 7) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2403_240306


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l2403_240387

theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l2403_240387


namespace NUMINAMATH_CALUDE_curve_properties_l2403_240341

-- Define the curve
def on_curve (x y : ℝ) : Prop := Real.sqrt x + Real.sqrt y = 1

-- Theorem statement
theorem curve_properties :
  (∀ a b : ℝ, on_curve a b → on_curve b a) ∧
  on_curve 0 1 ∧
  on_curve 1 0 ∧
  on_curve (1/4) (1/4) :=
by sorry

end NUMINAMATH_CALUDE_curve_properties_l2403_240341


namespace NUMINAMATH_CALUDE_smallest_product_smallest_product_is_neg_32_l2403_240336

def S : Finset Int := {-8, -3, -2, 2, 4}

theorem smallest_product (a b : Int) (ha : a ∈ S) (hb : b ∈ S) :
  ∃ (x y : Int), x ∈ S ∧ y ∈ S ∧ x * y ≤ a * b :=
by
  sorry

theorem smallest_product_is_neg_32 :
  ∃ (a b : Int), a ∈ S ∧ b ∈ S ∧ a * b = -32 ∧
  ∀ (x y : Int), x ∈ S → y ∈ S → a * b ≤ x * y :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_product_smallest_product_is_neg_32_l2403_240336


namespace NUMINAMATH_CALUDE_gym_guests_ratio_l2403_240366

/-- Represents the number of guests entering the gym each hour -/
structure GymGuests where
  first_hour : ℕ
  second_hour : ℕ
  third_hour : ℕ
  fourth_hour : ℕ

/-- Calculates the total number of guests -/
def total_guests (g : GymGuests) : ℕ :=
  g.first_hour + g.second_hour + g.third_hour + g.fourth_hour

theorem gym_guests_ratio (total_towels : ℕ) (g : GymGuests) : 
  total_towels = 300 →
  g.first_hour = 50 →
  g.second_hour = (120 * g.first_hour) / 100 →
  g.third_hour = (125 * g.second_hour) / 100 →
  g.fourth_hour > g.third_hour →
  total_guests g = 285 →
  (g.fourth_hour - g.third_hour) * 3 = g.third_hour := by
  sorry

#check gym_guests_ratio

end NUMINAMATH_CALUDE_gym_guests_ratio_l2403_240366


namespace NUMINAMATH_CALUDE_investment_calculation_l2403_240368

/-- Calculates the total investment in shares given the following conditions:
  * Face value of shares is 100 rupees
  * Shares are bought at a 20% premium
  * Company declares a 6% dividend
  * Total dividend received is 720 rupees
-/
def calculate_investment (face_value : ℕ) (premium_percent : ℕ) (dividend_percent : ℕ) (total_dividend : ℕ) : ℕ :=
  let premium_price := face_value + face_value * premium_percent / 100
  let dividend_per_share := face_value * dividend_percent / 100
  let num_shares := total_dividend / dividend_per_share
  num_shares * premium_price

/-- Theorem stating that under the given conditions, the total investment is 14400 rupees -/
theorem investment_calculation :
  calculate_investment 100 20 6 720 = 14400 := by
  sorry

end NUMINAMATH_CALUDE_investment_calculation_l2403_240368


namespace NUMINAMATH_CALUDE_cubic_sum_prime_power_l2403_240303

theorem cubic_sum_prime_power (a b p n : ℕ) : 
  0 < a ∧ 0 < b ∧ 0 < p ∧ 0 < n ∧ 
  Nat.Prime p ∧ 
  a^3 + b^3 = p^n →
  (∃ k : ℕ, (a = 2^k ∧ b = 2^k ∧ p = 2 ∧ n = 3*k + 1) ∨
             (a = 2*(3^k) ∧ b = 3^k ∧ p = 3 ∧ n = 3*k + 2) ∨
             (a = 3^k ∧ b = 2*(3^k) ∧ p = 3 ∧ n = 3*k + 2)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_sum_prime_power_l2403_240303


namespace NUMINAMATH_CALUDE_candy_sampling_percentage_l2403_240349

/-- The percentage of customers caught sampling candy -/
def caught_percent : ℝ := 22

/-- The percentage of candy samplers who are not caught -/
def not_caught_percent : ℝ := 20

/-- The total percentage of customers who sample candy -/
def total_sample_percent : ℝ := 28

/-- Theorem stating that the total percentage of customers who sample candy is 28% -/
theorem candy_sampling_percentage :
  total_sample_percent = caught_percent / (1 - not_caught_percent / 100) :=
by sorry

end NUMINAMATH_CALUDE_candy_sampling_percentage_l2403_240349


namespace NUMINAMATH_CALUDE_jordan_wins_l2403_240373

theorem jordan_wins (peter_wins peter_losses emma_wins emma_losses jordan_losses : ℕ)
  (h1 : peter_wins = 5)
  (h2 : peter_losses = 4)
  (h3 : emma_wins = 4)
  (h4 : emma_losses = 5)
  (h5 : jordan_losses = 2) :
  ∃ jordan_wins : ℕ,
    jordan_wins = 2 ∧
    2 * (peter_wins + peter_losses + emma_wins + emma_losses + jordan_wins + jordan_losses) =
    peter_wins + emma_wins + jordan_wins + peter_losses + emma_losses + jordan_losses :=
by sorry

end NUMINAMATH_CALUDE_jordan_wins_l2403_240373


namespace NUMINAMATH_CALUDE_second_derivative_parametric_function_l2403_240367

/-- The second-order derivative of a parametrically defined function -/
theorem second_derivative_parametric_function (t : ℝ) (h : t ≠ 0) :
  let x := 1 / t
  let y := 1 / (1 + t^2)
  let y''_xx := (2 * (t^2 - 3) * t^4) / ((1 + t^2)^3)
  ∃ (d2y_dx2 : ℝ), d2y_dx2 = y''_xx := by
  sorry

end NUMINAMATH_CALUDE_second_derivative_parametric_function_l2403_240367


namespace NUMINAMATH_CALUDE_segment_length_is_52_l2403_240324

/-- A right triangle with sides 10, 24, and 26 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  side_a : a = 10
  side_b : b = 24
  side_c : c = 26

/-- Three identical circles inscribed in the triangle -/
structure InscribedCircles where
  radius : ℝ
  radius_value : radius = 2
  touches_sides : Bool
  touches_other_circles : Bool

/-- The total length of segments from vertices to tangency points -/
def total_segment_length (t : RightTriangle) (circles : InscribedCircles) : ℝ :=
  (t.a - circles.radius) + (t.b - circles.radius) + (t.c - 2 * circles.radius)

theorem segment_length_is_52 (t : RightTriangle) (circles : InscribedCircles) :
  total_segment_length t circles = 52 :=
sorry

end NUMINAMATH_CALUDE_segment_length_is_52_l2403_240324


namespace NUMINAMATH_CALUDE_calculation_proof_l2403_240333

theorem calculation_proof : 
  47 * ((4 + 3/7) - (5 + 1/3)) / ((3 + 1/2) + (2 + 1/5)) = -(7 + 119/171) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2403_240333


namespace NUMINAMATH_CALUDE_triangle_side_value_l2403_240338

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem triangle_side_value (A B C : ℝ) (a b c : ℝ) :
  f A = 2 →
  b = 1 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 →
  a = Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_value_l2403_240338


namespace NUMINAMATH_CALUDE_second_account_interest_rate_l2403_240301

/-- Proves that the interest rate of the second account is 4% given the problem conditions --/
theorem second_account_interest_rate :
  ∀ (first_amount second_amount first_rate second_rate total_interest : ℝ),
    first_amount = 1000 →
    second_amount = first_amount + 800 →
    first_rate = 0.02 →
    total_interest = 92 →
    total_interest = first_rate * first_amount + second_rate * second_amount →
    second_rate = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_second_account_interest_rate_l2403_240301


namespace NUMINAMATH_CALUDE_square_side_length_l2403_240396

/-- Proves that a square with perimeter 52 cm and area 169 square cm has sides of length 13 cm -/
theorem square_side_length (s : ℝ) 
  (perimeter : s * 4 = 52) 
  (area : s * s = 169) : 
  s = 13 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2403_240396


namespace NUMINAMATH_CALUDE_task_completion_condition_l2403_240361

/-- Represents the completion of a task given the number of people working in two phases -/
def task_completion (x : ℝ) : Prop :=
  let total_time : ℝ := 40
  let phase1_time : ℝ := 4
  let phase2_time : ℝ := 8
  let phase1_people : ℝ := x
  let phase2_people : ℝ := x + 2
  (phase1_time * phase1_people) / total_time + (phase2_time * phase2_people) / total_time = 1

/-- Theorem stating the condition for task completion -/
theorem task_completion_condition (x : ℝ) :
  task_completion x ↔ 4 * x / 40 + 8 * (x + 2) / 40 = 1 :=
by sorry

end NUMINAMATH_CALUDE_task_completion_condition_l2403_240361


namespace NUMINAMATH_CALUDE_birds_and_nests_difference_l2403_240358

theorem birds_and_nests_difference :
  let num_birds : ℕ := 6
  let num_nests : ℕ := 3
  num_birds - num_nests = 3 :=
by sorry

end NUMINAMATH_CALUDE_birds_and_nests_difference_l2403_240358


namespace NUMINAMATH_CALUDE_largest_y_in_special_right_triangle_l2403_240325

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem largest_y_in_special_right_triangle (x y z : ℕ) 
  (h1 : is_prime x ∧ is_prime y ∧ is_prime z)
  (h2 : x + y + z = 90)
  (h3 : y < x)
  (h4 : y > z) :
  y ≤ 47 ∧ ∃ (x' z' : ℕ), is_prime x' ∧ is_prime z' ∧ x' + 47 + z' = 90 ∧ 47 < x' ∧ 47 > z' :=
sorry

end NUMINAMATH_CALUDE_largest_y_in_special_right_triangle_l2403_240325


namespace NUMINAMATH_CALUDE_sum_four_consecutive_odd_divisible_by_two_l2403_240389

theorem sum_four_consecutive_odd_divisible_by_two (n : ℤ) : 
  ∃ k : ℤ, (2*n + 1) + (2*n + 3) + (2*n + 5) + (2*n + 7) = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_four_consecutive_odd_divisible_by_two_l2403_240389


namespace NUMINAMATH_CALUDE_sum_of_edges_for_given_pyramid_l2403_240359

/-- Regular hexagonal pyramid with given edge lengths -/
structure RegularHexagonalPyramid where
  base_edge : ℝ
  lateral_edge : ℝ

/-- Sum of all edges of a regular hexagonal pyramid -/
def sum_of_edges (p : RegularHexagonalPyramid) : ℝ :=
  6 * p.base_edge + 6 * p.lateral_edge

/-- Theorem: The sum of all edges of a regular hexagonal pyramid with base edge 8 and lateral edge 13 is 126 -/
theorem sum_of_edges_for_given_pyramid :
  let p : RegularHexagonalPyramid := ⟨8, 13⟩
  sum_of_edges p = 126 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_edges_for_given_pyramid_l2403_240359


namespace NUMINAMATH_CALUDE_arc_length_of_sector_l2403_240357

/-- Given a sector with a central angle of 60° and a radius of 6 cm, 
    the length of the arc is equal to 2π cm. -/
theorem arc_length_of_sector (α : Real) (r : Real) : 
  α = 60 * π / 180 → r = 6 → α * r = 2 * π := by sorry

end NUMINAMATH_CALUDE_arc_length_of_sector_l2403_240357


namespace NUMINAMATH_CALUDE_cube_root_of_two_solves_equation_l2403_240383

theorem cube_root_of_two_solves_equation :
  ∃ x : ℝ, x^3 = 2 ∧ x = Real.rpow 2 (1/3) :=
sorry

end NUMINAMATH_CALUDE_cube_root_of_two_solves_equation_l2403_240383


namespace NUMINAMATH_CALUDE_prob_score_over_14_is_0_3_expected_value_is_13_6_l2403_240314

-- Define the success rates and point values
def three_week_success_rate : ℝ := 0.7
def four_week_success_rate : ℝ := 0.3
def three_week_success_points : ℝ := 8
def three_week_failure_points : ℝ := 4
def four_week_success_points : ℝ := 15
def four_week_failure_points : ℝ := 6

-- Define the probability of scoring more than 14 points
-- in a sequence of a three-week jump followed by a four-week jump
def prob_score_over_14 : ℝ :=
  three_week_success_rate * four_week_success_rate +
  (1 - three_week_success_rate) * four_week_success_rate

-- Define the expected value of the total score for two consecutive three-week jumps
def expected_value_two_three_week_jumps : ℝ :=
  (1 - three_week_success_rate)^2 * (2 * three_week_failure_points) +
  2 * three_week_success_rate * (1 - three_week_success_rate) * (three_week_success_points + three_week_failure_points) +
  three_week_success_rate^2 * (2 * three_week_success_points)

-- Theorem statements
theorem prob_score_over_14_is_0_3 : prob_score_over_14 = 0.3 := by sorry

theorem expected_value_is_13_6 : expected_value_two_three_week_jumps = 13.6 := by sorry

end NUMINAMATH_CALUDE_prob_score_over_14_is_0_3_expected_value_is_13_6_l2403_240314


namespace NUMINAMATH_CALUDE_third_purchase_total_l2403_240315

/-- Represents the clothing purchase scenario -/
structure ClothingPurchase where
  initialCost : ℕ
  typeAIncrease : ℕ
  typeBIncrease : ℕ
  secondCostIncrease : ℕ
  averageIncrease : ℕ
  profitMargin : ℚ
  thirdTypeBCost : ℕ

/-- Theorem stating the total number of pieces in the third purchase -/
theorem third_purchase_total (cp : ClothingPurchase)
  (h1 : cp.initialCost = 3600)
  (h2 : cp.typeAIncrease = 20)
  (h3 : cp.typeBIncrease = 5)
  (h4 : cp.secondCostIncrease = 400)
  (h5 : cp.averageIncrease = 8)
  (h6 : cp.profitMargin = 35 / 100)
  (h7 : cp.thirdTypeBCost = 3000) :
  ∃ (x y : ℕ),
    x + y = 50 ∧
    20 * x + 5 * y = 400 ∧
    8 * (x + y) = 400 ∧
    (3600 + 400) * (1 + cp.profitMargin) = 5400 ∧
    x * 60 + y * 75 = 3600 ∧
    3000 / 75 = (5400 - 3000) / 60 ∧
    (3000 / 75 + 3000 / 75) = 80 :=
  sorry


end NUMINAMATH_CALUDE_third_purchase_total_l2403_240315


namespace NUMINAMATH_CALUDE_division_remainder_proof_l2403_240382

theorem division_remainder_proof (a b : ℕ) 
  (h1 : a - b = 2415)
  (h2 : a = 2520)
  (h3 : a / b = 21) : 
  a % b = 315 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l2403_240382


namespace NUMINAMATH_CALUDE_distribute_five_students_three_classes_l2403_240308

/-- The number of ways to distribute students into classes -/
def distribute_students (total_students : ℕ) (num_classes : ℕ) (pre_assigned : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of distributions for the given problem -/
theorem distribute_five_students_three_classes : 
  distribute_students 5 3 1 = 56 := by sorry

end NUMINAMATH_CALUDE_distribute_five_students_three_classes_l2403_240308


namespace NUMINAMATH_CALUDE_triangle_third_side_l2403_240356

theorem triangle_third_side (a b c : ℝ) (angle : ℝ) : 
  a = 9 → b = 12 → angle = 150 * π / 180 → 
  c^2 = a^2 + b^2 - 2*a*b*(angle.cos) → 
  c = Real.sqrt (225 + 108 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_side_l2403_240356
