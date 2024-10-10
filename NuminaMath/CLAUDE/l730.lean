import Mathlib

namespace parabola_vertex_l730_73008

/-- The parabola defined by the equation y = x^2 + 2x + 5 -/
def parabola (x : ℝ) : ℝ := x^2 + 2*x + 5

/-- The x-coordinate of the vertex of the parabola -/
def vertex_x : ℝ := -1

/-- The y-coordinate of the vertex of the parabola -/
def vertex_y : ℝ := 4

/-- Theorem stating that (vertex_x, vertex_y) is the vertex of the parabola -/
theorem parabola_vertex :
  (∀ x : ℝ, parabola x ≥ parabola vertex_x) ∧
  parabola vertex_x = vertex_y :=
sorry

end parabola_vertex_l730_73008


namespace increasing_positive_function_inequality_l730_73016

theorem increasing_positive_function_inequality 
  (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y)
  (h_positive : ∀ x, f x > 0)
  (h_differentiable : Differentiable ℝ f) :
  3 * f (-2) > 2 * f (-3) := by
  sorry

end increasing_positive_function_inequality_l730_73016


namespace sin_600_degrees_l730_73031

theorem sin_600_degrees : Real.sin (600 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_600_degrees_l730_73031


namespace evaluate_expression_l730_73033

theorem evaluate_expression : 5^2 - 5 + (6^2 - 6) - (7^2 - 7) + (8^2 - 8) = 64 := by
  sorry

end evaluate_expression_l730_73033


namespace tan_45_degrees_l730_73075

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end tan_45_degrees_l730_73075


namespace multiple_valid_scenarios_exist_l730_73012

/-- Represents the ticket sales scenario for the Red Rose Theatre -/
structure TicketSales where
  total_tickets : ℕ
  total_sales : ℚ
  tickets_at_price1 : ℕ
  price1 : ℚ
  price2 : ℚ

/-- Checks if the given ticket sales scenario is valid -/
def is_valid_scenario (s : TicketSales) : Prop :=
  s.total_tickets = s.tickets_at_price1 + (s.total_tickets - s.tickets_at_price1) ∧
  s.total_sales = s.tickets_at_price1 * s.price1 + (s.total_tickets - s.tickets_at_price1) * s.price2

/-- States that multiple valid scenarios can exist for the same input data -/
theorem multiple_valid_scenarios_exist (total_tickets : ℕ) (total_sales : ℚ) (tickets_at_price1 : ℕ) :
  ∃ (s1 s2 : TicketSales),
    s1.total_tickets = total_tickets ∧
    s1.total_sales = total_sales ∧
    s1.tickets_at_price1 = tickets_at_price1 ∧
    s2.total_tickets = total_tickets ∧
    s2.total_sales = total_sales ∧
    s2.tickets_at_price1 = tickets_at_price1 ∧
    is_valid_scenario s1 ∧
    is_valid_scenario s2 ∧
    s1.price1 ≠ s2.price1 :=
  sorry

#check multiple_valid_scenarios_exist 380 1972.5 205

end multiple_valid_scenarios_exist_l730_73012


namespace seating_arrangements_count_l730_73046

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to seat 8 people in a row with restrictions. -/
def seatingArrangements : ℕ :=
  let totalArrangements := factorial 8
  let wilmaAndPaulTogether := factorial 7 * factorial 2
  let adamAndEveTogether := factorial 7 * factorial 2
  let bothPairsTogether := factorial 6 * factorial 2 * factorial 2
  totalArrangements - (wilmaAndPaulTogether + adamAndEveTogether - bothPairsTogether)

/-- Theorem stating that the number of seating arrangements is 23040. -/
theorem seating_arrangements_count :
  seatingArrangements = 23040 := by sorry

end seating_arrangements_count_l730_73046


namespace paint_time_per_room_l730_73040

theorem paint_time_per_room 
  (total_rooms : ℕ) 
  (painted_rooms : ℕ) 
  (remaining_time : ℕ) 
  (h1 : total_rooms = 12) 
  (h2 : painted_rooms = 5) 
  (h3 : remaining_time = 49) : 
  remaining_time / (total_rooms - painted_rooms) = 7 := by
  sorry

end paint_time_per_room_l730_73040


namespace sum_of_six_consecutive_odd_iff_l730_73001

/-- Predicate to check if an integer is odd -/
def IsOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

/-- Predicate to check if an integer can be written as the sum of six consecutive odd integers -/
def IsSumOfSixConsecutiveOdd (S : ℤ) : Prop :=
  IsOdd ((S - 30) / 6)

theorem sum_of_six_consecutive_odd_iff (S : ℤ) :
  IsSumOfSixConsecutiveOdd S ↔
  ∃ n : ℤ, S = n + (n + 2) + (n + 4) + (n + 6) + (n + 8) + (n + 10) ∧ IsOdd n :=
sorry

end sum_of_six_consecutive_odd_iff_l730_73001


namespace proportional_segments_l730_73074

theorem proportional_segments (a b c d : ℝ) :
  b = 3 → c = 4 → d = 6 → (a / b = c / d) → a = 2 := by sorry

end proportional_segments_l730_73074


namespace percentage_on_rent_is_14_l730_73045

/-- Calculates the percentage of remaining income spent on house rent --/
def percentage_on_rent (total_income : ℚ) (petrol_expense : ℚ) (rent_expense : ℚ) : ℚ :=
  let remaining_income := total_income - petrol_expense
  (rent_expense / remaining_income) * 100

/-- Theorem: Given the conditions, the percentage spent on house rent is 14% --/
theorem percentage_on_rent_is_14 :
  ∀ (total_income : ℚ),
  total_income > 0 →
  total_income * (30 / 100) = 300 →
  percentage_on_rent total_income 300 98 = 14 := by
  sorry

#eval percentage_on_rent 1000 300 98

end percentage_on_rent_is_14_l730_73045


namespace chess_tournament_participants_l730_73019

theorem chess_tournament_participants : ∃ n : ℕ, n > 0 ∧ n * (n - 1) / 2 = 15 := by
  sorry

end chess_tournament_participants_l730_73019


namespace train_length_l730_73015

/-- The length of a train given its speed and the time it takes to cross a bridge of known length. -/
theorem train_length (v : ℝ) (t : ℝ) (bridge_length : ℝ) (h1 : v = 36 * (1000 / 3600)) (h2 : t = 26.997840172786177) (h3 : bridge_length = 150) :
  v * t - bridge_length = 119.97840172786177 :=
by sorry

end train_length_l730_73015


namespace trailing_zeros_remainder_l730_73003

/-- Calculate the number of trailing zeros in the product of factorials from 1 to n -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- The remainder when the number of trailing zeros in 1!2!3!...50! is divided by 500 -/
theorem trailing_zeros_remainder : trailingZeros 50 % 500 = 12 := by
  sorry

end trailing_zeros_remainder_l730_73003


namespace secant_triangle_area_l730_73080

theorem secant_triangle_area (r : ℝ) (d : ℝ) (θ : ℝ) (S_ABC : ℝ) :
  r = 3 →
  d = 5 →
  θ = 30 * π / 180 →
  S_ABC = 10 →
  ∃ (S_AKL : ℝ), S_AKL = 8 / 5 :=
by sorry

end secant_triangle_area_l730_73080


namespace a_plus_b_values_l730_73069

/-- A strictly increasing sequence of positive integers -/
def StrictlyIncreasingPositiveSeq (s : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → s n < s m

/-- The theorem statement -/
theorem a_plus_b_values
  (a b : ℕ → ℕ)
  (h_a_incr : StrictlyIncreasingPositiveSeq a)
  (h_b_incr : StrictlyIncreasingPositiveSeq b)
  (h_eq : a 10 = b 10)
  (h_lt_2017 : a 10 < 2017)
  (h_a_rec : ∀ n : ℕ, a (n + 2) = a (n + 1) + a n)
  (h_b_rec : ∀ n : ℕ, b (n + 1) = 2 * b n) :
  (a 1 + b 1 = 13) ∨ (a 1 + b 1 = 20) :=
sorry

end a_plus_b_values_l730_73069


namespace smallest_five_digit_cube_sum_l730_73025

theorem smallest_five_digit_cube_sum (x : ℕ) : 
  x ≥ 10000 ∧ x < 100000 ∧ 
  (∃ k : ℕ, (343 * x) / 90 = k^3) ∧
  (∀ y : ℕ, y ≥ 10000 ∧ y < x → ¬(∃ k : ℕ, (343 * y) / 90 = k^3)) →
  x = 11250 := by
sorry

end smallest_five_digit_cube_sum_l730_73025


namespace solution_part_I_solution_part_II_l730_73064

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + |x + 1|

-- Theorem for part I
theorem solution_part_I :
  ∀ x : ℝ, f 1 x < 3 ↔ -1 < x ∧ x < 1 :=
by sorry

-- Theorem for part II
theorem solution_part_II :
  (∃ x : ℝ, ∀ y : ℝ, f a y ≥ f a x ∧ f a x = 1) ↔ a = -4 ∨ a = 0 :=
by sorry

end solution_part_I_solution_part_II_l730_73064


namespace matrix_product_equality_l730_73043

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -3; 1, 2]
def B : Matrix (Fin 2) (Fin 1) ℝ := !![4; -6]
def result : Matrix (Fin 2) (Fin 1) ℝ := !![26; -8]

theorem matrix_product_equality : A * B = result := by sorry

end matrix_product_equality_l730_73043


namespace preferred_groups_2000_l730_73055

/-- The number of non-empty subsets of {1, 2, ..., n} whose sum is divisible by 5 -/
def preferred_groups (n : ℕ) : ℕ := sorry

/-- The formula for the number of preferred groups when n = 2000 -/
def preferred_groups_formula : ℕ := 
  2^400 * ((1 / 5) * (2^1600 - 1) + 1) - 1

theorem preferred_groups_2000 : 
  preferred_groups 2000 = preferred_groups_formula := by sorry

end preferred_groups_2000_l730_73055


namespace book_profit_percentage_l730_73095

/-- Given a book's cost price and additional information about its profit, 
    calculate the initial profit percentage. -/
theorem book_profit_percentage 
  (cost_price : ℝ) 
  (additional_profit : ℝ) 
  (new_profit_percentage : ℝ) :
  cost_price = 2400 →
  additional_profit = 120 →
  new_profit_percentage = 15 →
  ∃ (initial_profit_percentage : ℝ),
    initial_profit_percentage = 10 ∧
    cost_price * (1 + new_profit_percentage / 100) = 
      cost_price * (1 + initial_profit_percentage / 100) + additional_profit :=
by sorry

end book_profit_percentage_l730_73095


namespace root_cubic_expression_l730_73092

theorem root_cubic_expression (m : ℝ) : 
  m^2 + 3*m - 2023 = 0 → m^3 + 2*m^2 - 2026*m - 2023 = -4046 := by
  sorry

end root_cubic_expression_l730_73092


namespace cases_in_1990_l730_73023

def linearDecrease (initial : ℕ) (final : ℕ) (totalYears : ℕ) (yearsPassed : ℕ) : ℕ :=
  initial - (initial - final) * yearsPassed / totalYears

theorem cases_in_1990 : 
  let initial := 600000
  let final := 2000
  let totalYears := 30
  let yearsPassed := 20
  linearDecrease initial final totalYears yearsPassed = 201333 :=
by sorry

end cases_in_1990_l730_73023


namespace calculate_expression_l730_73054

theorem calculate_expression : 3 * 3^3 + 4^7 / 4^5 = 97 := by
  sorry

end calculate_expression_l730_73054


namespace bus_passengers_l730_73006

theorem bus_passengers (total : ℕ) 
  (h1 : 3 * total = 5 * (total / 5 * 3))  -- 3/5 of total are Dutch
  (h2 : (total / 5 * 3) / 2 * 2 = total / 5 * 3)  -- 1/2 of Dutch are American
  (h3 : ((total / 5 * 3) / 2) / 3 * 3 = (total / 5 * 3) / 2)  -- 1/3 of Dutch Americans got window seats
  (h4 : ((total / 5 * 3) / 2) / 3 = 9)  -- Number of Dutch Americans at windows is 9
  : total = 90 := by
  sorry

end bus_passengers_l730_73006


namespace truck_kinetic_energy_l730_73029

/-- The initial kinetic energy of a truck with mass m, initial velocity v, and braking force F
    that stops after traveling a distance x, is equal to Fx. -/
theorem truck_kinetic_energy
  (m : ℝ) (v : ℝ) (F : ℝ) (x : ℝ) (t : ℝ)
  (h1 : m > 0)
  (h2 : v > 0)
  (h3 : F > 0)
  (h4 : x > 0)
  (h5 : t > 0)
  (h6 : F * x = (1/2) * m * v^2) :
  (1/2) * m * v^2 = F * x := by
sorry

end truck_kinetic_energy_l730_73029


namespace limit_special_function_l730_73018

/-- The limit of (4^(5x) - 9^(-2x)) / (sin(x) - tan(x^3)) as x approaches 0 is ln(1024 * 81) -/
theorem limit_special_function :
  ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ →
    |(4^(5*x) - 9^(-2*x)) / (Real.sin x - Real.tan (x^3)) - Real.log (1024 * 81)| < ε :=
sorry

end limit_special_function_l730_73018


namespace harry_sister_stamps_l730_73044

theorem harry_sister_stamps (total : ℕ) (harry_ratio : ℕ) (sister_stamps : ℕ) : 
  total = 240 → 
  harry_ratio = 3 → 
  sister_stamps + harry_ratio * sister_stamps = total → 
  sister_stamps = 60 := by
sorry

end harry_sister_stamps_l730_73044


namespace amritsar_bombay_encounters_l730_73024

/-- Represents a train journey from Amritsar to Bombay -/
structure TrainJourney where
  startTime : Nat  -- Start time in minutes after midnight
  duration : Nat   -- Duration of journey in minutes
  dailyDepartures : Nat  -- Number of trains departing each day

/-- Calculates the number of trains encountered during the journey -/
def encountersCount (journey : TrainJourney) : Nat :=
  sorry

/-- Theorem stating that a train journey with given conditions encounters 5 other trains -/
theorem amritsar_bombay_encounters :
  ∀ (journey : TrainJourney),
    journey.startTime = 9 * 60 →  -- 9 am start time
    journey.duration = 3 * 24 * 60 + 30 →  -- 3 days and 30 minutes duration
    journey.dailyDepartures = 1 →  -- One train departs each day
    encountersCount journey = 5 :=
  sorry

end amritsar_bombay_encounters_l730_73024


namespace exchange_rate_problem_l730_73084

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- Proves that under the given exchange rate and spending conditions, 
    the sum of digits of the initial U.S. dollars is 8 -/
theorem exchange_rate_problem (d : ℕ) : 
  (8 * d) / 5 - 75 = d → sum_of_digits d = 8 := by
  sorry

end exchange_rate_problem_l730_73084


namespace no_solution_implies_a_leq_one_l730_73078

theorem no_solution_implies_a_leq_one :
  (∀ x : ℝ, ¬(x + 2 > 3 ∧ x < a)) → a ≤ 1 := by
  sorry

end no_solution_implies_a_leq_one_l730_73078


namespace degree_of_our_monomial_l730_73072

/-- The degree of a monomial is the sum of the exponents of its variables. -/
def degree_of_monomial (m : String) : ℕ :=
  sorry

/-- The monomial -2/5 * x^2 * y -/
def our_monomial : String := "-2/5x^2y"

theorem degree_of_our_monomial :
  degree_of_monomial our_monomial = 3 := by
  sorry

end degree_of_our_monomial_l730_73072


namespace complex_number_quadrant_l730_73027

theorem complex_number_quadrant (a b : ℝ) (h : (1 : ℂ) + a * I = (b + I) * (1 + I)) :
  a > 0 ∧ b > 0 :=
by sorry

end complex_number_quadrant_l730_73027


namespace balloon_count_l730_73009

/-- Calculates the total number of balloons given the number of gold, silver, and black balloons -/
def total_balloons (gold : ℕ) (silver : ℕ) (black : ℕ) : ℕ :=
  gold + silver + black

/-- Proves that the total number of balloons is 573 given the specified conditions -/
theorem balloon_count : 
  let gold : ℕ := 141
  let silver : ℕ := 2 * gold
  let black : ℕ := 150
  total_balloons gold silver black = 573 := by
  sorry

end balloon_count_l730_73009


namespace complex_equation_solution_l730_73049

theorem complex_equation_solution (a b : ℝ) : 
  (a - 2 * Complex.I = (b + Complex.I) * Complex.I) → (a = -1 ∧ b = -2) := by
  sorry

end complex_equation_solution_l730_73049


namespace inequality_proof_l730_73066

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  Real.sqrt (3 * x^2 + x * y) + Real.sqrt (3 * y^2 + y * z) + Real.sqrt (3 * z^2 + z * x) ≤ 2 * (x + y + z) := by
  sorry

end inequality_proof_l730_73066


namespace nabla_calculation_l730_73077

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- Theorem statement
theorem nabla_calculation : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end nabla_calculation_l730_73077


namespace john_scores_42_points_l730_73002

/-- Calculates the total points scored by John given the specified conditions -/
def total_points_scored (shots_per_interval : ℕ) (points_per_shot : ℕ) (three_point_shots : ℕ) 
                        (interval_duration : ℕ) (num_periods : ℕ) (period_duration : ℕ) : ℕ :=
  let total_time := num_periods * period_duration
  let num_intervals := total_time / interval_duration
  let points_per_interval := shots_per_interval * points_per_shot + three_point_shots * 3
  num_intervals * points_per_interval

/-- Theorem stating that John scores 42 points under the given conditions -/
theorem john_scores_42_points : 
  total_points_scored 2 2 1 4 2 12 = 42 := by
  sorry


end john_scores_42_points_l730_73002


namespace am_hm_difference_bound_l730_73067

theorem am_hm_difference_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x < y) :
  ((x - y)^2) / (2*(x + y)) < ((y - x)^2) / (8*x) := by
  sorry

end am_hm_difference_bound_l730_73067


namespace square_area_and_perimeter_l730_73065

/-- Given a square with diagonal length 12√2 cm, prove its area and perimeter -/
theorem square_area_and_perimeter (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  (s ^ 2 = 144) ∧ (4 * s = 48) := by sorry

end square_area_and_perimeter_l730_73065


namespace sum_of_differences_equals_6999993_l730_73039

/-- Calculates the local value of a digit in a number based on its position --/
def localValue (digit : ℕ) (position : ℕ) : ℕ :=
  digit * (10 ^ position)

/-- Calculates the difference between local value and face value for a digit --/
def valueDifference (digit : ℕ) (position : ℕ) : ℕ :=
  localValue digit position - digit

/-- The numeral we're working with --/
def numeral : ℕ := 657932657

/-- Positions of 7 in the numeral (0-indexed from right) --/
def sevenPositions : List ℕ := [0, 6]

/-- Sum of differences between local and face values for all 7s in the numeral --/
def sumOfDifferences : ℕ :=
  (sevenPositions.map (valueDifference 7)).sum

theorem sum_of_differences_equals_6999993 :
  sumOfDifferences = 6999993 := by sorry

end sum_of_differences_equals_6999993_l730_73039


namespace line_equation_k_value_l730_73030

/-- Given a line passing through points (m, n) and (m + 2, n + 0.4), 
    with equation x = ky + 5, prove that k = 5 -/
theorem line_equation_k_value (m n : ℝ) : 
  let p : ℝ := 0.4
  let point1 : ℝ × ℝ := (m, n)
  let point2 : ℝ × ℝ := (m + 2, n + p)
  let k : ℝ := (point2.2 - point1.2) / (point2.1 - point1.1)
  ∀ x y : ℝ, x = k * y + 5 → k = 5 := by
sorry

end line_equation_k_value_l730_73030


namespace system_solution_unique_l730_73073

theorem system_solution_unique :
  ∃! (x y : ℚ), 37 * x + 92 * y = 5043 ∧ 92 * x + 37 * y = 2568 :=
by
  -- The proof goes here
  sorry

end system_solution_unique_l730_73073


namespace bus_trip_speed_l730_73088

theorem bus_trip_speed (distance : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) :
  distance = 450 ∧ speed_increase = 5 ∧ time_decrease = 1 →
  ∃ (original_speed : ℝ),
    distance / original_speed - time_decrease = distance / (original_speed + speed_increase) ∧
    original_speed = 45 := by
  sorry

end bus_trip_speed_l730_73088


namespace normal_distribution_symmetry_l730_73052

/-- Represents a normally distributed random variable -/
structure NormalRV (μ : ℝ) (σ : ℝ) where
  (μ_pos : μ > 0)
  (σ_pos : σ > 0)

/-- The probability of a random variable being in an interval -/
noncomputable def prob (X : Type) (a b : ℝ) : ℝ := sorry

theorem normal_distribution_symmetry 
  (a σ : ℝ) (ξ : NormalRV a σ) 
  (h : prob (NormalRV a σ) 0 a = 0.3) : 
  prob (NormalRV a σ) 0 (2 * a) = 0.6 :=
sorry

end normal_distribution_symmetry_l730_73052


namespace parabola_directrix_l730_73070

/-- Given a parabola with equation y = (x^2 - 4x + 3) / 8, its directrix is y = -9/8 -/
theorem parabola_directrix (x y : ℝ) : 
  (y = (x^2 - 4*x + 3) / 8) → 
  (∃ (d : ℝ), d = -9/8 ∧ 
    ∀ (p : ℝ × ℝ), 
      p.1 = x ∧ p.2 = y → 
      ∃ (f : ℝ × ℝ), 
        (f.1 - p.1)^2 + (f.2 - p.2)^2 = (p.2 - d)^2 ∧
        ∀ (q : ℝ × ℝ), q.2 = d → 
          (f.1 - p.1)^2 + (f.2 - p.2)^2 ≤ (q.1 - p.1)^2 + (q.2 - p.2)^2) :=
by sorry

end parabola_directrix_l730_73070


namespace total_score_is_219_l730_73057

/-- Represents a player's score in a basketball game -/
structure PlayerScore where
  twoPointers : Nat
  threePointers : Nat
  freeThrows : Nat

/-- Calculates the total score for a player -/
def calculatePlayerScore (score : PlayerScore) : Nat :=
  2 * score.twoPointers + 3 * score.threePointers + score.freeThrows

/-- Theorem: The total points scored by all players is 219 -/
theorem total_score_is_219 
  (sam : PlayerScore)
  (alex : PlayerScore)
  (jake : PlayerScore)
  (lily : PlayerScore)
  (h_sam : sam = { twoPointers := 20, threePointers := 5, freeThrows := 10 })
  (h_alex : alex = { twoPointers := 15, threePointers := 6, freeThrows := 8 })
  (h_jake : jake = { twoPointers := 10, threePointers := 8, freeThrows := 5 })
  (h_lily : lily = { twoPointers := 12, threePointers := 3, freeThrows := 16 }) :
  calculatePlayerScore sam + calculatePlayerScore alex + 
  calculatePlayerScore jake + calculatePlayerScore lily = 219 := by
  sorry

end total_score_is_219_l730_73057


namespace min_area_rectangle_l730_73068

/-- Given a rectangle with integer length and width, and a perimeter of 120 units,
    the minimum possible area is 59 square units. -/
theorem min_area_rectangle (l w : ℕ) : 
  (2 * l + 2 * w = 120) → (l * w ≥ 59) := by
  sorry

end min_area_rectangle_l730_73068


namespace arithmetic_mean_of_special_set_l730_73087

theorem arithmetic_mean_of_special_set (n : ℕ) (hn : n > 2) :
  let set := List.replicate (n - 2) 1 ++ List.replicate 2 (1 - 1 / n)
  (List.sum set) / n = 1 - 2 / n^2 := by sorry

end arithmetic_mean_of_special_set_l730_73087


namespace arithmetic_calculation_l730_73085

theorem arithmetic_calculation : 6^2 - 4*5 + 2^2 = 20 := by
  sorry

end arithmetic_calculation_l730_73085


namespace watermelon_sale_proof_l730_73028

/-- Calculates the total money made from selling watermelons -/
def total_money_from_watermelons (weight : ℕ) (price_per_pound : ℕ) (num_watermelons : ℕ) : ℕ :=
  weight * price_per_pound * num_watermelons

/-- Proves that selling 18 watermelons weighing 23 pounds each at $2 per pound yields $828 -/
theorem watermelon_sale_proof :
  total_money_from_watermelons 23 2 18 = 828 := by
  sorry

end watermelon_sale_proof_l730_73028


namespace inscribed_circle_distance_l730_73081

theorem inscribed_circle_distance (a b : ℝ) (h1 : a = 6) (h2 : b = 8) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a + b + c) / 2
  let m := s - b
  2 * Real.sqrt ((a^2 + m^2 - 2 * a * m * (a / c)) / 5) = 2 * Real.sqrt (29 / 5) :=
by sorry

end inscribed_circle_distance_l730_73081


namespace circles_intersect_circles_satisfy_intersection_condition_l730_73022

/-- Circle C₁ with equation x² + y² + 2x + 8y - 8 = 0 -/
def C₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 + 8*p.2 - 8 = 0}

/-- Circle C₂ with equation x² + y² - 4x - 4y - 1 = 0 -/
def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 - 4*p.2 - 1 = 0}

/-- The center of circle C₁ -/
def center₁ : ℝ × ℝ := (-1, -4)

/-- The center of circle C₂ -/
def center₂ : ℝ × ℝ := (2, 2)

/-- The radius of circle C₁ -/
def radius₁ : ℝ := 5

/-- The radius of circle C₂ -/
def radius₂ : ℝ := 3

/-- Theorem stating that circles C₁ and C₂ intersect -/
theorem circles_intersect : ∃ p : ℝ × ℝ, p ∈ C₁ ∩ C₂ := by
  sorry

/-- Lemma stating the condition for intersecting circles -/
lemma intersecting_circles_condition (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ) :
  let d := Real.sqrt ((c₂.1 - c₁.1)^2 + (c₂.2 - c₁.2)^2)
  abs (r₁ - r₂) < d ∧ d < r₁ + r₂ → ∃ p : ℝ × ℝ, p ∈ C₁ ∩ C₂ := by
  sorry

/-- Proof that C₁ and C₂ satisfy the intersecting circles condition -/
theorem circles_satisfy_intersection_condition :
  let d := Real.sqrt ((center₂.1 - center₁.1)^2 + (center₂.2 - center₁.2)^2)
  abs (radius₁ - radius₂) < d ∧ d < radius₁ + radius₂ := by
  sorry

end circles_intersect_circles_satisfy_intersection_condition_l730_73022


namespace function_value_proof_l730_73042

theorem function_value_proof (f : ℝ → ℝ) (h : ∀ x, f (1 - 2*x) = 1 / x^2) :
  f (1/2) = 16 := by
  sorry

end function_value_proof_l730_73042


namespace donation_distribution_l730_73007

theorem donation_distribution (total : ℝ) (community_ratio : ℝ) (crisis_ratio : ℝ) (livelihood_ratio : ℝ)
  (h_total : total = 240)
  (h_community : community_ratio = 1/3)
  (h_crisis : crisis_ratio = 1/2)
  (h_livelihood : livelihood_ratio = 1/4) :
  let community := total * community_ratio
  let crisis := total * crisis_ratio
  let remaining := total - community - crisis
  let livelihood := remaining * livelihood_ratio
  total - community - crisis - livelihood = 30 := by
sorry

end donation_distribution_l730_73007


namespace hot_sauce_duration_l730_73091

-- Define constants
def serving_size : ℚ := 1/2
def servings_per_day : ℕ := 3
def quart_in_ounces : ℕ := 32
def container_size_difference : ℕ := 2

-- Define the container size
def container_size : ℕ := quart_in_ounces - container_size_difference

-- Define daily usage
def daily_usage : ℚ := serving_size * servings_per_day

-- Theorem to prove
theorem hot_sauce_duration :
  (container_size : ℚ) / daily_usage = 20 := by sorry

end hot_sauce_duration_l730_73091


namespace magic_8_ball_probability_l730_73035

def num_questions : ℕ := 7
def num_positive : ℕ := 3
def prob_positive : ℚ := 3/7

theorem magic_8_ball_probability :
  (Nat.choose num_questions num_positive : ℚ) *
  (prob_positive ^ num_positive) *
  ((1 - prob_positive) ^ (num_questions - num_positive)) =
  242112/823543 := by sorry

end magic_8_ball_probability_l730_73035


namespace work_problem_l730_73013

/-- Proves that 8 men became absent given the conditions of the work problem -/
theorem work_problem (total_men : ℕ) (original_days : ℕ) (actual_days : ℕ) 
  (h1 : total_men = 42)
  (h2 : original_days = 17)
  (h3 : actual_days = 21)
  (h4 : total_men * original_days = (total_men - absent_men) * actual_days) :
  absent_men = 8 := by
  sorry

#check work_problem

end work_problem_l730_73013


namespace equal_squares_count_l730_73094

/-- Represents a cell in the grid -/
inductive Cell
| White
| Black

/-- Represents the 5x5 grid -/
def Grid := Fin 5 → Fin 5 → Cell

/-- Defines the specific coloring pattern of the grid -/
def initial_grid : Grid :=
  fun i j => 
    if (i = 2 ∧ j = 2) ∨ 
       (i = 1 ∧ j = 3) ∨ 
       (i = 3 ∧ j = 1) ∨ 
       (i = 3 ∧ j = 3) ∨ 
       (i = 3 ∧ j = 5) 
    then Cell.Black 
    else Cell.White

/-- Checks if a square in the grid has equal number of black and white cells -/
def has_equal_cells (g : Grid) (top_left_i top_left_j size : Nat) : Bool :=
  sorry

/-- Counts the number of squares with equal black and white cells -/
def count_equal_squares (g : Grid) : Nat :=
  sorry

/-- The main theorem stating that there are exactly 16 squares with equal black and white cells -/
theorem equal_squares_count : count_equal_squares initial_grid = 16 :=
  sorry

end equal_squares_count_l730_73094


namespace curve_properties_l730_73060

/-- The curve y = ax³ + bx passing through point (2,2) with tangent slope 9 -/
def Curve (a b : ℝ) : Prop :=
  2 * a * 8 + 2 * b = 2 ∧ 3 * a * 4 + b = 9

/-- The function f(x) = ax³ + bx -/
def f (a b x : ℝ) : ℝ := a * x^3 + b * x

theorem curve_properties :
  ∀ a b : ℝ, Curve a b →
  (a * b = -3) ∧
  (∀ x : ℝ, -3/2 ≤ x ∧ x ≤ 3 → -2 ≤ f a b x ∧ f a b x ≤ 18) :=
by sorry

end curve_properties_l730_73060


namespace trapezoid_segment_property_l730_73076

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  midpoint_segment : ℝ
  equal_area_segment : ℝ
  base_difference : shorter_base + 150 = longer_base
  midpoint_area_ratio : (shorter_base + midpoint_segment) / (longer_base + midpoint_segment) = 3 / 4
  equal_area_condition : (shorter_base + equal_area_segment) * (height / 2) = 
                         (shorter_base + longer_base) * height / 2

/-- The theorem statement -/
theorem trapezoid_segment_property (t : Trapezoid) : 
  ⌊t.equal_area_segment^2 / 150⌋ = 304 := by sorry

end trapezoid_segment_property_l730_73076


namespace no_consecutive_heads_probability_sum_of_numerator_and_denominator_l730_73061

/-- Number of valid sequences of length n ending in Tails -/
def a : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n+2) => a (n+1) + a n

/-- Number of valid sequences of length n ending in Heads -/
def b : ℕ → ℕ
| 0 => 0
| (n+1) => a n

/-- Total number of valid sequences of length n -/
def valid_sequences (n : ℕ) : ℕ := a n + b n

/-- Total number of possible sequences of length n -/
def total_sequences (n : ℕ) : ℕ := 2^n

theorem no_consecutive_heads_probability :
  (valid_sequences 10 : ℚ) / (total_sequences 10 : ℚ) = 9 / 64 :=
sorry

theorem sum_of_numerator_and_denominator : 9 + 64 = 73 :=
sorry

end no_consecutive_heads_probability_sum_of_numerator_and_denominator_l730_73061


namespace smallest_valid_sequence_length_l730_73062

def is_valid_sequence (a : List Int) : Prop :=
  a.sum = 2005 ∧ a.prod = 2005

theorem smallest_valid_sequence_length :
  (∃ (n : Nat) (a : List Int), n > 1 ∧ a.length = n ∧ is_valid_sequence a) ∧
  (∀ (m : Nat) (b : List Int), m > 1 ∧ m < 5 ∧ b.length = m → ¬is_valid_sequence b) ∧
  (∃ (c : List Int), c.length = 5 ∧ is_valid_sequence c) :=
by sorry

end smallest_valid_sequence_length_l730_73062


namespace red_shells_count_l730_73099

theorem red_shells_count (total : ℕ) (green : ℕ) (not_red_or_green : ℕ) 
  (h1 : total = 291)
  (h2 : green = 49)
  (h3 : not_red_or_green = 166) :
  total - green - not_red_or_green = 76 := by
sorry

end red_shells_count_l730_73099


namespace mean_equality_implies_z_l730_73000

theorem mean_equality_implies_z (z : ℚ) : 
  (5 + 10 + 20) / 3 = (15 + z) / 2 → z = 25 / 3 :=
by sorry

end mean_equality_implies_z_l730_73000


namespace taxi_driver_problem_l730_73037

def distances : List Int := [8, -6, 3, -4, 8, -4, 4, -3]

def total_time : Rat := 4/3

theorem taxi_driver_problem (distances : List Int) (total_time : Rat) :
  (distances.sum = 6) ∧
  (((distances.map abs).sum : Rat) / total_time = 30) :=
by sorry

end taxi_driver_problem_l730_73037


namespace tenth_term_of_sequence_l730_73036

def arithmetic_sequence (a₁ a₂ a₃ : ℚ) : ℕ → ℚ :=
  λ n => a₁ + (n - 1) * (a₂ - a₁)

theorem tenth_term_of_sequence (h₁ : arithmetic_sequence (1/2) (5/6) (7/6) 2 = 5/6) 
                               (h₂ : arithmetic_sequence (1/2) (5/6) (7/6) 3 = 7/6) :
  arithmetic_sequence (1/2) (5/6) (7/6) 10 = 7/2 := by
  sorry

end tenth_term_of_sequence_l730_73036


namespace last_score_must_be_86_l730_73004

def scores : List ℕ := [73, 78, 84, 86, 97]

def is_integer (n : ℚ) : Prop := ∃ m : ℤ, n = m

def average_is_integer (entered_scores : List ℕ) : Prop :=
  ∀ k : ℕ, k > 0 → k ≤ entered_scores.length →
    is_integer ((entered_scores.take k).sum / k)

theorem last_score_must_be_86 :
  ∀ perm : List ℕ, perm.length = 5 →
  perm.toFinset = scores.toFinset →
  average_is_integer perm →
  perm.getLast? = some 86 :=
sorry

end last_score_must_be_86_l730_73004


namespace rationalize_denominator_l730_73017

theorem rationalize_denominator :
  ∃ (A B C D E F : ℚ) (X : ℕ),
    (1 : ℝ) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11) =
    (A * Real.sqrt 2 + B * Real.sqrt 3 + C * Real.sqrt 5 + D * Real.sqrt 11 + E * Real.sqrt X) / F ∧
    F > 0 ∧
    A + B + C + D + E + F = 20 :=
by sorry

end rationalize_denominator_l730_73017


namespace max_value_theorem_l730_73047

theorem max_value_theorem (x y : ℝ) (h : 4 * x^2 + y^2 + x * y = 1) :
  ∃ (max : ℝ), (∀ (x' y' : ℝ), 4 * x'^2 + y'^2 + x' * y' = 1 → 2 * x' + y' ≤ max) ∧
                max = 2 * Real.sqrt 10 / 5 :=
sorry

end max_value_theorem_l730_73047


namespace combined_average_age_l730_73005

theorem combined_average_age (room_a_count room_b_count room_c_count : ℕ)
                             (room_a_avg room_b_avg room_c_avg : ℝ) :
  room_a_count = 8 →
  room_b_count = 5 →
  room_c_count = 7 →
  room_a_avg = 30 →
  room_b_avg = 35 →
  room_c_avg = 40 →
  let total_count := room_a_count + room_b_count + room_c_count
  let total_age := room_a_count * room_a_avg + room_b_count * room_b_avg + room_c_count * room_c_avg
  (total_age / total_count : ℝ) = 34.75 := by
sorry

end combined_average_age_l730_73005


namespace range_of_c_l730_73034

theorem range_of_c (a b c : ℝ) :
  (∀ x : ℝ, (Real.sqrt (2 * x^2 + a * x + b) > x - c) ↔ (x ≤ 0 ∨ x > 1)) →
  c ∈ Set.Ioo 0 1 :=
by sorry

end range_of_c_l730_73034


namespace complex_exp_13pi_div_2_l730_73010

open Complex

theorem complex_exp_13pi_div_2 : exp (13 * π / 2 * I) = I := by sorry

end complex_exp_13pi_div_2_l730_73010


namespace even_sum_from_even_expression_l730_73026

theorem even_sum_from_even_expression (n m : ℤ) : 
  Even (n^2 + m^2 + n*m) → Even (n + m) := by
  sorry

end even_sum_from_even_expression_l730_73026


namespace parabola_line_intersection_minimum_l730_73056

/-- A parabola with equation y^2 = 16x --/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- A line passing through a point --/
structure Line where
  passingPoint : ℝ × ℝ

/-- Intersection points of a line and a parabola --/
structure Intersection where
  M : ℝ × ℝ
  N : ℝ × ℝ

/-- Distance between two points --/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- The main theorem --/
theorem parabola_line_intersection_minimum (p : Parabola) (l : Line) (i : Intersection) :
  p.equation = (fun x y => y^2 = 16*x) →
  p.focus = (4, 0) →
  l.passingPoint = p.focus →
  (∃ (x y : ℝ), p.equation x y ∧ (x, y) = i.M) →
  (∃ (x y : ℝ), p.equation x y ∧ (x, y) = i.N) →
  (∀ NF MF : ℝ, NF = distance i.N p.focus → MF = distance i.M p.focus →
    NF / 9 - 4 / MF ≥ 1 / 3) ∧
  (∃ NF MF : ℝ, NF = distance i.N p.focus ∧ MF = distance i.M p.focus ∧
    NF / 9 - 4 / MF = 1 / 3) :=
sorry

end parabola_line_intersection_minimum_l730_73056


namespace max_crosses_4x10_impossible_5x10_l730_73083

/-- Represents a table with crosses --/
structure CrossTable (m n : ℕ) :=
  (crosses : Fin m → Fin n → Bool)

/-- Checks if a row has an odd number of crosses --/
def rowHasOddCrosses (t : CrossTable m n) (i : Fin m) : Prop :=
  (Finset.filter (λ j => t.crosses i j) (Finset.univ : Finset (Fin n))).card % 2 = 1

/-- Checks if a column has an odd number of crosses --/
def colHasOddCrosses (t : CrossTable m n) (j : Fin n) : Prop :=
  (Finset.filter (λ i => t.crosses i j) (Finset.univ : Finset (Fin m))).card % 2 = 1

/-- Checks if all rows and columns have odd number of crosses --/
def allOddCrosses (t : CrossTable m n) : Prop :=
  (∀ i, rowHasOddCrosses t i) ∧ (∀ j, colHasOddCrosses t j)

/-- Counts the total number of crosses in the table --/
def totalCrosses (t : CrossTable m n) : ℕ :=
  (Finset.filter (λ (i, j) => t.crosses i j) (Finset.univ : Finset (Fin m × Fin n))).card

/-- Theorem: The maximum number of crosses in a 4x10 table with odd crosses in each row and column is 30 --/
theorem max_crosses_4x10 :
  (∃ t : CrossTable 4 10, allOddCrosses t ∧ totalCrosses t = 30) ∧
  (∀ t : CrossTable 4 10, allOddCrosses t → totalCrosses t ≤ 30) := by sorry

/-- Theorem: It's impossible to place crosses in a 5x10 table with odd crosses in each row and column --/
theorem impossible_5x10 :
  ¬ ∃ t : CrossTable 5 10, allOddCrosses t := by sorry

end max_crosses_4x10_impossible_5x10_l730_73083


namespace pauls_money_duration_l730_73079

/-- Represents the duration (in weeks) that money lasts given earnings and weekly spending. -/
def money_duration (lawn_earnings weed_eating_earnings weekly_spending : ℚ) : ℚ :=
  (lawn_earnings + weed_eating_earnings) / weekly_spending

/-- Theorem stating that Paul's money lasts for 2 weeks given his earnings and spending. -/
theorem pauls_money_duration :
  money_duration 3 3 3 = 2 := by
  sorry

end pauls_money_duration_l730_73079


namespace cyclist_speed_proof_l730_73063

/-- The distance between Town A and Town B in miles -/
def distance_AB : ℝ := 80

/-- The speed difference between Cyclist Y and Cyclist X in mph -/
def speed_difference : ℝ := 6

/-- The distance from Town B where the cyclists meet after Cyclist Y turns back, in miles -/
def meeting_distance : ℝ := 20

/-- The speed of Cyclist X in mph -/
def speed_X : ℝ := 9

/-- The speed of Cyclist Y in mph -/
def speed_Y : ℝ := speed_X + speed_difference

theorem cyclist_speed_proof :
  speed_X * ((distance_AB + meeting_distance) / speed_Y) = distance_AB - meeting_distance :=
sorry

end cyclist_speed_proof_l730_73063


namespace unique_sequence_exists_l730_73014

def is_valid_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧
  a 2 > 1 ∧
  ∀ n : ℕ, n ≥ 1 → (a (n + 1))^3 + 1 = (a n) * (a (n + 2))

theorem unique_sequence_exists : ∃! a : ℕ → ℤ, is_valid_sequence a := by
  sorry

end unique_sequence_exists_l730_73014


namespace f_one_less_than_f_two_necessary_not_sufficient_l730_73058

def increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem f_one_less_than_f_two_necessary_not_sufficient :
  ∃ f : ℝ → ℝ, (increasing f → f 1 < f 2) ∧
  ¬(f 1 < f 2 → increasing f) :=
by sorry

end f_one_less_than_f_two_necessary_not_sufficient_l730_73058


namespace fraction_inequality_solution_set_l730_73096

theorem fraction_inequality_solution_set :
  {x : ℝ | (x - 1) / x < 0} = Set.Ioo 0 1 := by sorry

end fraction_inequality_solution_set_l730_73096


namespace subset_union_existence_l730_73093

theorem subset_union_existence (n m : ℕ) (h1 : n ≥ 3) (h2 : m ≥ 2^(n-1) + 1) :
  ∀ (A : Fin m → Set (Fin n)), 
    (∀ j, A j ≠ ∅) → 
    (∀ i j, i ≠ j → A i ≠ A j) → 
    ∃ i j k, A i ∪ A j = A k := by
  sorry

end subset_union_existence_l730_73093


namespace remainder_of_binary_number_mod_4_l730_73086

def binary_number : ℕ := 111100010111

theorem remainder_of_binary_number_mod_4 :
  binary_number % 4 = 3 := by sorry

end remainder_of_binary_number_mod_4_l730_73086


namespace largest_n_for_factorable_quadratic_l730_73098

/-- A structure representing a quadratic expression ax^2 + bx + c -/
structure Quadratic where
  a : ℤ
  b : ℤ
  c : ℤ

/-- A structure representing a linear factor ax + b -/
structure LinearFactor where
  a : ℤ
  b : ℤ

/-- Function to check if a quadratic can be factored into two linear factors -/
def isFactorable (q : Quadratic) (l1 l2 : LinearFactor) : Prop :=
  q.a = l1.a * l2.a ∧
  q.b = l1.a * l2.b + l1.b * l2.a ∧
  q.c = l1.b * l2.b

/-- The main theorem stating the largest value of n -/
theorem largest_n_for_factorable_quadratic :
  ∃ (n : ℤ),
    n = 451 ∧
    (∀ m : ℤ, m > n → 
      ¬∃ (l1 l2 : LinearFactor), 
        isFactorable ⟨5, m, 90⟩ l1 l2) ∧
    (∃ (l1 l2 : LinearFactor), 
      isFactorable ⟨5, n, 90⟩ l1 l2) :=
by sorry

end largest_n_for_factorable_quadratic_l730_73098


namespace f_shifted_l730_73090

def f (x : ℝ) : ℝ := 2 * x + 1

theorem f_shifted (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 3) :
  ∃ y, f (y - 1) = 2 * y - 1 ∧ 2 ≤ y ∧ y ≤ 4 :=
by sorry

end f_shifted_l730_73090


namespace candies_given_away_l730_73051

/-- Given a girl who initially had 60 candies and now has 20 left, 
    prove that she gave away 40 candies. -/
theorem candies_given_away (initial : ℕ) (remaining : ℕ) (given_away : ℕ) : 
  initial = 60 → remaining = 20 → given_away = initial - remaining → given_away = 40 := by
  sorry

end candies_given_away_l730_73051


namespace hyperbola_equation_l730_73059

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1 ∧ a > 0 ∧ b > 0

-- Define the focus
def focus (x y : ℝ) : Prop := x = 2 ∧ y = 0

-- Define the eccentricity
def eccentricity (e : ℝ) : Prop := e = 2

-- Theorem statement
theorem hyperbola_equation (a b : ℝ) :
  (∃ x y : ℝ, hyperbola a b x y ∧ focus x y) →
  eccentricity 2 →
  (∀ x y : ℝ, hyperbola a b x y ↔ x^2 - y^2/3 = 1) :=
by sorry

end hyperbola_equation_l730_73059


namespace road_repair_theorem_l730_73050

/-- The number of persons in the first group -/
def first_group : ℕ := 42

/-- The number of days the first group works -/
def first_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_hours : ℕ := 5

/-- The number of days the second group works -/
def second_days : ℕ := 14

/-- The number of hours per day the second group works -/
def second_hours : ℕ := 6

/-- The total man-hours required to complete the work -/
def total_man_hours : ℕ := first_group * first_days * first_hours

/-- The number of persons in the second group -/
def second_group : ℕ := total_man_hours / (second_days * second_hours)

theorem road_repair_theorem : second_group = 30 := by
  sorry

end road_repair_theorem_l730_73050


namespace power_function_value_l730_73020

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x^α

-- Define the theorem
theorem power_function_value (f : ℝ → ℝ) :
  isPowerFunction f → f 4 = 1/2 → f (1/4) = 2 := by
  sorry

end power_function_value_l730_73020


namespace average_weight_abc_l730_73011

theorem average_weight_abc (a b c : ℝ) : 
  (a + b) / 2 = 42 →
  (b + c) / 2 = 43 →
  b = 35 →
  (a + b + c) / 3 = 45 := by
sorry

end average_weight_abc_l730_73011


namespace smallest_X_value_l730_73032

/-- A function that checks if a natural number is composed only of 0s and 1s -/
def isComposedOf0sAnd1s (n : ℕ) : Prop := sorry

/-- The smallest positive integer composed of 0s and 1s that is divisible by 6 -/
def smallestValidT : ℕ := 1110

theorem smallest_X_value :
  ∀ T : ℕ,
  T > 0 →
  isComposedOf0sAnd1s T →
  T % 6 = 0 →
  T / 6 ≥ 185 :=
sorry

end smallest_X_value_l730_73032


namespace geometric_sequence_formula_l730_73038

/-- Given a geometric sequence {a_n} with sum S_n of first n terms, prove the general formula. -/
theorem geometric_sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  a 3 = 3/2 →                   -- given a_3
  S 3 = 9/2 →                   -- given S_3
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- sum formula for geometric sequence
  (q = 1 ∨ q = -1/2) ∧
  (∀ n, (q = 1 → a n = 3/2) ∧ 
        (q = -1/2 → a n = 6 * (-1/2)^(n-1))) :=
by sorry

end geometric_sequence_formula_l730_73038


namespace smallest_block_with_360_hidden_l730_73071

/-- Given a rectangular block made of unit cubes, this function calculates
    the number of hidden cubes when three surfaces are visible. -/
def hidden_cubes (l m n : ℕ) : ℕ := (l - 1) * (m - 1) * (n - 1)

/-- The total number of cubes in the rectangular block. -/
def total_cubes (l m n : ℕ) : ℕ := l * m * n

/-- Theorem stating that the smallest possible number of cubes in a rectangular block
    with 360 hidden cubes when three surfaces are visible is 560. -/
theorem smallest_block_with_360_hidden : 
  (∃ l m n : ℕ, 
    l > 1 ∧ m > 1 ∧ n > 1 ∧ 
    hidden_cubes l m n = 360 ∧
    (∀ l' m' n' : ℕ, 
      l' > 1 → m' > 1 → n' > 1 → 
      hidden_cubes l' m' n' = 360 → 
      total_cubes l m n ≤ total_cubes l' m' n')) ∧
  (∀ l m n : ℕ,
    l > 1 → m > 1 → n > 1 →
    hidden_cubes l m n = 360 →
    total_cubes l m n ≥ 560) := by
  sorry

end smallest_block_with_360_hidden_l730_73071


namespace shelf_board_length_l730_73021

/-- Proves that given a board of 143 cm, after cutting 25 cm and then 7 cm, 
    the length of other boards before the 7 cm cut was 125 cm. -/
theorem shelf_board_length 
  (initial_length : ℕ) 
  (first_cut : ℕ) 
  (final_adjustment : ℕ) 
  (h1 : initial_length = 143)
  (h2 : first_cut = 25)
  (h3 : final_adjustment = 7) :
  initial_length - first_cut - final_adjustment + final_adjustment = 125 :=
by
  sorry

end shelf_board_length_l730_73021


namespace integer_solution_range_l730_73048

theorem integer_solution_range (b : ℝ) : 
  (∀ x : ℤ, |3 * (x : ℝ) - b| < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) → 
  (5 < b ∧ b < 7) :=
by sorry

end integer_solution_range_l730_73048


namespace wrong_mark_value_l730_73089

/-- Proves that the wrongly entered mark is 85 given the conditions of the problem -/
theorem wrong_mark_value (n : ℕ) (correct_mark : ℕ) (average_increase : ℚ) 
  (h1 : n = 104) 
  (h2 : correct_mark = 33) 
  (h3 : average_increase = 1/2) : 
  ∃ x : ℕ, x = 85 ∧ (x - correct_mark : ℚ) = average_increase * n := by
  sorry

end wrong_mark_value_l730_73089


namespace candy_store_revenue_l730_73097

/-- Calculates the revenue of a candy store given specific sales conditions --/
theorem candy_store_revenue :
  let fudge_pounds : ℕ := 37
  let fudge_price : ℚ := 5/2
  let truffle_count : ℕ := 82
  let truffle_price : ℚ := 3/2
  let pretzel_count : ℕ := 48
  let pretzel_price : ℚ := 2
  let fudge_discount : ℚ := 1/10
  let sales_tax : ℚ := 1/20
  let truffle_promo : ℕ := 3  -- buy 3, get 1 free

  let fudge_revenue := (1 - fudge_discount) * (fudge_pounds : ℚ) * fudge_price
  let truffle_revenue := (truffle_count - truffle_count / (truffle_promo + 1)) * truffle_price
  let pretzel_revenue := (pretzel_count : ℚ) * pretzel_price
  
  let total_before_tax := fudge_revenue + truffle_revenue + pretzel_revenue
  let total_after_tax := total_before_tax * (1 + sales_tax)

  total_after_tax = 28586 / 100
  := by sorry

end candy_store_revenue_l730_73097


namespace two_fifths_in_three_fourths_l730_73041

theorem two_fifths_in_three_fourths : (3 : ℚ) / 4 / ((2 : ℚ) / 5) = 15 / 8 := by
  sorry

end two_fifths_in_three_fourths_l730_73041


namespace fibonacci_factorial_sum_last_two_digits_l730_73053

def fibonacci_factorial_series := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

def last_two_digits (n : ℕ) : ℕ := n % 100

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem fibonacci_factorial_sum_last_two_digits : 
  (fibonacci_factorial_series.map (λ x => last_two_digits (factorial x))).sum = 50 := by
  sorry

end fibonacci_factorial_sum_last_two_digits_l730_73053


namespace range_of_a_l730_73082

-- Define the sets p and q
def p (x : ℝ) : Prop := x^2 - x - 2 ≥ 0
def q (a x : ℝ) : Prop := 2*a - 1 ≤ x ∧ x ≤ a + 3

-- Define the property that ¬p is a necessary but not sufficient condition for q
def not_p_necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, q a x → ¬(p x)) ∧ (∃ x, ¬(p x) ∧ ¬(q a x))

-- State the theorem
theorem range_of_a (a : ℝ) :
  not_p_necessary_not_sufficient a ↔ -1 ≤ a ∧ a ≤ 0 :=
sorry

end range_of_a_l730_73082
