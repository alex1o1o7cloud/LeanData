import Mathlib

namespace max_value_product_l1705_170508

theorem max_value_product (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) :
  (x^2 + x*y + y^2) * (x^2 + x*z + z^2) * (y^2 + y*z + z^2) ≤ 27 :=
by sorry

end max_value_product_l1705_170508


namespace units_digit_of_expression_l1705_170575

theorem units_digit_of_expression (k : ℕ) : k = 2025^2 + 3^2025 → (k^2 + 3^k) % 10 = 5 := by
  sorry

end units_digit_of_expression_l1705_170575


namespace snow_probability_l1705_170533

theorem snow_probability : 
  let p1 : ℚ := 1/5  -- probability of snow for each of the first 5 days
  let p2 : ℚ := 1/3  -- probability of snow for each of the next 5 days
  let days1 : ℕ := 5  -- number of days with probability p1
  let days2 : ℕ := 5  -- number of days with probability p2
  let prob_at_least_one_snow : ℚ := 1 - (1 - p1)^days1 * (1 - p2)^days2
  prob_at_least_one_snow = 726607/759375 := by
sorry

end snow_probability_l1705_170533


namespace largest_of_three_consecutive_integers_sum_18_l1705_170574

theorem largest_of_three_consecutive_integers_sum_18 (a b c : ℤ) : 
  (b = a + 1) →  -- b is the next consecutive integer after a
  (c = b + 1) →  -- c is the next consecutive integer after b
  (a + b + c = 18) →  -- sum of the three integers is 18
  (c = 7) -- c (the largest) is 7
:= by sorry

end largest_of_three_consecutive_integers_sum_18_l1705_170574


namespace cuts_for_331_pieces_l1705_170511

/-- The number of cuts needed to transform initial sheets into a given number of pieces -/
def number_of_cuts (initial_sheets : ℕ) (final_pieces : ℕ) : ℕ :=
  (final_pieces - initial_sheets) / 6

/-- Theorem stating that 54 cuts are needed to transform 7 sheets into 331 pieces -/
theorem cuts_for_331_pieces : number_of_cuts 7 331 = 54 := by
  sorry

end cuts_for_331_pieces_l1705_170511


namespace constant_sequence_l1705_170529

def is_prime (n : ℤ) : Prop := Nat.Prime n.natAbs

theorem constant_sequence
  (a : ℕ → ℤ)  -- Sequence of integers
  (d : ℤ)      -- Integer d
  (h1 : ∀ n, is_prime (a n))  -- |a_n| is prime for all n
  (h2 : ∀ n, a (n + 2) = a (n + 1) + a n + d)  -- Recurrence relation
  : ∀ n, a n = a 0  -- Conclusion: sequence is constant
  := by sorry

end constant_sequence_l1705_170529


namespace sum_of_reciprocal_equations_l1705_170559

theorem sum_of_reciprocal_equations (x y : ℚ) 
  (h1 : x⁻¹ + y⁻¹ = 3)
  (h2 : x⁻¹ - y⁻¹ = -7) : 
  x + y = -3/10 := by
  sorry

end sum_of_reciprocal_equations_l1705_170559


namespace rhombus_field_area_l1705_170538

/-- Represents the length of the long diagonal of a rhombus-shaped field in miles. -/
def long_diagonal : ℝ := 2500

/-- Represents the area of the rhombus-shaped field in square miles. -/
def field_area : ℝ := 1562500

/-- Theorem stating that the area of the rhombus-shaped field is 1562500 square miles. -/
theorem rhombus_field_area : field_area = (1 / 2) * long_diagonal * (long_diagonal / 2) := by
  sorry

#check rhombus_field_area

end rhombus_field_area_l1705_170538


namespace garden_area_l1705_170548

/-- The total area of a garden with a semicircle and an attached square -/
theorem garden_area (diameter : ℝ) (h : diameter = 8) : 
  let radius := diameter / 2
  let semicircle_area := π * radius^2 / 2
  let square_area := radius^2
  semicircle_area + square_area = 8 * π + 16 := by
  sorry

#check garden_area

end garden_area_l1705_170548


namespace expected_value_of_event_A_l1705_170500

theorem expected_value_of_event_A (p : ℝ) (h1 : (1 - p) ^ 4 = 16 / 81) :
  4 * p = 4 / 3 := by
  sorry

end expected_value_of_event_A_l1705_170500


namespace eight_flavors_twentyeight_sundaes_l1705_170560

/-- The number of unique two scoop sundaes with distinct flavors given n flavors of ice cream -/
def uniqueSundaes (n : ℕ) : ℕ := Nat.choose n 2

/-- Theorem stating that with 8 flavors, there are 28 unique two scoop sundaes -/
theorem eight_flavors_twentyeight_sundaes : uniqueSundaes 8 = 28 := by
  sorry

end eight_flavors_twentyeight_sundaes_l1705_170560


namespace min_value_of_expression_l1705_170507

theorem min_value_of_expression (x : ℝ) :
  ∃ (m : ℝ), m = -784 ∧ ∀ (y : ℝ), (15 - y) * (13 - y) * (15 + y) * (13 + y) ≥ m :=
by sorry

end min_value_of_expression_l1705_170507


namespace musical_group_seats_l1705_170554

/-- Represents the number of seats needed for a musical group --/
def total_seats (F T Tr D C H S P V G : ℕ) : ℕ :=
  F + T + Tr + D + C + H + S + P + V + G

/-- Theorem stating the total number of seats needed for the musical group --/
theorem musical_group_seats :
  ∀ (F T Tr D C H S P V G : ℕ),
    F = 5 →
    T = 3 * F →
    Tr = T - 8 →
    D = Tr + 11 →
    C = 2 * F →
    H = Tr + 3 →
    S = (T + Tr) / 2 →
    P = D + 2 →
    V = H - C →
    G = 3 * F →
    total_seats F T Tr D C H S P V G = 111 :=
by
  sorry

end musical_group_seats_l1705_170554


namespace problem_statement_l1705_170543

theorem problem_statement (x y n : ℝ) : 
  x = 3 → y = 0 → n = x - y^(x+y) → n = 3 := by sorry

end problem_statement_l1705_170543


namespace employee_recorder_price_l1705_170588

-- Define the wholesale cost
def wholesale_cost : ℝ := 200

-- Define the markup percentage
def markup_percentage : ℝ := 0.20

-- Define the employee discount percentage
def employee_discount_percentage : ℝ := 0.05

-- Define the retail price calculation
def retail_price : ℝ := wholesale_cost * (1 + markup_percentage)

-- Define the employee price calculation
def employee_price : ℝ := retail_price * (1 - employee_discount_percentage)

-- Theorem statement
theorem employee_recorder_price : employee_price = 228 := by
  sorry

end employee_recorder_price_l1705_170588


namespace largest_inscribed_rectangle_area_l1705_170578

theorem largest_inscribed_rectangle_area (r : ℝ) (h : r = 6) :
  let d := 2 * r
  let s := d / Real.sqrt 2
  s * s = 72 := by sorry

end largest_inscribed_rectangle_area_l1705_170578


namespace seating_arrangement_l1705_170526

theorem seating_arrangement (total_people : ℕ) (total_rows : ℕ) 
  (h1 : total_people = 97) 
  (h2 : total_rows = 13) : 
  ∃ (rows_with_8 : ℕ), 
    rows_with_8 * 8 + (total_rows - rows_with_8) * 7 = total_people ∧ 
    rows_with_8 = 6 := by
  sorry

end seating_arrangement_l1705_170526


namespace range_of_a_l1705_170502

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x) ∧ 
  (∃ x : ℝ, x^2 + 4*x + a = 0) → 
  a ∈ Set.Icc (Real.exp 1) 4 := by
sorry

end range_of_a_l1705_170502


namespace kind_wizard_succeeds_for_odd_n_l1705_170568

/-- Represents a friendship between two dwarves -/
structure Friendship :=
  (dwarf1 : ℕ)
  (dwarf2 : ℕ)

/-- Creates a list of friendships based on the wizard's pairing strategy -/
def createFriendships (n : ℕ) : List Friendship := sorry

/-- Breaks n friendships from the list -/
def breakFriendships (friendships : List Friendship) (n : ℕ) : List Friendship := sorry

/-- Checks if the remaining friendships can form a valid circular arrangement -/
def canFormCircularArrangement (friendships : List Friendship) : Prop := sorry

theorem kind_wizard_succeeds_for_odd_n (n : ℕ) (h : Odd n) :
  ∀ (broken : List Friendship),
    broken.length = n →
    canFormCircularArrangement (breakFriendships (createFriendships n) n) :=
sorry

end kind_wizard_succeeds_for_odd_n_l1705_170568


namespace quadratic_one_solution_l1705_170513

theorem quadratic_one_solution (m : ℝ) : 
  (∃! x, 16 * x^2 + m * x + 4 = 0) ↔ m = 16 :=
by sorry

end quadratic_one_solution_l1705_170513


namespace coffee_shop_spending_l1705_170510

theorem coffee_shop_spending (A B : ℝ) : 
  B = 0.5 * A → A = B + 15 → A + B = 45 := by sorry

end coffee_shop_spending_l1705_170510


namespace compound_interest_principal_l1705_170520

/-- Given a sum of 5292 after 2 years with an interest rate of 5% per annum compounded yearly, 
    prove that the principal amount is 4800. -/
theorem compound_interest_principal (sum : ℝ) (years : ℕ) (rate : ℝ) (principal : ℝ) : 
  sum = 5292 →
  years = 2 →
  rate = 0.05 →
  sum = principal * (1 + rate) ^ years →
  principal = 4800 := by
  sorry

end compound_interest_principal_l1705_170520


namespace probability_at_least_one_black_l1705_170501

def total_balls : ℕ := 10
def red_balls : ℕ := 6
def black_balls : ℕ := 4
def selected_balls : ℕ := 4

theorem probability_at_least_one_black :
  let total_ways := Nat.choose total_balls selected_balls
  let all_red_ways := Nat.choose red_balls selected_balls
  let at_least_one_black_ways := total_ways - all_red_ways
  (at_least_one_black_ways : ℚ) / total_ways = 13 / 14 := by
  sorry

end probability_at_least_one_black_l1705_170501


namespace four_greater_than_sqrt_fourteen_l1705_170553

theorem four_greater_than_sqrt_fourteen : 4 > Real.sqrt 14 := by
  sorry

end four_greater_than_sqrt_fourteen_l1705_170553


namespace percentage_increase_proof_l1705_170598

theorem percentage_increase_proof (initial_earnings new_earnings : ℝ) 
  (h1 : initial_earnings = 60)
  (h2 : new_earnings = 110) :
  (new_earnings - initial_earnings) / initial_earnings * 100 = 83.33 := by
  sorry

end percentage_increase_proof_l1705_170598


namespace wise_men_strategy_l1705_170586

/-- Represents the color of a hat -/
inductive HatColor
| White
| Black

/-- Represents a wise man with a hat -/
structure WiseMan where
  hat : HatColor

/-- Represents the line of wise men -/
def WiseMenLine := List WiseMan

/-- A strategy is a function that takes the visible hats and returns a guess -/
def Strategy := (visible : WiseMenLine) → HatColor

/-- Counts the number of correct guesses given a line of wise men and a strategy -/
def countCorrectGuesses (line : WiseMenLine) (strategy : Strategy) : Nat :=
  sorry

/-- The main theorem: there exists a strategy where at least n-1 wise men guess correctly -/
theorem wise_men_strategy (n : Nat) :
  ∃ (strategy : Strategy), ∀ (line : WiseMenLine),
    line.length = n →
    countCorrectGuesses line strategy ≥ n - 1 :=
  sorry

end wise_men_strategy_l1705_170586


namespace unique_cube_pair_l1705_170535

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def has_unique_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = digits.toFinset.card

theorem unique_cube_pair :
  ∃! (a b : ℕ),
    1000 ≤ a ∧ a < 10000 ∧
    100 ≤ b ∧ b < 1000 ∧
    is_perfect_cube a ∧
    is_perfect_cube b ∧
    a / 100 = b / 10 ∧
    has_unique_digits a ∧
    has_unique_digits b ∧
    a = 1728 ∧
    b = 125 := by
  sorry

end unique_cube_pair_l1705_170535


namespace remainder_2567139_div_6_l1705_170527

theorem remainder_2567139_div_6 : 2567139 % 6 = 3 := by
  sorry

end remainder_2567139_div_6_l1705_170527


namespace archer_expected_hits_l1705_170577

/-- The expected value of a binomial distribution with n trials and probability p -/
def binomial_expectation (n : ℕ) (p : ℝ) : ℝ := n * p

/-- The number of shots taken by the archer -/
def num_shots : ℕ := 10

/-- The probability of hitting the bullseye -/
def hit_probability : ℝ := 0.9

/-- Theorem: The expected number of bullseye hits for the archer -/
theorem archer_expected_hits : 
  binomial_expectation num_shots hit_probability = 9 := by
  sorry

end archer_expected_hits_l1705_170577


namespace parabola_shift_theorem_l1705_170570

/-- Represents a parabola of the form y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a, h := p.h + dx, k := p.k + dy }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = 3 ∧ p.h = 1 ∧ p.k = 1 →
  let shifted := shift (shift p 2 0) 0 2
  shifted.a = 3 ∧ shifted.h = 3 ∧ shifted.k = 3 := by
  sorry

#check parabola_shift_theorem

end parabola_shift_theorem_l1705_170570


namespace product_equivalence_l1705_170544

theorem product_equivalence (h : 213 * 16 = 3408) : 1.6 * 2.13 = 3.408 := by
  sorry

end product_equivalence_l1705_170544


namespace day_of_week_N_minus_1_l1705_170552

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific day in a year -/
structure YearDay where
  year : Int
  dayNumber : Nat

/-- Function to determine the day of the week for a given YearDay -/
def dayOfWeek (yd : YearDay) : DayOfWeek :=
  sorry

theorem day_of_week_N_minus_1 
  (N : Int)
  (h1 : dayOfWeek ⟨N, 250⟩ = DayOfWeek.Friday)
  (h2 : dayOfWeek ⟨N+1, 150⟩ = DayOfWeek.Friday) :
  dayOfWeek ⟨N-1, 250⟩ = DayOfWeek.Saturday :=
sorry

end day_of_week_N_minus_1_l1705_170552


namespace increasing_power_function_m_l1705_170539

/-- A power function f(x) = (m^2 - 3)x^(m+1) is increasing on (0, +∞) -/
def is_increasing_power_function (m : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → Monotone (fun x => (m^2 - 3) * x^(m+1))

/-- The value of m for which the power function is increasing -/
theorem increasing_power_function_m : 
  ∃ m : ℝ, is_increasing_power_function m ∧ m = 2 :=
sorry

end increasing_power_function_m_l1705_170539


namespace matrix_paths_count_l1705_170564

/-- Represents a position on the 5x5 grid -/
structure Position :=
  (row : Fin 5)
  (col : Fin 5)

/-- Represents a letter in the word "MATRIX" -/
inductive Letter
| M | A | T | R | I | X

/-- Represents the 5x5 grid of letters -/
def grid : Position → Letter := sorry

/-- Checks if two positions are adjacent (horizontally, vertically, or diagonally) -/
def adjacent (p1 p2 : Position) : Prop := sorry

/-- Represents a valid path spelling "MATRIX" -/
def valid_path (path : List Position) : Prop := sorry

/-- Counts the number of valid paths starting from a given position -/
def count_paths_from (start : Position) : ℕ := sorry

/-- Counts the total number of valid paths in the grid -/
def total_paths : ℕ := sorry

/-- Theorem stating that the total number of paths spelling "MATRIX" is 48 -/
theorem matrix_paths_count :
  total_paths = 48 := by sorry

end matrix_paths_count_l1705_170564


namespace bert_profit_l1705_170596

def selling_price : ℝ := 90
def markup : ℝ := 10
def tax_rate : ℝ := 0.1

theorem bert_profit : 
  let cost_price := selling_price - markup
  let tax := selling_price * tax_rate
  selling_price - cost_price - tax = 1 := by
  sorry

end bert_profit_l1705_170596


namespace polynomial_product_bound_l1705_170518

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The property that |P(x) * P(1/x)| ≤ 1 for all positive real x -/
def HasBoundedProduct (P : RealPolynomial) : Prop :=
  ∀ x : ℝ, x > 0 → |P x * P (1/x)| ≤ 1

/-- The form c * x^n where |c| ≤ 1 and n is a non-negative integer -/
def IsMonomial (P : RealPolynomial) : Prop :=
  ∃ (c : ℝ) (n : ℕ), (|c| ≤ 1) ∧ (∀ x : ℝ, P x = c * x^n)

/-- The main theorem -/
theorem polynomial_product_bound (P : RealPolynomial) :
  HasBoundedProduct P → IsMonomial P := by
  sorry

end polynomial_product_bound_l1705_170518


namespace train_meeting_point_train_A_distance_l1705_170583

theorem train_meeting_point (total_distance : ℝ) (time_A time_B : ℝ) (h1 : total_distance = 75) 
  (h2 : time_A = 3) (h3 : time_B = 2) : ℝ :=
  let speed_A := total_distance / time_A
  let speed_B := total_distance / time_B
  let relative_speed := speed_A + speed_B
  let meeting_time := total_distance / relative_speed
  speed_A * meeting_time

theorem train_A_distance (total_distance : ℝ) (time_A time_B : ℝ) (h1 : total_distance = 75) 
  (h2 : time_A = 3) (h3 : time_B = 2) : 
  train_meeting_point total_distance time_A time_B h1 h2 h3 = 30 := by
  sorry

end train_meeting_point_train_A_distance_l1705_170583


namespace divisibility_after_subtraction_l1705_170506

theorem divisibility_after_subtraction :
  ∃ (n : ℕ), n = 15 ∧ (427398 - 3) % n = 0 :=
by
  sorry

end divisibility_after_subtraction_l1705_170506


namespace exponential_inequality_l1705_170532

theorem exponential_inequality (x : ℝ) : 3^x < (1:ℝ)/27 ↔ x < -3 := by
  sorry

end exponential_inequality_l1705_170532


namespace unique_prime_with_remainder_l1705_170528

theorem unique_prime_with_remainder : ∃! n : ℕ, 
  40 < n ∧ n < 50 ∧ 
  Nat.Prime n ∧ 
  n % 9 = 5 :=
by
  sorry

end unique_prime_with_remainder_l1705_170528


namespace smallest_among_four_rationals_l1705_170536

theorem smallest_among_four_rationals :
  let S : Set ℚ := {-1, 0, 1, 2}
  ∀ x ∈ S, -1 ≤ x
  ∧ ∃ y ∈ S, y = -1 := by
  sorry

end smallest_among_four_rationals_l1705_170536


namespace sienas_initial_bookmarks_l1705_170557

/-- Calculates the number of pages Siena had before March, given her daily bookmarking rate and final page count. -/
theorem sienas_initial_bookmarks (daily_bookmarks : ℕ) (march_days : ℕ) (final_count : ℕ) : 
  daily_bookmarks = 30 → 
  march_days = 31 → 
  final_count = 1330 → 
  final_count - (daily_bookmarks * march_days) = 400 :=
by
  sorry

#check sienas_initial_bookmarks

end sienas_initial_bookmarks_l1705_170557


namespace polynomial_factorization_l1705_170582

theorem polynomial_factorization (m n a b : ℝ) : 
  (|m - 4| + (n^2 - 8*n + 16) = 0) → 
  (a^2 + 4*b^2 - m*a*b - n = (a - 2*b + 2) * (a - 2*b - 2)) := by
sorry

end polynomial_factorization_l1705_170582


namespace people_per_car_l1705_170521

theorem people_per_car (total_people : ℕ) (num_cars : ℕ) (h1 : total_people = 63) (h2 : num_cars = 9) :
  total_people / num_cars = 7 := by
sorry

end people_per_car_l1705_170521


namespace sin_2a_minus_pi_6_l1705_170550

theorem sin_2a_minus_pi_6 (a : ℝ) (h : Real.sin (π / 3 - a) = 1 / 4) :
  Real.sin (2 * a - π / 6) = 7 / 8 := by
  sorry

end sin_2a_minus_pi_6_l1705_170550


namespace cyclist_speed_l1705_170523

/-- Proves that given a hiker walking at 4 km/h and a cyclist who stops 5 minutes after passing the hiker,
    if it takes the hiker 17.5 minutes to catch up to the cyclist, then the cyclist's speed is 14 km/h. -/
theorem cyclist_speed (hiker_speed : ℝ) (cyclist_ride_time : ℝ) (hiker_catch_up_time : ℝ) :
  hiker_speed = 4 →
  cyclist_ride_time = 5 / 60 →
  hiker_catch_up_time = 17.5 / 60 →
  ∃ (cyclist_speed : ℝ),
    cyclist_speed * cyclist_ride_time = hiker_speed * (cyclist_ride_time + hiker_catch_up_time) ∧
    cyclist_speed = 14 := by
  sorry

end cyclist_speed_l1705_170523


namespace no_less_equal_two_mo_l1705_170566

theorem no_less_equal_two_mo (N O M : ℝ) (h : N * O ≤ 2 * M * O) : N * O ≤ 2 * M * O := by
  sorry

end no_less_equal_two_mo_l1705_170566


namespace percentage_increase_l1705_170592

theorem percentage_increase (x : ℝ) : 
  x > 98 ∧ x = 117.6 → (x - 98) / 98 * 100 = 20 := by
  sorry

end percentage_increase_l1705_170592


namespace not_all_monotonic_functions_have_extremum_l1705_170585

-- Define a monotonic function
def MonotonicFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

-- Define the existence of an extremum value
def HasExtremum (f : ℝ → ℝ) : Prop :=
  ∃ x, ∀ y, f y ≤ f x ∨ f x ≤ f y

-- Theorem statement
theorem not_all_monotonic_functions_have_extremum :
  ∃ f : ℝ → ℝ, MonotonicFunction f ∧ ¬HasExtremum f := by
  sorry

end not_all_monotonic_functions_have_extremum_l1705_170585


namespace simplified_expression_l1705_170555

theorem simplified_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2 * x⁻¹ + 3 * y⁻¹)⁻¹ = (x * y) / (2 * y + 3 * x) :=
by sorry

end simplified_expression_l1705_170555


namespace rajdhani_speed_calculation_l1705_170569

/-- The speed of Bombay Express in km/h -/
def bombay_speed : ℝ := 60

/-- The time difference between the departures of the two trains in hours -/
def time_difference : ℝ := 2

/-- The distance at which the two trains meet in km -/
def meeting_distance : ℝ := 480

/-- The speed of Rajdhani Express in km/h -/
def rajdhani_speed : ℝ := 80

theorem rajdhani_speed_calculation :
  let distance_covered_by_bombay : ℝ := bombay_speed * time_difference
  let remaining_distance : ℝ := meeting_distance - distance_covered_by_bombay
  let time_to_meet : ℝ := remaining_distance / bombay_speed
  rajdhani_speed = meeting_distance / time_to_meet :=
by sorry

end rajdhani_speed_calculation_l1705_170569


namespace quadratic_equation_solution_l1705_170525

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := fun x ↦ 5 * x^2 - 2 * x
  ∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 2/5 ∧ 
    (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end quadratic_equation_solution_l1705_170525


namespace cos15_cos45_minus_sin165_sin45_l1705_170522

theorem cos15_cos45_minus_sin165_sin45 :
  Real.cos (15 * π / 180) * Real.cos (45 * π / 180) - 
  Real.sin (165 * π / 180) * Real.sin (45 * π / 180) = 1 / 2 := by
  sorry

end cos15_cos45_minus_sin165_sin45_l1705_170522


namespace isosceles_trapezoid_theorem_l1705_170561

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The length of the smaller base -/
  smallerBase : ℝ
  /-- The height of the trapezoid -/
  height : ℝ
  /-- The diagonal is perpendicular to the lateral side -/
  diagonalPerpendicular : Bool

/-- Properties of the isosceles trapezoid -/
def trapezoidProperties : IsoscelesTrapezoid :=
  { smallerBase := 3
  , height := 2
  , diagonalPerpendicular := true }

/-- The theorem stating the properties of the isosceles trapezoid -/
theorem isosceles_trapezoid_theorem (t : IsoscelesTrapezoid) 
  (h1 : t = trapezoidProperties) :
  ∃ (largerBase acuteAngle : ℝ),
    largerBase = 5 ∧ 
    acuteAngle = Real.arctan 2 :=
sorry

end isosceles_trapezoid_theorem_l1705_170561


namespace triangle_shape_l1705_170515

theorem triangle_shape (a b : ℝ) (A B : ℝ) :
  0 < A → A < π →
  0 < B → B < π →
  a^2 * Real.tan B = b^2 * Real.tan A →
  (A = B ∨ A + B = π / 2) :=
sorry

end triangle_shape_l1705_170515


namespace least_three_digit_multiple_of_13_l1705_170599

theorem least_three_digit_multiple_of_13 : ∃ n : ℕ, 
  13 * n = 104 ∧ 
  104 ≥ 100 ∧
  104 < 1000 ∧
  ∀ m : ℕ, (13 * m ≥ 100 ∧ 13 * m < 1000) → 13 * m ≥ 104 := by
  sorry

end least_three_digit_multiple_of_13_l1705_170599


namespace elvin_internet_charge_l1705_170576

/-- Represents Elvin's monthly telephone bill structure -/
structure TelephoneBill where
  fixedCharge : ℝ
  callCharge : ℝ

/-- Calculates the total bill for a given month -/
def totalBill (bill : TelephoneBill) : ℝ :=
  bill.fixedCharge + bill.callCharge

theorem elvin_internet_charge : 
  ∀ (jan : TelephoneBill) (feb : TelephoneBill),
    totalBill jan = 50 →
    totalBill feb = 76 →
    feb.callCharge = 2 * jan.callCharge →
    jan.fixedCharge = feb.fixedCharge →
    jan.fixedCharge = 24 := by
  sorry


end elvin_internet_charge_l1705_170576


namespace max_value_of_sum_and_reciprocal_l1705_170541

theorem max_value_of_sum_and_reciprocal (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 15 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 15 :=
by sorry

end max_value_of_sum_and_reciprocal_l1705_170541


namespace tensor_self_zero_tensor_dot_product_identity_l1705_170593

/-- Definition of the ⊗ operation for 2D vectors -/
def tensor (a b : ℝ × ℝ) : ℝ := a.1 * a.2 - b.1 * b.2

/-- The dot product of two 2D vectors -/
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

theorem tensor_self_zero (a : ℝ × ℝ) : tensor a a = 0 := by sorry

theorem tensor_dot_product_identity (a b : ℝ × ℝ) :
  (tensor a b)^2 + (dot_product a b)^2 = (a.1^2 + b.2^2) * (a.2^2 + b.1^2) := by sorry

end tensor_self_zero_tensor_dot_product_identity_l1705_170593


namespace nearest_integer_to_three_plus_sqrt_three_fourth_l1705_170571

theorem nearest_integer_to_three_plus_sqrt_three_fourth (x : ℝ) : 
  x = (3 + Real.sqrt 3)^4 → 
  ∃ n : ℤ, n = 504 ∧ ∀ m : ℤ, |x - n| ≤ |x - m| := by
  sorry

end nearest_integer_to_three_plus_sqrt_three_fourth_l1705_170571


namespace oil_price_reduction_l1705_170590

/-- Proves that given a 10% reduction in the price of oil, if a housewife can obtain 6 kgs more 
    for Rs. 900 after the reduction, then the reduced price per kg of oil is Rs. 15. -/
theorem oil_price_reduction (original_price : ℝ) : 
  let reduced_price := original_price * 0.9
  let original_quantity := 900 / original_price
  let new_quantity := 900 / reduced_price
  new_quantity = original_quantity + 6 →
  reduced_price = 15 := by
  sorry

end oil_price_reduction_l1705_170590


namespace tower_surface_area_l1705_170537

/-- Calculates the visible surface area of a cube in the tower -/
def visibleSurfaceArea (sideLength : ℕ) (isTop : Bool) : ℕ :=
  if isTop then 5 * sideLength^2 else 4 * sideLength^2

/-- Represents the tower of cubes -/
def cubesTower : List ℕ := [9, 1, 7, 3, 5, 4, 6, 8]

/-- Calculates the total visible surface area of the tower -/
def totalVisibleSurfaceArea (tower : List ℕ) : ℕ :=
  let n := tower.length
  tower.enum.foldl (fun acc (i, sideLength) =>
    acc + visibleSurfaceArea sideLength (i == n - 1)) 0

theorem tower_surface_area :
  totalVisibleSurfaceArea cubesTower = 1408 := by
  sorry

#eval totalVisibleSurfaceArea cubesTower

end tower_surface_area_l1705_170537


namespace f_monotonicity_and_m_range_l1705_170503

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x - b * x

theorem f_monotonicity_and_m_range :
  ∀ (a b : ℝ),
  (∀ (x : ℝ), x > 0 → f a b x = a * Real.log x - b * x) →
  (b = 1 →
    ((a ≤ 0 → ∀ (x y : ℝ), 0 < x → x < y → f a b y < f a b x) ∧
     (a > 0 → (∀ (x y : ℝ), 0 < x → x < y → y < a → f a b x < f a b y) ∧
              (∀ (x y : ℝ), a < x → x < y → f a b y < f a b x)))) ∧
  (a = 1 →
    (∀ (x : ℝ), x > 0 → f a b x ≤ -1) →
    (∀ (x : ℝ), x > 0 → f a b x ≤ x * Real.exp x - (b + 1) * x - 1) →
    b ≤ 1) :=
by sorry

end f_monotonicity_and_m_range_l1705_170503


namespace rectangle_area_unchanged_l1705_170547

/-- Given a rectangle with area 432 square centimeters, prove that decreasing the length by 20%
    and increasing the width by 25% results in the same area. -/
theorem rectangle_area_unchanged (l w : ℝ) (h : l * w = 432) :
  (0.8 * l) * (1.25 * w) = 432 := by
  sorry

end rectangle_area_unchanged_l1705_170547


namespace water_depth_ratio_l1705_170546

theorem water_depth_ratio (dean_height : ℝ) (water_depth_difference : ℝ) :
  dean_height = 9 →
  water_depth_difference = 81 →
  (dean_height + water_depth_difference) / dean_height = 10 :=
by
  sorry

end water_depth_ratio_l1705_170546


namespace calculate_books_arlo_book_count_l1705_170514

/-- Given a ratio of books to pens and a total number of items, calculate the number of books. -/
theorem calculate_books (book_ratio : ℕ) (pen_ratio : ℕ) (total_items : ℕ) : ℕ :=
  let total_ratio := book_ratio + pen_ratio
  let items_per_part := total_items / total_ratio
  book_ratio * items_per_part

/-- Prove that given a ratio of books to pens of 7:3 and a total of 400 stationery items, the number of books is 280. -/
theorem arlo_book_count : calculate_books 7 3 400 = 280 := by
  sorry

end calculate_books_arlo_book_count_l1705_170514


namespace max_house_paintable_area_l1705_170594

/-- The total area of walls to be painted in Max's house -/
def total_paintable_area (num_bedrooms : ℕ) (length width height : ℝ) (non_paintable_area : ℝ) : ℝ :=
  num_bedrooms * (2 * (length * height + width * height) - non_paintable_area)

/-- Theorem stating the total area of walls to be painted in Max's house -/
theorem max_house_paintable_area :
  total_paintable_area 4 15 12 9 80 = 1624 := by
  sorry

end max_house_paintable_area_l1705_170594


namespace last_three_digits_of_7_power_10000_l1705_170558

theorem last_three_digits_of_7_power_10000 (h : 7^250 ≡ 1 [ZMOD 1250]) :
  7^10000 ≡ 1 [ZMOD 1000] := by
  sorry

end last_three_digits_of_7_power_10000_l1705_170558


namespace equal_roots_quadratic_l1705_170517

theorem equal_roots_quadratic (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 4 * x + 1 = 0 ∧ 
   ∀ y : ℝ, a * y^2 - 4 * y + 1 = 0 → y = x) → 
  a = 4 := by
sorry

end equal_roots_quadratic_l1705_170517


namespace quadratic_equation_solution_l1705_170556

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ y => y^2 + 7*y + 10 + (y + 2)*(y + 8)
  (f (-2) = 0 ∧ f (-13/2) = 0) ∧
  ∀ y : ℝ, f y = 0 → (y = -2 ∨ y = -13/2) :=
by sorry

end quadratic_equation_solution_l1705_170556


namespace payroll_tax_threshold_l1705_170549

/-- The payroll tax problem -/
theorem payroll_tax_threshold (tax_rate : ℝ) (tax_paid : ℝ) (total_payroll : ℝ) (T : ℝ) : 
  tax_rate = 0.002 →
  tax_paid = 200 →
  total_payroll = 300000 →
  tax_paid = (total_payroll - T) * tax_rate →
  T = 200000 := by
  sorry


end payroll_tax_threshold_l1705_170549


namespace twenty_seven_power_divided_by_nine_l1705_170531

theorem twenty_seven_power_divided_by_nine (m : ℕ) :
  m = 27^1001 → m / 9 = 3^3001 := by
  sorry

end twenty_seven_power_divided_by_nine_l1705_170531


namespace only_30_40_50_is_pythagorean_triple_l1705_170516

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem only_30_40_50_is_pythagorean_triple :
  (is_pythagorean_triple 30 40 50) ∧
  ¬(is_pythagorean_triple 1 1 2) ∧
  ¬(is_pythagorean_triple 1 2 2) ∧
  ¬(is_pythagorean_triple 7 14 15) :=
by sorry

end only_30_40_50_is_pythagorean_triple_l1705_170516


namespace closed_path_vector_sum_l1705_170524

/-- The sum of vectors forming a closed path in a plane is equal to the zero vector. -/
theorem closed_path_vector_sum (A B C D E F : ℝ × ℝ) : 
  (B.1 - A.1, B.2 - A.2) + (C.1 - B.1, C.2 - B.2) + (D.1 - C.1, D.2 - C.2) + 
  (E.1 - D.1, E.2 - D.2) + (F.1 - E.1, F.2 - E.2) + (A.1 - F.1, A.2 - F.2) = (0, 0) := by
sorry

end closed_path_vector_sum_l1705_170524


namespace parabola_equation_l1705_170540

/-- A parabola with vertex at the origin and focus on the y-axis. -/
structure Parabola where
  p : ℝ  -- The focal parameter of the parabola

/-- The line y = 2x + 1 -/
def line (x : ℝ) : ℝ := 2 * x + 1

/-- The chord length intercepted by the line y = 2x + 1 on the parabola -/
def chordLength (p : Parabola) : ℝ := sorry

theorem parabola_equation (p : Parabola) :
  chordLength p = Real.sqrt 15 →
  (∀ x y : ℝ, y = p.p * x^2 ∨ y = -3 * p.p * x^2) :=
sorry

end parabola_equation_l1705_170540


namespace parallel_implies_m_eq_neg_one_perpendicular_implies_m_eq_one_plus_minus_two_sqrt_two_l1705_170534

-- Define the vectors
def OA : Fin 2 → ℝ := ![(-1 : ℝ), 3]
def OB : Fin 2 → ℝ := ![3, -1]
def OC (m : ℝ) : Fin 2 → ℝ := ![m, 1]

-- Define vector operations
def vector_sub (v w : Fin 2 → ℝ) : Fin 2 → ℝ := λ i => v i - w i
def parallel (v w : Fin 2 → ℝ) : Prop := ∃ k : ℝ, ∀ i, v i = k * w i
def perpendicular (v w : Fin 2 → ℝ) : Prop := v 0 * w 0 + v 1 * w 1 = 0

-- Define the theorems
theorem parallel_implies_m_eq_neg_one (m : ℝ) :
  parallel (vector_sub OB OA) (OC m) → m = -1 := by sorry

theorem perpendicular_implies_m_eq_one_plus_minus_two_sqrt_two (m : ℝ) :
  perpendicular (vector_sub (OC m) OA) (vector_sub (OC m) OB) →
  (m = 1 + 2 * Real.sqrt 2 ∨ m = 1 - 2 * Real.sqrt 2) := by sorry

end parallel_implies_m_eq_neg_one_perpendicular_implies_m_eq_one_plus_minus_two_sqrt_two_l1705_170534


namespace probability_both_genders_selected_l1705_170505

/-- The probability of selecting both boys and girls when randomly choosing 3 students from 2 boys and 3 girls -/
theorem probability_both_genders_selected (total_students : ℕ) (boys : ℕ) (girls : ℕ) (selected : ℕ) : 
  total_students = boys + girls →
  boys = 2 →
  girls = 3 →
  selected = 3 →
  (1 - (Nat.choose girls selected : ℚ) / (Nat.choose total_students selected : ℚ)) = 9/10 :=
by sorry

end probability_both_genders_selected_l1705_170505


namespace office_staff_composition_l1705_170512

/-- Represents the number of officers in an office. -/
def num_officers : ℕ := 15

/-- Represents the number of non-officers in the office. -/
def num_non_officers : ℕ := 480

/-- Represents the average salary of all employees in Rs/month. -/
def avg_salary_all : ℕ := 120

/-- Represents the average salary of officers in Rs/month. -/
def avg_salary_officers : ℕ := 440

/-- Represents the average salary of non-officers in Rs/month. -/
def avg_salary_non_officers : ℕ := 110

/-- Theorem stating that the number of officers is 15, given the conditions of the problem. -/
theorem office_staff_composition :
  num_officers = 15 ∧
  num_non_officers = 480 ∧
  avg_salary_all * (num_officers + num_non_officers) = 
    avg_salary_officers * num_officers + avg_salary_non_officers * num_non_officers :=
by sorry


end office_staff_composition_l1705_170512


namespace connected_paper_area_l1705_170567

/-- The area of connected square papers -/
theorem connected_paper_area 
  (num_papers : ℕ) 
  (side_length : ℝ) 
  (overlap : ℝ) 
  (h_num : num_papers = 6)
  (h_side : side_length = 30)
  (h_overlap : overlap = 7) : 
  (side_length + (num_papers - 1) * (side_length - overlap)) * side_length = 4350 :=
sorry

end connected_paper_area_l1705_170567


namespace beaver_carrot_count_l1705_170519

/-- Represents the number of carrots stored per burrow by the beaver -/
def beaver_carrots_per_burrow : ℕ := 4

/-- Represents the number of carrots stored per burrow by the rabbit -/
def rabbit_carrots_per_burrow : ℕ := 5

/-- Represents the difference in the number of burrows between the beaver and the rabbit -/
def burrow_difference : ℕ := 3

theorem beaver_carrot_count (beaver_burrows rabbit_burrows total_carrots : ℕ) :
  beaver_burrows = rabbit_burrows + burrow_difference →
  beaver_carrots_per_burrow * beaver_burrows = total_carrots →
  rabbit_carrots_per_burrow * rabbit_burrows = total_carrots →
  total_carrots = 60 := by
  sorry

end beaver_carrot_count_l1705_170519


namespace gcd_problems_l1705_170551

theorem gcd_problems :
  (Nat.gcd 840 1764 = 84) ∧ (Nat.gcd 459 357 = 51) := by
  sorry

end gcd_problems_l1705_170551


namespace simplify_trig_fraction_l1705_170597

theorem simplify_trig_fraction (x : ℝ) :
  (2 + 2 * Real.sin x - 2 * Real.cos x) / (2 + 2 * Real.sin x + 2 * Real.cos x) = Real.tan (x / 2) := by
  sorry

end simplify_trig_fraction_l1705_170597


namespace sum_equality_l1705_170581

theorem sum_equality (a b : ℝ) (h : a/b + a/b^2 + a/b^3 + a/b^4 + a/b^5 = 3) :
  (∑' n, (2*a) / (a+b)^n) = (6*(1 - 1/b^5)) / (4 - 1/b^5) :=
by sorry

end sum_equality_l1705_170581


namespace equation_solution_l1705_170572

theorem equation_solution (x : ℝ) (h : x > 4) :
  (Real.sqrt (x - 4 * Real.sqrt (x - 4)) + 2 = Real.sqrt (x + 4 * Real.sqrt (x - 4)) - 2) ↔ x ≥ 8 := by
sorry

end equation_solution_l1705_170572


namespace no_persistent_numbers_l1705_170509

/-- A number is persistent if, when multiplied by any positive integer, 
    the result always contains all ten digits 0,1,...,9. -/
def IsPersistent (n : ℕ) : Prop :=
  ∀ k : ℕ+, ∀ d : Fin 10, ∃ m : ℕ, (n * k : ℕ) / 10^m % 10 = d

/-- There are no persistent numbers. -/
theorem no_persistent_numbers : ¬∃ n : ℕ, IsPersistent n := by
  sorry


end no_persistent_numbers_l1705_170509


namespace sum_factorials_25_divisible_by_26_l1705_170595

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials (n : ℕ) : ℕ :=
  match n with
  | 0 => factorial 0
  | n + 1 => factorial (n + 1) + sum_factorials n

theorem sum_factorials_25_divisible_by_26 :
  ∃ k : ℕ, sum_factorials 25 = 26 * k :=
sorry

end sum_factorials_25_divisible_by_26_l1705_170595


namespace polynomial_identity_l1705_170587

theorem polynomial_identity (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end polynomial_identity_l1705_170587


namespace michael_quiz_score_l1705_170589

theorem michael_quiz_score (existing_scores : List ℕ) (target_mean : ℕ) (required_score : ℕ) : 
  existing_scores = [84, 78, 95, 88, 91] →
  target_mean = 90 →
  required_score = 104 →
  (existing_scores.sum + required_score) / (existing_scores.length + 1) = target_mean :=
by sorry

end michael_quiz_score_l1705_170589


namespace basketball_teams_l1705_170530

theorem basketball_teams (total : ℕ) (bad : ℕ) (rich : ℕ) (both : ℕ) : 
  total = 60 → 
  bad = (3 * total) / 5 →
  rich = (2 * total) / 3 →
  both ≤ bad :=
by sorry

end basketball_teams_l1705_170530


namespace lcm_problem_l1705_170542

theorem lcm_problem (n : ℕ+) (h1 : Nat.lcm 40 n = 200) (h2 : Nat.lcm n 45 = 180) : n = 180 := by
  sorry

end lcm_problem_l1705_170542


namespace division_problem_l1705_170565

theorem division_problem (A : ℕ) (h1 : 26 = A * 8 + 2) : A = 3 := by
  sorry

end division_problem_l1705_170565


namespace sqrt_equation_solution_l1705_170584

theorem sqrt_equation_solution (x : ℚ) :
  (Real.sqrt (6 * x) / Real.sqrt (5 * (x - 2)) = 3) → x = 30 / 13 := by
  sorry

end sqrt_equation_solution_l1705_170584


namespace events_mutually_exclusive_not_complementary_l1705_170504

-- Define the set of people
inductive Person : Type
  | A | B | C | D

-- Define the set of balls
inductive Ball : Type
  | one | two | three | four

-- Define a distribution as a function from Person to Ball
def Distribution := Person → Ball

-- Define the event "Person A gets ball number 1"
def event_A (d : Distribution) : Prop := d Person.A = Ball.one

-- Define the event "Person B gets ball number 1"
def event_B (d : Distribution) : Prop := d Person.B = Ball.one

-- Define mutually exclusive events
def mutually_exclusive (e1 e2 : Distribution → Prop) : Prop :=
  ∀ d : Distribution, ¬(e1 d ∧ e2 d)

-- Define complementary events
def complementary (e1 e2 : Distribution → Prop) : Prop :=
  ∀ d : Distribution, e1 d ↔ ¬(e2 d)

-- Theorem statement
theorem events_mutually_exclusive_not_complementary :
  mutually_exclusive event_A event_B ∧ ¬(complementary event_A event_B) := by
  sorry

end events_mutually_exclusive_not_complementary_l1705_170504


namespace promotion_difference_l1705_170591

/-- Represents a shoe promotion strategy -/
inductive Promotion
  | A  -- Buy one pair, get second pair half price
  | B  -- Buy one pair, get $15 off second pair

/-- Calculate the total cost of two pairs of shoes under a given promotion -/
def calculateCost (p : Promotion) (price : ℕ) : ℕ :=
  match p with
  | Promotion.A => price + price / 2
  | Promotion.B => price + price - 15

/-- The difference in cost between Promotion B and Promotion A is $5 -/
theorem promotion_difference (shoePrice : ℕ) (h : shoePrice = 40) :
  calculateCost Promotion.B shoePrice - calculateCost Promotion.A shoePrice = 5 := by
  sorry

#eval calculateCost Promotion.B 40 - calculateCost Promotion.A 40

end promotion_difference_l1705_170591


namespace nuts_in_masons_car_l1705_170579

/-- The number of busy squirrels -/
def busy_squirrels : ℕ := 2

/-- The number of nuts stored by each busy squirrel per day -/
def busy_nuts_per_day : ℕ := 30

/-- The number of days busy squirrels have been storing nuts -/
def busy_days : ℕ := 35

/-- The number of slightly lazy squirrels -/
def lazy_squirrels : ℕ := 3

/-- The number of nuts stored by each slightly lazy squirrel per day -/
def lazy_nuts_per_day : ℕ := 20

/-- The number of days slightly lazy squirrels have been storing nuts -/
def lazy_days : ℕ := 40

/-- The number of extremely sleepy squirrels -/
def sleepy_squirrels : ℕ := 1

/-- The number of nuts stored by the extremely sleepy squirrel per day -/
def sleepy_nuts_per_day : ℕ := 10

/-- The number of days the extremely sleepy squirrel has been storing nuts -/
def sleepy_days : ℕ := 45

/-- The total number of nuts in Mason's car -/
def total_nuts : ℕ := busy_squirrels * busy_nuts_per_day * busy_days +
                      lazy_squirrels * lazy_nuts_per_day * lazy_days +
                      sleepy_squirrels * sleepy_nuts_per_day * sleepy_days

theorem nuts_in_masons_car : total_nuts = 4950 := by
  sorry

end nuts_in_masons_car_l1705_170579


namespace d_is_zero_l1705_170580

def d (n m : ℕ) : ℚ :=
  if m = 0 ∨ m = n then 0
  else if 0 < m ∧ m < n then
    (m * d (n-1) m + (2*n - m) * d (n-1) (m-1)) / m
  else 0

theorem d_is_zero (n m : ℕ) (h : m ≤ n) : d n m = 0 := by
  sorry

end d_is_zero_l1705_170580


namespace total_slices_is_16_l1705_170562

/-- The number of pizzas Mrs. Hilt bought -/
def num_pizzas : ℕ := 2

/-- The number of slices per pizza -/
def slices_per_pizza : ℕ := 8

/-- The total number of pizza slices Mrs. Hilt had -/
def total_slices : ℕ := num_pizzas * slices_per_pizza

/-- Theorem stating that the total number of pizza slices is 16 -/
theorem total_slices_is_16 : total_slices = 16 := by
  sorry

end total_slices_is_16_l1705_170562


namespace project_hours_difference_l1705_170563

theorem project_hours_difference (total_hours kate_hours pat_hours mark_hours : ℕ) : 
  total_hours = 117 →
  pat_hours = 2 * kate_hours →
  pat_hours * 3 = mark_hours →
  total_hours = kate_hours + pat_hours + mark_hours →
  mark_hours - kate_hours = 65 := by
  sorry

end project_hours_difference_l1705_170563


namespace bo_words_per_day_l1705_170573

def words_per_day (total_flashcards : ℕ) (known_percentage : ℚ) (days_to_learn : ℕ) : ℚ :=
  (total_flashcards : ℚ) * (1 - known_percentage) / days_to_learn

theorem bo_words_per_day :
  words_per_day 800 (1/5) 40 = 16 := by sorry

end bo_words_per_day_l1705_170573


namespace symmetry_condition_l1705_170545

theorem symmetry_condition (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  (∀ x y : ℝ, y = (p * x + q) / (r * x + s) → -x = (p * (-y) + q) / (r * (-y) + s)) →
  p - s = 0 := by
  sorry

end symmetry_condition_l1705_170545
