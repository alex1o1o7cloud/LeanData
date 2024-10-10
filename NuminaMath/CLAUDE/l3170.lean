import Mathlib

namespace factorization_equality_l3170_317065

theorem factorization_equality (a b : ℝ) : a * b^2 - 4 * a = a * (b + 2) * (b - 2) := by
  sorry

end factorization_equality_l3170_317065


namespace circle_radius_from_longest_chord_l3170_317055

theorem circle_radius_from_longest_chord (c : ℝ) (h : c > 0) :
  (∃ (r : ℝ), r > 0 ∧ c = 24 ∧ c = 2 * r) → c / 2 = 12 := by
  sorry

end circle_radius_from_longest_chord_l3170_317055


namespace equation_solutions_l3170_317013

theorem equation_solutions :
  (∀ x : ℝ, 3 * (x - 1)^2 = 27 ↔ x = 4 ∨ x = -2) ∧
  (∀ x : ℝ, x^3 / 8 + 2 = 3 ↔ x = 2) := by
  sorry

end equation_solutions_l3170_317013


namespace day_in_consecutive_years_l3170_317019

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  number : ℕ
  is_leap : Bool

/-- Function to get the day of the week for a given day number in a year -/
def day_of_week (y : Year) (day_number : ℕ) : DayOfWeek :=
  sorry

/-- Function to check if a given day number is a Friday -/
def is_friday (y : Year) (day_number : ℕ) : Prop :=
  day_of_week y day_number = DayOfWeek.Friday

/-- Theorem stating the relationship between the days in consecutive years -/
theorem day_in_consecutive_years 
  (n : ℕ) 
  (year_n : Year)
  (year_n_plus_1 : Year)
  (year_n_minus_1 : Year)
  (h1 : year_n.number = n)
  (h2 : year_n_plus_1.number = n + 1)
  (h3 : year_n_minus_1.number = n - 1)
  (h4 : is_friday year_n 250)
  (h5 : is_friday year_n_plus_1 150) :
  day_of_week year_n_minus_1 50 = DayOfWeek.Monday :=
sorry

end day_in_consecutive_years_l3170_317019


namespace green_peaches_count_l3170_317022

/-- Given a basket of peaches, prove that the number of green peaches is 3 -/
theorem green_peaches_count (total : ℕ) (red : ℕ) (h1 : total = 16) (h2 : red = 13) :
  total - red = 3 := by
  sorry

end green_peaches_count_l3170_317022


namespace marbles_left_l3170_317038

theorem marbles_left (red : ℕ) (blue : ℕ) (broken : ℕ) 
  (h1 : red = 156) 
  (h2 : blue = 267) 
  (h3 : broken = 115) : 
  red + blue - broken = 308 := by
  sorry

end marbles_left_l3170_317038


namespace min_regions_for_12_intersections_l3170_317069

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A set of circles in a plane --/
def CircleSet := Set Circle

/-- The number of intersection points between circles in a set --/
def intersectionPoints (s : CircleSet) : ℕ := sorry

/-- The number of regions into which a set of circles divides the plane --/
def regions (s : CircleSet) : ℕ := sorry

/-- The theorem stating the minimum number of regions --/
theorem min_regions_for_12_intersections (s : CircleSet) :
  intersectionPoints s = 12 → regions s ≥ 14 :=
by sorry

end min_regions_for_12_intersections_l3170_317069


namespace decimal_51_to_binary_l3170_317010

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec to_binary_aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else to_binary_aux (m / 2) ((m % 2) :: acc)
    to_binary_aux n []

theorem decimal_51_to_binary :
  decimal_to_binary 51 = [1, 1, 0, 0, 1, 1] := by
  sorry

end decimal_51_to_binary_l3170_317010


namespace second_test_score_proof_l3170_317049

def first_test_score : ℝ := 78
def new_average : ℝ := 81

theorem second_test_score_proof :
  ∃ (second_score : ℝ), (first_test_score + second_score) / 2 = new_average ∧ second_score = 84 :=
by sorry

end second_test_score_proof_l3170_317049


namespace least_addition_for_divisibility_least_addition_to_1024_for_25_divisibility_l3170_317052

theorem least_addition_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n + x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % d ≠ 0 :=
by sorry

theorem least_addition_to_1024_for_25_divisibility :
  ∃ (x : ℕ), x < 25 ∧ (1024 + x) % 25 = 0 ∧ ∀ (y : ℕ), y < x → (1024 + y) % 25 ≠ 0 ∧ x = 1 :=
by sorry

end least_addition_for_divisibility_least_addition_to_1024_for_25_divisibility_l3170_317052


namespace system_solvability_l3170_317097

-- Define the system of equations
def system (x y p : ℝ) : Prop :=
  (x - p)^2 = 16 * (y - 3 + p) ∧
  y^2 + ((x - 3) / (|x| - 3))^2 = 1 ∧
  |x| ≠ 3

-- Define the set of valid p values
def valid_p_set : Set ℝ :=
  {p | (3 < p ∧ p ≤ 4) ∨ (12 ≤ p ∧ p < 19) ∨ (p > 19)}

-- Theorem statement
theorem system_solvability (p : ℝ) :
  (∃ x y, system x y p) ↔ p ∈ valid_p_set :=
sorry

end system_solvability_l3170_317097


namespace noah_age_in_ten_years_l3170_317098

/-- Calculates Noah's age after a given number of years -/
def noah_age_after (joe_age : ℕ) (years_passed : ℕ) : ℕ :=
  2 * joe_age + years_passed

/-- Proves that Noah will be 22 years old after 10 years, given the initial conditions -/
theorem noah_age_in_ten_years (joe_age : ℕ) (h : joe_age = 6) :
  noah_age_after joe_age 10 = 22 := by
  sorry

end noah_age_in_ten_years_l3170_317098


namespace bob_always_has_valid_move_l3170_317005

-- Define the game board
def GameBoard (n : ℕ) := ℤ × ℤ

-- Define the possible moves for Bob and Alice
def BobMove (p : ℤ × ℤ) : Set (ℤ × ℤ) :=
  {(p.1 + 2, p.2 + 1), (p.1 + 2, p.2 - 1), (p.1 - 2, p.2 + 1), (p.1 - 2, p.2 - 1)}

def AliceMove (p : ℤ × ℤ) : Set (ℤ × ℤ) :=
  {(p.1 + 1, p.2 + 2), (p.1 + 1, p.2 - 2), (p.1 - 1, p.2 + 2), (p.1 - 1, p.2 - 2)}

-- Define the modulo condition
def ModuloCondition (n : ℕ) (a b c d : ℤ) : Prop :=
  c % n = a % n ∧ d % n = b % n

-- Define a valid move
def ValidMove (n : ℕ) (occupied : Set (ℤ × ℤ)) (p : ℤ × ℤ) : Prop :=
  ∀ (a b : ℤ), (a, b) ∈ occupied → ¬(ModuloCondition n a b p.1 p.2)

-- Theorem: Bob always has a valid move
theorem bob_always_has_valid_move (n : ℕ) (h : n = 2018 ∨ n = 2019) 
  (occupied : Set (ℤ × ℤ)) (last_move : ℤ × ℤ) :
  ∃ (next_move : ℤ × ℤ), next_move ∈ BobMove last_move ∧ ValidMove n occupied next_move :=
sorry

end bob_always_has_valid_move_l3170_317005


namespace fraction_equality_l3170_317046

theorem fraction_equality : 
  (14 : ℚ) / 12 = 7 / 6 ∧
  (1 : ℚ) + 1 / 6 = 7 / 6 ∧
  (21 : ℚ) / 18 = 7 / 6 ∧
  (1 : ℚ) + 2 / 12 = 7 / 6 ∧
  (1 : ℚ) + 1 / 3 ≠ 7 / 6 :=
by sorry

end fraction_equality_l3170_317046


namespace complex_expressions_equality_l3170_317009

theorem complex_expressions_equality : 
  let z₁ : ℂ := (-2 * Real.sqrt 3 * I + 1) / (1 + 2 * Real.sqrt 3 * I) + ((Real.sqrt 2) / (1 + I)) ^ 2000 + (1 + I) / (3 - I)
  let z₂ : ℂ := (5 * (4 + I)^2) / (I * (2 + I)) + 2 / (1 - I)^2
  z₁ = 6/65 + (39/65) * I ∧ z₂ = -1 + 39 * I :=
by
  sorry

end complex_expressions_equality_l3170_317009


namespace sam_remaining_money_l3170_317021

/-- Given an initial amount, number of books, and cost per book, 
    calculate the remaining amount after purchase. -/
def remaining_amount (initial : ℕ) (num_books : ℕ) (cost_per_book : ℕ) : ℕ :=
  initial - (num_books * cost_per_book)

/-- Theorem stating that given the specific conditions of Sam's purchase,
    the remaining amount is 16 dollars. -/
theorem sam_remaining_money :
  remaining_amount 79 9 7 = 16 := by
  sorry

end sam_remaining_money_l3170_317021


namespace distance_focus_to_asymptote_l3170_317039

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 3 = 1

-- Define the focus of the hyperbola
def focus (F : ℝ × ℝ) : Prop := 
  F.1^2 - F.2^2 = 6 ∧ F.2 = 0

-- Define an asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := y = x ∨ y = -x

-- Theorem statement
theorem distance_focus_to_asymptote :
  ∀ (F : ℝ × ℝ) (x y : ℝ),
  focus F → hyperbola x y → asymptote x y →
  ∃ (d : ℝ), d = Real.sqrt 3 ∧ 
  d = Real.sqrt ((F.1 - x)^2 + (F.2 - y)^2) := by sorry

end distance_focus_to_asymptote_l3170_317039


namespace smallest_three_digit_congruence_l3170_317057

theorem smallest_three_digit_congruence :
  ∃ (n : ℕ), 
    (n ≥ 100 ∧ n < 1000) ∧ 
    (70 * n) % 350 = 210 ∧
    (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ (70 * m) % 350 = 210 → n ≤ m) ∧
    n = 103 := by
  sorry

end smallest_three_digit_congruence_l3170_317057


namespace large_bus_most_cost_effective_l3170_317089

/-- Represents the transportation options for the field trip --/
inductive TransportOption
  | Van
  | Minibus
  | LargeBus

/-- Calculates the number of vehicles needed for a given option --/
def vehiclesNeeded (option : TransportOption) : ℕ :=
  match option with
  | .Van => 6
  | .Minibus => 3
  | .LargeBus => 1

/-- Calculates the total cost for a given option --/
def totalCost (option : TransportOption) : ℕ :=
  match option with
  | .Van => 50 * vehiclesNeeded .Van
  | .Minibus => 100 * vehiclesNeeded .Minibus
  | .LargeBus => 250

/-- States that the large bus is the most cost-effective option --/
theorem large_bus_most_cost_effective :
  ∀ option : TransportOption, totalCost .LargeBus ≤ totalCost option :=
by sorry

end large_bus_most_cost_effective_l3170_317089


namespace triangle_properties_l3170_317032

open Real

theorem triangle_properties 
  (a b c A B C : ℝ) 
  (h1 : sin A / (sin B + sin C) = 1 - (a - b) / (a - c))
  (h2 : b = Real.sqrt 3)
  (h3 : 0 < A ∧ A < 2 * π / 3) :
  ∃ (area : ℝ) (range_lower range_upper : ℝ),
    (∀ perimeter, perimeter ≤ 3 * Real.sqrt 3 → 
      area * 2 ≤ perimeter * Real.sqrt (perimeter * (perimeter - 2*a) * (perimeter - 2*b) * (perimeter - 2*c)) / 4) ∧
    (area = 3 * Real.sqrt 3 / 4) ∧
    (∀ m_dot_n : ℝ, 
      (∃ A', 0 < A' ∧ A' < 2 * π / 3 ∧ 
        m_dot_n = 6 * sin A' * cos B + cos (2 * A')) → 
      (range_lower < m_dot_n ∧ m_dot_n ≤ range_upper)) ∧
    (range_lower = 1 ∧ range_upper = 17/8) :=
by sorry

end triangle_properties_l3170_317032


namespace first_day_next_year_monday_l3170_317093

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  value : Nat
  is_leap : Bool

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Theorem: If a non-leap year has 53 Sundays, then the first day of the following year is a Monday -/
theorem first_day_next_year_monday 
  (year : Year) 
  (h1 : year.is_leap = false) 
  (h2 : ∃ (sundays : Nat), sundays = 53) : 
  nextDay DayOfWeek.Sunday = DayOfWeek.Monday := by
  sorry

#check first_day_next_year_monday

end first_day_next_year_monday_l3170_317093


namespace sum_of_four_integers_with_product_5_4_l3170_317004

theorem sum_of_four_integers_with_product_5_4 (a b c d : ℕ+) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 5^4 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) + (d : ℕ) = 156 := by
  sorry

end sum_of_four_integers_with_product_5_4_l3170_317004


namespace jenny_lasagna_sales_l3170_317033

/-- The number of pans of lasagna Jenny makes and sells -/
def num_pans : ℕ := 20

/-- The cost to make each pan of lasagna -/
def cost_per_pan : ℚ := 10

/-- The selling price of each pan of lasagna -/
def price_per_pan : ℚ := 25

/-- The profit after expenses -/
def profit : ℚ := 300

/-- Theorem stating that the number of pans sold is correct given the conditions -/
theorem jenny_lasagna_sales : 
  (price_per_pan - cost_per_pan) * num_pans = profit := by sorry

end jenny_lasagna_sales_l3170_317033


namespace abs_is_even_and_increasing_l3170_317079

def f (x : ℝ) := abs x

theorem abs_is_even_and_increasing :
  (∀ x, f (-x) = f x) ∧
  (∀ x y, 0 < x → x < y → f x < f y) :=
sorry

end abs_is_even_and_increasing_l3170_317079


namespace total_spent_is_135_l3170_317023

/-- The amount Jen spent on pastries -/
def jen_spent : ℝ := sorry

/-- The amount Lisa spent on pastries -/
def lisa_spent : ℝ := sorry

/-- For every dollar Jen spent, Lisa spent 20 cents less -/
axiom lisa_spent_relation : lisa_spent = 0.8 * jen_spent

/-- Jen spent $15 more than Lisa -/
axiom jen_spent_more : jen_spent = lisa_spent + 15

/-- The total amount spent on pastries -/
def total_spent : ℝ := jen_spent + lisa_spent

/-- Theorem stating that the total amount spent is $135 -/
theorem total_spent_is_135 : total_spent = 135 := by sorry

end total_spent_is_135_l3170_317023


namespace total_daily_allowance_l3170_317040

theorem total_daily_allowance (total_students : ℕ) 
  (high_allowance : ℚ) (low_allowance : ℚ) :
  total_students = 60 →
  high_allowance = 6 →
  low_allowance = 4 →
  (2 : ℚ) / 3 * total_students * high_allowance + 
  (1 : ℚ) / 3 * total_students * low_allowance = 320 := by
sorry

end total_daily_allowance_l3170_317040


namespace non_congruent_triangles_count_l3170_317066

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a set of points -/
def PointSet : Type := List Point

/-- The set of nine points as described in the problem -/
def ninePoints : PointSet :=
  [
    {x := 0, y := 0}, {x := 1, y := 0}, {x := 2, y := 0},
    {x := 0, y := 0}, {x := 1, y := 1}, {x := 2, y := 2},
    {x := 0, y := 0}, {x := 0.5, y := 1}, {x := 1, y := 2}
  ]

/-- Checks if three points form a non-congruent triangle with respect to a set of triangles -/
def isNonCongruentTriangle (p1 p2 p3 : Point) (triangles : List (Point × Point × Point)) : Bool :=
  sorry

/-- Counts the number of non-congruent triangles that can be formed from a set of points -/
def countNonCongruentTriangles (points : PointSet) : Nat :=
  sorry

/-- The main theorem stating that the number of non-congruent triangles is 5 -/
theorem non_congruent_triangles_count :
  countNonCongruentTriangles ninePoints = 5 :=
sorry

end non_congruent_triangles_count_l3170_317066


namespace male_salmon_count_l3170_317006

theorem male_salmon_count (total : ℕ) (female : ℕ) (h1 : total = 971639) (h2 : female = 259378) :
  total - female = 712261 := by
  sorry

end male_salmon_count_l3170_317006


namespace simplify_trig_expression_l3170_317036

theorem simplify_trig_expression :
  1 / Real.sin (15 * π / 180) - 1 / Real.cos (15 * π / 180) = 2 * Real.sqrt 2 := by
  sorry

end simplify_trig_expression_l3170_317036


namespace correct_division_l3170_317061

theorem correct_division (dividend : ℕ) (wrong_divisor correct_divisor wrong_quotient : ℕ) 
  (h1 : wrong_divisor = 87)
  (h2 : correct_divisor = 36)
  (h3 : wrong_quotient = 24)
  (h4 : dividend = wrong_divisor * wrong_quotient) :
  dividend / correct_divisor = 58 := by
sorry

end correct_division_l3170_317061


namespace jeff_probability_multiple_of_four_l3170_317018

/-- The number of cards --/
def num_cards : ℕ := 12

/-- The probability of moving left on a single spin --/
def prob_left : ℚ := 1/2

/-- The probability of moving right on a single spin --/
def prob_right : ℚ := 1/2

/-- The number of spaces moved left --/
def spaces_left : ℕ := 1

/-- The number of spaces moved right --/
def spaces_right : ℕ := 2

/-- The probability of ending up at a multiple of 4 --/
def prob_multiple_of_four : ℚ := 5/32

theorem jeff_probability_multiple_of_four :
  let start_at_multiple_of_four := (num_cards / 4 : ℚ) / num_cards
  let start_two_more_than_multiple_of_four := (num_cards / 4 : ℚ) / num_cards
  let start_two_less_than_multiple_of_four := (num_cards / 4 : ℚ) / num_cards
  let end_at_multiple_of_four_from_multiple_of_four := prob_left * prob_right + prob_right * prob_left
  let end_at_multiple_of_four_from_two_more := prob_right * prob_right
  let end_at_multiple_of_four_from_two_less := prob_left * prob_left
  start_at_multiple_of_four * end_at_multiple_of_four_from_multiple_of_four +
  start_two_more_than_multiple_of_four * end_at_multiple_of_four_from_two_more +
  start_two_less_than_multiple_of_four * end_at_multiple_of_four_from_two_less =
  prob_multiple_of_four := by
  sorry

end jeff_probability_multiple_of_four_l3170_317018


namespace temperature_conversion_l3170_317011

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → t = 50 → k = 122 := by
  sorry

end temperature_conversion_l3170_317011


namespace prime_power_sum_l3170_317008

theorem prime_power_sum (w x y z : ℕ) :
  2^w * 3^x * 5^y * 7^z = 588 → 2*w + 3*x + 5*y + 7*z = 21 := by
  sorry

end prime_power_sum_l3170_317008


namespace perfect_square_count_l3170_317024

theorem perfect_square_count : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, n > 0 ∧ n ≤ 2000 ∧ ∃ k : ℕ, 10 * n = k^2) ∧ 
  (∀ n : ℕ, n > 0 ∧ n ≤ 2000 ∧ (∃ k : ℕ, 10 * n = k^2) → n ∈ S) ∧
  Finset.card S = 14 :=
sorry

end perfect_square_count_l3170_317024


namespace selection_schemes_count_l3170_317092

def num_students : ℕ := 6
def num_tasks : ℕ := 4
def num_restricted_students : ℕ := 2

theorem selection_schemes_count :
  (num_students.factorial / (num_students - num_tasks).factorial) -
  (num_restricted_students * (num_students - 1).factorial / (num_students - num_tasks).factorial) = 240 :=
by sorry

end selection_schemes_count_l3170_317092


namespace hexagon_diagonals_l3170_317085

/-- The number of diagonals in a polygon with N sides -/
def num_diagonals (N : ℕ) : ℕ := N * (N - 3) / 2

/-- A regular hexagon has 9 diagonals -/
theorem hexagon_diagonals :
  num_diagonals 6 = 9 := by
  sorry

end hexagon_diagonals_l3170_317085


namespace arithmetic_sequence_12th_term_l3170_317041

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_3rd : a 3 = 10)
  (h_6th : a 6 = 20) :
  a 12 = 40 := by
sorry

end arithmetic_sequence_12th_term_l3170_317041


namespace remainder_of_binary_div_8_l3170_317068

def binary_number : ℕ := 110110111010

-- Define a function to get the last three bits of a binary number
def last_three_bits (n : ℕ) : ℕ := n % 8

-- Theorem statement
theorem remainder_of_binary_div_8 :
  binary_number % 8 = 2 := by sorry

end remainder_of_binary_div_8_l3170_317068


namespace total_pizza_slices_l3170_317000

theorem total_pizza_slices (num_pizzas : ℕ) (slices_per_pizza : ℕ) 
  (h1 : num_pizzas = 36) 
  (h2 : slices_per_pizza = 12) : 
  num_pizzas * slices_per_pizza = 432 :=
by sorry

end total_pizza_slices_l3170_317000


namespace parallelepiped_base_sides_l3170_317003

/-- Given a rectangular parallelepiped with a cross-section having diagonals of 20 and 8 units
    intersecting at a 60° angle, the lengths of the sides of its base are 2√5 and √30. -/
theorem parallelepiped_base_sides (d₁ d₂ : ℝ) (θ : ℝ) 
  (h₁ : d₁ = 20) (h₂ : d₂ = 8) (h₃ : θ = Real.pi / 3) :
  ∃ (a b : ℝ), a = 2 * Real.sqrt 5 ∧ b = Real.sqrt 30 ∧ 
  (a * a + b * b = d₁ * d₁) ∧ 
  (d₂ * d₂ = 2 * a * b * Real.cos θ) := by
sorry


end parallelepiped_base_sides_l3170_317003


namespace max_sum_cubes_constraint_max_sum_cubes_constraint_achievable_l3170_317037

theorem max_sum_cubes_constraint (p q r s t : ℝ) 
  (h : p^2 + q^2 + r^2 + s^2 + t^2 = 5) : 
  p^3 + q^3 + r^3 + s^3 + t^3 ≤ 5 * Real.sqrt 5 := by
  sorry

theorem max_sum_cubes_constraint_achievable : 
  ∃ (p q r s t : ℝ), p^2 + q^2 + r^2 + s^2 + t^2 = 5 ∧ 
  p^3 + q^3 + r^3 + s^3 + t^3 = 5 * Real.sqrt 5 := by
  sorry

end max_sum_cubes_constraint_max_sum_cubes_constraint_achievable_l3170_317037


namespace k_value_l3170_317067

def length (k : ℕ) : ℕ := sorry

theorem k_value (k : ℕ) (h1 : k > 1) (h2 : length k = 4) (h3 : k = 2 * 2 * 2 * 3) : k = 24 := by
  sorry

end k_value_l3170_317067


namespace members_playing_both_l3170_317058

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  totalMembers : ℕ
  badmintonPlayers : ℕ
  tennisPlayers : ℕ
  neitherPlayers : ℕ

/-- Calculates the number of members playing both badminton and tennis -/
def playBoth (club : SportsClub) : ℕ :=
  club.badmintonPlayers + club.tennisPlayers - (club.totalMembers - club.neitherPlayers)

/-- Theorem stating the number of members playing both sports in the given scenario -/
theorem members_playing_both (club : SportsClub)
    (h1 : club.totalMembers = 50)
    (h2 : club.badmintonPlayers = 25)
    (h3 : club.tennisPlayers = 32)
    (h4 : club.neitherPlayers = 5) :
    playBoth club = 12 := by
  sorry


end members_playing_both_l3170_317058


namespace jasons_cousins_l3170_317012

theorem jasons_cousins (cupcakes_bought : ℕ) (cupcakes_per_cousin : ℕ) : 
  cupcakes_bought = 4 * 12 → cupcakes_per_cousin = 3 → 
  cupcakes_bought / cupcakes_per_cousin = 16 := by
  sorry

end jasons_cousins_l3170_317012


namespace monotonicity_f_range_of_a_l3170_317017

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -a * x^2 + Real.log x

-- State the theorems
theorem monotonicity_f (a : ℝ) :
  (a ≤ 0 → ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → f a x₁ < f a x₂) ∧
  (a > 0 → ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → x₂ < 1 / Real.sqrt (2 * a) → f a x₁ < f a x₂) ∧
  (a > 0 → ∀ x₁ x₂, 1 / Real.sqrt (2 * a) < x₁ → x₁ < x₂ → f a x₁ > f a x₂) :=
sorry

theorem range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, x > 1 ∧ f a x > -a) ↔ a < 1/2 :=
sorry

end monotonicity_f_range_of_a_l3170_317017


namespace average_daily_sales_l3170_317043

/-- Theorem: Average daily sales of cups over a 12-day period -/
theorem average_daily_sales (day_one_sales : ℕ) (other_days_sales : ℕ) (total_days : ℕ) :
  day_one_sales = 86 →
  other_days_sales = 50 →
  total_days = 12 →
  (day_one_sales + (total_days - 1) * other_days_sales) / total_days = 53 :=
by sorry

end average_daily_sales_l3170_317043


namespace circular_competition_rounds_l3170_317078

theorem circular_competition_rounds (m : ℕ) (h : m ≥ 17) :
  ∃ (n : ℕ), n = m - 1 ∧
  (∀ (schedule : ℕ → Fin (2*m) → Fin (2*m) → Prop),
    (∀ (i : Fin (2*m)), ∀ (j : Fin (2*m)), i ≠ j → ∃ (k : Fin (2*m - 1)), schedule k i j) →
    (∀ (k : Fin (2*m - 1)), ∀ (i : Fin (2*m)), ∃! (j : Fin (2*m)), i ≠ j ∧ schedule k i j) →
    (∀ (a b c d : Fin (2*m)), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d →
      (∀ (k : Fin n), ¬(schedule k a b ∨ schedule k a c ∨ schedule k a d ∨ schedule k b c ∨ schedule k b d ∨ schedule k c d)) ∨
      (∃ (k₁ k₂ : Fin n), k₁ ≠ k₂ ∧
        ((schedule k₁ a b ∧ schedule k₂ c d) ∨
         (schedule k₁ a c ∧ schedule k₂ b d) ∨
         (schedule k₁ a d ∧ schedule k₂ b c))))) ∧
  (∀ (n' : ℕ), n' < n →
    ∃ (schedule : ℕ → Fin (2*m) → Fin (2*m) → Prop),
      (∀ (i : Fin (2*m)), ∀ (j : Fin (2*m)), i ≠ j → ∃ (k : Fin (2*m - 1)), schedule k i j) ∧
      (∀ (k : Fin (2*m - 1)), ∀ (i : Fin (2*m)), ∃! (j : Fin (2*m)), i ≠ j ∧ schedule k i j) ∧
      (∃ (a b c d : Fin (2*m)), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d ∧
        (∀ (k : Fin n'), ¬(schedule k a b ∨ schedule k a c ∨ schedule k a d ∨ schedule k b c ∨ schedule k b d ∨ schedule k c d)) ∧
        ¬(∃ (k₁ k₂ : Fin n'), k₁ ≠ k₂ ∧
          ((schedule k₁ a b ∧ schedule k₂ c d) ∨
           (schedule k₁ a c ∧ schedule k₂ b d) ∨
           (schedule k₁ a d ∧ schedule k₂ b c))))) :=
by
  sorry


end circular_competition_rounds_l3170_317078


namespace smallest_overlap_percentage_l3170_317002

theorem smallest_overlap_percentage (coffee_drinkers tea_drinkers : ℝ) 
  (h1 : coffee_drinkers = 60)
  (h2 : tea_drinkers = 90) :
  coffee_drinkers + tea_drinkers - 100 = 50 := by
  sorry

end smallest_overlap_percentage_l3170_317002


namespace least_positive_four_digit_solution_l3170_317086

theorem least_positive_four_digit_solution (x : ℕ) : 
  (10 * x ≡ 30 [ZMOD 20]) ∧ 
  (2 * x + 10 ≡ 19 [ZMOD 9]) ∧ 
  (-3 * x + 1 ≡ x [ZMOD 19]) ∧ 
  (x ≥ 1000) ∧ (x < 10000) →
  x ≥ 1296 := by
  sorry

end least_positive_four_digit_solution_l3170_317086


namespace math_team_selection_ways_l3170_317096

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem math_team_selection_ways :
  let boys := 7
  let girls := 9
  let team_boys := 3
  let team_girls := 3
  (choose boys team_boys) * (choose girls team_girls) = 2940 := by
sorry

end math_team_selection_ways_l3170_317096


namespace no_integers_product_zeros_l3170_317091

theorem no_integers_product_zeros : 
  ¬∃ (x y : ℤ), 
    (x % 10 ≠ 0) ∧ 
    (y % 10 ≠ 0) ∧ 
    (x * y = 100000) := by
  sorry

end no_integers_product_zeros_l3170_317091


namespace negation_is_false_l3170_317076

theorem negation_is_false : 
  ¬(∀ x y : ℝ, (x > 2 ∧ y > 3) → x + y > 5) = False :=
sorry

end negation_is_false_l3170_317076


namespace max_ac_without_racing_stripes_l3170_317073

/-- Represents the properties of a car group -/
structure CarGroup where
  total : ℕ
  without_ac : ℕ
  with_racing_stripes : ℕ
  (total_valid : total = 100)
  (without_ac_valid : without_ac = 37)
  (racing_stripes_valid : with_racing_stripes ≥ 41)

/-- Theorem: The greatest number of cars that could have air conditioning but not racing stripes -/
theorem max_ac_without_racing_stripes (group : CarGroup) : 
  (group.total - group.without_ac) - group.with_racing_stripes ≤ 22 :=
sorry

end max_ac_without_racing_stripes_l3170_317073


namespace crayon_difference_l3170_317080

theorem crayon_difference (red : ℕ) (yellow : ℕ) (blue : ℕ) : 
  red = 14 → 
  yellow = 32 → 
  yellow = 2 * blue - 6 → 
  blue - red = 5 := by
sorry

end crayon_difference_l3170_317080


namespace stating_min_gloves_for_matching_pair_l3170_317015

/-- Represents the number of different glove patterns -/
def num_patterns : ℕ := 4

/-- Represents the number of pairs for each pattern -/
def pairs_per_pattern : ℕ := 3

/-- Represents the total number of gloves in the wardrobe -/
def total_gloves : ℕ := num_patterns * pairs_per_pattern * 2

/-- 
Theorem stating the minimum number of gloves needed to ensure a matching pair
-/
theorem min_gloves_for_matching_pair : 
  ∃ (n : ℕ), n = num_patterns * pairs_per_pattern + 1 ∧ 
  (∀ (m : ℕ), m < n → ∃ (pattern : Fin num_patterns), 
    (m.choose 2 : ℕ) < pairs_per_pattern) ∧
  n ≤ total_gloves := by
  sorry

end stating_min_gloves_for_matching_pair_l3170_317015


namespace octagon_area_in_circle_l3170_317051

theorem octagon_area_in_circle (r : ℝ) (h : r = 2.5) : 
  let octagon_area := 8 * (r^2 * Real.sin (π/8) * Real.cos (π/8))
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ |octagon_area - 17.672| < ε :=
by
  sorry

end octagon_area_in_circle_l3170_317051


namespace sqrt_inequality_l3170_317007

theorem sqrt_inequality (a : ℝ) (h : a ≥ 3) :
  Real.sqrt (a - 2) - Real.sqrt (a - 3) > Real.sqrt a - Real.sqrt (a - 1) := by
  sorry

end sqrt_inequality_l3170_317007


namespace parabola_axis_of_symmetry_l3170_317062

-- Define the parabola equation
def parabola_eq (x y : ℝ) : Prop :=
  x^2 + 2*x*y + y^2 + 3*x + y = 0

-- Define the axis of symmetry
def axis_of_symmetry (x y : ℝ) : Prop :=
  x + y + 1 = 0

-- Theorem statement
theorem parabola_axis_of_symmetry :
  ∀ (x y : ℝ), parabola_eq x y → axis_of_symmetry x y :=
by sorry

end parabola_axis_of_symmetry_l3170_317062


namespace base_irrelevant_l3170_317084

theorem base_irrelevant (b : ℝ) : 
  ∃ (x y : ℝ), 3^x * b^y = 19683 ∧ x - y = 9 ∧ x = 9 → 3^9 * b^0 = 19683 := by
  sorry

end base_irrelevant_l3170_317084


namespace monotone_increasing_condition_l3170_317082

/-- A function f(x) = kx - ln x is monotonically increasing on (1/2, +∞) if and only if k ≥ 2 -/
theorem monotone_increasing_condition (k : ℝ) :
  (∀ x > (1/2 : ℝ), Monotone (λ x => k * x - Real.log x)) ↔ k ≥ 2 :=
sorry

end monotone_increasing_condition_l3170_317082


namespace sufficient_not_necessary_l3170_317031

theorem sufficient_not_necessary (p q : Prop) : 
  (¬(p ∨ q) → ¬p) ∧ ¬(∀ p q, ¬p → ¬(p ∨ q)) := by
  sorry

end sufficient_not_necessary_l3170_317031


namespace pradeep_failed_marks_l3170_317050

def total_marks : ℕ := 550
def passing_percentage : ℚ := 40 / 100
def pradeep_marks : ℕ := 200

theorem pradeep_failed_marks : 
  (total_marks * passing_percentage).floor - pradeep_marks = 20 := by
  sorry

end pradeep_failed_marks_l3170_317050


namespace pyramid_section_is_trapezoid_l3170_317075

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D
  d : ℝ

/-- Represents a pyramid -/
structure Pyramid where
  apex : Point3D
  base : List Point3D

/-- Represents a parallelogram -/
structure Parallelogram where
  vertices : List Point3D

/-- Represents a trapezoid -/
structure Trapezoid where
  vertices : List Point3D

def is_parallelogram (p : Parallelogram) : Prop := sorry

def is_point_on_edge (p : Point3D) (e1 e2 : Point3D) : Prop := sorry

def intersection_is_trapezoid (plane : Plane) (pyr : Pyramid) : Prop := sorry

theorem pyramid_section_is_trapezoid 
  (S A B C D M : Point3D) 
  (base : Parallelogram) 
  (pyr : Pyramid) 
  (plane : Plane) :
  is_parallelogram base →
  pyr.apex = S →
  pyr.base = base.vertices →
  is_point_on_edge M S C →
  plane.normal = sorry → -- Define the normal vector of plane ABM
  plane.d = sorry → -- Define the d value for plane ABM
  intersection_is_trapezoid plane pyr := by
  sorry

#check pyramid_section_is_trapezoid

end pyramid_section_is_trapezoid_l3170_317075


namespace one_can_per_person_day1_l3170_317071

/-- Represents the food bank scenario --/
structure FoodBank where
  initialStock : ℕ
  day1People : ℕ
  day1Restock : ℕ
  day2People : ℕ
  day2CansPerPerson : ℕ
  day2Restock : ℕ
  totalGivenAway : ℕ

/-- Calculates the number of cans each person took on the first day --/
def cansPerPersonDay1 (fb : FoodBank) : ℕ :=
  (fb.totalGivenAway - fb.day2People * fb.day2CansPerPerson) / fb.day1People

/-- Theorem stating that each person took 1 can on the first day --/
theorem one_can_per_person_day1 (fb : FoodBank)
    (h1 : fb.initialStock = 2000)
    (h2 : fb.day1People = 500)
    (h3 : fb.day1Restock = 1500)
    (h4 : fb.day2People = 1000)
    (h5 : fb.day2CansPerPerson = 2)
    (h6 : fb.day2Restock = 3000)
    (h7 : fb.totalGivenAway = 2500) :
    cansPerPersonDay1 fb = 1 := by
  sorry

#eval cansPerPersonDay1 {
  initialStock := 2000,
  day1People := 500,
  day1Restock := 1500,
  day2People := 1000,
  day2CansPerPerson := 2,
  day2Restock := 3000,
  totalGivenAway := 2500
}

end one_can_per_person_day1_l3170_317071


namespace cats_after_sale_l3170_317025

/-- The number of cats left after a sale in a pet store -/
theorem cats_after_sale (siamese_cats house_cats sold_cats : ℕ) :
  siamese_cats = 12 →
  house_cats = 20 →
  sold_cats = 20 →
  siamese_cats + house_cats - sold_cats = 12 := by
  sorry

end cats_after_sale_l3170_317025


namespace apple_ratio_l3170_317045

theorem apple_ratio (blue_apples : ℕ) (yellow_apples : ℕ) : 
  blue_apples = 5 →
  yellow_apples + blue_apples - (yellow_apples + blue_apples) / 5 = 12 →
  yellow_apples / blue_apples = 2 :=
by sorry

end apple_ratio_l3170_317045


namespace total_books_on_shelves_l3170_317095

theorem total_books_on_shelves (num_shelves : ℕ) (books_per_shelf : ℚ) : 
  num_shelves = 520 → books_per_shelf = 37.5 → num_shelves * books_per_shelf = 19500 := by
  sorry

end total_books_on_shelves_l3170_317095


namespace two_integers_sum_l3170_317027

theorem two_integers_sum (a b : ℕ) : 
  a > 0 → b > 0 → 
  a * b + a + b = 255 → 
  (Odd a ∨ Odd b) → 
  a < 30 → b < 30 → 
  a + b = 30 := by sorry

end two_integers_sum_l3170_317027


namespace factorization_ax2_minus_4ay2_l3170_317077

theorem factorization_ax2_minus_4ay2 (a x y : ℝ) :
  a * x^2 - 4 * a * y^2 = a * (x + 2 * y) * (x - 2 * y) := by
  sorry

end factorization_ax2_minus_4ay2_l3170_317077


namespace expression_simplification_l3170_317020

theorem expression_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let x := ((a * (b^(1/3))) / (b * (a^3)^(1/2)))^(3/2) + ((a^(1/2)) / (a * (b^3)^(1/8)))^2
  x / (a^(1/4) + b^(1/4)) = 1 / (a * b) := by
  sorry

end expression_simplification_l3170_317020


namespace probability_at_least_one_female_l3170_317070

def total_students : ℕ := 5
def male_students : ℕ := 3
def female_students : ℕ := 2
def students_to_select : ℕ := 2

theorem probability_at_least_one_female :
  (1 : ℚ) - (Nat.choose male_students students_to_select : ℚ) / (Nat.choose total_students students_to_select : ℚ) = 7 / 10 := by
  sorry

end probability_at_least_one_female_l3170_317070


namespace arithmetic_square_root_of_16_l3170_317059

theorem arithmetic_square_root_of_16 : ∃ (x : ℝ), x ≥ 0 ∧ x^2 = 16 ∧ x = 4 := by
  sorry

end arithmetic_square_root_of_16_l3170_317059


namespace symmetry_probability_one_third_l3170_317030

/-- A square grid with n^2 points -/
structure SquareGrid (n : ℕ) where
  points : Fin n → Fin n → Bool

/-- The center point of a square grid -/
def centerPoint (n : ℕ) : Fin n × Fin n :=
  (⟨n / 2, sorry⟩, ⟨n / 2, sorry⟩)

/-- A line of symmetry for a square grid -/
def isSymmetryLine (n : ℕ) (grid : SquareGrid n) (p q : Fin n × Fin n) : Prop :=
  sorry

/-- The number of symmetry lines through the center point -/
def numSymmetryLines (n : ℕ) (grid : SquareGrid n) : ℕ :=
  sorry

theorem symmetry_probability_one_third (grid : SquareGrid 11) :
  let center := centerPoint 11
  let totalPoints := 121
  let nonCenterPoints := totalPoints - 1
  let symmetryLines := numSymmetryLines 11 grid
  (symmetryLines : ℚ) / nonCenterPoints = 1 / 3 :=
sorry

end symmetry_probability_one_third_l3170_317030


namespace isosceles_triangle_condition_l3170_317001

theorem isosceles_triangle_condition (a b c A B C : ℝ) :
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (a > 0) → (b > 0) → (c > 0) →
  (a * Real.cos C + c * Real.cos B = b) →
  (a = b ∧ A = B) :=
sorry

end isosceles_triangle_condition_l3170_317001


namespace price_increase_percentage_l3170_317090

theorem price_increase_percentage (old_price new_price : ℝ) 
  (h1 : old_price = 300)
  (h2 : new_price = 360) :
  (new_price - old_price) / old_price * 100 = 20 := by
  sorry

end price_increase_percentage_l3170_317090


namespace prob_of_three_l3170_317083

/-- The decimal representation of 8/13 -/
def decimal_rep : ℚ := 8 / 13

/-- The length of the repeating block in the decimal representation -/
def block_length : ℕ := 6

/-- The count of digit 3 in one repeating block -/
def count_of_threes : ℕ := 1

/-- The probability of randomly selecting the digit 3 from the decimal representation of 8/13 -/
theorem prob_of_three (decimal_rep : ℚ) (block_length : ℕ) (count_of_threes : ℕ) :
  decimal_rep = 8 / 13 →
  block_length = 6 →
  count_of_threes = 1 →
  (count_of_threes : ℚ) / (block_length : ℚ) = 1 / 6 := by
  sorry

end prob_of_three_l3170_317083


namespace solution_set_implies_ab_value_l3170_317081

theorem solution_set_implies_ab_value (a b : ℝ) : 
  (∀ x, x^2 + 2*a*x - 4*b ≤ 0 ↔ -2 ≤ x ∧ x ≤ 6) → 
  a^b = -8 := by
sorry

end solution_set_implies_ab_value_l3170_317081


namespace allison_craft_items_l3170_317014

/-- Represents the number of craft items bought by a person -/
structure CraftItems where
  glueSticks : ℕ
  constructionPaper : ℕ

/-- Calculates the total number of craft items -/
def totalItems (items : CraftItems) : ℕ :=
  items.glueSticks + items.constructionPaper

theorem allison_craft_items (marie : CraftItems) 
    (marie_glue : marie.glueSticks = 15)
    (marie_paper : marie.constructionPaper = 30)
    (allison : CraftItems)
    (glue_diff : allison.glueSticks = marie.glueSticks + 8)
    (paper_ratio : marie.constructionPaper = 6 * allison.constructionPaper) :
    totalItems allison = 28 := by
  sorry

end allison_craft_items_l3170_317014


namespace line_passes_through_fixed_point_l3170_317026

/-- The parabola y² = 4x in the cartesian plane -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- A line in the cartesian plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The intersection points of a line with the parabola -/
def intersection (l : Line) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p ∈ Parabola ∧ p.1 = l.slope * p.2 + l.intercept}

/-- The dot product of two points in ℝ² -/
def dot_product (p q : ℝ × ℝ) : ℝ :=
  p.1 * q.1 + p.2 * q.2

theorem line_passes_through_fixed_point (l : Line) 
    (h_distinct : ∃ A B : ℝ × ℝ, A ≠ B ∧ A ∈ intersection l ∧ B ∈ intersection l)
    (h_dot_product : ∃ A B : ℝ × ℝ, A ∈ intersection l ∧ B ∈ intersection l ∧ 
                     dot_product A B = -4) :
    (2, 0) ∈ {p : ℝ × ℝ | p.1 = l.slope * p.2 + l.intercept} :=
  sorry

end line_passes_through_fixed_point_l3170_317026


namespace pages_left_to_write_l3170_317087

-- Define the total number of pages for the book
def total_pages : ℕ := 500

-- Define the number of pages written on each day
def day1_pages : ℕ := 25
def day2_pages : ℕ := 2 * day1_pages
def day3_pages : ℕ := 2 * day2_pages
def day4_pages : ℕ := 10

-- Define the total number of pages written so far
def pages_written : ℕ := day1_pages + day2_pages + day3_pages + day4_pages

-- Define the number of pages left to write
def pages_left : ℕ := total_pages - pages_written

-- Theorem stating that the number of pages left to write is 315
theorem pages_left_to_write : pages_left = 315 := by sorry

end pages_left_to_write_l3170_317087


namespace combination_equality_l3170_317028

theorem combination_equality (n : ℕ) : 
  (Nat.choose (n + 1) 7 - Nat.choose n 7 = Nat.choose n 8) → n = 14 := by
  sorry

end combination_equality_l3170_317028


namespace fifth_term_of_arithmetic_sequence_l3170_317060

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem fifth_term_of_arithmetic_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 = 1 →
  (∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d) →
  a 5 = 9 := by
sorry

end fifth_term_of_arithmetic_sequence_l3170_317060


namespace max_watching_count_is_five_l3170_317054

/-- Represents a direction a guard can look -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Represents a position on the board -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents a guard on the board -/
structure Guard :=
  (pos : Position)
  (dir : Direction)

/-- The type of a board configuration -/
def Board := Fin 8 → Fin 8 → Guard

/-- Count of guards watching a specific position -/
def watchingCount (b : Board) (p : Position) : Nat :=
  sorry

/-- The maximum k for which every guard is watched by at least k other guards -/
def maxWatchingCount (b : Board) : Nat :=
  sorry

theorem max_watching_count_is_five :
  ∀ b : Board, maxWatchingCount b ≤ 5 ∧ ∃ b' : Board, maxWatchingCount b' = 5 :=
sorry

end max_watching_count_is_five_l3170_317054


namespace quadratic_equation_roots_and_triangle_l3170_317034

theorem quadratic_equation_roots_and_triangle (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - (m+3)*x₁ + m + 1 = 0 ∧ 
    x₂^2 - (m+3)*x₂ + m + 1 = 0) ∧ 
  (∃ x : ℝ, x^2 - (m+3)*x + m + 1 = 0 ∧ x = 4 → 
    ∃ y : ℝ, y^2 - (m+3)*y + m + 1 = 0 ∧ y ≠ 4 ∧ 
    4 + 4 + y = 26/3) :=
sorry

end quadratic_equation_roots_and_triangle_l3170_317034


namespace cookie_days_count_l3170_317029

-- Define the total number of school days
def total_days : ℕ := 5

-- Define the number of days with peanut butter sandwiches
def peanut_butter_days : ℕ := 2

-- Define the number of days with ham sandwiches
def ham_days : ℕ := 3

-- Define the number of days with cake
def cake_days : ℕ := 1

-- Define the probability of ham sandwich and cake on the same day
def ham_cake_prob : ℚ := 12 / 100

-- Theorem to prove
theorem cookie_days_count : 
  total_days - cake_days - peanut_butter_days = 2 :=
sorry

end cookie_days_count_l3170_317029


namespace gcd_360_504_l3170_317064

theorem gcd_360_504 : Nat.gcd 360 504 = 72 := by
  sorry

end gcd_360_504_l3170_317064


namespace cylinder_minus_cones_volume_l3170_317048

/-- The volume of a cylinder minus the volumes of two cones -/
theorem cylinder_minus_cones_volume (r h₁ h₂ h : ℝ) (hr : r = 10) (hh₁ : h₁ = 10) (hh₂ : h₂ = 16) (hh : h = 26) :
  π * r^2 * h - (1/3 * π * r^2 * h₁ + 1/3 * π * r^2 * h₂) = 2600/3 * π := by
  sorry

end cylinder_minus_cones_volume_l3170_317048


namespace geometric_sequence_properties_l3170_317047

/-- Given a geometric sequence {a_n} with sum of first n terms S_n, 
    if S_1, S_3, and S_2 form an arithmetic sequence, and a_1 - a_3 = 3, 
    then the common ratio q = -1/2 and a_1 = 4 -/
theorem geometric_sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * (a 2 / a 1))
  (h_sum : ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1)))
  (h_arith : S 3 - S 2 = S 2 - S 1)
  (h_diff : a 1 - a 3 = 3) :
  a 2 / a 1 = -1/2 ∧ a 1 = 4 := by
sorry

end geometric_sequence_properties_l3170_317047


namespace floor_power_equality_l3170_317056

theorem floor_power_equality (a b : ℝ) (h : a > 0) (h' : b > 0)
  (h_infinite : ∃ᶠ k : ℕ in atTop, ⌊a^k⌋ + ⌊b^k⌋ = ⌊a⌋^k + ⌊b⌋^k) :
  ⌊a^2014⌋ + ⌊b^2014⌋ = ⌊a⌋^2014 + ⌊b⌋^2014 := by
sorry

end floor_power_equality_l3170_317056


namespace marks_fruit_consumption_l3170_317099

/-- Given the conditions of Mark's fruit consumption, prove that he ate 5 pieces in the first four days --/
theorem marks_fruit_consumption
  (total : ℕ)
  (kept_for_next_week : ℕ)
  (brought_on_friday : ℕ)
  (h1 : total = 10)
  (h2 : kept_for_next_week = 2)
  (h3 : brought_on_friday = 3) :
  total - kept_for_next_week - brought_on_friday = 5 := by
  sorry

end marks_fruit_consumption_l3170_317099


namespace problem_statement_l3170_317074

theorem problem_statement (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) :
  (x + y) / (x - y) = Real.sqrt 3 := by
  sorry

end problem_statement_l3170_317074


namespace range_of_a_l3170_317016

-- Define the propositions p and q
def p (x : ℝ) : Prop := |4*x - 3| ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the theorem
theorem range_of_a :
  (∀ x, ¬(p x) → ¬(q x a)) ∧ 
  (∃ x, ¬(q x a) ∧ p x) →
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

end range_of_a_l3170_317016


namespace even_quadratic_implies_m_zero_l3170_317042

/-- A quadratic function f(x) = (m-1)x^2 - 2mx + 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 - 2 * m * x + 3

/-- Definition of an even function -/
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_quadratic_implies_m_zero (m : ℝ) :
  is_even (f m) → m = 0 := by
  sorry

end even_quadratic_implies_m_zero_l3170_317042


namespace solution_set_correct_l3170_317035

/-- The set of all solutions to the equation x² + y² = 3 · 2016ᶻ + 77 where x, y, and z are natural numbers -/
def SolutionSet : Set (ℕ × ℕ × ℕ) :=
  {(77, 14, 1), (14, 77, 1), (70, 35, 1), (35, 70, 1), (8, 4, 0), (4, 8, 0)}

/-- Predicate that checks if a triplet (x, y, z) satisfies the equation -/
def SatisfiesEquation (t : ℕ × ℕ × ℕ) : Prop :=
  let (x, y, z) := t
  x^2 + y^2 = 3 * 2016^z + 77

theorem solution_set_correct :
  ∀ t : ℕ × ℕ × ℕ, SatisfiesEquation t ↔ t ∈ SolutionSet := by
  sorry

end solution_set_correct_l3170_317035


namespace triangle_side_equations_l3170_317072

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The equation of a line given two points -/
def lineEquation (p1 p2 : Point) : Line :=
  let a := p2.y - p1.y
  let b := p1.x - p2.x
  let c := p2.x * p1.y - p1.x * p2.y
  { a := a, b := b, c := c }

theorem triangle_side_equations (A B C : Point)
  (hA : A = { x := -5, y := 0 })
  (hB : B = { x := 3, y := -3 })
  (hC : C = { x := 0, y := 2 }) :
  let AB := lineEquation A B
  let AC := lineEquation A C
  let BC := lineEquation B C
  AB = { a := 3, b := 8, c := 15 } ∧
  AC = { a := 2, b := -5, c := 10 } ∧
  BC = { a := 5, b := 3, c := -6 } :=
sorry

end triangle_side_equations_l3170_317072


namespace total_balloons_proof_l3170_317094

def sam_initial_balloons : ℝ := 6.0
def sam_given_balloons : ℝ := 5.0
def mary_balloons : ℝ := 7.0

theorem total_balloons_proof :
  sam_initial_balloons - sam_given_balloons + mary_balloons = 8 :=
by sorry

end total_balloons_proof_l3170_317094


namespace flagstaff_shadow_length_l3170_317063

/-- Given a flagstaff and a building casting shadows under similar conditions,
    prove that the length of the shadow cast by the flagstaff is 40.1 m. -/
theorem flagstaff_shadow_length 
  (flagstaff_height : ℝ) 
  (building_height : ℝ) 
  (building_shadow : ℝ) 
  (h1 : flagstaff_height = 17.5)
  (h2 : building_height = 12.5)
  (h3 : building_shadow = 28.75) :
  flagstaff_height / (flagstaff_height * building_shadow / building_height) = 17.5 / 40.1 :=
by sorry

end flagstaff_shadow_length_l3170_317063


namespace f_negative_one_value_l3170_317088

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The definition of f for positive x -/
def f_pos (x : ℝ) : ℝ :=
  2 * x^2 - 1

theorem f_negative_one_value
    (f : ℝ → ℝ)
    (h_odd : IsOdd f)
    (h_pos : ∀ x > 0, f x = f_pos x) :
    f (-1) = -1 := by
  sorry

end f_negative_one_value_l3170_317088


namespace star_difference_equals_45_l3170_317044

/-- The star operation defined as x ★ y = x^2y - 3x -/
def star (x y : ℝ) : ℝ := x^2 * y - 3 * x

/-- Theorem stating that (6 ★ 3) - (3 ★ 6) = 45 -/
theorem star_difference_equals_45 : star 6 3 - star 3 6 = 45 := by
  sorry

end star_difference_equals_45_l3170_317044


namespace min_value_sum_reciprocals_l3170_317053

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1)
  (h_sum : a * b + b * c + a * c = 1) :
  (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) ≥ (9 + 3 * Real.sqrt 3) / 2 := by
  sorry

end min_value_sum_reciprocals_l3170_317053
