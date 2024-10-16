import Mathlib

namespace NUMINAMATH_CALUDE_larger_number_proof_l3826_382646

theorem larger_number_proof (a b : ℕ+) 
  (hcf_eq : Nat.gcd a b = 23)
  (lcm_eq : Nat.lcm a b = 23 * 14 * 15) :
  max a b = 345 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3826_382646


namespace NUMINAMATH_CALUDE_circle_rectangle_area_relation_l3826_382649

theorem circle_rectangle_area_relation (x : ℝ) :
  let circle_radius : ℝ := x - 2
  let rectangle_length : ℝ := x - 3
  let rectangle_width : ℝ := x + 4
  let circle_area : ℝ := π * circle_radius ^ 2
  let rectangle_area : ℝ := rectangle_length * rectangle_width
  rectangle_area = 3 * circle_area →
  (12 * π + 1) / (3 * π - 1) = x + (-(12 * π + 1) / (2 * (1 - 3 * π)) + (12 * π + 1) / (2 * (1 - 3 * π))) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_rectangle_area_relation_l3826_382649


namespace NUMINAMATH_CALUDE_product_of_divisors_60_l3826_382670

-- Define the number we're working with
def n : ℕ := 60

-- Define a function to get all divisors of a natural number
def divisors (m : ℕ) : Finset ℕ :=
  sorry

-- Define the product of all divisors
def product_of_divisors (m : ℕ) : ℕ :=
  (divisors m).prod id

-- Theorem statement
theorem product_of_divisors_60 :
  product_of_divisors n = 46656000000000 := by
  sorry

end NUMINAMATH_CALUDE_product_of_divisors_60_l3826_382670


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3826_382650

/-- Represents a chess tournament with players of two ranks -/
structure ChessTournament where
  a_players : Nat
  b_players : Nat

/-- Calculates the total number of games in a chess tournament -/
def total_games (t : ChessTournament) : Nat :=
  t.a_players * t.b_players

/-- Theorem: In a chess tournament with 3 'A' players and 3 'B' players, 
    where each 'A' player faces all 'B' players, the total number of games is 9 -/
theorem chess_tournament_games :
  ∀ (t : ChessTournament), 
  t.a_players = 3 → t.b_players = 3 → total_games t = 9 := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_games_l3826_382650


namespace NUMINAMATH_CALUDE_shoebox_height_l3826_382663

/-- The height of a rectangular shoebox given specific conditions -/
theorem shoebox_height (width : ℝ) (block_side : ℝ) (uncovered_area : ℝ)
  (h_width : width = 6)
  (h_block : block_side = 4)
  (h_uncovered : uncovered_area = 8)
  : width * (block_side * block_side + uncovered_area) / width = 4 := by
  sorry

end NUMINAMATH_CALUDE_shoebox_height_l3826_382663


namespace NUMINAMATH_CALUDE_correlation_theorem_l3826_382647

/-- A function to represent the relationship between x and y -/
def f (x : ℝ) : ℝ := 0.1 * x - 10

/-- Definition of positive correlation -/
def positively_correlated (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂

/-- Definition of negative correlation -/
def negatively_correlated (f g : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → g x₁ > g x₂

/-- The main theorem -/
theorem correlation_theorem (z : ℝ → ℝ) 
  (h : negatively_correlated f z) :
  positively_correlated f ∧ negatively_correlated id z := by
  sorry

end NUMINAMATH_CALUDE_correlation_theorem_l3826_382647


namespace NUMINAMATH_CALUDE_complex_star_angle_sum_l3826_382642

/-- An n-pointed complex star is formed from a regular n-gon by extending every third side --/
structure ComplexStar where
  n : ℕ
  is_even : Even n
  n_ge_6 : n ≥ 6

/-- The sum of interior angles at the n intersections of a complex star --/
def interior_angle_sum (star : ComplexStar) : ℝ :=
  180 * (star.n - 6)

/-- Theorem: The sum of interior angles at the n intersections of a complex star is 180° * (n-6) --/
theorem complex_star_angle_sum (star : ComplexStar) :
  interior_angle_sum star = 180 * (star.n - 6) := by
  sorry

end NUMINAMATH_CALUDE_complex_star_angle_sum_l3826_382642


namespace NUMINAMATH_CALUDE_starting_number_sequence_l3826_382654

theorem starting_number_sequence (n : ℕ) : 
  (n ≤ 79) →                          -- Last number is less than or equal to 79
  (n % 11 = 0) →                      -- Last number is divisible by 11
  (∃ (m : ℕ), n = m * 11) →           -- n is a multiple of 11
  (∃ (k : ℕ), n = 11 * 7 - k * 11) →  -- n is the 7th number in the sequence
  (11 : ℕ) = n - 6 * 11               -- Starting number is 11
  := by sorry

end NUMINAMATH_CALUDE_starting_number_sequence_l3826_382654


namespace NUMINAMATH_CALUDE_unique_solution_l3826_382677

def is_solution (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ Nat.gcd a b = 1 ∧ (a + 12) * b = 3 * a * (b + 12)

theorem unique_solution : ∀ a b : ℕ, is_solution a b ↔ a = 2 ∧ b = 9 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l3826_382677


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3826_382697

theorem cube_root_equation_solution (y : ℝ) :
  (6 - 2 / y) ^ (1/3 : ℝ) = -3 → y = 2/33 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3826_382697


namespace NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l3826_382606

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
structure ArithmeticSequence (α : Type*) [Add α] where
  a : ℕ → α  -- The sequence
  d : α      -- The common difference
  h : ∀ n, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the 3rd term is 3/8 and the 15th term is 7/9, 
    the 9th term is equal to 83/144. -/
theorem ninth_term_of_arithmetic_sequence 
  (seq : ArithmeticSequence ℚ) 
  (h3 : seq.a 3 = 3/8) 
  (h15 : seq.a 15 = 7/9) : 
  seq.a 9 = 83/144 := by
sorry

end NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l3826_382606


namespace NUMINAMATH_CALUDE_jim_gave_eight_sets_to_brother_l3826_382632

/-- The number of trading cards in one set -/
def cards_per_set : ℕ := 13

/-- The number of sets Jim gave to his sister -/
def sets_to_sister : ℕ := 5

/-- The number of sets Jim gave to his friend -/
def sets_to_friend : ℕ := 2

/-- The total number of trading cards Jim had initially -/
def initial_cards : ℕ := 365

/-- The total number of trading cards Jim gave away -/
def total_given_away : ℕ := 195

/-- The number of sets Jim gave to his brother -/
def sets_to_brother : ℕ := (total_given_away - (sets_to_sister + sets_to_friend) * cards_per_set) / cards_per_set

theorem jim_gave_eight_sets_to_brother : sets_to_brother = 8 := by
  sorry

end NUMINAMATH_CALUDE_jim_gave_eight_sets_to_brother_l3826_382632


namespace NUMINAMATH_CALUDE_min_white_pairs_8x8_20black_l3826_382622

/-- Represents a grid with black and white cells -/
structure Grid :=
  (size : ℕ)
  (black_cells : ℕ)

/-- Calculates the total number of adjacent cell pairs in a square grid -/
def total_pairs (g : Grid) : ℕ :=
  2 * (g.size - 1) * g.size

/-- Calculates the maximum number of central black cells that can be placed without adjacency -/
def max_central_black (g : Grid) : ℕ :=
  (g.size - 2)^2 / 2

/-- Calculates the minimum number of adjacent white cell pairs -/
def min_white_pairs (g : Grid) : ℕ :=
  total_pairs g - (60 + min g.black_cells (max_central_black g))

/-- Theorem stating the minimum number of adjacent white cell pairs for an 8x8 grid with 20 black cells -/
theorem min_white_pairs_8x8_20black :
  let g : Grid := { size := 8, black_cells := 20 }
  min_white_pairs g = 34 := by
  sorry

end NUMINAMATH_CALUDE_min_white_pairs_8x8_20black_l3826_382622


namespace NUMINAMATH_CALUDE_compound_oxygen_atoms_l3826_382617

/-- The number of Oxygen atoms in a compound with given properties -/
def oxygenAtoms (molecularWeight : ℕ) (hydrogenAtoms carbonAtoms : ℕ) 
  (atomicWeightH atomicWeightC atomicWeightO : ℕ) : ℕ :=
  (molecularWeight - (hydrogenAtoms * atomicWeightH + carbonAtoms * atomicWeightC)) / atomicWeightO

/-- Theorem stating the number of Oxygen atoms in the compound -/
theorem compound_oxygen_atoms :
  oxygenAtoms 62 2 1 1 12 16 = 3 := by
  sorry

end NUMINAMATH_CALUDE_compound_oxygen_atoms_l3826_382617


namespace NUMINAMATH_CALUDE_mango_rate_problem_l3826_382669

/-- Calculates the rate per kg of mangoes given the total amount paid, grape weight, grape rate, and mango weight -/
def mango_rate (total_paid : ℕ) (grape_weight : ℕ) (grape_rate : ℕ) (mango_weight : ℕ) : ℕ :=
  (total_paid - grape_weight * grape_rate) / mango_weight

theorem mango_rate_problem :
  mango_rate 1376 14 54 10 = 62 := by
  sorry

end NUMINAMATH_CALUDE_mango_rate_problem_l3826_382669


namespace NUMINAMATH_CALUDE_smallest_cube_divisible_by_primes_l3826_382634

theorem smallest_cube_divisible_by_primes (p q r : ℕ) : 
  Prime p → Prime q → Prime r → p ≠ q → p ≠ r → q ≠ r → p ≠ 1 → q ≠ 1 → r ≠ 1 →
  (pqr2_cube : ℕ) → pqr2_cube = (p * q * r^2)^3 →
  (∀ m : ℕ, m^3 ∣ p^2 * q^3 * r^5 → m^3 ≥ pqr2_cube) :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_divisible_by_primes_l3826_382634


namespace NUMINAMATH_CALUDE_f_2015_value_l3826_382656

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2015_value (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : ∀ x, f (x + 2) = -f x)
  (h_01 : ∀ x, x ∈ Set.Icc 0 1 → f x = 3^x - 1) :
  f 2015 = -2 := by
sorry

end NUMINAMATH_CALUDE_f_2015_value_l3826_382656


namespace NUMINAMATH_CALUDE_rounding_down_less_than_exact_sum_l3826_382625

def fraction_a : ℚ := 2 / 3
def fraction_b : ℚ := 5 / 4

def round_down (q : ℚ) : ℤ := ⌊q⌋

theorem rounding_down_less_than_exact_sum :
  (round_down fraction_a : ℚ) + (round_down fraction_b : ℚ) ≤ fraction_a + fraction_b := by
  sorry

end NUMINAMATH_CALUDE_rounding_down_less_than_exact_sum_l3826_382625


namespace NUMINAMATH_CALUDE_return_trip_time_l3826_382600

/-- Represents the flight scenario between two towns -/
structure FlightScenario where
  d : ℝ  -- distance between towns
  p : ℝ  -- speed of plane in still air
  w : ℝ  -- speed of wind
  against_wind_time : ℝ  -- time of flight against wind
  still_air_time : ℝ  -- time of flight in still air

/-- The flight conditions as given in the problem -/
def flight_conditions (s : FlightScenario) : Prop :=
  s.against_wind_time = 90 ∧
  s.d = s.against_wind_time * (s.p - s.w) ∧
  s.d / (s.p + s.w) = s.still_air_time - 15

/-- The theorem stating that the return trip takes either 30 or 45 minutes -/
theorem return_trip_time (s : FlightScenario) :
  flight_conditions s →
  (s.d / (s.p + s.w) = 30 ∨ s.d / (s.p + s.w) = 45) :=
by sorry

end NUMINAMATH_CALUDE_return_trip_time_l3826_382600


namespace NUMINAMATH_CALUDE_area_swept_is_14_l3826_382621

/-- The area swept by a line segment during a transformation -/
def area_swept (length1 width1 length2 width2 : ℝ) : ℝ :=
  length1 * width1 + length2 * width2

/-- Theorem: The area swept by the line segment is 14 -/
theorem area_swept_is_14 :
  area_swept 4 2 3 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_area_swept_is_14_l3826_382621


namespace NUMINAMATH_CALUDE_initial_number_exists_l3826_382688

theorem initial_number_exists : ∃ N : ℝ, ∃ k : ℤ, N + 69.00000000008731 = 330 * (k : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_initial_number_exists_l3826_382688


namespace NUMINAMATH_CALUDE_fifteen_minutes_measurable_seven_intervals_with_three_ropes_l3826_382636

/-- Represents a rope that burns for 1 hour when lit from one end -/
structure Rope :=
  (burn_time : ℝ)
  (uneven : Bool)

/-- Represents the state of a burning rope -/
inductive BurningState
  | Unlit
  | LitOneEnd
  | LitBothEnds

/-- Measures time intervals using ropes -/
def measure_time (ropes : List Rope) (state : List BurningState) : ℝ :=
  sorry

/-- Counts the number of distinct time intervals that can be measured -/
def count_distinct_intervals (n : ℕ) : ℕ :=
  sorry

theorem fifteen_minutes_measurable (rope1 rope2 : Rope) 
  (h1 : rope1.burn_time = 1) (h2 : rope2.burn_time = 1)
  (h3 : rope1.uneven) (h4 : rope2.uneven) :
  ∃ (s1 s2 : BurningState), measure_time [rope1, rope2] [s1, s2] = 0.25 :=
sorry

theorem seven_intervals_with_three_ropes :
  count_distinct_intervals 3 = 7 :=
sorry

end NUMINAMATH_CALUDE_fifteen_minutes_measurable_seven_intervals_with_three_ropes_l3826_382636


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3826_382610

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ + 6*a₆ + 7*a₇ = -14 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3826_382610


namespace NUMINAMATH_CALUDE_final_price_in_euros_l3826_382679

-- Define the pin prices
def pin_prices : List ℝ := [23, 18, 20, 15, 25, 22, 19, 16, 24, 17]

-- Define the discount rate
def discount_rate : ℝ := 0.15

-- Define the sales tax rate
def sales_tax_rate : ℝ := 0.08

-- Define the exchange rate (USD to Euro)
def exchange_rate : ℝ := 0.85

-- Theorem statement
theorem final_price_in_euros :
  let original_price := pin_prices.sum
  let discounted_price := original_price * (1 - discount_rate)
  let price_with_tax := discounted_price * (1 + sales_tax_rate)
  let final_price := price_with_tax * exchange_rate
  ∃ ε > 0, |final_price - 155.28| < ε :=
sorry

end NUMINAMATH_CALUDE_final_price_in_euros_l3826_382679


namespace NUMINAMATH_CALUDE_arc_measure_constant_l3826_382641

/-- A right isosceles triangle with a rotating circle -/
structure RightIsoscelesWithCircle where
  -- The side length of the right isosceles triangle
  s : ℝ
  -- Ensure s is positive
  s_pos : 0 < s

/-- The measure of arc MBM' in degrees -/
def arcMeasure (t : RightIsoscelesWithCircle) : ℝ := 180

/-- Theorem: The arc MBM' always measures 180° -/
theorem arc_measure_constant (t : RightIsoscelesWithCircle) :
  arcMeasure t = 180 := by
  sorry

end NUMINAMATH_CALUDE_arc_measure_constant_l3826_382641


namespace NUMINAMATH_CALUDE_sqrt_inequality_reciprocal_sum_inequality_l3826_382693

-- Part 1
theorem sqrt_inequality (b : ℝ) (h : b ≥ 2) :
  Real.sqrt (b + 1) - Real.sqrt b < Real.sqrt (b - 1) - Real.sqrt (b - 2) :=
sorry

-- Part 2
theorem reciprocal_sum_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) (h4 : a + b = 2) :
  1 / a + 1 / b > 2 :=
sorry

end NUMINAMATH_CALUDE_sqrt_inequality_reciprocal_sum_inequality_l3826_382693


namespace NUMINAMATH_CALUDE_prob_a_before_b_is_one_third_l3826_382618

-- Define the set of people
inductive Person : Type
  | A
  | B
  | C

-- Define a duty arrangement as a list of people
def DutyArrangement := List Person

-- Define the set of all possible duty arrangements
def allArrangements : List DutyArrangement :=
  [[Person.A, Person.B, Person.C],
   [Person.A, Person.C, Person.B],
   [Person.C, Person.A, Person.B]]

-- Define a function to check if A is immediately before B in an arrangement
def isABeforeB (arrangement : DutyArrangement) : Bool :=
  match arrangement with
  | [Person.A, Person.B, _] => true
  | _ => false

-- Define the probability of A being immediately before B
def probABeforeB : ℚ :=
  (allArrangements.filter isABeforeB).length / allArrangements.length

-- Theorem statement
theorem prob_a_before_b_is_one_third :
  probABeforeB = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_a_before_b_is_one_third_l3826_382618


namespace NUMINAMATH_CALUDE_new_average_is_34_l3826_382689

/-- Represents a batsman's performance -/
structure BatsmanPerformance where
  innings : ℕ
  lastInningScore : ℕ
  averageIncrease : ℕ

/-- Calculates the new average score after the last inning -/
def newAverage (performance : BatsmanPerformance) : ℕ :=
  performance.lastInningScore + (performance.innings - 1) * (performance.lastInningScore / performance.innings + performance.averageIncrease - 3)

/-- Theorem stating that the new average is 34 for the given conditions -/
theorem new_average_is_34 (performance : BatsmanPerformance) 
  (h1 : performance.innings = 17)
  (h2 : performance.lastInningScore = 82)
  (h3 : performance.averageIncrease = 3) :
  newAverage performance = 34 := by
  sorry

end NUMINAMATH_CALUDE_new_average_is_34_l3826_382689


namespace NUMINAMATH_CALUDE_mimi_picked_24_shells_l3826_382613

/-- The number of seashells Mimi picked up -/
def mimi_shells : ℕ := 24

/-- The number of seashells Kyle found -/
def kyle_shells : ℕ := 2 * mimi_shells

/-- The number of seashells Leigh grabbed -/
def leigh_shells : ℕ := 16

/-- Theorem stating that Mimi picked up 24 seashells -/
theorem mimi_picked_24_shells : mimi_shells = 24 :=
by
  have h1 : kyle_shells = 2 * mimi_shells := by rfl
  have h2 : leigh_shells = kyle_shells / 3 := by sorry
  have h3 : leigh_shells = 16 := by rfl
  sorry


end NUMINAMATH_CALUDE_mimi_picked_24_shells_l3826_382613


namespace NUMINAMATH_CALUDE_sum_of_squares_of_solutions_l3826_382674

theorem sum_of_squares_of_solutions : ∃ (s₁ s₂ : ℝ), 
  (s₁^2 - 17*s₁ + 22 = 0) ∧ 
  (s₂^2 - 17*s₂ + 22 = 0) ∧ 
  (s₁^2 + s₂^2 = 245) := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_solutions_l3826_382674


namespace NUMINAMATH_CALUDE_solve_earnings_l3826_382664

def earnings_problem (first_month_daily : ℝ) : Prop :=
  let second_month_daily := 2 * first_month_daily
  let third_month_daily := second_month_daily
  let first_month_total := 30 * first_month_daily
  let second_month_total := 30 * second_month_daily
  let third_month_total := 15 * third_month_daily
  first_month_total + second_month_total + third_month_total = 1200

theorem solve_earnings : ∃ (x : ℝ), earnings_problem x ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_earnings_l3826_382664


namespace NUMINAMATH_CALUDE_quadratic_root_value_l3826_382653

theorem quadratic_root_value (d : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + d = 0 ↔ x = (3 + Real.sqrt d) / 2 ∨ x = (3 - Real.sqrt d) / 2) →
  d = 9/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l3826_382653


namespace NUMINAMATH_CALUDE_function_composition_result_l3826_382675

-- Define the functions f and g
def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem function_composition_result : f (g (Real.sqrt 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_result_l3826_382675


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3826_382639

theorem quadratic_inequality_solution (m : ℝ) :
  {x : ℝ | x^2 + (2*m+1)*x + m^2 + m > 0} = {x : ℝ | x > -m ∨ x < -m-1} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3826_382639


namespace NUMINAMATH_CALUDE_unique_monic_polynomial_l3826_382602

/-- A monic polynomial of degree 3 satisfying f(0) = 3 and f(2) = 19 -/
def f : ℝ → ℝ :=
  fun x ↦ x^3 + x^2 + 2*x + 3

/-- Theorem stating that f is the unique monic polynomial of degree 3 satisfying the given conditions -/
theorem unique_monic_polynomial :
  (∀ x, f x = x^3 + x^2 + 2*x + 3) ∧
  (∀ p : ℝ → ℝ, (∃ a b c : ℝ, ∀ x, p x = x^3 + a*x^2 + b*x + c) →
    p 0 = 3 → p 2 = 19 → p = f) := by
  sorry

end NUMINAMATH_CALUDE_unique_monic_polynomial_l3826_382602


namespace NUMINAMATH_CALUDE_consecutive_squares_sum_l3826_382661

theorem consecutive_squares_sum (x : ℤ) :
  (x + 1)^2 - x^2 = 199 → x^2 + (x + 1)^2 = 19801 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_sum_l3826_382661


namespace NUMINAMATH_CALUDE_max_pairs_sum_l3826_382631

theorem max_pairs_sum (n : ℕ) (h : n = 3011) : 
  (∃ (k : ℕ) (pairs : List (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 < p.2 ∧ p.1 ∈ Finset.range n ∧ p.2 ∈ Finset.range n) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ n) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    pairs.length = k) ∧
  (∀ (m : ℕ) (pairs : List (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 < p.2 ∧ p.1 ∈ Finset.range n ∧ p.2 ∈ Finset.range n) →
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) →
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ n) →
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) →
    pairs.length = m →
    m ≤ k) →
  k = 1204 := by
sorry

end NUMINAMATH_CALUDE_max_pairs_sum_l3826_382631


namespace NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l3826_382614

theorem smallest_k_with_remainder_one : ∃! k : ℕ,
  k > 1 ∧
  k % 19 = 1 ∧
  k % 7 = 1 ∧
  k % 3 = 1 ∧
  ∀ m : ℕ, m > 1 ∧ m % 19 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1 → k ≤ m :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l3826_382614


namespace NUMINAMATH_CALUDE_parabola_equation_l3826_382603

/-- Represents a parabola with equation y^2 = 2px -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- A point on a parabola -/
structure ParabolaPoint (para : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * para.p * x

theorem parabola_equation (para : Parabola) 
  (point : ParabolaPoint para)
  (h_ordinate : point.y = -4 * Real.sqrt 2)
  (h_distance : point.x + para.p / 2 = 6) :
  para.p = 4 ∨ para.p = 8 :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l3826_382603


namespace NUMINAMATH_CALUDE_lending_time_combined_l3826_382691

-- Define the lending time for chocolate bars and bonbons
def lending_time_chocolate (bars : ℚ) : ℚ := (3 / 2) * bars

def lending_time_bonbons (bonbons : ℚ) : ℚ := (1 / 6) * bonbons

-- Theorem to prove
theorem lending_time_combined : 
  lending_time_chocolate 1 + lending_time_bonbons 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lending_time_combined_l3826_382691


namespace NUMINAMATH_CALUDE_john_max_books_l3826_382644

/-- The maximum number of books John can buy given his money and the price per book -/
def max_books_buyable (total_money : ℕ) (price_per_book : ℕ) : ℕ :=
  total_money / price_per_book

/-- Proof that John can buy at most 14 books -/
theorem john_max_books :
  let john_money : ℕ := 4575  -- 45 dollars and 75 cents in cents
  let book_price : ℕ := 325   -- 3 dollars and 25 cents in cents
  max_books_buyable john_money book_price = 14 := by
sorry

end NUMINAMATH_CALUDE_john_max_books_l3826_382644


namespace NUMINAMATH_CALUDE_divisible_by_six_ones_digits_l3826_382667

theorem divisible_by_six_ones_digits : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, n < 10 ∧ ∃ m : ℕ, 6 ∣ (10 * m + n)) ∧ S.card = 5 :=
sorry

end NUMINAMATH_CALUDE_divisible_by_six_ones_digits_l3826_382667


namespace NUMINAMATH_CALUDE_largest_n_satisfying_condition_l3826_382684

def satisfies_condition (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ k, 2 ≤ k ∧ k ≤ n - 1 →
    a (k + 1) = (a k ^ 2 + 1) / (a (k - 1) + 1) - 1

theorem largest_n_satisfying_condition : 
  (∃ (a : ℕ → ℕ), satisfies_condition a 4) ∧
  (∀ n : ℕ, n > 4 → ¬∃ (a : ℕ → ℕ), satisfies_condition a n) :=
sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_condition_l3826_382684


namespace NUMINAMATH_CALUDE_goat_redistribution_impossibility_l3826_382678

theorem goat_redistribution_impossibility :
  ¬ ∃ (n m : ℕ), n + 7 * m = 150 ∧ 7 * n + m = 150 :=
by sorry

end NUMINAMATH_CALUDE_goat_redistribution_impossibility_l3826_382678


namespace NUMINAMATH_CALUDE_not_all_primes_from_cards_l3826_382623

/-- A card with two digits -/
structure Card :=
  (front : Nat)
  (back : Nat)

/-- Check if a number is prime -/
def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

/-- Generate all two-digit numbers from two cards -/
def twoDigitNumbers (card1 card2 : Card) : List Nat :=
  [
    10 * card1.front + card2.front,
    10 * card1.front + card2.back,
    10 * card1.back + card2.front,
    10 * card1.back + card2.back,
    10 * card2.front + card1.front,
    10 * card2.front + card1.back,
    10 * card2.back + card1.front,
    10 * card2.back + card1.back
  ]

/-- Main theorem -/
theorem not_all_primes_from_cards :
  ∀ (card1 card2 : Card),
    card1.front ≠ card1.back ∧
    card2.front ≠ card2.back ∧
    card1.front ≠ card2.front ∧
    card1.front ≠ card2.back ∧
    card1.back ≠ card2.front ∧
    card1.back ≠ card2.back ∧
    card1.front < 10 ∧ card1.back < 10 ∧ card2.front < 10 ∧ card2.back < 10 →
    ∃ (n : Nat), n ∈ twoDigitNumbers card1 card2 ∧ ¬isPrime n :=
by sorry

end NUMINAMATH_CALUDE_not_all_primes_from_cards_l3826_382623


namespace NUMINAMATH_CALUDE_weather_ratings_theorem_l3826_382638

-- Define the weather observation
structure WeatherObservation :=
  (morning : Bool)
  (afternoon : Bool)
  (evening : Bool)

-- Define the rating system for each child
def firstChildRating (w : WeatherObservation) : Bool :=
  ¬(w.morning ∨ w.afternoon ∨ w.evening)

def secondChildRating (w : WeatherObservation) : Bool :=
  ¬w.morning ∨ ¬w.afternoon ∨ ¬w.evening

-- Define the combined rating
def combinedRating (w : WeatherObservation) : Bool × Bool :=
  (firstChildRating w, secondChildRating w)

-- Define the set of all possible weather observations
def allWeatherObservations : Set WeatherObservation :=
  {w | w.morning = true ∨ w.morning = false ∧
       w.afternoon = true ∨ w.afternoon = false ∧
       w.evening = true ∨ w.evening = false}

-- Theorem statement
theorem weather_ratings_theorem :
  {(true, true), (true, false), (false, true), (false, false)} =
  {r | ∃ w ∈ allWeatherObservations, combinedRating w = r} :=
by sorry

end NUMINAMATH_CALUDE_weather_ratings_theorem_l3826_382638


namespace NUMINAMATH_CALUDE_wrong_number_calculation_l3826_382615

theorem wrong_number_calculation (n : Nat) (initial_avg correct_avg correct_num : ℝ) 
  (h1 : n = 10)
  (h2 : initial_avg = 21)
  (h3 : correct_avg = 22)
  (h4 : correct_num = 36) :
  ∃ wrong_num : ℝ,
    n * correct_avg - n * initial_avg = correct_num - wrong_num ∧
    wrong_num = 26 := by
  sorry

end NUMINAMATH_CALUDE_wrong_number_calculation_l3826_382615


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l3826_382627

/-- Two real numbers are inversely proportional if their product is constant. -/
def InverselyProportional (x y : ℝ → ℝ) :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

theorem inverse_proportion_ratio
  (x y : ℝ → ℝ)
  (h_inv_prop : InverselyProportional x y)
  (x₁ x₂ y₁ y₂ : ℝ)
  (h_x_nonzero : x₁ ≠ 0 ∧ x₂ ≠ 0)
  (h_y_nonzero : y₁ ≠ 0 ∧ y₂ ≠ 0)
  (h_x_ratio : x₁ / x₂ = 4 / 5)
  (h_y_corr : y₁ = y (x.invFun x₁) ∧ y₂ = y (x.invFun x₂)) :
  y₁ / y₂ = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l3826_382627


namespace NUMINAMATH_CALUDE_min_value_theorem_l3826_382698

theorem min_value_theorem (a b : ℝ) (h : a^2 + 2*b^2 = 6) :
  ∃ (m : ℝ), m = -2*Real.sqrt 3 ∧ ∀ x y : ℝ, x^2 + 2*y^2 = 6 → m ≤ x + Real.sqrt 2 * y :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3826_382698


namespace NUMINAMATH_CALUDE_unique_number_exists_l3826_382668

-- Define the properties of x
def is_reciprocal_not_less_than_1 (x : ℝ) : Prop := 1 / x ≥ 1
def does_not_contain_6 (x : ℕ) : Prop := ¬ (∃ d : ℕ, d < 10 ∧ d = 6 ∧ ∃ k : ℕ, x = 10 * k + d)
def cube_less_than_221 (x : ℝ) : Prop := x^3 < 221
def is_even (x : ℕ) : Prop := ∃ k : ℕ, x = 2 * k
def is_prime (x : ℕ) : Prop := Nat.Prime x
def is_multiple_of_5 (x : ℕ) : Prop := ∃ k : ℕ, x = 5 * k
def is_irrational (x : ℝ) : Prop := ¬ (∃ p q : ℤ, q ≠ 0 ∧ x = p / q)
def is_less_than_6 (x : ℝ) : Prop := x < 6
def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, x = k^2
def is_greater_than_20 (x : ℝ) : Prop := x > 20
def log_base_10_at_least_2 (x : ℝ) : Prop := Real.log x / Real.log 10 ≥ 2
def is_not_less_than_10 (x : ℝ) : Prop := x ≥ 10

-- Define the theorem
theorem unique_number_exists : ∃! x : ℕ, 
  (is_reciprocal_not_less_than_1 x ∨ does_not_contain_6 x ∨ cube_less_than_221 x) ∧
  (¬is_reciprocal_not_less_than_1 x ∨ ¬does_not_contain_6 x ∨ ¬cube_less_than_221 x) ∧
  (is_even x ∨ is_prime x ∨ is_multiple_of_5 x) ∧
  (¬is_even x ∨ ¬is_prime x ∨ ¬is_multiple_of_5 x) ∧
  (is_irrational x ∨ is_less_than_6 x ∨ is_perfect_square x) ∧
  (¬is_irrational x ∨ ¬is_less_than_6 x ∨ ¬is_perfect_square x) ∧
  (is_greater_than_20 x ∨ log_base_10_at_least_2 x ∨ is_not_less_than_10 x) ∧
  (¬is_greater_than_20 x ∨ ¬log_base_10_at_least_2 x ∨ ¬is_not_less_than_10 x) :=
by sorry


end NUMINAMATH_CALUDE_unique_number_exists_l3826_382668


namespace NUMINAMATH_CALUDE_sqrt_23_minus_1_lt_4_l3826_382681

theorem sqrt_23_minus_1_lt_4 : Real.sqrt 23 - 1 < 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_23_minus_1_lt_4_l3826_382681


namespace NUMINAMATH_CALUDE_bales_in_barn_l3826_382648

/-- The number of bales in the barn after Tim stacked new bales -/
def total_bales (initial_bales new_bales : ℕ) : ℕ :=
  initial_bales + new_bales

/-- Theorem stating that the total number of bales is 82 -/
theorem bales_in_barn : total_bales 54 28 = 82 := by
  sorry

end NUMINAMATH_CALUDE_bales_in_barn_l3826_382648


namespace NUMINAMATH_CALUDE_largest_number_l3826_382640

def a : ℚ := 883/1000
def b : ℚ := 8839/10000
def c : ℚ := 88/100
def d : ℚ := 839/1000
def e : ℚ := 889/1000

theorem largest_number : b > a ∧ b > c ∧ b > d ∧ b > e := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l3826_382640


namespace NUMINAMATH_CALUDE_one_million_divided_by_one_fourth_l3826_382601

theorem one_million_divided_by_one_fourth : 
  (1000000 : ℝ) / (1/4 : ℝ) = 4000000 := by sorry

end NUMINAMATH_CALUDE_one_million_divided_by_one_fourth_l3826_382601


namespace NUMINAMATH_CALUDE_horner_method_v3_equals_55_l3826_382620

def horner_polynomial (x : ℝ) : ℝ := 3*x^5 + 8*x^4 - 3*x^3 + 5*x^2 + 12*x - 6

def horner_v3 (x : ℝ) : ℝ :=
  let v0 := 3
  let v1 := v0 * x + 8
  let v2 := v1 * x - 3
  v2 * x + 5

theorem horner_method_v3_equals_55 :
  horner_v3 2 = 55 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v3_equals_55_l3826_382620


namespace NUMINAMATH_CALUDE_problem_solution_l3826_382655

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

-- Theorem statement
theorem problem_solution (a : ℝ) (h : a > 0) :
  -- Part I
  (∀ x : ℝ, f 1 x ≥ 3 * x + 2 ↔ x ≥ 3 ∨ x ≤ -1) ∧
  -- Part II
  ((∀ x : ℝ, f a x ≤ 0 ↔ x ≤ -1) → a = 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3826_382655


namespace NUMINAMATH_CALUDE_cubic_three_distinct_roots_in_ap_l3826_382645

/-- A cubic polynomial with coefficients a and b -/
def cubic_polynomial (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x + b

/-- Predicate for a cubic polynomial having three distinct roots in arithmetic progression -/
def has_three_distinct_roots_in_ap (a b : ℝ) : Prop :=
  ∃ (r d : ℝ), d ≠ 0 ∧
    cubic_polynomial a b (-d) = 0 ∧
    cubic_polynomial a b 0 = 0 ∧
    cubic_polynomial a b d = 0

/-- Theorem stating the condition for a cubic polynomial to have three distinct roots in arithmetic progression -/
theorem cubic_three_distinct_roots_in_ap (a b : ℝ) :
  has_three_distinct_roots_in_ap a b ↔ b = 0 ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_three_distinct_roots_in_ap_l3826_382645


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l3826_382630

theorem nested_fraction_equality : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = 21 / 55 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l3826_382630


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3826_382665

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p ^ 2 - 5 * p - 14 = 0) →
  (3 * q ^ 2 - 5 * q - 14 = 0) →
  p ≠ q →
  (3 * p ^ 2 - 3 * q ^ 2) / (p - q) = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3826_382665


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_complementary_l3826_382686

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define the set of cards
inductive Card : Type
| Red : Card
| Yellow : Card
| Green : Card
| Blue : Card

-- Define a distribution as a function from Person to Card
def Distribution := Person → Card

-- Define the event "A receives the red card"
def A_gets_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "B receives the red card"
def B_gets_red (d : Distribution) : Prop := d Person.B = Card.Red

-- State the theorem
theorem events_mutually_exclusive_but_not_complementary :
  (∀ d : Distribution, ¬(A_gets_red d ∧ B_gets_red d)) ∧
  (∃ d : Distribution, ¬(A_gets_red d ∨ B_gets_red d)) :=
by sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_complementary_l3826_382686


namespace NUMINAMATH_CALUDE_triangle_existence_l3826_382628

theorem triangle_existence (n : ℕ) (h : n ≥ 2) : 
  ∃ (points : Finset (Fin (2*n))) (segments : Finset (Fin (2*n) × Fin (2*n))),
    Finset.card segments = n^2 + 1 →
    ∃ (a b c : Fin (2*n)), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
      (a, b) ∈ segments ∧ (b, c) ∈ segments ∧ (a, c) ∈ segments :=
by sorry


end NUMINAMATH_CALUDE_triangle_existence_l3826_382628


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l3826_382683

/-- Represents the dimensions of a framed painting -/
structure FramedPainting where
  painting_height : ℝ
  painting_width : ℝ
  frame_side_width : ℝ

/-- Calculates the framed dimensions of the painting -/
def framed_dimensions (fp : FramedPainting) : ℝ × ℝ :=
  (fp.painting_width + 2 * fp.frame_side_width, 
   fp.painting_height + 6 * fp.frame_side_width)

/-- Calculates the area of the framed painting -/
def framed_area (fp : FramedPainting) : ℝ :=
  let (w, h) := framed_dimensions fp
  w * h

/-- Theorem stating the ratio of smaller to larger dimension of the framed painting -/
theorem framed_painting_ratio 
  (fp : FramedPainting)
  (h1 : fp.painting_height = 30)
  (h2 : fp.painting_width = 20)
  (h3 : framed_area fp = fp.painting_height * fp.painting_width) :
  let (w, h) := framed_dimensions fp
  min w h / max w h = 4 / 7 := by
    sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l3826_382683


namespace NUMINAMATH_CALUDE_power_of_power_l3826_382633

theorem power_of_power (a : ℝ) : (a^3)^4 = a^12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3826_382633


namespace NUMINAMATH_CALUDE_expression_value_l3826_382680

theorem expression_value (x : ℝ) (h : x^2 + 3*x + 5 = 7) : 3*x^2 + 9*x - 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3826_382680


namespace NUMINAMATH_CALUDE_unique_solution_for_n_equals_one_l3826_382692

theorem unique_solution_for_n_equals_one (n : ℕ+) :
  (∃ x : ℤ, x^(n : ℕ) + (2 + x)^(n : ℕ) + (2 - x)^(n : ℕ) = 0) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_n_equals_one_l3826_382692


namespace NUMINAMATH_CALUDE_square_root_squared_l3826_382605

theorem square_root_squared (x : ℝ) : (Real.sqrt x)^2 = 49 → x = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_root_squared_l3826_382605


namespace NUMINAMATH_CALUDE_triangle_problem_l3826_382619

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
  (h1 : Real.sin (t.A + t.C) = 8 * (Real.sin (t.B / 2))^2)
  (h2 : t.a + t.c = 6)
  (h3 : (1/2) * t.a * t.c * Real.sin t.B = 2) : 
  Real.cos t.B = 15/17 ∧ t.b = 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l3826_382619


namespace NUMINAMATH_CALUDE_x_minus_y_equals_nine_l3826_382682

theorem x_minus_y_equals_nine
  (x y : ℕ)
  (h1 : 3^x * 4^y = 19683)
  (h2 : x = 9) :
  x - y = 9 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_nine_l3826_382682


namespace NUMINAMATH_CALUDE_triangle_theorem_l3826_382690

noncomputable def triangle_problem (a b c : ℝ) (A B C : ℝ) : Prop :=
  let m : ℝ × ℝ := (a, b + c)
  let n : ℝ × ℝ := (1, Real.cos C + Real.sqrt 3 * Real.sin C)
  (∀ k : ℝ, m = k • n) ∧  -- m is parallel to n
  3 * b * c = 16 - a^2 ∧
  A = Real.pi / 3 ∧
  (∀ S : ℝ, S = 1/2 * b * c * Real.sin A → S ≤ Real.sqrt 3)

theorem triangle_theorem (a b c : ℝ) (A B C : ℝ) :
  triangle_problem a b c A B C → 
    A = Real.pi / 3 ∧
    (∃ S : ℝ, S = 1/2 * b * c * Real.sin A ∧ S = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3826_382690


namespace NUMINAMATH_CALUDE_seating_arrangements_l3826_382659

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of girls -/
def num_girls : ℕ := 3

/-- The total number of children -/
def total_children : ℕ := num_boys + num_girls

/-- Calculates the number of permutations of n elements taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of ways boys can sit together -/
def boys_together : ℕ := permutations num_boys num_boys * permutations (num_girls + 1) (num_girls + 1)

/-- The number of arrangements where no two girls sit next to each other -/
def girls_not_adjacent : ℕ := permutations num_boys num_boys * permutations (num_boys + 1) num_girls

/-- The number of ways boys can sit together and girls can sit together -/
def boys_and_girls_together : ℕ := permutations num_boys num_boys * permutations num_girls num_girls * permutations 2 2

/-- The number of arrangements where a specific boy doesn't sit at the beginning and a specific girl doesn't sit at the end -/
def specific_positions : ℕ := permutations total_children total_children - 2 * permutations (total_children - 1) (total_children - 1) + permutations (total_children - 2) (total_children - 2)

theorem seating_arrangements :
  boys_together = 576 ∧
  girls_not_adjacent = 1440 ∧
  boys_and_girls_together = 288 ∧
  specific_positions = 3720 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l3826_382659


namespace NUMINAMATH_CALUDE_probability_not_above_x_axis_is_half_l3826_382660

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Defines the specific parallelogram ABCD from the problem -/
def ABCD : Parallelogram := {
  A := { x := 3, y := 3 }
  B := { x := -3, y := -3 }
  C := { x := -9, y := -3 }
  D := { x := -3, y := 3 }
}

/-- Probability of a point being not above the x-axis in the parallelogram -/
def probability_not_above_x_axis (p : Parallelogram) : ℚ :=
  1/2

/-- Theorem stating that the probability of a randomly selected point 
    from the region determined by parallelogram ABCD being not above 
    the x-axis is 1/2 -/
theorem probability_not_above_x_axis_is_half : 
  probability_not_above_x_axis ABCD = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_probability_not_above_x_axis_is_half_l3826_382660


namespace NUMINAMATH_CALUDE_sum_of_multiples_of_4_between_63_and_151_l3826_382611

def sumOfMultiplesOf4 (lower upper : ℕ) : ℕ :=
  let first := (lower + 3) / 4 * 4
  let last := upper / 4 * 4
  let n := (last - first) / 4 + 1
  n * (first + last) / 2

theorem sum_of_multiples_of_4_between_63_and_151 :
  sumOfMultiplesOf4 63 151 = 2332 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_of_4_between_63_and_151_l3826_382611


namespace NUMINAMATH_CALUDE_flower_vase_problem_l3826_382657

/-- Calculates the number of vases needed to hold flowers given the vase capacity and flower counts. -/
def vases_needed (vase_capacity : ℕ) (carnations : ℕ) (roses : ℕ) : ℕ :=
  (carnations + roses + vase_capacity - 1) / vase_capacity

/-- Proves that given 9 flowers per vase, 4 carnations, and 23 roses, 3 vases are needed. -/
theorem flower_vase_problem : vases_needed 9 4 23 = 3 := by
  sorry

#eval vases_needed 9 4 23

end NUMINAMATH_CALUDE_flower_vase_problem_l3826_382657


namespace NUMINAMATH_CALUDE_percentage_problem_l3826_382643

theorem percentage_problem (x : ℝ) (p : ℝ) 
  (h1 : 0.2 * x = 80) 
  (h2 : p / 100 * x = 160) : 
  p = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3826_382643


namespace NUMINAMATH_CALUDE_propositions_truth_l3826_382626

theorem propositions_truth :
  (∀ a b : ℝ, a * b > 0 → a > b → 1 / a < 1 / b) ∧
  (∀ a b : ℝ, a > abs b → a^2 > b^2) ∧
  (¬ ∀ a b c d : ℝ, a > b → c > d → a - c > b - d) ∧
  (∀ a b m : ℝ, 0 < a → a < b → m > 0 → a / b < (a + m) / (b + m)) :=
by sorry

end NUMINAMATH_CALUDE_propositions_truth_l3826_382626


namespace NUMINAMATH_CALUDE_chicken_fried_steak_cost_l3826_382607

theorem chicken_fried_steak_cost (steak_egg_cost : ℝ) (james_payment : ℝ) 
  (tip_percentage : ℝ) (chicken_fried_steak_cost : ℝ) :
  steak_egg_cost = 16 →
  james_payment = 21 →
  tip_percentage = 0.20 →
  james_payment = (steak_egg_cost + chicken_fried_steak_cost) / 2 + 
    tip_percentage * (steak_egg_cost + chicken_fried_steak_cost) →
  chicken_fried_steak_cost = 14 := by
  sorry

end NUMINAMATH_CALUDE_chicken_fried_steak_cost_l3826_382607


namespace NUMINAMATH_CALUDE_escalator_steps_l3826_382671

/-- The number of steps Xiaolong takes to go down the escalator -/
def steps_down : ℕ := 30

/-- The number of steps Xiaolong takes to go up the escalator -/
def steps_up : ℕ := 90

/-- The ratio of Xiaolong's speed going up compared to going down -/
def speed_ratio : ℕ := 3

/-- The total number of visible steps on the escalator -/
def total_steps : ℕ := 60

theorem escalator_steps :
  ∃ (x : ℚ),
    (steps_down : ℚ) + (steps_down : ℚ) * x = (steps_up : ℚ) - (steps_up : ℚ) / speed_ratio * x ∧
    x = 1 ∧
    total_steps = steps_down + steps_down := by sorry

end NUMINAMATH_CALUDE_escalator_steps_l3826_382671


namespace NUMINAMATH_CALUDE_cost_price_per_meter_l3826_382608

-- Define the problem parameters
def total_meters : ℕ := 75
def selling_price : ℕ := 4950
def profit_per_meter : ℕ := 15

-- Define the theorem
theorem cost_price_per_meter :
  (selling_price - total_meters * profit_per_meter) / total_meters = 51 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_per_meter_l3826_382608


namespace NUMINAMATH_CALUDE_average_adjacent_pairs_l3826_382672

/-- Represents a row of people --/
structure Row where
  boys : ℕ
  girls : ℕ

/-- Calculates the expected number of boy-girl or girl-boy pairs in a row --/
def expectedPairs (r : Row) : ℚ :=
  let total := r.boys + r.girls
  let prob := (r.boys : ℚ) * r.girls / (total * (total - 1))
  2 * prob * (total - 1)

/-- The problem statement --/
theorem average_adjacent_pairs (row1 row2 : Row)
  (h1 : row1 = ⟨10, 12⟩)
  (h2 : row2 = ⟨15, 5⟩) :
  expectedPairs row1 + expectedPairs row2 = 2775 / 154 := by
  sorry

#eval expectedPairs ⟨10, 12⟩ + expectedPairs ⟨15, 5⟩

end NUMINAMATH_CALUDE_average_adjacent_pairs_l3826_382672


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3826_382685

/-- An isosceles triangle with sides of 4cm and 8cm has a perimeter of 20cm -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive sides
  (a = 4 ∧ b = 8) ∨ (a = 8 ∧ b = 4) →  -- given side lengths
  (a = b ∨ b = c ∨ a = c) →  -- isosceles condition
  a + b + c = 20 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3826_382685


namespace NUMINAMATH_CALUDE_problem_2019_1981_l3826_382612

theorem problem_2019_1981 : (2019 + 1981)^2 / 121 = 132231 := by
  sorry

end NUMINAMATH_CALUDE_problem_2019_1981_l3826_382612


namespace NUMINAMATH_CALUDE_x_div_y_equals_neg_two_l3826_382609

theorem x_div_y_equals_neg_two (x y : ℝ) 
  (h1 : |x| = 4)
  (h2 : |y| = 2)
  (h3 : x < y) :
  x / y = -2 := by
sorry

end NUMINAMATH_CALUDE_x_div_y_equals_neg_two_l3826_382609


namespace NUMINAMATH_CALUDE_octagon_perimeter_l3826_382687

/-- The perimeter of an octagon with alternating side lengths -/
theorem octagon_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 2 * Real.sqrt 2) :
  4 * a + 4 * b = 12 + 8 * Real.sqrt 2 := by
  sorry

#check octagon_perimeter

end NUMINAMATH_CALUDE_octagon_perimeter_l3826_382687


namespace NUMINAMATH_CALUDE_valid_fractions_are_complete_l3826_382604

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_fraction_in_range (n d : ℕ) : Prop :=
  22 / 3 < n / d ∧ n / d < 15 / 2

def is_valid_fraction (n d : ℕ) : Prop :=
  is_two_digit n ∧ is_two_digit d ∧ is_fraction_in_range n d ∧ Nat.gcd n d = 1

def valid_fractions : Set (ℕ × ℕ) :=
  {(81, 11), (82, 11), (89, 12), (96, 13), (97, 13)}

theorem valid_fractions_are_complete :
  ∀ (n d : ℕ), is_valid_fraction n d ↔ (n, d) ∈ valid_fractions := by sorry

end NUMINAMATH_CALUDE_valid_fractions_are_complete_l3826_382604


namespace NUMINAMATH_CALUDE_power_quotient_plus_five_l3826_382673

theorem power_quotient_plus_five : 23^12 / 23^5 + 5 = 148035894 := by
  sorry

end NUMINAMATH_CALUDE_power_quotient_plus_five_l3826_382673


namespace NUMINAMATH_CALUDE_dollar_four_negative_one_l3826_382662

-- Define the $ operation
def dollar (a b : ℤ) : ℤ := a * (b + 1) + a * b

-- Theorem statement
theorem dollar_four_negative_one :
  dollar 4 (-1) = -4 := by
  sorry

end NUMINAMATH_CALUDE_dollar_four_negative_one_l3826_382662


namespace NUMINAMATH_CALUDE_circle_radius_l3826_382666

theorem circle_radius (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 8*x + y^2 + 2*y + k = 0 → 
   ∃ h a : ℝ, (x - h)^2 + (y - a)^2 = 5^2) ↔ 
  k = -8 := by sorry

end NUMINAMATH_CALUDE_circle_radius_l3826_382666


namespace NUMINAMATH_CALUDE_classroom_books_count_l3826_382616

theorem classroom_books_count (num_children : ℕ) (books_per_child : ℕ) (teacher_books : ℕ) : 
  num_children = 10 → books_per_child = 7 → teacher_books = 8 →
  num_children * books_per_child + teacher_books = 78 := by
sorry

end NUMINAMATH_CALUDE_classroom_books_count_l3826_382616


namespace NUMINAMATH_CALUDE_annie_aaron_visibility_time_l3826_382637

/-- The time (in minutes) Annie can see Aaron given their speeds and distances -/
theorem annie_aaron_visibility_time : 
  let annie_speed : ℝ := 10  -- Annie's speed in miles per hour
  let aaron_speed : ℝ := 6   -- Aaron's speed in miles per hour
  let initial_distance : ℝ := 1/4  -- Initial distance between Annie and Aaron in miles
  let final_distance : ℝ := 1/4   -- Final distance between Annie and Aaron in miles
  let relative_speed : ℝ := annie_speed - aaron_speed
  let time_hours : ℝ := (initial_distance + final_distance) / relative_speed
  let time_minutes : ℝ := time_hours * 60
  time_minutes = 7.5
  := by sorry


end NUMINAMATH_CALUDE_annie_aaron_visibility_time_l3826_382637


namespace NUMINAMATH_CALUDE_base3_to_base9_first_digit_l3826_382635

/-- Represents a number in base 3 --/
def base3_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

/-- Converts a decimal number to its first digit in base 9 --/
def first_digit_base9 (n : Nat) : Nat :=
  if n < 9 then n else first_digit_base9 (n / 9)

/-- The main theorem --/
theorem base3_to_base9_first_digit :
  let y := base3_to_decimal [1,1,2,2,0,0,1,1,2,2]
  first_digit_base9 y = 5 := by
  sorry


end NUMINAMATH_CALUDE_base3_to_base9_first_digit_l3826_382635


namespace NUMINAMATH_CALUDE_regina_van_rental_cost_l3826_382696

/-- Calculates the total cost of renting a van given the rental conditions -/
def vanRentalCost (dailyRate : ℚ) (mileageRate : ℚ) (days : ℕ) (miles : ℕ) (fixedFee : ℚ) : ℚ :=
  dailyRate * days + mileageRate * miles + fixedFee

theorem regina_van_rental_cost :
  vanRentalCost 30 0.25 3 450 15 = 217.5 := by
  sorry

end NUMINAMATH_CALUDE_regina_van_rental_cost_l3826_382696


namespace NUMINAMATH_CALUDE_transistors_in_2010_l3826_382676

/-- Moore's law doubling period in years -/
def doubling_period : ℕ := 2

/-- Initial year for the calculation -/
def initial_year : ℕ := 1995

/-- Final year for the calculation -/
def final_year : ℕ := 2010

/-- Initial number of transistors in 1995 -/
def initial_transistors : ℕ := 2000000

/-- Calculate the number of transistors based on Moore's law -/
def moores_law_transistors (years : ℕ) : ℕ :=
  initial_transistors * 2^(years / doubling_period)

/-- Theorem stating the number of transistors in 2010 according to Moore's law -/
theorem transistors_in_2010 :
  moores_law_transistors (final_year - initial_year) = 256000000 := by
  sorry

end NUMINAMATH_CALUDE_transistors_in_2010_l3826_382676


namespace NUMINAMATH_CALUDE_binomial_1294_2_l3826_382629

theorem binomial_1294_2 : Nat.choose 1294 2 = 836161 := by sorry

end NUMINAMATH_CALUDE_binomial_1294_2_l3826_382629


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3826_382695

theorem other_root_of_quadratic (m : ℝ) : 
  (2^2 - 2 + m = 0) → ((-1)^2 - (-1) + m = 0) := by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3826_382695


namespace NUMINAMATH_CALUDE_runners_meeting_point_l3826_382651

/-- Represents the marathon track setup and runners' speeds -/
structure MarathonTrack where
  totalLength : ℝ
  uphillLength : ℝ
  jackHeadStart : ℝ
  jackUphillSpeed : ℝ
  jackDownhillSpeed : ℝ
  jillUphillSpeed : ℝ
  jillDownhillSpeed : ℝ

/-- Calculates the distance from the top of the hill where runners meet -/
def distanceFromTop (track : MarathonTrack) : ℝ :=
  sorry

/-- Theorem stating the distance from the top where runners meet -/
theorem runners_meeting_point (track : MarathonTrack)
  (h1 : track.totalLength = 16)
  (h2 : track.uphillLength = 8)
  (h3 : track.jackHeadStart = 0.25)
  (h4 : track.jackUphillSpeed = 12)
  (h5 : track.jackDownhillSpeed = 18)
  (h6 : track.jillUphillSpeed = 14)
  (h7 : track.jillDownhillSpeed = 20) :
  distanceFromTop track = 511 / 32 := by
  sorry

end NUMINAMATH_CALUDE_runners_meeting_point_l3826_382651


namespace NUMINAMATH_CALUDE_greatest_b_for_no_negative_seven_in_range_l3826_382694

theorem greatest_b_for_no_negative_seven_in_range : 
  ∃ (b : ℤ), b = 10 ∧ 
  (∀ (x : ℝ), x^2 + (b : ℝ) * x + 20 ≠ -7) ∧
  (∀ (b' : ℤ), b' > b → ∃ (x : ℝ), x^2 + (b' : ℝ) * x + 20 = -7) :=
by sorry

end NUMINAMATH_CALUDE_greatest_b_for_no_negative_seven_in_range_l3826_382694


namespace NUMINAMATH_CALUDE_part_one_part_two_l3826_382658

-- Define combinatorial and permutation functions
def C (n k : ℕ) : ℕ := Nat.choose n k
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Part 1: Prove that C₁₀⁴ - C₇³A₃³ = 0
theorem part_one : C 10 4 - C 7 3 * A 3 3 = 0 := by sorry

-- Part 2: Prove that the solution to 3A₈ˣ = 4A₉ˣ⁻¹ is x = 6
theorem part_two : ∃ (x : ℕ), 3 * A 8 x = 4 * A 9 (x - 1) ∧ x = 6 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3826_382658


namespace NUMINAMATH_CALUDE_max_balloons_proof_l3826_382624

/-- Represents the maximum number of balloons that can be purchased given a budget and pricing scheme. -/
def max_balloons (budget : ℕ) (regular_price : ℕ) (set_price : ℕ) : ℕ :=
  (budget / set_price) * 3

/-- Proves that given $120 to spend, with balloons priced at $4 each, and a special sale where every set of 3 balloons costs $7, the maximum number of balloons that can be purchased is 51. -/
theorem max_balloons_proof :
  max_balloons 120 4 7 = 51 := by
  sorry

end NUMINAMATH_CALUDE_max_balloons_proof_l3826_382624


namespace NUMINAMATH_CALUDE_largest_number_l3826_382699

-- Define a function to convert a number from base n to decimal
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

-- Define the numbers in their respective bases
def num_A : Nat := to_decimal [5, 8] 9
def num_B : Nat := to_decimal [0, 0, 2] 6
def num_C : Nat := to_decimal [8, 6] 11
def num_D : Nat := 70

-- Theorem statement
theorem largest_number :
  num_A = max num_A (max num_B (max num_C num_D)) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l3826_382699


namespace NUMINAMATH_CALUDE_boy_age_theorem_l3826_382652

/-- The age of the boy not included in either group -/
def X (A : ℝ) : ℝ := 606 - 11 * A

/-- Theorem stating the relationship between X and A -/
theorem boy_age_theorem (A : ℝ) :
  let first_six_total : ℝ := 6 * 49
  let last_six_total : ℝ := 6 * 52
  let total_boys : ℕ := 11
  X A = first_six_total + last_six_total - total_boys * A := by
  sorry

end NUMINAMATH_CALUDE_boy_age_theorem_l3826_382652
