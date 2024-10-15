import Mathlib

namespace NUMINAMATH_CALUDE_fraction_product_proof_l3800_380080

theorem fraction_product_proof : 
  (7 : ℚ) / 4 * 8 / 14 * 14 / 8 * 16 / 40 * 35 / 20 * 18 / 45 * 49 / 28 * 32 / 64 = 49 / 200 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_proof_l3800_380080


namespace NUMINAMATH_CALUDE_not_A_implies_not_all_mc_or_not_three_math_l3800_380071

-- Define the predicates
def got_all_mc_right (student : String) : Prop := sorry
def solved_at_least_three_math (student : String) : Prop := sorry
def received_A (student : String) : Prop := sorry

-- Ms. Carroll's rule
axiom ms_carroll_rule (student : String) :
  got_all_mc_right student ∧ solved_at_least_three_math student → received_A student

-- Theorem to prove
theorem not_A_implies_not_all_mc_or_not_three_math (student : String) :
  ¬(received_A student) → ¬(got_all_mc_right student) ∨ ¬(solved_at_least_three_math student) :=
by sorry

end NUMINAMATH_CALUDE_not_A_implies_not_all_mc_or_not_three_math_l3800_380071


namespace NUMINAMATH_CALUDE_equation_solution_l3800_380076

theorem equation_solution :
  ∃ x : ℚ, (0.05 * x + 0.12 * (30 + x) = 15.6) ∧ (x = 1200 / 17) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3800_380076


namespace NUMINAMATH_CALUDE_larry_remaining_cards_l3800_380066

/-- Given that Larry has 352 cards initially and Dennis takes 47 cards away,
    prove that Larry will have 305 cards remaining. -/
theorem larry_remaining_cards (initial_cards : ℕ) (cards_taken : ℕ) :
  initial_cards = 352 →
  cards_taken = 47 →
  initial_cards - cards_taken = 305 := by
  sorry

end NUMINAMATH_CALUDE_larry_remaining_cards_l3800_380066


namespace NUMINAMATH_CALUDE_smores_cost_example_l3800_380095

/-- The cost of supplies for S'mores given the number of people, S'mores per person, and cost per set of S'mores. -/
def smoresCost (numPeople : ℕ) (smoresPerPerson : ℕ) (costPerSet : ℚ) (smoresPerSet : ℕ) : ℚ :=
  (numPeople * smoresPerPerson : ℚ) / smoresPerSet * costPerSet

/-- Theorem: The cost of S'mores supplies for 8 people eating 3 S'mores each, where 4 S'mores cost $3, is $18. -/
theorem smores_cost_example : smoresCost 8 3 3 4 = 18 := by
  sorry

#eval smoresCost 8 3 3 4

end NUMINAMATH_CALUDE_smores_cost_example_l3800_380095


namespace NUMINAMATH_CALUDE_equation_solution_l3800_380079

theorem equation_solution :
  ∃ (a b p q : ℝ),
    (∀ x : ℝ, (2*x - 1)^20 - (a*x + b)^20 = (x^2 + p*x + q)^10) ↔
    ((a = (2^20 - 1)^(1/20) ∧ b = -(2^20 - 1)^(1/20)/2) ∨
     (a = -(2^20 - 1)^(1/20) ∧ b = (2^20 - 1)^(1/20)/2)) ∧
    p = -1 ∧ q = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3800_380079


namespace NUMINAMATH_CALUDE_max_m_value_eight_is_achievable_max_m_is_eight_l3800_380027

/-- The function f(x) = x^2 + 2x -/
def f (x : ℝ) : ℝ := x^2 + 2*x

/-- The theorem stating the maximum value of m -/
theorem max_m_value (t : ℝ) (m : ℝ) (h : ∀ x ∈ Set.Icc 1 m, f (x + t) ≤ 3*x) :
  m ≤ 8 :=
sorry

/-- The theorem stating that 8 is achievable -/
theorem eight_is_achievable :
  ∃ t : ℝ, ∀ x ∈ Set.Icc 1 8, f (x + t) ≤ 3*x :=
sorry

/-- The main theorem combining the above results -/
theorem max_m_is_eight :
  (∃ m : ℝ, ∃ t : ℝ, (∀ x ∈ Set.Icc 1 m, f (x + t) ≤ 3*x) ∧
    (∀ m' > m, ¬∃ t' : ℝ, ∀ x ∈ Set.Icc 1 m', f (x + t') ≤ 3*x)) ∧
  (∃ t : ℝ, ∀ x ∈ Set.Icc 1 8, f (x + t) ≤ 3*x) :=
sorry

end NUMINAMATH_CALUDE_max_m_value_eight_is_achievable_max_m_is_eight_l3800_380027


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3800_380061

/-- Represents a hyperbola in 2D space -/
structure Hyperbola where
  /-- The equation of the hyperbola in the form x²/a² - y²/b² = 1 -/
  equation : ℝ → ℝ → Prop

/-- Properties of a specific hyperbola -/
def hyperbola_properties (h : Hyperbola) : Prop :=
  ∃ (a b : ℝ),
    -- The center is at the origin
    h.equation 0 0 ∧
    -- The right focus is at (3,0)
    (∃ (x y : ℝ), h.equation x y ∧ x = 3 ∧ y = 0) ∧
    -- The eccentricity is 3/2
    (3 / a = 3 / 2) ∧
    -- The equation of the hyperbola
    (∀ (x y : ℝ), h.equation x y ↔ x^2 / a^2 - y^2 / b^2 = 1)

/-- Theorem: The hyperbola with given properties has the equation x²/4 - y²/5 = 1 -/
theorem hyperbola_equation (h : Hyperbola) (hp : hyperbola_properties h) :
  ∀ (x y : ℝ), h.equation x y ↔ x^2 / 4 - y^2 / 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3800_380061


namespace NUMINAMATH_CALUDE_temperature_drop_l3800_380078

/-- Given an initial temperature and a temperature drop, calculate the final temperature. -/
def final_temperature (initial : ℤ) (drop : ℕ) : ℤ :=
  initial - drop

/-- Theorem: When the initial temperature is 3℃ and it drops by 5℃, the final temperature is -2℃. -/
theorem temperature_drop : final_temperature 3 5 = -2 := by
  sorry

end NUMINAMATH_CALUDE_temperature_drop_l3800_380078


namespace NUMINAMATH_CALUDE_fixed_point_of_parabola_family_l3800_380067

/-- The fixed point of the family of parabolas y = 4x^2 + 2tx - 3t is (1.5, 9) -/
theorem fixed_point_of_parabola_family (t : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 + 2 * t * x - 3 * t
  f 1.5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_parabola_family_l3800_380067


namespace NUMINAMATH_CALUDE_max_value_on_interval_l3800_380008

def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem max_value_on_interval : 
  ∃ (c : ℝ), c ∈ Set.Icc 0 4 ∧ 
  (∀ x, x ∈ Set.Icc 0 4 → f x ≤ f c) ∧
  f c = 10 :=
sorry

end NUMINAMATH_CALUDE_max_value_on_interval_l3800_380008


namespace NUMINAMATH_CALUDE_distinct_exponentiation_values_l3800_380074

-- Define a function to represent different parenthesizations of 3^3^3^3
def exponentiation_order (n : Nat) : Nat :=
  match n with
  | 0 => 3^(3^(3^3))  -- standard order
  | 1 => 3^((3^3)^3)
  | 2 => (3^3)^(3^3)
  | 3 => (3^(3^3))^3
  | _ => ((3^3)^3)^3

-- Theorem statement
theorem distinct_exponentiation_values :
  ∃ (S : Finset Nat), (Finset.card S = 5) ∧ 
  (∀ (i : Nat), i < 5 → exponentiation_order i ∈ S) ∧
  (∀ (x : Nat), x ∈ S → ∃ (i : Nat), i < 5 ∧ exponentiation_order i = x) :=
sorry

end NUMINAMATH_CALUDE_distinct_exponentiation_values_l3800_380074


namespace NUMINAMATH_CALUDE_frost_39_cupcakes_in_6_minutes_l3800_380059

/-- The number of cupcakes frosted by three people in a given time -/
def cupcakes_frosted (bob_rate cagney_rate lacey_rate time : ℚ) : ℚ :=
  (bob_rate + cagney_rate + lacey_rate) * time

/-- Theorem stating that Bob, Cagney, and Lacey can frost 39 cupcakes in 6 minutes -/
theorem frost_39_cupcakes_in_6_minutes :
  cupcakes_frosted (1/40) (1/20) (1/30) 360 = 39 := by
  sorry

end NUMINAMATH_CALUDE_frost_39_cupcakes_in_6_minutes_l3800_380059


namespace NUMINAMATH_CALUDE_mr_green_potato_yield_l3800_380058

/-- Calculates the expected potato yield from a rectangular garden --/
def expected_potato_yield (length_steps : ℕ) (width_steps : ℕ) (feet_per_step : ℕ) (yield_per_sqft : ℚ) : ℚ :=
  let length_feet := length_steps * feet_per_step
  let width_feet := width_steps * feet_per_step
  let area_sqft := length_feet * width_feet
  (area_sqft : ℚ) * yield_per_sqft

/-- Theorem stating the expected potato yield for Mr. Green's garden --/
theorem mr_green_potato_yield :
  expected_potato_yield 15 20 2 (1/2) = 600 := by
  sorry

end NUMINAMATH_CALUDE_mr_green_potato_yield_l3800_380058


namespace NUMINAMATH_CALUDE_leanna_cd_purchase_l3800_380045

/-- Represents the number of CDs Leanna can buy -/
def max_cds (total : ℕ) (cd_price : ℕ) (cassette_price : ℕ) : ℕ :=
  (total - cassette_price) / cd_price

/-- The cassette price satisfies the given condition -/
def cassette_price_condition (cd_price : ℕ) (cassette_price : ℕ) : Prop :=
  cd_price + 2 * cassette_price + 5 = 37

theorem leanna_cd_purchase :
  ∀ (total : ℕ) (cd_price : ℕ) (cassette_price : ℕ),
    total = 37 →
    cd_price = 14 →
    cassette_price_condition cd_price cassette_price →
    max_cds total cd_price cassette_price = 2 :=
by sorry

end NUMINAMATH_CALUDE_leanna_cd_purchase_l3800_380045


namespace NUMINAMATH_CALUDE_probability_circle_or_square_l3800_380054

-- Define the total number of figures
def total_figures : ℕ := 10

-- Define the number of circles
def num_circles : ℕ := 3

-- Define the number of squares
def num_squares : ℕ := 4

-- Define the number of triangles
def num_triangles : ℕ := 3

-- Theorem statement
theorem probability_circle_or_square :
  (num_circles + num_squares : ℚ) / total_figures = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_circle_or_square_l3800_380054


namespace NUMINAMATH_CALUDE_permutation_five_three_l3800_380014

/-- The number of permutations of n objects taken r at a time -/
def permutation (n : ℕ) (r : ℕ) : ℕ :=
  if r > n then 0
  else (n - r + 1).factorial / (n - r).factorial

theorem permutation_five_three :
  permutation 5 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_permutation_five_three_l3800_380014


namespace NUMINAMATH_CALUDE_min_hindi_speakers_l3800_380086

theorem min_hindi_speakers (total : ℕ) (english : ℕ) (both : ℕ) (hindi : ℕ) : 
  total = 40 → 
  english = 20 → 
  both ≥ 10 → 
  hindi = total + both - english →
  hindi ≥ 30 :=
by sorry

end NUMINAMATH_CALUDE_min_hindi_speakers_l3800_380086


namespace NUMINAMATH_CALUDE_uniform_prices_theorem_l3800_380090

/-- Represents a servant's employment terms and compensation --/
structure Servant where
  annual_salary : ℕ  -- Annual salary in Rupees
  service_months : ℕ  -- Months of service completed
  partial_payment : ℕ  -- Partial payment received in Rupees

/-- Calculates the price of a uniform given a servant's terms and compensation --/
def uniform_price (s : Servant) : ℕ :=
  (s.service_months * s.annual_salary - 12 * s.partial_payment) / (12 - s.service_months)

theorem uniform_prices_theorem (servant_a servant_b servant_c : Servant) 
  (h_a : servant_a = { annual_salary := 500, service_months := 9, partial_payment := 250 })
  (h_b : servant_b = { annual_salary := 800, service_months := 6, partial_payment := 300 })
  (h_c : servant_c = { annual_salary := 1200, service_months := 4, partial_payment := 200 }) :
  uniform_price servant_a = 500 ∧ 
  uniform_price servant_b = 200 ∧ 
  uniform_price servant_c = 300 := by
  sorry

#eval uniform_price { annual_salary := 500, service_months := 9, partial_payment := 250 }
#eval uniform_price { annual_salary := 800, service_months := 6, partial_payment := 300 }
#eval uniform_price { annual_salary := 1200, service_months := 4, partial_payment := 200 }

end NUMINAMATH_CALUDE_uniform_prices_theorem_l3800_380090


namespace NUMINAMATH_CALUDE_senior_mean_score_l3800_380007

theorem senior_mean_score 
  (total_students : ℕ) 
  (overall_mean : ℝ) 
  (senior_count : ℝ) 
  (non_senior_count : ℝ) 
  (senior_mean : ℝ) 
  (non_senior_mean : ℝ) :
  total_students = 120 →
  overall_mean = 150 →
  non_senior_count = senior_count + 0.75 * senior_count →
  senior_mean = 2 * non_senior_mean →
  senior_count + non_senior_count = total_students →
  senior_count * senior_mean + non_senior_count * non_senior_mean = total_students * overall_mean →
  senior_mean = 220 :=
by sorry

end NUMINAMATH_CALUDE_senior_mean_score_l3800_380007


namespace NUMINAMATH_CALUDE_probability_triangle_or_hexagon_l3800_380037

theorem probability_triangle_or_hexagon :
  let total_figures : ℕ := 12
  let triangles : ℕ := 3
  let squares : ℕ := 4
  let circles : ℕ := 3
  let hexagons : ℕ := 2
  let favorable_outcomes : ℕ := triangles + hexagons
  (favorable_outcomes : ℚ) / total_figures = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_triangle_or_hexagon_l3800_380037


namespace NUMINAMATH_CALUDE_parking_lot_perimeter_l3800_380024

theorem parking_lot_perimeter (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧ 
  a^2 + b^2 = 28^2 ∧ 
  a * b = 180 → 
  2 * (a + b) = 68 := by
sorry

end NUMINAMATH_CALUDE_parking_lot_perimeter_l3800_380024


namespace NUMINAMATH_CALUDE_rectangle_split_divisibility_l3800_380084

/-- The number of ways to split a 3 × n rectangle into 1 × 2 rectangles -/
def N (n : ℕ) : ℕ :=
  sorry

/-- The number of ways to split a 3 × n rectangle into 1 × 2 rectangles,
    where the last row has exactly two cells filled -/
def γ (n : ℕ) : ℕ :=
  sorry

theorem rectangle_split_divisibility (n : ℕ) (h : n = 200) :
  3 ∣ N n := by
  sorry

end NUMINAMATH_CALUDE_rectangle_split_divisibility_l3800_380084


namespace NUMINAMATH_CALUDE_equation_solution_l3800_380099

theorem equation_solution :
  ∃ x : ℚ, 6 * (2 * x + 3) - 4 = -3 * (2 - 5 * x) + 3 * x ∧ x = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3800_380099


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_M_over_100_l3800_380003

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def M : ℚ := (factorial 2 * factorial 19) * (
  1 / (factorial 3 * factorial 18) +
  1 / (factorial 4 * factorial 17) +
  1 / (factorial 5 * factorial 16) +
  1 / (factorial 6 * factorial 15) +
  1 / (factorial 7 * factorial 14) +
  1 / (factorial 8 * factorial 13) +
  1 / (factorial 9 * factorial 12) +
  1 / (factorial 10 * factorial 11)
)

theorem greatest_integer_less_than_M_over_100 : 
  ⌊M / 100⌋ = 49 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_M_over_100_l3800_380003


namespace NUMINAMATH_CALUDE_sum_coordinates_of_endpoint_l3800_380015

/-- Given a line segment CD with midpoint M(5,5) and endpoint C(7,3),
    the sum of the coordinates of the other endpoint D is 10. -/
theorem sum_coordinates_of_endpoint (C D M : ℝ × ℝ) : 
  M = (5, 5) →
  C = (7, 3) →
  M.1 = (C.1 + D.1) / 2 →
  M.2 = (C.2 + D.2) / 2 →
  D.1 + D.2 = 10 := by
  sorry

#check sum_coordinates_of_endpoint

end NUMINAMATH_CALUDE_sum_coordinates_of_endpoint_l3800_380015


namespace NUMINAMATH_CALUDE_notebook_pen_cost_l3800_380011

/-- The cost of notebooks and pens -/
theorem notebook_pen_cost (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 7.40)
  (h2 : 2 * x + 5 * y = 9.75) :
  x + 3 * y = 5.53 := by
  sorry

end NUMINAMATH_CALUDE_notebook_pen_cost_l3800_380011


namespace NUMINAMATH_CALUDE_jellybean_probability_l3800_380017

def total_jellybeans : ℕ := 12
def red_jellybeans : ℕ := 5
def blue_jellybeans : ℕ := 2
def yellow_jellybeans : ℕ := 5
def picked_jellybeans : ℕ := 4

theorem jellybean_probability :
  (Nat.choose red_jellybeans 3 * Nat.choose (blue_jellybeans + yellow_jellybeans) 1) /
  Nat.choose total_jellybeans picked_jellybeans = 14 / 99 :=
by sorry

end NUMINAMATH_CALUDE_jellybean_probability_l3800_380017


namespace NUMINAMATH_CALUDE_students_play_both_football_and_cricket_l3800_380032

/-- The number of students who play both football and cricket -/
def students_play_both (total students_football students_cricket students_neither : ℕ) : ℕ :=
  students_football + students_cricket - (total - students_neither)

/-- Theorem: Given the conditions, 130 students play both football and cricket -/
theorem students_play_both_football_and_cricket :
  students_play_both 420 325 175 50 = 130 := by
  sorry

#eval students_play_both 420 325 175 50

end NUMINAMATH_CALUDE_students_play_both_football_and_cricket_l3800_380032


namespace NUMINAMATH_CALUDE_tournament_max_k_l3800_380069

def num_teams : ℕ := 20

-- Ice Hockey scoring system
def ice_hockey_max_k (n : ℕ) : ℕ := n - 2

-- Volleyball scoring system
def volleyball_max_k (n : ℕ) : ℕ :=
  if n % 2 = 0 then n - 5 else n - 4

theorem tournament_max_k :
  ice_hockey_max_k num_teams = 18 ∧
  volleyball_max_k num_teams = 15 := by
  sorry

#eval ice_hockey_max_k num_teams
#eval volleyball_max_k num_teams

end NUMINAMATH_CALUDE_tournament_max_k_l3800_380069


namespace NUMINAMATH_CALUDE_line_parameterization_solution_l3800_380039

/-- The line equation y = 2x - 8 -/
def line_eq (x y : ℝ) : Prop := y = 2 * x - 8

/-- The parameterization of the line -/
def parameterization (s m t : ℝ) : ℝ × ℝ :=
  (s + 6 * t, 5 + m * t)

/-- The theorem stating that s = 13/2 and m = 11 satisfy the conditions -/
theorem line_parameterization_solution :
  let s : ℝ := 13/2
  let m : ℝ := 11
  ∃ t : ℝ, 
    let (x, y) := parameterization s m t
    x = 12 ∧ line_eq x y :=
  sorry

end NUMINAMATH_CALUDE_line_parameterization_solution_l3800_380039


namespace NUMINAMATH_CALUDE_solution_system_equations_l3800_380097

theorem solution_system_equations :
  let x : ℝ := -1
  let y : ℝ := 2
  ((x^2 + y) * Real.sqrt (y - 2*x) - 4 = 2*x^2 + 2*x + y) ∧
  (x^3 - x^2 - y + 6 = 4 * Real.sqrt (x + 1) + 2 * Real.sqrt (y - 1)) := by
  sorry

end NUMINAMATH_CALUDE_solution_system_equations_l3800_380097


namespace NUMINAMATH_CALUDE_investment_growth_l3800_380020

/-- The compound interest formula -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem investment_growth :
  let principal : ℝ := 2500
  let rate : ℝ := 0.06
  let time : ℕ := 21
  let final_amount := compound_interest principal rate time
  ∃ ε > 0, |final_amount - 8280.91| < ε :=
by sorry

end NUMINAMATH_CALUDE_investment_growth_l3800_380020


namespace NUMINAMATH_CALUDE_non_sunday_avg_is_120_l3800_380010

/-- Represents a library's visitor statistics for a month. -/
structure LibraryStats where
  total_days : Nat
  sunday_count : Nat
  sunday_avg : Nat
  overall_avg : Nat

/-- Calculates the average number of visitors on non-Sunday days. -/
def non_sunday_avg (stats : LibraryStats) : Rat :=
  let non_sunday_days := stats.total_days - stats.sunday_count
  let total_visitors := stats.overall_avg * stats.total_days
  let sunday_visitors := stats.sunday_avg * stats.sunday_count
  (total_visitors - sunday_visitors) / non_sunday_days

/-- Theorem stating the average number of visitors on non-Sunday days. -/
theorem non_sunday_avg_is_120 (stats : LibraryStats) 
  (h1 : stats.total_days = 30)
  (h2 : stats.sunday_count = 5)
  (h3 : stats.sunday_avg = 150)
  (h4 : stats.overall_avg = 125) :
  non_sunday_avg stats = 120 := by
  sorry

#eval non_sunday_avg ⟨30, 5, 150, 125⟩

end NUMINAMATH_CALUDE_non_sunday_avg_is_120_l3800_380010


namespace NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l3800_380051

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem tenth_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a2 : a 2 = 2) 
  (h_a3 : a 3 = 4) : 
  a 10 = 18 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l3800_380051


namespace NUMINAMATH_CALUDE_expected_value_is_one_l3800_380021

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- The probability of getting heads or tails -/
def flip_probability : CoinFlip → ℚ
| CoinFlip.Heads => 1/2
| CoinFlip.Tails => 1/2

/-- The payoff for each outcome -/
def payoff : CoinFlip → ℤ
| CoinFlip.Heads => 5
| CoinFlip.Tails => -3

/-- The expected value of a single coin flip -/
def expected_value : ℚ :=
  (flip_probability CoinFlip.Heads * payoff CoinFlip.Heads) +
  (flip_probability CoinFlip.Tails * payoff CoinFlip.Tails)

theorem expected_value_is_one :
  expected_value = 1 := by sorry

end NUMINAMATH_CALUDE_expected_value_is_one_l3800_380021


namespace NUMINAMATH_CALUDE_equal_roots_implies_m_value_l3800_380009

/-- Prove that if the equation (x(x-1)-(m+1))/((x-1)(m-1)) = x/m has all equal roots, then m = -1/2 -/
theorem equal_roots_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, (x * (x - 1) - (m + 1)) / ((x - 1) * (m - 1)) = x / m) →
  (∃! x : ℝ, x * (x - 1) - (m + 1) = 0) →
  m = -1/2 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_implies_m_value_l3800_380009


namespace NUMINAMATH_CALUDE_ordered_pairs_1764_l3800_380046

/-- The number of ordered pairs of positive integers (x,y) that satisfy xy = n,
    where n has the prime factorization p₁^a₁ * p₂^a₂ * ... * pₖ^aₖ -/
def count_ordered_pairs (n : ℕ) (primes : List ℕ) (exponents : List ℕ) : ℕ :=
  sorry

theorem ordered_pairs_1764 :
  count_ordered_pairs 1764 [2, 3, 7] [2, 2, 2] = 27 :=
sorry

end NUMINAMATH_CALUDE_ordered_pairs_1764_l3800_380046


namespace NUMINAMATH_CALUDE_unique_solution_l3800_380092

/-- A quadratic polynomial with exactly one root -/
structure UniqueRootQuadratic where
  a : ℝ
  b : ℝ
  has_unique_root : ∃! x : ℝ, x^2 + a * x + b = 0

/-- The composite polynomial with exactly one root -/
def composite_poly (g : UniqueRootQuadratic) (x : ℝ) : ℝ :=
  g.a * (x^5 + 2*x - 1) + g.b + g.a * (x^5 + 3*x + 1) + g.b

/-- Theorem stating the unique solution for a and b -/
theorem unique_solution (g : UniqueRootQuadratic) 
  (h : ∃! x : ℝ, composite_poly g x = 0) : 
  g.a = 74 ∧ g.b = 1369 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3800_380092


namespace NUMINAMATH_CALUDE_equal_angles_in_intersecting_circles_l3800_380053

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the point type
def Point := ℝ × ℝ

-- Define the angle type
def Angle := ℝ

-- Define the function to check if three points are collinear
def collinear (p1 p2 p3 : Point) : Prop := sorry

-- Define the function to check if a point lies on a circle
def on_circle (p : Point) (c : Circle) : Prop := sorry

-- Define the function to calculate the angle between three points
def angle (p1 p2 p3 : Point) : Angle := sorry

-- Define the theorem
theorem equal_angles_in_intersecting_circles 
  (c1 c2 : Circle) 
  (K M A B C D : Point) : 
  (∃ (K M : Point), on_circle K c1 ∧ on_circle K c2 ∧ on_circle M c1 ∧ on_circle M c2) →
  (on_circle A c1 ∧ on_circle B c2 ∧ collinear K A B) →
  (on_circle C c1 ∧ on_circle D c2 ∧ collinear K C D) →
  angle M A B = angle M C D := by
  sorry

end NUMINAMATH_CALUDE_equal_angles_in_intersecting_circles_l3800_380053


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3800_380094

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (-3, 1/2) and (7, 9) is equal to 6.75 -/
theorem midpoint_coordinate_sum : 
  let x₁ : ℝ := -3
  let y₁ : ℝ := 1/2
  let x₂ : ℝ := 7
  let y₂ : ℝ := 9
  let mx : ℝ := (x₁ + x₂) / 2
  let my : ℝ := (y₁ + y₂) / 2
  mx + my = 6.75 := by sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3800_380094


namespace NUMINAMATH_CALUDE_road_system_exists_road_system_impossible_l3800_380050

/-- A graph representing the road system in the kingdom --/
structure RoadSystem where
  cities : Finset ℕ
  roads : cities → cities → Prop

/-- The distance between two cities in the road system --/
def distance (G : RoadSystem) (a b : G.cities) : ℕ :=
  sorry

/-- The degree (number of outgoing roads) of a city in the road system --/
def degree (G : RoadSystem) (a : G.cities) : ℕ :=
  sorry

/-- Theorem stating the existence of a road system satisfying the king's requirements --/
theorem road_system_exists :
  ∃ (G : RoadSystem),
    G.cities.card = 16 ∧
    (∀ a b : G.cities, a ≠ b → distance G a b ≤ 2) ∧
    (∀ a : G.cities, degree G a ≤ 5) :=
  sorry

/-- Theorem stating the impossibility of a road system with reduced maximum degree --/
theorem road_system_impossible :
  ¬∃ (G : RoadSystem),
    G.cities.card = 16 ∧
    (∀ a b : G.cities, a ≠ b → distance G a b ≤ 2) ∧
    (∀ a : G.cities, degree G a ≤ 4) :=
  sorry

end NUMINAMATH_CALUDE_road_system_exists_road_system_impossible_l3800_380050


namespace NUMINAMATH_CALUDE_permutation_combination_equality_l3800_380077

theorem permutation_combination_equality (n : ℕ) : (n.factorial / (n - 3).factorial = 6 * n.choose 4) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_permutation_combination_equality_l3800_380077


namespace NUMINAMATH_CALUDE_present_age_of_B_l3800_380096

/-- Given two natural numbers A and B representing ages, proves that B is 41 years old
    given the conditions:
    1) In 10 years, A will be twice as old as B was 10 years ago.
    2) A is now 11 years older than B. -/
theorem present_age_of_B (A B : ℕ) 
    (h1 : A + 10 = 2 * (B - 10))
    (h2 : A = B + 11) : 
  B = 41 := by
  sorry


end NUMINAMATH_CALUDE_present_age_of_B_l3800_380096


namespace NUMINAMATH_CALUDE_president_secretary_selection_l3800_380073

/-- The number of ways to select 2 people from n people and assign them to 2 distinct roles -/
def permutation_two_roles (n : ℕ) : ℕ := n * (n - 1)

/-- There are 6 people to choose from -/
def number_of_people : ℕ := 6

theorem president_secretary_selection :
  permutation_two_roles number_of_people = 30 := by
  sorry

end NUMINAMATH_CALUDE_president_secretary_selection_l3800_380073


namespace NUMINAMATH_CALUDE_no_real_solutions_l3800_380081

theorem no_real_solutions (n : ℝ) : 
  (∀ x : ℝ, (x + 6) * (x - 3) ≠ n + 4 * x) ↔ n < -73/4 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3800_380081


namespace NUMINAMATH_CALUDE_arun_weight_average_l3800_380068

def arun_weight_range (w : ℝ) : Prop :=
  62 < w ∧ w < 72 ∧ 60 < w ∧ w < 70 ∧ w ≤ 65

theorem arun_weight_average :
  ∃ (min max : ℝ),
    (∀ w, arun_weight_range w → min ≤ w ∧ w ≤ max) ∧
    (∃ w₁ w₂, arun_weight_range w₁ ∧ arun_weight_range w₂ ∧ w₁ = min ∧ w₂ = max) ∧
    (min + max) / 2 = 63.5 :=
sorry

end NUMINAMATH_CALUDE_arun_weight_average_l3800_380068


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l3800_380072

theorem hot_dogs_remainder : 35252983 % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l3800_380072


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l3800_380026

theorem floor_ceiling_sum : ⌊(1.999 : ℝ)⌋ + ⌈(3.001 : ℝ)⌉ = 5 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l3800_380026


namespace NUMINAMATH_CALUDE_equation_solution_l3800_380044

theorem equation_solution : ∃ x : ℚ, (x - 3) / 2 - (2 * x) / 3 = 1 ∧ x = -15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3800_380044


namespace NUMINAMATH_CALUDE_fundraiser_total_amount_l3800_380034

/-- The total amount promised in a fundraiser -/
theorem fundraiser_total_amount (received : ℕ) (sally_owed : ℕ) (carl_owed : ℕ) (amy_owed : ℕ) :
  received = 285 →
  sally_owed = 35 →
  carl_owed = 35 →
  amy_owed = 30 →
  received + sally_owed + carl_owed + amy_owed + amy_owed / 2 = 400 := by
  sorry

#check fundraiser_total_amount

end NUMINAMATH_CALUDE_fundraiser_total_amount_l3800_380034


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_k_l3800_380091

/-- A polynomial is a perfect square trinomial if it can be expressed as (ax + b)^2 -/
def IsPerfectSquareTrinomial (p : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, p x = (a * x + b)^2

/-- The given polynomial -/
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - 8*x + k

theorem perfect_square_trinomial_k (k : ℝ) :
  IsPerfectSquareTrinomial (f k) → k = 16 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_k_l3800_380091


namespace NUMINAMATH_CALUDE_shopping_visit_problem_l3800_380052

theorem shopping_visit_problem (
  num_stores : ℕ
  ) (total_visits : ℕ)
  (two_store_visitors : ℕ)
  (h1 : num_stores = 8)
  (h2 : total_visits = 21)
  (h3 : two_store_visitors = 8)
  (h4 : two_store_visitors * 2 ≤ total_visits) :
  ∃ (max_stores_visited : ℕ) (total_shoppers : ℕ),
    max_stores_visited = 5 ∧
    total_shoppers = 9 ∧
    max_stores_visited ≤ num_stores ∧
    total_shoppers * 1 ≤ total_visits ∧
    total_shoppers ≥ two_store_visitors + 1 :=
by sorry

end NUMINAMATH_CALUDE_shopping_visit_problem_l3800_380052


namespace NUMINAMATH_CALUDE_wendy_polished_glasses_l3800_380004

def small_glasses : ℕ := 50
def large_glasses : ℕ := small_glasses + 10

theorem wendy_polished_glasses : small_glasses + large_glasses = 110 := by
  sorry

end NUMINAMATH_CALUDE_wendy_polished_glasses_l3800_380004


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3800_380048

/-- The polynomial p(x) = x^4 - x^3 - 4x + 7 -/
def p (x : ℝ) : ℝ := x^4 - x^3 - 4*x + 7

/-- The remainder when p(x) is divided by (x - 3) -/
def remainder : ℝ := p 3

theorem polynomial_remainder : remainder = 49 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3800_380048


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3800_380075

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (m : ℝ), m = b / a ∧ m = Real.sqrt 3) →
  (∃ (d : ℝ), d = 2 * Real.sqrt 3 ∧ d = b) →
  (∀ (x y : ℝ), x^2 / 4 - y^2 / 12 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3800_380075


namespace NUMINAMATH_CALUDE_sum_formula_l3800_380042

/-- Given a sequence {a_n}, S_n is the sum of the first n terms and satisfies S_n = 2a_n - 2^n -/
def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  2 * a n - 2^n

/-- Theorem stating that S_n = n * 2^n -/
theorem sum_formula (a : ℕ → ℝ) (n : ℕ) : S a n = n * 2^n := by
  sorry

end NUMINAMATH_CALUDE_sum_formula_l3800_380042


namespace NUMINAMATH_CALUDE_table_capacity_l3800_380062

theorem table_capacity (invited : ℕ) (no_show : ℕ) (tables : ℕ) : 
  invited = 68 → no_show = 50 → tables = 6 → 
  (invited - no_show) / tables = 3 := by
  sorry

end NUMINAMATH_CALUDE_table_capacity_l3800_380062


namespace NUMINAMATH_CALUDE_amp_composition_l3800_380047

def amp (x : ℤ) : ℤ := 9 - x
def amp_bar (x : ℤ) : ℤ := x - 9

theorem amp_composition : amp (amp_bar 15) = 15 := by
  sorry

end NUMINAMATH_CALUDE_amp_composition_l3800_380047


namespace NUMINAMATH_CALUDE_endpoint_sum_endpoint_sum_proof_l3800_380070

/-- Given a line segment with one endpoint (6, 1) and midpoint (5, 7),
    the sum of the coordinates of the other endpoint is 17. -/
theorem endpoint_sum : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → Prop :=
  fun x1 y1 mx my x2 y2 =>
    x1 = 6 ∧ y1 = 1 ∧ mx = 5 ∧ my = 7 ∧
    (x1 + x2) / 2 = mx ∧ (y1 + y2) / 2 = my →
    x2 + y2 = 17

theorem endpoint_sum_proof : endpoint_sum 6 1 5 7 4 13 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_sum_endpoint_sum_proof_l3800_380070


namespace NUMINAMATH_CALUDE_point_p_final_position_l3800_380083

def final_position (initial : ℤ) (right_move : ℤ) (left_move : ℤ) : ℤ :=
  initial + right_move - left_move

theorem point_p_final_position :
  final_position (-2) 5 4 = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_p_final_position_l3800_380083


namespace NUMINAMATH_CALUDE_joggers_problem_l3800_380030

theorem joggers_problem (tyson alexander christopher : ℕ) : 
  alexander = tyson + 22 →
  christopher = 20 * tyson →
  christopher = alexander + 54 →
  christopher = 80 :=
by sorry

end NUMINAMATH_CALUDE_joggers_problem_l3800_380030


namespace NUMINAMATH_CALUDE_carol_invitation_packs_l3800_380093

theorem carol_invitation_packs (invitations_per_pack : ℕ) (total_invitations : ℕ) (h1 : invitations_per_pack = 9) (h2 : total_invitations = 45) :
  total_invitations / invitations_per_pack = 5 :=
by sorry

end NUMINAMATH_CALUDE_carol_invitation_packs_l3800_380093


namespace NUMINAMATH_CALUDE_tangent_product_equality_l3800_380065

theorem tangent_product_equality : 
  Real.tan (55 * π / 180) * Real.tan (65 * π / 180) * Real.tan (75 * π / 180) = Real.tan (85 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_equality_l3800_380065


namespace NUMINAMATH_CALUDE_greatest_consecutive_integers_sum_120_l3800_380056

theorem greatest_consecutive_integers_sum_120 :
  (∀ n : ℕ, n > 15 → ¬∃ a : ℕ, (Finset.range n).sum (λ i => a + i) = 120) ∧
  ∃ a : ℕ, (Finset.range 15).sum (λ i => a + i) = 120 :=
by sorry

end NUMINAMATH_CALUDE_greatest_consecutive_integers_sum_120_l3800_380056


namespace NUMINAMATH_CALUDE_rowing_time_with_current_l3800_380043

/-- The time to cover a distance with, against, and without current -/
structure RowingTimes where
  with_current : ℚ
  against_current : ℚ
  no_current : ℚ

/-- The conditions of the rowing problem -/
def rowing_conditions (t : RowingTimes) : Prop :=
  t.against_current = 60 / 7 ∧ t.no_current = t.with_current - 7

/-- The theorem stating the time to cover the distance with the current -/
theorem rowing_time_with_current (t : RowingTimes) :
  rowing_conditions t → t.with_current = 60 / 7 := by
  sorry

end NUMINAMATH_CALUDE_rowing_time_with_current_l3800_380043


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l3800_380082

theorem geometric_sequence_solution (x : ℝ) : 
  (1 : ℝ) < x ∧ x < 9 ∧ x^2 = 9 → x = 3 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_solution_l3800_380082


namespace NUMINAMATH_CALUDE_short_video_length_proof_l3800_380088

/-- Represents the length of short videos in minutes -/
def short_video_length : ℝ := 2

/-- Represents the number of videos released per day -/
def videos_per_day : ℕ := 3

/-- Represents the length multiplier for the long video -/
def long_video_multiplier : ℕ := 6

/-- Represents the number of days in a week -/
def days_per_week : ℕ := 7

/-- Represents the total video length per week in minutes -/
def total_weekly_length : ℝ := 112

theorem short_video_length_proof :
  short_video_length * (videos_per_day - 1 + long_video_multiplier) * days_per_week = total_weekly_length :=
by sorry

end NUMINAMATH_CALUDE_short_video_length_proof_l3800_380088


namespace NUMINAMATH_CALUDE_three_statements_true_l3800_380025

/-- A sequence a, b, c is geometric if b/a = c/b when a and b are non-zero -/
def IsGeometricSequence (a b c : ℝ) : Prop :=
  (a ≠ 0 ∧ b ≠ 0) → (b / a = c / b)

/-- The four statements about geometric sequences and their square relationship -/
def Statements (a b c : ℝ) : Fin 4 → Prop
  | 0 => IsGeometricSequence a b c → b^2 = a*c
  | 1 => b^2 = a*c → IsGeometricSequence a b c
  | 2 => ¬(IsGeometricSequence a b c) → b^2 ≠ a*c
  | 3 => b^2 ≠ a*c → ¬(IsGeometricSequence a b c)

/-- The theorem stating that exactly 3 of the 4 statements are true -/
theorem three_statements_true : 
  ∃ (correct : Finset (Fin 4)), correct.card = 3 ∧ 
    (∀ i : Fin 4, i ∈ correct ↔ ∀ a b c : ℝ, Statements a b c i) :=
sorry

end NUMINAMATH_CALUDE_three_statements_true_l3800_380025


namespace NUMINAMATH_CALUDE_fourth_place_votes_l3800_380005

theorem fourth_place_votes (total_votes : ℕ) (winner_margin1 winner_margin2 winner_margin3 : ℕ) :
  total_votes = 979 →
  winner_margin1 = 53 →
  winner_margin2 = 79 →
  winner_margin3 = 105 →
  ∃ (winner_votes fourth_place_votes : ℕ),
    winner_votes - winner_margin1 + winner_votes - winner_margin2 + winner_votes - winner_margin3 + fourth_place_votes = total_votes ∧
    fourth_place_votes = 199 :=
by sorry

end NUMINAMATH_CALUDE_fourth_place_votes_l3800_380005


namespace NUMINAMATH_CALUDE_ratio_problem_l3800_380064

theorem ratio_problem (a b : ℝ) (h1 : a / b = 5) (h2 : a = 40) : b = 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3800_380064


namespace NUMINAMATH_CALUDE_talent_school_problem_l3800_380006

theorem talent_school_problem (total : ℕ) (cant_sing cant_dance cant_act : ℕ) :
  total = 150 ∧ 
  cant_sing = 90 ∧ 
  cant_dance = 100 ∧ 
  cant_act = 60 →
  ∃ (two_talents : ℕ),
    two_talents = 50 ∧
    two_talents = total - (total - cant_sing) - (total - cant_dance) - (total - cant_act) + 2 * total - cant_sing - cant_dance - cant_act :=
by sorry

end NUMINAMATH_CALUDE_talent_school_problem_l3800_380006


namespace NUMINAMATH_CALUDE_opposite_sign_quadratic_solution_l3800_380031

theorem opposite_sign_quadratic_solution :
  ∀ m n : ℝ,
  (|2*m + n| + Real.sqrt (3*n + 12) = 0) →
  (m = 2 ∧ n = -4) ∧
  (∀ x : ℝ, m*x^2 + 4*n*x - 2 = 0 ↔ x = 4 + Real.sqrt 17 ∨ x = 4 - Real.sqrt 17) :=
by sorry

end NUMINAMATH_CALUDE_opposite_sign_quadratic_solution_l3800_380031


namespace NUMINAMATH_CALUDE_row_length_theorem_l3800_380035

/-- The length of a row of boys standing with 1 meter between adjacent boys -/
def row_length (n : ℕ) : ℕ := n - 1

/-- Theorem: For n boys standing in a row with 1 meter between adjacent boys,
    the length of the row in meters is equal to n - 1 -/
theorem row_length_theorem (n : ℕ) (h : n > 0) : row_length n = n - 1 := by
  sorry

end NUMINAMATH_CALUDE_row_length_theorem_l3800_380035


namespace NUMINAMATH_CALUDE_initial_books_count_l3800_380023

theorem initial_books_count (initial_books additional_books total_books : ℕ) 
  (h1 : additional_books = 23)
  (h2 : total_books = 77)
  (h3 : initial_books + additional_books = total_books) :
  initial_books = 54 := by
  sorry

end NUMINAMATH_CALUDE_initial_books_count_l3800_380023


namespace NUMINAMATH_CALUDE_parabola_shift_l3800_380087

/-- A parabola shifted 1 unit to the left -/
def shifted_parabola (x : ℝ) : ℝ := (x + 1)^2

/-- The original parabola -/
def original_parabola (x : ℝ) : ℝ := x^2

theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_l3800_380087


namespace NUMINAMATH_CALUDE_student_count_last_year_l3800_380013

theorem student_count_last_year 
  (increase_rate : ℝ) 
  (current_count : ℕ) 
  (h1 : increase_rate = 0.2) 
  (h2 : current_count = 960) : 
  ℕ :=
  by
    -- Proof goes here
    sorry

#check student_count_last_year

end NUMINAMATH_CALUDE_student_count_last_year_l3800_380013


namespace NUMINAMATH_CALUDE_find_z_value_l3800_380089

theorem find_z_value (z : ℝ) : (12^3 * z^3) / 432 = 864 → z = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_z_value_l3800_380089


namespace NUMINAMATH_CALUDE_orange_juice_fraction_l3800_380033

/-- Represents the capacity of each pitcher in milliliters -/
def pitcher_capacity : ℕ := 800

/-- Represents the fraction of orange juice in the first pitcher -/
def first_pitcher_fraction : ℚ := 1/2

/-- Represents the fraction of orange juice in the second pitcher -/
def second_pitcher_fraction : ℚ := 1/4

/-- Calculates the total volume of orange juice in both pitchers -/
def total_orange_juice : ℚ := 
  pitcher_capacity * first_pitcher_fraction + pitcher_capacity * second_pitcher_fraction

/-- Calculates the total volume of the mixture after filling both pitchers completely -/
def total_mixture : ℕ := 2 * pitcher_capacity

/-- Theorem stating that the fraction of orange juice in the final mixture is 3/8 -/
theorem orange_juice_fraction : 
  (total_orange_juice : ℚ) / (total_mixture : ℚ) = 3/8 := by sorry

end NUMINAMATH_CALUDE_orange_juice_fraction_l3800_380033


namespace NUMINAMATH_CALUDE_count_is_2530_l3800_380000

/-- Sum of digits function -/
def s (n : ℕ) : ℕ := sorry

/-- The count of positive integers n ≤ 10^4 satisfying s(11n) = 2s(n) -/
def count : ℕ := sorry

/-- Theorem stating the count is 2530 -/
theorem count_is_2530 : count = 2530 := by sorry

end NUMINAMATH_CALUDE_count_is_2530_l3800_380000


namespace NUMINAMATH_CALUDE_firm_ratio_proof_l3800_380002

/-- Represents the number of partners in the firm -/
def partners : ℕ := 20

/-- Represents the additional associates to be hired -/
def additional_associates : ℕ := 50

/-- Represents the ratio of partners to associates after hiring additional associates -/
def new_ratio : ℚ := 1 / 34

/-- Calculates the initial number of associates in the firm -/
def initial_associates : ℕ := partners * 34 - additional_associates

/-- Represents the initial ratio of partners to associates -/
def initial_ratio : ℚ := partners / initial_associates

theorem firm_ratio_proof :
  initial_ratio = 2 / 63 := by
  sorry

end NUMINAMATH_CALUDE_firm_ratio_proof_l3800_380002


namespace NUMINAMATH_CALUDE_ben_cards_l3800_380016

theorem ben_cards (B : ℕ) (tim_cards : ℕ) : 
  tim_cards = 20 → B + 3 = 2 * tim_cards → B = 37 := by sorry

end NUMINAMATH_CALUDE_ben_cards_l3800_380016


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l3800_380055

/-- A cubic polynomial that satisfies specific conditions -/
def q (x : ℚ) : ℚ := (15/8) * x^3 + (5/4) * x^2 - (13/8) * x + 3

/-- Theorem stating that q satisfies the given conditions -/
theorem q_satisfies_conditions :
  q 0 = 3 ∧ q 1 = 5 ∧ q 2 = 13 ∧ q 3 = 41 := by
  sorry

#eval q 0
#eval q 1
#eval q 2
#eval q 3

end NUMINAMATH_CALUDE_q_satisfies_conditions_l3800_380055


namespace NUMINAMATH_CALUDE_power_of_two_plus_one_square_or_cube_l3800_380022

theorem power_of_two_plus_one_square_or_cube (n : ℕ) :
  (∃ m : ℕ, 2^n + 1 = m^2) ∨ (∃ m : ℕ, 2^n + 1 = m^3) ↔ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_plus_one_square_or_cube_l3800_380022


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3800_380060

theorem polynomial_expansion (z : ℝ) :
  (3 * z^3 + 4 * z^2 - 7 * z + 1) * (2 * z^4 - 3 * z^2 + 2) =
  6 * z^7 + 8 * z^6 - 23 * z^5 - 10 * z^4 + 27 * z^3 + 5 * z^2 - 14 * z + 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3800_380060


namespace NUMINAMATH_CALUDE_equation_solution_l3800_380063

theorem equation_solution : ∃ x : ℝ, (3 / x - 2 / (x + 1) = 0) ∧ (x = -3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3800_380063


namespace NUMINAMATH_CALUDE_average_of_seventeen_numbers_l3800_380029

-- Define the problem parameters
def total_count : ℕ := 17
def first_nine_avg : ℚ := 56
def last_nine_avg : ℚ := 63
def ninth_number : ℚ := 68

-- Theorem statement
theorem average_of_seventeen_numbers :
  let first_nine_sum := 9 * first_nine_avg
  let last_nine_sum := 9 * last_nine_avg
  let total_sum := first_nine_sum + last_nine_sum - ninth_number
  total_sum / total_count = 59 := by
sorry

end NUMINAMATH_CALUDE_average_of_seventeen_numbers_l3800_380029


namespace NUMINAMATH_CALUDE_smallest_integer_in_ratio_l3800_380019

theorem smallest_integer_in_ratio (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 72 →
  b = 3 * a →
  c = 4 * a →
  a = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_in_ratio_l3800_380019


namespace NUMINAMATH_CALUDE_toy_purchase_cost_l3800_380038

theorem toy_purchase_cost (num_toys : ℕ) (cost_per_toy : ℝ) (discount_percent : ℝ) :
  num_toys = 5 →
  cost_per_toy = 3 →
  discount_percent = 20 →
  (num_toys : ℝ) * cost_per_toy * (1 - discount_percent / 100) = 12 := by
  sorry

end NUMINAMATH_CALUDE_toy_purchase_cost_l3800_380038


namespace NUMINAMATH_CALUDE_f_properties_l3800_380049

-- Define the function f(x) = x^3 + ax^2 + 1
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 1

-- State the theorem
theorem f_properties (a : ℝ) (h : a > 0) :
  -- f(x) has exactly two critical points
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, deriv (f a) x = 0 ↔ x = x₁ ∨ x = x₂)) ∧
  -- The point (-a/3, f(-a/3)) is the center of symmetry
  (∀ x : ℝ, f a (-a/3 + x) = f a (-a/3 - x)) ∧
  -- There exists a point where y = x is tangent to y = f(x)
  (∃ x₀ : ℝ, deriv (f a) x₀ = 1 ∧ f a x₀ = x₀) :=
by sorry


end NUMINAMATH_CALUDE_f_properties_l3800_380049


namespace NUMINAMATH_CALUDE_binary_to_octal_conversion_l3800_380001

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldr (fun bit acc => 2 * acc + if bit then 1 else 0) 0

def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

def binary_101101011 : List Bool :=
  [true, false, true, true, false, true, false, true, true]

theorem binary_to_octal_conversion :
  decimal_to_octal (binary_to_decimal binary_101101011) = [3, 2, 3, 1] := by
  sorry

end NUMINAMATH_CALUDE_binary_to_octal_conversion_l3800_380001


namespace NUMINAMATH_CALUDE_initial_amount_theorem_l3800_380028

/-- The initial amount of money given the lending conditions --/
theorem initial_amount_theorem (amount_to_B : ℝ) 
  (h1 : amount_to_B = 4000.0000000000005)
  (h2 : ∃ amount_to_A : ℝ, 
    amount_to_A * 0.15 * 2 = amount_to_B * 0.18 * 2 + 360) :
  ∃ initial_amount : ℝ, initial_amount = 10000.000000000002 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_theorem_l3800_380028


namespace NUMINAMATH_CALUDE_sandy_walk_l3800_380036

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a direction -/
inductive Direction
  | North
  | South
  | East
  | West

/-- Moves a point in a given direction by a specified distance -/
def move (p : Point) (dir : Direction) (distance : ℝ) : Point :=
  match dir with
  | Direction.North => { x := p.x, y := p.y + distance }
  | Direction.South => { x := p.x, y := p.y - distance }
  | Direction.East => { x := p.x + distance, y := p.y }
  | Direction.West => { x := p.x - distance, y := p.y }

/-- Sandy's walk -/
theorem sandy_walk (start : Point) : 
  let p1 := move start Direction.South 20
  let p2 := move p1 Direction.East 20
  let p3 := move p2 Direction.North 20
  let final := move p3 Direction.East 20
  final.x = start.x + 40 ∧ final.y = start.y :=
by
  sorry

#check sandy_walk

end NUMINAMATH_CALUDE_sandy_walk_l3800_380036


namespace NUMINAMATH_CALUDE_fish_sales_hours_l3800_380041

/-- The number of hours fish are sold for, given peak and low season sales rates,
    price per pack, and daily revenue difference between seasons. -/
theorem fish_sales_hours 
  (peak_rate : ℕ) 
  (low_rate : ℕ) 
  (price_per_pack : ℕ) 
  (daily_revenue_diff : ℕ) 
  (h_peak_rate : peak_rate = 6)
  (h_low_rate : low_rate = 4)
  (h_price : price_per_pack = 60)
  (h_revenue_diff : daily_revenue_diff = 1800) :
  (peak_rate - low_rate) * price_per_pack * h = daily_revenue_diff → h = 15 :=
by sorry

end NUMINAMATH_CALUDE_fish_sales_hours_l3800_380041


namespace NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l3800_380057

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l3800_380057


namespace NUMINAMATH_CALUDE_det_of_matrix_l3800_380012

def matrix : Matrix (Fin 2) (Fin 2) ℤ := !![7, -2; -3, 5]

theorem det_of_matrix : Matrix.det matrix = 29 := by
  sorry

end NUMINAMATH_CALUDE_det_of_matrix_l3800_380012


namespace NUMINAMATH_CALUDE_ones_digit_of_35_power_ones_digit_of_35_large_power_l3800_380018

theorem ones_digit_of_35_power (n : ℕ) : n > 0 → (35^n) % 10 = 5 := by sorry

theorem ones_digit_of_35_large_power : (35^(35*(17^17))) % 10 = 5 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_35_power_ones_digit_of_35_large_power_l3800_380018


namespace NUMINAMATH_CALUDE_alice_book_payment_percentage_l3800_380040

/-- The percentage of the suggested retail price that Alice paid for a book -/
theorem alice_book_payment_percentage 
  (suggested_retail_price : ℝ)
  (marked_price : ℝ)
  (alice_paid : ℝ)
  (h1 : marked_price = 0.6 * suggested_retail_price)
  (h2 : alice_paid = 0.4 * marked_price) :
  alice_paid / suggested_retail_price = 0.24 := by
sorry

end NUMINAMATH_CALUDE_alice_book_payment_percentage_l3800_380040


namespace NUMINAMATH_CALUDE_complement_of_union_l3800_380098

def U : Finset Nat := {1,2,3,4,5,6,7,8}
def M : Finset Nat := {1,3,5,7}
def N : Finset Nat := {5,6,7}

theorem complement_of_union :
  (U \ (M ∪ N)) = {2,4,8} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l3800_380098


namespace NUMINAMATH_CALUDE_tan_double_angle_special_case_l3800_380085

theorem tan_double_angle_special_case (α : ℝ) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.cos α + Real.sin α = -1/5) : Real.tan (2 * α) = -24/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_special_case_l3800_380085
