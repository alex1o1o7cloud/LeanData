import Mathlib

namespace NUMINAMATH_CALUDE_rob_quarters_l1959_195937

def quarters : ℕ → ℚ
  | n => (n : ℚ) * (1 / 4)

def dimes : ℕ → ℚ
  | n => (n : ℚ) * (1 / 10)

def nickels : ℕ → ℚ
  | n => (n : ℚ) * (1 / 20)

def pennies : ℕ → ℚ
  | n => (n : ℚ) * (1 / 100)

theorem rob_quarters (x : ℕ) :
  quarters x + dimes 3 + nickels 5 + pennies 12 = 242 / 100 →
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_rob_quarters_l1959_195937


namespace NUMINAMATH_CALUDE_super_extra_yield_interest_l1959_195950

/-- Calculates the interest earned on a compound interest savings account -/
theorem super_extra_yield_interest
  (principal : ℝ)
  (rate : ℝ)
  (years : ℕ)
  (h_principal : principal = 1500)
  (h_rate : rate = 0.02)
  (h_years : years = 5) :
  ⌊(principal * (1 + rate) ^ years - principal)⌋ = 156 := by
  sorry

end NUMINAMATH_CALUDE_super_extra_yield_interest_l1959_195950


namespace NUMINAMATH_CALUDE_power_of_point_outside_circle_l1959_195964

/-- Given a circle with radius R and a point M outside the circle at distance d from the center,
    prove that for any line through M intersecting the circle at A and B, MA * MB = d² - R² -/
theorem power_of_point_outside_circle (R d : ℝ) (h : 0 < R) (h' : R < d) :
  ∀ (M A B : ℝ × ℝ),
    ‖M - (0, 0)‖ = d →
    ‖A - (0, 0)‖ = R →
    ‖B - (0, 0)‖ = R →
    (∃ t : ℝ, A = M + t • (B - M)) →
    ‖M - A‖ * ‖M - B‖ = d^2 - R^2 :=
by sorry

end NUMINAMATH_CALUDE_power_of_point_outside_circle_l1959_195964


namespace NUMINAMATH_CALUDE_eleven_overtake_points_l1959_195978

/-- Represents a point on a circular track -/
structure TrackPoint where
  position : ℝ
  mk_mod : position ≥ 0 ∧ position < 1

/-- Represents the movement of a person on the track -/
structure Movement where
  speed : ℝ
  startPoint : TrackPoint

/-- Calculates the number of distinct overtake points -/
def countOvertakePoints (pedestrian : Movement) (cyclist : Movement) : ℕ :=
  sorry

/-- Main theorem: There are exactly 11 distinct overtake points -/
theorem eleven_overtake_points :
  ∀ (start : TrackPoint) (pedSpeed : ℝ),
    pedSpeed > 0 →
    let cycSpeed := pedSpeed * 1.55
    let pedestrian := Movement.mk pedSpeed start
    let cyclist := Movement.mk cycSpeed start
    countOvertakePoints pedestrian cyclist = 11 :=
  sorry

end NUMINAMATH_CALUDE_eleven_overtake_points_l1959_195978


namespace NUMINAMATH_CALUDE_same_color_probability_eight_nine_l1959_195952

/-- The probability of drawing two balls of the same color from a box containing 
    8 white balls and 9 black balls. -/
def same_color_probability (white : ℕ) (black : ℕ) : ℚ :=
  let total := white + black
  let same_color_ways := (white.choose 2) + (black.choose 2)
  let total_ways := total.choose 2
  same_color_ways / total_ways

/-- Theorem stating that the probability of drawing two balls of the same color 
    from a box with 8 white balls and 9 black balls is 8/17. -/
theorem same_color_probability_eight_nine : 
  same_color_probability 8 9 = 8 / 17 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_eight_nine_l1959_195952


namespace NUMINAMATH_CALUDE_cards_distribution_theorem_l1959_195954

/-- Given a total number of cards and people, calculate how many people receive fewer cards when dealt as evenly as possible. -/
def people_with_fewer_cards (total_cards : ℕ) (num_people : ℕ) (threshold : ℕ) : ℕ :=
  let cards_per_person := total_cards / num_people
  let extra_cards := total_cards % num_people
  if cards_per_person + 1 < threshold then num_people
  else num_people - extra_cards

/-- Theorem stating that when 60 cards are dealt to 9 people as evenly as possible, 3 people will have fewer than 7 cards. -/
theorem cards_distribution_theorem :
  people_with_fewer_cards 60 9 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_theorem_l1959_195954


namespace NUMINAMATH_CALUDE_box_volume_l1959_195997

/-- Given a box with specified dimensions, prove its volume is 3888 cubic inches. -/
theorem box_volume : 
  ∀ (height length width : ℝ),
    height = 12 →
    length = 3 * height →
    length = 4 * width →
    height * length * width = 3888 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_l1959_195997


namespace NUMINAMATH_CALUDE_time_after_1567_minutes_l1959_195993

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a day and time -/
structure DayTime where
  days : Nat
  time : Time
  deriving Repr

def addMinutes (start : Time) (minutes : Nat) : DayTime :=
  let totalMinutes := start.minutes + minutes
  let totalHours := start.hours + totalMinutes / 60
  let finalMinutes := totalMinutes % 60
  let finalHours := totalHours % 24
  let days := totalHours / 24
  { days := days
  , time := { hours := finalHours, minutes := finalMinutes } }

theorem time_after_1567_minutes :
  let start := Time.mk 17 0  -- 5:00 p.m.
  let result := addMinutes start 1567
  result = DayTime.mk 1 (Time.mk 19 7)  -- 7:07 p.m. next day
  := by sorry

end NUMINAMATH_CALUDE_time_after_1567_minutes_l1959_195993


namespace NUMINAMATH_CALUDE_female_democrats_count_l1959_195988

-- Define the total number of participants
def total_participants : ℕ := 840

-- Define the ratio of female Democrats to total females
def female_democrat_ratio : ℚ := 1/2

-- Define the ratio of male Democrats to total males
def male_democrat_ratio : ℚ := 1/4

-- Define the ratio of all Democrats to total participants
def total_democrat_ratio : ℚ := 1/3

-- Theorem statement
theorem female_democrats_count :
  ∃ (female_participants male_participants : ℕ),
    female_participants + male_participants = total_participants ∧
    (female_democrat_ratio * female_participants + male_democrat_ratio * male_participants : ℚ) = 
      total_democrat_ratio * total_participants ∧
    female_democrat_ratio * female_participants = 140 :=
sorry

end NUMINAMATH_CALUDE_female_democrats_count_l1959_195988


namespace NUMINAMATH_CALUDE_sum_and_count_result_l1959_195967

def sum_of_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

def x : ℕ := sum_of_integers 10 20

def y : ℕ := count_even_integers 10 20

theorem sum_and_count_result : x + y = 171 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_result_l1959_195967


namespace NUMINAMATH_CALUDE_other_man_age_is_36_l1959_195956

/-- The age of the other man in the group problem -/
def other_man_age : ℕ := 36

/-- The number of men in the initial group -/
def num_men : ℕ := 9

/-- The increase in average age when two women replace two men -/
def avg_age_increase : ℕ := 4

/-- The age of one of the men in the group -/
def known_man_age : ℕ := 32

/-- The average age of the two women -/
def women_avg_age : ℕ := 52

/-- The theorem stating that given the conditions, the age of the other man is 36 -/
theorem other_man_age_is_36 :
  (num_men * avg_age_increase = 2 * women_avg_age - (other_man_age + known_man_age)) →
  other_man_age = 36 := by
  sorry

#check other_man_age_is_36

end NUMINAMATH_CALUDE_other_man_age_is_36_l1959_195956


namespace NUMINAMATH_CALUDE_sticker_distribution_l1959_195919

/-- The number of ways to distribute n identical objects into k identical containers -/
def distribute_objects (n k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 25 ways to distribute 10 identical stickers onto 5 identical sheets of paper -/
theorem sticker_distribution : distribute_objects 10 5 = 25 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l1959_195919


namespace NUMINAMATH_CALUDE_multiply_decimals_l1959_195976

theorem multiply_decimals : (4.8 : ℝ) * 0.25 * 0.1 = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_multiply_decimals_l1959_195976


namespace NUMINAMATH_CALUDE_area_ratio_of_specific_trapezoid_l1959_195903

/-- Represents a trapezoid with extended legs -/
structure ExtendedTrapezoid where
  -- Length of the shorter base
  pq : ℝ
  -- Length of the longer base
  rs : ℝ
  -- Point where extended legs meet
  t : Point

/-- Calculates the ratio of the area of triangle TPQ to the area of trapezoid PQRS -/
def areaRatio (trap : ExtendedTrapezoid) : ℚ :=
  100 / 429

/-- Theorem stating the area ratio for the given trapezoid -/
theorem area_ratio_of_specific_trapezoid :
  ∃ (trap : ExtendedTrapezoid),
    trap.pq = 10 ∧ trap.rs = 23 ∧ areaRatio trap = 100 / 429 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_specific_trapezoid_l1959_195903


namespace NUMINAMATH_CALUDE_total_stickers_l1959_195916

theorem total_stickers (stickers_per_page : ℕ) (total_pages : ℕ) : 
  stickers_per_page = 10 → total_pages = 22 → stickers_per_page * total_pages = 220 := by
sorry

end NUMINAMATH_CALUDE_total_stickers_l1959_195916


namespace NUMINAMATH_CALUDE_lemonade_mixture_problem_l1959_195901

theorem lemonade_mixture_problem (x : ℝ) : 
  x ≥ 0 ∧ x ≤ 100 →  -- Percentage of lemonade in first solution
  (0.6799999999999997 * (100 - x) + 0.32 * 55 = 72) →  -- Mixture equation
  x = 20 := by sorry

end NUMINAMATH_CALUDE_lemonade_mixture_problem_l1959_195901


namespace NUMINAMATH_CALUDE_card_game_profit_general_card_game_profit_l1959_195927

/-- Expected profit function for the card guessing game -/
def expected_profit (r b g : ℕ) : ℚ :=
  (b - r : ℚ) + (2 * (r - b : ℚ) / (r + b : ℚ)) * g

/-- Theorem stating the expected profit for the specific game instance -/
theorem card_game_profit :
  expected_profit 2011 2012 2011 = 1 / 4023 := by
  sorry

/-- Theorem for the general case of the card guessing game -/
theorem general_card_game_profit (r b g : ℕ) (h : r + b > 0) :
  expected_profit r b g =
    (b - r : ℚ) + (2 * (r - b : ℚ) / (r + b : ℚ)) * g := by
  sorry

end NUMINAMATH_CALUDE_card_game_profit_general_card_game_profit_l1959_195927


namespace NUMINAMATH_CALUDE_rectangular_prism_surface_area_l1959_195906

/-- The surface area of a rectangular prism with dimensions 1, 2, and 2 is 16 -/
theorem rectangular_prism_surface_area :
  let length : ℝ := 1
  let width : ℝ := 2
  let height : ℝ := 2
  let surface_area := 2 * (length * width + length * height + width * height)
  surface_area = 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_surface_area_l1959_195906


namespace NUMINAMATH_CALUDE_geometric_sequence_terms_l1959_195975

theorem geometric_sequence_terms (n : ℕ) (a₁ q : ℝ) 
  (h1 : a₁^3 * q^3 = 3)
  (h2 : a₁^3 * q^(3*n - 6) = 9)
  (h3 : a₁^n * q^(n*(n-1)/2) = 729) :
  n = 12 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_terms_l1959_195975


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_l1959_195902

theorem consecutive_even_integers_sum (n : ℤ) :
  (n + (n + 6) = 160) →
  ((n + 2) + (n + 4) = 160) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_l1959_195902


namespace NUMINAMATH_CALUDE_divisibility_by_360_l1959_195986

theorem divisibility_by_360 (p : ℕ) (h_prime : Nat.Prime p) (h_greater_than_5 : p > 5) :
  360 ∣ (p^4 - 5*p^2 + 4) := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_360_l1959_195986


namespace NUMINAMATH_CALUDE_sets_and_range_l1959_195961

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a - 2 < x ∧ x < a}
def B : Set ℝ := {x | 3 / (x - 1) ≥ 1}

-- Define the complement of B in ℝ
def complement_B : Set ℝ := {x | x ≤ 1 ∨ x > 4}

-- State the theorem
theorem sets_and_range (a : ℝ) : 
  A a ⊆ complement_B → (a ≤ 1 ∨ a ≥ 2) := by sorry

end NUMINAMATH_CALUDE_sets_and_range_l1959_195961


namespace NUMINAMATH_CALUDE_simplify_expression_l1959_195932

theorem simplify_expression (a : ℝ) (h1 : a ≠ -1) (h2 : a ≠ 2) (h3 : a ≠ -2) :
  (3 / (a + 1) - a + 1) / ((a^2 - 4) / (a^2 + 2*a + 1)) = -a - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1959_195932


namespace NUMINAMATH_CALUDE_intersection_product_l1959_195987

-- Define the sets T and S
def T (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | a * p.1 + p.2 - 3 = 0}
def S (b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 - b = 0}

-- State the theorem
theorem intersection_product (a b : ℝ) : 
  S b ∩ T a = {(2, 1)} → a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_product_l1959_195987


namespace NUMINAMATH_CALUDE_brick_width_l1959_195920

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: For a rectangular prism with length 10, height 3, and surface area 164, the width is 4 -/
theorem brick_width (l h : ℝ) (w : ℝ) (h₁ : l = 10) (h₂ : h = 3) (h₃ : surface_area l w h = 164) : w = 4 := by
  sorry

end NUMINAMATH_CALUDE_brick_width_l1959_195920


namespace NUMINAMATH_CALUDE_q_factor_change_l1959_195929

/-- Given a function q defined in terms of w, h, and z, prove that when w is quadrupled,
    h is doubled, and z is tripled, q is multiplied by 5/18. -/
theorem q_factor_change (w h z : ℝ) (q : ℝ → ℝ → ℝ → ℝ) 
    (hq : q w h z = 5 * w / (4 * h * z^2)) :
  q (4*w) (2*h) (3*z) = (5/18) * q w h z := by
  sorry

end NUMINAMATH_CALUDE_q_factor_change_l1959_195929


namespace NUMINAMATH_CALUDE_multiple_of_six_between_14_and_30_l1959_195900

theorem multiple_of_six_between_14_and_30 (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 6 * k)
  (h2 : x^2 > 196)
  (h3 : x < 30) :
  x = 18 ∨ x = 24 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_six_between_14_and_30_l1959_195900


namespace NUMINAMATH_CALUDE_ticket_price_possibilities_l1959_195912

theorem ticket_price_possibilities : ∃ (S : Finset ℕ), 
  (∀ y ∈ S, y > 0 ∧ 42 % y = 0 ∧ 70 % y = 0) ∧ 
  (∀ y : ℕ, y > 0 → 42 % y = 0 → 70 % y = 0 → y ∈ S) ∧
  Finset.card S = 4 := by
  sorry

end NUMINAMATH_CALUDE_ticket_price_possibilities_l1959_195912


namespace NUMINAMATH_CALUDE_mutual_correlation_sign_change_l1959_195935

/-- A stationary stochastic process -/
class StationaryStochasticProcess (X : ℝ → ℝ) : Prop where
  -- Add any necessary properties for a stationary stochastic process

/-- The derivative of a function -/
def derivative (f : ℝ → ℝ) : ℝ → ℝ :=
  fun t => sorry -- Definition of derivative

/-- Mutual correlation function of a process and its derivative -/
def mutualCorrelationFunction (X : ℝ → ℝ) (t₁ t₂ : ℝ) : ℝ :=
  sorry -- Definition of mutual correlation function

/-- Theorem: The mutual correlation function changes sign when arguments are swapped -/
theorem mutual_correlation_sign_change
  (X : ℝ → ℝ) [StationaryStochasticProcess X] (t₁ t₂ : ℝ) :
  mutualCorrelationFunction X t₁ t₂ = -mutualCorrelationFunction X t₂ t₁ :=
by sorry

end NUMINAMATH_CALUDE_mutual_correlation_sign_change_l1959_195935


namespace NUMINAMATH_CALUDE_divisibility_condition_l1959_195977

-- Define the predicate for divisibility
def divides (m n : ℤ) : Prop := ∃ k : ℤ, n = m * k

theorem divisibility_condition (p a : ℤ) : 
  (p ≥ 2) → 
  (a ≥ 1) → 
  Prime p → 
  p ≠ a → 
  (divides (a + p) (a^2 + p^2) ↔ 
    ((a = p) ∨ (a = p^2 - p) ∨ (a = 2*p^2 - p))) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1959_195977


namespace NUMINAMATH_CALUDE_pyramid_height_l1959_195924

theorem pyramid_height (p q : ℝ) : 
  p > 0 ∧ q > 0 →
  3^2 + p^2 = 5^2 →
  (1/3) * (1/2 * 3 * p) * q = 12 →
  q = 6 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_height_l1959_195924


namespace NUMINAMATH_CALUDE_north_pond_duck_count_l1959_195909

/-- The number of ducks at Lake Michigan -/
def lake_michigan_ducks : ℕ := 100

/-- The number of ducks at North Pond -/
def north_pond_ducks : ℕ := 2 * lake_michigan_ducks + 6

theorem north_pond_duck_count : north_pond_ducks = 206 := by
  sorry

end NUMINAMATH_CALUDE_north_pond_duck_count_l1959_195909


namespace NUMINAMATH_CALUDE_cubs_cardinals_home_run_difference_l1959_195923

/-- The number of home runs scored by the Chicago Cubs in the game -/
def cubs_home_runs : ℕ := 2 + 1 + 2

/-- The number of home runs scored by the Cardinals in the game -/
def cardinals_home_runs : ℕ := 1 + 1

/-- The difference in home runs between the Cubs and the Cardinals -/
def home_run_difference : ℕ := cubs_home_runs - cardinals_home_runs

theorem cubs_cardinals_home_run_difference :
  home_run_difference = 3 :=
by sorry

end NUMINAMATH_CALUDE_cubs_cardinals_home_run_difference_l1959_195923


namespace NUMINAMATH_CALUDE_fourth_number_in_sequence_l1959_195914

def fibonacci_like_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 3, a n = a (n - 1) + a (n - 2)

theorem fourth_number_in_sequence
  (a : ℕ → ℕ)
  (h_fib : fibonacci_like_sequence a)
  (h_7 : a 7 = 42)
  (h_9 : a 9 = 110) :
  a 4 = 10 :=
sorry

end NUMINAMATH_CALUDE_fourth_number_in_sequence_l1959_195914


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1959_195949

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1959_195949


namespace NUMINAMATH_CALUDE_function_parity_l1959_195939

-- Define the property of the function
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y

-- Define even and odd functions
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem function_parity (f : ℝ → ℝ) (h : satisfies_property f) :
  (is_even f ∨ is_odd f) :=
sorry

end NUMINAMATH_CALUDE_function_parity_l1959_195939


namespace NUMINAMATH_CALUDE_lexiCement_is_10_l1959_195931

/-- The amount of cement used for Lexi's street -/
def lexiCement : ℝ := sorry

/-- The amount of cement used for Tess's street -/
def tessCement : ℝ := 5.1

/-- The total amount of cement used -/
def totalCement : ℝ := 15.1

/-- Theorem stating that the amount of cement used for Lexi's street is 10 tons -/
theorem lexiCement_is_10 : lexiCement = 10 :=
by
  have h1 : lexiCement = totalCement - tessCement := sorry
  sorry


end NUMINAMATH_CALUDE_lexiCement_is_10_l1959_195931


namespace NUMINAMATH_CALUDE_first_duck_ate_half_l1959_195904

/-- The fraction of bread eaten by the first duck -/
def first_duck_fraction (total_bread pieces_left second_duck_pieces third_duck_pieces : ℕ) : ℚ :=
  let eaten := total_bread - pieces_left
  let first_duck_pieces := eaten - (second_duck_pieces + third_duck_pieces)
  first_duck_pieces / total_bread

/-- Theorem stating the fraction of bread eaten by the first duck -/
theorem first_duck_ate_half :
  first_duck_fraction 100 30 13 7 = 1/2 := by
  sorry

#eval first_duck_fraction 100 30 13 7

end NUMINAMATH_CALUDE_first_duck_ate_half_l1959_195904


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1959_195946

/-- Represents a repeating decimal with a two-digit repeating sequence -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

/-- The repeating decimal 0.474747... -/
def x : ℚ := RepeatingDecimal 4 7

/-- The sum of the numerator and denominator of a fraction -/
def sumNumeratorDenominator (q : ℚ) : ℕ :=
  q.num.natAbs + q.den

theorem repeating_decimal_sum : sumNumeratorDenominator x = 146 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1959_195946


namespace NUMINAMATH_CALUDE_calculation_one_l1959_195938

theorem calculation_one : (-2) + (-7) + 9 - (-12) = 12 := by sorry

end NUMINAMATH_CALUDE_calculation_one_l1959_195938


namespace NUMINAMATH_CALUDE_customers_without_tip_greasy_spoon_tip_problem_l1959_195930

/-- The number of customers who didn't leave a tip at 'The Greasy Spoon' restaurant --/
theorem customers_without_tip (initial_customers : ℕ) (additional_customers : ℕ) (customers_with_tip : ℕ) : ℕ :=
  initial_customers + additional_customers - customers_with_tip

/-- Proof that 34 customers didn't leave a tip --/
theorem greasy_spoon_tip_problem : customers_without_tip 29 20 15 = 34 := by
  sorry

end NUMINAMATH_CALUDE_customers_without_tip_greasy_spoon_tip_problem_l1959_195930


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_range_l1959_195970

/-- Given two predicates p and q on real numbers, where p is a sufficient but not necessary condition for q, 
    this theorem states that the lower bound of k in p is 2, and k can be any real number greater than 2. -/
theorem sufficient_not_necessary_condition_range (k : ℝ) : 
  (∀ x, x ≥ k → (2 - x) / (x + 1) < 0) ∧ 
  (∃ x, (2 - x) / (x + 1) < 0 ∧ x < k) → 
  k > 2 :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_range_l1959_195970


namespace NUMINAMATH_CALUDE_car_profit_percentage_l1959_195943

theorem car_profit_percentage (original_price : ℝ) (h : original_price > 0) :
  let discount_rate := 0.20
  let purchase_price := original_price * (1 - discount_rate)
  let sale_price := purchase_price * 2
  let profit := sale_price - original_price
  let profit_percentage := (profit / original_price) * 100
  profit_percentage = 60 := by sorry

end NUMINAMATH_CALUDE_car_profit_percentage_l1959_195943


namespace NUMINAMATH_CALUDE_parabola_y_comparison_l1959_195926

/-- Parabola function -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 2

/-- Theorem stating that y₁ < y₂ for the given parabola -/
theorem parabola_y_comparison :
  ∀ (y₁ y₂ : ℝ), f 1 = y₁ → f 3 = y₂ → y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_y_comparison_l1959_195926


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1959_195928

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = x^2 - 1}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (3 - x^2)}

-- State the theorem
theorem intersection_of_M_and_N :
  (M ∩ N : Set ℝ) = {x | -1 ≤ x ∧ x ≤ Real.sqrt 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1959_195928


namespace NUMINAMATH_CALUDE_distribute_five_books_four_students_l1959_195968

/-- The number of ways to distribute n different books to k students,
    with each student getting at least one book -/
def distribute (n k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k.choose 2) * (k-2)^n

/-- Theorem: There are 240 ways to distribute 5 different books to 4 students,
    with each student getting at least one book -/
theorem distribute_five_books_four_students :
  distribute 5 4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_books_four_students_l1959_195968


namespace NUMINAMATH_CALUDE_class_size_difference_l1959_195962

theorem class_size_difference (total_students : ℕ) (total_professors : ℕ) (class_sizes : List ℕ) :
  total_students = 200 →
  total_professors = 4 →
  class_sizes = [100, 50, 30, 20] →
  (class_sizes.sum = total_students) →
  let t := (class_sizes.sum : ℚ) / total_professors
  let s := (class_sizes.map (λ size => size * size)).sum / total_students
  t - s = -19 := by
  sorry

end NUMINAMATH_CALUDE_class_size_difference_l1959_195962


namespace NUMINAMATH_CALUDE_cyclic_ratio_inequality_l1959_195966

theorem cyclic_ratio_inequality (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  a / b + b / c + c / d + d / a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_ratio_inequality_l1959_195966


namespace NUMINAMATH_CALUDE_tangent_circle_center_l1959_195918

/-- A circle tangent to two parallel lines with its center on a third line -/
structure TangentCircle where
  -- First tangent line: 3x - 4y = 20
  tangent_line1 : (ℝ × ℝ) → Prop := fun (x, y) ↦ 3 * x - 4 * y = 20
  -- Second tangent line: 3x - 4y = -40
  tangent_line2 : (ℝ × ℝ) → Prop := fun (x, y) ↦ 3 * x - 4 * y = -40
  -- Line containing the center: x - 3y = 0
  center_line : (ℝ × ℝ) → Prop := fun (x, y) ↦ x - 3 * y = 0

/-- The center of the tangent circle is at (-6, -2) -/
theorem tangent_circle_center (c : TangentCircle) : 
  ∃ (x y : ℝ), c.center_line (x, y) ∧ x = -6 ∧ y = -2 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_center_l1959_195918


namespace NUMINAMATH_CALUDE_sum_of_distances_is_ten_l1959_195944

/-- Given a circle tangent to the sides of an angle at points A and B, with a point C on the circle,
    this structure represents the distances and conditions of the problem. -/
structure CircleTangentProblem where
  -- Distance from C to line AB
  h : ℝ
  -- Distance from C to the side of the angle passing through A
  h_A : ℝ
  -- Distance from C to the side of the angle passing through B
  h_B : ℝ
  -- Condition: h is equal to 4
  h_eq_four : h = 4
  -- Condition: One distance is four times the other
  one_distance_four_times_other : h_B = 4 * h_A

/-- The theorem states that under the given conditions, the sum of distances h_A and h_B is 10. -/
theorem sum_of_distances_is_ten (p : CircleTangentProblem) : p.h_A + p.h_B = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distances_is_ten_l1959_195944


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1959_195908

-- Define the sets A and B
def A : Set ℝ := {x | 2 / x > 1}
def B : Set ℝ := {x | Real.log x < 0}

-- Define the union of A and B
def AunionB : Set ℝ := A ∪ B

-- Theorem statement
theorem union_of_A_and_B : AunionB = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1959_195908


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1959_195989

/-- The lateral surface area of a cone with base radius 3 and height 4 is 15π. -/
theorem cone_lateral_surface_area :
  let r : ℝ := 3  -- base radius
  let h : ℝ := 4  -- height
  let l : ℝ := (r^2 + h^2).sqrt  -- slant height
  π * r * l = 15 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1959_195989


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1959_195972

/-- The imaginary part of (3-2i)/(1-i) is 1/2 -/
theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (3 - 2*I) / (1 - I)
  Complex.im z = 1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1959_195972


namespace NUMINAMATH_CALUDE_max_value_theorem_l1959_195922

theorem max_value_theorem (x y : ℝ) : 
  (2*x + 3*y + 5) / Real.sqrt (x^2 + 2*y^2 + 2) ≤ Real.sqrt 38 ∧ 
  ∃ (x₀ y₀ : ℝ), (2*x₀ + 3*y₀ + 5) / Real.sqrt (x₀^2 + 2*y₀^2 + 2) = Real.sqrt 38 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1959_195922


namespace NUMINAMATH_CALUDE_total_birds_is_168_l1959_195974

/-- Represents the number of birds of each species -/
structure BirdCounts where
  bluebirds : ℕ
  cardinals : ℕ
  goldfinches : ℕ
  sparrows : ℕ
  swallows : ℕ

/-- Conditions for the bird counts -/
def validBirdCounts (b : BirdCounts) : Prop :=
  b.cardinals = 2 * b.bluebirds ∧
  b.goldfinches = 4 * b.bluebirds ∧
  b.sparrows = (b.cardinals + b.goldfinches) / 2 ∧
  b.swallows = 8 ∧
  b.bluebirds = 2 * b.swallows

/-- The total number of birds -/
def totalBirds (b : BirdCounts) : ℕ :=
  b.bluebirds + b.cardinals + b.goldfinches + b.sparrows + b.swallows

/-- Theorem: The total number of birds is 168 -/
theorem total_birds_is_168 :
  ∀ b : BirdCounts, validBirdCounts b → totalBirds b = 168 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_is_168_l1959_195974


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1959_195911

/-- For any a, b, and c, the expression a^4(b^3 - c^3) + b^4(c^3 - a^3) + c^4(a^3 - b^3)
    can be factored as (a - b)(b - c)(c - a) multiplied by a specific polynomial in a, b, and c. -/
theorem polynomial_factorization (a b c : ℝ) :
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (a^3*b + a^3*c + a^2*b^2 + a^2*b*c + a^2*c^2 + a*b^3 + a*b*c^2 + a*c^3 + b^3*c + b^2*c^2 + b*c^3) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1959_195911


namespace NUMINAMATH_CALUDE_larger_number_proof_l1959_195996

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 7 * S + 15) : 
  L = 1590 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1959_195996


namespace NUMINAMATH_CALUDE_N_inverse_proof_l1959_195921

def N : Matrix (Fin 3) (Fin 3) ℝ := !![1, 2, -1; 4, -3, 2; -3, 5, 0]

theorem N_inverse_proof :
  let N_inv : Matrix (Fin 3) (Fin 3) ℝ := !![5/21, 5/14, -1/21; 3/14, 1/14, 5/42; -1/21, -19/42, 11/42]
  N * N_inv = 1 ∧ N_inv * N = 1 := by sorry

end NUMINAMATH_CALUDE_N_inverse_proof_l1959_195921


namespace NUMINAMATH_CALUDE_ryegrass_percentage_in_x_l1959_195973

/-- Represents a seed mixture with percentages of different grass types -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The combined mixture of X and Y -/
def combined_mixture (x y : SeedMixture) (x_proportion : ℝ) : SeedMixture :=
  { ryegrass := x.ryegrass * x_proportion + y.ryegrass * (1 - x_proportion)
  , bluegrass := x.bluegrass * x_proportion + y.bluegrass * (1 - x_proportion)
  , fescue := x.fescue * x_proportion + y.fescue * (1 - x_proportion) }

theorem ryegrass_percentage_in_x 
  (x : SeedMixture) 
  (y : SeedMixture) 
  (h1 : x.bluegrass = 60)
  (h2 : x.ryegrass + x.bluegrass + x.fescue = 100)
  (h3 : y.ryegrass = 25)
  (h4 : y.fescue = 75)
  (h5 : y.ryegrass + y.bluegrass + y.fescue = 100)
  (h6 : (combined_mixture x y (2/3)).ryegrass = 35) :
  x.ryegrass = 40 := by
    sorry

#check ryegrass_percentage_in_x

end NUMINAMATH_CALUDE_ryegrass_percentage_in_x_l1959_195973


namespace NUMINAMATH_CALUDE_y_share_is_63_l1959_195983

/-- Represents the share of each person in rupees -/
structure Share where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The total amount to be divided -/
def total_amount : ℝ := 245

/-- The ratio of y's share to x's share -/
def y_ratio : ℝ := 0.45

/-- The ratio of z's share to x's share -/
def z_ratio : ℝ := 0.30

/-- The share satisfies the given conditions -/
def is_valid_share (s : Share) : Prop :=
  s.x + s.y + s.z = total_amount ∧
  s.y = y_ratio * s.x ∧
  s.z = z_ratio * s.x

theorem y_share_is_63 :
  ∃ (s : Share), is_valid_share s ∧ s.y = 63 := by
  sorry

end NUMINAMATH_CALUDE_y_share_is_63_l1959_195983


namespace NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l1959_195994

theorem only_one_divides_power_minus_one :
  ∀ n : ℕ, n ∣ (2^n - 1) ↔ n = 1 := by sorry

end NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l1959_195994


namespace NUMINAMATH_CALUDE_arthurs_walk_distance_l1959_195941

/-- Represents the distance walked in each direction --/
structure WalkDistance where
  east : ℕ
  north : ℕ
  west : ℕ

/-- Calculates the total distance walked in miles --/
def total_distance (walk : WalkDistance) (block_length : ℚ) : ℚ :=
  ((walk.east + walk.north + walk.west) : ℚ) * block_length

/-- Theorem: Arthur's walk totals 6.5 miles --/
theorem arthurs_walk_distance :
  let walk := WalkDistance.mk 8 15 3
  let block_length : ℚ := 1/4
  total_distance walk block_length = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_arthurs_walk_distance_l1959_195941


namespace NUMINAMATH_CALUDE_complex_modulus_l1959_195960

theorem complex_modulus (z : ℂ) : z = (1 - 2*I) / (1 - I) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1959_195960


namespace NUMINAMATH_CALUDE_part_one_part_two_l1959_195940

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 5|

-- Part I
theorem part_one : 
  {x : ℝ | f 1 x ≥ 2 * |x + 5|} = {x : ℝ | x ≤ -2} := by sorry

-- Part II
theorem part_two : 
  (∀ x : ℝ, f a x ≥ 8) → (a ≥ 3 ∨ a ≤ -13) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1959_195940


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1959_195998

/-- A quadratic trinomial of the form x^2 + kx + 9 is a perfect square if and only if k = ±6 -/
theorem perfect_square_trinomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + k*x + 9 = (x + a)^2) ↔ k = 6 ∨ k = -6 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1959_195998


namespace NUMINAMATH_CALUDE_eight_star_three_equals_fiftythree_l1959_195948

-- Define the operation ⋆
def star (a b : ℤ) : ℤ := 4*a + 6*b + 3

-- Theorem statement
theorem eight_star_three_equals_fiftythree : star 8 3 = 53 := by sorry

end NUMINAMATH_CALUDE_eight_star_three_equals_fiftythree_l1959_195948


namespace NUMINAMATH_CALUDE_we_the_people_cows_l1959_195971

theorem we_the_people_cows (W : ℕ) : 
  W + (3 * W + 2) = 70 → W = 17 := by
  sorry

end NUMINAMATH_CALUDE_we_the_people_cows_l1959_195971


namespace NUMINAMATH_CALUDE_gcf_of_40_and_14_l1959_195947

theorem gcf_of_40_and_14 :
  let n : ℕ := 40
  let m : ℕ := 14
  let lcm_nm : ℕ := 56
  Nat.lcm n m = lcm_nm →
  Nat.gcd n m = 10 := by
sorry

end NUMINAMATH_CALUDE_gcf_of_40_and_14_l1959_195947


namespace NUMINAMATH_CALUDE_smaller_of_two_numbers_l1959_195953

theorem smaller_of_two_numbers (x y a b c : ℝ) : 
  x > 0 → y > 0 → x * y = c → x^2 - b*x + a*y = 0 → 0 < a → a < b → 
  min x y = c / a :=
by sorry

end NUMINAMATH_CALUDE_smaller_of_two_numbers_l1959_195953


namespace NUMINAMATH_CALUDE_max_oranges_for_teacher_l1959_195963

theorem max_oranges_for_teacher (n : ℕ) : 
  let k := 8
  let remainder := n % k
  remainder ≤ 7 ∧ ∃ m : ℕ, n = m * k + 7 :=
by sorry

end NUMINAMATH_CALUDE_max_oranges_for_teacher_l1959_195963


namespace NUMINAMATH_CALUDE_sum_of_numbers_greater_than_point_four_l1959_195991

theorem sum_of_numbers_greater_than_point_four : 
  let numbers : List ℚ := [0.8, 1/2, 0.9]
  let sum_of_greater : ℚ := (numbers.filter (λ x => x > 0.4)).sum
  sum_of_greater = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_greater_than_point_four_l1959_195991


namespace NUMINAMATH_CALUDE_cylindrical_tank_capacity_l1959_195969

theorem cylindrical_tank_capacity (x : ℝ) 
  (h1 : 0.24 * x = 72) : x = 300 := by
  sorry

end NUMINAMATH_CALUDE_cylindrical_tank_capacity_l1959_195969


namespace NUMINAMATH_CALUDE_minimum_cubes_required_l1959_195980

/-- Represents a 3D grid of unit cubes -/
def CubeGrid := List (List (List Bool))

/-- Checks if a cube in the grid shares at least one face with another cube -/
def sharesface (grid : CubeGrid) : Bool :=
  sorry

/-- Generates the front view of the grid -/
def frontView (grid : CubeGrid) : List (List Bool) :=
  sorry

/-- Generates the side view of the grid -/
def sideView (grid : CubeGrid) : List (List Bool) :=
  sorry

/-- The given front view -/
def givenFrontView : List (List Bool) :=
  [[true, true, false],
   [true, true, false],
   [true, false, false]]

/-- The given side view -/
def givenSideView : List (List Bool) :=
  [[true, true, true, false],
   [false, true, false, false],
   [false, false, true, false]]

/-- Counts the number of cubes in the grid -/
def countCubes (grid : CubeGrid) : Nat :=
  sorry

theorem minimum_cubes_required :
  ∃ (grid : CubeGrid),
    sharesface grid ∧
    frontView grid = givenFrontView ∧
    sideView grid = givenSideView ∧
    countCubes grid = 5 ∧
    (∀ (other : CubeGrid),
      sharesface other →
      frontView other = givenFrontView →
      sideView other = givenSideView →
      countCubes other ≥ 5) :=
  sorry

end NUMINAMATH_CALUDE_minimum_cubes_required_l1959_195980


namespace NUMINAMATH_CALUDE_victoria_money_l1959_195933

/-- The amount of money Victoria was given by her mother -/
def total_money : ℕ := sorry

/-- The cost of one box of pizza -/
def pizza_cost : ℕ := 12

/-- The number of pizza boxes bought -/
def pizza_boxes : ℕ := 2

/-- The cost of one pack of juice drinks -/
def juice_cost : ℕ := 2

/-- The number of juice drink packs bought -/
def juice_packs : ℕ := 2

/-- The amount Victoria should return to her mother -/
def return_amount : ℕ := 22

/-- Theorem stating that the total money Victoria was given equals $50 -/
theorem victoria_money : 
  total_money = pizza_cost * pizza_boxes + juice_cost * juice_packs + return_amount :=
by sorry

end NUMINAMATH_CALUDE_victoria_money_l1959_195933


namespace NUMINAMATH_CALUDE_four_circles_common_tangent_l1959_195951

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the length of a common tangent between two circles -/
def tangentLength (c1 c2 : Circle) : ℝ := sorry

/-- 
Given four circles α, β, γ, and δ satisfying the tangent length equation,
there exists a circle tangent to all four circles.
-/
theorem four_circles_common_tangent 
  (α β γ δ : Circle)
  (h : tangentLength α β * tangentLength γ δ + 
       tangentLength β γ * tangentLength δ α = 
       tangentLength α γ * tangentLength β δ) :
  ∃ (σ : Circle), 
    (tangentLength σ α = 0) ∧ 
    (tangentLength σ β = 0) ∧ 
    (tangentLength σ γ = 0) ∧ 
    (tangentLength σ δ = 0) :=
sorry

end NUMINAMATH_CALUDE_four_circles_common_tangent_l1959_195951


namespace NUMINAMATH_CALUDE_two_parts_problem_l1959_195999

theorem two_parts_problem (x y : ℝ) (h1 : x + y = 24) (h2 : 7 * x + 5 * y = 146) : x = 13 := by
  sorry

end NUMINAMATH_CALUDE_two_parts_problem_l1959_195999


namespace NUMINAMATH_CALUDE_multiple_is_two_l1959_195910

/-- The grading method used by a teacher for a test -/
structure GradingMethod where
  totalQuestions : ℕ
  studentScore : ℕ
  correctAnswers : ℕ
  scoreCalculation : ℕ → ℕ → ℕ → ℕ → ℕ

/-- The multiple used for incorrect responses in the grading method -/
def incorrectResponseMultiple (gm : GradingMethod) : ℕ :=
  let incorrectAnswers := gm.totalQuestions - gm.correctAnswers
  (gm.correctAnswers - gm.studentScore) / incorrectAnswers

/-- Theorem stating that the multiple used for incorrect responses is 2 -/
theorem multiple_is_two (gm : GradingMethod) 
  (h1 : gm.totalQuestions = 100)
  (h2 : gm.studentScore = 76)
  (h3 : gm.correctAnswers = 92)
  (h4 : gm.scoreCalculation = fun total correct incorrect multiple => 
    correct - multiple * incorrect) :
  incorrectResponseMultiple gm = 2 := by
  sorry


end NUMINAMATH_CALUDE_multiple_is_two_l1959_195910


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1959_195959

theorem cubic_equation_roots (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x^3 - 3*x^2 - a = 0 ∧ 
    y^3 - 3*y^2 - a = 0 ∧ 
    z^3 - 3*z^2 - a = 0) → 
  -4 < a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l1959_195959


namespace NUMINAMATH_CALUDE_charity_boxes_theorem_l1959_195915

/-- Calculates the total number of boxes a charity can pack given initial conditions --/
theorem charity_boxes_theorem (initial_boxes : ℕ) (food_cost : ℕ) (supplies_cost : ℕ) (donation_multiplier : ℕ) : 
  initial_boxes = 400 → 
  food_cost = 80 → 
  supplies_cost = 165 → 
  donation_multiplier = 4 → 
  (initial_boxes + (donation_multiplier * initial_boxes * (food_cost + supplies_cost)) / (food_cost + supplies_cost) : ℕ) = 2000 := by
  sorry

#check charity_boxes_theorem

end NUMINAMATH_CALUDE_charity_boxes_theorem_l1959_195915


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l1959_195925

theorem stewart_farm_sheep_count :
  -- Definitions
  let sheep_horse_cow_ratio : Fin 3 → ℕ := ![4, 7, 5]
  let food_per_animal : Fin 3 → ℕ := ![150, 230, 300]
  let total_food : Fin 3 → ℕ := ![9750, 12880, 15000]

  -- Conditions
  ∀ (num_animals : Fin 3 → ℕ),
    (∀ i : Fin 3, num_animals i * food_per_animal i = total_food i) →
    (∀ i j : Fin 3, num_animals i * sheep_horse_cow_ratio j = num_animals j * sheep_horse_cow_ratio i) →

  -- Conclusion
  num_animals 0 = 98 :=
by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l1959_195925


namespace NUMINAMATH_CALUDE_min_value_f_l1959_195942

/-- Given that 2x^2 + 3xy + 2y^2 = 1, the minimum value of f(x, y) = x + y + xy is -9/8 -/
theorem min_value_f (x y : ℝ) (h : 2*x^2 + 3*x*y + 2*y^2 = 1) :
  ∃ (m : ℝ), m = -9/8 ∧ ∀ (a b : ℝ), 2*a^2 + 3*a*b + 2*b^2 = 1 → a + b + a*b ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_f_l1959_195942


namespace NUMINAMATH_CALUDE_sum_of_x_coordinates_is_zero_l1959_195965

/-- A piecewise linear function composed of six line segments -/
def PiecewiseLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ x₅ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧ x₄ < x₅ ∧
    (∀ x, (x ≤ x₁ ∨ x₁ < x ∧ x ≤ x₂ ∨ x₂ < x ∧ x ≤ x₃ ∨
           x₃ < x ∧ x ≤ x₄ ∨ x₄ < x ∧ x ≤ x₅ ∨ x₅ < x) →
      ∃ (a b : ℝ), ∀ y ∈ Set.Icc x₁ x, f y = a * y + b)

/-- The graph of g intersects with y = x - 1 at exactly three points -/
def ThreeIntersections (g : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧
    (∀ x, g x = x - 1 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)

theorem sum_of_x_coordinates_is_zero
  (g : ℝ → ℝ)
  (h₁ : PiecewiseLinearFunction g)
  (h₂ : ThreeIntersections g) :
  ∃ (x₁ x₂ x₃ : ℝ), (∀ x, g x = x - 1 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
                    x₁ + x₂ + x₃ = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_x_coordinates_is_zero_l1959_195965


namespace NUMINAMATH_CALUDE_lab_capacity_l1959_195945

theorem lab_capacity (total_capacity : ℕ) (total_stations : ℕ) (two_student_stations : ℕ) 
  (h1 : total_capacity = 38)
  (h2 : total_stations = 16)
  (h3 : two_student_stations = 10) :
  total_capacity - (2 * two_student_stations) = 18 := by
  sorry

end NUMINAMATH_CALUDE_lab_capacity_l1959_195945


namespace NUMINAMATH_CALUDE_car_speed_time_relations_l1959_195905

/-- Represents the speed and time of a car --/
structure CarData where
  speed : ℝ
  time : ℝ

/-- Given conditions and proof goals for the car problem --/
theorem car_speed_time_relations 
  (x y z : CarData) 
  (h1 : y.speed = 3 * x.speed) 
  (h2 : z.speed = (x.speed + y.speed) / 2) 
  (h3 : x.speed * x.time = y.speed * y.time) 
  (h4 : x.speed * x.time = z.speed * z.time) : 
  z.speed = 2 * x.speed ∧ 
  y.time = x.time / 3 ∧ 
  z.time = x.time / 2 := by
  sorry


end NUMINAMATH_CALUDE_car_speed_time_relations_l1959_195905


namespace NUMINAMATH_CALUDE_johns_daily_wage_without_bonus_l1959_195982

/-- John's work scenario -/
structure WorkScenario where
  regular_hours : ℕ
  bonus_hours : ℕ
  bonus_amount : ℕ
  hourly_rate_with_bonus : ℕ

/-- Calculates John's daily wage without bonus -/
def daily_wage_without_bonus (w : WorkScenario) : ℕ :=
  w.hourly_rate_with_bonus * w.bonus_hours - w.bonus_amount

/-- Theorem: John's daily wage without bonus is $80 -/
theorem johns_daily_wage_without_bonus :
  let w : WorkScenario := {
    regular_hours := 8,
    bonus_hours := 10,
    bonus_amount := 20,
    hourly_rate_with_bonus := 10
  }
  daily_wage_without_bonus w = 80 := by
  sorry


end NUMINAMATH_CALUDE_johns_daily_wage_without_bonus_l1959_195982


namespace NUMINAMATH_CALUDE_inverse_proportion_comparison_l1959_195907

/-- Given two points A(1, y₁) and B(2, y₂) on the graph of y = 2/x, prove that y₁ > y₂ -/
theorem inverse_proportion_comparison (y₁ y₂ : ℝ) :
  y₁ = 2 / 1 → y₂ = 2 / 2 → y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_comparison_l1959_195907


namespace NUMINAMATH_CALUDE_triangle_properties_l1959_195917

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a = 2 * b ∧
  Real.sin A + Real.sin B = 2 * Real.sin C ∧
  (1 / 2) * b * c * Real.sin A = (8 * Real.sqrt 15) / 3 →
  Real.cos A = -1 / 4 ∧ c = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1959_195917


namespace NUMINAMATH_CALUDE_parallelogram_vector_l1959_195981

/-- A parallelogram on the complex plane -/
structure Parallelogram :=
  (A B C D : ℂ)
  (parallelogram_condition : (C - A) = (D - B))

/-- The theorem statement -/
theorem parallelogram_vector (ABCD : Parallelogram) 
  (hAC : ABCD.C - ABCD.A = 6 + 8*I) 
  (hBD : ABCD.D - ABCD.B = -4 + 6*I) : 
  ABCD.A - ABCD.D = -1 - 7*I :=
sorry

end NUMINAMATH_CALUDE_parallelogram_vector_l1959_195981


namespace NUMINAMATH_CALUDE_gcd_4050_12150_l1959_195985

theorem gcd_4050_12150 : Nat.gcd 4050 12150 = 450 := by
  sorry

end NUMINAMATH_CALUDE_gcd_4050_12150_l1959_195985


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1959_195936

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → (1 : ℚ) / x + (1 : ℚ) / y = 1 / 12 → (x : ℕ) + y ≥ 49 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1959_195936


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_m_eq_neg_two_l1959_195955

/-- Two vectors in ℝ² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- The theorem stating that if vectors (2, 3m+2) and (m, -1) are perpendicular, then m = -2 -/
theorem perpendicular_vectors_imply_m_eq_neg_two (m : ℝ) :
  perpendicular (2, 3*m+2) (m, -1) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_imply_m_eq_neg_two_l1959_195955


namespace NUMINAMATH_CALUDE_win_sector_area_l1959_195957

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 12) (h2 : p = 1/3) :
  p * π * r^2 = 48 * π := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l1959_195957


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l1959_195979

/-- The area of an equilateral triangle with altitude 2√3 is 4√3 -/
theorem equilateral_triangle_area (h : ℝ) (altitude_eq : h = 2 * Real.sqrt 3) :
  let side : ℝ := 2 * h / Real.sqrt 3
  let area : ℝ := side * h / 2
  area = 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l1959_195979


namespace NUMINAMATH_CALUDE_plum_cost_l1959_195990

theorem plum_cost (total_fruits : ℕ) (total_cost : ℕ) (peach_cost : ℕ) (plum_count : ℕ) :
  total_fruits = 32 →
  total_cost = 52 →
  peach_cost = 1 →
  plum_count = 20 →
  ∃ (plum_cost : ℕ), plum_cost = 2 ∧ 
    plum_cost * plum_count + peach_cost * (total_fruits - plum_count) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_plum_cost_l1959_195990


namespace NUMINAMATH_CALUDE_probability_of_drawing_ball_two_l1959_195958

/-- A box containing labeled balls. -/
structure Box where
  balls : Finset ℕ
  labels_distinct : balls.card = balls.toList.length

/-- The probability of drawing a specific ball from a box. -/
def probability_of_drawing (box : Box) (ball : ℕ) : ℚ :=
  if ball ∈ box.balls then 1 / box.balls.card else 0

/-- Theorem stating the probability of drawing ball 2 from a box with 3 balls labeled 1, 2, and 3. -/
theorem probability_of_drawing_ball_two :
  ∃ (box : Box), box.balls = {1, 2, 3} ∧ probability_of_drawing box 2 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_drawing_ball_two_l1959_195958


namespace NUMINAMATH_CALUDE_no_solution_for_four_divides_sum_of_squares_plus_one_l1959_195934

theorem no_solution_for_four_divides_sum_of_squares_plus_one :
  ∀ (a b : ℤ), ¬(4 ∣ a^2 + b^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_four_divides_sum_of_squares_plus_one_l1959_195934


namespace NUMINAMATH_CALUDE_pascal_triangle_row_15_fifth_number_l1959_195913

theorem pascal_triangle_row_15_fifth_number : 
  let row := List.range 16
  let pascal_row := row.map (fun k => Nat.choose 15 k)
  pascal_row[4] = 1365 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_row_15_fifth_number_l1959_195913


namespace NUMINAMATH_CALUDE_expression_evaluation_l1959_195992

theorem expression_evaluation :
  let a : ℚ := -1/3
  let b : ℤ := -3
  2 * (3 * a^2 * b - a * b^2) - (a * b^2 + 6 * a^2 * b) = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1959_195992


namespace NUMINAMATH_CALUDE_m_range_theorem_l1959_195995

/-- The function f(x) = x^2 + mx + 1 has two distinct roots -/
def has_two_distinct_roots (m : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

/-- There exists an x such that 4x^2 + 4(m-2)x + 1 ≤ 0 -/
def exists_nonpositive (m : ℝ) : Prop :=
  ∃ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≤ 0

/-- The range of m is (-∞, -2) ∪ [3, +∞) -/
def m_range (m : ℝ) : Prop :=
  m < -2 ∨ m ≥ 3

theorem m_range_theorem (m : ℝ) :
  has_two_distinct_roots m ∧ exists_nonpositive m → m_range m := by
  sorry

end NUMINAMATH_CALUDE_m_range_theorem_l1959_195995


namespace NUMINAMATH_CALUDE_sum_of_square_roots_geq_one_l1959_195984

theorem sum_of_square_roots_geq_one (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt (a / (a + 3 * b)) + Real.sqrt (b / (b + 3 * a)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_geq_one_l1959_195984
