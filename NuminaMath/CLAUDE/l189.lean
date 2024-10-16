import Mathlib

namespace NUMINAMATH_CALUDE_part1_part2_l189_18921

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a * x - 1) * (x - 1)

-- Part 1: Prove that a = 1/2 given the conditions
theorem part1 (a : ℝ) : 
  (∀ x : ℝ, f a x < 0 ↔ 1 < x ∧ x < 2) → a = 1/2 := by sorry

-- Part 2: Characterize the solution set for f(x) < 0 when a > 0
theorem part2 (a : ℝ) (h : a > 0) : 
  (∀ x : ℝ, f a x < 0 ↔ 
    ((0 < a ∧ a < 1 ∧ 1 < x ∧ x < 1/a) ∨
     (a = 1 ∧ False) ∨
     (a > 1 ∧ 1/a < x ∧ x < 1))) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l189_18921


namespace NUMINAMATH_CALUDE_power_sum_l189_18973

theorem power_sum (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(m+n) = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_l189_18973


namespace NUMINAMATH_CALUDE_cricket_run_rate_theorem_l189_18958

/-- Represents a cricket game scenario -/
structure CricketGame where
  total_overs : ℕ
  first_part_overs : ℕ
  first_part_run_rate : ℚ
  target_runs : ℕ

/-- Calculates the required run rate for the remaining overs -/
def required_run_rate (game : CricketGame) : ℚ :=
  let remaining_overs := game.total_overs - game.first_part_overs
  let runs_scored := game.first_part_run_rate * game.first_part_overs
  let runs_needed := game.target_runs - runs_scored
  runs_needed / remaining_overs

/-- Theorem stating the required run rate for the given cricket game scenario -/
theorem cricket_run_rate_theorem (game : CricketGame)
  (h1 : game.total_overs = 50)
  (h2 : game.first_part_overs = 10)
  (h3 : game.first_part_run_rate = 3.8)
  (h4 : game.target_runs = 282) :
  required_run_rate game = 6.1 := by
  sorry

#eval required_run_rate {
  total_overs := 50,
  first_part_overs := 10,
  first_part_run_rate := 3.8,
  target_runs := 282
}

end NUMINAMATH_CALUDE_cricket_run_rate_theorem_l189_18958


namespace NUMINAMATH_CALUDE_genuine_purses_and_handbags_l189_18997

theorem genuine_purses_and_handbags 
  (total_purses : ℕ) 
  (total_handbags : ℕ) 
  (fake_purses_ratio : ℚ) 
  (fake_handbags_ratio : ℚ) 
  (h1 : total_purses = 26) 
  (h2 : total_handbags = 24) 
  (h3 : fake_purses_ratio = 1/2) 
  (h4 : fake_handbags_ratio = 1/4) :
  (total_purses - total_purses * fake_purses_ratio) + 
  (total_handbags - total_handbags * fake_handbags_ratio) = 31 := by
sorry

end NUMINAMATH_CALUDE_genuine_purses_and_handbags_l189_18997


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l189_18912

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a * b > 0) (h2 : a + 4 * b = 1) :
  (1 / a + 1 / b) ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l189_18912


namespace NUMINAMATH_CALUDE_units_digit_of_17_squared_times_29_l189_18989

theorem units_digit_of_17_squared_times_29 : 
  (17^2 * 29) % 10 = 1 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_17_squared_times_29_l189_18989


namespace NUMINAMATH_CALUDE_star_operation_divisors_l189_18936

-- Define the star operation
def star (a b : ℤ) : ℚ := (a^2 : ℚ) / b

-- Define the count of positive integer divisors of a number
def countPositiveDivisors (n : ℕ) : ℕ := sorry

-- Define the count of integer x for which (20 ★ x) is a positive integer
def countValidX : ℕ := sorry

-- Theorem statement
theorem star_operation_divisors : 
  countPositiveDivisors 400 = countValidX := by sorry

end NUMINAMATH_CALUDE_star_operation_divisors_l189_18936


namespace NUMINAMATH_CALUDE_grocery_bagging_l189_18910

/-- The number of ways to distribute n distinct objects into k indistinguishable containers,
    allowing empty containers. -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- There are 5 different items and 3 identical bags. -/
theorem grocery_bagging : distribute 5 3 = 41 := by
  sorry

end NUMINAMATH_CALUDE_grocery_bagging_l189_18910


namespace NUMINAMATH_CALUDE_unique_m_existence_l189_18970

theorem unique_m_existence : ∃! m : ℤ,
  50 ≤ m ∧ m ≤ 120 ∧
  m % 7 = 0 ∧
  m % 8 = 5 ∧
  m % 5 = 4 ∧
  m = 189 := by
  sorry

end NUMINAMATH_CALUDE_unique_m_existence_l189_18970


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l189_18911

/-- The diagonal of a rectangle with perimeter 40 inches and length to width ratio of 3:2 --/
theorem rectangle_diagonal (l w : ℝ) : 
  (2 * l + 2 * w = 40) →  -- Perimeter condition
  (l = (3/2) * w) →       -- Length to width ratio condition
  Real.sqrt (l^2 + w^2) = Real.sqrt 208 := by
sorry


end NUMINAMATH_CALUDE_rectangle_diagonal_l189_18911


namespace NUMINAMATH_CALUDE_max_cards_in_original_position_l189_18952

/-- Represents a two-digit number card -/
structure Card :=
  (tens : Nat)
  (ones : Nat)
  (h1 : tens < 10)
  (h2 : ones < 10)

/-- The list of all cards from 00 to 99 in ascending order -/
def initial_arrangement : List Card := sorry

/-- Checks if two cards are adjacent according to the rearrangement rule -/
def are_adjacent (c1 c2 : Card) : Prop := sorry

/-- A valid rearrangement of cards -/
def valid_rearrangement (arrangement : List Card) : Prop :=
  arrangement.length = 100 ∧
  ∀ i, i < 99 → are_adjacent (arrangement.get ⟨i, sorry⟩) (arrangement.get ⟨i+1, sorry⟩)

/-- The number of cards in their original positions after rearrangement -/
def cards_in_original_position (arrangement : List Card) : Nat := sorry

/-- Theorem stating the maximum number of cards that can remain in their original positions -/
theorem max_cards_in_original_position :
  ∀ arrangement : List Card,
    valid_rearrangement arrangement →
    cards_in_original_position arrangement ≤ 50 :=
sorry

end NUMINAMATH_CALUDE_max_cards_in_original_position_l189_18952


namespace NUMINAMATH_CALUDE_principal_calculation_l189_18948

/-- Calculates the principal given simple interest, rate, and time -/
def calculate_principal (simple_interest : ℚ) (rate : ℚ) (time : ℕ) : ℚ :=
  simple_interest / (rate * time)

/-- Theorem: Given the specified conditions, the principal is 44625 -/
theorem principal_calculation :
  let simple_interest : ℚ := 4016.25
  let rate : ℚ := 1 / 100
  let time : ℕ := 9
  calculate_principal simple_interest rate time = 44625 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l189_18948


namespace NUMINAMATH_CALUDE_daxton_water_usage_l189_18943

theorem daxton_water_usage 
  (tank_capacity : ℝ)
  (initial_fill_ratio : ℝ)
  (refill_ratio : ℝ)
  (final_volume : ℝ)
  (h1 : tank_capacity = 8000)
  (h2 : initial_fill_ratio = 3/4)
  (h3 : refill_ratio = 0.3)
  (h4 : final_volume = 4680) :
  let initial_volume := tank_capacity * initial_fill_ratio
  let usage_percentage := 
    (initial_volume - (final_volume - refill_ratio * (initial_volume - usage_volume))) / initial_volume
  let usage_volume := usage_percentage * initial_volume
  usage_percentage = 0.4 := by
sorry

end NUMINAMATH_CALUDE_daxton_water_usage_l189_18943


namespace NUMINAMATH_CALUDE_parallelogram_side_sum_l189_18961

/-- A parallelogram with sides measuring 10, 12, 5y-2, and 3x+6 units consecutively has x + y = 22/5 -/
theorem parallelogram_side_sum (x y : ℚ) : 
  3 * x + 6 = 12 → 5 * y - 2 = 10 → x + y = 22 / 5 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_sum_l189_18961


namespace NUMINAMATH_CALUDE_second_workshop_production_l189_18962

/-- Represents the production and sampling data for three workshops -/
structure WorkshopData where
  total_production : ℕ
  sample_1 : ℕ
  sample_2 : ℕ
  sample_3 : ℕ

/-- Checks if three numbers form an arithmetic sequence -/
def isArithmeticSequence (a b c : ℕ) : Prop :=
  b - a = c - b

/-- Calculates the production of the second workshop based on sampling data -/
def productionOfSecondWorkshop (data : WorkshopData) : ℕ :=
  data.sample_2 * data.total_production / (data.sample_1 + data.sample_2 + data.sample_3)

/-- Theorem stating the production of the second workshop is 1200 given the conditions -/
theorem second_workshop_production
  (data : WorkshopData)
  (h_total : data.total_production = 3600)
  (h_arithmetic : isArithmeticSequence data.sample_1 data.sample_2 data.sample_3) :
  productionOfSecondWorkshop data = 1200 := by
  sorry


end NUMINAMATH_CALUDE_second_workshop_production_l189_18962


namespace NUMINAMATH_CALUDE_one_correct_statement_l189_18909

theorem one_correct_statement (a b : ℤ) : 
  (∃! n : Nat, n < 3 ∧ n > 0 ∧
    ((n = 1 → (Even (a + 5*b) → Even (a - 7*b))) ∧
     (n = 2 → ((a + b) % 3 = 0 → a % 3 = 0 ∧ b % 3 = 0)) ∧
     (n = 3 → (Prime (a + b) → ¬ Prime (a - b))))) := by
  sorry

end NUMINAMATH_CALUDE_one_correct_statement_l189_18909


namespace NUMINAMATH_CALUDE_circle_area_ratio_l189_18900

/-- A square with side length 2 -/
structure Square :=
  (side_length : ℝ)
  (is_two : side_length = 2)

/-- A circle outside the square -/
structure Circle :=
  (radius : ℝ)
  (center : ℝ × ℝ)

/-- The configuration of the two circles and the square -/
structure Configuration :=
  (square : Square)
  (circle1 : Circle)
  (circle2 : Circle)
  (tangent_to_PQ : circle1.center.2 = square.side_length / 2)
  (tangent_to_RS : circle2.center.2 = -square.side_length / 2)
  (tangent_to_QR_extension : circle1.center.1 + circle1.radius = circle2.center.1 - circle2.radius)

/-- The theorem stating the ratio of the areas -/
theorem circle_area_ratio (config : Configuration) : 
  (π * config.circle2.radius^2) / (π * config.circle1.radius^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l189_18900


namespace NUMINAMATH_CALUDE_alice_next_birthday_age_l189_18904

theorem alice_next_birthday_age :
  ∀ (a b c : ℝ),
  a = 1.25 * b →                -- Alice is 25% older than Bob
  b = 0.7 * c →                 -- Bob is 30% younger than Carlos
  a + b + c = 30 →              -- Sum of their ages is 30 years
  ⌊a⌋ + 1 = 11 :=               -- Alice's age on her next birthday
by
  sorry


end NUMINAMATH_CALUDE_alice_next_birthday_age_l189_18904


namespace NUMINAMATH_CALUDE_water_transfer_l189_18969

theorem water_transfer (a b x : ℝ) : 
  a = 13.2 ∧ 
  (13.2 - x = (1/3) * (b + x)) ∧ 
  (b - x = (1/2) * (13.2 + x)) → 
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_water_transfer_l189_18969


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l189_18935

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 26 cm and height 14 cm is 364 square centimeters -/
theorem parallelogram_area_example : parallelogram_area 26 14 = 364 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l189_18935


namespace NUMINAMATH_CALUDE_orange_count_l189_18928

/-- The number of oranges initially in the bin -/
def initial_oranges : ℕ := sorry

/-- The number of oranges thrown away -/
def thrown_away : ℕ := 20

/-- The number of new oranges added -/
def new_oranges : ℕ := 13

/-- The final number of oranges in the bin -/
def final_oranges : ℕ := 27

theorem orange_count : initial_oranges = 34 :=
  by sorry

end NUMINAMATH_CALUDE_orange_count_l189_18928


namespace NUMINAMATH_CALUDE_abs_five_minus_sqrt_two_l189_18983

theorem abs_five_minus_sqrt_two : |5 - Real.sqrt 2| = 5 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_five_minus_sqrt_two_l189_18983


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l189_18940

theorem complex_magnitude_problem (z w : ℂ) : 
  Complex.abs (3 * z - w) = 30 →
  Complex.abs (z + 3 * w) = 6 →
  Complex.abs (z + w) = 3 →
  ∃! (abs_z : ℝ), abs_z > 0 ∧ Complex.abs z = abs_z :=
by sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l189_18940


namespace NUMINAMATH_CALUDE_paths_equal_choose_l189_18908

/-- The number of paths on a 3x3 grid from top-left to bottom-right -/
def num_paths : ℕ := sorry

/-- The number of ways to choose 3 items from a set of 6 items -/
def choose_3_from_6 : ℕ := Nat.choose 6 3

/-- Theorem stating that the number of paths is equal to choosing 3 from 6 -/
theorem paths_equal_choose :
  num_paths = choose_3_from_6 := by sorry

end NUMINAMATH_CALUDE_paths_equal_choose_l189_18908


namespace NUMINAMATH_CALUDE_whole_substitution_problems_l189_18917

theorem whole_substitution_problems :
  -- Problem 1
  (∀ m n : ℝ, m - n = -1 → 2 * (m - n)^2 + 18 = 20) ∧
  -- Problem 2
  (∀ m n : ℝ, m^2 + 2*m*n = 10 ∧ n^2 + 3*m*n = 6 → 2*m^2 + n^2 + 7*m*n = 26) ∧
  -- Problem 3
  (∀ a b c m : ℝ, a*(-1)^5 + b*(-1)^3 + c*(-1) - 5 = m → 
    a*(1)^5 + b*(1)^3 + c*(1) - 5 = -m - 10) :=
by sorry

end NUMINAMATH_CALUDE_whole_substitution_problems_l189_18917


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_five_l189_18946

theorem largest_four_digit_divisible_by_five : ∃ n : ℕ, 
  (n ≤ 9999 ∧ n ≥ 1000) ∧ 
  n % 5 = 0 ∧
  ∀ m : ℕ, (m ≤ 9999 ∧ m ≥ 1000 ∧ m % 5 = 0) → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_five_l189_18946


namespace NUMINAMATH_CALUDE_inverse_f_84_l189_18918

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 + 3

-- State the theorem
theorem inverse_f_84 : 
  ∃ (y : ℝ), f y = 84 ∧ y = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_f_84_l189_18918


namespace NUMINAMATH_CALUDE_harmonic_series_term_count_l189_18966

theorem harmonic_series_term_count (k : ℕ) (h : k ≥ 2) :
  (Finset.range (2^(k+1) - 1)).card - (Finset.range (2^k - 1)).card = 2^k := by
  sorry

end NUMINAMATH_CALUDE_harmonic_series_term_count_l189_18966


namespace NUMINAMATH_CALUDE_trig_identity_l189_18956

theorem trig_identity (α : Real) (h : π < α ∧ α < 3*π/2) :
  Real.sqrt (1/2 + 1/2 * Real.sqrt (1/2 + 1/2 * Real.cos (2*α))) = Real.sin (α/2) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l189_18956


namespace NUMINAMATH_CALUDE_even_odd_difference_3000_l189_18950

/-- Sum of the first n odd numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n * n

/-- Sum of the first n even numbers -/
def sumEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- The difference between the sum of the first n even numbers and the sum of the first n odd numbers -/
def evenOddDifference (n : ℕ) : ℕ := sumEvenNumbers n - sumOddNumbers n

theorem even_odd_difference_3000 : evenOddDifference 3000 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_difference_3000_l189_18950


namespace NUMINAMATH_CALUDE_pitchers_needed_l189_18985

def glasses_per_pitcher : ℝ := 4.5
def total_glasses_served : ℕ := 30

theorem pitchers_needed : 
  ∃ (n : ℕ), n * glasses_per_pitcher ≥ total_glasses_served ∧ 
  ∀ (m : ℕ), m * glasses_per_pitcher ≥ total_glasses_served → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_pitchers_needed_l189_18985


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_perimeter_l189_18929

/-- Given a rectangle with sides a and b, and an inscribed quadrilateral with vertices on each side
of the rectangle, the perimeter of the quadrilateral is greater than or equal to 2√(a² + b²). -/
theorem inscribed_quadrilateral_perimeter (a b : ℝ) (x y z t : ℝ)
  (hx : 0 ≤ x ∧ x ≤ a) (hy : 0 ≤ y ∧ y ≤ b) (hz : 0 ≤ z ∧ z ≤ a) (ht : 0 ≤ t ∧ t ≤ b) :
  let perimeter := Real.sqrt ((a - x)^2 + t^2) + Real.sqrt ((b - t)^2 + z^2) +
                   Real.sqrt ((a - z)^2 + (b - y)^2) + Real.sqrt (x^2 + y^2)
  perimeter ≥ 2 * Real.sqrt (a^2 + b^2) := by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_perimeter_l189_18929


namespace NUMINAMATH_CALUDE_exists_n_factorial_starts_with_2015_l189_18971

/-- Given a natural number n, returns the first four digits of n! as a natural number -/
def firstFourDigitsOfFactorial (n : ℕ) : ℕ :=
  sorry

/-- Theorem: There exists a positive integer n such that the first four digits of n! are 2015 -/
theorem exists_n_factorial_starts_with_2015 : ∃ n : ℕ+, firstFourDigitsOfFactorial n.val = 2015 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_factorial_starts_with_2015_l189_18971


namespace NUMINAMATH_CALUDE_eddie_pies_l189_18975

/-- The number of pies Eddie's sister can bake per day -/
def sister_pies : ℕ := 6

/-- The number of pies Eddie's mother can bake per day -/
def mother_pies : ℕ := 8

/-- The total number of pies they can bake in 7 days -/
def total_pies : ℕ := 119

/-- The number of days they bake pies -/
def days : ℕ := 7

/-- Eddie can bake 3 pies a day -/
theorem eddie_pies : ∃ (eddie_pies : ℕ), 
  eddie_pies = 3 ∧ 
  days * (eddie_pies + sister_pies + mother_pies) = total_pies := by
  sorry

end NUMINAMATH_CALUDE_eddie_pies_l189_18975


namespace NUMINAMATH_CALUDE_remainder_of_60_div_18_l189_18993

theorem remainder_of_60_div_18 : ∃ q : ℕ, 60 = 18 * q + 6 := by
  sorry

#check remainder_of_60_div_18

end NUMINAMATH_CALUDE_remainder_of_60_div_18_l189_18993


namespace NUMINAMATH_CALUDE_conference_handshakes_l189_18927

/-- Represents a conference with two groups of people -/
structure Conference :=
  (total : ℕ)
  (group1 : ℕ)
  (group2 : ℕ)
  (h_total : total = group1 + group2)

/-- Calculates the number of handshakes in the conference -/
def handshakes (conf : Conference) : ℕ :=
  conf.group2 * (conf.group1 + conf.group2 - 1)

theorem conference_handshakes :
  ∃ (conf : Conference),
    conf.total = 40 ∧
    conf.group1 = 25 ∧
    conf.group2 = 15 ∧
    handshakes conf = 480 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l189_18927


namespace NUMINAMATH_CALUDE_total_rent_is_7800_l189_18965

/-- Represents the rent shares of four people renting a house -/
structure RentShares where
  purity : ℝ
  sheila : ℝ
  rose : ℝ
  john : ℝ

/-- Calculates the total rent based on the given rent shares -/
def totalRent (shares : RentShares) : ℝ :=
  shares.purity + shares.sheila + shares.rose + shares.john

/-- Theorem stating that the total rent is $7,800 given the conditions -/
theorem total_rent_is_7800 :
  ∀ (shares : RentShares),
    shares.sheila = 5 * shares.purity →
    shares.rose = 3 * shares.purity →
    shares.john = 4 * shares.purity →
    shares.rose = 1800 →
    totalRent shares = 7800 := by
  sorry

#check total_rent_is_7800

end NUMINAMATH_CALUDE_total_rent_is_7800_l189_18965


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l189_18933

/-- Compound interest calculation --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Problem statement --/
theorem interest_rate_calculation (principal time : ℕ) (final_amount : ℝ) 
  (h1 : principal = 6000)
  (h2 : time = 2)
  (h3 : final_amount = 7260) :
  ∃ (rate : ℝ), compound_interest principal rate time = final_amount ∧ rate = 0.1 :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l189_18933


namespace NUMINAMATH_CALUDE_program_output_l189_18957

-- Define the program steps as a function
def program (a b : Int) : (Int × Int × Int) :=
  let a' := if a < 0 then -a else a
  let b' := b * b
  let a'' := a' + b'
  let c := a'' - 2 * b'
  let a''' := a'' / c
  let b'' := b' * c + 1
  (a''', b'', c)

-- State the theorem
theorem program_output : program (-6) 2 = (5, 9, 2) := by
  sorry

end NUMINAMATH_CALUDE_program_output_l189_18957


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l189_18992

/-- The total surface area of a cylinder with height 12 and radius 4 is 128π. -/
theorem cylinder_surface_area :
  let h : ℝ := 12
  let r : ℝ := 4
  let circle_area := π * r^2
  let lateral_area := 2 * π * r * h
  circle_area * 2 + lateral_area = 128 * π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l189_18992


namespace NUMINAMATH_CALUDE_valid_fractions_characterization_l189_18915

def is_valid_fraction (a b : ℕ) : Prop :=
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧
  (a % 10 = b / 10) ∧
  (a : ℚ) / b = (a / 10 : ℚ) / (b % 10)

def valid_fractions : Set (ℕ × ℕ) :=
  {(a, b) | is_valid_fraction a b}

theorem valid_fractions_characterization :
  valid_fractions = {(19, 95), (49, 98), (11, 11), (22, 22), (33, 33),
                     (44, 44), (55, 55), (66, 66), (77, 77), (88, 88),
                     (99, 99), (16, 64), (26, 65)} :=
by sorry

end NUMINAMATH_CALUDE_valid_fractions_characterization_l189_18915


namespace NUMINAMATH_CALUDE_complex_modulus_l189_18954

theorem complex_modulus (z : ℂ) : z = (1 + 3*I) / (3 - I) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l189_18954


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l189_18994

/-- An isosceles triangle with two sides of lengths 3 and 6 has a perimeter of 15. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 6 ∧ c = 6 →  -- Two sides are 6, one side is 3
  (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
  a + b + c = 15 :=
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l189_18994


namespace NUMINAMATH_CALUDE_files_per_folder_l189_18982

theorem files_per_folder (initial_files : ℕ) (deleted_files : ℕ) (num_folders : ℕ) 
  (h1 : initial_files = 27)
  (h2 : deleted_files = 9)
  (h3 : num_folders = 3)
  (h4 : num_folders > 0) :
  (initial_files - deleted_files) / num_folders = 6 := by
  sorry

end NUMINAMATH_CALUDE_files_per_folder_l189_18982


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_progression_first_term_is_three_common_difference_is_six_l189_18937

/-- The sum of the first n terms of a sequence -/
def S (n : ℕ) : ℝ := 3 * n^2

/-- The n-th term of the sequence -/
def u (n : ℕ) : ℝ := S n - S (n-1)

theorem sequence_is_arithmetic_progression :
  ∃ (a d : ℝ), ∀ n : ℕ, u n = a + (n - 1) * d :=
sorry

theorem first_term_is_three : u 1 = 3 :=
sorry

theorem common_difference_is_six :
  ∀ n : ℕ, n > 1 → u n - u (n-1) = 6 :=
sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_progression_first_term_is_three_common_difference_is_six_l189_18937


namespace NUMINAMATH_CALUDE_green_pill_cost_proof_l189_18991

/-- The cost of a green pill in dollars -/
def green_pill_cost : ℚ := 22.5

/-- The cost of a pink pill in dollars -/
def pink_pill_cost : ℚ := green_pill_cost - 2

/-- The number of days in the treatment period -/
def treatment_days : ℕ := 21

/-- The total cost of the treatment in dollars -/
def total_cost : ℚ := 903

theorem green_pill_cost_proof :
  green_pill_cost = 22.5 ∧
  pink_pill_cost = green_pill_cost - 2 ∧
  treatment_days = 21 ∧
  total_cost = 903 ∧
  total_cost = treatment_days * (green_pill_cost + pink_pill_cost) :=
by sorry

end NUMINAMATH_CALUDE_green_pill_cost_proof_l189_18991


namespace NUMINAMATH_CALUDE_total_cost_is_1975_l189_18953

def first_laptop_cost : ℝ := 500
def second_laptop_multiplier : ℝ := 3
def discount_rate : ℝ := 0.15
def external_hard_drive_cost : ℝ := 80
def mouse_cost : ℝ := 20

def total_cost : ℝ :=
  let second_laptop_cost := first_laptop_cost * second_laptop_multiplier
  let discounted_second_laptop_cost := second_laptop_cost * (1 - discount_rate)
  let accessories_cost := external_hard_drive_cost + mouse_cost
  first_laptop_cost + discounted_second_laptop_cost + 2 * accessories_cost

theorem total_cost_is_1975 : total_cost = 1975 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_1975_l189_18953


namespace NUMINAMATH_CALUDE_white_pairs_coincide_l189_18951

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCounts where
  red : Nat
  blue : Nat
  white : Nat

/-- Represents the number of coinciding pairs of each type when the figure is folded -/
structure CoincidingPairs where
  red_red : Nat
  blue_blue : Nat
  red_white : Nat
  white_white : Nat

/-- The main theorem to prove -/
theorem white_pairs_coincide (counts : TriangleCounts) (pairs : CoincidingPairs) : 
  counts.red = 4 ∧ 
  counts.blue = 7 ∧ 
  counts.white = 10 ∧
  pairs.red_red = 3 ∧
  pairs.blue_blue = 4 ∧
  pairs.red_white = 3 →
  pairs.white_white = 4 := by
  sorry

end NUMINAMATH_CALUDE_white_pairs_coincide_l189_18951


namespace NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l189_18974

/-- The volume of a cylinder formed by rotating a rectangle about its longer side -/
theorem cylinder_volume_from_rectangle (length width : ℝ) (length_ge_width : length ≥ width) :
  let radius := length / 2
  let height := width
  let volume := π * radius^2 * height
  length = 20 ∧ width = 10 → volume = 1000 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l189_18974


namespace NUMINAMATH_CALUDE_strawberry_price_proof_l189_18959

/-- The cost of strawberries in dollars per pound -/
def strawberry_cost : ℝ := sorry

/-- The cost of cherries in dollars per pound -/
def cherry_cost : ℝ := sorry

/-- The total cost of 5 pounds of strawberries and 5 pounds of cherries -/
def total_cost : ℝ := sorry

theorem strawberry_price_proof :
  (cherry_cost = 6 * strawberry_cost) →
  (total_cost = 5 * strawberry_cost + 5 * cherry_cost) →
  (total_cost = 77) →
  (strawberry_cost = 2.2) := by
  sorry

end NUMINAMATH_CALUDE_strawberry_price_proof_l189_18959


namespace NUMINAMATH_CALUDE_solution_y_percent_a_l189_18923

/-- Represents a chemical solution with a given percentage of chemical A -/
structure Solution where
  percent_a : ℝ
  h_percent_range : 0 ≤ percent_a ∧ percent_a ≤ 1

/-- Represents a mixture of two solutions -/
structure Mixture where
  solution_x : Solution
  solution_y : Solution
  proportion_x : ℝ
  h_proportion_range : 0 ≤ proportion_x ∧ proportion_x ≤ 1

/-- Calculates the percentage of chemical A in a mixture -/
def mixture_percent_a (m : Mixture) : ℝ :=
  m.proportion_x * m.solution_x.percent_a + (1 - m.proportion_x) * m.solution_y.percent_a

theorem solution_y_percent_a (x : Solution) (y : Solution) (m : Mixture) 
  (h_x : x.percent_a = 0.3)
  (h_m : m.solution_x = x ∧ m.solution_y = y ∧ m.proportion_x = 0.8)
  (h_mixture : mixture_percent_a m = 0.32) :
  y.percent_a = 0.4 := by
  sorry


end NUMINAMATH_CALUDE_solution_y_percent_a_l189_18923


namespace NUMINAMATH_CALUDE_trip_distance_calculation_l189_18925

theorem trip_distance_calculation (total_distance : ℝ) (speed1 speed2 avg_speed : ℝ) 
  (h1 : total_distance = 70)
  (h2 : speed1 = 48)
  (h3 : speed2 = 24)
  (h4 : avg_speed = 32) :
  ∃ (first_part : ℝ),
    first_part = 35 ∧
    first_part / speed1 + (total_distance - first_part) / speed2 = total_distance / avg_speed :=
by sorry

end NUMINAMATH_CALUDE_trip_distance_calculation_l189_18925


namespace NUMINAMATH_CALUDE_polygon_division_euler_characteristic_l189_18947

/-- A polygon division represents the result of dividing a polygon into several polygons. -/
structure PolygonDivision where
  p : ℕ  -- number of resulting polygons
  q : ℕ  -- number of segments that are the sides of these polygons
  r : ℕ  -- number of points that are their vertices

/-- The Euler characteristic of a polygon division is always 1. -/
theorem polygon_division_euler_characteristic (d : PolygonDivision) : 
  d.p - d.q + d.r = 1 := by
  sorry

end NUMINAMATH_CALUDE_polygon_division_euler_characteristic_l189_18947


namespace NUMINAMATH_CALUDE_number_of_sets_l189_18942

/-- Represents a four-digit number in the game "Set" -/
def SetNumber := Fin 4 → Fin 3

/-- Checks if three numbers form a valid set in the game "Set" -/
def is_valid_set (a b c : SetNumber) : Prop :=
  ∀ i : Fin 4, (a i = b i ∧ b i = c i) ∨ (a i ≠ b i ∧ b i ≠ c i ∧ a i ≠ c i)

/-- The set of all possible four-digit numbers in the game "Set" -/
def all_set_numbers : Finset SetNumber :=
  sorry

/-- The set of all valid sets in the game "Set" -/
def all_valid_sets : Finset (Finset SetNumber) :=
  sorry

/-- The main theorem stating the number of valid sets in the game "Set" -/
theorem number_of_sets : Finset.card all_valid_sets = 1080 :=
  sorry

end NUMINAMATH_CALUDE_number_of_sets_l189_18942


namespace NUMINAMATH_CALUDE_total_wristbands_distributed_l189_18944

/-- Represents the number of wristbands given to each spectator -/
def wristbands_per_spectator : ℕ := 2

/-- Represents the total number of wristbands distributed -/
def total_wristbands : ℕ := 125

/-- Theorem stating that the total number of wristbands distributed is 125 -/
theorem total_wristbands_distributed :
  total_wristbands = 125 := by sorry

end NUMINAMATH_CALUDE_total_wristbands_distributed_l189_18944


namespace NUMINAMATH_CALUDE_reciprocal_difference_problem_l189_18972

theorem reciprocal_difference_problem (m : ℚ) (hm : m ≠ 1) (h : 1 / (m - 1) = m) :
  m^4 + 1 / m^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_difference_problem_l189_18972


namespace NUMINAMATH_CALUDE_expand_expression_l189_18901

theorem expand_expression (x y : ℝ) : (x + 7) * (3 * y + 8) = 3 * x * y + 8 * x + 21 * y + 56 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l189_18901


namespace NUMINAMATH_CALUDE_covered_number_is_eight_l189_18922

/-- A circular arrangement of 15 numbers -/
def CircularArrangement := Fin 15 → ℕ

/-- The property that the sum of any six consecutive numbers is 50 -/
def SumProperty (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 15, (arr i + arr (i + 1) + arr (i + 2) + arr (i + 3) + arr (i + 4) + arr (i + 5)) = 50

/-- The property that two adjacent numbers are 7 and 10 with a number between them -/
def AdjacentProperty (arr : CircularArrangement) : Prop :=
  ∃ i : Fin 15, arr i = 7 ∧ arr (i + 2) = 10

theorem covered_number_is_eight (arr : CircularArrangement) 
  (h1 : SumProperty arr) (h2 : AdjacentProperty arr) : 
  ∃ i : Fin 15, arr i = 7 ∧ arr (i + 1) = 8 ∧ arr (i + 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_covered_number_is_eight_l189_18922


namespace NUMINAMATH_CALUDE_prime_power_equation_l189_18955

theorem prime_power_equation (p q s : Nat) (y : Nat) (hp : Prime p) (hq : Prime q) (hs : Prime s) (hy : y > 1) 
  (h : 2^s * q = p^y - 1) : p = 3 ∨ p = 5 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_equation_l189_18955


namespace NUMINAMATH_CALUDE_area_of_specific_quadrilateral_l189_18999

/-- Represents a convex quadrilateral ABCD with given side lengths and angle properties -/
structure Quadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  angle_CBA_is_right : Bool
  tan_angle_ACD : ℝ

/-- Calculates the area of the quadrilateral ABCD -/
def area (q : Quadrilateral) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific quadrilateral is 122/3 -/
theorem area_of_specific_quadrilateral :
  let q : Quadrilateral := {
    AB := 6,
    BC := 8,
    CD := 5,
    DA := 10,
    angle_CBA_is_right := true,
    tan_angle_ACD := 4/3
  }
  area q = 122/3 := by sorry

end NUMINAMATH_CALUDE_area_of_specific_quadrilateral_l189_18999


namespace NUMINAMATH_CALUDE_log_three_five_l189_18976

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_three_five (a : ℝ) (h : log 5 45 = a) : log 5 3 = (a - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_three_five_l189_18976


namespace NUMINAMATH_CALUDE_max_valid_triples_l189_18941

/-- A function that checks if four positive integers can be arranged in a circle
    with all neighbors being coprime -/
def can_arrange_coprime (a₁ a₂ a₃ a₄ : ℕ+) : Prop :=
  (Nat.gcd a₁.val a₂.val = 1 ∧ Nat.gcd a₂.val a₃.val = 1 ∧ Nat.gcd a₃.val a₄.val = 1 ∧ Nat.gcd a₄.val a₁.val = 1) ∨
  (Nat.gcd a₁.val a₂.val = 1 ∧ Nat.gcd a₂.val a₄.val = 1 ∧ Nat.gcd a₄.val a₃.val = 1 ∧ Nat.gcd a₃.val a₁.val = 1) ∨
  (Nat.gcd a₁.val a₃.val = 1 ∧ Nat.gcd a₃.val a₂.val = 1 ∧ Nat.gcd a₂.val a₄.val = 1 ∧ Nat.gcd a₄.val a₁.val = 1) ∨
  (Nat.gcd a₁.val a₃.val = 1 ∧ Nat.gcd a₃.val a₄.val = 1 ∧ Nat.gcd a₄.val a₂.val = 1 ∧ Nat.gcd a₂.val a₁.val = 1) ∨
  (Nat.gcd a₁.val a₄.val = 1 ∧ Nat.gcd a₄.val a₂.val = 1 ∧ Nat.gcd a₂.val a₃.val = 1 ∧ Nat.gcd a₃.val a₁.val = 1) ∨
  (Nat.gcd a₁.val a₄.val = 1 ∧ Nat.gcd a₄.val a₃.val = 1 ∧ Nat.gcd a₃.val a₂.val = 1 ∧ Nat.gcd a₂.val a₁.val = 1)

/-- A function that counts the number of valid triples (i,j,k) where (gcd(aᵢ,a_j))² | a_k -/
def count_valid_triples (a₁ a₂ a₃ a₄ : ℕ+) : ℕ :=
  let check (i j k : ℕ+) : Bool :=
    i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ (Nat.gcd i.val j.val)^2 ∣ k.val
  (if check a₁ a₂ a₃ then 1 else 0) +
  (if check a₁ a₂ a₄ then 1 else 0) +
  (if check a₁ a₃ a₂ then 1 else 0) +
  (if check a₁ a₃ a₄ then 1 else 0) +
  (if check a₁ a₄ a₂ then 1 else 0) +
  (if check a₁ a₄ a₃ then 1 else 0) +
  (if check a₂ a₃ a₁ then 1 else 0) +
  (if check a₂ a₃ a₄ then 1 else 0) +
  (if check a₂ a₄ a₁ then 1 else 0) +
  (if check a₂ a₄ a₃ then 1 else 0) +
  (if check a₃ a₄ a₁ then 1 else 0) +
  (if check a₃ a₄ a₂ then 1 else 0)

theorem max_valid_triples (a₁ a₂ a₃ a₄ : ℕ+) :
  ¬(can_arrange_coprime a₁ a₂ a₃ a₄) → count_valid_triples a₁ a₂ a₃ a₄ ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_max_valid_triples_l189_18941


namespace NUMINAMATH_CALUDE_pascal_triangle_51_numbers_l189_18945

theorem pascal_triangle_51_numbers (n : ℕ) : 
  n = 50 → (n.choose 4) = 230150 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_51_numbers_l189_18945


namespace NUMINAMATH_CALUDE_curve_equivalence_l189_18906

-- Define the set of points satisfying the original equation
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + p.2 - 1) * Real.sqrt (p.1^2 + p.2^2 - 4) = 0}

-- Define the set of points on the line outside or on the circle
def L : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 1 ∧ p.1^2 + p.2^2 ≥ 4}

-- Define the set of points on the circle
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

-- Theorem stating the equivalence of the sets
theorem curve_equivalence : S = L ∪ C := by sorry

end NUMINAMATH_CALUDE_curve_equivalence_l189_18906


namespace NUMINAMATH_CALUDE_divisibility_implies_r_value_l189_18987

/-- The polynomial in question -/
def p (x : ℝ) : ℝ := 10 * x^3 - 5 * x^2 - 52 * x + 56

/-- Divisibility condition -/
def is_divisible_by_square (r : ℝ) : Prop :=
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - r)^2 * q x

theorem divisibility_implies_r_value :
  ∀ r : ℝ, is_divisible_by_square r → r = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_implies_r_value_l189_18987


namespace NUMINAMATH_CALUDE_union_M_N_l189_18907

-- Define the universe set U
def U : Set ℝ := {x | -3 ≤ x ∧ x < 2}

-- Define set M
def M : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the complement of N with respect to U
def complement_N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define set N
def N : Set ℝ := U \ complement_N

-- Theorem statement
theorem union_M_N : M ∪ N = {x : ℝ | -3 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_union_M_N_l189_18907


namespace NUMINAMATH_CALUDE_associated_equation_k_range_l189_18916

/-- Definition of an associated equation -/
def is_associated_equation (eq_sol : ℝ) (ineq_set : Set ℝ) : Prop :=
  eq_sol ∈ ineq_set

/-- The system of inequalities -/
def inequality_system (x : ℝ) : Prop :=
  (x - 3) / 2 ≥ x ∧ (2 * x + 5) / 2 > x / 2

/-- The solution set of the system of inequalities -/
def solution_set : Set ℝ :=
  {x | inequality_system x}

/-- The equation 2x - k = 6 -/
def equation (k : ℝ) (x : ℝ) : Prop :=
  2 * x - k = 6

theorem associated_equation_k_range :
  ∀ k : ℝ, (∃ x : ℝ, equation k x ∧ is_associated_equation x solution_set) →
    -16 < k ∧ k ≤ -12 :=
by sorry

end NUMINAMATH_CALUDE_associated_equation_k_range_l189_18916


namespace NUMINAMATH_CALUDE_base5_division_proof_l189_18939

-- Define a function to convert from base 5 to decimal
def base5ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

-- Define a function to convert from decimal to base 5
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

-- Theorem statement
theorem base5_division_proof :
  let dividend := [4, 0, 1, 2]  -- 2104₅ in reverse order
  let divisor := [3, 2]         -- 23₅ in reverse order
  let quotient := [1, 4]        -- 41₅ in reverse order
  (base5ToDecimal dividend) / (base5ToDecimal divisor) = base5ToDecimal quotient :=
by sorry

end NUMINAMATH_CALUDE_base5_division_proof_l189_18939


namespace NUMINAMATH_CALUDE_min_guests_banquet_l189_18914

def total_food : ℝ := 337
def max_food_per_guest : ℝ := 2

theorem min_guests_banquet :
  ∃ (min_guests : ℕ), 
    (min_guests : ℝ) * max_food_per_guest ≥ total_food ∧
    ∀ (n : ℕ), (n : ℝ) * max_food_per_guest ≥ total_food → n ≥ min_guests ∧
    min_guests = 169 :=
by sorry

end NUMINAMATH_CALUDE_min_guests_banquet_l189_18914


namespace NUMINAMATH_CALUDE_linear_correlation_classification_l189_18932

-- Define the relationships
def parent_child_height : ℝ → ℝ := sorry
def cylinder_volume_radius : ℝ → ℝ := sorry
def car_weight_fuel_efficiency : ℝ → ℝ := sorry
def household_income_expenditure : ℝ → ℝ := sorry

-- Define linear correlation
def is_linearly_correlated (f : ℝ → ℝ) : Prop := 
  ∃ (a b : ℝ), ∀ x, f x = a * x + b

-- Theorem statement
theorem linear_correlation_classification :
  is_linearly_correlated parent_child_height ∧
  is_linearly_correlated car_weight_fuel_efficiency ∧
  is_linearly_correlated household_income_expenditure ∧
  ¬ is_linearly_correlated cylinder_volume_radius :=
sorry

end NUMINAMATH_CALUDE_linear_correlation_classification_l189_18932


namespace NUMINAMATH_CALUDE_janet_movie_cost_l189_18913

/-- The cost per minute to film Janet's previous movie -/
def previous_cost_per_minute : ℝ := 5

/-- The length of Janet's previous movie in minutes -/
def previous_movie_length : ℝ := 120

/-- The length of Janet's new movie in minutes -/
def new_movie_length : ℝ := previous_movie_length * 1.6

/-- The cost per minute to film Janet's new movie -/
def new_cost_per_minute : ℝ := 2 * previous_cost_per_minute

/-- The total cost to film Janet's new movie -/
def new_movie_total_cost : ℝ := 1920

theorem janet_movie_cost : 
  new_movie_length * new_cost_per_minute = new_movie_total_cost :=
by sorry

end NUMINAMATH_CALUDE_janet_movie_cost_l189_18913


namespace NUMINAMATH_CALUDE_inequality_proof_l189_18963

theorem inequality_proof (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n > 0) :
  x + n^n / x^n ≥ n + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l189_18963


namespace NUMINAMATH_CALUDE_triangle_angle_side_ratio_l189_18920

/-- In a triangle ABC, if the ratio of angles A:B:C is 3:1:2, then the ratio of sides a:b:c is 2:1:√3 -/
theorem triangle_angle_side_ratio (A B C a b c : ℝ) (h_triangle : A + B + C = π) 
  (h_angle_ratio : A = 3 * B ∧ C = 2 * B) : 
  ∃ (k : ℝ), a = 2 * k ∧ b = k ∧ c = Real.sqrt 3 * k := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_side_ratio_l189_18920


namespace NUMINAMATH_CALUDE_max_perfect_squares_among_products_l189_18934

theorem max_perfect_squares_among_products (a b : ℕ) (h : a ≠ b) : 
  let products := {a * (a + 2), a * b, a * (b + 2), (a + 2) * b, (a + 2) * (b + 2), b * (b + 2)}
  (∃ (s : Finset ℕ), s ⊆ products ∧ (∀ x ∈ s, ∃ y : ℕ, x = y * y) ∧ s.card = 2) ∧
  (∀ (s : Finset ℕ), s ⊆ products → (∀ x ∈ s, ∃ y : ℕ, x = y * y) → s.card ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_max_perfect_squares_among_products_l189_18934


namespace NUMINAMATH_CALUDE_semicircle_in_quarter_circle_l189_18964

theorem semicircle_in_quarter_circle (r : ℝ) (hr : r > 0) :
  let s := r * Real.sqrt 3
  let quarter_circle_area := π * s^2 / 4
  let semicircle_area := π * r^2 / 2
  semicircle_area / quarter_circle_area = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_semicircle_in_quarter_circle_l189_18964


namespace NUMINAMATH_CALUDE_billy_homework_ratio_l189_18977

/-- Represents the number of questions solved in each hour -/
structure QuestionsSolved where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents Billy's homework solving session -/
def BillyHomework (qs : QuestionsSolved) : Prop :=
  qs.third = 132 ∧
  qs.third = 2 * qs.second ∧
  qs.first + qs.second + qs.third = 242

theorem billy_homework_ratio (qs : QuestionsSolved) 
  (h : BillyHomework qs) : qs.third / qs.first = 3 := by
  sorry

end NUMINAMATH_CALUDE_billy_homework_ratio_l189_18977


namespace NUMINAMATH_CALUDE_johns_weekly_water_intake_l189_18924

/-- Proves that John drinks 42 quarts of water in a week -/
theorem johns_weekly_water_intake :
  let daily_intake_gallons : ℚ := 3/2
  let gallons_to_quarts : ℚ := 4
  let days_in_week : ℕ := 7
  let weekly_intake_quarts : ℚ := daily_intake_gallons * gallons_to_quarts * days_in_week
  weekly_intake_quarts = 42 := by
  sorry

end NUMINAMATH_CALUDE_johns_weekly_water_intake_l189_18924


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l189_18978

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(1 + b²/a²) -/
theorem hyperbola_eccentricity (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let e := Real.sqrt (1 + b^2 / a^2)
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  hyperbola = fun (x y : ℝ) ↦ x^2 / 4 - y^2 / 5 = 1 →
  e = 3/2 := by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l189_18978


namespace NUMINAMATH_CALUDE_grid_has_ten_rows_l189_18967

/-- Represents a grid of colored squares. -/
structure ColoredGrid where
  squares_per_row : ℕ
  red_squares : ℕ
  blue_squares : ℕ
  green_squares : ℕ

/-- Calculates the number of rows in the grid. -/
def number_of_rows (grid : ColoredGrid) : ℕ :=
  (grid.red_squares + grid.blue_squares + grid.green_squares) / grid.squares_per_row

/-- Theorem stating that a grid with the given properties has 10 rows. -/
theorem grid_has_ten_rows (grid : ColoredGrid) 
  (h1 : grid.squares_per_row = 15)
  (h2 : grid.red_squares = 24)
  (h3 : grid.blue_squares = 60)
  (h4 : grid.green_squares = 66) : 
  number_of_rows grid = 10 := by
  sorry

end NUMINAMATH_CALUDE_grid_has_ten_rows_l189_18967


namespace NUMINAMATH_CALUDE_distance_foci_to_asymptotes_for_given_hyperbola_l189_18931

/-- The distance from the foci to the asymptotes of a hyperbola -/
def distance_foci_to_asymptotes (a b : ℝ) : ℝ := b

/-- The equation of a hyperbola in standard form -/
def is_hyperbola (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

theorem distance_foci_to_asymptotes_for_given_hyperbola :
  ∀ x y : ℝ, is_hyperbola x y 1 3 → distance_foci_to_asymptotes 1 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_foci_to_asymptotes_for_given_hyperbola_l189_18931


namespace NUMINAMATH_CALUDE_tan_difference_pi_12_pi_6_l189_18960

theorem tan_difference_pi_12_pi_6 : 
  Real.tan (π / 12) - Real.tan (π / 6) = 7 - 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_pi_12_pi_6_l189_18960


namespace NUMINAMATH_CALUDE_work_completion_time_l189_18926

theorem work_completion_time (a_time b_time joint_time b_remaining_time : ℝ) 
  (ha : a_time = 45)
  (hjoint : joint_time = 9)
  (hb_remaining : b_remaining_time = 23)
  (h_work_rate : (joint_time * (1 / a_time + 1 / b_time)) + 
                 (b_remaining_time * (1 / b_time)) = 1) :
  b_time = 40 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l189_18926


namespace NUMINAMATH_CALUDE_remaining_safe_caffeine_l189_18919

/-- The maximum safe amount of caffeine that can be consumed per day in milligrams. -/
def max_safe_caffeine : ℕ := 500

/-- The amount of caffeine in each energy drink in milligrams. -/
def caffeine_per_drink : ℕ := 120

/-- The number of energy drinks Brandy consumes. -/
def drinks_consumed : ℕ := 4

/-- The remaining safe amount of caffeine Brandy can consume that day in milligrams. -/
theorem remaining_safe_caffeine : 
  max_safe_caffeine - (caffeine_per_drink * drinks_consumed) = 20 := by
  sorry

end NUMINAMATH_CALUDE_remaining_safe_caffeine_l189_18919


namespace NUMINAMATH_CALUDE_teachers_daughter_age_l189_18990

theorem teachers_daughter_age 
  (P : ℤ → ℤ)  -- P is a function from integers to integers
  (a : ℕ)      -- a is a natural number (age)
  (p : ℕ)      -- p is a natural number (prime)
  (h_poly : ∀ x y : ℤ, (x - y) ∣ (P x - P y))  -- P is a polynomial with integer coefficients
  (h_pa : P a = a)     -- P(a) = a
  (h_p0 : P 0 = p)     -- P(0) = p
  (h_prime : Nat.Prime p)  -- p is prime
  (h_p_gt_a : p > a)   -- p > a
  : a = 1 := by
sorry

end NUMINAMATH_CALUDE_teachers_daughter_age_l189_18990


namespace NUMINAMATH_CALUDE_sequence_convergence_l189_18998

def S (seq : List Int) : List Int :=
  let n := seq.length
  List.zipWith (· * ·) seq ((seq.drop 1).append [seq.head!])

def all_ones (seq : List Int) : Prop :=
  seq.all (· = 1)

theorem sequence_convergence (n : Nat) (seq : List Int) :
  seq.length = 2 * n →
  (∀ i, i ∈ seq → i = 1 ∨ i = -1) →
  ∃ k : Nat, all_ones (Nat.iterate S k seq) := by
  sorry

end NUMINAMATH_CALUDE_sequence_convergence_l189_18998


namespace NUMINAMATH_CALUDE_book_reading_permutations_l189_18980

theorem book_reading_permutations :
  let n : ℕ := 5  -- total number of books
  let r : ℕ := 3  -- number of books to read
  Nat.factorial n / Nat.factorial (n - r) = 60 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_permutations_l189_18980


namespace NUMINAMATH_CALUDE_drums_per_day_l189_18930

/-- Given that 2916 drums of grapes are filled in 9 days, 
    prove that 324 drums are filled per day. -/
theorem drums_per_day : 
  ∀ (total_drums : ℕ) (total_days : ℕ) (drums_per_day : ℕ),
    total_drums = 2916 →
    total_days = 9 →
    drums_per_day = total_drums / total_days →
    drums_per_day = 324 := by
  sorry

end NUMINAMATH_CALUDE_drums_per_day_l189_18930


namespace NUMINAMATH_CALUDE_identity_is_unique_satisfying_function_l189_18988

/-- A function satisfying the given property -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, x^3 + f y * x + f z = 0 → f x^3 + y * f x + z = 0

/-- The main theorem stating that the identity function is the only function satisfying the property -/
theorem identity_is_unique_satisfying_function :
  ∀ f : ℝ → ℝ, SatisfyingFunction f → f = id := by sorry

end NUMINAMATH_CALUDE_identity_is_unique_satisfying_function_l189_18988


namespace NUMINAMATH_CALUDE_sqrt_transformation_l189_18984

theorem sqrt_transformation (n : ℕ) (h : n ≥ 1) : 
  Real.sqrt ((1 : ℝ) / n * ((1 : ℝ) / (n + 1) - (1 : ℝ) / (n + 2))) = 
  (1 : ℝ) / (n + 1) * Real.sqrt ((n + 1 : ℝ) / (n * (n + 2))) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_transformation_l189_18984


namespace NUMINAMATH_CALUDE_radio_loss_percentage_l189_18905

/-- Calculates the loss percentage given the cost price and selling price -/
def loss_percentage (cost_price selling_price : ℕ) : ℚ :=
  (cost_price - selling_price : ℚ) / cost_price * 100

theorem radio_loss_percentage :
  let cost_price := 1500
  let selling_price := 1260
  loss_percentage cost_price selling_price = 16 := by
sorry

end NUMINAMATH_CALUDE_radio_loss_percentage_l189_18905


namespace NUMINAMATH_CALUDE_workshop_workers_count_workshop_workers_count_proof_l189_18938

/-- Given a workshop with workers, prove that the total number of workers is 28 -/
theorem workshop_workers_count : ℕ :=
  let total_average : ℚ := 8000
  let technician_count : ℕ := 7
  let technician_average : ℚ := 14000
  let non_technician_average : ℚ := 6000
  28

/-- Proof of the theorem -/
theorem workshop_workers_count_proof :
  let total_average : ℚ := 8000
  let technician_count : ℕ := 7
  let technician_average : ℚ := 14000
  let non_technician_average : ℚ := 6000
  workshop_workers_count = 28 := by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_count_workshop_workers_count_proof_l189_18938


namespace NUMINAMATH_CALUDE_radius_circle_q_is_ten_l189_18903

/-- A triangle ABC with two equal sides and a circle P tangent to two sides -/
structure IsoscelesTriangleWithTangentCircle where
  /-- The length of the equal sides AB and AC -/
  side_length : ℝ
  /-- The length of the base BC -/
  base_length : ℝ
  /-- The radius of the circle P tangent to AC and BC -/
  circle_p_radius : ℝ

/-- The radius of circle Q, which is externally tangent to P and tangent to AB and BC -/
def radius_circle_q (t : IsoscelesTriangleWithTangentCircle) : ℝ := sorry

/-- The main theorem: In the given configuration, the radius of circle Q is 10 -/
theorem radius_circle_q_is_ten
  (t : IsoscelesTriangleWithTangentCircle)
  (h1 : t.side_length = 120)
  (h2 : t.base_length = 90)
  (h3 : t.circle_p_radius = 30) :
  radius_circle_q t = 10 := by sorry

end NUMINAMATH_CALUDE_radius_circle_q_is_ten_l189_18903


namespace NUMINAMATH_CALUDE_floor_product_equals_twelve_l189_18995

theorem floor_product_equals_twelve (x : ℝ) : 
  ⌊x * ⌊x / 2⌋⌋ = 12 ↔ x ≥ 4.9 ∧ x < 5.1 := by sorry

end NUMINAMATH_CALUDE_floor_product_equals_twelve_l189_18995


namespace NUMINAMATH_CALUDE_worksheets_graded_l189_18968

theorem worksheets_graded (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (problems_left : ℕ) : 
  total_worksheets = 15 →
  problems_per_worksheet = 3 →
  problems_left = 24 →
  (total_worksheets * problems_per_worksheet - problems_left) / problems_per_worksheet = 7 := by
sorry

end NUMINAMATH_CALUDE_worksheets_graded_l189_18968


namespace NUMINAMATH_CALUDE_min_value_theorem_l189_18949

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1/a + 4/b ≥ 9/2 := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l189_18949


namespace NUMINAMATH_CALUDE_T_is_three_rays_with_common_point_l189_18981

/-- The set T of points in the coordinate plane satisfying the given conditions -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x - 1 ∧ y + 3 ≤ 5) ∨
               (5 = y + 3 ∧ x - 1 ≤ 5) ∨
               (x - 1 = y + 3 ∧ 5 ≤ x - 1)}

/-- A ray in the coordinate plane -/
structure Ray where
  start : ℝ × ℝ
  direction : ℝ × ℝ

/-- The three rays that make up set T -/
def rays : List Ray :=
  [{ start := (6, 2), direction := (0, -1) },   -- Vertical ray downward
   { start := (6, 2), direction := (-1, 0) },   -- Horizontal ray leftward
   { start := (6, 2), direction := (1, 1) }]    -- Diagonal ray upward

/-- Theorem stating that T consists of three rays with a common point -/
theorem T_is_three_rays_with_common_point :
  ∃ (common_point : ℝ × ℝ) (rs : List Ray),
    common_point = (6, 2) ∧
    rs.length = 3 ∧
    (∀ r ∈ rs, r.start = common_point) ∧
    T = ⋃ r ∈ rs, {p | ∃ t ≥ 0, p = r.start + t • r.direction} :=
  sorry

end NUMINAMATH_CALUDE_T_is_three_rays_with_common_point_l189_18981


namespace NUMINAMATH_CALUDE_read_time_is_two_hours_l189_18979

/-- Calculates the time taken to read a given number of pages at an increased reading speed. -/
def time_to_read (normal_speed : ℕ) (speed_increase : ℕ) (total_pages : ℕ) : ℚ :=
  total_pages / (normal_speed * speed_increase)

/-- Theorem stating that given the conditions from the problem, the time taken to read is 2 hours. -/
theorem read_time_is_two_hours (normal_speed : ℕ) (speed_increase : ℕ) (total_pages : ℕ)
  (h1 : normal_speed = 12)
  (h2 : speed_increase = 3)
  (h3 : total_pages = 72) :
  time_to_read normal_speed speed_increase total_pages = 2 := by
  sorry

end NUMINAMATH_CALUDE_read_time_is_two_hours_l189_18979


namespace NUMINAMATH_CALUDE_least_multiple_75_with_digit_product_75_l189_18986

def is_multiple_of_75 (n : ℕ) : Prop := ∃ k : ℕ, n = 75 * k

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_multiple_of_75 n ∧ is_multiple_of_75 (digit_product n)

theorem least_multiple_75_with_digit_product_75 :
  satisfies_conditions 75375 ∧ ∀ m : ℕ, m < 75375 → ¬(satisfies_conditions m) :=
sorry

end NUMINAMATH_CALUDE_least_multiple_75_with_digit_product_75_l189_18986


namespace NUMINAMATH_CALUDE_nancy_folders_l189_18902

theorem nancy_folders (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) : 
  initial_files = 80 → deleted_files = 31 → files_per_folder = 7 → 
  (initial_files - deleted_files) / files_per_folder = 7 := by
  sorry

end NUMINAMATH_CALUDE_nancy_folders_l189_18902


namespace NUMINAMATH_CALUDE_no_square_in_triangle_grid_l189_18996

/-- A point in the plane represented by its coordinates -/
structure Point where
  x : ℚ
  y : ℝ

/-- The grid of equilateral triangles -/
structure TriangleGrid where
  side_length : ℝ
  is_valid : side_length > 0

/-- A function that checks if a point is a valid vertex in the triangle grid -/
def is_vertex (grid : TriangleGrid) (p : Point) : Prop :=
  ∃ (k l : ℤ), p.x = k * (grid.side_length / 2) ∧ p.y = l * (Real.sqrt 3 * grid.side_length / 2)

/-- A function that checks if four points form a square -/
def is_square (a b c d : Point) : Prop :=
  let dist (p q : Point) := Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)
  (dist a b = dist b c) ∧ (dist b c = dist c d) ∧ (dist c d = dist d a) ∧
  (dist a c = dist b d) ∧
  ((b.x - a.x) * (c.x - b.x) + (b.y - a.y) * (c.y - b.y) = 0)

/-- The main theorem stating that it's impossible to form a square from vertices of the triangle grid -/
theorem no_square_in_triangle_grid (grid : TriangleGrid) :
  ¬∃ (a b c d : Point), is_vertex grid a ∧ is_vertex grid b ∧ is_vertex grid c ∧ is_vertex grid d ∧ is_square a b c d :=
sorry

end NUMINAMATH_CALUDE_no_square_in_triangle_grid_l189_18996
