import Mathlib

namespace NUMINAMATH_CALUDE_factorial_ratio_equals_119_factorial_l3487_348760

theorem factorial_ratio_equals_119_factorial : (Nat.factorial (Nat.factorial 5)) / (Nat.factorial 5) = Nat.factorial 119 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_119_factorial_l3487_348760


namespace NUMINAMATH_CALUDE_greatest_3digit_base9_divisible_by_7_l3487_348708

/-- Converts a base 9 number to decimal --/
def base9ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (9 ^ i)) 0

/-- Checks if a number is a valid 3-digit base 9 number --/
def isValid3DigitBase9 (n : Nat) : Prop :=
  ∃ (d₁ d₂ d₃ : Nat), d₁ ≠ 0 ∧ d₁ < 9 ∧ d₂ < 9 ∧ d₃ < 9 ∧ n = base9ToDecimal [d₃, d₂, d₁]

theorem greatest_3digit_base9_divisible_by_7 :
  let n := base9ToDecimal [8, 8, 8]
  isValid3DigitBase9 n ∧ n % 7 = 0 ∧
  ∀ m, isValid3DigitBase9 m → m % 7 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_3digit_base9_divisible_by_7_l3487_348708


namespace NUMINAMATH_CALUDE_slope_MN_constant_l3487_348771

/-- Curve C defined by y² = 4x -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Point D on curve C -/
def D : ℝ × ℝ := (1, 2)

/-- Line with slope k passing through D -/
def line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * (p.1 - D.1) + D.2}

/-- Theorem: The slope of line MN is constant -/
theorem slope_MN_constant (k : ℝ) :
  k ≠ 0 →
  D ∈ C →
  ∃ (M N : ℝ × ℝ),
    M ∈ C ∧ M ∈ line k ∧
    N ∈ C ∧ N ∈ line (-1/k) ∧
    M ≠ D ∧ N ≠ D →
    (N.2 - M.2) / (N.1 - M.1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_slope_MN_constant_l3487_348771


namespace NUMINAMATH_CALUDE_room_width_is_two_l3487_348700

-- Define the room's properties
def room_area : ℝ := 10
def room_length : ℝ := 5

-- Theorem statement
theorem room_width_is_two : 
  ∃ (width : ℝ), room_area = room_length * width ∧ width = 2 := by
  sorry

end NUMINAMATH_CALUDE_room_width_is_two_l3487_348700


namespace NUMINAMATH_CALUDE_elite_academy_games_l3487_348758

/-- The number of teams in the Elite Academy Basketball League -/
def num_teams : ℕ := 8

/-- The number of times each team plays every other team -/
def games_per_pair : ℕ := 3

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 3

/-- The total number of games in a season for the Elite Academy Basketball League -/
def total_games : ℕ := (num_teams.choose 2 * games_per_pair) + (num_teams * non_conference_games)

theorem elite_academy_games :
  total_games = 108 := by sorry

end NUMINAMATH_CALUDE_elite_academy_games_l3487_348758


namespace NUMINAMATH_CALUDE_alex_jellybean_possibilities_l3487_348782

def total_money : ℕ := 100  -- in pence

def toffee_price : ℕ := 5
def bubblegum_price : ℕ := 3
def jellybean_price : ℕ := 2

def min_toffee_spend : ℕ := 35  -- ⌈100 / 3⌉ rounded up to nearest multiple of 5
def min_bubblegum_spend : ℕ := 27  -- ⌈100 / 4⌉ rounded up to nearest multiple of 3
def min_jellybean_spend : ℕ := 10  -- 100 / 10

def possible_jellybean_counts : Set ℕ := {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19}

theorem alex_jellybean_possibilities :
  ∀ n : ℕ, n ∈ possible_jellybean_counts ↔
    ∃ (t b j : ℕ),
      t * toffee_price + b * bubblegum_price + n * jellybean_price = total_money ∧
      t * toffee_price ≥ min_toffee_spend ∧
      b * bubblegum_price ≥ min_bubblegum_spend ∧
      n * jellybean_price ≥ min_jellybean_spend :=
by sorry

end NUMINAMATH_CALUDE_alex_jellybean_possibilities_l3487_348782


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3487_348754

-- Define the quadratic equation
def quadratic_equation (x k : ℝ) : Prop :=
  x^2 + 2*x - k = 0

-- Define the condition for real roots
def has_real_roots (k : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_equation x k

-- Theorem statement
theorem quadratic_real_roots_condition (k : ℝ) :
  has_real_roots k ↔ k ≥ -1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3487_348754


namespace NUMINAMATH_CALUDE_sqrt_sum_greater_than_sqrt_of_sum_l3487_348761

theorem sqrt_sum_greater_than_sqrt_of_sum {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  Real.sqrt a + Real.sqrt b > Real.sqrt (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_greater_than_sqrt_of_sum_l3487_348761


namespace NUMINAMATH_CALUDE_percentage_of_men_speaking_french_l3487_348733

theorem percentage_of_men_speaking_french (E : ℝ) (E_pos : E > 0) :
  let men_percentage : ℝ := 70
  let french_speaking_percentage : ℝ := 40
  let women_not_speaking_french_percentage : ℝ := 83.33333333333331
  let men_count : ℝ := (men_percentage / 100) * E
  let french_speaking_count : ℝ := (french_speaking_percentage / 100) * E
  let women_count : ℝ := E - men_count
  let women_speaking_french_count : ℝ := (1 - women_not_speaking_french_percentage / 100) * women_count
  let men_speaking_french_count : ℝ := french_speaking_count - women_speaking_french_count
  (men_speaking_french_count / men_count) * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_men_speaking_french_l3487_348733


namespace NUMINAMATH_CALUDE_winning_strategy_works_l3487_348750

/-- Represents the game state with blue and white balls --/
structure GameState where
  blue : ℕ
  white : ℕ

/-- Represents a player's move --/
inductive Move
  | TakeBlue
  | TakeWhite

/-- Applies a move to the game state --/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.TakeBlue => { blue := state.blue - 3, white := state.white }
  | Move.TakeWhite => { blue := state.blue, white := state.white - 2 }

/-- Checks if the game is over (no balls left) --/
def isGameOver (state : GameState) : Prop :=
  state.blue = 0 ∧ state.white = 0

/-- Represents the winning strategy --/
def winningStrategy (state : GameState) : Prop :=
  3 * state.white = 2 * state.blue

/-- The main theorem to prove --/
theorem winning_strategy_works (initialState : GameState)
  (h_initial : initialState.blue = 15 ∧ initialState.white = 12) :
  ∃ (firstMove : Move),
    let stateAfterFirstMove := applyMove initialState firstMove
    winningStrategy stateAfterFirstMove ∧
    (∀ (opponentMove : Move),
      let stateAfterOpponent := applyMove stateAfterFirstMove opponentMove
      ∃ (response : Move),
        let stateAfterResponse := applyMove stateAfterOpponent response
        winningStrategy stateAfterResponse ∨ isGameOver stateAfterResponse) :=
sorry

end NUMINAMATH_CALUDE_winning_strategy_works_l3487_348750


namespace NUMINAMATH_CALUDE_range_of_4x_plus_2y_l3487_348725

theorem range_of_4x_plus_2y (x y : ℝ) 
  (h1 : 1 ≤ x + y) (h2 : x + y ≤ 3) 
  (h3 : -1 ≤ x - y) (h4 : x - y ≤ 1) : 
  2 ≤ 4*x + 2*y ∧ 4*x + 2*y ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_range_of_4x_plus_2y_l3487_348725


namespace NUMINAMATH_CALUDE_puppies_per_dog_l3487_348755

theorem puppies_per_dog (num_dogs : ℕ) (sold_fraction : ℚ) (price_per_puppy : ℕ) (total_revenue : ℕ) :
  num_dogs = 2 →
  sold_fraction = 3 / 4 →
  price_per_puppy = 200 →
  total_revenue = 3000 →
  (total_revenue / price_per_puppy : ℚ) / sold_fraction / num_dogs = 10 :=
by sorry

end NUMINAMATH_CALUDE_puppies_per_dog_l3487_348755


namespace NUMINAMATH_CALUDE_double_money_in_20_years_l3487_348795

/-- The simple interest rate that doubles a sum of money in 20 years -/
def double_money_rate : ℚ := 5 / 100

/-- The time period in years -/
def time_period : ℕ := 20

/-- The final amount after applying simple interest -/
def final_amount (principal : ℚ) : ℚ :=
  principal * (1 + double_money_rate * time_period)

theorem double_money_in_20_years (principal : ℚ) (h : principal > 0) :
  final_amount principal = 2 * principal := by
  sorry

#check double_money_in_20_years

end NUMINAMATH_CALUDE_double_money_in_20_years_l3487_348795


namespace NUMINAMATH_CALUDE_lcm_of_numbers_with_given_hcf_and_product_l3487_348720

theorem lcm_of_numbers_with_given_hcf_and_product (a b : ℕ+) : 
  Nat.gcd a b = 16 → 
  a * b = 2560 → 
  Nat.lcm a b = 160 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_numbers_with_given_hcf_and_product_l3487_348720


namespace NUMINAMATH_CALUDE_sum_of_B_coordinates_l3487_348702

-- Define the points
def A : ℝ × ℝ := (5, -1)
def M : ℝ × ℝ := (4, 3)

-- Define B as a variable point
variable (B : ℝ × ℝ)

-- Define the midpoint condition
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Theorem statement
theorem sum_of_B_coordinates :
  is_midpoint M A B → B.1 + B.2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_B_coordinates_l3487_348702


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l3487_348764

/-- 
Given a quadratic equation (m-1)x^2 - 2mx + m + 3 = 0,
prove that it has two real roots if and only if m ≤ 3/2 and m ≠ 1.
-/
theorem quadratic_two_real_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    (m - 1) * x^2 - 2 * m * x + m + 3 = 0 ∧ 
    (m - 1) * y^2 - 2 * m * y + m + 3 = 0) ↔ 
  (m ≤ 3/2 ∧ m ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l3487_348764


namespace NUMINAMATH_CALUDE_minimal_vertices_2007_gon_l3487_348711

/-- Given a regular polygon with n sides, returns the minimal number k such that
    among every k vertices of the polygon, there always exists 4 vertices forming
    a convex quadrilateral with 3 sides being sides of the polygon. -/
def minimalVerticesForQuadrilateral (n : ℕ) : ℕ :=
  ⌈(3 * n : ℚ) / 4⌉₊

theorem minimal_vertices_2007_gon :
  minimalVerticesForQuadrilateral 2007 = 1506 := by
  sorry

#eval minimalVerticesForQuadrilateral 2007

end NUMINAMATH_CALUDE_minimal_vertices_2007_gon_l3487_348711


namespace NUMINAMATH_CALUDE_smallest_product_l3487_348704

def S : Finset ℤ := {-9, -5, -1, 1, 4}

theorem smallest_product (a b : ℤ) (ha : a ∈ S) (hb : b ∈ S) (hab : a ≠ b) :
  ∃ (x y : ℤ) (hx : x ∈ S) (hy : y ∈ S) (hxy : x ≠ y), 
    x * y ≤ a * b ∧ x * y = -36 := by
  sorry

end NUMINAMATH_CALUDE_smallest_product_l3487_348704


namespace NUMINAMATH_CALUDE_negative_less_than_positive_l3487_348757

theorem negative_less_than_positive : ∀ x y : ℝ, x < 0 → 0 < y → x < y := by sorry

end NUMINAMATH_CALUDE_negative_less_than_positive_l3487_348757


namespace NUMINAMATH_CALUDE_notebook_length_for_12cm_span_l3487_348763

/-- Given a hand span and a notebook with a long side twice the span, calculate the length of the notebook's long side. -/
def notebook_length (hand_span : ℝ) : ℝ := 2 * hand_span

/-- Theorem stating that for a hand span of 12 cm, the notebook's long side is 24 cm. -/
theorem notebook_length_for_12cm_span :
  notebook_length 12 = 24 := by sorry

end NUMINAMATH_CALUDE_notebook_length_for_12cm_span_l3487_348763


namespace NUMINAMATH_CALUDE_movie_spending_ratio_l3487_348751

/-- Proves that the ratio of movie spending to weekly allowance is 1:2 --/
theorem movie_spending_ratio (weekly_allowance car_wash_earnings final_amount : ℕ) :
  weekly_allowance = 8 →
  car_wash_earnings = 8 →
  final_amount = 12 →
  (weekly_allowance + car_wash_earnings - final_amount) / weekly_allowance = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_movie_spending_ratio_l3487_348751


namespace NUMINAMATH_CALUDE_no_adjacent_women_correct_a_not_first_b_not_last_correct_fixed_sequence_correct_a_left_of_b_correct_l3487_348726

def num_men : Nat := 4
def num_women : Nat := 3
def total_people : Nat := num_men + num_women

-- Function to calculate the number of arrangements where no two women are adjacent
def arrangements_no_adjacent_women : Nat :=
  Nat.factorial num_men * Nat.descFactorial (num_men + 1) num_women

-- Function to calculate the number of arrangements where Man A is not first and Man B is not last
def arrangements_a_not_first_b_not_last : Nat :=
  Nat.factorial total_people - 2 * Nat.factorial (total_people - 1) + Nat.factorial (total_people - 2)

-- Function to calculate the number of arrangements where Men A, B, and C are in a fixed sequence
def arrangements_fixed_sequence : Nat :=
  Nat.factorial total_people / Nat.factorial 3

-- Function to calculate the number of arrangements where Man A is to the left of Man B
def arrangements_a_left_of_b : Nat :=
  Nat.factorial total_people / 2

-- Theorems to prove
theorem no_adjacent_women_correct :
  arrangements_no_adjacent_women = 1440 := by sorry

theorem a_not_first_b_not_last_correct :
  arrangements_a_not_first_b_not_last = 3720 := by sorry

theorem fixed_sequence_correct :
  arrangements_fixed_sequence = 840 := by sorry

theorem a_left_of_b_correct :
  arrangements_a_left_of_b = 2520 := by sorry

end NUMINAMATH_CALUDE_no_adjacent_women_correct_a_not_first_b_not_last_correct_fixed_sequence_correct_a_left_of_b_correct_l3487_348726


namespace NUMINAMATH_CALUDE_max_value_expression_l3487_348748

theorem max_value_expression (x y z : ℝ) (h1 : x + 2*y + z = 7) (h2 : y ≥ 0) :
  ∃ M : ℝ, M = (10.5 : ℝ) ∧ ∀ x' y' z' : ℝ, x' + 2*y' + z' = 7 → y' ≥ 0 →
    x'*y' + x'*z' + y'*z' + y'^2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l3487_348748


namespace NUMINAMATH_CALUDE_T_values_l3487_348735

theorem T_values (θ : Real) :
  (∃ T : Real, T = Real.sqrt (1 + Real.sin (2 * θ))) →
  (((Real.sin (π - θ) = 3/5 ∧ π/2 < θ ∧ θ < π) →
    Real.sqrt (1 + Real.sin (2 * θ)) = 1/5) ∧
   ((Real.cos (π/2 - θ) = m ∧ π/2 < θ ∧ θ < 3*π/4) →
    Real.sqrt (1 + Real.sin (2 * θ)) = m - Real.sqrt (1 - m^2)) ∧
   ((Real.cos (π/2 - θ) = m ∧ 3*π/4 < θ ∧ θ < π) →
    Real.sqrt (1 + Real.sin (2 * θ)) = -m + Real.sqrt (1 - m^2))) :=
by sorry

end NUMINAMATH_CALUDE_T_values_l3487_348735


namespace NUMINAMATH_CALUDE_existence_of_xy_for_function_l3487_348744

open Set

theorem existence_of_xy_for_function (f : ℝ → ℝ) 
  (hf : ∀ x, x > 0 → f x > 0) : 
  ∃ x y, x > 0 ∧ y > 0 ∧ f (x + y) < y * f (f x) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_xy_for_function_l3487_348744


namespace NUMINAMATH_CALUDE_beidou_satellite_altitude_scientific_notation_l3487_348775

theorem beidou_satellite_altitude_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 21500000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.15 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_beidou_satellite_altitude_scientific_notation_l3487_348775


namespace NUMINAMATH_CALUDE_general_admission_price_l3487_348778

/-- Proves that the price of a general admission seat is $21.85 -/
theorem general_admission_price : 
  ∀ (total_tickets : ℕ) 
    (total_revenue : ℚ) 
    (vip_price : ℚ) 
    (gen_price : ℚ),
  total_tickets = 320 →
  total_revenue = 7500 →
  vip_price = 45 →
  (∃ (vip_tickets gen_tickets : ℕ),
    vip_tickets + gen_tickets = total_tickets ∧
    vip_tickets = gen_tickets - 276 ∧
    vip_price * vip_tickets + gen_price * gen_tickets = total_revenue) →
  gen_price = 21.85 := by
sorry


end NUMINAMATH_CALUDE_general_admission_price_l3487_348778


namespace NUMINAMATH_CALUDE_total_capital_calculation_l3487_348732

/-- Represents the total capital at the end of the first year given an initial investment and profit rate. -/
def totalCapitalEndOfYear (initialInvestment : ℝ) (profitRate : ℝ) : ℝ :=
  initialInvestment * (1 + profitRate)

/-- Theorem stating that for an initial investment of 50 ten thousand yuan and profit rate P,
    the total capital at the end of the first year is 50(1+P) ten thousand yuan. -/
theorem total_capital_calculation (P : ℝ) :
  totalCapitalEndOfYear 50 P = 50 * (1 + P) := by
  sorry

end NUMINAMATH_CALUDE_total_capital_calculation_l3487_348732


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l3487_348783

theorem square_area_from_diagonal (d : ℝ) (h : d = 10) : 
  (d^2 / 2 : ℝ) = 50 := by
  sorry

#check square_area_from_diagonal

end NUMINAMATH_CALUDE_square_area_from_diagonal_l3487_348783


namespace NUMINAMATH_CALUDE_base_conversion_l3487_348729

theorem base_conversion (b : ℝ) : 
  b > 0 ∧ (5 * 6 + 4 = 1 * b^2 + 2 * b + 1) → b = -1 + Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_l3487_348729


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3487_348719

theorem imaginary_part_of_z (z : ℂ) (h : z * (2 + Complex.I) = 1) :
  z.im = -1/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3487_348719


namespace NUMINAMATH_CALUDE_one_thirds_in_nine_thirds_l3487_348747

theorem one_thirds_in_nine_thirds : (9 : ℚ) / 3 / (1 : ℚ) / 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_one_thirds_in_nine_thirds_l3487_348747


namespace NUMINAMATH_CALUDE_alexander_rearrangements_l3487_348713

theorem alexander_rearrangements (name_length : ℕ) (rearrangements_per_minute : ℕ) : 
  name_length = 9 → rearrangements_per_minute = 15 → 
  (Nat.factorial name_length / rearrangements_per_minute : ℚ) / 60 = 403.2 := by
  sorry

end NUMINAMATH_CALUDE_alexander_rearrangements_l3487_348713


namespace NUMINAMATH_CALUDE_cubed_49_plus_1_l3487_348791

theorem cubed_49_plus_1 : 49^3 + 3*(49^2) + 3*49 + 1 = 125000 := by
  sorry

end NUMINAMATH_CALUDE_cubed_49_plus_1_l3487_348791


namespace NUMINAMATH_CALUDE_polynomial_division_result_l3487_348788

-- Define the polynomials f and d
def f (x : ℝ) : ℝ := 3 * x^4 - 9 * x^3 + 6 * x^2 + 2 * x - 5
def d (x : ℝ) : ℝ := x^2 - 2 * x + 1

-- State the theorem
theorem polynomial_division_result :
  ∃ (q r : ℝ → ℝ), 
    (∀ x, f x = q x * d x + r x) ∧ 
    (∀ x, r x = 14) ∧
    (q 1 + r (-1) = 17) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_result_l3487_348788


namespace NUMINAMATH_CALUDE_first_month_sale_l3487_348707

theorem first_month_sale (sales_2 sales_3 sales_4 sales_5 sales_6 : ℕ)
  (h1 : sales_2 = 6927)
  (h2 : sales_3 = 6855)
  (h3 : sales_4 = 7230)
  (h4 : sales_5 = 6562)
  (h5 : sales_6 = 4791)
  (desired_average : ℕ)
  (h6 : desired_average = 6500)
  (num_months : ℕ)
  (h7 : num_months = 6) :
  ∃ (sales_1 : ℕ), sales_1 = 6635 ∧
    (sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / num_months = desired_average :=
by
  sorry

end NUMINAMATH_CALUDE_first_month_sale_l3487_348707


namespace NUMINAMATH_CALUDE_license_plate_count_l3487_348770

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The total number of characters in the license plate -/
def total_chars : ℕ := 8

/-- The number of digits in the license plate -/
def num_plate_digits : ℕ := 5

/-- The number of letters in the license plate -/
def num_plate_letters : ℕ := 3

/-- The number of possible starting positions for the letter block -/
def letter_block_positions : ℕ := 6

/-- Calculates the total number of distinct license plates -/
def total_license_plates : ℕ :=
  letter_block_positions * num_digits ^ num_plate_digits * num_letters ^ num_plate_letters

theorem license_plate_count :
  total_license_plates = 10545600000 := by sorry

end NUMINAMATH_CALUDE_license_plate_count_l3487_348770


namespace NUMINAMATH_CALUDE_president_and_vice_captain_selection_l3487_348780

/-- The number of people to choose from -/
def n : ℕ := 5

/-- The number of positions to fill -/
def k : ℕ := 2

/-- Theorem: The number of ways to select a class president and a vice-captain 
    from a group of n people, where one person cannot hold both positions, 
    is equal to n * (n - 1) -/
theorem president_and_vice_captain_selection : n * (n - 1) = 20 := by
  sorry

end NUMINAMATH_CALUDE_president_and_vice_captain_selection_l3487_348780


namespace NUMINAMATH_CALUDE_number_of_elements_in_set_l3487_348769

theorem number_of_elements_in_set (S : ℝ) (n : ℕ) 
  (h1 : (S + 26) / n = 5)
  (h2 : (S + 36) / n = 6) :
  n = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_of_elements_in_set_l3487_348769


namespace NUMINAMATH_CALUDE_series_sum_equals_three_l3487_348762

theorem series_sum_equals_three (k : ℝ) (hk : k > 1) :
  (∑' n : ℕ, (n^2 + 3*n - 2) / k^n) = 2 → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_three_l3487_348762


namespace NUMINAMATH_CALUDE_square_sum_equals_twenty_l3487_348756

theorem square_sum_equals_twenty (x y : ℝ) 
  (h1 : (x + y)^2 = 36) 
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_twenty_l3487_348756


namespace NUMINAMATH_CALUDE_range_of_m_l3487_348759

theorem range_of_m (m : ℝ) : 
  (∃ (x₁ x₂ x₃ x₄ : ℤ), 
    (∀ x : ℤ, (3 * ↑x - m > 0 ∧ ↑x - 1 ≤ 5) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄))) →
  (6 ≤ m ∧ m < 9) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3487_348759


namespace NUMINAMATH_CALUDE_oliver_dish_count_l3487_348773

/-- Represents the buffet and Oliver's preferences -/
structure Buffet where
  total_dishes : ℕ
  mango_salsa_dishes : ℕ
  fresh_mango_dishes : ℕ
  mango_jelly_dishes : ℕ
  strawberry_dishes : ℕ
  pineapple_dishes : ℕ
  mango_dishes_oliver_can_eat : ℕ

/-- Calculates the number of dishes Oliver can eat -/
def dishes_for_oliver (b : Buffet) : ℕ :=
  b.total_dishes -
  (b.mango_salsa_dishes + b.fresh_mango_dishes + b.mango_jelly_dishes - b.mango_dishes_oliver_can_eat) -
  min b.strawberry_dishes b.pineapple_dishes

/-- Theorem stating the number of dishes Oliver can eat -/
theorem oliver_dish_count (b : Buffet) : dishes_for_oliver b = 28 :=
  by
    have h1 : b.total_dishes = 42 := by sorry
    have h2 : b.mango_salsa_dishes = 5 := by sorry
    have h3 : b.fresh_mango_dishes = 7 := by sorry
    have h4 : b.mango_jelly_dishes = 2 := by sorry
    have h5 : b.strawberry_dishes = 3 := by sorry
    have h6 : b.pineapple_dishes = 5 := by sorry
    have h7 : b.mango_dishes_oliver_can_eat = 3 := by sorry
    sorry

#eval dishes_for_oliver {
  total_dishes := 42,
  mango_salsa_dishes := 5,
  fresh_mango_dishes := 7,
  mango_jelly_dishes := 2,
  strawberry_dishes := 3,
  pineapple_dishes := 5,
  mango_dishes_oliver_can_eat := 3
}

end NUMINAMATH_CALUDE_oliver_dish_count_l3487_348773


namespace NUMINAMATH_CALUDE_runner_stops_in_D_l3487_348739

/-- Represents the quarters of the circular track -/
inductive Quarter : Type
| A : Quarter
| B : Quarter
| C : Quarter
| D : Quarter

/-- The circular track -/
structure Track :=
  (circumference : ℝ)
  (start : Quarter)

/-- Determines the quarter where a runner stops after running a given distance -/
def stop_quarter (t : Track) (distance : ℝ) : Quarter :=
  sorry

/-- The main theorem -/
theorem runner_stops_in_D (t : Track) (distance : ℝ) :
  t.circumference = 40 ∧ t.start = Quarter.D ∧ distance = 1600 →
  stop_quarter t distance = Quarter.D :=
sorry

end NUMINAMATH_CALUDE_runner_stops_in_D_l3487_348739


namespace NUMINAMATH_CALUDE_simplify_nested_expression_l3487_348712

theorem simplify_nested_expression (x : ℝ) :
  2 * (1 - (2 * (1 - (1 + (2 - (3 * x)))))) = -10 + 12 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_expression_l3487_348712


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3487_348781

/-- An arithmetic sequence with given conditions -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  h1 : a 3 = 10
  h2 : a 12 = 31

/-- Properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (seq.a 1 = 16/3) ∧ 
  (∀ n : ℕ, seq.a (n + 1) - seq.a n = 7/3) ∧
  (∀ n : ℕ, seq.a n = 7/3 * n + 3) ∧
  (seq.a 18 = 45) ∧
  (∀ n : ℕ, seq.a n ≠ 85) := by
  sorry

#check arithmetic_sequence_properties

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3487_348781


namespace NUMINAMATH_CALUDE_seventh_triangular_is_28_l3487_348706

/-- Triangular number function -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The seventh triangular number is 28 -/
theorem seventh_triangular_is_28 : triangular 7 = 28 := by sorry

end NUMINAMATH_CALUDE_seventh_triangular_is_28_l3487_348706


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3487_348768

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 500)
  (h2 : profit_percentage = 25) :
  selling_price / (1 + profit_percentage / 100) = 400 := by
sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3487_348768


namespace NUMINAMATH_CALUDE_bisecting_line_tangent_lines_l3487_348786

-- Define the point P and circle C
def P : ℝ × ℝ := (-1, 4)
def C : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 - 3)^2 = 1}

-- Define the center of circle C
def center : ℝ × ℝ := (2, 3)

-- Define the lines
def line1 (x y : ℝ) : Prop := x + 3*y - 11 = 0
def line2 (x y : ℝ) : Prop := y - 4 = 0
def line3 (x y : ℝ) : Prop := 3*x + 4*y - 13 = 0

-- Theorem statements
theorem bisecting_line :
  line1 P.1 P.2 ∧ line1 center.1 center.2 := by sorry

theorem tangent_lines :
  (line2 P.1 P.2 ∧ (∃ (p : ℝ × ℝ), p ∈ C ∧ line2 p.1 p.2 ∧ (∀ (q : ℝ × ℝ), q ∈ C → line2 q.1 q.2 → q = p))) ∧
  (line3 P.1 P.2 ∧ (∃ (p : ℝ × ℝ), p ∈ C ∧ line3 p.1 p.2 ∧ (∀ (q : ℝ × ℝ), q ∈ C → line3 q.1 q.2 → q = p))) := by sorry

end NUMINAMATH_CALUDE_bisecting_line_tangent_lines_l3487_348786


namespace NUMINAMATH_CALUDE_trajectory_is_line_segment_l3487_348746

/-- The trajectory of a point M satisfying |MF₁| + |MF₂| = 8 is a line segment -/
theorem trajectory_is_line_segment (M : ℝ × ℝ) : 
  let F₁ : ℝ × ℝ := (-4, 0)
  let F₂ : ℝ × ℝ := (4, 0)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (dist M F₁ + dist M F₂ = 8) →
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (t * F₂.1 + (1 - t) * F₁.1, t * F₂.2 + (1 - t) * F₁.2) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_is_line_segment_l3487_348746


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3487_348789

/-- Function f(x) = ax² + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x + 1

/-- Condition: a ≥ 4 or a ≤ 0 -/
def condition (a : ℝ) : Prop := a ≥ 4 ∨ a ≤ 0

/-- f has zero points -/
def has_zero_points (a : ℝ) : Prop := ∃ x : ℝ, f a x = 0

theorem condition_necessary_not_sufficient :
  (∀ a : ℝ, has_zero_points a → condition a) ∧
  (∃ a : ℝ, condition a ∧ ¬has_zero_points a) :=
sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3487_348789


namespace NUMINAMATH_CALUDE_sum_of_digits_main_expression_l3487_348798

/-- Represents a string of digits --/
structure DigitString :=
  (length : ℕ)
  (digit : ℕ)
  (digit_valid : digit < 10)

/-- Calculates the product of two DigitStrings --/
def multiply_digit_strings (a b : DigitString) : ℕ := sorry

/-- Calculates the sum of digits in a natural number --/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Represents the expression (80 eights × 80 fives + 80 ones) --/
def main_expression : ℕ :=
  let eights : DigitString := ⟨80, 8, by norm_num⟩
  let fives : DigitString := ⟨80, 5, by norm_num⟩
  let ones : DigitString := ⟨80, 1, by norm_num⟩
  multiply_digit_strings eights fives + ones.length * ones.digit

/-- The main theorem to be proved --/
theorem sum_of_digits_main_expression :
  sum_of_digits main_expression = 400 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_main_expression_l3487_348798


namespace NUMINAMATH_CALUDE_transformed_area_l3487_348740

/-- Given a region T in the plane with area 9, prove that when transformed
    by the matrix [[3, 0], [8, 3]], the resulting region T' has an area of 81. -/
theorem transformed_area (T : Set (Fin 2 → ℝ)) (harea : MeasureTheory.volume T = 9) :
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![3, 0; 8, 3]
  let T' := (fun p ↦ M.mulVec p) '' T
  MeasureTheory.volume T' = 81 := by
sorry

end NUMINAMATH_CALUDE_transformed_area_l3487_348740


namespace NUMINAMATH_CALUDE_automobile_dealer_revenue_l3487_348784

/-- Represents the revenue calculation for an automobile dealer's sale --/
theorem automobile_dealer_revenue :
  ∀ (num_suvs : ℕ),
    num_suvs + (num_suvs + 50) + (2 * num_suvs) = 150 →
    20000 * (num_suvs + 50) + 30000 * (2 * num_suvs) + 40000 * num_suvs = 4000000 :=
by
  sorry

end NUMINAMATH_CALUDE_automobile_dealer_revenue_l3487_348784


namespace NUMINAMATH_CALUDE_double_perimeter_polygon_exists_l3487_348787

/-- A grid point in 2D space --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A line segment on the grid --/
inductive GridSegment
  | Horizontal : GridPoint → GridPoint → GridSegment
  | Vertical : GridPoint → GridPoint → GridSegment
  | Diagonal1x1 : GridPoint → GridPoint → GridSegment
  | Diagonal1x2 : GridPoint → GridPoint → GridSegment

/-- A polygon on the grid --/
structure GridPolygon where
  vertices : List GridPoint
  edges : List GridSegment

/-- A triangle on the grid --/
structure GridTriangle where
  vertices : Fin 3 → GridPoint
  edges : Fin 3 → GridSegment

/-- Calculate the perimeter of a grid polygon --/
def perimeterOfPolygon (p : GridPolygon) : ℕ :=
  sorry

/-- Calculate the perimeter of a grid triangle --/
def perimeterOfTriangle (t : GridTriangle) : ℕ :=
  sorry

/-- Check if a polygon has double the perimeter of a triangle --/
def hasDoublePerimeter (p : GridPolygon) (t : GridTriangle) : Prop :=
  perimeterOfPolygon p = 2 * perimeterOfTriangle t

/-- Main theorem: Given a triangle on a grid, there exists a polygon with double its perimeter --/
theorem double_perimeter_polygon_exists (t : GridTriangle) : 
  ∃ (p : GridPolygon), hasDoublePerimeter p t :=
sorry

end NUMINAMATH_CALUDE_double_perimeter_polygon_exists_l3487_348787


namespace NUMINAMATH_CALUDE_range_of_a_l3487_348799

/-- The inequality (a-3)x^2 + 2(a-3)x - 4 < 0 has a solution set of all real numbers for x -/
def has_all_real_solutions (a : ℝ) : Prop :=
  ∀ x : ℝ, (a - 3) * x^2 + 2 * (a - 3) * x - 4 < 0

/-- The main theorem stating the range of a -/
theorem range_of_a (a : ℝ) : 
  has_all_real_solutions a ↔ -1 < a ∧ a < 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3487_348799


namespace NUMINAMATH_CALUDE_distinct_roots_isosceles_triangle_k_values_l3487_348745

/-- The quadratic equation x^2 - (2k+1)x + k^2 + k = 0 has two distinct real roots for all k -/
theorem distinct_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - (2*k+1)*x₁ + k^2 + k = 0 ∧ x₂^2 - (2*k+1)*x₂ + k^2 + k = 0 :=
sorry

/-- When two roots of x^2 - (2k+1)x + k^2 + k = 0 form two sides of an isosceles triangle 
    with the third side of length 5, k = 4 or k = 5 -/
theorem isosceles_triangle_k_values :
  ∃ x₁ x₂ : ℝ, 
    x₁^2 - (2*4+1)*x₁ + 4^2 + 4 = 0 ∧
    x₂^2 - (2*4+1)*x₂ + 4^2 + 4 = 0 ∧
    ((x₁ = 5 ∧ x₂ = x₁) ∨ (x₂ = 5 ∧ x₁ = x₂))
  ∧
  ∃ y₁ y₂ : ℝ,
    y₁^2 - (2*5+1)*y₁ + 5^2 + 5 = 0 ∧
    y₂^2 - (2*5+1)*y₂ + 5^2 + 5 = 0 ∧
    ((y₁ = 5 ∧ y₂ = y₁) ∨ (y₂ = 5 ∧ y₁ = y₂))
  ∧
  ∀ k : ℝ, k ≠ 4 → k ≠ 5 →
    ¬∃ z₁ z₂ : ℝ,
      z₁^2 - (2*k+1)*z₁ + k^2 + k = 0 ∧
      z₂^2 - (2*k+1)*z₂ + k^2 + k = 0 ∧
      ((z₁ = 5 ∧ z₂ = z₁) ∨ (z₂ = 5 ∧ z₁ = z₂)) :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_isosceles_triangle_k_values_l3487_348745


namespace NUMINAMATH_CALUDE_range_of_a_l3487_348741

theorem range_of_a (p q : ℝ → Prop) (a : ℝ) :
  (∀ x, p x ↔ (3*x - 1)/(x - 2) ≤ 1) →
  (∀ x, q x ↔ x^2 - (2*a + 1)*x + a*(a + 1) < 0) →
  (∀ x, ¬(q x) → ¬(p x)) →
  (∃ x, ¬(q x) ∧ p x) →
  -1/2 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3487_348741


namespace NUMINAMATH_CALUDE_nested_a_value_l3487_348737

-- Define the function a
def a (k : ℕ) : ℕ := (k + 1)^2

-- State the theorem
theorem nested_a_value :
  let k : ℕ := 1
  a (a (a (a k))) = 458329 := by
  sorry

end NUMINAMATH_CALUDE_nested_a_value_l3487_348737


namespace NUMINAMATH_CALUDE_sum_congruence_l3487_348722

def large_sum : ℕ := 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999

theorem sum_congruence : large_sum % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_l3487_348722


namespace NUMINAMATH_CALUDE_inequality_proof_l3487_348752

theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a < b) :
  1 / (a * b^2) > 1 / (a^2 * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3487_348752


namespace NUMINAMATH_CALUDE_largest_divided_by_smallest_l3487_348715

theorem largest_divided_by_smallest : 
  let numbers : List ℝ := [10, 11, 12, 13]
  (List.maximum numbers).get! / (List.minimum numbers).get! = 1.3 := by
sorry

end NUMINAMATH_CALUDE_largest_divided_by_smallest_l3487_348715


namespace NUMINAMATH_CALUDE_max_x_plus_y_on_circle_l3487_348776

theorem max_x_plus_y_on_circle :
  let S := {(x, y) : ℝ × ℝ | x^2 + y^2 - 3*y - 1 = 0}
  ∃ (max : ℝ), ∀ (p : ℝ × ℝ), p ∈ S → p.1 + p.2 ≤ max ∧
  ∃ (q : ℝ × ℝ), q ∈ S ∧ q.1 + q.2 = max ∧
  max = (Real.sqrt 26 + 3) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_x_plus_y_on_circle_l3487_348776


namespace NUMINAMATH_CALUDE_marbles_given_l3487_348772

theorem marbles_given (initial_marbles : ℕ) (remaining_marbles : ℕ) : 
  initial_marbles = 87 → remaining_marbles = 79 → initial_marbles - remaining_marbles = 8 := by
sorry

end NUMINAMATH_CALUDE_marbles_given_l3487_348772


namespace NUMINAMATH_CALUDE_work_completion_time_l3487_348718

theorem work_completion_time (a b : ℝ) (h1 : b = 20) 
  (h2 : 4 * (1/a + 1/b) = 0.4666666666666667) : a = 15 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3487_348718


namespace NUMINAMATH_CALUDE_remainder_divisibility_l3487_348701

theorem remainder_divisibility (x y : ℤ) (h : 9 ∣ (x + 2*y)) :
  ∃ k : ℤ, 2*(5*x - 8*y - 4) = 9*k + (-8) ∨ 2*(5*x - 8*y - 4) = 9*k + 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l3487_348701


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l3487_348738

/-- Prove that the initial volume of a milk-water mixture is 155 liters -/
theorem initial_mixture_volume (milk : ℝ) (water : ℝ) : 
  milk / water = 3 / 2 →  -- Initial ratio of milk to water
  milk / (water + 62) = 3 / 4 →  -- New ratio after adding 62 liters of water
  milk + water = 155 :=  -- Initial volume of the mixture
by sorry

end NUMINAMATH_CALUDE_initial_mixture_volume_l3487_348738


namespace NUMINAMATH_CALUDE_sandy_marks_per_correct_sum_l3487_348731

theorem sandy_marks_per_correct_sum :
  ∀ (marks_per_correct : ℕ) (marks_per_incorrect : ℕ) (total_attempts : ℕ) (total_marks : ℕ) (correct_attempts : ℕ),
    marks_per_incorrect = 2 →
    total_attempts = 30 →
    total_marks = 50 →
    correct_attempts = 22 →
    marks_per_correct * correct_attempts - marks_per_incorrect * (total_attempts - correct_attempts) = total_marks →
    marks_per_correct = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandy_marks_per_correct_sum_l3487_348731


namespace NUMINAMATH_CALUDE_cyclic_inequality_l3487_348766

theorem cyclic_inequality (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (h : x * y + y * z + z * x = x + y + z) : 
  1 / (x^2 + y + 1) + 1 / (y^2 + z + 1) + 1 / (z^2 + x + 1) ≤ 1 ∧ 
  (1 / (x^2 + y + 1) + 1 / (y^2 + z + 1) + 1 / (z^2 + x + 1) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l3487_348766


namespace NUMINAMATH_CALUDE_traffic_light_change_probability_l3487_348793

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle duration -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the duration of color changes -/
def changeDuration (cycle : TrafficLightCycle) : ℕ :=
  3 * 5 -- 5 seconds at the end of each color

/-- Theorem: Probability of observing a color change -/
theorem traffic_light_change_probability (cycle : TrafficLightCycle)
    (h1 : cycle.green = 45)
    (h2 : cycle.yellow = 5)
    (h3 : cycle.red = 50)
    (h4 : cycleDuration cycle = 100) :
    (changeDuration cycle : ℚ) / (cycleDuration cycle : ℚ) = 3 / 20 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_change_probability_l3487_348793


namespace NUMINAMATH_CALUDE_f_inequality_solution_f_minimum_value_condition_l3487_348723

def f (a : ℝ) (x : ℝ) : ℝ := |3*x - 1| + a*x + 3

theorem f_inequality_solution (a : ℝ) (h : a = 1) :
  {x : ℝ | f a x ≤ 4} = {x : ℝ | 0 ≤ x ∧ x ≤ 1/2} :=
sorry

theorem f_minimum_value_condition (a : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), f a x ≤ f a y) ↔ -3 ≤ a ∧ a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_f_inequality_solution_f_minimum_value_condition_l3487_348723


namespace NUMINAMATH_CALUDE_positive_A_value_l3487_348710

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h : hash A 6 = 200) : A = 2 * Real.sqrt 41 :=
sorry

end NUMINAMATH_CALUDE_positive_A_value_l3487_348710


namespace NUMINAMATH_CALUDE_remainder_theorem_l3487_348730

theorem remainder_theorem (x y u v : ℕ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x = u * y + v) (h4 : v < y) : 
  (x + (2 * u + 1) * y) % y = v := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3487_348730


namespace NUMINAMATH_CALUDE_part_one_part_two_l3487_348716

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ
  -- Add necessary conditions
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  pos_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π
  -- Add the given condition
  given_condition : 2 * Real.cos C + 2 * Real.cos A = 5 * b / 2

theorem part_one (t : Triangle) : 2 * (t.a + t.c) = 3 * t.b := by sorry

theorem part_two (t : Triangle) (h1 : Real.cos t.B = 1/4) (h2 : t.S = Real.sqrt 15) : t.b = 4 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3487_348716


namespace NUMINAMATH_CALUDE_ping_pong_rackets_sold_l3487_348785

/-- The number of pairs of ping pong rackets sold -/
def num_pairs : ℕ := 55

/-- The total amount made from selling rackets in dollars -/
def total_amount : ℚ := 539

/-- The average price of a pair of rackets in dollars -/
def avg_price : ℚ := 9.8

/-- Theorem: The number of pairs of ping pong rackets sold is 55 -/
theorem ping_pong_rackets_sold :
  (total_amount / avg_price : ℚ) = num_pairs := by sorry

end NUMINAMATH_CALUDE_ping_pong_rackets_sold_l3487_348785


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3487_348743

/-- The eccentricity of a hyperbola with equation x²/4 - y²/3 = 1 is √7/2 -/
theorem hyperbola_eccentricity : 
  let a : ℝ := 2
  let b : ℝ := Real.sqrt 3
  let c : ℝ := Real.sqrt (a^2 + b^2)
  let e : ℝ := c / a
  e = Real.sqrt 7 / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3487_348743


namespace NUMINAMATH_CALUDE_clementine_baked_72_cookies_l3487_348796

/-- The number of cookies Clementine baked -/
def clementine_cookies : ℕ := 72

/-- The number of cookies Jake baked -/
def jake_cookies : ℕ := 2 * clementine_cookies

/-- The number of cookies Tory baked -/
def tory_cookies : ℕ := (clementine_cookies + jake_cookies) / 2

/-- The price of each cookie in dollars -/
def cookie_price : ℕ := 2

/-- The total amount of money made from selling cookies in dollars -/
def total_money : ℕ := 648

theorem clementine_baked_72_cookies :
  clementine_cookies = 72 ∧
  jake_cookies = 2 * clementine_cookies ∧
  tory_cookies = (clementine_cookies + jake_cookies) / 2 ∧
  cookie_price = 2 ∧
  total_money = 648 ∧
  total_money = cookie_price * (clementine_cookies + jake_cookies + tory_cookies) :=
by sorry

end NUMINAMATH_CALUDE_clementine_baked_72_cookies_l3487_348796


namespace NUMINAMATH_CALUDE_persimmons_count_l3487_348797

theorem persimmons_count (tangerines : ℕ) (total : ℕ) (h1 : tangerines = 19) (h2 : total = 37) :
  total - tangerines = 18 := by
  sorry

end NUMINAMATH_CALUDE_persimmons_count_l3487_348797


namespace NUMINAMATH_CALUDE_constant_term_product_l3487_348717

-- Define polynomials p, q, and r
variable (p q r : ℝ[X])

-- Define the relationship between r, p, and q
variable (h_prod : r = p * q)

-- Define the constant terms of p and r
variable (h_p_const : p.coeff 0 = 5)
variable (h_r_const : r.coeff 0 = -10)

-- Theorem statement
theorem constant_term_product :
  q.eval 0 = -2 := by sorry

end NUMINAMATH_CALUDE_constant_term_product_l3487_348717


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3487_348714

theorem algebraic_expression_value : 
  let x : ℝ := -1
  3 * x^2 + 2 * x - 1 = 0 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3487_348714


namespace NUMINAMATH_CALUDE_range_of_a_l3487_348709

-- Define set A
def A : Set ℝ := {x | 1 < |x - 2| ∧ |x - 2| < 2}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x^2 - (a + 1) * x + a < 0}

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, x ∈ A ∩ B a) ↔ a ∈ Set.Iio 1 ∪ Set.Ioi 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3487_348709


namespace NUMINAMATH_CALUDE_arithmetic_sum_l3487_348794

theorem arithmetic_sum : 5 * 12 + 7 * 9 + 8 * 4 + 6 * 7 + 2 * 13 = 223 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_l3487_348794


namespace NUMINAMATH_CALUDE_range_of_a_l3487_348742

def M : Set ℝ := {x | |x| < 1}
def N (a : ℝ) : Set ℝ := {a}

theorem range_of_a (a : ℝ) (h : M ∪ N a = M) : a ∈ Set.Ioo (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3487_348742


namespace NUMINAMATH_CALUDE_birds_on_fence_l3487_348753

theorem birds_on_fence : 
  let initial_sparrows : ℕ := 4
  let initial_storks : ℕ := 46
  let pigeons_joined : ℕ := 6
  let sparrows_left : ℕ := 3
  let storks_left : ℕ := 5
  let swans_came : ℕ := 8
  let ducks_came : ℕ := 2
  
  let total_birds : ℕ := 
    (initial_sparrows + initial_storks + pigeons_joined - sparrows_left - storks_left + swans_came + ducks_came)
  
  total_birds = 58 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l3487_348753


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l3487_348767

/-- The sum of the first n terms of a sequence -/
def S (n : ℕ) (c : ℝ) : ℝ := 3^n - c

/-- The nth term of the sequence -/
def a (n : ℕ) (c : ℝ) : ℝ := S (n + 1) c - S n c

/-- A sequence is geometric if the ratio between consecutive terms is constant -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_condition (c : ℝ) :
  (c = 1 ↔ IsGeometric (a · c)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l3487_348767


namespace NUMINAMATH_CALUDE_least_odd_prime_factor_of_2023_pow_6_plus_1_l3487_348703

theorem least_odd_prime_factor_of_2023_pow_6_plus_1 :
  (Nat.minFac (2023^6 + 1)) = 37 := by
  sorry

end NUMINAMATH_CALUDE_least_odd_prime_factor_of_2023_pow_6_plus_1_l3487_348703


namespace NUMINAMATH_CALUDE_f_composition_result_l3487_348705

noncomputable def f (z : ℂ) : ℂ :=
  if z.im = 0 then -z^3 else z^3

theorem f_composition_result :
  f (f (f (f (-1 + I)))) = -1.79841759e14 - 2.75930025e10 * I :=
by sorry

end NUMINAMATH_CALUDE_f_composition_result_l3487_348705


namespace NUMINAMATH_CALUDE_ice_cream_cost_is_7_l3487_348724

/-- The cost of one portion of ice cream in kopecks -/
def ice_cream_cost : ℕ := 7

/-- Fedya's money in kopecks -/
def fedya_money : ℕ := ice_cream_cost - 7

/-- Masha's money in kopecks -/
def masha_money : ℕ := ice_cream_cost - 1

theorem ice_cream_cost_is_7 :
  (fedya_money + masha_money < ice_cream_cost) ∧
  (fedya_money = ice_cream_cost - 7) ∧
  (masha_money = ice_cream_cost - 1) →
  ice_cream_cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_cost_is_7_l3487_348724


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_roots_l3487_348765

theorem isosceles_right_triangle_roots (p q : ℂ) (z₁ z₂ : ℂ) : 
  z₁^2 + 2*p*z₁ + q = 0 →
  z₂^2 + 2*p*z₂ + q = 0 →
  z₂ = Complex.I * z₁ →
  p^2 / q = 2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_roots_l3487_348765


namespace NUMINAMATH_CALUDE_distance_to_center_l3487_348779

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 8*x - 4*y + 16

/-- The center of the circle -/
def circle_center : ℝ × ℝ := sorry

/-- The given point -/
def given_point : ℝ × ℝ := (3, -1)

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem distance_to_center : distance circle_center given_point = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_distance_to_center_l3487_348779


namespace NUMINAMATH_CALUDE_at_least_two_in_same_group_l3487_348727

theorem at_least_two_in_same_group 
  (n : ℕ) 
  (h_n : n = 28) 
  (partition1 partition2 partition3 : Fin n → Fin 3) 
  (h_diff1 : partition1 ≠ partition2) 
  (h_diff2 : partition2 ≠ partition3) 
  (h_diff3 : partition1 ≠ partition3) :
  ∃ i j : Fin n, i ≠ j ∧ 
    partition1 i = partition1 j ∧ 
    partition2 i = partition2 j ∧ 
    partition3 i = partition3 j :=
sorry

end NUMINAMATH_CALUDE_at_least_two_in_same_group_l3487_348727


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3487_348774

open Set

def U : Set ℝ := univ
def A : Set ℝ := {1, 2, 3, 4}
def B : Set ℝ := {x | x ≤ 2}

theorem intersection_complement_equality :
  A ∩ (U \ B) = {3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3487_348774


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3487_348792

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) :
  a 7 - (1/2) * a 8 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3487_348792


namespace NUMINAMATH_CALUDE_defective_products_m1_l3487_348721

theorem defective_products_m1 (m1_production m2_production m3_production : ℝ)
  (m2_defective_rate m3_defective_rate total_defective_rate : ℝ) :
  m1_production = 0.4 →
  m2_production = 0.3 →
  m3_production = 0.3 →
  m2_defective_rate = 0.01 →
  m3_defective_rate = 0.07 →
  total_defective_rate = 0.036 →
  ∃ (m1_defective_rate : ℝ),
    m1_defective_rate * m1_production +
    m2_defective_rate * m2_production +
    m3_defective_rate * m3_production = total_defective_rate ∧
    m1_defective_rate = 0.03 := by
  sorry

end NUMINAMATH_CALUDE_defective_products_m1_l3487_348721


namespace NUMINAMATH_CALUDE_train_distance_problem_l3487_348777

theorem train_distance_problem :
  let fast_train_time : ℝ := 5
  let slow_train_time : ℝ := fast_train_time * (1 + 1/5)
  let stop_time : ℝ := 2
  let additional_distance : ℝ := 40
  let distance : ℝ := 150
  let fast_train_speed : ℝ := distance / fast_train_time
  let slow_train_speed : ℝ := distance / slow_train_time
  let fast_train_distance : ℝ := fast_train_speed * stop_time
  let slow_train_distance : ℝ := slow_train_speed * stop_time
  let remaining_distance : ℝ := distance - (fast_train_distance + slow_train_distance)
  remaining_distance = additional_distance :=
by
  sorry

#check train_distance_problem

end NUMINAMATH_CALUDE_train_distance_problem_l3487_348777


namespace NUMINAMATH_CALUDE_triangle_area_in_circle_l3487_348728

/-- The area of a triangle inscribed in a circle with given radius and side ratio --/
theorem triangle_area_in_circle (r : ℝ) (a b c : ℝ) (h_radius : r = 2 * Real.sqrt 3) 
  (h_ratio : ∃ (k : ℝ), a = 3 * k ∧ b = 5 * k ∧ c = 7 * k) :
  ∃ (area : ℝ), area = (135 * Real.sqrt 3) / 49 ∧ 
  area = (1 / 2) * a * b * Real.sin (2 * π / 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_in_circle_l3487_348728


namespace NUMINAMATH_CALUDE_system_solution_exists_l3487_348790

theorem system_solution_exists (b : ℝ) : 
  (∃ (a x y : ℝ), 
    x = 7 / b - abs (y + b) ∧ 
    x^2 + y^2 + 96 = -a * (2 * y + a) - 20 * x) ↔ 
  (b ≤ -7/12 ∨ b > 0) :=
sorry

end NUMINAMATH_CALUDE_system_solution_exists_l3487_348790


namespace NUMINAMATH_CALUDE_mad_hatter_waiting_time_l3487_348749

/-- The rate at which the Mad Hatter's clock runs compared to normal time -/
def mad_hatter_rate : ℚ := 5/4

/-- The rate at which the March Hare's clock runs compared to normal time -/
def march_hare_rate : ℚ := 5/6

/-- The agreed meeting time in hours after noon -/
def meeting_time : ℚ := 5

theorem mad_hatter_waiting_time :
  let mad_hatter_arrival := meeting_time / mad_hatter_rate
  let march_hare_arrival := meeting_time / march_hare_rate
  march_hare_arrival - mad_hatter_arrival = 2 := by sorry

end NUMINAMATH_CALUDE_mad_hatter_waiting_time_l3487_348749


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3487_348736

/-- Given an equilateral triangle with perimeter 45 and an isosceles triangle sharing one side
    with the equilateral triangle and having a base of length 10, the perimeter of the isosceles
    triangle is 40. -/
theorem isosceles_triangle_perimeter
  (equilateral_perimeter : ℝ)
  (isosceles_base : ℝ)
  (h_equilateral_perimeter : equilateral_perimeter = 45)
  (h_isosceles_base : isosceles_base = 10)
  (h_shared_side : ∃ (side : ℝ), side = equilateral_perimeter / 3 ∧
                   ∃ (leg : ℝ), leg = side) :
  ∃ (isosceles_perimeter : ℝ), isosceles_perimeter = 40 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3487_348736


namespace NUMINAMATH_CALUDE_difference_of_squares_l3487_348734

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3487_348734
