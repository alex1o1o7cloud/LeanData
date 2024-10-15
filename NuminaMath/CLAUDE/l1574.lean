import Mathlib

namespace NUMINAMATH_CALUDE_sine_transformation_l1574_157408

theorem sine_transformation (x : ℝ) : 
  Real.sin (2 * (x + π/4) + π/6) = Real.sin (2*x + 2*π/3) := by
  sorry

end NUMINAMATH_CALUDE_sine_transformation_l1574_157408


namespace NUMINAMATH_CALUDE_no_valid_sum_of_consecutive_integers_l1574_157479

def sum_of_consecutive_integers (k : ℕ) : ℕ := 150 * k + 11175

def given_integers : List ℕ := [1625999850, 2344293800, 3578726150, 4691196050, 5815552000]

theorem no_valid_sum_of_consecutive_integers : 
  ∀ n ∈ given_integers, ¬ ∃ k : ℕ, sum_of_consecutive_integers k = n :=
by sorry

end NUMINAMATH_CALUDE_no_valid_sum_of_consecutive_integers_l1574_157479


namespace NUMINAMATH_CALUDE_area_of_overlapping_rotated_squares_exists_l1574_157486

/-- Represents a square in 2D space -/
structure Square where
  sideLength : ℝ
  rotation : ℝ -- in radians

/-- Calculates the area of a polygon formed by overlapping squares -/
noncomputable def areaOfOverlappingSquares (squares : List Square) : ℝ :=
  sorry

theorem area_of_overlapping_rotated_squares_exists : 
  ∃ (A : ℝ), 
    let squares := [
      { sideLength := 4, rotation := 0 },
      { sideLength := 5, rotation := π/4 },
      { sideLength := 6, rotation := -π/6 }
    ]
    A = areaOfOverlappingSquares squares ∧ A > 0 := by
  sorry

end NUMINAMATH_CALUDE_area_of_overlapping_rotated_squares_exists_l1574_157486


namespace NUMINAMATH_CALUDE_gcd_of_2_powers_l1574_157459

theorem gcd_of_2_powers : Nat.gcd (2^2021 - 1) (2^2000 - 1) = 2^21 - 1 := by sorry

end NUMINAMATH_CALUDE_gcd_of_2_powers_l1574_157459


namespace NUMINAMATH_CALUDE_factorial_last_nonzero_digit_not_periodic_l1574_157472

/-- The last nonzero digit of n! -/
def lastNonzeroDigit (n : ℕ) : ℕ :=
  sorry

/-- The sequence of last nonzero digits of factorials is not eventually periodic -/
theorem factorial_last_nonzero_digit_not_periodic :
  ¬ ∃ (p d : ℕ), p > 0 ∧ d > 0 ∧ 
  ∀ n ≥ d, lastNonzeroDigit n = lastNonzeroDigit (n + p) :=
sorry

end NUMINAMATH_CALUDE_factorial_last_nonzero_digit_not_periodic_l1574_157472


namespace NUMINAMATH_CALUDE_second_month_sale_l1574_157454

def average_sale : ℕ := 6500
def num_months : ℕ := 6
def first_month_sale : ℕ := 6535
def third_month_sale : ℕ := 6855
def fourth_month_sale : ℕ := 7230
def fifth_month_sale : ℕ := 6562
def sixth_month_sale : ℕ := 4891

theorem second_month_sale :
  ∃ (second_month_sale : ℕ),
    second_month_sale = average_sale * num_months - 
      (first_month_sale + third_month_sale + fourth_month_sale + 
       fifth_month_sale + sixth_month_sale) ∧
    second_month_sale = 6927 := by
  sorry

end NUMINAMATH_CALUDE_second_month_sale_l1574_157454


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1574_157436

-- Problem 1
theorem problem_1 : (-8) + 10 - 2 + (-1) = -1 := by sorry

-- Problem 2
theorem problem_2 : 12 - 7 * (-4) + 8 / (-2) = 36 := by sorry

-- Problem 3
theorem problem_3 : (1/2 + 1/3 - 1/6) / (-1/18) = -12 := by sorry

-- Problem 4
theorem problem_4 : -1^4 - (1 + 0.5) * (1/3) / (-4)^2 = -33/32 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1574_157436


namespace NUMINAMATH_CALUDE_translated_line_through_origin_l1574_157457

/-- A line in the Cartesian plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line vertically by a given amount -/
def translate_line (l : Line) (dy : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - dy }

/-- Check if a line passes through a point -/
def passes_through (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem translated_line_through_origin (b : ℝ) :
  let original_line : Line := { slope := 2, intercept := b }
  let translated_line := translate_line original_line 2
  passes_through translated_line 0 0 → b = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_translated_line_through_origin_l1574_157457


namespace NUMINAMATH_CALUDE_tea_consumption_l1574_157462

/-- Represents the relationship between hours spent reading and liters of tea consumed -/
structure ReadingTeaData where
  hours : ℝ
  liters : ℝ

/-- The constant of proportionality for the inverse relationship -/
def proportionality_constant (data : ReadingTeaData) : ℝ :=
  data.hours * data.liters

theorem tea_consumption (wednesday thursday friday : ReadingTeaData)
  (h_wednesday : wednesday.hours = 8 ∧ wednesday.liters = 3)
  (h_thursday : thursday.hours = 5)
  (h_friday : friday.hours = 10)
  (h_inverse_prop : proportionality_constant wednesday = proportionality_constant thursday
                  ∧ proportionality_constant wednesday = proportionality_constant friday) :
  thursday.liters = 4.8 ∧ friday.liters = 2.4 := by
  sorry

#check tea_consumption

end NUMINAMATH_CALUDE_tea_consumption_l1574_157462


namespace NUMINAMATH_CALUDE_dylans_mother_hotdogs_l1574_157426

theorem dylans_mother_hotdogs (helens_hotdogs : ℕ) (total_hotdogs : ℕ) 
  (h1 : helens_hotdogs = 101)
  (h2 : total_hotdogs = 480) :
  total_hotdogs - helens_hotdogs = 379 := by
  sorry

end NUMINAMATH_CALUDE_dylans_mother_hotdogs_l1574_157426


namespace NUMINAMATH_CALUDE_max_hands_for_54_coincidences_l1574_157493

/-- Represents a clock with minute hands moving in opposite directions -/
structure Clock :=
  (hands_clockwise : ℕ)
  (hands_counterclockwise : ℕ)

/-- The number of coincidences between pairs of hands in one hour -/
def coincidences (c : Clock) : ℕ :=
  2 * c.hands_clockwise * c.hands_counterclockwise

/-- The total number of hands on the clock -/
def total_hands (c : Clock) : ℕ :=
  c.hands_clockwise + c.hands_counterclockwise

/-- Theorem stating that if there are 54 coincidences in an hour,
    the maximum number of hands is 28 -/
theorem max_hands_for_54_coincidences :
  ∀ c : Clock, coincidences c = 54 → total_hands c ≤ 28 :=
by sorry

end NUMINAMATH_CALUDE_max_hands_for_54_coincidences_l1574_157493


namespace NUMINAMATH_CALUDE_correct_commission_calculation_l1574_157406

/-- Calculates the total commission for a salesperson selling appliances -/
def calculate_commission (num_appliances : ℕ) (total_selling_price : ℚ) : ℚ :=
  let fixed_commission := 50 * num_appliances
  let percentage_commission := 0.1 * total_selling_price
  fixed_commission + percentage_commission

/-- Theorem stating the correct commission calculation for the given scenario -/
theorem correct_commission_calculation :
  calculate_commission 6 3620 = 662 := by
  sorry

end NUMINAMATH_CALUDE_correct_commission_calculation_l1574_157406


namespace NUMINAMATH_CALUDE_unique_base_solution_l1574_157444

/-- Converts a number from base b to decimal --/
def to_decimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Converts a number from decimal to base b --/
def from_decimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Checks if a number is valid in base b --/
def is_valid_in_base (n : ℕ) (b : ℕ) : Prop := sorry

theorem unique_base_solution :
  ∃! b : ℕ, b > 6 ∧ 
    is_valid_in_base 243 b ∧
    is_valid_in_base 156 b ∧
    is_valid_in_base 411 b ∧
    to_decimal 243 b + to_decimal 156 b = to_decimal 411 b ∧
    b = 10 := by sorry

end NUMINAMATH_CALUDE_unique_base_solution_l1574_157444


namespace NUMINAMATH_CALUDE_gambler_win_rate_is_40_percent_l1574_157400

/-- Represents the gambler's statistics -/
structure GamblerStats where
  games_played : ℕ
  future_games : ℕ
  future_win_rate : ℚ
  target_win_rate : ℚ

/-- Calculates the current win rate of the gambler -/
def current_win_rate (stats : GamblerStats) : ℚ :=
  let total_games := stats.games_played + stats.future_games
  let future_wins := stats.future_win_rate * stats.future_games
  let total_wins := stats.target_win_rate * total_games
  (total_wins - future_wins) / stats.games_played

/-- Theorem stating the gambler's current win rate is 40% under given conditions -/
theorem gambler_win_rate_is_40_percent (stats : GamblerStats) 
  (h1 : stats.games_played = 40)
  (h2 : stats.future_games = 80)
  (h3 : stats.future_win_rate = 7/10)
  (h4 : stats.target_win_rate = 6/10) :
  current_win_rate stats = 4/10 := by
  sorry

#eval current_win_rate { games_played := 40, future_games := 80, future_win_rate := 7/10, target_win_rate := 6/10 }

end NUMINAMATH_CALUDE_gambler_win_rate_is_40_percent_l1574_157400


namespace NUMINAMATH_CALUDE_sandy_marks_lost_l1574_157413

theorem sandy_marks_lost (marks_per_correct : ℕ) (total_attempts : ℕ) (total_marks : ℕ) (correct_sums : ℕ) :
  marks_per_correct = 3 →
  total_attempts = 30 →
  total_marks = 45 →
  correct_sums = 21 →
  ∃ (marks_lost_per_incorrect : ℕ), 
    marks_lost_per_incorrect = 2 ∧
    total_marks = correct_sums * marks_per_correct - (total_attempts - correct_sums) * marks_lost_per_incorrect :=
by sorry

end NUMINAMATH_CALUDE_sandy_marks_lost_l1574_157413


namespace NUMINAMATH_CALUDE_triangle_side_length_l1574_157432

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  b = 7 →
  c = 3 →
  Real.cos (B - C) = 17/18 →
  a = 40/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1574_157432


namespace NUMINAMATH_CALUDE_abs_inequality_and_fraction_inequality_l1574_157407

theorem abs_inequality_and_fraction_inequality :
  (∀ x : ℝ, |x + 3| - |x - 2| ≥ 3 ↔ x ≥ 1) ∧
  (∀ a b : ℝ, a > b ∧ b > 0 → (a^2 - b^2) / (a^2 + b^2) > (a - b) / (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_and_fraction_inequality_l1574_157407


namespace NUMINAMATH_CALUDE_floor_equation_solution_l1574_157453

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊2 * x⌋ - (1 / 2)⌋ = ⌊x + 3⌋ ↔ x ∈ Set.Icc (3.5 : ℝ) (4.5 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l1574_157453


namespace NUMINAMATH_CALUDE_max_quarters_kevin_l1574_157442

/-- Represents the value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- Represents the value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- Represents the total amount of money Kevin has in dollars -/
def total_money : ℚ := 4.85

/-- 
Given that Kevin has $4.85 in U.S. coins and twice as many nickels as quarters,
prove that the maximum number of quarters he could have is 13.
-/
theorem max_quarters_kevin : 
  ∃ (q : ℕ), 
    q ≤ 13 ∧ 
    q * quarter_value + 2 * q * nickel_value ≤ total_money ∧
    ∀ (n : ℕ), n * quarter_value + 2 * n * nickel_value ≤ total_money → n ≤ q :=
sorry

end NUMINAMATH_CALUDE_max_quarters_kevin_l1574_157442


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l1574_157445

/-- A polynomial is a perfect square if it can be expressed as (ax + b)^2 for some real numbers a and b -/
def is_perfect_square (p : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, p x = (a * x + b)^2

/-- The given polynomial -/
def polynomial (m : ℝ) (x : ℝ) : ℝ := m - 10*x + x^2

theorem perfect_square_polynomial (m : ℝ) :
  is_perfect_square (polynomial m) → m = 25 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l1574_157445


namespace NUMINAMATH_CALUDE_f_min_at_4_l1574_157443

/-- The quadratic function f(x) = x^2 - 8x + 15 -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 15

/-- Theorem: The function f(x) = x^2 - 8x + 15 has a minimum value when x = 4 -/
theorem f_min_at_4 : ∀ y : ℝ, f 4 ≤ f y := by
  sorry

end NUMINAMATH_CALUDE_f_min_at_4_l1574_157443


namespace NUMINAMATH_CALUDE_interest_rate_proof_l1574_157410

/-- Proves that for given conditions, the annual interest rate is 10% -/
theorem interest_rate_proof (principal : ℝ) (time : ℝ) (diff : ℝ) : 
  principal = 1700 → 
  time = 1 → 
  diff = 4.25 → 
  ∃ (rate : ℝ), 
    rate = 10 ∧ 
    principal * ((1 + rate / 200)^2 - 1) - principal * rate * time / 100 = diff :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l1574_157410


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1574_157480

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (∀ a b, a > b → a > b - 1) ∧ 
  (∃ a b, a > b - 1 ∧ ¬(a > b)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1574_157480


namespace NUMINAMATH_CALUDE_solution_set_equality_l1574_157483

-- Define a decreasing function on ℝ
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Define the set of x that satisfy the inequality
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | f (|1/x|) < f 1}

-- Theorem statement
theorem solution_set_equality (f : ℝ → ℝ) (h : DecreasingFunction f) :
  SolutionSet f = {x | -1 < x ∧ x < 0} ∪ {x | 0 < x ∧ x < 1} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equality_l1574_157483


namespace NUMINAMATH_CALUDE_triangle_formation_l1574_157405

/-- A function that checks if three stick lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The theorem stating which set of stick lengths can form a triangle -/
theorem triangle_formation :
  can_form_triangle 2 3 4 ∧
  ¬can_form_triangle 3 7 2 ∧
  ¬can_form_triangle 3 3 7 ∧
  ¬can_form_triangle 1 2 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l1574_157405


namespace NUMINAMATH_CALUDE_select_blocks_count_l1574_157476

def grid_size : ℕ := 6
def blocks_to_select : ℕ := 4

/-- The number of ways to select 4 blocks from a 6x6 grid, 
    such that no two blocks are in the same row or column -/
def select_blocks : ℕ := Nat.choose grid_size blocks_to_select * 
                         Nat.choose grid_size blocks_to_select * 
                         Nat.factorial blocks_to_select

theorem select_blocks_count : select_blocks = 5400 := by
  sorry

end NUMINAMATH_CALUDE_select_blocks_count_l1574_157476


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_and_ellipse_l1574_157441

-- Define the equation
def equation (y z : ℝ) : Prop :=
  z^4 - 6*y^4 = 3*z^2 - 8

-- Define what it means for the equation to represent a hyperbola
def represents_hyperbola (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (a b c d e f : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
    ∀ (y z : ℝ), eq y z ↔ a*y^2 + b*z^2 + c*y*z + d*y + e*z + f = 0

-- Define what it means for the equation to represent an ellipse
def represents_ellipse (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (a b c d e f : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (y z : ℝ), eq y z ↔ a*y^2 + b*z^2 + c*y*z + d*y + e*z + f = 0

-- Theorem statement
theorem equation_represents_hyperbola_and_ellipse :
  represents_hyperbola equation ∧ represents_ellipse equation :=
sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_and_ellipse_l1574_157441


namespace NUMINAMATH_CALUDE_tan_alpha_plus_beta_l1574_157492

theorem tan_alpha_plus_beta (α β : ℝ) 
  (h1 : 3 * Real.tan (α / 2) + Real.tan (α / 2) ^ 2 = 1)
  (h2 : Real.sin β = 3 * Real.sin (2 * α + β)) :
  Real.tan (α + β) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_beta_l1574_157492


namespace NUMINAMATH_CALUDE_initial_deposit_proof_l1574_157466

/-- Proves that the initial deposit is correct given the total savings goal,
    saving period, and weekly saving amount. -/
theorem initial_deposit_proof (total_goal : ℕ) (weeks : ℕ) (weekly_saving : ℕ) 
    (h1 : total_goal = 500)
    (h2 : weeks = 19)
    (h3 : weekly_saving = 17) : 
  total_goal - (weeks * weekly_saving) = 177 := by
  sorry

end NUMINAMATH_CALUDE_initial_deposit_proof_l1574_157466


namespace NUMINAMATH_CALUDE_factorization_of_18x_squared_minus_8_l1574_157428

theorem factorization_of_18x_squared_minus_8 (x : ℝ) : 18 * x^2 - 8 = 2 * (3*x + 2) * (3*x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_18x_squared_minus_8_l1574_157428


namespace NUMINAMATH_CALUDE_race_probability_l1574_157446

structure Race where
  total_cars : ℕ
  prob_x : ℚ
  prob_y : ℚ
  prob_z : ℚ
  no_dead_heat : Bool

def Race.prob_one_wins (r : Race) : ℚ :=
  r.prob_x + r.prob_y + r.prob_z

theorem race_probability (r : Race) 
  (h1 : r.total_cars = 10)
  (h2 : r.prob_x = 1 / 7)
  (h3 : r.prob_y = 1 / 3)
  (h4 : r.prob_z = 1 / 5)
  (h5 : r.no_dead_heat = true) :
  r.prob_one_wins = 71 / 105 := by
  sorry

end NUMINAMATH_CALUDE_race_probability_l1574_157446


namespace NUMINAMATH_CALUDE_frequency_in_interval_l1574_157440

def sample_capacity : ℕ := 100

def group_frequencies : List ℕ := [12, 13, 24, 15, 16, 13, 7]

def interval_sum : ℕ := 12 + 13 + 24 + 15

theorem frequency_in_interval :
  (interval_sum : ℚ) / sample_capacity = 0.64 := by sorry

end NUMINAMATH_CALUDE_frequency_in_interval_l1574_157440


namespace NUMINAMATH_CALUDE_min_value_inequality_l1574_157425

theorem min_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + 3 * c) + b / (8 * c + 4 * a) + 9 * c / (3 * a + 2 * b) ≥ 47 / 48 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1574_157425


namespace NUMINAMATH_CALUDE_club_membership_l1574_157469

theorem club_membership (total : ℕ) (lit : ℕ) (hist : ℕ) (both : ℕ) 
  (h1 : total = 80)
  (h2 : lit = 50)
  (h3 : hist = 40)
  (h4 : both = 25) :
  total - (lit + hist - both) = 15 := by
  sorry

end NUMINAMATH_CALUDE_club_membership_l1574_157469


namespace NUMINAMATH_CALUDE_sum_of_numbers_l1574_157423

theorem sum_of_numbers (a b : ℕ+) 
  (hcf : Nat.gcd a b = 5)
  (lcm : Nat.lcm a b = 120)
  (sum_reciprocals : (1 : ℚ) / a + (1 : ℚ) / b = 11 / 120) :
  a + b = 55 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l1574_157423


namespace NUMINAMATH_CALUDE_max_volume_box_l1574_157450

/-- Represents a rectangular box without a lid -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The volume of a box -/
def volume (b : Box) : ℝ := b.length * b.width * b.height

/-- The surface area of a box without a lid -/
def surfaceArea (b : Box) : ℝ := 
  b.length * b.width + 2 * b.height * (b.length + b.width)

/-- Theorem: Maximum volume of a box with given constraints -/
theorem max_volume_box : 
  ∃ (b : Box), 
    b.width = 2 ∧ 
    surfaceArea b = 32 ∧ 
    (∀ (b' : Box), b'.width = 2 → surfaceArea b' = 32 → volume b' ≤ volume b) ∧
    volume b = 16 := by
  sorry


end NUMINAMATH_CALUDE_max_volume_box_l1574_157450


namespace NUMINAMATH_CALUDE_scaled_standard_deviation_l1574_157467

def data := List ℝ

def variance (d : data) : ℝ := sorry

def standardDeviation (d : data) : ℝ := sorry

def scaleData (d : data) (k : ℝ) : data := sorry

theorem scaled_standard_deviation 
  (d : data) 
  (h : variance d = 2) : 
  standardDeviation (scaleData d 2) = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_scaled_standard_deviation_l1574_157467


namespace NUMINAMATH_CALUDE_range_of_m_l1574_157499

def f (x : ℝ) := -x^2 + 4*x

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc m 4, f x ∈ Set.Icc 0 4) ∧
  (∀ y ∈ Set.Icc 0 4, ∃ x ∈ Set.Icc m 4, f x = y) →
  m ∈ Set.Icc 0 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1574_157499


namespace NUMINAMATH_CALUDE_remaining_lives_l1574_157488

def initial_lives : ℕ := 98
def lives_lost : ℕ := 25

theorem remaining_lives : initial_lives - lives_lost = 73 := by
  sorry

end NUMINAMATH_CALUDE_remaining_lives_l1574_157488


namespace NUMINAMATH_CALUDE_water_needed_for_growth_medium_l1574_157477

/-- Given a growth medium mixture with initial volumes of nutrient concentrate and water,
    calculate the amount of water needed for a specified total volume. -/
theorem water_needed_for_growth_medium 
  (nutrient_vol : ℝ) 
  (initial_water_vol : ℝ) 
  (total_vol : ℝ) 
  (h1 : nutrient_vol = 0.08)
  (h2 : initial_water_vol = 0.04)
  (h3 : total_vol = 1) :
  (total_vol * initial_water_vol) / (nutrient_vol + initial_water_vol) = 1/3 := by
  sorry

#check water_needed_for_growth_medium

end NUMINAMATH_CALUDE_water_needed_for_growth_medium_l1574_157477


namespace NUMINAMATH_CALUDE_f_properties_l1574_157470

noncomputable def f (x : ℝ) := (2 * x - x^2) * Real.exp x

theorem f_properties :
  (∀ x ∈ Set.Ioo 0 2, f x > 0) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-Real.sqrt 2 - ε) (-Real.sqrt 2 + ε), f (-Real.sqrt 2) ≤ f x) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (Real.sqrt 2 - ε) (Real.sqrt 2 + ε), f x ≤ f (Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1574_157470


namespace NUMINAMATH_CALUDE_no_nonneg_integer_solution_l1574_157437

theorem no_nonneg_integer_solution (a b : ℕ) (ha : a ≠ b) :
  let d := Nat.gcd a b
  let a' := a / d
  let b' := b / d
  ∀ n : ℕ, (∀ x y : ℕ, a * x + b * y ≠ n) ↔ n = d * (a' * b' - a' - b') := by
  sorry

end NUMINAMATH_CALUDE_no_nonneg_integer_solution_l1574_157437


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l1574_157465

theorem hemisphere_surface_area (V : ℝ) (h : V = (500 / 3) * Real.pi) :
  ∃ (r : ℝ), V = (2 / 3) * Real.pi * r^3 ∧
             (2 * Real.pi * r^2 + Real.pi * r^2) = 3 * Real.pi * 250^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l1574_157465


namespace NUMINAMATH_CALUDE_sheila_hourly_wage_l1574_157415

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_long_day : ℕ  -- Hours worked on long days
  hours_short_day : ℕ -- Hours worked on short days
  long_days : ℕ       -- Number of long workdays per week
  short_days : ℕ      -- Number of short workdays per week
  weekly_earnings : ℕ -- Weekly earnings in dollars

/-- Calculates the hourly wage given a work schedule --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  let total_hours := schedule.hours_long_day * schedule.long_days + 
                     schedule.hours_short_day * schedule.short_days
  schedule.weekly_earnings / total_hours

/-- Sheila's actual work schedule --/
def sheila_schedule : WorkSchedule := {
  hours_long_day := 8,
  hours_short_day := 6,
  long_days := 3,
  short_days := 2,
  weekly_earnings := 468
}

/-- Theorem stating that Sheila's hourly wage is $13 --/
theorem sheila_hourly_wage : hourly_wage sheila_schedule = 13 := by
  sorry

end NUMINAMATH_CALUDE_sheila_hourly_wage_l1574_157415


namespace NUMINAMATH_CALUDE_derivative_even_implies_b_zero_l1574_157496

/-- A cubic function with a constant term of 2 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

/-- The derivative of f -/
def f' (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

/-- A function is even if f(x) = f(-x) for all x -/
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

theorem derivative_even_implies_b_zero (a b c : ℝ) :
  is_even (f' a b c) → b = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_even_implies_b_zero_l1574_157496


namespace NUMINAMATH_CALUDE_movie_of_the_year_threshold_l1574_157412

theorem movie_of_the_year_threshold (total_members : ℕ) (threshold_fraction : ℚ) : 
  total_members = 795 →
  threshold_fraction = 1/4 →
  ∃ n : ℕ, n ≥ total_members * threshold_fraction ∧ 
    ∀ m : ℕ, m ≥ total_members * threshold_fraction → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_movie_of_the_year_threshold_l1574_157412


namespace NUMINAMATH_CALUDE_johns_tax_rate_l1574_157463

theorem johns_tax_rate (john_income ingrid_income : ℝ)
  (ingrid_tax_rate combined_tax_rate : ℝ)
  (h1 : john_income = 56000)
  (h2 : ingrid_income = 74000)
  (h3 : ingrid_tax_rate = 0.4)
  (h4 : combined_tax_rate = 0.3569) :
  let total_income := john_income + ingrid_income
  let total_tax := combined_tax_rate * total_income
  let ingrid_tax := ingrid_tax_rate * ingrid_income
  let john_tax := total_tax - ingrid_tax
  john_tax / john_income = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_johns_tax_rate_l1574_157463


namespace NUMINAMATH_CALUDE_cauliflower_increase_40401_l1574_157456

/-- Represents the increase in cauliflower production from one year to the next,
    given a square garden where each cauliflower takes 1 square foot. -/
def cauliflower_increase (this_year_production : ℕ) : ℕ :=
  this_year_production - (Nat.sqrt this_year_production - 1)^2

/-- Theorem stating that for a square garden with 40401 cauliflowers this year,
    the increase in production from last year is 401 cauliflowers. -/
theorem cauliflower_increase_40401 :
  cauliflower_increase 40401 = 401 := by
  sorry

#eval cauliflower_increase 40401

end NUMINAMATH_CALUDE_cauliflower_increase_40401_l1574_157456


namespace NUMINAMATH_CALUDE_andre_carl_speed_ratio_l1574_157473

/-- 
Given two runners, Carl and André, with the following conditions:
- Carl runs at a constant speed of x meters per second
- André runs at a constant speed of y meters per second
- André starts running 20 seconds after Carl
- André catches up to Carl after running for 10 seconds

Prove that the ratio of André's speed to Carl's speed is 3:1
-/
theorem andre_carl_speed_ratio 
  (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h_catchup : 10 * y = 30 * x) : 
  y / x = 3 := by
sorry

end NUMINAMATH_CALUDE_andre_carl_speed_ratio_l1574_157473


namespace NUMINAMATH_CALUDE_product_equals_one_l1574_157421

theorem product_equals_one (x y : ℝ) 
  (h : x * y - x / (y^2) - y / (x^2) + x^2 / (y^3) = 4) : 
  (x - 2) * (y - 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_one_l1574_157421


namespace NUMINAMATH_CALUDE_fruit_theorem_l1574_157489

def fruit_problem (apples pears plums cherries : ℕ) : Prop :=
  apples = 180 ∧
  apples = 3 * plums ∧
  pears = 2 * plums ∧
  cherries = 4 * apples ∧
  251 = apples - (13 * apples / 15) +
        plums - (5 * plums / 6) +
        pears - (3 * pears / 4) +
        cherries - (37 * cherries / 50)

theorem fruit_theorem :
  ∃ (apples pears plums cherries : ℕ),
    fruit_problem apples pears plums cherries := by
  sorry

end NUMINAMATH_CALUDE_fruit_theorem_l1574_157489


namespace NUMINAMATH_CALUDE_min_ones_in_valid_grid_l1574_157414

/-- A grid of zeros and ones -/
def Grid := Matrix (Fin 11) (Fin 11) Bool

/-- The sum of elements in a 2x2 subgrid is odd -/
def valid_subgrid (g : Grid) (i j : Fin 10) : Prop :=
  (g i j).toNat + (g i (j+1)).toNat + (g (i+1) j).toNat + (g (i+1) (j+1)).toNat % 2 = 1

/-- A grid is valid if all its 2x2 subgrids have odd sum -/
def valid_grid (g : Grid) : Prop :=
  ∀ i j : Fin 10, valid_subgrid g i j

/-- Count the number of ones in a grid -/
def count_ones (g : Grid) : Nat :=
  (Finset.univ.sum fun i => (Finset.univ.sum fun j => (g i j).toNat))

/-- The main theorem: the minimum number of ones in a valid 11x11 grid is 25 -/
theorem min_ones_in_valid_grid :
  ∃ (g : Grid), valid_grid g ∧ count_ones g = 25 ∧
  ∀ (h : Grid), valid_grid h → count_ones h ≥ 25 :=
sorry

end NUMINAMATH_CALUDE_min_ones_in_valid_grid_l1574_157414


namespace NUMINAMATH_CALUDE_problem_statement_l1574_157481

theorem problem_statement (x y z : ℝ) (h : (5 : ℝ) ^ x = (9 : ℝ) ^ y ∧ (9 : ℝ) ^ y = (225 : ℝ) ^ z) : 
  1 / z = 2 / x + 1 / y := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1574_157481


namespace NUMINAMATH_CALUDE_megan_markers_count_l1574_157474

/-- The total number of markers Megan has after receiving more from Robert -/
def total_markers (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Megan's total markers is the sum of her initial markers and those received from Robert -/
theorem megan_markers_count (initial : ℕ) (received : ℕ) :
  total_markers initial received = initial + received :=
by
  sorry

end NUMINAMATH_CALUDE_megan_markers_count_l1574_157474


namespace NUMINAMATH_CALUDE_function_properties_l1574_157491

def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def IsOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def IsSymmetricAboutPoint (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

def IsSymmetricAboutYAxis (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem function_properties (f : ℝ → ℝ) 
    (h1 : ∀ x, f (x + 3/2) + f x = 0)
    (h2 : IsOddFunction (fun x ↦ f (x - 3/4))) :
  (IsPeriodic f 3 ∧ 
   ¬ IsPeriodic f (3/2)) ∧ 
  IsSymmetricAboutPoint f (-3/4) ∧ 
  ¬ IsSymmetricAboutYAxis f :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1574_157491


namespace NUMINAMATH_CALUDE_log_five_twelve_l1574_157424

theorem log_five_twelve (a b : ℝ) (h1 : Real.log 2 = a * Real.log 10) (h2 : Real.log 3 = b * Real.log 10) :
  Real.log 12 / Real.log 5 = (2*a + b) / (1 - a) := by
  sorry

end NUMINAMATH_CALUDE_log_five_twelve_l1574_157424


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l1574_157416

/-- Theorem: Given a total of 6000 votes and a candidate losing by 1800 votes,
    the percentage of votes the candidate received is 35%. -/
theorem candidate_vote_percentage
  (total_votes : ℕ)
  (vote_difference : ℕ)
  (h_total : total_votes = 6000)
  (h_diff : vote_difference = 1800) :
  (total_votes - vote_difference) * 100 / (2 * total_votes) = 35 := by
  sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l1574_157416


namespace NUMINAMATH_CALUDE_leftover_coin_value_l1574_157439

/-- Represents the number of coins in a complete roll --/
structure RollSize where
  quarters : Nat
  dimes : Nat

/-- Represents the number of coins a person has --/
structure CoinCount where
  quarters : Nat
  dimes : Nat

/-- Calculates the value of coins in dollars --/
def coinValue (quarters dimes : Nat) : Rat :=
  (quarters * 25 + dimes * 10) / 100

theorem leftover_coin_value
  (charles marta : CoinCount)
  (roll_size : RollSize)
  (h1 : charles.quarters = 57)
  (h2 : charles.dimes = 216)
  (h3 : marta.quarters = 88)
  (h4 : marta.dimes = 193)
  (h5 : roll_size.quarters = 50)
  (h6 : roll_size.dimes = 40) :
  let total_quarters := charles.quarters + marta.quarters
  let total_dimes := charles.dimes + marta.dimes
  let leftover_quarters := total_quarters % roll_size.quarters
  let leftover_dimes := total_dimes % roll_size.dimes
  coinValue leftover_quarters leftover_dimes = 1215 / 100 := by
  sorry

end NUMINAMATH_CALUDE_leftover_coin_value_l1574_157439


namespace NUMINAMATH_CALUDE_train_length_l1574_157455

/-- The length of a train given its speed, platform length, and crossing time -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 96 * (5 / 18) →
  platform_length = 480 →
  crossing_time = 36 →
  ∃ (train_length : ℝ), abs (train_length - 480.12) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1574_157455


namespace NUMINAMATH_CALUDE_intersection_and_parallel_line_l1574_157434

/-- Given two lines in R², prove their intersection point and a parallel line through that point. -/
theorem intersection_and_parallel_line 
  (l₁ : Set (ℝ × ℝ)) 
  (l₂ : Set (ℝ × ℝ))
  (h₁ : l₁ = {p : ℝ × ℝ | p.1 + 8 * p.2 + 7 = 0})
  (h₂ : l₂ = {p : ℝ × ℝ | 2 * p.1 + p.2 - 1 = 0})
  (l₃ : Set (ℝ × ℝ))
  (h₃ : l₃ = {p : ℝ × ℝ | p.1 + p.2 + 1 = 0}) :
  (∃! p : ℝ × ℝ, p ∈ l₁ ∧ p ∈ l₂ ∧ p = (1, -1)) ∧
  (∃ l : Set (ℝ × ℝ), l = {p : ℝ × ℝ | p.1 + p.2 = 0} ∧ 
    (1, -1) ∈ l ∧ 
    ∀ (p q : ℝ × ℝ), p ∈ l ∧ q ∈ l → p.1 - q.1 = q.2 - p.2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_parallel_line_l1574_157434


namespace NUMINAMATH_CALUDE_sequence_ratio_l1574_157497

def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def arithmetic_square_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n * (2 * a₁^2 + (n - 1) * d^2 + (n - 1) * d * a₁)) / 2

theorem sequence_ratio :
  let n := (38 - 4) / 2 + 1
  arithmetic_sum 4 2 n / arithmetic_square_sum 3 3 n = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_sequence_ratio_l1574_157497


namespace NUMINAMATH_CALUDE_sandy_correct_sums_l1574_157449

theorem sandy_correct_sums (total_sums : ℕ) (total_marks : ℤ) 
  (correct_marks : ℕ) (incorrect_marks : ℕ) :
  total_sums = 30 →
  total_marks = 50 →
  correct_marks = 3 →
  incorrect_marks = 2 →
  ∃ (correct : ℕ) (incorrect : ℕ),
    correct + incorrect = total_sums ∧
    correct_marks * correct - incorrect_marks * incorrect = total_marks ∧
    correct = 22 := by
  sorry

end NUMINAMATH_CALUDE_sandy_correct_sums_l1574_157449


namespace NUMINAMATH_CALUDE_walters_age_2009_l1574_157475

theorem walters_age_2009 (walter_age_2004 : ℝ) (grandmother_age_2004 : ℝ) : 
  walter_age_2004 = grandmother_age_2004 / 3 →
  (2004 - walter_age_2004) + (2004 - grandmother_age_2004) = 4018 →
  walter_age_2004 + 5 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_walters_age_2009_l1574_157475


namespace NUMINAMATH_CALUDE_multiply_difference_of_cubes_l1574_157431

theorem multiply_difference_of_cubes (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_difference_of_cubes_l1574_157431


namespace NUMINAMATH_CALUDE_at_least_one_zero_l1574_157402

theorem at_least_one_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^9 + b^9) * (b^9 + c^9) * (c^9 + a^9) = (a * b * c)^9) :
  a = 0 ∨ b = 0 ∨ c = 0 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_zero_l1574_157402


namespace NUMINAMATH_CALUDE_passes_through_neg1_0_two_a_plus_c_positive_roots_between_neg3_and_1_l1574_157452

/-- A parabola defined by y = ax^2 - 2ax + c, where a and c are constants, a ≠ 0, c > 0,
    and the parabola passes through the point (3,0) -/
structure Parabola where
  a : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  c_positive : c > 0
  passes_through_3_0 : a * 3^2 - 2 * a * 3 + c = 0

/-- The parabola passes through the point (-1,0) -/
theorem passes_through_neg1_0 (p : Parabola) : p.a * (-1)^2 - 2 * p.a * (-1) + p.c = 0 := by sorry

/-- 2a + c > 0 -/
theorem two_a_plus_c_positive (p : Parabola) : 2 * p.a + p.c > 0 := by sorry

/-- If m and n (m < n) are the two roots of ax^2 + 2ax + c = p, where p > 0,
    then -3 < m < n < 1 -/
theorem roots_between_neg3_and_1 (p : Parabola) (m n : ℝ) (p_pos : ℝ) 
  (h_roots : m < n ∧ p.a * m^2 + 2 * p.a * m + p.c = p_pos ∧ p.a * n^2 + 2 * p.a * n + p.c = p_pos)
  (h_p_pos : p_pos > 0) : -3 < m ∧ m < n ∧ n < 1 := by sorry

end NUMINAMATH_CALUDE_passes_through_neg1_0_two_a_plus_c_positive_roots_between_neg3_and_1_l1574_157452


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_90_l1574_157498

theorem distinct_prime_factors_of_90 : Nat.card (Nat.factors 90).toFinset = 3 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_90_l1574_157498


namespace NUMINAMATH_CALUDE_pencil_buyers_difference_l1574_157478

theorem pencil_buyers_difference (price : ℕ) 
  (h1 : price > 0)
  (h2 : 234 % price = 0)
  (h3 : 325 % price = 0) :
  325 / price - 234 / price = 7 := by
  sorry

end NUMINAMATH_CALUDE_pencil_buyers_difference_l1574_157478


namespace NUMINAMATH_CALUDE_min_distance_to_origin_l1574_157451

/-- The minimum distance between a point on the line x - 2y + 2 = 0 and the origin -/
theorem min_distance_to_origin : 
  ∃ (d : ℝ), d = 2 * Real.sqrt 5 / 5 ∧ 
  ∀ (P : ℝ × ℝ), P.1 - 2 * P.2 + 2 = 0 → 
  Real.sqrt (P.1^2 + P.2^2) ≥ d := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_origin_l1574_157451


namespace NUMINAMATH_CALUDE_smallest_integer_y_minus_three_is_smallest_l1574_157485

theorem smallest_integer_y (y : ℤ) : 3 - 5 * y < 23 ↔ y ≥ -3 :=
  sorry

theorem minus_three_is_smallest : ∃ (y : ℤ), 3 - 5 * y < 23 ∧ ∀ (z : ℤ), 3 - 5 * z < 23 → z ≥ y :=
  sorry

end NUMINAMATH_CALUDE_smallest_integer_y_minus_three_is_smallest_l1574_157485


namespace NUMINAMATH_CALUDE_rectangle_rotation_l1574_157433

theorem rectangle_rotation (w : ℝ) (a : ℝ) (l : ℝ) : 
  w = 6 →
  (1/4) * Real.pi * (l^2 + w^2) = a →
  a = 45 * Real.pi →
  l = 12 := by
sorry

end NUMINAMATH_CALUDE_rectangle_rotation_l1574_157433


namespace NUMINAMATH_CALUDE_geometric_arithmetic_progression_sum_l1574_157495

theorem geometric_arithmetic_progression_sum : 
  ∃ (a b : ℝ), 
    3 < a ∧ a < b ∧ b < 9 ∧ 
    (∃ (r : ℝ), r > 0 ∧ a = 3 * r ∧ b = 3 * r^2) ∧ 
    (∃ (d : ℝ), b = a + d ∧ 9 = b + d) ∧ 
    a + b = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_progression_sum_l1574_157495


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l1574_157458

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 15 → ¬(Nat.Prime p ∧ p ∣ n)

theorem smallest_composite_no_small_factors : 
  (is_composite 289 ∧ has_no_small_prime_factors 289) ∧ 
  (∀ m : ℕ, m < 289 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l1574_157458


namespace NUMINAMATH_CALUDE_range_of_a_p_or_q_range_of_a_p_or_q_not_p_and_q_l1574_157490

def p (a : ℝ) : Prop := 
  (a > 3 ∨ (1 < a ∧ a < 2))

def q (a : ℝ) : Prop := 
  (2 < a ∧ a < 4)

theorem range_of_a_p_or_q (a : ℝ) : 
  p a ∨ q a → a ∈ Set.union (Set.Ioo 1 2) (Set.Ioi 2) := by
  sorry

theorem range_of_a_p_or_q_not_p_and_q (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → 
  a ∈ Set.union (Set.union (Set.Ioo 1 2) (Set.Ico 2 3)) (Set.Ici 4) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_p_or_q_range_of_a_p_or_q_not_p_and_q_l1574_157490


namespace NUMINAMATH_CALUDE_sqrt_of_square_of_negative_l1574_157427

theorem sqrt_of_square_of_negative : ∀ (x : ℝ), x < 0 → Real.sqrt (x^2) = -x := by sorry

end NUMINAMATH_CALUDE_sqrt_of_square_of_negative_l1574_157427


namespace NUMINAMATH_CALUDE_zhuhai_visitors_scientific_notation_l1574_157430

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem zhuhai_visitors_scientific_notation :
  toScientificNotation 3001000 = ScientificNotation.mk 3.001 6 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_zhuhai_visitors_scientific_notation_l1574_157430


namespace NUMINAMATH_CALUDE_nehas_mother_twice_age_l1574_157419

/-- Represents the age difference between Neha's mother and Neha when the mother will be twice Neha's age -/
def AgeDifference (n : ℕ) : Prop :=
  ∃ (neha_age : ℕ),
    -- Neha's mother's current age is 60
    60 = neha_age + n ∧
    -- 12 years ago, Neha's mother was 4 times Neha's age
    (60 - 12) = 4 * (neha_age - 12) ∧
    -- In n years, Neha's mother will be twice as old as Neha
    (60 + n) = 2 * (neha_age + n)

/-- The number of years until Neha's mother is twice as old as Neha is 12 -/
theorem nehas_mother_twice_age : AgeDifference 12 := by
  sorry

end NUMINAMATH_CALUDE_nehas_mother_twice_age_l1574_157419


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l1574_157471

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l1574_157471


namespace NUMINAMATH_CALUDE_vote_increase_l1574_157435

/-- Represents the voting scenario for a bill --/
structure VotingScenario where
  total_members : ℕ
  initial_for : ℕ
  initial_against : ℕ
  revote_for : ℕ
  revote_against : ℕ

/-- Conditions for the voting scenario --/
def voting_conditions (v : VotingScenario) : Prop :=
  v.total_members = 500 ∧
  v.initial_for + v.initial_against = v.total_members ∧
  v.initial_against > v.initial_for ∧
  v.revote_for + v.revote_against = v.total_members ∧
  v.revote_for = (10 * v.initial_against) / 9 ∧
  (v.revote_for - v.revote_against) = 3 * (v.initial_against - v.initial_for)

/-- Theorem stating the increase in votes for the bill --/
theorem vote_increase (v : VotingScenario) (h : voting_conditions v) :
  v.revote_for - v.initial_for = 59 :=
sorry

end NUMINAMATH_CALUDE_vote_increase_l1574_157435


namespace NUMINAMATH_CALUDE_ellipse_equation_l1574_157404

/-- An ellipse with given properties -/
structure Ellipse where
  -- Foci are on the x-axis
  foci_on_x_axis : Bool
  -- Passes through (0,1) and (3,0)
  passes_through_points : (ℝ × ℝ) → (ℝ × ℝ) → Prop
  -- Eccentricity is 3/5
  eccentricity : ℚ
  -- Length of minor axis is 8
  minor_axis_length : ℝ

/-- The standard equation of an ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 16 = 1

/-- Theorem: The standard equation of the ellipse with given properties -/
theorem ellipse_equation (e : Ellipse) 
  (h1 : e.foci_on_x_axis = true)
  (h2 : e.passes_through_points (0, 1) (3, 0))
  (h3 : e.eccentricity = 3/5)
  (h4 : e.minor_axis_length = 8) :
  ∀ x y : ℝ, standard_equation e x y :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1574_157404


namespace NUMINAMATH_CALUDE_smallest_perfect_cube_l1574_157484

/-- Given distinct prime numbers p, q, and r, prove that (p * q * r^2)^3 is the smallest positive
    perfect cube that includes the factor n = p * q^2 * r^4 -/
theorem smallest_perfect_cube (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r)
  (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  ∃ (k : ℕ), k > 0 ∧ (p * q * r^2)^3 = k^3 ∧
  ∀ (m : ℕ), m > 0 → m^3 ≥ (p * q * r^2)^3 → (p * q^2 * r^4) ∣ m^3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_perfect_cube_l1574_157484


namespace NUMINAMATH_CALUDE_order_of_abc_l1574_157420

theorem order_of_abc (a b c : ℝ) : 
  a = 5^(1/5) → b = Real.log 3 / Real.log π → c = Real.log 0.2 / Real.log 5 → a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_order_of_abc_l1574_157420


namespace NUMINAMATH_CALUDE_teacup_cost_function_l1574_157494

-- Define the cost of a single teacup
def teacup_cost : ℚ := 2.5

-- Define the function for the total cost
def total_cost (x : ℕ+) : ℚ := x.val * teacup_cost

-- Theorem statement
theorem teacup_cost_function (x : ℕ+) (y : ℚ) :
  y = total_cost x ↔ y = 2.5 * x.val := by sorry

end NUMINAMATH_CALUDE_teacup_cost_function_l1574_157494


namespace NUMINAMATH_CALUDE_zero_count_in_circular_sequence_l1574_157482

/-- Represents a circular sequence without repetitions -/
structure CircularSequence (α : Type) where
  elements : List α
  no_repetitions : elements.Nodup
  circular : elements ≠ []

/-- Counts the number of occurrences of an element in a list -/
def count (α : Type) [DecidableEq α] (l : List α) (x : α) : Nat :=
  l.filter (· = x) |>.length

/-- Theorem: The number of zeroes in a circular sequence without repetitions is 0, 1, 2, or 4 -/
theorem zero_count_in_circular_sequence (m : ℕ) (seq : CircularSequence ℕ) :
  let zero_count := count ℕ seq.elements 0
  zero_count = 0 ∨ zero_count = 1 ∨ zero_count = 2 ∨ zero_count = 4 :=
sorry

end NUMINAMATH_CALUDE_zero_count_in_circular_sequence_l1574_157482


namespace NUMINAMATH_CALUDE_monthly_salary_calculation_l1574_157461

/-- Proves that a person's monthly salary is 6000 given the specified savings conditions -/
theorem monthly_salary_calculation (salary : ℝ) : 
  (salary * 0.2 = salary - (salary * 0.8 * 1.2 + 240)) → salary = 6000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_salary_calculation_l1574_157461


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l1574_157468

/-- Represents a vertical shift of a parabola -/
def vertical_shift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x => f x + shift

/-- The original parabola function -/
def original_parabola : ℝ → ℝ := λ x => -x^2

theorem parabola_shift_theorem :
  vertical_shift original_parabola 2 = λ x => -x^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l1574_157468


namespace NUMINAMATH_CALUDE_rock_paper_scissors_games_l1574_157401

/-- The number of students in the group -/
def num_students : ℕ := 9

/-- The number of neighbors each student doesn't play with -/
def neighbors : ℕ := 2

/-- The number of games each student plays -/
def games_per_student : ℕ := num_students - 1 - neighbors

/-- The total number of games played, counting each game twice -/
def total_games : ℕ := num_students * games_per_student

/-- The number of unique games played -/
def unique_games : ℕ := total_games / 2

theorem rock_paper_scissors_games :
  unique_games = 27 :=
sorry

end NUMINAMATH_CALUDE_rock_paper_scissors_games_l1574_157401


namespace NUMINAMATH_CALUDE_solution_count_33_l1574_157417

/-- The number of solutions to 3x + 2y + z = n in positive integers x, y, z -/
def solution_count (n : ℕ+) : ℕ := sorry

/-- The set of possible values for n -/
def possible_values : Set ℕ+ := {22, 24, 25}

/-- Theorem: If the equation 3x + 2y + z = n has exactly 33 solutions in positive integers x, y, and z,
    then n is in the set {22, 24, 25} -/
theorem solution_count_33 (n : ℕ+) : solution_count n = 33 → n ∈ possible_values := by sorry

end NUMINAMATH_CALUDE_solution_count_33_l1574_157417


namespace NUMINAMATH_CALUDE_chebyshev_birth_year_l1574_157403

def is_valid_year (year : Nat) : Prop :=
  -- Year is in the 19th century
  1800 ≤ year ∧ year < 1900 ∧
  -- Sum of hundreds and thousands digits is 3 times sum of units and tens digits
  (year / 100 + (year / 1000) % 10) = 3 * ((year % 10) + (year / 10) % 10) ∧
  -- Tens digit is greater than units digit
  (year / 10) % 10 > year % 10 ∧
  -- Chebyshev lived for 73 years and died in the same century
  year + 73 < 1900

theorem chebyshev_birth_year :
  ∀ year : Nat, is_valid_year year ↔ year = 1821 := by sorry

end NUMINAMATH_CALUDE_chebyshev_birth_year_l1574_157403


namespace NUMINAMATH_CALUDE_ring_arrangements_count_l1574_157460

def choose (n k : ℕ) : ℕ := Nat.choose n k

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem ring_arrangements_count : 
  let total_rings : ℕ := 10
  let arranged_rings : ℕ := 6
  let fingers : ℕ := 4
  choose total_rings arranged_rings * factorial arranged_rings * choose (arranged_rings + fingers - 1) (fingers - 1) = 9130560 := by
  sorry

end NUMINAMATH_CALUDE_ring_arrangements_count_l1574_157460


namespace NUMINAMATH_CALUDE_proportion_reciprocal_outer_terms_l1574_157418

theorem proportion_reciprocal_outer_terms (a b c d : ℚ) : 
  (a / b = c / d) →  -- proportion
  (b * c = 1) →      -- middle terms are reciprocals
  (a = 7 / 9) →      -- one outer term is 7/9
  (d = 9 / 7) :=     -- other outer term is 9/7
by
  sorry


end NUMINAMATH_CALUDE_proportion_reciprocal_outer_terms_l1574_157418


namespace NUMINAMATH_CALUDE_conjugate_2023_l1574_157464

/-- Conjugate point in 2D space -/
def conjugate (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2 + 1, p.1 + 1)

/-- Sequence of conjugate points -/
def conjugateSequence : ℕ → ℝ × ℝ
  | 0 => (2, 2)
  | n + 1 => conjugate (conjugateSequence n)

theorem conjugate_2023 :
  conjugateSequence 2023 = (-2, 0) := by sorry

end NUMINAMATH_CALUDE_conjugate_2023_l1574_157464


namespace NUMINAMATH_CALUDE_cone_section_height_ratio_l1574_157409

/-- Given a cone with height h and base radius r, if a cross-section parallel to the base
    has an area that is half of the base area, then the ratio of the height of this section
    to the remaining height of the cone is 1:(√2 - 1). -/
theorem cone_section_height_ratio (h r x : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  let base_area := π * r^2
  let section_area := π * (r * x / h)^2
  section_area = base_area / 2 →
  x / (h - x) = 1 / (Real.sqrt 2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_cone_section_height_ratio_l1574_157409


namespace NUMINAMATH_CALUDE_base_number_proof_l1574_157429

theorem base_number_proof (x n : ℕ) (h1 : 4 * x^(2*n) = 4^26) (h2 : n = 25) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l1574_157429


namespace NUMINAMATH_CALUDE_square_sum_from_product_and_sum_l1574_157487

theorem square_sum_from_product_and_sum (r s : ℝ) 
  (h1 : r * s = 24) 
  (h2 : r + s = 10) : 
  r^2 + s^2 = 52 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_product_and_sum_l1574_157487


namespace NUMINAMATH_CALUDE_stability_of_nonlinear_eq_l1574_157411

/-- The nonlinear differential equation dx/dt = 1 - x^2(t) -/
def diff_eq (x : ℝ → ℝ) : Prop :=
  ∀ t, deriv x t = 1 - (x t)^2

/-- Definition of an equilibrium point -/
def is_equilibrium_point (x : ℝ) (eq : (ℝ → ℝ) → Prop) : Prop :=
  eq (λ _ => x)

/-- Definition of asymptotic stability -/
def is_asymptotically_stable (x : ℝ) (eq : (ℝ → ℝ) → Prop) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x₀, |x₀ - x| < δ → 
    ∀ sol, eq sol → sol 0 = x₀ → ∀ t ≥ 0, |sol t - x| < ε

/-- Definition of instability -/
def is_unstable (x : ℝ) (eq : (ℝ → ℝ) → Prop) : Prop :=
  ∃ ε > 0, ∀ δ > 0, ∃ x₀, |x₀ - x| < δ ∧
    ∃ sol, eq sol ∧ sol 0 = x₀ ∧ ∃ t ≥ 0, |sol t - x| ≥ ε

/-- Theorem about the stability of the nonlinear differential equation -/
theorem stability_of_nonlinear_eq :
  (is_equilibrium_point 1 diff_eq ∧ is_equilibrium_point (-1) diff_eq) ∧
  (is_asymptotically_stable 1 diff_eq) ∧
  (is_unstable (-1) diff_eq) :=
sorry

end NUMINAMATH_CALUDE_stability_of_nonlinear_eq_l1574_157411


namespace NUMINAMATH_CALUDE_jack_queen_king_prob_in_standard_deck_l1574_157448

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (jacks : ℕ)
  (queens : ℕ)
  (kings : ℕ)

/-- Calculates the probability of drawing a specific card from a deck -/
def draw_probability (n : ℕ) (total : ℕ) : ℚ :=
  n / total

/-- Calculates the probability of drawing a Jack, then a Queen, then a King -/
def jack_queen_king_probability (d : Deck) : ℚ :=
  (draw_probability d.jacks d.total_cards) *
  (draw_probability d.queens (d.total_cards - 1)) *
  (draw_probability d.kings (d.total_cards - 2))

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52
  , jacks := 4
  , queens := 4
  , kings := 4 }

theorem jack_queen_king_prob_in_standard_deck :
  jack_queen_king_probability standard_deck = 8 / 16575 :=
by sorry

end NUMINAMATH_CALUDE_jack_queen_king_prob_in_standard_deck_l1574_157448


namespace NUMINAMATH_CALUDE_math_contest_problem_l1574_157447

theorem math_contest_problem (a b c d e f g : ℕ) : 
  a + b + c + d + e + f + g = 25 →
  b + d = 2 * (c + d) →
  a = 1 + (e + f + g) →
  a = b + c →
  b = 6 :=
by sorry

end NUMINAMATH_CALUDE_math_contest_problem_l1574_157447


namespace NUMINAMATH_CALUDE_g_of_two_eq_zero_l1574_157422

/-- Given a function g(x) = x^2 - 4 for all real x, prove that g(2) = 0 -/
theorem g_of_two_eq_zero (g : ℝ → ℝ) (h : ∀ x, g x = x^2 - 4) : g 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_of_two_eq_zero_l1574_157422


namespace NUMINAMATH_CALUDE_carls_cupcake_goal_l1574_157438

/-- Given Carl's cupcake selling goal and payment obligation, prove the number of cupcakes he must sell per day. -/
theorem carls_cupcake_goal (goal : ℕ) (days : ℕ) (payment : ℕ) (cupcakes_per_day : ℕ) 
    (h1 : goal = 96) 
    (h2 : days = 2) 
    (h3 : payment = 24) 
    (h4 : cupcakes_per_day * days = goal + payment) : 
  cupcakes_per_day = 60 := by
  sorry

#check carls_cupcake_goal

end NUMINAMATH_CALUDE_carls_cupcake_goal_l1574_157438
