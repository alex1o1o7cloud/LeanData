import Mathlib

namespace NUMINAMATH_CALUDE_triangle_50_40_l2017_201737

-- Define the triangle operation
def triangle (a b : ℤ) : ℤ := a * b + (a - b) + 6

-- Theorem statement
theorem triangle_50_40 : triangle 50 40 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_triangle_50_40_l2017_201737


namespace NUMINAMATH_CALUDE_cow_problem_l2017_201766

theorem cow_problem (purchase_price daily_food_cost additional_costs selling_price profit : ℕ) 
  (h1 : purchase_price = 600)
  (h2 : daily_food_cost = 20)
  (h3 : additional_costs = 500)
  (h4 : selling_price = 2500)
  (h5 : profit = 600) :
  ∃ days : ℕ, days = (selling_price - profit - purchase_price - additional_costs) / daily_food_cost ∧ days = 40 :=
sorry

end NUMINAMATH_CALUDE_cow_problem_l2017_201766


namespace NUMINAMATH_CALUDE_power_mod_thirteen_l2017_201786

theorem power_mod_thirteen :
  5^2023 ≡ 8 [ZMOD 13] := by
sorry

end NUMINAMATH_CALUDE_power_mod_thirteen_l2017_201786


namespace NUMINAMATH_CALUDE_pen_problem_solution_l2017_201731

/-- Represents the number of pens of each color in Maria's desk drawer. -/
structure PenCounts where
  red : ℕ
  black : ℕ
  blue : ℕ

/-- The conditions of the pen problem. -/
def penProblem (p : PenCounts) : Prop :=
  p.red = 8 ∧
  p.black > p.red ∧
  p.blue = p.red + 7 ∧
  p.red + p.black + p.blue = 41

/-- The theorem stating the solution to the pen problem. -/
theorem pen_problem_solution (p : PenCounts) (h : penProblem p) : 
  p.black - p.red = 10 := by
  sorry

end NUMINAMATH_CALUDE_pen_problem_solution_l2017_201731


namespace NUMINAMATH_CALUDE_slope_product_of_triple_angle_and_slope_l2017_201720

/-- Given two non-horizontal lines with slopes m and n, where one line forms
    three times as large an angle with the horizontal as the other and has
    three times the slope, prove that mn = 9/4 -/
theorem slope_product_of_triple_angle_and_slope
  (m n : ℝ) -- slopes of the lines
  (h₁ : m ≠ 0) -- L₁ is not horizontal
  (h₂ : n ≠ 0) -- L₂ is not horizontal
  (h₃ : ∃ θ₁ θ₂ : ℝ, θ₁ = 3 * θ₂ ∧ m = Real.tan θ₁ ∧ n = Real.tan θ₂) -- angle relation
  (h₄ : m = 3 * n) -- slope relation
  : m * n = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_slope_product_of_triple_angle_and_slope_l2017_201720


namespace NUMINAMATH_CALUDE_shopping_mall_investment_strategy_l2017_201788

/-- Profit when selling at the beginning of the month -/
def profit_beginning (x : ℝ) : ℝ := x * (1 + 0.15) * (1 + 0.10) - x

/-- Profit when selling at the end of the month -/
def profit_end (x : ℝ) : ℝ := x * (1 + 0.30) - x - 700

theorem shopping_mall_investment_strategy :
  (profit_beginning 15000 > profit_end 15000) ∧
  (profit_end 30000 > profit_beginning 30000) ∧
  (∀ x y : ℝ, profit_beginning x = 6000 ∧ profit_end y = 6000 → y < x) ∧
  (∀ x y : ℝ, profit_beginning x = 5300 ∧ profit_end y = 5300 → y < x) :=
sorry

end NUMINAMATH_CALUDE_shopping_mall_investment_strategy_l2017_201788


namespace NUMINAMATH_CALUDE_two_digit_factorizations_of_2210_l2017_201732

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def valid_factorization (a b : ℕ) : Prop :=
  is_two_digit a ∧ is_two_digit b ∧ a * b = 2210

def distinct_factorizations (f1 f2 : ℕ × ℕ) : Prop :=
  f1.1 ≠ f2.1 ∧ f1.1 ≠ f2.2

theorem two_digit_factorizations_of_2210 :
  ∃ (f1 f2 : ℕ × ℕ),
    valid_factorization f1.1 f1.2 ∧
    valid_factorization f2.1 f2.2 ∧
    distinct_factorizations f1 f2 ∧
    ∀ (f : ℕ × ℕ), valid_factorization f.1 f.2 →
      (f = f1 ∨ f = f2 ∨ f = (f1.2, f1.1) ∨ f = (f2.2, f2.1)) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_factorizations_of_2210_l2017_201732


namespace NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l2017_201702

theorem lcm_from_hcf_and_product (a b : ℕ+) : 
  Nat.gcd a b = 14 → a * b = 2562 → Nat.lcm a b = 183 := by
sorry

end NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l2017_201702


namespace NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l2017_201758

theorem smallest_factor_for_perfect_square : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (m : ℕ), 1152 * n = m^2) ∧ 
  (∀ (k : ℕ), k > 0 → k < n → ¬∃ (m : ℕ), 1152 * k = m^2) ∧
  n = 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l2017_201758


namespace NUMINAMATH_CALUDE_probability_all_players_have_initial_coins_l2017_201708

/-- Represents a player in the game -/
inductive Player : Type
| Alice : Player
| Bob : Player
| Charlie : Player
| Dana : Player

/-- Represents a ball color -/
inductive BallColor : Type
| Blue : BallColor
| Red : BallColor
| White : BallColor
| Yellow : BallColor

/-- Represents the state of the game -/
structure GameState :=
  (coins : Player → ℕ)
  (round : ℕ)

/-- Represents a single round of the game -/
def play_round (state : GameState) : GameState :=
  sorry

/-- Probability of a specific outcome in a single round -/
def round_probability : ℚ :=
  12 / 120

/-- The game consists of 5 rounds -/
def num_rounds : ℕ := 5

/-- The initial number of coins for each player -/
def initial_coins : ℕ := 5

/-- Theorem stating the probability of all players having the initial number of coins after the game -/
theorem probability_all_players_have_initial_coins :
  (round_probability ^ num_rounds : ℚ) = 1 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_players_have_initial_coins_l2017_201708


namespace NUMINAMATH_CALUDE_sin_B_in_triangle_l2017_201746

theorem sin_B_in_triangle (A B C : Real) (AB BC : Real) :
  A = 2 * π / 3 →  -- 120° in radians
  AB = 5 →
  BC = 7 →
  Real.sin B = 3 * Real.sqrt 3 / 14 :=
by sorry

end NUMINAMATH_CALUDE_sin_B_in_triangle_l2017_201746


namespace NUMINAMATH_CALUDE_total_money_l2017_201797

theorem total_money (A B C : ℕ) : 
  A + C = 200 →
  B + C = 340 →
  C = 40 →
  A + B + C = 500 := by
sorry

end NUMINAMATH_CALUDE_total_money_l2017_201797


namespace NUMINAMATH_CALUDE_cube_lateral_surface_area_l2017_201754

theorem cube_lateral_surface_area (volume : ℝ) (lateral_surface_area : ℝ) :
  volume = 125 →
  lateral_surface_area = 4 * (volume ^ (1/3))^2 →
  lateral_surface_area = 100 := by
  sorry

end NUMINAMATH_CALUDE_cube_lateral_surface_area_l2017_201754


namespace NUMINAMATH_CALUDE_discount_calculation_l2017_201733

/-- Given an article with a cost price of 100 units, if the selling price is marked 12% above 
    the cost price and the trader suffers a loss of 1% at the time of selling, 
    then the discount allowed is 13 units. -/
theorem discount_calculation (cost_price : ℝ) (marked_price : ℝ) (selling_price : ℝ) : 
  cost_price = 100 →
  marked_price = cost_price * 1.12 →
  selling_price = cost_price * 0.99 →
  marked_price - selling_price = 13 :=
by sorry

end NUMINAMATH_CALUDE_discount_calculation_l2017_201733


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l2017_201701

/-- A right triangle with sides 6, 8, and 10 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  a_eq : a = 6
  b_eq : b = 8
  c_eq : c = 10

/-- Square inscribed in the triangle with one vertex at the right angle -/
def inscribed_square_right_angle (t : RightTriangle) (x : ℝ) : Prop :=
  x > 0 ∧ x ≤ t.a ∧ x ≤ t.b ∧ x / t.a = x / t.b

/-- Square inscribed in the triangle with one side along the hypotenuse -/
def inscribed_square_hypotenuse (t : RightTriangle) (y : ℝ) : Prop :=
  y > 0 ∧ y ≤ t.c ∧ y / t.c = (6/5 * y + 8/5 * y) / (t.a + t.b)

theorem inscribed_squares_ratio (t : RightTriangle) (x y : ℝ) 
  (hx : inscribed_square_right_angle t x) (hy : inscribed_square_hypotenuse t y) : 
  x / y = 111 / 175 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l2017_201701


namespace NUMINAMATH_CALUDE_janice_working_days_l2017_201745

-- Define the problem parameters
def regular_pay : ℕ := 30
def overtime_pay : ℕ := 15
def overtime_shifts : ℕ := 3
def total_earnings : ℕ := 195

-- Define the function to calculate the number of working days
def calculate_working_days (regular_pay overtime_pay overtime_shifts total_earnings : ℕ) : ℕ :=
  (total_earnings - overtime_pay * overtime_shifts) / regular_pay

-- Theorem statement
theorem janice_working_days :
  calculate_working_days regular_pay overtime_pay overtime_shifts total_earnings = 5 := by
  sorry


end NUMINAMATH_CALUDE_janice_working_days_l2017_201745


namespace NUMINAMATH_CALUDE_tripling_radius_and_negative_quantity_l2017_201723

theorem tripling_radius_and_negative_quantity : ∀ (r : ℝ) (x : ℝ), 
  r > 0 → x < 0 → 
  (π * (3 * r)^2 ≠ 3 * (π * r^2)) ∧ (3 * x ≤ x) := by sorry

end NUMINAMATH_CALUDE_tripling_radius_and_negative_quantity_l2017_201723


namespace NUMINAMATH_CALUDE_count_cow_herds_l2017_201709

/-- Given a farm with cows organized into herds, this theorem proves
    the number of herds given the total number of cows and the number
    of cows per herd. -/
theorem count_cow_herds (total_cows : ℕ) (cows_per_herd : ℕ) 
    (h1 : total_cows = 320) (h2 : cows_per_herd = 40) :
    total_cows / cows_per_herd = 8 := by
  sorry

end NUMINAMATH_CALUDE_count_cow_herds_l2017_201709


namespace NUMINAMATH_CALUDE_dana_soda_consumption_l2017_201762

/-- The number of milliliters in one liter -/
def ml_per_liter : ℕ := 1000

/-- The size of the soda bottle in liters -/
def bottle_size : ℕ := 2

/-- The number of days the bottle lasts -/
def days_lasted : ℕ := 4

/-- Dana's daily soda consumption in milliliters -/
def daily_consumption : ℕ := (bottle_size * ml_per_liter) / days_lasted

theorem dana_soda_consumption :
  daily_consumption = 500 :=
sorry

end NUMINAMATH_CALUDE_dana_soda_consumption_l2017_201762


namespace NUMINAMATH_CALUDE_magical_stack_size_157_l2017_201722

/-- A stack of cards is magical if it satisfies certain conditions --/
structure MagicalStack :=
  (n : ℕ)
  (total_cards : ℕ := 2 * n)
  (card_157_position : ℕ)
  (card_157_retains_position : card_157_position = 157)

/-- The number of cards in a magical stack where card 157 retains its position --/
def magical_stack_size (stack : MagicalStack) : ℕ := stack.total_cards

/-- Theorem: The size of a magical stack where card 157 retains its position is 470 --/
theorem magical_stack_size_157 (stack : MagicalStack) : 
  magical_stack_size stack = 470 := by sorry

end NUMINAMATH_CALUDE_magical_stack_size_157_l2017_201722


namespace NUMINAMATH_CALUDE_largest_four_digit_square_base7_l2017_201760

/-- The largest integer whose square has exactly 4 digits in base 7 -/
def M : ℕ := 48

/-- Conversion of a natural number to its base 7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

theorem largest_four_digit_square_base7 :
  (M * M ≥ 7^3) ∧ 
  (M * M < 7^4) ∧ 
  (∀ n : ℕ, n > M → n * n ≥ 7^4) ∧
  (toBase7 M = [6, 6]) := by sorry

end NUMINAMATH_CALUDE_largest_four_digit_square_base7_l2017_201760


namespace NUMINAMATH_CALUDE_a_to_b_equals_negative_one_l2017_201763

theorem a_to_b_equals_negative_one (a b : ℝ) (h : |a + 1| = -(b - 3)^2) : a^b = -1 := by
  sorry

end NUMINAMATH_CALUDE_a_to_b_equals_negative_one_l2017_201763


namespace NUMINAMATH_CALUDE_twenty_first_term_is_4641_l2017_201730

/-- The sequence where each term is the sum of consecutive integers, 
    and the number of integers in each group increases by 1 -/
def sequence_term (n : ℕ) : ℕ :=
  let first_num := 1 + (n * (n - 1)) / 2
  let last_num := first_num + n - 1
  n * (first_num + last_num) / 2

/-- The 21st term of the sequence is 4641 -/
theorem twenty_first_term_is_4641 : sequence_term 21 = 4641 := by
  sorry

end NUMINAMATH_CALUDE_twenty_first_term_is_4641_l2017_201730


namespace NUMINAMATH_CALUDE_program_output_l2017_201725

def program (a b : ℕ) : ℕ × ℕ :=
  let a' := a + b
  let b' := b * a'
  (a', b')

theorem program_output : program 1 3 = (4, 12) := by
  sorry

end NUMINAMATH_CALUDE_program_output_l2017_201725


namespace NUMINAMATH_CALUDE_quadratic_opens_downward_l2017_201771

def f (x : ℝ) := -x^2 + 3

theorem quadratic_opens_downward :
  ∃ (a : ℝ), ∀ (x : ℝ), x > a → f x < f a :=
sorry

end NUMINAMATH_CALUDE_quadratic_opens_downward_l2017_201771


namespace NUMINAMATH_CALUDE_unique_number_l2017_201744

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem unique_number : ∃! n : ℕ, 
  is_two_digit n ∧ 
  n % 2 = 1 ∧ 
  n % 13 = 0 ∧ 
  is_perfect_square (digit_product n) ∧
  n = 91 := by
sorry

end NUMINAMATH_CALUDE_unique_number_l2017_201744


namespace NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l2017_201739

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem diagonals_30_sided_polygon : num_diagonals 30 = 405 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l2017_201739


namespace NUMINAMATH_CALUDE_calculation_proof_l2017_201753

theorem calculation_proof : (40 * 1505 - 20 * 1505) / 5 = 6020 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2017_201753


namespace NUMINAMATH_CALUDE_range_of_x_l2017_201752

theorem range_of_x (x : ℝ) : 
  (¬ (x ∈ Set.Icc 2 5 ∨ x < 1 ∨ x > 4)) → (1 ≤ x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_l2017_201752


namespace NUMINAMATH_CALUDE_tom_car_washing_earnings_l2017_201780

/-- 
Given:
- Tom had $74 last week
- Tom has $160 now
Prove that Tom made $86 by washing cars over the weekend.
-/
theorem tom_car_washing_earnings :
  let initial_money : ℕ := 74
  let current_money : ℕ := 160
  let money_earned : ℕ := current_money - initial_money
  money_earned = 86 := by sorry

end NUMINAMATH_CALUDE_tom_car_washing_earnings_l2017_201780


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l2017_201761

-- Define a line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to check if a line passes through a point
def passes_through (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

-- Define a function to check if a line has equal intercepts
def has_equal_intercepts (l : Line) : Prop :=
  l.a = l.b ∨ (l.a = -l.b ∧ l.c = 0)

-- State the theorem
theorem line_through_point_with_equal_intercepts :
  ∀ l : Line,
    passes_through l 3 (-6) →
    has_equal_intercepts l →
    (l = Line.mk 1 1 3 ∨ l = Line.mk 2 1 0) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l2017_201761


namespace NUMINAMATH_CALUDE_min_value_on_circle_l2017_201749

theorem min_value_on_circle :
  ∃ (min : ℝ), min = -5 ∧
  (∀ x y : ℝ, x^2 + y^2 = 1 → 3*x + 4*y ≥ min) ∧
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ 3*x + 4*y = min) := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l2017_201749


namespace NUMINAMATH_CALUDE_range_of_b_l2017_201769

theorem range_of_b (a b c : ℝ) (h1 : a * c = b^2) (h2 : a + b + c = 3) :
  -3 ≤ b ∧ b ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_b_l2017_201769


namespace NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l2017_201774

/-- The length of a bridge given train specifications --/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof of the bridge length problem --/
theorem bridge_length_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |bridge_length 180 60 45 - 570.15| < ε :=
sorry

end NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l2017_201774


namespace NUMINAMATH_CALUDE_wire_length_problem_l2017_201756

theorem wire_length_problem (total_wires : ℕ) (total_avg_length : ℝ) 
  (quarter_avg_length : ℝ) (third_avg_length : ℝ) :
  total_wires = 12 →
  total_avg_length = 95 →
  quarter_avg_length = 120 →
  third_avg_length = 75 →
  let quarter_wires := total_wires / 4
  let third_wires := total_wires / 3
  let remaining_wires := total_wires - quarter_wires - third_wires
  let total_length := total_wires * total_avg_length
  let quarter_length := quarter_wires * quarter_avg_length
  let third_length := third_wires * third_avg_length
  let remaining_length := total_length - quarter_length - third_length
  remaining_length / remaining_wires = 96 := by
sorry

end NUMINAMATH_CALUDE_wire_length_problem_l2017_201756


namespace NUMINAMATH_CALUDE_rainfall_difference_l2017_201768

/-- The difference in rainfall between March and April -/
theorem rainfall_difference (march_rainfall april_rainfall : ℝ) 
  (h1 : march_rainfall = 0.81)
  (h2 : april_rainfall = 0.46) : 
  march_rainfall - april_rainfall = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_difference_l2017_201768


namespace NUMINAMATH_CALUDE_soccer_league_female_fraction_l2017_201734

/-- Represents the number of participants in a soccer league for two consecutive years -/
structure LeagueParticipation where
  malesLastYear : ℕ
  femalesLastYear : ℕ
  malesThisYear : ℕ
  femalesThisYear : ℕ

/-- Calculates the fraction of female participants this year -/
def femaleFraction (lp : LeagueParticipation) : Rat :=
  lp.femalesThisYear / (lp.malesThisYear + lp.femalesThisYear)

theorem soccer_league_female_fraction 
  (lp : LeagueParticipation)
  (male_increase : lp.malesThisYear = (110 * lp.malesLastYear) / 100)
  (female_increase : lp.femalesThisYear = (125 * lp.femalesLastYear) / 100)
  (total_increase : lp.malesThisYear + lp.femalesThisYear = 
    (115 * (lp.malesLastYear + lp.femalesLastYear)) / 100)
  (males_last_year : lp.malesLastYear = 30)
  : femaleFraction lp = 19 / 52 := by
  sorry

#check soccer_league_female_fraction

end NUMINAMATH_CALUDE_soccer_league_female_fraction_l2017_201734


namespace NUMINAMATH_CALUDE_passing_marks_l2017_201704

theorem passing_marks (T : ℝ) (P : ℝ) 
  (h1 : 0.20 * T = P - 40)
  (h2 : 0.30 * T = P + 20) : 
  P = 160 := by
sorry

end NUMINAMATH_CALUDE_passing_marks_l2017_201704


namespace NUMINAMATH_CALUDE_phi_value_l2017_201700

noncomputable def f (x φ : ℝ) : ℝ := Real.cos (2 * x + φ)

theorem phi_value (φ : ℝ) :
  -π ≤ φ ∧ φ < π →
  (∀ x, f (x - π/2) φ = Real.sin x * Real.cos x + (Real.sqrt 3 / 2) * Real.cos x) →
  |φ| = 5*π/6 := by
  sorry

end NUMINAMATH_CALUDE_phi_value_l2017_201700


namespace NUMINAMATH_CALUDE_zoe_mp3_songs_l2017_201747

theorem zoe_mp3_songs (initial_songs : ℕ) (deleted_songs : ℕ) (added_songs : ℕ) :
  initial_songs = 6 →
  deleted_songs = 3 →
  added_songs = 20 →
  initial_songs - deleted_songs + added_songs = 23 :=
by sorry

end NUMINAMATH_CALUDE_zoe_mp3_songs_l2017_201747


namespace NUMINAMATH_CALUDE_student_arrangement_count_l2017_201787

/-- The number of ways to arrange 6 students in a line with 3 friends not adjacent -/
def arrangement_count : ℕ := 576

/-- Total number of students -/
def total_students : ℕ := 6

/-- Number of friends who refuse to stand next to each other -/
def friend_count : ℕ := 3

/-- Number of non-friend students -/
def non_friend_count : ℕ := total_students - friend_count

theorem student_arrangement_count :
  arrangement_count =
    (Nat.factorial total_students) -
    ((Nat.factorial non_friend_count) *
     (Nat.choose (non_friend_count + 1) friend_count) *
     (Nat.factorial friend_count)) :=
by sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l2017_201787


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l2017_201783

/-- A hyperbola passing through the point (4, √3) with asymptote equation y = 1/2x has the standard equation x²/4 - y² = 1 -/
theorem hyperbola_standard_equation (x y : ℝ) :
  (∃ k : ℝ, x^2 / 4 - y^2 = k) →  -- Assuming the general form of the hyperbola equation
  (∀ x, y = 1/2 * x) →           -- Asymptote equation
  (4^2 / 4 - (Real.sqrt 3)^2 = 1) →  -- The hyperbola passes through (4, √3)
  x^2 / 4 - y^2 = 1 :=            -- Standard equation of the hyperbola
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l2017_201783


namespace NUMINAMATH_CALUDE_tom_marbles_groups_l2017_201779

/-- Represents the colors of marbles --/
inductive MarbleColor
  | Red
  | Green
  | Blue
  | Yellow

/-- Represents Tom's collection of marbles --/
structure MarbleCollection where
  red : Nat
  green : Nat
  blue : Nat
  yellow : Nat

/-- Calculates the number of different groups of two marbles that can be chosen --/
def countDifferentGroups (collection : MarbleCollection) : Nat :=
  sorry

/-- Theorem stating that Tom's specific collection results in 12 different groups --/
theorem tom_marbles_groups :
  let toms_collection : MarbleCollection := {
    red := 1,
    green := 1,
    blue := 2,
    yellow := 3
  }
  countDifferentGroups toms_collection = 12 := by
  sorry

end NUMINAMATH_CALUDE_tom_marbles_groups_l2017_201779


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l2017_201764

def num_red_balls : ℕ := 2
def num_white_balls : ℕ := 6

def total_balls : ℕ := num_red_balls + num_white_balls

theorem probability_of_red_ball :
  (num_red_balls : ℚ) / (total_balls : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l2017_201764


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2017_201706

theorem chess_tournament_games (n : ℕ) (h : n = 25) : 
  n * (n - 1) = 600 → 2 * (n * (n - 1)) = 1200 := by
  sorry

#check chess_tournament_games

end NUMINAMATH_CALUDE_chess_tournament_games_l2017_201706


namespace NUMINAMATH_CALUDE_factorization_equality_l2017_201726

theorem factorization_equality (a b : ℝ) : 2 * a^3 - 8 * a * b^2 = 2 * a * (a + 2*b) * (a - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2017_201726


namespace NUMINAMATH_CALUDE_division_remainder_l2017_201755

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = divisor * quotient + remainder →
  dividend = 199 →
  divisor = 18 →
  quotient = 11 →
  remainder = 1 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l2017_201755


namespace NUMINAMATH_CALUDE_janettes_remaining_jerky_l2017_201770

def camping_days : ℕ := 5
def initial_jerky : ℕ := 40
def breakfast_jerky : ℕ := 1
def lunch_jerky : ℕ := 1
def dinner_jerky : ℕ := 2

def daily_consumption : ℕ := breakfast_jerky + lunch_jerky + dinner_jerky

def total_consumed : ℕ := daily_consumption * camping_days

def remaining_after_trip : ℕ := initial_jerky - total_consumed

def given_to_brother : ℕ := remaining_after_trip / 2

theorem janettes_remaining_jerky :
  initial_jerky - total_consumed - given_to_brother = 10 := by
  sorry

end NUMINAMATH_CALUDE_janettes_remaining_jerky_l2017_201770


namespace NUMINAMATH_CALUDE_remainder_sum_mod_59_l2017_201794

theorem remainder_sum_mod_59 (a b c : ℕ+) 
  (ha : a ≡ 28 [ZMOD 59])
  (hb : b ≡ 34 [ZMOD 59])
  (hc : c ≡ 5 [ZMOD 59]) :
  (a + b + c) ≡ 8 [ZMOD 59] := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_59_l2017_201794


namespace NUMINAMATH_CALUDE_divisibility_by_five_l2017_201793

theorem divisibility_by_five (a b : ℕ) : 
  (5 ∣ (a * b)) → (5 ∣ a) ∨ (5 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l2017_201793


namespace NUMINAMATH_CALUDE_sum_of_numbers_l2017_201738

-- Define the range of numbers
def valid_number (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 50

-- Define primality
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define Alice's number
def alice_number (a : ℕ) : Prop := valid_number a

-- Define Bob's number
def bob_number (b : ℕ) : Prop := valid_number b ∧ is_prime b

-- Alice can't determine who has the larger number
def alice_uncertainty (a b : ℕ) : Prop :=
  alice_number a → bob_number b → ¬(a > b ∨ b > a)

-- Bob can determine who has the larger number after Alice's statement
def bob_certainty (a b : ℕ) : Prop :=
  alice_number a → bob_number b → (a > b ∨ b > a)

-- 200 * Bob's number + Alice's number is a perfect square
def perfect_square_condition (a b : ℕ) : Prop :=
  alice_number a → bob_number b → ∃ k : ℕ, 200 * b + a = k * k

-- Theorem statement
theorem sum_of_numbers (a b : ℕ) :
  alice_number a →
  bob_number b →
  alice_uncertainty a b →
  bob_certainty a b →
  perfect_square_condition a b →
  a + b = 43 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l2017_201738


namespace NUMINAMATH_CALUDE_building_C_floors_l2017_201799

/-- The number of floors in Building A -/
def floors_A : ℕ := 4

/-- The number of floors in Building B -/
def floors_B : ℕ := floors_A + 9

/-- The number of floors in Building C -/
def floors_C : ℕ := 5 * floors_B - 6

theorem building_C_floors : floors_C = 59 := by
  sorry

end NUMINAMATH_CALUDE_building_C_floors_l2017_201799


namespace NUMINAMATH_CALUDE_metal_bars_per_set_l2017_201728

theorem metal_bars_per_set (total_bars : ℕ) (num_sets : ℕ) (bars_per_set : ℕ) : 
  total_bars = 14 → num_sets = 2 → total_bars = num_sets * bars_per_set → bars_per_set = 7 := by
  sorry

end NUMINAMATH_CALUDE_metal_bars_per_set_l2017_201728


namespace NUMINAMATH_CALUDE_two_sides_and_angle_unique_two_angles_and_side_unique_three_sides_unique_two_sides_and_included_angle_not_unique_l2017_201798

/-- Represents a triangle -/
structure Triangle :=
  (a b c : ℝ)  -- sides
  (α β γ : ℝ)  -- angles

/-- Determines if a triangle is uniquely defined -/
def is_unique_triangle (t : Triangle) : Prop := sorry

/-- Two sides and an angle uniquely determine a triangle -/
theorem two_sides_and_angle_unique (a b : ℝ) (α : ℝ) : 
  ∃! t : Triangle, t.a = a ∧ t.b = b ∧ t.α = α := sorry

/-- Two angles and a side uniquely determine a triangle -/
theorem two_angles_and_side_unique (α β : ℝ) (a : ℝ) : 
  ∃! t : Triangle, t.α = α ∧ t.β = β ∧ t.a = a := sorry

/-- Three sides uniquely determine a triangle -/
theorem three_sides_unique (a b c : ℝ) : 
  ∃! t : Triangle, t.a = a ∧ t.b = b ∧ t.c = c := sorry

/-- Two sides and their included angle do not uniquely determine a triangle -/
theorem two_sides_and_included_angle_not_unique (a b : ℝ) (γ : ℝ) : 
  ¬(∃! t : Triangle, t.a = a ∧ t.b = b ∧ t.γ = γ) := sorry

end NUMINAMATH_CALUDE_two_sides_and_angle_unique_two_angles_and_side_unique_three_sides_unique_two_sides_and_included_angle_not_unique_l2017_201798


namespace NUMINAMATH_CALUDE_sum_in_base6_l2017_201767

/-- Represents a number in base 6 -/
def Base6 : Type := List Nat

/-- Converts a base 6 number to its decimal representation -/
def to_decimal (n : Base6) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a decimal number to its base 6 representation -/
def to_base6 (n : Nat) : Base6 :=
  sorry

/-- Adds two base 6 numbers -/
def add_base6 (a b : Base6) : Base6 :=
  to_base6 (to_decimal a + to_decimal b)

theorem sum_in_base6 (a b c : Base6) :
  a = [0, 5, 6] ∧ b = [5, 0, 1] ∧ c = [2] →
  add_base6 (add_base6 a b) c = [1, 1, 1, 1] :=
sorry

end NUMINAMATH_CALUDE_sum_in_base6_l2017_201767


namespace NUMINAMATH_CALUDE_first_note_denomination_l2017_201727

/-- Proves that given the conditions of the problem, the denomination of the first type of notes must be 1 rupee -/
theorem first_note_denomination (total_amount : ℕ) (total_notes : ℕ) (x : ℕ) : 
  total_amount = 400 →
  total_notes = 75 →
  total_amount = (total_notes / 3 * x) + (total_notes / 3 * 5) + (total_notes / 3 * 10) →
  x = 1 := by
  sorry

#check first_note_denomination

end NUMINAMATH_CALUDE_first_note_denomination_l2017_201727


namespace NUMINAMATH_CALUDE_abs_minus_one_eq_zero_l2017_201736

theorem abs_minus_one_eq_zero (a : ℝ) : |a| - 1 = 0 → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_abs_minus_one_eq_zero_l2017_201736


namespace NUMINAMATH_CALUDE_billy_hike_distance_l2017_201716

theorem billy_hike_distance (east north : ℝ) (h1 : east = 7) (h2 : north = 8 * (Real.sqrt 2) / 2) :
  Real.sqrt (east^2 + north^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_billy_hike_distance_l2017_201716


namespace NUMINAMATH_CALUDE_planes_perpendicular_from_line_perpendicular_planes_from_parallel_l2017_201772

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_perpendicular : Line → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)

-- Theorem 1
theorem planes_perpendicular_from_line 
  (α β : Plane) (l : Line) :
  line_perpendicular l α → line_parallel l β → perpendicular α β := by sorry

-- Theorem 2
theorem perpendicular_planes_from_parallel 
  (α β γ : Plane) :
  parallel α β → perpendicular α γ → perpendicular β γ := by sorry

end NUMINAMATH_CALUDE_planes_perpendicular_from_line_perpendicular_planes_from_parallel_l2017_201772


namespace NUMINAMATH_CALUDE_local_maximum_at_two_l2017_201710

/-- The function f(x) defined as x(x-c)^2 --/
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

/-- The derivative of f(x) with respect to x --/
def f_derivative (c : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*c*x + c^2

/-- Theorem stating that the value of c for which f(x) has a local maximum at x=2 is 6 --/
theorem local_maximum_at_two (c : ℝ) : 
  (∀ x, x ≠ 2 → ∃ δ > 0, ∀ y, |y - 2| < δ → f c y ≤ f c 2) → c = 6 :=
sorry

end NUMINAMATH_CALUDE_local_maximum_at_two_l2017_201710


namespace NUMINAMATH_CALUDE_number_of_planes_l2017_201715

/-- Given an air exhibition with commercial planes, prove the number of planes
    when the total number of wings and wings per plane are known. -/
theorem number_of_planes (total_wings : ℕ) (wings_per_plane : ℕ) (h1 : total_wings = 90) (h2 : wings_per_plane = 2) :
  total_wings / wings_per_plane = 45 := by
  sorry

#check number_of_planes

end NUMINAMATH_CALUDE_number_of_planes_l2017_201715


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2017_201784

/-- Given a hyperbola with equation x² - y²/b² = 1 where b > 0,
    if one of its asymptotes has the equation y = 2x, then b = 2 -/
theorem hyperbola_asymptote (b : ℝ) (hb : b > 0) :
  (∀ x y : ℝ, x^2 - y^2 / b^2 = 1) →
  (∃ x y : ℝ, y = 2*x ∧ x^2 - y^2 / b^2 = 1) →
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2017_201784


namespace NUMINAMATH_CALUDE_baseball_cards_count_l2017_201785

theorem baseball_cards_count (num_friends : ℕ) (cards_per_friend : ℕ) : 
  num_friends = 5 → cards_per_friend = 91 → num_friends * cards_per_friend = 455 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_count_l2017_201785


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2017_201714

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 4*x₁ - 2 = 0) → 
  (x₂^2 - 4*x₂ - 2 = 0) → 
  (x₁ + x₂ = 4) := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2017_201714


namespace NUMINAMATH_CALUDE_triangle_problem_l2017_201765

open Real

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  b = 2 * a * sin A / sin B →
  c = 2 * a * sin C / sin B →
  b / 2 = (2 * a * sin A * cos C + c * sin (2 * A)) / 2 →
  (A = π/6 ∧
   (a = 2 →
    ∀ (b' c' : ℝ),
      b' > 0 ∧ c' > 0 →
      b' = 2 * sin A / sin B →
      c' = 2 * sin C / sin B →
      1/2 * b' * c' * sin A ≤ 2 + sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2017_201765


namespace NUMINAMATH_CALUDE_lumberjack_firewood_l2017_201703

/-- Calculates the total number of firewood pieces produced by a lumberjack --/
theorem lumberjack_firewood (trees : ℕ) (logs_per_tree : ℕ) (pieces_per_log : ℕ) 
  (h1 : logs_per_tree = 4)
  (h2 : pieces_per_log = 5)
  (h3 : trees = 25) :
  trees * logs_per_tree * pieces_per_log = 500 := by
  sorry

#check lumberjack_firewood

end NUMINAMATH_CALUDE_lumberjack_firewood_l2017_201703


namespace NUMINAMATH_CALUDE_bianca_recycling_points_l2017_201778

/-- Calculates the points earned by Bianca for recycling bottles and cans --/
def points_earned (aluminum_points plastic_points glass_points : ℕ)
                  (aluminum_bags plastic_bags glass_bags : ℕ)
                  (aluminum_not_recycled plastic_not_recycled glass_not_recycled : ℕ) : ℕ :=
  (aluminum_points * (aluminum_bags - aluminum_not_recycled)) +
  (plastic_points * (plastic_bags - plastic_not_recycled)) +
  (glass_points * (glass_bags - glass_not_recycled))

theorem bianca_recycling_points :
  points_earned 5 8 10 10 5 5 3 2 1 = 99 := by
  sorry

end NUMINAMATH_CALUDE_bianca_recycling_points_l2017_201778


namespace NUMINAMATH_CALUDE_surface_area_difference_specific_l2017_201789

/-- Calculates the surface area difference when removing a cube from a rectangular solid -/
def surface_area_difference (l w h : ℝ) (cube_side : ℝ) : ℝ :=
  let original_surface_area := 2 * (l * w + l * h + w * h)
  let new_faces_area := 2 * cube_side * cube_side
  let removed_faces_area := 5 * cube_side * cube_side
  new_faces_area - removed_faces_area

/-- The surface area difference for the specific problem -/
theorem surface_area_difference_specific :
  surface_area_difference 6 5 4 2 = -12 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_difference_specific_l2017_201789


namespace NUMINAMATH_CALUDE_initial_hamburgers_count_l2017_201724

/-- Proves that the number of hamburgers made initially equals 9 -/
theorem initial_hamburgers_count (initial : ℕ) (additional : ℕ) (total : ℕ)
  (h1 : additional = 3)
  (h2 : total = 12)
  (h3 : initial + additional = total) :
  initial = 9 := by
  sorry

end NUMINAMATH_CALUDE_initial_hamburgers_count_l2017_201724


namespace NUMINAMATH_CALUDE_polynomial_solution_l2017_201711

theorem polynomial_solution (a : ℝ) (ha : a ≠ -1) 
  (h : a^5 + 5*a^4 + 10*a^3 + 3*a^2 - 9*a - 6 = 0) : 
  (a + 1)^3 = 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_solution_l2017_201711


namespace NUMINAMATH_CALUDE_bookshelf_capacity_l2017_201751

theorem bookshelf_capacity (num_bookshelves : ℕ) (layers_per_bookshelf : ℕ) (books_per_layer : ℕ) 
  (h1 : num_bookshelves = 8) 
  (h2 : layers_per_bookshelf = 5) 
  (h3 : books_per_layer = 85) : 
  num_bookshelves * layers_per_bookshelf * books_per_layer = 3400 := by
  sorry

end NUMINAMATH_CALUDE_bookshelf_capacity_l2017_201751


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2017_201790

-- Problem 1
theorem problem_1 : (1) - 1^2 + Real.sqrt 12 + Real.sqrt (4/3) = -1 + (8 * Real.sqrt 3) / 3 := by sorry

-- Problem 2
theorem problem_2 : ∀ x : ℝ, 2*x^2 - x - 1 = 0 ↔ x = -1/2 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2017_201790


namespace NUMINAMATH_CALUDE_fran_ate_15_green_macaroons_l2017_201777

/-- The number of green macaroons Fran ate -/
def green_eaten : ℕ := sorry

/-- The number of red macaroons Fran baked -/
def red_baked : ℕ := 50

/-- The number of green macaroons Fran baked -/
def green_baked : ℕ := 40

/-- The number of macaroons remaining -/
def remaining : ℕ := 45

theorem fran_ate_15_green_macaroons :
  green_eaten = 15 ∧
  red_baked = 50 ∧
  green_baked = 40 ∧
  remaining = 45 ∧
  red_baked + green_baked = green_eaten + 2 * green_eaten + remaining :=
sorry

end NUMINAMATH_CALUDE_fran_ate_15_green_macaroons_l2017_201777


namespace NUMINAMATH_CALUDE_comparison_of_expressions_l2017_201717

theorem comparison_of_expressions (x : ℝ) (h : x ≠ 1) :
  (x > 1 → 1 + x > 1 / (1 - x)) ∧ (x < 1 → 1 + x < 1 / (1 - x)) := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_expressions_l2017_201717


namespace NUMINAMATH_CALUDE_pet_shop_pricing_l2017_201782

theorem pet_shop_pricing (puppy_cost kitten_cost parakeet_cost : ℚ) : 
  puppy_cost = 3 * parakeet_cost →
  parakeet_cost = kitten_cost / 2 →
  2 * puppy_cost + 2 * kitten_cost + 3 * parakeet_cost = 130 →
  parakeet_cost = 10 := by
sorry

end NUMINAMATH_CALUDE_pet_shop_pricing_l2017_201782


namespace NUMINAMATH_CALUDE_floor_of_sum_l2017_201742

theorem floor_of_sum (x : ℝ) (h : x = -3.7 + 1.5) : ⌊x⌋ = -3 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_sum_l2017_201742


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l2017_201741

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ j : ℕ, j ≤ 10 ∧ j > 0 ∧ m % j ≠ 0) ∧
  n = 2520 :=
by sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l2017_201741


namespace NUMINAMATH_CALUDE_earth_angle_calculation_l2017_201796

/-- The angle between two points on a spherical Earth given their coordinates --/
def spherical_angle (lat1 : Real) (lon1 : Real) (lat2 : Real) (lon2 : Real) : Real :=
  sorry

theorem earth_angle_calculation :
  let p_lat : Real := 0
  let p_lon : Real := 100
  let q_lat : Real := 30
  let q_lon : Real := -100 -- Negative for West longitude
  spherical_angle p_lat p_lon q_lat q_lon = 160 := by
  sorry

end NUMINAMATH_CALUDE_earth_angle_calculation_l2017_201796


namespace NUMINAMATH_CALUDE_min_bills_for_payment_l2017_201718

/-- Represents the available denominations of bills and coins --/
structure Denominations :=
  (ten_dollar : ℕ)
  (five_dollar : ℕ)
  (two_dollar : ℕ)
  (one_dollar : ℕ)
  (fifty_cent : ℕ)

/-- Calculates the minimum number of bills and coins needed to pay a given amount --/
def min_bills_and_coins (d : Denominations) (amount : ℚ) : ℕ :=
  sorry

/-- Tim's available bills and coins --/
def tims_denominations : Denominations :=
  { ten_dollar := 15
  , five_dollar := 7
  , two_dollar := 12
  , one_dollar := 20
  , fifty_cent := 10 }

/-- The theorem stating that Tim needs 17 bills and coins to pay $152.50 --/
theorem min_bills_for_payment :
  min_bills_and_coins tims_denominations (152.5 : ℚ) = 17 :=
sorry

end NUMINAMATH_CALUDE_min_bills_for_payment_l2017_201718


namespace NUMINAMATH_CALUDE_largest_valid_n_l2017_201759

/-- Represents the color of a ball -/
inductive Color
| Black
| White

/-- Represents a coloring function for balls -/
def Coloring := ℕ → Color

/-- Checks if a coloring satisfies the given condition -/
def ValidColoring (c : Coloring) (n : ℕ) : Prop :=
  ∀ a₁ a₂ a₃ a₄ : ℕ,
    a₁ ≤ n ∧ a₂ ≤ n ∧ a₃ ≤ n ∧ a₄ ≤ n →
    a₁ + a₂ + a₃ = a₄ →
    (c a₁ = Color.Black ∨ c a₂ = Color.Black ∨ c a₃ = Color.Black) ∧
    (c a₁ = Color.White ∨ c a₂ = Color.White ∨ c a₃ = Color.White)

/-- The theorem stating that 10 is the largest possible value of n -/
theorem largest_valid_n :
  (∃ c : Coloring, ValidColoring c 10) ∧
  (∀ n > 10, ¬∃ c : Coloring, ValidColoring c n) :=
sorry

end NUMINAMATH_CALUDE_largest_valid_n_l2017_201759


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2017_201795

theorem quadratic_equation_solution : ∃ x : ℝ, (10 - x)^2 = x^2 + 6 ∧ x = 4.7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2017_201795


namespace NUMINAMATH_CALUDE_largest_percent_error_rectangular_plot_l2017_201776

theorem largest_percent_error_rectangular_plot (length width : ℝ) 
  (h_length : length = 15)
  (h_width : width = 10)
  (h_error : ℝ) (h_error_bound : h_error = 0.1) : 
  let actual_area := length * width
  let max_length := length * (1 + h_error)
  let max_width := width * (1 + h_error)
  let max_area := max_length * max_width
  let max_percent_error := (max_area - actual_area) / actual_area * 100
  max_percent_error = 21 := by sorry

end NUMINAMATH_CALUDE_largest_percent_error_rectangular_plot_l2017_201776


namespace NUMINAMATH_CALUDE_largest_sum_of_digits_24hour_pm_l2017_201713

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours ≥ 0 ∧ hours ≤ 23
  minute_valid : minutes ≥ 0 ∧ minutes ≤ 59

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits in a Time24 -/
def sumOfDigitsTime24 (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- Checks if a Time24 is between 12:00 and 23:59 -/
def isBetween12And2359 (t : Time24) : Prop :=
  t.hours ≥ 12 ∧ t.hours ≤ 23

theorem largest_sum_of_digits_24hour_pm :
  ∃ (t : Time24), isBetween12And2359 t ∧
    ∀ (t' : Time24), isBetween12And2359 t' →
      sumOfDigitsTime24 t' ≤ sumOfDigitsTime24 t ∧
      sumOfDigitsTime24 t = 24 :=
sorry

end NUMINAMATH_CALUDE_largest_sum_of_digits_24hour_pm_l2017_201713


namespace NUMINAMATH_CALUDE_water_jars_count_l2017_201707

theorem water_jars_count (total_water : ℚ) (quart_jars half_gal_jars one_gal_jars two_gal_jars : ℕ) : 
  total_water = 56 →
  quart_jars = 16 →
  half_gal_jars = 12 →
  one_gal_jars = 8 →
  two_gal_jars = 4 →
  ∃ (three_gal_jars : ℕ), 
    (quart_jars : ℚ) * (1/4) + 
    (half_gal_jars : ℚ) * (1/2) + 
    (one_gal_jars : ℚ) + 
    (two_gal_jars : ℚ) * 2 + 
    (three_gal_jars : ℚ) * 3 = total_water ∧
    quart_jars + half_gal_jars + one_gal_jars + two_gal_jars + three_gal_jars = 50 :=
by sorry

end NUMINAMATH_CALUDE_water_jars_count_l2017_201707


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2017_201743

/-- The speed of a boat in still water, given its travel times with varying current and wind conditions. -/
theorem boat_speed_in_still_water 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (current_start : ℝ) 
  (current_end : ℝ) 
  (wind_slowdown : ℝ) 
  (h1 : downstream_time = 3)
  (h2 : upstream_time = 4.5)
  (h3 : current_start = 2)
  (h4 : current_end = 4)
  (h5 : wind_slowdown = 1) :
  ∃ (boat_speed : ℝ), boat_speed = 16 ∧ 
  (boat_speed + (current_start + current_end) / 2 - wind_slowdown) * downstream_time = 
  (boat_speed - (current_start + current_end) / 2 - wind_slowdown) * upstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2017_201743


namespace NUMINAMATH_CALUDE_twenty_new_homes_l2017_201781

/-- Calculates the number of new trailer homes added -/
def new_trailer_homes (initial_count : ℕ) (initial_avg_age : ℕ) (time_passed : ℕ) (current_avg_age : ℕ) : ℕ :=
  let total_age := initial_count * (initial_avg_age + time_passed)
  let k := (total_age - initial_count * current_avg_age) / (current_avg_age - time_passed)
  k

/-- Theorem stating that 20 new trailer homes were added -/
theorem twenty_new_homes :
  new_trailer_homes 30 15 3 12 = 20 := by
  sorry

#eval new_trailer_homes 30 15 3 12

end NUMINAMATH_CALUDE_twenty_new_homes_l2017_201781


namespace NUMINAMATH_CALUDE_solve_josie_problem_l2017_201750

def josie_problem (initial_amount : ℕ) (cassette_cost : ℕ) (num_cassettes : ℕ) (remaining_amount : ℕ) : Prop :=
  let total_cassette_cost := cassette_cost * num_cassettes
  let amount_after_cassettes := initial_amount - total_cassette_cost
  let headphone_cost := amount_after_cassettes - remaining_amount
  headphone_cost = 25

theorem solve_josie_problem :
  josie_problem 50 9 2 7 := by sorry

end NUMINAMATH_CALUDE_solve_josie_problem_l2017_201750


namespace NUMINAMATH_CALUDE_calculation_proof_l2017_201757

theorem calculation_proof : (-3)^2 - Real.sqrt 4 + (1/2)⁻¹ = 9 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2017_201757


namespace NUMINAMATH_CALUDE_smallest_value_zero_l2017_201775

-- Define the function y
def y (x p q : ℝ) : ℝ := x^3 + x^2 + p*x + q

-- State the theorem
theorem smallest_value_zero (p : ℝ) :
  ∃ q : ℝ, (∀ x : ℝ, y x p q ≥ 0) ∧ (∃ x : ℝ, y x p q = 0) ∧ q = -2/27 :=
sorry

end NUMINAMATH_CALUDE_smallest_value_zero_l2017_201775


namespace NUMINAMATH_CALUDE_total_profit_is_56700_l2017_201729

/-- Given a profit sharing ratio and c's profit, calculate the total profit -/
def calculate_total_profit (ratio_a ratio_b ratio_c : ℕ) (profit_c : ℕ) : ℕ :=
  let total_parts := ratio_a + ratio_b + ratio_c
  let part_value := profit_c / ratio_c
  total_parts * part_value

/-- Theorem: The total profit is $56,700 given the specified conditions -/
theorem total_profit_is_56700 :
  calculate_total_profit 8 9 10 21000 = 56700 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_56700_l2017_201729


namespace NUMINAMATH_CALUDE_blue_ball_probability_l2017_201748

/-- Represents a container with red and blue balls -/
structure Container where
  red : ℕ
  blue : ℕ

/-- The probability of selecting a blue ball from a container -/
def blueProbability (c : Container) : ℚ :=
  c.blue / (c.red + c.blue)

/-- The containers X, Y, and Z -/
def X : Container := ⟨3, 7⟩
def Y : Container := ⟨5, 5⟩
def Z : Container := ⟨6, 4⟩

/-- The list of all containers -/
def containers : List Container := [X, Y, Z]

/-- The probability of selecting each container -/
def containerProbability : ℚ := 1 / containers.length

/-- The overall probability of selecting a blue ball -/
def overallBlueProbability : ℚ :=
  (containers.map blueProbability).sum / containers.length

theorem blue_ball_probability :
  overallBlueProbability = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_blue_ball_probability_l2017_201748


namespace NUMINAMATH_CALUDE_mild_curries_count_mild_curries_proof_l2017_201712

/-- The number of peppers needed for different curry types -/
def peppers_per_curry : List Nat := [3, 2, 1]

/-- The number of curries of each type previously bought -/
def previous_curries : List Nat := [30, 30, 10]

/-- The number of spicy curries now bought -/
def current_spicy_curries : Nat := 15

/-- The reduction in total peppers bought -/
def pepper_reduction : Nat := 40

/-- Calculate the total number of peppers previously bought -/
def previous_total_peppers : Nat :=
  List.sum (List.zipWith (· * ·) peppers_per_curry previous_curries)

/-- Calculate the current total number of peppers bought -/
def current_total_peppers : Nat := previous_total_peppers - pepper_reduction

/-- Calculate the number of peppers used for current spicy curries -/
def current_spicy_peppers : Nat := peppers_per_curry[1] * current_spicy_curries

theorem mild_curries_count : Nat :=
  current_total_peppers - current_spicy_peppers

theorem mild_curries_proof : mild_curries_count = 90 := by
  sorry

end NUMINAMATH_CALUDE_mild_curries_count_mild_curries_proof_l2017_201712


namespace NUMINAMATH_CALUDE_real_roots_of_polynomial_l2017_201735

/-- The polynomial function we're investigating -/
def f (x : ℝ) : ℝ := x^4 - 4*x^3 + 3*x^2 + 2*x - 6

/-- Theorem stating that -1 and 3 are the only real roots of the polynomial -/
theorem real_roots_of_polynomial :
  (∀ x : ℝ, f x = 0 ↔ x = -1 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_real_roots_of_polynomial_l2017_201735


namespace NUMINAMATH_CALUDE_inequality_solution_l2017_201740

def inequality (x : ℝ) : Prop :=
  2*x^4 + x^2 - 2*x - 3*x^2*|x-1| + 1 ≥ 0

def solution_set : Set ℝ :=
  {x | x ≤ -(1 + Real.sqrt 5)/2 ∨ 
       (-1 ≤ x ∧ x ≤ 1/2) ∨ 
       x ≥ (Real.sqrt 5 - 1)/2}

theorem inequality_solution : 
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2017_201740


namespace NUMINAMATH_CALUDE_points_on_line_l2017_201773

/-- Three points lie on the same line if and only if the slope between any two pairs of points is equal. -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

/-- The theorem states that if the given points are collinear, then a = -7/23 -/
theorem points_on_line (a : ℝ) : 
  collinear (3, -5) (-a + 2, 3) (2*a + 3, 2) → a = -7/23 := by
  sorry

end NUMINAMATH_CALUDE_points_on_line_l2017_201773


namespace NUMINAMATH_CALUDE_trail_mix_weight_l2017_201721

/-- The weight of peanuts in pounds -/
def peanuts : ℝ := 0.16666666666666666

/-- The weight of chocolate chips in pounds -/
def chocolate_chips : ℝ := 0.16666666666666666

/-- The weight of raisins in pounds -/
def raisins : ℝ := 0.08333333333333333

/-- The total weight of trail mix in pounds -/
def total_weight : ℝ := peanuts + chocolate_chips + raisins

/-- Theorem stating that the total weight of trail mix is equal to 0.41666666666666663 pounds -/
theorem trail_mix_weight : total_weight = 0.41666666666666663 := by sorry

end NUMINAMATH_CALUDE_trail_mix_weight_l2017_201721


namespace NUMINAMATH_CALUDE_product_mb_range_l2017_201705

/-- Given a line y = mx + b with slope m = 3/4 and y-intercept b = -1/3,
    the product mb satisfies -1 < mb < 0. -/
theorem product_mb_range (m b : ℚ) : 
  m = 3/4 → b = -1/3 → -1 < m * b ∧ m * b < 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mb_range_l2017_201705


namespace NUMINAMATH_CALUDE_price_problem_solution_l2017_201792

/-- The price of sugar and salt -/
def price_problem (sugar_price salt_price : ℝ) : Prop :=
  let sugar_3kg_salt_1kg := 3 * sugar_price + salt_price
  sugar_price = 1.5 ∧ sugar_3kg_salt_1kg = 5 →
  2 * sugar_price + 5 * salt_price = 5.5

/-- The solution to the price problem -/
theorem price_problem_solution :
  ∃ (sugar_price salt_price : ℝ), price_problem sugar_price salt_price :=
sorry

end NUMINAMATH_CALUDE_price_problem_solution_l2017_201792


namespace NUMINAMATH_CALUDE_candy_probability_theorem_l2017_201791

-- Define the type for a packet of candies
structure Packet where
  blue : ℕ
  total : ℕ

-- Define the function to calculate the probability of drawing a blue candy from a box
def boxProbability (p1 p2 : Packet) : ℚ :=
  (p1.blue + p2.blue : ℚ) / (p1.total + p2.total : ℚ)

-- Theorem statement
theorem candy_probability_theorem :
  ∃ (p1 p2 p3 p4 : Packet),
    (boxProbability p1 p2 = 5/13 ∨ boxProbability p1 p2 = 7/18) ∧
    (boxProbability p3 p4 ≠ 17/40) ∧
    (∀ (p5 p6 : Packet), 3/8 ≤ boxProbability p5 p6 ∧ boxProbability p5 p6 ≤ 2/5) :=
by sorry

end NUMINAMATH_CALUDE_candy_probability_theorem_l2017_201791


namespace NUMINAMATH_CALUDE_wally_initial_tickets_l2017_201719

/-- Proves that Wally had 400 tickets initially given the conditions of the problem -/
theorem wally_initial_tickets : 
  ∀ (total : ℕ) (jensen finley : ℕ),
  (3 : ℚ) / 4 * total = jensen + finley →
  jensen * 11 = finley * 4 →
  finley = 220 →
  total = 400 :=
by sorry

end NUMINAMATH_CALUDE_wally_initial_tickets_l2017_201719
