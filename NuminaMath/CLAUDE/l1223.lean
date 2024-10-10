import Mathlib

namespace smallest_b_in_arithmetic_sequence_l1223_122326

theorem smallest_b_in_arithmetic_sequence (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- a, b, c are positive
  ∃ (d : ℝ), a = b - d ∧ c = b + d →  -- a, b, c form an arithmetic sequence
  a * b * c = 216 →  -- product is 216
  ∀ (x : ℝ), (∃ (y z : ℝ), 0 < y ∧ 0 < x ∧ 0 < z ∧  -- for any positive x, y, z in arithmetic sequence
    ∃ (e : ℝ), y = x - e ∧ z = x + e ∧  
    y * x * z = 216) →  -- with product 216
  x ≥ b →  -- if x is greater than or equal to b
  b = 6  -- then b must be 6
  := by sorry

end smallest_b_in_arithmetic_sequence_l1223_122326


namespace cos_four_theta_l1223_122362

theorem cos_four_theta (θ : Real) 
  (h : Real.exp (Real.log 2 * (-2 + 3 * Real.cos θ)) + 1 = Real.exp (Real.log 2 * (1/2 + 2 * Real.cos θ))) : 
  Real.cos (4 * θ) = -1/2 := by
sorry

end cos_four_theta_l1223_122362


namespace light_green_yellow_percentage_l1223_122391

-- Define the variables
def light_green_volume : ℝ := 5
def dark_green_volume : ℝ := 1.66666666667
def dark_green_yellow_percentage : ℝ := 0.4
def mixture_yellow_percentage : ℝ := 0.25

-- Define the theorem
theorem light_green_yellow_percentage :
  ∃ x : ℝ, 
    x * light_green_volume + dark_green_yellow_percentage * dark_green_volume = 
    mixture_yellow_percentage * (light_green_volume + dark_green_volume) ∧ 
    x = 0.2 := by
  sorry

end light_green_yellow_percentage_l1223_122391


namespace john_paid_1273_l1223_122358

/-- Calculates the amount John paid out of pocket for his purchases --/
def john_out_of_pocket (exchange_rate : ℝ) (computer_cost gaming_chair_cost accessories_cost : ℝ)
  (computer_discount gaming_chair_discount : ℝ) (sales_tax : ℝ)
  (playstation_value playstation_discount bicycle_price : ℝ) : ℝ :=
  let discounted_computer := computer_cost * (1 - computer_discount)
  let discounted_chair := gaming_chair_cost * (1 - gaming_chair_discount)
  let total_before_tax := discounted_computer + discounted_chair + accessories_cost
  let total_after_tax := total_before_tax * (1 + sales_tax)
  let playstation_sale := playstation_value * (1 - playstation_discount)
  let total_sales := playstation_sale + bicycle_price
  total_after_tax - total_sales

/-- Theorem stating that John paid $1273 out of pocket --/
theorem john_paid_1273 :
  john_out_of_pocket 100 1500 400 300 0.2 0.1 0.05 600 0.2 200 = 1273 := by
  sorry

end john_paid_1273_l1223_122358


namespace video_votes_theorem_l1223_122367

theorem video_votes_theorem (total_votes : ℕ) (score : ℤ) (like_percentage : ℚ) : 
  like_percentage = 3/4 ∧ 
  score = 120 ∧ 
  (like_percentage * total_votes : ℚ).num * 2 - total_votes = score → 
  total_votes = 240 := by
sorry

end video_votes_theorem_l1223_122367


namespace positive_real_inequality_general_real_inequality_l1223_122315

-- Part 1
theorem positive_real_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 / b ≥ 2*a - b := by sorry

-- Part 2
theorem general_real_inequality (a b : ℝ) :
  a^2 + b^2 + 3 ≥ a*b + Real.sqrt 3 * (a + b) := by sorry

end positive_real_inequality_general_real_inequality_l1223_122315


namespace emilys_purchase_cost_l1223_122305

/-- The total cost of Emily's purchase including installation service -/
theorem emilys_purchase_cost : 
  let curtain_pairs : ℕ := 2
  let curtain_price : ℚ := 30
  let wall_prints : ℕ := 9
  let wall_print_price : ℚ := 15
  let installation_fee : ℚ := 50
  (curtain_pairs : ℚ) * curtain_price + 
  (wall_prints : ℚ) * wall_print_price + 
  installation_fee = 245 :=
by sorry

end emilys_purchase_cost_l1223_122305


namespace four_digit_divisible_by_fourteen_l1223_122383

theorem four_digit_divisible_by_fourteen (n : Nat) : 
  n < 10 ∧ 945 * n < 10000 ∧ 945 * n ≥ 1000 ∧ (945 * n) % 14 = 0 → n = 8 :=
by sorry

end four_digit_divisible_by_fourteen_l1223_122383


namespace fraction_equals_point_eight_seven_five_l1223_122388

theorem fraction_equals_point_eight_seven_five (a : ℕ+) :
  (a : ℚ) / (a + 35 : ℚ) = 7/8 → a = 245 := by
  sorry

end fraction_equals_point_eight_seven_five_l1223_122388


namespace integral_even_function_l1223_122372

/-- Given that f(x) = ax^2 + (a-2)x + a^2 is an even function, 
    prove that the integral of (x^2 + x + √(4 - x^2)) from -a to a equals 16/3 + 2π -/
theorem integral_even_function (a : ℝ) 
  (h : ∀ x, a*x^2 + (a-2)*x + a^2 = a*(-x)^2 + (a-2)*(-x) + a^2) :
  ∫ x in (-a)..a, (x^2 + x + Real.sqrt (4 - x^2)) = 16/3 + 2*Real.pi := by
  sorry

end integral_even_function_l1223_122372


namespace sqrt_2023_bound_l1223_122324

theorem sqrt_2023_bound (n : ℤ) : n < Real.sqrt 2023 ∧ Real.sqrt 2023 < n + 1 → n = 44 := by
  sorry

end sqrt_2023_bound_l1223_122324


namespace remainder_problem_l1223_122396

theorem remainder_problem (x y : ℤ) 
  (h1 : x % 82 = 5)
  (h2 : (x + y^2) % 41 = 0) :
  (x + y^3 + 7) % 61 = 45 := by
  sorry

end remainder_problem_l1223_122396


namespace largest_integer_y_f_42_is_integer_largest_y_is_42_l1223_122328

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def f (y : ℤ) : ℚ := (y^2 + 3*y + 10) / (y - 4)

theorem largest_integer_y : ∀ y : ℤ, is_integer (f y) → y ≤ 42 :=
by sorry

theorem f_42_is_integer : is_integer (f 42) :=
by sorry

theorem largest_y_is_42 : 
  (∃ y : ℤ, is_integer (f y)) ∧ 
  (∀ y : ℤ, is_integer (f y) → y ≤ 42) ∧ 
  is_integer (f 42) :=
by sorry

end largest_integer_y_f_42_is_integer_largest_y_is_42_l1223_122328


namespace max_popsicles_with_10_dollars_l1223_122373

/-- Represents the number of popsicles that can be bought with a given amount of money -/
def max_popsicles (single_cost : ℕ) (box4_cost : ℕ) (box6_cost : ℕ) (budget : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of popsicles that can be bought with $10 is 14 -/
theorem max_popsicles_with_10_dollars :
  max_popsicles 1 3 4 10 = 14 :=
sorry

end max_popsicles_with_10_dollars_l1223_122373


namespace box_weights_sum_l1223_122380

/-- Given four boxes with weights a, b, c, and d, where:
    - The weight of box d is 60 pounds
    - The combined weight of (a,b) is 132 pounds
    - The combined weight of (a,c) is 136 pounds
    - The combined weight of (b,c) is 138 pounds
    Prove that the sum of weights a, b, and c is 203 pounds. -/
theorem box_weights_sum (a b c d : ℝ) 
    (hd : d = 60)
    (hab : a + b = 132)
    (hac : a + c = 136)
    (hbc : b + c = 138) :
    a + b + c = 203 := by
  sorry

end box_weights_sum_l1223_122380


namespace rectangle_areas_l1223_122343

theorem rectangle_areas (square_area : ℝ) (ratio1_width ratio1_length ratio2_width ratio2_length : ℕ) :
  square_area = 98 →
  ratio1_width = 2 →
  ratio1_length = 3 →
  ratio2_width = 3 →
  ratio2_length = 8 →
  ∃ (rect1_perim rect2_perim : ℝ),
    4 * Real.sqrt square_area = rect1_perim + rect2_perim ∧
    (rect1_perim * ratio1_width * rect1_perim * ratio1_length) / ((ratio1_width + ratio1_length) ^ 2) =
    (rect2_perim * ratio2_width * rect2_perim * ratio2_length) / ((ratio2_width + ratio2_length) ^ 2) →
  (rect1_perim * ratio1_width * rect1_perim * ratio1_length) / ((ratio1_width + ratio1_length) ^ 2) = 64 / 3 :=
by sorry

end rectangle_areas_l1223_122343


namespace thanksgiving_to_christmas_l1223_122378

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents months of the year -/
inductive Month
  | November
  | December

/-- Represents a date in a month -/
structure Date where
  month : Month
  day : Nat

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

/-- Function to get the day of the week after n days -/
def dayAfter (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => dayAfter (nextDay d) n

/-- Theorem: If Thanksgiving is on Thursday, November 28, then December 25 falls on a Wednesday -/
theorem thanksgiving_to_christmas 
  (thanksgiving : Date)
  (thanksgiving_day : DayOfWeek)
  (h1 : thanksgiving.month = Month.November)
  (h2 : thanksgiving.day = 28)
  (h3 : thanksgiving_day = DayOfWeek.Thursday) :
  dayAfter thanksgiving_day 27 = DayOfWeek.Wednesday :=
by
  sorry

end thanksgiving_to_christmas_l1223_122378


namespace total_stamps_is_72_l1223_122384

/-- Calculates the total number of stamps needed for Valerie's mailing --/
def total_stamps : ℕ :=
  let thank_you_cards := 5
  let thank_you_stamps_per_card := 2
  let water_bill_stamps := 3
  let electric_bill_stamps := 2
  let internet_bill_stamps := 5
  let rebate_stamps_per_envelope := 2
  let job_app_stamps_per_envelope := 1
  let bill_types := 3
  let additional_rebates := 3

  let bill_stamps := water_bill_stamps + electric_bill_stamps + internet_bill_stamps
  let rebates := bill_types + additional_rebates
  let job_applications := 2 * rebates

  thank_you_cards * thank_you_stamps_per_card +
  bill_stamps +
  rebates * rebate_stamps_per_envelope +
  job_applications * job_app_stamps_per_envelope

theorem total_stamps_is_72 : total_stamps = 72 := by
  sorry

end total_stamps_is_72_l1223_122384


namespace no_valid_a_l1223_122345

theorem no_valid_a : ¬∃ a : ℝ, ∀ x : ℝ, x^2 + a*x + a - 2 > 0 := by
  sorry

end no_valid_a_l1223_122345


namespace scientific_notation_of_8500_l1223_122399

theorem scientific_notation_of_8500 : 
  ∃ (a : ℝ) (n : ℤ), 8500 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 :=
by
  -- The proof would go here
  sorry

end scientific_notation_of_8500_l1223_122399


namespace percentage_reduction_price_increase_l1223_122337

-- Define the original price
def original_price : ℝ := 50

-- Define the final price after two reductions
def final_price : ℝ := 32

-- Define the initial profit per kilogram
def initial_profit : ℝ := 10

-- Define the initial daily sales
def initial_sales : ℝ := 500

-- Define the sales decrease rate
def sales_decrease_rate : ℝ := 20

-- Define the target daily profit
def target_profit : ℝ := 6000

-- Theorem for the percentage reduction
theorem percentage_reduction (x : ℝ) : 
  original_price * (1 - x)^2 = final_price → x = 0.2 := by sorry

-- Theorem for the price increase
theorem price_increase (x : ℝ) :
  (initial_profit + x) * (initial_sales - sales_decrease_rate * x) = target_profit →
  x > 0 →
  ∀ y, y > 0 → 
  (initial_profit + y) * (initial_sales - sales_decrease_rate * y) = target_profit →
  x ≤ y →
  x = 5 := by sorry

end percentage_reduction_price_increase_l1223_122337


namespace arithmetic_problems_l1223_122327

theorem arithmetic_problems :
  (270 * 9 = 2430) ∧
  (735 / 7 = 105) ∧
  (99 * 9 = 891) := by
sorry

end arithmetic_problems_l1223_122327


namespace ball_probability_l1223_122323

theorem ball_probability (total : Nat) (white green yellow red purple blue black : Nat)
  (h_total : total = 200)
  (h_white : white = 50)
  (h_green : green = 40)
  (h_yellow : yellow = 20)
  (h_red : red = 30)
  (h_purple : purple = 30)
  (h_blue : blue = 10)
  (h_black : black = 20)
  (h_sum : total = white + green + yellow + red + purple + blue + black) :
  (white + green + yellow + blue : ℚ) / total = 0.6 := by
  sorry

end ball_probability_l1223_122323


namespace arithmetic_sequence_length_l1223_122303

/-- 
An arithmetic sequence starting at 5, with a common difference of 3, 
and ending at 140, contains 46 terms.
-/
theorem arithmetic_sequence_length : 
  ∀ (a : ℕ → ℕ), 
    a 0 = 5 → 
    (∀ n, a (n + 1) = a n + 3) → 
    (∃ m, a m = 140) → 
    (∃ n, a n = 140 ∧ n = 45) :=
by sorry

end arithmetic_sequence_length_l1223_122303


namespace max_frisbee_receipts_l1223_122300

/-- Represents the total receipts from frisbee sales -/
def total_receipts (x y : ℕ) : ℕ := 3 * x + 4 * y

/-- Proves that the maximum total receipts from frisbee sales is $204 -/
theorem max_frisbee_receipts :
  ∀ x y : ℕ,
  x + y = 60 →
  y ≥ 24 →
  total_receipts x y ≤ 204 :=
by
  sorry

#eval total_receipts 36 24  -- Should output 204

end max_frisbee_receipts_l1223_122300


namespace factor_sum_l1223_122329

theorem factor_sum (R S : ℝ) : 
  (∃ d e : ℝ, (X^4 + R*X^2 + S) = (X^2 - 3*X + 7) * (X^2 + d*X + e)) → 
  R + S = 54 :=
sorry

end factor_sum_l1223_122329


namespace intersection_solution_set_l1223_122312

theorem intersection_solution_set (a b : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + b < 0 ↔ (x^2 - 2*x - 3 < 0 ∧ x^2 + x - 6 < 0)) → 
  a + b = -3 := by
sorry

end intersection_solution_set_l1223_122312


namespace selling_price_ratio_l1223_122365

theorem selling_price_ratio (c x y : ℝ) (hx : x = 0.80 * c) (hy : y = 1.25 * c) :
  y / x = 25 / 16 := by
  sorry

end selling_price_ratio_l1223_122365


namespace tribe_leadership_arrangements_l1223_122320

def tribe_size : ℕ := 15
def num_chiefs : ℕ := 1
def num_supporting_chiefs : ℕ := 2
def inferior_officers_per_chief : ℕ := 3

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem tribe_leadership_arrangements :
  tribe_size * (tribe_size - 1) * (tribe_size - 2) *
  (choose (tribe_size - 3) inferior_officers_per_chief) *
  (choose (tribe_size - 3 - inferior_officers_per_chief) inferior_officers_per_chief) = 3243240 :=
by sorry

end tribe_leadership_arrangements_l1223_122320


namespace parabola_directrix_l1223_122316

/-- Given a parabola y = -3x^2 + 6x - 7, its directrix is y = -47/12 -/
theorem parabola_directrix (x y : ℝ) : 
  y = -3 * x^2 + 6 * x - 7 → 
  ∃ (k : ℝ), k = -47/12 ∧ (∀ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y → 
    (p.1 - 1)^2 + (p.2 - k)^2 = (p.2 + 4)^2 / 9) :=
by sorry

end parabola_directrix_l1223_122316


namespace sufficient_not_necessary_condition_l1223_122356

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, (x + 1) * (x - 3) < 0 → x < 3) ∧
  (∃ x, x < 3 ∧ (x + 1) * (x - 3) ≥ 0) :=
sorry

end sufficient_not_necessary_condition_l1223_122356


namespace coin_bag_total_l1223_122364

theorem coin_bag_total (p : ℕ) : ∃ (p : ℕ), 
  (0.01 * p + 0.05 * (3 * p) + 0.50 * (12 * p) : ℚ) = 616 := by
  sorry

end coin_bag_total_l1223_122364


namespace hiking_team_selection_l1223_122344

theorem hiking_team_selection (n : ℕ) (k : ℕ) : n = 10 → k = 3 → Nat.choose n k = 120 := by
  sorry

end hiking_team_selection_l1223_122344


namespace sufficient_not_necessary_l1223_122306

theorem sufficient_not_necessary (m n : ℝ) :
  (∀ m n : ℝ, m / n - 1 = 0 → m - n = 0) ∧
  (∃ m n : ℝ, m - n = 0 ∧ ¬(m / n - 1 = 0)) :=
by sorry

end sufficient_not_necessary_l1223_122306


namespace base_prime_182_l1223_122382

/-- Represents a number in base prime notation --/
def BasePrime : Type := List Nat

/-- Converts a natural number to its base prime representation --/
def toBasePrime (n : Nat) : BasePrime :=
  sorry

/-- Theorem: The base prime representation of 182 is [1, 0, 0, 1, 0, 1] --/
theorem base_prime_182 : toBasePrime 182 = [1, 0, 0, 1, 0, 1] := by
  sorry

end base_prime_182_l1223_122382


namespace f_monotone_condition_l1223_122319

-- Define the piecewise function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 2 else Real.log (|x - m|)

-- Define monotonically increasing property
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

-- State the theorem
theorem f_monotone_condition (m : ℝ) :
  (∀ x y, 0 ≤ x ∧ x < y → f m x ≤ f m y) ↔ m ≤ 9/10 :=
sorry

end f_monotone_condition_l1223_122319


namespace cindy_calculation_l1223_122308

def original_number : ℝ := (23 * 5) + 7

theorem cindy_calculation : 
  Int.floor ((original_number + 7) / 5) = 26 := by
  sorry

end cindy_calculation_l1223_122308


namespace fraction_to_decimal_l1223_122361

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := by sorry

end fraction_to_decimal_l1223_122361


namespace probability_same_group_l1223_122381

/-- The probability that two specific cards (5 and 14) are in the same group
    when drawing 4 cards from 20, where groups are determined by card value. -/
theorem probability_same_group : 
  let total_cards : ℕ := 20
  let cards_drawn : ℕ := 4
  let remaining_cards : ℕ := total_cards - cards_drawn + 2  -- +2 because 5 and 14 are known
  let favorable_outcomes : ℕ := (remaining_cards - 14 + 1) * (remaining_cards - 14) + 
                                (5 - 1) * (5 - 2)
  let total_outcomes : ℕ := remaining_cards * (remaining_cards - 1)
  (favorable_outcomes : ℚ) / total_outcomes = 7 / 51 := by
  sorry

end probability_same_group_l1223_122381


namespace brendas_age_is_real_l1223_122304

/-- Represents the ages of individuals --/
structure Ages where
  addison : ℝ
  brenda : ℝ
  carlos : ℝ
  janet : ℝ

/-- The conditions given in the problem --/
def age_conditions (ages : Ages) : Prop :=
  ages.addison = 4 * ages.brenda ∧
  ages.carlos = 2 * ages.brenda ∧
  ages.addison = ages.janet

/-- Theorem stating that Brenda's age is a positive real number --/
theorem brendas_age_is_real (ages : Ages) (h : age_conditions ages) :
  ∃ (B : ℝ), B > 0 ∧ ages.brenda = B :=
sorry

end brendas_age_is_real_l1223_122304


namespace diagonals_from_vertex_l1223_122354

/-- For a polygon with interior angles summing to 540°, 
    the number of diagonals that can be drawn from one vertex is 2. -/
theorem diagonals_from_vertex (n : ℕ) : 
  (n - 2) * 180 = 540 → (n - 3 : ℕ) = 2 := by
  sorry

end diagonals_from_vertex_l1223_122354


namespace nine_appears_once_l1223_122395

def multiply_987654321_by_9 : ℕ := 987654321 * 9

def count_digit (n : ℕ) (d : ℕ) : ℕ :=
  n.digits 10
    |>.filter (· = d)
    |>.length

theorem nine_appears_once :
  count_digit multiply_987654321_by_9 9 = 1 := by
  sorry

end nine_appears_once_l1223_122395


namespace other_root_of_quadratic_l1223_122346

theorem other_root_of_quadratic (a : ℝ) : 
  (1^2 + a*1 + 2 = 0) → ∃ b : ℝ, b ≠ 1 ∧ b^2 + a*b + 2 = 0 ∧ b = 2 :=
sorry

end other_root_of_quadratic_l1223_122346


namespace complex_sum_problem_l1223_122332

theorem complex_sum_problem (x y u v w z : ℝ) : 
  y = 5 →
  w = -x - u →
  Complex.I * (x + y + u + v + w + z) = 4 * Complex.I →
  v + z = -1 := by
sorry

end complex_sum_problem_l1223_122332


namespace calculation_proof_l1223_122353

theorem calculation_proof : (1/2)⁻¹ - 3 * Real.tan (30 * π / 180) + (1 - Real.pi)^0 + Real.sqrt 12 = Real.sqrt 3 + 3 := by
  sorry

end calculation_proof_l1223_122353


namespace common_difference_is_two_l1223_122341

/-- An arithmetic sequence with specified terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m
  fifth_term : a 5 = 6
  third_term : a 3 = 2

/-- The common difference of an arithmetic sequence -/
def commonDifference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem common_difference_is_two (seq : ArithmeticSequence) :
  commonDifference seq = 2 := by
  sorry

end common_difference_is_two_l1223_122341


namespace sunflower_seeds_weight_l1223_122390

/-- The weight of a bag of sunflower seeds in grams -/
def bag_weight : ℝ := 250

/-- The number of bags -/
def num_bags : ℕ := 8

/-- Conversion factor from grams to kilograms -/
def grams_to_kg : ℝ := 1000

theorem sunflower_seeds_weight :
  (bag_weight * num_bags) / grams_to_kg = 2 := by
  sorry

end sunflower_seeds_weight_l1223_122390


namespace sum_of_squares_and_products_l1223_122359

theorem sum_of_squares_and_products (a b c : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → 
  a^2 + b^2 + c^2 = 48 → 
  a*b + b*c + c*a = 24 → 
  a + b + c = 4 * Real.sqrt 6 := by
sorry

end sum_of_squares_and_products_l1223_122359


namespace all_digits_are_perfect_cube_units_l1223_122376

-- Define the set of possible units digits of perfect cubes modulo 10
def PerfectCubeUnitsDigits : Set ℕ :=
  {d | ∃ n : ℤ, (n^3 : ℤ) % 10 = d}

-- Theorem statement
theorem all_digits_are_perfect_cube_units : PerfectCubeUnitsDigits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} := by
  sorry

end all_digits_are_perfect_cube_units_l1223_122376


namespace equation_equivalence_l1223_122387

theorem equation_equivalence (p q : ℝ) 
  (hp1 : p ≠ 0) (hp2 : p ≠ 5) (hq1 : q ≠ 0) (hq2 : q ≠ 7) :
  (3 / p + 4 / q = 1 / 3) ↔ (9 * q / (q - 12) = p) :=
by sorry

end equation_equivalence_l1223_122387


namespace log_inequality_l1223_122309

theorem log_inequality (x : ℝ) :
  (Real.log x / Real.log (1/2) - Real.sqrt (2 - Real.log x / Real.log 4) + 1 ≤ 0) ↔
  (1 / Real.sqrt 2 ≤ x ∧ x ≤ 16) :=
by sorry

end log_inequality_l1223_122309


namespace arithmetic_sequence_sum_2_to_20_l1223_122313

def arithmetic_sequence_sum (a : ℕ) (l : ℕ) (d : ℕ) : ℕ :=
  let n : ℕ := (l - a) / d + 1
  n * (a + l) / 2

theorem arithmetic_sequence_sum_2_to_20 :
  arithmetic_sequence_sum 2 20 2 = 110 := by
  sorry

end arithmetic_sequence_sum_2_to_20_l1223_122313


namespace inequality_system_solution_l1223_122394

theorem inequality_system_solution :
  let S := {x : ℝ | 3 < x ∧ x ≤ 4}
  S = {x : ℝ | 3*x + 4 ≥ 4*x ∧ 2*(x - 1) + x > 7} := by sorry

end inequality_system_solution_l1223_122394


namespace exactly_one_survives_l1223_122322

theorem exactly_one_survives (p q : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) (hq : 0 ≤ q ∧ q ≤ 1) :
  (p * (1 - q)) + ((1 - p) * q) = p + q - p * q := by
  sorry

end exactly_one_survives_l1223_122322


namespace constant_term_is_70_l1223_122355

/-- 
Given a natural number n, this function represents the coefficient 
of the r-th term in the expansion of (x + 1/x)^(2n)
-/
def binomialCoeff (n : ℕ) (r : ℕ) : ℕ := Nat.choose (2 * n) r

/-- 
This theorem states that if the coefficients of the fourth and sixth terms 
in the expansion of (x + 1/x)^(2n) are equal, then the constant term 
in the expansion is 70
-/
theorem constant_term_is_70 (n : ℕ) 
  (h : binomialCoeff n 3 = binomialCoeff n 5) : 
  binomialCoeff n 4 = 70 := by
  sorry


end constant_term_is_70_l1223_122355


namespace isosceles_triangle_leg_length_l1223_122363

/-- Represents an isosceles triangle with given properties -/
structure IsoscelesTriangle where
  perimeter : ℝ
  side_ratio : ℝ
  leg_length : ℝ
  h_perimeter_positive : perimeter > 0
  h_ratio_positive : side_ratio > 0
  h_leg_length_positive : leg_length > 0
  h_perimeter_eq : perimeter = (1 + 2 * side_ratio) * leg_length / side_ratio

/-- Theorem stating the possible leg lengths of the isosceles triangle -/
theorem isosceles_triangle_leg_length 
  (triangle : IsoscelesTriangle) 
  (h_perimeter : triangle.perimeter = 70) 
  (h_ratio : triangle.side_ratio = 3) : 
  triangle.leg_length = 14 ∨ triangle.leg_length = 30 := by
  sorry

#check isosceles_triangle_leg_length

end isosceles_triangle_leg_length_l1223_122363


namespace factorial_ratio_l1223_122310

theorem factorial_ratio : Nat.factorial 45 / Nat.factorial 42 = 85140 := by sorry

end factorial_ratio_l1223_122310


namespace jelly_bean_count_l1223_122342

/-- The number of jelly beans in jar X -/
def jarX (total : ℕ) (y : ℕ) : ℕ := 3 * y - 400

/-- The number of jelly beans in jar Y -/
def jarY (total : ℕ) (x : ℕ) : ℕ := total - x

theorem jelly_bean_count (total : ℕ) (h : total = 1200) :
  ∃ y : ℕ, jarX total y + jarY total (jarX total y) = total ∧ jarX total y = 800 := by
sorry

end jelly_bean_count_l1223_122342


namespace original_average_proof_l1223_122377

theorem original_average_proof (n : ℕ) (original_avg new_avg : ℝ) : 
  n = 30 → 
  new_avg = 90 → 
  new_avg = 2 * original_avg → 
  original_avg = 45 := by
sorry

end original_average_proof_l1223_122377


namespace min_tiles_for_floor_l1223_122334

/-- Calculates the minimum number of rectangular tiles needed to cover a rectangular floor -/
def min_tiles_needed (tile_length inch_per_foot tile_width floor_length floor_width : ℕ) : ℕ :=
  let floor_area := (floor_length * inch_per_foot) * (floor_width * inch_per_foot)
  let tile_area := tile_length * tile_width
  (floor_area + tile_area - 1) / tile_area

/-- The minimum number of 5x6 inch tiles needed to cover a 3x4 foot floor is 58 -/
theorem min_tiles_for_floor :
  min_tiles_needed 5 12 6 3 4 = 58 := by
  sorry

end min_tiles_for_floor_l1223_122334


namespace quadratic_inequality_solution_l1223_122330

-- Define the quadratic function
def f (x : ℝ) := x^2 - 6*x + 8

-- Define the solution set
def solution_set : Set ℝ := {x | x < 2 ∨ x > 4}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x > 0} = solution_set :=
sorry

end quadratic_inequality_solution_l1223_122330


namespace second_player_winning_strategy_l1223_122314

/-- Represents the n-gon coloring game -/
structure ColoringGame where
  n : ℕ  -- number of sides in the n-gon

/-- Defines when the second player has a winning strategy -/
def second_player_wins (game : ColoringGame) : Prop :=
  ∃ k : ℕ, game.n = 4 + 3 * k

/-- Theorem: The second player has a winning strategy if and only if n = 4 + 3k, where k ≥ 0 -/
theorem second_player_winning_strategy (game : ColoringGame) :
  second_player_wins game ↔ ∃ k : ℕ, game.n = 4 + 3 * k :=
by sorry

end second_player_winning_strategy_l1223_122314


namespace cylinder_height_comparison_l1223_122340

-- Define the cylinders
structure Cylinder where
  radius : ℝ
  height : ℝ

-- Define the theorem
theorem cylinder_height_comparison (c1 c2 : Cylinder) 
  (h_volume : π * c1.radius^2 * c1.height = π * c2.radius^2 * c2.height)
  (h_radius : c2.radius = 1.2 * c1.radius) :
  c1.height = 1.44 * c2.height := by
  sorry

end cylinder_height_comparison_l1223_122340


namespace initial_nickels_correct_l1223_122392

/-- The number of nickels Mike had initially -/
def initial_nickels : ℕ := 87

/-- The number of nickels Mike's dad borrowed -/
def borrowed_nickels : ℕ := 75

/-- The number of nickels Mike was left with -/
def remaining_nickels : ℕ := 12

/-- Theorem stating that the initial number of nickels is correct -/
theorem initial_nickels_correct : initial_nickels = borrowed_nickels + remaining_nickels := by
  sorry

end initial_nickels_correct_l1223_122392


namespace prob_win_5_eq_prob_win_total_eq_l1223_122389

/-- Probability of Team A winning a single game -/
def p : ℝ := 0.6

/-- Probability of Team B winning a single game -/
def q : ℝ := 1 - p

/-- Number of games in the series -/
def n : ℕ := 7

/-- Number of games needed to win the series -/
def k : ℕ := 4

/-- Probability of Team A winning the championship after exactly 5 games -/
def prob_win_5 : ℝ := Nat.choose 4 3 * p^4 * q

/-- Probability of Team A winning the championship -/
def prob_win_total : ℝ := 
  p^4 + Nat.choose 4 3 * p^4 * q + Nat.choose 5 3 * p^4 * q^2 + Nat.choose 6 3 * p^4 * q^3

/-- Theorem stating the probability of Team A winning after exactly 5 games -/
theorem prob_win_5_eq : prob_win_5 = 0.20736 := by sorry

/-- Theorem stating the overall probability of Team A winning the championship -/
theorem prob_win_total_eq : prob_win_total = 0.710208 := by sorry

end prob_win_5_eq_prob_win_total_eq_l1223_122389


namespace dodecahedron_intersection_area_l1223_122385

/-- The area of a regular pentagon formed by intersecting a plane with a regular dodecahedron -/
theorem dodecahedron_intersection_area (s : ℝ) :
  let dodecahedron_side_length : ℝ := s
  let intersection_pentagon_side_length : ℝ := s / 2
  let intersection_pentagon_area : ℝ := (5 / 4) * (intersection_pentagon_side_length ^ 2) * ((Real.sqrt 5 + 1) / 2)
  intersection_pentagon_area = (5 * s^2 * (Real.sqrt 5 + 1)) / 16 := by
  sorry

end dodecahedron_intersection_area_l1223_122385


namespace counterexample_exists_l1223_122347

theorem counterexample_exists : ∃ n : ℕ, ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n - 3)) := by
  sorry

end counterexample_exists_l1223_122347


namespace max_product_with_geometric_mean_l1223_122338

theorem max_product_with_geometric_mean (a b : ℝ) : 
  a > 0 → b > 0 → (3 : ℝ)^((a + b) / 2) = Real.sqrt 3 → ab ≤ (1 / 4 : ℝ) := by
  sorry

end max_product_with_geometric_mean_l1223_122338


namespace middle_term_of_five_term_arithmetic_sequence_l1223_122375

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem middle_term_of_five_term_arithmetic_sequence 
  (a : ℕ → ℝ) (h_arith : arithmetic_sequence a) 
  (h_first : a 1 = 21) (h_last : a 5 = 53) : 
  a 3 = 37 := by
sorry

end middle_term_of_five_term_arithmetic_sequence_l1223_122375


namespace i_power_2016_l1223_122397

-- Define i as a complex number
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem i_power_2016 : i^2016 = 1 :=
  -- Assume the given conditions
  have h1 : i^1 = i := by sorry
  have h2 : i^2 = -1 := by sorry
  have h3 : i^3 = -i := by sorry
  have h4 : i^4 = 1 := by sorry
  have h5 : i^5 = i := by sorry
  
  -- Proof goes here
  sorry

end i_power_2016_l1223_122397


namespace multiplication_by_hundred_l1223_122321

theorem multiplication_by_hundred : 38 * 100 = 3800 := by
  sorry

end multiplication_by_hundred_l1223_122321


namespace quadratic_inequality_range_l1223_122398

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a-1)*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 3 := by sorry

end quadratic_inequality_range_l1223_122398


namespace f_min_at_4_l1223_122336

/-- The quadratic function f(x) = x^2 - 8x + 15 -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 15

/-- The theorem stating that f(x) attains its minimum at x = 4 -/
theorem f_min_at_4 :
  ∀ x : ℝ, f x ≥ f 4 :=
sorry

end f_min_at_4_l1223_122336


namespace f_pi_half_value_l1223_122318

theorem f_pi_half_value : 
  let f : ℝ → ℝ := fun x ↦ x * Real.sin x + Real.cos x
  f (Real.pi / 2) = Real.pi / 2 := by
  sorry

end f_pi_half_value_l1223_122318


namespace pears_left_l1223_122369

/-- 
Given that Jason picked 46 pears, Keith picked 47 pears, and Mike ate 12 pears,
prove that the number of pears left is 81.
-/
theorem pears_left (jason_pears keith_pears : ℕ) (mike_ate : ℕ) 
  (h1 : jason_pears = 46)
  (h2 : keith_pears = 47)
  (h3 : mike_ate = 12) :
  jason_pears + keith_pears - mike_ate = 81 := by
  sorry

end pears_left_l1223_122369


namespace modified_square_boundary_length_l1223_122386

/-- The boundary length of a modified square figure --/
theorem modified_square_boundary_length :
  ∀ (square_area : ℝ) (num_segments : ℕ),
    square_area = 100 →
    num_segments = 4 →
    ∃ (boundary_length : ℝ),
      boundary_length = 5 * Real.pi + 10 := by
  sorry

end modified_square_boundary_length_l1223_122386


namespace bench_cost_l1223_122393

theorem bench_cost (bench_cost table_cost : ℕ) : 
  bench_cost + table_cost = 750 → 
  table_cost = 2 * bench_cost →
  bench_cost = 250 := by
  sorry

end bench_cost_l1223_122393


namespace or_not_implies_right_l1223_122357

theorem or_not_implies_right (p q : Prop) : (p ∨ q) → ¬p → q := by
  sorry

end or_not_implies_right_l1223_122357


namespace polygon_sides_l1223_122370

theorem polygon_sides (n : ℕ) : n > 2 →
  ∃ (x : ℝ), x > 0 ∧ x < 180 ∧ (n - 2) * 180 + x = 1350 →
  n = 9 := by
sorry

end polygon_sides_l1223_122370


namespace polynomial_roots_and_factorization_l1223_122350

theorem polynomial_roots_and_factorization (m : ℤ) :
  (∀ x : ℤ, 2 * x^4 + m * x^2 + 8 = 0 → (∃ y : ℤ, x = y)) →
  (m = -10 ∧
   ∀ x : ℝ, 2 * x^4 + m * x^2 + 8 = 2 * (x + 1) * (x - 1) * (x + 2) * (x - 2)) :=
by sorry

end polynomial_roots_and_factorization_l1223_122350


namespace polynomial_expansion_l1223_122352

/-- Proves the expansion of (3z^3 + 4z^2 - 2z + 1)(2z^2 - 3z + 5) -/
theorem polynomial_expansion (z : ℝ) :
  (3 * z^3 + 4 * z^2 - 2 * z + 1) * (2 * z^2 - 3 * z + 5) =
  10 * z^5 - 8 * z^4 + 11 * z^3 + 5 * z^2 - 10 * z + 5 := by
  sorry

end polynomial_expansion_l1223_122352


namespace quadratic_rational_solutions_l1223_122371

/-- A function that checks if a quadratic equation with rational coefficients has rational solutions -/
def has_rational_solutions (a b c : ℚ) : Prop :=
  ∃ x : ℚ, a * x^2 + b * x + c = 0

/-- The set of positive integer values of d for which 3x^2 + 7x + d = 0 has rational solutions -/
def D : Set ℕ+ :=
  {d : ℕ+ | has_rational_solutions 3 7 d.val}

theorem quadratic_rational_solutions :
  (∃ d1 d2 : ℕ+, d1 ≠ d2 ∧ D = {d1, d2}) ∧
  (∀ d1 d2 : ℕ+, d1 ∈ D → d2 ∈ D → d1.val * d2.val = 8) :=
sorry

end quadratic_rational_solutions_l1223_122371


namespace balloon_arrangements_l1223_122302

/-- The number of distinct arrangements of letters in a word -/
def distinctArrangements (totalLetters : Nat) (repeatedLetters : List (Nat)) : Nat :=
  Nat.factorial totalLetters / (repeatedLetters.map Nat.factorial).prod

theorem balloon_arrangements :
  distinctArrangements 7 [2, 2] = 1260 := by
  sorry

end balloon_arrangements_l1223_122302


namespace inverse_of_A_l1223_122351

def A : Matrix (Fin 2) (Fin 2) ℚ := !![5, -3; 3, -2]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![2, -3; 3, -5]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end inverse_of_A_l1223_122351


namespace exactly_one_zero_in_interval_l1223_122349

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

theorem exactly_one_zero_in_interval (a : ℝ) (h : a > 3) :
  ∃! x, x ∈ (Set.Ioo 0 2) ∧ f a x = 0 := by
sorry

end exactly_one_zero_in_interval_l1223_122349


namespace lcm_18_42_l1223_122307

theorem lcm_18_42 : Nat.lcm 18 42 = 126 := by sorry

end lcm_18_42_l1223_122307


namespace candy_fraction_of_earnings_l1223_122301

/-- Proves that the fraction of earnings spent on candy is 1/6 -/
theorem candy_fraction_of_earnings : 
  ∀ (candy_bar_price lollipop_price driveway_charge : ℚ)
    (candy_bars lollipops driveways : ℕ),
  candy_bar_price = 3/4 →
  lollipop_price = 1/4 →
  driveway_charge = 3/2 →
  candy_bars = 2 →
  lollipops = 4 →
  driveways = 10 →
  (candy_bar_price * candy_bars + lollipop_price * lollipops) / 
  (driveway_charge * driveways) = 1/6 :=
by sorry

end candy_fraction_of_earnings_l1223_122301


namespace max_segment_sum_l1223_122374

/-- A rhombus constructed from two equal equilateral triangles, divided into 2n^2 smaller triangles --/
structure Rhombus (n : ℕ) where
  triangles : Fin (2 * n^2) → ℕ
  triangle_values : ∀ i, 1 ≤ triangles i ∧ triangles i ≤ 2 * n^2
  distinct_values : ∀ i j, i ≠ j → triangles i ≠ triangles j

/-- The sum of positive differences on common segments of the rhombus --/
def segmentSum (n : ℕ) (r : Rhombus n) : ℕ :=
  sorry

/-- Theorem: The maximum sum of positive differences on common segments is 3n^4 - 4n^2 + 4n - 2 --/
theorem max_segment_sum (n : ℕ) : 
  (∀ r : Rhombus n, segmentSum n r ≤ 3 * n^4 - 4 * n^2 + 4 * n - 2) ∧
  (∃ r : Rhombus n, segmentSum n r = 3 * n^4 - 4 * n^2 + 4 * n - 2) :=
sorry

end max_segment_sum_l1223_122374


namespace sandy_puppies_given_to_friends_l1223_122368

/-- Given the initial number of puppies and the number of puppies left,
    calculate the number of puppies given to friends. -/
def puppies_given_to_friends (initial_puppies left_puppies : ℕ) : ℕ :=
  initial_puppies - left_puppies

/-- Theorem stating that for Sandy's specific case, the number of puppies
    given to friends is 4. -/
theorem sandy_puppies_given_to_friends :
  puppies_given_to_friends 8 4 = 4 := by
  sorry

end sandy_puppies_given_to_friends_l1223_122368


namespace stock_trading_profit_l1223_122317

/-- Represents the stock trading scenario described in the problem -/
def stock_trading (initial_investment : ℝ) (profit_rate : ℝ) (loss_rate : ℝ) (final_sale_rate : ℝ) : ℝ :=
  let first_sale := initial_investment * (1 + profit_rate)
  let second_sale := first_sale * (1 - loss_rate)
  let third_sale := second_sale * final_sale_rate
  let first_profit := first_sale - initial_investment
  let final_loss := second_sale - third_sale
  first_profit - final_loss

/-- Theorem stating that given the conditions in the problem, A's overall profit is 10 yuan -/
theorem stock_trading_profit :
  stock_trading 10000 0.1 0.1 0.9 = 10 := by
  sorry

end stock_trading_profit_l1223_122317


namespace warehouse_solution_l1223_122366

/-- Represents the problem of determining the number of warehouses on a straight road. -/
def WarehouseProblem (n : ℕ) : Prop :=
  -- n is odd
  ∃ k : ℕ, n = 2*k + 1 ∧
  -- Distance between warehouses is 1 km
  -- Each warehouse contains 8 tons of goods
  -- Truck capacity is 8 tons
  -- These conditions are implicit in the problem setup
  -- Optimal route distance is 300 km
  2 * k * (k + 1) - k = 300

/-- The solution to the warehouse problem is 25 warehouses. -/
theorem warehouse_solution : WarehouseProblem 25 := by
  sorry

#check warehouse_solution

end warehouse_solution_l1223_122366


namespace division_of_fractions_l1223_122339

theorem division_of_fractions : (5 : ℚ) / 6 / ((2 : ℚ) / 3) = (5 : ℚ) / 4 := by
  sorry

end division_of_fractions_l1223_122339


namespace product_of_integers_l1223_122348

theorem product_of_integers (a b : ℕ+) 
  (h1 : (a : ℚ) / (b : ℚ) = 12)
  (h2 : a + b = 144) :
  (a : ℚ) * (b : ℚ) = 248832 / 169 := by
sorry

end product_of_integers_l1223_122348


namespace harry_monday_speed_l1223_122331

/-- Harry's marathon running speeds throughout the week -/
def harry_speeds (monday_speed : ℝ) : Fin 5 → ℝ
  | 0 => monday_speed  -- Monday
  | 1 => 1.5 * monday_speed  -- Tuesday
  | 2 => 1.5 * monday_speed  -- Wednesday
  | 3 => 1.5 * monday_speed  -- Thursday
  | 4 => 1.6 * 1.5 * monday_speed  -- Friday

theorem harry_monday_speed :
  ∃ (monday_speed : ℝ), 
    (harry_speeds monday_speed 4 = 24) ∧ 
    (monday_speed = 10) := by
  sorry

end harry_monday_speed_l1223_122331


namespace oranges_per_group_l1223_122311

def total_oranges : ℕ := 356
def orange_groups : ℕ := 178

theorem oranges_per_group : total_oranges / orange_groups = 2 := by
  sorry

end oranges_per_group_l1223_122311


namespace second_place_prize_l1223_122325

theorem second_place_prize (total_prize : ℕ) (num_winners : ℕ) (first_prize : ℕ) (third_prize : ℕ) (other_prize : ℕ) :
  total_prize = 800 →
  num_winners = 18 →
  first_prize = 200 →
  third_prize = 120 →
  other_prize = 22 →
  (num_winners - 3) * other_prize + first_prize + third_prize + 150 = total_prize :=
by sorry

end second_place_prize_l1223_122325


namespace remainder_11_power_1995_mod_5_l1223_122333

theorem remainder_11_power_1995_mod_5 : 11^1995 % 5 = 1 := by
  sorry

end remainder_11_power_1995_mod_5_l1223_122333


namespace inscribed_square_area_l1223_122335

/-- A square inscribed in a semicircle with radius 1 -/
structure InscribedSquare where
  /-- The side length of the square -/
  side : ℝ
  /-- One side of the square is flush with the diameter of the semicircle -/
  flush_with_diameter : True
  /-- The square is inscribed in the semicircle -/
  inscribed : side^2 + (side/2)^2 = 1

/-- The area of an inscribed square is 4/5 -/
theorem inscribed_square_area (s : InscribedSquare) : s.side^2 = 4/5 := by
  sorry

#check inscribed_square_area

end inscribed_square_area_l1223_122335


namespace james_earnings_ratio_l1223_122360

theorem james_earnings_ratio :
  ∀ (february_earnings : ℕ),
    4000 + february_earnings + (february_earnings - 2000) = 18000 →
    february_earnings / 4000 = 2 := by
  sorry

end james_earnings_ratio_l1223_122360


namespace petrol_consumption_reduction_l1223_122379

theorem petrol_consumption_reduction (P C : ℝ) (h : P > 0) (h' : C > 0) :
  let new_price := 1.25 * P
  let new_consumption := C * (P / new_price)
  (P * C = new_price * new_consumption) → (new_consumption / C = 0.8) :=
by sorry

end petrol_consumption_reduction_l1223_122379
