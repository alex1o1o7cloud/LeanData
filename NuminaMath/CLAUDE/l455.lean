import Mathlib

namespace race_probability_l455_45588

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

end race_probability_l455_45588


namespace triangle_side_length_l455_45508

theorem triangle_side_length 
  (AB : ℝ) 
  (time_AB time_BC_CA : ℝ) 
  (h1 : AB = 1992)
  (h2 : time_AB = 24)
  (h3 : time_BC_CA = 166)
  : ∃ (BC : ℝ), BC = 6745 :=
by
  sorry

end triangle_side_length_l455_45508


namespace silver_dollar_difference_l455_45515

/-- The number of silver dollars owned by Mr. Chiu -/
def chiu_dollars : ℕ := 56

/-- The number of silver dollars owned by Mr. Phung -/
def phung_dollars : ℕ := chiu_dollars + 16

/-- The number of silver dollars owned by Mr. Ha -/
def ha_dollars : ℕ := 205 - phung_dollars - chiu_dollars

theorem silver_dollar_difference : ha_dollars - phung_dollars = 5 := by
  sorry

end silver_dollar_difference_l455_45515


namespace shirt_price_reduction_l455_45518

theorem shirt_price_reduction (original_price : ℝ) (first_reduction_percent : ℝ) (second_reduction_percent : ℝ) : 
  original_price = 20 →
  first_reduction_percent = 20 →
  second_reduction_percent = 40 →
  (1 - second_reduction_percent / 100) * ((1 - first_reduction_percent / 100) * original_price) = 9.60 := by
sorry

end shirt_price_reduction_l455_45518


namespace m_shaped_area_l455_45525

/-- The area of the M-shaped region formed by folding a 12 × 18 rectangle along its diagonal -/
theorem m_shaped_area (width : ℝ) (height : ℝ) (diagonal : ℝ) (m_area : ℝ) : 
  width = 12 → 
  height = 18 → 
  diagonal = (width^2 + height^2).sqrt →
  m_area = 138 → 
  m_area = (width * height / 2) + 2 * (width * height / 2 - (13 / 36) * (width * height / 2)) :=
by sorry

end m_shaped_area_l455_45525


namespace floor_equation_solution_l455_45579

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊2 * x⌋ - (1 / 2)⌋ = ⌊x + 3⌋ ↔ x ∈ Set.Icc (3.5 : ℝ) (4.5 : ℝ) :=
sorry

end floor_equation_solution_l455_45579


namespace thief_speed_l455_45536

/-- Proves that given the initial conditions, the speed of the thief is 8 km/hr -/
theorem thief_speed (initial_distance : ℝ) (policeman_speed : ℝ) (thief_distance : ℝ)
  (h1 : initial_distance = 175 / 1000) -- Convert 175 meters to kilometers
  (h2 : policeman_speed = 10)
  (h3 : thief_distance = 700 / 1000) -- Convert 700 meters to kilometers
  : ∃ (thief_speed : ℝ), thief_speed = 8 := by
  sorry

end thief_speed_l455_45536


namespace intersection_when_m_neg_one_intersection_empty_iff_m_nonnegative_l455_45599

def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < 1-m}

theorem intersection_when_m_neg_one :
  B (-1) ∩ A = {x | 1 < x ∧ x < 2} := by sorry

theorem intersection_empty_iff_m_nonnegative (m : ℝ) :
  A ∩ B m = ∅ ↔ m ≥ 0 := by sorry

end intersection_when_m_neg_one_intersection_empty_iff_m_nonnegative_l455_45599


namespace cannot_make_24_l455_45500

/-- Represents the four basic arithmetic operations -/
inductive Operation
| Add
| Sub
| Mul
| Div

/-- Applies an operation to two rational numbers -/
def applyOp (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => if b ≠ 0 then a / b else 0

/-- Checks if it's possible to get 24 using the given numbers and operations -/
def canMake24 (a b c d : ℚ) : Prop :=
  ∃ (op1 op2 op3 : Operation),
    (applyOp op3 (applyOp op2 (applyOp op1 a b) c) d = 24) ∨
    (applyOp op3 (applyOp op2 (applyOp op1 a b) d) c = 24) ∨
    (applyOp op3 (applyOp op2 (applyOp op1 a c) b) d = 24) ∨
    (applyOp op3 (applyOp op2 (applyOp op1 a c) d) b = 24) ∨
    (applyOp op3 (applyOp op2 (applyOp op1 a d) b) c = 24) ∨
    (applyOp op3 (applyOp op2 (applyOp op1 a d) c) b = 24)

theorem cannot_make_24 : ¬ canMake24 1 6 8 7 := by
  sorry

end cannot_make_24_l455_45500


namespace fraction_equality_l455_45541

theorem fraction_equality (a b : ℚ) (h : a / b = 2 / 3) : a / (a + b) = 2 / 5 := by
  sorry

end fraction_equality_l455_45541


namespace chebyshev_birth_year_l455_45584

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

end chebyshev_birth_year_l455_45584


namespace log_eight_negative_seven_fourths_l455_45593

theorem log_eight_negative_seven_fourths (x : ℝ) : 
  Real.log x / Real.log 8 = -1.75 → x = 1/64 := by
  sorry

end log_eight_negative_seven_fourths_l455_45593


namespace sum_of_numbers_l455_45586

theorem sum_of_numbers (a b : ℕ+) 
  (hcf : Nat.gcd a b = 5)
  (lcm : Nat.lcm a b = 120)
  (sum_reciprocals : (1 : ℚ) / a + (1 : ℚ) / b = 11 / 120) :
  a + b = 55 := by
  sorry

end sum_of_numbers_l455_45586


namespace quarter_probability_is_3_28_l455_45547

/-- Represents the types of coins in the jar -/
inductive Coin
| Quarter
| Nickel
| Penny
| Dime

/-- The value of each coin type in cents -/
def coin_value : Coin → ℕ
| Coin.Quarter => 25
| Coin.Nickel => 5
| Coin.Penny => 1
| Coin.Dime => 10

/-- The total value of each coin type in cents -/
def total_value : Coin → ℕ
| Coin.Quarter => 1200
| Coin.Nickel => 500
| Coin.Penny => 200
| Coin.Dime => 1000

/-- The number of coins of each type -/
def coin_count (c : Coin) : ℕ := total_value c / coin_value c

/-- The total number of coins in the jar -/
def total_coins : ℕ := coin_count Coin.Quarter + coin_count Coin.Nickel + 
                       coin_count Coin.Penny + coin_count Coin.Dime

/-- The probability of choosing a quarter -/
def quarter_probability : ℚ := coin_count Coin.Quarter / total_coins

theorem quarter_probability_is_3_28 : quarter_probability = 3 / 28 := by
  sorry

end quarter_probability_is_3_28_l455_45547


namespace conjugate_2023_l455_45561

/-- Conjugate point in 2D space -/
def conjugate (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2 + 1, p.1 + 1)

/-- Sequence of conjugate points -/
def conjugateSequence : ℕ → ℝ × ℝ
  | 0 => (2, 2)
  | n + 1 => conjugate (conjugateSequence n)

theorem conjugate_2023 :
  conjugateSequence 2023 = (-2, 0) := by sorry

end conjugate_2023_l455_45561


namespace instrument_probability_l455_45516

theorem instrument_probability (total : ℕ) (at_least_one : ℚ) (two_or_more : ℕ) : 
  total = 800 → 
  at_least_one = 3/5 → 
  two_or_more = 96 → 
  (((at_least_one * total) - two_or_more) : ℚ) / total = 48/100 := by
sorry

end instrument_probability_l455_45516


namespace symmetric_point_exists_l455_45532

def S : Set ℤ := {n : ℤ | ∃ (a b : ℤ), a > 0 ∧ b > 0 ∧ n = 19*a + 85*b}

theorem symmetric_point_exists : 
  ∃ (A : ℝ), ∀ (x y : ℤ), (x + y : ℝ) / 2 = A → 
    (x ∈ S ↔ y ∉ S) :=
sorry

end symmetric_point_exists_l455_45532


namespace simplify_absolute_value_expression_l455_45551

theorem simplify_absolute_value_expression 
  (a b c : ℝ) 
  (ha : |a| + a = 0) 
  (hab : |a * b| = a * b) 
  (hc : |c| - c = 0) : 
  |b| - |a + b| - |c - b| + |a - c| = b := by
  sorry

end simplify_absolute_value_expression_l455_45551


namespace distinct_prime_factors_of_90_l455_45554

theorem distinct_prime_factors_of_90 : Nat.card (Nat.factors 90).toFinset = 3 := by
  sorry

end distinct_prime_factors_of_90_l455_45554


namespace min_distance_to_origin_l455_45566

/-- The minimum distance between a point on the line x - 2y + 2 = 0 and the origin -/
theorem min_distance_to_origin : 
  ∃ (d : ℝ), d = 2 * Real.sqrt 5 / 5 ∧ 
  ∀ (P : ℝ × ℝ), P.1 - 2 * P.2 + 2 = 0 → 
  Real.sqrt (P.1^2 + P.2^2) ≥ d := by
  sorry

end min_distance_to_origin_l455_45566


namespace sqrt_49284_squared_times_3_l455_45505

theorem sqrt_49284_squared_times_3 : (Real.sqrt 49284)^2 * 3 = 147852 := by
  sorry

end sqrt_49284_squared_times_3_l455_45505


namespace five_integers_sum_20_product_420_l455_45507

theorem five_integers_sum_20_product_420 : 
  ∃ (a b c d e : ℕ+), 
    (a.val + b.val + c.val + d.val + e.val = 20) ∧ 
    (a.val * b.val * c.val * d.val * e.val = 420) := by
  sorry

end five_integers_sum_20_product_420_l455_45507


namespace zeta_sum_eight_l455_45546

theorem zeta_sum_eight (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 8)
  (h3 : ζ₁^4 + ζ₂^4 + ζ₃^4 = 26) :
  ζ₁^8 + ζ₂^8 + ζ₃^8 = 219 := by sorry

end zeta_sum_eight_l455_45546


namespace girls_equal_barefoot_children_l455_45503

theorem girls_equal_barefoot_children (B_b G_b G_s : ℕ) :
  B_b = G_s →
  B_b + G_b = G_b + G_s :=
by sorry

end girls_equal_barefoot_children_l455_45503


namespace equidistant_line_from_three_parallel_lines_l455_45502

/-- Given three parallel lines in the form Ax + By = Cᵢ, 
    this theorem states that the line Ax + By = (C₁ + 2C₂ + C₃) / 4 
    is equidistant from all three lines. -/
theorem equidistant_line_from_three_parallel_lines 
  (A B C₁ C₂ C₃ : ℝ) 
  (h_distinct₁ : C₁ ≠ C₂) 
  (h_distinct₂ : C₂ ≠ C₃) 
  (h_distinct₃ : C₁ ≠ C₃) :
  let d₁₂ := |C₂ - C₁| / Real.sqrt (A^2 + B^2)
  let d₂₃ := |C₃ - C₂| / Real.sqrt (A^2 + B^2)
  let d₁₃ := |C₃ - C₁| / Real.sqrt (A^2 + B^2)
  let M := (C₁ + 2*C₂ + C₃) / 4
  ∀ x y, A*x + B*y = M → 
    |A*x + B*y - C₁| / Real.sqrt (A^2 + B^2) = 
    |A*x + B*y - C₂| / Real.sqrt (A^2 + B^2) ∧
    |A*x + B*y - C₂| / Real.sqrt (A^2 + B^2) = 
    |A*x + B*y - C₃| / Real.sqrt (A^2 + B^2) := by
  sorry

end equidistant_line_from_three_parallel_lines_l455_45502


namespace gcd_digits_bound_l455_45596

theorem gcd_digits_bound (a b : ℕ) : 
  (1000000 ≤ a ∧ a < 10000000) →
  (1000000 ≤ b ∧ b < 10000000) →
  (100000000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 1000000000000) →
  Nat.gcd a b < 1000 := by
  sorry

end gcd_digits_bound_l455_45596


namespace log_five_twelve_l455_45562

theorem log_five_twelve (a b : ℝ) (h1 : Real.log 2 = a * Real.log 10) (h2 : Real.log 3 = b * Real.log 10) :
  Real.log 12 / Real.log 5 = (2*a + b) / (1 - a) := by
  sorry

end log_five_twelve_l455_45562


namespace car_speed_problem_l455_45592

theorem car_speed_problem (train_speed_ratio : ℝ) (distance : ℝ) (train_stop_time : ℝ) :
  train_speed_ratio = 1.5 →
  distance = 75 →
  train_stop_time = 12.5 / 60 →
  ∃ (car_speed : ℝ),
    car_speed = 80 ∧
    distance = car_speed * (distance / car_speed) ∧
    distance = (train_speed_ratio * car_speed) * (distance / car_speed - train_stop_time) :=
by sorry

end car_speed_problem_l455_45592


namespace complex_fraction_equality_l455_45511

theorem complex_fraction_equality : Complex.I ^ 2 + Complex.I ^ 3 + Complex.I ^ 4 = (1 / 2 - Complex.I / 2) * (1 - Complex.I) := by
  sorry

end complex_fraction_equality_l455_45511


namespace teddy_bear_shelves_l455_45537

theorem teddy_bear_shelves (total_bears : ℕ) (shelf_capacity : ℕ) (filled_shelves : ℕ) : 
  total_bears = 98 → 
  shelf_capacity = 7 → 
  filled_shelves = total_bears / shelf_capacity →
  filled_shelves = 14 := by
sorry

end teddy_bear_shelves_l455_45537


namespace chocolate_milk_probability_l455_45524

theorem chocolate_milk_probability :
  let n : ℕ := 5  -- number of days
  let k : ℕ := 4  -- number of successful days
  let p : ℚ := 2/3  -- probability of success on each day
  Nat.choose n k * p^k * (1-p)^(n-k) = 80/243 := by
sorry

end chocolate_milk_probability_l455_45524


namespace dot_product_bound_l455_45527

theorem dot_product_bound (a b c m n p : ℝ) 
  (sum_abc : a + b + c = 1) 
  (sum_mnp : m + n + p = 1) : 
  -1 ≤ a*m + b*n + c*p ∧ a*m + b*n + c*p ≤ 1 := by
sorry

end dot_product_bound_l455_45527


namespace expression_evaluation_l455_45522

theorem expression_evaluation : 
  (1.2 : ℝ)^3 - (0.9 : ℝ)^3 / (1.2 : ℝ)^2 + 1.08 + (0.9 : ℝ)^2 = 3.11175 := by
  sorry

end expression_evaluation_l455_45522


namespace range_of_a_l455_45528

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x + Real.exp x - 1 / Real.exp x

theorem range_of_a (a : ℝ) :
  (f (a - 1) + f (2 * a^2) ≤ 0) → (-1 ≤ a ∧ a ≤ 1/2) :=
by sorry

end range_of_a_l455_45528


namespace f_one_half_equals_two_l455_45543

-- Define the function f
noncomputable def f (y : ℝ) : ℝ := (4 : ℝ) ^ y

-- State the theorem
theorem f_one_half_equals_two :
  f (1/2) = 2 :=
sorry

end f_one_half_equals_two_l455_45543


namespace train_length_l455_45589

/-- The length of a train given its speed, platform length, and crossing time -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 96 * (5 / 18) →
  platform_length = 480 →
  crossing_time = 36 →
  ∃ (train_length : ℝ), abs (train_length - 480.12) < 0.01 := by
  sorry

end train_length_l455_45589


namespace units_digit_of_sum_of_cubes_l455_45548

theorem units_digit_of_sum_of_cubes : (24^3 + 42^3) % 10 = 2 := by
  sorry

end units_digit_of_sum_of_cubes_l455_45548


namespace watch_cost_price_proof_l455_45514

/-- The cost price of a watch satisfying certain selling conditions -/
def watch_cost_price : ℝ := 1400

/-- The selling price at a 10% loss -/
def selling_price_loss (cost : ℝ) : ℝ := cost * 0.9

/-- The selling price at a 4% gain -/
def selling_price_gain (cost : ℝ) : ℝ := cost * 1.04

theorem watch_cost_price_proof :
  (selling_price_gain watch_cost_price - selling_price_loss watch_cost_price = 196) ∧
  (watch_cost_price = 1400) := by
  sorry

end watch_cost_price_proof_l455_45514


namespace g_of_two_eq_zero_l455_45585

/-- Given a function g(x) = x^2 - 4 for all real x, prove that g(2) = 0 -/
theorem g_of_two_eq_zero (g : ℝ → ℝ) (h : ∀ x, g x = x^2 - 4) : g 2 = 0 := by
  sorry

end g_of_two_eq_zero_l455_45585


namespace second_month_sale_l455_45580

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

end second_month_sale_l455_45580


namespace derivative_of_x_minus_sin_l455_45519

open Real

theorem derivative_of_x_minus_sin (x : ℝ) : 
  deriv (fun x => x - sin x) x = 1 - cos x := by
sorry

end derivative_of_x_minus_sin_l455_45519


namespace movie_of_the_year_threshold_l455_45570

theorem movie_of_the_year_threshold (total_members : ℕ) (threshold_fraction : ℚ) : 
  total_members = 795 →
  threshold_fraction = 1/4 →
  ∃ n : ℕ, n ≥ total_members * threshold_fraction ∧ 
    ∀ m : ℕ, m ≥ total_members * threshold_fraction → m ≥ n :=
by sorry

end movie_of_the_year_threshold_l455_45570


namespace yellow_apples_probability_l455_45504

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The probability of an event -/
def probability (favorable_outcomes total_outcomes : ℕ) : ℚ :=
  favorable_outcomes / total_outcomes

theorem yellow_apples_probability :
  let total_apples : ℕ := 10
  let yellow_apples : ℕ := 5
  let selected_apples : ℕ := 3
  probability (choose yellow_apples selected_apples) (choose total_apples selected_apples) = 1 / 12 := by
  sorry

end yellow_apples_probability_l455_45504


namespace solve_equations_l455_45521

/-- Solutions to the quadratic equation x^2 - 6x + 3 = 0 -/
def solutions_eq1 : Set ℝ := {3 + Real.sqrt 6, 3 - Real.sqrt 6}

/-- Solutions to the equation x(x-2) = x-2 -/
def solutions_eq2 : Set ℝ := {2, 1}

theorem solve_equations :
  (∀ x ∈ solutions_eq1, x^2 - 6*x + 3 = 0) ∧
  (∀ x ∈ solutions_eq2, x*(x-2) = x-2) :=
by sorry

end solve_equations_l455_45521


namespace pencil_buyers_difference_l455_45575

theorem pencil_buyers_difference (price : ℕ) 
  (h1 : price > 0)
  (h2 : 234 % price = 0)
  (h3 : 325 % price = 0) :
  325 / price - 234 / price = 7 := by
  sorry

end pencil_buyers_difference_l455_45575


namespace at_least_one_zero_l455_45583

theorem at_least_one_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^9 + b^9) * (b^9 + c^9) * (c^9 + a^9) = (a * b * c)^9) :
  a = 0 ∨ b = 0 ∨ c = 0 := by
sorry

end at_least_one_zero_l455_45583


namespace rational_absolute_value_and_negative_numbers_l455_45517

theorem rational_absolute_value_and_negative_numbers :
  (∀ x : ℚ, |x| ≥ 0 ∧ (|x| = 0 ↔ x = 0)) ∧
  (∀ x : ℝ, -x > x → x < 0) := by
  sorry

end rational_absolute_value_and_negative_numbers_l455_45517


namespace min_value_inequality_l455_45563

theorem min_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + 3 * c) + b / (8 * c + 4 * a) + 9 * c / (3 * a + 2 * b) ≥ 47 / 48 := by
  sorry

end min_value_inequality_l455_45563


namespace max_hands_for_54_coincidences_l455_45573

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

end max_hands_for_54_coincidences_l455_45573


namespace line_parabola_intersection_l455_45510

/-- The line x = k intersects the parabola x = -3y^2 + 2y + 7 at exactly one point if and only if k = 22/3 -/
theorem line_parabola_intersection (k : ℝ) : 
  (∃! y : ℝ, k = -3 * y^2 + 2 * y + 7) ↔ k = 22/3 := by
sorry

end line_parabola_intersection_l455_45510


namespace square_sum_from_product_and_sum_l455_45556

theorem square_sum_from_product_and_sum (r s : ℝ) 
  (h1 : r * s = 24) 
  (h2 : r + s = 10) : 
  r^2 + s^2 = 52 := by
sorry

end square_sum_from_product_and_sum_l455_45556


namespace difference_of_squares_403_397_l455_45523

theorem difference_of_squares_403_397 : 403^2 - 397^2 = 4800 := by
  sorry

end difference_of_squares_403_397_l455_45523


namespace initial_deposit_proof_l455_45572

/-- Proves that the initial deposit is correct given the total savings goal,
    saving period, and weekly saving amount. -/
theorem initial_deposit_proof (total_goal : ℕ) (weeks : ℕ) (weekly_saving : ℕ) 
    (h1 : total_goal = 500)
    (h2 : weeks = 19)
    (h3 : weekly_saving = 17) : 
  total_goal - (weeks * weekly_saving) = 177 := by
  sorry

end initial_deposit_proof_l455_45572


namespace function_fixed_point_l455_45550

def iterateF (f : ℝ → ℝ) : ℕ → (ℝ → ℝ)
  | 0 => id
  | n + 1 => f ∘ (iterateF f n)

theorem function_fixed_point
  (f : ℝ → ℝ)
  (hf : Continuous f)
  (h : ∀ x : ℝ, ∃ n : ℕ, iterateF f n x = 1) :
  f 1 = 1 := by
  sorry

end function_fixed_point_l455_45550


namespace cannot_form_70_cents_l455_45501

/-- Represents the types of coins available in the piggy bank -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents -/
def coinValue (c : Coin) : Nat :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- Represents a combination of coins -/
def CoinCombination := List Coin

/-- Calculates the total value of a coin combination in cents -/
def totalValue (comb : CoinCombination) : Nat :=
  comb.map coinValue |>.sum

/-- Predicate to check if a coin combination has exactly six coins -/
def hasSixCoins (comb : CoinCombination) : Prop :=
  comb.length = 6

theorem cannot_form_70_cents :
  ¬∃ (comb : CoinCombination), hasSixCoins comb ∧ totalValue comb = 70 :=
sorry

end cannot_form_70_cents_l455_45501


namespace max_volume_box_l455_45565

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


end max_volume_box_l455_45565


namespace rope_length_theorem_l455_45531

/-- Represents a rope that can be folded in a specific manner. -/
structure Rope where
  /-- The distance between points (2) and (3) in the final folding. -/
  distance_2_3 : ℝ
  /-- Assertion that the distance between points (2) and (3) is positive. -/
  distance_positive : distance_2_3 > 0

/-- Calculates the total length of the rope based on its properties. -/
def total_length (rope : Rope) : ℝ :=
  6 * rope.distance_2_3

/-- Theorem stating that for a rope with distance between points (2) and (3) equal to 20,
    the total length is 120. -/
theorem rope_length_theorem (rope : Rope) (h : rope.distance_2_3 = 20) :
  total_length rope = 120 := by
  sorry

end rope_length_theorem_l455_45531


namespace system_solution_exists_l455_45597

theorem system_solution_exists (a b : ℤ) (h1 : 5 * a ≥ 7 * b) (h2 : 7 * b ≥ 0) :
  ∃ (x y z u : ℕ), x + 2 * y + 3 * z + 7 * u = a ∧ y + 2 * z + 5 * u = b := by
  sorry

end system_solution_exists_l455_45597


namespace base_number_proof_l455_45568

theorem base_number_proof (x n : ℕ) (h1 : 4 * x^(2*n) = 4^26) (h2 : n = 25) : x = 2 := by
  sorry

end base_number_proof_l455_45568


namespace total_budget_allocation_l455_45534

def budget_groceries : ℝ := 0.6
def budget_eating_out : ℝ := 0.2
def budget_transportation : ℝ := 0.1
def budget_rent : ℝ := 0.05
def budget_utilities : ℝ := 0.05

theorem total_budget_allocation :
  budget_groceries + budget_eating_out + budget_transportation + budget_rent + budget_utilities = 1 := by
  sorry

end total_budget_allocation_l455_45534


namespace factorization_of_18x_squared_minus_8_l455_45567

theorem factorization_of_18x_squared_minus_8 (x : ℝ) : 18 * x^2 - 8 = 2 * (3*x + 2) * (3*x - 2) := by
  sorry

end factorization_of_18x_squared_minus_8_l455_45567


namespace sine_transformation_l455_45577

theorem sine_transformation (x : ℝ) : 
  Real.sin (2 * (x + π/4) + π/6) = Real.sin (2*x + 2*π/3) := by
  sorry

end sine_transformation_l455_45577


namespace fraction_simplification_l455_45529

theorem fraction_simplification (x y : ℝ) (h : x ≠ y) :
  (x^6 - y^6) / (x^3 - y^3) = x^3 + y^3 := by
  sorry

end fraction_simplification_l455_45529


namespace complex_equation_solution_l455_45545

theorem complex_equation_solution (i : ℂ) (m : ℝ) : 
  i * i = -1 → (1 - m * i) / (i^3) = 1 + i → m = 1 := by
sorry

end complex_equation_solution_l455_45545


namespace sequence_ratio_l455_45553

def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def arithmetic_square_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n * (2 * a₁^2 + (n - 1) * d^2 + (n - 1) * d * a₁)) / 2

theorem sequence_ratio :
  let n := (38 - 4) / 2 + 1
  arithmetic_sum 4 2 n / arithmetic_square_sum 3 3 n = 2 / 15 := by
  sorry

end sequence_ratio_l455_45553


namespace area_of_overlapping_rotated_squares_exists_l455_45555

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

end area_of_overlapping_rotated_squares_exists_l455_45555


namespace no_valid_sum_of_consecutive_integers_l455_45576

def sum_of_consecutive_integers (k : ℕ) : ℕ := 150 * k + 11175

def given_integers : List ℕ := [1625999850, 2344293800, 3578726150, 4691196050, 5815552000]

theorem no_valid_sum_of_consecutive_integers : 
  ∀ n ∈ given_integers, ¬ ∃ k : ℕ, sum_of_consecutive_integers k = n :=
by sorry

end no_valid_sum_of_consecutive_integers_l455_45576


namespace line_through_point_parallel_to_given_line_l455_45506

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in the form ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def areParallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The main theorem to prove -/
theorem line_through_point_parallel_to_given_line :
  let A : Point2D := ⟨2, 3⟩
  let givenLine : Line2D := ⟨2, 4, -3⟩
  let resultLine : Line2D := ⟨1, 2, -8⟩
  pointOnLine A resultLine ∧ areParallel resultLine givenLine := by sorry

end line_through_point_parallel_to_given_line_l455_45506


namespace distance_walked_l455_45542

theorem distance_walked (x t : ℝ) 
  (h1 : (x + 1) * (3/4 * t) = x * t) 
  (h2 : (x - 1) * (t + 3) = x * t) : 
  x * t = 18 := by
  sorry

end distance_walked_l455_45542


namespace teacup_cost_function_l455_45574

-- Define the cost of a single teacup
def teacup_cost : ℚ := 2.5

-- Define the function for the total cost
def total_cost (x : ℕ+) : ℚ := x.val * teacup_cost

-- Theorem statement
theorem teacup_cost_function (x : ℕ+) (y : ℚ) :
  y = total_cost x ↔ y = 2.5 * x.val := by sorry

end teacup_cost_function_l455_45574


namespace existence_of_non_one_start_l455_45526

def begins_with_same_digit (x : ℕ) : Prop :=
  ∃ d : ℕ, d ≠ 0 ∧ d < 10 ∧
  ∀ n : ℕ, n ≤ 2015 →
    ∃ k : ℕ, x^n = d * 10^k + (x^n - d * 10^k) ∧
             d * 10^k ≤ x^n ∧
             x^n < (d + 1) * 10^k

theorem existence_of_non_one_start :
  ∃ x : ℕ, begins_with_same_digit x ∧
    ∃ d : ℕ, d ≠ 1 ∧ d < 10 ∧
    ∀ n : ℕ, n ≤ 2015 →
      ∃ k : ℕ, x^n = d * 10^k + (x^n - d * 10^k) ∧
               d * 10^k ≤ x^n ∧
               x^n < (d + 1) * 10^k :=
by sorry

end existence_of_non_one_start_l455_45526


namespace monthly_salary_calculation_l455_45557

/-- Proves that a person's monthly salary is 6000 given the specified savings conditions -/
theorem monthly_salary_calculation (salary : ℝ) : 
  (salary * 0.2 = salary - (salary * 0.8 * 1.2 + 240)) → salary = 6000 := by
  sorry

end monthly_salary_calculation_l455_45557


namespace andre_carl_speed_ratio_l455_45582

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

end andre_carl_speed_ratio_l455_45582


namespace system_consistency_l455_45512

/-- The system of equations is consistent if and only if a is 0, -2, or 54 -/
theorem system_consistency (x a : ℝ) : 
  (∃ x, (10 * x^2 + x - a - 11 = 0) ∧ (4 * x^2 + (a + 4) * x - 3 * a - 8 = 0)) ↔ 
  (a = 0 ∨ a = -2 ∨ a = 54) :=
by sorry

end system_consistency_l455_45512


namespace point_on_y_axis_l455_45520

/-- A point P with coordinates (m+2, m+1) lies on the y-axis if and only if its coordinates are (0, -1) -/
theorem point_on_y_axis (m : ℝ) : 
  (m + 2 = 0 ∧ ∃ y, (0, y) = (m + 2, m + 1)) ↔ (0, -1) = (m + 2, m + 1) :=
by sorry

end point_on_y_axis_l455_45520


namespace no_nonneg_integer_solution_l455_45559

theorem no_nonneg_integer_solution (a b : ℕ) (ha : a ≠ b) :
  let d := Nat.gcd a b
  let a' := a / d
  let b' := b / d
  ∀ n : ℕ, (∀ x y : ℕ, a * x + b * y ≠ n) ↔ n = d * (a' * b' - a' - b') := by
  sorry

end no_nonneg_integer_solution_l455_45559


namespace sandy_correct_sums_l455_45564

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

end sandy_correct_sums_l455_45564


namespace walters_age_2009_l455_45595

theorem walters_age_2009 (walter_age_2004 : ℝ) (grandmother_age_2004 : ℝ) : 
  walter_age_2004 = grandmother_age_2004 / 3 →
  (2004 - walter_age_2004) + (2004 - grandmother_age_2004) = 4018 →
  walter_age_2004 + 5 = 7.5 := by
  sorry

end walters_age_2009_l455_45595


namespace basketball_tournament_equation_l455_45513

/-- The number of games played in a basketball tournament -/
def num_games : ℕ := 28

/-- Theorem: In a basketball tournament with x teams, where each pair of teams plays exactly one game,
    and a total of 28 games are played, the equation ½x(x-1) = 28 holds true. -/
theorem basketball_tournament_equation (x : ℕ) (h : x > 1) :
  (x * (x - 1)) / 2 = num_games :=
sorry

end basketball_tournament_equation_l455_45513


namespace no_solution_exists_l455_45509

theorem no_solution_exists : ¬∃ (x y z : ℕ+), x^(x.val) + y^(y.val) = 9^(z.val) := by
  sorry

end no_solution_exists_l455_45509


namespace min_value_of_trig_function_l455_45530

theorem min_value_of_trig_function :
  ∃ (min : ℝ), min = -Real.sqrt 2 - 1 ∧
  ∀ (x : ℝ), 2 * Real.sin x * Real.cos x - 2 * (Real.sin x)^2 ≥ min :=
sorry

end min_value_of_trig_function_l455_45530


namespace sqrt_floor_equality_l455_45538

theorem sqrt_floor_equality (n : ℕ) :
  ⌊Real.sqrt (4 * n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ ∧
  ⌊Real.sqrt (4 * n + 2)⌋ = ⌊Real.sqrt (4 * n + 3)⌋ :=
by sorry

end sqrt_floor_equality_l455_45538


namespace unattainable_y_value_l455_45535

theorem unattainable_y_value (x : ℝ) :
  x ≠ -2/3 → (x - 3) / (3 * x + 2) ≠ 1/3 := by
  sorry

end unattainable_y_value_l455_45535


namespace cooking_probability_l455_45540

-- Define the set of courses
def Courses := Finset.range 4

-- Define the probability of selecting a specific course
def prob_select (course : Courses) : ℚ :=
  1 / Courses.card

-- State the theorem
theorem cooking_probability :
  ∃ (cooking : Courses), prob_select cooking = 1 / 4 :=
sorry

end cooking_probability_l455_45540


namespace count_numbers_with_three_700_l455_45539

def contains_three (n : Nat) : Bool :=
  n.repr.any (· = '3')

def count_numbers_with_three (upper_bound : Nat) : Nat :=
  (List.range upper_bound).filter contains_three |>.length

theorem count_numbers_with_three_700 :
  count_numbers_with_three 700 = 214 := by
  sorry

end count_numbers_with_three_700_l455_45539


namespace larger_number_proof_l455_45549

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 6 * S + 15) :
  L = 1635 := by
  sorry

end larger_number_proof_l455_45549


namespace megan_markers_count_l455_45594

/-- The total number of markers Megan has after receiving more from Robert -/
def total_markers (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Megan's total markers is the sum of her initial markers and those received from Robert -/
theorem megan_markers_count (initial : ℕ) (received : ℕ) :
  total_markers initial received = initial + received :=
by
  sorry

end megan_markers_count_l455_45594


namespace mp3_price_reduction_l455_45591

/-- Given an item with a sale price of 112 after a 20% reduction,
    prove that its price after a 30% reduction would be 98. -/
theorem mp3_price_reduction (original_price : ℝ) : 
  (original_price * 0.8 = 112) → (original_price * 0.7 = 98) := by
  sorry

end mp3_price_reduction_l455_45591


namespace range_of_m_l455_45587

def f (x : ℝ) := -x^2 + 4*x

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc m 4, f x ∈ Set.Icc 0 4) ∧
  (∀ y ∈ Set.Icc 0 4, ∃ x ∈ Set.Icc m 4, f x = y) →
  m ∈ Set.Icc 0 2 :=
sorry

end range_of_m_l455_45587


namespace rational_sqrt_one_minus_ab_l455_45544

theorem rational_sqrt_one_minus_ab (a b : ℚ) 
  (h : a^3 * b + a * b^3 + 2 * a^2 * b^2 + 2 * a + 2 * b + 1 = 0) : 
  ∃ q : ℚ, q^2 = 1 - a * b := by
  sorry

end rational_sqrt_one_minus_ab_l455_45544


namespace carls_cupcake_goal_l455_45552

/-- Given Carl's cupcake selling goal and payment obligation, prove the number of cupcakes he must sell per day. -/
theorem carls_cupcake_goal (goal : ℕ) (days : ℕ) (payment : ℕ) (cupcakes_per_day : ℕ) 
    (h1 : goal = 96) 
    (h2 : days = 2) 
    (h3 : payment = 24) 
    (h4 : cupcakes_per_day * days = goal + payment) : 
  cupcakes_per_day = 60 := by
  sorry

#check carls_cupcake_goal

end carls_cupcake_goal_l455_45552


namespace stability_of_nonlinear_eq_l455_45569

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

end stability_of_nonlinear_eq_l455_45569


namespace cauliflower_increase_40401_l455_45590

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

end cauliflower_increase_40401_l455_45590


namespace ofelias_to_rileys_mistakes_ratio_l455_45533

theorem ofelias_to_rileys_mistakes_ratio 
  (total_questions : ℕ) 
  (rileys_mistakes : ℕ) 
  (team_incorrect : ℕ) 
  (h1 : total_questions = 35)
  (h2 : rileys_mistakes = 3)
  (h3 : team_incorrect = 17) :
  (team_incorrect - rileys_mistakes) / rileys_mistakes = 14 / 3 := by
sorry

end ofelias_to_rileys_mistakes_ratio_l455_45533


namespace tea_consumption_l455_45558

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

end tea_consumption_l455_45558


namespace hemisphere_surface_area_l455_45571

theorem hemisphere_surface_area (V : ℝ) (h : V = (500 / 3) * Real.pi) :
  ∃ (r : ℝ), V = (2 / 3) * Real.pi * r^3 ∧
             (2 * Real.pi * r^2 + Real.pi * r^2) = 3 * Real.pi * 250^(2/3) := by
  sorry

end hemisphere_surface_area_l455_45571


namespace inscribed_square_side_length_l455_45578

/-- Triangle ABC with inscribed square PQRS -/
structure InscribedSquareTriangle where
  /-- Side length of AB -/
  ab : ℝ
  /-- Side length of BC -/
  bc : ℝ
  /-- Side length of CA -/
  ca : ℝ
  /-- Point P lies on BC -/
  p_on_bc : Bool
  /-- Point R lies on BC -/
  r_on_bc : Bool
  /-- Point Q lies on CA -/
  q_on_ca : Bool
  /-- Point S lies on AB -/
  s_on_ab : Bool

/-- The side length of the inscribed square PQRS -/
def squareSideLength (t : InscribedSquareTriangle) : ℝ := sorry

/-- Theorem: The side length of the inscribed square is 42 -/
theorem inscribed_square_side_length 
  (t : InscribedSquareTriangle) 
  (h1 : t.ab = 13) 
  (h2 : t.bc = 14) 
  (h3 : t.ca = 15) 
  (h4 : t.p_on_bc = true) 
  (h5 : t.r_on_bc = true) 
  (h6 : t.q_on_ca = true) 
  (h7 : t.s_on_ab = true) : 
  squareSideLength t = 42 := by sorry

end inscribed_square_side_length_l455_45578


namespace passes_through_neg1_0_two_a_plus_c_positive_roots_between_neg3_and_1_l455_45581

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

end passes_through_neg1_0_two_a_plus_c_positive_roots_between_neg3_and_1_l455_45581


namespace frequency_in_interval_l455_45598

def sample_capacity : ℕ := 100

def group_frequencies : List ℕ := [12, 13, 24, 15, 16, 13, 7]

def interval_sum : ℕ := 12 + 13 + 24 + 15

theorem frequency_in_interval :
  (interval_sum : ℚ) / sample_capacity = 0.64 := by sorry

end frequency_in_interval_l455_45598


namespace johns_tax_rate_l455_45560

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

end johns_tax_rate_l455_45560
