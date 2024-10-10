import Mathlib

namespace divisibility_condition_l453_45373

theorem divisibility_condition (n p : ℕ+) (h_prime : Nat.Prime p) (h_bound : n ≤ 2 * p) :
  (((p : ℤ) - 1) ^ (n : ℕ) + 1) % (n ^ (p - 1 : ℕ)) = 0 ↔ 
  ((n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) ∨ (n = 1)) := by
  sorry

end divisibility_condition_l453_45373


namespace on_time_passengers_l453_45355

theorem on_time_passengers (total : ℕ) (late : ℕ) (on_time : ℕ) : 
  total = 14720 → late = 213 → on_time = total - late → on_time = 14507 := by
  sorry

end on_time_passengers_l453_45355


namespace weekend_reading_l453_45372

/-- The number of pages Bekah needs to read for history class -/
def total_pages : ℕ := 408

/-- The number of days left to finish reading -/
def days_left : ℕ := 5

/-- The number of pages Bekah needs to read each day for the remaining days -/
def pages_per_day : ℕ := 59

/-- The number of pages Bekah read over the weekend -/
def pages_read_weekend : ℕ := total_pages - (days_left * pages_per_day)

theorem weekend_reading :
  pages_read_weekend = 113 := by sorry

end weekend_reading_l453_45372


namespace anne_distance_l453_45341

/-- Given a speed and time, calculates the distance traveled -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Anne's distance traveled is 6 miles -/
theorem anne_distance :
  let speed : ℝ := 2  -- miles per hour
  let time : ℝ := 3   -- hours
  distance speed time = 6 := by sorry

end anne_distance_l453_45341


namespace rectangle_to_square_side_half_length_l453_45306

/-- Given a rectangle with dimensions 7 × 21 that is cut into two congruent shapes
    and rearranged into a square, half the length of a side of the resulting square
    is equal to 7√3/2. -/
theorem rectangle_to_square_side_half_length :
  let rectangle_length : ℝ := 21
  let rectangle_width : ℝ := 7
  let rectangle_area := rectangle_length * rectangle_width
  let square_side := Real.sqrt rectangle_area
  let y := square_side / 2
  y = 7 * Real.sqrt 3 / 2 := by sorry

end rectangle_to_square_side_half_length_l453_45306


namespace unique_function_theorem_l453_45317

-- Define the function type
def IntFunction := ℤ → ℤ

-- Define the property that the function must satisfy
def SatisfiesEquation (f : IntFunction) : Prop :=
  ∀ m n : ℤ, f (f m + n) + f m = f n + f (3 * m) + 2014

-- State the theorem
theorem unique_function_theorem :
  ∀ f : IntFunction, SatisfiesEquation f → ∀ n : ℤ, f n = 2 * n + 1007 := by
  sorry

end unique_function_theorem_l453_45317


namespace simplify_sqrt_expression_l453_45336

theorem simplify_sqrt_expression : 
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end simplify_sqrt_expression_l453_45336


namespace initial_quarters_count_l453_45326

-- Define the problem parameters
def cents_left : ℕ := 300
def cents_spent : ℕ := 50
def cents_per_quarter : ℕ := 25

-- Theorem statement
theorem initial_quarters_count : 
  (cents_left + cents_spent) / cents_per_quarter = 14 := by
  sorry

end initial_quarters_count_l453_45326


namespace three_slice_toast_l453_45396

/-- Represents a slice of bread with two sides -/
structure Bread :=
  (side1 : Bool)
  (side2 : Bool)

/-- Represents the state of the toaster -/
structure ToasterState :=
  (slot1 : Option Bread)
  (slot2 : Option Bread)

/-- Represents the toasting process -/
def toast (initial : List Bread) (time : Nat) : List Bread → Prop :=
  sorry

theorem three_slice_toast :
  ∀ (initial : List Bread),
    initial.length = 3 →
    ∀ (b : Bread), b ∈ initial → ¬b.side1 ∧ ¬b.side2 →
    ∃ (final : List Bread),
      toast initial 3 final ∧
      final.length = 3 ∧
      ∀ (b : Bread), b ∈ final → b.side1 ∧ b.side2 :=
by sorry

end three_slice_toast_l453_45396


namespace base_10_satisfies_equation_l453_45361

def base_x_addition (x : ℕ) (a b c : ℕ) : Prop :=
  a + b = c

def to_base_10 (x : ℕ) (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d1 * x^3 + d2 * x^2 + d3 * x + d4

theorem base_10_satisfies_equation : 
  ∃ x : ℕ, x > 1 ∧ base_x_addition x 
    (to_base_10 x 8374) 
    (to_base_10 x 6250) 
    (to_base_10 x 15024) :=
by
  sorry

end base_10_satisfies_equation_l453_45361


namespace expression_simplification_l453_45380

theorem expression_simplification (a b c : ℚ) 
  (ha : a = 1/3) (hb : b = 1/2) (hc : c = 1) : 
  (2*a^2 - b) - (a^2 - 4*b) - (b + c) = 1/9 := by
  sorry

end expression_simplification_l453_45380


namespace floor_plus_self_eq_seventeen_fourths_l453_45390

theorem floor_plus_self_eq_seventeen_fourths :
  ∃! (y : ℚ), ⌊y⌋ + y = 17 / 4 :=
by sorry

end floor_plus_self_eq_seventeen_fourths_l453_45390


namespace intersection_of_A_and_B_l453_45330

def A : Set Int := {-1, 0, 1, 2}
def B : Set Int := {-2, 0, 2, 4}

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by
  sorry

end intersection_of_A_and_B_l453_45330


namespace angle_D_is_120_l453_45366

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ)
  (sum_360 : A + B + C + D = 360)
  (all_positive : A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0)

-- Define the ratio condition
def ratio_condition (q : Quadrilateral) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ q.A = k ∧ q.B = 2*k ∧ q.C = k ∧ q.D = 2*k

-- Theorem statement
theorem angle_D_is_120 (q : Quadrilateral) (h : ratio_condition q) : q.D = 120 := by
  sorry

end angle_D_is_120_l453_45366


namespace binary_multiplication_division_l453_45334

/-- Convert a binary string to a natural number -/
def binary_to_nat (s : String) : ℕ :=
  s.foldl (fun acc c => 2 * acc + (if c = '1' then 1 else 0)) 0

/-- Convert a natural number to a binary string -/
def nat_to_binary (n : ℕ) : String :=
  if n = 0 then "0" else
  let rec aux (m : ℕ) : String :=
    if m = 0 then "" else aux (m / 2) ++ (if m % 2 = 1 then "1" else "0")
  aux n

theorem binary_multiplication_division :
  let a := binary_to_nat "11100"
  let b := binary_to_nat "11010"
  let c := binary_to_nat "100"
  nat_to_binary ((a * b) / c) = "10100110" := by
  sorry

end binary_multiplication_division_l453_45334


namespace absolute_value_simplification_l453_45351

theorem absolute_value_simplification : |(-4^2 + (5 - 2))| = 13 := by
  sorry

end absolute_value_simplification_l453_45351


namespace yellow_marbles_count_l453_45305

theorem yellow_marbles_count (total red blue green yellow : ℕ) : 
  total = 110 →
  red = 8 →
  blue = 4 * red →
  green = 2 * blue →
  yellow = total - (red + blue + green) →
  yellow = 6 := by
sorry

end yellow_marbles_count_l453_45305


namespace x_plus_q_in_terms_of_q_l453_45386

theorem x_plus_q_in_terms_of_q (x q : ℝ) (h1 : |x - 5| = q) (h2 : x > 5) : x + q = 2*q + 5 := by
  sorry

end x_plus_q_in_terms_of_q_l453_45386


namespace world_cup_matches_l453_45339

/-- The number of matches played in a group of teams where each pair plays twice -/
def number_of_matches (n : ℕ) : ℕ :=
  n * (n - 1)

/-- Theorem: In a group of 6 teams where each pair plays twice, 30 matches are played -/
theorem world_cup_matches : number_of_matches 6 = 30 := by
  sorry

end world_cup_matches_l453_45339


namespace sequence_sum_l453_45356

theorem sequence_sum (a : ℕ → ℝ) (a_pos : ∀ n, a n > 0) 
  (h1 : a 1 = 2) (h2 : a 2 = 3) (h3 : a 3 = 4) (h5 : a 5 = 6) :
  ∃ (a_val t : ℝ), a_val > 0 ∧ t > 0 ∧ a_val = a 5 ∧ t = a_val^2 - 1 ∧ a_val + t = 41 := by
  sorry

end sequence_sum_l453_45356


namespace stewart_farm_ratio_l453_45318

theorem stewart_farm_ratio : ∀ (num_sheep num_horses : ℕ) (horse_food_per_day total_horse_food : ℕ),
  num_sheep = 24 →
  horse_food_per_day = 230 →
  total_horse_food = 12880 →
  num_horses * horse_food_per_day = total_horse_food →
  num_sheep * 7 = num_horses * 3 :=
by sorry

end stewart_farm_ratio_l453_45318


namespace modulus_of_z_is_sqrt_two_l453_45333

/-- Given a complex number z defined as z = 2/(1+i) + (1+i)^2, prove that its modulus |z| is equal to √2 -/
theorem modulus_of_z_is_sqrt_two : 
  let z : ℂ := 2 / (1 + Complex.I) + (1 + Complex.I)^2
  ‖z‖ = Real.sqrt 2 := by sorry

end modulus_of_z_is_sqrt_two_l453_45333


namespace complex_number_equality_l453_45367

theorem complex_number_equality : ∀ (i : ℂ), i^2 = -1 →
  (2 * i) / (2 + i) = 2/5 + 4/5 * i := by
  sorry

end complex_number_equality_l453_45367


namespace babylonian_square_58_l453_45320

/-- Represents the Babylonian method of expressing squares --/
def babylonian_square (n : ℕ) : ℕ × ℕ :=
  let square := n * n
  let quotient := square / 60
  let remainder := square % 60
  if remainder = 0 then (quotient, 60) else (quotient, remainder)

/-- The theorem to be proved --/
theorem babylonian_square_58 : babylonian_square 58 = (56, 4) := by
  sorry

#eval babylonian_square 58  -- To check the result

end babylonian_square_58_l453_45320


namespace drivers_distance_difference_l453_45307

/-- Calculates the difference in distance traveled between two drivers meeting on a highway --/
theorem drivers_distance_difference
  (initial_distance : ℝ)
  (speed_a : ℝ)
  (speed_b : ℝ)
  (delay : ℝ)
  (h1 : initial_distance = 787)
  (h2 : speed_a = 90)
  (h3 : speed_b = 80)
  (h4 : delay = 1) :
  let remaining_distance := initial_distance - speed_a * delay
  let relative_speed := speed_a + speed_b
  let meeting_time := remaining_distance / relative_speed
  let distance_a := speed_a * (meeting_time + delay)
  let distance_b := speed_b * meeting_time
  distance_a - distance_b = 131 := by sorry

end drivers_distance_difference_l453_45307


namespace conditional_probability_l453_45362

/-- Represents the probability space for the household appliance problem -/
structure ApplianceProbability where
  /-- Probability that the appliance lasts for 3 years -/
  three_years : ℝ
  /-- Probability that the appliance lasts for 4 years -/
  four_years : ℝ
  /-- Assumption that the probability of lasting 3 years is 0.8 -/
  three_years_prob : three_years = 0.8
  /-- Assumption that the probability of lasting 4 years is 0.4 -/
  four_years_prob : four_years = 0.4
  /-- Assumption that probabilities are between 0 and 1 -/
  prob_bounds : 0 ≤ three_years ∧ three_years ≤ 1 ∧ 0 ≤ four_years ∧ four_years ≤ 1

/-- The main theorem stating the conditional probability -/
theorem conditional_probability (ap : ApplianceProbability) :
  (ap.four_years / ap.three_years) = 0.5 := by
  sorry


end conditional_probability_l453_45362


namespace parallel_lines_m_value_l453_45371

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {a b c d : ℝ} : 
  (∀ x y, a*x + b*y + c = 0 ↔ d*x - y = 0) → a/b = -d

/-- The value of m for parallel lines -/
theorem parallel_lines_m_value : 
  (∀ x y, x + 2*y - 1 = 0 ↔ m*x - y = 0) → m = -1/2 := by sorry

end parallel_lines_m_value_l453_45371


namespace cubic_expression_value_l453_45347

theorem cubic_expression_value (α : ℝ) (h1 : α > 0) (h2 : α^2 - 8*α - 5 = 0) :
  α^3 - 7*α^2 - 13*α + 6 = 11 := by
sorry

end cubic_expression_value_l453_45347


namespace range_of_m_l453_45323

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + y = x * y) :
  (∃ m : ℝ, x + y / 4 < m^2 + 3 * m) ↔ ∃ m : ℝ, m < -4 ∨ m > 1 := by
  sorry

end range_of_m_l453_45323


namespace arrangement_plans_count_l453_45358

/-- The number of ways to arrange teachers into classes -/
def arrangement_count (n m : ℕ) : ℕ :=
  -- n: total number of teachers
  -- m: number of classes
  sorry

/-- Xiao Li must be in class one -/
def xiao_li_in_class_one : Prop :=
  sorry

/-- Each class must have at least one teacher -/
def at_least_one_teacher_per_class : Prop :=
  sorry

/-- The main theorem stating the number of arrangement plans -/
theorem arrangement_plans_count :
  arrangement_count 5 3 = 50 ∧ xiao_li_in_class_one ∧ at_least_one_teacher_per_class :=
sorry

end arrangement_plans_count_l453_45358


namespace smallest_ab_value_l453_45322

theorem smallest_ab_value (a b : ℤ) (h : (a : ℚ) / 2 + (b : ℚ) / 1009 = 1 / 2018) :
  ∃ (a₀ b₀ : ℤ), (a₀ : ℚ) / 2 + (b₀ : ℚ) / 1009 = 1 / 2018 ∧ |a₀ * b₀| = 504 ∧
    ∀ (a' b' : ℤ), (a' : ℚ) / 2 + (b' : ℚ) / 1009 = 1 / 2018 → |a' * b'| ≥ 504 :=
by sorry

end smallest_ab_value_l453_45322


namespace seventeenth_term_is_two_l453_45311

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  sum : ℕ → ℝ -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, sum n = n * (a 1 + a n) / 2
  sum_13 : sum 13 = 78
  sum_7_12 : a 7 + a 12 = 10

/-- The 17th term of the arithmetic sequence is 2 -/
theorem seventeenth_term_is_two (seq : ArithmeticSequence) : seq.a 17 = 2 := by
  sorry

end seventeenth_term_is_two_l453_45311


namespace mlb_game_hits_and_misses_l453_45395

theorem mlb_game_hits_and_misses (hits misses : ℕ) : 
  misses = 3 * hits → 
  misses = 50 → 
  hits + misses = 200 := by
  sorry

end mlb_game_hits_and_misses_l453_45395


namespace scarf_cost_l453_45349

theorem scarf_cost (initial_amount : ℕ) (toy_car_cost : ℕ) (num_toy_cars : ℕ) (beanie_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 53 →
  toy_car_cost = 11 →
  num_toy_cars = 2 →
  beanie_cost = 14 →
  remaining_amount = 7 →
  initial_amount - (num_toy_cars * toy_car_cost + beanie_cost + remaining_amount) = 10 := by
sorry

end scarf_cost_l453_45349


namespace equipment_value_after_three_years_l453_45388

/-- The value of equipment after n years, given an initial value and annual depreciation rate. -/
def equipment_value (initial_value : ℝ) (depreciation_rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 - depreciation_rate) ^ years

/-- Theorem: The value of equipment initially worth 10,000 yuan, depreciating by 50% annually, will be 1,250 yuan after 3 years. -/
theorem equipment_value_after_three_years :
  equipment_value 10000 0.5 3 = 1250 := by
  sorry

end equipment_value_after_three_years_l453_45388


namespace number_of_ambiguous_dates_l453_45368

/-- The number of days that cannot be uniquely determined by date notation -/
def ambiguous_dates : ℕ :=
  let total_possible_ambiguous := 12 * 12  -- Days 1-12 for each of the 12 months
  let non_ambiguous := 12  -- Dates where day and month are the same (e.g., 1.1, 2.2, ..., 12.12)
  total_possible_ambiguous - non_ambiguous

/-- Theorem stating that the number of ambiguous dates is 132 -/
theorem number_of_ambiguous_dates : ambiguous_dates = 132 := by
  sorry


end number_of_ambiguous_dates_l453_45368


namespace athena_spent_14_dollars_l453_45316

/-- The total amount Athena spent on snacks for her friends -/
def total_spent (sandwich_price : ℝ) (sandwich_quantity : ℕ) (drink_price : ℝ) (drink_quantity : ℕ) : ℝ :=
  sandwich_price * sandwich_quantity + drink_price * drink_quantity

/-- Theorem stating that Athena spent $14 in total -/
theorem athena_spent_14_dollars :
  let sandwich_price : ℝ := 3
  let sandwich_quantity : ℕ := 3
  let drink_price : ℝ := 2.5
  let drink_quantity : ℕ := 2
  total_spent sandwich_price sandwich_quantity drink_price drink_quantity = 14 := by
sorry

end athena_spent_14_dollars_l453_45316


namespace probability_perfect_square_sum_l453_45344

def roll_outcomes : ℕ := 64

def favorable_outcomes : ℕ := 12

theorem probability_perfect_square_sum (roll_outcomes : ℕ) (favorable_outcomes : ℕ) :
  (favorable_outcomes : ℚ) / (roll_outcomes : ℚ) = 3 / 16 :=
by sorry

end probability_perfect_square_sum_l453_45344


namespace smallest_n_divisible_by_2013_l453_45300

def product_of_consecutive_evens (n : ℕ) : ℕ :=
  Finset.prod (Finset.range (n/2)) (fun i => 2 * (i + 1))

theorem smallest_n_divisible_by_2013 :
  ∀ n : ℕ, n % 2 = 0 →
    (product_of_consecutive_evens n % 2013 = 0 →
      n ≥ 122) ∧
    (n ≥ 122 →
      product_of_consecutive_evens n % 2013 = 0) :=
sorry

end smallest_n_divisible_by_2013_l453_45300


namespace min_keystrokes_to_243_l453_45343

-- Define the allowed operations
def add_one (n : ℕ) : ℕ := n + 1
def multiply_two (n : ℕ) : ℕ := n * 2
def multiply_three (n : ℕ) : ℕ := if n % 3 = 0 then n * 3 else n

-- Define a function to represent a sequence of operations
def apply_operations (ops : List (ℕ → ℕ)) (start : ℕ) : ℕ :=
  ops.foldl (λ acc op => op acc) start

-- Define the theorem
theorem min_keystrokes_to_243 :
  ∃ (ops : List (ℕ → ℕ)), 
    (∀ op ∈ ops, op ∈ [add_one, multiply_two, multiply_three]) ∧
    apply_operations ops 1 = 243 ∧
    ops.length = 5 ∧
    (∀ (other_ops : List (ℕ → ℕ)), 
      (∀ op ∈ other_ops, op ∈ [add_one, multiply_two, multiply_three]) →
      apply_operations other_ops 1 = 243 →
      other_ops.length ≥ 5) :=
sorry

end min_keystrokes_to_243_l453_45343


namespace investor_profit_l453_45346

def total_investment : ℝ := 1900
def investment_fund1 : ℝ := 1700
def profit_rate_fund1 : ℝ := 0.09
def profit_rate_fund2 : ℝ := 0.02

def investment_fund2 : ℝ := total_investment - investment_fund1

def profit_fund1 : ℝ := investment_fund1 * profit_rate_fund1
def profit_fund2 : ℝ := investment_fund2 * profit_rate_fund2

def total_profit : ℝ := profit_fund1 + profit_fund2

theorem investor_profit : total_profit = 157 := by
  sorry

end investor_profit_l453_45346


namespace sequence_properties_l453_45378

/-- Given a sequence {a_n} where S_n is the sum of the first n terms -/
def S (n : ℕ+) : ℚ := 3 * n.val^2 + 4 * n.val

/-- The nth term of the sequence -/
def a (n : ℕ+) : ℚ := S n - S (n - 1)

theorem sequence_properties :
  (∀ n : ℕ+, a n = 6 * n.val + 1) ∧
  (∀ n : ℕ+, n ≥ 2 → a n - a (n - 1) = 6) :=
by sorry

end sequence_properties_l453_45378


namespace infections_exceed_threshold_l453_45391

/-- The number of people infected after two rounds of infection -/
def infected_after_two_rounds : ℕ := 81

/-- The average number of people infected by one person in each round -/
def average_infections_per_round : ℕ := 8

/-- The threshold number of infections we want to exceed after three rounds -/
def infection_threshold : ℕ := 700

/-- Theorem stating that the number of infected people after three rounds exceeds the threshold -/
theorem infections_exceed_threshold : 
  infected_after_two_rounds * (1 + average_infections_per_round) > infection_threshold := by
  sorry


end infections_exceed_threshold_l453_45391


namespace units_digit_of_2_power_10_l453_45365

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The function to calculate 2 to the power of n -/
def powerOfTwo (n : ℕ) : ℕ := 2^n

theorem units_digit_of_2_power_10 : unitsDigit (powerOfTwo 10) = 4 := by
  sorry

end units_digit_of_2_power_10_l453_45365


namespace shopkeeper_loss_percent_l453_45335

theorem shopkeeper_loss_percent 
  (profit_rate : ℝ) 
  (theft_rate : ℝ) 
  (initial_value : ℝ) 
  (profit_rate_is_10_percent : profit_rate = 0.1)
  (theft_rate_is_60_percent : theft_rate = 0.6)
  (initial_value_positive : initial_value > 0) : 
  let remaining_goods := initial_value * (1 - theft_rate)
  let final_value := remaining_goods * (1 + profit_rate)
  let loss := initial_value - final_value
  let loss_percent := (loss / initial_value) * 100
  loss_percent = 56 := by
sorry

end shopkeeper_loss_percent_l453_45335


namespace least_perimeter_l453_45315

/-- Represents a triangle with two known sides and an integral third side -/
structure Triangle where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ
  is_triangle : side1 + side2 > side3 ∧ side1 + side3 > side2 ∧ side2 + side3 > side1

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℕ := t.side1 + t.side2 + t.side3

/-- The specific triangle from the problem -/
def problem_triangle : Triangle → Prop
  | t => t.side1 = 24 ∧ t.side2 = 51

theorem least_perimeter :
  ∀ t : Triangle, problem_triangle t →
  ∀ u : Triangle, problem_triangle u →
  perimeter t ≥ 103 ∧ (∃ v : Triangle, problem_triangle v ∧ perimeter v = 103) :=
by sorry

end least_perimeter_l453_45315


namespace inverse_f_84_l453_45398

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 + 3

-- State the theorem
theorem inverse_f_84 : 
  ∃ (y : ℝ), f y = 84 ∧ y = 3 := by sorry

end inverse_f_84_l453_45398


namespace jones_trip_time_comparison_l453_45350

/-- Proves that the time taken for the third trip is three times the time taken for the first trip
    given the conditions of Jones' three trips. -/
theorem jones_trip_time_comparison 
  (v : ℝ) -- Original speed
  (h1 : v > 0) -- Assumption that speed is positive
  (d1 : ℝ) (h2 : d1 = 40) -- Distance of first trip
  (d2 : ℝ) (h3 : d2 = 200) -- Distance of second trip
  (d3 : ℝ) (h4 : d3 = 480) -- Distance of third trip
  (v2 : ℝ) (h5 : v2 = 2 * v) -- Speed of second trip
  (v3 : ℝ) (h6 : v3 = 2 * v2) -- Speed of third trip
  : (d3 / v3) = 3 * (d1 / v) := by
  sorry

end jones_trip_time_comparison_l453_45350


namespace boy_age_problem_l453_45354

theorem boy_age_problem (total_boys : Nat) (avg_age_all : Nat) (avg_age_first_six : Nat) (avg_age_last_six : Nat)
  (h1 : total_boys = 11)
  (h2 : avg_age_all = 50)
  (h3 : avg_age_first_six = 49)
  (h4 : avg_age_last_six = 52) :
  total_boys * avg_age_all = 6 * avg_age_first_six + 6 * avg_age_last_six - 56 := by
  sorry

#check boy_age_problem

end boy_age_problem_l453_45354


namespace quadratic_equation_solution_l453_45382

theorem quadratic_equation_solution (p q : ℤ) :
  (∃ x : ℝ, 36 * x^2 - 4 * (p^2 + 11) * x + 135 * (p + q) + 576 = 0) →
  p + q = 20 := by
sorry

end quadratic_equation_solution_l453_45382


namespace intersection_and_union_when_a_is_3_intersection_empty_iff_a_less_than_1_l453_45381

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 - a ≤ x ∧ x ≤ 2 + a}
def B : Set ℝ := {x | x^2 - 5*x + 4 ≥ 0}

-- Define the universal set U (assuming it's the real numbers)
def U : Set ℝ := Set.univ

-- Theorem for part (1)
theorem intersection_and_union_when_a_is_3 :
  (A 3 ∩ B = {x | (-1 ≤ x ∧ x ≤ 1) ∨ (4 ≤ x ∧ x ≤ 5)}) ∧
  (A 3 ∪ (U \ B) = {x | -1 ≤ x ∧ x ≤ 5}) := by sorry

-- Theorem for part (2)
theorem intersection_empty_iff_a_less_than_1 :
  ∀ a : ℝ, (A a ∩ B = ∅) ↔ a < 1 := by sorry

end intersection_and_union_when_a_is_3_intersection_empty_iff_a_less_than_1_l453_45381


namespace basketball_donations_l453_45312

theorem basketball_donations (total_donations : ℕ) 
  (basketball_hoops : ℕ) (pool_floats : ℕ) (footballs : ℕ) (tennis_balls : ℕ) :
  total_donations = 300 →
  basketball_hoops = 60 →
  pool_floats = 120 →
  footballs = 50 →
  tennis_balls = 40 →
  ∃ (basketballs : ℕ),
    basketballs = total_donations - (basketball_hoops + (pool_floats - pool_floats / 4) + footballs + tennis_balls) + basketball_hoops / 2 ∧
    basketballs = 90 :=
by sorry

end basketball_donations_l453_45312


namespace picture_frame_width_l453_45337

theorem picture_frame_width 
  (height : ℝ) 
  (circumference : ℝ) 
  (h_height : height = 12) 
  (h_circumference : circumference = 38) : 
  let width := (circumference - 2 * height) / 2
  width = 7 := by
sorry

end picture_frame_width_l453_45337


namespace remaining_safe_caffeine_l453_45399

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

end remaining_safe_caffeine_l453_45399


namespace students_without_glasses_l453_45369

theorem students_without_glasses (total : ℕ) (with_glasses_percent : ℚ) 
  (h1 : total = 325) 
  (h2 : with_glasses_percent = 40 / 100) : 
  ↑total * (1 - with_glasses_percent) = 195 := by
  sorry

end students_without_glasses_l453_45369


namespace negation_equivalence_l453_45324

theorem negation_equivalence (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 2*x + a ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + a > 0) := by sorry

end negation_equivalence_l453_45324


namespace sqrt_triangle_exists_abs_diff_plus_one_triangle_exists_l453_45328

-- Define a triangle with sides a, b, and c
structure Triangle :=
  (a b c : ℝ)
  (h₁ : a > 0)
  (h₂ : b > 0)
  (h₃ : c > 0)
  (triangle_inequality₁ : a + b > c)
  (triangle_inequality₂ : b + c > a)
  (triangle_inequality₃ : c + a > b)

-- Theorem 1: A triangle with sides √a, √b, and √c always exists
theorem sqrt_triangle_exists (t : Triangle) : 
  ∃ (t' : Triangle), t'.a = Real.sqrt t.a ∧ t'.b = Real.sqrt t.b ∧ t'.c = Real.sqrt t.c :=
sorry

-- Theorem 2: A triangle with sides |a-b|+1, |b-c|+1, and |c-a|+1 always exists
theorem abs_diff_plus_one_triangle_exists (t : Triangle) :
  ∃ (t' : Triangle), t'.a = |t.a - t.b| + 1 ∧ t'.b = |t.b - t.c| + 1 ∧ t'.c = |t.c - t.a| + 1 :=
sorry

end sqrt_triangle_exists_abs_diff_plus_one_triangle_exists_l453_45328


namespace complement_intersection_theorem_l453_45338

open Set

def I : Finset Nat := {1,2,3,4,5}
def A : Finset Nat := {2,3,5}
def B : Finset Nat := {1,2}

theorem complement_intersection_theorem :
  (I \ B) ∩ A = {3,5} := by sorry

end complement_intersection_theorem_l453_45338


namespace exponent_fraction_equality_l453_45392

theorem exponent_fraction_equality : (3^2015 + 3^2013) / (3^2015 - 3^2013) = 5/4 := by
  sorry

end exponent_fraction_equality_l453_45392


namespace log_product_range_l453_45360

theorem log_product_range : ∃ y : ℝ,
  y = Real.log 6 / Real.log 5 * Real.log 7 / Real.log 6 * Real.log 8 / Real.log 7 * Real.log 9 / Real.log 8 * Real.log 10 / Real.log 9 ∧
  1 < y ∧ y < 2 := by
  sorry

end log_product_range_l453_45360


namespace smallest_integer_satisfying_inequality_l453_45376

theorem smallest_integer_satisfying_inequality : 
  (∀ y : ℤ, y < 8 → (y : ℚ) / 4 + 3 / 7 ≤ 9 / 4) ∧ 
  (8 : ℚ) / 4 + 3 / 7 > 9 / 4 := by
  sorry

end smallest_integer_satisfying_inequality_l453_45376


namespace sum_of_first_eight_multiples_of_eleven_l453_45331

/-- The sum of the first n distinct positive integer multiples of m -/
def sum_of_multiples (n m : ℕ) : ℕ := 
  m * n * (n + 1) / 2

/-- Theorem: The sum of the first 8 distinct positive integer multiples of 11 is 396 -/
theorem sum_of_first_eight_multiples_of_eleven : 
  sum_of_multiples 8 11 = 396 := by
  sorry

end sum_of_first_eight_multiples_of_eleven_l453_45331


namespace expression_value_l453_45345

theorem expression_value (a b : ℝ) (h : a * 1 + b * 2 = 3) : 2 * a + 4 * b - 5 = 1 := by
  sorry

end expression_value_l453_45345


namespace work_together_duration_l453_45321

/-- Given two workers A and B, where A can complete a job in 15 days and B in 20 days,
    this theorem proves that if they work together until 5/12 of the job is left,
    then they worked together for 5 days. -/
theorem work_together_duration (a_rate b_rate : ℚ) (work_left : ℚ) (days_worked : ℕ) :
  a_rate = 1 / 15 →
  b_rate = 1 / 20 →
  work_left = 5 / 12 →
  (a_rate + b_rate) * days_worked = 1 - work_left →
  days_worked = 5 :=
by sorry

end work_together_duration_l453_45321


namespace range_of_f_l453_45313

-- Define the function f
def f (x : ℝ) : ℝ := x + |x - 2|

-- State the theorem about the range of f
theorem range_of_f :
  {y : ℝ | ∃ x : ℝ, f x = y} = Set.Ici 2 := by sorry

end range_of_f_l453_45313


namespace radical_simplification_l453_45385

theorem radical_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q) = 21 * q * Real.sqrt (2 * q) := by
  sorry

end radical_simplification_l453_45385


namespace part1_part2_l453_45352

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x - a| + a

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x, f a x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) → a = 1 := by sorry

-- Part 2
theorem part2 (m : ℝ) :
  (∃ n : ℝ, |2 * n - 1| + 1 ≤ m - (|2 * (-n) - 1| + 1)) → m ≥ 4 := by sorry

end part1_part2_l453_45352


namespace gcf_of_75_and_90_l453_45329

theorem gcf_of_75_and_90 : Nat.gcd 75 90 = 15 := by
  sorry

end gcf_of_75_and_90_l453_45329


namespace tyler_puppies_l453_45332

/-- The number of puppies Tyler has after a week -/
def total_puppies (total_dogs : ℕ) 
                  (dogs_with_5_5 : ℕ) 
                  (dogs_with_8 : ℕ) 
                  (puppies_per_dog_1 : ℚ) 
                  (puppies_per_dog_2 : ℕ) 
                  (puppies_per_dog_3 : ℕ) 
                  (dogs_with_extra : ℕ) 
                  (extra_puppies : ℚ) : ℚ := 
  let remaining_dogs := total_dogs - dogs_with_5_5 - dogs_with_8
  dogs_with_5_5 * puppies_per_dog_1 + 
  dogs_with_8 * puppies_per_dog_2 + 
  remaining_dogs * puppies_per_dog_3 + 
  dogs_with_extra * extra_puppies

theorem tyler_puppies : 
  total_puppies 35 15 10 (5.5) 8 6 5 (2.5) = 235 := by
  sorry

end tyler_puppies_l453_45332


namespace whipped_cream_cans_needed_l453_45394

/-- The number of pies Billie bakes per day -/
def pies_per_day : ℕ := 3

/-- The number of days Billie bakes pies -/
def baking_days : ℕ := 11

/-- The number of cans of whipped cream needed to cover one pie -/
def cans_per_pie : ℕ := 2

/-- The number of pies Tiffany eats -/
def pies_eaten : ℕ := 4

/-- The total number of pies Billie bakes -/
def total_pies : ℕ := pies_per_day * baking_days

/-- The number of pies remaining after Tiffany eats -/
def remaining_pies : ℕ := total_pies - pies_eaten

/-- The number of cans of whipped cream needed to cover the remaining pies -/
def cans_needed : ℕ := remaining_pies * cans_per_pie

theorem whipped_cream_cans_needed : cans_needed = 58 := by
  sorry

end whipped_cream_cans_needed_l453_45394


namespace largest_d_for_negative_five_in_range_l453_45377

-- Define the function g
def g (x d : ℝ) : ℝ := x^2 + 5*x + d

-- State the theorem
theorem largest_d_for_negative_five_in_range :
  (∃ (d : ℝ), ∀ (d' : ℝ), 
    (∃ (x : ℝ), g x d = -5) → 
    (∃ (x : ℝ), g x d' = -5) → 
    d' ≤ d) ∧
  (∃ (x : ℝ), g x (5/4) = -5) :=
sorry

end largest_d_for_negative_five_in_range_l453_45377


namespace solve_for_B_l453_45309

theorem solve_for_B : ∃ B : ℚ, (3 * B - 5 = 23) ∧ (B = 28 / 3) := by sorry

end solve_for_B_l453_45309


namespace judes_chair_expenditure_l453_45301

/-- Proves that the amount spent on chairs is $36 given the conditions of Jude's purchase --/
theorem judes_chair_expenditure
  (table_cost : ℕ)
  (plate_set_cost : ℕ)
  (num_plate_sets : ℕ)
  (money_given : ℕ)
  (change_received : ℕ)
  (h1 : table_cost = 50)
  (h2 : plate_set_cost = 20)
  (h3 : num_plate_sets = 2)
  (h4 : money_given = 130)
  (h5 : change_received = 4) :
  money_given - change_received - (table_cost + num_plate_sets * plate_set_cost) = 36 := by
  sorry

#check judes_chair_expenditure

end judes_chair_expenditure_l453_45301


namespace cos_seven_pi_sixths_l453_45302

theorem cos_seven_pi_sixths : Real.cos (7 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end cos_seven_pi_sixths_l453_45302


namespace sequence_on_line_is_arithmetic_l453_45308

/-- Given a sequence {a_n} where for any n ∈ ℕ*, the point P_n(n, a_n) lies on the line y = 2x + 1,
    prove that {a_n} is an arithmetic sequence with a common difference of 2. -/
theorem sequence_on_line_is_arithmetic (a : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → a n = 2 * n + 1) →
  ∃ (a₀ : ℝ), ∀ n : ℕ, a n = a₀ + 2 * n :=
by sorry

end sequence_on_line_is_arithmetic_l453_45308


namespace car_distribution_l453_45393

theorem car_distribution (total_cars : ℕ) (first_supplier : ℕ) (fourth_fifth_each : ℕ) :
  total_cars = 5650000 →
  first_supplier = 1000000 →
  fourth_fifth_each = 325000 →
  ∃ (second_supplier : ℕ),
    second_supplier + first_supplier + (second_supplier + first_supplier) + 2 * fourth_fifth_each = total_cars ∧
    second_supplier = first_supplier + 500000 := by
  sorry

end car_distribution_l453_45393


namespace breakfast_cost_is_correct_l453_45397

/-- Calculates the total cost of breakfast for Francis and Kiera -/
def breakfast_cost : ℝ :=
  let muffin_price : ℝ := 2
  let fruit_cup_price : ℝ := 3
  let coffee_price : ℝ := 1.5
  let discount_rate : ℝ := 0.1
  
  let francis_muffins : ℕ := 2
  let francis_fruit_cups : ℕ := 2
  let francis_coffee : ℕ := 1
  
  let kiera_muffins : ℕ := 2
  let kiera_fruit_cups : ℕ := 1
  let kiera_coffee : ℕ := 2
  
  let francis_cost : ℝ := 
    muffin_price * francis_muffins + 
    fruit_cup_price * francis_fruit_cups + 
    coffee_price * francis_coffee
  
  let kiera_cost_before_discount : ℝ := 
    muffin_price * kiera_muffins + 
    fruit_cup_price * kiera_fruit_cups + 
    coffee_price * kiera_coffee
  
  let discount_amount : ℝ := 
    discount_rate * (muffin_price * 2 + fruit_cup_price)
  
  let kiera_cost : ℝ := kiera_cost_before_discount - discount_amount
  
  francis_cost + kiera_cost

theorem breakfast_cost_is_correct : breakfast_cost = 20.8 := by
  sorry

end breakfast_cost_is_correct_l453_45397


namespace div_fraction_equality_sum_fraction_equality_l453_45319

-- Define variables
variable (a b : ℝ)

-- Assume a ≠ b and a ≠ 0 to avoid division by zero
variable (h1 : a ≠ b) (h2 : a ≠ 0)

-- Theorem 1
theorem div_fraction_equality : (4 * b^3 / a) / (2 * b / a^2) = 2 * a * b^2 := by sorry

-- Theorem 2
theorem sum_fraction_equality : a^2 / (a - b) + b^2 / (a - b) - 2 * a * b / (a - b) = a - b := by sorry

end div_fraction_equality_sum_fraction_equality_l453_45319


namespace bird_count_2003_l453_45304

/-- The number of birds in the Weishui Development Zone over three years -/
structure BirdCount where
  year2001 : ℝ
  year2002 : ℝ
  year2003 : ℝ

/-- The conditions of the bird count problem -/
def bird_count_conditions (bc : BirdCount) : Prop :=
  bc.year2002 = 1.5 * bc.year2001 ∧ 
  bc.year2003 = 2 * bc.year2002

/-- Theorem stating that under the given conditions, the number of birds in 2003 is 3 times the number in 2001 -/
theorem bird_count_2003 (bc : BirdCount) (h : bird_count_conditions bc) : 
  bc.year2003 = 3 * bc.year2001 := by
  sorry

end bird_count_2003_l453_45304


namespace intersection_when_a_is_one_intersection_equals_B_iff_l453_45375

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < a+3}

-- Statement I
theorem intersection_when_a_is_one :
  (Set.univ \ A) ∩ (B 1) = {x | 3 < x ∧ x < 4} := by sorry

-- Statement II
theorem intersection_equals_B_iff (a : ℝ) :
  (Set.univ \ A) ∩ (B a) = B a ↔ a ≤ -2 ∨ a ≥ 3/2 := by sorry

end intersection_when_a_is_one_intersection_equals_B_iff_l453_45375


namespace julia_played_with_16_kids_l453_45340

def kids_on_tuesday : ℕ := 4

def kids_difference : ℕ := 12

def kids_on_monday : ℕ := kids_on_tuesday + kids_difference

theorem julia_played_with_16_kids : kids_on_monday = 16 := by
  sorry

end julia_played_with_16_kids_l453_45340


namespace population_change_l453_45364

/-- Theorem: Given an initial population that increases by 30% in the first year
    and then decreases by x% in the second year, if the initial population is 15000
    and the final population is 13650, then x = 30. -/
theorem population_change (x : ℝ) : 
  let initial_population : ℝ := 15000
  let first_year_increase : ℝ := 0.3
  let final_population : ℝ := 13650
  let population_after_first_year : ℝ := initial_population * (1 + first_year_increase)
  let population_after_second_year : ℝ := population_after_first_year * (1 - x / 100)
  population_after_second_year = final_population → x = 30 := by
sorry

end population_change_l453_45364


namespace geometric_sequence_problem_l453_45303

-- Define the geometric sequence
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n - 1)

-- State the theorem
theorem geometric_sequence_problem (a₁ a₄ : ℝ) (m : ℤ) :
  a₁ = 2 →
  a₄ = 1/4 →
  m = -15 →
  (∃ r : ℝ, ∀ n : ℕ, geometric_sequence a₁ r n = 2^(2 - n)) →
  m = 14 := by
  sorry


end geometric_sequence_problem_l453_45303


namespace peanut_butter_servings_l453_45353

/-- The amount of peanut butter in the jar in tablespoons -/
def jar_amount : ℚ := 45 + 2/3

/-- The size of one serving of peanut butter in tablespoons -/
def serving_size : ℚ := 1 + 1/3

/-- The number of servings in the jar -/
def servings : ℚ := jar_amount / serving_size

theorem peanut_butter_servings : servings = 34 + 1/4 := by
  sorry

end peanut_butter_servings_l453_45353


namespace range_of_k_value_of_k_with_condition_l453_45370

-- Define the quadratic equation
def quadratic (k x : ℝ) : ℝ := x^2 + (2*k - 1)*x + k^2 - 1

-- Define the condition for two real roots
def has_two_real_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0

-- Define the condition for the sum of squares
def sum_of_squares_condition (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, quadratic k x₁ = 0 ∧ quadratic k x₂ = 0 ∧ x₁^2 + x₂^2 = 16 + x₁*x₂

-- Theorem for the range of k
theorem range_of_k :
  ∀ k : ℝ, has_two_real_roots k → k ≤ 5/4 :=
sorry

-- Theorem for the value of k when sum of squares condition is satisfied
theorem value_of_k_with_condition :
  ∀ k : ℝ, has_two_real_roots k → sum_of_squares_condition k → k = -2 :=
sorry

end range_of_k_value_of_k_with_condition_l453_45370


namespace inequality_proof_l453_45325

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end inequality_proof_l453_45325


namespace smallest_sum_reciprocals_l453_45374

theorem smallest_sum_reciprocals (x y : ℕ+) (hxy : x ≠ y) (hsum : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 10) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 10 ∧ (↑a + ↑b : ℕ) = 40 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 10 → (↑c + ↑d : ℕ) ≥ 40 :=
sorry

end smallest_sum_reciprocals_l453_45374


namespace sqrt_x_plus_3_real_l453_45310

theorem sqrt_x_plus_3_real (x : ℝ) : (∃ y : ℝ, y^2 = x + 3) ↔ x ≥ -3 := by sorry

end sqrt_x_plus_3_real_l453_45310


namespace sams_dimes_l453_45327

/-- Given that Sam had 9 dimes initially and received 7 more dimes from his dad,
    prove that the total number of dimes Sam has now is 16. -/
theorem sams_dimes (initial_dimes : ℕ) (received_dimes : ℕ) (total_dimes : ℕ) : 
  initial_dimes = 9 → received_dimes = 7 → total_dimes = initial_dimes + received_dimes → total_dimes = 16 := by
  sorry

end sams_dimes_l453_45327


namespace gcf_seven_eight_factorial_l453_45363

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem gcf_seven_eight_factorial :
  Nat.gcd (factorial 7) (factorial 8) = factorial 7 := by
  sorry

end gcf_seven_eight_factorial_l453_45363


namespace consecutive_integers_divisibility_l453_45383

theorem consecutive_integers_divisibility (a₁ a₂ a₃ : ℕ) 
  (h1 : a₁ + 1 = a₂) 
  (h2 : a₂ + 1 = a₃) 
  (h3 : 0 < a₁) : 
  a₂^3 ∣ (a₁ * a₂ * a₃ + a₂) := by
  sorry

end consecutive_integers_divisibility_l453_45383


namespace mary_overtime_pay_increase_l453_45379

/-- Represents Mary's work schedule and pay structure -/
structure WorkSchedule where
  maxHours : Nat
  regularHours : Nat
  regularRate : ℚ
  totalEarnings : ℚ

/-- Calculates the percentage increase in overtime pay given a work schedule -/
def overtimePayIncrease (schedule : WorkSchedule) : ℚ :=
  let regularEarnings := schedule.regularHours * schedule.regularRate
  let overtimeEarnings := schedule.totalEarnings - regularEarnings
  let overtimeHours := schedule.maxHours - schedule.regularHours
  let overtimeRate := overtimeEarnings / overtimeHours
  ((overtimeRate - schedule.regularRate) / schedule.regularRate) * 100

/-- Theorem stating that Mary's overtime pay increase is 25% -/
theorem mary_overtime_pay_increase :
  let mary_schedule : WorkSchedule := {
    maxHours := 45,
    regularHours := 20,
    regularRate := 8,
    totalEarnings := 410
  }
  overtimePayIncrease mary_schedule = 25 := by
  sorry


end mary_overtime_pay_increase_l453_45379


namespace shortest_path_length_is_28b_l453_45314

/-- Represents a 3x3 grid of blocks with side length b -/
structure Grid :=
  (b : ℝ)
  (size : ℕ := 3)

/-- The number of street segments in the grid -/
def Grid.streetSegments (g : Grid) : ℕ := 24

/-- The number of intersections with odd degree -/
def Grid.oddDegreeIntersections (g : Grid) : ℕ := 8

/-- The extra segments that need to be traversed twice -/
def Grid.extraSegments (g : Grid) : ℕ := g.oddDegreeIntersections / 2

/-- The shortest path length to pave all streets in the grid -/
def Grid.shortestPathLength (g : Grid) : ℝ :=
  (g.streetSegments + g.extraSegments) * g.b

/-- Theorem stating that the shortest path length is 28b -/
theorem shortest_path_length_is_28b (g : Grid) :
  g.shortestPathLength = 28 * g.b := by
  sorry

end shortest_path_length_is_28b_l453_45314


namespace correct_small_glasses_l453_45342

/-- Calculates the number of small drinking glasses given the following conditions:
  * 50 jelly beans fill a large glass
  * 25 jelly beans fill a small glass
  * There are 5 large glasses
  * A total of 325 jelly beans are used
-/
def number_of_small_glasses (large_glass_beans : ℕ) (small_glass_beans : ℕ) 
  (num_large_glasses : ℕ) (total_beans : ℕ) : ℕ :=
  (total_beans - large_glass_beans * num_large_glasses) / small_glass_beans

theorem correct_small_glasses : 
  number_of_small_glasses 50 25 5 325 = 3 := by
  sorry

end correct_small_glasses_l453_45342


namespace greatest_x_value_l453_45384

theorem greatest_x_value (x : ℤ) (h : (2.134 : ℝ) * (10 : ℝ) ^ (x : ℝ) < 210000) :
  x ≤ 4 ∧ ∃ y : ℤ, y > 4 → (2.134 : ℝ) * (10 : ℝ) ^ (y : ℝ) ≥ 210000 :=
by sorry

end greatest_x_value_l453_45384


namespace comic_book_frames_per_page_l453_45359

/-- Given a comic book with a total number of frames and pages, 
    calculate the number of frames per page. -/
def frames_per_page (total_frames : ℕ) (total_pages : ℕ) : ℕ :=
  total_frames / total_pages

/-- Theorem stating that for a comic book with 143 frames and 13 pages, 
    the number of frames per page is 11. -/
theorem comic_book_frames_per_page :
  frames_per_page 143 13 = 11 := by
  sorry

end comic_book_frames_per_page_l453_45359


namespace prob_all_players_odd_sum_l453_45357

/-- The number of tiles --/
def n : ℕ := 12

/-- The number of odd tiles --/
def odd_tiles : ℕ := n / 2

/-- The number of even tiles --/
def even_tiles : ℕ := n / 2

/-- The number of tiles each player selects --/
def tiles_per_player : ℕ := 4

/-- The number of players --/
def num_players : ℕ := 3

/-- The probability of all players getting an odd sum --/
def prob_all_odd_sum : ℚ := 800 / 963

/-- Theorem stating the probability of all players getting an odd sum --/
theorem prob_all_players_odd_sum :
  let total_distributions := Nat.choose n tiles_per_player * 
                             Nat.choose (n - tiles_per_player) tiles_per_player * 
                             Nat.choose (n - 2 * tiles_per_player) tiles_per_player
  let odd_sum_distributions := (Nat.choose odd_tiles 3 * Nat.choose even_tiles 1)^num_players / 
                               Nat.factorial num_players
  (odd_sum_distributions : ℚ) / total_distributions = prob_all_odd_sum := by
  sorry

end prob_all_players_odd_sum_l453_45357


namespace absolute_value_inequality_l453_45387

theorem absolute_value_inequality (x : ℝ) : 
  |2*x - 1| - |x + 1| < 1 ↔ -1/3 < x ∧ x < 3 :=
sorry

end absolute_value_inequality_l453_45387


namespace orange_price_theorem_l453_45348

/-- The cost of fruits and the discount policy at a store --/
structure FruitStore where
  apple_cost : ℚ
  banana_cost : ℚ
  discount_per_five : ℚ

/-- A customer's purchase of fruits --/
structure Purchase where
  apples : ℕ
  oranges : ℕ
  bananas : ℕ

/-- Calculate the total cost of a purchase given the store's prices and an orange price --/
def totalCost (store : FruitStore) (purchase : Purchase) (orange_price : ℚ) : ℚ :=
  store.apple_cost * purchase.apples +
  orange_price * purchase.oranges +
  store.banana_cost * purchase.bananas -
  store.discount_per_five * ((purchase.apples + purchase.oranges + purchase.bananas) / 5)

/-- The theorem stating the price of oranges based on Mary's purchase --/
theorem orange_price_theorem (store : FruitStore) (purchase : Purchase) :
  store.apple_cost = 1 →
  store.banana_cost = 3 →
  store.discount_per_five = 1 →
  purchase.apples = 5 →
  purchase.oranges = 3 →
  purchase.bananas = 2 →
  totalCost store purchase (8/3) = 15 :=
by sorry

end orange_price_theorem_l453_45348


namespace function_local_extrema_l453_45389

/-- The function f(x) = (x^2 + ax + 2)e^x has both a local maximum and a local minimum
    if and only if a > 2 or a < -2 -/
theorem function_local_extrema (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    IsLocalMax (fun x => (x^2 + a*x + 2) * Real.exp x) x₁ ∧
    IsLocalMin (fun x => (x^2 + a*x + 2) * Real.exp x) x₂) ↔
  (a > 2 ∨ a < -2) :=
sorry

end function_local_extrema_l453_45389
