import Mathlib

namespace competitive_exam_candidates_l815_81510

theorem competitive_exam_candidates (x : ℝ)
  (A_selected : ℝ := 0.06 * x) 
  (B_selected : ℝ := 0.07 * x) 
  (h : B_selected = A_selected + 81) :
  x = 8100 := by
  sorry

end competitive_exam_candidates_l815_81510


namespace numberOfValidFiveDigitNumbers_l815_81512

namespace MathProof

def isFiveDigitNumber (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def isDivisibleBy5 (n : ℕ) : Prop := n % 5 = 0

def firstAndLastDigitsEqual (n : ℕ) : Prop := 
  let firstDigit := (n / 10000) % 10
  let lastDigit := n % 10
  firstDigit = lastDigit

def sumOfDigitsDivisibleBy5 (n : ℕ) : Prop := 
  let d1 := (n / 10000) % 10
  let d2 := (n / 1000) % 10
  let d3 := (n / 100) % 10
  let d4 := (n / 10) % 10
  let d5 := n % 10
  (d1 + d2 + d3 + d4 + d5) % 5 = 0

theorem numberOfValidFiveDigitNumbers :
  ∃ (count : ℕ), count = 200 ∧ 
  count = Nat.card {n : ℕ // isFiveDigitNumber n ∧ 
                                isDivisibleBy5 n ∧ 
                                firstAndLastDigitsEqual n ∧ 
                                sumOfDigitsDivisibleBy5 n} :=
by
  sorry

end MathProof

end numberOfValidFiveDigitNumbers_l815_81512


namespace star_vertex_angle_l815_81582

-- Defining a function that calculates the star vertex angle for odd n-sided concave regular polygon
theorem star_vertex_angle (n : ℕ) (hn_odd : n % 2 = 1) (hn_gt3 : 3 < n) : 
  (180 - 360 / n) = (n - 2) * 180 / n := 
sorry

end star_vertex_angle_l815_81582


namespace batsman_average_increase_l815_81592

theorem batsman_average_increase
  (A : ℕ)  -- Assume the initial average is a non-negative integer
  (h1 : 11 * A + 70 = 12 * (A + 3))  -- Condition derived from the problem
  : A + 3 = 37 := 
by {
  -- The actual proof would go here, but is replaced by sorry to skip the proof
  sorry
}

end batsman_average_increase_l815_81592


namespace range_of_x_l815_81590

theorem range_of_x (x : ℝ) (p : x^2 - 2 * x - 3 < 0) (q : 1 / (x - 2) < 0) : -1 < x ∧ x < 2 :=
by
  sorry

end range_of_x_l815_81590


namespace find_a_given_conditions_l815_81542

theorem find_a_given_conditions (a : ℤ)
  (hA : ∃ (x : ℤ), x = 12 ∨ x = a^2 + 4 * a ∨ x = a - 2)
  (hA_contains_minus3 : ∃ (x : ℤ), (-3 = x) ∧ (x = 12 ∨ x = a^2 + 4 * a ∨ x = a - 2)) : a = -3 := 
by
  sorry

end find_a_given_conditions_l815_81542


namespace NOQZ_has_same_product_as_MNOQ_l815_81531

/-- Each letter of the alphabet is assigned a value (A=1, B=2, C=3, ..., Z=26). -/
def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5 | 'F' => 6 | 'G' => 7
  | 'H' => 8 | 'I' => 9 | 'J' => 10 | 'K' => 11 | 'L' => 12 | 'M' => 13
  | 'N' => 14 | 'O' => 15 | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19
  | 'T' => 20 | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25 | 'Z' => 26
  | _   => 0  -- We'll assume only uppercase letters are inputs

/-- The product of a four-letter list is the product of the values of its four letters. -/
def list_product (lst : List Char) : ℕ :=
  lst.map letter_value |>.foldl (· * ·) 1

/-- The product of the list MNOQ is calculated. -/
def product_MNOQ : ℕ := list_product ['M', 'N', 'O', 'Q']
/-- The product of the list BEHK is calculated. -/
def product_BEHK : ℕ := list_product ['B', 'E', 'H', 'K']
/-- The product of the list NOQZ is calculated. -/
def product_NOQZ : ℕ := list_product ['N', 'O', 'Q', 'Z']

theorem NOQZ_has_same_product_as_MNOQ :
  product_NOQZ = product_MNOQ := by
  sorry

end NOQZ_has_same_product_as_MNOQ_l815_81531


namespace shortest_distance_point_on_circle_to_line_l815_81589

theorem shortest_distance_point_on_circle_to_line
  (P : ℝ × ℝ)
  (hP : (P.1 + 1)^2 + (P.2 - 2)^2 = 1) :
  ∃ (d : ℝ), d = 3 :=
sorry

end shortest_distance_point_on_circle_to_line_l815_81589


namespace division_result_l815_81562

theorem division_result : 210 / (15 + 12 * 3 - 6) = 210 / 45 :=
by
  sorry

end division_result_l815_81562


namespace museum_paintings_discarded_l815_81555

def initial_paintings : ℕ := 2500
def percentage_to_discard : ℝ := 0.35
def paintings_discarded : ℝ := initial_paintings * percentage_to_discard

theorem museum_paintings_discarded : paintings_discarded = 875 :=
by
  -- Lean automatically simplifies this using basic arithmetic rules
  sorry

end museum_paintings_discarded_l815_81555


namespace fewer_onions_grown_l815_81552

def num_tomatoes := 2073
def num_cobs_of_corn := 4112
def num_onions := 985

theorem fewer_onions_grown : num_tomatoes + num_cobs_of_corn - num_onions = 5200 := by
  sorry

end fewer_onions_grown_l815_81552


namespace p_sufficient_not_necessary_for_q_l815_81583

def p (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 ≤ 2
def q (x y : ℝ) : Prop := y ≥ x - 1 ∧ y ≥ 1 - x ∧ y ≤ 1

theorem p_sufficient_not_necessary_for_q :
  (∀ x y : ℝ, q x y → p x y) ∧ ¬(∀ x y : ℝ, p x y → q x y) := by
  sorry

end p_sufficient_not_necessary_for_q_l815_81583


namespace sum_of_interior_angles_of_regular_polygon_l815_81560

theorem sum_of_interior_angles_of_regular_polygon (n : ℕ) (h : 60 = 360 / n) : (n - 2) * 180 = 720 :=
by
  sorry

end sum_of_interior_angles_of_regular_polygon_l815_81560


namespace deposit_percentage_l815_81529

noncomputable def last_year_cost : ℝ := 250
noncomputable def increase_percentage : ℝ := 0.40
noncomputable def amount_paid_at_pickup : ℝ := 315
noncomputable def total_cost := last_year_cost * (1 + increase_percentage)
noncomputable def deposit := total_cost - amount_paid_at_pickup
noncomputable def percentage_deposit := deposit / total_cost * 100

theorem deposit_percentage :
  percentage_deposit = 10 := 
  by
    sorry

end deposit_percentage_l815_81529


namespace arithmetic_sequence_z_value_l815_81550

theorem arithmetic_sequence_z_value :
  ∃ z : ℤ, (3 ^ 2 = 9 ∧ 3 ^ 4 = 81) ∧ z = (9 + 81) / 2 :=
by
  -- the proof goes here
  sorry

end arithmetic_sequence_z_value_l815_81550


namespace younger_by_17_l815_81509

variables (A B C : ℕ)

-- Given condition
axiom age_condition : A + B = B + C + 17

-- To show
theorem younger_by_17 : A - C = 17 :=
by
  sorry

end younger_by_17_l815_81509


namespace problem_statement_l815_81595

def f (x : ℤ) : ℤ := 3*x + 4
def g (x : ℤ) : ℤ := 4*x - 3

theorem problem_statement : (f (g (f 2))) / (g (f (g 2))) = 115 / 73 :=
by
  sorry

end problem_statement_l815_81595


namespace fraction_of_loss_l815_81598

theorem fraction_of_loss
  (SP CP : ℚ) (hSP : SP = 16) (hCP : CP = 17) :
  (CP - SP) / CP = 1 / 17 :=
by
  sorry

end fraction_of_loss_l815_81598


namespace polynomial_factorization_l815_81588

theorem polynomial_factorization :
  ∀ x : ℤ, x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) :=
by sorry

end polynomial_factorization_l815_81588


namespace max_abc_l815_81566

theorem max_abc : ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  (a * b + b * c = 518) ∧ 
  (a * b - a * c = 360) ∧ 
  (a * b * c = 1008) := 
by {
  -- Definitions of a, b, c satisfying the given conditions.
  -- Proof of the maximum value will be placed here (not required as per instructions).
  sorry
}

end max_abc_l815_81566


namespace xyz_expression_l815_81533

theorem xyz_expression (x y z : ℝ) 
  (h1 : x^2 - y * z = 2)
  (h2 : y^2 - z * x = 2)
  (h3 : z^2 - x * y = 2) :
  x * y + y * z + z * x = -2 :=
sorry

end xyz_expression_l815_81533


namespace hall_length_l815_81541

theorem hall_length
  (width : ℝ)
  (stone_length : ℝ)
  (stone_width : ℝ)
  (num_stones : ℕ)
  (h₁ : width = 15)
  (h₂ : stone_length = 0.8)
  (h₃ : stone_width = 0.5)
  (h₄ : num_stones = 1350) :
  ∃ length : ℝ, length = 36 :=
by
  sorry

end hall_length_l815_81541


namespace p_value_for_roots_l815_81554

theorem p_value_for_roots (α β : ℝ) (h1 : 3 * α^2 + 5 * α + 2 = 0) (h2 : 3 * β^2 + 5 * β + 2 = 0)
  (hαβ : α + β = -5/3) (hαβ_prod : α * β = 2/3) : p = -49/9 :=
by
  sorry

end p_value_for_roots_l815_81554


namespace interest_calculation_l815_81596

theorem interest_calculation :
  ∃ n : ℝ, 
  (1000 * 0.03 * n + 1400 * 0.05 * n = 350) →
  n = 3.5 := 
by 
  sorry

end interest_calculation_l815_81596


namespace tan_alpha_value_l815_81565

open Real

theorem tan_alpha_value (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 := 
sorry

end tan_alpha_value_l815_81565


namespace arithmetic_mean_reciprocal_primes_l815_81563

theorem arithmetic_mean_reciprocal_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := (1 : ℚ) / p1
  let r2 := (1 : ℚ) / p2
  let r3 := (1 : ℚ) / p3
  let r4 := (1 : ℚ) / p4
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 := by
sorry

end arithmetic_mean_reciprocal_primes_l815_81563


namespace complement_intersection_l815_81501

open Set

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {0, 2, 4}
def B : Set ℕ := {1, 2, 5}

theorem complement_intersection :
  ((U \ A) ∩ B) = {1, 5} :=
by
  sorry

end complement_intersection_l815_81501


namespace circle_tangent_lines_l815_81519

theorem circle_tangent_lines (h k : ℝ) (r : ℝ) (h_gt_10 : h > 10) (k_gt_10 : k > 10)
  (tangent_y_eq_10 : k - 10 = r)
  (tangent_y_eq_x : r = (|h - k| / Real.sqrt 2)) :
  (h, k) = (10 + (1 + Real.sqrt 2) * r, 10 + r) :=
by
  sorry

end circle_tangent_lines_l815_81519


namespace coloring_integers_l815_81527

theorem coloring_integers 
  (color : ℤ → ℕ) 
  (x y : ℤ) 
  (hx : x % 2 = 1) 
  (hy : y % 2 = 1) 
  (h_neq : |x| ≠ |y|) 
  (h_color_range : ∀ n : ℤ, color n < 4) :
  ∃ a b : ℤ, color a = color b ∧ (a - b = x ∨ a - b = y ∨ a - b = x + y ∨ a - b = x - y) :=
sorry

end coloring_integers_l815_81527


namespace dishonest_dealer_weight_l815_81557

noncomputable def dealer_weight_equiv (cost_price : ℝ) (selling_price : ℝ) (profit_percent : ℝ) : ℝ :=
  (1 - profit_percent / 100) * cost_price / selling_price

theorem dishonest_dealer_weight :
  dealer_weight_equiv 1 2 100 = 0.5 :=
by
  sorry

end dishonest_dealer_weight_l815_81557


namespace x_plus_p_l815_81505

theorem x_plus_p (x p : ℝ) (h1 : |x - 3| = p) (h2 : x > 3) : x + p = 2 * p + 3 :=
by
  sorry

end x_plus_p_l815_81505


namespace cost_price_percentage_l815_81570

variables (CP MP SP : ℝ) (x : ℝ)

theorem cost_price_percentage (h1 : CP = (x / 100) * MP)
                             (h2 : SP = 0.5 * MP)
                             (h3 : SP = 2 * CP) :
                             x = 25 := by
  sorry

end cost_price_percentage_l815_81570


namespace time_reading_per_week_l815_81500

-- Define the given conditions
def time_meditating_per_day : ℕ := 1
def time_reading_per_day : ℕ := 2 * time_meditating_per_day
def days_in_week : ℕ := 7

-- Define the target property to prove
theorem time_reading_per_week : time_reading_per_day * days_in_week = 14 :=
by
  sorry

end time_reading_per_week_l815_81500


namespace car_b_speed_l815_81586

noncomputable def SpeedOfCarB (Speed_A Time_A Time_B d_ratio: ℝ) : ℝ :=
  let Distance_A := Speed_A * Time_A
  let Distance_B := Distance_A / d_ratio
  Distance_B / Time_B

theorem car_b_speed
  (Speed_A : ℝ) (Time_A : ℝ) (Time_B : ℝ) (d_ratio : ℝ)
  (h1 : Speed_A = 70) (h2 : Time_A = 10) (h3 : Time_B = 10) (h4 : d_ratio = 2) :
  SpeedOfCarB Speed_A Time_A Time_B d_ratio = 35 :=
by
  sorry

end car_b_speed_l815_81586


namespace quarters_value_percentage_l815_81543

theorem quarters_value_percentage (dimes_count quarters_count dimes_value quarters_value : ℕ) (h1 : dimes_count = 75)
    (h2 : quarters_count = 30) (h3 : dimes_value = 10) (h4 : quarters_value = 25) :
    (quarters_count * quarters_value * 100) / (dimes_count * dimes_value + quarters_count * quarters_value) = 50 := 
by
    sorry

end quarters_value_percentage_l815_81543


namespace num_tosses_l815_81571

theorem num_tosses (n : ℕ) (h : (1 - (7 / 8 : ℝ)^n) = 0.111328125) : n = 7 :=
by
  sorry

end num_tosses_l815_81571


namespace leftover_balls_when_placing_60_in_tetrahedral_stack_l815_81549

def tetrahedral_number (n : ℕ) : ℕ :=
  n * (n + 1) * (n + 2) / 6

/--
  When placing 60 balls in a tetrahedral stack, the number of leftover balls is 4.
-/
theorem leftover_balls_when_placing_60_in_tetrahedral_stack :
  ∃ n, tetrahedral_number n ≤ 60 ∧ 60 - tetrahedral_number n = 4 := by
  sorry

end leftover_balls_when_placing_60_in_tetrahedral_stack_l815_81549


namespace probability_red_or_blue_marbles_l815_81584

theorem probability_red_or_blue_marbles (red blue green total : ℕ) (h_red : red = 4) (h_blue : blue = 3) (h_green : green = 6) (h_total : total = red + blue + green) :
  (red + blue) / total = 7 / 13 :=
by
  sorry

end probability_red_or_blue_marbles_l815_81584


namespace cranberries_left_l815_81514

def initial_cranberries : ℕ := 60000
def harvested_by_humans : ℕ := initial_cranberries * 40 / 100
def eaten_by_elk : ℕ := 20000

theorem cranberries_left (c : ℕ) : c = initial_cranberries - harvested_by_humans - eaten_by_elk → c = 16000 := by
  sorry

end cranberries_left_l815_81514


namespace flour_in_cupboard_l815_81528

theorem flour_in_cupboard :
  let flour_on_counter := 100
  let flour_in_pantry := 100
  let flour_per_loaf := 200
  let loaves := 2
  let total_flour_needed := loaves * flour_per_loaf
  let flour_outside_cupboard := flour_on_counter + flour_in_pantry
  let flour_in_cupboard := total_flour_needed - flour_outside_cupboard
  flour_in_cupboard = 200 :=
by
  sorry

end flour_in_cupboard_l815_81528


namespace max_angle_OAB_l815_81573

/-- Let OA = a, OB = b, and OM = x on the right angle XOY, where a < b. 
    The value of x which maximizes the angle ∠AMB is sqrt(ab). -/
theorem max_angle_OAB (a b x : ℝ) (h : a < b) (h1 : x = Real.sqrt (a * b)) :
  x = Real.sqrt (a * b) :=
sorry

end max_angle_OAB_l815_81573


namespace part_I_part_II_l815_81577

-- Part (I)
theorem part_I (x a : ℝ) (h_a : a = 3) (h : abs (x - a) + abs (x + 5) ≥ 2 * abs (x + 5)) : x ≤ -1 := 
sorry

-- Part (II)
theorem part_II (a : ℝ) (h : ∀ x : ℝ, abs (x - a) + abs (x + 5) ≥ 6) : a ≥ 1 ∨ a ≤ -11 := 
sorry

end part_I_part_II_l815_81577


namespace find_number_l815_81572

theorem find_number (x : ℝ) :
  0.15 * x = 0.25 * 16 + 2 → x = 40 :=
by
  -- skipping the proof steps
  sorry

end find_number_l815_81572


namespace comparison_of_powers_l815_81508

theorem comparison_of_powers : 6 ^ 0.7 > 0.7 ^ 6 ∧ 0.7 ^ 6 > 0.6 ^ 7 := by
  sorry

end comparison_of_powers_l815_81508


namespace lashawn_three_times_kymbrea_l815_81503

-- Definitions based on the conditions
def kymbrea_collection (months : ℕ) : ℕ := 50 + 3 * months
def lashawn_collection (months : ℕ) : ℕ := 20 + 5 * months

-- Theorem stating the core of the problem
theorem lashawn_three_times_kymbrea (x : ℕ) 
  (h : lashawn_collection x = 3 * kymbrea_collection x) : x = 33 := 
sorry

end lashawn_three_times_kymbrea_l815_81503


namespace probability_two_girls_l815_81587

-- Define the conditions
def total_students := 8
def total_girls := 5
def total_boys := 3
def choose_two_from_n (n : ℕ) := n * (n - 1) / 2

-- Define the question as a statement that the probability equals 5/14
theorem probability_two_girls
    (h1 : choose_two_from_n total_students = 28)
    (h2 : choose_two_from_n total_girls = 10) :
    (choose_two_from_n total_girls : ℚ) / choose_two_from_n total_students = 5 / 14 :=
by
  sorry

end probability_two_girls_l815_81587


namespace call_processing_ratio_l815_81524

variables (A B C : ℝ)
variable (total_calls : ℝ)
variable (calls_processed_by_A_per_member calls_processed_by_B_per_member : ℝ)

-- Given conditions
def team_A_agents_ratio : Prop := A = (5 / 8) * B
def team_B_calls_ratio : Prop := calls_processed_by_B_per_member * B = (4 / 7) * total_calls
def team_A_calls_ratio : Prop := calls_processed_by_A_per_member * A = (3 / 7) * total_calls

-- Proving the ratio of calls processed by each member
theorem call_processing_ratio
    (hA : team_A_agents_ratio A B)
    (hB_calls : team_B_calls_ratio B total_calls calls_processed_by_B_per_member)
    (hA_calls : team_A_calls_ratio A total_calls calls_processed_by_A_per_member) :
  calls_processed_by_A_per_member / calls_processed_by_B_per_member = 6 / 5 :=
by
  sorry

end call_processing_ratio_l815_81524


namespace price_per_liter_after_discount_l815_81525

-- Define the initial conditions
def num_bottles : ℕ := 6
def liters_per_bottle : ℝ := 2
def original_total_cost : ℝ := 15
def discounted_total_cost : ℝ := 12

-- Calculate the total number of liters
def total_liters : ℝ := num_bottles * liters_per_bottle

-- Define the expected price per liter after discount
def expected_price_per_liter : ℝ := 1

-- Lean query to verify the expected price per liter
theorem price_per_liter_after_discount : (discounted_total_cost / total_liters) = expected_price_per_liter := by
  sorry

end price_per_liter_after_discount_l815_81525


namespace Kiran_money_l815_81517

theorem Kiran_money (R G K : ℕ) (h1 : R / G = 6 / 7) (h2 : G / K = 6 / 15) (h3 : R = 36) : K = 105 := by
  sorry

end Kiran_money_l815_81517


namespace total_packages_l815_81558

theorem total_packages (num_trucks : ℕ) (packages_per_truck : ℕ) (h1 : num_trucks = 7) (h2 : packages_per_truck = 70) : num_trucks * packages_per_truck = 490 := by
  sorry

end total_packages_l815_81558


namespace brianne_savings_ratio_l815_81534

theorem brianne_savings_ratio
  (r : ℝ)
  (H1 : 10 * r^4 = 160) :
  r = 2 :=
by 
  sorry

end brianne_savings_ratio_l815_81534


namespace percentage_increase_l815_81515

theorem percentage_increase
  (W R : ℝ)
  (H1 : 0.70 * R = 1.04999999999999982 * W) :
  (R - W) / W * 100 = 50 :=
by
  sorry

end percentage_increase_l815_81515


namespace third_dog_average_daily_miles_l815_81535

/-- Bingo has three dogs. On average, they walk a total of 100 miles a week.

    The first dog walks an average of 2 miles a day.

    The second dog walks 1 mile if it is an odd day of the month and 3 miles if it is an even day of the month.

    Considering a 30-day month, the goal is to find the average daily miles of the third dog. -/
theorem third_dog_average_daily_miles :
  let total_dogs := 3
  let weekly_total_miles := 100
  let first_dog_daily_miles := 2
  let second_dog_odd_day_miles := 1
  let second_dog_even_day_miles := 3
  let days_in_month := 30
  let odd_days_in_month := 15
  let even_days_in_month := 15
  let weeks_in_month := days_in_month / 7
  let first_dog_monthly_miles := days_in_month * first_dog_daily_miles
  let second_dog_monthly_miles := (second_dog_odd_day_miles * odd_days_in_month) + (second_dog_even_day_miles * even_days_in_month)
  let third_dog_monthly_miles := (weekly_total_miles * weeks_in_month) - (first_dog_monthly_miles + second_dog_monthly_miles)
  let third_dog_daily_miles := third_dog_monthly_miles / days_in_month
  third_dog_daily_miles = 10.33 :=
by
  sorry

end third_dog_average_daily_miles_l815_81535


namespace intersection_complement_eq_l815_81530

open Set

universe u

def U : Set ℝ := univ

def A : Set ℝ := { x | x < 0 }

def B : Set ℝ := { x | x ≤ -1 }

theorem intersection_complement_eq : A ∩ (U \ B) = { x | -1 < x ∧ x < 0 } :=
by
  sorry

end intersection_complement_eq_l815_81530


namespace price_per_glass_first_day_l815_81513

theorem price_per_glass_first_day
    (O W : ℝ) (P1 P2 : ℝ)
    (h1 : O = W)
    (h2 : P2 = 0.40)
    (h3 : 2 * O * P1 = 3 * O * P2) :
    P1 = 0.60 :=
by
    sorry

end price_per_glass_first_day_l815_81513


namespace number_of_true_propositions_l815_81548

-- Define the original condition
def original_proposition (a b : ℝ) : Prop := (a + b = 1) → (a * b ≤ 1 / 4)

-- Define contrapositive
def contrapositive (a b : ℝ) : Prop := (a * b > 1 / 4) → (a + b ≠ 1)

-- Define inverse
def inverse (a b : ℝ) : Prop := (a * b ≤ 1 / 4) → (a + b = 1)

-- Define converse
def converse (a b : ℝ) : Prop := (a + b ≠ 1) → (a * b > 1 / 4)

-- State the problem
theorem number_of_true_propositions (a b : ℝ) :
  (original_proposition a b ∧ contrapositive a b ∧ ¬inverse a b ∧ ¬converse a b) → 
  (∃ n : ℕ, n = 1) :=
by sorry

end number_of_true_propositions_l815_81548


namespace sixty_percent_of_40_greater_than_four_fifths_of_25_l815_81502

theorem sixty_percent_of_40_greater_than_four_fifths_of_25 :
  let x := (60 / 100 : ℝ) * 40
  let y := (4 / 5 : ℝ) * 25
  x - y = 4 :=
by
  sorry

end sixty_percent_of_40_greater_than_four_fifths_of_25_l815_81502


namespace Jamie_needs_to_climb_40_rungs_l815_81523

-- Define the conditions
def height_of_new_tree : ℕ := 20
def rungs_climbed_previous : ℕ := 12
def height_of_previous_tree : ℕ := 6
def rungs_per_foot := rungs_climbed_previous / height_of_previous_tree

-- Define the theorem
theorem Jamie_needs_to_climb_40_rungs :
  height_of_new_tree * rungs_per_foot = 40 :=
by
  -- Proof placeholder
  sorry

end Jamie_needs_to_climb_40_rungs_l815_81523


namespace marie_initial_erasers_l815_81579

def erasers_problem : Prop :=
  ∃ initial_erasers : ℝ, initial_erasers + 42.0 = 137

theorem marie_initial_erasers : erasers_problem :=
  sorry

end marie_initial_erasers_l815_81579


namespace max_value_of_expression_l815_81564

theorem max_value_of_expression (x y : ℝ) (h : 2 * x^2 - 6 * x + y^2 = 0) : 
  ∃ m, m = 15 ∧ x^2 + y^2 + 2 * x ≤ m := 
sorry

end max_value_of_expression_l815_81564


namespace prime_pairs_perfect_square_l815_81522

theorem prime_pairs_perfect_square (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  ∃ k : ℕ, p^(q-1) + q^(p-1) = k^2 ↔ (p = 2 ∧ q = 2) :=
by
  sorry

end prime_pairs_perfect_square_l815_81522


namespace find_digits_l815_81576

theorem find_digits (a b : ℕ) (h1 : (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9)) :
  (∃ (c : ℕ), 10000 * a + 6790 + b = 72 * c) ↔ (a = 3 ∧ b = 2) :=
by
  sorry

end find_digits_l815_81576


namespace min_value_m_plus_2n_exists_min_value_l815_81559

variable (n : ℝ) -- Declare n as a real number.

-- Define m in terms of n
def m (n : ℝ) : ℝ := n^2

-- State and prove that the minimum value of m + 2n is -1
theorem min_value_m_plus_2n : (m n + 2 * n) ≥ -1 :=
by sorry

-- Show there exists an n such that m + 2n = -1
theorem exists_min_value : ∃ n : ℝ, m n + 2 * n = -1 :=
by sorry

end min_value_m_plus_2n_exists_min_value_l815_81559


namespace arithmetic_sequence_properties_l815_81567

theorem arithmetic_sequence_properties (a b c : ℝ) (h1 : ∃ d : ℝ, [2, a, b, c, 9] = [2, 2 + d, 2 + 2 * d, 2 + 3 * d, 2 + 4 * d]) : 
  c - a = 7 / 2 := 
by
  -- We assume the proof here
  sorry

end arithmetic_sequence_properties_l815_81567


namespace sum_remainders_mod_13_l815_81538

theorem sum_remainders_mod_13 :
  ∀ (a b c d e : ℕ),
  a % 13 = 3 →
  b % 13 = 5 →
  c % 13 = 7 →
  d % 13 = 9 →
  e % 13 = 11 →
  (a + b + c + d + e) % 13 = 9 :=
by
  intros a b c d e ha hb hc hd he
  sorry

end sum_remainders_mod_13_l815_81538


namespace rational_sum_is_negative_then_at_most_one_positive_l815_81540

theorem rational_sum_is_negative_then_at_most_one_positive (a b : ℚ) (h : a + b < 0) :
  (a > 0 ∧ b ≤ 0) ∨ (a ≤ 0 ∧ b > 0) ∨ (a ≤ 0 ∧ b ≤ 0) :=
by
  sorry

end rational_sum_is_negative_then_at_most_one_positive_l815_81540


namespace problem_l815_81507

-- Definitions and conditions
def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ (∀ n, 2 ≤ n → 2 * a n / (a n * (Finset.sum (Finset.range n) a) - (Finset.sum (Finset.range n) a) ^ 2) = 1)

-- Sum of the first n terms
def S (a : ℕ → ℚ) (n : ℕ) : ℚ := Finset.sum (Finset.range n) a

-- The proof statement
theorem problem (a : ℕ → ℚ) (h : seq a) : S a 2017 = 1 / 1009 := sorry

end problem_l815_81507


namespace groups_partition_count_l815_81594

-- Definitions based on the conditions
def num_dogs : ℕ := 12
def group1_size : ℕ := 4
def group2_size : ℕ := 6
def group3_size : ℕ := 2

-- Given specific names for groups based on problem statement
def Fluffy_group_size : ℕ := group1_size
def Nipper_group_size : ℕ := group2_size

-- The total number of ways to form the groups given the conditions
def total_ways (n k : ℕ) : ℕ := Nat.choose n k

-- Statement of the problem
theorem groups_partition_count :
  total_ways 10 3 * total_ways 7 5 = 2520 := sorry

end groups_partition_count_l815_81594


namespace cone_bead_path_l815_81545

theorem cone_bead_path (r h : ℝ) (h_sqrt : h / r = 3 * Real.sqrt 11) : 3 + 11 = 14 := by
  sorry

end cone_bead_path_l815_81545


namespace hyperbola_asymptotes_l815_81575

open Real

noncomputable def hyperbola (x y m : ℝ) : Prop := (x^2 / 9) - (y^2 / m) = 1

noncomputable def on_line (x y : ℝ) : Prop := x + y = 5

theorem hyperbola_asymptotes (m : ℝ) (hm : 9 + m = 25) :
    (∃ x y : ℝ, hyperbola x y m ∧ on_line x y) →
    (∀ x : ℝ, on_line x ((4 / 3) * x) ∧ on_line x (-(4 / 3) * x)) :=
by
  sorry

end hyperbola_asymptotes_l815_81575


namespace chlorine_needed_l815_81578

variable (Methane moles_HCl moles_Cl₂ : ℕ)

-- Given conditions
def reaction_started_with_one_mole_of_methane : Prop :=
  Methane = 1

def reaction_produces_two_moles_of_HCl : Prop :=
  moles_HCl = 2

-- Question to be proved
def number_of_moles_of_Chlorine_combined : Prop :=
  moles_Cl₂ = 2

theorem chlorine_needed
  (h1 : reaction_started_with_one_mole_of_methane Methane)
  (h2 : reaction_produces_two_moles_of_HCl moles_HCl)
  : number_of_moles_of_Chlorine_combined moles_Cl₂ :=
sorry

end chlorine_needed_l815_81578


namespace probability_of_circle_in_square_l815_81516

open Real Set

theorem probability_of_circle_in_square :
  ∃ (p : ℝ), (∀ x y : ℝ, x ∈ Icc (-1 : ℝ) 1 → y ∈ Icc (-1 : ℝ) 1 → (x^2 + y^2 < 1/4) → True)
  → p = π / 16 :=
by
  use π / 16
  sorry

end probability_of_circle_in_square_l815_81516


namespace napkin_coloring_l815_81553

structure Napkin where
  top : ℝ
  bottom : ℝ
  left : ℝ
  right : ℝ

def intersects_vertically (n1 n2 : Napkin) : Prop :=
  n1.left ≤ n2.right ∧ n2.left ≤ n1.right

def intersects_horizontally (n1 n2 : Napkin) : Prop :=
  n1.bottom ≤ n2.top ∧ n2.bottom ≤ n1.top

def can_be_crossed_by_line (n1 n2 : Napkin) : Prop :=
  intersects_vertically n1 n2 ∨ intersects_horizontally n1 n2

theorem napkin_coloring
  (blue_napkins green_napkins : List Napkin)
  (h_cross : ∀ (b : Napkin) (g : Napkin), 
    b ∈ blue_napkins → g ∈ green_napkins → can_be_crossed_by_line b g) :
  ∃ (color : String) (h1 h2 : ℝ) (v : ℝ), 
    (color = "blue" ∧ ∀ b ∈ blue_napkins, (b.bottom ≤ h1 ∧ h1 ≤ b.top) ∨ (b.bottom ≤ h2 ∧ h2 ≤ b.top) ∨ (b.left ≤ v ∧ v ≤ b.right)) ∨
    (color = "green" ∧ ∀ g ∈ green_napkins, (g.bottom ≤ h1 ∧ h1 ≤ g.top) ∨ (g.bottom ≤ h2 ∧ h2 ≤ g.top) ∨ (g.left ≤ v ∧ v ≤ g.right)) :=
sorry

end napkin_coloring_l815_81553


namespace has_two_zeros_of_f_l815_81511

noncomputable def f (x a : ℝ) : ℝ := (x + 1) * Real.exp x - a

theorem has_two_zeros_of_f (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ (-1 / Real.exp 2 < a ∧ a < 0) := by
sorry

end has_two_zeros_of_f_l815_81511


namespace fraction_halfway_between_l815_81537

theorem fraction_halfway_between (a b : ℚ) (h₁ : a = 1 / 6) (h₂ : b = 2 / 5) : (a + b) / 2 = 17 / 60 :=
by {
  sorry
}

end fraction_halfway_between_l815_81537


namespace shaded_figure_perimeter_l815_81547

theorem shaded_figure_perimeter (a b : ℝ) (area_overlap : ℝ) (side_length : ℝ) (side_length_overlap : ℝ):
    a = 5 → b = 5 → area_overlap = 4 → side_length_overlap * side_length_overlap = area_overlap →
    side_length_overlap = 2 →
    ((4 * a) + (4 * b) - (4 * side_length_overlap)) = 32 :=
by
  intros
  sorry

end shaded_figure_perimeter_l815_81547


namespace inequality_transpose_l815_81599

variable (a b : ℝ)

theorem inequality_transpose (h : a < b) (hab : b < 0) : (1 / a) > (1 / b) := by
  sorry

end inequality_transpose_l815_81599


namespace polygon_E_largest_area_l815_81593

def unit_square_area : ℕ := 1
def right_triangle_area : ℚ := 1 / 2
def rectangle_area : ℕ := 2

def polygon_A_area : ℚ := 3 * unit_square_area + 2 * right_triangle_area
def polygon_B_area : ℚ := 2 * unit_square_area + 4 * right_triangle_area
def polygon_C_area : ℚ := 4 * unit_square_area + 1 * rectangle_area
def polygon_D_area : ℚ := 3 * rectangle_area
def polygon_E_area : ℚ := 2 * unit_square_area + 2 * right_triangle_area + 2 * rectangle_area

theorem polygon_E_largest_area :
  polygon_E_area = max polygon_A_area (max polygon_B_area (max polygon_C_area (max polygon_D_area polygon_E_area))) := by
  sorry

end polygon_E_largest_area_l815_81593


namespace geometric_sequence_seventh_term_l815_81597

noncomputable def a_7 (a₁ q : ℝ) : ℝ :=
  a₁ * q^6

theorem geometric_sequence_seventh_term :
  a_7 3 (Real.sqrt 2) = 24 :=
by
  sorry

end geometric_sequence_seventh_term_l815_81597


namespace window_dimensions_l815_81504

-- Given conditions
def panes := 12
def rows := 3
def columns := 4
def height_to_width_ratio := 3
def border_width := 2

-- Definitions based on given conditions
def width_per_pane (x : ℝ) := x
def height_per_pane (x : ℝ) := 3 * x

def total_width (x : ℝ) := columns * width_per_pane x + (columns + 1) * border_width
def total_height (x : ℝ) := rows * height_per_pane x + (rows + 1) * border_width

-- Theorem statement: width and height of the window
theorem window_dimensions (x : ℝ) : 
  total_width x = 4 * x + 10 ∧ 
  total_height x = 9 * x + 8 := by
  sorry

end window_dimensions_l815_81504


namespace yura_picture_dimensions_l815_81546

-- Definitions based on the problem conditions
variable {a b : ℕ} -- dimensions of the picture
variable (hasFrame : ℕ × ℕ → Prop) -- definition sketch

-- The main statement to prove
theorem yura_picture_dimensions (h : (a + 2) * (b + 2) - a * b = 2 * a * b) :
  (a = 3 ∧ b = 10) ∨ (a = 10 ∧ b = 3) ∨ (a = 4 ∧ b = 6) ∨ (a = 6 ∧ b = 4) :=
  sorry

end yura_picture_dimensions_l815_81546


namespace annie_building_time_l815_81580

theorem annie_building_time (b p : ℕ) (h1 : b = 3 * p - 5) (h2 : b + p = 67) : b = 49 :=
by
  sorry

end annie_building_time_l815_81580


namespace horizontal_length_tv_screen_l815_81520

theorem horizontal_length_tv_screen : 
  ∀ (a b : ℝ), (a / b = 4 / 3) → (a ^ 2 + b ^ 2 = 27 ^ 2) → a = 21.5 := 
by 
  sorry

end horizontal_length_tv_screen_l815_81520


namespace problem_statement_l815_81561

theorem problem_statement (x : ℝ) (h : x^2 + 3 * x - 1 = 0) : x^3 + 5 * x^2 + 5 * x + 18 = 20 :=
by
  sorry

end problem_statement_l815_81561


namespace correct_answer_l815_81532

variable (x : ℝ)

theorem correct_answer : {x : ℝ | x^2 + 2*x + 1 = 0} = {-1} :=
by sorry -- the actual proof is not required, just the statement

end correct_answer_l815_81532


namespace calculate_fraction_value_l815_81569

theorem calculate_fraction_value :
  1 + 1 / (1 + 1 / (1 + 1 / (1 + 2))) = 11 / 7 := 
  sorry

end calculate_fraction_value_l815_81569


namespace parallel_lines_l815_81536

def line1 (x : ℝ) : ℝ := 5 * x + 3
def line2 (x k : ℝ) : ℝ := 3 * k * x + 7

theorem parallel_lines (k : ℝ) : (∀ x : ℝ, line1 x = line2 x k) → k = 5 / 3 := 
by
  intros h_parallel
  sorry

end parallel_lines_l815_81536


namespace arithmetic_progression_impossible_geometric_progression_possible_l815_81556

theorem arithmetic_progression_impossible (a b c : ℝ) (h1 : a = 2) (h2 : b = Real.sqrt 6) (h3 : c = 4.5) : 
  2 * b ≠ a + c :=
by {
    sorry
}

theorem geometric_progression_possible (a b c : ℝ) (h1 : a = 2) (h2 : b = Real.sqrt 6) (h3 : c = 4.5) : 
  ∃ r m : ℤ, (b / a)^r = (c / a)^m :=
by {
    sorry
}

end arithmetic_progression_impossible_geometric_progression_possible_l815_81556


namespace rectangle_length_l815_81526

theorem rectangle_length (P W : ℝ) (hP : P = 40) (hW : W = 8) : ∃ L : ℝ, 2 * (L + W) = P ∧ L = 12 := 
by 
  sorry

end rectangle_length_l815_81526


namespace boat_travel_distance_downstream_l815_81506

-- Define the conditions given in the problem
def speed_boat_still_water := 22 -- in km/hr
def speed_stream := 5 -- in km/hr
def time_downstream := 2 -- in hours

-- Define a function to compute the effective speed downstream
def effective_speed_downstream (speed_boat: ℝ) (speed_stream: ℝ) : ℝ :=
  speed_boat + speed_stream

-- Define a function to compute the distance travelled downstream
def distance_downstream (speed: ℝ) (time: ℝ) : ℝ :=
  speed * time

-- The main theorem to prove
theorem boat_travel_distance_downstream :
  distance_downstream (effective_speed_downstream speed_boat_still_water speed_stream) time_downstream = 54 :=
by
  -- Proof is to be filled in later
  sorry

end boat_travel_distance_downstream_l815_81506


namespace greatest_multiple_of_5_and_7_less_than_800_l815_81539

theorem greatest_multiple_of_5_and_7_less_than_800 : 
    ∀ n : ℕ, (n < 800 ∧ 35 ∣ n) → n ≤ 770 := 
by
  -- Proof steps go here
  sorry

end greatest_multiple_of_5_and_7_less_than_800_l815_81539


namespace oprq_possible_figures_l815_81574

theorem oprq_possible_figures (x1 y1 x2 y2 : ℝ) (h : (x1, y1) ≠ (x2, y2)) : 
  -- Define the points P, Q, and R
  let P := (x1, y1)
  let Q := (x2, y2)
  let R := (x1 - x2, y1 - y2)
  -- Proving the geometric possibilities
  (∃ k : ℝ, x1 = k * x2 ∧ y1 = k * y2) ∨
  -- When the points are collinear
  ((x1 + x2, y1 + y2) = (x1, y1)) :=
sorry

end oprq_possible_figures_l815_81574


namespace polynomial_divisible_2520_l815_81581

theorem polynomial_divisible_2520 (n : ℕ) : (n^7 - 14 * n^5 + 49 * n^3 - 36 * n) % 2520 = 0 := 
sorry

end polynomial_divisible_2520_l815_81581


namespace manufacturing_section_degrees_l815_81591

theorem manufacturing_section_degrees (percentage : ℝ) (total_degrees : ℝ) (h1 : total_degrees = 360) (h2 : percentage = 35) : 
  ((percentage / 100) * total_degrees) = 126 :=
by
  sorry

end manufacturing_section_degrees_l815_81591


namespace john_alone_finishes_in_48_days_l815_81568

theorem john_alone_finishes_in_48_days (J R : ℝ) (h1 : J + R = 1 / 24)
  (h2 : 16 * (J + R) = 16 / 24) (h3 : ∀ T : ℝ, J * T = 1 → T = 48) : 
  (J = 1 / 48) → (∀ T : ℝ, J * T = 1 → T = 48) :=
by
  intro hJohn
  sorry

end john_alone_finishes_in_48_days_l815_81568


namespace constant_expression_l815_81551

theorem constant_expression 
  (x y : ℝ) 
  (h₁ : x + y = 1) 
  (h₂ : x ≠ 1) 
  (h₃ : y ≠ 1) : 
  (x / (y^3 - 1) + y / (1 - x^3) + 2 * (x - y) / (x^2 * y^2 + 3)) = 0 :=
by 
  sorry

end constant_expression_l815_81551


namespace total_weight_of_rice_l815_81518

theorem total_weight_of_rice :
  (29 * 4) / 16 = 7.25 := by
sorry

end total_weight_of_rice_l815_81518


namespace probability_of_selecting_one_is_correct_l815_81544

-- Define the number of elements in the first 20 rows of Pascal's triangle
def totalElementsInPascalFirst20Rows : ℕ := 210

-- Define the number of ones in the first 20 rows of Pascal's triangle
def totalOnesInPascalFirst20Rows : ℕ := 39

-- The probability as a rational number
def probabilityOfSelectingOne : ℚ := totalOnesInPascalFirst20Rows / totalElementsInPascalFirst20Rows

theorem probability_of_selecting_one_is_correct :
  probabilityOfSelectingOne = 13 / 70 :=
by
  -- Proof is omitted
  sorry

end probability_of_selecting_one_is_correct_l815_81544


namespace lowest_cost_per_ton_l815_81585

-- Define the conditions given in the problem statement
variable (x : ℝ) (y : ℝ)

-- Define the annual production range
def production_range (x : ℝ) : Prop := x ≥ 150 ∧ x ≤ 250

-- Define the relationship between total annual production cost and annual production
def production_cost_relation (x y : ℝ) : Prop := y = (x^2 / 10) - 30 * x + 4000

-- State the main theorem: the annual production when the cost per ton is the lowest is 200 tons
theorem lowest_cost_per_ton (x : ℝ) (y : ℝ) (h1 : production_range x) (h2 : production_cost_relation x y) : x = 200 :=
sorry

end lowest_cost_per_ton_l815_81585


namespace weights_less_than_90_l815_81521

variable (a b c : ℝ)
-- conditions
axiom h1 : a + b = 100
axiom h2 : a + c = 101
axiom h3 : b + c = 102

theorem weights_less_than_90 (a b c : ℝ) (h1 : a + b = 100) (h2 : a + c = 101) (h3 : b + c = 102) : a < 90 ∧ b < 90 ∧ c < 90 := 
by sorry

end weights_less_than_90_l815_81521
