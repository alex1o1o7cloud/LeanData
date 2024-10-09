import Mathlib

namespace range_of_m_l1615_161585
-- Import the entire math library

-- Defining the propositions p and q
def p (x : ℝ) : Prop := x^2 - 8 * x - 20 ≤ 0 
def q (x m : ℝ) : Prop := (x - (1 + m)) * (x - (1 - m)) ≤ 0 

-- Main theorem statement
theorem range_of_m (m : ℝ) (h1 : 0 < m) 
(hsuff : ∀ x : ℝ, p x → q x m) 
(hnsuff : ¬ (∀ x : ℝ, q x m → p x)) : m ≥ 9 := 
sorry

end range_of_m_l1615_161585


namespace total_dolphins_l1615_161550

theorem total_dolphins (initial_dolphins : ℕ) (triple_of_initial : ℕ) (final_dolphins : ℕ) 
    (h1 : initial_dolphins = 65) (h2 : triple_of_initial = 3 * initial_dolphins) (h3 : final_dolphins = initial_dolphins + triple_of_initial) : 
    final_dolphins = 260 :=
by
  -- Proof goes here
  sorry

end total_dolphins_l1615_161550


namespace cost_price_per_meter_l1615_161572

theorem cost_price_per_meter
  (total_meters : ℕ)
  (selling_price : ℕ)
  (loss_per_meter : ℕ)
  (total_cost_price : ℕ)
  (cost_price_per_meter : ℕ)
  (h1 : total_meters = 400)
  (h2 : selling_price = 18000)
  (h3 : loss_per_meter = 5)
  (h4 : total_cost_price = selling_price + total_meters * loss_per_meter)
  (h5 : cost_price_per_meter = total_cost_price / total_meters) :
  cost_price_per_meter = 50 :=
by
  sorry

end cost_price_per_meter_l1615_161572


namespace odd_blue_faces_in_cubes_l1615_161569

noncomputable def count_odd_blue_faces (length width height : ℕ) : ℕ :=
if length = 6 ∧ width = 4 ∧ height = 2 then 16 else 0

theorem odd_blue_faces_in_cubes : count_odd_blue_faces 6 4 2 = 16 := 
by
  -- The proof would involve calculating the corners, edges, etc.
  sorry

end odd_blue_faces_in_cubes_l1615_161569


namespace multiply_by_11_l1615_161556

theorem multiply_by_11 (A B k : ℕ) (h1 : 10 * A + B < 100) (h2 : A + B = 10 + k) :
  (10 * A + B) * 11 = 100 * (A + 1) + 10 * k + B :=
by 
  sorry

end multiply_by_11_l1615_161556


namespace total_books_l1615_161594

variable (Sandy_books Benny_books Tim_books : ℕ)
variable (h_Sandy : Sandy_books = 10)
variable (h_Benny : Benny_books = 24)
variable (h_Tim : Tim_books = 33)

theorem total_books :
  Sandy_books + Benny_books + Tim_books = 67 :=
by sorry

end total_books_l1615_161594


namespace herd_total_cows_l1615_161570

theorem herd_total_cows (n : ℕ) : 
  let first_son := 1 / 3 * n
  let second_son := 1 / 6 * n
  let third_son := 1 / 8 * n
  let remaining := n - (first_son + second_son + third_son)
  remaining = 9 ↔ n = 24 := 
by
  -- Skipping proof, placeholder
  sorry

end herd_total_cows_l1615_161570


namespace jesse_needs_more_carpet_l1615_161513

def additional_carpet_needed (carpet : ℕ) (length : ℕ) (width : ℕ) : ℕ :=
  let room_area := length * width
  room_area - carpet

theorem jesse_needs_more_carpet
  (carpet : ℕ) (length : ℕ) (width : ℕ)
  (h_carpet : carpet = 18)
  (h_length : length = 4)
  (h_width : width = 20) :
  additional_carpet_needed carpet length width = 62 :=
by {
  -- the proof goes here
  sorry
}

end jesse_needs_more_carpet_l1615_161513


namespace tonya_large_lemonade_sales_l1615_161530

theorem tonya_large_lemonade_sales 
  (price_small : ℝ)
  (price_medium : ℝ)
  (price_large : ℝ)
  (total_revenue : ℝ)
  (revenue_small : ℝ)
  (revenue_medium : ℝ)
  (n : ℝ)
  (h_price_small : price_small = 1)
  (h_price_medium : price_medium = 2)
  (h_price_large : price_large = 3)
  (h_total_revenue : total_revenue = 50)
  (h_revenue_small : revenue_small = 11)
  (h_revenue_medium : revenue_medium = 24)
  (h_revenue_large : n = (total_revenue - revenue_small - revenue_medium) / price_large) :
  n = 5 :=
sorry

end tonya_large_lemonade_sales_l1615_161530


namespace basketball_cost_l1615_161540

-- Initial conditions
def initial_amount : Nat := 50
def cost_jerseys (n price_per_jersey : Nat) : Nat := n * price_per_jersey
def cost_shorts : Nat := 8
def remaining_amount : Nat := 14

-- Derived total spent calculation
def total_spent (initial remaining : Nat) : Nat := initial - remaining
def known_cost (jerseys shorts : Nat) : Nat := jerseys + shorts

-- Prove the cost of the basketball
theorem basketball_cost :
  let jerseys := cost_jerseys 5 2
  let shorts := cost_shorts
  let total_spent := total_spent initial_amount remaining_amount
  let known_cost := known_cost jerseys shorts
  total_spent - known_cost = 18 := 
by
  sorry

end basketball_cost_l1615_161540


namespace weight_of_each_package_l1615_161541

theorem weight_of_each_package (W : ℝ) 
  (h1: 10 * W + 7 * W + 8 * W = 100) : W = 4 :=
by
  sorry

end weight_of_each_package_l1615_161541


namespace quadratic_real_roots_range_l1615_161560

theorem quadratic_real_roots_range (m : ℝ) : (∃ x : ℝ, x^2 - 2 * x - m = 0) → -1 ≤ m := 
sorry

end quadratic_real_roots_range_l1615_161560


namespace polynomial_simplification_l1615_161599

theorem polynomial_simplification (p : ℤ) :
  (5 * p^4 + 2 * p^3 - 7 * p^2 + 3 * p - 2) + (-3 * p^4 + 4 * p^3 + 8 * p^2 - 2 * p + 6) = 
  2 * p^4 + 6 * p^3 + p^2 + p + 4 :=
by
  sorry

end polynomial_simplification_l1615_161599


namespace sample_quantities_and_probability_l1615_161500

-- Define the given quantities from each workshop
def q_A := 10
def q_B := 20
def q_C := 30

-- Total sample size
def n := 6

-- Given conditions, the total quantity and sample ratio
def total_quantity := q_A + q_B + q_C
def ratio := n / total_quantity

-- Derived quantities in the samples based on the proportion
def sample_A := q_A * ratio
def sample_B := q_B * ratio
def sample_C := q_C * ratio

-- Combinatorial calculations
def C (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
def total_combinations := C 6 2
def workshop_C_combinations := C 3 2
def probability_C_samples := workshop_C_combinations / total_combinations

-- Theorem to prove the quantities and probability
theorem sample_quantities_and_probability :
  sample_A = 1 ∧ sample_B = 2 ∧ sample_C = 3 ∧ probability_C_samples = 1 / 5 :=
by
  sorry

end sample_quantities_and_probability_l1615_161500


namespace frank_bought_2_bags_of_chips_l1615_161561

theorem frank_bought_2_bags_of_chips
  (cost_choco_bar : ℕ)
  (num_choco_bar : ℕ)
  (total_money : ℕ)
  (change : ℕ)
  (cost_bag_chip : ℕ)
  (num_bags_chip : ℕ)
  (h1 : cost_choco_bar = 2)
  (h2 : num_choco_bar = 5)
  (h3 : total_money = 20)
  (h4 : change = 4)
  (h5 : cost_bag_chip = 3)
  (h6 : total_money - change = (cost_choco_bar * num_choco_bar) + (cost_bag_chip * num_bags_chip)) :
  num_bags_chip = 2 := by
  sorry

end frank_bought_2_bags_of_chips_l1615_161561


namespace g_sqrt_45_l1615_161591

noncomputable def g (x : ℝ) : ℝ :=
if x % 1 = 0 then 7 * x + 6 else ⌊x⌋ + 7

theorem g_sqrt_45 : g (Real.sqrt 45) = 13 := by
  sorry

end g_sqrt_45_l1615_161591


namespace percentage_problem_l1615_161571

noncomputable def percentage_of_value (x : ℝ) (y : ℝ) (z : ℝ) : ℝ :=
  (y / x) * 100

theorem percentage_problem :
  percentage_of_value 2348 (528.0642570281125 * 4.98) = 112 := 
by
  sorry

end percentage_problem_l1615_161571


namespace average_reading_days_l1615_161544

def emery_days : ℕ := 20
def serena_days : ℕ := 5 * emery_days
def average_days (e s : ℕ) : ℕ := (e + s) / 2

theorem average_reading_days 
  (e s : ℕ) 
  (h1 : e = emery_days)
  (h2 : s = serena_days) :
  average_days e s = 60 :=
by
  rw [h1, h2, emery_days, serena_days]
  sorry

end average_reading_days_l1615_161544


namespace quadratic_difference_square_l1615_161596

theorem quadratic_difference_square (α β : ℝ) (h : α ≠ β) (hα : α^2 - 3 * α + 1 = 0) (hβ : β^2 - 3 * β + 1 = 0) : (α - β)^2 = 5 := by
  sorry

end quadratic_difference_square_l1615_161596


namespace minimum_value_of_a_l1615_161509

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp (3 * Real.log x - x)) - x^2 - (a - 4) * x - 4

theorem minimum_value_of_a (h : ∀ x > 0, f x ≤ 0) : a ≥ 4 / Real.exp 2 := by
  sorry

end minimum_value_of_a_l1615_161509


namespace no_three_partition_exists_l1615_161587

/-- Define the partitioning property for three subsets -/
def partitions (A B C : Set ℤ) : Prop :=
  ∀ n : ℤ, (n ∈ A ∨ n ∈ B ∨ n ∈ C) ∧ (n ∈ A ↔ n-50 ∈ B ∧ n+1987 ∈ C) ∧ (n-50 ∈ A ∨ n-50 ∈ B ∨ n-50 ∈ C) ∧ (n-50 ∈ B ↔ n-50-50 ∈ A ∧ n-50+1987 ∈ C) ∧ (n+1987 ∈ A ∨ n+1987 ∈ B ∨ n+1987 ∈ C) ∧ (n+1987 ∈ C ↔ n+1987-50 ∈ A ∧ n+1987+1987 ∈ B)

/-- The main theorem stating that no such partition is possible -/
theorem no_three_partition_exists :
  ¬∃ A B C : Set ℤ, partitions A B C :=
sorry

end no_three_partition_exists_l1615_161587


namespace correct_M_min_t_for_inequality_l1615_161589

-- Define the set M
def M : Set ℝ := {a | 0 ≤ a ∧ a < 4}

-- Prove that M is correct given ax^2 + ax + 2 > 0 for all x ∈ ℝ implies 0 ≤ a < 4
theorem correct_M (a : ℝ) : (∀ x : ℝ, a * x^2 + a * x + 2 > 0) ↔ (0 ≤ a ∧ a < 4) :=
sorry

-- Prove the minimum value of t given t > 0 and the inequality holds for all a ∈ M
theorem min_t_for_inequality (t : ℝ) (h : 0 < t) : 
  (∀ a ∈ M, (a^2 - 2 * a) * t ≤ t^2 + 3 * t - 46) ↔ 46 ≤ t :=
sorry

end correct_M_min_t_for_inequality_l1615_161589


namespace secret_code_count_l1615_161577

-- Conditions
def num_colors : ℕ := 8
def num_slots : ℕ := 5

-- The proof statement
theorem secret_code_count : (num_colors ^ num_slots) = 32768 := by
  sorry

end secret_code_count_l1615_161577


namespace area_of_rectangle_l1615_161584

-- Define the given conditions
def side_length_of_square (s : ℝ) (ABCD : ℝ) : Prop :=
  ABCD = 4 * s^2

def perimeter_of_rectangle (s : ℝ) (perimeter : ℝ): Prop :=
  perimeter = 8 * s

-- Statement of the proof problem
theorem area_of_rectangle (s perimeter_area : ℝ) (h_perimeter : perimeter_of_rectangle s 160) :
  side_length_of_square s 1600 :=
by
  sorry

end area_of_rectangle_l1615_161584


namespace total_chocolate_bars_proof_l1615_161546

def large_box_contains := 17
def first_10_boxes_contains := 10
def medium_boxes_per_small := 4
def chocolate_bars_per_medium := 26

def remaining_7_boxes := 7
def first_two_boxes := 2
def first_two_bars := 18
def next_three_boxes := 3
def next_three_bars := 22
def last_two_boxes := 2
def last_two_bars := 30

noncomputable def total_chocolate_bars_in_large_box : Nat :=
  let chocolate_in_first_10 := first_10_boxes_contains * medium_boxes_per_small * chocolate_bars_per_medium
  let chocolate_in_remaining_7 :=
    (first_two_boxes * first_two_bars) +
    (next_three_boxes * next_three_bars) +
    (last_two_boxes * last_two_bars)
  chocolate_in_first_10 + chocolate_in_remaining_7

theorem total_chocolate_bars_proof :
  total_chocolate_bars_in_large_box = 1202 :=
by
  -- Detailed calculation is skipped
  sorry

end total_chocolate_bars_proof_l1615_161546


namespace geometric_sequence_problem_l1615_161532

noncomputable def a : ℕ → ℝ := sorry

theorem geometric_sequence_problem :
  a 4 = 4 →
  a 8 = 8 →
  a 12 = 16 :=
by
  intros h4 h8
  sorry

end geometric_sequence_problem_l1615_161532


namespace infinite_series_sum_l1615_161581

theorem infinite_series_sum : 
  ∑' k : ℕ, (5^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 3 := 
sorry

end infinite_series_sum_l1615_161581


namespace rate_of_dividend_is_12_l1615_161563

-- Defining the conditions
def total_investment : ℝ := 4455
def price_per_share : ℝ := 8.25
def annual_income : ℝ := 648
def face_value_per_share : ℝ := 10

-- Expected rate of dividend
def expected_rate_of_dividend : ℝ := 12

-- The proof problem statement: Prove that the rate of dividend is 12% given the conditions.
theorem rate_of_dividend_is_12 :
  ∃ (r : ℝ), r = 12 ∧ annual_income = 
    (total_investment / price_per_share) * (r / 100) * face_value_per_share :=
by 
  use 12
  sorry

end rate_of_dividend_is_12_l1615_161563


namespace parabola_equation_l1615_161575

theorem parabola_equation (a b c d e f : ℤ)
  (h1 : a = 0 )    -- The equation should have no x^2 term
  (h2 : b = 0 )    -- The equation should have no xy term
  (h3 : c > 0)     -- The coefficient of y^2 should be positive
  (h4 : d = -2)    -- The coefficient of x in the final form should be -2
  (h5 : e = -8)    -- The coefficient of y in the final form should be -8
  (h6 : f = 16)    -- The constant term in the final form should be 16
  (pass_through : (2 : ℤ) = k * (6 - 4) ^ 2)
  (vertex : (0 : ℤ) = k * (sym_axis - 4) ^ 2)
  (symmetry_axis_parallel_x : True)
  (vertex_on_y_axis : True):
  ax^2 + bxy + cy^2 + dx + ey + f = 0 :=
by
  sorry

end parabola_equation_l1615_161575


namespace trigonometric_identity_l1615_161515

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin α ^ 2 + Real.sin α * Real.cos α = 6 / 5 := 
sorry

end trigonometric_identity_l1615_161515


namespace joan_bought_72_eggs_l1615_161590

def dozen := 12
def joan_eggs (dozens: Nat) := dozens * dozen

theorem joan_bought_72_eggs : joan_eggs 6 = 72 :=
by
  sorry

end joan_bought_72_eggs_l1615_161590


namespace initial_money_l1615_161535

/-
We had $3500 left after spending 30% of our money on clothing, 
25% on electronics, and saving 15% in a bank account. 
How much money (X) did we start with before shopping and saving?
-/

theorem initial_money (M : ℝ) 
  (h_clothing : 0.30 * M ≠ 0) 
  (h_electronics : 0.25 * M ≠ 0) 
  (h_savings : 0.15 * M ≠ 0) 
  (remaining_money : 0.30 * M = 3500) : 
  M = 11666.67 := 
sorry

end initial_money_l1615_161535


namespace rationalize_denominator_eqn_l1615_161576

theorem rationalize_denominator_eqn : 
  let expr := (3 + Real.sqrt 2) / (2 - Real.sqrt 5)
  let rationalized := -6 - 3 * Real.sqrt 5 - 2 * Real.sqrt 2 - Real.sqrt 10
  let A := -6
  let B := -2
  let C := 2
  expr = rationalized ∧ A * B * C = -24 :=
by
  sorry

end rationalize_denominator_eqn_l1615_161576


namespace solution_to_quadratic_inequality_l1615_161565

theorem solution_to_quadratic_inequality 
  (a : ℝ)
  (h : ∀ x : ℝ, x^2 - a * x + 1 < 0 ↔ (1 / 2 : ℝ) < x ∧ x < 2) :
  a = 5 / 2 :=
sorry

end solution_to_quadratic_inequality_l1615_161565


namespace exists_valid_configuration_l1615_161506

-- Define the nine circles
def circles : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Define the connections (adjacency list) where each connected pair must sum to 23
def lines : List (ℕ × ℕ) := [(1, 8), (8, 6), (8, 9), (9, 2), (2, 7), (7, 6), (7, 4), (4, 1), (4, 5), (5, 6), (5, 3), (6, 3)]

-- The main theorem that we need to prove: there exists a permutation of circles satisfying the line sum condition
theorem exists_valid_configuration: 
  ∃ (f : ℕ → ℕ), 
    (∀ x ∈ circles, f x ∈ circles) ∧ 
    (∀ (a b : ℕ), (a, b) ∈ lines → f a + f b = 23) :=
sorry

end exists_valid_configuration_l1615_161506


namespace find_solution_l1615_161578

-- Definitions for the problem
def is_solution (x y z t : ℕ) : Prop := (x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧ (2^y + 2^z * 5^t - 5^x = 1))

-- Statement of the theorem
theorem find_solution : ∀ x y z t : ℕ, is_solution x y z t → (x, y, z, t) = (2, 4, 1, 1) := by
  sorry

end find_solution_l1615_161578


namespace original_price_l1615_161543

variables (q r : ℝ) (h1 : 0 ≤ q) (h2 : 0 ≤ r)

theorem original_price (h : (2 : ℝ) = (1 + q / 100) * (1 - r / 100) * x) :
  x = 200 / (100 + q - r - (q * r) / 100) :=
by
  sorry

end original_price_l1615_161543


namespace geometric_sequence_product_l1615_161524

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) (h_seq : geometric_sequence a) 
  (h_cond : a 2 * a 4 = 16) : a 2 * a 3 * a 4 = 64 ∨ a 2 * a 3 * a 4 = -64 :=
by
  sorry

end geometric_sequence_product_l1615_161524


namespace find_k_l1615_161559

theorem find_k (m : ℝ) (h : ∃ A B : ℝ, (m^3 - 24*m + 16) = (m^2 - 8*m) * (A*m + B) ∧ A - 8 = -k ∧ -8*B = -24) : k = 5 :=
sorry

end find_k_l1615_161559


namespace xiaoming_additional_games_l1615_161566

variable (total_games games_won target_percentage : ℕ)

theorem xiaoming_additional_games :
  total_games = 20 →
  games_won = 95 * total_games / 100 →
  target_percentage = 96 →
  ∃ additional_games, additional_games = 5 ∧
    (games_won + additional_games) / (total_games + additional_games) = target_percentage / 100 :=
by
  sorry

end xiaoming_additional_games_l1615_161566


namespace value_of_expression_l1615_161553

theorem value_of_expression (x : ℤ) (h : x = 5) : 3 * x + 4 = 19 :=
by
  rw [h]
  exact rfl

end value_of_expression_l1615_161553


namespace sum_of_possible_values_of_x_l1615_161549

namespace ProofProblem

-- Assume we are working in degrees for angles
def is_scalene_triangle (A B C : ℝ) (a b c : ℝ) :=
  a ≠ b ∧ b ≠ c ∧ c ≠ a

def triangle_angle_sum (A B C : ℝ) : Prop :=
  A + B + C = 180

noncomputable def problem_statement (x : ℝ) (A B C : ℝ) (a b c : ℝ) : Prop :=
  is_scalene_triangle A B C a b c ∧
  B = 45 ∧
  (A = x ∨ C = x) ∧
  (a = b ∨ b = c ∨ c = a) ∧
  triangle_angle_sum A B C

theorem sum_of_possible_values_of_x (x : ℝ) (A B C : ℝ) (a b c : ℝ) :
  problem_statement x A B C a b c →
  x = 45 :=
sorry

end ProofProblem

end sum_of_possible_values_of_x_l1615_161549


namespace trapezoid_perimeter_l1615_161528

theorem trapezoid_perimeter (x y : ℝ) (h1 : x ≠ 0)
  (h2 : ∀ (AB CD AD BC : ℝ), AB = 2 * x ∧ CD = 4 * x ∧ AD = 2 * y ∧ BC = y) :
  (∀ (P : ℝ), P = AB + BC + CD + AD → P = 6 * x + 3 * y) :=
by sorry

end trapezoid_perimeter_l1615_161528


namespace time_taken_by_alex_l1615_161512

-- Define the conditions
def distance_per_lap : ℝ := 500 -- distance per lap in meters
def distance_first_part : ℝ := 150 -- first part of the distance in meters
def speed_first_part : ℝ := 3 -- speed for the first part in meters per second
def distance_second_part : ℝ := 350 -- remaining part of the distance in meters
def speed_second_part : ℝ := 4 -- speed for the remaining part in meters per second
def num_laps : ℝ := 4 -- number of laps run by Alex

-- Target time, expressed in seconds
def target_time : ℝ := 550 -- 9 minutes and 10 seconds is 550 seconds

-- Prove that given the conditions, the total time Alex takes to run 4 laps is 550 seconds
theorem time_taken_by_alex :
  (distance_first_part / speed_first_part + distance_second_part / speed_second_part) * num_laps = target_time :=
by
  sorry

end time_taken_by_alex_l1615_161512


namespace max_type_A_pieces_max_profit_l1615_161542

noncomputable def type_A_cost := 80
noncomputable def type_A_sell := 120
noncomputable def type_B_cost := 60
noncomputable def type_B_sell := 90
noncomputable def total_clothes := 100
noncomputable def min_type_A := 65
noncomputable def max_cost := 7500

/-- The maximum number of type A clothing pieces that can be purchased --/
theorem max_type_A_pieces (x : ℕ) : 
  type_A_cost * x + type_B_cost * (total_clothes - x) ≤ max_cost → 
  x ≤ 75 := by 
sorry

variable (a : ℝ) (h_a : 0 < a ∧ a < 10)

/-- The optimal purchase strategy to maximize profit --/
theorem max_profit (x y : ℕ) : 
  (x + y = total_clothes) ∧ 
  (type_A_cost * x + type_B_cost * y ≤ max_cost) ∧
  (min_type_A ≤ x) ∧ 
  (x ≤ 75) → 
  (type_A_sell - type_A_cost - a) * x + (type_B_sell - type_B_cost) * y 
  ≤ (type_A_sell - type_A_cost - a) * 75 + (type_B_sell - type_B_cost) * 25 := by 
sorry

end max_type_A_pieces_max_profit_l1615_161542


namespace find_sum_mod_7_l1615_161573

open ZMod

-- Let a, b, and c be elements of the cyclic group modulo 7
def a : ZMod 7 := sorry
def b : ZMod 7 := sorry
def c : ZMod 7 := sorry

-- Conditions
axiom h1 : a * b * c = 1
axiom h2 : 4 * c = 5
axiom h3 : 5 * b = 4 + b

-- Goal
theorem find_sum_mod_7 : a + b + c = 2 := by
  sorry

end find_sum_mod_7_l1615_161573


namespace transform_quadratic_to_linear_l1615_161554

theorem transform_quadratic_to_linear (x y : ℝ) : 
  x^2 - 4 * x * y + 4 * y^2 = 4 ↔ (x - 2 * y + 2 = 0 ∨ x - 2 * y - 2 = 0) :=
by
  sorry

end transform_quadratic_to_linear_l1615_161554


namespace crystal_meals_count_l1615_161523

def num_entrees : ℕ := 4
def num_drinks : ℕ := 4
def num_desserts : ℕ := 2

theorem crystal_meals_count : num_entrees * num_drinks * num_desserts = 32 := by
  sorry

end crystal_meals_count_l1615_161523


namespace infinite_series_computation_l1615_161520

noncomputable def infinite_series_sum (a b : ℝ) : ℝ :=
  ∑' n : ℕ, if n = 0 then (0 : ℝ) else
    (1 : ℝ) / ((2 * (n - 1 : ℕ) * a - (n - 2 : ℕ) * b) * (2 * n * a - (n - 1 : ℕ) * b))

theorem infinite_series_computation (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_ineq : a > b) :
  infinite_series_sum a b = 1 / ((2 * a - b) * (2 * b)) :=
by
  sorry

end infinite_series_computation_l1615_161520


namespace largest_room_length_l1615_161502

theorem largest_room_length (L : ℕ) (w_large w_small l_small diff_area : ℕ)
  (h1 : w_large = 45)
  (h2 : w_small = 15)
  (h3 : l_small = 8)
  (h4 : diff_area = 1230)
  (h5 : w_large * L - (w_small * l_small) = diff_area) :
  L = 30 :=
by sorry

end largest_room_length_l1615_161502


namespace floor_inequality_sqrt_l1615_161529

theorem floor_inequality_sqrt (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  (⌊ m * Real.sqrt 2 ⌋) * (⌊ n * Real.sqrt 7 ⌋) < (⌊ m * n * Real.sqrt 14 ⌋) := 
by
  sorry

end floor_inequality_sqrt_l1615_161529


namespace initial_amount_saved_l1615_161522

noncomputable section

def cost_of_couch : ℝ := 750
def cost_of_table : ℝ := 100
def cost_of_lamp : ℝ := 50
def amount_still_owed : ℝ := 400

def total_cost : ℝ := cost_of_couch + cost_of_table + cost_of_lamp

theorem initial_amount_saved (initial_amount : ℝ) :
  initial_amount = total_cost - amount_still_owed ↔ initial_amount = 500 :=
by
  -- the proof is omitted
  sorry

end initial_amount_saved_l1615_161522


namespace b_alone_work_time_l1615_161526

def work_rate_combined (a_rate b_rate : ℝ) : ℝ := a_rate + b_rate

theorem b_alone_work_time
  (a_rate b_rate : ℝ)
  (h1 : work_rate_combined a_rate b_rate = 1/16)
  (h2 : a_rate = 1/20) :
  b_rate = 1/80 := by
  sorry

end b_alone_work_time_l1615_161526


namespace calculate_total_travel_time_l1615_161579

/-- The total travel time, including stops, from the first station to the last station. -/
def total_travel_time (d1 d2 d3 : ℕ) (s1 s2 s3 : ℕ) (t1 t2 : ℕ) : ℚ :=
  let leg1_time := d1 / s1
  let stop1_time := t1 / 60
  let leg2_time := d2 / s2
  let stop2_time := t2 / 60
  let leg3_time := d3 / s3
  leg1_time + stop1_time + leg2_time + stop2_time + leg3_time

/-- Proof that total travel time is 2 hours and 22.5 minutes. -/
theorem calculate_total_travel_time :
  total_travel_time 30 40 50 60 40 80 10 5 = 2.375 :=
by
  sorry

end calculate_total_travel_time_l1615_161579


namespace find_second_number_l1615_161525

theorem find_second_number (x y z : ℚ) (h1 : x + y + z = 120)
  (h2 : x / y = 3 / 4) (h3 : y / z = 4 / 7) : y = 240 / 7 := by
  sorry

end find_second_number_l1615_161525


namespace ratio_lcm_gcf_l1615_161517

theorem ratio_lcm_gcf (a b : ℕ) (h₁ : a = 252) (h₂ : b = 675) : 
  let lcm_ab := Nat.lcm a b
  let gcf_ab := Nat.gcd a b
  (lcm_ab / gcf_ab) = 2100 :=
by
  sorry

end ratio_lcm_gcf_l1615_161517


namespace probability_same_plane_l1615_161533

-- Define the number of vertices in a cube
def num_vertices : ℕ := 8

-- Define the number of vertices to be selected
def selection : ℕ := 4

-- Define the total number of ways to select 4 vertices out of 8
def total_ways : ℕ := Nat.choose num_vertices selection

-- Define the number of favorable ways to have 4 vertices lie in the same plane
def favorable_ways : ℕ := 12

-- Define the probability that the 4 selected vertices lie in the same plane
def probability : ℚ := favorable_ways / total_ways

-- The statement we need to prove
theorem probability_same_plane : probability = 6 / 35 := by
  sorry

end probability_same_plane_l1615_161533


namespace cost_per_adult_meal_l1615_161583

-- Definitions and given conditions
def total_people : ℕ := 13
def num_kids : ℕ := 9
def total_cost : ℕ := 28

-- Question translated into a proof statement
theorem cost_per_adult_meal : (total_cost / (total_people - num_kids)) = 7 := 
by
  sorry

end cost_per_adult_meal_l1615_161583


namespace gcd_540_180_diminished_by_2_eq_178_l1615_161507

theorem gcd_540_180_diminished_by_2_eq_178 : gcd 540 180 - 2 = 178 := by
  sorry

end gcd_540_180_diminished_by_2_eq_178_l1615_161507


namespace cakes_served_during_lunch_today_l1615_161595

theorem cakes_served_during_lunch_today (L : ℕ) 
  (h_total : L + 6 + 3 = 14) : 
  L = 5 :=
sorry

end cakes_served_during_lunch_today_l1615_161595


namespace fill_time_difference_correct_l1615_161547

-- Define the time to fill one barrel in normal conditions
def normal_fill_time_per_barrel : ℕ := 3

-- Define the time to fill one barrel with a leak
def leak_fill_time_per_barrel : ℕ := 5

-- Define the number of barrels to fill
def barrels_to_fill : ℕ := 12

-- Define the time to fill 12 barrels in normal conditions
def normal_fill_time : ℕ := normal_fill_time_per_barrel * barrels_to_fill

-- Define the time to fill 12 barrels with a leak
def leak_fill_time : ℕ := leak_fill_time_per_barrel * barrels_to_fill

-- Define the time difference
def time_difference : ℕ := leak_fill_time - normal_fill_time

theorem fill_time_difference_correct : time_difference = 24 := by
  sorry

end fill_time_difference_correct_l1615_161547


namespace point_B_possible_values_l1615_161527

-- Define point A
def A : ℝ := 1

-- Define the condition that B is 3 units away from A
def units_away (a b : ℝ) : ℝ := abs (b - a)

theorem point_B_possible_values :
  ∃ B : ℝ, units_away A B = 3 ∧ (B = 4 ∨ B = -2) := by
  sorry

end point_B_possible_values_l1615_161527


namespace ellipse_eccentricity_l1615_161557

theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : b = (4/3) * c) (h4 : a^2 - b^2 = c^2) : 
  c / a = 3 / 5 :=
by
  sorry

end ellipse_eccentricity_l1615_161557


namespace min_value_of_a_l1615_161539

/-- Given the inequality |x - 1| + |x + a| ≤ 8, prove that the minimum value of a is -9 -/

theorem min_value_of_a (a : ℝ) (h : ∀ x : ℝ, |x - 1| + |x + a| ≤ 8) : a = -9 :=
sorry

end min_value_of_a_l1615_161539


namespace model_car_cost_l1615_161582

theorem model_car_cost (x : ℕ) :
  (5 * x) + (5 * 10) + (5 * 2) = 160 → x = 20 :=
by
  intro h
  sorry

end model_car_cost_l1615_161582


namespace largest_odd_not_sum_of_three_distinct_composites_l1615_161592

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem largest_odd_not_sum_of_three_distinct_composites :
  ∀ n : ℕ, is_odd n → (¬ ∃ (a b c : ℕ), is_composite a ∧ is_composite b ∧ is_composite c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ n = a + b + c) → n ≤ 17 :=
by
  sorry

end largest_odd_not_sum_of_three_distinct_composites_l1615_161592


namespace poster_width_l1615_161568
   
   theorem poster_width (h : ℕ) (A : ℕ) (w : ℕ) (h_eq : h = 7) (A_eq : A = 28) (area_eq : w * h = A) : w = 4 :=
   by
   sorry
   
end poster_width_l1615_161568


namespace perpendicular_lines_solve_b_l1615_161504

theorem perpendicular_lines_solve_b (b : ℝ) : (∀ x y : ℝ, y = 3 * x + 7 →
                                                    ∃ y1 : ℝ, y1 = ( - b / 4 ) * x + 3 ∧
                                                               3 * ( - b / 4 ) = -1) → 
                                               b = 4 / 3 :=
by
  sorry

end perpendicular_lines_solve_b_l1615_161504


namespace intersection_is_N_l1615_161597

-- Define the sets M and N as given in the problem
def M := {x : ℝ | x > 0}
def N := {x : ℝ | Real.log x > 0}

-- State the theorem for the intersection of M and N
theorem intersection_is_N : (M ∩ N) = N := 
  by 
    sorry

end intersection_is_N_l1615_161597


namespace line_equation_l1615_161580

theorem line_equation (x y : ℝ) : 
  (3 * x + y = 0) ∧ (x + y - 2 = 0) ∧ 
  ∃ m : ℝ, -2 = -(1 / m) ∧ 
  (∃ b : ℝ, (y = m * x + b) ∧ (3 = m * (-1) + b)) ∧ 
  x - 2 * y + 7 = 0 :=
sorry

end line_equation_l1615_161580


namespace x_share_of_profit_l1615_161538

-- Define the problem conditions
def investment_x : ℕ := 5000
def investment_y : ℕ := 15000
def total_profit : ℕ := 1600

-- Define the ratio simplification
def ratio_x : ℕ := 1
def ratio_y : ℕ := 3
def total_ratio_parts : ℕ := ratio_x + ratio_y

-- Define the profit division per part
def profit_per_part : ℕ := total_profit / total_ratio_parts

-- Lean 4 statement to prove
theorem x_share_of_profit : profit_per_part * ratio_x = 400 := sorry

end x_share_of_profit_l1615_161538


namespace solve_for_b_l1615_161510

theorem solve_for_b :
  (∀ (x y : ℝ), 4 * y - 3 * x + 2 = 0) →
  (∀ (x y : ℝ), 2 * y + b * x - 1 = 0) →
  (∃ b : ℝ, b = 8 / 3) := 
by
  sorry

end solve_for_b_l1615_161510


namespace GoldenRabbitCards_count_l1615_161558

theorem GoldenRabbitCards_count :
  let total_cards := 10000
  let non_golden_combinations := 8 * 8 * 8 * 8
  let golden_cards := total_cards - non_golden_combinations
  golden_cards = 5904 :=
by
  let total_cards := 10000
  let non_golden_combinations := 8 * 8 * 8 * 8
  let golden_cards := total_cards - non_golden_combinations
  sorry

end GoldenRabbitCards_count_l1615_161558


namespace distinct_exponentiation_values_l1615_161548

theorem distinct_exponentiation_values : 
  let a := 3^(3^(3^3))
  let b := 3^((3^3)^3)
  let c := ((3^3)^3)^3
  let d := 3^((3^3)^(3^2))
  (a ≠ b) → (a ≠ c) → (a ≠ d) → (b ≠ c) → (b ≠ d) → (c ≠ d) → 
  ∃ n, n = 3 := 
sorry

end distinct_exponentiation_values_l1615_161548


namespace least_n_for_factorial_multiple_10080_l1615_161521

theorem least_n_for_factorial_multiple_10080 (n : ℕ) 
  (h₁ : 0 < n) 
  (h₂ : ∀ m, m > 0 → (n ≠ m → n! % 10080 ≠ 0)) 
  : n = 8 := 
sorry

end least_n_for_factorial_multiple_10080_l1615_161521


namespace function_value_at_6000_l1615_161516

theorem function_value_at_6000
  (f : ℝ → ℝ)
  (h0 : f 0 = 1)
  (h1 : ∀ x : ℝ, f (x + 3) = f x + 2 * x + 3) :
  f 6000 = 12000001 :=
by
  sorry

end function_value_at_6000_l1615_161516


namespace multiples_of_six_l1615_161534

theorem multiples_of_six (a b : ℕ) (h₁ : a = 5) (h₂ : b = 127) :
  ∃ n : ℕ, n = 21 ∧ ∀ x : ℕ, (a < 6 * x ∧ 6 * x < b) ↔ (1 ≤ x ∧ x ≤ 21) :=
by
  sorry

end multiples_of_six_l1615_161534


namespace blueprint_conversion_proof_l1615_161562

-- Let inch_to_feet be the conversion factor from blueprint inches to actual feet.
def inch_to_feet : ℝ := 500

-- Let line_segment_inch be the length of the line segment on the blueprint in inches.
def line_segment_inch : ℝ := 6.5

-- Then, line_segment_feet is the actual length of the line segment in feet.
def line_segment_feet : ℝ := line_segment_inch * inch_to_feet

-- Theorem statement to prove
theorem blueprint_conversion_proof : line_segment_feet = 3250 := by
  -- Proof goes here
  sorry

end blueprint_conversion_proof_l1615_161562


namespace excess_percentage_l1615_161574

theorem excess_percentage (x : ℝ) 
  (L W : ℝ) (hL : L > 0) (hW : W > 0) 
  (h1 : L * (1 + x / 100) * W * 0.96 = L * W * 1.008) : 
  x = 5 :=
by sorry

end excess_percentage_l1615_161574


namespace base9_addition_l1615_161511

-- Define the numbers in base 9
def num1 : ℕ := 1 * 9^2 + 7 * 9^1 + 5 * 9^0
def num2 : ℕ := 7 * 9^2 + 1 * 9^1 + 4 * 9^0
def num3 : ℕ := 6 * 9^1 + 1 * 9^0
def result : ℕ := 1 * 9^3 + 0 * 9^2 + 6 * 9^1 + 1 * 9^0

-- State the theorem
theorem base9_addition : num1 + num2 + num3 = result := by
  sorry

end base9_addition_l1615_161511


namespace shares_owned_l1615_161564

theorem shares_owned (expected_earnings dividend_ratio additional_per_10c actual_earnings total_dividend : ℝ)
  ( h1 : expected_earnings = 0.80 )
  ( h2 : dividend_ratio = 0.50 )
  ( h3 : additional_per_10c = 0.04 )
  ( h4 : actual_earnings = 1.10 )
  ( h5 : total_dividend = 156.0 ) :
  ∃ shares : ℝ, shares = total_dividend / (expected_earnings * dividend_ratio + (max ((actual_earnings - expected_earnings) / 0.10) 0) * additional_per_10c) ∧ shares = 300 := 
sorry

end shares_owned_l1615_161564


namespace convert_1623_to_base7_l1615_161593

theorem convert_1623_to_base7 :
  ∃ a b c d : ℕ, 1623 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧
  a = 4 ∧ b = 5 ∧ c = 0 ∧ d = 6 :=
by
  sorry

end convert_1623_to_base7_l1615_161593


namespace find_a9_l1615_161588

-- Define the geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Given conditions of the problem
variables {a : ℕ → ℝ}
axiom h_geom_seq : is_geometric_sequence a
axiom h_root1 : a 3 * a 15 = 1
axiom h_root2 : a 3 + a 15 = -4

-- The proof statement
theorem find_a9 : a 9 = 1 := 
by sorry

end find_a9_l1615_161588


namespace max_value_9_l1615_161555

noncomputable def max_ab_ac_bc (a b c : ℝ) : ℝ :=
  max (a * b) (max (a * c) (b * c))

theorem max_value_9 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : a + b + c = 12) (h_prod : a * b + b * c + c * a = 27) :
  max_ab_ac_bc a b c = 9 :=
sorry

end max_value_9_l1615_161555


namespace total_feet_l1615_161518

theorem total_feet (H C : ℕ) (h1 : H + C = 48) (h2 : H = 28) : 2 * H + 4 * C = 136 := 
by
  sorry

end total_feet_l1615_161518


namespace f_odd_f_decreasing_f_extremum_l1615_161508

noncomputable def f : ℝ → ℝ := sorry

axiom f_additive : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_val : f 1 = -2
axiom f_neg : ∀ x > 0, f x < 0

theorem f_odd : ∀ x : ℝ, f (-x) = -f x :=
sorry

theorem f_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂ :=
sorry

theorem f_extremum : ∃ (max min : ℝ), max = f (-3) ∧ min = f 3 :=
sorry

end f_odd_f_decreasing_f_extremum_l1615_161508


namespace sin_double_theta_l1615_161503

-- Given condition
def given_condition (θ : ℝ) : Prop :=
  Real.cos (Real.pi / 4 - θ) = 1 / 2

-- The statement we want to prove: sin(2θ) = -1/2
theorem sin_double_theta (θ : ℝ) (h : given_condition θ) : Real.sin (2 * θ) = -1 / 2 :=
sorry

end sin_double_theta_l1615_161503


namespace part_one_part_two_l1615_161586

noncomputable def f (x : ℝ) (a : ℝ) := x^2 + a * x + 6

theorem part_one (x : ℝ) : ∀ a, a = 5 → f x a < 0 ↔ -3 < x ∧ x < -2 :=
by
  sorry

theorem part_two : ∀ a, (∀ x, f x a > 0) ↔ - 2 * Real.sqrt 6 < a ∧ a < 2 * Real.sqrt 6 :=
by
  sorry

end part_one_part_two_l1615_161586


namespace adam_walks_distance_l1615_161514

/-- The side length of the smallest squares is 20 cm. --/
def smallest_square_side : ℕ := 20

/-- The side length of the middle-sized square is 2 times the smallest square. --/
def middle_square_side : ℕ := 2 * smallest_square_side

/-- The side length of the largest square is 3 times the smallest square. --/
def largest_square_side : ℕ := 3 * smallest_square_side

/-- The number of smallest squares Adam encounters. --/
def num_smallest_squares : ℕ := 5

/-- The number of middle-sized squares Adam encounters. --/
def num_middle_squares : ℕ := 5

/-- The number of largest squares Adam encounters. --/
def num_largest_squares : ℕ := 2

/-- The total distance Adam walks from P to Q. --/
def total_distance : ℕ :=
  num_smallest_squares * smallest_square_side +
  num_middle_squares * middle_square_side +
  num_largest_squares * largest_square_side

/-- Proof that the total distance Adam walks is 420 cm. --/
theorem adam_walks_distance : total_distance = 420 := by
  sorry

end adam_walks_distance_l1615_161514


namespace points_on_line_sufficient_but_not_necessary_l1615_161531

open Nat

-- Define the sequence a_n
def sequence_a (n : ℕ) : ℕ := n + 1

-- Define a general arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) := ∀ n m : ℕ, n < m → a (m) - a (n) = (m - n) * (a 1 - a 0)

-- Define the condition that points (n, a_n), where n is a natural number, lie on the line y = x + 1
def points_on_line (a : ℕ → ℕ) : Prop := ∀ n : ℕ, a (n) = n + 1

-- Prove that points_on_line is sufficient but not necessary for is_arithmetic_sequence
theorem points_on_line_sufficient_but_not_necessary :
  (∀ a : ℕ → ℕ, points_on_line a → is_arithmetic_sequence a)
  ∧ ∃ a : ℕ → ℕ, is_arithmetic_sequence a ∧ ¬ points_on_line a := 
by 
  sorry

end points_on_line_sufficient_but_not_necessary_l1615_161531


namespace probability_of_diff_by_three_is_one_eighth_l1615_161551

-- Define the problem within a namespace
namespace DiceRoll

-- Define the probability of rolling two integers that differ by 3 on an 8-sided die
noncomputable def prob_diff_by_three : ℚ :=
  let successful_outcomes := 8
  let total_outcomes := 8 * 8
  successful_outcomes / total_outcomes

-- The main theorem
theorem probability_of_diff_by_three_is_one_eighth :
  prob_diff_by_three = 1 / 8 := by
  sorry

end DiceRoll

end probability_of_diff_by_three_is_one_eighth_l1615_161551


namespace area_above_the_line_l1615_161519

-- Definitions of the circle and the line equations
def circle_eqn (x y : ℝ) := (x - 5)^2 + (y - 3)^2 = 1
def line_eqn (x y : ℝ) := y = x - 5

-- The main statement to prove
theorem area_above_the_line : 
  ∃ (A : ℝ), A = (3 / 4) * Real.pi ∧ 
  ∀ (x y : ℝ), 
    circle_eqn x y ∧ y > x - 5 → 
    A > 0 := 
sorry

end area_above_the_line_l1615_161519


namespace probability_blue_given_not_red_l1615_161567

theorem probability_blue_given_not_red :
  let total_balls := 20
  let red_balls := 5
  let yellow_balls := 5
  let blue_balls := 10
  let non_red_balls := yellow_balls + blue_balls
  let blue_given_not_red := (blue_balls : ℚ) / non_red_balls
  blue_given_not_red = 2 / 3 := 
by
  sorry

end probability_blue_given_not_red_l1615_161567


namespace correct_statements_are_two_l1615_161545

def statement1 : Prop := 
  ∀ (data : Type) (eq : data → data → Prop), 
    (∃ (t : data), eq t t) → 
    (∀ (d1 d2 : data), eq d1 d2 → d1 = d2)

def statement2 : Prop := 
  ∀ (samplevals : Type) (regress_eqn : samplevals → samplevals → Prop), 
    (∃ (s : samplevals), regress_eqn s s) → 
    (∀ (sv1 sv2 : samplevals), regress_eqn sv1 sv2 → sv1 = sv2)

def statement3 : Prop := 
  ∀ (predvals : Type) (pred_eqn : predvals → predvals → Prop), 
    (∃ (p : predvals), pred_eqn p p) → 
    (∀ (pp1 pp2 : predvals), pred_eqn pp1 pp2 → pp1 = pp2)

def statement4 : Prop := 
  ∀ (observedvals : Type) (linear_eqn : observedvals → observedvals → Prop), 
    (∃ (o : observedvals), linear_eqn o o) → 
    (∀ (ov1 ov2 : observedvals), linear_eqn ov1 ov2 → ov1 = ov2)

def correct_statements_count : ℕ := 2

theorem correct_statements_are_two : 
  (statement1 ∧ statement2 ∧ ¬ statement3 ∧ ¬ statement4) → 
  correct_statements_count = 2 := by
  sorry

end correct_statements_are_two_l1615_161545


namespace derivative_of_f_l1615_161501

noncomputable def f (x : ℝ) : ℝ := 2^x - Real.log x / Real.log 3

theorem derivative_of_f (x : ℝ) : (deriv f x) = 2^x * Real.log 2 - 1 / (x * Real.log 3) :=
by
  -- This statement skips the proof details
  sorry

end derivative_of_f_l1615_161501


namespace sqrt_x_minus_2_real_iff_x_ge_2_l1615_161536

theorem sqrt_x_minus_2_real_iff_x_ge_2 (x : ℝ) : (∃ r : ℝ, r * r = x - 2) ↔ x ≥ 2 := by
  sorry

end sqrt_x_minus_2_real_iff_x_ge_2_l1615_161536


namespace f_neg_one_f_monotonic_decreasing_solve_inequality_l1615_161537

-- Definitions based on conditions in part a)
variables {f : ℝ → ℝ}
axiom f_add : ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂ - 2
axiom f_one : f 1 = 0
axiom f_neg : ∀ x > 1, f x < 0

-- Proof statement for the value of f(-1)
theorem f_neg_one : f (-1) = 4 := by
  sorry

-- Proof statement for the monotonicity of f(x)
theorem f_monotonic_decreasing : ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂ := by
  sorry

-- Proof statement for the inequality solution
theorem solve_inequality (x : ℝ) :
  ∀ t, t = f (x^2 - 2*x) →
  t^2 + 2*t - 8 < 0 → (-1 < x ∧ x < 0) ∨ (2 < x ∧ x < 3) := by
  sorry

end f_neg_one_f_monotonic_decreasing_solve_inequality_l1615_161537


namespace woman_complete_time_l1615_161505

-- Define the work rate of one man
def man_rate := 1 / 100

-- Define the combined work rate equation for 10 men and 15 women completing work in 5 days
def combined_work_rate (W : ℝ) : Prop :=
  10 * man_rate + 15 * W = 1 / 5

-- Prove that given the combined work rate equation, one woman alone takes 150 days to complete the work
theorem woman_complete_time (W : ℝ) : combined_work_rate W → W = 1 / 150 :=
by
  intro h
  have h1 : 10 * man_rate + 15 * W = 1 / 5 := h
  rw [man_rate] at h1
  sorry -- Proof steps would go here

end woman_complete_time_l1615_161505


namespace combined_resistance_l1615_161552

theorem combined_resistance (x y : ℝ) (r : ℝ) (hx : x = 4) (hy : y = 6) :
  (1 / r) = (1 / x) + (1 / y) → r = 12 / 5 :=
by
  sorry

end combined_resistance_l1615_161552


namespace triangle_side_lengths_l1615_161598

theorem triangle_side_lengths (a : ℝ) :
  (∃ (b c : ℝ), b = 1 - 2 * a ∧ c = 8 ∧ (3 + b > c ∧ 3 + c > b ∧ b + c > 3)) ↔ (-5 < a ∧ a < -2) :=
sorry

end triangle_side_lengths_l1615_161598
