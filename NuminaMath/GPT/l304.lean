import Mathlib

namespace problem1_problem2_l304_30432

-- Problem 1: Prove (-a^3)^2 * (-a^2)^3 / a = -a^11 given a is a real number.
theorem problem1 (a : ℝ) : (-a^3)^2 * (-a^2)^3 / a = -a^11 :=
  sorry

-- Problem 2: Prove (m - n)^3 * (n - m)^4 * (n - m)^5 = - (n - m)^12 given m, n are real numbers.
theorem problem2 (m n : ℝ) : (m - n)^3 * (n - m)^4 * (n - m)^5 = - (n - m)^12 :=
  sorry

end problem1_problem2_l304_30432


namespace find_number_l304_30480

theorem find_number (N : ℝ) (h : 0.4 * (3 / 5) * N = 36) : N = 150 := 
sorry

end find_number_l304_30480


namespace length_of_bridge_l304_30406

theorem length_of_bridge
  (length_of_train : ℕ)
  (speed_km_hr : ℝ)
  (time_sec : ℝ)
  (h_train_length : length_of_train = 155)
  (h_train_speed : speed_km_hr = 45)
  (h_time : time_sec = 30) :
  ∃ (length_of_bridge : ℝ),
    length_of_bridge = 220 :=
by
  sorry

end length_of_bridge_l304_30406


namespace func_increasing_l304_30424

noncomputable def func (x : ℝ) : ℝ :=
  x^3 + x + 1

theorem func_increasing : ∀ x : ℝ, deriv func x > 0 := by
  sorry

end func_increasing_l304_30424


namespace geometric_sequence_seventh_term_l304_30404

theorem geometric_sequence_seventh_term (a r : ℝ) 
    (h1 : a * r^3 = 8) 
    (h2 : a * r^9 = 2) : 
    a * r^6 = 1 := 
by 
    sorry

end geometric_sequence_seventh_term_l304_30404


namespace units_digit_of_k_squared_plus_2_to_the_k_l304_30422

def k : ℕ := 2021^2 + 2^2021 + 3

theorem units_digit_of_k_squared_plus_2_to_the_k :
    (k^2 + 2^k) % 10 = 0 :=
by
    sorry

end units_digit_of_k_squared_plus_2_to_the_k_l304_30422


namespace find_sum_of_digits_l304_30494

theorem find_sum_of_digits (a b c d : ℕ) 
  (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h2 : a = 1)
  (h3 : 1000 * a + 100 * b + 10 * c + d - (100 * b + 10 * c + d) < 100)
  : a + b + c + d = 2 := 
sorry

end find_sum_of_digits_l304_30494


namespace eight_digit_descending_numbers_count_l304_30444

theorem eight_digit_descending_numbers_count : (Nat.choose 10 2) = 45 :=
by
  sorry

end eight_digit_descending_numbers_count_l304_30444


namespace number_of_black_cats_l304_30439

-- Definitions of the conditions.
def white_cats : Nat := 2
def gray_cats : Nat := 3
def total_cats : Nat := 15

-- The theorem we want to prove.
theorem number_of_black_cats : ∃ B : Nat, B = total_cats - (white_cats + gray_cats) ∧ B = 10 := by
  -- Proof will go here.
  sorry

end number_of_black_cats_l304_30439


namespace fraction_stamp_collection_l304_30458

theorem fraction_stamp_collection (sold_amount total_value : ℝ) (sold_for : sold_amount = 28) (total : total_value = 49) : sold_amount / total_value = 4 / 7 :=
by
  sorry

end fraction_stamp_collection_l304_30458


namespace count_divisors_2022_2022_l304_30428

noncomputable def num_divisors_2022_2022 : ℕ :=
  let fac2022 := 2022
  let factor_triplets := [(2, 3, 337), (3, 337, 2), (2, 337, 3), (337, 2, 3), (337, 3, 2), (3, 2, 337)]
  factor_triplets.length

theorem count_divisors_2022_2022 :
  num_divisors_2022_2022 = 6 :=
  by {
    sorry
  }

end count_divisors_2022_2022_l304_30428


namespace remaining_problems_to_grade_l304_30435

-- Define the conditions
def problems_per_worksheet : ℕ := 3
def total_worksheets : ℕ := 15
def graded_worksheets : ℕ := 7

-- The remaining worksheets to grade
def remaining_worksheets : ℕ := total_worksheets - graded_worksheets

-- Theorems stating the amount of problems left to grade
theorem remaining_problems_to_grade : problems_per_worksheet * remaining_worksheets = 24 :=
by
  sorry

end remaining_problems_to_grade_l304_30435


namespace proof_problem_l304_30402

theorem proof_problem (a b : ℤ) (h1 : ∃ k, a = 5 * k) (h2 : ∃ m, b = 10 * m) :
  (∃ n, b = 5 * n) ∧ (∃ p, a - b = 5 * p) :=
by
  sorry

end proof_problem_l304_30402


namespace gain_percent_is_approx_30_11_l304_30464

-- Definitions for cost price (CP) and selling price (SP)
def CP : ℕ := 930
def SP : ℕ := 1210

-- Definition for gain percent
noncomputable def gain_percent : ℚ :=
  ((SP - CP : ℚ) / CP) * 100

-- Statement to prove the gain percent is approximately 30.11%
theorem gain_percent_is_approx_30_11 :
  abs (gain_percent - 30.11) < 0.01 := by
  sorry

end gain_percent_is_approx_30_11_l304_30464


namespace train_speed_l304_30490

theorem train_speed :
  let train_length := 200 -- in meters
  let platform_length := 175.03 -- in meters
  let time_taken := 25 -- in seconds
  let total_distance := train_length + platform_length -- total distance in meters
  let speed_mps := total_distance / time_taken -- speed in meters per second
  let speed_kmph := speed_mps * 3.6 -- converting speed to kilometers per hour
  speed_kmph = 54.00432 := sorry

end train_speed_l304_30490


namespace inscribed_sphere_radius_of_tetrahedron_l304_30429

variables (V S1 S2 S3 S4 R : ℝ)

theorem inscribed_sphere_radius_of_tetrahedron
  (hV_pos : 0 < V)
  (hS_pos : 0 < S1) (hS2_pos : 0 < S2) (hS3_pos : 0 < S3) (hS4_pos : 0 < S4) :
  R = 3 * V / (S1 + S2 + S3 + S4) :=
sorry

end inscribed_sphere_radius_of_tetrahedron_l304_30429


namespace symmetric_points_on_parabola_l304_30416

theorem symmetric_points_on_parabola (x1 x2 y1 y2 m : ℝ)
  (h1: y1 = 2 * x1 ^ 2)
  (h2: y2 = 2 * x2 ^ 2)
  (h3: x1 * x2 = -1 / 2)
  (h4: y2 - y1 = 2 * (x2 ^ 2 - x1 ^ 2))
  (h5: (x1 + x2) / 2 = -1 / 4)
  (h6: (y1 + y2) / 2 = (x1 + x2) / 2 + m) :
  m = 3 / 2 :=
by
  sorry

end symmetric_points_on_parabola_l304_30416


namespace complex_ratio_proof_l304_30453

noncomputable def complex_ratio (x y : ℂ) : ℂ :=
  ((x^6 + y^6) / (x^6 - y^6)) - ((x^6 - y^6) / (x^6 + y^6))

theorem complex_ratio_proof (x y : ℂ) (h : ((x - y) / (x + y)) - ((x + y) / (x - y)) = 2) :
  complex_ratio x y = L :=
  sorry

end complex_ratio_proof_l304_30453


namespace longest_side_enclosure_l304_30410

variable (l w : ℝ)

theorem longest_side_enclosure (h1 : 2 * l + 2 * w = 240) (h2 : l * w = 1920) : max l w = 101 :=
sorry

end longest_side_enclosure_l304_30410


namespace inequality_a5_b5_c5_l304_30455

theorem inequality_a5_b5_c5 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^5 + b^5 + c^5 ≥ a^3 * b * c + a * b^3 * c + a * b * c^3 :=
by
  sorry

end inequality_a5_b5_c5_l304_30455


namespace max_product_of_xy_l304_30409

open Real

theorem max_product_of_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 4 * y = 1) :
  x * y ≤ 1 / 16 := 
sorry

end max_product_of_xy_l304_30409


namespace solve_equation_l304_30448

theorem solve_equation (x : ℝ) : 
  (9 - 3 * x) * (3 ^ x) - (x - 2) * (x ^ 2 - 5 * x + 6) = 0 ↔ x = 3 :=
by sorry

end solve_equation_l304_30448


namespace cassie_water_bottle_ounces_l304_30468

-- Define the given quantities
def cups_per_day : ℕ := 12
def ounces_per_cup : ℕ := 8
def refills_per_day : ℕ := 6

-- Define the total ounces of water Cassie drinks per day
def total_ounces_per_day := cups_per_day * ounces_per_cup

-- Define the ounces her water bottle holds
def ounces_per_bottle := total_ounces_per_day / refills_per_day

-- Prove the statement
theorem cassie_water_bottle_ounces : 
  ounces_per_bottle = 16 := by 
  sorry

end cassie_water_bottle_ounces_l304_30468


namespace intersection_of_sets_l304_30492

noncomputable def A : Set ℤ := {x | x^2 - 1 = 0}
def B : Set ℤ := {-1, 2, 5}

theorem intersection_of_sets : A ∩ B = {-1} :=
by
  sorry

end intersection_of_sets_l304_30492


namespace unit_digit_power3_58_l304_30499

theorem unit_digit_power3_58 : (3 ^ 58) % 10 = 9 := by
  -- proof steps will be provided here
  sorry

end unit_digit_power3_58_l304_30499


namespace total_weight_of_towels_is_40_lbs_l304_30426

def number_of_towels_Mary := 24
def factor_Mary_Frances := 4
def weight_Frances_towels_oz := 128
def pounds_per_ounce := 1 / 16

def number_of_towels_Frances := number_of_towels_Mary / factor_Mary_Frances

def total_number_of_towels := number_of_towels_Mary + number_of_towels_Frances
def weight_per_towel_oz := weight_Frances_towels_oz / number_of_towels_Frances

def total_weight_oz := total_number_of_towels * weight_per_towel_oz
def total_weight_lbs := total_weight_oz * pounds_per_ounce

theorem total_weight_of_towels_is_40_lbs :
  total_weight_lbs = 40 :=
sorry

end total_weight_of_towels_is_40_lbs_l304_30426


namespace b_power_a_equals_nine_l304_30452

theorem b_power_a_equals_nine (a b : ℝ) (h : |a - 2| + (b + 3)^2 = 0) : b^a = 9 := by
  sorry

end b_power_a_equals_nine_l304_30452


namespace fruits_in_good_condition_percentage_l304_30400

theorem fruits_in_good_condition_percentage (total_oranges total_bananas rotten_oranges_percentage rotten_bananas_percentage : ℝ) 
  (h1 : total_oranges = 600) 
  (h2 : total_bananas = 400) 
  (h3 : rotten_oranges_percentage = 0.15) 
  (h4 : rotten_bananas_percentage = 0.08) : 
  (1 - ((rotten_oranges_percentage * total_oranges + rotten_bananas_percentage * total_bananas) / (total_oranges + total_bananas))) * 100 = 87.8 :=
by 
  sorry

end fruits_in_good_condition_percentage_l304_30400


namespace foci_distance_of_hyperbola_l304_30456

theorem foci_distance_of_hyperbola :
  let a_sq := 25
  let b_sq := 9
  let c := Real.sqrt (a_sq + b_sq)
  2 * c = 2 * Real.sqrt 34 :=
by
  let a_sq := 25
  let b_sq := 9
  let c := Real.sqrt (a_sq + b_sq)
  sorry

end foci_distance_of_hyperbola_l304_30456


namespace contradiction_proof_l304_30481

theorem contradiction_proof (x y : ℝ) (h1 : x + y < 2) (h2 : 1 < x) (h3 : 1 < y) : false := 
by 
  sorry

end contradiction_proof_l304_30481


namespace range_of_a1_l304_30447

theorem range_of_a1 (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = 1 / (2 - a n)) 
  (h_pos : ∀ n, a (n + 1) > a n) : a 1 < 1 := 
sorry

end range_of_a1_l304_30447


namespace faster_pump_rate_ratio_l304_30441

theorem faster_pump_rate_ratio (S F : ℝ) 
  (h1 : S + F = 1/5) 
  (h2 : S = 1/12.5) : F / S = 1.5 :=
by
  sorry

end faster_pump_rate_ratio_l304_30441


namespace minimize_feed_costs_l304_30465

theorem minimize_feed_costs 
  (x y : ℝ)
  (h1: 5 * x + 3 * y ≥ 30)
  (h2: 2.5 * x + 3 * y ≥ 22.5)
  (h3: x ≥ 0)
  (h4: y ≥ 0)
  : (x = 3 ∧ y = 5) ∧ (x + y = 8) := 
sorry

end minimize_feed_costs_l304_30465


namespace largest_base4_is_largest_l304_30475

theorem largest_base4_is_largest 
  (n1 : ℕ) (n2 : ℕ) (n3 : ℕ) (n4 : ℕ)
  (h1 : n1 = 31) (h2 : n2 = 52) (h3 : n3 = 54) (h4 : n4 = 46) :
  n3 = Nat.max (Nat.max n1 n2) (Nat.max n3 n4) :=
by
  sorry

end largest_base4_is_largest_l304_30475


namespace greatest_whole_number_satisfying_inequalities_l304_30438

theorem greatest_whole_number_satisfying_inequalities :
  ∃ x : ℕ, 3 * (x : ℤ) - 5 < 1 - x ∧ 2 * (x : ℤ) + 4 ≤ 8 ∧ ∀ y : ℕ, y > x → ¬ (3 * (y : ℤ) - 5 < 1 - y ∧ 2 * (y : ℤ) + 4 ≤ 8) :=
sorry

end greatest_whole_number_satisfying_inequalities_l304_30438


namespace no_solution_ineq_l304_30445

theorem no_solution_ineq (m : ℝ) :
  (¬ ∃ (x : ℝ), x - 1 > 1 ∧ x < m) → m ≤ 2 :=
by
  sorry

end no_solution_ineq_l304_30445


namespace cylinder_surface_area_l304_30412

theorem cylinder_surface_area (r : ℝ) (l : ℝ) (h1 : r = 2) (h2 : l = 2 * r) : 
  2 * Real.pi * r^2 + 2 * Real.pi * r * l = 24 * Real.pi :=
by
  subst h1
  subst h2
  sorry

end cylinder_surface_area_l304_30412


namespace cookies_per_bag_l304_30407

theorem cookies_per_bag (b T : ℕ) (h1 : b = 37) (h2 : T = 703) : (T / b) = 19 :=
by
  -- Placeholder for proof
  sorry

end cookies_per_bag_l304_30407


namespace longest_tape_length_l304_30427

theorem longest_tape_length (a b c : ℕ) (h1 : a = 600) (h2 : b = 500) (h3 : c = 1200) : Nat.gcd (Nat.gcd a b) c = 100 :=
by
  sorry

end longest_tape_length_l304_30427


namespace cost_of_new_game_l304_30454

theorem cost_of_new_game (initial_money : ℕ) (money_left : ℕ) (toy_cost : ℕ) (toy_count : ℕ)
  (h_initial : initial_money = 68) (h_toy_cost : toy_cost = 7) (h_toy_count : toy_count = 3) 
  (h_money_left : money_left = toy_count * toy_cost) :
  initial_money - money_left = 47 :=
by {
  sorry
}

end cost_of_new_game_l304_30454


namespace school_minimum_payment_l304_30469

noncomputable def individual_ticket_price : ℝ := 6
noncomputable def group_ticket_price : ℝ := 40
noncomputable def discount : ℝ := 0.9
noncomputable def students : ℕ := 1258

-- Define the minimum amount the school should pay
noncomputable def minimum_amount := 4536

theorem school_minimum_payment :
  (students / 10 : ℝ) * group_ticket_price * discount + 
  (students % 10) * individual_ticket_price * discount = minimum_amount := sorry

end school_minimum_payment_l304_30469


namespace circle_radius_5_l304_30470

-- The circle equation given
def circle_eq (x y : ℝ) (c : ℝ) : Prop :=
  x^2 + 4 * x + y^2 + 8 * y + c = 0

-- The radius condition given
def radius_condition : Prop :=
  5 = (25 : ℝ).sqrt

-- The final proof statement
theorem circle_radius_5 (c : ℝ) : 
  (∀ x y : ℝ, circle_eq x y c) → radius_condition → c = -5 := 
by
  sorry

end circle_radius_5_l304_30470


namespace countDivisorsOf72Pow8_l304_30460

-- Definitions of conditions in Lean 4
def isPerfectSquare (a b : ℕ) : Prop := a % 2 = 0 ∧ b % 2 = 0
def isPerfectCube (a b : ℕ) : Prop := a % 3 = 0 ∧ b % 3 = 0
def isPerfectSixthPower (a b : ℕ) : Prop := a % 6 = 0 ∧ b % 6 = 0

def countPerfectSquares : ℕ := 13 * 9
def countPerfectCubes : ℕ := 9 * 6
def countPerfectSixthPowers : ℕ := 5 * 3

-- The proof problem to prove the number of such divisors is 156
theorem countDivisorsOf72Pow8:
  (countPerfectSquares + countPerfectCubes - countPerfectSixthPowers) = 156 :=
by
  sorry

end countDivisorsOf72Pow8_l304_30460


namespace range_of_a_l304_30443

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a*x^2 + (a+1)*x + a < 0) → a ∈ Set.Iio (-2 / 3) := 
sorry

end range_of_a_l304_30443


namespace remainder_when_divided_by_100_l304_30450

-- Define the given m
def m : ℕ := 76^2006 - 76

-- State the theorem
theorem remainder_when_divided_by_100 : m % 100 = 0 :=
by
  sorry

end remainder_when_divided_by_100_l304_30450


namespace modulus_complex_number_l304_30459

theorem modulus_complex_number (i : ℂ) (h : i = Complex.I) : 
  Complex.abs (1 / (i - 1)) = Real.sqrt 2 / 2 :=
by
  sorry

end modulus_complex_number_l304_30459


namespace total_books_count_l304_30497

theorem total_books_count (total_cost : ℕ) (math_book_cost : ℕ) (history_book_cost : ℕ) 
    (math_books_count : ℕ) (history_books_count : ℕ) (total_books : ℕ) :
    total_cost = 390 ∧ math_book_cost = 4 ∧ history_book_cost = 5 ∧ 
    math_books_count = 10 ∧ total_books = math_books_count + history_books_count ∧ 
    total_cost = (math_book_cost * math_books_count) + (history_book_cost * history_books_count) →
    total_books = 80 := by
  sorry

end total_books_count_l304_30497


namespace david_english_marks_l304_30466

theorem david_english_marks :
  let Mathematics := 45
  let Physics := 72
  let Chemistry := 77
  let Biology := 75
  let AverageMarks := 68.2
  let TotalSubjects := 5
  let TotalMarks := AverageMarks * TotalSubjects
  let MarksInEnglish := TotalMarks - (Mathematics + Physics + Chemistry + Biology)
  MarksInEnglish = 72 :=
by
  sorry

end david_english_marks_l304_30466


namespace fundraiser_total_money_l304_30473

def fundraiser_money : ℝ :=
  let brownies_students := 70
  let brownies_each := 20
  let brownies_price := 1.50
  let cookies_students := 40
  let cookies_each := 30
  let cookies_price := 2.25
  let donuts_students := 35
  let donuts_each := 18
  let donuts_price := 3.00
  let cupcakes_students := 25
  let cupcakes_each := 12
  let cupcakes_price := 2.50
  let total_brownies := brownies_students * brownies_each
  let total_cookies := cookies_students * cookies_each
  let total_donuts := donuts_students * donuts_each
  let total_cupcakes := cupcakes_students * cupcakes_each
  let money_brownies := total_brownies * brownies_price
  let money_cookies := total_cookies * cookies_price
  let money_donuts := total_donuts * donuts_price
  let money_cupcakes := total_cupcakes * cupcakes_price
  money_brownies + money_cookies + money_donuts + money_cupcakes

theorem fundraiser_total_money : fundraiser_money = 7440 := sorry

end fundraiser_total_money_l304_30473


namespace expand_expression_l304_30442

variable {x y z : ℝ}

theorem expand_expression :
  (2 * x + 5) * (3 * y + 15 + 4 * z) = 6 * x * y + 30 * x + 8 * x * z + 15 * y + 20 * z + 75 :=
by
  sorry

end expand_expression_l304_30442


namespace quadratic_roots_l304_30495

theorem quadratic_roots (m : ℝ) (h1 : m > 4) :
  (∃ x y : ℝ, x ≠ y ∧ (m-5) * x^2 - 2 * (m + 2) * x + m = 0 ∧ (m-5) * y^2 - 2 * (m + 2) * y + m = 0)
  ∨ (m = 5 ∧ ∃ x : ℝ, (m-5) * x^2 - 2 * (m + 2) * x + m = 0)
  ∨ (¬((∃ x y : ℝ, x ≠ y ∧ (m-5) * x^2 - 2 * (m + 2) * x + m = 0) ∨ (m = 5 ∧ ∃ x : ℝ, (m-5) * x^2 - 2 * (m + 2) * x + m = 0))) :=
by
  sorry

end quadratic_roots_l304_30495


namespace system_real_solutions_l304_30413

theorem system_real_solutions (a b c : ℝ) :
  (∃ x : ℝ, 
    a * x^2 + b * x + c = 0 ∧ 
    b * x^2 + c * x + a = 0 ∧ 
    c * x^2 + a * x + b = 0) ↔ 
  a + b + c = 0 :=
sorry

end system_real_solutions_l304_30413


namespace days_of_supply_l304_30408

-- Define the conditions as Lean definitions
def visits_per_day : ℕ := 3
def squares_per_visit : ℕ := 5
def total_rolls : ℕ := 1000
def squares_per_roll : ℕ := 300

-- Define the daily usage calculation
def daily_usage : ℕ := squares_per_visit * visits_per_day

-- Define the total squares calculation
def total_squares : ℕ := total_rolls * squares_per_roll

-- Define the proof statement for the number of days Bill's supply will last
theorem days_of_supply : (total_squares / daily_usage) = 20000 :=
by
  -- Placeholder for the actual proof, which is not required per instructions
  sorry

end days_of_supply_l304_30408


namespace set_proof_l304_30477

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {5, 6, 7}

theorem set_proof :
  (U \ A) ∩ (U \ B) = {4, 8} := by
  sorry

end set_proof_l304_30477


namespace multiples_of_7_are_128_l304_30414

theorem multiples_of_7_are_128 : 
  let range_start := 100
  let range_end := 999
  let multiple_7_smallest := 7 * 15
  let multiple_7_largest := 7 * 142
  let n_terms := (142 - 15 + 1)
  n_terms = 128 := sorry

end multiples_of_7_are_128_l304_30414


namespace find_multiple_of_q_l304_30488

theorem find_multiple_of_q
  (q : ℕ)
  (x : ℕ := 55 + 2 * q)
  (y : ℕ)
  (m : ℕ)
  (h1 : y = m * q + 41)
  (h2 : x = y)
  (h3 : q = 7) : m = 4 :=
by
  sorry

end find_multiple_of_q_l304_30488


namespace eqn_intersecting_straight_lines_l304_30487

theorem eqn_intersecting_straight_lines (x y : ℝ) : 
  x^2 - y^2 = 0 → (y = x ∨ y = -x) :=
by
  intros h
  sorry

end eqn_intersecting_straight_lines_l304_30487


namespace large_planter_holds_seeds_l304_30478

theorem large_planter_holds_seeds (total_seeds : ℕ) (small_planter_capacity : ℕ) (num_small_planters : ℕ) (num_large_planters : ℕ) 
  (h1 : total_seeds = 200)
  (h2 : small_planter_capacity = 4)
  (h3 : num_small_planters = 30)
  (h4 : num_large_planters = 4) : 
  (total_seeds - num_small_planters * small_planter_capacity) / num_large_planters = 20 := by
  sorry

end large_planter_holds_seeds_l304_30478


namespace sales_professionals_count_l304_30449

theorem sales_professionals_count :
  (∀ (C : ℕ) (MC : ℕ) (M : ℕ), C = 500 → MC = 10 → M = 5 → C / M / MC = 10) :=
by
  intros C MC M hC hMC hM
  sorry

end sales_professionals_count_l304_30449


namespace jian_wins_cases_l304_30431

inductive Move
| rock : Move
| paper : Move
| scissors : Move

def wins (jian shin : Move) : Prop :=
  (jian = Move.rock ∧ shin = Move.scissors) ∨
  (jian = Move.paper ∧ shin = Move.rock) ∨
  (jian = Move.scissors ∧ shin = Move.paper)

theorem jian_wins_cases : ∃ n : Nat, n = 3 ∧ (∀ jian shin, wins jian shin → n = 3) :=
by
  sorry

end jian_wins_cases_l304_30431


namespace range_of_a_l304_30419

theorem range_of_a {A : Set ℝ} (h1: ∀ x ∈ A, 2 * x + a > 0) (h2: 1 ∉ A) (h3: 2 ∈ A) : -4 < a ∧ a ≤ -2 := 
sorry

end range_of_a_l304_30419


namespace ratio_of_larger_to_smaller_l304_30440

noncomputable def ratio_of_numbers (a b : ℝ) : ℝ :=
a / b

theorem ratio_of_larger_to_smaller (a b : ℝ) (h1 : a + b = 7 * (a - b)) (h2 : a * b = 50) (h3 : a > b) :
  ratio_of_numbers a b = 4 / 3 :=
sorry

end ratio_of_larger_to_smaller_l304_30440


namespace find_numbers_l304_30405

theorem find_numbers (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sqrt (a * b) = Real.sqrt 5) ∧ 
  (2 * a * b / (a + b) = 5 / 3) → 
  (a = 5 ∧ b = 1) ∨ (a = 1 ∧ b = 5) := 
sorry

end find_numbers_l304_30405


namespace alcohol_quantity_l304_30434

theorem alcohol_quantity (A W : ℕ) (h1 : 4 * W = 3 * A) (h2 : 4 * (W + 8) = 5 * A) : A = 16 := 
by
  sorry

end alcohol_quantity_l304_30434


namespace cost_prices_max_units_B_possible_scenarios_l304_30415

-- Part 1: Prove cost prices of Product A and B
theorem cost_prices (x : ℝ) (A B : ℝ) 
  (h₁ : B = x ∧ A = x - 2) 
  (h₂ : 80 / A = 100 / B) 
  : B = 10 ∧ A = 8 :=
by 
  sorry

-- Part 2: Prove maximum units of product B that can be purchased
theorem max_units_B (y : ℕ) 
  (h₁ : ∀ y : ℕ, 3 * y - 5 + y ≤ 95) 
  : y ≤ 25 :=
by 
  sorry

-- Part 3: Prove possible scenarios for purchasing products A and B
theorem possible_scenarios (y : ℕ) 
  (h₁ : y > 23 * 9/17 ∧ y ≤ 25) 
  : y = 24 ∨ y = 25 :=
by 
  sorry

end cost_prices_max_units_B_possible_scenarios_l304_30415


namespace quadratic_inequality_solution_l304_30462

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 3 * x - 18 < 0} = {x : ℝ | -3 < x ∧ x < 6} :=
by
  sorry

end quadratic_inequality_solution_l304_30462


namespace electricity_bill_written_as_decimal_l304_30486

-- Definitions as conditions
def number : ℝ := 71.08

-- Proof statement
theorem electricity_bill_written_as_decimal : number = 71.08 :=
by sorry

end electricity_bill_written_as_decimal_l304_30486


namespace average_speed_correct_l304_30489

noncomputable def average_speed (d v_up v_down : ℝ) : ℝ :=
  let t_up := d / v_up
  let t_down := d / v_down
  let total_distance := 2 * d
  let total_time := t_up + t_down
  total_distance / total_time

theorem average_speed_correct :
  average_speed 0.2 24 36 = 28.8 := by {
  sorry
}

end average_speed_correct_l304_30489


namespace number_of_black_balls_l304_30467

variable (T : ℝ)
variable (red_balls : ℝ := 21)
variable (prop_red : ℝ := 0.42)
variable (prop_white : ℝ := 0.28)
variable (white_balls : ℝ := 0.28 * T)

noncomputable def total_balls : ℝ := red_balls / prop_red

theorem number_of_black_balls :
  T = total_balls → 
  ∃ black_balls : ℝ, black_balls = total_balls - red_balls - white_balls ∧ black_balls = 15 := 
by
  intro hT
  let black_balls := total_balls - red_balls - white_balls
  use black_balls
  simp [total_balls]
  sorry

end number_of_black_balls_l304_30467


namespace macey_needs_to_save_three_more_weeks_l304_30401

def cost_of_shirt : ℝ := 3.0
def amount_saved : ℝ := 1.5
def saving_per_week : ℝ := 0.5

theorem macey_needs_to_save_three_more_weeks :
  ∃ W : ℝ, W * saving_per_week = cost_of_shirt - amount_saved ∧ W = 3 := by
  sorry

end macey_needs_to_save_three_more_weeks_l304_30401


namespace largest_circle_radius_l304_30463

noncomputable def largest_inscribed_circle_radius (AB BC CD DA : ℝ) : ℝ :=
  let s := (AB + BC + CD + DA) / 2
  let A := Real.sqrt ((s - AB) * (s - BC) * (s - CD) * (s - DA))
  A / s

theorem largest_circle_radius {AB BC CD DA : ℝ} (hAB : AB = 10) (hBC : BC = 11) (hCD : CD = 6) (hDA : DA = 13)
  : largest_inscribed_circle_radius AB BC CD DA = 3 * Real.sqrt 245 / 10 :=
by
  simp [largest_inscribed_circle_radius, hAB, hBC, hCD, hDA]
  sorry

end largest_circle_radius_l304_30463


namespace range_of_a_l304_30446

noncomputable def A : Set ℝ := {x | x ≥ abs (x^2 - 2 * x)}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 - 2 * a * x + a ≤ 0}

theorem range_of_a (a : ℝ) : (A ∩ B a = B a) → (0 ≤ a ∧ a ≤ 1) :=
by
  sorry

end range_of_a_l304_30446


namespace graph_of_x2_minus_y2_eq_0_is_two_intersecting_lines_l304_30482

theorem graph_of_x2_minus_y2_eq_0_is_two_intersecting_lines :
  ∀ x y : ℝ, (x^2 - y^2 = 0) ↔ (y = x ∨ y = -x) := 
by
  sorry

end graph_of_x2_minus_y2_eq_0_is_two_intersecting_lines_l304_30482


namespace margo_donation_l304_30479

variable (M J : ℤ)

theorem margo_donation (h1: J = 4700) (h2: (|J - M| / 2) = 200) : M = 4300 :=
sorry

end margo_donation_l304_30479


namespace find_rate_of_interest_l304_30423

-- Define the problem conditions
def principal_B : ℝ := 4000
def principal_C : ℝ := 2000
def time_B : ℝ := 2
def time_C : ℝ := 4
def total_interest : ℝ := 2200

-- Define the unknown rate of interest per annum
noncomputable def rate_of_interest (R : ℝ) : Prop :=
  let interest_B := (principal_B * R * time_B) / 100
  let interest_C := (principal_C * R * time_C) / 100
  interest_B + interest_C = total_interest

-- Statement to prove that the rate of interest is 13.75%
theorem find_rate_of_interest : rate_of_interest 13.75 := by
  sorry

end find_rate_of_interest_l304_30423


namespace used_mystery_books_l304_30430

theorem used_mystery_books (total_books used_adventure_books new_crime_books : ℝ)
  (h1 : total_books = 45)
  (h2 : used_adventure_books = 13.0)
  (h3 : new_crime_books = 15.0) :
  total_books - (used_adventure_books + new_crime_books) = 17.0 := by
  sorry

end used_mystery_books_l304_30430


namespace total_pink_crayons_l304_30461

-- Define the conditions
def Mara_crayons : ℕ := 40
def Mara_pink_percent : ℕ := 10
def Luna_crayons : ℕ := 50
def Luna_pink_percent : ℕ := 20

-- Define the proof problem statement
theorem total_pink_crayons : 
  (Mara_crayons * Mara_pink_percent / 100) + (Luna_crayons * Luna_pink_percent / 100) = 14 := 
by sorry

end total_pink_crayons_l304_30461


namespace remainder_of_concatenated_number_l304_30491

def concatenated_number : ℕ :=
  -- Definition of the concatenated number
  -- That is 123456789101112...4344
  -- For simplicity, we'll just assign it directly
  1234567891011121314151617181920212223242526272829303132333435363738394041424344

theorem remainder_of_concatenated_number :
  concatenated_number % 45 = 9 :=
sorry

end remainder_of_concatenated_number_l304_30491


namespace find_x_l304_30472

theorem find_x (a b x : ℕ) (h1 : a = 105) (h2 : b = 147) (h3 : a^3 = 21 * x * 15 * b) : x = 25 :=
by
  -- This is where the proof would go
  sorry

end find_x_l304_30472


namespace ball_height_intersect_l304_30474

noncomputable def ball_height (h : ℝ) (t₁ t₂ : ℝ) (h₁ h₂ : ℝ → ℝ) : Prop :=
  (∀ t, h₁ t = h₂ (t - 1) ↔ t = t₁) ∧
  (h₁ t₁ = h ∧ h₂ t₁ = h) ∧ 
  (∀ t, h₂ (t - 1) = h₁ t) ∧ 
  (h₁ (1.1) = h ∧ h₂ (1.1) = h)

theorem ball_height_intersect (h : ℝ)
  (h₁ h₂ : ℝ → ℝ)
  (h_max : ∀ t₁ t₂, ball_height h t₁ t₂ h₁ h₂) :
  (∃ t₁, t₁ = 1.6) :=
sorry

end ball_height_intersect_l304_30474


namespace problem1_problem2_l304_30498

-- Definitions of the conditions
def periodic_func (f: ℝ → ℝ) (a: ℝ) (x: ℝ) : Prop :=
(∀ x, f (x + 3) = f x) ∧ 
(∀ x, -2 ≤ x ∧ x < 0 → f x = x + a) ∧ 
(∀ x, 0 ≤ x ∧ x < 1 → f x = (1/2)^x)

-- 1. Prove f(13/2) = sqrt(2)/2
theorem problem1 (f: ℝ → ℝ) (a: ℝ) (h: periodic_func f a x) : f (13/2) = (Real.sqrt 2) / 2 := 
sorry

-- 2. Prove that if f(x) has a minimum value but no maximum value, then 1 < a ≤ 5/2
theorem problem2 (f: ℝ → ℝ) (a: ℝ) (h: periodic_func f a x) (hmin: ∃ m, ∀ x, f x ≥ m) (hmax: ¬∃ M, ∀ x, f x ≤ M) : 1 < a ∧ a ≤ 5/2 :=
sorry

end problem1_problem2_l304_30498


namespace work_completion_l304_30425

theorem work_completion 
  (x_work_days : ℕ) 
  (y_work_days : ℕ) 
  (y_worked_days : ℕ) 
  (x_rate := 1 / (x_work_days : ℚ)) 
  (y_rate := 1 / (y_work_days : ℚ)) 
  (work_remaining := 1 - y_rate * y_worked_days) 
  (remaining_work_days := work_remaining / x_rate) : 
  x_work_days = 18 → 
  y_work_days = 15 → 
  y_worked_days = 5 → 
  remaining_work_days = 12 := 
by
  intros
  sorry

end work_completion_l304_30425


namespace order_of_values_l304_30437

noncomputable def a : ℝ := Real.log 2 / 2
noncomputable def b : ℝ := Real.log 3 / 3
noncomputable def c : ℝ := Real.log Real.pi / Real.pi
noncomputable def d : ℝ := Real.log 2.72 / 2.72
noncomputable def f : ℝ := (Real.sqrt 10 * Real.log 10) / 20

theorem order_of_values : a < f ∧ f < c ∧ c < b ∧ b < d :=
by
  sorry

end order_of_values_l304_30437


namespace smallest_b_value_l304_30476

noncomputable def smallest_b (a b : ℝ) : ℝ :=
if a > 2 ∧ 2 < a ∧ a < b 
   ∧ (2 + a ≤ b) 
   ∧ ((1 / a) + (1 / b) ≤ 1 / 2) 
then b else 0

theorem smallest_b_value : ∀ (a b : ℝ), 
  (2 < a) → (a < b) → (2 + a ≤ b) → 
  ((1 / a) + (1 / b) ≤ 1 / 2) → 
  b = 3 + Real.sqrt 5 := sorry

end smallest_b_value_l304_30476


namespace total_simple_interest_is_correct_l304_30493

noncomputable def principal : ℝ := 15041.875
noncomputable def rate : ℝ := 8
noncomputable def time : ℝ := 5
noncomputable def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

theorem total_simple_interest_is_correct :
  simple_interest principal rate time = 6016.75 := 
sorry

end total_simple_interest_is_correct_l304_30493


namespace least_common_multiple_increments_l304_30411

theorem least_common_multiple_increments :
  let a := 4; let b := 6; let c := 12; let d := 18
  let a' := a + 1; let b' := b + 1; let c' := c + 1; let d' := d + 1
  Nat.lcm (Nat.lcm (Nat.lcm a' b') c') d' = 8645 :=
by
  let a := 4; let b := 6; let c := 12; let d := 18
  let a' := a + 1; let b' := b + 1; let c' := c + 1; let d' := d + 1
  sorry

end least_common_multiple_increments_l304_30411


namespace total_prime_ending_starting_numerals_l304_30417

def single_digit_primes : List ℕ := [2, 3, 5, 7]
def number_of_possible_digits := 10

def count_3digit_numerals : ℕ :=
  4 * number_of_possible_digits * 4

def count_4digit_numerals : ℕ :=
  4 * number_of_possible_digits * number_of_possible_digits * 4

theorem total_prime_ending_starting_numerals : 
  count_3digit_numerals + count_4digit_numerals = 1760 := by
sorry

end total_prime_ending_starting_numerals_l304_30417


namespace abc_divisibility_l304_30421

theorem abc_divisibility (a b c : ℕ) (h₁ : a ∣ (b * c - 1)) (h₂ : b ∣ (c * a - 1)) (h₃ : c ∣ (a * b - 1)) : 
  (a = 2 ∧ b = 3 ∧ c = 5) ∨ (a = 1 ∧ b = 1 ∧ ∃ n : ℕ, n ≥ 1 ∧ c = n) :=
by
  sorry

end abc_divisibility_l304_30421


namespace measure_of_alpha_l304_30436

theorem measure_of_alpha
  (A B D α : ℝ)
  (hA : A = 50)
  (hB : B = 150)
  (hD : D = 140)
  (quadrilateral_sum : A + B + D + α = 360) : α = 20 :=
by
  rw [hA, hB, hD] at quadrilateral_sum
  sorry

end measure_of_alpha_l304_30436


namespace first_number_in_expression_l304_30451

theorem first_number_in_expression (a b c d e : ℝ)
  (h_expr : (a * b * c) / d + e = 2229) :
  a = 26.3 :=
  sorry

end first_number_in_expression_l304_30451


namespace evaluate_expression_l304_30484

theorem evaluate_expression :
  let a := 12
  let b := 14
  let c := 18
  (144 * ((1:ℝ)/b - (1:ℝ)/c) + 196 * ((1:ℝ)/c - (1:ℝ)/a) + 324 * ((1:ℝ)/a - (1:ℝ)/b)) /
  (a * ((1:ℝ)/b - (1:ℝ)/c) + b * ((1:ℝ)/c - (1:ℝ)/a) + c * ((1:ℝ)/a - (1:ℝ)/b)) = a + b + c := by
  sorry

end evaluate_expression_l304_30484


namespace pets_beds_calculation_l304_30496

theorem pets_beds_calculation
  (initial_beds : ℕ)
  (additional_beds : ℕ)
  (total_pets : ℕ)
  (H1 : initial_beds = 12)
  (H2 : additional_beds = 8)
  (H3 : total_pets = 10) :
  (initial_beds + additional_beds) / total_pets = 2 := 
by 
  sorry

end pets_beds_calculation_l304_30496


namespace equipment_total_cost_l304_30457

def cost_jersey : ℝ := 25
def cost_shorts : ℝ := 15.20
def cost_socks : ℝ := 6.80
def cost_cleats : ℝ := 40
def cost_water_bottle : ℝ := 12
def cost_one_player := cost_jersey + cost_shorts + cost_socks + cost_cleats + cost_water_bottle
def num_players : ℕ := 25
def total_cost_for_team : ℝ := cost_one_player * num_players

theorem equipment_total_cost :
  total_cost_for_team = 2475 := by
  sorry

end equipment_total_cost_l304_30457


namespace exists_prime_q_and_positive_n_l304_30420

theorem exists_prime_q_and_positive_n (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : p > 5) :
  ∃ q n : ℕ, Nat.Prime q ∧ q < p ∧ 0 < n ∧ p ∣ (n^2 - q) :=
by
  sorry

end exists_prime_q_and_positive_n_l304_30420


namespace sum_of_x_and_y_l304_30418

theorem sum_of_x_and_y (x y : ℤ) (h1 : 3 + x = 5) (h2 : -3 + y = 5) : x + y = 10 :=
by
  sorry

end sum_of_x_and_y_l304_30418


namespace sum_of_reciprocals_of_numbers_l304_30485

theorem sum_of_reciprocals_of_numbers (x y : ℕ) (h_sum : x + y = 45) (h_hcf : Nat.gcd x y = 3)
    (h_lcm : Nat.lcm x y = 100) : 1/x + 1/y = 3/20 := 
by 
  sorry

end sum_of_reciprocals_of_numbers_l304_30485


namespace average_monthly_bill_l304_30471

-- Definitions based on conditions
def first_4_months_average := 30
def last_2_months_average := 24
def first_4_months_total := 4 * first_4_months_average
def last_2_months_total := 2 * last_2_months_average
def total_spent := first_4_months_total + last_2_months_total
def total_months := 6

-- The theorem statement
theorem average_monthly_bill : total_spent / total_months = 28 := by
  sorry

end average_monthly_bill_l304_30471


namespace total_animals_l304_30403

-- Define the number of pigs and giraffes
def num_pigs : ℕ := 7
def num_giraffes : ℕ := 6

-- Theorem stating the total number of giraffes and pigs
theorem total_animals : num_pigs + num_giraffes = 13 :=
by sorry

end total_animals_l304_30403


namespace students_more_than_rabbits_l304_30433

/- Define constants for the problem. -/
def students_per_class : ℕ := 20
def rabbits_per_class : ℕ := 3
def num_classes : ℕ := 5

/- Define total counts based on given conditions. -/
def total_students : ℕ := students_per_class * num_classes
def total_rabbits : ℕ := rabbits_per_class * num_classes

/- The theorem we need to prove: The difference between total students and total rabbits is 85. -/
theorem students_more_than_rabbits : total_students - total_rabbits = 85 := by
  sorry

end students_more_than_rabbits_l304_30433


namespace find_missing_edge_l304_30483

-- Define the known parameters
def volume : ℕ := 80
def edge1 : ℕ := 2
def edge3 : ℕ := 8

-- Define the missing edge
def missing_edge : ℕ := 5

-- State the problem
theorem find_missing_edge (volume : ℕ) (edge1 : ℕ) (edge3 : ℕ) (missing_edge : ℕ) :
  volume = edge1 * missing_edge * edge3 →
  missing_edge = 5 :=
by
  sorry

end find_missing_edge_l304_30483
