import Mathlib

namespace incorrect_operation_B_l1872_187265

variables (a b c : ℝ)

theorem incorrect_operation_B : (c - 2 * (a + b)) ≠ (c - 2 * a + 2 * b) := by
  sorry

end incorrect_operation_B_l1872_187265


namespace grapefruits_orchards_proof_l1872_187270

/-- 
Given the following conditions:
1. There are 40 orchards in total.
2. 15 orchards are dedicated to lemons.
3. The number of orchards for oranges is two-thirds of the number of orchards for lemons.
4. Limes and grapefruits have an equal number of orchards.
5. Mandarins have half as many orchards as limes or grapefruits.
Prove that the number of citrus orchards growing grapefruits is 6.
-/
def num_grapefruit_orchards (TotalOrchards Lemons Oranges L G M : ℕ) : Prop :=
  TotalOrchards = 40 ∧
  Lemons = 15 ∧
  Oranges = 2 * Lemons / 3 ∧
  L = G ∧
  M = G / 2 ∧
  L + G + M = TotalOrchards - (Lemons + Oranges) ∧
  G = 6

theorem grapefruits_orchards_proof : ∃ (TotalOrchards Lemons Oranges L G M : ℕ), num_grapefruit_orchards TotalOrchards Lemons Oranges L G M :=
by
  sorry

end grapefruits_orchards_proof_l1872_187270


namespace ceiling_and_floor_calculation_l1872_187223

theorem ceiling_and_floor_calculation : 
  let a := (15 : ℚ) / 8
  let b := (-34 : ℚ) / 4
  Int.ceil (a * b) - Int.floor (a * Int.floor b) = 2 :=
by
  sorry

end ceiling_and_floor_calculation_l1872_187223


namespace evaluate_b3_l1872_187263

variable (b1 q : ℤ)
variable (b1_cond : b1 = 5 ∨ b1 = -5)
variable (q_cond : q = 3 ∨ q = -3)
def b3 : ℤ := b1 * q^2

theorem evaluate_b3 (h : b1^2 * (1 + q^2 + q^4) = 2275) : b3 = 45 ∨ b3 = -45 :=
by sorry

end evaluate_b3_l1872_187263


namespace ice_cream_cost_l1872_187225

theorem ice_cream_cost
  (num_pennies : ℕ) (num_nickels : ℕ) (num_dimes : ℕ) (num_quarters : ℕ) 
  (leftover_cents : ℤ) (num_family_members : ℕ)
  (h_pennies : num_pennies = 123)
  (h_nickels : num_nickels = 85)
  (h_dimes : num_dimes = 35)
  (h_quarters : num_quarters = 26)
  (h_leftover : leftover_cents = 48)
  (h_members : num_family_members = 5) :
  (123 * 0.01 + 85 * 0.05 + 35 * 0.1 + 26 * 0.25 - 0.48) / 5 = 3 :=
by
  sorry

end ice_cream_cost_l1872_187225


namespace minimum_soldiers_to_add_l1872_187233

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  ∃ k : ℕ, 84 * k + 2 - N = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l1872_187233


namespace clock_angle_230_l1872_187232

theorem clock_angle_230 (h12 : ℕ := 12) (deg360 : ℕ := 360) 
  (hour_mark_deg : ℕ := deg360 / h12) (hour_halfway : ℕ := hour_mark_deg / 2)
  (hour_deg_230 : ℕ := hour_mark_deg * 3) (total_angle : ℕ := hour_halfway + hour_deg_230) :
  total_angle = 105 := 
by
  sorry

end clock_angle_230_l1872_187232


namespace pet_store_customers_buy_different_pets_l1872_187284

theorem pet_store_customers_buy_different_pets :
  let puppies := 20
  let kittens := 10
  let hamsters := 12
  let rabbits := 5
  let customers := 4
  (puppies * kittens * hamsters * rabbits * Nat.factorial customers = 288000) := 
by
  sorry

end pet_store_customers_buy_different_pets_l1872_187284


namespace dress_assignment_l1872_187227

variables {Girl : Type} [Finite Girl]
variables (Katya Olya Liza Rita Pink Green Yellow Blue : Girl)
variables (standing_between : Girl → Girl → Girl → Prop)

-- Conditions
variable (cond1 : Katya ≠ Pink ∧ Katya ≠ Blue)
variable (cond2 : standing_between Green Liza Yellow)
variable (cond3 : Rita ≠ Green ∧ Rita ≠ Blue)
variable (cond4 : standing_between Olya Rita Pink)

-- Theorem statement
theorem dress_assignment :
  Katya = Green ∧ Olya = Blue ∧ Liza = Pink ∧ Rita = Yellow := 
sorry

end dress_assignment_l1872_187227


namespace no_solution_l1872_187201

theorem no_solution : ∀ x y z t : ℕ, 16^x + 21^y + 26^z ≠ t^2 :=
by
  intro x y z t
  sorry

end no_solution_l1872_187201


namespace felix_chopped_down_trees_l1872_187230

theorem felix_chopped_down_trees
  (sharpening_cost : ℕ)
  (trees_per_sharpening : ℕ)
  (total_spent : ℕ)
  (times_sharpened : ℕ)
  (trees_chopped_down : ℕ)
  (h1 : sharpening_cost = 5)
  (h2 : trees_per_sharpening = 13)
  (h3 : total_spent = 35)
  (h4 : times_sharpened = total_spent / sharpening_cost)
  (h5 : trees_chopped_down = trees_per_sharpening * times_sharpened) :
  trees_chopped_down ≥ 91 :=
by
  sorry

end felix_chopped_down_trees_l1872_187230


namespace exponential_inequality_l1872_187235

theorem exponential_inequality (a x1 x2 : ℝ) (h1 : 1 < a) (h2 : x1 < x2) :
  |a ^ ((1 / 2) * (x1 + x2)) - a ^ x1| < |a ^ x2 - a ^ ((1 / 2) * (x1 + x2))| :=
by
  sorry

end exponential_inequality_l1872_187235


namespace add_decimals_l1872_187204

theorem add_decimals :
  5.467 + 3.92 = 9.387 :=
by
  sorry

end add_decimals_l1872_187204


namespace library_visit_period_l1872_187260

noncomputable def dance_class_days := 6
noncomputable def karate_class_days := 12
noncomputable def common_days := 36

theorem library_visit_period (library_days : ℕ) 
  (hdance : ∀ (n : ℕ), n * dance_class_days = common_days)
  (hkarate : ∀ (n : ℕ), n * karate_class_days = common_days)
  (hcommon : ∀ (n : ℕ), n * library_days = common_days) : 
  library_days = 18 := 
sorry

end library_visit_period_l1872_187260


namespace central_angle_measures_l1872_187281

-- Definitions for the conditions
def perimeter_eq (r l : ℝ) : Prop := l + 2 * r = 6
def area_eq (r l : ℝ) : Prop := (1 / 2) * l * r = 2
def central_angle (r l α : ℝ) : Prop := α = l / r

-- The final proof statement
theorem central_angle_measures (r l α : ℝ) (h1 : perimeter_eq r l) (h2 : area_eq r l) :
  central_angle r l α → (α = 1 ∨ α = 4) :=
sorry

end central_angle_measures_l1872_187281


namespace solve_system_of_equations_l1872_187202

theorem solve_system_of_equations (x y : Real) : 
  (3 * x^2 + 3 * y^2 - 2 * x^2 * y^2 = 3) ∧ 
  (x^4 + y^4 + (2/3) * x^2 * y^2 = 17) ↔
  ( (x = Real.sqrt 2 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3 )) ∨ 
    (x = -Real.sqrt 2 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3)) ∨ 
    (x = Real.sqrt 3 ∧ (y = Real.sqrt 2 ∨ y = -Real.sqrt 2 )) ∨ 
    (x = -Real.sqrt 3 ∧ (y = Real.sqrt 2 ∨ y = -Real.sqrt 2 )) ) := 
  by
    sorry

end solve_system_of_equations_l1872_187202


namespace faster_car_distance_l1872_187288

theorem faster_car_distance (d v : ℝ) (h_dist: d + 2 * d = 4) (h_faster: 2 * v = 2 * (d / v)) : 
  d = 4 / 3 → 2 * d = 8 / 3 :=
by sorry

end faster_car_distance_l1872_187288


namespace probability_range_inequality_l1872_187287

theorem probability_range_inequality :
  ∀ p : ℝ, 0 ≤ p → p ≤ 1 →
  (4 * p * (1 - p)^3 ≤ 6 * p^2 * (1 - p)^2 → 0.4 ≤ p ∧ p < 1) := sorry

end probability_range_inequality_l1872_187287


namespace non_zero_digits_of_fraction_l1872_187285

def fraction : ℚ := 80 / (2^4 * 5^9)

def decimal_expansion (x : ℚ) : String :=
  -- some function to compute the decimal expansion of a fraction as a string
  "0.00000256" -- placeholder

def non_zero_digits_to_right (s : String) : ℕ :=
  -- some function to count the number of non-zero digits to the right of the decimal point in the string
  3 -- placeholder

theorem non_zero_digits_of_fraction : non_zero_digits_to_right (decimal_expansion fraction) = 3 := by
  sorry

end non_zero_digits_of_fraction_l1872_187285


namespace laura_change_l1872_187215

-- Define the cost of a pair of pants and a shirt.
def cost_of_pants := 54
def cost_of_shirts := 33

-- Define the number of pants and shirts Laura bought.
def num_pants := 2
def num_shirts := 4

-- Define the amount Laura gave to the cashier.
def amount_given := 250

-- Calculate the total cost.
def total_cost := num_pants * cost_of_pants + num_shirts * cost_of_shirts

-- Define the expected change.
def expected_change := 10

-- The main theorem stating the problem and its solution.
theorem laura_change :
  amount_given - total_cost = expected_change :=
by
  sorry

end laura_change_l1872_187215


namespace total_weight_is_28_87_l1872_187279

def blue_ball_weight : ℝ := 6
def brown_ball_weight : ℝ := 3.12
def green_ball_weight : ℝ := 4.25

def red_ball_weight : ℝ := 2 * green_ball_weight
def yellow_ball_weight : ℝ := red_ball_weight - 1.5

def total_weight : ℝ := blue_ball_weight + brown_ball_weight + green_ball_weight + red_ball_weight + yellow_ball_weight

theorem total_weight_is_28_87 : total_weight = 28.87 :=
by
  /- proof goes here -/
  sorry

end total_weight_is_28_87_l1872_187279


namespace Bryce_raisins_l1872_187290

theorem Bryce_raisins (B C : ℚ) (h1 : B = C + 10) (h2 : C = B / 4) : B = 40 / 3 :=
by
 -- The proof goes here, but we skip it for now
 sorry

end Bryce_raisins_l1872_187290


namespace digit_x_base_7_l1872_187205

theorem digit_x_base_7 (x : ℕ) : 
    (4 * 7^3 + 5 * 7^2 + x * 7 + 2) % 9 = 0 → x = 4 := 
by {
    sorry
}

end digit_x_base_7_l1872_187205


namespace books_and_games_left_to_experience_l1872_187203

def booksLeft (B_total B_read : Nat) : Nat := B_total - B_read
def gamesLeft (G_total G_played : Nat) : Nat := G_total - G_played
def totalLeft (B_total B_read G_total G_played : Nat) : Nat := booksLeft B_total B_read + gamesLeft G_total G_played

theorem books_and_games_left_to_experience :
  totalLeft 150 74 50 17 = 109 := by
  sorry

end books_and_games_left_to_experience_l1872_187203


namespace wrapping_paper_area_l1872_187237

theorem wrapping_paper_area 
  (l w h : ℝ) :
  (l + 4 + 2 * h) ^ 2 = l^2 + 8 * l + 16 + 4 * l * h + 16 * h + 4 * h^2 := 
by 
  sorry

end wrapping_paper_area_l1872_187237


namespace max_candy_leftover_l1872_187217

theorem max_candy_leftover (x : ℕ) : (∃ k : ℕ, x = 12 * k + 11) → (x % 12 = 11) :=
by
  sorry

end max_candy_leftover_l1872_187217


namespace min_employees_birthday_Wednesday_l1872_187236

theorem min_employees_birthday_Wednesday (W D : ℕ) (h_eq : W + 6 * D = 50) (h_gt : W > D) : W = 8 :=
sorry

end min_employees_birthday_Wednesday_l1872_187236


namespace min_dot_product_l1872_187283

noncomputable def ellipse_eq_p (x y : ℝ) : Prop :=
    x^2 / 9 + y^2 / 8 = 1

noncomputable def dot_product_op_fp (x y : ℝ) : ℝ :=
    x^2 + x + y^2

theorem min_dot_product : 
    (∀ x y : ℝ, ellipse_eq_p x y → dot_product_op_fp x y = 6) := 
sorry

end min_dot_product_l1872_187283


namespace solution_set_inequality_l1872_187209

theorem solution_set_inequality (x : ℝ) : |3 * x + 1| - |x - 1| < 0 ↔ -1 < x ∧ x < 0 := 
sorry

end solution_set_inequality_l1872_187209


namespace second_discount_percentage_l1872_187207

theorem second_discount_percentage
  (original_price : ℝ)
  (first_discount : ℝ)
  (final_price : ℝ)
  (second_discount : ℝ) :
  original_price = 10000 →
  first_discount = 0.20 →
  final_price = 6840 →
  second_discount = 14.5 :=
by
  sorry

end second_discount_percentage_l1872_187207


namespace find_certain_number_multiplied_by_24_l1872_187286

-- Define the conditions
theorem find_certain_number_multiplied_by_24 :
  (∃ x : ℤ, 37 - x = 24) →
  ∀ x : ℤ, (37 - x = 24) → (x * 24 = 312) :=
by
  intros h x hx
  -- Here we will have the proof using the assumption and the theorem.
  sorry

end find_certain_number_multiplied_by_24_l1872_187286


namespace arithmetic_sequence_a6_l1872_187254

theorem arithmetic_sequence_a6 (a : ℕ → ℕ)
  (h_arith_seq : ∀ n, ∃ d, a (n+1) = a n + d)
  (h_sum : a 4 + a 8 = 16) : a 6 = 8 :=
sorry

end arithmetic_sequence_a6_l1872_187254


namespace term_with_largest_binomial_coeffs_and_largest_coefficient_l1872_187213

theorem term_with_largest_binomial_coeffs_and_largest_coefficient :
  ∀ x : ℝ,
    (∀ k : ℕ, k = 2 → (Nat.choose 5 k) * (x ^ (2 / 3)) ^ (5 - k) * (3 * x ^ 2) ^ k = 90 * x ^ 6) ∧
    (∀ k : ℕ, k = 3 → (Nat.choose 5 k) * (x ^ (2 / 3)) ^ (5 - k) * (3 * x ^ 2) ^ k = 270 * x ^ (22 / 3)) ∧
    (∀ r : ℕ, r = 4 → (Nat.choose 5 4) * (x ^ (2 / 3)) ^ (5 - 4) * (3 * x ^ 2) ^ 4 = 405 * x ^ (26 / 3)) :=
by sorry

end term_with_largest_binomial_coeffs_and_largest_coefficient_l1872_187213


namespace sqrt_of_9_fact_over_84_eq_24_sqrt_15_l1872_187293

theorem sqrt_of_9_fact_over_84_eq_24_sqrt_15 :
  Real.sqrt (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1 / (2^2 * 3 * 7)) = 24 * Real.sqrt 15 :=
by
  sorry

end sqrt_of_9_fact_over_84_eq_24_sqrt_15_l1872_187293


namespace negation_of_p_l1872_187289

theorem negation_of_p :
  (∃ x : ℝ, x < 0 ∧ x + (1 / x) > -2) ↔ ¬ (∀ x : ℝ, x < 0 → x + (1 / x) ≤ -2) :=
by {
  sorry
}

end negation_of_p_l1872_187289


namespace smallest_b_factor_2020_l1872_187278

theorem smallest_b_factor_2020 :
  ∃ b : ℕ, b > 0 ∧
  (∃ r s : ℕ, r > s ∧ r * s = 2020 ∧ b = r + s) ∧
  (∀ c : ℕ, c > 0 → (∃ r s : ℕ, r > s ∧ r * s = 2020 ∧ c = r + s) → b ≤ c) ∧
  b = 121 :=
sorry

end smallest_b_factor_2020_l1872_187278


namespace trapezoid_perimeter_l1872_187246

theorem trapezoid_perimeter (AB CD BC DA : ℝ) (BCD_angle : ℝ)
  (h1 : AB = 60) (h2 : CD = 40) (h3 : BC = DA) (h4 : BCD_angle = 120) :
  AB + BC + CD + DA = 220 := 
sorry

end trapezoid_perimeter_l1872_187246


namespace max_cookies_Andy_can_eat_l1872_187239

theorem max_cookies_Andy_can_eat 
  (x y : ℕ) 
  (h1 : x + y = 36)
  (h2 : y ≥ 2 * x) : 
  x ≤ 12 := by
  sorry

end max_cookies_Andy_can_eat_l1872_187239


namespace appears_more_than_three_times_in_Pascal_appears_more_than_four_times_in_Pascal_l1872_187249

-- Definitions for binomial coefficient and Pascal's triangle

-- Define binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Check occurrences in Pascal's triangle more than three times
theorem appears_more_than_three_times_in_Pascal (n : ℕ) :
  n = 10 ∨ n = 15 ∨ n = 21 → ∃ a b c : ℕ, 
    (1 < a) ∧ (1 < b) ∧ (1 < c) ∧ 
    (binomial_coeff a 2 = n ∨ binomial_coeff a 3 = n) ∧
    (binomial_coeff b 2 = n ∨ binomial_coeff b 3 = n) ∧
    (binomial_coeff c 2 = n ∨ binomial_coeff c 3 = n) := 
by
  sorry

-- Check occurrences in Pascal's triangle more than four times
theorem appears_more_than_four_times_in_Pascal (n : ℕ) :
  n = 120 ∨ n = 210 ∨ n = 3003 → ∃ a b c d : ℕ, 
    (1 < a) ∧ (1 < b) ∧ (1 < c) ∧ (1 < d) ∧ 
    (binomial_coeff a 3 = n ∨ binomial_coeff a 4 = n) ∧
    (binomial_coeff b 3 = n ∨ binomial_coeff b 4 = n) ∧
    (binomial_coeff c 3 = n ∨ binomial_coeff c 4 = n) ∧
    (binomial_coeff d 3 = n ∨ binomial_coeff d 4 = n) := 
by
  sorry

end appears_more_than_three_times_in_Pascal_appears_more_than_four_times_in_Pascal_l1872_187249


namespace citizen_income_l1872_187214

theorem citizen_income (I : ℝ) 
  (h1 : I > 0)
  (h2 : 0.12 * 40000 + 0.20 * (I - 40000) = 8000) : 
  I = 56000 := 
sorry

end citizen_income_l1872_187214


namespace meeting_time_l1872_187297

def time_Cassie_leaves : ℕ := 495 -- 8:15 AM in minutes past midnight
def speed_Cassie : ℕ := 12 -- mph
def break_Cassie : ℚ := 0.25 -- hours
def time_Brian_leaves : ℕ := 540 -- 9:00 AM in minutes past midnight
def speed_Brian : ℕ := 14 -- mph
def total_distance : ℕ := 74 -- miles

def time_in_minutes (h m : ℕ) : ℕ := h * 60 + m

theorem meeting_time : time_Cassie_leaves + (87 : ℚ) / 26 * 60 = time_in_minutes 11 37 := 
by sorry

end meeting_time_l1872_187297


namespace interval_between_prizes_l1872_187256

theorem interval_between_prizes (total_prize : ℝ) (first_place : ℝ) (interval : ℝ) :
  total_prize = 4800 ∧
  first_place = 2000 ∧
  (first_place - interval) + (first_place - 2 * interval) = total_prize - 2000 →
  interval = 400 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  rw [h1, h2] at h3
  sorry

end interval_between_prizes_l1872_187256


namespace rational_numbers_inequality_l1872_187242

theorem rational_numbers_inequality (a b : ℚ) (h1 : a * b > 0) (h2 : a + b < 0) : a < 0 :=
sorry

end rational_numbers_inequality_l1872_187242


namespace smallest_m_l1872_187280

-- Defining the remainder function
def r (m n : ℕ) : ℕ := m % n

-- Main theorem stating the problem needed to be proved
theorem smallest_m (m : ℕ) (h : m > 0) 
  (H : (r m 1 + r m 2 + r m 3 + r m 4 + r m 5 + r m 6 + r m 7 + r m 8 + r m 9 + r m 10) = 4) : 
  m = 120 :=
sorry

end smallest_m_l1872_187280


namespace max_value_fraction_l1872_187266

theorem max_value_fraction (x : ℝ) : 
  (∃ x, (x^4 / (x^8 + 4 * x^6 - 8 * x^4 + 16 * x^2 + 64)) = (1 / 24)) := 
sorry

end max_value_fraction_l1872_187266


namespace find_additional_student_number_l1872_187231

def classSize : ℕ := 52
def sampleSize : ℕ := 4
def sampledNumbers : List ℕ := [5, 31, 44]
def additionalStudentNumber : ℕ := 18

theorem find_additional_student_number (classSize sampleSize : ℕ) 
    (sampledNumbers : List ℕ) : additionalStudentNumber ∈ (5 :: 31 :: 44 :: []) →
    (sampledNumbers = [5, 31, 44]) →
    (additionalStudentNumber = 18) := by
  sorry

end find_additional_student_number_l1872_187231


namespace polynomial_evaluation_l1872_187255

theorem polynomial_evaluation (a : ℝ) (h : a^2 + 3 * a = 2) : 2 * a^2 + 6 * a - 10 = -6 := by
  sorry

end polynomial_evaluation_l1872_187255


namespace minimum_inhabitants_to_ask_l1872_187211

def knights_count : ℕ := 50
def civilians_count : ℕ := 15

theorem minimum_inhabitants_to_ask (knights civilians : ℕ) (h_knights : knights = knights_count) (h_civilians : civilians = civilians_count) :
  ∃ n, (∀ asked : ℕ, (asked ≥ n) → asked - civilians > civilians) ∧ n = 31 :=
by
  sorry

end minimum_inhabitants_to_ask_l1872_187211


namespace donation_value_l1872_187257

def donation_in_yuan (usd: ℝ) (exchange_rate: ℝ): ℝ :=
  usd * exchange_rate

theorem donation_value :
  donation_in_yuan 1.2 6.25 = 7.5 :=
by
  -- Proof to be filled in
  sorry

end donation_value_l1872_187257


namespace prob_each_class_receives_one_prob_at_least_one_class_empty_prob_exactly_one_class_empty_l1872_187262

-- Definitions
def classes := 4
def students := 4
def total_distributions := classes ^ students

-- Problem 1
theorem prob_each_class_receives_one : 
  (A_4 ^ 4) / total_distributions = 3 / 32 := sorry

-- Problem 2
theorem prob_at_least_one_class_empty : 
  1 - (A_4 ^ 4) / total_distributions = 29 / 32 := sorry

-- Problem 3
theorem prob_exactly_one_class_empty :
  (C_4 ^ 1 * C_4 ^ 2 * C_3 ^ 1 * C_2 ^ 1) / total_distributions = 9 / 16 := sorry

end prob_each_class_receives_one_prob_at_least_one_class_empty_prob_exactly_one_class_empty_l1872_187262


namespace daragh_initial_bears_l1872_187206

variables (initial_bears eden_initial_bears eden_final_bears favorite_bears shared_bears_per_sister : ℕ)
variables (sisters : ℕ)

-- Given conditions
axiom h1 : eden_initial_bears = 10
axiom h2 : eden_final_bears = 14
axiom h3 : favorite_bears = 8
axiom h4 : sisters = 3

-- Derived condition
axiom h5 : shared_bears_per_sister = eden_final_bears - eden_initial_bears
axiom h6 : initial_bears = favorite_bears + (shared_bears_per_sister * sisters)

-- The theorem to prove
theorem daragh_initial_bears : initial_bears = 20 :=
by
  -- Insert proof here
  sorry

end daragh_initial_bears_l1872_187206


namespace max_candies_per_student_l1872_187222

theorem max_candies_per_student (n_students : ℕ) (mean_candies : ℕ) (min_candies : ℕ) (max_candies : ℕ) :
  n_students = 50 ∧
  mean_candies = 7 ∧
  min_candies = 1 ∧
  max_candies = 20 →
  ∃ m : ℕ, m ≤ max_candies :=
by
  intro h
  use 20
  sorry

end max_candies_per_student_l1872_187222


namespace equation_two_roots_iff_l1872_187296

theorem equation_two_roots_iff (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a) ↔ a > -1 :=
by
  sorry

end equation_two_roots_iff_l1872_187296


namespace initial_weight_l1872_187252

theorem initial_weight (W : ℝ) (current_weight : ℝ) (future_weight : ℝ) (months : ℝ) (additional_months : ℝ) 
  (constant_rate : Prop) :
  current_weight = 198 →
  future_weight = 170 →
  months = 3 →
  additional_months = 3.5 →
  constant_rate →
  W = 222 :=
by
  intros h_current_weight h_future_weight h_months h_additional_months h_constant_rate
  -- proof would go here
  sorry

end initial_weight_l1872_187252


namespace speed_in_still_water_l1872_187267

-- Given conditions
def upstream_speed : ℝ := 25
def downstream_speed : ℝ := 41

-- Question: Prove the speed of the man in still water is 33 kmph.
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 33 := 
by 
  sorry

end speed_in_still_water_l1872_187267


namespace angle_BAC_eq_angle_DAE_l1872_187208

-- Define types and points A, B, C, D, E
variables (A B C D E : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
variables (P Q R S T : Point)

-- Define angles
variable {α β γ δ θ ω : Angle}

-- Establish the conditions
axiom angle_ABC_eq_angle_ADE : α = θ
axiom angle_AEC_eq_angle_ADB : β = ω

-- State the theorem
theorem angle_BAC_eq_angle_DAE
  (h1 : α = θ) -- Given \(\angle ABC = \angle ADE\)
  (h2 : β = ω) -- Given \(\angle AEC = \angle ADB\)
  : γ = δ := sorry

end angle_BAC_eq_angle_DAE_l1872_187208


namespace log_27_gt_point_53_l1872_187219

open Real

theorem log_27_gt_point_53 :
  log 27 > 0.53 :=
by
  sorry

end log_27_gt_point_53_l1872_187219


namespace lee_sold_action_figures_l1872_187253

-- Defining variables and conditions based on the problem
def sneaker_cost : ℕ := 90
def saved_money : ℕ := 15
def price_per_action_figure : ℕ := 10
def remaining_money : ℕ := 25

-- Theorem statement asserting that Lee sold 10 action figures
theorem lee_sold_action_figures : 
  (sneaker_cost - saved_money + remaining_money) / price_per_action_figure = 10  :=
by
  sorry

end lee_sold_action_figures_l1872_187253


namespace triangle_inequality_l1872_187200

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (triangle_cond : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2*(b + c - a) + b^2*(c + a - b) + c^2*(a + b - c) ≤ 3*a*b*c :=
by
  sorry

end triangle_inequality_l1872_187200


namespace white_longer_than_blue_l1872_187241

noncomputable def whiteLineInches : ℝ := 7.666666666666667
noncomputable def blueLineInches : ℝ := 3.3333333333333335
noncomputable def inchToCm : ℝ := 2.54
noncomputable def cmToMm : ℝ := 10

theorem white_longer_than_blue :
  let whiteLineCm := whiteLineInches * inchToCm
  let blueLineCm := blueLineInches * inchToCm
  let differenceCm := whiteLineCm - blueLineCm
  let differenceMm := differenceCm * cmToMm
  differenceMm = 110.05555555555553 := by
  sorry

end white_longer_than_blue_l1872_187241


namespace range_of_m_l1872_187295

def cond1 (x : ℝ) : Prop := x^2 - 4 * x + 3 < 0
def cond2 (x : ℝ) : Prop := x^2 - 6 * x + 8 < 0
def cond3 (x m : ℝ) : Prop := 2 * x^2 - 9 * x + m < 0

theorem range_of_m (m : ℝ) : (∀ x, cond1 x → cond2 x → cond3 x m) → m < 9 :=
by
  sorry

end range_of_m_l1872_187295


namespace anayet_speed_is_61_l1872_187264

-- Define the problem conditions
def amoli_speed : ℝ := 42
def amoli_time : ℝ := 3
def anayet_time : ℝ := 2
def total_distance : ℝ := 369
def remaining_distance : ℝ := 121

-- Calculate derived values
def amoli_distance : ℝ := amoli_speed * amoli_time
def covered_distance : ℝ := total_distance - remaining_distance
def anayet_distance : ℝ := covered_distance - amoli_distance

-- Define the theorem to prove Anayet's speed
theorem anayet_speed_is_61 : anayet_distance / anayet_time = 61 :=
by
  -- sorry is a placeholder for the proof
  sorry

end anayet_speed_is_61_l1872_187264


namespace common_root_rational_l1872_187259

variable (a b c d e f g : ℚ) -- coefficient variables

def poly1 (x : ℚ) : ℚ := 90 * x^4 + a * x^3 + b * x^2 + c * x + 18

def poly2 (x : ℚ) : ℚ := 18 * x^5 + d * x^4 + e * x^3 + f * x^2 + g * x + 90

theorem common_root_rational (k : ℚ) (h1 : poly1 a b c k = 0) (h2 : poly2 d e f g k = 0) 
  (hn : k < 0) (hi : ∀ (m n : ℤ), k ≠ m / n) : k = -1/3 := sorry

end common_root_rational_l1872_187259


namespace reciprocal_of_complex_power_l1872_187258

noncomputable def complex_num_reciprocal : ℂ :=
  (Complex.I) ^ 2023

theorem reciprocal_of_complex_power :
  ∀ z : ℂ, z = (Complex.I) ^ 2023 -> (1 / z) = Complex.I :=
by
  intro z
  intro hz
  have h_power : z = Complex.I ^ 2023 := by assumption
  sorry

end reciprocal_of_complex_power_l1872_187258


namespace probability_same_color_l1872_187234

/-
Problem statement:
Given a bag contains 6 green balls and 7 white balls,
if two balls are drawn simultaneously, prove that the probability 
that both balls are the same color is 6/13.
-/

theorem probability_same_color
  (total_balls : ℕ := 6 + 7)
  (green_balls : ℕ := 6)
  (white_balls : ℕ := 7)
  (two_balls_drawn_simultaneously : Prop := true) :
  ((green_balls / total_balls) * ((green_balls - 1) / (total_balls - 1))) +
  ((white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1))) = 6 / 13 :=
sorry

end probability_same_color_l1872_187234


namespace C_plus_D_l1872_187229

theorem C_plus_D (C D : ℝ) (h : ∀ x : ℝ, x ≠ 3 → C / (x - 3) + D * (x + 2) = (-4 * x^2 + 18 * x + 32) / (x - 3)) : 
  C + D = 28 := sorry

end C_plus_D_l1872_187229


namespace Cl_invalid_electrons_l1872_187240

noncomputable def Cl_mass_number : ℕ := 35
noncomputable def Cl_protons : ℕ := 17
noncomputable def Cl_neutrons : ℕ := Cl_mass_number - Cl_protons
noncomputable def Cl_electrons : ℕ := Cl_protons

theorem Cl_invalid_electrons : Cl_electrons ≠ 18 :=
by
  sorry

end Cl_invalid_electrons_l1872_187240


namespace rectangle_width_is_4_l1872_187292

-- Definitions of conditions
variable (w : ℝ) -- width of the rectangle
def length := w + 2 -- length of the rectangle
def perimeter := 2 * w + 2 * (w + 2) -- perimeter of the rectangle, using given conditions

-- The theorem to be proved
theorem rectangle_width_is_4 (h : perimeter = 20) : w = 4 :=
by {
  sorry -- To be proved
}

end rectangle_width_is_4_l1872_187292


namespace area_enclosed_by_curve_l1872_187220

theorem area_enclosed_by_curve :
  ∃ (area : ℝ), (∀ (x y : ℝ), |x - 1| + |y - 1| = 1 → area = 2) :=
sorry

end area_enclosed_by_curve_l1872_187220


namespace symm_diff_complement_l1872_187224

variable {U : Type} -- Universal set U
variable (A B : Set U) -- Sets A and B

-- Definition of symmetric difference
def symm_diff (X Y : Set U) : Set U := (X ∪ Y) \ (X ∩ Y)

theorem symm_diff_complement (A B : Set U) :
  (symm_diff A B) = (symm_diff (Aᶜ) (Bᶜ)) :=
sorry

end symm_diff_complement_l1872_187224


namespace option_c_same_function_l1872_187273

-- Definitions based on conditions
def f_c (x : ℝ) : ℝ := x^2
def g_c (x : ℝ) : ℝ := 3 * x^6

-- Theorem statement that Option C f(x) and g(x) represent the same function
theorem option_c_same_function : ∀ x : ℝ, f_c x = g_c x := by
  sorry

end option_c_same_function_l1872_187273


namespace width_of_river_l1872_187274

def ferry_problem (v1 v2 W t1 t2 : ℝ) : Prop :=
  v1 * t1 + v2 * t1 = W ∧
  v1 * t1 = 720 ∧
  v2 * t1 = W - 720 ∧
  (v1 * t2 + v2 * t2 = 3 * W) ∧
  v1 * t2 = 2 * W - 400 ∧
  v2 * t2 = W + 400

theorem width_of_river 
  (v1 v2 W t1 t2 : ℝ)
  (h : ferry_problem v1 v2 W t1 t2) :
  W = 1280 :=
by
  sorry

end width_of_river_l1872_187274


namespace symmetry_of_transformed_graphs_l1872_187218

noncomputable def y_eq_f_x_symmetric_line (f : ℝ → ℝ) : Prop :=
∀ (x : ℝ), f (x - 19) = f (99 - x) ↔ x = 59

theorem symmetry_of_transformed_graphs (f : ℝ → ℝ) :
  y_eq_f_x_symmetric_line f :=
by {
  sorry
}

end symmetry_of_transformed_graphs_l1872_187218


namespace largest_term_at_k_31_l1872_187277

noncomputable def B_k (k : ℕ) : ℝ := (Nat.choose 500 k) * (0.15)^k

theorem largest_term_at_k_31 : 
  ∀ k : ℕ, (k ≤ 500) →
    (B_k 31 ≥ B_k k) :=
by
  intro k hk
  sorry

end largest_term_at_k_31_l1872_187277


namespace value_of_y_l1872_187243

theorem value_of_y (y : ℝ) (α : ℝ) (h₁ : (-3, y) = (x, y)) (h₂ : Real.sin α = -3 / 4) : 
  y = -9 * Real.sqrt 7 / 7 := 
  sorry

end value_of_y_l1872_187243


namespace decimal_6_to_binary_is_110_l1872_187261

def decimal_to_binary (n : ℕ) : ℕ :=
  -- This is just a placeholder definition. Adjust as needed for formalization.
  sorry

theorem decimal_6_to_binary_is_110 :
  decimal_to_binary 6 = 110 := 
sorry

end decimal_6_to_binary_is_110_l1872_187261


namespace symmetric_graph_l1872_187291

variable (f : ℝ → ℝ)
variable (c : ℝ)
variable (h_nonzero : c ≠ 0)
variable (h_fx_plus_y : ∀ (x y : ℝ), f (x + y) + f (x - y) = 2 * f x * f y)
variable (h_f_half_c : f (c / 2) = 0)
variable (h_f_zero : f 0 ≠ 0)

theorem symmetric_graph (k : ℤ) : 
  ∀ (x : ℝ), f (x) = f (2*k*c - x) :=
sorry

end symmetric_graph_l1872_187291


namespace cat_mouse_position_after_300_moves_l1872_187250

def move_pattern_cat_mouse :=
  let cat_cycle_length := 4
  let mouse_cycle_length := 8
  let cat_moves := 300
  let mouse_moves := (3 / 2) * cat_moves
  let cat_position := (cat_moves % cat_cycle_length)
  let mouse_position := (mouse_moves % mouse_cycle_length)
  (cat_position, mouse_position)

theorem cat_mouse_position_after_300_moves :
  move_pattern_cat_mouse = (0, 2) :=
by
  sorry

end cat_mouse_position_after_300_moves_l1872_187250


namespace train_length_is_250_l1872_187294

noncomputable def train_length (speed_kmh : ℕ) (time_sec : ℕ) (station_length : ℕ) : ℕ :=
  (speed_kmh * 1000 / 3600 * time_sec) - station_length

theorem train_length_is_250 :
  train_length 36 45 200 = 250 :=
by
  sorry

end train_length_is_250_l1872_187294


namespace range_of_a_l1872_187238

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + 2 * x - a = 0) ↔ a ≥ -1 :=
by
  sorry

end range_of_a_l1872_187238


namespace problem1_solution_problem2_solution_l1872_187244

-- Statement for Problem 1
theorem problem1_solution (x : ℝ) : (1 / 2 * (x - 3) ^ 2 = 18) ↔ (x = 9 ∨ x = -3) :=
by sorry

-- Statement for Problem 2
theorem problem2_solution (x : ℝ) : (x ^ 2 + 6 * x = 5) ↔ (x = -3 + Real.sqrt 14 ∨ x = -3 - Real.sqrt 14) :=
by sorry

end problem1_solution_problem2_solution_l1872_187244


namespace remainder_sum_division_by_9_l1872_187272

theorem remainder_sum_division_by_9 :
  (9151 + 9152 + 9153 + 9154 + 9155 + 9156 + 9157) % 9 = 6 :=
by
  sorry

end remainder_sum_division_by_9_l1872_187272


namespace find_third_number_l1872_187271

noncomputable def third_number := 9.110300000000005

theorem find_third_number :
  12.1212 + 17.0005 - third_number = 20.011399999999995 :=
sorry

end find_third_number_l1872_187271


namespace number_of_representations_l1872_187269

-- Definitions of the conditions
def is_valid_b (b : ℕ) : Prop :=
  b ≤ 99

def is_representation (b3 b2 b1 b0 : ℕ) : Prop :=
  3152 = b3 * 10^3 + b2 * 10^2 + b1 * 10 + b0

-- The theorem to prove
theorem number_of_representations : 
  ∃ (N' : ℕ), (N' = 316) ∧ 
  (∀ (b3 b2 b1 b0 : ℕ), is_representation b3 b2 b1 b0 → is_valid_b b0 → is_valid_b b1 → is_valid_b b2 → is_valid_b b3) :=
sorry

end number_of_representations_l1872_187269


namespace arithmetic_sequence_a5_l1872_187268

noncomputable def a (n : ℕ) (a₁ d : ℝ) : ℝ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_a5 (a₁ d : ℝ) (h1 : a 2 a₁ d = 2 * a 3 a₁ d + 1) (h2 : a 4 a₁ d = 2 * a 3 a₁ d + 7) :
  a 5 a₁ d = 2 :=
by
  sorry

end arithmetic_sequence_a5_l1872_187268


namespace average_remaining_ropes_l1872_187226

theorem average_remaining_ropes 
  (n : ℕ) 
  (m : ℕ) 
  (l_avg : ℕ) 
  (l1_avg : ℕ) 
  (l2_avg : ℕ) 
  (h1 : n = 6)
  (h2 : m = 2)
  (hl_avg : l_avg = 80)
  (hl1_avg : l1_avg = 70)
  (htotal : l_avg * n = 480)
  (htotal1 : l1_avg * m = 140)
  (htotal2 : l_avg * n - l1_avg * m = 340):
  (340 : ℕ) / (4 : ℕ) = 85 := by
  sorry

end average_remaining_ropes_l1872_187226


namespace tan_75_eq_2_plus_sqrt_3_l1872_187216

theorem tan_75_eq_2_plus_sqrt_3 : Real.tan (75 * Real.pi / 180) = 2 + Real.sqrt 3 := 
sorry

end tan_75_eq_2_plus_sqrt_3_l1872_187216


namespace minimum_value_of_sum_of_squares_l1872_187245

theorem minimum_value_of_sum_of_squares (x y z : ℝ) (h : 4 * x + 3 * y + 12 * z = 1) : 
  x^2 + y^2 + z^2 ≥ 1 / 169 :=
by
  sorry

end minimum_value_of_sum_of_squares_l1872_187245


namespace fraction_of_married_men_l1872_187282

-- We start by defining the conditions given in the problem.
def only_single_women_and_married_couples (total_women total_married_women : ℕ) :=
  total_women - total_married_women + total_married_women * 2

def probability_single_woman_single (total_women total_single_women : ℕ) :=
  total_single_women / total_women = 3 / 7

-- The main theorem we need to prove under the given conditions.
theorem fraction_of_married_men (total_women total_married_women : ℕ)
  (h1 : probability_single_woman_single total_women (total_women - total_married_women))
  : (total_married_women * 2) / (total_women + total_married_women) = 4 / 11 := sorry

end fraction_of_married_men_l1872_187282


namespace application_outcomes_l1872_187276

theorem application_outcomes :
  let choices_A := 3
  let choices_B := 2
  let choices_C := 3
  (choices_A * choices_B * choices_C) = 18 :=
by
  let choices_A := 3
  let choices_B := 2
  let choices_C := 3
  show (choices_A * choices_B * choices_C = 18)
  sorry

end application_outcomes_l1872_187276


namespace find_g_neg_3_l1872_187275

def g (x : ℤ) : ℤ :=
if x < 1 then 3 * x - 4 else x + 6

theorem find_g_neg_3 : g (-3) = -13 :=
by
  -- proof omitted: sorry
  sorry

end find_g_neg_3_l1872_187275


namespace zeroes_in_base_81_l1872_187251

-- Definitions based on the conditions:
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Question: How many zeroes does 15! end with in base 81?
-- Lean 4 proof statement:
theorem zeroes_in_base_81 (n : ℕ) : n = 15 → Nat.factorial n = 
  (81 : ℕ) ^ k * m → k = 1 :=
by
  sorry

end zeroes_in_base_81_l1872_187251


namespace original_number_proof_l1872_187298

-- Define the conditions
variables (x y : ℕ)
-- Given conditions
def condition1 : Prop := y = 13
def condition2 : Prop := 7 * x + 5 * y = 146

-- Goal: the original number (sum of the parts x and y)
def original_number : ℕ := x + y

-- State the problem as a theorem
theorem original_number_proof (x y : ℕ) (h1 : condition1 y) (h2 : condition2 x y) : original_number x y = 24 := by
  -- The proof will be written here
  sorry

end original_number_proof_l1872_187298


namespace a_plus_b_is_24_l1872_187228

theorem a_plus_b_is_24 (a b : ℤ) (h1 : 0 < b) (h2 : b < a) (h3 : a * (a + 3 * b) = 550) : a + b = 24 :=
sorry

end a_plus_b_is_24_l1872_187228


namespace quadrilateral_with_equal_angles_is_parallelogram_l1872_187221

axiom Quadrilateral (a b c d : Type) : Prop
axiom Parallelogram (a b c d : Type) : Prop
axiom equal_angles (a b c d : Type) : Prop

theorem quadrilateral_with_equal_angles_is_parallelogram 
  (a b c d : Type) 
  (q : Quadrilateral a b c d)
  (h : equal_angles a b c d) : Parallelogram a b c d := 
sorry

end quadrilateral_with_equal_angles_is_parallelogram_l1872_187221


namespace grocer_display_rows_l1872_187248

theorem grocer_display_rows (n : ℕ)
  (h1 : ∃ k, k = 2 + 3 * (n - 1))
  (h2 : ∃ s, s = (n / 2) * (2 + (3 * n - 1))):
  (3 * n^2 + n) / 2 = 225 → n = 12 :=
by
  sorry

end grocer_display_rows_l1872_187248


namespace proof_two_digit_number_l1872_187247

noncomputable def two_digit_number := {n : ℤ // 10 ≤ n ∧ n ≤ 99}

theorem proof_two_digit_number (n : two_digit_number) :
  (n.val % 2 = 0) ∧ 
  ((n.val + 1) % 3 = 0) ∧
  ((n.val + 2) % 4 = 0) ∧
  ((n.val + 3) % 5 = 0) →
  n.val = 62 :=
by sorry

end proof_two_digit_number_l1872_187247


namespace sum_of_cubes_l1872_187212

theorem sum_of_cubes (x y : ℝ) (h_sum : x + y = 3) (h_prod : x * y = 2) : x^3 + y^3 = 9 :=
by
  sorry

end sum_of_cubes_l1872_187212


namespace necessary_not_sufficient_condition_t_for_b_l1872_187299

variable (x y : ℝ)

def condition_t : Prop := x ≤ 12 ∨ y ≤ 16
def condition_b : Prop := x + y ≤ 28 ∨ x * y ≤ 192

theorem necessary_not_sufficient_condition_t_for_b (h : condition_b x y) : condition_t x y ∧ ¬ (condition_t x y → condition_b x y) := by
  sorry

end necessary_not_sufficient_condition_t_for_b_l1872_187299


namespace veronica_pre_selected_photos_l1872_187210

-- Definition: Veronica needs to include 3 or 4 of her pictures
def needs_3_or_4_photos : Prop := True

-- Definition: Veronica has pre-selected a certain number of photos
def pre_selected_photos : ℕ := 15

-- Definition: She has 15 choices
def choices : ℕ := 15

-- The proof statement
theorem veronica_pre_selected_photos : needs_3_or_4_photos → choices = pre_selected_photos :=
by
  intros
  sorry

end veronica_pre_selected_photos_l1872_187210
