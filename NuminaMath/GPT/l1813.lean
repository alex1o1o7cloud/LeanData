import Mathlib

namespace area_ratio_l1813_181359

-- Define the problem conditions
def Square (s : ℝ) := s > 0
def Rectangle (longer shorter : ℝ) := longer = 1.2 * shorter ∧ shorter = 0.8 * shorter

-- Define a function to calculate the area of square
def area_square (s : ℝ) : ℝ := s * s

-- Define a function to calculate the area of rectangle
def area_rectangle (longer shorter : ℝ) : ℝ := longer * shorter

-- State the proof problem
theorem area_ratio (s : ℝ) (h_square : Square s) :
  let longer := 1.2 * s
  let shorter := 0.8 * s
  area_rectangle longer shorter / area_square s = 24 / 25 :=
by
  sorry

end area_ratio_l1813_181359


namespace landA_area_and_ratio_l1813_181396

/-
  a = 3, b = 5, c = 6
  p = 1/2 * (a + b + c)
  S = sqrt(p * (p - a) * (p - b) * (p - c))
  S_A = 2 * sqrt(14)
  S_B = 3/2 * sqrt(14)
  S_A / S_B = 4 / 3
-/
theorem landA_area_and_ratio :
  let a := 3
  let b := 5
  let c := 6
  let p := (a + b + c) / 2
  let S_A := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  let S_B := 3 / 2 * Real.sqrt 14
  S_A = 2 * Real.sqrt 14 ∧ S_A / S_B = 4 / 3 :=
by
  sorry

end landA_area_and_ratio_l1813_181396


namespace matilda_jellybeans_l1813_181304

/-- Suppose Matilda has half as many jellybeans as Matt.
    Suppose Matt has ten times as many jellybeans as Steve.
    Suppose Steve has 84 jellybeans.
    Then Matilda has 420 jellybeans. -/
theorem matilda_jellybeans
    (matilda_jellybeans : ℕ)
    (matt_jellybeans : ℕ)
    (steve_jellybeans : ℕ)
    (h1 : matilda_jellybeans = matt_jellybeans / 2)
    (h2 : matt_jellybeans = 10 * steve_jellybeans)
    (h3 : steve_jellybeans = 84) : matilda_jellybeans = 420 := 
sorry

end matilda_jellybeans_l1813_181304


namespace polygonal_chain_max_length_not_exceed_200_l1813_181312

-- Define the size of the board
def board_size : ℕ := 15

-- Define the concept of a polygonal chain length on a symmetric board
def polygonal_chain_length (n : ℕ) : ℕ := sorry -- length function yet to be defined

-- Define the maximum length constant to be compared with
def max_length : ℕ := 200

-- Define the theorem statement including all conditions and constraints
theorem polygonal_chain_max_length_not_exceed_200 :
  ∃ (n : ℕ), n = board_size ∧ 
             (∀ (length : ℕ),
             length = polygonal_chain_length n →
             length ≤ max_length) :=
sorry

end polygonal_chain_max_length_not_exceed_200_l1813_181312


namespace hyperbola_equation_l1813_181373

-- Lean 4 statement
theorem hyperbola_equation (a b : ℝ) (hpos_a : a > 0) (hpos_b : b > 0)
    (length_imag_axis : 2 * b = 2)
    (asymptote : ∃ (k : ℝ), ∀ x : ℝ, y = k * x ↔ y = (1 / 2) * x) :
  (x y : ℝ) → (x^2 / a^2) - (y^2 / b^2) = 1 ↔ (x^2 / 4) - (y^2 / 1) = 1 :=
by 
  intros
  sorry

end hyperbola_equation_l1813_181373


namespace no_real_solution_l1813_181377

theorem no_real_solution :
  ¬ ∃ x : ℝ, (3 * x ^ 2 / (x - 2) - (5 * x + 4) / 4 + (10 - 9 * x) / (x - 2) + 2 = 0) :=
sorry

end no_real_solution_l1813_181377


namespace find_point_M_l1813_181334

/-- Define the function f(x) = x^3 + x - 2. -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- Define the derivative of the function, f'(x). -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

/-- Define the condition that the slope of the tangent line is perpendicular to y = -1/4x - 1. -/
def slope_perpendicular_condition (m : ℝ) : Prop := m = 4

/-- Main theorem: The coordinates of the point M are (1, 0) and (-1, -4). -/
theorem find_point_M : 
  ∃ (x₀ y₀ : ℝ), f x₀ = y₀ ∧ slope_perpendicular_condition (f' x₀) ∧ 
  ((x₀ = 1 ∧ y₀ = 0) ∨ (x₀ = -1 ∧ y₀ = -4)) := 
sorry

end find_point_M_l1813_181334


namespace tank_filled_fraction_l1813_181305

noncomputable def initial_quantity (total_capacity : ℕ) := (3 / 4 : ℚ) * total_capacity

noncomputable def final_quantity (initial : ℚ) (additional : ℚ) := initial + additional

noncomputable def fraction_of_capacity (quantity : ℚ) (total_capacity : ℕ) := quantity / total_capacity

theorem tank_filled_fraction (total_capacity : ℕ) (additional_gas : ℚ)
  (initial_fraction : ℚ) (final_fraction : ℚ) :
  initial_fraction = initial_quantity total_capacity →
  final_fraction = fraction_of_capacity (final_quantity initial_fraction additional_gas) total_capacity →
  total_capacity = 42 →
  additional_gas = 7 →
  initial_fraction = 31.5 →
  final_fraction = (833 / 909 : ℚ) :=
by
  sorry

end tank_filled_fraction_l1813_181305


namespace three_topping_pizzas_l1813_181300

theorem three_topping_pizzas : Nat.choose 8 3 = 56 := by
  sorry

end three_topping_pizzas_l1813_181300


namespace labourer_saving_after_debt_clearance_l1813_181395

variable (averageExpenditureFirst6Months : ℕ)
variable (monthlyIncome : ℕ)
variable (reducedMonthlyExpensesNext4Months : ℕ)

theorem labourer_saving_after_debt_clearance (h1 : averageExpenditureFirst6Months = 90)
                                              (h2 : monthlyIncome = 81)
                                              (h3 : reducedMonthlyExpensesNext4Months = 60) :
    (monthlyIncome * 4) - ((reducedMonthlyExpensesNext4Months * 4) + 
    ((averageExpenditureFirst6Months * 6) - (monthlyIncome * 6))) = 30 := by
  sorry

end labourer_saving_after_debt_clearance_l1813_181395


namespace find_larger_number_l1813_181302

-- Define the conditions
variables (L S : ℕ)
axiom condition1 : L - S = 1365
axiom condition2 : L = 6 * S + 35

-- State the theorem
theorem find_larger_number : L = 1631 :=
by
  sorry

end find_larger_number_l1813_181302


namespace find_y_of_x_pow_l1813_181394

theorem find_y_of_x_pow (x y : ℝ) (h1 : x = 2) (h2 : x^(3*y - 1) = 8) : y = 4 / 3 :=
by
  -- skipping proof
  sorry

end find_y_of_x_pow_l1813_181394


namespace distance_from_origin_to_point_on_parabola_l1813_181336

theorem distance_from_origin_to_point_on_parabola
  (y x : ℝ)
  (focus : ℝ × ℝ := (4, 0))
  (on_parabola : y^2 = 8 * x)
  (distance_to_focus : Real.sqrt ((x - 4)^2 + y^2) = 4) :
  Real.sqrt (x^2 + y^2) = 2 * Real.sqrt 5 :=
by
  sorry

end distance_from_origin_to_point_on_parabola_l1813_181336


namespace simplify_expression_l1813_181317

variable (y : ℝ)

theorem simplify_expression : (3 * y^4)^2 = 9 * y^8 :=
by 
  sorry

end simplify_expression_l1813_181317


namespace tan_sum_identity_l1813_181303

theorem tan_sum_identity
  (A B C : ℝ)
  (h1 : A + B + C = Real.pi)
  (h2 : Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C) :
  Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C := 
sorry

end tan_sum_identity_l1813_181303


namespace evaluate_fraction_l1813_181343

theorem evaluate_fraction :
  (20 - 18 + 16 - 14 + 12 - 10 + 8 - 6 + 4 - 2) / (2 - 4 + 6 - 8 + 10 - 12 + 14 - 16 + 18) = 1 :=
by
  sorry

end evaluate_fraction_l1813_181343


namespace number_of_people_l1813_181327

variable (P M : ℕ)

-- Conditions
def cond1 : Prop := (500 = P * M)
def cond2 : Prop := (500 = (P + 5) * (M - 2))

-- Goal
theorem number_of_people (h1 : cond1 P M) (h2 : cond2 P M) : P = 33 :=
sorry

end number_of_people_l1813_181327


namespace fiveLetterWordsWithAtLeastOneVowel_l1813_181374

-- Definitions for the given conditions
def letters : List Char := ['A', 'B', 'C', 'D', 'E', 'F']
def vowels : List Char := ['A', 'E']
def consonants : List Char := ['B', 'C', 'D', 'F']

-- Total number of 5-letter words with no restrictions
def totalWords := 6^5

-- Total number of 5-letter words containing no vowels
def noVowelWords := 4^5

-- Prove that the number of 5-letter words with at least one vowel is 6752
theorem fiveLetterWordsWithAtLeastOneVowel : (totalWords - noVowelWords) = 6752 := by
  sorry

end fiveLetterWordsWithAtLeastOneVowel_l1813_181374


namespace problem_l1813_181361

theorem problem (x : ℕ) (h : 2^x + 2^x + 2^x = 256) : x * (x + 1) = 72 :=
sorry

end problem_l1813_181361


namespace win_percentage_of_people_with_envelopes_l1813_181313

theorem win_percentage_of_people_with_envelopes (total_people : ℕ) (percent_with_envelopes : ℝ) (winners : ℕ) (num_with_envelopes : ℕ) : 
  total_people = 100 ∧ percent_with_envelopes = 0.40 ∧ num_with_envelopes = total_people * percent_with_envelopes ∧ winners = 8 → 
    (winners / num_with_envelopes) * 100 = 20 :=
by
  intros
  sorry

end win_percentage_of_people_with_envelopes_l1813_181313


namespace num_users_in_china_in_2022_l1813_181329

def num_users_scientific (n : ℝ) : Prop :=
  n = 1.067 * 10^9

theorem num_users_in_china_in_2022 :
  num_users_scientific 1.067e9 :=
by
  sorry

end num_users_in_china_in_2022_l1813_181329


namespace find_x_l1813_181339

noncomputable def positive_real (a : ℝ) := 0 < a

theorem find_x (x y : ℝ) (h1 : positive_real x) (h2 : positive_real y)
  (h3 : 6 * x^3 + 12 * x^2 * y = 2 * x^4 + 3 * x^3 * y)
  (h4 : x + y = 3) : x = 2 :=
by
  sorry

end find_x_l1813_181339


namespace problem1_problem2_l1813_181380

-- Definitions for conditions
def p (x : ℝ) : Prop := x^2 - 7 * x + 10 < 0
def q (x : ℝ) (m : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 < 0

-- Problem 1: For m = 4, p ∧ q implies 4 < x < 5
theorem problem1 (x : ℝ) (h : 4 < x ∧ x < 5) : 
  p x ∧ q x 4 :=
sorry

-- Problem 2: ∃ m, m > 0, m ≤ 2, and 3m ≥ 5 implies (5/3 ≤ m ≤ 2)
theorem problem2 (m : ℝ) (h1 : m > 0) (h2 : m ≤ 2) (h3 : 3 * m ≥ 5) : 
  5 / 3 ≤ m ∧ m ≤ 2 :=
sorry

end problem1_problem2_l1813_181380


namespace incorrect_relation_when_agtb_l1813_181328

theorem incorrect_relation_when_agtb (a b : ℝ) (c : ℝ) (h : a > b) : c = 0 → ¬ (a * c^2 > b * c^2) :=
by
  -- Not providing the proof here as specified in the instructions.
  sorry

end incorrect_relation_when_agtb_l1813_181328


namespace price_reduction_percentage_l1813_181352

theorem price_reduction_percentage (original_price new_price : ℕ) 
  (h_original : original_price = 250) 
  (h_new : new_price = 200) : 
  (original_price - new_price) * 100 / original_price = 20 := 
by 
  -- include the proof when needed
  sorry

end price_reduction_percentage_l1813_181352


namespace workers_time_together_l1813_181390

theorem workers_time_together (T : ℝ) (h1 : ∀ t : ℝ, (T + 8) = t → 1 / t = 1 / (T + 8))
                                (h2 : ∀ t : ℝ, (T + 4.5) = t → 1 / t = 1 / (T + 4.5))
                                (h3 : 1 / (T + 8) + 1 / (T + 4.5) = 1 / T) : T = 6 :=
sorry

end workers_time_together_l1813_181390


namespace sin_product_l1813_181326

theorem sin_product :
  (Real.sin (12 * Real.pi / 180)) * 
  (Real.sin (36 * Real.pi / 180)) *
  (Real.sin (72 * Real.pi / 180)) *
  (Real.sin (84 * Real.pi / 180)) = 1 / 16 := 
by
  sorry

end sin_product_l1813_181326


namespace least_positive_integer_n_l1813_181306

theorem least_positive_integer_n (n : ℕ) (hn : n = 10) :
  (2:ℝ)^(1 / 5 * (n * (n + 1) / 2)) > 1000 :=
by
  sorry

end least_positive_integer_n_l1813_181306


namespace count_noncongruent_triangles_l1813_181324

theorem count_noncongruent_triangles :
  ∃ (n : ℕ), n = 13 ∧
  ∀ (a b c : ℕ), a < b ∧ b < c ∧ a + b > c ∧ a + b + c < 20 ∧ ¬(a * a + b * b = c * c)
  → n = 13 := by {
  sorry
}

end count_noncongruent_triangles_l1813_181324


namespace simplify_fraction_l1813_181379

theorem simplify_fraction (b : ℕ) (hb : b = 2) : (15 * b ^ 4) / (45 * b ^ 3) = 2 / 3 :=
by
  sorry

end simplify_fraction_l1813_181379


namespace find_y_value_l1813_181321
-- Import the necessary Lean library

-- Define the conditions and the target theorem
theorem find_y_value (h : 6 * y + 3 * y + y + 4 * y = 360) : y = 180 / 7 :=
by
  sorry

end find_y_value_l1813_181321


namespace find_f_sqrt2_l1813_181388

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain : ∀ x, x > 0 → (∃ y, f y = x ∨ y = x)

axiom f_multiplicative : ∀ x y : ℝ, x > 0 → y > 0 → f (x * y) = f x + f y
axiom f_at_8 : f 8 = 3

-- Define the problem statement
theorem find_f_sqrt2 : f (Real.sqrt 2) = 1 / 2 := sorry

end find_f_sqrt2_l1813_181388


namespace gcd_of_polynomials_l1813_181369

theorem gcd_of_polynomials (b : ℤ) (h : ∃ k : ℤ, b = 2 * 5959 * k) :
  Int.gcd (4 * b^2 + 73 * b + 156) (4 * b + 15) = 1 :=
by
  sorry

end gcd_of_polynomials_l1813_181369


namespace fraction_evaluation_l1813_181338

theorem fraction_evaluation : (1 - (1 / 4)) / (1 - (1 / 3)) = (9 / 8) :=
by
  sorry

end fraction_evaluation_l1813_181338


namespace ratio_of_speeds_l1813_181347

theorem ratio_of_speeds
  (speed_of_tractor : ℝ)
  (speed_of_bike : ℝ)
  (speed_of_car : ℝ)
  (h1 : speed_of_tractor = 575 / 25)
  (h2 : speed_of_car = 331.2 / 4)
  (h3 : speed_of_bike = 2 * speed_of_tractor) :
  speed_of_car / speed_of_bike = 1.8 :=
by
  sorry

end ratio_of_speeds_l1813_181347


namespace number_of_elements_in_set_l1813_181342

-- We define the conditions in terms of Lean definitions.
variable (n : ℕ) (S : ℕ)

-- Define the initial wrong average condition
def wrong_avg_condition : Prop := (S + 26) / n = 18

-- Define the corrected average condition
def correct_avg_condition : Prop := (S + 36) / n = 19

-- The main theorem to be proved
theorem number_of_elements_in_set (h1 : wrong_avg_condition n S) (h2 : correct_avg_condition n S) : n = 10 := 
sorry

end number_of_elements_in_set_l1813_181342


namespace bisect_angle_BAX_l1813_181348

-- Definitions and conditions
variables {A B C M X : Point}
variable (is_scalene_triangle : ScaleneTriangle A B C)
variable (is_midpoint : Midpoint M B C)
variable (is_parallel : Parallel (Line C X) (Line A B))
variable (angle_right : Angle AM X = 90)

-- The theorem statement to be proven
theorem bisect_angle_BAX (h1 : is_scalene_triangle)
                         (h2 : is_midpoint)
                         (h3 : is_parallel)
                         (h4 : angle_right) :
  Bisects (Line A M) (Angle B A X) :=
sorry

end bisect_angle_BAX_l1813_181348


namespace keith_turnips_l1813_181309

theorem keith_turnips (Alyssa_turnips Keith_turnips : ℕ) 
  (total_turnips : Alyssa_turnips + Keith_turnips = 15) 
  (alyssa_grew : Alyssa_turnips = 9) : Keith_turnips = 6 :=
by
  sorry

end keith_turnips_l1813_181309


namespace brownie_to_bess_ratio_l1813_181337

-- Define daily milk production
def bess_daily_milk : ℕ := 2
def daisy_daily_milk : ℕ := bess_daily_milk + 1

-- Calculate weekly milk production
def bess_weekly_milk : ℕ := bess_daily_milk * 7
def daisy_weekly_milk : ℕ := daisy_daily_milk * 7

-- Given total weekly milk production
def total_weekly_milk : ℕ := 77
def combined_bess_daisy_weekly_milk : ℕ := bess_weekly_milk + daisy_weekly_milk
def brownie_weekly_milk : ℕ := total_weekly_milk - combined_bess_daisy_weekly_milk

-- Main proof statement
theorem brownie_to_bess_ratio : brownie_weekly_milk / bess_weekly_milk = 3 :=
by
  -- Skip the proof
  sorry

end brownie_to_bess_ratio_l1813_181337


namespace second_group_product_number_l1813_181350

theorem second_group_product_number (a₀ : ℕ) (h₀ : 0 ≤ a₀ ∧ a₀ < 20)
  (h₁ : 4 * 20 + a₀ = 94) : 1 * 20 + a₀ = 34 :=
by
  sorry

end second_group_product_number_l1813_181350


namespace train_speed_is_60_kmph_l1813_181362

-- Define the distance and time
def train_length : ℕ := 400
def bridge_length : ℕ := 800
def time_to_pass_bridge : ℕ := 72

-- Define the distances and calculations
def total_distance : ℕ := train_length + bridge_length
def speed_m_per_s : ℚ := total_distance / time_to_pass_bridge
def speed_km_per_h : ℚ := speed_m_per_s * 3.6

-- State and prove the theorem
theorem train_speed_is_60_kmph : speed_km_per_h = 60 := by
  sorry

end train_speed_is_60_kmph_l1813_181362


namespace range_of_a_l1813_181397

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * Real.exp 1 * x - Real.log x / x + a

theorem range_of_a (a : ℝ) :
  (∃ x > 0, f x a = 0) → a ≤ Real.exp 2 + 1 / Real.exp 1 := by
  sorry

end range_of_a_l1813_181397


namespace xy_expr_value_l1813_181360

variable (x y : ℝ)

-- Conditions
def cond1 : Prop := x - y = 2
def cond2 : Prop := x * y = 3

-- Statement to prove
theorem xy_expr_value (h1 : cond1 x y) (h2 : cond2 x y) : x * y^2 - x^2 * y = -6 :=
by
  sorry

end xy_expr_value_l1813_181360


namespace solution_set_of_inequality_l1813_181385

theorem solution_set_of_inequality :
  { x : ℝ | ∃ (h : x ≠ 1), 1 / (x - 1) ≥ -1 } = { x : ℝ | x ≤ 0 ∨ 1 < x } :=
by sorry

end solution_set_of_inequality_l1813_181385


namespace scientific_notation_of_100000000_l1813_181393

theorem scientific_notation_of_100000000 :
  100000000 = 1 * 10^8 :=
sorry

end scientific_notation_of_100000000_l1813_181393


namespace value_of_f_2012_l1813_181381

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

noncomputable def f : ℝ → ℝ := sorry

axiom odd_fn : odd_function f
axiom f_at_2 : f 2 = 0
axiom functional_eq : ∀ x : ℝ, f (x + 4) = f x + f 4

theorem value_of_f_2012 : f 2012 = 0 :=
by
  sorry

end value_of_f_2012_l1813_181381


namespace contest_correct_answers_l1813_181399

/-- 
In a mathematics contest with ten problems, a student gains 
5 points for a correct answer and loses 2 points for an 
incorrect answer. If Olivia answered every problem 
and her score was 29, how many correct answers did she have?
-/
theorem contest_correct_answers (c w : ℕ) (h1 : c + w = 10) (h2 : 5 * c - 2 * w = 29) : c = 7 :=
by 
  sorry

end contest_correct_answers_l1813_181399


namespace beaker_water_division_l1813_181375

-- Given conditions
variable (buckets : ℕ) (bucket_capacity : ℕ) (remaining_water : ℝ)
  (total_buckets : ℕ := 2) (capacity : ℕ := 120) (remaining : ℝ := 2.4)

-- Theorem statement
theorem beaker_water_division (h1 : buckets = total_buckets)
                             (h2 : bucket_capacity = capacity)
                             (h3 : remaining_water = remaining) :
                             (total_water : ℝ := buckets * bucket_capacity + remaining_water ) → 
                             (water_per_beaker : ℝ := total_water / 3) →
                             water_per_beaker = 80.8 :=
by
  -- Skipping the proof steps here, will use sorry
  sorry

end beaker_water_division_l1813_181375


namespace range_of_f_on_interval_l1813_181372

-- Definition of the function
def f (x : ℝ) : ℝ := x^2 - 2 * x + 2

-- Definition of the interval
def domain (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

-- The main statement
theorem range_of_f_on_interval : 
  ∀ y, (∃ x, domain x ∧ f x = y) ↔ (1 ≤ y ∧ y ≤ 10) :=
by
  sorry

end range_of_f_on_interval_l1813_181372


namespace bus_driver_regular_rate_l1813_181356

theorem bus_driver_regular_rate (hours := 60) (total_pay := 1200) (regular_hours := 40) (overtime_rate_factor := 1.75) :
  ∃ R : ℝ, 40 * R + 20 * (1.75 * R) = 1200 ∧ R = 16 := 
by
  sorry

end bus_driver_regular_rate_l1813_181356


namespace winning_margin_l1813_181384

theorem winning_margin (total_votes : ℝ) (winning_votes : ℝ) (winning_percent : ℝ) (losing_percent : ℝ) 
  (win_votes_eq: winning_votes = winning_percent * total_votes)
  (perc_eq: winning_percent + losing_percent = 1)
  (win_votes_given: winning_votes = 550)
  (winning_percent_given: winning_percent = 0.55)
  (losing_percent_given: losing_percent = 0.45) :
  winning_votes - (losing_percent * total_votes) = 100 := 
by
  sorry

end winning_margin_l1813_181384


namespace base_5_to_decimal_l1813_181353

theorem base_5_to_decimal : 
  let b5 := [1, 2, 3, 4] -- base-5 number 1234 in list form
  let decimal := 194
  (b5[0] * 5^3 + b5[1] * 5^2 + b5[2] * 5^1 + b5[3] * 5^0) = decimal :=
by
  -- Proof details go here
  sorry

end base_5_to_decimal_l1813_181353


namespace problem_part1_29_13_problem_part2_mn_problem_part3_k_36_problem_part4_min_val_l1813_181367

def is_perfect_number (n : ℕ) : Prop :=
  ∃ a b : ℤ, n = a^2 + b^2

theorem problem_part1_29_13 : is_perfect_number 29 ∧ is_perfect_number 13 := by
  sorry

theorem problem_part2_mn : 
  ∃ m n : ℤ, (∀ a : ℤ, a^2 - 4 * a + 8 = (a - m)^2 + n^2) ∧ (m * n = 4 ∨ m * n = -4) := by
  sorry

theorem problem_part3_k_36 (a b : ℤ) : 
  ∃ k : ℤ, (∀ k : ℤ, a^2 + 4*a*b + 5*b^2 - 12*b + k = (a + 2*b)^2 + (b-6)^2) ∧ k = 36 := by
  sorry

theorem problem_part4_min_val (a b : ℝ) : 
  (∀ (a b : ℝ), -a^2 + 5*a + b - 7 = 0 → ∃ a' b', (a + b = (a'-2)^2 + 3) ∧ a' + b' = 3) := by
  sorry

end problem_part1_29_13_problem_part2_mn_problem_part3_k_36_problem_part4_min_val_l1813_181367


namespace alex_ahead_of_max_after_even_l1813_181392

theorem alex_ahead_of_max_after_even (x : ℕ) (h1 : x - 200 + 170 + 440 = 1110) : x = 300 :=
sorry

end alex_ahead_of_max_after_even_l1813_181392


namespace sum_of_ages_l1813_181346

variables (P M Mo : ℕ)

theorem sum_of_ages (h1 : 5 * P = 3 * M)
                    (h2 : 5 * M = 3 * Mo)
                    (h3 : Mo - P = 32) :
  P + M + Mo = 98 :=
by
  sorry

end sum_of_ages_l1813_181346


namespace total_pieces_equiv_231_l1813_181311

-- Define the arithmetic progression for rods.
def rods_arithmetic_sequence : ℕ → ℕ
| 0 => 0
| n + 1 => 3 * (n + 1)

-- Define the sum of the first 10 terms of the sequence.
def rods_total (n : ℕ) : ℕ :=
  let a := 3
  let d := 3
  n / 2 * (2 * a + (n - 1) * d)

def rods_count : ℕ :=
  rods_total 10

-- Define the 11th triangular number for connectors.
def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def connectors_count : ℕ :=
  triangular_number 11

-- Define the total number of pieces.
def total_pieces : ℕ :=
  rods_count + connectors_count

-- The theorem we aim to prove.
theorem total_pieces_equiv_231 : total_pieces = 231 := by
  sorry

end total_pieces_equiv_231_l1813_181311


namespace find_X_l1813_181387

theorem find_X (X : ℚ) (h : (1/3 : ℚ) * (1/4 : ℚ) * X = (1/4 : ℚ) * (1/6 : ℚ) * 120) : X = 60 := 
sorry

end find_X_l1813_181387


namespace augmented_matrix_solution_l1813_181319

theorem augmented_matrix_solution :
  ∀ (m n : ℝ),
  (∃ (x y : ℝ), (m * x = 6 ∧ 3 * y = n) ∧ (x = -3 ∧ y = 4)) →
  m + n = 10 :=
by
  intros m n h
  sorry

end augmented_matrix_solution_l1813_181319


namespace avg_diff_l1813_181358

theorem avg_diff (n : ℕ) (m : ℝ) (mistake : ℝ) (true_value : ℝ)
   (h_n : n = 30) (h_mistake : mistake = 15) (h_true_value : true_value = 105) 
   (h_m : m = true_value - mistake) : 
   (m / n) = 3 := 
by
  sorry

end avg_diff_l1813_181358


namespace find_x_find_y_find_p_q_r_l1813_181333

-- Condition: The number on the line connecting two circles is the sum of the two numbers in the circles.

-- For part (a):
theorem find_x (a b : ℝ) (x : ℝ) (h1 : a + 4 = 13) (h2 : a + b = 10) (h3 : b + 4 = x) : x = 5 :=
by {
  -- Proof can be filled in here to show x = 5 by solving the equations.
  sorry
}

-- For part (b):
theorem find_y (w y : ℝ) (h1 : 3 * w + w = y) (h2 : 6 * w = 48) : y = 32 := 
by {
  -- Proof can be filled in here to show y = 32 by solving the equations.
  sorry
}

-- For part (c):
theorem find_p_q_r (p q r : ℝ) (h1 : p + r = 3) (h2 : p + q = 18) (h3 : q + r = 13) : p = 4 ∧ q = 14 ∧ r = -1 :=
by {
  -- Proof can be filled in here to show p = 4, q = 14, r = -1 by solving the equations.
  sorry
}

end find_x_find_y_find_p_q_r_l1813_181333


namespace batsman_average_increase_l1813_181320

theorem batsman_average_increase
  (A : ℕ)
  (h_average_after_17th : (16 * A + 90) / 17 = 42) :
  42 - A = 3 :=
by
  sorry

end batsman_average_increase_l1813_181320


namespace select_student_based_on_variance_l1813_181307

-- Define the scores for students A and B
def scoresA : List ℚ := [12.1, 12.1, 12.0, 11.9, 11.8, 12.1]
def scoresB : List ℚ := [12.2, 12.0, 11.8, 12.0, 12.3, 11.7]

-- Define the function to calculate the mean of a list of rational numbers
def mean (scores : List ℚ) : ℚ := (scores.foldr (· + ·) 0) / scores.length

-- Define the function to calculate the variance of a list of rational numbers
def variance (scores : List ℚ) : ℚ :=
  let m := mean scores
  (scores.foldr (λ x acc => acc + (x - m) ^ 2) 0) / scores.length

-- Prove that the variance of student A's scores is less than the variance of student B's scores
theorem select_student_based_on_variance :
  variance scoresA < variance scoresB := by
  sorry

end select_student_based_on_variance_l1813_181307


namespace problem_true_propositions_l1813_181308

-- Definitions
def is_square (q : ℕ) : Prop := q = 4
def is_trapezoid (q : ℕ) : Prop := q ≠ 4
def is_parallelogram (q : ℕ) : Prop := q = 2

-- Propositions
def prop_negation (p : Prop) : Prop := ¬ p
def prop_contrapositive (p q : Prop) : Prop := ¬ q → ¬ p
def prop_inverse (p q : Prop) : Prop := p → q

-- True propositions
theorem problem_true_propositions (a b c : ℕ) (h1 : ¬ (is_square 4)) (h2 : ¬ (is_parallelogram 3)) (h3 : ¬ (a * c^2 > b * c^2 → a > b)) : 
    (prop_negation (is_square 4) ∧ prop_contrapositive (is_trapezoid 3) (is_parallelogram 3)) ∧ ¬ prop_inverse (a * c^2 > b * c^2) (a > b) := 
by
    sorry

end problem_true_propositions_l1813_181308


namespace find_q_l1813_181318

-- Define the conditions and the statement to prove
theorem find_q (p q : ℝ) (hp1 : p > 1) (hq1 : q > 1) 
  (h1 : 1 / p + 1 / q = 3 / 2)
  (h2 : p * q = 9) : q = 6 := 
sorry

end find_q_l1813_181318


namespace largest_multiple_of_8_smaller_than_neg_80_l1813_181323

theorem largest_multiple_of_8_smaller_than_neg_80 :
  ∃ n : ℤ, (8 ∣ n) ∧ n < -80 ∧ ∀ m : ℤ, (8 ∣ m ∧ m < -80 → m ≤ n) :=
sorry

end largest_multiple_of_8_smaller_than_neg_80_l1813_181323


namespace Jenny_recycling_l1813_181389

theorem Jenny_recycling:
  let bottle_weight := 6
  let can_weight := 2
  let glass_jar_weight := 8
  let max_weight := 100
  let num_cans := 20
  let bottle_value := 10
  let can_value := 3
  let glass_jar_value := 12
  let total_money := (num_cans * can_value) + (7 * glass_jar_value) + (0 * bottle_value)
  total_money = 144 ∧ num_cans = 20 ∧ glass_jars = 7 ∧ bottles = 0 := by sorry

end Jenny_recycling_l1813_181389


namespace bus_speed_excluding_stoppages_l1813_181363

theorem bus_speed_excluding_stoppages (v : ℝ) (stoppage_time : ℝ) (speed_incl_stoppages : ℝ) :
  stoppage_time = 15 / 60 ∧ speed_incl_stoppages = 48 → v = 64 :=
by
  intro h
  sorry

end bus_speed_excluding_stoppages_l1813_181363


namespace machine_production_in_10_seconds_l1813_181315

def items_per_minute : ℕ := 150
def seconds_per_minute : ℕ := 60
def production_rate_per_second : ℚ := items_per_minute / seconds_per_minute
def production_time_in_seconds : ℕ := 10
def expected_production_in_ten_seconds : ℚ := 25

theorem machine_production_in_10_seconds :
  (production_rate_per_second * production_time_in_seconds) = expected_production_in_ten_seconds :=
sorry

end machine_production_in_10_seconds_l1813_181315


namespace totalCups_l1813_181340

-- Let's state our definitions based on the conditions:
def servingsPerBox : ℕ := 9
def cupsPerServing : ℕ := 2

-- Our goal is to prove the following statement.
theorem totalCups (hServings: servingsPerBox = 9) (hCups: cupsPerServing = 2) : servingsPerBox * cupsPerServing = 18 := by
  -- The detailed proof will go here.
  sorry

end totalCups_l1813_181340


namespace no_valid_partition_of_nat_l1813_181366

-- Definitions of the sets A, B, and C as nonempty subsets of positive integers
variable (A B C : Set ℕ)

-- Definition to capture the key condition in the problem
def valid_partition (A B C : Set ℕ) : Prop :=
  (∀ x ∈ A, ∀ y ∈ B, (x^2 - x * y + y^2) ∈ C) 

-- The main theorem stating that such a partition is impossible
theorem no_valid_partition_of_nat : 
  (∃ A B C : Set ℕ, A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅ ∧ (∀ x ∈ A, ∀ y ∈ B, (x^2 - x * y + y^2) ∈ C)) → False :=
by
  sorry

end no_valid_partition_of_nat_l1813_181366


namespace rational_root_neg_one_third_l1813_181330

def P (x : ℚ) : ℚ := 3 * x^5 - 4 * x^3 - 7 * x^2 + 2 * x + 1

theorem rational_root_neg_one_third : P (-1/3) = 0 :=
by
  have : (-1/3 : ℚ) ≠ 0 := by norm_num
  sorry

end rational_root_neg_one_third_l1813_181330


namespace percentage_of_same_grade_is_48_l1813_181376

def students_with_same_grade (grades : ℕ × ℕ → ℕ) : ℕ :=
  grades (0, 0) + grades (1, 1) + grades (2, 2) + grades (3, 3) + grades (4, 4)

theorem percentage_of_same_grade_is_48
  (grades : ℕ × ℕ → ℕ)
  (h : grades (0, 0) = 3 ∧ grades (1, 1) = 6 ∧ grades (2, 2) = 8 ∧ grades (3, 3) = 4 ∧ grades (4, 4) = 3)
  (total_students : ℕ) (h_students : total_students = 50) :
  (students_with_same_grade grades / 50 : ℚ) * 100 = 48 :=
by
  sorry

end percentage_of_same_grade_is_48_l1813_181376


namespace smaller_solution_of_quadratic_eq_l1813_181383

theorem smaller_solution_of_quadratic_eq : 
  (exists x y : ℝ, x < y ∧ x^2 - 13 * x + 36 = 0 ∧ y^2 - 13 * y + 36 = 0 ∧ x = 4) :=
by sorry

end smaller_solution_of_quadratic_eq_l1813_181383


namespace equivalent_problem_l1813_181301

variable (a b : ℤ)

def condition1 : Prop :=
  a * (-2)^3 + b * (-2) - 7 = 9

def condition2 : Prop :=
  8 * a + 2 * b - 7 = -23

theorem equivalent_problem (h : condition1 a b) : condition2 a b :=
sorry

end equivalent_problem_l1813_181301


namespace intersection_complement_N_M_eq_singleton_two_l1813_181357

def M : Set ℝ := {y | y ≥ 2}
def N : Set ℝ := {x | x > 2}
def C_R_N : Set ℝ := {x | x ≤ 2}

theorem intersection_complement_N_M_eq_singleton_two :
  (C_R_N ∩ M = {2}) :=
by
  sorry

end intersection_complement_N_M_eq_singleton_two_l1813_181357


namespace tetrahedron_volume_l1813_181341

-- Definition of the required constants and variables
variables {S1 S2 S3 S4 r : ℝ}

-- The volume formula we need to prove
theorem tetrahedron_volume :
  (V = 1/3 * (S1 + S2 + S3 + S4) * r) :=
sorry

end tetrahedron_volume_l1813_181341


namespace other_x_intercept_of_parabola_l1813_181316

theorem other_x_intercept_of_parabola (a b c : ℝ) :
  (∃ x : ℝ, y = a * x ^ 2 + b * x + c) ∧ (2, 10) ∈ {p | ∃ x : ℝ, p = (x, a * x ^ 2 + b * x + c)} ∧ (1, 0) ∈ {p | ∃ x : ℝ, p = (x, a * x ^ 2 + b * x + c)}
  → ∃ x : ℝ, x = 3 ∧ (x, 0) ∈ {p | ∃ x : ℝ, p = (x, a * x ^ 2 + b * x + c)} :=
by
  sorry

end other_x_intercept_of_parabola_l1813_181316


namespace number_of_pictures_deleted_l1813_181355

-- Definitions based on the conditions
def total_files_deleted : ℕ := 17
def songs_deleted : ℕ := 8
def text_files_deleted : ℕ := 7

-- The question rewritten as a Lean theorem statement
theorem number_of_pictures_deleted : 
  (total_files_deleted - songs_deleted - text_files_deleted) = 2 := 
by
  sorry

end number_of_pictures_deleted_l1813_181355


namespace f_when_x_lt_4_l1813_181386

noncomputable def f : ℝ → ℝ := sorry

theorem f_when_x_lt_4 (x : ℝ) (h1 : ∀ y : ℝ, y > 4 → f y = 2^(y-1)) (h2 : ∀ y : ℝ, f (4-y) = f (4+y)) (hx : x < 4) : f x = 2^(7-x) :=
by
  sorry

end f_when_x_lt_4_l1813_181386


namespace tan_of_x_is_3_l1813_181314

theorem tan_of_x_is_3 (x : ℝ) (h : Real.tan x = 3) (hx : Real.cos x ≠ 0) : 
  (Real.sin x + 3 * Real.cos x) / (2 * Real.sin x - 3 * Real.cos x) = 2 :=
by
  sorry

end tan_of_x_is_3_l1813_181314


namespace box_surface_area_correct_l1813_181398

-- Define the dimensions of the original cardboard.
def original_length : ℕ := 25
def original_width : ℕ := 40

-- Define the size of the squares removed from each corner.
def square_side : ℕ := 8

-- Define the surface area function.
def surface_area (length width : ℕ) (square_side : ℕ) : ℕ :=
  let area_remaining := (length * width) - 4 * (square_side * square_side)
  area_remaining

-- The theorem statement to prove
theorem box_surface_area_correct : surface_area original_length original_width square_side = 744 :=
by
  sorry

end box_surface_area_correct_l1813_181398


namespace solve_problem_l1813_181364

open Classical

-- Definition of the problem conditions
def problem_conditions (x y : ℝ) : Prop :=
  5 * y^2 + 3 * y + 2 = 2 * (10 * x^2 + 3 * y + 3) ∧ y = 3 * x + 1

-- Definition of the quadratic solution considering the quadratic formula
def quadratic_solution (x : ℝ) : Prop :=
  x = (-21 + Real.sqrt 641) / 50 ∨ x = (-21 - Real.sqrt 641) / 50

-- Main theorem statement
theorem solve_problem :
  ∃ x y : ℝ, problem_conditions x y ∧ quadratic_solution x :=
by
  sorry

end solve_problem_l1813_181364


namespace casey_savings_l1813_181349

-- Define the constants given in the problem conditions
def wage_employee_1 : ℝ := 20
def wage_employee_2 : ℝ := 22
def subsidy : ℝ := 6
def hours_per_week : ℝ := 40

-- Define the weekly cost of each employee
def weekly_cost_employee_1 := wage_employee_1 * hours_per_week
def weekly_cost_employee_2 := (wage_employee_2 - subsidy) * hours_per_week

-- Define the savings by hiring the cheaper employee
def savings := weekly_cost_employee_1 - weekly_cost_employee_2

-- Theorem stating the expected savings
theorem casey_savings : savings = 160 := by
  -- Proof is not included
  sorry

end casey_savings_l1813_181349


namespace exists_xyz_l1813_181368

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem exists_xyz :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  (x + sum_of_digits x = y + sum_of_digits y ∧ y + sum_of_digits y = z + sum_of_digits z) :=
by {
  sorry
}

end exists_xyz_l1813_181368


namespace frequencies_of_first_class_products_confidence_in_difference_of_quality_l1813_181378

theorem frequencies_of_first_class_products 
  (aA bA aB bB : ℕ) 
  (total_A total_B total_products : ℕ) 
  (hA : aA = 150) (hB : bA = 50) (hC : aB = 120) (hD : bB = 80) 
  (hE : total_A = 200) (hF : total_B = 200) (h_total : total_products = total_A + total_B) : 
  aA / total_A = 3 / 4 ∧ aB / total_B = 3 / 5 := 
by
  sorry

theorem confidence_in_difference_of_quality 
  (n a b c d : ℕ) 
  (h_n : n = 400) (hA : a = 150) (hB : b = 50) (hC : c = 120) (hD : d = 80) 
  (K2 : ℝ) 
  (hK2 : K2 = n * ((a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))) : 
  K2 > 6.635 ∧ K2 < 10.828 := 
by
  sorry

end frequencies_of_first_class_products_confidence_in_difference_of_quality_l1813_181378


namespace fraction_addition_solution_is_six_l1813_181331

theorem fraction_addition_solution_is_six :
  (1 / 9) + (1 / 18) = 1 / 6 := 
sorry

end fraction_addition_solution_is_six_l1813_181331


namespace all_three_digits_same_two_digits_same_all_digits_different_l1813_181351

theorem all_three_digits_same (a : ℕ) (h1 : a < 10) (h2 : 3 * a = 24) : a = 8 :=
by sorry

theorem two_digits_same (a b : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : 2 * a + b = 24 ∨ a + 2 * b = 24) : 
  (a = 9 ∧ b = 6) ∨ (a = 6 ∧ b = 9) :=
by sorry

theorem all_digits_different (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10)
  (h4 : a ≠ b) (h5 : a ≠ c) (h6 : b ≠ c) (h7 : a + b + c = 24) :
  (a, b, c) = (7, 8, 9) ∨ (a, b, c) = (7, 9, 8) ∨ (a, b, c) = (8, 7, 9) ∨ (a, b, c) = (8, 9, 7) ∨ (a, b, c) = (9, 7, 8) ∨ (a, b, c) = (9, 8, 7) :=
by sorry

end all_three_digits_same_two_digits_same_all_digits_different_l1813_181351


namespace num_right_angle_triangles_l1813_181325

-- Step d): Lean 4 statement
theorem num_right_angle_triangles {C : ℝ × ℝ} (hC : C.2 = 0) :
  (C = (-2, 0) ∨ C = (4, 0) ∨ C = (1, 0)) ↔ ∃ A B : ℝ × ℝ,
  (A = (-2, 3)) ∧ (B = (4, 3)) ∧ 
  (A.2 = B.2) ∧ (A.1 ≠ B.1) ∧ 
  (((C.1-A.1)*(B.1-A.1) + (C.2-A.2)*(B.2-A.2) = 0) ∨ 
   ((C.1-B.1)*(A.1-B.1) + (C.2-B.2)*(A.2-B.2) = 0)) :=
sorry

end num_right_angle_triangles_l1813_181325


namespace angles_equal_l1813_181391

theorem angles_equal (α θ γ : Real) (hα : 0 < α ∧ α < π / 2) (hθ : 0 < θ ∧ θ < π / 2) (hγ : 0 < γ ∧ γ < π / 2)
  (h : Real.sin (α + γ) * Real.tan α = Real.sin (θ + γ) * Real.tan θ) : α = θ :=
by
  sorry

end angles_equal_l1813_181391


namespace no_common_interior_points_l1813_181382

open Metric

-- Define the distance conditions for two convex polygons F1 and F2
variables {F1 F2 : Set (EuclideanSpace ℝ (Fin 2))}

-- F1 is a convex polygon
def is_convex (S : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∀ {x y : EuclideanSpace ℝ (Fin 2)} {a b : ℝ},
    x ∈ S → y ∈ S → 0 ≤ a → 0 ≤ b → a + b = 1 → a • x + b • y ∈ S

-- Conditions provided in the problem
def condition1 (F : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∀ {x y : EuclideanSpace ℝ (Fin 2)}, x ∈ F → y ∈ F → dist x y ≤ 1

def condition2 (F1 F2 : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∀ {x : EuclideanSpace ℝ (Fin 2)} {y : EuclideanSpace ℝ (Fin 2)}, x ∈ F1 → y ∈ F2 → dist x y > 1 / Real.sqrt 2

-- The theorem to prove
theorem no_common_interior_points (h1 : is_convex F1) (h2 : is_convex F2) 
  (h3 : condition1 F1) (h4 : condition1 F2) (h5 : condition2 F1 F2) :
  ∀ p ∈ interior F1, ∀ q ∈ interior F2, p ≠ q :=
sorry

end no_common_interior_points_l1813_181382


namespace ratio_cost_to_marked_price_l1813_181370

theorem ratio_cost_to_marked_price (p : ℝ) (hp : p > 0) :
  let selling_price := (3 / 4) * p
  let cost_price := (5 / 6) * selling_price
  cost_price / p = 5 / 8 :=
by 
  sorry

end ratio_cost_to_marked_price_l1813_181370


namespace payment_first_trip_payment_second_trip_l1813_181371

-- Define conditions and questions
variables {x y : ℝ}

-- Conditions: discounts and expenditure
def discount_1st_trip (x : ℝ) := 0.9 * x
def discount_2nd_trip (y : ℝ) := 300 * 0.9 + (y - 300) * 0.8

def combined_discount (x y : ℝ) := 300 * 0.9 + (x + y - 300) * 0.8

-- Given conditions as equations
axiom eq1 : discount_1st_trip x + discount_2nd_trip y - combined_discount x y = 19
axiom eq2 : x + y - (discount_1st_trip x + discount_2nd_trip y) = 67

-- The proof statements
theorem payment_first_trip : discount_1st_trip 190 = 171 := by sorry

theorem payment_second_trip : discount_2nd_trip 390 = 342 := by sorry

end payment_first_trip_payment_second_trip_l1813_181371


namespace correct_calculation_l1813_181365

theorem correct_calculation :
  (∃ (x y : ℝ), 5 * x + 2 * y ≠ 7 * x * y) ∧
  (∃ (x : ℝ), 3 * x - 2 * x ≠ 1) ∧
  (∃ (x : ℝ), x^2 + x^5 ≠ x^7) →
  (∀ (x y : ℝ), 3 * x^2 * y - 4 * y * x^2 = -x^2 * y) :=
by
  sorry

end correct_calculation_l1813_181365


namespace find_m_l1813_181310

noncomputable def quadratic_eq (x : ℝ) (m : ℝ) : ℝ := 2 * x^2 + 4 * x + m

theorem find_m (x₁ x₂ m : ℝ) 
  (h1 : quadratic_eq x₁ m = 0)
  (h2 : quadratic_eq x₂ m = 0)
  (h3 : 16 - 8 * m ≥ 0)
  (h4 : x₁^2 + x₂^2 + 2 * x₁ * x₂ - x₁^2 * x₂^2 = 0) 
  : m = -4 :=
sorry

end find_m_l1813_181310


namespace hannahs_grapes_per_day_l1813_181335

-- Definitions based on conditions
def oranges_per_day : ℕ := 20
def days : ℕ := 30
def total_fruits : ℕ := 1800
def total_oranges : ℕ := oranges_per_day * days

-- The math proof problem to be targeted
theorem hannahs_grapes_per_day : 
  (total_fruits - total_oranges) / days = 40 := 
by
  -- Proof to be filled in here
  sorry

end hannahs_grapes_per_day_l1813_181335


namespace camilla_jellybeans_l1813_181344

theorem camilla_jellybeans (b c : ℕ) (h1 : b = 3 * c) (h2 : b - 20 = 4 * (c - 20)) :
  b = 180 :=
by
  -- Proof steps would go here
  sorry

end camilla_jellybeans_l1813_181344


namespace find_a1_over_d_l1813_181332

variable {a : ℕ → ℝ} (d : ℝ)

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem find_a1_over_d 
  (d_ne_zero : d ≠ 0) 
  (seq : arithmetic_sequence a d) 
  (h : a 2021 = a 20 + a 21) : 
  a 1 / d = 1981 :=
by 
  sorry

end find_a1_over_d_l1813_181332


namespace sufficient_not_necessary_l1813_181354

theorem sufficient_not_necessary (p q: Prop) :
  ¬ (p ∨ q) → ¬ p ∧ (¬ p → ¬(¬ p ∧ ¬ q)) := sorry

end sufficient_not_necessary_l1813_181354


namespace sales_on_second_street_l1813_181322

noncomputable def commission_per_system : ℕ := 25
noncomputable def total_commission : ℕ := 175
noncomputable def total_systems_sold : ℕ := total_commission / commission_per_system

def first_street_sales (S : ℕ) : ℕ := S
def second_street_sales (S : ℕ) : ℕ := 2 * S
def third_street_sales : ℕ := 0
def fourth_street_sales : ℕ := 1

def total_sales (S : ℕ) : ℕ := first_street_sales S + second_street_sales S + third_street_sales + fourth_street_sales

theorem sales_on_second_street (S : ℕ) : total_sales S = total_systems_sold → second_street_sales S = 4 := by
  sorry

end sales_on_second_street_l1813_181322


namespace evaluate_expression_l1813_181345

theorem evaluate_expression :
  let a := 5 ^ 1001
  let b := 6 ^ 1002
  (a + b) ^ 2 - (a - b) ^ 2 = 24 * 30 ^ 1001 :=
by
  sorry

end evaluate_expression_l1813_181345
