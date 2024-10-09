import Mathlib

namespace sandwiches_bought_l1272_127245

theorem sandwiches_bought (sandwich_cost soda_cost total_cost_sodas total_cost : ℝ)
  (h1 : sandwich_cost = 2.45)
  (h2 : soda_cost = 0.87)
  (h3 : total_cost_sodas = 4 * soda_cost)
  (h4 : total_cost = 8.38) :
  ∃ (S : ℕ), sandwich_cost * S + total_cost_sodas = total_cost ∧ S = 2 :=
by
  use 2
  simp [h1, h2, h3, h4]
  sorry

end sandwiches_bought_l1272_127245


namespace probability_of_odd_sum_l1272_127267

open Nat

def first_twelve_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_of_odd_sum :
  (binomial 11 3) / (binomial 12 4) = 1 / 3 := by
sorry

end probability_of_odd_sum_l1272_127267


namespace solve_olympics_problem_max_large_sets_l1272_127290

-- Definitions based on the conditions
variables (x y : ℝ)

-- Condition 1: 2 small sets cost $20 less than 1 large set
def condition1 : Prop := y - 2 * x = 20

-- Condition 2: 3 small sets and 2 large sets cost $390
def condition2 : Prop := 3 * x + 2 * y = 390

-- Finding unit prices
def unit_prices : Prop := x = 50 ∧ y = 120

-- Condition 3: Budget constraint for purchasing sets
def budget_constraint (m : ℕ) : Prop := m ≤ 7

-- Prove unit prices and purchasing constraints
theorem solve_olympics_problem :
  condition1 x y ∧ condition2 x y → unit_prices x y :=
by
  sorry

theorem max_large_sets :
  budget_constraint 7 :=
by
  sorry

end solve_olympics_problem_max_large_sets_l1272_127290


namespace q_is_20_percent_less_than_p_l1272_127242

theorem q_is_20_percent_less_than_p (p q : ℝ) (h : p = 1.25 * q) : (q - p) / p * 100 = -20 := by
  sorry

end q_is_20_percent_less_than_p_l1272_127242


namespace distance_bob_walked_when_met_l1272_127249

theorem distance_bob_walked_when_met (distance_XY walk_rate_Yolanda walk_rate_Bob : ℕ)
  (start_time_Yolanda start_time_Bob : ℕ) (y_distance b_distance : ℕ) (t : ℕ)
  (h1 : distance_XY = 65)
  (h2 : walk_rate_Yolanda = 5)
  (h3 : walk_rate_Bob = 7)
  (h4 : start_time_Yolanda = 0)
  (h5 : start_time_Bob = 1)
  (h6 : y_distance = walk_rate_Yolanda * (t + start_time_Bob))
  (h7 : b_distance = walk_rate_Bob * t)
  (h8 : y_distance + b_distance = distance_XY) : 
  b_distance = 35 := 
sorry

end distance_bob_walked_when_met_l1272_127249


namespace problem_statement_l1272_127230

theorem problem_statement (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (1 / Real.sqrt (2011 + Real.sqrt (2011^2 - 1)) = Real.sqrt m - Real.sqrt n) →
  m + n = 2011 :=
sorry

end problem_statement_l1272_127230


namespace part1_part2_l1272_127209

open Real

-- Definitions used in the proof
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0 ∧ a > 0
def q (x : ℝ) : Prop := abs (x - 1) ≤ 2 ∧ (x + 3) / (x - 2) ≥ 0

theorem part1 (x : ℝ) : (p 1 x ∧ q x) → 2 < x ∧ x ≤ 3 := by
  sorry

theorem part2 (a : ℝ) : (¬ (∃ x, p a x) → ¬ (∃ x, q x)) → a > 3 / 2 := by
  sorry

end part1_part2_l1272_127209


namespace intersection_M_N_l1272_127244

open Set

noncomputable def M : Set ℝ := {x | x ≥ 2}

noncomputable def N : Set ℝ := {x | x^2 - 6*x + 5 < 0}

theorem intersection_M_N : M ∩ N = {x | 2 ≤ x ∧ x < 5} :=
by
  sorry

end intersection_M_N_l1272_127244


namespace units_digit_product_composites_l1272_127299

theorem units_digit_product_composites :
  (4 * 6 * 8 * 9 * 10) % 10 = 0 :=
sorry

end units_digit_product_composites_l1272_127299


namespace no_positive_abc_exists_l1272_127291

theorem no_positive_abc_exists 
  (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h1 : b^2 ≥ 4 * a * c)
  (h2 : c^2 ≥ 4 * b * a)
  (h3 : a^2 ≥ 4 * b * c)
  : false :=
sorry

end no_positive_abc_exists_l1272_127291


namespace problem_statement_l1272_127223

theorem problem_statement {f : ℝ → ℝ}
  (Hodd : ∀ x, f (-x) = -f x)
  (Hdecreasing : ∀ x y, x < y → f x > f y)
  (a b : ℝ) (H : f a + f b > 0) : a + b < 0 :=
sorry

end problem_statement_l1272_127223


namespace color_of_241st_marble_l1272_127254

def sequence_color (n : ℕ) : String :=
  if n % 14 < 6 then "blue"
  else if n % 14 < 11 then "red"
  else "green"

theorem color_of_241st_marble : sequence_color 240 = "blue" :=
  by
  sorry

end color_of_241st_marble_l1272_127254


namespace fraction_proof_l1272_127287

-- Define the fractions as constants
def a := 1 / 3
def b := 1 / 4
def c := 1 / 2
def d := 1 / 3

-- Prove the main statement
theorem fraction_proof : (a - b) / (c - d) = 1 / 2 := by
  sorry

end fraction_proof_l1272_127287


namespace same_root_a_eq_3_l1272_127262

theorem same_root_a_eq_3 {x a : ℝ} (h1 : 3 * x - 2 * a = 0) (h2 : 2 * x + 3 * a - 13 = 0) : a = 3 :=
by
  sorry

end same_root_a_eq_3_l1272_127262


namespace square_side_length_l1272_127288

theorem square_side_length (a : ℚ) (s : ℚ) (h : a = 9/16) (h_area : s^2 = a) : s = 3/4 :=
by {
  -- proof omitted
  sorry
}

end square_side_length_l1272_127288


namespace boat_speed_in_still_water_l1272_127200

/--
The speed of the stream is 6 kmph.
The boat can cover 48 km downstream or 32 km upstream in the same time.
We want to prove that the speed of the boat in still water is 30 kmph.
-/
theorem boat_speed_in_still_water (x : ℝ)
  (h1 : ∃ t : ℝ, t = 48 / (x + 6) ∧ t = 32 / (x - 6)) : x = 30 :=
by
  sorry

end boat_speed_in_still_water_l1272_127200


namespace compare_nsquare_pow2_pos_int_l1272_127226

-- Proposition that captures the given properties of comparing n^2 and 2^n
theorem compare_nsquare_pow2_pos_int (n : ℕ) (hn : n > 0) : 
  (n = 1 → n^2 < 2^n) ∧
  (n = 2 → n^2 = 2^n) ∧
  (n = 3 → n^2 > 2^n) ∧
  (n = 4 → n^2 = 2^n) ∧
  (n ≥ 5 → n^2 < 2^n) :=
by
  sorry

end compare_nsquare_pow2_pos_int_l1272_127226


namespace triangle_area_l1272_127219

-- Define a triangle as a structure with vertices A, B, and C, where the lengths AB, AC, and BC are provided
structure Triangle :=
  (A B C : ℝ)
  (AB AC BC : ℝ)
  (is_isosceles : AB = AC)
  (BC_length : BC = 20)
  (AB_length : AB = 26)

-- Define the length bisector and Pythagorean properties
def bisects_base (t : Triangle) : Prop :=
  ∃ D : ℝ, (t.B - D) = (D - t.C) ∧ 2 * D = t.B + t.C

def pythagorean_theorem_AD (t : Triangle) (D : ℝ) (AD : ℝ) : Prop :=
  t.AB^2 = AD^2 + (t.B - D)^2

-- State the problem as a theorem
theorem triangle_area (t : Triangle) (D : ℝ) (AD : ℝ) (h1 : bisects_base t) (h2 : pythagorean_theorem_AD t D AD) :
  AD = 24 ∧ (1 / 2) * t.BC * AD = 240 :=
sorry

end triangle_area_l1272_127219


namespace nhai_highway_construction_l1272_127284

/-- Problem definition -/
def total_man_hours (men1 men2 days1 days2 hours1 hours2 : Nat) : Nat := 
  (men1 * days1 * hours1) + (men2 * days2 * hours2)

theorem nhai_highway_construction :
  let men := 100
  let days1 := 25
  let days2 := 25
  let hours1 := 8
  let hours2 := 10
  let additional_men := 60
  let total_days := 50
  total_man_hours men (men + additional_men) total_days total_days hours1 hours2 = 
  2 * total_man_hours men men days1 days2 hours1 hours1 :=
  sorry

end nhai_highway_construction_l1272_127284


namespace isosceles_triangle_if_perpendiculars_intersect_at_single_point_l1272_127229

noncomputable def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

theorem isosceles_triangle_if_perpendiculars_intersect_at_single_point
  (a b c : ℝ)
  (D E F P Q R H : Type)
  (intersection_point: P = Q ∧ Q = R ∧ P = R ∧ P = H) :
  is_isosceles_triangle a b c := 
sorry

end isosceles_triangle_if_perpendiculars_intersect_at_single_point_l1272_127229


namespace original_number_l1272_127233

theorem original_number (x : ℝ) (h : 1.47 * x = 1214.33) : x = 826.14 :=
sorry

end original_number_l1272_127233


namespace quinn_free_donuts_l1272_127214

-- Definitions based on conditions
def books_per_week : ℕ := 2
def weeks : ℕ := 10
def books_needed_for_donut : ℕ := 5

-- Calculation based on conditions
def total_books_read : ℕ := books_per_week * weeks
def free_donuts (total_books : ℕ) : ℕ := total_books / books_needed_for_donut

-- Proof statement
theorem quinn_free_donuts : free_donuts total_books_read = 4 := by
  sorry

end quinn_free_donuts_l1272_127214


namespace solve_diamond_l1272_127278

theorem solve_diamond (d : ℕ) (h : d * 6 + 5 = d * 7 + 2) : d = 3 :=
by
  sorry

end solve_diamond_l1272_127278


namespace find_divisor_l1272_127258

def div_remainder (a b r : ℕ) : Prop :=
  ∃ k : ℕ, a = k * b + r

theorem find_divisor :
  ∃ D : ℕ, (div_remainder 242 D 15) ∧ (div_remainder 698 D 27) ∧ (div_remainder (242 + 698) D 5) ∧ D = 37 := 
by
  sorry

end find_divisor_l1272_127258


namespace solution_fraction_l1272_127215

-- Conditions and definition of x
def initial_quantity : ℝ := 1
def concentration_70 : ℝ := 0.70
def concentration_25 : ℝ := 0.25
def concentration_new : ℝ := 0.35

-- Definition of the fraction of the solution replaced
def x (fraction : ℝ) : Prop :=
  concentration_70 * initial_quantity - concentration_70 * fraction + concentration_25 * fraction = concentration_new * initial_quantity

-- The theorem we need to prove
theorem solution_fraction : ∃ (fraction : ℝ), x fraction ∧ fraction = 7 / 9 :=
by
  use 7 / 9
  simp [x]
  sorry  -- Proof steps would be filled here

end solution_fraction_l1272_127215


namespace broken_crayons_l1272_127298

theorem broken_crayons (total new used : Nat) (h1 : total = 14) (h2 : new = 2) (h3 : used = 4) :
  total = new + used + 8 :=
by
  -- Proof omitted
  sorry

end broken_crayons_l1272_127298


namespace largest_sum_of_ABC_l1272_127240

-- Define the variables and the conditions
def A := 533
def B := 5
def C := 1

-- Define the product condition
def product_condition : Prop := (A * B * C = 2665)

-- Define the distinct positive integers condition
def distinct_positive_integers_condition : Prop := (A > 0 ∧ B > 0 ∧ C > 0 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C)

-- State the theorem
theorem largest_sum_of_ABC : product_condition → distinct_positive_integers_condition → A + B + C = 539 := by
  intros _ _
  sorry

end largest_sum_of_ABC_l1272_127240


namespace book_shelf_arrangement_l1272_127277

-- Definitions for the problem conditions
def math_books := 3
def english_books := 4
def science_books := 2

-- The total number of ways to arrange the books
def total_arrangements :=
  (Nat.factorial (math_books + english_books + science_books - 6)) * -- For the groups
  (Nat.factorial math_books) * -- For math books within the group
  (Nat.factorial english_books) * -- For English books within the group
  (Nat.factorial science_books) -- For science books within the group

theorem book_shelf_arrangement :
  total_arrangements = 1728 := by
  -- Proof starts here
  sorry

end book_shelf_arrangement_l1272_127277


namespace problem_inequality_l1272_127204

theorem problem_inequality (n a b : ℕ) (h₁ : n ≥ 2) 
  (h₂ : ∀ m, 2^m ∣ 5^n - 3^n → m ≤ a) 
  (h₃ : ∀ m, 2^m ≤ n → m ≤ b) : a ≤ b + 3 :=
sorry

end problem_inequality_l1272_127204


namespace flowers_to_embroider_l1272_127246

-- Defining constants based on the problem conditions
def stitches_per_minute : ℕ := 4
def stitches_per_flower : ℕ := 60
def stitches_per_unicorn : ℕ := 180
def stitches_per_godzilla : ℕ := 800
def num_unicorns : ℕ := 3
def num_godzillas : ℕ := 1
def total_minutes : ℕ := 1085

-- Theorem statement to prove the number of flowers Carolyn wants to embroider
theorem flowers_to_embroider : 
  (total_minutes * stitches_per_minute - (num_godzillas * stitches_per_godzilla + num_unicorns * stitches_per_unicorn)) / stitches_per_flower = 50 :=
by
  sorry

end flowers_to_embroider_l1272_127246


namespace S_10_minus_S_7_l1272_127234

-- Define the first term and common difference of the arithmetic sequence
variables (a₁ d : ℕ)

-- Define the arithmetic sequence based on the first term and common difference
def arithmetic_sequence (n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

-- Conditions given in the problem
axiom a_5_eq : a₁ + 4 * d = 8
axiom S_3_eq : sum_arithmetic_sequence a₁ 3 = 6

-- The goal: prove that S_10 - S_7 = 48
theorem S_10_minus_S_7 : sum_arithmetic_sequence a₁ 10 - sum_arithmetic_sequence a₁ 7 = 48 :=
sorry

end S_10_minus_S_7_l1272_127234


namespace sin_cos_equation_solution_l1272_127270

open Real

theorem sin_cos_equation_solution (x : ℝ): 
  (∃ n : ℤ, x = (π / 4050) + (π * n / 2025)) ∨ (∃ k : ℤ, x = (π * k / 9)) ↔ 
  sin (2025 * x) ^ 4 + (cos (2016 * x) ^ 2019) * (cos (2025 * x) ^ 2018) = 1 := 
by 
  sorry

end sin_cos_equation_solution_l1272_127270


namespace Dan_work_hours_l1272_127247

theorem Dan_work_hours (x : ℝ) :
  (1 / 15) * x + 3 / 5 = 1 → x = 6 :=
by
  intro h
  sorry

end Dan_work_hours_l1272_127247


namespace problem1_problem2_problem3_l1272_127276

def A : Set ℝ := Set.Icc (-1) 1
def B : Set ℝ := Set.Icc (-2) 2
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 + m * x - 1
def g (a m x : ℝ) : ℝ := 2 * abs (x - a) - x^2 - m * x

theorem problem1 (m : ℝ) : (∀ x, f m x ≤ 0 → x ∈ A) → m ∈ Set.Icc (-1) 1 :=
sorry

theorem problem2 (f_eq : ∀ x, f (-4) (1-x) = f (-4) (1+x)) : 
  Set.range (f (-4) ∘ id) ⊆ Set.Icc (-3) 15 :=
sorry

theorem problem3 (a : ℝ) (m : ℝ) :
  (a ≤ -1 → ∃ x, f m x + g a m x = -2*a - 2) ∧
  (-1 < a ∧ a < 1 → ∃ x, f m x + g a m x = a^2 - 1) ∧
  (a ≥ 1 → ∃ x, f m x + g a m x = 2*a - 2) :=
sorry

end problem1_problem2_problem3_l1272_127276


namespace min_value_expression_l1272_127292

theorem min_value_expression : ∃ (x y : ℝ), x^2 + 2*x*y + 3*y^2 - 6*x - 2*y = -11 := by
  sorry

end min_value_expression_l1272_127292


namespace number_of_players_l1272_127227

/-- Jane bought 600 minnows, each prize has 3 minnows, 15% of the players win a prize, 
and 240 minnows are left over. To find the total number of players -/
theorem number_of_players (total_minnows left_over_minnows minnows_per_prize prizes_win_percent : ℕ) 
(h1 : total_minnows = 600) 
(h2 : minnows_per_prize = 3)
(h3 : prizes_win_percent * 100 = 15)
(h4 : left_over_minnows = 240) : 
total_minnows - left_over_minnows = 360 → 
  360 / minnows_per_prize = 120 → 
  (prizes_win_percent * 100 / 100) * P = 120 → 
  P = 800 := 
by 
  sorry

end number_of_players_l1272_127227


namespace trig_expression_value_l1272_127205

theorem trig_expression_value (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 :=
by
  sorry

end trig_expression_value_l1272_127205


namespace race_winner_l1272_127238

-- Definitions and conditions based on the problem statement
def tortoise_speed : ℕ := 5  -- Tortoise speed in meters per minute
def hare_speed_1 : ℕ := 20  -- Hare initial speed in meters per minute
def hare_time_1 : ℕ := 3  -- Hare initial running time in minutes
def hare_speed_2 : ℕ := 10  -- Hare speed when going back in meters per minute
def hare_time_2 : ℕ := 2  -- Hare back running time in minutes
def hare_sleep_time : ℕ := 5  -- Hare sleeping time in minutes
def hare_speed_3 : ℕ := 25  -- Hare final speed in meters per minute
def track_length : ℕ := 130  -- Total length of the race track in meters

-- The problem statement
theorem race_winner :
  track_length / tortoise_speed > hare_time_1 + hare_time_2 + hare_sleep_time + (track_length - (hare_speed_1 * hare_time_1 - hare_speed_2 * hare_time_2)) / hare_speed_3 :=
sorry

end race_winner_l1272_127238


namespace arithmetic_series_sum_l1272_127250

theorem arithmetic_series_sum :
  let first_term := -25
  let common_difference := 2
  let last_term := 19
  let n := (last_term - first_term) / common_difference + 1
  let sum := n * (first_term + last_term) / 2
  sum = -69 :=
by
  sorry

end arithmetic_series_sum_l1272_127250


namespace alexandra_magazines_l1272_127256

theorem alexandra_magazines :
  let friday_magazines := 8
  let saturday_magazines := 12
  let sunday_magazines := 4 * friday_magazines
  let dog_chewed_magazines := 4
  let total_magazines_before_dog := friday_magazines + saturday_magazines + sunday_magazines
  let total_magazines_now := total_magazines_before_dog - dog_chewed_magazines
  total_magazines_now = 48 := by
  sorry

end alexandra_magazines_l1272_127256


namespace new_average_age_l1272_127220

theorem new_average_age (avg_age : ℕ) (num_students : ℕ) (teacher_age : ℕ) (new_num_individuals : ℕ) (new_avg_age : ℕ) :
  avg_age = 15 ∧ num_students = 20 ∧ teacher_age = 36 ∧ new_num_individuals = 21 →
  new_avg_age = (num_students * avg_age + teacher_age) / new_num_individuals → new_avg_age = 16 :=
by
  intros
  sorry

end new_average_age_l1272_127220


namespace ratio_bound_exceeds_2023_power_l1272_127259

theorem ratio_bound_exceeds_2023_power (a b : ℕ → ℝ) (h_pos : ∀ n, 0 < a n ∧ 0 < b n)
  (h1 : ∀ n, (a (n + 1)) * (b (n + 1)) = (a n)^2 + (b n)^2)
  (h2 : ∀ n, (a (n + 1)) + (b (n + 1)) = (a n) * (b n))
  (h3 : ∀ n, a n ≥ b n) :
  ∃ n, (a n) / (b n) > 2023^2023 :=
by
  sorry

end ratio_bound_exceeds_2023_power_l1272_127259


namespace total_amount_spent_correct_l1272_127224

noncomputable def total_amount_spent (mango_cost pineapple_cost cost_pineapple total_people : ℕ) : ℕ :=
  let pineapple_people := cost_pineapple / pineapple_cost
  let mango_people := total_people - pineapple_people
  let mango_cost_total := mango_people * mango_cost
  cost_pineapple + mango_cost_total

theorem total_amount_spent_correct :
  total_amount_spent 5 6 54 17 = 94 := by
  -- This is where the proof would go, but it's omitted per instructions
  sorry

end total_amount_spent_correct_l1272_127224


namespace parsley_rows_l1272_127228

-- Define the conditions laid out in the problem
def garden_rows : ℕ := 20
def plants_per_row : ℕ := 10
def rosemary_rows : ℕ := 2
def chives_planted : ℕ := 150

-- Define the target statement to prove
theorem parsley_rows :
  let total_plants := garden_rows * plants_per_row
  let remaining_rows := garden_rows - rosemary_rows
  let chives_rows := chives_planted / plants_per_row
  let parsley_rows := remaining_rows - chives_rows
  parsley_rows = 3 :=
by
  sorry

end parsley_rows_l1272_127228


namespace rotted_tomatoes_is_correct_l1272_127225

noncomputable def shipment_1 : ℕ := 1000
noncomputable def sold_Saturday : ℕ := 300
noncomputable def shipment_2 : ℕ := 2 * shipment_1
noncomputable def tomatoes_Tuesday : ℕ := 2500

-- Define remaining tomatoes after the first shipment accounting for Saturday's sales
noncomputable def remaining_tomatoes_1 : ℕ := shipment_1 - sold_Saturday

-- Define total tomatoes after second shipment arrives
noncomputable def total_tomatoes_after_second_shipment : ℕ := remaining_tomatoes_1 + shipment_2

-- Define the amount of tomatoes that rotted
noncomputable def rotted_tomatoes : ℕ :=
  total_tomatoes_after_second_shipment - tomatoes_Tuesday

theorem rotted_tomatoes_is_correct :
  rotted_tomatoes = 200 := by
  sorry

end rotted_tomatoes_is_correct_l1272_127225


namespace mean_variance_transformation_l1272_127222

variable (n : ℕ)
variable (x : Fin n → ℝ)
variable (mean_original variance_original : ℝ)
variable (meam_new variance_new : ℝ)
variable (offset : ℝ)

theorem mean_variance_transformation (hmean : mean_original = 2.8) (hvariance : variance_original = 3.6) 
  (hoffset : offset = 60) : 
  (mean_new = mean_original + offset) ∧ (variance_new = variance_original) :=
  sorry

end mean_variance_transformation_l1272_127222


namespace expected_worth_coin_flip_l1272_127272

def prob_head : ℚ := 2 / 3
def prob_tail : ℚ := 1 / 3
def gain_head : ℚ := 5
def loss_tail : ℚ := -12

theorem expected_worth_coin_flip : ∃ E : ℚ, E = round (((prob_head * gain_head) + (prob_tail * loss_tail)) * 100) / 100 ∧ E = - (2 / 3) :=
by
  sorry

end expected_worth_coin_flip_l1272_127272


namespace floor_sqrt_12_squared_l1272_127293

theorem floor_sqrt_12_squared : (Int.floor (Real.sqrt 12))^2 = 9 := by
  sorry

end floor_sqrt_12_squared_l1272_127293


namespace cost_of_one_hockey_stick_l1272_127279

theorem cost_of_one_hockey_stick (x : ℝ)
    (h1 : x * 2 + 25 = 68) : x = 21.50 :=
by
  sorry

end cost_of_one_hockey_stick_l1272_127279


namespace exists_set_with_property_l1272_127269

theorem exists_set_with_property (n : ℕ) (h : n > 0) :
  ∃ S : Finset ℕ, S.card = n ∧
  (∀ {a b}, a ∈ S → b ∈ S → a ≠ b → (a - b) ∣ a ∧ (a - b) ∣ b) ∧
  (∀ {a b c}, a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → ¬ ((a - b) ∣ c)) :=
sorry

end exists_set_with_property_l1272_127269


namespace find_y_l1272_127211

theorem find_y (y : ℚ) (h : 1/3 - 1/4 = 4/y) : y = 48 := sorry

end find_y_l1272_127211


namespace compute_pqr_l1272_127243

theorem compute_pqr (p q r : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) 
  (h_sum : p + q + r = 26) (h_eq : (1 : ℚ) / ↑p + (1 : ℚ) / ↑q + (1 : ℚ) / ↑r + 360 / (p * q * r) = 1) : 
  p * q * r = 576 := 
sorry

end compute_pqr_l1272_127243


namespace common_ratio_of_geometric_sequence_l1272_127264

open BigOperators

theorem common_ratio_of_geometric_sequence
  (a1 : ℝ) (q : ℝ)
  (h1 : 2 * (a1 * q^5) = 3 * (a1 * (1 - q^4) / (1 - q)) + 1)
  (h2 : a1 * q^6 = 3 * (a1 * (1 - q^5) / (1 - q)) + 1)
  (h_pos : a1 > 0) :
  q = 3 :=
sorry

end common_ratio_of_geometric_sequence_l1272_127264


namespace quadrilateral_area_is_two_l1272_127285

def A : (Int × Int) := (0, 0)
def B : (Int × Int) := (2, 0)
def C : (Int × Int) := (2, 3)
def D : (Int × Int) := (0, 2)

noncomputable def area (p1 p2 p3 p4 : (Int × Int)) : ℚ :=
  (1 / 2 : ℚ) * (abs ((p1.1 * p2.2 + p2.1 * p3.2 + p3.1 * p4.2 + p4.1 * p1.2) - 
                      (p1.2 * p2.1 + p2.2 * p3.1 + p3.2 * p4.1 + p4.2 * p1.1)))

theorem quadrilateral_area_is_two : 
  area A B C D = 2 := by
  sorry

end quadrilateral_area_is_two_l1272_127285


namespace arithmetic_sequence_equal_sum_l1272_127231

variable (a d : ℕ) -- defining first term and common difference as natural numbers
variable (n : ℕ) -- defining n as a natural number

noncomputable def sum_arithmetic_sequence (n: ℕ) (a d: ℕ): ℕ := (n * (2 * a + (n - 1) * d) ) / 2

theorem arithmetic_sequence_equal_sum (a d n : ℕ) :
  sum_arithmetic_sequence (10 * n) a d = sum_arithmetic_sequence (15 * n) a d - sum_arithmetic_sequence (10 * n) a d :=
by
  sorry

end arithmetic_sequence_equal_sum_l1272_127231


namespace max_ab_min_inv_a_plus_4_div_b_l1272_127239

theorem max_ab (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_sum : a + 4 * b = 4) : 
  ab ≤ 1 :=
by
  sorry

theorem min_inv_a_plus_4_div_b (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_sum : a + 4 * b = 4) :
  1 / a + 4 / b ≥ 25 / 4 :=
by
  sorry

end max_ab_min_inv_a_plus_4_div_b_l1272_127239


namespace find_digit_e_l1272_127275

theorem find_digit_e (A B C D E F : ℕ) (h1 : A * 10 + B + (C * 10 + D) = A * 10 + E) (h2 : A * 10 + B - (D * 10 + C) = A * 10 + F) : E = 9 :=
sorry

end find_digit_e_l1272_127275


namespace y_is_one_y_is_neg_two_thirds_l1272_127263

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (1, 3)
def vector_b (y : ℝ) : ℝ × ℝ := (2, y)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Prove y = 1 given dot_product(vector_a, vector_b(y)) = 5
theorem y_is_one (h : dot_product vector_a (vector_b y) = 5) : y = 1 :=
by
  -- We assume the proof (otherwise it would go here)
  sorry

-- Prove y = -2/3 given |vector_a + vector_b(y)| = |vector_a - vector_b(y)|
theorem y_is_neg_two_thirds (h : (vector_a.1 + (vector_b y).1)^2 + (vector_a.2 + (vector_b y).2)^2 =
                                (vector_a.1 - (vector_b y).1)^2 + (vector_a.2 - (vector_b y).2)^2) : y = -2/3 :=
by
  -- We assume the proof (otherwise it would go here)
  sorry

end y_is_one_y_is_neg_two_thirds_l1272_127263


namespace find_a_parallel_lines_l1272_127210

theorem find_a_parallel_lines (a : ℝ) (l1_parallel_l2 : x + a * y + 6 = 0 → (a - 1) * x + 2 * y + 3 * a = 0 → Parallel) : a = -1 :=
sorry

end find_a_parallel_lines_l1272_127210


namespace find_k_l1272_127255

def g (a b c x : ℤ) := a * x^2 + b * x + c

theorem find_k
  (a b c : ℤ)
  (h1 : g a b c 1 = 0)
  (h2 : 20 < g a b c 5 ∧ g a b c 5 < 30)
  (h3 : 40 < g a b c 6 ∧ g a b c 6 < 50)
  (h4 : ∃ k : ℤ, 3000 * k < g a b c 100 ∧ g a b c 100 < 3000 * (k + 1)) :
  ∃ k : ℤ, k = 9 :=
by
  sorry

end find_k_l1272_127255


namespace function_positive_on_interval_l1272_127206

theorem function_positive_on_interval (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → (2 - a^2) * x + a > 0) ↔ 0 < a ∧ a < 2 :=
by
  sorry

end function_positive_on_interval_l1272_127206


namespace candidate_a_valid_votes_l1272_127252

/-- In an election, candidate A got 80% of the total valid votes.
If 15% of the total votes were declared invalid and the total number of votes is 560,000,
find the number of valid votes polled in favor of candidate A. -/
theorem candidate_a_valid_votes :
  let total_votes := 560000
  let invalid_percentage := 0.15
  let valid_percentage := 0.85
  let candidate_a_percentage := 0.80
  let valid_votes := (valid_percentage * total_votes : ℝ)
  let candidate_a_votes := (candidate_a_percentage * valid_votes : ℝ)
  candidate_a_votes = 380800 :=
by
  sorry

end candidate_a_valid_votes_l1272_127252


namespace find_x_l1272_127294

theorem find_x (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 6 * x^2 + 12 * x * y + 6 * y^2 = x^3 + 3 * x^2 * y + 3 * x * y^2) : x = 24 / 7 :=
by
  sorry

end find_x_l1272_127294


namespace range_of_a_for_f_zero_l1272_127260

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - 2 * x + a

theorem range_of_a_for_f_zero (a : ℝ) :
  (∃ x : ℝ, f x a = 0) ↔ a ≤ 2 * Real.log 2 - 2 :=
by
  sorry

end range_of_a_for_f_zero_l1272_127260


namespace second_ball_probability_l1272_127257

-- Definitions and conditions
def red_balls := 3
def white_balls := 2
def black_balls := 5
def total_balls := red_balls + white_balls + black_balls

def first_ball_white_condition : Prop := (white_balls / total_balls) = (2 / 10)
def second_ball_red_given_first_white (first_ball_white : Prop) : Prop :=
  (first_ball_white → (red_balls / (total_balls - 1)) = (1 / 3))

-- Mathematical equivalence proof problem statement in Lean
theorem second_ball_probability : 
  first_ball_white_condition ∧ second_ball_red_given_first_white first_ball_white_condition :=
by
  sorry

end second_ball_probability_l1272_127257


namespace find_chosen_number_l1272_127217

theorem find_chosen_number (x : ℤ) (h : 2 * x - 138 = 106) : x = 122 :=
by
  sorry

end find_chosen_number_l1272_127217


namespace purely_imaginary_m_no_m_in_fourth_quadrant_l1272_127221

def z (m : ℝ) : ℂ := ⟨m^2 - 8 * m + 15, m^2 - 5 * m⟩

theorem purely_imaginary_m :
  (∀ m : ℝ, z m = ⟨0, m^2 - 5 * m⟩ ↔ m = 3) :=
by
  sorry

theorem no_m_in_fourth_quadrant :
  ¬ ∃ m : ℝ, (m^2 - 8 * m + 15 > 0) ∧ (m^2 - 5 * m < 0) :=
by
  sorry

end purely_imaginary_m_no_m_in_fourth_quadrant_l1272_127221


namespace original_sticker_price_l1272_127295

-- Define the conditions in Lean
variables {x : ℝ} -- x is the original sticker price of the laptop

-- Definitions based on the problem conditions
def store_A_price (x : ℝ) : ℝ := 0.80 * x - 50
def store_B_price (x : ℝ) : ℝ := 0.70 * x
def heather_saves (x : ℝ) : Prop := store_B_price x - store_A_price x = 30

-- The theorem to prove
theorem original_sticker_price (x : ℝ) (h : heather_saves x) : x = 200 :=
by
  sorry

end original_sticker_price_l1272_127295


namespace limonia_largest_unachievable_l1272_127236

noncomputable def largest_unachievable_amount (n : ℕ) : ℕ :=
  12 * n^2 + 14 * n - 1

theorem limonia_largest_unachievable (n : ℕ) :
  ∀ k, ¬ ∃ a b c d : ℕ, 
    k = a * (6 * n + 1) + b * (6 * n + 4) + c * (6 * n + 7) + d * (6 * n + 10) 
    → k = largest_unachievable_amount n :=
sorry

end limonia_largest_unachievable_l1272_127236


namespace probability_four_collinear_dots_l1272_127253

noncomputable def probability_collinear_four_dots : ℚ :=
  let total_dots := 25
  let choose_4 := (total_dots.choose 4)
  let successful_outcomes := 60
  successful_outcomes / choose_4

theorem probability_four_collinear_dots :
  probability_collinear_four_dots = 12 / 2530 :=
by
  sorry

end probability_four_collinear_dots_l1272_127253


namespace range_of_m_l1272_127265

theorem range_of_m (x y m : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 / x + 1 / y = 1) (h4 : x + 2 * y > m^2 + 2 * m) :
  -4 < m ∧ m < 2 :=
sorry

end range_of_m_l1272_127265


namespace ratio_of_fruit_salads_l1272_127218

theorem ratio_of_fruit_salads 
  (salads_Alaya : ℕ) 
  (total_salads : ℕ) 
  (h1 : salads_Alaya = 200) 
  (h2 : total_salads = 600) : 
  (total_salads - salads_Alaya) / salads_Alaya = 2 :=
by 
  sorry

end ratio_of_fruit_salads_l1272_127218


namespace jacob_river_water_collection_l1272_127286

/-- Definitions: 
1. Capacity of the tank in milliliters
2. Daily water collected from the rain in milliliters
3. Number of days to fill the tank
4. To be proved: Daily water collected from the river in milliliters
-/
def tank_capacity_ml : Int := 50000
def daily_rain_ml : Int := 800
def days_to_fill : Int := 20
def daily_river_ml : Int := 1700

/-- Prove that the amount of water Jacob collects from the river every day equals 1700 milliliters.
-/
theorem jacob_river_water_collection (total_water: Int) 
  (rain_water: Int) (days: Int) (correct_river_water: Int) : 
  total_water = tank_capacity_ml → 
  rain_water = daily_rain_ml → 
  days = days_to_fill → 
  correct_river_water = daily_river_ml → 
  (total_water - rain_water * days) / days = correct_river_water := 
by 
  intros; 
  sorry

end jacob_river_water_collection_l1272_127286


namespace geometric_sequence_properties_l1272_127201

-- Define the first term and common ratio
def first_term : ℕ := 12
def common_ratio : ℚ := 1/2

-- Define the formula for the n-th term of the geometric sequence
def nth_term (a : ℕ) (r : ℚ) (n : ℕ) := a * r^(n-1)

-- The 8th term in the sequence
def term_8 := nth_term first_term common_ratio 8

-- Half of the 8th term
def half_term_8 := (1/2) * term_8

-- Prove that the 8th term is 3/32 and half of the 8th term is 3/64
theorem geometric_sequence_properties : 
  (term_8 = (3/32)) ∧ (half_term_8 = (3/64)) := 
by 
  sorry

end geometric_sequence_properties_l1272_127201


namespace tom_paid_1145_l1272_127216

-- Define the quantities
def quantity_apples : ℕ := 8
def rate_apples : ℕ := 70
def quantity_mangoes : ℕ := 9
def rate_mangoes : ℕ := 65

-- Calculate costs
def cost_apples : ℕ := quantity_apples * rate_apples
def cost_mangoes : ℕ := quantity_mangoes * rate_mangoes

-- Calculate the total amount paid
def total_amount_paid : ℕ := cost_apples + cost_mangoes

-- The theorem to prove
theorem tom_paid_1145 :
  total_amount_paid = 1145 :=
by sorry

end tom_paid_1145_l1272_127216


namespace bear_population_l1272_127241

theorem bear_population (black_bears white_bears brown_bears total_bears : ℕ) 
(h1 : black_bears = 60)
(h2 : white_bears = black_bears / 2)
(h3 : brown_bears = black_bears + 40) :
total_bears = black_bears + white_bears + brown_bears :=
sorry

end bear_population_l1272_127241


namespace instantaneous_velocity_at_t_2_l1272_127280

def y (t : ℝ) : ℝ := 3 * t^2 + 4

theorem instantaneous_velocity_at_t_2 :
  deriv y 2 = 12 :=
by
  sorry

end instantaneous_velocity_at_t_2_l1272_127280


namespace part_a_part_b_l1272_127281

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def valid_permutation (P : Fin 16 → ℕ) : Prop :=
  (∀ i : Fin 15, is_perfect_square (P i + P (i + 1))) ∧
  ∀ i, P i ∈ (Finset.range 16).image (λ x => x + 1)

def valid_cyclic_permutation (C : Fin 16 → ℕ) : Prop :=
  (∀ i : Fin 15, is_perfect_square (C i + C (i + 1))) ∧
  is_perfect_square (C 15 + C 0) ∧
  ∀ i, C i ∈ (Finset.range 16).image (λ x => x + 1)

theorem part_a :
  ∃ P : Fin 16 → ℕ, valid_permutation P := sorry

theorem part_b :
  ¬ ∃ C : Fin 16 → ℕ, valid_cyclic_permutation C := sorry

end part_a_part_b_l1272_127281


namespace solve_equation_l1272_127248

theorem solve_equation (x : ℝ) : 
  (9 - x - 2 * (31 - x) = 27) → (x = 80) :=
by
  sorry

end solve_equation_l1272_127248


namespace linear_equation_variables_l1272_127232

theorem linear_equation_variables (m n : ℤ) (h1 : 3 * m - 2 * n = 1) (h2 : n - m = 1) : m = 0 ∧ n = 1 :=
by {
  sorry
}

end linear_equation_variables_l1272_127232


namespace problem_statement_l1272_127203

noncomputable def omega : ℂ := sorry -- Definition placeholder for a specific nonreal root of x^4 = 1. 

theorem problem_statement (h1 : omega ^ 4 = 1) (h2 : omega ^ 2 = -1) : 
  (1 - omega + omega ^ 3) ^ 4 + (1 + omega - omega ^ 3) ^ 4 = -14 := 
sorry

end problem_statement_l1272_127203


namespace divisible_by_6_l1272_127297

theorem divisible_by_6 (n : ℤ) (h1 : n % 3 = 0) (h2 : n % 2 = 0) : n % 6 = 0 :=
sorry

end divisible_by_6_l1272_127297


namespace ram_birthday_l1272_127268

theorem ram_birthday
    (L : ℕ) (L1 : ℕ) (Llast : ℕ) (d : ℕ) (languages_learned_per_day : ℕ) (days_in_month : ℕ) :
    (L = 1000) →
    (L1 = 820) →
    (Llast = 1100) →
    (days_in_month = 28 ∨ days_in_month = 29 ∨ days_in_month = 30 ∨ days_in_month = 31) →
    (d = days_in_month - 1) →
    (languages_learned_per_day = (Llast - L1) / d) →
    ∃ n : ℕ, n = 19 :=
by
  intros hL hL1 hLlast hDays hm_d hLearned
  existsi 19
  sorry

end ram_birthday_l1272_127268


namespace intersection_M_N_l1272_127282

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | -1 < x ∧ x < 1}

theorem intersection_M_N : (M ∩ N) = {x | 0 ≤ x ∧ x < 1} :=
by {
  sorry
}

end intersection_M_N_l1272_127282


namespace car_catch_up_distance_l1272_127208

-- Define the problem: Prove that Car A catches Car B at the specified location, given the conditions

theorem car_catch_up_distance
  (distance : ℝ := 300)
  (v_A v_B t_A : ℝ)
  (start_A_late : ℝ := 1) -- Car A leaves 1 hour later
  (arrive_A_early : ℝ := 1) -- Car A arrives 1 hour earlier
  (t : ℝ := 2) -- Time when Car A catches up with Car B
  (dist_eq_A : distance = v_A * t_A)
  (dist_eq_B : distance = v_B * (t_A + 2))
  (catch_up_time_eq : v_A * t = v_B * (t + 1)):
  (distance - v_A * t) = 150 := sorry

end car_catch_up_distance_l1272_127208


namespace ball_probability_l1272_127271

theorem ball_probability (n : ℕ) (h : (n : ℚ) / (n + 2) = 1 / 3) : n = 1 :=
sorry

end ball_probability_l1272_127271


namespace trig_proof_l1272_127237

variable {α a : ℝ}

theorem trig_proof (h₁ : (∃ a : ℝ, a < 0 ∧ (4 * a, -3 * a) = (4 * a, -3 * a)))
                    (h₂ : a < 0) :
  2 * Real.sin α + Real.cos α = 2 / 5 := 
sorry

end trig_proof_l1272_127237


namespace clara_climbs_stone_blocks_l1272_127212

-- Define the number of steps per level
def steps_per_level : Nat := 8

-- Define the number of blocks per step
def blocks_per_step : Nat := 3

-- Define the number of levels in the tower
def levels : Nat := 4

-- Define a function to compute the total number of blocks given the constants
def total_blocks (steps_per_level blocks_per_step levels : Nat) : Nat :=
  steps_per_level * blocks_per_step * levels

-- Statement of the theorem
theorem clara_climbs_stone_blocks :
  total_blocks steps_per_level blocks_per_step levels = 96 :=
by
  -- Lean requires 'sorry' as a placeholder for the proof.
  sorry

end clara_climbs_stone_blocks_l1272_127212


namespace incorrect_inequality_l1272_127266

-- Given definitions
variables {a b : ℝ}
axiom h : a < b ∧ b < 0

-- Equivalent theorem statement
theorem incorrect_inequality (ha : a < b) (hb : b < 0) : (1 / (a - b)) < (1 / a) := 
sorry

end incorrect_inequality_l1272_127266


namespace ellipse_find_m_l1272_127207

theorem ellipse_find_m (a b m e : ℝ) 
  (h1 : a^2 = 4) 
  (h2 : b^2 = m)
  (h3 : e = 1/2) :
  m = 3 := 
by
  sorry

end ellipse_find_m_l1272_127207


namespace calories_in_250_grams_is_106_l1272_127274

noncomputable def total_calories_apple : ℝ := 150 * (46 / 100)
noncomputable def total_calories_orange : ℝ := 50 * (45 / 100)
noncomputable def total_calories_carrot : ℝ := 300 * (40 / 100)
noncomputable def total_calories_mix : ℝ := total_calories_apple + total_calories_orange + total_calories_carrot
noncomputable def total_weight_mix : ℝ := 150 + 50 + 300
noncomputable def caloric_density : ℝ := total_calories_mix / total_weight_mix
noncomputable def calories_in_250_grams : ℝ := 250 * caloric_density

theorem calories_in_250_grams_is_106 : calories_in_250_grams = 106 :=
by
  sorry

end calories_in_250_grams_is_106_l1272_127274


namespace lowest_sale_price_percentage_l1272_127289

theorem lowest_sale_price_percentage :
  ∃ (p : ℝ) (h1 : 30 / 100 * p ≤ 70 / 100 * p) (h2 : p = 80),
  (p - 70 / 100 * p - 20 / 100 * p) / p * 100 = 10 := by
sorry

end lowest_sale_price_percentage_l1272_127289


namespace expected_expenditure_l1272_127213

-- Define the parameters and conditions
def b : ℝ := 0.8
def a : ℝ := 2
def e_condition (e : ℝ) : Prop := |e| < 0.5
def revenue : ℝ := 10

-- Define the expenditure function based on the conditions
def expenditure (x e : ℝ) : ℝ := b * x + a + e

-- The expected expenditure should not exceed 10.5
theorem expected_expenditure (e : ℝ) (h : e_condition e) : expenditure revenue e ≤ 10.5 :=
sorry

end expected_expenditure_l1272_127213


namespace probability_one_project_not_selected_l1272_127261

noncomputable def calc_probability : ℚ :=
  let n := 4 ^ 4
  let m := Nat.choose 4 2 * Nat.factorial 4
  let p := m / n
  p

theorem probability_one_project_not_selected :
  calc_probability = 9 / 16 :=
by
  sorry

end probability_one_project_not_selected_l1272_127261


namespace SandySpentTotal_l1272_127251

theorem SandySpentTotal :
  let shorts := 13.99
  let shirt := 12.14
  let jacket := 7.43
  shorts + shirt + jacket = 33.56 := by
  sorry

end SandySpentTotal_l1272_127251


namespace integer_solutions_count_l1272_127283

theorem integer_solutions_count (x : ℤ) : 
  (x^2 - 3 * x + 2)^2 - 3 * (x^2 - 3 * x) - 4 = 0 ↔ 0 = 0 :=
by sorry

end integer_solutions_count_l1272_127283


namespace sum_of_solutions_eq_zero_l1272_127202

theorem sum_of_solutions_eq_zero :
  let f (x : ℝ) := 2^|x| + 4 * |x|
  (∀ x : ℝ, f x = 20) →
  (∃ x₁ x₂ : ℝ, f x₁ = 20 ∧ f x₂ = 20 ∧ x₁ + x₂ = 0) :=
sorry

end sum_of_solutions_eq_zero_l1272_127202


namespace range_sin_cos_two_x_is_minus2_to_9_over_8_l1272_127296

noncomputable def range_of_function : Set ℝ :=
  { y : ℝ | ∃ x : ℝ, y = Real.sin x + Real.cos (2 * x) }

theorem range_sin_cos_two_x_is_minus2_to_9_over_8 :
  range_of_function = Set.Icc (-2) (9 / 8) := 
by
  sorry

end range_sin_cos_two_x_is_minus2_to_9_over_8_l1272_127296


namespace scientific_notation_216000_l1272_127235

theorem scientific_notation_216000 : 216000 = 2.16 * 10^5 :=
by
  -- proof will be provided here
  sorry

end scientific_notation_216000_l1272_127235


namespace proposition_correctness_l1272_127273

theorem proposition_correctness :
  (∀ a b : ℝ, a < b ∧ b < 0 → ¬ (1 / a < 1 / b)) ∧
  (∀ a b : ℝ, a > 0 ∧ b > 0 → (a + b) / 2 ≥ Real.sqrt (a * b) ∧ Real.sqrt (a * b) ≥ a * b / (a + b)) ∧
  (∀ a b : ℝ, a < b ∧ b < 0 → a^2 > a * b ∧ a * b > b^2) ∧
  (Real.log 9 * Real.log 11 < 1) ∧
  (∀ a b : ℝ, a > b ∧ 1 / a > 1 / b → a > 0 ∧ b < 0) ∧
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / x + 1 / y = 1 → ¬(x + 2 * y = 6)) :=
sorry

end proposition_correctness_l1272_127273
