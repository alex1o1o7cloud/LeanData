import Mathlib

namespace NUMINAMATH_GPT_plain_b_area_l371_37178

theorem plain_b_area : 
  ∃ x : ℕ, (x + (x - 50) = 350) ∧ x = 200 :=
by
  sorry

end NUMINAMATH_GPT_plain_b_area_l371_37178


namespace NUMINAMATH_GPT_min_spend_for_free_delivery_l371_37157

theorem min_spend_for_free_delivery : 
  let chicken_price := 1.5 * 6.00
  let lettuce_price := 3.00
  let tomato_price := 2.50
  let sweet_potato_price := 4 * 0.75
  let broccoli_price := 2 * 2.00
  let brussel_sprouts_price := 2.50
  let current_total := chicken_price + lettuce_price + tomato_price + sweet_potato_price + broccoli_price + brussel_sprouts_price
  let additional_needed := 11.00 
  let minimum_spend := current_total + additional_needed
  minimum_spend = 35.00 :=
by
  sorry

end NUMINAMATH_GPT_min_spend_for_free_delivery_l371_37157


namespace NUMINAMATH_GPT_time_to_empty_is_109_89_hours_l371_37191

noncomputable def calculate_time_to_empty_due_to_leak : ℝ :=
  let R := 1 / 10 -- filling rate in tank/hour
  let Reffective := 1 / 11 -- effective filling rate in tank/hour
  let L := R - Reffective -- leak rate in tank/hour
  1 / L -- time to empty in hours

theorem time_to_empty_is_109_89_hours : calculate_time_to_empty_due_to_leak = 109.89 :=
by
  rw [calculate_time_to_empty_due_to_leak]
  sorry -- Proof steps can be filled in later

end NUMINAMATH_GPT_time_to_empty_is_109_89_hours_l371_37191


namespace NUMINAMATH_GPT_number_is_more_than_sum_l371_37119

theorem number_is_more_than_sum : 20.2 + 33.8 - 5.1 = 48.9 :=
by
  sorry

end NUMINAMATH_GPT_number_is_more_than_sum_l371_37119


namespace NUMINAMATH_GPT_swimming_pool_surface_area_l371_37177

def length : ℝ := 20
def width : ℝ := 15

theorem swimming_pool_surface_area : length * width = 300 := 
by
  -- The mathematical proof would go here; we'll skip it with "sorry" per instructions.
  sorry

end NUMINAMATH_GPT_swimming_pool_surface_area_l371_37177


namespace NUMINAMATH_GPT_product_of_points_l371_37146

def f (n : ℕ) : ℕ :=
  if n % 6 = 0 then 6
  else if n % 3 = 0 then 3
  else if n % 2 = 0 then 2
  else 1

def allie_rolls := [5, 6, 1, 2, 3]
def betty_rolls := [6, 1, 1, 2, 3]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.foldl (fun acc n => acc + f n) 0

theorem product_of_points :
  total_points allie_rolls * total_points betty_rolls = 169 :=
by
  sorry

end NUMINAMATH_GPT_product_of_points_l371_37146


namespace NUMINAMATH_GPT_jane_mean_after_extra_credit_l371_37173

-- Define Jane's original scores
def original_scores : List ℤ := [82, 90, 88, 95, 91]

-- Define the extra credit points
def extra_credit : ℤ := 2

-- Define the mean calculation after extra credit
def mean_after_extra_credit (scores : List ℤ) (extra : ℤ) : ℚ :=
  let total_sum := scores.sum + (scores.length * extra)
  total_sum / scores.length

theorem jane_mean_after_extra_credit :
  mean_after_extra_credit original_scores extra_credit = 91.2 := by
  sorry

end NUMINAMATH_GPT_jane_mean_after_extra_credit_l371_37173


namespace NUMINAMATH_GPT_gcd_153_119_eq_17_l371_37144

theorem gcd_153_119_eq_17 : Nat.gcd 153 119 = 17 := by
  sorry

end NUMINAMATH_GPT_gcd_153_119_eq_17_l371_37144


namespace NUMINAMATH_GPT_bees_second_day_l371_37102

-- Define the number of bees on the first day
def bees_on_first_day : ℕ := 144 

-- Define the multiplier for the second day
def multiplier : ℕ := 3

-- Define the number of bees on the second day
def bees_on_second_day : ℕ := bees_on_first_day * multiplier

-- Theorem stating the number of bees seen on the second day
theorem bees_second_day : bees_on_second_day = 432 := by
  -- Proof is pending.
  sorry

end NUMINAMATH_GPT_bees_second_day_l371_37102


namespace NUMINAMATH_GPT_largest_constant_inequality_l371_37141

theorem largest_constant_inequality (C : ℝ) (h : ∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z)) : 
  C ≤ 2 / Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_largest_constant_inequality_l371_37141


namespace NUMINAMATH_GPT_decryption_correct_l371_37169

theorem decryption_correct (a b : ℤ) (h1 : a - 2 * b = 1) (h2 : 2 * a + b = 7) : a = 3 ∧ b = 1 :=
by
  sorry

end NUMINAMATH_GPT_decryption_correct_l371_37169


namespace NUMINAMATH_GPT_maximal_possible_degree_difference_l371_37109

theorem maximal_possible_degree_difference (n_vertices : ℕ) (n_edges : ℕ) (disjoint_edge_pairs : ℕ) 
    (h1 : n_vertices = 30) (h2 : n_edges = 105) (h3 : disjoint_edge_pairs = 4822) : 
    ∃ (max_diff : ℕ), max_diff = 22 :=
by
  sorry

end NUMINAMATH_GPT_maximal_possible_degree_difference_l371_37109


namespace NUMINAMATH_GPT_range_of_a_l371_37107

variable {x a : ℝ}

theorem range_of_a (h1 : x > 1) (h2 : a ≤ x + 1 / (x - 1)) : a ≤ 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l371_37107


namespace NUMINAMATH_GPT_julia_money_left_l371_37143

def initial_amount : ℕ := 40

def amount_spent_on_game (initial : ℕ) : ℕ := initial / 2

def amount_left_after_game (initial : ℕ) (spent_game : ℕ) : ℕ := initial - spent_game

def amount_spent_on_in_game (left_after_game : ℕ) : ℕ := left_after_game / 4

def final_amount (left_after_game : ℕ) (spent_in_game : ℕ) : ℕ := left_after_game - spent_in_game

theorem julia_money_left (initial : ℕ) 
  (h_init : initial = initial_amount)
  (spent_game : ℕ)
  (h_spent_game : spent_game = amount_spent_on_game initial)
  (left_after_game : ℕ)
  (h_left_after_game : left_after_game = amount_left_after_game initial spent_game)
  (spent_in_game : ℕ)
  (h_spent_in_game : spent_in_game = amount_spent_on_in_game left_after_game)
  : final_amount left_after_game spent_in_game = 15 := by 
  sorry

end NUMINAMATH_GPT_julia_money_left_l371_37143


namespace NUMINAMATH_GPT_tan_45_deg_l371_37147

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end NUMINAMATH_GPT_tan_45_deg_l371_37147


namespace NUMINAMATH_GPT_one_cow_one_bag_l371_37195

-- Define parameters
def cows : ℕ := 26
def bags : ℕ := 26
def days_for_all_cows : ℕ := 26

-- Theorem to prove the number of days for one cow to eat one bag of husk
theorem one_cow_one_bag (cows bags days_for_all_cows : ℕ) (h : cows = bags) (h2 : days_for_all_cows = 26) : days_for_one_cow_one_bag = 26 :=
by {
    sorry -- Proof to be filled in
}

end NUMINAMATH_GPT_one_cow_one_bag_l371_37195


namespace NUMINAMATH_GPT_find_the_number_l371_37170

theorem find_the_number :
  ∃ x : ℤ, 65 + (x * 12) / (180 / 3) = 66 ∧ x = 5 :=
by
  existsi (5 : ℤ)
  sorry

end NUMINAMATH_GPT_find_the_number_l371_37170


namespace NUMINAMATH_GPT_number_minus_six_l371_37187

variable (x : ℤ)

theorem number_minus_six
  (h : x / 5 = 2) : x - 6 = 4 := 
sorry

end NUMINAMATH_GPT_number_minus_six_l371_37187


namespace NUMINAMATH_GPT_find_a4_l371_37152

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

axiom hyp1 : is_arithmetic_sequence a d
axiom hyp2 : a 5 = 9
axiom hyp3 : a 7 + a 8 = 28

-- Goal
theorem find_a4 : a 4 = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_a4_l371_37152


namespace NUMINAMATH_GPT_quadruple_solution_l371_37185

theorem quadruple_solution (x y z w : ℝ) (h1: x + y + z + w = 0) (h2: x^7 + y^7 + z^7 + w^7 = 0) :
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ w = 0) ∨ (x = -y ∧ z = -w) ∨ (x = -z ∧ y = -w) ∨ (x = -w ∧ y = -z) :=
by
  sorry

end NUMINAMATH_GPT_quadruple_solution_l371_37185


namespace NUMINAMATH_GPT_animals_on_farm_l371_37124

theorem animals_on_farm (cows : ℕ) (sheep : ℕ) (pigs : ℕ) 
  (h1 : cows = 12) 
  (h2 : sheep = 2 * cows) 
  (h3 : pigs = 3 * sheep) : 
  cows + sheep + pigs = 108 := 
by
  sorry

end NUMINAMATH_GPT_animals_on_farm_l371_37124


namespace NUMINAMATH_GPT_range_of_a_for_function_min_max_l371_37180

theorem range_of_a_for_function_min_max 
  (a : ℝ) 
  (h_min : ∀ x ∈ [-1, 1], x = -1 → x^2 + a * x + 3 ≤ y) 
  (h_max : ∀ x ∈ [-1, 1], x = 1 → x^2 + a * x + 3 ≥ y) : 
  2 ≤ a := 
sorry

end NUMINAMATH_GPT_range_of_a_for_function_min_max_l371_37180


namespace NUMINAMATH_GPT_proof_statements_correct_l371_37163

variable (candidates : Nat) (sample_size : Nat)

def is_sampling_survey (survey_type : String) : Prop :=
  survey_type = "sampling"

def is_population (pop_size sample_size : Nat) : Prop :=
  (pop_size = 60000) ∧ (sample_size = 1000)

def is_sample (sample_size pop_size : Nat) : Prop :=
  sample_size < pop_size

def sample_size_correct (sample_size : Nat) : Prop :=
  sample_size = 1000

theorem proof_statements_correct :
  ∀ (survey_type : String) (pop_size sample_size : Nat),
  is_sampling_survey survey_type →
  is_population pop_size sample_size →
  is_sample sample_size pop_size →
  sample_size_correct sample_size →
  survey_type = "sampling" ∧
  pop_size = 60000 ∧
  sample_size = 1000 :=
by
  intros survey_type pop_size sample_size hs hp hsamp hsiz
  sorry

end NUMINAMATH_GPT_proof_statements_correct_l371_37163


namespace NUMINAMATH_GPT_knights_win_35_l371_37155

noncomputable def Sharks : ℕ := sorry
noncomputable def Falcons : ℕ := sorry
noncomputable def Knights : ℕ := 35
noncomputable def Wolves : ℕ := sorry
noncomputable def Royals : ℕ := sorry

-- Conditions
axiom h1 : Sharks > Falcons
axiom h2 : Wolves > 25
axiom h3 : Wolves < Knights ∧ Knights < Royals

-- Prove: Knights won 35 games
theorem knights_win_35 : Knights = 35 := 
by sorry

end NUMINAMATH_GPT_knights_win_35_l371_37155


namespace NUMINAMATH_GPT_domain_of_f_zeros_of_f_l371_37196

def log_a (a : ℝ) (x : ℝ) : ℝ := sorry -- Assume definition of logarithm base 'a'.

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log_a a (2 - x)

theorem domain_of_f (a : ℝ) : ∀ x : ℝ, 2 - x > 0 ↔ x < 2 :=
by
  sorry

theorem zeros_of_f (a : ℝ) : f a 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_zeros_of_f_l371_37196


namespace NUMINAMATH_GPT_probability_same_color_boxes_l371_37182

def num_neckties := 6
def num_shirts := 5
def num_hats := 4
def num_socks := 3

def num_common_colors := 3

def total_combinations : ℕ := num_neckties * num_shirts * num_hats * num_socks

def same_color_combinations : ℕ := num_common_colors

def same_color_probability : ℚ :=
  same_color_combinations / total_combinations

theorem probability_same_color_boxes :
  same_color_probability = 1 / 120 :=
  by
    -- Proof would go here
    sorry

end NUMINAMATH_GPT_probability_same_color_boxes_l371_37182


namespace NUMINAMATH_GPT_fraction_zero_imp_x_eq_two_l371_37156
open Nat Real

theorem fraction_zero_imp_x_eq_two (x : ℝ) (h: (2 - abs x) / (x + 2) = 0) : x = 2 :=
by
  have h1 : 2 - abs x = 0 := sorry
  have h2 : x + 2 ≠ 0 := sorry
  sorry

end NUMINAMATH_GPT_fraction_zero_imp_x_eq_two_l371_37156


namespace NUMINAMATH_GPT_dividend_rate_of_stock_l371_37127

variable (MarketPrice : ℝ) (YieldPercent : ℝ) (DividendPercent : ℝ)
variable (NominalValue : ℝ) (AnnualDividend : ℝ)

def stock_dividend_rate_condition (YieldPercent MarketPrice NominalValue DividendPercent : ℝ) 
  (AnnualDividend : ℝ) : Prop :=
  YieldPercent = 20 ∧ MarketPrice = 125 ∧ DividendPercent = 0.25 ∧ NominalValue = 100 ∧
  AnnualDividend = (YieldPercent / 100) * MarketPrice

theorem dividend_rate_of_stock :
  stock_dividend_rate_condition 20 125 100 0.25 25 → (DividendPercent * NominalValue) = 25 :=
by 
  sorry

end NUMINAMATH_GPT_dividend_rate_of_stock_l371_37127


namespace NUMINAMATH_GPT_least_positive_x_l371_37145

variable (a b : ℝ)

noncomputable def tan_inv (x : ℝ) : ℝ := Real.arctan x

theorem least_positive_x (x k : ℝ) 
  (h1 : Real.tan x = a / b)
  (h2 : Real.tan (2 * x) = b / (a + b))
  (h3 : Real.tan (3 * x) = (a - b) / (a + b))
  (h4 : x = tan_inv k)
  : k = 13 / 9 := sorry

end NUMINAMATH_GPT_least_positive_x_l371_37145


namespace NUMINAMATH_GPT_apples_used_l371_37112

theorem apples_used (initial_apples remaining_apples : ℕ) (h_initial : initial_apples = 40) (h_remaining : remaining_apples = 39) : initial_apples - remaining_apples = 1 := 
by
  sorry

end NUMINAMATH_GPT_apples_used_l371_37112


namespace NUMINAMATH_GPT_trig_problem_l371_37172

theorem trig_problem 
  (α : ℝ) 
  (h1 : Real.cos α = -1/2) 
  (h2 : 180 * (Real.pi / 180) < α ∧ α < 270 * (Real.pi / 180)) : 
  α = 240 * (Real.pi / 180) :=
sorry

end NUMINAMATH_GPT_trig_problem_l371_37172


namespace NUMINAMATH_GPT_scientific_notation_of_634000000_l371_37194

theorem scientific_notation_of_634000000 :
  634000000 = 6.34 * 10 ^ 8 := 
sorry

end NUMINAMATH_GPT_scientific_notation_of_634000000_l371_37194


namespace NUMINAMATH_GPT_sticker_ratio_l371_37159

variable (Dan Tom Bob : ℕ)

theorem sticker_ratio 
  (h1 : Dan = 2 * Tom) 
  (h2 : Tom = Bob) 
  (h3 : Bob = 12) 
  (h4 : Dan = 72) : 
  Tom = Bob :=
by
  sorry

end NUMINAMATH_GPT_sticker_ratio_l371_37159


namespace NUMINAMATH_GPT_find_N_l371_37126

theorem find_N (x y : ℝ) (h1 : 2 * x + y = 6) (h2 : x + 2 * y = 5) :
  (x + y) / 3 = 1.222222222222222 := 
by
  -- We state the conditions.
  -- Lean will check whether these assumptions are consistent 
  sorry

end NUMINAMATH_GPT_find_N_l371_37126


namespace NUMINAMATH_GPT_average_speed_monday_to_wednesday_l371_37150

theorem average_speed_monday_to_wednesday :
  ∃ x : ℝ, (∀ (total_hours total_distance thursday_friday_distance : ℝ),
    total_hours = 2 * 5 ∧
    thursday_friday_distance = 9 * 2 * 2 ∧
    total_distance = 108 ∧
    total_distance - thursday_friday_distance = x * (2 * 3))
    → x = 12 :=
sorry

end NUMINAMATH_GPT_average_speed_monday_to_wednesday_l371_37150


namespace NUMINAMATH_GPT_equilibrium_constant_l371_37171

theorem equilibrium_constant (C_NO2 C_O2 C_NO : ℝ) (h_NO2 : C_NO2 = 0.4) (h_O2 : C_O2 = 0.3) (h_NO : C_NO = 0.2) :
  (C_NO2^2 / (C_O2 * C_NO^2)) = 13.3 := by
  rw [h_NO2, h_O2, h_NO]
  sorry

end NUMINAMATH_GPT_equilibrium_constant_l371_37171


namespace NUMINAMATH_GPT_percent_of_150_is_60_l371_37165

def percent_is_correct (Part Whole : ℝ) : Prop :=
  (Part / Whole) * 100 = 250

theorem percent_of_150_is_60 :
  percent_is_correct 150 60 :=
by
  sorry

end NUMINAMATH_GPT_percent_of_150_is_60_l371_37165


namespace NUMINAMATH_GPT_decreasing_power_function_l371_37101

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x ^ k

theorem decreasing_power_function (k : ℝ) : 
  (∀ x : ℝ, 0 < x → (f k x) ≤ 0) ↔ k < 0 ∧ k ≠ 0 := sorry

end NUMINAMATH_GPT_decreasing_power_function_l371_37101


namespace NUMINAMATH_GPT_original_speed_of_Person_A_l371_37111

variable (v_A v_B : ℝ)

-- Define the conditions
def condition1 : Prop := v_B = 2 * v_A
def condition2 : Prop := v_A + 10 = 4 * (v_B - 5)

-- Define the theorem to prove
theorem original_speed_of_Person_A (h1 : condition1 v_A v_B) (h2 : condition2 v_A v_B) : v_A = 18 := 
by
  sorry

end NUMINAMATH_GPT_original_speed_of_Person_A_l371_37111


namespace NUMINAMATH_GPT_mason_courses_not_finished_l371_37162

-- Each necessary condition is listed as a definition.
def coursesPerWall := 6
def bricksPerCourse := 10
def numOfWalls := 4
def totalBricksUsed := 220

-- Creating an entity to store the problem and prove it.
theorem mason_courses_not_finished : 
  (numOfWalls * coursesPerWall * bricksPerCourse - totalBricksUsed) / bricksPerCourse = 2 := 
by
  sorry

end NUMINAMATH_GPT_mason_courses_not_finished_l371_37162


namespace NUMINAMATH_GPT_find_q_l371_37168

theorem find_q (p q : ℚ) (h1 : 5 * p + 6 * q = 20) (h2 : 6 * p + 5 * q = 29) : q = -25 / 11 :=
by
  sorry

end NUMINAMATH_GPT_find_q_l371_37168


namespace NUMINAMATH_GPT_puppies_per_cage_l371_37190

-- Conditions
variables (total_puppies sold_puppies cages initial_puppies per_cage : ℕ)
variables (h_total : total_puppies = 13)
variables (h_sold : sold_puppies = 7)
variables (h_cages : cages = 3)
variables (h_equal_cages : total_puppies - sold_puppies = cages * per_cage)

-- Question
theorem puppies_per_cage :
  per_cage = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_puppies_per_cage_l371_37190


namespace NUMINAMATH_GPT_fish_left_in_sea_l371_37117

theorem fish_left_in_sea : 
  let westward_initial := 1800
  let eastward_initial := 3200
  let north_initial := 500
  let eastward_caught := (2 / 5) * eastward_initial
  let westward_caught := (3 / 4) * westward_initial
  let eastward_left := eastward_initial - eastward_caught
  let westward_left := westward_initial - westward_caught
  let north_left := north_initial
  eastward_left + westward_left + north_left = 2870 := 
by 
  sorry

end NUMINAMATH_GPT_fish_left_in_sea_l371_37117


namespace NUMINAMATH_GPT_students_before_Yoongi_l371_37121

theorem students_before_Yoongi (total_students : ℕ) (students_after_Yoongi : ℕ) 
  (condition1 : total_students = 20) (condition2 : students_after_Yoongi = 11) :
  total_students - students_after_Yoongi - 1 = 8 :=
by 
  sorry

end NUMINAMATH_GPT_students_before_Yoongi_l371_37121


namespace NUMINAMATH_GPT_baskets_of_peaches_l371_37134

theorem baskets_of_peaches (n : ℕ) :
  (∀ x : ℕ, (n * 2 = 14) → (n = x)) := by
  sorry

end NUMINAMATH_GPT_baskets_of_peaches_l371_37134


namespace NUMINAMATH_GPT_intersection_A_B_l371_37103

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := { x | 0 < 2 - x ∧ 2 - x < 3 }

theorem intersection_A_B :
  A ∩ B = {0, 1} := sorry

end NUMINAMATH_GPT_intersection_A_B_l371_37103


namespace NUMINAMATH_GPT_distance_diff_is_0_point3_l371_37164

def john_walk_distance : ℝ := 0.7
def nina_walk_distance : ℝ := 0.4
def distance_difference_john_nina : ℝ := john_walk_distance - nina_walk_distance

theorem distance_diff_is_0_point3 : distance_difference_john_nina = 0.3 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_distance_diff_is_0_point3_l371_37164


namespace NUMINAMATH_GPT_area_conversion_correct_l371_37149

-- Define the legs of the right triangle
def leg1 : ℕ := 60
def leg2 : ℕ := 80

-- Define the conversion factor
def square_feet_in_square_yard : ℕ := 9

-- Calculate the area of the triangle in square feet
def area_in_square_feet : ℕ := (leg1 * leg2) / 2

-- Calculate the area of the triangle in square yards
def area_in_square_yards : ℚ := area_in_square_feet / square_feet_in_square_yard

-- The theorem stating the problem
theorem area_conversion_correct : area_in_square_yards = 266 + 2 / 3 := by
  sorry

end NUMINAMATH_GPT_area_conversion_correct_l371_37149


namespace NUMINAMATH_GPT_ratio_e_a_l371_37175

theorem ratio_e_a (a b c d e : ℚ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 3)
  (h4 : d / e = 1 / 4) :
  e / a = 8 / 15 := 
by
  sorry

end NUMINAMATH_GPT_ratio_e_a_l371_37175


namespace NUMINAMATH_GPT_total_movies_shown_l371_37199

-- Define the conditions of the problem
def screens := 6
def open_hours := 8
def movie_duration := 2

-- Define the statement to prove
theorem total_movies_shown : screens * (open_hours / movie_duration) = 24 := 
by
  sorry

end NUMINAMATH_GPT_total_movies_shown_l371_37199


namespace NUMINAMATH_GPT_jimmy_eats_7_cookies_l371_37198

def cookies_and_calories (c: ℕ) : Prop :=
  50 * c + 150 = 500

theorem jimmy_eats_7_cookies : cookies_and_calories 7 :=
by {
  -- This would be where the proof steps go, but we replace it with:
  sorry
}

end NUMINAMATH_GPT_jimmy_eats_7_cookies_l371_37198


namespace NUMINAMATH_GPT_find_x_l371_37132

theorem find_x (x : ℝ) (h : 61 + 5 * 12 / (x / 3) = 62) : x = 180 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l371_37132


namespace NUMINAMATH_GPT_skipping_ropes_l371_37113

theorem skipping_ropes (length1 length2 : ℕ) (h1 : length1 = 18) (h2 : length2 = 24) :
  ∃ (max_length : ℕ) (num_ropes : ℕ),
    max_length = Nat.gcd length1 length2 ∧
    max_length = 6 ∧
    num_ropes = length1 / max_length + length2 / max_length ∧
    num_ropes = 7 :=
by
  have max_length : ℕ := Nat.gcd length1 length2
  have num_ropes : ℕ := length1 / max_length + length2 / max_length
  use max_length, num_ropes
  sorry

end NUMINAMATH_GPT_skipping_ropes_l371_37113


namespace NUMINAMATH_GPT_total_food_items_donated_l371_37183

def FosterFarmsDonation : ℕ := 45
def AmericanSummitsDonation : ℕ := 2 * FosterFarmsDonation
def HormelDonation : ℕ := 3 * FosterFarmsDonation
def BoudinButchersDonation : ℕ := HormelDonation / 3
def DelMonteFoodsDonation : ℕ := AmericanSummitsDonation - 30

theorem total_food_items_donated :
  FosterFarmsDonation + AmericanSummitsDonation + HormelDonation + BoudinButchersDonation + DelMonteFoodsDonation = 375 :=
by
  sorry

end NUMINAMATH_GPT_total_food_items_donated_l371_37183


namespace NUMINAMATH_GPT_complex_expression_simplification_l371_37118

-- Given: i is the imaginary unit
def i := Complex.I

-- Prove that the expression simplifies to -1
theorem complex_expression_simplification : (i^3 * (i + 1)) / (i - 1) = -1 := by
  -- We are skipping the proof and adding sorry for now
  sorry

end NUMINAMATH_GPT_complex_expression_simplification_l371_37118


namespace NUMINAMATH_GPT_range_of_a_l371_37106

variable (a : ℝ)

def discriminant (a : ℝ) : ℝ := 4 * a ^ 2 - 12

theorem range_of_a
  (h : discriminant a > 0) :
  a < -Real.sqrt 3 ∨ a > Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l371_37106


namespace NUMINAMATH_GPT_arithmetic_series_sum_l371_37120

theorem arithmetic_series_sum :
  let a1 : ℚ := 22
  let d : ℚ := 3 / 7
  let an : ℚ := 73
  let n := (an - a1) / d + 1
  let S := n * (a1 + an) / 2
  S = 5700 := by
  sorry

end NUMINAMATH_GPT_arithmetic_series_sum_l371_37120


namespace NUMINAMATH_GPT_perpendicular_bisector_eq_l371_37186

theorem perpendicular_bisector_eq (A B: (ℝ × ℝ)) (hA: A = (1, 3)) (hB: B = (-5, 1)) :
  ∃ m c, (m = -3) ∧ (c = 4) ∧ (∀ x y, y = m * x + c ↔ 3 * x + y + 4 = 0) := 
by
  sorry

end NUMINAMATH_GPT_perpendicular_bisector_eq_l371_37186


namespace NUMINAMATH_GPT_tank_capacity_l371_37104

variable (C : ℝ)

theorem tank_capacity (h : (3/4) * C + 9 = (7/8) * C) : C = 72 :=
by
  sorry

end NUMINAMATH_GPT_tank_capacity_l371_37104


namespace NUMINAMATH_GPT_stream_current_rate_proof_l371_37189

noncomputable def stream_current_rate (c : ℝ) : Prop :=
  ∃ (c : ℝ), (6 / (8 - c) + 6 / (8 + c) = 2) ∧ c = 4

theorem stream_current_rate_proof : stream_current_rate 4 :=
by {
  -- Proof to be provided here.
  sorry
}

end NUMINAMATH_GPT_stream_current_rate_proof_l371_37189


namespace NUMINAMATH_GPT_simplify_expression_l371_37129

theorem simplify_expression (x : ℝ) : 
  (3 * x^2 + 4 * x - 5) * (x - 2) + (x - 2) * (2 * x^2 - 3 * x + 9) - (4 * x - 7) * (x - 2) * (x - 3) 
  = x^3 + x^2 + 12 * x - 36 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l371_37129


namespace NUMINAMATH_GPT_apples_shared_l371_37139

-- Definitions and conditions based on problem statement
def initial_apples : ℕ := 89
def remaining_apples : ℕ := 84

-- The goal to prove that Ruth shared 5 apples with Peter
theorem apples_shared : initial_apples - remaining_apples = 5 := by
  sorry

end NUMINAMATH_GPT_apples_shared_l371_37139


namespace NUMINAMATH_GPT_option_c_correct_l371_37122

theorem option_c_correct (a b : ℝ) (h : a < b) : a - 1 < b - 1 :=
sorry

end NUMINAMATH_GPT_option_c_correct_l371_37122


namespace NUMINAMATH_GPT_compute_value_l371_37136

open Nat Real

theorem compute_value (A B : ℝ × ℝ) (hA : A = (15, 10)) (hB : B = (-5, 6)) :
  let C : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ∃ (x y : ℝ), C = (x, y) ∧ 2 * x - 4 * y = -22 := by
  sorry

end NUMINAMATH_GPT_compute_value_l371_37136


namespace NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l371_37176

noncomputable def f (x a : ℝ) : ℝ := -x^2 + a * x + 4

def g (x : ℝ) : ℝ := abs (x + 1) + abs (x - 1)

theorem part1_solution_set (a : ℝ := 1) :
  {x : ℝ | f x a ≥ g x} = { x : ℝ | -1 ≤ x ∧ x ≤ (Real.sqrt 17 - 1) / 2 } :=
by
  sorry

theorem part2_range_of_a (a : ℝ) :
  (∀ x ∈ [-1,1], f x a ≥ g x) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l371_37176


namespace NUMINAMATH_GPT_anya_hair_growth_l371_37128

theorem anya_hair_growth (wash_loss : ℕ) (brush_loss : ℕ) (total_loss : ℕ) : wash_loss = 32 → brush_loss = wash_loss / 2 → total_loss = wash_loss + brush_loss → total_loss + 1 = 49 :=
by
  sorry

end NUMINAMATH_GPT_anya_hair_growth_l371_37128


namespace NUMINAMATH_GPT_intersection_M_N_l371_37166

noncomputable def M : Set ℝ := { x | x^2 + x - 2 = 0 }
def N : Set ℝ := { x | x < 0 }

theorem intersection_M_N : M ∩ N = { -2 } := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l371_37166


namespace NUMINAMATH_GPT_identity_map_a_plus_b_l371_37142

theorem identity_map_a_plus_b (a b : ℝ) (h : ∀ x ∈ ({-1, b / a, 1} : Set ℝ), x ∈ ({a, b, b - a} : Set ℝ)) : a + b = -1 ∨ a + b = 1 :=
by
  sorry

end NUMINAMATH_GPT_identity_map_a_plus_b_l371_37142


namespace NUMINAMATH_GPT_g_at_6_l371_37161

def g (x : ℝ) : ℝ := 2 * x^4 - 13 * x^3 + 28 * x^2 - 32 * x - 48

theorem g_at_6 : g 6 = 552 :=
by sorry

end NUMINAMATH_GPT_g_at_6_l371_37161


namespace NUMINAMATH_GPT_max_knights_is_seven_l371_37115

-- Definitions of conditions
def students : ℕ := 11
def total_statements : ℕ := students * (students - 1)
def liar_statements : ℕ := 56

-- Definition translating the problem statement
theorem max_knights_is_seven : ∃ (k li : ℕ), 
  (k + li = students) ∧ 
  (k * li = liar_statements) ∧ 
  (k = 7) := 
by
  sorry

end NUMINAMATH_GPT_max_knights_is_seven_l371_37115


namespace NUMINAMATH_GPT_square_area_dimensions_l371_37140

theorem square_area_dimensions (x : ℝ) (n : ℝ) : 
  (x^2 + (x + 12)^2 = 2120) → 
  (n = x + 12) → 
  (x = 26) → 
  (n = 38) := 
by
  sorry

end NUMINAMATH_GPT_square_area_dimensions_l371_37140


namespace NUMINAMATH_GPT_total_games_in_season_is_correct_l371_37125

-- Definitions based on given conditions
def games_per_month : ℕ := 7
def season_months : ℕ := 2

-- The theorem to prove
theorem total_games_in_season_is_correct : 
  (games_per_month * season_months = 14) :=
by
  sorry

end NUMINAMATH_GPT_total_games_in_season_is_correct_l371_37125


namespace NUMINAMATH_GPT_period_f_2pi_max_value_f_exists_max_f_l371_37116

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.cos x) + Real.cos (Real.sin x)

theorem period_f_2pi : ∀ x : ℝ, f (x + 2 * Real.pi) = f x := by
  sorry

theorem max_value_f : ∀ x : ℝ, f x ≤ Real.sin 1 + 1 := by
  sorry

-- Optional: Existence of the maximum value.
theorem exists_max_f : ∃ x : ℝ, f x = Real.sin 1 + 1 := by
  sorry

end NUMINAMATH_GPT_period_f_2pi_max_value_f_exists_max_f_l371_37116


namespace NUMINAMATH_GPT_Emily_walks_more_distance_than_Troy_l371_37174

theorem Emily_walks_more_distance_than_Troy (Troy_distance Emily_distance : ℕ) (days : ℕ) 
  (hTroy : Troy_distance = 75) (hEmily : Emily_distance = 98) (hDays : days = 5) : 
  ((Emily_distance * 2 - Troy_distance * 2) * days) = 230 :=
by
  sorry

end NUMINAMATH_GPT_Emily_walks_more_distance_than_Troy_l371_37174


namespace NUMINAMATH_GPT_grocery_cost_l371_37154

/-- Potatoes and celery costs problem. -/
theorem grocery_cost (a b : ℝ) (potato_cost_per_kg celery_cost_per_kg : ℝ) 
(h1 : potato_cost_per_kg = 1) (h2 : celery_cost_per_kg = 0.7) :
  potato_cost_per_kg * a + celery_cost_per_kg * b = a + 0.7 * b :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_grocery_cost_l371_37154


namespace NUMINAMATH_GPT_count_congruent_3_mod_8_l371_37184

theorem count_congruent_3_mod_8 (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 300) :
  ∃ k : ℕ, (1 ≤ 8 * k + 3 ∧ 8 * k + 3 ≤ 300) ∧ n = 38 :=
by
  sorry

end NUMINAMATH_GPT_count_congruent_3_mod_8_l371_37184


namespace NUMINAMATH_GPT_problem_statement_l371_37193

variable (x : ℝ)

-- Definitions based on the conditions
def a := 2005 * x + 2009
def b := 2005 * x + 2010
def c := 2005 * x + 2011

-- Assertion for the problem
theorem problem_statement : a ^ 2 + b ^ 2 + c ^ 2 - a * b - b * c - c * a = 3 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l371_37193


namespace NUMINAMATH_GPT_difference_of_two_numbers_l371_37110

theorem difference_of_two_numbers (a b : ℕ) (h₀ : a + b = 25800) (h₁ : b = 12 * a) (h₂ : b % 10 = 0) (h₃ : b / 10 = a) : b - a = 21824 :=
by 
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_difference_of_two_numbers_l371_37110


namespace NUMINAMATH_GPT_f_99_eq_1_l371_37100

-- Define an even function on ℝ
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- The conditions to be satisfied by the function f
variables (f : ℝ → ℝ)
variable (h_even : is_even_function f)
variable (h_f1 : f 1 = 1)
variable (h_period : ∀ x, f (x + 4) = f x)

-- Prove that f(99) = 1
theorem f_99_eq_1 : f 99 = 1 :=
by
  sorry

end NUMINAMATH_GPT_f_99_eq_1_l371_37100


namespace NUMINAMATH_GPT_original_area_area_after_translation_l371_37133

-- Defining vectors v, w, and t
def v : ℝ × ℝ := (6, -4)
def w : ℝ × ℝ := (-8, 3)
def t : ℝ × ℝ := (3, 2)

-- Function to compute the determinant of two vectors in R^2
def det (v w : ℝ × ℝ) : ℝ := v.1 * w.2 - v.2 * w.1

-- The area of a parallelogram is the absolute value of the determinant
def parallelogram_area (v w : ℝ × ℝ) : ℝ := |det v w|

-- Proving the original area is 14
theorem original_area : parallelogram_area v w = 14 := by
  sorry

-- Proving the area remains the same after translation
theorem area_after_translation : parallelogram_area v w = parallelogram_area (v.1 + t.1, v.2 + t.2) (w.1 + t.1, w.2 + t.2) := by
  sorry

end NUMINAMATH_GPT_original_area_area_after_translation_l371_37133


namespace NUMINAMATH_GPT_sequence_strictly_monotonic_increasing_l371_37192

noncomputable def a (n : ℕ) : ℝ := ((n + 1) ^ n * n ^ (2 - n)) / (7 * n ^ 2 + 1)

theorem sequence_strictly_monotonic_increasing :
  ∀ n : ℕ, a n < a (n + 1) := 
by {
  sorry
}

end NUMINAMATH_GPT_sequence_strictly_monotonic_increasing_l371_37192


namespace NUMINAMATH_GPT_measure_of_angle_A_l371_37148

-- Defining the measures of angles
def angle_B : ℝ := 50
def angle_C : ℝ := 40
def angle_D : ℝ := 30

-- Prove that measure of angle A is 120 degrees given the conditions
theorem measure_of_angle_A (B C D : ℝ) (hB : B = angle_B) (hC : C = angle_C) (hD : D = angle_D) : B + C + D + 60 = 180 -> 180 - (B + C + D + 60) = 120 :=
by sorry

end NUMINAMATH_GPT_measure_of_angle_A_l371_37148


namespace NUMINAMATH_GPT_find_set_of_points_B_l371_37151

noncomputable def is_incenter (A B C I : Point) : Prop :=
  -- define the incenter condition
  sorry

noncomputable def angle_less_than (A B C : Point) (α : ℝ) : Prop :=
  -- define the condition that all angles of triangle ABC are less than α
  sorry

theorem find_set_of_points_B (A I : Point) (α : ℝ) (hα1 : 60 < α) (hα2 : α < 90) :
  ∃ B : Point, ∃ C : Point,
    is_incenter A B C I ∧ angle_less_than A B C α :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_find_set_of_points_B_l371_37151


namespace NUMINAMATH_GPT_find_alpha_l371_37181

theorem find_alpha (n : ℕ) (h : ∀ x : ℤ, x * x * x + α * x + 4 - 2 * 2016 ^ n = 0 → ∀ r : ℤ, x = r)
  : α = -3 :=
sorry

end NUMINAMATH_GPT_find_alpha_l371_37181


namespace NUMINAMATH_GPT_solve_exponential_equation_l371_37114

theorem solve_exponential_equation (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  3^x + 4^y = 5^z ↔ x = 2 ∧ y = 2 ∧ z = 2 :=
by sorry

end NUMINAMATH_GPT_solve_exponential_equation_l371_37114


namespace NUMINAMATH_GPT_rectangle_area_l371_37138

theorem rectangle_area (l w : ℕ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) 
  : l * w = 1600 := 
by 
  sorry

end NUMINAMATH_GPT_rectangle_area_l371_37138


namespace NUMINAMATH_GPT_smallest_n_congruence_l371_37158

theorem smallest_n_congruence :
  ∃ n : ℕ+, 537 * (n : ℕ) % 30 = 1073 * (n : ℕ) % 30 ∧ (∀ m : ℕ+, 537 * (m : ℕ) % 30 = 1073 * (m : ℕ) % 30 → (m : ℕ) < n → false) :=
  sorry

end NUMINAMATH_GPT_smallest_n_congruence_l371_37158


namespace NUMINAMATH_GPT_division_problem_l371_37188

theorem division_problem (D : ℕ) (Quotient Dividend Remainder : ℕ) 
    (h1 : Quotient = 36) 
    (h2 : Dividend = 3086) 
    (h3 : Remainder = 26) 
    (h_div : Dividend = (D * Quotient) + Remainder) : 
    D = 85 := 
by 
  -- Steps to prove the theorem will go here
  sorry

end NUMINAMATH_GPT_division_problem_l371_37188


namespace NUMINAMATH_GPT_expected_number_of_socks_l371_37179

noncomputable def expected_socks_to_pick (n : ℕ) : ℚ := (2 * (n + 1)) / 3

theorem expected_number_of_socks (n : ℕ) (h : n ≥ 2) : 
  (expected_socks_to_pick n) = (2 * (n + 1)) / 3 := 
by
  sorry

end NUMINAMATH_GPT_expected_number_of_socks_l371_37179


namespace NUMINAMATH_GPT_solve_system_of_equations_l371_37197

theorem solve_system_of_equations :
  ∃ y : ℝ, (2 * 2 + y = 0) ∧ (2 + y = 3) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l371_37197


namespace NUMINAMATH_GPT_annual_growth_rate_l371_37123

theorem annual_growth_rate (u_2021 u_2023 : ℝ) (x : ℝ) : 
    u_2021 = 1 ∧ u_2023 = 1.69 ∧ x > 0 → (u_2023 / u_2021) = (1 + x)^2 → x * 100 = 30 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_annual_growth_rate_l371_37123


namespace NUMINAMATH_GPT_incorrect_vertex_is_false_l371_37153

-- Definition of the given parabola
def parabola (x : ℝ) : ℝ := -2 * (x - 2)^2 + 1

-- Define the incorrect hypothesis: Vertex at (-2, 1)
def incorrect_vertex (x y : ℝ) : Prop := (x, y) = (-2, 1)

-- Proposition to prove that the vertex is not at (-2, 1)
theorem incorrect_vertex_is_false : ¬ ∃ x y, (x, y) = (-2, 1) ∧ parabola x = y :=
by
  sorry

end NUMINAMATH_GPT_incorrect_vertex_is_false_l371_37153


namespace NUMINAMATH_GPT_factorization_correctness_l371_37130

theorem factorization_correctness :
  (∀ x : ℝ, (x + 1) * (x - 1) = x^2 - 1 → false) ∧
  (∀ x : ℝ, x^2 - 4 * x + 4 = x * (x - 4) + 4 → false) ∧
  (∀ x : ℝ, (x + 3) * (x - 4) = x^2 - x - 12 → false) ∧
  (∀ x : ℝ, x^2 - 4 = (x + 2) * (x - 2)) :=
by
  sorry

end NUMINAMATH_GPT_factorization_correctness_l371_37130


namespace NUMINAMATH_GPT_A_beats_B_by_160_meters_l371_37105

-- Definitions used in conditions
def distance_A := 400 -- meters
def time_A := 60 -- seconds
def distance_B := 400 -- meters
def time_B := 100 -- seconds
def speed_B := distance_B / time_B -- B's speed in meters/second
def time_for_B_in_A_time := time_A -- B's time for the duration A took to finish the race
def distance_B_in_A_time := speed_B * time_for_B_in_A_time -- Distance B covers in A's time

-- Statement to prove
theorem A_beats_B_by_160_meters : distance_A - distance_B_in_A_time = 160 :=
by
  -- This is a placeholder for an eventual proof
  sorry

end NUMINAMATH_GPT_A_beats_B_by_160_meters_l371_37105


namespace NUMINAMATH_GPT_number_of_triples_l371_37160

theorem number_of_triples : 
  {n : ℕ // ∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ ab = c ∧ bc = a ∧ ca = b ∧ n = 4} :=
sorry

end NUMINAMATH_GPT_number_of_triples_l371_37160


namespace NUMINAMATH_GPT_balls_into_boxes_l371_37137

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end NUMINAMATH_GPT_balls_into_boxes_l371_37137


namespace NUMINAMATH_GPT_number_of_people_in_group_is_21_l371_37167

-- Definitions based directly on the conditions
def pins_contribution_per_day := 10
def pins_deleted_per_week_per_person := 5
def group_initial_pins := 1000
def final_pins_after_month := 6600
def weeks_in_a_month := 4

-- To be proved: number of people in the group is 21
theorem number_of_people_in_group_is_21 (P : ℕ)
  (h1 : final_pins_after_month - group_initial_pins = 5600)
  (h2 : weeks_in_a_month * (pins_contribution_per_day * 7 - pins_deleted_per_week_per_person) = 260)
  (h3 : 5600 / 260 = 21) :
  P = 21 := 
sorry

end NUMINAMATH_GPT_number_of_people_in_group_is_21_l371_37167


namespace NUMINAMATH_GPT_roots_real_l371_37108

variable {x p q k : ℝ}
variable {x1 x2 : ℝ}

theorem roots_real 
  (h1 : x^2 + p * x + q = 0) 
  (h2 : p = -(x1 + x2)) 
  (h3 : q = x1 * x2) 
  (h4 : x1 ≠ x2) 
  (h5 :  x1^2 - 2*x1*x2 + x2^2 + 4*q = 0):
  (∃ y1 y2, y1 = k * x1 + (1 / k) * x2 ∧ y2 = k * x2 + (1 / k) * x1 ∧ 
    (y1^2 + (k + 1/k) * p * y1 + (p^2 + q * ((k - 1/k)^2)) = 0) ∧ 
    (y2^2 + (k + 1/k) * p * y2 + (p^2 + q * ((k - 1/k)^2)) = 0)) → 
  (∃ z1 z2, z1 = k * x1 ∧ z2 = 1/k * x2 ∧ 
    (z1^2 - y1 * z1 + q = 0) ∧ 
    (z2^2 - y2 * z2 + q = 0)) :=
sorry

end NUMINAMATH_GPT_roots_real_l371_37108


namespace NUMINAMATH_GPT_distance_dormitory_to_city_l371_37131

variable (D : ℝ)
variable (c : ℝ := 12)
variable (f := (1/5) * D)
variable (b := (2/3) * D)

theorem distance_dormitory_to_city (h : f + b + c = D) : D = 90 := by
  sorry

end NUMINAMATH_GPT_distance_dormitory_to_city_l371_37131


namespace NUMINAMATH_GPT_fourth_intersection_point_l371_37135

noncomputable def fourth_point_of_intersection : Prop :=
  let hyperbola (x y : ℝ) := x * y = 1
  let circle (x y : ℝ) := (x - 1)^2 + (y + 1)^2 = 10
  let known_points : List (ℝ × ℝ) := [(3, 1/3), (-4, -1/4), (1/2, 2)]
  let fourth_point := (-1/6, -6)
  (hyperbola 3 (1/3)) ∧ (hyperbola (-4) (-1/4)) ∧ (hyperbola (1/2) 2) ∧
  (circle 3 (1/3)) ∧ (circle (-4) (-1/4)) ∧ (circle (1/2) 2) ∧ 
  (hyperbola (-1/6) (-6)) ∧ (circle (-1/6) (-6)) ∧ 
  ∀ (x y : ℝ), (hyperbola x y) → (circle x y) → ((x, y) = fourth_point ∨ (x, y) ∈ known_points)
  
theorem fourth_intersection_point :
  fourth_point_of_intersection :=
sorry

end NUMINAMATH_GPT_fourth_intersection_point_l371_37135
