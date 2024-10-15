import Mathlib

namespace NUMINAMATH_GPT_production_cost_percentage_l53_5305

theorem production_cost_percentage
    (initial_cost final_cost : ℝ)
    (final_cost_eq : final_cost = 48)
    (initial_cost_eq : initial_cost = 50)
    (h : (initial_cost + 0.5 * x) * (1 - x / 100) = final_cost) :
    x = 20 :=
by
  sorry

end NUMINAMATH_GPT_production_cost_percentage_l53_5305


namespace NUMINAMATH_GPT_factor_expression_l53_5384

variable (b : ℝ)

theorem factor_expression : 221 * b * b + 17 * b = 17 * b * (13 * b + 1) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l53_5384


namespace NUMINAMATH_GPT_perfect_square_solution_l53_5389

theorem perfect_square_solution (n : ℕ) : ∃ a : ℕ, n * 2^(n+1) + 1 = a^2 ↔ n = 0 ∨ n = 3 := by
  sorry

end NUMINAMATH_GPT_perfect_square_solution_l53_5389


namespace NUMINAMATH_GPT_average_attendance_percentage_l53_5392

theorem average_attendance_percentage :
  let total_laborers := 300
  let day1_present := 150
  let day2_present := 225
  let day3_present := 180
  let day1_percentage := (day1_present / total_laborers) * 100
  let day2_percentage := (day2_present / total_laborers) * 100
  let day3_percentage := (day3_present / total_laborers) * 100
  let average_percentage := (day1_percentage + day2_percentage + day3_percentage) / 3
  average_percentage = 61.7 := by
  sorry

end NUMINAMATH_GPT_average_attendance_percentage_l53_5392


namespace NUMINAMATH_GPT_exam_total_students_l53_5359
-- Import the necessary Lean libraries

-- Define the problem conditions and the proof goal
theorem exam_total_students (T : ℕ) (h1 : 27 * T / 100 ≤ T) (h2 : 54 * T / 100 ≤ T) (h3 : 57 = 19 * T / 100) :
  T = 300 :=
  sorry  -- Proof is omitted here.

end NUMINAMATH_GPT_exam_total_students_l53_5359


namespace NUMINAMATH_GPT_lemonade_water_quarts_l53_5343

theorem lemonade_water_quarts :
  let ratioWaterLemon := (4 : ℕ) / (1 : ℕ)
  let totalParts := 4 + 1
  let totalVolumeInGallons := 3
  let quartsPerGallon := 4
  let totalVolumeInQuarts := totalVolumeInGallons * quartsPerGallon
  let volumePerPart := totalVolumeInQuarts / totalParts
  let volumeWater := 4 * volumePerPart
  volumeWater = 9.6 :=
by
  -- placeholder for actual proof
  sorry

end NUMINAMATH_GPT_lemonade_water_quarts_l53_5343


namespace NUMINAMATH_GPT_find_divisor_l53_5373

theorem find_divisor (N D k : ℤ) (h1 : N = 5 * D) (h2 : N % 11 = 2) : D = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l53_5373


namespace NUMINAMATH_GPT_largest_four_digit_number_with_digits_sum_25_l53_5391

def four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  (n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + (n % 10) = s)

theorem largest_four_digit_number_with_digits_sum_25 :
  ∃ n, four_digit n ∧ digits_sum_to n 25 ∧ ∀ m, four_digit m → digits_sum_to m 25 → m ≤ n :=
sorry

end NUMINAMATH_GPT_largest_four_digit_number_with_digits_sum_25_l53_5391


namespace NUMINAMATH_GPT_adam_teaches_650_students_in_10_years_l53_5371

noncomputable def students_in_n_years (n : ℕ) : ℕ :=
  if n = 1 then 40
  else if n = 2 then 60
  else if n = 3 then 70
  else if n <= 10 then 70
  else 0 -- beyond the scope of this problem

theorem adam_teaches_650_students_in_10_years :
  (students_in_n_years 1 + students_in_n_years 2 + students_in_n_years 3 +
   students_in_n_years 4 + students_in_n_years 5 + students_in_n_years 6 +
   students_in_n_years 7 + students_in_n_years 8 + students_in_n_years 9 +
   students_in_n_years 10) = 650 :=
by
  sorry

end NUMINAMATH_GPT_adam_teaches_650_students_in_10_years_l53_5371


namespace NUMINAMATH_GPT_find_p_l53_5332

variables (a b c p : ℝ)

theorem find_p 
  (h1 : 9 / (a + b) = 13 / (c - b)) : 
  p = 22 :=
sorry

end NUMINAMATH_GPT_find_p_l53_5332


namespace NUMINAMATH_GPT_chromium_percentage_in_second_alloy_l53_5379

theorem chromium_percentage_in_second_alloy (x : ℝ) :
  (15 * 0.12) + (35 * (x / 100)) = 50 * 0.106 → x = 10 :=
by
  sorry

end NUMINAMATH_GPT_chromium_percentage_in_second_alloy_l53_5379


namespace NUMINAMATH_GPT_find_number_l53_5350

theorem find_number (x : ℝ) (h : 0.4 * x + 60 = x) : x = 100 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l53_5350


namespace NUMINAMATH_GPT_jennifer_total_discount_is_28_l53_5333

-- Define the conditions in the Lean context

def initial_whole_milk_cans : ℕ := 40 
def mark_whole_milk_cans : ℕ := 30 
def mark_skim_milk_cans : ℕ := 15 
def almond_milk_per_3_whole_milk : ℕ := 2 
def whole_milk_per_5_skim_milk : ℕ := 4 
def discount_per_10_whole_milk : ℕ := 4 
def discount_per_7_almond_milk : ℕ := 3 
def discount_per_3_almond_milk : ℕ := 1

def jennifer_additional_almond_milk := (mark_whole_milk_cans / 3) * almond_milk_per_3_whole_milk
def jennifer_additional_whole_milk := (mark_skim_milk_cans / 5) * whole_milk_per_5_skim_milk

def jennifer_whole_milk_cans := initial_whole_milk_cans + jennifer_additional_whole_milk
def jennifer_almond_milk_cans := jennifer_additional_almond_milk

def jennifer_whole_milk_discount := (jennifer_whole_milk_cans / 10) * discount_per_10_whole_milk
def jennifer_almond_milk_discount := 
  (jennifer_almond_milk_cans / 7) * discount_per_7_almond_milk + 
  ((jennifer_almond_milk_cans % 7) / 3) * discount_per_3_almond_milk

def total_jennifer_discount := jennifer_whole_milk_discount + jennifer_almond_milk_discount

-- Theorem stating the total discount 
theorem jennifer_total_discount_is_28 : total_jennifer_discount = 28 := by
  sorry

end NUMINAMATH_GPT_jennifer_total_discount_is_28_l53_5333


namespace NUMINAMATH_GPT_solve_for_x_l53_5395

theorem solve_for_x (x : ℝ) : (x - 20) / 3 = (4 - 3 * x) / 4 → x = 7.08 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l53_5395


namespace NUMINAMATH_GPT_carrie_payment_l53_5302

def num_shirts := 8
def cost_per_shirt := 12
def total_shirt_cost := num_shirts * cost_per_shirt

def num_pants := 4
def cost_per_pant := 25
def total_pant_cost := num_pants * cost_per_pant

def num_jackets := 4
def cost_per_jacket := 75
def total_jacket_cost := num_jackets * cost_per_jacket

def num_skirts := 3
def cost_per_skirt := 30
def total_skirt_cost := num_skirts * cost_per_skirt

def num_shoes := 2
def cost_per_shoe := 50
def total_shoe_cost := num_shoes * cost_per_shoe

def total_cost := total_shirt_cost + total_pant_cost + total_jacket_cost + total_skirt_cost + total_shoe_cost

def mom_share := (2 / 3 : ℚ) * total_cost
def carrie_share := total_cost - mom_share

theorem carrie_payment : carrie_share = 228.67 :=
by
  sorry

end NUMINAMATH_GPT_carrie_payment_l53_5302


namespace NUMINAMATH_GPT_problem_statement_l53_5380

noncomputable def original_expression (x : ℕ) : ℚ :=
(1 - 1 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1))

theorem problem_statement (x : ℕ) (hx1 : 3 - x ≥ 0) (hx2 : x ≠ 2) (hx3 : x ≠ 1) :
  original_expression 3 = 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l53_5380


namespace NUMINAMATH_GPT_find_sum_l53_5356

variable (a b : ℚ)

theorem find_sum :
  2 * a + 5 * b = 31 ∧ 4 * a + 3 * b = 35 → a + b = 68 / 7 := by
  sorry

end NUMINAMATH_GPT_find_sum_l53_5356


namespace NUMINAMATH_GPT_single_interval_condition_l53_5338

-- Definitions: k and l are integers
variables (k l : ℤ)

-- Condition: The given condition for l
theorem single_interval_condition : l = Int.floor (k ^ 2 / 4) :=
sorry

end NUMINAMATH_GPT_single_interval_condition_l53_5338


namespace NUMINAMATH_GPT_prove_angle_sum_l53_5358

open Real

theorem prove_angle_sum (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : cos α / sin β + cos β / sin α = 2) : 
  α + β = π / 2 := 
sorry

end NUMINAMATH_GPT_prove_angle_sum_l53_5358


namespace NUMINAMATH_GPT_average_marks_for_class_l53_5355

theorem average_marks_for_class (total_students : ℕ) (marks_group1 marks_group2 marks_group3 : ℕ) (num_students_group1 num_students_group2 num_students_group3 : ℕ) 
  (h1 : total_students = 50) 
  (h2 : num_students_group1 = 10) 
  (h3 : marks_group1 = 90) 
  (h4 : num_students_group2 = 15) 
  (h5 : marks_group2 = 80) 
  (h6 : num_students_group3 = total_students - num_students_group1 - num_students_group2) 
  (h7 : marks_group3 = 60) : 
  (10 * 90 + 15 * 80 + (total_students - 10 - 15) * 60) / total_students = 72 := 
by
  sorry

end NUMINAMATH_GPT_average_marks_for_class_l53_5355


namespace NUMINAMATH_GPT_next_term_geometric_sequence_l53_5313

noncomputable def geometric_term (a r : ℕ) (n : ℕ) : ℕ :=
a * r^n

theorem next_term_geometric_sequence (y : ℕ) :
  ∀ a₁ a₂ a₃ a₄, a₁ = 3 → a₂ = 9 * y → a₃ = 27 * y^2 → a₄ = 81 * y^3 →
  geometric_term 3 (3 * y) 4 = 243 * y^4 :=
by
  intros a₁ a₂ a₃ a₄ h₁ h₂ h₃ h₄
  sorry

end NUMINAMATH_GPT_next_term_geometric_sequence_l53_5313


namespace NUMINAMATH_GPT_intersection_complement_l53_5310

-- Define the sets A and B
def A : Set ℝ := { x : ℝ | -1 ≤ x ∧ x ≤ 1 }
def B : Set ℝ := { x : ℝ | x > 0 }

-- Define the complement of B
def complement_B : Set ℝ := { x : ℝ | x ≤ 0 }

-- The theorem we need to prove
theorem intersection_complement :
  A ∩ complement_B = { x : ℝ | -1 ≤ x ∧ x ≤ 0 } := 
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l53_5310


namespace NUMINAMATH_GPT_noah_billed_amount_l53_5324

theorem noah_billed_amount
  (minutes_per_call : ℕ)
  (cost_per_minute : ℝ)
  (weeks_per_year : ℕ)
  (total_cost : ℝ)
  (h_minutes_per_call : minutes_per_call = 30)
  (h_cost_per_minute : cost_per_minute = 0.05)
  (h_weeks_per_year : weeks_per_year = 52)
  (h_total_cost : total_cost = 78) :
  (minutes_per_call * cost_per_minute * weeks_per_year = total_cost) :=
by
  sorry

end NUMINAMATH_GPT_noah_billed_amount_l53_5324


namespace NUMINAMATH_GPT_find_N_value_l53_5368

-- Definitions based on given conditions
def M (n : ℕ) : ℕ := 4^n
def N (n : ℕ) : ℕ := 2^n
def condition (n : ℕ) : Prop := M n - N n = 240

-- Theorem statement to prove N == 16 given the conditions
theorem find_N_value (n : ℕ) (h : condition n) : N n = 16 := 
  sorry

end NUMINAMATH_GPT_find_N_value_l53_5368


namespace NUMINAMATH_GPT_margo_total_distance_travelled_l53_5336

noncomputable def total_distance_walked (walking_time_in_minutes: ℝ) (stopping_time_in_minutes: ℝ) (additional_walking_time_in_minutes: ℝ) (walking_speed: ℝ) : ℝ :=
  walking_speed * ((walking_time_in_minutes + stopping_time_in_minutes + additional_walking_time_in_minutes) / 60)

noncomputable def total_distance_cycled (cycling_time_in_minutes: ℝ) (cycling_speed: ℝ) : ℝ :=
  cycling_speed * (cycling_time_in_minutes / 60)

theorem margo_total_distance_travelled :
  let walking_time := 10
  let stopping_time := 15
  let additional_walking_time := 10
  let cycling_time := 15
  let walking_speed := 4
  let cycling_speed := 10

  total_distance_walked walking_time stopping_time additional_walking_time walking_speed +
  total_distance_cycled cycling_time cycling_speed = 4.8333 := 
by 
  sorry

end NUMINAMATH_GPT_margo_total_distance_travelled_l53_5336


namespace NUMINAMATH_GPT_first_term_of_geometric_series_l53_5331

theorem first_term_of_geometric_series (r a S : ℝ) (h_r : r = 1 / 4) (h_S : S = 40) 
  (h_geometric_sum : S = a / (1 - r)) : a = 30 :=
by
  -- The proof would go here, but we place a sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_first_term_of_geometric_series_l53_5331


namespace NUMINAMATH_GPT_part_one_part_two_l53_5330

-- Part (1)
theorem part_one (m : ℝ) : 
  (∀ x, -1 ≤ x ∧ x ≤ 2 → (2 * m < x ∧ x < 1 → -1 ≤ x ∧ x ≤ 2 ∧ - (1 / 2) ≤ m)) → 
  (m ≥ - (1 / 2)) :=
by sorry

-- Part (2)
theorem part_two (m : ℝ) : 
  (∃ x : ℤ, (2 * m < x ∧ x < 1) ∧ (x < -1 ∨ x > 2)) ∧ 
  (∀ y : ℤ, (2 * m < y ∧ y < 1) ∧ (y < -1 ∨ y > 2) → y = x) → 
  (- (3 / 2) ≤ m ∧ m < -1) :=
by sorry

end NUMINAMATH_GPT_part_one_part_two_l53_5330


namespace NUMINAMATH_GPT_marbles_in_bag_l53_5390

theorem marbles_in_bag (r b : ℕ) : 
  (r - 2) * 10 = (r + b - 2) →
  (r * 6 = (r + b - 3)) →
  ((r - 2) * 8 = (r + b - 4)) →
  r + b = 42 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_marbles_in_bag_l53_5390


namespace NUMINAMATH_GPT_decimal_to_fraction_l53_5366

theorem decimal_to_fraction {a b c : ℚ} (H1 : a = 2.75) (H2 : b = 11) (H3 : c = 4) : (a = b / c) :=
by {
  sorry
}

end NUMINAMATH_GPT_decimal_to_fraction_l53_5366


namespace NUMINAMATH_GPT_square_binomial_formula_l53_5323

variable {x y : ℝ}

theorem square_binomial_formula :
  (2 * x + y) * (y - 2 * x) = y^2 - 4 * x^2 := 
  sorry

end NUMINAMATH_GPT_square_binomial_formula_l53_5323


namespace NUMINAMATH_GPT_smallest_six_digit_odd_div_by_125_l53_5345

theorem smallest_six_digit_odd_div_by_125 : 
  ∃ n : ℕ, n = 111375 ∧ 
           100000 ≤ n ∧ n < 1000000 ∧ 
           (∀ d : ℕ, d ∈ (n.digits 10) → d % 2 = 1) ∧ 
           n % 125 = 0 :=
by
  sorry

end NUMINAMATH_GPT_smallest_six_digit_odd_div_by_125_l53_5345


namespace NUMINAMATH_GPT_students_later_than_Yoongi_l53_5364

theorem students_later_than_Yoongi (total_students finished_before_Yoongi : ℕ) (h1 : total_students = 20) (h2 : finished_before_Yoongi = 11) :
  total_students - (finished_before_Yoongi + 1) = 8 :=
by {
  -- Proof is omitted as it's not required.
  sorry
}

end NUMINAMATH_GPT_students_later_than_Yoongi_l53_5364


namespace NUMINAMATH_GPT_solve_quadratic_eq_l53_5349

theorem solve_quadratic_eq (x : ℝ) : 4 * x ^ 2 - (x - 1) ^ 2 = 0 ↔ x = -1 ∨ x = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l53_5349


namespace NUMINAMATH_GPT_first_place_points_l53_5367

-- Definitions for the conditions
def num_teams : Nat := 4
def points_win : Nat := 2
def points_draw : Nat := 1
def points_loss : Nat := 0

def games_played (n : Nat) : Nat :=
  let pairs := n * (n - 1) / 2  -- Binomial coefficient C(n, 2)
  2 * pairs  -- Each pair plays twice

def total_points_distributed (n : Nat) (points_per_game : Nat) : Nat :=
  (games_played n) * points_per_game

def last_place_points : Nat := 5

-- The theorem to prove
theorem first_place_points : ∃ a b c : Nat, a + b + c = total_points_distributed num_teams points_win - last_place_points ∧ (a = 7 ∨ b = 7 ∨ c = 7) :=
by
  sorry

end NUMINAMATH_GPT_first_place_points_l53_5367


namespace NUMINAMATH_GPT_tournament_games_count_l53_5357

-- Defining the problem conditions
def num_players : Nat := 12
def plays_twice : Bool := true

-- Theorem statement
theorem tournament_games_count (n : Nat) (plays_twice : Bool) (h : n = num_players ∧ plays_twice = true) :
  (n * (n - 1) * 2) = 264 := by
  sorry

end NUMINAMATH_GPT_tournament_games_count_l53_5357


namespace NUMINAMATH_GPT_change_in_mean_and_median_l53_5351

-- Original attendance data
def original_data : List ℕ := [15, 23, 17, 19, 17, 20]

-- Corrected attendance data
def corrected_data : List ℕ := [15, 23, 17, 19, 17, 25]

-- Function to compute mean
def mean (data: List ℕ) : ℚ := (data.sum : ℚ) / data.length

-- Function to compute median
def median (data: List ℕ) : ℚ :=
  let sorted := data.toArray.qsort (· ≤ ·) |>.toList
  if sorted.length % 2 == 0 then
    (sorted.get! (sorted.length / 2 - 1) + sorted.get! (sorted.length / 2)) / 2
  else
    sorted.get! (sorted.length / 2)

-- Lean statement verifying the expected change in mean and median
theorem change_in_mean_and_median :
  mean corrected_data - mean original_data = 1 ∧ median corrected_data = median original_data :=
by -- Note the use of 'by' to structure the proof
  sorry -- Proof omitted

end NUMINAMATH_GPT_change_in_mean_and_median_l53_5351


namespace NUMINAMATH_GPT_hypotenuse_length_is_13_l53_5320

theorem hypotenuse_length_is_13 (a b c : ℝ) (ha : a = 5) (hb : b = 12)
  (hrt : a ^ 2 + b ^ 2 = c ^ 2) : c = 13 :=
by
  -- to complete the proof, fill in the details here
  sorry

end NUMINAMATH_GPT_hypotenuse_length_is_13_l53_5320


namespace NUMINAMATH_GPT_jill_water_filled_jars_l53_5376

variable (gallons : ℕ) (quart_halfGallon_gallon : ℕ)
variable (h_eq : gallons = 14)
variable (h_eq_n : quart_halfGallon_gallon = 3 * 8)
variable (h_total : quart_halfGallon_gallon = 24)

theorem jill_water_filled_jars :
  3 * (gallons * 4 / 7) = 24 :=
sorry

end NUMINAMATH_GPT_jill_water_filled_jars_l53_5376


namespace NUMINAMATH_GPT_find_n_series_sum_l53_5399

theorem find_n_series_sum 
  (first_term_I : ℝ) (second_term_I : ℝ) (first_term_II : ℝ) (second_term_II : ℝ) (sum_multiplier : ℝ) (n : ℝ)
  (h_I_first_term : first_term_I = 12)
  (h_I_second_term : second_term_I = 4)
  (h_II_first_term : first_term_II = 12)
  (h_II_second_term : second_term_II = 4 + n)
  (h_sum_multiplier : sum_multiplier = 5) :
  n = 152 :=
by
  sorry

end NUMINAMATH_GPT_find_n_series_sum_l53_5399


namespace NUMINAMATH_GPT_ratio_of_sums_l53_5340

theorem ratio_of_sums (total_sums : ℕ) (correct_sums : ℕ) (incorrect_sums : ℕ)
  (h1 : total_sums = 75)
  (h2 : incorrect_sums = 2 * correct_sums)
  (h3 : total_sums = correct_sums + incorrect_sums) :
  incorrect_sums / correct_sums = 2 :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_ratio_of_sums_l53_5340


namespace NUMINAMATH_GPT_yancheng_marathon_half_marathon_estimated_probability_l53_5317

noncomputable def estimated_probability
  (surveyed_participants_frequencies : List (ℕ × Real)) : Real :=
by
  -- Define the surveyed participants and their corresponding frequencies
  -- In this example, [(20, 0.35), (50, 0.40), (100, 0.39), (200, 0.415), (500, 0.418), (2000, 0.411)]
  sorry

theorem yancheng_marathon_half_marathon_estimated_probability :
  let surveyed_participants_frequencies := [
    (20, 0.350),
    (50, 0.400),
    (100, 0.390),
    (200, 0.415),
    (500, 0.418),
    (2000, 0.411)
  ]
  estimated_probability surveyed_participants_frequencies = 0.40 :=
by
  sorry

end NUMINAMATH_GPT_yancheng_marathon_half_marathon_estimated_probability_l53_5317


namespace NUMINAMATH_GPT_cans_purchased_l53_5353

variable (N P T : ℕ)

theorem cans_purchased (N P T : ℕ) : N * (5 * (T - 1)) / P = 5 * N * (T - 1) / P :=
by
  sorry

end NUMINAMATH_GPT_cans_purchased_l53_5353


namespace NUMINAMATH_GPT_total_cost_l53_5381

-- Define the cost of a neutral pen and a pencil
variables (x y : ℝ)

-- The total cost of buying 5 neutral pens and 3 pencils
theorem total_cost (x y : ℝ) : 5 * x + 3 * y = 5 * x + 3 * y :=
by
  -- The statement is self-evident, hence can be written directly
  sorry

end NUMINAMATH_GPT_total_cost_l53_5381


namespace NUMINAMATH_GPT_sum_of_primes_less_than_20_eq_77_l53_5397

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_primes_less_than_20_eq_77_l53_5397


namespace NUMINAMATH_GPT_car_travel_distance_l53_5334

theorem car_travel_distance (v d : ℕ) 
  (h1 : d = v * 7)
  (h2 : d = (v + 12) * 5) : 
  d = 210 := by 
  sorry

end NUMINAMATH_GPT_car_travel_distance_l53_5334


namespace NUMINAMATH_GPT_isabella_hair_length_end_of_year_l53_5337

/--
Isabella's initial hair length.
-/
def initial_hair_length : ℕ := 18

/--
Isabella's hair growth over the year.
-/
def hair_growth : ℕ := 6

/--
Prove that Isabella's hair length at the end of the year is 24 inches.
-/
theorem isabella_hair_length_end_of_year : initial_hair_length + hair_growth = 24 := by
  sorry

end NUMINAMATH_GPT_isabella_hair_length_end_of_year_l53_5337


namespace NUMINAMATH_GPT_number_of_girls_in_school_l53_5326

/-- Statement: There are 408 boys and some girls in a school which are to be divided into equal sections
of either boys or girls alone. The total number of sections thus formed is 26. Prove that the number 
of girls is 216. -/
theorem number_of_girls_in_school (n : ℕ) (n_boys : ℕ := 408) (total_sections : ℕ := 26)
  (h1 : n_boys = 408)
  (h2 : ∃ b g : ℕ, b + g = total_sections ∧ 408 / b = n / g ∧ b ∣ 408 ∧ g ∣ n) :
  n = 216 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_number_of_girls_in_school_l53_5326


namespace NUMINAMATH_GPT_cube_volume_surface_area_l53_5382

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ s : ℝ, s^3 = 8 * x ∧ 6 * s^2 = 2 * x) → x = 0 :=
by
  sorry

end NUMINAMATH_GPT_cube_volume_surface_area_l53_5382


namespace NUMINAMATH_GPT_game_cost_l53_5387

theorem game_cost
    (initial_amount : ℕ)
    (cost_per_toy : ℕ)
    (num_toys : ℕ)
    (remaining_amount := initial_amount - cost_per_toy * num_toys)
    (cost_of_game := initial_amount - remaining_amount)
    (h1 : initial_amount = 57)
    (h2 : cost_per_toy = 6)
    (h3 : num_toys = 5) :
  cost_of_game = 27 :=
by
  sorry

end NUMINAMATH_GPT_game_cost_l53_5387


namespace NUMINAMATH_GPT_price_reduction_l53_5369

theorem price_reduction (P : ℝ) : 
  let first_day_reduction := 0.91 * P
  let second_day_reduction := 0.90 * first_day_reduction
  second_day_reduction = 0.819 * P :=
by 
  sorry

end NUMINAMATH_GPT_price_reduction_l53_5369


namespace NUMINAMATH_GPT_range_of_a_l53_5341

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x ≤ 4 → 3 * x - a ≥ 0) → a ≤ 6 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_range_of_a_l53_5341


namespace NUMINAMATH_GPT_bug_total_distance_l53_5360

theorem bug_total_distance 
  (p₀ p₁ p₂ p₃ : ℤ) 
  (h₀ : p₀ = 0) 
  (h₁ : p₁ = 4) 
  (h₂ : p₂ = -3) 
  (h₃ : p₃ = 7) : 
  |p₁ - p₀| + |p₂ - p₁| + |p₃ - p₂| = 21 :=
by 
  sorry

end NUMINAMATH_GPT_bug_total_distance_l53_5360


namespace NUMINAMATH_GPT_determine_w_arithmetic_seq_l53_5388

theorem determine_w_arithmetic_seq (w : ℝ) (h : (w ≠ 0) ∧ 
  (1 / w - 1 / 2 = 1 / 2 - 1 / 3) ∧ (1 / 2 - 1 / 3 = 1 / 3 - 1 / 6)) :
  w = 3 / 2 := 
sorry

end NUMINAMATH_GPT_determine_w_arithmetic_seq_l53_5388


namespace NUMINAMATH_GPT_triangle_max_third_side_l53_5315

theorem triangle_max_third_side (D E F : ℝ) (a b : ℝ) (h1 : a = 8) (h2 : b = 15) 
(h_cos : Real.cos (3 * D) + Real.cos (3 * E) + Real.cos (3 * F) = 1) 
: ∃ c : ℝ, c = 13 :=
by
  sorry

end NUMINAMATH_GPT_triangle_max_third_side_l53_5315


namespace NUMINAMATH_GPT_f_x_minus_1_pass_through_l53_5335

variable (a : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + x

theorem f_x_minus_1_pass_through (a : ℝ) : f a (1 - 1) = 0 :=
by
  -- Proof is omitted here
  sorry

end NUMINAMATH_GPT_f_x_minus_1_pass_through_l53_5335


namespace NUMINAMATH_GPT_tub_emptying_time_l53_5318

variables (x C D T : ℝ) (hx : x > 0) (hC : C > 0) (hD : D > 0)

theorem tub_emptying_time (h1 : 4 * (D - x) = (5 / 7) * C) :
  T = 8 / (5 + (28 * x) / C) :=
by sorry

end NUMINAMATH_GPT_tub_emptying_time_l53_5318


namespace NUMINAMATH_GPT_min_squares_to_cover_5x5_l53_5307

theorem min_squares_to_cover_5x5 : 
  (∀ (cover : ℕ → ℕ), (cover 1 + cover 2 + cover 3 + cover 4) * (1^2 + 2^2 + 3^2 + 4^2) = 25 → 
  cover 1 + cover 2 + cover 3 + cover 4 = 10) :=
sorry

end NUMINAMATH_GPT_min_squares_to_cover_5x5_l53_5307


namespace NUMINAMATH_GPT_calc_625_to_4_div_5_l53_5374

theorem calc_625_to_4_div_5 :
  (625 : ℝ)^(4/5) = 238 :=
sorry

end NUMINAMATH_GPT_calc_625_to_4_div_5_l53_5374


namespace NUMINAMATH_GPT_num_triangles_with_perimeter_9_l53_5329

-- Definitions for side lengths and their conditions
def is_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def perimeter (a b c : ℕ) : Prop :=
  a + b + c = 9

-- The main theorem
theorem num_triangles_with_perimeter_9 : 
  ∃ (n : ℕ), n = 2 ∧ ∀ (a b c : ℕ), a + b + c = 9 → a ≤ b ∧ b ≤ c → is_triangle a b c ↔ 
  (a = 2 ∧ b = 3 ∧ c = 4) ∨ (a = 3 ∧ b = 3 ∧ c = 3) :=
by
  sorry

end NUMINAMATH_GPT_num_triangles_with_perimeter_9_l53_5329


namespace NUMINAMATH_GPT_crayons_per_box_l53_5362

-- Define the conditions
def crayons : ℕ := 80
def boxes : ℕ := 10

-- State the proof problem
theorem crayons_per_box : (crayons / boxes) = 8 := by
  sorry

end NUMINAMATH_GPT_crayons_per_box_l53_5362


namespace NUMINAMATH_GPT_quadratic_eq_two_distinct_real_roots_l53_5377

theorem quadratic_eq_two_distinct_real_roots :
    ∃ x y : ℝ, x ≠ y ∧ (x^2 + x - 1 = 0) ∧ (y^2 + y - 1 = 0) :=
by
    sorry

end NUMINAMATH_GPT_quadratic_eq_two_distinct_real_roots_l53_5377


namespace NUMINAMATH_GPT_yellow_not_greater_than_green_l53_5354

theorem yellow_not_greater_than_green
    (G Y S : ℕ)
    (h1 : G + Y + S = 100)
    (h2 : G + S / 2 = 50)
    (h3 : Y + S / 2 = 50) : ¬ Y > G :=
sorry

end NUMINAMATH_GPT_yellow_not_greater_than_green_l53_5354


namespace NUMINAMATH_GPT_hexagon_largest_angle_l53_5319

-- Definitions for conditions
def hexagon_interior_angle_sum : ℝ := 720  -- Sum of all interior angles of hexagon

def angle_A : ℝ := 100
def angle_B : ℝ := 120

-- Define x for angles C and D
variables (x : ℝ)
def angle_C : ℝ := x
def angle_D : ℝ := x
def angle_F : ℝ := 3 * x + 10

-- The formal statement to prove
theorem hexagon_largest_angle (x : ℝ) : 
  100 + 120 + x + x + (3 * x + 10) = 720 → 
  3 * x + 10 = 304 :=
by 
  sorry

end NUMINAMATH_GPT_hexagon_largest_angle_l53_5319


namespace NUMINAMATH_GPT_exercise_l53_5352

-- Define a, b, and the identity
def a : ℕ := 45
def b : ℕ := 15

-- Theorem statement
theorem exercise :
  ((a + b) ^ 2 - (a ^ 2 + b ^ 2) = 1350) :=
by
  sorry

end NUMINAMATH_GPT_exercise_l53_5352


namespace NUMINAMATH_GPT_total_handshakes_five_people_l53_5308

theorem total_handshakes_five_people : 
  let n := 5
  let total_handshakes (n : ℕ) : ℕ := (n * (n - 1)) / 2
  total_handshakes 5 = 10 :=
by sorry

end NUMINAMATH_GPT_total_handshakes_five_people_l53_5308


namespace NUMINAMATH_GPT_intersection_of_perpendicular_lines_l53_5314

theorem intersection_of_perpendicular_lines (x y : ℝ) : 
  (y = 3 * x + 4) ∧ (y = -1/3 * x + 4) → (x = 0 ∧ y = 4) :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_perpendicular_lines_l53_5314


namespace NUMINAMATH_GPT_flower_bed_length_l53_5325

theorem flower_bed_length (a b : ℝ) :
  ∀ width : ℝ, (6 * a^2 - 4 * a * b + 2 * a = 2 * a * width) → width = 3 * a - 2 * b + 1 :=
by
  intros width h
  sorry

end NUMINAMATH_GPT_flower_bed_length_l53_5325


namespace NUMINAMATH_GPT_remainder_division_l53_5346

theorem remainder_division (exists_quotient : ∃ q r : ℕ, r < 5 ∧ N = 5 * 5 + r)
    (exists_quotient_prime : ∃ k : ℕ, N = 11 * k + 3) :
  ∃ r : ℕ, r = 0 ∧ N % 5 = r := 
sorry

end NUMINAMATH_GPT_remainder_division_l53_5346


namespace NUMINAMATH_GPT_people_in_group_10_l53_5347

-- Let n represent the number of people in the group.
def number_of_people_in_group (n : ℕ) : Prop :=
  let average_increase : ℚ := 3.2
  let weight_of_replaced_person : ℚ := 65
  let weight_of_new_person : ℚ := 97
  let weight_increase : ℚ := weight_of_new_person - weight_of_replaced_person
  weight_increase = average_increase * n

theorem people_in_group_10 :
  ∃ n : ℕ, number_of_people_in_group n ∧ n = 10 :=
by
  sorry

end NUMINAMATH_GPT_people_in_group_10_l53_5347


namespace NUMINAMATH_GPT_union_of_A_and_B_intersection_of_A_and_B_complement_of_intersection_in_U_l53_5300

open Set

noncomputable def U : Set ℤ := {x | -2 < x ∧ x < 2}
def A : Set ℤ := {x | x^2 - 5 * x - 6 = 0}
def B : Set ℤ := {x | x^2 = 1}

theorem union_of_A_and_B : A ∪ B = {-1, 1, 6} :=
by
  sorry

theorem intersection_of_A_and_B : A ∩ B = {-1} :=
by
  sorry

theorem complement_of_intersection_in_U : U \ (A ∩ B) = {0, 1} :=
by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_intersection_of_A_and_B_complement_of_intersection_in_U_l53_5300


namespace NUMINAMATH_GPT_like_terms_m_n_sum_l53_5394

theorem like_terms_m_n_sum :
  ∃ (m n : ℕ), (2 : ℤ) * x ^ (3 * n) * y ^ (m + 4) = (-3 : ℤ) * x ^ 9 * y ^ (2 * n) ∧ m + n = 5 :=
by 
  sorry

end NUMINAMATH_GPT_like_terms_m_n_sum_l53_5394


namespace NUMINAMATH_GPT_question1_question2_l53_5370

def f (x : ℝ) : ℝ := |2 * x - 1|

theorem question1 (m : ℝ) (h1 : m > 0) 
(h2 : ∀ (x : ℝ), f (x + 1/2) ≤ 2 * m + 1 ↔ x ∈ [-2, 2]) : m = 3 / 2 := 
sorry

theorem question2 (x y : ℝ) : f x ≤ 2^y + 4 / 2^y + |2 * x + 3| := 
sorry

end NUMINAMATH_GPT_question1_question2_l53_5370


namespace NUMINAMATH_GPT_trajectory_is_ellipse_l53_5328

noncomputable def trajectory_of_P (P : ℝ × ℝ) : Prop :=
  ∃ (N : ℝ × ℝ), N.fst^2 + N.snd^2 = 8 ∧ 
                 ∃ (M : ℝ × ℝ), M.fst = 0 ∧ M.snd = N.snd ∧
                 P.fst = N.fst / 2 ∧ P.snd = N.snd

theorem trajectory_is_ellipse (P : ℝ × ℝ) (h : trajectory_of_P P) : 
  P.fst^2 / 2 + P.snd^2 / 8 = 1 :=
by
  sorry

end NUMINAMATH_GPT_trajectory_is_ellipse_l53_5328


namespace NUMINAMATH_GPT_nickel_chocolates_l53_5322

theorem nickel_chocolates (N : ℕ) (h : 7 = N + 2) : N = 5 :=
by
  sorry

end NUMINAMATH_GPT_nickel_chocolates_l53_5322


namespace NUMINAMATH_GPT_trigonometric_identity_l53_5304

theorem trigonometric_identity (x : ℝ) (h : Real.tan x = 2) : 
  (6 * Real.sin (2 * x) + 2 * Real.cos (2 * x)) / (Real.cos (2 * x) - 3 * Real.sin (2 * x)) = -2 / 5 := by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l53_5304


namespace NUMINAMATH_GPT_directrix_of_parabola_l53_5339

-- Define the variables and constants
variables (x y a : ℝ) (h₁ : x^2 = 4 * a * y) (h₂ : x = -2) (h₃ : y = 1)

theorem directrix_of_parabola (h : (-2)^2 = 4 * a * 1) : y = -1 := 
by
  -- Our proof will happen here, but we omit the details
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l53_5339


namespace NUMINAMATH_GPT_matrix_expression_l53_5385
open Matrix

variables {n : Type*} [Fintype n] [DecidableEq n]
variables (B : Matrix n n ℝ) (I : Matrix n n ℝ)

noncomputable def B_inverse := B⁻¹

-- Condition 1: B is a matrix with an inverse
variable [Invertible B]

-- Condition 2: (B - 3*I) * (B - 5*I) = 0
variable (H : (B - (3 : ℝ) • I) * (B - (5 : ℝ) • I) = 0)

-- Theorem to prove
theorem matrix_expression (B: Matrix n n ℝ) [Invertible B] 
  (H : (B - (3 : ℝ) • I) * (B - (5 : ℝ) • I) = 0) : 
  B + 10 * (B_inverse B) = (160 / 15 : ℝ) • I := 
sorry

end NUMINAMATH_GPT_matrix_expression_l53_5385


namespace NUMINAMATH_GPT_g_positive_l53_5393

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 1 / 2 + 1 / (2^x - 1) else 0

noncomputable def g (x : ℝ) : ℝ :=
  x^3 * f x

theorem g_positive (x : ℝ) (hx : x ≠ 0) : g x > 0 :=
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_g_positive_l53_5393


namespace NUMINAMATH_GPT_mod_37_5_l53_5378

theorem mod_37_5 : 37 % 5 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_mod_37_5_l53_5378


namespace NUMINAMATH_GPT_vertex_of_parabola_l53_5396

theorem vertex_of_parabola :
  ∃ (x y : ℝ), y^2 - 8*x + 6*y + 17 = 0 ∧ (x, y) = (1, -3) :=
by
  use 1, -3
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_l53_5396


namespace NUMINAMATH_GPT_ice_cream_arrangements_is_correct_l53_5321

-- Let us define the problem: counting the number of unique stacks of ice cream flavors
def ice_cream_scoops_arrangements : ℕ :=
  let total_scoops := 5
  let vanilla_scoops := 2
  Nat.factorial total_scoops / Nat.factorial vanilla_scoops

-- Assertion that needs to be proved
theorem ice_cream_arrangements_is_correct : ice_cream_scoops_arrangements = 60 := by
  -- Proof to be filled in; current placeholder
  sorry

end NUMINAMATH_GPT_ice_cream_arrangements_is_correct_l53_5321


namespace NUMINAMATH_GPT_factor_difference_of_squares_l53_5375

theorem factor_difference_of_squares (t : ℤ) : t^2 - 64 = (t - 8) * (t + 8) :=
by {
  sorry
}

end NUMINAMATH_GPT_factor_difference_of_squares_l53_5375


namespace NUMINAMATH_GPT_no_integers_p_and_q_l53_5398

theorem no_integers_p_and_q (p q : ℤ) : ¬(∀ x : ℤ, 3 ∣ (x^2 + p * x + q)) :=
by
  sorry

end NUMINAMATH_GPT_no_integers_p_and_q_l53_5398


namespace NUMINAMATH_GPT_find_second_number_l53_5363

-- Defining the ratios and sum condition
def ratio (a b c : ℕ) := 5*a = 3*b ∧ 3*b = 4*c

theorem find_second_number (a b c : ℕ) (h_ratio : ratio a b c) (h_sum : a + b + c = 108) : b = 27 :=
by
  sorry

end NUMINAMATH_GPT_find_second_number_l53_5363


namespace NUMINAMATH_GPT_find_a5_find_a31_div_a29_l53_5301

noncomputable def geo_diff_seq (a : ℕ → ℕ) (d : ℕ) :=
∀ n : ℕ, n > 0 → (a (n + 2) / a (n + 1)) - (a (n + 1) / a n) = d

theorem find_a5 (a : ℕ → ℕ) (d : ℕ) (h_geo_diff : geo_diff_seq a d)
  (h_init : a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 3) : a 5 = 105 :=
sorry

theorem find_a31_div_a29 (a : ℕ → ℕ) (d : ℕ) (h_geo_diff : geo_diff_seq a d)
  (h_init : a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 3) : a 31 / a 29 = 3363 :=
sorry

end NUMINAMATH_GPT_find_a5_find_a31_div_a29_l53_5301


namespace NUMINAMATH_GPT_gaoan_total_revenue_in_scientific_notation_l53_5327

theorem gaoan_total_revenue_in_scientific_notation :
  (21 * 10^9 : ℝ) = 2.1 * 10^9 :=
sorry

end NUMINAMATH_GPT_gaoan_total_revenue_in_scientific_notation_l53_5327


namespace NUMINAMATH_GPT_xiao_ying_should_pay_l53_5312

variable (x y z : ℝ)

def equation1 := 3 * x + 7 * y + z = 14
def equation2 := 4 * x + 10 * y + z = 16
def equation3 := 2 * (x + y + z) = 20

theorem xiao_ying_should_pay :
  equation1 x y z →
  equation2 x y z →
  equation3 x y z :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_xiao_ying_should_pay_l53_5312


namespace NUMINAMATH_GPT_undefined_expr_iff_l53_5386

theorem undefined_expr_iff (a : ℝ) : (∃ x, x = (a^2 - 9) ∧ x = 0) ↔ (a = -3 ∨ a = 3) :=
by
  sorry

end NUMINAMATH_GPT_undefined_expr_iff_l53_5386


namespace NUMINAMATH_GPT_blue_black_pen_ratio_l53_5311

theorem blue_black_pen_ratio (B K R : ℕ) 
  (h1 : B + K + R = 31) 
  (h2 : B = 18) 
  (h3 : K = R + 5) : 
  B / Nat.gcd B K = 2 ∧ K / Nat.gcd B K = 1 := 
by 
  sorry

end NUMINAMATH_GPT_blue_black_pen_ratio_l53_5311


namespace NUMINAMATH_GPT_capacity_of_bucket_in_first_scenario_l53_5306

theorem capacity_of_bucket_in_first_scenario (x : ℝ) 
  (h1 : 28 * x = 378) : x = 13.5 :=
by
  sorry

end NUMINAMATH_GPT_capacity_of_bucket_in_first_scenario_l53_5306


namespace NUMINAMATH_GPT_future_age_ratio_l53_5348

theorem future_age_ratio (j e x : ℕ) 
  (h1 : j - 3 = 5 * (e - 3)) 
  (h2 : j - 7 = 6 * (e - 7)) 
  (h3 : x = 17) : (j + x) / (e + x) = 3 := 
by
  sorry

end NUMINAMATH_GPT_future_age_ratio_l53_5348


namespace NUMINAMATH_GPT_unique_representation_l53_5365

theorem unique_representation (n : ℕ) (h_pos : 0 < n) : 
  ∃! (a b : ℚ), a = 1 / n ∧ b = 1 / (n + 1) ∧ (a + b = (2 * n + 1) / (n * (n + 1))) :=
by
  sorry

end NUMINAMATH_GPT_unique_representation_l53_5365


namespace NUMINAMATH_GPT_inverse_proportion_l53_5303

theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x = 3) (h3 : y = 15) (h4 : y = -30) : x = -3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_l53_5303


namespace NUMINAMATH_GPT_total_participating_students_l53_5372

-- Define the given conditions
def field_events_participants : ℕ := 15
def track_events_participants : ℕ := 13
def both_events_participants : ℕ := 5

-- Define the total number of students calculation
def total_students_participating : ℕ :=
  (field_events_participants - both_events_participants) + 
  (track_events_participants - both_events_participants) + 
  both_events_participants

-- State the theorem that needs to be proved
theorem total_participating_students : total_students_participating = 23 := by
  sorry

end NUMINAMATH_GPT_total_participating_students_l53_5372


namespace NUMINAMATH_GPT_determine_radius_l53_5309

variable (R r : ℝ)

theorem determine_radius (h1 : R = 10) (h2 : π * R^2 = 2 * (π * R^2 - π * r^2)) : r = 5 * Real.sqrt 2 :=
  sorry

end NUMINAMATH_GPT_determine_radius_l53_5309


namespace NUMINAMATH_GPT_value_of_a_plus_d_l53_5342

variable (a b c d : ℝ)

theorem value_of_a_plus_d 
  (h1 : a + b = 4) 
  (h2 : b + c = 5) 
  (h3 : c + d = 3) : 
  a + d = 1 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_a_plus_d_l53_5342


namespace NUMINAMATH_GPT_arith_seq_sum_first_110_l53_5316

variable {α : Type*} [OrderedRing α]

theorem arith_seq_sum_first_110 (a₁ d : α) :
  (10 * a₁ + 45 * d = 100) →
  (100 * a₁ + 4950 * d = 10) →
  (110 * a₁ + 5995 * d = -110) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_arith_seq_sum_first_110_l53_5316


namespace NUMINAMATH_GPT_square_of_square_root_l53_5383

theorem square_of_square_root (x : ℝ) (hx : (Real.sqrt x)^2 = 49) : x = 49 :=
by 
  sorry

end NUMINAMATH_GPT_square_of_square_root_l53_5383


namespace NUMINAMATH_GPT_non_zero_real_y_satisfies_l53_5361

theorem non_zero_real_y_satisfies (y : ℝ) (h : y ≠ 0) : (8 * y) ^ 3 = (16 * y) ^ 2 → y = 1 / 2 :=
by
  -- Lean code placeholders
  sorry

end NUMINAMATH_GPT_non_zero_real_y_satisfies_l53_5361


namespace NUMINAMATH_GPT_find_x_l53_5344

theorem find_x
  (p q : ℝ)
  (h1 : 3 / p = x)
  (h2 : 3 / q = 18)
  (h3 : p - q = 0.33333333333333337) :
  x = 6 :=
sorry

end NUMINAMATH_GPT_find_x_l53_5344
