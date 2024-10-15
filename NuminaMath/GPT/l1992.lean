import Mathlib

namespace NUMINAMATH_GPT_A_plays_D_third_day_l1992_199260

section GoTournament

variables (Player : Type) (A B C D : Player) 

-- Define the condition that each player competes with every other player exactly once.
def each_plays_once (P : Player → Player → Prop) : Prop :=
  ∀ x y, x ≠ y → (P x y ∨ P y x)

-- Define the tournament setup and the play conditions.
variables (P : Player → Player → Prop)
variable [∀ x y, Decidable (P x y)] -- Assuming decidability for the play relation

-- The given conditions of the problem
axiom A_plays_C_first_day : P A C
axiom C_plays_D_second_day : P C D
axiom only_one_match_per_day : ∀ x, ∃! y, P x y

-- We aim to prove that A will play against D on the third day.
theorem A_plays_D_third_day : P A D :=
sorry

end GoTournament

end NUMINAMATH_GPT_A_plays_D_third_day_l1992_199260


namespace NUMINAMATH_GPT_remainder_of_3_pow_19_mod_10_l1992_199236

-- Definition of the problem and conditions
def q := 3^19

-- Statement to prove
theorem remainder_of_3_pow_19_mod_10 : q % 10 = 7 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_3_pow_19_mod_10_l1992_199236


namespace NUMINAMATH_GPT_christine_siri_total_money_l1992_199274

-- Define the conditions
def christine_has_more_than_siri : ℝ := 20 -- Christine has 20 rs more than Siri
def christine_amount : ℝ := 20.5 -- Christine has 20.5 rs

-- Define the proof problem
theorem christine_siri_total_money :
  (∃ (siri_amount : ℝ), christine_amount = siri_amount + christine_has_more_than_siri) →
  ∃ total : ℝ, total = christine_amount + (christine_amount - christine_has_more_than_siri) ∧ total = 21 :=
by sorry

end NUMINAMATH_GPT_christine_siri_total_money_l1992_199274


namespace NUMINAMATH_GPT_root_k_value_l1992_199215

theorem root_k_value
  (k : ℝ)
  (h : Polynomial.eval 4 (Polynomial.C 2 * Polynomial.X^2 + Polynomial.C 3 * Polynomial.X - Polynomial.C k) = 0) :
  k = 44 :=
sorry

end NUMINAMATH_GPT_root_k_value_l1992_199215


namespace NUMINAMATH_GPT_gcf_72_108_l1992_199273

theorem gcf_72_108 : Nat.gcd 72 108 = 36 := by
  sorry

end NUMINAMATH_GPT_gcf_72_108_l1992_199273


namespace NUMINAMATH_GPT_sequence_value_l1992_199257

theorem sequence_value : 
  ∃ (x y r : ℝ), 
    (4096 * r = 1024) ∧ 
    (1024 * r = 256) ∧ 
    (256 * r = x) ∧ 
    (x * r = y) ∧ 
    (y * r = 4) ∧  
    (4 * r = 1) ∧ 
    (x + y = 80) :=
by
  sorry

end NUMINAMATH_GPT_sequence_value_l1992_199257


namespace NUMINAMATH_GPT_can_measure_all_weights_l1992_199298

theorem can_measure_all_weights (a b c : ℕ) 
  (h_sum : a + b + c = 10) 
  (h_unique : (a = 1 ∧ b = 2 ∧ c = 7) ∨ (a = 1 ∧ b = 3 ∧ c = 6)) : 
  ∀ w : ℕ, 1 ≤ w ∧ w ≤ 10 → 
    ∃ (k l m : ℤ), w = k * a + l * b + m * c ∨ w = k * -a + l * -b + m * -c :=
  sorry

end NUMINAMATH_GPT_can_measure_all_weights_l1992_199298


namespace NUMINAMATH_GPT_lowest_score_for_average_l1992_199280

theorem lowest_score_for_average
  (score1 score2 score3 : ℕ)
  (h1 : score1 = 81)
  (h2 : score2 = 72)
  (h3 : score3 = 93)
  (max_score : ℕ := 100)
  (desired_average : ℕ := 86)
  (number_of_exams : ℕ := 5) :
  ∃ x y : ℕ, x ≤ 100 ∧ y ≤ 100 ∧ (score1 + score2 + score3 + x + y) / number_of_exams = desired_average ∧ min x y = 84 :=
by
  sorry

end NUMINAMATH_GPT_lowest_score_for_average_l1992_199280


namespace NUMINAMATH_GPT_initial_apples_proof_l1992_199243

-- Define the variables and conditions
def initial_apples (handed_out: ℕ) (pies: ℕ) (apples_per_pie: ℕ): ℕ := 
  handed_out + pies * apples_per_pie

-- Define the proof statement
theorem initial_apples_proof : initial_apples 30 7 8 = 86 := by 
  sorry

end NUMINAMATH_GPT_initial_apples_proof_l1992_199243


namespace NUMINAMATH_GPT_pictures_per_album_l1992_199272

-- Definitions based on the conditions
def phone_pics := 35
def camera_pics := 5
def total_pics := phone_pics + camera_pics
def albums := 5 

-- Statement that needs to be proven
theorem pictures_per_album : total_pics / albums = 8 := by
  sorry

end NUMINAMATH_GPT_pictures_per_album_l1992_199272


namespace NUMINAMATH_GPT_element_in_set_l1992_199291

open Set

noncomputable def A : Set ℝ := { x | x < 2 * Real.sqrt 3 }
def a : ℝ := 2

theorem element_in_set : a ∈ A := by
  sorry

end NUMINAMATH_GPT_element_in_set_l1992_199291


namespace NUMINAMATH_GPT_range_of_function_l1992_199247

theorem range_of_function :
  ∀ y : ℝ, (∃ x : ℝ, y = (1 / 2) ^ (x^2 + 2 * x - 1)) ↔ (0 < y ∧ y ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_function_l1992_199247


namespace NUMINAMATH_GPT_units_digit_of_150_factorial_is_zero_l1992_199242

theorem units_digit_of_150_factorial_is_zero : (Nat.factorial 150) % 10 = 0 := by
sorry

end NUMINAMATH_GPT_units_digit_of_150_factorial_is_zero_l1992_199242


namespace NUMINAMATH_GPT_cars_difference_proof_l1992_199212

theorem cars_difference_proof (U M : ℕ) :
  let initial_cars := 150
  let total_cars := 196
  let cars_from_uncle := U
  let cars_from_grandpa := 2 * U
  let cars_from_dad := 10
  let cars_from_auntie := U + 1
  let cars_from_mum := M
  let total_given_cars := cars_from_dad + cars_from_auntie + cars_from_uncle + cars_from_grandpa + cars_from_mum
  initial_cars + total_given_cars = total_cars ->
  (cars_from_mum - cars_from_dad = 5) := 
by
  sorry

end NUMINAMATH_GPT_cars_difference_proof_l1992_199212


namespace NUMINAMATH_GPT_find_p_fifth_plus_3_l1992_199286

theorem find_p_fifth_plus_3 (p : ℕ) (hp : Nat.Prime p) (h : Nat.Prime (p^4 + 3)) :
  p^5 + 3 = 35 :=
sorry

end NUMINAMATH_GPT_find_p_fifth_plus_3_l1992_199286


namespace NUMINAMATH_GPT_price_per_foot_of_fencing_l1992_199281

theorem price_per_foot_of_fencing
  (area : ℝ) (total_cost : ℝ) (price_per_foot : ℝ)
  (h1 : area = 36) (h2 : total_cost = 1392) :
  price_per_foot = 58 :=
by
  sorry

end NUMINAMATH_GPT_price_per_foot_of_fencing_l1992_199281


namespace NUMINAMATH_GPT_carpet_area_l1992_199255

def width : ℝ := 8
def length : ℝ := 1.5

theorem carpet_area : width * length = 12 := by
  sorry

end NUMINAMATH_GPT_carpet_area_l1992_199255


namespace NUMINAMATH_GPT_total_cost_maria_l1992_199239

-- Define the cost of the pencil
def cost_pencil : ℕ := 8

-- Define the cost of the pen as half the price of the pencil
def cost_pen : ℕ := cost_pencil / 2

-- Define the total cost for both the pen and the pencil
def total_cost : ℕ := cost_pencil + cost_pen

-- Prove that total cost is equal to 12
theorem total_cost_maria : total_cost = 12 := 
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_total_cost_maria_l1992_199239


namespace NUMINAMATH_GPT_gwendolyn_read_time_l1992_199200

theorem gwendolyn_read_time :
  let rate := 200 -- sentences per hour
  let paragraphs_per_page := 30
  let sentences_per_paragraph := 15
  let pages := 100
  let sentences_per_page := sentences_per_paragraph * paragraphs_per_page
  let total_sentences := sentences_per_page * pages
  let total_time := total_sentences / rate
  total_time = 225 :=
by
  sorry

end NUMINAMATH_GPT_gwendolyn_read_time_l1992_199200


namespace NUMINAMATH_GPT_minimum_value_of_T_l1992_199237

theorem minimum_value_of_T (a b c : ℝ) (h1 : ∀ x : ℝ, (1 / a) * x^2 + b * x + c ≥ 0) (h2 : a * b > 1) :
  ∃ T : ℝ, T = 4 ∧ T = (1 / (2 * (a * b - 1))) + (a * (b + 2 * c) / (a * b - 1)) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_T_l1992_199237


namespace NUMINAMATH_GPT_quadratic_equal_roots_l1992_199268

theorem quadratic_equal_roots (a : ℝ) :
  (∃ x : ℝ, x ≠ 0 ∧ (x * (x + 1) + a * x = 0) ∧ ((1 + a)^2 = 0)) →
  a = -1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equal_roots_l1992_199268


namespace NUMINAMATH_GPT_total_pay_is_correct_l1992_199277

-- Define the weekly pay for employee B
def pay_B : ℝ := 228

-- Define the multiplier for employee A's pay relative to employee B's pay
def multiplier_A : ℝ := 1.5

-- Define the weekly pay for employee A
def pay_A : ℝ := multiplier_A * pay_B

-- Define the total weekly pay for both employees
def total_pay : ℝ := pay_A + pay_B

-- Prove the total pay
theorem total_pay_is_correct : total_pay = 570 := by
  -- Use the definitions and compute the total pay
  sorry

end NUMINAMATH_GPT_total_pay_is_correct_l1992_199277


namespace NUMINAMATH_GPT_train_length_calculation_l1992_199203

noncomputable def length_of_train (time : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  speed_ms * time

theorem train_length_calculation : 
  length_of_train 4.99960003199744 72 = 99.9920006399488 :=
by 
  sorry  -- proof of the actual calculation

end NUMINAMATH_GPT_train_length_calculation_l1992_199203


namespace NUMINAMATH_GPT_quadratic_solution_l1992_199258

theorem quadratic_solution (x : ℝ) (h : x^2 - 4 * x + 2 = 0) : x + 2 / x = 4 :=
by sorry

end NUMINAMATH_GPT_quadratic_solution_l1992_199258


namespace NUMINAMATH_GPT_min_value_x_plus_2_div_x_minus_2_l1992_199233

theorem min_value_x_plus_2_div_x_minus_2 (x : ℝ) (h : x > 2) : 
  ∃ m, m = 2 + 2 * Real.sqrt 2 ∧ x + 2/(x-2) ≥ m :=
by sorry

end NUMINAMATH_GPT_min_value_x_plus_2_div_x_minus_2_l1992_199233


namespace NUMINAMATH_GPT_fractional_sum_identity_l1992_199249

noncomputable def distinct_real_roots (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem fractional_sum_identity :
  ∀ (p q r A B C : ℝ),
  (x^3 - 22*x^2 + 80*x - 67 = (x - p) * (x - q) * (x - r)) →
  distinct_real_roots (λ x => x^3 - 22*x^2 + 80*x - 67) p q r →
  (∀ (s : ℝ), s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 22*s^2 + 80*s - 67) = A / (s - p) + B / (s - q) + C / (s - r)) →
  (1 / (A) + 1 / (B) + 1 / (C) = 244) :=
by 
  intros p q r A B C h_poly h_distinct h_fractional
  sorry

end NUMINAMATH_GPT_fractional_sum_identity_l1992_199249


namespace NUMINAMATH_GPT_max_xy_under_constraint_l1992_199238

theorem max_xy_under_constraint (x y : ℝ) (h1 : x + 2 * y = 1) (h2 : x > 0) (h3 : y > 0) : 
  xy ≤ 1 / 8 
  := sorry

end NUMINAMATH_GPT_max_xy_under_constraint_l1992_199238


namespace NUMINAMATH_GPT_find_y_l1992_199253

theorem find_y (x y : ℝ) (h1 : x - y = 20) (h2 : x + y = 10) : y = -5 := 
sorry

end NUMINAMATH_GPT_find_y_l1992_199253


namespace NUMINAMATH_GPT_larger_number_is_seventy_two_l1992_199234

def five_times_larger_is_six_times_smaller (x y : ℕ) : Prop := 5 * y = 6 * x
def difference_is_twelve (x y : ℕ) : Prop := y - x = 12

theorem larger_number_is_seventy_two (x y : ℕ) 
  (h1 : five_times_larger_is_six_times_smaller x y)
  (h2 : difference_is_twelve x y) : y = 72 :=
sorry

end NUMINAMATH_GPT_larger_number_is_seventy_two_l1992_199234


namespace NUMINAMATH_GPT_sum_of_squares_l1992_199235

theorem sum_of_squares (a b : ℝ) (h1 : a + b = 16) (h2 : a * b = 20) : a^2 + b^2 = 216 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l1992_199235


namespace NUMINAMATH_GPT_ratio_of_girls_to_boys_l1992_199267

variables (g b : ℕ)

theorem ratio_of_girls_to_boys (h₁ : b = g - 6) (h₂ : g + b = 36) :
  (g / gcd g b) / (b / gcd g b) = 7 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_girls_to_boys_l1992_199267


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1992_199290

theorem isosceles_triangle_perimeter (a b : ℝ) (h₁ : a = 6) (h₂ : b = 5) :
  ∃ p : ℝ, (p = a + a + b ∨ p = b + b + a) ∧ (p = 16 ∨ p = 17) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1992_199290


namespace NUMINAMATH_GPT_cut_wire_l1992_199229

theorem cut_wire (x y : ℕ) : 
  15 * x + 12 * y = 102 ↔ (x = 2 ∧ y = 6) ∨ (x = 6 ∧ y = 1) :=
by
  sorry

end NUMINAMATH_GPT_cut_wire_l1992_199229


namespace NUMINAMATH_GPT_gcd_228_1995_l1992_199271

theorem gcd_228_1995 : Int.gcd 228 1995 = 57 := by
  sorry

end NUMINAMATH_GPT_gcd_228_1995_l1992_199271


namespace NUMINAMATH_GPT_number_of_girls_in_first_year_l1992_199223

theorem number_of_girls_in_first_year
  (total_students : ℕ)
  (sample_size : ℕ)
  (boys_in_sample : ℕ)
  (girls_in_first_year : ℕ) :
  total_students = 2400 →
  sample_size = 80 →
  boys_in_sample = 42 →
  girls_in_first_year = total_students * (sample_size - boys_in_sample) / sample_size →
  girls_in_first_year = 1140 :=
by 
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_number_of_girls_in_first_year_l1992_199223


namespace NUMINAMATH_GPT_emily_required_sixth_score_is_99_l1992_199295

/-- Emily's quiz scores and the required mean score -/
def emily_scores : List ℝ := [85, 90, 88, 92, 98]
def required_mean_score : ℝ := 92

/-- The function to calculate the required sixth quiz score for Emily -/
def required_sixth_score (scores : List ℝ) (mean : ℝ) : ℝ :=
  let sum_current := scores.sum
  let total_required := mean * (scores.length + 1)
  total_required - sum_current

/-- Emily needs to score 99 on her sixth quiz for an average of 92 -/
theorem emily_required_sixth_score_is_99 : 
  required_sixth_score emily_scores required_mean_score = 99 :=
by
  sorry

end NUMINAMATH_GPT_emily_required_sixth_score_is_99_l1992_199295


namespace NUMINAMATH_GPT_european_savings_correct_l1992_199206

noncomputable def movie_ticket_price : ℝ := 8
noncomputable def popcorn_price : ℝ := 8 - 3
noncomputable def drink_price : ℝ := popcorn_price + 1
noncomputable def candy_price : ℝ := drink_price / 2
noncomputable def hotdog_price : ℝ := 5

noncomputable def monday_discount_popcorn : ℝ := 0.15 * popcorn_price
noncomputable def wednesday_discount_candy : ℝ := 0.10 * candy_price
noncomputable def friday_discount_drink : ℝ := 0.05 * drink_price

noncomputable def monday_price : ℝ := 22
noncomputable def wednesday_price : ℝ := 20
noncomputable def friday_price : ℝ := 25
noncomputable def weekend_price : ℝ := 25
noncomputable def monday_exchange_rate : ℝ := 0.85
noncomputable def wednesday_exchange_rate : ℝ := 0.85
noncomputable def friday_exchange_rate : ℝ := 0.83
noncomputable def weekend_exchange_rate : ℝ := 0.81

noncomputable def total_cost_monday : ℝ := movie_ticket_price + (popcorn_price - monday_discount_popcorn) + drink_price + candy_price + hotdog_price
noncomputable def savings_monday_usd : ℝ := total_cost_monday - monday_price
noncomputable def savings_monday_eur : ℝ := savings_monday_usd * monday_exchange_rate

noncomputable def total_cost_wednesday : ℝ := movie_ticket_price + popcorn_price + drink_price + (candy_price - wednesday_discount_candy) + hotdog_price
noncomputable def savings_wednesday_usd : ℝ := total_cost_wednesday - wednesday_price
noncomputable def savings_wednesday_eur : ℝ := savings_wednesday_usd * wednesday_exchange_rate

noncomputable def total_cost_friday : ℝ := movie_ticket_price + popcorn_price + (drink_price - friday_discount_drink) + candy_price + hotdog_price
noncomputable def savings_friday_usd : ℝ := total_cost_friday - friday_price
noncomputable def savings_friday_eur : ℝ := savings_friday_usd * friday_exchange_rate

noncomputable def total_cost_weekend : ℝ := movie_ticket_price + popcorn_price + drink_price + candy_price + hotdog_price
noncomputable def savings_weekend_usd : ℝ := total_cost_weekend - weekend_price
noncomputable def savings_weekend_eur : ℝ := savings_weekend_usd * weekend_exchange_rate

theorem european_savings_correct :
  savings_monday_eur = 3.61 ∧ 
  savings_wednesday_eur = 5.70 ∧ 
  savings_friday_eur = 1.41 ∧ 
  savings_weekend_eur = 1.62 :=
by
  sorry

end NUMINAMATH_GPT_european_savings_correct_l1992_199206


namespace NUMINAMATH_GPT_four_positive_reals_inequality_l1992_199294

theorem four_positive_reals_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a^3 + b^3 + c^3 + d^3 ≥ a^2 * b + b^2 * c + c^2 * d + d^2 * a :=
sorry

end NUMINAMATH_GPT_four_positive_reals_inequality_l1992_199294


namespace NUMINAMATH_GPT_division_of_fractions_l1992_199263

theorem division_of_fractions :
  (10 / 21) / (4 / 9) = 15 / 14 :=
by
  -- Proof will be provided here 
  sorry

end NUMINAMATH_GPT_division_of_fractions_l1992_199263


namespace NUMINAMATH_GPT_lines_intersect_and_not_perpendicular_l1992_199285

theorem lines_intersect_and_not_perpendicular (a : ℝ) :
  (∃ (x y : ℝ), 3 * x + 3 * y + a = 0 ∧ 3 * x - 2 * y + 1 = 0) ∧ 
  ¬ (∃ k1 k2 : ℝ, k1 = -1 ∧ k2 = 3 / 2 ∧ k1 ≠ k2 ∧ k1 * k2 = -1) :=
by
  sorry

end NUMINAMATH_GPT_lines_intersect_and_not_perpendicular_l1992_199285


namespace NUMINAMATH_GPT_ratio_jerky_l1992_199205

/-
  Given conditions:
  1. Janette camps for 5 days.
  2. She has an initial 40 pieces of beef jerky.
  3. She eats 4 pieces of beef jerky per day.
  4. She will have 10 pieces of beef jerky left after giving some to her brother.

  Prove that the ratio of the pieces of beef jerky she gives to her brother 
  to the remaining pieces is 1:1.
-/

theorem ratio_jerky (days : ℕ) (initial_jerky : ℕ) (jerky_per_day : ℕ) (jerky_left_after_trip : ℕ)
  (h1 : days = 5) (h2 : initial_jerky = 40) (h3 : jerky_per_day = 4) (h4 : jerky_left_after_trip = 10) :
  (initial_jerky - days * jerky_per_day - jerky_left_after_trip) = jerky_left_after_trip :=
by
  sorry

end NUMINAMATH_GPT_ratio_jerky_l1992_199205


namespace NUMINAMATH_GPT_probability_red_or_white_l1992_199265

-- Definitions based on the conditions
def total_marbles := 20
def blue_marbles := 5
def red_marbles := 9
def white_marbles := total_marbles - (blue_marbles + red_marbles)

-- Prove that the probability of selecting a red or white marble is 3/4
theorem probability_red_or_white : (red_marbles + white_marbles : ℚ) / total_marbles = 3 / 4 :=
by sorry

end NUMINAMATH_GPT_probability_red_or_white_l1992_199265


namespace NUMINAMATH_GPT_total_students_l1992_199217

theorem total_students (students_per_classroom : ℕ) (num_classrooms : ℕ) (h1 : students_per_classroom = 30) (h2 : num_classrooms = 13) : students_per_classroom * num_classrooms = 390 :=
by
  -- Begin the proof
  sorry

end NUMINAMATH_GPT_total_students_l1992_199217


namespace NUMINAMATH_GPT_number_of_roses_picked_later_l1992_199214

-- Given definitions
def initial_roses : ℕ := 50
def sold_roses : ℕ := 15
def final_roses : ℕ := 56

-- Compute the number of roses left after selling.
def roses_left := initial_roses - sold_roses

-- Define the final goal: number of roses picked later.
def picked_roses_later := final_roses - roses_left

-- State the theorem
theorem number_of_roses_picked_later : picked_roses_later = 21 :=
by
  sorry

end NUMINAMATH_GPT_number_of_roses_picked_later_l1992_199214


namespace NUMINAMATH_GPT_B_alone_can_do_work_in_9_days_l1992_199225

-- Define the conditions
def A_completes_work_in : ℕ := 15
def A_completes_portion_in (days : ℕ) : ℚ := days / 15
def portion_of_work_left (days : ℕ) : ℚ := 1 - A_completes_portion_in days
def B_completes_remaining_work_in_left_days (days_left : ℕ) : ℕ := 6
def B_completes_work_in (days_left : ℕ) : ℚ := B_completes_remaining_work_in_left_days days_left / (portion_of_work_left 5)

-- Define the theorem to be proven
theorem B_alone_can_do_work_in_9_days (days_left : ℕ) : B_completes_work_in days_left = 9 := by
  sorry

end NUMINAMATH_GPT_B_alone_can_do_work_in_9_days_l1992_199225


namespace NUMINAMATH_GPT_distance_travelled_l1992_199231

variables (S D : ℝ)

-- conditions
def cond1 : Prop := D = S * 7
def cond2 : Prop := D = (S + 12) * 5

-- Define the main theorem
theorem distance_travelled (h1 : cond1 S D) (h2 : cond2 S D) : D = 210 :=
by {
  sorry
}

end NUMINAMATH_GPT_distance_travelled_l1992_199231


namespace NUMINAMATH_GPT_average_expenditure_whole_week_l1992_199230

theorem average_expenditure_whole_week (a b : ℕ) (h₁ : a = 3 * 350) (h₂ : b = 4 * 420) : 
  (a + b) / 7 = 390 :=
by 
  sorry

end NUMINAMATH_GPT_average_expenditure_whole_week_l1992_199230


namespace NUMINAMATH_GPT_moles_of_hcl_l1992_199210

-- Definitions according to the conditions
def methane := 1 -- 1 mole of methane (CH₄)
def chlorine := 2 -- 2 moles of chlorine (Cl₂)
def hcl := 1 -- The expected number of moles of Hydrochloric acid (HCl)

-- The Lean 4 statement (no proof required)
theorem moles_of_hcl (methane chlorine : ℕ) : hcl = 1 :=
by sorry

end NUMINAMATH_GPT_moles_of_hcl_l1992_199210


namespace NUMINAMATH_GPT_cylinder_is_defined_sphere_is_defined_hyperbolic_cylinder_is_defined_parabolic_cylinder_is_defined_l1992_199251

-- 1) Cylinder
theorem cylinder_is_defined (R : ℝ) :
  ∀ (x y z : ℝ), x^2 + y^2 = R^2 → ∃ (r : ℝ), r = R ∧ x^2 + y^2 = r^2 :=
sorry

-- 2) Sphere
theorem sphere_is_defined (R : ℝ) :
  ∀ (x y z : ℝ), x^2 + y^2 + z^2 = R^2 → ∃ (r : ℝ), r = R ∧ x^2 + y^2 + z^2 = r^2 :=
sorry

-- 3) Hyperbolic Cylinder
theorem hyperbolic_cylinder_is_defined (m : ℝ) :
  ∀ (x y z : ℝ), xy = m → ∃ (k : ℝ), k = m ∧ xy = k :=
sorry

-- 4) Parabolic Cylinder
theorem parabolic_cylinder_is_defined :
  ∀ (x z : ℝ), z = x^2 → ∃ (k : ℝ), k = 1 ∧ z = k*x^2 :=
sorry

end NUMINAMATH_GPT_cylinder_is_defined_sphere_is_defined_hyperbolic_cylinder_is_defined_parabolic_cylinder_is_defined_l1992_199251


namespace NUMINAMATH_GPT_determine_f_101_l1992_199276

theorem determine_f_101 (f : ℕ → ℕ) (h : ∀ m n : ℕ, m * n + 1 ∣ f m * f n + 1) : 
  ∃ k : ℕ, k % 2 = 1 ∧ f 101 = 101 ^ k :=
sorry

end NUMINAMATH_GPT_determine_f_101_l1992_199276


namespace NUMINAMATH_GPT_andy_tomatoes_l1992_199296

theorem andy_tomatoes (P : ℕ) (h1 : ∀ P, 7 * P / 3 = 42) : P = 18 := by
  sorry

end NUMINAMATH_GPT_andy_tomatoes_l1992_199296


namespace NUMINAMATH_GPT_factor_polynomial_l1992_199224

theorem factor_polynomial (x y : ℝ) : 
  2*x^2 - x*y - 15*y^2 = (2*x - 5*y) * (x - 3*y) :=
sorry

end NUMINAMATH_GPT_factor_polynomial_l1992_199224


namespace NUMINAMATH_GPT_not_integer_division_l1992_199220

def P : ℕ := 1
def Q : ℕ := 2

theorem not_integer_division : ¬ (∃ (n : ℤ), (P : ℤ) / (Q : ℤ) = n) := by
sorry

end NUMINAMATH_GPT_not_integer_division_l1992_199220


namespace NUMINAMATH_GPT_final_cost_cooking_gear_sets_l1992_199284

-- Definitions based on conditions
def hand_mitts_cost : ℕ := 14
def apron_cost : ℕ := 16
def utensils_cost : ℕ := 10
def knife_cost : ℕ := 2 * utensils_cost
def discount_rate : ℚ := 0.25
def sales_tax_rate : ℚ := 0.08
def number_of_recipients : ℕ := 3 + 5

-- Proof statement: calculate the final cost
theorem final_cost_cooking_gear_sets :
  let total_cost_before_discount := hand_mitts_cost + apron_cost + utensils_cost + knife_cost
  let discounted_cost_per_set := (total_cost_before_discount : ℚ) * (1 - discount_rate)
  let total_cost_for_recipients := (discounted_cost_per_set * number_of_recipients : ℚ)
  let final_cost := total_cost_for_recipients * (1 + sales_tax_rate)
  final_cost = 388.80 :=
by
  sorry

end NUMINAMATH_GPT_final_cost_cooking_gear_sets_l1992_199284


namespace NUMINAMATH_GPT_find_t_l1992_199256

-- Definitions of the vectors and parallel condition
def a : ℝ × ℝ := (-1, 1)
def b (t : ℝ) : ℝ × ℝ := (3, t)
def is_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = k • v ∨ v = k • u

-- The theorem statement
theorem find_t (t : ℝ) (h : is_parallel (b t) (a + b t)) : t = -3 := by
  sorry

end NUMINAMATH_GPT_find_t_l1992_199256


namespace NUMINAMATH_GPT_gear_revolutions_difference_l1992_199264

noncomputable def gear_revolution_difference (t : ℕ) : ℕ :=
  let p := 10 * t
  let q := 40 * t
  q - p

theorem gear_revolutions_difference (t : ℕ) : gear_revolution_difference t = 30 * t :=
by
  sorry

end NUMINAMATH_GPT_gear_revolutions_difference_l1992_199264


namespace NUMINAMATH_GPT_Lisa_days_l1992_199204

theorem Lisa_days (L : ℝ) (h : 1/4 + 1/2 + 1/L = 1/1.09090909091) : L = 2.93333333333 :=
by sorry

end NUMINAMATH_GPT_Lisa_days_l1992_199204


namespace NUMINAMATH_GPT_expected_value_shorter_gentlemen_l1992_199207

-- Definitions based on the problem conditions
def expected_shorter_gentlemen (n : ℕ) : ℚ :=
  (n - 1) / 2

-- The main theorem statement based on the problem translation
theorem expected_value_shorter_gentlemen (n : ℕ) : 
  expected_shorter_gentlemen n = (n - 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_expected_value_shorter_gentlemen_l1992_199207


namespace NUMINAMATH_GPT_calculate_expression_value_l1992_199248

theorem calculate_expression_value : 
  3 - ((-3 : ℚ) ^ (-3 : ℤ) * 2) = 83 / 27 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_value_l1992_199248


namespace NUMINAMATH_GPT_symmetry_axis_of_sine_function_l1992_199219

theorem symmetry_axis_of_sine_function (x : ℝ) :
  (∃ k : ℤ, 2 * x + π / 4 = k * π + π / 2) ↔ x = π / 8 :=
by sorry

end NUMINAMATH_GPT_symmetry_axis_of_sine_function_l1992_199219


namespace NUMINAMATH_GPT_simplify_expression_l1992_199213

theorem simplify_expression :
  (Real.sqrt 15 + Real.sqrt 45 - (Real.sqrt (4/3) - Real.sqrt 108)) = 
  (Real.sqrt 15 + 3 * Real.sqrt 5 + 16 * Real.sqrt 3 / 3) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1992_199213


namespace NUMINAMATH_GPT_initial_erasers_calculation_l1992_199261

variable (initial_erasers added_erasers total_erasers : ℕ)

theorem initial_erasers_calculation
  (total_erasers_eq : total_erasers = 270)
  (added_erasers_eq : added_erasers = 131) :
  initial_erasers = total_erasers - added_erasers → initial_erasers = 139 := by
  intro h
  rw [total_erasers_eq, added_erasers_eq] at h
  simp at h
  exact h

end NUMINAMATH_GPT_initial_erasers_calculation_l1992_199261


namespace NUMINAMATH_GPT_largest_6_digit_div_by_88_l1992_199252

theorem largest_6_digit_div_by_88 : ∃ n : ℕ, 100000 ≤ n ∧ n ≤ 999999 ∧ 88 ∣ n ∧ (∀ m : ℕ, 100000 ≤ m ∧ m ≤ 999999 ∧ 88 ∣ m → m ≤ n) ∧ n = 999944 :=
by
  sorry

end NUMINAMATH_GPT_largest_6_digit_div_by_88_l1992_199252


namespace NUMINAMATH_GPT_gcd_n_squared_plus_4_n_plus_3_l1992_199250

theorem gcd_n_squared_plus_4_n_plus_3 (n : ℕ) (hn_gt_four : n > 4) : 
  (gcd (n^2 + 4) (n + 3)) = if n % 13 = 10 then 13 else 1 := 
sorry

end NUMINAMATH_GPT_gcd_n_squared_plus_4_n_plus_3_l1992_199250


namespace NUMINAMATH_GPT_remainder_sum_of_squares_25_mod_6_l1992_199279

def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

theorem remainder_sum_of_squares_25_mod_6 :
  (sum_of_squares 25) % 6 = 5 :=
by
  sorry

end NUMINAMATH_GPT_remainder_sum_of_squares_25_mod_6_l1992_199279


namespace NUMINAMATH_GPT_find_b_value_l1992_199218

theorem find_b_value
  (b : ℝ)
  (eq1 : ∀ y x, 3 * y - 3 * b = 9 * x)
  (eq2 : ∀ y x, y - 2 = (b + 9) * x)
  (parallel : ∀ y1 y2 x1 x2, 
    (3 * y1 - 3 * b = 9 * x1) ∧ (y2 - 2 = (b + 9) * x2) → 
    ((3 * x1 = (b + 9) * x2) ↔ (3 = b + 9)))
  : b = -6 := 
  sorry

end NUMINAMATH_GPT_find_b_value_l1992_199218


namespace NUMINAMATH_GPT_intersection_A_B_l1992_199275

def A := {x : ℤ | ∃ k : ℤ, x = 2 * k + 1}
def B := {x : ℤ | 0 < x ∧ x < 5}

theorem intersection_A_B : A ∩ B = {1, 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1992_199275


namespace NUMINAMATH_GPT_tan_150_eq_neg_inv_sqrt_3_l1992_199278

theorem tan_150_eq_neg_inv_sqrt_3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 := by
  -- Using the given conditions
  sorry

end NUMINAMATH_GPT_tan_150_eq_neg_inv_sqrt_3_l1992_199278


namespace NUMINAMATH_GPT_mike_spent_l1992_199246

def trumpet_price : ℝ := 145.16
def song_book_price : ℝ := 5.84
def total_price : ℝ := 151.00

theorem mike_spent :
  trumpet_price + song_book_price = total_price :=
by
  sorry

end NUMINAMATH_GPT_mike_spent_l1992_199246


namespace NUMINAMATH_GPT_platform_length_is_correct_l1992_199244

noncomputable def length_of_platform (time_to_pass_man : ℝ) (time_to_cross_platform : ℝ) (length_of_train : ℝ) : ℝ := 
  length_of_train * time_to_cross_platform / time_to_pass_man - length_of_train

theorem platform_length_is_correct : length_of_platform 8 20 178 = 267 := 
  sorry

end NUMINAMATH_GPT_platform_length_is_correct_l1992_199244


namespace NUMINAMATH_GPT_alcohol_percentage_in_second_vessel_l1992_199208

open Real

theorem alcohol_percentage_in_second_vessel (x : ℝ) (h : (0.2 * 2) + (0.01 * x * 6) = 8 * 0.28) : 
  x = 30.666666666666668 :=
by 
  sorry

end NUMINAMATH_GPT_alcohol_percentage_in_second_vessel_l1992_199208


namespace NUMINAMATH_GPT_sum_of_primes_less_than_twenty_is_77_l1992_199241

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_primes_less_than_twenty_is_77_l1992_199241


namespace NUMINAMATH_GPT_school_supply_cost_l1992_199293

theorem school_supply_cost (num_students : ℕ) (pens_per_student : ℕ) (pen_cost : ℝ) 
  (notebooks_per_student : ℕ) (notebook_cost : ℝ) 
  (binders_per_student : ℕ) (binder_cost : ℝ) 
  (highlighters_per_student : ℕ) (highlighter_cost : ℝ) 
  (teacher_discount : ℝ) : 
  num_students = 30 →
  pens_per_student = 5 →
  pen_cost = 0.50 →
  notebooks_per_student = 3 →
  notebook_cost = 1.25 →
  binders_per_student = 1 →
  binder_cost = 4.25 →
  highlighters_per_student = 2 →
  highlighter_cost = 0.75 →
  teacher_discount = 100 →
  (num_students * 
    (pens_per_student * pen_cost + notebooks_per_student * notebook_cost + 
    binders_per_student * binder_cost + highlighters_per_student * highlighter_cost) - 
    teacher_discount) = 260 :=
by
  intros _ _ _ _ _ _ _ _ _ _

  -- Sorry added to skip the proof
  sorry

end NUMINAMATH_GPT_school_supply_cost_l1992_199293


namespace NUMINAMATH_GPT_difference_students_guinea_pigs_l1992_199287

-- Define the conditions as constants
def students_per_classroom : Nat := 20
def guinea_pigs_per_classroom : Nat := 3
def number_of_classrooms : Nat := 6

-- Calculate the total number of students
def total_students : Nat := students_per_classroom * number_of_classrooms

-- Calculate the total number of guinea pigs
def total_guinea_pigs : Nat := guinea_pigs_per_classroom * number_of_classrooms

-- Define the theorem to prove the equality
theorem difference_students_guinea_pigs :
  total_students - total_guinea_pigs = 102 :=
by
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_difference_students_guinea_pigs_l1992_199287


namespace NUMINAMATH_GPT_ring_stack_vertical_distance_l1992_199269

theorem ring_stack_vertical_distance :
  let ring_thickness := 2
  let top_ring_outer_diameter := 36
  let bottom_ring_outer_diameter := 12
  let decrement := 2
  ∃ n, (top_ring_outer_diameter - bottom_ring_outer_diameter) / decrement + 1 = n ∧
       n * ring_thickness = 260 :=
by {
  let ring_thickness := 2
  let top_ring_outer_diameter := 36
  let bottom_ring_outer_diameter := 12
  let decrement := 2
  sorry
}

end NUMINAMATH_GPT_ring_stack_vertical_distance_l1992_199269


namespace NUMINAMATH_GPT_pythagorean_set_A_l1992_199227

theorem pythagorean_set_A : 
  let x := Real.sqrt 2
  let y := Real.sqrt 3
  let z := Real.sqrt 5
  x^2 + y^2 = z^2 := 
by
  let x := Real.sqrt 2
  let y := Real.sqrt 3
  let z := Real.sqrt 5
  sorry

end NUMINAMATH_GPT_pythagorean_set_A_l1992_199227


namespace NUMINAMATH_GPT_least_value_QGK_l1992_199245

theorem least_value_QGK :
  ∃ (G K Q : ℕ), (10 * G + G) * G = 100 * Q + 10 * G + K ∧ G ≠ K ∧ (10 * G + G) ≥ 10 ∧ (10 * G + G) < 100 ∧  ∃ x, x = 44 ∧ 100 * G + 10 * 4 + 4 = (100 * Q + 10 * G + K) ∧ 100 * 0 + 10 * 4 + 4 = 044  :=
by
  sorry

end NUMINAMATH_GPT_least_value_QGK_l1992_199245


namespace NUMINAMATH_GPT_Nathan_daily_hours_l1992_199201

theorem Nathan_daily_hours (x : ℝ) 
  (h1 : 14 * x + 35 = 77) : 
  x = 3 := 
by 
  sorry

end NUMINAMATH_GPT_Nathan_daily_hours_l1992_199201


namespace NUMINAMATH_GPT_oblique_projection_intuitive_diagrams_correct_l1992_199289

-- Definitions based on conditions
structure ObliqueProjection :=
  (lines_parallel_x_axis_same_length : Prop)
  (lines_parallel_y_axis_halved_length : Prop)
  (perpendicular_relationship_becomes_45_angle : Prop)

-- Definitions based on statements
def intuitive_triangle_projection (P : ObliqueProjection) : Prop :=
  P.lines_parallel_x_axis_same_length ∧ 
  P.lines_parallel_y_axis_halved_length ∧ 
  P.perpendicular_relationship_becomes_45_angle

def intuitive_parallelogram_projection (P : ObliqueProjection) : Prop := 
  P.lines_parallel_x_axis_same_length ∧ 
  P.lines_parallel_y_axis_halved_length ∧ 
  P.perpendicular_relationship_becomes_45_angle

def intuitive_square_projection (P : ObliqueProjection) : Prop := 
  P.lines_parallel_x_axis_same_length ∧ 
  P.lines_parallel_y_axis_halved_length ∧ 
  P.perpendicular_relationship_becomes_45_angle

def intuitive_rhombus_projection (P : ObliqueProjection) : Prop := 
  P.lines_parallel_x_axis_same_length ∧ 
  P.lines_parallel_y_axis_halved_length ∧ 
  P.perpendicular_relationship_becomes_45_angle

-- Theorem stating which intuitive diagrams are correctly represented under the oblique projection method.
theorem oblique_projection_intuitive_diagrams_correct : 
  ∀ (P : ObliqueProjection), 
    intuitive_triangle_projection P ∧ 
    intuitive_parallelogram_projection P ∧
    ¬intuitive_square_projection P ∧
    ¬intuitive_rhombus_projection P :=
by 
  sorry

end NUMINAMATH_GPT_oblique_projection_intuitive_diagrams_correct_l1992_199289


namespace NUMINAMATH_GPT_tori_current_height_l1992_199270

theorem tori_current_height :
  let original_height := 4.4
  let growth := 2.86
  original_height + growth = 7.26 := 
by
  sorry

end NUMINAMATH_GPT_tori_current_height_l1992_199270


namespace NUMINAMATH_GPT_algebraic_expression_value_l1992_199262

theorem algebraic_expression_value (m x n : ℝ)
  (h1 : (m + 3) * x ^ (|m| - 2) + 6 * m = 0)
  (h2 : n * x - 5 = x * (3 - n))
  (h3 : |m| = 2)
  (h4 : (m + 3) ≠ 0) :
  (m + x) ^ 2000 * (-m ^ 2 * n + x * n ^ 2) + 1 = 1 := by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1992_199262


namespace NUMINAMATH_GPT_students_per_van_l1992_199282

def number_of_boys : ℕ := 60
def number_of_girls : ℕ := 80
def number_of_vans : ℕ := 5

theorem students_per_van : (number_of_boys + number_of_girls) / number_of_vans = 28 := by
  sorry

end NUMINAMATH_GPT_students_per_van_l1992_199282


namespace NUMINAMATH_GPT_fraction_sum_l1992_199259

theorem fraction_sum : (1 / 4 : ℚ) + (3 / 8) = 5 / 8 :=
by
  sorry

end NUMINAMATH_GPT_fraction_sum_l1992_199259


namespace NUMINAMATH_GPT_value_of_xyz_l1992_199228

variable (x y z : ℝ)

theorem value_of_xyz (h1 : (x + y + z) * (x * y + x * z + y * z) = 36)
                     (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 22) 
                     : x * y * z = 14 / 3 := 
sorry

end NUMINAMATH_GPT_value_of_xyz_l1992_199228


namespace NUMINAMATH_GPT_sum_of_integers_product_neg17_l1992_199216

theorem sum_of_integers_product_neg17 (a b c : ℤ) (h : a * b * c = -17) : a + b + c = -15 ∨ a + b + c = 17 :=
sorry

end NUMINAMATH_GPT_sum_of_integers_product_neg17_l1992_199216


namespace NUMINAMATH_GPT_monotonic_interval_range_l1992_199288

noncomputable def f (a x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

theorem monotonic_interval_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < 2 → 1 < x₂ → x₂ < 2 → x₁ < x₂ → f a x₁ ≤ f a x₂ ∨ f a x₁ ≥ f a x₂) ↔
  (a ∈ Set.Iic (-1) ∪ Set.Ici 0) :=
sorry

end NUMINAMATH_GPT_monotonic_interval_range_l1992_199288


namespace NUMINAMATH_GPT_smallest_square_value_l1992_199211

theorem smallest_square_value (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h₁ : ∃ r : ℕ, 15 * a + 16 * b = r^2) (h₂ : ∃ s : ℕ, 16 * a - 15 * b = s^2) :
  ∃ (m : ℕ), m = 481^2 ∧ (15 * a + 16 * b = m ∨ 16 * a - 15 * b = m) :=
  sorry

end NUMINAMATH_GPT_smallest_square_value_l1992_199211


namespace NUMINAMATH_GPT_problem_l1992_199226

theorem problem (a b c : ℂ) 
  (h1 : a + b + c = 3)
  (h2 : a^2 + b^2 + c^2 = 3)
  (h3 : a^3 + b^3 + c^3 = 6) :
  (a - 1)^(2023) + (b - 1)^(2023) + (c - 1)^(2023) = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_l1992_199226


namespace NUMINAMATH_GPT_geometric_sequence_sum_of_first_four_terms_l1992_199266

theorem geometric_sequence_sum_of_first_four_terms 
  (a q : ℝ)
  (h1 : a * (1 + q) = 7)
  (h2 : a * (q^6 - 1) / (q - 1) = 91) :
  a * (1 + q + q^2 + q^3) = 28 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_of_first_four_terms_l1992_199266


namespace NUMINAMATH_GPT_rationalize_denominator_eq_l1992_199221

noncomputable def rationalize_denominator : ℝ :=
  18 / (Real.sqrt 36 + Real.sqrt 2)

theorem rationalize_denominator_eq : rationalize_denominator = (54 / 17) - (9 * Real.sqrt 2 / 17) := 
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_eq_l1992_199221


namespace NUMINAMATH_GPT_solve_frac_eq_l1992_199299

theorem solve_frac_eq (x : ℝ) (h : 3 - 5 / x + 2 / (x^2) = 0) : 
  ∃ y : ℝ, (y = 3 / x ∧ (y = 9 / 2 ∨ y = 3)) :=
sorry

end NUMINAMATH_GPT_solve_frac_eq_l1992_199299


namespace NUMINAMATH_GPT_sum_of_selected_primes_divisible_by_3_probability_l1992_199240

def first_fifteen_primes : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

def count_combinations_divisible_3 (nums : List ℕ) (k : ℕ) : ℕ :=
sorry -- Combines over the list to count combinations summing divisible by 3

noncomputable def probability_divisible_by_3 : ℚ :=
  let total_combinations := (Nat.choose 15 4)
  let favorable_combinations := count_combinations_divisible_3 first_fifteen_primes 4
  favorable_combinations / total_combinations

theorem sum_of_selected_primes_divisible_by_3_probability :
  probability_divisible_by_3 = 1/3 :=
sorry

end NUMINAMATH_GPT_sum_of_selected_primes_divisible_by_3_probability_l1992_199240


namespace NUMINAMATH_GPT_solution_to_system_l1992_199232

theorem solution_to_system : ∃ x y : ℤ, (2 * x + 3 * y = -11 ∧ 6 * x - 5 * y = 9) ↔ (x = -1 ∧ y = -3) :=
by
  sorry

end NUMINAMATH_GPT_solution_to_system_l1992_199232


namespace NUMINAMATH_GPT_choir_members_count_l1992_199292

theorem choir_members_count (n : ℕ) (h1 : n % 10 = 4) (h2 : n % 11 = 5) (h3 : 200 ≤ n) (h4 : n ≤ 300) : n = 234 := 
sorry

end NUMINAMATH_GPT_choir_members_count_l1992_199292


namespace NUMINAMATH_GPT_multiple_of_four_l1992_199209

theorem multiple_of_four (n : ℕ) (h1 : ∃ k : ℕ, 12 + 4 * k = n) (h2 : 21 = (n - 12) / 4 + 1) : n = 96 := 
sorry

end NUMINAMATH_GPT_multiple_of_four_l1992_199209


namespace NUMINAMATH_GPT_burger_cost_l1992_199254

theorem burger_cost
  (B P : ℝ)
  (h₁ : P = 2 * B)
  (h₂ : P + 3 * B = 45) :
  B = 9 := by
  sorry

end NUMINAMATH_GPT_burger_cost_l1992_199254


namespace NUMINAMATH_GPT_coeff_a_zero_l1992_199297

theorem coeff_a_zero
  (a b c : ℝ)
  (h : ∀ p : ℝ, 0 < p → ∀ (x : ℝ), (a * x^2 + b * x + c + p = 0) → x > 0) :
  a = 0 :=
sorry

end NUMINAMATH_GPT_coeff_a_zero_l1992_199297


namespace NUMINAMATH_GPT_avg_GPA_is_93_l1992_199222

def avg_GPA_school (GPA_6th GPA_8th : ℕ) (GPA_diff : ℕ) : ℕ :=
  (GPA_6th + (GPA_6th + GPA_diff) + GPA_8th) / 3

theorem avg_GPA_is_93 :
  avg_GPA_school 93 91 2 = 93 :=
by
  -- The proof can be handled here 
  sorry

end NUMINAMATH_GPT_avg_GPA_is_93_l1992_199222


namespace NUMINAMATH_GPT_number_of_clerks_l1992_199202

theorem number_of_clerks 
  (num_officers : ℕ) 
  (num_clerks : ℕ) 
  (avg_salary_staff : ℕ) 
  (avg_salary_officers : ℕ) 
  (avg_salary_clerks : ℕ)
  (h1 : avg_salary_staff = 90)
  (h2 : avg_salary_officers = 600)
  (h3 : avg_salary_clerks = 84)
  (h4 : num_officers = 2)
  : num_clerks = 170 :=
sorry

end NUMINAMATH_GPT_number_of_clerks_l1992_199202


namespace NUMINAMATH_GPT_jill_trips_to_fill_tank_l1992_199283

def tank_capacity : ℕ := 600
def bucket_volume : ℕ := 5
def jack_buckets_per_trip : ℕ := 2
def jill_buckets_per_trip : ℕ := 1
def jack_to_jill_trip_ratio : ℕ := 3 / 2

theorem jill_trips_to_fill_tank : (tank_capacity / bucket_volume) = 120 → 
                                   ((jack_to_jill_trip_ratio * jack_buckets_per_trip) + 2 * jill_buckets_per_trip) = 8 →
                                   15 * 2 = 30 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_jill_trips_to_fill_tank_l1992_199283
