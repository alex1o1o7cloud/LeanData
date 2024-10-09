import Mathlib

namespace sum_series_a_sum_series_b_sum_series_c_l2172_217224

-- Part (a)
theorem sum_series_a : (∑' n : ℕ, (1 / 2) ^ (n + 1)) = 1 := by
  --skip proof
  sorry

-- Part (b)
theorem sum_series_b : (∑' n : ℕ, (1 / 3) ^ (n + 1)) = 1/2 := by
  --skip proof
  sorry

-- Part (c)
theorem sum_series_c : (∑' n : ℕ, (1 / 4) ^ (n + 1)) = 1/3 := by
  --skip proof
  sorry

end sum_series_a_sum_series_b_sum_series_c_l2172_217224


namespace john_total_cost_l2172_217223

def base_cost : ℤ := 25
def text_cost_per_message : ℤ := 8
def extra_minute_cost_per_minute : ℤ := 15
def international_minute_cost : ℤ := 100

def texts_sent : ℤ := 200
def total_hours : ℤ := 42
def international_minutes : ℤ := 10

-- Calculate the number of extra minutes
def extra_minutes : ℤ := (total_hours - 40) * 60

noncomputable def total_cost : ℤ :=
  base_cost +
  (texts_sent * text_cost_per_message) / 100 +
  (extra_minutes * extra_minute_cost_per_minute) / 100 +
  international_minutes * (international_minute_cost / 100)

theorem john_total_cost :
  total_cost = 69 := by
    sorry

end john_total_cost_l2172_217223


namespace quadratic_inequality_solution_l2172_217260

theorem quadratic_inequality_solution 
  (a b c : ℝ)
  (h1 : ∀ x, -3 < x ∧ x < 1/2 ↔ cx^2 + bx + a < 0) :
  ∀ x, -1/3 ≤ x ∧ x ≤ 2 ↔ ax^2 + bx + c ≥ 0 :=
sorry

end quadratic_inequality_solution_l2172_217260


namespace find_x_l2172_217229

-- Define the conditions
def cherryGum := 25
def grapeGum := 35
def packs (x : ℚ) := x -- Each pack contains exactly x pieces of gum

-- Define the ratios after losing one pack of cherry gum and finding 6 packs of grape gum
def ratioAfterLosingCherryPack (x : ℚ) := (cherryGum - packs x) / grapeGum
def ratioAfterFindingGrapePacks (x : ℚ) := cherryGum / (grapeGum + 6 * packs x)

-- State the theorem to be proved
theorem find_x (x : ℚ) (h : ratioAfterLosingCherryPack x = ratioAfterFindingGrapePacks x) : x = 115 / 6 :=
by
  sorry

end find_x_l2172_217229


namespace man_speed_in_still_water_l2172_217247

theorem man_speed_in_still_water
  (vm vs : ℝ)
  (h1 : vm + vs = 6)  -- effective speed downstream
  (h2 : vm - vs = 4)  -- effective speed upstream
  : vm = 5 := 
by
  sorry

end man_speed_in_still_water_l2172_217247


namespace students_joined_l2172_217272

theorem students_joined (A X : ℕ) (h1 : 100 * A = 5000) (h2 : (100 + X) * (A - 10) = 5400) :
  X = 35 :=
by
  sorry

end students_joined_l2172_217272


namespace cost_of_building_fence_eq_3944_l2172_217259

def area_square : ℕ := 289
def price_per_foot : ℕ := 58

theorem cost_of_building_fence_eq_3944 : 
  let side_length := (area_square : ℝ) ^ (1/2)
  let perimeter := 4 * side_length
  let cost := perimeter * (price_per_foot : ℝ)
  cost = 3944 :=
by
  sorry

end cost_of_building_fence_eq_3944_l2172_217259


namespace pet_store_initial_gerbils_l2172_217297

-- Define sold gerbils
def sold_gerbils : ℕ := 69

-- Define left gerbils
def left_gerbils : ℕ := 16

-- Define the initial number of gerbils
def initial_gerbils : ℕ := sold_gerbils + left_gerbils

-- State the theorem to be proved
theorem pet_store_initial_gerbils : initial_gerbils = 85 := by
  -- This is where the proof would go
  sorry

end pet_store_initial_gerbils_l2172_217297


namespace repetend_of_frac_4_div_17_is_235294_l2172_217210

noncomputable def decimalRepetend_of_4_div_17 : String :=
  let frac := 4 / 17
  let repetend := "235294"
  repetend

theorem repetend_of_frac_4_div_17_is_235294 :
  (∃ n m : ℕ, (4 / 17 : ℚ) = n + (m / 10^6) ∧ m % 10^6 = 235294) :=
sorry

end repetend_of_frac_4_div_17_is_235294_l2172_217210


namespace both_hit_exactly_one_hits_at_least_one_hits_l2172_217282

noncomputable def prob_A : ℝ := 0.8
noncomputable def prob_B : ℝ := 0.9

theorem both_hit : prob_A * prob_B = 0.72 := by
  sorry

theorem exactly_one_hits : prob_A * (1 - prob_B) + (1 - prob_A) * prob_B = 0.26 := by
  sorry

theorem at_least_one_hits : 1 - (1 - prob_A) * (1 - prob_B) = 0.98 := by
  sorry

end both_hit_exactly_one_hits_at_least_one_hits_l2172_217282


namespace optimal_messenger_strategy_l2172_217296

theorem optimal_messenger_strategy (p : ℝ) (hp : 0 < p ∧ p < 1) :
  (p < 1/3 → ∃ n : ℕ, n = 4 ∧ ∀ (k : ℕ), k = 10) ∧ 
  (1/3 ≤ p → ∃ n : ℕ, n = 2 ∧ ∀ (m : ℕ), m = 20) :=
by
  sorry

end optimal_messenger_strategy_l2172_217296


namespace wall_length_l2172_217234

theorem wall_length (mirror_side length width : ℝ) (h_mirror : mirror_side = 21) (h_width : width = 28) 
  (h_area_relation : (mirror_side * mirror_side) * 2 = width * length) : length = 31.5 :=
by
  -- here you start the proof, but it's not required for the statement
  sorry

end wall_length_l2172_217234


namespace functional_equality_l2172_217261

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equality
  (h1 : ∀ x : ℝ, f x ≤ x)
  (h2 : ∀ x y : ℝ, f (x + y) ≤ f x + f y) :
  ∀ x : ℝ, f x = x :=
by
  sorry

end functional_equality_l2172_217261


namespace towers_remainder_l2172_217206

noncomputable def count_towers (k : ℕ) : ℕ := sorry

theorem towers_remainder : (count_towers 9) % 1000 = 768 := sorry

end towers_remainder_l2172_217206


namespace Cedar_school_earnings_l2172_217219

noncomputable def total_earnings_Cedar_school : ℝ :=
  let total_payment := 774
  let total_student_days := 6 * 4 + 5 * 6 + 3 * 10
  let daily_wage := total_payment / total_student_days
  let Cedar_student_days := 3 * 10
  daily_wage * Cedar_student_days

theorem Cedar_school_earnings :
  total_earnings_Cedar_school = 276.43 :=
by
  sorry

end Cedar_school_earnings_l2172_217219


namespace find_interest_rate_l2172_217212

noncomputable def interest_rate (total_investment remaining_investment interest_earned part_interest : ℝ) : ℝ :=
  (interest_earned - part_interest) / remaining_investment

theorem find_interest_rate :
  let total_investment := 9000
  let invested_at_8_percent := 4000
  let total_interest := 770
  let interest_at_8_percent := invested_at_8_percent * 0.08
  let remaining_investment := total_investment - invested_at_8_percent
  let interest_from_remaining := total_interest - interest_at_8_percent
  interest_rate total_investment remaining_investment total_interest interest_at_8_percent = 0.09 :=
by
  sorry

end find_interest_rate_l2172_217212


namespace alpha_beta_property_l2172_217203

theorem alpha_beta_property
  (α β : ℝ)
  (hαβ_roots : ∀ x : ℝ, (x = α ∨ x = β) → x^2 + x - 2023 = 0) :
  α^2 + 2 * α + β = 2022 :=
by
  sorry

end alpha_beta_property_l2172_217203


namespace min_value_of_expression_l2172_217242

theorem min_value_of_expression (x y : ℝ) (h : x^2 + y^2 + x * y = 315) :
  ∃ m : ℝ, m = x^2 + y^2 - x * y ∧ m ≥ 105 :=
by
  sorry

end min_value_of_expression_l2172_217242


namespace factorization_correct_l2172_217267

theorem factorization_correct (x y : ℝ) : 
  x^2 + y^2 + 2*x*y - 1 = (x + y + 1) * (x + y - 1) := 
by
  sorry

end factorization_correct_l2172_217267


namespace sum_of_two_numbers_l2172_217245

theorem sum_of_two_numbers (x y : ℕ) (hxy : x > y) (h1 : x - y = 4) (h2 : x * y = 156) : x + y = 28 :=
by {
  sorry
}

end sum_of_two_numbers_l2172_217245


namespace find_real_pairs_l2172_217243

theorem find_real_pairs (x y : ℝ) (h1 : x + y = 1) (h2 : x^3 + y^3 = 19) :
  (x = 3 ∧ y = -2) ∨ (x = -2 ∧ y = 3) :=
sorry

end find_real_pairs_l2172_217243


namespace man_completes_in_9_days_l2172_217244

-- Definitions of the work rates and the conditions given
def M : ℚ := sorry
def W : ℚ := 1 / 6
def B : ℚ := 1 / 18
def combined_rate : ℚ := 1 / 3

-- Statement that the man alone can complete the work in 9 days
theorem man_completes_in_9_days
  (h_combined : M + W + B = combined_rate) : 1 / M = 9 :=
  sorry

end man_completes_in_9_days_l2172_217244


namespace average_of_rstu_l2172_217208

theorem average_of_rstu (r s t u : ℝ) (h : (5 / 4) * (r + s + t + u) = 15) : (r + s + t + u) / 4 = 3 :=
by
  sorry

end average_of_rstu_l2172_217208


namespace simplify_and_evaluate_l2172_217227

theorem simplify_and_evaluate (a : ℚ) (h : a = -1/6) : 
  2 * (a + 1) * (a - 1) - a * (2 * a - 3) = -5 / 2 := by
  rw [h]
  sorry

end simplify_and_evaluate_l2172_217227


namespace odd_n_divides_pow_fact_sub_one_l2172_217274

theorem odd_n_divides_pow_fact_sub_one
  {n : ℕ} (hn_pos : n > 0) (hn_odd : n % 2 = 1)
  : n ∣ (2 ^ (Nat.factorial n) - 1) :=
sorry

end odd_n_divides_pow_fact_sub_one_l2172_217274


namespace expression_value_l2172_217257

theorem expression_value : (1 * 3 * 5 * 7) / (1^2 + 2^2 + 3^2 + 4^2) = 7 / 2 := by
  sorry

end expression_value_l2172_217257


namespace simplify_expression1_simplify_expression2_l2172_217284

-- Problem 1
theorem simplify_expression1 (x y : ℤ) :
  (-3) * x + 2 * y - 5 * x - 7 * y = -8 * x - 5 * y :=
by sorry

-- Problem 2
theorem simplify_expression2 (a b : ℤ) :
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) = 3 * a^2 * b - a * b^2 :=
by sorry

end simplify_expression1_simplify_expression2_l2172_217284


namespace valid_passwords_count_l2172_217205

def total_passwords : Nat := 10 ^ 5
def restricted_passwords : Nat := 10

theorem valid_passwords_count : total_passwords - restricted_passwords = 99990 := by
  sorry

end valid_passwords_count_l2172_217205


namespace courses_choice_l2172_217230

theorem courses_choice (total_courses : ℕ) (chosen_courses : ℕ)
  (h_total_courses : total_courses = 5)
  (h_chosen_courses : chosen_courses = 2) :
  ∃ (ways : ℕ), ways = 60 ∧
    (ways = ((Nat.choose total_courses chosen_courses)^2) - 
            (Nat.choose total_courses chosen_courses) - 
            ((Nat.choose total_courses chosen_courses) * 
             (Nat.choose (total_courses - chosen_courses) chosen_courses))) :=
by
  sorry

end courses_choice_l2172_217230


namespace parabola_condition_max_area_triangle_l2172_217238

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

theorem parabola_condition (p : ℝ) (h₀ : 0 < p) : 
  ((p / 2 + 4 - 1 = 4) → (p = 2)) :=
by sorry

theorem max_area_triangle (P : ℝ × ℝ) (k b : ℝ) 
  (h₀ : P.1 ^ 2 + (P.2 + 4) ^ 2 = 1) 
  (h₁ : P.1 = 2 * k) 
  (h₂ : -P.2 = b) 
  (h₃ : k ^ 2 + (b - 4) ^ 2 < 1) :
  4 * ((k ^ 2 + b) ^ (3 / 2)) = 20 * Real.sqrt 5 :=
by sorry

end parabola_condition_max_area_triangle_l2172_217238


namespace largest_consecutive_sum_is_nine_l2172_217279

-- Define the conditions: a sequence of positive consecutive integers summing to 45
def is_consecutive_sum (n k : ℕ) : Prop :=
  (k > 0) ∧ (n > 0) ∧ ((k * (2 * n + k - 1)) = 90)

-- The theorem statement proving k = 9 is the largest
theorem largest_consecutive_sum_is_nine :
  ∃ n k : ℕ, is_consecutive_sum n k ∧ ∀ k', is_consecutive_sum n k' → k' ≤ k :=
sorry

end largest_consecutive_sum_is_nine_l2172_217279


namespace sampling_prob_equal_l2172_217209

theorem sampling_prob_equal (N n : ℕ) (P_1 P_2 P_3 : ℝ)
  (H_random : ∀ i, 1 ≤ i ∧ i ≤ N → P_1 = 1 / N)
  (H_systematic : ∀ i, 1 ≤ i ∧ i ≤ N → P_2 = 1 / N)
  (H_stratified : ∀ i, 1 ≤ i ∧ i ≤ N → P_3 = 1 / N) :
  P_1 = P_2 ∧ P_2 = P_3 :=
by
  sorry

end sampling_prob_equal_l2172_217209


namespace acute_angle_at_315_equals_7_5_l2172_217250

/-- The degrees in a full circle -/
def fullCircle := 360

/-- The number of hours on a clock -/
def hoursOnClock := 12

/-- The measure in degrees of the acute angle formed by the minute hand and the hour hand at 3:15 -/
def acuteAngleAt315 : ℝ :=
  let degreesPerHour := fullCircle / hoursOnClock
  let hourHandAt3 := degreesPerHour * 3
  let additionalDegrees := (15 / 60) * degreesPerHour
  let hourHandPosition := hourHandAt3 + additionalDegrees
  let minuteHandPosition := (15 / 60) * fullCircle
  abs (hourHandPosition - minuteHandPosition)

theorem acute_angle_at_315_equals_7_5 : acuteAngleAt315 = 7.5 := by
  sorry

end acute_angle_at_315_equals_7_5_l2172_217250


namespace joe_total_paint_used_l2172_217246

-- Define the initial amount of paint Joe buys.
def initial_paint : ℕ := 360

-- Define the fraction of paint used during the first week.
def first_week_fraction := 1 / 4

-- Define the fraction of remaining paint used during the second week.
def second_week_fraction := 1 / 2

-- Define the total paint used by Joe in the first week.
def paint_used_first_week := first_week_fraction * initial_paint

-- Define the remaining paint after the first week.
def remaining_paint_after_first_week := initial_paint - paint_used_first_week

-- Define the total paint used by Joe in the second week.
def paint_used_second_week := second_week_fraction * remaining_paint_after_first_week

-- Define the total paint used by Joe.
def total_paint_used := paint_used_first_week + paint_used_second_week

-- The theorem to be proven: the total amount of paint Joe has used is 225 gallons.
theorem joe_total_paint_used : total_paint_used = 225 := by
  sorry

end joe_total_paint_used_l2172_217246


namespace magic_square_sum_l2172_217249

variable {a b c d e : ℕ}

-- Given conditions:
-- It's a magic square and the sums of the numbers in each row, column, and diagonal are equal.
-- Positions and known values specified:
theorem magic_square_sum (h : 15 + 24 = 18 + c ∧ 18 + c = 27 + a ∧ c = 21 ∧ a = 12 ∧ e = 17 ∧ d = 30 ∧ b = 25)
: d + e = 47 :=
by
  -- Sorry used to skip the proof
  sorry

end magic_square_sum_l2172_217249


namespace aladdin_no_profit_l2172_217216

theorem aladdin_no_profit (x : ℕ) :
  (x + 1023000) / 1024 <= x :=
by
  sorry

end aladdin_no_profit_l2172_217216


namespace cube_root_of_64_l2172_217214

theorem cube_root_of_64 : ∃ x : ℝ, x^3 = 64 ∧ x = 4 :=
by
  sorry

end cube_root_of_64_l2172_217214


namespace part1_l2172_217294

theorem part1 (a b c d : ℤ) (h : a * d - b * c = 1) : Int.gcd (a + b) (c + d) = 1 :=
sorry

end part1_l2172_217294


namespace option_C_is_always_odd_l2172_217289

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem option_C_is_always_odd (k : ℤ) : is_odd (2007 + 2 * k ^ 2) :=
sorry

end option_C_is_always_odd_l2172_217289


namespace sachin_age_l2172_217273
-- Import the necessary library

-- Lean statement defining the problem conditions and result
theorem sachin_age :
  ∃ (S R : ℝ), (R = S + 7) ∧ (S / R = 7 / 9) ∧ (S = 24.5) :=
by
  sorry

end sachin_age_l2172_217273


namespace repeating_decimal_fraction_l2172_217295

theorem repeating_decimal_fraction (x : ℚ) (h : x = 7.5656) : x = 749 / 99 :=
by
  sorry

end repeating_decimal_fraction_l2172_217295


namespace f_2009_l2172_217283

def f (x : ℝ) : ℝ := x^3 -- initial definition for x in [-1, 1]

axiom odd_function : ∀ x : ℝ, f (-x) = -f (x)
axiom symmetric_around_1 : ∀ x : ℝ, f (1 + x) = f (1 - x)
axiom f_cubed : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x = x^3

theorem f_2009 : f 2009 = 1 := by {
  -- The body of the theorem will be filled with proof steps
  sorry
}

end f_2009_l2172_217283


namespace intersection_M_N_eq_segment_l2172_217207

def M : Set ℝ := {x | x^2 + x - 6 < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N_eq_segment : M ∩ N = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_eq_segment_l2172_217207


namespace tangent_line_at_neg1_l2172_217237

-- Define the function given in the condition.
def f (x : ℝ) : ℝ := x^2 + 4 * x + 2

-- Define the point of tangency given in the condition.
def point_of_tangency : ℝ × ℝ := (-1, f (-1))

-- Define the derivative of the function.
def derivative_f (x : ℝ) : ℝ := 2 * x + 4

-- The proof statement: the equation of the tangent line at x = -1 is y = 2x + 1
theorem tangent_line_at_neg1 :
  ∃ (m b : ℝ), (∀ (x y : ℝ), y = f x → derivative_f (-1) = m ∧ point_of_tangency.fst = -1 ∧ y = m * (x + 1) + b) :=
sorry

end tangent_line_at_neg1_l2172_217237


namespace triangle_angle_bisectors_l2172_217268

theorem triangle_angle_bisectors {a b c : ℝ} (ht : (a = 2 ∧ b = 3 ∧ c < 5)) : 
  (∃ h_a h_b h_c : ℝ, h_a + h_b > h_c ∧ h_a + h_c > h_b ∧ h_b + h_c > h_a) →
  ¬ (∃ ell_a ell_b ell_c : ℝ, ell_a + ell_b > ell_c ∧ ell_a + ell_c > ell_b ∧ ell_b + ell_c > ell_a) :=
by
  sorry

end triangle_angle_bisectors_l2172_217268


namespace GCF_of_LCMs_l2172_217251

def GCF : ℕ → ℕ → ℕ := Nat.gcd
def LCM : ℕ → ℕ → ℕ := Nat.lcm

theorem GCF_of_LCMs :
  GCF (LCM 9 21) (LCM 10 15) = 3 :=
by
  sorry

end GCF_of_LCMs_l2172_217251


namespace inequality_always_holds_l2172_217217

theorem inequality_always_holds
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h_def : ∀ x, f x = (1 - 2^x) / (1 + 2^x))
  (h_odd : ∀ x, f (-x) = -f x)
  (h_decreasing : ∀ x y, x < y → f x > f y)
  (h_ineq : f (2 * a + b) + f (4 - 3 * b) > 0)
  : b - a > 2 :=
sorry

end inequality_always_holds_l2172_217217


namespace Tony_fever_l2172_217256

theorem Tony_fever :
  ∀ (normal_temp sickness_increase fever_threshold : ℕ),
    normal_temp = 95 →
    sickness_increase = 10 →
    fever_threshold = 100 →
    (normal_temp + sickness_increase) - fever_threshold = 5 :=
by
  intros normal_temp sickness_increase fever_threshold h1 h2 h3
  sorry

end Tony_fever_l2172_217256


namespace probability_all_operating_probability_shutdown_l2172_217255

-- Define the events and their probabilities
def P_A : ℝ := 0.9
def P_B : ℝ := 0.8
def P_C : ℝ := 0.85

-- Prove that the probability of all three machines operating without supervision is 0.612
theorem probability_all_operating : P_A * P_B * P_C = 0.612 := 
by sorry

-- Prove that the probability of a shutdown is 0.059
theorem probability_shutdown :
    P_A * (1 - P_B) * (1 - P_C) +
    (1 - P_A) * P_B * (1 - P_C) +
    (1 - P_A) * (1 - P_B) * P_C +
    (1 - P_A) * (1 - P_B) * (1 - P_C) = 0.059 :=
by sorry

end probability_all_operating_probability_shutdown_l2172_217255


namespace red_stars_eq_35_l2172_217233

-- Define the conditions
noncomputable def number_of_total_stars (x : ℕ) : ℕ := x + 20 + 15
noncomputable def red_star_frequency (x : ℕ) : ℚ := x / (number_of_total_stars x : ℚ)

-- Define the theorem statement
theorem red_stars_eq_35 : ∃ x : ℕ, red_star_frequency x = 0.5 ↔ x = 35 := sorry

end red_stars_eq_35_l2172_217233


namespace ratio_of_capitals_l2172_217266

-- Variables for the capitals of Ashok and Pyarelal
variables (A P : ℕ)

-- Given conditions
def total_loss := 670
def pyarelal_loss := 603
def ashok_loss := total_loss - pyarelal_loss

-- Proof statement: the ratio of Ashok's capital to Pyarelal's capital
theorem ratio_of_capitals : ashok_loss * P = total_loss * pyarelal_loss - pyarelal_loss * P → A * pyarelal_loss = P * ashok_loss :=
by
  sorry

end ratio_of_capitals_l2172_217266


namespace diagonal_cubes_140_320_360_l2172_217285

-- Define the problem parameters 
def length_x : ℕ := 140
def length_y : ℕ := 320
def length_z : ℕ := 360

-- Define the function to calculate the number of unit cubes the internal diagonal passes through.
def num_cubes_diagonal (x y z : ℕ) : ℕ :=
  x + y + z - Nat.gcd x y - Nat.gcd y z - Nat.gcd z x + Nat.gcd (Nat.gcd x y) z

-- The target theorem to be proven
theorem diagonal_cubes_140_320_360 :
  num_cubes_diagonal length_x length_y length_z = 760 :=
by
  sorry

end diagonal_cubes_140_320_360_l2172_217285


namespace rectangle_length_width_difference_l2172_217299

theorem rectangle_length_width_difference :
  ∃ (length width : ℕ), (length * width = 864) ∧ (length + width = 60) ∧ (length - width = 12) :=
by
  sorry

end rectangle_length_width_difference_l2172_217299


namespace circle_equation_l2172_217239

theorem circle_equation 
  (x y : ℝ)
  (passes_origin : (x, y) = (0, 0))
  (intersects_line : ∃ (x y : ℝ), 2 * x - y + 1 = 0)
  (intersects_circle : ∃ (x y :ℝ), x^2 + y^2 - 2 * x - 15 = 0) : 
  x^2 + y^2 + 28 * x - 15 * y = 0 :=
sorry

end circle_equation_l2172_217239


namespace evaluate_dollar_l2172_217281

variable {R : Type} [CommRing R]

def dollar (a b : R) : R := (a - b) ^ 2

theorem evaluate_dollar (x y : R) : 
  dollar (x^2 - y^2) (y^2 - x^2) = 4 * (x^4 - 2 * x^2 * y^2 + y^4) :=
by
  sorry

end evaluate_dollar_l2172_217281


namespace fold_string_twice_l2172_217228

theorem fold_string_twice (initial_length : ℕ) (half_folds : ℕ) (result_length : ℕ) 
  (h1 : initial_length = 12)
  (h2 : half_folds = 2)
  (h3 : result_length = initial_length / (2 ^ half_folds)) :
  result_length = 3 := 
by
  -- This is where the proof would go
  sorry

end fold_string_twice_l2172_217228


namespace exists_ints_a_b_l2172_217241

theorem exists_ints_a_b (n : ℤ) (h : n % 4 ≠ 2) : ∃ a b : ℤ, n + a^2 = b^2 :=
by
  sorry

end exists_ints_a_b_l2172_217241


namespace problem_solution_l2172_217236

theorem problem_solution (a : ℝ) (h : a = Real.sqrt 5 - 1) :
  2 * a^3 + 7 * a^2 - 2 * a - 12 = 0 :=
by 
  sorry  -- Proof placeholder

end problem_solution_l2172_217236


namespace max_sum_x_y_l2172_217202

theorem max_sum_x_y (x y : ℝ) (h : (2015 + x^2) * (2015 + y^2) = 2 ^ 22) : 
  x + y ≤ 2 * Real.sqrt 33 :=
sorry

end max_sum_x_y_l2172_217202


namespace find_x_l2172_217220

theorem find_x (x : ℝ) (h : 0.65 * x = 0.20 * 682.50) : x = 210 :=
by
  sorry

end find_x_l2172_217220


namespace base8_to_base10_conversion_l2172_217287

def base8_to_base10 (n : Nat) : Nat := 
  match n with
  | 246 => 2 * 8^2 + 4 * 8^1 + 6 * 8^0
  | _ => 0  -- We define this only for the number 246_8

theorem base8_to_base10_conversion : base8_to_base10 246 = 166 := by 
  sorry

end base8_to_base10_conversion_l2172_217287


namespace area_of_triangles_equal_l2172_217231

theorem area_of_triangles_equal {a b c d : ℝ} (h_hyperbola_a : a ≠ 0) (h_hyperbola_b : b ≠ 0) 
    (h_hyperbola_c : c ≠ 0) (h_hyperbola_d : d ≠ 0) (h_parallel : a * b = c * d) :
  (1 / 2) * ((a + c) * (a + c) / (a * c)) = (1 / 2) * ((b + d) * (b + d) / (b * d)) :=
by
  sorry

end area_of_triangles_equal_l2172_217231


namespace find_rate_percent_l2172_217293

-- Definitions
def simpleInterest (P R T : ℕ) : ℕ := (P * R * T) / 100

-- Given conditions
def principal : ℕ := 900
def time : ℕ := 4
def simpleInterestValue : ℕ := 160

-- Rate percent
theorem find_rate_percent : 
  ∃ R : ℕ, simpleInterest principal R time = simpleInterestValue :=
by
  sorry

end find_rate_percent_l2172_217293


namespace h_of_neg2_eq_11_l2172_217201

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x ^ 2 + 1
def h (x : ℝ) : ℝ := f (g x)

theorem h_of_neg2_eq_11 : h (-2) = 11 := by
  sorry

end h_of_neg2_eq_11_l2172_217201


namespace positive_number_square_sum_eq_210_l2172_217288

theorem positive_number_square_sum_eq_210 (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end positive_number_square_sum_eq_210_l2172_217288


namespace initial_tickets_l2172_217270

-- Definitions of the conditions
def ferris_wheel_rides : ℕ := 2
def roller_coaster_rides : ℕ := 3
def log_ride_rides : ℕ := 7

def ferris_wheel_cost : ℕ := 2
def roller_coaster_cost : ℕ := 5
def log_ride_cost : ℕ := 1

def additional_tickets_needed : ℕ := 6

-- Calculate the total number of tickets needed
def total_tickets_needed : ℕ := 
  (ferris_wheel_rides * ferris_wheel_cost) +
  (roller_coaster_rides * roller_coaster_cost) +
  (log_ride_rides * log_ride_cost)

-- The proof statement
theorem initial_tickets : ∀ (initial_tickets : ℕ), 
  total_tickets_needed - additional_tickets_needed = initial_tickets → 
  initial_tickets = 20 :=
by
  intros initial_tickets h
  sorry

end initial_tickets_l2172_217270


namespace units_digit_of_7_power_19_l2172_217271

theorem units_digit_of_7_power_19 : (7^19) % 10 = 3 := by
  sorry

end units_digit_of_7_power_19_l2172_217271


namespace math_competition_correct_answers_l2172_217253

theorem math_competition_correct_answers (qA qB cA cB : ℕ) 
  (h_total_questions : qA + qB = 10)
  (h_score_A : cA * 5 - (qA - cA) * 2 = 36)
  (h_score_B : cB * 5 - (qB - cB) * 2 = 22) 
  (h_combined_score : cA * 5 - (qA - cA) * 2 + cB * 5 - (qB - cB) * 2 = 58)
  (h_score_difference : cA * 5 - (qA - cA) * 2 - (cB * 5 - (qB - cB) * 2) = 14) : 
  cA = 8 :=
by {
  sorry
}

end math_competition_correct_answers_l2172_217253


namespace larger_number_is_28_l2172_217235

theorem larger_number_is_28
  (x y : ℕ)
  (h1 : 4 * y = 7 * x)
  (h2 : y - x = 12) : y = 28 :=
sorry

end larger_number_is_28_l2172_217235


namespace solve_system_l2172_217240

variable (x y z : ℝ)

theorem solve_system :
  (y + z = 20 - 4 * x) →
  (x + z = -18 - 4 * y) →
  (x + y = 10 - 4 * z) →
  (2 * x + 2 * y + 2 * z = 4) :=
by
  intros h1 h2 h3
  sorry

end solve_system_l2172_217240


namespace sin_beta_value_l2172_217225

theorem sin_beta_value (a β : ℝ) (ha : 0 < a ∧ a < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (hcos_a : Real.cos a = 4 / 5)
  (hcos_a_plus_beta : Real.cos (a + β) = 5 / 13) :
  Real.sin β = 63 / 65 :=
sorry

end sin_beta_value_l2172_217225


namespace dora_rate_correct_l2172_217221

noncomputable def betty_rate : ℕ := 10
noncomputable def dora_rate : ℕ := 8
noncomputable def total_time : ℕ := 5
noncomputable def betty_break_time : ℕ := 2
noncomputable def cupcakes_difference : ℕ := 10

theorem dora_rate_correct :
  ∃ D : ℕ, 
  (D = dora_rate) ∧ 
  ((total_time - betty_break_time) * betty_rate = 30) ∧ 
  (total_time * D - 30 = cupcakes_difference) :=
sorry

end dora_rate_correct_l2172_217221


namespace cylinder_volume_l2172_217218

theorem cylinder_volume (r h : ℝ) (π : ℝ) 
  (h_pos : 0 < π) 
  (cond1 : 2 * π * r * h = 100 * π) 
  (cond2 : 4 * r^2 + h^2 = 200) : 
  (π * r^2 * h = 250 * π) := 
by 
  sorry

end cylinder_volume_l2172_217218


namespace percentage_markup_l2172_217254

theorem percentage_markup (selling_price cost_price : ℝ) (h_selling : selling_price = 2000) (h_cost : cost_price = 1250) :
  ((selling_price - cost_price) / cost_price) * 100 = 60 := by
  sorry

end percentage_markup_l2172_217254


namespace speed_of_stream_l2172_217275

variable (b s : ℝ)

-- Define the conditions from the problem
def downstream_condition := (100 : ℝ) / 4 = b + s
def upstream_condition := (75 : ℝ) / 15 = b - s

theorem speed_of_stream (h1 : downstream_condition b s) (h2: upstream_condition b s) : s = 10 := 
by 
  sorry

end speed_of_stream_l2172_217275


namespace subtract_decimal_l2172_217204

theorem subtract_decimal : 3.75 - 1.46 = 2.29 :=
by
  sorry

end subtract_decimal_l2172_217204


namespace rewrite_expression_l2172_217213

theorem rewrite_expression : ∀ x : ℝ, x^2 + 4 * x + 1 = (x + 2)^2 - 3 :=
by
  intros
  sorry

end rewrite_expression_l2172_217213


namespace total_jellybeans_l2172_217269

def nephews := 3
def nieces := 2
def jellybeans_per_child := 14
def children := nephews + nieces

theorem total_jellybeans : children * jellybeans_per_child = 70 := by
  sorry

end total_jellybeans_l2172_217269


namespace dustin_reads_more_pages_l2172_217248

theorem dustin_reads_more_pages (dustin_rate_per_hour : ℕ) (sam_rate_per_hour : ℕ) : 
  (dustin_rate_per_hour = 75) → (sam_rate_per_hour = 24) → 
  (dustin_rate_per_hour * 40 / 60 - sam_rate_per_hour * 40 / 60 = 34) :=
by
  sorry

end dustin_reads_more_pages_l2172_217248


namespace number_of_valid_selections_l2172_217276

theorem number_of_valid_selections : 
  ∃ combinations : Finset (Finset ℕ), 
    combinations = {
      {2, 6, 3, 5}, 
      {2, 6, 1, 7}, 
      {2, 4, 1, 5}, 
      {4, 1, 3}, 
      {6, 1, 5}, 
      {4, 6, 3, 7}, 
      {2, 4, 6, 5, 7}
    } ∧ combinations.card = 7 :=
by sorry

end number_of_valid_selections_l2172_217276


namespace green_pill_cost_l2172_217291

-- Define the conditions 
variables (pinkCost greenCost : ℝ)
variable (totalCost : ℝ := 819) -- total cost for three weeks
variable (days : ℝ := 21) -- number of days in three weeks

-- Establish relationships between pink and green pill costs
axiom greenIsMore : greenCost = pinkCost + 1
axiom dailyCost : 2 * greenCost + pinkCost = 39

-- Define the theorem to prove the cost of one green pill
theorem green_pill_cost : greenCost = 40/3 :=
by
  -- Proof would go here, but is omitted for now.
  sorry

end green_pill_cost_l2172_217291


namespace place_pawns_distinct_5x5_l2172_217222

noncomputable def number_of_ways_place_pawns : ℕ :=
  5 * 4 * 3 * 2 * 1 * 120

theorem place_pawns_distinct_5x5 : number_of_ways_place_pawns = 14400 := by
  sorry

end place_pawns_distinct_5x5_l2172_217222


namespace probability_three_consecutive_cards_l2172_217252

-- Definitions of the conditions
def total_ways_to_draw_three : ℕ := Nat.choose 52 3

def sets_of_consecutive_ranks : ℕ := 10

def ways_to_choose_three_consecutive : ℕ := 64

def favorable_outcomes : ℕ := sets_of_consecutive_ranks * ways_to_choose_three_consecutive

def probability_consecutive_ranks : ℚ := favorable_outcomes / total_ways_to_draw_three

-- The main statement to prove
theorem probability_three_consecutive_cards :
  probability_consecutive_ranks = 32 / 1105 := 
sorry

end probability_three_consecutive_cards_l2172_217252


namespace masha_mushrooms_l2172_217262

theorem masha_mushrooms (B1 B2 B3 B4 G1 G2 G3 : ℕ) (total : B1 + B2 + B3 + B4 + G1 + G2 + G3 = 70)
  (girls_distinct : G1 ≠ G2 ∧ G1 ≠ G3 ∧ G2 ≠ G3)
  (boys_threshold : ∀ {A B C D : ℕ}, (A = B1 ∨ A = B2 ∨ A = B3 ∨ A = B4) →
                    (B = B1 ∨ B = B2 ∨ B = B3 ∨ B = B4) →
                    (C = B1 ∨ C = B2 ∨ C = B3 ∨ C = B4) → 
                    (A ≠ B ∧ A ≠ C ∧ B ≠ C) →
                    A + B + C ≥ 43)
  (diff_no_more_than_five_times : ∀ {x y : ℕ}, (x = B1 ∨ x = B2 ∨ x = B3 ∨ x = B4 ∨ x = G1 ∨ x = G2 ∨ x = G3) →
                                  (y = B1 ∨ y = B2 ∨ y = B3 ∨ y = B4 ∨ y = G1 ∨ y = G2 ∨ y = G3) →
                                  x ≠ y → x ≤ 5 * y ∧ y ≤ 5 * x)
  (masha_max_girl : G3 = max G1 (max G2 G3))
  : G3 = 5 :=
sorry

end masha_mushrooms_l2172_217262


namespace xiao_gang_steps_l2172_217263

theorem xiao_gang_steps (x : ℕ) (H1 : 9000 / x = 13500 / (x + 15)) : x = 30 :=
by
  sorry

end xiao_gang_steps_l2172_217263


namespace rulers_left_in_drawer_l2172_217286

theorem rulers_left_in_drawer (initial_rulers taken_rulers : ℕ) (h1 : initial_rulers = 46) (h2 : taken_rulers = 25) :
  initial_rulers - taken_rulers = 21 :=
by
  sorry

end rulers_left_in_drawer_l2172_217286


namespace watch_A_accurate_l2172_217258

variable (T : ℕ) -- Standard time, represented as natural numbers for simplicity
variable (A B : ℕ) -- Watches A and B, also represented as natural numbers
variable (h1 : A = B + 2) -- Watch A is 2 minutes faster than Watch B
variable (h2 : B = T - 2) -- Watch B is 2 minutes slower than the standard time

theorem watch_A_accurate : A = T :=
by
  -- The proof would go here
  sorry

end watch_A_accurate_l2172_217258


namespace rectangular_field_area_l2172_217280

theorem rectangular_field_area :
  ∃ (w l : ℝ), w = l / 3 ∧ 2 * (w + l) = 72 ∧ w * l = 243 :=
by
  sorry

end rectangular_field_area_l2172_217280


namespace words_per_page_l2172_217200

theorem words_per_page (p : ℕ) (h1 : 150 * p ≡ 210 [MOD 221]) (h2 : p ≤ 120) : p = 195 := by
  sorry

end words_per_page_l2172_217200


namespace captain_age_is_24_l2172_217292

theorem captain_age_is_24 (C W : ℕ) 
  (hW : W = C + 7)
  (h_total_team_age : 23 * 11 = 253)
  (h_total_9_players_age : 22 * 9 = 198)
  (h_team_age_equation : 253 = 198 + C + W)
  : C = 24 :=
sorry

end captain_age_is_24_l2172_217292


namespace jeremy_home_to_school_distance_l2172_217278

theorem jeremy_home_to_school_distance (v d : ℝ) (h1 : 30 / 60 = 1 / 2) (h2 : 15 / 60 = 1 / 4)
  (h3 : d = v * (1 / 2)) (h4 : d = (v + 12) * (1 / 4)):
  d = 6 :=
by
  -- We assume that the conditions given lead to the distance being 6 miles
  sorry

end jeremy_home_to_school_distance_l2172_217278


namespace find_n_modulo_conditions_l2172_217264

theorem find_n_modulo_conditions :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 10 ∧ n % 7 = -3137 % 7 ∧ (n = 1 ∨ n = 8) := sorry

end find_n_modulo_conditions_l2172_217264


namespace workshop_male_workers_l2172_217290

variables (F M : ℕ)

theorem workshop_male_workers :
  (M = F + 45) ∧ (M - 5 = 3 * F) → M = 65 :=
by
  intros h
  sorry

end workshop_male_workers_l2172_217290


namespace lines_are_coplanar_l2172_217277

/- Define the parameterized lines -/
def L1 (s : ℝ) (k : ℝ) : ℝ × ℝ × ℝ := (1 + 2 * s, 4 - k * s, 2 + 2 * k * s)
def L2 (t : ℝ) : ℝ × ℝ × ℝ := (2 + t, 7 + 3 * t, 1 - 2 * t)

/- Prove that k = 0 ensures the lines are coplanar -/
theorem lines_are_coplanar (k : ℝ) : k = 0 ↔ 
  ∃ (s t : ℝ), L1 s k = L2 t :=
by {
  sorry
}

end lines_are_coplanar_l2172_217277


namespace total_salmon_count_l2172_217298

def chinook_males := 451228
def chinook_females := 164225
def sockeye_males := 212001
def sockeye_females := 76914
def coho_males := 301008
def coho_females := 111873
def pink_males := 518001
def pink_females := 182945
def chum_males := 230023
def chum_females := 81321

theorem total_salmon_count : 
  chinook_males + chinook_females + 
  sockeye_males + sockeye_females + 
  coho_males + coho_females + 
  pink_males + pink_females + 
  chum_males + chum_females = 2329539 := 
by
  sorry

end total_salmon_count_l2172_217298


namespace plane_through_line_and_point_l2172_217215

-- Definitions from the conditions
def line (x y z : ℝ) : Prop :=
  (x - 1) / 2 = (y - 3) / 4 ∧ (x - 1) / 2 = z / (-1)

def pointP1 : ℝ × ℝ × ℝ := (1, 5, 2)

-- Correct answer
def plane_eqn (x y z : ℝ) : Prop :=
  5 * x - 2 * y + 2 * z + 1 = 0

-- The theorem to prove
theorem plane_through_line_and_point (x y z : ℝ) :
  line x y z → plane_eqn x y z := by
  sorry

end plane_through_line_and_point_l2172_217215


namespace range_of_m_l2172_217226

theorem range_of_m (a : ℝ) (h : a ≠ 0) (x1 x2 y1 y2 : ℝ) (m : ℝ)
  (hx1 : -2 < x1 ∧ x1 < 0) (hx2 : m < x2 ∧ x2 < m + 1)
  (h_on_parabola_A : y1 = a * x1^2 - 2 * a * x1 - 3)
  (h_on_parabola_B : y2 = a * x2^2 - 2 * a * x2 - 3)
  (h_diff_y : y1 ≠ y2) :
  (0 < m ∧ m ≤ 1) ∨ m ≥ 4 :=
sorry

end range_of_m_l2172_217226


namespace product_4_7_25_l2172_217232

theorem product_4_7_25 : 4 * 7 * 25 = 700 :=
by sorry

end product_4_7_25_l2172_217232


namespace angle_complement_supplement_l2172_217265

theorem angle_complement_supplement (x : ℝ) (h1 : 90 - x = (1 / 2) * (180 - x)) : x = 90 := by
  sorry

end angle_complement_supplement_l2172_217265


namespace student_correct_answers_l2172_217211

theorem student_correct_answers (C I : ℕ) (h1 : C + I = 100) (h2 : C - 2 * I = 73) : C = 91 :=
sorry

end student_correct_answers_l2172_217211
