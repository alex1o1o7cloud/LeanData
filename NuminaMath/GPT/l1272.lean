import Mathlib

namespace NUMINAMATH_GPT_problem_statement_l1272_127281

theorem problem_statement (p x : ℝ) (h : 0 ≤ p ∧ p ≤ 4) :
  (x^2 + p*x > 4*x + p - 3) ↔ (x > 3 ∨ x < -1) := by
sorry

end NUMINAMATH_GPT_problem_statement_l1272_127281


namespace NUMINAMATH_GPT_arithmetic_sequence_ninth_term_l1272_127253

theorem arithmetic_sequence_ninth_term
  (a d : ℤ)
  (h1 : a + 2 * d = 23)
  (h2 : a + 5 * d = 29) :
  a + 8 * d = 35 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_ninth_term_l1272_127253


namespace NUMINAMATH_GPT_distance_probability_l1272_127215

theorem distance_probability :
  let speed := 5
  let num_roads := 8
  let total_outcomes := num_roads * (num_roads - 1)
  let favorable_outcomes := num_roads * 3
  let probability := (favorable_outcomes : ℝ) / total_outcomes
  probability = 0.375 :=
by
  sorry

end NUMINAMATH_GPT_distance_probability_l1272_127215


namespace NUMINAMATH_GPT_total_number_of_crickets_l1272_127246

def initial_crickets : ℝ := 7.0
def additional_crickets : ℝ := 11.0
def total_crickets : ℝ := 18.0

theorem total_number_of_crickets :
  initial_crickets + additional_crickets = total_crickets :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_crickets_l1272_127246


namespace NUMINAMATH_GPT_ratio_of_x_y_l1272_127277

theorem ratio_of_x_y (x y : ℝ) (h : x + y = 3 * (x - y)) : x / y = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_x_y_l1272_127277


namespace NUMINAMATH_GPT_fraction_complex_eq_l1272_127223

theorem fraction_complex_eq (z : ℂ) (h : z = 2 + I) : 2 * I / (z - 1) = 1 + I := by
  sorry

end NUMINAMATH_GPT_fraction_complex_eq_l1272_127223


namespace NUMINAMATH_GPT_chad_ice_cost_l1272_127287

theorem chad_ice_cost
  (n : ℕ) -- Number of people
  (p : ℕ) -- Pounds of ice per person
  (c : ℝ) -- Cost per 10 pound bag of ice
  (h1 : n = 20) 
  (h2 : p = 3)
  (h3 : c = 4.5) :
  (3 * 20 / 10) * 4.5 = 27 :=
by
  sorry

end NUMINAMATH_GPT_chad_ice_cost_l1272_127287


namespace NUMINAMATH_GPT_total_pages_written_is_24_l1272_127297

def normal_letter_interval := 3
def time_per_normal_letter := 20
def time_per_page := 10
def additional_time_factor := 2
def time_spent_long_letter := 80
def days_in_month := 30

def normal_letters_written := days_in_month / normal_letter_interval
def pages_per_normal_letter := time_per_normal_letter / time_per_page
def total_pages_normal_letters := normal_letters_written * pages_per_normal_letter

def time_per_page_long_letter := additional_time_factor * time_per_page
def pages_long_letter := time_spent_long_letter / time_per_page_long_letter

def total_pages_written := total_pages_normal_letters + pages_long_letter

theorem total_pages_written_is_24 : total_pages_written = 24 := by
  sorry

end NUMINAMATH_GPT_total_pages_written_is_24_l1272_127297


namespace NUMINAMATH_GPT_max_bishops_1000x1000_l1272_127203

def bishop_max_non_attacking (n : ℕ) : ℕ :=
  2 * (n - 1)

theorem max_bishops_1000x1000 : bishop_max_non_attacking 1000 = 1998 :=
by sorry

end NUMINAMATH_GPT_max_bishops_1000x1000_l1272_127203


namespace NUMINAMATH_GPT_inequality_proof_l1272_127254

open Real

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 / (x^2 + y * z) + 1 / (y^2 + z * x) + 1 / (z^2 + x * y)) ≤ 
  (1 / 2) * (1 / (x * y) + 1 / (y * z) + 1 / (z * x)) :=
by sorry

end NUMINAMATH_GPT_inequality_proof_l1272_127254


namespace NUMINAMATH_GPT_johnny_hours_second_job_l1272_127276

theorem johnny_hours_second_job (x : ℕ) (h_eq : 5 * (69 + 10 * x) = 445) : x = 2 :=
by 
  -- The proof will go here, but we skip it as per the instructions
  sorry

end NUMINAMATH_GPT_johnny_hours_second_job_l1272_127276


namespace NUMINAMATH_GPT_rainfall_third_day_is_18_l1272_127282

-- Define the conditions including the rainfall for each day
def rainfall_first_day : ℕ := 4
def rainfall_second_day : ℕ := 5 * rainfall_first_day
def rainfall_third_day : ℕ := (rainfall_first_day + rainfall_second_day) - 6

-- Prove that the rainfall on the third day is 18 inches
theorem rainfall_third_day_is_18 : rainfall_third_day = 18 :=
by
  -- Use the definitions and directly state that the proof follows
  sorry

end NUMINAMATH_GPT_rainfall_third_day_is_18_l1272_127282


namespace NUMINAMATH_GPT_utensils_in_each_pack_l1272_127200

/-- Prove that given John needs to buy 5 packs to get 50 spoons
    and each pack contains an equal number of knives, forks, and spoons,
    the total number of utensils in each pack is 30. -/
theorem utensils_in_each_pack
  (packs : ℕ)
  (total_spoons : ℕ)
  (equal_parts : ∀ p : ℕ, p = total_spoons / packs)
  (knives forks spoons : ℕ)
  (equal_utensils : ∀ u : ℕ, u = spoons)
  (knives_forks : knives = forks)
  (knives_spoons : knives = spoons)
  (packs_needed : packs = 5)
  (total_utensils_needed : total_spoons = 50) :
  knives + forks + spoons = 30 := by
  sorry

end NUMINAMATH_GPT_utensils_in_each_pack_l1272_127200


namespace NUMINAMATH_GPT_luna_badges_correct_l1272_127229

-- conditions
def total_badges : ℕ := 83
def hermione_badges : ℕ := 14
def celestia_badges : ℕ := 52

-- question and answer
theorem luna_badges_correct : total_badges - (hermione_badges + celestia_badges) = 17 :=
by
  sorry

end NUMINAMATH_GPT_luna_badges_correct_l1272_127229


namespace NUMINAMATH_GPT_average_students_l1272_127269

def ClassGiraffe : ℕ := 225

def ClassElephant (giraffe: ℕ) : ℕ := giraffe + 48

def ClassRabbit (giraffe: ℕ) : ℕ := giraffe - 24

theorem average_students (giraffe : ℕ) (elephant : ℕ) (rabbit : ℕ) :
  giraffe = 225 → elephant = giraffe + 48 → rabbit = giraffe - 24 →
  (giraffe + elephant + rabbit) / 3 = 233 := by
  sorry

end NUMINAMATH_GPT_average_students_l1272_127269


namespace NUMINAMATH_GPT_sum_of_digits_of_n_l1272_127224

theorem sum_of_digits_of_n :
  ∃ n : ℕ,
    n > 2000 ∧
    n + 135 % 75 = 15 ∧
    n + 75 % 135 = 45 ∧
    (n = 2025 ∧ (2 + 0 + 2 + 5 = 9)) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_n_l1272_127224


namespace NUMINAMATH_GPT_ratio_a7_b7_l1272_127219

variables (a b : ℕ → ℤ) (Sa Tb : ℕ → ℤ)
variables (h1 : ∀ n : ℕ, a n = a 0 + n * (a 1 - a 0))
variables (h2 : ∀ n : ℕ, b n = b 0 + n * (b 1 - b 0))
variables (h3 : ∀ n : ℕ, Sa n = n * (a 0 + a n) / 2)
variables (h4 : ∀ n : ℕ, Tb n = n * (b 0 + b n) / 2)
variables (h5 : ∀ n : ℕ, n > 0 → Sa n / Tb n = (7 * n + 1) / (4 * n + 27))

theorem ratio_a7_b7 : ∀ n : ℕ, n = 7 → a 7 / b 7 = 92 / 79 :=
by
  intros n hn_eq
  sorry

end NUMINAMATH_GPT_ratio_a7_b7_l1272_127219


namespace NUMINAMATH_GPT_distance_traveled_l1272_127233

-- Define the conditions
def rate : Real := 60  -- rate of 60 miles per hour
def total_break_time : Real := 1  -- total break time of 1 hour
def total_trip_time : Real := 9  -- total trip time of 9 hours

-- The theorem to prove the distance traveled
theorem distance_traveled : rate * (total_trip_time - total_break_time) = 480 := 
by
  sorry

end NUMINAMATH_GPT_distance_traveled_l1272_127233


namespace NUMINAMATH_GPT_alpha_plus_beta_l1272_127266

noncomputable def alpha_beta (α β : ℝ) : Prop :=
  ∀ x : ℝ, ((x - α) / (x + β)) = ((x^2 - 54 * x + 621) / (x^2 + 42 * x - 1764))

theorem alpha_plus_beta : ∃ α β : ℝ, α + β = 86 ∧ alpha_beta α β :=
by
  sorry

end NUMINAMATH_GPT_alpha_plus_beta_l1272_127266


namespace NUMINAMATH_GPT_company_match_percentage_l1272_127231

theorem company_match_percentage (total_contribution : ℝ) (holly_contribution_per_paycheck : ℝ) (total_paychecks : ℕ) (total_contribution_one_year : ℝ) : 
  let holly_contribution := holly_contribution_per_paycheck * total_paychecks
  let company_contribution := total_contribution_one_year - holly_contribution
  (company_contribution / holly_contribution) * 100 = 6 :=
by
  let holly_contribution := holly_contribution_per_paycheck * total_paychecks
  let company_contribution := total_contribution_one_year - holly_contribution
  have h : holly_contribution = 2600 := by sorry
  have c : company_contribution = 156 := by sorry
  exact sorry

end NUMINAMATH_GPT_company_match_percentage_l1272_127231


namespace NUMINAMATH_GPT_symmetry_axis_one_of_cos_2x_minus_sin_2x_l1272_127299

noncomputable def symmetry_axis (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (k * Real.pi / 2) - Real.pi / 8

theorem symmetry_axis_one_of_cos_2x_minus_sin_2x :
  symmetry_axis (-Real.pi / 8) :=
by
  use 0
  simp
  sorry

end NUMINAMATH_GPT_symmetry_axis_one_of_cos_2x_minus_sin_2x_l1272_127299


namespace NUMINAMATH_GPT_universal_quantifiers_are_true_l1272_127283

-- Declare the conditions as hypotheses
theorem universal_quantifiers_are_true :
  (∀ x : ℝ, x^2 - x + 0.25 ≥ 0) ∧ (∀ x : ℝ, x^2 + 2 * x + 2 > 0) :=
by
  sorry -- Proof skipped

end NUMINAMATH_GPT_universal_quantifiers_are_true_l1272_127283


namespace NUMINAMATH_GPT_shortest_distance_between_tracks_l1272_127291

noncomputable def rational_man_track (x y : ℝ) : Prop :=
x^2 + y^2 = 1

noncomputable def irrational_man_track (x y : ℝ) : Prop :=
(x + 1)^2 + y^2 = 9

noncomputable def shortest_distance : ℝ :=
0

theorem shortest_distance_between_tracks :
  ∀ (A B : ℝ × ℝ), 
  rational_man_track A.1 A.2 → 
  irrational_man_track B.1 B.2 → 
  dist A B = shortest_distance := sorry

end NUMINAMATH_GPT_shortest_distance_between_tracks_l1272_127291


namespace NUMINAMATH_GPT_games_within_division_l1272_127209

theorem games_within_division (N M : ℕ) (h1 : N > 3 * M) (h2 : M > 5) (h3 : 2 * N + 6 * M = 76) : 2 * N = 40 :=
by {
  sorry
}

end NUMINAMATH_GPT_games_within_division_l1272_127209


namespace NUMINAMATH_GPT_quadratic_polynomial_exists_l1272_127262

theorem quadratic_polynomial_exists (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ p : ℝ → ℝ, (∀ x, p x = (a^2 + ab + b^2 + ac + bc + c^2) * x^2 
                   - (a + b) * (b + c) * (a + c) * x 
                   + abc * (a + b + c))
              ∧ p a = a^4 
              ∧ p b = b^4 
              ∧ p c = c^4 := 
sorry

end NUMINAMATH_GPT_quadratic_polynomial_exists_l1272_127262


namespace NUMINAMATH_GPT_AngiesClassGirlsCount_l1272_127272

theorem AngiesClassGirlsCount (n_girls n_boys : ℕ) (total_students : ℕ)
  (h1 : n_girls = 2 * (total_students / 5))
  (h2 : n_boys = 3 * (total_students / 5))
  (h3 : n_girls + n_boys = 20)
  : n_girls = 8 :=
by
  sorry

end NUMINAMATH_GPT_AngiesClassGirlsCount_l1272_127272


namespace NUMINAMATH_GPT_arithmetic_square_root_of_sqrt_16_l1272_127295

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_square_root_of_sqrt_16_l1272_127295


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1272_127212

theorem solution_set_of_inequality :
  {x : ℝ | 4 * x ^ 2 - 4 * x + 1 ≤ 0} = {1 / 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1272_127212


namespace NUMINAMATH_GPT_probability_roots_real_l1272_127273

-- Define the polynomial
def polynomial (b : ℝ) (x : ℝ) : ℝ :=
  x^4 + 3*b*x^3 + (3*b - 5)*x^2 + (-6*b + 4)*x - 3

-- Define the intervals for b
def interval_b1 := Set.Icc (-(15:ℝ)) (20:ℝ)
def interval_b2 := Set.Icc (-(15:ℝ)) (-2/3)
def interval_b3 := Set.Icc (4/3) (20:ℝ)

-- Calculate the lengths of the intervals
def length_interval (a b : ℝ) : ℝ := b - a

noncomputable def length_b1 := length_interval (-(15:ℝ)) (20:ℝ)
noncomputable def length_b2 := length_interval (-(15:ℝ)) (-2/3)
noncomputable def length_b3 := length_interval (4/3) (20:ℝ)
noncomputable def effective_length := length_b2 + length_b3

-- The probability is the ratio of effective lengths
noncomputable def probability := effective_length / length_b1

-- The theorem we want to prove
theorem probability_roots_real : probability = 33/35 :=
  sorry

end NUMINAMATH_GPT_probability_roots_real_l1272_127273


namespace NUMINAMATH_GPT_time_addition_correct_l1272_127268

def start_time := (3, 0, 0) -- Representing 3:00:00 PM as (hours, minutes, seconds)
def additional_time := (315, 78, 30) -- Representing additional time as (hours, minutes, seconds)

noncomputable def resulting_time (start add : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  let (sh, sm, ss) := start -- start hours, minutes, seconds
  let (ah, am, as) := add -- additional hours, minutes, seconds
  let total_seconds := ss + as
  let extra_minutes := total_seconds / 60
  let remaining_seconds := total_seconds % 60
  let total_minutes := sm + am + extra_minutes
  let extra_hours := total_minutes / 60
  let remaining_minutes := total_minutes % 60
  let total_hours := sh + ah + extra_hours
  let resulting_hours := (total_hours % 12) -- Modulo 12 for wrap-around
  (resulting_hours, remaining_minutes, remaining_seconds)

theorem time_addition_correct :
  let (A, B, C) := resulting_time start_time additional_time
  A + B + C = 55 := by
  sorry

end NUMINAMATH_GPT_time_addition_correct_l1272_127268


namespace NUMINAMATH_GPT_minimum_value_expression_l1272_127252

theorem minimum_value_expression (p q r s t u v w : ℝ) (h1 : p > 0) (h2 : q > 0) 
    (h3 : r > 0) (h4 : s > 0) (h5 : t > 0) (h6 : u > 0) (h7 : v > 0) (h8 : w > 0)
    (hpqrs : p * q * r * s = 16) (htuvw : t * u * v * w = 25) 
    (hptqu : p * t = q * u ∧ q * u = r * v ∧ r * v = s * w) : 
    (p * t) ^ 2 + (q * u) ^ 2 + (r * v) ^ 2 + (s * w) ^ 2 = 80 := sorry

end NUMINAMATH_GPT_minimum_value_expression_l1272_127252


namespace NUMINAMATH_GPT_volume_range_of_rectangular_solid_l1272_127216

theorem volume_range_of_rectangular_solid
  (a b c : ℝ)
  (h1 : 2 * (a * b + b * c + c * a) = 48)
  (h2 : 4 * (a + b + c) = 36) :
  (16 : ℝ) ≤ a * b * c ∧ a * b * c ≤ 20 :=
by sorry

end NUMINAMATH_GPT_volume_range_of_rectangular_solid_l1272_127216


namespace NUMINAMATH_GPT_probability_of_two_same_color_l1272_127245

noncomputable def probability_at_least_two_same_color (reds whites blues greens : ℕ) (total_draws : ℕ) : ℚ :=
  have total_marbles := reds + whites + blues + greens
  let total_combinations := Nat.choose total_marbles total_draws
  let two_reds := Nat.choose reds 2 * (total_marbles - 2)
  let two_whites := Nat.choose whites 2 * (total_marbles - 2)
  let two_blues := Nat.choose blues 2 * (total_marbles - 2)
  let two_greens := Nat.choose greens 2 * (total_marbles - 2)
  
  let all_reds := Nat.choose reds 3
  let all_whites := Nat.choose whites 3
  let all_blues := Nat.choose blues 3
  let all_greens := Nat.choose greens 3
  
  let desired_outcomes := two_reds + two_whites + two_blues + two_greens +
                          all_reds + all_whites + all_blues + all_greens
                          
  (desired_outcomes : ℚ) / (total_combinations : ℚ)

theorem probability_of_two_same_color : probability_at_least_two_same_color 6 7 8 4 3 = 69 / 115 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_two_same_color_l1272_127245


namespace NUMINAMATH_GPT_bucket_holds_120_ounces_l1272_127222

theorem bucket_holds_120_ounces :
  ∀ (fill_buckets remove_buckets baths_per_day ounces_per_week : ℕ),
    fill_buckets = 14 →
    remove_buckets = 3 →
    baths_per_day = 7 →
    ounces_per_week = 9240 →
    baths_per_day * (fill_buckets - remove_buckets) * (ounces_per_week / (baths_per_day * (fill_buckets - remove_buckets))) = ounces_per_week →
    (ounces_per_week / (baths_per_day * (fill_buckets - remove_buckets))) = 120 :=
by
  intros fill_buckets remove_buckets baths_per_day ounces_per_week Hfill Hremove Hbaths Hounces Hcalc
  sorry

end NUMINAMATH_GPT_bucket_holds_120_ounces_l1272_127222


namespace NUMINAMATH_GPT_samuel_distance_from_hotel_l1272_127206

/-- Samuel's driving problem conditions. -/
structure DrivingConditions where
  total_distance : ℕ -- in miles
  first_speed : ℕ -- in miles per hour
  first_time : ℕ -- in hours
  second_speed : ℕ -- in miles per hour
  second_time : ℕ -- in hours

def distance_remaining (c : DrivingConditions) : ℕ :=
  let distance_covered := (c.first_speed * c.first_time) + (c.second_speed * c.second_time)
  c.total_distance - distance_covered

/-- Prove that Samuel is 130 miles from the hotel. -/
theorem samuel_distance_from_hotel : 
  ∀ (c : DrivingConditions), 
    c.total_distance = 600 ∧
    c.first_speed = 50 ∧
    c.first_time = 3 ∧
    c.second_speed = 80 ∧
    c.second_time = 4 → distance_remaining c = 130 := by
  intros c h
  cases h
  sorry

end NUMINAMATH_GPT_samuel_distance_from_hotel_l1272_127206


namespace NUMINAMATH_GPT_difference_max_min_students_l1272_127240

-- Definitions for problem conditions
def total_students : ℕ := 50
def shanghai_university_min : ℕ := 40
def shanghai_university_max : ℕ := 45
def shanghai_normal_university_min : ℕ := 16
def shanghai_normal_university_max : ℕ := 20

-- Lean statement for the math proof problem
theorem difference_max_min_students :
  (∀ (a b : ℕ), shanghai_university_min ≤ a ∧ a ≤ shanghai_university_max →
                shanghai_normal_university_min ≤ b ∧ b ≤ shanghai_normal_university_max →
                15 ≤ a + b - total_students ∧ a + b - total_students ≤ 15) →
  (∀ (a b : ℕ), shanghai_university_min ≤ a ∧ a ≤ shanghai_university_max →
                shanghai_normal_university_min ≤ b ∧ b ≤ shanghai_normal_university_max →
                6 ≤ a + b - total_students ∧ a + b - total_students ≤ 6) →
  (∃ M m : ℕ, 
    (M = 15) ∧ 
    (m = 6) ∧ 
    (M - m = 9)) :=
by
  sorry

end NUMINAMATH_GPT_difference_max_min_students_l1272_127240


namespace NUMINAMATH_GPT_reduced_rectangle_area_l1272_127250

theorem reduced_rectangle_area
  (w h : ℕ) (hw : w = 5) (hh : h = 7)
  (new_w : ℕ) (h_reduced_area : new_w = w - 2 ∧ new_w * h = 21)
  (reduced_h : ℕ) (hr : reduced_h = h - 1) :
  (new_w * reduced_h = 18) :=
by
  sorry

end NUMINAMATH_GPT_reduced_rectangle_area_l1272_127250


namespace NUMINAMATH_GPT_cattle_area_correct_l1272_127278

-- Definitions based on the problem conditions
def length_km := 3.6
def width_km := 2.5 * length_km
def total_area_km2 := length_km * width_km
def cattle_area_km2 := total_area_km2 / 2

-- Theorem statement
theorem cattle_area_correct : cattle_area_km2 = 16.2 := by
  sorry

end NUMINAMATH_GPT_cattle_area_correct_l1272_127278


namespace NUMINAMATH_GPT_calc_expr_l1272_127204

theorem calc_expr : 4⁻¹ - Real.sqrt (1 / 16) + (3 - Real.sqrt 2)^0 = 1 := by
  sorry

end NUMINAMATH_GPT_calc_expr_l1272_127204


namespace NUMINAMATH_GPT_evaporation_days_l1272_127221

theorem evaporation_days
    (initial_water : ℝ)
    (evap_rate : ℝ)
    (percent_evaporated : ℝ)
    (evaporated_water : ℝ)
    (days : ℝ)
    (h1 : initial_water = 10)
    (h2 : evap_rate = 0.012)
    (h3 : percent_evaporated = 0.06)
    (h4 : evaporated_water = initial_water * percent_evaporated)
    (h5 : days = evaporated_water / evap_rate) :
  days = 50 :=
by
  sorry

end NUMINAMATH_GPT_evaporation_days_l1272_127221


namespace NUMINAMATH_GPT_ruby_height_is_192_l1272_127249

def height_janet := 62
def height_charlene := 2 * height_janet
def height_pablo := height_charlene + 70
def height_ruby := height_pablo - 2

theorem ruby_height_is_192 : height_ruby = 192 := by
  sorry

end NUMINAMATH_GPT_ruby_height_is_192_l1272_127249


namespace NUMINAMATH_GPT_zero_people_with_fewer_than_six_cards_l1272_127296

theorem zero_people_with_fewer_than_six_cards (cards people : ℕ) (h_cards : cards = 60) (h_people : people = 9) :
  let avg := cards / people
  let remainder := cards % people
  remainder < people → ∃ n, n = 0 := by
  sorry

end NUMINAMATH_GPT_zero_people_with_fewer_than_six_cards_l1272_127296


namespace NUMINAMATH_GPT_pipeA_fills_tank_in_56_minutes_l1272_127256

-- Define the relevant variables and conditions.
variable (t : ℕ) -- Time for Pipe A to fill the tank in minutes

-- Condition: Pipe B fills the tank 7 times faster than Pipe A
def pipeB_time (t : ℕ) := t / 7

-- Combined rate of Pipe A and Pipe B filling the tank in 7 minutes
def combined_rate (t : ℕ) := (1 / t) + (1 / pipeB_time t)

-- Given the combined rate fills the tank in 7 minutes
def combined_rate_equals (t : ℕ) := combined_rate t = 1 / 7

-- The proof statement
theorem pipeA_fills_tank_in_56_minutes (t : ℕ) (h : combined_rate_equals t) : t = 56 :=
sorry

end NUMINAMATH_GPT_pipeA_fills_tank_in_56_minutes_l1272_127256


namespace NUMINAMATH_GPT_find_a_l1272_127263

def are_parallel (a : ℝ) : Prop :=
  (a + 1) = (2 - a)

theorem find_a (a : ℝ) (h : are_parallel a) : a = 0 :=
sorry

end NUMINAMATH_GPT_find_a_l1272_127263


namespace NUMINAMATH_GPT_reciprocal_of_neg2_l1272_127228

-- Define the number
def num : ℤ := -2

-- Define the reciprocal function
def reciprocal (x : ℤ) : ℚ := 1 / x

-- State the theorem
theorem reciprocal_of_neg2 : reciprocal num = -1 / 2 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg2_l1272_127228


namespace NUMINAMATH_GPT_cos_half_pi_plus_alpha_l1272_127280

open Real

noncomputable def alpha : ℝ := sorry

theorem cos_half_pi_plus_alpha :
  let a := (1 / 3, tan alpha)
  let b := (cos alpha, 1)
  ((1 / 3) / (cos alpha) = (tan alpha) / 1) →
  cos (pi / 2 + alpha) = -1 / 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cos_half_pi_plus_alpha_l1272_127280


namespace NUMINAMATH_GPT_ratio_of_arithmetic_seqs_l1272_127205

noncomputable def arithmetic_seq_sum (a_1 a_n : ℕ) (n : ℕ) : ℝ := (n * (a_1 + a_n)) / 2

theorem ratio_of_arithmetic_seqs (a_1 a_6 a_11 b_1 b_6 b_11 : ℕ) :
  (∀ n : ℕ, (arithmetic_seq_sum a_1 a_n n) / (arithmetic_seq_sum b_1 b_n n) = n / (2 * n + 1))
  → (a_1 + a_6) / (b_1 + b_6) = 6 / 13
  → (a_1 + a_11) / (b_1 + b_11) = 11 / 23
  → (a_6 : ℝ) / (b_6 : ℝ) = 11 / 23 :=
  by
    intros h₁₁ h₆ h₁₁b
    sorry

end NUMINAMATH_GPT_ratio_of_arithmetic_seqs_l1272_127205


namespace NUMINAMATH_GPT_marissa_tied_boxes_l1272_127248

theorem marissa_tied_boxes 
  (r_total : ℝ) (r_per_box : ℝ) (r_left : ℝ) (h_total : r_total = 4.5)
  (h_per_box : r_per_box = 0.7) (h_left : r_left = 1) :
  (r_total - r_left) / r_per_box = 5 :=
by
  sorry

end NUMINAMATH_GPT_marissa_tied_boxes_l1272_127248


namespace NUMINAMATH_GPT_nth_term_pattern_l1272_127217

theorem nth_term_pattern (a : ℕ → ℕ) (h : ∀ n, a n = n * (n - 1)) : 
  (a 0 = 0) ∧ (a 1 = 2) ∧ (a 2 = 6) ∧ (a 3 = 12) ∧ (a 4 = 20) ∧ 
  (a 5 = 30) ∧ (a 6 = 42) ∧ (a 7 = 56) ∧ (a 8 = 72) ∧ (a 9 = 90) := sorry

end NUMINAMATH_GPT_nth_term_pattern_l1272_127217


namespace NUMINAMATH_GPT_intersection_volume_l1272_127225

noncomputable def volume_of_intersection (k : ℝ) : ℝ :=
  ∫ x in -k..k, 4 * (k^2 - x^2)

theorem intersection_volume (k : ℝ) : volume_of_intersection k = 16 * k^3 / 3 :=
  by
  sorry

end NUMINAMATH_GPT_intersection_volume_l1272_127225


namespace NUMINAMATH_GPT_buratino_spent_dollars_l1272_127293

theorem buratino_spent_dollars (x y : ℕ) (h1 : x + y = 50) (h2 : 2 * x = 3 * y) : 
  (y * 5 - x * 3) = 10 :=
by
  sorry

end NUMINAMATH_GPT_buratino_spent_dollars_l1272_127293


namespace NUMINAMATH_GPT_solve_eq_l1272_127208

theorem solve_eq (x y : ℕ) (h : x^2 - 2 * x * y + y^2 + 5 * x + 5 * y = 1500) :
  (x = 150 ∧ y = 150) ∨ (x = 150 ∧ y = 145) ∨ (x = 145 ∧ y = 135) ∨
  (x = 135 ∧ y = 120) ∨ (x = 120 ∧ y = 100) ∨ (x = 100 ∧ y = 75) ∨
  (x = 75 ∧ y = 45) ∨ (x = 45 ∧ y = 10) ∨ (x = 145 ∧ y = 150) ∨
  (x = 135 ∧ y = 145) ∨ (x = 120 ∧ y = 135) ∨ (x = 100 ∧ y = 120) ∨
  (x = 75 ∧ y = 100) ∨ (x = 45 ∧ y = 75) ∨ (x = 10 ∧ y = 45) :=
sorry

end NUMINAMATH_GPT_solve_eq_l1272_127208


namespace NUMINAMATH_GPT_limit_of_hours_for_overtime_l1272_127264

theorem limit_of_hours_for_overtime
  (R : Real) (O : Real) (total_compensation : Real) (total_hours_worked : Real) (L : Real)
  (hR : R = 14)
  (hO : O = 1.75 * R)
  (hTotalCompensation : total_compensation = 998)
  (hTotalHoursWorked : total_hours_worked = 57.88)
  (hEquation : (R * L) + ((total_hours_worked - L) * O) = total_compensation) :
  L = 40 := 
  sorry

end NUMINAMATH_GPT_limit_of_hours_for_overtime_l1272_127264


namespace NUMINAMATH_GPT_initial_population_of_town_l1272_127247

theorem initial_population_of_town 
  (final_population : ℝ) 
  (growth_rate : ℝ) 
  (years : ℕ) 
  (initial_population : ℝ) 
  (h : final_population = initial_population * (1 + growth_rate) ^ years) : 
  initial_population = 297500 / (1 + 0.07) ^ 10 :=
by
  sorry

end NUMINAMATH_GPT_initial_population_of_town_l1272_127247


namespace NUMINAMATH_GPT_number_of_new_students_l1272_127226

theorem number_of_new_students (initial_students end_students students_left : ℕ) 
  (h_initial: initial_students = 33) 
  (h_left: students_left = 18) 
  (h_end: end_students = 29) : 
  initial_students - students_left + (end_students - (initial_students - students_left)) = 14 :=
by
  sorry

end NUMINAMATH_GPT_number_of_new_students_l1272_127226


namespace NUMINAMATH_GPT_max_regions_with_five_lines_l1272_127201

def max_regions (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * (n + 1) / 2 + 1

theorem max_regions_with_five_lines (n : ℕ) (h : n = 5) : max_regions n = 16 :=
by {
  rw [h, max_regions];
  norm_num;
  done
}

end NUMINAMATH_GPT_max_regions_with_five_lines_l1272_127201


namespace NUMINAMATH_GPT_find_geometric_sequence_term_l1272_127255

noncomputable def geometric_sequence_term (a q : ℝ) (n : ℕ) : ℝ := a * q ^ (n - 1)

theorem find_geometric_sequence_term (a : ℝ) (q : ℝ)
  (h1 : a * (1 - q ^ 3) / (1 - q) = 7)
  (h2 : a * (1 - q ^ 6) / (1 - q) = 63) :
  ∀ n : ℕ, geometric_sequence_term a q n = 2^(n-1) :=
by
  sorry

end NUMINAMATH_GPT_find_geometric_sequence_term_l1272_127255


namespace NUMINAMATH_GPT_count_perfect_squares_divisible_by_36_l1272_127271

theorem count_perfect_squares_divisible_by_36 :
  let N := 10000
  let max_square := 10^8
  let multiple := 36
  let valid_divisor := 1296
  let count_multiples := 277
  (∀ N : ℕ, N^2 < max_square → (∃ k : ℕ, N = k * multiple ∧ k < N)) → 
  ∃ cnt : ℕ, cnt = count_multiples := 
by {
  sorry
}

end NUMINAMATH_GPT_count_perfect_squares_divisible_by_36_l1272_127271


namespace NUMINAMATH_GPT_g_one_minus_g_four_l1272_127207

theorem g_one_minus_g_four (g : ℝ → ℝ)
  (h_linear : ∀ x y : ℝ, g (x + y) = g x + g y)
  (h_diff : ∀ x : ℝ, g (x + 1) - g x = 5) :
  g 1 - g 4 = -15 :=
sorry

end NUMINAMATH_GPT_g_one_minus_g_four_l1272_127207


namespace NUMINAMATH_GPT_total_wings_l1272_127220

-- Conditions
def money_per_grandparent : ℕ := 50
def number_of_grandparents : ℕ := 4
def bird_cost : ℕ := 20
def wings_per_bird : ℕ := 2

-- Calculate the total amount of money John received:
def total_money_received : ℕ := number_of_grandparents * money_per_grandparent

-- Determine the number of birds John can buy:
def number_of_birds : ℕ := total_money_received / bird_cost

-- Prove that the total number of wings all the birds have is 20:
theorem total_wings : number_of_birds * wings_per_bird = 20 :=
by
  sorry

end NUMINAMATH_GPT_total_wings_l1272_127220


namespace NUMINAMATH_GPT_number_of_elements_less_than_2004_l1272_127274

theorem number_of_elements_less_than_2004 (f : ℕ → ℕ) 
    (h0 : f 0 = 0) 
    (h1 : ∀ n : ℕ, (f (2 * n + 1)) ^ 2 - (f (2 * n)) ^ 2 = 6 * f n + 1) 
    (h2 : ∀ n : ℕ, f (2 * n) > f n) 
  : ∃ m : ℕ,  m = 128 ∧ ∀ x : ℕ, f x < 2004 → x < m := sorry

end NUMINAMATH_GPT_number_of_elements_less_than_2004_l1272_127274


namespace NUMINAMATH_GPT_max_possible_median_l1272_127284

/-- 
Given:
1. The Beverage Barn sold 300 cans of soda to 120 customers.
2. Every customer bought at least 1 can of soda but no more than 5 cans.
Prove that the maximum possible median number of cans of soda bought per customer is 5.
-/
theorem max_possible_median (total_cans : ℕ) (customers : ℕ) (min_can_per_customer : ℕ) (max_can_per_customer : ℕ) :
  total_cans = 300 ∧ customers = 120 ∧ min_can_per_customer = 1 ∧ max_can_per_customer = 5 →
  (∃ median : ℕ, median = 5) :=
by
  sorry

end NUMINAMATH_GPT_max_possible_median_l1272_127284


namespace NUMINAMATH_GPT_matt_without_calculator_5_minutes_l1272_127241

-- Define the conditions
def time_with_calculator (problems : Nat) : Nat := 2 * problems
def time_without_calculator (problems : Nat) (x : Nat) : Nat := x * problems
def time_saved (problems : Nat) (x : Nat) : Nat := time_without_calculator problems x - time_with_calculator problems

-- State the problem
theorem matt_without_calculator_5_minutes (x : Nat) :
  (time_saved 20 x = 60) → x = 5 := by
  sorry

end NUMINAMATH_GPT_matt_without_calculator_5_minutes_l1272_127241


namespace NUMINAMATH_GPT_completing_square_result_l1272_127230

theorem completing_square_result:
  ∀ x : ℝ, (x^2 + 4 * x - 1 = 0) → (x + 2)^2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_completing_square_result_l1272_127230


namespace NUMINAMATH_GPT_calculate_result_l1272_127275

def binary_op (x y : ℝ) : ℝ := x^2 + y^2

theorem calculate_result (h : ℝ) : binary_op (binary_op h h) (binary_op h h) = 8 * h^4 :=
by
  sorry

end NUMINAMATH_GPT_calculate_result_l1272_127275


namespace NUMINAMATH_GPT_denominator_divisor_zero_l1272_127259

theorem denominator_divisor_zero (n : ℕ) : n ≠ 0 → (∀ d, d ≠ 0 → d / n ≠ d / 0) :=
by
  sorry

end NUMINAMATH_GPT_denominator_divisor_zero_l1272_127259


namespace NUMINAMATH_GPT_possible_original_numbers_l1272_127292

def four_digit_original_number (N : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 
    N = 1000 * a + 100 * b + 10 * c + d ∧ 
    (a+1) * (b+2) * (c+3) * (d+4) = 234 ∧ 
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10

theorem possible_original_numbers : 
  four_digit_original_number 1109 ∨ four_digit_original_number 2009 :=
sorry

end NUMINAMATH_GPT_possible_original_numbers_l1272_127292


namespace NUMINAMATH_GPT_log_sqrt_defined_in_interval_l1272_127298

def defined_interval (x : ℝ) : Prop :=
  ∃ y, y = (5 - x) ∧ y > 0 ∧ (x - 2) ≥ 0

theorem log_sqrt_defined_in_interval {x : ℝ} :
  defined_interval x ↔ (2 < x ∧ x < 5) :=
sorry

end NUMINAMATH_GPT_log_sqrt_defined_in_interval_l1272_127298


namespace NUMINAMATH_GPT_right_triangle_area_l1272_127211

def roots (a b : ℝ) : Prop :=
  a * b = 12 ∧ a + b = 7

def area (A : ℝ) : Prop :=
  A = 6 ∨ A = 3 * Real.sqrt 7 / 2

theorem right_triangle_area (a b A : ℝ) (h : roots a b) : area A := 
by 
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_right_triangle_area_l1272_127211


namespace NUMINAMATH_GPT_solving_linear_equations_count_l1272_127294

def total_problems : ℕ := 140
def algebra_percentage : ℝ := 0.40
def algebra_problems := (total_problems : ℝ) * algebra_percentage
def solving_linear_equations_percentage : ℝ := 0.50
def solving_linear_equations_problems := algebra_problems * solving_linear_equations_percentage

theorem solving_linear_equations_count :
  solving_linear_equations_problems = 28 :=
by
  sorry

end NUMINAMATH_GPT_solving_linear_equations_count_l1272_127294


namespace NUMINAMATH_GPT_smallest_range_l1272_127210

theorem smallest_range {x1 x2 x3 x4 x5 : ℝ} 
  (h1 : (x1 + x2 + x3 + x4 + x5) = 100)
  (h2 : x3 = 18)
  (h3 : 2 * x1 + 2 * x5 + 18 = 100): 
  x5 - x1 = 19 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_range_l1272_127210


namespace NUMINAMATH_GPT_sqrt_calculation_l1272_127270

theorem sqrt_calculation : Real.sqrt ((5: ℝ)^2 - (4: ℝ)^2 - (3: ℝ)^2) = 0 := 
by
  -- The proof is skipped
  sorry

end NUMINAMATH_GPT_sqrt_calculation_l1272_127270


namespace NUMINAMATH_GPT_value_of_k_l1272_127285

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

noncomputable def f (k x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

theorem value_of_k (k : ℝ) :
  is_even_function (f k) → k = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_k_l1272_127285


namespace NUMINAMATH_GPT_totalCoatsCollected_l1272_127260

-- Definitions from the conditions
def highSchoolCoats : Nat := 6922
def elementarySchoolCoats : Nat := 2515

-- Theorem that proves the total number of coats collected
theorem totalCoatsCollected : highSchoolCoats + elementarySchoolCoats = 9437 := by
  sorry

end NUMINAMATH_GPT_totalCoatsCollected_l1272_127260


namespace NUMINAMATH_GPT_find_specific_linear_function_l1272_127238

-- Define the linear function with given conditions
def linear_function (k b : ℝ) (x : ℝ) := k * x + b

-- Define the condition that the point lies on the line
def passes_through (k b : ℝ) (x y : ℝ) := y = linear_function k b x

-- Define the condition that slope is negative
def slope_negative (k : ℝ) := k < 0

-- The specific function we want to prove
def specific_linear_function (x : ℝ) := -x + 1

-- The theorem to prove
theorem find_specific_linear_function : 
  ∃ (k b : ℝ), slope_negative k ∧ passes_through k b 0 1 ∧ 
  ∀ x, linear_function k b x = specific_linear_function x :=
by
  sorry

end NUMINAMATH_GPT_find_specific_linear_function_l1272_127238


namespace NUMINAMATH_GPT_vertex_angle_isosceles_triangle_l1272_127279

theorem vertex_angle_isosceles_triangle (B V : ℝ) (h1 : 2 * B + V = 180) (h2 : B = 40) : V = 100 :=
by
  sorry

end NUMINAMATH_GPT_vertex_angle_isosceles_triangle_l1272_127279


namespace NUMINAMATH_GPT_num_sets_N_l1272_127232

open Set

-- Define the set M and the set U
def M : Set ℕ := {1, 2}
def U : Set ℕ := {1, 2, 3, 4}

-- The statement to prove
theorem num_sets_N : 
  ∃ count : ℕ, count = 4 ∧ 
  (∀ N : Set ℕ, M ∪ N = U → N = {3, 4} ∨ N = {1, 3, 4} ∨ N = {2, 3, 4} ∨ N = {1, 2, 3, 4}) :=
by
  sorry

end NUMINAMATH_GPT_num_sets_N_l1272_127232


namespace NUMINAMATH_GPT_solve_for_z_l1272_127243

theorem solve_for_z (z : ℂ) (h : 5 - 3 * (I * z) = 3 + 5 * (I * z)) : z = I / 4 :=
sorry

end NUMINAMATH_GPT_solve_for_z_l1272_127243


namespace NUMINAMATH_GPT_sin_cos_solution_set_l1272_127258
open Real

theorem sin_cos_solution_set :
  {x : ℝ | ∃ k : ℤ, x = k * π + (-1)^k * (π / 6) - (π / 3)} =
  {x : ℝ | sin x + sqrt 3 * cos x = 1} :=
by sorry

end NUMINAMATH_GPT_sin_cos_solution_set_l1272_127258


namespace NUMINAMATH_GPT_cereal_discount_l1272_127289

theorem cereal_discount (milk_normal_cost milk_discounted_cost total_savings milk_quantity cereal_quantity: ℝ) 
  (total_milk_savings cereal_savings_per_box: ℝ) 
  (h1: milk_normal_cost = 3)
  (h2: milk_discounted_cost = 2)
  (h3: total_savings = 8)
  (h4: milk_quantity = 3)
  (h5: cereal_quantity = 5)
  (h6: total_milk_savings = milk_quantity * (milk_normal_cost - milk_discounted_cost)) 
  (h7: total_milk_savings + cereal_quantity * cereal_savings_per_box = total_savings):
  cereal_savings_per_box = 1 :=
by 
  sorry

end NUMINAMATH_GPT_cereal_discount_l1272_127289


namespace NUMINAMATH_GPT_ratio_surface_area_cube_to_octahedron_l1272_127290

noncomputable def cube_side_length := 1

noncomputable def surface_area_cube (s : ℝ) : ℝ := 6 * s^2

noncomputable def edge_length_octahedron := 1

-- Surface area formula for a regular octahedron with side length e is 2 * sqrt(3) * e^2
noncomputable def surface_area_octahedron (e : ℝ) : ℝ := 2 * Real.sqrt 3 * e^2

-- Finally, we want to prove that the ratio of the surface area of the cube to that of the octahedron is sqrt(3)
theorem ratio_surface_area_cube_to_octahedron :
  surface_area_cube cube_side_length / surface_area_octahedron edge_length_octahedron = Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_ratio_surface_area_cube_to_octahedron_l1272_127290


namespace NUMINAMATH_GPT_john_total_hours_l1272_127235

def wall_area (length : ℕ) (width : ℕ) := length * width

def total_area (num_walls : ℕ) (wall_area : ℕ) := num_walls * wall_area

def time_to_paint (area : ℕ) (time_per_square_meter : ℕ) := area * time_per_square_meter

def hours_to_minutes (hours : ℕ) := hours * 60

def total_hours (painting_time : ℕ) (spare_time : ℕ) := painting_time + spare_time

theorem john_total_hours 
  (length width num_walls time_per_square_meter spare_hours : ℕ) 
  (H_length : length = 2) 
  (H_width : width = 3) 
  (H_num_walls : num_walls = 5)
  (H_time_per_square_meter : time_per_square_meter = 10)
  (H_spare_hours : spare_hours = 5) :
  total_hours (time_to_paint (total_area num_walls (wall_area length width)) time_per_square_meter / hours_to_minutes 1) spare_hours = 10 := 
by 
    rw [H_length, H_width, H_num_walls, H_time_per_square_meter, H_spare_hours]
    sorry

end NUMINAMATH_GPT_john_total_hours_l1272_127235


namespace NUMINAMATH_GPT_inverse_of_composite_l1272_127202

-- Define the function g
def g (x : ℕ) : ℕ :=
  if x = 1 then 4 else
  if x = 2 then 3 else
  if x = 3 then 1 else
  if x = 4 then 5 else
  if x = 5 then 2 else
  0  -- g is not defined for values other than 1 to 5

-- Define the inverse g_inv
def g_inv (x : ℕ) : ℕ :=
  if x = 4 then 1 else
  if x = 3 then 2 else
  if x = 1 then 3 else
  if x = 5 then 4 else
  if x = 2 then 5 else
  0  -- g_inv is not defined for values other than 1 to 5

theorem inverse_of_composite :
  g_inv (g_inv (g_inv 3)) = 4 :=
by
  sorry

end NUMINAMATH_GPT_inverse_of_composite_l1272_127202


namespace NUMINAMATH_GPT_example_function_indeterminate_unbounded_l1272_127214

theorem example_function_indeterminate_unbounded:
  (∀ x, ∃ f : ℝ → ℝ, (f x = (x^2 + x - 2) / (x^3 + 2 * x + 1)) ∧ 
                      (f 1 = (0 / (1^3 + 2 * 1 + 1))) ∧
                      (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 1) < δ → abs (f x) > ε)) :=
by
  sorry

end NUMINAMATH_GPT_example_function_indeterminate_unbounded_l1272_127214


namespace NUMINAMATH_GPT_find_sequence_l1272_127286

noncomputable def sequence_satisfies (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (1 / 2) * (a n + 1 / (a n))

theorem find_sequence (a : ℕ → ℝ) (S : ℕ → ℝ)
    (h_pos : ∀ n, 0 < a n)
    (h_S : sequence_satisfies a S) :
    ∀ n, a n = Real.sqrt n - Real.sqrt (n - 1) :=
sorry

end NUMINAMATH_GPT_find_sequence_l1272_127286


namespace NUMINAMATH_GPT_jerry_current_average_l1272_127213

theorem jerry_current_average (A : ℚ) (h1 : 3 * A + 89 = 4 * (A + 2)) : A = 81 := 
by
  sorry

end NUMINAMATH_GPT_jerry_current_average_l1272_127213


namespace NUMINAMATH_GPT_exact_sequence_a2007_l1272_127239

theorem exact_sequence_a2007 (a : ℕ → ℤ) 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 0) 
  (exact : ∀ n m : ℕ, n > m → a n ^ 2 - a m ^ 2 = a (n - m) * a (n + m)) :
  a 2007 = -1 := 
sorry

end NUMINAMATH_GPT_exact_sequence_a2007_l1272_127239


namespace NUMINAMATH_GPT_positional_relationship_l1272_127234

-- Definitions of the lines l1 and l2
def l1 (m x y : ℝ) : Prop := (m + 3) * x + 5 * y = 5 - 3 * m
def l2 (m x y : ℝ) : Prop := 2 * x + (m + 6) * y = 8

theorem positional_relationship (m : ℝ) :
  (∃ x y : ℝ, l1 m x y ∧ l2 m x y) ∨ (∀ x y : ℝ, l1 m x y ↔ l2 m x y) ∨
  ¬(∃ x y : ℝ, l1 m x y ∨ l2 m x y) :=
sorry

end NUMINAMATH_GPT_positional_relationship_l1272_127234


namespace NUMINAMATH_GPT_total_trees_planted_l1272_127227

theorem total_trees_planted (apple_trees orange_trees : ℕ) (h₁ : apple_trees = 47) (h₂ : orange_trees = 27) : apple_trees + orange_trees = 74 := 
by
  -- We skip the proof step
  sorry

end NUMINAMATH_GPT_total_trees_planted_l1272_127227


namespace NUMINAMATH_GPT_loaves_of_bread_l1272_127244

variable (B : ℕ) -- Number of loaves of bread Erik bought
variable (total_money : ℕ := 86) -- Money given to Erik
variable (money_left : ℕ := 59) -- Money left after purchase
variable (cost_bread : ℕ := 3) -- Cost of each loaf of bread
variable (cost_oj : ℕ := 6) -- Cost of each carton of orange juice
variable (num_oj : ℕ := 3) -- Number of cartons of orange juice bought

theorem loaves_of_bread (h1 : total_money - money_left = num_oj * cost_oj + B * cost_bread) : B = 3 := 
by sorry

end NUMINAMATH_GPT_loaves_of_bread_l1272_127244


namespace NUMINAMATH_GPT_solution_problem_l1272_127257

noncomputable def problem :=
  ∀ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 1 →
  2 ≤ (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ∧
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≤ (1 + a) * (1 + b) * (1 + c)

theorem solution_problem : problem :=
  sorry

end NUMINAMATH_GPT_solution_problem_l1272_127257


namespace NUMINAMATH_GPT_max_s_value_l1272_127288

variables (X Y Z P X' Y' Z' : Type)
variables (p q r XX' YY' ZZ' s : ℝ)

-- Defining the conditions
def triangle_XYZ (p q r : ℝ) : Prop :=
p ≤ r ∧ r ≤ q ∧ p + q > r ∧ p + r > q ∧ q + r > p

def point_P_inside (X Y Z P : Type) : Prop :=
true -- Simplified assumption since point P is given to be inside

def segments_XX'_YY'_ZZ' (XX' YY' ZZ' : ℝ) : ℝ :=
XX' + YY' + ZZ'

def given_ratio (p q r : ℝ) : Prop :=
(p / (q + r)) = (r / (p + q))

-- The maximum value of s being 3p
def max_value_s_eq_3p (s p : ℝ) : Prop :=
s = 3 * p

-- The final theorem statement
theorem max_s_value 
  (p q r XX' YY' ZZ' s : ℝ)
  (h_triangle : triangle_XYZ p q r)
  (h_ratio : given_ratio p q r)
  (h_segments : s = segments_XX'_YY'_ZZ' XX' YY' ZZ') : 
  max_value_s_eq_3p s p :=
by
  sorry

end NUMINAMATH_GPT_max_s_value_l1272_127288


namespace NUMINAMATH_GPT_number_of_whole_numbers_l1272_127236

theorem number_of_whole_numbers (x y : ℝ) (hx : 2 < x ∧ x < 3) (hy : 8 < y ∧ y < 9) : 
  ∃ (n : ℕ), n = 6 := by
  sorry

end NUMINAMATH_GPT_number_of_whole_numbers_l1272_127236


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1272_127218

theorem necessary_but_not_sufficient_condition {a b : ℝ} (ha : a > 0) (hb : b > 0) : 
  ((a + b > 1) ↔ (ab > 1)) → false :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1272_127218


namespace NUMINAMATH_GPT_ratio_of_coconut_flavored_red_jelly_beans_l1272_127242

theorem ratio_of_coconut_flavored_red_jelly_beans :
  ∀ (total_jelly_beans jelly_beans_coconut_flavored : ℕ)
    (three_fourths_red : total_jelly_beans > 0 ∧ (3/4 : ℝ) * total_jelly_beans = 3 * (total_jelly_beans / 4))
    (h1 : jelly_beans_coconut_flavored = 750)
    (h2 : total_jelly_beans = 4000),
  (250 : ℝ)/(3000 : ℝ) = 1/4 :=
by
  intros total_jelly_beans jelly_beans_coconut_flavored three_fourths_red h1 h2
  sorry

end NUMINAMATH_GPT_ratio_of_coconut_flavored_red_jelly_beans_l1272_127242


namespace NUMINAMATH_GPT_smallest_positive_angle_l1272_127261

theorem smallest_positive_angle (α : ℝ) (h : (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3)) = (Real.sin α, Real.cos α)) : 
  α = 11 * Real.pi / 6 := by
sorry

end NUMINAMATH_GPT_smallest_positive_angle_l1272_127261


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_cond_l1272_127265

open Set

variable {α : Type*} (A B C : Set α)

/-- Mathematical equivalent proof problem statement -/
theorem necessary_but_not_sufficient_cond (h1 : A ∪ B = C) (h2 : ¬ B ⊆ A) (hA : A.Nonempty) (hB : B.Nonempty) (hC : C.Nonempty) :
  (∀ x, x ∈ A → x ∈ C) ∧ (∃ y ∈ C, y ∉ A) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_cond_l1272_127265


namespace NUMINAMATH_GPT_weights_sum_l1272_127251

theorem weights_sum (e f g h : ℕ) (h₁ : e + f = 280) (h₂ : f + g = 230) (h₃ : e + h = 300) : g + h = 250 := 
by 
  sorry

end NUMINAMATH_GPT_weights_sum_l1272_127251


namespace NUMINAMATH_GPT_most_precise_value_l1272_127267

def D := 3.27645
def error := 0.00518
def D_upper := D + error
def D_lower := D - error
def rounded_D_upper := Float.round (D_upper * 10) / 10
def rounded_D_lower := Float.round (D_lower * 10) / 10

theorem most_precise_value :
  rounded_D_upper = 3.3 ∧ rounded_D_lower = 3.3 → rounded_D_upper = 3.3 :=
by sorry

end NUMINAMATH_GPT_most_precise_value_l1272_127267


namespace NUMINAMATH_GPT_hh_of_2_eq_91265_l1272_127237

def h (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 - x + 1

theorem hh_of_2_eq_91265 : h (h 2) = 91265 := by
  sorry

end NUMINAMATH_GPT_hh_of_2_eq_91265_l1272_127237
