import Mathlib

namespace A_more_likely_than_B_l813_81394

-- Define the conditions
variables (n : ℕ) (k : ℕ)
-- n is the total number of programs, k is the chosen number of programs
def total_programs : ℕ := 10
def selected_programs : ℕ := 3
-- Probability of person B correctly completing each program
def probability_B_correct : ℚ := 3/5
-- Person A can correctly complete 6 out of 10 programs
def person_A_correct : ℕ := 6

-- The probability of person B successfully completing the challenge
def probability_B_success : ℚ := (3 * (9/25) * (2/5)) + (27/125)

-- Define binomial coefficient function for easier combination calculations
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The probabilities for the number of correct programs for person A
def P_X_0 : ℚ := (choose 4 3 : ℕ) / (choose 10 3 : ℕ)
def P_X_1 : ℚ := (choose 6 1 * choose 4 2 : ℕ) / (choose 10 3 : ℕ)
def P_X_2 : ℚ := (choose 6 2 * choose 4 1 : ℕ) / (choose 10 3 : ℕ)
def P_X_3 : ℚ := (choose 6 3 : ℕ) / (choose 10 3 : ℕ)

-- The distribution and expectation of X for person A
def E_X : ℚ := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 + 3 * P_X_3

-- The probability of person A successfully completing the challenge
def P_A_success : ℚ := P_X_2 + P_X_3

-- Final comparisons to determine who is more likely to succeed
def compare_success : Prop := P_A_success > probability_B_success

-- Lean statement
theorem A_more_likely_than_B : compare_success := by
  sorry

end A_more_likely_than_B_l813_81394


namespace sector_properties_l813_81327

-- Definitions for the conditions
def central_angle (α : ℝ) : Prop := α = 2 * Real.pi / 3

def radius (r : ℝ) : Prop := r = 6

def sector_perimeter (l r : ℝ) : Prop := l + 2 * r = 20

-- The statement encapsulating the proof problem
theorem sector_properties :
  (central_angle (2 * Real.pi / 3) ∧ radius 6 →
    ∃ l S, l = 4 * Real.pi ∧ S = 12 * Real.pi) ∧
  (∃ l r, sector_perimeter l r ∧ 
    ∃ α S, α = 2 ∧ S = 25) := by
  sorry

end sector_properties_l813_81327


namespace smaller_side_of_rectangle_l813_81302

theorem smaller_side_of_rectangle (r : ℝ) (h1 : r = 42) 
                                   (h2 : ∀ L W : ℝ, L / W = 6 / 5 → 2 * (L + W) = 2 * π * r) : 
                                   ∃ W : ℝ, W = (210 * π) / 11 := 
by {
    sorry
}

end smaller_side_of_rectangle_l813_81302


namespace maximize_greenhouse_planting_area_l813_81330

theorem maximize_greenhouse_planting_area
    (a b : ℝ)
    (h : a * b = 800)
    (planting_area : ℝ := (a - 4) * (b - 2)) :
  (a = 40 ∧ b = 20) ↔ planting_area = 648 :=
by
  sorry

end maximize_greenhouse_planting_area_l813_81330


namespace numbers_with_digit_one_are_more_numerous_l813_81338

theorem numbers_with_digit_one_are_more_numerous :
  let n := 9999998
  let total_numbers := 10^7
  let numbers_without_one := 9^7 
  total_numbers - numbers_without_one > numbers_without_one :=
by
  let n := 9999998
  let total_numbers := 10^7
  let numbers_without_one := 9^7 
  show total_numbers - numbers_without_one > numbers_without_one
  sorry

end numbers_with_digit_one_are_more_numerous_l813_81338


namespace find_three_digit_number_in_decimal_l813_81304

theorem find_three_digit_number_in_decimal :
  ∃ (A B C : ℕ), ∀ (hA : A ≠ 0 ∧ A < 7) (hB : B ≠ 0 ∧ B < 7) (hC : C ≠ 0 ∧ C < 7) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
    (h1 : (7 * A + B) + C = 7 * C)
    (h2 : (7 * A + B) + (7 * B + A) = 7 * B + 6), 
    A * 100 + B * 10 + C = 425 :=
by
  sorry

end find_three_digit_number_in_decimal_l813_81304


namespace weaving_output_first_day_l813_81347

theorem weaving_output_first_day (x : ℝ) :
  (x + 2*x + 4*x + 8*x + 16*x = 5) → x = 5 / 31 :=
by
  intros h
  sorry

end weaving_output_first_day_l813_81347


namespace quotient_zero_l813_81314

theorem quotient_zero (D d R Q : ℕ) (hD : D = 12) (hd : d = 17) (hR : R = 8) (h : D = d * Q + R) : Q = 0 :=
by
  sorry

end quotient_zero_l813_81314


namespace find_ratio_of_radii_l813_81300

noncomputable def ratio_of_radii (a b : ℝ) (h1 : π * b ^ 2 - π * a ^ 2 = 4 * π * a ^ 2) : Prop :=
  a / b = Real.sqrt 5 / 5

theorem find_ratio_of_radii (a b : ℝ) (h1 : π * b ^ 2 - π * a ^ 2 = 4 * π * a ^ 2) :
  ratio_of_radii a b h1 :=
sorry

end find_ratio_of_radii_l813_81300


namespace probability_of_snow_at_least_once_l813_81385

/-- Probability of snow during the first week of January -/
theorem probability_of_snow_at_least_once :
  let p_no_snow_first_four_days := (3/4 : ℚ)
  let p_no_snow_next_three_days := (2/3 : ℚ)
  let p_no_snow_first_week := p_no_snow_first_four_days^4 * p_no_snow_next_three_days^3
  let p_snow_at_least_once := 1 - p_no_snow_first_week
  p_snow_at_least_once = 68359 / 100000 :=
by
  sorry

end probability_of_snow_at_least_once_l813_81385


namespace giyoon_above_average_subjects_l813_81337

def points_korean : ℕ := 80
def points_mathematics : ℕ := 94
def points_social_studies : ℕ := 82
def points_english : ℕ := 76
def points_science : ℕ := 100
def number_of_subjects : ℕ := 5

def total_points : ℕ := points_korean + points_mathematics + points_social_studies + points_english + points_science
def average_points : ℚ := total_points / number_of_subjects

def count_above_average_points : ℕ := 
  (if points_korean > average_points then 1 else 0) + 
  (if points_mathematics > average_points then 1 else 0) +
  (if points_social_studies > average_points then 1 else 0) +
  (if points_english > average_points then 1 else 0) +
  (if points_science > average_points then 1 else 0)

theorem giyoon_above_average_subjects : count_above_average_points = 2 := by
  sorry

end giyoon_above_average_subjects_l813_81337


namespace total_cost_is_96_l813_81377

noncomputable def hair_updo_cost : ℕ := 50
noncomputable def manicure_cost : ℕ := 30
noncomputable def tip_rate : ℚ := 0.20

def total_cost_with_tip (hair_cost manicure_cost : ℕ) (tip_rate : ℚ) : ℚ :=
  let hair_tip := hair_cost * tip_rate
  let manicure_tip := manicure_cost * tip_rate
  let total_tips := hair_tip + manicure_tip
  let total_before_tips := (hair_cost : ℚ) + (manicure_cost : ℚ)
  total_before_tips + total_tips

theorem total_cost_is_96 :
  total_cost_with_tip hair_updo_cost manicure_cost tip_rate = 96 := by
  sorry

end total_cost_is_96_l813_81377


namespace exponential_inequality_l813_81363

-- Define the conditions
variables {n : ℤ} {x : ℝ}

theorem exponential_inequality 
  (h1 : n ≥ 2) 
  (h2 : |x| < 1) 
  : 2^n > (1 - x)^n + (1 + x)^n :=
sorry

end exponential_inequality_l813_81363


namespace smallest_n_l813_81397

noncomputable def smallest_positive_integer (x y : ℤ) (h1 : (x + 1) % 7 = 0) (h2 : (y - 5) % 7 = 0) : ℕ :=
  if 3 % 7 = 0 then 7 else 7

theorem smallest_n (x y : ℤ) (h1 : (x + 1) % 7 = 0) (h2 : (y - 5) % 7 = 0) : smallest_positive_integer x y h1 h2 = 7 := 
  by
  admit

end smallest_n_l813_81397


namespace age_difference_l813_81372

theorem age_difference (a b c : ℕ) (h₁ : b = 8) (h₂ : c = b / 2) (h₃ : a + b + c = 22) : a - b = 2 :=
by
  sorry

end age_difference_l813_81372


namespace r_expansion_l813_81393

theorem r_expansion (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by
  sorry

end r_expansion_l813_81393


namespace porter_previous_painting_price_l813_81348

-- definitions from the conditions
def most_recent_sale : ℕ := 44000

-- definitions for the problem statement
def sale_equation (P : ℕ) : Prop :=
  most_recent_sale = 5 * P - 1000

theorem porter_previous_painting_price (P : ℕ) (h : sale_equation P) : P = 9000 :=
by {
  sorry
}

end porter_previous_painting_price_l813_81348


namespace latoya_initial_payment_l813_81369

variable (cost_per_minute : ℝ) (call_duration : ℝ) (remaining_credit : ℝ) 
variable (initial_credit : ℝ)

theorem latoya_initial_payment : 
  ∀ (cost_per_minute call_duration remaining_credit initial_credit : ℝ),
  cost_per_minute = 0.16 →
  call_duration = 22 →
  remaining_credit = 26.48 →
  initial_credit = (cost_per_minute * call_duration) + remaining_credit →
  initial_credit = 30 :=
by
  intros cost_per_minute call_duration remaining_credit initial_credit
  sorry

end latoya_initial_payment_l813_81369


namespace negate_one_even_l813_81349

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

theorem negate_one_even (a b c : ℕ) :
  (∃! x, x = a ∨ x = b ∨ x = c ∧ is_even x) ↔
  (∃ x y, x = a ∨ x = b ∨ x = c ∧ y = a ∨ y = b ∨ y = c ∧
    x ≠ y ∧ is_even x ∧ is_even y) ∨
  (is_odd a ∧ is_odd b ∧ is_odd c) :=
by {
  sorry
}

end negate_one_even_l813_81349


namespace tom_searching_days_l813_81312

variable (d : ℕ) (total_cost : ℕ)

theorem tom_searching_days :
  (∀ n, n ≤ 5 → total_cost = n * 100 + (d - n) * 60) →
  (∀ n, n > 5 → total_cost = 5 * 100 + (d - 5) * 60) →
  total_cost = 800 →
  d = 10 :=
by
  intros h1 h2 h3
  sorry

end tom_searching_days_l813_81312


namespace evaluate_power_l813_81352

theorem evaluate_power (a : ℝ) (b : ℝ) (hb : b = 16) (hc : b = a ^ 4) : (b ^ (1 / 4)) ^ 12 = 4096 := by
  sorry

end evaluate_power_l813_81352


namespace sum_of_three_consecutive_even_numbers_l813_81360

theorem sum_of_three_consecutive_even_numbers (a : ℤ) (h : a * (a + 2) * (a + 4) = 960) : a + (a + 2) + (a + 4) = 30 := by
  sorry

end sum_of_three_consecutive_even_numbers_l813_81360


namespace lines_parallel_coeff_l813_81351

theorem lines_parallel_coeff (a : ℝ) :
  (∀ x y: ℝ, a * x + 2 * y = 0 → 3 * x + (a + 1) * y + 1 = 0) ↔ (a = -3 ∨ a = 2) :=
by
  sorry

end lines_parallel_coeff_l813_81351


namespace min_employees_wednesday_l813_81339

noncomputable def minWednesdayBirthdays (total_employees : ℕ) 
  (diff_birthdays : (List ℕ → Prop)) 
  (max_birthdays : (ℕ → List ℕ → Prop)) :
  ℕ :=
  40

theorem min_employees_wednesday (total_employees : ℕ) 
  (diff_birthdays : (List ℕ → Prop)) 
  (max_birthdays : (ℕ → List ℕ → Prop)) 
  (h1 : total_employees = 61) 
  (h2 : ∃ lst, diff_birthdays lst ∧ max_birthdays 40 lst) :
  minWednesdayBirthdays total_employees diff_birthdays max_birthdays = 40 := 
sorry

end min_employees_wednesday_l813_81339


namespace max_rectangle_area_l813_81388

theorem max_rectangle_area (l w : ℕ) (h1 : 2 * l + 2 * w = 40) : l * w ≤ 100 :=
by
  have h2 : l + w = 20 := by linarith
  -- Further steps would go here but we're just stating it
  sorry

end max_rectangle_area_l813_81388


namespace necessary_not_sufficient_l813_81374

theorem necessary_not_sufficient (x : ℝ) : (x^2 ≥ 1) ↔ (x ≥ 1 ∨ x ≤ -1) ≠ (x ≥ 1) :=
by
  sorry

end necessary_not_sufficient_l813_81374


namespace total_canoes_boatsRUs_l813_81307

-- Definitions for the conditions
def initial_production := 10
def common_ratio := 3
def months := 6

-- The function to compute the total number of canoes built using the geometric sequence sum formula
noncomputable def total_canoes (a : ℕ) (r : ℕ) (n : ℕ) := a * (r^n - 1) / (r - 1)

-- Statement of the theorem
theorem total_canoes_boatsRUs : 
  total_canoes initial_production common_ratio months = 3640 :=
sorry

end total_canoes_boatsRUs_l813_81307


namespace find_cost_price_l813_81340

noncomputable def cost_price (CP SP_loss SP_gain : ℝ) : Prop :=
SP_loss = 0.90 * CP ∧
SP_gain = 1.05 * CP ∧
(SP_gain - SP_loss = 225)

theorem find_cost_price (CP : ℝ) (h : cost_price CP (0.90 * CP) (1.05 * CP)) : CP = 1500 :=
by
  sorry

end find_cost_price_l813_81340


namespace last_digit_m_is_9_l813_81362

def x (n : ℕ) : ℕ := 2^(2^n) + 1

def m : ℕ := List.foldr Nat.lcm 1 (List.map x (List.range' 2 (1971 - 2 + 1)))

theorem last_digit_m_is_9 : m % 10 = 9 :=
  by
    sorry

end last_digit_m_is_9_l813_81362


namespace vertical_angles_always_equal_l813_81332

theorem vertical_angles_always_equal (a b : ℝ) (h : a = b) : 
  (∀ θ1 θ2, θ1 + θ2 = 180 ∧ θ1 = a ∧ θ2 = b → θ1 = θ2) :=
by 
  intro θ1 θ2 
  intro h 
  sorry

end vertical_angles_always_equal_l813_81332


namespace false_proposition_C_l813_81358

variable (a b c : ℝ)
noncomputable def f (x : ℝ) := a * x^2 + b * x + c

theorem false_proposition_C 
  (ha : a > 0)
  (x0 : ℝ)
  (hx0 : x0 = -b / (2 * a)) :
  ¬ ∀ x : ℝ, f a b c x ≤ f a b c x0 :=
by
  sorry

end false_proposition_C_l813_81358


namespace probability_odd_divisor_15_factorial_l813_81368

theorem probability_odd_divisor_15_factorial :
  let number_of_divisors_15_fact : ℕ := (11 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  let number_of_odd_divisors_15_fact : ℕ := (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  (number_of_odd_divisors_15_fact : ℝ) / (number_of_divisors_15_fact : ℝ) = 1 / 12 :=
by
  sorry

end probability_odd_divisor_15_factorial_l813_81368


namespace sqrt_problem_l813_81357

theorem sqrt_problem (a m : ℝ) (ha : 0 < a) 
  (h1 : a = (3 * m - 1) ^ 2) 
  (h2 : a = (-2 * m - 2) ^ 2) : 
  a = 64 ∨ a = 64 / 25 := 
sorry

end sqrt_problem_l813_81357


namespace correct_option_l813_81326

-- Conditions
def option_A (a : ℕ) : Prop := (a^5)^2 = a^7
def option_B (a : ℕ) : Prop := a + 2 * a = 3 * a^2
def option_C (a : ℕ) : Prop := (2 * a)^3 = 6 * a^3
def option_D (a : ℕ) : Prop := a^6 / a^2 = a^4

-- Theorem statement
theorem correct_option (a : ℕ) : ¬ option_A a ∧ ¬ option_B a ∧ ¬ option_C a ∧ option_D a := by
  sorry

end correct_option_l813_81326


namespace solution_interval_l813_81318

theorem solution_interval (x : ℝ) : 
  (3/8 + |x - 1/4| < 7/8) ↔ (-1/4 < x ∧ x < 3/4) := 
sorry

end solution_interval_l813_81318


namespace calculate_area_of_shaded_region_l813_81350

namespace Proof

noncomputable def AreaOfShadedRegion (width height : ℝ) (divisions : ℕ) : ℝ :=
  let small_width := width
  let small_height := height / divisions
  let area_of_small := small_width * small_height
  let shaded_in_small := area_of_small / 2
  let total_shaded := divisions * shaded_in_small
  total_shaded

theorem calculate_area_of_shaded_region :
  AreaOfShadedRegion 3 14 4 = 21 := by
  sorry

end Proof

end calculate_area_of_shaded_region_l813_81350


namespace is_exact_time_now_321_l813_81354

noncomputable def current_time_is_321 : Prop :=
  exists t : ℝ, 0 < t ∧ t < 60 ∧ |(6 * t + 48) - (90 + 0.5 * (t - 4))| = 180

theorem is_exact_time_now_321 : current_time_is_321 := 
  sorry

end is_exact_time_now_321_l813_81354


namespace area_enclosed_by_curve_l813_81359

theorem area_enclosed_by_curve :
  let s : ℝ := 3
  let arc_length : ℝ := (3 * Real.pi) / 4
  let octagon_area : ℝ := (1 + Real.sqrt 2) * s^2
  let sector_area : ℝ := (3 / 8) * Real.pi
  let total_area : ℝ := 8 * sector_area + octagon_area
  total_area = 9 + 9 * Real.sqrt 2 + 3 * Real.pi :=
by
  let s := 3
  let arc_length := (3 * Real.pi) / 4
  let r := arc_length / ((3 * Real.pi) / 4)
  have r_eq : r = 1 := by
    sorry
  let full_circle_area := Real.pi * r^2
  let sector_area := (3 / 8) * Real.pi
  have sector_area_eq : sector_area = (3 / 8) * Real.pi := by
    sorry
  let total_sector_area := 8 * sector_area
  have total_sector_area_eq : total_sector_area = 3 * Real.pi := by
    sorry
  let octagon_area := (1 + Real.sqrt 2) * s^2
  have octagon_area_eq : octagon_area = 9 * (1 + Real.sqrt 2) := by
    sorry
  let total_area := total_sector_area + octagon_area
  have total_area_eq : total_area = 9 + 9 * Real.sqrt 2 + 3 * Real.pi := by
    sorry
  exact total_area_eq

end area_enclosed_by_curve_l813_81359


namespace average_of_first_12_results_l813_81370

theorem average_of_first_12_results
  (average_25_results : ℝ)
  (average_last_12_results : ℝ)
  (result_13th : ℝ)
  (total_results : ℕ)
  (num_first_12 : ℕ)
  (num_last_12 : ℕ)
  (total_sum : ℝ)
  (sum_first_12 : ℝ)
  (sum_last_12 : ℝ)
  (A : ℝ)
  (h1 : average_25_results = 24)
  (h2 : average_last_12_results = 17)
  (h3 : result_13th = 228)
  (h4 : total_results = 25)
  (h5 : num_first_12 = 12)
  (h6 : num_last_12 = 12)
  (h7 : total_sum = average_25_results * total_results)
  (h8 : sum_last_12 = average_last_12_results * num_last_12)
  (h9 : total_sum = sum_first_12 + result_13th + sum_last_12)
  (h10 : sum_first_12 = A * num_first_12) :
  A = 14 :=
by
  sorry

end average_of_first_12_results_l813_81370


namespace willie_stickers_l813_81334

theorem willie_stickers (initial_stickers : ℕ) (given_stickers : ℕ) (final_stickers : ℕ) 
  (h1 : initial_stickers = 124) 
  (h2 : given_stickers = 43) 
  (h3 : final_stickers = initial_stickers - given_stickers) :
  final_stickers = 81 :=
sorry

end willie_stickers_l813_81334


namespace gray_part_area_l813_81396

theorem gray_part_area (area_rect1 area_rect2 area_black area_white gray_part_area : ℕ)
  (h_rect1 : area_rect1 = 80)
  (h_rect2 : area_rect2 = 108)
  (h_black : area_black = 37)
  (h_white : area_white = area_rect1 - area_black)
  (h_white_correct : area_white = 43)
  : gray_part_area = area_rect2 - area_white :=
by
  sorry

end gray_part_area_l813_81396


namespace remainder_13_plus_x_l813_81395

theorem remainder_13_plus_x (x : ℕ) (h1 : 7 * x % 31 = 1) : (13 + x) % 31 = 22 := 
by
  sorry

end remainder_13_plus_x_l813_81395


namespace chess_tournament_participants_l813_81398

-- Define the number of grandmasters
variables (x : ℕ)

-- Define the number of masters as three times the number of grandmasters
def num_masters : ℕ := 3 * x

-- Condition on total points scored: Master's points is 1.2 times the Grandmaster's points
def points_condition (g m : ℕ) : Prop := m = 12 * g / 10

-- Proposition that the total number of participants is 12
theorem chess_tournament_participants (x_nonnegative: 0 < x) (g m : ℕ)
  (masters_points: points_condition g m) : 
  4 * x = 12 := 
sorry

end chess_tournament_participants_l813_81398


namespace least_number_with_remainder_l813_81311

theorem least_number_with_remainder (x : ℕ) :
  (x % 6 = 4) ∧ (x % 7 = 4) ∧ (x % 9 = 4) ∧ (x % 18 = 4) ↔ x = 130 :=
by
  sorry

end least_number_with_remainder_l813_81311


namespace ratio_of_volumes_l813_81361

theorem ratio_of_volumes (s : ℝ) (hs : s > 0) :
  let r_s := s / 2
  let r_c := s / 2
  let V_sphere := (4 / 3) * π * (r_s ^ 3)
  let V_cylinder := π * (r_c ^ 2) * s
  let V_total := V_sphere + V_cylinder
  let V_cube := s ^ 3
  V_total / V_cube = (5 * π) / 12 := by {
    -- Given the conditions and expressions
    sorry
  }

end ratio_of_volumes_l813_81361


namespace correct_propositions_l813_81367

theorem correct_propositions :
  let proposition1 := (∀ A B C : ℝ, C = (A + B) / 2 → C = (A + B) / 2)
  let proposition2 := (∀ a : ℝ, a - |a| = 0 → a ≥ 0)
  let proposition3 := false
  let proposition4 := (∀ a b : ℝ, |a| = |b| → a = -b)
  let proposition5 := (∀ a : ℝ, -a < 0)
  (cond1 : proposition1 = false) →
  (cond2 : proposition2 = false) →
  (cond3 : proposition3 = false) →
  (cond4 : proposition4 = true) →
  (cond5 : proposition5 = false) →
  1 = 1 :=
by
  intros
  sorry

end correct_propositions_l813_81367


namespace zeke_estimate_smaller_l813_81399

variable (x y k : ℝ)
variable (hx_pos : 0 < x)
variable (hy_pos : 0 < y)
variable (h_inequality : x > 2 * y)
variable (hk_pos : 0 < k)

theorem zeke_estimate_smaller : (x + k) - 2 * (y + k) < x - 2 * y :=
by
  sorry

end zeke_estimate_smaller_l813_81399


namespace sum_constants_l813_81383

theorem sum_constants (a b x : ℝ) 
  (h1 : (x - a) / (x + b) = (x^2 - 50 * x + 621) / (x^2 + 75 * x - 3400))
  (h2 : x^2 - 50 * x + 621 = (x - 27) * (x - 23))
  (h3 : x^2 + 75 * x - 3400 = (x - 40) * (x + 85)) :
  a + b = 112 :=
sorry

end sum_constants_l813_81383


namespace parabola_vertex_trajectory_eq_l813_81366

noncomputable def parabola_vertex_trajectory : Prop :=
  ∀ (m : ℝ), ∃ (x y : ℝ), (y = 2 * m) ∧ (x = -m^2) ∧ (y - 4 * x - 4 * m * y = 0)

theorem parabola_vertex_trajectory_eq :
  (∀ x y : ℝ, (∃ m : ℝ, y = 2 * m ∧ x = -m^2) → y^2 = -4 * x) :=
by
  sorry

end parabola_vertex_trajectory_eq_l813_81366


namespace polynomial_identity_l813_81387

theorem polynomial_identity (a_0 a_1 a_2 a_3 a_4 : ℝ) (x : ℝ) 
  (h : (2 * x + 1)^4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4) : 
  a_0 - a_1 + a_2 - a_3 + a_4 = 1 :=
by
  sorry

end polynomial_identity_l813_81387


namespace exchanges_count_l813_81355

theorem exchanges_count (n : ℕ) :
  ∀ (initial_pencils_XZ initial_pens_XL : ℕ) 
    (pencils_per_exchange pens_per_exchange : ℕ)
    (final_pencils_multiplier : ℕ)
    (pz : initial_pencils_XZ = 200) 
    (pl : initial_pens_XL = 20)
    (pe : pencils_per_exchange = 6)
    (se : pens_per_exchange = 1)
    (fm : final_pencils_multiplier = 11),
    (initial_pencils_XZ - pencils_per_exchange * n = final_pencils_multiplier * (initial_pens_XL - pens_per_exchange * n)) ↔ n = 4 :=
by
  intros initial_pencils_XZ initial_pens_XL pencils_per_exchange pens_per_exchange final_pencils_multiplier pz pl pe se fm
  sorry

end exchanges_count_l813_81355


namespace solve_inequality_l813_81316

theorem solve_inequality (a x : ℝ) :
  ((x - a) * (x - 2 * a) < 0) ↔ 
  ((a < 0 ∧ 2 * a < x ∧ x < a) ∨ (a = 0 ∧ false) ∨ (a > 0 ∧ a < x ∧ x < 2 * a)) :=
by sorry

end solve_inequality_l813_81316


namespace locus_square_l813_81335

open Real

variables {x y c1 c2 d1 d2 : ℝ}

/-- The locus of points in a square -/
theorem locus_square (h_square: d1 < d2 ∧ c1 < c2) (h_x: d1 ≤ x ∧ x ≤ d2) (h_y: c1 ≤ y ∧ y ≤ c2) :
  |y - c1| + |y - c2| = |x - d1| + |x - d2| :=
by sorry

end locus_square_l813_81335


namespace unique_8_tuple_real_l813_81303

theorem unique_8_tuple_real (x : Fin 8 → ℝ) :
  (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + (x 6 - x 7)^2 + x 7^2 = 1 / 8 →
  ∃! (y : Fin 8 → ℝ), (1 - y 0)^2 + (y 0 - y 1)^2 + (y 1 - y 2)^2 + (y 2 - y 3)^2 + (y 3 - y 4)^2 + (y 4 - y 5)^2 + (y 5 - y 6)^2 + (y 6 - y 7)^2 + y 7^2 = 1 / 8 :=
by
  sorry

end unique_8_tuple_real_l813_81303


namespace arithmetic_sequence_sum_l813_81346

-- Define the arithmetic sequence properties
def seq : List ℕ := [81, 83, 85, 87, 89, 91, 93, 95, 97, 99]
def first := 81
def last := 99
def common_diff := 2
def n := 10

-- Main theorem statement proving the desired property
theorem arithmetic_sequence_sum :
  2 * (seq.sum) = 1800 := by
  sorry

end arithmetic_sequence_sum_l813_81346


namespace find_ab_l813_81321

theorem find_ab (a b : ℝ) (h1 : a + b = 4) (h2 : a^3 + b^3 = 100) : a * b = -3 :=
by
sorry

end find_ab_l813_81321


namespace value_of_a_if_perpendicular_l813_81328

theorem value_of_a_if_perpendicular (a l : ℝ) :
  (∀ x y : ℝ, (a + l) * x + 2 * y = 0 → x - a * y = 1 → false) → a = 1 :=
by
  -- Proof is omitted
  sorry

end value_of_a_if_perpendicular_l813_81328


namespace marble_203_is_green_l813_81305

-- Define the conditions
def total_marbles : ℕ := 240
def cycle_length : ℕ := 15
def red_count : ℕ := 6
def green_count : ℕ := 5
def blue_count : ℕ := 4
def marble_pattern (n : ℕ) : String :=
  if n % cycle_length < red_count then "red"
  else if n % cycle_length < red_count + green_count then "green"
  else "blue"

-- Define the color of the 203rd marble
def marble_203 : String := marble_pattern 202

-- State the theorem
theorem marble_203_is_green : marble_203 = "green" :=
by
  sorry

end marble_203_is_green_l813_81305


namespace solve_inequality_l813_81325

noncomputable def g (x : ℝ) : ℝ := (3 * x - 8) * (x - 2) / (x - 1)

theorem solve_inequality : 
  { x : ℝ | g x ≥ 0 } = { x : ℝ | x < 1 } ∪ { x : ℝ | x ≥ 2 } :=
by
  sorry

end solve_inequality_l813_81325


namespace find_other_number_l813_81319

theorem find_other_number (x y : ℕ) (h1 : x + y = 72) (h2 : y = x + 12) (h3 : y = 42) : x = 30 := by
  sorry

end find_other_number_l813_81319


namespace solve_inequality_l813_81306

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_add (m n : ℝ) : f (m + n) = f m * f n
axiom f_pos (x : ℝ) : 0 < x → 0 < f x ∧ f x < 1

theorem solve_inequality (x : ℝ) : f (x^2) * f (2 * x - 3) > 1 ↔ -3 < x ∧ x < 1 := sorry

end solve_inequality_l813_81306


namespace highest_value_of_a_for_divisibility_l813_81344

/-- Given a number in the format of 365a2_, where 'a' is a digit (0 through 9),
prove that the highest value of 'a' that makes the number divisible by 8 is 9. -/
theorem highest_value_of_a_for_divisibility :
  ∃ (a : ℕ), a ≤ 9 ∧ (∃ (d : ℕ), d < 10 ∧ (365 * 100 + a * 10 + 20 + d) % 8 = 0 ∧ a = 9) :=
sorry

end highest_value_of_a_for_divisibility_l813_81344


namespace tracy_sold_paintings_l813_81364

-- Definitions based on conditions
def total_customers := 20
def customers_2_paintings := 4
def customers_1_painting := 12
def customers_4_paintings := 4

def paintings_per_customer_2 := 2
def paintings_per_customer_1 := 1
def paintings_per_customer_4 := 4

-- Theorem statement
theorem tracy_sold_paintings : 
  (customers_2_paintings * paintings_per_customer_2) + 
  (customers_1_painting * paintings_per_customer_1) + 
  (customers_4_paintings * paintings_per_customer_4) = 36 := 
by
  sorry

end tracy_sold_paintings_l813_81364


namespace mary_take_home_pay_l813_81384

def hourly_wage : ℝ := 8
def regular_hours : ℝ := 20
def first_overtime_hours : ℝ := 10
def second_overtime_hours : ℝ := 10
def third_overtime_hours : ℝ := 10
def remaining_overtime_hours : ℝ := 20
def social_security_tax_rate : ℝ := 0.08
def medicare_tax_rate : ℝ := 0.02
def insurance_premium : ℝ := 50

def regular_earnings := regular_hours * hourly_wage
def first_overtime_earnings := first_overtime_hours * (hourly_wage * 1.25)
def second_overtime_earnings := second_overtime_hours * (hourly_wage * 1.5)
def third_overtime_earnings := third_overtime_hours * (hourly_wage * 1.75)
def remaining_overtime_earnings := remaining_overtime_hours * (hourly_wage * 2)

def total_earnings := 
    regular_earnings + 
    first_overtime_earnings + 
    second_overtime_earnings + 
    third_overtime_earnings + 
    remaining_overtime_earnings

def social_security_tax := total_earnings * social_security_tax_rate
def medicare_tax := total_earnings * medicare_tax_rate
def total_taxes := social_security_tax + medicare_tax

def earnings_after_taxes := total_earnings - total_taxes
def earnings_take_home := earnings_after_taxes - insurance_premium

theorem mary_take_home_pay : earnings_take_home = 706 := by
  sorry

end mary_take_home_pay_l813_81384


namespace spoiled_apples_l813_81301

theorem spoiled_apples (S G : ℕ) (h1 : S + G = 8) (h2 : (G * (G - 1)) / 2 = 21) : S = 1 :=
by
  sorry

end spoiled_apples_l813_81301


namespace three_gt_sqrt_seven_l813_81322

theorem three_gt_sqrt_seven : 3 > Real.sqrt 7 := by
  sorry

end three_gt_sqrt_seven_l813_81322


namespace mark_last_shots_l813_81356

theorem mark_last_shots (h1 : 0.60 * 15 = 9) (h2 : 0.65 * 25 = 16.25) : 
  ∀ (successful_shots_first_15 successful_shots_total: ℤ),
  successful_shots_first_15 = 9 ∧ 
  successful_shots_total = 16 → 
  successful_shots_total - successful_shots_first_15 = 7 := by
  sorry

end mark_last_shots_l813_81356


namespace increase_in_output_with_assistant_l813_81353

theorem increase_in_output_with_assistant (B H : ℝ) (hB : 0 < B) (hH : 0 < H) :
  ((1.80 * B) / (0.90 * H)) / (B / H) - 1 = 1 :=
by {
  sorry
}

end increase_in_output_with_assistant_l813_81353


namespace find_ab_unique_l813_81324

theorem find_ab_unique (a b : ℕ) (h1 : a > 1) (h2 : b > a) (h3 : a ≤ 20) (h4 : b ≤ 20) (h5 : a * b = 52) (h6 : a + b = 17) : a = 4 ∧ b = 13 :=
by {
  -- Proof goes here
  sorry
}

end find_ab_unique_l813_81324


namespace largest_number_in_set_l813_81345

theorem largest_number_in_set (b : ℕ) (h₀ : 2 + 6 + b = 18) (h₁ : 2 ≤ 6 ∧ 6 ≤ b):
  b = 10 :=
sorry

end largest_number_in_set_l813_81345


namespace gabriel_pages_correct_l813_81309

-- Given conditions
def beatrix_pages : ℕ := 704

def cristobal_pages (b : ℕ) : ℕ := 3 * b + 15

def gabriel_pages (c b : ℕ) : ℕ := 3 * (c + b)

-- Problem statement
theorem gabriel_pages_correct : gabriel_pages (cristobal_pages beatrix_pages) beatrix_pages = 8493 :=
by 
  sorry

end gabriel_pages_correct_l813_81309


namespace train_speed_l813_81373

theorem train_speed (length_train time_cross : ℝ)
  (h1 : length_train = 180)
  (h2 : time_cross = 9) : 
  (length_train / time_cross) * 3.6 = 72 :=
by
  -- This is just a placeholder proof. Replace with the actual proof.
  sorry

end train_speed_l813_81373


namespace find_x_in_isosceles_triangle_l813_81331

def is_isosceles (a b c : ℝ) : Prop := (a = b) ∨ (b = c) ∨ (a = c)

def triangle_inequality (a b c : ℝ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem find_x_in_isosceles_triangle (x : ℝ) :
  is_isosceles (x + 3) (2 * x + 1) 11 ∧ triangle_inequality (x + 3) (2 * x + 1) 11 →
  (x = 8) ∨ (x = 5) :=
sorry

end find_x_in_isosceles_triangle_l813_81331


namespace market_value_of_13_percent_stock_yielding_8_percent_l813_81389

noncomputable def market_value_of_stock (yield rate dividend_per_share : ℝ) : ℝ :=
  (dividend_per_share / yield) * 100

theorem market_value_of_13_percent_stock_yielding_8_percent
  (yield_rate : ℝ) (dividend_per_share : ℝ) (market_value : ℝ)
  (h_yield_rate : yield_rate = 0.08)
  (h_dividend_per_share : dividend_per_share = 13) :
  market_value = 162.50 :=
by
  sorry

end market_value_of_13_percent_stock_yielding_8_percent_l813_81389


namespace value_of_a_minus_b_l813_81376

theorem value_of_a_minus_b (a b : ℝ) (h1 : 2 * a + b = 7) (h2 : 2 * a - b = 1) : a - b = -1 :=
by
  sorry

end value_of_a_minus_b_l813_81376


namespace income_distribution_after_tax_l813_81342

theorem income_distribution_after_tax (x : ℝ) (hx : 10 * x = 100) :
  let poor_income_initial := x
  let middle_income_initial := 4 * x
  let rich_income_initial := 5 * x
  let tax_rate := (x^2 / 4) + x
  let tax_collected := tax_rate * rich_income_initial / 100
  let poor_income_after := poor_income_initial + 3 / 4 * tax_collected
  let middle_income_after := middle_income_initial + 1 / 4 * tax_collected
  let rich_income_after := rich_income_initial - tax_collected
  poor_income_after = 0.23125 * 100 ∧
  middle_income_after = 0.44375 * 100 ∧
  rich_income_after = 0.325 * 100 :=
by {
  sorry
}

end income_distribution_after_tax_l813_81342


namespace projectile_time_to_meet_l813_81310

theorem projectile_time_to_meet
  (d v1 v2 : ℝ)
  (hd : d = 1455)
  (hv1 : v1 = 470)
  (hv2 : v2 = 500) :
  (d / (v1 + v2)) * 60 = 90 := by
  sorry

end projectile_time_to_meet_l813_81310


namespace angle_D_is_20_degrees_l813_81386

theorem angle_D_is_20_degrees (A B C D : ℝ) 
  (h1 : A + B = 180)
  (h2 : C = D)
  (h3 : A = 40)
  (h4 : B + C = 160) : D = 20 :=
by
  sorry

end angle_D_is_20_degrees_l813_81386


namespace number_of_snakes_l813_81381

-- Define the variables
variable (S : ℕ) -- Number of snakes

-- Define the cost constants
def cost_per_gecko := 15
def cost_per_iguana := 5
def cost_per_snake := 10

-- Define the number of each pet
def num_geckos := 3
def num_iguanas := 2

-- Define the yearly cost
def yearly_cost := 1140

-- Calculate the total monthly cost
def monthly_cost := num_geckos * cost_per_gecko + num_iguanas * cost_per_iguana + S * cost_per_snake

-- Calculate the total yearly cost
def total_yearly_cost := 12 * monthly_cost

-- Prove the number of snakes
theorem number_of_snakes : total_yearly_cost = yearly_cost → S = 4 := by
  sorry

end number_of_snakes_l813_81381


namespace sin_double_angle_half_l813_81320

theorem sin_double_angle_half (θ : ℝ) (h : Real.tan θ + 1 / Real.tan θ = 4) : Real.sin (2 * θ) = 1 / 2 :=
by
  sorry

end sin_double_angle_half_l813_81320


namespace Tyler_has_200_puppies_l813_81371

-- Define the number of dogs
def numDogs : ℕ := 25

-- Define the number of puppies per dog
def puppiesPerDog : ℕ := 8

-- Define the total number of puppies
def totalPuppies : ℕ := numDogs * puppiesPerDog

-- State the theorem we want to prove
theorem Tyler_has_200_puppies : totalPuppies = 200 := by
  exact (by norm_num : 25 * 8 = 200)

end Tyler_has_200_puppies_l813_81371


namespace pencil_count_l813_81341

def total_pencils (drawer : Nat) (desk_0 : Nat) (add_dan : Nat) (remove_sarah : Nat) : Nat :=
  let desk_1 := desk_0 + add_dan
  let desk_2 := desk_1 - remove_sarah
  drawer + desk_2

theorem pencil_count :
  total_pencils 43 19 16 7 = 71 :=
by
  sorry

end pencil_count_l813_81341


namespace objective_function_range_l813_81375

theorem objective_function_range (x y : ℝ) 
  (h1 : x + 2 * y > 2) 
  (h2 : 2 * x + y ≤ 4) 
  (h3 : 4 * x - y ≥ 1) : 
  ∃ z_min z_max : ℝ, (∀ z : ℝ, z = 3 * x + y → z_min ≤ z ∧ z ≤ z_max) ∧ z_min = 1 ∧ z_max = 6 := 
sorry

end objective_function_range_l813_81375


namespace remainder_when_divided_by_8_l813_81390

theorem remainder_when_divided_by_8 (x k : ℤ) (h : x = 63 * k + 27) : x % 8 = 3 :=
sorry

end remainder_when_divided_by_8_l813_81390


namespace min_trips_to_fill_hole_l813_81392

def hole_filling_trips (initial_gallons : ℕ) (required_gallons : ℕ) (capacity_2gallon : ℕ)
  (capacity_5gallon : ℕ) (capacity_8gallon : ℕ) (time_limit : ℕ) (time_per_trip : ℕ) : ℕ :=
  if initial_gallons < required_gallons then
    let remaining_gallons := required_gallons - initial_gallons
    let num_8gallon := remaining_gallons / capacity_8gallon
    let remaining_after_8gallon := remaining_gallons % capacity_8gallon
    let num_2gallon := if remaining_after_8gallon = 3 then 1 else 0
    let num_5gallon := if remaining_after_8gallon = 3 then 1 else remaining_after_8gallon / capacity_5gallon
    let total_trips := num_8gallon + num_2gallon + num_5gallon
    if total_trips <= time_limit / time_per_trip then
      total_trips
    else
      sorry -- If calculations overflow time limit
  else
    0

theorem min_trips_to_fill_hole : 
  hole_filling_trips 676 823 2 5 8 45 1 = 20 :=
by rfl

end min_trips_to_fill_hole_l813_81392


namespace range_of_a_l813_81391

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x < - (Real.sqrt 3) / 3 ∨ x > (Real.sqrt 3) / 3 →
    a * (3 * x^2 - 1) > 0) →
  a > 0 :=
by
  sorry

end range_of_a_l813_81391


namespace ram_shyam_weight_ratio_l813_81378

theorem ram_shyam_weight_ratio
    (R S : ℝ)
    (h1 : 1.10 * R + 1.22 * S = 82.8)
    (h2 : R + S = 72) :
    R / S = 7 / 5 :=
by sorry

end ram_shyam_weight_ratio_l813_81378


namespace abs_eq_abs_of_unique_solution_l813_81329

variables {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0)
theorem abs_eq_abs_of_unique_solution
  (h : ∃ x : ℝ, ∀ y : ℝ, a * (y - a)^2 + b * (y - b)^2 = 0 ↔ y = x) :
  |a| = |b| :=
sorry

end abs_eq_abs_of_unique_solution_l813_81329


namespace part_one_union_sets_l813_81336

theorem part_one_union_sets (a : ℝ) (A B : Set ℝ) :
  (a = 2) →
  A = {x | x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0} →
  B = {x | -2 < x ∧ x < 2} →
  A ∪ B = {x | -2 < x ∧ x ≤ 3} :=
by
  sorry

end part_one_union_sets_l813_81336


namespace arithmetic_sequence_problem_l813_81333

theorem arithmetic_sequence_problem
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (a1 : a 1 = 1)
  (a3 : a 3 = 5)
  (Sn : ∀ n, S n = n * (2 + (n - 1) * 2) / 2)
  (S_diff : ∀ k, S (k + 2) - S k = 36)
  : ∃ k : ℕ, k = 8 :=
by
  sorry

end arithmetic_sequence_problem_l813_81333


namespace monomials_like_terms_l813_81382

theorem monomials_like_terms (a b : ℝ) (m n : ℤ) 
  (h1 : 2 * (a^4) * (b^(-2 * m + 7)) = 3 * (a^(2 * m)) * (b^(n + 2))) :
  m + n = 3 := 
by {
  -- Our proof will be placed here
  sorry
}

end monomials_like_terms_l813_81382


namespace problem1_problem2_l813_81317

-- define problem 1 as a theorem
theorem problem1: 
  ((-0.4) * (-0.8) * (-1.25) * 2.5 = -1) :=
  sorry

-- define problem 2 as a theorem
theorem problem2: 
  ((- (5:ℚ) / 8) * (3 / 14) * ((-16) / 5) * ((-7) / 6) = -1 / 2) :=
  sorry

end problem1_problem2_l813_81317


namespace return_kittens_due_to_rehoming_problems_l813_81323

def num_breeding_rabbits : Nat := 10
def kittens_first_spring : Nat := num_breeding_rabbits * num_breeding_rabbits
def kittens_adopted_first_spring : Nat := kittens_first_spring / 2
def kittens_second_spring : Nat := 60
def kittens_adopted_second_spring : Nat := 4
def total_rabbits : Nat := 121

def non_breeding_rabbits_from_first_spring : Nat :=
  total_rabbits - num_breeding_rabbits - kittens_second_spring

def kittens_returned_to_lola : Prop :=
  non_breeding_rabbits_from_first_spring - kittens_adopted_first_spring = 1

theorem return_kittens_due_to_rehoming_problems : kittens_returned_to_lola :=
sorry

end return_kittens_due_to_rehoming_problems_l813_81323


namespace prove_pqrstu_eq_416_l813_81365

-- Define the condition 1728 * x^4 + 64 = (p * x^3 + q * x^2 + r * x + s) * (t * x + u) + v
def condition (p q r s t u v : ℤ) (x : ℤ) : Prop :=
  1728 * x^4 + 64 = (p * x^3 + q * x^2 + r * x + s) * (t * x + u) + v

-- State the theorem to prove p^2 + q^2 + r^2 + s^2 + t^2 + u^2 + v^2 = 416
theorem prove_pqrstu_eq_416 (p q r s t u v : ℤ) (h : ∀ x, condition p q r s t u v x) : 
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 + v^2 = 416 :=
sorry

end prove_pqrstu_eq_416_l813_81365


namespace smallest_rectangle_area_l813_81343

-- Definitions based on conditions
def diameter : ℝ := 10
def length : ℝ := diameter
def width : ℝ := diameter + 2

-- Theorem statement
theorem smallest_rectangle_area : (length * width) = 120 :=
by
  -- The proof would go here, but we provide sorry for now
  sorry

end smallest_rectangle_area_l813_81343


namespace unique_a_values_l813_81379

theorem unique_a_values :
  ∃ a_values : Finset ℝ,
    (∀ a ∈ a_values, ∃ r s : ℤ, (r + s = -a) ∧ (r * s = 8 * a)) ∧ a_values.card = 4 :=
by
  sorry

end unique_a_values_l813_81379


namespace largest_common_multiple_of_7_8_l813_81380

noncomputable def largest_common_multiple_of_7_8_sub_2 (n : ℕ) : ℕ :=
  if n <= 100 then n else 0

theorem largest_common_multiple_of_7_8 :
  ∃ x : ℕ, x <= 100 ∧ (x - 2) % Nat.lcm 7 8 = 0 ∧ x = 58 :=
by
  let x := 58
  use x
  have h1 : x <= 100 := by norm_num
  have h2 : (x - 2) % Nat.lcm 7 8 = 0 := by norm_num
  have h3 : x = 58 := by norm_num
  exact ⟨h1, h2, h3⟩

end largest_common_multiple_of_7_8_l813_81380


namespace least_positive_integer_l813_81313

theorem least_positive_integer (n : ℕ) (h1 : n > 1) 
  (h2 : n % 2 = 1) (h3 : n % 3 = 1) (h4 : n % 5 = 1) 
  (h5 : n % 7 = 1) (h6 : n % 11 = 1): 
  n = 2311 := 
by
  sorry

end least_positive_integer_l813_81313


namespace ramu_profit_percent_l813_81308

noncomputable def profit_percent (purchase_price repair_cost selling_price : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost
  let profit := selling_price - total_cost
  (profit * 100) / total_cost

theorem ramu_profit_percent :
  profit_percent 42000 13000 64900 = 18 := by
  sorry

end ramu_profit_percent_l813_81308


namespace tan_beta_minus_2alpha_l813_81315

noncomputable def tan_alpha := 1 / 2
noncomputable def tan_beta_minus_alpha := 2 / 5
theorem tan_beta_minus_2alpha (α β : ℝ) (h1 : Real.tan α = tan_alpha) (h2 : Real.tan (β - α) = tan_beta_minus_alpha) :
  Real.tan (β - 2 * α) = -1 / 12 := 
by
  sorry

end tan_beta_minus_2alpha_l813_81315
