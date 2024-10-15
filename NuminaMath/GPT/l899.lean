import Mathlib

namespace NUMINAMATH_GPT_total_distance_l899_89965

/--
John's journey is from point (-3, 4) to (2, 2) to (6, -3).
Prove that the total distance John travels is the sum of distances
from (-3, 4) to (2, 2) and from (2, 2) to (6, -3).
-/
theorem total_distance : 
  let d1 := Real.sqrt ((-3 - 2)^2 + (4 - 2)^2)
  let d2 := Real.sqrt ((6 - 2)^2 + (-3 - 2)^2)
  d1 + d2 = Real.sqrt 29 + Real.sqrt 41 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_l899_89965


namespace NUMINAMATH_GPT_max_value_of_expr_l899_89948

theorem max_value_of_expr 
  (x y z : ℝ) 
  (h₀ : 0 < x) 
  (h₁ : 0 < y) 
  (h₂ : 0 < z)
  (h : x^2 + y^2 + z^2 = 1) : 
  3 * x * y + y * z ≤ (Real.sqrt 10) / 2 := 
  sorry

end NUMINAMATH_GPT_max_value_of_expr_l899_89948


namespace NUMINAMATH_GPT_product_mod_five_l899_89960

theorem product_mod_five (a b c : ℕ) (h₁ : a = 1236) (h₂ : b = 7483) (h₃ : c = 53) :
  (a * b * c) % 5 = 4 :=
by
  sorry

end NUMINAMATH_GPT_product_mod_five_l899_89960


namespace NUMINAMATH_GPT_hotel_charge_difference_l899_89950

variables (G P R : ℝ)

-- Assumptions based on the problem conditions
variables
  (hR : R = 2 * G) -- Charge for a single room at hotel R is 100% greater than at hotel G
  (hP : P = 0.9 * G) -- Charge for a single room at hotel P is 10% less than at hotel G

theorem hotel_charge_difference :
  ((R - P) / R) * 100 = 55 :=
by
  -- Calculation
  sorry

end NUMINAMATH_GPT_hotel_charge_difference_l899_89950


namespace NUMINAMATH_GPT_total_handshakes_eq_900_l899_89976

def num_boys : ℕ := 25
def handshakes_per_pair : ℕ := 3

theorem total_handshakes_eq_900 : (num_boys * (num_boys - 1) / 2) * handshakes_per_pair = 900 := by
  sorry

end NUMINAMATH_GPT_total_handshakes_eq_900_l899_89976


namespace NUMINAMATH_GPT_quadratic_has_real_roots_find_specific_k_l899_89995

-- Part 1: Prove the range of values for k
theorem quadratic_has_real_roots (k : ℝ) : (k ≥ 2) ↔ ∃ x1 x2 : ℝ, x1 ^ 2 - 4 * x1 - 2 * k + 8 = 0 ∧ x2 ^ 2 - 4 * x2 - 2 * k + 8 = 0 := 
sorry

-- Part 2: Prove the specific value of k given the additional condition
theorem find_specific_k (k : ℝ) (x1 x2 : ℝ) : (x1 ^ 3 * x2 + x1 * x2 ^ 3 = 24) ∧ x1 ^ 2 - 4 * x1 - 2 * k + 8 = 0 ∧ x2 ^ 2 - 4 * x2 - 2 * k + 8 = 0 → k = 3 :=
sorry

end NUMINAMATH_GPT_quadratic_has_real_roots_find_specific_k_l899_89995


namespace NUMINAMATH_GPT_fraction_of_selected_color_films_equals_five_twenty_sixths_l899_89991

noncomputable def fraction_of_selected_color_films (x y : ℕ) : ℚ :=
  let bw_films := 40 * x
  let color_films := 10 * y
  let selected_bw_films := (y / x * 1 / 100) * bw_films
  let selected_color_films := color_films
  let total_selected_films := selected_bw_films + selected_color_films
  selected_color_films / total_selected_films

theorem fraction_of_selected_color_films_equals_five_twenty_sixths (x y : ℕ) (h1 : x > 0) (h2 : y > 0) :
  fraction_of_selected_color_films x y = 5 / 26 := by
  sorry

end NUMINAMATH_GPT_fraction_of_selected_color_films_equals_five_twenty_sixths_l899_89991


namespace NUMINAMATH_GPT_circle_represents_range_l899_89994

theorem circle_represents_range (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + m * x - 2 * y + 3 = 0 → (m > 2 * Real.sqrt 2 ∨ m < -2 * Real.sqrt 2)) :=
by
  sorry

end NUMINAMATH_GPT_circle_represents_range_l899_89994


namespace NUMINAMATH_GPT_range_of_a_l899_89997

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(1 + a * x) - x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f a (f a x) - x

theorem range_of_a (a : ℝ) : (F a 0 = 0 → F a e = 0) → 
  (0 < a ∧ a < (1 / (Real.exp 1 * Real.log 2))) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l899_89997


namespace NUMINAMATH_GPT_racing_magic_circle_time_l899_89936

theorem racing_magic_circle_time
  (T : ℕ) -- Time taken by the racing magic to circle the track once
  (bull_rounds_per_hour : ℕ := 40) -- Rounds the Charging Bull makes in an hour
  (meet_time_minutes : ℕ := 6) -- Time in minutes to meet at starting point
  (charging_bull_seconds_per_round : ℕ := 3600 / bull_rounds_per_hour) -- Time in seconds per Charging Bull round
  (meet_time_seconds : ℕ := meet_time_minutes * 60) -- Time in seconds to meet at starting point
  (rounds_by_bull : ℕ := meet_time_seconds / charging_bull_seconds_per_round) -- Rounds completed by the Charging Bull to meet again
  (rounds_by_magic : ℕ := meet_time_seconds / T) -- Rounds completed by the Racing Magic to meet again
  (h1 : rounds_by_magic = 1) -- Racing Magic completes 1 round in the meet time
  : T = 360 := -- Racing Magic takes 360 seconds to circle the track once
  sorry

end NUMINAMATH_GPT_racing_magic_circle_time_l899_89936


namespace NUMINAMATH_GPT_amount_to_add_l899_89930

theorem amount_to_add (students : ℕ) (total_cost : ℕ) (h1 : students = 9) (h2 : total_cost = 143) : 
  ∃ k : ℕ, total_cost + k = students * (total_cost / students + 1) :=
by
  sorry

end NUMINAMATH_GPT_amount_to_add_l899_89930


namespace NUMINAMATH_GPT_luncheon_cost_l899_89959

theorem luncheon_cost (s c p : ℝ)
  (h1 : 2 * s + 5 * c + p = 3.00)
  (h2 : 5 * s + 8 * c + p = 5.40)
  (h3 : 3 * s + 4 * c + p = 3.60) :
  2 * s + 2 * c + p = 2.60 :=
sorry

end NUMINAMATH_GPT_luncheon_cost_l899_89959


namespace NUMINAMATH_GPT_angle_between_tangents_l899_89955

theorem angle_between_tangents (R1 R2 : ℝ) (k : ℝ) (h_ratio : R1 = 2 * k ∧ R2 = 3 * k)
  (h_touching : (∃ O1 O2 : ℝ, (R2 - R1 = k))) : 
  ∃ θ : ℝ, θ = 90 := sorry

end NUMINAMATH_GPT_angle_between_tangents_l899_89955


namespace NUMINAMATH_GPT_room_width_is_12_l899_89990

variable (w : ℕ)

-- Definitions of given conditions
def room_length := 19
def veranda_width := 2
def veranda_area := 140

-- Statement that needs to be proven
theorem room_width_is_12
  (h1 : veranda_width = 2)
  (h2 : veranda_area = 140)
  (h3 : room_length = 19) :
  w = 12 :=
by
  sorry

end NUMINAMATH_GPT_room_width_is_12_l899_89990


namespace NUMINAMATH_GPT_gcd_m_n_is_one_l899_89938

/-- Definition of m -/
def m : ℕ := 130^2 + 241^2 + 352^2

/-- Definition of n -/
def n : ℕ := 129^2 + 240^2 + 353^2 + 2^3

/-- Proof statement: The greatest common divisor of m and n is 1 -/
theorem gcd_m_n_is_one : Nat.gcd m n = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_m_n_is_one_l899_89938


namespace NUMINAMATH_GPT_ganesh_ram_sohan_work_time_l899_89946

theorem ganesh_ram_sohan_work_time (G R S : ℝ)
  (H1 : G + R = 1 / 24)
  (H2 : S = 1 / 48) : (G + R + S = 1 / 16) ∧ (1 / (G + R + S) = 16) :=
by
  sorry

end NUMINAMATH_GPT_ganesh_ram_sohan_work_time_l899_89946


namespace NUMINAMATH_GPT_dancer_count_l899_89908

theorem dancer_count (n : ℕ) : 
  ((n + 5) % 12 = 0) ∧ ((n + 5) % 10 = 0) ∧ (200 ≤ n) ∧ (n ≤ 300) → (n = 235 ∨ n = 295) := 
by
  sorry

end NUMINAMATH_GPT_dancer_count_l899_89908


namespace NUMINAMATH_GPT_tailor_cut_difference_l899_89925

def skirt_cut : ℝ := 0.75
def pants_cut : ℝ := 0.5

theorem tailor_cut_difference : skirt_cut - pants_cut = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_tailor_cut_difference_l899_89925


namespace NUMINAMATH_GPT_compute_expression_l899_89903

noncomputable def given_cubic (x : ℝ) : Prop :=
  x ^ 3 - 7 * x ^ 2 + 12 * x = 18

theorem compute_expression (a b c : ℝ) (ha : given_cubic a) (hb : given_cubic b) (hc : given_cubic c) :
  (a + b + c = 7) → 
  (a * b + b * c + c * a = 12) → 
  (a * b * c = 18) → 
  (a * b / c + b * c / a + c * a / b = -6) :=
by 
  sorry

end NUMINAMATH_GPT_compute_expression_l899_89903


namespace NUMINAMATH_GPT_road_length_in_km_l899_89953

/-- The actual length of the road in kilometers is 7.5, given the scale of 1:50000 
    and the map length of 15 cm. -/

theorem road_length_in_km (s : ℕ) (map_length_cm : ℕ) (actual_length_cm : ℕ) (actual_length_km : ℝ) 
  (h_scale : s = 50000) (h_map_length : map_length_cm = 15) (h_conversion : actual_length_km = actual_length_cm / 100000) :
  actual_length_km = 7.5 :=
  sorry

end NUMINAMATH_GPT_road_length_in_km_l899_89953


namespace NUMINAMATH_GPT_base_10_to_base_7_conversion_l899_89902

theorem base_10_to_base_7_conversion :
  ∃ (digits : ℕ → ℕ), 789 = digits 3 * 7^3 + digits 2 * 7^2 + digits 1 * 7^1 + digits 0 * 7^0 ∧
  digits 3 = 2 ∧ digits 2 = 2 ∧ digits 1 = 0 ∧ digits 0 = 5 :=
sorry

end NUMINAMATH_GPT_base_10_to_base_7_conversion_l899_89902


namespace NUMINAMATH_GPT_log_difference_example_l899_89924

theorem log_difference_example :
  ∀ (log : ℕ → ℝ),
    log 3 * 24 - log 3 * 8 = 1 := 
by
sorry

end NUMINAMATH_GPT_log_difference_example_l899_89924


namespace NUMINAMATH_GPT_non_parallel_lines_a_l899_89921

theorem non_parallel_lines_a (a : ℝ) :
  ¬ (a * -(1 / (a+2))) = a →
  ¬ (-1 / (a+2)) = 2 →
  a = 0 ∨ a = -3 :=
by
  sorry

end NUMINAMATH_GPT_non_parallel_lines_a_l899_89921


namespace NUMINAMATH_GPT_train_length_l899_89975

noncomputable def L_train : ℝ :=
  let speed_kmph : ℝ := 60
  let speed_mps : ℝ := (speed_kmph * 1000 / 3600)
  let time : ℝ := 30
  let length_bridge : ℝ := 140
  let total_distance : ℝ := speed_mps * time
  total_distance - length_bridge

theorem train_length : L_train = 360.1 :=
by
  -- Sorry statement to skip the proof
  sorry

end NUMINAMATH_GPT_train_length_l899_89975


namespace NUMINAMATH_GPT_monomial_same_type_m_n_sum_l899_89963

theorem monomial_same_type_m_n_sum (m n : ℕ) (x y : ℤ) 
  (h1 : 2 * x ^ (m - 1) * y ^ 2 = 1/3 * x ^ 2 * y ^ (n + 1)) : 
  m + n = 4 := 
sorry

end NUMINAMATH_GPT_monomial_same_type_m_n_sum_l899_89963


namespace NUMINAMATH_GPT_area_of_shaded_region_l899_89998

open Real

noncomputable def line1 (x : ℝ) : ℝ := -3/10 * x + 5
noncomputable def line2 (x : ℝ) : ℝ := -1.5 * x + 9

theorem area_of_shaded_region : 
  ∫ x in (2:ℝ)..6, (line2 x - line1 x) = 8 :=
by
  sorry

end NUMINAMATH_GPT_area_of_shaded_region_l899_89998


namespace NUMINAMATH_GPT_min_faces_n2_min_faces_n3_l899_89962

noncomputable def minimum_faces (n : ℕ) : ℕ := 
  if n = 2 then 2 
  else if n = 3 then 12 
  else sorry 

theorem min_faces_n2 : minimum_faces 2 = 2 := 
  by 
  simp [minimum_faces]

theorem min_faces_n3 : minimum_faces 3 = 12 := 
  by 
  simp [minimum_faces]

end NUMINAMATH_GPT_min_faces_n2_min_faces_n3_l899_89962


namespace NUMINAMATH_GPT_candy_necklaces_per_pack_l899_89916

theorem candy_necklaces_per_pack (packs_total packs_opened packs_left candies_left necklaces_per_pack : ℕ) 
  (h_total : packs_total = 9) 
  (h_opened : packs_opened = 4) 
  (h_left : packs_left = packs_total - packs_opened) 
  (h_candies_left : candies_left = 40) 
  (h_necklaces_per_pack : candies_left = packs_left * necklaces_per_pack) :
  necklaces_per_pack = 8 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_candy_necklaces_per_pack_l899_89916


namespace NUMINAMATH_GPT_amy_remaining_money_l899_89947

-- Definitions based on conditions
def initial_money : ℕ := 100
def doll_cost : ℕ := 1
def number_of_dolls : ℕ := 3

-- The theorem we want to prove
theorem amy_remaining_money : initial_money - number_of_dolls * doll_cost = 97 :=
by 
  sorry

end NUMINAMATH_GPT_amy_remaining_money_l899_89947


namespace NUMINAMATH_GPT_perpendicular_bisector_eq_l899_89928

theorem perpendicular_bisector_eq (x y : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 2 * x - 5 = 0 ∧ x^2 + y^2 + 2 * x - 4 * y - 4 = 0) →
  x + y - 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_bisector_eq_l899_89928


namespace NUMINAMATH_GPT_determine_numbers_l899_89945

theorem determine_numbers (a b : ℕ) (S : ℕ) (h1 : a ≠ b) (h2 : a > 11) (h3 : b > 11)
  (h4 : S = a + b) (h5 : (∀ (x y : ℕ), x + y = S → x ≠ y → (x = a ∧ y = b) ∨ (x = b ∧ y = a)) = false)
  (h6 : a % 2 = 0 ∨ b % 2 = 0) : 
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := 
  sorry

end NUMINAMATH_GPT_determine_numbers_l899_89945


namespace NUMINAMATH_GPT_parking_spaces_remaining_l899_89989

-- Define the conditions as variables
variable (total_spaces : Nat := 30)
variable (spaces_per_caravan : Nat := 2)
variable (num_caravans : Nat := 3)

-- Prove the number of vehicles that can still park equals 24
theorem parking_spaces_remaining (total_spaces spaces_per_caravan num_caravans : Nat) :
    total_spaces - spaces_per_caravan * num_caravans = 24 :=
by
  -- Filling in the proof is required to fully complete this, but as per instruction we add 'sorry'
  sorry

end NUMINAMATH_GPT_parking_spaces_remaining_l899_89989


namespace NUMINAMATH_GPT_celebrity_baby_photo_probability_l899_89941

theorem celebrity_baby_photo_probability : 
  let total_arrangements := Nat.factorial 4
  let correct_arrangements := 1
  let probability := correct_arrangements / total_arrangements
  probability = 1/24 :=
by
  sorry

end NUMINAMATH_GPT_celebrity_baby_photo_probability_l899_89941


namespace NUMINAMATH_GPT_calc_fraction_l899_89934

theorem calc_fraction : (36 + 12) / (6 - 3) = 16 :=
by
  sorry

end NUMINAMATH_GPT_calc_fraction_l899_89934


namespace NUMINAMATH_GPT_find_distance_from_home_to_airport_l899_89984

variable (d t : ℝ)

-- Conditions
def condition1 := d = 40 * (t + 0.75)
def condition2 := d - 40 = 60 * (t - 1.25)

-- Proof statement
theorem find_distance_from_home_to_airport (hd : condition1 d t) (ht : condition2 d t) : d = 160 :=
by
  sorry

end NUMINAMATH_GPT_find_distance_from_home_to_airport_l899_89984


namespace NUMINAMATH_GPT_newton_method_approximation_bisection_method_approximation_l899_89992

noncomputable def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 3
noncomputable def f' (x : ℝ) : ℝ := 3*x^2 + 4*x + 3

theorem newton_method_approximation :
  let x0 := -1
  let x1 := x0 - f x0 / f' x0
  let x2 := x1 - f x1 / f' x1
  x2 = -7 / 5 := sorry

theorem bisection_method_approximation :
  let a := -2
  let b := -1
  let midpoint1 := (a + b) / 2
  let new_a := if f midpoint1 < 0 then midpoint1 else a
  let new_b := if f midpoint1 < 0 then b else midpoint1
  let midpoint2 := (new_a + new_b) / 2
  midpoint2 = -11 / 8 := sorry

end NUMINAMATH_GPT_newton_method_approximation_bisection_method_approximation_l899_89992


namespace NUMINAMATH_GPT_sum_of_coefficients_l899_89920

theorem sum_of_coefficients (a_5 a_4 a_3 a_2 a_1 a_0 : ℤ) :
  (x-2)^5 = a_5*x^5 + a_4*x^4 + a_3*x^3 + a_2*x^2 + a_1*x + a_0 →
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l899_89920


namespace NUMINAMATH_GPT_chips_sales_l899_89978

theorem chips_sales (total_chips : ℕ) (first_week : ℕ) (second_week : ℕ) (third_week : ℕ) (fourth_week : ℕ)
  (h1 : total_chips = 100)
  (h2 : first_week = 15)
  (h3 : second_week = 3 * first_week)
  (h4 : third_week = fourth_week)
  (h5 : total_chips = first_week + second_week + third_week + fourth_week) : third_week = 20 :=
by
  sorry

end NUMINAMATH_GPT_chips_sales_l899_89978


namespace NUMINAMATH_GPT_Jiyeol_average_score_l899_89964

theorem Jiyeol_average_score (K M E : ℝ)
  (h1 : (K + M) / 2 = 26.5)
  (h2 : (M + E) / 2 = 34.5)
  (h3 : (K + E) / 2 = 29) :
  (K + M + E) / 3 = 30 := 
sorry

end NUMINAMATH_GPT_Jiyeol_average_score_l899_89964


namespace NUMINAMATH_GPT_jake_present_weight_l899_89996

theorem jake_present_weight (J S : ℕ) 
  (h1 : J - 32 = 2 * S) 
  (h2 : J + S = 212) : 
  J = 152 := 
by 
  sorry

end NUMINAMATH_GPT_jake_present_weight_l899_89996


namespace NUMINAMATH_GPT_c_share_l899_89929

theorem c_share (a b c : ℝ) (h1 : a = b / 2) (h2 : b = c / 2) (h3 : a + b + c = 700) : c = 400 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_c_share_l899_89929


namespace NUMINAMATH_GPT_train_speed_is_64_kmh_l899_89973

noncomputable def train_speed_kmh (train_length platform_length time_seconds : ℕ) : ℕ :=
  let total_distance := train_length + platform_length
  let speed_mps := total_distance / time_seconds
  let speed_kmh := speed_mps * 36 / 10
  speed_kmh

theorem train_speed_is_64_kmh
  (train_length : ℕ)
  (platform_length : ℕ)
  (time_seconds : ℕ)
  (h_train_length : train_length = 240)
  (h_platform_length : platform_length = 240)
  (h_time_seconds : time_seconds = 27) :
  train_speed_kmh train_length platform_length time_seconds = 64 := by
  sorry

end NUMINAMATH_GPT_train_speed_is_64_kmh_l899_89973


namespace NUMINAMATH_GPT_samantha_birth_year_l899_89910

theorem samantha_birth_year
  (first_amc8_year : ℕ := 1985)
  (held_annually : ∀ (n : ℕ), n ≥ 0 → first_amc8_year + n = 1985 + n)
  (samantha_age_7th_amc8 : ℕ := 12) :
  ∃ (birth_year : ℤ), birth_year = 1979 :=
by
  sorry

end NUMINAMATH_GPT_samantha_birth_year_l899_89910


namespace NUMINAMATH_GPT_parabola_sum_l899_89986

def original_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

def reflected_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x ^ 2 - b * x + c

def f (a b c : ℝ) (x : ℝ) : ℝ := a * (x + 7) ^ 2 - b * (x + 7) + c

def g (a b c : ℝ) (x : ℝ) : ℝ := a * (x - 3) ^ 2 - b * (x - 3) + c

def fg (a b c : ℝ) (x : ℝ) : ℝ := f a b c x + g a b c x

theorem parabola_sum (a b c x : ℝ) : fg a b c x = 2 * a * x ^ 2 + (8 * a - 2 * b) * x + (58 * a - 4 * b + 2 * c) := by
  sorry

end NUMINAMATH_GPT_parabola_sum_l899_89986


namespace NUMINAMATH_GPT_probability_four_squares_form_square_l899_89970

noncomputable def probability_form_square (n k : ℕ) :=
  if (k = 4) ∧ (n = 6) then (1 / 561 : ℚ) else 0

theorem probability_four_squares_form_square :
  probability_form_square 6 4 = (1 / 561 : ℚ) :=
by
  -- Here we would usually include the detailed proof
  -- corresponding to the solution steps from the problem,
  -- but we leave it as sorry for now.
  sorry

end NUMINAMATH_GPT_probability_four_squares_form_square_l899_89970


namespace NUMINAMATH_GPT_minimum_value_ineq_l899_89939

noncomputable def minimum_value (x y z : ℝ) := x^2 + 4 * x * y + 4 * y^2 + 4 * z^2

theorem minimum_value_ineq (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 64) : minimum_value x y z ≥ 192 :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_value_ineq_l899_89939


namespace NUMINAMATH_GPT_tangent_line_to_circle_l899_89919

noncomputable def r_tangent_to_circle : ℝ := 4

theorem tangent_line_to_circle
  (x y r : ℝ)
  (circle_eq : x^2 + y^2 = 2 * r)
  (line_eq : x - y = r) :
  r = r_tangent_to_circle :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_to_circle_l899_89919


namespace NUMINAMATH_GPT_lcm_12_18_l899_89935

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := 
by
  sorry

end NUMINAMATH_GPT_lcm_12_18_l899_89935


namespace NUMINAMATH_GPT_multiple_of_second_lock_time_l899_89931

def first_lock_time := 5
def second_lock_time := 3 * first_lock_time - 3
def combined_lock_time := 60

theorem multiple_of_second_lock_time : combined_lock_time / second_lock_time = 5 := by
  -- Adding a proof placeholder using sorry
  sorry

end NUMINAMATH_GPT_multiple_of_second_lock_time_l899_89931


namespace NUMINAMATH_GPT_star_4_3_l899_89905

def star (a b : ℕ) : ℕ := a^2 + a * b - b^3

theorem star_4_3 : star 4 3 = 1 := 
by
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_star_4_3_l899_89905


namespace NUMINAMATH_GPT_isosceles_triangle_time_between_9_30_and_10_l899_89993

theorem isosceles_triangle_time_between_9_30_and_10 (time : ℕ) (h_time_range : 30 ≤ time ∧ time < 60)
  (h_isosceles : ∃ x : ℝ, 0 ≤ x ∧ x + 2 * x + 2 * x = 180) :
  time = 36 :=
  sorry

end NUMINAMATH_GPT_isosceles_triangle_time_between_9_30_and_10_l899_89993


namespace NUMINAMATH_GPT_num_pairs_satisfying_equation_l899_89914

theorem num_pairs_satisfying_equation :
  ∃! (x y : ℕ), 0 < x ∧ 0 < y ∧ x^2 - y^2 = 204 :=
by
  sorry

end NUMINAMATH_GPT_num_pairs_satisfying_equation_l899_89914


namespace NUMINAMATH_GPT_fulfill_customer_order_in_nights_l899_89909

structure JerkyCompany where
  batch_size : ℕ
  nightly_batches : ℕ

def customerOrder (ordered : ℕ) (current_stock : ℕ) : ℕ :=
  ordered - current_stock

def batchesNeeded (required : ℕ) (batch_size : ℕ) : ℕ :=
  required / batch_size

def daysNeeded (batches_needed : ℕ) (nightly_batches : ℕ) : ℕ :=
  batches_needed / nightly_batches

theorem fulfill_customer_order_in_nights :
  ∀ (ordered current_stock : ℕ) (jc : JerkyCompany),
    jc.batch_size = 10 →
    jc.nightly_batches = 1 →
    ordered = 60 →
    current_stock = 20 →
    daysNeeded (batchesNeeded (customerOrder ordered current_stock) jc.batch_size) jc.nightly_batches = 4 :=
by
  intros ordered current_stock jc h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_fulfill_customer_order_in_nights_l899_89909


namespace NUMINAMATH_GPT_equation_has_solution_implies_a_ge_2_l899_89974

theorem equation_has_solution_implies_a_ge_2 (a : ℝ) :
  (∃ x : ℝ, 4^x - a * 2^x - a + 3 = 0) → a ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_equation_has_solution_implies_a_ge_2_l899_89974


namespace NUMINAMATH_GPT_money_has_48_l899_89988

-- Definitions derived from conditions:
def money (p : ℝ) := 
  p = (1/3 * p) + 32

-- The main theorem statement
theorem money_has_48 (p : ℝ) : money p → p = 48 := by
  intro h
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_money_has_48_l899_89988


namespace NUMINAMATH_GPT_stamps_problem_l899_89972

theorem stamps_problem (x y : ℕ) : 
  2 * x + 6 * x + 5 * y / 2 = 60 → x = 5 ∧ y = 8 ∧ 6 * x = 30 :=
by 
  sorry

end NUMINAMATH_GPT_stamps_problem_l899_89972


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l899_89904

def A : Set ℝ := { x | 0 < x }
def B : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | 0 < x ∧ x ≤ 1 } := 
sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l899_89904


namespace NUMINAMATH_GPT_cube_axes_of_symmetry_regular_tetrahedron_axes_of_symmetry_l899_89900

-- Definitions for geometric objects
def cube : Type := sorry
def regular_tetrahedron : Type := sorry

-- Definitions for axes of symmetry
def axes_of_symmetry (shape : Type) : Nat := sorry

-- Theorem statements
theorem cube_axes_of_symmetry : axes_of_symmetry cube = 13 := 
by 
  sorry

theorem regular_tetrahedron_axes_of_symmetry : axes_of_symmetry regular_tetrahedron = 7 :=
by 
  sorry

end NUMINAMATH_GPT_cube_axes_of_symmetry_regular_tetrahedron_axes_of_symmetry_l899_89900


namespace NUMINAMATH_GPT_visual_range_increase_l899_89911

def percent_increase (original new : ℕ) : ℕ :=
  ((new - original) * 100) / original

theorem visual_range_increase :
  percent_increase 50 150 = 200 := 
by
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_visual_range_increase_l899_89911


namespace NUMINAMATH_GPT_constructible_triangle_l899_89913

theorem constructible_triangle (k c delta : ℝ) (h1 : 2 * c < k) :
  ∃ (a b : ℝ), a + b + c = k ∧ a + b > c ∧ ∃ (α β : ℝ), α - β = delta :=
by
  sorry

end NUMINAMATH_GPT_constructible_triangle_l899_89913


namespace NUMINAMATH_GPT_first_worker_time_budget_l899_89933

theorem first_worker_time_budget
  (total_time : ℝ := 1)
  (second_worker_time : ℝ := 1 / 3)
  (third_worker_time : ℝ := 1 / 3)
  (x : ℝ) :
  x + second_worker_time + third_worker_time = total_time → x = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_first_worker_time_budget_l899_89933


namespace NUMINAMATH_GPT_expected_number_of_digits_l899_89952

noncomputable def expectedNumberDigits : ℝ :=
  let oneDigitProbability := (9 : ℝ) / 16
  let twoDigitProbability := (7 : ℝ) / 16
  (oneDigitProbability * 1) + (twoDigitProbability * 2)

theorem expected_number_of_digits :
  expectedNumberDigits = 1.4375 := by
  sorry

end NUMINAMATH_GPT_expected_number_of_digits_l899_89952


namespace NUMINAMATH_GPT_max_knights_among_10_l899_89977

def is_knight (p : ℕ → Prop) (n : ℕ) : Prop :=
  ∀ m : ℕ, (p m ↔ (m ≥ n))

def is_liar (p : ℕ → Prop) (n : ℕ) : Prop :=
  ∀ m : ℕ, (¬ p m ↔ (m ≥ n))

def greater_than (k : ℕ) (n : ℕ) := n > k

def less_than (k : ℕ) (n : ℕ) := n < k

def person_statement_1 (i : ℕ) (n : ℕ) : Prop :=
  match i with
  | 1 => greater_than 1 n
  | 2 => greater_than 2 n
  | 3 => greater_than 3 n
  | 4 => greater_than 4 n
  | 5 => greater_than 5 n
  | 6 => greater_than 6 n
  | 7 => greater_than 7 n
  | 8 => greater_than 8 n
  | 9 => greater_than 9 n
  | 10 => greater_than 10 n
  | _ => false

def person_statement_2 (i : ℕ) (n : ℕ) : Prop :=
  match i with
  | 1 => less_than 1 n
  | 2 => less_than 2 n
  | 3 => less_than 3 n
  | 4 => less_than 4 n
  | 5 => less_than 5 n
  | 6 => less_than 6 n
  | 7 => less_than 7 n
  | 8 => less_than 8 n
  | 9 => less_than 9 n
  | 10 => less_than 10 n
  | _ => false

theorem max_knights_among_10 (knights : ℕ) : 
  (∀ i < 10, (is_knight (person_statement_1 (i + 1)) (i + 1) ∨ is_liar (person_statement_1 (i + 1)) (i + 1))) ∧
  (∀ i < 10, (is_knight (person_statement_2 (i + 1)) (i + 1) ∨ is_liar (person_statement_2 (i + 1)) (i + 1))) →
  knights ≤ 8 := sorry

end NUMINAMATH_GPT_max_knights_among_10_l899_89977


namespace NUMINAMATH_GPT_h_at_4_l899_89956

noncomputable def f (x : ℝ) := 4 / (3 - x)

noncomputable def f_inv (x : ℝ) := 3 - (4 / x)

noncomputable def h (x : ℝ) := (1 / f_inv x) + 10

theorem h_at_4 : h 4 = 10.5 :=
by
  sorry

end NUMINAMATH_GPT_h_at_4_l899_89956


namespace NUMINAMATH_GPT_candy_difference_l899_89926

theorem candy_difference (Frankie_candies Max_candies : ℕ) (hF : Frankie_candies = 74) (hM : Max_candies = 92) :
  Max_candies - Frankie_candies = 18 :=
by
  sorry

end NUMINAMATH_GPT_candy_difference_l899_89926


namespace NUMINAMATH_GPT_smallest_of_five_consecutive_numbers_l899_89979

theorem smallest_of_five_consecutive_numbers (n : ℕ) : 
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 100) → 
  n = 18 :=
by sorry

end NUMINAMATH_GPT_smallest_of_five_consecutive_numbers_l899_89979


namespace NUMINAMATH_GPT_Jasmine_initial_percentage_is_5_l899_89954

noncomputable def initial_percentage_of_jasmine 
  (V_initial : ℕ := 90) 
  (V_added_jasmine : ℕ := 8) 
  (V_added_water : ℕ := 2) 
  (V_final : ℕ := 100) 
  (P_final : ℚ := 12.5 / 100) : ℚ := 
  (P_final * V_final - V_added_jasmine) / V_initial * 100

theorem Jasmine_initial_percentage_is_5 :
  initial_percentage_of_jasmine = 5 := 
by 
  sorry

end NUMINAMATH_GPT_Jasmine_initial_percentage_is_5_l899_89954


namespace NUMINAMATH_GPT_find_pairs_l899_89999

theorem find_pairs (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  (∃ k m : ℕ, k ≠ 0 ∧ m ≠ 0 ∧ x + 1 = k * y ∧ y + 1 = m * x) ↔
  (x = 2 ∧ y = 1) ∨ (x = 3 ∧ y = 2) ∨ (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_l899_89999


namespace NUMINAMATH_GPT_determine_a_l899_89922

def A := {x : ℝ | x < 6}
def B (a : ℝ) := {x : ℝ | x - a < 0}

theorem determine_a (a : ℝ) (h : A ⊆ B a) : 6 ≤ a := 
sorry

end NUMINAMATH_GPT_determine_a_l899_89922


namespace NUMINAMATH_GPT_seq_a_ge_two_pow_nine_nine_l899_89943

theorem seq_a_ge_two_pow_nine_nine (a : ℕ → ℤ) 
  (h0 : a 1 > a 0)
  (h1 : a 1 > 0)
  (h2 : ∀ r : ℕ, r ≤ 98 → a (r + 2) = 3 * a (r + 1) - 2 * a r) : 
  a 100 > 2^99 :=
sorry

end NUMINAMATH_GPT_seq_a_ge_two_pow_nine_nine_l899_89943


namespace NUMINAMATH_GPT_mrs_hilt_rocks_proof_l899_89940

def num_rocks_already_placed : ℝ := 125.0
def total_num_rocks_planned : ℝ := 189
def num_more_rocks_needed : ℝ := 64

theorem mrs_hilt_rocks_proof : total_num_rocks_planned - num_rocks_already_placed = num_more_rocks_needed :=
by
  sorry

end NUMINAMATH_GPT_mrs_hilt_rocks_proof_l899_89940


namespace NUMINAMATH_GPT_difference_is_167_l899_89944

-- Define the number of boys and girls in each village
def A_village_boys : ℕ := 204
def A_village_girls : ℕ := 468
def B_village_boys : ℕ := 334
def B_village_girls : ℕ := 516
def C_village_boys : ℕ := 427
def C_village_girls : ℕ := 458
def D_village_boys : ℕ := 549
def D_village_girls : ℕ := 239

-- Define total number of boys and girls
def total_boys := A_village_boys + B_village_boys + C_village_boys + D_village_boys
def total_girls := A_village_girls + B_village_girls + C_village_girls + D_village_girls

-- Define the difference between total girls and total boys
def difference := total_girls - total_boys

-- The theorem to prove the difference is 167
theorem difference_is_167 : difference = 167 := by
  sorry

end NUMINAMATH_GPT_difference_is_167_l899_89944


namespace NUMINAMATH_GPT_circle_area_l899_89927

theorem circle_area (r : ℝ) (h : 8 * (1 / (2 * π * r)) = 2 * r) : π * r^2 = 2 := by
  sorry

end NUMINAMATH_GPT_circle_area_l899_89927


namespace NUMINAMATH_GPT_problem_l899_89915

theorem problem (m : ℝ) (h : m + 1/m = 10) : m^3 + 1/m^3 + 6 = 976 :=
by
  sorry

end NUMINAMATH_GPT_problem_l899_89915


namespace NUMINAMATH_GPT_total_spent_is_correct_l899_89932

-- Declare the constants for the prices and quantities
def wallet_cost : ℕ := 50
def sneakers_cost_per_pair : ℕ := 100
def sneakers_pairs : ℕ := 2
def backpack_cost : ℕ := 100
def jeans_cost_per_pair : ℕ := 50
def jeans_pairs : ℕ := 2

-- Define the total amounts spent by Leonard and Michael
def leonard_total : ℕ := wallet_cost + sneakers_cost_per_pair * sneakers_pairs
def michael_total : ℕ := backpack_cost + jeans_cost_per_pair * jeans_pairs

-- The total amount spent by Leonard and Michael
def total_spent : ℕ := leonard_total + michael_total

-- The proof statement
theorem total_spent_is_correct : total_spent = 450 :=
by 
  -- This part is where the proof would go
  sorry

end NUMINAMATH_GPT_total_spent_is_correct_l899_89932


namespace NUMINAMATH_GPT_clare_money_left_l899_89985

noncomputable def cost_of_bread : ℝ := 4 * 2
noncomputable def cost_of_milk : ℝ := 2 * 2
noncomputable def cost_of_cereal : ℝ := 3 * 3
noncomputable def cost_of_apples : ℝ := 1 * 4

noncomputable def total_cost_before_discount : ℝ := cost_of_bread + cost_of_milk + cost_of_cereal + cost_of_apples
noncomputable def discount_amount : ℝ := total_cost_before_discount * 0.1
noncomputable def total_cost_after_discount : ℝ := total_cost_before_discount - discount_amount
noncomputable def sales_tax : ℝ := total_cost_after_discount * 0.05
noncomputable def total_cost_after_discount_and_tax : ℝ := total_cost_after_discount + sales_tax

noncomputable def initial_amount : ℝ := 47
noncomputable def money_left : ℝ := initial_amount - total_cost_after_discount_and_tax

theorem clare_money_left : money_left = 23.37 := by
  sorry

end NUMINAMATH_GPT_clare_money_left_l899_89985


namespace NUMINAMATH_GPT_train_passing_time_l899_89951

theorem train_passing_time :
  ∀ (length : ℕ) (speed_km_hr : ℕ), length = 300 ∧ speed_km_hr = 90 →
  (length / (speed_km_hr * (1000 / 3600)) = 12) := 
by
  intros length speed_km_hr h
  have h_length : length = 300 := h.1
  have h_speed : speed_km_hr = 90 := h.2
  sorry

end NUMINAMATH_GPT_train_passing_time_l899_89951


namespace NUMINAMATH_GPT_train_overtakes_motorbike_time_l899_89917

theorem train_overtakes_motorbike_time :
  let train_speed_kmph := 100
  let motorbike_speed_kmph := 64
  let train_length_m := 120.0096
  let relative_speed_kmph := train_speed_kmph - motorbike_speed_kmph
  let relative_speed_m_s := (relative_speed_kmph : ℝ) * (1 / 3.6)
  let time_seconds := train_length_m / relative_speed_m_s
  time_seconds = 12.00096 :=
sorry

end NUMINAMATH_GPT_train_overtakes_motorbike_time_l899_89917


namespace NUMINAMATH_GPT_determine_cost_price_l899_89981

def selling_price := 16
def loss_fraction := 1 / 6

noncomputable def cost_price (CP : ℝ) : Prop :=
  selling_price = CP - (loss_fraction * CP)

theorem determine_cost_price (CP : ℝ) (h: cost_price CP) : CP = 19.2 := by
  sorry

end NUMINAMATH_GPT_determine_cost_price_l899_89981


namespace NUMINAMATH_GPT_sum_of_digits_l899_89918

variables {a b c d : ℕ}

theorem sum_of_digits (h1 : ∀ (x y z w : ℕ), x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w)
                      (h2 : c + a = 10)
                      (h3 : b + c = 9)
                      (h4 : a + d = 10) :
  a + b + c + d = 18 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_l899_89918


namespace NUMINAMATH_GPT_hyperbola_center_l899_89966

theorem hyperbola_center : 
  (∃ x y : ℝ, (4 * y + 6)^2 / 16 - (5 * x - 3)^2 / 9 = 1) →
  (∃ h k : ℝ, h = 3 / 5 ∧ k = -3 / 2 ∧ 
    (∀ x' y', (4 * y' + 6)^2 / 16 - (5 * x' - 3)^2 / 9 = 1 → x' = h ∧ y' = k)) :=
sorry

end NUMINAMATH_GPT_hyperbola_center_l899_89966


namespace NUMINAMATH_GPT_find_white_towels_l899_89937

variable {W : ℕ} -- Define W as a natural number

-- Define the conditions as Lean definitions
def initial_towel_count (W : ℕ) : ℕ := 35 + W
def remaining_towel_count (W : ℕ) : ℕ := initial_towel_count W - 34

-- Theorem statement: Proving that W = 21 given the conditions
theorem find_white_towels (h : remaining_towel_count W = 22) : W = 21 :=
by
  sorry

end NUMINAMATH_GPT_find_white_towels_l899_89937


namespace NUMINAMATH_GPT_number_of_triangles_l899_89983

theorem number_of_triangles (n : ℕ) : 
  ∃ k : ℕ, k = ⌊((n + 1) * (n + 3) * (2 * n + 1) : ℝ) / 24⌋ := sorry

end NUMINAMATH_GPT_number_of_triangles_l899_89983


namespace NUMINAMATH_GPT_math_problem_l899_89980

theorem math_problem :
  (Int.ceil ((15: ℚ) / 8 * (-34: ℚ) / 4) - Int.floor ((15: ℚ) / 8 * Int.floor ((-34: ℚ) / 4))) = 2 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l899_89980


namespace NUMINAMATH_GPT_units_digit_of_x_l899_89901

theorem units_digit_of_x (p x : ℕ): 
  (p * x = 32 ^ 10) → 
  (p % 10 = 6) → 
  (x % 4 = 0) → 
  (x % 10 = 1) :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_x_l899_89901


namespace NUMINAMATH_GPT_base_5_to_base_10_conversion_l899_89906

/-- An alien creature communicated that it produced 263_5 units of a resource. 
    Convert this quantity to base 10. -/
theorem base_5_to_base_10_conversion : ∀ (n : ℕ), n = 2 * 5^2 + 6 * 5^1 + 3 * 5^0 → n = 83 :=
by
  intros n h
  rw [h]
  sorry

end NUMINAMATH_GPT_base_5_to_base_10_conversion_l899_89906


namespace NUMINAMATH_GPT_initial_distance_l899_89958

/-- Suppose Jack walks at a speed of 3 feet per second toward Christina,
    Christina walks at a speed of 3 feet per second toward Jack, and their dog Lindy
    runs at a speed of 10 feet per second back and forth between Jack and Christina.
    Given that Lindy travels a total of 400 feet when they meet, prove that the initial
    distance between Jack and Christina is 240 feet. -/
theorem initial_distance (initial_distance_jack_christina : ℝ)
  (jack_speed : ℝ := 3)
  (christina_speed : ℝ := 3)
  (lindy_speed : ℝ := 10)
  (lindy_total_distance : ℝ := 400):
  initial_distance_jack_christina = 240 :=
sorry

end NUMINAMATH_GPT_initial_distance_l899_89958


namespace NUMINAMATH_GPT_first_die_sides_l899_89982

theorem first_die_sides (n : ℕ) 
  (h_prob : (1 : ℝ) / n * (1 : ℝ) / 7 = 0.023809523809523808) : 
  n = 6 := by
  sorry

end NUMINAMATH_GPT_first_die_sides_l899_89982


namespace NUMINAMATH_GPT_opposite_of_neg_eight_l899_89961

theorem opposite_of_neg_eight : (-(-8)) = 8 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_eight_l899_89961


namespace NUMINAMATH_GPT_inequality_proof_l899_89923

theorem inequality_proof (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b < 2) : 
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a * b)) ∧ 
  (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a * b) ↔ 0 < a ∧ a = b ∧ a < 1) := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l899_89923


namespace NUMINAMATH_GPT_min_value_ab2_cd_l899_89971

noncomputable def arithmetic_seq (x a b y : ℝ) : Prop :=
  2 * a = x + b ∧ 2 * b = a + y

noncomputable def geometric_seq (x c d y : ℝ) : Prop :=
  c^2 = x * d ∧ d^2 = c * y

theorem min_value_ab2_cd (x y a b c d : ℝ) :
  (x > 0) → (y > 0) → arithmetic_seq x a b y → geometric_seq x c d y → 
  (a + b) ^ 2 / (c * d) ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_ab2_cd_l899_89971


namespace NUMINAMATH_GPT_find_c_value_l899_89907

theorem find_c_value (x y n m c : ℕ) 
  (h1 : 10 * x + y = 8 * n) 
  (h2 : 10 + x + y = 9 * m) 
  (h3 : c = x + y) : 
  c = 8 := 
by
  sorry

end NUMINAMATH_GPT_find_c_value_l899_89907


namespace NUMINAMATH_GPT_cos_sum_seventh_roots_of_unity_l899_89987

noncomputable def cos_sum (α : ℝ) : ℝ := 
  Real.cos α + Real.cos (2 * α) + Real.cos (4 * α)

theorem cos_sum_seventh_roots_of_unity (z : ℂ) (α : ℝ)
  (hz : z^7 = 1) (hz_ne_one : z ≠ 1) (hα : z = Complex.exp (Complex.I * α)) :
  cos_sum α = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_cos_sum_seventh_roots_of_unity_l899_89987


namespace NUMINAMATH_GPT_percentage_of_adults_is_40_l899_89969

variables (A C : ℕ)

-- Given conditions as definitions
def total_members := 120
def more_children_than_adults := 24
def percentage_of_adults (A : ℕ) := (A.toFloat / total_members.toFloat) * 100

-- Lean 4 statement to prove the percentage of adults
theorem percentage_of_adults_is_40 (h1 : A + C = 120)
                                   (h2 : C = A + 24) :
  percentage_of_adults A = 40 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_adults_is_40_l899_89969


namespace NUMINAMATH_GPT_negation_of_exists_gt_implies_forall_leq_l899_89942

theorem negation_of_exists_gt_implies_forall_leq (x : ℝ) (h : 0 < x) :
  ¬ (∃ x : ℝ, 0 < x ∧ x^3 - x + 1 > 0) ↔ ∀ x : ℝ, 0 < x → x^3 - x + 1 ≤ 0 :=
by sorry

end NUMINAMATH_GPT_negation_of_exists_gt_implies_forall_leq_l899_89942


namespace NUMINAMATH_GPT_PE_bisects_CD_given_conditions_l899_89957

variables {A B C D E P : Type*}

noncomputable def cyclic_quadrilateral (A B C D : Type*) : Prop := sorry

noncomputable def AD_squared_plus_BC_squared_eq_AB_squared (A B C D : Type*) : Prop := sorry

noncomputable def angles_equality_condition (A B C D P : Type*) : Prop := sorry

noncomputable def line_PE_bisects_CD (P E C D : Type*) : Prop := sorry

theorem PE_bisects_CD_given_conditions
  (h1 : cyclic_quadrilateral A B C D)
  (h2 : AD_squared_plus_BC_squared_eq_AB_squared A B C D)
  (h3 : angles_equality_condition A B C D P) :
  line_PE_bisects_CD P E C D :=
sorry

end NUMINAMATH_GPT_PE_bisects_CD_given_conditions_l899_89957


namespace NUMINAMATH_GPT_line_length_after_erasing_l899_89967

theorem line_length_after_erasing :
  ∀ (initial_length_m : ℕ) (conversion_factor : ℕ) (erased_length_cm : ℕ),
  initial_length_m = 1 → conversion_factor = 100 → erased_length_cm = 33 →
  initial_length_m * conversion_factor - erased_length_cm = 67 :=
by {
  sorry
}

end NUMINAMATH_GPT_line_length_after_erasing_l899_89967


namespace NUMINAMATH_GPT_tammy_total_distance_l899_89949

-- Define the times and speeds for each segment and breaks
def initial_speed : ℝ := 55   -- miles per hour
def initial_time : ℝ := 2     -- hours
def road_speed : ℝ := 40      -- miles per hour
def road_time : ℝ := 5        -- hours
def first_break : ℝ := 1      -- hour
def drive_after_break_speed : ℝ := 50  -- miles per hour
def drive_after_break_time : ℝ := 15   -- hours
def hilly_speed : ℝ := 35     -- miles per hour
def hilly_time : ℝ := 3       -- hours
def second_break : ℝ := 0.5   -- hours
def finish_speed : ℝ := 60    -- miles per hour
def total_journey_time : ℝ := 36 -- hours

-- Define a function to calculate the segment distance
def distance (speed time : ℝ) : ℝ := speed * time

-- Define the total distance calculation
def total_distance : ℝ :=
  distance initial_speed initial_time +
  distance road_speed road_time +
  distance drive_after_break_speed drive_after_break_time +
  distance hilly_speed hilly_time +
  distance finish_speed (total_journey_time - (initial_time + road_time + drive_after_break_time + hilly_time + first_break + second_break))

-- The final proof statement
theorem tammy_total_distance : total_distance = 1735 :=
  sorry

end NUMINAMATH_GPT_tammy_total_distance_l899_89949


namespace NUMINAMATH_GPT_total_apples_l899_89912

def pinky_apples : ℕ := 36
def danny_apples : ℕ := 73

theorem total_apples :
  pinky_apples + danny_apples = 109 :=
by
  sorry

end NUMINAMATH_GPT_total_apples_l899_89912


namespace NUMINAMATH_GPT_ceil_sqrt_200_eq_15_l899_89968

theorem ceil_sqrt_200_eq_15 : Int.ceil (Real.sqrt 200) = 15 := by
  sorry

end NUMINAMATH_GPT_ceil_sqrt_200_eq_15_l899_89968
