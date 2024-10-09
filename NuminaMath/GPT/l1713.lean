import Mathlib

namespace geometric_sequence_sum_twenty_terms_l1713_171368

noncomputable def geom_seq_sum : ℕ → ℕ → ℕ := λ a r =>
  if r = 1 then a * (1 + 20 - 1) else a * ((1 - r^20) / (1 - r))

theorem geometric_sequence_sum_twenty_terms (a₁ q : ℕ) (h1 : a₁ * (q + 2) = 4) (h2 : (a₃:ℕ) * (q ^ 4) = (a₁ : ℕ) * (q ^ 4)) :
  geom_seq_sum a₁ q = 2^20 - 1 :=
sorry

end geometric_sequence_sum_twenty_terms_l1713_171368


namespace no_opposite_meanings_in_C_l1713_171398

def opposite_meanings (condition : String) : Prop :=
  match condition with
  | "A" => true
  | "B" => true
  | "C" => false
  | "D" => true
  | _   => false

theorem no_opposite_meanings_in_C :
  opposite_meanings "C" = false :=
by
  -- proof goes here
  sorry

end no_opposite_meanings_in_C_l1713_171398


namespace kolya_or_leva_l1713_171334

theorem kolya_or_leva (k l : ℝ) (hkl : k > 0) (hll : l > 0) : 
  (k > l → ∃ a b c : ℝ, a = l + (2 / 3) * (k - l) ∧ b = (1 / 6) * (k - l) ∧ c = (1 / 6) * (k - l) ∧ a > b + c + l ∧ ¬(a < b + c + a)) ∨ 
  (k ≤ l → ∃ k1 k2 k3 : ℝ, k1 ≥ k2 ∧ k2 ≥ k3 ∧ k = k1 + k2 + k3 ∧ ∃ a' b' c' : ℝ, a' = k1 ∧ b' = (l - k1) / 2 ∧ c' = (l - k1) / 2 ∧ a' + a' > k2 ∧ b' + b' > k3) :=
by sorry

end kolya_or_leva_l1713_171334


namespace max_area_of_house_l1713_171382

-- Definitions for conditions
def height_of_plates : ℝ := 2.5
def price_per_meter_colored : ℝ := 450
def price_per_meter_composite : ℝ := 200
def roof_cost_per_sqm : ℝ := 200
def cost_limit : ℝ := 32000

-- Definitions for the variables
variables (x y : ℝ) (P S : ℝ)

-- Definition for the material cost P
def material_cost (x y : ℝ) : ℝ := 900 * x + 400 * y + 200 * x * y

-- Maximum area S and corresponding x
theorem max_area_of_house (x y : ℝ) (h : material_cost x y ≤ cost_limit) :
  S = 100 ∧ x = 20 / 3 :=
sorry

end max_area_of_house_l1713_171382


namespace domain_g_l1713_171323

noncomputable def g (x : ℝ) : ℝ := (x - 3) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_g : {x : ℝ | g x = (x - 3) / Real.sqrt (x^2 - 5 * x + 6)} = 
  {x : ℝ | x < 2 ∨ x > 3} :=
by
  sorry

end domain_g_l1713_171323


namespace donuts_per_student_l1713_171357

theorem donuts_per_student 
    (dozens_of_donuts : ℕ)
    (students_in_class : ℕ)
    (percentage_likes_donuts : ℕ)
    (students_who_like_donuts : ℕ)
    (total_donuts : ℕ)
    (donuts_per_student : ℕ) :
    dozens_of_donuts = 4 →
    students_in_class = 30 →
    percentage_likes_donuts = 80 →
    students_who_like_donuts = (percentage_likes_donuts * students_in_class) / 100 →
    total_donuts = dozens_of_donuts * 12 →
    donuts_per_student = total_donuts / students_who_like_donuts →
    donuts_per_student = 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end donuts_per_student_l1713_171357


namespace sufficient_condition_for_min_value_not_necessary_condition_for_min_value_l1713_171306

noncomputable def f (x b : ℝ) : ℝ := x^2 + b*x

theorem sufficient_condition_for_min_value (b : ℝ) : b < 0 → ∀ x, min (f (f x b) b) = min (f x b) :=
sorry

theorem not_necessary_condition_for_min_value (b : ℝ) : (b < 0) ∧ (∀ x, min (f (f x b) b) = min (f x b)) → b ≤ 0 ∨ b ≥ 2 := 
sorry

end sufficient_condition_for_min_value_not_necessary_condition_for_min_value_l1713_171306


namespace unpainted_cubes_count_l1713_171396

theorem unpainted_cubes_count :
  let L := 6
  let W := 6
  let H := 3
  (L - 2) * (W - 2) * (H - 2) = 16 :=
by
  sorry

end unpainted_cubes_count_l1713_171396


namespace total_cost_calculation_l1713_171391

def total_transportation_cost (x : ℝ) : ℝ :=
  let cost_A_to_C := 20 * x
  let cost_A_to_D := 30 * (240 - x)
  let cost_B_to_C := 24 * (200 - x)
  let cost_B_to_D := 32 * (60 + x)
  cost_A_to_C + cost_A_to_D + cost_B_to_C + cost_B_to_D

theorem total_cost_calculation (x : ℝ) :
  total_transportation_cost x = 13920 - 2 * x := by
  sorry

end total_cost_calculation_l1713_171391


namespace gcd_of_gy_and_y_l1713_171309

theorem gcd_of_gy_and_y (y : ℕ) (h : ∃ k : ℕ, y = k * 3456) :
  gcd ((5 * y + 4) * (9 * y + 1) * (12 * y + 6) * (3 * y + 9)) y = 216 :=
by {
  sorry
}

end gcd_of_gy_and_y_l1713_171309


namespace find_f_six_l1713_171358

noncomputable def f : ℕ → ℤ := sorry

axiom f_one_eq_one : f 1 = 1
axiom f_add (x y : ℕ) : f (x + y) = f x + f y + 8 * x * y - 2
axiom f_seven_eq_163 : f 7 = 163

theorem find_f_six : f 6 = 116 := 
by {
  sorry
}

end find_f_six_l1713_171358


namespace original_stone_count_145_l1713_171373

theorem original_stone_count_145 : 
  ∃ (n : ℕ), (n ≡ 1 [MOD 18]) ∧ (n = 145) :=
by
  sorry

end original_stone_count_145_l1713_171373


namespace closed_fishing_season_purpose_sustainable_l1713_171385

-- Defining the options for the purpose of the closed fishing season
inductive FishingPurpose
| sustainable_development : FishingPurpose
| inspect_fishing_vessels : FishingPurpose
| prevent_red_tides : FishingPurpose
| zoning_management : FishingPurpose

-- Defining rational utilization of resources involving fishing seasons
def rational_utilization (closed_fishing_season: Bool) : FishingPurpose := 
  if closed_fishing_season then FishingPurpose.sustainable_development 
  else FishingPurpose.inspect_fishing_vessels -- fallback for contradiction; shouldn't be used

-- The theorem we want to prove
theorem closed_fishing_season_purpose_sustainable :
  rational_utilization true = FishingPurpose.sustainable_development :=
sorry

end closed_fishing_season_purpose_sustainable_l1713_171385


namespace parallel_lines_m_values_l1713_171302

theorem parallel_lines_m_values (m : ℝ) :
  (∀ x y : ℝ, (3 + m) * x + 4 * y = 5) ∧ (2 * x + (5 + m) * y = 8) → (m = -1 ∨ m = -7) :=
by
  sorry

end parallel_lines_m_values_l1713_171302


namespace greatest_int_with_gcd_five_l1713_171340

theorem greatest_int_with_gcd_five (x : ℕ) (h1 : x < 150) (h2 : Nat.gcd x 30 = 5) : x ≤ 145 :=
by
  sorry

end greatest_int_with_gcd_five_l1713_171340


namespace factorize_x_squared_minus_25_l1713_171318

theorem factorize_x_squared_minus_25 : ∀ (x : ℝ), (x^2 - 25) = (x + 5) * (x - 5) :=
by
  intros x
  sorry

end factorize_x_squared_minus_25_l1713_171318


namespace remaining_sweets_in_packet_l1713_171342

theorem remaining_sweets_in_packet 
  (C : ℕ) (S : ℕ) (P : ℕ) (R : ℕ) (L : ℕ)
  (HC : C = 30) (HS : S = 100) (HP : P = 60) (HR : R = 25) (HL : L = 150) 
  : (C - (2 * C / 5) - ((C - P / 4) / 3)) 
  + (S - (S / 4)) 
  + (P - (3 * P / 5)) 
  + ((max 0 (R - (3 * R / 2))))
  + (L - (3 * (S / 4) / 2)) = 232 :=
by
  sorry

end remaining_sweets_in_packet_l1713_171342


namespace no_solution_for_steers_and_cows_purchase_l1713_171336

theorem no_solution_for_steers_and_cows_purchase :
  ¬ ∃ (s c : ℕ), 30 * s + 32 * c = 1200 ∧ c > s :=
by
  sorry

end no_solution_for_steers_and_cows_purchase_l1713_171336


namespace calc_difference_of_squares_l1713_171344

theorem calc_difference_of_squares :
  625^2 - 375^2 = 250000 :=
by sorry

end calc_difference_of_squares_l1713_171344


namespace original_amount_water_l1713_171345

theorem original_amount_water (O : ℝ) (h1 : (0.75 = 0.05 * O)) : O = 15 :=
by sorry

end original_amount_water_l1713_171345


namespace complement_intersection_l1713_171331

def M : Set ℝ := { x | x ≥ 1 }
def N : Set ℝ := { x | x < 2 }
def CR (S : Set ℝ) : Set ℝ := { x | x ∉ S }

theorem complement_intersection :
  CR (M ∩ N) = { x | x < 1 } ∪ { x | x ≥ 2 } := by
  sorry

end complement_intersection_l1713_171331


namespace only_zero_solution_l1713_171347

theorem only_zero_solution (a b c n : ℤ) (h_gcd : Int.gcd (Int.gcd (Int.gcd a b) c) n = 1)
  (h_eq : 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 :=
sorry

end only_zero_solution_l1713_171347


namespace seated_people_count_l1713_171317

theorem seated_people_count (n : ℕ) :
  (∀ (i : ℕ), i > 0 → i ≤ n) ∧
  (∀ (k : ℕ), k > 0 → k ≤ n → ∃ (p q : ℕ), 
         p = 31 ∧ q = 7 ∧ (p < n) ∧ (q < n) ∧
         p + 16 + 1 = q ∨ 
         p = 31 ∧ q = 14 ∧ (p < n) ∧ (q < n) ∧ 
         p - (n - q) + 1 = 16) → 
  n = 41 := 
by 
  sorry

end seated_people_count_l1713_171317


namespace evaluate_product_roots_of_unity_l1713_171310

theorem evaluate_product_roots_of_unity :
  let w := Complex.exp (2 * Real.pi * Complex.I / 13)
  (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) *
  (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * (3 - w^11) * (3 - w^12) =
  (3^12 + 3^11 + 3^10 + 3^9 + 3^8 + 3^7 + 3^6 + 3^5 + 3^4 + 3^3 + 3^2 + 3 + 1) :=
by
  sorry

end evaluate_product_roots_of_unity_l1713_171310


namespace igor_reach_top_time_l1713_171311

-- Define the conditions
def cabins_numbered_consecutively := (1, 99)
def igor_initial_cabin := 42
def first_aligned_cabin := 13
def second_aligned_cabin := 12
def alignment_time := 15
def total_cabins := 99
def expected_time := 17 * 60 + 15

-- State the problem as a theorem
theorem igor_reach_top_time :
  ∃ t, t = expected_time ∧
  -- Assume the cabins are numbered consecutively
  cabins_numbered_consecutively = (1, total_cabins) ∧
  -- Igor starts in cabin #42
  igor_initial_cabin = 42 ∧
  -- Cabin #42 first aligns with cabin #13, then aligns with cabin #12, 15 seconds later
  first_aligned_cabin = 13 ∧
  second_aligned_cabin = 12 ∧
  alignment_time = 15 :=
sorry

end igor_reach_top_time_l1713_171311


namespace triangle_perimeter_l1713_171322

-- Definitions of the geometric problem conditions
def inscribed_circle_tangent (A B C P : Type) : Prop := sorry
def radius_of_inscribed_circle (r : ℕ) : Prop := r = 24
def segment_lengths (AP PB : ℕ) : Prop := AP = 25 ∧ PB = 29

-- Main theorem to prove the perimeter of the triangle ABC
theorem triangle_perimeter (A B C P : Type) (r AP PB : ℕ)
  (H1 : inscribed_circle_tangent A B C P)
  (H2 : radius_of_inscribed_circle r)
  (H3 : segment_lengths AP PB) :
  2 * (54 + 208.72) = 525.44 :=
  sorry

end triangle_perimeter_l1713_171322


namespace Isabella_speed_is_correct_l1713_171392

-- Definitions based on conditions
def distance_km : ℝ := 17.138
def time_s : ℝ := 38

-- Conversion factor
def conversion_factor : ℝ := 1000

-- Distance in meters
def distance_m : ℝ := distance_km * conversion_factor

-- Correct answer (speed in m/s)
def correct_speed : ℝ := 451

-- Statement to prove
theorem Isabella_speed_is_correct : distance_m / time_s = correct_speed :=
by
  sorry

end Isabella_speed_is_correct_l1713_171392


namespace solve_equation_l1713_171312

theorem solve_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -6) :
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) ↔ x = -4 ∨ x = -2 :=
by
  sorry

end solve_equation_l1713_171312


namespace hypotenuse_length_l1713_171393

theorem hypotenuse_length (a c : ℝ) (h_perimeter : 2 * a + c = 36) (h_area : (1 / 2) * a^2 = 24) : c = 4 * Real.sqrt 6 :=
by
  sorry

end hypotenuse_length_l1713_171393


namespace part1_part2_part3_l1713_171361

noncomputable def f (x a : ℝ) : ℝ := x^2 + (x - 1) * |x - a|

-- Part 1
theorem part1 (a : ℝ) (x : ℝ) (h : a = -1) : 
  (f x a = 1) ↔ (x ≤ -1 ∨ x = 1) :=
sorry

-- Part 2
theorem part2 (a : ℝ) : 
  (∀ x y : ℝ, x < y → f x a < f y a) ↔ (a ≥ 1 / 3) :=
sorry

-- Part 3
theorem part3 (a : ℝ) (h1 : a < 1) (h2 : ∀ x : ℝ, f x a ≥ 2 * x - 3) : 
  -3 ≤ a ∧ a < 1 :=
sorry

end part1_part2_part3_l1713_171361


namespace total_files_deleted_l1713_171369

theorem total_files_deleted 
  (initial_files : ℕ) (initial_apps : ℕ)
  (deleted_files1 : ℕ) (deleted_apps1 : ℕ)
  (added_files1 : ℕ) (added_apps1 : ℕ)
  (deleted_files2 : ℕ) (deleted_apps2 : ℕ)
  (added_files2 : ℕ) (added_apps2 : ℕ)
  (final_files : ℕ) (final_apps : ℕ)
  (h_initial_files : initial_files = 24)
  (h_initial_apps : initial_apps = 13)
  (h_deleted_files1 : deleted_files1 = 5)
  (h_deleted_apps1 : deleted_apps1 = 3)
  (h_added_files1 : added_files1 = 7)
  (h_added_apps1 : added_apps1 = 4)
  (h_deleted_files2 : deleted_files2 = 10)
  (h_deleted_apps2 : deleted_apps2 = 4)
  (h_added_files2 : added_files2 = 5)
  (h_added_apps2 : added_apps2 = 7)
  (h_final_files : final_files = 21)
  (h_final_apps : final_apps = 17) :
  (deleted_files1 + deleted_files2 = 15) := 
by
  sorry

end total_files_deleted_l1713_171369


namespace derivative_f_eq_l1713_171304

noncomputable def f (x : ℝ) : ℝ :=
  (7^x * (3 * Real.sin (3 * x) + Real.cos (3 * x) * Real.log 7)) / (9 + Real.log 7 ^ 2)

theorem derivative_f_eq :
  ∀ x : ℝ, deriv f x = 7^x * Real.cos (3 * x) :=
by
  intro x
  sorry

end derivative_f_eq_l1713_171304


namespace largest_base5_number_conversion_l1713_171349

noncomputable def largest_base5_number_in_base10 : ℕ := 3124

theorem largest_base5_number_conversion :
  (4 * 5^4) + (4 * 5^3) + (4 * 5^2) + (4 * 5^1) + (4 * 5^0) = largest_base5_number_in_base10 :=
by
  -- Proof would go here
  sorry

end largest_base5_number_conversion_l1713_171349


namespace smallest_n_modulo_l1713_171379

theorem smallest_n_modulo (
  n : ℕ
) (h1 : 17 * n ≡ 5678 [MOD 11]) : n = 4 :=
by sorry

end smallest_n_modulo_l1713_171379


namespace Manny_lasagna_pieces_l1713_171329

-- Define variables and conditions
variable (M : ℕ) -- Manny's desired number of pieces
variable (A : ℕ := 0) -- Aaron's pieces
variable (K : ℕ := 2 * M) -- Kai's pieces
variable (R : ℕ := M / 2) -- Raphael's pieces
variable (L : ℕ := 2 + R) -- Lisa's pieces

-- Prove that Manny wants 1 piece of lasagna
theorem Manny_lasagna_pieces (M : ℕ) (A : ℕ := 0) (K : ℕ := 2 * M) (R : ℕ := M / 2) (L : ℕ := 2 + R) 
  (h : M + A + K + R + L = 6) : M = 1 :=
by
  sorry

end Manny_lasagna_pieces_l1713_171329


namespace speed_downstream_l1713_171383

variables (V_m V_s V_u V_d : ℕ)
variables (h1 : V_u = 12)
variables (h2 : V_m = 25)
variables (h3 : V_u = V_m - V_s)

theorem speed_downstream (h1 : V_u = 12) (h2 : V_m = 25) (h3 : V_u = V_m - V_s) :
  V_d = V_m + V_s :=
by
  -- The proof goes here
  sorry

end speed_downstream_l1713_171383


namespace length_of_d_in_proportion_l1713_171388

variable (a b c d : ℝ)

theorem length_of_d_in_proportion
  (h1 : a = 3) 
  (h2 : b = 2)
  (h3 : c = 6)
  (h_prop : a / b = c / d) : 
  d = 4 :=
by
  sorry

end length_of_d_in_proportion_l1713_171388


namespace largest_divisor_of_n4_sub_4n2_is_4_l1713_171300

theorem largest_divisor_of_n4_sub_4n2_is_4 (n : ℤ) : 4 ∣ (n^4 - 4 * n^2) :=
sorry

end largest_divisor_of_n4_sub_4n2_is_4_l1713_171300


namespace difference_of_extreme_valid_numbers_l1713_171360

theorem difference_of_extreme_valid_numbers :
  ∃ (largest smallest : ℕ),
    (largest = 222210 ∧ smallest = 100002) ∧ 
    (largest % 3 = 0 ∧ smallest % 3 = 0) ∧ 
    (largest ≥ 100000 ∧ largest < 1000000) ∧
    (smallest ≥ 100000 ∧ smallest < 1000000) ∧
    (∀ d, d ∈ [0, 1, 2] → (d ∈ [largest / 100000 % 10, largest / 10000 % 10, largest / 1000 % 10, largest / 100 % 10, largest / 10 % 10, largest % 10])) ∧
    (∀ d, d ∈ [0, 1, 2] → (d ∈ [smallest / 100000 % 10, smallest / 10000 % 10, smallest / 1000 % 10, smallest / 100 % 10, smallest / 10 % 10, smallest % 10])) ∧ 
    (∀ d ∈ [largest / 100000 % 10, largest / 10000 % 10, largest / 1000 % 10, largest / 100 % 10, largest / 10 % 10, largest % 10], d ∈ [0, 1, 2]) ∧
    (∀ d ∈ [smallest / 100000 % 10, smallest / 10000 % 10, smallest / 1000 % 10, smallest / 100 % 10, smallest / 10 % 10, smallest % 10], d ∈ [0, 1, 2]) ∧
    (largest - smallest = 122208) :=
by
  sorry

end difference_of_extreme_valid_numbers_l1713_171360


namespace trains_cross_in_12_seconds_l1713_171320

noncomputable def length := 120 -- Length of each train in meters
noncomputable def time_train1 := 10 -- Time taken by the first train to cross the post in seconds
noncomputable def time_train2 := 15 -- Time taken by the second train to cross the post in seconds

noncomputable def speed_train1 := length / time_train1 -- Speed of the first train in m/s
noncomputable def speed_train2 := length / time_train2 -- Speed of the second train in m/s

noncomputable def relative_speed := speed_train1 + speed_train2 -- Relative speed when traveling in opposite directions in m/s
noncomputable def total_length := 2 * length -- Total distance covered when crossing each other

noncomputable def crossing_time := total_length / relative_speed -- Time to cross each other in seconds

theorem trains_cross_in_12_seconds : crossing_time = 12 := by
  sorry

end trains_cross_in_12_seconds_l1713_171320


namespace geometric_common_ratio_l1713_171374

theorem geometric_common_ratio (a1 d : ℝ) (h1 : d ≠ 0) (h2 : (a1 + 5 * d)^2 = a1 * (a1 + 20 * d)) : 
  (a1 + 5 * d) / a1 = 3 :=
by
  sorry

end geometric_common_ratio_l1713_171374


namespace find_y_l1713_171307

theorem find_y (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : x = 4) : y = 3 := 
by sorry

end find_y_l1713_171307


namespace imo_1990_q31_l1713_171319

def A (n : ℕ) : ℕ := sorry -- definition of A(n)
def B (n : ℕ) : ℕ := sorry -- definition of B(n)
def f (n : ℕ) : ℕ := if B n = 1 then 1 else -- largest prime factor of B(n)
  sorry -- logic to find the largest prime factor of B(n)

theorem imo_1990_q31 :
  ∃ (M : ℕ), (∀ n : ℕ, f n ≤ M) ∧ (∀ N, (∀ n, f n ≤ N) → M ≤ N) ∧ M = 1999 :=
by sorry

end imo_1990_q31_l1713_171319


namespace probability_of_drawing_three_white_balls_l1713_171394

theorem probability_of_drawing_three_white_balls
  (total_balls white_balls black_balls: ℕ)
  (h_total: total_balls = 15)
  (h_white: white_balls = 7)
  (h_black: black_balls = 8)
  (draws: ℕ)
  (h_draws: draws = 3) :
  (Nat.choose white_balls draws / Nat.choose total_balls draws) = (7 / 91) :=
by sorry

end probability_of_drawing_three_white_balls_l1713_171394


namespace find_value_of_expression_l1713_171359

theorem find_value_of_expression (x : ℝ) (h : 5 * x - 3 = 7) : 3 * x^2 + 2 = 14 :=
by {
  sorry
}

end find_value_of_expression_l1713_171359


namespace tree_age_difference_l1713_171324

theorem tree_age_difference
  (groups_rings : ℕ)
  (rings_per_group : ℕ)
  (first_tree_groups : ℕ)
  (second_tree_groups : ℕ)
  (rings_per_year : ℕ)
  (h_rg : rings_per_group = 6)
  (h_ftg : first_tree_groups = 70)
  (h_stg : second_tree_groups = 40)
  (h_rpy : rings_per_year = 1) :
  ((first_tree_groups * rings_per_group) - (second_tree_groups * rings_per_group)) = 180 := 
by
  sorry

end tree_age_difference_l1713_171324


namespace erased_number_is_one_or_twenty_l1713_171316

theorem erased_number_is_one_or_twenty (x : ℕ) (h₁ : 1 ≤ x ∧ x ≤ 20)
  (h₂ : (210 - x) % 19 = 0) : x = 1 ∨ x = 20 :=
  by sorry

end erased_number_is_one_or_twenty_l1713_171316


namespace digit_at_position_2020_l1713_171363

def sequence_digit (n : Nat) : Nat :=
  -- Function to return the nth digit of the sequence formed by concatenating the integers from 1 to 1000
  sorry

theorem digit_at_position_2020 : sequence_digit 2020 = 7 :=
  sorry

end digit_at_position_2020_l1713_171363


namespace simplify_and_multiply_l1713_171327

theorem simplify_and_multiply :
  let a := 3
  let b := 17
  let d1 := 504
  let d2 := 72
  let m := 5
  let n := 7
  let fraction1 := a / d1
  let fraction2 := b / d2
  ((fraction1 - (b * n / (d2 * n))) * (m / n)) = (-145 / 882) :=
by
  sorry

end simplify_and_multiply_l1713_171327


namespace convert_20202_3_l1713_171387

def ternary_to_decimal (a4 a3 a2 a1 a0 : ℕ) : ℕ :=
  a4 * 3^4 + a3 * 3^3 + a2 * 3^2 + a1 * 3^1 + a0 * 3^0

theorem convert_20202_3 : ternary_to_decimal 2 0 2 0 2 = 182 :=
  sorry

end convert_20202_3_l1713_171387


namespace min_absolute_sum_value_l1713_171353

def absolute_sum (x : ℝ) : ℝ :=
  abs (x + 3) + abs (x + 6) + abs (x + 7)

theorem min_absolute_sum_value : ∃ x, absolute_sum x = 4 :=
sorry

end min_absolute_sum_value_l1713_171353


namespace smallest_b_for_quadratic_factors_l1713_171399

theorem smallest_b_for_quadratic_factors :
  ∃ b : ℕ, (∀ r s : ℤ, (r * s = 1764 → r + s = b) → b = 84) :=
sorry

end smallest_b_for_quadratic_factors_l1713_171399


namespace coordinate_sum_l1713_171381

theorem coordinate_sum (f : ℝ → ℝ) (x y : ℝ) (h₁ : f 9 = 7) (h₂ : 3 * y = f (3 * x) / 3 + 3) (h₃ : x = 3) : 
  x + y = 43 / 9 :=
by
  -- Proof goes here
  sorry

end coordinate_sum_l1713_171381


namespace intersection_of_M_and_N_l1713_171305

-- Definitions of the sets
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x : ℤ | -2 < x ∧ x < 2}

-- The theorem to prove
theorem intersection_of_M_and_N : M ∩ N = {-1, 0, 1} := by
  sorry

end intersection_of_M_and_N_l1713_171305


namespace maximize_fraction_l1713_171370

theorem maximize_fraction (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y + z)^2 / (x^2 + y^2 + z^2) ≤ 3 :=
sorry

end maximize_fraction_l1713_171370


namespace eve_spending_l1713_171372

-- Definitions of the conditions
def cost_mitt : ℝ := 14.00
def cost_apron : ℝ := 16.00
def cost_utensils : ℝ := 10.00
def cost_knife : ℝ := 2 * cost_utensils -- Twice the amount of the utensils
def discount_rate : ℝ := 0.25
def num_nieces : ℝ := 3

-- Total cost before the discount for one kit
def total_cost_one_kit : ℝ :=
  cost_mitt + cost_apron + cost_utensils + cost_knife

-- Discount for one kit
def discount_one_kit : ℝ := 
  total_cost_one_kit * discount_rate

-- Discounted price for one kit
def discounted_cost_one_kit : ℝ :=
  total_cost_one_kit - discount_one_kit

-- Total cost for all kits
def total_cost_all_kits : ℝ :=
  num_nieces * discounted_cost_one_kit

-- The theorem statement
theorem eve_spending : total_cost_all_kits = 135.00 :=
by sorry

end eve_spending_l1713_171372


namespace TableCostEquals_l1713_171389

-- Define the given conditions and final result
def total_spent : ℕ := 56
def num_chairs : ℕ := 2
def chair_cost : ℕ := 11
def table_cost : ℕ := 34

-- State the assertion to be proved
theorem TableCostEquals :
  table_cost = total_spent - (num_chairs * chair_cost) := 
by 
  sorry

end TableCostEquals_l1713_171389


namespace walkway_area_correct_l1713_171376

/-- Definitions as per problem conditions --/
def bed_length : ℕ := 8
def bed_width : ℕ := 3
def walkway_bed_width : ℕ := 2
def walkway_row_width : ℕ := 1
def num_beds_in_row : ℕ := 3
def num_rows : ℕ := 4

/-- Total dimensions including walkways --/
def total_width := num_beds_in_row * bed_length + (num_beds_in_row + 1) * walkway_bed_width
def total_height := num_rows * bed_width + (num_rows + 1) * walkway_row_width

/-- Total areas --/
def total_area := total_width * total_height
def bed_area := bed_length * bed_width
def total_bed_area := num_beds_in_row * num_rows * bed_area
def walkway_area := total_area - total_bed_area

theorem walkway_area_correct : walkway_area = 256 := by
  /- Import necessary libraries and skip the proof -/
  sorry

end walkway_area_correct_l1713_171376


namespace mike_pens_given_l1713_171390

noncomputable def pens_remaining (initial_pens mike_pens : ℕ) : ℕ :=
  2 * (initial_pens + mike_pens) - 19

theorem mike_pens_given 
  (initial_pens : ℕ)
  (mike_pens final_pens : ℕ) 
  (H1 : initial_pens = 7)
  (H2 : final_pens = 39) 
  (H3 : pens_remaining initial_pens mike_pens = final_pens) : 
  mike_pens = 22 := sorry

end mike_pens_given_l1713_171390


namespace find_M_l1713_171346

theorem find_M (a b M : ℝ) (h : (a + 2 * b)^2 = (a - 2 * b)^2 + M) : M = 8 * a * b :=
by sorry

end find_M_l1713_171346


namespace min_prime_factors_of_expression_l1713_171321

theorem min_prime_factors_of_expression (m n : ℕ) : 
  ∃ p1 p2 : ℕ, Prime p1 ∧ Prime p2 ∧ p1 ≠ p2 ∧ p1 ∣ (m * (n + 9) * (m + 2 * n^2 + 3)) ∧ p2 ∣ (m * (n + 9) * (m + 2 * n^2 + 3)) := 
sorry

end min_prime_factors_of_expression_l1713_171321


namespace equal_sunday_tuesday_count_l1713_171352

theorem equal_sunday_tuesday_count (h : ∀ (d : ℕ), d < 7 → d ≠ 0 → d ≠ 1 → d ≠ 2 → d ≠ 3) :
  ∃! d, d = 4 :=
by
  -- proof here
  sorry

end equal_sunday_tuesday_count_l1713_171352


namespace unique_solution_l1713_171384

-- Define the system of equations
def system_of_equations (m x y : ℝ) := 
  (m + 1) * x - y - 3 * m = 0 ∧ 4 * x + (m - 1) * y + 7 = 0

-- Define the determinant condition
def determinant_nonzero (m : ℝ) := m^2 + 3 ≠ 0

-- Theorem to prove there is exactly one solution
theorem unique_solution (m x y : ℝ) : 
  determinant_nonzero m → ∃! (x y : ℝ), system_of_equations m x y :=
by
  sorry

end unique_solution_l1713_171384


namespace molecular_weight_CCl4_l1713_171375

theorem molecular_weight_CCl4 (MW_7moles_CCl4 : ℝ) (h : MW_7moles_CCl4 = 1064) : 
  MW_7moles_CCl4 / 7 = 152 :=
by
  sorry

end molecular_weight_CCl4_l1713_171375


namespace number_of_candidates_l1713_171308

theorem number_of_candidates (n : ℕ) (h : n * (n - 1) = 132) : n = 12 :=
by
  sorry

end number_of_candidates_l1713_171308


namespace find_m_l1713_171332

variable (m : ℝ)

def vector_oa : ℝ × ℝ := (-1, 2)
def vector_ob : ℝ × ℝ := (3, m)

def orthogonal (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem find_m
  (h : orthogonal (vector_oa) (vector_ob m)) :
  m = 3 / 2 := by
  sorry

end find_m_l1713_171332


namespace no_polynomial_deg_ge_3_satisfies_conditions_l1713_171386

theorem no_polynomial_deg_ge_3_satisfies_conditions :
  ¬ ∃ f : Polynomial ℝ, f.degree ≥ 3 ∧ f.eval (x^2) = (f.eval x)^2 ∧ f.coeff 2 = 0 :=
sorry

end no_polynomial_deg_ge_3_satisfies_conditions_l1713_171386


namespace barrel_capacity_l1713_171377

theorem barrel_capacity (x y : ℝ) (h1 : y = 45 / (3/5)) (h2 : 0.6*x = y*3/5) (h3 : 0.4*x = 18) : 
  y = 75 :=
by
  sorry

end barrel_capacity_l1713_171377


namespace maximize_area_CDFE_l1713_171348

-- Given the side lengths of the rectangle
def AB : ℝ := 2
def AD : ℝ := 1

-- Definitions for points E and F
def AE (x : ℝ) : ℝ := x
def AF (x : ℝ) : ℝ := x

-- The formula for the area of quadrilateral CDFE
def area_CDFE (x : ℝ) : ℝ := 
  0.5 * x * (3 - 2 * x)

theorem maximize_area_CDFE : 
  ∃ x : ℝ, x = 3 / 4 ∧ area_CDFE x = 9 / 16 :=
by 
  sorry

end maximize_area_CDFE_l1713_171348


namespace find_a_l1713_171337

open Set

theorem find_a :
  ∀ (A B : Set ℕ) (a : ℕ),
    A = {1, 2, 3} →
    B = {2, a} →
    A ∪ B = {0, 1, 2, 3} →
    a = 0 :=
by
  intros A B a hA hB hUnion
  rw [hA, hB] at hUnion
  sorry

end find_a_l1713_171337


namespace arithmetic_sequence_index_l1713_171338

theorem arithmetic_sequence_index {a : ℕ → ℕ} (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n + 3) (h₃ : a n = 2014) : n = 672 :=
by
  sorry

end arithmetic_sequence_index_l1713_171338


namespace sum_of_integers_l1713_171397

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 15) (h3 : x * y = 56) : x + y = Real.sqrt 449 :=
by
  sorry

end sum_of_integers_l1713_171397


namespace range_2a_minus_b_and_a_div_b_range_3x_minus_y_l1713_171313

-- Proof for finding the range of 2a - b and a / b
theorem range_2a_minus_b_and_a_div_b (a b : ℝ) (h_a : 12 < a ∧ a < 60) (h_b : 15 < b ∧ b < 36) : 
  -12 < 2 * a - b ∧ 2 * a - b < 105 ∧ 1 / 3 < a / b ∧ a / b < 4 :=
by
  sorry

-- Proof for finding the range of 3x - y
theorem range_3x_minus_y (x y : ℝ) (h_xy_diff : -1 / 2 < x - y ∧ x - y < 1 / 2) (h_xy_sum : 0 < x + y ∧ x + y < 1) : 
  -1 < 3 * x - y ∧ 3 * x - y < 2 :=
by
  sorry

end range_2a_minus_b_and_a_div_b_range_3x_minus_y_l1713_171313


namespace solve_equation_l1713_171364

theorem solve_equation : ∀ x : ℝ, (x + 2) / 4 - 1 = (2 * x + 1) / 3 → x = -2 :=
by
  intro x
  intro h
  sorry  

end solve_equation_l1713_171364


namespace correct_calculation_l1713_171315

theorem correct_calculation (x : ℕ) (h1 : 21 * x = 63) : x + 40 = 43 :=
by
  -- proof steps would go here, but we skip them with 'sorry'
  sorry

end correct_calculation_l1713_171315


namespace altitude_circumradius_relation_l1713_171355

variable (a b c R ha : ℝ)
-- Assume S is the area of the triangle
variable (S : ℝ)
-- conditions
axiom area_circumradius : S = (a * b * c) / (4 * R)
axiom area_altitude : S = (a * ha) / 2

-- Prove the equivalence
theorem altitude_circumradius_relation 
  (area_circumradius : S = (a * b * c) / (4 * R)) 
  (area_altitude : S = (a * ha) / 2) : 
  ha = (b * c) / (2 * R) :=
sorry

end altitude_circumradius_relation_l1713_171355


namespace calc_f_xh_min_f_x_l1713_171356

def f (x : ℝ) : ℝ := 5 * x^2 - 2 * x - 1

theorem calc_f_xh_min_f_x (x h : ℝ) : f (x + h) - f x = h * (10 * x + 5 * h - 2) := 
by
  sorry

end calc_f_xh_min_f_x_l1713_171356


namespace john_spent_on_sweets_l1713_171354

theorem john_spent_on_sweets (initial_amount : ℝ) (amount_given_per_friend : ℝ) (friends : ℕ) (amount_left : ℝ) (total_spent_on_sweets : ℝ) :
  initial_amount = 20.10 →
  amount_given_per_friend = 1.00 →
  friends = 2 →
  amount_left = 17.05 →
  total_spent_on_sweets = initial_amount - (amount_given_per_friend * friends) - amount_left →
  total_spent_on_sweets = 1.05 :=
by
  intros h_initial h_given h_friends h_left h_spent
  sorry

end john_spent_on_sweets_l1713_171354


namespace gcd_51457_37958_l1713_171328

theorem gcd_51457_37958 : Nat.gcd 51457 37958 = 1 := 
  sorry

end gcd_51457_37958_l1713_171328


namespace partial_fraction_product_l1713_171343

theorem partial_fraction_product :
  ∃ (A B C : ℚ), 
  (∀ x : ℚ, x ≠ 1 ∧ x ≠ -3 ∧ x ≠ 4 → 
    (x^2 - 4) / (x^3 + x^2 - 11 * x - 13) = A / (x - 1) + B / (x + 3) + C / (x - 4)) ∧
  A * B * C = 5 / 196 :=
sorry

end partial_fraction_product_l1713_171343


namespace square_difference_l1713_171341

variable (n : ℕ)

theorem square_difference (n : ℕ) : (n + 1)^2 - n^2 = 2 * n + 1 :=
sorry

end square_difference_l1713_171341


namespace total_morning_afternoon_emails_l1713_171351

-- Define the conditions
def morning_emails : ℕ := 5
def afternoon_emails : ℕ := 8
def evening_emails : ℕ := 72

-- State the proof problem
theorem total_morning_afternoon_emails : 
  morning_emails + afternoon_emails = 13 := by
  sorry

end total_morning_afternoon_emails_l1713_171351


namespace sequence_count_l1713_171367

theorem sequence_count :
  ∃ n : ℕ, 
  (∀ a : Fin 101 → ℤ, 
    a 1 = 0 ∧ 
    a 100 = 475 ∧ 
    (∀ k : ℕ, 1 ≤ k ∧ k < 100 → |a (k + 1) - a k| = 5) → 
    n = 4851) := 
sorry

end sequence_count_l1713_171367


namespace find_possible_values_of_y_l1713_171314

noncomputable def solve_y (x : ℝ) : ℝ :=
  ((x - 3) ^ 2 * (x + 4)) / (2 * x - 4)

theorem find_possible_values_of_y (x : ℝ) 
  (h : x ^ 2 + 9 * (x / (x - 3)) ^ 2 = 90) : 
  solve_y x = 0 ∨ solve_y x = 105.23 := 
sorry

end find_possible_values_of_y_l1713_171314


namespace water_depth_correct_l1713_171362

noncomputable def water_depth (ron_height : ℝ) (dean_shorter_by : ℝ) : ℝ :=
  let dean_height := ron_height - dean_shorter_by
  2.5 * dean_height + 3

theorem water_depth_correct :
  water_depth 14.2 8.3 = 17.75 :=
by
  let ron_height := 14.2
  let dean_shorter_by := 8.3
  let dean_height := ron_height - dean_shorter_by
  let depth := 2.5 * dean_height + 3
  simp [water_depth, dean_height, depth]
  sorry

end water_depth_correct_l1713_171362


namespace train_passes_man_in_approx_18_seconds_l1713_171366

noncomputable def train_length : ℝ := 300 -- meters
noncomputable def train_speed : ℝ := 68 -- km/h
noncomputable def man_speed : ℝ := 8 -- km/h
noncomputable def kmh_to_mps (v : ℝ) : ℝ := v * 1000 / 3600
noncomputable def relative_speed_mps : ℝ := kmh_to_mps (train_speed - man_speed)
noncomputable def time_to_pass_man : ℝ := train_length / relative_speed_mps

theorem train_passes_man_in_approx_18_seconds :
  abs (time_to_pass_man - 18) < 1 :=
by
  sorry

end train_passes_man_in_approx_18_seconds_l1713_171366


namespace ratio_expression_value_l1713_171395

theorem ratio_expression_value (A B C : ℚ) (hA : A = 3 * B / 2) (hC : C = 5 * B / 2) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by sorry

end ratio_expression_value_l1713_171395


namespace eesha_late_by_15_minutes_l1713_171326

theorem eesha_late_by_15_minutes 
  (T usual_time : ℕ) (delay : ℕ) (slower_factor : ℚ) (T' : ℕ) 
  (usual_time_eq : usual_time = 60)
  (delay_eq : delay = 30)
  (slower_factor_eq : slower_factor = 0.75)
  (new_time_eq : T' = unusual_time * slower_factor) 
  (T'' : ℕ) (total_time_eq: T'' = T' + delay)
  (time_taken : ℕ) (time_diff_eq : time_taken = T'' - usual_time) :
  time_taken = 15 :=
by
  -- Proof construction
  sorry

end eesha_late_by_15_minutes_l1713_171326


namespace sum_of_solutions_l1713_171371

theorem sum_of_solutions (x : ℝ) (h : x + (25 / x) = 10) : x = 5 :=
by
  sorry

end sum_of_solutions_l1713_171371


namespace prime_has_property_p_l1713_171335

theorem prime_has_property_p (n : ℕ) (hn : Prime n) (a : ℕ) (h : n ∣ a^n - 1) : n^2 ∣ a^n - 1 := by
  sorry

end prime_has_property_p_l1713_171335


namespace original_price_l1713_171301

theorem original_price (P : ℝ) (h₁ : P - 0.30 * P = 0.70 * P) (h₂ : P - 0.20 * P = 0.80 * P) (h₃ : 0.70 * P + 0.80 * P = 50) :
  P = 100 / 3 :=
by
  -- Proof skipped
  sorry

end original_price_l1713_171301


namespace least_sum_exponents_of_1000_l1713_171365

def sum_least_exponents (n : ℕ) : ℕ :=
  if n = 1000 then 38 else 0 -- Since we only care about the case for 1000.

theorem least_sum_exponents_of_1000 :
  sum_least_exponents 1000 = 38 := by
  sorry

end least_sum_exponents_of_1000_l1713_171365


namespace coeff_x3_in_x_mul_1_add_x_pow_6_l1713_171333

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

theorem coeff_x3_in_x_mul_1_add_x_pow_6 :
  ∀ x : ℕ, (∃ c : ℕ, c * x^3 = x * (1 + x)^6 ∧ c = 15) :=
by
  sorry

end coeff_x3_in_x_mul_1_add_x_pow_6_l1713_171333


namespace expression_value_l1713_171339

def a : ℤ := 5
def b : ℤ := -3
def c : ℕ := 2

theorem expression_value : (3 * c) / (a + b) + c = 5 := by
  sorry

end expression_value_l1713_171339


namespace repeated_mul_eq_pow_l1713_171378

-- Define the repeated multiplication of 2, n times
def repeated_mul (n : ℕ) : ℕ :=
  (List.replicate n 2).prod

-- State the theorem to prove
theorem repeated_mul_eq_pow (n : ℕ) : repeated_mul n = 2 ^ n :=
by
  sorry

end repeated_mul_eq_pow_l1713_171378


namespace tile_arrangement_probability_l1713_171330

theorem tile_arrangement_probability :
  let X := 4  -- Number of tiles marked X
  let O := 2  -- Number of tiles marked O
  let total := 6  -- Total number of tiles
  let arrangement := [true, true, false, true, false, true]  -- XXOXOX represented as [X, X, O, X, O, X]
  (↑(X / total) * ↑((X - 1) / (total - 1)) * ↑((O / (total - 2))) * ↑((X - 2) / (total - 3)) * ↑((O - 1) / (total - 4)) * 1 : ℚ) = 1 / 15 :=
sorry

end tile_arrangement_probability_l1713_171330


namespace melissa_bananas_l1713_171303

theorem melissa_bananas (a b : ℕ) (h1 : a = 88) (h2 : b = 4) : a - b = 84 :=
by
  sorry

end melissa_bananas_l1713_171303


namespace batsman_inning_problem_l1713_171380

-- Define the problem in Lean 4
theorem batsman_inning_problem (n R : ℕ) (h1 : R = 55 * n) (h2 : R + 110 = 60 * (n + 1)) : n + 1 = 11 := 
  sorry

end batsman_inning_problem_l1713_171380


namespace eliot_account_balance_l1713_171350

theorem eliot_account_balance (A E : ℝ) 
  (h1 : A > E)
  (h2 : A - E = (1 / 12) * (A + E))
  (h3 : 1.10 * A - 1.15 * E = 22) : 
  E = 146.67 :=
by
  sorry

end eliot_account_balance_l1713_171350


namespace amare_additional_fabric_needed_l1713_171325

-- Defining the conditions
def yards_per_dress : ℝ := 5.5
def num_dresses : ℝ := 4
def initial_fabric_feet : ℝ := 7
def yard_to_feet : ℝ := 3

-- The theorem to prove
theorem amare_additional_fabric_needed : 
  (yards_per_dress * num_dresses * yard_to_feet) - initial_fabric_feet = 59 := 
by
  sorry

end amare_additional_fabric_needed_l1713_171325
