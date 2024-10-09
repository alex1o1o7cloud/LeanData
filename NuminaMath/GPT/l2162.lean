import Mathlib

namespace daniel_age_is_13_l2162_216207

-- Define Aunt Emily's age
def aunt_emily_age : ℕ := 48

-- Define Brianna's age as a third of Aunt Emily's age
def brianna_age : ℕ := aunt_emily_age / 3

-- Define that Daniel's age is 3 years less than Brianna's age
def daniel_age : ℕ := brianna_age - 3

-- Theorem to prove Daniel's age is 13 given the conditions
theorem daniel_age_is_13 :
  brianna_age = aunt_emily_age / 3 →
  daniel_age = brianna_age - 3 →
  daniel_age = 13 :=
  sorry

end daniel_age_is_13_l2162_216207


namespace find_a_l2162_216208

-- Definitions of sets A, B, and C
def setA : Set ℝ := { x | x^2 + 2 * x - 8 = 0 }
def setB : Set ℝ := { x | Real.log (x^2 - 5 * x + 8) / Real.log 2 = 1 }
def setC (a : ℝ) : Set ℝ := { x | x^2 - a * x + a^2 - 19 = 0 }

-- Proof statement to find the value of a
theorem find_a (a : ℝ) : setA ∩ setC a = ∅ → setB ∩ setC a ≠ ∅ → a = -2 := by
  sorry

end find_a_l2162_216208


namespace div_sub_eq_l2162_216257

theorem div_sub_eq : 0.24 / 0.004 - 0.1 = 59.9 := by
  sorry

end div_sub_eq_l2162_216257


namespace area_of_trapezium_eq_336_l2162_216204

-- Define the lengths of the parallel sides and the distance between them
def a := 30 -- length of one parallel side in cm
def b := 12 -- length of the other parallel side in cm
def h := 16 -- distance between the parallel sides (height) in cm

-- Define the expected area
def expectedArea := 336 -- area in square cm

-- State the theorem to prove
theorem area_of_trapezium_eq_336 : (1/2 : ℝ) * (a + b) * h = expectedArea := 
by 
  -- The proof is omitted
  sorry

end area_of_trapezium_eq_336_l2162_216204


namespace khali_shovels_snow_l2162_216202

theorem khali_shovels_snow :
  let section1_length := 30
  let section1_width := 3
  let section1_depth := 1
  let section2_length := 15
  let section2_width := 2
  let section2_depth := 0.5
  let volume1 := section1_length * section1_width * section1_depth
  let volume2 := section2_length * section2_width * section2_depth
  volume1 + volume2 = 105 :=
by 
  sorry

end khali_shovels_snow_l2162_216202


namespace function_characterization_l2162_216253

def isRelativelyPrime (x y : ℕ) : Prop := Nat.gcd x y = 1

theorem function_characterization (f : ℕ → ℤ) (hyp : ∀ x y, isRelativelyPrime x y → f (x + y) = f (x + 1) + f (y + 1)) :
  ∃ a b : ℤ, ∀ n : ℕ, f (2 * n) = (n - 1) * b ∧ f (2 * n + 1) = (n - 1) * b + a :=
by
  sorry

end function_characterization_l2162_216253


namespace geometric_sequence_property_l2162_216255

theorem geometric_sequence_property (a : ℕ → ℝ) (h : ∀ n, a (n + 1) / a n = a 1 / a 0) (h₁ : a 5 * a 14 = 5) : 
  a 8 * a 9 * a 10 * a 11 = 25 :=
by
  sorry

end geometric_sequence_property_l2162_216255


namespace necessary_not_sufficient_condition_not_sufficient_condition_l2162_216256

theorem necessary_not_sufficient_condition (x : ℝ) :
  (1 < x ∧ x < 4) → (|x - 2| < 1) := sorry

theorem not_sufficient_condition (x : ℝ) :
  (|x - 2| < 1) → (1 < x ∧ x < 4) := sorry

end necessary_not_sufficient_condition_not_sufficient_condition_l2162_216256


namespace num_units_from_batch_B_l2162_216297

theorem num_units_from_batch_B
  (A B C : ℝ) -- quantities of products from batches A, B, and C
  (h_arith_seq : B - A = C - B) -- batches A, B, and C form an arithmetic sequence
  (h_total : A + B + C = 240)    -- total units from three batches
  (h_sample_size : A + B + C = 60)  -- sample size drawn equals 60
  : B = 20 := 
by {
  sorry
}

end num_units_from_batch_B_l2162_216297


namespace statues_at_end_of_fourth_year_l2162_216213

def initial_statues : ℕ := 4
def statues_after_second_year : ℕ := initial_statues * 4
def statues_added_third_year : ℕ := 12
def broken_statues_third_year : ℕ := 3
def statues_removed_third_year : ℕ := broken_statues_third_year
def statues_added_fourth_year : ℕ := broken_statues_third_year * 2

def statues_end_of_first_year : ℕ := initial_statues
def statues_end_of_second_year : ℕ := statues_after_second_year
def statues_end_of_third_year : ℕ := statues_end_of_second_year + statues_added_third_year - statues_removed_third_year
def statues_end_of_fourth_year : ℕ := statues_end_of_third_year + statues_added_fourth_year

theorem statues_at_end_of_fourth_year : statues_end_of_fourth_year = 31 :=
by
  sorry

end statues_at_end_of_fourth_year_l2162_216213


namespace greatest_two_digit_multiple_of_17_l2162_216278

theorem greatest_two_digit_multiple_of_17 : ∃ N, N = 85 ∧ 
  (∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85) :=
by 
  sorry

end greatest_two_digit_multiple_of_17_l2162_216278


namespace find_setC_l2162_216206

def setA := {x : ℝ | x^2 - 3 * x + 2 = 0}
def setB (a : ℝ) := {x : ℝ | a * x - 2 = 0}
def union_condition (a : ℝ) : Prop := (setA ∪ setB a) = setA
def setC := {a : ℝ | union_condition a}

theorem find_setC : setC = {0, 1, 2} :=
by
  sorry

end find_setC_l2162_216206


namespace coin_flip_heads_probability_l2162_216231

theorem coin_flip_heads_probability :
  let coins : List String := ["penny", "nickel", "dime", "quarter", "half-dollar"]
  let independent_event (coin : String) : Prop := True
  let outcomes := (2 : ℕ) ^ (List.length coins)
  let successful_outcomes := 4
  let probability := successful_outcomes / outcomes
  probability = 1 / 8 := 
by
  sorry

end coin_flip_heads_probability_l2162_216231


namespace min_value_fraction_l2162_216258

theorem min_value_fraction (a b c : ℝ) (h1 : a > 0) (h2 : a < (2 / 3) * b) (h3 : c ≥ b^2 / (3 * a)) : 
  ∃ x : ℝ, (∀ y : ℝ, y ≥ x → y ≥ 1) ∧ (x = 1) :=
by
  sorry

end min_value_fraction_l2162_216258


namespace frog_eyes_count_l2162_216229

def total_frog_eyes (a b c : ℕ) (eyesA eyesB eyesC : ℕ) : ℕ :=
  a * eyesA + b * eyesB + c * eyesC

theorem frog_eyes_count :
  let a := 2
  let b := 1
  let c := 3
  let eyesA := 2
  let eyesB := 3
  let eyesC := 4
  total_frog_eyes a b c eyesA eyesB eyesC = 19 := by
  sorry

end frog_eyes_count_l2162_216229


namespace cathy_can_win_l2162_216238

theorem cathy_can_win (n k : ℕ) (hn : 0 < n) (hk : 0 < k) : 
  (∃ (f : ℕ → ℕ) (hf : ∀ i, f i < n + 1), (∀ i j, (i < j) → (f i < f j) → (f j = f i + 1)) → n ≤ 2^(k-1)) :=
sorry

end cathy_can_win_l2162_216238


namespace calc_result_l2162_216283

theorem calc_result (initial_number : ℕ) (square : ℕ → ℕ) (subtract_five : ℕ → ℕ) : 
  initial_number = 7 ∧ (square 7 = 49) ∧ (subtract_five 49 = 44) → 
  subtract_five (square initial_number) = 44 := 
by
  sorry

end calc_result_l2162_216283


namespace compare_sqrt_difference_minimize_materials_plan_compare_a_inv_l2162_216209

-- Problem 1
theorem compare_sqrt_difference : 3 - Real.sqrt 2 > 4 - 2 * Real.sqrt 2 := 
  sorry

-- Problem 2
theorem minimize_materials_plan (x y : ℝ) (h : x > y) : 
  4 * x + 6 * y > 3 * x + 7 * y := 
  sorry

-- Problem 3
theorem compare_a_inv (a : ℝ) (h : a > 0) : 
  (0 < a ∧ a < 1) → a < 1 / a ∧ (a = 1 → a = 1 / a) ∧ (a > 1 → a > 1 / a) :=
  sorry

end compare_sqrt_difference_minimize_materials_plan_compare_a_inv_l2162_216209


namespace xiaolong_average_speed_l2162_216235

noncomputable def averageSpeed (dist_home_store : ℕ) (time_home_store : ℕ) 
                               (speed_store_playground : ℕ) (time_store_playground : ℕ) 
                               (dist_playground_school : ℕ) (speed_playground_school : ℕ) 
                               (total_time : ℕ) : ℕ :=
  let dist_store_playground := speed_store_playground * time_store_playground
  let time_playground_school := dist_playground_school / speed_playground_school
  let total_distance := dist_home_store + dist_store_playground + dist_playground_school
  total_distance / total_time

theorem xiaolong_average_speed :
  averageSpeed 500 7 80 8 300 60 20 = 72 := by
  sorry

end xiaolong_average_speed_l2162_216235


namespace increasing_on_interval_l2162_216232

theorem increasing_on_interval (a : ℝ) : (∀ x : ℝ, x > 1/2 → (2 * x + a + 1 / x^2) ≥ 0) → a ≥ -3 :=
by
  intros h
  -- Rest of the proof would go here
  sorry

end increasing_on_interval_l2162_216232


namespace speed_of_second_car_l2162_216245

/-!
Two cars started from the same point, at 5 am, traveling in opposite directions. 
One car was traveling at 50 mph, and they were 450 miles apart at 10 am. 
Prove that the speed of the other car is 40 mph.
-/

variable (S : ℝ) -- Speed of the second car

theorem speed_of_second_car
    (h1 : ∀ t : ℝ, t = 5) -- The time of travel from 5 am to 10 am is 5 hours 
    (h2 : ∀ d₁ : ℝ, d₁ = 50 * 5) -- Distance traveled by the first car
    (h3 : ∀ d₂ : ℝ, d₂ = S * 5) -- Distance traveled by the second car
    (h4 : 450 = 50 * 5 + S * 5) -- Total distance between the two cars
    : S = 40 := sorry

end speed_of_second_car_l2162_216245


namespace mult_63_37_l2162_216234

theorem mult_63_37 : 63 * 37 = 2331 :=
by {
  sorry
}

end mult_63_37_l2162_216234


namespace initial_bottle_caps_l2162_216299

theorem initial_bottle_caps 
    (x : ℝ) 
    (Nancy_bottle_caps : ℝ) 
    (Marilyn_current_bottle_caps : ℝ) 
    (h1 : Nancy_bottle_caps = 36.0)
    (h2 : Marilyn_current_bottle_caps = 87)
    (h3 : x + Nancy_bottle_caps = Marilyn_current_bottle_caps) : 
    x = 51 := 
by 
  sorry

end initial_bottle_caps_l2162_216299


namespace num_factors_of_1320_l2162_216211

theorem num_factors_of_1320 : ∃ n : ℕ, (n = 24) ∧ (∃ a b c d : ℕ, 1320 = 2^a * 3^b * 5^c * 11^d ∧ (a + 1) * (b + 1) * (c + 1) * (d + 1) = 24) :=
by
  sorry

end num_factors_of_1320_l2162_216211


namespace minimum_value_frac_sum_l2162_216216

theorem minimum_value_frac_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + 2 / y = 3) :
  (2 / x + y) ≥ 8 / 3 :=
sorry

end minimum_value_frac_sum_l2162_216216


namespace bicycle_distance_l2162_216272

def distance : ℝ := 15

theorem bicycle_distance :
  ∀ (x y : ℝ),
  (x + 6) * (y - 5 / 60) = x * y →
  (x - 5) * (y + 6 / 60) = x * y →
  x * y = distance :=
by
  intros x y h1 h2
  sorry

end bicycle_distance_l2162_216272


namespace Karen_has_fewer_nail_polishes_than_Kim_l2162_216230

theorem Karen_has_fewer_nail_polishes_than_Kim :
  ∀ (Kim Heidi Karen : ℕ), Kim = 12 → Heidi = Kim + 5 → Karen + Heidi = 25 → (Kim - Karen) = 4 :=
by
  intros Kim Heidi Karen hK hH hKH
  sorry

end Karen_has_fewer_nail_polishes_than_Kim_l2162_216230


namespace find_m_squared_plus_n_squared_l2162_216292

theorem find_m_squared_plus_n_squared (m n : ℝ) (h1 : (m - n) ^ 2 = 8) (h2 : (m + n) ^ 2 = 2) : m ^ 2 + n ^ 2 = 5 :=
by
  sorry

end find_m_squared_plus_n_squared_l2162_216292


namespace number_is_twenty_l2162_216240

-- We state that if \( \frac{30}{100}x = \frac{15}{100} \times 40 \), then \( x = 20 \)
theorem number_is_twenty (x : ℝ) (h : (30 / 100) * x = (15 / 100) * 40) : x = 20 :=
by
  sorry

end number_is_twenty_l2162_216240


namespace rooms_already_painted_l2162_216227

-- Define the conditions as variables and hypotheses
variables (total_rooms : ℕ) (hours_per_room : ℕ) (remaining_hours : ℕ)
variables (h1 : total_rooms = 10)
variables (h2 : hours_per_room = 8)
variables (h3 : remaining_hours = 16)

-- Define the theorem stating the number of rooms already painted
theorem rooms_already_painted (total_rooms : ℕ) (hours_per_room : ℕ) (remaining_hours : ℕ)
  (h1 : total_rooms = 10) (h2 : hours_per_room = 8) (h3 : remaining_hours = 16) :
  (total_rooms - (remaining_hours / hours_per_room) = 8) :=
sorry

end rooms_already_painted_l2162_216227


namespace find_s2_length_l2162_216200

variables (s r : ℝ)

def condition1 : Prop := 2 * r + s = 2420
def condition2 : Prop := 2 * r + 3 * s = 4040

theorem find_s2_length (h1 : condition1 s r) (h2 : condition2 s r) : s = 810 :=
sorry

end find_s2_length_l2162_216200


namespace sin_thirteen_pi_over_six_l2162_216287

-- Define a lean statement for the proof problem
theorem sin_thirteen_pi_over_six : Real.sin (13 * Real.pi / 6) = 1 / 2 := 
by 
  -- Add the proof later (or keep sorry if the proof is not needed)
  sorry

end sin_thirteen_pi_over_six_l2162_216287


namespace g_g_x_has_two_distinct_real_roots_iff_l2162_216210

noncomputable def g (d x : ℝ) := x^2 - 4 * x + d

def has_two_distinct_real_roots (f : ℝ → ℝ) : Prop := 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

theorem g_g_x_has_two_distinct_real_roots_iff (d : ℝ) :
  has_two_distinct_real_roots (g d ∘ g d) ↔ d = 8 := sorry

end g_g_x_has_two_distinct_real_roots_iff_l2162_216210


namespace stacy_history_paper_length_l2162_216286

theorem stacy_history_paper_length
  (days : ℕ)
  (pages_per_day : ℕ)
  (h_days : days = 6)
  (h_pages_per_day : pages_per_day = 11) :
  (days * pages_per_day) = 66 :=
by {
  sorry -- Proof goes here
}

end stacy_history_paper_length_l2162_216286


namespace area_of_largest_circle_l2162_216226

theorem area_of_largest_circle (side_length : ℝ) (h : side_length = 2) : 
  (Real.pi * (side_length / 2)^2 = 3.14) :=
by
  sorry

end area_of_largest_circle_l2162_216226


namespace friends_pets_ratio_l2162_216244

theorem friends_pets_ratio (pets_total : ℕ) (pets_taylor : ℕ) (pets_friend4 : ℕ) (pets_friend5 : ℕ)
  (pets_first3_total : ℕ) : pets_total = 32 → pets_taylor = 4 → pets_friend4 = 2 → pets_friend5 = 2 →
  pets_first3_total = pets_total - pets_taylor - pets_friend4 - pets_friend5 →
  (pets_first3_total : ℚ) / pets_taylor = 6 :=
by
  sorry

end friends_pets_ratio_l2162_216244


namespace smallest_x_l2162_216268

theorem smallest_x (x : ℕ) (h900 : 900 = 2^2 * 3^2 * 5^2) (h1152 : 1152 = 2^7 * 3^2) : 
  (900 * x) % 1152 = 0 ↔ x = 32 := 
by
  sorry

end smallest_x_l2162_216268


namespace spring_expenses_l2162_216223

noncomputable def expense_by_end_of_february : ℝ := 0.6
noncomputable def expense_by_end_of_may : ℝ := 1.8
noncomputable def spending_during_spring_months := expense_by_end_of_may - expense_by_end_of_february

-- Lean statement for the proof problem
theorem spring_expenses : spending_during_spring_months = 1.2 := by
  sorry

end spring_expenses_l2162_216223


namespace chewing_gum_company_revenue_l2162_216259

theorem chewing_gum_company_revenue (R : ℝ) :
  let projected_revenue := 1.25 * R
  let actual_revenue := 0.75 * R
  (actual_revenue / projected_revenue) * 100 = 60 := 
by
  sorry

end chewing_gum_company_revenue_l2162_216259


namespace Tim_pencils_value_l2162_216224

variable (Sarah_pencils : ℕ)
variable (Tyrah_pencils : ℕ)
variable (Tim_pencils : ℕ)

axiom Tyrah_condition : Tyrah_pencils = 6 * Sarah_pencils
axiom Tim_condition : Tim_pencils = 8 * Sarah_pencils
axiom Tyrah_pencils_value : Tyrah_pencils = 12

theorem Tim_pencils_value : Tim_pencils = 16 :=
by
  sorry

end Tim_pencils_value_l2162_216224


namespace opposite_neg_9_l2162_216246

theorem opposite_neg_9 : 
  ∃ y : Int, -9 + y = 0 ∧ y = 9 :=
by
  sorry

end opposite_neg_9_l2162_216246


namespace boxes_needed_for_loose_crayons_l2162_216271

-- Definitions based on conditions
def boxes_francine : ℕ := 5
def loose_crayons_francine : ℕ := 5
def loose_crayons_friend : ℕ := 27
def total_crayons_francine : ℕ := 85
def total_boxes_needed : ℕ := 2

-- The theorem to prove
theorem boxes_needed_for_loose_crayons 
  (hf : total_crayons_francine = boxes_francine * 16 + loose_crayons_francine)
  (htotal_loose : loose_crayons_francine + loose_crayons_friend = 32)
  (hboxes : boxes_francine = 5) : 
  total_boxes_needed = 2 :=
sorry

end boxes_needed_for_loose_crayons_l2162_216271


namespace find_fraction_l2162_216274

theorem find_fraction (x : ℝ) (h1 : 7 = (1 / 10) / 100 * 7000) (h2 : x * 7000 - 7 = 700) : x = 707 / 7000 :=
by sorry

end find_fraction_l2162_216274


namespace ball_hits_ground_time_l2162_216236

theorem ball_hits_ground_time (h : ℝ → ℝ) (t : ℝ) :
  (∀ (t : ℝ), h t = -16 * t ^ 2 - 30 * t + 200) → h t = 0 → t = 2.5 :=
by
  -- Placeholder for the formal proof
  sorry

end ball_hits_ground_time_l2162_216236


namespace square_of_binomial_l2162_216291

theorem square_of_binomial (a b : ℝ) : 
  (a - 5 * b)^2 = a^2 - 10 * a * b + 25 * b^2 :=
by
  sorry

end square_of_binomial_l2162_216291


namespace farm_field_proof_l2162_216289

section FarmField

variables 
  (planned_rate daily_rate : ℕ) -- planned_rate is 260 hectares/day, daily_rate is 85 hectares/day 
  (extra_days remaining_hectares : ℕ) -- extra_days is 2, remaining_hectares is 40
  (max_hours_per_day : ℕ) -- max_hours_per_day is 12

-- Definitions for soils
variables
  (A_percent B_percent C_percent : ℚ) (A_hours B_hours C_hours : ℕ)
  -- A_percent is 0.4, B_percent is 0.3, C_percent is 0.3
  -- A_hours is 4, B_hours is 6, C_hours is 3

-- Given conditions
axiom planned_rate_eq : planned_rate = 260
axiom daily_rate_eq : daily_rate = 85
axiom extra_days_eq : extra_days = 2
axiom remaining_hectares_eq : remaining_hectares = 40
axiom max_hours_per_day_eq : max_hours_per_day = 12

axiom A_percent_eq : A_percent = 0.4
axiom B_percent_eq : B_percent = 0.3
axiom C_percent_eq : C_percent = 0.3

axiom A_hours_eq : A_hours = 4
axiom B_hours_eq : B_hours = 6
axiom C_hours_eq : C_hours = 3

-- Theorem stating the problem
theorem farm_field_proof :
  ∃ (total_area initial_days : ℕ),
    total_area = 340 ∧ initial_days = 2 :=
by
  sorry

end FarmField

end farm_field_proof_l2162_216289


namespace calculate_sum_l2162_216277

open Real

theorem calculate_sum :
  (-1: ℝ) ^ 2023 + (1/2) ^ (-2: ℝ) + 3 * tan (pi / 6) - (3 - pi) ^ 0 + |sqrt 3 - 2| = 4 :=
by
  sorry

end calculate_sum_l2162_216277


namespace perpendicular_vectors_l2162_216249

/-- Given vectors a and b, prove that m = 6 if a is perpendicular to b -/
theorem perpendicular_vectors {m : ℝ} (h₁ : (1, 5, -2) = (1, 5, -2)) (h₂ : ∃ m : ℝ, (m, 2, m+2) = (m, 2, m+2)) (h₃ : (1 * m + 5 * 2 + (-2) * (m + 2) = 0)) :
  m = 6 :=
sorry

end perpendicular_vectors_l2162_216249


namespace hexagon_inscribed_circumscribed_symmetric_l2162_216215

-- Define the conditions of the problem
variables (R r c : ℝ)

-- Define the main assertion of the problem
theorem hexagon_inscribed_circumscribed_symmetric :
  3 * (R^2 - c^2)^4 - 4 * r^2 * (R^2 - c^2)^2 * (R^2 + c^2) - 16 * R^2 * c^2 * r^4 = 0 :=
by
  -- skipping proof
  sorry

end hexagon_inscribed_circumscribed_symmetric_l2162_216215


namespace find_k_l2162_216242

-- Defining the conditions used in the problem context
def line_condition (k a b : ℝ) : Prop :=
  (b = 4 * k + 1) ∧ (5 = k * a + 1) ∧ (b + 1 = k * a + 1)

-- The statement of the theorem
theorem find_k (a b k : ℝ) (h : line_condition k a b) : k = 3 / 4 :=
by sorry

end find_k_l2162_216242


namespace x_y_value_l2162_216218

theorem x_y_value (x y : ℝ) (h : x^2 + y^2 = 8 * x - 4 * y - 30) : x + y = 2 :=
sorry

end x_y_value_l2162_216218


namespace domain_of_f_l2162_216261

open Set

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 1)

theorem domain_of_f :
  {x : ℝ | x^2 - 1 > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 1} :=
by
  sorry

end domain_of_f_l2162_216261


namespace find_value_of_x_l2162_216294

theorem find_value_of_x (x : ℕ) (h : (50 + x / 90) * 90 = 4520) : x = 4470 :=
sorry

end find_value_of_x_l2162_216294


namespace magnesium_is_limiting_l2162_216222

-- Define the conditions
def moles_Mg : ℕ := 4
def moles_CO2 : ℕ := 2
def moles_O2 : ℕ := 2 -- represent excess O2, irrelevant to limiting reagent
def mag_ox_reaction (mg : ℕ) (o2 : ℕ) (mgo : ℕ) : Prop := 2 * mg + o2 = 2 * mgo
def mag_carbon_reaction (mg : ℕ) (co2 : ℕ) (mgco3 : ℕ) : Prop := mg + co2 = mgco3

-- Assume Magnesium is the limiting reagent for both reactions
theorem magnesium_is_limiting (mgo : ℕ) (mgco3 : ℕ) :
  mag_ox_reaction moles_Mg moles_O2 mgo ∧ mag_carbon_reaction moles_Mg moles_CO2 mgco3 →
  mgo = 4 ∧ mgco3 = 4 :=
by
  sorry

end magnesium_is_limiting_l2162_216222


namespace floor_length_l2162_216237

variable (b l : ℝ)

theorem floor_length :
  (l = 3 * b) →
  (3 * b ^ 2 = 128) →
  l = 19.59 :=
by
  intros h1 h2
  sorry

end floor_length_l2162_216237


namespace least_n_for_reducible_fraction_l2162_216293

theorem least_n_for_reducible_fraction :
  ∃ n : ℕ, 0 < n ∧ (∃ k : ℤ, n - 13 = 71 * k) ∧ n = 84 := by
  sorry

end least_n_for_reducible_fraction_l2162_216293


namespace clock_displays_unique_digits_minutes_l2162_216267

def minutes_with_unique_digits (h1 h2 m1 m2 : ℕ) : Prop :=
  h1 ≠ h2 ∧ h1 ≠ m1 ∧ h1 ≠ m2 ∧ h2 ≠ m1 ∧ h2 ≠ m2 ∧ m1 ≠ m2

def count_unique_digit_minutes (total_minutes : ℕ) :=
  let range0_19 := 1200
  let valid_0_19 := 504
  let range20_23 := 240
  let valid_20_23 := 84
  valid_0_19 + valid_20_23 = total_minutes

theorem clock_displays_unique_digits_minutes :
  count_unique_digit_minutes 588 :=
  by
    sorry

end clock_displays_unique_digits_minutes_l2162_216267


namespace solve_for_x_l2162_216203

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end solve_for_x_l2162_216203


namespace proposition_false_at_4_l2162_216225

theorem proposition_false_at_4 (P : ℕ → Prop) (hp : ∀ k : ℕ, k > 0 → (P k → P (k + 1))) (h4 : ¬ P 5) : ¬ P 4 :=
by {
    sorry
}

end proposition_false_at_4_l2162_216225


namespace balls_per_pack_l2162_216220

theorem balls_per_pack (total_packs total_cost cost_per_ball total_balls balls_per_pack : ℕ)
  (h1 : total_packs = 4)
  (h2 : total_cost = 24)
  (h3 : cost_per_ball = 2)
  (h4 : total_balls = total_cost / cost_per_ball)
  (h5 : total_balls = 12)
  (h6 : balls_per_pack = total_balls / total_packs) :
  balls_per_pack = 3 := by 
  sorry

end balls_per_pack_l2162_216220


namespace boys_amount_per_person_l2162_216290

theorem boys_amount_per_person (total_money : ℕ) (total_children : ℕ) (per_girl : ℕ) (number_of_boys : ℕ) (amount_per_boy : ℕ) : 
  total_money = 460 ∧
  total_children = 41 ∧
  per_girl = 8 ∧
  number_of_boys = 33 → 
  amount_per_boy = 12 :=
by sorry

end boys_amount_per_person_l2162_216290


namespace triangular_25_eq_325_l2162_216239

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem triangular_25_eq_325 : triangular_number 25 = 325 :=
by
  -- proof would go here
  sorry

end triangular_25_eq_325_l2162_216239


namespace multiplier_eq_l2162_216262

-- Definitions of the given conditions
def length (w : ℝ) (m : ℝ) : ℝ := m * w + 2
def perimeter (l : ℝ) (w : ℝ) : ℝ := 2 * l + 2 * w

-- Condition definitions
def l : ℝ := 38
def P : ℝ := 100

-- Proof statement
theorem multiplier_eq (m w : ℝ) (h1 : length w m = l) (h2 : perimeter l w = P) : m = 3 :=
by
  sorry

end multiplier_eq_l2162_216262


namespace total_cost_correct_l2162_216265

-- Define the parameters
variables (a : ℕ) -- the number of books
-- Define the constants and the conditions
def unit_price : ℝ := 8
def shipping_fee_percentage : ℝ := 0.10

-- Define the total cost including the shipping fee
def total_cost (a : ℕ) : ℝ := unit_price * (1 + shipping_fee_percentage) * a

-- Prove that the total cost is equal to the expected amount
theorem total_cost_correct : total_cost a = 8 * (1 + 0.10) * a := by
  sorry

end total_cost_correct_l2162_216265


namespace probability_at_tree_correct_expected_distance_correct_l2162_216282

-- Define the initial conditions
def initial_tree (n : ℕ) : ℕ := n + 1
def total_trees (n : ℕ) : ℕ := 2 * n + 1

-- Define the probability that the drunkard is at each tree T_i (1 <= i <= 2n+1) at the end of the nth minute
def probability_at_tree (n i : ℕ) : ℚ :=
  if 1 ≤ i ∧ i ≤ total_trees n then
    (Nat.choose (2*n) (i-1)) / (2^(2*n))
  else
    0

-- Define the expected distance between the final position and the initial tree T_{n+1}
def expected_distance (n : ℕ) : ℚ :=
  n * (Nat.choose (2*n) n) / (2^(2*n))

-- Statements to prove
theorem probability_at_tree_correct (n i : ℕ) (hi : 1 ≤ i ∧ i ≤ total_trees n)  :
  probability_at_tree n i = (Nat.choose (2*n) (i-1)) / (2^(2*n)) :=
by
  sorry

theorem expected_distance_correct (n : ℕ) :
  expected_distance n = n * (Nat.choose (2*n) n) / (2^(2*n)) :=
by
  sorry

end probability_at_tree_correct_expected_distance_correct_l2162_216282


namespace titu_andreescu_inequality_l2162_216221

theorem titu_andreescu_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 :=
sorry

end titu_andreescu_inequality_l2162_216221


namespace temperature_problem_l2162_216285

theorem temperature_problem (N : ℤ) (M L : ℤ) :
  M = L + N →
  (M - 10) - (L + 6) = 4 ∨ (M - 10) - (L + 6) = -4 →
  (N - 16 = 4 ∨ 16 - N = 4) →
  ((N = 20 ∨ N = 12) → 20 * 12 = 240) :=
by
   sorry

end temperature_problem_l2162_216285


namespace correct_statements_l2162_216205

-- Define the function and the given conditions
def f : ℝ → ℝ := sorry

lemma not_constant (h: ∃ x y: ℝ, x ≠ y ∧ f x ≠ f y) : true := sorry
lemma periodic (x : ℝ) : f (x - 1) = f (x + 1) := sorry
lemma symmetric (x : ℝ) : f (2 - x) = f x := sorry

-- The statements we want to prove
theorem correct_statements : 
  (∀ x, f (-x) = f x) ∧ 
  (∀ x, f (1 - x) = f (1 + x)) ∧ 
  (∃ T > 0, ∀ x, f (x + T) = f x)
:= by
  sorry

end correct_statements_l2162_216205


namespace smallest_positive_integer_l2162_216264

theorem smallest_positive_integer (
  a : ℕ
) : 
  (a ≡ 5 [MOD 6]) ∧ (a ≡ 7 [MOD 8]) → a = 23 :=
by sorry

end smallest_positive_integer_l2162_216264


namespace sum_of_A_B_C_l2162_216214

theorem sum_of_A_B_C (A B C : ℕ) (h_pos_A : 0 < A) (h_pos_B : 0 < B) (h_pos_C : 0 < C) (h_rel_prime : Nat.gcd A (Nat.gcd B C) = 1) (h_eq : A * Real.log 3 / Real.log 180 + B * Real.log 5 / Real.log 180 = C) : A + B + C = 4 :=
sorry

end sum_of_A_B_C_l2162_216214


namespace smallest_n_for_factors_l2162_216270

theorem smallest_n_for_factors (k : ℕ) (hk : (∃ p : ℕ, k = 2^p) ) :
  ∃ (n : ℕ), ( 5^2 ∣ n * k * 36 * 343 ) ∧ ( 3^3 ∣ n * k * 36 * 343 ) ∧ n = 75 :=
by
  sorry

end smallest_n_for_factors_l2162_216270


namespace x_plus_inv_x_eq_two_implies_x_pow_six_eq_one_l2162_216298

theorem x_plus_inv_x_eq_two_implies_x_pow_six_eq_one
  (x : ℝ) (h : x + 1/x = 2) : x^6 = 1 :=
sorry

end x_plus_inv_x_eq_two_implies_x_pow_six_eq_one_l2162_216298


namespace parabola_focus_l2162_216275

theorem parabola_focus (y x : ℝ) (h : y^2 = 4 * x) : x = 1 → y = 0 → (1, 0) = (1, 0) :=
by 
  sorry

end parabola_focus_l2162_216275


namespace average_of_first_n_multiples_of_8_is_88_l2162_216251

theorem average_of_first_n_multiples_of_8_is_88 (n : ℕ) (h : (n / 2) * (8 + 8 * n) / n = 88) : n = 21 :=
sorry

end average_of_first_n_multiples_of_8_is_88_l2162_216251


namespace sandy_initial_carrots_l2162_216252

-- Defining the conditions
def sam_took : ℕ := 3
def sandy_left : ℕ := 3

-- The statement to be proven
theorem sandy_initial_carrots :
  (sandy_left + sam_took = 6) :=
by
  sorry

end sandy_initial_carrots_l2162_216252


namespace students_distribution_l2162_216260

theorem students_distribution (students villages : ℕ) (h_students : students = 4) (h_villages : villages = 3) :
  ∃ schemes : ℕ, schemes = 36 := 
sorry

end students_distribution_l2162_216260


namespace range_of_a_l2162_216233

-- Define sets P and M
def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}
def M (a : ℝ) : Set ℝ := {x | (2 - a) ≤ x ∧ x ≤ (1 + a)}

-- Prove the range of a
theorem range_of_a (a : ℝ) : (P ∩ (M a) = P) ↔ (a ≥ 1) :=
by 
  sorry

end range_of_a_l2162_216233


namespace ratio_of_friday_to_thursday_l2162_216247

theorem ratio_of_friday_to_thursday
  (wednesday_copies : ℕ)
  (total_copies : ℕ)
  (ratio : ℚ)
  (h1 : wednesday_copies = 15)
  (h2 : total_copies = 69)
  (h3 : ratio = 1 / 5) :
  (total_copies - wednesday_copies - 3 * wednesday_copies) / (3 * wednesday_copies) = ratio :=
by
  -- proof goes here
  sorry

end ratio_of_friday_to_thursday_l2162_216247


namespace polynomial_div_6_l2162_216243

theorem polynomial_div_6 (n : ℕ) : 6 ∣ (2 * n ^ 3 + 9 * n ^ 2 + 13 * n) := 
sorry

end polynomial_div_6_l2162_216243


namespace multiple_of_four_and_six_prime_sum_even_l2162_216263

theorem multiple_of_four_and_six_prime_sum_even {a b : ℤ} 
  (h_a : ∃ m : ℤ, a = 4 * m) 
  (h_b1 : ∃ n : ℤ, b = 6 * n) 
  (h_b2 : Prime b) : 
  Even (a + b) := 
  by sorry

end multiple_of_four_and_six_prime_sum_even_l2162_216263


namespace find_starting_number_l2162_216254

theorem find_starting_number :
  ∃ startnum : ℕ, startnum % 5 = 0 ∧ (∀ k : ℕ, 0 ≤ k ∧ k < 20 → startnum + 5 * k ≤ 100) ∧ startnum = 10 :=
sorry

end find_starting_number_l2162_216254


namespace normal_price_of_article_l2162_216279

theorem normal_price_of_article (P : ℝ) (sale_price : ℝ) (discount1 discount2 : ℝ) 
  (h1 : discount1 = 0.10) 
  (h2 : discount2 = 0.20) 
  (h3 : sale_price = 72) 
  (h4 : sale_price = (P * (1 - discount1)) * (1 - discount2)) : 
  P = 100 :=
by 
  sorry

end normal_price_of_article_l2162_216279


namespace smallest_five_digit_divisible_by_2_3_8_9_l2162_216212

-- Definitions for the conditions given in the problem
def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000
def divisible_by (n d : ℕ) : Prop := d ∣ n

-- The main theorem stating the problem
theorem smallest_five_digit_divisible_by_2_3_8_9 :
  ∃ n : ℕ, is_five_digit n ∧ divisible_by n 2 ∧ divisible_by n 3 ∧ divisible_by n 8 ∧ divisible_by n 9 ∧ n = 10008 :=
sorry

end smallest_five_digit_divisible_by_2_3_8_9_l2162_216212


namespace tangent_series_identity_l2162_216248

noncomputable def series_tangent (x : ℝ) : ℝ := ∑' n, (1 / (2 ^ n)) * Real.tan (x / (2 ^ n))

theorem tangent_series_identity (x : ℝ) : 
  (1 / x) - (1 / Real.tan x) = series_tangent x := 
sorry

end tangent_series_identity_l2162_216248


namespace three_digit_numbers_m_l2162_216296

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem three_digit_numbers_m (n : ℕ) :
  100 ≤ n ∧ n ≤ 999 ∧ sum_of_digits n = 12 ∧ 100 ≤ 2 * n ∧ 2 * n ≤ 999 ∧ sum_of_digits (2 * n) = 6 → ∃! (m : ℕ), n = m :=
sorry

end three_digit_numbers_m_l2162_216296


namespace prop_A_prop_B_prop_C_prop_D_l2162_216273

-- Proposition A: For all x ∈ ℝ, x² - x + 1 > 0
theorem prop_A (x : ℝ) : x^2 - x + 1 > 0 :=
sorry

-- Proposition B: a² + a = 0 is not a sufficient and necessary condition for a = 0
theorem prop_B : ¬(∀ a : ℝ, (a^2 + a = 0 ↔ a = 0)) :=
sorry

-- Proposition C: a > 1 and b > 1 is a sufficient and necessary condition for a + b > 2 and ab > 1
theorem prop_C (a b : ℝ) : (a > 1 ∧ b > 1) ↔ (a + b > 2 ∧ a * b > 1) :=
sorry

-- Proposition D: a > 4 is a necessary and sufficient condition for the roots of the equation x² - ax + a = 0 to be all positive
theorem prop_D (a : ℝ) : (a > 4) ↔ (∀ x : ℝ, x ≠ 0 → (x^2 - a*x + a = 0 → x > 0)) :=
sorry

end prop_A_prop_B_prop_C_prop_D_l2162_216273


namespace yoongi_number_division_l2162_216228

theorem yoongi_number_division (n : ℕ) (h : n / 4 = 12) : n / 3 = 16 :=
by
  sorry

end yoongi_number_division_l2162_216228


namespace find_nm_l2162_216280

theorem find_nm (h : 62^2 + 122^2 = 18728) : 
  ∃ (n m : ℕ), (n = 92 ∧ m = 30) ∨ (n = 30 ∧ m = 92) ∧ n^2 + m^2 = 9364 := 
by 
  sorry

end find_nm_l2162_216280


namespace probability_two_cards_l2162_216250

noncomputable def probability_first_spade_second_ace : ℚ :=
  let total_cards := 52
  let total_spades := 13
  let total_aces := 4
  let remaining_cards := total_cards - 1
  
  let first_spade_non_ace := (total_spades - 1) / total_cards
  let second_ace_after_non_ace := total_aces / remaining_cards
  
  let probability_case1 := first_spade_non_ace * second_ace_after_non_ace
  
  let first_ace_spade := 1 / total_cards
  let second_ace_after_ace := (total_aces - 1) / remaining_cards
  
  let probability_case2 := first_ace_spade * second_ace_after_ace
  
  probability_case1 + probability_case2

theorem probability_two_cards {p : ℚ} (h : p = 1 / 52) : 
  probability_first_spade_second_ace = p := 
by 
  simp only [probability_first_spade_second_ace]
  sorry

end probability_two_cards_l2162_216250


namespace arrange_moon_l2162_216281

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def ways_to_arrange_moon : ℕ :=
  let total_letters := 4
  let repeated_O_count := 2
  factorial total_letters / factorial repeated_O_count

theorem arrange_moon : ways_to_arrange_moon = 12 := 
by {
  sorry -- Proof is omitted as instructed
}

end arrange_moon_l2162_216281


namespace convert_speed_to_mps_l2162_216276

-- Define given speeds and conversion factors
def speed_kmph : ℝ := 63
def kilometers_to_meters : ℝ := 1000
def hours_to_seconds : ℝ := 3600

-- Assert the conversion
theorem convert_speed_to_mps : speed_kmph * (kilometers_to_meters / hours_to_seconds) = 17.5 := by
  sorry

end convert_speed_to_mps_l2162_216276


namespace find_n_tangent_eq_1234_l2162_216201

theorem find_n_tangent_eq_1234 (n : ℤ) (h1 : -90 < n ∧ n < 90) (h2 : Real.tan (n * Real.pi / 180) = Real.tan (1234 * Real.pi / 180)) : n = -26 := 
by 
  sorry

end find_n_tangent_eq_1234_l2162_216201


namespace min_cost_for_boxes_l2162_216241

theorem min_cost_for_boxes
  (box_length: ℕ) (box_width: ℕ) (box_height: ℕ)
  (cost_per_box: ℝ) (total_volume: ℝ)
  (h1 : box_length = 20)
  (h2 : box_width = 20)
  (h3 : box_height = 15)
  (h4 : cost_per_box = 1.30)
  (h5 : total_volume = 3060000) :
  ∃ cost: ℝ, cost = 663 :=
by
  sorry

end min_cost_for_boxes_l2162_216241


namespace solve_for_p_l2162_216266

-- Conditions
def C1 (n : ℕ) : Prop := (3 : ℚ) / 4 = n / 48
def C2 (m n : ℕ) : Prop := (3 : ℚ) / 4 = (m + n) / 96
def C3 (p m : ℕ) : Prop := (3 : ℚ) / 4 = (p - m) / 160

-- Theorem to prove
theorem solve_for_p (n m p : ℕ) (h1 : C1 n) (h2 : C2 m n) (h3 : C3 p m) : p = 156 := 
by 
    sorry

end solve_for_p_l2162_216266


namespace find_angle_measure_l2162_216217

def complement_more_condition (x : ℝ) : Prop :=
  90 - x = (1 / 7) * x + 26

theorem find_angle_measure (x : ℝ) (h : complement_more_condition x) : x = 56 :=
sorry

end find_angle_measure_l2162_216217


namespace maximize_annual_profit_l2162_216284

theorem maximize_annual_profit : 
  ∃ n : ℕ, n ≠ 0 ∧ (∀ m : ℕ, m ≠ 0 → (110 * n - (n * n + n) - 90) / n ≥ (110 * m - (m * m + m) - 90) / m) ↔ n = 5 := 
by 
  -- Proof steps would go here
  sorry

end maximize_annual_profit_l2162_216284


namespace charles_whistles_l2162_216295

theorem charles_whistles (S : ℕ) (C : ℕ) (h1 : S = 223) (h2 : S = C + 95) : C = 128 :=
by
  sorry

end charles_whistles_l2162_216295


namespace sum_of_solutions_of_quadratic_l2162_216288

theorem sum_of_solutions_of_quadratic :
  let a := 18
  let b := -27
  let c := -45
  let roots_sum := (-b / a : ℚ)
  roots_sum = 3 / 2 :=
by
  let a := 18
  let b := -27
  let c := -45
  let roots_sum := (-b / a : ℚ)
  have h1 : roots_sum = 3 / 2 := by sorry
  exact h1

end sum_of_solutions_of_quadratic_l2162_216288


namespace gray_eyed_black_haired_students_l2162_216219

theorem gray_eyed_black_haired_students (total_students : ℕ) 
  (green_eyed_red_haired : ℕ) (black_haired : ℕ) (gray_eyed : ℕ) 
  (h_total : total_students = 50)
  (h_green_eyed_red_haired : green_eyed_red_haired = 17)
  (h_black_haired : black_haired = 27)
  (h_gray_eyed : gray_eyed = 23) :
  ∃ (gray_eyed_black_haired : ℕ), gray_eyed_black_haired = 17 :=
by sorry

end gray_eyed_black_haired_students_l2162_216219


namespace perpendicular_bisector_eq_l2162_216269

-- Define the circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 6 * y = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 6 * x = 0

-- Prove that the perpendicular bisector of line segment AB has the equation 3x - y - 9 = 0
theorem perpendicular_bisector_eq :
  (∀ x y : ℝ, C1 x y → C2 x y → 3 * x - y - 9 = 0) :=
by
  sorry

end perpendicular_bisector_eq_l2162_216269
