import Mathlib

namespace NUMINAMATH_GPT_solve_equation_l2175_217597

theorem solve_equation :
  ∃! (x y z : ℝ), 2 * x^4 + 2 * y^4 - 4 * x^3 * y + 6 * x^2 * y^2 - 4 * x * y^3 + 7 * y^2 + 7 * z^2 - 14 * y * z - 70 * y + 70 * z + 175 = 0 ∧ x = 0 ∧ y = 0 ∧ z = -5 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l2175_217597


namespace NUMINAMATH_GPT_intersection_points_with_x_axis_l2175_217546

theorem intersection_points_with_x_axis (a : ℝ) :
    (∃ x : ℝ, a * x^2 - a * x + 3 * x + 1 = 0 ∧ 
              ∀ x' : ℝ, (x' ≠ x → a * x'^2 - a * x' + 3 * x' + 1 ≠ 0)) ↔ 
    (a = 0 ∨ a = 1 ∨ a = 9) := by 
  sorry

end NUMINAMATH_GPT_intersection_points_with_x_axis_l2175_217546


namespace NUMINAMATH_GPT_count_divisible_digits_l2175_217512

def is_divisible (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

theorem count_divisible_digits :
  ∃! (s : Finset ℕ), s = {n | n ∈ Finset.range 10 ∧ n ≠ 0 ∧ is_divisible (25 * n) n} ∧ (Finset.card s = 3) := 
by
  sorry

end NUMINAMATH_GPT_count_divisible_digits_l2175_217512


namespace NUMINAMATH_GPT_marvin_substitute_correct_l2175_217572

theorem marvin_substitute_correct {a b c d f : ℤ} (ha : a = 3) (hb : b = 4) (hc : c = 7) (hd : d = 5) :
  (a + (b - (c + (d - f))) = 5 - f) → f = 5 :=
sorry

end NUMINAMATH_GPT_marvin_substitute_correct_l2175_217572


namespace NUMINAMATH_GPT_sin_eq_one_fifth_l2175_217571

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem sin_eq_one_fifth (ϕ : ℝ)
  (h : binomial_coefficient 5 3 * (Real.cos ϕ)^2 = 4) :
  Real.sin (2 * ϕ - π / 2) = 1 / 5 := sorry

end NUMINAMATH_GPT_sin_eq_one_fifth_l2175_217571


namespace NUMINAMATH_GPT_slope_of_tangent_at_minus_1_l2175_217588

theorem slope_of_tangent_at_minus_1
  (c : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = (x - 2) * (x^2 + c))
  (h_extremum : deriv f 1 = 0) :
  deriv f (-1) = 8 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_tangent_at_minus_1_l2175_217588


namespace NUMINAMATH_GPT_hillary_descending_rate_correct_l2175_217507

-- Define the conditions in Lean
def base_to_summit := 5000 -- height from base camp to the summit
def departure_time := 6 -- departure time in hours after midnight (6:00)
def summit_time_hillary := 5 -- time taken by Hillary to reach 1000 ft short of the summit
def passing_time := 12 -- time when Hillary and Eddy pass each other (12:00)
def climb_rate_hillary := 800 -- Hillary's climbing rate in ft/hr
def climb_rate_eddy := 500 -- Eddy's climbing rate in ft/hr
def stop_short := 1000 -- distance short of the summit Hillary stops at

-- Define the correct answer based on the conditions
def descending_rate_hillary := 1000 -- Hillary's descending rate in ft/hr

-- Create the theorem to prove Hillary's descending rate
theorem hillary_descending_rate_correct (base_to_summit departure_time summit_time_hillary passing_time climb_rate_hillary climb_rate_eddy stop_short descending_rate_hillary : ℕ) :
  (descending_rate_hillary = 1000) :=
sorry

end NUMINAMATH_GPT_hillary_descending_rate_correct_l2175_217507


namespace NUMINAMATH_GPT_cos_value_given_sin_l2175_217564

theorem cos_value_given_sin (α : ℝ) (h : Real.sin (α + π / 6) = 2 * Real.sqrt 5 / 5) :
  Real.cos (π / 3 - α) = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_GPT_cos_value_given_sin_l2175_217564


namespace NUMINAMATH_GPT_number_of_months_in_season_l2175_217511

def games_per_month : ℝ := 323.0
def total_games : ℝ := 5491.0

theorem number_of_months_in_season : total_games / games_per_month = 17 := 
by
  sorry

end NUMINAMATH_GPT_number_of_months_in_season_l2175_217511


namespace NUMINAMATH_GPT_initial_markup_percentage_l2175_217522

theorem initial_markup_percentage (C : ℝ) (M : ℝ) 
  (h1 : ∀ S_1 : ℝ, S_1 = C * (1 + M))
  (h2 : ∀ S_2 : ℝ, S_2 = C * (1 + M) * 1.25)
  (h3 : ∀ S_3 : ℝ, S_3 = C * (1 + M) * 1.25 * 0.94)
  (h4 : ∀ S_3 : ℝ, S_3 = C * 1.41) : 
  M = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_initial_markup_percentage_l2175_217522


namespace NUMINAMATH_GPT_circle_line_intersection_zero_l2175_217525

theorem circle_line_intersection_zero (x_0 y_0 r : ℝ) (hP : x_0^2 + y_0^2 < r^2) :
  ∀ (x y : ℝ), (x^2 + y^2 = r^2) → (x_0 * x + y_0 * y = r^2) → false :=
by
  sorry

end NUMINAMATH_GPT_circle_line_intersection_zero_l2175_217525


namespace NUMINAMATH_GPT_johns_salary_before_raise_l2175_217517

variable (x : ℝ)

theorem johns_salary_before_raise (h : x + 0.3333 * x = 80) : x = 60 :=
by
  sorry

end NUMINAMATH_GPT_johns_salary_before_raise_l2175_217517


namespace NUMINAMATH_GPT_mark_reading_time_l2175_217515

variable (x y : ℕ)

theorem mark_reading_time (x y : ℕ) : 
  7 * x + y = 7 * x + y :=
by
  sorry

end NUMINAMATH_GPT_mark_reading_time_l2175_217515


namespace NUMINAMATH_GPT_sum_of_squares_of_sums_l2175_217536

axiom roots_of_polynomial (p q r : ℝ) : p^3 - 15*p^2 + 25*p - 12 = 0 ∧ q^3 - 15*q^2 + 25*q - 12 = 0 ∧ r^3 - 15*r^2 + 25*r - 12 = 0

theorem sum_of_squares_of_sums (p q r : ℝ)
  (h_roots : p^3 - 15*p^2 + 25*p - 12 = 0 ∧ q^3 - 15*q^2 + 25*q - 12 = 0 ∧ r^3 - 15*r^2 + 25*r - 12 = 0) :
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 := 
sorry

end NUMINAMATH_GPT_sum_of_squares_of_sums_l2175_217536


namespace NUMINAMATH_GPT_indoor_players_count_l2175_217532

theorem indoor_players_count (T O B I : ℕ) 
  (hT : T = 400) 
  (hO : O = 350) 
  (hB : B = 60) 
  (hEq : T = (O - B) + (I - B) + B) : 
  I = 110 := 
by sorry

end NUMINAMATH_GPT_indoor_players_count_l2175_217532


namespace NUMINAMATH_GPT_kirill_is_62_5_l2175_217538

variable (K : ℝ)

def kirill_height := K
def brother_height := K + 14
def sister_height := 2 * K
def total_height := K + (K + 14) + 2 * K

theorem kirill_is_62_5 (h1 : total_height K = 264) : K = 62.5 := by
  sorry

end NUMINAMATH_GPT_kirill_is_62_5_l2175_217538


namespace NUMINAMATH_GPT_moles_of_H2_required_l2175_217594

theorem moles_of_H2_required 
  (moles_C : ℕ) 
  (moles_O2 : ℕ) 
  (moles_CH4 : ℕ) 
  (moles_CO2 : ℕ) 
  (balanced_reaction_1 : ℕ → ℕ → ℕ → Prop)
  (balanced_reaction_2 : ℕ → ℕ → ℕ → ℕ → Prop)
  (H_balanced : balanced_reaction_2 2 4 2 1)
  (H_form_CO2 : balanced_reaction_1 1 1 1) :
  moles_C = 2 ∧ moles_O2 = 1 ∧ moles_CH4 = 2 ∧ moles_CO2 = 1 → (∃ moles_H2, moles_H2 = 4) :=
by sorry

end NUMINAMATH_GPT_moles_of_H2_required_l2175_217594


namespace NUMINAMATH_GPT_eggs_left_l2175_217591

def initial_eggs := 20
def mother_used := 5
def father_used := 3
def chicken1_laid := 4
def chicken2_laid := 3
def chicken3_laid := 2
def oldest_took := 2

theorem eggs_left :
  initial_eggs - (mother_used + father_used) + (chicken1_laid + chicken2_laid + chicken3_laid) - oldest_took = 19 := 
by
  sorry

end NUMINAMATH_GPT_eggs_left_l2175_217591


namespace NUMINAMATH_GPT_percentage_increase_is_20_l2175_217518

def number_of_students_this_year : ℕ := 960
def number_of_students_last_year : ℕ := 800

theorem percentage_increase_is_20 :
  ((number_of_students_this_year - number_of_students_last_year : ℕ) / number_of_students_last_year * 100) = 20 := 
by
  sorry

end NUMINAMATH_GPT_percentage_increase_is_20_l2175_217518


namespace NUMINAMATH_GPT_final_balance_l2175_217559

noncomputable def initial_balance : ℕ := 10
noncomputable def charity_donation : ℕ := 4
noncomputable def prize_amount : ℕ := 90
noncomputable def lost_at_first_slot : ℕ := 50
noncomputable def lost_at_second_slot : ℕ := 10
noncomputable def lost_at_last_slot : ℕ := 5
noncomputable def cost_of_water : ℕ := 1
noncomputable def cost_of_lottery_ticket : ℕ := 1
noncomputable def lottery_win : ℕ := 65

theorem final_balance : 
  initial_balance - charity_donation + prize_amount - (lost_at_first_slot + lost_at_second_slot + lost_at_last_slot) - (cost_of_water + cost_of_lottery_ticket) + lottery_win = 94 := 
by 
  -- This is the lean statement, the proof is not required as per instructions.
  sorry

end NUMINAMATH_GPT_final_balance_l2175_217559


namespace NUMINAMATH_GPT_no_integer_solutions_l2175_217505

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ),
    x^6 + x^3 + x^3 * y + y = 147^157 ∧
    x^3 + x^3 * y + y^2 + y + z^9 = 157^147 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l2175_217505


namespace NUMINAMATH_GPT_number_of_classes_l2175_217540

variable (s : ℕ) (h_s : s > 0)
-- Define the conditions
def student_books_year : ℕ := 4 * 12
def total_books_read : ℕ := 48
def class_books_year (s : ℕ) : ℕ := s * student_books_year
def total_classes (c s : ℕ) (h_s : s > 0) : ℕ := 1

-- Define the main theorem
theorem number_of_classes (h : total_books_read = 48) (h_s : s > 0)
  (h1 : c * class_books_year s = 48) : c = 1 := by
  sorry

end NUMINAMATH_GPT_number_of_classes_l2175_217540


namespace NUMINAMATH_GPT_least_positive_integer_l2175_217595

theorem least_positive_integer (n : ℕ) (h1 : n > 1)
  (h2 : n % 3 = 2) (h3 : n % 4 = 2) (h4 : n % 5 = 2) (h5 : n % 11 = 2) :
  n = 662 :=
sorry

end NUMINAMATH_GPT_least_positive_integer_l2175_217595


namespace NUMINAMATH_GPT_largest_multiple_of_6_neg_greater_than_neg_150_l2175_217502

theorem largest_multiple_of_6_neg_greater_than_neg_150 : 
  ∃ m : ℤ, m % 6 = 0 ∧ -m > -150 ∧ m = 144 :=
by
  sorry

end NUMINAMATH_GPT_largest_multiple_of_6_neg_greater_than_neg_150_l2175_217502


namespace NUMINAMATH_GPT_monica_total_savings_l2175_217584

noncomputable def weekly_savings : ℕ := 15
noncomputable def weeks_to_fill_moneybox : ℕ := 60
noncomputable def num_repeats : ℕ := 5
noncomputable def total_savings (weekly_savings weeks_to_fill_moneybox num_repeats : ℕ) : ℕ :=
  (weekly_savings * weeks_to_fill_moneybox) * num_repeats

theorem monica_total_savings :
  total_savings 15 60 5 = 4500 := by
  sorry

end NUMINAMATH_GPT_monica_total_savings_l2175_217584


namespace NUMINAMATH_GPT_g_2002_equals_1_l2175_217516

theorem g_2002_equals_1 (f : ℝ → ℝ)
  (hf1 : f 1 = 1)
  (hf2 : ∀ x : ℝ, f (x + 5) ≥ f x + 5)
  (hf3 : ∀ x : ℝ, f (x + 1) ≤ f x + 1)
  (g : ℝ → ℝ := fun x => f x + 1 - x)
  : g 2002 = 1 :=
by
  sorry

end NUMINAMATH_GPT_g_2002_equals_1_l2175_217516


namespace NUMINAMATH_GPT_original_cost_of_pencil_l2175_217524

theorem original_cost_of_pencil (final_price discount: ℝ) (h_final: final_price = 3.37) (h_disc: discount = 0.63) : 
  final_price + discount = 4 :=
by
  sorry

end NUMINAMATH_GPT_original_cost_of_pencil_l2175_217524


namespace NUMINAMATH_GPT_average_age_first_and_fifth_dogs_l2175_217534

-- Define the conditions
def first_dog_age : ℕ := 10
def second_dog_age : ℕ := first_dog_age - 2
def third_dog_age : ℕ := second_dog_age + 4
def fourth_dog_age : ℕ := third_dog_age / 2
def fifth_dog_age : ℕ := fourth_dog_age + 20

-- Define the goal statement
theorem average_age_first_and_fifth_dogs : 
  (first_dog_age + fifth_dog_age) / 2 = 18 :=
by
  sorry

end NUMINAMATH_GPT_average_age_first_and_fifth_dogs_l2175_217534


namespace NUMINAMATH_GPT_gcd_min_value_l2175_217560

-- Definitions of the conditions
def is_positive_integer (x : ℕ) := x > 0

def gcd_cond (m n : ℕ) := Nat.gcd m n = 18

-- The main theorem statement
theorem gcd_min_value (m n : ℕ) (hm : is_positive_integer m) (hn : is_positive_integer n) (hgcd : gcd_cond m n) : 
  Nat.gcd (12 * m) (20 * n) = 72 :=
sorry

end NUMINAMATH_GPT_gcd_min_value_l2175_217560


namespace NUMINAMATH_GPT_rhind_papyrus_smallest_portion_l2175_217547

theorem rhind_papyrus_smallest_portion :
  ∀ (a1 d : ℚ),
    5 * a1 + (5 * 4 / 2) * d = 10 ∧
    (3 * a1 + 9 * d) / 7 = a1 + (a1 + d) →
    a1 = 1 / 6 :=
by sorry

end NUMINAMATH_GPT_rhind_papyrus_smallest_portion_l2175_217547


namespace NUMINAMATH_GPT_milk_in_jugs_l2175_217577

theorem milk_in_jugs (x y : ℝ) (h1 : x + y = 70) (h2 : y + 0.125 * x = 0.875 * x) :
  x = 40 ∧ y = 30 := 
sorry

end NUMINAMATH_GPT_milk_in_jugs_l2175_217577


namespace NUMINAMATH_GPT_cubes_difference_divisible_91_l2175_217550

theorem cubes_difference_divisible_91 (cubes : Fin 16 → ℤ) (h : ∀ n : Fin 16, ∃ m : ℤ, cubes n = m^3) :
  ∃ (a b : Fin 16), a ≠ b ∧ 91 ∣ (cubes a - cubes b) :=
sorry

end NUMINAMATH_GPT_cubes_difference_divisible_91_l2175_217550


namespace NUMINAMATH_GPT_peacocks_in_zoo_l2175_217582

theorem peacocks_in_zoo :
  ∃ p t : ℕ, 2 * p + 4 * t = 54 ∧ p + t = 17 ∧ p = 7 :=
by
  sorry

end NUMINAMATH_GPT_peacocks_in_zoo_l2175_217582


namespace NUMINAMATH_GPT_distance_earth_sun_l2175_217567

theorem distance_earth_sun (speed_of_light : ℝ) (time_to_earth: ℝ) 
(h1 : speed_of_light = 3 * 10^8) 
(h2 : time_to_earth = 5 * 10^2) :
  speed_of_light * time_to_earth = 1.5 * 10^11 := 
by 
  -- proof steps can be filled here
  sorry

end NUMINAMATH_GPT_distance_earth_sun_l2175_217567


namespace NUMINAMATH_GPT_subset_of_difference_empty_l2175_217581

theorem subset_of_difference_empty {α : Type*} (A B : Set α) :
  (A \ B = ∅) → (A ⊆ B) :=
by
  sorry

end NUMINAMATH_GPT_subset_of_difference_empty_l2175_217581


namespace NUMINAMATH_GPT_total_situps_l2175_217562

def situps (b c j : ℕ) : ℕ := b * 1 + c * 2 + j * 3

theorem total_situps :
  ∀ (b c j : ℕ),
    b = 45 →
    c = 2 * b →
    j = c + 5 →
    situps b c j = 510 :=
by intros b c j hb hc hj
   sorry

end NUMINAMATH_GPT_total_situps_l2175_217562


namespace NUMINAMATH_GPT_paint_two_faces_red_l2175_217521

theorem paint_two_faces_red (f : Fin 8 → ℕ) (H : ∀ i, 1 ≤ f i ∧ f i ≤ 8) : 
  (∃ pair_count : ℕ, pair_count = 9 ∧
    ∀ i j, i < j → f i + f j ≤ 7 → true) :=
sorry

end NUMINAMATH_GPT_paint_two_faces_red_l2175_217521


namespace NUMINAMATH_GPT_largest_integer_x_l2175_217583

theorem largest_integer_x (x : ℤ) : (x / 4 + 3 / 5 < 7 / 4) → x ≤ 4 := sorry

end NUMINAMATH_GPT_largest_integer_x_l2175_217583


namespace NUMINAMATH_GPT_albums_either_but_not_both_l2175_217537

-- Definition of the problem conditions
def shared_albums : Nat := 11
def andrew_total_albums : Nat := 20
def bob_exclusive_albums : Nat := 8

-- Calculate Andrew's exclusive albums
def andrew_exclusive_albums : Nat := andrew_total_albums - shared_albums

-- Question: Prove the total number of albums in either Andrew's or Bob's collection but not both is 17
theorem albums_either_but_not_both : 
  andrew_exclusive_albums + bob_exclusive_albums = 17 := 
by
  sorry

end NUMINAMATH_GPT_albums_either_but_not_both_l2175_217537


namespace NUMINAMATH_GPT_river_flow_speed_l2175_217513

theorem river_flow_speed (v : ℝ) :
  (6 - v ≠ 0) ∧ (6 + v ≠ 0) ∧ ((48 / (6 - v)) + (48 / (6 + v)) = 18) → v = 2 := 
by
  sorry

end NUMINAMATH_GPT_river_flow_speed_l2175_217513


namespace NUMINAMATH_GPT_pattern_equation_l2175_217587

theorem pattern_equation (n : ℕ) (h : 0 < n) : n * (n + 2) + 1 = (n + 1) ^ 2 := by
  sorry

end NUMINAMATH_GPT_pattern_equation_l2175_217587


namespace NUMINAMATH_GPT_sum_of_powers_of_minus_one_l2175_217570

theorem sum_of_powers_of_minus_one : (-1) ^ 2010 + (-1) ^ 2011 + 1 ^ 2012 - 1 ^ 2013 + (-1) ^ 2014 = -1 := by
  sorry

end NUMINAMATH_GPT_sum_of_powers_of_minus_one_l2175_217570


namespace NUMINAMATH_GPT_correct_NR_A_correct_NR_B_correct_NR_C_NR_B_highest_l2175_217574

-- Define the given percentages for each ship
def P_A : ℝ := 0.30
def C_A : ℝ := 0.25
def P_B : ℝ := 0.50
def C_B : ℝ := 0.15
def P_C : ℝ := 0.20
def C_C : ℝ := 0.35

-- Define the derived non-car round-trip percentages 
def NR_A : ℝ := P_A - (P_A * C_A)
def NR_B : ℝ := P_B - (P_B * C_B)
def NR_C : ℝ := P_C - (P_C * C_C)

-- Statements to be proved
theorem correct_NR_A : NR_A = 0.225 := sorry
theorem correct_NR_B : NR_B = 0.425 := sorry
theorem correct_NR_C : NR_C = 0.13 := sorry

-- Proof that NR_B is the highest percentage
theorem NR_B_highest : NR_B > NR_A ∧ NR_B > NR_C := sorry

end NUMINAMATH_GPT_correct_NR_A_correct_NR_B_correct_NR_C_NR_B_highest_l2175_217574


namespace NUMINAMATH_GPT_sms_message_fraudulent_l2175_217527

-- Define the conditions as properties
def messageArrivedNumberKnown (msg : String) (numberKnown : Bool) : Prop :=
  msg = "SMS message has already arrived" ∧ numberKnown = true

def fraudDefinition (acquisition : String -> Prop) : Prop :=
  ∀ (s : String), acquisition s = (s = "acquisition of property by third parties through deception or gaining the trust of the victim")

-- Define the main proof problem statement
theorem sms_message_fraudulent (msg : String) (numberKnown : Bool) (acquisition : String -> Prop) :
  messageArrivedNumberKnown msg numberKnown ∧ fraudDefinition acquisition →
  acquisition "acquisition of property by third parties through deception or gaining the trust of the victim" :=
  sorry

end NUMINAMATH_GPT_sms_message_fraudulent_l2175_217527


namespace NUMINAMATH_GPT_vlad_score_l2175_217573

theorem vlad_score :
  ∀ (rounds wins : ℕ) (totalPoints taroPoints vladPoints : ℕ),
    rounds = 30 →
    (wins = 5) →
    (totalPoints = rounds * wins) →
    (taroPoints = (3 * totalPoints) / 5 - 4) →
    (vladPoints = totalPoints - taroPoints) →
    vladPoints = 64 :=
by
  intros rounds wins totalPoints taroPoints vladPoints h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_vlad_score_l2175_217573


namespace NUMINAMATH_GPT_combination_divisible_by_30_l2175_217544

theorem combination_divisible_by_30 (n : ℤ) : ∃ k : ℤ, n^5 - n = 30 * k :=
by
  sorry

end NUMINAMATH_GPT_combination_divisible_by_30_l2175_217544


namespace NUMINAMATH_GPT_two_digit_number_representation_l2175_217554

-- Define the conditions and the problem statement in Lean 4
def units_digit (n : ℕ) := n % 10
def tens_digit (n : ℕ) := (n / 10) % 10

theorem two_digit_number_representation (x : ℕ) (h : x < 10) :
  ∃ n : ℕ, units_digit n = x ∧ tens_digit n = 2 * x ^ 2 ∧ n = 20 * x ^ 2 + x :=
by {
  sorry
}

end NUMINAMATH_GPT_two_digit_number_representation_l2175_217554


namespace NUMINAMATH_GPT_part1_part2_part3_l2175_217599

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (a * x^2 - Real.log x) * (x - Real.log x) + 1

variable {a : ℝ}

-- Prove that for all x > 0, if ax^2 > ln x, then f(x) ≥ ax^2 - ln x + 1
theorem part1 (h : ∀ x > 0, a*x^2 > Real.log x) (x : ℝ) (hx : x > 0) :
  f a x ≥ a*x^2 - Real.log x + 1 := sorry

-- Find the maximum value of a given there exists x₀ ∈ (0, +∞) where f(x₀) = 1 + x₀ ln x₀ - ln² x₀
theorem part2 (h : ∃ x₀ > 0, f a x₀ = 1 + x₀ * Real.log x₀ - (Real.log x₀)^2) :
  a ≤ 1 / Real.exp 1 := sorry

-- Prove that for all 1 < x < 2, we have f(x) > ax(2-ax)
theorem part3 (h : ∀ x, 1 < x ∧ x < 2) (x : ℝ) (hx1 : 1 < x) (hx2 : x < 2) :
  f a x > a * x * (2 - a * x) := sorry

end NUMINAMATH_GPT_part1_part2_part3_l2175_217599


namespace NUMINAMATH_GPT_symmetric_points_x_axis_l2175_217520

theorem symmetric_points_x_axis (m n : ℤ) (h1 : m + 1 = 1) (h2 : 3 = -(n - 2)) : m - n = 1 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_points_x_axis_l2175_217520


namespace NUMINAMATH_GPT_number_of_tables_l2175_217506

theorem number_of_tables (x : ℕ) (h : 2 * (x - 1) + 3 = 65) : x = 32 :=
sorry

end NUMINAMATH_GPT_number_of_tables_l2175_217506


namespace NUMINAMATH_GPT_n_m_odd_implies_sum_odd_l2175_217523

theorem n_m_odd_implies_sum_odd {n m : ℤ} (h : Odd (n^2 + m^2)) : Odd (n + m) :=
by
  sorry

end NUMINAMATH_GPT_n_m_odd_implies_sum_odd_l2175_217523


namespace NUMINAMATH_GPT_parallel_line_segment_length_l2175_217561

theorem parallel_line_segment_length (AB : ℝ) (S : ℝ) (x : ℝ) 
  (h1 : AB = 36) 
  (h2 : S = (S / 2) * 2)
  (h3 : x / AB = (↑(1 : ℝ) / 2 * S / S) ^ (1 / 2)) : 
  x = 18 * Real.sqrt 2 :=
by 
    sorry 

end NUMINAMATH_GPT_parallel_line_segment_length_l2175_217561


namespace NUMINAMATH_GPT_solve_equation_l2175_217593

theorem solve_equation:
  ∃ (x y : ℕ), 
    x > 0 ∧ y > 0 ∧ 
    (x - y - x / y - (x^3 / y^3) + (x^4 / y^4) = 2017) ∧ 
    ((x = 2949 ∧ y = 983) ∨ (x = 4022 ∧ y = 2011)) :=
sorry

end NUMINAMATH_GPT_solve_equation_l2175_217593


namespace NUMINAMATH_GPT_inscribed_circle_radius_DEF_l2175_217504

noncomputable def radius_inscribed_circle (DE DF EF : ℕ) : ℝ :=
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  K / s

theorem inscribed_circle_radius_DEF :
  radius_inscribed_circle 26 16 20 = 5 * Real.sqrt 511.5 / 31 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_DEF_l2175_217504


namespace NUMINAMATH_GPT_find_x7_plus_32x2_l2175_217586

theorem find_x7_plus_32x2 (x : ℝ) (h : x^3 + 2 * x = 4) : x^7 + 32 * x^2 = 64 :=
sorry

end NUMINAMATH_GPT_find_x7_plus_32x2_l2175_217586


namespace NUMINAMATH_GPT_determine_function_l2175_217569

theorem determine_function (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 → f (1 / x) = 1 / (x + 1)) →
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 → f x = x / (x + 1)) :=
by
  sorry

end NUMINAMATH_GPT_determine_function_l2175_217569


namespace NUMINAMATH_GPT_parabola_directrix_y_neg1_l2175_217509

-- We define the problem given the conditions.
def parabola_directrix (p : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 = 4 * y → y = -p

-- Now we state what needs to be proved.
theorem parabola_directrix_y_neg1 : parabola_directrix 1 :=
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_y_neg1_l2175_217509


namespace NUMINAMATH_GPT_square_roots_of_16_l2175_217519

theorem square_roots_of_16 :
  {y : ℤ | y^2 = 16} = {4, -4} :=
by
  sorry

end NUMINAMATH_GPT_square_roots_of_16_l2175_217519


namespace NUMINAMATH_GPT_scores_fraction_difference_l2175_217579

theorem scores_fraction_difference (y : ℕ) (white_ratio : ℕ) (black_ratio : ℕ) (total : ℕ) 
(h1 : white_ratio = 7) (h2 : black_ratio = 6) (h3 : total = 78) 
(h4 : y = white_ratio + black_ratio) : 
  ((white_ratio * total / y) - (black_ratio * total / y)) / total = 1 / 13 :=
by
 sorry

end NUMINAMATH_GPT_scores_fraction_difference_l2175_217579


namespace NUMINAMATH_GPT_number_of_outfits_l2175_217580

def red_shirts : ℕ := 6
def green_shirts : ℕ := 7
def number_pants : ℕ := 9
def blue_hats : ℕ := 10
def red_hats : ℕ := 10

theorem number_of_outfits :
  (red_shirts * number_pants * blue_hats) + (green_shirts * number_pants * red_hats) = 1170 :=
by
  sorry

end NUMINAMATH_GPT_number_of_outfits_l2175_217580


namespace NUMINAMATH_GPT_average_sequence_x_l2175_217508

theorem average_sequence_x (x : ℚ) (h : (5050 + x) / 101 = 50 * x) : x = 5050 / 5049 :=
by
  sorry

end NUMINAMATH_GPT_average_sequence_x_l2175_217508


namespace NUMINAMATH_GPT_problem_l2175_217557

-- Define the main problem conditions
variables {a b c : ℝ}
axiom h1 : a^2 + b^2 + c^2 = 63
axiom h2 : 2 * a + 3 * b + 6 * c = 21 * Real.sqrt 7

-- Define the goal
theorem problem :
  (a / c) ^ (a / b) = (1 / 3) ^ (2 / 3) :=
sorry

end NUMINAMATH_GPT_problem_l2175_217557


namespace NUMINAMATH_GPT_length_of_platform_is_300_meters_l2175_217598

-- Definitions used in the proof
def kmph_to_mps (v: ℕ) : ℕ := (v * 1000) / 3600

def speed := kmph_to_mps 72

def time_cross_man := 15

def length_train := speed * time_cross_man

def time_cross_platform := 30

def total_distance_cross_platform := speed * time_cross_platform

def length_platform := total_distance_cross_platform - length_train

theorem length_of_platform_is_300_meters :
  length_platform = 300 :=
by
  sorry

end NUMINAMATH_GPT_length_of_platform_is_300_meters_l2175_217598


namespace NUMINAMATH_GPT_orange_balloons_count_l2175_217553

variable (original_orange_balloons : ℝ)
variable (found_orange_balloons : ℝ)
variable (total_orange_balloons : ℝ)

theorem orange_balloons_count :
  original_orange_balloons = 9.0 →
  found_orange_balloons = 2.0 →
  total_orange_balloons = original_orange_balloons + found_orange_balloons →
  total_orange_balloons = 11.0 := by
  sorry

end NUMINAMATH_GPT_orange_balloons_count_l2175_217553


namespace NUMINAMATH_GPT_determine_function_l2175_217555

theorem determine_function (f : ℝ → ℝ)
    (h1 : f 1 = 0)
    (h2 : ∀ x y : ℝ, |f x - f y| = |x - y|) :
    (∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = 1 - x) := by
  sorry

end NUMINAMATH_GPT_determine_function_l2175_217555


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l2175_217526

theorem simplify_and_evaluate_expression (x : ℤ) (h : x = -2) : 
  2 * x * (x - 3) - (x - 2) * (x + 1) = 16 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l2175_217526


namespace NUMINAMATH_GPT_circus_capacity_l2175_217530

theorem circus_capacity (sections : ℕ) (people_per_section : ℕ) (h1 : sections = 4) (h2 : people_per_section = 246) :
  sections * people_per_section = 984 :=
by
  sorry

end NUMINAMATH_GPT_circus_capacity_l2175_217530


namespace NUMINAMATH_GPT_length_of_train_correct_l2175_217543

noncomputable def length_of_train (speed_kmh : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  speed_ms * time_sec

theorem length_of_train_correct :
  length_of_train 60 18 = 300.06 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_length_of_train_correct_l2175_217543


namespace NUMINAMATH_GPT_initial_capacity_of_drum_x_l2175_217558

theorem initial_capacity_of_drum_x (C x : ℝ) (h_capacity_y : 2 * x = 2 * 0.75 * C) :
  x = 0.75 * C :=
sorry

end NUMINAMATH_GPT_initial_capacity_of_drum_x_l2175_217558


namespace NUMINAMATH_GPT_total_balls_l2175_217585

theorem total_balls {balls_per_box boxes : ℕ} (h1 : balls_per_box = 3) (h2 : boxes = 2) : balls_per_box * boxes = 6 :=
by
  sorry

end NUMINAMATH_GPT_total_balls_l2175_217585


namespace NUMINAMATH_GPT_sum_of_coefficients_l2175_217578

theorem sum_of_coefficients:
  (x^3 + 2*x + 1) * (3*x^2 + 4) = 28 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l2175_217578


namespace NUMINAMATH_GPT_population_increase_rate_l2175_217568

theorem population_increase_rate (P₀ P₁ : ℕ) (rate : ℚ) (h₁ : P₀ = 220) (h₂ : P₁ = 242) :
  rate = ((P₁ - P₀ : ℚ) / P₀) * 100 := by
  sorry

end NUMINAMATH_GPT_population_increase_rate_l2175_217568


namespace NUMINAMATH_GPT_paint_pyramid_l2175_217551

theorem paint_pyramid (colors : Finset ℕ) (n : ℕ) (h : colors.card = 5) :
  let ways_to_paint := 5 * 4 * 3 * 2 * 1
  n = ways_to_paint
:=
sorry

end NUMINAMATH_GPT_paint_pyramid_l2175_217551


namespace NUMINAMATH_GPT_uncool_students_in_two_classes_l2175_217529

theorem uncool_students_in_two_classes
  (students_class1 : ℕ)
  (cool_dads_class1 : ℕ)
  (cool_moms_class1 : ℕ)
  (both_cool_class1 : ℕ)
  (students_class2 : ℕ)
  (cool_dads_class2 : ℕ)
  (cool_moms_class2 : ℕ)
  (both_cool_class2 : ℕ)
  (h1 : students_class1 = 45)
  (h2 : cool_dads_class1 = 22)
  (h3 : cool_moms_class1 = 25)
  (h4 : both_cool_class1 = 11)
  (h5 : students_class2 = 35)
  (h6 : cool_dads_class2 = 15)
  (h7 : cool_moms_class2 = 18)
  (h8 : both_cool_class2 = 7) :
  (students_class1 - ((cool_dads_class1 - both_cool_class1) + (cool_moms_class1 - both_cool_class1) + both_cool_class1) +
   students_class2 - ((cool_dads_class2 - both_cool_class2) + (cool_moms_class2 - both_cool_class2) + both_cool_class2)
  ) = 18 :=
sorry

end NUMINAMATH_GPT_uncool_students_in_two_classes_l2175_217529


namespace NUMINAMATH_GPT_value_of_expression_l2175_217528

variable (m : ℝ)

theorem value_of_expression (h : 2 * m^2 + 3 * m - 1 = 0) : 
  4 * m^2 + 6 * m - 2019 = -2017 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2175_217528


namespace NUMINAMATH_GPT_simplify_and_evaluate_l2175_217542

theorem simplify_and_evaluate (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 2) (hx3 : x ≠ -2) (hx4 : x = -1) :
  (2 / (x^2 - 4)) / (1 / (x^2 - 2*x)) = -2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l2175_217542


namespace NUMINAMATH_GPT_y_intercept_of_line_b_l2175_217575

theorem y_intercept_of_line_b
  (m : ℝ) (c₁ : ℝ) (c₂ : ℝ) (x₁ : ℝ) (y₁ : ℝ)
  (h_parallel : m = 3/2)
  (h_point : (4, 2) ∈ { p : ℝ × ℝ | p.2 = m * p.1 + c₂ }) :
  c₂ = -4 := by
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_b_l2175_217575


namespace NUMINAMATH_GPT_quadrant_of_angle_l2175_217531

theorem quadrant_of_angle (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
  ∃ n : ℤ, n = 1 ∧ α = (n * π + π / 2) :=
sorry

end NUMINAMATH_GPT_quadrant_of_angle_l2175_217531


namespace NUMINAMATH_GPT_numberOfBigBoats_l2175_217590

-- Conditions
variable (students : Nat) (bigBoatCapacity : Nat) (smallBoatCapacity : Nat) (totalBoats : Nat)
variable (students_eq : students = 52)
variable (bigBoatCapacity_eq : bigBoatCapacity = 8)
variable (smallBoatCapacity_eq : smallBoatCapacity = 4)
variable (totalBoats_eq : totalBoats = 9)

theorem numberOfBigBoats : bigBoats + smallBoats = totalBoats → 
                         bigBoatCapacity * bigBoats + smallBoatCapacity * smallBoats = students → 
                         bigBoats = 4 := 
by
  intros h1 h2
  -- Proof steps
  sorry


end NUMINAMATH_GPT_numberOfBigBoats_l2175_217590


namespace NUMINAMATH_GPT_divisibility_proof_l2175_217563

theorem divisibility_proof (n : ℕ) (hn : 0 < n) (h : n ∣ (10^n - 1)) : 
  n ∣ ((10^n - 1) / 9) :=
  sorry

end NUMINAMATH_GPT_divisibility_proof_l2175_217563


namespace NUMINAMATH_GPT_final_sale_price_l2175_217576

theorem final_sale_price (P P₁ P₂ P₃ : ℝ) (d₁ d₂ d₃ dx : ℝ) (x : ℝ)
  (h₁ : P = 600) 
  (h_d₁ : d₁ = 20) (h_d₂ : d₂ = 15) (h_d₃ : d₃ = 10)
  (h₁₁ : P₁ = P * (1 - d₁ / 100))
  (h₁₂ : P₂ = P₁ * (1 - d₂ / 100))
  (h₁₃ : P₃ = P₂ * (1 - d₃ / 100))
  (h_P₃_final : P₃ = 367.2) :
  P₃ * (100 - dx) / 100 = 367.2 * (100 - x) / 100 :=
by
  sorry

end NUMINAMATH_GPT_final_sale_price_l2175_217576


namespace NUMINAMATH_GPT_ratio_of_inscribed_squares_in_isosceles_right_triangle_l2175_217565

def isosceles_right_triangle (a b : ℝ) (leg : ℝ) : Prop :=
  let a_square_inscribed := a = leg
  let b_square_inscribed := b = leg
  a_square_inscribed ∧ b_square_inscribed

theorem ratio_of_inscribed_squares_in_isosceles_right_triangle (a b leg : ℝ)
  (h : isosceles_right_triangle a b leg) :
  leg = 6 ∧ a = leg ∧ b = leg → a / b = 1 := 
by {
  sorry -- the proof will go here
}

end NUMINAMATH_GPT_ratio_of_inscribed_squares_in_isosceles_right_triangle_l2175_217565


namespace NUMINAMATH_GPT_ackermann_3_2_l2175_217514

-- Define the Ackermann function
def ackermann : ℕ → ℕ → ℕ
| 0, n => n + 1
| (m + 1), 0 => ackermann m 1
| (m + 1), (n + 1) => ackermann m (ackermann (m + 1) n)

-- Prove that A(3, 2) = 29
theorem ackermann_3_2 : ackermann 3 2 = 29 := by
  sorry

end NUMINAMATH_GPT_ackermann_3_2_l2175_217514


namespace NUMINAMATH_GPT_power_multiplication_l2175_217539

theorem power_multiplication : 3000 * (3000 ^ 2999) = 3000 ^ 3000 := by
  sorry

end NUMINAMATH_GPT_power_multiplication_l2175_217539


namespace NUMINAMATH_GPT_total_games_in_season_l2175_217535

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem total_games_in_season
  (teams : ℕ)
  (games_per_pair : ℕ)
  (h_teams : teams = 30)
  (h_games_per_pair : games_per_pair = 6) :
  (choose 30 2 * games_per_pair) = 2610 :=
  by
    sorry

end NUMINAMATH_GPT_total_games_in_season_l2175_217535


namespace NUMINAMATH_GPT_value_of_k_if_two_equal_real_roots_l2175_217566

theorem value_of_k_if_two_equal_real_roots (k : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x + k = 0 → x^2 - 2 * x + k = 0) → k = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_k_if_two_equal_real_roots_l2175_217566


namespace NUMINAMATH_GPT_explicit_formula_is_even_tangent_line_at_1_tangent_line_equation_l2175_217552

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 + (2 - a) * x + (a - 1)

-- Proof needed for the first question:
theorem explicit_formula_is_even (a : ℝ) : (∀ x : ℝ, f x a = f (-x) a) → a = 2 ∧ ∀ x : ℝ, f x a = x^2 + 1 :=
by sorry

-- Proof needed for the second question:
theorem tangent_line_at_1 (f : ℝ → ℝ) : (∀ x : ℝ, f x = x^2 + 1) → ∀ x : ℝ, deriv f 1 = 2 :=
by sorry

-- The tangent line equation at x = 1 in the required form
theorem tangent_line_equation (f : ℝ → ℝ) : (∀ x : ℝ, f x = x^2 + 1) → ∀ x : ℝ, deriv f 1 = 2 → (f 1 - deriv f 1 * 1 + deriv f 1 * x = 2 * x) :=
by sorry

end NUMINAMATH_GPT_explicit_formula_is_even_tangent_line_at_1_tangent_line_equation_l2175_217552


namespace NUMINAMATH_GPT_parabola_focus_l2175_217500

theorem parabola_focus : 
  ∀ x y : ℝ, y = - (1 / 16) * x^2 → ∃ f : ℝ × ℝ, f = (0, -4) := 
by
  sorry

end NUMINAMATH_GPT_parabola_focus_l2175_217500


namespace NUMINAMATH_GPT_problem_l2175_217503

variable (a b : ℝ)

theorem problem (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 4) : a^2 + b^2 ≥ 8 := 
sorry

end NUMINAMATH_GPT_problem_l2175_217503


namespace NUMINAMATH_GPT_quarter_sector_area_l2175_217592

theorem quarter_sector_area (d : ℝ) (h : d = 10) : (π * (d / 2)^2) / 4 = 6.25 * π :=
by 
  sorry

end NUMINAMATH_GPT_quarter_sector_area_l2175_217592


namespace NUMINAMATH_GPT_find_number_l2175_217541

variable (number x : ℝ)

theorem find_number (h1 : number * x = 1600) (h2 : x = -8) : number = -200 := by
  sorry

end NUMINAMATH_GPT_find_number_l2175_217541


namespace NUMINAMATH_GPT_travel_rate_on_foot_l2175_217596

theorem travel_rate_on_foot
  (total_distance : ℝ)
  (total_time : ℝ)
  (distance_on_foot : ℝ)
  (rate_on_bicycle : ℝ)
  (rate_on_foot : ℝ) :
  total_distance = 80 ∧ total_time = 7 ∧ distance_on_foot = 32 ∧ rate_on_bicycle = 16 →
  rate_on_foot = 8 := by
  sorry

end NUMINAMATH_GPT_travel_rate_on_foot_l2175_217596


namespace NUMINAMATH_GPT_hitting_first_third_fifth_probability_hitting_exactly_three_out_of_five_probability_l2175_217501

noncomputable def probability_hitting_first_third_fifth (P : ℚ) : ℚ :=
  P * (1 - P) * P * (1 - P) * P

noncomputable def binomial_coefficient (n k : ℕ) : ℕ := 
  Nat.choose n k

noncomputable def probability_hitting_exactly_three_out_of_five (P : ℚ) : ℚ :=
  binomial_coefficient 5 3 * P^3 * (1 - P)^2

theorem hitting_first_third_fifth_probability :
  probability_hitting_first_third_fifth (3/5) = 108/3125 := by
  sorry

theorem hitting_exactly_three_out_of_five_probability :
  probability_hitting_exactly_three_out_of_five (3/5) = 216/625 := by
  sorry

end NUMINAMATH_GPT_hitting_first_third_fifth_probability_hitting_exactly_three_out_of_five_probability_l2175_217501


namespace NUMINAMATH_GPT_odd_tiling_numbers_l2175_217589

def f (n k : ℕ) : ℕ := sorry -- Assume f(n, 2k) is defined appropriately.

theorem odd_tiling_numbers (n : ℕ) : (∀ k : ℕ, f n (2*k) % 2 = 1) ↔ ∃ i : ℕ, n = 2^i - 1 := sorry

end NUMINAMATH_GPT_odd_tiling_numbers_l2175_217589


namespace NUMINAMATH_GPT_expected_total_rain_l2175_217549

theorem expected_total_rain :
  let p_sun := 0.30
  let p_rain5 := 0.30
  let p_rain12 := 0.40
  let rain_sun := 0
  let rain_rain5 := 5
  let rain_rain12 := 12
  let days := 6
  let E_rain := p_sun * rain_sun + p_rain5 * rain_rain5 + p_rain12 * rain_rain12
  E_rain * days = 37.8 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_expected_total_rain_l2175_217549


namespace NUMINAMATH_GPT_inequality_abc_l2175_217510

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    1 / (a * b * c) + 1 ≥ 3 * (1 / (a^2 + b^2 + c^2) + 1 / (a + b + c)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_abc_l2175_217510


namespace NUMINAMATH_GPT_length_of_symmedian_l2175_217545

theorem length_of_symmedian (a b c : ℝ) (AS : ℝ) :
  AS = (2 * b * c^2) / (b^2 + c^2) := sorry

end NUMINAMATH_GPT_length_of_symmedian_l2175_217545


namespace NUMINAMATH_GPT_average_interest_rate_l2175_217556

theorem average_interest_rate (x : ℝ) (h1 : 0 < x ∧ x < 6000)
  (h2 : 0.03 * (6000 - x) = 0.055 * x) :
  ((0.03 * (6000 - x) + 0.055 * x) / 6000) = 0.0388 :=
by
  sorry

end NUMINAMATH_GPT_average_interest_rate_l2175_217556


namespace NUMINAMATH_GPT_problem1_problem2_l2175_217548

-- Problem 1: Prove that the given expression evaluates to the correct answer
theorem problem1 :
  2 * Real.sin (Real.pi / 6) - (2015 - Real.pi)^0 + abs (1 - Real.tan (Real.pi / 3)) = abs (1 - Real.sqrt 3) :=
sorry

-- Problem 2: Prove that the solutions to the given equation are correct
theorem problem2 (x : ℝ) :
  (x-2)^2 = 3 * (x-2) → x = 2 ∨ x = 5 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l2175_217548


namespace NUMINAMATH_GPT_range_of_a_l2175_217533

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (a + 1) * x > a + 1 ↔ x > 1) : a > -1 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2175_217533
