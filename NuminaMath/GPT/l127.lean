import Mathlib

namespace NUMINAMATH_GPT_runners_never_meet_l127_12798

theorem runners_never_meet
    (x : ℕ)  -- Speed of first runner
    (a : ℕ)  -- 1/3 of the circumference of the track
    (C : ℕ)  -- Circumference of the track
    (hC : C = 3 * a)  -- Given that C = 3 * a
    (h_speeds : 1 * x = x ∧ 2 * x = 2 * x ∧ 4 * x = 4 * x)  -- Speed ratios: 1:2:4
    (t : ℕ)  -- Time variable
: ¬(∃ t, (x * t % C = 2 * x * t % C ∧ 2 * x * t % C = 4 * x * t % C)) :=
by sorry

end NUMINAMATH_GPT_runners_never_meet_l127_12798


namespace NUMINAMATH_GPT_complex_square_l127_12771

theorem complex_square (i : ℂ) (h : i^2 = -1) : (1 - i)^2 = -2 * i :=
by
  sorry

end NUMINAMATH_GPT_complex_square_l127_12771


namespace NUMINAMATH_GPT_abs_expression_value_l127_12713

theorem abs_expression_value (x : ℤ) (h : x = -2023) : 
  abs (abs (abs (abs x - 2 * x) - abs x) - x) = 6069 :=
by sorry

end NUMINAMATH_GPT_abs_expression_value_l127_12713


namespace NUMINAMATH_GPT_evaluate_expression_l127_12794

theorem evaluate_expression :
  |7 - (8^2) * (3 - 12)| - |(5^3) - (Real.sqrt 11)^4| = 579 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l127_12794


namespace NUMINAMATH_GPT_find_family_ages_l127_12737

theorem find_family_ages :
  ∃ (a b father_age mother_age : ℕ), 
    (a < 21) ∧
    (b < 21) ∧
    (a^3 + b^2 > 1900) ∧
    (a^3 + b^2 < 1978) ∧
    (father_age = 1978 - (a^3 + b^2)) ∧
    (mother_age = father_age - 8) ∧
    (a = 12) ∧
    (b = 14) ∧
    (father_age = 54) ∧
    (mother_age = 46) := 
by 
  use 12, 14, 54, 46
  sorry

end NUMINAMATH_GPT_find_family_ages_l127_12737


namespace NUMINAMATH_GPT_find_two_digit_number_l127_12799

theorem find_two_digit_number (N : ℕ) (h1 : 10 ≤ N ∧ N < 100) 
                              (h2 : N % 2 = 0) (h3 : N % 11 = 0) 
                              (h4 : ∃ k : ℕ, (N / 10) * (N % 10) = k^3) :
  N = 88 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_two_digit_number_l127_12799


namespace NUMINAMATH_GPT_probability_three_same_color_is_one_seventeenth_l127_12727

def standard_deck := {cards : Finset ℕ // cards.card = 52 ∧ ∃ reds blacks, reds.card = 26 ∧ blacks.card = 26 ∧ (reds ∪ blacks = cards)}

def num_ways_to_pick_3_same_color : ℕ :=
  (26 * 25 * 24) + (26 * 25 * 24)

def total_ways_to_pick_3 : ℕ :=
  52 * 51 * 50

def probability_top_three_same_color := (num_ways_to_pick_3_same_color / total_ways_to_pick_3 : ℚ)

theorem probability_three_same_color_is_one_seventeenth :
  probability_top_three_same_color = (1 / 17 : ℚ) := by sorry

end NUMINAMATH_GPT_probability_three_same_color_is_one_seventeenth_l127_12727


namespace NUMINAMATH_GPT_same_color_probability_l127_12705

def sides := 12
def violet_sides := 3
def orange_sides := 4
def lime_sides := 5

def prob_violet := violet_sides / sides
def prob_orange := orange_sides / sides
def prob_lime := lime_sides / sides

theorem same_color_probability :
  (prob_violet * prob_violet) + (prob_orange * prob_orange) + (prob_lime * prob_lime) = 25 / 72 :=
by
  sorry

end NUMINAMATH_GPT_same_color_probability_l127_12705


namespace NUMINAMATH_GPT_verify_a_l127_12710

def g (x : ℝ) : ℝ := 5 * x - 7

theorem verify_a (a : ℝ) : g a = 0 ↔ a = 7 / 5 := by
  sorry

end NUMINAMATH_GPT_verify_a_l127_12710


namespace NUMINAMATH_GPT_three_irrational_numbers_l127_12700

theorem three_irrational_numbers (a b c d e : ℝ) 
  (ha : ¬ ∃ q1 q2 : ℚ, a = q1 + q2) 
  (hb : ¬ ∃ q1 q2 : ℚ, b = q1 + q2) 
  (hc : ¬ ∃ q1 q2 : ℚ, c = q1 + q2) 
  (hd : ¬ ∃ q1 q2 : ℚ, d = q1 + q2) 
  (he : ¬ ∃ q1 q2 : ℚ, e = q1 + q2) : 
  ∃ x y z, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) 
  ∧ (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) 
  ∧ (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e)
  ∧ (¬ ∃ q1 q2 : ℚ, x + y = q1 + q2) 
  ∧ (¬ ∃ q1 q2 : ℚ, y + z = q1 + q2) 
  ∧ (¬ ∃ q1 q2 : ℚ, z + x = q1 + q2) :=
sorry

end NUMINAMATH_GPT_three_irrational_numbers_l127_12700


namespace NUMINAMATH_GPT_num_integers_in_set_x_l127_12757

-- Definition and conditions
variable (x y : Finset ℤ)
variable (h1 : y.card = 10)
variable (h2 : (x ∩ y).card = 6)
variable (h3 : (x.symmDiff y).card = 6)

-- Proof statement
theorem num_integers_in_set_x : x.card = 8 := by
  sorry

end NUMINAMATH_GPT_num_integers_in_set_x_l127_12757


namespace NUMINAMATH_GPT_gcd_779_209_589_eq_19_l127_12756

theorem gcd_779_209_589_eq_19 : Nat.gcd 779 (Nat.gcd 209 589) = 19 := by
  sorry

end NUMINAMATH_GPT_gcd_779_209_589_eq_19_l127_12756


namespace NUMINAMATH_GPT_infinite_squares_in_arithmetic_sequence_l127_12796

open Nat Int

theorem infinite_squares_in_arithmetic_sequence
  (a d : ℤ) (h_d_nonneg : d ≥ 0) (x : ℤ) 
  (hx_square : ∃ n : ℕ, a + n * d = x * x) :
  ∃ (infinitely_many_n : ℕ → Prop), 
    (∀ k : ℕ, ∃ n : ℕ, infinitely_many_n n ∧ a + n * d = (x + k * d) * (x + k * d)) :=
sorry

end NUMINAMATH_GPT_infinite_squares_in_arithmetic_sequence_l127_12796


namespace NUMINAMATH_GPT_price_of_food_before_tax_and_tip_l127_12774

noncomputable def actual_price_of_food (total_paid : ℝ) (tip_rate tax_rate : ℝ) : ℝ :=
  total_paid / (1 + tip_rate) / (1 + tax_rate)

theorem price_of_food_before_tax_and_tip :
  actual_price_of_food 211.20 0.20 0.10 = 160 :=
by
  sorry

end NUMINAMATH_GPT_price_of_food_before_tax_and_tip_l127_12774


namespace NUMINAMATH_GPT_Lenora_scored_30_points_l127_12784

variable (x y : ℕ)
variable (hx : x + y = 40)
variable (three_point_success_rate : ℚ := 25 / 100)
variable (free_throw_success_rate : ℚ := 50 / 100)
variable (points_three_point : ℚ := 3)
variable (points_free_throw : ℚ := 1)
variable (three_point_contribution : ℚ := three_point_success_rate * points_three_point * x)
variable (free_throw_contribution : ℚ := free_throw_success_rate * points_free_throw * y)
variable (total_points : ℚ := three_point_contribution + free_throw_contribution)

theorem Lenora_scored_30_points : total_points = 30 :=
by
  sorry

end NUMINAMATH_GPT_Lenora_scored_30_points_l127_12784


namespace NUMINAMATH_GPT_sector_area_l127_12718

noncomputable def area_of_sector (r : ℝ) (theta : ℝ) : ℝ :=
  1 / 2 * r * r * theta

theorem sector_area (r : ℝ) (theta : ℝ) (h_r : r = Real.pi) (h_theta : theta = 2 * Real.pi / 3) :
  area_of_sector r theta = Real.pi^3 / 6 :=
by
  sorry

end NUMINAMATH_GPT_sector_area_l127_12718


namespace NUMINAMATH_GPT_min_mn_sum_l127_12764

theorem min_mn_sum :
  ∃ (m n : ℕ), n > m ∧ m ≥ 1 ∧ 
  (1978^n % 1000 = 1978^m % 1000) ∧ (m + n = 106) :=
sorry

end NUMINAMATH_GPT_min_mn_sum_l127_12764


namespace NUMINAMATH_GPT_Donny_change_l127_12776

theorem Donny_change (tank_capacity : ℕ) (initial_fuel : ℕ) (money_available : ℕ) (fuel_cost_per_liter : ℕ) 
  (h1 : tank_capacity = 150) 
  (h2 : initial_fuel = 38) 
  (h3 : money_available = 350) 
  (h4 : fuel_cost_per_liter = 3) : 
  money_available - (tank_capacity - initial_fuel) * fuel_cost_per_liter = 14 := 
by 
  sorry

end NUMINAMATH_GPT_Donny_change_l127_12776


namespace NUMINAMATH_GPT_team_members_run_distance_l127_12746

-- Define the given conditions
def total_distance : ℕ := 150
def members : ℕ := 5

-- Prove the question == answer given the conditions
theorem team_members_run_distance :
  total_distance / members = 30 :=
by
  sorry

end NUMINAMATH_GPT_team_members_run_distance_l127_12746


namespace NUMINAMATH_GPT_ratio_of_milk_water_in_larger_vessel_l127_12706

-- Definitions of conditions
def volume1 (V : ℝ) : ℝ := 3 * V
def volume2 (V : ℝ) : ℝ := 5 * V

def ratio_milk_water_1 : ℝ × ℝ := (1, 2)
def ratio_milk_water_2 : ℝ × ℝ := (3, 2)

-- Define the problem statement
theorem ratio_of_milk_water_in_larger_vessel (V : ℝ) (hV : V > 0) :
  (volume1 V / (ratio_milk_water_1.1 + ratio_milk_water_1.2)) = V ∧ 
  2 * (volume1 V / (ratio_milk_water_1.1 + ratio_milk_water_1.2)) = 2 * V ∧ 
  3 * (volume2 V / (ratio_milk_water_2.1 + ratio_milk_water_2.2)) = 3 * V ∧ 
  2 * (volume2 V / (ratio_milk_water_2.1 + ratio_milk_water_2.2)) = 2 * V →
  (4 * V) / (4 * V) = 1 :=
sorry

end NUMINAMATH_GPT_ratio_of_milk_water_in_larger_vessel_l127_12706


namespace NUMINAMATH_GPT_proof_correct_chemical_information_l127_12755

def chemical_formula_starch : String := "(C_{6}H_{10}O_{5})_{n}"
def structural_formula_glycine : String := "H_{2}N-CH_{2}-COOH"
def element_in_glass_ceramics_cement : String := "Si"
def elements_cause_red_tide : List String := ["N", "P"]

theorem proof_correct_chemical_information :
  chemical_formula_starch = "(C_{6}H_{10}O_{5})_{n}" ∧
  structural_formula_glycine = "H_{2}N-CH_{2}-COOH" ∧
  element_in_glass_ceramics_cement = "Si" ∧
  elements_cause_red_tide = ["N", "P"] :=
by
  sorry

end NUMINAMATH_GPT_proof_correct_chemical_information_l127_12755


namespace NUMINAMATH_GPT_train_speed_in_kmph_l127_12747

theorem train_speed_in_kmph
  (train_length : ℝ) (bridge_length : ℝ) (time_seconds : ℝ)
  (H1: train_length = 200) (H2: bridge_length = 150) (H3: time_seconds = 34.997200223982084) :
  train_length + bridge_length = 200 + 150 →
  (train_length + bridge_length) / time_seconds * 3.6 = 36 :=
sorry

end NUMINAMATH_GPT_train_speed_in_kmph_l127_12747


namespace NUMINAMATH_GPT_journey_length_25_km_l127_12731

theorem journey_length_25_km:
  ∀ (D T : ℝ),
  (D = 100 * T) →
  (D = 50 * (T + 15/60)) →
  D = 25 :=
by
  intros D T h1 h2
  sorry

end NUMINAMATH_GPT_journey_length_25_km_l127_12731


namespace NUMINAMATH_GPT_part1_part2_part3_l127_12785

-- Defining the quadratic function
def quadratic (t : ℝ) (x : ℝ) : ℝ := x^2 - 2 * t * x + 3

-- Part (1)
theorem part1 (t : ℝ) (h : quadratic t 2 = 1) : t = 3 / 2 :=
by sorry

-- Part (2)
theorem part2 (t : ℝ) (h : ∀x, 0 ≤ x → x ≤ 3 → (quadratic t x) ≥ -2) : t = Real.sqrt 5 :=
by sorry

-- Part (3)
theorem part3 (m a b : ℝ) (hA : quadratic t (m - 2) = a) (hB : quadratic t 4 = b) 
              (hC : quadratic t m = a) (ha : a < b) (hb : b < 3) (ht : t > 0) : 
              (3 < m ∧ m < 4) ∨ (m > 6) :=
by sorry

end NUMINAMATH_GPT_part1_part2_part3_l127_12785


namespace NUMINAMATH_GPT_greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l127_12789

theorem greatest_integer_less_than_150_with_gcd_30_eq_5_is_145 :
  ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ (∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ n) :=
sorry

end NUMINAMATH_GPT_greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l127_12789


namespace NUMINAMATH_GPT_cassie_nail_cutting_l127_12793

structure AnimalCounts where
  dogs : ℕ
  dog_nails_per_foot : ℕ
  dog_feet : ℕ
  parrots : ℕ
  parrot_claws_per_leg : ℕ
  parrot_legs : ℕ
  extra_toe_parrots : ℕ
  extra_toe_claws : ℕ

def total_nails (counts : AnimalCounts) : ℕ :=
  counts.dogs * counts.dog_nails_per_foot * counts.dog_feet +
  counts.parrots * counts.parrot_claws_per_leg * counts.parrot_legs +
  counts.extra_toe_parrots * counts.extra_toe_claws

theorem cassie_nail_cutting :
  total_nails {
    dogs := 4,
    dog_nails_per_foot := 4,
    dog_feet := 4,
    parrots := 8,
    parrot_claws_per_leg := 3,
    parrot_legs := 2,
    extra_toe_parrots := 1,
    extra_toe_claws := 1
  } = 113 :=
by sorry

end NUMINAMATH_GPT_cassie_nail_cutting_l127_12793


namespace NUMINAMATH_GPT_unit_vector_perpendicular_l127_12701

theorem unit_vector_perpendicular (x y : ℝ)
  (h1 : 4 * x + 2 * y = 0) 
  (h2 : x^2 + y^2 = 1) :
  (x = (Real.sqrt 5) / 5 ∧ y = -(2 * (Real.sqrt 5) / 5)) ∨ 
  (x = -(Real.sqrt 5) / 5 ∧ y = 2 * (Real.sqrt 5) / 5) :=
sorry

end NUMINAMATH_GPT_unit_vector_perpendicular_l127_12701


namespace NUMINAMATH_GPT_sum_of_consecutive_odds_l127_12754

theorem sum_of_consecutive_odds (a : ℤ) (h : (a - 2) * a * (a + 2) = 9177) : (a - 2) + a + (a + 2) = 63 := 
sorry

end NUMINAMATH_GPT_sum_of_consecutive_odds_l127_12754


namespace NUMINAMATH_GPT_B_fraction_l127_12769

theorem B_fraction (A_s B_s C_s : ℕ) (h1 : A_s = 600) (h2 : A_s = (2 / 5) * (B_s + C_s))
  (h3 : A_s + B_s + C_s = 1800) :
  B_s / (A_s + C_s) = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_B_fraction_l127_12769


namespace NUMINAMATH_GPT_arith_seq_a15_l127_12725

variable {α : Type} [LinearOrderedField α]

def is_arith_seq (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arith_seq_a15 (a : ℕ → α) (k l m : ℕ) (x y : α) 
  (h_seq : is_arith_seq a)
  (h_k : a k = x)
  (h_l : a l = y) :
  a (l + (l - k)) = 2 * y - x := 
  sorry

end NUMINAMATH_GPT_arith_seq_a15_l127_12725


namespace NUMINAMATH_GPT_sameTypeTerm_l127_12717

variable (a b : ℝ) -- Assume a and b are real numbers 

-- Definitions for each term in the conditions
def term1 : ℝ := 2 * a * b^2
def term2 : ℝ := -a^2 * b
def term3 : ℝ := -2 * a * b
def term4 : ℝ := 5 * a^2

-- The term we are comparing against
def compareTerm : ℝ := 3 * a^2 * b

-- The condition we want to prove
theorem sameTypeTerm : term2 = compareTerm :=
  sorry


end NUMINAMATH_GPT_sameTypeTerm_l127_12717


namespace NUMINAMATH_GPT_sum_of_possible_values_of_x_l127_12721

theorem sum_of_possible_values_of_x :
  let sq_side := (x - 4)
  let rect_length := (x - 5)
  let rect_width := (x + 6)
  let sq_area := (sq_side)^2
  let rect_area := rect_length * rect_width
  (3 * (sq_area) = rect_area) → ∃ (x1 x2 : ℝ), (3 * (x1 - 4) ^ 2 = (x1 - 5) * (x1 + 6)) ∧ (3 * (x2 - 4) ^ 2 = (x2 - 5) * (x2 + 6)) ∧ (x1 + x2 = 12.5) := 
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_values_of_x_l127_12721


namespace NUMINAMATH_GPT_total_plants_in_garden_l127_12744

-- Definitions based on conditions
def basil_plants : ℕ := 5
def oregano_plants : ℕ := 2 + 2 * basil_plants

-- Theorem statement
theorem total_plants_in_garden : basil_plants + oregano_plants = 17 := by
  -- Skipping the proof with sorry
  sorry

end NUMINAMATH_GPT_total_plants_in_garden_l127_12744


namespace NUMINAMATH_GPT_evaluate_expression_l127_12736

theorem evaluate_expression :
  (2 * 4 * 6) * (1 / 2 + 1 / 4 + 1 / 6) = 44 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l127_12736


namespace NUMINAMATH_GPT_equality_of_costs_l127_12723

theorem equality_of_costs (x : ℕ) :
  (800 + 30 * x = 500 + 35 * x) ↔ x = 60 := by
  sorry

end NUMINAMATH_GPT_equality_of_costs_l127_12723


namespace NUMINAMATH_GPT_photos_in_each_album_l127_12795

theorem photos_in_each_album (total_photos : ℕ) (number_of_albums : ℕ) (photos_per_album : ℕ) 
    (h1 : total_photos = 2560) 
    (h2 : number_of_albums = 32) 
    (h3 : total_photos = number_of_albums * photos_per_album) : 
    photos_per_album = 80 := 
by 
    sorry

end NUMINAMATH_GPT_photos_in_each_album_l127_12795


namespace NUMINAMATH_GPT_sum_of_possible_values_of_N_l127_12745

theorem sum_of_possible_values_of_N (N : ℤ) : 
  (N * (N - 8) = 16) -> (∃ a b, N^2 - 8 * N - 16 = 0 ∧ (a + b = 8)) :=
sorry

end NUMINAMATH_GPT_sum_of_possible_values_of_N_l127_12745


namespace NUMINAMATH_GPT_corridor_length_correct_l127_12715

/-- Scale representation in the blueprint: 1 cm represents 10 meters. --/
def scale_cm_to_m (cm: ℝ): ℝ := cm * 10

/-- Length of the corridor in the blueprint. --/
def blueprint_length_cm: ℝ := 9.5

/-- Real-life length of the corridor. --/
def real_life_length: ℝ := 95

/-- Proof that the real-life length of the corridor is correctly calculated. --/
theorem corridor_length_correct :
  scale_cm_to_m blueprint_length_cm = real_life_length :=
by
  sorry

end NUMINAMATH_GPT_corridor_length_correct_l127_12715


namespace NUMINAMATH_GPT_trip_time_difference_l127_12786

-- Define the speed of the motorcycle
def speed : ℤ := 60

-- Define the distances for the two trips
def distance1 : ℤ := 360
def distance2 : ℤ := 420

-- Define the time calculation function
def time (distance speed : ℤ) : ℤ := distance / speed

-- Prove the problem statement
theorem trip_time_difference : (time distance2 speed - time distance1 speed) * 60 = 60 := by
  -- Provide the proof here
  sorry

end NUMINAMATH_GPT_trip_time_difference_l127_12786


namespace NUMINAMATH_GPT_div_condition_l127_12762

theorem div_condition (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  4 * (m * n + 1) % (m + n)^2 = 0 ↔ m = n := 
sorry

end NUMINAMATH_GPT_div_condition_l127_12762


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l127_12760

-- Part 1: Substitution Method
theorem system1_solution (x y : ℤ) :
  2 * x - y = 3 ∧ 3 * x + 2 * y = 8 ↔ x = 2 ∧ y = 1 :=
by
  sorry

-- Part 2: Elimination Method
theorem system2_solution (x y : ℚ) :
  2 * x + y = 2 ∧ 8 * x + 3 * y = 9 ↔ x = 3 / 2 ∧ y = -1 :=
by
  sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l127_12760


namespace NUMINAMATH_GPT_find_x_value_l127_12782

theorem find_x_value (x y z k: ℚ)
  (h1 : x = k * (z^3) / (y^2))
  (h2 : y = 2) (h3 : z = 3)
  (h4 : x = 1)
  : x = (4 / 27) * (4^3) / (6^2) := by
  sorry

end NUMINAMATH_GPT_find_x_value_l127_12782


namespace NUMINAMATH_GPT_number_of_bananas_l127_12740

-- Define costs as constants
def cost_per_banana := 1
def cost_per_apple := 2
def cost_per_twelve_strawberries := 4
def cost_per_avocado := 3
def cost_per_half_bunch_grapes := 2
def total_cost := 28

-- Define quantities as constants
def number_of_apples := 3
def number_of_strawberries := 24
def number_of_avocados := 2
def number_of_half_bunches_grapes := 2

-- Define calculated costs
def cost_of_apples := number_of_apples * cost_per_apple
def cost_of_strawberries := (number_of_strawberries / 12) * cost_per_twelve_strawberries
def cost_of_avocados := number_of_avocados * cost_per_avocado
def cost_of_grapes := number_of_half_bunches_grapes * cost_per_half_bunch_grapes

-- Define total cost of other fruits
def total_cost_of_other_fruits := cost_of_apples + cost_of_strawberries + cost_of_avocados + cost_of_grapes

-- Define the remaining cost for bananas
def remaining_cost := total_cost - total_cost_of_other_fruits

-- Prove the number of bananas
theorem number_of_bananas : remaining_cost / cost_per_banana = 4 :=
by
  -- This is a placeholder to indicate a non-implemented proof
  sorry

end NUMINAMATH_GPT_number_of_bananas_l127_12740


namespace NUMINAMATH_GPT_wheels_motion_is_rotation_l127_12770

def motion_wheel_car := "rotation"
def question_wheels_motion := "What is the type of motion exhibited by the wheels of a moving car?"

theorem wheels_motion_is_rotation :
  (question_wheels_motion = "What is the type of motion exhibited by the wheels of a moving car?" ∧ 
   motion_wheel_car = "rotation") → motion_wheel_car = "rotation" :=
by
  sorry

end NUMINAMATH_GPT_wheels_motion_is_rotation_l127_12770


namespace NUMINAMATH_GPT_sum_not_prime_if_product_equality_l127_12734

theorem sum_not_prime_if_product_equality 
  (a b c d : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : a * b = c * d) : ¬Nat.Prime (a + b + c + d) := 
by
  sorry

end NUMINAMATH_GPT_sum_not_prime_if_product_equality_l127_12734


namespace NUMINAMATH_GPT_lines_perpendicular_slope_l127_12714

theorem lines_perpendicular_slope (k : ℝ) :
  (∀ (x : ℝ), k * 2 = -1) → k = (-1:ℝ)/2 :=
by
  sorry

end NUMINAMATH_GPT_lines_perpendicular_slope_l127_12714


namespace NUMINAMATH_GPT_unique_solution_a_eq_sqrt_three_l127_12750

theorem unique_solution_a_eq_sqrt_three {a : ℝ} (h1 : ∀ x y : ℝ, x^2 + a * abs x + a^2 - 3 = 0 ∧ y^2 + a * abs y + a^2 - 3 = 0 → x = y)
  (h2 : a > 0) : a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_unique_solution_a_eq_sqrt_three_l127_12750


namespace NUMINAMATH_GPT_solution_set_of_inequality_l127_12777

theorem solution_set_of_inequality : 
  { x : ℝ | (1 : ℝ) * (2 * x + 1) < (x + 1) } = { x | -1 < x ∧ x < 0 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l127_12777


namespace NUMINAMATH_GPT_max_true_statements_l127_12779

theorem max_true_statements (a b : ℝ) :
  ((a < b) → (b < 0) → (a < 0) → ¬(1 / a < 1 / b)) ∧
  ((a < b) → (b < 0) → (a < 0) → ¬(a^2 < b^2)) →
  3 = 3
:=
by
  intros
  sorry

end NUMINAMATH_GPT_max_true_statements_l127_12779


namespace NUMINAMATH_GPT_length_of_each_train_l127_12780

theorem length_of_each_train (L : ℝ) 
  (speed_faster : ℝ := 45 * 5 / 18) -- converting 45 km/hr to m/s
  (speed_slower : ℝ := 36 * 5 / 18) -- converting 36 km/hr to m/s
  (time : ℝ := 36) 
  (relative_speed : ℝ := speed_faster - speed_slower) 
  (total_distance : ℝ := relative_speed * time) 
  (length_each_train : ℝ := total_distance / 2) 
  : length_each_train = 45 := 
by 
  sorry

end NUMINAMATH_GPT_length_of_each_train_l127_12780


namespace NUMINAMATH_GPT_remainder_division_l127_12712

-- Definition of the number in terms of its components
def num : ℤ := 98 * 10^6 + 76 * 10^4 + 54 * 10^2 + 32

-- The modulus
def m : ℤ := 25

-- The given problem restated as a hypothesis and goal
theorem remainder_division : num % m = 7 :=
by
  sorry

end NUMINAMATH_GPT_remainder_division_l127_12712


namespace NUMINAMATH_GPT_percentage_increase_numerator_l127_12751

variable (N D : ℝ) (P : ℝ)
variable (h1 : N / D = 0.75)
variable (h2 : (N * (1 + P / 100)) / (D * 0.92) = 15 / 16)

theorem percentage_increase_numerator :
  P = 15 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_numerator_l127_12751


namespace NUMINAMATH_GPT_rotated_and_shifted_line_eq_l127_12722

theorem rotated_and_shifted_line_eq :
  let rotate_line_90 (x y : ℝ) := ( -y, x )
  let shift_right (x y : ℝ) := (x + 1, y)
  ∃ (new_a new_b new_c : ℝ), 
  (∀ (x y : ℝ), (y = 3 * x → x * new_a + y * new_b + new_c = 0)) ∧ 
  (new_a = 1) ∧ (new_b = 3) ∧ (new_c = -1) := by
  sorry

end NUMINAMATH_GPT_rotated_and_shifted_line_eq_l127_12722


namespace NUMINAMATH_GPT_matrix_sum_correct_l127_12792

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![4, 3], ![-2, 1]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![-1, 5], ![8, -3]]
def C : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 8], ![6, -2]]

theorem matrix_sum_correct : A + B = C := by
  sorry

end NUMINAMATH_GPT_matrix_sum_correct_l127_12792


namespace NUMINAMATH_GPT_math_problem_l127_12797

variable (x Q : ℝ)

theorem math_problem (h : 5 * (6 * x - 3 * Real.pi) = Q) :
  15 * (18 * x - 9 * Real.pi) = 9 * Q := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l127_12797


namespace NUMINAMATH_GPT_max_sum_of_factors_l127_12763

theorem max_sum_of_factors (a b : ℕ) (h : a * b = 42) : a + b ≤ 43 :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_max_sum_of_factors_l127_12763


namespace NUMINAMATH_GPT_sequence_sum_l127_12783

theorem sequence_sum (S : ℕ → ℕ) (a : ℕ → ℕ) : 
  (∀ n, S n = 2^n) →
  (a 1 = S 1) ∧ (∀ n, n ≥ 2 → a n = S n - S (n-1)) →
  a 3 + a 4 = 12 :=
by
  sorry

end NUMINAMATH_GPT_sequence_sum_l127_12783


namespace NUMINAMATH_GPT_no_solution_condition_l127_12759

theorem no_solution_condition (m : ℝ) : (∀ x : ℝ, (3 * x - m) / (x - 2) ≠ 1) → m = 6 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_condition_l127_12759


namespace NUMINAMATH_GPT_bakery_new_cakes_count_l127_12765

def cakes_sold := 91
def more_cakes_bought := 63

theorem bakery_new_cakes_count : (91 + 63) = 154 :=
by
  sorry

end NUMINAMATH_GPT_bakery_new_cakes_count_l127_12765


namespace NUMINAMATH_GPT_problem_1_problem_2_l127_12766

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (1 / 2) * a * x^2 + 2 * x
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f x - g x a
noncomputable def h' (x : ℝ) (a : ℝ) : ℝ := (1 / x) - a * x - 2
noncomputable def G (x : ℝ) : ℝ := ((1 / x) - 1) ^ 2 - 1

theorem problem_1 (a : ℝ): 
  (∃ x : ℝ, 0 < x ∧ h' x a < 0) ↔ a > -1 :=
by sorry

theorem problem_2 (a : ℝ):
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → h' x a ≤ 0) ↔ a ≥ -(7 / 16) :=
by sorry

end NUMINAMATH_GPT_problem_1_problem_2_l127_12766


namespace NUMINAMATH_GPT_maximum_value_at_2001_l127_12735
noncomputable def a_n (n : ℕ) : ℝ := n^2 / (1.001^n)

theorem maximum_value_at_2001 : ∃ n : ℕ, n = 2001 ∧ ∀ k : ℕ, a_n k ≤ a_n 2001 := by
  sorry

end NUMINAMATH_GPT_maximum_value_at_2001_l127_12735


namespace NUMINAMATH_GPT_smallest_integer_k_no_real_roots_l127_12772

def quadratic_no_real_roots (a b c : ℝ) : Prop := b^2 - 4 * a * c < 0

theorem smallest_integer_k_no_real_roots :
  ∃ k : ℤ, (∀ x : ℝ, quadratic_no_real_roots (2 * k - 1) (-8) 6) ∧ (k = 2) :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_k_no_real_roots_l127_12772


namespace NUMINAMATH_GPT_no_solution_iff_discriminant_l127_12724

theorem no_solution_iff_discriminant (k : ℝ) : 
  (∀ x : ℝ, x^2 + k*x + 1 ≥ 0) ↔ -2 ≤ k ∧ k ≤ 2 := by
  sorry

end NUMINAMATH_GPT_no_solution_iff_discriminant_l127_12724


namespace NUMINAMATH_GPT_values_of_x_l127_12741

def f (x : ℝ) : ℝ := x^2 - 5 * x

theorem values_of_x (x : ℝ) : f (f x) = f x → x = 0 ∨ x = -2 ∨ x = 5 ∨ x = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_values_of_x_l127_12741


namespace NUMINAMATH_GPT_area_of_PDCE_l127_12743

/-- A theorem to prove the area of quadrilateral PDCE given conditions in triangle ABC. -/
theorem area_of_PDCE
  (ABC_area : ℝ)
  (BD_to_CD_ratio : ℝ)
  (E_is_midpoint : Prop)
  (AD_intersects_BE : Prop)
  (P : Prop)
  (area_PDCE : ℝ) :
  (ABC_area = 1) →
  (BD_to_CD_ratio = 2 / 1) →
  E_is_midpoint →
  AD_intersects_BE →
  ∃ P, P →
    area_PDCE = 7 / 30 :=
by sorry

end NUMINAMATH_GPT_area_of_PDCE_l127_12743


namespace NUMINAMATH_GPT_georg_can_identify_fake_coins_l127_12711

theorem georg_can_identify_fake_coins :
  ∀ (coins : ℕ) (baron : ℕ → ℕ → ℕ) (queries : ℕ),
    coins = 100 →
    ∃ (fake_count : ℕ → ℕ) (exaggeration : ℕ),
      (∀ group_size : ℕ, 10 ≤ group_size ∧ group_size ≤ 20) →
      (∀ (show_coins : ℕ), show_coins ≤ group_size → fake_count show_coins = baron show_coins exaggeration) →
      queries < 120 :=
by
  sorry

end NUMINAMATH_GPT_georg_can_identify_fake_coins_l127_12711


namespace NUMINAMATH_GPT_triangle_inequality_part_a_triangle_inequality_part_b_l127_12708

variable {a b c S : ℝ}

/-- Part (a): Prove that for any triangle ABC, the inequality a^2 + b^2 + c^2 ≥ 4 √3 S holds
    where equality holds if and only if ABC is an equilateral triangle. -/
theorem triangle_inequality_part_a (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (area_S : S > 0) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S :=
sorry

/-- Part (b): Prove that for any triangle ABC,
    the inequality a^2 + b^2 + c^2 - (a - b)^2 - (b - c)^2 - (c - a)^2 ≥ 4 √3 S
    holds where equality also holds if and only if a = b = c. -/
theorem triangle_inequality_part_b (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (area_S : S > 0) :
  a^2 + b^2 + c^2 - (a - b)^2 - (b - c)^2 - (c - a)^2 ≥ 4 * Real.sqrt 3 * S :=
sorry

end NUMINAMATH_GPT_triangle_inequality_part_a_triangle_inequality_part_b_l127_12708


namespace NUMINAMATH_GPT_age_difference_l127_12752

theorem age_difference (x : ℕ) (older_age younger_age : ℕ) 
  (h1 : 3 * x = older_age)
  (h2 : 2 * x = younger_age)
  (h3 : older_age + younger_age = 60) : 
  older_age - younger_age = 12 := 
by
  sorry

end NUMINAMATH_GPT_age_difference_l127_12752


namespace NUMINAMATH_GPT_sam_seashell_count_l127_12775

/-!
# Problem statement:
-/
def initialSeashells := 35
def seashellsGivenToJoan := 18
def seashellsFoundToday := 20
def seashellsGivenToTom := 5

/-!
# Proof goal: Prove that the current number of seashells Sam has is 32.
-/
theorem sam_seashell_count :
  initialSeashells - seashellsGivenToJoan + seashellsFoundToday - seashellsGivenToTom = 32 :=
  sorry

end NUMINAMATH_GPT_sam_seashell_count_l127_12775


namespace NUMINAMATH_GPT_percentage_error_in_square_area_l127_12703

-- Given an error of 1% in excess while measuring the side of a square,
-- prove that the percentage of error in the calculated area of the square is 2.01%.

theorem percentage_error_in_square_area (s : ℝ) (h : s ≠ 0) :
  let measured_side := 1.01 * s
  let actual_area := s ^ 2
  let calculated_area := (1.01 * s) ^ 2
  let error_in_area := calculated_area - actual_area
  let percentage_error := (error_in_area / actual_area) * 100
  percentage_error = 2.01 :=
by {
  let measured_side := 1.01 * s;
  let actual_area := s ^ 2;
  let calculated_area := (1.01 * s) ^ 2;
  let error_in_area := calculated_area - actual_area;
  let percentage_error := (error_in_area / actual_area) * 100;
  sorry
}

end NUMINAMATH_GPT_percentage_error_in_square_area_l127_12703


namespace NUMINAMATH_GPT_smallest_N_value_l127_12768

theorem smallest_N_value (a b c d : ℕ)
  (h1 : gcd a b = 1 ∧ gcd a c = 2 ∧ gcd a d = 4 ∧ gcd b c = 5 ∧ gcd b d = 3 ∧ gcd c d = N)
  (h2 : N > 5) : N = 14 := sorry

end NUMINAMATH_GPT_smallest_N_value_l127_12768


namespace NUMINAMATH_GPT_average_percentage_of_15_students_l127_12702

open Real

theorem average_percentage_of_15_students :
  ∀ (x : ℝ),
  (15 + 10 = 25) →
  (10 * 90 = 900) →
  (25 * 84 = 2100) →
  (15 * x + 900 = 2100) →
  x = 80 :=
by
  intro x h_sum h_10_avg h_25_avg h_total
  sorry

end NUMINAMATH_GPT_average_percentage_of_15_students_l127_12702


namespace NUMINAMATH_GPT_janet_daily_search_time_l127_12742

-- Define the conditions
def minutes_looking_for_keys_per_day (x : ℕ) := 
  let total_time_per_day := x + 3
  let total_time_per_week := 7 * total_time_per_day
  total_time_per_week = 77

-- State the theorem
theorem janet_daily_search_time : 
  ∃ x : ℕ, minutes_looking_for_keys_per_day x ∧ x = 8 := by
  sorry

end NUMINAMATH_GPT_janet_daily_search_time_l127_12742


namespace NUMINAMATH_GPT_sum_of_cubes_divisible_by_9_l127_12778

theorem sum_of_cubes_divisible_by_9 (n : ℕ) : 9 ∣ (n^3 + (n + 1)^3 + (n + 2)^3) := 
  sorry

end NUMINAMATH_GPT_sum_of_cubes_divisible_by_9_l127_12778


namespace NUMINAMATH_GPT_cars_to_sell_l127_12728

theorem cars_to_sell (n : ℕ) 
  (h1 : ∀ c, c ∈ {c' : ℕ | c' ≤ n} → ∃ m, m = 3)
  (h2 : ∀ c, c ∈ {c' : ℕ | c' ≤ n} → c ∈ {c' : ℕ | c' < 3})
  (h3 : 15 * 3 = 45)
  (h4 : ∀ n, n * 3 = 45 → n = 15):
  n = 15 := 
  by
    have n_eq: n * 3 = 45 := sorry
    exact h4 n n_eq

end NUMINAMATH_GPT_cars_to_sell_l127_12728


namespace NUMINAMATH_GPT_daps_equiv_dirps_l127_12739

noncomputable def dops_equiv_daps : ℝ := 5 / 4
noncomputable def dips_equiv_dops : ℝ := 3 / 10
noncomputable def dirps_equiv_dips : ℝ := 2

theorem daps_equiv_dirps (n : ℝ) : 20 = (dops_equiv_daps * dips_equiv_dops * dirps_equiv_dips) * n → n = 15 :=
by sorry

end NUMINAMATH_GPT_daps_equiv_dirps_l127_12739


namespace NUMINAMATH_GPT_converse_xy_implies_x_is_true_l127_12709

/-- Prove that the converse of the proposition "If \(xy = 0\), then \(x = 0\)" is true. -/
theorem converse_xy_implies_x_is_true {x y : ℝ} (h : x = 0) : x * y = 0 :=
by sorry

end NUMINAMATH_GPT_converse_xy_implies_x_is_true_l127_12709


namespace NUMINAMATH_GPT_sufficient_condition_perpendicular_l127_12720

variables {Plane Line : Type}
variables (l : Line) (α β : Plane)

-- Definitions for perpendicularity and parallelism
def perp (l : Line) (α : Plane) : Prop := sorry
def parallel (α β : Plane) : Prop := sorry

theorem sufficient_condition_perpendicular
  (h1 : perp l α) 
  (h2 : parallel α β) : 
  perp l β :=
sorry

end NUMINAMATH_GPT_sufficient_condition_perpendicular_l127_12720


namespace NUMINAMATH_GPT_probability_log_interval_l127_12733

open Set Real

noncomputable def probability_in_interval (a b c d : ℝ) (I J : Set ℝ) := 
  (b - a) / (d - c)

theorem probability_log_interval : 
  probability_in_interval 2 4 0 6 (Icc 0 6) (Ioo 2 4) = 1 / 3 := 
sorry

end NUMINAMATH_GPT_probability_log_interval_l127_12733


namespace NUMINAMATH_GPT_minimum_sum_of_reciprocals_l127_12749

open BigOperators

theorem minimum_sum_of_reciprocals (b : Fin 15 → ℝ) (h_pos : ∀ i, 0 < b i)
    (h_sum : ∑ i, b i = 1) :
    ∑ i, 1 / (b i) ≥ 225 := sorry

end NUMINAMATH_GPT_minimum_sum_of_reciprocals_l127_12749


namespace NUMINAMATH_GPT_sum_of_nonneg_numbers_ineq_l127_12748

theorem sum_of_nonneg_numbers_ineq
  (a b c d : ℝ)
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0)
  (h_sum : a + b + c + d = 4) :
  (a * b + c * d) * (a * c + b * d) * (a * d + b * c) ≤ 8 := sorry

end NUMINAMATH_GPT_sum_of_nonneg_numbers_ineq_l127_12748


namespace NUMINAMATH_GPT_banana_permutations_l127_12732

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end NUMINAMATH_GPT_banana_permutations_l127_12732


namespace NUMINAMATH_GPT_simplify_expression_l127_12730

theorem simplify_expression (y : ℝ) : (3 * y^4)^4 = 81 * y^16 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l127_12730


namespace NUMINAMATH_GPT_lucy_total_fish_l127_12761

theorem lucy_total_fish (current fish_needed : ℕ) (h1 : current = 212) (h2 : fish_needed = 68) : 
  current + fish_needed = 280 := 
by
  sorry

end NUMINAMATH_GPT_lucy_total_fish_l127_12761


namespace NUMINAMATH_GPT_graveling_cost_l127_12787

theorem graveling_cost
  (length_lawn : ℝ) (width_lawn : ℝ)
  (width_road : ℝ)
  (cost_per_sq_m : ℝ)
  (h1: length_lawn = 80) (h2: width_lawn = 40) (h3: width_road = 10) (h4: cost_per_sq_m = 3) :
  (length_lawn * width_road + width_lawn * width_road - width_road * width_road) * cost_per_sq_m = 3900 := 
by
  sorry

end NUMINAMATH_GPT_graveling_cost_l127_12787


namespace NUMINAMATH_GPT_gcd_136_1275_l127_12726

theorem gcd_136_1275 : Nat.gcd 136 1275 = 17 := by
sorry

end NUMINAMATH_GPT_gcd_136_1275_l127_12726


namespace NUMINAMATH_GPT_sujis_age_l127_12719

theorem sujis_age (x : ℕ) (Abi Suji : ℕ)
  (h1 : Abi = 5 * x)
  (h2 : Suji = 4 * x)
  (h3 : (Abi + 3) / (Suji + 3) = 11 / 9) : 
  Suji = 24 := 
by 
  sorry

end NUMINAMATH_GPT_sujis_age_l127_12719


namespace NUMINAMATH_GPT_max_k_inequality_l127_12758

open Real

theorem max_k_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a + b) * (a * b + 1) * (b + 1) ≥ (27 / 4) * a * b^2 :=
by
  sorry

end NUMINAMATH_GPT_max_k_inequality_l127_12758


namespace NUMINAMATH_GPT_cats_remaining_proof_l127_12781

def initial_siamese : ℕ := 38
def initial_house : ℕ := 25
def sold_cats : ℕ := 45

def total_cats (s : ℕ) (h : ℕ) : ℕ := s + h
def remaining_cats (total : ℕ) (sold : ℕ) : ℕ := total - sold

theorem cats_remaining_proof : remaining_cats (total_cats initial_siamese initial_house) sold_cats = 18 :=
by
  sorry

end NUMINAMATH_GPT_cats_remaining_proof_l127_12781


namespace NUMINAMATH_GPT_earnings_percentage_difference_l127_12716

-- Defining the conditions
def MikeEarnings : ℕ := 12
def PhilEarnings : ℕ := 6

-- Proving the percentage difference
theorem earnings_percentage_difference :
  ((MikeEarnings - PhilEarnings: ℕ) * 100 / MikeEarnings = 50) :=
by 
  sorry

end NUMINAMATH_GPT_earnings_percentage_difference_l127_12716


namespace NUMINAMATH_GPT_hangar_length_l127_12767

-- Define the conditions
def num_planes := 7
def length_per_plane := 40 -- in feet

-- Define the main theorem to be proven
theorem hangar_length : num_planes * length_per_plane = 280 := by
  -- Proof omitted with sorry
  sorry

end NUMINAMATH_GPT_hangar_length_l127_12767


namespace NUMINAMATH_GPT_find_AX_length_l127_12729

noncomputable def AX_length (AC BC BX : ℕ) : ℚ :=
AC * (BX / BC)

theorem find_AX_length :
  let AC := 25
  let BC := 35
  let BX := 30
  AX_length AC BC BX = 150 / 7 :=
by
  -- proof is omitted using 'sorry'
  sorry

end NUMINAMATH_GPT_find_AX_length_l127_12729


namespace NUMINAMATH_GPT_Trevor_future_age_when_brother_is_three_times_now_l127_12704

def Trevor_current_age := 11
def Brother_current_age := 20

theorem Trevor_future_age_when_brother_is_three_times_now :
  ∃ (X : ℕ), Brother_current_age + (X - Trevor_current_age) = 3 * Trevor_current_age :=
by
  use 24
  sorry

end NUMINAMATH_GPT_Trevor_future_age_when_brother_is_three_times_now_l127_12704


namespace NUMINAMATH_GPT_find_f_neg_9_over_2_l127_12753

noncomputable def f : ℝ → ℝ
| x => if 0 ≤ x ∧ x ≤ 1 then 2^x else sorry

theorem find_f_neg_9_over_2
  (hf_even : ∀ x : ℝ, f (-x) = f x)
  (hf_periodic : ∀ x : ℝ, f (x + 2) = f x)
  (hf_definition : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → f x = 2^x) :
  f (-9 / 2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_find_f_neg_9_over_2_l127_12753


namespace NUMINAMATH_GPT_func_symmetry_monotonicity_range_of_m_l127_12790

open Real

theorem func_symmetry_monotonicity (f : ℝ → ℝ)
  (h1 : ∀ x, f (3 + x) = f (1 - x))
  (h2 : ∀ x1 x2, 2 < x1 → 2 < x2 → (f x1 - f x2) / (x1 - x2) > 0) :
  (∀ x, f (2 + x) = f (2 - x)) ∧
  (∀ x1 x2, (x1 > 2 ∧ x2 > 2 → f x1 < f x2 → x1 < x2) ∧
            (x2 > 2 ∧ x1 > x2 → f x2 < f x1 → x2 < x1)) := 
sorry

theorem range_of_m (f : ℝ → ℝ)
  (h : ∀ θ : ℝ, f (cos θ ^ 2 + 2 * (m : ℝ) ^ 2 + 2) < f (sin θ + m ^ 2 - 3 * m - 2)) :
  ∀ m, (3 - sqrt 42) / 6 < m ∧ m < (3 + sqrt 42) / 6 :=
sorry

end NUMINAMATH_GPT_func_symmetry_monotonicity_range_of_m_l127_12790


namespace NUMINAMATH_GPT_alphabet_letters_l127_12738

theorem alphabet_letters (DS S_only Total D_only : ℕ) 
  (h_DS : DS = 9) 
  (h_S_only : S_only = 24) 
  (h_Total : Total = 40) 
  (h_eq : Total = D_only + S_only + DS) 
  : D_only = 7 := 
by
  sorry

end NUMINAMATH_GPT_alphabet_letters_l127_12738


namespace NUMINAMATH_GPT_books_more_than_figures_l127_12773

-- Definitions of initial conditions
def initial_action_figures := 2
def initial_books := 10
def added_action_figures := 4

-- Problem statement to prove
theorem books_more_than_figures :
  initial_books - (initial_action_figures + added_action_figures) = 4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_books_more_than_figures_l127_12773


namespace NUMINAMATH_GPT_third_butcher_delivered_8_packages_l127_12788

variables (x y z t1 t2 t3 : ℕ)

-- Given Conditions
axiom h1 : x = 10
axiom h2 : y = 7
axiom h3 : 4 * x + 4 * y + 4 * z = 100
axiom t1_time : t1 = 8
axiom t2_time : t2 = 10
axiom t3_time : t3 = 18

-- Proof Problem
theorem third_butcher_delivered_8_packages :
  z = 8 :=
by
  -- proof to be filled
  sorry

end NUMINAMATH_GPT_third_butcher_delivered_8_packages_l127_12788


namespace NUMINAMATH_GPT_inequality_proof_l127_12707

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ( (2 + x) / (1 + x) )^2 + ( (2 + y) / (1 + y) )^2 ≥ 9 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l127_12707


namespace NUMINAMATH_GPT_sum_first_8_terms_eq_8_l127_12791

noncomputable def arithmetic_sequence_sum (n : ℕ) (a₁ d : ℤ) : ℤ :=
  n / 2 * (2 * a₁ + (n - 1) * d)

theorem sum_first_8_terms_eq_8
  (a : ℕ → ℤ)
  (h_arith_seq : ∀ n : ℕ, a (n + 1) = a 1 + ↑n * d)
  (h_a1 : a 1 = 8)
  (h_a4_a6 : a 4 + a 6 = 0) :
  arithmetic_sequence_sum 8 8 (-2) = 8 := 
by
  sorry

end NUMINAMATH_GPT_sum_first_8_terms_eq_8_l127_12791
