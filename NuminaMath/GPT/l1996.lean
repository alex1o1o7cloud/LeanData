import Mathlib

namespace NUMINAMATH_GPT_find_x_l1996_199655

variables (x : ℝ)

theorem find_x : (x / 4) * 12 = 9 → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1996_199655


namespace NUMINAMATH_GPT_bounded_expression_l1996_199671

theorem bounded_expression (x y : ℝ) (h : 3 * x^2 + 2 * y^2 ≤ 6) : 2 * x + y ≤ Real.sqrt 11 :=
sorry

end NUMINAMATH_GPT_bounded_expression_l1996_199671


namespace NUMINAMATH_GPT_find_digits_l1996_199648

theorem find_digits (A B C D : ℕ) (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
(h2 : 1 ≤ A ∧ A ≤ 9)
(h3 : 1 ≤ B ∧ B ≤ 9)
(h4 : 1 ≤ C ∧ C ≤ 9)
(h5 : 1 ≤ D ∧ D ≤ 9)
(h6 : (10 * A + B) * (10 * C + B) = 111 * D)
(h7 : (10 * A + B) < (10 * C + B)) :
A = 2 ∧ B = 7 ∧ C = 3 ∧ D = 9 :=
sorry

end NUMINAMATH_GPT_find_digits_l1996_199648


namespace NUMINAMATH_GPT_green_peaches_are_six_l1996_199656

/-- There are 5 red peaches in the basket. -/
def red_peaches : ℕ := 5

/-- There are 14 yellow peaches in the basket. -/
def yellow_peaches : ℕ := 14

/-- There are total of 20 green and yellow peaches in the basket. -/
def green_and_yellow_peaches : ℕ := 20

/-- The number of green peaches is calculated as the difference between the total number of green and yellow peaches and the number of yellow peaches. -/
theorem green_peaches_are_six :
  (green_and_yellow_peaches - yellow_peaches) = 6 :=
by
  sorry

end NUMINAMATH_GPT_green_peaches_are_six_l1996_199656


namespace NUMINAMATH_GPT_oranges_per_box_calculation_l1996_199616

def total_oranges : ℕ := 2650
def total_boxes : ℕ := 265

theorem oranges_per_box_calculation (h : total_oranges % total_boxes = 0) : total_oranges / total_boxes = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_oranges_per_box_calculation_l1996_199616


namespace NUMINAMATH_GPT_problem1_problem2_l1996_199614

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin x * (Real.sqrt 3 * Real.cos x + Real.sin x) - 2

-- Problem 1
theorem problem1 (α : ℝ) (hα1 : Real.sin α = -1 / 2) (hα2 : Real.cos α = Real.sqrt 3 / 2) :
  f α = -3 := sorry

-- Problem 2
theorem problem2 (h0 : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -2) :
  ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = -2 := sorry

end NUMINAMATH_GPT_problem1_problem2_l1996_199614


namespace NUMINAMATH_GPT_determinant_of_trig_matrix_l1996_199646

theorem determinant_of_trig_matrix (α β : ℝ) : 
  Matrix.det ![
    ![Real.sin α, Real.cos α], 
    ![Real.cos β, Real.sin β]
  ] = -Real.cos (α - β) :=
by sorry

end NUMINAMATH_GPT_determinant_of_trig_matrix_l1996_199646


namespace NUMINAMATH_GPT_calculate_total_cost_l1996_199697

def initial_price_orange : ℝ := 40
def initial_price_mango : ℝ := 50
def price_increase_percentage : ℝ := 0.15

-- Hypotheses
def new_price (initial_price : ℝ) (percentage_increase : ℝ) : ℝ :=
  initial_price * (1 + percentage_increase)

noncomputable def total_cost (num_oranges num_mangoes : ℕ) : ℝ :=
  (num_oranges * new_price initial_price_orange price_increase_percentage) +
  (num_mangoes * new_price initial_price_mango price_increase_percentage)

theorem calculate_total_cost :
  total_cost 10 10 = 1035 := by
  sorry

end NUMINAMATH_GPT_calculate_total_cost_l1996_199697


namespace NUMINAMATH_GPT_second_number_l1996_199612

theorem second_number (A B : ℝ) (h1 : 0.50 * A = 0.40 * B + 180) (h2 : A = 456) : B = 120 := 
by
  sorry

end NUMINAMATH_GPT_second_number_l1996_199612


namespace NUMINAMATH_GPT_sector_angle_sector_max_area_l1996_199667

-- Part (1)
theorem sector_angle (r l : ℝ) (α : ℝ) :
  2 * r + l = 10 → (1 / 2) * l * r = 4 → α = l / r → α = 1 / 2 :=
by
  intro h1 h2 h3
  sorry

-- Part (2)
theorem sector_max_area (r l : ℝ) (α S : ℝ) :
  2 * r + l = 40 → α = l / r → S = (1 / 2) * l * r →
  (∀ r' l' α' S', 2 * r' + l' = 40 → α' = l' / r' → S' = (1 / 2) * l' * r' → S ≤ S') →
  r = 10 ∧ α = 2 ∧ S = 100 :=
by
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_sector_angle_sector_max_area_l1996_199667


namespace NUMINAMATH_GPT_time_to_fill_pool_l1996_199609

theorem time_to_fill_pool :
  let R1 := 1
  let R2 := 1 / 2
  let R3 := 1 / 3
  let R4 := 1 / 4
  let R_total := R1 + R2 + R3 + R4
  let T := 1 / R_total
  T = 12 / 25 := 
by
  sorry

end NUMINAMATH_GPT_time_to_fill_pool_l1996_199609


namespace NUMINAMATH_GPT_total_students_in_circle_l1996_199653

theorem total_students_in_circle (N : ℕ) (h1 : ∃ (students : Finset ℕ), students.card = N)
  (h2 : ∃ (a b : ℕ), a = 6 ∧ b = 16 ∧ b - a = N / 2): N = 18 :=
by
  sorry

end NUMINAMATH_GPT_total_students_in_circle_l1996_199653


namespace NUMINAMATH_GPT_ratio_of_voters_l1996_199631

open Real

theorem ratio_of_voters (X Y : ℝ) (h1 : 0.64 * X + 0.46 * Y = 0.58 * (X + Y)) : X / Y = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_voters_l1996_199631


namespace NUMINAMATH_GPT_turnips_total_l1996_199693

def melanie_turnips := 139
def benny_turnips := 113

def total_turnips (melanie_turnips benny_turnips : Nat) : Nat :=
  melanie_turnips + benny_turnips

theorem turnips_total :
  total_turnips melanie_turnips benny_turnips = 252 :=
by
  sorry

end NUMINAMATH_GPT_turnips_total_l1996_199693


namespace NUMINAMATH_GPT_part1_part2_l1996_199690

def f (x : ℝ) : ℝ := abs (x - 5) + abs (x + 4)

theorem part1 (x : ℝ) : f x ≥ 12 ↔ x ≥ 13 / 2 ∨ x ≤ -11 / 2 :=
by
    sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x - 2 ^ (1 - 3 * a) - 1 ≥ 0) ↔ -2 / 3 ≤ a :=
by
    sorry

end NUMINAMATH_GPT_part1_part2_l1996_199690


namespace NUMINAMATH_GPT_shenzhen_vaccination_count_l1996_199683

theorem shenzhen_vaccination_count :
  2410000 = 2.41 * 10^6 :=
  sorry

end NUMINAMATH_GPT_shenzhen_vaccination_count_l1996_199683


namespace NUMINAMATH_GPT_first_candidate_votes_percentage_l1996_199627

theorem first_candidate_votes_percentage 
( total_votes : ℕ ) 
( second_candidate_votes : ℕ ) 
( P : ℕ ) 
( h1 : total_votes = 2400 ) 
( h2 : second_candidate_votes = 480 ) 
( h3 : (P/100 : ℝ) * total_votes + second_candidate_votes = total_votes ) : 
  P = 80 := 
sorry

end NUMINAMATH_GPT_first_candidate_votes_percentage_l1996_199627


namespace NUMINAMATH_GPT_axis_of_symmetry_of_quadratic_l1996_199676

theorem axis_of_symmetry_of_quadratic (m : ℝ) :
  (∀ x : ℝ, -x^2 + 2 * m * x - m^2 + 3 = -x^2 + 2 * m * x - m^2 + 3) ∧ (∃ x : ℝ, x + 2 = 0) → m = -2 :=
by
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_of_quadratic_l1996_199676


namespace NUMINAMATH_GPT_aluminum_iodide_mass_produced_l1996_199668

theorem aluminum_iodide_mass_produced
  (mass_Al : ℝ) -- the mass of Aluminum used
  (molar_mass_Al : ℝ) -- molar mass of Aluminum
  (molar_mass_AlI3 : ℝ) -- molar mass of Aluminum Iodide
  (reaction_eq : ∀ (moles_Al : ℝ) (moles_AlI3 : ℝ), 2 * moles_Al = 2 * moles_AlI3) -- reaction equation which indicates a 1:1 molar ratio
  (mass_Al_value : mass_Al = 25.0) 
  (molar_mass_Al_value : molar_mass_Al = 26.98) 
  (molar_mass_AlI3_value : molar_mass_AlI3 = 407.68) :
  ∃ mass_AlI3 : ℝ, mass_AlI3 = 377.52 := by
  sorry

end NUMINAMATH_GPT_aluminum_iodide_mass_produced_l1996_199668


namespace NUMINAMATH_GPT_ratio_seconds_l1996_199604

theorem ratio_seconds (x : ℕ) (h : 12 / x = 6 / 240) : x = 480 :=
sorry

end NUMINAMATH_GPT_ratio_seconds_l1996_199604


namespace NUMINAMATH_GPT_proportion_of_solution_x_in_mixture_l1996_199659

-- Definitions for the conditions in given problem
def solution_x_contains_perc_a : ℚ := 0.20
def solution_y_contains_perc_a : ℚ := 0.30
def solution_z_contains_perc_a : ℚ := 0.40

def solution_y_to_z_ratio : ℚ := 3 / 2
def final_mixture_perc_a : ℚ := 0.25

-- Proving the proportion of solution x in the mixture equals 9/14
theorem proportion_of_solution_x_in_mixture
  (x y z : ℚ) (k : ℚ) (hx : x = 9 * k) (hy : y = 3 * k) (hz : z = 2 * k) :
  solution_x_contains_perc_a * x + solution_y_contains_perc_a * y + solution_z_contains_perc_a * z
  = final_mixture_perc_a * (x + y + z) →
  x / (x + y + z) = 9 / 14 :=
by
  intros h
  -- leaving the proof as a placeholder
  sorry

end NUMINAMATH_GPT_proportion_of_solution_x_in_mixture_l1996_199659


namespace NUMINAMATH_GPT_find_a_for_unique_solution_l1996_199606

theorem find_a_for_unique_solution :
  ∃ a : ℝ, (∀ x : ℝ, 0 ≤ x^2 - a * x + a ∧ x^2 - a * x + a ≤ 1) ↔ a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_for_unique_solution_l1996_199606


namespace NUMINAMATH_GPT_moles_of_H2O_formed_l1996_199619

def NH4NO3 (n : ℕ) : Prop := n = 1
def NaOH (n : ℕ) : Prop := ∃ m : ℕ, m = n
def H2O (n : ℕ) : Prop := n = 1

theorem moles_of_H2O_formed :
  ∀ (n : ℕ), NH4NO3 1 → NaOH n → H2O 1 := 
by
  intros n hNH4NO3 hNaOH
  exact sorry

end NUMINAMATH_GPT_moles_of_H2O_formed_l1996_199619


namespace NUMINAMATH_GPT_largest_4_digit_congruent_to_17_mod_26_l1996_199672

theorem largest_4_digit_congruent_to_17_mod_26 :
  ∃ x, x < 10000 ∧ x ≥ 1000 ∧ x % 26 = 17 ∧ (∀ y, y < 10000 ∧ y ≥ 1000 ∧ y % 26 = 17 → y ≤ x) ∧ x = 9972 := 
by
  sorry

end NUMINAMATH_GPT_largest_4_digit_congruent_to_17_mod_26_l1996_199672


namespace NUMINAMATH_GPT_round_trip_and_car_percent_single_trip_and_motorcycle_percent_l1996_199635

noncomputable def totalPassengers := 100
noncomputable def roundTripPercent := 35
noncomputable def singleTripPercent := 100 - roundTripPercent

noncomputable def roundTripCarPercent := 40
noncomputable def roundTripMotorcyclePercent := 15
noncomputable def roundTripNoVehiclePercent := 60

noncomputable def singleTripCarPercent := 25
noncomputable def singleTripMotorcyclePercent := 10
noncomputable def singleTripNoVehiclePercent := 45

theorem round_trip_and_car_percent : 
  ((roundTripCarPercent / 100) * (roundTripPercent / 100) * totalPassengers) = 14 :=
by
  sorry

theorem single_trip_and_motorcycle_percent :
  ((singleTripMotorcyclePercent / 100) * (singleTripPercent / 100) * totalPassengers) = 6 :=
by
  sorry

end NUMINAMATH_GPT_round_trip_and_car_percent_single_trip_and_motorcycle_percent_l1996_199635


namespace NUMINAMATH_GPT_proof_problem_l1996_199636

variable {R : Type*} [LinearOrderedField R]

theorem proof_problem 
  (a1 a2 a3 b1 b2 b3 : R)
  (h1 : a1 < a2) (h2 : a2 < a3) (h3 : b1 < b2) (h4 : b2 < b3)
  (h_sum : a1 + a2 + a3 = b1 + b2 + b3)
  (h_pair_sum : a1 * a2 + a1 * a3 + a2 * a3 = b1 * b2 + b1 * b3 + b2 * b3)
  (h_a1_lt_b1 : a1 < b1) :
  (b2 < a2) ∧ (a3 < b3) ∧ (a1 * a2 * a3 < b1 * b2 * b3) ∧ ((1 - a1) * (1 - a2) * (1 - a3) > (1 - b1) * (1 - b2) * (1 - b3)) :=
by {
  sorry
}

end NUMINAMATH_GPT_proof_problem_l1996_199636


namespace NUMINAMATH_GPT_members_not_playing_any_sport_l1996_199686

theorem members_not_playing_any_sport {total_members badminton_players tennis_players both_players : ℕ}
  (h_total : total_members = 28)
  (h_badminton : badminton_players = 17)
  (h_tennis : tennis_players = 19)
  (h_both : both_players = 10) :
  total_members - (badminton_players + tennis_players - both_players) = 2 :=
by
  sorry

end NUMINAMATH_GPT_members_not_playing_any_sport_l1996_199686


namespace NUMINAMATH_GPT_ivan_years_l1996_199644

theorem ivan_years (years months weeks days hours : ℕ) (h1 : years = 48) (h2 : months = 48)
    (h3 : weeks = 48) (h4 : days = 48) (h5 : hours = 48) :
    (53 : ℕ) = (years + (months / 12) + ((weeks * 7 + days) / 365) + ((hours / 24) / 365)) := by
  sorry

end NUMINAMATH_GPT_ivan_years_l1996_199644


namespace NUMINAMATH_GPT_valid_common_ratios_count_l1996_199661

noncomputable def num_valid_common_ratios (a₁ : ℝ) (q : ℝ) : ℝ :=
  let a₅ := a₁ * q^4
  let a₃ := a₁ * q^2
  if 2 * a₅ = 4 * a₁ + (-2) * a₃ then 1 else 0

theorem valid_common_ratios_count (a₁ : ℝ) : 
  (num_valid_common_ratios a₁ 1) + (num_valid_common_ratios a₁ (-1)) = 2 :=
by sorry

end NUMINAMATH_GPT_valid_common_ratios_count_l1996_199661


namespace NUMINAMATH_GPT_right_triangle_shorter_leg_l1996_199605
-- Import all necessary libraries

-- Define the problem
theorem right_triangle_shorter_leg (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c = 65) (h4 : a^2 + b^2 = c^2) :
  a = 25 :=
sorry

end NUMINAMATH_GPT_right_triangle_shorter_leg_l1996_199605


namespace NUMINAMATH_GPT_nate_reading_percentage_l1996_199647

-- Given conditions
def total_pages := 400
def pages_to_read := 320

-- Calculate the number of pages he has already read
def pages_read := total_pages - pages_to_read

-- Prove the percentage of the book Nate has finished reading
theorem nate_reading_percentage : (pages_read / total_pages) * 100 = 20 := by
  sorry

end NUMINAMATH_GPT_nate_reading_percentage_l1996_199647


namespace NUMINAMATH_GPT_time_after_3108_hours_l1996_199652

/-- The current time is 3 o'clock. On a 12-hour clock, 
 what time will it be 3108 hours from now? -/
theorem time_after_3108_hours : (3 + 3108) % 12 = 3 := 
by
  sorry

end NUMINAMATH_GPT_time_after_3108_hours_l1996_199652


namespace NUMINAMATH_GPT_john_total_jury_duty_days_l1996_199682

-- Definitions based on the given conditions
def jury_selection_days : ℕ := 2
def trial_length_multiplier : ℕ := 4
def deliberation_days : ℕ := 6
def deliberation_hours_per_day : ℕ := 16
def hours_in_a_day : ℕ := 24

-- Define the total days John spends on jury duty
def total_days_on_jury_duty : ℕ :=
  jury_selection_days +
  (trial_length_multiplier * jury_selection_days) +
  (deliberation_days * deliberation_hours_per_day) / hours_in_a_day

-- The theorem to prove
theorem john_total_jury_duty_days : total_days_on_jury_duty = 14 := by
  sorry

end NUMINAMATH_GPT_john_total_jury_duty_days_l1996_199682


namespace NUMINAMATH_GPT_negation_of_prop_l1996_199651

theorem negation_of_prop (P : Prop) :
  (¬ ∀ x > 0, x - 1 ≥ Real.log x) ↔ ∃ x > 0, x - 1 < Real.log x :=
by
  sorry

end NUMINAMATH_GPT_negation_of_prop_l1996_199651


namespace NUMINAMATH_GPT_number_of_players_in_association_l1996_199633

-- Define the variables and conditions based on the given problem
def socks_cost : ℕ := 6
def tshirt_cost := socks_cost + 8
def hat_cost := tshirt_cost - 3
def total_expenditure : ℕ := 4950
def cost_per_player := 2 * (socks_cost + tshirt_cost + hat_cost)

-- The statement to prove
theorem number_of_players_in_association :
  total_expenditure / cost_per_player = 80 := by
  sorry

end NUMINAMATH_GPT_number_of_players_in_association_l1996_199633


namespace NUMINAMATH_GPT_total_spent_l1996_199696

def cost_sandwich : ℕ := 2
def cost_hamburger : ℕ := 2
def cost_hotdog : ℕ := 1
def cost_fruit_juice : ℕ := 2

def selene_sandwiches : ℕ := 3
def selene_fruit_juice : ℕ := 1
def tanya_hamburgers : ℕ := 2
def tanya_fruit_juice : ℕ := 2

def total_selene_spent : ℕ := (selene_sandwiches * cost_sandwich) + (selene_fruit_juice * cost_fruit_juice)
def total_tanya_spent : ℕ := (tanya_hamburgers * cost_hamburger) + (tanya_fruit_juice * cost_fruit_juice)

theorem total_spent : total_selene_spent + total_tanya_spent = 16 := by
  sorry

end NUMINAMATH_GPT_total_spent_l1996_199696


namespace NUMINAMATH_GPT_time_taken_l1996_199615

-- Define the function T which takes the number of cats, the number of rats, and returns the time in minutes
def T (n m : ℕ) : ℕ := if n = m then 4 else sorry

-- The theorem states that, given n cats and n rats, the time taken is 4 minutes
theorem time_taken (n : ℕ) : T n n = 4 :=
by simp [T]

end NUMINAMATH_GPT_time_taken_l1996_199615


namespace NUMINAMATH_GPT_aldehyde_formula_l1996_199608

-- Define the problem starting with necessary variables
variables (n : ℕ)

-- Given conditions
def general_formula_aldehyde (n : ℕ) : String :=
  "CₙH_{2n}O"

def mass_percent_hydrogen (n : ℕ) : ℚ :=
  (2 * n) / (14 * n + 16)

-- Given the percentage of hydrogen in the aldehyde
def given_hydrogen_percent : ℚ := 0.12

-- The main theorem
theorem aldehyde_formula :
  (exists n : ℕ, mass_percent_hydrogen n = given_hydrogen_percent ∧ n = 6) ->
  general_formula_aldehyde 6 = "C₆H_{12}O" :=
by
  sorry

end NUMINAMATH_GPT_aldehyde_formula_l1996_199608


namespace NUMINAMATH_GPT_product_of_x_values_l1996_199628

noncomputable def find_product_of_x : ℚ :=
  let x1 := -20
  let x2 := -20 / 7
  (x1 * x2)

theorem product_of_x_values :
  (∃ x : ℚ, abs (20 / x + 4) = 3) ->
  find_product_of_x = 400 / 7 :=
by
  sorry

end NUMINAMATH_GPT_product_of_x_values_l1996_199628


namespace NUMINAMATH_GPT_socks_selection_l1996_199665

theorem socks_selection :
  (Nat.choose 7 3) - (Nat.choose 6 3) = 15 :=
by sorry

end NUMINAMATH_GPT_socks_selection_l1996_199665


namespace NUMINAMATH_GPT_share_of_y_l1996_199641

theorem share_of_y (A y z : ℝ)
  (hx : y = 0.45 * A)
  (hz : z = 0.30 * A)
  (h_total : A + y + z = 140) :
  y = 36 := by
  sorry

end NUMINAMATH_GPT_share_of_y_l1996_199641


namespace NUMINAMATH_GPT_work_hours_together_l1996_199643

theorem work_hours_together (t : ℚ) :
  (1 / 9) * (9 : ℚ) = 1 ∧ (1 / 12) * (12 : ℚ) = 1 ∧
  (7 / 36) * t + (1 / 9) * (15 / 4) = 1 → t = 3 :=
by
  sorry

end NUMINAMATH_GPT_work_hours_together_l1996_199643


namespace NUMINAMATH_GPT_percentage_of_paycheck_went_to_taxes_l1996_199602

-- Definitions
def original_paycheck : ℝ := 125
def savings : ℝ := 20
def spend_percentage : ℝ := 0.80
def save_percentage : ℝ := 0.20

-- Statement that needs to be proved
theorem percentage_of_paycheck_went_to_taxes (T : ℝ) :
  (0.20 * (1 - T / 100) * original_paycheck = savings) → T = 20 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_paycheck_went_to_taxes_l1996_199602


namespace NUMINAMATH_GPT_biography_percentage_increase_l1996_199678

variable {T : ℝ}
variable (hT : T > 0 ∧ T ≤ 10000)
variable (B : ℝ := 0.20 * T)
variable (B' : ℝ := 0.32 * T)
variable (percentage_increase : ℝ := ((B' - B) / B) * 100)

theorem biography_percentage_increase :
  percentage_increase = 60 :=
by
  sorry

end NUMINAMATH_GPT_biography_percentage_increase_l1996_199678


namespace NUMINAMATH_GPT_problem_part_1_problem_part_2_l1996_199662

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin x, 3 / 2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)

noncomputable def tan_2x_when_parallel (x : ℝ) (h : (Real.sin x + 3 / 2 * Real.cos x = 0)) : Prop :=
    Real.tan (2 * x) = 12 / 5

noncomputable def f (x : ℝ) : ℝ :=
  let a := vector_a x
  let b := vector_b x
  (a.1 + b.1) * b.1 + (a.2 + b.2) * b.2

def range_f_on_interval : Prop :=
  ∀ x ∈ Set.Icc (-Real.pi / 2) 0, -Real.sqrt 2 / 2 ≤ f x ∧ f x ≤ 1 / 2

theorem problem_part_1 (x : ℝ) (h : (Real.sin x + 3 / 2 * Real.cos x = 0)) : Real.tan (2 * x) = 12 / 5 :=
by
  sorry

theorem problem_part_2 : range_f_on_interval :=
by
  sorry

end NUMINAMATH_GPT_problem_part_1_problem_part_2_l1996_199662


namespace NUMINAMATH_GPT_problem_1_problem_2_l1996_199679

noncomputable def distance_between_parallel_lines (A B C1 C2 : ℝ) : ℝ :=
  let numerator := |C1 - C2|
  let denominator := Real.sqrt (A^2 + B^2)
  numerator / denominator

noncomputable def distance_point_to_line (A B C x0 y0 : ℝ) : ℝ :=
  let numerator := |A * x0 + B * y0 + C|
  let denominator := Real.sqrt (A^2 + B^2)
  numerator / denominator

theorem problem_1 : distance_between_parallel_lines 2 1 (-1) 1 = 2 * Real.sqrt 5 / 5 :=
  by sorry

theorem problem_2 : distance_point_to_line 2 1 (-1) 0 2 = Real.sqrt 5 / 5 :=
  by sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1996_199679


namespace NUMINAMATH_GPT_pizza_payment_l1996_199657

theorem pizza_payment (n : ℕ) (cost : ℕ) (total : ℕ) 
  (h1 : n = 3) 
  (h2 : cost = 8) 
  (h3 : total = n * cost) : 
  total = 24 :=
by 
  rw [h1, h2] at h3 
  exact h3

end NUMINAMATH_GPT_pizza_payment_l1996_199657


namespace NUMINAMATH_GPT_negation_of_proposition_l1996_199694

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x^2 - x + 2 ≥ 0)) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1996_199694


namespace NUMINAMATH_GPT_fraction_meaningful_iff_x_ne_pm1_l1996_199664

theorem fraction_meaningful_iff_x_ne_pm1 (x : ℝ) : (x^2 - 1 ≠ 0) ↔ (x ≠ 1 ∧ x ≠ -1) :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_iff_x_ne_pm1_l1996_199664


namespace NUMINAMATH_GPT_like_terms_mn_eq_neg1_l1996_199654

variable (m n : ℤ)

theorem like_terms_mn_eq_neg1
  (hx : m + 3 = 4)
  (hy : n + 3 = 1) :
  m + n = -1 :=
sorry

end NUMINAMATH_GPT_like_terms_mn_eq_neg1_l1996_199654


namespace NUMINAMATH_GPT_total_cards_in_box_l1996_199670

-- Definitions based on conditions
def xiaoMingCountsFaster (m h : ℕ) := 6 * h = 4 * m
def xiaoHuaForgets (h1 h2 : ℕ) := h1 + h2 = 112
def finalCardLeft (t : ℕ) := t - 1 = 112

-- Main theorem stating that the total number of cards is 353
theorem total_cards_in_box : ∃ N : ℕ, 
    (∃ m h1 h2 : ℕ,
        xiaoMingCountsFaster m h1 ∧
        xiaoHuaForgets h1 h2 ∧
        finalCardLeft N) ∧
    N = 353 :=
sorry

end NUMINAMATH_GPT_total_cards_in_box_l1996_199670


namespace NUMINAMATH_GPT_probability_three_dice_sum_to_fourth_l1996_199675

-- Define the probability problem conditions
def total_outcomes : ℕ := 8^4
def favorable_outcomes : ℕ := 1120

-- Final probability for the problem
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

-- Lean statement for the proof problem
theorem probability_three_dice_sum_to_fourth :
  probability favorable_outcomes total_outcomes = 35 / 128 :=
by sorry

end NUMINAMATH_GPT_probability_three_dice_sum_to_fourth_l1996_199675


namespace NUMINAMATH_GPT_total_steps_l1996_199600

theorem total_steps (up_steps down_steps : ℕ) (h1 : up_steps = 567) (h2 : down_steps = 325) : up_steps + down_steps = 892 := by
  sorry

end NUMINAMATH_GPT_total_steps_l1996_199600


namespace NUMINAMATH_GPT_compute_usage_difference_l1996_199640

theorem compute_usage_difference
  (usage_last_week : ℕ)
  (usage_per_day : ℕ)
  (days_in_week : ℕ)
  (h1 : usage_last_week = 91)
  (h2 : usage_per_day = 8)
  (h3 : days_in_week = 7) :
  (usage_last_week - usage_per_day * days_in_week) = 35 :=
  sorry

end NUMINAMATH_GPT_compute_usage_difference_l1996_199640


namespace NUMINAMATH_GPT_maximum_value_m_l1996_199622

def f (x : ℝ) : ℝ := x^2 + 2*x + 1

noncomputable def exists_t_and_max_m (m : ℝ) : Prop :=
  ∃ t : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → f (x + t) ≤ x

theorem maximum_value_m : ∃ m : ℝ, exists_t_and_max_m m ∧ (∀ m' : ℝ, exists_t_and_max_m m' → m' ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_m_l1996_199622


namespace NUMINAMATH_GPT_eq_sum_of_factorial_fractions_l1996_199642

theorem eq_sum_of_factorial_fractions (b2 b3 b5 b6 b7 b8 : ℤ)
  (h2 : 0 ≤ b2 ∧ b2 < 2)
  (h3 : 0 ≤ b3 ∧ b3 < 3)
  (h5 : 0 ≤ b5 ∧ b5 < 5)
  (h6 : 0 ≤ b6 ∧ b6 < 6)
  (h7 : 0 ≤ b7 ∧ b7 < 7)
  (h8 : 0 ≤ b8 ∧ b8 < 8)
  (h_eq : (3 / 8 : ℚ) = (b2 / (2 * 1) + b3 / (3 * 2 * 1) + b5 / (5 * 4 * 3 * 2 * 1) +
                          b6 / (6 * 5 * 4 * 3 * 2 * 1) + b7 / (7 * 6 * 5 * 4 * 3 * 2 * 1) +
                          b8 / (8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) : ℚ)) :
  b2 + b3 + b5 + b6 + b7 + b8 = 12 :=
by
  sorry

end NUMINAMATH_GPT_eq_sum_of_factorial_fractions_l1996_199642


namespace NUMINAMATH_GPT_commute_time_absolute_difference_l1996_199691

theorem commute_time_absolute_difference (x y : ℝ)
  (h1 : (x + y + 10 + 11 + 9) / 5 = 10)
  (h2 : (x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2 = 10) :
  |x - y| = 4 :=
by
  sorry

end NUMINAMATH_GPT_commute_time_absolute_difference_l1996_199691


namespace NUMINAMATH_GPT_power_function_passes_through_1_1_l1996_199618

theorem power_function_passes_through_1_1 (n : ℝ) : (1 : ℝ) ^ n = 1 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_power_function_passes_through_1_1_l1996_199618


namespace NUMINAMATH_GPT_train_length_l1996_199607

noncomputable def speed_km_hr : ℝ := 60
noncomputable def time_seconds : ℝ := 36
noncomputable def speed_m_s := speed_km_hr * (5/18 : ℝ)
noncomputable def distance := speed_m_s * time_seconds

-- Theorem statement
theorem train_length : distance = 600.12 := by
  sorry

end NUMINAMATH_GPT_train_length_l1996_199607


namespace NUMINAMATH_GPT_cooper_age_l1996_199685

variable (Cooper Dante Maria : ℕ)

-- Conditions
def sum_of_ages : Prop := Cooper + Dante + Maria = 31
def dante_twice_cooper : Prop := Dante = 2 * Cooper
def maria_one_year_older : Prop := Maria = Dante + 1

theorem cooper_age (h1 : sum_of_ages Cooper Dante Maria) (h2 : dante_twice_cooper Cooper Dante) (h3 : maria_one_year_older Dante Maria) : Cooper = 6 :=
by
  sorry

end NUMINAMATH_GPT_cooper_age_l1996_199685


namespace NUMINAMATH_GPT_rectangular_sheet_integers_l1996_199639

noncomputable def at_least_one_integer (a b : ℝ) : Prop :=
  ∃ i : ℤ, a = i ∨ b = i

theorem rectangular_sheet_integers (a b : ℝ)
  (h_positive_a : a > 0)
  (h_positive_b : b > 0)
  (h_cut_lines : ∀ x y : ℝ, (∃ k : ℤ, x = k ∧ y = 1 ∨ y = k ∧ x = 1) → (∃ z : ℤ, x = z ∨ y = z)) :
  at_least_one_integer a b :=
sorry

end NUMINAMATH_GPT_rectangular_sheet_integers_l1996_199639


namespace NUMINAMATH_GPT_symmetric_point_coordinates_l1996_199634

theorem symmetric_point_coordinates (M : ℝ × ℝ) (N : ℝ × ℝ) (hM : M = (1, -2)) (h_sym : N = (-M.1, -M.2)) :
  N = (-1, 2) :=
by sorry

end NUMINAMATH_GPT_symmetric_point_coordinates_l1996_199634


namespace NUMINAMATH_GPT_factorial_fraction_eq_zero_l1996_199680

theorem factorial_fraction_eq_zero :
  ((5 * (Nat.factorial 7) - 35 * (Nat.factorial 6)) / Nat.factorial 8 = 0) :=
by
  sorry

end NUMINAMATH_GPT_factorial_fraction_eq_zero_l1996_199680


namespace NUMINAMATH_GPT_total_amount_shared_l1996_199630

theorem total_amount_shared (a b c : ℕ) (h_ratio : a = 3 * b / 5 ∧ c = 9 * b / 5) (h_b : b = 50) : a + b + c = 170 :=
by sorry

end NUMINAMATH_GPT_total_amount_shared_l1996_199630


namespace NUMINAMATH_GPT_carrots_weight_l1996_199692

theorem carrots_weight (carrots_bed1: ℕ) (carrots_bed2: ℕ) (carrots_bed3: ℕ) (carrots_per_pound: ℕ)
  (h_bed1: carrots_bed1 = 55)
  (h_bed2: carrots_bed2 = 101)
  (h_bed3: carrots_bed3 = 78)
  (h_c_per_p: carrots_per_pound = 6) :
  (carrots_bed1 + carrots_bed2 + carrots_bed3) / carrots_per_pound = 39 := by
  sorry

end NUMINAMATH_GPT_carrots_weight_l1996_199692


namespace NUMINAMATH_GPT_cars_meet_first_time_l1996_199669

-- Definitions based on conditions
def car (t : ℕ) (v : ℕ) : ℕ := t * v
def car_meet (t : ℕ) (v1 v2 : ℕ) : Prop := ∃ n, v1 * t + v2 * t = n

-- Given conditions
variables (v_A v_B v_C v_D : ℕ) (pairwise_different : v_A ≠ v_B ∧ v_B ≠ v_C ∧ v_C ≠ v_D ∧ v_D ≠ v_A)
variables (t1 t2 t3 : ℕ) (time_AC : t1 = 7) (time_BD : t1 = 7) (time_AB : t2 = 53)
variables (condition1 : car_meet t1 v_A v_C) (condition2 : car_meet t1 v_B v_D)
variables (condition3 : ∃ k, (v_A - v_B) * t2 = k)

-- Theorem statement
theorem cars_meet_first_time : ∃ t, (t = 371) := sorry

end NUMINAMATH_GPT_cars_meet_first_time_l1996_199669


namespace NUMINAMATH_GPT_deepak_wife_speed_l1996_199620

-- Definitions and conditions
def track_circumference_km : ℝ := 0.66
def deepak_speed_kmh : ℝ := 4.5
def time_to_meet_hr : ℝ := 0.08

-- Theorem statement
theorem deepak_wife_speed
  (track_circumference_km : ℝ)
  (deepak_speed_kmh : ℝ)
  (time_to_meet_hr : ℝ)
  (deepak_distance : ℝ := deepak_speed_kmh * time_to_meet_hr)
  (wife_distance : ℝ := track_circumference_km - deepak_distance)
  (wife_speed_kmh : ℝ := wife_distance / time_to_meet_hr) : 
  wife_speed_kmh = 3.75 :=
sorry

end NUMINAMATH_GPT_deepak_wife_speed_l1996_199620


namespace NUMINAMATH_GPT_infinite_solutions_iff_c_is_5_over_2_l1996_199674

theorem infinite_solutions_iff_c_is_5_over_2 (c : ℝ) :
  (∀ y : ℝ, 3 * (2 + 2 * c * y) = 15 * y + 6) ↔ c = 5 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_infinite_solutions_iff_c_is_5_over_2_l1996_199674


namespace NUMINAMATH_GPT_centroid_coordinates_satisfy_l1996_199617

noncomputable def P : ℝ × ℝ := (2, 5)
noncomputable def Q : ℝ × ℝ := (-1, 3)
noncomputable def R : ℝ × ℝ := (4, -2)

noncomputable def S : ℝ × ℝ := (
  (P.1 + Q.1 + R.1) / 3,
  (P.2 + Q.2 + R.2) / 3
)

theorem centroid_coordinates_satisfy :
  4 * S.1 + 3 * S.2 = 38 / 3 :=
by
  -- Proof will be added here
  sorry

end NUMINAMATH_GPT_centroid_coordinates_satisfy_l1996_199617


namespace NUMINAMATH_GPT_double_espresso_cost_l1996_199621

-- Define the cost of coffee, days, and total spent as constants
def iced_coffee : ℝ := 2.5
def total_days : ℝ := 20
def total_spent : ℝ := 110

-- Define the cost of double espresso as variable E
variable (E : ℝ)

-- The proposition to prove
theorem double_espresso_cost : (total_days * (E + iced_coffee) = total_spent) → (E = 3) :=
by
  sorry

end NUMINAMATH_GPT_double_espresso_cost_l1996_199621


namespace NUMINAMATH_GPT_total_wage_is_75_l1996_199610

noncomputable def wages_total (man_wage : ℕ) : ℕ :=
  let men := 5
  let women := (5 : ℕ)
  let boys := 8
  (man_wage * men) + (man_wage * men) + (man_wage * men)

theorem total_wage_is_75
  (W : ℕ)
  (man_wage : ℕ := 5)
  (h1 : 5 = W) 
  (h2 : W = 8) 
  : wages_total man_wage = 75 := by
  sorry

end NUMINAMATH_GPT_total_wage_is_75_l1996_199610


namespace NUMINAMATH_GPT_num_ways_to_use_100_yuan_l1996_199603

noncomputable def x : ℕ → ℝ
| 0       => 0
| 1       => 1
| 2       => 3
| (n + 3) => x (n + 2) + 2 * x (n + 1)

theorem num_ways_to_use_100_yuan :
  x 100 = (1 / 3) * (2 ^ 101 + 1) :=
sorry

end NUMINAMATH_GPT_num_ways_to_use_100_yuan_l1996_199603


namespace NUMINAMATH_GPT_abs_eq_4_l1996_199698

theorem abs_eq_4 (a : ℝ) : |a| = 4 ↔ a = 4 ∨ a = -4 :=
by
  sorry

end NUMINAMATH_GPT_abs_eq_4_l1996_199698


namespace NUMINAMATH_GPT_rhombus_area_l1996_199688

theorem rhombus_area (a b : ℝ) (h : (a - 1) ^ 2 + Real.sqrt (b - 4) = 0) : (1 / 2) * a * b = 2 := by
  sorry

end NUMINAMATH_GPT_rhombus_area_l1996_199688


namespace NUMINAMATH_GPT_factorizations_of_2079_l1996_199658

theorem factorizations_of_2079 :
  ∃ (a b : ℕ), 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ a * b = 2079 ∧ (a, b) = (21, 99) ∨ (a, b) = (33, 63) :=
sorry

end NUMINAMATH_GPT_factorizations_of_2079_l1996_199658


namespace NUMINAMATH_GPT_total_students_l1996_199699

theorem total_students (a b c d e f : ℕ)  (h : a + b = 15) (h1 : a = 5) (h2 : b = 10) 
(h3 : c = 15) (h4 : d = 10) (h5 : e = 5) (h6 : f = 0) (h_total : a + b + c + d + e + f = 50) : a + b + c + d + e + f = 50 :=
by {exact h_total}

end NUMINAMATH_GPT_total_students_l1996_199699


namespace NUMINAMATH_GPT_find_m_l1996_199638

-- Define the conditions
def function_is_decreasing (m : ℝ) : Prop := 
  (m^2 - m - 1 = 1) ∧ (1 - m < 0)

-- The proof problem: prove m = 2 given the conditions
theorem find_m (m : ℝ) (h : function_is_decreasing m) : m = 2 := 
by
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_find_m_l1996_199638


namespace NUMINAMATH_GPT_distinct_terms_count_l1996_199695

theorem distinct_terms_count
  (x y z w p q r s t : Prop)
  (h1 : ¬(x = y ∨ x = z ∨ x = w ∨ y = z ∨ y = w ∨ z = w))
  (h2 : ¬(p = q ∨ p = r ∨ p = s ∨ p = t ∨ q = r ∨ q = s ∨ q = t ∨ r = s ∨ r = t ∨ s = t)) :
  ∃ (n : ℕ), n = 20 := by
  sorry

end NUMINAMATH_GPT_distinct_terms_count_l1996_199695


namespace NUMINAMATH_GPT_volume_of_inscribed_sphere_l1996_199681

theorem volume_of_inscribed_sphere {cube_edge : ℝ} (h : cube_edge = 6) : 
  ∃ V : ℝ, V = 36 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_volume_of_inscribed_sphere_l1996_199681


namespace NUMINAMATH_GPT_divisor_iff_even_l1996_199673

noncomputable def hasDivisor (k : ℕ) : Prop := 
  ∃ n : ℕ, n > 0 ∧ (8 * k * n - 1) ∣ (4 * k ^ 2 - 1) ^ 2

theorem divisor_iff_even (k : ℕ) (h : k > 0) : hasDivisor k ↔ (k % 2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_divisor_iff_even_l1996_199673


namespace NUMINAMATH_GPT_eq1_eq2_eq3_eq4_l1996_199666

theorem eq1 : ∀ x : ℝ, x = 6 → 3 * x - 8 = x + 4 := by
  intros x hx
  rw [hx]
  sorry

theorem eq2 : ∀ x : ℝ, x = -2 → 1 - 3 * (x + 1) = 2 * (1 - 0.5 * x) := by
  intros x hx
  rw [hx]
  sorry

theorem eq3 : ∀ x : ℝ, x = -20 → (1 / 6) * (3 * x - 6) = (2 / 5) * x - 3 := by
  intros x hx
  rw [hx]
  sorry

theorem eq4 : ∀ y : ℝ, y = -1 → (3 * y - 1) / 4 - 1 = (5 * y - 7) / 6 := by
  intros y hy
  rw [hy]
  sorry

end NUMINAMATH_GPT_eq1_eq2_eq3_eq4_l1996_199666


namespace NUMINAMATH_GPT_problem1_problem2_l1996_199611

theorem problem1 : 3 * Real.sqrt 3 - Real.sqrt 8 + Real.sqrt 2 - Real.sqrt 27 = -Real.sqrt 2 := 
by sorry

theorem problem2 : (Real.sqrt 5 - Real.sqrt 3) * (Real.sqrt 5 + Real.sqrt 3) = 2 := 
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1996_199611


namespace NUMINAMATH_GPT_eliana_total_steps_l1996_199650

noncomputable def day1_steps : ℕ := 200 + 300
noncomputable def day2_steps : ℕ := 2 * day1_steps
noncomputable def day3_steps : ℕ := day1_steps + day2_steps + 100

theorem eliana_total_steps : day3_steps = 1600 := by
  sorry

end NUMINAMATH_GPT_eliana_total_steps_l1996_199650


namespace NUMINAMATH_GPT_original_price_l1996_199645

theorem original_price (p q: ℝ) (h₁ : p ≠ 0) (h₂ : q ≠ 0) : 
  let x := 20000 / (10000^2 - (p^2 + q^2) * 10000 + p^2 * q^2)
  (x : ℝ) * (1 - p^2 / 10000) * (1 - q^2 / 10000) = 2 :=
by
  sorry

end NUMINAMATH_GPT_original_price_l1996_199645


namespace NUMINAMATH_GPT_difference_of_squares_l1996_199684

theorem difference_of_squares {a b : ℝ} (h1 : a + b = 75) (h2 : a - b = 15) : a^2 - b^2 = 1125 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_l1996_199684


namespace NUMINAMATH_GPT_roots_of_poly_l1996_199629

noncomputable def poly (x : ℝ) : ℝ := x^3 - 4 * x^2 - x + 4

theorem roots_of_poly :
  (poly 1 = 0) ∧ (poly (-1) = 0) ∧ (poly 4 = 0) ∧
  (∀ x, poly x = 0 → x = 1 ∨ x = -1 ∨ x = 4) :=
by
  sorry

end NUMINAMATH_GPT_roots_of_poly_l1996_199629


namespace NUMINAMATH_GPT_adam_lessons_on_monday_l1996_199649

theorem adam_lessons_on_monday :
  (∃ (time_monday time_tuesday time_wednesday : ℝ) (n_monday_lessons : ℕ),
    time_tuesday = 3 ∧
    time_wednesday = 2 * time_tuesday ∧
    time_monday + time_tuesday + time_wednesday = 12 ∧
    n_monday_lessons = time_monday / 0.5 ∧
    n_monday_lessons = 6) :=
by
  sorry

end NUMINAMATH_GPT_adam_lessons_on_monday_l1996_199649


namespace NUMINAMATH_GPT_marble_ratio_l1996_199632

theorem marble_ratio (A V X : ℕ) 
  (h1 : A + 5 = V - 5)
  (h2 : V + X = (A - X) + 30) : X / 5 = 2 :=
by
  sorry

end NUMINAMATH_GPT_marble_ratio_l1996_199632


namespace NUMINAMATH_GPT_members_in_third_shift_l1996_199624

-- Defining the given conditions
def total_first_shift : ℕ := 60
def percent_first_shift_pension : ℝ := 0.20

def total_second_shift : ℕ := 50
def percent_second_shift_pension : ℝ := 0.40

variable (T : ℕ)
def percent_third_shift_pension : ℝ := 0.10

def percent_total_pension_program : ℝ := 0.24

noncomputable def number_of_members_third_shift : ℕ :=
  T

-- Using the conditions to declare the theorem
theorem members_in_third_shift :
  ((60 * 0.20) + (50 * 0.40) + (number_of_members_third_shift T * percent_third_shift_pension)) / (60 + 50 + number_of_members_third_shift T) = percent_total_pension_program →
  number_of_members_third_shift T = 40 :=
sorry

end NUMINAMATH_GPT_members_in_third_shift_l1996_199624


namespace NUMINAMATH_GPT_number_of_real_solutions_l1996_199687

-- Definition of the equation
def equation (x : ℝ) : Prop := x / 50 = Real.cos x

-- The main theorem stating the number of solutions
theorem number_of_real_solutions : 
  ∃ (n : ℕ), n = 32 ∧ ∀ x : ℝ, equation x → -50 ≤ x ∧ x ≤ 50 :=
sorry

end NUMINAMATH_GPT_number_of_real_solutions_l1996_199687


namespace NUMINAMATH_GPT_solve_for_x_l1996_199637

theorem solve_for_x (x : ℚ) : (2/5 : ℚ) - (1/4 : ℚ) = 1/x → x = 20/3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l1996_199637


namespace NUMINAMATH_GPT_desiree_age_l1996_199689

variables (D C : ℕ)
axiom condition1 : D = 2 * C
axiom condition2 : D + 30 = (2 * (C + 30)) / 3 + 14

theorem desiree_age : D = 6 :=
by
  sorry

end NUMINAMATH_GPT_desiree_age_l1996_199689


namespace NUMINAMATH_GPT_initial_speeds_l1996_199601

/-- Motorcyclists Vasya and Petya ride at constant speeds around a circular track 1 km long.
    Petya overtakes Vasya every 2 minutes. Then Vasya doubles his speed and now he himself 
    overtakes Petya every 2 minutes. What were the initial speeds of Vasya and Petya? 
    Answer: 1000 and 1500 meters per minute.
-/

theorem initial_speeds (V_v V_p : ℕ) (track_length : ℕ) (time_interval : ℕ) 
  (h1 : track_length = 1000)
  (h2 : time_interval = 2)
  (h3 : V_p - V_v = track_length / time_interval)
  (h4 : 2 * V_v - V_p = track_length / time_interval):
  V_v = 1000 ∧ V_p = 1500 :=
by
  sorry

end NUMINAMATH_GPT_initial_speeds_l1996_199601


namespace NUMINAMATH_GPT_amount_per_person_l1996_199677

theorem amount_per_person (total_amount : ℕ) (num_persons : ℕ) (amount_each : ℕ)
  (h1 : total_amount = 42900) (h2 : num_persons = 22) (h3 : amount_each = 1950) :
  total_amount / num_persons = amount_each :=
by
  -- Proof to be filled
  sorry

end NUMINAMATH_GPT_amount_per_person_l1996_199677


namespace NUMINAMATH_GPT_decimal_arithmetic_l1996_199625

theorem decimal_arithmetic : 0.45 - 0.03 + 0.008 = 0.428 := by
  sorry

end NUMINAMATH_GPT_decimal_arithmetic_l1996_199625


namespace NUMINAMATH_GPT_real_roots_determinant_l1996_199660

variable (a b c k : ℝ)
variable (k_pos : k > 0)
variable (a_nonzero : a ≠ 0) 
variable (b_nonzero : b ≠ 0)
variable (c_nonzero : c ≠ 0)
variable (k_nonzero : k ≠ 0)

theorem real_roots_determinant : 
  ∃! x : ℝ, (Matrix.det ![![x, k * c, -k * b], ![-k * c, x, k * a], ![k * b, -k * a, x]] = 0) :=
sorry

end NUMINAMATH_GPT_real_roots_determinant_l1996_199660


namespace NUMINAMATH_GPT_supplement_complement_l1996_199626

theorem supplement_complement (angle1 angle2 : ℝ) 
  (h_complementary : angle1 + angle2 = 90) : 
   180 - angle1 = 90 + angle2 := by
  sorry

end NUMINAMATH_GPT_supplement_complement_l1996_199626


namespace NUMINAMATH_GPT_quadratic_inequality_sufficient_necessary_l1996_199613

theorem quadratic_inequality_sufficient_necessary (a : ℝ) : 
  (∀ x : ℝ, x^2 - a * x + a > 0) ↔ 0 < a ∧ a < 4 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_quadratic_inequality_sufficient_necessary_l1996_199613


namespace NUMINAMATH_GPT_white_cats_count_l1996_199663

theorem white_cats_count (total_cats : ℕ) (black_cats : ℕ) (gray_cats : ℕ) (white_cats : ℕ)
  (h1 : total_cats = 15)
  (h2 : black_cats = 10)
  (h3 : gray_cats = 3)
  (h4 : total_cats = black_cats + gray_cats + white_cats) : 
  white_cats = 2 := 
  by
    -- proof or sorry here
    sorry

end NUMINAMATH_GPT_white_cats_count_l1996_199663


namespace NUMINAMATH_GPT_only_negative_integer_among_list_l1996_199623

namespace NegativeIntegerProblem

def is_integer (n : ℚ) : Prop := ∃ m : ℤ, n = m

theorem only_negative_integer_among_list :
  (∃ x, x ∈ [0, -1, 2, -1.5] ∧ (x < 0) ∧ is_integer x) ↔ (x = -1) :=
by
  sorry

end NegativeIntegerProblem

end NUMINAMATH_GPT_only_negative_integer_among_list_l1996_199623
