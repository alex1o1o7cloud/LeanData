import Mathlib

namespace square_side_length_l1213_121389

variable (s : ℕ)
variable (P A : ℕ)

theorem square_side_length (h1 : P = 52) (h2 : A = 169) (h3 : P = 4 * s) (h4 : A = s * s) : s = 13 :=
sorry

end square_side_length_l1213_121389


namespace distance_D_to_plane_l1213_121341

-- Given conditions about the distances from points A, B, and C to plane M
variables (a b c : ℝ)

-- Formalizing the distance from vertex D to plane M
theorem distance_D_to_plane (a b c : ℝ) : 
  ∃ d : ℝ, d = |a + b + c| ∨ d = |a + b - c| ∨ d = |a - b + c| ∨ d = |-a + b + c| ∨ 
                    d = |a - b - c| ∨ d = |-a - b + c| ∨ d = |-a + b - c| ∨ d = |-a - b - c| := sorry

end distance_D_to_plane_l1213_121341


namespace sum_of_roots_l1213_121345

theorem sum_of_roots (a b : ℝ) (h1 : a^2 - 4*a - 2023 = 0) (h2 : b^2 - 4*b - 2023 = 0) : a + b = 4 :=
sorry

end sum_of_roots_l1213_121345


namespace max_extra_packages_l1213_121362

/-- Max's delivery performance --/
def max_daily_packages : Nat := 35

/-- (1) Max delivered the maximum number of packages on two days --/
def max_2_days : Nat := 2 * max_daily_packages

/-- (2) On two other days, Max unloaded a total of 50 packages --/
def two_days_50 : Nat := 50

/-- (3) On one day, Max delivered one-seventh of the maximum possible daily performance --/
def one_seventh_day : Nat := max_daily_packages / 7

/-- (4) On the last two days, the sum of packages was four-fifths of the maximum daily performance --/
def last_2_days : Nat := 2 * (4 * max_daily_packages / 5)

/-- (5) Total packages delivered in the week --/
def total_delivered : Nat := max_2_days + two_days_50 + one_seventh_day + last_2_days

/-- (6) Total possible packages in a week if worked at maximum performance --/
def total_possible : Nat := 7 * max_daily_packages

/-- (7) Difference between total possible and total delivered packages --/
def difference : Nat := total_possible - total_delivered

/-- Proof problem: Prove the difference is 64 --/
theorem max_extra_packages : difference = 64 := by
  sorry

end max_extra_packages_l1213_121362


namespace find_fahrenheit_l1213_121320

variable (F : ℝ)
variable (C : ℝ)

theorem find_fahrenheit (h : C = 40) (h' : C = 5 / 9 * (F - 32)) : F = 104 := by
  sorry

end find_fahrenheit_l1213_121320


namespace parabola_directrix_l1213_121375

theorem parabola_directrix (x : ℝ) : 
  (6 * x^2 + 5 = y) → (y = 6 * x^2 + 5) → (y = 6 * 0^2 + 5) → (y = (119 : ℝ) / 24) := 
sorry

end parabola_directrix_l1213_121375


namespace imaginary_part_of_z_l1213_121319

open Complex

theorem imaginary_part_of_z (z : ℂ) (h : z = 2 / (-1 + I)) : z.im = -1 := 
by
  sorry

end imaginary_part_of_z_l1213_121319


namespace find_a_l1213_121326

noncomputable def f (a x : ℝ) : ℝ := Real.log (Real.sqrt (1 + a * x ^ 2) - x)

theorem find_a (a : ℝ) :
  (∀ (x : ℝ), f a (-x) = -f a x) ↔ a = 1 :=
by
  sorry

end find_a_l1213_121326


namespace logan_usual_cartons_l1213_121346

theorem logan_usual_cartons 
  (C : ℕ)
  (h1 : ∀ cartons, (∀ jars : ℕ, jars = 20 * cartons) → jars = 20 * C)
  (h2 : ∀ cartons, cartons = C - 20)
  (h3 : ∀ damaged_jars, (∀ cartons : ℕ, cartons = 5) → damaged_jars = 3 * 5)
  (h4 : ∀ completely_damaged_jars, completely_damaged_jars = 20)
  (h5 : ∀ good_jars, good_jars = 565) :
  C = 50 :=
by
  sorry

end logan_usual_cartons_l1213_121346


namespace find_f_prime_at_2_l1213_121361

noncomputable def f (f' : ℝ → ℝ) (x : ℝ) : ℝ :=
  x^2 + 2 * x * f' 2 - Real.log x

theorem find_f_prime_at_2 (f' : ℝ → ℝ) (h : ∀ x, deriv (f f') x = f' x) :
  f' 2 = -7 / 2 :=
by
  have H := h 2
  sorry

end find_f_prime_at_2_l1213_121361


namespace base_height_calculation_l1213_121356

noncomputable def height_of_sculpture : ℚ := 2 + 5/6 -- 2 feet 10 inches in feet
noncomputable def total_height : ℚ := 3.5
noncomputable def height_of_base : ℚ := 2/3

theorem base_height_calculation (h1 : height_of_sculpture = 17/6) (h2 : total_height = 21/6):
  height_of_base = total_height - height_of_sculpture := by
  sorry

end base_height_calculation_l1213_121356


namespace AM_GM_inequality_example_l1213_121332

theorem AM_GM_inequality_example (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) : 
  (a + b) * (a + c) * (b + c) ≥ 8 * a * b * c :=
by
  sorry

end AM_GM_inequality_example_l1213_121332


namespace number_of_terms_in_arithmetic_sequence_l1213_121368

-- Define the necessary conditions
def a := 2
def d := 5
def l := 1007  -- last term

-- Prove the number of terms in the sequence
theorem number_of_terms_in_arithmetic_sequence : 
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 202 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l1213_121368


namespace fifth_pyTriple_is_correct_l1213_121324

-- Definitions based on conditions from part (a)
def pyTriple (n : ℕ) : ℕ × ℕ × ℕ :=
  let a := 2 * n + 1
  let b := 2 * n * (n + 1)
  let c := b + 1
  (a, b, c)

-- Question: Prove that the 5th Pythagorean triple is (11, 60, 61)
theorem fifth_pyTriple_is_correct : pyTriple 5 = (11, 60, 61) :=
  by
    -- Skip the proof
    sorry

end fifth_pyTriple_is_correct_l1213_121324


namespace number_of_ways_l1213_121330

theorem number_of_ways (h_walk : ℕ) (h_drive : ℕ) (h_eq1 : h_walk = 3) (h_eq2 : h_drive = 4) : h_walk + h_drive = 7 :=
by 
  sorry

end number_of_ways_l1213_121330


namespace solve_for_m_l1213_121328

def f (x : ℝ) (m : ℝ) := x^3 - m * x + 3

def f_prime (x : ℝ) (m : ℝ) := 3 * x^2 - m

theorem solve_for_m (m : ℝ) : f_prime 1 m = 0 → m = 3 :=
by
  sorry

end solve_for_m_l1213_121328


namespace problem1_problem2_l1213_121371

-- Problem 1:
theorem problem1 (P : ℝ × ℝ) (hP : P = (-1, 0)) :
  (∃ m b, (∀ x y, y = m * x + b ↔ (3 * x - y + 3 = 0)) ∧ ∀ x y, (x = -1 → y = 0 → y = m * x + b)) → 
  ∃ m b, ∀ x y, y = m * x + b ↔ (3 * x - y + 3 = 0) :=
sorry

-- Problem 2:
theorem problem2 (L1 : ℝ → ℝ → Prop) (hL1 : ∀ x y, L1 x y ↔ 3 * x + 4 * y - 12 = 0) (d : ℝ) (hd : d = 7) :
  (∃ c, ∀ x y, (3 * x + 4 * y + c = 0 ∨ 3 * x + 4 * y - 47 = 0) ↔ L1 x y ∧ d = 7) :=
sorry

end problem1_problem2_l1213_121371


namespace no_integers_for_sum_of_squares_l1213_121307

theorem no_integers_for_sum_of_squares :
  ¬ ∃ a b : ℤ, a^2 + b^2 = 10^100 + 3 :=
by
  sorry

end no_integers_for_sum_of_squares_l1213_121307


namespace domain_f_l1213_121390

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x^2 - 5 * x + 6)

theorem domain_f :
  {x : ℝ | f x ≠ f x} = {x : ℝ | (x < 2) ∨ (2 < x ∧ x < 3) ∨ (3 < x)} :=
by sorry

end domain_f_l1213_121390


namespace inequality_subtraction_l1213_121354

variable (a b : ℝ)

-- Given conditions
axiom nonzero_a : a ≠ 0 
axiom nonzero_b : b ≠ 0 
axiom a_lt_b : a < b 

-- Proof statement
theorem inequality_subtraction : a - 3 < b - 3 := 
by 
  sorry

end inequality_subtraction_l1213_121354


namespace complete_square_l1213_121323

theorem complete_square (x : ℝ) : x^2 + 4*x + 1 = 0 -> (x + 2)^2 = 3 :=
by sorry

end complete_square_l1213_121323


namespace stratified_sampling_male_students_l1213_121379

theorem stratified_sampling_male_students (total_students : ℕ) (female_students : ℕ) (sample_size : ℕ)
  (h1 : total_students = 900) (h2 : female_students = 0) (h3 : sample_size = 45) : 
  ((total_students - female_students) * sample_size / total_students) = 25 := 
by {
  sorry
}

end stratified_sampling_male_students_l1213_121379


namespace cricket_team_members_eq_11_l1213_121394

-- Definitions based on conditions:
def captain_age : ℕ := 26
def wicket_keeper_age : ℕ := 31
def avg_age_whole_team : ℕ := 24
def avg_age_remaining_players : ℕ := 23

-- Definition of n based on the problem conditions
def number_of_members (n : ℕ) : Prop :=
  n * avg_age_whole_team = (n - 2) * avg_age_remaining_players + (captain_age + wicket_keeper_age)

-- The proof statement:
theorem cricket_team_members_eq_11 : ∃ n, number_of_members n ∧ n = 11 := 
by
  use 11
  unfold number_of_members
  sorry

end cricket_team_members_eq_11_l1213_121394


namespace combined_pre_tax_and_pre_tip_cost_l1213_121331

theorem combined_pre_tax_and_pre_tip_cost (x y : ℝ) 
  (hx : 1.28 * x = 35.20) 
  (hy : 1.19 * y = 22.00) : 
  x + y = 46 := 
by
  sorry

end combined_pre_tax_and_pre_tip_cost_l1213_121331


namespace derivative_f_at_1_l1213_121312

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * Real.sin x

theorem derivative_f_at_1 : (deriv f 1) = 2 + 2 * Real.cos 1 := 
sorry

end derivative_f_at_1_l1213_121312


namespace smallest_value_of_n_l1213_121374

theorem smallest_value_of_n 
  (n : ℕ) 
  (h1 : ∀ θ : ℝ, θ = (n - 2) * 180 / n) 
  (h2 : ∀ α : ℝ, α = 360 / n) 
  (h3 : 28 = 180 / n) :
  n = 45 :=
sorry

end smallest_value_of_n_l1213_121374


namespace tangent_line_to_parabola_l1213_121388

noncomputable def parabola (x : ℝ) : ℝ := 4 * x^2

def derivative_parabola (x : ℝ) : ℝ := 8 * x

def tangent_line_eq (x y : ℝ) : Prop := 8 * x - y - 4 = 0

theorem tangent_line_to_parabola (x : ℝ) (hx : x = 1) (hy : parabola x = 4) :
    tangent_line_eq 1 4 :=
by 
  -- Sorry to skip the detailed proof, but it should follow the steps outlined in the solution.
  sorry

end tangent_line_to_parabola_l1213_121388


namespace find_F_l1213_121378

theorem find_F (F C : ℝ) (h1 : C = 4/7 * (F - 40)) (h2 : C = 28) : F = 89 := 
by
  sorry

end find_F_l1213_121378


namespace perimeter_difference_l1213_121300

theorem perimeter_difference (x : ℝ) :
  let small_square_perimeter := 4 * x
  let large_square_perimeter := 4 * (x + 8)
  large_square_perimeter - small_square_perimeter = 32 :=
by
  sorry

end perimeter_difference_l1213_121300


namespace chord_on_ellipse_midpoint_l1213_121350

theorem chord_on_ellipse_midpoint :
  ∀ (A B : ℝ × ℝ)
    (hx1 : (A.1^2) / 2 + A.2^2 = 1)
    (hx2 : (B.1^2) / 2 + B.2^2 = 1)
    (mid : (A.1 + B.1) / 2 = 1/2 ∧ (A.2 + B.2) / 2 = 1/2),
  ∃ (k : ℝ), ∀ (x y : ℝ), y - 1/2 = k * (x - 1/2) ↔ 2 * x + 4 * y = 3 := 
sorry

end chord_on_ellipse_midpoint_l1213_121350


namespace total_amount_spent_l1213_121385

variable (B D : ℝ)

-- Conditions
def condition1 : Prop := D = (1/2) * B
def condition2 : Prop := B = D + 15

-- Proof statement
theorem total_amount_spent (h1 : condition1 B D) (h2 : condition2 B D) : B + D = 45 := by
  sorry

end total_amount_spent_l1213_121385


namespace distance_from_plate_to_bottom_edge_l1213_121327

theorem distance_from_plate_to_bottom_edge :
    ∀ (d : ℕ), 10 + 63 = 20 + d → d = 53 :=
by
  intros d h
  sorry

end distance_from_plate_to_bottom_edge_l1213_121327


namespace find_num_oranges_l1213_121373

def num_oranges (O : ℝ) (x : ℕ) : Prop :=
  6 * 0.21 + O * (x : ℝ) = 1.77 ∧ 2 * 0.21 + 5 * O = 1.27
  ∧ 0.21 = 0.21

theorem find_num_oranges (O : ℝ) (x : ℕ) (h : num_oranges O x) : x = 3 :=
  sorry

end find_num_oranges_l1213_121373


namespace james_lifting_heavy_after_39_days_l1213_121367

noncomputable def JamesInjuryHealingTime : Nat := 3
noncomputable def HealingTimeFactor : Nat := 5
noncomputable def WaitingTimeAfterHealing : Nat := 3
noncomputable def AdditionalWaitingTimeWeeks : Nat := 3

theorem james_lifting_heavy_after_39_days :
  let healing_time := JamesInjuryHealingTime * HealingTimeFactor
  let total_time_before_workout := healing_time + WaitingTimeAfterHealing
  let additional_waiting_time_days := AdditionalWaitingTimeWeeks * 7
  let total_time_before_lifting_heavy := total_time_before_workout + additional_waiting_time_days
  total_time_before_lifting_heavy = 39 := by
  sorry

end james_lifting_heavy_after_39_days_l1213_121367


namespace rectangle_area_l1213_121359

theorem rectangle_area (w : ℝ) (h : ℝ) (area : ℝ) 
  (h1 : w = 5)
  (h2 : h = 2 * w) :
  area = h * w := by
  sorry

end rectangle_area_l1213_121359


namespace fraction_is_square_l1213_121384

theorem fraction_is_square (a b : ℕ) (hpos_a : a > 0) (hpos_b : b > 0) 
  (hdiv : (ab + 1) ∣ (a^2 + b^2)) :
  ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) :=
sorry

end fraction_is_square_l1213_121384


namespace number_of_dogs_l1213_121397

-- Conditions
def ratio_cats_dogs : ℚ := 3 / 4
def number_cats : ℕ := 18

-- Define the theorem to prove
theorem number_of_dogs : ∃ (dogs : ℕ), dogs = 24 :=
by
  -- Proof steps will go here, but we can use sorry for now to skip actual proving.
  sorry

end number_of_dogs_l1213_121397


namespace isosceles_triangle_base_angles_l1213_121318

theorem isosceles_triangle_base_angles 
  (α β : ℝ) -- α and β are the base angles
  (h : α = β)
  (height leg : ℝ)
  (h_height_leg : height = leg / 2) : 
  α = 75 ∨ α = 15 :=
by
  sorry

end isosceles_triangle_base_angles_l1213_121318


namespace janice_initial_sentences_l1213_121358

theorem janice_initial_sentences :
  ∀ (initial_sentences total_sentences erased_sentences: ℕ)
    (typed_rate before_break_minutes additional_minutes after_meeting_minutes: ℕ),
  typed_rate = 6 →
  before_break_minutes = 20 →
  additional_minutes = 15 →
  after_meeting_minutes = 18 →
  erased_sentences = 40 →
  total_sentences = 536 →
  (total_sentences - (before_break_minutes * typed_rate + (before_break_minutes + additional_minutes) * typed_rate + after_meeting_minutes * typed_rate - erased_sentences)) = initial_sentences →
  initial_sentences = 138 :=
by
  intros initial_sentences total_sentences erased_sentences typed_rate before_break_minutes additional_minutes after_meeting_minutes
  intros h_rate h_before h_additional h_after_meeting h_erased h_total h_eqn
  rw [h_rate, h_before, h_additional, h_after_meeting, h_erased, h_total] at h_eqn
  linarith

end janice_initial_sentences_l1213_121358


namespace problem1_problem2_l1213_121337

theorem problem1 (a b : ℝ) : ((a * b) ^ 6 / (a * b) ^ 2 * (a * b) ^ 4) = a^8 * b^8 := 
by sorry

theorem problem2 (x : ℝ) : ((3 * x^3)^2 * x^5 - (-x^2)^6 / x) = 8 * x^11 :=
by sorry

end problem1_problem2_l1213_121337


namespace savings_in_july_l1213_121347

-- Definitions based on the conditions
def savings_june : ℕ := 27
def savings_august : ℕ := 21
def expenses_books : ℕ := 5
def expenses_shoes : ℕ := 17
def final_amount_left : ℕ := 40

-- Main theorem stating the problem
theorem savings_in_july (J : ℕ) : 
  savings_june + J + savings_august - (expenses_books + expenses_shoes) = final_amount_left → 
  J = 14 :=
by
  sorry

end savings_in_july_l1213_121347


namespace disjoint_union_A_B_l1213_121303

def A : Set ℕ := {x | x^2 - 3*x + 2 = 0}
def B : Set ℕ := {y | ∃ x ∈ A, y = x^2 - 2*x + 3}

def symmetric_difference (M P : Set ℕ) : Set ℕ :=
  {x | (x ∈ M ∨ x ∈ P) ∧ x ∉ M ∩ P}

theorem disjoint_union_A_B :
  symmetric_difference A B = {1, 3} := by
  sorry

end disjoint_union_A_B_l1213_121303


namespace largest_n_cube_condition_l1213_121387

theorem largest_n_cube_condition :
  ∃ n : ℕ, (n^3 + 4 * n^2 - 15 * n - 18 = k^3) ∧ ∀ m : ℕ, (m^3 + 4 * m^2 - 15 * m - 18 = k^3 → m ≤ n) → n = 19 :=
by
  sorry

end largest_n_cube_condition_l1213_121387


namespace smallest_positive_four_digit_integer_equivalent_to_3_mod_4_l1213_121383

theorem smallest_positive_four_digit_integer_equivalent_to_3_mod_4 : 
  ∃ n : ℤ, n ≥ 1000 ∧ n % 4 = 3 ∧ n = 1003 := 
by {
  sorry
}

end smallest_positive_four_digit_integer_equivalent_to_3_mod_4_l1213_121383


namespace least_value_of_expression_l1213_121305

theorem least_value_of_expression : ∃ (x y : ℝ), (2 * x - y + 3)^2 + (x + 2 * y - 1)^2 = 295 / 72 := sorry

end least_value_of_expression_l1213_121305


namespace solve_car_production_l1213_121355

def car_production_problem : Prop :=
  ∃ (NorthAmericaCars : ℕ) (TotalCars : ℕ) (EuropeCars : ℕ),
    NorthAmericaCars = 3884 ∧
    TotalCars = 6755 ∧
    EuropeCars = TotalCars - NorthAmericaCars ∧
    EuropeCars = 2871

theorem solve_car_production : car_production_problem := by
  sorry

end solve_car_production_l1213_121355


namespace unique_A_value_l1213_121310

theorem unique_A_value (A : ℝ) (x1 x2 : ℂ) (hx1_ne : x1 ≠ x2) :
  (x1 * (x1 + 1) = A) ∧ (x2 * (x2 + 1) = A) ∧ (A * x1^4 + 3 * x1^3 + 5 * x1 = x2^4 + 3 * x2^3 + 5 * x2) 
  → A = -7 := by
  sorry

end unique_A_value_l1213_121310


namespace correct_transformation_l1213_121335

theorem correct_transformation (m : ℤ) (h : 2 * m - 1 = 3) : 2 * m = 4 :=
by
  sorry

end correct_transformation_l1213_121335


namespace jacket_purchase_price_l1213_121336

theorem jacket_purchase_price (S D P : ℝ) 
  (h1 : S = P + 0.30 * S)
  (h2 : D = 0.80 * S)
  (h3 : 6.000000000000007 = D - P) :
  P = 42 :=
by
  sorry

end jacket_purchase_price_l1213_121336


namespace cost_to_plant_flowers_l1213_121351

theorem cost_to_plant_flowers :
  let cost_flowers := 9
  let cost_clay_pot := cost_flowers + 20
  let cost_soil := cost_flowers - 2
  cost_flowers + cost_clay_pot + cost_soil = 45 := 
by
  let cost_flowers := 9
  let cost_clay_pot := cost_flowers + 20
  let cost_soil := cost_flowers - 2
  show cost_flowers + cost_clay_pot + cost_soil = 45
  sorry

end cost_to_plant_flowers_l1213_121351


namespace percentage_spent_on_household_items_eq_50_l1213_121382

-- Definitions for the conditions in the problem
def MonthlyIncome : ℝ := 90000
def ClothesPercentage : ℝ := 0.25
def MedicinesPercentage : ℝ := 0.15
def Savings : ℝ := 9000

-- Definition of the statement where we need to calculate the percentage spent on household items
theorem percentage_spent_on_household_items_eq_50 :
  let ClothesExpense := ClothesPercentage * MonthlyIncome
  let MedicinesExpense := MedicinesPercentage * MonthlyIncome
  let TotalExpense := ClothesExpense + MedicinesExpense + Savings
  let HouseholdItemsExpense := MonthlyIncome - TotalExpense
  let TotalIncome := MonthlyIncome
  (HouseholdItemsExpense / TotalIncome) * 100 = 50 :=
by
  sorry

end percentage_spent_on_household_items_eq_50_l1213_121382


namespace rectangular_prism_volume_l1213_121380

theorem rectangular_prism_volume
  (l w h : ℝ)
  (face1 : l * w = 6)
  (face2 : w * h = 8)
  (face3 : l * h = 12) : l * w * h = 24 := sorry

end rectangular_prism_volume_l1213_121380


namespace fred_change_received_l1213_121395

theorem fred_change_received :
  let ticket_price := 5.92
  let ticket_count := 2
  let borrowed_movie_price := 6.79
  let amount_paid := 20.00
  let total_cost := (ticket_price * ticket_count) + borrowed_movie_price
  let change := amount_paid - total_cost
  change = 1.37 :=
by
  sorry

end fred_change_received_l1213_121395


namespace direct_variation_y_value_l1213_121352

theorem direct_variation_y_value (x y k : ℝ) (h1 : y = k * x) (h2 : ∀ x, x = 5 → y = 10) 
                                 (h3 : ∀ x, x < 0 → k = 4) (hx : x = -6) : y = -24 :=
sorry

end direct_variation_y_value_l1213_121352


namespace expression_value_l1213_121372

theorem expression_value (x : ℤ) (h : x = 2) : (2 * x + 5)^3 = 729 := by
  sorry

end expression_value_l1213_121372


namespace origin_moves_distance_l1213_121340

noncomputable def origin_distance_moved : ℝ :=
  let B := (3, 1)
  let B' := (7, 9)
  let k := 1.5
  let center_of_dilation := (-1, -3)
  let d0 := Real.sqrt ((-1)^2 + (-3)^2)
  let d1 := k * d0
  d1 - d0

theorem origin_moves_distance :
  origin_distance_moved = 0.5 * Real.sqrt 10 :=
by 
  sorry

end origin_moves_distance_l1213_121340


namespace quoted_price_correct_l1213_121353

noncomputable def after_tax_yield (yield : ℝ) (tax_rate : ℝ) : ℝ :=
  yield * (1 - tax_rate)

noncomputable def real_yield (after_tax_yield : ℝ) (inflation_rate : ℝ) : ℝ :=
  after_tax_yield - inflation_rate

noncomputable def quoted_price (dividend_rate : ℝ) (real_yield : ℝ) (commission_rate : ℝ) : ℝ :=
  real_yield / (dividend_rate / (1 + commission_rate))

theorem quoted_price_correct :
  quoted_price 0.16 (real_yield (after_tax_yield 0.08 0.15) 0.03) 0.02 = 24.23 :=
by
  -- This is the proof statement. Since the task does not require us to prove it, we use 'sorry'.
  sorry

end quoted_price_correct_l1213_121353


namespace sum_of_ages_l1213_121317

theorem sum_of_ages (a b c : ℕ) (twin : a = b) (product : a * b * c = 256) : a + b + c = 20 := by
  sorry

end sum_of_ages_l1213_121317


namespace units_digit_fraction_l1213_121357

theorem units_digit_fraction :
  (30 * 31 * 32 * 33 * 34 * 35) % 10000 % 10 = 4 :=
by
  -- Placeholder for actual proof
  sorry

end units_digit_fraction_l1213_121357


namespace complex_abs_of_sqrt_l1213_121338

variable (z : ℂ)

theorem complex_abs_of_sqrt (h : z^2 = 16 - 30 * Complex.I) : Complex.abs z = Real.sqrt 34 := by
  sorry

end complex_abs_of_sqrt_l1213_121338


namespace symmetric_point_coordinates_l1213_121392

noncomputable def symmetric_with_respect_to_y_axis (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match p with
  | (x, y, z) => (-x, y, -z)

theorem symmetric_point_coordinates : symmetric_with_respect_to_y_axis (-2, 1, 4) = (2, 1, -4) :=
by sorry

end symmetric_point_coordinates_l1213_121392


namespace longest_side_is_48_l1213_121366

noncomputable def longest_side_of_triangle (a b c : ℝ) (ha : a/b = 1/2) (hb : b/c = 1/3) (hc : c/a = 1/4) (hp : a + b + c = 104) : ℝ :=
  a

theorem longest_side_is_48 {a b c : ℝ} (ha : a/b = 1/2) (hb : b/c = 1/3) (hc : c/a = 1/4) (hp : a + b + c = 104) : 
  longest_side_of_triangle a b c ha hb hc hp = 48 :=
sorry

end longest_side_is_48_l1213_121366


namespace train_speed_is_60_kmph_l1213_121329

noncomputable def speed_of_train_in_kmph (length_meters time_seconds : ℝ) : ℝ :=
  (length_meters / time_seconds) * 3.6

theorem train_speed_is_60_kmph (length_meters time_seconds : ℝ) :
  length_meters = 50 → time_seconds = 3 → speed_of_train_in_kmph length_meters time_seconds = 60 :=
by
  intros h_length h_time
  simp [speed_of_train_in_kmph, h_length, h_time]
  norm_num
  sorry

end train_speed_is_60_kmph_l1213_121329


namespace Lance_workdays_per_week_l1213_121377

theorem Lance_workdays_per_week (weekly_hours hourly_wage daily_earnings : ℕ) 
  (h1 : weekly_hours = 35)
  (h2 : hourly_wage = 9)
  (h3 : daily_earnings = 63) :
  weekly_hours / (daily_earnings / hourly_wage) = 5 := by
  sorry

end Lance_workdays_per_week_l1213_121377


namespace find_train_parameters_l1213_121370

-- Definitions based on the problem statement
def bridge_length : ℕ := 1000
def time_total : ℕ := 60
def time_on_bridge : ℕ := 40
def speed_train (x : ℕ) := (40 * x = bridge_length)
def length_train (x y : ℕ) := (60 * x = bridge_length + y)

-- Stating the problem to be proved
theorem find_train_parameters (x y : ℕ) (h₁ : speed_train x) (h₂ : length_train x y) :
  x = 20 ∧ y = 200 :=
sorry

end find_train_parameters_l1213_121370


namespace hyperbola_n_range_l1213_121301

noncomputable def hyperbola_range_n (m n : ℝ) : Set ℝ :=
  {n | ∃ (m : ℝ), (m^2 + n) + (3 * m^2 - n) = 4 ∧ ((m^2 + n) * (3 * m^2 - n) > 0) }

theorem hyperbola_n_range : ∀ n : ℝ, n ∈ hyperbola_range_n m n ↔ -1 < n ∧ n < 3 :=
by
  sorry

end hyperbola_n_range_l1213_121301


namespace pedoe_inequality_l1213_121316

variables {a b c a' b' c' Δ Δ' : ℝ} {A A' : ℝ}

theorem pedoe_inequality :
  a' ^ 2 * (-a ^ 2 + b ^ 2 + c ^ 2) +
  b' ^ 2 * (a ^ 2 - b ^ 2 + c ^ 2) +
  c' ^ 2 * (a ^ 2 + b ^ 2 - c ^ 2) -
  16 * Δ * Δ' =
  2 * (b * c' - b' * c) ^ 2 +
  8 * b * b' * c * c' * (Real.sin ((A - A') / 2)) ^ 2 := sorry

end pedoe_inequality_l1213_121316


namespace find_positive_integer_cube_root_divisible_by_21_l1213_121398

theorem find_positive_integer_cube_root_divisible_by_21 (m : ℕ) (h1: m = 735) :
  m % 21 = 0 ∧ 9 < (m : ℝ)^(1/3) ∧ (m : ℝ)^(1/3) < 9.1 :=
by {
  sorry
}

end find_positive_integer_cube_root_divisible_by_21_l1213_121398


namespace sum_max_min_expr_l1213_121339

theorem sum_max_min_expr (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) : 
    let expr := (x / |x|) + (|y| / y) - (|x * y| / (x * y))
    max (max expr (expr)) (min expr expr) = -2 :=
sorry

end sum_max_min_expr_l1213_121339


namespace discount_difference_l1213_121306

theorem discount_difference (P : ℝ) (h₁ : 0 < P) : 
  let actual_combined_discount := 1 - (0.75 * 0.85)
  let claimed_discount := 0.40
  actual_combined_discount - claimed_discount = 0.0375 :=
by 
  sorry

end discount_difference_l1213_121306


namespace determine_h_l1213_121399

theorem determine_h (h : ℝ) : (∃ x : ℝ, x = 3 ∧ x^3 - 2 * h * x + 15 = 0) → h = 7 :=
by
  intro hx
  sorry

end determine_h_l1213_121399


namespace solve_a_range_m_l1213_121376

def f (x : ℝ) (a : ℝ) : ℝ := |x - a|

theorem solve_a :
  (∀ x : ℝ, f x 2 ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) ↔ (2 = 2) :=
by {
  sorry
}

theorem range_m :
  (∀ x : ℝ, f (3 * x) 2 + f (x + 3) 2 ≥ m) ↔ (m ≤ 5 / 3) :=
by {
  sorry
}

end solve_a_range_m_l1213_121376


namespace age_ratio_in_4_years_l1213_121325

-- Definitions based on conditions
def Age6YearsAgoVimal := 12
def Age6YearsAgoSaroj := 10
def CurrentAgeSaroj := 16
def CurrentAgeVimal := Age6YearsAgoVimal + 6

-- Lean statement to prove the problem
theorem age_ratio_in_4_years (x : ℕ) 
  (h_ratio : (CurrentAgeVimal + x) / (CurrentAgeSaroj + x) = 11 / 10) :
  x = 4 := 
sorry

end age_ratio_in_4_years_l1213_121325


namespace parking_garage_capacity_l1213_121391

open Nat

-- Definitions from the conditions
def first_level_spaces : Nat := 90
def second_level_spaces : Nat := first_level_spaces + 8
def third_level_spaces : Nat := second_level_spaces + 12
def fourth_level_spaces : Nat := third_level_spaces - 9
def initial_parked_cars : Nat := 100

-- The proof statement
theorem parking_garage_capacity : 
  (first_level_spaces + second_level_spaces + third_level_spaces + fourth_level_spaces - initial_parked_cars) = 299 := 
  by 
    sorry

end parking_garage_capacity_l1213_121391


namespace probability_sequence_l1213_121308

def total_cards := 52
def first_card_is_six_of_diamonds := 1 / total_cards
def remaining_cards := total_cards - 1
def second_card_is_queen_of_hearts (first_card_was_six_of_diamonds : Prop) := 1 / remaining_cards
def probability_six_of_diamonds_and_queen_of_hearts : ℝ :=
  first_card_is_six_of_diamonds * second_card_is_queen_of_hearts sorry

theorem probability_sequence : 
  probability_six_of_diamonds_and_queen_of_hearts = 1 / 2652 := sorry

end probability_sequence_l1213_121308


namespace find_a6_l1213_121344

open Nat

noncomputable def arith_seq (a1 d : ℤ) (n : ℕ) : ℤ :=
  a1 + (n - 1) * d

def a2 := 4
def a4 := 2

theorem find_a6 (a1 d : ℤ) (h_a2 : arith_seq a1 d 2 = a2) (h_a4 : arith_seq a1 d 4 = a4) : 
  arith_seq a1 d 6 = 0 := by
  sorry

end find_a6_l1213_121344


namespace total_marbles_l1213_121321

def mary_marbles := 9
def joan_marbles := 3
def john_marbles := 7

theorem total_marbles :
  mary_marbles + joan_marbles + john_marbles = 19 :=
by
  sorry

end total_marbles_l1213_121321


namespace cone_volume_l1213_121348

theorem cone_volume (p q : ℕ) (a α : ℝ) :
  V = (2 * π * a^3) / (3 * (Real.sin (2 * α)) * (Real.cos (180 * q / (p + q)))^2 * (Real.cos α)) :=
sorry

end cone_volume_l1213_121348


namespace unique_solution_of_system_l1213_121311

theorem unique_solution_of_system :
  ∀ (a : ℝ), (∃ (x y : ℝ), x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0) →
  ((a = 1 ∧ ∃ x y : ℝ, x = -1 ∧ y = 0 ∧ x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0) ∨
  (a = -3 ∧ ∃ x y : ℝ, x = 1 ∧ y = 2 ∧ x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0)) :=
by
  sorry

end unique_solution_of_system_l1213_121311


namespace integer_points_on_segment_l1213_121349

open Int

def is_integer_point (x y : ℝ) : Prop := ∃ (a b : ℤ), x = a ∧ y = b

def f (n : ℕ) : ℕ := 
  if 3 ∣ n then 2
  else 0

theorem integer_points_on_segment (n : ℕ) (hn : 0 < n) :
  (f n) = if 3 ∣ n then 2 else 0 := 
  sorry

end integer_points_on_segment_l1213_121349


namespace puppy_food_total_correct_l1213_121314

def daily_food_first_two_weeks : ℚ := 3 / 4
def weekly_food_first_two_weeks : ℚ := 7 * daily_food_first_two_weeks
def total_food_first_two_weeks : ℚ := 2 * weekly_food_first_two_weeks

def daily_food_following_two_weeks : ℚ := 1
def weekly_food_following_two_weeks : ℚ := 7 * daily_food_following_two_weeks
def total_food_following_two_weeks : ℚ := 2 * weekly_food_following_two_weeks

def today_food : ℚ := 1 / 2

def total_food_over_4_weeks : ℚ :=
  total_food_first_two_weeks + total_food_following_two_weeks + today_food

theorem puppy_food_total_correct :
  total_food_over_4_weeks = 25 := by
  sorry

end puppy_food_total_correct_l1213_121314


namespace probability_neither_red_nor_purple_l1213_121342

theorem probability_neither_red_nor_purple 
    (total_balls : ℕ)
    (white_balls : ℕ) 
    (green_balls : ℕ) 
    (yellow_balls : ℕ) 
    (red_balls : ℕ) 
    (purple_balls : ℕ) 
    (h_total : total_balls = white_balls + green_balls + yellow_balls + red_balls + purple_balls)
    (h_counts : white_balls = 50 ∧ green_balls = 30 ∧ yellow_balls = 8 ∧ red_balls = 9 ∧ purple_balls = 3):
    (88 : ℚ) / 100 = 0.88 :=
by
  sorry

end probability_neither_red_nor_purple_l1213_121342


namespace solve_system_l1213_121360

theorem solve_system :
  ∀ (a1 a2 c1 c2 x y : ℝ),
  (a1 * 5 + 10 = c1) →
  (a2 * 5 + 10 = c2) →
  (a1 * x + 2 * y = a1 - c1) →
  (a2 * x + 2 * y = a2 - c2) →
  (x = -4) ∧ (y = -5) := by
  intros a1 a2 c1 c2 x y h1 h2 h3 h4
  sorry

end solve_system_l1213_121360


namespace pounds_in_a_ton_l1213_121304

-- Definition of variables based on the given conditions
variables (T E D : ℝ)

-- Condition 1: The elephant weighs 3 tons.
def elephant_weight := E = 3 * T

-- Condition 2: The donkey weighs 90% less than the elephant.
def donkey_weight := D = 0.1 * E

-- Condition 3: Their combined weight is 6600 pounds.
def combined_weight := E + D = 6600

-- Main theorem to prove
theorem pounds_in_a_ton (h1 : elephant_weight T E) (h2 : donkey_weight E D) (h3 : combined_weight E D) : T = 2000 :=
by
  sorry

end pounds_in_a_ton_l1213_121304


namespace angle_DEF_EDF_proof_l1213_121322

theorem angle_DEF_EDF_proof (angle_DOE : ℝ) (angle_EOD : ℝ) 
  (h1 : angle_DOE = 130) (h2 : angle_EOD = 90) :
  let angle_DEF := 45
  let angle_EDF := 45
  angle_DEF = 45 ∧ angle_EDF = 45 :=
by
  sorry

end angle_DEF_EDF_proof_l1213_121322


namespace find_solution_set_l1213_121381

-- Define the problem
def absolute_value_equation_solution_set (x : ℝ) : Prop :=
  |x - 2| + |2 * x - 3| = |3 * x - 5|

-- Define the expected solution set
def solution_set (x : ℝ) : Prop :=
  x ≤ 3 / 2 ∨ 2 ≤ x

-- The proof problem statement
theorem find_solution_set :
  ∀ x : ℝ, absolute_value_equation_solution_set x ↔ solution_set x :=
sorry -- No proof required, so we use 'sorry' to skip the proof

end find_solution_set_l1213_121381


namespace range_m_l1213_121365

noncomputable def circle_c (x y : ℝ) : Prop := (x - 4) ^ 2 + (y - 3) ^ 2 = 4

def point_A (m : ℝ) : ℝ × ℝ := (-m, 0)
def point_B (m : ℝ) : ℝ × ℝ := (m, 0)

theorem range_m (m : ℝ) (P : ℝ × ℝ) :
  circle_c P.1 P.2 ∧ m > 0 ∧ (∃ (a b : ℝ), P = (a, b) ∧ (a + m) * (a - m) + b ^ 2 = 0) → m ∈ Set.Icc 3 7 :=
sorry

end range_m_l1213_121365


namespace quadratic_inequality_solution_set_l1213_121364

theorem quadratic_inequality_solution_set (x : ℝ) : (x + 3) * (2 - x) < 0 ↔ x < -3 ∨ x > 2 := 
sorry

end quadratic_inequality_solution_set_l1213_121364


namespace initial_distance_is_18_l1213_121343

-- Step a) Conditions and Definitions
def distance_covered (v t d : ℝ) : Prop := 
  d = v * t

def increased_speed_time (v t d : ℝ) : Prop := 
  d = (v + 1) * (3 / 4 * t)

def decreased_speed_time (v t d : ℝ) : Prop := 
  d = (v - 1) * (t + 3)

-- Step c) Mathematically Equivalent Proof Problem
theorem initial_distance_is_18 (v t d : ℝ) 
  (h1 : distance_covered v t d) 
  (h2 : increased_speed_time v t d) 
  (h3 : decreased_speed_time v t d) : 
  d = 18 :=
sorry

end initial_distance_is_18_l1213_121343


namespace prime_in_choices_l1213_121313

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def twenty := 20
def twenty_one := 21
def twenty_three := 23
def twenty_five := 25
def twenty_seven := 27

theorem prime_in_choices :
  is_prime twenty_three ∧ ¬ is_prime twenty ∧ ¬ is_prime twenty_one ∧ ¬ is_prime twenty_five ∧ ¬ is_prime twenty_seven :=
by
  sorry

end prime_in_choices_l1213_121313


namespace distance_between_stripes_correct_l1213_121309

noncomputable def distance_between_stripes : ℝ :=
  let base1 := 20
  let height1 := 50
  let base2 := 65
  let area := base1 * height1
  let d := area / base2
  d

theorem distance_between_stripes_correct : distance_between_stripes = 200 / 13 := by
  sorry

end distance_between_stripes_correct_l1213_121309


namespace most_cost_effective_80_oranges_l1213_121333

noncomputable def cost_of_oranges (p1 p2 p3 : ℕ) (q1 q2 q3 : ℕ) : ℕ :=
  let cost_per_orange_p1 := p1 / q1
  let cost_per_orange_p2 := p2 / q2
  let cost_per_orange_p3 := p3 / q3
  if cost_per_orange_p3 ≤ cost_per_orange_p2 ∧ cost_per_orange_p3 ≤ cost_per_orange_p1 then
    (80 / q3) * p3
  else if cost_per_orange_p2 ≤ cost_per_orange_p1 then
    (80 / q2) * p2
  else
    (80 / q1) * p1

theorem most_cost_effective_80_oranges :
  cost_of_oranges 35 45 95 6 9 20 = 380 :=
by sorry

end most_cost_effective_80_oranges_l1213_121333


namespace man_speed_with_stream_is_4_l1213_121302

noncomputable def man's_speed_with_stream (Vm Vs : ℝ) : ℝ := Vm + Vs

theorem man_speed_with_stream_is_4 (Vm : ℝ) (Vs : ℝ) 
  (h1 : Vm - Vs = 4) 
  (h2 : Vm = 4) : man's_speed_with_stream Vm Vs = 4 :=
by 
  -- The proof is omitted as per instructions
  sorry

end man_speed_with_stream_is_4_l1213_121302


namespace double_counted_toddlers_l1213_121369

def number_of_toddlers := 21
def missed_toddlers := 3
def billed_count := 26

theorem double_counted_toddlers : 
  ∃ (D : ℕ), (number_of_toddlers + D - missed_toddlers = billed_count) ∧ D = 8 :=
by
  sorry

end double_counted_toddlers_l1213_121369


namespace find_a_and_b_l1213_121334

-- Given conditions
def curve (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

theorem find_a_and_b (a b : ℝ) :
  (∀ x, ∀ y, tangent_line x y → y = b ∧ x = 0) ∧
  (∀ x, ∀ y, y = curve x a b) →
  a = 1 ∧ b = 1 :=
by
  sorry

end find_a_and_b_l1213_121334


namespace calculate_expression_l1213_121386

theorem calculate_expression :
  4 + ((-2)^2) * 2 + (-36) / 4 = 3 := by
  sorry

end calculate_expression_l1213_121386


namespace sum_of_integers_eq_28_24_23_l1213_121363

theorem sum_of_integers_eq_28_24_23 
  (a b : ℕ) 
  (h1 : a * b + a + b = 143)
  (h2 : Nat.gcd a b = 1)
  (h3 : a < 30)
  (h4 : b < 30) 
  : a + b = 28 ∨ a + b = 24 ∨ a + b = 23 :=
by
  sorry

end sum_of_integers_eq_28_24_23_l1213_121363


namespace mary_donated_books_l1213_121393

theorem mary_donated_books 
  (s : ℕ) (b_c : ℕ) (b_b : ℕ) (b_y : ℕ) (g_d : ℕ) (g_m : ℕ) (e : ℕ) (s_s : ℕ) 
  (total : ℕ) (out_books : ℕ) (d : ℕ)
  (h1 : s = 72)
  (h2 : b_c = 12)
  (h3 : b_b = 5)
  (h4 : b_y = 2)
  (h5 : g_d = 1)
  (h6 : g_m = 4)
  (h7 : e = 81)
  (h8 : s_s = 3)
  (ht : total = s + b_c + b_b + b_y + g_d + g_m)
  (ho : out_books = total - e)
  (hd : d = out_books - s_s) :
  d = 12 :=
by { sorry }

end mary_donated_books_l1213_121393


namespace jackson_weeks_of_school_l1213_121315

def jackson_sandwich_per_week : ℕ := 2

def missed_wednesdays : ℕ := 1
def missed_fridays : ℕ := 2
def total_missed_sandwiches : ℕ := missed_wednesdays + missed_fridays

def total_sandwiches_eaten : ℕ := 69

def total_sandwiches_without_missing : ℕ := total_sandwiches_eaten + total_missed_sandwiches

def calculate_weeks_of_school (total_sandwiches : ℕ) (sandwiches_per_week : ℕ) : ℕ :=
total_sandwiches / sandwiches_per_week

theorem jackson_weeks_of_school : calculate_weeks_of_school total_sandwiches_without_missing jackson_sandwich_per_week = 36 :=
by
  sorry

end jackson_weeks_of_school_l1213_121315


namespace like_apple_orange_mango_l1213_121396

theorem like_apple_orange_mango (A B C: ℕ) 
  (h1: A = 40) 
  (h2: B = 7) 
  (h3: C = 10) 
  (total: ℕ) 
  (h_total: total = 47) 
: ∃ x: ℕ, 40 + (10 - x) + x = 47 ∧ x = 3 := 
by 
  sorry

end like_apple_orange_mango_l1213_121396
