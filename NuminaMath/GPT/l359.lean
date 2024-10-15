import Mathlib

namespace NUMINAMATH_GPT_animals_left_in_barn_l359_35949

-- Define the conditions
def num_pigs : Nat := 156
def num_cows : Nat := 267
def num_sold : Nat := 115

-- Define the question
def num_left := num_pigs + num_cows - num_sold

-- State the theorem
theorem animals_left_in_barn : num_left = 308 :=
by
  sorry

end NUMINAMATH_GPT_animals_left_in_barn_l359_35949


namespace NUMINAMATH_GPT_determine_gallons_l359_35968

def current_amount : ℝ := 7.75
def desired_total : ℝ := 14.75
def needed_to_add (x : ℝ) : Prop := desired_total = current_amount + x

theorem determine_gallons : needed_to_add 7 :=
by
  sorry

end NUMINAMATH_GPT_determine_gallons_l359_35968


namespace NUMINAMATH_GPT_remainder_when_divided_by_x_minus_1_is_minus_2_l359_35919

def p (x : ℝ) : ℝ := x^4 - 2*x^2 + 4*x - 5

theorem remainder_when_divided_by_x_minus_1_is_minus_2 : (p 1) = -2 := 
by 
  -- Proof not required
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_x_minus_1_is_minus_2_l359_35919


namespace NUMINAMATH_GPT_train_A_start_time_l359_35980

theorem train_A_start_time :
  let distance := 155 -- km
  let speed_A := 20 -- km/h
  let speed_B := 25 -- km/h
  let start_B := 8 -- a.m.
  let meet_time := 11 -- a.m.
  let travel_time_B := meet_time - start_B -- time in hours for train B from 8 a.m. to 11 a.m.
  let distance_B := speed_B * travel_time_B -- distance covered by train B
  let distance_A := distance - distance_B -- remaining distance covered by train A
  let travel_time_A := distance_A / speed_A -- time for train A to cover its distance
  let start_A := meet_time - travel_time_A -- start time for train A
  start_A = 7 := by
  sorry

end NUMINAMATH_GPT_train_A_start_time_l359_35980


namespace NUMINAMATH_GPT_quadratic_equiv_original_correct_transformation_l359_35960

theorem quadratic_equiv_original :
  (5 + 3*Real.sqrt 2) * x^2 + (3 + Real.sqrt 2) * x - 3 = 
  (7 + 4 * Real.sqrt 3) * x^2 + (2 + Real.sqrt 3) * x - 2 :=
sorry

theorem correct_transformation :
  ∃ r : ℝ, r = (9 / 7) - (4 * Real.sqrt 2 / 7) ∧ 
  ((5 + 3 * Real.sqrt 2) * x^2 + (3 + Real.sqrt 2) * x - 3) = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_equiv_original_correct_transformation_l359_35960


namespace NUMINAMATH_GPT_yellow_more_than_green_l359_35928

-- Given conditions
def G : ℕ := 90               -- Number of green buttons
def B : ℕ := 85               -- Number of blue buttons
def T : ℕ := 275              -- Total number of buttons
def Y : ℕ := 100              -- Number of yellow buttons (derived from conditions)

-- Mathematically equivalent proof problem
theorem yellow_more_than_green : (90 + 100 + 85 = 275) → (100 - 90 = 10) :=
by sorry

end NUMINAMATH_GPT_yellow_more_than_green_l359_35928


namespace NUMINAMATH_GPT_find_rate_percent_l359_35945

theorem find_rate_percent
  (P : ℝ) (SI : ℝ) (T : ℝ) (R : ℝ) 
  (hP : P = 1600)
  (hSI : SI = 200)
  (hT : T = 4)
  (hSI_eq : SI = (P * R * T) / 100) :
  R = 3.125 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_rate_percent_l359_35945


namespace NUMINAMATH_GPT_donation_total_correct_l359_35905

noncomputable def total_donation (t : ℝ) (y : ℝ) (x : ℝ) : ℝ :=
  t + t + x
  
theorem donation_total_correct (t : ℝ) (y : ℝ) (x : ℝ)
  (h1 : t = 570.00) (h2 : y = 140.00) (h3 : t = x + y) : total_donation t y x = 1570.00 :=
by
  sorry

end NUMINAMATH_GPT_donation_total_correct_l359_35905


namespace NUMINAMATH_GPT_percentage_non_honda_red_cars_l359_35935

theorem percentage_non_honda_red_cars 
  (total_cars : ℕ)
  (honda_cars : ℕ)
  (toyota_cars : ℕ)
  (ford_cars : ℕ)
  (other_cars : ℕ)
  (perc_red_honda : ℕ)
  (perc_red_toyota : ℕ)
  (perc_red_ford : ℕ)
  (perc_red_other : ℕ)
  (perc_total_red : ℕ)
  (hyp_total_cars : total_cars = 900)
  (hyp_honda_cars : honda_cars = 500)
  (hyp_toyota_cars : toyota_cars = 200)
  (hyp_ford_cars : ford_cars = 150)
  (hyp_other_cars : other_cars = 50)
  (hyp_perc_red_honda : perc_red_honda = 90)
  (hyp_perc_red_toyota : perc_red_toyota = 75)
  (hyp_perc_red_ford : perc_red_ford = 30)
  (hyp_perc_red_other : perc_red_other = 20)
  (hyp_perc_total_red : perc_total_red = 60) :
  (205 / 400) * 100 = 51.25 := 
by {
  sorry
}

end NUMINAMATH_GPT_percentage_non_honda_red_cars_l359_35935


namespace NUMINAMATH_GPT_range_of_m_l359_35982

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x ≠ 1 ∧ (m + 3) / (x - 1) = 1) ↔ m > -4 ∧ m ≠ -3 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_m_l359_35982


namespace NUMINAMATH_GPT_rectangular_field_area_l359_35916

theorem rectangular_field_area (a b c : ℕ) (h1 : a = 15) (h2 : c = 17)
  (h3 : a * a + b * b = c * c) : a * b = 120 := by
  sorry

end NUMINAMATH_GPT_rectangular_field_area_l359_35916


namespace NUMINAMATH_GPT_symmetry_condition_l359_35929

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin (x + π / 3)

theorem symmetry_condition (ϕ : ℝ) (hϕ : |ϕ| ≤ π / 2)
    (hxy: ∀ x : ℝ, f (x + ϕ) = f (-x + ϕ)) : ϕ = π / 6 :=
by
  -- Since the problem specifically asks for the statement only and not the proof steps,
  -- a "sorry" is used to skip the proof content.
  sorry

end NUMINAMATH_GPT_symmetry_condition_l359_35929


namespace NUMINAMATH_GPT_number_of_cds_l359_35999

-- Define the constants
def total_money : ℕ := 37
def cd_price : ℕ := 14
def cassette_price : ℕ := 9

theorem number_of_cds (total_money cd_price cassette_price : ℕ) (h_total_money : total_money = 37) (h_cd_price : cd_price = 14) (h_cassette_price : cassette_price = 9) :
  ∃ n : ℕ, n * cd_price + cassette_price = total_money ∧ n = 2 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end NUMINAMATH_GPT_number_of_cds_l359_35999


namespace NUMINAMATH_GPT_Koschei_no_equal_coins_l359_35970

theorem Koschei_no_equal_coins (a : Fin 6 → ℕ)
  (initial_condition : a 0 = 1 ∧ a 1 = 0 ∧ a 2 = 0 ∧ a 3 = 0 ∧ a 4 = 0 ∧ a 5 = 0) :
  ¬ ( ∃ k : ℕ, ( ( ∀ i : Fin 6, a i = k ) ) ) :=
by
  sorry

end NUMINAMATH_GPT_Koschei_no_equal_coins_l359_35970


namespace NUMINAMATH_GPT_Bobby_candy_chocolate_sum_l359_35996

/-
  Bobby ate 33 pieces of candy, then ate 4 more, and he also ate 14 pieces of chocolate.
  Prove that the total number of pieces of candy and chocolate he ate altogether is 51.
-/

theorem Bobby_candy_chocolate_sum :
  let initial_candy := 33
  let more_candy := 4
  let chocolate := 14
  let total_candy := initial_candy + more_candy
  total_candy + chocolate = 51 :=
by
  -- The theorem asserts the problem; apologies, the proof is not required here.
  sorry

end NUMINAMATH_GPT_Bobby_candy_chocolate_sum_l359_35996


namespace NUMINAMATH_GPT_professional_tax_correct_l359_35967

-- Define the total income and professional deductions
def total_income : ℝ := 50000
def professional_deductions : ℝ := 35000

-- Define the tax rates
def tax_rate_ndfl : ℝ := 0.13
def tax_rate_simplified_income : ℝ := 0.06
def tax_rate_simplified_income_minus_exp : ℝ := 0.15
def tax_rate_professional_income : ℝ := 0.04

-- Define the expected tax amount
def expected_tax_professional_income : ℝ := 2000

-- Define a function to calculate the professional income tax for self-employed individuals
def calculate_professional_income_tax (income : ℝ) (rate : ℝ) : ℝ :=
  income * rate

-- Define the main theorem to assert the correctness of the tax calculation
theorem professional_tax_correct :
  calculate_professional_income_tax total_income tax_rate_professional_income = expected_tax_professional_income :=
by
  sorry

end NUMINAMATH_GPT_professional_tax_correct_l359_35967


namespace NUMINAMATH_GPT_angle_of_inclination_range_l359_35948

theorem angle_of_inclination_range (a : ℝ) :
  (∃ m : ℝ, ax + (a + 1)*m + 2 = 0 ∧ (m < 0 ∨ m > 1)) ↔ (a < -1/2 ∨ a > 0) := sorry

end NUMINAMATH_GPT_angle_of_inclination_range_l359_35948


namespace NUMINAMATH_GPT_greatest_gcd_f_l359_35992

def f (n : ℕ) : ℕ := 70 + n^2

def g (n : ℕ) : ℕ := Nat.gcd (f n) (f (n + 1))

theorem greatest_gcd_f (n : ℕ) (h : 0 < n) : g n = 281 :=
  sorry

end NUMINAMATH_GPT_greatest_gcd_f_l359_35992


namespace NUMINAMATH_GPT_largest_value_among_expressions_l359_35936

def expA : ℕ := 3 + 1 + 2 + 4
def expB : ℕ := 3 * 1 + 2 + 4
def expC : ℕ := 3 + 1 * 2 + 4
def expD : ℕ := 3 + 1 + 2 * 4
def expE : ℕ := 3 * 1 * 2 * 4

theorem largest_value_among_expressions :
  expE > expA ∧ expE > expB ∧ expE > expC ∧ expE > expD :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_largest_value_among_expressions_l359_35936


namespace NUMINAMATH_GPT_sam_dad_gave_39_nickels_l359_35900

-- Define the initial conditions
def initial_pennies : ℕ := 49
def initial_nickels : ℕ := 24
def given_quarters : ℕ := 31
def dad_given_nickels : ℕ := 63 - initial_nickels

-- Statement to prove
theorem sam_dad_gave_39_nickels 
    (total_nickels_after : ℕ) 
    (initial_nickels : ℕ) 
    (final_nickels : ℕ := total_nickels_after - initial_nickels) : 
    final_nickels = 39 :=
sorry

end NUMINAMATH_GPT_sam_dad_gave_39_nickels_l359_35900


namespace NUMINAMATH_GPT_lucas_notation_sum_l359_35946

-- Define what each representation in Lucas's notation means
def lucasValue : String → Int
| "0" => 0
| s => -((s.length) - 1)

-- Define the question as a Lean theorem
theorem lucas_notation_sum :
  lucasValue "000" + lucasValue "0000" = lucasValue "000000" :=
by
  sorry

end NUMINAMATH_GPT_lucas_notation_sum_l359_35946


namespace NUMINAMATH_GPT_min_value_y_l359_35933

theorem min_value_y (x : ℝ) (hx : x > 2) : 
  ∃ x, x > 2 ∧ (∀ y, y = (x^2 - 4*x + 8) / (x - 2) → y ≥ 4 ∧ y = 4 ↔ x = 4) :=
sorry

end NUMINAMATH_GPT_min_value_y_l359_35933


namespace NUMINAMATH_GPT_find_r_l359_35917

variable (n : ℕ) (q r : ℝ)

-- n must be a positive natural number
axiom n_pos : n > 0

-- q is a positive real number and not equal to 1
axiom q_pos : q > 0
axiom q_ne_one : q ≠ 1

-- Define the sequence sum S_n according to the problem statement
def S_n (n : ℕ) (q r : ℝ) : ℝ := q^n + r

-- The goal is to prove that the correct value of r is -1
theorem find_r : r = -1 :=
sorry

end NUMINAMATH_GPT_find_r_l359_35917


namespace NUMINAMATH_GPT_handshake_problem_l359_35969

-- Defining the necessary elements:
def num_people : Nat := 12
def num_handshakes_per_person : Nat := num_people - 2

-- Defining the total number of handshakes. Each handshake is counted twice.
def total_handshakes : Nat := (num_people * num_handshakes_per_person) / 2

-- The theorem statement:
theorem handshake_problem : total_handshakes = 60 :=
by
  sorry

end NUMINAMATH_GPT_handshake_problem_l359_35969


namespace NUMINAMATH_GPT_min_height_box_l359_35923

noncomputable def min_height (x : ℝ) : ℝ :=
  if h : x ≥ (5 : ℝ) then x + 5 else 0

theorem min_height_box (x : ℝ) (hx : 3*x^2 + 10*x - 65 ≥ 0) : min_height x = 10 :=
by
  sorry

end NUMINAMATH_GPT_min_height_box_l359_35923


namespace NUMINAMATH_GPT_cube_volume_and_diagonal_l359_35901

theorem cube_volume_and_diagonal (A : ℝ) (s : ℝ) (V : ℝ) (d : ℝ) 
  (h1 : A = 864)
  (h2 : 6 * s^2 = A)
  (h3 : V = s^3)
  (h4 : d = s * Real.sqrt 3) :
  V = 1728 ∧ d = 12 * Real.sqrt 3 :=
by 
  sorry

end NUMINAMATH_GPT_cube_volume_and_diagonal_l359_35901


namespace NUMINAMATH_GPT_greatest_integer_y_l359_35934

theorem greatest_integer_y (y : ℤ) : abs (3 * y - 4) ≤ 21 → y ≤ 8 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_y_l359_35934


namespace NUMINAMATH_GPT_no_bounded_sequence_a1_gt_2015_l359_35984

theorem no_bounded_sequence_a1_gt_2015 (a1 : ℚ) (h_a1 : a1 > 2015) : 
  ∀ (a_n : ℕ → ℚ), a_n 1 = a1 → 
  (∀ (n : ℕ), ∃ (p_n q_n : ℕ), p_n > 0 ∧ q_n > 0 ∧ (p_n.gcd q_n = 1) ∧ (a_n n = p_n / q_n) ∧ 
  (a_n (n + 1) = (p_n^2 + 2015) / (p_n * q_n))) → 
  ∃ (M : ℚ), ∀ (n : ℕ), a_n n ≤ M → 
  False :=
sorry

end NUMINAMATH_GPT_no_bounded_sequence_a1_gt_2015_l359_35984


namespace NUMINAMATH_GPT_no_real_m_for_parallel_lines_l359_35985

theorem no_real_m_for_parallel_lines : 
  ∀ (m : ℝ), ∃ (l1 l2 : ℝ × ℝ × ℝ), 
  (l1 = (2, (m + 1), 4)) ∧ (l2 = (m, 3, 4)) ∧ 
  ( ∀ (m : ℝ), -2 / (m + 1) = -m / 3 → false ) :=
by sorry

end NUMINAMATH_GPT_no_real_m_for_parallel_lines_l359_35985


namespace NUMINAMATH_GPT_speed_of_second_half_l359_35930

theorem speed_of_second_half (total_time : ℕ) (speed_first_half : ℕ) (total_distance : ℕ)
  (h1 : total_time = 15) (h2 : speed_first_half = 21) (h3 : total_distance = 336) :
  2 * total_distance / total_time - speed_first_half * (total_time / 2) / (total_time / 2) = 24 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_speed_of_second_half_l359_35930


namespace NUMINAMATH_GPT_problem1_problem2_l359_35922

theorem problem1 (x : ℝ) (a : ℝ) (h : a = 1) (hp : a < x ∧ x < 3 * a) (hq : 2 < x ∧ x < 3) : 2 < x ∧ x < 3 := 
by
  sorry

theorem problem2 (x : ℝ) (a : ℝ) (hp : 0 < a ∧ a < x ∧ x < 3 * a) (hq : 2 < x ∧ x < 3) (hsuff : ∀ (a x : ℝ), (2 < x ∧ x < 3) → a < x ∧ x < 3 * a) : 1 ≤ a ∧ a ≤ 2 := 
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l359_35922


namespace NUMINAMATH_GPT_rect_tiling_l359_35963

theorem rect_tiling (a b : ℕ) : ∃ (w h : ℕ), w = max 1 (2 * a) ∧ h = 2 * b ∧ (∃ f : ℕ → ℕ → (ℕ × ℕ), ∀ i j, (i < w ∧ j < h → f i j = (a, b))) := sorry

end NUMINAMATH_GPT_rect_tiling_l359_35963


namespace NUMINAMATH_GPT_sally_total_fries_is_50_l359_35987

-- Definitions for the conditions
def sally_initial_fries : ℕ := 14
def mark_initial_fries : ℕ := 3 * 12
def mark_fraction_given_to_sally : ℕ := mark_initial_fries / 3
def jessica_total_cm_of_fries : ℕ := 240
def fry_length_cm : ℕ := 5
def jessica_total_fries : ℕ := jessica_total_cm_of_fries / fry_length_cm
def jessica_fraction_given_to_sally : ℕ := jessica_total_fries / 2

-- Definition for the question
def total_fries_sally_has (sally_initial_fries mark_fraction_given_to_sally jessica_fraction_given_to_sally : ℕ) : ℕ :=
  sally_initial_fries + mark_fraction_given_to_sally + jessica_fraction_given_to_sally

-- The theorem to be proved
theorem sally_total_fries_is_50 :
  total_fries_sally_has sally_initial_fries mark_fraction_given_to_sally jessica_fraction_given_to_sally = 50 :=
sorry

end NUMINAMATH_GPT_sally_total_fries_is_50_l359_35987


namespace NUMINAMATH_GPT_weight_of_newcomer_l359_35974

theorem weight_of_newcomer (avg_old W_initial : ℝ) 
  (h_weight_range : 400 ≤ W_initial ∧ W_initial ≤ 420)
  (h_avg_increase : avg_old + 3.5 = (W_initial - 47 + W_new) / 6)
  (h_person_replaced : 47 = 47) :
  W_new = 68 := 
sorry

end NUMINAMATH_GPT_weight_of_newcomer_l359_35974


namespace NUMINAMATH_GPT_intersection_M_N_l359_35924

-- Definitions of the sets M and N
def M : Set ℤ := {-3, -2, -1}
def N : Set ℤ := { x | -2 < x ∧ x < 3 }

-- The theorem stating that the intersection of M and N is {-1}
theorem intersection_M_N : M ∩ N = {-1} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l359_35924


namespace NUMINAMATH_GPT_problem_1_problem_2_l359_35908

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l359_35908


namespace NUMINAMATH_GPT_directrix_of_parabola_l359_35959

-- Define the given condition: the equation of the parabola
def given_parabola (x : ℝ) : ℝ := 4 * x ^ 2

-- State the theorem to be proven
theorem directrix_of_parabola : 
  (∀ x : ℝ, given_parabola x = 4 * x ^ 2) → 
  (y = -1 / 16) :=
sorry

end NUMINAMATH_GPT_directrix_of_parabola_l359_35959


namespace NUMINAMATH_GPT_find_x9_y9_l359_35937

theorem find_x9_y9 (x y : ℝ) (h1 : x^3 + y^3 = 7) (h2 : x^6 + y^6 = 49) : x^9 + y^9 = 343 :=
by
  sorry

end NUMINAMATH_GPT_find_x9_y9_l359_35937


namespace NUMINAMATH_GPT_largest_divisor_60_36_divisible_by_3_l359_35918

theorem largest_divisor_60_36_divisible_by_3 : 
  ∃ x, (x ∣ 60) ∧ (x ∣ 36) ∧ (3 ∣ x) ∧ (∀ y, (y ∣ 60) → (y ∣ 36) → (3 ∣ y) → y ≤ x) ∧ x = 12 :=
sorry

end NUMINAMATH_GPT_largest_divisor_60_36_divisible_by_3_l359_35918


namespace NUMINAMATH_GPT_P_intersect_Q_empty_l359_35983

def is_element_of_P (x : ℝ) : Prop :=
  ∃ (k : ℤ), x = k / 2 + 1 / 4

def is_element_of_Q (x : ℝ) : Prop :=
  ∃ (k : ℤ), x = k / 2 + 1 / 2

theorem P_intersect_Q_empty : ∀ x, is_element_of_P x → is_element_of_Q x → false :=
by
  intro x hP hQ
  sorry

end NUMINAMATH_GPT_P_intersect_Q_empty_l359_35983


namespace NUMINAMATH_GPT_mass_percentage_H_calculation_l359_35943

noncomputable def molar_mass_CaH2 : ℝ := 42.09
noncomputable def molar_mass_H2O : ℝ := 18.015
noncomputable def molar_mass_H2SO4 : ℝ := 98.079

noncomputable def moles_CaH2 : ℕ := 3
noncomputable def moles_H2O : ℕ := 4
noncomputable def moles_H2SO4 : ℕ := 2

noncomputable def mass_H_CaH2 : ℝ := 3 * 2 * 1.008
noncomputable def mass_H_H2O : ℝ := 4 * 2 * 1.008
noncomputable def mass_H_H2SO4 : ℝ := 2 * 2 * 1.008

noncomputable def total_mass_H : ℝ :=
  mass_H_CaH2 + mass_H_H2O + mass_H_H2SO4

noncomputable def total_mass_mixture : ℝ :=
  (moles_CaH2 * molar_mass_CaH2) + (moles_H2O * molar_mass_H2O) + (moles_H2SO4 * molar_mass_H2SO4)

noncomputable def mass_percentage_H : ℝ :=
  (total_mass_H / total_mass_mixture) * 100

theorem mass_percentage_H_calculation :
  abs (mass_percentage_H - 4.599) < 0.001 :=
by
  sorry

end NUMINAMATH_GPT_mass_percentage_H_calculation_l359_35943


namespace NUMINAMATH_GPT_tangent_line_equation_at_1_l359_35976

-- Define the function f and the point of tangency
def f (x : ℝ) : ℝ := x^2 + 2 * x
def p : ℝ × ℝ := (1, f 1)

-- Statement of the theorem
theorem tangent_line_equation_at_1 :
  ∃ a b c : ℝ, (∀ x y : ℝ, y = f x → y - p.2 = a * (x - p.1)) ∧
               4 * (p.1 : ℝ) - (p.2 : ℝ) - 1 = 0 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_tangent_line_equation_at_1_l359_35976


namespace NUMINAMATH_GPT_train_rate_first_hour_l359_35990

-- Define the conditions
def rateAtFirstHour (r : ℕ) : Prop :=
  (11 / 2) * (r + (r + 100)) = 660

-- Prove the rate is 10 mph
theorem train_rate_first_hour (r : ℕ) : rateAtFirstHour r → r = 10 :=
by 
  sorry

end NUMINAMATH_GPT_train_rate_first_hour_l359_35990


namespace NUMINAMATH_GPT_part1_part2_l359_35953

def unitPrices (x : ℕ) (y : ℕ) : Prop :=
  (20 * x = 16 * (y + 20)) ∧ (x = y + 20)

def maxBoxes (a : ℕ) : Prop :=
  ∀ b, (100 * a + 80 * b ≤ 4600) → (a + b = 50)

theorem part1 (x : ℕ) :
  unitPrices x (x - 20) → x = 100 ∧ (x - 20 = 80) :=
by
  sorry

theorem part2 :
  maxBoxes 30 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l359_35953


namespace NUMINAMATH_GPT_sum_of_coeff_l359_35951

theorem sum_of_coeff (x y : ℕ) (n : ℕ) (h : 2 * x + y = 3) : (2 * x + y) ^ n = 3^n := 
by
  sorry

end NUMINAMATH_GPT_sum_of_coeff_l359_35951


namespace NUMINAMATH_GPT_neg_exists_equiv_forall_l359_35975

theorem neg_exists_equiv_forall :
  (¬ ∃ x : ℝ, x^2 - x + 4 < 0) ↔ (∀ x : ℝ, x^2 - x + 4 ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_neg_exists_equiv_forall_l359_35975


namespace NUMINAMATH_GPT_intersection_condition_l359_35914

noncomputable def M : Set ℝ := {x | -1 ≤ x ∧ x < 2}
noncomputable def N (k : ℝ) : Set ℝ := {x | x ≤ k}

theorem intersection_condition (k : ℝ) (h : M ⊆ N k) : k ≥ 2 :=
  sorry

end NUMINAMATH_GPT_intersection_condition_l359_35914


namespace NUMINAMATH_GPT_count_L_shapes_l359_35942

theorem count_L_shapes (m n : ℕ) (hm : 1 ≤ m) (hn : 1 ≤ n) : 
  ∃ k, k = 4 * (m - 1) * (n - 1) :=
by
  sorry

end NUMINAMATH_GPT_count_L_shapes_l359_35942


namespace NUMINAMATH_GPT_integer_xyz_zero_l359_35989

theorem integer_xyz_zero (x y z : ℤ) (h : x^2 + y^2 + z^2 = 2 * x * y * z) : x = 0 ∧ y = 0 ∧ z = 0 := 
sorry

end NUMINAMATH_GPT_integer_xyz_zero_l359_35989


namespace NUMINAMATH_GPT_math_solution_l359_35977

noncomputable def math_problem (x y z : ℝ) : Prop :=
  (0 ≤ x) ∧ (0 ≤ y) ∧ (0 ≤ z) ∧ (x + y + z = 1) → 
  (x^2 * y^2 + y^2 * z^2 + z^2 * x^2 + x^2 * y^2 * z^2 ≤ 1 / 16)

theorem math_solution (x y z : ℝ) :
  math_problem x y z := 
by
  sorry

end NUMINAMATH_GPT_math_solution_l359_35977


namespace NUMINAMATH_GPT_laptop_sticker_price_l359_35978

theorem laptop_sticker_price (x : ℝ) (h1 : 0.8 * x - 120 = y) (h2 : 0.7 * x = z) (h3 : y + 25 = z) : x = 950 :=
sorry

end NUMINAMATH_GPT_laptop_sticker_price_l359_35978


namespace NUMINAMATH_GPT_project_completion_time_l359_35971

-- Definitions for conditions
def a_rate : ℚ := 1 / 20
def b_rate : ℚ := 1 / 30
def combined_rate : ℚ := a_rate + b_rate

-- Total days to complete the project
def total_days (x : ℚ) : Prop :=
  (x - 5) * a_rate + x * b_rate = 1

-- The theorem to be proven
theorem project_completion_time : ∃ (x : ℚ), total_days x ∧ x = 15 := by
  sorry

end NUMINAMATH_GPT_project_completion_time_l359_35971


namespace NUMINAMATH_GPT_fraction_of_second_year_students_not_declared_major_l359_35909

theorem fraction_of_second_year_students_not_declared_major (T : ℕ) :
  (1 / 2 : ℝ) * (1 - (1 / 3 * (1 / 5))) = 7 / 15 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_second_year_students_not_declared_major_l359_35909


namespace NUMINAMATH_GPT_length_of_AC_l359_35962

-- Define the conditions: lengths and angle
def AB : ℝ := 10
def BC : ℝ := 10
def CD : ℝ := 15
def DA : ℝ := 15
def angle_ADC : ℝ := 120

-- Prove the length of diagonal AC is 15*sqrt(3)
theorem length_of_AC : 
  (CD ^ 2 + DA ^ 2 - 2 * CD * DA * Real.cos (angle_ADC * Real.pi / 180)) = (15 * Real.sqrt 3) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_length_of_AC_l359_35962


namespace NUMINAMATH_GPT_roger_current_money_l359_35944

def roger_initial_money : ℕ := 16
def roger_birthday_money : ℕ := 28
def roger_spent_money : ℕ := 25

theorem roger_current_money : roger_initial_money + roger_birthday_money - roger_spent_money = 19 := 
by sorry

end NUMINAMATH_GPT_roger_current_money_l359_35944


namespace NUMINAMATH_GPT_law_school_student_count_l359_35964

theorem law_school_student_count 
    (business_students : ℕ)
    (sibling_pairs : ℕ)
    (selection_probability : ℚ)
    (L : ℕ)
    (h1 : business_students = 500)
    (h2 : sibling_pairs = 30)
    (h3 : selection_probability = 7.500000000000001e-5) :
    L = 8000 :=
by
  sorry

end NUMINAMATH_GPT_law_school_student_count_l359_35964


namespace NUMINAMATH_GPT_max_items_sum_l359_35931

theorem max_items_sum (m n : ℕ) (h : 5 * m + 17 * n = 203) : m + n ≤ 31 :=
sorry

end NUMINAMATH_GPT_max_items_sum_l359_35931


namespace NUMINAMATH_GPT_minimum_value_of_x_plus_y_l359_35955

theorem minimum_value_of_x_plus_y
  (x y : ℝ)
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : (1 / y) + (4 / x) = 1) : 
  x + y = 9 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_x_plus_y_l359_35955


namespace NUMINAMATH_GPT_sum_of_integers_75_to_95_l359_35902

def arithmeticSumOfIntegers (a l : ℕ) : ℕ :=
  let n := l - a + 1
  n / 2 * (a + l)

theorem sum_of_integers_75_to_95 : arithmeticSumOfIntegers 75 95 = 1785 :=
  by
  sorry

end NUMINAMATH_GPT_sum_of_integers_75_to_95_l359_35902


namespace NUMINAMATH_GPT_find_x_l359_35913

theorem find_x (x : ℝ) : 
  let a := (4, 2)
  let b := (x, 3)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = -3 / 2 :=
by
  intros a b h
  sorry

end NUMINAMATH_GPT_find_x_l359_35913


namespace NUMINAMATH_GPT_roger_money_in_january_l359_35903

theorem roger_money_in_january (x : ℝ) (h : (x - 20) + 46 = 71) : x = 45 :=
sorry

end NUMINAMATH_GPT_roger_money_in_january_l359_35903


namespace NUMINAMATH_GPT_geometric_sequence_l359_35966

theorem geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 1) 
  (h2 : (3 * S 1, 2 * S 2, S 3) = (3 * S 1, 2 * S 2, S 3) ∧ (4 * S 2 = 3 * S 1 + S 3)) 
  (hq_pos : q ≠ 0) 
  (hq : ∀ n, a (n + 1) = a n * q):
  ∀ n, a n = 3^(n-1) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_l359_35966


namespace NUMINAMATH_GPT_ratio_men_to_women_on_team_l359_35981

theorem ratio_men_to_women_on_team (M W : ℕ) 
  (h1 : W = M + 6) 
  (h2 : M + W = 24) : 
  M / W = 3 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_men_to_women_on_team_l359_35981


namespace NUMINAMATH_GPT_triangle_median_length_l359_35921

variable (XY XZ XM YZ : ℝ)

theorem triangle_median_length :
  XY = 6 →
  XZ = 8 →
  XM = 5 →
  YZ = 10 := by
  sorry

end NUMINAMATH_GPT_triangle_median_length_l359_35921


namespace NUMINAMATH_GPT_fruits_total_l359_35972

def remaining_fruits (frank_apples susan_blueberries henry_apples karen_grapes : ℤ) : ℤ :=
  let frank_remaining := 36 - (36 / 3)
  let susan_remaining := 120 - (120 / 2)
  let henry_collected := 2 * 120
  let henry_after_eating := henry_collected - (henry_collected / 4)
  let henry_remaining := henry_after_eating - (henry_after_eating / 10)
  let karen_collected := henry_collected / 2
  let karen_after_spoilage := karen_collected - (15 * karen_collected / 100)
  let karen_after_giving_away := karen_after_spoilage - (karen_after_spoilage / 3)
  let karen_remaining := karen_after_giving_away - (Int.sqrt karen_after_giving_away)
  frank_remaining + susan_remaining + henry_remaining + karen_remaining

theorem fruits_total : remaining_fruits 36 120 240 120 = 254 :=
by sorry

end NUMINAMATH_GPT_fruits_total_l359_35972


namespace NUMINAMATH_GPT_circle_center_tangent_lines_l359_35932

theorem circle_center_tangent_lines 
    (center : ℝ × ℝ)
    (h1 : 3 * center.1 + 4 * center.2 = 10)
    (h2 : center.1 = 3 * center.2) : 
    center = (30 / 13, 10 / 13) := 
by {
  sorry
}

end NUMINAMATH_GPT_circle_center_tangent_lines_l359_35932


namespace NUMINAMATH_GPT_correct_average_l359_35991

theorem correct_average 
  (n : ℕ) (initial_average : ℚ) (wrong_number : ℚ) (correct_number : ℚ) (wrong_average : ℚ)
  (h_n : n = 10) 
  (h_initial : initial_average = 14) 
  (h_wrong_number : wrong_number = 26) 
  (h_correct_number : correct_number = 36) 
  (h_wrong_average : wrong_average = 14) : 
  (initial_average * n - wrong_number + correct_number) / n = 15 := 
by
  sorry

end NUMINAMATH_GPT_correct_average_l359_35991


namespace NUMINAMATH_GPT_fraction_of_fractions_l359_35961

theorem fraction_of_fractions : (1 / 8) / (1 / 3) = 3 / 8 := by
  sorry

end NUMINAMATH_GPT_fraction_of_fractions_l359_35961


namespace NUMINAMATH_GPT_analogical_reasoning_correct_l359_35907

theorem analogical_reasoning_correct (a b c : ℝ) (hc : c ≠ 0) : (a + b) * c = a * c + b * c → (a + b) / c = a / c + b / c :=
by
  sorry

end NUMINAMATH_GPT_analogical_reasoning_correct_l359_35907


namespace NUMINAMATH_GPT_question_solution_l359_35958

noncomputable def segment_ratio : (ℝ × ℝ) :=
  let m := 7
  let n := 2
  let x := - (2 / (m - n))
  let y := 7 / (m - n)
  (x, y)

theorem question_solution : segment_ratio = (-2/5, 7/5) :=
  by
  -- prove that the pair (x, y) calculated using given m and n equals (-2/5, 7/5)
  sorry

end NUMINAMATH_GPT_question_solution_l359_35958


namespace NUMINAMATH_GPT_bianca_deleted_text_files_l359_35988

theorem bianca_deleted_text_files (pictures songs total : ℕ) (h₁ : pictures = 2) (h₂ : songs = 8) (h₃ : total = 17) :
  total - (pictures + songs) = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_bianca_deleted_text_files_l359_35988


namespace NUMINAMATH_GPT_expr_D_is_diff_of_squares_l359_35941

-- Definitions for the expressions
def expr_A (a b : ℤ) : ℤ := (a + 2 * b) * (-a - 2 * b)
def expr_B (m n : ℤ) : ℤ := (2 * m - 3 * n) * (3 * n - 2 * m)
def expr_C (x y : ℤ) : ℤ := (2 * x - 3 * y) * (3 * x + 2 * y)
def expr_D (a b : ℤ) : ℤ := (a - b) * (-b - a)

-- Theorem stating that Expression D can be calculated using the difference of squares formula
theorem expr_D_is_diff_of_squares (a b : ℤ) : expr_D a b = a^2 - b^2 :=
by sorry

end NUMINAMATH_GPT_expr_D_is_diff_of_squares_l359_35941


namespace NUMINAMATH_GPT_remaining_paint_fraction_l359_35910

theorem remaining_paint_fraction (x : ℝ) (h : 1.2 * x = 1 / 2) : (1 / 2) - x = 1 / 12 :=
by 
  sorry

end NUMINAMATH_GPT_remaining_paint_fraction_l359_35910


namespace NUMINAMATH_GPT_frisbee_sales_total_receipts_l359_35940

theorem frisbee_sales_total_receipts 
  (total_frisbees : ℕ) 
  (price_3_frisbee : ℕ) 
  (price_4_frisbee : ℕ) 
  (sold_3 : ℕ) 
  (sold_4 : ℕ) 
  (total_receipts : ℕ) 
  (h1 : total_frisbees = 60) 
  (h2 : price_3_frisbee = 3)
  (h3 : price_4_frisbee = 4) 
  (h4 : sold_3 + sold_4 = total_frisbees) 
  (h5 : sold_4 ≥ 24)
  (h6 : total_receipts = sold_3 * price_3_frisbee + sold_4 * price_4_frisbee) :
  total_receipts = 204 :=
sorry

end NUMINAMATH_GPT_frisbee_sales_total_receipts_l359_35940


namespace NUMINAMATH_GPT_find_n_divisible_by_35_l359_35950

-- Define the five-digit number for some digit n
def num (n : ℕ) : ℕ := 80000 + n * 1000 + 975

-- Define the conditions
def divisible_by_5 (d : ℕ) : Prop := d % 5 = 0
def divisible_by_7 (d : ℕ) : Prop := d % 7 = 0
def divisible_by_35 (d : ℕ) : Prop := divisible_by_5 d ∧ divisible_by_7 d

-- Statement of the problem for proving given conditions and the correct answer
theorem find_n_divisible_by_35 : ∃ (n : ℕ), (num n % 35 = 0) ∧ n = 6 := by
  sorry

end NUMINAMATH_GPT_find_n_divisible_by_35_l359_35950


namespace NUMINAMATH_GPT_madeline_has_five_boxes_l359_35947

theorem madeline_has_five_boxes 
    (total_crayons_per_box : ℕ)
    (not_used_fraction1 : ℚ)
    (not_used_fraction2 : ℚ)
    (used_fraction2 : ℚ)
    (total_boxes_not_used : ℚ)
    (total_unused_crayons : ℕ)
    (unused_in_last_box : ℚ)
    (total_boxes : ℕ) :
    total_crayons_per_box = 24 →
    not_used_fraction1 = 5 / 8 →
    not_used_fraction2 = 1 / 3 →
    used_fraction2 = 2 / 3 →
    total_boxes_not_used = 4 →
    total_unused_crayons = 70 →
    total_boxes = 5 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_madeline_has_five_boxes_l359_35947


namespace NUMINAMATH_GPT_exists_sum_of_divisibles_l359_35956

theorem exists_sum_of_divisibles : ∃ (a b: ℕ), a + b = 316 ∧ (13 ∣ a) ∧ (11 ∣ b) :=
by
  existsi 52
  existsi 264
  sorry

end NUMINAMATH_GPT_exists_sum_of_divisibles_l359_35956


namespace NUMINAMATH_GPT_number_of_subsets_l359_35993

theorem number_of_subsets (M : Finset ℕ) (h : M.card = 5) : 2 ^ M.card = 32 := by
  sorry

end NUMINAMATH_GPT_number_of_subsets_l359_35993


namespace NUMINAMATH_GPT_divide_segment_l359_35912

theorem divide_segment (a : ℝ) (n : ℕ) (h : 0 < n) : 
  ∃ P : ℝ, P = a / (n + 1) ∧ P > 0 :=
by
  sorry

end NUMINAMATH_GPT_divide_segment_l359_35912


namespace NUMINAMATH_GPT_union_complements_eq_l359_35997

-- Definitions for the universal set U and subsets A and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5, 7}
def B : Set ℕ := {3, 4, 5}

-- Definition of the complements of A and B with respect to U
def complement_U_A : Set ℕ := {x ∈ U | x ∉ A}
def complement_U_B : Set ℕ := {x ∈ U | x ∉ B}

-- The union of the two complements
def union_complements : Set ℕ := complement_U_A ∪ complement_U_B

-- The target proof statement
theorem union_complements_eq : union_complements = {1, 2, 3, 6, 7} := by
  sorry

end NUMINAMATH_GPT_union_complements_eq_l359_35997


namespace NUMINAMATH_GPT_specific_values_exist_l359_35925

def expr_equal_for_specific_values (a b c : ℝ) : Prop :=
  a + b^2 * c = (a^2 + b) * (a + c)

theorem specific_values_exist :
  ∃ a b c : ℝ, expr_equal_for_specific_values a b c :=
sorry

end NUMINAMATH_GPT_specific_values_exist_l359_35925


namespace NUMINAMATH_GPT_percentage_of_students_with_same_grades_l359_35998

noncomputable def same_grade_percentage (students_class : ℕ) (grades_A : ℕ) (grades_B : ℕ) (grades_C : ℕ) (grades_D : ℕ) (grades_E : ℕ) : ℚ :=
  ((grades_A + grades_B + grades_C + grades_D + grades_E : ℚ) / students_class) * 100

theorem percentage_of_students_with_same_grades :
  let students_class := 40
  let grades_A := 3
  let grades_B := 5
  let grades_C := 6
  let grades_D := 2
  let grades_E := 1
  same_grade_percentage students_class grades_A grades_B grades_C grades_D grades_E = 42.5 := by
  sorry

end NUMINAMATH_GPT_percentage_of_students_with_same_grades_l359_35998


namespace NUMINAMATH_GPT_intersection_eq_l359_35915

def M := {x : ℝ | x < 1}
def N := {x : ℝ | Real.log x / Real.log 2 < 1}

theorem intersection_eq : {x : ℝ | x ∈ M ∧ x ∈ N} = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_eq_l359_35915


namespace NUMINAMATH_GPT_min_value_of_squares_attains_min_value_l359_35926

theorem min_value_of_squares (a b c t : ℝ) (h : a + b + c = t) :
  (a^2 + b^2 + c^2) ≥ (t^2 / 3) :=
sorry

theorem attains_min_value (a b c t : ℝ) (h : a = t / 3 ∧ b = t / 3 ∧ c = t / 3) :
  (a^2 + b^2 + c^2) = (t^2 / 3) :=
sorry

end NUMINAMATH_GPT_min_value_of_squares_attains_min_value_l359_35926


namespace NUMINAMATH_GPT_leak_time_to_empty_tank_l359_35965

theorem leak_time_to_empty_tank :
  let rateA := 1 / 2  -- rate at which pipe A fills the tank (tanks per hour)
  let rateB := 2 / 3  -- rate at which pipe B fills the tank (tanks per hour)
  let combined_rate_without_leak := rateA + rateB  -- combined rate without leak
  let combined_rate_with_leak := 1 / 1.75  -- combined rate with leak (tanks per hour)
  let leak_rate := combined_rate_without_leak - combined_rate_with_leak  -- rate of the leak (tanks per hour)
  60 / leak_rate = 100.8 :=  -- time to empty the tank by the leak (minutes)
    by sorry

end NUMINAMATH_GPT_leak_time_to_empty_tank_l359_35965


namespace NUMINAMATH_GPT_alyssa_ate_limes_l359_35906

def mikes_limes : ℝ := 32.0
def limes_left : ℝ := 7.0

theorem alyssa_ate_limes : mikes_limes - limes_left = 25.0 := by
  sorry

end NUMINAMATH_GPT_alyssa_ate_limes_l359_35906


namespace NUMINAMATH_GPT_min_students_l359_35911

theorem min_students (b g : ℕ) (hb : (3 / 5 : ℚ) * b = (5 / 6 : ℚ) * g) :
  b + g = 43 :=
sorry

end NUMINAMATH_GPT_min_students_l359_35911


namespace NUMINAMATH_GPT_number_of_square_tiles_l359_35939

/-- A box contains a collection of triangular tiles, square tiles, and pentagonal tiles. 
    There are a total of 30 tiles in the box and a total of 100 edges. 
    We need to show that the number of square tiles is 10. --/
theorem number_of_square_tiles (a b c : ℕ) (h1 : a + b + c = 30) (h2 : 3 * a + 4 * b + 5 * c = 100) : b = 10 := by
  sorry

end NUMINAMATH_GPT_number_of_square_tiles_l359_35939


namespace NUMINAMATH_GPT_simplify_and_evaluate_l359_35994

theorem simplify_and_evaluate 
  (a b : ℤ)
  (h1 : a = 2)
  (h2 : b = -1) : 
  (2 * a^2 * b - 4 * a * b^2) - 2 * (a * b^2 + a^2 * b) = -12 := 
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l359_35994


namespace NUMINAMATH_GPT_population_doubling_time_l359_35995

open Real

noncomputable def net_growth_rate (birth_rate : ℝ) (death_rate : ℝ) : ℝ :=
birth_rate - death_rate

noncomputable def percentage_growth_rate (net_growth_rate : ℝ) (population_base : ℝ) : ℝ :=
(net_growth_rate / population_base) * 100

noncomputable def doubling_time (percentage_growth_rate : ℝ) : ℝ :=
70 / percentage_growth_rate

theorem population_doubling_time :
    let birth_rate := 39.4
    let death_rate := 19.4
    let population_base := 1000
    let net_growth := net_growth_rate birth_rate death_rate
    let percentage_growth := percentage_growth_rate net_growth population_base
    doubling_time percentage_growth = 35 := 
by
    sorry

end NUMINAMATH_GPT_population_doubling_time_l359_35995


namespace NUMINAMATH_GPT_distribute_pencils_l359_35938

theorem distribute_pencils (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 4) :
  (∃ ways : ℕ, ways = Nat.choose (n - 1) (k - 1) ∧ ways = 35) :=
by
  sorry

end NUMINAMATH_GPT_distribute_pencils_l359_35938


namespace NUMINAMATH_GPT_matrix_C_power_50_l359_35954

open Matrix

theorem matrix_C_power_50 (C : Matrix (Fin 2) (Fin 2) ℤ) 
  (hC : C = !![3, 2; -8, -5]) : 
  C^50 = !![1, 0; 0, 1] :=
by {
  -- External proof omitted.
  sorry
}

end NUMINAMATH_GPT_matrix_C_power_50_l359_35954


namespace NUMINAMATH_GPT_horses_legs_problem_l359_35904

theorem horses_legs_problem 
    (m h a b : ℕ) 
    (h_eq : h = m) 
    (men_to_A : m = 3 * a) 
    (men_to_B : m = 4 * b) 
    (total_legs : 2 * m + 4 * (h / 2) + 3 * a + 4 * b = 200) : 
    h = 25 :=
  sorry

end NUMINAMATH_GPT_horses_legs_problem_l359_35904


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_with_product_812_l359_35979

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_with_product_812_l359_35979


namespace NUMINAMATH_GPT_largest_common_value_l359_35973

theorem largest_common_value :
  ∃ (a : ℕ), (∃ (n m : ℕ), a = 4 + 5 * n ∧ a = 5 + 10 * m) ∧ a < 1000 ∧ a = 994 :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_common_value_l359_35973


namespace NUMINAMATH_GPT_minimum_value_inequality_equality_condition_exists_l359_35952

theorem minimum_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  6 * c / (3 * a + b) + 6 * a / (b + 3 * c) + 2 * b / (a + c) ≥ 12 := by
  sorry

theorem equality_condition_exists : 
  ∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ (6 * c / (3 * a + b) + 6 * a / (b + 3 * c) + 2 * b / (a + c) = 12) := by
  sorry

end NUMINAMATH_GPT_minimum_value_inequality_equality_condition_exists_l359_35952


namespace NUMINAMATH_GPT_tteokbokki_cost_l359_35927

theorem tteokbokki_cost (P : ℝ) (h1 : P / 2 - P * (3 / 16) = 2500) : P / 2 = 4000 :=
by
  sorry

end NUMINAMATH_GPT_tteokbokki_cost_l359_35927


namespace NUMINAMATH_GPT_problem_M_l359_35957

theorem problem_M (M : ℤ) (h : 1989 + 1991 + 1993 + 1995 + 1997 + 1999 + 2001 = 14000 - M) : M = 35 :=
by
  sorry

end NUMINAMATH_GPT_problem_M_l359_35957


namespace NUMINAMATH_GPT_min_dist_sum_l359_35920

theorem min_dist_sum (x y : ℝ) :
  let M := (1, 3)
  let N := (7, 5)
  let P_on_M := (x - 1)^2 + (y - 3)^2 = 1
  let Q_on_N := (x - 7)^2 + (y - 5)^2 = 4
  let A_on_x_axis := y = 0
  ∃ (P Q : ℝ × ℝ), P_on_M ∧ Q_on_N ∧ ∀ A : ℝ × ℝ, A_on_x_axis → (|dist A P| + |dist A Q|) = 7 := 
sorry

end NUMINAMATH_GPT_min_dist_sum_l359_35920


namespace NUMINAMATH_GPT_bucky_savings_excess_l359_35986

def cost_of_game := 60
def saved_amount := 15
def fish_earnings_weekends (fish : String) : ℕ :=
  match fish with
  | "trout" => 5
  | "bluegill" => 4
  | "bass" => 7
  | "catfish" => 6
  | _ => 0

def fish_earnings_weekdays (fish : String) : ℕ :=
  match fish with
  | "trout" => 10
  | "bluegill" => 8
  | "bass" => 14
  | "catfish" => 12
  | _ => 0

def sunday_fish := 10
def weekday_fish := 3
def weekdays := 2

def sunday_fish_distribution := [
  ("trout", 3),
  ("bluegill", 2),
  ("bass", 4),
  ("catfish", 1)
]

noncomputable def sunday_earnings : ℕ :=
  sunday_fish_distribution.foldl (λ acc (fish, count) =>
    acc + count * fish_earnings_weekends fish) 0

noncomputable def weekday_earnings : ℕ :=
  weekdays * weekday_fish * (
    fish_earnings_weekdays "trout" +
    fish_earnings_weekdays "bluegill" +
    fish_earnings_weekdays "bass")

noncomputable def total_earnings : ℕ :=
  sunday_earnings + weekday_earnings

noncomputable def total_savings : ℕ :=
  total_earnings + saved_amount

theorem bucky_savings_excess :
  total_savings - cost_of_game = 76 :=
by sorry

end NUMINAMATH_GPT_bucky_savings_excess_l359_35986
