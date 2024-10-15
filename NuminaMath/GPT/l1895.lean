import Mathlib

namespace NUMINAMATH_GPT_solve_system_eq_l1895_189570

theorem solve_system_eq (x y : ℝ) :
  x^2 + y^2 + 6 * x * y = 68 ∧ 2 * x^2 + 2 * y^2 - 3 * x * y = 16 ↔
  (x = 4 ∧ y = 2) ∨ (x = 2 ∧ y = 4) ∨ (x = -4 ∧ y = -2) ∨ (x = -2 ∧ y = -4) := 
by
  sorry

end NUMINAMATH_GPT_solve_system_eq_l1895_189570


namespace NUMINAMATH_GPT_fraction_simplification_l1895_189505

theorem fraction_simplification (a b : ℝ) : 9 * b / (6 * a + 3) = 3 * b / (2 * a + 1) :=
by sorry

end NUMINAMATH_GPT_fraction_simplification_l1895_189505


namespace NUMINAMATH_GPT_prob_each_student_gets_each_snack_l1895_189558

-- Define the total number of snacks and their types
def total_snacks := 16
def snack_types := 4

-- Define the conditions for the problem
def students := 4
def snacks_per_type := 4

-- Define the probability calculation.
-- We would typically use combinatorial functions here, but for simplicity, use predefined values from the solution.
def prob_student_1 := 64 / 455
def prob_student_2 := 9 / 55
def prob_student_3 := 8 / 35
def prob_student_4 := 1 -- Always 1 for the final student's remaining snacks

-- Calculate the total probability
def total_prob := prob_student_1 * prob_student_2 * prob_student_3 * prob_student_4

-- The statement to prove the desired probability outcome
theorem prob_each_student_gets_each_snack : total_prob = (64 / 1225) :=
by
  sorry

end NUMINAMATH_GPT_prob_each_student_gets_each_snack_l1895_189558


namespace NUMINAMATH_GPT_min_value_of_m_l1895_189557

theorem min_value_of_m (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a * b + b * c + c * a = -1) (h3 : a * b * c = -m) : 
    m = - (min (-a ^ 3 + a ^ 2 + a ) (- (1 / 27))) := 
sorry

end NUMINAMATH_GPT_min_value_of_m_l1895_189557


namespace NUMINAMATH_GPT_value_after_increase_l1895_189503

-- Definition of original number and percentage increase
def original_number : ℝ := 600
def percentage_increase : ℝ := 0.10

-- Theorem stating that after a 10% increase, the value is 660
theorem value_after_increase : original_number * (1 + percentage_increase) = 660 := by
  sorry

end NUMINAMATH_GPT_value_after_increase_l1895_189503


namespace NUMINAMATH_GPT_find_all_x_satisfying_condition_l1895_189544

theorem find_all_x_satisfying_condition :
  ∃ (x : Fin 2016 → ℝ), 
  (∀ i : Fin 2016, x (i + 1) % 2016 = x 0) ∧
  (∀ i : Fin 2016, x i ^ 2 + x i - 1 = x ((i + 1) % 2016)) ∧
  (∀ i : Fin 2016, x i = 1 ∨ x i = -1) :=
sorry

end NUMINAMATH_GPT_find_all_x_satisfying_condition_l1895_189544


namespace NUMINAMATH_GPT_large_rectangle_perimeter_correct_l1895_189525

def perimeter_of_square (p : ℕ) : ℕ :=
  p / 4

def perimeter_of_rectangle (p : ℕ) (l : ℕ) : ℕ :=
  (p - 2 * l) / 2

def perimeter_of_large_rectangle (side_length_of_square side_length_of_rectangle : ℕ) : ℕ :=
  let height := side_length_of_square + 2 * side_length_of_rectangle
  let width := 3 * side_length_of_square
  2 * (height + width)

theorem large_rectangle_perimeter_correct :
  let side_length_of_square := perimeter_of_square 24
  let side_length_of_rectangle := perimeter_of_rectangle 16 side_length_of_square
  perimeter_of_large_rectangle side_length_of_square side_length_of_rectangle = 52 :=
by
  sorry

end NUMINAMATH_GPT_large_rectangle_perimeter_correct_l1895_189525


namespace NUMINAMATH_GPT_total_legs_of_camden_dogs_l1895_189524

-- Defining the number of dogs Justin has
def justin_dogs : ℕ := 14

-- Defining the number of dogs Rico has
def rico_dogs : ℕ := justin_dogs + 10

-- Defining the number of dogs Camden has
def camden_dogs : ℕ := 3 * rico_dogs / 4

-- Defining the total number of legs Camden's dogs have
def camden_dogs_legs : ℕ := camden_dogs * 4

-- The proof statement
theorem total_legs_of_camden_dogs : camden_dogs_legs = 72 :=
by
  -- skip proof
  sorry

end NUMINAMATH_GPT_total_legs_of_camden_dogs_l1895_189524


namespace NUMINAMATH_GPT_age_difference_l1895_189534

theorem age_difference (h b m : ℕ) (ratio : h = 4 * m ∧ b = 3 * m ∧ 4 * m + 3 * m + 7 * m = 126) : h - b = 9 :=
by
  -- proof will be filled here
  sorry

end NUMINAMATH_GPT_age_difference_l1895_189534


namespace NUMINAMATH_GPT_hypotenuse_length_l1895_189516

theorem hypotenuse_length (a b c : ℝ)
  (h_a : a = 12)
  (h_area : 54 = 1 / 2 * a * b)
  (h_py : c^2 = a^2 + b^2) :
    c = 15 := by
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l1895_189516


namespace NUMINAMATH_GPT_problem_equiv_l1895_189509

def dollar (a b : ℝ) : ℝ := (a - b) ^ 2

theorem problem_equiv (x y : ℝ) : dollar ((2 * x + y) ^ 2) ((x - 2 * y) ^ 2) = (3 * x ^ 2 + 8 * x * y - 3 * y ^ 2) ^ 2 := by
  sorry

end NUMINAMATH_GPT_problem_equiv_l1895_189509


namespace NUMINAMATH_GPT_tan_identity_l1895_189517

open Real

-- Definition of conditions
def isPureImaginary (z : Complex) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

theorem tan_identity (theta : ℝ) :
  isPureImaginary ((cos theta - 4/5) + (sin theta - 3/5) * Complex.I) →
  tan (theta - π / 4) = -7 :=
by
  sorry

end NUMINAMATH_GPT_tan_identity_l1895_189517


namespace NUMINAMATH_GPT_sqrt27_add_sqrt3_sub_sqrt12_eq_2sqrt3_sqrt24_inverse_add_abs_sqrt6_sub_3_add_half_inverse_sub_2016_pow0_eq_4_min_11sqrt6_div_12_sqrt3_add_sqrt2_squared_sub_sqrt3_sub_sqrt2_squared_eq_4sqrt6_l1895_189593

-- Proof 1: 
theorem sqrt27_add_sqrt3_sub_sqrt12_eq_2sqrt3 :
  Real.sqrt 27 + Real.sqrt 3 - Real.sqrt 12 = 2 * Real.sqrt 3 :=
by
  sorry

-- Proof 2:
theorem sqrt24_inverse_add_abs_sqrt6_sub_3_add_half_inverse_sub_2016_pow0_eq_4_min_11sqrt6_div_12 :
  1 / Real.sqrt 24 + abs (Real.sqrt 6 - 3) + (1 / 2)⁻¹ - 2016 ^ 0 = 4 - 11 * Real.sqrt 6 / 12 :=
by
  sorry

-- Proof 3:
theorem sqrt3_add_sqrt2_squared_sub_sqrt3_sub_sqrt2_squared_eq_4sqrt6 :
  (Real.sqrt 3 + Real.sqrt 2) ^ 2 - (Real.sqrt 3 - Real.sqrt 2) ^ 2 = 4 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_sqrt27_add_sqrt3_sub_sqrt12_eq_2sqrt3_sqrt24_inverse_add_abs_sqrt6_sub_3_add_half_inverse_sub_2016_pow0_eq_4_min_11sqrt6_div_12_sqrt3_add_sqrt2_squared_sub_sqrt3_sub_sqrt2_squared_eq_4sqrt6_l1895_189593


namespace NUMINAMATH_GPT_n_calculation_l1895_189547

theorem n_calculation (n : ℕ) (hn : 0 < n)
  (h1 : Int.lcm 24 n = 72)
  (h2 : Int.lcm n 27 = 108) :
  n = 36 :=
sorry

end NUMINAMATH_GPT_n_calculation_l1895_189547


namespace NUMINAMATH_GPT_tom_new_collection_l1895_189553

theorem tom_new_collection (initial_stamps mike_gift : ℕ) (harry_gift : ℕ := 2 * mike_gift + 10) (sarah_gift : ℕ := 3 * mike_gift - 5) (total_gifts : ℕ := mike_gift + harry_gift + sarah_gift) (new_collection : ℕ := initial_stamps + total_gifts) 
  (h_initial_stamps : initial_stamps = 3000) (h_mike_gift : mike_gift = 17) :
  new_collection = 3107 := by
  sorry

end NUMINAMATH_GPT_tom_new_collection_l1895_189553


namespace NUMINAMATH_GPT_smallest_number_starts_with_four_and_decreases_four_times_l1895_189565

theorem smallest_number_starts_with_four_and_decreases_four_times :
  ∃ (X : ℕ), ∃ (A n : ℕ), (X = 4 * 10^n + A ∧ X = 4 * (10 * A + 4)) ∧ X = 410256 := 
by
  sorry

end NUMINAMATH_GPT_smallest_number_starts_with_four_and_decreases_four_times_l1895_189565


namespace NUMINAMATH_GPT_no_extrema_1_1_l1895_189579

noncomputable def f (x : ℝ) : ℝ :=
  x^3 - 3 * x

theorem no_extrema_1_1 : ∀ x : ℝ, (x > -1) ∧ (x < 1) → ¬ (∃ c : ℝ, c ∈ Set.Ioo (-1) (1) ∧ (∀ y ∈ Set.Ioo (-1) (1), f y ≤ f c ∨ f c ≤ f y)) :=
by
  sorry

end NUMINAMATH_GPT_no_extrema_1_1_l1895_189579


namespace NUMINAMATH_GPT_karen_average_speed_l1895_189507

noncomputable def total_distance : ℚ := 198
noncomputable def start_time : ℚ := (9 * 60 + 40) / 60
noncomputable def end_time : ℚ := (13 * 60 + 20) / 60
noncomputable def total_time : ℚ := end_time - start_time
noncomputable def average_speed (distance : ℚ) (time : ℚ) : ℚ := distance / time

theorem karen_average_speed :
  average_speed total_distance total_time = 54 := by
  sorry

end NUMINAMATH_GPT_karen_average_speed_l1895_189507


namespace NUMINAMATH_GPT_problem_solution_l1895_189568

open Set

variable {U : Set ℕ} (M : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hC : U \ M = {1, 3})

theorem problem_solution : 2 ∈ M :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1895_189568


namespace NUMINAMATH_GPT_students_taking_chem_or_phys_not_both_l1895_189556

def students_taking_both : ℕ := 12
def students_taking_chemistry : ℕ := 30
def students_taking_only_physics : ℕ := 18

theorem students_taking_chem_or_phys_not_both : 
  (students_taking_chemistry - students_taking_both) + students_taking_only_physics = 36 := 
by
  sorry

end NUMINAMATH_GPT_students_taking_chem_or_phys_not_both_l1895_189556


namespace NUMINAMATH_GPT_probability_of_rolling_greater_than_five_l1895_189539

def probability_of_greater_than_five (dice_faces : Finset ℕ) (greater_than : ℕ) : ℚ := 
  let favorable_outcomes := dice_faces.filter (λ x => x > greater_than)
  favorable_outcomes.card / dice_faces.card

theorem probability_of_rolling_greater_than_five:
  probability_of_greater_than_five ({1, 2, 3, 4, 5, 6} : Finset ℕ) 5 = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_rolling_greater_than_five_l1895_189539


namespace NUMINAMATH_GPT_new_container_volume_l1895_189518

-- Define the original volume of the container 
def original_volume : ℝ := 4

-- Define the scale factor of each dimension (quadrupled)
def scale_factor : ℝ := 4

-- Define the new volume, which is original volume * (scale factor ^ 3)
def new_volume (orig_vol : ℝ) (scale : ℝ) : ℝ := orig_vol * (scale ^ 3)

-- The theorem we want to prove
theorem new_container_volume : new_volume original_volume scale_factor = 256 :=
by
  sorry

end NUMINAMATH_GPT_new_container_volume_l1895_189518


namespace NUMINAMATH_GPT_angle_BAC_in_isosceles_triangle_l1895_189599

theorem angle_BAC_in_isosceles_triangle
  (A B C D : Type)
  (AB AC : ℝ)
  (BD DC : ℝ)
  (angle_BDA : ℝ)
  (isosceles_triangle : AB = AC)
  (midpoint_D : BD = DC)
  (external_angle_D : angle_BDA = 120) :
  ∃ (angle_BAC : ℝ), angle_BAC = 60 :=
by
  sorry

end NUMINAMATH_GPT_angle_BAC_in_isosceles_triangle_l1895_189599


namespace NUMINAMATH_GPT_range_of_b_l1895_189521

def M := {p : ℝ × ℝ | p.1 ^ 2 + 2 * p.2 ^ 2 = 3}
def N (m b : ℝ) := {p : ℝ × ℝ | p.2 = m * p.1 + b}

theorem range_of_b (b : ℝ) : (∀ (m : ℝ), (∃ (p : ℝ × ℝ), p ∈ M ∧ p ∈ N m b)) ↔ 
  -Real.sqrt (6) / 2 ≤ b ∧ b ≤ Real.sqrt (6) / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_b_l1895_189521


namespace NUMINAMATH_GPT_arithmetic_seq_sum_l1895_189514

theorem arithmetic_seq_sum(S : ℕ → ℝ) (d : ℝ) (h1 : S 5 < S 6) 
    (h2 : S 6 = S 7) (h3 : S 7 > S 8) : S 9 < S 5 := 
sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l1895_189514


namespace NUMINAMATH_GPT_det_A_is_neg9_l1895_189540

noncomputable def A : Matrix (Fin 2) (Fin 2) ℤ := ![![-7, 5], ![6, -3]]

theorem det_A_is_neg9 : Matrix.det A = -9 := 
by 
  sorry

end NUMINAMATH_GPT_det_A_is_neg9_l1895_189540


namespace NUMINAMATH_GPT_jackson_money_l1895_189501

theorem jackson_money (W : ℝ) (H1 : 5 * W + W = 150) : 5 * W = 125 :=
by
  sorry

end NUMINAMATH_GPT_jackson_money_l1895_189501


namespace NUMINAMATH_GPT_tan_three_theta_l1895_189506

theorem tan_three_theta (θ : Real) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end NUMINAMATH_GPT_tan_three_theta_l1895_189506


namespace NUMINAMATH_GPT_typist_current_salary_l1895_189513

-- Define the initial conditions as given in the problem
def initial_salary : ℝ := 6000
def raise_percentage : ℝ := 0.10
def reduction_percentage : ℝ := 0.05

-- Define the calculations for raised and reduced salaries
def raised_salary := initial_salary * (1 + raise_percentage)
def current_salary := raised_salary * (1 - reduction_percentage)

-- State the theorem to prove the current salary
theorem typist_current_salary : current_salary = 6270 := 
by
  -- Sorry is used to skip proof, overriding with the statement to ensure code builds successfully
  sorry

end NUMINAMATH_GPT_typist_current_salary_l1895_189513


namespace NUMINAMATH_GPT_fish_population_estimate_l1895_189559

theorem fish_population_estimate :
  ∃ N : ℕ, (60 * 60) / 2 = N ∧ (2 / 60 : ℚ) = (60 / N : ℚ) :=
by
  use 1800
  simp
  sorry

end NUMINAMATH_GPT_fish_population_estimate_l1895_189559


namespace NUMINAMATH_GPT_base_k_to_decimal_is_5_l1895_189587

theorem base_k_to_decimal_is_5 (k : ℕ) (h : 1 * k^2 + 3 * k + 2 = 42) : k = 5 := sorry

end NUMINAMATH_GPT_base_k_to_decimal_is_5_l1895_189587


namespace NUMINAMATH_GPT_find_f_of_2_l1895_189598

theorem find_f_of_2 (f : ℝ → ℝ) (h : ∀ (x : ℝ), x > 0 → f (Real.log x / Real.log 2) = 2 ^ x) : f 2 = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_f_of_2_l1895_189598


namespace NUMINAMATH_GPT_Margie_can_drive_200_miles_l1895_189576

/--
  Margie's car can go 40 miles per gallon of gas, and the price of gas is $5 per gallon.
  Prove that Margie can drive 200 miles with $25 worth of gas.
-/
theorem Margie_can_drive_200_miles (miles_per_gallon price_per_gallon money_available : ℕ) 
  (h1 : miles_per_gallon = 40) (h2 : price_per_gallon = 5) (h3 : money_available = 25) : 
  (money_available / price_per_gallon) * miles_per_gallon = 200 :=
by 
  /- The proof goes here -/
  sorry

end NUMINAMATH_GPT_Margie_can_drive_200_miles_l1895_189576


namespace NUMINAMATH_GPT_problem_quadrilateral_inscribed_in_circle_l1895_189510

theorem problem_quadrilateral_inscribed_in_circle
  (r : ℝ)
  (AB BC CD DA : ℝ)
  (h_radius : r = 300 * Real.sqrt 2)
  (h_AB : AB = 300)
  (h_BC : BC = 150)
  (h_CD : CD = 150) :
  DA = 750 :=
sorry

end NUMINAMATH_GPT_problem_quadrilateral_inscribed_in_circle_l1895_189510


namespace NUMINAMATH_GPT_gcd_of_three_numbers_l1895_189563

theorem gcd_of_three_numbers : Nat.gcd (Nat.gcd 324 243) 135 = 27 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_of_three_numbers_l1895_189563


namespace NUMINAMATH_GPT_parabola_tangent_angle_l1895_189520

noncomputable def tangent_slope_angle : Real :=
  let x := (1 / 2 : ℝ)
  let y := x^2
  let slope := (deriv (fun x => x^2)) x
  Real.arctan slope

theorem parabola_tangent_angle :
  tangent_slope_angle = Real.pi / 4 :=
by
sorry

end NUMINAMATH_GPT_parabola_tangent_angle_l1895_189520


namespace NUMINAMATH_GPT_intersection_M_S_l1895_189529

namespace ProofProblem

def M : Set ℕ := { x | 0 < x ∧ x < 4 }

def S : Set ℕ := { 2, 3, 5 }

theorem intersection_M_S :
  M ∩ S = { 2, 3 } := by
  sorry

end ProofProblem

end NUMINAMATH_GPT_intersection_M_S_l1895_189529


namespace NUMINAMATH_GPT_equal_split_l1895_189561

theorem equal_split (A B C : ℝ) (h1 : A < B) (h2 : B < C) : 
  (B + C - 2 * A) / 3 = (A + B + C) / 3 - A :=
by
  sorry

end NUMINAMATH_GPT_equal_split_l1895_189561


namespace NUMINAMATH_GPT_value_of_f_2014_l1895_189519

def f : ℕ → ℕ := sorry

theorem value_of_f_2014 : (∀ n : ℕ, f (f n) + f n = 2 * n + 3) → (f 0 = 1) → (f 2014 = 2015) := by
  intro h₁ h₀
  have h₂ := h₀
  sorry

end NUMINAMATH_GPT_value_of_f_2014_l1895_189519


namespace NUMINAMATH_GPT_will_earnings_l1895_189543

-- Defining the conditions
def hourly_wage : ℕ := 8
def monday_hours : ℕ := 8
def tuesday_hours : ℕ := 2

-- Calculating the earnings
def monday_earnings := monday_hours * hourly_wage
def tuesday_earnings := tuesday_hours * hourly_wage
def total_earnings := monday_earnings + tuesday_earnings

-- Stating the problem
theorem will_earnings : total_earnings = 80 := by
  -- sorry is used to skip the actual proof
  sorry

end NUMINAMATH_GPT_will_earnings_l1895_189543


namespace NUMINAMATH_GPT_remy_gallons_used_l1895_189574

def roman_usage (R : ℕ) : Prop := R + (3 * R + 1) = 33

def remy_usage (R : ℕ) (Remy : ℕ) : Prop := Remy = 3 * R + 1

theorem remy_gallons_used :
  ∃ R Remy : ℕ, roman_usage R ∧ remy_usage R Remy ∧ Remy = 25 :=
  by
    sorry

end NUMINAMATH_GPT_remy_gallons_used_l1895_189574


namespace NUMINAMATH_GPT_sum_series_eq_l1895_189528

theorem sum_series_eq : 
  (∑' k : ℕ, (k + 1) * (1/4)^(k + 1)) = 4 / 9 :=
by sorry

end NUMINAMATH_GPT_sum_series_eq_l1895_189528


namespace NUMINAMATH_GPT_eldorado_license_plates_count_l1895_189546

def is_vowel (c : Char) : Prop := c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U'

def valid_license_plates_count : Nat :=
  let num_vowels := 5
  let num_letters := 26
  let num_digits := 10
  num_vowels * num_letters * num_letters * num_digits * num_digits

theorem eldorado_license_plates_count : valid_license_plates_count = 338000 := by
  sorry

end NUMINAMATH_GPT_eldorado_license_plates_count_l1895_189546


namespace NUMINAMATH_GPT_troy_buys_beef_l1895_189536

theorem troy_buys_beef (B : ℕ) 
  (veg_pounds : ℕ := 6)
  (veg_cost_per_pound : ℕ := 2)
  (beef_cost_per_pound : ℕ := 3 * veg_cost_per_pound)
  (total_cost : ℕ := 36) :
  6 * veg_cost_per_pound + B * beef_cost_per_pound = total_cost → B = 4 :=
by
  sorry

end NUMINAMATH_GPT_troy_buys_beef_l1895_189536


namespace NUMINAMATH_GPT_sandy_siding_cost_l1895_189522

theorem sandy_siding_cost
  (wall_length wall_height roof_base roof_height : ℝ)
  (siding_length siding_height siding_cost : ℝ)
  (num_walls num_roof_faces num_siding_sections : ℝ)
  (total_cost : ℝ)
  (h_wall_length : wall_length = 10)
  (h_wall_height : wall_height = 7)
  (h_roof_base : roof_base = 10)
  (h_roof_height : roof_height = 6)
  (h_siding_length : siding_length = 10)
  (h_siding_height : siding_height = 15)
  (h_siding_cost : siding_cost = 35)
  (h_num_walls : num_walls = 2)
  (h_num_roof_faces : num_roof_faces = 1)
  (h_num_siding_sections : num_siding_sections = 2)
  (h_total_cost : total_cost = 70) :
  (siding_cost * num_siding_sections) = total_cost := 
by
  sorry

end NUMINAMATH_GPT_sandy_siding_cost_l1895_189522


namespace NUMINAMATH_GPT_greatest_prime_factor_of_n_l1895_189595

noncomputable def n : ℕ := 4^17 - 2^29

theorem greatest_prime_factor_of_n :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ n → q ≤ p :=
sorry

end NUMINAMATH_GPT_greatest_prime_factor_of_n_l1895_189595


namespace NUMINAMATH_GPT_ratio_w_to_y_l1895_189526

variables {w x y z : ℝ}

theorem ratio_w_to_y
  (h1 : w / x = 4 / 3)
  (h2 : y / z = 3 / 2)
  (h3 : z / x = 1 / 9) :
  w / y = 8 :=
by
  sorry

end NUMINAMATH_GPT_ratio_w_to_y_l1895_189526


namespace NUMINAMATH_GPT_labor_union_trees_l1895_189531

theorem labor_union_trees (x : ℕ) :
  (∃ t : ℕ, t = 2 * x + 21) ∧ (∃ t' : ℕ, t' = 3 * x - 24) →
  2 * x + 21 = 3 * x - 24 :=
by
  sorry

end NUMINAMATH_GPT_labor_union_trees_l1895_189531


namespace NUMINAMATH_GPT_fraction_field_planted_l1895_189533

-- Define the problem conditions
structure RightTriangle (leg1 leg2 hypotenuse : ℝ) : Prop :=
  (right_angle : ∃ (A B C : ℝ), A = 5 ∧ B = 12 ∧ hypotenuse = 13 ∧ A^2 + B^2 = hypotenuse^2)

structure SquarePatch (shortest_distance : ℝ) : Prop :=
  (distance_to_hypotenuse : shortest_distance = 3)

-- Define the statement
theorem fraction_field_planted (T : RightTriangle 5 12 13) (P : SquarePatch 3) : 
  ∃ (fraction : ℚ), fraction = 7 / 10 :=
by
  sorry

end NUMINAMATH_GPT_fraction_field_planted_l1895_189533


namespace NUMINAMATH_GPT_value_of_g_neg2_l1895_189527

def g (x : ℝ) : ℝ := x^3 - 2 * x + 1

theorem value_of_g_neg2 : g (-2) = -3 := 
by sorry

end NUMINAMATH_GPT_value_of_g_neg2_l1895_189527


namespace NUMINAMATH_GPT_missing_digit_is_0_l1895_189523

/- Define the known digits of the number. -/
def digit1 : ℕ := 6
def digit2 : ℕ := 5
def digit3 : ℕ := 3
def digit4 : ℕ := 4

/- Define the condition that ensures the divisibility by 9. -/
def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

/- The main theorem to prove: the value of the missing digit d is 0. -/
theorem missing_digit_is_0 (d : ℕ) 
  (h : is_divisible_by_9 (digit1 + digit2 + digit3 + digit4 + d)) : 
  d = 0 :=
sorry

end NUMINAMATH_GPT_missing_digit_is_0_l1895_189523


namespace NUMINAMATH_GPT_largest_lcm_among_given_pairs_l1895_189535

theorem largest_lcm_among_given_pairs : 
  max (max (max (max (max (Nat.lcm 18 3) (Nat.lcm 18 6)) (Nat.lcm 18 9)) (Nat.lcm 18 12)) (Nat.lcm 18 15)) (Nat.lcm 18 18) = 90 :=
by
  sorry

end NUMINAMATH_GPT_largest_lcm_among_given_pairs_l1895_189535


namespace NUMINAMATH_GPT_evaluate_sixth_iteration_of_g_at_2_l1895_189584

def g (x : ℤ) : ℤ := x^2 - 4 * x + 1

theorem evaluate_sixth_iteration_of_g_at_2 :
  g (g (g (g (g (g 2))))) = 59162302643740737293922 := by
  sorry

end NUMINAMATH_GPT_evaluate_sixth_iteration_of_g_at_2_l1895_189584


namespace NUMINAMATH_GPT_find_y_values_l1895_189578

def A (y : ℝ) : ℝ := 1 - y - 2 * y^2

theorem find_y_values (y : ℝ) (h₁ : y ≤ 1) (h₂ : y ≠ 0) (h₃ : y ≠ -1) (h₄ : y ≠ 0.5) :
  y^2 * A y / (y * A y) ≤ 1 ↔
  y ∈ Set.Iio (-1) ∪ Set.Ioo (-1) (1/2) ∪ Set.Ioc (1/2) 1 :=
by
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_find_y_values_l1895_189578


namespace NUMINAMATH_GPT_part1_max_min_part2_triangle_inequality_l1895_189582

noncomputable def f (x k : ℝ) : ℝ :=
  (x^4 + k*x^2 + 1) / (x^4 + x^2 + 1)

theorem part1_max_min (k : ℝ): 
  (∀ x : ℝ, k ≥ 1 → 1 ≤ f x k ∧ f x k ≤ (1/3) * (k + 2)) ∧ 
  (∀ x : ℝ, k < 1 → (1/3) * (k + 2) ≤ f x k ∧ f x k ≤ 1) := 
sorry

theorem part2_triangle_inequality (k : ℝ) : 
  -1/2 < k ∧ k < 4 ↔ (∀ a b c : ℝ, (f a k + f b k > f c k) ∧ (f b k + f c k > f a k) ∧ (f c k + f a k > f b k)) :=
sorry

end NUMINAMATH_GPT_part1_max_min_part2_triangle_inequality_l1895_189582


namespace NUMINAMATH_GPT_inequality1_in_triangle_inequality2_in_triangle_l1895_189502

theorem inequality1_in_triangle (a b c s : ℝ)
  (h1 : a + b + c = s) :
  (13 / 27) * s^2 ≤ a^2 + b^2 + c^2 + (4 / s) * a * b * c ∧ 
  a^2 + b^2 + c^2 + (4 / s) * a * b * c < s^2 / 2 :=
sorry

theorem inequality2_in_triangle (a b c s : ℝ)
  (h1 : a + b + c = s) :
  s^2 / 4 < a * b + b * c + c * a - (2 / s) * a * b * c ∧ 
  a * b + b * c + c * a - (2 / s) * a * b * c ≤ (7 / 27) * s^2 :=
sorry

end NUMINAMATH_GPT_inequality1_in_triangle_inequality2_in_triangle_l1895_189502


namespace NUMINAMATH_GPT_y_give_z_start_l1895_189532

variables (Vx Vy Vz T : ℝ)
variables (D : ℝ)

-- Conditions
def condition1 : Prop := Vx * T = Vy * T + 100
def condition2 : Prop := Vx * T = Vz * T + 200
def condition3 : Prop := T > 0

theorem y_give_z_start (h1 : condition1 Vx Vy T) (h2 : condition2 Vx Vz T) (h3 : condition3 T) : (Vy - Vz) * T = 200 := 
by
  sorry

end NUMINAMATH_GPT_y_give_z_start_l1895_189532


namespace NUMINAMATH_GPT_walter_bus_time_l1895_189504

noncomputable def walter_schedule : Prop :=
  let wake_up_time := 6  -- Walter gets up at 6:00 a.m.
  let leave_home_time := 7  -- Walter catches the school bus at 7:00 a.m.
  let arrival_home_time := 17  -- Walter arrives home at 5:00 p.m.
  let num_classes := 8  -- Walter has 8 classes
  let class_duration := 45  -- Each class lasts 45 minutes
  let lunch_duration := 40  -- Walter has 40 minutes for lunch
  let additional_activities_hours := 2.5  -- Walter has 2.5 hours of additional activities

  -- Total time calculation
  let total_away_hours := arrival_home_time - leave_home_time
  let total_away_minutes := total_away_hours * 60

  -- School-related activities calculation
  let total_class_minutes := num_classes * class_duration
  let total_additional_activities_minutes := additional_activities_hours * 60
  let total_school_activity_minutes := total_class_minutes + lunch_duration + total_additional_activities_minutes

  -- Time spent on the bus
  let bus_time := total_away_minutes - total_school_activity_minutes
  bus_time = 50

-- Statement to prove
theorem walter_bus_time : walter_schedule :=
  sorry

end NUMINAMATH_GPT_walter_bus_time_l1895_189504


namespace NUMINAMATH_GPT_total_students_in_class_l1895_189571

theorem total_students_in_class :
  ∃ x, (10 * 90 + 15 * 80 + x * 60) / (10 + 15 + x) = 72 → 10 + 15 + x = 50 :=
by
  -- Providing an existence proof and required conditions
  use 25
  intro h
  sorry

end NUMINAMATH_GPT_total_students_in_class_l1895_189571


namespace NUMINAMATH_GPT_weight_of_5_moles_H₂CO₃_l1895_189508

-- Definitions based on the given conditions
def atomic_weight_H : ℝ := 1.008
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999

def num_H₂CO₃_H : ℕ := 2
def num_H₂CO₃_C : ℕ := 1
def num_H₂CO₃_O : ℕ := 3

def molecular_weight (num_H num_C num_O : ℕ) 
                     (weight_H weight_C weight_O : ℝ) : ℝ :=
  num_H * weight_H + num_C * weight_C + num_O * weight_O

-- Main proof statement
theorem weight_of_5_moles_H₂CO₃ :
  5 * molecular_weight num_H₂CO₃_H num_H₂CO₃_C num_H₂CO₃_O 
                       atomic_weight_H atomic_weight_C atomic_weight_O 
  = 310.12 := by
  sorry

end NUMINAMATH_GPT_weight_of_5_moles_H₂CO₃_l1895_189508


namespace NUMINAMATH_GPT_days_to_learn_all_vowels_l1895_189549

-- Defining the number of vowels
def number_of_vowels : Nat := 5

-- Defining the days Charles takes to learn one alphabet
def days_per_vowel : Nat := 7

-- Prove that Charles needs 35 days to learn all the vowels
theorem days_to_learn_all_vowels : number_of_vowels * days_per_vowel = 35 := by
  sorry

end NUMINAMATH_GPT_days_to_learn_all_vowels_l1895_189549


namespace NUMINAMATH_GPT_candidate_1_fails_by_40_marks_l1895_189575

-- Definitions based on the conditions
def total_marks (T : ℕ) := T
def passing_marks (pass : ℕ) := pass = 160
def candidate_1_failed_by (marks_failed_by : ℕ) := ∃ (T : ℕ), (0.4 : ℝ) * T = 0.4 * T ∧ (0.6 : ℝ) * T - 20 = 160

-- Theorem to prove the first candidate fails by 40 marks
theorem candidate_1_fails_by_40_marks (marks_failed_by : ℕ) : candidate_1_failed_by marks_failed_by → marks_failed_by = 40 :=
by
  sorry

end NUMINAMATH_GPT_candidate_1_fails_by_40_marks_l1895_189575


namespace NUMINAMATH_GPT_total_days_2001_to_2004_l1895_189530

def regular_year_days : ℕ := 365
def leap_year_days : ℕ := 366
def num_regular_years : ℕ := 3
def num_leap_years : ℕ := 1

theorem total_days_2001_to_2004 : 
  (num_regular_years * regular_year_days) + (num_leap_years * leap_year_days) = 1461 :=
by
  sorry

end NUMINAMATH_GPT_total_days_2001_to_2004_l1895_189530


namespace NUMINAMATH_GPT_B_and_C_together_l1895_189548

theorem B_and_C_together (A B C : ℕ) (h1 : A + B + C = 1000) (h2 : A + C = 700) (h3 : C = 300) :
  B + C = 600 :=
by
  sorry

end NUMINAMATH_GPT_B_and_C_together_l1895_189548


namespace NUMINAMATH_GPT_largest_number_of_gold_coins_l1895_189590

theorem largest_number_of_gold_coins 
  (num_friends : ℕ)
  (extra_coins : ℕ)
  (total_coins : ℕ) :
  num_friends = 13 →
  extra_coins = 3 →
  total_coins < 150 →
  ∀ k : ℕ, total_coins = num_friends * k + extra_coins →
  total_coins ≤ 146 :=
by
  sorry

end NUMINAMATH_GPT_largest_number_of_gold_coins_l1895_189590


namespace NUMINAMATH_GPT_num_shirts_sold_l1895_189573

theorem num_shirts_sold (p_jeans : ℕ) (c_shirt : ℕ) (total_earnings : ℕ) (h1 : p_jeans = 10) (h2 : c_shirt = 10) (h3 : total_earnings = 400) : ℕ :=
  let c_jeans := 2 * c_shirt
  let n_shirts := 20
  have h4 : p_jeans * c_jeans + n_shirts * c_shirt = total_earnings := by sorry
  n_shirts

end NUMINAMATH_GPT_num_shirts_sold_l1895_189573


namespace NUMINAMATH_GPT_sufficient_not_necessary_l1895_189500

theorem sufficient_not_necessary (x : ℝ) : (x^2 - 3 * x + 2 ≠ 0) → (x ≠ 1) ∧ ¬((x ≠ 1) → (x^2 - 3 * x + 2 ≠ 0)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l1895_189500


namespace NUMINAMATH_GPT_meaningful_fraction_implies_neq_neg4_l1895_189564

theorem meaningful_fraction_implies_neq_neg4 (x : ℝ) : (x + 4 ≠ 0) ↔ (x ≠ -4) := 
by
  sorry

end NUMINAMATH_GPT_meaningful_fraction_implies_neq_neg4_l1895_189564


namespace NUMINAMATH_GPT_number_of_rows_l1895_189591

-- Definitions of the conditions
def total_students : ℕ := 23
def students_in_restroom : ℕ := 2
def students_absent : ℕ := 3 * students_in_restroom - 1
def students_per_desk : ℕ := 6
def fraction_full (r : ℕ) := (2 * r) / 3

-- The statement we need to prove 
theorem number_of_rows : (total_students - students_in_restroom - students_absent) / (students_per_desk * 2 / 3) = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_rows_l1895_189591


namespace NUMINAMATH_GPT_BretCatchesFrogs_l1895_189572

-- Define the number of frogs caught by Alster, Quinn, and Bret.
def AlsterFrogs : Nat := 2
def QuinnFrogs (a : Nat) : Nat := 2 * a
def BretFrogs (q : Nat) : Nat := 3 * q

-- The main theorem to prove
theorem BretCatchesFrogs : BretFrogs (QuinnFrogs AlsterFrogs) = 12 :=
by
  sorry

end NUMINAMATH_GPT_BretCatchesFrogs_l1895_189572


namespace NUMINAMATH_GPT_find_x_l1895_189596

theorem find_x (x : ℝ) (h : ⌈x⌉ * x = 210) : x = 14 := by
  sorry

end NUMINAMATH_GPT_find_x_l1895_189596


namespace NUMINAMATH_GPT_sum_of_two_integers_l1895_189512

theorem sum_of_two_integers (a b : ℕ) (h₁ : a * b + a + b = 135) (h₂ : Nat.gcd a b = 1) (h₃ : a < 30) (h₄ : b < 30) : a + b = 23 :=
sorry

end NUMINAMATH_GPT_sum_of_two_integers_l1895_189512


namespace NUMINAMATH_GPT_find_k_l1895_189583

noncomputable def g (x : ℕ) : ℤ := 2 * x^2 - 8 * x + 8

theorem find_k :
  (g 2 = 0) ∧ 
  (90 < g 9) ∧ (g 9 < 100) ∧
  (120 < g 10) ∧ (g 10 < 130) ∧
  ∃ (k : ℤ), 7000 * k < g 150 ∧ g 150 < 7000 * (k + 1)
  → ∃ (k : ℤ), k = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1895_189583


namespace NUMINAMATH_GPT_count_true_statements_l1895_189592

open Set

variable {M P : Set α}

theorem count_true_statements (h : ¬ ∀ x ∈ M, x ∈ P) (hne : Nonempty M) :
  (¬ ∃ x, x ∈ M ∧ x ∈ P ∨ ∀ x, x ∈ M → x ∈ P) ∧ (∃ x, x ∈ M ∧ x ∉ P) ∧ 
  ¬ (∃ x, x ∈ M ∧ x ∈ P) ∧ (¬ ∀ x, x ∈ M → x ∈ P) :=
sorry

end NUMINAMATH_GPT_count_true_statements_l1895_189592


namespace NUMINAMATH_GPT_part1_solution_set_l1895_189567

theorem part1_solution_set (a : ℝ) (x : ℝ) : a = -2 → (a - 2) * x ^ 2 + 2 * (a - 2) * x - 4 < 0 ↔ x ≠ -1 :=
by sorry

end NUMINAMATH_GPT_part1_solution_set_l1895_189567


namespace NUMINAMATH_GPT_xn_plus_inv_xn_l1895_189537

theorem xn_plus_inv_xn (θ : ℝ) (x : ℝ) (n : ℕ) (h₀ : 0 < θ) (h₁ : θ < π / 2)
  (h₂ : x + 1 / x = -2 * Real.sin θ) (hn_pos : 0 < n) :
  x ^ n + x⁻¹ ^ n = -2 * Real.sin (n * θ) := by
  sorry

end NUMINAMATH_GPT_xn_plus_inv_xn_l1895_189537


namespace NUMINAMATH_GPT_correct_evaluation_l1895_189577

noncomputable def evaluate_expression : ℚ :=
  - (2 : ℚ) ^ 3 + (6 / 5) * (2 / 5)

theorem correct_evaluation : evaluate_expression = -7 - 13 / 25 :=
by
  unfold evaluate_expression
  sorry

end NUMINAMATH_GPT_correct_evaluation_l1895_189577


namespace NUMINAMATH_GPT_johns_total_distance_l1895_189569

theorem johns_total_distance :
  let monday := 1700
  let tuesday := monday + 200
  let wednesday := 0.7 * tuesday
  let thursday := 2 * wednesday
  let friday := 3.5 * 1000
  let saturday := 0
  monday + tuesday + wednesday + thursday + friday + saturday = 10090 := 
by
  sorry

end NUMINAMATH_GPT_johns_total_distance_l1895_189569


namespace NUMINAMATH_GPT_find_real_x_l1895_189588

theorem find_real_x (x : ℝ) : 
  (2 ≤ 3 * x / (3 * x - 7)) ∧ (3 * x / (3 * x - 7) < 6) ↔ (7 / 3 < x ∧ x < 42 / 15) :=
by
  sorry

end NUMINAMATH_GPT_find_real_x_l1895_189588


namespace NUMINAMATH_GPT_initial_alcohol_solution_percentage_l1895_189550

noncomputable def initial_percentage_of_alcohol (P : ℝ) :=
  let initial_volume := 6 -- initial volume of solution in liters
  let added_alcohol := 1.2 -- added volume of pure alcohol in liters
  let final_volume := initial_volume + added_alcohol -- final volume in liters
  let final_percentage := 0.5 -- final percentage of alcohol
  ∃ P, (initial_volume * (P / 100) + added_alcohol) / final_volume = final_percentage

theorem initial_alcohol_solution_percentage : initial_percentage_of_alcohol 40 :=
by 
  -- Prove that initial percentage P is 40
  have hs : initial_percentage_of_alcohol 40 := by sorry
  exact hs

end NUMINAMATH_GPT_initial_alcohol_solution_percentage_l1895_189550


namespace NUMINAMATH_GPT_number_of_typists_needed_l1895_189589

theorem number_of_typists_needed :
  (∃ t : ℕ, (20 * 40) / 20 * 60 * t = 180) ↔ t = 30 :=
by sorry

end NUMINAMATH_GPT_number_of_typists_needed_l1895_189589


namespace NUMINAMATH_GPT_number_times_quarter_squared_eq_four_cubed_l1895_189554

theorem number_times_quarter_squared_eq_four_cubed : 
  ∃ (number : ℕ), number * (1 / 4 : ℚ) ^ 2 = (4 : ℚ) ^ 3 ∧ number = 1024 :=
by 
  use 1024
  sorry

end NUMINAMATH_GPT_number_times_quarter_squared_eq_four_cubed_l1895_189554


namespace NUMINAMATH_GPT_pirates_total_coins_l1895_189552

theorem pirates_total_coins :
  ∀ (x : ℕ), (x * (x + 1)) / 2 = 5 * x → 6 * x = 54 :=
by
  intro x
  intro h
  -- proof omitted
  sorry

end NUMINAMATH_GPT_pirates_total_coins_l1895_189552


namespace NUMINAMATH_GPT_number_of_valid_sets_l1895_189580

universe u

def U : Set ℕ := {1,2,3,4,5,6,7,8,9,10}
def valid_set (A : Set ℕ) : Prop :=
  ∃ a1 a2 a3, A = {a1, a2, a3} ∧ a3 ∈ U ∧ a2 ∈ U ∧ a1 ∈ U ∧ a3 ≥ a2 + 1 ∧ a2 ≥ a1 + 4

theorem number_of_valid_sets : ∃ (n : ℕ), n = 56 ∧ ∃ S : Finset (Set ℕ), (∀ A ∈ S, valid_set A) ∧ S.card = n := by
  sorry

end NUMINAMATH_GPT_number_of_valid_sets_l1895_189580


namespace NUMINAMATH_GPT_range_of_m_l1895_189566

noncomputable def P (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
noncomputable def Q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) : 
  (∀ x, P x → Q x m ∧ P x) ↔ (0 < m ∧ m ≤ 3) :=
sorry

end NUMINAMATH_GPT_range_of_m_l1895_189566


namespace NUMINAMATH_GPT_product_of_solutions_l1895_189560

theorem product_of_solutions (α β : ℝ) (h : 2 * α^2 + 8 * α - 45 = 0 ∧ 2 * β^2 + 8 * β - 45 = 0 ∧ α ≠ β) :
  α * β = -22.5 :=
sorry

end NUMINAMATH_GPT_product_of_solutions_l1895_189560


namespace NUMINAMATH_GPT_number_of_members_l1895_189515

theorem number_of_members (n : ℕ) (h : n * n = 8649) : n = 93 :=
by
  sorry

end NUMINAMATH_GPT_number_of_members_l1895_189515


namespace NUMINAMATH_GPT_exterior_angle_of_polygon_l1895_189542

theorem exterior_angle_of_polygon (n : ℕ) (h₁ : (n - 2) * 180 = 1800) (h₂ : n > 2) :
  360 / n = 30 := by
    sorry

end NUMINAMATH_GPT_exterior_angle_of_polygon_l1895_189542


namespace NUMINAMATH_GPT_quotient_division_l1895_189551

noncomputable def poly_division_quotient : Polynomial ℚ :=
  Polynomial.div (9 * Polynomial.X ^ 4 + 8 * Polynomial.X ^ 3 - 12 * Polynomial.X ^ 2 - 7 * Polynomial.X + 4) (3 * Polynomial.X ^ 2 + 2 * Polynomial.X + 5)

theorem quotient_division :
  poly_division_quotient = (3 * Polynomial.X ^ 2 - 2 * Polynomial.X + 2) :=
sorry

end NUMINAMATH_GPT_quotient_division_l1895_189551


namespace NUMINAMATH_GPT_pirates_treasure_l1895_189581

theorem pirates_treasure (m : ℝ) :
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by
  sorry

end NUMINAMATH_GPT_pirates_treasure_l1895_189581


namespace NUMINAMATH_GPT_students_in_5th_6th_grades_l1895_189594

-- Definitions for problem conditions
def is_three_digit_number (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000

def six_two_digit_sum_eq_twice (n : ℕ) : Prop :=
  ∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧
               a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
               (10 * a + b) + (10 * b + a) + (10 * a + c) + (10 * c + a) + (10 * b + c) + (10 * c + b) = 2 * n

-- The proof problem statement in Lean 4
theorem students_in_5th_6th_grades :
  ∃ n : ℕ, is_three_digit_number n ∧ six_two_digit_sum_eq_twice n ∧ n = 198 :=
by
  sorry

end NUMINAMATH_GPT_students_in_5th_6th_grades_l1895_189594


namespace NUMINAMATH_GPT_infinitely_many_composite_z_l1895_189597

theorem infinitely_many_composite_z (m n : ℕ) (h_m : m > 1) : ¬ (Nat.Prime (n^4 + 4*m^4)) :=
by
  sorry

end NUMINAMATH_GPT_infinitely_many_composite_z_l1895_189597


namespace NUMINAMATH_GPT_work_days_together_l1895_189538

variable (d : ℝ) (j : ℝ)

theorem work_days_together (hd : d = 1 / 5) (hj : j = 1 / 9) :
  1 / (d + j) = 45 / 14 := by
  sorry

end NUMINAMATH_GPT_work_days_together_l1895_189538


namespace NUMINAMATH_GPT_wanda_blocks_l1895_189541

theorem wanda_blocks (initial_blocks: ℕ) (additional_blocks: ℕ) (total_blocks: ℕ) : 
  initial_blocks = 4 → additional_blocks = 79 → total_blocks = initial_blocks + additional_blocks → total_blocks = 83 :=
by
  intros hi ha ht
  rw [hi, ha] at ht
  exact ht

end NUMINAMATH_GPT_wanda_blocks_l1895_189541


namespace NUMINAMATH_GPT_seventy_second_number_in_S_is_573_l1895_189562

open Nat

def S : Set Nat := { k | k % 8 = 5 }

theorem seventy_second_number_in_S_is_573 : ∃ k ∈ (Finset.range 650), k = 8 * 71 + 5 :=
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_seventy_second_number_in_S_is_573_l1895_189562


namespace NUMINAMATH_GPT_min_xy_solution_l1895_189586

theorem min_xy_solution (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 2 * x + 8 * y) :
  (x = 16 ∧ y = 4) :=
by
  sorry

end NUMINAMATH_GPT_min_xy_solution_l1895_189586


namespace NUMINAMATH_GPT_sum_geometric_series_l1895_189545

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_sum_geometric_series_l1895_189545


namespace NUMINAMATH_GPT_linda_age_l1895_189511

theorem linda_age 
  (J : ℕ)  -- Jane's current age
  (H1 : ∃ J, 2 * J + 3 = 13) -- Linda is 3 more than 2 times the age of Jane
  (H2 : (J + 5) + ((2 * J + 3) + 5) = 28) -- In 5 years, the sum of their ages will be 28
  : 2 * J + 3 = 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_linda_age_l1895_189511


namespace NUMINAMATH_GPT_basketball_points_l1895_189585

/-
In a basketball league, each game must have a winner and a loser. 
A team earns 2 points for a win and 1 point for a loss. 
A certain team expects to earn at least 48 points in all 32 games of 
the 2012-2013 season in order to have a chance to enter the playoffs. 
If this team wins x games in the upcoming matches, prove that
the relationship that x should satisfy to reach the goal is:
    2x + (32 - x) ≥ 48.
-/
theorem basketball_points (x : ℕ) (h : 0 ≤ x ∧ x ≤ 32) :
    2 * x + (32 - x) ≥ 48 :=
sorry

end NUMINAMATH_GPT_basketball_points_l1895_189585


namespace NUMINAMATH_GPT_range_of_a_l1895_189555

variable (a : ℝ)

def condition1 : Prop := a < 0
def condition2 : Prop := -a / 2 ≥ 1
def condition3 : Prop := -1 - a - 5 ≤ a

theorem range_of_a :
  condition1 a ∧ condition2 a ∧ condition3 a → -3 ≤ a ∧ a ≤ -2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1895_189555
