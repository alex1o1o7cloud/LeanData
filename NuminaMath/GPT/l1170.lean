import Mathlib

namespace percentage_error_in_area_l1170_117080

theorem percentage_error_in_area (s : ℝ) (h_s_pos: s > 0) :
  let measured_side := 1.01 * s
  let actual_area := s ^ 2
  let measured_area := measured_side ^ 2
  let error_in_area := measured_area - actual_area
  (error_in_area / actual_area) * 100 = 2.01 :=
by
  sorry

end percentage_error_in_area_l1170_117080


namespace provider_choices_count_l1170_117021

theorem provider_choices_count :
  let num_providers := 25
  let num_s_providers := 6
  let remaining_providers_after_laura := num_providers - 1
  let remaining_providers_after_brother := remaining_providers_after_laura - 1

  (num_providers * num_s_providers * remaining_providers_after_laura * remaining_providers_after_brother) = 75900 :=
by
  sorry

end provider_choices_count_l1170_117021


namespace parallel_slope_l1170_117095

theorem parallel_slope (x y : ℝ) : (∃ b : ℝ, 3 * x - 6 * y = 12) → (∀ (x' y' : ℝ), (∃ b' : ℝ, 3 * x' - 6 * y' = b') → (∃ m : ℝ, m = 1 / 2)) :=
by
  sorry

end parallel_slope_l1170_117095


namespace mistaken_multiplier_is_34_l1170_117002

-- Define the main conditions
def correct_number : ℕ := 135
def correct_multiplier : ℕ := 43
def difference : ℕ := 1215

-- Define what we need to prove
theorem mistaken_multiplier_is_34 :
  (correct_number * correct_multiplier - correct_number * x = difference) →
  x = 34 :=
by
  sorry

end mistaken_multiplier_is_34_l1170_117002


namespace min_value_x_y_l1170_117008

open Real

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / (x + 2) + 1 / (y + 2) = 1 / 6) : 
  x + y ≥ 20 :=
sorry

end min_value_x_y_l1170_117008


namespace find_y_l1170_117086

theorem find_y (x y : ℕ) (h1 : x^2 = y + 3) (h2 : x = 6) : y = 33 := 
by
  sorry

end find_y_l1170_117086


namespace eccentricity_of_ellipse_l1170_117011

noncomputable def e (a b c : ℝ) : ℝ := c / a

theorem eccentricity_of_ellipse (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : (a - c) * (a + c) = (2 * c)^2) : e a b c = (Real.sqrt 5) / 5 := 
by
  sorry

end eccentricity_of_ellipse_l1170_117011


namespace option_c_equals_9_l1170_117012

theorem option_c_equals_9 : (3 * 3 - 3 + 3) = 9 :=
by
  sorry

end option_c_equals_9_l1170_117012


namespace total_ticket_sales_cost_l1170_117090

theorem total_ticket_sales_cost
  (num_orchestra num_balcony : ℕ)
  (price_orchestra price_balcony : ℕ)
  (total_tickets total_revenue : ℕ)
  (h1 : num_orchestra + num_balcony = 370)
  (h2 : num_balcony = num_orchestra + 190)
  (h3 : price_orchestra = 12)
  (h4 : price_balcony = 8)
  (h5 : total_tickets = 370)
  : total_revenue = 3320 := by
  sorry

end total_ticket_sales_cost_l1170_117090


namespace operation_5_7_eq_35_l1170_117089

noncomputable def operation (x y : ℝ) : ℝ := sorry

axiom condition1 :
  ∀ (x y : ℝ), (x * y > 0) → (operation (x * y) y = x * (operation y y))

axiom condition2 :
  ∀ (x : ℝ), (x > 0) → (operation (operation x 1) x = operation x 1)

axiom condition3 :
  (operation 1 1 = 2)

theorem operation_5_7_eq_35 : operation 5 7 = 35 :=
by
  sorry

end operation_5_7_eq_35_l1170_117089


namespace find_K_l1170_117036

theorem find_K (surface_area_cube : ℝ) (volume_sphere : ℝ) (r : ℝ) (K : ℝ) 
  (cube_side_length : ℝ) (surface_area_sphere_eq : surface_area_cube = 4 * Real.pi * (r ^ 2))
  (volume_sphere_eq : volume_sphere = (4 / 3) * Real.pi * (r ^ 3)) 
  (surface_area_cube_eq : surface_area_cube = 6 * (cube_side_length ^ 2)) 
  (volume_sphere_form : volume_sphere = (K * Real.sqrt 6) / Real.sqrt Real.pi) :
  K = 8 :=
by
  sorry

end find_K_l1170_117036


namespace isosceles_right_triangle_example_l1170_117024

theorem isosceles_right_triangle_example :
  (5 = 5) ∧ (5^2 + 5^2 = (5 * Real.sqrt 2)^2) :=
by {
  sorry
}

end isosceles_right_triangle_example_l1170_117024


namespace common_difference_l1170_117081

theorem common_difference (a : ℕ → ℝ) (d : ℝ) (h_seq : ∀ n, a n = 1 + (n - 1) * d) 
  (h_geom : (a 3) ^ 2 = (a 1) * (a 13)) (h_ne_zero: d ≠ 0) : d = 2 :=
by
  sorry

end common_difference_l1170_117081


namespace solve_eq1_solve_system_l1170_117014

theorem solve_eq1 : ∃ x y : ℝ, (3 / x) + (2 / y) = 4 :=
by
  use 1
  use 2
  sorry

theorem solve_system :
  ∃ x y : ℝ,
    (3 / x + 2 / y = 4) ∧ (5 / x - 6 / y = 2) ∧ (x = 1) ∧ (y = 2) :=
by
  use 1
  use 2
  sorry

end solve_eq1_solve_system_l1170_117014


namespace solve_equation1_solve_equation2_l1170_117092

-- Define the equations and the problem.
def equation1 (x : ℝ) : Prop := (3 / (x^2 - 9)) + (x / (x - 3)) = 1
def equation2 (x : ℝ) : Prop := 2 - (1 / (2 - x)) = ((3 - x) / (x - 2))

-- Proof problem for the first equation: Prove that x = -4 is the solution.
theorem solve_equation1 : ∀ x : ℝ, equation1 x → x = -4 :=
by {
  sorry
}

-- Proof problem for the second equation: Prove that there are no solutions.
theorem solve_equation2 : ∀ x : ℝ, ¬equation2 x :=
by {
  sorry
}

end solve_equation1_solve_equation2_l1170_117092


namespace sum_of_repeating_decimals_l1170_117065

-- Definitions of repeating decimals x and y
def x : ℚ := 25 / 99
def y : ℚ := 87 / 99

-- The assertion that the sum of these repeating decimals is equal to 112/99 as a fraction
theorem sum_of_repeating_decimals: x + y = 112 / 99 := by
  sorry

end sum_of_repeating_decimals_l1170_117065


namespace integer_solutions_exist_l1170_117017

theorem integer_solutions_exist (R₀ : ℝ) : 
  ∃ (x₁ x₂ x₃ : ℤ), (x₁^2 + x₂^2 + x₃^2 = x₁ * x₂ * x₃) ∧ (R₀ < x₁) ∧ (R₀ < x₂) ∧ (R₀ < x₃) := 
sorry

end integer_solutions_exist_l1170_117017


namespace net_percentage_error_l1170_117026

noncomputable section
def calculate_percentage_error (true_side excess_error deficit_error : ℝ) : ℝ :=
  let measured_side1 := true_side * (1 + excess_error / 100)
  let measured_side2 := measured_side1 * (1 - deficit_error / 100)
  let true_area := true_side ^ 2
  let calculated_area := measured_side2 * true_side
  let percentage_error := ((true_area - calculated_area) / true_area) * 100
  percentage_error

theorem net_percentage_error 
  (S : ℝ) (h1 : S > 0) : calculate_percentage_error S 3 (-4) = 1.12 := by
  sorry

end net_percentage_error_l1170_117026


namespace average_earning_week_l1170_117083

theorem average_earning_week (D1 D2 D3 D4 D5 D6 D7 : ℝ) 
  (h1 : (D1 + D2 + D3 + D4) / 4 = 18)
  (h2 : (D4 + D5 + D6 + D7) / 4 = 22)
  (h3 : D4 = 13) : 
  (D1 + D2 + D3 + D4 + D5 + D6 + D7) / 7 = 22.86 := 
by 
  sorry

end average_earning_week_l1170_117083


namespace line_of_intersection_canonical_form_l1170_117004

def canonical_form_of_line (A B : ℝ) (x y z : ℝ) :=
  (x / A) = (y / B) ∧ (y / B) = (z)

theorem line_of_intersection_canonical_form :
  ∀ (x y z : ℝ),
  x + y - 2*z - 2 = 0 →
  x - y + z + 2 = 0 →
  canonical_form_of_line (-1) (-3) x (y - 2) (-2) :=
by
  intros x y z h_eq1 h_eq2
  sorry

end line_of_intersection_canonical_form_l1170_117004


namespace solveEquation_l1170_117032

noncomputable def findNonZeroSolution (z : ℝ) : Prop :=
  (5 * z) ^ 10 = (20 * z) ^ 5 ∧ z ≠ 0

theorem solveEquation : ∃ z : ℝ, findNonZeroSolution z ∧ z = 4 / 5 := by
  exists 4 / 5
  simp [findNonZeroSolution]
  sorry

end solveEquation_l1170_117032


namespace inequality_ab_l1170_117079

theorem inequality_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) :
  1 / (a^2 + 1) + 1 / (b^2 + 1) ≥ 1 := 
sorry

end inequality_ab_l1170_117079


namespace total_oranges_proof_l1170_117071

def jeremyMonday : ℕ := 100
def jeremyTuesdayPlusBrother : ℕ := 3 * jeremyMonday
def jeremyWednesdayPlusBrotherPlusCousin : ℕ := 2 * jeremyTuesdayPlusBrother
def jeremyThursday : ℕ := (70 * jeremyMonday) / 100
def cousinWednesday : ℕ := jeremyTuesdayPlusBrother - (20 * jeremyTuesdayPlusBrother) / 100
def cousinThursday : ℕ := cousinWednesday + (30 * cousinWednesday) / 100

def total_oranges : ℕ :=
  jeremyMonday + jeremyTuesdayPlusBrother + jeremyWednesdayPlusBrotherPlusCousin + (jeremyThursday + (jeremyWednesdayPlusBrotherPlusCousin - cousinWednesday) + cousinThursday)

theorem total_oranges_proof : total_oranges = 1642 :=
by
  sorry

end total_oranges_proof_l1170_117071


namespace triangle_side_length_x_l1170_117029

theorem triangle_side_length_x (x : ℤ) (hpos : x > 0) (hineq1 : 7 < x^2) (hineq2 : x^2 < 17) :
    x = 3 ∨ x = 4 :=
by {
  apply sorry
}

end triangle_side_length_x_l1170_117029


namespace sum_of_coeffs_in_expansion_l1170_117098

theorem sum_of_coeffs_in_expansion (n : ℕ) : 
  (1 - 2 : ℤ)^n = (-1 : ℤ)^n :=
by
  sorry

end sum_of_coeffs_in_expansion_l1170_117098


namespace volume_tetrahedron_OABC_correct_l1170_117084

noncomputable def volume_tetrahedron_OABC : ℝ :=
  let a := Real.sqrt 33
  let b := 4
  let c := 4 * Real.sqrt 3
  (1 / 6) * a * b * c

theorem volume_tetrahedron_OABC_correct :
  let a := Real.sqrt 33
  let b := 4
  let c := 4 * Real.sqrt 3
  let volume := (1 / 6) * a * b * c
  volume = 8 * Real.sqrt 99 / 3 :=
by
  sorry

end volume_tetrahedron_OABC_correct_l1170_117084


namespace slope_angle_of_y_eq_0_l1170_117034

theorem slope_angle_of_y_eq_0  :
  ∀ (α : ℝ), (∀ (y x : ℝ), y = 0) → α = 0 :=
by
  intros α h
  sorry

end slope_angle_of_y_eq_0_l1170_117034


namespace solution_l1170_117046

noncomputable def inequality_prove (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 4) : Prop :=
  (1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5)

noncomputable def equality_condition (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 4) : Prop :=
  (1 / (x + 3) + 1 / (y + 3) = 2 / 5) ↔ (x = 2 ∧ y = 2)

theorem solution (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 4) : 
  inequality_prove x y h1 h2 h3 ∧ equality_condition x y h1 h2 h3 := by
  sorry

end solution_l1170_117046


namespace intersection_A_B_l1170_117023

-- Definitions of sets A and B
def A := { x : ℝ | x ≥ -1 }
def B := { y : ℝ | y < 1 }

-- Statement to prove the intersection of A and B
theorem intersection_A_B : A ∩ B = { x : ℝ | -1 ≤ x ∧ x < 1 } :=
by
  sorry

end intersection_A_B_l1170_117023


namespace total_balloons_correct_l1170_117045

-- Define the number of blue balloons Joan and Melanie have
def Joan_balloons : ℕ := 40
def Melanie_balloons : ℕ := 41

-- Define the total number of blue balloons
def total_balloons : ℕ := Joan_balloons + Melanie_balloons

-- Prove that the total number of blue balloons is 81
theorem total_balloons_correct : total_balloons = 81 := by
  sorry

end total_balloons_correct_l1170_117045


namespace solve_equation_l1170_117001

theorem solve_equation :
  ∃ x : ℝ, (x - 2)^3 + (x - 6)^3 = 54 ∧ x = 7 := by
sorry

end solve_equation_l1170_117001


namespace quadratic_factoring_even_a_l1170_117016

theorem quadratic_factoring_even_a (a : ℤ) :
  (∃ (m p n q : ℤ), 21 * x^2 + a * x + 21 = (m * x + n) * (p * x + q) ∧ m * p = 21 ∧ n * q = 21 ∧ (∃ (k : ℤ), a = 2 * k)) :=
sorry

end quadratic_factoring_even_a_l1170_117016


namespace remainder_of_1234567_div_257_l1170_117031

theorem remainder_of_1234567_div_257 : 1234567 % 257 = 123 := by
  sorry

end remainder_of_1234567_div_257_l1170_117031


namespace total_pages_in_book_l1170_117009

def pagesReadMonday := 23
def pagesReadTuesday := 38
def pagesReadWednesday := 61
def pagesReadThursday := 12
def pagesReadFriday := 2 * pagesReadThursday

def totalPagesRead := pagesReadMonday + pagesReadTuesday + pagesReadWednesday + pagesReadThursday + pagesReadFriday

theorem total_pages_in_book :
  totalPagesRead = 158 :=
by
  sorry

end total_pages_in_book_l1170_117009


namespace stephen_total_distance_l1170_117055

def speed_first_segment := 16 -- miles per hour
def time_first_segment := 10 / 60 -- hours

def speed_second_segment := 12 -- miles per hour
def headwind := 2 -- miles per hour
def actual_speed_second_segment := speed_second_segment - headwind
def time_second_segment := 20 / 60 -- hours

def speed_third_segment := 20 -- miles per hour
def tailwind := 4 -- miles per hour
def actual_speed_third_segment := speed_third_segment + tailwind
def time_third_segment := 15 / 60 -- hours

def distance_first_segment := speed_first_segment * time_first_segment
def distance_second_segment := actual_speed_second_segment * time_second_segment
def distance_third_segment := actual_speed_third_segment * time_third_segment

theorem stephen_total_distance : distance_first_segment + distance_second_segment + distance_third_segment = 12 := by
  sorry

end stephen_total_distance_l1170_117055


namespace boys_to_admit_or_expel_l1170_117091

-- Definitions from the conditions
def total_students : ℕ := 500

def girls_percent (x : ℕ) : ℕ := (x * total_students) / 100

-- Definition of the calculation under the new policy
def required_boys : ℕ := (total_students * 3) / 5

-- Main statement we need to prove
theorem boys_to_admit_or_expel (x : ℕ) (htotal : x + girls_percent x = total_students) :
  required_boys - x = 217 := by
  sorry

end boys_to_admit_or_expel_l1170_117091


namespace moles_of_water_formed_l1170_117005

-- Definitions (conditions)
def reaction : String := "NaOH + HCl → NaCl + H2O"

def initial_moles_NaOH : ℕ := 1
def initial_moles_HCl : ℕ := 1
def mole_ratio_NaOH_HCl : ℕ := 1
def mole_ratio_NaOH_H2O : ℕ := 1

-- The proof problem
theorem moles_of_water_formed :
  initial_moles_NaOH = mole_ratio_NaOH_HCl →
  initial_moles_HCl = mole_ratio_NaOH_HCl →
  mole_ratio_NaOH_H2O * initial_moles_NaOH = 1 :=
by
  intros h1 h2
  sorry

end moles_of_water_formed_l1170_117005


namespace solve_equation_l1170_117075

theorem solve_equation (x : ℤ) (h1 : x ≠ 2) : x - 8 / (x - 2) = 5 - 8 / (x - 2) → x = 5 := by
  sorry

end solve_equation_l1170_117075


namespace smallest_y_for_perfect_cube_l1170_117051

-- Define the given conditions
def x : ℕ := 5 * 24 * 36

-- State the theorem to prove
theorem smallest_y_for_perfect_cube (y : ℕ) (h : y = 50) : 
  ∃ y, (x * y) % (y * y * y) = 0 :=
by
  sorry

end smallest_y_for_perfect_cube_l1170_117051


namespace function_behavior_l1170_117064

noncomputable def f (x : ℝ) : ℝ := abs (2^x - 2)

theorem function_behavior :
  (∀ x y : ℝ, x < y ∧ y ≤ 1 → f x ≥ f y) ∧ (∀ x y : ℝ, x < y ∧ x ≥ 1 → f x ≤ f y) :=
by
  sorry

end function_behavior_l1170_117064


namespace find_some_number_l1170_117015

-- Definitions based on the given condition
def some_number : ℝ := sorry
def equation := some_number * 3.6 / (0.04 * 0.1 * 0.007) = 990.0000000000001

-- An assertion/proof that given the equation, some_number equals 7.7
theorem find_some_number (h : equation) : some_number = 7.7 :=
sorry

end find_some_number_l1170_117015


namespace angle_division_l1170_117058

theorem angle_division (α : ℝ) (n : ℕ) (θ : ℝ) (h : α = 78) (hn : n = 26) (ht : θ = 3) :
  α / n = θ :=
by
  sorry

end angle_division_l1170_117058


namespace valid_three_digit_numbers_no_seven_nine_l1170_117020

noncomputable def count_valid_three_digit_numbers : Nat := 
  let hundredsChoices := 7
  let tensAndUnitsChoices := 8
  hundredsChoices * tensAndUnitsChoices * tensAndUnitsChoices

theorem valid_three_digit_numbers_no_seven_nine : 
  count_valid_three_digit_numbers = 448 := by
  sorry

end valid_three_digit_numbers_no_seven_nine_l1170_117020


namespace find_k_l1170_117076

theorem find_k : 
  ∀ (k : ℤ), 2^4 - 6 = 3^3 + k ↔ k = -17 :=
by sorry

end find_k_l1170_117076


namespace inequality_solution_l1170_117022

theorem inequality_solution :
  {x : ℝ | (x - 1) * (x - 4) * (x - 5)^2 / ((x - 3) * (x^2 - 9)) > 0} = { x : ℝ | -3 < x ∧ x < 3 } :=
sorry

end inequality_solution_l1170_117022


namespace max_value_a4_a6_l1170_117088

theorem max_value_a4_a6 (a : ℕ → ℝ) (d : ℝ) (h1 : d ≥ 0) (h2 : ∀ n, a n > 0) (h3 : a 3 + 2 * a 6 = 6) :
  ∃ m, ∀ (a : ℕ → ℝ) (d : ℝ) (h1 : d ≥ 0) (h2 : ∀ n, a n > 0) (h3 : a 3 + 2 * a 6 = 6), a 4 * a 6 ≤ m :=
sorry

end max_value_a4_a6_l1170_117088


namespace cos_identity_l1170_117060

noncomputable def f (x : ℝ) : ℝ :=
  let a := (2 * Real.cos x, (Real.sqrt 3) / 2)
  let b := (Real.sin (x - Real.pi / 3), 1)
  a.1 * b.1 + a.2 * b.2

theorem cos_identity (x0 : ℝ) (hx0 : x0 ∈ Set.Icc (5 * Real.pi / 12) (2 * Real.pi / 3))
  (hf : f x0 = 4 / 5) :
  Real.cos (2 * x0 - Real.pi / 12) = -7 * Real.sqrt 2 / 10 :=
sorry

end cos_identity_l1170_117060


namespace abs_inequality_no_solution_l1170_117097

theorem abs_inequality_no_solution (a : ℝ) : (∀ x : ℝ, |x - 5| + |x + 3| ≥ a) ↔ a ≤ 8 :=
by sorry

end abs_inequality_no_solution_l1170_117097


namespace existence_of_unique_root_l1170_117035

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x - 5

theorem existence_of_unique_root :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  f 0 = -4 ∧
  f 2 = Real.exp 2 - 1 →
  ∃! c, f c = 0 :=
by
  sorry

end existence_of_unique_root_l1170_117035


namespace high_quality_chip_prob_l1170_117059

variable (chipsA chipsB chipsC : ℕ)
variable (qualityA qualityB qualityC : ℝ)
variable (totalChips : ℕ)

noncomputable def probability_of_high_quality_chip (chipsA chipsB chipsC : ℕ) (qualityA qualityB qualityC : ℝ) (totalChips : ℕ) : ℝ :=
  (chipsA / totalChips) * qualityA + (chipsB / totalChips) * qualityB + (chipsC / totalChips) * qualityC

theorem high_quality_chip_prob :
  let chipsA := 5
  let chipsB := 10
  let chipsC := 10
  let qualityA := 0.8
  let qualityB := 0.8
  let qualityC := 0.7
  let totalChips := 25
  probability_of_high_quality_chip chipsA chipsB chipsC qualityA qualityB qualityC totalChips = 0.76 :=
by
  sorry

end high_quality_chip_prob_l1170_117059


namespace log_expression_value_l1170_117077

theorem log_expression_value (lg : ℕ → ℤ) :
  (lg 4 = 2 * lg 2) →
  (lg 20 = lg 4 + lg 5) →
  lg 4 + lg 5 * lg 20 + (lg 5)^2 = 2 :=
by
  intros h1 h2
  sorry

end log_expression_value_l1170_117077


namespace opposite_of_2023_is_neg_2023_l1170_117067

theorem opposite_of_2023_is_neg_2023 (x : ℝ) (h : x = 2023) : -x = -2023 :=
by
  /- proof begins here, but we are skipping it with sorry -/
  sorry

end opposite_of_2023_is_neg_2023_l1170_117067


namespace base_conversion_l1170_117094

theorem base_conversion (C D : ℕ) (h₁ : 0 ≤ C ∧ C < 8) (h₂ : 0 ≤ D ∧ D < 5) (h₃ : 7 * C = 4 * D) :
  8 * C + D = 0 := by
  sorry

end base_conversion_l1170_117094


namespace seven_digit_palindromes_count_l1170_117030

theorem seven_digit_palindromes_count : 
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  (a_choices * b_choices * c_choices * d_choices) = 9000 := by
  sorry

end seven_digit_palindromes_count_l1170_117030


namespace train_length_l1170_117007

def speed_kmph := 72   -- Speed in kilometers per hour
def time_sec := 14     -- Time in seconds

/-- Function to convert speed from km/hr to m/s -/
def convert_speed (speed : ℕ) : ℕ :=
  speed * 1000 / 3600

/-- Function to calculate distance given speed and time -/
def calculate_distance (speed : ℕ) (time : ℕ) : ℕ :=
  speed * time

theorem train_length :
  calculate_distance (convert_speed speed_kmph) time_sec = 280 :=
by
  sorry

end train_length_l1170_117007


namespace delores_initial_money_l1170_117052

def computer_price : ℕ := 400
def printer_price : ℕ := 40
def headphones_price : ℕ := 60
def discount_percentage : ℕ := 10
def left_money : ℕ := 10

theorem delores_initial_money :
  ∃ initial_money : ℕ,
    initial_money = printer_price + headphones_price + (computer_price - (discount_percentage * computer_price / 100)) + left_money :=
  sorry

end delores_initial_money_l1170_117052


namespace maximum_modest_number_l1170_117025

-- Definitions and Conditions
def is_modest (a b c d : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
  5 * a = b + c + d ∧
  d % 2 = 0

def G (a b c d : ℕ) : ℕ :=
  (1000 * a + 100 * b + 10 * c + d - (1000 * c + 100 * d + 10 * a + b)) / 99

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def is_divisible_by_3 (abc : ℕ) : Prop :=
  abc % 3 = 0

-- Theorem statement
theorem maximum_modest_number :
  ∃ a b c d : ℕ, is_modest a b c d ∧ is_divisible_by_11 (G a b c d) ∧ is_divisible_by_3 (100 * a + 10 * b + c) ∧ 
  (1000 * a + 100 * b + 10 * c + d) = 3816 := 
sorry

end maximum_modest_number_l1170_117025


namespace find_p_q_l1170_117054

theorem find_p_q (p q : ℤ) 
    (h1 : (3:ℤ)^5 - 2 * (3:ℤ)^4 + 3 * (3:ℤ)^3 - p * (3:ℤ)^2 + q * (3:ℤ) - 12 = 0)
    (h2 : (-1:ℤ)^5 - 2 * (-1:ℤ)^4 + 3 * (-1:ℤ)^3 - p * (-1:ℤ)^2 + q * (-1:ℤ) - 12 = 0) : 
    (p, q) = (-8, -10) :=
by
  sorry

end find_p_q_l1170_117054


namespace consecutive_odd_integers_sum_l1170_117057

theorem consecutive_odd_integers_sum (a b c : ℤ) (h1 : a % 2 = 1) (h2 : b % 2 = 1) (h3 : c % 2 = 1) (h4 : a < b) (h5 : b < c) (h6 : c = -47) : a + b + c = -141 := 
sorry

end consecutive_odd_integers_sum_l1170_117057


namespace most_likely_outcome_l1170_117096

-- Defining the conditions
def equally_likely (n : ℕ) (k : ℕ) := (Nat.choose n k) * (1 / 2)^n

-- Defining the problem statement
theorem most_likely_outcome :
  (equally_likely 5 3 = 5 / 16 ∧ equally_likely 5 2 = 5 / 16) :=
sorry

end most_likely_outcome_l1170_117096


namespace ratio_of_people_on_buses_l1170_117041

theorem ratio_of_people_on_buses (P_2 P_3 P_4 : ℕ) 
  (h1 : P_1 = 12) 
  (h2 : P_3 = P_2 - 6) 
  (h3 : P_4 = P_1 + 9) 
  (h4 : P_1 + P_2 + P_3 + P_4 = 75) : 
  P_2 / P_1 = 2 := 
by
  sorry

end ratio_of_people_on_buses_l1170_117041


namespace g_two_eq_one_l1170_117053

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x - y) = g x * g y
axiom g_nonzero (x : ℝ) : g x ≠ 0

theorem g_two_eq_one : g 2 = 1 := by
  sorry

end g_two_eq_one_l1170_117053


namespace no_corner_cut_possible_l1170_117066

-- Define the cube and the triangle sides
def cube_edge_length : ℝ := 15
def triangle_side1 : ℝ := 5
def triangle_side2 : ℝ := 6
def triangle_side3 : ℝ := 8

-- Main statement: Prove that it's not possible to cut off a corner of the cube to form the given triangle
theorem no_corner_cut_possible :
  ¬ (∃ (a b c : ℝ),
    a^2 + b^2 = triangle_side1^2 ∧
    b^2 + c^2 = triangle_side2^2 ∧
    c^2 + a^2 = triangle_side3^2 ∧
    a^2 + b^2 + c^2 = 62.5) :=
sorry

end no_corner_cut_possible_l1170_117066


namespace radius_large_circle_l1170_117078

/-- Let R be the radius of the large circle. Assume three circles of radius 2 are externally 
tangent to each other. Two of these circles are internally tangent to the larger circle, 
and the third circle is tangent to the larger circle both internally and externally. 
Prove that the radius of the large circle is 4 + 2 * sqrt 3. -/
theorem radius_large_circle (R : ℝ)
  (h1 : ∃ (C1 C2 C3 : ℝ × ℝ), 
    dist C1 C2 = 4 ∧ dist C2 C3 = 4 ∧ dist C3 C1 = 4 ∧ 
    (∃ (O : ℝ × ℝ), 
      (dist O C1 = R - 2) ∧ 
      (dist O C2 = R - 2) ∧ 
      (dist O C3 = R + 2) ∧ 
      (dist C1 C2 = 4) ∧ (dist C2 C3 = 4) ∧ (dist C3 C1 = 4))):
  R = 4 + 2 * Real.sqrt 3 := 
sorry

end radius_large_circle_l1170_117078


namespace Sam_has_walked_25_miles_l1170_117028

variables (d : ℕ) (v_fred v_sam : ℕ)

def Fred_and_Sam_meet (d : ℕ) (v_fred v_sam : ℕ) := 
  d / (v_fred + v_sam) * v_sam

theorem Sam_has_walked_25_miles :
  Fred_and_Sam_meet 50 5 5 = 25 :=
by
  sorry

end Sam_has_walked_25_miles_l1170_117028


namespace total_number_of_parts_l1170_117044

-- Identify all conditions in the problem: sample size and probability
def sample_size : ℕ := 30
def probability : ℝ := 0.25

-- Statement of the proof problem: The total number of parts N is 120 given the conditions
theorem total_number_of_parts (N : ℕ) (h : (sample_size : ℝ) / N = probability) : N = 120 :=
sorry

end total_number_of_parts_l1170_117044


namespace difference_of_numbers_l1170_117043

theorem difference_of_numbers 
  (L S : ℤ) (hL : L = 1636) (hdiv : L = 6 * S + 10) : 
  L - S = 1365 :=
sorry

end difference_of_numbers_l1170_117043


namespace Jeff_total_laps_l1170_117042

theorem Jeff_total_laps (laps_saturday : ℕ) (laps_sunday_morning : ℕ) (laps_remaining : ℕ)
  (h1 : laps_saturday = 27) (h2 : laps_sunday_morning = 15) (h3 : laps_remaining = 56) :
  (laps_saturday + laps_sunday_morning + laps_remaining) = 98 := 
by
  sorry

end Jeff_total_laps_l1170_117042


namespace sin_half_alpha_plus_beta_eq_sqrt2_div_2_l1170_117013

open Real

theorem sin_half_alpha_plus_beta_eq_sqrt2_div_2
  (α β : ℝ)
  (hα : α ∈ Set.Icc (π / 2) (3 * π / 2))
  (hβ : β ∈ Set.Icc (-π / 2) 0)
  (h1 : (α - π / 2)^3 - sin α - 2 = 0)
  (h2 : 8 * β^3 + 2 * (cos β)^2 + 1 = 0) :
  sin (α / 2 + β) = sqrt 2 / 2 := 
sorry

end sin_half_alpha_plus_beta_eq_sqrt2_div_2_l1170_117013


namespace rectangle_perimeter_of_right_triangle_l1170_117047

-- Define the conditions for the triangle and the rectangle
def rightTriangleArea (a b c : ℕ) (h : a^2 + b^2 = c^2) : ℕ :=
  (1 / 2) * a * b

def rectanglePerimeter (width area : ℕ) : ℕ :=
  2 * ((area / width) + width)

theorem rectangle_perimeter_of_right_triangle :
  ∀ (a b c width : ℕ) (h_a : a = 5) (h_b : b = 12) (h_c : c = 13)
    (h_pyth : a^2 + b^2 = c^2) (h_width : width = 5)
    (h_area_eq : rightTriangleArea a b c h_pyth = width * (rightTriangleArea a b c h_pyth / width)),
  rectanglePerimeter width (rightTriangleArea a b c h_pyth) = 22 :=
by
  intros
  sorry

end rectangle_perimeter_of_right_triangle_l1170_117047


namespace table_tennis_total_rounds_l1170_117038

-- Mathematical equivalent proof problem in Lean 4 statement
theorem table_tennis_total_rounds
  (A_played : ℕ) (B_played : ℕ) (C_referee : ℕ) (total_rounds : ℕ)
  (hA : A_played = 5) (hB : B_played = 4) (hC : C_referee = 2) :
  total_rounds = 7 :=
by
  -- Proof omitted
  sorry

end table_tennis_total_rounds_l1170_117038


namespace dimes_count_l1170_117061

-- Definitions of types of coins and their values in cents.
def penny := 1
def nickel := 5
def dime := 10
def quarter := 25
def halfDollar := 50

-- Condition statements as assumptions
variables (num_pennies num_nickels num_dimes num_quarters num_halfDollars : ℕ)

-- Sum of all coins and their values (in cents)
def total_value := num_pennies * penny + num_nickels * nickel + num_dimes * dime + num_quarters * quarter + num_halfDollars * halfDollar

-- Total number of coins
def total_coins := num_pennies + num_nickels + num_dimes + num_quarters + num_halfDollars

-- Proving the number of dimes is 5 given the conditions.
theorem dimes_count : 
  total_value = 163 ∧ 
  total_coins = 12 ∧ 
  num_pennies ≥ 1 ∧ 
  num_nickels ≥ 1 ∧ 
  num_dimes ≥ 1 ∧ 
  num_quarters ≥ 1 ∧ 
  num_halfDollars ≥ 1 → 
  num_dimes = 5 :=
by
  sorry

end dimes_count_l1170_117061


namespace find_ab_l1170_117082

-- Define the "¤" operation
def op (x y : ℝ) := (x + y)^2 - (x - y)^2

-- The Lean 4 theorem statement
theorem find_ab (a b : ℝ) (h : op a b = 24) : a * b = 6 := 
by
  -- We leave the proof as an exercise
  sorry

end find_ab_l1170_117082


namespace businessmen_drink_neither_l1170_117062

theorem businessmen_drink_neither : 
  ∀ (total coffee tea both : ℕ), 
    total = 30 → 
    coffee = 15 → 
    tea = 13 → 
    both = 8 → 
    total - (coffee - both + tea - both + both) = 10 := 
by 
  intros total coffee tea both h_total h_coffee h_tea h_both
  sorry

end businessmen_drink_neither_l1170_117062


namespace number_of_girls_l1170_117050

variable (b g d : ℕ)

-- Conditions
axiom boys_count : b = 1145
axiom difference : d = 510
axiom boys_equals_girls_plus_difference : b = g + d

-- Theorem to prove
theorem number_of_girls : g = 635 := by
  sorry

end number_of_girls_l1170_117050


namespace find_value_added_l1170_117056

open Classical

variable (n : ℕ) (avg_initial avg_final : ℝ)

-- Initial conditions
axiom avg_then_sum (n : ℕ) (avg : ℝ) : n * avg = 600

axiom avg_after_addition (n : ℕ) (avg : ℝ) : n * avg = 825

theorem find_value_added (n : ℕ) (avg_initial avg_final : ℝ) (h1 : n * avg_initial = 600) (h2 : n * avg_final = 825) :
  avg_final - avg_initial = 15 := by
  -- Proof goes here
  sorry

end find_value_added_l1170_117056


namespace find_x_l1170_117072

theorem find_x (x : ℝ) (hx1 : x > 0) 
  (h1 : 0.20 * x + 14 = (1 / 3) * ((3 / 4) * x + 21)) : x = 140 :=
sorry

end find_x_l1170_117072


namespace systematic_sampling_first_group_l1170_117063

theorem systematic_sampling_first_group (S : ℕ) (n : ℕ) (students_per_group : ℕ) (group_number : ℕ)
(h1 : n = 160)
(h2 : students_per_group = 8)
(h3 : group_number = 16)
(h4 : S + (group_number - 1) * students_per_group = 126)
: S = 6 := by
  sorry

end systematic_sampling_first_group_l1170_117063


namespace periodic_function_l1170_117006

variable {α : Type*} [AddGroup α] {f : α → α} {a b : α}

def symmetric_around (c : α) (f : α → α) : Prop := ∀ x, f (c - x) = f (c + x)

theorem periodic_function (h1 : symmetric_around a f) (h2 : symmetric_around b f) (h_ab : a ≠ b) : ∃ T, (∀ x, f (x + T) = f x) := 
sorry

end periodic_function_l1170_117006


namespace reporters_covering_local_politics_l1170_117049

theorem reporters_covering_local_politics (R : ℕ) (P Q A B : ℕ)
  (h1 : P = 70)
  (h2 : Q = 100 - P)
  (h3 : A = 40)
  (h4 : B = 100 - A) :
  B % 30 = 18 :=
by
  sorry

end reporters_covering_local_politics_l1170_117049


namespace fraction_of_salary_spent_on_house_rent_l1170_117070

theorem fraction_of_salary_spent_on_house_rent
    (S : ℕ) (H : ℚ)
    (cond1 : S = 180000)
    (cond2 : S / 5 + H * S + 3 * S / 5 + 18000 = S) :
    H = 1 / 10 := by
  sorry

end fraction_of_salary_spent_on_house_rent_l1170_117070


namespace shaded_triangle_ratio_is_correct_l1170_117010

noncomputable def ratio_of_shaded_triangle_to_large_square (total_area : ℝ) 
  (midpoint_area_ratio : ℝ := 1 / 24) : ℝ :=
  midpoint_area_ratio * total_area

theorem shaded_triangle_ratio_is_correct 
  (shaded_area total_area : ℝ)
  (n : ℕ)
  (h1 : n = 36)
  (grid_area : ℝ)
  (condition1 : grid_area = total_area / n)
  (condition2 : shaded_area = grid_area / 2 * 3)
  : shaded_area / total_area = 1 / 24 :=
by
  sorry

end shaded_triangle_ratio_is_correct_l1170_117010


namespace subset_contains_square_l1170_117039

theorem subset_contains_square {A : Finset ℕ} (hA₁ : A ⊆ Finset.range 101) (hA₂ : A.card = 50) (hA₃ : ∀ x ∈ A, ∀ y ∈ A, x + y ≠ 100) : 
  ∃ x ∈ A, ∃ k : ℕ, x = k^2 := 
sorry

end subset_contains_square_l1170_117039


namespace jesse_initial_blocks_l1170_117087

def total_blocks_initial (blocks_cityscape blocks_farmhouse blocks_zoo blocks_first_area blocks_second_area blocks_third_area blocks_left : ℕ) : ℕ :=
  blocks_cityscape + blocks_farmhouse + blocks_zoo + blocks_first_area + blocks_second_area + blocks_third_area + blocks_left

theorem jesse_initial_blocks :
  total_blocks_initial 80 123 95 57 43 62 84 = 544 :=
sorry

end jesse_initial_blocks_l1170_117087


namespace coordinates_at_5PM_l1170_117033

noncomputable def particle_coords_at_5PM : ℝ × ℝ :=
  let t1 : ℝ := 7  -- 7 AM
  let t2 : ℝ := 9  -- 9 AM
  let t3 : ℝ := 17  -- 5 PM in 24-hour format
  let coord1 : ℝ × ℝ := (1, 2)
  let coord2 : ℝ × ℝ := (3, -2)
  let dx : ℝ := (coord2.1 - coord1.1) / (t2 - t1)
  let dy : ℝ := (coord2.2 - coord1.2) / (t2 - t1)
  (coord2.1 + dx * (t3 - t2), coord2.2 + dy * (t3 - t2))

theorem coordinates_at_5PM
  (t1 t2 t3 : ℝ)
  (coord1 coord2 : ℝ × ℝ)
  (h_t1 : t1 = 7)
  (h_t2 : t2 = 9)
  (h_t3 : t3 = 17)
  (h_coord1 : coord1 = (1, 2))
  (h_coord2 : coord2 = (3, -2))
  (h_dx : (coord2.1 - coord1.1) / (t2 - t1) = 1)
  (h_dy : (coord2.2 - coord1.2) / (t2 - t1) = -2)
  : particle_coords_at_5PM = (11, -18) :=
by
  sorry

end coordinates_at_5PM_l1170_117033


namespace rhombus_diagonals_not_equal_l1170_117068

-- Define what a rhombus is
structure Rhombus where
  sides_equal : ∀ a b : ℝ, a = b  -- all sides are equal
  symmetrical : Prop -- it is a symmetrical figure
  centrally_symmetrical : Prop -- it is a centrally symmetrical figure

-- Theorem to state that the diagonals of a rhombus are not necessarily equal
theorem rhombus_diagonals_not_equal (r : Rhombus) : ¬(∀ a b : ℝ, a = b) := by
  sorry

end rhombus_diagonals_not_equal_l1170_117068


namespace eggs_used_to_bake_cake_l1170_117085

theorem eggs_used_to_bake_cake
    (initial_eggs : ℕ)
    (omelet_eggs : ℕ)
    (aunt_eggs : ℕ)
    (meal_eggs : ℕ)
    (num_meals : ℕ)
    (remaining_eggs_after_omelet : initial_eggs - omelet_eggs = 22)
    (eggs_given_to_aunt : 2 * aunt_eggs = initial_eggs - omelet_eggs)
    (remaining_eggs_after_aunt : initial_eggs - omelet_eggs - aunt_eggs = 11)
    (total_eggs_for_meals : meal_eggs * num_meals = 9)
    (remaining_eggs_after_meals : initial_eggs - omelet_eggs - aunt_eggs - meal_eggs * num_meals = 2) :
  initial_eggs - omelet_eggs - aunt_eggs - meal_eggs * num_meals = 2 :=
sorry

end eggs_used_to_bake_cake_l1170_117085


namespace find_number_of_boys_l1170_117048

noncomputable def number_of_boys (B G : ℕ) : Prop :=
  (B : ℚ) / (G : ℚ) = 7.5 / 15.4 ∧ G = B + 174

theorem find_number_of_boys : ∃ B G : ℕ, number_of_boys B G ∧ B = 165 := 
by 
  sorry

end find_number_of_boys_l1170_117048


namespace tan_cos_identity_15deg_l1170_117000

theorem tan_cos_identity_15deg :
  (1 - (Real.tan (Real.pi / 12))^2) * (Real.cos (Real.pi / 12))^2 = Real.sqrt 3 / 2 :=
by
  sorry

end tan_cos_identity_15deg_l1170_117000


namespace parallel_vectors_l1170_117073

theorem parallel_vectors (m : ℝ) : (m = 1) ↔ (∃ k : ℝ, (m, 1) = k • (1, m)) := sorry

end parallel_vectors_l1170_117073


namespace cows_sold_l1170_117003

/-- 
A man initially had 39 cows, 25 of them died last year, he sold some remaining cows, this year,
the number of cows increased by 24, he bought 43 more cows, his friend gave him 8 cows.
Now, he has 83 cows. How many cows did he sell last year?
-/
theorem cows_sold (S : ℕ) : (39 - 25 - S + 24 + 43 + 8 = 83) → S = 6 :=
by
  intro h
  sorry

end cows_sold_l1170_117003


namespace chess_team_selection_l1170_117027

theorem chess_team_selection
  (players : Finset ℕ) (twin1 twin2 : ℕ)
  (H1 : players.card = 10)
  (H2 : twin1 ∈ players)
  (H3 : twin2 ∈ players) :
  ∃ n : ℕ, n = 182 ∧ 
  (∃ team : Finset ℕ, team.card = 4 ∧
    (twin1 ∉ team ∨ twin2 ∉ team)) ∧
  n = (players.card.choose 4 - 
      ((players.erase twin1).erase twin2).card.choose 2) := sorry

end chess_team_selection_l1170_117027


namespace complex_div_eq_half_sub_half_i_l1170_117018

theorem complex_div_eq_half_sub_half_i (i : ℂ) (hi : i^2 = -1) : 
  (i^3 / (1 - i)) = (1 / 2) - (1 / 2) * i :=
by
  sorry

end complex_div_eq_half_sub_half_i_l1170_117018


namespace part1_solution_set_part2_range_a_l1170_117099

noncomputable def f (x a : ℝ) := 5 - abs (x + a) - abs (x - 2)

-- Part 1
theorem part1_solution_set (x : ℝ) (a : ℝ) (h : a = 1) :
  (f x a ≥ 0) ↔ (-2 ≤ x ∧ x ≤ 3) := sorry

-- Part 2
theorem part2_range_a (a : ℝ) :
  (∀ x, f x a ≤ 1) ↔ (a ≤ -6 ∨ a ≥ 2) := sorry

end part1_solution_set_part2_range_a_l1170_117099


namespace combined_weight_proof_l1170_117040

-- Definitions of atomic weights
def weight_C : ℝ := 12.01
def weight_H : ℝ := 1.01
def weight_O : ℝ := 16.00
def weight_S : ℝ := 32.07

-- Definitions of molar masses of compounds
def molar_mass_C6H8O7 : ℝ := (6 * weight_C) + (8 * weight_H) + (7 * weight_O)
def molar_mass_H2SO4 : ℝ := (2 * weight_H) + weight_S + (4 * weight_O)

-- Definitions of number of moles
def moles_C6H8O7 : ℝ := 8
def moles_H2SO4 : ℝ := 4

-- Combined weight
def combined_weight : ℝ := (moles_C6H8O7 * molar_mass_C6H8O7) + (moles_H2SO4 * molar_mass_H2SO4)

theorem combined_weight_proof : combined_weight = 1929.48 :=
by
  -- calculations as explained in the problem
  let wC6H8O7 := moles_C6H8O7 * molar_mass_C6H8O7
  let wH2SO4 := moles_H2SO4 * molar_mass_H2SO4
  have h1 : wC6H8O7 = 8 * 192.14 := by sorry
  have h2 : wH2SO4 = 4 * 98.09 := by sorry
  have h3 : combined_weight = wC6H8O7 + wH2SO4 := by simp [combined_weight, wC6H8O7, wH2SO4]
  rw [h3, h1, h2]
  simp
  sorry -- finish the proof as necessary

end combined_weight_proof_l1170_117040


namespace evaluate_expression_l1170_117093

theorem evaluate_expression : ∃ x : ℝ, (x = Real.sqrt (18 + x)) ∧ (x = (1 + Real.sqrt 73) / 2) := by
  sorry

end evaluate_expression_l1170_117093


namespace division_of_squares_l1170_117037

theorem division_of_squares {a b : ℕ} (h1 : a < 1000) (h2 : b > 0) (h3 : b^10 ∣ a^21) : b ∣ a^2 := 
sorry

end division_of_squares_l1170_117037


namespace simplify_fraction_l1170_117074

theorem simplify_fraction (a b gcd : ℕ) (h1 : a = 72) (h2 : b = 108) (h3 : gcd = Nat.gcd a b) : (a / gcd) / (b / gcd) = 2 / 3 :=
by
  -- the proof is omitted here
  sorry

end simplify_fraction_l1170_117074


namespace remainder_of_876539_div_7_l1170_117069

theorem remainder_of_876539_div_7 : 876539 % 7 = 6 :=
by
  sorry

end remainder_of_876539_div_7_l1170_117069


namespace evaluate_expression_l1170_117019

theorem evaluate_expression : 3 - 5 * (2^3 + 3) * 2 = -107 := by
  sorry

end evaluate_expression_l1170_117019
