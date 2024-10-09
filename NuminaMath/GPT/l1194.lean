import Mathlib

namespace average_of_roots_l1194_119447

theorem average_of_roots (p q : ℝ) (h : ∃ x1 x2 : ℝ, 3*p*x1^2 - 6*p*x1 + q = 0 ∧ 3*p*x2^2 - 6*p*x2 + q = 0 ∧ x1 ≠ x2):
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 3*p*x1^2 - 6*p*x1 + q = 0 ∧ 3*p*x2^2 - 6*p*x2 + q = 0) → 
  (x1 + x2) / 2 = 1 :=
by
  sorry

end average_of_roots_l1194_119447


namespace area_of_concentric_ring_l1194_119418

theorem area_of_concentric_ring (r_large : ℝ) (r_small : ℝ) 
  (h1 : r_large = 10) 
  (h2 : r_small = 6) : 
  (π * r_large^2 - π * r_small^2) = 64 * π :=
by {
  sorry
}

end area_of_concentric_ring_l1194_119418


namespace divide_one_meter_into_100_parts_l1194_119492

theorem divide_one_meter_into_100_parts :
  (1 / 100 : ℝ) = 1 / 100 := 
by
  sorry

end divide_one_meter_into_100_parts_l1194_119492


namespace monotone_increasing_interval_l1194_119477

noncomputable def f (x : ℝ) : ℝ := (x / (x^2 + 1)) + 1

theorem monotone_increasing_interval :
  ∀ x : ℝ, (-1 < x ∧ x < 1) ↔ ∀ ε > 0, ∃ δ > 0, ∀ x₁ x₂, (-1 < x₁ ∧ x₁ < 1 ∧ -1 < x₂ ∧ x₂ < 1 ∧ |x₁ - x₂| < δ) → f x₁ ≤ f x₂ + ε := 
sorry

end monotone_increasing_interval_l1194_119477


namespace sum_of_extreme_values_of_g_l1194_119475

def g (x : ℝ) : ℝ := abs (x - 1) + abs (x - 5) - 2 * abs (x - 3)

theorem sum_of_extreme_values_of_g :
  ∃ (min_val max_val : ℝ), 
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 6 → g x ≥ min_val) ∧ 
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 6 → g x ≤ max_val) ∧ 
    (min_val = -8) ∧ 
    (max_val = 0) ∧ 
    (min_val + max_val = -8) := 
by
  sorry

end sum_of_extreme_values_of_g_l1194_119475


namespace sequence_geometric_progression_iff_b1_eq_b2_l1194_119422

theorem sequence_geometric_progression_iff_b1_eq_b2 
  (b : ℕ → ℝ) 
  (h0 : ∀ n, b n > 0)
  (h1 : ∀ n, b (n + 2) = 3 * b n * b (n + 1)) :
  (∃ r : ℝ, ∀ n, b (n + 1) = r * b n) ↔ b 1 = b 0 :=
sorry

end sequence_geometric_progression_iff_b1_eq_b2_l1194_119422


namespace abs_neg_two_equals_two_l1194_119444

theorem abs_neg_two_equals_two : |(-2 : ℤ)| = 2 := 
by 
  sorry

end abs_neg_two_equals_two_l1194_119444


namespace shaded_area_between_circles_l1194_119458

theorem shaded_area_between_circles (r1 r2 : ℝ) (h1 : r1 = 4) (h2 : r2 = 5)
  (tangent : True) -- This represents that the circles are externally tangent
  (circumscribed : True) -- This represents the third circle circumscribing the two circles
  : ∃ r3 : ℝ, r3 = 9 ∧ π * r3^2 - (π * r1^2 + π * r2^2) = 40 * π :=
  sorry

end shaded_area_between_circles_l1194_119458


namespace quadratic_function_order_l1194_119421

theorem quadratic_function_order (a b c : ℝ) (h_neg_a : a < 0) 
  (h_sym : ∀ x, (a * (x + 2)^2 + b * (x + 2) + c) = (a * (2 - x)^2 + b * (2 - x) + c)) :
  (a * (-1992)^2 + b * (-1992) + c) < (a * (1992)^2 + b * (1992) + c) ∧
  (a * (1992)^2 + b * (1992) + c) < (a * (0)^2 + b * (0) + c) :=
by
  sorry

end quadratic_function_order_l1194_119421


namespace graph_of_4x2_minus_9y2_is_pair_of_straight_lines_l1194_119468

theorem graph_of_4x2_minus_9y2_is_pair_of_straight_lines :
  (∀ x y : ℝ, (4 * x^2 - 9 * y^2 = 0) → (x / y = 3 / 2 ∨ x / y = -3 / 2)) :=
by
  sorry

end graph_of_4x2_minus_9y2_is_pair_of_straight_lines_l1194_119468


namespace train_speed_proof_l1194_119439

theorem train_speed_proof : 
  ∀ (V_A V_B : ℝ) (T_A T_B : ℝ), 
  T_A = 9 ∧ 
  T_B = 4 ∧ 
  V_B = 90 ∧ 
  (V_A / V_B = T_B / T_A) → 
  V_A = 40 := 
by
  intros V_A V_B T_A T_B h
  obtain ⟨hT_A, hT_B, hV_B, hprop⟩ := h
  sorry

end train_speed_proof_l1194_119439


namespace intersection_reciprocal_sum_l1194_119481

open Real

theorem intersection_reciprocal_sum :
    ∀ (a b : ℝ),
    (∃ x : ℝ, x - 1 = a ∧ 3 / x = b) ∧
    (a * b = 3) →
    ∃ s : ℝ, (s = (a + b) / 3 ∨ s = -(a + b) / 3) ∧ (1 / a + 1 / b = s) := by
  sorry

end intersection_reciprocal_sum_l1194_119481


namespace wall_building_time_l1194_119485

theorem wall_building_time (n t : ℕ) (h1 : n * t = 48) (h2 : n = 4) : t = 12 :=
by
  -- appropriate proof steps would go here
  sorry

end wall_building_time_l1194_119485


namespace find_a_b_l1194_119438

theorem find_a_b (a b : ℝ)
  (h1 : a < 0)
  (h2 : (-b / a) = -((1 / 2) - (1 / 3)))
  (h3 : (2 / a) = -((1 / 2) * (1 / 3))) : 
  a + b = -14 :=
sorry

end find_a_b_l1194_119438


namespace independent_and_dependent_variables_l1194_119442

variable (R V : ℝ)

theorem independent_and_dependent_variables (h : V = (4 / 3) * Real.pi * R^3) :
  (∃ R : ℝ, ∀ V : ℝ, V = (4 / 3) * Real.pi * R^3) ∧ (∃ V : ℝ, ∃ R' : ℝ, V = (4 / 3) * Real.pi * R'^3) :=
by
  sorry

end independent_and_dependent_variables_l1194_119442


namespace numberOfZeros_l1194_119470

noncomputable def g (x : ℝ) : ℝ := Real.sin (Real.log x)

theorem numberOfZeros :
  ∃ x ∈ Set.Ioo 1 (Real.exp Real.pi), g x = 0 ∧ ∀ y ∈ Set.Ioo 1 (Real.exp Real.pi), g y = 0 → y = x := 
sorry

end numberOfZeros_l1194_119470


namespace sum_of_arithmetic_series_51_to_100_l1194_119454

theorem sum_of_arithmetic_series_51_to_100 :
  let first_term := 51
  let last_term := 100
  let n := (last_term - first_term) + 1
  2 * (n / 2) * (first_term + last_term) / 2 = 3775 :=
by
  sorry

end sum_of_arithmetic_series_51_to_100_l1194_119454


namespace percentage_born_in_july_l1194_119414

def total_scientists : ℕ := 150
def scientists_born_in_july : ℕ := 15

theorem percentage_born_in_july : (scientists_born_in_july * 100 / total_scientists) = 10 := by
  sorry

end percentage_born_in_july_l1194_119414


namespace company_employee_count_l1194_119462

theorem company_employee_count (E : ℝ) (H1 : E > 0) (H2 : 0.60 * E = 0.55 * (E + 30)) : E + 30 = 360 :=
by
  -- The proof steps would go here, but that is not required.
  sorry

end company_employee_count_l1194_119462


namespace refrigerator_volume_unit_l1194_119437

theorem refrigerator_volume_unit (V : ℝ) (u : String) : 
  V = 150 → (u = "Liters" ∨ u = "Milliliters" ∨ u = "Cubic meters") → 
  u = "Liters" :=
by
  intro hV hu
  sorry

end refrigerator_volume_unit_l1194_119437


namespace total_snails_and_frogs_l1194_119419

-- Define the number of snails and frogs in the conditions.
def snails : Nat := 5
def frogs : Nat := 2

-- State the problem: proving that the total number of snails and frogs equals 7.
theorem total_snails_and_frogs : snails + frogs = 7 := by
  -- Proof is omitted as the user requested only the statement.
  sorry

end total_snails_and_frogs_l1194_119419


namespace solution_unique_2014_l1194_119423

theorem solution_unique_2014 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (2 * x - 2 * y + 1 / z = 1 / 2014) ∧
  (2 * y - 2 * z + 1 / x = 1 / 2014) ∧
  (2 * z - 2 * x + 1 / y = 1 / 2014) →
  x = 2014 ∧ y = 2014 ∧ z = 2014 :=
by
  sorry

end solution_unique_2014_l1194_119423


namespace inf_solutions_l1194_119487

theorem inf_solutions (x y z : ℤ) : 
  ∃ (infinitely many relatively prime solutions : ℕ), x^2 + y^2 = z^5 + z :=
sorry

end inf_solutions_l1194_119487


namespace coordinates_of_F_double_prime_l1194_119486

-- Definitions of transformations
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Definition of initial point F
def F : ℝ × ℝ := (1, 1)

-- Definition of the transformations applied to point F
def F_prime : ℝ × ℝ := reflect_x F
def F_double_prime : ℝ × ℝ := reflect_y_eq_x F_prime

-- Theorem statement
theorem coordinates_of_F_double_prime : F_double_prime = (-1, 1) :=
by
  sorry

end coordinates_of_F_double_prime_l1194_119486


namespace bags_needed_l1194_119426

theorem bags_needed (expected_people extra_people extravagant_bags average_bags : ℕ) 
    (h1 : expected_people = 50) 
    (h2 : extra_people = 40) 
    (h3 : extravagant_bags = 10) 
    (h4 : average_bags = 20) : 
    (expected_people + extra_people - (extravagant_bags + average_bags) = 60) :=
by {
  sorry
}

end bags_needed_l1194_119426


namespace daves_initial_apps_l1194_119401

theorem daves_initial_apps : ∃ (X : ℕ), X + 11 - 17 = 4 ∧ X = 10 :=
by {
  sorry
}

end daves_initial_apps_l1194_119401


namespace circle_area_l1194_119420

open Real

noncomputable def radius_square (x : ℝ) (DE : ℝ) (EF : ℝ) : ℝ :=
  let DE_square := DE^2
  let r_square_1 := x^2 + DE_square
  let product_DE_EF := DE * EF
  let r_square_2 := product_DE_EF + x^2
  r_square_2

theorem circle_area (x : ℝ) (h1 : OE = x) (h2 : DE = 8) (h3 : EF = 4) :
  π * radius_square x 8 4 = 96 * π :=
by
  sorry

end circle_area_l1194_119420


namespace arithmetic_sequence_30th_term_l1194_119490

theorem arithmetic_sequence_30th_term (a1 a2 a3 d a30 : ℤ) 
 (h1 : a1 = 3) (h2 : a2 = 12) (h3 : a3 = 21) 
 (h4 : d = a2 - a1) (h5 : a3 = a1 + 2 * d) 
 (h6 : a30 = a1 + 29 * d) : 
 a30 = 264 :=
by
  sorry

end arithmetic_sequence_30th_term_l1194_119490


namespace main_problem_l1194_119412

-- Define the set A
def A (a : ℝ) : Set ℝ :=
  {0, 1, a^2 - 2 * a}

-- Define the main problem as a theorem
theorem main_problem (a : ℝ) (h : a ∈ A a) : a = 1 ∨ a = 3 :=
  sorry

end main_problem_l1194_119412


namespace regression_slope_interpretation_l1194_119482

-- Define the variables and their meanings
variable {x y : ℝ}

-- Define the regression line equation
def regression_line (x : ℝ) : ℝ := 0.8 * x + 4.6

-- Define the proof statement
theorem regression_slope_interpretation (hx : ∀ x, y = regression_line x) :
  ∀ delta_x : ℝ, delta_x = 1 → (regression_line (x + delta_x) - regression_line x) = 0.8 :=
by
  intros delta_x h_delta_x
  rw [h_delta_x, regression_line, regression_line]
  simp
  sorry

end regression_slope_interpretation_l1194_119482


namespace tony_schooling_years_l1194_119483

theorem tony_schooling_years:
  let first_degree := 4
  let additional_degrees := 2 * 4
  let graduate_degree := 2
  first_degree + additional_degrees + graduate_degree = 14 :=
by {
  let first_degree := 4
  let additional_degrees := 2 * 4
  let graduate_degree := 2
  show first_degree + additional_degrees + graduate_degree = 14
  sorry
}

end tony_schooling_years_l1194_119483


namespace transaction_gain_per_year_l1194_119467

theorem transaction_gain_per_year
  (principal : ℝ) (borrow_rate : ℝ) (lend_rate : ℝ) (time : ℕ)
  (principal_eq : principal = 5000)
  (borrow_rate_eq : borrow_rate = 0.04)
  (lend_rate_eq : lend_rate = 0.06)
  (time_eq : time = 2) :
  (principal * lend_rate * time - principal * borrow_rate * time) / time = 100 := by
  sorry

end transaction_gain_per_year_l1194_119467


namespace evaluate_expression_l1194_119434

theorem evaluate_expression :
  ((3.5 / 0.7) * (5 / 3) + (7.2 / 0.36) - ((5 / 3) * (0.75 / 0.25))) = 23.3335 :=
by
  sorry

end evaluate_expression_l1194_119434


namespace find_y_intercept_of_second_parabola_l1194_119489

theorem find_y_intercept_of_second_parabola :
  ∃ D : ℝ × ℝ, D = (0, 9) ∧ 
    (∃ A : ℝ × ℝ, A = (10, 4) ∧ 
     ∃ B : ℝ × ℝ, B = (6, 0) ∧ 
     (∀ x y : ℝ, y = (-1/4) * x ^ 2 + 5 * x - 21 → A = (10, 4)) ∧ 
     (∀ x y : ℝ, y = (1/4) * (x - B.1) ^ 2 + B.2 ∧ y = 4 ∧ B = (6, 0) → A = (10, 4))) :=
  sorry

end find_y_intercept_of_second_parabola_l1194_119489


namespace ratio_Raphael_to_Manny_l1194_119406

-- Define the pieces of lasagna each person will eat
def Manny_pieces : ℕ := 1
def Kai_pieces : ℕ := 2
def Lisa_pieces : ℕ := 2
def Aaron_pieces : ℕ := 0
def Total_pieces : ℕ := 6

-- Calculate the remaining pieces for Raphael
def Raphael_pieces : ℕ := Total_pieces - (Manny_pieces + Kai_pieces + Lisa_pieces + Aaron_pieces)

-- Prove that the ratio of Raphael's pieces to Manny's pieces is 1:1
theorem ratio_Raphael_to_Manny : Raphael_pieces = Manny_pieces :=
by
  -- Provide the actual proof logic, but currently leaving it as a placeholder
  sorry

end ratio_Raphael_to_Manny_l1194_119406


namespace problem1_problem2_l1194_119461

-- Problem 1 Lean Statement
theorem problem1 (m n : ℕ) (h1 : 3 ^ m = 6) (h2 : 9 ^ n = 2) : 3 ^ (m - 2 * n) = 3 :=
by
  sorry

-- Problem 2 Lean Statement
theorem problem2 (x : ℝ) (n : ℕ) (h : x ^ (2 * n) = 3) : (x ^ (3 * n)) ^ 2 - (x ^ 2) ^ (2 * n) = 18 :=
by
  sorry

end problem1_problem2_l1194_119461


namespace units_digit_sum_even_20_to_80_l1194_119452

theorem units_digit_sum_even_20_to_80 :
  let a := 20
  let d := 2
  let l := 80
  let n := ((l - a) / d) + 1 -- Given by the formula l = a + (n-1)d => n = (l - a) / d + 1
  let sum := (n * (a + l)) / 2
  (sum % 10) = 0 := sorry

end units_digit_sum_even_20_to_80_l1194_119452


namespace leak_empties_tank_in_10_hours_l1194_119407

theorem leak_empties_tank_in_10_hours :
  (∀ (A L : ℝ), (A = 1/5) → (A - L = 1/10) → (1 / L = 10)) 
  := by
  intros A L hA hAL
  sorry

end leak_empties_tank_in_10_hours_l1194_119407


namespace triangle_area_formula_l1194_119472

theorem triangle_area_formula (a b c R : ℝ) (α β γ : ℝ) 
    (h1 : a / (Real.sin α) = 2 * R) 
    (h2 : b / (Real.sin β) = 2 * R) 
    (h3 : c / (Real.sin γ) = 2 * R) :
    let S := (1 / 2) * a * b * (Real.sin γ)
    S = a * b * c / (4 * R) := 
by 
  sorry

end triangle_area_formula_l1194_119472


namespace cosine_135_eq_neg_sqrt_2_div_2_l1194_119408

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l1194_119408


namespace evaluate_root_power_l1194_119431

theorem evaluate_root_power : (Real.sqrt (Real.sqrt 9))^12 = 729 := 
by sorry

end evaluate_root_power_l1194_119431


namespace option_A_is_incorrect_option_B_is_incorrect_option_C_is_correct_option_D_is_incorrect_l1194_119402

variable (x y: ℝ)

theorem option_A_is_incorrect : 5 - 3 * (x + 1) ≠ 5 - 3 * x - 1 := 
by sorry

theorem option_B_is_incorrect : 2 - 4 * (x + 1/4) ≠ 2 - 4 * x + 1 := 
by sorry

theorem option_C_is_correct : 2 - 4 * (1/4 * x + 1) = 2 - x - 4 := 
by sorry

theorem option_D_is_incorrect : 2 * (x - 2) - 3 * (y - 1) ≠ 2 * x - 4 - 3 * y - 3 := 
by sorry

end option_A_is_incorrect_option_B_is_incorrect_option_C_is_correct_option_D_is_incorrect_l1194_119402


namespace fraction_of_automobile_installment_credit_extended_by_finance_companies_l1194_119417

theorem fraction_of_automobile_installment_credit_extended_by_finance_companies
  (total_consumer_credit : ℝ)
  (percentage_auto_credit : ℝ)
  (credit_extended_by_finance_companies : ℝ)
  (total_auto_credit_fraction : percentage_auto_credit = 0.36)
  (total_consumer_credit_value : total_consumer_credit = 475)
  (credit_extended_by_finance_companies_value : credit_extended_by_finance_companies = 57) :
  credit_extended_by_finance_companies / (percentage_auto_credit * total_consumer_credit) = 1 / 3 :=
by
  -- The proof part will go here.
  sorry

end fraction_of_automobile_installment_credit_extended_by_finance_companies_l1194_119417


namespace arithmetic_sequence_26th_term_l1194_119480

theorem arithmetic_sequence_26th_term (a d : ℤ) (h1 : a = 3) (h2 : a + d = 13) (h3 : a + 2 * d = 23) : 
  a + 25 * d = 253 :=
by
  -- specifications for variables a, d, and hypotheses h1, h2, h3
  sorry

end arithmetic_sequence_26th_term_l1194_119480


namespace solution_to_inequality_l1194_119443

theorem solution_to_inequality :
  { x : ℝ | ((x^2 - 1) / (x - 4)^2) ≥ 0 } = { x : ℝ | x ≤ -1 ∨ (1 ≤ x ∧ x < 4) ∨ x > 4 } := 
sorry

end solution_to_inequality_l1194_119443


namespace probability_odd_sum_l1194_119450

-- Definitions based on the conditions
def cards : List ℕ := [1, 2, 3, 4, 5]

def is_odd_sum (a b : ℕ) : Prop := (a + b) % 2 = 1

def combinations (n k : ℕ) : ℕ := (Nat.choose n k)

-- Main statement
theorem probability_odd_sum :
  (combinations 5 2) = 10 → -- Total combinations of 2 cards from 5
  (∃ N, N = 6 ∧ (N:ℚ)/(combinations 5 2) = 3/5) :=
by 
  sorry

end probability_odd_sum_l1194_119450


namespace min_abs_difference_on_hyperbola_l1194_119456

theorem min_abs_difference_on_hyperbola : 
  ∀ (x y : ℝ), (x^2 / 8 - y^2 / 4 = 1) → abs (x - y) ≥ 2 := 
by
  intros x y hxy
  sorry

end min_abs_difference_on_hyperbola_l1194_119456


namespace yellow_balls_in_bag_l1194_119411

theorem yellow_balls_in_bag (y : ℕ) (r : ℕ) (P_red : ℚ) (h_r : r = 8) (h_P_red : P_red = 1 / 3) 
  (h_prob : P_red = r / (r + y)) : y = 16 :=
by
  sorry

end yellow_balls_in_bag_l1194_119411


namespace marshmallow_total_l1194_119410

-- Define the number of marshmallows each kid can hold
def Haley := 8
def Michael := 3 * Haley
def Brandon := Michael / 2

-- Prove the total number of marshmallows held by all three is 44
theorem marshmallow_total : Haley + Michael + Brandon = 44 := by
  sorry

end marshmallow_total_l1194_119410


namespace train_speed_kmph_l1194_119427

/-- Given that the length of the train is 200 meters and it crosses a pole in 9 seconds,
the speed of the train in km/hr is 80. -/
theorem train_speed_kmph (length : ℝ) (time : ℝ) (length_eq : length = 200) (time_eq : time = 9) : 
  (length / time) * (3600 / 1000) = 80 :=
by
  sorry

end train_speed_kmph_l1194_119427


namespace license_plate_combinations_l1194_119496

-- Definition for the conditions of the problem
def num_license_plate_combinations : ℕ :=
  let num_letters := 26
  let num_digits := 10
  let choose_two_distinct_letters := (num_letters * (num_letters - 1)) / 2
  let arrange_pairs := 2
  let choose_positions := 6
  let digit_permutations := num_digits ^ 2
  choose_two_distinct_letters * arrange_pairs * choose_positions * digit_permutations

-- The theorem we are proving
theorem license_plate_combinations :
  num_license_plate_combinations = 390000 :=
by
  -- The proof would be provided here.
  sorry

end license_plate_combinations_l1194_119496


namespace carmen_candles_needed_l1194_119449

-- Definitions based on the conditions

def candle_lifespan_1_hour : Nat := 8  -- a candle lasts 8 nights when burned 1 hour each night
def nights_total : Nat := 24  -- total nights

-- Question: How many candles are needed if burned 2 hours a night?

theorem carmen_candles_needed (h : candle_lifespan_1_hour / 2 = 4) :
  nights_total / 4 = 6 := 
  sorry

end carmen_candles_needed_l1194_119449


namespace trajectory_of_P_l1194_119413
-- Import entire library for necessary definitions and theorems.

-- Define the properties of the conic sections.
def ellipse (x y : ℝ) (n : ℝ) : Prop :=
  x^2 / 4 + y^2 / n = 1

def hyperbola (x y : ℝ) (m : ℝ) : Prop :=
  x^2 / 8 - y^2 / m = 1

-- Define the condition where the conic sections share the same foci.
def shared_foci (n m : ℝ) : Prop :=
  4 - n = 8 + m

-- The main theorem stating the relationship between m and n forming a straight line.
theorem trajectory_of_P : ∀ (n m : ℝ), shared_foci n m → (m + n + 4 = 0) :=
by
  intros n m h
  sorry

end trajectory_of_P_l1194_119413


namespace pet_center_final_count_l1194_119499

/-!
# Problem: Count the total number of pets in a pet center after a series of adoption and collection events.
-/

def initialDogs : Nat := 36
def initialCats : Nat := 29
def initialRabbits : Nat := 15
def initialBirds : Nat := 10

def dogsAdopted1 : Nat := 20
def rabbitsAdopted1 : Nat := 5

def catsCollected : Nat := 12
def rabbitsCollected : Nat := 8
def birdsCollected : Nat := 5

def catsAdopted2 : Nat := 10
def birdsAdopted2 : Nat := 4

def finalDogs : Nat :=
  initialDogs - dogsAdopted1

def finalCats : Nat :=
  initialCats + catsCollected - catsAdopted2

def finalRabbits : Nat :=
  initialRabbits - rabbitsAdopted1 + rabbitsCollected

def finalBirds : Nat :=
  initialBirds + birdsCollected - birdsAdopted2

def totalPets (d c r b : Nat) : Nat :=
  d + c + r + b

theorem pet_center_final_count : 
  totalPets finalDogs finalCats finalRabbits finalBirds = 76 := by
  -- This is where we would provide the proof, but it's skipped as per the instructions.
  sorry

end pet_center_final_count_l1194_119499


namespace uki_total_earnings_l1194_119404

def cupcake_price : ℝ := 1.50
def cookie_price : ℝ := 2.00
def biscuit_price : ℝ := 1.00
def daily_cupcakes : ℕ := 20
def daily_cookies : ℕ := 10
def daily_biscuits : ℕ := 20
def days : ℕ := 5

theorem uki_total_earnings :
  5 * ((daily_cupcakes * cupcake_price) + (daily_cookies * cookie_price) + (daily_biscuits * biscuit_price)) = 350 :=
by
  -- This is a placeholder for the proof
  sorry

end uki_total_earnings_l1194_119404


namespace B_subset_A_implies_m_values_l1194_119497

noncomputable def A : Set ℝ := { x | x^2 + x - 6 = 0 }
noncomputable def B (m : ℝ) : Set ℝ := { x | m * x + 1 = 0 }
def possible_m_values : Set ℝ := {1/3, -1/2}

theorem B_subset_A_implies_m_values (m : ℝ) : B m ⊆ A → m ∈ possible_m_values := by
  sorry

end B_subset_A_implies_m_values_l1194_119497


namespace magnitude_of_b_is_5_l1194_119416

variable (a b : ℝ × ℝ)
variable (h_a : a = (3, -2))
variable (h_ab : a + b = (0, 2))

theorem magnitude_of_b_is_5 : ‖b‖ = 5 :=
by
  sorry

end magnitude_of_b_is_5_l1194_119416


namespace equal_areas_triangle_height_l1194_119451

theorem equal_areas_triangle_height (l b h : ℝ) (hlb : l > b) 
  (H1 : l * b = (1/2) * l * h) : h = 2 * b :=
by 
  -- skipping proof
  sorry

end equal_areas_triangle_height_l1194_119451


namespace opposite_of_neg2023_l1194_119453

def opposite (x : Int) := -x

theorem opposite_of_neg2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg2023_l1194_119453


namespace possible_values_x_l1194_119474

-- Define the conditions
def gold_coin_worth (x y : ℕ) (g s : ℝ) : Prop :=
  g = (1 + x / 100.0) * s ∧ s = (1 - y / 100.0) * g

-- Define the main theorem statement
theorem possible_values_x : ∀ (x y : ℕ) (g s : ℝ), gold_coin_worth x y g s → 
  (∃ (n : ℕ), n = 12) :=
by
  -- Definitions based on given conditions
  intro x y g s h
  obtain ⟨hx, hy⟩ := h

  -- Placeholder for proof; skip with sorry
  sorry

end possible_values_x_l1194_119474


namespace rhombus_area_l1194_119403

noncomputable def area_of_rhombus (d1 d2 : ℝ) : ℝ :=
  0.5 * d1 * d2

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 3) (h2 : d2 = 4) : area_of_rhombus d1 d2 = 6 :=
by
  sorry

end rhombus_area_l1194_119403


namespace percentage_not_pens_pencils_erasers_l1194_119494

-- Define the given percentages
def percentPens : ℝ := 42
def percentPencils : ℝ := 25
def percentErasers : ℝ := 12
def totalPercent : ℝ := 100

-- The goal is to prove that the percentage of sales that were not pens, pencils, or erasers is 21%
theorem percentage_not_pens_pencils_erasers :
  totalPercent - (percentPens + percentPencils + percentErasers) = 21 := by
  sorry

end percentage_not_pens_pencils_erasers_l1194_119494


namespace entree_cost_l1194_119465

theorem entree_cost (E : ℝ) :
  let appetizer := 9
  let dessert := 11
  let tip_rate := 0.30
  let total_cost_with_tip := 78
  let total_cost_before_tip := appetizer + 2 * E + dessert
  total_cost_with_tip = total_cost_before_tip + (total_cost_before_tip * tip_rate) →
  E = 20 :=
by
  intros appetizer dessert tip_rate total_cost_with_tip total_cost_before_tip h
  sorry

end entree_cost_l1194_119465


namespace ordered_pair_for_quadratic_with_same_roots_l1194_119460

theorem ordered_pair_for_quadratic_with_same_roots (b c : ℝ) :
  (∀ x : ℝ, |x - 4| = 3 ↔ (x = 7 ∨ x = 1)) →
  (∀ x : ℝ, x^2 + b * x + c = 0 ↔ (x = 7 ∨ x = 1)) →
  (b, c) = (-8, 7) :=
by
  intro h1 h2
  sorry

end ordered_pair_for_quadratic_with_same_roots_l1194_119460


namespace greatest_number_of_bouquets_l1194_119495

def cherry_lollipops := 4
def orange_lollipops := 6
def raspberry_lollipops := 8
def lemon_lollipops := 10
def candy_canes := 12
def chocolate_coins := 14

theorem greatest_number_of_bouquets : 
  Nat.gcd cherry_lollipops (Nat.gcd orange_lollipops (Nat.gcd raspberry_lollipops (Nat.gcd lemon_lollipops (Nat.gcd candy_canes chocolate_coins)))) = 2 := 
by 
  sorry

end greatest_number_of_bouquets_l1194_119495


namespace exists_integers_for_S_geq_100_l1194_119498

theorem exists_integers_for_S_geq_100 (S : ℤ) (hS : S ≥ 100) :
  ∃ (T C B : ℤ) (P : ℤ),
    T > 0 ∧ C > 0 ∧ B > 0 ∧
    T > C ∧ C > B ∧
    T + C + B = S ∧
    T * C * B = P ∧
    (∀ (T₁ C₁ B₁ T₂ C₂ B₂ : ℤ), 
      T₁ > 0 ∧ C₁ > 0 ∧ B₁ > 0 ∧ 
      T₂ > 0 ∧ C₂ > 0 ∧ B₂ > 0 ∧ 
      T₁ > C₁ ∧ C₁ > B₁ ∧ 
      T₂ > C₂ ∧ C₂ > B₂ ∧ 
      T₁ + C₁ + B₁ = S ∧ 
      T₂ + C₂ + B₂ = S ∧ 
      T₁ * C₁ * B₁ = T₂ * C₂ * B₂ → 
      (T₁ = T₂) ∧ (C₁ = C₂) ∧ (B₁ = B₂) → false) :=
sorry

end exists_integers_for_S_geq_100_l1194_119498


namespace prove_market_demand_prove_tax_revenue_prove_per_unit_tax_rate_prove_tax_revenue_specified_l1194_119476

noncomputable def market_supply_function (P : ℝ) : ℝ := 6 * P - 312

noncomputable def market_demand_function (a b P : ℝ) : ℝ := a - b * P

noncomputable def price_elasticity_supply (P_e Q_e : ℝ) : ℝ := 6 * (P_e / Q_e)

noncomputable def price_elasticity_demand (b P_e Q_e : ℝ) : ℝ := -b * (P_e / Q_e)

noncomputable def tax_rate := 30

noncomputable def consumer_price_after_tax := 118

theorem prove_market_demand (a P_e Q_e : ℝ) :
  1.5 * |price_elasticity_demand 4 P_e Q_e| = price_elasticity_supply P_e Q_e →
  market_demand_function a 4 P_e = a - 4 * P_e := sorry

theorem prove_tax_revenue (Q_d : ℝ) :
  Q_d = 216 →
  Q_d * tax_rate = 6480 := sorry

theorem prove_per_unit_tax_rate (t : ℝ) :
  t = 60 → 4 * t = 240 := sorry

theorem prove_tax_revenue_specified (t : ℝ) :
  t = 60 →
  (288 * t - 2.4 * t^2) = 8640 := sorry

end prove_market_demand_prove_tax_revenue_prove_per_unit_tax_rate_prove_tax_revenue_specified_l1194_119476


namespace sequence_relation_l1194_119440

theorem sequence_relation (b : ℕ → ℚ) : 
  b 1 = 2 ∧ b 2 = 5 / 11 ∧ (∀ n ≥ 3, b n = b (n-2) * b (n-1) / (3 * b (n-2) - b (n-1)))
  ↔ b 2023 = 5 / 12137 :=
by sorry

end sequence_relation_l1194_119440


namespace total_calories_box_l1194_119464

-- Definitions from the conditions
def bags := 6
def cookies_per_bag := 25
def calories_per_cookie := 18

-- Given the conditions, prove the total calories equals 2700
theorem total_calories_box : bags * cookies_per_bag * calories_per_cookie = 2700 := by
  sorry

end total_calories_box_l1194_119464


namespace intersection_complement_l1194_119488

open Set

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hA : A = {1, 3, 4})
variable (hB : B = {4, 5})

theorem intersection_complement :
  A ∩ (U \ B) = {1, 3} :=
by
  rw [hU, hA, hB]
  ext
  simp
  sorry

end intersection_complement_l1194_119488


namespace solve_problem_l1194_119433

noncomputable def problem_statement : ℤ :=
  (-3)^6 / 3^4 - 4^3 * 2^2 + 9^2

theorem solve_problem : problem_statement = -166 :=
by 
  -- Proof omitted
  sorry

end solve_problem_l1194_119433


namespace cannot_reach_eighth_vertex_l1194_119491

def Point := ℕ × ℕ × ℕ

def symmetry (p1 p2 : Point) : Point :=
  let (a, b, c) := p1
  let (a', b', c') := p2
  (2 * a' - a, 2 * b' - b, 2 * c' - c)

def vertices : List Point :=
  [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

theorem cannot_reach_eighth_vertex : ∀ (p : Point), p ∈ vertices → ∀ (q : Point), q ∈ vertices → 
  ¬(symmetry p q = (1, 1, 1)) :=
by
  sorry

end cannot_reach_eighth_vertex_l1194_119491


namespace expanded_form_correct_l1194_119478

theorem expanded_form_correct :
  (∃ a b c : ℤ, (∀ x : ℚ, 2 * (x - 3)^2 - 12 = a * x^2 + b * x + c) ∧ (10 * a - b - 4 * c = 8)) :=
by
  sorry

end expanded_form_correct_l1194_119478


namespace johnny_practice_l1194_119425

variable (P : ℕ) -- Current amount of practice in days
variable (h : P = 40) -- Given condition translating Johnny's practice amount
variable (d : ℕ) -- Additional days needed

theorem johnny_practice : d = 80 :=
by
  have goal : 3 * P = P + d := sorry
  have initial_condition : P = 40 := sorry
  have required : d = 3 * 40 - 40 := sorry
  sorry

end johnny_practice_l1194_119425


namespace minimum_value_expression_l1194_119409

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ x : ℝ, 
    (∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → 
    x = (a + 3 * c) / (a + 2 * b + c) + (4 * b) / (a + b + 2 * c) - (8 * c) / (a + b + 3 * c)) ∧
    x = -17 + 12 * Real.sqrt 2 := 
sorry

end minimum_value_expression_l1194_119409


namespace Mira_trips_to_fill_tank_l1194_119430

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

noncomputable def volume_of_cube (a : ℝ) : ℝ :=
  a^3

noncomputable def number_of_trips (cube_side : ℝ) (sphere_diameter : ℝ) : ℕ :=
  let r := sphere_diameter / 2
  let sphere_volume := volume_of_sphere r
  let cube_volume := volume_of_cube cube_side
  Nat.ceil (cube_volume / sphere_volume)

theorem Mira_trips_to_fill_tank : number_of_trips 8 6 = 5 :=
by
  sorry

end Mira_trips_to_fill_tank_l1194_119430


namespace audio_space_per_hour_l1194_119428

/-
The digital music library holds 15 days of music.
The library occupies 20,000 megabytes of disk space.
The library contains both audio and video files.
Video files take up twice as much space per hour as audio files.
There is an equal number of hours for audio and video.
-/

theorem audio_space_per_hour (total_days : ℕ) (total_space : ℕ) (equal_hours : Prop) (video_space : ℕ → ℕ) 
  (H1 : total_days = 15)
  (H2 : total_space = 20000)
  (H3 : equal_hours)
  (H4 : ∀ x, video_space x = 2 * x) :
  ∃ x, x = 37 :=
by
  sorry

end audio_space_per_hour_l1194_119428


namespace actual_diameter_of_tissue_is_0_03_mm_l1194_119446

-- Defining necessary conditions
def magnified_diameter_meters : ℝ := 0.15
def magnification_factor : ℝ := 5000
def meters_to_millimeters : ℝ := 1000

-- Prove that the actual diameter of the tissue is 0.03 millimeters
theorem actual_diameter_of_tissue_is_0_03_mm :
  (magnified_diameter_meters * meters_to_millimeters) / magnification_factor = 0.03 := 
  sorry

end actual_diameter_of_tissue_is_0_03_mm_l1194_119446


namespace find_cd_l1194_119484

def g (c d x : ℝ) := c * x^3 - 7 * x^2 + d * x - 4

theorem find_cd : ∃ c d : ℝ, (g c d 2 = -4) ∧ (g c d (-1) = -22) ∧ (c = 19/3) ∧ (d = -8/3) := 
by
  sorry

end find_cd_l1194_119484


namespace right_triangle_area_l1194_119466

theorem right_triangle_area :
  ∃ (a b c : ℕ), (c^2 = a^2 + b^2) ∧ (2 * b^2 - 23 * b + 11 = 0) ∧ (a * b / 2 = 330) :=
sorry

end right_triangle_area_l1194_119466


namespace simon_gift_bags_l1194_119457

theorem simon_gift_bags (rate_per_day : ℕ) (days : ℕ) (total_bags : ℕ) :
  rate_per_day = 42 → days = 13 → total_bags = rate_per_day * days → total_bags = 546 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end simon_gift_bags_l1194_119457


namespace scientific_notation_of_trade_volume_l1194_119432

-- Define the total trade volume
def total_trade_volume : ℕ := 175000000000

-- Define the expected scientific notation result
def expected_result : ℝ := 1.75 * 10^11

-- Theorem stating the problem
theorem scientific_notation_of_trade_volume :
  (total_trade_volume : ℝ) = expected_result := by
  sorry

end scientific_notation_of_trade_volume_l1194_119432


namespace carol_packs_l1194_119463

theorem carol_packs (n_invites n_per_pack : ℕ) (h1 : n_invites = 12) (h2 : n_per_pack = 4) : n_invites / n_per_pack = 3 :=
by
  sorry

end carol_packs_l1194_119463


namespace intersection_complement_eq_l1194_119429

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {4, 5}

-- Theorem
theorem intersection_complement_eq : (A ∩ (U \ B)) = {2, 3} :=
by 
  sorry

end intersection_complement_eq_l1194_119429


namespace geom_sequence_sum_first_ten_terms_l1194_119441

noncomputable def geom_sequence_sum (a1 q n : ℕ) : ℕ :=
  a1 * (1 - q^n) / (1 - q)

theorem geom_sequence_sum_first_ten_terms (a : ℕ) (q : ℕ) (h1 : a * (1 + q) = 6) (h2 : a * q^3 * (1 + q) = 48) :
  geom_sequence_sum a q 10 = 2046 :=
sorry

end geom_sequence_sum_first_ten_terms_l1194_119441


namespace range_arcsin_x_squared_minus_x_l1194_119424

noncomputable def range_of_arcsin : Set ℝ :=
  {x | -Real.arcsin (1 / 4) ≤ x ∧ x ≤ Real.pi / 2}

theorem range_arcsin_x_squared_minus_x :
  ∀ x : ℝ, ∃ y ∈ range_of_arcsin, y = Real.arcsin (x^2 - x) :=
by
  sorry

end range_arcsin_x_squared_minus_x_l1194_119424


namespace triangle_final_position_after_rotation_l1194_119469

-- Definitions for the initial conditions
def square_rolls_clockwise_around_octagon : Prop := 
  true -- placeholder definition, assume this defines the motion correctly

def triangle_initial_position : ℕ := 0 -- representing bottom as 0

-- Defining the proof problem
theorem triangle_final_position_after_rotation :
  square_rolls_clockwise_around_octagon →
  triangle_initial_position = 0 →
  triangle_initial_position = 0 :=
by
  intros
  sorry

end triangle_final_position_after_rotation_l1194_119469


namespace correct_grid_l1194_119405

def A := 8
def B := 6
def C := 4
def D := 2

def grid := [[A, 1, 9],
             [3, 5, D],
             [B, C, 7]]

theorem correct_grid :
  (A + 1 < 12) ∧ (A + 3 < 12) ∧ (1 + 9 < 12) ∧
  (1 + 5 < 12) ∧ (3 + 5 < 12) ∧ (3 + B < 12) ∧
  (5 + D < 12) ∧ (5 + C < 12) ∧ (9 + D < 12) ∧
  (B + C < 12) ∧ (C + 7 < 12) :=
by
  -- This is to provide a sketch dummy theorem, we'd prove each step here  
  sorry

end correct_grid_l1194_119405


namespace original_profit_percentage_l1194_119400

theorem original_profit_percentage
  (C : ℝ) -- original cost
  (S : ℝ) -- selling price
  (y : ℝ) -- original profit percentage
  (hS : S = C * (1 + 0.01 * y)) -- condition for selling price based on original cost
  (hC' : S = 0.85 * C * (1 + 0.01 * (y + 20))) -- condition for selling price based on reduced cost
  : y = -89 :=
by
  sorry

end original_profit_percentage_l1194_119400


namespace birds_in_marsh_end_of_day_l1194_119459

def geese_initial : Nat := 58
def ducks : Nat := 37
def geese_flew_away : Nat := 15
def swans : Nat := 22
def herons : Nat := 2

theorem birds_in_marsh_end_of_day : 
  58 - 15 + 37 + 22 + 2 = 104 := by
  sorry

end birds_in_marsh_end_of_day_l1194_119459


namespace max_discount_rate_l1194_119473

theorem max_discount_rate (c p m : ℝ) (h₀ : c = 4) (h₁ : p = 5) (h₂ : m = 0.4) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (p * (1 - x / 100) - c ≥ m) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l1194_119473


namespace num_individuals_eliminated_l1194_119448

theorem num_individuals_eliminated (pop_size : ℕ) (sample_size : ℕ) :
  (pop_size % sample_size) = 2 :=
by
  -- Given conditions
  let pop_size := 1252
  let sample_size := 50
  -- Proof skipped
  sorry

end num_individuals_eliminated_l1194_119448


namespace bee_honeycomb_path_l1194_119471

theorem bee_honeycomb_path (x1 x2 x3 : ℕ) (honeycomb_grid : Prop)
  (shortest_path : ℕ) (honeycomb_property : shortest_path = 100)
  (path_decomposition : x1 + x2 + x3 = 100) : x1 = 50 ∧ x2 + x3 = 50 := 
sorry

end bee_honeycomb_path_l1194_119471


namespace solution_to_water_l1194_119436

theorem solution_to_water (A W S T: ℝ) (h1: A = 0.04) (h2: W = 0.02) (h3: S = 0.06) (h4: T = 0.48) :
  (T * (W / S) = 0.16) :=
by
  sorry

end solution_to_water_l1194_119436


namespace sum_of_cubes_l1194_119479

theorem sum_of_cubes (k : ℤ) : 
  24 * k = (k + 2)^3 + (-k)^3 + (-k)^3 + (k - 2)^3 :=
by
  sorry

end sum_of_cubes_l1194_119479


namespace equilateral_triangle_area_l1194_119415

theorem equilateral_triangle_area (h : ∀ (a : ℝ), a = 2 * Real.sqrt 3) : 
  ∃ (a : ℝ), a = 4 * Real.sqrt 3 := 
sorry

end equilateral_triangle_area_l1194_119415


namespace sum_of_perimeters_l1194_119445

theorem sum_of_perimeters (x y : ℝ) (h1 : x^2 + y^2 = 85) (h2 : x^2 - y^2 = 41) :
  4 * (Real.sqrt 63 + Real.sqrt 22) = 4 * (Real.sqrt x^2 + Real.sqrt y^2) :=
by
  sorry

end sum_of_perimeters_l1194_119445


namespace billy_raspberry_juice_billy_raspberry_juice_quarts_l1194_119435

theorem billy_raspberry_juice (V : ℚ) (h : V / 12 + 1 = 3) : V = 24 :=
by sorry

theorem billy_raspberry_juice_quarts (V : ℚ) (h : V / 12 + 1 = 3) : V / 4 = 6 :=
by sorry

end billy_raspberry_juice_billy_raspberry_juice_quarts_l1194_119435


namespace temp_on_Monday_l1194_119455

variable (M T W Th F : ℤ)

-- Given conditions
axiom sum_MTWT : M + T + W + Th = 192
axiom sum_TWTF : T + W + Th + F = 184
axiom temp_F : F = 34
axiom exists_day_temp_42 : ∃ (day : String), 
  (day = "Monday" ∨ day = "Tuesday" ∨ day = "Wednesday" ∨ day = "Thursday" ∨ day = "Friday") ∧
  (if day = "Monday" then M else if day = "Tuesday" then T else if day = "Wednesday" then W else if day = "Thursday" then Th else F) = 42

-- Prove temperature of Monday is 42
theorem temp_on_Monday : M = 42 := 
by
  sorry

end temp_on_Monday_l1194_119455


namespace even_perfect_square_factors_l1194_119493

theorem even_perfect_square_factors :
  let factors := 2^6 * 5^4 * 7^3
  ∃ (count : ℕ), count = (3 * 3 * 2) ∧
  ∀ (a b c : ℕ), (0 ≤ a ∧ a ≤ 6 ∧ 0 ≤ c ∧ c ≤ 4 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 
  a % 2 = 0 ∧ 2 ≤ a ∧ c % 2 = 0 ∧ b % 2 = 0) → 
  a * b * c < count :=
by
  sorry

end even_perfect_square_factors_l1194_119493
