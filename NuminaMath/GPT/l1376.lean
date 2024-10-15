import Mathlib

namespace NUMINAMATH_GPT_spherical_to_rectangular_example_l1376_137617

noncomputable def spherical_to_rectangular (ρ θ ϕ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin ϕ * Real.cos θ, ρ * Real.sin ϕ * Real.sin θ, ρ * Real.cos ϕ)

theorem spherical_to_rectangular_example :
  spherical_to_rectangular 4 (Real.pi / 4) (Real.pi / 6) = (Real.sqrt 2, Real.sqrt 2, 2 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_spherical_to_rectangular_example_l1376_137617


namespace NUMINAMATH_GPT_polynomial_terms_equal_l1376_137628

theorem polynomial_terms_equal (p q : ℝ) (hp : 0 < p) (hq : 0 < q) (h : p + q = 1) :
  (9 * p^8 * q = 36 * p^7 * q^2) → p = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_terms_equal_l1376_137628


namespace NUMINAMATH_GPT_distance_walked_l1376_137672

theorem distance_walked (D : ℝ) (t1 t2 : ℝ): 
  (t1 = D / 4) → 
  (t2 = D / 3) → 
  (t2 - t1 = 1 / 2) → 
  D = 6 := 
by
  sorry

end NUMINAMATH_GPT_distance_walked_l1376_137672


namespace NUMINAMATH_GPT_quadratic_residue_l1376_137624

theorem quadratic_residue (a : ℤ) (p : ℕ) (hp : p > 2) (ha_nonzero : a ≠ 0) :
  (∃ b : ℤ, b^2 ≡ a [ZMOD p] → a^((p - 1) / 2) ≡ 1 [ZMOD p]) ∧
  (¬ ∃ b : ℤ, b^2 ≡ a [ZMOD p] → a^((p - 1) / 2) ≡ -1 [ZMOD p]) :=
sorry

end NUMINAMATH_GPT_quadratic_residue_l1376_137624


namespace NUMINAMATH_GPT_rachel_more_than_adam_l1376_137683

variable (R J A : ℕ)

def condition1 := R = 75
def condition2 := R = J - 6
def condition3 := R > A
def condition4 := (R + J + A) / 3 = 72

theorem rachel_more_than_adam
  (h1 : condition1 R)
  (h2 : condition2 R J)
  (h3 : condition3 R A)
  (h4 : condition4 R J A) : 
  R - A = 15 := 
by
  sorry

end NUMINAMATH_GPT_rachel_more_than_adam_l1376_137683


namespace NUMINAMATH_GPT_ariana_average_speed_l1376_137626

theorem ariana_average_speed
  (sadie_speed : ℝ)
  (sadie_time : ℝ)
  (ariana_time : ℝ)
  (sarah_speed : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (sadie_speed_eq : sadie_speed = 3)
  (sadie_time_eq : sadie_time = 2)
  (ariana_time_eq : ariana_time = 0.5)
  (sarah_speed_eq : sarah_speed = 4)
  (total_time_eq : total_time = 4.5)
  (total_distance_eq : total_distance = 17) :
  ∃ ariana_speed : ℝ, ariana_speed = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_ariana_average_speed_l1376_137626


namespace NUMINAMATH_GPT_contest_end_time_l1376_137661

-- Definitions for the conditions
def start_time_pm : Nat := 15 -- 3:00 p.m. in 24-hour format
def duration_min : Nat := 720

-- Proof that the contest ended at 3:00 a.m.
theorem contest_end_time :
  let end_time := (start_time_pm + (duration_min / 60)) % 24
  end_time = 3 :=
by
  -- This would be the place to provide the proof
  sorry

end NUMINAMATH_GPT_contest_end_time_l1376_137661


namespace NUMINAMATH_GPT_product_of_20_random_digits_ends_with_zero_l1376_137684

noncomputable def probability_product_ends_in_zero : ℝ := 
  (1 - (9 / 10)^20) +
  (9 / 10)^20 * (1 - (5 / 9)^20) * (1 - (8 / 9)^19)

theorem product_of_20_random_digits_ends_with_zero : 
  abs (probability_product_ends_in_zero - 0.988) < 0.001 :=
by
  sorry

end NUMINAMATH_GPT_product_of_20_random_digits_ends_with_zero_l1376_137684


namespace NUMINAMATH_GPT_range_of_x_for_sqrt_meaningful_l1376_137659

theorem range_of_x_for_sqrt_meaningful (x : ℝ) (h : x + 2 ≥ 0) : x ≥ -2 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_x_for_sqrt_meaningful_l1376_137659


namespace NUMINAMATH_GPT_red_marbles_in_bag_l1376_137615

theorem red_marbles_in_bag (T R : ℕ) (hT : T = 84)
    (probability_not_red : ((T - R : ℚ) / T)^2 = 36 / 49) : 
    R = 12 := 
sorry

end NUMINAMATH_GPT_red_marbles_in_bag_l1376_137615


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l1376_137605

-- Part (a)
theorem part_a : (7 * (2 / 3) + 16 * (5 / 12)) = (34 / 3) :=
by
  sorry

-- Part (b)
theorem part_b : (5 - (2 / (5 / 3))) = (19 / 5) :=
by
  sorry

-- Part (c)
theorem part_c : (1 + (2 / (1 + (3 / (1 + 4))))) = (9 / 4) :=
by
  sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l1376_137605


namespace NUMINAMATH_GPT_range_of_m_non_perpendicular_tangent_l1376_137658

noncomputable def f (m x : ℝ) : ℝ := Real.exp x - m * x

theorem range_of_m_non_perpendicular_tangent (m : ℝ) :
  (∀ x : ℝ, (deriv (f m) x ≠ -2)) → m ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_non_perpendicular_tangent_l1376_137658


namespace NUMINAMATH_GPT_locus_of_projection_l1376_137641

theorem locus_of_projection {a b c : ℝ} (h : (1 / a ^ 2) + (1 / b ^ 2) = 1 / c ^ 2) :
  ∀ (x y : ℝ), (x, y) ∈ ({P : ℝ × ℝ | ∃ a b : ℝ, P = ((a * b^2) / (a^2 + b^2), (a^2 * b) / (a^2 + b^2)) ∧ (1 / a ^ 2) + (1 / b ^ 2) = 1 / c ^ 2}) → 
    x^2 + y^2 = c^2 := 
sorry

end NUMINAMATH_GPT_locus_of_projection_l1376_137641


namespace NUMINAMATH_GPT_pythagorean_theorem_l1376_137682

theorem pythagorean_theorem (a b c : ℝ) (h : a^2 + b^2 = c^2) : a^2 + b^2 = c^2 :=
by
  sorry

end NUMINAMATH_GPT_pythagorean_theorem_l1376_137682


namespace NUMINAMATH_GPT_math_problem_l1376_137668

theorem math_problem :
  3^(5+2) + 4^(1+3) = 39196 ∧
  2^(9+2) - 3^(4+1) = 3661 ∧
  1^(8+6) + 3^(2+3) = 250 ∧
  6^(5+4) - 4^(5+1) = 409977 → 
  5^(7+2) - 2^(5+3) = 1952869 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1376_137668


namespace NUMINAMATH_GPT_janet_earns_more_as_freelancer_l1376_137610

-- Definitions for the problem conditions
def current_job_weekly_hours : ℕ := 40
def current_job_hourly_rate : ℕ := 30

def freelance_client_a_hours_per_week : ℕ := 15
def freelance_client_a_hourly_rate : ℕ := 45

def freelance_client_b_hours_project1_per_week : ℕ := 5
def freelance_client_b_hours_project2_per_week : ℕ := 10
def freelance_client_b_hourly_rate : ℕ := 40

def freelance_client_c_hours_per_week : ℕ := 20
def freelance_client_c_rate_range : ℕ × ℕ := (35, 42)

def weekly_fica_taxes : ℕ := 25
def monthly_healthcare_premiums : ℕ := 400
def monthly_increased_rent : ℕ := 750
def monthly_business_phone_internet : ℕ := 150
def business_expense_percentage : ℕ := 10

def weeks_in_month : ℕ := 4

-- Define the calculations
def current_job_monthly_earnings := current_job_weekly_hours * current_job_hourly_rate * weeks_in_month

def freelance_client_a_weekly_earnings := freelance_client_a_hours_per_week * freelance_client_a_hourly_rate
def freelance_client_b_weekly_earnings := (freelance_client_b_hours_project1_per_week + freelance_client_b_hours_project2_per_week) * freelance_client_b_hourly_rate
def freelance_client_c_weekly_earnings := freelance_client_c_hours_per_week * ((freelance_client_c_rate_range.1 + freelance_client_c_rate_range.2) / 2)

def total_freelance_weekly_earnings := freelance_client_a_weekly_earnings + freelance_client_b_weekly_earnings + freelance_client_c_weekly_earnings
def total_freelance_monthly_earnings := total_freelance_weekly_earnings * weeks_in_month

def total_additional_expenses := (weekly_fica_taxes * weeks_in_month) + monthly_healthcare_premiums + monthly_increased_rent + monthly_business_phone_internet

def business_expense_deduction := (total_freelance_monthly_earnings * business_expense_percentage) / 100
def adjusted_freelance_earnings_after_deduction := total_freelance_monthly_earnings - business_expense_deduction
def adjusted_freelance_earnings_after_expenses := adjusted_freelance_earnings_after_deduction - total_additional_expenses

def earnings_difference := adjusted_freelance_earnings_after_expenses - current_job_monthly_earnings

-- The theorem to be proved
theorem janet_earns_more_as_freelancer :
  earnings_difference = 1162 :=
sorry

end NUMINAMATH_GPT_janet_earns_more_as_freelancer_l1376_137610


namespace NUMINAMATH_GPT_erica_time_is_65_l1376_137646

-- Definitions for the conditions
def dave_time : ℕ := 10
def chuck_time : ℕ := 5 * dave_time
def erica_time : ℕ := chuck_time + 3 * chuck_time / 10

-- The proof statement
theorem erica_time_is_65 : erica_time = 65 := by
  sorry

end NUMINAMATH_GPT_erica_time_is_65_l1376_137646


namespace NUMINAMATH_GPT_olivia_total_cost_l1376_137679

-- Definitions based on conditions given in the problem.
def daily_rate : ℕ := 30 -- daily rate in dollars per day
def mileage_rate : ℕ := 25 -- mileage rate in cents per mile (converted to cents to avoid fractions)
def rental_days : ℕ := 3 -- number of days the car is rented
def miles_driven : ℕ := 500 -- number of miles driven

-- Calculate costs in cents to avoid fractions in the Lean theorem statement.
def daily_rental_cost : ℕ := daily_rate * rental_days * 100
def mileage_cost : ℕ := mileage_rate * miles_driven
def total_cost : ℕ := daily_rental_cost + mileage_cost

-- Final statement to be proved, converting total cost back to dollars.
theorem olivia_total_cost : (total_cost / 100) = 215 := by
  sorry

end NUMINAMATH_GPT_olivia_total_cost_l1376_137679


namespace NUMINAMATH_GPT_value_of_b_minus_a_l1376_137648

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x / 2)

theorem value_of_b_minus_a (a b : ℝ) (h1 : ∀ x ∈ Set.Icc a b, f x ∈ Set.Icc (-1 : ℝ) 2) (h2 : ∀ x, f x = 2 * Real.sin (x / 2)) : 
  b - a ≠ 14 * Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_value_of_b_minus_a_l1376_137648


namespace NUMINAMATH_GPT_find_x_l1376_137666

theorem find_x :
  ∃ x : ℚ, (1 / 3) * ((x + 8) + (8*x + 3) + (3*x + 9)) = 5*x - 9 ∧ x = 47 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1376_137666


namespace NUMINAMATH_GPT_incorrect_propositions_l1376_137685

theorem incorrect_propositions :
  ¬ (∀ P : Prop, P → P) ∨
  (¬ (∀ x : ℝ, x^2 - x ≤ 0) ↔ (∃ x : ℝ, x^2 - x > 0)) ∨
  (∀ (R : Type) (f : R → Prop), (∀ r, f r → ∃ r', f r') = ∃ r, f r ∧ ∃ r', f r') ∨
  (∀ (x : ℝ), x ≠ 3 → abs x = 3 → x = 3) :=
by sorry

end NUMINAMATH_GPT_incorrect_propositions_l1376_137685


namespace NUMINAMATH_GPT_geometric_sum_over_term_l1376_137693

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q ^ n) / (1 - q)

noncomputable def geometric_term (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

theorem geometric_sum_over_term (a₁ : ℝ) (q : ℝ) (h₁ : q = 3) :
  (geometric_sum a₁ q 4) / (geometric_term a₁ q 4) = 40 / 27 := by
  sorry

end NUMINAMATH_GPT_geometric_sum_over_term_l1376_137693


namespace NUMINAMATH_GPT_barrels_of_pitch_needed_l1376_137633

-- Define the basic properties and conditions
def total_length_road := 16
def truckloads_per_mile := 3
def bags_of_gravel_per_truckload := 2
def gravel_to_pitch_ratio := 5
def miles_paved_first_day := 4
def miles_paved_second_day := 2 * miles_paved_first_day - 1
def miles_already_paved := miles_paved_first_day + miles_paved_second_day
def remaining_miles := total_length_road - miles_already_paved
def total_truckloads := truckloads_per_mile * remaining_miles
def total_bags_of_gravel := bags_of_gravel_per_truckload * total_truckloads
def barrels_of_pitch := total_bags_of_gravel / gravel_to_pitch_ratio

-- State the theorem to prove the number of barrels of pitch needed
theorem barrels_of_pitch_needed :
    barrels_of_pitch = 6 :=
by
    sorry

end NUMINAMATH_GPT_barrels_of_pitch_needed_l1376_137633


namespace NUMINAMATH_GPT_find_a_l1376_137618

noncomputable def A (a : ℝ) : Set ℝ :=
  {a + 2, (a + 1)^2, a^2 + 3 * a + 3}

theorem find_a (a : ℝ) (h : 1 ∈ A a) : a = 0 :=
  sorry

end NUMINAMATH_GPT_find_a_l1376_137618


namespace NUMINAMATH_GPT_bridge_length_correct_l1376_137690

noncomputable def length_of_bridge 
  (train_length : ℝ) 
  (time_to_cross : ℝ) 
  (train_speed_kmph : ℝ) : ℝ :=
  (train_speed_kmph * (5 / 18) * time_to_cross) - train_length

theorem bridge_length_correct :
  length_of_bridge 120 31.99744020478362 36 = 199.9744020478362 :=
by
  -- Skipping the proof details
  sorry

end NUMINAMATH_GPT_bridge_length_correct_l1376_137690


namespace NUMINAMATH_GPT_inequality_of_negatives_l1376_137601

theorem inequality_of_negatives (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a * b ∧ a * b > b^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_of_negatives_l1376_137601


namespace NUMINAMATH_GPT_poly_expansion_l1376_137647

def poly1 (z : ℝ) := 5 * z^3 + 4 * z^2 - 3 * z + 7
def poly2 (z : ℝ) := 2 * z^4 - z^3 + z - 2
def poly_product (z : ℝ) := 10 * z^7 + 6 * z^6 - 10 * z^5 + 22 * z^4 - 13 * z^3 - 11 * z^2 + 13 * z - 14

theorem poly_expansion (z : ℝ) : poly1 z * poly2 z = poly_product z := by
  sorry

end NUMINAMATH_GPT_poly_expansion_l1376_137647


namespace NUMINAMATH_GPT_expression_value_is_one_l1376_137620

theorem expression_value_is_one :
  let a1 := 121
  let b1 := 19
  let a2 := 91
  let b2 := 13
  (a1^2 - b1^2) / (a2^2 - b2^2) * ((a2 - b2) * (a2 + b2)) / ((a1 - b1) * (a1 + b1)) = 1 := by
  sorry

end NUMINAMATH_GPT_expression_value_is_one_l1376_137620


namespace NUMINAMATH_GPT_problem_statement_l1376_137619

theorem problem_statement (a b : ℤ) (h1 : b = 7) (h2: a * b = 2 * (a + b) + 1) :
  b - a = 4 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1376_137619


namespace NUMINAMATH_GPT_number_of_children_l1376_137650

-- Define the number of adults and their ticket price
def num_adults := 9
def adult_ticket_price := 11

-- Define the children's ticket price and the total cost difference
def child_ticket_price := 7
def cost_difference := 50

-- Define the total cost for adult tickets
def total_adult_cost := num_adults * adult_ticket_price

-- Given the conditions, prove that the number of children is 7
theorem number_of_children : ∃ c : ℕ, total_adult_cost = c * child_ticket_price + cost_difference ∧ c = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_of_children_l1376_137650


namespace NUMINAMATH_GPT_brendan_threw_back_l1376_137600

-- Brendan's catches in the morning, throwing back x fish and catching more in the afternoon
def brendan_morning (x : ℕ) : ℕ := 8 - x
def brendan_afternoon : ℕ := 5

-- Brendan's and his dad's total catches
def brendan_total (x : ℕ) : ℕ := brendan_morning x + brendan_afternoon
def dad_total : ℕ := 13

-- Combined total fish caught by both
def total_fish (x : ℕ) : ℕ := brendan_total x + dad_total

-- The number of fish thrown back by Brendan
theorem brendan_threw_back : ∃ x : ℕ, total_fish x = 23 ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_brendan_threw_back_l1376_137600


namespace NUMINAMATH_GPT_assembly_line_average_output_l1376_137696

theorem assembly_line_average_output :
  (60 / 90) + (60 / 60) = (5 / 3) →
  60 + 60 = 120 →
  120 / (5 / 3) = 72 :=
by
  intros h1 h2
  -- Proof follows, but we will end with 'sorry' to indicate further proof steps need to be done.
  sorry

end NUMINAMATH_GPT_assembly_line_average_output_l1376_137696


namespace NUMINAMATH_GPT_pair_C_product_not_36_l1376_137613

-- Definitions of the pairs
def pair_A : ℤ × ℤ := (-4, -9)
def pair_B : ℤ × ℤ := (-3, -12)
def pair_C : ℚ × ℚ := (1/2, -72)
def pair_D : ℤ × ℤ := (1, 36)
def pair_E : ℚ × ℚ := (3/2, 24)

-- Mathematical statement for the proof problem
theorem pair_C_product_not_36 :
  pair_C.fst * pair_C.snd ≠ 36 :=
by
  sorry

end NUMINAMATH_GPT_pair_C_product_not_36_l1376_137613


namespace NUMINAMATH_GPT_original_number_is_15_l1376_137671

theorem original_number_is_15 (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) (N : ℕ) (h4 : 100 * a + 10 * b + c = m)
  (h5 : 100 * a +  10 * b +   c +
        100 * a +   c + 10 * b + 
        100 * b +  10 * a +   c +
        100 * b +   c + 10 * a + 
        100 * c +  10 * a +   b +
        100 * c +   b + 10 * a = 3315) :
  m = 15 :=
sorry

end NUMINAMATH_GPT_original_number_is_15_l1376_137671


namespace NUMINAMATH_GPT_students_in_first_class_l1376_137686

variable (x : ℕ)
variable (avg_marks_first_class : ℕ := 40)
variable (num_students_second_class : ℕ := 28)
variable (avg_marks_second_class : ℕ := 60)
variable (avg_marks_all : ℕ := 54)

theorem students_in_first_class : (40 * x + 60 * 28) / (x + 28) = 54 → x = 12 := 
by 
  sorry

end NUMINAMATH_GPT_students_in_first_class_l1376_137686


namespace NUMINAMATH_GPT_minimum_draws_divisible_by_3_or_5_l1376_137657

theorem minimum_draws_divisible_by_3_or_5 (n : ℕ) (h : n = 90) :
  ∃ k, k = 49 ∧ ∀ (draws : ℕ), draws < k → ¬ (∃ x, 1 ≤ x ∧ x ≤ n ∧ (x % 3 = 0 ∨ x % 5 = 0)) :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_draws_divisible_by_3_or_5_l1376_137657


namespace NUMINAMATH_GPT_pipe_B_fill_time_l1376_137611

theorem pipe_B_fill_time (t : ℝ) :
  (1/10) + (2/t) - (2/15) = 1 ↔ t = 60/31 :=
by
  sorry

end NUMINAMATH_GPT_pipe_B_fill_time_l1376_137611


namespace NUMINAMATH_GPT_probability_three_dice_same_number_is_1_div_36_l1376_137654

noncomputable def probability_same_number_three_dice : ℚ :=
  let first_die := 1
  let second_die := 1 / 6
  let third_die := 1 / 6
  first_die * second_die * third_die

theorem probability_three_dice_same_number_is_1_div_36 : probability_same_number_three_dice = 1 / 36 :=
  sorry

end NUMINAMATH_GPT_probability_three_dice_same_number_is_1_div_36_l1376_137654


namespace NUMINAMATH_GPT_cricket_average_l1376_137663

theorem cricket_average (A : ℝ) (h : 20 * A + 120 = 21 * (A + 4)) : A = 36 :=
by sorry

end NUMINAMATH_GPT_cricket_average_l1376_137663


namespace NUMINAMATH_GPT_initial_soup_weight_l1376_137639

theorem initial_soup_weight (W: ℕ) (h: W / 16 = 5): W = 40 :=
by
  sorry

end NUMINAMATH_GPT_initial_soup_weight_l1376_137639


namespace NUMINAMATH_GPT_original_garden_side_length_l1376_137687

theorem original_garden_side_length (a : ℝ) (h : (a + 3)^2 = 2 * a^2 + 9) : a = 6 :=
by
  sorry

end NUMINAMATH_GPT_original_garden_side_length_l1376_137687


namespace NUMINAMATH_GPT_rectangle_area_ratio_l1376_137656

theorem rectangle_area_ratio (a b c d : ℝ) 
  (h1 : a / c = 3 / 5) 
  (h2 : b / d = 3 / 5) :
  (a * b) / (c * d) = 9 / 25 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_ratio_l1376_137656


namespace NUMINAMATH_GPT_journey_divided_into_portions_l1376_137678

theorem journey_divided_into_portions
  (total_distance : ℕ)
  (speed : ℕ)
  (time : ℝ)
  (portion_distance : ℕ)
  (portions_covered : ℕ)
  (h1 : total_distance = 35)
  (h2 : speed = 40)
  (h3 : time = 0.7)
  (h4 : portions_covered = 4)
  (distance_covered := speed * time)
  (one_portion_distance := distance_covered / portions_covered)
  (total_portions := total_distance / one_portion_distance) :
  total_portions = 5 := 
sorry

end NUMINAMATH_GPT_journey_divided_into_portions_l1376_137678


namespace NUMINAMATH_GPT_chimney_base_radius_l1376_137692

-- Given conditions
def tinplate_length := 219.8
def tinplate_width := 125.6
def pi_approx := 3.14

def radius_length (circumference : Float) : Float :=
  circumference / (2 * pi_approx)

def radius_width (circumference : Float) : Float :=
  circumference / (2 * pi_approx)

theorem chimney_base_radius :
  radius_length tinplate_length = 35 ∧ radius_width tinplate_width = 20 :=
by 
  sorry

end NUMINAMATH_GPT_chimney_base_radius_l1376_137692


namespace NUMINAMATH_GPT_pictures_per_coloring_book_l1376_137642

theorem pictures_per_coloring_book
    (total_colored : ℕ)
    (remaining_pictures : ℕ)
    (two_books : ℕ)
    (h1 : total_colored = 20) 
    (h2 : remaining_pictures = 68) 
    (h3 : two_books = 2) :
  (total_colored + remaining_pictures) / two_books = 44 :=
by
  sorry

end NUMINAMATH_GPT_pictures_per_coloring_book_l1376_137642


namespace NUMINAMATH_GPT_not_divisible_by_5_l1376_137643

theorem not_divisible_by_5 (b : ℕ) : b = 6 ↔ ¬ (5 ∣ (2 * b ^ 3 - 2 * b ^ 2 + 2 * b - 1)) :=
sorry

end NUMINAMATH_GPT_not_divisible_by_5_l1376_137643


namespace NUMINAMATH_GPT_square_divided_into_40_smaller_squares_l1376_137602

theorem square_divided_into_40_smaller_squares : ∃ squares : ℕ, squares = 40 :=
by
  sorry

end NUMINAMATH_GPT_square_divided_into_40_smaller_squares_l1376_137602


namespace NUMINAMATH_GPT_soccer_campers_l1376_137667

theorem soccer_campers (total_campers : ℕ) (basketball_campers : ℕ) (football_campers : ℕ) (h1 : total_campers = 88) (h2 : basketball_campers = 24) (h3 : football_campers = 32) : 
  total_campers - (basketball_campers + football_campers) = 32 := 
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_soccer_campers_l1376_137667


namespace NUMINAMATH_GPT_maximize_lower_houses_l1376_137627

theorem maximize_lower_houses (x y : ℕ) 
    (h1 : x + 2 * y = 30)
    (h2 : 0 < y)
    (h3 : (∃ k, k = 112)) :
  ∃ x y, (x + 2 * y = 30) ∧ ((x * y)) = 112 :=
by
  sorry

end NUMINAMATH_GPT_maximize_lower_houses_l1376_137627


namespace NUMINAMATH_GPT_total_marks_prove_total_marks_l1376_137606

def average_marks : ℝ := 40
def number_of_candidates : ℕ := 50

theorem total_marks (average_marks : ℝ) (number_of_candidates : ℕ) : Real :=
  average_marks * number_of_candidates

theorem prove_total_marks : total_marks average_marks number_of_candidates = 2000 := 
by
  sorry

end NUMINAMATH_GPT_total_marks_prove_total_marks_l1376_137606


namespace NUMINAMATH_GPT_enthalpy_of_formation_C6H6_l1376_137697

theorem enthalpy_of_formation_C6H6 :
  ∀ (enthalpy_C2H2 : ℝ) (enthalpy_C6H6 : ℝ)
  (enthalpy_C6H6_C6H6 : ℝ) (Hess_law : Prop),
  (enthalpy_C2H2 = 226.7) →
  (enthalpy_C6H6 = 631.1) →
  (enthalpy_C6H6_C6H6 = -33.9) →
  Hess_law →
  -- Using the given conditions to accumulate the enthalpy change for the formation of C6H6.
  ∃ Q_formation : ℝ, Q_formation = -82.9 := by
  sorry

end NUMINAMATH_GPT_enthalpy_of_formation_C6H6_l1376_137697


namespace NUMINAMATH_GPT_empty_set_is_d_l1376_137652

open Set

theorem empty_set_is_d : {x : ℝ | x^2 - x + 1 = 0} = ∅ :=
by
  sorry

end NUMINAMATH_GPT_empty_set_is_d_l1376_137652


namespace NUMINAMATH_GPT_perimeter_of_original_rectangle_l1376_137621

-- Define the rectangle's dimensions based on the given condition
def length_of_rectangle := 2 * 8 -- because it forms two squares of side 8 cm each
def width_of_rectangle := 8 -- side of the squares

-- Using the formula for the perimeter of a rectangle: P = 2 * (length + width)
def perimeter_of_rectangle := 2 * (length_of_rectangle + width_of_rectangle)

-- The statement we need to prove
theorem perimeter_of_original_rectangle : perimeter_of_rectangle = 48 := by
  sorry

end NUMINAMATH_GPT_perimeter_of_original_rectangle_l1376_137621


namespace NUMINAMATH_GPT_probability_allison_greater_l1376_137688

theorem probability_allison_greater (A D S : ℕ) (prob_derek_less_than_4 : ℚ) (prob_sophie_less_than_4 : ℚ) : 
  (A > D) ∧ (A > S) → prob_derek_less_than_4 = 1 / 2 ∧ prob_sophie_less_than_4 = 2 / 3 → 
  (1 / 2 : ℚ) * (2 / 3 : ℚ) = (1 / 3 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_probability_allison_greater_l1376_137688


namespace NUMINAMATH_GPT_find_other_x_intercept_l1376_137649

theorem find_other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, x = 2 → y = -3) (h_x_intercept : ∀ x, x = 5 → y = 0) : 
  ∃ x, x = -1 ∧ y = 0 := 
sorry

end NUMINAMATH_GPT_find_other_x_intercept_l1376_137649


namespace NUMINAMATH_GPT_problem1_problem2_l1376_137631

-- Definitions based on conditions in the problem
def seq_sum (a : ℕ) (n : ℕ) : ℕ := a * 2^n - 1
def a1 (a : ℕ) : ℕ := seq_sum a 1
def a4 (a : ℕ) : ℕ := seq_sum a 4 - seq_sum a 3

-- Problem statement 1
theorem problem1 (a : ℕ) (h : a = 3) : a1 a = 5 ∧ a4 a = 24 := by 
  sorry

-- Geometric sequence conditions
def is_geometric (a_n : ℕ → ℕ) : Prop :=
  ∃ q ≠ 1, ∀ n, a_n (n + 1) = q * a_n n

-- Definitions for the geometric sequence part
def a_n (a : ℕ) (n : ℕ) : ℕ :=
  if n = 1 then 2 * a - 1
  else if n = 2 then 2 * a
  else if n = 3 then 4 * a
  else 0 -- Simplifying for the first few terms only

-- Problem statement 2
theorem problem2 : (∃ a : ℕ, is_geometric (a_n a)) → ∃ a : ℕ, a = 1 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1376_137631


namespace NUMINAMATH_GPT_polynomial_sum_l1376_137674

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : 
  f x + g x + h x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_sum_l1376_137674


namespace NUMINAMATH_GPT_donation_amount_per_person_l1376_137629

theorem donation_amount_per_person (m n : ℕ) 
  (h1 : m + 11 = n + 9) 
  (h2 : ∃ d : ℕ, (m * n + 9 * m + 11 * n + 145) = d * (m + 11)) 
  (h3 : ∃ d : ℕ, (m * n + 9 * m + 11 * n + 145) = d * (n + 9))
  : ∃ k : ℕ, k = 25 ∨ k = 47 :=
by
  sorry

end NUMINAMATH_GPT_donation_amount_per_person_l1376_137629


namespace NUMINAMATH_GPT_ram_actual_distance_from_base_l1376_137634

def map_distance_between_mountains : ℝ := 312
def actual_distance_between_mountains : ℝ := 136
def ram_map_distance_from_base : ℝ := 28

theorem ram_actual_distance_from_base :
  ram_map_distance_from_base * (actual_distance_between_mountains / map_distance_between_mountains) = 12.205 :=
by sorry

end NUMINAMATH_GPT_ram_actual_distance_from_base_l1376_137634


namespace NUMINAMATH_GPT_factor_expression_l1376_137669

theorem factor_expression (a : ℝ) :
  (9 * a^4 + 105 * a^3 - 15 * a^2 + 1) - (-2 * a^4 + 3 * a^3 - 4 * a^2 + 2 * a - 5) =
  (a - 3) * (11 * a^2 * (a + 1) - 2) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1376_137669


namespace NUMINAMATH_GPT_speed_of_man_in_still_water_l1376_137622

theorem speed_of_man_in_still_water 
  (v_m v_s : ℝ)
  (h1 : 32 = 4 * (v_m + v_s))
  (h2 : 24 = 4 * (v_m - v_s)) :
  v_m = 7 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_man_in_still_water_l1376_137622


namespace NUMINAMATH_GPT_shirt_pants_outfits_l1376_137673

theorem shirt_pants_outfits
  (num_shirts : ℕ) (num_pants : ℕ) (num_formal_pants : ℕ) (num_casual_pants : ℕ) (num_assignee_shirts : ℕ) :
  num_shirts = 5 →
  num_pants = 6 →
  num_formal_pants = 3 →
  num_casual_pants = 3 →
  num_assignee_shirts = 3 →
  (num_casual_pants * num_shirts) + (num_formal_pants * num_assignee_shirts) = 24 :=
by
  intros h_shirts h_pants h_formal h_casual h_assignee
  sorry

end NUMINAMATH_GPT_shirt_pants_outfits_l1376_137673


namespace NUMINAMATH_GPT_value_of_x_l1376_137603

theorem value_of_x : (2015^2 + 2015 - 1) / (2015 : ℝ) = 2016 - 1 / 2015 := 
  sorry

end NUMINAMATH_GPT_value_of_x_l1376_137603


namespace NUMINAMATH_GPT_vector_subtraction_l1376_137665

/-
Define the vectors we are working with.
-/
def v1 : Matrix (Fin 2) (Fin 1) ℤ := ![![3], ![-8]]
def v2 : Matrix (Fin 2) (Fin 1) ℤ := ![![2], ![-6]]
def scalar : ℤ := 5
def result : Matrix (Fin 2) (Fin 1) ℤ := ![![-7], ![22]]

/-
The statement of the proof problem.
-/
theorem vector_subtraction : v1 - scalar • v2 = result := 
by
  sorry

end NUMINAMATH_GPT_vector_subtraction_l1376_137665


namespace NUMINAMATH_GPT_polynomial_square_b_value_l1376_137644

theorem polynomial_square_b_value (a b p q : ℝ) (h : (∀ x : ℝ, x^4 + x^3 - x^2 + a * x + b = (x^2 + p * x + q)^2)) : b = 25/64 := by
  sorry

end NUMINAMATH_GPT_polynomial_square_b_value_l1376_137644


namespace NUMINAMATH_GPT_determine_F_l1376_137651

theorem determine_F (A H S M F : ℕ) (ha : 0 < A) (hh : 0 < H) (hs : 0 < S) (hm : 0 < M) (hf : 0 < F):
  (A * x + H * y = z) →
  (S * x + M * y = z) →
  (F * x = z) →
  (H > A) →
  (A ≠ H) →
  (S ≠ M) →
  (F ≠ A) →
  (F ≠ H) →
  (F ≠ S) →
  (F ≠ M) →
  x = z / F →
  y = ((F - A) / H * z) / z →
  F = (A * F - S * H) / (M - H) := sorry

end NUMINAMATH_GPT_determine_F_l1376_137651


namespace NUMINAMATH_GPT_train_length_l1376_137680

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (bridge_length_m : ℕ) (conversion_factor : ℝ) :
  speed_kmh = 54 →
  time_s = 33333333333333336 / 1000000000000000 →
  bridge_length_m = 140 →
  conversion_factor = 1000 / 3600 →
  ∃ (train_length_m : ℝ), 
    speed_kmh * conversion_factor * time_s + bridge_length_m = train_length_m + bridge_length_m :=
by
  intros
  use 360
  sorry

end NUMINAMATH_GPT_train_length_l1376_137680


namespace NUMINAMATH_GPT_smallest_positive_number_is_x2_l1376_137635

noncomputable def x1 : ℝ := 14 - 4 * Real.sqrt 17
noncomputable def x2 : ℝ := 4 * Real.sqrt 17 - 14
noncomputable def x3 : ℝ := 23 - 7 * Real.sqrt 14
noncomputable def x4 : ℝ := 65 - 12 * Real.sqrt 34
noncomputable def x5 : ℝ := 12 * Real.sqrt 34 - 65

theorem smallest_positive_number_is_x2 :
  x2 = 4 * Real.sqrt 17 - 14 ∧
  (0 < x1 ∨ 0 < x2 ∨ 0 < x3 ∨ 0 < x4 ∨ 0 < x5) ∧
  (∀ x : ℝ, (x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4 ∨ x = x5) → 0 < x → x2 ≤ x) := sorry

end NUMINAMATH_GPT_smallest_positive_number_is_x2_l1376_137635


namespace NUMINAMATH_GPT_find_tax_percentage_l1376_137607

noncomputable def net_income : ℝ := 12000
noncomputable def total_income : ℝ := 13000
noncomputable def non_taxable_income : ℝ := 3000
noncomputable def taxable_income : ℝ := total_income - non_taxable_income
noncomputable def tax_percentage (T : ℝ) := total_income - (T * taxable_income)

theorem find_tax_percentage : ∃ T : ℝ, tax_percentage T = net_income :=
by
  sorry

end NUMINAMATH_GPT_find_tax_percentage_l1376_137607


namespace NUMINAMATH_GPT_div_ad_bc_l1376_137662

theorem div_ad_bc (a b c d : ℤ) (h : (a - c) ∣ (a * b + c * d)) : (a - c) ∣ (a * d + b * c) :=
sorry

end NUMINAMATH_GPT_div_ad_bc_l1376_137662


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l1376_137655

variable (B S : ℝ)

-- conditions
def condition1 : Prop := B + S = 6
def condition2 : Prop := B - S = 2

-- question to answer
theorem boat_speed_in_still_water (h1 : condition1 B S) (h2 : condition2 B S) : B = 4 :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l1376_137655


namespace NUMINAMATH_GPT_Theresa_game_scores_l1376_137653

theorem Theresa_game_scores 
  (h_sum_10 : 9 + 5 + 4 + 7 + 6 + 2 + 4 + 8 + 3 + 7 = 55)
  (h_p11 : ∀ p11 : ℕ, p11 < 10 → (55 + p11) % 11 = 0)
  (h_p12 : ∀ p11 p12 : ℕ, p11 < 10 → p12 < 10 → ((55 + p11 + p12) % 12 = 0)) :
  ∃ p11 p12 : ℕ, p11 < 10 ∧ p12 < 10 ∧ (55 + p11) % 11 = 0 ∧ (55 + p11 + p12) % 12 = 0 ∧ p11 * p12 = 0 :=
by
  sorry

end NUMINAMATH_GPT_Theresa_game_scores_l1376_137653


namespace NUMINAMATH_GPT_cricket_bat_profit_percentage_correct_football_profit_percentage_correct_l1376_137632

noncomputable def cricket_bat_selling_price : ℝ := 850
noncomputable def cricket_bat_profit : ℝ := 215
noncomputable def cricket_bat_cost_price : ℝ := cricket_bat_selling_price - cricket_bat_profit
noncomputable def cricket_bat_profit_percentage : ℝ := (cricket_bat_profit / cricket_bat_cost_price) * 100

noncomputable def football_selling_price : ℝ := 120
noncomputable def football_profit : ℝ := 45
noncomputable def football_cost_price : ℝ := football_selling_price - football_profit
noncomputable def football_profit_percentage : ℝ := (football_profit / football_cost_price) * 100

theorem cricket_bat_profit_percentage_correct :
  |cricket_bat_profit_percentage - 33.86| < 1e-2 :=
by sorry

theorem football_profit_percentage_correct :
  football_profit_percentage = 60 :=
by sorry

end NUMINAMATH_GPT_cricket_bat_profit_percentage_correct_football_profit_percentage_correct_l1376_137632


namespace NUMINAMATH_GPT_circle_equation_exists_shortest_chord_line_l1376_137608

-- Condition 1: Points A and B
def point_A : (ℝ × ℝ) := (1, -2)
def point_B : (ℝ × ℝ) := (-1, 0)

-- Condition 2: Circle passes through A and B and sum of intercepts is 2
def passes_through (x y : ℝ) (D E F : ℝ) : Prop := 
  (x^2 + y^2 + D * x + E * y + F = 0)

def satisfies_intercepts (D E : ℝ) : Prop := (-D - E = 2)

-- Prove
theorem circle_equation_exists : 
  ∃ D E F, passes_through 1 (-2) D E F ∧ passes_through (-1) 0 D E F ∧ satisfies_intercepts D E :=
sorry

-- Given that P(2, 0.5) is inside the circle from above theorem
def point_P : (ℝ × ℝ) := (2, 0.5)

-- Prove the equation of the shortest chord line l
theorem shortest_chord_line :
  ∃ m b, m = -2 ∧ point_P.2 = m * (point_P.1 - 2) + b ∧ (∀ (x y : ℝ), 4 * x + 2 * y - 9 = 0) :=
sorry

end NUMINAMATH_GPT_circle_equation_exists_shortest_chord_line_l1376_137608


namespace NUMINAMATH_GPT_product_of_three_numbers_l1376_137616

-- Define the problem conditions as variables and assumptions
variables (a b c : ℚ)
axiom h1 : a + b + c = 30
axiom h2 : a = 3 * (b + c)
axiom h3 : b = 6 * c

-- State the theorem to be proven
theorem product_of_three_numbers : a * b * c = 10125 / 14 :=
by
  sorry

end NUMINAMATH_GPT_product_of_three_numbers_l1376_137616


namespace NUMINAMATH_GPT_fourth_student_seat_number_l1376_137695

theorem fourth_student_seat_number (n : ℕ) (pop_size sample_size : ℕ)
  (s1 s2 s3 : ℕ)
  (h_pop_size : pop_size = 52)
  (h_sample_size : sample_size = 4)
  (h_6_in_sample : s1 = 6)
  (h_32_in_sample : s2 = 32)
  (h_45_in_sample : s3 = 45)
  : ∃ s4 : ℕ, s4 = 19 :=
by
  sorry

end NUMINAMATH_GPT_fourth_student_seat_number_l1376_137695


namespace NUMINAMATH_GPT_rhombus_area_in_rectangle_l1376_137640

theorem rhombus_area_in_rectangle :
  ∀ (l w : ℝ), 
  (∀ (A B C D : ℝ), 
    (2 * w = l) ∧ 
    (l * w = 72) →
    let diag1 := w 
    let diag2 := l 
    (1/2 * diag1 * diag2 = 36)) :=
by
  intros
  sorry

end NUMINAMATH_GPT_rhombus_area_in_rectangle_l1376_137640


namespace NUMINAMATH_GPT_total_fruit_salads_is_1800_l1376_137637

def Alaya_fruit_salads := 200
def Angel_fruit_salads := 2 * Alaya_fruit_salads
def Betty_fruit_salads := 3 * Angel_fruit_salads
def Total_fruit_salads := Alaya_fruit_salads + Angel_fruit_salads + Betty_fruit_salads

theorem total_fruit_salads_is_1800 : Total_fruit_salads = 1800 := by
  sorry

end NUMINAMATH_GPT_total_fruit_salads_is_1800_l1376_137637


namespace NUMINAMATH_GPT_henry_final_price_l1376_137670

-- Definitions based on the conditions in the problem
def price_socks : ℝ := 5
def price_tshirt : ℝ := price_socks + 10
def price_jeans : ℝ := 2 * price_tshirt
def discount_jeans : ℝ := 0.15 * price_jeans
def discounted_price_jeans : ℝ := price_jeans - discount_jeans
def sales_tax_jeans : ℝ := 0.08 * discounted_price_jeans
def final_price_jeans : ℝ := discounted_price_jeans + sales_tax_jeans

-- Statement to prove
theorem henry_final_price : final_price_jeans = 27.54 := by
  sorry

end NUMINAMATH_GPT_henry_final_price_l1376_137670


namespace NUMINAMATH_GPT_product_variation_l1376_137609

theorem product_variation (a b c : ℕ) (h1 : a * b = c) (h2 : b' = 10 * b) (h3 : ∃ d : ℕ, d = a * b') : d = 720 :=
by
  sorry

end NUMINAMATH_GPT_product_variation_l1376_137609


namespace NUMINAMATH_GPT_contrapositive_of_proposition_l1376_137675

theorem contrapositive_of_proposition (a b : ℝ) : (a > b → a + 1 > b) ↔ (a + 1 ≤ b → a ≤ b) :=
sorry

end NUMINAMATH_GPT_contrapositive_of_proposition_l1376_137675


namespace NUMINAMATH_GPT_Leela_Hotel_all_three_reunions_l1376_137630

theorem Leela_Hotel_all_three_reunions
  (A B C : Finset ℕ)
  (hA : A.card = 80)
  (hB : B.card = 90)
  (hC : C.card = 70)
  (hAB : (A ∩ B).card = 30)
  (hAC : (A ∩ C).card = 25)
  (hBC : (B ∩ C).card = 20)
  (hABC : ((A ∪ B ∪ C)).card = 150) : 
  (A ∩ B ∩ C).card = 15 :=
by
  sorry

end NUMINAMATH_GPT_Leela_Hotel_all_three_reunions_l1376_137630


namespace NUMINAMATH_GPT_probability_of_drawing_two_red_shoes_l1376_137660

/-- Given there are 7 red shoes and 3 green shoes, 
    and a total of 10 shoes, if two shoes are drawn randomly,
    prove that the probability of drawing both shoes as red is 7/15. -/
theorem probability_of_drawing_two_red_shoes :
  let total_shoes := 10
  let red_shoes := 7
  let green_shoes := 3
  let total_ways := Nat.choose total_shoes 2
  let red_ways := Nat.choose red_shoes 2
  (1 : ℚ) * red_ways / total_ways = 7 / 15  := by
  sorry

end NUMINAMATH_GPT_probability_of_drawing_two_red_shoes_l1376_137660


namespace NUMINAMATH_GPT_find_value_of_10n_l1376_137612

theorem find_value_of_10n (n : ℝ) (h : 2 * n = 14) : 10 * n = 70 :=
sorry

end NUMINAMATH_GPT_find_value_of_10n_l1376_137612


namespace NUMINAMATH_GPT_gcd_polynomial_multiple_l1376_137664

theorem gcd_polynomial_multiple (b : ℤ) (h : b % 2373 = 0) : Int.gcd (b^2 + 13 * b + 40) (b + 5) = 5 :=
by
  sorry

end NUMINAMATH_GPT_gcd_polynomial_multiple_l1376_137664


namespace NUMINAMATH_GPT_waiter_initial_tables_l1376_137645

theorem waiter_initial_tables
  (T : ℝ)
  (H1 : (T - 12.0) * 8.0 = 256) :
  T = 44.0 :=
sorry

end NUMINAMATH_GPT_waiter_initial_tables_l1376_137645


namespace NUMINAMATH_GPT_maxwell_meets_brad_l1376_137638

theorem maxwell_meets_brad :
  ∃ t : ℝ, t = 2 ∧ 
  (∀ distance max_speed brad_speed start_time, 
   distance = 14 ∧ 
   max_speed = 4 ∧ 
   brad_speed = 6 ∧ 
   start_time = 1 → 
   max_speed * (t + start_time) + brad_speed * t = distance) :=
by
  use 1
  sorry

end NUMINAMATH_GPT_maxwell_meets_brad_l1376_137638


namespace NUMINAMATH_GPT_angle_magnification_l1376_137689

theorem angle_magnification (α : ℝ) (h : α = 20) : α = 20 := by
  sorry

end NUMINAMATH_GPT_angle_magnification_l1376_137689


namespace NUMINAMATH_GPT_relationship_of_ys_l1376_137623

variables {k y1 y2 y3 : ℝ}

theorem relationship_of_ys (h : k < 0) 
  (h1 : y1 = k / -4) 
  (h2 : y2 = k / 2) 
  (h3 : y3 = k / 3) : 
  y1 > y3 ∧ y3 > y2 :=
by 
  sorry

end NUMINAMATH_GPT_relationship_of_ys_l1376_137623


namespace NUMINAMATH_GPT_ticket_representation_l1376_137698

-- Define a structure for representing a movie ticket
structure Ticket where
  rows : Nat
  seats : Nat

-- Define the specific instance of representing 7 rows and 5 seats
def ticket_7_5 : Ticket := ⟨7, 5⟩

-- The theorem stating our problem: the representation of 7 rows and 5 seats is (7,5)
theorem ticket_representation : ticket_7_5 = ⟨7, 5⟩ :=
  by
    -- Proof goes here (omitted as per instructions)
    sorry

end NUMINAMATH_GPT_ticket_representation_l1376_137698


namespace NUMINAMATH_GPT_least_integer_sol_l1376_137676

theorem least_integer_sol (x : ℤ) (h : |(2 : ℤ) * x + 7| ≤ 16) : x ≥ -11 := sorry

end NUMINAMATH_GPT_least_integer_sol_l1376_137676


namespace NUMINAMATH_GPT_complex_number_on_line_l1376_137625

theorem complex_number_on_line (a : ℝ) (h : (3 : ℝ) = (a - 1) + 2) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_complex_number_on_line_l1376_137625


namespace NUMINAMATH_GPT_bob_first_six_probability_l1376_137604

noncomputable def probability_bob_first_six (p : ℚ) : ℚ :=
  (1 - p) * p / (1 - ( (1 - p) * (1 - p)))

theorem bob_first_six_probability :
  probability_bob_first_six (1/6) = 5/11 :=
by
  sorry

end NUMINAMATH_GPT_bob_first_six_probability_l1376_137604


namespace NUMINAMATH_GPT_arrange_abc_l1376_137677

theorem arrange_abc : 
  let a := Real.log 5 / Real.log 0.6
  let b := 2 ^ (4 / 5)
  let c := Real.sin 1
  a < c ∧ c < b := 
by
  sorry

end NUMINAMATH_GPT_arrange_abc_l1376_137677


namespace NUMINAMATH_GPT_max_band_members_l1376_137694

theorem max_band_members (n : ℤ) (h1 : 22 * n % 24 = 2) (h2 : 22 * n < 1000) : 22 * n = 770 :=
  sorry

end NUMINAMATH_GPT_max_band_members_l1376_137694


namespace NUMINAMATH_GPT_find_f_of_one_l1376_137691

def f (x : ℝ) : ℝ := 3 * x - 1

theorem find_f_of_one : f 1 = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_f_of_one_l1376_137691


namespace NUMINAMATH_GPT_ratio_kittens_to_breeding_rabbits_l1376_137614

def breeding_rabbits : ℕ := 10
def kittens_first_spring (k : ℕ) : ℕ := k * breeding_rabbits
def adopted_kittens_first_spring (k : ℕ) : ℕ := 5 * k
def returned_kittens : ℕ := 5
def remaining_kittens_first_spring (k : ℕ) : ℕ := (k * breeding_rabbits) / 2 + returned_kittens

def kittens_second_spring : ℕ := 60
def adopted_kittens_second_spring : ℕ := 4
def remaining_kittens_second_spring : ℕ := kittens_second_spring - adopted_kittens_second_spring

def total_rabbits (k : ℕ) : ℕ := 
  breeding_rabbits + remaining_kittens_first_spring k + remaining_kittens_second_spring

theorem ratio_kittens_to_breeding_rabbits (k : ℕ) (h : total_rabbits k = 121) :
  k = 10 :=
sorry

end NUMINAMATH_GPT_ratio_kittens_to_breeding_rabbits_l1376_137614


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1376_137681

def setA (x : ℝ) : Prop := x^2 < 4
def setB : Set ℝ := {0, 1}

theorem intersection_of_A_and_B :
  {x : ℝ | setA x} ∩ setB = setB := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1376_137681


namespace NUMINAMATH_GPT_total_pieces_of_gum_and_candy_l1376_137699

theorem total_pieces_of_gum_and_candy 
  (packages_A : ℕ) (pieces_A : ℕ) (packages_B : ℕ) (pieces_B : ℕ) 
  (packages_C : ℕ) (pieces_C : ℕ) (packages_X : ℕ) (pieces_X : ℕ)
  (packages_Y : ℕ) (pieces_Y : ℕ) 
  (hA : packages_A = 10) (hA_pieces : pieces_A = 4)
  (hB : packages_B = 5) (hB_pieces : pieces_B = 8)
  (hC : packages_C = 13) (hC_pieces : pieces_C = 12)
  (hX : packages_X = 8) (hX_pieces : pieces_X = 6)
  (hY : packages_Y = 6) (hY_pieces : pieces_Y = 10) : 
  packages_A * pieces_A + packages_B * pieces_B + packages_C * pieces_C + 
  packages_X * pieces_X + packages_Y * pieces_Y = 344 := 
by
  sorry

end NUMINAMATH_GPT_total_pieces_of_gum_and_candy_l1376_137699


namespace NUMINAMATH_GPT_rose_initial_rice_l1376_137636

theorem rose_initial_rice : 
  ∀ (R : ℝ), (R - 9 / 10 * R - 1 / 4 * (R - 9 / 10 * R) = 0.75) → (R = 10) :=
by
  intro R h
  sorry

end NUMINAMATH_GPT_rose_initial_rice_l1376_137636
