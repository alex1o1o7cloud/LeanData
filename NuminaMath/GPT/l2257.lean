import Mathlib

namespace NUMINAMATH_GPT_smallest_n_mod_equality_l2257_225776

theorem smallest_n_mod_equality :
  ∃ n : ℕ, n > 0 ∧ (7^n ≡ n^7 [MOD 5]) ∧ (∀ m : ℕ, 0 < m ∧ m < n → (7^m ≡ m^7 [MOD 5]) = false) :=
  sorry

end NUMINAMATH_GPT_smallest_n_mod_equality_l2257_225776


namespace NUMINAMATH_GPT_equilateral_triangle_perimeter_l2257_225765

theorem equilateral_triangle_perimeter (p_ADC : ℝ) (h_ratio : ∀ s1 s2 : ℝ, s1 / s2 = 1 / 2) :
  p_ADC = 9 + 3 * Real.sqrt 3 → (3 * (2 * (3 + Real.sqrt 3)) = 18 + 6 * Real.sqrt 3) :=
by
  intro h
  have h1 : 3 * (2 * (3 + Real.sqrt 3)) = 18 + 6 * Real.sqrt 3 := sorry
  exact h1

end NUMINAMATH_GPT_equilateral_triangle_perimeter_l2257_225765


namespace NUMINAMATH_GPT_black_area_fraction_after_three_changes_l2257_225782

theorem black_area_fraction_after_three_changes
  (initial_black_area : ℚ)
  (change_factor : ℚ)
  (h1 : initial_black_area = 1)
  (h2 : change_factor = 2 / 3)
  : (change_factor ^ 3) * initial_black_area = 8 / 27 := 
by
  sorry

end NUMINAMATH_GPT_black_area_fraction_after_three_changes_l2257_225782


namespace NUMINAMATH_GPT_part_a_part_b_case1_part_b_case2_l2257_225722

theorem part_a (x1 x2 p : ℝ) 
  (h1 : 4 * (x1 ^ 2) + x1 + 4 * p = 0)
  (h2 : 4 * (x2 ^ 2) + x2 + 4 * p = 0) 
  (h3 : x1 / x2 + x2 / x1 = -9 / 4) : 
  p = -1 / 23 :=
sorry

theorem part_b_case1 (x1 x2 p : ℝ) 
  (h1 : 4 * (x1 ^ 2) + x1 + 4 * p = 0)
  (h2 : 4 * (x2 ^ 2) + x2 + 4 * p = 0) 
  (h3 : x2 = x1^2 - 1) : 
  p = -3 / 8 :=
sorry

theorem part_b_case2 (x1 x2 p : ℝ) 
  (h1 : 4 * (x1 ^ 2) + x1 + 4 * p = 0)
  (h2 : 4 * (x2 ^ 2) + x2 + 4 * p = 0) 
  (h3 : x2 = x1^2 - 1) : 
  p = -15 / 8 :=
sorry

end NUMINAMATH_GPT_part_a_part_b_case1_part_b_case2_l2257_225722


namespace NUMINAMATH_GPT_largest_divisor_n_l2257_225787

theorem largest_divisor_n (n : ℕ) (h₁ : n > 0) (h₂ : 650 ∣ n^3) : 130 ∣ n :=
sorry

end NUMINAMATH_GPT_largest_divisor_n_l2257_225787


namespace NUMINAMATH_GPT_xy_sum_l2257_225725

namespace ProofExample

variable (x y : ℚ)

def condition1 : Prop := (1 / x) + (1 / y) = 4
def condition2 : Prop := (1 / x) - (1 / y) = -6

theorem xy_sum : condition1 x y → condition2 x y → (x + y = -4 / 5) := by
  intros
  sorry

end ProofExample

end NUMINAMATH_GPT_xy_sum_l2257_225725


namespace NUMINAMATH_GPT_cats_in_house_l2257_225785

-- Define the conditions
def total_cats (C : ℕ) : Prop :=
  let num_white_cats := 2
  let num_black_cats := C / 4
  let num_grey_cats := 10
  C = num_white_cats + num_black_cats + num_grey_cats

-- State the theorem
theorem cats_in_house : ∃ C : ℕ, total_cats C ∧ C = 16 := 
by
  sorry

end NUMINAMATH_GPT_cats_in_house_l2257_225785


namespace NUMINAMATH_GPT_line_passes_second_and_third_quadrants_l2257_225735

theorem line_passes_second_and_third_quadrants 
  (a b c p : ℝ)
  (h1 : a * b * c ≠ 0)
  (h2 : (a + b) / c = p)
  (h3 : (b + c) / a = p)
  (h4 : (c + a) / b = p) :
  ∀ (x y : ℝ), y = p * x + p → 
  ((x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) :=
sorry

end NUMINAMATH_GPT_line_passes_second_and_third_quadrants_l2257_225735


namespace NUMINAMATH_GPT_cost_price_of_item_l2257_225731

theorem cost_price_of_item 
  (retail_price : ℝ) (reduction_percentage : ℝ) 
  (additional_discount : ℝ) (profit_percentage : ℝ) 
  (selling_price : ℝ) 
  (h1 : retail_price = 900)
  (h2 : reduction_percentage = 0.1)
  (h3 : additional_discount = 48)
  (h4 : profit_percentage = 0.2)
  (h5 : selling_price = 762) :
  ∃ x : ℝ, selling_price = 1.2 * x ∧ x = 635 := 
by {
  sorry
}

end NUMINAMATH_GPT_cost_price_of_item_l2257_225731


namespace NUMINAMATH_GPT_addition_correct_l2257_225757

theorem addition_correct :
  1357 + 2468 + 3579 + 4680 + 5791 = 17875 := 
by
  sorry

end NUMINAMATH_GPT_addition_correct_l2257_225757


namespace NUMINAMATH_GPT_f_properties_l2257_225713

variable (f : ℝ → ℝ)
variable (f_pos : ∀ x : ℝ, f x > 0)
variable (f_eq : ∀ a b : ℝ, f a * f b = f (a + b))

theorem f_properties :
  (f 0 = 1) ∧
  (∀ a : ℝ, f (-a) = 1 / f a) ∧
  (∀ a : ℝ, f a = (f (3 * a))^(1/3)) :=
by {
  sorry
}

end NUMINAMATH_GPT_f_properties_l2257_225713


namespace NUMINAMATH_GPT_find_B_l2257_225745

theorem find_B (A C B : ℕ) (hA : A = 520) (hC : C = A + 204) (hCB : C = B + 179) : B = 545 :=
by
  sorry

end NUMINAMATH_GPT_find_B_l2257_225745


namespace NUMINAMATH_GPT_rectangle_area_l2257_225703

theorem rectangle_area (x : ℝ) (w : ℝ) (h : ℝ) (H1 : x^2 = w^2 + h^2) (H2 : h = 3 * w) : 
  (w * h = (3 * x^2) / 10) :=
by sorry

end NUMINAMATH_GPT_rectangle_area_l2257_225703


namespace NUMINAMATH_GPT_negation_of_exists_lt_l2257_225756

theorem negation_of_exists_lt :
  (¬ ∃ x : ℝ, x^2 + 2 * x + 3 < 0) = (∀ x : ℝ, x^2 + 2 * x + 3 ≥ 0) :=
by sorry

end NUMINAMATH_GPT_negation_of_exists_lt_l2257_225756


namespace NUMINAMATH_GPT_find_f_comp_f_l2257_225755

def f (x : ℚ) : ℚ :=
  if x ≤ 1 then x + 1 else -x + 3

theorem find_f_comp_f (h : f (f (5/2)) = 3/2) :
  f (f (5/2)) = 3/2 := by
  sorry

end NUMINAMATH_GPT_find_f_comp_f_l2257_225755


namespace NUMINAMATH_GPT_color_opposite_lightgreen_is_red_l2257_225701

-- Define the colors
inductive Color
| Red | White | Green | Brown | LightGreen | Purple

open Color

-- Define the condition
def is_opposite (a b : Color) : Prop := sorry

-- Main theorem
theorem color_opposite_lightgreen_is_red :
  is_opposite LightGreen Red :=
sorry

end NUMINAMATH_GPT_color_opposite_lightgreen_is_red_l2257_225701


namespace NUMINAMATH_GPT_total_games_played_l2257_225746

-- Defining the conditions
def games_won : ℕ := 18
def games_lost : ℕ := games_won + 21

-- Problem statement
theorem total_games_played : games_won + games_lost = 57 := by
  sorry

end NUMINAMATH_GPT_total_games_played_l2257_225746


namespace NUMINAMATH_GPT_inequality_positive_reals_l2257_225778

theorem inequality_positive_reals (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  (x / (1 + x^2)) + (y / (1 + y^2)) + (z / (1 + z^2)) ≤ (3 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_GPT_inequality_positive_reals_l2257_225778


namespace NUMINAMATH_GPT_jogger_ahead_distance_l2257_225791

-- Definitions of conditions
def jogger_speed : ℝ := 9  -- km/hr
def train_speed : ℝ := 45  -- km/hr
def train_length : ℝ := 150  -- meters
def passing_time : ℝ := 39  -- seconds

-- The main statement that we want to prove
theorem jogger_ahead_distance : 
  let relative_speed := (train_speed - jogger_speed) * (5 / 18)  -- conversion to m/s
  let distance_covered := relative_speed * passing_time
  let jogger_ahead := distance_covered - train_length
  jogger_ahead = 240 :=
by
  sorry

end NUMINAMATH_GPT_jogger_ahead_distance_l2257_225791


namespace NUMINAMATH_GPT_min_value_p_plus_q_l2257_225729

theorem min_value_p_plus_q (p q : ℕ) (hp : 1 < p) (hq : 1 < q) 
  (h : 17 * (p + 1) = 20 * (q + 1)) : p + q = 37 :=
sorry

end NUMINAMATH_GPT_min_value_p_plus_q_l2257_225729


namespace NUMINAMATH_GPT_expression_undefined_iff_l2257_225769

theorem expression_undefined_iff (a : ℝ) : (a^2 - 9 = 0) ↔ (a = 3 ∨ a = -3) :=
sorry

end NUMINAMATH_GPT_expression_undefined_iff_l2257_225769


namespace NUMINAMATH_GPT_stella_profit_loss_l2257_225723

theorem stella_profit_loss :
  let dolls := 6
  let clocks := 4
  let glasses := 8
  let vases := 3
  let postcards := 10
  let dolls_price := 8
  let clocks_price := 25
  let glasses_price := 6
  let vases_price := 12
  let postcards_price := 3
  let cost := 250
  let clocks_discount_threshold := 2
  let clocks_discount := 10 / 100
  let glasses_bundle := 3
  let glasses_bundle_price := 2 * glasses_price
  let sales_tax_rate := 5 / 100
  let dolls_revenue := dolls * dolls_price
  let clocks_revenue_full := clocks * clocks_price
  let clocks_discounts_count := clocks / clocks_discount_threshold
  let clocks_discount_amount := clocks_discounts_count * clocks_discount * clocks_discount_threshold * clocks_price
  let clocks_revenue := clocks_revenue_full - clocks_discount_amount
  let glasses_discount_quantity := glasses / glasses_bundle
  let glasses_revenue := (glasses - glasses_discount_quantity) * glasses_price
  let vases_revenue := vases * vases_price
  let postcards_revenue := postcards * postcards_price
  let total_revenue_without_discounts := dolls_revenue + clocks_revenue_full + glasses_revenue + vases_revenue + postcards_revenue
  let total_revenue_with_discounts := dolls_revenue + clocks_revenue + glasses_revenue + vases_revenue + postcards_revenue
  let sales_tax := sales_tax_rate * total_revenue_with_discounts
  let profit := total_revenue_with_discounts - cost - sales_tax
  profit = -17.25 := by sorry

end NUMINAMATH_GPT_stella_profit_loss_l2257_225723


namespace NUMINAMATH_GPT_cars_needed_to_double_earnings_l2257_225716

-- Define the conditions
def baseSalary : Int := 1000
def commissionPerCar : Int := 200
def januaryEarnings : Int := 1800

-- The proof goal
theorem cars_needed_to_double_earnings : 
  ∃ (carsSoldInFeb : Int), 
    1000 + commissionPerCar * carsSoldInFeb = 2 * januaryEarnings :=
by
  sorry

end NUMINAMATH_GPT_cars_needed_to_double_earnings_l2257_225716


namespace NUMINAMATH_GPT_find_f_g_3_l2257_225793

def f (x : ℝ) : ℝ := x^2 + 2
def g (x : ℝ) : ℝ := 3 * x - 2

theorem find_f_g_3 : f (g 3) = 51 := 
by 
  sorry

end NUMINAMATH_GPT_find_f_g_3_l2257_225793


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_iff_l2257_225796

theorem quadratic_has_two_distinct_real_roots_iff (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 * x1 - 2 * x1 + k - 1 = 0 ∧ x2 * x2 - 2 * x2 + k - 1 = 0) ↔ k < 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_iff_l2257_225796


namespace NUMINAMATH_GPT_find_a_l2257_225738

theorem find_a (a b c : ℕ) (h1 : a ≥ b ∧ b ≥ c)  
  (h2 : (a:ℤ) ^ 2 - b ^ 2 - c ^ 2 + a * b = 2011) 
  (h3 : (a:ℤ) ^ 2 + 3 * b ^ 2 + 3 * c ^ 2 - 3 * a * b - 2 * a * c - 2 * b * c = -1997) : 
a = 253 := 
sorry

end NUMINAMATH_GPT_find_a_l2257_225738


namespace NUMINAMATH_GPT_circle_tangent_to_yaxis_and_line_l2257_225792

theorem circle_tangent_to_yaxis_and_line :
  (∃ C : ℝ → ℝ → Prop, 
    (∀ x y r : ℝ, C x y ↔ (x - 3) ^ 2 + (y - 2) ^ 2 = 9 ∨ (x + 1 / 3) ^ 2 + (y - 2) ^ 2 = 1 / 9) ∧ 
    (∀ y : ℝ, C 0 y → y = 2) ∧ 
    (∀ x y: ℝ, C x y → (∃ x1 : ℝ, 4 * x - 3 * y + 9 = 0 → 4 * x1 + 3 = 0))) :=
sorry

end NUMINAMATH_GPT_circle_tangent_to_yaxis_and_line_l2257_225792


namespace NUMINAMATH_GPT_first_person_job_completion_time_l2257_225708

noncomputable def job_completion_time :=
  let A := 1 - (1/5)
  let C := 1/8
  let combined_rate := A + C
  have h1 : combined_rate = 0.325 := by
    sorry
  have h2 : A ≠ 0 := by
    sorry
  (1 / A : ℝ)
  
theorem first_person_job_completion_time :
  job_completion_time = 1.25 :=
by
  sorry

end NUMINAMATH_GPT_first_person_job_completion_time_l2257_225708


namespace NUMINAMATH_GPT_count_perfect_cubes_l2257_225734

theorem count_perfect_cubes (a b : ℤ) (h₁ : 100 < a) (h₂ : b < 1000) : 
  ∃ n m : ℤ, (n^3 > 100 ∧ m^3 < 1000) ∧ m - n + 1 = 5 := 
by
  -- The proof is omitted and replaced with sorry.
  sorry

end NUMINAMATH_GPT_count_perfect_cubes_l2257_225734


namespace NUMINAMATH_GPT_heidi_paints_fraction_in_10_minutes_l2257_225714

variable (Heidi_paint_rate : ℕ → ℝ)
variable (t : ℕ)
variable (fraction : ℝ)

theorem heidi_paints_fraction_in_10_minutes 
  (h1 : Heidi_paint_rate 30 = 1) 
  (h2 : t = 10) 
  (h3 : fraction = 1 / 3) : 
  Heidi_paint_rate t = fraction := 
sorry

end NUMINAMATH_GPT_heidi_paints_fraction_in_10_minutes_l2257_225714


namespace NUMINAMATH_GPT_chess_tournament_participants_l2257_225717

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 171) : n = 19 := 
by sorry

end NUMINAMATH_GPT_chess_tournament_participants_l2257_225717


namespace NUMINAMATH_GPT_teachers_no_conditions_percentage_l2257_225751

theorem teachers_no_conditions_percentage :
  let total_teachers := 150
  let high_blood_pressure := 90
  let heart_trouble := 60
  let both_hbp_ht := 30
  let diabetes := 10
  let both_diabetes_ht := 5
  let both_diabetes_hbp := 8
  let all_three := 3

  let only_hbp := high_blood_pressure - both_hbp_ht - both_diabetes_hbp - all_three
  let only_ht := heart_trouble - both_hbp_ht - both_diabetes_ht - all_three
  let only_diabetes := diabetes - both_diabetes_hbp - both_diabetes_ht - all_three
  let both_hbp_ht_only := both_hbp_ht - all_three
  let both_hbp_diabetes_only := both_diabetes_hbp - all_three
  let both_ht_diabetes_only := both_diabetes_ht - all_three
  let any_condition := only_hbp + only_ht + only_diabetes + both_hbp_ht_only + both_hbp_diabetes_only + both_ht_diabetes_only + all_three
  let no_conditions := total_teachers - any_condition

  (no_conditions / total_teachers * 100) = 28 :=
by
  sorry

end NUMINAMATH_GPT_teachers_no_conditions_percentage_l2257_225751


namespace NUMINAMATH_GPT_figure_perimeter_l2257_225742

-- Define the side length of the square and the triangles.
def square_side_length : ℕ := 3
def triangle_side_length : ℕ := 2

-- Calculate the perimeter of the figure
def perimeter (a b : ℕ) : ℕ := 2 * a + 2 * b

-- Statement to prove
theorem figure_perimeter : perimeter square_side_length triangle_side_length = 10 := 
by 
  -- "sorry" denotes that the proof is omitted.
  sorry

end NUMINAMATH_GPT_figure_perimeter_l2257_225742


namespace NUMINAMATH_GPT_green_space_equation_l2257_225724

theorem green_space_equation (x : ℝ) (h_area : x * (x - 30) = 1000) :
  x * (x - 30) = 1000 := 
by
  exact h_area

end NUMINAMATH_GPT_green_space_equation_l2257_225724


namespace NUMINAMATH_GPT_temperature_range_for_5_percent_deviation_l2257_225759

noncomputable def approx_formula (C : ℝ) : ℝ := 2 * C + 30
noncomputable def exact_formula (C : ℝ) : ℝ := (9/5 : ℝ) * C + 32
noncomputable def deviation (C : ℝ) : ℝ := approx_formula C - exact_formula C
noncomputable def percentage_deviation (C : ℝ) : ℝ := abs (deviation C / exact_formula C)

theorem temperature_range_for_5_percent_deviation :
  ∀ (C : ℝ), 1 + 11 / 29 ≤ C ∧ C ≤ 32 + 8 / 11 ↔ percentage_deviation C ≤ 0.05 := sorry

end NUMINAMATH_GPT_temperature_range_for_5_percent_deviation_l2257_225759


namespace NUMINAMATH_GPT_badges_before_exchange_l2257_225795

theorem badges_before_exchange (V T : ℕ) (h1 : V = T + 5) (h2 : 76 * V + 20 * T = 80 * T + 24 * V - 100) :
  V = 50 ∧ T = 45 :=
by
  sorry

end NUMINAMATH_GPT_badges_before_exchange_l2257_225795


namespace NUMINAMATH_GPT_distance_covered_by_train_l2257_225766

-- Define the average speed and the total duration of the journey
def speed : ℝ := 10
def time : ℝ := 8

-- Use these definitions to state and prove the distance covered by the train
theorem distance_covered_by_train : speed * time = 80 := by
  sorry

end NUMINAMATH_GPT_distance_covered_by_train_l2257_225766


namespace NUMINAMATH_GPT_incorrect_conclusion_l2257_225733

noncomputable def data_set : List ℕ := [4, 1, 6, 2, 9, 5, 8]
def mean_x : ℝ := 2
def mean_y : ℝ := 20
def regression_eq (x : ℝ) : ℝ := 9.1 * x + 1.8
def chi_squared_value : ℝ := 9.632
def alpha : ℝ := 0.001
def critical_value : ℝ := 10.828

theorem incorrect_conclusion : ¬(chi_squared_value ≥ critical_value) := by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_incorrect_conclusion_l2257_225733


namespace NUMINAMATH_GPT_mutually_exclusive_event_of_hitting_target_at_least_once_l2257_225798

-- Definitions from conditions
def two_shots_fired : Prop := true

def complementary_events (E F : Prop) : Prop :=
  E ∨ F ∧ ¬(E ∧ F)

def hitting_target_at_least_once : Prop := true -- Placeholder for the event of hitting at least one target
def both_shots_miss : Prop := true              -- Placeholder for the event that both shots miss

-- Statement to prove
theorem mutually_exclusive_event_of_hitting_target_at_least_once
  (h1 : two_shots_fired)
  (h2 : complementary_events hitting_target_at_least_once both_shots_miss) :
  hitting_target_at_least_once = ¬both_shots_miss := 
sorry

end NUMINAMATH_GPT_mutually_exclusive_event_of_hitting_target_at_least_once_l2257_225798


namespace NUMINAMATH_GPT_inclination_angle_of_line_l2257_225700

theorem inclination_angle_of_line (m : ℝ) (b : ℝ) (h : b = -3) (h_line : ∀ x : ℝ, x - 3 = m * x + b) : 
  (Real.arctan m * 180 / Real.pi) = 45 := 
by sorry

end NUMINAMATH_GPT_inclination_angle_of_line_l2257_225700


namespace NUMINAMATH_GPT_no_solution_frac_eq_l2257_225754

theorem no_solution_frac_eq (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1) : 
  3 / x + 6 / (x - 1) - (x + 5) / (x * (x - 1)) ≠ 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_no_solution_frac_eq_l2257_225754


namespace NUMINAMATH_GPT_total_profit_is_2560_l2257_225777

noncomputable def basicWashPrice : ℕ := 5
noncomputable def deluxeWashPrice : ℕ := 10
noncomputable def premiumWashPrice : ℕ := 15

noncomputable def basicCarsWeekday : ℕ := 50
noncomputable def deluxeCarsWeekday : ℕ := 40
noncomputable def premiumCarsWeekday : ℕ := 20

noncomputable def employeeADailyWage : ℕ := 110
noncomputable def employeeBDailyWage : ℕ := 90
noncomputable def employeeCDailyWage : ℕ := 100
noncomputable def employeeDDailyWage : ℕ := 80

noncomputable def operatingExpenseWeekday : ℕ := 200

noncomputable def totalProfit : ℕ := 
  let revenueWeekday := (basicCarsWeekday * basicWashPrice) + 
                        (deluxeCarsWeekday * deluxeWashPrice) + 
                        (premiumCarsWeekday * premiumWashPrice)
  let totalRevenue := revenueWeekday * 5
  let wageA := employeeADailyWage * 5
  let wageB := employeeBDailyWage * 2
  let wageC := employeeCDailyWage * 3
  let wageD := employeeDDailyWage * 2
  let totalWages := wageA + wageB + wageC + wageD
  let totalOperatingExpenses := operatingExpenseWeekday * 5
  totalRevenue - (totalWages + totalOperatingExpenses)

theorem total_profit_is_2560 : totalProfit = 2560 := by
  sorry

end NUMINAMATH_GPT_total_profit_is_2560_l2257_225777


namespace NUMINAMATH_GPT_find_values_l2257_225727

theorem find_values (a b : ℕ) (h1 : (a + 1) % b = 0) (h2 : a = 2 * b + 5) (h3 : Nat.Prime (a + 7 * b)) : (a = 9 ∧ b = 2) ∨ (a = 17 ∧ b = 6) :=
sorry

end NUMINAMATH_GPT_find_values_l2257_225727


namespace NUMINAMATH_GPT_latus_rectum_of_parabola_l2257_225784

theorem latus_rectum_of_parabola : 
  ∀ x y : ℝ, x^2 = -y → y = 1/4 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_latus_rectum_of_parabola_l2257_225784


namespace NUMINAMATH_GPT_no_solution_for_inequalities_l2257_225783

theorem no_solution_for_inequalities (x : ℝ) :
  ¬(5 * x^2 - 7 * x + 1 < 0 ∧ x^2 - 9 * x + 30 < 0) :=
sorry

end NUMINAMATH_GPT_no_solution_for_inequalities_l2257_225783


namespace NUMINAMATH_GPT_geo_seq_fifth_term_l2257_225720

theorem geo_seq_fifth_term (a r : ℝ) (a_pos : 0 < a) (r_pos : 0 < r)
  (h3 : a * r^2 = 8) (h7 : a * r^6 = 18) : a * r^4 = 12 :=
sorry

end NUMINAMATH_GPT_geo_seq_fifth_term_l2257_225720


namespace NUMINAMATH_GPT_compute_factorial_expression_l2257_225732

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

theorem compute_factorial_expression :
  factorial 9 - factorial 8 - factorial 7 + factorial 6 = 318240 := by
  sorry

end NUMINAMATH_GPT_compute_factorial_expression_l2257_225732


namespace NUMINAMATH_GPT_fill_entire_bucket_l2257_225706

theorem fill_entire_bucket (h : (2/3 : ℝ) * t = 2) : t = 3 :=
sorry

end NUMINAMATH_GPT_fill_entire_bucket_l2257_225706


namespace NUMINAMATH_GPT_rod_length_l2257_225712

theorem rod_length (L : ℝ) (weight : ℝ → ℝ) (weight_6m : weight 6 = 14.04) (weight_L : weight L = 23.4) :
  L = 10 :=
by 
  sorry

end NUMINAMATH_GPT_rod_length_l2257_225712


namespace NUMINAMATH_GPT_mean_temperature_l2257_225715

def temperatures : List ℝ := [-6.5, -2, -3.5, -1, 0.5, 4, 1.5]

theorem mean_temperature : (temperatures.sum / temperatures.length) = -1 := by
  sorry

end NUMINAMATH_GPT_mean_temperature_l2257_225715


namespace NUMINAMATH_GPT_x_squared_minus_y_squared_l2257_225743

theorem x_squared_minus_y_squared (x y : ℚ) (h1 : x + y = 2 / 5) (h2 : x - y = 1 / 10) : x ^ 2 - y ^ 2 = 1 / 25 :=
by
  sorry

end NUMINAMATH_GPT_x_squared_minus_y_squared_l2257_225743


namespace NUMINAMATH_GPT_swimmers_meeting_times_l2257_225764

theorem swimmers_meeting_times (l : ℕ) (vA vB t : ℕ) (T : ℝ) :
  l = 120 →
  vA = 4 →
  vB = 3 →
  t = 15 →
  T = 21 :=
  sorry

end NUMINAMATH_GPT_swimmers_meeting_times_l2257_225764


namespace NUMINAMATH_GPT_minimum_a_l2257_225762

open Real

theorem minimum_a (a : ℝ) : (∀ (x y : ℝ), x > 0 → y > 0 → (1 / x + a / y) ≥ (16 / (x + y))) → a ≥ 9 := by
sorry

end NUMINAMATH_GPT_minimum_a_l2257_225762


namespace NUMINAMATH_GPT_Vovochka_correct_pairs_count_l2257_225781

def no_carry_pairs_count (digit_sum: ℕ → ℕ → Prop) : ℕ :=
  let count_pairs (lim: ℕ) : ℕ := (lim * (lim + 1)) / 2
  let digit_valid_pairs : ℕ := count_pairs 9
  (digit_valid_pairs * digit_valid_pairs) * 81

def digit_sum (x y: ℕ) : Prop := (x + y ≤ 9)

theorem Vovochka_correct_pairs_count :
  no_carry_pairs_count digit_sum = 244620 := by
  sorry

end NUMINAMATH_GPT_Vovochka_correct_pairs_count_l2257_225781


namespace NUMINAMATH_GPT_no_int_satisfies_both_congruences_l2257_225799

theorem no_int_satisfies_both_congruences :
  ¬ ∃ n : ℤ, (n ≡ 5 [ZMOD 6]) ∧ (n ≡ 1 [ZMOD 21]) :=
sorry

end NUMINAMATH_GPT_no_int_satisfies_both_congruences_l2257_225799


namespace NUMINAMATH_GPT_prob_at_least_one_wrong_l2257_225702

-- Defining the conditions in mathlib
def prob_wrong : ℝ := 0.1
def num_questions : ℕ := 3

-- Proving the main statement
theorem prob_at_least_one_wrong : 1 - (1 - prob_wrong) ^ num_questions = 0.271 := by
  sorry

end NUMINAMATH_GPT_prob_at_least_one_wrong_l2257_225702


namespace NUMINAMATH_GPT_find_k_l2257_225748

-- Definitions for the vectors and collinearity condition.

def vector := ℝ × ℝ

def collinear (v1 v2 : vector) : Prop :=
  v1.1 * v2.2 - v1.2 * v2.1 = 0

-- Given vectors a and b.
def a (k : ℝ) : vector := (1, k)
def b : vector := (2, 2)

-- Vector addition.
def add (v1 v2 : vector) : vector := (v1.1 + v2.1, v1.2 + v2.2)

-- Problem statement
theorem find_k (k : ℝ) (h : collinear (add (a k) b) (a k)) : k = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l2257_225748


namespace NUMINAMATH_GPT_jacob_initial_fish_count_l2257_225747

theorem jacob_initial_fish_count : 
  ∃ J : ℕ, 
    (∀ A : ℕ, A = 7 * J) → 
    (A' = A - 23) → 
    (J + 26 = A' + 1) → 
    J = 8 := 
by 
  sorry

end NUMINAMATH_GPT_jacob_initial_fish_count_l2257_225747


namespace NUMINAMATH_GPT_expression_simplification_l2257_225718

open Real

theorem expression_simplification (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : 3*x + y / 3 ≠ 0) :
  (3*x + y/3)⁻¹ * ((3*x)⁻¹ + (y/3)⁻¹) = 1 / (3 * (x * y)) :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_expression_simplification_l2257_225718


namespace NUMINAMATH_GPT_age_of_Rahim_l2257_225790

theorem age_of_Rahim (R : ℕ) (h1 : ∀ (a : ℕ), a = (R + 1) → (a + 5) = (2 * R)) (h2 : ∀ (a : ℕ), a = (R + 1) → a = R + 1) :
  R = 6 := by
  sorry

end NUMINAMATH_GPT_age_of_Rahim_l2257_225790


namespace NUMINAMATH_GPT_lcm_of_4_8_9_10_l2257_225780

theorem lcm_of_4_8_9_10 : Nat.lcm (Nat.lcm 4 8) (Nat.lcm 9 10) = 360 := by
  sorry

end NUMINAMATH_GPT_lcm_of_4_8_9_10_l2257_225780


namespace NUMINAMATH_GPT_andrew_current_age_l2257_225737

-- Definitions based on conditions.
def initial_age := 11  -- Andrew started donating at age 11
def donation_per_year := 7  -- Andrew donates 7k each year on his birthday
def total_donation := 133  -- Andrew has donated a total of 133k till now

-- The theorem stating the problem and the conclusion.
theorem andrew_current_age : 
  ∃ (A : ℕ), donation_per_year * (A - initial_age) = total_donation :=
by {
  sorry
}

end NUMINAMATH_GPT_andrew_current_age_l2257_225737


namespace NUMINAMATH_GPT_inequality_f_l2257_225707

noncomputable def f (x y z : ℝ) : ℝ :=
  x * y + y * z + z * x - 2 * x * y * z

theorem inequality_f (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 1) :
  0 ≤ f x y z ∧ f x y z ≤ 7 / 27 :=
  sorry

end NUMINAMATH_GPT_inequality_f_l2257_225707


namespace NUMINAMATH_GPT_common_property_rhombus_rectangle_diagonals_l2257_225771

-- Define a structure for Rhombus and its property
structure Rhombus (R : Type) :=
  (diagonals_perpendicular : Prop)
  (diagonals_bisect : Prop)

-- Define a structure for Rectangle and its property
structure Rectangle (R : Type) :=
  (diagonals_equal_length : Prop)
  (diagonals_bisect : Prop)

-- Define the theorem that states the common property between diagonals of both shapes
theorem common_property_rhombus_rectangle_diagonals (R : Type) 
  (rhombus_properties : Rhombus R) 
  (rectangle_properties : Rectangle R) :
  rhombus_properties.diagonals_bisect ∧ rectangle_properties.diagonals_bisect :=
by {
  -- Since the solution steps are not to be included, we conclude the proof with 'sorry'
  sorry
}

end NUMINAMATH_GPT_common_property_rhombus_rectangle_diagonals_l2257_225771


namespace NUMINAMATH_GPT_average_speed_for_remaining_part_l2257_225763

theorem average_speed_for_remaining_part (D : ℝ) (v : ℝ) 
  (h1 : 0.8 * D / 80 + 0.2 * D / v = D / 50) : v = 20 :=
sorry

end NUMINAMATH_GPT_average_speed_for_remaining_part_l2257_225763


namespace NUMINAMATH_GPT_geometric_series_sum_frac_l2257_225773

open BigOperators

theorem geometric_series_sum_frac (q : ℚ) (a1 : ℚ) (a_list: List ℚ) (h_theta : q = 1 / 2) 
(h_a_list : a_list ⊆ [-4, -3, -2, 0, 1, 23, 4]) : 
  a1 * (1 + q^5) / (1 - q) = 33 / 4 := by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_frac_l2257_225773


namespace NUMINAMATH_GPT_find_real_pairs_l2257_225730

theorem find_real_pairs (x y : ℝ) (h : 2 * x / (1 + x^2) = (1 + y^2) / (2 * y)) : 
  (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) :=
sorry

end NUMINAMATH_GPT_find_real_pairs_l2257_225730


namespace NUMINAMATH_GPT_find_increase_in_perimeter_l2257_225797

variable (L B y : ℕ)

theorem find_increase_in_perimeter (h1 : 2 * (L + y + (B + y)) = 2 * (L + B) + 16) : y = 4 := by
  sorry

end NUMINAMATH_GPT_find_increase_in_perimeter_l2257_225797


namespace NUMINAMATH_GPT_inequality_holds_l2257_225709

variable {f : ℝ → ℝ}

-- Conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_monotonic_on_nonneg_interval (f : ℝ → ℝ) : Prop := ∀ x y, (0 ≤ x ∧ x < y ∧ y < 8) → f y ≤ f x

axiom condition1 : is_even f
axiom condition2 : is_monotonic_on_nonneg_interval f
axiom condition3 : f (-3) < f 2

-- The statement to be proven
theorem inequality_holds : f 5 < f (-3) ∧ f (-3) < f (-1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l2257_225709


namespace NUMINAMATH_GPT_min_value_of_a_plus_2b_l2257_225736

theorem min_value_of_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) : a + 2*b = 3 + 2*Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_min_value_of_a_plus_2b_l2257_225736


namespace NUMINAMATH_GPT_speed_of_A_l2257_225768
-- Import necessary library

-- Define conditions
def initial_distance : ℝ := 25  -- initial distance between A and B
def speed_B : ℝ := 13  -- speed of B in kmph
def meeting_time : ℝ := 1  -- time duration in hours

-- The speed of A which is to be proven
def speed_A : ℝ := 12

-- The theorem to be proved
theorem speed_of_A (d : ℝ) (vB : ℝ) (t : ℝ) (vA : ℝ) : d = 25 → vB = 13 → t = 1 → 
  d = vA * t + vB * t → vA = 12 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  -- Enforcing the statement to be proved
  have := Eq.symm h4
  simp [speed_A, *] at *
  sorry

end NUMINAMATH_GPT_speed_of_A_l2257_225768


namespace NUMINAMATH_GPT_evaluate_expression_l2257_225739

theorem evaluate_expression (x y : ℕ) (h1 : x = 2) (h2 : y = 3) : 3 * x^y + 4 * y^x = 60 :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2257_225739


namespace NUMINAMATH_GPT_complement_P_l2257_225767

def U : Set ℝ := Set.univ

def P : Set ℝ := {x | x^2 < 1}

theorem complement_P : (U \ P) = Set.Iic (-1) ∪ Set.Ici 1 := by
  sorry

end NUMINAMATH_GPT_complement_P_l2257_225767


namespace NUMINAMATH_GPT_find_x_l2257_225711

theorem find_x (b x : ℝ) (hb : b > 1) (hx : x > 0)
  (h : (3 * x)^(Real.log 3 / Real.log b) - (5 * x)^(Real.log 5 / Real.log b) = 0) :
  x = 1 / 5^(Real.log b / Real.log (5 / 3)) :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2257_225711


namespace NUMINAMATH_GPT_machine_A_production_rate_l2257_225786

theorem machine_A_production_rate :
  ∀ (A B T_A T_B : ℝ),
    500 = A * T_A →
    500 = B * T_B →
    B = 1.25 * A →
    T_A = T_B + 15 →
    A = 100 / 15 :=
by
  intros A B T_A T_B hA hB hRate hTime
  sorry

end NUMINAMATH_GPT_machine_A_production_rate_l2257_225786


namespace NUMINAMATH_GPT_total_daisies_l2257_225704

theorem total_daisies (white pink red : ℕ) (h1 : pink = 9 * white) (h2 : red = 4 * pink - 3) (h3 : white = 6) : 
    white + pink + red = 273 :=
by
  sorry

end NUMINAMATH_GPT_total_daisies_l2257_225704


namespace NUMINAMATH_GPT_grace_age_l2257_225774

/-- Grace's age calculation based on given family ages. -/
theorem grace_age (mother_age : ℕ) (grandmother_age : ℕ) (grace_age : ℕ)
  (h1 : mother_age = 80)
  (h2 : grandmother_age = 2 * mother_age)
  (h3 : grace_age = (3 * grandmother_age) / 8) : grace_age = 60 :=
by
  sorry

end NUMINAMATH_GPT_grace_age_l2257_225774


namespace NUMINAMATH_GPT_tank_overflows_after_24_minutes_l2257_225710

theorem tank_overflows_after_24_minutes 
  (rateA : ℝ) (rateB : ℝ) (t : ℝ) 
  (hA : rateA = 1) 
  (hB : rateB = 4) :
  t - 1/4 * rateB + t * rateA = 1 → t = 2/5 :=
by 
  intros h
  -- the proof steps go here
  sorry

end NUMINAMATH_GPT_tank_overflows_after_24_minutes_l2257_225710


namespace NUMINAMATH_GPT_calc_expr_l2257_225719

theorem calc_expr : 
  (-1: ℝ)^4 - 2 * Real.tan (Real.pi / 3) + (Real.sqrt 3 - Real.sqrt 2)^0 + Real.sqrt 12 = 2 := 
by
  sorry

end NUMINAMATH_GPT_calc_expr_l2257_225719


namespace NUMINAMATH_GPT_log_sum_l2257_225740

open Real

theorem log_sum : log 2 + log 5 = 1 :=
sorry

end NUMINAMATH_GPT_log_sum_l2257_225740


namespace NUMINAMATH_GPT_number_of_valid_x_l2257_225761

theorem number_of_valid_x (x : ℕ) : 
  ((x + 3) * (x - 3) * (x ^ 2 + 9) < 500) ∧ (x - 3 > 0) ↔ x = 4 :=
sorry

end NUMINAMATH_GPT_number_of_valid_x_l2257_225761


namespace NUMINAMATH_GPT_power_of_power_rule_l2257_225749

theorem power_of_power_rule (h : 128 = 2^7) : (128: ℝ)^(4/7) = 16 := by
  sorry

end NUMINAMATH_GPT_power_of_power_rule_l2257_225749


namespace NUMINAMATH_GPT_value_of_N_l2257_225772

theorem value_of_N (N : ℕ) (h : Nat.choose N 5 = 231) : N = 11 := sorry

end NUMINAMATH_GPT_value_of_N_l2257_225772


namespace NUMINAMATH_GPT_Jackie_apples_count_l2257_225744

variable (Adam_apples Jackie_apples : ℕ)
variable (h1 : Adam_apples = 10)
variable (h2 : Adam_apples = Jackie_apples + 8)

theorem Jackie_apples_count : Jackie_apples = 2 := by
  sorry

end NUMINAMATH_GPT_Jackie_apples_count_l2257_225744


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l2257_225779

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℕ)
  (d : ℚ)
  (h_arith_seq : ∀ (n m : ℕ), (n > 0) → (m > 0) → (a n) / n - (a m) / m = (n - m) * d)
  (h_a3 : a 3 = 2)
  (h_a9 : a 9 = 12) :
  d = 1 / 9 ∧ a 12 = 20 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l2257_225779


namespace NUMINAMATH_GPT_variance_of_ξ_l2257_225726

noncomputable def probability_distribution (ξ : ℕ) : ℚ :=
  if ξ = 2 ∨ ξ = 4 ∨ ξ = 6 ∨ ξ = 8 ∨ ξ = 10 then 1/5 else 0

def expected_value (ξ_values : List ℕ) (prob : ℕ → ℚ) : ℚ :=
  ξ_values.map (λ ξ => ξ * prob ξ) |>.sum

def variance (ξ_values : List ℕ) (prob : ℕ → ℚ) (Eξ : ℚ) : ℚ :=
  ξ_values.map (λ ξ => prob ξ * (ξ - Eξ) ^ 2) |>.sum

theorem variance_of_ξ :
  let ξ_values := [2, 4, 6, 8, 10]
  let prob := probability_distribution
  let Eξ := expected_value ξ_values prob
  variance ξ_values prob Eξ = 8 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_variance_of_ξ_l2257_225726


namespace NUMINAMATH_GPT_sum_as_common_fraction_l2257_225794

/-- The sum of 0.2 + 0.04 + 0.006 + 0.0008 + 0.00010 as a common fraction -/
theorem sum_as_common_fraction : (0.2 + 0.04 + 0.006 + 0.0008 + 0.00010) = (12345 / 160000) := by
  sorry

end NUMINAMATH_GPT_sum_as_common_fraction_l2257_225794


namespace NUMINAMATH_GPT_merchant_problem_l2257_225753

theorem merchant_problem (P C : ℝ) (h1 : P + C = 60) (h2 : 2.40 * P + 6.00 * C = 180) : C = 10 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_merchant_problem_l2257_225753


namespace NUMINAMATH_GPT_bianca_made_after_selling_l2257_225788

def bianca_initial_cupcakes : ℕ := 14
def bianca_sold_cupcakes : ℕ := 6
def bianca_final_cupcakes : ℕ := 25

theorem bianca_made_after_selling :
  (bianca_initial_cupcakes - bianca_sold_cupcakes) + (bianca_final_cupcakes - (bianca_initial_cupcakes - bianca_sold_cupcakes)) = bianca_final_cupcakes :=
by
  sorry

end NUMINAMATH_GPT_bianca_made_after_selling_l2257_225788


namespace NUMINAMATH_GPT_find_investment_sum_l2257_225775

variable (P : ℝ)

def simple_interest (rate time : ℝ) (principal : ℝ) : ℝ :=
  principal * rate * time

theorem find_investment_sum (h : simple_interest 0.18 2 P - simple_interest 0.12 2 P = 240) :
  P = 2000 :=
by
  sorry

end NUMINAMATH_GPT_find_investment_sum_l2257_225775


namespace NUMINAMATH_GPT_beef_weight_loss_l2257_225760

theorem beef_weight_loss (weight_before weight_after: ℕ) 
                         (h1: weight_before = 400) 
                         (h2: weight_after = 240) : 
                         ((weight_before - weight_after) * 100 / weight_before = 40) :=
by 
  sorry

end NUMINAMATH_GPT_beef_weight_loss_l2257_225760


namespace NUMINAMATH_GPT_misha_grade_students_l2257_225705

theorem misha_grade_students (n : ℕ) (h1 : n = 75) (h2 : n = 75) : 2 * n - 1 = 149 := 
by
  sorry

end NUMINAMATH_GPT_misha_grade_students_l2257_225705


namespace NUMINAMATH_GPT_days_c_worked_l2257_225789

theorem days_c_worked 
    (days_a : ℕ) (days_b : ℕ) (wage_ratio_a : ℚ) (wage_ratio_b : ℚ) (wage_ratio_c : ℚ)
    (total_earnings : ℚ) (wage_c : ℚ) :
    days_a = 16 →
    days_b = 9 →
    wage_ratio_a = 3 →
    wage_ratio_b = 4 →
    wage_ratio_c = 5 →
    wage_c = 71.15384615384615 →
    total_earnings = 1480 →
    ∃ days_c : ℕ, (total_earnings = (wage_ratio_a / wage_ratio_c * wage_c * days_a) + 
                                 (wage_ratio_b / wage_ratio_c * wage_c * days_b) + 
                                 (wage_c * days_c)) ∧ days_c = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_days_c_worked_l2257_225789


namespace NUMINAMATH_GPT_circle_properties_l2257_225750

theorem circle_properties (D r C A : ℝ) (h1 : D = 15)
  (h2 : r = 7.5)
  (h3 : C = 15 * Real.pi)
  (h4 : A = 56.25 * Real.pi) :
  (9 ^ 2 + 12 ^ 2 = D ^ 2) ∧ (D = 2 * r) ∧ (C = Real.pi * D) ∧ (A = Real.pi * r ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_circle_properties_l2257_225750


namespace NUMINAMATH_GPT_gcd_of_power_of_two_plus_one_l2257_225758

theorem gcd_of_power_of_two_plus_one (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : m ≠ n) : 
  Nat.gcd (2^(2^m) + 1) (2^(2^n) + 1) = 1 := 
sorry

end NUMINAMATH_GPT_gcd_of_power_of_two_plus_one_l2257_225758


namespace NUMINAMATH_GPT_part1_part2_l2257_225752

def f (x : ℝ) (a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2*a + 1)

theorem part1 (x : ℝ) : f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 :=
by sorry

theorem part2 (a : ℝ) : (∀ x, f x a ≥ 4) ↔ (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end NUMINAMATH_GPT_part1_part2_l2257_225752


namespace NUMINAMATH_GPT_find_v_value_l2257_225770

theorem find_v_value (x : ℝ) (v : ℝ) (h1 : x = 3.0) (h2 : 5 * x + v = 19) : v = 4 := by
  sorry

end NUMINAMATH_GPT_find_v_value_l2257_225770


namespace NUMINAMATH_GPT_number_of_jars_pasta_sauce_l2257_225728

-- Conditions
def pasta_cost_per_kg := 1.5
def pasta_weight_kg := 2.0
def ground_beef_cost_per_kg := 8.0
def ground_beef_weight_kg := 1.0 / 4.0
def quesadilla_cost := 6.0
def jar_sauce_cost := 2.0
def total_money := 15.0

-- Helper definitions for total costs
def pasta_total_cost := pasta_weight_kg * pasta_cost_per_kg
def ground_beef_total_cost := ground_beef_weight_kg * ground_beef_cost_per_kg
def other_total_cost := quesadilla_cost + pasta_total_cost + ground_beef_total_cost
def remaining_money := total_money - other_total_cost

-- Proof statement
theorem number_of_jars_pasta_sauce :
  (remaining_money / jar_sauce_cost) = 2 := by
  sorry

end NUMINAMATH_GPT_number_of_jars_pasta_sauce_l2257_225728


namespace NUMINAMATH_GPT_parallel_lines_m_value_l2257_225721

/-- Given two lines l_1: (3 + m) * x + 4 * y = 5 - 3 * m, and l_2: 2 * x + (5 + m) * y = 8,
the value of m for which l_1 is parallel to l_2 is -7. -/
theorem parallel_lines_m_value (m : ℝ) :
  (∀ x y : ℝ, (3 + m) * x + 4 * y = 5 - 3 * m) →
  (∀ x y : ℝ, 2 * x + (5 + m) * y = 8) →
  m = -7 :=
sorry

end NUMINAMATH_GPT_parallel_lines_m_value_l2257_225721


namespace NUMINAMATH_GPT_valid_numbers_l2257_225741

noncomputable def is_valid_number (a : ℕ) : Prop :=
  ∃ b c d x y : ℕ, 
    a = b * c + d ∧
    a = 10 * x + y ∧
    x > 0 ∧ x ≤ 9 ∧ y ≤ 9 ∧
    10 * x + y = 4 * x + 4 * y

theorem valid_numbers : 
  ∃ a : ℕ, (a = 12 ∨ a = 24 ∨ a = 36 ∨ a = 48) ∧ is_valid_number a :=
by
  sorry

end NUMINAMATH_GPT_valid_numbers_l2257_225741
