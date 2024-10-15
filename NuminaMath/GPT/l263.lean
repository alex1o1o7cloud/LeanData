import Mathlib

namespace NUMINAMATH_GPT_find_k_value_l263_26345

theorem find_k_value (k : ℕ) :
  3 * 6 * 4 * k = Nat.factorial 8 → k = 560 :=
by
  sorry

end NUMINAMATH_GPT_find_k_value_l263_26345


namespace NUMINAMATH_GPT_collinear_points_sum_xy_solution_l263_26399

theorem collinear_points_sum_xy_solution (x y : ℚ)
  (h1 : (B : ℚ × ℚ) = (-2, y))
  (h2 : (A : ℚ × ℚ) = (x, 5))
  (h3 : (C : ℚ × ℚ) = (1, 1))
  (h4 : dist (B.1, B.2) (C.1, C.2) = 2 * dist (A.1, A.2) (C.1, C.2))
  (h5 : (y - 5) / (-2 - x) = (1 - 5) / (1 - x)) :
  x + y = -9 / 2 ∨ x + y = 17 / 2 :=
by sorry

end NUMINAMATH_GPT_collinear_points_sum_xy_solution_l263_26399


namespace NUMINAMATH_GPT_tv_price_with_tax_l263_26307

-- Define the original price of the TV
def originalPrice : ℝ := 1700

-- Define the value-added tax rate
def taxRate : ℝ := 0.15

-- Calculate the total price including tax
theorem tv_price_with_tax : originalPrice * (1 + taxRate) = 1955 :=
by
  sorry

end NUMINAMATH_GPT_tv_price_with_tax_l263_26307


namespace NUMINAMATH_GPT_find_increase_x_l263_26335

noncomputable def initial_radius : ℝ := 7
noncomputable def initial_height : ℝ := 5
variable (x : ℝ)

theorem find_increase_x (hx : x > 0)
  (volume_eq : π * (initial_radius + x) ^ 2 * initial_height =
               π * initial_radius ^ 2 * (initial_height + 2 * x)) :
  x = 28 / 5 :=
by
  sorry

end NUMINAMATH_GPT_find_increase_x_l263_26335


namespace NUMINAMATH_GPT_cloth_sold_worth_l263_26371

-- Define the commission rate and commission received
def commission_rate := 0.05
def commission_received := 12.50

-- State the theorem to be proved
theorem cloth_sold_worth : commission_received / commission_rate = 250 :=
by
  sorry

end NUMINAMATH_GPT_cloth_sold_worth_l263_26371


namespace NUMINAMATH_GPT_simplify_expression_l263_26331

variable (b : ℝ)

theorem simplify_expression :
  (2 * b + 6 - 5 * b) / 2 = -3 / 2 * b + 3 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l263_26331


namespace NUMINAMATH_GPT_dave_guitar_strings_l263_26393

noncomputable def strings_per_night : ℕ := 2
noncomputable def shows_per_week : ℕ := 6
noncomputable def weeks : ℕ := 12

theorem dave_guitar_strings : 
  (strings_per_night * shows_per_week * weeks) = 144 := 
by
  sorry

end NUMINAMATH_GPT_dave_guitar_strings_l263_26393


namespace NUMINAMATH_GPT_morio_current_age_l263_26338

-- Given conditions
def teresa_current_age : ℕ := 59
def morio_age_when_michiko_born : ℕ := 38
def teresa_age_when_michiko_born : ℕ := 26

-- Definitions derived from the conditions
def michiko_age : ℕ := teresa_current_age - teresa_age_when_michiko_born

-- Statement to prove Morio's current age
theorem morio_current_age : (michiko_age + morio_age_when_michiko_born) = 71 :=
by
  sorry

end NUMINAMATH_GPT_morio_current_age_l263_26338


namespace NUMINAMATH_GPT_correct_option_l263_26379

-- Define the variable 'a' as a real number
variable (a : ℝ)

-- Define propositions for each option
def option_A : Prop := 5 * a ^ 2 - 4 * a ^ 2 = 1
def option_B : Prop := (a ^ 7) / (a ^ 4) = a ^ 3
def option_C : Prop := (a ^ 3) ^ 2 = a ^ 5
def option_D : Prop := a ^ 2 * a ^ 3 = a ^ 6

-- State the main proposition asserting that option B is correct and others are incorrect
theorem correct_option :
  option_B a ∧ ¬option_A a ∧ ¬option_C a ∧ ¬option_D a :=
  by sorry

end NUMINAMATH_GPT_correct_option_l263_26379


namespace NUMINAMATH_GPT_sum_of_transformed_roots_equals_one_l263_26346

theorem sum_of_transformed_roots_equals_one 
  {α β γ : ℝ} 
  (hα : α^3 - α - 1 = 0) 
  (hβ : β^3 - β - 1 = 0) 
  (hγ : γ^3 - γ - 1 = 0) : 
  (1 - α) / (1 + α) + (1 - β) / (1 + β) + (1 - γ) / (1 + γ) = 1 :=
sorry

end NUMINAMATH_GPT_sum_of_transformed_roots_equals_one_l263_26346


namespace NUMINAMATH_GPT_alternative_plan_cost_is_eleven_l263_26352

-- Defining current cost
def current_cost : ℕ := 12

-- Defining the alternative plan cost in terms of current cost
def alternative_cost : ℕ := current_cost - 1

-- Theorem stating the alternative cost is $11
theorem alternative_plan_cost_is_eleven : alternative_cost = 11 :=
by
  -- This is the proof, which we are skipping with sorry
  sorry

end NUMINAMATH_GPT_alternative_plan_cost_is_eleven_l263_26352


namespace NUMINAMATH_GPT_calculation_result_l263_26322

theorem calculation_result : 
  2003^3 - 2001^3 - 6 * 2003^2 + 24 * 1001 = -4 := 
by 
  sorry

end NUMINAMATH_GPT_calculation_result_l263_26322


namespace NUMINAMATH_GPT_probability_of_forming_triangle_l263_26351

def segment_lengths : List ℕ := [1, 3, 5, 7, 9]
def valid_combinations : List (ℕ × ℕ × ℕ) := [(3, 5, 7), (3, 7, 9), (5, 7, 9)]
def total_combinations := Nat.choose 5 3

theorem probability_of_forming_triangle :
  (valid_combinations.length : ℚ) / total_combinations = 3 / 10 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_forming_triangle_l263_26351


namespace NUMINAMATH_GPT_rectangle_cut_dimensions_l263_26395

-- Define the original dimensions of the rectangle as constants.
def original_length : ℕ := 12
def original_height : ℕ := 6

-- Define the dimensions of the new rectangle after slicing parallel to the longer side.
def new_length := original_length / 2
def new_height := original_height

-- The theorem statement.
theorem rectangle_cut_dimensions :
  new_length = 6 ∧ new_height = 6 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_cut_dimensions_l263_26395


namespace NUMINAMATH_GPT_breadth_of_boat_l263_26381

theorem breadth_of_boat :
  ∀ (L : ℝ) (h : ℝ) (m : ℝ) (g : ℝ) (rho : ℝ),
    L = 8 → h = 0.01 → m = 160 → g = 9.81 → rho = 1000 →
    (L * 2 * h = (m * g) / (rho * g)) :=
by
  intros L h m g rho hL hh hm hg hrho
  sorry

end NUMINAMATH_GPT_breadth_of_boat_l263_26381


namespace NUMINAMATH_GPT_bee_loss_rate_l263_26340

theorem bee_loss_rate (initial_bees : ℕ) (days : ℕ) (remaining_bees : ℕ) :
  initial_bees = 80000 → 
  days = 50 → 
  remaining_bees = initial_bees / 4 → 
  (initial_bees - remaining_bees) / days = 1200 :=
by
  intros h₁ h₂ h₃
  sorry

end NUMINAMATH_GPT_bee_loss_rate_l263_26340


namespace NUMINAMATH_GPT_unique_two_digit_u_l263_26368

theorem unique_two_digit_u:
  ∃! u : ℤ, 10 ≤ u ∧ u < 100 ∧ 
            (15 * u) % 100 = 45 ∧ 
            u % 17 = 7 :=
by
  -- To be completed in proof
  sorry

end NUMINAMATH_GPT_unique_two_digit_u_l263_26368


namespace NUMINAMATH_GPT_rick_iron_hours_l263_26308

def can_iron_dress_shirts (h : ℕ) : ℕ := 4 * h

def can_iron_dress_pants (hours : ℕ) : ℕ := 3 * hours

def total_clothes_ironed (h : ℕ) : ℕ := can_iron_dress_shirts h + can_iron_dress_pants 5

theorem rick_iron_hours (h : ℕ) (H : total_clothes_ironed h = 27) : h = 3 :=
by sorry

end NUMINAMATH_GPT_rick_iron_hours_l263_26308


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_l263_26387

theorem right_triangle_hypotenuse (a b c : ℕ) (h : a = 3) (h' : b = 4) (hc : c^2 = a^2 + b^2) : c = 5 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_l263_26387


namespace NUMINAMATH_GPT_annie_spent_on_candies_l263_26310

theorem annie_spent_on_candies : 
  ∀ (num_classmates : ℕ) (candies_per_classmate : ℕ) (candies_left : ℕ) (cost_per_candy : ℚ),
  num_classmates = 35 →
  candies_per_classmate = 2 →
  candies_left = 12 →
  cost_per_candy = 0.1 →
  (num_classmates * candies_per_classmate + candies_left) * cost_per_candy = 8.2 :=
by
  intros num_classmates candies_per_classmate candies_left cost_per_candy
         h_classmates h_candies_per_classmate h_candies_left h_cost_per_candy
  simp [h_classmates, h_candies_per_classmate, h_candies_left, h_cost_per_candy]
  sorry

end NUMINAMATH_GPT_annie_spent_on_candies_l263_26310


namespace NUMINAMATH_GPT_apothem_comparison_l263_26398

noncomputable def pentagon_side_length : ℝ := 4 / Real.tan (54 * Real.pi / 180)

noncomputable def pentagon_apothem : ℝ := pentagon_side_length / (2 * Real.tan (54 * Real.pi / 180))

noncomputable def hexagon_side_length : ℝ := 4 / Real.sqrt 3

noncomputable def hexagon_apothem : ℝ := (Real.sqrt 3 / 2) * hexagon_side_length

theorem apothem_comparison : pentagon_apothem = 1.06 * hexagon_apothem :=
by
  sorry

end NUMINAMATH_GPT_apothem_comparison_l263_26398


namespace NUMINAMATH_GPT_sin_double_angle_l263_26320

theorem sin_double_angle (α : ℝ)
  (h : Real.cos (α + π / 6) = Real.sqrt 3 / 3) :
  Real.sin (2 * α - π / 6) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l263_26320


namespace NUMINAMATH_GPT_find_x_and_verify_l263_26317

theorem find_x_and_verify (x : ℤ) (h : (x - 14) / 10 = 4) : (x - 5) / 7 = 7 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_and_verify_l263_26317


namespace NUMINAMATH_GPT_fifteenth_term_arithmetic_sequence_l263_26327

theorem fifteenth_term_arithmetic_sequence (a d : ℤ) : 
  (a + 20 * d = 17) ∧ (a + 21 * d = 20) → (a + 14 * d = -1) := by
  sorry

end NUMINAMATH_GPT_fifteenth_term_arithmetic_sequence_l263_26327


namespace NUMINAMATH_GPT_badger_hid_35_l263_26304

-- Define the variables
variables (h_b h_f x : ℕ)

-- Define the conditions based on the problem
def badger_hides : Prop := 5 * h_b = x
def fox_hides : Prop := 7 * h_f = x
def fewer_holes : Prop := h_b = h_f + 2

-- The main theorem to prove the badger hid 35 walnuts
theorem badger_hid_35 (h_b h_f x : ℕ) :
  badger_hides h_b x ∧ fox_hides h_f x ∧ fewer_holes h_b h_f → x = 35 :=
by sorry

end NUMINAMATH_GPT_badger_hid_35_l263_26304


namespace NUMINAMATH_GPT_line_eq_l263_26349

theorem line_eq (P : ℝ × ℝ) (hP : P = (1, 2)) (h_perp : ∀ x y : ℝ, 2 * x + y - 1 = 0 → x - 2 * y + c = 0) : 
  ∃ c : ℝ, (x - 2 * y + c = 0 ∧ P ∈ {(x, y) | x - 2 * y + c = 0}) ∧ c = 3 :=
  sorry

end NUMINAMATH_GPT_line_eq_l263_26349


namespace NUMINAMATH_GPT_certain_number_sum_421_l263_26361

theorem certain_number_sum_421 :
  ∃ n, (∃ k, n = 423 * k) ∧ k = 2 →
  n + 421 = 1267 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_sum_421_l263_26361


namespace NUMINAMATH_GPT_largest_four_digit_number_l263_26302

theorem largest_four_digit_number
  (n : ℕ) (hn1 : n % 8 = 2) (hn2 : n % 7 = 4) (hn3 : 1000 ≤ n) (hn4 : n ≤ 9999) :
  n = 9990 :=
sorry

end NUMINAMATH_GPT_largest_four_digit_number_l263_26302


namespace NUMINAMATH_GPT_max_g_8_l263_26325

noncomputable def g (x : ℝ) : ℝ := sorry -- To be filled with the specific polynomial

theorem max_g_8 (g : ℝ → ℝ)
  (h_nonneg : ∀ x, 0 ≤ g x)
  (h4 : g 4 = 16)
  (h16 : g 16 = 1024) : g 8 ≤ 128 :=
sorry

end NUMINAMATH_GPT_max_g_8_l263_26325


namespace NUMINAMATH_GPT_program_arrangement_possible_l263_26369

theorem program_arrangement_possible (initial_programs : ℕ) (additional_programs : ℕ) 
  (h_initial: initial_programs = 6) (h_additional: additional_programs = 2) : 
  ∃ arrangements, arrangements = 56 :=
by
  sorry

end NUMINAMATH_GPT_program_arrangement_possible_l263_26369


namespace NUMINAMATH_GPT_calculate_coeffs_l263_26354

noncomputable def quadratic_coeffs (p q : ℝ) : Prop :=
  if p = 1 then true else if p = -2 then q = -1 else false

theorem calculate_coeffs (p q : ℝ) :
    (∃ p q, (x^2 + p * x + q = 0) ∧ (x^2 - p^2 * x + p * q = 0)) →
    quadratic_coeffs p q :=
by sorry

end NUMINAMATH_GPT_calculate_coeffs_l263_26354


namespace NUMINAMATH_GPT_original_numbers_geometric_sequence_l263_26358

theorem original_numbers_geometric_sequence (a q : ℝ) :
  (2 * (a * q + 8) = a + a * q^2) →
  ((a * q + 8) ^ 2 = a * (a * q^2 + 64)) →
  (a, a * q, a * q^2) = (4, 12, 36) ∨ (a, a * q, a * q^2) = (4 / 9, -20 / 9, 100 / 9) :=
by {
  sorry
}

end NUMINAMATH_GPT_original_numbers_geometric_sequence_l263_26358


namespace NUMINAMATH_GPT_max_sum_clock_digits_l263_26355

theorem max_sum_clock_digits : ∃ t : ℕ, 0 ≤ t ∧ t < 24 ∧ 
  (∃ h1 h2 m1 m2 : ℕ, t = h1 * 10 + h2 + m1 * 10 + m2 ∧ 
   (0 ≤ h1 ∧ h1 ≤ 2) ∧ (0 ≤ h2 ∧ h2 ≤ 9) ∧ (0 ≤ m1 ∧ m1 ≤ 5) ∧ (0 ≤ m2 ∧ m2 ≤ 9) ∧ 
   h1 + h2 + m1 + m2 = 24) := sorry

end NUMINAMATH_GPT_max_sum_clock_digits_l263_26355


namespace NUMINAMATH_GPT_system_of_equations_solution_l263_26384

theorem system_of_equations_solution :
  ∃ x y : ℝ, 7 * x - 3 * y = 2 ∧ 2 * x + y = 8 ∧ x = 2 ∧ y = 4 :=
by
  use 2
  use 4
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l263_26384


namespace NUMINAMATH_GPT_fraction_part_of_twenty_five_l263_26329

open Nat

def eighty_percent (x : ℕ) : ℕ := (85 * x) / 100

theorem fraction_part_of_twenty_five (x y : ℕ) (h1 : eighty_percent 40 = 34) (h2 : 34 - y = 14) (h3 : y = (4 * 25) / 5) : y = 20 :=
by 
  -- Given h1: eighty_percent 40 = 34
  -- And h2: 34 - y = 14
  -- And h3: y = (4 * 25) / 5
  -- Show y = 20
  sorry

end NUMINAMATH_GPT_fraction_part_of_twenty_five_l263_26329


namespace NUMINAMATH_GPT_tim_income_less_than_juan_l263_26353

-- Definitions of the conditions
variables {T J M : ℝ}
def mart_income_condition1 (M T : ℝ) : Prop := M = 1.40 * T
def mart_income_condition2 (M J : ℝ) : Prop := M = 0.84 * J

-- The proof goal
theorem tim_income_less_than_juan (T J M : ℝ) 
(h1: mart_income_condition1 M T) 
(h2: mart_income_condition2 M J) : 
T = 0.60 * J :=
by
  sorry

end NUMINAMATH_GPT_tim_income_less_than_juan_l263_26353


namespace NUMINAMATH_GPT_first_even_number_of_8_sum_424_l263_26324

theorem first_even_number_of_8_sum_424 (x : ℕ) (h : x + (x + 2) + (x + 4) + (x + 6) + 
                   (x + 8) + (x + 10) + (x + 12) + (x + 14) = 424) : x = 46 :=
by sorry

end NUMINAMATH_GPT_first_even_number_of_8_sum_424_l263_26324


namespace NUMINAMATH_GPT_slope_of_line_l263_26328

theorem slope_of_line : ∀ (x y : ℝ), (x - y + 1 = 0) → (1 = 1) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_slope_of_line_l263_26328


namespace NUMINAMATH_GPT_problem_conditions_l263_26374

noncomputable def f (x : ℝ) : ℝ := -x - x^3

variables (x₁ x₂ : ℝ)

theorem problem_conditions (h₁ : x₁ + x₂ ≤ 0) :
  (f x₁ * f (-x₁) ≤ 0) ∧
  (¬ (f x₂ * f (-x₂) > 0)) ∧
  (¬ (f x₁ + f x₂ ≤ f (-x₁) + f (-x₂))) ∧
  (f x₁ + f x₂ ≥ f (-x₁) + f (-x₂)) :=
sorry

end NUMINAMATH_GPT_problem_conditions_l263_26374


namespace NUMINAMATH_GPT_jerry_remaining_money_l263_26350

-- Define initial money
def initial_money := 18

-- Define amount spent on video games
def spent_video_games := 6

-- Define amount spent on a snack
def spent_snack := 3

-- Define total amount spent
def total_spent := spent_video_games + spent_snack

-- Define remaining money after spending
def remaining_money := initial_money - total_spent

theorem jerry_remaining_money : remaining_money = 9 :=
by
  sorry

end NUMINAMATH_GPT_jerry_remaining_money_l263_26350


namespace NUMINAMATH_GPT_weight_of_dry_grapes_l263_26389

theorem weight_of_dry_grapes (w_fresh : ℝ) (perc_water_fresh perc_water_dried : ℝ) (w_non_water : ℝ) (w_dry : ℝ) :
  w_fresh = 5 →
  perc_water_fresh = 0.90 →
  perc_water_dried = 0.20 →
  w_non_water = w_fresh * (1 - perc_water_fresh) →
  w_non_water = w_dry * (1 - perc_water_dried) →
  w_dry = 0.625 :=
by sorry

end NUMINAMATH_GPT_weight_of_dry_grapes_l263_26389


namespace NUMINAMATH_GPT_billy_sleep_total_hours_l263_26319

theorem billy_sleep_total_hours : 
    let first_night := 6
    let second_night := 2 * first_night
    let third_night := second_night - 3
    let fourth_night := 3 * third_night
    first_night + second_night + third_night + fourth_night = 54
  := by
    sorry

end NUMINAMATH_GPT_billy_sleep_total_hours_l263_26319


namespace NUMINAMATH_GPT_geometric_progression_fourth_term_l263_26378

theorem geometric_progression_fourth_term (x : ℚ)
  (h : (3 * x + 3) / x = (5 * x + 5) / (3 * x + 3)) :
  (5 / 3) * (5 * x + 5) = -125/12 :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_fourth_term_l263_26378


namespace NUMINAMATH_GPT_monthly_expenses_last_month_was_2888_l263_26385

def basic_salary : ℕ := 1250
def commission_rate : ℚ := 0.10
def total_sales : ℕ := 23600
def savings_rate : ℚ := 0.20

theorem monthly_expenses_last_month_was_2888 :
  let commission := commission_rate * total_sales
  let total_earnings := basic_salary + commission
  let savings := savings_rate * total_earnings
  let monthly_expenses := total_earnings - savings
  monthly_expenses = 2888 := by
  sorry

end NUMINAMATH_GPT_monthly_expenses_last_month_was_2888_l263_26385


namespace NUMINAMATH_GPT_no_intersect_x_axis_intersection_points_m_minus3_l263_26321

-- Define the quadratic function y = x^2 - 6x + 2m - 1
def quadratic_function (x m : ℝ) : ℝ := x^2 - 6 * x + 2 * m - 1

-- Theorem for Question 1: The function does not intersect the x-axis if and only if m > 5
theorem no_intersect_x_axis (m : ℝ) : (∀ x : ℝ, quadratic_function x m ≠ 0) ↔ m > 5 := sorry

-- Specific case when m = -3
def quadratic_function_m_minus3 (x : ℝ) : ℝ := x^2 - 6 * x - 7

-- Theorem for Question 2: Intersection points with coordinate axes for m = -3
theorem intersection_points_m_minus3 :
  ((∃ x : ℝ, quadratic_function_m_minus3 x = 0 ∧ (x = -1 ∨ x = 7)) ∧
   quadratic_function_m_minus3 0 = -7) := sorry

end NUMINAMATH_GPT_no_intersect_x_axis_intersection_points_m_minus3_l263_26321


namespace NUMINAMATH_GPT_hh3_value_l263_26312

noncomputable def h (x : ℤ) : ℤ := 3 * x^3 + 3 * x^2 - x - 1

theorem hh3_value : h (h 3) = 3406935 := by
  sorry

end NUMINAMATH_GPT_hh3_value_l263_26312


namespace NUMINAMATH_GPT_intersection_in_quadrants_I_and_II_l263_26377

open Set

def in_quadrants_I_and_II (x y : ℝ) : Prop :=
  (0 ≤ x ∧ 0 ≤ y) ∨ (x ≤ 0 ∧ 0 ≤ y)

theorem intersection_in_quadrants_I_and_II :
  ∀ (x y : ℝ),
    y > 3 * x → y > -2 * x + 3 → in_quadrants_I_and_II x y :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_intersection_in_quadrants_I_and_II_l263_26377


namespace NUMINAMATH_GPT_gasoline_added_l263_26330

theorem gasoline_added (total_capacity : ℝ) (initial_fraction final_fraction : ℝ) 
(h1 : initial_fraction = 3 / 4)
(h2 : final_fraction = 9 / 10)
(h3 : total_capacity = 29.999999999999996) : 
(final_fraction * total_capacity - initial_fraction * total_capacity = 4.499999999999999) :=
by sorry

end NUMINAMATH_GPT_gasoline_added_l263_26330


namespace NUMINAMATH_GPT_distance_not_all_odd_l263_26392

theorem distance_not_all_odd (A B C D : ℝ × ℝ) : 
  ∃ (P Q : ℝ × ℝ), dist P Q % 2 = 0 := by sorry

end NUMINAMATH_GPT_distance_not_all_odd_l263_26392


namespace NUMINAMATH_GPT_log_expression_eval_find_m_from_conditions_l263_26311

-- (1) Prove that lg (5^2) + (2/3) * lg 8 + lg 5 * lg 20 + (lg 2)^2 = 3.
theorem log_expression_eval : 
  Real.logb 10 (5^2) + (2 / 3) * Real.logb 10 8 + Real.logb 10 5 * Real.logb 10 20 + (Real.logb 10 2)^2 = 3 := 
sorry

-- (2) Given 2^a = 5^b = m and 1/a + 1/b = 2, prove that m = sqrt(10).
theorem find_m_from_conditions (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 :=
sorry

end NUMINAMATH_GPT_log_expression_eval_find_m_from_conditions_l263_26311


namespace NUMINAMATH_GPT_percentage_honda_red_l263_26333

theorem percentage_honda_red (total_cars : ℕ) (honda_cars : ℕ) (percentage_red_total : ℚ)
  (percentage_red_non_honda : ℚ) (percentage_red_honda : ℚ) :
  total_cars = 9000 →
  honda_cars = 5000 →
  percentage_red_total = 0.60 →
  percentage_red_non_honda = 0.225 →
  percentage_red_honda = 0.90 →
  ((honda_cars * percentage_red_honda) / total_cars) * 100 = ((total_cars * percentage_red_total - (total_cars - honda_cars) * percentage_red_non_honda) / honda_cars) * 100 :=
by
  sorry

end NUMINAMATH_GPT_percentage_honda_red_l263_26333


namespace NUMINAMATH_GPT_correct_statement_about_K_l263_26380

-- Defining the possible statements about the chemical equilibrium constant K
def K (n : ℕ) : String :=
  match n with
  | 1 => "The larger the K, the smaller the conversion rate of the reactants."
  | 2 => "K is related to the concentration of the reactants."
  | 3 => "K is related to the concentration of the products."
  | 4 => "K is related to temperature."
  | _ => "Invalid statement"

-- Given that the correct answer is that K is related to temperature
theorem correct_statement_about_K : K 4 = "K is related to temperature." :=
by
  rfl

end NUMINAMATH_GPT_correct_statement_about_K_l263_26380


namespace NUMINAMATH_GPT_chess_tournament_games_l263_26309

def number_of_games (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chess_tournament_games :
  number_of_games 20 = 190 :=
by
  sorry

end NUMINAMATH_GPT_chess_tournament_games_l263_26309


namespace NUMINAMATH_GPT_problem_solution_l263_26390

open Set

def A : Set ℕ := {x | x > 0 ∧ x < 9}
def B : Set ℕ := {1, 2, 3}
def C : Set ℕ := {3, 4, 5, 6}

theorem problem_solution :
  A ∩ B = {1, 2, 3} ∧
  A ∩ C = {3, 4, 5, 6} ∧
  A ∩ (B ∪ C) = {1, 2, 3, 4, 5, 6} ∧
  A ∪ (B ∩ C) = {1, 2, 3, 4, 5, 6, 7, 8} :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l263_26390


namespace NUMINAMATH_GPT_actual_revenue_percentage_l263_26397

def last_year_revenue (R : ℝ) := R
def projected_revenue (R : ℝ) := 1.25 * R
def actual_revenue (R : ℝ) := 0.75 * R

theorem actual_revenue_percentage (R : ℝ) : 
  (actual_revenue R / projected_revenue R) * 100 = 60 :=
by
  sorry

end NUMINAMATH_GPT_actual_revenue_percentage_l263_26397


namespace NUMINAMATH_GPT_find_price_per_craft_l263_26348

-- Definitions based on conditions
def price_per_craft (x : ℝ) : Prop :=
  let crafts_sold := 3
  let extra_money := 7
  let deposit := 18
  let remaining_money := 25
  let total_before_deposit := 43
  3 * x + extra_money = total_before_deposit

-- Statement of the problem to prove x = 12 given conditions
theorem find_price_per_craft : ∃ x : ℝ, price_per_craft x ∧ x = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_price_per_craft_l263_26348


namespace NUMINAMATH_GPT_base_7_minus_base_8_to_decimal_l263_26367

theorem base_7_minus_base_8_to_decimal : 
  (5 * 7^4 + 4 * 7^3 + 3 * 7^2 + 2 * 7^1 + 1 * 7^0) - (1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0) = 8190 :=
by sorry

end NUMINAMATH_GPT_base_7_minus_base_8_to_decimal_l263_26367


namespace NUMINAMATH_GPT_tangent_line_equation_at_point_l263_26316

-- Defining the function and the point
def f (x : ℝ) : ℝ := x^2 + 2 * x
def point : ℝ × ℝ := (1, 3)

-- Main theorem stating the tangent line equation at the given point
theorem tangent_line_equation_at_point : 
  ∃ m b, (m = (2 * 1 + 2)) ∧ 
         (b = (3 - m * 1)) ∧ 
         (∀ x y, y = f x → y = m * x + b → 4 * x - y - 1 = 0) :=
by
  -- Proof is omitted and can be filled in later
  sorry

end NUMINAMATH_GPT_tangent_line_equation_at_point_l263_26316


namespace NUMINAMATH_GPT_smallest_k_for_polygon_l263_26373

-- Definitions and conditions
def equiangular_decagon_interior_angle : ℝ := 144

-- Question transformation into a proof problem
theorem smallest_k_for_polygon (k : ℕ) (hk : k > 1) :
  (∀ (n2 : ℕ), n2 = 10 * k → ∃ (interior_angle : ℝ), interior_angle = k * equiangular_decagon_interior_angle ∧
  n2 ≥ 3) → k = 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_for_polygon_l263_26373


namespace NUMINAMATH_GPT_tan_A_value_l263_26372

open Real

theorem tan_A_value (A : ℝ) (h1 : sin A * (sin A + sqrt 3 * cos A) = -1 / 2) (h2 : 0 < A ∧ A < π) :
  tan A = -sqrt 3 / 3 :=
sorry

end NUMINAMATH_GPT_tan_A_value_l263_26372


namespace NUMINAMATH_GPT_divisors_remainder_5_l263_26360

theorem divisors_remainder_5 (d : ℕ) : d ∣ 2002 ∧ d > 5 ↔ d = 7 ∨ d = 11 ∨ d = 13 ∨ d = 14 ∨ 
                                      d = 22 ∨ d = 26 ∨ d = 77 ∨ d = 91 ∨ 
                                      d = 143 ∨ d = 154 ∨ d = 182 ∨ d = 286 ∨ 
                                      d = 1001 ∨ d = 2002 :=
by sorry

end NUMINAMATH_GPT_divisors_remainder_5_l263_26360


namespace NUMINAMATH_GPT_polynomial_coefficients_sum_even_odd_coefficients_difference_square_l263_26388

theorem polynomial_coefficients_sum (a : Fin 8 → ℝ):
  (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) = 3^7 - 1 :=
by
  -- assume the polynomial (1 + 2x)^7 has coefficients a 0, a 1, ..., a 7
  -- such that (1 + 2x)^7 = a 0 + a 1 * x + a 2 * x^2 + ... + a 7 * x^7
  sorry

theorem even_odd_coefficients_difference_square (a : Fin 8 → ℝ):
  (a 0 + a 2 + a 4 + a 6)^2 - (a 1 + a 3 + a 5 + a 7)^2 = -3^7 :=
by
  -- assume the polynomial (1 + 2x)^7 has coefficients a 0, a 1, ..., a 7
  -- such that (1 + 2x)^7 = a 0 + a 1 * x + a 2 * x^2 + ... + a 7 * x^7
  sorry

end NUMINAMATH_GPT_polynomial_coefficients_sum_even_odd_coefficients_difference_square_l263_26388


namespace NUMINAMATH_GPT_five_twos_make_24_l263_26396

theorem five_twos_make_24 :
  ∃ a b c d e : ℕ, a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 ∧ e = 2 ∧
  ((a + b + c) * (d + e) = 24) :=
by
  sorry

end NUMINAMATH_GPT_five_twos_make_24_l263_26396


namespace NUMINAMATH_GPT_area_of_closed_shape_l263_26386

theorem area_of_closed_shape :
  ∫ y in (-2 : ℝ)..3, ((2:ℝ)^y + 2 - (2:ℝ)^y) = 10 := by
  sorry

end NUMINAMATH_GPT_area_of_closed_shape_l263_26386


namespace NUMINAMATH_GPT_num_sets_B_l263_26342

open Set

theorem num_sets_B (A B : Set ℕ) (hA : A = {1, 2}) (h_union : A ∪ B = {1, 2, 3}) : ∃ n, n = 4 :=
by
  sorry

end NUMINAMATH_GPT_num_sets_B_l263_26342


namespace NUMINAMATH_GPT_solve_inequality_l263_26359

theorem solve_inequality 
  (k_0 k b m n : ℝ)
  (hM1 : -1 = k_0 * m + b) (hM2 : -1 = k^2 / m)
  (hN1 : 2 = k_0 * n + b) (hN2 : 2 = k^2 / n) :
  {x : ℝ | x^2 > k_0 * k^2 + b * x} = {x : ℝ | x < -1 ∨ x > 2} :=
  sorry

end NUMINAMATH_GPT_solve_inequality_l263_26359


namespace NUMINAMATH_GPT_largest_constant_inequality_equality_condition_l263_26376

theorem largest_constant_inequality (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆) ^ 2 ≥
    3 * (x₁ * (x₂ + x₃) + x₂ * (x₃ + x₄) + x₃ * (x₄ + x₅) + x₄ * (x₅ + x₆) + x₅ * (x₆ + x₁) + x₆ * (x₁ + x₂)) :=
sorry

theorem equality_condition (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  (x₁ + x₄ = x₂ + x₅) ∧ (x₂ + x₅ = x₃ + x₆) :=
sorry

end NUMINAMATH_GPT_largest_constant_inequality_equality_condition_l263_26376


namespace NUMINAMATH_GPT_smallest_pos_integer_n_l263_26341

theorem smallest_pos_integer_n 
  (x y : ℤ)
  (hx: ∃ k : ℤ, x = 8 * k - 2)
  (hy : ∃ l : ℤ, y = 8 * l + 2) :
  ∃ n : ℤ, n > 0 ∧ ∃ (m : ℤ), x^2 - x*y + y^2 + n = 8 * m ∧ n = 4 := by
  sorry

end NUMINAMATH_GPT_smallest_pos_integer_n_l263_26341


namespace NUMINAMATH_GPT_difference_square_consecutive_l263_26375

theorem difference_square_consecutive (x : ℕ) (h : x * (x + 1) = 812) : (x + 1)^2 - x = 813 :=
sorry

end NUMINAMATH_GPT_difference_square_consecutive_l263_26375


namespace NUMINAMATH_GPT_average_interest_rate_equal_4_09_percent_l263_26383

-- Define the given conditions
def investment_total : ℝ := 5000
def interest_rate_at_3_percent : ℝ := 0.03
def interest_rate_at_5_percent : ℝ := 0.05
def return_relationship (x : ℝ) : Prop := 
  interest_rate_at_5_percent * x = 2 * interest_rate_at_3_percent * (investment_total - x)

-- Define the final statement
theorem average_interest_rate_equal_4_09_percent :
  ∃ x : ℝ, return_relationship x ∧ 
  ((interest_rate_at_5_percent * x + interest_rate_at_3_percent * (investment_total - x)) / investment_total) = 0.04091 := 
by
  sorry

end NUMINAMATH_GPT_average_interest_rate_equal_4_09_percent_l263_26383


namespace NUMINAMATH_GPT_min_cuts_for_payment_7_days_l263_26332

theorem min_cuts_for_payment_7_days (n : ℕ) (h : n = 7) : ∃ k, k = 1 :=
by sorry

end NUMINAMATH_GPT_min_cuts_for_payment_7_days_l263_26332


namespace NUMINAMATH_GPT_distance_between_stripes_l263_26363

theorem distance_between_stripes
  (curb_distance : ℝ) (length_curb : ℝ) (stripe_length : ℝ) (distance_stripes : ℝ)
  (h1 : curb_distance = 60)
  (h2 : length_curb = 20)
  (h3 : stripe_length = 50)
  (h4 : distance_stripes = (length_curb * curb_distance) / stripe_length) :
  distance_stripes = 24 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_stripes_l263_26363


namespace NUMINAMATH_GPT_log2_125_eq_9y_l263_26339

theorem log2_125_eq_9y (y : ℝ) (h : Real.log 5 / Real.log 8 = y) : Real.log 125 / Real.log 2 = 9 * y :=
by
  sorry

end NUMINAMATH_GPT_log2_125_eq_9y_l263_26339


namespace NUMINAMATH_GPT_four_distinct_real_roots_l263_26344

noncomputable def f (x c : ℝ) : ℝ := x^2 + 4 * x + c

-- We need to prove that if c is in the interval (-1, 3), f(f(x)) has exactly 4 distinct real roots
theorem four_distinct_real_roots (c : ℝ) : (-1 < c) ∧ (c < 3) → 
  ∃ (x₁ x₂ x₃ x₄ : ℝ), (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) 
  ∧ (f (f x₁ c) c = 0 ∧ f (f x₂ c) c = 0 ∧ f (f x₃ c) c = 0 ∧ f (f x₄ c) c = 0) :=
by sorry

end NUMINAMATH_GPT_four_distinct_real_roots_l263_26344


namespace NUMINAMATH_GPT_total_revenue_correct_l263_26313

noncomputable def total_revenue : ℚ := 
  let revenue_v1 := 23 * 5 * 0.50
  let revenue_v2 := 28 * 6 * 0.60
  let revenue_v3 := 35 * 7 * 0.50
  let revenue_v4 := 43 * 8 * 0.60
  let revenue_v5 := 50 * 9 * 0.50
  let revenue_v6 := 64 * 10 * 0.60
  revenue_v1 + revenue_v2 + revenue_v3 + revenue_v4 + revenue_v5 + revenue_v6

theorem total_revenue_correct : total_revenue = 1096.20 := 
by
  sorry

end NUMINAMATH_GPT_total_revenue_correct_l263_26313


namespace NUMINAMATH_GPT_total_cupcakes_l263_26314

theorem total_cupcakes (children : ℕ) (cupcakes_per_child : ℕ) (h1 : children = 8) (h2 : cupcakes_per_child = 12) : children * cupcakes_per_child = 96 :=
by
  sorry

end NUMINAMATH_GPT_total_cupcakes_l263_26314


namespace NUMINAMATH_GPT_peg_stickers_total_l263_26318

def stickers_in_red_folder : ℕ := 10 * 3
def stickers_in_green_folder : ℕ := 10 * 2
def stickers_in_blue_folder : ℕ := 10 * 1

def total_stickers : ℕ := stickers_in_red_folder + stickers_in_green_folder + stickers_in_blue_folder

theorem peg_stickers_total : total_stickers = 60 := by
  sorry

end NUMINAMATH_GPT_peg_stickers_total_l263_26318


namespace NUMINAMATH_GPT_average_speed_of_train_l263_26343

theorem average_speed_of_train (d1 d2: ℝ) (t1 t2: ℝ) (h_d1: d1 = 250) (h_d2: d2 = 350) (h_t1: t1 = 2) (h_t2: t2 = 4) :
  (d1 + d2) / (t1 + t2) = 100 := by
  sorry

end NUMINAMATH_GPT_average_speed_of_train_l263_26343


namespace NUMINAMATH_GPT_quadratic_graph_nature_l263_26366

theorem quadratic_graph_nature (a b : Real) (h : a ≠ 0) :
  ∀ (x : Real), (a * x^2 + b * x + (b^2 / (2 * a)) > 0) ∨ (a * x^2 + b * x + (b^2 / (2 * a)) < 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_graph_nature_l263_26366


namespace NUMINAMATH_GPT_expand_and_simplify_l263_26356

theorem expand_and_simplify :
  ∀ (x : ℝ), 5 * (6 * x^3 - 3 * x^2 + 4 * x - 2) = 30 * x^3 - 15 * x^2 + 20 * x - 10 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_expand_and_simplify_l263_26356


namespace NUMINAMATH_GPT_monotonic_intervals_range_of_k_l263_26301

noncomputable def f (x a : ℝ) : ℝ :=
  (x - a - 1) * Real.exp x - (1/2) * x^2 + a * x

-- Conditions: a > 0
variables (a : ℝ) (h_a : 0 < a)

-- Part (1): Monotonic Intervals
theorem monotonic_intervals :
  (∀ x, f x a < f (x + 1) a ↔ x < 0 ∨ a < x) ∧
  (∀ x, f (x + 1) a < f x a ↔ 0 < x ∧ x < a) :=
  sorry

-- Part (2): Range of k
theorem range_of_k (x1 x2 : ℝ) (hx1 : f x1 a = 0) (hx2 : f x2 a = 0) :
  (f x1 a - f x2 a < k * a^3) ↔ k ≥ -1/6 :=
  sorry

end NUMINAMATH_GPT_monotonic_intervals_range_of_k_l263_26301


namespace NUMINAMATH_GPT_not_product_of_consecutives_l263_26305

theorem not_product_of_consecutives (n k : ℕ) : 
  ¬ (∃ a b: ℕ, a + 1 = b ∧ (2 * n^(3 * k) + 4 * n^k + 10 = a * b)) :=
by sorry

end NUMINAMATH_GPT_not_product_of_consecutives_l263_26305


namespace NUMINAMATH_GPT_mod_50_remainder_of_b86_l263_26303

def b (n : ℕ) : ℕ := 7^n + 9^n

theorem mod_50_remainder_of_b86 : (b 86) % 50 = 40 := 
by 
-- Given definition of b and the problem is to prove the remainder of b_86 when divided by 50 is 40
sorry

end NUMINAMATH_GPT_mod_50_remainder_of_b86_l263_26303


namespace NUMINAMATH_GPT_sequence_a_100_l263_26336

theorem sequence_a_100 : 
  (∃ a : ℕ → ℕ, a 1 = 2 ∧ (∀ n ≥ 1, a (n + 1) = a n + 2 * n) ∧ a 100 = 9902) :=
by
  sorry

end NUMINAMATH_GPT_sequence_a_100_l263_26336


namespace NUMINAMATH_GPT_largest_two_digit_divisible_by_6_ending_in_4_l263_26370

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, is_two_digit n ∧ is_divisible_by_6 n ∧ ends_in_4 n ∧
  ∀ m : ℕ, is_two_digit m ∧ is_divisible_by_6 m ∧ ends_in_4 m → m ≤ n :=
  sorry

end NUMINAMATH_GPT_largest_two_digit_divisible_by_6_ending_in_4_l263_26370


namespace NUMINAMATH_GPT_payment_to_C_l263_26364

-- Work rates definition
def work_rate_A : ℚ := 1 / 6
def work_rate_B : ℚ := 1 / 8
def combined_work_rate_A_B : ℚ := work_rate_A + work_rate_B
def combined_work_rate_A_B_C : ℚ := 1 / 3

-- C's work rate calculation
def work_rate_C : ℚ := combined_work_rate_A_B_C - combined_work_rate_A_B

-- Payment calculation
def total_payment : ℚ := 3200
def C_payment_ratio : ℚ := work_rate_C / combined_work_rate_A_B_C
def C_payment : ℚ := total_payment * C_payment_ratio

-- Theorem stating the result
theorem payment_to_C : C_payment = 400 := by
  sorry

end NUMINAMATH_GPT_payment_to_C_l263_26364


namespace NUMINAMATH_GPT_roots_opposite_signs_l263_26347

theorem roots_opposite_signs (p : ℝ) (hp : p > 0) :
  ( ∃ (x₁ x₂ : ℝ), (x₁ * x₂ < 0) ∧ (5 * x₁^2 - 4 * (p + 3) * x₁ + 4 = p^2) ∧  
      (5 * x₂^2 - 4 * (p + 3) * x₂ + 4 = p^2) ) ↔ p > 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_roots_opposite_signs_l263_26347


namespace NUMINAMATH_GPT_complement_of_intersection_eq_l263_26337

-- Definitions of sets with given conditions
def U : Set ℝ := {x | 0 ≤ x ∧ x < 10}
def A : Set ℝ := {x | 2 < x ∧ x ≤ 4}
def B : Set ℝ := {x | 3 < x ∧ x ≤ 5}

-- Complement of a set with respect to U
def complement_U (S : Set ℝ) : Set ℝ := {x | x ∈ U ∧ x ∉ S}

-- Intersect two sets
def intersection (S1 S2 : Set ℝ) : Set ℝ := {x | x ∈ S1 ∧ x ∈ S2}

theorem complement_of_intersection_eq :
  complement_U (intersection A B) = {x | (0 ≤ x ∧ x ≤ 2) ∨ (5 < x ∧ x < 10)} := 
by
  sorry

end NUMINAMATH_GPT_complement_of_intersection_eq_l263_26337


namespace NUMINAMATH_GPT_range_of_a_l263_26306

variable {a b c d : ℝ}

theorem range_of_a (h1 : a + b + c + d = 3) (h2 : a^2 + 2 * b^2 + 3 * c^2 + 6 * d^2 = 5) : 1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l263_26306


namespace NUMINAMATH_GPT_max_value_frac_l263_26382
noncomputable section

open Real

variables (a b x y : ℝ)

theorem max_value_frac :
  a > 1 → b > 1 → 
  a^x = 2 → b^y = 2 →
  a + sqrt b = 4 →
  (2/x + 1/y) ≤ 4 :=
by
  intros ha hb hax hby hab
  sorry

end NUMINAMATH_GPT_max_value_frac_l263_26382


namespace NUMINAMATH_GPT_chess_tournament_participants_l263_26391

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 105) : n = 15 := by
  sorry

end NUMINAMATH_GPT_chess_tournament_participants_l263_26391


namespace NUMINAMATH_GPT_relationship_between_y_values_l263_26394

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + |b| * x + c

theorem relationship_between_y_values
  (a b c y1 y2 y3 : ℝ)
  (h1 : quadratic_function a b c (-14 / 3) = y1)
  (h2 : quadratic_function a b c (5 / 2) = y2)
  (h3 : quadratic_function a b c 3 = y3)
  (axis_symmetry : -(|b| / (2 * a)) = -1)
  (h_pos : 0 < a) :
  y2 < y1 ∧ y1 < y3 :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_y_values_l263_26394


namespace NUMINAMATH_GPT_problem_solution_l263_26326

theorem problem_solution (a b c d : ℝ) 
  (h1 : 3 * a + 2 * b + 4 * c + 8 * d = 40)
  (h2 : 4 * (d + c) = b)
  (h3 : 2 * b + 2 * c = a)
  (h4 : c + 1 = d) :
  a * b * c * d = 0 :=
sorry

end NUMINAMATH_GPT_problem_solution_l263_26326


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l263_26300

variable {a : ℝ}

theorem sufficient_but_not_necessary (h : a > 1) : a^2 > a :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l263_26300


namespace NUMINAMATH_GPT_price_of_75_cans_l263_26323

/-- The price of 75 cans of a certain brand of soda purchased in 24-can cases,
    given the regular price per can is $0.15 and a 10% discount is applied when
    purchased in 24-can cases, is $10.125.
-/
theorem price_of_75_cans (regular_price : ℝ) (discount : ℝ) (cases_needed : ℕ) (remaining_cans : ℕ) 
  (discounted_price : ℝ) (total_price : ℝ) :
  regular_price = 0.15 →
  discount = 0.10 →
  discounted_price = regular_price - (discount * regular_price) →
  cases_needed = 75 / 24 ∧ remaining_cans = 75 % 24 →
  total_price = (cases_needed * 24 + remaining_cans) * discounted_price →
  total_price = 10.125 :=
by
  sorry

end NUMINAMATH_GPT_price_of_75_cans_l263_26323


namespace NUMINAMATH_GPT_greatest_positive_integer_x_l263_26365

theorem greatest_positive_integer_x (x : ℕ) (h₁ : x^2 < 12) (h₂ : ∀ y: ℕ, y^2 < 12 → y ≤ x) : 
  x = 3 := 
by
  sorry

end NUMINAMATH_GPT_greatest_positive_integer_x_l263_26365


namespace NUMINAMATH_GPT_keith_score_l263_26362

theorem keith_score (K : ℕ) (h : K + 3 * K + (3 * K + 5) = 26) : K = 3 :=
by
  sorry

end NUMINAMATH_GPT_keith_score_l263_26362


namespace NUMINAMATH_GPT_axis_of_symmetry_parabola_l263_26334

theorem axis_of_symmetry_parabola : 
  ∀ (x : ℝ), ∃ y : ℝ, y = (x - 5)^2 → x = 5 := 
by 
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_parabola_l263_26334


namespace NUMINAMATH_GPT_books_sold_on_tuesday_l263_26315

theorem books_sold_on_tuesday (total_stock : ℕ) (monday_sold : ℕ) (wednesday_sold : ℕ)
  (thursday_sold : ℕ) (friday_sold : ℕ) (percent_unsold : ℚ) (tuesday_sold : ℕ) :
  total_stock = 1100 →
  monday_sold = 75 →
  wednesday_sold = 64 →
  thursday_sold = 78 →
  friday_sold = 135 →
  percent_unsold = 63.45 →
  tuesday_sold = total_stock - (monday_sold + wednesday_sold + thursday_sold + friday_sold + (total_stock * percent_unsold / 100)) :=
by sorry

end NUMINAMATH_GPT_books_sold_on_tuesday_l263_26315


namespace NUMINAMATH_GPT_expansion_coefficient_a2_l263_26357

theorem expansion_coefficient_a2 : 
  (∃ (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ), 
    (1 - 2*x)^7 = a + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 + a_6*x^6 + a_7*x^7 -> 
    a_2 = 84) :=
sorry

end NUMINAMATH_GPT_expansion_coefficient_a2_l263_26357
