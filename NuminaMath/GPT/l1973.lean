import Mathlib

namespace NUMINAMATH_GPT_min_value_of_function_l1973_197314

theorem min_value_of_function : ∃ x : ℝ, ∀ x : ℝ, x * (x + 1) * (x + 2) * (x + 3) ≥ -1 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_function_l1973_197314


namespace NUMINAMATH_GPT_graph_passes_through_point_l1973_197364

theorem graph_passes_through_point :
  ∀ (a : ℝ), 0 < a ∧ a < 1 → (∃ (x y : ℝ), (x = 2) ∧ (y = -1) ∧ (y = 2 * a * x - 1)) :=
by
  sorry

end NUMINAMATH_GPT_graph_passes_through_point_l1973_197364


namespace NUMINAMATH_GPT_sum_y_coordinates_of_other_vertices_of_parallelogram_l1973_197363

theorem sum_y_coordinates_of_other_vertices_of_parallelogram :
  let x1 := 4
  let y1 := 26
  let x2 := 12
  let y2 := -8
  let midpoint_y := (y1 + y2) / 2
  2 * midpoint_y = 18 := by
    sorry

end NUMINAMATH_GPT_sum_y_coordinates_of_other_vertices_of_parallelogram_l1973_197363


namespace NUMINAMATH_GPT_sandwiches_count_l1973_197326

theorem sandwiches_count (M : ℕ) (C : ℕ) (S : ℕ) (hM : M = 12) (hC : C = 12) (hS : S = 5) :
  M * (C * (C - 1) / 2) * S = 3960 := 
  by sorry

end NUMINAMATH_GPT_sandwiches_count_l1973_197326


namespace NUMINAMATH_GPT_difference_in_square_sides_square_side_length_square_area_greater_than_rectangle_l1973_197302

-- Exploration 1
theorem difference_in_square_sides (a b : ℝ) (h1 : a + b = 20) (h2 : a^2 - b^2 = 40) : a - b = 2 :=
by sorry

-- Exploration 2
theorem square_side_length (x y : ℝ) : (2 * x + 2 * y) / 4 = (x + y) / 2 :=
by sorry

theorem square_area_greater_than_rectangle (x y : ℝ) (h : x > y) : ( (x + y) / 2 ) ^ 2 > x * y :=
by sorry

end NUMINAMATH_GPT_difference_in_square_sides_square_side_length_square_area_greater_than_rectangle_l1973_197302


namespace NUMINAMATH_GPT_identify_wise_l1973_197313

def total_people : ℕ := 30

def is_wise (p : ℕ) : Prop := True   -- This can be further detailed to specify wise characteristics
def is_fool (p : ℕ) : Prop := True    -- This can be further detailed to specify fool characteristics

def wise_count (w : ℕ) : Prop := True -- This indicates the count of wise people
def fool_count (f : ℕ) : Prop := True -- This indicates the count of fool people

def sum_of_groups (wise_groups fool_groups : ℕ) : Prop :=
  wise_groups + fool_groups = total_people

def sum_of_fools (fool_groups : ℕ) (F : ℕ) : Prop :=
  fool_groups = F

theorem identify_wise (F : ℕ) (h1 : F ≤ 8) :
  ∃ (wise_person : ℕ), (wise_person < 30 ∧ is_wise wise_person) :=
by
  sorry

end NUMINAMATH_GPT_identify_wise_l1973_197313


namespace NUMINAMATH_GPT_logan_money_left_l1973_197369

-- Defining the given conditions
def income : ℕ := 65000
def rent_expense : ℕ := 20000
def groceries_expense : ℕ := 5000
def gas_expense : ℕ := 8000
def additional_income_needed : ℕ := 10000

-- Calculating total expenses
def total_expense : ℕ := rent_expense + groceries_expense + gas_expense

-- Desired income
def desired_income : ℕ := income + additional_income_needed

-- The theorem to prove
theorem logan_money_left : (desired_income - total_expense) = 42000 :=
by
  -- A placeholder for the proof
  sorry

end NUMINAMATH_GPT_logan_money_left_l1973_197369


namespace NUMINAMATH_GPT_algebraic_expression_value_l1973_197340

variables (m n x y : ℤ)

def condition1 := m - n = 100
def condition2 := x + y = -1

theorem algebraic_expression_value :
  condition1 m n → condition2 x y → (n + x) - (m - y) = -101 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1973_197340


namespace NUMINAMATH_GPT_sequence_general_term_l1973_197332

theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 2^n) :
  ∀ n : ℕ, a n = 2^n :=
sorry

end NUMINAMATH_GPT_sequence_general_term_l1973_197332


namespace NUMINAMATH_GPT_contrapositive_statement_l1973_197320

theorem contrapositive_statement (m : ℝ) : 
  (¬ ∃ (x : ℝ), x^2 + x - m = 0) → m > 0 :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_statement_l1973_197320


namespace NUMINAMATH_GPT_inequality_proof_l1973_197394

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a^3 + b^3 = 2) :
  (1 / a) + (1 / b) ≥ 2 * (a^2 - a + 1) * (b^2 - b + 1) := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1973_197394


namespace NUMINAMATH_GPT_min_intersection_l1973_197361

open Finset

-- Definition of subset count function
def n (S : Finset ℕ) : ℕ :=
  2 ^ S.card

theorem min_intersection {A B C : Finset ℕ} (hA : A.card = 100) (hB : B.card = 100) 
  (h_subsets : n A + n B + n C = n (A ∪ B ∪ C)) :
  (A ∩ B ∩ C).card ≥ 97 := by
  sorry

end NUMINAMATH_GPT_min_intersection_l1973_197361


namespace NUMINAMATH_GPT_real_roots_range_real_roots_specific_value_l1973_197321

-- Part 1
theorem real_roots_range (a b m : ℝ) (h_eq : a ≠ 0) (h_discriminant : b^2 - 4 * a * m ≥ 0) :
  m ≤ (b^2) / (4 * a) :=
sorry

-- Part 2
theorem real_roots_specific_value (x1 x2 m : ℝ) (h_sum : x1 + x2 = 4) (h_product : x1 * x2 = m)
  (h_condition : x1^2 + x2^2 + (x1 * x2)^2 = 40) (h_range : m ≤ 4) :
  m = -4 :=
sorry

end NUMINAMATH_GPT_real_roots_range_real_roots_specific_value_l1973_197321


namespace NUMINAMATH_GPT_journey_speed_first_half_l1973_197343

noncomputable def speed_first_half (total_time : ℝ) (total_distance : ℝ) (second_half_speed : ℝ) : ℝ :=
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let second_half_time := second_half_distance / second_half_speed
  let first_half_time := total_time - second_half_time
  first_half_distance / first_half_time

theorem journey_speed_first_half
  (total_time : ℝ) (total_distance : ℝ) (second_half_speed : ℝ)
  (h1 : total_time = 10)
  (h2 : total_distance = 224)
  (h3 : second_half_speed = 24) :
  speed_first_half total_time total_distance second_half_speed = 21 := by
  sorry

end NUMINAMATH_GPT_journey_speed_first_half_l1973_197343


namespace NUMINAMATH_GPT_simplify_expression_l1973_197344

noncomputable def expr1 := (Real.sqrt 462) / (Real.sqrt 330)
noncomputable def expr2 := (Real.sqrt 245) / (Real.sqrt 175)
noncomputable def expr_simplified := (12 * Real.sqrt 35) / 25

theorem simplify_expression :
  expr1 + expr2 = expr_simplified :=
sorry

end NUMINAMATH_GPT_simplify_expression_l1973_197344


namespace NUMINAMATH_GPT_max_blocks_fit_l1973_197336

theorem max_blocks_fit :
  ∃ (blocks : ℕ), blocks = 12 ∧ 
  (∀ (a b c : ℕ), a = 3 ∧ b = 2 ∧ c = 1 → 
  ∀ (x y z : ℕ), x = 5 ∧ y = 4 ∧ z = 4 → 
  blocks = (x * y * z) / (a * b * c) ∧
  blocks = (y * z / (b * c) * (5 / a))) :=
sorry

end NUMINAMATH_GPT_max_blocks_fit_l1973_197336


namespace NUMINAMATH_GPT_trigonometric_identity_l1973_197370

open Real

theorem trigonometric_identity (α : ℝ) (h : α ∈ Set.Ioo (-π) (-π / 2)) : 
  sqrt ((1 + cos α) / (1 - cos α)) - sqrt ((1 - cos α) / (1 + cos α)) = 2 / tan α := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1973_197370


namespace NUMINAMATH_GPT_relay_race_total_time_l1973_197365

theorem relay_race_total_time :
  let t1 := 55
  let t2 := t1 + 0.25 * t1
  let t3 := t2 - 0.20 * t2
  let t4 := t1 + 0.30 * t1
  let t5 := 80
  let t6 := t5 - 0.20 * t5
  let t7 := t5 + 0.15 * t5
  let t8 := t7 - 0.05 * t7
  t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 = 573.65 :=
by
  sorry

end NUMINAMATH_GPT_relay_race_total_time_l1973_197365


namespace NUMINAMATH_GPT_min_value_expression_l1973_197319

theorem min_value_expression (x : ℝ) (h : x > 1) : x + 9 / x - 2 ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1973_197319


namespace NUMINAMATH_GPT_line_through_intersection_and_origin_l1973_197315

-- Define the equations of the lines
def line1 (x y : ℝ) : Prop := 2023 * x - 2022 * y - 1 = 0
def line2 (x y : ℝ) : Prop := 2022 * x + 2023 * y + 1 = 0

-- Define the line passing through the origin
def line_pass_origin (x y : ℝ) : Prop := 4045 * x + y = 0

-- Define the intersection point of the two lines
def intersection (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define the theorem stating the desired property
theorem line_through_intersection_and_origin (x y : ℝ)
    (h1 : intersection x y)
    (h2 : x = 0 ∧ y = 0) :
    line_pass_origin x y :=
by
    sorry

end NUMINAMATH_GPT_line_through_intersection_and_origin_l1973_197315


namespace NUMINAMATH_GPT_smallest_k_divisibility_l1973_197346

theorem smallest_k_divisibility : ∃ (k : ℕ), k > 1 ∧ (k % 19 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) ∧ k = 400 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_divisibility_l1973_197346


namespace NUMINAMATH_GPT_max_marks_l1973_197350

theorem max_marks (score shortfall passing_threshold : ℝ) (h1 : score = 212) (h2 : shortfall = 19) (h3 : passing_threshold = 0.30) :
  ∃ M, M = 770 :=
by
  sorry

end NUMINAMATH_GPT_max_marks_l1973_197350


namespace NUMINAMATH_GPT_range_of_a_l1973_197383

theorem range_of_a (a : ℝ) : 
  (2 * (-1) + 0 + a) * (2 * 2 + (-1) + a) < 0 ↔ -3 < a ∧ a < 2 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l1973_197383


namespace NUMINAMATH_GPT_maximum_value_x2y_y2z_z2x_l1973_197307

theorem maximum_value_x2y_y2z_z2x (x y z : ℝ) (h_sum : x + y + z = 0) (h_squares : x^2 + y^2 + z^2 = 6) :
  x^2 * y + y^2 * z + z^2 * x ≤ 6 :=
sorry

end NUMINAMATH_GPT_maximum_value_x2y_y2z_z2x_l1973_197307


namespace NUMINAMATH_GPT_total_cows_l1973_197338

theorem total_cows (cows_per_herd : Nat) (herds : Nat) (total_cows : Nat) : 
  cows_per_herd = 40 → herds = 8 → total_cows = cows_per_herd * herds → total_cows = 320 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_total_cows_l1973_197338


namespace NUMINAMATH_GPT_vertical_asymptote_l1973_197387

theorem vertical_asymptote (x : ℝ) : (y = (2*x - 3) / (4*x + 5)) → (4*x + 5 = 0) → x = -5/4 := 
by 
  intros h1 h2
  sorry

end NUMINAMATH_GPT_vertical_asymptote_l1973_197387


namespace NUMINAMATH_GPT_max_watches_two_hours_l1973_197375

noncomputable def show_watched_each_day : ℕ := 30 -- Time in minutes
def days_watched : ℕ := 4 -- Monday to Thursday

theorem max_watches_two_hours :
  (days_watched * show_watched_each_day) / 60 = 2 := by
  sorry

end NUMINAMATH_GPT_max_watches_two_hours_l1973_197375


namespace NUMINAMATH_GPT_positive_difference_two_numbers_l1973_197397

theorem positive_difference_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 80) : |x - y| = 8 := by
  sorry

end NUMINAMATH_GPT_positive_difference_two_numbers_l1973_197397


namespace NUMINAMATH_GPT_common_difference_l1973_197329

noncomputable def a_n (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

theorem common_difference (d : ℕ) (a1 : ℕ) (h1 : a1 = 18) (h2 : d ≠ 0) 
  (h3 : (a1 + 3 * d)^2 = a1 * (a1 + 7 * d)) : d = 2 :=
by
  sorry

end NUMINAMATH_GPT_common_difference_l1973_197329


namespace NUMINAMATH_GPT_f_periodic_l1973_197390

noncomputable def f (x : ℝ) : ℝ := sorry

theorem f_periodic (f : ℝ → ℝ)
  (h_bound : ∀ x : ℝ, |f x| ≤ 1)
  (h_func : ∀ x : ℝ, f (x + 13 / 42) + f x = f (x + 1 / 6) + f (x + 1 / 7)) :
  ∀ x : ℝ, f (x + 1) = f x :=
sorry

end NUMINAMATH_GPT_f_periodic_l1973_197390


namespace NUMINAMATH_GPT_bhupathi_amount_l1973_197399

variable (A B : ℝ)

theorem bhupathi_amount :
  (A + B = 1210 ∧ (4 / 15) * A = (2 / 5) * B) → B = 484 :=
by
  sorry

end NUMINAMATH_GPT_bhupathi_amount_l1973_197399


namespace NUMINAMATH_GPT_commercial_break_total_time_l1973_197396

theorem commercial_break_total_time (c1 c2 c3 : ℕ) (c4 : ℕ → ℕ) (interrupt restart : ℕ) 
  (h1 : c1 = 5) (h2 : c2 = 6) (h3 : c3 = 7) 
  (h4 : ∀ i, i < 11 → c4 i = 2) 
  (h_interrupt : interrupt = 3)
  (h_restart : restart = 2) :
  c1 + c2 + c3 + (11 * 2) + interrupt + 2 * restart = 47 := 
  by
  sorry

end NUMINAMATH_GPT_commercial_break_total_time_l1973_197396


namespace NUMINAMATH_GPT_intersection_points_l1973_197378

noncomputable def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 10)^2 = 50
noncomputable def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2 * (x - y) - 18 = 0

theorem intersection_points : 
  (circle1 3 3 ∧ circle2 3 3) ∧ (circle1 (-3) 5 ∧ circle2 (-3) 5) :=
by sorry

end NUMINAMATH_GPT_intersection_points_l1973_197378


namespace NUMINAMATH_GPT_Haley_sweaters_l1973_197362

theorem Haley_sweaters (machine_capacity loads shirts sweaters : ℕ) 
    (h_capacity : machine_capacity = 7)
    (h_loads : loads = 5)
    (h_shirts : shirts = 2)
    (h_sweaters_total : sweaters = loads * machine_capacity - shirts) :
  sweaters = 33 :=
by 
  rw [h_capacity, h_loads, h_shirts] at h_sweaters_total
  exact h_sweaters_total

end NUMINAMATH_GPT_Haley_sweaters_l1973_197362


namespace NUMINAMATH_GPT_remaining_pictures_l1973_197379

-- Definitions based on the conditions
def pictures_in_first_book : ℕ := 44
def pictures_in_second_book : ℕ := 35
def pictures_in_third_book : ℕ := 52
def pictures_in_fourth_book : ℕ := 48
def colored_pictures : ℕ := 37

-- Statement of the theorem based on the question and correct answer
theorem remaining_pictures :
  pictures_in_first_book + pictures_in_second_book + pictures_in_third_book + pictures_in_fourth_book - colored_pictures = 142 := by
  sorry

end NUMINAMATH_GPT_remaining_pictures_l1973_197379


namespace NUMINAMATH_GPT_aero_flight_tees_per_package_l1973_197305

theorem aero_flight_tees_per_package {A : ℕ} :
  (∀ (num_people : ℕ), num_people = 4 → 20 * num_people ≤ A * 28 + 2 * 12) →
  A * 28 ≥ 56 →
  A = 2 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_aero_flight_tees_per_package_l1973_197305


namespace NUMINAMATH_GPT_fbox_eval_correct_l1973_197384

-- Define the function according to the condition
def fbox (a b c : ℕ) : ℕ := a^b - b^c + c^a

-- Propose the theorem 
theorem fbox_eval_correct : fbox 2 0 3 = 10 := 
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_fbox_eval_correct_l1973_197384


namespace NUMINAMATH_GPT_expand_and_simplify_l1973_197352

variable (x : ℝ)

theorem expand_and_simplify : (7 * x - 3) * 3 * x^2 = 21 * x^3 - 9 * x^2 := by
  sorry

end NUMINAMATH_GPT_expand_and_simplify_l1973_197352


namespace NUMINAMATH_GPT_money_left_after_purchases_l1973_197324

variable (initial_money : ℝ) (fraction_for_cupcakes : ℝ) (money_spent_on_milkshake : ℝ)

theorem money_left_after_purchases (h_initial : initial_money = 10)
  (h_fraction : fraction_for_cupcakes = 1/5)
  (h_milkshake : money_spent_on_milkshake = 5) :
  initial_money - (initial_money * fraction_for_cupcakes) - money_spent_on_milkshake = 3 := 
by
  sorry

end NUMINAMATH_GPT_money_left_after_purchases_l1973_197324


namespace NUMINAMATH_GPT_probability_Xavier_Yvonne_not_Zelda_l1973_197316

theorem probability_Xavier_Yvonne_not_Zelda
    (P_Xavier : ℚ)
    (P_Yvonne : ℚ)
    (P_Zelda : ℚ)
    (hXavier : P_Xavier = 1/3)
    (hYvonne : P_Yvonne = 1/2)
    (hZelda : P_Zelda = 5/8) :
    (P_Xavier * P_Yvonne * (1 - P_Zelda) = 1/16) :=
  by
  rw [hXavier, hYvonne, hZelda]
  sorry

end NUMINAMATH_GPT_probability_Xavier_Yvonne_not_Zelda_l1973_197316


namespace NUMINAMATH_GPT_min_percentage_of_people_owning_95_percent_money_l1973_197351

theorem min_percentage_of_people_owning_95_percent_money 
  (total_people: ℕ) (total_money: ℕ) 
  (P: ℕ) (M: ℕ) 
  (H1: P = total_people * 10 / 100) 
  (H2: M = total_money * 90 / 100)
  (H3: ∀ (people_owning_90_percent: ℕ), people_owning_90_percent = P → people_owning_90_percent * some_money = M) :
      P = total_people * 55 / 100 := 
sorry

end NUMINAMATH_GPT_min_percentage_of_people_owning_95_percent_money_l1973_197351


namespace NUMINAMATH_GPT_P_2017_eq_14_l1973_197377

def sumOfDigits (n : Nat) : Nat :=
  n.digits 10 |>.sum

def numberOfDigits (n : Nat) : Nat :=
  n.digits 10 |>.length

def P (n : Nat) : Nat :=
  sumOfDigits n + numberOfDigits n

theorem P_2017_eq_14 : P 2017 = 14 :=
by
  sorry

end NUMINAMATH_GPT_P_2017_eq_14_l1973_197377


namespace NUMINAMATH_GPT_solve_quadratic_l1973_197335

theorem solve_quadratic : 
  ∃ x1 x2 : ℝ, (x1 = -2 + Real.sqrt 2) ∧ (x2 = -2 - Real.sqrt 2) ∧ (∀ x : ℝ, x^2 + 4 * x + 2 = 0 → (x = x1 ∨ x = x2)) :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_quadratic_l1973_197335


namespace NUMINAMATH_GPT_units_digit_product_l1973_197306

theorem units_digit_product (k l : ℕ) (h1 : ∀ n : ℕ, (5^n % 10) = 5) (h2 : ∀ m < 4, (6^m % 10) = 6) :
  ((5^k * 6^l) % 10) = 0 :=
by
  have h5 : (5^k % 10) = 5 := h1 k
  have h6 : (6^4 % 10) = 6 := h2 4 (by sorry)
  have h_product : (5^k * 6^l % 10) = ((5 % 10) * (6 % 10) % 10) := sorry
  norm_num at h_product
  exact h_product

end NUMINAMATH_GPT_units_digit_product_l1973_197306


namespace NUMINAMATH_GPT_new_member_younger_by_160_l1973_197392

theorem new_member_younger_by_160 
  (A : ℕ)  -- average age 8 years ago and today
  (O N : ℕ)  -- age of the old member and the new member respectively
  (h1 : 20 * A = 20 * A + O - N)  -- condition derived from the problem
  (h2 : 20 * 8 = 160)  -- age increase over 8 years for 20 members
  (h3 : O - N = 160) : O - N = 160 :=
by
  sorry

end NUMINAMATH_GPT_new_member_younger_by_160_l1973_197392


namespace NUMINAMATH_GPT_total_shaded_area_l1973_197391

def rectangle_area (R : ℝ) : ℝ := R * R
def square_area (S : ℝ) : ℝ := S * S

theorem total_shaded_area 
  (R S : ℝ)
  (h1 : 18 = 2 * R)
  (h2 : R = 4 * S) :
  rectangle_area R + 12 * square_area S = 141.75 := 
  by 
    sorry

end NUMINAMATH_GPT_total_shaded_area_l1973_197391


namespace NUMINAMATH_GPT_circumscribed_sphere_surface_area_l1973_197333

-- Define the setup and conditions for the right circular cone and its circumscribed sphere
theorem circumscribed_sphere_surface_area (PA PB PC AB R : ℝ)
  (h1 : AB = Real.sqrt 2)
  (h2 : PA = 1)
  (h3 : PB = 1)
  (h4 : PC = 1)
  (h5 : R = Real.sqrt 3 / 2 * PA) :
  4 * Real.pi * R ^ 2 = 3 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_circumscribed_sphere_surface_area_l1973_197333


namespace NUMINAMATH_GPT_no_integer_solution_l1973_197308

theorem no_integer_solution (x y z : ℤ) (h : x ≠ 0) : ¬(2 * x^4 + 2 * x^2 * y^2 + y^4 = z^2) :=
sorry

end NUMINAMATH_GPT_no_integer_solution_l1973_197308


namespace NUMINAMATH_GPT_shaded_region_area_l1973_197349

theorem shaded_region_area (r : ℝ) (h : r = 5) : 
  8 * (π * r * r / 4 - r * r / 2) / 2 = 50 * (π - 2) :=
by
  sorry

end NUMINAMATH_GPT_shaded_region_area_l1973_197349


namespace NUMINAMATH_GPT_part_a_l1973_197341

theorem part_a (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) : 
  |a - b| + |b - c| + |c - a| ≤ 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_part_a_l1973_197341


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1973_197342

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_a6 : a 6 = 5) (h_a10 : a 10 = 6) : 
  (a 10 - a 6) / 4 = 1 / 4 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1973_197342


namespace NUMINAMATH_GPT_only_one_statement_is_true_l1973_197380

theorem only_one_statement_is_true (A B C D E: Prop)
  (hA : A ↔ B)
  (hB : B ↔ ¬ E)
  (hC : C ↔ (A ∧ B ∧ C ∧ D ∧ E))
  (hD : D ↔ ¬ (A ∨ B ∨ C ∨ D ∨ E))
  (hE : E ↔ ¬ A)
  (h_unique : ∃! x, x = A ∨ x = B ∨ x = C ∨ x = D ∨ x = E ∧ x = True) : E :=
by
  sorry

end NUMINAMATH_GPT_only_one_statement_is_true_l1973_197380


namespace NUMINAMATH_GPT_bruno_pens_l1973_197356

-- Define Bruno's purchase of pens
def one_dozen : Nat := 12
def half_dozen : Nat := one_dozen / 2
def two_and_half_dozens : Nat := 2 * one_dozen + half_dozen

-- State the theorem to be proved
theorem bruno_pens : two_and_half_dozens = 30 :=
by sorry

end NUMINAMATH_GPT_bruno_pens_l1973_197356


namespace NUMINAMATH_GPT_math_proof_problem_l1973_197309

-- Defining the problem condition
def condition (x y z : ℝ) := 
  x^3 + y^3 + z^3 - 3 * x * y * z - 3 * (x^2 + y^2 + z^2 - x * y - y * z - z * x) = 0

-- Adding constraints to x, y, z
def constraints (x y z : ℝ) :=
  0 < x ∧ 0 < y ∧ 0 < z ∧ (x ≠ y ∨ y ≠ z ∨ z ≠ x)

-- Stating the main theorem
theorem math_proof_problem (x y z : ℝ) (h_condition : condition x y z) (h_constraints : constraints x y z) :
  x + y + z = 3 ∧ x^2 * (1 + y) + y^2 * (1 + z) + z^2 * (1 + x) > 6 := 
sorry

end NUMINAMATH_GPT_math_proof_problem_l1973_197309


namespace NUMINAMATH_GPT_triangle_with_sticks_l1973_197327

theorem triangle_with_sticks (c : ℕ) (h₁ : 4 + 9 > c) (h₂ : 9 - 4 < c) :
  c = 9 :=
by
  sorry

end NUMINAMATH_GPT_triangle_with_sticks_l1973_197327


namespace NUMINAMATH_GPT_pos_sum_of_powers_l1973_197358

theorem pos_sum_of_powers (a b c : ℝ) (n : ℕ) (h1 : a * b * c > 0) (h2 : a + b + c > 0) : 
  a^n + b^n + c^n > 0 :=
sorry

end NUMINAMATH_GPT_pos_sum_of_powers_l1973_197358


namespace NUMINAMATH_GPT_opposite_of_2023_l1973_197355

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end NUMINAMATH_GPT_opposite_of_2023_l1973_197355


namespace NUMINAMATH_GPT_probability_calculation_l1973_197300

noncomputable def probability_same_color (pairs_black pairs_brown pairs_gray : ℕ) : ℚ :=
  let total_shoes := 2 * (pairs_black + pairs_brown + pairs_gray)
  let prob_black := (2 * pairs_black : ℚ) / total_shoes * (pairs_black : ℚ) / (total_shoes - 1)
  let prob_brown := (2 * pairs_brown : ℚ) / total_shoes * (pairs_brown : ℚ) / (total_shoes - 1)
  let prob_gray := (2 * pairs_gray : ℚ) / total_shoes * (pairs_gray : ℚ) / (total_shoes - 1)
  prob_black + prob_brown + prob_gray

theorem probability_calculation :
  probability_same_color 7 4 3 = 37 / 189 :=
by
  sorry

end NUMINAMATH_GPT_probability_calculation_l1973_197300


namespace NUMINAMATH_GPT_bridge_length_is_235_l1973_197366

noncomputable def length_of_bridge (train_length : ℕ) (train_speed_kmh : ℕ) (time_sec : ℕ) : ℕ :=
  let train_speed_ms := (train_speed_kmh * 1000) / 3600
  let total_distance := train_speed_ms * time_sec
  let bridge_length := total_distance - train_length
  bridge_length

theorem bridge_length_is_235 :
  length_of_bridge 140 45 30 = 235 :=
by 
  sorry

end NUMINAMATH_GPT_bridge_length_is_235_l1973_197366


namespace NUMINAMATH_GPT_find_m_n_l1973_197325

theorem find_m_n (m n : ℤ) (h : m^2 - 2 * m * n + 2 * n^2 - 8 * n + 16 = 0) : m = 4 ∧ n = 4 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_m_n_l1973_197325


namespace NUMINAMATH_GPT_geometric_sequence_a4_l1973_197389

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ {m n p q}, m + n = p + q → a m * a n = a p * a q

theorem geometric_sequence_a4 (a : ℕ → ℝ) (h : geometric_sequence a) (h2 : a 2 = 4) (h6 : a 6 = 16) :
  a 4 = 8 :=
by {
  -- Here you can provide the proof steps if needed
  sorry
}

end NUMINAMATH_GPT_geometric_sequence_a4_l1973_197389


namespace NUMINAMATH_GPT_model_x_completion_time_l1973_197322

theorem model_x_completion_time (T_x : ℝ) : 
  (24 : ℕ) * (1 / T_x + 1 / 36) = 1 → T_x = 72 := 
by 
  sorry

end NUMINAMATH_GPT_model_x_completion_time_l1973_197322


namespace NUMINAMATH_GPT_LCM_of_numbers_with_HCF_and_ratio_l1973_197334

theorem LCM_of_numbers_with_HCF_and_ratio (a b x : ℕ)
  (h1 : a = 3 * x) 
  (h2 : b = 4 * x)
  (h3 : ∀ y : ℕ, y ∣ a → y ∣ b → y ∣ x)
  (hx : x = 5) :
  Nat.lcm a b = 60 := 
by
  sorry

end NUMINAMATH_GPT_LCM_of_numbers_with_HCF_and_ratio_l1973_197334


namespace NUMINAMATH_GPT_min_female_students_l1973_197353

theorem min_female_students (males females : ℕ) (total : ℕ) (percent_participated : ℕ) (participated : ℕ) (min_females : ℕ)
  (h1 : males = 22) 
  (h2 : females = 18) 
  (h3 : total = males + females)
  (h4 : percent_participated = 60) 
  (h5 : participated = (percent_participated * total) / 100)
  (h6 : min_females = participated - males) :
  min_females = 2 := 
sorry

end NUMINAMATH_GPT_min_female_students_l1973_197353


namespace NUMINAMATH_GPT_solve_quadratic_eq_l1973_197371

theorem solve_quadratic_eq (x : ℝ) : (x - 1) * (x + 2) = 0 ↔ x = 1 ∨ x = -2 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l1973_197371


namespace NUMINAMATH_GPT_cost_of_marker_l1973_197311

theorem cost_of_marker (s c m : ℕ) (h1 : s > 12) (h2 : m > 1) (h3 : c > m) (h4 : s * c * m = 924) : c = 11 :=
sorry

end NUMINAMATH_GPT_cost_of_marker_l1973_197311


namespace NUMINAMATH_GPT_measure_of_angle_B_in_triangle_l1973_197301

theorem measure_of_angle_B_in_triangle
  {a b c : ℝ} {A B C : ℝ} 
  (h1 : a * c = b^2 - a^2)
  (h2 : A = Real.pi / 6)
  (h3 : a / Real.sin A = b / Real.sin B) 
  (h4 : b / Real.sin B = c / Real.sin C)
  (h5 : A + B + C = Real.pi) :
  B = Real.pi / 3 :=
by sorry

end NUMINAMATH_GPT_measure_of_angle_B_in_triangle_l1973_197301


namespace NUMINAMATH_GPT_bucket_full_weight_l1973_197372

variable (c d : ℝ)

def total_weight_definition (x y : ℝ) := x + y

theorem bucket_full_weight (x y : ℝ) 
  (h₁ : x + 3/4 * y = c) 
  (h₂ : x + 1/3 * y = d) : 
  total_weight_definition x y = (8 * c - 3 * d) / 5 :=
sorry

end NUMINAMATH_GPT_bucket_full_weight_l1973_197372


namespace NUMINAMATH_GPT_village_population_l1973_197386

variable (Px : ℕ)
variable (py : ℕ := 42000)
variable (years : ℕ := 16)
variable (rate_decrease_x : ℕ := 1200)
variable (rate_increase_y : ℕ := 800)

theorem village_population (Px : ℕ) (py : ℕ := 42000)
  (years : ℕ := 16) (rate_decrease_x : ℕ := 1200)
  (rate_increase_y : ℕ := 800) :
  Px - rate_decrease_x * years = py + rate_increase_y * years → Px = 74000 := by
  sorry

end NUMINAMATH_GPT_village_population_l1973_197386


namespace NUMINAMATH_GPT_math_problem_l1973_197339

theorem math_problem (x y : ℝ) 
  (h1 : 1/5 + x + y = 1) 
  (h2 : 1/5 * 1 + 2 * x + 3 * y = 11/5) : 
  (x = 2/5) ∧ 
  (y = 2/5) ∧ 
  (1/5 + x = 3/5) ∧ 
  ((1 - 11/5)^2 * (1/5) + (2 - 11/5)^2 * (2/5) + (3 - 11/5)^2 * (2/5) = 14/25) :=
by {
  sorry
}

end NUMINAMATH_GPT_math_problem_l1973_197339


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l1973_197310

theorem quadratic_has_two_distinct_real_roots (k : ℝ) (h1 : 4 + 4 * k > 0) (h2 : k ≠ 0) :
  k > -1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l1973_197310


namespace NUMINAMATH_GPT_total_rowing_campers_l1973_197337

theorem total_rowing_campers (morning_rowing afternoon_rowing : ℕ) : 
  morning_rowing = 13 -> 
  afternoon_rowing = 21 -> 
  morning_rowing + afternoon_rowing = 34 :=
by
  sorry

end NUMINAMATH_GPT_total_rowing_campers_l1973_197337


namespace NUMINAMATH_GPT_ninety_one_square_friendly_unique_square_friendly_l1973_197304

-- Given conditions
def square_friendly (c : ℤ) : Prop :=
  ∀ m : ℤ, ∃ n : ℤ, m^2 + 18 * m + c = n^2

-- Part (a)
theorem ninety_one_square_friendly : square_friendly 81 :=
sorry

-- Part (b)
theorem unique_square_friendly (c c' : ℤ) (h_c : square_friendly c) (h_c' : square_friendly c') : c = c' :=
sorry

end NUMINAMATH_GPT_ninety_one_square_friendly_unique_square_friendly_l1973_197304


namespace NUMINAMATH_GPT_question1_question2_l1973_197347

noncomputable def f (x : ℝ) : ℝ :=
  if x < -4 then -x - 9
  else if x < 1 then 3 * x + 7
  else x + 9

theorem question1 (x : ℝ) (h : -10 ≤ x ∧ x ≤ -2) : f x ≤ 1 := sorry

theorem question2 (x a : ℝ) (hx : x > 1) (h : f x > -x^2 + a * x) : a < 7 := sorry

end NUMINAMATH_GPT_question1_question2_l1973_197347


namespace NUMINAMATH_GPT_lines_intersect_l1973_197359

theorem lines_intersect (m b : ℝ) (h1 : 17 = 2 * m * 4 + 5) (h2 : 17 = 4 * 4 + b) : b + m = 2.5 :=
by {
    sorry
}

end NUMINAMATH_GPT_lines_intersect_l1973_197359


namespace NUMINAMATH_GPT_find_divisor_l1973_197368

theorem find_divisor (d : ℕ) (h1 : d ∣ (9671 - 1)) : d = 9670 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l1973_197368


namespace NUMINAMATH_GPT_domain_of_f_l1973_197348

noncomputable def domain_of_function (x : ℝ) : Set ℝ :=
  {x | 4 - x ^ 2 ≥ 0 ∧ x ≠ 1}

theorem domain_of_f (x : ℝ) : domain_of_function x = {x | -2 ≤ x ∧ x < 1 ∨ 1 < x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l1973_197348


namespace NUMINAMATH_GPT_algebraic_expression_value_l1973_197360

theorem algebraic_expression_value (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 2 * x^2 + 3 * x - 7 = -6 :=
by sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1973_197360


namespace NUMINAMATH_GPT_car_speed_40_kmph_l1973_197357

theorem car_speed_40_kmph (v : ℝ) (h : 1 / v = 1 / 48 + 15 / 3600) : v = 40 := 
sorry

end NUMINAMATH_GPT_car_speed_40_kmph_l1973_197357


namespace NUMINAMATH_GPT_find_abc_l1973_197367

noncomputable def x (t : ℝ) := 3 * Real.cos t - 2 * Real.sin t
noncomputable def y (t : ℝ) := 3 * Real.sin t

theorem find_abc :
  ∃ a b c : ℝ, 
  (a = 1/9) ∧ 
  (b = 4/27) ∧ 
  (c = 5/27) ∧ 
  (∀ t : ℝ, a * (x t)^2 + b * (x t) * (y t) + c * (y t)^2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_abc_l1973_197367


namespace NUMINAMATH_GPT_ratio_new_values_l1973_197385

theorem ratio_new_values (x y x2 y2 : ℝ) (h1 : x / y = 7 / 5) (h2 : x2 = x * y) (h3 : y2 = y * x) : x2 / y2 = 1 := by
  sorry

end NUMINAMATH_GPT_ratio_new_values_l1973_197385


namespace NUMINAMATH_GPT_odd_numbers_divisibility_l1973_197345

theorem odd_numbers_divisibility 
  (a b c : ℤ) 
  (h_a_odd : a % 2 = 1) 
  (h_b_odd : b % 2 = 1) 
  (h_c_odd : c % 2 = 1) 
  : (ab - 1) % 4 = 0 ∨ (bc - 1) % 4 = 0 ∨ (ca - 1) % 4 = 0 := 
sorry

end NUMINAMATH_GPT_odd_numbers_divisibility_l1973_197345


namespace NUMINAMATH_GPT_binom_15_4_eq_1365_l1973_197331

theorem binom_15_4_eq_1365 : (Nat.choose 15 4) = 1365 := 
by 
  sorry

end NUMINAMATH_GPT_binom_15_4_eq_1365_l1973_197331


namespace NUMINAMATH_GPT_find_a_for_even_function_l1973_197388

theorem find_a_for_even_function (a : ℝ) (f : ℝ → ℝ) (hf : ∀ x : ℝ, f x = (x + 1) * (x + a) ∧ f (-x) = f x) : a = -1 := by 
  sorry

end NUMINAMATH_GPT_find_a_for_even_function_l1973_197388


namespace NUMINAMATH_GPT_loss_percentage_l1973_197312

theorem loss_percentage (C : ℝ) (h : 40 * C = 100 * C) : 
  ∃ L : ℝ, L = 60 := 
sorry

end NUMINAMATH_GPT_loss_percentage_l1973_197312


namespace NUMINAMATH_GPT_combined_degrees_l1973_197318

variable (Summer_deg Jolly_deg : ℕ)

def Summer_has_150_degrees := Summer_deg = 150

def Summer_has_5_more_degrees_than_Jolly := Summer_deg = Jolly_deg + 5

theorem combined_degrees (h1 : Summer_has_150_degrees Summer_deg) (h2 : Summer_has_5_more_degrees_than_Jolly Summer_deg Jolly_deg) :
  Summer_deg + Jolly_deg = 295 :=
by
  sorry

end NUMINAMATH_GPT_combined_degrees_l1973_197318


namespace NUMINAMATH_GPT_gardener_b_time_l1973_197374

theorem gardener_b_time :
  ∃ x : ℝ, (1 / 3 + 1 / x = 1 / 1.875) → (x = 5) := by
  sorry

end NUMINAMATH_GPT_gardener_b_time_l1973_197374


namespace NUMINAMATH_GPT_p_at_0_l1973_197330

noncomputable def p : Polynomial ℚ := sorry

theorem p_at_0 :
  (∀ n : ℕ, n ≤ 6 → p.eval (2^n) = 1 / (2^n))
  ∧ p.degree = 6 → 
  p.eval 0 = 127 / 64 :=
sorry

end NUMINAMATH_GPT_p_at_0_l1973_197330


namespace NUMINAMATH_GPT_number_is_a_l1973_197303

theorem number_is_a (x y z a : ℝ) (h1 : x + y + z = a) (h2 : (1 / x) + (1 / y) + (1 / z) = 1 / a) : 
  x = a ∨ y = a ∨ z = a :=
sorry

end NUMINAMATH_GPT_number_is_a_l1973_197303


namespace NUMINAMATH_GPT_beads_per_necklace_l1973_197328

-- Definitions based on conditions
def total_beads_used (N : ℕ) : ℕ :=
  10 * N + 2 * N + 50 + 35

-- Main theorem to prove the number of beads needed for one beaded necklace
theorem beads_per_necklace (N : ℕ) (h : total_beads_used N = 325) : N = 20 :=
by
  sorry

end NUMINAMATH_GPT_beads_per_necklace_l1973_197328


namespace NUMINAMATH_GPT_ratio_of_investments_l1973_197398

theorem ratio_of_investments (P Q : ℝ)
  (h_ratio_profits : (20 * P) / (40 * Q) = 7 / 10) : P / Q = 7 / 5 := 
sorry

end NUMINAMATH_GPT_ratio_of_investments_l1973_197398


namespace NUMINAMATH_GPT_rectangle_area_inscribed_circle_l1973_197393

theorem rectangle_area_inscribed_circle {r w l : ℕ} (h1 : r = 7) (h2 : w = 2 * r) (h3 : l = 3 * w) : l * w = 588 :=
by 
  -- The proof details are omitted as per instructions.
  sorry

end NUMINAMATH_GPT_rectangle_area_inscribed_circle_l1973_197393


namespace NUMINAMATH_GPT_slope_of_chord_in_ellipse_l1973_197323

noncomputable def slope_of_chord (x1 y1 x2 y2 : ℝ) : ℝ :=
  (y1 - y2) / (x1 - x2)

theorem slope_of_chord_in_ellipse :
  ∀ (x1 y1 x2 y2 : ℝ),
    (x1^2 / 16 + y1^2 / 9 = 1) →
    (x2^2 / 16 + y2^2 / 9 = 1) →
    ((x1 + x2) = -2) →
    ((y1 + y2) = 4) →
    slope_of_chord x1 y1 x2 y2 = 9 / 32 :=
by
  intro x1 y1 x2 y2 h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_slope_of_chord_in_ellipse_l1973_197323


namespace NUMINAMATH_GPT_bamboo_sections_volume_l1973_197395

theorem bamboo_sections_volume (a : ℕ → ℚ) (d : ℚ) :
  (∀ n, a n = a 0 + n * d) →
  (a 0 + a 1 + a 2 = 4) →
  (a 5 + a 6 + a 7 + a 8 = 3) →
  (a 3 + a 4 = 2 + 3 / 22) :=
sorry

end NUMINAMATH_GPT_bamboo_sections_volume_l1973_197395


namespace NUMINAMATH_GPT_fraction_difference_of_squares_l1973_197373

theorem fraction_difference_of_squares :
  (175^2 - 155^2) / 20 = 330 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_fraction_difference_of_squares_l1973_197373


namespace NUMINAMATH_GPT_length_of_the_train_l1973_197376

noncomputable def train_speed_kmph : ℝ := 45
noncomputable def time_to_cross_seconds : ℝ := 30
noncomputable def bridge_length_meters : ℝ := 205

noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600
noncomputable def distance_crossed_meters : ℝ := train_speed_mps * time_to_cross_seconds

theorem length_of_the_train 
  (h1 : train_speed_kmph = 45)
  (h2 : time_to_cross_seconds = 30)
  (h3 : bridge_length_meters = 205) : 
  distance_crossed_meters - bridge_length_meters = 170 := 
by
  sorry

end NUMINAMATH_GPT_length_of_the_train_l1973_197376


namespace NUMINAMATH_GPT_area_of_circle_l1973_197317

-- Define the given conditions
def pi_approx : ℝ := 3
def radius : ℝ := 0.6

-- Prove that the area is 1.08 given the conditions
theorem area_of_circle : π = pi_approx → radius = 0.6 → 
  (pi_approx * radius^2 = 1.08) :=
by
  intros hπ hr
  sorry

end NUMINAMATH_GPT_area_of_circle_l1973_197317


namespace NUMINAMATH_GPT_trader_gain_percentage_l1973_197382

-- Definition of the given conditions
def cost_per_pen (C : ℝ) := C
def num_pens_sold := 90
def gain_from_sale (C : ℝ) := 15 * C
def total_cost (C : ℝ) := 90 * C

-- Statement of the problem
theorem trader_gain_percentage (C : ℝ) : 
  (((gain_from_sale C) / (total_cost C)) * 100) = 16.67 :=
by
  -- This part will contain the step-by-step proof, omitted here
  sorry

end NUMINAMATH_GPT_trader_gain_percentage_l1973_197382


namespace NUMINAMATH_GPT_shaded_solid_volume_l1973_197354

noncomputable def volume_rectangular_prism (length width height : ℕ) : ℕ :=
  length * width * height

theorem shaded_solid_volume :
  volume_rectangular_prism 4 5 6 - volume_rectangular_prism 1 2 4 = 112 :=
by
  sorry

end NUMINAMATH_GPT_shaded_solid_volume_l1973_197354


namespace NUMINAMATH_GPT_midpoint_trajectory_l1973_197381

theorem midpoint_trajectory (x y p q : ℝ) (h_parabola : p^2 = 4 * q)
  (h_focus : ∀ (p q : ℝ), p^2 = 4 * q → q = (p/2)^2) 
  (h_midpoint_x : x = (p + 1) / 2)
  (h_midpoint_y : y = q / 2):
  y^2 = 2 * x - 1 :=
by
  sorry

end NUMINAMATH_GPT_midpoint_trajectory_l1973_197381
