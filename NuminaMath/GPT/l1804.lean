import Mathlib

namespace NUMINAMATH_GPT_scientific_notation_135000_l1804_180435

theorem scientific_notation_135000 :
  135000 = 1.35 * 10^5 := sorry

end NUMINAMATH_GPT_scientific_notation_135000_l1804_180435


namespace NUMINAMATH_GPT_range_a_of_function_has_two_zeros_l1804_180466

noncomputable def f (a x : ℝ) : ℝ := a^x - x - a

theorem range_a_of_function_has_two_zeros (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) : 
  1 < a :=
sorry

end NUMINAMATH_GPT_range_a_of_function_has_two_zeros_l1804_180466


namespace NUMINAMATH_GPT_savings_per_bagel_in_cents_l1804_180404

theorem savings_per_bagel_in_cents (cost_individual : ℝ) (cost_dozen : ℝ) (dozen : ℕ) (cents_per_dollar : ℕ) :
  cost_individual = 2.25 →
  cost_dozen = 24 →
  dozen = 12 →
  cents_per_dollar = 100 →
  (cost_individual * cents_per_dollar - (cost_dozen / dozen) * cents_per_dollar) = 25 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_savings_per_bagel_in_cents_l1804_180404


namespace NUMINAMATH_GPT_number_of_students_in_the_course_l1804_180482

variable (T : ℝ)

theorem number_of_students_in_the_course
  (h1 : (1/5) * T + (1/4) * T + (1/2) * T + 40 = T) :
  T = 800 :=
sorry

end NUMINAMATH_GPT_number_of_students_in_the_course_l1804_180482


namespace NUMINAMATH_GPT_simplify_negative_exponents_l1804_180478

theorem simplify_negative_exponents (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x + y)⁻¹ * (x⁻¹ + y⁻¹) = x⁻¹ * y⁻¹ :=
  sorry

end NUMINAMATH_GPT_simplify_negative_exponents_l1804_180478


namespace NUMINAMATH_GPT_find_sam_age_l1804_180444

variable (Sam Drew : ℕ)

-- Conditions as definitions in Lean 4
def combined_age (Sam Drew : ℕ) : Prop := Sam + Drew = 54
def sam_half_drew (Sam Drew : ℕ) : Prop := Sam = Drew / 2

theorem find_sam_age (Sam Drew : ℕ) (h1 : combined_age Sam Drew) (h2 : sam_half_drew Sam Drew) : Sam = 18 :=
sorry

end NUMINAMATH_GPT_find_sam_age_l1804_180444


namespace NUMINAMATH_GPT_total_action_figures_l1804_180451

theorem total_action_figures (initial_figures cost_per_figure total_cost needed_figures : ℕ)
  (h1 : initial_figures = 7)
  (h2 : cost_per_figure = 8)
  (h3 : total_cost = 72)
  (h4 : needed_figures = total_cost / cost_per_figure)
  : initial_figures + needed_figures = 16 :=
by
  sorry

end NUMINAMATH_GPT_total_action_figures_l1804_180451


namespace NUMINAMATH_GPT_shelves_needed_l1804_180401

theorem shelves_needed (initial_stock : ℕ) (additional_shipment : ℕ) (bears_per_shelf : ℕ) (total_bears : ℕ) (shelves : ℕ) :
  initial_stock = 4 → 
  additional_shipment = 10 → 
  bears_per_shelf = 7 → 
  total_bears = initial_stock + additional_shipment →
  total_bears / bears_per_shelf = shelves →
  shelves = 2 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_shelves_needed_l1804_180401


namespace NUMINAMATH_GPT_number_of_groups_eq_five_l1804_180403

-- Define conditions
def total_eggs : ℕ := 35
def eggs_per_group : ℕ := 7

-- Statement to prove the number of groups
theorem number_of_groups_eq_five : total_eggs / eggs_per_group = 5 := by
  sorry

end NUMINAMATH_GPT_number_of_groups_eq_five_l1804_180403


namespace NUMINAMATH_GPT_jeans_cost_proof_l1804_180476

def cheaper_jeans_cost (coat_price: Float) (backpack_price: Float) (shoes_price: Float) (subtotal: Float) (difference: Float): Float :=
  let known_items_cost := coat_price + backpack_price + shoes_price
  let jeans_total_cost := subtotal - known_items_cost
  let x := (jeans_total_cost - difference) / 2
  x

def more_expensive_jeans_cost (cheaper_price : Float) (difference: Float): Float :=
  cheaper_price + difference

theorem jeans_cost_proof : ∀ (coat_price backpack_price shoes_price subtotal difference : Float),
  coat_price = 45 →
  backpack_price = 25 →
  shoes_price = 30 →
  subtotal = 139 →
  difference = 15 →
  cheaper_jeans_cost coat_price backpack_price shoes_price subtotal difference = 12 ∧
  more_expensive_jeans_cost (cheaper_jeans_cost coat_price backpack_price shoes_price subtotal difference) difference = 27 :=
by
  intros coat_price backpack_price shoes_price subtotal difference
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_jeans_cost_proof_l1804_180476


namespace NUMINAMATH_GPT_final_bill_is_correct_l1804_180473

def Alicia_order := [7.50, 4.00, 5.00]
def Brant_order := [10.00, 4.50, 6.00]
def Josh_order := [8.50, 4.00, 3.50]
def Yvette_order := [9.00, 4.50, 6.00]

def discount_rate := 0.10
def sales_tax_rate := 0.08
def tip_rate := 0.20

noncomputable def calculate_final_bill : Float :=
  let subtotal := (Alicia_order.sum + Brant_order.sum + Josh_order.sum + Yvette_order.sum)
  let discount := discount_rate * subtotal
  let discounted_total := subtotal - discount
  let sales_tax := sales_tax_rate * discounted_total
  let pre_tax_and_discount_total := subtotal
  let tip := tip_rate * pre_tax_and_discount_total
  discounted_total + sales_tax + tip

theorem final_bill_is_correct : calculate_final_bill = 84.97 := by
  sorry

end NUMINAMATH_GPT_final_bill_is_correct_l1804_180473


namespace NUMINAMATH_GPT_original_price_computer_l1804_180421

noncomputable def first_store_price (P : ℝ) : ℝ := 0.94 * P

noncomputable def second_store_price (exchange_rate : ℝ) : ℝ := (920 / 0.95) * exchange_rate

theorem original_price_computer 
  (exchange_rate : ℝ)
  (h : exchange_rate = 1.1) 
  (H : (first_store_price P - second_store_price exchange_rate = 19)) :
  P = 1153.47 :=
by
  sorry

end NUMINAMATH_GPT_original_price_computer_l1804_180421


namespace NUMINAMATH_GPT_max_n_for_factorable_poly_l1804_180410

/-- 
  Let p(x) = 6x^2 + n * x + 48 be a quadratic polynomial.
  We want to find the maximum value of n such that p(x) can be factored into
  the product of two linear factors with integer coefficients.
-/
theorem max_n_for_factorable_poly :
  ∃ (n : ℤ), (∀ (A B : ℤ), 6 * B + A = n → A * B = 48) ∧ n = 289 := 
by
  sorry

end NUMINAMATH_GPT_max_n_for_factorable_poly_l1804_180410


namespace NUMINAMATH_GPT_initial_books_count_l1804_180420

theorem initial_books_count (x : ℕ) (h : x + 10 = 48) : x = 38 := 
by
  sorry

end NUMINAMATH_GPT_initial_books_count_l1804_180420


namespace NUMINAMATH_GPT_gcd_of_8a_plus_3_and_5a_plus_2_l1804_180428

theorem gcd_of_8a_plus_3_and_5a_plus_2 (a : ℕ) : Nat.gcd (8 * a + 3) (5 * a + 2) = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_8a_plus_3_and_5a_plus_2_l1804_180428


namespace NUMINAMATH_GPT_rubber_boat_fall_time_l1804_180477

variable {a b x : ℝ}

theorem rubber_boat_fall_time
  (h1 : 5 - x = (a - b) / (a + b))
  (h2 : 6 - x = b / (a + b)) :
  x = 4 := by
  sorry

end NUMINAMATH_GPT_rubber_boat_fall_time_l1804_180477


namespace NUMINAMATH_GPT_largest_base5_eq_124_l1804_180406

-- Define largest base-5 number with three digits
def largest_base5_three_digits : ℕ := 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_eq_124 : largest_base5_three_digits = 124 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_largest_base5_eq_124_l1804_180406


namespace NUMINAMATH_GPT_white_balls_count_l1804_180485

theorem white_balls_count
  (total_balls : ℕ)
  (white_balls blue_balls red_balls : ℕ)
  (h1 : total_balls = 100)
  (h2 : white_balls + blue_balls + red_balls = total_balls)
  (h3 : blue_balls = white_balls + 12)
  (h4 : red_balls = 2 * blue_balls) : white_balls = 16 := by
  sorry

end NUMINAMATH_GPT_white_balls_count_l1804_180485


namespace NUMINAMATH_GPT_dave_shirts_not_washed_l1804_180467

variable (short_sleeve_shirts long_sleeve_shirts washed_shirts : ℕ)

theorem dave_shirts_not_washed (h1 : short_sleeve_shirts = 9) (h2 : long_sleeve_shirts = 27) (h3 : washed_shirts = 20) :
  (short_sleeve_shirts + long_sleeve_shirts - washed_shirts = 16) :=
by {
  -- sorry indicates the proof is omitted
  sorry
}

end NUMINAMATH_GPT_dave_shirts_not_washed_l1804_180467


namespace NUMINAMATH_GPT_solve_expression_l1804_180429

def f (x : ℝ) : ℝ := 2 * x - 1
def g (x : ℝ) : ℝ := x^2 + 2*x + 1

theorem solve_expression : f (g 3) - g (f 3) = -5 := by
  sorry

end NUMINAMATH_GPT_solve_expression_l1804_180429


namespace NUMINAMATH_GPT_diff_implies_continuous_l1804_180496

def differentiable_imp_continuous (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  DifferentiableAt ℝ f x₀ → ContinuousAt f x₀

-- Problem statement: if f is differentiable at x₀, then it is continuous at x₀.
theorem diff_implies_continuous (f : ℝ → ℝ) (x₀ : ℝ) : differentiable_imp_continuous f x₀ :=
by
  sorry

end NUMINAMATH_GPT_diff_implies_continuous_l1804_180496


namespace NUMINAMATH_GPT_find_multiplier_l1804_180431

theorem find_multiplier (x : ℝ) : 3 - 3 * x < 14 ↔ x = -3 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_multiplier_l1804_180431


namespace NUMINAMATH_GPT_students_interested_both_l1804_180455

/-- total students surveyed -/
def U : ℕ := 50

/-- students who liked watching table tennis matches -/
def A : ℕ := 35

/-- students who liked watching badminton matches -/
def B : ℕ := 30

/-- students not interested in either -/
def nU_not_interest : ℕ := 5

theorem students_interested_both : (A + B - (U - nU_not_interest)) = 20 :=
by sorry

end NUMINAMATH_GPT_students_interested_both_l1804_180455


namespace NUMINAMATH_GPT_function_fixed_point_l1804_180412

theorem function_fixed_point {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1) : (2, 2) ∈ { p : ℝ × ℝ | ∃ x, p = (x, a^(x-2) + 1) } :=
by
  sorry

end NUMINAMATH_GPT_function_fixed_point_l1804_180412


namespace NUMINAMATH_GPT_real_root_ineq_l1804_180430

theorem real_root_ineq (a b : ℝ) (x₀ : ℝ) (h : x₀^4 - a * x₀^3 + 2 * x₀^2 - b * x₀ + 1 = 0) :
  a^2 + b^2 ≥ 8 :=
by
  sorry

end NUMINAMATH_GPT_real_root_ineq_l1804_180430


namespace NUMINAMATH_GPT_minimize_distance_school_l1804_180475

-- Define the coordinates for the towns X, Y, and Z
def X_coord : ℕ × ℕ := (0, 0)
def Y_coord : ℕ × ℕ := (200, 0)
def Z_coord : ℕ × ℕ := (0, 300)

-- Define the population of the towns
def X_population : ℕ := 100
def Y_population : ℕ := 200
def Z_population : ℕ := 300

theorem minimize_distance_school : ∃ (x y : ℕ), x + y = 300 := by
  -- This should follow from the problem setup and conditions.
  sorry

end NUMINAMATH_GPT_minimize_distance_school_l1804_180475


namespace NUMINAMATH_GPT_parallelogram_base_length_l1804_180465

theorem parallelogram_base_length :
  ∀ (A H : ℝ), (A = 480) → (H = 15) → (A = Base * H) → (Base = 32) := 
by 
  intros A H hA hH hArea 
  sorry

end NUMINAMATH_GPT_parallelogram_base_length_l1804_180465


namespace NUMINAMATH_GPT_gift_sequence_count_l1804_180442

noncomputable def number_of_gift_sequences (students : ℕ) (classes_per_week : ℕ) : ℕ :=
  (students * students) ^ classes_per_week

theorem gift_sequence_count :
  number_of_gift_sequences 15 3 = 11390625 :=
by
  sorry

end NUMINAMATH_GPT_gift_sequence_count_l1804_180442


namespace NUMINAMATH_GPT_find_m_b_sum_does_not_prove_l1804_180497

theorem find_m_b_sum_does_not_prove :
  ∃ m b : ℝ, 
  let original_point := (2, 3)
  let image_point := (10, 9)
  let midpoint := ((original_point.1 + image_point.1) / 2, (original_point.2 + image_point.2) / 2)
  m = -4 / 3 ∧ 
  midpoint = (6, 6) ∧ 
  6 = m * 6 + b 
  ∧ m + b = 38 / 3 := sorry

end NUMINAMATH_GPT_find_m_b_sum_does_not_prove_l1804_180497


namespace NUMINAMATH_GPT_sam_dimes_proof_l1804_180457

def initial_dimes : ℕ := 9
def remaining_dimes : ℕ := 2
def dimes_given : ℕ := 7

theorem sam_dimes_proof : initial_dimes - remaining_dimes = dimes_given :=
by
  sorry

end NUMINAMATH_GPT_sam_dimes_proof_l1804_180457


namespace NUMINAMATH_GPT_book_pages_l1804_180402

noncomputable def totalPages := 240

theorem book_pages : 
  ∀ P : ℕ, 
    (1 / 2) * P + (1 / 4) * P + (1 / 6) * P + 20 = P → 
    P = totalPages :=
by
  intro P
  intros h
  sorry

end NUMINAMATH_GPT_book_pages_l1804_180402


namespace NUMINAMATH_GPT_sum_of_digits_6608_condition_l1804_180437

theorem sum_of_digits_6608_condition :
  ∀ n1 n2 : ℕ, (6 * 1000 + n1 * 100 + n2 * 10 + 8) % 236 = 0 → n1 + n2 = 6 :=
by 
  intros n1 n2 h
  -- This is where the proof would go. Since we're not proving it, we skip it with "sorry".
  sorry

end NUMINAMATH_GPT_sum_of_digits_6608_condition_l1804_180437


namespace NUMINAMATH_GPT_find_fraction_l1804_180499

theorem find_fraction (x y : ℤ) (h1 : x + 2 = y + 1) (h2 : 2 * (x + 4) = y + 2) : 
  x = -5 ∧ y = -4 := 
sorry

end NUMINAMATH_GPT_find_fraction_l1804_180499


namespace NUMINAMATH_GPT_dave_time_correct_l1804_180494

-- Definitions for the given conditions
def chuck_time (dave_time : ℕ) := 5 * dave_time
def erica_time (chuck_time : ℕ) := chuck_time + (3 * chuck_time / 10)
def erica_fixed_time := 65

-- Statement to prove
theorem dave_time_correct : ∃ (dave_time : ℕ), erica_time (chuck_time dave_time) = erica_fixed_time ∧ dave_time = 10 := by
  sorry

end NUMINAMATH_GPT_dave_time_correct_l1804_180494


namespace NUMINAMATH_GPT_least_possible_value_l1804_180491

theorem least_possible_value (x y z : ℕ) (hx : 2 * x = 5 * y) (hy : 5 * y = 8 * z) (hz : 8 * z = 2 * x) (hnz_x: x > 0) (hnz_y: y > 0) (hnz_z: z > 0) :
  x + y + z = 33 :=
sorry

end NUMINAMATH_GPT_least_possible_value_l1804_180491


namespace NUMINAMATH_GPT_greatest_value_of_x_for_7x_factorial_100_l1804_180486

open Nat

theorem greatest_value_of_x_for_7x_factorial_100 : 
  ∃ x : ℕ, (∀ y : ℕ, 7^y ∣ factorial 100 → y ≤ x) ∧ x = 16 :=
by
  sorry

end NUMINAMATH_GPT_greatest_value_of_x_for_7x_factorial_100_l1804_180486


namespace NUMINAMATH_GPT_power_function_decreasing_l1804_180469

theorem power_function_decreasing (m : ℝ) (f : ℝ → ℝ)
  (h : ∀ x : ℝ, 0 < x → f x = (m^2 + m - 11) * x^(m - 1))
  (hm : m^2 + m - 11 > 0)
  (hm' : m - 1 < 0)
  (hx : 0 < 1):
  f (-1) = -1 := by 
sorry

end NUMINAMATH_GPT_power_function_decreasing_l1804_180469


namespace NUMINAMATH_GPT_greatest_possible_third_term_l1804_180460

theorem greatest_possible_third_term :
  ∃ (a d : ℕ), (a > 0) ∧ (d > 0) ∧ (4 * a + 6 * d = 50) ∧ (∀ (a' d' : ℕ), (a' > 0) ∧ (d' > 0) ∧ (4 * a' + 6 * d' = 50) → (a + 2 * d ≥ a' + 2 * d')) ∧ (a + 2 * d = 16) :=
sorry

end NUMINAMATH_GPT_greatest_possible_third_term_l1804_180460


namespace NUMINAMATH_GPT_trigonometric_identity_l1804_180426

theorem trigonometric_identity (α : ℝ) :
    (1 / Real.sin (-α) - Real.sin (Real.pi + α)) /
    (1 / Real.cos (3 * Real.pi - α) + Real.cos (2 * Real.pi - α)) =
    1 / Real.tan α ^ 3 :=
    sorry

end NUMINAMATH_GPT_trigonometric_identity_l1804_180426


namespace NUMINAMATH_GPT_marsha_pay_per_mile_l1804_180408

variable (distance1 distance2 payment : ℝ)
variable (distance3 : ℝ := distance2 / 2)
variable (totalDistance := distance1 + distance2 + distance3)

noncomputable def payPerMile (payment : ℝ) (totalDistance : ℝ) : ℝ :=
  payment / totalDistance

theorem marsha_pay_per_mile
  (distance1: ℝ := 10)
  (distance2: ℝ := 28)
  (payment: ℝ := 104)
  (distance3: ℝ := distance2 / 2)
  (totalDistance: ℝ := distance1 + distance2 + distance3)
  : payPerMile payment totalDistance = 2 := by
  sorry

end NUMINAMATH_GPT_marsha_pay_per_mile_l1804_180408


namespace NUMINAMATH_GPT_number_of_cars_washed_l1804_180409

theorem number_of_cars_washed (cars trucks suvs total raised_per_car raised_per_truck raised_per_suv : ℕ)
  (hc : cars = 5)
  (ht : trucks = 5)
  (ha : cars + trucks + suvs = total)
  (h_cost_car : raised_per_car = 5)
  (h_cost_truck : raised_per_truck = 6)
  (h_cost_suv : raised_per_suv = 7)
  (h_amount_total : total = 100)
  (h_raised_trucks : trucks * raised_per_truck = 30)
  (h_raised_suvs : suvs * raised_per_suv = 35) :
  suvs + trucks + cars = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cars_washed_l1804_180409


namespace NUMINAMATH_GPT_number_of_positions_forming_cube_with_missing_face_l1804_180462

-- Define the polygon formed by 6 congruent squares in a cross shape
inductive Square
| center : Square
| top : Square
| bottom : Square
| left : Square
| right : Square

-- Define the indices for the additional square positions
inductive Position
| pos1 : Position
| pos2 : Position
| pos3 : Position
| pos4 : Position
| pos5 : Position
| pos6 : Position
| pos7 : Position
| pos8 : Position
| pos9 : Position
| pos10 : Position
| pos11 : Position

-- Define a function that takes a position and returns whether the polygon can form the missing-face cube
def can_form_cube_missing_face : Position → Bool
  | Position.pos1   => true
  | Position.pos2   => true
  | Position.pos3   => true
  | Position.pos4   => true
  | Position.pos5   => false
  | Position.pos6   => false
  | Position.pos7   => false
  | Position.pos8   => false
  | Position.pos9   => true
  | Position.pos10  => true
  | Position.pos11  => true

-- Count valid positions for forming the cube with one face missing
def count_valid_positions : Nat :=
  List.length (List.filter can_form_cube_missing_face 
    [Position.pos1, Position.pos2, Position.pos3, Position.pos4, Position.pos5, Position.pos6, Position.pos7, Position.pos8, Position.pos9, Position.pos10, Position.pos11])

-- Prove that the number of valid positions is 7
theorem number_of_positions_forming_cube_with_missing_face : count_valid_positions = 7 :=
  by
    -- Implementation of the proof
    sorry

end NUMINAMATH_GPT_number_of_positions_forming_cube_with_missing_face_l1804_180462


namespace NUMINAMATH_GPT_total_books_l1804_180423

-- Lean 4 Statement
theorem total_books (stu_books : ℝ) (albert_ratio : ℝ) (albert_books : ℝ) (total_books : ℝ) 
  (h1 : stu_books = 9) 
  (h2 : albert_ratio = 4.5) 
  (h3 : albert_books = stu_books * albert_ratio) 
  (h4 : total_books = stu_books + albert_books) : 
  total_books = 49.5 := 
sorry

end NUMINAMATH_GPT_total_books_l1804_180423


namespace NUMINAMATH_GPT_arithmetic_mean_calculation_l1804_180407

theorem arithmetic_mean_calculation (x : ℝ) 
  (h : (x + 10 + 20 + 3 * x + 15 + 3 * x + 6) / 5 = 30) : 
  x = 14.142857 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_calculation_l1804_180407


namespace NUMINAMATH_GPT_linda_savings_l1804_180441

theorem linda_savings :
  let original_price_per_notebook := 3.75
  let discount_rate := 0.15
  let quantity := 12
  let total_price_without_discount := quantity * original_price_per_notebook
  let discount_amount_per_notebook := original_price_per_notebook * discount_rate
  let discounted_price_per_notebook := original_price_per_notebook - discount_amount_per_notebook
  let total_price_with_discount := quantity * discounted_price_per_notebook
  let total_savings := total_price_without_discount - total_price_with_discount
  total_savings = 6.75 :=
by {
  sorry
}

end NUMINAMATH_GPT_linda_savings_l1804_180441


namespace NUMINAMATH_GPT_number_of_cans_on_third_day_l1804_180490

-- Definition of an arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + d * (n - 1)

theorem number_of_cans_on_third_day :
  (arithmetic_sequence 4 5 2 = 9) →   -- on the second day, he found 9 cans
  (arithmetic_sequence 4 5 7 = 34) →  -- on the seventh day, he found 34 cans
  (arithmetic_sequence 4 5 3 = 14) :=  -- therefore, on the third day, he found 14 cans
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_number_of_cans_on_third_day_l1804_180490


namespace NUMINAMATH_GPT_problem1_problem2_l1804_180452

-- Problem 1
theorem problem1 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hneq : a ≠ b) : 
  (a / Real.sqrt b) + (b / Real.sqrt a) > Real.sqrt a + Real.sqrt b :=
sorry

-- Problem 2
theorem problem2 (x : ℝ) (hx : x > -1) (m : ℕ) (hm : 0 < m) : 
  (1 + x)^m ≥ 1 + m * x :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1804_180452


namespace NUMINAMATH_GPT_earth_surface_area_scientific_notation_l1804_180425

theorem earth_surface_area_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 780000000 = a * 10^n ∧ a = 7.8 ∧ n = 8 :=
by
  sorry

end NUMINAMATH_GPT_earth_surface_area_scientific_notation_l1804_180425


namespace NUMINAMATH_GPT_sponge_cake_eggs_l1804_180463

theorem sponge_cake_eggs (eggs flour sugar total desiredCakeMass : ℕ) 
  (h_recipe : eggs = 300) 
  (h_flour : flour = 120)
  (h_sugar : sugar = 100) 
  (h_total : total = 520) 
  (h_desiredMass : desiredCakeMass = 2600) :
  (eggs * desiredCakeMass / total) = 1500 := by
  sorry

end NUMINAMATH_GPT_sponge_cake_eggs_l1804_180463


namespace NUMINAMATH_GPT_sqrt_16_eq_plus_minus_4_l1804_180481

theorem sqrt_16_eq_plus_minus_4 : ∀ x : ℝ, (x^2 = 16) ↔ (x = 4 ∨ x = -4) :=
by sorry

end NUMINAMATH_GPT_sqrt_16_eq_plus_minus_4_l1804_180481


namespace NUMINAMATH_GPT_rational_inequalities_l1804_180416

theorem rational_inequalities (a b c d : ℚ)
  (h : a^3 - 2005 = b^3 + 2027 ∧ b^3 + 2027 = c^3 - 2822 ∧ c^3 - 2822 = d^3 + 2820) :
  c > a ∧ a > b ∧ b > d :=
by
  sorry

end NUMINAMATH_GPT_rational_inequalities_l1804_180416


namespace NUMINAMATH_GPT_treaty_signed_on_wednesday_l1804_180432

-- This function calculates the weekday after a given number of days since a known weekday.
def weekday_after (start_day: ℕ) (days: ℕ) : ℕ :=
  (start_day + days) % 7

-- Given the problem conditions:
-- The war started on a Friday: 5th day of the week (considering Sunday as 0)
def war_start_day_of_week : ℕ := 5

-- The number of days after which the treaty was signed
def days_until_treaty : ℕ := 926

-- Expected final day (Wednesday): 3rd day of the week (considering Sunday as 0)
def treaty_day_of_week : ℕ := 3

-- The theorem to be proved:
theorem treaty_signed_on_wednesday :
  weekday_after war_start_day_of_week days_until_treaty = treaty_day_of_week :=
by
  sorry

end NUMINAMATH_GPT_treaty_signed_on_wednesday_l1804_180432


namespace NUMINAMATH_GPT_andrea_rhinestones_ratio_l1804_180411

theorem andrea_rhinestones_ratio :
  (∃ (B : ℕ), B = 45 - (1 / 5 * 45) - 21) →
  (1/5 * 45 : ℕ) + B + 21 = 45 →
  (B : ℕ) / 45 = 1 / 3 := 
sorry

end NUMINAMATH_GPT_andrea_rhinestones_ratio_l1804_180411


namespace NUMINAMATH_GPT_smallest_distance_zero_l1804_180492

theorem smallest_distance_zero :
  let r_track (t : ℝ) := (Real.cos t, Real.sin t)
  let i_track (t : ℝ) := (Real.cos (t / 2), Real.sin (t / 2))
  ∀ t₁ t₂ : ℝ, dist (r_track t₁) (i_track t₂) = 0 := by
  sorry

end NUMINAMATH_GPT_smallest_distance_zero_l1804_180492


namespace NUMINAMATH_GPT_shiela_drawings_l1804_180433

theorem shiela_drawings (neighbors : ℕ) (drawings_per_neighbor : ℕ) (total_drawings : ℕ) 
  (h1 : neighbors = 6) (h2 : drawings_per_neighbor = 9) : total_drawings = 54 :=
  by 
    have h : total_drawings = neighbors * drawings_per_neighbor := sorry
    rw [h1, h2] at h
    exact h
    -- Proof skipped with sorry.

end NUMINAMATH_GPT_shiela_drawings_l1804_180433


namespace NUMINAMATH_GPT_final_solution_concentration_l1804_180447

def concentration (mass : ℕ) (volume : ℕ) : ℕ := 
  (mass * 100) / volume

theorem final_solution_concentration :
  let volume1 := 4
  let conc1 := 4 -- percentage
  let volume2 := 2
  let conc2 := 10 -- percentage
  let mass1 := volume1 * conc1 / 100
  let mass2 := volume2 * conc2 / 100
  let total_mass := mass1 + mass2
  let total_volume := volume1 + volume2
  concentration total_mass total_volume = 6 :=
by
  sorry

end NUMINAMATH_GPT_final_solution_concentration_l1804_180447


namespace NUMINAMATH_GPT_min_sum_of_dimensions_l1804_180446

/-- A theorem to find the minimum possible sum of the three dimensions of a rectangular box 
with given volume 1729 inch³ and positive integer dimensions. -/
theorem min_sum_of_dimensions (x y z : ℕ) (h1 : x * y * z = 1729) : x + y + z ≥ 39 :=
by
  sorry

end NUMINAMATH_GPT_min_sum_of_dimensions_l1804_180446


namespace NUMINAMATH_GPT_min_xy_l1804_180495

theorem min_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + 8 * y - x * y = 0) : xy ≥ 64 :=
by sorry

end NUMINAMATH_GPT_min_xy_l1804_180495


namespace NUMINAMATH_GPT_Tom_initial_investment_l1804_180400

noncomputable def Jose_investment : ℝ := 45000
noncomputable def Jose_investment_time : ℕ := 10
noncomputable def total_profit : ℝ := 36000
noncomputable def Jose_share : ℝ := 20000
noncomputable def Tom_share : ℝ := total_profit - Jose_share
noncomputable def Tom_investment_time : ℕ := 12
noncomputable def proportion_Tom : ℝ := (4 : ℝ) / 5
noncomputable def Tom_expected_investment : ℝ := 6000

theorem Tom_initial_investment (T : ℝ) (h1 : Jose_investment = 45000)
                               (h2 : Jose_investment_time = 10)
                               (h3 : total_profit = 36000)
                               (h4 : Jose_share = 20000)
                               (h5 : Tom_investment_time = 12)
                               (h6 : Tom_share = 16000)
                               (h7 : proportion_Tom = (4 : ℝ) / 5)
                               : T = Tom_expected_investment :=
by
  sorry

end NUMINAMATH_GPT_Tom_initial_investment_l1804_180400


namespace NUMINAMATH_GPT_lemon_heads_distribution_l1804_180458

-- Conditions
def total_lemon_heads := 72
def number_of_friends := 6

-- Desired answer
def lemon_heads_per_friend := 12

-- Lean 4 statement
theorem lemon_heads_distribution : total_lemon_heads / number_of_friends = lemon_heads_per_friend := by 
  sorry

end NUMINAMATH_GPT_lemon_heads_distribution_l1804_180458


namespace NUMINAMATH_GPT_find_a_l1804_180464

noncomputable def geometric_sum_expression (n : ℕ) (a : ℝ) : ℝ :=
  3 * 2^n + a

theorem find_a (a : ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = geometric_sum_expression n a) → a = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1804_180464


namespace NUMINAMATH_GPT_find_integer_for_combination_of_square_l1804_180479

theorem find_integer_for_combination_of_square (y : ℝ) :
  ∃ (k : ℝ), (y^2 + 14*y + 60) = (y + 7)^2 + k ∧ k = 11 :=
by
  use 11
  sorry

end NUMINAMATH_GPT_find_integer_for_combination_of_square_l1804_180479


namespace NUMINAMATH_GPT_probability_common_letters_l1804_180414

open Set

def letters_GEOMETRY : Finset Char := {'G', 'E', 'O', 'M', 'T', 'R', 'Y'}
def letters_RHYME : Finset Char := {'R', 'H', 'Y', 'M', 'E'}

def common_letters : Finset Char := letters_GEOMETRY ∩ letters_RHYME

theorem probability_common_letters :
  (common_letters.card : ℚ) / (letters_GEOMETRY.card : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_probability_common_letters_l1804_180414


namespace NUMINAMATH_GPT_union_of_sets_l1804_180484

def A : Set ℝ := {x | 3 < x ∧ x ≤ 7}
def B : Set ℝ := {x | 4 < x ∧ x ≤ 10}

theorem union_of_sets :
  A ∪ B = {x | 3 < x ∧ x ≤ 10} :=
by
  sorry

end NUMINAMATH_GPT_union_of_sets_l1804_180484


namespace NUMINAMATH_GPT_table_height_is_five_l1804_180449

def height_of_table (l h w : ℕ) : Prop :=
  l + h + w = 45 ∧ 2 * w + h = 40

theorem table_height_is_five (l w : ℕ) : height_of_table l 5 w :=
by
  sorry

end NUMINAMATH_GPT_table_height_is_five_l1804_180449


namespace NUMINAMATH_GPT_how_many_large_glasses_l1804_180468

theorem how_many_large_glasses (cost_small cost_large : ℕ) 
                               (total_money money_left change : ℕ) 
                               (num_small : ℕ) : 
  cost_small = 3 -> 
  cost_large = 5 -> 
  total_money = 50 -> 
  money_left = 26 ->
  change = 1 ->
  num_small = 8 ->
  (money_left - change) / cost_large = 5 := 
by 
  intros h1 h2 h3 h4 h5 h6 
  sorry

end NUMINAMATH_GPT_how_many_large_glasses_l1804_180468


namespace NUMINAMATH_GPT_solve_x_floor_x_eq_72_l1804_180471

theorem solve_x_floor_x_eq_72 : ∃ x : ℝ, 0 < x ∧ x * (⌊x⌋) = 72 ∧ x = 9 :=
by
  sorry

end NUMINAMATH_GPT_solve_x_floor_x_eq_72_l1804_180471


namespace NUMINAMATH_GPT_complex_number_identity_l1804_180487

theorem complex_number_identity (m : ℝ) (h : m + ((m ^ 2 - 4) * Complex.I) = Complex.re 0 + 1 * Complex.I ↔ m > 0): 
  (Complex.mk m 2 * Complex.mk 2 (-2)⁻¹) = Complex.I := sorry

end NUMINAMATH_GPT_complex_number_identity_l1804_180487


namespace NUMINAMATH_GPT_round_trip_ticket_percentage_l1804_180434

theorem round_trip_ticket_percentage (p : ℕ → Prop) : 
  (∀ n, p n → n = 375) → (∀ n, p n → n = 375) :=
by
  sorry

end NUMINAMATH_GPT_round_trip_ticket_percentage_l1804_180434


namespace NUMINAMATH_GPT_find_a_l1804_180415

theorem find_a (a : ℝ) (x : ℝ) :
  (∃ b : ℝ, (9 * x^2 - 18 * x + a) = (3 * x + b) ^ 2) → a = 9 := by
  sorry

end NUMINAMATH_GPT_find_a_l1804_180415


namespace NUMINAMATH_GPT_find_number_l1804_180472

-- Definitions and conditions
def unknown_number (x : ℝ) : Prop :=
  (14 / 100) * x = 98

-- Theorem to prove
theorem find_number (x : ℝ) : unknown_number x → x = 700 := by
  sorry

end NUMINAMATH_GPT_find_number_l1804_180472


namespace NUMINAMATH_GPT_solution_set_l1804_180489

def f (x : ℝ) : ℝ := abs x - x + 1

theorem solution_set (x : ℝ) : f (1 - x^2) > f (1 - 2 * x) ↔ x > 2 ∨ x < -1 := by
  sorry

end NUMINAMATH_GPT_solution_set_l1804_180489


namespace NUMINAMATH_GPT_team_selection_ways_l1804_180417

theorem team_selection_ways :
  let ways (n k : ℕ) := Nat.choose n k
  (ways 6 3) * (ways 6 3) = 400 := 
by
  let ways := Nat.choose
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_team_selection_ways_l1804_180417


namespace NUMINAMATH_GPT_square_area_l1804_180450

-- Define the radius of the circles
def circle_radius : ℝ := 3

-- Define the side length of the square based on the arrangement of circles
def square_side_length : ℝ := 2 * (2 * circle_radius)

-- State the theorem to prove the area of the square
theorem square_area : (square_side_length * square_side_length) = 144 :=
by
  sorry

end NUMINAMATH_GPT_square_area_l1804_180450


namespace NUMINAMATH_GPT_sum_of_possible_values_of_x_l1804_180413

-- Conditions
def radius (x : ℝ) : ℝ := x - 2
def semiMajor (x : ℝ) : ℝ := x - 3
def semiMinor (x : ℝ) : ℝ := x + 4

-- Theorem to be proved
theorem sum_of_possible_values_of_x (x : ℝ) :
  (π * semiMajor x * semiMinor x = 2 * π * (radius x) ^ 2) →
  (x = 5 ∨ x = 4) →
  5 + 4 = 9 :=
by
  intros
  rfl

end NUMINAMATH_GPT_sum_of_possible_values_of_x_l1804_180413


namespace NUMINAMATH_GPT_stella_spent_amount_l1804_180436

-- Definitions
def num_dolls : ℕ := 3
def num_clocks : ℕ := 2
def num_glasses : ℕ := 5

def price_doll : ℕ := 5
def price_clock : ℕ := 15
def price_glass : ℕ := 4

def profit : ℕ := 25

-- Calculation of total revenue from profit
def total_revenue : ℕ := num_dolls * price_doll + num_clocks * price_clock + num_glasses * price_glass

-- Proposition to be proved
theorem stella_spent_amount : total_revenue - profit = 40 :=
by sorry

end NUMINAMATH_GPT_stella_spent_amount_l1804_180436


namespace NUMINAMATH_GPT_isosceles_right_triangle_area_l1804_180459

noncomputable def triangle_area (p : ℝ) : ℝ :=
  (1 / 8) * ((p + p * Real.sqrt 2 + 2) * (2 - Real.sqrt 2)) ^ 2

theorem isosceles_right_triangle_area (p : ℝ) :
  let perimeter := p + p * Real.sqrt 2 + 2
  let x := (p + p * Real.sqrt 2 + 2) * (2 - Real.sqrt 2) / 2
  let area := 1 / 2 * x ^ 2
  area = triangle_area p :=
by
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_area_l1804_180459


namespace NUMINAMATH_GPT_dividend_is_correct_l1804_180419

def quotient : ℕ := 36
def divisor : ℕ := 85
def remainder : ℕ := 26

theorem dividend_is_correct : divisor * quotient + remainder = 3086 := by
  sorry

end NUMINAMATH_GPT_dividend_is_correct_l1804_180419


namespace NUMINAMATH_GPT_complete_the_square_l1804_180480

-- Define the quadratic expression as a function.
def quad_expr (k : ℚ) : ℚ := 8 * k^2 + 12 * k + 18

-- Define the completed square form.
def completed_square_expr (k : ℚ) : ℚ := 8 * (k + 3 / 4)^2 + 27 / 2

-- Theorem stating the equality of the original expression in completed square form and the value of r + s.
theorem complete_the_square : ∀ k : ℚ, quad_expr k = completed_square_expr k ∧ (3 / 4 + 27 / 2 = 57 / 4) :=
by
  intro k
  sorry

end NUMINAMATH_GPT_complete_the_square_l1804_180480


namespace NUMINAMATH_GPT_speed_of_man_l1804_180427

/-
  Problem Statement:
  A train 100 meters long takes 6 seconds to cross a man walking at a certain speed in the direction opposite to that of the train. The speed of the train is 54.99520038396929 kmph. What is the speed of the man in kmph?
-/
 
theorem speed_of_man :
  ∀ (length_of_train : ℝ) (time_to_cross : ℝ) (speed_of_train_kmph : ℝ) (relative_speed_mps : ℝ),
    length_of_train = 100 →
    time_to_cross = 6 →
    speed_of_train_kmph = 54.99520038396929 →
    relative_speed_mps = length_of_train / time_to_cross →
    (relative_speed_mps - (speed_of_train_kmph * (1000 / 3600))) * (3600 / 1000) = 5.00479961403071 :=
by
  intros length_of_train time_to_cross speed_of_train_kmph relative_speed_mps
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_speed_of_man_l1804_180427


namespace NUMINAMATH_GPT_green_red_socks_ratio_l1804_180456

theorem green_red_socks_ratio 
  (r : ℕ) -- Number of pairs of red socks originally ordered
  (y : ℕ) -- Price per pair of red socks
  (green_socks_price : ℕ := 3 * y) -- Price per pair of green socks, 3 times the red socks
  (C_original : ℕ := 6 * green_socks_price + r * y) -- Cost of the original order
  (C_interchanged : ℕ := r * green_socks_price + 6 * y) -- Cost of the interchanged order
  (exchange_rate : ℚ := 1.2) -- 20% increase
  (cost_relation : C_interchanged = exchange_rate * C_original) -- Cost relation given by the problem
  : (6 : ℚ) / (r : ℚ) = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_green_red_socks_ratio_l1804_180456


namespace NUMINAMATH_GPT_part1_solution_set_part2_comparison_l1804_180488

noncomputable def f (x : ℝ) := -|x| - |x + 2|

theorem part1_solution_set (x : ℝ) : f x < -4 ↔ x < -3 ∨ x > 1 :=
by sorry

theorem part2_comparison (a b x : ℝ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + b = Real.sqrt 5) : 
  a^2 + b^2 / 4 ≥ f x + 3 :=
by sorry

end NUMINAMATH_GPT_part1_solution_set_part2_comparison_l1804_180488


namespace NUMINAMATH_GPT_least_number_subtracted_l1804_180483

theorem least_number_subtracted (n k : ℕ) (h₁ : n = 123457) (h₂ : k = 79) : ∃ r, n % k = r ∧ r = 33 :=
by
  sorry

end NUMINAMATH_GPT_least_number_subtracted_l1804_180483


namespace NUMINAMATH_GPT_inequality_semi_perimeter_l1804_180448

variables {R r p : Real}

theorem inequality_semi_perimeter (h1 : 0 < R) (h2 : 0 < r) (h3 : 0 < p) :
  16 * R * r - 5 * r^2 ≤ p^2 ∧ p^2 ≤ 4 * R^2 + 4 * R * r + 3 * r^2 :=
sorry

end NUMINAMATH_GPT_inequality_semi_perimeter_l1804_180448


namespace NUMINAMATH_GPT_proof_2_fx_minus_11_eq_f_x_minus_d_l1804_180422

def f (x : ℝ) : ℝ := 2 * x - 3
def d : ℝ := 2

theorem proof_2_fx_minus_11_eq_f_x_minus_d :
  2 * (f 5) - 11 = f (5 - d) := by
  sorry

end NUMINAMATH_GPT_proof_2_fx_minus_11_eq_f_x_minus_d_l1804_180422


namespace NUMINAMATH_GPT_percentage_solution_P_mixture_l1804_180498

-- Define constants for volumes and percentages
variables (P Q : ℝ)

-- Define given conditions
def percentage_lemonade_P : ℝ := 0.2
def percentage_carbonated_P : ℝ := 0.8
def percentage_lemonade_Q : ℝ := 0.45
def percentage_carbonated_Q : ℝ := 0.55
def percentage_carbonated_mixture : ℝ := 0.72

-- Prove that the percentage of the volume of the mixture that is Solution P is 68%
theorem percentage_solution_P_mixture : 
  (percentage_carbonated_P * P + percentage_carbonated_Q * Q = percentage_carbonated_mixture * (P + Q)) → 
  ((P / (P + Q)) * 100 = 68) :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_percentage_solution_P_mixture_l1804_180498


namespace NUMINAMATH_GPT_damaged_cartons_per_customer_l1804_180418

theorem damaged_cartons_per_customer (total_cartons : ℕ) (num_customers : ℕ) (total_accepted : ℕ) 
    (h1 : total_cartons = 400) (h2 : num_customers = 4) (h3 : total_accepted = 160) 
    : (total_cartons - total_accepted) / num_customers = 60 :=
by
  sorry

end NUMINAMATH_GPT_damaged_cartons_per_customer_l1804_180418


namespace NUMINAMATH_GPT_number_of_students_and_average_output_l1804_180405

theorem number_of_students_and_average_output 
  (total_potatoes : ℕ)
  (days : ℕ)
  (x y : ℕ) 
  (h1 : total_potatoes = 45715) 
  (h2 : days = 5)
  (h3 : x * y * days = total_potatoes) : 
  x = 41 ∧ y = 223 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_and_average_output_l1804_180405


namespace NUMINAMATH_GPT_pairs_satisfying_condition_l1804_180470

theorem pairs_satisfying_condition (x y : ℤ) (h : x + y ≠ 0) :
  (x^2 + y^2)/(x + y) = 10 ↔ (x, y) = (12, 6) ∨ (x, y) = (-2, 6) ∨ (x, y) = (12, 4) ∨ (x, y) = (-2, 4) ∨ (x, y) = (10, 10) ∨ (x, y) = (0, 10) ∨ (x, y) = (10, 0) :=
sorry

end NUMINAMATH_GPT_pairs_satisfying_condition_l1804_180470


namespace NUMINAMATH_GPT_f_l1804_180439

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^4 + b * x^2 - x

-- Define the derivative f'(x)
def f' (a b x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x - 1

-- Problem statement: Prove that f'(-1) = -5 given the conditions
theorem f'_neg_one_value (a b : ℝ) (h : f' a b 1 = 3) : f' a b (-1) = -5 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_f_l1804_180439


namespace NUMINAMATH_GPT_sector_angle_l1804_180424

-- Defining the conditions
def perimeter (r l : ℝ) : Prop := 2 * r + l = 8
def area (r l : ℝ) : Prop := (1 / 2) * l * r = 4

-- Lean theorem statement
theorem sector_angle (r l θ : ℝ) :
  (perimeter r l) → (area r l) → (θ = l / r) → |θ| = 2 :=
by sorry

end NUMINAMATH_GPT_sector_angle_l1804_180424


namespace NUMINAMATH_GPT_find_m_l1804_180474

-- Define the vector
def vec2 := (ℝ × ℝ)

-- Given vectors
def a : vec2 := (2, -1)
def c : vec2 := (-1, 2)

-- Definition of parallel vectors
def parallel (v1 v2 : vec2) := ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

-- Problem Statement
theorem find_m (m : ℝ) (b : vec2 := (-1, m)) (h : parallel (a.1 + b.1, a.2 + b.2) c) : m = -1 :=
sorry

end NUMINAMATH_GPT_find_m_l1804_180474


namespace NUMINAMATH_GPT_floor_trig_sum_l1804_180461

theorem floor_trig_sum :
  Int.floor (Real.sin 1) + Int.floor (Real.cos 2) + Int.floor (Real.tan 3) +
  Int.floor (Real.sin 4) + Int.floor (Real.cos 5) + Int.floor (Real.tan 6) = -4 := by
  sorry

end NUMINAMATH_GPT_floor_trig_sum_l1804_180461


namespace NUMINAMATH_GPT_stereographic_projection_reflection_l1804_180438

noncomputable def sphere : Type := sorry
noncomputable def point_on_sphere (P : sphere) : Prop := sorry
noncomputable def reflection_on_sphere (P P' : sphere) (e : sphere) : Prop := sorry
noncomputable def arbitrary_point (E : sphere) (P P' : sphere) : Prop := E ≠ P ∧ E ≠ P'
noncomputable def tangent_plane (E : sphere) : Type := sorry
noncomputable def stereographic_projection (E : sphere) (δ : Type) : sphere → sorry := sorry
noncomputable def circle_on_plane (e : sphere) (E : sphere) (δ : Type) : Type := sorry
noncomputable def inversion_in_circle (P P' : sphere) (e_1 : Type) : Prop := sorry

theorem stereographic_projection_reflection (P P' E : sphere) (e : sphere) (δ : Type) (e_1 : Type) :
  point_on_sphere P ∧
  reflection_on_sphere P P' e ∧
  arbitrary_point E P P' ∧
  circle_on_plane e E δ = e_1 →
  inversion_in_circle P P' e_1 :=
sorry

end NUMINAMATH_GPT_stereographic_projection_reflection_l1804_180438


namespace NUMINAMATH_GPT_systematic_sampling_first_number_l1804_180443

theorem systematic_sampling_first_number
    (n : ℕ)  -- total number of products
    (k : ℕ)  -- sample size
    (common_diff : ℕ)  -- common difference in the systematic sample
    (x : ℕ)  -- an element in the sample
    (first_num : ℕ)  -- first product number in the sample
    (h1 : n = 80)  -- total number of products is 80
    (h2 : k = 5)  -- sample size is 5
    (h3 : common_diff = 16)  -- common difference is 16
    (h4 : x = 42)  -- 42 is in the sample
    (h5 : x = common_diff * 2 + first_num)  -- position of 42 in the arithmetic sequence
: first_num = 10 := 
sorry

end NUMINAMATH_GPT_systematic_sampling_first_number_l1804_180443


namespace NUMINAMATH_GPT_solve_inequality_l1804_180440

noncomputable def solutionSet := { x : ℝ | 0 < x ∧ x < 1 }

theorem solve_inequality (x : ℝ) : x^2 < x ↔ x ∈ solutionSet := 
sorry

end NUMINAMATH_GPT_solve_inequality_l1804_180440


namespace NUMINAMATH_GPT_teacher_age_l1804_180453

theorem teacher_age (avg_age_students : ℕ) (num_students : ℕ) (avg_age_total : ℕ) (num_total : ℕ) (h1 : avg_age_students = 21) (h2 : num_students = 20) (h3 : avg_age_total = 22) (h4 : num_total = 21) :
  let total_age_students := avg_age_students * num_students
  let total_age_class := avg_age_total * num_total
  let teacher_age := total_age_class - total_age_students
  teacher_age = 42 :=
by
  sorry

end NUMINAMATH_GPT_teacher_age_l1804_180453


namespace NUMINAMATH_GPT_red_flowers_count_l1804_180445

theorem red_flowers_count (w r : ℕ) (h1 : w = 555) (h2 : w = r + 208) : r = 347 :=
by {
  -- Proof steps will be here
  sorry
}

end NUMINAMATH_GPT_red_flowers_count_l1804_180445


namespace NUMINAMATH_GPT_total_bike_price_l1804_180454

theorem total_bike_price 
  (marion_bike_cost : ℝ := 356)
  (stephanie_bike_base_cost : ℝ := 2 * marion_bike_cost)
  (stephanie_discount_rate : ℝ := 0.10)
  (patrick_bike_base_cost : ℝ := 3 * marion_bike_cost)
  (patrick_discount_rate : ℝ := 0.75)
  (stephanie_bike_cost : ℝ := stephanie_bike_base_cost * (1 - stephanie_discount_rate))
  (patrick_bike_cost : ℝ := patrick_bike_base_cost * patrick_discount_rate):
  marion_bike_cost + stephanie_bike_cost + patrick_bike_cost = 1797.80 := 
by 
  sorry

end NUMINAMATH_GPT_total_bike_price_l1804_180454


namespace NUMINAMATH_GPT_ursula_annual_salary_l1804_180493

def hourly_wage : ℝ := 8.50
def hours_per_day : ℝ := 8
def days_per_month : ℝ := 20
def months_per_year : ℝ := 12

noncomputable def daily_earnings : ℝ := hourly_wage * hours_per_day
noncomputable def monthly_earnings : ℝ := daily_earnings * days_per_month
noncomputable def annual_salary : ℝ := monthly_earnings * months_per_year

theorem ursula_annual_salary : annual_salary = 16320 := 
  by sorry

end NUMINAMATH_GPT_ursula_annual_salary_l1804_180493
