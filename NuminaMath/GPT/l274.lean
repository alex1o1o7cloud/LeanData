import Mathlib

namespace NUMINAMATH_GPT_solve_inequality_l274_27407

theorem solve_inequality : 
  {x : ℝ | -3 * x^2 + 9 * x + 6 < 0} = {x : ℝ | -2 / 3 < x ∧ x < 3} :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_inequality_l274_27407


namespace NUMINAMATH_GPT_arithmetic_sequence_num_terms_l274_27423

theorem arithmetic_sequence_num_terms 
  (a : ℕ) (d : ℕ) (l : ℕ) (n : ℕ)
  (h1 : a = 20)
  (h2 : d = 5)
  (h3 : l = 150)
  (h4 : 150 = 20 + (n-1) * 5) :
  n = 27 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_num_terms_l274_27423


namespace NUMINAMATH_GPT_cube_side_length_increase_20_percent_l274_27414

variable {s : ℝ} (initial_side_length_increase : ℝ) (percentage_surface_area_increase : ℝ) (percentage_volume_increase : ℝ)
variable (new_surface_area : ℝ) (new_volume : ℝ)

theorem cube_side_length_increase_20_percent :
  ∀ (s : ℝ),
  (initial_side_length_increase = 1.2 * s) →
  (new_surface_area = 6 * (1.2 * s)^2) →
  (new_volume = (1.2 * s)^3) →
  (percentage_surface_area_increase = ((new_surface_area - (6 * s^2)) / (6 * s^2)) * 100) →
  (percentage_volume_increase = ((new_volume - s^3) / s^3) * 100) →
  5 * (percentage_volume_increase - percentage_surface_area_increase) = 144 := by
  sorry

end NUMINAMATH_GPT_cube_side_length_increase_20_percent_l274_27414


namespace NUMINAMATH_GPT_total_cost_of_plates_and_cups_l274_27425

theorem total_cost_of_plates_and_cups (P C : ℝ) 
  (h : 20 * P + 40 * C = 1.50) : 
  100 * P + 200 * C = 7.50 :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_total_cost_of_plates_and_cups_l274_27425


namespace NUMINAMATH_GPT_eval_sum_l274_27458

theorem eval_sum : 333 + 33 + 3 = 369 :=
by
  sorry

end NUMINAMATH_GPT_eval_sum_l274_27458


namespace NUMINAMATH_GPT_trapezoid_ratio_l274_27426

structure Trapezoid (α : Type) [LinearOrderedField α] :=
  (AB CD : α)
  (areas : List α)
  (AB_gt_CD : AB > CD)
  (areas_eq : areas = [3, 5, 6, 8])

open Trapezoid

theorem trapezoid_ratio (α : Type) [LinearOrderedField α] (T : Trapezoid α) :
  ∃ ρ : α, T.AB / T.CD = ρ ∧ ρ = 8 / 3 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_ratio_l274_27426


namespace NUMINAMATH_GPT_power_of_i_2016_l274_27481
-- Importing necessary libraries to handle complex numbers

theorem power_of_i_2016 (i : ℂ) (h1 : i^2 = -1) (h2 : i^4 = 1) : 
  (i^2016 = 1) :=
sorry

end NUMINAMATH_GPT_power_of_i_2016_l274_27481


namespace NUMINAMATH_GPT_gross_profit_percentage_is_correct_l274_27401

def selling_price : ℝ := 28
def wholesale_cost : ℝ := 24.56
def gross_profit : ℝ := selling_price - wholesale_cost

-- Define the expected profit percentage as a constant value.
def expected_profit_percentage : ℝ := 14.01

theorem gross_profit_percentage_is_correct :
  ((gross_profit / wholesale_cost) * 100) = expected_profit_percentage :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_gross_profit_percentage_is_correct_l274_27401


namespace NUMINAMATH_GPT_remainder_of_4n_squared_l274_27473

theorem remainder_of_4n_squared {n : ℤ} (h : n % 13 = 7) : (4 * n^2) % 13 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_4n_squared_l274_27473


namespace NUMINAMATH_GPT_find_k_for_line_l274_27446

theorem find_k_for_line : 
  ∃ k : ℚ, (∀ x y : ℚ, (-1 / 3 - 3 * k * x = 4 * y) ∧ (x = 1 / 3) ∧ (y = -8)) → k = 95 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_for_line_l274_27446


namespace NUMINAMATH_GPT_average_shift_l274_27465

variable (a b c : ℝ)

-- Given condition: The average of the data \(a\), \(b\), \(c\) is 5.
def average_is_five := (a + b + c) / 3 = 5

-- Define the statement to prove: The average of the data \(a-2\), \(b-2\), \(c-2\) is 3.
theorem average_shift (h : average_is_five a b c) : ((a - 2) + (b - 2) + (c - 2)) / 3 = 3 :=
by
  sorry

end NUMINAMATH_GPT_average_shift_l274_27465


namespace NUMINAMATH_GPT_eval_to_one_l274_27402

noncomputable def evalExpression (a b c : ℝ) : ℝ :=
  let numerator := (1 / a + 1 / b - 2 * c / (a * b)) * (a + b + 2 * c)
  let denominator := 1 / a^2 + 1 / b^2 + 2 / (a * b) - 4 * c^2 / (a^2 * b^2)
  numerator / denominator

theorem eval_to_one : 
  evalExpression 7.4 (5 / 37) c = 1 := 
by 
  sorry

end NUMINAMATH_GPT_eval_to_one_l274_27402


namespace NUMINAMATH_GPT_fraction_of_money_left_l274_27460

theorem fraction_of_money_left (m : ℝ) (b : ℝ) (h1 : (1 / 4) * m = (1 / 2) * b) :
  m - b - 50 = m / 2 - 50 → (m - b - 50) / m = 1 / 2 - 50 / m :=
by sorry

end NUMINAMATH_GPT_fraction_of_money_left_l274_27460


namespace NUMINAMATH_GPT_num_of_sets_eq_four_l274_27480

open Finset

theorem num_of_sets_eq_four : ∀ B : Finset ℕ, (insert 1 (insert 2 B) = {1, 2, 3, 4, 5}) → (B = {3, 4, 5} ∨ B = {1, 3, 4, 5} ∨ B = {2, 3, 4, 5} ∨ B = {1, 2, 3, 4, 5}) := 
by
  sorry

end NUMINAMATH_GPT_num_of_sets_eq_four_l274_27480


namespace NUMINAMATH_GPT_ratio_of_chocolate_to_regular_milk_l274_27413

def total_cartons : Nat := 24
def regular_milk_cartons : Nat := 3
def chocolate_milk_cartons : Nat := total_cartons - regular_milk_cartons

theorem ratio_of_chocolate_to_regular_milk (h1 : total_cartons = 24) (h2 : regular_milk_cartons = 3) :
  chocolate_milk_cartons / regular_milk_cartons = 7 :=
by 
  -- Skipping proof with sorry
  sorry

end NUMINAMATH_GPT_ratio_of_chocolate_to_regular_milk_l274_27413


namespace NUMINAMATH_GPT_football_team_gain_l274_27450

theorem football_team_gain (G : ℤ) :
  (-5 + G = 2) → (G = 7) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_football_team_gain_l274_27450


namespace NUMINAMATH_GPT_unique_x0_implies_a_in_range_l274_27495

noncomputable def f (x : ℝ) (a : ℝ) := Real.exp x * (3 * x - 1) - a * x + a

theorem unique_x0_implies_a_in_range :
  ∃ x0 : ℤ, f x0 a ≤ 0 ∧ a < 1 -> a ∈ Set.Ico (2 / Real.exp 1) 1 := 
sorry

end NUMINAMATH_GPT_unique_x0_implies_a_in_range_l274_27495


namespace NUMINAMATH_GPT_collinear_points_l274_27471

theorem collinear_points (x y : ℝ) (h_collinear : ∃ k : ℝ, (x + 1, y, 3) = (2 * k, 4 * k, 6 * k)) : x - y = -2 := 
by 
  sorry

end NUMINAMATH_GPT_collinear_points_l274_27471


namespace NUMINAMATH_GPT_alex_correct_percentage_l274_27410

theorem alex_correct_percentage 
  (score_quiz : ℤ) (problems_quiz : ℤ)
  (score_test : ℤ) (problems_test : ℤ)
  (score_exam : ℤ) (problems_exam : ℤ)
  (h1 : score_quiz = 75) (h2 : problems_quiz = 30)
  (h3 : score_test = 85) (h4 : problems_test = 50)
  (h5 : score_exam = 80) (h6 : problems_exam = 20) :
  (75 * 30 + 85 * 50 + 80 * 20) / (30 + 50 + 20) = 81 := 
sorry

end NUMINAMATH_GPT_alex_correct_percentage_l274_27410


namespace NUMINAMATH_GPT_fg_of_neg2_l274_27437

def f (x : ℤ) : ℤ := x^2
def g (x : ℤ) : ℤ := 2 * x + 5

theorem fg_of_neg2 : f (g (-2)) = 1 := by
  sorry

end NUMINAMATH_GPT_fg_of_neg2_l274_27437


namespace NUMINAMATH_GPT_quadratic_floor_eq_solutions_count_l274_27429

theorem quadratic_floor_eq_solutions_count : 
  ∃ s : Finset ℝ, (∀ x : ℝ, x^2 - 4 * ⌊x⌋ + 3 = 0 → x ∈ s) ∧ s.card = 3 :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_floor_eq_solutions_count_l274_27429


namespace NUMINAMATH_GPT_eight_S_three_l274_27456

def custom_operation_S (a b : ℤ) : ℤ := 4 * a + 6 * b + 3

theorem eight_S_three : custom_operation_S 8 3 = 53 := by
  sorry

end NUMINAMATH_GPT_eight_S_three_l274_27456


namespace NUMINAMATH_GPT_xy_diff_l274_27416

theorem xy_diff {x y : ℝ} (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 :=
by
  sorry

end NUMINAMATH_GPT_xy_diff_l274_27416


namespace NUMINAMATH_GPT_distance_from_C_to_A_is_8_l274_27454

-- Define points A, B, and C as real numbers representing positions
def A : ℝ := 0  -- Starting point
def B : ℝ := A - 15  -- 15 meters west from A
def C : ℝ := B + 23  -- 23 meters east from B

-- Prove that the distance from point C to point A is 8 meters
theorem distance_from_C_to_A_is_8 : abs (C - A) = 8 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_C_to_A_is_8_l274_27454


namespace NUMINAMATH_GPT_borrow_years_l274_27455

/-- A person borrows Rs. 5000 at 4% p.a simple interest and lends it at 6% p.a simple interest.
His gain in the transaction per year is Rs. 100. Prove that he borrowed the money for 1 year. --/
theorem borrow_years
  (principal : ℝ)
  (borrow_rate : ℝ)
  (lend_rate : ℝ)
  (gain : ℝ)
  (interest_paid_per_year : ℝ)
  (interest_earned_per_year : ℝ) :
  (principal = 5000) →
  (borrow_rate = 0.04) →
  (lend_rate = 0.06) →
  (gain = 100) →
  (interest_paid_per_year = principal * borrow_rate) →
  (interest_earned_per_year = principal * lend_rate) →
  (interest_earned_per_year - interest_paid_per_year = gain) →
  1 = 1 := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_borrow_years_l274_27455


namespace NUMINAMATH_GPT_incorrect_inequality_l274_27434

variable (a b : ℝ)

theorem incorrect_inequality (h : a > b) : ¬ (-2 * a > -2 * b) :=
by sorry

end NUMINAMATH_GPT_incorrect_inequality_l274_27434


namespace NUMINAMATH_GPT_trigonometric_inequality_l274_27400

open Real

theorem trigonometric_inequality 
  (x y z : ℝ) 
  (h1 : 0 < x) 
  (h2 : x < y) 
  (h3 : y < z) 
  (h4 : z < π / 2) : 
  π / 2 + 2 * sin x * cos y + 2 * sin y * cos z > sin (2 * x) + sin (2 * y) + sin (2 * z) :=
  sorry

end NUMINAMATH_GPT_trigonometric_inequality_l274_27400


namespace NUMINAMATH_GPT_hexagon_circle_ratio_correct_l274_27432

noncomputable def hexagon_circle_area_ratio (s r : ℝ) (h : 6 * s = 2 * π * r) : ℝ :=
  let A_hex := (3 * Real.sqrt 3 / 2) * s^2
  let A_circ := π * r^2
  (A_hex / A_circ)

theorem hexagon_circle_ratio_correct (s r : ℝ) (h : 6 * s = 2 * π * r) :
    hexagon_circle_area_ratio s r h = (π * Real.sqrt 3 / 6) :=
sorry

end NUMINAMATH_GPT_hexagon_circle_ratio_correct_l274_27432


namespace NUMINAMATH_GPT_garden_area_increase_l274_27411

-- Problem: Prove that changing a 40 ft by 10 ft rectangular garden into a square,
-- using the same fencing, increases the area by 225 sq ft.

theorem garden_area_increase :
  let length_orig := 40
  let width_orig := 10
  let perimeter := 2 * (length_orig + width_orig)
  let side_square := perimeter / 4
  let area_orig := length_orig * width_orig
  let area_square := side_square * side_square
  (area_square - area_orig) = 225 := 
sorry

end NUMINAMATH_GPT_garden_area_increase_l274_27411


namespace NUMINAMATH_GPT_total_people_after_one_hour_l274_27403

variable (x y Z : ℕ)

def ferris_wheel_line_initial := 50
def bumper_cars_line_initial := 50
def roller_coaster_line_initial := 50

def ferris_wheel_line_after_half_hour := ferris_wheel_line_initial - x
def bumper_cars_line_after_half_hour := bumper_cars_line_initial + y

axiom Z_eq : Z = ferris_wheel_line_after_half_hour + bumper_cars_line_after_half_hour

theorem total_people_after_one_hour : (Z = (50 - x) + (50 + y)) -> (Z + 100) = ((50 - x) + (50 + y) + 100) :=
by {
  sorry
}

end NUMINAMATH_GPT_total_people_after_one_hour_l274_27403


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l274_27493

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum (h_arith : arithmetic_sequence a)
    (h_a2 : a 2 = 3)
    (h_a1_a6 : a 1 + a 6 = 12) : a 7 + a 8 + a 9 = 45 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l274_27493


namespace NUMINAMATH_GPT_find_common_ratio_l274_27431

variable {α : Type*} [LinearOrderedField α]

def is_geometric_sequence (a : ℕ → α) : Prop :=
∀ n m, ∃ q, a (n + 1) = a n * q ∧ a (m + 1) = a m * q

theorem find_common_ratio 
  (a : ℕ → α) 
  (h : is_geometric_sequence a) 
  (h_a3 : a 3 = 2)
  (h_a6 : a 6 = 1 / 4) : 
  ∃ q, q = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_common_ratio_l274_27431


namespace NUMINAMATH_GPT_little_john_height_l274_27427

theorem little_john_height :
  let m := 2 
  let cm_to_m := 8 * 0.01
  let mm_to_m := 3 * 0.001
  m + cm_to_m + mm_to_m = 2.083 := 
by
  sorry

end NUMINAMATH_GPT_little_john_height_l274_27427


namespace NUMINAMATH_GPT_positive_difference_of_two_numbers_l274_27476

theorem positive_difference_of_two_numbers :
  ∃ (x y : ℝ), x + y = 10 ∧ x^2 - y^2 = 24 ∧ |x - y| = 12 / 5 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_of_two_numbers_l274_27476


namespace NUMINAMATH_GPT_caricatures_sold_on_sunday_l274_27444

def caricature_price : ℕ := 20
def saturday_sales : ℕ := 24
def total_earnings : ℕ := 800

theorem caricatures_sold_on_sunday :
  (total_earnings - saturday_sales * caricature_price) / caricature_price = 16 :=
by
  sorry  -- Proof goes here

end NUMINAMATH_GPT_caricatures_sold_on_sunday_l274_27444


namespace NUMINAMATH_GPT_min_value_xy_l274_27421

theorem min_value_xy (x y : ℝ) (h1 : x + y = -1) (h2 : x < 0) (h3 : y < 0) :
  ∃ (xy_min : ℝ), (∀ (xy : ℝ), xy = x * y → xy + 1 / xy ≥ xy_min) ∧ xy_min = 17 / 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_xy_l274_27421


namespace NUMINAMATH_GPT_find_digit_A_l274_27475

-- Define the six-digit number for any digit A
def six_digit_number (A : ℕ) : ℕ := 103200 + A * 10 + 4
-- Define the condition that a number is prime
def is_prime (n : ℕ) : Prop := (2 ≤ n) ∧ ∀ m : ℕ, 2 ≤ m → m * m ≤ n → ¬ (m ∣ n)

-- The main theorem stating that A must equal 1 for the number to be prime
theorem find_digit_A (A : ℕ) : A = 1 ↔ is_prime (six_digit_number A) :=
by
  sorry -- Proof to be filled in


end NUMINAMATH_GPT_find_digit_A_l274_27475


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l274_27499

theorem hyperbola_eccentricity (k : ℝ) (h_eq : ∀ x y : ℝ, x^2 + k * y^2 = 1) (h_eccentricity : ∀ e : ℝ, e = 2) :
    k = -1 / 3 := 
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l274_27499


namespace NUMINAMATH_GPT_find_a_for_even_function_l274_27472

theorem find_a_for_even_function (a : ℝ) :
  (∀ x : ℝ, x^3 * (a * 2^x - 2^(-x)) = (-x)^3 * (a * 2^(-x) - 2^x)) →
  a = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_for_even_function_l274_27472


namespace NUMINAMATH_GPT_cost_of_orange_juice_l274_27409

theorem cost_of_orange_juice (total_money : ℕ) (bread_qty : ℕ) (orange_qty : ℕ) 
  (bread_cost : ℕ) (money_left : ℕ) (total_spent : ℕ) (orange_cost : ℕ) 
  (h1 : total_money = 86) (h2 : bread_qty = 3) (h3 : orange_qty = 3) 
  (h4 : bread_cost = 3) (h5 : money_left = 59) :
  (total_money - money_left - bread_qty * bread_cost) / orange_qty = 6 :=
by
  have h6 : total_spent = total_money - money_left := by sorry
  have h7 : total_spent - bread_qty * bread_cost = orange_qty * orange_cost := by sorry
  have h8 : orange_cost = 6 := by sorry
  exact sorry

end NUMINAMATH_GPT_cost_of_orange_juice_l274_27409


namespace NUMINAMATH_GPT_smaller_number_l274_27464

theorem smaller_number (x y : ℤ) (h1 : x + y = 12) (h2 : x - y = 20) : y = -4 := 
by 
  sorry

end NUMINAMATH_GPT_smaller_number_l274_27464


namespace NUMINAMATH_GPT_bear_small_animal_weight_l274_27445

theorem bear_small_animal_weight :
  let total_weight_needed := 1200
  let berries_weight := 1/5 * total_weight_needed
  let insects_weight := 1/10 * total_weight_needed
  let acorns_weight := 2 * berries_weight
  let honey_weight := 3 * insects_weight
  let total_weight_gained := berries_weight + insects_weight + acorns_weight + honey_weight
  let remaining_weight := total_weight_needed - total_weight_gained
  remaining_weight = 0 -> 0 = 0 := by
  intros total_weight_needed berries_weight insects_weight acorns_weight honey_weight
         total_weight_gained remaining_weight h
  exact Eq.refl 0

end NUMINAMATH_GPT_bear_small_animal_weight_l274_27445


namespace NUMINAMATH_GPT_barbara_wins_iff_multiple_of_6_l274_27419

-- Define the conditions and the statement to be proved
theorem barbara_wins_iff_multiple_of_6 (n : ℕ) (h : n > 1) :
  (∃ a b : ℕ, a > 0 ∧ b > 1 ∧ (b ∣ a ∨ a ∣ b) ∧ ∀ k ≤ 50, (b + k = n ∨ b - k = n)) ↔ 6 ∣ n :=
sorry

end NUMINAMATH_GPT_barbara_wins_iff_multiple_of_6_l274_27419


namespace NUMINAMATH_GPT_product_signs_l274_27486

theorem product_signs (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) : 
  ( 
    (((-a * b > 0) ∧ (a * c < 0) ∧ (b * d < 0) ∧ (c * d < 0)) ∨ 
    ((-a * b < 0) ∧ (a * c > 0) ∧ (b * d > 0) ∧ (c * d > 0))) ∨
    (((-a * b < 0) ∧ (a * c > 0) ∧ (b * d < 0) ∧ (c * d > 0)) ∨ 
    ((-a * b > 0) ∧ (a * c < 0) ∧ (b * d > 0) ∧ (c * d < 0))) 
  ) := 
sorry

end NUMINAMATH_GPT_product_signs_l274_27486


namespace NUMINAMATH_GPT_find_quotient_from_conditions_l274_27442

variable (x y : ℕ)
variable (k : ℕ)

theorem find_quotient_from_conditions :
  y - x = 1360 ∧ y = 1614 ∧ y % x = 15 → y / x = 6 :=
by
  intro h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end NUMINAMATH_GPT_find_quotient_from_conditions_l274_27442


namespace NUMINAMATH_GPT_smallest_interesting_rectangle_area_l274_27492

/-- 
  A rectangle is interesting if both its side lengths are integers and 
  it contains exactly four lattice points strictly in its interior.
  Prove that the area of the smallest such interesting rectangle is 10.
-/
theorem smallest_interesting_rectangle_area :
  ∃ (a b : ℕ), (a - 1) * (b - 1) = 4 ∧ a * b = 10 :=
by
  sorry

end NUMINAMATH_GPT_smallest_interesting_rectangle_area_l274_27492


namespace NUMINAMATH_GPT_cubic_sum_identity_l274_27484

theorem cubic_sum_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
sorry

end NUMINAMATH_GPT_cubic_sum_identity_l274_27484


namespace NUMINAMATH_GPT_yo_yos_collected_l274_27482

-- Define the given conditions
def stuffed_animals : ℕ := 14
def frisbees : ℕ := 18
def total_prizes : ℕ := 50

-- Define the problem to prove that the number of yo-yos is 18
theorem yo_yos_collected : (total_prizes - (stuffed_animals + frisbees) = 18) :=
by
  sorry

end NUMINAMATH_GPT_yo_yos_collected_l274_27482


namespace NUMINAMATH_GPT_pool_width_l274_27462

variable (length : ℝ) (depth : ℝ) (chlorine_per_cubic_foot : ℝ) (chlorine_cost_per_quart : ℝ) (total_spent : ℝ)
variable (w : ℝ)

-- defining the conditions
def pool_conditions := length = 10 ∧ depth = 6 ∧ chlorine_per_cubic_foot = 120 ∧ chlorine_cost_per_quart = 3 ∧ total_spent = 12

-- goal statement
theorem pool_width : pool_conditions length depth chlorine_per_cubic_foot chlorine_cost_per_quart total_spent →
  w = 8 :=
by
  sorry

end NUMINAMATH_GPT_pool_width_l274_27462


namespace NUMINAMATH_GPT_first_digit_base8_of_473_l274_27433

theorem first_digit_base8_of_473 : 
  ∃ (d : ℕ), (d < 8) ∧ (473 = d * 64 + r ∧ r < 64) ∧ 473 = 7 * 64 + 25 :=
sorry

end NUMINAMATH_GPT_first_digit_base8_of_473_l274_27433


namespace NUMINAMATH_GPT_runs_in_last_match_l274_27461

theorem runs_in_last_match (W : ℕ) (R x : ℝ) 
    (hW : W = 85) 
    (hR : R = 12.4 * W) 
    (new_average : (R + x) / (W + 5) = 12) : 
    x = 26 := 
by 
  sorry

end NUMINAMATH_GPT_runs_in_last_match_l274_27461


namespace NUMINAMATH_GPT_product_of_935421_and_625_l274_27477

theorem product_of_935421_and_625 : 935421 * 625 = 584638125 :=
by
  sorry

end NUMINAMATH_GPT_product_of_935421_and_625_l274_27477


namespace NUMINAMATH_GPT_XAXAXA_divisible_by_seven_l274_27406

theorem XAXAXA_divisible_by_seven (X A : ℕ) (hX : X < 10) (hA : A < 10) : 
  (101010 * X + 10101 * A) % 7 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_XAXAXA_divisible_by_seven_l274_27406


namespace NUMINAMATH_GPT_Ryan_funding_goal_l274_27487

theorem Ryan_funding_goal 
  (avg_fund_per_person : ℕ := 10) 
  (people_recruited : ℕ := 80)
  (pre_existing_fund : ℕ := 200) :
  (avg_fund_per_person * people_recruited + pre_existing_fund = 1000) :=
by
  sorry

end NUMINAMATH_GPT_Ryan_funding_goal_l274_27487


namespace NUMINAMATH_GPT_carson_gardening_time_l274_27485

-- Definitions of the problem conditions
def lines_to_mow : ℕ := 40
def minutes_per_line : ℕ := 2
def rows_of_flowers : ℕ := 8
def flowers_per_row : ℕ := 7
def minutes_per_flower : ℚ := 0.5

-- Total time calculation for the proof 
theorem carson_gardening_time : 
  (lines_to_mow * minutes_per_line) + (rows_of_flowers * flowers_per_row * minutes_per_flower) = 108 := 
by 
  sorry

end NUMINAMATH_GPT_carson_gardening_time_l274_27485


namespace NUMINAMATH_GPT_orange_ratio_l274_27405

variable {R U : ℕ}

theorem orange_ratio (h1 : R + U = 96) 
                    (h2 : (3 / 4 : ℝ) * R + (7 / 8 : ℝ) * U = 78) :
  (R : ℝ) / (R + U : ℝ) = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_orange_ratio_l274_27405


namespace NUMINAMATH_GPT_find_initial_average_price_l274_27453

noncomputable def average_initial_price (P : ℚ) : Prop :=
  let total_cost_of_4_cans := 120
  let total_cost_of_returned_cans := 99
  let total_cost_of_6_cans := 6 * P
  total_cost_of_6_cans - total_cost_of_4_cans = total_cost_of_returned_cans

theorem find_initial_average_price (P : ℚ) :
    average_initial_price P → 
    P = 36.5 := sorry

end NUMINAMATH_GPT_find_initial_average_price_l274_27453


namespace NUMINAMATH_GPT_bobby_has_candy_left_l274_27420

def initial_candy := 36
def candy_eaten_first := 17
def candy_eaten_second := 15

theorem bobby_has_candy_left : 
  initial_candy - (candy_eaten_first + candy_eaten_second) = 4 := 
by
  sorry


end NUMINAMATH_GPT_bobby_has_candy_left_l274_27420


namespace NUMINAMATH_GPT_convert_units_l274_27459

theorem convert_units :
  (0.56 * 10 = 5.6 ∧ 0.6 * 10 = 6) ∧
  (2.05 = 2 + 0.05 ∧ 0.05 * 100 = 5) :=
by 
  sorry

end NUMINAMATH_GPT_convert_units_l274_27459


namespace NUMINAMATH_GPT_dave_diner_total_cost_l274_27490

theorem dave_diner_total_cost (burger_count : ℕ) (fries_count : ℕ)
  (burger_cost : ℕ) (fries_cost : ℕ)
  (discount_threshold : ℕ) (discount_amount : ℕ)
  (h1 : burger_count >= discount_threshold) :
  burger_count = 6 → fries_count = 5 → burger_cost = 4 → fries_cost = 3 →
  discount_threshold = 4 → discount_amount = 2 →
  (burger_count * (burger_cost - discount_amount) + fries_count * fries_cost) = 27 :=
by
  intros hbc hfc hbcost hfcs dth da
  sorry

end NUMINAMATH_GPT_dave_diner_total_cost_l274_27490


namespace NUMINAMATH_GPT_books_per_shelf_l274_27489

theorem books_per_shelf (total_books : ℕ) (total_shelves : ℕ) (h_total_books : total_books = 2250) (h_total_shelves : total_shelves = 150) :
  total_books / total_shelves = 15 :=
by
  sorry

end NUMINAMATH_GPT_books_per_shelf_l274_27489


namespace NUMINAMATH_GPT_earnings_correct_l274_27439

def phonePrice : Nat := 11
def laptopPrice : Nat := 15
def computerPrice : Nat := 18
def tabletPrice : Nat := 12
def smartwatchPrice : Nat := 8

def phoneRepairs : Nat := 9
def laptopRepairs : Nat := 5
def computerRepairs : Nat := 4
def tabletRepairs : Nat := 6
def smartwatchRepairs : Nat := 8

def totalEarnings : Nat := 
  phoneRepairs * phonePrice + 
  laptopRepairs * laptopPrice + 
  computerRepairs * computerPrice + 
  tabletRepairs * tabletPrice + 
  smartwatchRepairs * smartwatchPrice

theorem earnings_correct : totalEarnings = 382 := by
  sorry

end NUMINAMATH_GPT_earnings_correct_l274_27439


namespace NUMINAMATH_GPT_sergeant_distance_travel_l274_27452

noncomputable def sergeant_distance (x k : ℝ) : ℝ :=
  let t₁ := 1 / (x * (k - 1))
  let t₂ := 1 / (x * (k + 1))
  let t := t₁ + t₂
  let d := k * 4 / 3
  d

theorem sergeant_distance_travel (x k : ℝ) (h1 : (4 * k) / (k^2 - 1) = 4 / 3) :
  sergeant_distance x k = 8 / 3 := by
  sorry

end NUMINAMATH_GPT_sergeant_distance_travel_l274_27452


namespace NUMINAMATH_GPT_sequence_term_20_l274_27468

theorem sequence_term_20 :
  ∀ (a : ℕ → ℕ), (a 1 = 1) → (∀ n, a (n+1) = a n + 2) → (a 20 = 39) := by
  intros a h1 h2
  sorry

end NUMINAMATH_GPT_sequence_term_20_l274_27468


namespace NUMINAMATH_GPT_largest_of_20_consecutive_even_integers_l274_27449

theorem largest_of_20_consecutive_even_integers (x : ℕ) 
  (h : 20 * (x + 19) = 8000) : (x + 38) = 419 :=
  sorry

end NUMINAMATH_GPT_largest_of_20_consecutive_even_integers_l274_27449


namespace NUMINAMATH_GPT_three_digit_number_is_657_l274_27438

theorem three_digit_number_is_657 :
  ∃ (a b c : ℕ), (100 * a + 10 * b + c = 657) ∧ (a + b + c = 18) ∧ (a = b + 1) ∧ (c = b + 2) :=
by
  sorry

end NUMINAMATH_GPT_three_digit_number_is_657_l274_27438


namespace NUMINAMATH_GPT_rainfall_second_week_l274_27417

theorem rainfall_second_week (x : ℝ) (h1 : x + 1.5 * x = 20) : 1.5 * x = 12 := 
by {
  sorry
}

end NUMINAMATH_GPT_rainfall_second_week_l274_27417


namespace NUMINAMATH_GPT_nell_initial_ace_cards_l274_27478

def initial_ace_cards (initial_baseball_cards final_ace_cards final_baseball_cards given_difference : ℕ) : ℕ :=
  final_ace_cards + (initial_baseball_cards - final_baseball_cards)

theorem nell_initial_ace_cards : 
  initial_ace_cards 239 376 111 265 = 504 :=
by
  /- This is to show that the initial count of Ace cards Nell had is 504 given the conditions -/
  sorry

end NUMINAMATH_GPT_nell_initial_ace_cards_l274_27478


namespace NUMINAMATH_GPT_sequence_sum_is_100_then_n_is_10_l274_27494

theorem sequence_sum_is_100_then_n_is_10 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (a 1 = 1) →
  (∀ n, a (n + 1) = a n + 2) →
  (∀ n, S n = n * a 1 + n * (n - 1)) →
  (∃ n, S n = 100) → 
  n = 10 :=
by sorry

end NUMINAMATH_GPT_sequence_sum_is_100_then_n_is_10_l274_27494


namespace NUMINAMATH_GPT_correct_choice_l274_27483

theorem correct_choice (a : ℝ) : -(-a)^2 * a^4 = -a^6 := 
sorry

end NUMINAMATH_GPT_correct_choice_l274_27483


namespace NUMINAMATH_GPT_area_ratio_gt_two_ninths_l274_27498

variables {A B C P Q R : Type*}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P] [Inhabited Q] [Inhabited R]

def divides_perimeter_eq (A B C : Type*) (P Q R : Type*) : Prop :=
-- Definition that P, Q, and R divide the perimeter into three equal parts
sorry

def is_on_side_AB (A B C P Q : Type*) : Prop :=
-- Definition that points P and Q are on side AB
sorry

theorem area_ratio_gt_two_ninths (A B C P Q R : Type*)
  (H1 : divides_perimeter_eq A B C P Q R)
  (H2 : is_on_side_AB A B C P Q) :
  -- Statement to prove that the area ratio is greater than 2/9
  (S_ΔPQR / S_ΔABC) > (2 / 9) :=
sorry

end NUMINAMATH_GPT_area_ratio_gt_two_ninths_l274_27498


namespace NUMINAMATH_GPT_michael_final_revenue_l274_27497

noncomputable def total_revenue_before_discount : ℝ :=
  (3 * 45) + (5 * 22) + (7 * 16) + (8 * 10) + (10 * 5)

noncomputable def discount : ℝ := 0.10 * total_revenue_before_discount

noncomputable def discounted_revenue : ℝ := total_revenue_before_discount - discount

noncomputable def sales_tax : ℝ := 0.06 * discounted_revenue

noncomputable def final_revenue : ℝ := discounted_revenue + sales_tax

theorem michael_final_revenue : final_revenue = 464.60 :=
by
  sorry

end NUMINAMATH_GPT_michael_final_revenue_l274_27497


namespace NUMINAMATH_GPT_john_work_days_l274_27496

theorem john_work_days (J : ℕ) (H1 : 1 / J + 1 / 480 = 1 / 192) : J = 320 :=
sorry

end NUMINAMATH_GPT_john_work_days_l274_27496


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l274_27440

-- Problem 1: 27 - 16 + (-7) - 18 = -14
theorem problem1 : 27 - 16 + (-7) - 18 = -14 := 
by 
  sorry

-- Problem 2: (-6) * (-3/4) / (-3/2) = -3
theorem problem2 : (-6) * (-3/4) / (-3/2) = -3 := 
by
  sorry

-- Problem 3: (1/2 - 3 + 5/6 - 7/12) / (-1/36) = 81
theorem problem3 : (1/2 - 3 + 5/6 - 7/12) / (-1/36) = 81 := 
by
  sorry

-- Problem 4: -2^4 + 3 * (-1)^4 - (-2)^3 = -5
theorem problem4 : -2^4 + 3 * (-1)^4 - (-2)^3 = -5 := 
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l274_27440


namespace NUMINAMATH_GPT_solve_floor_equation_l274_27436

theorem solve_floor_equation (x : ℚ) 
  (h : ⌊(5 + 6 * x) / 8⌋ = (15 * x - 7) / 5) : 
  x = 7 / 15 ∨ x = 4 / 5 := 
sorry

end NUMINAMATH_GPT_solve_floor_equation_l274_27436


namespace NUMINAMATH_GPT_unique_solution_l274_27470

theorem unique_solution (x y z : ℝ) (h₁ : x^2 + y^2 + z^2 = 2) (h₂ : x = z + 2) :
  x = 1 ∧ y = 0 ∧ z = -1 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_l274_27470


namespace NUMINAMATH_GPT_total_muffins_l274_27466

-- Define initial conditions
def initial_muffins : ℕ := 35
def additional_muffins : ℕ := 48

-- Define the main theorem we want to prove
theorem total_muffins : initial_muffins + additional_muffins = 83 :=
by
  sorry

end NUMINAMATH_GPT_total_muffins_l274_27466


namespace NUMINAMATH_GPT_f_neg_a_l274_27415

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 :=
by
  sorry

end NUMINAMATH_GPT_f_neg_a_l274_27415


namespace NUMINAMATH_GPT_negation_of_prop_l274_27441

theorem negation_of_prop :
  (¬ ∀ (x y : ℝ), x^2 + y^2 ≥ 0) ↔ (∃ (x y : ℝ), x^2 + y^2 < 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_prop_l274_27441


namespace NUMINAMATH_GPT_inequality_holds_l274_27408

variable {x y : ℝ}

theorem inequality_holds (h₀ : 0 < x) (h₁ : x < 1) (h₂ : 0 < y) (h₃ : y < 1) :
  (x^2 / (x + y)) + (y^2 / (1 - x)) + ((1 - x - y)^2 / (1 - y)) ≥ 1 / 2 := by
  sorry

end NUMINAMATH_GPT_inequality_holds_l274_27408


namespace NUMINAMATH_GPT_yanna_kept_36_apples_l274_27443

-- Define the initial number of apples Yanna has
def initial_apples : ℕ := 60

-- Define the number of apples given to Zenny
def apples_given_to_zenny : ℕ := 18

-- Define the number of apples given to Andrea
def apples_given_to_andrea : ℕ := 6

-- The proof statement that Yanna kept 36 apples
theorem yanna_kept_36_apples : initial_apples - apples_given_to_zenny - apples_given_to_andrea = 36 := by
  sorry

end NUMINAMATH_GPT_yanna_kept_36_apples_l274_27443


namespace NUMINAMATH_GPT_number_of_terms_in_expansion_l274_27479

theorem number_of_terms_in_expansion (A B : Finset ℕ) (h1 : A.card = 4) (h2 : B.card = 5) :
  (A.product B).card = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_terms_in_expansion_l274_27479


namespace NUMINAMATH_GPT_ways_to_select_computers_l274_27435

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the number of Type A and Type B computers
def num_type_a := 4
def num_type_b := 5

-- Define the total number of computers to select
def total_selected := 3

-- Define the calculation for number of ways to select the computers ensuring both types are included
def ways_to_select := binomial num_type_a 2 * binomial num_type_b 1 + binomial num_type_a 1 * binomial num_type_b 2

-- State the theorem
theorem ways_to_select_computers : ways_to_select = 70 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_ways_to_select_computers_l274_27435


namespace NUMINAMATH_GPT_ben_remaining_money_l274_27430

variable (initial_capital : ℝ := 2000) 
variable (payment_to_supplier : ℝ := 600)
variable (payment_from_debtor : ℝ := 800)
variable (maintenance_cost : ℝ := 1200)
variable (remaining_capital : ℝ := 1000)

theorem ben_remaining_money
  (h1 : initial_capital = 2000)
  (h2 : payment_to_supplier = 600)
  (h3 : payment_from_debtor = 800)
  (h4 : maintenance_cost = 1200) :
  remaining_capital = (initial_capital - payment_to_supplier + payment_from_debtor - maintenance_cost) :=
sorry

end NUMINAMATH_GPT_ben_remaining_money_l274_27430


namespace NUMINAMATH_GPT_part_a_impossible_part_b_possible_l274_27422

-- Part (a)
theorem part_a_impossible (a : ℝ) (h₁ : 1 < a) (h₂ : a ≠ 2) :
  ¬ ∀ (x : ℝ), (1 < x ∧ x < a) ∧ (a < 2*x ∧ 2*x < a^2) :=
sorry

-- Part (b)
theorem part_b_possible (a : ℝ) (h₁ : 1 < a) (h₂ : a ≠ 2) :
  ∃ (x : ℝ), (a < 2*x ∧ 2*x < a^2) ∧ ¬ (1 < x ∧ x < a) :=
sorry

end NUMINAMATH_GPT_part_a_impossible_part_b_possible_l274_27422


namespace NUMINAMATH_GPT_find_A_and_area_l274_27428

open Real

variable (A B C a b c : ℝ)
variable (h1 : 2 * sin A * cos B = 2 * sin C - sin B)
variable (h2 : a = 4 * sqrt 3)
variable (h3 : b + c = 8)
variable (h4 : a^2 = b^2 + c^2 - 2*b*c* cos A)

theorem find_A_and_area :
  A = π / 3 ∧ (1/2 * b * c * sin A = 4 * sqrt 3 / 3) :=
by
  sorry

end NUMINAMATH_GPT_find_A_and_area_l274_27428


namespace NUMINAMATH_GPT_find_wall_width_l274_27404

-- Define the volume of one brick
def volume_of_one_brick : ℚ := 100 * 11.25 * 6

-- Define the total number of bricks
def number_of_bricks : ℕ := 1600

-- Define the volume of all bricks combined
def total_volume_of_bricks : ℚ := volume_of_one_brick * number_of_bricks

-- Define dimensions of the wall
def wall_height : ℚ := 800 -- in cm (since 8 meters = 800 cm)
def wall_depth : ℚ := 22.5 -- in cm

-- Theorem to prove the width of the wall
theorem find_wall_width : ∃ width : ℚ, total_volume_of_bricks = wall_height * width * wall_depth ∧ width = 600 :=
by
  -- skipping the actual proof
  sorry

end NUMINAMATH_GPT_find_wall_width_l274_27404


namespace NUMINAMATH_GPT_boys_laps_eq_27_l274_27424

noncomputable def miles_per_lap : ℝ := 3 / 4
noncomputable def girls_miles : ℝ := 27
noncomputable def girls_extra_laps : ℝ := 9

theorem boys_laps_eq_27 :
  (∃ boys_laps girls_laps : ℝ, 
    girls_laps = girls_miles / miles_per_lap ∧ 
    boys_laps = girls_laps - girls_extra_laps ∧ 
    boys_laps = 27) :=
by
  sorry

end NUMINAMATH_GPT_boys_laps_eq_27_l274_27424


namespace NUMINAMATH_GPT_star_evaluation_l274_27491

noncomputable def star (a b : ℝ) : ℝ := (a + b) / (a - b)

theorem star_evaluation : (star (star 2 3) 4) = 1 / 9 := 
by sorry

end NUMINAMATH_GPT_star_evaluation_l274_27491


namespace NUMINAMATH_GPT_vasya_days_without_purchases_l274_27418

theorem vasya_days_without_purchases 
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) : 
  w = 7 := 
sorry

end NUMINAMATH_GPT_vasya_days_without_purchases_l274_27418


namespace NUMINAMATH_GPT_sum_of_squares_ne_sum_of_fourth_powers_l274_27412

theorem sum_of_squares_ne_sum_of_fourth_powers :
  ∀ (a b : ℤ), a^2 + (a + 1)^2 ≠ b^4 + (b + 1)^4 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_squares_ne_sum_of_fourth_powers_l274_27412


namespace NUMINAMATH_GPT_sin_neg_390_eq_neg_half_l274_27457

theorem sin_neg_390_eq_neg_half : Real.sin (-390 * Real.pi / 180) = -1 / 2 :=
  sorry

end NUMINAMATH_GPT_sin_neg_390_eq_neg_half_l274_27457


namespace NUMINAMATH_GPT_change_back_l274_27451

theorem change_back (price_laptop : ℤ) (price_smartphone : ℤ) (qty_laptops : ℤ) (qty_smartphones : ℤ) (initial_amount : ℤ) (total_cost : ℤ) (change : ℤ) :
  price_laptop = 600 →
  price_smartphone = 400 →
  qty_laptops = 2 →
  qty_smartphones = 4 →
  initial_amount = 3000 →
  total_cost = (price_laptop * qty_laptops) + (price_smartphone * qty_smartphones) →
  change = initial_amount - total_cost →
  change = 200 := by
  sorry

end NUMINAMATH_GPT_change_back_l274_27451


namespace NUMINAMATH_GPT_cost_of_each_soccer_ball_l274_27488

theorem cost_of_each_soccer_ball (total_amount_paid : ℕ) (change_received : ℕ) (number_of_balls : ℕ)
  (amount_spent := total_amount_paid - change_received)
  (unit_price := amount_spent / number_of_balls) :
  total_amount_paid = 100 →
  change_received = 20 →
  number_of_balls = 2 →
  unit_price = 40 := by
  sorry

end NUMINAMATH_GPT_cost_of_each_soccer_ball_l274_27488


namespace NUMINAMATH_GPT_minimum_problem_l274_27447

open BigOperators

theorem minimum_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + 1 / y) * (x + 1 / y - 2020) + (y + 1 / x) * (y + 1 / x - 2020) ≥ -2040200 := 
sorry

end NUMINAMATH_GPT_minimum_problem_l274_27447


namespace NUMINAMATH_GPT_angle_B_measure_l274_27474

theorem angle_B_measure (a b : ℝ) (A B : ℝ) (h₁ : a = 4) (h₂ : b = 4 * Real.sqrt 3) (h₃ : A = Real.pi / 6) : 
  B = Real.pi / 3 ∨ B = 2 * Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_angle_B_measure_l274_27474


namespace NUMINAMATH_GPT_train_crossing_time_l274_27448

noncomputable def time_to_cross_bridge (l_train : ℕ) (v_train_kmh : ℕ) (l_bridge : ℕ) : ℚ :=
  let total_distance := l_train + l_bridge
  let v_train_ms := (v_train_kmh * 1000 : ℚ) / 3600
  total_distance / v_train_ms

theorem train_crossing_time :
  time_to_cross_bridge 110 72 136 = 12.3 := 
by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l274_27448


namespace NUMINAMATH_GPT_problem_l274_27467

theorem problem (a b n : ℕ) (h : ∀ k : ℕ, k ≠ b → b - k ∣ a - k^n) : a = b^n := by
  sorry

end NUMINAMATH_GPT_problem_l274_27467


namespace NUMINAMATH_GPT_find_coordinates_l274_27469

def pointA : ℝ × ℝ := (2, -4)
def pointB : ℝ × ℝ := (0, 6)
def pointC : ℝ × ℝ := (-8, 10)

def vector (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  (p2.1 - p1.1, p2.2 - p1.2)

def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

theorem find_coordinates :
  scalar_mult (1/2) (vector pointA pointC) - 
  scalar_mult (1/4) (vector pointB pointC) = (-3, 6) :=
by
  sorry

end NUMINAMATH_GPT_find_coordinates_l274_27469


namespace NUMINAMATH_GPT_salary_after_cuts_l274_27463

noncomputable def finalSalary (init_salary : ℝ) (cuts : List ℝ) : ℝ :=
  cuts.foldl (λ salary cut => salary * (1 - cut)) init_salary

theorem salary_after_cuts :
  finalSalary 5000 [0.0525, 0.0975, 0.146, 0.128] = 3183.63 :=
by
  sorry

end NUMINAMATH_GPT_salary_after_cuts_l274_27463
