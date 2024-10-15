import Mathlib

namespace NUMINAMATH_GPT_percentage_of_students_in_band_l932_93265

theorem percentage_of_students_in_band 
  (students_in_band : ℕ)
  (total_students : ℕ)
  (students_in_band_eq : students_in_band = 168)
  (total_students_eq : total_students = 840) :
  (students_in_band / total_students : ℚ) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_students_in_band_l932_93265


namespace NUMINAMATH_GPT_lcm_48_75_l932_93274

theorem lcm_48_75 : Nat.lcm 48 75 = 1200 := by
  sorry

end NUMINAMATH_GPT_lcm_48_75_l932_93274


namespace NUMINAMATH_GPT_quarter_pounder_cost_l932_93243

theorem quarter_pounder_cost :
  let fries_cost := 2 * 1.90
  let milkshakes_cost := 2 * 2.40
  let min_purchase := 18
  let current_total := fries_cost + milkshakes_cost
  let amount_needed := min_purchase - current_total
  let additional_spending := 3
  let total_cost := amount_needed + additional_spending
  total_cost = 12.40 :=
by
  sorry

end NUMINAMATH_GPT_quarter_pounder_cost_l932_93243


namespace NUMINAMATH_GPT_find_hourly_rate_l932_93213

theorem find_hourly_rate (x : ℝ) (h1 : 40 * x + 10.75 * 16 = 622) : x = 11.25 :=
sorry

end NUMINAMATH_GPT_find_hourly_rate_l932_93213


namespace NUMINAMATH_GPT_jump_difference_l932_93233

def frog_jump := 39
def grasshopper_jump := 17

theorem jump_difference :
  frog_jump - grasshopper_jump = 22 := by
  sorry

end NUMINAMATH_GPT_jump_difference_l932_93233


namespace NUMINAMATH_GPT_slips_with_3_l932_93204

theorem slips_with_3 (x : ℤ) 
    (h1 : 15 > 0) 
    (h2 : 3 > 0 ∧ 9 > 0) 
    (h3 : (3 * x + 9 * (15 - x)) / 15 = 5) : 
    x = 10 := 
sorry

end NUMINAMATH_GPT_slips_with_3_l932_93204


namespace NUMINAMATH_GPT_particular_solution_satisfies_initial_conditions_l932_93267

noncomputable def x_solution : ℝ → ℝ := λ t => (-4/3) * Real.exp t + (7/3) * Real.exp (-2 * t)
noncomputable def y_solution : ℝ → ℝ := λ t => (-1/3) * Real.exp t + (7/3) * Real.exp (-2 * t)

def x_prime (x y : ℝ) := 2 * x - 4 * y
def y_prime (x y : ℝ) := x - 3 * y

theorem particular_solution_satisfies_initial_conditions :
  (∀ t, deriv x_solution t = x_prime (x_solution t) (y_solution t)) ∧
  (∀ t, deriv y_solution t = y_prime (x_solution t) (y_solution t)) ∧
  (x_solution 0 = 1) ∧
  (y_solution 0 = 2) := by
  sorry

end NUMINAMATH_GPT_particular_solution_satisfies_initial_conditions_l932_93267


namespace NUMINAMATH_GPT_perpendicular_lines_a_value_l932_93239

theorem perpendicular_lines_a_value (a : ℝ) :
  (∀ x y : ℝ, (x + a * y - a = 0) ∧ (a * x - (2 * a - 3) * y - 1 = 0) → x ≠ y) →
  a = 0 ∨ a = 2 :=
sorry

end NUMINAMATH_GPT_perpendicular_lines_a_value_l932_93239


namespace NUMINAMATH_GPT_even_fn_increasing_max_val_l932_93258

variable {f : ℝ → ℝ}

theorem even_fn_increasing_max_val (h_even : ∀ x, f x = f (-x))
    (h_inc_0_5 : ∀ x y, 0 ≤ x → x ≤ y → y ≤ 5 → f x ≤ f y)
    (h_dec_5_inf : ∀ x y, 5 ≤ x → x ≤ y → f y ≤ f x)
    (h_f5 : f 5 = 2) :
    (∀ x y, -5 ≤ x → x ≤ y → y ≤ 0 → f x ≤ f y) ∧ (∀ x, -5 ≤ x → x ≤ 0 → f x ≤ 2) :=
by
    sorry

end NUMINAMATH_GPT_even_fn_increasing_max_val_l932_93258


namespace NUMINAMATH_GPT_inequality_proof_l932_93263

theorem inequality_proof (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) :
  (a^2 - b * c) / (2 * a^2 + b * c) + (b^2 - c * a) / (2 * b^2 + c * a) + (c^2 - a * b) / (2 * c^2 + a * b) ≤ 0 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l932_93263


namespace NUMINAMATH_GPT_tuna_per_customer_l932_93214

noncomputable def total_customers := 100
noncomputable def total_tuna := 10
noncomputable def weight_per_tuna := 200
noncomputable def customers_without_fish := 20

theorem tuna_per_customer : (total_tuna * weight_per_tuna) / (total_customers - customers_without_fish) = 25 := by
  sorry

end NUMINAMATH_GPT_tuna_per_customer_l932_93214


namespace NUMINAMATH_GPT_point_outside_circle_l932_93208

theorem point_outside_circle (D E F x0 y0 : ℝ) (h : (x0 + D / 2)^2 + (y0 + E / 2)^2 > (D^2 + E^2 - 4 * F) / 4) :
  x0^2 + y0^2 + D * x0 + E * y0 + F > 0 :=
sorry

end NUMINAMATH_GPT_point_outside_circle_l932_93208


namespace NUMINAMATH_GPT_area_of_triangle_ABC_sinA_value_l932_93218

noncomputable def cosC := 3 / 4
noncomputable def sinC := Real.sqrt (1 - cosC ^ 2)
noncomputable def a := 1
noncomputable def b := 2
noncomputable def c := Real.sqrt (a ^ 2 + b ^ 2 - 2 * a * b * cosC)
noncomputable def area := (1 / 2) * a * b * sinC
noncomputable def sinA := (a * sinC) / c

theorem area_of_triangle_ABC : area = Real.sqrt 7 / 4 :=
by sorry

theorem sinA_value : sinA = Real.sqrt 14 / 8 :=
by sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_sinA_value_l932_93218


namespace NUMINAMATH_GPT_jane_rejects_percent_l932_93212

theorem jane_rejects_percent :
  -- Declare the conditions as hypotheses
  ∀ (P : ℝ) (J : ℝ) (john_frac_reject : ℝ) (total_reject_percent : ℝ) (jane_inspect_frac : ℝ),
  john_frac_reject = 0.005 →
  total_reject_percent = 0.0075 →
  jane_inspect_frac = 5 / 6 →
  -- Given the rejection equation
  (john_frac_reject * (1 / 6) * P + (J / 100) * jane_inspect_frac * P = total_reject_percent * P) →
  -- Prove that Jane rejected 0.8% of the products she inspected
  J = 0.8 :=
by {
  sorry
}

end NUMINAMATH_GPT_jane_rejects_percent_l932_93212


namespace NUMINAMATH_GPT_first_triangular_number_year_in_21st_century_l932_93285

theorem first_triangular_number_year_in_21st_century :
  ∃ n : ℕ, (n * (n + 1)) / 2 = 2016 ∧ 2000 ≤ 2016 ∧ 2016 < 2100 :=
by
  sorry

end NUMINAMATH_GPT_first_triangular_number_year_in_21st_century_l932_93285


namespace NUMINAMATH_GPT_point_P_path_length_l932_93201

/-- A rectangle PQRS in the plane with points P Q R S, where PQ = RS = 2 and QR = SP = 6. 
    The rectangle is rotated 90 degrees twice: first about point R and then 
    about the new position of point S after the first rotation. 
    The goal is to prove that the length of the path P travels is (3 + sqrt 10) * pi. -/
theorem point_P_path_length :
  ∀ (P Q R S : ℝ × ℝ), 
    dist P Q = 2 ∧ dist Q R = 6 ∧ dist R S = 2 ∧ dist S P = 6 →
    ∃ path_length : ℝ, path_length = (3 + Real.sqrt 10) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_point_P_path_length_l932_93201


namespace NUMINAMATH_GPT_border_area_correct_l932_93268

theorem border_area_correct :
  let photo_height := 9
  let photo_width := 12
  let border_width := 3
  let photo_area := photo_height * photo_width
  let framed_height := photo_height + 2 * border_width
  let framed_width := photo_width + 2 * border_width
  let framed_area := framed_height * framed_width
  let border_area := framed_area - photo_area
  border_area = 162 :=
by sorry

end NUMINAMATH_GPT_border_area_correct_l932_93268


namespace NUMINAMATH_GPT_consecutive_integers_no_two_l932_93261

theorem consecutive_integers_no_two (a n : ℕ) : 
  ¬(∃ (b : ℤ), (b : ℤ) = 2) :=
sorry

end NUMINAMATH_GPT_consecutive_integers_no_two_l932_93261


namespace NUMINAMATH_GPT_abs_diff_roots_quad_eq_l932_93217

theorem abs_diff_roots_quad_eq : 
  ∀ (r1 r2 : ℝ), 
  (r1 * r2 = 12) ∧ (r1 + r2 = 7) → |r1 - r2| = 1 :=
by
  intro r1 r2 h
  sorry

end NUMINAMATH_GPT_abs_diff_roots_quad_eq_l932_93217


namespace NUMINAMATH_GPT_books_left_unchanged_l932_93222

theorem books_left_unchanged (initial_books : ℕ) (initial_pens : ℕ) (pens_sold : ℕ) (pens_left : ℕ) :
  initial_books = 51 → initial_pens = 106 → pens_sold = 92 → pens_left = 14 → initial_books = 51 := 
by
  intros h_books h_pens h_sold h_left
  exact h_books

end NUMINAMATH_GPT_books_left_unchanged_l932_93222


namespace NUMINAMATH_GPT_total_rainfall_l932_93203

theorem total_rainfall
  (r₁ r₂ : ℕ)
  (T t₁ : ℕ)
  (H1 : r₁ = 30)
  (H2 : r₂ = 15)
  (H3 : T = 45)
  (H4 : t₁ = 20) :
  r₁ * t₁ + r₂ * (T - t₁) = 975 := by
  sorry

end NUMINAMATH_GPT_total_rainfall_l932_93203


namespace NUMINAMATH_GPT_angle_triple_supplement_l932_93290

theorem angle_triple_supplement (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end NUMINAMATH_GPT_angle_triple_supplement_l932_93290


namespace NUMINAMATH_GPT_complement_A_complement_B_intersection_A_B_complement_union_A_B_l932_93223

open Set

variable (U : Set ℝ) (A B : Set ℝ)

def set_U : Set ℝ := {x | true}  -- This represents U = ℝ
def set_A : Set ℝ := {x | x < -2 ∨ x > 5}
def set_B : Set ℝ := {x | 4 ≤ x ∧ x ≤ 6}

theorem complement_A :
  ∀ x : ℝ, x ∈ set_U \ set_A ↔ -2 ≤ x ∧ x ≤ 5 :=
by
  intro x
  sorry

theorem complement_B :
  ∀ x : ℝ, x ∉ set_B ↔ x < 4 ∨ x > 6 :=
by
  intro x
  sorry

theorem intersection_A_B :
  ∀ x : ℝ, x ∈ set_A ∩ set_B ↔ 5 < x ∧ x ≤ 6 :=
by
  intro x
  sorry

theorem complement_union_A_B :
  ∀ x : ℝ, x ∈ set_U \ (set_A ∪ set_B) ↔ -2 ≤ x ∧ x < 4 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_complement_A_complement_B_intersection_A_B_complement_union_A_B_l932_93223


namespace NUMINAMATH_GPT_length_of_other_parallel_side_l932_93255

theorem length_of_other_parallel_side (a b h area : ℝ) 
  (h_area : area = 190) 
  (h_parallel1 : b = 18) 
  (h_height : h = 10) : 
  a = 20 :=
by
  sorry

end NUMINAMATH_GPT_length_of_other_parallel_side_l932_93255


namespace NUMINAMATH_GPT_loaves_per_hour_in_one_oven_l932_93294

-- Define the problem constants and variables
def loaves_in_3_weeks : ℕ := 1740
def ovens : ℕ := 4
def weekday_hours : ℕ := 5
def weekend_hours : ℕ := 2
def weekdays_per_week : ℕ := 5
def weekends_per_week : ℕ := 2
def weeks : ℕ := 3

-- Calculate the total hours per week
def hours_per_week : ℕ := (weekdays_per_week * weekday_hours) + (weekends_per_week * weekend_hours)

-- Calculate the total oven-hours for 3 weeks
def total_oven_hours : ℕ := hours_per_week * ovens * weeks

-- Provide the proof statement
theorem loaves_per_hour_in_one_oven : (loaves_in_3_weeks = 5 * total_oven_hours) :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_loaves_per_hour_in_one_oven_l932_93294


namespace NUMINAMATH_GPT_at_least_one_is_zero_l932_93295

theorem at_least_one_is_zero (a b : ℕ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : false := by sorry

end NUMINAMATH_GPT_at_least_one_is_zero_l932_93295


namespace NUMINAMATH_GPT_arithmetic_seq_20th_term_l932_93238

variable (a : ℕ → ℤ) -- a_n is an arithmetic sequence
variable (d : ℤ) -- common difference of the arithmetic sequence

-- Condition for arithmetic sequence
variable (h_seq : ∀ n, a (n+1) = a n + d)

-- Given conditions
axiom h1 : a 1 + a 3 + a 5 = 105
axiom h2 : a 2 + a 4 + a 6 = 99

-- Goal: prove that a 20 = 1
theorem arithmetic_seq_20th_term :
  a 20 = 1 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_20th_term_l932_93238


namespace NUMINAMATH_GPT_compute_fraction_mul_l932_93216

theorem compute_fraction_mul :
  (1 / 3) ^ 2 * (1 / 8) = 1 / 72 :=
by
  sorry

end NUMINAMATH_GPT_compute_fraction_mul_l932_93216


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_and_general_formula_l932_93249

variable (a : ℕ → ℝ)

theorem geometric_sequence_common_ratio_and_general_formula (h₁ : a 1 = 1) (h₃ : a 3 = 4) : 
  (∃ q : ℝ, q = 2 ∨ q = -2 ∧ (∀ n : ℕ, a n = 2^(n-1) ∨ a n = (-2)^(n-1))) := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_and_general_formula_l932_93249


namespace NUMINAMATH_GPT_cockatiel_weekly_consumption_is_50_l932_93210

def boxes_bought : ℕ := 3
def boxes_existing : ℕ := 5
def grams_per_box : ℕ := 225
def parrot_weekly_consumption : ℕ := 100
def weeks_supply : ℕ := 12

def total_boxes : ℕ := boxes_bought + boxes_existing
def total_birdseed_grams : ℕ := total_boxes * grams_per_box
def parrot_total_consumption : ℕ := parrot_weekly_consumption * weeks_supply
def cockatiel_total_consumption : ℕ := total_birdseed_grams - parrot_total_consumption
def cockatiel_weekly_consumption : ℕ := cockatiel_total_consumption / weeks_supply

theorem cockatiel_weekly_consumption_is_50 :
  cockatiel_weekly_consumption = 50 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cockatiel_weekly_consumption_is_50_l932_93210


namespace NUMINAMATH_GPT_intersection_A_B_l932_93297

variable (x : ℝ)

def setA : Set ℝ := { x | x^2 - 4*x - 5 < 0 }
def setB : Set ℝ := { x | -2 < x ∧ x < 2 }

theorem intersection_A_B :
  { x | x^2 - 4*x - 5 < 0 } ∩ { x | -2 < x ∧ x < 2 } = { x | -1 < x ∧ x < 2 } :=
by
  -- Here would be the proof, but we use sorry to skip it
  sorry

end NUMINAMATH_GPT_intersection_A_B_l932_93297


namespace NUMINAMATH_GPT_equilateral_given_inequality_l932_93226

open Real

-- Define the primary condition to be used in the theorem
def inequality (a b c : ℝ) : Prop :=
  (1 / a * sqrt (1 / b + 1 / c) + 1 / b * sqrt (1 / c + 1 / a) + 1 / c * sqrt (1 / a + 1 / b)) ≥
  (3 / 2 * sqrt ((1 / a + 1 / b) * (1 / b + 1 / c) * (1 / c + 1 / a)))

-- Define the theorem that states the sides form an equilateral triangle under the given condition
theorem equilateral_given_inequality (a b c : ℝ) (habc : inequality a b c) (htriangle : a > 0 ∧ b > 0 ∧ c > 0):
  a = b ∧ b = c ∧ c = a := 
sorry

end NUMINAMATH_GPT_equilateral_given_inequality_l932_93226


namespace NUMINAMATH_GPT_quadratic_two_equal_real_roots_l932_93298

theorem quadratic_two_equal_real_roots (m : ℝ) :
  (∃ (x : ℝ), x^2 + m * x + m = 0 ∧ ∀ (y : ℝ), x = y → x^2 + m * y + m = 0) →
  (m = 0 ∨ m = 4) :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_two_equal_real_roots_l932_93298


namespace NUMINAMATH_GPT_simplify_polynomial_l932_93230

theorem simplify_polynomial (q : ℚ) :
  (4 * q^3 - 7 * q^2 + 3 * q - 2) + (5 * q^2 - 9 * q + 8) = 4 * q^3 - 2 * q^2 - 6 * q + 6 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l932_93230


namespace NUMINAMATH_GPT_find_cubic_expression_l932_93242

theorem find_cubic_expression (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 648 := by
  sorry

end NUMINAMATH_GPT_find_cubic_expression_l932_93242


namespace NUMINAMATH_GPT_g_of_50_eq_zero_l932_93241

theorem g_of_50_eq_zero (g : ℝ → ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y → x * g y - 3 * y * g x = g (x / y)) : g 50 = 0 :=
sorry

end NUMINAMATH_GPT_g_of_50_eq_zero_l932_93241


namespace NUMINAMATH_GPT_percent_gain_correct_l932_93293

theorem percent_gain_correct :
  ∀ (x : ℝ), (900 * x + 50 * (900 * x / 850) - 900 * x) / (900 * x) * 100 = 58.82 :=
by sorry

end NUMINAMATH_GPT_percent_gain_correct_l932_93293


namespace NUMINAMATH_GPT_carriage_problem_l932_93228

theorem carriage_problem (x : ℕ) : 
  3 * (x - 2) = 2 * x + 9 := 
sorry

end NUMINAMATH_GPT_carriage_problem_l932_93228


namespace NUMINAMATH_GPT_entree_cost_14_l932_93209

-- Define the conditions as given in part a)
def total_cost (e d : ℕ) : Prop := e + d = 23
def entree_more (e d : ℕ) : Prop := e = d + 5

-- The theorem to be proved
theorem entree_cost_14 (e d : ℕ) (h1 : total_cost e d) (h2 : entree_more e d) : e = 14 := 
by 
  sorry

end NUMINAMATH_GPT_entree_cost_14_l932_93209


namespace NUMINAMATH_GPT_find_n_values_l932_93257

theorem find_n_values (n : ℚ) :
  ( 4 * n ^ 2 + 3 * n + 2 = 2 * n + 2 ∨ 4 * n ^ 2 + 3 * n + 2 = 5 * n + 4 ) →
  ( n = 0 ∨ n = 1 ) :=
by
  sorry

end NUMINAMATH_GPT_find_n_values_l932_93257


namespace NUMINAMATH_GPT_cosine_range_l932_93291

theorem cosine_range {x : ℝ} (h1 : 0 ≤ x) (h2 : x ≤ 2 * Real.pi) (h3 : Real.cos x ≤ 1 / 2) : 
  x ∈ Set.Icc (Real.pi / 3) (5 * Real.pi / 3) :=
by
  sorry

end NUMINAMATH_GPT_cosine_range_l932_93291


namespace NUMINAMATH_GPT_most_colored_pencils_l932_93246

theorem most_colored_pencils (total red blue yellow : ℕ) 
  (h_total : total = 24)
  (h_red : red = total / 4)
  (h_blue : blue = red + 6)
  (h_yellow : yellow = total - (red + blue)) :
  blue = 12 :=
by
  sorry

end NUMINAMATH_GPT_most_colored_pencils_l932_93246


namespace NUMINAMATH_GPT_reciprocal_div_calculate_fraction_reciprocal_div_result_l932_93271

-- Part 1
theorem reciprocal_div {a b c : ℚ} (h : (a + b) / c = -2) : c / (a + b) = -1 / 2 :=
sorry

-- Part 2
theorem calculate_fraction : (5 / 12 - 1 / 9 + 2 / 3) / (1 / 36) = 35 :=
sorry

-- Part 3
theorem reciprocal_div_result : (5 / 12 - 1 / 9 + 2 / 3) / (1 / 36) = 35 →
 (-1 / 36) / (5 / 12 - 1 / 9 + 2 / 3) = -1 / 35 :=
sorry

end NUMINAMATH_GPT_reciprocal_div_calculate_fraction_reciprocal_div_result_l932_93271


namespace NUMINAMATH_GPT_good_numbers_count_1_to_50_l932_93275

def is_good_number (n : ℕ) : Prop :=
  ∃ (k l : ℕ), k ≠ 0 ∧ l ≠ 0 ∧ n = k * l + l - k

theorem good_numbers_count_1_to_50 : ∃ cnt, cnt = 49 ∧ (∀ n, n ∈ (Finset.range 51).erase 0 → is_good_number n) :=
  sorry

end NUMINAMATH_GPT_good_numbers_count_1_to_50_l932_93275


namespace NUMINAMATH_GPT_range_of_square_root_l932_93262

theorem range_of_square_root (x : ℝ) : x + 4 ≥ 0 → x ≥ -4 :=
by
  intro h
  linarith

end NUMINAMATH_GPT_range_of_square_root_l932_93262


namespace NUMINAMATH_GPT_strawberry_quality_meets_standard_l932_93256

def acceptable_weight_range (w : ℝ) : Prop :=
  4.97 ≤ w ∧ w ≤ 5.03

theorem strawberry_quality_meets_standard :
  acceptable_weight_range 4.98 :=
by
  sorry

end NUMINAMATH_GPT_strawberry_quality_meets_standard_l932_93256


namespace NUMINAMATH_GPT_bubbleSort_iter_count_l932_93282

/-- Bubble sort iterates over the list repeatedly, swapping adjacent elements if they are in the wrong order. -/
def bubbleSortSteps (lst : List Int) : List (List Int) :=
sorry -- Implementation of bubble sort to capture each state after each iteration

/-- Prove that sorting [6, -3, 0, 15] in descending order using bubble sort requires exactly 3 iterations. -/
theorem bubbleSort_iter_count : 
  (bubbleSortSteps [6, -3, 0, 15]).length = 3 :=
sorry

end NUMINAMATH_GPT_bubbleSort_iter_count_l932_93282


namespace NUMINAMATH_GPT_problem_equivalent_l932_93225

theorem problem_equivalent :
  2^1998 - 2^1997 - 2^1996 + 2^1995 = 3 * 2^1995 :=
by
  sorry

end NUMINAMATH_GPT_problem_equivalent_l932_93225


namespace NUMINAMATH_GPT_one_half_of_scientific_notation_l932_93266

theorem one_half_of_scientific_notation :
  (1 / 2) * (1.2 * 10 ^ 30) = 6.0 * 10 ^ 29 :=
by
  sorry

end NUMINAMATH_GPT_one_half_of_scientific_notation_l932_93266


namespace NUMINAMATH_GPT_cows_total_l932_93296

theorem cows_total (M F : ℕ) 
  (h1 : F = 2 * M) 
  (h2 : F / 2 = M / 2 + 50) : 
  M + F = 300 :=
by
  sorry

end NUMINAMATH_GPT_cows_total_l932_93296


namespace NUMINAMATH_GPT_find_range_of_m_l932_93284

-- Define propositions p and q based on the problem description
def p (m : ℝ) : Prop :=
  ∀ x y : ℝ, m ≠ 0 → (x - 2 * y + 3 = 0 ∧ y * y ≠ m * x)

def q (m : ℝ) : Prop :=
  5 - 2 * m ≠ 0 ∧ m ≠ 0 ∧ (∃ x y : ℝ, (x * x) / (5 - 2 * m) + (y * y) / m = 1)

-- Given conditions
def condition1 (m : ℝ) : Prop := p m ∨ q m
def condition2 (m : ℝ) : Prop := ¬ (p m ∧ q m)

-- The range of m that satisfies the given problem
def valid_m (m : ℝ) : Prop :=
  (m ≥ 3) ∨ (m < 0) ∨ (0 < m ∧ m ≤ 2.5)

theorem find_range_of_m (m : ℝ) : condition1 m → condition2 m → valid_m m := 
  sorry

end NUMINAMATH_GPT_find_range_of_m_l932_93284


namespace NUMINAMATH_GPT_fill_pool_time_l932_93205

theorem fill_pool_time (R : ℝ) (T : ℝ) (hSlowerPipe : R = 1 / 9) (hFasterPipe : 1.25 * R = 1.25 / 9)
                     (hCombinedRate : 2.25 * R = 2.25 / 9) : T = 4 := by
  sorry

end NUMINAMATH_GPT_fill_pool_time_l932_93205


namespace NUMINAMATH_GPT_largest_multiple_of_7_whose_negation_greater_than_neg80_l932_93211

theorem largest_multiple_of_7_whose_negation_greater_than_neg80 : ∃ (n : ℤ), n = 77 ∧ (∃ (k : ℤ), n = k * 7) ∧ (-n > -80) :=
by
  sorry

end NUMINAMATH_GPT_largest_multiple_of_7_whose_negation_greater_than_neg80_l932_93211


namespace NUMINAMATH_GPT_divisor_condition_l932_93229

def M (n : ℤ) : Set ℤ := {n, n+1, n+2, n+3, n+4}

def S (n : ℤ) : ℤ := 5*n^2 + 20*n + 30

def P (n : ℤ) : ℤ := (n * (n+1) * (n+2) * (n+3) * (n+4))^2

theorem divisor_condition (n : ℤ) : S n ∣ P n ↔ n = 3 := 
by
  sorry

end NUMINAMATH_GPT_divisor_condition_l932_93229


namespace NUMINAMATH_GPT_tangent_line_to_curve_l932_93221

theorem tangent_line_to_curve (a : ℝ) : (∀ (x : ℝ), y = x → y = a + Real.log x) → a = 1 := 
sorry

end NUMINAMATH_GPT_tangent_line_to_curve_l932_93221


namespace NUMINAMATH_GPT_jake_needs_total_hours_to_pay_off_debts_l932_93220

-- Define the conditions for the debts and payments
variable (debtA debtB debtC : ℝ)
variable (paymentA paymentB paymentC : ℝ)
variable (task1P task2P task3P task4P task5P task6P : ℝ)
variable (task2Payoff task4Payoff task6Payoff : ℝ)

-- Assume provided values
noncomputable def total_hours_needed : ℝ :=
  let remainingA := debtA - paymentA
  let remainingB := debtB - paymentB
  let remainingC := debtC - paymentC
  let hoursTask1 := (remainingA - task2Payoff) / task1P
  let hoursTask2 := task2Payoff / task2P
  let hoursTask3 := (remainingB - task4Payoff) / task3P
  let hoursTask4 := task4Payoff / task4P
  let hoursTask5 := (remainingC - task6Payoff) / task5P
  let hoursTask6 := task6Payoff / task6P
  hoursTask1 + hoursTask2 + hoursTask3 + hoursTask4 + hoursTask5 + hoursTask6

-- Given our specific problem conditions
theorem jake_needs_total_hours_to_pay_off_debts :
  total_hours_needed 150 200 250 60 80 100 15 12 20 10 25 30 30 40 60 = 20.1 :=
by
  sorry

end NUMINAMATH_GPT_jake_needs_total_hours_to_pay_off_debts_l932_93220


namespace NUMINAMATH_GPT_sum_of_coefficients_l932_93278

noncomputable def expand_and_sum_coefficients (d : ℝ) : ℝ :=
  let poly := -2 * (4 - d) * (d + 3 * (4 - d))
  let expanded := -4 * d^2 + 40 * d - 96
  let sum_coefficients := (-4) + 40 + (-96)
  sum_coefficients

theorem sum_of_coefficients (d : ℝ) : expand_and_sum_coefficients d = -60 := by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l932_93278


namespace NUMINAMATH_GPT_fourth_term_sum_eq_40_l932_93244

theorem fourth_term_sum_eq_40 : 3^0 + 3^1 + 3^2 + 3^3 = 40 := by
  sorry

end NUMINAMATH_GPT_fourth_term_sum_eq_40_l932_93244


namespace NUMINAMATH_GPT_carpet_area_in_yards_l932_93250

def main_length_feet : ℕ := 15
def main_width_feet : ℕ := 12
def extension_length_feet : ℕ := 6
def extension_width_feet : ℕ := 5
def feet_per_yard : ℕ := 3

def main_length_yards : ℕ := main_length_feet / feet_per_yard
def main_width_yards : ℕ := main_width_feet / feet_per_yard
def extension_length_yards : ℕ := extension_length_feet / feet_per_yard
def extension_width_yards : ℕ := extension_width_feet / feet_per_yard

def main_area_yards : ℕ := main_length_yards * main_width_yards
def extension_area_yards : ℕ := extension_length_yards * extension_width_yards

theorem carpet_area_in_yards : (main_area_yards : ℚ) + (extension_area_yards : ℚ) = 23.33 := 
by
  apply sorry

end NUMINAMATH_GPT_carpet_area_in_yards_l932_93250


namespace NUMINAMATH_GPT_circles_intersection_distance_squared_l932_93240

open Real

-- Definitions of circles
def circle1 (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 1)^2 = 25

def circle2 (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 6)^2 = 9

-- Theorem to prove
theorem circles_intersection_distance_squared :
  ∃ A B : (ℝ × ℝ), circle1 A.1 A.2 ∧ circle2 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 B.1 B.2 ∧ (A ≠ B) ∧
  (dist A B)^2 = 675 / 49 :=
sorry

end NUMINAMATH_GPT_circles_intersection_distance_squared_l932_93240


namespace NUMINAMATH_GPT_bottles_have_200_mL_l932_93215

def liters_to_milliliters (liters : ℕ) : ℕ :=
  liters * 1000

def total_milliliters (liters : ℕ) : ℕ :=
  liters_to_milliliters liters

def milliliters_per_bottle (total_mL : ℕ) (num_bottles : ℕ) : ℕ :=
  total_mL / num_bottles

theorem bottles_have_200_mL (num_bottles : ℕ) (total_oil_liters : ℕ) (h1 : total_oil_liters = 4) (h2 : num_bottles = 20) :
  milliliters_per_bottle (total_milliliters total_oil_liters) num_bottles = 200 := 
by
  sorry

end NUMINAMATH_GPT_bottles_have_200_mL_l932_93215


namespace NUMINAMATH_GPT_train_length_l932_93252

theorem train_length (speed_fast speed_slow : ℝ) (time_pass : ℝ)
  (L : ℝ)
  (hf : speed_fast = 46 * (1000/3600))
  (hs : speed_slow = 36 * (1000/3600))
  (ht : time_pass = 36)
  (hL : (2 * L = (speed_fast - speed_slow) * time_pass)) :
  L = 50 := by
  sorry

end NUMINAMATH_GPT_train_length_l932_93252


namespace NUMINAMATH_GPT_total_colors_over_two_hours_l932_93232

def colors_in_first_hour : Nat :=
  let quick_colors := 5 * 3
  let slow_colors := 2 * 3
  quick_colors + slow_colors

def colors_in_second_hour : Nat :=
  let quick_colors := (5 * 2) * 3
  let slow_colors := (2 * 2) * 3
  quick_colors + slow_colors

theorem total_colors_over_two_hours : colors_in_first_hour + colors_in_second_hour = 63 := by
  sorry

end NUMINAMATH_GPT_total_colors_over_two_hours_l932_93232


namespace NUMINAMATH_GPT_gondor_total_earnings_l932_93283

-- Defining the earnings from repairing a phone and a laptop
def phone_earning : ℕ := 10
def laptop_earning : ℕ := 20

-- Defining the number of repairs
def monday_phone_repairs : ℕ := 3
def tuesday_phone_repairs : ℕ := 5
def wednesday_laptop_repairs : ℕ := 2
def thursday_laptop_repairs : ℕ := 4

-- Calculating total earnings
def monday_earnings : ℕ := monday_phone_repairs * phone_earning
def tuesday_earnings : ℕ := tuesday_phone_repairs * phone_earning
def wednesday_earnings : ℕ := wednesday_laptop_repairs * laptop_earning
def thursday_earnings : ℕ := thursday_laptop_repairs * laptop_earning

def total_earnings : ℕ := monday_earnings + tuesday_earnings + wednesday_earnings + thursday_earnings

-- The theorem to be proven
theorem gondor_total_earnings : total_earnings = 200 := by
  sorry

end NUMINAMATH_GPT_gondor_total_earnings_l932_93283


namespace NUMINAMATH_GPT_calculate_glass_area_l932_93277

-- Given conditions as definitions
def long_wall_length : ℕ := 30
def long_wall_height : ℕ := 12
def short_wall_length : ℕ := 20

-- Total area of glass required (what we want to prove)
def total_glass_area : ℕ := 960

-- The theorem to prove
theorem calculate_glass_area
  (a1 : long_wall_length = 30)
  (a2 : long_wall_height = 12)
  (a3 : short_wall_length = 20) :
  2 * (long_wall_length * long_wall_height) + (short_wall_length * long_wall_height) = total_glass_area :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_calculate_glass_area_l932_93277


namespace NUMINAMATH_GPT_ratio_of_height_to_radius_l932_93237

theorem ratio_of_height_to_radius (r h : ℝ)
  (h_cone : r > 0 ∧ h > 0)
  (circumference_cone_base : 20 * 2 * Real.pi * r = 2 * Real.pi * Real.sqrt (r^2 + h^2))
  : h / r = Real.sqrt 399 := by
  sorry

end NUMINAMATH_GPT_ratio_of_height_to_radius_l932_93237


namespace NUMINAMATH_GPT_range_of_k_for_real_roots_l932_93273

theorem range_of_k_for_real_roots (k : ℝ) : 
  (∃ x, 2 * x^2 - 3 * x = k) ↔ k ≥ -9/8 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_for_real_roots_l932_93273


namespace NUMINAMATH_GPT_broken_seashells_count_l932_93202

def total_seashells : Nat := 6
def unbroken_seashells : Nat := 2
def broken_seashells : Nat := total_seashells - unbroken_seashells

theorem broken_seashells_count :
  broken_seashells = 4 :=
by
  -- The proof would go here, but for now, we use 'sorry' to denote it.
  sorry

end NUMINAMATH_GPT_broken_seashells_count_l932_93202


namespace NUMINAMATH_GPT_max_contribution_l932_93292

theorem max_contribution (total_contribution : ℝ) (num_people : ℕ) (min_contribution_each : ℝ) (h1 : total_contribution = 45.00) (h2 : num_people = 25) (h3 : min_contribution_each = 1.00) : 
  ∃ max_cont : ℝ, max_cont = 21.00 :=
by
  sorry

end NUMINAMATH_GPT_max_contribution_l932_93292


namespace NUMINAMATH_GPT_one_quarters_in_one_eighth_l932_93280

theorem one_quarters_in_one_eighth : (1 / 8) / (1 / 4) = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_one_quarters_in_one_eighth_l932_93280


namespace NUMINAMATH_GPT_pell_infinite_solutions_l932_93281

theorem pell_infinite_solutions : ∃ m : ℕ, ∃ a b c : ℕ, 
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ 
  (∀ n : ℕ, ∃ an bn cn : ℕ, 
    (1 / an + 1 / bn + 1 / cn + 1 / (an * bn * cn) = m / (an + bn + cn))) := 
sorry

end NUMINAMATH_GPT_pell_infinite_solutions_l932_93281


namespace NUMINAMATH_GPT_sum_reciprocals_seven_l932_93264

variable (x y : ℝ)

theorem sum_reciprocals_seven (h : x + y = 7 * x * y) (hx : x ≠ 0) (hy : y ≠ 0) :
  (1 / x) + (1 / y) = 7 := 
sorry

end NUMINAMATH_GPT_sum_reciprocals_seven_l932_93264


namespace NUMINAMATH_GPT_area_of_trapezoid_l932_93251

variable (a d : ℝ)
variable (h b1 b2 : ℝ)

def is_arithmetic_progression (a d : ℝ) (h b1 b2 : ℝ) : Prop :=
  h = a ∧ b1 = a + d ∧ b2 = a - d

theorem area_of_trapezoid (a d : ℝ) (h b1 b2 : ℝ) (hAP : is_arithmetic_progression a d h b1 b2) :
  ∃ J : ℝ, J = a^2 ∧ ∀ x : ℝ, 0 ≤ x → (J = x → x ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_area_of_trapezoid_l932_93251


namespace NUMINAMATH_GPT_weather_conclusion_l932_93254

variables (T C : ℝ) (visitors : ℕ)

def condition1 : Prop :=
  (T ≥ 75.0 ∧ C < 10) → visitors > 100

def condition2 : Prop :=
  visitors ≤ 100

theorem weather_conclusion (h1 : condition1 T C visitors) (h2 : condition2 visitors) : 
  T < 75.0 ∨ C ≥ 10 :=
by 
  sorry

end NUMINAMATH_GPT_weather_conclusion_l932_93254


namespace NUMINAMATH_GPT_total_regular_and_diet_soda_bottles_l932_93207

-- Definitions from the conditions
def regular_soda_bottles := 49
def diet_soda_bottles := 40

-- The statement to prove
theorem total_regular_and_diet_soda_bottles :
  regular_soda_bottles + diet_soda_bottles = 89 :=
by
  sorry

end NUMINAMATH_GPT_total_regular_and_diet_soda_bottles_l932_93207


namespace NUMINAMATH_GPT_subcommittees_with_at_least_one_teacher_l932_93234

theorem subcommittees_with_at_least_one_teacher
  (total_members teachers : ℕ)
  (total_members_eq : total_members = 12)
  (teachers_eq : teachers = 5)
  (subcommittee_size : ℕ)
  (subcommittee_size_eq : subcommittee_size = 5) :
  ∃ (n : ℕ), n = 771 :=
by
  sorry

end NUMINAMATH_GPT_subcommittees_with_at_least_one_teacher_l932_93234


namespace NUMINAMATH_GPT_additional_days_use_l932_93286

variable (m a : ℝ)

theorem additional_days_use (hm : m > 0) (ha : a > 1) : 
  (m / (a - 1) - m / a) = m / (a * (a - 1)) :=
sorry

end NUMINAMATH_GPT_additional_days_use_l932_93286


namespace NUMINAMATH_GPT_sum_of_integers_70_to_85_l932_93289

theorem sum_of_integers_70_to_85 :
  let range_start := 70
  let range_end := 85
  let n := range_end - range_start + 1
  let sum := (range_start + range_end) * n / 2
  sum = 1240 :=
by
  let range_start := 70
  let range_end := 85
  let n := range_end - range_start + 1
  let sum := (range_start + range_end) * n / 2
  sorry

end NUMINAMATH_GPT_sum_of_integers_70_to_85_l932_93289


namespace NUMINAMATH_GPT_sufficient_condition_for_equation_l932_93200

theorem sufficient_condition_for_equation (x y z : ℤ) (h1 : x = y + 1) (h2 : z = y) :
    x * (x - y) + y * (y - z) + z * (z - x) = 1 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_sufficient_condition_for_equation_l932_93200


namespace NUMINAMATH_GPT_water_for_bathing_per_horse_per_day_l932_93253

-- Definitions of the given conditions
def initial_horses : ℕ := 3
def additional_horses : ℕ := 5
def total_horses : ℕ := initial_horses + additional_horses
def drink_water_per_horse_per_day : ℕ := 5
def total_days : ℕ := 28
def total_water_needed : ℕ := 1568

-- The proven statement
theorem water_for_bathing_per_horse_per_day :
  ((total_water_needed - (total_horses * drink_water_per_horse_per_day * total_days)) / (total_horses * total_days)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_water_for_bathing_per_horse_per_day_l932_93253


namespace NUMINAMATH_GPT_sqrt_9_is_rational_l932_93270

theorem sqrt_9_is_rational : ∃ q : ℚ, (q : ℝ) = 3 := by
  sorry

end NUMINAMATH_GPT_sqrt_9_is_rational_l932_93270


namespace NUMINAMATH_GPT_probability_not_all_dice_same_l932_93287

theorem probability_not_all_dice_same :
  let total_outcomes := 6^5
  let same_number_outcomes := 6
  let probability_same_number := same_number_outcomes / total_outcomes
  let probability_not_same_number := 1 - probability_same_number
  probability_not_same_number = (1295 : ℚ) / 1296 :=
by
  sorry

end NUMINAMATH_GPT_probability_not_all_dice_same_l932_93287


namespace NUMINAMATH_GPT_ratio_gluten_free_l932_93236

theorem ratio_gluten_free (total_cupcakes vegan_cupcakes non_vegan_gluten cupcakes_gluten_free : ℕ)
    (H1 : total_cupcakes = 80)
    (H2 : vegan_cupcakes = 24)
    (H3 : non_vegan_gluten = 28)
    (H4 : cupcakes_gluten_free = vegan_cupcakes / 2) :
    (cupcakes_gluten_free : ℚ) / (total_cupcakes : ℚ) = 3 / 20 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ratio_gluten_free_l932_93236


namespace NUMINAMATH_GPT_area_of_rectangle_inscribed_in_triangle_l932_93224

theorem area_of_rectangle_inscribed_in_triangle :
  ∀ (E F G A B C D : ℝ) (EG altitude_ABCD : ℝ),
    E < F ∧ F < G ∧ A < B ∧ B < C ∧ C < D ∧ A < D ∧ D < G ∧ A < G ∧
    EG = 10 ∧ 
    altitude_ABCD = 7 ∧ 
    B = C ∧ 
    A + D = EG ∧ 
    A + 2 * B = EG →
    ((A * B) = (1225 / 72)) :=
by
  intros E F G A B C D EG altitude_ABCD
  intro h
  sorry

end NUMINAMATH_GPT_area_of_rectangle_inscribed_in_triangle_l932_93224


namespace NUMINAMATH_GPT_marks_per_correct_answer_l932_93235

-- Definitions based on the conditions
def total_questions : ℕ := 60
def total_marks : ℕ := 160
def correct_questions : ℕ := 44
def wrong_mark_loss : ℕ := 1

-- The number of correct answers multiplies the marks per correct answer,
-- minus the loss from wrong answers, equals the total marks.
theorem marks_per_correct_answer (x : ℕ) :
  correct_questions * x - (total_questions - correct_questions) * wrong_mark_loss = total_marks → x = 4 := by
sorry

end NUMINAMATH_GPT_marks_per_correct_answer_l932_93235


namespace NUMINAMATH_GPT_imaginary_part_is_neg_two_l932_93269

open Complex

noncomputable def imaginary_part_of_square : ℂ := (1 - I)^2

theorem imaginary_part_is_neg_two : imaginary_part_of_square.im = -2 := by
  sorry

end NUMINAMATH_GPT_imaginary_part_is_neg_two_l932_93269


namespace NUMINAMATH_GPT_greatest_two_digit_product_is_12_l932_93206

theorem greatest_two_digit_product_is_12 : 
  ∃ (n : ℕ), (∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12 ∧ 10 ≤ n ∧ n < 100) ∧ 
              ∀ (m : ℕ), (∃ (e1 e2 : ℕ), m = 10 * e1 + e2 ∧ e1 * e2 = 12 ∧ 10 ≤ m ∧ m < 100) → m ≤ n :=
sorry

end NUMINAMATH_GPT_greatest_two_digit_product_is_12_l932_93206


namespace NUMINAMATH_GPT_largest_value_a_plus_b_plus_c_l932_93227

open Nat
open Function

def sum_of_digits (n : ℕ) : ℕ :=
  (digits 10 n).sum

theorem largest_value_a_plus_b_plus_c :
  ∃ (a b c : ℕ),
    10 ≤ a ∧ a < 100 ∧
    100 ≤ b ∧ b < 1000 ∧
    1000 ≤ c ∧ c < 10000 ∧
    sum_of_digits (a + b) = 2 ∧
    sum_of_digits (b + c) = 2 ∧
    (a + b + c = 10199) := sorry

end NUMINAMATH_GPT_largest_value_a_plus_b_plus_c_l932_93227


namespace NUMINAMATH_GPT_probability_one_defective_l932_93276

theorem probability_one_defective (g d : Nat) (h1 : g = 3) (h2 : d = 1) : 
  let total_combinations := (g + d).choose 2
  let favorable_outcomes := g * d
  favorable_outcomes / total_combinations = 1 / 2 := by
sorry

end NUMINAMATH_GPT_probability_one_defective_l932_93276


namespace NUMINAMATH_GPT_find_length_of_polaroid_l932_93247

theorem find_length_of_polaroid 
  (C : ℝ) (W : ℝ) (L : ℝ)
  (hC : C = 40) (hW : W = 8) 
  (hFormula : C = 2 * (L + W)) : 
  L = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_length_of_polaroid_l932_93247


namespace NUMINAMATH_GPT_solve_system_of_equations_l932_93299

theorem solve_system_of_equations:
  (∀ (x y : ℝ), 2 * y - x - 2 * x * y = -1 ∧ 4 * x ^ 2 * y ^ 2 + x ^ 2 + 4 * y ^ 2 - 4 * x * y = 61 →
  (x, y) = (-6, -1/2) ∨ (x, y) = (1, 3) ∨ (x, y) = (1, -5/2) ∨ (x, y) = (5, -1/2)) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l932_93299


namespace NUMINAMATH_GPT_compare_neg_rational_l932_93219

def neg_one_third : ℚ := -1 / 3
def neg_one_half : ℚ := -1 / 2

theorem compare_neg_rational : neg_one_third > neg_one_half :=
by sorry

end NUMINAMATH_GPT_compare_neg_rational_l932_93219


namespace NUMINAMATH_GPT_line_perpendicular_to_plane_l932_93248

-- Define a structure for vectors in 3D
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define line l with the given direction vector
def direction_vector_l : Vector3D := ⟨1, -1, -2⟩

-- Define plane α with the given normal vector
def normal_vector_alpha : Vector3D := ⟨2, -2, -4⟩

-- Prove that line l is perpendicular to plane α
theorem line_perpendicular_to_plane :
  let a := direction_vector_l
  let b := normal_vector_alpha
  (b.x = 2 * a.x) ∧ (b.y = 2 * a.y) ∧ (b.z = 2 * a.z) → 
  (a.x * b.x + a.y * b.y + a.z * b.z = 0) :=
by
  intro a b h
  sorry

end NUMINAMATH_GPT_line_perpendicular_to_plane_l932_93248


namespace NUMINAMATH_GPT_train_length_is_correct_l932_93260

noncomputable def train_length (speed_kmph : ℝ) (crossing_time_s : ℝ) (platform_length_m : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  let total_distance := speed_mps * crossing_time_s
  total_distance - platform_length_m

theorem train_length_is_correct :
  train_length 60 14.998800095992321 150 = 100 := by
  sorry

end NUMINAMATH_GPT_train_length_is_correct_l932_93260


namespace NUMINAMATH_GPT_proof_of_independence_l932_93288

/-- A line passing through the plane of two parallel lines and intersecting one of them also intersects the other. -/
def independent_of_parallel_postulate (statement : String) : Prop :=
  statement = "A line passing through the plane of two parallel lines and intersecting one of them also intersects the other."

theorem proof_of_independence :
  independent_of_parallel_postulate "A line passing through the plane of two parallel lines and intersecting one of them also intersects the other." :=
sorry

end NUMINAMATH_GPT_proof_of_independence_l932_93288


namespace NUMINAMATH_GPT_divisible_by_9_l932_93259

-- Definition of the sum of digits function S
def sum_of_digits (n : ℕ) : ℕ := sorry  -- Assume we have a function that sums the digits of n

theorem divisible_by_9 (a : ℕ) (h₁ : sum_of_digits a = sum_of_digits (2 * a)) 
  (h₂ : a % 9 = sum_of_digits a % 9) (h₃ : (2 * a) % 9 = sum_of_digits (2 * a) % 9) : 
  a % 9 = 0 :=
by
  sorry

end NUMINAMATH_GPT_divisible_by_9_l932_93259


namespace NUMINAMATH_GPT_problem_I_problem_II_l932_93272

open Set

-- Definitions of the sets A and B, and their intersections would be needed
def A := {x : ℝ | x ≤ 1 ∨ x ≥ 2}
def B (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ 3 * a}

-- (I) When a = 1, find A ∩ B
theorem problem_I : A ∩ (B 1) = {x : ℝ | (2 ≤ x ∧ x ≤ 3) ∨ x = 1} := by
  sorry

-- (II) When A ∩ B = B, find the range of a
theorem problem_II : {a : ℝ | a > 0 ∧ ∀ x, x ∈ B a → x ∈ A} = {a : ℝ | (0 < a ∧ a ≤ 1 / 3) ∨ a ≥ 2} := by
  sorry

end NUMINAMATH_GPT_problem_I_problem_II_l932_93272


namespace NUMINAMATH_GPT_halloween_candy_l932_93279

theorem halloween_candy (katie_candy : ℕ) (sister_candy : ℕ) (remaining_candy : ℕ) (total_candy : ℕ) (eaten_candy : ℕ)
  (h1 : katie_candy = 10) 
  (h2 : sister_candy = 6) 
  (h3 : remaining_candy = 7) 
  (h4 : total_candy = katie_candy + sister_candy) 
  (h5 : eaten_candy = total_candy - remaining_candy) : 
  eaten_candy = 9 :=
by sorry

end NUMINAMATH_GPT_halloween_candy_l932_93279


namespace NUMINAMATH_GPT_right_triangle_other_side_l932_93231

theorem right_triangle_other_side (a b c : ℕ) (h : a^2 + b^2 = c^2) (h_c : c = 17) (h_a : a = 15) : b = 8 := 
by
  sorry

end NUMINAMATH_GPT_right_triangle_other_side_l932_93231


namespace NUMINAMATH_GPT_quadratic_inequality_min_value_l932_93245

noncomputable def min_value (a b: ℝ) : ℝ := 2 * a^2 + b^2

theorem quadratic_inequality_min_value
  (a b: ℝ) (hx: ∀ x : ℝ, a * x^2 + 2 * x + b ≥ 0)
  (x0: ℝ) (hx0: a * x0^2 + 2 * x0 + b = 0) :
  a > b → min_value a b = 2 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_quadratic_inequality_min_value_l932_93245
