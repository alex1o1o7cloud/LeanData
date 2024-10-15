import Mathlib

namespace NUMINAMATH_GPT_polynomial_value_at_2_l1383_138324

def f (x : ℕ) : ℕ := 8 * x^7 + 5 * x^6 + 3 * x^4 + 2 * x + 1

theorem polynomial_value_at_2 : f 2 = 1397 := by
  sorry

end NUMINAMATH_GPT_polynomial_value_at_2_l1383_138324


namespace NUMINAMATH_GPT_ratio_initial_to_doubled_l1383_138321

theorem ratio_initial_to_doubled (x : ℕ) (h : 3 * (2 * x + 9) = 63) : x / (2 * x) = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_ratio_initial_to_doubled_l1383_138321


namespace NUMINAMATH_GPT_hyperbola_eccentricity_is_sqrt2_l1383_138301

noncomputable def hyperbola_eccentricity (a b : ℝ) (hyp1 : a > 0) (hyp2 : b > 0) 
(hyp3 : b = a) : ℝ :=
    let c := Real.sqrt (2) * a
    c / a

theorem hyperbola_eccentricity_is_sqrt2 
(a b : ℝ) (hyp1 : a > 0) (hyp2 : b > 0) (hyp3 : b = a) :
hyperbola_eccentricity a b hyp1 hyp2 hyp3 = Real.sqrt 2 := sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_is_sqrt2_l1383_138301


namespace NUMINAMATH_GPT_vectors_parallel_l1383_138331

theorem vectors_parallel (m : ℝ) : 
    (∃ k : ℝ, (m, 4) = (k * 5, k * -2)) → m = -10 := 
by
  sorry

end NUMINAMATH_GPT_vectors_parallel_l1383_138331


namespace NUMINAMATH_GPT_ben_has_10_fewer_stickers_than_ryan_l1383_138376

theorem ben_has_10_fewer_stickers_than_ryan :
  ∀ (Karl_stickers Ryan_stickers Ben_stickers total_stickers : ℕ),
    Karl_stickers = 25 →
    Ryan_stickers = Karl_stickers + 20 →
    total_stickers = Karl_stickers + Ryan_stickers + Ben_stickers →
    total_stickers = 105 →
    (Ryan_stickers - Ben_stickers) = 10 :=
by
  intros Karl_stickers Ryan_stickers Ben_stickers total_stickers h1 h2 h3 h4
  -- Conditions mentioned in a)
  exact sorry

end NUMINAMATH_GPT_ben_has_10_fewer_stickers_than_ryan_l1383_138376


namespace NUMINAMATH_GPT_find_balls_l1383_138309

theorem find_balls (x y : ℕ) (h1 : (x + y : ℚ) / (x + y + 18) = (x + 18) / (x + y + 18) - 1 / 15)
                   (h2 : (y + 18 : ℚ) / (x + y + 18) = (x + 18) / (x + y + 18) * 11 / 10) :
  x = 12 ∧ y = 15 :=
sorry

end NUMINAMATH_GPT_find_balls_l1383_138309


namespace NUMINAMATH_GPT_exists_circle_touching_given_circles_and_line_l1383_138373

-- Define the given radii
def r1 := 1
def r2 := 3
def r3 := 4

-- Prove that there exists a circle with a specific radius touching the given circles and line AB
theorem exists_circle_touching_given_circles_and_line (x : ℝ) :
  ∃ (r : ℝ), r > 0 ∧ (r + r1) = x ∧ (r + r2) = x ∧ (r + r3) = x :=
sorry

end NUMINAMATH_GPT_exists_circle_touching_given_circles_and_line_l1383_138373


namespace NUMINAMATH_GPT_no_integer_solution_k_range_l1383_138391

theorem no_integer_solution_k_range (k : ℝ) :
  (∀ x : ℤ, ¬ ((k * x - k^2 - 4) * (x - 4) < 0)) → (1 ≤ k ∧ k ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solution_k_range_l1383_138391


namespace NUMINAMATH_GPT_proper_subset_A_B_l1383_138350

theorem proper_subset_A_B (a : ℝ) : 
  (∀ x, 1 < x ∧ x < 2 → x < a) ∧ (∃ b, b < a ∧ ¬(1 < b ∧ b < 2)) ↔ 2 ≤ a :=
by
  sorry

end NUMINAMATH_GPT_proper_subset_A_B_l1383_138350


namespace NUMINAMATH_GPT_sequence_inequality_l1383_138363
open Nat

variable (a : ℕ → ℝ)

noncomputable def conditions := 
  (a 1 ≥ 1) ∧ (∀ k : ℕ, a (k + 1) - a k ≥ 1)

theorem sequence_inequality (h : conditions a) : 
  ∀ n : ℕ, a (n + 1) ≥ n + 1 :=
sorry

end NUMINAMATH_GPT_sequence_inequality_l1383_138363


namespace NUMINAMATH_GPT_chess_pieces_present_l1383_138342

theorem chess_pieces_present (total_pieces : ℕ) (missing_pieces : ℕ) (h1 : total_pieces = 32) (h2 : missing_pieces = 4) : (total_pieces - missing_pieces) = 28 := 
by sorry

end NUMINAMATH_GPT_chess_pieces_present_l1383_138342


namespace NUMINAMATH_GPT_temperature_at_4km_l1383_138365

theorem temperature_at_4km (ground_temp : ℤ) (drop_rate : ℤ) (altitude : ℕ) (ΔT : ℤ) : 
  ground_temp = 15 ∧ drop_rate = -5 ∧ ΔT = altitude * drop_rate ∧ altitude = 4 → 
  ground_temp + ΔT = -5 :=
by
  sorry

end NUMINAMATH_GPT_temperature_at_4km_l1383_138365


namespace NUMINAMATH_GPT_polygon_interior_angle_sum_l1383_138332

theorem polygon_interior_angle_sum (n : ℕ) (hn : 3 ≤ n) :
  (n - 2) * 180 + 180 = 2007 → n = 13 := by
  sorry

end NUMINAMATH_GPT_polygon_interior_angle_sum_l1383_138332


namespace NUMINAMATH_GPT_percent_to_decimal_l1383_138387

theorem percent_to_decimal : (2 : ℝ) / 100 = 0.02 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_percent_to_decimal_l1383_138387


namespace NUMINAMATH_GPT_Sally_bought_20_pokemon_cards_l1383_138397

theorem Sally_bought_20_pokemon_cards
  (initial_cards : ℕ)
  (cards_from_dan : ℕ)
  (total_cards : ℕ)
  (bought_cards : ℕ)
  (h1 : initial_cards = 27)
  (h2 : cards_from_dan = 41)
  (h3 : total_cards = 88)
  (h4 : total_cards = initial_cards + cards_from_dan + bought_cards) :
  bought_cards = 20 := 
by
  sorry

end NUMINAMATH_GPT_Sally_bought_20_pokemon_cards_l1383_138397


namespace NUMINAMATH_GPT_total_books_sold_l1383_138360

theorem total_books_sold (tuesday_books wednesday_books thursday_books : Nat) 
  (h1 : tuesday_books = 7) 
  (h2 : wednesday_books = 3 * tuesday_books) 
  (h3 : thursday_books = 3 * wednesday_books) : 
  tuesday_books + wednesday_books + thursday_books = 91 := 
by 
  sorry

end NUMINAMATH_GPT_total_books_sold_l1383_138360


namespace NUMINAMATH_GPT_trapezoid_perimeter_and_area_l1383_138399

theorem trapezoid_perimeter_and_area (PQ RS QR PS : ℝ) (hPQ_RS : PQ = RS)
  (hPQ_RS_positive : PQ > 0) (hQR : QR = 10) (hPS : PS = 20) (height : ℝ)
  (h_height : height = 5) :
  PQ = 5 * Real.sqrt 2 ∧
  QR = 10 ∧
  PS = 20 ∧ 
  height = 5 ∧
  (PQ + QR + RS + PS = 30 + 10 * Real.sqrt 2) ∧
  (1 / 2 * (QR + PS) * height = 75) :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_perimeter_and_area_l1383_138399


namespace NUMINAMATH_GPT_percentage_failed_both_l1383_138357

theorem percentage_failed_both (p_hindi p_english p_pass_both x : ℝ)
  (h₁ : p_hindi = 0.25)
  (h₂ : p_english = 0.5)
  (h₃ : p_pass_both = 0.5)
  (h₄ : (p_hindi + p_english - x) = 0.5) : 
  x = 0.25 := 
sorry

end NUMINAMATH_GPT_percentage_failed_both_l1383_138357


namespace NUMINAMATH_GPT_escalator_length_l1383_138302

theorem escalator_length
  (escalator_speed : ℝ)
  (person_speed : ℝ)
  (time_taken : ℝ)
  (combined_speed := escalator_speed + person_speed)
  (distance := combined_speed * time_taken) :
  escalator_speed = 10 → person_speed = 4 → time_taken = 8 → distance = 112 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end NUMINAMATH_GPT_escalator_length_l1383_138302


namespace NUMINAMATH_GPT_symmetric_points_sum_l1383_138327

theorem symmetric_points_sum (a b : ℝ) (P Q : ℝ × ℝ) 
    (hP : P = (3, a)) (hQ : Q = (b, 2))
    (symm : P = (-Q.1, Q.2)) : a + b = -1 := by
  sorry

end NUMINAMATH_GPT_symmetric_points_sum_l1383_138327


namespace NUMINAMATH_GPT_rd_expense_necessary_for_increase_l1383_138323

theorem rd_expense_necessary_for_increase :
  ∀ (R_and_D_t : ℝ) (delta_APL_t1 : ℝ),
  R_and_D_t = 3289.31 → delta_APL_t1 = 1.55 →
  R_and_D_t / delta_APL_t1 = 2122 := 
by
  intros R_and_D_t delta_APL_t1 hR hD
  rw [hR, hD]
  norm_num
  sorry

end NUMINAMATH_GPT_rd_expense_necessary_for_increase_l1383_138323


namespace NUMINAMATH_GPT_percentage_of_games_lost_l1383_138362

theorem percentage_of_games_lost (games_won games_lost games_tied total_games : ℕ)
  (h_ratio : 5 * games_lost = 3 * games_won)
  (h_tied : games_tied * 5 = total_games) :
  (games_lost * 10 / total_games) = 3 :=
by sorry

end NUMINAMATH_GPT_percentage_of_games_lost_l1383_138362


namespace NUMINAMATH_GPT_total_amount_paid_l1383_138371

-- Definitions from the conditions
def quantity_grapes : ℕ := 8
def rate_grapes : ℕ := 70
def quantity_mangoes : ℕ := 9
def rate_mangoes : ℕ := 60

-- Main statement to prove
theorem total_amount_paid :
  (quantity_grapes * rate_grapes) + (quantity_mangoes * rate_mangoes) = 1100 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_paid_l1383_138371


namespace NUMINAMATH_GPT_solve_expression_l1383_138356

theorem solve_expression :
  (27 ^ (2 / 3) - 2 ^ (Real.log 3 / Real.log 2) * (Real.logb 2 (1 / 8)) +
    Real.logb 10 4 + Real.logb 10 25 = 20) :=
by
  sorry

end NUMINAMATH_GPT_solve_expression_l1383_138356


namespace NUMINAMATH_GPT_calculate_expression_is_correct_l1383_138320

noncomputable def calculate_expression : ℝ :=
  -(-2) + 2 * Real.cos (Real.pi / 3) + (-1 / 8)⁻¹ + (Real.pi - 3.14) ^ 0

theorem calculate_expression_is_correct :
  calculate_expression = -4 :=
by
  -- the conditions as definitions
  have h1 : Real.cos (Real.pi / 3) = 1 / 2 := by sorry
  have h2 : (Real.pi - 3.14) ^ 0 = 1 := by sorry
  -- use these conditions to prove the main statement
  sorry

end NUMINAMATH_GPT_calculate_expression_is_correct_l1383_138320


namespace NUMINAMATH_GPT_property_tax_increase_is_800_l1383_138389

-- Define conditions as constants
def tax_rate : ℝ := 0.10
def initial_value : ℝ := 20000
def new_value : ℝ := 28000

-- Define the increase in property tax
def tax_increase : ℝ := (new_value * tax_rate) - (initial_value * tax_rate)

-- Statement to be proved
theorem property_tax_increase_is_800 : tax_increase = 800 :=
by
  sorry

end NUMINAMATH_GPT_property_tax_increase_is_800_l1383_138389


namespace NUMINAMATH_GPT_log_inequality_sqrt_inequality_l1383_138311

-- Proof problem for part (1)
theorem log_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.log ((a + b) / 2) ≥ (Real.log a + Real.log b) / 2 :=
sorry

-- Proof problem for part (2)
theorem sqrt_inequality :
  Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_log_inequality_sqrt_inequality_l1383_138311


namespace NUMINAMATH_GPT_volunteers_meet_again_in_360_days_l1383_138378

theorem volunteers_meet_again_in_360_days :
  let Sasha := 5
  let Leo := 8
  let Uma := 9
  let Kim := 10
  Nat.lcm Sasha (Nat.lcm Leo (Nat.lcm Uma Kim)) = 360 :=
by
  sorry

end NUMINAMATH_GPT_volunteers_meet_again_in_360_days_l1383_138378


namespace NUMINAMATH_GPT_sum_of_edges_l1383_138300

theorem sum_of_edges (a r : ℝ) 
  (h_volume : (a^3 = 512))
  (h_surface_area : (2 * (a^2 / r + a^2 + a^2 * r) = 384))
  (h_geometric_progression : true) :
  (4 * ((a / r) + a + (a * r)) = 96) :=
by
  -- It is only necessary to provide the theorem statement
  sorry

end NUMINAMATH_GPT_sum_of_edges_l1383_138300


namespace NUMINAMATH_GPT_gum_lcm_l1383_138394

theorem gum_lcm (strawberry blueberry cherry : ℕ) (h₁ : strawberry = 6) (h₂ : blueberry = 5) (h₃ : cherry = 8) :
  Nat.lcm (Nat.lcm strawberry blueberry) cherry = 120 :=
by
  rw [h₁, h₂, h₃]
  -- LCM(6, 5, 8) = LCM(LCM(6, 5), 8)
  sorry

end NUMINAMATH_GPT_gum_lcm_l1383_138394


namespace NUMINAMATH_GPT_geometric_sequence_conditions_l1383_138379

variable (a : ℕ → ℝ) (q : ℝ)

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_conditions (a : ℕ → ℝ) (q : ℝ)
  (h1 : geometric_sequence a q)
  (h2 : -1 < q)
  (h3 : q < 0) :
  (∀ n, a n * a (n + 1) < 0) ∧ (∀ n, |a n| > |a (n + 1)|) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_conditions_l1383_138379


namespace NUMINAMATH_GPT_parker_net_income_after_taxes_l1383_138319

noncomputable def parker_income : Real := sorry

theorem parker_net_income_after_taxes :
  let daily_pay := 63
  let hours_per_day := 8
  let hourly_rate := daily_pay / hours_per_day
  let overtime_rate := 1.5 * hourly_rate
  let overtime_hours_per_weekend_day := 3
  let weekends_in_6_weeks := 6
  let days_per_week := 7
  let total_days_in_6_weeks := days_per_week * weekends_in_6_weeks
  let regular_earnings := daily_pay * total_days_in_6_weeks
  let total_overtime_earnings := overtime_rate * overtime_hours_per_weekend_day * 2 * weekends_in_6_weeks
  let gross_income := regular_earnings + total_overtime_earnings
  let tax_rate := 0.1
  let net_income_after_taxes := gross_income * (1 - tax_rate)
  net_income_after_taxes = 2764.125 := by sorry

end NUMINAMATH_GPT_parker_net_income_after_taxes_l1383_138319


namespace NUMINAMATH_GPT_train_length_l1383_138307

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (h_speed : speed_kmph = 36) (h_time : time_sec = 6.5) : 
  (speed_kmph * 1000 / 3600) * time_sec = 65 := 
by {
  -- Placeholder for proof
  sorry
}

end NUMINAMATH_GPT_train_length_l1383_138307


namespace NUMINAMATH_GPT_manuscript_typing_cost_l1383_138336

theorem manuscript_typing_cost :
  let total_pages := 100
  let revised_once_pages := 30
  let revised_twice_pages := 20
  let cost_first_time := 10
  let cost_revision := 5
  let cost_first_typing := total_pages * cost_first_time
  let cost_revisions_once := revised_once_pages * cost_revision
  let cost_revisions_twice := revised_twice_pages * (2 * cost_revision)
  cost_first_typing + cost_revisions_once + cost_revisions_twice = 1350 :=
by
  let total_pages := 100
  let revised_once_pages := 30
  let revised_twice_pages := 20
  let cost_first_time := 10
  let cost_revision := 5
  let cost_first_typing := total_pages * cost_first_time
  let cost_revisions_once := revised_once_pages * cost_revision
  let cost_revisions_twice := revised_twice_pages * (2 * cost_revision)
  have : cost_first_typing + cost_revisions_once + cost_revisions_twice = 1350 := sorry
  exact this

end NUMINAMATH_GPT_manuscript_typing_cost_l1383_138336


namespace NUMINAMATH_GPT_school_boys_number_l1383_138384

theorem school_boys_number (B G : ℕ) (h1 : B / G = 5 / 13) (h2 : G = B + 80) : B = 50 :=
by
  sorry

end NUMINAMATH_GPT_school_boys_number_l1383_138384


namespace NUMINAMATH_GPT_largest_n_digit_number_divisible_by_61_correct_l1383_138333

def largest_n_digit_number (n : ℕ) : ℕ :=
10^n - 1

def largest_n_digit_number_divisible_by_61 (n : ℕ) : ℕ :=
largest_n_digit_number n - (largest_n_digit_number n % 61)

theorem largest_n_digit_number_divisible_by_61_correct (n : ℕ) :
  ∃ k : ℕ, largest_n_digit_number_divisible_by_61 n = 61 * k :=
by
  sorry

end NUMINAMATH_GPT_largest_n_digit_number_divisible_by_61_correct_l1383_138333


namespace NUMINAMATH_GPT_multiple_of_three_l1383_138344

theorem multiple_of_three (a b : ℤ) : ∃ k : ℤ, (a + b = 3 * k) ∨ (ab = 3 * k) ∨ (a - b = 3 * k) :=
sorry

end NUMINAMATH_GPT_multiple_of_three_l1383_138344


namespace NUMINAMATH_GPT_reading_time_difference_in_minutes_l1383_138313

noncomputable def xanthia_reading_speed : ℝ := 120 -- pages per hour
noncomputable def molly_reading_speed : ℝ := 60 -- pages per hour
noncomputable def book_length : ℝ := 360 -- pages

theorem reading_time_difference_in_minutes :
  let time_for_xanthia := book_length / xanthia_reading_speed
  let time_for_molly := book_length / molly_reading_speed
  let difference_in_hours := time_for_molly - time_for_xanthia
  difference_in_hours * 60 = 180 :=
by
  sorry

end NUMINAMATH_GPT_reading_time_difference_in_minutes_l1383_138313


namespace NUMINAMATH_GPT_greatest_possible_value_l1383_138366

theorem greatest_possible_value (A B C D : ℕ) 
    (h1 : A + B + C + D = 200) 
    (h2 : A + B = 70) 
    (h3 : 0 < A) 
    (h4 : 0 < B) 
    (h5 : 0 < C) 
    (h6 : 0 < D) : 
    C ≤ 129 := 
sorry

end NUMINAMATH_GPT_greatest_possible_value_l1383_138366


namespace NUMINAMATH_GPT_repeating_block_digits_l1383_138306

theorem repeating_block_digits (n d : ℕ) (h1 : n = 3) (h2 : d = 11) : 
  (∃ repeating_block_length, repeating_block_length = 2) := by
  sorry

end NUMINAMATH_GPT_repeating_block_digits_l1383_138306


namespace NUMINAMATH_GPT_speed_of_man_in_still_water_l1383_138303

theorem speed_of_man_in_still_water (v_m v_s : ℝ) (h1 : v_m + v_s = 18) (h2 : v_m - v_s = 13) : v_m = 15.5 :=
by {
  -- Proof is not required as per the instructions
  sorry
}

end NUMINAMATH_GPT_speed_of_man_in_still_water_l1383_138303


namespace NUMINAMATH_GPT_sin_cos_inequality_for_any_x_l1383_138361

noncomputable def largest_valid_n : ℕ := 8

theorem sin_cos_inequality_for_any_x (n : ℕ) (h : n = largest_valid_n) :
  ∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 1 / n :=
sorry

end NUMINAMATH_GPT_sin_cos_inequality_for_any_x_l1383_138361


namespace NUMINAMATH_GPT_operation_not_equal_33_l1383_138368

-- Definitions for the given conditions
def single_digit_positive_integer (n : ℤ) : Prop := 1 ≤ n ∧ n ≤ 9
def x (a : ℤ) := 1 / 5 * a
def z (b : ℤ) := 1 / 5 * b

-- The theorem to show that the operations involving x and z cannot equal 33
theorem operation_not_equal_33 (a b : ℤ) (ha : single_digit_positive_integer a) 
(hb : single_digit_positive_integer b) : 
((x a - z b = 33) ∨ (z b - x a = 33) ∨ (x a / z b = 33) ∨ (z b / x a = 33)) → false :=
by
  sorry

end NUMINAMATH_GPT_operation_not_equal_33_l1383_138368


namespace NUMINAMATH_GPT_find_triple_l1383_138322
-- Import necessary libraries

-- Define the required predicates and conditions
def satisfies_conditions (x y z : ℕ) : Prop :=
  x ≤ y ∧ y ≤ z ∧ x^3 * (y^3 + z^3) = 2012 * (x * y * z + 2)

-- The main theorem statement
theorem find_triple : 
  ∀ (x y z : ℕ), satisfies_conditions x y z → (x, y, z) = (2, 251, 252) :=
by
  sorry

end NUMINAMATH_GPT_find_triple_l1383_138322


namespace NUMINAMATH_GPT_ellipse_problem_l1383_138314

noncomputable def point_coordinates (x y b : ℝ) : Prop :=
  x = 1 ∧ y = 1 ∧ (4 * x^2 = 4) ∧ (4 * b^2 / (4 + b^2) = 1)

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  (Real.sqrt (a^2 - b^2)) / a

theorem ellipse_problem (b : ℝ) (h₁ : 4 * b^2 / (4 + b^2) = 1) :
  ∃ x y, point_coordinates x y b 
  ∧ eccentricity 2 b = Real.sqrt 6 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_ellipse_problem_l1383_138314


namespace NUMINAMATH_GPT_ratio_of_cows_to_bulls_l1383_138349

-- Define the total number of cattle
def total_cattle := 555

-- Define the number of bulls
def number_of_bulls := 405

-- Compute the number of cows
def number_of_cows := total_cattle - number_of_bulls

-- Define the expected ratio of cows to bulls
def expected_ratio_cows_to_bulls := (10, 27)

-- Prove that the ratio of cows to bulls is equal to the expected ratio
theorem ratio_of_cows_to_bulls : 
  (number_of_cows / (gcd number_of_cows number_of_bulls), number_of_bulls / (gcd number_of_cows number_of_bulls)) = expected_ratio_cows_to_bulls :=
sorry

end NUMINAMATH_GPT_ratio_of_cows_to_bulls_l1383_138349


namespace NUMINAMATH_GPT_classroom_not_1_hectare_l1383_138304

def hectare_in_sq_meters : ℕ := 10000
def classroom_area_approx : ℕ := 60

theorem classroom_not_1_hectare : ¬ (classroom_area_approx = hectare_in_sq_meters) :=
by 
  sorry

end NUMINAMATH_GPT_classroom_not_1_hectare_l1383_138304


namespace NUMINAMATH_GPT_expression_eq_49_l1383_138355

theorem expression_eq_49 (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 := 
by 
  sorry

end NUMINAMATH_GPT_expression_eq_49_l1383_138355


namespace NUMINAMATH_GPT_find_c_if_lines_parallel_l1383_138343

theorem find_c_if_lines_parallel (c : ℝ) : 
  (∀ x : ℝ, 5 * x - 3 = (3 * c) * x + 1) → 
  c = 5 / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_c_if_lines_parallel_l1383_138343


namespace NUMINAMATH_GPT_exponential_inequality_l1383_138352

theorem exponential_inequality (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a^m < a^n) : ¬ (m < n) := 
sorry

end NUMINAMATH_GPT_exponential_inequality_l1383_138352


namespace NUMINAMATH_GPT_smallest_solution_is_neg_sqrt_13_l1383_138381

noncomputable def smallest_solution (x : ℝ) : Prop :=
  x^4 - 26 * x^2 + 169 = 0 ∧ ∀ y : ℝ, y^4 - 26 * y^2 + 169 = 0 → x ≤ y

theorem smallest_solution_is_neg_sqrt_13 :
  smallest_solution (-Real.sqrt 13) :=
by
  sorry

end NUMINAMATH_GPT_smallest_solution_is_neg_sqrt_13_l1383_138381


namespace NUMINAMATH_GPT_largest_4_digit_integer_congruent_to_25_mod_26_l1383_138374

theorem largest_4_digit_integer_congruent_to_25_mod_26 : ∃ x : ℕ, x < 10000 ∧ x ≥ 1000 ∧ x % 26 = 25 ∧ ∀ y : ℕ, y < 10000 ∧ y ≥ 1000 ∧ y % 26 = 25 → y ≤ x := by
  sorry

end NUMINAMATH_GPT_largest_4_digit_integer_congruent_to_25_mod_26_l1383_138374


namespace NUMINAMATH_GPT_sequence_bound_100_l1383_138316

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n ≥ 2, a n = a (n - 1) + 1 / a (n - 1)

theorem sequence_bound_100 (a : ℕ → ℝ) (h : seq a) : 
  14 < a 100 ∧ a 100 < 18 := 
sorry

end NUMINAMATH_GPT_sequence_bound_100_l1383_138316


namespace NUMINAMATH_GPT_find_female_employees_l1383_138358

-- Definitions from conditions
def total_employees (E : ℕ) := True
def female_employees (F : ℕ) := True
def male_employees (M : ℕ) := True
def female_managers (F_mgrs : ℕ) := F_mgrs = 280
def fraction_of_managers : ℚ := 2 / 5
def fraction_of_male_managers : ℚ := 2 / 5

-- Statements as conditions in Lean
def managers_total (E M : ℕ) := (fraction_of_managers * E : ℚ) = (fraction_of_male_managers * M : ℚ) + 280
def employees_total (E F M : ℕ) := E = F + M

-- The proof target
theorem find_female_employees (E F M : ℕ) (F_mgrs : ℕ)
    (h1 : female_managers F_mgrs)
    (h2 : managers_total E M)
    (h3 : employees_total E F M) : F = 700 := by
  sorry

end NUMINAMATH_GPT_find_female_employees_l1383_138358


namespace NUMINAMATH_GPT_green_disks_count_l1383_138312

-- Definitions of the conditions given in the problem
def total_disks : ℕ := 14
def red_disks (g : ℕ) : ℕ := 2 * g
def blue_disks (g : ℕ) : ℕ := g / 2

-- The theorem statement to prove
theorem green_disks_count (g : ℕ) (h : 2 * g + g + g / 2 = total_disks) : g = 4 :=
sorry

end NUMINAMATH_GPT_green_disks_count_l1383_138312


namespace NUMINAMATH_GPT_discount_percentage_l1383_138390

theorem discount_percentage (MP CP SP : ℝ)
  (h1 : CP = 0.64 * MP)
  (h2 : SP = CP * 1.375)
  (gain_percent : 37.5 = (SP - CP) / CP * 100) :
  (MP - SP) / MP * 100 = 12 :=
by
  sorry

end NUMINAMATH_GPT_discount_percentage_l1383_138390


namespace NUMINAMATH_GPT_not_function_of_x_l1383_138334

theorem not_function_of_x : 
  ∃ x : ℝ, ∃ y1 y2 : ℝ, (|y1| = 2 * x ∧ |y2| = 2 * x ∧ y1 ≠ y2) := sorry

end NUMINAMATH_GPT_not_function_of_x_l1383_138334


namespace NUMINAMATH_GPT_number_of_movies_in_series_l1383_138347

variables (watched_movies remaining_movies total_movies : ℕ)

theorem number_of_movies_in_series 
  (h_watched : watched_movies = 4) 
  (h_remaining : remaining_movies = 4) :
  total_movies = watched_movies + remaining_movies :=
by
  sorry

end NUMINAMATH_GPT_number_of_movies_in_series_l1383_138347


namespace NUMINAMATH_GPT_negation_of_proposition_l1383_138339

theorem negation_of_proposition (a b c : ℝ) :
  ¬ (a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) ↔ (a + b + c ≠ 3 → a^2 + b^2 + c^2 < 3) :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l1383_138339


namespace NUMINAMATH_GPT_smallest_integer_represented_as_AA6_and_BB8_l1383_138335

def valid_digit_in_base (d : ℕ) (b : ℕ) : Prop := d < b

theorem smallest_integer_represented_as_AA6_and_BB8 :
  ∃ (n : ℕ) (A B : ℕ),
  valid_digit_in_base A 6 ∧ valid_digit_in_base B 8 ∧ 
  n = 7 * A ∧ n = 9 * B ∧ n = 63 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_represented_as_AA6_and_BB8_l1383_138335


namespace NUMINAMATH_GPT_g_solution_l1383_138318

noncomputable def g : ℝ → ℝ := sorry

axiom g_0 : g 0 = 2
axiom g_functional : ∀ x y : ℝ, g (x * y) = g ((x^2 + y^2) / 2) + (x - y)^2 + x^2

theorem g_solution :
  ∀ x : ℝ, g x = 2 - 2 * x := sorry

end NUMINAMATH_GPT_g_solution_l1383_138318


namespace NUMINAMATH_GPT_smallest_non_factor_product_l1383_138370

open Nat

def is_factor (n d : ℕ) := d > 0 ∧ n % d = 0

theorem smallest_non_factor_product (a b : ℕ) (h1 : a ≠ b) (h2 : is_factor 48 a) (h3 : is_factor 48 b) (h4 : ¬ is_factor 48 (a * b)) : a * b = 18 :=
by
  sorry

end NUMINAMATH_GPT_smallest_non_factor_product_l1383_138370


namespace NUMINAMATH_GPT_intersect_of_given_circles_l1383_138369

noncomputable def circle_center (a b c : ℝ) : ℝ × ℝ :=
  let x := -a / 2
  let y := -b / 2
  (x, y)

noncomputable def radius_squared (a b c : ℝ) : ℝ :=
  (a / 2) ^ 2 + (b / 2) ^ 2 - c

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def circles_intersect (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  let center1 := circle_center a1 b1 c1
  let center2 := circle_center a2 b2 c2
  let r1 := Real.sqrt (radius_squared a1 b1 c1)
  let r2 := Real.sqrt (radius_squared a2 b2 c2)
  let d := distance center1 center2
  r1 - r2 < d ∧ d < r1 + r2

theorem intersect_of_given_circles :
  circles_intersect 4 3 2 2 3 1 :=
sorry

end NUMINAMATH_GPT_intersect_of_given_circles_l1383_138369


namespace NUMINAMATH_GPT_platform_length_correct_l1383_138308

noncomputable def platform_length : ℝ :=
  let T := 180
  let v_kmph := 72
  let t := 20
  let v_ms := v_kmph * 1000 / 3600
  let total_distance := v_ms * t
  total_distance - T

theorem platform_length_correct : platform_length = 220 := by
  sorry

end NUMINAMATH_GPT_platform_length_correct_l1383_138308


namespace NUMINAMATH_GPT_proof_problem_solution_l1383_138395

noncomputable def proof_problem (a b c d : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ (a * b + b * c + c * d + d * a = 1) →
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1 / 3)

theorem proof_problem_solution (a b c d : ℝ) : proof_problem a b c d :=
  sorry

end NUMINAMATH_GPT_proof_problem_solution_l1383_138395


namespace NUMINAMATH_GPT_challenging_math_problem_l1383_138393

theorem challenging_math_problem :
  ((9^2 + (3^3 - 1) * 4^2) % 6) * Real.sqrt 49 + (15 - 3 * 5) = 35 :=
by
  sorry

end NUMINAMATH_GPT_challenging_math_problem_l1383_138393


namespace NUMINAMATH_GPT_smallest_four_digit_multiple_of_18_l1383_138388

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 18 ∣ n ∧ ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ 18 ∣ m → n ≤ m := by
  use 1008
  sorry

end NUMINAMATH_GPT_smallest_four_digit_multiple_of_18_l1383_138388


namespace NUMINAMATH_GPT_total_annual_gain_l1383_138351

-- Definitions based on given conditions
variable (A B C : Type) [Field ℝ]

-- Assume initial investments and time factors
variable (x : ℝ) (A_share : ℝ := 5000) -- A's share is Rs. 5000

-- Total annual gain to be proven
theorem total_annual_gain (x : ℝ) (A_share B_share C_share Total_Profit : ℝ) :
  A_share = 5000 → 
  B_share = (2 * x) * (6 / 12) → 
  C_share = (3 * x) * (4 / 12) → 
  (A_share / (x * 12)) * Total_Profit = 5000 → -- A's determined share from profit
  Total_Profit = 15000 := 
by 
  sorry

end NUMINAMATH_GPT_total_annual_gain_l1383_138351


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_l1383_138346

noncomputable def problem1 (α : ℝ) (h : Real.tan α = -2) : Real :=
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α)

theorem problem1_solution (α : ℝ) (h : Real.tan α = -2) : problem1 α h = 5 := by
  sorry

noncomputable def problem2 (α : ℝ) (h : Real.tan α = -2) : Real :=
  1 / (Real.sin α * Real.cos α)

theorem problem2_solution (α : ℝ) (h : Real.tan α = -2) : problem2 α h = -5 / 2 := by
  sorry

end NUMINAMATH_GPT_problem1_solution_problem2_solution_l1383_138346


namespace NUMINAMATH_GPT_right_triangle_legs_l1383_138328

theorem right_triangle_legs (c a b : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : ab = c^2 / 4) :
  a = c * (Real.sqrt 6 + Real.sqrt 2) / 4 ∧ b = c * (Real.sqrt 6 - Real.sqrt 2) / 4 := 
sorry

end NUMINAMATH_GPT_right_triangle_legs_l1383_138328


namespace NUMINAMATH_GPT_reduced_price_per_kg_l1383_138375

variables (P P' : ℝ)

-- Given conditions
def condition1 := P' = P / 2
def condition2 := 800 / P' = 800 / P + 5

-- Proof problem statement
theorem reduced_price_per_kg (P P' : ℝ) (h1 : condition1 P P') (h2 : condition2 P P') :
  P' = 80 :=
by
  sorry

end NUMINAMATH_GPT_reduced_price_per_kg_l1383_138375


namespace NUMINAMATH_GPT_maximize_probability_sum_8_l1383_138385

def L : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12]

theorem maximize_probability_sum_8 :
  (∀ x ∈ L, x ≠ 4 → (∃ y ∈ (List.erase L x), y = 8 - x)) ∧ 
  (∀ y ∈ List.erase L 4, ¬(∃ x ∈ List.erase L 4, x + y = 8)) :=
sorry

end NUMINAMATH_GPT_maximize_probability_sum_8_l1383_138385


namespace NUMINAMATH_GPT_sum_of_x_values_l1383_138380

theorem sum_of_x_values (x : ℝ) (h : x ≠ -1) : 
  (∃ x, 3 = (x^3 - 3*x^2 - 4*x)/(x + 1)) →
  (x = 6) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_x_values_l1383_138380


namespace NUMINAMATH_GPT_jessica_age_proof_l1383_138315

-- Definitions based on conditions
def grandmother_age (j : ℚ) : ℚ := 15 * j
def age_difference (g j : ℚ) : Prop := g - j = 60

-- Proposed age of Jessica
def jessica_age : ℚ := 30 / 7

-- Main statement to prove
theorem jessica_age_proof : ∃ j : ℚ, grandmother_age j = 15 * j ∧ age_difference (grandmother_age j) j ∧ j = jessica_age :=
by sorry

end NUMINAMATH_GPT_jessica_age_proof_l1383_138315


namespace NUMINAMATH_GPT_smallest_y_l1383_138364

theorem smallest_y (y : ℝ) :
  (3 * y ^ 2 + 33 * y - 90 = y * (y + 16)) → y = -10 :=
sorry

end NUMINAMATH_GPT_smallest_y_l1383_138364


namespace NUMINAMATH_GPT_response_rate_percentage_l1383_138398

theorem response_rate_percentage (number_of_responses_needed number_of_questionnaires_mailed : ℕ) 
  (h1 : number_of_responses_needed = 300) 
  (h2 : number_of_questionnaires_mailed = 500) : 
  (number_of_responses_needed / number_of_questionnaires_mailed : ℚ) * 100 = 60 :=
by 
  sorry

end NUMINAMATH_GPT_response_rate_percentage_l1383_138398


namespace NUMINAMATH_GPT_jill_commute_time_l1383_138382

theorem jill_commute_time :
  let dave_steps_per_min := 80
  let dave_cm_per_step := 70
  let dave_time_min := 20
  let dave_speed :=
    dave_steps_per_min * dave_cm_per_step
  let dave_distance :=
    dave_speed * dave_time_min
  let jill_steps_per_min := 120
  let jill_cm_per_step := 50
  let jill_speed :=
    jill_steps_per_min * jill_cm_per_step
  let jill_time :=
    dave_distance / jill_speed
  jill_time = 18 + 2 / 3 := by
  sorry

end NUMINAMATH_GPT_jill_commute_time_l1383_138382


namespace NUMINAMATH_GPT_find_principal_amount_l1383_138348

theorem find_principal_amount
  (P r : ℝ) -- P for Principal amount, r for interest rate
  (simple_interest : 800 = P * r / 100 * 2) -- Condition 1: Simple Interest Formula
  (compound_interest : 820 = P * ((1 + r / 100) ^ 2 - 1)) -- Condition 2: Compound Interest Formula
  : P = 8000 := 
sorry

end NUMINAMATH_GPT_find_principal_amount_l1383_138348


namespace NUMINAMATH_GPT_least_positive_integer_x_20y_l1383_138345

theorem least_positive_integer_x_20y (x y : ℤ) (h : Int.gcd x (20 * y) = 4) : 
  ∃ k : ℕ, k > 0 ∧ k * (x + 20 * y) = 4 := 
sorry

end NUMINAMATH_GPT_least_positive_integer_x_20y_l1383_138345


namespace NUMINAMATH_GPT_complex_number_in_first_quadrant_l1383_138383

open Complex

theorem complex_number_in_first_quadrant (z : ℂ) (h : z = 1 / (1 - I)) : 
  z.re > 0 ∧ z.im > 0 :=
by
  sorry

end NUMINAMATH_GPT_complex_number_in_first_quadrant_l1383_138383


namespace NUMINAMATH_GPT_find_side_length_S2_l1383_138317

-- Define the variables and conditions
variables (r s : ℕ)
def is_solution (r s : ℕ) : Prop :=
  2 * r + s = 2160 ∧ 2 * r + 3 * s = 3450

-- Define the problem statement
theorem find_side_length_S2 (r s : ℕ) (h : is_solution r s) : s = 645 :=
sorry

end NUMINAMATH_GPT_find_side_length_S2_l1383_138317


namespace NUMINAMATH_GPT_garden_stone_calculation_l1383_138372

/-- A rectangular garden with dimensions 15m by 2m and patio stones of dimensions 0.5m by 0.5m requires 120 stones to be fully covered -/
theorem garden_stone_calculation :
  let garden_length := 15
  let garden_width := 2
  let stone_length := 0.5
  let stone_width := 0.5
  let area_garden := garden_length * garden_width
  let area_stone := stone_length * stone_width
  let num_stones := area_garden / area_stone
  num_stones = 120 :=
by
  sorry

end NUMINAMATH_GPT_garden_stone_calculation_l1383_138372


namespace NUMINAMATH_GPT_range_of_a_l1383_138325

noncomputable def f (a x : ℝ) : ℝ :=
  x^3 + 3 * a * x^2 + 3 * ((a + 2) * x + 1)

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ deriv (f a) x = 0 ∧ deriv (f a) y = 0) ↔ a < -1 ∨ a > 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1383_138325


namespace NUMINAMATH_GPT_grading_options_count_l1383_138353

theorem grading_options_count :
  (4 ^ 15) = 1073741824 :=
by
  sorry

end NUMINAMATH_GPT_grading_options_count_l1383_138353


namespace NUMINAMATH_GPT_order_of_abc_l1383_138305

theorem order_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h1 : a^2 + b^2 < a^2 + c^2) (h2 : a^2 + c^2 < b^2 + c^2) : a < b ∧ b < c := 
by
  sorry

end NUMINAMATH_GPT_order_of_abc_l1383_138305


namespace NUMINAMATH_GPT_P_at_7_eq_5760_l1383_138396

noncomputable def P (x : ℝ) : ℝ :=
  12 * (x - 1) * (x - 2) * (x - 3)^2 * (x - 6)^4

theorem P_at_7_eq_5760 : P 7 = 5760 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_P_at_7_eq_5760_l1383_138396


namespace NUMINAMATH_GPT_min_boxes_to_eliminate_for_one_third_chance_l1383_138392

-- Define the number of boxes
def total_boxes := 26

-- Define the number of boxes with at least $250,000
def boxes_with_at_least_250k := 6

-- Define the condition for having a 1/3 chance
def one_third_chance (remaining_boxes : ℕ) : Prop :=
  6 / remaining_boxes = 1 / 3

-- Define the target number of boxes to eliminate
def boxes_to_eliminate := total_boxes - 18

theorem min_boxes_to_eliminate_for_one_third_chance :
  ∃ remaining_boxes : ℕ, one_third_chance remaining_boxes ∧ total_boxes - remaining_boxes = boxes_to_eliminate :=
sorry

end NUMINAMATH_GPT_min_boxes_to_eliminate_for_one_third_chance_l1383_138392


namespace NUMINAMATH_GPT_cost_of_5_pound_bag_is_2_l1383_138367

-- Define costs of each type of bag
def cost_10_pound_bag : ℝ := 20.40
def cost_25_pound_bag : ℝ := 32.25
def least_total_cost : ℝ := 98.75

-- Define the total weight constraint
def min_weight : ℕ := 65
def max_weight : ℕ := 80
def weight_25_pound_bags : ℕ := 75

-- Given condition: The total purchase fulfils the condition of minimum cost
def total_cost_3_bags_25 : ℝ := 3 * cost_25_pound_bag
def remaining_cost : ℝ := least_total_cost - total_cost_3_bags_25

-- Prove the cost of the 5-pound bag is $2.00
theorem cost_of_5_pound_bag_is_2 :
  ∃ (cost_5_pound_bag : ℝ), cost_5_pound_bag = remaining_cost :=
by
  sorry

end NUMINAMATH_GPT_cost_of_5_pound_bag_is_2_l1383_138367


namespace NUMINAMATH_GPT_root_in_interval_l1383_138340

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 2

theorem root_in_interval : 
  (f 1 < 0) → (f 2 > 0) → ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_root_in_interval_l1383_138340


namespace NUMINAMATH_GPT_compare_neg5_neg2_compare_neg_third_neg_half_compare_absneg5_0_l1383_138326

theorem compare_neg5_neg2 : -5 < -2 :=
by sorry

theorem compare_neg_third_neg_half : -(1/3) > -(1/2) :=
by sorry

theorem compare_absneg5_0 : abs (-5) > 0 :=
by sorry

end NUMINAMATH_GPT_compare_neg5_neg2_compare_neg_third_neg_half_compare_absneg5_0_l1383_138326


namespace NUMINAMATH_GPT_total_servings_of_vegetables_l1383_138341

def carrot_plant_serving : ℕ := 4
def num_green_bean_plants : ℕ := 10
def num_carrot_plants : ℕ := 8
def num_corn_plants : ℕ := 12
def num_tomato_plants : ℕ := 15
def corn_plant_serving : ℕ := 5 * carrot_plant_serving
def green_bean_plant_serving : ℕ := corn_plant_serving / 2
def tomato_plant_serving : ℕ := carrot_plant_serving + 3

theorem total_servings_of_vegetables :
  (num_carrot_plants * carrot_plant_serving) +
  (num_corn_plants * corn_plant_serving) +
  (num_green_bean_plants * green_bean_plant_serving) +
  (num_tomato_plants * tomato_plant_serving) = 477 := by
  sorry

end NUMINAMATH_GPT_total_servings_of_vegetables_l1383_138341


namespace NUMINAMATH_GPT_find_m_values_l1383_138337

theorem find_m_values :
  ∃ m : ℝ, (∀ (α β : ℝ), (3 * α^2 + m * α - 4 = 0 ∧ 3 * β^2 + m * β - 4 = 0) ∧ (α * β = -4 / 3) ∧ (α + β = -m / 3) ∧ (α * β = 2 * (α^3 + β^3))) ↔
  (m = -1.5 ∨ m = 6 ∨ m = -2.4) :=
sorry

end NUMINAMATH_GPT_find_m_values_l1383_138337


namespace NUMINAMATH_GPT_max_value_y2_minus_x2_plus_x_plus_5_l1383_138386

theorem max_value_y2_minus_x2_plus_x_plus_5 (x y : ℝ) (h : y^2 + x - 2 = 0) : 
  ∃ M, M = 7 ∧ ∀ u v, v^2 + u - 2 = 0 → y^2 - x^2 + x + 5 ≤ M :=
by
  sorry

end NUMINAMATH_GPT_max_value_y2_minus_x2_plus_x_plus_5_l1383_138386


namespace NUMINAMATH_GPT_trig_function_properties_l1383_138359

theorem trig_function_properties :
  ∀ x : ℝ, 
    (1 - 2 * (Real.sin (x - π / 4))^2) = Real.sin (2 * x) ∧ 
    (∀ x : ℝ, Real.sin (2 * (-x)) = -Real.sin (2 * x)) ∧ 
    2 * π / 2 = π :=
by
  sorry

end NUMINAMATH_GPT_trig_function_properties_l1383_138359


namespace NUMINAMATH_GPT_codecracker_total_combinations_l1383_138354

theorem codecracker_total_combinations (colors slots : ℕ) (h_colors : colors = 6) (h_slots : slots = 5) :
  colors ^ slots = 7776 :=
by
  rw [h_colors, h_slots]
  norm_num

end NUMINAMATH_GPT_codecracker_total_combinations_l1383_138354


namespace NUMINAMATH_GPT_log_identity_l1383_138329

noncomputable def my_log (base x : ℝ) := Real.log x / Real.log base

theorem log_identity (x : ℝ) (h : x > 0) (h1 : x ≠ 1) : 
  (my_log 4 x) * (my_log x 5) = my_log 4 5 :=
by
  sorry

end NUMINAMATH_GPT_log_identity_l1383_138329


namespace NUMINAMATH_GPT_least_number_to_add_divisible_l1383_138338

theorem least_number_to_add_divisible (n d : ℕ) (h1 : n = 929) (h2 : d = 30) : 
  ∃ x, (n + x) % d = 0 ∧ x = 1 := 
by 
  sorry

end NUMINAMATH_GPT_least_number_to_add_divisible_l1383_138338


namespace NUMINAMATH_GPT_determine_b_div_a_l1383_138310

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a * x^2 + b * x - a^2 - 7 * a

theorem determine_b_div_a
  (a b : ℝ)
  (hf_deriv : ∀ x : ℝ, (deriv (f a b)) x = 3 * x^2 + 2 * a * x + b)
  (hf_max : f a b 1 = 10)
  (hf_deriv_at_1 : (deriv (f a b)) 1 = 0) :
  b / a = -3 / 2 :=
sorry

end NUMINAMATH_GPT_determine_b_div_a_l1383_138310


namespace NUMINAMATH_GPT_number_of_technicians_l1383_138377

-- Definitions of the conditions
def average_salary_all_workers := 10000
def average_salary_technicians := 12000
def average_salary_rest := 8000
def total_workers := 14

-- Variables for the number of technicians and the rest of the workers
variable (T R : ℕ)

-- Problem statement in Lean
theorem number_of_technicians :
  (T + R = total_workers) →
  (T * average_salary_technicians + R * average_salary_rest = total_workers * average_salary_all_workers) →
  T = 7 :=
by
  -- leaving the proof as sorry
  sorry

end NUMINAMATH_GPT_number_of_technicians_l1383_138377


namespace NUMINAMATH_GPT_evaluate_expression_l1383_138330

theorem evaluate_expression (x : ℝ) (h : 3 * x - 2 = 13) : 6 * x - 4 = 26 :=
by {
    sorry
}

end NUMINAMATH_GPT_evaluate_expression_l1383_138330
