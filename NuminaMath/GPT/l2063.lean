import Mathlib

namespace NUMINAMATH_GPT_symmetric_point_origin_l2063_206322

theorem symmetric_point_origin (A : ℝ × ℝ) (A_sym : ℝ × ℝ) (h : A = (3, -2)) (h_sym : A_sym = (-A.1, -A.2)) : A_sym = (-3, 2) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_origin_l2063_206322


namespace NUMINAMATH_GPT_max_marks_l2063_206375

theorem max_marks (M : ℝ) (h1 : 0.33 * M = 92 + 40) : M = 400 :=
by
  sorry

end NUMINAMATH_GPT_max_marks_l2063_206375


namespace NUMINAMATH_GPT_find_p_l2063_206384

variable (p q : ℝ) (k : ℕ)

theorem find_p (h_sum : ∀ (α β : ℝ), α + β = 2) (h_prod : ∀ (α β : ℝ), α * β = k) (hk : k > 0) :
  p = -2 := by
  sorry

end NUMINAMATH_GPT_find_p_l2063_206384


namespace NUMINAMATH_GPT_probability_of_selecting_boy_given_girl_A_selected_l2063_206310

-- Define the total number of girls and boys
def total_girls : ℕ := 5
def total_boys : ℕ := 2

-- Define the group size to be selected
def group_size : ℕ := 3

-- Define the probability of selecting at least one boy given girl A is selected
def probability_at_least_one_boy_given_girl_A : ℚ := 3 / 5

-- Math problem reformulated as a Lean theorem
theorem probability_of_selecting_boy_given_girl_A_selected : 
  (total_girls = 5) → (total_boys = 2) → (group_size = 3) → 
  (probability_at_least_one_boy_given_girl_A = 3 / 5) :=
by sorry

end NUMINAMATH_GPT_probability_of_selecting_boy_given_girl_A_selected_l2063_206310


namespace NUMINAMATH_GPT_complex_modulus_square_l2063_206392

open Complex

theorem complex_modulus_square (z : ℂ) (h : z^2 + abs z ^ 2 = 7 + 6 * I) : abs z ^ 2 = 85 / 14 :=
sorry

end NUMINAMATH_GPT_complex_modulus_square_l2063_206392


namespace NUMINAMATH_GPT_monotonic_intervals_l2063_206331

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem monotonic_intervals :
  (∀ x (h : 0 < x ∧ x < Real.exp 1), 0 < f x) ∧
  (∀ x (h : Real.exp 1 < x), f x < 0) :=
by
  sorry

end NUMINAMATH_GPT_monotonic_intervals_l2063_206331


namespace NUMINAMATH_GPT_solve_for_x_l2063_206367

variables (x y z : ℝ)

def condition : Prop :=
  1 / (x + y) + 1 / (x - y) = z / (x - y)

theorem solve_for_x (h : condition x y z) : x = z / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2063_206367


namespace NUMINAMATH_GPT_kanul_total_amount_l2063_206302

def kanul_spent : ℝ := 3000 + 1000
def kanul_spent_percentage (T : ℝ) : ℝ := 0.30 * T

theorem kanul_total_amount (T : ℝ) (h : T = kanul_spent + kanul_spent_percentage T) :
  T = 5714.29 := sorry

end NUMINAMATH_GPT_kanul_total_amount_l2063_206302


namespace NUMINAMATH_GPT_solve_diophantine_eq_l2063_206323

theorem solve_diophantine_eq (a b : ℤ) (ha : a ≥ 0) (hb : b ≥ 0) : 
  a^2 = b * (b + 7) ↔ (a = 12 ∧ b = 9) ∨ (a = 0 ∧ b = 0) := 
by 
  sorry

end NUMINAMATH_GPT_solve_diophantine_eq_l2063_206323


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2063_206380

def p (a : ℝ) : Prop := ∃ (x : ℝ), x^2 + 2 * a * x - a ≤ 0

def q (a : ℝ) : Prop := a > 0 ∨ a < -1

theorem necessary_but_not_sufficient (a : ℝ) : (∃ (x : ℝ), x^2 + 2 * a * x - a ≤ 0) → (a > 0 ∨ a < -1) ∧ ¬((a > 0 ∨ a < -1) → (∃ (x : ℝ), x^2 + 2 * a * x - a ≤ 0)) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2063_206380


namespace NUMINAMATH_GPT_probability_even_in_5_of_7_rolls_is_21_over_128_l2063_206386

noncomputable def probability_even_in_5_of_7_rolls : ℚ :=
  let n := 7
  let k := 5
  let p := (1:ℚ) / 2
  let binomial (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k
  (binomial n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_even_in_5_of_7_rolls_is_21_over_128 :
  probability_even_in_5_of_7_rolls = 21 / 128 :=
by
  sorry

end NUMINAMATH_GPT_probability_even_in_5_of_7_rolls_is_21_over_128_l2063_206386


namespace NUMINAMATH_GPT_min_m_n_sum_l2063_206335

theorem min_m_n_sum (m n : ℕ) (h_pos_m : m > 0) (h_pos_n : n > 0) (h_eq : 45 * m = n^3) : m + n = 90 :=
sorry

end NUMINAMATH_GPT_min_m_n_sum_l2063_206335


namespace NUMINAMATH_GPT_sequence_formula_minimum_m_l2063_206370

variable (a_n : ℕ → ℕ) (S_n : ℕ → ℕ)

/-- The sequence a_n with sum of its first n terms S_n, the first term a_1 = 1, and the terms
   1, a_n, S_n forming an arithmetic sequence, satisfies a_n = 2^(n-1). -/
theorem sequence_formula (h1 : a_n 1 = 1)
    (h2 : ∀ n : ℕ, 1 + n * (a_n n - 1) = S_n n) :
    ∀ n : ℕ, a_n n = 2 ^ (n - 1) := by
  sorry

/-- T_n being the sum of the sequence {n / a_n}, if T_n < (m - 4) / 3 for all n in ℕ*, 
    then the minimum value of m is 16. -/
theorem minimum_m (T_n : ℕ → ℝ) (m : ℕ)
    (hT : ∀ n : ℕ, n > 0 → T_n n < (m - 4) / 3) :
    m ≥ 16 := by
  sorry

end NUMINAMATH_GPT_sequence_formula_minimum_m_l2063_206370


namespace NUMINAMATH_GPT_arithmetic_sequence_ratio_l2063_206351

-- Definitions and conditions from the problem
variable (a b : ℕ → ℕ)
variable (S T : ℕ → ℕ)
variable (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
variable (h2 : ∀ n, T n = n * (b 1 + b n) / 2)
variable (h3 : ∀ n, S n / T n = (3 * n - 1) / (n + 3))

-- The theorem that will give us the required answer
theorem arithmetic_sequence_ratio : 
  (a 8) / (b 5 + b 11) = 11 / 9 := by 
  have h4 := h3 15
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_ratio_l2063_206351


namespace NUMINAMATH_GPT_digits_arithmetic_l2063_206314

theorem digits_arithmetic :
  (12 / 3 / 4) * (56 / 7 / 8) = 1 :=
by
  sorry

end NUMINAMATH_GPT_digits_arithmetic_l2063_206314


namespace NUMINAMATH_GPT_find_x_squared_add_y_squared_l2063_206345

noncomputable def x_squared_add_y_squared (x y : ℝ) : ℝ :=
  x^2 + y^2

theorem find_x_squared_add_y_squared (x y : ℝ) 
  (h1 : x + y = 48)
  (h2 : x * y = 168) :
  x_squared_add_y_squared x y = 1968 :=
by
  sorry

end NUMINAMATH_GPT_find_x_squared_add_y_squared_l2063_206345


namespace NUMINAMATH_GPT_tan_sum_to_expression_l2063_206313

theorem tan_sum_to_expression (α : ℝ) (h : Real.tan (π / 4 + α) = 1 / 2) : 
  (Real.sin (2 * α) - Real.cos α ^ 2) / (1 + Real.cos (2 * α)) = -5 / 6 :=
by 
  sorry

end NUMINAMATH_GPT_tan_sum_to_expression_l2063_206313


namespace NUMINAMATH_GPT_find_other_endpoint_of_diameter_l2063_206349

noncomputable def circle_center : (ℝ × ℝ) := (4, -2)
noncomputable def one_endpoint_of_diameter : (ℝ × ℝ) := (7, 5)
noncomputable def other_endpoint_of_diameter : (ℝ × ℝ) := (1, -9)

theorem find_other_endpoint_of_diameter :
  let (cx, cy) := circle_center
  let (x1, y1) := one_endpoint_of_diameter
  let (x2, y2) := other_endpoint_of_diameter
  (x2, y2) = (2 * cx - x1, 2 * cy - y1) :=
by
  sorry

end NUMINAMATH_GPT_find_other_endpoint_of_diameter_l2063_206349


namespace NUMINAMATH_GPT_cashier_total_bills_l2063_206399

theorem cashier_total_bills
  (total_value : ℕ)
  (num_ten_bills : ℕ)
  (num_twenty_bills : ℕ)
  (h1 : total_value = 330)
  (h2 : num_ten_bills = 27)
  (h3 : num_twenty_bills = 3) :
  num_ten_bills + num_twenty_bills = 30 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cashier_total_bills_l2063_206399


namespace NUMINAMATH_GPT_fraction_calculation_l2063_206348

theorem fraction_calculation :
  let a := (1 / 2) + (1 / 3)
  let b := (2 / 7) + (1 / 4)
  ((a / b) * (3 / 5)) = (14 / 15) :=
by
  sorry

end NUMINAMATH_GPT_fraction_calculation_l2063_206348


namespace NUMINAMATH_GPT_final_result_l2063_206324

/-- A student chose a number, multiplied it by 5, then subtracted 138 
from the result. The number he chose was 48. What was the final result 
after subtracting 138? -/
theorem final_result (x : ℕ) (h1 : x = 48) : (x * 5) - 138 = 102 := by
  sorry

end NUMINAMATH_GPT_final_result_l2063_206324


namespace NUMINAMATH_GPT_more_sightings_than_triple_cape_may_l2063_206360

def daytona_shark_sightings := 26
def cape_may_shark_sightings := 7

theorem more_sightings_than_triple_cape_may :
  daytona_shark_sightings - 3 * cape_may_shark_sightings = 5 :=
by
  sorry

end NUMINAMATH_GPT_more_sightings_than_triple_cape_may_l2063_206360


namespace NUMINAMATH_GPT_find_valid_N_l2063_206365

def is_divisible_by_10_consec (N : ℕ) : Prop :=
  ∀ m : ℕ, (N % (List.prod (List.range' m 10)) = 0)

def is_not_divisible_by_11_consec (N : ℕ) : Prop :=
  ∀ m : ℕ, ¬ (N % (List.prod (List.range' m 11)) = 0)

theorem find_valid_N (N : ℕ) :
  (is_divisible_by_10_consec N ∧ is_not_divisible_by_11_consec N) ↔
  (∃ k : ℕ, (k > 0) ∧ ¬ (k % 11 = 0) ∧ N = k * Nat.factorial 10) :=
sorry

end NUMINAMATH_GPT_find_valid_N_l2063_206365


namespace NUMINAMATH_GPT_loss_per_metre_l2063_206304

theorem loss_per_metre
  (total_metres : ℕ)
  (selling_price : ℕ)
  (cost_price_per_m: ℕ)
  (selling_price_total : selling_price = 18000)
  (cost_price_per_m_def : cost_price_per_m = 95)
  (total_metres_def : total_metres = 200) :
  ((cost_price_per_m * total_metres - selling_price) / total_metres) = 5 :=
by
  sorry

end NUMINAMATH_GPT_loss_per_metre_l2063_206304


namespace NUMINAMATH_GPT_math_problem_l2063_206373

variables {x y : ℝ}

theorem math_problem (h1 : x * y = 16) (h2 : 1 / x = 3 * (1 / y)) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x < y) : 
  (2 * y - x) = 24 - (4 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_GPT_math_problem_l2063_206373


namespace NUMINAMATH_GPT_right_triangle_ratio_l2063_206387

theorem right_triangle_ratio (a b c r s : ℝ) (h : a / b = 2 / 5)
  (h_c : c^2 = a^2 + b^2)
  (h_r : r = a^2 / c)
  (h_s : s = b^2 / c) :
  r / s = 4 / 25 := by
  sorry

end NUMINAMATH_GPT_right_triangle_ratio_l2063_206387


namespace NUMINAMATH_GPT_age_twice_in_2_years_l2063_206389

/-
Conditions:
1. The man is 24 years older than his son.
2. The present age of the son is 22 years.
3. In a certain number of years, the man's age will be twice the age of his son.
-/
def man_is_24_years_older (S M : ℕ) : Prop := M = S + 24
def present_age_son : ℕ := 22
def age_twice_condition (Y S M : ℕ) : Prop := M + Y = 2 * (S + Y)

/-
Prove that in 2 years, the man's age will be twice the age of his son.
-/
theorem age_twice_in_2_years : ∃ (Y : ℕ), 
  (man_is_24_years_older present_age_son M) → 
  (age_twice_condition Y present_age_son M) →
  Y = 2 :=
by
  sorry

end NUMINAMATH_GPT_age_twice_in_2_years_l2063_206389


namespace NUMINAMATH_GPT_one_fourth_in_one_eighth_l2063_206341

theorem one_fourth_in_one_eighth : (1/8 : ℚ) / (1/4) = (1/2) := 
by
  sorry

end NUMINAMATH_GPT_one_fourth_in_one_eighth_l2063_206341


namespace NUMINAMATH_GPT_total_pages_read_l2063_206332

def pages_read_yesterday : ℕ := 21
def pages_read_today : ℕ := 17

theorem total_pages_read : pages_read_yesterday + pages_read_today = 38 :=
by
  sorry

end NUMINAMATH_GPT_total_pages_read_l2063_206332


namespace NUMINAMATH_GPT_min_value_xyz_l2063_206377

theorem min_value_xyz (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) : x^2 + y^2 + z^2 ≥ 1 / 14 := 
by
  sorry

end NUMINAMATH_GPT_min_value_xyz_l2063_206377


namespace NUMINAMATH_GPT_find_value_of_expression_l2063_206398

open Real

theorem find_value_of_expression (x y z w : ℝ) (h1 : x + y + z + w = 0) (h2 : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 := by
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l2063_206398


namespace NUMINAMATH_GPT_max_tickets_l2063_206334


theorem max_tickets (cost_regular : ℕ) (cost_discounted : ℕ) (threshold : ℕ) (total_money : ℕ) 
  (h1 : cost_regular = 15) 
  (h2 : cost_discounted = 12) 
  (h3 : threshold = 5)
  (h4 : total_money = 150) 
  : (total_money / cost_regular ≤ 10) ∧ 
    ((total_money - threshold * cost_regular) / cost_discounted + threshold = 11) :=
by
  sorry

end NUMINAMATH_GPT_max_tickets_l2063_206334


namespace NUMINAMATH_GPT_ellipse_foci_distance_l2063_206327

theorem ellipse_foci_distance 
  (h : ∀ x y : ℝ, 9 * x^2 + y^2 = 144) : 
  ∃ c : ℝ, c = 16 * Real.sqrt 2 :=
  sorry

end NUMINAMATH_GPT_ellipse_foci_distance_l2063_206327


namespace NUMINAMATH_GPT_solve_quadratic_l2063_206382

theorem solve_quadratic (h₁ : 48 * (3/4:ℚ)^2 - 74 * (3/4:ℚ) + 47 = 0) :
  ∃ x : ℚ, x ≠ 3/4 ∧ 48 * x^2 - 74 * x + 47 = 0 ∧ x = 11/12 := 
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l2063_206382


namespace NUMINAMATH_GPT_fraction_subtraction_l2063_206337

theorem fraction_subtraction :
  (9 / 19) - (5 / 57) - (2 / 38) = 1 / 3 := by
sorry

end NUMINAMATH_GPT_fraction_subtraction_l2063_206337


namespace NUMINAMATH_GPT_arithmetic_sequence_a10_l2063_206303

theorem arithmetic_sequence_a10 (a : ℕ → ℤ) (d : ℤ) (a1 : ℤ) 
  (h1 : a 7 = 9) (h2 : a 13 = -3) 
  (ha : ∀ n, a n = a1 + (n - 1) * d) :
  a 10 = 3 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_a10_l2063_206303


namespace NUMINAMATH_GPT_train_cross_bridge_time_l2063_206381

-- Length of the train in meters
def train_length : ℕ := 165

-- Length of the bridge in meters
def bridge_length : ℕ := 660

-- Speed of the train in kmph
def train_speed_kmph : ℕ := 54

-- Conversion factor from kmph to m/s
def kmph_to_mps : ℚ := 5 / 18

-- Total distance to be traveled by the train to cross the bridge
def total_distance : ℕ := train_length + bridge_length

-- Speed of the train in meters per second (m/s)
def train_speed_mps : ℚ := train_speed_kmph * kmph_to_mps

-- Time taken for the train to cross the bridge (in seconds)
def time_to_cross_bridge : ℚ := total_distance / train_speed_mps

-- Prove that the time taken for the train to cross the bridge is 55 seconds
theorem train_cross_bridge_time : time_to_cross_bridge = 55 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_train_cross_bridge_time_l2063_206381


namespace NUMINAMATH_GPT_range_of_x_for_y1_gt_y2_l2063_206363

noncomputable def y1 (x : ℝ) : ℝ := x - 3
noncomputable def y2 (x : ℝ) : ℝ := 4 / x

theorem range_of_x_for_y1_gt_y2 :
  ∀ x : ℝ, (y1 x > y2 x) ↔ ((-1 < x ∧ x < 0) ∨ (x > 4)) := by
  sorry

end NUMINAMATH_GPT_range_of_x_for_y1_gt_y2_l2063_206363


namespace NUMINAMATH_GPT_lateral_surface_area_of_prism_l2063_206300

theorem lateral_surface_area_of_prism (h : ℝ) (angle : ℝ) (h_pos : 0 < h) (angle_eq : angle = 60) :
  ∃ S : ℝ, S = 6 * h^2 :=
by
  sorry

end NUMINAMATH_GPT_lateral_surface_area_of_prism_l2063_206300


namespace NUMINAMATH_GPT_both_players_score_same_points_l2063_206379

theorem both_players_score_same_points :
  let P_A_score := 0.5 
  let P_B_score := 0.8 
  let P_A_miss := 1 - P_A_score
  let P_B_miss := 1 - P_B_score
  let P_both_miss := P_A_miss * P_B_miss
  let P_both_score := P_A_score * P_B_score
  let P_same_points := P_both_miss + P_both_score
  P_same_points = 0.5 := 
by {
  -- Actual proof should be here
  sorry
}

end NUMINAMATH_GPT_both_players_score_same_points_l2063_206379


namespace NUMINAMATH_GPT_isosceles_perimeter_l2063_206359

theorem isosceles_perimeter (peri_eqt : ℕ) (side_eqt : ℕ) (base_iso : ℕ) (side_iso : ℕ)
    (h1 : peri_eqt = 60)
    (h2 : side_eqt = peri_eqt / 3)
    (h3 : side_iso = side_eqt)
    (h4 : base_iso = 25) :
  2 * side_iso + base_iso = 65 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_perimeter_l2063_206359


namespace NUMINAMATH_GPT_count_powers_of_2_not_4_under_2000000_l2063_206308

theorem count_powers_of_2_not_4_under_2000000 :
  ∃ n, ∀ x, x < 2000000 → (∃ k, x = 2 ^ k ∧ (∀ m, x ≠ 4 ^ m)) ↔ x > 0 ∧ x < 2 ^ (n + 1) := by
  sorry

end NUMINAMATH_GPT_count_powers_of_2_not_4_under_2000000_l2063_206308


namespace NUMINAMATH_GPT_complement_of_A_in_B_l2063_206374

def set_A : Set ℤ := {x | 2 * x = x^2}
def set_B : Set ℤ := {x | -x^2 + x + 2 ≥ 0}

theorem complement_of_A_in_B :
  (set_B \ set_A) = {-1, 1} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_B_l2063_206374


namespace NUMINAMATH_GPT_point_in_second_quadrant_l2063_206354

theorem point_in_second_quadrant (x : ℝ) (h1 : 6 - 2 * x < 0) (h2 : x - 5 > 0) : x > 5 :=
by
  sorry

end NUMINAMATH_GPT_point_in_second_quadrant_l2063_206354


namespace NUMINAMATH_GPT_girls_exceed_boys_by_402_l2063_206350

theorem girls_exceed_boys_by_402 : 
  let girls := 739
  let boys := 337
  girls - boys = 402 :=
by
  sorry

end NUMINAMATH_GPT_girls_exceed_boys_by_402_l2063_206350


namespace NUMINAMATH_GPT_number_of_possible_monograms_l2063_206319

-- Define the set of letters before 'M'
def letters_before_M : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'}

-- Define the set of letters after 'M'
def letters_after_M : Finset Char := {'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}

-- State the theorem 
theorem number_of_possible_monograms : 
  (letters_before_M.card * letters_after_M.card) = 156 :=
by
  sorry

end NUMINAMATH_GPT_number_of_possible_monograms_l2063_206319


namespace NUMINAMATH_GPT_find_x_plus_y_l2063_206397

theorem find_x_plus_y
  (x y : ℝ)
  (h1 : x + Real.cos y = 2010)
  (h2 : x + 2010 * Real.sin y = 2009)
  (h3 : (π / 2) ≤ y ∧ y ≤ π) :
  x + y = 2011 + π :=
sorry

end NUMINAMATH_GPT_find_x_plus_y_l2063_206397


namespace NUMINAMATH_GPT_min_value_abs_x1_x2_l2063_206329

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x - Real.sqrt 3 * Real.cos x

theorem min_value_abs_x1_x2 
  (a : ℝ) (x1 x2 : ℝ)
  (h_symm : ∃ k : ℤ, -π / 6 - (Real.arctan (Real.sqrt 3 / a)) = (k * π + π / 2))
  (h_diff : f a x1 - f a x2 = -4) :
  |x1 + x2| = (2 * π) / 3 := 
sorry

end NUMINAMATH_GPT_min_value_abs_x1_x2_l2063_206329


namespace NUMINAMATH_GPT_range_x_plus_y_l2063_206356

theorem range_x_plus_y (x y : ℝ) (h : x^3 + y^3 = 2) : 0 < x + y ∧ x + y ≤ 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_x_plus_y_l2063_206356


namespace NUMINAMATH_GPT_inscribed_squares_equilateral_triangle_l2063_206321

theorem inscribed_squares_equilateral_triangle (a b c h_a h_b h_c : ℝ) 
  (h1 : a * h_a / (a + h_a) = b * h_b / (b + h_b))
  (h2 : b * h_b / (b + h_b) = c * h_c / (c + h_c)) :
  a = b ∧ b = c ∧ h_a = h_b ∧ h_b = h_c :=
sorry

end NUMINAMATH_GPT_inscribed_squares_equilateral_triangle_l2063_206321


namespace NUMINAMATH_GPT_regular_price_of_tire_l2063_206376

theorem regular_price_of_tire (p : ℝ) (h : 2 * p + p / 2 = 270) : p = 108 :=
sorry

end NUMINAMATH_GPT_regular_price_of_tire_l2063_206376


namespace NUMINAMATH_GPT_intersection_complement_eq_l2063_206368

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {3, 4, 5}
def U : Set ℝ := Set.univ  -- Universal set U is the set of all real numbers

theorem intersection_complement_eq : P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_eq_l2063_206368


namespace NUMINAMATH_GPT_tangent_normal_lines_l2063_206393

noncomputable def x (t : ℝ) : ℝ := (1 / 2) * t^2 - (1 / 4) * t^4
noncomputable def y (t : ℝ) : ℝ := (1 / 2) * t^2 + (1 / 3) * t^3
def t0 : ℝ := 0

theorem tangent_normal_lines :
  (∃ m : ℝ, ∀ t : ℝ, t = t0 → y t = m * x t) ∧
  (∃ n : ℝ, ∀ t : ℝ, t = t0 → y t = n * x t ∧ n = -1 / m) :=
sorry

end NUMINAMATH_GPT_tangent_normal_lines_l2063_206393


namespace NUMINAMATH_GPT_people_landed_in_virginia_l2063_206342

def initial_passengers : ℕ := 124
def texas_out : ℕ := 58
def texas_in : ℕ := 24
def north_carolina_out : ℕ := 47
def north_carolina_in : ℕ := 14
def crew_members : ℕ := 10

def final_passengers := initial_passengers - texas_out + texas_in - north_carolina_out + north_carolina_in
def total_people_landed := final_passengers + crew_members

theorem people_landed_in_virginia : total_people_landed = 67 :=
by
  sorry

end NUMINAMATH_GPT_people_landed_in_virginia_l2063_206342


namespace NUMINAMATH_GPT_seconds_in_3_hours_45_minutes_l2063_206325

theorem seconds_in_3_hours_45_minutes :
  let hours := 3
  let minutes := 45
  let minutes_in_hour := 60
  let seconds_in_minute := 60
  (hours * minutes_in_hour + minutes) * seconds_in_minute = 13500 := by
  sorry

end NUMINAMATH_GPT_seconds_in_3_hours_45_minutes_l2063_206325


namespace NUMINAMATH_GPT_smallest_common_students_l2063_206366

theorem smallest_common_students 
    (z : ℕ) (k : ℕ) (j : ℕ) 
    (hz : z = k ∧ k = j) 
    (hz_ratio : ∃ x : ℕ, z = 3 * x ∧ k = 2 * x ∧ j = 5 * x)
    (hz_group : ∃ y : ℕ, z = 14 * y) 
    (hk_group : ∃ w : ℕ, k = 10 * w) 
    (hj_group : ∃ v : ℕ, j = 15 * v) : 
    z = 630 ∧ k = 420 ∧ j = 1050 :=
    sorry

end NUMINAMATH_GPT_smallest_common_students_l2063_206366


namespace NUMINAMATH_GPT_A_wins_if_perfect_square_or_prime_l2063_206316

theorem A_wins_if_perfect_square_or_prime (n : ℕ) (h_pos : 0 < n) : 
  (∃ A_wins : Bool, A_wins = true ↔ (∃ k : ℕ, n = k^2) ∨ (∃ p : ℕ, Nat.Prime p ∧ n = p)) :=
by
  sorry

end NUMINAMATH_GPT_A_wins_if_perfect_square_or_prime_l2063_206316


namespace NUMINAMATH_GPT_solve_system_of_equations_l2063_206344

theorem solve_system_of_equations (a b c x y z : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  (a * y + b * x = c) ∧ (c * x + a * z = b) ∧ (b * z + c * y = a) →
  (x = (b^2 + c^2 - a^2) / (2 * b * c)) ∧
  (y = (a^2 + c^2 - b^2) / (2 * a * c)) ∧
  (z = (a^2 + b^2 - c^2) / (2 * a * b)) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l2063_206344


namespace NUMINAMATH_GPT_chandra_valid_pairings_l2063_206385

noncomputable def valid_pairings (total_items : Nat) (invalid_pairing : Nat) : Nat :=
total_items * total_items - invalid_pairing

theorem chandra_valid_pairings : valid_pairings 5 1 = 24 := by
  sorry

end NUMINAMATH_GPT_chandra_valid_pairings_l2063_206385


namespace NUMINAMATH_GPT_no_positive_reals_satisfy_conditions_l2063_206369

theorem no_positive_reals_satisfy_conditions :
  ¬ ∃ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  4 * (a * b + b * c + c * a) - 1 ≥ a^2 + b^2 + c^2 ∧ 
  a^2 + b^2 + c^2 ≥ 3 * (a^3 + b^3 + c^3) :=
by
  sorry

end NUMINAMATH_GPT_no_positive_reals_satisfy_conditions_l2063_206369


namespace NUMINAMATH_GPT_volume_of_given_sphere_l2063_206357

noncomputable def volume_of_sphere (A d : ℝ) : ℝ :=
  (4 / 3) * Real.pi * (Real.sqrt (d^2 + A / Real.pi))^3

theorem volume_of_given_sphere
  (hA : 2 * Real.pi = 2 * Real.pi)
  (hd : 1 = 1):
  volume_of_sphere (2 * Real.pi) 1 = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_volume_of_given_sphere_l2063_206357


namespace NUMINAMATH_GPT_factorize_expression_l2063_206395

variable (a : ℝ) -- assuming a is a real number

theorem factorize_expression (a : ℝ) : a^2 + 3 * a = a * (a + 3) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_factorize_expression_l2063_206395


namespace NUMINAMATH_GPT_count_even_integers_between_l2063_206318

theorem count_even_integers_between : 
    let lower := 18 / 5
    let upper := 45 / 2
    ∃ (count : ℕ), (∀ n : ℕ, lower < n ∧ n < upper → n % 2 = 0 → n = 4 ∨ n = 6 ∨ n = 8 ∨ n = 10 ∨ n = 12 ∨ n = 14 ∨ n = 16 ∨ n = 18 ∨ n = 20 ∨ n = 22) ∧ count = 10 :=
by
  sorry

end NUMINAMATH_GPT_count_even_integers_between_l2063_206318


namespace NUMINAMATH_GPT_series_sum_eq_l2063_206333

noncomputable def series_term (n : ℕ) : ℝ :=
  (4 * ↑n + 3) / ((4 * ↑n + 1)^2 * (4 * ↑n + 5)^2)

theorem series_sum_eq :
  (∑' n, series_term n) = 1 / 800 :=
by
  sorry

end NUMINAMATH_GPT_series_sum_eq_l2063_206333


namespace NUMINAMATH_GPT_proof_problem_l2063_206362

variable {x y : ℝ}

def conditions (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + y = 1

theorem proof_problem (h : conditions x y) :
  x + y - 4 * x * y ≥ 0 ∧ (1 / x) + 4 / (1 + y) ≥ 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l2063_206362


namespace NUMINAMATH_GPT_unique_sequence_l2063_206328

theorem unique_sequence (a : ℕ → ℝ) 
  (h1 : a 0 = 1) 
  (h2 : ∀ n : ℕ, a n > 0) 
  (h3 : ∀ n : ℕ, a n - a (n + 1) = a (n + 2)) : 
  ∀ n : ℕ, a n = ( (-1 + Real.sqrt 5) / 2)^n := 
sorry

end NUMINAMATH_GPT_unique_sequence_l2063_206328


namespace NUMINAMATH_GPT_find_p_q_of_divisible_polynomial_l2063_206364

theorem find_p_q_of_divisible_polynomial :
  ∃ p q : ℤ, (p, q) = (-7, -12) ∧
    (∀ x : ℤ, (x^5 - x^4 + x^3 - p*x^2 + q*x + 4 = 0) → (x = -2 ∨ x = 1)) :=
by
  sorry

end NUMINAMATH_GPT_find_p_q_of_divisible_polynomial_l2063_206364


namespace NUMINAMATH_GPT_compound_interest_calculation_l2063_206347

theorem compound_interest_calculation :
  let P_SI := 1750.0000000000018
  let r_SI := 0.08
  let t_SI := 3
  let r_CI := 0.10
  let t_CI := 2
  let SI := P_SI * r_SI * t_SI
  let CI (P_CI : ℝ) := P_CI * ((1 + r_CI) ^ t_CI - 1)
  (SI = 420.0000000000004) →
  (SI = (1 / 2) * CI P_CI) →
  P_CI = 4000.000000000004 :=
by
  intros P_SI r_SI t_SI r_CI t_CI SI CI h1 h2
  sorry

end NUMINAMATH_GPT_compound_interest_calculation_l2063_206347


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2063_206305

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 0 → x^2 > 0) ∧ ¬(x^2 > 0 → x > 0) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2063_206305


namespace NUMINAMATH_GPT_jean_jail_time_l2063_206396

/-- Jean has 3 counts of arson -/
def arson_count : ℕ := 3

/-- Each arson count has a 36-month sentence -/
def arson_sentence : ℕ := 36

/-- Jean has 2 burglary charges -/
def burglary_charges : ℕ := 2

/-- Each burglary charge has an 18-month sentence -/
def burglary_sentence : ℕ := 18

/-- Jean has six times as many petty larceny charges as burglary charges -/
def petty_larceny_multiplier : ℕ := 6

/-- Each petty larceny charge is 1/3 as long as a burglary charge -/
def petty_larceny_sentence : ℕ := burglary_sentence / 3

/-- Calculate all charges in months -/
def total_charges : ℕ :=
  (arson_count * arson_sentence) +
  (burglary_charges * burglary_sentence) +
  (petty_larceny_multiplier * burglary_charges * petty_larceny_sentence)

/-- Prove the total jail time for Jean is 216 months -/
theorem jean_jail_time : total_charges = 216 := by
  sorry

end NUMINAMATH_GPT_jean_jail_time_l2063_206396


namespace NUMINAMATH_GPT_total_feed_amount_l2063_206340

theorem total_feed_amount (x : ℝ) : 
  (17 * 0.18) + (x * 0.53) = (17 + x) * 0.36 → 17 + x = 35 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_total_feed_amount_l2063_206340


namespace NUMINAMATH_GPT_small_cubes_with_two_faces_painted_l2063_206353

theorem small_cubes_with_two_faces_painted :
  let n := 5
  let total_small_cubes := n ^ 3
  let small_cube_edge_length := 1
  let small_cubes_with_two_faces := 12 * (n - 2)
  12 * (n - 2) = 36 :=
by
  let n := 5
  let total_small_cubes := n ^ 3
  let small_cube_edge_length := 1
  let small_cubes_with_two_faces := 12 * (n - 2)
  exact sorry

end NUMINAMATH_GPT_small_cubes_with_two_faces_painted_l2063_206353


namespace NUMINAMATH_GPT_vertices_after_removal_l2063_206338

theorem vertices_after_removal (a b : ℕ) (h₁ : a = 5) (h₂ : b = 2) : 
  let initial_vertices := 8
  let removed_vertices := initial_vertices
  let new_vertices := 8 * 9
  let final_vertices := new_vertices - removed_vertices
  final_vertices = 64 :=
by
  sorry

end NUMINAMATH_GPT_vertices_after_removal_l2063_206338


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2063_206339

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x ≥ 1) → (|x + 1| + |x - 1| = 2 * |x|) ∧ ¬((x ≥ 1) ↔ (|x + 1| + |x - 1| = 2 * |x|)) := by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2063_206339


namespace NUMINAMATH_GPT_solution_proof_l2063_206361

noncomputable def problem_statement : Prop :=
  ((16^(1/4) * 32^(1/5)) + 64^(1/6)) = 6

theorem solution_proof : problem_statement :=
by
  sorry

end NUMINAMATH_GPT_solution_proof_l2063_206361


namespace NUMINAMATH_GPT_mappings_count_A_to_B_l2063_206372

open Finset

def A : Finset ℕ := {1, 2}
def B : Finset ℕ := {3, 4}

theorem mappings_count_A_to_B : (card B) ^ (card A) = 4 :=
by
  -- This line will state that the proof is skipped for now.
  sorry

end NUMINAMATH_GPT_mappings_count_A_to_B_l2063_206372


namespace NUMINAMATH_GPT_problem_1_problem_2_l2063_206317

def setP (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
def setS (x : ℝ) (m : ℝ) : Prop := |x - 1| ≤ m

theorem problem_1 (m : ℝ) : (m ∈ Set.Iic (3)) → ∀ x, (setP x ∨ setS x m) → setP x := sorry

theorem problem_2 : ¬ ∃ m : ℝ, ∀ x : ℝ, (setP x ↔ setS x m) := sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2063_206317


namespace NUMINAMATH_GPT_sample_size_is_50_l2063_206391

theorem sample_size_is_50 (n : ℕ) :
  (n > 0) → 
  (10 / n = 2 / (2 + 3 + 5)) → 
  n = 50 := 
by
  sorry

end NUMINAMATH_GPT_sample_size_is_50_l2063_206391


namespace NUMINAMATH_GPT_worker_bees_in_hive_l2063_206307

variable (initialWorkerBees leavingWorkerBees returningWorkerBees : ℕ)

def finalWorkerBees (initialWorkerBees leavingWorkerBees returningWorkerBees : ℕ) : ℕ :=
  initialWorkerBees - leavingWorkerBees + returningWorkerBees

theorem worker_bees_in_hive
  (initialWorkerBees : ℕ := 400)
  (leavingWorkerBees : ℕ := 28)
  (returningWorkerBees : ℕ := 15) :
  finalWorkerBees initialWorkerBees leavingWorkerBees returningWorkerBees = 387 := by
  sorry

end NUMINAMATH_GPT_worker_bees_in_hive_l2063_206307


namespace NUMINAMATH_GPT_inequality_problem_l2063_206378

theorem inequality_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (a + 1 / (2 * b^2)) * (b + 1 / (2 * a^2)) ≥ 25 / 4 := 
sorry

end NUMINAMATH_GPT_inequality_problem_l2063_206378


namespace NUMINAMATH_GPT_lower_limit_of_b_l2063_206355

theorem lower_limit_of_b (a : ℤ) (b : ℤ) (h₁ : 8 < a ∧ a < 15) (h₂ : ∃ x, x < b ∧ b < 21) (h₃ : (14 : ℚ) / b - (9 : ℚ) / b = 1.55) : b = 4 :=
by
  sorry

end NUMINAMATH_GPT_lower_limit_of_b_l2063_206355


namespace NUMINAMATH_GPT_adam_paper_tearing_l2063_206383

theorem adam_paper_tearing (n : ℕ) :
  let starts_with_one_piece : ℕ := 1
  let increment_to_four : ℕ := 3
  let increment_to_ten : ℕ := 9
  let target_pieces : ℕ := 20000
  let start_modulo : ℤ := 1

  -- Modulo 3 analysis
  starts_with_one_piece % 3 = start_modulo ∧
  increment_to_four % 3 = 0 ∧ 
  increment_to_ten % 3 = 0 ∧ 
  target_pieces % 3 = 2 → 
  n % 3 = start_modulo ∧ ∀ m, m % 3 = 0 → n + m ≠ target_pieces :=
sorry

end NUMINAMATH_GPT_adam_paper_tearing_l2063_206383


namespace NUMINAMATH_GPT_donna_additional_flyers_l2063_206394

theorem donna_additional_flyers (m d a : ℕ) (h1 : m = 33) (h2 : d = 2 * m + a) (h3 : d = 71) : a = 5 :=
by
  have m_val : m = 33 := h1
  rw [m_val] at h2
  linarith [h3, h2]

end NUMINAMATH_GPT_donna_additional_flyers_l2063_206394


namespace NUMINAMATH_GPT_percentage_of_students_on_trip_l2063_206390

variable (students : ℕ) -- Total number of students at the school
variable (students_trip_and_more_than_100 : ℕ) -- Number of students who went to the camping trip and took more than $100
variable (percent_trip_and_more_than_100 : ℚ) -- Percent of students who went to camping trip and took more than $100

-- Given Conditions
def cond1 : students_trip_and_more_than_100 = (percent_trip_and_more_than_100 * students) := 
  by
    sorry  -- This will represent the first condition: 18% of students went to a camping trip and took more than $100.

variable (percent_did_not_take_more_than_100 : ℚ) -- Percent of students who went to camping trip and did not take more than $100

-- second condition
def cond2 : percent_did_not_take_more_than_100 = 0.75 := 
  by
    sorry  -- Represent the second condition: 75% of students who went to the camping trip did not take more than $100.

-- Prove
theorem percentage_of_students_on_trip : 
  (students_trip_and_more_than_100 / (0.25 * students)) * 100 = (72 : ℚ) := 
  by
    sorry

end NUMINAMATH_GPT_percentage_of_students_on_trip_l2063_206390


namespace NUMINAMATH_GPT_find_a_for_parabola_l2063_206352

theorem find_a_for_parabola (a : ℝ) :
  (∃ y : ℝ, y = a * (-1 / 2)^2) → a = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_for_parabola_l2063_206352


namespace NUMINAMATH_GPT_relationship_among_abc_l2063_206312

noncomputable def a : ℝ := Real.log 4 / Real.log 3
noncomputable def b : ℝ := Real.log 3 / Real.log 0.4
noncomputable def c : ℝ := 0.4 ^ 3

theorem relationship_among_abc : a > c ∧ c > b := by
  sorry

end NUMINAMATH_GPT_relationship_among_abc_l2063_206312


namespace NUMINAMATH_GPT_points_do_not_exist_l2063_206336

/-- 
  If \( A, B, C, D \) are four points in space and 
  \( AB = 8 \) cm, 
  \( CD = 8 \) cm, 
  \( AC = 10 \) cm, 
  \( BD = 10 \) cm, 
  \( AD = 13 \) cm, 
  \( BC = 13 \) cm, 
  then such points \( A, B, C, D \) cannot exist.
-/
theorem points_do_not_exist 
  (A B C D : Type)
  (AB CD AC BD AD BC : ℝ) 
  (h1 : AB = 8) 
  (h2 : CD = 8) 
  (h3 : AC = 10)
  (h4 : BD = 10)
  (h5 : AD = 13)
  (h6 : BC = 13) : 
  false :=
sorry

end NUMINAMATH_GPT_points_do_not_exist_l2063_206336


namespace NUMINAMATH_GPT_best_fitting_model_is_model_2_l2063_206346

-- Variables representing the correlation coefficients of the four models
def R2_model_1 : ℝ := 0.86
def R2_model_2 : ℝ := 0.96
def R2_model_3 : ℝ := 0.73
def R2_model_4 : ℝ := 0.66

-- Statement asserting that Model 2 has the best fitting effect
theorem best_fitting_model_is_model_2 :
  R2_model_2 = 0.96 ∧ R2_model_2 > R2_model_1 ∧ R2_model_2 > R2_model_3 ∧ R2_model_2 > R2_model_4 :=
by {
  sorry
}

end NUMINAMATH_GPT_best_fitting_model_is_model_2_l2063_206346


namespace NUMINAMATH_GPT_f_neg_one_value_l2063_206301

theorem f_neg_one_value (f : ℝ → ℝ) (b : ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_def : ∀ x : ℝ, 0 ≤ x → f x = 2^x + 2 * x + b) :
  f (-1) = -3 := by
sorry

end NUMINAMATH_GPT_f_neg_one_value_l2063_206301


namespace NUMINAMATH_GPT_triangle_square_side_length_ratio_l2063_206320

theorem triangle_square_side_length_ratio (t s : ℝ) (ht : 3 * t = 12) (hs : 4 * s = 12) : 
  t / s = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_square_side_length_ratio_l2063_206320


namespace NUMINAMATH_GPT_a_0_eq_2_pow_n_S_n_eq_3_pow_n_minus_2_pow_n_S_n_magnitude_comparison_l2063_206358

noncomputable def a_0 (n : ℕ) : ℕ := 2^n
noncomputable def S_n (n : ℕ) : ℕ := 3^n - 2^n
noncomputable def T_n (n : ℕ) : ℕ := (n - 2) * 2^n + 2 * n^2

theorem a_0_eq_2_pow_n (n : ℕ) (h : n > 0) : a_0 n = 2^n := sorry

theorem S_n_eq_3_pow_n_minus_2_pow_n (n : ℕ) (h : n > 0) : S_n n = 3^n - 2^n := sorry

theorem S_n_magnitude_comparison : 
  ∀ (n : ℕ), 
    (n = 1 → S_n n > T_n n) ∧
    (n = 2 ∨ n = 3 → S_n n < T_n n) ∧
    (n ≥ 4 → S_n n > T_n n) := sorry

end NUMINAMATH_GPT_a_0_eq_2_pow_n_S_n_eq_3_pow_n_minus_2_pow_n_S_n_magnitude_comparison_l2063_206358


namespace NUMINAMATH_GPT_original_days_l2063_206315

-- Definitions based on the given problem conditions
def totalLaborers : ℝ := 17.5
def absentLaborers : ℝ := 7
def workingLaborers : ℝ := totalLaborers - absentLaborers
def workDaysByWorkingLaborers : ℝ := 10
def totalLaborDays : ℝ := workingLaborers * workDaysByWorkingLaborers

theorem original_days (D : ℝ) (h : totalLaborers * D = totalLaborDays) : D = 6 := sorry

end NUMINAMATH_GPT_original_days_l2063_206315


namespace NUMINAMATH_GPT_Sasha_can_write_2011_l2063_206326

theorem Sasha_can_write_2011 (N : ℕ) (hN : N > 1) : 
    ∃ (s : ℕ → ℕ), (s 0 = N) ∧ (∃ n, s n = 2011) ∧ 
    (∀ k, ∃ d, d > 1 ∧ (s (k + 1) = s k + d ∨ s (k + 1) = s k - d)) :=
sorry

end NUMINAMATH_GPT_Sasha_can_write_2011_l2063_206326


namespace NUMINAMATH_GPT_third_candidate_votes_l2063_206309

-- Definition of the problem's conditions
variables (total_votes winning_votes candidate2_votes : ℕ)
variables (winning_percentage : ℚ)

-- Conditions given in the problem
def conditions : Prop :=
  winning_votes = 11628 ∧
  winning_percentage = 0.4969230769230769 ∧
  (total_votes : ℚ) = winning_votes / winning_percentage ∧
  candidate2_votes = 7636

-- The theorem we need to prove
theorem third_candidate_votes (total_votes winning_votes candidate2_votes : ℕ)
    (winning_percentage : ℚ)
    (h : conditions total_votes winning_votes candidate2_votes winning_percentage) :
    total_votes - (winning_votes + candidate2_votes) = 4136 := 
  sorry

end NUMINAMATH_GPT_third_candidate_votes_l2063_206309


namespace NUMINAMATH_GPT_average_visitors_per_day_in_month_l2063_206388

theorem average_visitors_per_day_in_month (avg_visitors_sunday : ℕ) (avg_visitors_other_days : ℕ) (days_in_month : ℕ) (starts_sunday : Bool) :
  avg_visitors_sunday = 140 → avg_visitors_other_days = 80 → days_in_month = 30 → starts_sunday = true → 
  (∀ avg_visitors, avg_visitors = (4 * avg_visitors_sunday + 26 * avg_visitors_other_days) / days_in_month → avg_visitors = 88) :=
by
  intros h1 h2 h3 h4
  have total_visitors : ℕ := 4 * avg_visitors_sunday + 26 * avg_visitors_other_days
  have avg := total_visitors / days_in_month
  have visitors : ℕ := 2640
  sorry

end NUMINAMATH_GPT_average_visitors_per_day_in_month_l2063_206388


namespace NUMINAMATH_GPT_n_minus_m_eq_200_l2063_206311

-- Define the parameters
variable (m n x : ℝ)

-- State the conditions
def condition1 : Prop := m ≤ 8 * x - 1 ∧ 8 * x - 1 ≤ n 
def condition2 : Prop := (n + 1)/8 - (m + 1)/8 = 25

-- State the theorem to prove
theorem n_minus_m_eq_200 (h1 : condition1 m n x) (h2 : condition2 m n) : n - m = 200 := 
by 
  -- This is where the proof would go, but we use sorry to skip it.
  sorry

end NUMINAMATH_GPT_n_minus_m_eq_200_l2063_206311


namespace NUMINAMATH_GPT_modulus_of_complex_l2063_206371

-- Some necessary imports for complex numbers and proofs in Lean
open Complex

theorem modulus_of_complex (x y : ℝ) (h : (1 + I) * x = 1 + y * I) : abs (x + y * I) = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_modulus_of_complex_l2063_206371


namespace NUMINAMATH_GPT_solution_set_ineq_l2063_206343

theorem solution_set_ineq (m : ℝ) (hm : m > 1) :
  {x : ℝ | x^2 + (m-1) * x - m >= 0} = {x : ℝ | x <= -m ∨ x >= 1} :=
sorry

end NUMINAMATH_GPT_solution_set_ineq_l2063_206343


namespace NUMINAMATH_GPT_value_of_a_l2063_206306

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, (5-x)/(x-2) ≥ 0 ↔ -3 < x ∧ x < a) → a > 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_value_of_a_l2063_206306


namespace NUMINAMATH_GPT_sum_of_reciprocals_factors_12_l2063_206330

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_factors_12_l2063_206330
