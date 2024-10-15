import Mathlib

namespace NUMINAMATH_GPT_functional_eq_linear_l633_63333

theorem functional_eq_linear {f : ℝ → ℝ} (h : ∀ x y : ℝ, f (x ^ 2 - y ^ 2) = (x + y) * (f x - f y)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end NUMINAMATH_GPT_functional_eq_linear_l633_63333


namespace NUMINAMATH_GPT_bicycle_final_price_l633_63364

-- Define initial conditions
def original_price : ℝ := 200
def wednesday_discount : ℝ := 0.40
def friday_increase : ℝ := 0.20
def saturday_discount : ℝ := 0.25

-- Statement to prove that the final price, after all discounts and increases, is $108
theorem bicycle_final_price :
  (original_price * (1 - wednesday_discount) * (1 + friday_increase) * (1 - saturday_discount)) = 108 := by
  sorry

end NUMINAMATH_GPT_bicycle_final_price_l633_63364


namespace NUMINAMATH_GPT_geometric_extraction_from_arithmetic_l633_63342

theorem geometric_extraction_from_arithmetic (a b : ℤ) :
  ∃ k : ℕ → ℤ, (∀ n : ℕ, k n = a * (b + 1) ^ n) ∧ (∀ n : ℕ, ∃ m : ℕ, k n = a + b * m) :=
by sorry

end NUMINAMATH_GPT_geometric_extraction_from_arithmetic_l633_63342


namespace NUMINAMATH_GPT_find_x2_plus_y2_l633_63357

theorem find_x2_plus_y2 
  (x y : ℕ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h1 : x * y + x + y = 117) 
  (h2 : x^2 * y + x * y^2 = 1512) : 
  x^2 + y^2 = 549 := 
sorry

end NUMINAMATH_GPT_find_x2_plus_y2_l633_63357


namespace NUMINAMATH_GPT_double_root_condition_l633_63347

theorem double_root_condition (a : ℝ) : 
  (∃! x : ℝ, (x+2)^2 * (x+7)^2 + a = 0) ↔ a = -625 / 16 :=
sorry

end NUMINAMATH_GPT_double_root_condition_l633_63347


namespace NUMINAMATH_GPT_prob_bigger_number_correct_l633_63314

def bernardo_picks := {n | 1 ≤ n ∧ n ≤ 10}
def silvia_picks := {n | 1 ≤ n ∧ n ≤ 8}

noncomputable def prob_bigger_number : ℚ :=
  let prob_bern_picks_10 : ℚ := 3 / 10
  let prob_bern_not_10_larger_silvia : ℚ := 55 / 112
  let prob_bern_not_picks_10 : ℚ := 7 / 10
  prob_bern_picks_10 + prob_bern_not_10_larger_silvia * prob_bern_not_picks_10

theorem prob_bigger_number_correct :
  prob_bigger_number = 9 / 14 := by
  sorry

end NUMINAMATH_GPT_prob_bigger_number_correct_l633_63314


namespace NUMINAMATH_GPT_calc_problem1_calc_problem2_l633_63306

-- Proof Problem 1
theorem calc_problem1 : 
  (Real.sqrt 3 + 2 * Real.sqrt 2) - (3 * Real.sqrt 3 + Real.sqrt 2) = -2 * Real.sqrt 3 + Real.sqrt 2 := 
by 
  sorry

-- Proof Problem 2
theorem calc_problem2 : 
  Real.sqrt 2 * (Real.sqrt 2 + 1 / Real.sqrt 2) - abs (2 - Real.sqrt 6) = 5 - Real.sqrt 6 := 
by 
  sorry

end NUMINAMATH_GPT_calc_problem1_calc_problem2_l633_63306


namespace NUMINAMATH_GPT_coin_combinations_l633_63301

-- Define the coins and their counts
def one_cent_count := 1
def two_cent_count := 1
def five_cent_count := 1
def ten_cent_count := 4
def fifty_cent_count := 2

-- Define the expected number of different possible amounts
def expected_amounts := 119

-- Prove that the expected number of possible amounts can be achieved given the coins
theorem coin_combinations : 
  (∃ sums : Finset ℕ, 
    sums.card = expected_amounts ∧ 
    (∀ n ∈ sums, n = one_cent_count * 1 + 
                          two_cent_count * 2 + 
                          five_cent_count * 5 + 
                          ten_cent_count * 10 + 
                          fifty_cent_count * 50)) :=
sorry

end NUMINAMATH_GPT_coin_combinations_l633_63301


namespace NUMINAMATH_GPT_math_problem_l633_63316

theorem math_problem
  (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : x = z * (1 / y)) : 
  (x - z / x) * (y + 1 / (z * y)) = (x^4 - z^3 + x^2 * (z^2 - z)) / (z * x^2) :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l633_63316


namespace NUMINAMATH_GPT_range_of_m_l633_63355
-- Import the essential libraries

-- Define the problem conditions and state the theorem
theorem range_of_m (f : ℝ → ℝ) (h_even : ∀ x : ℝ, f x = f (-x)) (h_mono_dec : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f y ≤ f x)
  (m : ℝ) (h_ineq : f m > f (1 - m)) : m < 1 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l633_63355


namespace NUMINAMATH_GPT_area_under_f_l633_63389

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2 - 3 * x

noncomputable def f' (x : ℝ) : ℝ := 1 / x + 2 * x - 3

theorem area_under_f' : 
  - ∫ x in (1/2 : ℝ)..1, f' x = (3 / 4) - Real.log 2 := 
by
  sorry

end NUMINAMATH_GPT_area_under_f_l633_63389


namespace NUMINAMATH_GPT_lily_petals_l633_63304

theorem lily_petals (L : ℕ) (h1 : 8 * L + 15 = 63) : L = 6 :=
by sorry

end NUMINAMATH_GPT_lily_petals_l633_63304


namespace NUMINAMATH_GPT_quadratic_no_discriminant_23_l633_63328

theorem quadratic_no_discriminant_23 (a b c : ℤ) (h_eq : b^2 - 4 * a * c = 23) : False := sorry

end NUMINAMATH_GPT_quadratic_no_discriminant_23_l633_63328


namespace NUMINAMATH_GPT_max_possible_value_l633_63331

-- Define the number of cities and the structure of roads.
def numCities : ℕ := 110

-- Condition: Each city has either a road or no road to another city
def Road (city1 city2 : ℕ) : Prop := sorry  -- A placeholder definition for the road relationship

-- Condition: Number of roads leading out of each city.
def numRoads (city : ℕ) : ℕ := sorry  -- A placeholder for the actual function counting the number of roads from a city

-- Condition: The driver starts at a city with exactly one road leading out.
def startCity : ℕ := sorry  -- A placeholder for the starting city

-- Main theorem statement to prove the maximum possible value of N is 107
theorem max_possible_value : ∃ N : ℕ, N ≤ 107 ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ N → numRoads k = k) :=
by
  sorry  -- Actual proof is not required, hence we use sorry to indicate the proof step is skipped.

end NUMINAMATH_GPT_max_possible_value_l633_63331


namespace NUMINAMATH_GPT_probability_x_y_less_than_3_l633_63322

theorem probability_x_y_less_than_3 :
  let A := 6 * 2
  let triangle_area := (1 / 2) * 3 * 2
  let P := triangle_area / A
  P = 1 / 4 := by sorry

end NUMINAMATH_GPT_probability_x_y_less_than_3_l633_63322


namespace NUMINAMATH_GPT_percent_swans_non_ducks_l633_63356

def percent_ducks : ℝ := 35
def percent_swans : ℝ := 30
def percent_herons : ℝ := 20
def percent_geese : ℝ := 15
def percent_non_ducks := 100 - percent_ducks

theorem percent_swans_non_ducks : (percent_swans / percent_non_ducks) * 100 = 46.15 := 
by
  sorry

end NUMINAMATH_GPT_percent_swans_non_ducks_l633_63356


namespace NUMINAMATH_GPT_quadratic_distinct_roots_l633_63339

theorem quadratic_distinct_roots (c : ℝ) (h : c < 1 / 4) : 
  ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 * r1 + 2 * r1 + 4 * c = 0) ∧ (r2 * r2 + 2 * r2 + 4 * c = 0) :=
by 
sorry

end NUMINAMATH_GPT_quadratic_distinct_roots_l633_63339


namespace NUMINAMATH_GPT_stars_total_is_correct_l633_63382

-- Define the given conditions
def number_of_stars_per_student : ℕ := 6
def number_of_students : ℕ := 210

-- Define total number of stars calculation
def total_number_of_stars : ℕ := number_of_stars_per_student * number_of_students

-- Proof statement that the total number of stars is correct
theorem stars_total_is_correct : total_number_of_stars = 1260 := by
  sorry

end NUMINAMATH_GPT_stars_total_is_correct_l633_63382


namespace NUMINAMATH_GPT_expression_undefined_at_12_l633_63330

theorem expression_undefined_at_12 :
  ¬ ∃ x : ℝ, x = 12 ∧ (x^2 - 24 * x + 144 = 0) →
  (∃ y : ℝ, y = (3 * x^3 + 5) / (x^2 - 24 * x + 144)) :=
by
  sorry

end NUMINAMATH_GPT_expression_undefined_at_12_l633_63330


namespace NUMINAMATH_GPT_value_of_a_l633_63313

-- Definition of the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

-- Definition of the derivative f'(-1)
def f_prime_at_neg1 (a : ℝ) : ℝ := 3 * a - 6

-- The theorem to prove the value of a
theorem value_of_a (a : ℝ) (h : f_prime_at_neg1 a = 3) : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l633_63313


namespace NUMINAMATH_GPT_dodecagon_area_constraint_l633_63311

theorem dodecagon_area_constraint 
    (a : ℕ) -- side length of the square
    (N : ℕ) -- a large number with 2017 digits, breaking it down as 2 * (10^2017 - 1) / 9
    (hN : N = (2 * (10^2017 - 1)) / 9) 
    (H : ∃ n : ℕ, (n * n) = 3 * a^2 / 2) :
    False :=
by
    sorry

end NUMINAMATH_GPT_dodecagon_area_constraint_l633_63311


namespace NUMINAMATH_GPT_radius_inscribed_circle_ABC_l633_63376

noncomputable def radius_of_inscribed_circle (AB AC BC : ℝ) : ℝ :=
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  K / s

theorem radius_inscribed_circle_ABC (hAB : AB = 18) (hAC : AC = 18) (hBC : BC = 24) :
  radius_of_inscribed_circle 18 18 24 = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_GPT_radius_inscribed_circle_ABC_l633_63376


namespace NUMINAMATH_GPT_andy_late_l633_63368

theorem andy_late
  (school_start : ℕ := 480) -- 8:00 AM in minutes since midnight
  (normal_travel_time : ℕ := 30)
  (red_lights : ℕ := 4)
  (red_light_wait_time : ℕ := 3)
  (construction_wait_time : ℕ := 10)
  (departure_time : ℕ := 435) -- 7:15 AM in minutes since midnight
  : ((school_start - departure_time) < (normal_travel_time + (red_lights * red_light_wait_time) + construction_wait_time)) →
    school_start + (normal_travel_time + (red_lights * red_light_wait_time) + construction_wait_time - (school_start - departure_time)) = school_start + 7 :=
by
  -- This skips the proof part
  sorry

end NUMINAMATH_GPT_andy_late_l633_63368


namespace NUMINAMATH_GPT_pizzas_served_during_lunch_l633_63370

def total_pizzas : ℕ := 15
def dinner_pizzas : ℕ := 6

theorem pizzas_served_during_lunch :
  ∃ lunch_pizzas : ℕ, lunch_pizzas = total_pizzas - dinner_pizzas :=
by
  use 9
  exact rfl

end NUMINAMATH_GPT_pizzas_served_during_lunch_l633_63370


namespace NUMINAMATH_GPT_sum_of_two_numbers_l633_63338

theorem sum_of_two_numbers (x y : ℕ) (h : x = 11) (h1 : y = 3 * x + 11) : x + y = 55 := by
  sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l633_63338


namespace NUMINAMATH_GPT_simplify_expression_of_triangle_side_lengths_l633_63319

theorem simplify_expression_of_triangle_side_lengths
  (a b c : ℝ) 
  (h1 : a + b > c)
  (h2 : a + c > b)
  (h3 : b + c > a) :
  |a - b - c| - |c - a + b| = 0 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_of_triangle_side_lengths_l633_63319


namespace NUMINAMATH_GPT_sum_of_fifths_divisible_by_30_l633_63390

open BigOperators

theorem sum_of_fifths_divisible_by_30 {a : ℕ → ℕ} {n : ℕ} 
  (h : 30 ∣ ∑ i in Finset.range n, a i) : 
  30 ∣ ∑ i in Finset.range n, (a i) ^ 5 := 
by sorry

end NUMINAMATH_GPT_sum_of_fifths_divisible_by_30_l633_63390


namespace NUMINAMATH_GPT_calculate_expression_l633_63346

theorem calculate_expression :
  4 * Real.sqrt 24 * (Real.sqrt 6 / 8) / Real.sqrt 3 - 3 * Real.sqrt 3 = -Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l633_63346


namespace NUMINAMATH_GPT_problem_statement_l633_63351

def setA : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}
def setB : Set ℝ := {x : ℝ | x ≥ 2}
def setC (a : ℝ) : Set ℝ := {x : ℝ | 2 * x + a ≥ 0}

theorem problem_statement (a : ℝ):
  (setA ∩ setB = {x : ℝ | 2 ≤ x ∧ x < 3}) ∧ 
  (setA ∪ setB = {x : ℝ | x ≥ -1}) ∧ 
  (setB ⊆ setC a → a > -4) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l633_63351


namespace NUMINAMATH_GPT_axis_of_symmetry_parabola_l633_63302

theorem axis_of_symmetry_parabola : 
  ∀ (x : ℝ), 2 * (x - 3)^2 - 5 = 2 * (x - 3)^2 - 5 → (∃ h : ℝ, h = 3 ∧ ∀ x : ℝ, h = 3) :=
by
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_parabola_l633_63302


namespace NUMINAMATH_GPT_field_area_l633_63399

-- Define the given conditions and prove the area of the field
theorem field_area (x y : ℕ) 
  (h1 : 2*(x + 20) + 2*y = 2*(2*x + 2*y))
  (h2 : 2*x + 2*(2*y) = 2*x + 2*y + 18) : x * y = 99 := by 
{
  sorry
}

end NUMINAMATH_GPT_field_area_l633_63399


namespace NUMINAMATH_GPT_system_infinite_solutions_l633_63383

theorem system_infinite_solutions :
  ∃ (x y : ℚ), (3 * x - 4 * y = 5) ∧ (9 * x - 12 * y = 15) ↔ (3 * x - 4 * y = 5) :=
by
  sorry

end NUMINAMATH_GPT_system_infinite_solutions_l633_63383


namespace NUMINAMATH_GPT_range_of_MF_plus_MN_l633_63362

open Real

noncomputable def point_on_parabola (x y : ℝ) : Prop := y^2 = 4 * x

theorem range_of_MF_plus_MN (M : ℝ × ℝ) (N : ℝ × ℝ) (F : ℝ × ℝ) (hM : point_on_parabola M.1 M.2) (hN : N = (2, 2)) (hF : F = (1, 0)) :
  ∃ y : ℝ, y ≥ 3 ∧ ∀ MF MN : ℝ, MF = abs (M.1 - F.1) + abs (M.2 - F.2) ∧ MN = abs (M.1 - N.1) + abs (M.2 - N.2) → MF + MN = y :=
sorry

end NUMINAMATH_GPT_range_of_MF_plus_MN_l633_63362


namespace NUMINAMATH_GPT_problem_statement_l633_63375

-- Define the function f(x)
variable (f : ℝ → ℝ)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 4) = -f x
axiom increasing_on_0_2 : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 2 → f x < f y

-- Theorem to prove
theorem problem_statement : f (-10) < f 40 ∧ f 40 < f 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l633_63375


namespace NUMINAMATH_GPT_product_of_areas_eq_square_of_volume_l633_63379

theorem product_of_areas_eq_square_of_volume (w : ℝ) :
  let l := 2 * w
  let h := 3 * w
  let A_bottom := l * w
  let A_side := w * h
  let A_front := l * h
  let volume := l * w * h
  A_bottom * A_side * A_front = volume^2 :=
by
  sorry

end NUMINAMATH_GPT_product_of_areas_eq_square_of_volume_l633_63379


namespace NUMINAMATH_GPT_parallel_lines_if_and_only_if_perpendicular_lines_if_and_only_if_l633_63329

variable (m x y : ℝ)

def l1_eq : Prop := (3 - m) * x + 2 * m * y + 1 = 0
def l2_eq : Prop := 2 * m * x + 2 * y + m = 0

theorem parallel_lines_if_and_only_if : l1_eq m x y → l2_eq m x y → (m = -3/2) :=
by sorry

theorem perpendicular_lines_if_and_only_if : l1_eq m x y → l2_eq m x y → (m = 0 ∨ m = 5) :=
by sorry

end NUMINAMATH_GPT_parallel_lines_if_and_only_if_perpendicular_lines_if_and_only_if_l633_63329


namespace NUMINAMATH_GPT_length_of_bridge_is_correct_l633_63393

def length_of_bridge (train_length : ℕ) (train_speed_kmh : ℕ) (crossing_time_s : ℕ) : ℕ :=
  let train_speed_ms := (train_speed_kmh * 1000) / 3600
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

theorem length_of_bridge_is_correct : 
  length_of_bridge 170 45 30 = 205 :=
by
  -- we state the translation and prove here (proof omitted, just the structure is present)
  sorry

end NUMINAMATH_GPT_length_of_bridge_is_correct_l633_63393


namespace NUMINAMATH_GPT_speed_in_still_water_l633_63334

variable (v_m v_s : ℝ)

def swims_downstream (v_m v_s : ℝ) : Prop :=
  54 = (v_m + v_s) * 3

def swims_upstream (v_m v_s : ℝ) : Prop :=
  18 = (v_m - v_s) * 3

theorem speed_in_still_water : swims_downstream v_m v_s ∧ swims_upstream v_m v_s → v_m = 12 :=
by
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l633_63334


namespace NUMINAMATH_GPT_maximum_range_of_temperatures_l633_63307

variable (T1 T2 T3 T4 T5 : ℝ)

-- Given conditions
def average_condition : Prop := (T1 + T2 + T3 + T4 + T5) / 5 = 50
def lowest_temperature_condition : Prop := T1 = 45

-- Question to prove
def possible_maximum_range : Prop := T5 - T1 = 25

-- The final theorem statement
theorem maximum_range_of_temperatures 
  (h_avg : average_condition T1 T2 T3 T4 T5) 
  (h_lowest : lowest_temperature_condition T1) 
  : possible_maximum_range T1 T5 := by
  sorry

end NUMINAMATH_GPT_maximum_range_of_temperatures_l633_63307


namespace NUMINAMATH_GPT_multiply_by_5_l633_63341

theorem multiply_by_5 (x : ℤ) (h : x - 7 = 9) : x * 5 = 80 := by
  sorry

end NUMINAMATH_GPT_multiply_by_5_l633_63341


namespace NUMINAMATH_GPT_range_of_a_l633_63353

theorem range_of_a (x y a : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 / x + 1 / y = 1) : 
  (x + y + a > 0) ↔ (a > -3 - 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l633_63353


namespace NUMINAMATH_GPT_find_line_eq_l633_63337

theorem find_line_eq (l : ℝ → ℝ → Prop) :
  (∃ A B : ℝ × ℝ, l A.fst A.snd ∧ l B.fst B.snd ∧ ((A.fst + 1)^2 + (A.snd - 2)^2 = 100 ∧ (B.fst + 1)^2 + (B.snd - 2)^2 = 100)) ∧
  (∃ M : ℝ × ℝ, M = (-2, 3) ∧ (l M.fst M.snd)) →
  (∀ x y : ℝ, l x y ↔ x - y + 5 = 0) :=
by
  sorry

end NUMINAMATH_GPT_find_line_eq_l633_63337


namespace NUMINAMATH_GPT_locus_of_midpoint_of_chord_l633_63377

theorem locus_of_midpoint_of_chord
  (x y : ℝ)
  (hx : (x - 1)^2 + y^2 ≠ 0)
  : (x - 1) * (x - 1) + y * y = 1 :=
by
  sorry

end NUMINAMATH_GPT_locus_of_midpoint_of_chord_l633_63377


namespace NUMINAMATH_GPT_probabilities_equal_l633_63308

def roll := {n : ℕ // n ≥ 1 ∧ n ≤ 6}

def is_successful (r : roll) : Prop := r.val ≥ 3

def prob_successful : ℚ := 4 / 6

def prob_unsuccessful : ℚ := 1 - prob_successful

def prob_at_least_one_success_two_rolls : ℚ := 1 - (prob_unsuccessful ^ 2)

def prob_at_least_two_success_four_rolls : ℚ :=
  let zero_success := prob_unsuccessful ^ 4
  let one_success := 4 * (prob_unsuccessful ^ 3) * prob_successful
  1 - (zero_success + one_success)

theorem probabilities_equal :
  prob_at_least_one_success_two_rolls = prob_at_least_two_success_four_rolls := by
  sorry

end NUMINAMATH_GPT_probabilities_equal_l633_63308


namespace NUMINAMATH_GPT_max_distance_between_vertices_l633_63396

theorem max_distance_between_vertices (inner_perimeter outer_perimeter : ℕ) 
  (inner_perimeter_eq : inner_perimeter = 20) 
  (outer_perimeter_eq : outer_perimeter = 28) : 
  ∃ x y, x + y = 7 ∧ x^2 + y^2 = 25 ∧ (x^2 + (x + y)^2 = 65) :=
by
  sorry

end NUMINAMATH_GPT_max_distance_between_vertices_l633_63396


namespace NUMINAMATH_GPT_sum_of_other_endpoint_coords_l633_63326

theorem sum_of_other_endpoint_coords (x y : ℝ) (hx : (6 + x) / 2 = 5) (hy : (2 + y) / 2 = 7) : x + y = 16 := 
  sorry

end NUMINAMATH_GPT_sum_of_other_endpoint_coords_l633_63326


namespace NUMINAMATH_GPT_fibonacci_sequence_x_l633_63340

theorem fibonacci_sequence_x {a : ℕ → ℕ} 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 2) 
  (h3 : a 3 = 3) 
  (h_fib : ∀ n, n ≥ 3 → a (n + 1) = a n + a (n - 1)) : 
  a 5 = 8 := 
sorry

end NUMINAMATH_GPT_fibonacci_sequence_x_l633_63340


namespace NUMINAMATH_GPT_equivalent_statements_l633_63343
  
variables {A B : Prop}

theorem equivalent_statements :
  ((A ∧ B) → ¬ (A ∨ B)) ↔ ((A ∨ B) → ¬ (A ∧ B)) :=
sorry

end NUMINAMATH_GPT_equivalent_statements_l633_63343


namespace NUMINAMATH_GPT_range_of_f_l633_63381

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ |x + 1|

theorem range_of_f : Set.Ioo 0 1 ∪ {1} = {y : ℝ | ∃ x : ℝ, f x = y} :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l633_63381


namespace NUMINAMATH_GPT_find_g5_l633_63366

def g : ℤ → ℤ := sorry

axiom g1 : g 1 > 1
axiom g2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
sorry

end NUMINAMATH_GPT_find_g5_l633_63366


namespace NUMINAMATH_GPT_find_a_l633_63361

theorem find_a {S : ℕ → ℤ} (a : ℤ)
  (hS : ∀ n : ℕ, S n = 5 ^ (n + 1) + a) : a = -5 :=
sorry

end NUMINAMATH_GPT_find_a_l633_63361


namespace NUMINAMATH_GPT_length_of_hypotenuse_l633_63358

theorem length_of_hypotenuse (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 2450) (h2 : c = b + 10) (h3 : a^2 + b^2 = c^2) : c = 35 :=
by
  sorry

end NUMINAMATH_GPT_length_of_hypotenuse_l633_63358


namespace NUMINAMATH_GPT_relationship_x_y_l633_63324

theorem relationship_x_y (a b c : ℝ) (h₀ : a > b) (h₁ : b > c) (h₂ : x = Real.sqrt ((a - b) * (b - c))) (h₃ : y = (a - c) / 2) : 
  x ≤ y :=
by
  sorry

end NUMINAMATH_GPT_relationship_x_y_l633_63324


namespace NUMINAMATH_GPT_osmotic_pressure_independence_l633_63398

-- definitions for conditions
def osmotic_pressure_depends_on (osmotic_pressure protein_content Na_content Cl_content : Prop) : Prop :=
  (osmotic_pressure = protein_content ∧ osmotic_pressure = Na_content ∧ osmotic_pressure = Cl_content)

-- statement of the problem to be proved
theorem osmotic_pressure_independence 
  (osmotic_pressure : Prop) 
  (protein_content : Prop) 
  (Na_content : Prop) 
  (Cl_content : Prop) 
  (mw_plasma_protein : Prop)
  (dependence : osmotic_pressure_depends_on osmotic_pressure protein_content Na_content Cl_content) :
  ¬(osmotic_pressure = mw_plasma_protein) :=
sorry

end NUMINAMATH_GPT_osmotic_pressure_independence_l633_63398


namespace NUMINAMATH_GPT_relationship_abc_l633_63318

noncomputable def a : ℝ := 5 ^ (Real.log 3.4 / Real.log 3)
noncomputable def b : ℝ := 5 ^ (Real.log 3.6 / Real.log 3)
noncomputable def c : ℝ := (1 / 5) ^ (Real.log 0.5 / Real.log 3)

theorem relationship_abc : b > a ∧ a > c :=
by
  -- Assumptions derived from logarithmic properties.
  have h1 : Real.log 2 < Real.log 3.4 := sorry
  have h2 : Real.log 3.4 < Real.log 3.6 := sorry
  have h3 : Real.log 0.5 < 0 := sorry
  have h4 : Real.log 2 / Real.log 3 = Real.log 2 := sorry
  have h5 : Real.log 0.5 / Real.log 3 = -Real.log 2 := sorry

  -- Monotonicity of exponential function.
  apply And.intro
  { exact sorry }
  { exact sorry }

end NUMINAMATH_GPT_relationship_abc_l633_63318


namespace NUMINAMATH_GPT_zoo_animals_total_l633_63335

-- Conditions as definitions
def initial_animals : ℕ := 68
def gorillas_sent_away : ℕ := 6
def hippopotamus_adopted : ℕ := 1
def rhinos_taken_in : ℕ := 3
def lion_cubs_born : ℕ := 8
def meerkats_per_cub : ℕ := 2

-- Theorem to prove the resulting number of animals
theorem zoo_animals_total :
  (initial_animals - gorillas_sent_away + hippopotamus_adopted + rhinos_taken_in + lion_cubs_born + meerkats_per_cub * lion_cubs_born) = 90 :=
by 
  sorry

end NUMINAMATH_GPT_zoo_animals_total_l633_63335


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_intersections_l633_63372

theorem sufficient_but_not_necessary_condition_for_intersections
  (k : ℝ) (h : 0 < k ∧ k < 3) :
  ∃ x y : ℝ, (x - y - k = 0) ∧ ((x - 1)^2 + y^2 = 2) :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_intersections_l633_63372


namespace NUMINAMATH_GPT_sandy_found_additional_money_l633_63336

-- Define the initial amount of money Sandy had
def initial_amount : ℝ := 13.99

-- Define the cost of the shirt
def shirt_cost : ℝ := 12.14

-- Define the cost of the jacket
def jacket_cost : ℝ := 9.28

-- Define the remaining amount after buying the shirt
def remaining_after_shirt : ℝ := initial_amount - shirt_cost

-- Define the additional money found in Sandy's pocket
def additional_found_money : ℝ := jacket_cost - remaining_after_shirt

-- State the theorem to prove the amount of additional money found
theorem sandy_found_additional_money :
  additional_found_money = 11.13 :=
by sorry

end NUMINAMATH_GPT_sandy_found_additional_money_l633_63336


namespace NUMINAMATH_GPT_problem1_problem2_l633_63360

-- Problem I
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 12 = 0}
def B : Set ℝ := {-2, 4}

theorem problem1 (a : ℝ) (h : A a = B) : a = 2 :=
sorry

-- Problem II
def C (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}
def B' : Set ℝ := {-2, 4}

theorem problem2 (m : ℝ) (h : B' ∪ C m = B') : 
  m = -1/2 ∨ m = -1/4 ∨ m = 0 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l633_63360


namespace NUMINAMATH_GPT_star_property_l633_63300

-- Define the custom operation
def star (a b : ℝ) : ℝ := a^2 - b

-- Define the property to prove
theorem star_property (x y : ℝ) : star (x - y) (x + y) = x^2 - x - 2 * x * y + y^2 - y :=
by sorry

end NUMINAMATH_GPT_star_property_l633_63300


namespace NUMINAMATH_GPT_next_podcast_duration_l633_63373

def minutes_in_an_hour : ℕ := 60

def first_podcast_minutes : ℕ := 45
def second_podcast_minutes : ℕ := 2 * first_podcast_minutes
def third_podcast_minutes : ℕ := 105
def fourth_podcast_minutes : ℕ := 60

def total_podcast_minutes : ℕ := first_podcast_minutes + second_podcast_minutes + third_podcast_minutes + fourth_podcast_minutes

def drive_minutes : ℕ := 6 * minutes_in_an_hour

theorem next_podcast_duration :
  (drive_minutes - total_podcast_minutes) / minutes_in_an_hour = 1 :=
by
  sorry

end NUMINAMATH_GPT_next_podcast_duration_l633_63373


namespace NUMINAMATH_GPT_min_value_of_m_l633_63323

noncomputable def a_n (n : ℕ) : ℝ := 2 * 3^(n - 1)
noncomputable def b_n (n : ℕ) : ℝ := 2 * n - 9
noncomputable def c_n (n : ℕ) : ℝ := b_n n / a_n n

theorem min_value_of_m (m : ℝ) : (∀ n : ℕ, c_n n ≤ m) → m ≥ 1/162 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_m_l633_63323


namespace NUMINAMATH_GPT_moles_of_HCl_needed_l633_63349

theorem moles_of_HCl_needed : ∀ (moles_KOH : ℕ), moles_KOH = 2 →
  (moles_HCl : ℕ) → moles_HCl = 2 :=
by
  sorry

end NUMINAMATH_GPT_moles_of_HCl_needed_l633_63349


namespace NUMINAMATH_GPT_max_gas_tank_capacity_l633_63309

-- Definitions based on conditions
def start_gas : ℕ := 10
def gas_used_store : ℕ := 6
def gas_used_doctor : ℕ := 2
def refill_needed : ℕ := 10

-- Theorem statement based on the equivalence proof problem
theorem max_gas_tank_capacity : 
  start_gas - (gas_used_store + gas_used_doctor) + refill_needed = 12 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_max_gas_tank_capacity_l633_63309


namespace NUMINAMATH_GPT_number_is_2_point_5_l633_63365

theorem number_is_2_point_5 (x : ℝ) (h: x^2 + 50 = (x - 10)^2) : x = 2.5 := 
by
  sorry

end NUMINAMATH_GPT_number_is_2_point_5_l633_63365


namespace NUMINAMATH_GPT_value_of_f_at_2_l633_63386

def f (x : ℝ) : ℝ := x^2 + 2*x - 3

theorem value_of_f_at_2 : f 2 = 5 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_value_of_f_at_2_l633_63386


namespace NUMINAMATH_GPT_arrangement_count_l633_63345

-- Definitions corresponding to the given problem conditions
def numMathBooks : Nat := 3
def numPhysicsBooks : Nat := 2
def numChemistryBooks : Nat := 1
def totalArrangements : Nat := 2592

-- Statement of the theorem
theorem arrangement_count :
  ∃ (numM numP numC : Nat), 
    numM = 3 ∧ 
    numP = 2 ∧ 
    numC = 1 ∧ 
    (numM + numP + numC = 6) ∧ 
    allMathBooksAdjacent ∧ 
    physicsBooksNonAdjacent → 
    totalArrangements = 2592 :=
by
  sorry

end NUMINAMATH_GPT_arrangement_count_l633_63345


namespace NUMINAMATH_GPT_shorts_cost_l633_63363

theorem shorts_cost :
  let football_cost := 3.75
  let shoes_cost := 11.85
  let zachary_money := 10
  let additional_needed := 8
  ∃ S, football_cost + shoes_cost + S = zachary_money + additional_needed ∧ S = 2.40 :=
by
  sorry

end NUMINAMATH_GPT_shorts_cost_l633_63363


namespace NUMINAMATH_GPT_value_of_k_l633_63388

theorem value_of_k (k : ℝ) : 
  (∃ P Q R : ℝ × ℝ, P = (5, 12) ∧ Q = (0, k) ∧ dist (0, 0) P = dist (0, 0) Q + 5) → 
  k = 8 := 
by
  sorry

end NUMINAMATH_GPT_value_of_k_l633_63388


namespace NUMINAMATH_GPT_smallest_m_for_integral_roots_l633_63354

theorem smallest_m_for_integral_roots :
  ∃ (m : ℕ), (∃ (p q : ℤ), p * q = 30 ∧ m = 12 * (p + q)) ∧ m = 132 := by
  sorry

end NUMINAMATH_GPT_smallest_m_for_integral_roots_l633_63354


namespace NUMINAMATH_GPT_braking_distance_non_alcohol_reaction_time_after_alcohol_braking_distance_after_alcohol_increase_in_braking_distance_max_reaction_time_l633_63380

section braking_distance

variables {t k v s : ℝ}

-- Problem 1
theorem braking_distance_non_alcohol: 
  (t = 0.5) ∧ (v = 10) ∧ (k = 0.1) → (s = t * v + k * v^2) → (s = 15) :=
by intros; sorry

-- Problem 2a
theorem reaction_time_after_alcohol:
  (v = 15) ∧ (s = 52.5) ∧ (k = 0.1) → (s = t * v + k * v^2) → (t = 2) :=
by intros; sorry

-- Problem 2b
theorem braking_distance_after_alcohol:
  (t = 2) ∧ (v = 10) ∧ (k = 0.1) → (s = t * v + k * v^2) → (s = 30) :=
by intros; sorry

-- Problem 2c
theorem increase_in_braking_distance:
  (s_after = 30) ∧ (s_before = 15) → (diff = s_after - s_before) → (diff = 15) :=
by intros; sorry

-- Problem 3
theorem max_reaction_time:
  (v = 12) ∧ (k = 0.1) ∧ (s ≤ 42) → (s = t * v + k * v^2) → (t ≤ 2.3) :=
by intros; sorry

end braking_distance

end NUMINAMATH_GPT_braking_distance_non_alcohol_reaction_time_after_alcohol_braking_distance_after_alcohol_increase_in_braking_distance_max_reaction_time_l633_63380


namespace NUMINAMATH_GPT_probability_two_green_marbles_l633_63352

open Classical

section
variable (num_red num_green num_white num_blue : ℕ)
variable (total_marbles : ℕ := num_red + num_green + num_white + num_blue)

def probability_green_two_draws (num_green : ℕ) (total_marbles : ℕ) : ℚ :=
  (num_green / total_marbles : ℚ) * ((num_green - 1) / (total_marbles - 1))

theorem probability_two_green_marbles :
  probability_green_two_draws 4 (3 + 4 + 8 + 5) = 3 / 95 := by
  sorry
end

end NUMINAMATH_GPT_probability_two_green_marbles_l633_63352


namespace NUMINAMATH_GPT_quadratic_roots_interval_l633_63317

theorem quadratic_roots_interval (b : ℝ) :
  ∃ (x : ℝ), x^2 + b * x + 25 = 0 → b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_interval_l633_63317


namespace NUMINAMATH_GPT_range_k_fx_greater_than_ln_l633_63385

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem range_k (k : ℝ) : 0 ≤ k ∧ k ≤ Real.exp 1 ↔ ∀ x : ℝ, f x ≥ k * x := 
by 
  sorry

theorem fx_greater_than_ln (t : ℝ) (x : ℝ) : t ≤ 2 ∧ 0 < x → f x > t + Real.log x :=
by
  sorry

end NUMINAMATH_GPT_range_k_fx_greater_than_ln_l633_63385


namespace NUMINAMATH_GPT_ratio_third_to_others_l633_63325

-- Definitions of the heights
def H1 := 600
def H2 := 2 * H1
def H3 := 7200 - (H1 + H2)

-- Definition of the ratio to be proved
def ratio := H3 / (H1 + H2)

-- The theorem statement in Lean 4
theorem ratio_third_to_others : ratio = 3 := by
  have hH1 : H1 = 600 := rfl
  have hH2 : H2 = 2 * 600 := rfl
  have hH3 : H3 = 7200 - (600 + 1200) := rfl
  have h_total : 600 + 1200 + H3 = 7200 := sorry
  have h_ratio : (7200 - (600 + 1200)) / (600 + 1200) = 3 := by sorry
  sorry

end NUMINAMATH_GPT_ratio_third_to_others_l633_63325


namespace NUMINAMATH_GPT_number_of_ways_to_choose_teams_l633_63321

theorem number_of_ways_to_choose_teams : 
  ∃ (n : ℕ), n = Nat.choose 5 2 ∧ n = 10 :=
by
  have h : Nat.choose 5 2 = 10 := by sorry
  use 10
  exact ⟨h, rfl⟩

end NUMINAMATH_GPT_number_of_ways_to_choose_teams_l633_63321


namespace NUMINAMATH_GPT_james_total_money_l633_63312

section
-- Conditions
def number_of_bills : ℕ := 3
def value_of_each_bill : ℕ := 20
def initial_wallet_amount : ℕ := 75

-- Question:
-- What is the total amount of money James has now?
def total_value_of_bills : ℕ := number_of_bills * value_of_each_bill
def total_money_now : ℕ := initial_wallet_amount + total_value_of_bills

-- Theorem stating that he has $135 now.
theorem james_total_money : total_money_now = 135 := 
  by
    sorry
end

end NUMINAMATH_GPT_james_total_money_l633_63312


namespace NUMINAMATH_GPT_race_head_start_l633_63394

variable (vA vB L h : ℝ)
variable (hva_vb : vA = (16 / 15) * vB)

theorem race_head_start (hL_pos : L > 0) (hvB_pos : vB > 0) 
    (h_times_eq : (L / vA) = ((L - h) / vB)) : h = L / 16 :=
by
  sorry

end NUMINAMATH_GPT_race_head_start_l633_63394


namespace NUMINAMATH_GPT_acute_triangle_probability_correct_l633_63350

noncomputable def acute_triangle_probability : ℝ :=
  let l_cube_vol := 1
  let quarter_cone_vol := (1/4) * (1/3) * Real.pi * (1^2) * 1
  let total_unfavorable_vol := 3 * quarter_cone_vol
  let favorable_vol := l_cube_vol - total_unfavorable_vol
  favorable_vol / l_cube_vol

theorem acute_triangle_probability_correct : abs (acute_triangle_probability - 0.2146) < 0.0001 :=
  sorry

end NUMINAMATH_GPT_acute_triangle_probability_correct_l633_63350


namespace NUMINAMATH_GPT_correct_calculation_l633_63327

theorem correct_calculation :
  (- (4 + 2 / 3) - (1 + 5 / 6) - (- (18 + 1 / 2)) + (- (13 + 3 / 4))) = - (7 / 4) :=
by 
  sorry

end NUMINAMATH_GPT_correct_calculation_l633_63327


namespace NUMINAMATH_GPT_evaluate_expression_l633_63320

theorem evaluate_expression : 
  (16 = 2^4) → 
  (32 = 2^5) → 
  (16^24 / 32^12 = 8^12) :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l633_63320


namespace NUMINAMATH_GPT_no_2000_digit_perfect_square_with_1999_digits_of_5_l633_63397

theorem no_2000_digit_perfect_square_with_1999_digits_of_5 :
  ¬ (∃ n : ℕ,
      (Nat.digits 10 n).length = 2000 ∧
      ∃ k : ℕ, n = k * k ∧
      (Nat.digits 10 n).count 5 ≥ 1999) :=
sorry

end NUMINAMATH_GPT_no_2000_digit_perfect_square_with_1999_digits_of_5_l633_63397


namespace NUMINAMATH_GPT_andrey_wins_iff_irreducible_fraction_l633_63344

def is_irreducible_fraction (p : ℝ) : Prop :=
  ∃ m n : ℕ, p = m / 2^n ∧ gcd m (2^n) = 1

def can_reach_0_or_1 (p : ℝ) : Prop :=
  ∀ move : ℝ, ∃ dir : ℝ, (p + dir * move = 0 ∨ p + dir * move = 1)

theorem andrey_wins_iff_irreducible_fraction (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  (∃ move_sequence : ℕ → ℝ, ∀ n, can_reach_0_or_1 (move_sequence n)) ↔ is_irreducible_fraction p :=
sorry

end NUMINAMATH_GPT_andrey_wins_iff_irreducible_fraction_l633_63344


namespace NUMINAMATH_GPT_K_travel_time_40_miles_l633_63305

noncomputable def K_time (x : ℝ) : ℝ := 40 / x

theorem K_travel_time_40_miles (x : ℝ) (d : ℝ) (Δt : ℝ)
  (h1 : d = 40)
  (h2 : Δt = 1 / 3)
  (h3 : ∃ (Kmiles_r : ℝ) (Mmiles_r : ℝ), Kmiles_r = x ∧ Mmiles_r = x - 0.5)
  (h4 : ∃ (Ktime : ℝ) (Mtime : ℝ), Ktime = d / x ∧ Mtime = d / (x - 0.5) ∧ Mtime - Ktime = Δt) :
  K_time x = 5 := sorry

end NUMINAMATH_GPT_K_travel_time_40_miles_l633_63305


namespace NUMINAMATH_GPT_alpha_add_beta_eq_pi_div_two_l633_63367

open Real

theorem alpha_add_beta_eq_pi_div_two (α β : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : 0 < β ∧ β < π / 2) (h₃ : (sin α) ^ 4 / (cos β) ^ 2 + (cos α) ^ 4 / (sin β) ^ 2 = 1) :
  α + β = π / 2 :=
sorry

end NUMINAMATH_GPT_alpha_add_beta_eq_pi_div_two_l633_63367


namespace NUMINAMATH_GPT_mean_score_of_seniors_l633_63315

theorem mean_score_of_seniors 
  (s n : ℕ)
  (ms mn : ℝ)
  (h1 : s + n = 120)
  (h2 : n = 2 * s)
  (h3 : ms = 1.5 * mn)
  (h4 : (s : ℝ) * ms + (n : ℝ) * mn = 13200)
  : ms = 141.43 :=
by
  sorry

end NUMINAMATH_GPT_mean_score_of_seniors_l633_63315


namespace NUMINAMATH_GPT_assign_teachers_to_classes_l633_63392

-- Define the given conditions as variables and constants
theorem assign_teachers_to_classes :
  (∃ ways : ℕ, ways = 36) :=
by
  sorry

end NUMINAMATH_GPT_assign_teachers_to_classes_l633_63392


namespace NUMINAMATH_GPT_SUCCESSOR_arrangement_count_l633_63303

theorem SUCCESSOR_arrangement_count :
  (Nat.factorial 9) / (Nat.factorial 3 * Nat.factorial 2) = 30240 :=
by
  sorry

end NUMINAMATH_GPT_SUCCESSOR_arrangement_count_l633_63303


namespace NUMINAMATH_GPT_quadratic_solution_l633_63384

theorem quadratic_solution (m n x : ℝ)
  (h1 : (x - m)^2 + n = 0) 
  (h2 : ∃ (a b : ℝ), a ≠ b ∧ (x = a ∨ x = b) ∧ (a - m)^2 + n = 0 ∧ (b - m)^2 + n = 0
    ∧ (a = -1 ∨ a = 3) ∧ (b = -1 ∨ b = 3)) :
  x = -3 ∨ x = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_solution_l633_63384


namespace NUMINAMATH_GPT_binomial_coefficient_x5_l633_63378

theorem binomial_coefficient_x5 :
  let binomial_term (r : ℕ) : ℕ := Nat.choose 7 r * (21 - 4 * r)
  35 = binomial_term 4 :=
by
  sorry

end NUMINAMATH_GPT_binomial_coefficient_x5_l633_63378


namespace NUMINAMATH_GPT_sum_even_numbered_terms_l633_63371

variable (n : ℕ)

def a_n (n : ℕ) : ℕ := 2 * 3^(n-1)

def new_sequence (n : ℕ) : ℕ := a_n (2 * n)

def Sn (n : ℕ) : ℕ := (6 * (1 - 9^n)) / (1 - 9)

theorem sum_even_numbered_terms (n : ℕ) : Sn n = 3 * (9^n - 1) / 4 :=
by sorry

end NUMINAMATH_GPT_sum_even_numbered_terms_l633_63371


namespace NUMINAMATH_GPT_halfway_between_fractions_l633_63395

theorem halfway_between_fractions : ( (1/8 : ℚ) + (1/3 : ℚ) ) / 2 = 11 / 48 :=
by
  sorry

end NUMINAMATH_GPT_halfway_between_fractions_l633_63395


namespace NUMINAMATH_GPT_similar_triangles_iff_sides_proportional_l633_63332

theorem similar_triangles_iff_sides_proportional
  (a b c a1 b1 c1 : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < a1 ∧ 0 < b1 ∧ 0 < c1) :
  (Real.sqrt (a * a1) + Real.sqrt (b * b1) + Real.sqrt (c * c1) =
   Real.sqrt ((a + b + c) * (a1 + b1 + c1))) ↔
  (a / a1 = b / b1 ∧ b / b1 = c / c1) :=
by
  sorry

end NUMINAMATH_GPT_similar_triangles_iff_sides_proportional_l633_63332


namespace NUMINAMATH_GPT_ratio_boys_girls_l633_63359

theorem ratio_boys_girls
  (B G : ℕ)  -- Number of boys and girls
  (h_ratio : 75 * G = 80 * B)
  (h_total_no_scholarship : 100 * (3 * B + 4 * G) = 7772727272727272 * (B + G)) :
  B = 5 * G := sorry

end NUMINAMATH_GPT_ratio_boys_girls_l633_63359


namespace NUMINAMATH_GPT_polygon_sides_l633_63348

-- Define the conditions
def side_length : ℝ := 7
def perimeter : ℝ := 42

-- The statement to prove: number of sides is 6
theorem polygon_sides : (perimeter / side_length) = 6 := by
  sorry

end NUMINAMATH_GPT_polygon_sides_l633_63348


namespace NUMINAMATH_GPT_mult_base7_correct_l633_63391

def base7_to_base10 (n : ℕ) : ℕ :=
  -- assume conversion from base-7 to base-10 is already defined
  sorry 

def base10_to_base7 (n : ℕ) : ℕ :=
  -- assume conversion from base-10 to base-7 is already defined
  sorry

theorem mult_base7_correct : (base7_to_base10 325) * (base7_to_base10 4) = base7_to_base10 1656 :=
by
  sorry

end NUMINAMATH_GPT_mult_base7_correct_l633_63391


namespace NUMINAMATH_GPT_max_gcd_value_l633_63374

theorem max_gcd_value (n : ℕ) (hn : 0 < n) : ∃ k, k = gcd (13 * n + 4) (8 * n + 3) ∧ k <= 7 := sorry

end NUMINAMATH_GPT_max_gcd_value_l633_63374


namespace NUMINAMATH_GPT_complement_A_union_B_in_U_l633_63310

-- Define the universe set U
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | (x - 2) * (x + 1) ≤ 0}

-- Define set B
def B : Set ℝ := {x | 0 ≤ x ∧ x < 3}

-- Define the union of A and B
def A_union_B : Set ℝ := {x | (-1 ≤ x ∧ x < 3)}

-- Define the complement of A ∪ B in U
def C_U_A_union_B : Set ℝ := {x | x < -1 ∨ x ≥ 3}

-- Proof Statement
theorem complement_A_union_B_in_U :
  {x | x < -1 ∨ x ≥ 3} = {x | x ∈ U ∧ (x ∉ A_union_B)} :=
sorry

end NUMINAMATH_GPT_complement_A_union_B_in_U_l633_63310


namespace NUMINAMATH_GPT_scientific_notation_of_8_36_billion_l633_63387

theorem scientific_notation_of_8_36_billion : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 8.36 * 10^9 = a * 10^n := 
by
  use 8.36
  use 9
  simp
  sorry

end NUMINAMATH_GPT_scientific_notation_of_8_36_billion_l633_63387


namespace NUMINAMATH_GPT_roger_trays_l633_63369

theorem roger_trays (trays_per_trip trips trays_first_table : ℕ) 
  (h1 : trays_per_trip = 4) 
  (h2 : trips = 3) 
  (h3 : trays_first_table = 10) : 
  trays_per_trip * trips - trays_first_table = 2 :=
by
  -- Step proofs are omitted
  sorry

end NUMINAMATH_GPT_roger_trays_l633_63369
