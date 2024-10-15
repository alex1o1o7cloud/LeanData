import Mathlib

namespace NUMINAMATH_GPT_opposite_of_neg_11_l601_60189

-- Define the opposite (negative) of a number
def opposite (a : ℤ) : ℤ := -a

-- Prove that the opposite of -11 is 11
theorem opposite_of_neg_11 : opposite (-11) = 11 := 
by
  -- Proof not required, so using sorry as placeholder
  sorry

end NUMINAMATH_GPT_opposite_of_neg_11_l601_60189


namespace NUMINAMATH_GPT_catering_service_comparison_l601_60113

theorem catering_service_comparison :
  ∃ (x : ℕ), 150 + 18 * x > 250 + 15 * x ∧ (∀ y : ℕ, y < x -> (150 + 18 * y ≤ 250 + 15 * y)) ∧ x = 34 :=
sorry

end NUMINAMATH_GPT_catering_service_comparison_l601_60113


namespace NUMINAMATH_GPT_trees_distance_l601_60168

theorem trees_distance (num_trees : ℕ) (yard_length : ℕ) (trees_at_end : Prop) (tree_count : num_trees = 26) (yard_size : yard_length = 800) : 
  (yard_length / (num_trees - 1)) = 32 := 
by
  sorry

end NUMINAMATH_GPT_trees_distance_l601_60168


namespace NUMINAMATH_GPT_solve_sum_of_digits_eq_2018_l601_60153

def s (n : ℕ) : ℕ := (Nat.digits 10 n).sum

theorem solve_sum_of_digits_eq_2018 : ∃ n : ℕ, n + s n = 2018 := by
  sorry

end NUMINAMATH_GPT_solve_sum_of_digits_eq_2018_l601_60153


namespace NUMINAMATH_GPT_possible_values_of_b_l601_60184

theorem possible_values_of_b 
        (b : ℤ)
        (h : ∃ x : ℤ, (x ^ 3 + 2 * x ^ 2 + b * x + 8 = 0)) :
        b = -81 ∨ b = -26 ∨ b = -12 ∨ b = -6 ∨ b = 4 ∨ b = 9 ∨ b = 47 :=
  sorry

end NUMINAMATH_GPT_possible_values_of_b_l601_60184


namespace NUMINAMATH_GPT_min_value_function_l601_60183

theorem min_value_function (x : ℝ) (h : x > 0) : 
  ∃ y, y = (x^2 + x + 25) / x ∧ y ≥ 11 :=
sorry

end NUMINAMATH_GPT_min_value_function_l601_60183


namespace NUMINAMATH_GPT_intersection_A_B_l601_60146

open Set

def A : Set ℤ := {x | ∃ n : ℤ, x = 3 * n - 1}
def B : Set ℤ := {x | 0 < x ∧ x < 6}

theorem intersection_A_B : A ∩ B = {2, 5} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l601_60146


namespace NUMINAMATH_GPT_route_down_distance_l601_60114

theorem route_down_distance :
  ∀ (rate_up rate_down time_up time_down distance_up distance_down : ℝ),
    -- Conditions
    rate_down = 1.5 * rate_up →
    time_up = time_down →
    rate_up = 6 →
    time_up = 2 →
    distance_up = rate_up * time_up →
    distance_down = rate_down * time_down →
    -- Question: Prove the correct answer
    distance_down = 18 :=
by
  intros rate_up rate_down time_up time_down distance_up distance_down h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_route_down_distance_l601_60114


namespace NUMINAMATH_GPT_third_number_is_forty_four_l601_60101

theorem third_number_is_forty_four (a b c d e : ℕ) (h1 : a = e + 1) (h2 : b = e) 
  (h3 : c = e - 1) (h4 : d = e - 2) (h5 : e = e - 3) 
  (h6 : (a + b + c) / 3 = 45) (h7 : (c + d + e) / 3 = 43) : 
  c = 44 := 
sorry

end NUMINAMATH_GPT_third_number_is_forty_four_l601_60101


namespace NUMINAMATH_GPT_fifth_roll_six_probability_l601_60142
noncomputable def probability_fifth_roll_six : ℚ := sorry

theorem fifth_roll_six_probability :
  let fair_die_prob : ℚ := (1/6)^4
  let biased_die_6_prob : ℚ := (2/3)^3 * (1/15)
  let biased_die_3_prob : ℚ := (1/10)^3 * (1/2)
  let total_prob := (1/3) * fair_die_prob + (1/3) * biased_die_6_prob + (1/3) * biased_die_3_prob
  let normalized_biased_6_prob := (1/3) * biased_die_6_prob / total_prob
  let prob_of_fifth_six := normalized_biased_6_prob * (2/3)
  probability_fifth_roll_six = prob_of_fifth_six :=
sorry

end NUMINAMATH_GPT_fifth_roll_six_probability_l601_60142


namespace NUMINAMATH_GPT_solution_set_of_abs_inequality_l601_60157

theorem solution_set_of_abs_inequality :
  { x : ℝ | |x^2 - 2| < 2 } = { x : ℝ | -2 < x ∧ x < 0 ∨ 0 < x ∧ x < 2 } :=
sorry

end NUMINAMATH_GPT_solution_set_of_abs_inequality_l601_60157


namespace NUMINAMATH_GPT_find_number_l601_60111

def problem (x : ℝ) : Prop :=
  0.25 * x = 130 + 190

theorem find_number (x : ℝ) (h : problem x) : x = 1280 :=
by 
  sorry

end NUMINAMATH_GPT_find_number_l601_60111


namespace NUMINAMATH_GPT_lorenzo_cans_l601_60187

theorem lorenzo_cans (c : ℕ) (tacks_per_can : ℕ) (total_tacks : ℕ) (boards_tested : ℕ) (remaining_tacks : ℕ) :
  boards_tested = 120 →
  remaining_tacks = 30 →
  total_tacks = 450 →
  tacks_per_can = (boards_tested + remaining_tacks) →
  c * tacks_per_can = total_tacks →
  c = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_lorenzo_cans_l601_60187


namespace NUMINAMATH_GPT_set_equality_l601_60186

-- Define the universe U
def U := ℝ

-- Define the set M
def M := {x : ℝ | (x + 1) * (x - 2) ≤ 0}

-- Define the set N
def N := {x : ℝ | x > 1}

-- Define the set we want to prove is equal to the intersection of M and N
def target_set := {x : ℝ | 1 < x ∧ x ≤ 2}

theorem set_equality : target_set = M ∩ N := 
by sorry

end NUMINAMATH_GPT_set_equality_l601_60186


namespace NUMINAMATH_GPT_inequality_I_inequality_II_inequality_III_l601_60141

variable {a b c x y z : ℝ}

-- Assume the conditions
def conditions (a b c x y z : ℝ) : Prop :=
  x^2 < a ∧ y^2 < b ∧ z^2 < c

-- Prove the first inequality
theorem inequality_I (h : conditions a b c x y z) : x^2 * y^2 + y^2 * z^2 + z^2 * x^2 < a * b + b * c + c * a :=
sorry

-- Prove the second inequality
theorem inequality_II (h : conditions a b c x y z) : x^4 + y^4 + z^4 < a^2 + b^2 + c^2 :=
sorry

-- Prove the third inequality
theorem inequality_III (h : conditions a b c x y z) : x^2 * y^2 * z^2 < a * b * c :=
sorry

end NUMINAMATH_GPT_inequality_I_inequality_II_inequality_III_l601_60141


namespace NUMINAMATH_GPT_tax_is_one_l601_60180

-- Define costs
def cost_eggs : ℕ := 3
def cost_pancakes : ℕ := 2
def cost_cocoa : ℕ := 2

-- Initial order
def initial_eggs := 1
def initial_pancakes := 1
def initial_mugs_of_cocoa := 2

-- Additional order by Ben
def additional_pancakes := 1
def additional_mugs_of_cocoa := 1

-- Calculate costs
def initial_cost : ℕ := initial_eggs * cost_eggs + initial_pancakes * cost_pancakes + initial_mugs_of_cocoa * cost_cocoa
def additional_cost : ℕ := additional_pancakes * cost_pancakes + additional_mugs_of_cocoa * cost_cocoa
def total_cost_before_tax : ℕ := initial_cost + additional_cost

-- Payment and change
def total_paid : ℕ := 15
def change : ℕ := 1
def actual_payment : ℕ := total_paid - change

-- Calculate tax
def tax : ℕ := actual_payment - total_cost_before_tax

-- Prove that the tax is $1
theorem tax_is_one : tax = 1 :=
by
  sorry

end NUMINAMATH_GPT_tax_is_one_l601_60180


namespace NUMINAMATH_GPT_smallest_gcd_six_l601_60195

theorem smallest_gcd_six (x : ℕ) (hx1 : 70 ≤ x) (hx2 : x ≤ 90) (hx3 : Nat.gcd 24 x = 6) : x = 78 :=
by
  sorry

end NUMINAMATH_GPT_smallest_gcd_six_l601_60195


namespace NUMINAMATH_GPT_clairaut_equation_solution_l601_60126

open Real

noncomputable def clairaut_solution (f : ℝ → ℝ) (C : ℝ) : Prop :=
  (∀ x, f x = C * x + 1/(2 * C)) ∨ (∀ x, (f x)^2 = 2 * x)

theorem clairaut_equation_solution (y : ℝ → ℝ) :
  (∀ x, y x = x * (deriv y x) + 1/(2 * (deriv y x))) →
  ∃ C, clairaut_solution y C :=
sorry

end NUMINAMATH_GPT_clairaut_equation_solution_l601_60126


namespace NUMINAMATH_GPT_sergio_has_6_more_correct_answers_l601_60154

-- Define conditions
def total_questions : ℕ := 50
def incorrect_answers_sylvia : ℕ := total_questions / 5
def incorrect_answers_sergio : ℕ := 4

-- Calculate correct answers
def correct_answers_sylvia : ℕ := total_questions - incorrect_answers_sylvia
def correct_answers_sergio : ℕ := total_questions - incorrect_answers_sergio

-- The proof problem
theorem sergio_has_6_more_correct_answers :
  correct_answers_sergio - correct_answers_sylvia = 6 :=
by
  sorry

end NUMINAMATH_GPT_sergio_has_6_more_correct_answers_l601_60154


namespace NUMINAMATH_GPT_perimeter_hypotenuse_ratios_l601_60196

variable {x y : Real}
variable (h_pos_x : x > 0) (h_pos_y : y > 0)

theorem perimeter_hypotenuse_ratios
    (h_sides : (3 * x + 3 * y = (3 * x + 3 * y)) ∨ 
               (4 * x = (4 * x)) ∨
               (4 * y = (4 * y)))
    : 
    (∃ p : Real, p = 7 * (x + y) / (3 * (x + y)) ∨
                 p = 32 * y / (100 / 7 * y) ∨
                 p = 224 / 25 * y / 4 * y ∨ 
                 p = 7 / 3 ∨ 
                 p = 56 / 25) := by sorry

end NUMINAMATH_GPT_perimeter_hypotenuse_ratios_l601_60196


namespace NUMINAMATH_GPT_frustum_volume_correct_l601_60120

noncomputable def volume_of_frustum 
(base_edge_original_pyramid : ℝ) (height_original_pyramid : ℝ) 
(base_edge_smaller_pyramid : ℝ) (height_smaller_pyramid : ℝ) : ℝ :=
  let base_area_original := base_edge_original_pyramid ^ 2
  let volume_original := 1 / 3 * base_area_original * height_original_pyramid
  let similarity_ratio := base_edge_smaller_pyramid / base_edge_original_pyramid
  let volume_smaller := volume_original * (similarity_ratio ^ 3)
  volume_original - volume_smaller

theorem frustum_volume_correct 
(base_edge_original_pyramid : ℝ) (height_original_pyramid : ℝ) 
(base_edge_smaller_pyramid : ℝ) (height_smaller_pyramid : ℝ) 
(h_orig_base_edge : base_edge_original_pyramid = 16) 
(h_orig_height : height_original_pyramid = 10) 
(h_smaller_base_edge : base_edge_smaller_pyramid = 8) 
(h_smaller_height : height_smaller_pyramid = 5) : 
  volume_of_frustum base_edge_original_pyramid height_original_pyramid base_edge_smaller_pyramid height_smaller_pyramid = 746.66 :=
by 
  sorry

end NUMINAMATH_GPT_frustum_volume_correct_l601_60120


namespace NUMINAMATH_GPT_prove_product_reduced_difference_l601_60199

-- We are given two numbers x and y such that:
variable (x y : ℚ)
-- 1. The sum of the numbers is 6
axiom sum_eq_six : x + y = 6
-- 2. The quotient of the larger number by the smaller number is 6
axiom quotient_eq_six : x / y = 6

-- We need to prove that the product of these two numbers reduced by their difference is 6/49
theorem prove_product_reduced_difference (x y : ℚ) 
  (sum_eq_six : x + y = 6) (quotient_eq_six : x / y = 6) : 
  (x * y) - (x - y) = 6 / 49 := 
by
  sorry

end NUMINAMATH_GPT_prove_product_reduced_difference_l601_60199


namespace NUMINAMATH_GPT_planes_touch_three_spheres_count_l601_60129

-- Declare the conditions as definitions
def square_side_length : ℝ := 10
def radii : Fin 4 → ℝ
| 0 => 1
| 1 => 2
| 2 => 4
| 3 => 3

-- The proof problem statement
theorem planes_touch_three_spheres_count :
    ∃ (planes_that_touch_three_spheres : ℕ) (planes_that_intersect_fourth_sphere : ℕ),
    planes_that_touch_three_spheres = 26 ∧ planes_that_intersect_fourth_sphere = 8 := 
by
  -- sorry skips the proof
  sorry

end NUMINAMATH_GPT_planes_touch_three_spheres_count_l601_60129


namespace NUMINAMATH_GPT_maximum_k_value_l601_60125

noncomputable def max_value_k (a : ℝ) (b : ℝ) (k : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = 1 ∧ a^2 + b^2 ≥ k ∧ k = 1 / 2

theorem maximum_k_value (a b : ℝ) :
  (a > 0 ∧ b > 0 ∧ a + b = 1) → a^2 + b^2 ≥ 1 / 2 :=
by
  intro h
  obtain ⟨ha, hb, hab⟩ := h
  sorry

end NUMINAMATH_GPT_maximum_k_value_l601_60125


namespace NUMINAMATH_GPT_interest_rate_l601_60107

theorem interest_rate (P T R : ℝ) (SI CI : ℝ) (difference : ℝ)
  (hP : P = 1700)
  (hT : T = 1)
  (hdiff : difference = 4.25)
  (hSI : SI = P * R * T / 100)
  (hCI : CI = P * ((1 + R / 200)^2 - 1))
  (hDiff : CI - SI = difference) : 
  R = 10 := sorry

end NUMINAMATH_GPT_interest_rate_l601_60107


namespace NUMINAMATH_GPT_min_value_f_l601_60182

noncomputable def f (x : ℝ) : ℝ := x^3 + 9 * x + 81 / x^4

theorem min_value_f : ∃ x > 0, f x = 21 ∧ ∀ y > 0, f y ≥ 21 := by
  sorry

end NUMINAMATH_GPT_min_value_f_l601_60182


namespace NUMINAMATH_GPT_main_theorem_l601_60108

open Nat

-- Define the conditions
def conditions (p q n : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Odd p ∧ Odd q ∧ n > 1 ∧
  (q^(n+2) % p^n = 3^(n+2) % p^n) ∧ (p^(n+2) % q^n = 3^(n+2) % q^n)

-- Define the conclusion
def conclusion (p q n : ℕ) : Prop :=
  (p = 3 ∧ q = 3)

-- Define the main problem
theorem main_theorem : ∀ p q n : ℕ, conditions p q n → conclusion p q n :=
  by
    intros p q n h
    sorry

end NUMINAMATH_GPT_main_theorem_l601_60108


namespace NUMINAMATH_GPT_value_of_p_l601_60100

variable (m n p : ℝ)

-- The conditions from the problem
def first_point_on_line := m = (n / 6) - (2 / 5)
def second_point_on_line := m + p = ((n + 18) / 6) - (2 / 5)

-- The theorem to prove
theorem value_of_p (h1 : first_point_on_line m n) (h2 : second_point_on_line m n p) : p = 3 :=
  sorry

end NUMINAMATH_GPT_value_of_p_l601_60100


namespace NUMINAMATH_GPT_baker_weekend_hours_l601_60124

noncomputable def loaves_per_hour : ℕ := 5
noncomputable def ovens : ℕ := 4
noncomputable def weekday_hours : ℕ := 5
noncomputable def total_loaves : ℕ := 1740
noncomputable def weeks : ℕ := 3
noncomputable def weekday_days : ℕ := 5
noncomputable def weekend_days : ℕ := 2

theorem baker_weekend_hours :
  ((total_loaves - (weeks * weekday_days * weekday_hours * (loaves_per_hour * ovens))) / (weeks * (loaves_per_hour * ovens))) / weekend_days = 4 := by
  sorry

end NUMINAMATH_GPT_baker_weekend_hours_l601_60124


namespace NUMINAMATH_GPT_number_of_mowers_l601_60117

noncomputable section

def area_larger_meadow (A : ℝ) : ℝ := 2 * A

def team_half_day_work (K a : ℝ) : ℝ := (K * a) / 2

def team_remaining_larger_meadow (K a : ℝ) : ℝ := (K * a) / 2

def half_team_half_day_work (K a : ℝ) : ℝ := (K * a) / 4

def larger_meadow_area_leq_sum (K a A : ℝ) : Prop :=
  team_half_day_work K a + team_remaining_larger_meadow K a = 2 * A

def smaller_meadow_area_left (K a A : ℝ) : ℝ :=
  A - half_team_half_day_work K a

def one_mower_one_day_work_rate (K a : ℝ) : ℝ := (K * a) / 4

def eq_total_mowed_by_team (K a A : ℝ) : Prop :=
  larger_meadow_area_leq_sum K a A ∧ smaller_meadow_area_left K a A = (K * a) / 4

theorem number_of_mowers
  (K a A b : ℝ)
  (h1 : larger_meadow_area_leq_sum K a A)
  (h2 : smaller_meadow_area_left K a A = one_mower_one_day_work_rate K a)
  (h3 : one_mower_one_day_work_rate K a = b)
  (h4 : K * a = 2 * A)
  (h5 : 2 * A = 4 * b)
  : K = 8 :=
  sorry

end NUMINAMATH_GPT_number_of_mowers_l601_60117


namespace NUMINAMATH_GPT_verify_sum_of_fourth_powers_l601_60128

noncomputable def sum_of_squares (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def sum_of_fourth_powers (n : ℕ) : ℕ :=
  ((n * (n + 1) * (2 * n + 1) * (3 * n^2 + 3 * n - 1)) / 30)

noncomputable def square_of_sum (n : ℕ) : ℕ :=
  (n * (n + 1) / 2)^2

theorem verify_sum_of_fourth_powers (n : ℕ) :
  5 * sum_of_fourth_powers n = (4 * n + 2) * square_of_sum n - sum_of_squares n := 
  sorry

end NUMINAMATH_GPT_verify_sum_of_fourth_powers_l601_60128


namespace NUMINAMATH_GPT_exists_integer_in_seq_l601_60112

noncomputable def x_seq (x : ℕ → ℚ) := ∀ n : ℕ, x (n + 1) = x n + 1 / ⌊x n⌋

theorem exists_integer_in_seq {x : ℕ → ℚ} (h1 : 1 < x 1) (h2 : x_seq x) : 
  ∃ n : ℕ, ∃ k : ℤ, x n = k :=
sorry

end NUMINAMATH_GPT_exists_integer_in_seq_l601_60112


namespace NUMINAMATH_GPT_LittleJohnnyAnnualIncome_l601_60172

theorem LittleJohnnyAnnualIncome :
  ∀ (total_amount bank_amount bond_amount : ℝ) 
    (bank_interest bond_interest annual_income : ℝ),
    total_amount = 10000 →
    bank_amount = 6000 →
    bond_amount = 4000 →
    bank_interest = 0.05 →
    bond_interest = 0.09 →
    annual_income = bank_amount * bank_interest + bond_amount * bond_interest →
    annual_income = 660 :=
by
  intros total_amount bank_amount bond_amount bank_interest bond_interest annual_income 
  intros h_total_amount h_bank_amount h_bond_amount h_bank_interest h_bond_interest h_annual_income
  -- Proof is not required
  sorry

end NUMINAMATH_GPT_LittleJohnnyAnnualIncome_l601_60172


namespace NUMINAMATH_GPT_second_polygon_sides_l601_60174

-- Conditions as definitions
def perimeter_first_polygon (s : ℕ) := 50 * (3 * s)
def perimeter_second_polygon (N s : ℕ) := N * s
def same_perimeter (s N : ℕ) := perimeter_first_polygon s = perimeter_second_polygon N s

-- Theorem statement
theorem second_polygon_sides (s N : ℕ) :
  same_perimeter s N → N = 150 :=
by
  sorry

end NUMINAMATH_GPT_second_polygon_sides_l601_60174


namespace NUMINAMATH_GPT_bacteria_after_7_hours_l601_60155

noncomputable def bacteria_growth (initial : ℝ) (t : ℝ) (k : ℝ) : ℝ := initial * (10 * (Real.exp (k * t)))

noncomputable def solve_bacteria_problem : ℝ :=
let doubling_time := 1 / 60 -- In hours, since 60 minutes is 1 hour
-- Given that it doubles in 1 hour, we expect the growth to be such that y = initial * (2) in 1 hour.
let k := Real.log 2 -- Since when t = 1, we have 10 * e^(k * 1) = 2 * 10
bacteria_growth 10 7 k

theorem bacteria_after_7_hours :
  solve_bacteria_problem = 1280 :=
by
  sorry

end NUMINAMATH_GPT_bacteria_after_7_hours_l601_60155


namespace NUMINAMATH_GPT_find_xz_l601_60110

theorem find_xz (x y z : ℝ) (h1 : 2 * x + z = 15) (h2 : x - 2 * y = 8) : x + z = 15 :=
sorry

end NUMINAMATH_GPT_find_xz_l601_60110


namespace NUMINAMATH_GPT_shorter_piece_length_l601_60132

theorem shorter_piece_length (total_len : ℝ) (h1 : total_len = 60)
                            (short_len long_len : ℝ) (h2 : long_len = (1 / 2) * short_len)
                            (h3 : short_len + long_len = total_len) :
  short_len = 40 := 
  sorry

end NUMINAMATH_GPT_shorter_piece_length_l601_60132


namespace NUMINAMATH_GPT_weighted_average_remaining_two_l601_60136

theorem weighted_average_remaining_two (avg_10 : ℝ) (avg_2 : ℝ) (avg_3 : ℝ) (avg_3_next : ℝ) :
  avg_10 = 4.25 ∧ avg_2 = 3.4 ∧ avg_3 = 3.85 ∧ avg_3_next = 4.7 →
  (42.5 - (2 * 3.4 + 3 * 3.85 + 3 * 4.7)) / 2 = 5.025 :=
by
  intros
  sorry

end NUMINAMATH_GPT_weighted_average_remaining_two_l601_60136


namespace NUMINAMATH_GPT_totalCandlesInHouse_l601_60144

-- Definitions for the problem's conditions
def bedroomCandles : ℕ := 20
def livingRoomCandles : ℕ := bedroomCandles / 2
def donovanCandles : ℕ := 20

-- Problem to prove
theorem totalCandlesInHouse : bedroomCandles + livingRoomCandles + donovanCandles = 50 := by
  sorry

end NUMINAMATH_GPT_totalCandlesInHouse_l601_60144


namespace NUMINAMATH_GPT_units_digit_of_17_pow_28_l601_60122

theorem units_digit_of_17_pow_28 : ((17:ℕ) ^ 28) % 10 = 1 := by
  sorry

end NUMINAMATH_GPT_units_digit_of_17_pow_28_l601_60122


namespace NUMINAMATH_GPT_find_baseball_deck_price_l601_60150

variables (numberOfBasketballPacks : ℕ) (pricePerBasketballPack : ℝ) (numberOfBaseballDecks : ℕ)
           (totalMoney : ℝ) (changeReceived : ℝ) (totalSpent : ℝ) (spentOnBasketball : ℝ) (baseballDeckPrice : ℝ)

noncomputable def problem_conditions : Prop :=
  numberOfBasketballPacks = 2 ∧
  pricePerBasketballPack = 3 ∧
  numberOfBaseballDecks = 5 ∧
  totalMoney = 50 ∧
  changeReceived = 24 ∧
  totalSpent = totalMoney - changeReceived ∧
  spentOnBasketball = numberOfBasketballPacks * pricePerBasketballPack ∧
  totalSpent = spentOnBasketball + (numberOfBaseballDecks * baseballDeckPrice)

theorem find_baseball_deck_price (h : problem_conditions numberOfBasketballPacks pricePerBasketballPack numberOfBaseballDecks totalMoney changeReceived totalSpent spentOnBasketball baseballDeckPrice) :
  baseballDeckPrice = 4 :=
sorry

end NUMINAMATH_GPT_find_baseball_deck_price_l601_60150


namespace NUMINAMATH_GPT_trapezoid_sides_l601_60137

theorem trapezoid_sides (r kl: ℝ) (h1 : r = 5) (h2 : kl = 8) :
  ∃ (ab cd bc_ad : ℝ), ab = 5 ∧ cd = 20 ∧ bc_ad = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_sides_l601_60137


namespace NUMINAMATH_GPT_ratio_of_pieces_l601_60198

theorem ratio_of_pieces (total_length : ℝ) (shorter_piece : ℝ) : 
  total_length = 60 ∧ shorter_piece = 20 → shorter_piece / (total_length - shorter_piece) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_pieces_l601_60198


namespace NUMINAMATH_GPT_reflected_ray_equation_l601_60176

theorem reflected_ray_equation (x y : ℝ) (incident_ray : y = 2 * x + 1) (reflecting_line : y = x) :
  x - 2 * y - 1 = 0 :=
sorry

end NUMINAMATH_GPT_reflected_ray_equation_l601_60176


namespace NUMINAMATH_GPT_airplane_average_speed_l601_60156

-- Define the conditions
def miles_to_kilometers (miles : ℕ) : ℝ :=
  miles * 1.60934

def distance_miles : ℕ := 1584
def time_hours : ℕ := 24

-- Define the problem to prove
theorem airplane_average_speed : 
  (miles_to_kilometers distance_miles) / (time_hours : ℝ) = 106.24 :=
by
  sorry

end NUMINAMATH_GPT_airplane_average_speed_l601_60156


namespace NUMINAMATH_GPT_pipe_C_draining_rate_l601_60185

noncomputable def pipe_rate := 25

def tank_capacity := 2000
def pipe_A_rate := 200
def pipe_B_rate := 50
def pipe_C_duration_per_cycle := 2
def pipe_A_duration := 1
def pipe_B_duration := 2
def cycle_duration := pipe_A_duration + pipe_B_duration + pipe_C_duration_per_cycle
def total_time := 40
def number_of_cycles := total_time / cycle_duration
def water_filled_per_cycle := (pipe_A_rate * pipe_A_duration) + (pipe_B_rate * pipe_B_duration)
def total_water_filled := number_of_cycles * water_filled_per_cycle
def excess_water := total_water_filled - tank_capacity 
def pipe_C_rate := excess_water / (pipe_C_duration_per_cycle * number_of_cycles)

theorem pipe_C_draining_rate :
  pipe_C_rate = pipe_rate := by
  sorry

end NUMINAMATH_GPT_pipe_C_draining_rate_l601_60185


namespace NUMINAMATH_GPT_length_of_each_section_25_l601_60159

theorem length_of_each_section_25 (x : ℝ) 
  (h1 : ∃ x, x > 0)
  (h2 : 1000 / x = 15 / (1 / 2 * 3 / 4))
  : x = 25 := 
  sorry

end NUMINAMATH_GPT_length_of_each_section_25_l601_60159


namespace NUMINAMATH_GPT_circumradius_of_triangle_l601_60127

theorem circumradius_of_triangle (a b c : ℕ) (h₁ : a = 8) (h₂ : b = 6) (h₃ : c = 10) 
  (h₄ : a^2 + b^2 = c^2) : 
  (c : ℝ) / 2 = 5 := 
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_circumradius_of_triangle_l601_60127


namespace NUMINAMATH_GPT_leggings_needed_l601_60160

theorem leggings_needed (dogs : ℕ) (cats : ℕ) (dogs_legs : ℕ) (cats_legs : ℕ) (pair_of_leggings : ℕ) 
                        (hd : dogs = 4) (hc : cats = 3) (hl1 : dogs_legs = 4) (hl2 : cats_legs = 4) (lp : pair_of_leggings = 2)
                        : (dogs * dogs_legs + cats * cats_legs) / pair_of_leggings = 14 :=
by
  sorry

end NUMINAMATH_GPT_leggings_needed_l601_60160


namespace NUMINAMATH_GPT_divide_group_among_boats_l601_60103
noncomputable def number_of_ways_divide_group 
  (boatA_capacity : ℕ) 
  (boatB_capacity : ℕ) 
  (boatC_capacity : ℕ) 
  (num_adults : ℕ) 
  (num_children : ℕ) 
  (constraint : ∀ {boat : ℕ}, boat > 1 → num_children ≥ 1 → num_adults ≥ 1) : ℕ := 
    sorry

theorem divide_group_among_boats 
  (boatA_capacity : ℕ := 3) 
  (boatB_capacity : ℕ := 2) 
  (boatC_capacity : ℕ := 1) 
  (num_adults : ℕ := 2) 
  (num_children : ℕ := 2) 
  (constraint : ∀ {boat : ℕ}, boat > 1 → num_children ≥ 1 → num_adults ≥ 1) : 
  number_of_ways_divide_group boatA_capacity boatB_capacity boatC_capacity num_adults num_children constraint = 8 := 
sorry

end NUMINAMATH_GPT_divide_group_among_boats_l601_60103


namespace NUMINAMATH_GPT_Kim_min_score_for_target_l601_60165

noncomputable def Kim_exam_scores : List ℚ := [86, 82, 89]

theorem Kim_min_score_for_target :
  ∃ x : ℚ, ↑((Kim_exam_scores.sum + x) / (Kim_exam_scores.length + 1) ≥ (Kim_exam_scores.sum / Kim_exam_scores.length) + 2)
  ∧ x = 94 := sorry

end NUMINAMATH_GPT_Kim_min_score_for_target_l601_60165


namespace NUMINAMATH_GPT_a_16_value_l601_60170

-- Define the recurrence relation
def seq (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  match n with
  | 0       => 2
  | (n + 1) => (1 + a n) / (1 - a n)

-- State the theorem
theorem a_16_value :
  seq (a : ℕ → ℚ) 16 = -1/3 := 
sorry

end NUMINAMATH_GPT_a_16_value_l601_60170


namespace NUMINAMATH_GPT_A_eq_D_l601_60192

def A := {θ : ℝ | 0 < θ ∧ θ < 90}
def D := {θ : ℝ | 0 < θ ∧ θ < 90}

theorem A_eq_D : A = D :=
by
  sorry

end NUMINAMATH_GPT_A_eq_D_l601_60192


namespace NUMINAMATH_GPT_range_of_a_l601_60143

theorem range_of_a (a x : ℝ) (h_p : a - 4 < x ∧ x < a + 4) (h_q : (x - 2) * (x - 3) > 0) :
  a ≤ -2 ∨ a ≥ 7 :=
sorry

end NUMINAMATH_GPT_range_of_a_l601_60143


namespace NUMINAMATH_GPT_setB_is_correct_l601_60162

def setA : Set ℤ := {1, 0, -1, 2}
def setB : Set ℤ := { y | ∃ x ∈ setA, y = Int.natAbs x }

theorem setB_is_correct : setB = {0, 1, 2} := by
  sorry

end NUMINAMATH_GPT_setB_is_correct_l601_60162


namespace NUMINAMATH_GPT_entree_cost_l601_60161

/-- 
Prove that if the total cost is 23 and the entree costs 5 more than the dessert, 
then the cost of the entree is 14.
-/
theorem entree_cost (D : ℝ) (H1 : D + (D + 5) = 23) : D + 5 = 14 :=
by
  -- note: no proof required as per instructions
  sorry

end NUMINAMATH_GPT_entree_cost_l601_60161


namespace NUMINAMATH_GPT_johns_final_push_time_l601_60169

-- Definitions and assumptions
def john_speed : ℝ := 4.2
def steve_speed : ℝ := 3.8
def initial_gap : ℝ := 15
def final_gap : ℝ := 2

theorem johns_final_push_time :
  ∃ t : ℝ, john_speed * t = steve_speed * t + initial_gap + final_gap ∧ t = 42.5 :=
by
  sorry

end NUMINAMATH_GPT_johns_final_push_time_l601_60169


namespace NUMINAMATH_GPT_quadratic_second_root_l601_60158

noncomputable def second_root (p q : ℝ) : ℝ :=
  -2 * p / (p - 2)

theorem quadratic_second_root (p q : ℝ) (h1 : (p + q) * 1^2 + (p - q) * 1 + p * q = 0) :
  ∃ r : ℝ, r = second_root p q :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_second_root_l601_60158


namespace NUMINAMATH_GPT_value_of_y_l601_60130

variables (x y : ℝ)

theorem value_of_y (h1 : x - y = 16) (h2 : x + y = 8) : y = -4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_y_l601_60130


namespace NUMINAMATH_GPT_books_at_end_of_month_l601_60105

-- Definitions based on provided conditions
def initial_books : ℕ := 75
def loaned_books (x : ℕ) : ℕ := 40  -- Rounded from 39.99999999999999
def returned_books (x : ℕ) : ℕ := (loaned_books x * 70) / 100
def not_returned_books (x : ℕ) : ℕ := loaned_books x - returned_books x

-- The statement to be proved
theorem books_at_end_of_month (x : ℕ) : initial_books - not_returned_books x = 63 :=
by
  -- This will be filled in with the actual proof steps later
  sorry

end NUMINAMATH_GPT_books_at_end_of_month_l601_60105


namespace NUMINAMATH_GPT_find_integers_l601_60116

theorem find_integers (a b c : ℤ) (h1 : ∃ x : ℤ, a = 2 * x ∧ b = 5 * x ∧ c = 8 * x)
  (h2 : a + 6 = b / 3)
  (h3 : c - 10 = 5 * a / 4) :
  a = 36 ∧ b = 90 ∧ c = 144 :=
by
  sorry

end NUMINAMATH_GPT_find_integers_l601_60116


namespace NUMINAMATH_GPT_range_of_k_l601_60197

-- Given conditions
variables {k : ℝ} (h : ∃ (x y : ℝ), x^2 + k * y^2 = 2)

-- Theorem statement
theorem range_of_k : 0 < k ∧ k < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l601_60197


namespace NUMINAMATH_GPT_g_value_at_neg3_l601_60149

noncomputable def g : ℚ → ℚ := sorry

theorem g_value_at_neg3 (h : ∀ x : ℚ, x ≠ 0 → 4 * g (1 / x) + 3 * g x / x = 2 * x^2) : 
  g (-3) = 98 / 13 := 
sorry

end NUMINAMATH_GPT_g_value_at_neg3_l601_60149


namespace NUMINAMATH_GPT_tan_ratio_l601_60191

theorem tan_ratio (p q : Real) (hpq1 : Real.sin (p + q) = 0.6) (hpq2 : Real.sin (p - q) = 0.3) : 
  Real.tan p / Real.tan q = 3 := 
by
  sorry

end NUMINAMATH_GPT_tan_ratio_l601_60191


namespace NUMINAMATH_GPT_sarahs_trip_length_l601_60177

noncomputable def sarahsTrip (x : ℝ) : Prop :=
  x / 4 + 15 + x / 3 = x

theorem sarahs_trip_length : ∃ x : ℝ, sarahsTrip x ∧ x = 36 := by
  -- There should be a proof here, but it's omitted as per the task instructions
  sorry

end NUMINAMATH_GPT_sarahs_trip_length_l601_60177


namespace NUMINAMATH_GPT_james_final_sticker_count_l601_60152

-- Define the conditions
def initial_stickers := 478
def gift_stickers := 182
def given_away_stickers := 276

-- Define the correct answer
def final_stickers := 384

-- State the theorem
theorem james_final_sticker_count :
  initial_stickers + gift_stickers - given_away_stickers = final_stickers :=
by
  sorry

end NUMINAMATH_GPT_james_final_sticker_count_l601_60152


namespace NUMINAMATH_GPT_chemist_solution_l601_60133

theorem chemist_solution (x : ℝ) (h1 : ∃ x, 0 < x) 
  (h2 : x + 1 > 1) : 0.60 * x = 0.10 * (x + 1) → x = 0.2 := by
  sorry

end NUMINAMATH_GPT_chemist_solution_l601_60133


namespace NUMINAMATH_GPT_probability_of_blue_or_yellow_l601_60118

def num_red : ℕ := 6
def num_green : ℕ := 7
def num_yellow : ℕ := 8
def num_blue : ℕ := 9

def total_jelly_beans : ℕ := num_red + num_green + num_yellow + num_blue
def total_blue_or_yellow : ℕ := num_yellow + num_blue

theorem probability_of_blue_or_yellow (h : total_jelly_beans ≠ 0) : 
  (total_blue_or_yellow : ℚ) / (total_jelly_beans : ℚ) = 17 / 30 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_blue_or_yellow_l601_60118


namespace NUMINAMATH_GPT_pentagon_coloring_l601_60121

theorem pentagon_coloring (convex : Prop) (unequal_sides : Prop)
  (colors : Prop) (adjacent_diff_color : Prop) :
  ∃ n : ℕ, n = 30 := by
  -- Definitions for conditions (in practical terms, these might need to be more elaborate)
  let convex := true           -- Simplified representation
  let unequal_sides := true    -- Simplified representation
  let colors := true           -- Simplified representation
  let adjacent_diff_color := true -- Simplified representation
  
  -- Proof that the number of coloring methods is 30
  existsi 30
  sorry

end NUMINAMATH_GPT_pentagon_coloring_l601_60121


namespace NUMINAMATH_GPT_problems_per_page_l601_60188

theorem problems_per_page (total_problems finished_problems remaining_pages : Nat) (h1 : total_problems = 101) 
  (h2 : finished_problems = 47) (h3 : remaining_pages = 6) :
  (total_problems - finished_problems) / remaining_pages = 9 :=
by
  sorry

end NUMINAMATH_GPT_problems_per_page_l601_60188


namespace NUMINAMATH_GPT_initial_birds_count_l601_60179

theorem initial_birds_count (B : ℕ) :
  ∃ B, B + 4 = 5 + 2 → B = 3 :=
by
  sorry

end NUMINAMATH_GPT_initial_birds_count_l601_60179


namespace NUMINAMATH_GPT_find_P_l601_60106

variable (P : ℕ) 

-- Conditions
def cost_samosas : ℕ := 3 * 2
def cost_mango_lassi : ℕ := 2
def cost_per_pakora : ℕ := 3
def total_cost : ℕ := 25
def tip_rate : ℚ := 0.25

-- Total cost before tip
def total_cost_before_tip (P : ℕ) : ℕ := cost_samosas + cost_mango_lassi + cost_per_pakora * P

-- Total cost with tip
def total_cost_with_tip (P : ℕ) : ℚ := 
  (total_cost_before_tip P : ℚ) + (tip_rate * total_cost_before_tip P : ℚ)

-- Proof Goal
theorem find_P (h : total_cost_with_tip P = total_cost) : P = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_P_l601_60106


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_first_five_terms_l601_60104

theorem arithmetic_sequence_sum_first_five_terms:
  ∀ (a : ℕ → ℤ), a 2 = 1 → a 4 = 7 → (a 1 + a 5 = a 2 + a 4) → (5 * (a 1 + a 5) / 2 = 20) :=
by
  intros a h1 h2 h3
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_first_five_terms_l601_60104


namespace NUMINAMATH_GPT_final_score_proof_l601_60190

def final_score (bullseye_points : ℕ) (miss_points : ℕ) (half_bullseye_points : ℕ) : ℕ :=
  bullseye_points + miss_points + half_bullseye_points

theorem final_score_proof : final_score 50 0 25 = 75 :=
by
  -- Considering the given conditions
  -- bullseye_points = 50
  -- miss_points = 0
  -- half_bullseye_points = half of bullseye_points = 25
  -- Summing them up: 50 + 0 + 25 = 75
  sorry

end NUMINAMATH_GPT_final_score_proof_l601_60190


namespace NUMINAMATH_GPT_cyclic_quadrilateral_angle_D_l601_60115

theorem cyclic_quadrilateral_angle_D (A B C D : ℝ) (h1 : A + C = 180) (h2 : B + D = 180) (h3 : 3 * A = 4 * B) (h4 : 3 * A = 6 * C) : D = 100 :=
by
  sorry

end NUMINAMATH_GPT_cyclic_quadrilateral_angle_D_l601_60115


namespace NUMINAMATH_GPT_complement_union_l601_60119

def A : Set ℝ := { x | -1 < x ∧ x < 1 }
def B : Set ℝ := { x | x ≥ 1 }
def C (s : Set ℝ) : Set ℝ := { x | ¬ s x }

theorem complement_union :
  C (A ∪ B) = { x | x ≤ -1 } :=
by {
  sorry
}

end NUMINAMATH_GPT_complement_union_l601_60119


namespace NUMINAMATH_GPT_cows_in_group_l601_60109

theorem cows_in_group (D C : ℕ) 
  (h : 2 * D + 4 * C = 2 * (D + C) + 36) : 
  C = 18 :=
by
  sorry

end NUMINAMATH_GPT_cows_in_group_l601_60109


namespace NUMINAMATH_GPT_pages_read_on_Sunday_l601_60194

def total_pages : ℕ := 93
def pages_read_on_Saturday : ℕ := 30
def pages_remaining_after_Sunday : ℕ := 43

theorem pages_read_on_Sunday : total_pages - pages_read_on_Saturday - pages_remaining_after_Sunday = 20 := by
  sorry

end NUMINAMATH_GPT_pages_read_on_Sunday_l601_60194


namespace NUMINAMATH_GPT_equivalent_problem_l601_60148

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then x^2 else sorry

theorem equivalent_problem 
  (f_odd : ∀ x : ℝ, f (-x) = -f x)
  (f_periodic : ∀ x : ℝ, f (x + 2) = f x)
  (f_interval : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = x^2)
  : f (-3/2) + f 1 = 3/4 :=
sorry

end NUMINAMATH_GPT_equivalent_problem_l601_60148


namespace NUMINAMATH_GPT_gwen_spending_l601_60178

theorem gwen_spending : 
    ∀ (initial_amount spent remaining : ℕ), 
    initial_amount = 7 → remaining = 5 → initial_amount - remaining = 2 :=
by
    sorry

end NUMINAMATH_GPT_gwen_spending_l601_60178


namespace NUMINAMATH_GPT_joy_valid_rod_count_l601_60138

theorem joy_valid_rod_count : 
  let l := [4, 12, 21]
  let qs := [1, 2, 3, 5, 13, 20, 22, 40].filter (fun x => x != 4 ∧ x != 12 ∧ x != 21)
  (∀ d ∈ qs, 4 + 12 + 21 > d ∧ 4 + 12 + d > 21 ∧ 4 + 21 + d > 12 ∧ 12 + 21 + d > 4) → 
  ∃ n, n = 28 :=
by sorry

end NUMINAMATH_GPT_joy_valid_rod_count_l601_60138


namespace NUMINAMATH_GPT_sales_tax_difference_l601_60102

theorem sales_tax_difference
  (price : ℝ)
  (rate1 rate2 : ℝ)
  (h_rate1 : rate1 = 0.075)
  (h_rate2 : rate2 = 0.07)
  (h_price : price = 30) :
  (price * rate1 - price * rate2 = 0.15) :=
by
  sorry

end NUMINAMATH_GPT_sales_tax_difference_l601_60102


namespace NUMINAMATH_GPT_find_y_l601_60131

-- Definitions of the given conditions
def angle_ABC_is_straight_line := true  -- This is to ensure the angle is a straight line.
def angle_ABD_is_exterior_of_triangle_BCD := true -- This is to ensure ABD is an exterior angle.
def angle_ABD : ℝ := 118
def angle_BCD : ℝ := 82

-- Theorem to prove y = 36 given the conditions
theorem find_y (A B C D : Type) (y : ℝ) 
    (h1 : angle_ABC_is_straight_line)
    (h2 : angle_ABD_is_exterior_of_triangle_BCD)
    (h3 : angle_ABD = 118)
    (h4 : angle_BCD = 82) : 
            y = 36 :=
  by
  sorry

end NUMINAMATH_GPT_find_y_l601_60131


namespace NUMINAMATH_GPT_min_ab_value_l601_60140

theorem min_ab_value 
  (a b : ℝ) 
  (hab_pos : a * b > 0)
  (collinear_condition : 2 * a + 2 * b + a * b = 0) :
  a * b ≥ 16 := 
sorry

end NUMINAMATH_GPT_min_ab_value_l601_60140


namespace NUMINAMATH_GPT_how_many_cakes_each_friend_ate_l601_60151

-- Definitions pertaining to the problem conditions
def crackers : ℕ := 29
def cakes : ℕ := 30
def friends : ℕ := 2

-- The main theorem statement we aim to prove
theorem how_many_cakes_each_friend_ate 
  (h1 : crackers = 29)
  (h2 : cakes = 30)
  (h3 : friends = 2) : 
  (cakes / friends = 15) :=
by
  sorry

end NUMINAMATH_GPT_how_many_cakes_each_friend_ate_l601_60151


namespace NUMINAMATH_GPT_range_of_a_l601_60171

variable {x a : ℝ}

def A : Set ℝ := {x | x ≤ -3 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x ≤ a - 1}

theorem range_of_a (A_union_B_R : A ∪ B a = Set.univ) : a ∈ Set.Ici 3 :=
  sorry

end NUMINAMATH_GPT_range_of_a_l601_60171


namespace NUMINAMATH_GPT_sin_225_eq_neg_sqrt_two_div_two_l601_60123

theorem sin_225_eq_neg_sqrt_two_div_two :
  Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_225_eq_neg_sqrt_two_div_two_l601_60123


namespace NUMINAMATH_GPT_slower_speed_is_correct_l601_60139

/-- 
A person walks at 14 km/hr instead of a slower speed, 
and as a result, he would have walked 20 km more. 
The actual distance travelled by him is 50 km. 
What is the slower speed he usually walks at?
-/
theorem slower_speed_is_correct :
    ∃ x : ℝ, (14 * (50 / 14) - (x * (30 / x))) = 20 ∧ x = 8.4 :=
by
  sorry

end NUMINAMATH_GPT_slower_speed_is_correct_l601_60139


namespace NUMINAMATH_GPT_sequence_solution_l601_60166

theorem sequence_solution
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_a1 : a 1 = 10)
  (h_b1 : b 1 = 10)
  (h_recur_a : ∀ n : ℕ, a (n + 1) = 1 / (a n * b n))
  (h_recur_b : ∀ n : ℕ, b (n + 1) = (a n)^4 * b n) :
  (∀ n : ℕ, n > 0 → a n = 10^((2 - 3 * n) * (-1 : ℝ)^n) ∧ b n = 10^((6 * n - 7) * (-1 : ℝ)^n)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_solution_l601_60166


namespace NUMINAMATH_GPT_problem_statement_l601_60181

theorem problem_statement (x : ℤ) (h₁ : (x - 5) / 7 = 7) : (x - 24) / 10 = 3 := 
sorry

end NUMINAMATH_GPT_problem_statement_l601_60181


namespace NUMINAMATH_GPT_solve_mod_equation_l601_60193

theorem solve_mod_equation (x : ℤ) (h : 10 * x + 3 ≡ 7 [ZMOD 18]) : x ≡ 4 [ZMOD 9] :=
sorry

end NUMINAMATH_GPT_solve_mod_equation_l601_60193


namespace NUMINAMATH_GPT_total_cost_of_phone_l601_60167

theorem total_cost_of_phone (cost_per_phone : ℕ) (monthly_cost : ℕ) (months : ℕ) (phone_count : ℕ) :
  cost_per_phone = 2 → monthly_cost = 7 → months = 4 → phone_count = 1 →
  (cost_per_phone * phone_count + monthly_cost * months) = 30 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_total_cost_of_phone_l601_60167


namespace NUMINAMATH_GPT_sum_of_coefficients_l601_60147

theorem sum_of_coefficients :
  ∃ (a b c d e : ℤ), (512 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) ∧ (a + b + c + d + e = 60) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l601_60147


namespace NUMINAMATH_GPT_taimour_time_to_paint_alone_l601_60135

theorem taimour_time_to_paint_alone (T : ℝ) (h1 : Jamshid_time = T / 2)
  (h2 : (1 / T + 1 / (T / 2)) = 1 / 3) : T = 9 :=
sorry

end NUMINAMATH_GPT_taimour_time_to_paint_alone_l601_60135


namespace NUMINAMATH_GPT_garden_length_increase_l601_60164

variable (L W : ℝ)  -- Original length and width
variable (X : ℝ)    -- Percentage increase in length

theorem garden_length_increase :
  (1 + X / 100) * 0.8 = 1.1199999999999999 → X = 40 :=
by
  sorry

end NUMINAMATH_GPT_garden_length_increase_l601_60164


namespace NUMINAMATH_GPT_tan_alpha_eq_l601_60163

theorem tan_alpha_eq : ∀ (α : ℝ),
  (Real.tan (α - (5 * Real.pi / 4)) = 1 / 5) →
  Real.tan α = 3 / 2 :=
by
  intro α h
  sorry

end NUMINAMATH_GPT_tan_alpha_eq_l601_60163


namespace NUMINAMATH_GPT_distinct_arrangements_balloon_l601_60175

theorem distinct_arrangements_balloon : 
  let n := 7
  let r1 := 2
  let r2 := 2
  (Nat.factorial n) / ((Nat.factorial r1) * (Nat.factorial r2)) = 1260 :=
by
  sorry

end NUMINAMATH_GPT_distinct_arrangements_balloon_l601_60175


namespace NUMINAMATH_GPT_construct_3x3x3_cube_l601_60145

theorem construct_3x3x3_cube :
  ∃ (cubes_1x2x2 : Finset (Set (Fin 3 × Fin 3 × Fin 3))),
  ∃ (cubes_1x1x1 : Finset (Fin 3 × Fin 3 × Fin 3)),
  cubes_1x2x2.card = 6 ∧ 
  cubes_1x1x1.card = 3 ∧ 
  (∀ c ∈ cubes_1x2x2, ∃ a b : Fin 3, ∀ x, x = (a, b, 0) ∨ x = (a, b, 1) ∨ x = (a, b, 2)) ∧
  (∀ c ∈ cubes_1x1x1, ∃ a b c : Fin 3, ∀ x, x = (a, b, c)) :=
sorry

end NUMINAMATH_GPT_construct_3x3x3_cube_l601_60145


namespace NUMINAMATH_GPT_common_factor_polynomials_l601_60173

-- Define the two polynomials
def poly1 (x y z : ℝ) := 3 * x^2 * y^3 * z + 9 * x^3 * y^3 * z
def poly2 (x y z : ℝ) := 6 * x^4 * y * z^2

-- Define the common factor
def common_factor (x y z : ℝ) := 3 * x^2 * y * z

-- The statement to prove that the common factor of poly1 and poly2 is 3 * x^2 * y * z
theorem common_factor_polynomials (x y z : ℝ) :
  ∃ (f : ℝ → ℝ → ℝ → ℝ), (poly1 x y z) = (f x y z) * (common_factor x y z) ∧
                          (poly2 x y z) = (f x y z) * (common_factor x y z) :=
sorry

end NUMINAMATH_GPT_common_factor_polynomials_l601_60173


namespace NUMINAMATH_GPT_solve_for_m_l601_60134

namespace ProofProblem

def f (x m : ℝ) : ℝ := x^2 - 3*x + m
def g (x m : ℝ) : ℝ := x^2 - 3*x + 5*m

theorem solve_for_m (m : ℝ) : 3 * f 3 m = g 3 m → m = 0 := by
  sorry

end ProofProblem

end NUMINAMATH_GPT_solve_for_m_l601_60134
