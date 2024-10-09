import Mathlib

namespace calculate_sum_of_powers_l502_50277

theorem calculate_sum_of_powers :
  (6^2 - 3^2)^4 + (7^2 - 2^2)^4 = 4632066 :=
by
  sorry

end calculate_sum_of_powers_l502_50277


namespace perimeter_difference_l502_50287

-- Definitions as per conditions
def plywood_width : ℕ := 6
def plywood_height : ℕ := 9
def rectangles_count : ℕ := 6

-- The perimeter difference to be proved
theorem perimeter_difference : 
  ∃ (max_perimeter min_perimeter : ℕ), 
  max_perimeter = 22 ∧ min_perimeter = 12 ∧ (max_perimeter - min_perimeter = 10) :=
by
  sorry

end perimeter_difference_l502_50287


namespace value_of_x_l502_50294

theorem value_of_x (x y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 :=
by
  sorry

end value_of_x_l502_50294


namespace remainder_property_l502_50230

theorem remainder_property (a : ℤ) (h : ∃ k : ℤ, a = 45 * k + 36) :
  ∃ n : ℤ, a = 45 * n + 36 :=
by {
  sorry
}

end remainder_property_l502_50230


namespace least_number_to_subtract_l502_50242

theorem least_number_to_subtract (n : ℕ) (k : ℕ) (r : ℕ) (h : n = 3674958423) (div : k = 47) (rem : r = 30) :
  (n % k = r) → 3674958423 % 47 = 30 :=
by
  sorry

end least_number_to_subtract_l502_50242


namespace women_left_room_is_3_l502_50247

-- Definitions and conditions
variables (M W x : ℕ)
variables (ratio : M * 5 = W * 4) 
variables (men_entered : M + 2 = 14) 
variables (women_left : 2 * (W - x) = 24)

-- Theorem statement
theorem women_left_room_is_3 
  (ratio : M * 5 = W * 4) 
  (men_entered : M + 2 = 14) 
  (women_left : 2 * (W - x) = 24) : 
  x = 3 :=
sorry

end women_left_room_is_3_l502_50247


namespace determinant_calculation_l502_50257

variable {R : Type*} [CommRing R]

def matrix_example (a b c : R) : Matrix (Fin 3) (Fin 3) R :=
  ![![1, a, b], ![1, a + b, b + c], ![1, a, a + c]]

theorem determinant_calculation (a b c : R) :
  (matrix_example a b c).det = ab + b^2 + bc :=
by sorry

end determinant_calculation_l502_50257


namespace fill_time_is_13_seconds_l502_50259

-- Define the given conditions as constants
def flow_rate_in (t : ℝ) : ℝ := 24 * t -- 24 gallons/second
def leak_rate (t : ℝ) : ℝ := 4 * t -- 4 gallons/second
def basin_capacity : ℝ := 260 -- 260 gallons

-- Main theorem to be proven
theorem fill_time_is_13_seconds : 
  ∀ t : ℝ, (flow_rate_in t - leak_rate t) * (13) = basin_capacity := 
sorry

end fill_time_is_13_seconds_l502_50259


namespace alex_piles_of_jelly_beans_l502_50214

theorem alex_piles_of_jelly_beans : 
  ∀ (initial_weight eaten weight_per_pile remaining_weight piles : ℕ),
    initial_weight = 36 →
    eaten = 6 →
    weight_per_pile = 10 →
    remaining_weight = initial_weight - eaten →
    piles = remaining_weight / weight_per_pile →
    piles = 3 :=
by
  intros initial_weight eaten weight_per_pile remaining_weight piles h_init h_eat h_wpile h_remaining h_piles
  sorry

end alex_piles_of_jelly_beans_l502_50214


namespace total_digits_in_book_l502_50236

open Nat

theorem total_digits_in_book (n : Nat) (h : n = 10000) : 
    let pages_1_9 := 9
    let pages_10_99 := 90 * 2
    let pages_100_999 := 900 * 3
    let pages_1000_9999 := 9000 * 4
    let page_10000 := 5
    pages_1_9 + pages_10_99 + pages_100_999 + pages_1000_9999 + page_10000 = 38894 :=
by
    sorry

end total_digits_in_book_l502_50236


namespace find_annual_interest_rate_l502_50213

/-- 
  Given:
  - Principal P = 10000
  - Interest I = 450
  - Time period T = 0.75 years

  Prove that the annual interest rate is 0.08.
-/
theorem find_annual_interest_rate (P I : ℝ) (T : ℝ) (hP : P = 10000) (hI : I = 450) (hT : T = 0.75) : 
  (I / (P * T) / T) = 0.08 :=
by
  sorry

end find_annual_interest_rate_l502_50213


namespace custom_op_difference_l502_50260

def custom_op (x y : ℕ) : ℕ := x * y - (x + y)

theorem custom_op_difference : custom_op 7 4 - custom_op 4 7 = 0 :=
by
  sorry

end custom_op_difference_l502_50260


namespace determine_p_q_l502_50248

theorem determine_p_q (r1 r2 p q : ℝ) (h1 : r1 + r2 = 5) (h2 : r1 * r2 = 6) (h3 : r1^2 + r2^2 = -p) (h4 : r1^2 * r2^2 = q) : p = -13 ∧ q = 36 :=
by
  sorry

end determine_p_q_l502_50248


namespace solution_range_for_m_l502_50258

theorem solution_range_for_m (x m : ℝ) (h₁ : 2 * x - 1 > 3 * (x - 2)) (h₂ : x < m) : m ≥ 5 :=
by {
  sorry
}

end solution_range_for_m_l502_50258


namespace simplify_and_evaluate_expression_l502_50267

theorem simplify_and_evaluate_expression :
  let x := -1
  let y := Real.sqrt 2
  (x + y) * (x - y) - (4 * x^3 * y - 8 * x * y^3) / (2 * x * y) = 5 :=
by
  let x := -1
  let y := Real.sqrt 2
  sorry

end simplify_and_evaluate_expression_l502_50267


namespace average_number_of_carnations_l502_50249

-- Define the number of carnations in each bouquet
def n1 : ℤ := 9
def n2 : ℤ := 23
def n3 : ℤ := 13
def n4 : ℤ := 36
def n5 : ℤ := 28
def n6 : ℤ := 45

-- Define the number of bouquets
def number_of_bouquets : ℤ := 6

-- Prove that the average number of carnations in the bouquets is 25.67
theorem average_number_of_carnations :
  ((n1 + n2 + n3 + n4 + n5 + n6) : ℚ) / (number_of_bouquets : ℚ) = 25.67 := 
by
  sorry

end average_number_of_carnations_l502_50249


namespace kabob_cubes_calculation_l502_50229

-- Define the properties of a slab of beef
def cubes_per_slab := 80
def cost_per_slab := 25

-- Define Simon's usage and expenditure
def simons_budget := 50
def number_of_kabob_sticks := 40

-- Auxiliary calculations for proofs (making noncomputable if necessary)
noncomputable def cost_per_cube := cost_per_slab / cubes_per_slab
noncomputable def cubes_per_kabob_stick := (2 * cubes_per_slab) / number_of_kabob_sticks

-- The theorem we want to prove
theorem kabob_cubes_calculation :
  cubes_per_kabob_stick = 4 := by
  sorry

end kabob_cubes_calculation_l502_50229


namespace subway_speed_increase_l502_50256

theorem subway_speed_increase (s : ℝ) (h₀ : 0 ≤ s) (h₁ : s ≤ 7) : 
  (s^2 + 2 * s = 63) ↔ (s = 7) :=
by
  sorry 

end subway_speed_increase_l502_50256


namespace part_I_solution_set_part_II_prove_inequality_l502_50265

-- Definition for part (I)
def f (x: ℝ) := |x - 2|
def g (x: ℝ) := 4 - |x - 1|

-- Theorem for part (I)
theorem part_I_solution_set :
  {x : ℝ | f x ≥ g x} = {x : ℝ | x ≤ -1/2} ∪ {x : ℝ | x ≥ 7/2} :=
by sorry

-- Definition for part (II)
def satisfiable_range (a: ℝ) := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
def density_equation (m n a: ℝ) := (1 / m) + (1 / (2 * n)) = a

-- Theorem for part (II)
theorem part_II_prove_inequality (m n: ℝ) (hm: 0 < m) (hn: 0 < n) 
  (a: ℝ) (h_a: satisfiable_range a = {x : ℝ | abs (x - a) ≤ 1}) (h_density: density_equation m n a) :
  m + 2 * n ≥ 4 :=
by sorry

end part_I_solution_set_part_II_prove_inequality_l502_50265


namespace solution_set_of_inequality_l502_50253

theorem solution_set_of_inequality :
  {x : ℝ | (x - 1) / (x^2 - x - 6) ≥ 0} = {x : ℝ | (-2 < x ∧ x ≤ 1) ∨ (3 < x)} := 
sorry

end solution_set_of_inequality_l502_50253


namespace expression_even_nat_l502_50238

theorem expression_even_nat (m n : ℕ) : 
  2 ∣ (5 * m + n + 1) * (3 * m - n + 4) := 
sorry

end expression_even_nat_l502_50238


namespace parallel_line_through_point_l502_50210

theorem parallel_line_through_point :
  ∃ c : ℝ, ∀ x y : ℝ, (x = -1) → (y = 3) → (x - 2*y + 3 = 0) → (x - 2*y + c = 0) :=
sorry

end parallel_line_through_point_l502_50210


namespace trisha_interest_l502_50211

noncomputable def total_amount (P : ℝ) (r : ℝ) (D : ℝ) (t : ℕ) : ℝ :=
  let rec compute (n : ℕ) (A : ℝ) :=
    if n = 0 then A
    else let A_next := A * (1 + r) + D
         compute (n - 1) A_next
  compute t P

noncomputable def total_deposits (D : ℝ) (t : ℕ) : ℝ :=
  D * t

noncomputable def total_interest (P : ℝ) (r : ℝ) (D : ℝ) (t : ℕ) : ℝ :=
  total_amount P r D t - P - total_deposits D t

theorem trisha_interest :
  total_interest 2000 0.05 300 5 = 710.25 :=
by
  sorry

end trisha_interest_l502_50211


namespace Beth_crayons_proof_l502_50243

def Beth_packs_of_crayons (packs_crayons : ℕ) (total_crayons extra_crayons : ℕ) : ℕ :=
  total_crayons - extra_crayons

theorem Beth_crayons_proof
  (packs_crayons : ℕ)
  (each_pack_contains total_crayons extra_crayons : ℕ)
  (h_each_pack : each_pack_contains = 10) 
  (h_extra : extra_crayons = 6)
  (h_total : total_crayons = 40) 
  (valid_packs : packs_crayons = (Beth_packs_of_crayons total_crayons extra_crayons / each_pack_contains)) :
  packs_crayons = 3 :=
by
  rw [h_each_pack, h_extra, h_total] at valid_packs
  sorry

end Beth_crayons_proof_l502_50243


namespace find_factor_l502_50280

theorem find_factor (x f : ℕ) (hx : x = 110) (h : x * f - 220 = 110) : f = 3 :=
sorry

end find_factor_l502_50280


namespace find_center_of_circle_l502_50220

-- Condition 1: The circle is tangent to the lines 3x - 4y = 12 and 3x - 4y = -48
def tangent_line1 (x y : ℝ) : Prop := 3 * x - 4 * y = 12
def tangent_line2 (x y : ℝ) : Prop := 3 * x - 4 * y = -48

-- Condition 2: The center of the circle lies on the line x - 2y = 0
def center_line (x y : ℝ) : Prop := x - 2 * y = 0

-- The center of the circle
def circle_center (x y : ℝ) : Prop := 
  tangent_line1 x y ∧ tangent_line2 x y ∧ center_line x y

-- Statement to prove
theorem find_center_of_circle : 
  circle_center (-18) (-9) := 
sorry

end find_center_of_circle_l502_50220


namespace print_output_l502_50298

-- Conditions
def a : Nat := 10

/-- The print statement with the given conditions should output "a=10" -/
theorem print_output : "a=" ++ toString a = "a=10" :=
sorry

end print_output_l502_50298


namespace power_comparison_l502_50216

theorem power_comparison :
  2 ^ 16 = 256 * 16 ^ 2 := 
by
  sorry

end power_comparison_l502_50216


namespace students_passed_both_tests_l502_50288

theorem students_passed_both_tests :
  ∀ (total students_passed_long_jump students_passed_shot_put students_failed_both x : ℕ),
    total = 50 →
    students_passed_long_jump = 40 →
    students_passed_shot_put = 31 →
    students_failed_both = 4 →
    (students_passed_long_jump - x) + (students_passed_shot_put - x) + x + students_failed_both = total →
    x = 25 :=
by intros total students_passed_long_jump students_passed_shot_put students_failed_both x
   intro total_eq students_passed_long_jump_eq students_passed_shot_put_eq students_failed_both_eq sum_eq
   sorry

end students_passed_both_tests_l502_50288


namespace ratio_of_plums_to_peaches_is_three_l502_50291

theorem ratio_of_plums_to_peaches_is_three :
  ∃ (L P W : ℕ), W = 1 ∧ P = W + 12 ∧ L = 3 * P ∧ W + P + L = 53 ∧ (L / P) = 3 :=
by
  sorry

end ratio_of_plums_to_peaches_is_three_l502_50291


namespace find_f_of_2_l502_50251

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 5

theorem find_f_of_2 : f 2 = 5 := by
  sorry

end find_f_of_2_l502_50251


namespace uneaten_pancakes_time_l502_50283

theorem uneaten_pancakes_time:
  ∀ (production_rate_dad production_rate_mom consumption_rate_petya consumption_rate_vasya : ℕ) (k : ℕ),
    production_rate_dad = 70 →
    production_rate_mom = 100 →
    consumption_rate_petya = 10 * 4 → -- 10 pancakes in 15 minutes -> (10/15) * 60 = 40 per hour
    consumption_rate_vasya = 2 * consumption_rate_petya →
    k * ((production_rate_dad + production_rate_mom) / 60 - (consumption_rate_petya + consumption_rate_vasya) / 60) ≥ 20 →
    k ≥ 24 := 
by
  intros production_rate_dad production_rate_mom consumption_rate_petya consumption_rate_vasya k
  sorry

end uneaten_pancakes_time_l502_50283


namespace sqrt_sqrt_81_is_9_l502_50231

theorem sqrt_sqrt_81_is_9 : Real.sqrt (Real.sqrt 81) = 3 := sorry

end sqrt_sqrt_81_is_9_l502_50231


namespace percentage_decrease_l502_50297

variable {a b x m : ℝ} (p : ℝ)

theorem percentage_decrease (h₁ : a / b = 4 / 5)
                          (h₂ : x = 1.25 * a)
                          (h₃ : m = b * (1 - p / 100))
                          (h₄ : m / x = 0.8) :
  p = 20 :=
sorry

end percentage_decrease_l502_50297


namespace find_number_l502_50285

variable (a b x : ℕ)

theorem find_number
    (h1 : x * a = 7 * b)
    (h2 : x * a = 20)
    (h3 : 7 * b = 20) :
    x = 1 :=
sorry

end find_number_l502_50285


namespace rooms_with_two_beds_l502_50268

variable (x y : ℕ)

theorem rooms_with_two_beds:
  x + y = 13 →
  2 * x + 3 * y = 31 →
  x = 8 :=
by
  intros h1 h2
  sorry

end rooms_with_two_beds_l502_50268


namespace compute_inverse_10_mod_1729_l502_50221

def inverse_of_10_mod_1729 : ℕ :=
  1537

theorem compute_inverse_10_mod_1729 :
  (10 * inverse_of_10_mod_1729) % 1729 = 1 :=
by
  sorry

end compute_inverse_10_mod_1729_l502_50221


namespace missing_digit_divisibility_l502_50286

theorem missing_digit_divisibility (x : ℕ) (h1 : x < 10) :
  3 ∣ (1 + 3 + 5 + 7 + x + 2) ↔ x = 0 ∨ x = 3 ∨ x = 6 ∨ x = 9 := by
  sorry

end missing_digit_divisibility_l502_50286


namespace days_in_april_l502_50227

-- Hannah harvests 5 strawberries daily for the whole month of April.
def harvest_per_day : ℕ := 5
-- She gives away 20 strawberries.
def strawberries_given_away : ℕ := 20
-- 30 strawberries are stolen.
def strawberries_stolen : ℕ := 30
-- She has 100 strawberries by the end of April.
def strawberries_final : ℕ := 100

theorem days_in_april : 
  ∃ (days : ℕ), (days * harvest_per_day = strawberries_final + strawberries_given_away + strawberries_stolen) :=
by
  sorry

end days_in_april_l502_50227


namespace quadratic_inequality_l502_50222

theorem quadratic_inequality (a : ℝ) 
  (x₁ x₂ : ℝ) (h_roots : ∀ x, x^2 + (3 * a - 1) * x + a + 8 = 0) 
  (h_distinct : x₁ ≠ x₂)
  (h_x1_lt_1 : x₁ < 1) (h_x2_gt_1 : x₂ > 1) : 
  a < -2 := 
by
  sorry

end quadratic_inequality_l502_50222


namespace find_f_of_one_third_l502_50239

-- Define g function according to given condition
def g (x : ℝ) : ℝ := 1 - x^2

-- Define f function according to given condition, valid for x ≠ 0
noncomputable def f (x : ℝ) : ℝ := (1 - x) / x

-- State the theorem we need to prove
theorem find_f_of_one_third : f (1 / 3) = 1 / 2 :=
by
  -- Placeholder for the proof
  sorry

end find_f_of_one_third_l502_50239


namespace max_side_length_triangle_l502_50207

theorem max_side_length_triangle (a b c : ℕ) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_perimeter : a + b + c = 20) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : max a (max b c) = 9 := 
sorry

end max_side_length_triangle_l502_50207


namespace pretty_number_characterization_l502_50240

def is_pretty (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∀ k ℓ : ℕ, k < n → ℓ < n → k > 0 → ℓ > 0 → 
    (n ∣ 2*k - ℓ ∨ n ∣ 2*ℓ - k)

theorem pretty_number_characterization :
  ∀ n : ℕ, is_pretty n ↔ (Prime n ∨ n = 6 ∨ n = 9 ∨ n = 15) :=
by
  sorry

end pretty_number_characterization_l502_50240


namespace present_population_l502_50209

variable (P : ℝ)
variable (H1 : P * 1.20 = 2400)

theorem present_population (H1 : P * 1.20 = 2400) : P = 2000 :=
by {
  sorry
}

end present_population_l502_50209


namespace mutter_paid_correct_amount_l502_50272

def total_lagaan_collected : ℝ := 344000
def mutter_land_percentage : ℝ := 0.0023255813953488372
def mutter_lagaan_paid : ℝ := 800

theorem mutter_paid_correct_amount : 
  mutter_lagaan_paid = total_lagaan_collected * mutter_land_percentage := by
  sorry

end mutter_paid_correct_amount_l502_50272


namespace cos2_minus_sin2_pi_over_12_l502_50235

theorem cos2_minus_sin2_pi_over_12 : 
  (Real.cos (Real.pi / 12))^2 - (Real.sin (Real.pi / 12))^2 = Real.cos (Real.pi / 6) := 
by
  sorry

end cos2_minus_sin2_pi_over_12_l502_50235


namespace acute_triangle_sin_sum_gt_two_l502_50296

theorem acute_triangle_sin_sum_gt_two 
  {α β γ : ℝ} 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : 0 < γ ∧ γ < π / 2) 
  (h4 : α + β + γ = π) :
  (Real.sin α + Real.sin β + Real.sin γ > 2) :=
sorry

end acute_triangle_sin_sum_gt_two_l502_50296


namespace least_number_to_subtract_l502_50293

theorem least_number_to_subtract (n : ℕ) : 
  ∃ k : ℕ, k = 762429836 % 17 ∧ k = 15 := 
by sorry

end least_number_to_subtract_l502_50293


namespace students_not_visiting_any_l502_50212

-- Define the given conditions as Lean definitions
def total_students := 52
def visited_botanical := 12
def visited_animal := 26
def visited_technology := 23
def visited_botanical_animal := 5
def visited_botanical_technology := 2
def visited_animal_technology := 4
def visited_all_three := 1

-- Translate the problem statement and proof goal
theorem students_not_visiting_any :
  total_students - (visited_botanical + visited_animal + visited_technology 
  - visited_botanical_animal - visited_botanical_technology 
  - visited_animal_technology + visited_all_three) = 1 :=
by
  -- The proof is omitted
  sorry

end students_not_visiting_any_l502_50212


namespace product_M1_M2_l502_50262

theorem product_M1_M2 :
  (∃ M1 M2 : ℝ, (∀ x : ℝ, x ≠ 1 ∧ x ≠ 3 →
    (45 * x - 36) / (x^2 - 4 * x + 3) = M1 / (x - 1) + M2 / (x - 3)) ∧
    M1 * M2 = -222.75) :=
sorry

end product_M1_M2_l502_50262


namespace graph_three_lines_no_common_point_l502_50203

theorem graph_three_lines_no_common_point :
  ∀ x y : ℝ, x^2 * (x + 2*y - 3) = y^2 * (x + 2*y - 3) →
    x + 2*y - 3 = 0 ∨ x = y ∨ x = -y :=
by sorry

end graph_three_lines_no_common_point_l502_50203


namespace find_A_l502_50264

theorem find_A (A B : ℕ) (h : 632 - (100 * A + 10 * B) = 41) : A = 5 :=
by 
  sorry

end find_A_l502_50264


namespace total_third_graders_l502_50201

theorem total_third_graders (num_girls : ℕ) (num_boys : ℕ) (h1 : num_girls = 57) (h2 : num_boys = 66) : num_girls + num_boys = 123 :=
by
  sorry

end total_third_graders_l502_50201


namespace smallest_number_l502_50224

theorem smallest_number (a b c : ℕ) (h1 : b = 29) (h2 : c = b + 7) (h3 : (a + b + c) / 3 = 30) : a = 25 :=
by
  sorry

end smallest_number_l502_50224


namespace gen_term_seq_l502_50234

open Nat

def seq (a : ℕ → ℕ) : Prop := 
a 1 = 1 ∧ (∀ n : ℕ, n ≠ 0 → a (n + 1) = 2 * a n - 3)

theorem gen_term_seq (a : ℕ → ℕ) (h : seq a) : ∀ n : ℕ, a n = 3 - 2^n :=
by
  sorry

end gen_term_seq_l502_50234


namespace monotonically_increasing_interval_l502_50245

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)

theorem monotonically_increasing_interval (k : ℤ) :
  ∀ x : ℝ, (k * Real.pi - 5 * Real.pi / 12 <= x ∧ x <= k * Real.pi + Real.pi / 12) →
    (∃ r : ℝ, f (x + r) > f x ∨ f (x + r) < f x) := by
  sorry

end monotonically_increasing_interval_l502_50245


namespace parabola_hyperbola_tangent_l502_50237

theorem parabola_hyperbola_tangent : ∃ m : ℝ, 
  (∀ x y : ℝ, y = x^2 - 2 * x + 2 → y^2 - m * x^2 = 1) ↔ m = 1 :=
by
  sorry

end parabola_hyperbola_tangent_l502_50237


namespace short_answer_question_time_l502_50255

-- Definitions from the conditions
def minutes_per_paragraph := 15
def minutes_per_essay := 60
def num_essays := 2
def num_paragraphs := 5
def num_short_answer_questions := 15
def total_minutes := 4 * 60

-- Auxiliary calculations
def total_minutes_essays := num_essays * minutes_per_essay
def total_minutes_paragraphs := num_paragraphs * minutes_per_paragraph
def total_minutes_used := total_minutes_essays + total_minutes_paragraphs

-- The time per short-answer question is 3 minutes
theorem short_answer_question_time (x : ℕ) : (total_minutes - total_minutes_used) / num_short_answer_questions = 3 :=
by
  -- x is defined as the time per short-answer question
  let x := (total_minutes - total_minutes_used) / num_short_answer_questions
  have time_for_short_answer_questions : total_minutes - total_minutes_used = 45 := by sorry
  have time_per_short_answer_question : 45 / num_short_answer_questions = 3 := by sorry
  have x_equals_3 : x = 3 := by sorry
  exact x_equals_3

end short_answer_question_time_l502_50255


namespace turquoise_more_green_l502_50276

-- Definitions based on given conditions
def total_people : ℕ := 150
def more_blue : ℕ := 90
def both_blue_green : ℕ := 40
def neither_blue_green : ℕ := 20

-- Theorem statement to prove the number of people who believe turquoise is more green
theorem turquoise_more_green : (total_people - neither_blue_green - (more_blue - both_blue_green) - both_blue_green) + both_blue_green = 80 := by
  sorry

end turquoise_more_green_l502_50276


namespace least_subtracted_12702_is_26_l502_50274

theorem least_subtracted_12702_is_26 : 12702 % 99 = 26 :=
by
  sorry

end least_subtracted_12702_is_26_l502_50274


namespace max_blocks_in_box_l502_50241

def volume (l w h : ℕ) : ℕ := l * w * h

-- Define the dimensions of the box and the block
def box_length := 4
def box_width := 3
def box_height := 2
def block_length := 3
def block_width := 1
def block_height := 1

-- Define the volumes of the box and the block using the dimensions
def V_box : ℕ := volume box_length box_width box_height
def V_block : ℕ := volume block_length block_width block_height

theorem max_blocks_in_box : V_box / V_block = 8 :=
  sorry

end max_blocks_in_box_l502_50241


namespace range_of_a_l502_50200

noncomputable def func (x a : ℝ) : ℝ := -x^2 - 2 * a * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → func x a ≤ a^2) →
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ func x a = a^2) →
  -1 ≤ a ∧ a ≤ 0 :=
by
  sorry

end range_of_a_l502_50200


namespace determine_range_of_b_l502_50218

noncomputable def f (b x : ℝ) : ℝ := (Real.log x + (x - b) ^ 2) / x
noncomputable def f'' (b x : ℝ) : ℝ := (2 * Real.log x - 2) / x ^ 3

theorem determine_range_of_b (b : ℝ) (h : ∃ x ∈ Set.Icc (1 / 2) 2, f b x > -x * f'' b x) :
  b < 9 / 4 :=
by
  sorry

end determine_range_of_b_l502_50218


namespace solution_set_inequality_l502_50225

theorem solution_set_inequality (x : ℝ) : 
  (-x^2 + 3 * x - 2 ≥ 0) ↔ (1 ≤ x ∧ x ≤ 2) :=
sorry

end solution_set_inequality_l502_50225


namespace alpha_value_l502_50250

theorem alpha_value (α : ℝ) (h : 0 ≤ α ∧ α ≤ 2 * Real.pi 
    ∧ ∃β : ℝ, β = 2 * Real.pi / 3 ∧ (Real.sin β, Real.cos β) = (Real.sin α, Real.cos α)) : 
    α = 5 * Real.pi / 3 := 
  by
    sorry

end alpha_value_l502_50250


namespace sphere_surface_area_l502_50281

theorem sphere_surface_area (a : ℝ) (l R : ℝ)
  (h₁ : 6 * l^2 = a)
  (h₂ : l * Real.sqrt 3 = 2 * R) :
  4 * Real.pi * R^2 = (Real.pi / 2) * a :=
sorry

end sphere_surface_area_l502_50281


namespace solve_for_x_l502_50217

theorem solve_for_x (x : ℤ) (h : -3 * x - 8 = 8 * x + 3) : x = -1 :=
by
  sorry

end solve_for_x_l502_50217


namespace arithmetic_sequence_properties_l502_50223

def a_n (n : ℕ) : ℤ := 2 * n + 1

def S_n (n : ℕ) : ℤ := n * (n + 2)

theorem arithmetic_sequence_properties : 
  (a_n 3 = 7) ∧ (a_n 5 + a_n 7 = 26) :=
by {
  -- Proof to be filled
  sorry
}

end arithmetic_sequence_properties_l502_50223


namespace largest_on_edge_l502_50266

/-- On a grid, each cell contains a number which is the arithmetic mean of the four numbers around it 
    and all numbers are different. Prove that the largest number is located on the edge of the grid. -/
theorem largest_on_edge 
    (grid : ℕ → ℕ → ℝ) 
    (h_condition : ∀ (i j : ℕ), grid i j = (grid (i+1) j + grid (i-1) j + grid i (j+1) + grid i (j-1)) / 4)
    (h_unique : ∀ (i1 j1 i2 j2 : ℕ), (i1 ≠ i2 ∨ j1 ≠ j2) → grid i1 j1 ≠ grid i2 j2)
    : ∃ (i j : ℕ), (i = 0 ∨ j = 0 ∨ i = max_i ∨ j = max_j) ∧ ∀ (x y : ℕ), grid x y ≤ grid i j :=
sorry

end largest_on_edge_l502_50266


namespace speed_against_current_l502_50292

noncomputable def man's_speed_with_current : ℝ := 20
noncomputable def current_speed : ℝ := 1

theorem speed_against_current :
  (man's_speed_with_current - 2 * current_speed) = 18 := by
sorry

end speed_against_current_l502_50292


namespace missing_digits_pairs_l502_50232

theorem missing_digits_pairs (x y : ℕ) : (2 + 4 + 6 + x + y + 8) % 9 = 0 ↔ x + y = 7 := by
  sorry

end missing_digits_pairs_l502_50232


namespace ratio_of_students_to_professors_l502_50219

theorem ratio_of_students_to_professors (total : ℕ) (students : ℕ) (professors : ℕ)
  (h1 : total = 40000) (h2 : students = 37500) (h3 : total = students + professors) :
  students / professors = 15 :=
by
  sorry

end ratio_of_students_to_professors_l502_50219


namespace find_added_amount_l502_50226

theorem find_added_amount (x y : ℕ) (h1 : x = 18) (h2 : 3 * (2 * x + y) = 123) : y = 5 :=
by
  sorry

end find_added_amount_l502_50226


namespace largest_digit_M_divisible_by_6_l502_50270

theorem largest_digit_M_divisible_by_6 (M : ℕ) (h1 : 5172 * 10 + M % 2 = 0) (h2 : (5 + 1 + 7 + 2 + M) % 3 = 0) : M = 6 := by
  sorry

end largest_digit_M_divisible_by_6_l502_50270


namespace bc_over_ad_l502_50261

noncomputable def a : ℝ := 32 / 3
noncomputable def b : ℝ := 16 * Real.pi
noncomputable def c : ℝ := 24 * Real.pi
noncomputable def d : ℝ := 16 * Real.pi

theorem bc_over_ad : (b * c) / (a * d) = 9 / 4 := 
by 
  sorry

end bc_over_ad_l502_50261


namespace simplify_expression_l502_50215

theorem simplify_expression (x y : ℝ) (m : ℤ) : 
  ((x + y)^(2 * m + 1) / (x + y)^(m - 1) = (x + y)^(m + 2)) :=
by sorry

end simplify_expression_l502_50215


namespace pyramid_base_edge_length_l502_50289

theorem pyramid_base_edge_length
  (hemisphere_radius : ℝ) (pyramid_height : ℝ) (slant_height : ℝ) (is_tangent: Prop) :
  hemisphere_radius = 3 ∧ pyramid_height = 8 ∧ slant_height = 10 ∧ is_tangent →
  ∃ (base_edge_length : ℝ), base_edge_length = 6 * Real.sqrt 2 :=
by
  sorry

end pyramid_base_edge_length_l502_50289


namespace kenny_trumpet_hours_l502_50202

variables (x y : ℝ)
def basketball_hours := 10
def running_hours := 2 * basketball_hours
def trumpet_hours := 2 * running_hours

theorem kenny_trumpet_hours (x y : ℝ) (H : basketball_hours + running_hours + trumpet_hours = x + y) :
  trumpet_hours = 40 :=
by
  sorry

end kenny_trumpet_hours_l502_50202


namespace value_of_b_l502_50208

theorem value_of_b (a b c y1 y2 y3 : ℝ)
( h1 : y1 = a + b + c )
( h2 : y2 = a - b + c )
( h3 : y3 = 4 * a + 2 * b + c )
( h4 : y1 - y2 = 8 )
( h5 : y3 = y1 + 2 )
: b = 4 :=
sorry

end value_of_b_l502_50208


namespace max_zeros_in_product_of_three_natural_numbers_sum_1003_l502_50269

theorem max_zeros_in_product_of_three_natural_numbers_sum_1003 :
  ∀ (a b c : ℕ), a + b + c = 1003 →
    ∃ N, (a * b * c) % (10^N) = 0 ∧ N = 7 := by
  sorry

end max_zeros_in_product_of_three_natural_numbers_sum_1003_l502_50269


namespace marbles_total_l502_50205

theorem marbles_total (fabian kyle miles : ℕ) (h1 : fabian = 3 * kyle) (h2 : fabian = 5 * miles) (h3 : fabian = 15) : kyle + miles = 8 := by
  sorry

end marbles_total_l502_50205


namespace max_value_expression_le_380_l502_50244

noncomputable def max_value_expression (a b c d : ℝ) : ℝ :=
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_expression_le_380 (a b c d : ℝ)
  (ha : -9.5 ≤ a ∧ a ≤ 9.5)
  (hb : -9.5 ≤ b ∧ b ≤ 9.5)
  (hc : -9.5 ≤ c ∧ c ≤ 9.5)
  (hd : -9.5 ≤ d ∧ d ≤ 9.5) :
  max_value_expression a b c d ≤ 380 :=
sorry

end max_value_expression_le_380_l502_50244


namespace solve_rational_inequality_l502_50295

theorem solve_rational_inequality :
  {x : ℝ | (9*x^2 + 18 * x - 60) / ((3 * x - 4) * (x + 5)) < 4} =
  {x : ℝ | (-10 < x ∧ x < -5) ∨ (2/3 < x ∧ x < 4/3) ∨ (4/3 < x)} :=
by
  sorry

end solve_rational_inequality_l502_50295


namespace n_leq_1972_l502_50252

theorem n_leq_1972 (n : ℕ) (h1 : 4 ^ 27 + 4 ^ 1000 + 4 ^ n = k ^ 2) : n ≤ 1972 :=
by
  sorry

end n_leq_1972_l502_50252


namespace problem1_l502_50279

theorem problem1 (x y : ℝ) (h1 : x * (x + y) = 27) (h2 : y * (x + y) = 54) : (x + y)^2 = 81 := 
by
  sorry

end problem1_l502_50279


namespace roots_reciprocal_l502_50271

theorem roots_reciprocal {a b c x y : ℝ} (h1 : a ≠ 0) (h2 : c ≠ 0) :
  (a * x^2 + b * x + c = 0) ↔ (c * y^2 + b * y + a = 0) := by
sorry

end roots_reciprocal_l502_50271


namespace lowest_price_l502_50278

theorem lowest_price (cost_per_component shipping_cost_per_unit fixed_costs number_of_components produced_cost total_variable_cost total_cost lowest_price : ℝ)
  (h1 : cost_per_component = 80)
  (h2 : shipping_cost_per_unit = 2)
  (h3 : fixed_costs = 16200)
  (h4 : number_of_components = 150)
  (h5 : total_variable_cost = cost_per_component + shipping_cost_per_unit)
  (h6 : produced_cost = total_variable_cost * number_of_components)
  (h7 : total_cost = produced_cost + fixed_costs)
  (h8 : lowest_price = total_cost / number_of_components) :
  lowest_price = 190 :=
  by
  sorry

end lowest_price_l502_50278


namespace simplify_tan_expression_l502_50204

noncomputable def tan_15 := Real.tan (Real.pi / 12)
noncomputable def tan_30 := Real.tan (Real.pi / 6)

theorem simplify_tan_expression : (1 + tan_15) * (1 + tan_30) = 2 := by
  sorry

end simplify_tan_expression_l502_50204


namespace no_prime_p_such_that_22p2_plus_23_is_prime_l502_50282

theorem no_prime_p_such_that_22p2_plus_23_is_prime :
  ∀ p : ℕ, Prime p → ¬ Prime (22 * p ^ 2 + 23) :=
by
  sorry

end no_prime_p_such_that_22p2_plus_23_is_prime_l502_50282


namespace analysis_method_sufficient_conditions_l502_50290

theorem analysis_method_sufficient_conditions (P : Prop) (analysis_method : ∀ (Q : Prop), (Q → P) → Q) :
  ∀ Q, (Q → P) → Q :=
by
  -- Proof is skipped
  sorry

end analysis_method_sufficient_conditions_l502_50290


namespace rectangular_prism_volume_l502_50233

theorem rectangular_prism_volume
  (a b c : ℕ) 
  (h1 : 4 * ((a - 2) + (b - 2) + (c - 2)) = 40)
  (h2 : 2 * ((a - 2) * (b - 2) + (a - 2) * (c - 2) + (b - 2) * (c - 2)) = 66) :
  a * b * c = 150 :=
by sorry

end rectangular_prism_volume_l502_50233


namespace exists_prime_with_composite_sequence_l502_50228

theorem exists_prime_with_composite_sequence (n : ℕ) (hn : n ≠ 0) : 
  ∃ p : ℕ, Nat.Prime p ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → ¬ Nat.Prime (p + k) :=
sorry

end exists_prime_with_composite_sequence_l502_50228


namespace min_a_b_div_1176_l502_50246

theorem min_a_b_div_1176 (a b : ℕ) (h : b^3 = 1176 * a) : a = 63 :=
by sorry

end min_a_b_div_1176_l502_50246


namespace solve_nat_numbers_equation_l502_50273

theorem solve_nat_numbers_equation (n k l m : ℕ) (h_l : l > 1) 
  (h_eq : (1 + n^k)^l = 1 + n^m) : (n = 2) ∧ (k = 1) ∧ (l = 2) ∧ (m = 3) := 
by
  sorry

end solve_nat_numbers_equation_l502_50273


namespace rain_first_hour_l502_50275

theorem rain_first_hour (x : ℝ) 
  (h1 : 22 = x + (2 * x + 7)) : x = 5 :=
by
  sorry

end rain_first_hour_l502_50275


namespace points_on_single_circle_l502_50299

theorem points_on_single_circle (n : ℕ) (points : Fin n → ℝ × ℝ)
  (h : ∀ i j : Fin n, ∃ f : ℝ × ℝ → ℝ × ℝ, (∀ p, f p ≠ p) ∧ f (points i) = points j ∧ 
        (∀ k : Fin n, ∃ p, points k = f p)) :
  ∃ (O : ℝ × ℝ) (r : ℝ), ∀ i : Fin n, dist (points i) O = r := sorry

end points_on_single_circle_l502_50299


namespace hyperbola_eccentricity_range_l502_50206

theorem hyperbola_eccentricity_range (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  (∃ P₁ P₂ : { p : ℝ × ℝ // p ≠ (0, b) ∧ p ≠ (c, 0) ∧ ((0, b) - p).1 * ((c, 0) - p).1 + ((0, b) - p).2 * ((c, 0) - p).2 = 0},
   true) -- This encodes the existence of the required points P₁ and P₂ on line segment BF excluding endpoints
  → 1 < (Real.sqrt ((a^2 + b^2) / a^2)) ∧ (Real.sqrt ((a^2 + b^2) / a^2)) < (Real.sqrt 5 + 1)/2 :=
sorry

end hyperbola_eccentricity_range_l502_50206


namespace arc_intercept_length_l502_50284

noncomputable def side_length : ℝ := 4
noncomputable def diagonal_length : ℝ := Real.sqrt (side_length^2 + side_length^2)
noncomputable def radius : ℝ := diagonal_length / 2
noncomputable def circumference : ℝ := 2 * Real.pi * radius
noncomputable def arc_length_one_side : ℝ := circumference / 4

theorem arc_intercept_length :
  arc_length_one_side = Real.sqrt 2 * Real.pi :=
by
  sorry

end arc_intercept_length_l502_50284


namespace cars_fell_in_lot_l502_50254

theorem cars_fell_in_lot (initial_cars went_out_cars came_in_cars final_cars: ℕ) (h1 : initial_cars = 25) 
    (h2 : went_out_cars = 18) (h3 : came_in_cars = 12) (h4 : final_cars = initial_cars - went_out_cars + came_in_cars) :
    initial_cars - final_cars = 6 :=
    sorry

end cars_fell_in_lot_l502_50254


namespace rectangle_perimeter_l502_50263

theorem rectangle_perimeter (w : ℝ) (P : ℝ) (l : ℝ) (A : ℝ) 
  (h1 : l = 18)
  (h2 : A = l * w)
  (h3 : P = 2 * l + 2 * w) 
  (h4 : A + P = 2016) : 
  P = 234 :=
by
  sorry

end rectangle_perimeter_l502_50263
