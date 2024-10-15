import Mathlib

namespace NUMINAMATH_GPT_arithmetic_sequence_geometric_ratio_l286_28607

theorem arithmetic_sequence_geometric_ratio
  (a : ℕ → ℚ) (d : ℚ)
  (h_arith : ∀ n : ℕ, a (n+1) = a n + d)
  (h_nonzero_d : d ≠ 0)
  (h_geo : (a 2) * (a 9) = (a 3) ^ 2)
  : (a 4 + a 5 + a 6) / (a 2 + a 3 + a 4) = (8 / 3) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_geometric_ratio_l286_28607


namespace NUMINAMATH_GPT_quadratic_trinomial_unique_l286_28615

theorem quadratic_trinomial_unique
  (a b c : ℝ)
  (h1 : b^2 - 4*(a+1)*c = 0)
  (h2 : (b+1)^2 - 4*a*c = 0)
  (h3 : b^2 - 4*a*(c+1) = 0) :
  a = 1/8 ∧ b = -3/4 ∧ c = 1/8 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_trinomial_unique_l286_28615


namespace NUMINAMATH_GPT_sum_of_interior_numbers_eighth_row_l286_28616

def sum_of_interior_numbers (n : ℕ) : ℕ :=
  2^(n-1) - 2

theorem sum_of_interior_numbers_eighth_row : sum_of_interior_numbers 8 = 126 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_numbers_eighth_row_l286_28616


namespace NUMINAMATH_GPT_proportion_fourth_number_l286_28664

theorem proportion_fourth_number (x y : ℝ) (h_x : x = 0.6) (h_prop : 0.75 / x = 10 / y) : y = 8 :=
by
  sorry

end NUMINAMATH_GPT_proportion_fourth_number_l286_28664


namespace NUMINAMATH_GPT_green_bows_count_l286_28633

noncomputable def total_bows : ℕ := 36 * 4

def fraction_green : ℚ := 1/6

theorem green_bows_count (red blue green total yellow : ℕ) (h_red : red = total / 4)
  (h_blue : blue = total / 3) (h_green : green = total / 6)
  (h_yellow : yellow = total - red - blue - green)
  (h_yellow_count : yellow = 36) : green = 24 := by
  sorry

end NUMINAMATH_GPT_green_bows_count_l286_28633


namespace NUMINAMATH_GPT_triangle_circle_property_l286_28624

-- Let a, b, and c be the lengths of the sides of a right triangle, where c is the hypotenuse.
variables {a b c : ℝ}

-- Let varrho_b be the radius of the circle inscribed around the leg b of the triangle.
variable {varrho_b : ℝ}

-- Assume the relationship a^2 + b^2 = c^2 (Pythagorean theorem).
axiom right_triangle : a^2 + b^2 = c^2

-- Prove that b + c = a + 2 * varrho_b
theorem triangle_circle_property (h : a^2 + b^2 = c^2) (radius_condition : varrho_b = (a*b)/(a+c-b)) : 
  b + c = a + 2 * varrho_b :=
sorry

end NUMINAMATH_GPT_triangle_circle_property_l286_28624


namespace NUMINAMATH_GPT_water_level_decrease_3m_l286_28692

-- Definitions from conditions
def increase (amount : ℝ) : ℝ := amount
def decrease (amount : ℝ) : ℝ := -amount

-- The claim to be proven
theorem water_level_decrease_3m : decrease 3 = -3 :=
by
  sorry

end NUMINAMATH_GPT_water_level_decrease_3m_l286_28692


namespace NUMINAMATH_GPT_max_true_statements_maximum_true_conditions_l286_28647

theorem max_true_statements (x y : ℝ) (h1 : (1/x > 1/y)) (h2 : (x^2 < y^2)) (h3 : (x > y)) (h4 : (x > 0)) (h5 : (y > 0)) :
  false :=
  sorry

theorem maximum_true_conditions (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  ¬ ((1/x > 1/y) ∧ (x^2 < y^2)) :=
  sorry

#check max_true_statements
#check maximum_true_conditions

end NUMINAMATH_GPT_max_true_statements_maximum_true_conditions_l286_28647


namespace NUMINAMATH_GPT_max_ab_value_1_half_l286_28625

theorem max_ab_value_1_half 
  (a b : ℝ) 
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_eq : a + 2 * b = 1) :
  a = 1 / 2 → ab = 1 / 8 :=
sorry

end NUMINAMATH_GPT_max_ab_value_1_half_l286_28625


namespace NUMINAMATH_GPT_proof_problem_l286_28652

-- Define the operation table as a function in Lean 4
def op (a b : ℕ) : ℕ :=
  if a = 1 then
    if b = 1 then 2 else if b = 2 then 1 else if b = 3 then 4 else 3
  else if a = 2 then
    if b = 1 then 1 else if b = 2 then 3 else if b = 3 then 2 else 4
  else if a = 3 then
    if b = 1 then 4 else if b = 2 then 2 else if b = 3 then 1 else 3
  else
    if b = 1 then 3 else if b = 2 then 4 else if b = 3 then 3 else 2

-- State the theorem to prove
theorem proof_problem : op (op 3 1) (op 4 2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l286_28652


namespace NUMINAMATH_GPT_percent_change_is_minus_5_point_5_percent_l286_28604

noncomputable def overall_percent_change (initial_value : ℝ) : ℝ :=
  let day1_value := initial_value * 0.75
  let day2_value := day1_value * 1.4
  let final_value := day2_value * 0.9
  ((final_value / initial_value) - 1) * 100

theorem percent_change_is_minus_5_point_5_percent :
  ∀ (initial_value : ℝ), overall_percent_change initial_value = -5.5 :=
sorry

end NUMINAMATH_GPT_percent_change_is_minus_5_point_5_percent_l286_28604


namespace NUMINAMATH_GPT_complex_is_purely_imaginary_iff_a_eq_2_l286_28669

theorem complex_is_purely_imaginary_iff_a_eq_2 (a : ℝ) :
  (a = 2) ↔ ((a^2 - 4 = 0) ∧ (a + 2 ≠ 0)) :=
by sorry

end NUMINAMATH_GPT_complex_is_purely_imaginary_iff_a_eq_2_l286_28669


namespace NUMINAMATH_GPT_trajectory_of_midpoint_l286_28655

theorem trajectory_of_midpoint
  (M : ℝ × ℝ)
  (P : ℝ × ℝ) (Q : ℝ × ℝ)
  (hP : P = (4, 0))
  (hQ : Q.1^2 + Q.2^2 = 4)
  (M_is_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  (M.1 - 2)^2 + M.2^2 = 1 :=
sorry

end NUMINAMATH_GPT_trajectory_of_midpoint_l286_28655


namespace NUMINAMATH_GPT_no_solution_for_12k_plus_7_l286_28621

theorem no_solution_for_12k_plus_7 (k : ℤ) :
  ∀ (a b c : ℕ), 12 * k + 7 ≠ 2^a + 3^b - 5^c := 
by sorry

end NUMINAMATH_GPT_no_solution_for_12k_plus_7_l286_28621


namespace NUMINAMATH_GPT_solve_quadratic_equation_l286_28690

theorem solve_quadratic_equation (x : ℝ) :
  x^2 - 2 * x - 8 = 0 ↔ (x = 4 ∨ x = -2) :=
by sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l286_28690


namespace NUMINAMATH_GPT_negation_of_proposition_l286_28642

open Classical

theorem negation_of_proposition : (¬ ∀ x : ℝ, 2 * x + 4 ≥ 0) ↔ (∃ x : ℝ, 2 * x + 4 < 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l286_28642


namespace NUMINAMATH_GPT_sufficient_condition_for_inequality_l286_28636

theorem sufficient_condition_for_inequality (a : ℝ) (h : 0 < a ∧ a < 4) :
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_for_inequality_l286_28636


namespace NUMINAMATH_GPT_math_problem_l286_28661

variable {a b c d e f : ℕ}
variable (h1 : f < a)
variable (h2 : (a * b * d + 1) % c = 0)
variable (h3 : (a * c * e + 1) % b = 0)
variable (h4 : (b * c * f + 1) % a = 0)

theorem math_problem
  (h5 : (d : ℚ) / c < 1 - (e : ℚ) / b) :
  (d : ℚ) / c < 1 - (f : ℚ) / a :=
by {
  skip -- Adding "by" ... "sorry" to make the statement complete since no proof is required.
  sorry
}

end NUMINAMATH_GPT_math_problem_l286_28661


namespace NUMINAMATH_GPT_number_of_pairs_satisfying_x_sq_minus_y_sq_eq_100_l286_28627

theorem number_of_pairs_satisfying_x_sq_minus_y_sq_eq_100 :
  ∃! (n : ℕ), n = 3 ∧ ∀ (x y : ℕ), x > 0 → y > 0 → x^2 - y^2 = 100 ↔ (x, y) = (26, 24) ∨ (x, y) = (15, 10) ∨ (x, y) = (15, 5) :=
by
  sorry

end NUMINAMATH_GPT_number_of_pairs_satisfying_x_sq_minus_y_sq_eq_100_l286_28627


namespace NUMINAMATH_GPT_red_marbles_eq_14_l286_28623

theorem red_marbles_eq_14 (total_marbles : ℕ) (yellow_marbles : ℕ) (R : ℕ) (B : ℕ)
  (h1 : total_marbles = 85)
  (h2 : yellow_marbles = 29)
  (h3 : B = 3 * R)
  (h4 : (total_marbles - yellow_marbles) = R + B) :
  R = 14 :=
by
  sorry

end NUMINAMATH_GPT_red_marbles_eq_14_l286_28623


namespace NUMINAMATH_GPT_compute_sum_l286_28608
-- Import the necessary library to have access to the required definitions and theorems.

-- Define the integers involved based on the conditions.
def a : ℕ := 157
def b : ℕ := 43
def c : ℕ := 19
def d : ℕ := 81

-- State the theorem that computes the sum of these integers and equate it to 300.
theorem compute_sum : a + b + c + d = 300 := by
  sorry

end NUMINAMATH_GPT_compute_sum_l286_28608


namespace NUMINAMATH_GPT_book_costs_l286_28601

theorem book_costs (C1 C2 : ℝ) (h1 : C1 + C2 = 450) (h2 : 0.85 * C1 = 1.19 * C2) : C1 = 262.5 := 
sorry

end NUMINAMATH_GPT_book_costs_l286_28601


namespace NUMINAMATH_GPT_total_bill_l286_28645

theorem total_bill (m : ℝ) (h1 : m = 10 * (m / 10 + 3) - 27) : m = 270 :=
by
  sorry

end NUMINAMATH_GPT_total_bill_l286_28645


namespace NUMINAMATH_GPT_matching_pair_probability_correct_l286_28603

-- Define the basic assumptions (conditions)
def black_pairs : Nat := 7
def brown_pairs : Nat := 4
def gray_pairs : Nat := 3
def red_pairs : Nat := 2

def total_pairs : Nat := black_pairs + brown_pairs + gray_pairs + red_pairs
def total_shoes : Nat := 2 * total_pairs

-- The probability calculation will be shown as the final proof requirement
def matching_color_probability : Rat :=  (14 * 7 + 8 * 4 + 6 * 3 + 4 * 2 : Int) / (32 * 31 : Int)

-- The target statement to be proven
theorem matching_pair_probability_correct :
  matching_color_probability = (39 / 248 : Rat) :=
by
  sorry

end NUMINAMATH_GPT_matching_pair_probability_correct_l286_28603


namespace NUMINAMATH_GPT_fred_initial_balloons_l286_28613

def green_balloons_initial (given: Nat) (left: Nat) : Nat := 
  given + left

theorem fred_initial_balloons : green_balloons_initial 221 488 = 709 :=
by
  sorry

end NUMINAMATH_GPT_fred_initial_balloons_l286_28613


namespace NUMINAMATH_GPT_initial_integer_value_l286_28670

theorem initial_integer_value (x : ℤ) (h : (x + 2) * (x + 2) = x * x - 2016) : x = -505 := 
sorry

end NUMINAMATH_GPT_initial_integer_value_l286_28670


namespace NUMINAMATH_GPT_problem_solution_l286_28635

theorem problem_solution :
  3 * 995 + 4 * 996 + 5 * 997 + 6 * 998 + 7 * 999 - 4985 * 3 = 9980 := 
  by
  sorry

end NUMINAMATH_GPT_problem_solution_l286_28635


namespace NUMINAMATH_GPT_neg_q_true_l286_28680

theorem neg_q_true : (∃ x : ℝ, x^2 ≤ 0) :=
sorry

end NUMINAMATH_GPT_neg_q_true_l286_28680


namespace NUMINAMATH_GPT_correct_calculation_l286_28651

theorem correct_calculation (x y : ℝ) : (x^2 * y)^3 = x^6 * y^3 :=
  sorry

end NUMINAMATH_GPT_correct_calculation_l286_28651


namespace NUMINAMATH_GPT_ping_pong_matches_l286_28629

noncomputable def f (n k : ℕ) : ℕ :=
  Nat.ceil ((n : ℚ) / Nat.ceil ((k : ℚ) / 2))

theorem ping_pong_matches (n k : ℕ) (hn_pos : 0 < n) (hk_le : k ≤ 2 * n - 1) :
  f n k = Nat.ceil ((n : ℚ) / Nat.ceil ((k : ℚ) / 2)) :=
by
  sorry

end NUMINAMATH_GPT_ping_pong_matches_l286_28629


namespace NUMINAMATH_GPT_fewest_number_of_students_l286_28605

theorem fewest_number_of_students :
  ∃ n : ℕ, n ≡ 3 [MOD 6] ∧ n ≡ 5 [MOD 8] ∧ n ≡ 7 [MOD 9] ∧ ∀ m : ℕ, (m ≡ 3 [MOD 6] ∧ m ≡ 5 [MOD 8] ∧ m ≡ 7 [MOD 9]) → m ≥ n := by
  sorry

end NUMINAMATH_GPT_fewest_number_of_students_l286_28605


namespace NUMINAMATH_GPT_john_investment_years_l286_28682

theorem john_investment_years (P FVt : ℝ) (r1 r2 : ℝ) (n1 t : ℝ) :
  P = 2000 →
  r1 = 0.08 →
  r2 = 0.12 →
  n1 = 2 →
  FVt = 6620 →
  P * (1 + r1)^n1 * (1 + r2)^(t - n1) = FVt →
  t = 11 :=
by
  sorry

end NUMINAMATH_GPT_john_investment_years_l286_28682


namespace NUMINAMATH_GPT_unique_reconstruction_l286_28653

theorem unique_reconstruction (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (a b c d : ℝ) (Ha : x + y = a) (Hb : x - y = b) (Hc : x * y = c) (Hd : x / y = d) :
  ∃! (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x' + y' = a ∧ x' - y' = b ∧ x' * y' = c ∧ x' / y' = d := 
sorry

end NUMINAMATH_GPT_unique_reconstruction_l286_28653


namespace NUMINAMATH_GPT_toms_dad_gave_him_dimes_l286_28665

theorem toms_dad_gave_him_dimes (original_dimes final_dimes dimes_given : ℕ)
  (h1 : original_dimes = 15)
  (h2 : final_dimes = 48)
  (h3 : final_dimes = original_dimes + dimes_given) :
  dimes_given = 33 :=
by
  -- Since the main goal here is just the statement, proof is omitted with sorry
  sorry

end NUMINAMATH_GPT_toms_dad_gave_him_dimes_l286_28665


namespace NUMINAMATH_GPT_find_set_T_l286_28649

namespace MathProof 

theorem find_set_T (S : Finset ℕ) (hS : ∀ x ∈ S, x > 0) :
  ∃ T : Finset ℕ, S ⊆ T ∧ ∀ x ∈ T, x ∣ (T.sum id) :=
by
  sorry

end MathProof 

end NUMINAMATH_GPT_find_set_T_l286_28649


namespace NUMINAMATH_GPT_no_equilateral_integer_coords_l286_28662

theorem no_equilateral_integer_coords (x1 y1 x2 y2 x3 y3 : ℤ) : 
  ¬ ((x1 ≠ x2 ∨ y1 ≠ y2) ∧ 
     (x1 ≠ x3 ∨ y1 ≠ y3) ∧
     (x2 ≠ x3 ∨ y2 ≠ y3) ∧ 
     ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 = (x3 - x1) ^ 2 + (y3 - y1) ^ 2 ∧ 
      (x2 - x1) ^ 2 + (y2 - y1) ^ 2 = (x3 - x2) ^ 2 + (y3 - y2) ^ 2)) :=
by
  sorry

end NUMINAMATH_GPT_no_equilateral_integer_coords_l286_28662


namespace NUMINAMATH_GPT_abigail_money_left_l286_28617

def initial_amount : ℕ := 11
def spent_in_store : ℕ := 2
def amount_lost : ℕ := 6

theorem abigail_money_left :
  initial_amount - spent_in_store - amount_lost = 3 := 
by {
  sorry
}

end NUMINAMATH_GPT_abigail_money_left_l286_28617


namespace NUMINAMATH_GPT_time_to_odd_floor_l286_28675

-- Define the number of even-numbered floors
def evenFloors : Nat := 5

-- Define the number of odd-numbered floors
def oddFloors : Nat := 5

-- Define the time to climb one even-numbered floor
def timeEvenFloor : Nat := 15

-- Define the total time to reach the 10th floor
def totalTime : Nat := 120

-- Define the desired time per odd-numbered floor
def timeOddFloor : Nat := 9

-- Formalize the proof statement
theorem time_to_odd_floor : 
  (oddFloors * timeOddFloor = totalTime - (evenFloors * timeEvenFloor)) :=
by
  sorry

end NUMINAMATH_GPT_time_to_odd_floor_l286_28675


namespace NUMINAMATH_GPT_incorrect_rounding_statement_l286_28610

def rounded_to_nearest (n : ℝ) (accuracy : ℝ) : Prop :=
  ∃ (k : ℤ), abs (n - k * accuracy) < accuracy / 2

theorem incorrect_rounding_statement :
  ¬ rounded_to_nearest 23.9 10 :=
sorry

end NUMINAMATH_GPT_incorrect_rounding_statement_l286_28610


namespace NUMINAMATH_GPT_linear_inequality_solution_set_l286_28663

variable (x : ℝ)

theorem linear_inequality_solution_set :
  ∀ x : ℝ, (2 * x - 4 > 0) → (x > 2) := 
by
  sorry

end NUMINAMATH_GPT_linear_inequality_solution_set_l286_28663


namespace NUMINAMATH_GPT_remaining_water_l286_28685

theorem remaining_water (initial_water : ℚ) (used_water : ℚ) (remaining_water : ℚ) 
  (h1 : initial_water = 3) (h2 : used_water = 5/4) : remaining_water = 7/4 :=
by
  -- The proof would go here, but we are skipping it as per the instructions.
  sorry

end NUMINAMATH_GPT_remaining_water_l286_28685


namespace NUMINAMATH_GPT_pencils_inequalities_l286_28677

theorem pencils_inequalities (x y : ℕ) :
  (3 * x < 48 ∧ 48 < 4 * x) ∧ (4 * y < 48 ∧ 48 < 5 * y) :=
sorry

end NUMINAMATH_GPT_pencils_inequalities_l286_28677


namespace NUMINAMATH_GPT_circle_radius_l286_28619

theorem circle_radius (r : ℝ) (π : ℝ) (h1 : π > 0) (h2 : ∀ x, π * x^2 = 100*π → x = 10) : r = 10 :=
by
  have : π * r^2 = 100*π → r = 10 := h2 r
  exact sorry

end NUMINAMATH_GPT_circle_radius_l286_28619


namespace NUMINAMATH_GPT_probability_two_girls_l286_28659

theorem probability_two_girls (total_students girls boys : ℕ) (htotal : total_students = 6) (hg : girls = 4) (hb : boys = 2) :
  (Nat.choose girls 2 / Nat.choose total_students 2 : ℝ) = 2 / 5 := by
  sorry

end NUMINAMATH_GPT_probability_two_girls_l286_28659


namespace NUMINAMATH_GPT_greatest_divisor_l286_28640

theorem greatest_divisor (d : ℕ) :
  (690 % d = 10) ∧ (875 % d = 25) ∧ ∀ e : ℕ, (690 % e = 10) ∧ (875 % e = 25) → (e ≤ d) :=
  sorry

end NUMINAMATH_GPT_greatest_divisor_l286_28640


namespace NUMINAMATH_GPT_ratio_a_to_d_l286_28674

theorem ratio_a_to_d (a b c d : ℚ) 
  (h1 : a / b = 5 / 4) 
  (h2 : b / c = 2 / 3) 
  (h3 : c / d = 3 / 5) : 
  a / d = 1 / 2 :=
sorry

end NUMINAMATH_GPT_ratio_a_to_d_l286_28674


namespace NUMINAMATH_GPT_find_b_value_l286_28696

theorem find_b_value {b : ℚ} (h : -8 ^ 2 + b * -8 - 45 = 0) : b = 19 / 8 :=
sorry

end NUMINAMATH_GPT_find_b_value_l286_28696


namespace NUMINAMATH_GPT_number_of_cars_l286_28667

theorem number_of_cars (C : ℕ) : 
  let bicycles := 3
  let pickup_trucks := 8
  let tricycles := 1
  let car_tires := 4
  let bicycle_tires := 2
  let pickup_truck_tires := 4
  let tricycle_tires := 3
  let total_tires := 101
  (4 * C + 3 * bicycle_tires + 8 * pickup_truck_tires + 1 * tricycle_tires = total_tires) → C = 15 := by
  intros h
  sorry

end NUMINAMATH_GPT_number_of_cars_l286_28667


namespace NUMINAMATH_GPT_smallest_k_with_properties_l286_28612

noncomputable def exists_coloring_and_function (k : ℕ) : Prop :=
  ∃ (colors : ℤ → Fin k) (f : ℤ → ℤ),
    (∀ m n : ℤ, colors m = colors n → f (m + n) = f m + f n) ∧
    (∃ m n : ℤ, f (m + n) ≠ f m + f n)

theorem smallest_k_with_properties : ∃ (k : ℕ), k > 0 ∧ exists_coloring_and_function k ∧
                                         (∀ k' : ℕ, k' > 0 ∧ k' < k → ¬ exists_coloring_and_function k') :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_with_properties_l286_28612


namespace NUMINAMATH_GPT_weight_of_each_bar_l286_28657

theorem weight_of_each_bar 
  (num_bars : ℕ) 
  (cost_per_pound : ℝ) 
  (total_cost : ℝ) 
  (total_weight : ℝ) 
  (weight_per_bar : ℝ)
  (h1 : num_bars = 20)
  (h2 : cost_per_pound = 0.5)
  (h3 : total_cost = 15)
  (h4 : total_weight = total_cost / cost_per_pound)
  (h5 : weight_per_bar = total_weight / num_bars)
  : weight_per_bar = 1.5 := 
by
  sorry

end NUMINAMATH_GPT_weight_of_each_bar_l286_28657


namespace NUMINAMATH_GPT_company_needs_86_workers_l286_28650

def profit_condition (n : ℕ) : Prop :=
  147 * n > 600 + 140 * n

theorem company_needs_86_workers (n : ℕ) : profit_condition n → n ≥ 86 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_company_needs_86_workers_l286_28650


namespace NUMINAMATH_GPT_range_of_a_l286_28668

variables (a b c : ℝ)

theorem range_of_a (h₁ : a^2 - b * c - 8 * a + 7 = 0)
                   (h₂ : b^2 + c^2 + b * c - 6 * a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 :=
sorry

end NUMINAMATH_GPT_range_of_a_l286_28668


namespace NUMINAMATH_GPT_packages_of_gum_l286_28698

-- Define the conditions
variables (P : Nat) -- Number of packages Robin has

-- State the theorem
theorem packages_of_gum (h1 : 7 * P + 6 = 41) : P = 5 :=
by
  sorry

end NUMINAMATH_GPT_packages_of_gum_l286_28698


namespace NUMINAMATH_GPT_line_through_point_bisected_by_hyperbola_l286_28673

theorem line_through_point_bisected_by_hyperbola :
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (a * 3 + b * (-1) + c = 0) ∧
  (∀ x y : ℝ, (x^2 / 4 - y^2 = 1) → (a * x + b * y + c = 0)) ↔ (a = 3 ∧ b = 4 ∧ c = -5) :=
by
  sorry

end NUMINAMATH_GPT_line_through_point_bisected_by_hyperbola_l286_28673


namespace NUMINAMATH_GPT_colored_copies_count_l286_28622

theorem colored_copies_count :
  ∃ C W : ℕ, (C + W = 400) ∧ (10 * C + 5 * W = 2250) ∧ (C = 50) :=
by
  sorry

end NUMINAMATH_GPT_colored_copies_count_l286_28622


namespace NUMINAMATH_GPT_max_value_under_constraint_l286_28631

noncomputable def max_value_expression (a b c : ℝ) : ℝ :=
3 * a * b - 3 * b * c + 2 * c^2

theorem max_value_under_constraint
  (a b c : ℝ)
  (h : a^2 + b^2 + c^2 = 1) :
  max_value_expression a b c ≤ 3 :=
sorry

end NUMINAMATH_GPT_max_value_under_constraint_l286_28631


namespace NUMINAMATH_GPT_smallest_possible_sum_l286_28654

theorem smallest_possible_sum (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hneq : a ≠ b) 
  (heq : (1 / a : ℚ) + (1 / b) = 1 / 12) : a + b = 49 :=
sorry

end NUMINAMATH_GPT_smallest_possible_sum_l286_28654


namespace NUMINAMATH_GPT_area_triangle_l286_28611

theorem area_triangle (A B C: ℝ) (AB AC : ℝ) (h1 : Real.sin A = 4 / 5) (h2 : AB * AC * Real.cos A = 6) :
  (1 / 2) * AB * AC * Real.sin A = 4 :=
by
  sorry

end NUMINAMATH_GPT_area_triangle_l286_28611


namespace NUMINAMATH_GPT_sum_of_two_numbers_l286_28678

theorem sum_of_two_numbers (S : ℝ) (L : ℝ) (h1 : S = 3.5) (h2 : L = 3 * S) : S + L = 14 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l286_28678


namespace NUMINAMATH_GPT_determine_values_of_abc_l286_28656

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
noncomputable def f_inv (a b c : ℝ) (x : ℝ) : ℝ := c * x^2 + b * x + a

theorem determine_values_of_abc 
  (a b c : ℝ) 
  (h_f : ∀ x : ℝ, f a b c (f_inv a b c x) = x)
  (h_f_inv : ∀ x : ℝ, f_inv a b c (f a b c x) = x) : 
  a = -1 ∧ b = 1 ∧ c = 0 :=
by
  sorry

end NUMINAMATH_GPT_determine_values_of_abc_l286_28656


namespace NUMINAMATH_GPT_no_viable_schedule_l286_28614

theorem no_viable_schedule :
  ∀ (studentsA studentsB : ℕ), 
    studentsA = 29 → 
    studentsB = 32 → 
    ¬ ∃ (a b : ℕ),
      (a = 29 ∧ b = 32 ∧
      (a * b = studentsA * studentsB) ∧
      (∀ (x : ℕ), x < studentsA * studentsB →
        ∃ (iA iB : ℕ), 
          iA < studentsA ∧ 
          iB < studentsB ∧ 
          -- The condition that each pair is unique within this period
          ((iA + iB) % (studentsA * studentsB) = x))) := by
  sorry

end NUMINAMATH_GPT_no_viable_schedule_l286_28614


namespace NUMINAMATH_GPT_no_positive_sequence_exists_l286_28671

theorem no_positive_sequence_exists:
  ¬ (∃ (b : ℕ → ℝ), (∀ n, b n > 0) ∧ (∀ m : ℕ, (∑' k, b ((k + 1) * m)) = (1 / m))) :=
by
  sorry

end NUMINAMATH_GPT_no_positive_sequence_exists_l286_28671


namespace NUMINAMATH_GPT_function_increasing_interval_l286_28697

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem function_increasing_interval :
  ∀ x : ℝ, x > 0 → deriv f x > 0 := 
sorry

end NUMINAMATH_GPT_function_increasing_interval_l286_28697


namespace NUMINAMATH_GPT_jillian_max_apartment_size_l286_28632

theorem jillian_max_apartment_size :
  ∀ s : ℝ, (1.10 * s = 880) → s = 800 :=
by
  intros s h
  sorry

end NUMINAMATH_GPT_jillian_max_apartment_size_l286_28632


namespace NUMINAMATH_GPT_projection_is_negative_sqrt_10_l286_28620

noncomputable def projection_of_AB_in_direction_of_AC : ℝ :=
  let A := (1, 1)
  let B := (-3, 3)
  let C := (4, 2)
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  let dot_product := AB.1 * AC.1 + AB.2 * AC.2
  let magnitude_AC := Real.sqrt (AC.1^2 + AC.2^2)
  dot_product / magnitude_AC

theorem projection_is_negative_sqrt_10 :
  projection_of_AB_in_direction_of_AC = -Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_projection_is_negative_sqrt_10_l286_28620


namespace NUMINAMATH_GPT_more_time_running_than_skipping_l286_28679

def time_running : ℚ := 17 / 20
def time_skipping_rope : ℚ := 83 / 100

theorem more_time_running_than_skipping :
  time_running > time_skipping_rope :=
by
  -- sorry skips the proof
  sorry

end NUMINAMATH_GPT_more_time_running_than_skipping_l286_28679


namespace NUMINAMATH_GPT_Sandy_goal_water_l286_28643

-- Definitions based on the conditions in problem a)
def milliliters_per_interval := 500
def time_per_interval := 2
def total_time := 12
def milliliters_to_liters := 1000

-- The goal statement that proves the question == answer given conditions.
theorem Sandy_goal_water : (milliliters_per_interval * (total_time / time_per_interval)) / milliliters_to_liters = 3 := by
  sorry

end NUMINAMATH_GPT_Sandy_goal_water_l286_28643


namespace NUMINAMATH_GPT_arithmetic_seq_a7_l286_28600

theorem arithmetic_seq_a7 (a : ℕ → ℕ) (h1 : a 2 = 2) (h2 : a 4 + a 5 = 12) : a 7 = 10 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_a7_l286_28600


namespace NUMINAMATH_GPT_conic_section_is_ellipse_l286_28618

/-- Given two fixed points (0, 2) and (4, -1) and the equation 
    sqrt(x^2 + (y - 2)^2) + sqrt((x - 4)^2 + (y + 1)^2) = 12, 
    prove that the conic section is an ellipse. -/
theorem conic_section_is_ellipse 
  (x y : ℝ)
  (h : Real.sqrt (x^2 + (y - 2)^2) + Real.sqrt ((x - 4)^2 + (y + 1)^2) = 12) :
  ∃ (F1 F2 : ℝ × ℝ), 
    F1 = (0, 2) ∧ 
    F2 = (4, -1) ∧ 
    ∀ (P : ℝ × ℝ), P = (x, y) → 
      Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) + 
      Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = 12 := 
sorry

end NUMINAMATH_GPT_conic_section_is_ellipse_l286_28618


namespace NUMINAMATH_GPT_positive_difference_of_solutions_is_zero_l286_28684

theorem positive_difference_of_solutions_is_zero : ∀ (x : ℂ), (x ^ 2 + 3 * x + 4 = 0) → 
  ∀ (y : ℂ), (y ^ 2 + 3 * y + 4 = 0) → |y.re - x.re| = 0 :=
by
  intro x hx y hy
  sorry

end NUMINAMATH_GPT_positive_difference_of_solutions_is_zero_l286_28684


namespace NUMINAMATH_GPT_problem1_problem2_l286_28637

theorem problem1 (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  abs ((a + b) / (a - b)) + abs ((b + c) / (b - c)) + abs ((c + a) / (c - a)) ≥ 2 :=
sorry

theorem problem2 (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  abs ((a + b) / (a - b)) + abs ((b + c) / (b - c)) + abs ((c + a) / (c - a)) > 3 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l286_28637


namespace NUMINAMATH_GPT_trig_expression_value_l286_28639

theorem trig_expression_value (α : ℝ) (h : Real.tan α = 3) :
  2 * Real.sin α ^ 2 + 4 * Real.sin α * Real.cos α - 9 * Real.cos α ^ 2 = 21 / 10 :=
by
  sorry

end NUMINAMATH_GPT_trig_expression_value_l286_28639


namespace NUMINAMATH_GPT_scientific_notation_of_125000_l286_28681

theorem scientific_notation_of_125000 :
  125000 = 1.25 * 10^5 := sorry

end NUMINAMATH_GPT_scientific_notation_of_125000_l286_28681


namespace NUMINAMATH_GPT_number_of_groups_l286_28666

theorem number_of_groups (max min c : ℕ) (h_max : max = 140) (h_min : min = 50) (h_c : c = 10) : 
  (max - min) / c + 1 = 10 := 
by
  sorry

end NUMINAMATH_GPT_number_of_groups_l286_28666


namespace NUMINAMATH_GPT_photo_arrangement_l286_28634

noncomputable def valid_arrangements (teacher boys girls : ℕ) : ℕ :=
  if girls = 2 ∧ teacher = 1 ∧ boys = 2 then 24 else 0

theorem photo_arrangement :
  valid_arrangements 1 2 2 = 24 :=
by {
  -- The proof goes here.
  sorry
}

end NUMINAMATH_GPT_photo_arrangement_l286_28634


namespace NUMINAMATH_GPT_f_decreasing_on_0_1_l286_28648

noncomputable def f (x : ℝ) : ℝ := x + 1 / x

theorem f_decreasing_on_0_1 : ∀ (x1 x2 : ℝ), (x1 ∈ Set.Ioo 0 1) → (x2 ∈ Set.Ioo 0 1) → (x1 < x2) → (f x1 < f x2) := by
  sorry

end NUMINAMATH_GPT_f_decreasing_on_0_1_l286_28648


namespace NUMINAMATH_GPT_arrangement_of_accommodation_l286_28628

open Nat

noncomputable def num_arrangements_accommodation : ℕ :=
  (factorial 13) / ((factorial 2) * (factorial 2) * (factorial 2) * (factorial 2))

theorem arrangement_of_accommodation : num_arrangements_accommodation = 389188800 := by
  sorry

end NUMINAMATH_GPT_arrangement_of_accommodation_l286_28628


namespace NUMINAMATH_GPT_quiz_answer_key_count_l286_28689

theorem quiz_answer_key_count :
  let tf_combinations := 6 -- Combinations of true-false questions
  let mc_combinations := 4 ^ 3 -- Combinations of multiple-choice questions
  tf_combinations * mc_combinations = 384 := by
  -- The values and conditions are directly taken from the problem statement.
  let tf_combinations := 6
  let mc_combinations := 4 ^ 3
  sorry

end NUMINAMATH_GPT_quiz_answer_key_count_l286_28689


namespace NUMINAMATH_GPT_total_theme_parks_l286_28641

theorem total_theme_parks 
  (J V M N : ℕ) 
  (hJ : J = 35)
  (hV : V = J + 40)
  (hM : M = J + 60)
  (hN : N = 2 * M) 
  : J + V + M + N = 395 :=
sorry

end NUMINAMATH_GPT_total_theme_parks_l286_28641


namespace NUMINAMATH_GPT_cone_lateral_area_l286_28672

theorem cone_lateral_area (r l : ℝ) (h_r : r = 3) (h_l : l = 5) : 
  π * r * l = 15 * π := by
  sorry

end NUMINAMATH_GPT_cone_lateral_area_l286_28672


namespace NUMINAMATH_GPT_find_f_four_thirds_l286_28691

def f (y: ℝ) : ℝ := sorry  -- Placeholder for the function definition

theorem find_f_four_thirds : f (4 / 3) = - (7 / 2) := sorry

end NUMINAMATH_GPT_find_f_four_thirds_l286_28691


namespace NUMINAMATH_GPT_factorization_identity_l286_28683

theorem factorization_identity (m : ℝ) : m^3 - m = m * (m + 1) * (m - 1) :=
by
  sorry

end NUMINAMATH_GPT_factorization_identity_l286_28683


namespace NUMINAMATH_GPT_jellybeans_in_new_bag_l286_28646

theorem jellybeans_in_new_bag (average_per_bag : ℕ) (num_bags : ℕ) (additional_avg_increase : ℕ) (total_jellybeans_old : ℕ) (total_jellybeans_new : ℕ) (num_bags_new : ℕ) (new_bag_jellybeans : ℕ) : 
  average_per_bag = 117 → 
  num_bags = 34 → 
  additional_avg_increase = 7 → 
  total_jellybeans_old = num_bags * average_per_bag → 
  total_jellybeans_new = (num_bags + 1) * (average_per_bag + additional_avg_increase) → 
  new_bag_jellybeans = total_jellybeans_new - total_jellybeans_old → 
  new_bag_jellybeans = 362 := 
by 
  intros 
  sorry

end NUMINAMATH_GPT_jellybeans_in_new_bag_l286_28646


namespace NUMINAMATH_GPT_club_men_count_l286_28609

theorem club_men_count (M W : ℕ) (h1 : M + W = 30) (h2 : M + (W / 3 : ℕ) = 20) : M = 15 := by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_club_men_count_l286_28609


namespace NUMINAMATH_GPT_bird_families_flew_away_to_Africa_l286_28630

theorem bird_families_flew_away_to_Africa 
  (B : ℕ) (n : ℕ) (hB94 : B = 94) (hB_A_plus_n : B = n + 47) : n = 47 :=
by
  sorry

end NUMINAMATH_GPT_bird_families_flew_away_to_Africa_l286_28630


namespace NUMINAMATH_GPT_percent_of_dollar_in_pocket_l286_28638

def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def half_dollar_value : ℕ := 50

theorem percent_of_dollar_in_pocket :
  let total_cents := penny_value + nickel_value + dime_value + quarter_value + half_dollar_value
  total_cents = 91 := by
  sorry

end NUMINAMATH_GPT_percent_of_dollar_in_pocket_l286_28638


namespace NUMINAMATH_GPT_fill_bucket_time_l286_28644

theorem fill_bucket_time (time_full_bucket : ℕ) (fraction : ℚ) (time_two_thirds_bucket : ℕ) 
  (h1 : time_full_bucket = 150) (h2 : fraction = 2 / 3) : time_two_thirds_bucket = 100 :=
sorry

end NUMINAMATH_GPT_fill_bucket_time_l286_28644


namespace NUMINAMATH_GPT_range_of_m_l286_28626

noncomputable def equation_has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, 25^(-|x+1|) - 4 * 5^(-|x+1|) - m = 0

theorem range_of_m : ∀ m : ℝ, equation_has_real_roots m ↔ (-3 ≤ m ∧ m < 0) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_range_of_m_l286_28626


namespace NUMINAMATH_GPT_find_M_value_when_x_3_l286_28686

-- Definitions based on the given conditions
def polynomial (a b c d x : ℝ) : ℝ := a*x^5 + b*x^3 + c*x + d

-- Given conditions
variables (a b c d : ℝ)
axiom h₀ : polynomial a b c d 0 = -5
axiom h₁ : polynomial a b c d (-3) = 7

-- Desired statement: Prove that the value of polynomial at x = 3 is -17
theorem find_M_value_when_x_3 : polynomial a b c d 3 = -17 :=
by sorry

end NUMINAMATH_GPT_find_M_value_when_x_3_l286_28686


namespace NUMINAMATH_GPT_subset_relation_l286_28688

variables (M N : Set ℕ) 

theorem subset_relation (hM : M = {1, 2, 3, 4}) (hN : N = {2, 3, 4}) : N ⊆ M :=
sorry

end NUMINAMATH_GPT_subset_relation_l286_28688


namespace NUMINAMATH_GPT_frequency_of_group_5_l286_28693

/-- Let the total number of data points be 50, number of data points in groups 1, 2, 3, and 4 be
  2, 8, 15, and 5 respectively. Prove that the frequency of group 5 is 0.4. -/
theorem frequency_of_group_5 :
  let total_data_points := 50
  let group1_data_points := 2
  let group2_data_points := 8
  let group3_data_points := 15
  let group4_data_points := 5
  let group5_data_points := total_data_points - group1_data_points - group2_data_points - group3_data_points - group4_data_points
  let frequency_group5 := (group5_data_points : ℝ) / total_data_points
  frequency_group5 = 0.4 := 
by
  sorry

end NUMINAMATH_GPT_frequency_of_group_5_l286_28693


namespace NUMINAMATH_GPT_trigonometric_identity_l286_28694

theorem trigonometric_identity (α : Real) (h : Real.sin (Real.pi + α) = -1/3) : 
  (Real.sin (2 * α) / Real.cos α) = 2/3 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l286_28694


namespace NUMINAMATH_GPT_surface_area_of_solid_block_l286_28695

theorem surface_area_of_solid_block :
  let unit_cube_surface_area := 6
  let top_bottom_area := 2 * (3 * 5)
  let front_back_area := 2 * (3 * 5)
  let left_right_area := 2 * (3 * 1)
  top_bottom_area + front_back_area + left_right_area = 66 :=
by
  let unit_cube_surface_area := 6
  let top_bottom_area := 2 * (3 * 5)
  let front_back_area := 2 * (3 * 5)
  let left_right_area := 2 * (3 * 1)
  sorry

end NUMINAMATH_GPT_surface_area_of_solid_block_l286_28695


namespace NUMINAMATH_GPT_resulting_solid_faces_l286_28660

-- Define a cube structure with a given number of faces
structure Cube where
  faces : Nat

-- Define the problem conditions and prove the total faces of the resulting solid
def original_cube := Cube.mk 6

def new_faces_per_cube := 5

def total_new_faces := original_cube.faces * new_faces_per_cube

def total_faces_of_resulting_solid := total_new_faces + original_cube.faces

theorem resulting_solid_faces : total_faces_of_resulting_solid = 36 := by
  sorry

end NUMINAMATH_GPT_resulting_solid_faces_l286_28660


namespace NUMINAMATH_GPT_inequality_proof_l286_28658

theorem inequality_proof (a b : ℝ) (h : a > b ∧ b > 0) : 
  (a - b)^2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧ (a + b) / 2 - Real.sqrt (a * b) < (a - b)^2 / (8 * b) := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l286_28658


namespace NUMINAMATH_GPT_glass_volume_correct_l286_28699

-- Define the constants based on the problem conditions
def pessimist_empty_percent : ℝ := 0.60
def optimist_full_percent : ℝ := 0.60
def water_difference : ℝ := 46

-- Let V be the volume of the glass
def glass_volume (V : ℝ) : Prop :=
  let pessimist_full := (1 - pessimist_empty_percent) * V
  let optimist_full := optimist_full_percent * V
  optimist_full - pessimist_full = water_difference

-- The theorem to prove
theorem glass_volume_correct :
  ∃ V : ℝ, glass_volume V ∧ V = 230 :=
by
  sorry

end NUMINAMATH_GPT_glass_volume_correct_l286_28699


namespace NUMINAMATH_GPT_locus_of_centers_of_circles_l286_28676

structure Point (α : Type _) :=
(x : α)
(y : α)

noncomputable def perpendicular_bisector {α : Type _} [LinearOrderedField α] (A B : Point α) : Set (Point α) :=
  {C | ∃ m b : α, C.y = m * C.x + b ∧ A.y = m * A.x + b ∧ B.y = m * B.x + b ∧
                 (A.x - B.x) * C.x + (A.y - B.y) * C.y = (A.x^2 + A.y^2 - B.x^2 - B.y^2) / 2}

theorem locus_of_centers_of_circles {α : Type _} [LinearOrderedField α] (A B : Point α) :
  (∀ (C : Point α), (∃ r : α, r > 0 ∧ ∃ k: α, (C.x - A.x)^2 + (C.y - A.y)^2 = r^2 ∧ (C.x - B.x)^2 + (C.y - B.y)^2 = r^2) 
  → C ∈ perpendicular_bisector A B) :=
by
  sorry

end NUMINAMATH_GPT_locus_of_centers_of_circles_l286_28676


namespace NUMINAMATH_GPT_baseball_team_games_l286_28602

theorem baseball_team_games (P Q : ℕ) (hP : P > 3 * Q) (hQ : Q > 3) (hTotal : 2 * P + 6 * Q = 78) :
  2 * P = 54 :=
by
  -- placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_baseball_team_games_l286_28602


namespace NUMINAMATH_GPT_solution_set_abs_inequality_l286_28606

theorem solution_set_abs_inequality (x : ℝ) : (|x - 1| ≤ 2) ↔ (-1 ≤ x ∧ x ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_abs_inequality_l286_28606


namespace NUMINAMATH_GPT_triangle_area_l286_28687

theorem triangle_area (a b c : ℝ) (ha : a = 6) (hb : b = 5) (hc : c = 5) (isosceles : a = 2 * b) :
  let s := (a + b + c) / 2
  let area := (s * (s - a) * (s - b) * (s - c)).sqrt
  area = 12 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l286_28687
