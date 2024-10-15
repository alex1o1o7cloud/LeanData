import Mathlib

namespace NUMINAMATH_GPT_integer_base10_from_bases_l1536_153644

theorem integer_base10_from_bases (C D : ℕ) (hC : 0 ≤ C ∧ C ≤ 7) (hD : 0 ≤ D ∧ D ≤ 5)
    (h : 8 * C + D = 6 * D + C) : C = 0 ∧ D = 0 ∧ (8 * C + D = 0) := by
  sorry

end NUMINAMATH_GPT_integer_base10_from_bases_l1536_153644


namespace NUMINAMATH_GPT_inequality_solution_l1536_153671

theorem inequality_solution (x : ℝ) : 3 * x + 2 ≥ 5 ↔ x ≥ 1 :=
by sorry

end NUMINAMATH_GPT_inequality_solution_l1536_153671


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l1536_153687

-- Define the first system of equations and its solution
theorem system1_solution (x y : ℝ) : 
    (3 * (x - 1) = y + 5 ∧ 5 * (y - 1) = 3 * (x + 5)) ↔ (x = 5 ∧ y = 7) :=
sorry

-- Define the second system of equations and its solution
theorem system2_solution (x y a : ℝ) :
    (2 * x + 4 * y = a ∧ 7 * x - 2 * y = 3 * a) ↔ 
    (x = (7 / 16) * a ∧ y = (1 / 32) * a) :=
sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l1536_153687


namespace NUMINAMATH_GPT_remainder_when_divided_by_8_l1536_153652

theorem remainder_when_divided_by_8 (x : ℤ) (k : ℤ) (h : x = 72 * k + 19) : x % 8 = 3 :=
by sorry

end NUMINAMATH_GPT_remainder_when_divided_by_8_l1536_153652


namespace NUMINAMATH_GPT_initial_bottle_count_l1536_153650

variable (B: ℕ)

-- Conditions: Each bottle holds 15 stars, bought 3 more bottles, total 75 stars to fill
def bottle_capacity := 15
def additional_bottles := 3
def total_stars := 75

-- The main statement we want to prove
theorem initial_bottle_count (h : (B + additional_bottles) * bottle_capacity = total_stars) : 
    B = 2 :=
by sorry

end NUMINAMATH_GPT_initial_bottle_count_l1536_153650


namespace NUMINAMATH_GPT_sqrt_x_eq_0_123_l1536_153636

theorem sqrt_x_eq_0_123 (x : ℝ) (h1 : Real.sqrt 15129 = 123) (h2 : Real.sqrt x = 0.123) : x = 0.015129 := by
  -- proof goes here, but it is omitted
  sorry

end NUMINAMATH_GPT_sqrt_x_eq_0_123_l1536_153636


namespace NUMINAMATH_GPT_choice_of_b_l1536_153691

noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x - 2)
noncomputable def g (x : ℝ) : ℝ := f (x + 3)

theorem choice_of_b (b : ℝ) :
  (g (g x) = x) ↔ (b = -4) :=
sorry

end NUMINAMATH_GPT_choice_of_b_l1536_153691


namespace NUMINAMATH_GPT_original_fraction_2_7_l1536_153600

theorem original_fraction_2_7 (N D : ℚ) : 
  (1.40 * N) / (0.50 * D) = 4 / 5 → N / D = 2 / 7 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_original_fraction_2_7_l1536_153600


namespace NUMINAMATH_GPT_height_difference_l1536_153619

variable (h_A h_B h_D h_E h_F h_G : ℝ)

theorem height_difference :
  (h_A - h_D = 4.5) →
  (h_E - h_D = -1.7) →
  (h_F - h_E = -0.8) →
  (h_G - h_F = 1.9) →
  (h_B - h_G = 3.6) →
  (h_A - h_B > 0) :=
by
  intro h_AD h_ED h_FE h_GF h_BG
  sorry

end NUMINAMATH_GPT_height_difference_l1536_153619


namespace NUMINAMATH_GPT_find_numbers_l1536_153655

def is_7_digit (n : ℕ) : Prop := n ≥ 1000000 ∧ n < 10000000
def is_14_digit (n : ℕ) : Prop := n >= 10^13 ∧ n < 10^14

theorem find_numbers (x y z : ℕ) (hx7 : is_7_digit x) (hy7 : is_7_digit y) (hz14 : is_14_digit z) :
  3 * x * y = z ∧ z = 10^7 * x + y → 
  x = 1666667 ∧ y = 3333334 ∧ z = 16666673333334 := 
by
  sorry

end NUMINAMATH_GPT_find_numbers_l1536_153655


namespace NUMINAMATH_GPT_sin_double_angle_l1536_153614

theorem sin_double_angle (x : ℝ) (h : Real.tan (π / 4 - x) = 2) : Real.sin (2 * x) = -3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l1536_153614


namespace NUMINAMATH_GPT_average_gpa_of_whole_class_l1536_153676

-- Define the conditions
variables (n : ℕ)
def num_students_in_group1 := n / 3
def num_students_in_group2 := 2 * n / 3

def gpa_group1 := 15
def gpa_group2 := 18

-- Lean statement for the proof problem
theorem average_gpa_of_whole_class (hn_pos : 0 < n):
  ((num_students_in_group1 * gpa_group1) + (num_students_in_group2 * gpa_group2)) / n = 17 :=
sorry

end NUMINAMATH_GPT_average_gpa_of_whole_class_l1536_153676


namespace NUMINAMATH_GPT_evaluate_expression_evaluate_fraction_l1536_153685

theorem evaluate_expression (x y : ℕ) (hx : x = 3) (hy : y = 4) : 
  3 * x^3 + 4 * y^3 = 337 :=
by
  sorry

theorem evaluate_fraction (x y : ℕ) (hx : x = 3) (hy : y = 4) 
  (h : 3 * x^3 + 4 * y^3 = 337) :
  (3 * x^3 + 4 * y^3) / 9 = 37 + 4/9 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_evaluate_fraction_l1536_153685


namespace NUMINAMATH_GPT_max_sum_unique_digits_expression_equivalent_l1536_153667

theorem max_sum_unique_digits_expression_equivalent :
  ∃ (a b c d e : ℕ), (2 * 19 * 53 = 2014) ∧ 
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧
    (2 * (b + c) * (d + e) = 2014) ∧
    (a + b + c + d + e = 35) ∧ 
    (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10) :=
by
  sorry

end NUMINAMATH_GPT_max_sum_unique_digits_expression_equivalent_l1536_153667


namespace NUMINAMATH_GPT_length_of_side_b_l1536_153623

theorem length_of_side_b (B C : ℝ) (c b : ℝ) (hB : B = 45 * Real.pi / 180) (hC : C = 60 * Real.pi / 180) (hc : c = 1) :
  b = Real.sqrt 6 / 3 :=
by
  sorry

end NUMINAMATH_GPT_length_of_side_b_l1536_153623


namespace NUMINAMATH_GPT_area_of_square_on_RS_l1536_153628

theorem area_of_square_on_RS (PQ QR PS PS_square PQ_square QR_square : ℝ)
  (hPQ : PQ_square = 25) (hQR : QR_square = 49) (hPS : PS_square = 64)
  (hPQ_eq : PQ_square = PQ^2) (hQR_eq : QR_square = QR^2) (hPS_eq : PS_square = PS^2)
  : ∃ RS_square : ℝ, RS_square = 138 := by
  let PR_square := PQ^2 + QR^2
  let RS_square := PR_square + PS^2
  use RS_square
  sorry

end NUMINAMATH_GPT_area_of_square_on_RS_l1536_153628


namespace NUMINAMATH_GPT_compute_a_l1536_153633

theorem compute_a (a b : ℚ) 
  (h_root1 : (-1:ℚ) - 5 * (Real.sqrt 3) = -1 - 5 * (Real.sqrt 3))
  (h_rational1 : (-1:ℚ) + 5 * (Real.sqrt 3) = -1 + 5 * (Real.sqrt 3))
  (h_poly : ∀ x, x^3 + a*x^2 + b*x + 48 = 0) :
  a = 50 / 37 :=
by
  sorry

end NUMINAMATH_GPT_compute_a_l1536_153633


namespace NUMINAMATH_GPT_largest_number_not_sum_of_two_composites_l1536_153656

-- Define composite numbers
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

-- Define the problem predicate
def cannot_be_expressed_as_sum_of_two_composites (n : ℕ) : Prop :=
  ¬ ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_number_not_sum_of_two_composites : 
  ∃ n, cannot_be_expressed_as_sum_of_two_composites n ∧ 
  (∀ m, cannot_be_expressed_as_sum_of_two_composites m → m ≤ n) ∧ 
  n = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_number_not_sum_of_two_composites_l1536_153656


namespace NUMINAMATH_GPT_original_ratio_l1536_153651

theorem original_ratio (x y : ℤ) (h₁ : y = 72) (h₂ : (x + 6) / y = 1 / 3) : y / x = 4 := 
by
  sorry

end NUMINAMATH_GPT_original_ratio_l1536_153651


namespace NUMINAMATH_GPT_problem1_problem2_l1536_153612

theorem problem1 : ∃ (m : ℝ) (b : ℝ), ∀ (x y : ℝ),
  3 * x + 4 * y - 2 = 0 ∧ x - y + 4 = 0 →
  y = m * x + b ∧ (1 / m = -2) ∧ (y = - (2 * x + 2)) :=
sorry

theorem problem2 : ∀ (x y a : ℝ), (x = -1) ∧ (y = 3) → 
  (x + y = a) →
  a = 2 ∧ (x + y - 2 = 0) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1536_153612


namespace NUMINAMATH_GPT_solve_for_n_l1536_153640

theorem solve_for_n (n : ℕ) (h : (16^n) * (16^n) * (16^n) * (16^n) * (16^n) = 256^5) : n = 2 := by
  sorry

end NUMINAMATH_GPT_solve_for_n_l1536_153640


namespace NUMINAMATH_GPT_mrs_hilt_initial_marbles_l1536_153674

theorem mrs_hilt_initial_marbles (lost_marble : ℕ) (remaining_marble : ℕ) (h1 : lost_marble = 15) (h2 : remaining_marble = 23) : 
    (remaining_marble + lost_marble) = 38 :=
by
  sorry

end NUMINAMATH_GPT_mrs_hilt_initial_marbles_l1536_153674


namespace NUMINAMATH_GPT_ratio_of_runs_l1536_153632

theorem ratio_of_runs (A B C : ℕ) (h1 : B = C / 5) (h2 : A + B + C = 95) (h3 : C = 75) :
  A / B = 1 / 3 :=
by sorry

end NUMINAMATH_GPT_ratio_of_runs_l1536_153632


namespace NUMINAMATH_GPT_proportional_x_y2_y_z2_l1536_153673

variable {x y z k m c : ℝ}

theorem proportional_x_y2_y_z2 (h1 : x = k * y^2) (h2 : y = m / z^2) (h3 : x = 2) (hz4 : z = 4) (hz16 : z = 16):
  x = 1/128 :=
by
  sorry

end NUMINAMATH_GPT_proportional_x_y2_y_z2_l1536_153673


namespace NUMINAMATH_GPT_slices_left_for_Era_l1536_153634

def total_burgers : ℕ := 5
def slices_per_burger : ℕ := 8

def first_friend_slices : ℕ := 3
def second_friend_slices : ℕ := 8
def third_friend_slices : ℕ := 5
def fourth_friend_slices : ℕ := 11
def fifth_friend_slices : ℕ := 6

def total_slices : ℕ := total_burgers * slices_per_burger
def slices_given_to_friends : ℕ := first_friend_slices + second_friend_slices + third_friend_slices + fourth_friend_slices + fifth_friend_slices

theorem slices_left_for_Era : total_slices - slices_given_to_friends = 7 :=
by
  rw [total_slices, slices_given_to_friends]
  exact Eq.refl 7

#reduce slices_left_for_Era

end NUMINAMATH_GPT_slices_left_for_Era_l1536_153634


namespace NUMINAMATH_GPT_quadratic_root_sum_eight_l1536_153641

theorem quadratic_root_sum_eight (p r : ℝ) (hp : p > 0) (hr : r > 0) 
  (h : ∀ (x₁ x₂ : ℝ), (x₁ + x₂ = p) -> (x₁ * x₂ = r) -> (x₁ + x₂ = 8)) : r = 8 :=
sorry

end NUMINAMATH_GPT_quadratic_root_sum_eight_l1536_153641


namespace NUMINAMATH_GPT_rachel_math_homework_l1536_153665

def rachel_homework (M : ℕ) (reading : ℕ) (biology : ℕ) (total : ℕ) : Prop :=
reading = 3 ∧ biology = 10 ∧ total = 15 ∧ reading + biology + M = total

theorem rachel_math_homework: ∃ M : ℕ, rachel_homework M 3 10 15 ∧ M = 2 := 
by 
  sorry

end NUMINAMATH_GPT_rachel_math_homework_l1536_153665


namespace NUMINAMATH_GPT_candy_total_l1536_153688

theorem candy_total (n m : ℕ) (h1 : n = 2) (h2 : m = 8) : n * m = 16 :=
by
  -- This will contain the proof
  sorry

end NUMINAMATH_GPT_candy_total_l1536_153688


namespace NUMINAMATH_GPT_leak_takes_3_hours_to_empty_l1536_153696

noncomputable def leak_emptying_time (inlet_rate_per_minute: ℕ) (tank_empty_time_with_inlet: ℕ) (tank_capacity: ℕ) : ℕ :=
  let inlet_rate_per_hour := inlet_rate_per_minute * 60
  let effective_empty_rate := tank_capacity / tank_empty_time_with_inlet
  let leak_rate := inlet_rate_per_hour + effective_empty_rate
  tank_capacity / leak_rate

theorem leak_takes_3_hours_to_empty:
  leak_emptying_time 6 12 1440 = 3 := 
sorry

end NUMINAMATH_GPT_leak_takes_3_hours_to_empty_l1536_153696


namespace NUMINAMATH_GPT_find_ordered_triple_l1536_153607

theorem find_ordered_triple (a b c : ℝ) (h₁ : 2 < a) (h₂ : 2 < b) (h₃ : 2 < c)
    (h_eq : (a + 1)^2 / (b + c - 1) + (b + 2)^2 / (c + a - 3) + (c + 3)^2 / (a + b - 5) = 32) :
    (a = 8 ∧ b = 6 ∧ c = 5) :=
sorry

end NUMINAMATH_GPT_find_ordered_triple_l1536_153607


namespace NUMINAMATH_GPT_multiple_with_digits_l1536_153605

theorem multiple_with_digits (n : ℕ) (h : n > 0) :
  ∃ (m : ℕ), (m % n = 0) ∧ (m < 10 ^ n) ∧ (∀ d ∈ m.digits 10, d = 0 ∨ d = 1) :=
by
  sorry

end NUMINAMATH_GPT_multiple_with_digits_l1536_153605


namespace NUMINAMATH_GPT_find_number_l1536_153699

-- Definitions used in the given problem conditions
def condition (x : ℝ) : Prop := (3.242 * x) / 100 = 0.04863

-- Statement of the problem
theorem find_number (x : ℝ) (h : condition x) : x = 1.5 :=
by
  sorry
 
end NUMINAMATH_GPT_find_number_l1536_153699


namespace NUMINAMATH_GPT_sum_of_palindromes_l1536_153653

theorem sum_of_palindromes (a b : ℕ) (ha : a > 99) (ha' : a < 1000) (hb : b > 99) (hb' : b < 1000) 
  (hpal_a : ∀ i j k, a = 100*i + 10*j + k → a = 100*k + 10*j + i) 
  (hpal_b : ∀ i j k, b = 100*i + 10*j + k → b = 100*k + 10*j + i) 
  (hprod : a * b = 589185) : a + b = 1534 :=
sorry

end NUMINAMATH_GPT_sum_of_palindromes_l1536_153653


namespace NUMINAMATH_GPT_spoons_needed_to_fill_cup_l1536_153643

-- Define necessary conditions
def spoon_capacity : Nat := 5
def liter_to_milliliters : Nat := 1000

-- State the problem
theorem spoons_needed_to_fill_cup : liter_to_milliliters / spoon_capacity = 200 := 
by 
  -- Skip the actual proof
  sorry

end NUMINAMATH_GPT_spoons_needed_to_fill_cup_l1536_153643


namespace NUMINAMATH_GPT_ratio_of_percent_increase_to_decrease_l1536_153694

variable (P U V : ℝ)
variable (h1 : P * U = 0.25 * P * V)
variable (h2 : P ≠ 0)

theorem ratio_of_percent_increase_to_decrease (h : U = 0.25 * V) :
  ((V - U) / U) * 100 / 75 = 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_percent_increase_to_decrease_l1536_153694


namespace NUMINAMATH_GPT_points_among_transformations_within_square_l1536_153606

def projection_side1 (A : ℝ × ℝ) : ℝ × ℝ := (A.1, 2 - A.2)
def projection_side2 (A : ℝ × ℝ) : ℝ × ℝ := (-A.1, A.2)
def projection_side3 (A : ℝ × ℝ) : ℝ × ℝ := (A.1, -A.2)
def projection_side4 (A : ℝ × ℝ) : ℝ × ℝ := (2 - A.1, A.2)

def within_square (A : ℝ × ℝ) : Prop := 
  0 ≤ A.1 ∧ A.1 ≤ 1 ∧ 0 ≤ A.2 ∧ A.2 ≤ 1

theorem points_among_transformations_within_square (A : ℝ × ℝ)
  (H1 : within_square A)
  (H2 : within_square (projection_side1 A))
  (H3 : within_square (projection_side2 (projection_side1 A)))
  (H4 : within_square (projection_side3 (projection_side2 (projection_side1 A))))
  (H5 : within_square (projection_side4 (projection_side3 (projection_side2 (projection_side1 A))))) :
  A = (1 / 3, 1 / 3) := sorry

end NUMINAMATH_GPT_points_among_transformations_within_square_l1536_153606


namespace NUMINAMATH_GPT_jewelry_store_total_cost_l1536_153630

theorem jewelry_store_total_cost :
  let necklaces_needed := 7
  let rings_needed := 12
  let bracelets_needed := 7
  let necklace_price := 4
  let ring_price := 10
  let bracelet_price := 5
  let necklace_discount := if necklaces_needed >= 6 then 0.15 else if necklaces_needed >= 4 then 0.10 else 0
  let ring_discount := if rings_needed >= 20 then 0.10 else if rings_needed >= 10 then 0.05 else 0
  let bracelet_discount := if bracelets_needed >= 10 then 0.12 else if bracelets_needed >= 7 then 0.08 else 0
  let necklace_cost := necklaces_needed * (necklace_price * (1 - necklace_discount))
  let ring_cost := rings_needed * (ring_price * (1 - ring_discount))
  let bracelet_cost := bracelets_needed * (bracelet_price * (1 - bracelet_discount))
  let total_cost := necklace_cost + ring_cost + bracelet_cost
  total_cost = 170 := by
  -- calculation details omitted
  sorry

end NUMINAMATH_GPT_jewelry_store_total_cost_l1536_153630


namespace NUMINAMATH_GPT_logs_needed_l1536_153610

theorem logs_needed (needed_woodblocks : ℕ) (current_logs : ℕ) (woodblocks_per_log : ℕ) 
  (H1 : needed_woodblocks = 80) 
  (H2 : current_logs = 8) 
  (H3 : woodblocks_per_log = 5) : 
  current_logs * woodblocks_per_log < needed_woodblocks → 
  (needed_woodblocks - current_logs * woodblocks_per_log) / woodblocks_per_log = 8 := by
  sorry

end NUMINAMATH_GPT_logs_needed_l1536_153610


namespace NUMINAMATH_GPT_inequality_problem_l1536_153625

theorem inequality_problem (x a : ℝ) (h1 : x < a) (h2 : a < 0) : x^2 > ax ∧ ax > a^2 :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_problem_l1536_153625


namespace NUMINAMATH_GPT_man_double_son_in_years_l1536_153608

-- Definitions of conditions
def son_age : ℕ := 18
def man_age : ℕ := son_age + 20

-- The proof problem statement
theorem man_double_son_in_years :
  ∃ (X : ℕ), (man_age + X = 2 * (son_age + X)) ∧ X = 2 :=
by
  sorry

end NUMINAMATH_GPT_man_double_son_in_years_l1536_153608


namespace NUMINAMATH_GPT_solve_problem_l1536_153602

open Complex

noncomputable def problem_statement (a : ℝ) : Prop :=
  abs ((a : ℂ) + I) / abs I = 2
  
theorem solve_problem {a : ℝ} : problem_statement a → a = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_problem_l1536_153602


namespace NUMINAMATH_GPT_array_sum_remainder_l1536_153657

def entry_value (r c : ℕ) : ℚ :=
  (1 / (2 * 1013) ^ r) * (1 / 1013 ^ c)

def array_sum : ℚ :=
  (1 / (2 * 1013 - 1)) * (1 / (1013 - 1))

def m : ℤ := 1
def n : ℤ := 2046300
def mn_sum : ℤ := m + n

theorem array_sum_remainder :
  (mn_sum % 1013) = 442 :=
by
  sorry

end NUMINAMATH_GPT_array_sum_remainder_l1536_153657


namespace NUMINAMATH_GPT_find_a6_l1536_153695

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem find_a6 (a : ℕ → ℝ) (h : arithmetic_sequence a) (h2 : a 2 = 4) (h4 : a 4 = 2) : a 6 = 0 :=
by sorry

end NUMINAMATH_GPT_find_a6_l1536_153695


namespace NUMINAMATH_GPT_option_c_is_not_equal_l1536_153627

theorem option_c_is_not_equal :
  let A := 14 / 12
  let B := 1 + 1 / 6
  let C := 1 + 1 / 2
  let D := 1 + 7 / 42
  let E := 1 + 14 / 84
  A = 7 / 6 ∧ B = 7 / 6 ∧ D = 7 / 6 ∧ E = 7 / 6 ∧ C ≠ 7 / 6 :=
by
  sorry

end NUMINAMATH_GPT_option_c_is_not_equal_l1536_153627


namespace NUMINAMATH_GPT_inverse_mod_187_l1536_153603

theorem inverse_mod_187 : ∃ (x : ℤ), 0 ≤ x ∧ x ≤ 186 ∧ (2 * x) % 187 = 1 :=
by
  use 94
  sorry

end NUMINAMATH_GPT_inverse_mod_187_l1536_153603


namespace NUMINAMATH_GPT_Chemistry_marks_l1536_153624

theorem Chemistry_marks (english_marks mathematics_marks physics_marks biology_marks : ℕ) (avg_marks : ℝ) (num_subjects : ℕ) (total_marks : ℕ)
  (h1 : english_marks = 72)
  (h2 : mathematics_marks = 60)
  (h3 : physics_marks = 35)
  (h4 : biology_marks = 84)
  (h5 : avg_marks = 62.6)
  (h6 : num_subjects = 5)
  (h7 : total_marks = avg_marks * num_subjects) :
  (total_marks - (english_marks + mathematics_marks + physics_marks + biology_marks) = 62) :=
by
  sorry

end NUMINAMATH_GPT_Chemistry_marks_l1536_153624


namespace NUMINAMATH_GPT_not_divisible_by_5_square_plus_or_minus_1_divisible_by_5_l1536_153670

theorem not_divisible_by_5_square_plus_or_minus_1_divisible_by_5 (a : ℤ) (h : a % 5 ≠ 0) :
  (a^2 + 1) % 5 = 0 ∨ (a^2 - 1) % 5 = 0 :=
by
  sorry

end NUMINAMATH_GPT_not_divisible_by_5_square_plus_or_minus_1_divisible_by_5_l1536_153670


namespace NUMINAMATH_GPT_n_prime_of_divisors_l1536_153662

theorem n_prime_of_divisors (n k : ℕ) (h₁ : n > 1) 
  (h₂ : ∀ d : ℕ, d ∣ n → (d + k ∣ n) ∨ (d - k ∣ n)) : Prime n :=
  sorry

end NUMINAMATH_GPT_n_prime_of_divisors_l1536_153662


namespace NUMINAMATH_GPT_cube_sum_gt_l1536_153672

variable (a b c d : ℝ)
variable (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d)
variable (h1 : a + b = c + d)
variable (h2 : a^2 + b^2 > c^2 + d^2)

theorem cube_sum_gt : a^3 + b^3 > c^3 + d^3 := by
  sorry

end NUMINAMATH_GPT_cube_sum_gt_l1536_153672


namespace NUMINAMATH_GPT_solve_for_q_l1536_153601

theorem solve_for_q
  (n m q : ℚ)
  (h1 : 5 / 6 = n / 60)
  (h2 : 5 / 6 = (m - n) / 66)
  (h3 : 5 / 6 = (q - m) / 150) :
  q = 230 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_q_l1536_153601


namespace NUMINAMATH_GPT_percentage_difference_between_chef_and_dishwasher_l1536_153682

theorem percentage_difference_between_chef_and_dishwasher
    (manager_wage : ℝ)
    (dishwasher_wage : ℝ)
    (chef_wage : ℝ)
    (h1 : manager_wage = 6.50)
    (h2 : dishwasher_wage = manager_wage / 2)
    (h3 : chef_wage = manager_wage - 2.60) :
    (chef_wage - dishwasher_wage) / dishwasher_wage * 100 = 20 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_percentage_difference_between_chef_and_dishwasher_l1536_153682


namespace NUMINAMATH_GPT_domain_of_g_l1536_153647

theorem domain_of_g : ∀ t : ℝ, (t - 3)^2 + (t + 3)^2 + 1 ≠ 0 :=
by
  intro t
  sorry

end NUMINAMATH_GPT_domain_of_g_l1536_153647


namespace NUMINAMATH_GPT_intersection_lines_l1536_153621

theorem intersection_lines (a b : ℝ) (h1 : ∀ x y : ℝ, (x = 3 ∧ y = 1) → x = 1/3 * y + a)
                          (h2 : ∀ x y : ℝ, (x = 3 ∧ y = 1) → y = 1/3 * x + b) :
  a + b = 8 / 3 :=
sorry

end NUMINAMATH_GPT_intersection_lines_l1536_153621


namespace NUMINAMATH_GPT_no_solution_system_l1536_153664

noncomputable def system_inconsistent : Prop :=
  ∀ x y : ℝ, ¬ (3 * x - 4 * y = 8 ∧ 6 * x - 8 * y = 12)

theorem no_solution_system : system_inconsistent :=
by
  sorry

end NUMINAMATH_GPT_no_solution_system_l1536_153664


namespace NUMINAMATH_GPT_problem_proof_l1536_153631

variable {α : Type*}
noncomputable def op (a b : ℝ) : ℝ := 1/a + 1/b
theorem problem_proof (a b : ℝ) (h : op a (-b) = 2) : (3 * a * b) / (2 * a - 2 * b) = -3/4 :=
by
  sorry

end NUMINAMATH_GPT_problem_proof_l1536_153631


namespace NUMINAMATH_GPT_joan_balloons_l1536_153622

def initial_balloons : ℕ := 72
def additional_balloons : ℕ := 23
def total_balloons : ℕ := initial_balloons + additional_balloons

theorem joan_balloons : total_balloons = 95 := by
  sorry

end NUMINAMATH_GPT_joan_balloons_l1536_153622


namespace NUMINAMATH_GPT_find_c_l1536_153658

noncomputable def P (c : ℝ) (x : ℝ) : ℝ := x^3 - 3 * x^2 + c * x - 8

theorem find_c (c : ℝ) : (∀ x, P c (x + 2) = 0) → c = -14 :=
sorry

end NUMINAMATH_GPT_find_c_l1536_153658


namespace NUMINAMATH_GPT_selena_ran_24_miles_l1536_153677

theorem selena_ran_24_miles (S J : ℝ) (h1 : S + J = 36) (h2 : J = S / 2) : S = 24 := 
sorry

end NUMINAMATH_GPT_selena_ran_24_miles_l1536_153677


namespace NUMINAMATH_GPT_remainder_correct_l1536_153626

noncomputable def P : Polynomial ℝ := Polynomial.C 1 * Polynomial.X^6 
                                  + Polynomial.C 2 * Polynomial.X^5 
                                  - Polynomial.C 3 * Polynomial.X^4 
                                  + Polynomial.C 1 * Polynomial.X^3 
                                  - Polynomial.C 2 * Polynomial.X^2
                                  + Polynomial.C 5 * Polynomial.X 
                                  - Polynomial.C 1

noncomputable def D : Polynomial ℝ := (Polynomial.X - Polynomial.C 1) * 
                                      (Polynomial.X + Polynomial.C 2) * 
                                      (Polynomial.X - Polynomial.C 3)

noncomputable def R : Polynomial ℝ := 17 * Polynomial.X^2 - 52 * Polynomial.X + 38

theorem remainder_correct :
    ∀ (q : Polynomial ℝ), P = D * q + R :=
by sorry

end NUMINAMATH_GPT_remainder_correct_l1536_153626


namespace NUMINAMATH_GPT_range_of_a_l1536_153678

noncomputable def is_monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a * x^2 - x - 1

theorem range_of_a {a : ℝ} : is_monotonic (f a) ↔ -Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1536_153678


namespace NUMINAMATH_GPT_effect_on_revenue_decrease_l1536_153661

variable (P Q : ℝ)

def original_revenue (P Q : ℝ) : ℝ := P * Q

def new_price (P : ℝ) : ℝ := P * 1.40

def new_quantity (Q : ℝ) : ℝ := Q * 0.65

def new_revenue (P Q : ℝ) : ℝ := new_price P * new_quantity Q

theorem effect_on_revenue_decrease :
  new_revenue P Q = original_revenue P Q * 0.91 →
  new_revenue P Q - original_revenue P Q = original_revenue P Q * -0.09 :=
by
  sorry

end NUMINAMATH_GPT_effect_on_revenue_decrease_l1536_153661


namespace NUMINAMATH_GPT_distinct_digit_sums_l1536_153616

theorem distinct_digit_sums (A B C E D : ℕ) (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ E ∧ A ≠ D ∧ B ≠ C ∧ B ≠ E ∧ B ≠ D ∧ C ≠ E ∧ C ≠ D ∧ E ≠ D)
 (h_ab : A + B = D) (h_ab_lt_10 : A + B < 10) (h_ce : C + E = D) :
  ∃ (x : ℕ), x = 8 := 
sorry

end NUMINAMATH_GPT_distinct_digit_sums_l1536_153616


namespace NUMINAMATH_GPT_geom_prog_235_l1536_153638

theorem geom_prog_235 (q : ℝ) (k n : ℕ) (hk : 1 < k) (hn : k < n) : 
  ¬ (q > 0 ∧ q ≠ 1 ∧ 3 = 2 * q^(k - 1) ∧ 5 = 2 * q^(n - 1)) := 
by 
  sorry

end NUMINAMATH_GPT_geom_prog_235_l1536_153638


namespace NUMINAMATH_GPT_find_number_l1536_153642

theorem find_number (x : ℝ) (h : (168 / 100) * x / 6 = 354.2) : x = 1265 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l1536_153642


namespace NUMINAMATH_GPT_solve_cubic_equation_l1536_153659

theorem solve_cubic_equation :
  ∀ x : ℝ, x^3 = 13 * x + 12 ↔ x = 4 ∨ x = -1 ∨ x = -3 :=
by
  sorry

end NUMINAMATH_GPT_solve_cubic_equation_l1536_153659


namespace NUMINAMATH_GPT_calculation_correct_l1536_153615

theorem calculation_correct : -2 + 3 = 1 :=
by
  sorry

end NUMINAMATH_GPT_calculation_correct_l1536_153615


namespace NUMINAMATH_GPT_either_d_or_2d_is_perfect_square_l1536_153681

theorem either_d_or_2d_is_perfect_square
  (a c d : ℕ) (hrel_prime : Nat.gcd a c = 1) (hd : ∃ D : ℝ, D = d ∧ (D:ℝ) > 0)
  (hdiam : d^2 = 2 * a^2 + c^2) :
  ∃ m : ℕ, m^2 = d ∨ m^2 = 2 * d :=
by
  sorry

end NUMINAMATH_GPT_either_d_or_2d_is_perfect_square_l1536_153681


namespace NUMINAMATH_GPT_percent_area_square_in_rectangle_l1536_153611

theorem percent_area_square_in_rectangle 
  (s : ℝ) (rect_width : ℝ) (rect_length : ℝ) (h1 : rect_width = 2 * s) (h2 : rect_length = 2 * rect_width) : 
  (s^2 / (rect_length * rect_width)) * 100 = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_percent_area_square_in_rectangle_l1536_153611


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l1536_153654

noncomputable def find_prices (price_peanuts price_tea : ℝ) : Prop :=
price_peanuts + 40 = price_tea ∧
50 * price_peanuts = 10 * price_tea

theorem part1_solution :
  ∃ (price_peanuts price_tea : ℝ), find_prices price_peanuts price_tea :=
by
  sorry

def cost_function (m : ℝ) : ℝ :=
6 * m + 36 * (60 - m)

def profit_function (m : ℝ) : ℝ :=
(10 - 6) * m + (50 - 36) * (60 - m)

noncomputable def max_profit := 540

theorem part2_solution :
  ∃ (m t : ℝ), 30 ≤ m ∧ m ≤ 40 ∧ cost_function m ≤ 1260 ∧ profit_function m = max_profit :=
by
  sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_l1536_153654


namespace NUMINAMATH_GPT_square_division_l1536_153683

theorem square_division (n k : ℕ) (m : ℕ) (h : n * k = m * m) :
  ∃ u v d : ℕ, (gcd u v = 1) ∧ (n = d * u * u) ∧ (k = d * v * v) ∧ (m = d * u * v) :=
by sorry

end NUMINAMATH_GPT_square_division_l1536_153683


namespace NUMINAMATH_GPT_minimum_value_of_xy_l1536_153620

theorem minimum_value_of_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = x * y) :
  x * y ≥ 8 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_xy_l1536_153620


namespace NUMINAMATH_GPT_distance_between_foci_l1536_153646

-- Define the ellipse
def ellipse_eq (x y : ℝ) := 9 * x^2 + 36 * y^2 = 1296

-- Define the semi-major and semi-minor axes
def semi_major_axis := 12
def semi_minor_axis := 6

-- Distance between the foci of the ellipse
theorem distance_between_foci : 
  (∃ x y : ℝ, ellipse_eq x y) → 2 * Real.sqrt (semi_major_axis^2 - semi_minor_axis^2) = 12 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_foci_l1536_153646


namespace NUMINAMATH_GPT_first_reduction_percentage_l1536_153663

theorem first_reduction_percentage (P : ℝ) (x : ℝ) :
  P * (1 - x / 100) * 0.6 = P * 0.45 → x = 25 :=
by
  sorry

end NUMINAMATH_GPT_first_reduction_percentage_l1536_153663


namespace NUMINAMATH_GPT_condition1_condition2_condition3_l1536_153692

-- Condition 1 statement
theorem condition1: (number_of_ways_condition1 : ℕ) = 5520 := by
  -- Expected proof that number_of_ways_condition1 = 5520
  sorry

-- Condition 2 statement
theorem condition2: (number_of_ways_condition2 : ℕ) = 3360 := by
  -- Expected proof that number_of_ways_condition2 = 3360
  sorry

-- Condition 3 statement
theorem condition3: (number_of_ways_condition3 : ℕ) = 360 := by
  -- Expected proof that number_of_ways_condition3 = 360
  sorry

end NUMINAMATH_GPT_condition1_condition2_condition3_l1536_153692


namespace NUMINAMATH_GPT_intersection_A_B_l1536_153698

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := { x | 2 ≤ x ∧ x ≤ 5 }

theorem intersection_A_B : A ∩ B = {3, 5} :=
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1536_153698


namespace NUMINAMATH_GPT_typist_original_salary_l1536_153649

theorem typist_original_salary (S : ℝ) :
  (1.10 * S * 0.95 * 1.07 * 0.97 = 2090) → (S = 2090 / (1.10 * 0.95 * 1.07 * 0.97)) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_typist_original_salary_l1536_153649


namespace NUMINAMATH_GPT_cars_per_client_l1536_153629

-- Define the conditions
def num_cars : ℕ := 18
def selections_per_car : ℕ := 3
def num_clients : ℕ := 18

-- Define the proof problem as a theorem
theorem cars_per_client :
  (num_cars * selections_per_car) / num_clients = 3 :=
sorry

end NUMINAMATH_GPT_cars_per_client_l1536_153629


namespace NUMINAMATH_GPT_fishing_problem_l1536_153618

theorem fishing_problem :
  ∃ (x y : ℕ), 
    (x + y = 70) ∧ 
    (∃ k : ℕ, x = 9 * k) ∧ 
    (∃ m : ℕ, y = 17 * m) ∧ 
    x = 36 ∧ 
    y = 34 := 
by
  sorry

end NUMINAMATH_GPT_fishing_problem_l1536_153618


namespace NUMINAMATH_GPT_relationship_among_neg_a_neg_a3_a2_l1536_153660

theorem relationship_among_neg_a_neg_a3_a2 (a : ℝ) (h : a^2 + a < 0) : -a > a^2 ∧ a^2 > -a^3 :=
by sorry

end NUMINAMATH_GPT_relationship_among_neg_a_neg_a3_a2_l1536_153660


namespace NUMINAMATH_GPT_allocation_methods_count_l1536_153635

def number_of_allocation_methods (doctors nurses : ℕ) (hospitals : ℕ) (nurseA nurseB : ℕ) :=
  if (doctors = 3) ∧ (nurses = 6) ∧ (hospitals = 3) ∧ (nurseA = 1) ∧ (nurseB = 1) then 684 else 0

theorem allocation_methods_count :
  number_of_allocation_methods 3 6 3 2 2 = 684 :=
by
  sorry

end NUMINAMATH_GPT_allocation_methods_count_l1536_153635


namespace NUMINAMATH_GPT_smallest_possible_sum_l1536_153689

theorem smallest_possible_sum (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : Nat.gcd (a + b) 330 = 1) (h4 : b ^ b ∣ a ^ a) (h5 : ¬ b ∣ a) :
  a + b = 147 :=
sorry

end NUMINAMATH_GPT_smallest_possible_sum_l1536_153689


namespace NUMINAMATH_GPT_positive_integer_solutions_l1536_153613

theorem positive_integer_solutions:
  ∀ (x y : ℕ), (5 * x + y = 11) → (x > 0) → (y > 0) → (x = 1 ∧ y = 6) ∨ (x = 2 ∧ y = 1) :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_solutions_l1536_153613


namespace NUMINAMATH_GPT_work_completion_time_l1536_153666

-- Define the rate of work done by a, b, and c.
def rate_a := 1 / 4
def rate_b := 1 / 12
def rate_c := 1 / 6

-- Define the time each person starts working and the cycle pattern.
def start_time : ℕ := 6 -- in hours
def cycle_pattern := [rate_a, rate_b, rate_c]

-- Calculate the total amount of work done in one cycle of 3 hours.
def work_per_cycle := (rate_a + rate_b + rate_c)

-- Calculate the total time to complete the work.
def total_time_to_complete_work := 2 * 3 -- number of cycles times 3 hours per cycle

-- Calculate the time of completion.
def completion_time := start_time + total_time_to_complete_work

-- Theorem to prove the work completion time.
theorem work_completion_time : completion_time = 12 := 
by
  -- Proof can be filled in here
  sorry

end NUMINAMATH_GPT_work_completion_time_l1536_153666


namespace NUMINAMATH_GPT_part_a_part_b_l1536_153675

theorem part_a (m : ℕ) : m = 1 ∨ m = 2 ∨ m = 4 → (3^m - 1) % (2^m) = 0 := by
  sorry

theorem part_b (m : ℕ) : m = 1 ∨ m = 2 ∨ m = 4 ∨ m = 6 ∨ m = 8 → (31^m - 1) % (2^m) = 0 := by
  sorry

end NUMINAMATH_GPT_part_a_part_b_l1536_153675


namespace NUMINAMATH_GPT_prime_sum_divisible_l1536_153637

theorem prime_sum_divisible (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : q = p + 2) :
  (p ^ q + q ^ p) % (p + q) = 0 :=
by
  sorry

end NUMINAMATH_GPT_prime_sum_divisible_l1536_153637


namespace NUMINAMATH_GPT_one_less_than_neg_one_is_neg_two_l1536_153639

theorem one_less_than_neg_one_is_neg_two : (-1 - 1 = -2) :=
by
  sorry

end NUMINAMATH_GPT_one_less_than_neg_one_is_neg_two_l1536_153639


namespace NUMINAMATH_GPT_max_largest_int_of_avg_and_diff_l1536_153680

theorem max_largest_int_of_avg_and_diff (A B C D E : ℕ) (h1 : A ≤ B) (h2 : B ≤ C) (h3 : C ≤ D) (h4 : D ≤ E) 
  (h_avg : (A + B + C + D + E) / 5 = 70) (h_diff : E - A = 10) : E = 340 :=
by
  sorry

end NUMINAMATH_GPT_max_largest_int_of_avg_and_diff_l1536_153680


namespace NUMINAMATH_GPT_Zachary_sold_40_games_l1536_153604

theorem Zachary_sold_40_games 
  (R J Z : ℝ)
  (games_Zachary_sold : ℕ)
  (h1 : R = J + 50)
  (h2 : J = 1.30 * Z)
  (h3 : Z = 5 * games_Zachary_sold)
  (h4 : Z + J + R = 770) :
  games_Zachary_sold = 40 :=
by
  sorry

end NUMINAMATH_GPT_Zachary_sold_40_games_l1536_153604


namespace NUMINAMATH_GPT_slope_of_given_line_l1536_153668

theorem slope_of_given_line : ∀ (x y : ℝ), (4 / x + 5 / y = 0) → (y = (-5 / 4) * x) := 
by 
  intros x y h
  sorry

end NUMINAMATH_GPT_slope_of_given_line_l1536_153668


namespace NUMINAMATH_GPT_length_of_train_is_correct_l1536_153693

noncomputable def speed_in_m_per_s (speed_in_km_per_hr : ℝ) : ℝ := speed_in_km_per_hr * 1000 / 3600

noncomputable def total_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

noncomputable def length_of_train (total_distance : ℝ) (length_of_bridge : ℝ) : ℝ := total_distance - length_of_bridge

theorem length_of_train_is_correct :
  ∀ (speed_in_km_per_hr : ℝ) (time_to_cross_bridge : ℝ) (length_of_bridge : ℝ),
  speed_in_km_per_hr = 72 →
  time_to_cross_bridge = 12.199024078073753 →
  length_of_bridge = 134 →
  length_of_train (total_distance (speed_in_m_per_s speed_in_km_per_hr) time_to_cross_bridge) length_of_bridge = 110.98048156147506 :=
by 
  intros speed_in_km_per_hr time_to_cross_bridge length_of_bridge hs ht hl;
  rw [hs, ht, hl];
  sorry

end NUMINAMATH_GPT_length_of_train_is_correct_l1536_153693


namespace NUMINAMATH_GPT_square_pattern_1111111_l1536_153617

theorem square_pattern_1111111 :
  11^2 = 121 ∧ 111^2 = 12321 ∧ 1111^2 = 1234321 → 1111111^2 = 1234567654321 :=
by
  sorry

end NUMINAMATH_GPT_square_pattern_1111111_l1536_153617


namespace NUMINAMATH_GPT_sum_of_six_consecutive_integers_l1536_153648

theorem sum_of_six_consecutive_integers (m : ℤ) : 
  (m + (m + 1) + (m + 2) + (m + 3) + (m + 4) + (m + 5)) = 6 * m + 15 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_six_consecutive_integers_l1536_153648


namespace NUMINAMATH_GPT_protein_percentage_in_mixture_l1536_153645

theorem protein_percentage_in_mixture :
  let soybean_meal_weight := 240
  let cornmeal_weight := 40
  let mixture_weight := 280
  let soybean_protein_content := 0.14
  let cornmeal_protein_content := 0.07
  let total_protein := soybean_meal_weight * soybean_protein_content + cornmeal_weight * cornmeal_protein_content
  let protein_percentage := (total_protein / mixture_weight) * 100
  protein_percentage = 13 :=
by
  sorry

end NUMINAMATH_GPT_protein_percentage_in_mixture_l1536_153645


namespace NUMINAMATH_GPT_tournament_committee_count_l1536_153609

theorem tournament_committee_count :
  let teams := 6
  let members_per_team := 8
  let host_team_choices := Nat.choose 8 3
  let regular_non_host_choices := Nat.choose 8 2
  let special_non_host_choices := Nat.choose 8 3
  let total_regular_non_host_choices := regular_non_host_choices ^ 4 
  let combined_choices_non_host := total_regular_non_host_choices * special_non_host_choices
  let combined_choices_host_non_host := combined_choices_non_host * host_team_choices
  let total_choices := combined_choices_host_non_host * teams
  total_choices = 11568055296 := 
by {
  let teams := 6
  let members_per_team := 8
  let host_team_choices := Nat.choose 8 3
  let regular_non_host_choices := Nat.choose 8 2
  let special_non_host_choices := Nat.choose 8 3
  let total_regular_non_host_choices := regular_non_host_choices ^ 4 
  let combined_choices_non_host := total_regular_non_host_choices * special_non_host_choices
  let combined_choices_host_non_host := combined_choices_non_host * host_team_choices
  let total_choices := combined_choices_host_non_host * teams
  have h_total_choices_eq : total_choices = 11568055296 := sorry
  exact h_total_choices_eq
}

end NUMINAMATH_GPT_tournament_committee_count_l1536_153609


namespace NUMINAMATH_GPT_find_red_chairs_l1536_153690

noncomputable def red_chairs := Nat
noncomputable def yellow_chairs := Nat
noncomputable def blue_chairs := Nat

theorem find_red_chairs
    (R Y B : Nat)
    (h1 : Y = 2 * R)
    (h2 : B = Y - 2)
    (h3 : R + Y + B = 18) :
    R = 4 := by
  sorry

end NUMINAMATH_GPT_find_red_chairs_l1536_153690


namespace NUMINAMATH_GPT_inequality_proof_l1536_153684

variables {a b : ℝ}

theorem inequality_proof :
  a^2 + b^2 - 1 - a^2 * b^2 <= 0 ↔ (a^2 - 1) * (b^2 - 1) >= 0 :=
by sorry

end NUMINAMATH_GPT_inequality_proof_l1536_153684


namespace NUMINAMATH_GPT_slope_of_line_l1536_153669

-- Define the point and the line equation with a generic slope
def point : ℝ × ℝ := (-1, 2)

def line (a : ℝ) := a * (point.fst) + (point.snd) - 4 = 0

-- The main theorem statement
theorem slope_of_line (a : ℝ) (h : line a) : ∃ m : ℝ, m = 2 :=
by
  -- The slope of the line derived from the equation and condition
  sorry

end NUMINAMATH_GPT_slope_of_line_l1536_153669


namespace NUMINAMATH_GPT_numberOfBoys_playground_boys_count_l1536_153679

-- Definitions and conditions
def numberOfGirls : ℕ := 28
def totalNumberOfChildren : ℕ := 63

-- Theorem statement
theorem numberOfBoys (numberOfGirls : ℕ) (totalNumberOfChildren : ℕ) : ℕ :=
  totalNumberOfChildren - numberOfGirls

-- Proof statement
theorem playground_boys_count (numberOfGirls : ℕ) (totalNumberOfChildren : ℕ) (boysOnPlayground : ℕ) : 
  numberOfGirls = 28 → 
  totalNumberOfChildren = 63 → 
  boysOnPlayground = totalNumberOfChildren - numberOfGirls →
  boysOnPlayground = 35 :=
by
  intros
  -- since no proof is required, we use sorry here
  exact sorry

end NUMINAMATH_GPT_numberOfBoys_playground_boys_count_l1536_153679


namespace NUMINAMATH_GPT_sqrt_3_between_neg_1_and_2_l1536_153686

theorem sqrt_3_between_neg_1_and_2 : -1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 := by
  sorry

end NUMINAMATH_GPT_sqrt_3_between_neg_1_and_2_l1536_153686


namespace NUMINAMATH_GPT_kelly_raisins_l1536_153697

theorem kelly_raisins (weight_peanuts : ℝ) (total_weight_snacks : ℝ) (h1 : weight_peanuts = 0.1) (h2 : total_weight_snacks = 0.5) : total_weight_snacks - weight_peanuts = 0.4 := by
  sorry

end NUMINAMATH_GPT_kelly_raisins_l1536_153697
