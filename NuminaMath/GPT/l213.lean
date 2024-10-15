import Mathlib

namespace NUMINAMATH_GPT_all_plants_diseased_l213_21351

theorem all_plants_diseased (n : ℕ) (h : n = 1007) : 
  n * 2 = 2014 := by
  sorry

end NUMINAMATH_GPT_all_plants_diseased_l213_21351


namespace NUMINAMATH_GPT_perfect_squares_difference_l213_21392

theorem perfect_squares_difference : 
  let N : ℕ := 20000;
  let diff_squared (b : ℤ) : ℤ := (b+2)^2 - b^2;
  ∃ k : ℕ, (1 ≤ k ∧ k ≤ 70) ∧ (∀ m : ℕ, (m < N) → (∃ b : ℤ, m = diff_squared b) → m = (2 * k)^2)
:= sorry

end NUMINAMATH_GPT_perfect_squares_difference_l213_21392


namespace NUMINAMATH_GPT_number_of_boys_at_reunion_l213_21356

theorem number_of_boys_at_reunion (n : ℕ) (h : n * (n - 1) / 2 = 66) : n = 12 :=
sorry

end NUMINAMATH_GPT_number_of_boys_at_reunion_l213_21356


namespace NUMINAMATH_GPT_base_8_to_10_conversion_l213_21347

theorem base_8_to_10_conversion : (2 * 8^4 + 3 * 8^3 + 4 * 8^2 + 5 * 8^1 + 6 * 8^0) = 10030 := by 
  -- specify the summation directly 
  sorry

end NUMINAMATH_GPT_base_8_to_10_conversion_l213_21347


namespace NUMINAMATH_GPT_square_flag_side_length_side_length_of_square_flags_is_4_l213_21391

theorem square_flag_side_length 
  (total_fabric : ℕ)
  (fabric_left : ℕ)
  (num_square_flags : ℕ)
  (num_wide_flags : ℕ)
  (num_tall_flags : ℕ)
  (wide_flag_length : ℕ)
  (wide_flag_width : ℕ)
  (tall_flag_length : ℕ)
  (tall_flag_width : ℕ)
  (fabric_used_on_wide_and_tall_flags : ℕ)
  (fabric_used_on_all_flags : ℕ)
  (fabric_used_on_square_flags : ℕ)
  (square_flag_area : ℕ)
  (side_length : ℕ) : Prop :=
  total_fabric = 1000 ∧
  fabric_left = 294 ∧
  num_square_flags = 16 ∧
  num_wide_flags = 20 ∧
  num_tall_flags = 10 ∧
  wide_flag_length = 5 ∧
  wide_flag_width = 3 ∧
  tall_flag_length = 5 ∧
  tall_flag_width = 3 ∧
  fabric_used_on_wide_and_tall_flags = (num_wide_flags + num_tall_flags) * (wide_flag_length * wide_flag_width) ∧
  fabric_used_on_all_flags = total_fabric - fabric_left ∧
  fabric_used_on_square_flags = fabric_used_on_all_flags - fabric_used_on_wide_and_tall_flags ∧
  square_flag_area = fabric_used_on_square_flags / num_square_flags ∧
  side_length = Int.sqrt square_flag_area ∧
  side_length = 4

theorem side_length_of_square_flags_is_4 : 
  square_flag_side_length 1000 294 16 20 10 5 3 5 3 450 706 256 16 4 :=
  by
    sorry

end NUMINAMATH_GPT_square_flag_side_length_side_length_of_square_flags_is_4_l213_21391


namespace NUMINAMATH_GPT_whales_last_year_eq_4000_l213_21345

variable (W : ℕ) (last_year this_year next_year : ℕ)

theorem whales_last_year_eq_4000
    (h1 : this_year = 2 * last_year)
    (h2 : next_year = this_year + 800)
    (h3 : next_year = 8800) :
    last_year = 4000 := by
  sorry

end NUMINAMATH_GPT_whales_last_year_eq_4000_l213_21345


namespace NUMINAMATH_GPT_find_fraction_l213_21382

variable (n : ℚ) (x : ℚ)

theorem find_fraction (h1 : n = 0.5833333333333333) (h2 : n = 1/3 + x) : x = 0.25 := by
  sorry

end NUMINAMATH_GPT_find_fraction_l213_21382


namespace NUMINAMATH_GPT_interest_rate_l213_21319

theorem interest_rate (P : ℝ) (r : ℝ) (t : ℝ) (CI SI : ℝ) (diff : ℝ) 
    (hP : P = 1500)
    (ht : t = 2)
    (hdiff : diff = 15)
    (hCI : CI = P * (1 + r / 100)^t - P)
    (hSI : SI = P * r * t / 100)
    (hCI_SI_diff : CI - SI = diff) :
    r = 1 := 
by
  sorry -- proof goes here


end NUMINAMATH_GPT_interest_rate_l213_21319


namespace NUMINAMATH_GPT_lamp_probability_l213_21327

theorem lamp_probability (rope_length : ℝ) (pole_distance : ℝ) (h_pole_distance : pole_distance = 8) :
  let lamp_range := 2
  let favorable_segment_length := 4
  let total_rope_length := rope_length
  let probability := (favorable_segment_length / total_rope_length)
  rope_length = 8 → probability = 1 / 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_lamp_probability_l213_21327


namespace NUMINAMATH_GPT_unique_solution_l213_21374

theorem unique_solution (x : ℝ) (h : (1 / (x - 1)) = (3 / (2 * x - 3))) : x = 0 := 
sorry

end NUMINAMATH_GPT_unique_solution_l213_21374


namespace NUMINAMATH_GPT_water_usage_correct_l213_21302

variable (y : ℝ) (C₁ : ℝ) (C₂ : ℝ) (x : ℝ)

noncomputable def water_bill : ℝ :=
  if x ≤ 4 then C₁ * x else 4 * C₁ + C₂ * (x - 4)

theorem water_usage_correct (h1 : y = 12.8) (h2 : C₁ = 1.2) (h3 : C₂ = 1.6) : x = 9 :=
by
  have h4 : x > 4 := sorry
  sorry

end NUMINAMATH_GPT_water_usage_correct_l213_21302


namespace NUMINAMATH_GPT_yvettes_final_bill_l213_21328

theorem yvettes_final_bill :
  let alicia : ℝ := 7.5
  let brant : ℝ := 10
  let josh : ℝ := 8.5
  let yvette : ℝ := 9
  let tip_percentage : ℝ := 0.2
  ∃ final_bill : ℝ, final_bill = (alicia + brant + josh + yvette) * (1 + tip_percentage) ∧ final_bill = 42 :=
by
  sorry

end NUMINAMATH_GPT_yvettes_final_bill_l213_21328


namespace NUMINAMATH_GPT_arithmetic_and_geometric_sequence_statement_l213_21388

-- Arithmetic sequence definitions
def arithmetic_seq (a b d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Conditions
def a_2 : ℕ := 9
def a_5 : ℕ := 21

-- General formula and solution for part (Ⅰ)
def general_formula_arithmetic_sequence : Prop :=
  ∃ (a d : ℕ), (a + d = a_2 ∧ a + 4 * d = a_5) ∧ ∀ n : ℕ, arithmetic_seq a d n = 4 * n + 1

-- Definitions and conditions for geometric sequence derived from arithmetic sequence
def b_n (n : ℕ) : ℕ := 2 ^ (4 * n + 1)

-- Sum of the first n terms of the sequence {b_n}
def S_n (n : ℕ) : ℕ := (32 * (2 ^ (4 * n) - 1)) / 15

-- Statement that needs to be proven
theorem arithmetic_and_geometric_sequence_statement :
  general_formula_arithmetic_sequence ∧ (∀ n, S_n n = (32 * (2 ^ (4 * n) - 1)) / 15) := by
  sorry

end NUMINAMATH_GPT_arithmetic_and_geometric_sequence_statement_l213_21388


namespace NUMINAMATH_GPT_abs_inequality_solution_l213_21364

theorem abs_inequality_solution (x : ℝ) :
  |2 * x - 2| + |2 * x + 4| < 10 ↔ x ∈ Set.Ioo (-4 : ℝ) (2 : ℝ) := 
by sorry

end NUMINAMATH_GPT_abs_inequality_solution_l213_21364


namespace NUMINAMATH_GPT_quadratic_roots_ratio_l213_21343

theorem quadratic_roots_ratio (p x1 x2 : ℝ) (h_eq : x1^2 + p * x1 - 16 = 0) (h_ratio : x1 / x2 = -4) :
  p = 6 ∨ p = -6 :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_roots_ratio_l213_21343


namespace NUMINAMATH_GPT_greatest_two_digit_number_with_digit_product_16_l213_21398

def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def digit_product (n m : ℕ) : Prop :=
  n * m = 16

def from_digits (n m : ℕ) : ℕ :=
  10 * n + m

theorem greatest_two_digit_number_with_digit_product_16 :
  ∀ n m, is_two_digit_number (from_digits n m) → digit_product n m → (82 ≥ from_digits n m) :=
by
  intros n m h1 h2
  sorry

end NUMINAMATH_GPT_greatest_two_digit_number_with_digit_product_16_l213_21398


namespace NUMINAMATH_GPT_tan_double_angle_third_quadrant_l213_21361

open Real

theorem tan_double_angle_third_quadrant (α : ℝ) (h1 : π < α ∧ α < 3 * π / 2) 
  (h2 : sin (π - α) = - (3 / 5)) : 
  tan (2 * α) = 24 / 7 :=
by
  sorry

end NUMINAMATH_GPT_tan_double_angle_third_quadrant_l213_21361


namespace NUMINAMATH_GPT_find_number_l213_21379

theorem find_number (x : ℝ) (h : x / 3 = 1.005 * 400) : x = 1206 := 
by 
sorry

end NUMINAMATH_GPT_find_number_l213_21379


namespace NUMINAMATH_GPT_fruit_seller_price_l213_21323

theorem fruit_seller_price 
  (CP SP SP_profit : ℝ)
  (h1 : SP = CP * 0.88)
  (h2 : SP_profit = CP * 1.20)
  (h3 : SP_profit = 21.818181818181817) :
  SP = 16 := 
by 
  sorry

end NUMINAMATH_GPT_fruit_seller_price_l213_21323


namespace NUMINAMATH_GPT_remainder_of_power_mod_five_l213_21352

theorem remainder_of_power_mod_five : (4 ^ 11) % 5 = 4 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_power_mod_five_l213_21352


namespace NUMINAMATH_GPT_connie_needs_more_money_l213_21369

variable (cost_connie : ℕ) (cost_watch : ℕ)

theorem connie_needs_more_money 
  (h_connie : cost_connie = 39)
  (h_watch : cost_watch = 55) :
  cost_watch - cost_connie = 16 :=
by sorry

end NUMINAMATH_GPT_connie_needs_more_money_l213_21369


namespace NUMINAMATH_GPT_time_to_pass_tree_l213_21300

-- Define the conditions given in the problem
def train_length : ℕ := 1200
def platform_length : ℕ := 700
def time_to_pass_platform : ℕ := 190

-- Calculate the total distance covered while passing the platform
def distance_passed_platform : ℕ := train_length + platform_length

-- The main theorem we need to prove
theorem time_to_pass_tree : (distance_passed_platform / time_to_pass_platform) * train_length = 120 := 
by
  sorry

end NUMINAMATH_GPT_time_to_pass_tree_l213_21300


namespace NUMINAMATH_GPT_eval_expression_l213_21370

theorem eval_expression : ⌈- (7 / 3 : ℚ)⌉ + ⌊(7 / 3 : ℚ)⌋ = 0 := 
by 
  sorry

end NUMINAMATH_GPT_eval_expression_l213_21370


namespace NUMINAMATH_GPT_equation_of_parallel_line_l213_21305

theorem equation_of_parallel_line (x y : ℝ) :
  (∀ b : ℝ, 2 * x + y + b = 0 → x = -1 → y = 2 → b = 0) →
  (∀ x y b: ℝ, 2 * x + y + b = 0 → x = -1 → y = 2 → 2 * x + y = 0) :=
by
  sorry

end NUMINAMATH_GPT_equation_of_parallel_line_l213_21305


namespace NUMINAMATH_GPT_tina_made_more_140_dollars_l213_21304

def candy_bars_cost : ℕ := 2
def marvin_candy_bars : ℕ := 35
def tina_candy_bars : ℕ := 3 * marvin_candy_bars
def marvin_money : ℕ := marvin_candy_bars * candy_bars_cost
def tina_money : ℕ := tina_candy_bars * candy_bars_cost
def tina_extra_money : ℕ := tina_money - marvin_money

theorem tina_made_more_140_dollars :
  tina_extra_money = 140 := by
  sorry

end NUMINAMATH_GPT_tina_made_more_140_dollars_l213_21304


namespace NUMINAMATH_GPT_James_future_age_when_Thomas_reaches_James_current_age_l213_21344

-- Defining the given conditions
def Thomas_age := 6
def Shay_age := Thomas_age + 13
def James_age := Shay_age + 5

-- Goal: Proving James's age when Thomas reaches James's current age
theorem James_future_age_when_Thomas_reaches_James_current_age :
  let years_until_Thomas_is_James_current_age := James_age - Thomas_age
  let James_future_age := James_age + years_until_Thomas_is_James_current_age
  James_future_age = 42 :=
by
  sorry

end NUMINAMATH_GPT_James_future_age_when_Thomas_reaches_James_current_age_l213_21344


namespace NUMINAMATH_GPT_candidate_fails_by_50_marks_l213_21320

theorem candidate_fails_by_50_marks (T : ℝ) (pass_mark : ℝ) (h1 : pass_mark = 199.99999999999997)
    (h2 : 0.45 * T - 25 = 199.99999999999997) :
    199.99999999999997 - 0.30 * T = 50 :=
by
  sorry

end NUMINAMATH_GPT_candidate_fails_by_50_marks_l213_21320


namespace NUMINAMATH_GPT_range_of_a_l213_21338

-- Definitions of the sets U and A
def U := {x : ℝ | 0 < x ∧ x < 9}
def A (a : ℝ) := {x : ℝ | 1 < x ∧ x < a}

-- Theorem stating the range of a
theorem range_of_a (a : ℝ) (H_non_empty : A a ≠ ∅) (H_not_subset : ¬ ∀ x, x ∈ A a → x ∈ U) : 
  1 < a ∧ a ≤ 9 :=
sorry

end NUMINAMATH_GPT_range_of_a_l213_21338


namespace NUMINAMATH_GPT_max_value_of_quadratic_l213_21335

theorem max_value_of_quadratic :
  ∃ x_max : ℝ, x_max = 1.5 ∧
  ∀ x : ℝ, -3 * x^2 + 9 * x + 24 ≤ -3 * (1.5)^2 + 9 * 1.5 + 24 := by
  sorry

end NUMINAMATH_GPT_max_value_of_quadratic_l213_21335


namespace NUMINAMATH_GPT_cd_leq_one_l213_21387

variables {a b c d : ℝ}

theorem cd_leq_one (h1 : a * b = 1) (h2 : a * c + b * d = 2) : c * d ≤ 1 := 
sorry

end NUMINAMATH_GPT_cd_leq_one_l213_21387


namespace NUMINAMATH_GPT_constant_term_in_first_equation_l213_21332

/-- Given the system of equations:
  1. 5x + y = C
  2. x + 3y = 1
  3. 3x + 2y = 10
  Prove that the constant term C is 19.
-/
theorem constant_term_in_first_equation
  (x y C : ℝ)
  (h1 : 5 * x + y = C)
  (h2 : x + 3 * y = 1)
  (h3 : 3 * x + 2 * y = 10) :
  C = 19 :=
by
  sorry

end NUMINAMATH_GPT_constant_term_in_first_equation_l213_21332


namespace NUMINAMATH_GPT_distinct_arrangements_ballon_l213_21373

theorem distinct_arrangements_ballon : 
  let n := 6
  let repetitions := 2
  n! / repetitions! = 360 :=
by
  sorry

end NUMINAMATH_GPT_distinct_arrangements_ballon_l213_21373


namespace NUMINAMATH_GPT_pure_imaginary_number_solution_l213_21367

-- Definition of the problem
theorem pure_imaginary_number_solution (a : ℝ) (h1 : a^2 - 4 = 0) (h2 : a^2 - 3 * a + 2 ≠ 0) : a = -2 :=
sorry

end NUMINAMATH_GPT_pure_imaginary_number_solution_l213_21367


namespace NUMINAMATH_GPT_smallest_palindromic_odd_integer_in_base2_and_4_l213_21372

def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  let digits := n.digits base
  digits = digits.reverse

theorem smallest_palindromic_odd_integer_in_base2_and_4 :
  ∃ n : ℕ, n > 10 ∧ is_palindrome n 2 ∧ is_palindrome n 4 ∧ Odd n ∧ ∀ m : ℕ, (m > 10 ∧ is_palindrome m 2 ∧ is_palindrome m 4 ∧ Odd m) → n <= m :=
  sorry

end NUMINAMATH_GPT_smallest_palindromic_odd_integer_in_base2_and_4_l213_21372


namespace NUMINAMATH_GPT_value_of_other_bills_l213_21366

theorem value_of_other_bills (x : ℕ) : 
  (∃ (num_twenty num_x : ℕ), num_twenty = 3 ∧
                           num_x = 2 * num_twenty ∧
                           20 * num_twenty + x * num_x = 120) → 
  x * 6 = 60 :=
by
  intro h
  obtain ⟨num_twenty, num_x, h1, h2, h3⟩ := h
  have : num_twenty = 3 := h1
  have : num_x = 2 * num_twenty := h2
  have : x * 6 = 60 := sorry
  exact this

end NUMINAMATH_GPT_value_of_other_bills_l213_21366


namespace NUMINAMATH_GPT_total_length_of_segments_l213_21378

theorem total_length_of_segments
  (l1 l2 l3 l4 l5 l6 : ℕ) 
  (hl1 : l1 = 5) 
  (hl2 : l2 = 1) 
  (hl3 : l3 = 4) 
  (hl4 : l4 = 2) 
  (hl5 : l5 = 3) 
  (hl6 : l6 = 3) : 
  l1 + l2 + l3 + l4 + l5 + l6 = 18 := 
by 
  sorry

end NUMINAMATH_GPT_total_length_of_segments_l213_21378


namespace NUMINAMATH_GPT_solution_correct_l213_21318

-- Define the conditions
def abs_inequality (x : ℝ) : Prop := abs (x - 3) + abs (x + 4) < 8
def quadratic_eq (x : ℝ) : Prop := x^2 - x - 12 = 0

-- Define the main statement to prove
theorem solution_correct : ∃ (x : ℝ), abs_inequality x ∧ quadratic_eq x ∧ x = -3 := sorry

end NUMINAMATH_GPT_solution_correct_l213_21318


namespace NUMINAMATH_GPT_work_done_by_first_group_l213_21310

theorem work_done_by_first_group :
  (6 * 8 * 5 : ℝ) / W = (4 * 3 * 8 : ℝ) / 30 →
  W = 75 :=
by
  sorry

end NUMINAMATH_GPT_work_done_by_first_group_l213_21310


namespace NUMINAMATH_GPT_regular_polygon_sides_l213_21322

theorem regular_polygon_sides (n : ℕ) (h : ∀ n, (n > 2) → (360 / n = 20)) : n = 18 := sorry

end NUMINAMATH_GPT_regular_polygon_sides_l213_21322


namespace NUMINAMATH_GPT_math_problem_l213_21358

def calc_expr : Int := 
  54322 * 32123 - 54321 * 32123 + 54322 * 99000 - 54321 * 99001

theorem math_problem :
  calc_expr = 76802 := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l213_21358


namespace NUMINAMATH_GPT_money_left_l213_21368

theorem money_left (olivia_money nigel_money ticket_cost tickets_purchased : ℕ) 
  (h1 : olivia_money = 112) 
  (h2 : nigel_money = 139) 
  (h3 : ticket_cost = 28) 
  (h4 : tickets_purchased = 6) : 
  olivia_money + nigel_money - tickets_purchased * ticket_cost = 83 := 
by 
  sorry

end NUMINAMATH_GPT_money_left_l213_21368


namespace NUMINAMATH_GPT_exists_root_in_interval_l213_21317

-- Define the quadratic equation
def quadratic (a b c x : ℝ) : ℝ := a*x^2 + b*x + c

-- Conditions given in the problem
variables {a b c : ℝ}
variable  (h_a_nonzero : a ≠ 0)
variable  (h_neg_value : quadratic a b c 3.24 = -0.02)
variable  (h_pos_value : quadratic a b c 3.25 = 0.01)

-- Problem statement to be proved
theorem exists_root_in_interval : ∃ x : ℝ, 3.24 < x ∧ x < 3.25 ∧ quadratic a b c x = 0 :=
sorry

end NUMINAMATH_GPT_exists_root_in_interval_l213_21317


namespace NUMINAMATH_GPT_ratio_of_dolls_l213_21315

-- Definitions used in Lean 4 statement directly appear in the conditions
variable (I : ℕ) -- the number of dolls Ivy has
variable (Dina_dolls : ℕ := 60) -- Dina has 60 dolls
variable (Ivy_collectors : ℕ := 20) -- Ivy has 20 collector edition dolls

-- Condition based on given problem
axiom Ivy_collectors_condition : (2 / 3 : ℚ) * I = 20

-- Lean 4 statement for the proof problem
theorem ratio_of_dolls (h : 3 * Ivy_collectors = 2 * I) : Dina_dolls / I = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_dolls_l213_21315


namespace NUMINAMATH_GPT_abs_add_three_eq_two_l213_21329

theorem abs_add_three_eq_two (a : ℝ) (h : a = -1) : |a + 3| = 2 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_abs_add_three_eq_two_l213_21329


namespace NUMINAMATH_GPT_num_points_satisfying_inequalities_l213_21333

theorem num_points_satisfying_inequalities :
  ∃ (n : ℕ), n = 2551 ∧
  ∀ (x y : ℤ), (y ≤ 3 * x) ∧ (y ≥ x / 3) ∧ (x + y ≤ 100) → 
              ∃ (p : ℕ), p = n := 
by
  sorry

end NUMINAMATH_GPT_num_points_satisfying_inequalities_l213_21333


namespace NUMINAMATH_GPT_train_travel_distance_l213_21376

def speed (miles : ℕ) (minutes : ℕ) : ℕ :=
  miles / minutes

def minutes_in_hours (hours : ℕ) : ℕ :=
  hours * 60

def distance_traveled (rate : ℕ) (time : ℕ) : ℕ :=
  rate * time

theorem train_travel_distance :
  (speed 2 2 = 1) →
  (minutes_in_hours 3 = 180) →
  distance_traveled (speed 2 2) (minutes_in_hours 3) = 180 :=
by
  intros h_speed h_minutes
  rw [h_speed, h_minutes]
  sorry

end NUMINAMATH_GPT_train_travel_distance_l213_21376


namespace NUMINAMATH_GPT_probability_is_correct_l213_21383

-- Given definitions
def total_marbles : ℕ := 100
def red_marbles : ℕ := 35
def white_marbles : ℕ := 30
def green_marbles : ℕ := 10

-- Probe the probability
noncomputable def probability_red_white_green : ℚ :=
  (red_marbles + white_marbles + green_marbles : ℚ) / total_marbles

-- The theorem we need to prove
theorem probability_is_correct :
  probability_red_white_green = 0.75 := by
  sorry

end NUMINAMATH_GPT_probability_is_correct_l213_21383


namespace NUMINAMATH_GPT_nonnegative_values_ineq_l213_21375

theorem nonnegative_values_ineq {x : ℝ} : 
  (x^2 - 6*x + 9) / (9 - x^3) ≥ 0 ↔ x ∈ Set.Iic 3 := 
sorry

end NUMINAMATH_GPT_nonnegative_values_ineq_l213_21375


namespace NUMINAMATH_GPT_clara_gave_10_stickers_l213_21359

-- Defining the conditions
def initial_stickers : ℕ := 100
def remaining_after_boy (B : ℕ) : ℕ := initial_stickers - B
def remaining_after_friends (B : ℕ) : ℕ := (remaining_after_boy B) / 2

-- Theorem stating that Clara gave 10 stickers to the boy
theorem clara_gave_10_stickers (B : ℕ) (h : remaining_after_friends B = 45) : B = 10 :=
by
  sorry

end NUMINAMATH_GPT_clara_gave_10_stickers_l213_21359


namespace NUMINAMATH_GPT_factory_output_l213_21363

theorem factory_output :
  ∀ (J M : ℝ), M = J * 0.8 → J = M * 1.25 :=
by
  intros J M h
  sorry

end NUMINAMATH_GPT_factory_output_l213_21363


namespace NUMINAMATH_GPT_simplify_tangent_sum_l213_21340

theorem simplify_tangent_sum :
  (1 + Real.tan (10 * Real.pi / 180)) * (1 + Real.tan (35 * Real.pi / 180)) = 2 := by
  have h1 : Real.tan (45 * Real.pi / 180) = Real.tan ((10 + 35) * Real.pi / 180) := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have h3 : ∀ x y, Real.tan (x + y) = (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y) := by sorry
  sorry

end NUMINAMATH_GPT_simplify_tangent_sum_l213_21340


namespace NUMINAMATH_GPT_rectangle_area_l213_21380

theorem rectangle_area 
  (P : ℝ) (r : ℝ) (hP : P = 40) (hr : r = 3 / 2) : 
  ∃ (length width : ℝ), 2 * (length + width) = P ∧ length = 3 * (width / 2) ∧ (length * width) = 96 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l213_21380


namespace NUMINAMATH_GPT_number_of_deluxe_volumes_l213_21396

theorem number_of_deluxe_volumes (d s : ℕ) 
  (h1 : d + s = 15)
  (h2 : 30 * d + 20 * s = 390) : 
  d = 9 :=
by
  sorry

end NUMINAMATH_GPT_number_of_deluxe_volumes_l213_21396


namespace NUMINAMATH_GPT_remainder_2_pow_33_mod_9_l213_21399

theorem remainder_2_pow_33_mod_9 : 2^33 % 9 = 8 := by
  sorry

end NUMINAMATH_GPT_remainder_2_pow_33_mod_9_l213_21399


namespace NUMINAMATH_GPT_sum_of_abc_eq_11_l213_21314

theorem sum_of_abc_eq_11 (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_order : a < b ∧ b < c)
  (h_inv_sum : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = 1) : a + b + c = 11 :=
  sorry

end NUMINAMATH_GPT_sum_of_abc_eq_11_l213_21314


namespace NUMINAMATH_GPT_machines_produce_x_units_l213_21321

variable (x : ℕ) (d : ℕ)

-- Define the conditions
def four_machines_produce_in_d_days (x : ℕ) (d : ℕ) : Prop := 
  4 * (x / d) = x / d

def twelve_machines_produce_three_x_in_d_days (x : ℕ) (d : ℕ) : Prop := 
  12 * (x / d) = 3 * (x / d)

-- Given the conditions, prove the number of days for 4 machines to produce x units
theorem machines_produce_x_units (x : ℕ) (d : ℕ) 
  (H1 : four_machines_produce_in_d_days x d)
  (H2 : twelve_machines_produce_three_x_in_d_days x d) : 
  x / d = x / d := 
by 
  sorry

end NUMINAMATH_GPT_machines_produce_x_units_l213_21321


namespace NUMINAMATH_GPT_correct_statements_count_l213_21397

/-
  Question: How many students have given correct interpretations of the algebraic expression \( 7x \)?
  Conditions:
    - Xiaoming's Statement: \( 7x \) can represent the sum of \( 7 \) and \( x \).
    - Xiaogang's Statement: \( 7x \) can represent the product of \( 7 \) and \( x \).
    - Xiaoliang's Statement: \( 7x \) can represent the total price of buying \( x \) pens at a unit price of \( 7 \) yuan.
  Given these conditions, prove that the number of correct statements is \( 2 \).
-/

theorem correct_statements_count (x : ℕ) :
  (if 7 * x = 7 + x then 1 else 0) +
  (if 7 * x = 7 * x then 1 else 0) +
  (if 7 * x = 7 * x then 1 else 0) = 2 := sorry

end NUMINAMATH_GPT_correct_statements_count_l213_21397


namespace NUMINAMATH_GPT_original_triangle_area_l213_21306

theorem original_triangle_area (new_area : ℝ) (scaling_factor : ℝ) (area_ratio : ℝ) : 
  new_area = 32 → scaling_factor = 2 → 
  area_ratio = scaling_factor ^ 2 → 
  new_area / area_ratio = 8 := 
by
  intros
  -- insert your proof logic here
  sorry

end NUMINAMATH_GPT_original_triangle_area_l213_21306


namespace NUMINAMATH_GPT_age_ratio_l213_21355

theorem age_ratio (B_current A_current B_10_years_ago A_in_10_years : ℕ) 
  (h1 : B_current = 37) 
  (h2 : A_current = B_current + 7) 
  (h3 : B_10_years_ago = B_current - 10) 
  (h4 : A_in_10_years = A_current + 10) : 
  A_in_10_years / B_10_years_ago = 2 :=
by
  sorry

end NUMINAMATH_GPT_age_ratio_l213_21355


namespace NUMINAMATH_GPT_minimum_workers_needed_l213_21325

noncomputable def units_per_first_worker : Nat := 48
noncomputable def units_per_second_worker : Nat := 32
noncomputable def units_per_third_worker : Nat := 28

def minimum_workers_first_process : Nat := 14
def minimum_workers_second_process : Nat := 21
def minimum_workers_third_process : Nat := 24

def lcm_3_nat (a b c : Nat) : Nat :=
  Nat.lcm (Nat.lcm a b) c

theorem minimum_workers_needed (a b c : Nat) (w1 w2 w3 : Nat)
  (h1 : a = 48) (h2 : b = 32) (h3 : c = 28)
  (hw1 : w1 = minimum_workers_first_process )
  (hw2 : w2 = minimum_workers_second_process )
  (hw3 : w3 = minimum_workers_third_process ) :
  lcm_3_nat a b c / a = w1 ∧ lcm_3_nat a b c / b = w2 ∧ lcm_3_nat a b c / c = w3 :=
by
  sorry

end NUMINAMATH_GPT_minimum_workers_needed_l213_21325


namespace NUMINAMATH_GPT_max_value_fraction_l213_21326

theorem max_value_fraction (x y : ℝ) (hx : 1 / 3 ≤ x ∧ x ≤ 3 / 5) (hy : 1 / 4 ≤ y ∧ y ≤ 1 / 2) :
  (∃ x y, (1 / 3 ≤ x ∧ x ≤ 3 / 5) ∧ (1 / 4 ≤ y ∧ y ≤ 1 / 2) ∧ (xy / (x^2 + y^2) = 6 / 13)) :=
by
  sorry

end NUMINAMATH_GPT_max_value_fraction_l213_21326


namespace NUMINAMATH_GPT_num_play_both_l213_21303

-- Definitions based on the conditions
def total_members : ℕ := 30
def play_badminton : ℕ := 17
def play_tennis : ℕ := 19
def play_neither : ℕ := 2

-- The statement we want to prove
theorem num_play_both :
  play_badminton + play_tennis - 8 = total_members - play_neither := by
  -- Omitted proof
  sorry

end NUMINAMATH_GPT_num_play_both_l213_21303


namespace NUMINAMATH_GPT_sampling_methods_correct_l213_21309

def first_method_sampling : String :=
  "Simple random sampling"

def second_method_sampling : String :=
  "Systematic sampling"

theorem sampling_methods_correct :
  first_method_sampling = "Simple random sampling" ∧ second_method_sampling = "Systematic sampling" :=
by
  sorry

end NUMINAMATH_GPT_sampling_methods_correct_l213_21309


namespace NUMINAMATH_GPT_blue_pens_count_l213_21357

variable (x y : ℕ) -- Define x as the number of red pens and y as the number of blue pens.
variable (h1 : 5 * x + 7 * y = 102) -- Condition 1: Total cost equation.
variable (h2 : x + y = 16) -- Condition 2: Total number of pens equation.

theorem blue_pens_count : y = 11 :=
by
  sorry

end NUMINAMATH_GPT_blue_pens_count_l213_21357


namespace NUMINAMATH_GPT_angle_measure_l213_21384

theorem angle_measure (x : ℝ) : 
  (180 - x = 7 * (90 - x)) → x = 75 :=
by
  sorry

end NUMINAMATH_GPT_angle_measure_l213_21384


namespace NUMINAMATH_GPT_infinite_div_by_100_l213_21337

theorem infinite_div_by_100 : ∀ k : ℕ, ∃ n : ℕ, n > 0 ∧ (2 ^ n + n ^ 2) % 100 = 0 :=
by
  sorry

end NUMINAMATH_GPT_infinite_div_by_100_l213_21337


namespace NUMINAMATH_GPT_sum_first_2017_terms_l213_21393

theorem sum_first_2017_terms (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → S (n + 1) - S n = 3^n / a n) :
  S 2017 = 3^1009 - 2 := sorry

end NUMINAMATH_GPT_sum_first_2017_terms_l213_21393


namespace NUMINAMATH_GPT_value_of_f_of_g_l213_21395

def f (x : ℝ) : ℝ := 2 * x + 4
def g (x : ℝ) : ℝ := x^2 - 9

theorem value_of_f_of_g : f (g 3) = 4 :=
by
  -- The proof would go here. Since we are only defining the statement, we can leave this as 'sorry'.
  sorry

end NUMINAMATH_GPT_value_of_f_of_g_l213_21395


namespace NUMINAMATH_GPT_algebraic_identity_l213_21365

theorem algebraic_identity 
  (p q r a b c : ℝ)
  (h₁ : p + q + r = 1)
  (h₂ : 1 / p + 1 / q + 1 / r = 0) :
  a^2 + b^2 + c^2 = (p * a + q * b + r * c)^2 + (q * a + r * b + p * c)^2 + (r * a + p * b + q * c)^2 := by
  sorry

end NUMINAMATH_GPT_algebraic_identity_l213_21365


namespace NUMINAMATH_GPT_calculate_speed_of_stream_l213_21307

noncomputable def speed_of_stream (boat_speed : ℕ) (downstream_distance : ℕ) (upstream_distance : ℕ) : ℕ :=
  let x := (downstream_distance * boat_speed - boat_speed * upstream_distance) / (downstream_distance + upstream_distance)
  x

theorem calculate_speed_of_stream :
  speed_of_stream 20 26 14 = 6 := by
  sorry

end NUMINAMATH_GPT_calculate_speed_of_stream_l213_21307


namespace NUMINAMATH_GPT_find_constants_l213_21390

variable (x : ℝ)

/-- Restate the equation problem and the constants A, B, C, D to be found. -/
theorem find_constants 
  (A B C D : ℝ)
  (h : ∀ x, x^3 - 7 = A * (x - 3) * (x - 5) * (x - 7) + B * (x - 2) * (x - 5) * (x - 7) + C * (x - 2) * (x - 3) * (x - 7) + D * (x - 2) * (x - 3) * (x - 5)) :
  A = 1/15 ∧ B = 5/2 ∧ C = -59/6 ∧ D = 42/5 :=
  sorry

end NUMINAMATH_GPT_find_constants_l213_21390


namespace NUMINAMATH_GPT_unique_prime_p_l213_21389

def f (x : ℤ) : ℤ := x^3 + 7 * x^2 + 9 * x + 10

theorem unique_prime_p (p : ℕ) (hp : p = 5 ∨ p = 7 ∨ p = 11 ∨ p = 13 ∨ p = 17) :
  (∀ a b : ℤ, f a ≡ f b [ZMOD p] → a ≡ b [ZMOD p]) ↔ p = 11 :=
by
  sorry

end NUMINAMATH_GPT_unique_prime_p_l213_21389


namespace NUMINAMATH_GPT_smallest_number_of_eggs_over_150_l213_21342

theorem smallest_number_of_eggs_over_150 
  (d : ℕ) 
  (h1: 12 * d - 3 > 150) 
  (h2: ∀ k < d, 12 * k - 3 ≤ 150) :
  12 * d - 3 = 153 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_eggs_over_150_l213_21342


namespace NUMINAMATH_GPT_total_miles_Wednesday_l213_21313

-- The pilot flew 1134 miles on Tuesday and 1475 miles on Thursday.
def miles_flown_Tuesday : ℕ := 1134
def miles_flown_Thursday : ℕ := 1475

-- The miles flown on Wednesday is denoted as "x".
variable (x : ℕ)

-- The period is 4 weeks.
def weeks : ℕ := 4

-- We need to prove that the total miles flown on Wednesdays during this 4-week period is 4 * x.
theorem total_miles_Wednesday : 4 * x = 4 * x := by sorry

end NUMINAMATH_GPT_total_miles_Wednesday_l213_21313


namespace NUMINAMATH_GPT_length_of_generatrix_l213_21385

/-- Given that the base radius of a cone is sqrt(2), and its lateral surface is unfolded into a semicircle,
prove that the length of the generatrix of the cone is 2 sqrt(2). -/
theorem length_of_generatrix (r l : ℝ) (h1 : r = Real.sqrt 2)
    (h2 : 2 * Real.pi * r = Real.pi * l) : l = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_length_of_generatrix_l213_21385


namespace NUMINAMATH_GPT_min_value_x2_y2_z2_l213_21334

theorem min_value_x2_y2_z2 (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + 2 * y + 3 * z = 2) : 
  x^2 + y^2 + z^2 ≥ 2 / 7 :=
sorry

end NUMINAMATH_GPT_min_value_x2_y2_z2_l213_21334


namespace NUMINAMATH_GPT_valid_third_side_length_l213_21348

theorem valid_third_side_length : 4 < 6 ∧ 6 < 10 :=
by
  exact ⟨by norm_num, by norm_num⟩

end NUMINAMATH_GPT_valid_third_side_length_l213_21348


namespace NUMINAMATH_GPT_find_values_of_a_and_b_l213_21316

theorem find_values_of_a_and_b
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (x : ℝ) (hx : x > 1)
  (h : 9 * (Real.log x / Real.log a)^2 + 5 * (Real.log x / Real.log b)^2 = 17)
  (h2 : (Real.log b / Real.log a) * (Real.log a / Real.log b) = 2) :
  a = 10 ^ Real.sqrt 2 ∧ b = 10 := by
sorry

end NUMINAMATH_GPT_find_values_of_a_and_b_l213_21316


namespace NUMINAMATH_GPT_intersect_at_one_point_l213_21331

-- Define the equations as given in the conditions
def equation1 (b : ℝ) (x : ℝ) : ℝ := b * x ^ 2 + 2 * x + 2
def equation2 (x : ℝ) : ℝ := -2 * x - 2

-- Statement of the theorem
theorem intersect_at_one_point (b : ℝ) :
  (∀ x : ℝ, equation1 b x = equation2 x → x = 1) ↔ b = 1 := sorry

end NUMINAMATH_GPT_intersect_at_one_point_l213_21331


namespace NUMINAMATH_GPT_find_C_when_F_10_l213_21311

theorem find_C_when_F_10 : (∃ C : ℚ, ∀ F : ℚ, F = 10 → F = (9 / 5 : ℚ) * C + 32 → C = -110 / 9) :=
by
  sorry

end NUMINAMATH_GPT_find_C_when_F_10_l213_21311


namespace NUMINAMATH_GPT_log_increasing_a_gt_one_l213_21377

noncomputable def log (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_increasing_a_gt_one (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : log a 2 < log a 3) : a > 1 :=
by
  sorry

end NUMINAMATH_GPT_log_increasing_a_gt_one_l213_21377


namespace NUMINAMATH_GPT_fraction_eq_four_l213_21301

theorem fraction_eq_four (a b : ℝ) (h1 : a * b ≠ 0) (h2 : 3 * b = 2 * a) : 
  (2 * a + b) / b = 4 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_eq_four_l213_21301


namespace NUMINAMATH_GPT_area_of_unpainted_section_l213_21312

-- Define the conditions
def board1_width : ℝ := 5
def board2_width : ℝ := 7
def cross_angle : ℝ := 45
def negligible_holes : Prop := true

-- The main statement
theorem area_of_unpainted_section (h1 : board1_width = 5) (h2 : board2_width = 7) (h3 : cross_angle = 45) (h4 : negligible_holes) : 
  ∃ (area : ℝ), area = 35 := 
sorry

end NUMINAMATH_GPT_area_of_unpainted_section_l213_21312


namespace NUMINAMATH_GPT_inequality_proof_l213_21371

section
variable {a b x y : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) (hab : a + b = 1) :
  (1 / (a / x + b / y) ≤ a * x + b * y) ∧ (1 / (a / x + b / y) = a * x + b * y ↔ a * y = b * x) :=
  sorry
end

end NUMINAMATH_GPT_inequality_proof_l213_21371


namespace NUMINAMATH_GPT_sum_of_roots_of_quadratic_l213_21394

open Polynomial

theorem sum_of_roots_of_quadratic :
  ∀ (m n : ℝ), (m ≠ n ∧ (∀ x, x^2 + 2*x - 1 = 0 → x = m ∨ x = n)) → m + n = -2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_of_quadratic_l213_21394


namespace NUMINAMATH_GPT_ordered_triples_count_l213_21330

theorem ordered_triples_count : 
  let b := 3003
  let side_length_squared := b * b
  let num_divisors := (2 + 1) * (2 + 1) * (2 + 1) * (2 + 1)
  let half_divisors := num_divisors / 2
  half_divisors = 40 := by
  sorry

end NUMINAMATH_GPT_ordered_triples_count_l213_21330


namespace NUMINAMATH_GPT_sum_of_three_squares_power_l213_21341

theorem sum_of_three_squares_power (n a b c k : ℕ) (h : n = a^2 + b^2 + c^2) (h_pos : n > 0) (k_pos : k > 0) :
  ∃ A B C : ℕ, n^(2*k) = A^2 + B^2 + C^2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_squares_power_l213_21341


namespace NUMINAMATH_GPT_least_red_chips_l213_21354

/--
  There are 70 chips in a box. Each chip is either red or blue.
  If the sum of the number of red chips and twice the number of blue chips equals a prime number,
  proving that the least possible number of red chips is 69.
-/
theorem least_red_chips (r b : ℕ) (p : ℕ) (h1 : r + b = 70) (h2 : r + 2 * b = p) (hp : Nat.Prime p) :
  r = 69 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_least_red_chips_l213_21354


namespace NUMINAMATH_GPT_target1_target2_l213_21386

variable (α : ℝ)

-- Define the condition
def tan_alpha := Real.tan α = 2

-- State the first target with the condition considered
theorem target1 (h : tan_alpha α) : 
  (Real.sin α - 4 * Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = -1 / 6 := by
  sorry

-- State the second target with the condition considered
theorem target2 (h : tan_alpha α) : 
  4 * Real.sin α ^ 2 - 3 * Real.sin α * Real.cos α - 5 * Real.cos α ^ 2 = 1 := by
  sorry

end NUMINAMATH_GPT_target1_target2_l213_21386


namespace NUMINAMATH_GPT_second_customer_payment_l213_21339

def price_of_headphones : ℕ := 30
def total_cost_first_customer (P H : ℕ) : ℕ := 5 * P + 8 * H
def total_cost_second_customer (P H : ℕ) : ℕ := 3 * P + 4 * H

theorem second_customer_payment
  (P : ℕ)
  (H_eq : H = price_of_headphones)
  (first_customer_eq : total_cost_first_customer P H = 840) :
  total_cost_second_customer P H = 480 :=
by
  -- Proof to be filled in later
  sorry

end NUMINAMATH_GPT_second_customer_payment_l213_21339


namespace NUMINAMATH_GPT_scientific_notation_example_l213_21346

theorem scientific_notation_example :
  (0.000000007: ℝ) = 7 * 10^(-9 : ℝ) :=
sorry

end NUMINAMATH_GPT_scientific_notation_example_l213_21346


namespace NUMINAMATH_GPT_exists_three_naturals_sum_perfect_square_no_four_naturals_sum_perfect_square_l213_21360

-- Definition for the condition that ab + 10 is a perfect square
def is_perfect_square_sum (a b : ℕ) : Prop := ∃ k : ℕ, a * b + 10 = k * k

-- Problem: Existence of three different natural numbers for which the sum of the product of any two with 10 is a perfect square
theorem exists_three_naturals_sum_perfect_square :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ is_perfect_square_sum a b ∧ is_perfect_square_sum b c ∧ is_perfect_square_sum c a := sorry

-- Problem: Non-existence of four different natural numbers for which the sum of the product of any two with 10 is a perfect square
theorem no_four_naturals_sum_perfect_square :
  ¬ ∃ (a b c d : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ d ∧
    is_perfect_square_sum a b ∧ is_perfect_square_sum a c ∧ is_perfect_square_sum a d ∧
    is_perfect_square_sum b c ∧ is_perfect_square_sum b d ∧ is_perfect_square_sum c d := sorry

end NUMINAMATH_GPT_exists_three_naturals_sum_perfect_square_no_four_naturals_sum_perfect_square_l213_21360


namespace NUMINAMATH_GPT_parabola_intercept_sum_l213_21362

theorem parabola_intercept_sum : 
  let d := 4
  let e := (9 + Real.sqrt 33) / 6
  let f := (9 - Real.sqrt 33) / 6
  d + e + f = 7 :=
by 
  sorry

end NUMINAMATH_GPT_parabola_intercept_sum_l213_21362


namespace NUMINAMATH_GPT_book_distribution_l213_21308

theorem book_distribution (x : ℕ) (h1 : 9 * x + 7 < 11 * x) : 
  9 * x + 7 = totalBooks - 9 * x ∧ totalBooks - 9 * x = 7 :=
by
  sorry

end NUMINAMATH_GPT_book_distribution_l213_21308


namespace NUMINAMATH_GPT_divisibility_problem_l213_21350

theorem divisibility_problem (a b k : ℕ) :
  (a = 7 * k^2 ∧ b = 7 * k) ∨ (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) →
  a * b^2 + b + 7 ∣ a^2 * b + a + b := by
  intro h
  cases h
  case inl h1 =>
    rw [h1.1, h1.2]
    sorry
  case inr h2 =>
    cases h2
    case inl h21 =>
      rw [h21.1, h21.2]
      sorry
    case inr h22 =>
      rw [h22.1, h22.2]
      sorry

end NUMINAMATH_GPT_divisibility_problem_l213_21350


namespace NUMINAMATH_GPT_parabola_vertex_location_l213_21353

theorem parabola_vertex_location (a b c : ℝ) (h1 : ∀ x < 0, a * x^2 + b * x + c ≤ 0) (h2 : a < 0) : 
  -b / (2 * a) ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_location_l213_21353


namespace NUMINAMATH_GPT_find_a_l213_21324

variable (a : ℤ) -- We assume a is an integer for simplicity

def point_on_x_axis (P : Nat × ℤ) : Prop :=
  P.snd = 0

theorem find_a (h : point_on_x_axis (4, 2 * a + 6)) : a = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l213_21324


namespace NUMINAMATH_GPT_president_and_committee_l213_21381

def combinatorial (n k : ℕ) : ℕ := Nat.choose n k

theorem president_and_committee :
  let num_people := 10
  let num_president := 1
  let num_committee := 3
  let num_ways_president := 10
  let num_remaining_people := num_people - num_president
  let num_ways_committee := combinatorial num_remaining_people num_committee
  num_ways_president * num_ways_committee = 840 := 
by
  sorry

end NUMINAMATH_GPT_president_and_committee_l213_21381


namespace NUMINAMATH_GPT_seats_in_hall_l213_21336

theorem seats_in_hall (S : ℝ) (h1 : 0.50 * S = 300) : S = 600 :=
by
  sorry

end NUMINAMATH_GPT_seats_in_hall_l213_21336


namespace NUMINAMATH_GPT_tenth_term_of_sequence_l213_21349

theorem tenth_term_of_sequence : 
  let a_1 := 3
  let d := 6 
  let n := 10 
  (a_1 + (n-1) * d) = 57 := by
  sorry

end NUMINAMATH_GPT_tenth_term_of_sequence_l213_21349
