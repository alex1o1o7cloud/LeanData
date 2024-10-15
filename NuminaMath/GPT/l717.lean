import Mathlib

namespace NUMINAMATH_GPT_cafeteria_extra_fruits_l717_71731

def num_apples_red := 75
def num_apples_green := 35
def num_oranges := 40
def num_bananas := 20
def num_students := 17

def total_fruits := num_apples_red + num_apples_green + num_oranges + num_bananas
def fruits_taken_by_students := num_students
def extra_fruits := total_fruits - fruits_taken_by_students

theorem cafeteria_extra_fruits : extra_fruits = 153 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_cafeteria_extra_fruits_l717_71731


namespace NUMINAMATH_GPT_factorial_inequality_l717_71743

theorem factorial_inequality (n : ℕ) (h : n > 1) : n! < ( (n + 1) / 2 )^n := by
  sorry

end NUMINAMATH_GPT_factorial_inequality_l717_71743


namespace NUMINAMATH_GPT_min_value_of_function_l717_71786

theorem min_value_of_function (x : ℝ) (h: x > 1) :
  ∃ t > 0, x = t + 1 ∧ (t + 3 / t + 3) = 3 + 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_min_value_of_function_l717_71786


namespace NUMINAMATH_GPT_M_inter_N_eq_l717_71700

-- Definitions based on the problem conditions
def M : Set ℝ := { x | abs x ≥ 3 }
def N : Set ℝ := { y | ∃ x ∈ M, y = x^2 }

-- The statement we want to prove
theorem M_inter_N_eq : M ∩ N = { x : ℝ | x ≥ 3 } :=
by
  sorry

end NUMINAMATH_GPT_M_inter_N_eq_l717_71700


namespace NUMINAMATH_GPT_find_angle_between_vectors_l717_71724

open Real

noncomputable def vector_norm (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem find_angle_between_vectors 
  (a b : ℝ × ℝ)
  (a_nonzero : a ≠ (0, 0))
  (b_nonzero : b ≠ (0, 0))
  (ha : vector_norm a = 2)
  (hb : vector_norm b = 3)
  (h_sum : vector_norm (a.1 + b.1, a.2 + b.2) = 1)
  : arccos (dot_product a b / (vector_norm a * vector_norm b)) = π :=
sorry

end NUMINAMATH_GPT_find_angle_between_vectors_l717_71724


namespace NUMINAMATH_GPT_determine_x_l717_71754

theorem determine_x (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0) → x = 3 / 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_determine_x_l717_71754


namespace NUMINAMATH_GPT_solve_for_x_l717_71720

theorem solve_for_x (x : ℝ) : (3 : ℝ)^(4 * x^2 - 3 * x + 5) = (3 : ℝ)^(4 * x^2 + 9 * x - 6) ↔ x = 11 / 12 :=
by sorry

end NUMINAMATH_GPT_solve_for_x_l717_71720


namespace NUMINAMATH_GPT_inequality_proof_l717_71705

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ab * (a + b) + bc * (b + c) + ac * (a + c) ≥ 6 * abc := 
sorry

end NUMINAMATH_GPT_inequality_proof_l717_71705


namespace NUMINAMATH_GPT_maxwell_meets_brad_l717_71755

-- Define the given conditions
def distance_between_homes : ℝ := 94
def maxwell_speed : ℝ := 4
def brad_speed : ℝ := 6
def time_delay : ℝ := 1

-- Define the total time it takes Maxwell to meet Brad
theorem maxwell_meets_brad : ∃ t : ℝ, maxwell_speed * (t + time_delay) + brad_speed * t = distance_between_homes ∧ (t + time_delay = 10) :=
by
  sorry

end NUMINAMATH_GPT_maxwell_meets_brad_l717_71755


namespace NUMINAMATH_GPT_george_change_sum_l717_71701

theorem george_change_sum :
  ∃ n m : ℕ,
    0 ≤ n ∧ n < 19 ∧
    0 ≤ m ∧ m < 10 ∧
    (7 + 5 * n) = (4 + 10 * m) ∧
    (7 + 5 * 14) + (4 + 10 * 7) = 144 :=
by
  -- We declare the problem stating that there exist natural numbers n and m within
  -- the given ranges such that the sums of valid change amounts add up to 144 cents.
  sorry

end NUMINAMATH_GPT_george_change_sum_l717_71701


namespace NUMINAMATH_GPT_factorize_expression_l717_71719

theorem factorize_expression (x : ℝ) : x^3 - 4 * x^2 + 4 * x = x * (x - 2)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l717_71719


namespace NUMINAMATH_GPT_prove_two_minus_a_l717_71763

theorem prove_two_minus_a (a b : ℚ) 
  (h1 : 2 * a + 3 = 5 - b) 
  (h2 : 5 + 2 * b = 10 + a) : 
  2 - a = 11 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_prove_two_minus_a_l717_71763


namespace NUMINAMATH_GPT_solution_set_f_leq_6_sum_of_squares_geq_16_div_3_l717_71733

-- Problem 1: Solution set for the inequality \( f(x) ≤ 6 \)
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

theorem solution_set_f_leq_6 :
  {x : ℝ | f x ≤ 6} = {x : ℝ | -2 ≤ x ∧ x ≤ 4} :=
sorry

-- Problem 2: Prove \( a^2 + b^2 + c^2 ≥ 16/3 \)
variables (a b c : ℝ)
axiom pos_abc : a > 0 ∧ b > 0 ∧ c > 0
axiom sum_abc : a + b + c = 4

theorem sum_of_squares_geq_16_div_3 :
  a^2 + b^2 + c^2 ≥ 16 / 3 :=
sorry

end NUMINAMATH_GPT_solution_set_f_leq_6_sum_of_squares_geq_16_div_3_l717_71733


namespace NUMINAMATH_GPT_calculate_expression_l717_71776

theorem calculate_expression : (-1) ^ 47 + 2 ^ (3 ^ 3 + 4 ^ 2 - 6 ^ 2) = 127 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_expression_l717_71776


namespace NUMINAMATH_GPT_first_place_team_ties_l717_71746

noncomputable def teamPoints (wins ties: ℕ) : ℕ := 2 * wins + ties

theorem first_place_team_ties {T : ℕ} : 
  teamPoints 13 1 + teamPoints 8 10 + teamPoints 12 T = 81 → T = 4 :=
by
  sorry

end NUMINAMATH_GPT_first_place_team_ties_l717_71746


namespace NUMINAMATH_GPT_perfect_square_A_perfect_square_D_l717_71782

def is_even (n : ℕ) : Prop := n % 2 = 0

def A : ℕ := 2^10 * 3^12 * 7^14
def D : ℕ := 2^20 * 3^16 * 7^12

theorem perfect_square_A : ∃ k : ℕ, A = k^2 :=
by
  sorry

theorem perfect_square_D : ∃ k : ℕ, D = k^2 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_A_perfect_square_D_l717_71782


namespace NUMINAMATH_GPT_equality_proof_l717_71779

variable {a b c : ℝ}

theorem equality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b)) ≥ (1 / 2) * (a + b + c) :=
by
  sorry

end NUMINAMATH_GPT_equality_proof_l717_71779


namespace NUMINAMATH_GPT_product_of_consecutive_even_numbers_divisible_by_24_l717_71751

theorem product_of_consecutive_even_numbers_divisible_by_24 (n : ℕ) :
  (2 * n) * (2 * n + 2) * (2 * n + 4) % 24 = 0 :=
  sorry

end NUMINAMATH_GPT_product_of_consecutive_even_numbers_divisible_by_24_l717_71751


namespace NUMINAMATH_GPT_find_value_in_table_l717_71713

theorem find_value_in_table :
  let W := 'W'
  let L := 'L'
  let Q := 'Q'
  let table := [
    [W, '?', Q],
    [L, Q, W],
    [Q, W, L]
  ]
  table[0][1] = L :=
by
  sorry

end NUMINAMATH_GPT_find_value_in_table_l717_71713


namespace NUMINAMATH_GPT_complete_square_result_l717_71784

theorem complete_square_result (x : ℝ) :
  (∃ r s : ℝ, (16 * x ^ 2 + 32 * x - 1280 = 0) → ((x + r) ^ 2 = s) ∧ s = 81) :=
by
  sorry

end NUMINAMATH_GPT_complete_square_result_l717_71784


namespace NUMINAMATH_GPT_compute_diff_of_squares_l717_71773

theorem compute_diff_of_squares : (65^2 - 35^2 = 3000) :=
by
  sorry

end NUMINAMATH_GPT_compute_diff_of_squares_l717_71773


namespace NUMINAMATH_GPT_least_number_divisible_by_13_l717_71796

theorem least_number_divisible_by_13 (n : ℕ) :
  (∀ m : ℕ, 2 ≤ m ∧ m ≤ 7 → n % m = 2) ∧ (n % 13 = 0) → n = 1262 :=
by sorry

end NUMINAMATH_GPT_least_number_divisible_by_13_l717_71796


namespace NUMINAMATH_GPT_determine_guilty_resident_l717_71775

structure IslandResident where
  name : String
  is_guilty : Bool
  is_knight : Bool
  is_liar : Bool
  is_normal : Bool -- derived condition: ¬is_knight ∧ ¬is_liar

def A : IslandResident := { name := "A", is_guilty := false, is_knight := false, is_liar := false, is_normal := true }
def B : IslandResident := { name := "B", is_guilty := true, is_knight := true, is_liar := false, is_normal := false }
def C : IslandResident := { name := "C", is_guilty := false, is_knight := false, is_liar := true, is_normal := false }

-- Condition: Only one of them is guilty.
def one_guilty (A B C : IslandResident) : Prop :=
  A.is_guilty ≠ B.is_guilty ∧ A.is_guilty ≠ C.is_guilty ∧ B.is_guilty ≠ C.is_guilty ∧ (A.is_guilty ∨ B.is_guilty ∨ C.is_guilty)

-- Condition: The guilty one is a knight.
def guilty_is_knight (A B C : IslandResident) : Prop :=
  (A.is_guilty → A.is_knight) ∧ (B.is_guilty → B.is_knight) ∧ (C.is_guilty → C.is_knight)

-- Statements made by each resident.
def statements_made (A B C : IslandResident) : Prop :=
  (A.is_guilty = false) ∧ (B.is_guilty = false) ∧ (B.is_normal = false)

theorem determine_guilty_resident (A B C : IslandResident) :
  one_guilty A B C →
  guilty_is_knight A B C →
  statements_made A B C →
  B.is_guilty ∧ B.is_knight :=
by
  sorry

end NUMINAMATH_GPT_determine_guilty_resident_l717_71775


namespace NUMINAMATH_GPT_sum_of_squares_of_rates_l717_71725

theorem sum_of_squares_of_rates (b j s : ℕ) 
  (h1 : 3 * b + 2 * j + 4 * s = 66) 
  (h2 : 3 * j + 2 * s + 4 * b = 96) : 
  b^2 + j^2 + s^2 = 612 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_rates_l717_71725


namespace NUMINAMATH_GPT_smallest_integer_to_make_square_l717_71745

noncomputable def y : ℕ := 2^37 * 3^18 * 5^6 * 7^8

theorem smallest_integer_to_make_square : ∃ z : ℕ, z = 10 ∧ ∃ k : ℕ, (y * z) = k^2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_to_make_square_l717_71745


namespace NUMINAMATH_GPT_parabola_directrix_l717_71787

theorem parabola_directrix : 
  ∃ d : ℝ, (∀ (y : ℝ), x = - (1 / 4) * y^2 -> 
  ( - (1 / 4) * y^2 = d)) ∧ d = 1 :=
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_l717_71787


namespace NUMINAMATH_GPT_additional_track_length_needed_l717_71760

theorem additional_track_length_needed
  (vertical_rise : ℝ) (initial_grade final_grade : ℝ) (initial_horizontal_length final_horizontal_length : ℝ) : 
  vertical_rise = 400 →
  initial_grade = 0.04 →
  final_grade = 0.03 →
  initial_horizontal_length = (vertical_rise / initial_grade) →
  final_horizontal_length = (vertical_rise / final_grade) →
  final_horizontal_length - initial_horizontal_length = 3333 :=
by
  intros h_vertical_rise h_initial_grade h_final_grade h_initial_horizontal_length h_final_horizontal_length
  sorry

end NUMINAMATH_GPT_additional_track_length_needed_l717_71760


namespace NUMINAMATH_GPT_sequence_comparison_l717_71771

noncomputable def geom_seq (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)
noncomputable def arith_seq (b₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := b₁ + (n-1) * d

theorem sequence_comparison
  (a₁ b₁ q d : ℝ)
  (h₃ : geom_seq a₁ q 3 = arith_seq b₁ d 3)
  (h₇ : geom_seq a₁ q 7 = arith_seq b₁ d 7)
  (q_pos : 0 < q)
  (d_pos : 0 < d) :
  geom_seq a₁ q 5 < arith_seq b₁ d 5 ∧
  geom_seq a₁ q 1 > arith_seq b₁ d 1 ∧
  geom_seq a₁ q 9 > arith_seq b₁ d 9 :=
by
  sorry

end NUMINAMATH_GPT_sequence_comparison_l717_71771


namespace NUMINAMATH_GPT_solution_inequality_l717_71792

theorem solution_inequality (p p' q q' : ℝ) (hp : p ≠ 0) (hp' : p' ≠ 0)
    (h : -q / p > -q' / p') : q / p < q' / p' :=
by
  sorry

end NUMINAMATH_GPT_solution_inequality_l717_71792


namespace NUMINAMATH_GPT_total_cost_of_purchase_l717_71778

variable (x y z : ℝ)

theorem total_cost_of_purchase (h₁ : 4 * x + (9 / 2) * y + 12 * z = 6) (h₂ : 12 * x + 6 * y + 6 * z = 8) :
  4 * x + 3 * y + 6 * z = 4 :=
sorry

end NUMINAMATH_GPT_total_cost_of_purchase_l717_71778


namespace NUMINAMATH_GPT_board_game_cost_correct_l717_71732

-- Definitions
def jump_rope_cost : ℕ := 7
def ball_cost : ℕ := 4
def saved_money : ℕ := 6
def gift_money : ℕ := 13
def needed_money : ℕ := 4

-- Total money Dalton has
def total_money : ℕ := saved_money + gift_money

-- Total cost of all items
def total_cost : ℕ := total_money + needed_money

-- Combined cost of jump rope and ball
def combined_cost_jump_rope_ball : ℕ := jump_rope_cost + ball_cost

-- Cost of the board game
def board_game_cost : ℕ := total_cost - combined_cost_jump_rope_ball

-- Theorem to prove
theorem board_game_cost_correct : board_game_cost = 12 :=
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_board_game_cost_correct_l717_71732


namespace NUMINAMATH_GPT_Alma_test_score_l717_71793

-- Define the constants and conditions
variables (Alma_age Melina_age Alma_score : ℕ)

-- Conditions
axiom Melina_is_60 : Melina_age = 60
axiom Melina_3_times_Alma : Melina_age = 3 * Alma_age
axiom sum_ages_twice_score : Melina_age + Alma_age = 2 * Alma_score

-- Goal
theorem Alma_test_score : Alma_score = 40 :=
by
  sorry

end NUMINAMATH_GPT_Alma_test_score_l717_71793


namespace NUMINAMATH_GPT_jasmine_paperclips_l717_71764

theorem jasmine_paperclips :
  ∃ k : ℕ, (4 * 3^k > 500) ∧ (∀ n < k, 4 * 3^n ≤ 500) ∧ k = 5 ∧ (n = 6) :=
by {
  sorry
}

end NUMINAMATH_GPT_jasmine_paperclips_l717_71764


namespace NUMINAMATH_GPT_remainder_div_1356_l717_71742

theorem remainder_div_1356 :
  ∃ R : ℝ, ∃ L : ℝ, ∃ S : ℝ, S = 268.2 ∧ L - S = 1356 ∧ L = 6 * S + R ∧ R = 15 :=
by
  sorry

end NUMINAMATH_GPT_remainder_div_1356_l717_71742


namespace NUMINAMATH_GPT_probability_inside_circle_is_2_div_9_l717_71728

noncomputable def probability_point_in_circle : ℚ := 
  let total_points := 36
  let points_inside := 8
  points_inside / total_points

theorem probability_inside_circle_is_2_div_9 :
  probability_point_in_circle = 2 / 9 :=
by
  -- we acknowledge the mathematical computation here
  sorry

end NUMINAMATH_GPT_probability_inside_circle_is_2_div_9_l717_71728


namespace NUMINAMATH_GPT_find_k_l717_71799

-- Definitions for the given conditions
def slope_of_first_line : ℝ := 2
def alpha : ℝ := slope_of_first_line
def slope_of_second_line : ℝ := 2 * alpha

-- The proof goal
theorem find_k (k : ℝ) : slope_of_second_line = k ↔ k = 4 := by
  sorry

end NUMINAMATH_GPT_find_k_l717_71799


namespace NUMINAMATH_GPT_paving_stones_needed_l717_71703

def length_courtyard : ℝ := 60
def width_courtyard : ℝ := 14
def width_stone : ℝ := 2
def paving_stones_required : ℕ := 140

theorem paving_stones_needed (L : ℝ) 
  (h1 : length_courtyard * width_courtyard = 840) 
  (h2 : paving_stones_required = 140)
  (h3 : (140 * (L * 2)) = 840) : 
  (length_courtyard * width_courtyard) / (L * width_stone) = 140 := 
by sorry

end NUMINAMATH_GPT_paving_stones_needed_l717_71703


namespace NUMINAMATH_GPT_right_triangle_side_sums_l717_71790

theorem right_triangle_side_sums (a b c : ℕ) (h1 : a + b = c + 6) (h2 : a^2 + b^2 = c^2) :
  (a = 7 ∧ b = 24 ∧ c = 25) ∨ (a = 8 ∧ b = 15 ∧ c = 17) ∨ (a = 9 ∧ b = 12 ∧ c = 15) :=
sorry

end NUMINAMATH_GPT_right_triangle_side_sums_l717_71790


namespace NUMINAMATH_GPT_fencing_cost_per_meter_l717_71736

-- Definitions based on given conditions
def area : ℚ := 1200
def short_side : ℚ := 30
def total_cost : ℚ := 1800

-- Definition to represent the length of the long side
def long_side := area / short_side

-- Definition to represent the diagonal of the rectangle
def diagonal := (long_side^2 + short_side^2).sqrt

-- Definition to represent the total length of the fence
def total_length := long_side + short_side + diagonal

-- Definition to represent the cost per meter
def cost_per_meter := total_cost / total_length

-- Theorem statement asserting that cost_per_meter == 15
theorem fencing_cost_per_meter : cost_per_meter = 15 := 
by 
  sorry

end NUMINAMATH_GPT_fencing_cost_per_meter_l717_71736


namespace NUMINAMATH_GPT_percentage_of_all_students_with_cars_l717_71761

def seniors := 300
def percent_seniors_with_cars := 0.40
def lower_grades := 1500
def percent_lower_grades_with_cars := 0.10

theorem percentage_of_all_students_with_cars :
  (120 + 150) / 1800 * 100 = 15 := by
  sorry

end NUMINAMATH_GPT_percentage_of_all_students_with_cars_l717_71761


namespace NUMINAMATH_GPT_cosine_sum_identity_l717_71717

theorem cosine_sum_identity 
  (α : ℝ) 
  (h_sin : Real.sin α = 3 / 5) 
  (h_alpha_first_quad : 0 < α ∧ α < Real.pi / 2) : 
  Real.cos (Real.pi / 3 + α) = (4 - 3 * Real.sqrt 3) / 10 :=
by
  sorry

end NUMINAMATH_GPT_cosine_sum_identity_l717_71717


namespace NUMINAMATH_GPT_part1_solution_l717_71714

theorem part1_solution : ∀ n : ℕ, ∃ k : ℤ, 2^n + 3 = k^2 ↔ n = 0 :=
by sorry

end NUMINAMATH_GPT_part1_solution_l717_71714


namespace NUMINAMATH_GPT_find_f_2018_l717_71722

-- Define the function f, its periodicity and even property
variable (f : ℝ → ℝ)

-- Conditions
axiom f_periodicity : ∀ x : ℝ, f (x + 4) = -f x
axiom f_symmetric : ∀ x : ℝ, f x = f (-x)
axiom f_at_two : f 2 = 2

-- Theorem stating the desired property
theorem find_f_2018 : f 2018 = 2 :=
  sorry

end NUMINAMATH_GPT_find_f_2018_l717_71722


namespace NUMINAMATH_GPT_min_value_of_quadratic_l717_71753

theorem min_value_of_quadratic :
  ∀ (x : ℝ), ∃ (z : ℝ), z = 4 * x^2 + 8 * x + 16 ∧ z ≥ 12 ∧ (∃ c : ℝ, c = c → z = 12) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l717_71753


namespace NUMINAMATH_GPT_eliot_account_balance_l717_71740

theorem eliot_account_balance (A E : ℝ) 
  (h1 : A - E = (1/12) * (A + E))
  (h2 : A * 1.10 = E * 1.15 + 30) :
  E = 857.14 := by
  sorry

end NUMINAMATH_GPT_eliot_account_balance_l717_71740


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_is_constant_l717_71709

def is_constant (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, S n = c

theorem sum_of_arithmetic_sequence_is_constant
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 2 + a 6 + a 10 = a 1 + d + a 1 + 5 * d + a 1 + 9 * d)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d) :
  is_constant 11 a S :=
by
  sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_is_constant_l717_71709


namespace NUMINAMATH_GPT_percentage_of_b_l717_71721

variable (a b c p : ℝ)

theorem percentage_of_b :
  (0.04 * a = 8) →
  (p * b = 4) →
  (c = b / a) →
  p = 1 / (50 * c) :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_b_l717_71721


namespace NUMINAMATH_GPT_matching_pair_probability_l717_71744

-- Given conditions
def total_gray_socks : ℕ := 12
def total_white_socks : ℕ := 10
def total_socks : ℕ := total_gray_socks + total_white_socks

-- Proof statement
theorem matching_pair_probability (h_grays : total_gray_socks = 12) (h_whites : total_white_socks = 10) :
  (66 + 45) / (total_socks.choose 2) = 111 / 231 :=
by
  sorry

end NUMINAMATH_GPT_matching_pair_probability_l717_71744


namespace NUMINAMATH_GPT_simplify_fraction_1_simplify_fraction_2_l717_71726

variables (a b c : ℝ)

theorem simplify_fraction_1 :
  (a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c) / (a^2 - b^2 - c^2 - 2*b*c) = (a + b + c) / (a - b - c) :=
sorry

theorem simplify_fraction_2 :
  (a^2 - 3*a*b + a*c + 2*b^2 - 2*b*c) / (a^2 - b^2 + 2*b*c - c^2) = (a - 2*b) / (a + b - c) :=
sorry

end NUMINAMATH_GPT_simplify_fraction_1_simplify_fraction_2_l717_71726


namespace NUMINAMATH_GPT_intersection_eq_l717_71748

def A : Set ℝ := {-1, 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 2}
def C : Set ℝ := {2}

theorem intersection_eq : A ∩ B = C := 
by {
  sorry
}

end NUMINAMATH_GPT_intersection_eq_l717_71748


namespace NUMINAMATH_GPT_exactly_one_germinates_l717_71794

theorem exactly_one_germinates (pA pB : ℝ) (hA : pA = 0.8) (hB : pB = 0.9) : 
  (pA * (1 - pB) + (1 - pA) * pB) = 0.26 :=
by
  sorry

end NUMINAMATH_GPT_exactly_one_germinates_l717_71794


namespace NUMINAMATH_GPT_valid_for_expression_c_l717_71788

def expression_a_defined (x : ℝ) : Prop := x ≠ 2
def expression_b_defined (x : ℝ) : Prop := x ≠ 3
def expression_c_defined (x : ℝ) : Prop := x ≥ 2
def expression_d_defined (x : ℝ) : Prop := x ≥ 3

theorem valid_for_expression_c :
  (expression_a_defined 2 = false ∧ expression_a_defined 3 = true) ∧
  (expression_b_defined 2 = true ∧ expression_b_defined 3 = false) ∧
  (expression_c_defined 2 = true ∧ expression_c_defined 3 = true) ∧
  (expression_d_defined 2 = false ∧ expression_d_defined 3 = true) ∧
  (expression_c_defined 2 = true ∧ expression_c_defined 3 = true) := by
  sorry

end NUMINAMATH_GPT_valid_for_expression_c_l717_71788


namespace NUMINAMATH_GPT_completing_the_square_l717_71768

theorem completing_the_square (x : ℝ) :
  x^2 - 6 * x + 2 = 0 →
  (x - 3)^2 = 7 :=
by sorry

end NUMINAMATH_GPT_completing_the_square_l717_71768


namespace NUMINAMATH_GPT_difference_between_two_numbers_l717_71777

theorem difference_between_two_numbers : 
  ∃ (a b : ℕ),
    (a + b = 21780) ∧
    (a % 5 = 0) ∧
    ((a / 10) = b) ∧
    (a - b = 17825) :=
sorry

end NUMINAMATH_GPT_difference_between_two_numbers_l717_71777


namespace NUMINAMATH_GPT_lunks_needed_for_20_apples_l717_71783

-- Definitions based on given conditions
def lunks_to_kunks (lunks : ℕ) : ℕ := (lunks / 4) * 2
def kunks_to_apples (kunks : ℕ) : ℕ := (kunks / 3) * 5

-- The main statement to be proven
theorem lunks_needed_for_20_apples :
  ∃ l : ℕ, (kunks_to_apples (lunks_to_kunks l)) = 20 ∧ l = 24 :=
by
  sorry

end NUMINAMATH_GPT_lunks_needed_for_20_apples_l717_71783


namespace NUMINAMATH_GPT_problem_solution_l717_71715

variable (x y : ℝ)

theorem problem_solution :
  (x - y + 1) * (x - y - 1) = x^2 - 2 * x * y + y^2 - 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l717_71715


namespace NUMINAMATH_GPT_negation_correct_l717_71756

-- Define the statement to be negated
def original_statement (x : ℕ) : Prop := ∀ x : ℕ, x^2 ≠ 4

-- Define the negation of the original statement
def negated_statement (x : ℕ) : Prop := ∃ x : ℕ, x^2 = 4

-- Prove that the negation of the original statement is the given negated statement
theorem negation_correct : (¬ (∀ x : ℕ, x^2 ≠ 4)) ↔ (∃ x : ℕ, x^2 = 4) :=
by sorry

end NUMINAMATH_GPT_negation_correct_l717_71756


namespace NUMINAMATH_GPT_ratio_of_cone_to_sphere_l717_71730

theorem ratio_of_cone_to_sphere (r : ℝ) (h := 2 * r) : 
  (1 / 3 * π * r^2 * h) / ((4 / 3) * π * r^3) = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_cone_to_sphere_l717_71730


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l717_71780

variable (x y : ℝ)

theorem simplify_and_evaluate_expression (h₁ : x = -2) (h₂ : y = 1/2) :
  (x + 2 * y) ^ 2 - (x + y) * (3 * x - y) - 5 * y ^ 2 / (2 * x) = 2 + 1 / 2 := 
sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l717_71780


namespace NUMINAMATH_GPT_fraction_dehydrated_l717_71758

theorem fraction_dehydrated (total_men tripped fraction_dnf finished : ℕ) (fraction_tripped fraction_dehydrated_dnf : ℚ)
  (htotal_men : total_men = 80)
  (hfraction_tripped : fraction_tripped = 1 / 4)
  (htripped : tripped = total_men * fraction_tripped)
  (hfinished : finished = 52)
  (hfraction_dnf : fraction_dehydrated_dnf = 1 / 5)
  (hdnf : total_men - finished = tripped + fraction_dehydrated_dnf * (total_men - tripped) * x)
  (hx : x = 2 / 3) :
  x = 2 / 3 := sorry

end NUMINAMATH_GPT_fraction_dehydrated_l717_71758


namespace NUMINAMATH_GPT_part1_part2_l717_71772

variables (a b : ℝ) (f g : ℝ → ℝ)

-- Step 1: Given a > 0, b > 0 and f(x) = |x - a| - |x + b|, prove that if max(f) = 3, then a + b = 3.
theorem part1 (ha : a > 0) (hb : b > 0) (hf : ∀ x, f x = abs (x - a) - abs (x + b)) (hmax : ∀ x, f x ≤ 3) :
  a + b = 3 :=
sorry

-- Step 2: For g(x) = -x^2 - ax - b, if g(x) < f(x) for all x ≥ a, prove that 1/2 < a < 3.
theorem part2 (ha : a > 0) (hb : b > 0) (hf : ∀ x, f x = abs (x - a) - abs (x + b)) (hmax : ∀ x, f x ≤ 3)
    (hg : ∀ x, g x = -x^2 - a * x - b) (hcond : ∀ x, x ≥ a → g x < f x) :
    1 / 2 < a ∧ a < 3 :=
sorry

end NUMINAMATH_GPT_part1_part2_l717_71772


namespace NUMINAMATH_GPT_edward_initial_lives_l717_71797

def initialLives (lives_lost lives_left : Nat) : Nat :=
  lives_lost + lives_left

theorem edward_initial_lives (lost left : Nat) (H_lost : lost = 8) (H_left : left = 7) :
  initialLives lost left = 15 :=
by
  sorry

end NUMINAMATH_GPT_edward_initial_lives_l717_71797


namespace NUMINAMATH_GPT_initial_apples_count_l717_71702

theorem initial_apples_count (a b : ℕ) (h₁ : b = 13) (h₂ : b = a + 5) : a = 8 :=
by
  sorry

end NUMINAMATH_GPT_initial_apples_count_l717_71702


namespace NUMINAMATH_GPT_solve_for_x_l717_71750

theorem solve_for_x (x : ℝ) (h : (x^2 + 2*x + 3) / (x + 1) = x + 3) : x = 0 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l717_71750


namespace NUMINAMATH_GPT_expected_value_is_350_l717_71737

noncomputable def expected_value_of_winnings : ℚ :=
  ((1 / 8) * (8 - 1) + (1 / 8) * (8 - 2) + (1 / 8) * (8 - 3) + (1 / 8) * (8 - 4) +
  (1 / 8) * (8 - 5) + (1 / 8) * (8 - 6) + (1 / 8) * (8 - 7) + (1 / 8) * (8 - 8))

theorem expected_value_is_350 :
  expected_value_of_winnings = 3.50 := by
  sorry

end NUMINAMATH_GPT_expected_value_is_350_l717_71737


namespace NUMINAMATH_GPT_max_jogs_possible_l717_71770

theorem max_jogs_possible :
  ∃ (x y z : ℕ), (3 * x + 4 * y + 10 * z = 100) ∧ (x + y + z ≥ 20) ∧ (x ≥ 1) ∧ (y ≥ 1) ∧ (z ≥ 1) ∧
  (∀ (x' y' z' : ℕ), (3 * x' + 4 * y' + 10 * z' = 100) ∧ (x' + y' + z' ≥ 20) ∧ (x' ≥ 1) ∧ (y' ≥ 1) ∧ (z' ≥ 1) → z' ≤ z) :=
by
  sorry

end NUMINAMATH_GPT_max_jogs_possible_l717_71770


namespace NUMINAMATH_GPT_find_range_of_a_l717_71765

noncomputable def A (a : ℝ) := { x : ℝ | 1 ≤ x ∧ x ≤ a}
noncomputable def B (a : ℝ) := { y : ℝ | ∃ x : ℝ, y = 5 * x - 6 ∧ 1 ≤ x ∧ x ≤ a }
noncomputable def C (a : ℝ) := { m : ℝ | ∃ x : ℝ, m = x^2 ∧ 1 ≤ x ∧ x ≤ a }

theorem find_range_of_a (a : ℝ) (h : B a ∩ C a = C a) : 2 ≤ a ∧ a ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_find_range_of_a_l717_71765


namespace NUMINAMATH_GPT_value_of_k_l717_71769

open Real

theorem value_of_k {k : ℝ} : 
  (∃ x : ℝ, k * x ^ 2 - 2 * k * x + 4 = 0 ∧ (∀ y : ℝ, k * y ^ 2 - 2 * k * y + 4 = 0 → x = y)) → k = 4 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_value_of_k_l717_71769


namespace NUMINAMATH_GPT_percentage_of_boys_playing_soccer_l717_71704

theorem percentage_of_boys_playing_soccer
  (total_students : ℕ)
  (boys : ℕ)
  (students_playing_soccer : ℕ)
  (girl_students_not_playing_soccer : ℕ)
  (h1 : total_students = 420)
  (h2 : boys = 296)
  (h3 : students_playing_soccer = 250)
  (h4 : girl_students_not_playing_soccer = 89) :
  (students_playing_soccer - (total_students - boys - girl_students_not_playing_soccer)) * 100 / students_playing_soccer = 86 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_boys_playing_soccer_l717_71704


namespace NUMINAMATH_GPT_intersection_complement_eq_l717_71781

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define set M within U
def M : Set ℕ := {1, 3, 5, 7}

-- Define set N within U
def N : Set ℕ := {5, 6, 7}

-- Define the complement of M in U
def CU_M : Set ℕ := U \ M

-- Define the complement of N in U
def CU_N : Set ℕ := U \ N

-- Mathematically equivalent proof problem
theorem intersection_complement_eq : CU_M ∩ CU_N = {2, 4, 8} := by
  sorry

end NUMINAMATH_GPT_intersection_complement_eq_l717_71781


namespace NUMINAMATH_GPT_distance_rowed_downstream_l717_71798

def speed_of_boat_still_water : ℝ := 70 -- km/h
def distance_upstream : ℝ := 240 -- km
def time_upstream : ℝ := 6 -- hours
def time_downstream : ℝ := 5 -- hours

theorem distance_rowed_downstream :
  let V_b := speed_of_boat_still_water
  let V_upstream := distance_upstream / time_upstream
  let V_s := V_b - V_upstream
  let V_downstream := V_b + V_s
  V_downstream * time_downstream = 500 :=
by
  sorry

end NUMINAMATH_GPT_distance_rowed_downstream_l717_71798


namespace NUMINAMATH_GPT_probability_of_point_on_line_4_l717_71723

-- Definitions as per conditions
def total_outcomes : ℕ := 36
def favorable_points : Finset (ℕ × ℕ) := {(1, 3), (2, 2), (3, 1)}
def probability : ℚ := (favorable_points.card : ℚ) / total_outcomes

-- Problem statement to prove
theorem probability_of_point_on_line_4 :
  probability = 1 / 12 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_point_on_line_4_l717_71723


namespace NUMINAMATH_GPT_numberOfHandshakes_is_correct_l717_71738

noncomputable def numberOfHandshakes : ℕ :=
  let gremlins := 30
  let imps := 20
  let friendlyImps := 5
  let gremlinHandshakes := gremlins * (gremlins - 1) / 2
  let impGremlinHandshakes := imps * gremlins
  let friendlyImpHandshakes := friendlyImps * (friendlyImps - 1) / 2
  gremlinHandshakes + impGremlinHandshakes + friendlyImpHandshakes

theorem numberOfHandshakes_is_correct : numberOfHandshakes = 1045 := by
  sorry

end NUMINAMATH_GPT_numberOfHandshakes_is_correct_l717_71738


namespace NUMINAMATH_GPT_problem_l717_71707

structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

def NotParallel (v1 v2 : Vector3D) : Prop := ¬ ∃ k : ℝ, v2 = ⟨k * v1.x, k * v1.y, k * v1.z⟩

def a : Vector3D := ⟨1, 2, -2⟩
def b : Vector3D := ⟨-2, -4, 4⟩
def c : Vector3D := ⟨1, 0, 0⟩
def d : Vector3D := ⟨-3, 0, 0⟩
def g : Vector3D := ⟨-2, 3, 5⟩
def h : Vector3D := ⟨16, 24, 40⟩
def e : Vector3D := ⟨2, 3, 0⟩
def f : Vector3D := ⟨0, 0, 0⟩

theorem problem : NotParallel g h := by
  sorry

end NUMINAMATH_GPT_problem_l717_71707


namespace NUMINAMATH_GPT_maximize_box_volume_l717_71711

-- Define the volume function
def volume (x : ℝ) := (48 - 2 * x)^2 * x

-- Define the constraint on x
def constraint (x : ℝ) := 0 < x ∧ x < 24

-- The theorem stating the side length of the removed square that maximizes the volume
theorem maximize_box_volume : ∃ x : ℝ, constraint x ∧ (∀ y : ℝ, constraint y → volume y ≤ volume 8) :=
by
  sorry

end NUMINAMATH_GPT_maximize_box_volume_l717_71711


namespace NUMINAMATH_GPT_minute_hand_gain_per_hour_l717_71762

theorem minute_hand_gain_per_hour (h_start h_end : ℕ) (time_elapsed : ℕ) 
  (total_gain : ℕ) (gain_per_hour : ℕ) 
  (h_start_eq_9 : h_start = 9)
  (time_period_eq_8 : time_elapsed = 8)
  (total_gain_eq_40 : total_gain = 40)
  (time_elapsed_eq : h_end = h_start + time_elapsed)
  (gain_formula : gain_per_hour * time_elapsed = total_gain) :
  gain_per_hour = 5 := 
by 
  sorry

end NUMINAMATH_GPT_minute_hand_gain_per_hour_l717_71762


namespace NUMINAMATH_GPT_factors_of_48_are_multiples_of_6_l717_71710

theorem factors_of_48_are_multiples_of_6 : 
  ∃ (n : ℕ), n = 4 ∧ ∀ d, d ∣ 48 → (6 ∣ d ↔ d = 6 ∨ d = 12 ∨ d = 24 ∨ d = 48) := 
by { sorry }

end NUMINAMATH_GPT_factors_of_48_are_multiples_of_6_l717_71710


namespace NUMINAMATH_GPT_vieta_formula_l717_71791

-- Define what it means to be a root of a polynomial
noncomputable def is_root (p : ℝ) (a b c d : ℝ) : Prop :=
  a * p^3 + b * p^2 + c * p + d = 0

-- Setting up the variables and conditions for the polynomial
variables (p q r : ℝ)
variable (a b c d : ℝ)
variable (ha : a = 5)
variable (hb : b = -10)
variable (hc : c = 17)
variable (hd : d = -7)
variable (hp : is_root p a b c d)
variable (hq : is_root q a b c d)
variable (hr : is_root r a b c d)

-- Lean statement to prove the desired equality using Vieta's formulas
theorem vieta_formula : 
  pq + qr + rp = c / a :=
by
  -- Translate the problem into Lean structure
  sorry

end NUMINAMATH_GPT_vieta_formula_l717_71791


namespace NUMINAMATH_GPT_int_even_bijection_l717_71712

theorem int_even_bijection :
  ∃ (f : ℤ → ℤ), (∀ n : ℤ, ∃ m : ℤ, f n = m ∧ m % 2 = 0) ∧
                 (∀ m : ℤ, m % 2 = 0 → ∃ n : ℤ, f n = m) := 
sorry

end NUMINAMATH_GPT_int_even_bijection_l717_71712


namespace NUMINAMATH_GPT_product_of_zero_multiples_is_equal_l717_71759

theorem product_of_zero_multiples_is_equal :
  (6000 * 0 = 0) ∧ (6 * 0 = 0) → (6000 * 0 = 6 * 0) :=
by sorry

end NUMINAMATH_GPT_product_of_zero_multiples_is_equal_l717_71759


namespace NUMINAMATH_GPT_cubed_sum_identity_l717_71718

theorem cubed_sum_identity {x y : ℝ} (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end NUMINAMATH_GPT_cubed_sum_identity_l717_71718


namespace NUMINAMATH_GPT_sector_area_l717_71757

theorem sector_area (r : ℝ) (theta : ℝ) (h_r : r = 3) (h_theta : theta = 120) : 
  (theta / 360) * π * r^2 = 3 * π :=
by 
  sorry

end NUMINAMATH_GPT_sector_area_l717_71757


namespace NUMINAMATH_GPT_range_of_a_l717_71716

open Real

noncomputable def f (a x : ℝ) : ℝ := log x + (1 / 2) * x ^ 2 + a * x

theorem range_of_a (a : ℝ) : (∃ x > 0, deriv (f a) x = 3) ↔ a < 1 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l717_71716


namespace NUMINAMATH_GPT_jennifer_money_left_l717_71789

theorem jennifer_money_left (initial_amount sandwich_fraction museum_fraction book_fraction : ℚ)
    (initial_eq : initial_amount = 90) 
    (sandwich_eq : sandwich_fraction = 1/5) 
    (museum_eq : museum_fraction = 1/6) 
    (book_eq : book_fraction = 1/2) :
    initial_amount - (initial_amount * sandwich_fraction + initial_amount * museum_fraction + initial_amount * book_fraction) = 12 := 
by 
  sorry

end NUMINAMATH_GPT_jennifer_money_left_l717_71789


namespace NUMINAMATH_GPT_find_original_number_l717_71735

-- Definitions of the conditions
def isFiveDigitNumber (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

theorem find_original_number (n x y : ℕ) 
  (h1 : isFiveDigitNumber n) 
  (h2 : n = 10 * x + y) 
  (h3 : n - x = 54321) : 
  n = 60356 := 
sorry

end NUMINAMATH_GPT_find_original_number_l717_71735


namespace NUMINAMATH_GPT_problem_1_l717_71706

noncomputable def f (a x : ℝ) : ℝ := abs (x + 2) + abs (x - a)

theorem problem_1 (a : ℝ) (h : ∀ x : ℝ, f a x ≥ 3) : a ≤ -5 ∨ a ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_1_l717_71706


namespace NUMINAMATH_GPT_number_of_solutions_l717_71774

theorem number_of_solutions :
  (∀ (x : ℝ), (3 * x ^ 3 - 15 * x ^ 2) / (x ^ 2 - 5 * x) = 2 * x - 6 → x ≠ 0 ∧ x ≠ 5) →
  ∃! (x : ℝ), (3 * x ^ 3 - 15 * x ^ 2) / (x ^ 2 - 5 * x) = 2 * x - 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_solutions_l717_71774


namespace NUMINAMATH_GPT_rectangle_length_increase_decrease_l717_71734

theorem rectangle_length_increase_decrease
  (L : ℝ)
  (width : ℝ)
  (increase_percentage : ℝ)
  (decrease_percentage : ℝ)
  (new_width : ℝ)
  (initial_area : ℝ)
  (new_length : ℝ)
  (new_area : ℝ)
  (HLW : width = 40)
  (Hinc : increase_percentage = 0.30)
  (Hdec : decrease_percentage = 0.17692307692307693)
  (Hnew_width : new_width = 40 - (decrease_percentage * 40))
  (Hinitial_area : initial_area = L * 40)
  (Hnew_length : new_length = 1.30 * L)
  (Hequal_area : new_length * new_width = L * 40) :
  L = 30.76923076923077 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_length_increase_decrease_l717_71734


namespace NUMINAMATH_GPT_failed_both_l717_71747

-- Defining the conditions based on the problem statement
def failed_hindi : ℝ := 0.34
def failed_english : ℝ := 0.44
def passed_both : ℝ := 0.44

-- Defining a proposition to represent the problem and its solution
theorem failed_both (x : ℝ) (h1 : x = failed_hindi + failed_english - (1 - passed_both)) : 
  x = 0.22 :=
by
  sorry

end NUMINAMATH_GPT_failed_both_l717_71747


namespace NUMINAMATH_GPT_pirates_total_coins_l717_71795

theorem pirates_total_coins :
  ∀ (x : ℕ), (∃ (paul_coins pete_coins : ℕ), 
  paul_coins = x ∧ pete_coins = 5 * x ∧ pete_coins = (x * (x + 1)) / 2) → x + 5 * x = 54 := by
  sorry

end NUMINAMATH_GPT_pirates_total_coins_l717_71795


namespace NUMINAMATH_GPT_find_principal_l717_71785

-- Definitions based on conditions
def simple_interest (P R T : ℚ) : ℚ := (P * R * T) / 100

-- Given conditions
def SI : ℚ := 6016.75
def R : ℚ := 8
def T : ℚ := 5

-- Stating the proof problem
theorem find_principal : 
  ∃ P : ℚ, simple_interest P R T = SI ∧ P = 15041.875 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_principal_l717_71785


namespace NUMINAMATH_GPT_problem_solution_l717_71741

theorem problem_solution (p q r : ℝ) 
    (h1 : (p * r / (p + q) + q * p / (q + r) + r * q / (r + p)) = -8)
    (h2 : (q * r / (p + q) + r * p / (q + r) + p * q / (r + p)) = 9) 
    : (q / (p + q) + r / (q + r) + p / (r + p) = 10) := 
by
  sorry

end NUMINAMATH_GPT_problem_solution_l717_71741


namespace NUMINAMATH_GPT_max_non_managers_l717_71752

theorem max_non_managers (N : ℕ) (h : (9:ℝ) / (N:ℝ) > (7:ℝ) / (32:ℝ)) : N ≤ 41 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_max_non_managers_l717_71752


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_of_quadratic_l717_71766

theorem sum_of_squares_of_roots_of_quadratic :
  (∀ (s₁ s₂ : ℝ), (s₁ + s₂ = 15) → (s₁ * s₂ = 6) → (s₁^2 + s₂^2 = 213)) :=
by
  intros s₁ s₂ h_sum h_prod
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_of_quadratic_l717_71766


namespace NUMINAMATH_GPT_mixed_fraction_product_l717_71708

theorem mixed_fraction_product (X Y : ℕ) (hX : X ≠ 0) (hY : Y ≠ 0) :
  (5 + (1 / X : ℚ)) * (Y + (1 / 2 : ℚ)) = 43 ↔ X = 17 ∧ Y = 8 := 
by 
  sorry

end NUMINAMATH_GPT_mixed_fraction_product_l717_71708


namespace NUMINAMATH_GPT_no_possible_k_l717_71739
open Classical

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem no_possible_k : 
  ∀ (k : ℕ), 
    (∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ (p + q = 74) ∧ (x^2 - 74*x + k = 0)) -> False :=
by sorry

end NUMINAMATH_GPT_no_possible_k_l717_71739


namespace NUMINAMATH_GPT_arithmetic_sequence_l717_71767

-- Given conditions
variables {a x b : ℝ}

-- Statement of the problem in Lean 4
theorem arithmetic_sequence (h1 : x - a = b - x) (h2 : b - x = 2 * x - b) : a / b = 1 / 3 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_l717_71767


namespace NUMINAMATH_GPT_intersection_A_complement_B_eq_interval_l717_71729

-- We define universal set U as ℝ
def U := Set ℝ

-- Definitions provided in the problem
def A : Set ℝ := { x | x > 1 }
def B : Set ℝ := { y | y >= 2 }

-- Complement of B in U
def C_U_B : Set ℝ := { y | y < 2 }

-- Now we state the theorem
theorem intersection_A_complement_B_eq_interval :
  A ∩ C_U_B = { x | 1 < x ∧ x < 2 } :=
by 
  sorry

end NUMINAMATH_GPT_intersection_A_complement_B_eq_interval_l717_71729


namespace NUMINAMATH_GPT_remainder_div_x_plus_1_l717_71749

noncomputable def f (x : ℝ) : ℝ := x^8 + 3

theorem remainder_div_x_plus_1 : 
  (f (-1) = 4) := 
by
  sorry

end NUMINAMATH_GPT_remainder_div_x_plus_1_l717_71749


namespace NUMINAMATH_GPT_isosceles_triangle_vertex_angle_l717_71727

theorem isosceles_triangle_vertex_angle (α : ℝ) (β : ℝ) (γ : ℝ)
  (h1 : α = β)
  (h2: α = 70) 
  (h3 : α + β + γ = 180) : 
  γ = 40 :=
by {
  sorry
}

end NUMINAMATH_GPT_isosceles_triangle_vertex_angle_l717_71727
