import Mathlib

namespace NUMINAMATH_GPT_find_number_l1986_198684

theorem find_number (x : ℤ) (h : 16 * x = 32) : x = 2 :=
sorry

end NUMINAMATH_GPT_find_number_l1986_198684


namespace NUMINAMATH_GPT_correct_option_is_C_l1986_198612

namespace ExponentProof

-- Definitions of conditions
def optionA (a : ℝ) : Prop := a^3 * a^4 = a^12
def optionB (a : ℝ) : Prop := a^3 + a^4 = a^7
def optionC (a : ℝ) : Prop := a^5 / a^3 = a^2
def optionD (a : ℝ) : Prop := (-2 * a)^3 = -6 * a^3

-- Proof problem stating that optionC is the only correct one
theorem correct_option_is_C : ∀ (a : ℝ), ¬ optionA a ∧ ¬ optionB a ∧ optionC a ∧ ¬ optionD a :=
by
  intro a
  sorry

end ExponentProof

end NUMINAMATH_GPT_correct_option_is_C_l1986_198612


namespace NUMINAMATH_GPT_average_speed_for_trip_l1986_198698

theorem average_speed_for_trip (t₁ t₂ : ℝ) (v₁ v₂ : ℝ) (total_time : ℝ) 
  (h₁ : t₁ = 6) 
  (h₂ : v₁ = 30) 
  (h₃ : t₂ = 2) 
  (h₄ : v₂ = 46) 
  (h₅ : total_time = t₁ + t₂) 
  (h₆ : total_time = 8) :
  ((v₁ * t₁ + v₂ * t₂) / total_time) = 34 := 
  by 
    sorry

end NUMINAMATH_GPT_average_speed_for_trip_l1986_198698


namespace NUMINAMATH_GPT_smallest_num_rectangles_l1986_198669

theorem smallest_num_rectangles (a b : ℕ) (h_a : a = 3) (h_b : b = 4) : 
  ∃ n : ℕ, n = 12 ∧ ∀ s : ℕ, (s = lcm a b) → s^2 / (a * b) = 12 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_num_rectangles_l1986_198669


namespace NUMINAMATH_GPT_base3_last_two_digits_l1986_198654

open Nat

theorem base3_last_two_digits (a b c : ℕ) (h1 : a = 2005) (h2 : b = 2003) (h3 : c = 2004) :
  (2005 ^ (2003 ^ 2004 + 3) % 81) = 11 :=
by
  sorry

end NUMINAMATH_GPT_base3_last_two_digits_l1986_198654


namespace NUMINAMATH_GPT_Geli_pushups_total_l1986_198632

variable (x : ℕ)
variable (total_pushups : ℕ)

theorem Geli_pushups_total (h : 10 + (10 + x) + (10 + 2 * x) = 45) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_Geli_pushups_total_l1986_198632


namespace NUMINAMATH_GPT_arithmetic_sequence_third_term_l1986_198690

theorem arithmetic_sequence_third_term 
    (a d : ℝ) 
    (h1 : a = 2)
    (h2 : (a + d) + (a + 3 * d) = 10) : 
    a + 2 * d = 5 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_third_term_l1986_198690


namespace NUMINAMATH_GPT_proof_problem_l1986_198606

variables {a : ℕ → ℕ} -- sequence a_n is positive integers
variables {b : ℕ → ℕ} -- sequence b_n is integers
variables {q : ℕ} -- ratio for geometric sequence
variables {d : ℕ} -- difference for arithmetic sequence
variables {a1 b1 : ℕ} -- initial terms for the sequences

-- Additional conditions as per the problem statement
def geometric_seq (a : ℕ → ℕ) (a1 q : ℕ) : Prop :=
∀ n : ℕ, a n = a1 * q^(n-1)

def arithmetic_seq (b : ℕ → ℕ) (b1 d : ℕ) : Prop :=
∀ n : ℕ, b n = b1 + (n-1) * d

-- Given conditions
variable (geometric : geometric_seq a a1 q)
variable (arithmetic : arithmetic_seq b b1 d)
variable (equal_term : a 6 = b 7)

-- The proof task
theorem proof_problem : a 3 + a 9 ≥ b 4 + b 10 :=
by sorry

end NUMINAMATH_GPT_proof_problem_l1986_198606


namespace NUMINAMATH_GPT_dozen_Pokemon_cards_per_friend_l1986_198650

theorem dozen_Pokemon_cards_per_friend
  (total_cards : ℕ) (num_friends : ℕ) (cards_per_dozen : ℕ)
  (h1 : total_cards = 432)
  (h2 : num_friends = 4)
  (h3 : cards_per_dozen = 12) :
  (total_cards / num_friends) / cards_per_dozen = 9 := 
sorry

end NUMINAMATH_GPT_dozen_Pokemon_cards_per_friend_l1986_198650


namespace NUMINAMATH_GPT_no_two_distinct_real_roots_l1986_198692

-- Definitions of the conditions and question in Lean 4
theorem no_two_distinct_real_roots (a : ℝ) (h : a ≥ 1) : ¬ ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 2*x1 + a = 0) ∧ (x2^2 - 2*x2 + a = 0) :=
sorry

end NUMINAMATH_GPT_no_two_distinct_real_roots_l1986_198692


namespace NUMINAMATH_GPT_smallest_w_l1986_198648

theorem smallest_w (w : ℕ) (h1 : 2^4 ∣ 1452 * w) (h2 : 3^3 ∣ 1452 * w) (h3 : 13^3 ∣ 1452 * w) : w = 79132 :=
by
  sorry

end NUMINAMATH_GPT_smallest_w_l1986_198648


namespace NUMINAMATH_GPT_determine_digits_l1986_198656

theorem determine_digits (h t u : ℕ) (hu: h > u) (h_subtr: t = h - 5) (unit_result: u = 3) : (h = 9 ∧ t = 4 ∧ u = 3) := by
  sorry

end NUMINAMATH_GPT_determine_digits_l1986_198656


namespace NUMINAMATH_GPT_smallest_positive_integer_n_l1986_198610

theorem smallest_positive_integer_n (n : ℕ) (h : 527 * n ≡ 1083 * n [MOD 30]) : n = 2 :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_n_l1986_198610


namespace NUMINAMATH_GPT_sum_of_three_consecutive_even_numbers_is_162_l1986_198639

theorem sum_of_three_consecutive_even_numbers_is_162 (a b c : ℕ) 
  (h1 : a = 52) 
  (h2 : b = a + 2) 
  (h3 : c = b + 2) : 
  a + b + c = 162 := by
  sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_even_numbers_is_162_l1986_198639


namespace NUMINAMATH_GPT_find_f_2009_l1986_198693

-- Defining the function f and specifying the conditions
variable (f : ℝ → ℝ)
axiom h1 : f 3 = -Real.sqrt 3
axiom h2 : ∀ x : ℝ, f (x + 2) * (1 - f x) = 1 + f x

-- Proving the desired statement
theorem find_f_2009 : f 2009 = 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_find_f_2009_l1986_198693


namespace NUMINAMATH_GPT_gcd_14568_78452_l1986_198668

theorem gcd_14568_78452 : Nat.gcd 14568 78452 = 4 :=
sorry

end NUMINAMATH_GPT_gcd_14568_78452_l1986_198668


namespace NUMINAMATH_GPT_polar_circle_equation_l1986_198615

theorem polar_circle_equation (ρ θ : ℝ) (O pole : ℝ) (eq_line : ρ * Real.cos θ + ρ * Real.sin θ = 2) :
  (∃ ρ, ρ = 2 * Real.cos θ) :=
sorry

end NUMINAMATH_GPT_polar_circle_equation_l1986_198615


namespace NUMINAMATH_GPT_inequality_relations_l1986_198689

noncomputable def a : ℝ := Real.sqrt 3
noncomputable def b : ℝ := Real.sqrt 125 ^ (1 / 6)
noncomputable def c : ℝ := Real.log 7 / Real.log (1 / 6)

theorem inequality_relations :
  c < a ∧ a < b := 
by 
  sorry

end NUMINAMATH_GPT_inequality_relations_l1986_198689


namespace NUMINAMATH_GPT_sum_of_coordinates_l1986_198682

-- Definitions based on conditions
variable (f k : ℝ → ℝ)
variable (h₁ : f 4 = 8)
variable (h₂ : ∀ x, k x = (f x) ^ 3)

-- Statement of the theorem
theorem sum_of_coordinates : 4 + k 4 = 516 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_l1986_198682


namespace NUMINAMATH_GPT_percentage_of_a_added_to_get_x_l1986_198653

variable (a b x m : ℝ) (P : ℝ) (k : ℝ)
variable (h1 : a / b = 4 / 5)
variable (h2 : x = a * (1 + P / 100))
variable (h3 : m = b * 0.2)
variable (h4 : m / x = 0.14285714285714285)

theorem percentage_of_a_added_to_get_x :
  P = 75 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_a_added_to_get_x_l1986_198653


namespace NUMINAMATH_GPT_min_value_expression_l1986_198620

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (v : ℝ), (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z →
    8 * x^4 + 12 * y^4 + 18 * z^4 + 25 / (x * y * z) ≥ v) ∧ v = 30 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1986_198620


namespace NUMINAMATH_GPT_bank_robbery_car_l1986_198674

def car_statement (make color : String) : Prop :=
  (make = "Buick" ∨ color = "blue") ∧
  (make = "Chrysler" ∨ color = "black") ∧
  (make = "Ford" ∨ color ≠ "blue")

theorem bank_robbery_car : ∃ make color : String, car_statement make color ∧ make = "Buick" ∧ color = "black" :=
by
  sorry

end NUMINAMATH_GPT_bank_robbery_car_l1986_198674


namespace NUMINAMATH_GPT_positions_after_347_moves_l1986_198627

-- Define the possible positions for the cat
inductive CatPosition
| top_vertex
| right_upper_vertex
| right_lower_vertex
| left_lower_vertex
| left_upper_vertex

-- Define the possible positions for the mouse
inductive MousePosition
| top_left_edge
| left_upper_vertex
| left_middle_edge
| left_lower_vertex
| bottom_edge
| right_lower_vertex
| right_middle_edge
| right_upper_vertex
| top_right_edge
| top_vertex

-- Define the movement function for the cat
def cat_position_after_moves (moves : Nat) : CatPosition :=
  match moves % 5 with
  | 0 => CatPosition.top_vertex
  | 1 => CatPosition.right_upper_vertex
  | 2 => CatPosition.right_lower_vertex
  | 3 => CatPosition.left_lower_vertex
  | 4 => CatPosition.left_upper_vertex
  | _ => CatPosition.top_vertex  -- This case is unreachable due to % 5

-- Define the movement function for the mouse
def mouse_position_after_moves (moves : Nat) : MousePosition :=
  match moves % 10 with
  | 0 => MousePosition.top_left_edge
  | 1 => MousePosition.left_upper_vertex
  | 2 => MousePosition.left_middle_edge
  | 3 => MousePosition.left_lower_vertex
  | 4 => MousePosition.bottom_edge
  | 5 => MousePosition.right_lower_vertex
  | 6 => MousePosition.right_middle_edge
  | 7 => MousePosition.right_upper_vertex
  | 8 => MousePosition.top_right_edge
  | 9 => MousePosition.top_vertex
  | _ => MousePosition.top_left_edge  -- This case is unreachable due to % 10

-- Prove the positions after 347 moves
theorem positions_after_347_moves :
  cat_position_after_moves 347 = CatPosition.right_upper_vertex ∧
  mouse_position_after_moves 347 = MousePosition.right_middle_edge :=
by
  sorry

end NUMINAMATH_GPT_positions_after_347_moves_l1986_198627


namespace NUMINAMATH_GPT_central_angle_of_sector_l1986_198663

theorem central_angle_of_sector (r A : ℝ) (h₁ : r = 4) (h₂ : A = 4) :
  (1 / 2) * r^2 * (1 / 4) = A :=
by
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_l1986_198663


namespace NUMINAMATH_GPT_top_layer_lamps_l1986_198661

theorem top_layer_lamps (a : ℕ) :
  (a + 2 * a + 4 * a + 8 * a + 16 * a + 32 * a + 64 * a = 381) → a = 3 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_top_layer_lamps_l1986_198661


namespace NUMINAMATH_GPT_count_true_statements_l1986_198688

theorem count_true_statements (a b c d : ℝ) : 
  (∃ (H1 : a ≠ b) (H2 : c ≠ d), a + c = b + d) →
  ((a ≠ b) ∧ (c ≠ d) → a + c ≠ b + d) = false ∧ 
  ((a + c ≠ b + d) → (a ≠ b) ∧ (c ≠ d)) = false ∧ 
  (∃ (H3 : a = b) (H4 : c = d), a + c ≠ b + d) = false ∧ 
  ((a + c = b + d) → (a = b) ∨ (c = d)) = false → 
  number_of_true_statements = 0 := 
by
  sorry

end NUMINAMATH_GPT_count_true_statements_l1986_198688


namespace NUMINAMATH_GPT_average_writing_speed_time_to_write_10000_words_l1986_198642

-- Definitions based on the problem conditions
def total_words : ℕ := 60000
def total_hours : ℝ := 90.5
def writing_speed : ℝ := 663
def words_to_write : ℕ := 10000
def writing_time : ℝ := 15.08

-- Proposition that the average writing speed is 663 words per hour
theorem average_writing_speed :
  (total_words : ℝ) / total_hours = writing_speed :=
sorry

-- Proposition that the time to write 10,000 words at the given average speed is 15.08 hours
theorem time_to_write_10000_words :
  (words_to_write : ℝ) / writing_speed = writing_time :=
sorry

end NUMINAMATH_GPT_average_writing_speed_time_to_write_10000_words_l1986_198642


namespace NUMINAMATH_GPT_product_evaluation_l1986_198629

theorem product_evaluation :
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 :=
by sorry

end NUMINAMATH_GPT_product_evaluation_l1986_198629


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1986_198633

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum {a : ℕ → ℝ}
  (h_geo : is_geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_condition : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) :
  a 3 + a 5 = 5 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1986_198633


namespace NUMINAMATH_GPT_simplify_root_power_l1986_198634

theorem simplify_root_power :
  (7^(1/3))^6 = 49 := by
  sorry

end NUMINAMATH_GPT_simplify_root_power_l1986_198634


namespace NUMINAMATH_GPT_probability_females_not_less_than_males_l1986_198608

noncomputable def prob_female_not_less_than_male : ℚ :=
  let total_students := 5
  let females := 2
  let males := 3
  let total_combinations := Nat.choose total_students 2
  let favorable_combinations := Nat.choose females 2 + females * males
  favorable_combinations / total_combinations

theorem probability_females_not_less_than_males (total_students females males : ℕ) :
  total_students = 5 → females = 2 → males = 3 →
  prob_female_not_less_than_male = 7 / 10 :=
by intros; sorry

end NUMINAMATH_GPT_probability_females_not_less_than_males_l1986_198608


namespace NUMINAMATH_GPT_gcd_of_44_54_74_l1986_198643

theorem gcd_of_44_54_74 : gcd (gcd 44 54) 74 = 2 :=
by
    sorry

end NUMINAMATH_GPT_gcd_of_44_54_74_l1986_198643


namespace NUMINAMATH_GPT_number_of_students_l1986_198651

def total_students (a b : ℕ) : ℕ :=
  a + b

variables (a b : ℕ)

theorem number_of_students (h : 48 * a + 45 * b = 972) : total_students a b = 21 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_l1986_198651


namespace NUMINAMATH_GPT_dad_real_age_l1986_198660

theorem dad_real_age (x : ℝ) (h : (5/7) * x = 35) : x = 49 :=
by
  sorry

end NUMINAMATH_GPT_dad_real_age_l1986_198660


namespace NUMINAMATH_GPT_rectangle_problem_l1986_198676

theorem rectangle_problem (x : ℝ) (h1 : 4 * x = l) (h2 : x + 7 = w) (h3 : l * w = 2 * (2 * l + 2 * w)) : x = 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_rectangle_problem_l1986_198676


namespace NUMINAMATH_GPT_smith_trip_times_same_l1986_198699

theorem smith_trip_times_same (v : ℝ) (hv : v > 0) : 
  let t1 := 80 / v 
  let t2 := 160 / (2 * v) 
  t1 = t2 :=
by
  sorry

end NUMINAMATH_GPT_smith_trip_times_same_l1986_198699


namespace NUMINAMATH_GPT_taylor_one_basket_in_three_tries_l1986_198678

theorem taylor_one_basket_in_three_tries (P_no_make : ℚ) (h : P_no_make = 1/3) : 
  (∃ P_make : ℚ, P_make = 1 - P_no_make ∧ P_make * P_no_make * P_no_make * 3 = 2/9) := 
by
  sorry

end NUMINAMATH_GPT_taylor_one_basket_in_three_tries_l1986_198678


namespace NUMINAMATH_GPT_johns_average_speed_l1986_198631

def start_time := 8 * 60 + 15  -- 8:15 a.m. in minutes
def end_time := 14 * 60 + 45   -- 2:45 p.m. in minutes
def break_start := 12 * 60     -- 12:00 p.m. in minutes
def break_duration := 30       -- 30 minutes
def total_distance := 240      -- Total distance in miles

def total_driving_time : ℕ := 
  (break_start - start_time) + (end_time - (break_start + break_duration))

def average_speed (distance : ℕ) (time : ℕ) : ℕ :=
  distance / (time / 60)  -- converting time from minutes to hours

theorem johns_average_speed :
  average_speed total_distance total_driving_time = 40 :=
by
  sorry

end NUMINAMATH_GPT_johns_average_speed_l1986_198631


namespace NUMINAMATH_GPT_domain_of_log_function_l1986_198637

theorem domain_of_log_function : 
  { x : ℝ | x < 1 ∨ x > 2 } = { x : ℝ | 0 < x^2 - 3 * x + 2 } :=
by sorry

end NUMINAMATH_GPT_domain_of_log_function_l1986_198637


namespace NUMINAMATH_GPT_gcf_60_90_150_l1986_198619

theorem gcf_60_90_150 : Nat.gcd (Nat.gcd 60 90) 150 = 30 :=
by
  sorry

end NUMINAMATH_GPT_gcf_60_90_150_l1986_198619


namespace NUMINAMATH_GPT_num_integers_div_10_or_12_l1986_198649

-- Define the problem in Lean
theorem num_integers_div_10_or_12 (N : ℕ) : (1 ≤ N ∧ N ≤ 2007) ∧ (N % 10 = 0 ∨ N % 12 = 0) ↔ ∃ k, k = 334 := by
  sorry

end NUMINAMATH_GPT_num_integers_div_10_or_12_l1986_198649


namespace NUMINAMATH_GPT_find_number_l1986_198630

theorem find_number (x : ℕ) :
  ((4 * x) / 8 = 6) ∧ ((4 * x) % 8 = 4) → x = 13 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1986_198630


namespace NUMINAMATH_GPT_face_opposite_of_E_l1986_198672

-- Definitions of faces and their relationships
inductive Face : Type
| A | B | C | D | E | F | x

open Face

-- Adjacency relationship
def is_adjacent_to (f1 f2 : Face) : Prop :=
(f1 = x ∧ (f2 = A ∨ f2 = B ∨ f2 = C ∨ f2 = D)) ∨
(f2 = x ∧ (f1 = A ∨ f1 = B ∨ f1 = C ∨ f1 = D)) ∨
(f1 = E ∧ (f2 = A ∨ f2 = B ∨ f2 = C ∨ f2 = D)) ∨
(f2 = E ∧ (f1 = A ∨ f1 = B ∨ f1 = C ∨ f1 = D))

-- Non-adjacency relationship
def is_opposite (f1 f2 : Face) : Prop :=
∀ f : Face, is_adjacent_to f1 f → ¬ is_adjacent_to f2 f

-- Theorem to prove that F is opposite of E
theorem face_opposite_of_E : is_opposite E F :=
sorry

end NUMINAMATH_GPT_face_opposite_of_E_l1986_198672


namespace NUMINAMATH_GPT_solid_views_same_shape_and_size_l1986_198647

theorem solid_views_same_shape_and_size (solid : Type) (sphere triangular_pyramid cube cylinder : solid)
  (views_same_shape_and_size : solid → Bool) : 
  views_same_shape_and_size cylinder = false :=
sorry

end NUMINAMATH_GPT_solid_views_same_shape_and_size_l1986_198647


namespace NUMINAMATH_GPT_cafeteria_pies_l1986_198616

theorem cafeteria_pies (initial_apples handed_out_apples apples_per_pie : ℕ)
  (h_initial : initial_apples = 50)
  (h_handed_out : handed_out_apples = 5)
  (h_apples_per_pie : apples_per_pie = 5) :
  (initial_apples - handed_out_apples) / apples_per_pie = 9 := 
by
  sorry

end NUMINAMATH_GPT_cafeteria_pies_l1986_198616


namespace NUMINAMATH_GPT_max_value_of_a_plus_b_l1986_198683

theorem max_value_of_a_plus_b (a b : ℕ) 
  (h : 5 * a + 19 * b = 213) : a + b ≤ 37 :=
  sorry

end NUMINAMATH_GPT_max_value_of_a_plus_b_l1986_198683


namespace NUMINAMATH_GPT_range_of_m_value_of_m_l1986_198679

variables (m p x : ℝ)

-- Conditions: The quadratic equation x^2 - 2x + m - 1 = 0 must have two real roots.
def discriminant (m : ℝ) := (-2)^2 - 4 * 1 * (m - 1)

-- Part 1: Finding the range of values for m
theorem range_of_m (h : discriminant m ≥ 0) : m ≤ 2 := 
by sorry

-- Additional Condition: p is a real root of the equation x^2 - 2x + m - 1 = 0
def is_root (p m : ℝ) := p^2 - 2 * p + m - 1 = 0

-- Another condition: (p^2 - 2p + 3)(m + 4) = 7
def satisfies_condition (p m : ℝ) := (p^2 - 2 * p + 3) * (m + 4) = 7

-- Part 2: Finding the value of m given p is a real root and satisfies (p^2 - 2p + 3)(m + 4) = 7
theorem value_of_m (h1 : is_root p m) (h2 : satisfies_condition p m) : m = -3 := 
by sorry

end NUMINAMATH_GPT_range_of_m_value_of_m_l1986_198679


namespace NUMINAMATH_GPT_large_block_volume_l1986_198659

theorem large_block_volume (W D L : ℝ) (h1 : W * D * L = 3) : 
  (2 * W) * (2 * D) * (3 * L) = 36 := 
by 
  sorry

end NUMINAMATH_GPT_large_block_volume_l1986_198659


namespace NUMINAMATH_GPT_problem1_problem2_l1986_198618

theorem problem1 (x : ℝ) (h : 4 * x^2 - 9 = 0) : x = 3/2 ∨ x = -3/2 :=
by
  sorry

theorem problem2 (x : ℝ) (h : 64 * (x-2)^3 - 1 = 0) : x = 2 + 1/4 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1986_198618


namespace NUMINAMATH_GPT_temperature_or_daytime_not_sufficiently_high_l1986_198645

variable (T : ℝ) (Daytime Lively : Prop)
axiom h1 : (T ≥ 75 ∧ Daytime) → Lively
axiom h2 : ¬ Lively

theorem temperature_or_daytime_not_sufficiently_high : T < 75 ∨ ¬ Daytime :=
by
  -- proof steps
  sorry

end NUMINAMATH_GPT_temperature_or_daytime_not_sufficiently_high_l1986_198645


namespace NUMINAMATH_GPT_monthly_salary_l1986_198652

variables (S : ℝ) (savings : ℝ) (new_expenses : ℝ)

theorem monthly_salary (h1 : savings = 0.20 * S)
                      (h2 : new_expenses = 0.96 * S)
                      (h3 : S = 200 + new_expenses) :
                      S = 5000 :=
by
  sorry

end NUMINAMATH_GPT_monthly_salary_l1986_198652


namespace NUMINAMATH_GPT_value_of_a_l1986_198680

noncomputable def M : Set ℝ := {x | x^2 = 2}
noncomputable def N (a : ℝ) : Set ℝ := {x | a*x = 1}

theorem value_of_a (a : ℝ) : N a ⊆ M → a = 0 ∨ a = -Real.sqrt 2 / 2 ∨ a = Real.sqrt 2 / 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_value_of_a_l1986_198680


namespace NUMINAMATH_GPT_find_fixed_point_l1986_198600

theorem find_fixed_point (c d k : ℝ) 
(h : ∀ k : ℝ, d = 5 * c^2 + k * c - 3 * k) : (c, d) = (3, 45) :=
sorry

end NUMINAMATH_GPT_find_fixed_point_l1986_198600


namespace NUMINAMATH_GPT_integer_solution_of_inequality_system_l1986_198611

theorem integer_solution_of_inequality_system :
  ∃ x : ℤ, (2 * (x : ℝ) ≤ 1) ∧ ((x : ℝ) + 2 > 1) ∧ (x = 0) :=
by
  sorry

end NUMINAMATH_GPT_integer_solution_of_inequality_system_l1986_198611


namespace NUMINAMATH_GPT_negation_of_proposition_l1986_198622

namespace NegationProp

theorem negation_of_proposition :
  (∀ x : ℝ, 0 < x ∧ x < 1 → x^2 - x < 0) ↔
  (∃ x0 : ℝ, 0 < x0 ∧ x0 < 1 ∧ x0^2 - x0 ≥ 0) := by sorry

end NegationProp

end NUMINAMATH_GPT_negation_of_proposition_l1986_198622


namespace NUMINAMATH_GPT_greatest_integer_third_side_of_triangle_l1986_198636

theorem greatest_integer_third_side_of_triangle (x : ℕ) (h1 : 7 + 10 > x) (h2 : x > 3) : x = 16 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_third_side_of_triangle_l1986_198636


namespace NUMINAMATH_GPT_sum_first_8_terms_arithmetic_sequence_l1986_198624

theorem sum_first_8_terms_arithmetic_sequence (a : ℕ → ℝ) (h : a 4 + a 5 = 12) :
    (8 * (a 1 + a 8)) / 2 = 48 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_8_terms_arithmetic_sequence_l1986_198624


namespace NUMINAMATH_GPT_least_number_of_homeowners_l1986_198686

theorem least_number_of_homeowners (total_members : ℕ) 
(num_men : ℕ) (num_women : ℕ) 
(homeowners_men : ℕ) (homeowners_women : ℕ) 
(h_total : total_members = 5000)
(h_men_women : num_men + num_women = total_members) 
(h_percentage_men : homeowners_men = 15 * num_men / 100)
(h_percentage_women : homeowners_women = 25 * num_women / 100):
  homeowners_men + homeowners_women = 4 :=
sorry

end NUMINAMATH_GPT_least_number_of_homeowners_l1986_198686


namespace NUMINAMATH_GPT_width_of_metallic_sheet_is_36_l1986_198646

-- Given conditions
def length_of_metallic_sheet : ℕ := 48
def side_length_of_cutoff_square : ℕ := 8
def volume_of_box : ℕ := 5120

-- Proof statement
theorem width_of_metallic_sheet_is_36 :
  ∀ (w : ℕ), w - 2 * side_length_of_cutoff_square = 36 - 16 →  length_of_metallic_sheet - 2* side_length_of_cutoff_square = 32  →  5120 = 256 * (w - 16)  := sorry

end NUMINAMATH_GPT_width_of_metallic_sheet_is_36_l1986_198646


namespace NUMINAMATH_GPT_part_I_part_II_l1986_198691

-- Condition definitions:
def f (x : ℝ) (m : ℝ) : ℝ := m - |x - 2|

-- Part I: Prove m = 1
theorem part_I (m : ℝ) : (∀ x : ℝ, f (x + 2) m ≥ 0) ↔ m = 1 :=
by
  sorry

-- Part II: Prove a + 2b + 3c ≥ 9
theorem part_II (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) : a + 2 * b + 3 * c ≥ 9 :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l1986_198691


namespace NUMINAMATH_GPT_lemonade_water_l1986_198677

theorem lemonade_water (L S W : ℝ) (h1 : S = 1.5 * L) (h2 : W = 3 * S) (h3 : L = 4) : W = 18 :=
by
  sorry

end NUMINAMATH_GPT_lemonade_water_l1986_198677


namespace NUMINAMATH_GPT_waiter_customers_l1986_198602

theorem waiter_customers
    (initial_tables : ℝ)
    (left_tables : ℝ)
    (customers_per_table : ℝ)
    (remaining_tables : ℝ) 
    (total_customers : ℝ) 
    (h1 : initial_tables = 44.0)
    (h2 : left_tables = 12.0)
    (h3 : customers_per_table = 8.0)
    (remaining_tables_def : remaining_tables = initial_tables - left_tables)
    (total_customers_def : total_customers = remaining_tables * customers_per_table) :
    total_customers = 256.0 :=
by
  sorry

end NUMINAMATH_GPT_waiter_customers_l1986_198602


namespace NUMINAMATH_GPT_initial_men_colouring_l1986_198657

theorem initial_men_colouring (M : ℕ) : 
  (∀ m : ℕ, ∀ d : ℕ, ∀ l : ℕ, m * d = 48 * 2 → 8 * 0.75 = 6 → M = 4) :=
by
  sorry

end NUMINAMATH_GPT_initial_men_colouring_l1986_198657


namespace NUMINAMATH_GPT_mark_min_correct_problems_l1986_198670

noncomputable def mark_score (x : ℕ) : ℤ :=
  8 * x - 21

theorem mark_min_correct_problems (x : ℕ) :
  (4 * 2) + mark_score x ≥ 120 ↔ x ≥ 17 :=
by
  sorry

end NUMINAMATH_GPT_mark_min_correct_problems_l1986_198670


namespace NUMINAMATH_GPT_bella_grazing_area_l1986_198626

open Real

theorem bella_grazing_area:
  let leash_length := 5
  let barn_width := 4
  let barn_height := 6
  let sector_fraction := 3 / 4
  let area_circle := π * leash_length^2
  let grazed_area := sector_fraction * area_circle
  grazed_area = 75 / 4 * π := 
by
  sorry

end NUMINAMATH_GPT_bella_grazing_area_l1986_198626


namespace NUMINAMATH_GPT_axis_of_symmetry_l1986_198694

-- Given conditions
variables {b c : ℝ}
axiom eq_roots : ∃ (x1 x2 : ℝ), (x1 = -1 ∧ x2 = 2) ∧ (x1 + x2 = -b) ∧ (x1 * x2 = c)

-- Question translation to Lean statement
theorem axis_of_symmetry : 
  ∀ b c, 
  (∃ (x1 x2 : ℝ), x1 = -1 ∧ x2 = 2 ∧ x1 + x2 = -b ∧ x1 * x2 = c) 
  → -b / 2 = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_l1986_198694


namespace NUMINAMATH_GPT_largest_integer_odd_divides_expression_l1986_198614

theorem largest_integer_odd_divides_expression (x : ℕ) (h_odd : x % 2 = 1) : 
    ∃ k, k = 384 ∧ ∀ m, m ∣ (8*x + 6) * (8*x + 10) * (4*x + 4) → m ≤ k :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_integer_odd_divides_expression_l1986_198614


namespace NUMINAMATH_GPT_ratio_of_arithmetic_sequence_sums_l1986_198671

theorem ratio_of_arithmetic_sequence_sums :
  let a1 := 2
  let d1 := 3
  let l1 := 41
  let n1 := (l1 - a1) / d1 + 1
  let sum1 := n1 / 2 * (a1 + l1)

  let a2 := 4
  let d2 := 4
  let l2 := 60
  let n2 := (l2 - a2) / d2 + 1
  let sum2 := n2 / 2 * (a2 + l2)
  sum1 / sum2 = 301 / 480 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_arithmetic_sequence_sums_l1986_198671


namespace NUMINAMATH_GPT_probability_all_from_same_tribe_l1986_198641

-- Definitions based on the conditions of the problem
def total_people := 24
def tribe_count := 3
def people_per_tribe := 8
def quitters := 3

-- We assume each person has an equal chance of quitting and the quitters are chosen independently
-- The probability that all three people who quit belong to the same tribe

theorem probability_all_from_same_tribe :
  ((3 * (Nat.choose people_per_tribe quitters)) / (Nat.choose total_people quitters) : ℚ) = 1 / 12 := 
  by 
    sorry

end NUMINAMATH_GPT_probability_all_from_same_tribe_l1986_198641


namespace NUMINAMATH_GPT_Indians_drink_tea_is_zero_l1986_198662

-- Definitions based on given conditions and questions
variable (total_people : Nat)
variable (total_drink_tea : Nat)
variable (total_drink_coffee : Nat)
variable (answer_do_you_drink_coffee : Nat)
variable (answer_are_you_a_turk : Nat)
variable (answer_is_it_raining : Nat)
variable (Indians_drink_tea : Nat)
variable (Indians_drink_coffee : Nat)
variable (Turks_drink_coffee : Nat)
variable (Turks_drink_tea : Nat)

-- The given facts and conditions
axiom hx1 : total_people = 55
axiom hx2 : answer_do_you_drink_coffee = 44
axiom hx3 : answer_are_you_a_turk = 33
axiom hx4 : answer_is_it_raining = 22
axiom hx5 : Indians_drink_tea + Indians_drink_coffee + Turks_drink_coffee + Turks_drink_tea = total_people
axiom hx6 : Indians_drink_coffee + Turks_drink_coffee = answer_do_you_drink_coffee
axiom hx7 : Indians_drink_coffee + Turks_drink_tea = answer_are_you_a_turk
axiom hx8 : Indians_drink_tea + Turks_drink_coffee = answer_is_it_raining

-- Prove that the number of Indians drinking tea is 0
theorem Indians_drink_tea_is_zero : Indians_drink_tea = 0 :=
by {
    sorry
}

end NUMINAMATH_GPT_Indians_drink_tea_is_zero_l1986_198662


namespace NUMINAMATH_GPT_meaningful_expression_range_l1986_198644

theorem meaningful_expression_range (x : ℝ) :
  (2 - x ≥ 0) ∧ (x - 2 ≠ 0) → x < 2 :=
by
  sorry

end NUMINAMATH_GPT_meaningful_expression_range_l1986_198644


namespace NUMINAMATH_GPT_tangent_line_eqn_at_one_l1986_198667

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem tangent_line_eqn_at_one :
  let k := (Real.exp 1)
  let p := (1, Real.exp 1)
  ∃ m b : ℝ, (m = k) ∧ (b = p.2 - m * p.1) ∧ (∀ x, f x = y → y = m * x + b) :=
sorry

end NUMINAMATH_GPT_tangent_line_eqn_at_one_l1986_198667


namespace NUMINAMATH_GPT_additional_pairs_of_snakes_l1986_198623

theorem additional_pairs_of_snakes (total_snakes breeding_balls snakes_per_ball additional_snakes_per_pair : ℕ)
  (h1 : total_snakes = 36) 
  (h2 : breeding_balls = 3)
  (h3 : snakes_per_ball = 8) 
  (h4 : additional_snakes_per_pair = 2) :
  (total_snakes - (breeding_balls * snakes_per_ball)) / additional_snakes_per_pair = 6 :=
by
  sorry

end NUMINAMATH_GPT_additional_pairs_of_snakes_l1986_198623


namespace NUMINAMATH_GPT_probability_heads_mod_coin_l1986_198664

theorem probability_heads_mod_coin (p : ℝ) (h : 20 * p ^ 3 * (1 - p) ^ 3 = 1 / 20) : p = (1 - Real.sqrt 0.6816) / 2 :=
by
  sorry

end NUMINAMATH_GPT_probability_heads_mod_coin_l1986_198664


namespace NUMINAMATH_GPT_larger_number_is_8_l1986_198697

-- Define the conditions
def is_twice (x y : ℕ) : Prop := x = 2 * y
def product_is_40 (x y : ℕ) : Prop := x * y = 40
def sum_is_14 (x y : ℕ) : Prop := x + y = 14

-- The proof statement
theorem larger_number_is_8 (x y : ℕ) (h1 : is_twice x y) (h2 : product_is_40 x y) (h3 : sum_is_14 x y) : x = 8 :=
  sorry

end NUMINAMATH_GPT_larger_number_is_8_l1986_198697


namespace NUMINAMATH_GPT_fraction_of_bag_spent_on_lunch_l1986_198604

-- Definitions of conditions based on the problem
def initial_amount : ℕ := 158
def price_of_shoes : ℕ := 45
def price_of_bag : ℕ := price_of_shoes - 17
def amount_left : ℕ := 78
def money_before_lunch := amount_left + price_of_shoes + price_of_bag
def money_spent_on_lunch := initial_amount - money_before_lunch 

-- Statement of the problem in Lean
theorem fraction_of_bag_spent_on_lunch :
  (money_spent_on_lunch : ℚ) / price_of_bag = 1 / 4 :=
by
  -- Conditions decoded to match the solution provided
  have h1 : price_of_bag = 28 := by sorry
  have h2 : money_before_lunch = 151 := by sorry
  have h3 : money_spent_on_lunch = 7 := by sorry
  -- The main theorem statement
  exact sorry

end NUMINAMATH_GPT_fraction_of_bag_spent_on_lunch_l1986_198604


namespace NUMINAMATH_GPT_min_sum_of_bases_l1986_198696

theorem min_sum_of_bases (a b : ℕ) (h : 3 * a + 5 = 4 * b + 2) : a + b = 13 :=
sorry

end NUMINAMATH_GPT_min_sum_of_bases_l1986_198696


namespace NUMINAMATH_GPT_total_cups_of_ingredients_l1986_198628

theorem total_cups_of_ingredients
  (ratio_butter : ℕ) (ratio_flour : ℕ) (ratio_sugar : ℕ)
  (flour_cups : ℕ)
  (h_ratio : ratio_butter = 2 ∧ ratio_flour = 3 ∧ ratio_sugar = 5)
  (h_flour : flour_cups = 6) :
  let part_cups := flour_cups / ratio_flour
  let butter_cups := ratio_butter * part_cups
  let sugar_cups := ratio_sugar * part_cups
  let total_cups := butter_cups + flour_cups + sugar_cups
  total_cups = 20 :=
by
  sorry

end NUMINAMATH_GPT_total_cups_of_ingredients_l1986_198628


namespace NUMINAMATH_GPT_find_number_l1986_198673

theorem find_number
  (x : ℝ)
  (h : 0.90 * x = 0.50 * 1080) :
  x = 600 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1986_198673


namespace NUMINAMATH_GPT_initial_girls_count_l1986_198603

variable (p : ℕ) -- total number of people initially in the group
variable (girls_initial : ℕ) -- number of girls initially in the group
variable (girls_after : ℕ) -- number of girls after the change
variable (total_after : ℕ) -- total number of people after the change

/--
Initially, 50% of the group are girls. 
Later, five girls leave and five boys arrive, leading to 40% of the group now being girls.
--/
theorem initial_girls_count :
  (girls_initial = p / 2) →
  (total_after = p) →
  (girls_after = girls_initial - 5) →
  (girls_after = 2 * total_after / 5) →
  girls_initial = 25 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_initial_girls_count_l1986_198603


namespace NUMINAMATH_GPT_beads_counter_representation_l1986_198640

-- Given conditions
variable (a : ℕ) -- a is a natural number representing the beads in the tens place.
variable (h : a ≥ 0) -- Ensure a is non-negative since the number of beads cannot be negative.

-- The main statement to prove
theorem beads_counter_representation (a : ℕ) (h : a ≥ 0) : 10 * a + 4 = (10 * a) + 4 :=
by sorry

end NUMINAMATH_GPT_beads_counter_representation_l1986_198640


namespace NUMINAMATH_GPT_percent_brandA_in_mix_l1986_198695

theorem percent_brandA_in_mix (x : Real) :
  (0.60 * x + 0.35 * (100 - x) = 50) → x = 60 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_percent_brandA_in_mix_l1986_198695


namespace NUMINAMATH_GPT_solution_l1986_198687

noncomputable def polynomial (x m : ℝ) := 3 * x^2 - 5 * x + m

theorem solution (m : ℝ) : (∃ a : ℝ, a = 2 ∧ polynomial a m = 0) -> m = -2 := by
  sorry

end NUMINAMATH_GPT_solution_l1986_198687


namespace NUMINAMATH_GPT_find_number_l1986_198638

theorem find_number (x : ℤ) (h : (7 * (x + 10) / 5) - 5 = 44) : x = 25 :=
sorry

end NUMINAMATH_GPT_find_number_l1986_198638


namespace NUMINAMATH_GPT_weight_gain_difference_l1986_198655

theorem weight_gain_difference :
  let orlando_gain := 5
  let jose_gain := 2 * orlando_gain + 2
  let total_gain := 20
  let fernando_gain := total_gain - (orlando_gain + jose_gain)
  let half_jose_gain := jose_gain / 2
  half_jose_gain - fernando_gain = 3 :=
by
  sorry

end NUMINAMATH_GPT_weight_gain_difference_l1986_198655


namespace NUMINAMATH_GPT_sum_of_squares_inequality_l1986_198609

theorem sum_of_squares_inequality (a b c : ℝ) (h : a + 2 * b + 3 * c = 4) : a^2 + b^2 + c^2 ≥ 8 / 7 := by
  sorry

end NUMINAMATH_GPT_sum_of_squares_inequality_l1986_198609


namespace NUMINAMATH_GPT_polynomial_transformation_l1986_198601

variable {x y : ℝ}

theorem polynomial_transformation
  (h : y = x + 1/x) 
  (poly_eq_0 : x^4 + x^3 - 5*x^2 + x + 1 = 0) :
  x^2 * (y^2 + y - 7) = 0 :=
sorry

end NUMINAMATH_GPT_polynomial_transformation_l1986_198601


namespace NUMINAMATH_GPT_walking_time_proof_l1986_198685

-- Define the conditions from the problem
def bus_ride : ℕ := 75
def train_ride : ℕ := 360
def total_trip_time : ℕ := 480

-- Define the walking time as variable
variable (W : ℕ)

-- State the theorem as a Lean statement
theorem walking_time_proof :
  bus_ride + W + 2 * W + train_ride = total_trip_time → W = 15 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_walking_time_proof_l1986_198685


namespace NUMINAMATH_GPT_infinite_consecutive_pairs_l1986_198621

-- Define the relation
def related (x y : ℕ) : Prop :=
  ∀ d ∈ (Nat.digits 10 (x + y)), d = 0 ∨ d = 1

-- Define sets A and B
variable (A B : Set ℕ)

-- Define the conditions
axiom cond1 : ∀ a ∈ A, ∀ b ∈ B, related a b
axiom cond2 : ∀ c, (∀ a ∈ A, related c a) → c ∈ B
axiom cond3 : ∀ c, (∀ b ∈ B, related c b) → c ∈ A

-- Prove that one of the sets contains infinitely many pairs of consecutive numbers
theorem infinite_consecutive_pairs :
  (∃ a ∈ A, ∀ n : ℕ, a + n ∈ A ∧ a + n + 1 ∈ A) ∨ (∃ b ∈ B, ∀ n : ℕ, b + n ∈ B ∧ b + n + 1 ∈ B) :=
sorry

end NUMINAMATH_GPT_infinite_consecutive_pairs_l1986_198621


namespace NUMINAMATH_GPT_equivalence_of_expression_l1986_198613

theorem equivalence_of_expression (x y : ℝ) :
  ( (x^2 + y^2 + xy) / (x^2 + y^2 - xy) ) - ( (x^2 + y^2 - xy) / (x^2 + y^2 + xy) ) =
  ( 4 * xy * (x^2 + y^2) ) / ( x^4 + y^4 ) :=
by sorry

end NUMINAMATH_GPT_equivalence_of_expression_l1986_198613


namespace NUMINAMATH_GPT_pyramid_x_value_l1986_198681

theorem pyramid_x_value (x y : ℝ) 
  (h1 : 150 = 10 * x)
  (h2 : 225 = x * 15)
  (h3 : 1800 = 150 * y * 225) :
  x = 15 :=
sorry

end NUMINAMATH_GPT_pyramid_x_value_l1986_198681


namespace NUMINAMATH_GPT_circle_radius_zero_l1986_198625

-- Theorem statement
theorem circle_radius_zero :
  ∀ (x y : ℝ), 4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0 → 
  ∃ (c : ℝ) (r : ℝ), r = 0 ∧ (x - 1)^2 + (y + 2)^2 = r^2 :=
by sorry

end NUMINAMATH_GPT_circle_radius_zero_l1986_198625


namespace NUMINAMATH_GPT_regular_hexagon_area_l1986_198658

theorem regular_hexagon_area (A : ℝ) (r : ℝ) (hex_area : ℝ) :
  A = 100 * Real.pi → r = Real.sqrt 100 → 
  hex_area = 150 * Real.sqrt 3 → 
  150 * Real.sqrt 3 = 150 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_regular_hexagon_area_l1986_198658


namespace NUMINAMATH_GPT_original_combined_price_l1986_198617

theorem original_combined_price (C S : ℝ) 
  (candy_box_increased : C * 1.25 = 18.75)
  (soda_can_increased : S * 1.50 = 9) : 
  C + S = 21 :=
by
  sorry

end NUMINAMATH_GPT_original_combined_price_l1986_198617


namespace NUMINAMATH_GPT_nat_exponent_sum_eq_l1986_198607

theorem nat_exponent_sum_eq (n p q : ℕ) : n^p + n^q = n^2010 ↔ (n = 2 ∧ p = 2009 ∧ q = 2009) :=
by
  sorry

end NUMINAMATH_GPT_nat_exponent_sum_eq_l1986_198607


namespace NUMINAMATH_GPT_fraction_of_jenny_bounce_distance_l1986_198665

-- Definitions for the problem conditions
def jenny_initial_distance := 18
def jenny_bounce_fraction (f : ℚ) : ℚ := 18 * f
def jenny_total_distance (f : ℚ) : ℚ := jenny_initial_distance + jenny_bounce_fraction f

def mark_initial_distance := 15
def mark_bounce_distance := 2 * mark_initial_distance
def mark_total_distance : ℚ := mark_initial_distance + mark_bounce_distance

def distance_difference := 21

-- The theorem to prove
theorem fraction_of_jenny_bounce_distance (f : ℚ) :
  mark_total_distance = jenny_total_distance f + distance_difference →
  f = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_jenny_bounce_distance_l1986_198665


namespace NUMINAMATH_GPT_how_many_bigger_panda_bears_l1986_198635

-- Definitions for the conditions
def four_small_panda_bears_eat_daily : ℕ := 25
def one_small_panda_bear_eats_daily : ℚ := 25 / 4
def each_bigger_panda_bear_eats_daily : ℚ := 40
def total_bamboo_eaten_weekly : ℕ := 2100
def total_bamboo_eaten_daily : ℚ := 2100 / 7

-- The theorem statement to prove
theorem how_many_bigger_panda_bears :
  ∃ B : ℚ, one_small_panda_bear_eats_daily * 4 + each_bigger_panda_bear_eats_daily * B = total_bamboo_eaten_daily := by
  sorry

end NUMINAMATH_GPT_how_many_bigger_panda_bears_l1986_198635


namespace NUMINAMATH_GPT_quadratic_pairs_square_diff_exists_l1986_198666

open Nat Polynomial

theorem quadratic_pairs_square_diff_exists (P : Polynomial ℤ) (u v w a b n : ℤ) (n_pos : 0 < n)
    (hp : ∃ (u v w : ℤ), P = C u * X ^ 2 + C v * X + C w)
    (h_ab : P.eval a - P.eval b = n^2) : ∃ k > 10^6, ∃ m : ℕ, ∃ c d : ℤ, (c - d = a - b + 2 * k) ∧ 
    (P.eval c - P.eval d = n^2 * m ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_pairs_square_diff_exists_l1986_198666


namespace NUMINAMATH_GPT_smallest_rel_prime_120_l1986_198675

theorem smallest_rel_prime_120 : ∃ (x : ℕ), x > 1 ∧ Nat.gcd x 120 = 1 ∧ ∀ y, y > 1 ∧ Nat.gcd y 120 = 1 → x ≤ y :=
by
  use 7
  sorry

end NUMINAMATH_GPT_smallest_rel_prime_120_l1986_198675


namespace NUMINAMATH_GPT_sqrt_sum_simplify_l1986_198605

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_sqrt_sum_simplify_l1986_198605
