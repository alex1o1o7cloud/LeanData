import Mathlib

namespace smallest_portion_bread_l264_26477

theorem smallest_portion_bread (a d : ℚ) (h1 : 5 * a = 100) (h2 : 24 * d = 11 * a) :
  a - 2 * d = 5 / 3 :=
by
  -- Solution proof goes here...
  sorry -- placeholder for the proof

end smallest_portion_bread_l264_26477


namespace white_balls_probability_l264_26410

noncomputable def probability_all_white (total_balls white_balls draw_count : ℕ) : ℚ :=
  if h : total_balls >= draw_count ∧ white_balls >= draw_count then
    (Nat.choose white_balls draw_count : ℚ) / (Nat.choose total_balls draw_count : ℚ)
  else
    0

theorem white_balls_probability :
  probability_all_white 11 5 5 = 1 / 462 :=
by
  sorry

end white_balls_probability_l264_26410


namespace problem_proof_l264_26463

def mixed_to_improper (a b c : ℚ) : ℚ := a + b / c

noncomputable def evaluate_expression : ℚ :=
  100 - (mixed_to_improper 3 1 8) / (mixed_to_improper 2 1 12 - 5 / 8) * (8 / 5 + mixed_to_improper 2 2 3)

theorem problem_proof : evaluate_expression = 636 / 7 := 
  sorry

end problem_proof_l264_26463


namespace find_m_l264_26483

theorem find_m (a b c m : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_m : 0 < m) (h : a * b * c * m = 1 + a^2 + b^2 + c^2) : 
  m = 4 :=
sorry

end find_m_l264_26483


namespace least_subtract_divisible_by_8_l264_26480

def least_subtracted_to_divisible_by (n : ℕ) (d : ℕ) : ℕ :=
  n % d

theorem least_subtract_divisible_by_8 (n : ℕ) (d : ℕ) (h : n = 964807) (h_d : d = 8) :
  least_subtracted_to_divisible_by n d = 7 :=
by
  sorry

end least_subtract_divisible_by_8_l264_26480


namespace negation_of_exists_l264_26409

open Classical

theorem negation_of_exists (p : Prop) : 
  (∃ x : ℝ, 2^x ≥ 2 * x + 1) ↔ ¬ ∀ x : ℝ, 2^x < 2 * x + 1 :=
by
  sorry

end negation_of_exists_l264_26409


namespace x_intercept_of_line_l264_26429

theorem x_intercept_of_line :
  (∃ x : ℝ, 5 * x - 7 * 0 = 35 ∧ (x, 0) = (7, 0)) :=
by
  use 7
  simp
  sorry

end x_intercept_of_line_l264_26429


namespace proof_problem_l264_26486

variable (a b c d : ℝ)
variable (ω : ℂ)

-- Conditions
def conditions : Prop :=
  a ≠ -1 ∧ b ≠ -1 ∧ c ≠ -1 ∧ d ≠ -1 ∧
  ω^4 = 1 ∧ ω ≠ 1 ∧
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 4 / ω^2)

theorem proof_problem (h : conditions a b c d ω) :
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1)) = 2 := 
sorry

end proof_problem_l264_26486


namespace find_m_l264_26438

theorem find_m (n m : ℕ) (h1 : m = 13 * n + 8) (h2 : m = 15 * n) : m = 60 :=
  sorry

end find_m_l264_26438


namespace barbara_total_candies_l264_26411

theorem barbara_total_candies :
  let boxes1 := 9
  let candies_per_box1 := 25
  let boxes2 := 18
  let candies_per_box2 := 35
  boxes1 * candies_per_box1 + boxes2 * candies_per_box2 = 855 := 
by
  let boxes1 := 9
  let candies_per_box1 := 25
  let boxes2 := 18
  let candies_per_box2 := 35
  show boxes1 * candies_per_box1 + boxes2 * candies_per_box2 = 855
  sorry

end barbara_total_candies_l264_26411


namespace divisor_is_ten_l264_26493

variable (x y : ℝ)

theorem divisor_is_ten
  (h : ((5 * x - x / y) / (5 * x)) * 100 = 98) : y = 10 := by
  sorry

end divisor_is_ten_l264_26493


namespace median_eq_range_le_l264_26428

variables (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ)
-- x₁ is the minimum value
-- x₆ is the maximum value
-- Assume x₁ ≤ x₂ ≤ x₃ ≤ x₄ ≤ x₅ ≤ x₆

theorem median_eq : (x₃ + x₄) / 2 = (x₃ + x₄) / 2 := 
by sorry

theorem range_le : (x₅ - x₂) ≤ (x₆ - x₁) := 
by sorry

end median_eq_range_le_l264_26428


namespace factorize_expression_l264_26405

variable (a x y : ℝ)

theorem factorize_expression : a^2 * (x - y) + 9 * (y - x) = (x - y) * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l264_26405


namespace sequences_count_l264_26491

open BigOperators

def consecutive_blocks (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2 - 1) - 2

theorem sequences_count {n : ℕ} (h : n = 15) :
  consecutive_blocks n = 238 :=
by
  sorry

end sequences_count_l264_26491


namespace batsman_sixes_l264_26422

theorem batsman_sixes 
(scorer_runs : ℕ)
(boundaries : ℕ)
(run_contrib : ℕ → ℚ)
(score_by_boundary : ℕ)
(score : ℕ)
(h1 : scorer_runs = 125)
(h2 : boundaries = 5)
(h3 : ∀ (x : ℕ), run_contrib x = (0.60 * scorer_runs : ℚ))
(h4 : score_by_boundary = boundaries * 4)
(h5 : score = scorer_runs - score_by_boundary) : 
∃ (x : ℕ), x = 5 ∧ (scorer_runs = score + (x * 6)) :=
by
  sorry

end batsman_sixes_l264_26422


namespace balls_per_bag_l264_26412

theorem balls_per_bag (total_balls : ℕ) (total_bags : ℕ) (h1 : total_balls = 36) (h2 : total_bags = 9) : total_balls / total_bags = 4 :=
by
  sorry

end balls_per_bag_l264_26412


namespace Leah_coins_value_in_cents_l264_26435

theorem Leah_coins_value_in_cents (p n : ℕ) (h₁ : p + n = 15) (h₂ : p = n + 2) : p + 5 * n = 44 :=
by
  sorry

end Leah_coins_value_in_cents_l264_26435


namespace cyclist_speed_ratio_l264_26497

theorem cyclist_speed_ratio
  (d : ℝ) (t₁ t₂ : ℝ) 
  (v₁ v₂ : ℝ)
  (h1 : d = 8)
  (h2 : t₁ = 4)
  (h3 : t₂ = 1)
  (h4 : d = (v₁ - v₂) * t₁)
  (h5 : d = (v₁ + v₂) * t₂) :
  v₁ / v₂ = 5 / 3 :=
sorry

end cyclist_speed_ratio_l264_26497


namespace diane_total_loss_l264_26425

-- Define the starting amount of money Diane had.
def starting_amount : ℤ := 100

-- Define the amount of money Diane won.
def winnings : ℤ := 65

-- Define the amount of money Diane owed at the end.
def debt : ℤ := 50

-- Define the total amount of money Diane had after winnings.
def mid_game_total : ℤ := starting_amount + winnings

-- Define the total amount Diane lost.
def total_loss : ℤ := mid_game_total + debt

-- Theorem stating the total amount Diane lost is 215 dollars.
theorem diane_total_loss : total_loss = 215 := by
  sorry

end diane_total_loss_l264_26425


namespace digit_in_92nd_place_l264_26492

/-- The fraction 5/33 is expressed in decimal form as a repeating decimal 0.151515... -/
def fraction_to_decimal : ℚ := 5 / 33

/-- The repeated pattern in the decimal expansion of 5/33 is 15, which is a cycle of length 2 -/
def repeated_pattern (n : ℕ) : ℕ :=
  if n % 2 = 0 then 5 else 1

/-- The digit at the 92nd place in the decimal expansion of 5/33 is 5 -/
theorem digit_in_92nd_place : repeated_pattern 92 = 5 :=
by sorry

end digit_in_92nd_place_l264_26492


namespace desired_yearly_income_l264_26498

theorem desired_yearly_income (total_investment : ℝ) 
  (investment1 : ℝ) (rate1 : ℝ) 
  (investment2 : ℝ) (rate2 : ℝ) 
  (rate_remainder : ℝ) 
  (h_total : total_investment = 10000) 
  (h_invest1 : investment1 = 4000)
  (h_rate1 : rate1 = 0.05) 
  (h_invest2 : investment2 = 3500)
  (h_rate2 : rate2 = 0.04)
  (h_rate_remainder : rate_remainder = 0.064)
  : (rate1 * investment1 + rate2 * investment2 + rate_remainder * (total_investment - (investment1 + investment2))) = 500 := 
by
  sorry

end desired_yearly_income_l264_26498


namespace dodecahedron_interior_diagonals_l264_26420

-- Define the structure and properties of a dodecahedron
structure Dodecahedron :=
  (faces: ℕ := 12)
  (vertices: ℕ := 20)
  (vertices_per_face: ℕ := 5)
  (faces_per_vertex: ℕ := 3)

-- Total number of potential vertices to connect
def total_vertices (d: Dodecahedron) : ℕ := d.vertices - 1

-- Number of connected neighbors per vertex
def connected_neighbors (d: Dodecahedron) : ℕ := d.faces_per_vertex

-- Number of interior diagonals from one vertex
def interior_diagonals_per_vertex (d: Dodecahedron) : ℕ :=
  total_vertices d - connected_neighbors d

-- Total initial count of interior diagonals
def total_initial_interiors (d: Dodecahedron) : ℕ :=
  d.vertices * interior_diagonals_per_vertex d

-- Correct count of interior diagonals by accounting for overcounting
def correct_interior_diagonals (d: Dodecahedron) : ℕ :=
  total_initial_interiors d / 2

-- The theorem to prove
theorem dodecahedron_interior_diagonals (d: Dodecahedron) :
  correct_interior_diagonals d = 160 := by
  sorry

end dodecahedron_interior_diagonals_l264_26420


namespace sum_even_deg_coeff_l264_26444

theorem sum_even_deg_coeff (x : ℕ) : 
  (3 - 2*x)^3 * (2*x + 1)^4 = (3 - 2*x)^3 * (2*x + 1)^4 →
  (∀ (x : ℕ), (3 - 2*x)^3 * (2*1 + 1)^4 =  81 ∧ 
  (3 - 2*(-1))^3 * (2*(-1) + 1)^4 = 125 → 
  (81 + 125) / 2 = 103) :=
by
  sorry

end sum_even_deg_coeff_l264_26444


namespace value_of_m_l264_26406

theorem value_of_m (m : ℕ) : (5^m = 5 * 25^2 * 125^3) → m = 14 :=
by
  sorry

end value_of_m_l264_26406


namespace red_balls_l264_26495

theorem red_balls (w r : ℕ) (h1 : w = 12) (h2 : w * 3 = r * 4) : r = 9 :=
sorry

end red_balls_l264_26495


namespace length_of_BC_l264_26407

theorem length_of_BC (a : ℝ) (b_x b_y c_x c_y area : ℝ) 
  (h1 : b_y = b_x ^ 2)
  (h2 : c_y = c_x ^ 2)
  (h3 : b_y = c_y)
  (h4 : area = 64) :
  c_x - b_x = 8 := by
sorry

end length_of_BC_l264_26407


namespace product_of_two_numbers_l264_26452

theorem product_of_two_numbers (x y : ℕ) (h1 : x + y = 50) (h2 : x - y = 6) : x * y = 616 :=
sorry

end product_of_two_numbers_l264_26452


namespace set_equality_implies_sum_zero_l264_26454

theorem set_equality_implies_sum_zero
  (x y : ℝ)
  (A : Set ℝ := {x, y, x + y})
  (B : Set ℝ := {0, x^2, x * y}) :
  A = B → x + y = 0 :=
by
  sorry

end set_equality_implies_sum_zero_l264_26454


namespace base10_to_base4_addition_l264_26400

-- Define the base 10 numbers
def n1 : ℕ := 45
def n2 : ℕ := 28

-- Define the base 4 representations
def n1_base4 : ℕ := 2 * 4^2 + 3 * 4^1 + 1 * 4^0
def n2_base4 : ℕ := 1 * 4^2 + 3 * 4^1 + 0 * 4^0

-- The sum of the base 10 numbers
def sum_base10 : ℕ := n1 + n2

-- The expected sum in base 4
def sum_base4 : ℕ := 1 * 4^3 + 0 * 4^2 + 2 * 4^1 + 1 * 4^0

-- Prove the equivalence
theorem base10_to_base4_addition :
  (n1 + n2 = n1_base4  + n2_base4) →
  (sum_base10 = sum_base4) :=
by
  sorry

end base10_to_base4_addition_l264_26400


namespace union_sets_l264_26417

noncomputable def setA : Set ℝ := { x | x^2 - 3*x - 4 ≤ 0 }
noncomputable def setB : Set ℝ := { x | 1 < x ∧ x < 5 }

theorem union_sets :
  (setA ∪ setB) = { x | -1 ≤ x ∧ x < 5 } :=
by
  sorry

end union_sets_l264_26417


namespace log_inequality_solution_l264_26499

noncomputable def log_a (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_inequality_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (log_a a (3 / 5) < 1) ↔ (a ∈ Set.Ioo 0 (3 / 5) ∪ Set.Ioi 1) := 
by
  sorry

end log_inequality_solution_l264_26499


namespace right_triangle_hypotenuse_length_l264_26419

theorem right_triangle_hypotenuse_length
  (a b : ℝ)
  (ha : a = 12)
  (hb : b = 16) :
  c = 20 :=
by
  -- Placeholder for the proof
  sorry

end right_triangle_hypotenuse_length_l264_26419


namespace second_person_days_l264_26430

theorem second_person_days (P1 P2 : ℝ) (h1 : P1 = 1 / 24) (h2 : P1 + P2 = 1 / 8) : 1 / P2 = 12 :=
by
  sorry

end second_person_days_l264_26430


namespace football_game_spectators_l264_26439

-- Define the conditions and the proof goals
theorem football_game_spectators 
  (A C : ℕ) 
  (h_condition_1 : 2 * A + 2 * C + 40 = 310) 
  (h_condition_2 : C = A / 2) : 
  A = 90 ∧ C = 45 ∧ (A + C + 20) = 155 := 
by 
  sorry

end football_game_spectators_l264_26439


namespace goldie_worked_hours_last_week_l264_26413

variable (H : ℕ)
variable (money_per_hour : ℕ := 5)
variable (hours_this_week : ℕ := 30)
variable (total_earnings : ℕ := 250)

theorem goldie_worked_hours_last_week :
  H = (total_earnings - hours_this_week * money_per_hour) / money_per_hour :=
sorry

end goldie_worked_hours_last_week_l264_26413


namespace gcd_lcm_product_24_60_l264_26488

theorem gcd_lcm_product_24_60 : 
  (Nat.gcd 24 60) * (Nat.lcm 24 60) = 1440 := by
  sorry

end gcd_lcm_product_24_60_l264_26488


namespace minValue_Proof_l264_26467

noncomputable def minValue (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 2) : Prop :=
  ∃ m : ℝ, m = 4.5 ∧ (∀ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 2 → (1/a + 1/b + 1/c) ≥ 9/2)

theorem minValue_Proof :
  ∀ (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 2), 
    minValue x y z h1 h2 h3 h4 := by
  sorry

end minValue_Proof_l264_26467


namespace how_many_children_got_on_l264_26449

noncomputable def initial_children : ℝ := 42.5
noncomputable def children_got_off : ℝ := 21.3
noncomputable def final_children : ℝ := 35.8

theorem how_many_children_got_on : initial_children - children_got_off + (final_children - (initial_children - children_got_off)) = final_children := by
  sorry

end how_many_children_got_on_l264_26449


namespace solve_for_x_l264_26482

theorem solve_for_x (x y : ℝ) (h1 : x + y = 15) (h2 : x - y = 5) : x = 10 :=
by
  sorry

end solve_for_x_l264_26482


namespace largest_integral_x_l264_26408

theorem largest_integral_x (x : ℤ) : (1 / 4 : ℚ) < (x : ℚ) / 6 ∧ (x : ℚ) / 6 < 7 / 9 ↔ x = 4 :=
by 
  sorry

end largest_integral_x_l264_26408


namespace composite_sum_of_four_integers_l264_26461

theorem composite_sum_of_four_integers 
  (a b c d : ℕ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_pos_d : 0 < d) 
  (h_eq : a^2 + b^2 + a * b = c^2 + d^2 + c * d) : 
  ∃ n m : ℕ, 1 < a + b + c + d ∧ a + b + c + d = n * m ∧ 1 < n ∧ 1 < m := 
sorry

end composite_sum_of_four_integers_l264_26461


namespace a_n_plus_1_is_geometric_general_term_formula_l264_26459

-- Define the sequence a_n.
def a : ℕ → ℤ
| 0       => 0  -- a_0 is not given explicitly, we start the sequence from 1.
| (n + 1) => if n = 0 then 1 else 2 * a n + 1

-- Prove that the sequence {a_n + 1} is a geometric sequence.
theorem a_n_plus_1_is_geometric : ∃ r : ℤ, ∀ n : ℕ, (a (n + 1) + 1) / (a n + 1) = r := by
  sorry

-- Find the general formula for a_n.
theorem general_term_formula : ∃ f : ℕ → ℤ, ∀ n : ℕ, a n = f n := by
  sorry

end a_n_plus_1_is_geometric_general_term_formula_l264_26459


namespace ten_percent_of_number_l264_26446

theorem ten_percent_of_number (x : ℝ)
  (h : x - (1 / 4) * 2 - (1 / 3) * 3 - (1 / 7) * x = 27) :
  0.10 * x = 3.325 :=
sorry

end ten_percent_of_number_l264_26446


namespace average_marks_is_70_l264_26453

variable (P C M : ℕ)

-- Condition: The total marks in physics, chemistry, and mathematics is 140 more than the marks in physics
def total_marks_condition : Prop := P + C + M = P + 140

-- Definition of the average marks in chemistry and mathematics
def average_marks_C_M : ℕ := (C + M) / 2

theorem average_marks_is_70 (h : total_marks_condition P C M) : average_marks_C_M C M = 70 :=
sorry

end average_marks_is_70_l264_26453


namespace line_bisects_circle_l264_26441

theorem line_bisects_circle
  (C : Type)
  [MetricSpace C]
  (x y : ℝ)
  (h : ∀ {x y : ℝ}, x^2 + y^2 - 2*x - 4*y + 1 = 0) : 
  x - y + 1 = 0 → True :=
by
  intro h_line
  sorry

end line_bisects_circle_l264_26441


namespace inequality_holds_for_all_real_l264_26432

open Real -- Open the real numbers namespace

theorem inequality_holds_for_all_real (x : ℝ) : 
  2^((sin x)^2) + 2^((cos x)^2) ≥ 2 * sqrt 2 :=
by
  sorry

end inequality_holds_for_all_real_l264_26432


namespace radius_wheel_l264_26418

noncomputable def pi : ℝ := 3.14159

theorem radius_wheel (D : ℝ) (N : ℕ) (r : ℝ) (h1 : D = 760.57) (h2 : N = 500) :
  r = (D / N) / (2 * pi) :=
sorry

end radius_wheel_l264_26418


namespace sum_of_intercepts_l264_26478

theorem sum_of_intercepts (a b c : ℕ) :
  (∃ y, x = 2 * y^2 - 6 * y + 3 ∧ x = a ∧ y = 0) ∧
  (∃ y1 y2, x = 0 ∧ 2 * y1^2 - 6 * y1 + 3 = 0 ∧ 2 * y2^2 - 6 * y2 + 3 = 0 ∧ y1 + y2 = b + c) →
  a + b + c = 6 :=
by 
  sorry

end sum_of_intercepts_l264_26478


namespace generate_sequence_next_three_members_l264_26402

-- Define the function that generates the sequence
def f (n : ℕ) : ℕ := 2 * (n + 1) ^ 2 * (n + 2) ^ 2

-- Define the predicate that checks if a number can be expressed as the sum of squares of two positive integers
def is_sum_of_squares_of_two_positives (k : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = k

-- The problem statement to prove the equivalence
theorem generate_sequence_next_three_members :
  is_sum_of_squares_of_two_positives (f 1) ∧
  is_sum_of_squares_of_two_positives (f 2) ∧
  is_sum_of_squares_of_two_positives (f 3) ∧
  is_sum_of_squares_of_two_positives (f 4) ∧
  is_sum_of_squares_of_two_positives (f 5) ∧
  is_sum_of_squares_of_two_positives (f 6) ∧
  f 1 = 72 ∧
  f 2 = 288 ∧
  f 3 = 800 ∧
  f 4 = 1800 ∧
  f 5 = 3528 ∧
  f 6 = 6272 :=
sorry

end generate_sequence_next_three_members_l264_26402


namespace mean_of_combined_sets_l264_26474

theorem mean_of_combined_sets (mean_set1 mean_set2 : ℝ) (n1 n2 : ℕ) 
  (h1 : mean_set1 = 15) (h2 : mean_set2 = 20) (h3 : n1 = 5) (h4 : n2 = 8) :
  (n1 * mean_set1 + n2 * mean_set2) / (n1 + n2) = 235 / 13 :=
by
  sorry

end mean_of_combined_sets_l264_26474


namespace sum_second_largest_smallest_l264_26433

theorem sum_second_largest_smallest (a b c : ℕ) (order_cond : a < b ∧ b < c) : a + b = 21 :=
by
  -- Following the correct answer based on the provided conditions:
  -- 10, 11, and 12 with their ordering, we have the smallest a and the second largest b.
  sorry

end sum_second_largest_smallest_l264_26433


namespace five_n_plus_three_composite_l264_26481

theorem five_n_plus_three_composite (n x y : ℕ) 
  (h_pos : 0 < n)
  (h1 : 2 * n + 1 = x ^ 2)
  (h2 : 3 * n + 1 = y ^ 2) : 
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = 5 * n + 3 := 
sorry

end five_n_plus_three_composite_l264_26481


namespace total_scoops_l264_26471

-- Definitions
def single_cone_scoops : ℕ := 1
def double_cone_scoops : ℕ := 2 * single_cone_scoops
def banana_split_scoops : ℕ := 3 * single_cone_scoops
def waffle_bowl_scoops : ℕ := banana_split_scoops + 1

-- Theorem statement: Prove the total number of scoops is 10.
theorem total_scoops : single_cone_scoops + double_cone_scoops + banana_split_scoops + waffle_bowl_scoops = 10 :=
by
  -- Proof steps would go here
  sorry

end total_scoops_l264_26471


namespace intersection_line_l264_26469

-- Define the equations of the circles in Cartesian coordinates.
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + y = 0

-- The theorem to prove.
theorem intersection_line (x y : ℝ) : circle1 x y ∧ circle2 x y → y + 4 * x = 0 :=
by
  sorry

end intersection_line_l264_26469


namespace n_minus_m_l264_26423

theorem n_minus_m (m n : ℝ) (h1 : m^2 - n^2 = 6) (h2 : m + n = 3) : n - m = -2 :=
by
  sorry

end n_minus_m_l264_26423


namespace mean_of_set_is_16_6_l264_26484

theorem mean_of_set_is_16_6 (m : ℝ) (h : m + 7 = 16) :
  (9 + 11 + 16 + 20 + 27) / 5 = 16.6 :=
by
  -- Proof steps would go here, but we use sorry to skip the proof.
  sorry

end mean_of_set_is_16_6_l264_26484


namespace find_r_l264_26456

theorem find_r : ∃ r : ℕ, (5 + 7 * 8 + 1 * 8^2) = 120 + r ∧ r = 5 := 
by
  use 5
  sorry

end find_r_l264_26456


namespace sum_first_15_terms_l264_26416

variable {a : ℕ → ℝ} -- Define the arithmetic sequence as a function from natural numbers to real numbers

-- Define the conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

def a1_plus_a15_eq_three (a : ℕ → ℝ) : Prop :=
  a 1 + a 15 = 3

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (a 1 + a n)

theorem sum_first_15_terms (a : ℕ → ℝ) (h_arith: arithmetic_sequence a) (h_sum: a1_plus_a15_eq_three a) :
  sum_first_n_terms a 15 = 22.5 := by
  sorry

end sum_first_15_terms_l264_26416


namespace number_of_recipes_needed_l264_26401

noncomputable def cookies_per_student : ℕ := 3
noncomputable def total_students : ℕ := 150
noncomputable def recipe_yield : ℕ := 20
noncomputable def attendance_drop_rate : ℝ := 0.30

theorem number_of_recipes_needed : 
  ⌈ (total_students * (1 - attendance_drop_rate) * cookies_per_student) / recipe_yield ⌉ = 16 := by
  sorry

end number_of_recipes_needed_l264_26401


namespace root_count_sqrt_eq_l264_26437

open Real

theorem root_count_sqrt_eq (x : ℝ) :
  (∀ y, (y = sqrt (7 - 2 * x)) → y = x * y → (∃ x, x = 7 / 2 ∨ x = 1)) ∧
  (7 - 2 * x ≥ 0) →
  ∃ s, s = 1 ∧ (7 - 2 * s = 0) → x = 1 ∨ x = 7 / 2 :=
sorry

end root_count_sqrt_eq_l264_26437


namespace bicycle_car_speed_l264_26489

theorem bicycle_car_speed (x : Real) (h1 : x > 0) :
  10 / x - 10 / (2 * x) = 1 / 3 :=
by
  sorry

end bicycle_car_speed_l264_26489


namespace min_overlap_percent_l264_26431

theorem min_overlap_percent
  (M S : ℝ)
  (hM : M = 0.9)
  (hS : S = 0.85) :
  ∃ x, x = 0.75 ∧ (M + S - 1 ≤ x ∧ x ≤ min M S ∧ x = M + S - 1) :=
by
  sorry

end min_overlap_percent_l264_26431


namespace least_positive_n_l264_26440

theorem least_positive_n : ∃ n : ℕ, (1 / (n : ℝ) - 1 / (n + 1 : ℝ) < 1 / 12) ∧ (∀ m : ℕ, (1 / (m : ℝ) - 1 / (m + 1 : ℝ) < 1 / 12) → n ≤ m) :=
by {
  sorry
}

end least_positive_n_l264_26440


namespace maximum_ab_is_40_l264_26445

noncomputable def maximum_ab (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : 5 * a + 8 * b = 80) : ℝ :=
  max (a * b) 40

theorem maximum_ab_is_40 {a b : ℝ} (h₀ : a > 0) (h₁ : b > 0) (h₂ : 5 * a + 8 * b = 80) : maximum_ab a b h₀ h₁ h₂ = 40 := 
by 
  sorry

end maximum_ab_is_40_l264_26445


namespace plane_crash_probabilities_eq_l264_26451

noncomputable def crashing_probability_3_engines (p : ℝ) : ℝ :=
  3 * p^2 * (1 - p) + p^3

noncomputable def crashing_probability_5_engines (p : ℝ) : ℝ :=
  10 * p^3 * (1 - p)^2 + 5 * p^4 * (1 - p) + p^5

theorem plane_crash_probabilities_eq (p : ℝ) :
  crashing_probability_3_engines p = crashing_probability_5_engines p ↔ p = 0 ∨ p = 1/2 ∨ p = 1 :=
by
  sorry

end plane_crash_probabilities_eq_l264_26451


namespace positive_value_of_A_l264_26460

def my_relation (A B k : ℝ) : ℝ := A^2 + k * B^2

theorem positive_value_of_A (A : ℝ) (h1 : ∀ A B, my_relation A B 3 = A^2 + 3 * B^2) (h2 : my_relation A 7 3 = 196) :
  A = 7 := by
  sorry

end positive_value_of_A_l264_26460


namespace factorization_of_difference_of_squares_l264_26448

variable {R : Type} [CommRing R]

theorem factorization_of_difference_of_squares (m : R) : m^2 - 4 = (m + 2) * (m - 2) :=
by sorry

end factorization_of_difference_of_squares_l264_26448


namespace rectangle_dimensions_l264_26472

variable (w l : ℝ)
variable (h1 : l = w + 15)
variable (h2 : 2 * w + 2 * l = 150)

theorem rectangle_dimensions :
  w = 30 ∧ l = 45 :=
by
  sorry

end rectangle_dimensions_l264_26472


namespace ratio_S15_S5_l264_26470

variable {α : Type*} [LinearOrderedField α]

namespace ArithmeticSequence

def sum_of_first_n_terms (a : α) (d : α) (n : ℕ) : α :=
  n * (2 * a + (n - 1) * d) / 2

theorem ratio_S15_S5
  {a d : α}
  {S5 S10 S15 : α}
  (h1 : S5 = sum_of_first_n_terms a d 5)
  (h2 : S10 = sum_of_first_n_terms a d 10)
  (h3 : S15 = sum_of_first_n_terms a d 15)
  (h_ratio : S5 / S10 = 2 / 3) :
  S15 / S5 = 3 / 2 := 
sorry

end ArithmeticSequence

end ratio_S15_S5_l264_26470


namespace inequality_first_inequality_second_l264_26466

theorem inequality_first (x : ℝ) : 4 * x - 2 < 1 - 2 * x → x < 1 / 2 := 
sorry

theorem inequality_second (x : ℝ) : (3 - 2 * x ≥ x - 6) ∧ ((3 * x + 1) / 2 < 2 * x) → 1 < x ∧ x ≤ 3 :=
sorry

end inequality_first_inequality_second_l264_26466


namespace pencils_left_l264_26465

def ashton_boxes : Nat := 3
def pencils_per_box : Nat := 14
def pencils_given_to_brother : Nat := 6
def pencils_given_to_friends : Nat := 12

theorem pencils_left (h₁ : ashton_boxes = 3) 
                     (h₂ : pencils_per_box = 14)
                     (h₃ : pencils_given_to_brother = 6)
                     (h₄ : pencils_given_to_friends = 12) :
  (ashton_boxes * pencils_per_box - pencils_given_to_brother - pencils_given_to_friends) = 24 :=
by
  sorry

end pencils_left_l264_26465


namespace triangle_is_isosceles_right_triangle_l264_26414

theorem triangle_is_isosceles_right_triangle
  (a b c : ℝ)
  (h1 : (a - b)^2 + (Real.sqrt (2 * a - b - 3)) + (abs (c - 3 * Real.sqrt 2)) = 0) :
  (a = 3) ∧ (b = 3) ∧ (c = 3 * Real.sqrt 2) :=
by
  sorry

end triangle_is_isosceles_right_triangle_l264_26414


namespace largest_pies_without_any_ingredients_l264_26462

-- Define the conditions
def total_pies : ℕ := 60
def pies_with_strawberries : ℕ := total_pies / 4
def pies_with_bananas : ℕ := total_pies * 3 / 8
def pies_with_cherries : ℕ := total_pies / 2
def pies_with_pecans : ℕ := total_pies / 10

-- State the theorem to prove
theorem largest_pies_without_any_ingredients : (total_pies - pies_with_cherries) = 30 := by
  sorry

end largest_pies_without_any_ingredients_l264_26462


namespace find_m_if_parallel_l264_26458

-- Given vectors
def a (m : ℝ) : ℝ × ℝ := (2, m)
def b (m : ℝ) : ℝ × ℝ := (m, 2)

-- Parallel condition and the result that m must be -2 or 2
theorem find_m_if_parallel (m : ℝ) (h : ∃ k : ℝ, a m = (k * (b m).fst, k * (b m).snd)) : 
  m = -2 ∨ m = 2 :=
sorry

end find_m_if_parallel_l264_26458


namespace ted_age_proof_l264_26427

theorem ted_age_proof (s t : ℝ) (h1 : t = 3 * s - 20) (h2 : t + s = 78) : t = 53.5 :=
by
  sorry  -- Proof steps are not required, hence using sorry.

end ted_age_proof_l264_26427


namespace fraction_covered_by_pepperoni_l264_26457

theorem fraction_covered_by_pepperoni 
  (d_pizza : ℝ) (n_pepperoni_diameter : ℕ) (n_pepperoni : ℕ) (diameter_pepperoni : ℝ) 
  (radius_pepperoni : ℝ) (radius_pizza : ℝ)
  (area_one_pepperoni : ℝ) (total_area_pepperoni : ℝ) (area_pizza : ℝ)
  (fraction_covered : ℝ)
  (h1 : d_pizza = 16)
  (h2 : n_pepperoni_diameter = 14)
  (h3 : n_pepperoni = 42)
  (h4 : diameter_pepperoni = d_pizza / n_pepperoni_diameter)
  (h5 : radius_pepperoni = diameter_pepperoni / 2)
  (h6 : radius_pizza = d_pizza / 2)
  (h7 : area_one_pepperoni = π * radius_pepperoni ^ 2)
  (h8 : total_area_pepperoni = n_pepperoni * area_one_pepperoni)
  (h9 : area_pizza = π * radius_pizza ^ 2)
  (h10 : fraction_covered = total_area_pepperoni / area_pizza) :
  fraction_covered = 3 / 7 :=
sorry

end fraction_covered_by_pepperoni_l264_26457


namespace how_many_oxen_c_put_l264_26464

variables (oxen_a oxen_b months_a months_b rent total_rent c_share x : ℕ)
variable (H : 10 * 7 = oxen_a)
variable (H1 : 12 * 5 = oxen_b)
variable (H2 : 3 * x = months_a)
variable (H3 : 70 + 60 + 3 * x = months_b)
variable (H4 : 280 = total_rent)
variable (H5 : 72 = c_share)

theorem how_many_oxen_c_put : x = 15 :=
  sorry

end how_many_oxen_c_put_l264_26464


namespace intersection_of_A_and_B_is_2_l264_26404

-- Define the sets A and B based on the given conditions
def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B : Set ℝ := {2, 3}

-- State the theorem that needs to be proved
theorem intersection_of_A_and_B_is_2 : A ∩ B = {2} :=
by
  sorry

end intersection_of_A_and_B_is_2_l264_26404


namespace smallest_positive_multiple_of_45_l264_26426

theorem smallest_positive_multiple_of_45 : ∃ (n : ℕ), n > 0 ∧ ∃ (x : ℕ), x > 0 ∧ n = 45 * x ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l264_26426


namespace count_valid_pairs_is_7_l264_26434

def valid_pairs_count : Nat :=
  let pairs := [(1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (3, 3), (4, 4)]
  List.length pairs

theorem count_valid_pairs_is_7 (b c : ℕ) (hb : b > 0) (hc : c > 0) :
  (b^2 - 4 * c ≤ 0) → (c^2 - 4 * b ≤ 0) → valid_pairs_count = 7 :=
by
  sorry

end count_valid_pairs_is_7_l264_26434


namespace trig_expression_equality_l264_26468

theorem trig_expression_equality :
  (1 - 1 / (Real.cos (Real.pi / 6))) *
  (1 + 1 / (Real.sin (Real.pi / 3))) *
  (1 - 1 / (Real.sin (Real.pi / 6))) *
  (1 + 1 / (Real.cos (Real.pi / 3))) = 1 :=
by
  have hcos30 : Real.cos (Real.pi / 6) = (Real.sqrt 3) / 2 := sorry
  have hsin60 : Real.sin (Real.pi / 3) = (Real.sqrt 3) / 2 := sorry
  have hsin30 : Real.sin (Real.pi / 6) = 1 / 2 := sorry
  have hcos60 : Real.cos (Real.pi / 3) = 1 / 2 := sorry
  sorry

end trig_expression_equality_l264_26468


namespace cube_surface_area_l264_26485

-- Definitions based on conditions from the problem
def edge_length : ℕ := 7
def number_of_faces : ℕ := 6

-- Definition of the problem converted to a theorem in Lean 4
theorem cube_surface_area (edge_length : ℕ) (number_of_faces : ℕ) : 
  number_of_faces * (edge_length * edge_length) = 294 :=
by
  -- Proof steps are omitted, so we put sorry to indicate that the proof is required.
  sorry

end cube_surface_area_l264_26485


namespace rectangle_area_l264_26447

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 226) : L * B = 3060 :=
by sorry

end rectangle_area_l264_26447


namespace min_side_value_l264_26479

-- Definitions based on the conditions provided
variables (a b c : ℕ) (h1 : a - b = 5) (h2 : (a + b + c) % 2 = 0)

theorem min_side_value (h1 : a - b = 5) (h2 : (a + b + c) % 2 = 0) : c ≥ 7 :=
sorry

end min_side_value_l264_26479


namespace find_quadruples_l264_26496

open Nat

/-- Define the primality property -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- Define the problem conditions -/
def valid_quadruple (p1 p2 p3 p4 : ℕ) : Prop :=
  p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧
  is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧
  p1 * p2 + p2 * p3 + p3 * p4 + p4 * p1 = 882

/-- The final theorem stating the valid quadruples -/
theorem find_quadruples :
  ∀ (p1 p2 p3 p4 : ℕ), valid_quadruple p1 p2 p3 p4 ↔ 
  (p1 = 2 ∧ p2 = 5 ∧ p3 = 19 ∧ p4 = 37) ∨
  (p1 = 2 ∧ p2 = 11 ∧ p3 = 19 ∧ p4 = 31) ∨
  (p1 = 2 ∧ p2 = 13 ∧ p3 = 19 ∧ p4 = 29) :=
by
  sorry

end find_quadruples_l264_26496


namespace quadrilateral_with_equal_sides_is_rhombus_l264_26424

theorem quadrilateral_with_equal_sides_is_rhombus (a b c d : ℝ) (h1 : a = b) (h2 : b = c) (h3 : c = d) : a = d :=
by
  sorry

end quadrilateral_with_equal_sides_is_rhombus_l264_26424


namespace chords_in_circle_l264_26490

theorem chords_in_circle (n : ℕ) (h_n : n = 10) : (n.choose 2) = 45 := sorry

end chords_in_circle_l264_26490


namespace expression_evaluation_l264_26475

theorem expression_evaluation (a b : ℕ) (h1 : a = 25) (h2 : b = 15) : (a + b)^2 - (a^2 + b^2) = 750 :=
by
  sorry

end expression_evaluation_l264_26475


namespace dexter_filled_fewer_boxes_with_football_cards_l264_26473

-- Conditions
def boxes_with_basketball_cards : ℕ := 9
def cards_per_basketball_box : ℕ := 15
def cards_per_football_box : ℕ := 20
def total_cards : ℕ := 255

-- Definition of the main problem statement
def fewer_boxes_with_football_cards : Prop :=
  let basketball_cards := boxes_with_basketball_cards * cards_per_basketball_box
  let football_cards := total_cards - basketball_cards
  let boxes_with_football_cards := football_cards / cards_per_football_box
  boxes_with_basketball_cards - boxes_with_football_cards = 3

theorem dexter_filled_fewer_boxes_with_football_cards : fewer_boxes_with_football_cards :=
by
  sorry

end dexter_filled_fewer_boxes_with_football_cards_l264_26473


namespace part_i_solution_set_part_ii_minimum_value_l264_26436

-- Part (I)
theorem part_i_solution_set :
  (∀ (x : ℝ), 1 = 1 ∧ 2 = 2 → |x - 1| + |x + 2| ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2) :=
by { sorry }

-- Part (II)
theorem part_ii_minimum_value (a b x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 2 * a * b) :
  |x - a| + |x + b| ≥ 9 / 2 :=
by { sorry }

end part_i_solution_set_part_ii_minimum_value_l264_26436


namespace value_of_sum_l264_26443

theorem value_of_sum (a b c : ℚ) (h1 : 2 * a + 3 * b + c = 27) (h2 : 4 * a + 6 * b + 5 * c = 71) :
  a + b + c = 115 / 9 :=
sorry

end value_of_sum_l264_26443


namespace f_decreasing_max_k_value_l264_26450

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log (x + 1)) / x

theorem f_decreasing : ∀ x > 0, ∀ y > 0, x < y → f x > f y := by
  sorry

theorem max_k_value : ∀ x > 0, f x > k / (x + 1) → k ≤ 3 := by
  sorry

end f_decreasing_max_k_value_l264_26450


namespace opposite_points_number_line_l264_26476

theorem opposite_points_number_line (a : ℤ) (h : a - 6 = -a) : a = 3 := by
  sorry

end opposite_points_number_line_l264_26476


namespace carrie_weekly_earning_l264_26415

-- Definitions and conditions
def iphone_cost : ℕ := 800
def trade_in_value : ℕ := 240
def weeks_needed : ℕ := 7

-- Calculate the required weekly earning
def weekly_earning : ℕ := (iphone_cost - trade_in_value) / weeks_needed

-- Problem statement: Prove that Carrie makes $80 per week babysitting
theorem carrie_weekly_earning :
  weekly_earning = 80 := by
  sorry

end carrie_weekly_earning_l264_26415


namespace average_time_per_stop_l264_26442

theorem average_time_per_stop (pizzas : ℕ) 
                              (stops_for_two_pizzas : ℕ) 
                              (pizzas_per_stop_for_two : ℕ) 
                              (remaining_pizzas : ℕ) 
                              (total_stops : ℕ) 
                              (total_time : ℕ) 
                              (H1: pizzas = 12) 
                              (H2: stops_for_two_pizzas = 2) 
                              (H3: pizzas_per_stop_for_two = 2) 
                              (H4: remaining_pizzas = pizzas - stops_for_two_pizzas * pizzas_per_stop_for_two)
                              (H5: total_stops = stops_for_two_pizzas + remaining_pizzas)
                              (H6: total_time = 40) :
                              total_time / total_stops = 4 :=
by
  sorry

end average_time_per_stop_l264_26442


namespace game_necessarily_ends_winning_strategy_l264_26494

-- Definitions and conditions based on problem:
def Card := Fin 2009

def isWhite (c : Fin 2009) : Prop := sorry -- Placeholder for actual white card predicate

def validMove (k : Fin 2009) : Prop := k.val < 1969 ∧ isWhite k

def applyMove (k : Fin 2009) (cards : Fin 2009 → Prop) : Fin 2009 → Prop :=
  fun c => if c.val ≥ k.val ∧ c.val < k.val + 41 then ¬isWhite c else isWhite c

-- Theorem statements to match proof problem:
theorem game_necessarily_ends : ∃ n, n = 2009 → (∀ (cards : Fin 2009 → Prop), (∃ k < 1969, validMove k) → (∀ k < 1969, ¬(validMove k))) :=
sorry

theorem winning_strategy (cards : Fin 2009 → Prop) : ∃ strategy : (Fin 2009 → Prop) → Fin 2009, ∀ s, (s = applyMove (strategy s) s) → strategy s = sorry :=
sorry

end game_necessarily_ends_winning_strategy_l264_26494


namespace veranda_width_l264_26403

theorem veranda_width (w : ℝ) (h_room : 18 * 12 = 216) (h_veranda : 136 = 136) : 
  (18 + 2*w) * (12 + 2*w) = 352 → w = 2 :=
by
  sorry

end veranda_width_l264_26403


namespace sum_of_six_digits_is_31_l264_26421

-- Problem constants and definitions
def digits : Set ℕ := {0, 2, 3, 4, 5, 7, 8, 9}

-- Problem conditions expressed as hypotheses
variables (a b c d e f g : ℕ)
variables (h1 : a ∈ digits) (h2 : b ∈ digits) (h3 : c ∈ digits) 
          (h4 : d ∈ digits) (h5 : e ∈ digits) (h6 : f ∈ digits) (h7 : g ∈ digits)
          (h8 : a ≠ b) (h9 : a ≠ c) (h10 : a ≠ d) (h11 : a ≠ e) (h12 : a ≠ f) (h13 : a ≠ g)
          (h14 : b ≠ c) (h15 : b ≠ d) (h16 : b ≠ e) (h17 : b ≠ f) (h18 : b ≠ g)
          (h19 : c ≠ d) (h20 : c ≠ e) (h21 : c ≠ f) (h22 : c ≠ g)
          (h23 : d ≠ e) (h24 : d ≠ f) (h25 : d ≠ g)
          (h26 : e ≠ f) (h27 : e ≠ g) (h28 : f ≠ g)
variable (shared : b = e)
variables (h29 : a + b + c = 24) (h30 : d + e + f + g = 14)

-- Proposition to be proved
theorem sum_of_six_digits_is_31 : a + b + c + d + e + f = 31 :=
by 
  sorry

end sum_of_six_digits_is_31_l264_26421


namespace tied_part_length_l264_26455

theorem tied_part_length (length_of_each_string : ℕ) (num_strings : ℕ) (total_tied_length : ℕ) 
  (H1 : length_of_each_string = 217) (H2 : num_strings = 3) (H3 : total_tied_length = 627) : 
  (length_of_each_string * num_strings - total_tied_length) / (num_strings - 1) = 12 :=
by
  sorry

end tied_part_length_l264_26455


namespace donna_smallest_n_l264_26487

theorem donna_smallest_n (n : ℕ) : 15 * n - 1 % 6 = 0 ↔ n % 6 = 5 := sorry

end donna_smallest_n_l264_26487
