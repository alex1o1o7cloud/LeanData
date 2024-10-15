import Mathlib

namespace NUMINAMATH_GPT_number_of_terms_geometric_seq_l2207_220728

-- Given conditions
variables (a1 q : ℝ)  -- First term and common ratio of the sequence
variable  (n : ℕ)     -- Number of terms in the sequence

-- The product of the first three terms
axiom condition1 : a1^3 * q^3 = 3

-- The product of the last three terms
axiom condition2 : a1^3 * q^(3 * n - 6) = 9

-- The product of all terms
axiom condition3 : a1^n * q^(n * (n - 1) / 2) = 729

-- Proving the number of terms in the sequence
theorem number_of_terms_geometric_seq : n = 12 := by
  sorry

end NUMINAMATH_GPT_number_of_terms_geometric_seq_l2207_220728


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2207_220771

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) (n : ℕ) 
  (h_arith : ∀ n, a (n+1) = a n + 3)
  (h_a1_a2 : a 1 + a 2 = 7)
  (h_a3 : a 3 = 8)
  (h_bn : ∀ n, b n = 1 / (a n * a (n+1)))
  :
  (∀ n, a n = 3 * n - 1) ∧ (T n = n / (2 * (3 * n + 2))) :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2207_220771


namespace NUMINAMATH_GPT_moles_of_C2H6_l2207_220746

-- Define the reactive coefficients
def ratio_C := 2
def ratio_H2 := 3
def ratio_C2H6 := 1

-- Given conditions
def moles_C := 6
def moles_H2 := 9

-- Function to calculate moles of C2H6 formed
def moles_C2H6_formed (m_C : ℕ) (m_H2 : ℕ) : ℕ :=
  min (m_C * ratio_C2H6 / ratio_C) (m_H2 * ratio_C2H6 / ratio_H2)

-- Theorem statement: the number of moles of C2H6 formed is 3
theorem moles_of_C2H6 : moles_C2H6_formed moles_C moles_H2 = 3 :=
by {
  -- Sorry is used since we are not providing the proof here
  sorry
}

end NUMINAMATH_GPT_moles_of_C2H6_l2207_220746


namespace NUMINAMATH_GPT_janessa_kept_20_cards_l2207_220768

-- Definitions based on conditions
def initial_cards : Nat := 4
def father_cards : Nat := 13
def ebay_cards : Nat := 36
def bad_shape_cards : Nat := 4
def cards_given_to_dexter : Nat := 29

-- Prove that Janessa kept 20 cards for herself
theorem janessa_kept_20_cards :
  (initial_cards + father_cards  + ebay_cards - bad_shape_cards) - cards_given_to_dexter = 20 :=
by
  sorry

end NUMINAMATH_GPT_janessa_kept_20_cards_l2207_220768


namespace NUMINAMATH_GPT_max_squares_covered_by_card_l2207_220796

theorem max_squares_covered_by_card : 
  let checkerboard_square_size := 1
  let card_side := 2
  let card_diagonal := Real.sqrt (card_side ^ 2 + card_side ^ 2)
  ∃ n, n = 9 :=
by
  let checkerboard_square_size := 1
  let card_side := 2
  let card_diagonal := Real.sqrt (card_side ^ 2 + card_side ^ 2)
  existsi 9
  sorry

end NUMINAMATH_GPT_max_squares_covered_by_card_l2207_220796


namespace NUMINAMATH_GPT_ratio_is_one_third_l2207_220772

-- Definitions based on given conditions
def total_students : ℕ := 90
def initial_cafeteria_students : ℕ := (2 * total_students) / 3
def initial_outside_students : ℕ := total_students - initial_cafeteria_students
def moved_cafeteria_to_outside : ℕ := 3
def final_cafeteria_students : ℕ := 67
def students_ran_inside : ℕ := final_cafeteria_students - (initial_cafeteria_students - moved_cafeteria_to_outside)

-- Ratio calculation as a proof statement
def ratio_ran_inside_to_outside : ℚ := students_ran_inside / initial_outside_students

-- Proof that the ratio is 1/3
theorem ratio_is_one_third : ratio_ran_inside_to_outside = 1 / 3 :=
by sorry -- Proof omitted

end NUMINAMATH_GPT_ratio_is_one_third_l2207_220772


namespace NUMINAMATH_GPT_large_monkey_doll_cost_l2207_220711

theorem large_monkey_doll_cost :
  ∃ (L : ℝ), (300 / L - 300 / (L - 2) = 25) ∧ L > 0 := by
  sorry

end NUMINAMATH_GPT_large_monkey_doll_cost_l2207_220711


namespace NUMINAMATH_GPT_apples_per_pie_l2207_220791

/-- Let's define the parameters given in the problem -/
def initial_apples : ℕ := 62
def apples_given_to_students : ℕ := 8
def pies_made : ℕ := 6

/-- Define the remaining apples after handing out to students -/
def remaining_apples : ℕ := initial_apples - apples_given_to_students

/-- The statement we need to prove: each pie requires 9 apples -/
theorem apples_per_pie : remaining_apples / pies_made = 9 := by
  -- Add the proof here
  sorry

end NUMINAMATH_GPT_apples_per_pie_l2207_220791


namespace NUMINAMATH_GPT_total_points_scored_l2207_220777

theorem total_points_scored (points_per_round : ℕ) (rounds : ℕ) (h1 : points_per_round = 42) (h2 : rounds = 2) : 
  points_per_round * rounds = 84 :=
by
  sorry

end NUMINAMATH_GPT_total_points_scored_l2207_220777


namespace NUMINAMATH_GPT_base_conversion_subtraction_l2207_220744

namespace BaseConversion

def base9_to_base10 (n : ℕ) : ℕ :=
  3 * 9^2 + 2 * 9^1 + 4 * 9^0

def base6_to_base10 (n : ℕ) : ℕ :=
  1 * 6^2 + 5 * 6^1 + 6 * 6^0

theorem base_conversion_subtraction : (base9_to_base10 324) - (base6_to_base10 156) = 193 := by
  sorry

end BaseConversion

end NUMINAMATH_GPT_base_conversion_subtraction_l2207_220744


namespace NUMINAMATH_GPT_find_f_2_l2207_220740

def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 3

theorem find_f_2 (a b : ℝ) (hf_neg2 : f a b (-2) = 7) : f a b 2 = -13 :=
by
  sorry

end NUMINAMATH_GPT_find_f_2_l2207_220740


namespace NUMINAMATH_GPT_positive_real_as_sum_l2207_220739

theorem positive_real_as_sum (k : ℝ) (hk : k > 0) : 
  ∃ (a : ℕ → ℕ), (∀ n, a n > 0) ∧ (∀ n, a n < a (n + 1)) ∧ (∑' n, 1 / 10 ^ a n = k) :=
sorry

end NUMINAMATH_GPT_positive_real_as_sum_l2207_220739


namespace NUMINAMATH_GPT_number_four_units_away_from_neg_five_l2207_220769

theorem number_four_units_away_from_neg_five (x : ℝ) : 
    abs (x + 5) = 4 ↔ x = -9 ∨ x = -1 :=
by 
  sorry

end NUMINAMATH_GPT_number_four_units_away_from_neg_five_l2207_220769


namespace NUMINAMATH_GPT_intersection_proof_l2207_220761

-- Definitions of sets M and N
def M : Set ℝ := { x | x^2 < 4 }
def N : Set ℝ := { x | x < 1 }

-- The intersection of M and N
def intersection : Set ℝ := { x | -2 < x ∧ x < 1 }

-- Proposition to prove
theorem intersection_proof : M ∩ N = intersection :=
by sorry

end NUMINAMATH_GPT_intersection_proof_l2207_220761


namespace NUMINAMATH_GPT_integer_count_of_sqrt_x_l2207_220774

theorem integer_count_of_sqrt_x : ∃ (n : ℕ), n = 15 ∧ ∀ (x : ℤ), (9 < x ∧ x < 25) ↔ (10 ≤ x ∧ x ≤ 24) :=
by
  sorry

end NUMINAMATH_GPT_integer_count_of_sqrt_x_l2207_220774


namespace NUMINAMATH_GPT_cube_faces_sum_l2207_220715

open Nat

theorem cube_faces_sum (a b c d e f : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < e) (h6 : 0 < f) 
    (h7 : (a + d) * (b + e) * (c + f) = 1386) : 
    a + b + c + d + e + f = 38 := 
sorry

end NUMINAMATH_GPT_cube_faces_sum_l2207_220715


namespace NUMINAMATH_GPT_fewest_people_to_join_CBL_l2207_220760

theorem fewest_people_to_join_CBL (initial_people teamsize : ℕ) (even_teams : ℕ → Prop)
  (initial_people_eq : initial_people = 38)
  (teamsize_eq : teamsize = 9)
  (even_teams_def : ∀ n, even_teams n ↔ n % 2 = 0) :
  ∃(p : ℕ), (initial_people + p) % teamsize = 0 ∧ even_teams ((initial_people + p) / teamsize) ∧ p = 16 := by
  sorry

end NUMINAMATH_GPT_fewest_people_to_join_CBL_l2207_220760


namespace NUMINAMATH_GPT_cos_180_eq_neg1_l2207_220745

theorem cos_180_eq_neg1 : Real.cos (180 * Real.pi / 180) = -1 :=
by
  sorry

end NUMINAMATH_GPT_cos_180_eq_neg1_l2207_220745


namespace NUMINAMATH_GPT_max_profit_at_800_l2207_220757

open Nat

def P (x : ℕ) : ℝ :=
  if h : 0 < x ∧ x ≤ 100 then 80
  else if h : 100 < x ∧ x ≤ 1000 then 82 - 0.02 * x
  else 0

def f (x : ℕ) : ℝ :=
  if h : 0 < x ∧ x ≤ 100 then 30 * x
  else if h : 100 < x ∧ x ≤ 1000 then 32 * x - 0.02 * x^2
  else 0

theorem max_profit_at_800 :
  ∀ x : ℕ, f x ≤ 12800 ∧ f 800 = 12800 :=
sorry

end NUMINAMATH_GPT_max_profit_at_800_l2207_220757


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_l2207_220763

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℤ)
  (h1 : ∀ n m, a (n+1) - a n = a (m+1) - a m)
  (h2 : (a 2 + a 6) / 2 = 5)
  (h3 : (a 3 + a 7) / 2 = 7) :
  ∀ n, a n = 2 * n - 3 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_l2207_220763


namespace NUMINAMATH_GPT_correct_sqrt_evaluation_l2207_220782

theorem correct_sqrt_evaluation:
  2 * Real.sqrt 2 - Real.sqrt 2 = Real.sqrt 2 :=
by 
  sorry

end NUMINAMATH_GPT_correct_sqrt_evaluation_l2207_220782


namespace NUMINAMATH_GPT_exists_similarity_point_l2207_220767

variable {Point : Type} [MetricSpace Point]

noncomputable def similar_triangles (A B A' B' : Point) (O : Point) : Prop :=
  dist A O / dist A' O = dist A B / dist A' B' ∧ dist B O / dist B' O = dist A B / dist A' B'

theorem exists_similarity_point (A B A' B' : Point) (h1 : dist A B ≠ 0) (h2: dist A' B' ≠ 0) :
  ∃ O : Point, similar_triangles A B A' B' O :=
  sorry

end NUMINAMATH_GPT_exists_similarity_point_l2207_220767


namespace NUMINAMATH_GPT_rectangle_inscribed_area_l2207_220783

variables (b h x : ℝ) 

theorem rectangle_inscribed_area (hb : 0 < b) (hh : 0 < h) (hx : 0 < x) (hx_lt_h : x < h) :
  ∃ A, A = (b * x * (h - x)) / h :=
sorry

end NUMINAMATH_GPT_rectangle_inscribed_area_l2207_220783


namespace NUMINAMATH_GPT_integer_values_of_a_l2207_220722

variable (a b c x : ℤ)

theorem integer_values_of_a (h : (x - a) * (x - 12) + 4 = (x + b) * (x + c)) : a = 7 ∨ a = 17 := by
  sorry

end NUMINAMATH_GPT_integer_values_of_a_l2207_220722


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_solve_eq3_l2207_220731

def equation1 (x : ℝ) : Prop := x^2 - 6 * x + 5 = 0
def solution1 (x : ℝ) : Prop := x = 5 ∨ x = 1

theorem solve_eq1 : ∀ x : ℝ, equation1 x ↔ solution1 x := sorry

def equation2 (x : ℝ) : Prop := 3 * x * (2 * x - 1) = 4 * x - 2
def solution2 (x : ℝ) : Prop := x = 1/2 ∨ x = 2/3

theorem solve_eq2 : ∀ x : ℝ, equation2 x ↔ solution2 x := sorry

def equation3 (x : ℝ) : Prop := x^2 - 2 * Real.sqrt 2 * x - 2 = 0
def solution3 (x : ℝ) : Prop := x = Real.sqrt 2 + 2 ∨ x = Real.sqrt 2 - 2

theorem solve_eq3 : ∀ x : ℝ, equation3 x ↔ solution3 x := sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_solve_eq3_l2207_220731


namespace NUMINAMATH_GPT_correct_exponentiation_l2207_220723

theorem correct_exponentiation (a : ℝ) :
  (a^2 * a^3 = a^5) ∧
  (a^2 + a^3 ≠ a^5) ∧
  (a^6 + a^2 ≠ a^4) ∧
  (3 * a^3 - a^2 ≠ 2 * a) :=
by
  sorry

end NUMINAMATH_GPT_correct_exponentiation_l2207_220723


namespace NUMINAMATH_GPT_prairie_total_area_l2207_220720

theorem prairie_total_area (acres_dust_storm : ℕ) (acres_untouched : ℕ) (h₁ : acres_dust_storm = 64535) (h₂ : acres_untouched = 522) : acres_dust_storm + acres_untouched = 65057 :=
by
  sorry

end NUMINAMATH_GPT_prairie_total_area_l2207_220720


namespace NUMINAMATH_GPT_distribute_ways_l2207_220726

/-- There are 5 distinguishable balls and 4 distinguishable boxes.
The total number of ways to distribute these balls into the boxes is 1024. -/
theorem distribute_ways : (4 : ℕ) ^ (5 : ℕ) = 1024 := by
  sorry

end NUMINAMATH_GPT_distribute_ways_l2207_220726


namespace NUMINAMATH_GPT_hashtag_3_8_l2207_220708

-- Define the hashtag operation
def hashtag (a b : ℤ) : ℤ := a * b - b + b ^ 2

-- Prove that 3 # 8 equals 80
theorem hashtag_3_8 : hashtag 3 8 = 80 := by
  sorry

end NUMINAMATH_GPT_hashtag_3_8_l2207_220708


namespace NUMINAMATH_GPT_number_of_partitions_indistinguishable_balls_into_boxes_l2207_220716

/-- The number of distinct ways to partition 6 indistinguishable balls into 3 indistinguishable boxes is 7. -/
theorem number_of_partitions_indistinguishable_balls_into_boxes :
  ∃ n : ℕ, n = 7 := sorry

end NUMINAMATH_GPT_number_of_partitions_indistinguishable_balls_into_boxes_l2207_220716


namespace NUMINAMATH_GPT_total_purchase_cost_l2207_220734

-- Definitions for the quantities of the items
def quantity_chocolate_bars : ℕ := 10
def quantity_gummy_bears : ℕ := 10
def quantity_chocolate_chips : ℕ := 20

-- Definitions for the costs of the items
def cost_per_chocolate_bar : ℕ := 3
def cost_per_gummy_bear_pack : ℕ := 2
def cost_per_chocolate_chip_bag : ℕ := 5

-- Proof statement to be shown
theorem total_purchase_cost :
  (quantity_chocolate_bars * cost_per_chocolate_bar) + 
  (quantity_gummy_bears * cost_per_gummy_bear_pack) + 
  (quantity_chocolate_chips * cost_per_chocolate_chip_bag) = 150 :=
sorry

end NUMINAMATH_GPT_total_purchase_cost_l2207_220734


namespace NUMINAMATH_GPT_average_growth_rate_le_max_growth_rate_l2207_220737

variable (P : ℝ) (a : ℝ) (b : ℝ) (x : ℝ)

theorem average_growth_rate_le_max_growth_rate (h : (1 + x)^2 = (1 + a) * (1 + b)) :
  x ≤ max a b := 
sorry

end NUMINAMATH_GPT_average_growth_rate_le_max_growth_rate_l2207_220737


namespace NUMINAMATH_GPT_proposition_false_n4_l2207_220780

variable {P : ℕ → Prop}

theorem proposition_false_n4
  (h_ind : ∀ (k : ℕ), k ≠ 0 → P k → P (k + 1))
  (h_false_5 : P 5 = False) :
  P 4 = False :=
sorry

end NUMINAMATH_GPT_proposition_false_n4_l2207_220780


namespace NUMINAMATH_GPT_cost_per_minute_of_each_call_l2207_220753

theorem cost_per_minute_of_each_call :
  let calls_per_week := 50
  let hours_per_call := 1
  let weeks_per_month := 4
  let total_hours_in_month := calls_per_week * hours_per_call * weeks_per_month
  let total_cost := 600
  let cost_per_hour := total_cost / total_hours_in_month
  let minutes_per_hour := 60
  let cost_per_minute := cost_per_hour / minutes_per_hour
  cost_per_minute = 0.05 := 
by
  sorry

end NUMINAMATH_GPT_cost_per_minute_of_each_call_l2207_220753


namespace NUMINAMATH_GPT_quadratic_does_not_pass_third_quadrant_l2207_220717

-- Definitions of the functions
def linear_function (a b x : ℝ) : ℝ := -a * x + b
def quadratic_function (a b x : ℝ) : ℝ := -a * x^2 + b * x

-- Conditions
variables (a b : ℝ)
axiom a_nonzero : a ≠ 0
axiom passes_first_third_fourth : ∀ x, (linear_function a b x > 0 ∧ x > 0) ∨ (linear_function a b x < 0 ∧ x < 0) ∨ (linear_function a b x < 0 ∧ x > 0)

-- Theorem stating the problem
theorem quadratic_does_not_pass_third_quadrant :
  ¬ (∃ x, quadratic_function a b x < 0 ∧ x < 0) := 
sorry

end NUMINAMATH_GPT_quadratic_does_not_pass_third_quadrant_l2207_220717


namespace NUMINAMATH_GPT_third_root_of_polynomial_l2207_220735

variable (a b x : ℝ)
noncomputable def polynomial := a * x^3 + (a + 3 * b) * x^2 + (b - 4 * a) * x + (10 - a)

theorem third_root_of_polynomial (h1 : polynomial a b (-3) = 0) (h2 : polynomial a b 4 = 0) :
  ∃ r : ℝ, r = -17 / 10 ∧ polynomial a b r = 0 :=
by
  sorry

end NUMINAMATH_GPT_third_root_of_polynomial_l2207_220735


namespace NUMINAMATH_GPT_negation_of_proposition_l2207_220718

theorem negation_of_proposition :
  (∀ (x y : ℝ), x^2 + y^2 - 1 > 0) → (∃ (x y : ℝ), x^2 + y^2 - 1 ≤ 0) :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l2207_220718


namespace NUMINAMATH_GPT_sara_red_balloons_l2207_220730

theorem sara_red_balloons (initial_red : ℕ) (given_red : ℕ) 
  (h_initial : initial_red = 31) (h_given : given_red = 24) : 
  initial_red - given_red = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_sara_red_balloons_l2207_220730


namespace NUMINAMATH_GPT_volume_of_tetrahedron_PQRS_l2207_220743

-- Definitions of the given conditions for the tetrahedron
def PQ := 6
def PR := 4
def PS := 5
def QR := 5
def QS := 6
def RS := 15 / 2  -- RS is (15 / 2), i.e., 7.5
def area_PQR := 12

noncomputable def volume_tetrahedron (PQ PR PS QR QS RS area_PQR : ℝ) : ℝ := 1 / 3 * area_PQR * 4

theorem volume_of_tetrahedron_PQRS :
  volume_tetrahedron PQ PR PS QR QS RS area_PQR = 16 :=
by sorry

end NUMINAMATH_GPT_volume_of_tetrahedron_PQRS_l2207_220743


namespace NUMINAMATH_GPT_apples_eq_pears_l2207_220747

-- Define the conditions
def apples_eq_oranges (a o : ℕ) : Prop := 4 * a = 6 * o
def oranges_eq_pears (o p : ℕ) : Prop := 5 * o = 3 * p

-- The main problem statement
theorem apples_eq_pears (a o p : ℕ) (h1 : apples_eq_oranges a o) (h2 : oranges_eq_pears o p) :
  24 * a = 21 * p :=
sorry

end NUMINAMATH_GPT_apples_eq_pears_l2207_220747


namespace NUMINAMATH_GPT_total_notebooks_l2207_220784

theorem total_notebooks (num_boxes : ℕ) (parts_per_box : ℕ) (notebooks_per_part : ℕ) (h1 : num_boxes = 22)
  (h2 : parts_per_box = 6) (h3 : notebooks_per_part = 5) : 
  num_boxes * parts_per_box * notebooks_per_part = 660 := 
by
  sorry

end NUMINAMATH_GPT_total_notebooks_l2207_220784


namespace NUMINAMATH_GPT_binomial_evaluation_l2207_220729

-- Defining the binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.choose k

-- Theorem stating our problem
theorem binomial_evaluation : binomial 12 6 = 924 := 
by sorry

end NUMINAMATH_GPT_binomial_evaluation_l2207_220729


namespace NUMINAMATH_GPT_triangle_inequality_l2207_220788

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l2207_220788


namespace NUMINAMATH_GPT_ellipse_major_axis_length_l2207_220762

theorem ellipse_major_axis_length : 
  ∀ (x y : ℝ), x^2 + 2 * y^2 = 2 → 2 * Real.sqrt 2 = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_major_axis_length_l2207_220762


namespace NUMINAMATH_GPT_sum_of_squares_is_42_l2207_220764

variables (D T H : ℕ)

theorem sum_of_squares_is_42
  (h1 : 3 * D + T = 2 * H)
  (h2 : 2 * H^3 = 3 * D^3 + T^3)
  (coprime : Nat.gcd (Nat.gcd D T) H = 1) :
  (T^2 + D^2 + H^2 = 42) :=
sorry

end NUMINAMATH_GPT_sum_of_squares_is_42_l2207_220764


namespace NUMINAMATH_GPT_number_of_possible_triangles_with_side_5_not_shortest_l2207_220797

-- Define and prove the number of possible triangles (a, b, c) with a, b, c positive integers,
-- such that one side is length 5 and it is not the shortest side is 10.
theorem number_of_possible_triangles_with_side_5_not_shortest (a b c : ℕ) (h1: a + b > c) (h2: a + c > b) (h3: b + c > a) 
(h4: 0 < a) (h5: 0 < b) (h6: 0 < c) (h7: a = 5 ∨ b = 5 ∨ c = 5) (h8: ¬ (a < 5 ∧ b < 5 ∧ c < 5)) :
∃ n, n = 10 := 
sorry

end NUMINAMATH_GPT_number_of_possible_triangles_with_side_5_not_shortest_l2207_220797


namespace NUMINAMATH_GPT_joe_initial_paint_l2207_220793

noncomputable def total_paint (P : ℕ) : Prop :=
  let used_first_week := (1 / 4 : ℚ) * P
  let remaining_after_first := (3 / 4 : ℚ) * P
  let used_second_week := (1 / 6 : ℚ) * remaining_after_first
  let total_used := used_first_week + used_second_week
  total_used = 135

theorem joe_initial_paint (P : ℕ) (h : total_paint P) : P = 463 :=
sorry

end NUMINAMATH_GPT_joe_initial_paint_l2207_220793


namespace NUMINAMATH_GPT_intersection_point_of_lines_l2207_220751

theorem intersection_point_of_lines (x y : ℝ) :
  (2 * x - 3 * y = 3) ∧ (4 * x + 2 * y = 2) ↔ (x = 3/4) ∧ (y = -1/2) :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_of_lines_l2207_220751


namespace NUMINAMATH_GPT_set_contains_all_rationals_l2207_220799

variable (S : Set ℚ)
variable (h1 : (0 : ℚ) ∈ S)
variable (h2 : ∀ x ∈ S, x + 1 ∈ S ∧ x - 1 ∈ S)
variable (h3 : ∀ x ∈ S, x ≠ 0 → x ≠ 1 → 1 / (x * (x - 1)) ∈ S)

theorem set_contains_all_rationals : ∀ q : ℚ, q ∈ S :=
by
  sorry

end NUMINAMATH_GPT_set_contains_all_rationals_l2207_220799


namespace NUMINAMATH_GPT_determine_d_l2207_220778

theorem determine_d (m n d : ℝ) (p : ℝ) (hp : p = 0.6666666666666666) 
  (h1 : m = 3 * n + 5) (h2 : m + d = 3 * (n + p) + 5) : d = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_determine_d_l2207_220778


namespace NUMINAMATH_GPT_Mason_tables_needed_l2207_220736

theorem Mason_tables_needed
  (w_silverware_piece : ℕ := 4) 
  (n_silverware_piece_per_setting : ℕ := 3) 
  (w_plate : ℕ := 12) 
  (n_plates_per_setting : ℕ := 2) 
  (n_settings_per_table : ℕ := 8) 
  (n_backup_settings : ℕ := 20) 
  (total_weight : ℕ := 5040) : 
  ∃ (n_tables : ℕ), n_tables = 15 :=
by
  sorry

end NUMINAMATH_GPT_Mason_tables_needed_l2207_220736


namespace NUMINAMATH_GPT_rohan_house_rent_percentage_l2207_220755

noncomputable def house_rent_percentage (food_percentage entertainment_percentage conveyance_percentage salary savings: ℝ) : ℝ :=
  100 - (food_percentage + entertainment_percentage + conveyance_percentage + (savings / salary * 100))

-- Conditions
def food_percentage : ℝ := 40
def entertainment_percentage : ℝ := 10
def conveyance_percentage : ℝ := 10
def salary : ℝ := 10000
def savings : ℝ := 2000

-- Theorem
theorem rohan_house_rent_percentage :
  house_rent_percentage food_percentage entertainment_percentage conveyance_percentage salary savings = 20 := 
sorry

end NUMINAMATH_GPT_rohan_house_rent_percentage_l2207_220755


namespace NUMINAMATH_GPT_larger_of_two_numbers_l2207_220770

theorem larger_of_two_numbers (A B : ℕ) (hcf : A.gcd B = 47) (lcm_factors : A.lcm B = 47 * 49 * 11 * 13 * 4913) : max A B = 123800939 :=
sorry

end NUMINAMATH_GPT_larger_of_two_numbers_l2207_220770


namespace NUMINAMATH_GPT_exists_x_geq_zero_l2207_220779

theorem exists_x_geq_zero (h : ∀ x : ℝ, x^2 + x - 1 < 0) : ∃ x : ℝ, x^2 + x - 1 ≥ 0 :=
sorry

end NUMINAMATH_GPT_exists_x_geq_zero_l2207_220779


namespace NUMINAMATH_GPT_total_children_l2207_220773

variable (S C B T : ℕ)

theorem total_children (h1 : T < 19) 
                       (h2 : S = 3 * C) 
                       (h3 : B = S / 2) 
                       (h4 : T = B + S + 1) : 
                       T = 10 := 
  sorry

end NUMINAMATH_GPT_total_children_l2207_220773


namespace NUMINAMATH_GPT_tony_initial_amount_l2207_220787

-- Define the initial amount P
variable (P : ℝ)

-- Define the conditions
def initial_amount := P
def after_first_year := 1.20 * P
def after_half_taken := 0.60 * P
def after_second_year := 0.69 * P
def final_amount : ℝ := 690

-- State the theorem to prove
theorem tony_initial_amount : 
  (after_second_year P = final_amount) → (initial_amount P = 1000) :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_tony_initial_amount_l2207_220787


namespace NUMINAMATH_GPT_amount_after_two_years_l2207_220738

noncomputable def annual_increase (initial_amount : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial_amount * (1 + rate) ^ years

theorem amount_after_two_years :
  annual_increase 32000 (1/8) 2 = 40500 :=
by
  sorry

end NUMINAMATH_GPT_amount_after_two_years_l2207_220738


namespace NUMINAMATH_GPT_age_is_50_l2207_220741

-- Definitions only based on the conditions provided
def future_age (A: ℕ) := A + 5
def past_age (A: ℕ) := A - 5

theorem age_is_50 (A : ℕ) (h : 5 * future_age A - 5 * past_age A = A) : A = 50 := 
by 
  sorry  -- proof should be provided here

end NUMINAMATH_GPT_age_is_50_l2207_220741


namespace NUMINAMATH_GPT_max_jogs_l2207_220765

theorem max_jogs (x y z : ℕ) (h1 : 3 * x + 2 * y + 8 * z = 60) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) :
  z ≤ 6 := 
sorry

end NUMINAMATH_GPT_max_jogs_l2207_220765


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l2207_220702

theorem hyperbola_eccentricity (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (h₂ : ∀ x : ℝ, y = (3 / 4) * x → y = (b / a) * x) : 
  (b = (3 / 4) * a) → (e = 5 / 4) := 
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l2207_220702


namespace NUMINAMATH_GPT_missing_number_l2207_220749

theorem missing_number (x : ℤ) : 1234562 - 12 * x * 2 = 1234490 ↔ x = 3 :=
by
sorry

end NUMINAMATH_GPT_missing_number_l2207_220749


namespace NUMINAMATH_GPT_intersection_points_count_l2207_220794

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem intersection_points_count :
  ∃ y1 y2 : ℝ, y1 ≠ y2 ∧ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = g x1 ∧ f x2 = g x2) := sorry

end NUMINAMATH_GPT_intersection_points_count_l2207_220794


namespace NUMINAMATH_GPT_johns_height_in_feet_l2207_220798

def initial_height := 66 -- John's initial height in inches
def growth_rate := 2      -- Growth rate in inches per month
def growth_duration := 3  -- Growth duration in months
def inches_per_foot := 12 -- Conversion factor from inches to feet

def total_growth : ℕ := growth_rate * growth_duration

def final_height_in_inches : ℕ := initial_height + total_growth

-- Now, proof that the final height in feet is 6
theorem johns_height_in_feet : (final_height_in_inches / inches_per_foot) = 6 :=
by {
  -- We would provide the detailed proof here
  sorry
}

end NUMINAMATH_GPT_johns_height_in_feet_l2207_220798


namespace NUMINAMATH_GPT_fraction_identity_l2207_220707

theorem fraction_identity
  (m : ℝ)
  (h : (m - 1) / m = 3) : (m^2 + 1) / m^2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_identity_l2207_220707


namespace NUMINAMATH_GPT_no_nat_solutions_l2207_220732

theorem no_nat_solutions (x y z : ℕ) : x^2 + y^2 + z^2 ≠ 2 * x * y * z :=
sorry

end NUMINAMATH_GPT_no_nat_solutions_l2207_220732


namespace NUMINAMATH_GPT_total_potatoes_brought_home_l2207_220754

def number_of_potatoes_each : ℕ := 8

theorem total_potatoes_brought_home (jane_potatoes mom_potatoes dad_potatoes : ℕ) :
  jane_potatoes = number_of_potatoes_each →
  mom_potatoes = number_of_potatoes_each →
  dad_potatoes = number_of_potatoes_each →
  jane_potatoes + mom_potatoes + dad_potatoes = 24 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end NUMINAMATH_GPT_total_potatoes_brought_home_l2207_220754


namespace NUMINAMATH_GPT_relationship_between_m_and_n_l2207_220748

variable {X_1 X_2 k m n : ℝ}

-- Given conditions
def inverse_proportional_points (X_1 X_2 k : ℝ) (m n : ℝ) : Prop :=
  m = k / X_1 ∧ n = k / X_2 ∧ k > 0 ∧ X_1 < X_2

theorem relationship_between_m_and_n (h : inverse_proportional_points X_1 X_2 k m n) : m > n :=
by
  -- Insert proof here, skipping with sorry
  sorry

end NUMINAMATH_GPT_relationship_between_m_and_n_l2207_220748


namespace NUMINAMATH_GPT_number_of_herds_l2207_220700

-- Definitions from the conditions
def total_sheep : ℕ := 60
def sheep_per_herd : ℕ := 20

-- The statement to prove
theorem number_of_herds : total_sheep / sheep_per_herd = 3 := by
  sorry

end NUMINAMATH_GPT_number_of_herds_l2207_220700


namespace NUMINAMATH_GPT_zoe_total_cost_l2207_220706

theorem zoe_total_cost 
  (app_cost : ℕ)
  (monthly_cost : ℕ)
  (item_cost : ℕ)
  (feature_cost : ℕ)
  (months_played : ℕ)
  (h1 : app_cost = 5)
  (h2 : monthly_cost = 8)
  (h3 : item_cost = 10)
  (h4 : feature_cost = 12)
  (h5 : months_played = 2) :
  app_cost + (months_played * monthly_cost) + item_cost + feature_cost = 43 := 
by 
  sorry

end NUMINAMATH_GPT_zoe_total_cost_l2207_220706


namespace NUMINAMATH_GPT_sin_range_l2207_220742

theorem sin_range (p : Prop) (q : Prop) :
  (¬ ∃ x : ℝ, Real.sin x = 3/2) → (∀ x : ℝ, x^2 - 4 * x + 5 > 0) → (¬p ∧ q) :=
by
  sorry

end NUMINAMATH_GPT_sin_range_l2207_220742


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2207_220727

theorem solution_set_of_inequality :
  { x : ℝ | -x^2 + 3*x + 4 > 0 } = { x : ℝ | -1 < x ∧ x < 4 } := 
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2207_220727


namespace NUMINAMATH_GPT_q_can_complete_work_in_30_days_l2207_220704

theorem q_can_complete_work_in_30_days (W_p W_q W_r : ℝ)
  (h1 : W_p = W_q + W_r)
  (h2 : W_p + W_q = 1/10)
  (h3 : W_r = 1/30) :
  1 / W_q = 30 :=
by
  -- Note: You can add proof here, but it's not required in the task.
  sorry

end NUMINAMATH_GPT_q_can_complete_work_in_30_days_l2207_220704


namespace NUMINAMATH_GPT_div_pow_eq_l2207_220724

theorem div_pow_eq {a : ℝ} (h : a ≠ 0) : a^3 / a^2 = a :=
sorry

end NUMINAMATH_GPT_div_pow_eq_l2207_220724


namespace NUMINAMATH_GPT_time_taken_by_x_alone_l2207_220758

theorem time_taken_by_x_alone 
  (W : ℝ)
  (Rx Ry Rz : ℝ)
  (h1 : Ry = W / 24)
  (h2 : Ry + Rz = W / 6)
  (h3 : Rx + Rz = W / 4) :
  (W / Rx) = 16 :=
by
  sorry

end NUMINAMATH_GPT_time_taken_by_x_alone_l2207_220758


namespace NUMINAMATH_GPT_kendra_more_buttons_l2207_220792

theorem kendra_more_buttons {K M S : ℕ} (hM : M = 8) (hS : S = 22) (hHalfK : S = K / 2) :
  K - 5 * M = 4 :=
by
  sorry

end NUMINAMATH_GPT_kendra_more_buttons_l2207_220792


namespace NUMINAMATH_GPT_pure_ghee_percentage_l2207_220710

theorem pure_ghee_percentage (Q : ℝ) (P : ℝ) (H1 : Q = 10) (H2 : (P / 100) * Q + 10 = 0.80 * (Q + 10)) :
  P = 60 :=
sorry

end NUMINAMATH_GPT_pure_ghee_percentage_l2207_220710


namespace NUMINAMATH_GPT_correct_diagram_is_B_l2207_220781

-- Define the diagrams and their respected angles
def sector_angle_A : ℝ := 90
def sector_angle_B : ℝ := 135
def sector_angle_C : ℝ := 180

-- Define the target central angle for one third of the circle
def target_angle : ℝ := 120

-- The proof statement that Diagram B is the correct diagram with the sector angle closest to one third of the circle (120 degrees)
theorem correct_diagram_is_B (A B C : Prop) :
  (B = (sector_angle_A < target_angle ∧ target_angle < sector_angle_B)) := 
sorry

end NUMINAMATH_GPT_correct_diagram_is_B_l2207_220781


namespace NUMINAMATH_GPT_solution_set_f_lt_zero_a_two_solution_set_f_gt_zero_l2207_220703

-- Given function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := x^2 - (a - 1) * x - a

-- Problem 1: for a = 2, solution to f(x) < 0
theorem solution_set_f_lt_zero_a_two :
  { x : ℝ | f x 2 < 0 } = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

-- Problem 2: for any a in ℝ, solution to f(x) > 0
theorem solution_set_f_gt_zero (a : ℝ) :
  { x : ℝ | f x a > 0 } =
  if a > -1 then
    {x : ℝ | x < -1} ∪ {x : ℝ | x > a}
  else if a = -1 then
    {x : ℝ | x ≠ -1}
  else
    {x : ℝ | x < a} ∪ {x : ℝ | x > -1} :=
sorry

end NUMINAMATH_GPT_solution_set_f_lt_zero_a_two_solution_set_f_gt_zero_l2207_220703


namespace NUMINAMATH_GPT_cubic_eq_solutions_l2207_220789

theorem cubic_eq_solutions (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : ∀ x, x^3 + a * x^2 + b * x + c = 0 → (x = a ∨ x = -b ∨ x = c)) : (a, b, c) = (1, -1, -1) := 
by {
  -- Convert solution steps into a proof
  sorry
}

end NUMINAMATH_GPT_cubic_eq_solutions_l2207_220789


namespace NUMINAMATH_GPT_find_m_value_l2207_220756

theorem find_m_value (m : Real) (h : (3 * m + 8) * (m - 3) = 72) : m = (1 + Real.sqrt 1153) / 6 :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l2207_220756


namespace NUMINAMATH_GPT_value_of_p_l2207_220752

theorem value_of_p (m n p : ℝ) (h1 : m = 6 * n + 5) (h2 : m + 2 = 6 * (n + p) + 5) : p = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_p_l2207_220752


namespace NUMINAMATH_GPT_time_left_to_room_l2207_220750

theorem time_left_to_room (total_time minutes_to_gate minutes_to_building : ℕ) 
  (h1 : total_time = 30) 
  (h2 : minutes_to_gate = 15) 
  (h3 : minutes_to_building = 6) : 
  total_time - (minutes_to_gate + minutes_to_building) = 9 :=
by 
  sorry

end NUMINAMATH_GPT_time_left_to_room_l2207_220750


namespace NUMINAMATH_GPT_strictly_increasing_interval_l2207_220721

noncomputable def f (x : ℝ) : ℝ :=
  Real.logb (1/3) (x^2 - 4 * x + 3)

theorem strictly_increasing_interval : ∀ x y : ℝ, x < 1 → y < 1 → x < y → f x < f y :=
by
  sorry

end NUMINAMATH_GPT_strictly_increasing_interval_l2207_220721


namespace NUMINAMATH_GPT_poly_diff_independent_of_x_l2207_220790

theorem poly_diff_independent_of_x (x y: ℤ) (m n : ℤ) 
  (h1 : (1 - n = 0)) 
  (h2 : (m + 3 = 0)) :
  n - m = 4 := by
  sorry

end NUMINAMATH_GPT_poly_diff_independent_of_x_l2207_220790


namespace NUMINAMATH_GPT_unique_three_digit_multiple_of_66_ending_in_4_l2207_220705

theorem unique_three_digit_multiple_of_66_ending_in_4 :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 66 = 0 ∧ n % 10 = 4 := sorry

end NUMINAMATH_GPT_unique_three_digit_multiple_of_66_ending_in_4_l2207_220705


namespace NUMINAMATH_GPT_miles_flown_on_thursday_l2207_220766
-- Importing the necessary library

-- Defining the problem conditions and the proof goal
theorem miles_flown_on_thursday (x : ℕ) : 
  (∀ y, (3 * (1134 + y) = 7827) → y = x) → x = 1475 :=
by
  intro h
  specialize h 1475
  sorry

end NUMINAMATH_GPT_miles_flown_on_thursday_l2207_220766


namespace NUMINAMATH_GPT_certain_number_less_32_l2207_220785

theorem certain_number_less_32 (x : ℤ) (h : x - 48 = 22) : x - 32 = 38 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_less_32_l2207_220785


namespace NUMINAMATH_GPT_net_rate_of_pay_l2207_220786

theorem net_rate_of_pay
  (hours_travelled : ℕ)
  (speed : ℕ)
  (fuel_efficiency : ℕ)
  (pay_per_mile : ℝ)
  (price_per_gallon : ℝ)
  (net_rate_of_pay : ℝ) :
  hours_travelled = 3 →
  speed = 50 →
  fuel_efficiency = 25 →
  pay_per_mile = 0.60 →
  price_per_gallon = 2.50 →
  net_rate_of_pay = 25 := by
  sorry

end NUMINAMATH_GPT_net_rate_of_pay_l2207_220786


namespace NUMINAMATH_GPT_total_people_on_hike_l2207_220701

-- Definitions of the conditions
def n_cars : ℕ := 3
def n_people_per_car : ℕ := 4
def n_taxis : ℕ := 6
def n_people_per_taxi : ℕ := 6
def n_vans : ℕ := 2
def n_people_per_van : ℕ := 5

-- Statement of the problem
theorem total_people_on_hike : 
  n_cars * n_people_per_car + n_taxis * n_people_per_taxi + n_vans * n_people_per_van = 58 :=
by sorry

end NUMINAMATH_GPT_total_people_on_hike_l2207_220701


namespace NUMINAMATH_GPT_selection_methods_count_l2207_220776

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem selection_methods_count :
  let females := 8
  let males := 4
  (binomial females 2 * binomial males 1) + (binomial females 1 * binomial males 2) = 112 :=
by
  sorry

end NUMINAMATH_GPT_selection_methods_count_l2207_220776


namespace NUMINAMATH_GPT_book_organizing_activity_l2207_220775

theorem book_organizing_activity (x : ℕ) (h₁ : x > 0):
  (80 : ℝ) / (x + 5 : ℝ) = (70 : ℝ) / (x : ℝ) :=
sorry

end NUMINAMATH_GPT_book_organizing_activity_l2207_220775


namespace NUMINAMATH_GPT_find_n_divides_2n_plus_2_l2207_220713

theorem find_n_divides_2n_plus_2 :
  ∃ n : ℕ, (100 ≤ n ∧ n ≤ 1997 ∧ n ∣ (2 * n + 2)) ∧ n = 946 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_n_divides_2n_plus_2_l2207_220713


namespace NUMINAMATH_GPT_angle_sum_x_y_l2207_220712

theorem angle_sum_x_y 
  (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) (x : ℝ) (y : ℝ) 
  (hA : angle_A = 34) (hB : angle_B = 80) (hC : angle_C = 30) 
  (hexagon_property : ∀ A B x y : ℝ, A + B + 360 - x + 90 + 120 - y = 720) :
  x + y = 36 :=
by
  sorry

end NUMINAMATH_GPT_angle_sum_x_y_l2207_220712


namespace NUMINAMATH_GPT_percentage_subtraction_l2207_220714

variable (a b x m : ℝ) (p : ℝ)

-- Conditions extracted from the problem.
def ratio_a_to_b : Prop := a / b = 4 / 5
def definition_of_x : Prop := x = 1.75 * a
def definition_of_m : Prop := m = b * (1 - p / 100)
def value_m_div_x : Prop := m / x = 0.14285714285714285

-- The proof problem in the form of a Lean statement.
theorem percentage_subtraction 
  (h1 : ratio_a_to_b a b)
  (h2 : definition_of_x a x)
  (h3 : definition_of_m b m p)
  (h4 : value_m_div_x x m) : p = 80 := 
sorry

end NUMINAMATH_GPT_percentage_subtraction_l2207_220714


namespace NUMINAMATH_GPT_complement_U_A_l2207_220759

open Set

def U : Set ℤ := univ
def A : Set ℤ := { x | x^2 - x - 2 ≥ 0 }

theorem complement_U_A :
  (U \ A) = { 0, 1 } := by
  sorry

end NUMINAMATH_GPT_complement_U_A_l2207_220759


namespace NUMINAMATH_GPT_total_cost_is_21_l2207_220709

-- Definitions of the costs
def cost_almond_croissant : Float := 4.50
def cost_salami_and_cheese_croissant : Float := 4.50
def cost_plain_croissant : Float := 3.00
def cost_focaccia : Float := 4.00
def cost_latte : Float := 2.50

-- Theorem stating the total cost
theorem total_cost_is_21 :
  (cost_almond_croissant + cost_salami_and_cheese_croissant) + (2 * cost_latte) + cost_plain_croissant + cost_focaccia = 21.00 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_is_21_l2207_220709


namespace NUMINAMATH_GPT_bisection_second_iteration_value_l2207_220725

def f (x : ℝ) : ℝ := x^3 + 3 * x - 1

theorem bisection_second_iteration_value :
  f 0.25 = -0.234375 :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_bisection_second_iteration_value_l2207_220725


namespace NUMINAMATH_GPT_kelly_sony_games_solution_l2207_220719

def kelly_sony_games_left (n g : Nat) : Nat :=
  n - g

theorem kelly_sony_games_solution (initial : Nat) (given_away : Nat) 
  (h_initial : initial = 132)
  (h_given_away : given_away = 101) :
  kelly_sony_games_left initial given_away = 31 :=
by
  rw [h_initial, h_given_away]
  unfold kelly_sony_games_left
  norm_num

end NUMINAMATH_GPT_kelly_sony_games_solution_l2207_220719


namespace NUMINAMATH_GPT_interest_rate_difference_correct_l2207_220795

noncomputable def interest_rate_difference (P r R T : ℝ) :=
  let I := P * r * T
  let I' := P * R * T
  (I' - I) = 140

theorem interest_rate_difference_correct:
  ∀ (P r R T : ℝ),
  P = 1000 ∧ T = 7 ∧ interest_rate_difference P r R T →
  (R - r) = 0.02 :=
by
  intros P r R T h
  sorry

end NUMINAMATH_GPT_interest_rate_difference_correct_l2207_220795


namespace NUMINAMATH_GPT_cynthia_more_miles_l2207_220733

open Real

noncomputable def david_speed : ℝ := 55 / 5
noncomputable def cynthia_speed : ℝ := david_speed + 3

theorem cynthia_more_miles (t : ℝ) (ht : t = 5) :
  (cynthia_speed * t) - (david_speed * t) = 15 :=
by
  sorry

end NUMINAMATH_GPT_cynthia_more_miles_l2207_220733
