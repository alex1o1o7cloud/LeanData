import Mathlib

namespace simplify_expression_l1563_156359

theorem simplify_expression (x y : ℝ) :
  (2 * x + 25) + (150 * x + 40) + (5 * y + 10) = 152 * x + 5 * y + 75 :=
by sorry

end simplify_expression_l1563_156359


namespace binary_arithmetic_correct_l1563_156338

theorem binary_arithmetic_correct :
  (2^3 + 2^2 + 2^0) + (2^2 + 2^1 + 2^0) - (2^3 + 2^2 + 2^1) + (2^3 + 2^0) + (2^3 + 2^1) = 2^4 + 2^3 + 2^0 :=
by sorry

end binary_arithmetic_correct_l1563_156338


namespace union_prob_inconsistency_l1563_156392

noncomputable def p_a : ℚ := 2/15
noncomputable def p_b : ℚ := 4/15
noncomputable def p_b_given_a : ℚ := 3

theorem union_prob_inconsistency : p_a + p_b - p_b_given_a * p_a = 0 → false := by
  sorry

end union_prob_inconsistency_l1563_156392


namespace div_count_27n5_l1563_156341

theorem div_count_27n5 
  (n : ℕ) 
  (h : (120 * n^3).divisors.card = 120) 
  : (27 * n^5).divisors.card = 324 :=
sorry

end div_count_27n5_l1563_156341


namespace total_penalty_kicks_l1563_156371

theorem total_penalty_kicks (total_players : ℕ) (goalies : ℕ) (hoop_challenges : ℕ)
  (h_total : total_players = 25) (h_goalies : goalies = 5) (h_hoop_challenges : hoop_challenges = 10) :
  (goalies * (total_players - 1)) = 120 :=
by
  sorry

end total_penalty_kicks_l1563_156371


namespace balloons_difference_l1563_156334

theorem balloons_difference (yours friends : ℝ) (hyours : yours = -7) (hfriends : friends = 4.5) :
  friends - yours = 11.5 :=
by
  rw [hyours, hfriends]
  sorry

end balloons_difference_l1563_156334


namespace oranges_per_box_l1563_156345

theorem oranges_per_box (total_oranges : ℝ) (total_boxes : ℝ) (h1 : total_oranges = 26500) (h2 : total_boxes = 2650) : 
  total_oranges / total_boxes = 10 :=
by 
  sorry

end oranges_per_box_l1563_156345


namespace pizza_party_l1563_156308

theorem pizza_party (boys girls : ℕ) :
  (7 * boys + 3 * girls ≤ 59) ∧ (6 * boys + 2 * girls ≥ 49) ∧ (boys + girls ≤ 10) → 
  boys = 8 ∧ girls = 1 := 
by sorry

end pizza_party_l1563_156308


namespace probability_neither_red_nor_purple_l1563_156384

def total_balls : ℕ := 60
def white_balls : ℕ := 22
def green_balls : ℕ := 18
def yellow_balls : ℕ := 8
def red_balls : ℕ := 5
def purple_balls : ℕ := 7

theorem probability_neither_red_nor_purple : 
  (total_balls - (red_balls + purple_balls)) / total_balls = 4 / 5 :=
by sorry

end probability_neither_red_nor_purple_l1563_156384


namespace geometric_progression_complex_l1563_156375

theorem geometric_progression_complex (a b c m : ℂ) (r : ℂ) (hr : r ≠ 0) 
    (h1 : a = r) (h2 : b = r^2) (h3 : c = r^3) 
    (h4 : a / (1 - b) = m) (h5 : b / (1 - c) = m) (h6 : c / (1 - a) = m) : 
    ∃ m : ℂ, ∀ a b c : ℂ, ∃ r : ℂ, a = r ∧ b = r^2 ∧ c = r^3 
    ∧ r ≠ 0 
    ∧ (a / (1 - b) = m) 
    ∧ (b / (1 - c) = m) 
    ∧ (c / (1 - a) = m) := 
sorry

end geometric_progression_complex_l1563_156375


namespace cumulative_distribution_X_maximized_expected_score_l1563_156372

noncomputable def distribution_X (p_A : ℝ) (p_B : ℝ) : (ℝ × ℝ × ℝ) :=
(1 - p_A, p_A * (1 - p_B), p_A * p_B)

def expected_score (p_A : ℝ) (p_B : ℝ) (s_A : ℝ) (s_B : ℝ) : ℝ :=
0 * (1 - p_A) + s_A * (p_A * (1 - p_B)) + (s_A + s_B) * (p_A * p_B)

theorem cumulative_distribution_X :
  distribution_X 0.8 0.6 = (0.2, 0.32, 0.48) :=
sorry

theorem maximized_expected_score :
  expected_score 0.8 0.6 20 80 < expected_score 0.6 0.8 80 20 :=
sorry

end cumulative_distribution_X_maximized_expected_score_l1563_156372


namespace cheryl_same_color_probability_l1563_156320

/-- Defines the probability of Cheryl picking 3 marbles of the same color from the given box setup. -/
def probability_cheryl_picks_same_color : ℚ :=
  let total_ways := (Nat.choose 9 3) * (Nat.choose 6 3) * (Nat.choose 3 3)
  let favorable_ways := 3 * (Nat.choose 6 3)
  (favorable_ways : ℚ) / (total_ways : ℚ)

/-- Theorem stating the probability that Cheryl picks 3 marbles of the same color is 1/28. -/
theorem cheryl_same_color_probability :
  probability_cheryl_picks_same_color = 1 / 28 :=
by
  sorry

end cheryl_same_color_probability_l1563_156320


namespace fabric_difference_fabric_total_l1563_156381

noncomputable def fabric_used_coat : ℝ := 1.55
noncomputable def fabric_used_pants : ℝ := 1.05

theorem fabric_difference : fabric_used_coat - fabric_used_pants = 0.5 :=
by
  sorry

theorem fabric_total : fabric_used_coat + fabric_used_ppants = 2.6 :=
by
  sorry

end fabric_difference_fabric_total_l1563_156381


namespace min_value_512_l1563_156367

noncomputable def min_value (a b c d e f g h : ℝ) : ℝ :=
  (2 * a * e)^2 + (2 * b * f)^2 + (2 * c * g)^2 + (2 * d * h)^2

theorem min_value_512 
  (a b c d e f g h : ℝ)
  (H1 : a * b * c * d = 8)
  (H2 : e * f * g * h = 16) : 
  ∃ (min_val : ℝ), min_val = 512 ∧ min_value a b c d e f g h = min_val :=
sorry

end min_value_512_l1563_156367


namespace smallest_sum_of_factors_of_12_factorial_l1563_156307

theorem smallest_sum_of_factors_of_12_factorial :
  ∃ (x y z w : Nat), x * y * z * w = Nat.factorial 12 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ x + y + z + w = 147 :=
by
  sorry

end smallest_sum_of_factors_of_12_factorial_l1563_156307


namespace find_tricycles_l1563_156321

theorem find_tricycles (b t w : ℕ) 
  (sum_children : b + t + w = 10)
  (sum_wheels : 2 * b + 3 * t = 26) :
  t = 6 :=
by sorry

end find_tricycles_l1563_156321


namespace find_point_N_l1563_156391

theorem find_point_N 
  (M N : ℝ × ℝ) 
  (MN_length : Real.sqrt (((N.1 - M.1) ^ 2) + ((N.2 - M.2) ^ 2)) = 4)
  (MN_parallel_y_axis : N.1 = M.1)
  (M_coord : M = (-1, 2)) 
  : (N = (-1, 6)) ∨ (N = (-1, -2)) :=
sorry

end find_point_N_l1563_156391


namespace parallelogram_area_l1563_156326

def base := 12 -- in meters
def height := 6 -- in meters

theorem parallelogram_area : base * height = 72 := by
  sorry

end parallelogram_area_l1563_156326


namespace workers_contribution_l1563_156366

theorem workers_contribution (W C : ℕ) (h1 : W * C = 300000) (h2 : W * (C + 50) = 360000) : W = 1200 :=
by
  sorry

end workers_contribution_l1563_156366


namespace domain_of_log_base_5_range_of_3_pow_neg_l1563_156352

theorem domain_of_log_base_5 (x : ℝ) : (1 - x > 0) -> (x < 1) :=
sorry

theorem range_of_3_pow_neg (y : ℝ) : (∃ x : ℝ, y = 3 ^ (-x)) -> (y > 0) :=
sorry

end domain_of_log_base_5_range_of_3_pow_neg_l1563_156352


namespace proof_problem_l1563_156333

def from_base (b : ℕ) (digits : List ℕ) : ℕ :=
digits.foldr (λ (d acc) => d + b * acc) 0

def problem : Prop :=
  let a := from_base 8 [2, 3, 4, 5] -- 2345 base 8
  let b := from_base 5 [1, 4, 0]    -- 140 base 5
  let c := from_base 4 [1, 0, 3, 2] -- 1032 base 4
  let d := from_base 8 [2, 9, 1, 0] -- 2910 base 8
  let result := (a / b + c - d : ℤ)
  result = -1502

theorem proof_problem : problem :=
by
  sorry

end proof_problem_l1563_156333


namespace brocard_inequality_part_a_brocard_inequality_part_b_l1563_156315

variable (α β γ φ : ℝ)

theorem brocard_inequality_part_a (h_sum_angles : α + β + γ = π) (h_brocard : 0 < φ ∧ φ < π/2) :
  φ^3 ≤ (α - φ) * (β - φ) * (γ - φ) := 
sorry

theorem brocard_inequality_part_b (h_sum_angles : α + β + γ = π) (h_brocard : 0 < φ ∧ φ < π/2) :
  8 * φ^3 ≤ α * β * γ := 
sorry

end brocard_inequality_part_a_brocard_inequality_part_b_l1563_156315


namespace max_n_for_polynomial_l1563_156340

theorem max_n_for_polynomial (P : Polynomial ℤ) (hdeg : P.degree = 2022) :
  ∃ n ≤ 2022, ∀ {a : Fin n → ℤ}, 
    (∀ i, P.eval (a i) = i) ↔ n = 2022 :=
by sorry

end max_n_for_polynomial_l1563_156340


namespace exists_x_for_ax2_plus_2x_plus_a_lt_0_l1563_156350

theorem exists_x_for_ax2_plus_2x_plus_a_lt_0 (a : ℝ) : (∃ x : ℝ, a * x^2 + 2 * x + a < 0) ↔ a < 1 :=
by
  sorry

end exists_x_for_ax2_plus_2x_plus_a_lt_0_l1563_156350


namespace smallest_n_for_cubic_sum_inequality_l1563_156332

theorem smallest_n_for_cubic_sum_inequality :
  ∃ n : ℕ, (∀ (a b c : ℕ), (a + b + c) ^ 3 ≤ n * (a ^ 3 + b ^ 3 + c ^ 3)) ∧ n = 9 :=
sorry

end smallest_n_for_cubic_sum_inequality_l1563_156332


namespace determine_x_l1563_156300

variable {m x : ℝ}

theorem determine_x (h₁ : m > 25)
    (h₂ : ((m / 100) * m = (m - 20) / 100 * (m + x))) : 
    x = 20 * m / (m - 20) := 
sorry

end determine_x_l1563_156300


namespace minimum_apples_collected_l1563_156353

-- Anya, Vanya, Dania, Sanya, and Tanya each collected an integer percentage of the total number of apples,
-- with all these percentages distinct and greater than zero.
-- Prove that the minimum total number of apples is 20.

theorem minimum_apples_collected :
  ∃ (n : ℕ), (∀ (a v d s t : ℕ), 
    1 ≤ a ∧ 1 ≤ v ∧ 1 ≤ d ∧ 1 ≤ s ∧ 1 ≤ t ∧
    a ≠ v ∧ a ≠ d ∧ a ≠ s ∧ a ≠ t ∧ 
    v ≠ d ∧ v ≠ s ∧ v ≠ t ∧ 
    d ≠ s ∧ d ≠ t ∧ 
    s ≠ t ∧
    a + v + d + s + t = 100) →
  n ≥ 20 :=
by 
  sorry

end minimum_apples_collected_l1563_156353


namespace sweet_treats_per_student_l1563_156362

theorem sweet_treats_per_student :
  let cookies := 20
  let cupcakes := 25
  let brownies := 35
  let students := 20
  (cookies + cupcakes + brownies) / students = 4 :=
by 
  sorry

end sweet_treats_per_student_l1563_156362


namespace midpoint_product_l1563_156370

-- Defining the endpoints of the line segment
def x1 : ℤ := 4
def y1 : ℤ := 7
def x2 : ℤ := -8
def y2 : ℤ := 9

-- Proof goal: show that the product of the coordinates of the midpoint is -16
theorem midpoint_product : ((x1 + x2) / 2) * ((y1 + y2) / 2) = -16 := 
by sorry

end midpoint_product_l1563_156370


namespace pos_int_solns_to_eq_l1563_156379

open Int

theorem pos_int_solns_to_eq (x y z : ℤ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  x^2 + y^2 - z^2 = 9 - 2 * x * y ↔ 
    (x, y, z) = (5, 0, 4) ∨ (x, y, z) = (4, 1, 4) ∨ (x, y, z) = (3, 2, 4) ∨ 
    (x, y, z) = (2, 3, 4) ∨ (x, y, z) = (1, 4, 4) ∨ (x, y, z) = (0, 5, 4) ∨ 
    (x, y, z) = (3, 0, 0) ∨ (x, y, z) = (2, 1, 0) ∨ (x, y, z) = (1, 2, 0) ∨ 
    (x, y, z) = (0, 3, 0) :=
by sorry

end pos_int_solns_to_eq_l1563_156379


namespace div_equiv_l1563_156331

theorem div_equiv : (0.75 / 25) = (7.5 / 250) :=
by
  sorry

end div_equiv_l1563_156331


namespace dolls_completion_time_l1563_156312

def time_to_complete_dolls (craft_time_per_doll break_time_per_three_dolls total_dolls start_time : Nat) : Nat :=
  let total_craft_time := craft_time_per_doll * total_dolls
  let total_breaks := (total_dolls / 3) * break_time_per_three_dolls
  let total_time := total_craft_time + total_breaks
  (start_time + total_time) % 1440 -- 1440 is the number of minutes in a day

theorem dolls_completion_time :
  time_to_complete_dolls 105 30 10 600 = 300 := -- 600 is 10:00 AM in minutes, 300 is 5:00 AM in minutes
sorry

end dolls_completion_time_l1563_156312


namespace multiple_people_sharing_carriage_l1563_156324

theorem multiple_people_sharing_carriage (x : ℝ) : 
  (x / 3) + 2 = (x - 9) / 2 :=
sorry

end multiple_people_sharing_carriage_l1563_156324


namespace factor_tree_X_value_l1563_156396

def H : ℕ := 2 * 5
def J : ℕ := 3 * 7
def F : ℕ := 7 * H
def G : ℕ := 11 * J
def X : ℕ := F * G

theorem factor_tree_X_value : X = 16170 := by
  sorry

end factor_tree_X_value_l1563_156396


namespace rooks_non_attacking_kings_non_attacking_bishops_non_attacking_knights_non_attacking_queens_non_attacking_l1563_156363

-- Define the problem conditions: number of ways to place two same-color rooks that do not attack each other.
def num_ways_rooks : ℕ := 1568
theorem rooks_non_attacking : ∃ (n : ℕ), n = num_ways_rooks := by
  sorry

-- Define the problem conditions: number of ways to place two same-color kings that do not attack each other.
def num_ways_kings : ℕ := 1806
theorem kings_non_attacking : ∃ (n : ℕ), n = num_ways_kings := by
  sorry

-- Define the problem conditions: number of ways to place two same-color bishops that do not attack each other.
def num_ways_bishops : ℕ := 1736
theorem bishops_non_attacking : ∃ (n : ℕ), n = num_ways_bishops := by
  sorry

-- Define the problem conditions: number of ways to place two same-color knights that do not attack each other.
def num_ways_knights : ℕ := 1848
theorem knights_non_attacking : ∃ (n : ℕ), n = num_ways_knights := by
  sorry

-- Define the problem conditions: number of ways to place two same-color queens that do not attack each other.
def num_ways_queens : ℕ := 1288
theorem queens_non_attacking : ∃ (n : ℕ), n = num_ways_queens := by
  sorry

end rooks_non_attacking_kings_non_attacking_bishops_non_attacking_knights_non_attacking_queens_non_attacking_l1563_156363


namespace unique_ordered_triple_l1563_156314

theorem unique_ordered_triple (a b c : ℕ) (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_eq : a^3 + b^3 + c^3 + 648 = (a + b + c)^3) :
  (a, b, c) = (3, 3, 3) ∨ (a, b, c) = (3, 3, 3) ∨ (a, b, c) = (3, 3, 3) :=
sorry

end unique_ordered_triple_l1563_156314


namespace Dima_floor_l1563_156389

theorem Dima_floor (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 9)
  (h2 : 60 = (n - 1))
  (h3 : 70 = (n - 1) / (n - 1) * 60 + (n - n / 2) * 2 * 60)
  (h4 : ∀ m : ℕ, 1 ≤ m ∧ m ≤ 9 → (5 * n = 6 * m + 1) → (n = 7 ∧ m = 6)) :
  n = 7 :=
by
  sorry

end Dima_floor_l1563_156389


namespace twice_x_minus_3_gt_4_l1563_156310

theorem twice_x_minus_3_gt_4 (x : ℝ) : 2 * x - 3 > 4 :=
sorry

end twice_x_minus_3_gt_4_l1563_156310


namespace total_strings_correct_l1563_156348

-- Definitions based on conditions
def num_ukuleles : ℕ := 2
def num_guitars : ℕ := 4
def num_violins : ℕ := 2
def strings_per_ukulele : ℕ := 4
def strings_per_guitar : ℕ := 6
def strings_per_violin : ℕ := 4

-- Total number of strings
def total_strings : ℕ := num_ukuleles * strings_per_ukulele +
                         num_guitars * strings_per_guitar +
                         num_violins * strings_per_violin

-- The proof statement
theorem total_strings_correct : total_strings = 40 :=
by
  -- Proof omitted.
  sorry

end total_strings_correct_l1563_156348


namespace max_value_f_l1563_156368

noncomputable def f (x : ℝ) : ℝ := Real.sin (2*x) - 2 * Real.sqrt 3 * (Real.sin x)^2

theorem max_value_f : ∃ x : ℝ, f x = 2 - Real.sqrt 3 :=
  sorry

end max_value_f_l1563_156368


namespace loan_amount_l1563_156354

theorem loan_amount
  (P : ℝ)
  (SI : ℝ := 704)
  (R : ℝ := 8)
  (T : ℝ := 8)
  (h : SI = (P * R * T) / 100) : P = 1100 :=
by
  sorry

end loan_amount_l1563_156354


namespace perfect_square_trinomial_t_l1563_156349

theorem perfect_square_trinomial_t (a b t : ℝ) :
  (∃ (x y : ℝ), x = a ∧ y = 2 * b ∧ a^2 + (2 * t - 1) * a * b + 4 * b^2 = (x + y)^2) →
  (t = 5 / 2 ∨ t = -3 / 2) :=
by
  sorry

end perfect_square_trinomial_t_l1563_156349


namespace max_values_of_f_smallest_positive_period_of_f_intervals_where_f_is_monotonically_increasing_l1563_156397

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (x / 2) + Real.sqrt 3 * Real.cos (x / 2)

theorem max_values_of_f (k : ℤ) : 
  ∃ x, f x = 2 ∧ x = 4 * (k : ℝ) * Real.pi - (2 * Real.pi / 3) := 
sorry

theorem smallest_positive_period_of_f : 
  ∃ T, T = 4 * Real.pi := 
sorry

theorem intervals_where_f_is_monotonically_increasing (k : ℤ) : 
  ∀ x, (-(5 * Real.pi / 3) + 4 * (k : ℝ) * Real.pi ≤ x) ∧ (x ≤ Real.pi / 3 + 4 * (k : ℝ) * Real.pi) → 
  ∀ y, (-(5 * Real.pi / 3) + 4 * (k : ℝ) * Real.pi ≤ y) ∧ (y ≤ Real.pi / 3 + 4 * (k : ℝ) * Real.pi) → 
  (x ≤ y ↔ f x ≤ f y) :=
sorry

end max_values_of_f_smallest_positive_period_of_f_intervals_where_f_is_monotonically_increasing_l1563_156397


namespace solve_for_x_l1563_156369

theorem solve_for_x (y : ℝ) : 
  ∃ x : ℝ, 19 * (x + y) + 17 = 19 * (-x + y) - 21 ∧ 
           x = -21 / 38 :=
by
  sorry

end solve_for_x_l1563_156369


namespace athlete_A_most_stable_l1563_156301

noncomputable def athlete_A_variance : ℝ := 0.019
noncomputable def athlete_B_variance : ℝ := 0.021
noncomputable def athlete_C_variance : ℝ := 0.020
noncomputable def athlete_D_variance : ℝ := 0.022

theorem athlete_A_most_stable :
  athlete_A_variance < athlete_B_variance ∧
  athlete_A_variance < athlete_C_variance ∧
  athlete_A_variance < athlete_D_variance :=
by {
  sorry
}

end athlete_A_most_stable_l1563_156301


namespace length_of_train_l1563_156399

variable (L : ℝ) (S : ℝ)

-- Condition 1: The train crosses a 120 meters platform in 15 seconds
axiom condition1 : S = (L + 120) / 15

-- Condition 2: The train crosses a 250 meters platform in 20 seconds
axiom condition2 : S = (L + 250) / 20

-- The theorem to be proved
theorem length_of_train : L = 270 :=
by
  sorry

end length_of_train_l1563_156399


namespace intersection_area_l1563_156387

-- Define the square vertices
def vertex1 : (ℝ × ℝ) := (2, 8)
def vertex2 : (ℝ × ℝ) := (13, 8)
def vertex3 : (ℝ × ℝ) := (13, -3)
def vertex4 : (ℝ × ℝ) := (2, -3)  -- Derived from the conditions

-- Define the circle with center and radius
def circle_center : (ℝ × ℝ) := (2, -3)
def circle_radius : ℝ := 4

-- Define the square side length
def square_side_length : ℝ := 11  -- From vertex (2, 8) to vertex (2, -3)

-- Prove the intersection area
theorem intersection_area :
  let area := (1 / 4) * Real.pi * (circle_radius^2)
  area = 4 * Real.pi :=
by
  sorry

end intersection_area_l1563_156387


namespace calculate_total_area_of_figure_l1563_156393

-- Defining the lengths of the segments according to the problem conditions.
def length_1 : ℕ := 8
def length_2 : ℕ := 6
def length_3 : ℕ := 3
def length_4 : ℕ := 5
def length_5 : ℕ := 2
def length_6 : ℕ := 4

-- Using the given lengths to compute the areas of the smaller rectangles
def area_A : ℕ := length_1 * length_2
def area_B : ℕ := length_4 * (10 - 6)
def area_C : ℕ := (6 - 3) * (15 - 10)

-- The total area of the figure is the sum of the areas of the smaller rectangles
def total_area : ℕ := area_A + area_B + area_C

-- The statement to prove
theorem calculate_total_area_of_figure : total_area = 83 := by
  -- Proof goes here
  sorry

end calculate_total_area_of_figure_l1563_156393


namespace percentage_of_500_l1563_156342

theorem percentage_of_500 : (110 / 100) * 500 = 550 := 
  by
  -- Here we would provide the proof (placeholder)
  sorry

end percentage_of_500_l1563_156342


namespace unique_m_value_l1563_156398

theorem unique_m_value : ∀ m : ℝ,
  (m ^ 2 - 5 * m + 6 = 0 ∧ m ^ 2 - 3 * m + 2 = 0) →
  (m ^ 2 - 3 * m + 2 = 2 * (m ^ 2 - 5 * m + 6)) →
  ((m ^ 2 - 5 * m + 6) * (m ^ 2 - 3 * m + 2) > 0) →
  m = 2 :=
by
  sorry

end unique_m_value_l1563_156398


namespace total_spectators_after_halftime_l1563_156382

theorem total_spectators_after_halftime
  (initial_boys : ℕ := 300)
  (initial_girls : ℕ := 400)
  (initial_adults : ℕ := 300)
  (total_people : ℕ := 1000)
  (quarter_boys_leave_fraction : ℚ := 1 / 4)
  (quarter_girls_leave_fraction : ℚ := 1 / 8)
  (quarter_adults_leave_fraction : ℚ := 1 / 5)
  (halftime_new_boys : ℕ := 50)
  (halftime_new_girls : ℕ := 90)
  (halftime_adults_leave_fraction : ℚ := 3 / 100) :
  let boys_after_first_quarter := initial_boys - initial_boys * quarter_boys_leave_fraction
  let girls_after_first_quarter := initial_girls - initial_girls * quarter_girls_leave_fraction
  let adults_after_first_quarter := initial_adults - initial_adults * quarter_adults_leave_fraction
  let boys_after_halftime := boys_after_first_quarter + halftime_new_boys
  let girls_after_halftime := girls_after_first_quarter + halftime_new_girls
  let adults_after_halftime := adults_after_first_quarter * (1 - halftime_adults_leave_fraction)
  boys_after_halftime + girls_after_halftime + adults_after_halftime = 948 :=
by sorry

end total_spectators_after_halftime_l1563_156382


namespace calculate_savings_l1563_156374

noncomputable def monthly_salary : ℕ := 10000
noncomputable def spent_on_food (S : ℕ) : ℕ := (40 * S) / 100
noncomputable def spent_on_rent (S : ℕ) : ℕ := (20 * S) / 100
noncomputable def spent_on_entertainment (S : ℕ) : ℕ := (10 * S) / 100
noncomputable def spent_on_conveyance (S : ℕ) : ℕ := (10 * S) / 100
noncomputable def total_spent (S : ℕ) : ℕ := spent_on_food S + spent_on_rent S + spent_on_entertainment S + spent_on_conveyance S
noncomputable def amount_saved (S : ℕ) : ℕ := S - total_spent S

theorem calculate_savings : amount_saved monthly_salary = 2000 :=
by
  sorry

end calculate_savings_l1563_156374


namespace fg_3_eq_123_l1563_156364

def f (x : ℤ) : ℤ := x^2 + 2
def g (x : ℤ) : ℤ := 3 * x + 2

theorem fg_3_eq_123 : f (g 3) = 123 := by
  sorry

end fg_3_eq_123_l1563_156364


namespace fountain_area_l1563_156390

theorem fountain_area (A B D C : ℝ) (h₁ : B - A = 20) (h₂ : D = (A + B) / 2) (h₃ : C - D = 12) :
  ∃ R : ℝ, R^2 = 244 ∧ π * R^2 = 244 * π :=
by
  sorry

end fountain_area_l1563_156390


namespace calculate_expression_l1563_156343

theorem calculate_expression : 
  (-7 : ℤ)^7 / (7 : ℤ)^4 + 2^6 - 8^2 = -343 :=
by
  sorry

end calculate_expression_l1563_156343


namespace find_original_number_l1563_156365

theorem find_original_number (x : ℝ) :
  (((x / 2.5) - 10.5) * 0.3 = 5.85) -> x = 75 :=
by
  sorry

end find_original_number_l1563_156365


namespace sum_of_square_face_is_13_l1563_156302

-- Definitions based on conditions
variables (x₁ x₂ x₃ x₄ x₅ : ℕ)

-- Conditions
axiom h₁ : x₁ + x₂ + x₃ = 7
axiom h₂ : x₁ + x₂ + x₄ = 8
axiom h₃ : x₁ + x₃ + x₄ = 9
axiom h₄ : x₂ + x₃ + x₄ = 10

-- Properties
axiom h_sum : x₁ + x₂ + x₃ + x₄ + x₅ = 15

-- Goal to prove
theorem sum_of_square_face_is_13 (h₁ : x₁ + x₂ + x₃ = 7) (h₂ : x₁ + x₂ + x₄ = 8) 
  (h₃ : x₁ + x₃ + x₄ = 9) (h₄ : x₂ + x₃ + x₄ = 10) (h_sum : x₁ + x₂ + x₃ + x₄ + x₅ = 15): 
  x₅ + x₁ + x₂ + x₄ = 13 :=
sorry

end sum_of_square_face_is_13_l1563_156302


namespace find_time_for_compound_interest_l1563_156355

noncomputable def compound_interest_time 
  (A P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  (Real.log (A / P)) / (n * Real.log (1 + r / n))

theorem find_time_for_compound_interest :
  compound_interest_time 500 453.51473922902494 0.05 1 = 2 :=
sorry

end find_time_for_compound_interest_l1563_156355


namespace sufficient_condition_for_line_perpendicular_to_plane_l1563_156329

variables {Plane Line : Type}
variables (α β γ : Plane) (m n l : Line)

-- Definitions of perpendicularity and inclusion
def perp (l : Line) (p : Plane) : Prop := sorry -- definition of a line being perpendicular to a plane
def parallel (p₁ p₂ : Plane) : Prop := sorry -- definition of parallel planes
def incl (l : Line) (p : Plane) : Prop := sorry -- definition of a line being in a plane

-- The given conditions
axiom n_perp_α : perp n α
axiom n_perp_β : perp n β
axiom m_perp_α : perp m α

-- The proof goal
theorem sufficient_condition_for_line_perpendicular_to_plane :
  perp m β :=
by
    sorry

end sufficient_condition_for_line_perpendicular_to_plane_l1563_156329


namespace max_hot_dogs_with_300_dollars_l1563_156327

def num_hot_dogs (dollars : ℕ) 
  (cost_8 : ℚ) (count_8 : ℕ) 
  (cost_20 : ℚ) (count_20 : ℕ)
  (cost_250 : ℚ) (count_250 : ℕ) : ℕ :=
  sorry

theorem max_hot_dogs_with_300_dollars : 
  num_hot_dogs 300 1.55 8 3.05 20 22.95 250 = 3258 :=
sorry

end max_hot_dogs_with_300_dollars_l1563_156327


namespace parabola_int_x_axis_for_all_m_l1563_156325

theorem parabola_int_x_axis_for_all_m {n : ℝ} :
  (∀ m : ℝ, (9 * m^2 - 4 * m - 4 * n) ≥ 0) → (n ≤ -1 / 9) :=
by
  intro h
  sorry

end parabola_int_x_axis_for_all_m_l1563_156325


namespace petya_cannot_form_figure_c_l1563_156322

-- Define the rhombus and its properties, including rotation
noncomputable def is_rotatable_rhombus (r : ℕ) : Prop := sorry

-- Define the larger shapes and their properties in terms of whether they can be formed using rotations of the rhombus.
noncomputable def can_form_figure_a (rhombus : ℕ) : Prop := sorry
noncomputable def can_form_figure_b (rhombus : ℕ) : Prop := sorry
noncomputable def can_form_figure_c (rhombus : ℕ) : Prop := sorry
noncomputable def can_form_figure_d (rhombus : ℕ) : Prop := sorry

-- Statement: Petya cannot form the figure (c) using the rhombus and allowed transformations.
theorem petya_cannot_form_figure_c (rhombus : ℕ) (h : is_rotatable_rhombus rhombus) :
  ¬ can_form_figure_c rhombus := sorry

end petya_cannot_form_figure_c_l1563_156322


namespace range_x_plus_y_l1563_156303

theorem range_x_plus_y (x y: ℝ) (h: x^2 + y^2 - 4 * x + 3 = 0) : 
  2 - Real.sqrt 2 ≤ x + y ∧ x + y ≤ 2 + Real.sqrt 2 :=
by 
  sorry

end range_x_plus_y_l1563_156303


namespace part1_solution_count_part2_solution_count_l1563_156385

theorem part1_solution_count :
  ∃ (solutions : Finset (ℕ × ℕ × ℕ)), solutions.card = 7 ∧
    ∀ (m n r : ℕ), (m, n, r) ∈ solutions ↔ mn + nr + mr = 2 * (m + n + r) := sorry

theorem part2_solution_count (k : ℕ) (h : 1 < k) :
  ∃ (solutions : Finset (ℕ × ℕ × ℕ)), solutions.card ≥ 3 * k + 1 ∧
    ∀ (m n r : ℕ), (m, n, r) ∈ solutions ↔ mn + nr + mr = k * (m + n + r) := sorry

end part1_solution_count_part2_solution_count_l1563_156385


namespace expand_product_l1563_156304

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5 * x - 36 :=
by
  sorry

end expand_product_l1563_156304


namespace not_perfect_square_l1563_156388

theorem not_perfect_square (a b : ℤ) (ha : 0 < a) (hb : 0 < b) (h : ¬ (a^2 - b^2) % 4 = 0) : 
  ¬ ∃ k : ℤ, (a + 3*b) * (5*a + 7*b) = k^2 :=
sorry

end not_perfect_square_l1563_156388


namespace determinant_of_sine_matrix_is_zero_l1563_156378

theorem determinant_of_sine_matrix_is_zero : 
  let M : Matrix (Fin 3) (Fin 3) ℝ :=
    ![![Real.sin 2, Real.sin 3, Real.sin 4],
      ![Real.sin 5, Real.sin 6, Real.sin 7],
      ![Real.sin 8, Real.sin 9, Real.sin 10]]
  Matrix.det M = 0 := 
by sorry

end determinant_of_sine_matrix_is_zero_l1563_156378


namespace x_gt_1_implies_inv_x_lt_1_inv_x_lt_1_not_necessitates_x_gt_1_l1563_156360

theorem x_gt_1_implies_inv_x_lt_1 (x : ℝ) (h : x > 1) : 1 / x < 1 :=
by
  sorry

theorem inv_x_lt_1_not_necessitates_x_gt_1 (x : ℝ) (h : 1 / x < 1) : ¬(x > 1) ∨ (x ≤ 1) :=
by
  sorry

end x_gt_1_implies_inv_x_lt_1_inv_x_lt_1_not_necessitates_x_gt_1_l1563_156360


namespace similar_triangles_iff_l1563_156386

variables {a b c a' b' c' : ℂ}

theorem similar_triangles_iff :
  (∃ (z w : ℂ), a' = a * z + w ∧ b' = b * z + w ∧ c' = c * z + w) ↔
  a' * (b - c) + b' * (c - a) + c' * (a - b) = 0 :=
sorry

end similar_triangles_iff_l1563_156386


namespace fairy_tale_island_counties_l1563_156306

theorem fairy_tale_island_counties :
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  let first_year_elves := initial_elves
  let first_year_dwarves := initial_dwarves * 3
  let first_year_centaurs := initial_centaurs * 3

  let second_year_elves := first_year_elves * 4
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := first_year_centaurs * 4

  let third_year_elves := second_year_elves * 6
  let third_year_dwarves := second_year_dwarves * 6
  let third_year_centaurs := second_year_centaurs

  let total_counties := third_year_elves + third_year_dwarves + third_year_centaurs

  total_counties = 54 :=
by
  sorry

end fairy_tale_island_counties_l1563_156306


namespace ln_gt_ln_sufficient_for_x_gt_y_l1563_156309

noncomputable def ln : ℝ → ℝ := sorry  -- Assuming ln is imported from Mathlib

-- Conditions
variable (x y : ℝ)
axiom ln_gt_ln_of_x_gt_y (hxy : x > y) (hx_pos : 0 < x) (hy_pos : 0 < y) : ln x > ln y

theorem ln_gt_ln_sufficient_for_x_gt_y (h : ln x > ln y) : x > y := sorry

end ln_gt_ln_sufficient_for_x_gt_y_l1563_156309


namespace toby_peanut_butter_servings_l1563_156313

theorem toby_peanut_butter_servings :
  let bread_calories := 100
  let peanut_butter_calories_per_serving := 200
  let total_calories := 500
  let bread_pieces := 1
  ∃ (servings : ℕ), total_calories = (bread_calories * bread_pieces) + (peanut_butter_calories_per_serving * servings) → servings = 2 := by
  sorry

end toby_peanut_butter_servings_l1563_156313


namespace bumper_cars_number_of_tickets_l1563_156337

theorem bumper_cars_number_of_tickets (Ferris_Wheel Roller_Coaster Jeanne_Has Jeanne_Buys : ℕ)
  (h1 : Ferris_Wheel = 5)
  (h2 : Roller_Coaster = 4)
  (h3 : Jeanne_Has = 5)
  (h4 : Jeanne_Buys = 8) :
  Ferris_Wheel + Roller_Coaster + (13 - (Ferris_Wheel + Roller_Coaster)) = 13 - (Ferris_Wheel + Roller_Coaster) :=
by
  sorry

end bumper_cars_number_of_tickets_l1563_156337


namespace mitzi_amount_brought_l1563_156316

-- Define the amounts spent on different items
def ticket_cost : ℕ := 30
def food_cost : ℕ := 13
def tshirt_cost : ℕ := 23

-- Define the amount of money left
def amount_left : ℕ := 9

-- Define the total amount spent
def total_spent : ℕ :=
  ticket_cost + food_cost + tshirt_cost

-- Define the total amount brought to the amusement park
def amount_brought : ℕ :=
  total_spent + amount_left

-- Prove that the amount of money Mitzi brought to the amusement park is 75
theorem mitzi_amount_brought : amount_brought = 75 := by
  sorry

end mitzi_amount_brought_l1563_156316


namespace polynomial_real_roots_l1563_156317

theorem polynomial_real_roots :
  ∀ x : ℝ, (x^4 - 3 * x^3 + 3 * x^2 - x - 6 = 0) ↔ (x = 3 ∨ x = 2 ∨ x = -1) := 
by
  sorry

end polynomial_real_roots_l1563_156317


namespace partition_cities_l1563_156394

-- Define the type for cities and airlines.
variable (City : Type) (Airline : Type)

-- Define the number of cities and airlines
variable (n k : ℕ)

-- Define a relation to represent bidirectional direct flights
variable (flight : Airline → City → City → Prop)

-- Define the condition: Some pairs of cities are connected by exactly one direct flight operated by one of the airline companies
-- or there are no such flights between them.
axiom unique_flight : ∀ (a : Airline) (c1 c2 : City), flight a c1 c2 → ¬ (∃ (a' : Airline), flight a' c1 c2 ∧ a' ≠ a)

-- Define the condition: Any two direct flights operated by the same company share a common endpoint
axiom shared_endpoint :
  ∀ (a : Airline) (c1 c2 c3 c4 : City), flight a c1 c2 → flight a c3 c4 → (c1 = c3 ∨ c1 = c4 ∨ c2 = c3 ∨ c2 = c4)

-- The main theorem to prove
theorem partition_cities :
  ∃ (partition : City → Fin (k + 2)), ∀ (c1 c2 : City) (a : Airline), flight a c1 c2 → partition c1 ≠ partition c2 :=
sorry

end partition_cities_l1563_156394


namespace total_bins_used_l1563_156356

def bins_of_soup : ℝ := 0.12
def bins_of_vegetables : ℝ := 0.12
def bins_of_pasta : ℝ := 0.5

theorem total_bins_used : bins_of_soup + bins_of_vegetables + bins_of_pasta = 0.74 :=
by
  sorry

end total_bins_used_l1563_156356


namespace find_rate_squares_sum_l1563_156336

theorem find_rate_squares_sum {b j s : ℤ} 
(H1 : 3 * b + 2 * j + 2 * s = 112)
(H2 : 2 * b + 3 * j + 4 * s = 129) : b^2 + j^2 + s^2 = 1218 :=
by sorry

end find_rate_squares_sum_l1563_156336


namespace correct_statements_l1563_156383

theorem correct_statements (f : ℝ → ℝ) (t : ℝ)
  (h1 : ∀ x1 x2 : ℝ, x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) < 0)
  (h2 : (∀ x : ℝ, f x = f (-x)) ∧ (∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 < x2 → f x1 > f x2) ∧ f (-2) = 0)
  (h3 : ∀ x : ℝ, f (-x) = -f x)
  (h4 : ∀ x : ℝ, f (x - t) = f (x + t)) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 > f x2 ↔ x1 < x2) ∧
  (∀ x : ℝ, f x - f (|x|) = - (f (-x) - f (|x|))) :=
by
  sorry

end correct_statements_l1563_156383


namespace hoseok_needs_17_more_jumps_l1563_156346

/-- Define the number of jumps by Hoseok and Minyoung -/
def hoseok_jumps : ℕ := 34
def minyoung_jumps : ℕ := 51

/-- Define the number of additional jumps Hoseok needs -/
def additional_jumps_hoseok : ℕ := minyoung_jumps - hoseok_jumps

/-- Prove that the additional jumps Hoseok needs is equal to 17 -/
theorem hoseok_needs_17_more_jumps (h_jumps : ℕ := hoseok_jumps) (m_jumps : ℕ := minyoung_jumps) :
  additional_jumps_hoseok = 17 := by
  -- Proof goes here
  sorry

end hoseok_needs_17_more_jumps_l1563_156346


namespace total_value_of_coins_l1563_156344

theorem total_value_of_coins :
  (∀ (coins : List (String × ℕ)), coins.length = 12 →
    (∃ Q N : ℕ, 
      Q = 4 ∧ N = 8 ∧
      (∀ (coin : String × ℕ), coin ∈ coins → 
        (coin = ("quarter", Q) → Q = 4 ∧ (Q * 25 = 100)) ∧ 
        (coin = ("nickel", N) → N = 8 ∧ (N * 5 = 40)) ∧
      (Q * 25 + N * 5 = 140)))) :=
sorry

end total_value_of_coins_l1563_156344


namespace parabola_directrix_l1563_156330

theorem parabola_directrix (x y : ℝ) :
  (∃ a b c : ℝ, y = (a * x^2 + b * x + c) / 12 ∧ a = 1 ∧ b = -6 ∧ c = 5) →
  y = -10 / 3 :=
by
  sorry

end parabola_directrix_l1563_156330


namespace sum_of_cubes_condition_l1563_156377

theorem sum_of_cubes_condition (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_condition : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) :
  a^3 + b^3 + c^3 = -36 := 
by
  sorry

end sum_of_cubes_condition_l1563_156377


namespace compute_expression_l1563_156380

-- Definition of the imaginary unit i
class ImaginaryUnit (i : ℂ) where
  I_square : i * i = -1

-- Definition of non-zero real number a
variable (a : ℝ) (h_a : a ≠ 0)

-- Theorem to prove the equivalence
theorem compute_expression (i : ℂ) [ImaginaryUnit i] :
  (a * i - i⁻¹)⁻¹ = -i / (a + 1) :=
by
  sorry

end compute_expression_l1563_156380


namespace inequality_solution_l1563_156323

theorem inequality_solution (a x : ℝ) : 
  (a = 0 → ¬(x^2 - 2*a*x - 3*a^2 < 0)) ∧
  (a > 0 → (-a < x ∧ x < 3*a) ↔ (x^2 - 2*a*x - 3*a^2 < 0)) ∧
  (a < 0 → (3*a < x ∧ x < -a) ↔ (x^2 - 2*a*x - 3*a^2 < 0)) :=
by
  sorry

end inequality_solution_l1563_156323


namespace amount_per_friend_l1563_156361

-- Definitions based on conditions
def cost_of_erasers : ℝ := 5 * 200
def cost_of_pencils : ℝ := 7 * 800
def total_cost : ℝ := cost_of_erasers + cost_of_pencils
def number_of_friends : ℝ := 4

-- The proof statement
theorem amount_per_friend : (total_cost / number_of_friends) = 1650 := by
  sorry

end amount_per_friend_l1563_156361


namespace find_N_l1563_156376

theorem find_N : ∀ N : ℕ, (991 + 993 + 995 + 997 + 999 = 5000 - N) → N = 25 :=
by
  intro N h
  sorry

end find_N_l1563_156376


namespace smallest_value_of_linear_expression_l1563_156339

theorem smallest_value_of_linear_expression :
  (∃ a, 8 * a^2 + 6 * a + 5 = 7 ∧ (∃ b, b = 3 * a + 2 ∧ ∀ c, (8 * c^2 + 6 * c + 5 = 7 → 3 * c + 2 ≥ b))) → -1 = b :=
by
  sorry

end smallest_value_of_linear_expression_l1563_156339


namespace mixed_number_expression_l1563_156328

theorem mixed_number_expression :
  (7 + 1/2 - (5 + 3/4)) * (3 + 1/6 + (2 + 1/8)) = 9 + 25/96 :=
by
  -- here we would provide the proof steps
  sorry

end mixed_number_expression_l1563_156328


namespace age_of_son_l1563_156311

theorem age_of_son (S M : ℕ) 
  (h1 : M = S + 22)
  (h2 : M + 2 = 2 * (S + 2)) : 
  S = 20 := 
sorry

end age_of_son_l1563_156311


namespace ratio_A_B_l1563_156395

-- Define constants for non-zero numbers A and B
variables {A B : ℕ} (h1 : A ≠ 0) (h2 : B ≠ 0)

-- Define the given condition
theorem ratio_A_B (h : (2 * A) * 7 = (3 * B) * 3) : A / B = 9 / 14 := by
  sorry

end ratio_A_B_l1563_156395


namespace closest_point_on_line_l1563_156335

theorem closest_point_on_line :
  ∀ (x y : ℝ), (4, -2) = (4, -2) →
    y = 3 * x - 1 →
    (∃ (p : ℝ × ℝ), p = (-0.5, -2.5) ∧ p = (-0.5, -2.5))
  := by
    -- The proof of the theorem goes here
    sorry

end closest_point_on_line_l1563_156335


namespace real_part_of_z_l1563_156373

open Complex

theorem real_part_of_z (z : ℂ) (h : I * z = 1 + 2 * I) : z.re = 2 :=
sorry

end real_part_of_z_l1563_156373


namespace solve_system_l1563_156318

variable {a b c : ℝ}
variable {x y z : ℝ}
variable {e1 e2 e3 : ℤ} -- Sign variables should be integers to express ±1 more easily 

axiom ax1 : x * (x + y) + z * (x - y) = a
axiom ax2 : y * (y + z) + x * (y - z) = b
axiom ax3 : z * (z + x) + y * (z - x) = c

theorem solve_system :
  (e1 = 1 ∨ e1 = -1) ∧ (e2 = 1 ∨ e2 = -1) ∧ (e3 = 1 ∨ e3 = -1) →
  x = (1/2) * (e1 * Real.sqrt (a + b) - e2 * Real.sqrt (b + c) + e3 * Real.sqrt (c + a)) ∧
  y = (1/2) * (e1 * Real.sqrt (a + b) + e2 * Real.sqrt (b + c) - e3 * Real.sqrt (c + a)) ∧
  z = (1/2) * (-e1 * Real.sqrt (a + b) + e2 * Real.sqrt (b + c) + e3 * Real.sqrt (c + a)) :=
sorry -- proof goes here

end solve_system_l1563_156318


namespace evaluate_f_g_l1563_156305

def g (x : ℝ) : ℝ := 3 * x
def f (x : ℝ) : ℝ := x - 6

theorem evaluate_f_g :
  f (g 3) = 3 :=
by
  sorry

end evaluate_f_g_l1563_156305


namespace perpendicular_lines_slope_l1563_156358

theorem perpendicular_lines_slope (a : ℝ) (h1 :  a * (a + 2) = -1) : a = -1 :=
by 
-- Perpendicularity condition given
sorry

end perpendicular_lines_slope_l1563_156358


namespace cookies_remaining_in_jar_l1563_156351

-- Definition of the conditions
variable (initial_cookies : Nat)

def cookies_taken_by_Lou_Senior := 3 + 1
def cookies_taken_by_Louie_Junior := 7
def total_cookies_taken := cookies_taken_by_Lou_Senior + cookies_taken_by_Louie_Junior

-- Debra's assumption and the proof goal
theorem cookies_remaining_in_jar (half_cookies_removed : total_cookies_taken = initial_cookies / 2) : 
  initial_cookies - total_cookies_taken = 11 := by
  sorry

end cookies_remaining_in_jar_l1563_156351


namespace harry_friday_speed_l1563_156357

theorem harry_friday_speed :
  let monday_speed := 10
  let tuesday_thursday_speed := monday_speed + monday_speed * (50 / 100)
  let friday_speed := tuesday_thursday_speed + tuesday_thursday_speed * (60 / 100)
  friday_speed = 24 :=
by
  sorry

end harry_friday_speed_l1563_156357


namespace find_a_plus_b_l1563_156347

theorem find_a_plus_b (a b : ℤ) (h1 : a^2 = 16) (h2 : b^3 = -27) (h3 : |a - b| = a - b) : a + b = 1 := by
  sorry

end find_a_plus_b_l1563_156347


namespace students_count_l1563_156319

noncomputable def num_students (N T : ℕ) : Prop :=
  T = 72 * N ∧ (T - 200) / (N - 5) = 92

theorem students_count (N T : ℕ) : num_students N T → N = 13 :=
by
  sorry

end students_count_l1563_156319
