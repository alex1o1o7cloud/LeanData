import Mathlib

namespace problem1_problem2_l573_57313

open Set

variable {U : Set ℝ} (A B : Set ℝ)

def UA : U = univ := by sorry
def A_def : A = { x : ℝ | 0 < x ∧ x ≤ 2 } := by sorry
def B_def : B = { x : ℝ | x < -3 ∨ x > 1 } := by sorry

theorem problem1 : A ∩ B = { x : ℝ | 1 < x ∧ x ≤ 2 } := 
by sorry

theorem problem2 : (U \ A) ∩ (U \ B) = { x : ℝ | -3 ≤ x ∧ x ≤ 0 } := 
by sorry

end problem1_problem2_l573_57313


namespace integer_a_conditions_l573_57347

theorem integer_a_conditions (a : ℤ) :
  (∃ (x y : ℕ), x ≠ y ∧ (a * x * y + 1) ∣ (a * x^2 + 1) ^ 2) → a ≥ -1 :=
sorry

end integer_a_conditions_l573_57347


namespace tooth_extraction_cost_l573_57386

variable (c f b e : ℕ)

-- Conditions
def cost_cleaning := c = 70
def cost_filling := f = 120
def bill := b = 5 * f

-- Proof Problem
theorem tooth_extraction_cost (h_cleaning : cost_cleaning c) (h_filling : cost_filling f) (h_bill : bill b f) :
  e = b - (c + 2 * f) :=
sorry

end tooth_extraction_cost_l573_57386


namespace chord_PQ_eqn_l573_57361

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9
def midpoint_PQ (M : ℝ × ℝ) : Prop := M = (1, 2)
def line_PQ_eq (x y : ℝ) : Prop := x + 2 * y - 5 = 0

theorem chord_PQ_eqn : 
  (∃ P Q : ℝ × ℝ, circle_eq P.1 P.2 ∧ circle_eq Q.1 Q.2 ∧ midpoint_PQ ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) →
  ∃ x y : ℝ, line_PQ_eq x y := 
sorry

end chord_PQ_eqn_l573_57361


namespace square_side_length_l573_57367

theorem square_side_length (s : ℚ) (h : s^2 = 9/16) : s = 3/4 := 
by
sorry

end square_side_length_l573_57367


namespace right_triangle_hypotenuse_length_l573_57363

theorem right_triangle_hypotenuse_length :
  ∀ (a b h : ℕ), a = 15 → b = 36 → h^2 = a^2 + b^2 → h = 39 :=
by
  intros a b h ha hb hyp
  -- In the proof, we would use ha, hb, and hyp to show h = 39
  sorry

end right_triangle_hypotenuse_length_l573_57363


namespace value_of_a_range_of_m_l573_57394

def f (x a : ℝ) : ℝ := abs (x - a)

-- Given the following conditions
axiom cond1 (x : ℝ) (a : ℝ) : f x a = abs (x - a)
axiom cond2 (x : ℝ) (a : ℝ) : (f x a >= 3) ↔ (x <= 1 ∨ x >= 5)

-- Prove that a = 2
theorem value_of_a (a : ℝ) : (∀ x : ℝ, (f x a >= 3) ↔ (x <= 1 ∨ x >= 5)) → a = 2 := by
  sorry

-- Additional condition for m
axiom cond3 (x : ℝ) (a : ℝ) (m : ℝ) : ∀ x : ℝ, f x a + f (x + 4) a >= m

-- Prove that m ≤ 4
theorem range_of_m (a : ℝ) (m : ℝ) : (∀ x : ℝ, f x a + f (x + 4) a >= m) → a = 2 → m ≤ 4 := by
  sorry

end value_of_a_range_of_m_l573_57394


namespace reporters_not_covering_politics_l573_57368

-- Definitions of basic quantities
variables (R P : ℝ) (percentage_local : ℝ) (percentage_no_local : ℝ)

-- Conditions from the problem
def conditions : Prop :=
  R = 100 ∧
  percentage_local = 10 ∧
  percentage_no_local = 30 ∧
  percentage_local = 0.7 * P

-- Theorem statement for the problem
theorem reporters_not_covering_politics (h : conditions R P percentage_local percentage_no_local) :
  100 - P = 85.71 :=
by sorry

end reporters_not_covering_politics_l573_57368


namespace probability_of_selecting_storybook_l573_57339

theorem probability_of_selecting_storybook (reference_books storybooks picture_books : ℕ) 
  (h1 : reference_books = 5) (h2 : storybooks = 3) (h3 : picture_books = 2) :
  (storybooks : ℚ) / (reference_books + storybooks + picture_books) = 3 / 10 :=
by {
  sorry
}

end probability_of_selecting_storybook_l573_57339


namespace square_diagonal_l573_57320

theorem square_diagonal (s d : ℝ) (h : 4 * s = 40) : d = s * Real.sqrt 2 → d = 10 * Real.sqrt 2 :=
by
  sorry

end square_diagonal_l573_57320


namespace number_of_negative_x_values_l573_57393

theorem number_of_negative_x_values : 
  (∃ (n : ℕ), ∀ (x : ℤ), x = n^2 - 196 ∧ x < 0) ∧ (n ≤ 13) :=
by 
  -- To formalize our problem we need quantifiers, inequalities and integer properties.
  sorry

end number_of_negative_x_values_l573_57393


namespace max_brownies_l573_57373

theorem max_brownies (m n : ℕ) (h1 : (m-2)*(n-2) = 2*(2*m + 2*n - 4)) : m * n ≤ 294 :=
by sorry

end max_brownies_l573_57373


namespace angle_A_minimum_a_l573_57383

variable {α : Type} [LinearOrderedField α]

-- Part 1: Prove A = π / 3 given the specific equation in triangle ABC
theorem angle_A (a b c : α) (cos : α → α)
  (h : b^2 * c * cos c + c^2 * b * cos b = a * b^2 + a * c^2 - a^3) :
  ∃ A : α, A = π / 3 :=
sorry

-- Part 2: Prove the minimum value of a is 1 when b + c = 2
theorem minimum_a (a b c : α) (h : b + c = 2) :
  ∃ a : α, a = 1 :=
sorry

end angle_A_minimum_a_l573_57383


namespace no_valid_solution_l573_57336

theorem no_valid_solution (x y z : ℤ) (h1 : x = 11 * y + 4) 
  (h2 : 2 * x = 24 * y + 3) (h3 : x + z = 34 * y + 5) : 
  ¬ ∃ (y : ℤ), 13 * y - x + 7 * z = 0 :=
by
  sorry

end no_valid_solution_l573_57336


namespace total_notes_in_week_l573_57385

-- Define the conditions for day hours ring pattern
def day_notes (hour : ℕ) (minute : ℕ) : ℕ :=
  if minute = 15 then 2
  else if minute = 30 then 4
  else if minute = 45 then 6
  else if minute = 0 then 
    8 + (if hour % 2 = 0 then hour else hour / 2)
  else 0

-- Define the conditions for night hours ring pattern
def night_notes (hour : ℕ) (minute : ℕ) : ℕ :=
  if minute = 15 then 3
  else if minute = 30 then 5
  else if minute = 45 then 7
  else if minute = 0 then 
    9 + (if hour % 2 = 1 then hour else hour / 2)
  else 0

-- Define total notes over day period
def total_day_notes : ℕ := 
  (day_notes 6 0 + day_notes 7 0 + day_notes 8 0 + day_notes 9 0 + day_notes 10 0 + day_notes 11 0
 + day_notes 12 0 + day_notes 1 0 + day_notes 2 0 + day_notes 3 0 + day_notes 4 0 + day_notes 5 0)
 +
 (2 * 12 + 4 * 12 + 6 * 12)

-- Define total notes over night period
def total_night_notes : ℕ := 
  (night_notes 6 0 + night_notes 7 0 + night_notes 8 0 + night_notes 9 0 + night_notes 10 0 + night_notes 11 0
 + night_notes 12 0 + night_notes 1 0 + night_notes 2 0 + night_notes 3 0 + night_notes 4 0 + night_notes 5 0)
 +
 (3 * 12 + 5 * 12 + 7 * 12)

-- Define the total number of notes the clock will ring in a full week
def total_week_notes : ℕ :=
  7 * (total_day_notes + total_night_notes)

theorem total_notes_in_week : 
  total_week_notes = 3297 := 
  by 
  sorry

end total_notes_in_week_l573_57385


namespace radius_of_circle_l573_57310

theorem radius_of_circle (r x y : ℝ): 
  x = π * r^2 → 
  y = 2 * π * r → 
  x - y = 72 * π → 
  r = 12 := 
by 
  sorry

end radius_of_circle_l573_57310


namespace simplify_expression_l573_57365

theorem simplify_expression
  (a b c : ℝ) 
  (hnz_a : a ≠ 0) 
  (hnz_b : b ≠ 0) 
  (hnz_c : c ≠ 0) 
  (h_sum : a + b + c = 0) :
  (1 / (b^3 + c^3 - a^3)) + (1 / (a^3 + c^3 - b^3)) + (1 / (a^3 + b^3 - c^3)) = 1 / (a * b * c) :=
by
  sorry

end simplify_expression_l573_57365


namespace ab_non_positive_l573_57344

theorem ab_non_positive (a b : ℝ) (h : 2011 * a + 2012 * b = 0) : a * b ≤ 0 :=
sorry

end ab_non_positive_l573_57344


namespace remainder_3_101_add_5_mod_11_l573_57362

theorem remainder_3_101_add_5_mod_11 : (3 ^ 101 + 5) % 11 = 8 := 
by sorry

end remainder_3_101_add_5_mod_11_l573_57362


namespace marco_older_than_twice_marie_l573_57328

variable (M m x : ℕ)

def marie_age : ℕ := 12
def sum_of_ages : ℕ := 37

theorem marco_older_than_twice_marie :
  m = marie_age → (M = 2 * m + x) → (M + m = sum_of_ages) → x = 1 :=
by
  intros h1 h2 h3
  rw [h1] at h2 h3
  sorry

end marco_older_than_twice_marie_l573_57328


namespace polynomials_common_zero_k_l573_57317

theorem polynomials_common_zero_k
  (k : ℝ) :
  (∃ x : ℝ, (1988 * x^2 + k * x + 8891 = 0) ∧ (8891 * x^2 + k * x + 1988 = 0)) ↔ (k = 10879 ∨ k = -10879) :=
sorry

end polynomials_common_zero_k_l573_57317


namespace number_of_cipher_keys_l573_57374

theorem number_of_cipher_keys (n : ℕ) (h : n % 2 = 0) : 
  ∃ K : ℕ, K = 4^(n^2 / 4) :=
by 
  sorry

end number_of_cipher_keys_l573_57374


namespace quadratic_inequality_always_holds_l573_57323

theorem quadratic_inequality_always_holds (k : ℝ) (h : ∀ x : ℝ, (x^2 - k*x + 1) > 0) : -2 < k ∧ k < 2 :=
  sorry

end quadratic_inequality_always_holds_l573_57323


namespace problem_a_b_sum_l573_57392

-- Define the operation
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- Given conditions
variable (a b : ℝ)

-- Theorem statement: Prove that a + b = 4
theorem problem_a_b_sum :
  (∀ x, ((2 < x) ∧ (x < 3)) ↔ ((x - a) * (x - b - 1) < 0)) → a + b = 4 :=
by
  sorry

end problem_a_b_sum_l573_57392


namespace polygon_diagonals_l573_57391

-- Definitions of the conditions
def sum_of_angles (n : ℕ) : ℝ := (n - 2) * 180 + 360

def num_diagonals (n : ℕ) : ℤ := n * (n - 3) / 2

-- Theorem statement
theorem polygon_diagonals (n : ℕ) (h : sum_of_angles n = 2160) : num_diagonals n = 54 :=
sorry

end polygon_diagonals_l573_57391


namespace num_int_values_x_l573_57396

theorem num_int_values_x (x : ℕ) :
  (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) → ∃ n : ℕ, n = 3 :=
by
  sorry

end num_int_values_x_l573_57396


namespace num_monomials_degree_7_l573_57398

theorem num_monomials_degree_7 : 
  ∃ (count : Nat), 
    (∀ (a b c : ℕ), a + b + c = 7 → (1 : ℕ) = 1) ∧ 
    count = 15 := 
sorry

end num_monomials_degree_7_l573_57398


namespace repeating_decimal_as_fraction_l573_57357

theorem repeating_decimal_as_fraction :
  ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ Int.natAbs (Int.gcd a b) = 1 ∧ a + b = 15 ∧ (a : ℚ) / b = 0.3636363636363636 :=
by
  sorry

end repeating_decimal_as_fraction_l573_57357


namespace average_of_original_set_l573_57348

theorem average_of_original_set (A : ℝ) (h1 : 7 * A = 125 * 7 / 5) : A = 25 := 
sorry

end average_of_original_set_l573_57348


namespace sum_first_10_terms_l573_57330

-- Define the general arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a (n + 1) = a n + d

-- Define the conditions of the problem
def given_conditions (a : ℕ → ℤ) (d : ℤ) : Prop :=
  a 1 = 2 ∧ (a 2) ^ 2 = 2 * a 4 ∧ arithmetic_seq a d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(n * (a 1 + a n)) / 2

-- Statement of the problem
theorem sum_first_10_terms (a : ℕ → ℤ) (d : ℤ) (S₁₀ : ℤ) :
  given_conditions a d →
  (S₁₀ = 20 ∨ S₁₀ = 110) :=
sorry

end sum_first_10_terms_l573_57330


namespace quadractic_inequality_solution_l573_57359

theorem quadractic_inequality_solution (a b : ℝ) (h₁ : ∀ x : ℝ, -4 ≤ x ∧ x ≤ 3 → x^2 - (a+1) * x + b ≤ 0) : a + b = -14 :=
by 
  -- Proof construction is omitted
  sorry

end quadractic_inequality_solution_l573_57359


namespace find_g_720_l573_57390

noncomputable def g (n : ℕ) : ℕ := sorry

axiom g_multiplicative : ∀ (x y : ℕ), g (x * y) = g x + g y
axiom g_8 : g 8 = 12
axiom g_12 : g 12 = 16

theorem find_g_720 : g 720 = 44 := by sorry

end find_g_720_l573_57390


namespace dynaco_shares_l573_57300

theorem dynaco_shares (M D : ℕ) 
  (h1 : M + D = 300)
  (h2 : 36 * M + 44 * D = 12000) : 
  D = 150 :=
sorry

end dynaco_shares_l573_57300


namespace driving_time_per_trip_l573_57353

-- Define the conditions
def filling_time_per_trip : ℕ := 15
def number_of_trips : ℕ := 6
def total_moving_hours : ℕ := 7
def total_moving_time : ℕ := total_moving_hours * 60

-- Define the problem
theorem driving_time_per_trip :
  (total_moving_time - (filling_time_per_trip * number_of_trips)) / number_of_trips = 55 :=
by
  sorry

end driving_time_per_trip_l573_57353


namespace remainder_when_divided_by_100_l573_57302

theorem remainder_when_divided_by_100 (n : ℤ) (h : ∃ a : ℤ, n = 100 * a - 1) : 
  (n^3 + n^2 + 2 * n + 3) % 100 = 1 :=
by 
  sorry

end remainder_when_divided_by_100_l573_57302


namespace evaluate_expression_l573_57304

theorem evaluate_expression : 
  abs (abs (-abs (3 - 5) + 2) - 4) = 4 :=
by
  sorry

end evaluate_expression_l573_57304


namespace remainder_sum_div_l573_57337

theorem remainder_sum_div (n : ℤ) : ((9 - n) + (n + 5)) % 9 = 5 := by
  sorry

end remainder_sum_div_l573_57337


namespace sum_of_common_ratios_l573_57350

theorem sum_of_common_ratios (k p r : ℝ) (h₁ : k ≠ 0) (h₂ : p ≠ r) (h₃ : (k * (p ^ 2)) - (k * (r ^ 2)) = 4 * (k * p - k * r)) : 
  p + r = 4 :=
by
  -- Using the conditions provided, we can prove the sum of the common ratios is 4.
  sorry

end sum_of_common_ratios_l573_57350


namespace incorrect_option_D_l573_57354

-- Definitions based on conditions
def cumulative_progress (days : ℕ) : ℕ :=
  30 * days

-- The Lean statement representing the mathematically equivalent proof problem
theorem incorrect_option_D : cumulative_progress 11 = 330 ∧ ¬ (cumulative_progress 10 = 330) :=
by {
  sorry
}

end incorrect_option_D_l573_57354


namespace sum_x_y_eq_2_l573_57306

open Real

theorem sum_x_y_eq_2 (x y : ℝ) (h : x - 1 = 1 - y) : x + y = 2 :=
by
  sorry

end sum_x_y_eq_2_l573_57306


namespace jason_total_spent_l573_57384

def cost_of_flute : ℝ := 142.46
def cost_of_music_tool : ℝ := 8.89
def cost_of_song_book : ℝ := 7.00

def total_spent (flute_cost music_tool_cost song_book_cost : ℝ) : ℝ :=
  flute_cost + music_tool_cost + song_book_cost

theorem jason_total_spent :
  total_spent cost_of_flute cost_of_music_tool cost_of_song_book = 158.35 :=
by
  -- Proof omitted
  sorry

end jason_total_spent_l573_57384


namespace ellipse_foci_distance_2sqrt21_l573_57378

noncomputable def ellipse_foci_distance (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance_2sqrt21 :
  let center : ℝ × ℝ := (5, 2)
  let a := 5
  let b := 2
  ellipse_foci_distance a b = 2 * Real.sqrt 21 :=
by
  sorry

end ellipse_foci_distance_2sqrt21_l573_57378


namespace describe_graph_l573_57376

theorem describe_graph :
  ∀ (x y : ℝ), ((x + y) ^ 2 = x ^ 2 + y ^ 2 + 4 * x) ↔ (x = 0 ∨ y = 2) := 
by
  sorry

end describe_graph_l573_57376


namespace smallest_solution_to_equation_l573_57343

noncomputable def smallest_solution := (11 - Real.sqrt 445) / 6

theorem smallest_solution_to_equation:
  ∃ x : ℝ, (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ (x = smallest_solution) :=
sorry

end smallest_solution_to_equation_l573_57343


namespace find_n_l573_57314

theorem find_n (N : ℕ) (hN : 10 ≤ N ∧ N < 100)
  (h : ∃ a b : ℕ, N = 10 * a + b ∧ 4 * a + 2 * b = N / 2) : 
  N = 32 ∨ N = 64 ∨ N = 96 :=
sorry

end find_n_l573_57314


namespace inequality_solution_l573_57340

theorem inequality_solution (x : ℝ) :
  (x * (x + 2) > x * (3 - x) + 1) ↔ (x < -1/2 ∨ x > 1) :=
by sorry

end inequality_solution_l573_57340


namespace twenty_kopeck_greater_than_ten_kopeck_l573_57388

-- Definitions of the conditions
variables (x y z : ℕ)
axiom total_coins : x + y + z = 30 
axiom total_value : 10 * x + 15 * y + 20 * z = 500 

-- The proof statement
theorem twenty_kopeck_greater_than_ten_kopeck : z > x :=
sorry

end twenty_kopeck_greater_than_ten_kopeck_l573_57388


namespace shaded_area_l573_57352

theorem shaded_area (R : ℝ) (r : ℝ) (hR : R = 10) (hr : r = R / 2) : 
  π * R^2 - 2 * (π * r^2) = 50 * π :=
by
  sorry

end shaded_area_l573_57352


namespace lines_are_parallel_l573_57345

-- Definitions of the conditions
variable (θ a p : Real)
def line1 := θ = a
def line2 := p * Real.sin (θ - a) = 1

-- The proof problem: Prove the two lines are parallel
theorem lines_are_parallel (h1 : line1 θ a) (h2 : line2 θ a p) : False :=
by
  sorry

end lines_are_parallel_l573_57345


namespace coffee_blend_l573_57342

variable (pA pB : ℝ) (cA cB : ℝ) (total_cost : ℝ) 

theorem coffee_blend (hA : pA = 4.60) 
                     (hB : pB = 5.95) 
                     (h_ratio : cB = 2 * cA) 
                     (h_total : 4.60 * cA + 5.95 * cB = 511.50) : 
                     cA = 31 := 
by
  sorry

end coffee_blend_l573_57342


namespace total_reactions_eq_100_l573_57369

variable (x : ℕ) -- Total number of reactions.
variable (thumbs_up : ℕ) -- Number of "thumbs up" reactions.
variable (thumbs_down : ℕ) -- Number of "thumbs down" reactions.
variable (S : ℕ) -- Net Score.

-- Conditions
axiom thumbs_up_eq_75percent_reactions : thumbs_up = 3 * x / 4
axiom thumbs_down_eq_25percent_reactions : thumbs_down = x / 4
axiom score_definition : S = thumbs_up - thumbs_down
axiom initial_score : S = 50

theorem total_reactions_eq_100 : x = 100 :=
by 
  sorry

end total_reactions_eq_100_l573_57369


namespace prob_equal_even_odd_dice_l573_57370

def even_number_probability : ℚ := 1 / 2
def odd_number_probability : ℚ := 1 / 2

def probability_equal_even_odd (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k) * (even_number_probability) ^ n

theorem prob_equal_even_odd_dice : 
  probability_equal_even_odd 8 4 = 35 / 128 :=
by
  sorry

end prob_equal_even_odd_dice_l573_57370


namespace trip_time_total_l573_57308

noncomputable def wrong_direction_time : ℝ := 75 / 60
noncomputable def return_time : ℝ := 75 / 45
noncomputable def normal_trip_time : ℝ := 250 / 45

theorem trip_time_total :
  wrong_direction_time + return_time + normal_trip_time = 8.48 := by
  sorry

end trip_time_total_l573_57308


namespace probability_A_selected_l573_57381

def n : ℕ := 5
def k : ℕ := 2

def total_ways : ℕ := Nat.choose n k  -- C(n, k)

def favorable_ways : ℕ := Nat.choose (n - 1) (k - 1)  -- C(n-1, k-1)

theorem probability_A_selected : (favorable_ways : ℚ) / (total_ways : ℚ) = 2 / 5 :=
by
  sorry

end probability_A_selected_l573_57381


namespace pencil_total_length_l573_57371

-- Definitions of the colored sections
def purple_length : ℝ := 3.5
def black_length : ℝ := 2.8
def blue_length : ℝ := 1.6
def green_length : ℝ := 0.9
def yellow_length : ℝ := 1.2

-- The theorem stating the total length of the pencil
theorem pencil_total_length : purple_length + black_length + blue_length + green_length + yellow_length = 10 := 
by
  sorry

end pencil_total_length_l573_57371


namespace triangle_max_area_in_quarter_ellipse_l573_57389

theorem triangle_max_area_in_quarter_ellipse (a b c : ℝ) (h : c^2 = a^2 - b^2) :
  ∃ (T_max : ℝ), T_max = b / 2 :=
by sorry

end triangle_max_area_in_quarter_ellipse_l573_57389


namespace no_solution_abs_eq_quadratic_l573_57333

theorem no_solution_abs_eq_quadratic (x : ℝ) : ¬ (|x - 4| = x^2 + 6 * x + 8) :=
by
  sorry

end no_solution_abs_eq_quadratic_l573_57333


namespace cheesecake_factory_savings_l573_57301

noncomputable def combined_savings : ℕ := 3000

theorem cheesecake_factory_savings :
  let hourly_wage := 10
  let daily_hours := 10
  let working_days := 5
  let weekly_hours := daily_hours * working_days
  let weekly_salary := weekly_hours * hourly_wage
  let robby_savings := (2/5 : ℚ) * weekly_salary
  let jaylen_savings := (3/5 : ℚ) * weekly_salary
  let miranda_savings := (1/2 : ℚ) * weekly_salary
  let combined_weekly_savings := robby_savings + jaylen_savings + miranda_savings
  4 * combined_weekly_savings = combined_savings :=
by
  sorry

end cheesecake_factory_savings_l573_57301


namespace circumscribed_circle_radius_l573_57366

theorem circumscribed_circle_radius (r : ℝ) (π : ℝ)
  (isosceles_right_triangle : Type) 
  (perimeter : isosceles_right_triangle → ℝ )
  (area : ℝ → ℝ)
  (h : ∀ (t : isosceles_right_triangle), perimeter t = area r) :
  r = (1 + Real.sqrt 2) / π :=
sorry

end circumscribed_circle_radius_l573_57366


namespace percentage_of_whole_l573_57375

theorem percentage_of_whole (part whole percent : ℕ) (h1 : part = 120) (h2 : whole = 80) (h3 : percent = 150) : 
  part = (percent / 100) * whole :=
by
  sorry

end percentage_of_whole_l573_57375


namespace part_a_part_b_l573_57318

-- Part (a)
theorem part_a (n : ℕ) (a b : ℝ) : 
  a^(n+1) + b^(n+1) = (a + b) * (a^n + b^n) - a * b * (a^(n - 1) + b^(n - 1)) :=
by sorry

-- Part (b)
theorem part_b {a b : ℝ} (h1 : a + b = 1) (h2: a * b = -1) : 
  a^10 + b^10 = 123 :=
by sorry

end part_a_part_b_l573_57318


namespace total_insect_legs_l573_57335

/--
This Lean statement defines the conditions and question,
proving that given 5 insects in the laboratory and each insect
having 6 legs, the total number of insect legs is 30.
-/
theorem total_insect_legs (n_insects : Nat) (legs_per_insect : Nat) (h1 : n_insects = 5) (h2 : legs_per_insect = 6) : (n_insects * legs_per_insect) = 30 :=
by
  sorry

end total_insect_legs_l573_57335


namespace percentage_passed_in_both_l573_57351

def percentage_of_students_failing_hindi : ℝ := 30
def percentage_of_students_failing_english : ℝ := 42
def percentage_of_students_failing_both : ℝ := 28

theorem percentage_passed_in_both (P_H_E: percentage_of_students_failing_hindi + percentage_of_students_failing_english - percentage_of_students_failing_both = 44) : 
  100 - (percentage_of_students_failing_hindi + percentage_of_students_failing_english - percentage_of_students_failing_both) = 56 := by
  sorry

end percentage_passed_in_both_l573_57351


namespace mabel_tomatoes_l573_57311

theorem mabel_tomatoes (n1 n2 n3 n4 : ℕ)
  (h1 : n1 = 8)
  (h2 : n2 = n1 + 4)
  (h3 : n3 = 3 * (n1 + n2))
  (h4 : n4 = 3 * (n1 + n2)) :
  n1 + n2 + n3 + n4 = 140 := by
  sorry

end mabel_tomatoes_l573_57311


namespace geometric_sequence_y_l573_57324

theorem geometric_sequence_y (x y z : ℝ) (h1 : 1 ≠ 0) (h2 : x ≠ 0) (h3 : y ≠ 0) (h4 : z ≠ 0) (h5 : 9 ≠ 0)
  (h_seq : ∀ a b c d e : ℝ, (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ a * e = b * d ∧ b * d = c^2) →
           (a, b, c, d, e) = (1, x, y, z, 9)) :
  y = 3 :=
sorry

end geometric_sequence_y_l573_57324


namespace complex_number_purely_imaginary_l573_57326

theorem complex_number_purely_imaginary (a : ℝ) (i : ℂ) (h₁ : (a^2 - a - 2 : ℝ) = 0) (h₂ : (a + 1 ≠ 0)) : a = 2 := 
by {
  sorry
}

end complex_number_purely_imaginary_l573_57326


namespace digit_for_divisibility_by_45_l573_57399

theorem digit_for_divisibility_by_45 (n : ℕ) (h₀ : n < 10)
  (h₁ : 5 ∣ (5 + 10 * (7 + 4 * (1 + 5 * (8 + n))))) 
  (h₂ : 9 ∣ (5 + 7 + 4 + n + 5 + 8)) : 
  n = 7 :=
by { sorry }

end digit_for_divisibility_by_45_l573_57399


namespace albums_not_in_both_l573_57397

-- Definitions representing the problem conditions
def andrew_albums : ℕ := 23
def common_albums : ℕ := 11
def john_unique_albums : ℕ := 8

-- Proof statement (not the actual proof)
theorem albums_not_in_both : 
  (andrew_albums - common_albums) + john_unique_albums = 20 :=
by
  sorry

end albums_not_in_both_l573_57397


namespace locus_of_points_line_or_point_l573_57380

theorem locus_of_points_line_or_point {n : ℕ} (A B : ℕ → ℝ) (k : ℝ) (h : ∀ i, 1 ≤ i ∧ i < n → (A (i + 1) - A i) / (B (i + 1) - B i) = k) :
  ∃ l : ℝ, ∀ i, 1 ≤ i ∧ i ≤ n → (A i + l*B i) = A 1 + l*B 1 :=
by
  sorry

end locus_of_points_line_or_point_l573_57380


namespace score_seventy_five_can_be_achieved_three_ways_l573_57312

-- Defining the problem constraints and goal
def quiz_problem (c u i : ℕ) (S : ℝ) : Prop :=
  c + u + i = 20 ∧ S = 5 * (c : ℝ) + 1.5 * (u : ℝ)

theorem score_seventy_five_can_be_achieved_three_ways :
  ∃ (c1 u1 c2 u2 c3 u3 : ℕ), 0 ≤ (5 * (c1 : ℝ) + 1.5 * (u1 : ℝ)) ∧ (5 * (c1 : ℝ) + 1.5 * (u1 : ℝ)) ≤ 100 ∧
  (5 * (c2 : ℝ) + 1.5 * (u2 : ℝ)) = 75 ∧ (5 * (c3 : ℝ) + 1.5 * (u3 : ℝ)) = 75 ∧
  (c1 ≠ c2 ∧ u1 ≠ u2) ∧ (c2 ≠ c3 ∧ u2 ≠ u3) ∧ (c3 ≠ c1 ∧ u3 ≠ u1) ∧ 
  quiz_problem c1 u1 (20 - c1 - u1) 75 ∧
  quiz_problem c2 u2 (20 - c2 - u2) 75 ∧
  quiz_problem c3 u3 (20 - c3 - u3) 75 :=
sorry

end score_seventy_five_can_be_achieved_three_ways_l573_57312


namespace paper_plates_cost_l573_57349

theorem paper_plates_cost (P C x : ℝ) 
(h1 : 100 * P + 200 * C = 6.00) 
(h2 : x * P + 40 * C = 1.20) : 
x = 20 := 
sorry

end paper_plates_cost_l573_57349


namespace greatest_y_value_l573_57395

theorem greatest_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -6) : y ≤ 3 :=
sorry

end greatest_y_value_l573_57395


namespace correct_sum_of_satisfying_values_l573_57358

def g (x : Nat) : Nat :=
  match x with
  | 0 => 0
  | 1 => 2
  | 2 => 1
  | _ => 0  -- This handles the out-of-bounds case, though it's not needed here

def f (x : Nat) : Nat :=
  match x with
  | 0 => 2
  | 1 => 1
  | 2 => 0
  | _ => 0  -- This handles the out-of-bounds case, though it's not needed here

def satisfies_condition (x : Nat) : Bool :=
  f (g x) > g (f x)

def sum_of_satisfying_values : Nat :=
  List.sum (List.filter satisfies_condition [0, 1, 2])

theorem correct_sum_of_satisfying_values : sum_of_satisfying_values = 2 :=
  sorry

end correct_sum_of_satisfying_values_l573_57358


namespace ratio_sum_l573_57360

theorem ratio_sum {x y : ℚ} (h : x / y = 4 / 7) : (x + y) / y = 11 / 7 :=
sorry

end ratio_sum_l573_57360


namespace estimate_fitness_population_l573_57325

theorem estimate_fitness_population :
  ∀ (sample_size total_population : ℕ) (sample_met_standards : Nat) (percentage_met_standards estimated_met_standards : ℝ),
  sample_size = 1000 →
  total_population = 1200000 →
  sample_met_standards = 950 →
  percentage_met_standards = (sample_met_standards : ℝ) / (sample_size : ℝ) →
  estimated_met_standards = percentage_met_standards * (total_population : ℝ) →
  estimated_met_standards = 1140000 := by sorry

end estimate_fitness_population_l573_57325


namespace spider_legs_total_l573_57346

def num_spiders : ℕ := 4
def legs_per_spider : ℕ := 8
def total_legs : ℕ := num_spiders * legs_per_spider

theorem spider_legs_total : total_legs = 32 := by
  sorry -- proof is skipped with 'sorry'

end spider_legs_total_l573_57346


namespace proof_problem_l573_57331

-- Given conditions
variables {a b : Type}  -- Two non-coincident lines
variables {α β : Type}  -- Two non-coincident planes

-- Definitions of the relationships
def is_parallel_to (x y : Type) : Prop := sorry  -- Parallel relationship
def is_perpendicular_to (x y : Type) : Prop := sorry  -- Perpendicular relationship

-- Statements to verify
def statement1 (a α b : Type) : Prop := 
  (is_parallel_to a α ∧ is_parallel_to b α) → is_parallel_to a b

def statement2 (a α β : Type) : Prop :=
  (is_perpendicular_to a α ∧ is_perpendicular_to a β) → is_parallel_to α β

def statement3 (α β : Type) : Prop :=
  is_perpendicular_to α β → ∃ l : Type, is_perpendicular_to l α ∧ is_parallel_to l β

def statement4 (α β : Type) : Prop :=
  is_perpendicular_to α β → ∃ γ : Type, is_perpendicular_to γ α ∧ is_perpendicular_to γ β

-- Proof problem: verifying which statements are true.
theorem proof_problem :
  ¬ (statement1 a α b) ∧ statement2 a α β ∧ statement3 α β ∧ statement4 α β :=
by
  sorry

end proof_problem_l573_57331


namespace cost_of_paving_l573_57382

theorem cost_of_paving (L W R : ℝ) (hL : L = 6.5) (hW : W = 2.75) (hR : R = 600) : 
  L * W * R = 10725 := by
  rw [hL, hW, hR]
  -- To solve the theorem successively
  -- we would need to verify the product of the values
  -- given by the conditions.
  sorry

end cost_of_paving_l573_57382


namespace DebateClubOfficerSelection_l573_57356

-- Definitions based on the conditions
def members : Finset ℕ := Finset.range 25 -- Members are indexed from 0 to 24
def Simon := 0
def Rachel := 1
def John := 2

-- Conditions regarding the officers
def is_officer (x : ℕ) (pres sec tre : ℕ) : Prop := 
  x = pres ∨ x = sec ∨ x = tre

def Simon_condition (pres sec tre : ℕ) : Prop :=
  (is_officer Simon pres sec tre) → (is_officer Rachel pres sec tre)

def Rachel_condition (pres sec tre : ℕ) : Prop :=
  (is_officer Rachel pres sec tre) → (is_officer Simon pres sec tre) ∨ (is_officer John pres sec tre)

-- Statement of the problem in Lean
theorem DebateClubOfficerSelection : ∃ (pres sec tre : ℕ), 
  pres ≠ sec ∧ sec ≠ tre ∧ pres ≠ tre ∧ 
  pres ∈ members ∧ sec ∈ members ∧ tre ∈ members ∧ 
  Simon_condition pres sec tre ∧
  Rachel_condition pres sec tre :=
sorry

end DebateClubOfficerSelection_l573_57356


namespace meet_time_l573_57307

theorem meet_time 
  (circumference : ℝ) 
  (deepak_speed_kmph : ℝ) 
  (wife_speed_kmph : ℝ) 
  (deepak_speed_mpm : ℝ := deepak_speed_kmph * 1000 / 60) 
  (wife_speed_mpm : ℝ := wife_speed_kmph * 1000 / 60) 
  (relative_speed : ℝ := deepak_speed_mpm + wife_speed_mpm)
  (time_to_meet : ℝ := circumference / relative_speed) :
  circumference = 660 → 
  deepak_speed_kmph = 4.5 → 
  wife_speed_kmph = 3.75 → 
  time_to_meet = 4.8 :=
by 
  intros h1 h2 h3 
  sorry

end meet_time_l573_57307


namespace equilateral_triangles_count_in_grid_of_side_4_l573_57329

-- Define a function to calculate the number of equilateral triangles in a triangular grid of side length n
def countEquilateralTriangles (n : ℕ) : ℕ :=
  (n * (n + 1) * (n + 2) * (n + 3)) / 24

-- Define the problem statement for n = 4
theorem equilateral_triangles_count_in_grid_of_side_4 :
  countEquilateralTriangles 4 = 35 := by
  sorry

end equilateral_triangles_count_in_grid_of_side_4_l573_57329


namespace b_profit_share_l573_57305

theorem b_profit_share (total_capital : ℝ) (profit : ℝ) (A_invest : ℝ) (B_invest : ℝ) (C_invest : ℝ) (D_invest : ℝ)
 (A_time : ℝ) (B_time : ℝ) (C_time : ℝ) (D_time : ℝ) :
  total_capital = 100000 ∧
  A_invest = B_invest + 10000 ∧
  B_invest = C_invest + 5000 ∧
  D_invest = A_invest + 8000 ∧
  A_time = 12 ∧
  B_time = 10 ∧
  C_time = 8 ∧
  D_time = 6 ∧
  profit = 50000 →
  (B_invest * B_time / (A_invest * A_time + B_invest * B_time + C_invest * C_time + D_invest * D_time)) * profit = 10925 :=
by
  sorry

end b_profit_share_l573_57305


namespace sin_2gamma_proof_l573_57377

-- Assume necessary definitions and conditions
variables {A B C D P : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited P]
variables (a b c d: ℝ)
variables (α β γ: ℝ)

-- Assume points A, B, C, D, P lie on a circle in that order and AB = BC = CD
axiom points_on_circle : a = b ∧ b = c ∧ c = d
axiom cos_apc : Real.cos α = 3/5
axiom cos_bpd : Real.cos β = 1/5

noncomputable def sin_2gamma : ℝ :=
  2 * Real.sin γ * Real.cos γ

-- Statement to prove sin(2 * γ) given the conditions
theorem sin_2gamma_proof : sin_2gamma γ = 8 * Real.sqrt 5 / 25 :=
sorry

end sin_2gamma_proof_l573_57377


namespace number_of_perfect_square_divisors_of_450_l573_57387

theorem number_of_perfect_square_divisors_of_450 : 
    let p := 450;
    let factors := [(3, 2), (5, 2), (2, 1)];
    ∃ n, (n = 4 ∧ 
          ∀ (d : ℕ), d ∣ p → 
                     (∃ (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ 
                              (a = 0) ∧ (b = 0 ∨ b = 2) ∧ (c = 0 ∨ c = 2) → 
                              a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0)) :=
    sorry

end number_of_perfect_square_divisors_of_450_l573_57387


namespace James_total_area_l573_57338

def initial_length : ℕ := 13
def initial_width : ℕ := 18
def increase_dimension : ℕ := 2
def number_of_new_rooms : ℕ := 4
def larger_room_multiplier : ℕ := 2

noncomputable def new_length : ℕ := initial_length + increase_dimension
noncomputable def new_width : ℕ := initial_width + increase_dimension
noncomputable def area_of_one_new_room : ℕ := new_length * new_width
noncomputable def total_area_of_4_rooms : ℕ := area_of_one_new_room * number_of_new_rooms
noncomputable def area_of_larger_room : ℕ := area_of_one_new_room * larger_room_multiplier
noncomputable def total_area : ℕ := total_area_of_4_rooms + area_of_larger_room

theorem James_total_area : total_area = 1800 := 
by
  sorry

end James_total_area_l573_57338


namespace percentage_female_officers_on_duty_correct_l573_57316

-- Define the conditions
def total_officers_on_duty : ℕ := 144
def total_female_officers : ℕ := 400
def female_officers_on_duty : ℕ := total_officers_on_duty / 2

-- Define the percentage calculation
def percentage_female_officers_on_duty : ℕ :=
  (female_officers_on_duty * 100) / total_female_officers

-- The theorem that what we need to prove
theorem percentage_female_officers_on_duty_correct :
  percentage_female_officers_on_duty = 18 :=
by
  sorry

end percentage_female_officers_on_duty_correct_l573_57316


namespace point_B_not_on_curve_C_l573_57315

theorem point_B_not_on_curve_C {a : ℝ} : 
  ¬ ((2 * a) ^ 2 + (4 * a) ^ 2 + 6 * a * (2 * a) - 8 * a * (4 * a) = 0) :=
by 
  sorry

end point_B_not_on_curve_C_l573_57315


namespace triangle_value_l573_57341

variable (triangle p : ℝ)

theorem triangle_value : (triangle + p = 75 ∧ 3 * (triangle + p) - p = 198) → triangle = 48 :=
by
  sorry

end triangle_value_l573_57341


namespace smallest_number_conditions_l573_57332

theorem smallest_number_conditions :
  ∃ n : ℤ, (n > 0) ∧
           (n % 2 = 1) ∧
           (n % 3 = 1) ∧
           (n % 4 = 1) ∧
           (n % 5 = 1) ∧
           (n % 6 = 1) ∧
           (n % 11 = 0) ∧
           (∀ m : ℤ, (m > 0) → 
             (m % 2 = 1) ∧
             (m % 3 = 1) ∧
             (m % 4 = 1) ∧
             (m % 5 = 1) ∧
             (m % 6 = 1) ∧
             (m % 11 = 0) → 
             (n ≤ m)) :=
sorry

end smallest_number_conditions_l573_57332


namespace calculate_f_f_2_l573_57322

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 3 * x ^ 2 - 4
else if x = 0 then 2
else -1

theorem calculate_f_f_2 : f (f 2) = 188 :=
by
  sorry

end calculate_f_f_2_l573_57322


namespace janet_speed_l573_57327

def janet_sister_speed : ℝ := 12
def lake_width : ℝ := 60
def wait_time : ℝ := 3

theorem janet_speed :
  (lake_width / (lake_width / janet_sister_speed - wait_time)) = 30 := 
sorry

end janet_speed_l573_57327


namespace total_seeds_l573_57379

theorem total_seeds (A B C : ℕ) (h₁ : A = B + 10) (h₂ : B = 30) (h₃ : C = 30) : A + B + C = 100 :=
by
  sorry

end total_seeds_l573_57379


namespace complex_multiplication_l573_57372

def imaginary_unit := Complex.I

theorem complex_multiplication (h : imaginary_unit^2 = -1) : (3 + 2 * imaginary_unit) * imaginary_unit = -2 + 3 * imaginary_unit :=
by
  sorry

end complex_multiplication_l573_57372


namespace ranking_of_ABC_l573_57334

-- Define the ranking type
inductive Rank
| first
| second
| third

-- Define types for people
inductive Person
| A
| B
| C

open Rank Person

-- Alias for ranking of each person
def ranking := Person → Rank

-- Define the conditions
def A_statement (r : ranking) : Prop := r A ≠ first
def B_statement (r : ranking) : Prop := A_statement r ≠ false
def C_statement (r : ranking) : Prop := r C ≠ third

def B_lied : Prop := true
def C_told_truth : Prop := true

-- The equivalent problem, asked to prove the final result
theorem ranking_of_ABC (r : ranking) : 
  (B_lied ∧ C_told_truth ∧ B_statement r = false ∧ C_statement r = true) → 
  (r A = first ∧ r B = third ∧ r C = second) :=
sorry

end ranking_of_ABC_l573_57334


namespace riverside_high_badges_l573_57355

/-- Given the conditions on the sums of consecutive prime badge numbers of the debate team members,
prove that Giselle's badge number is 1014, given that the current year is 2025.
-/
theorem riverside_high_badges (p1 p2 p3 p4 : ℕ) (hp1 : Prime p1) (hp2 : Prime p2) (hp3 : Prime p3) (hp4 : Prime p4)
    (hconsec : p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧ p4 = p3 + 6)
    (h1 : ∃ x, p1 + p3 = x) (h2 : ∃ y, p1 + p2 = y) (h3 : ∃ z, p2 + p3 = z ∧ z ≤ 31) 
    (h4 : p3 + p4 = 2025) : p4 = 1014 :=
by sorry

end riverside_high_badges_l573_57355


namespace spent_more_on_candy_bar_l573_57319

-- Definitions of conditions
def money_Dan_has : ℕ := 2
def candy_bar_cost : ℕ := 6
def chocolate_cost : ℕ := 3

-- Statement of the proof problem
theorem spent_more_on_candy_bar : candy_bar_cost - chocolate_cost = 3 := by
  sorry

end spent_more_on_candy_bar_l573_57319


namespace simplify_polynomial_l573_57309

theorem simplify_polynomial (x : ℤ) :
  (3 * x - 2) * (6 * x^12 + 3 * x^11 + 5 * x^9 + x^8 + 7 * x^7) =
  18 * x^13 - 3 * x^12 + 15 * x^10 - 7 * x^9 + 19 * x^8 - 14 * x^7 :=
by
  sorry

end simplify_polynomial_l573_57309


namespace triangle_is_isosceles_l573_57364

theorem triangle_is_isosceles 
  (A B C : ℝ) 
  (h : (Real.sin A + Real.sin B) * (Real.cos A + Real.cos B) = 2 * Real.sin C) 
  (h₀ : A + B + C = π) :
  (A = B) := 
sorry

end triangle_is_isosceles_l573_57364


namespace Pyarelal_loss_is_1800_l573_57303

noncomputable def Ashok_and_Pyarelal_loss (P L : ℝ) : Prop :=
  let Ashok_cap := (1 / 9) * P
  let total_cap := P + Ashok_cap
  let Pyarelal_ratio := P / total_cap
  let total_loss := 2000
  let Pyarelal_loss := Pyarelal_ratio * total_loss
  Pyarelal_loss = 1800

theorem Pyarelal_loss_is_1800 (P : ℝ) (h1 : P > 0) (h2 : L = 2000) :
  Ashok_and_Pyarelal_loss P L := sorry

end Pyarelal_loss_is_1800_l573_57303


namespace parallel_lines_necessary_not_sufficient_l573_57321

variables {a1 b1 a2 b2 c1 c2 : ℝ}

def determinant (a1 b1 a2 b2 : ℝ) : ℝ := a1 * b2 - a2 * b1

theorem parallel_lines_necessary_not_sufficient
  (h1 : a1^2 + b1^2 ≠ 0)
  (h2 : a2^2 + b2^2 ≠ 0)
  : (determinant a1 b1 a2 b2 = 0) → 
    (a1 * x + b1 * y + c1 = 0 ∧ a2 * x + b2 * y + c2 =0 → exists k : ℝ, (a1 = k ∧ b1 = k)) ∧ 
    (determinant a1 b1 a2 b2 = 0 → (a2 * x + b2 * y + c2 = a1 * x + b1 * y + c1 → false)) :=
sorry

end parallel_lines_necessary_not_sufficient_l573_57321
