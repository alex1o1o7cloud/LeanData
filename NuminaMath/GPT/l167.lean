import Mathlib

namespace part_one_part_two_l167_167892

-- Define the function f
def f (x : ℝ) : ℝ := abs (x - 2) + abs (x + 1)

-- Define the inequality condition
def inequality_condition (x : ℝ) : Prop := f x ≥ 4 - x

-- Problem set (I)
theorem part_one (x : ℝ) : inequality_condition x ↔ (x ≤ -3 ∨ x ≥ 1) :=
sorry

-- Define range conditions for a and b
def range_condition (a b : ℝ) : Prop := a ≥ 3 ∧ b ≥ 3

-- Problem set (II)
theorem part_two (a b : ℝ) (h : range_condition a b) : 2 * (a + b) < a * b + 4 :=
sorry

end part_one_part_two_l167_167892


namespace probability_of_9_heads_in_12_flips_l167_167483

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| _, 0       => 1
| 0, _       => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^12

-- Define the number of ways to get exactly 9 heads in 12 flips
def favorable_outcomes : ℕ := binom 12 9

-- Define the probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 220/4096
theorem probability_of_9_heads_in_12_flips : probability = 220 / 4096 := by
  sorry

end probability_of_9_heads_in_12_flips_l167_167483


namespace soccer_team_students_l167_167341

theorem soccer_team_students :
  ∀ (n p b m : ℕ),
    n = 25 →
    p = 10 →
    b = 6 →
    n - (p - b) = m →
    m = 21 :=
by
  intros n p b m h_n h_p h_b h_trivial
  sorry

end soccer_team_students_l167_167341


namespace find_m_l167_167601

variables (m x y : ℤ)

-- Conditions
def cond1 := x = 3 * m + 1
def cond2 := y = 2 * m - 2
def cond3 := 4 * x - 3 * y = 10

theorem find_m (h1 : cond1 m x) (h2 : cond2 m y) (h3 : cond3 x y) : m = 0 :=
by sorry

end find_m_l167_167601


namespace girls_with_brown_eyes_and_light_brown_skin_l167_167008

theorem girls_with_brown_eyes_and_light_brown_skin 
  (total_girls : ℕ)
  (light_brown_skin_girls : ℕ)
  (blue_eyes_fair_skin_girls : ℕ)
  (brown_eyes_total : ℕ)
  (total_girls_50 : total_girls = 50)
  (light_brown_skin_31 : light_brown_skin_girls = 31)
  (blue_eyes_fair_skin_14 : blue_eyes_fair_skin_girls = 14)
  (brown_eyes_18 : brown_eyes_total = 18) :
  ∃ (brown_eyes_light_brown_skin_girls : ℕ), brown_eyes_light_brown_skin_girls = 13 :=
by sorry

end girls_with_brown_eyes_and_light_brown_skin_l167_167008


namespace degree_reduction_l167_167705

theorem degree_reduction (x : ℝ) (h1 : x^2 = x + 1) (h2 : 0 < x) : x^4 - 2 * x^3 + 3 * x = 1 + Real.sqrt 5 :=
by
  sorry

end degree_reduction_l167_167705


namespace total_puppies_is_74_l167_167189

-- Define the number of puppies adopted each week based on the given conditions
def number_of_puppies_first_week : Nat := 20
def number_of_puppies_second_week : Nat := 2 * number_of_puppies_first_week / 5
def number_of_puppies_third_week : Nat := 2 * number_of_puppies_second_week
def number_of_puppies_fourth_week : Nat := number_of_puppies_first_week + 10

-- Define the total number of puppies
def total_number_of_puppies : Nat :=
  number_of_puppies_first_week + number_of_puppies_second_week + number_of_puppies_third_week + number_of_puppies_fourth_week

-- Proof statement: Prove that the total number of puppies is 74
theorem total_puppies_is_74 : total_number_of_puppies = 74 := by
  sorry

end total_puppies_is_74_l167_167189


namespace solve_for_x_l167_167148

theorem solve_for_x : ∀ (x : ℝ), (4 * x^2 - 3 * x + 2) / (x + 2) = 4 * x - 5 → x = 2 :=
by
  intros x h
  sorry

end solve_for_x_l167_167148


namespace solution_set_of_inequality_l167_167865

theorem solution_set_of_inequality (a : ℝ) :
  (a > 1 → {x : ℝ | ax + 1 < a^2 + x} = {x : ℝ | x < a + 1}) ∧
  (a < 1 → {x : ℝ | ax + 1 < a^2 + x} = {x : ℝ | x > a + 1}) ∧
  (a = 1 → {x : ℝ | ax + 1 < a^2 + x} = ∅) := 
  sorry

end solution_set_of_inequality_l167_167865


namespace smallest_integer_y_l167_167308

theorem smallest_integer_y (y : ℤ) (h : 3 - 5 * y < 23) : -3 ≥ y :=
by {
  sorry
}

end smallest_integer_y_l167_167308


namespace probability_product_odd_l167_167835

theorem probability_product_odd :
  let A := {1, 2, 3}
  let B := {0, 1, 3}
  let outcomes := 3 * 3
  let favorable_outcomes := 2 * 2
  (favorable_outcomes : ℚ) / outcomes = 4 / 9 := by
{
  -- Definitions for A and B
  let A := {1, 2, 3}
  let B := {0, 1, 3}
  -- Total number of outcomes
  let outcomes := 3 * 3
  -- Favorable outcomes where both a and b are odd
  let favorable_outcomes := 2 * 2
  -- Expected probability
  have h : (favorable_outcomes : ℚ) / outcomes = 4 / 9 := sorry
  exact h
}

end probability_product_odd_l167_167835


namespace digits_with_five_or_seven_is_5416_l167_167597

/-- The total number of four-digit positive integers. -/
def total_four_digit_integers : ℕ := 9000

/-- The number of four-digit integers without the digits 5 or 7. -/
def no_five_seven_integers : ℕ := 3584

/-- The number of four-digit positive integers that have at least one digit that is a 5 or a 7. -/
def digits_with_five_or_seven : ℕ :=
  total_four_digit_integers - no_five_seven_integers

/-- Proof that the number of four-digit integers with at least one digit that is a 5 or 7 is 5416. -/
theorem digits_with_five_or_seven_is_5416 :
  digits_with_five_or_seven = 5416 :=
by
  sorry

end digits_with_five_or_seven_is_5416_l167_167597


namespace find_missing_number_l167_167111

theorem find_missing_number 
  (x : ℕ) 
  (avg : (744 + 745 + 747 + 748 + 749 + some_num + 753 + 755 + x) / 9 = 750)
  (hx : x = 755) : 
  some_num = 804 := 
  sorry

end find_missing_number_l167_167111


namespace total_books_l167_167872

theorem total_books (books_last_month : ℕ) (goal_factor : ℕ) (books_this_month : ℕ) (total_books : ℕ) 
  (h1 : books_last_month = 4) 
  (h2 : goal_factor = 2) 
  (h3 : books_this_month = goal_factor * books_last_month) 
  (h4 : total_books = books_last_month + books_this_month) 
  : total_books = 12 := 
by
  sorry

end total_books_l167_167872


namespace largest_possible_perimeter_l167_167335

theorem largest_possible_perimeter (x : ℕ) (h1 : 1 < x) (h2 : x < 15) : 
  (7 + 8 + x) ≤ 29 := 
sorry

end largest_possible_perimeter_l167_167335


namespace probability_neither_orange_nor_white_l167_167528

/-- Define the problem conditions. -/
def num_orange_balls : ℕ := 8
def num_black_balls : ℕ := 7
def num_white_balls : ℕ := 6

/-- Define the total number of balls. -/
def total_balls : ℕ := num_orange_balls + num_black_balls + num_white_balls

/-- Define the probability of picking a black ball (neither orange nor white). -/
noncomputable def probability_black_ball : ℚ := num_black_balls / total_balls

/-- The main statement to be proved: The probability is 1/3. -/
theorem probability_neither_orange_nor_white : probability_black_ball = 1 / 3 :=
sorry

end probability_neither_orange_nor_white_l167_167528


namespace distance_from_dorm_to_city_l167_167772

theorem distance_from_dorm_to_city (D : ℚ) (h1 : (1/3) * D = (1/3) * D) (h2 : (3/5) * D = (3/5) * D) (h3 : D - ((1 / 3) * D + (3 / 5) * D) = 2) :
  D = 30 := 
by sorry

end distance_from_dorm_to_city_l167_167772


namespace find_B_l167_167016

theorem find_B (A B : ℝ) : (1 / 4 * 1 / 8 = 1 / (4 * A) ∧ 1 / 32 = 1 / B) → B = 32 := by
  intros h
  sorry

end find_B_l167_167016


namespace factor_expression_l167_167044

theorem factor_expression (x : ℝ) :
  (16 * x^4 + 36 * x^2 - 9) - (4 * x^4 - 6 * x^2 - 9) = 6 * x^2 * (2 * x^2 + 7) :=
by
  sorry

end factor_expression_l167_167044


namespace xy_div_eq_one_third_l167_167252

theorem xy_div_eq_one_third (x y z : ℝ) 
  (h1 : x + y = 2 * x + z)
  (h2 : x - 2 * y = 4 * z)
  (h3 : x + y + z = 21)
  (h4 : y / z = 6) : 
  x / y = 1 / 3 :=
by
  sorry

end xy_div_eq_one_third_l167_167252


namespace smallest_period_pi_max_value_min_value_l167_167254

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + Real.cos (2 * x)

open Real

theorem smallest_period_pi : ∀ x, f (x + π) = f x := by
  unfold f
  intros
  sorry

theorem max_value : ∀ x, 0 ≤ x ∧ x ≤ π / 2 → f x ≤ 1 + sqrt 2 := by
  unfold f
  intros
  sorry

theorem min_value : ∀ x, 0 ≤ x ∧ x ≤ π / 2 → f x ≥ 0 := by
  unfold f
  intros
  sorry

end smallest_period_pi_max_value_min_value_l167_167254


namespace complement_U_A_l167_167390

def U : Set ℝ := { x | x^2 ≤ 4 }
def A : Set ℝ := { x | abs (x + 1) ≤ 1 }

theorem complement_U_A :
  (U \ A) = { x | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end complement_U_A_l167_167390


namespace ferry_speed_difference_l167_167061

open Nat

-- Define the time and speed of ferry P
def timeP := 3 -- hours
def speedP := 8 -- kilometers per hour

-- Define the distance of ferry P
def distanceP := speedP * timeP -- kilometers

-- Define the distance of ferry Q
def distanceQ := 3 * distanceP -- kilometers

-- Define the time of ferry Q
def timeQ := timeP + 5 -- hours

-- Define the speed of ferry Q
def speedQ := distanceQ / timeQ -- kilometers per hour

-- Define the speed difference
def speedDifference := speedQ - speedP -- kilometers per hour

-- The target theorem to prove
theorem ferry_speed_difference : speedDifference = 1 := by
  sorry

end ferry_speed_difference_l167_167061


namespace frog_arrangement_count_l167_167009

theorem frog_arrangement_count :
  let green_frogs := 2
  let red_frogs := 3
  let blue_frogs := 2
  let frogs := green_frogs + red_frogs + blue_frogs
  -- Descriptions:
  -- 1. green_frogs refuse to sit next to red_frogs
  -- 2. green_frogs and red_frogs are fine sitting next to blue_frogs
  -- 3. blue_frogs can sit next to each other
  frogs = 7 → 
  ∃ arrangements : ℕ, arrangements = 72 :=
by 
  sorry

end frog_arrangement_count_l167_167009


namespace james_veg_consumption_l167_167920

-- Define the given conditions in Lean
def asparagus_per_day : ℝ := 0.25
def broccoli_per_day : ℝ := 0.25
def days_in_week : ℝ := 7
def weeks : ℝ := 2
def kale_per_week : ℝ := 3

-- Define the amount of vegetables (initial, doubled, and added kale)
def initial_veg_per_day := asparagus_per_day + broccoli_per_day
def initial_veg_per_week := initial_veg_per_day * days_in_week
def double_veg_per_week := initial_veg_per_week * weeks
def total_veg_per_week_after_kale := double_veg_per_week + kale_per_week

-- Statement of the proof problem
theorem james_veg_consumption :
  total_veg_per_week_after_kale = 10 := by 
  sorry

end james_veg_consumption_l167_167920


namespace spot_area_l167_167943

/-- Proving the area of the accessible region outside the doghouse -/
theorem spot_area
  (pentagon_side : ℝ)
  (rope_length : ℝ)
  (accessible_area : ℝ) 
  (h1 : pentagon_side = 1) 
  (h2 : rope_length = 3)
  (h3 : accessible_area = (37 * π) / 5) :
  accessible_area = (π * (rope_length^2) * (288 / 360)) + 2 * (π * (pentagon_side^2) * (36 / 360)) := 
  sorry

end spot_area_l167_167943


namespace ratio_proof_l167_167386

theorem ratio_proof (a b c : ℝ) (ha : b / a = 3) (hb : c / b = 4) :
    (a + 2 * b) / (b + 2 * c) = 7 / 27 := by
  sorry

end ratio_proof_l167_167386


namespace common_sum_of_matrix_l167_167306

theorem common_sum_of_matrix :
  let S := (1 / 2 : ℝ) * 25 * (10 + 34)
  let adjusted_total := S + 10
  let common_sum := adjusted_total / 6
  common_sum = 93.33 :=
by
  sorry

end common_sum_of_matrix_l167_167306


namespace count_four_digit_integers_with_5_or_7_l167_167589

theorem count_four_digit_integers_with_5_or_7 : 
  (set.range (λ n: ℕ, n >= 1000 ∧ n <= 9999) 
   \ { n | ∀ d ∈ toDigits n, d ≠ 5 ∧ d ≠ 7 }).card = 5416 :=
by
  sorry

end count_four_digit_integers_with_5_or_7_l167_167589


namespace probability_heads_exactly_9_of_12_l167_167464

noncomputable def bin_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_heads_9_of_12_flips : ℚ :=
  (bin_coeff 12 9 : ℚ) / (2 ^ 12)

theorem probability_heads_exactly_9_of_12 :
  probability_heads_9_of_12_flips = 220 / 4096 :=
by
  sorry

end probability_heads_exactly_9_of_12_l167_167464


namespace scientific_notation_of_0_000815_l167_167819

theorem scientific_notation_of_0_000815 :
  (∃ (c : ℝ) (n : ℤ), 0.000815 = c * 10^n ∧ 1 ≤ c ∧ c < 10) ∧ (0.000815 = 8.15 * 10^(-4)) :=
by
  sorry

end scientific_notation_of_0_000815_l167_167819


namespace find_k_l167_167899

-- Given vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- Vectors expressions
def k_a_add_b (k : ℝ) : ℝ × ℝ := (k * a.1 + b.1, k * a.2 + b.2)
def a_sub_3b : ℝ × ℝ := (a.1 - 3 * b.1, a.2 - 3 * b.2)

-- Condition of collinearity
def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  (v1.1 = 0 ∨ v2.1 = 0 ∨ v1.1 * v2.2 = v1.2 * v2.1)

-- Statement to prove
theorem find_k :
  collinear (k_a_add_b (-1/3)) a_sub_3b :=
sorry

end find_k_l167_167899


namespace dark_more_than_light_l167_167334

-- Define the board size
def board_size : ℕ := 9

-- Define the number of dark squares in odd rows
def dark_in_odd_row : ℕ := 5

-- Define the number of light squares in odd rows
def light_in_odd_row : ℕ := 4

-- Define the number of dark squares in even rows
def dark_in_even_row : ℕ := 4

-- Define the number of light squares in even rows
def light_in_even_row : ℕ := 5

-- Calculate the total number of dark squares
def total_dark_squares : ℕ := (dark_in_odd_row * ((board_size + 1) / 2)) + (dark_in_even_row * (board_size / 2))

-- Calculate the total number of light squares
def total_light_squares : ℕ := (light_in_odd_row * ((board_size + 1) / 2)) + (light_in_even_row * (board_size / 2))

-- Define the main theorem
theorem dark_more_than_light : total_dark_squares - total_light_squares = 1 := by
  sorry

end dark_more_than_light_l167_167334


namespace exists_infinitely_many_composite_l167_167053

noncomputable def tau (a : ℕ) : ℕ := sorry -- Placeholder definition for tau

def f (n : ℕ) : ℕ := tau n! - tau (n-1)!

theorem exists_infinitely_many_composite (n : ℕ) : ∃ n : ℕ, Nat.Prime n ∧ 
  (∀ m < n, f(m) < f(n)) := sorry

end exists_infinitely_many_composite_l167_167053


namespace batsman_average_after_17th_inning_l167_167031

-- Definitions for the conditions
def runs_scored_in_17th_inning : ℝ := 95
def increase_in_average : ℝ := 2.5

-- Lean statement encapsulating the problem
theorem batsman_average_after_17th_inning (A : ℝ) (h : 16 * A + runs_scored_in_17th_inning = 17 * (A + increase_in_average)) :
  A + increase_in_average = 55 := 
sorry

end batsman_average_after_17th_inning_l167_167031


namespace new_sphere_radius_l167_167690

noncomputable def calculateVolume (R r : ℝ) : ℝ :=
  let originalSphereVolume := (4 / 3) * Real.pi * R^3
  let cylinderHeight := 2 * Real.sqrt (R^2 - r^2)
  let cylinderVolume := Real.pi * r^2 * cylinderHeight
  let capHeight := R - Real.sqrt (R^2 - r^2)
  let capVolume := (Real.pi * capHeight^2 * (3 * R - capHeight)) / 3
  let totalCapVolume := 2 * capVolume
  originalSphereVolume - cylinderVolume - totalCapVolume

theorem new_sphere_radius
  (R : ℝ) (r : ℝ) (h : ℝ) (new_sphere_radius : ℝ)
  (h_eq: h = 2 * Real.sqrt (R^2 - r^2))
  (new_sphere_volume_eq: calculateVolume R r = (4 / 3) * Real.pi * new_sphere_radius^3)
  : new_sphere_radius = 16 :=
sorry

end new_sphere_radius_l167_167690


namespace probability_heads_in_9_of_12_flips_l167_167500

theorem probability_heads_in_9_of_12_flips :
  let total_flips := 12 in
  let required_heads := 9 in
  let total_outcomes := 2^total_flips in
  let specific_outcomes := Nat.choose total_flips required_heads in
  (specific_outcomes : ℚ) / total_outcomes = 220 / 4096 := 
by 
  sorry

end probability_heads_in_9_of_12_flips_l167_167500


namespace benny_eggs_l167_167342

def dozen := 12

def total_eggs (n: Nat) := n * dozen

theorem benny_eggs:
  total_eggs 7 = 84 := 
by 
  sorry

end benny_eggs_l167_167342


namespace triangle_area_formed_by_lines_l167_167221

def line1 := { p : ℝ × ℝ | p.2 = p.1 - 4 }
def line2 := { p : ℝ × ℝ | p.2 = -p.1 - 4 }
def x_axis := { p : ℝ × ℝ | p.2 = 0 }

def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_formed_by_lines :
  ∃ (A B C : ℝ × ℝ), A ∈ line1 ∧ A ∈ line2 ∧ B ∈ line1 ∧ B ∈ x_axis ∧ C ∈ line2 ∧ C ∈ x_axis ∧ 
  triangle_area A B C = 8 :=
by
  sorry

end triangle_area_formed_by_lines_l167_167221


namespace log_sum_geometric_sequence_l167_167576

open Real

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ (n : ℕ), a n ≠ 0 ∧ a (n + 1) / a n = a 1 / a 0

theorem log_sum_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_pos : ∀ n, 0 < a n) 
  (h_geo : geometric_sequence a) 
  (h_eq : a 10 * a 11 + a 9 * a 12 = 2 * exp 5) : 
  log (a 1) + log (a 2) + log (a 3) + log (a 4) + log (a 5) + 
  log (a 6) + log (a 7) + log (a 8) + log (a 9) + log (a 10) + 
  log (a 11) + log (a 12) + log (a 13) + log (a 14) + log (a 15) + 
  log (a 16) + log (a 17) + log (a 18) + log (a 19) + log (a 20) = 50 :=
sorry

end log_sum_geometric_sequence_l167_167576


namespace blue_pairs_count_l167_167339

-- Define the problem and conditions
def faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def sum9_pairs : Finset (ℕ × ℕ) := { (1, 8), (2, 7), (3, 6), (4, 5), (8, 1), (7, 2), (6, 3), (5, 4) }

-- Definition for counting valid pairs excluding pairs summing to 9
noncomputable def count_valid_pairs : ℕ := 
  (faces.card * (faces.card - 2)) / 2

-- Theorem statement proving the number of valid pairs
theorem blue_pairs_count : count_valid_pairs = 24 := 
by
  sorry

end blue_pairs_count_l167_167339


namespace square_side_length_l167_167798

theorem square_side_length (d : ℝ) (sqrt_2_ne_zero : sqrt 2 ≠ 0) (h : d = 2 * sqrt 2) : 
  ∃ (s : ℝ), s = 2 ∧ d = s * sqrt 2 :=
by
  use 2
  split
  · rfl
  · rw [mul_comm, ←mul_assoc, eq_comm, mul_right_comm, mul_div_cancel, h, mul_comm]
    · exact sqrt_2_ne_zero
  sorry

end square_side_length_l167_167798


namespace largest_inscribed_square_length_l167_167124

noncomputable def inscribed_square_length (s : ℝ) (n : ℕ) : ℝ :=
  let t := s / n
  let h := (Real.sqrt 3 / 2) * t
  s - 2 * h

theorem largest_inscribed_square_length :
  inscribed_square_length 12 3 = 12 - 4 * Real.sqrt 3 :=
by
  sorry

end largest_inscribed_square_length_l167_167124


namespace homothety_transformation_l167_167582

variable {V : Type}
variable [AddCommGroup V] [Module ℝ V]

/-- Definition of a homothety transformation -/
def homothety (S A A' : V) (k : ℝ) : Prop :=
  A' = k • A + (1 - k) • S

theorem homothety_transformation (S A A' : V) (k : ℝ) :
  homothety S A A' k ↔ A' = k • A + (1 - k) • S := 
by
  sorry

end homothety_transformation_l167_167582


namespace arithmetic_sequence_diff_l167_167426

theorem arithmetic_sequence_diff (b : ℕ → ℚ) (h1 : ∀ n : ℕ, b (n + 1) - b n = b 1 - b 0)
  (h2 : (Finset.range 150).sum b = 150)
  (h3 : (Finset.Ico 150 300).sum b = 300) : b 2 - b 1 = 1 / 150 :=
by
  sorry

end arithmetic_sequence_diff_l167_167426


namespace percent_absent_of_students_l167_167143

theorem percent_absent_of_students
  (boys girls : ℕ)
  (total_students := boys + girls)
  (boys_absent_fraction girls_absent_fraction : ℚ)
  (boys_absent_fraction_eq : boys_absent_fraction = 1 / 8)
  (girls_absent_fraction_eq : girls_absent_fraction = 1 / 4)
  (total_students_eq : total_students = 160)
  (boys_eq : boys = 80)
  (girls_eq : girls = 80) :
  (boys_absent_fraction * boys + girls_absent_fraction * girls) / total_students * 100 = 18.75 :=
by
  sorry

end percent_absent_of_students_l167_167143


namespace avg_marks_l167_167826

theorem avg_marks (P C M : ℕ) (h : P + C + M = P + 150) : (C + M) / 2 = 75 :=
by
  -- Proof goes here
  sorry

end avg_marks_l167_167826


namespace employee_pays_216_l167_167687

def retail_price (wholesale_cost : ℝ) (markup_percentage : ℝ) : ℝ :=
    wholesale_cost + markup_percentage * wholesale_cost

def employee_payment (retail_price : ℝ) (discount_percentage : ℝ) : ℝ :=
    retail_price - discount_percentage * retail_price

theorem employee_pays_216 (wholesale_cost : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) :
    wholesale_cost = 200 ∧ markup_percentage = 0.20 ∧ discount_percentage = 0.10 →
    employee_payment (retail_price wholesale_cost markup_percentage) discount_percentage = 216 :=
by
  intro h
  rcases h with ⟨h_wholesale, h_markup, h_discount⟩
  rw [h_wholesale, h_markup, h_discount]
  -- Now we have to prove the final statement: employee_payment (retail_price 200 0.20) 0.10 = 216
  -- This follows directly by computation, so we leave it as a sorry for now
  sorry

end employee_pays_216_l167_167687


namespace fill_cistern_time_l167_167204

-- Define the rates of the taps
def rateA := (1 : ℚ) / 3  -- Tap A fills 1 cistern in 3 hours (rate is 1/3 per hour)
def rateB := -(1 : ℚ) / 6  -- Tap B empties 1 cistern in 6 hours (rate is -1/6 per hour)
def rateC := (1 : ℚ) / 2  -- Tap C fills 1 cistern in 2 hours (rate is 1/2 per hour)

-- Define the combined rate
def combinedRate := rateA + rateB + rateC

-- The time to fill the cistern when all taps are opened simultaneously
def timeToFill := 1 / combinedRate

-- The theorem stating that the time to fill the cistern is 1.5 hours
theorem fill_cistern_time : timeToFill = (3 : ℚ) / 2 := by
  sorry  -- The proof is omitted as per the instructions

end fill_cistern_time_l167_167204


namespace hyperbola_asymptote_perpendicular_to_line_l167_167398

variable {a : ℝ}

theorem hyperbola_asymptote_perpendicular_to_line (h : a > 0)
  (C : ∀ x y : ℝ, x^2 / a^2 - y^2 = 1)
  (l : ∀ x y : ℝ, 2 * x - y + 1 = 0) :
  a = 2 :=
by
  sorry

end hyperbola_asymptote_perpendicular_to_line_l167_167398


namespace bernold_wins_game_l167_167674

/-- A game is played on a 2007 x 2007 grid. Arnold's move consists of taking a 2 x 2 square,
 and Bernold's move consists of taking a 1 x 1 square. They alternate turns with Arnold starting.
  When Arnold can no longer move, Bernold takes all remaining squares. The goal is to prove that 
  Bernold can always win the game by ensuring that Arnold cannot make enough moves to win. --/
theorem bernold_wins_game (N : ℕ) (hN : N = 2007) :
  let admissible_points := (N - 1) * (N - 1)
  let arnold_moves_needed := (N / 2) * (N / 2 + 1) / 2 + 1
  admissible_points < arnold_moves_needed :=
by
  let admissible_points := 2006 * 2006
  let arnold_moves_needed := 1003 * 1004 / 2 + 1
  exact sorry

end bernold_wins_game_l167_167674


namespace bus_stoppage_time_per_hour_l167_167198

theorem bus_stoppage_time_per_hour
  (speed_excluding_stoppages : ℕ) 
  (speed_including_stoppages : ℕ)
  (h1 : speed_excluding_stoppages = 54) 
  (h2 : speed_including_stoppages = 45) 
  : (60 * (speed_excluding_stoppages - speed_including_stoppages) / speed_excluding_stoppages) = 10 :=
by sorry

end bus_stoppage_time_per_hour_l167_167198


namespace contribution_proof_l167_167201

theorem contribution_proof (total : ℕ) (a_months b_months : ℕ) (a_total b_total a_received b_received : ℕ) :
  total = 3400 →
  a_months = 12 →
  b_months = 16 →
  a_received = 2070 →
  b_received = 1920 →
  (∃ (a_contributed b_contributed : ℕ), a_contributed = 1800 ∧ b_contributed = 1600) :=
by
  sorry

end contribution_proof_l167_167201


namespace range_of_inverse_dist_sum_l167_167120

theorem range_of_inverse_dist_sum 
  (t α : ℝ) 
  (P Q A : ℝ × ℝ)
  (C1 : ℝ × ℝ → Prop := λ point, ∃ (θ : ℝ), point = ⟨2 * Real.cos θ * Real.cos θ, 2 * Real.cos θ * Real.sin θ⟩)
  (C2 : ℝ × ℝ → Prop := λ point, ∃ (t : ℝ), point = ⟨t * Real.cos α, 1 + t * Real.sin α⟩)
  (A_def : A = (0, 1))
  (intersections : C1 P ∧ C2 P ∧ C1 Q ∧ C2 Q) :
  2 < 1 / (Real.dist P A) + 1 / (Real.dist Q A) ∧ 
  1 / (Real.dist P A) + 1 / (Real.dist Q A) ≤ 2 * Real.sqrt 2 :=
sorry

end range_of_inverse_dist_sum_l167_167120


namespace swan_percentage_not_ducks_l167_167402

theorem swan_percentage_not_ducks (total_birds geese swans herons ducks : ℝ)
  (h_total : total_birds = 100)
  (h_geese : geese = 0.30 * total_birds)
  (h_swans : swans = 0.20 * total_birds)
  (h_herons : herons = 0.20 * total_birds)
  (h_ducks : ducks = 0.30 * total_birds) :
  (swans / (total_birds - ducks) * 100) = 28.57 :=
by
  sorry

end swan_percentage_not_ducks_l167_167402


namespace alex_points_l167_167610

variable {x y : ℕ} -- x is the number of three-point shots, y is the number of two-point shots
variable (success_rate_3 success_rate_2 : ℚ) -- success rates for three-point and two-point shots
variable (total_shots : ℕ) -- total number of shots

def alex_total_points (x y : ℕ) (success_rate_3 success_rate_2 : ℚ) : ℚ :=
  3 * success_rate_3 * x + 2 * success_rate_2 * y

axiom condition_1 : success_rate_3 = 0.25
axiom condition_2 : success_rate_2 = 0.20
axiom condition_3 : total_shots = 40
axiom condition_4 : x + y = total_shots

theorem alex_points : alex_total_points x y 0.25 0.20 = 30 :=
by
  -- The proof would go here
  sorry

end alex_points_l167_167610


namespace tv_horizontal_length_l167_167784

-- Conditions
def is_rectangular_tv (width height : ℝ) : Prop :=
width / height = 9 / 12

def diagonal_is (d : ℝ) : Prop :=
d = 32

-- Theorem to prove
theorem tv_horizontal_length (width height diagonal : ℝ) 
(h1 : is_rectangular_tv width height) 
(h2 : diagonal_is diagonal) : 
width = 25.6 := by 
sorry

end tv_horizontal_length_l167_167784


namespace necessary_but_not_sufficient_l167_167323

variable (a : ℝ)

theorem necessary_but_not_sufficient (h : a ≥ 2) : (a = 2 ∨ a > 2) ∧ ¬(a > 2 → a ≥ 2) := by
  sorry

end necessary_but_not_sufficient_l167_167323


namespace final_score_l167_167351

theorem final_score (questions_first_half questions_second_half : Nat)
  (points_correct points_incorrect : Int)
  (correct_first_half incorrect_first_half correct_second_half incorrect_second_half : Nat) :
  questions_first_half = 10 →
  questions_second_half = 15 →
  points_correct = 3 →
  points_incorrect = -1 →
  correct_first_half = 6 →
  incorrect_first_half = 4 →
  correct_second_half = 10 →
  incorrect_second_half = 5 →
  (points_correct * correct_first_half + points_incorrect * incorrect_first_half 
   + points_correct * correct_second_half + points_incorrect * incorrect_second_half) = 39 := 
by
  intros
  sorry

end final_score_l167_167351


namespace perfect_squares_m_l167_167388

theorem perfect_squares_m (m : ℕ) (hm_pos : m > 0) (hm_min4_square : ∃ a : ℕ, m - 4 = a^2) (hm_plus5_square : ∃ b : ℕ, m + 5 = b^2) : m = 20 ∨ m = 4 :=
by
  sorry

end perfect_squares_m_l167_167388


namespace height_on_fifth_bounce_l167_167660

-- Define initial conditions
def initial_height : ℝ := 96
def initial_efficiency : ℝ := 0.5
def efficiency_decrease : ℝ := 0.05
def air_resistance_loss : ℝ := 0.02

-- Recursive function to compute the height after each bounce
def bounce_height (height : ℝ) (efficiency : ℝ) : ℝ :=
  let height_after_bounce := height * efficiency
  height_after_bounce - (height_after_bounce * air_resistance_loss)

-- Function to compute the bounce efficiency after each bounce
def bounce_efficiency (initial_efficiency : ℝ) (n : ℕ) : ℝ :=
  initial_efficiency - n * efficiency_decrease

-- Function to calculate the height after n-th bounce
def height_after_n_bounces (n : ℕ) : ℝ :=
  match n with
  | 0     => initial_height
  | n + 1 => bounce_height (height_after_n_bounces n) (bounce_efficiency initial_efficiency n)

-- Lean statement to prove the problem
theorem height_on_fifth_bounce :
  height_after_n_bounces 5 = 0.82003694685696 := by
  sorry

end height_on_fifth_bounce_l167_167660


namespace line_segment_value_of_x_l167_167321

theorem line_segment_value_of_x (x : ℝ) (h1 : (1 - 4)^2 + (3 - x)^2 = 25) (h2 : x > 0) : x = 7 :=
sorry

end line_segment_value_of_x_l167_167321


namespace arithmetic_sequence_sum_equals_product_l167_167356

theorem arithmetic_sequence_sum_equals_product :
  ∃ (a_1 a_2 a_3 : ℤ), (a_2 = a_1 + d) ∧ (a_3 = a_1 + 2 * d) ∧ 
    a_1 ≠ 0 ∧ (a_1 + a_2 + a_3 = a_1 * a_2 * a_3) ∧ 
    (∃ d x : ℤ, x ≠ 0 ∧ d ≠ 0 ∧ 
    ((x = 1 ∧ d = 1) ∨ (x = -3 ∧ d = 1) ∨ (x = 3 ∧ d = -1) ∨ (x = -1 ∧ d = -1))) :=
sorry

end arithmetic_sequence_sum_equals_product_l167_167356


namespace probability_of_9_heads_in_12_l167_167475

def coin_flip_probability_9_heads_in_12_flips : Prop :=
  let total_outcomes := 2^12
  let success_outcomes := Nat.choose 12 9
  success_outcomes / total_outcomes = 220 / 4096

theorem probability_of_9_heads_in_12 : coin_flip_probability_9_heads_in_12_flips :=
  sorry

end probability_of_9_heads_in_12_l167_167475


namespace find_d_div_a_l167_167291
noncomputable def quad_to_square_form (x : ℝ) : ℝ :=
  x^2 + 1500 * x + 1800

theorem find_d_div_a : 
  ∃ (a d : ℝ), (∀ x : ℝ, quad_to_square_form x = (x + a)^2 + d) 
  ∧ a = 750 
  ∧ d = -560700 
  ∧ d / a = -560700 / 750 := 
sorry

end find_d_div_a_l167_167291


namespace trig_identity_l167_167240

theorem trig_identity (α : ℝ) (h : Real.sin (α - Real.pi / 3) = 2 / 3) : 
  Real.cos (2 * α + Real.pi / 3) = -1 / 9 :=
by
  sorry

end trig_identity_l167_167240


namespace add_second_largest_to_sum_l167_167970

def is_valid_digit (d : ℕ) : Prop := d = 2 ∨ d = 5 ∨ d = 8

def form_number (d1 d2 d3 : ℕ) : ℕ := 100 * d1 + 10 * d2 + d3

def largest_number : ℕ := form_number 8 5 2
def smallest_number : ℕ := form_number 2 5 8
def second_largest_number : ℕ := form_number 8 2 5

theorem add_second_largest_to_sum : 
  second_largest_number + (largest_number + smallest_number) = 1935 := 
  sorry

end add_second_largest_to_sum_l167_167970


namespace min_area_PRQSQ_l167_167573

-- Define the setup
variables {m : ℝ} (h : m > 0)

-- Line passing through A(1,1) with slope -m
def line := {p : ℝ × ℝ // ∃ x y, p = (1 + x/m, 1 - m*x)}

-- Intersection points P and Q
def P := (1 + 1/m, 0)
def Q := (0, 1 + m)

-- Perpendicular feet R and S from P and Q on 2x + y = 0
def line₂ := {p : ℝ × ℝ // p.1 + 2*p.2 = 0}
def R := orthogonal_projection line₂ (1 + 1/m, 0)
def S := orthogonal_projection line₂ (0, 1 + m)

-- Lengths
def PR := distance P R
def QS := distance Q S
def RS := distance R S

-- Minimum area of quadrilateral PRSQ
theorem min_area_PRQSQ : 
  let area := (1 / 5) * (m + (1 / m) + (9 / 4)) ^ 2 - (1 / 80) in
  ∃ (area_min : ℝ), area ≥ 3.6 :=
sorry

end min_area_PRQSQ_l167_167573


namespace least_number_to_add_1054_23_l167_167178

def least_number_to_add (n k : ℕ) : ℕ :=
  let remainder := n % k
  if remainder = 0 then 0 else k - remainder

theorem least_number_to_add_1054_23 : least_number_to_add 1054 23 = 4 :=
by
  -- This is a placeholder for the actual proof
  sorry

end least_number_to_add_1054_23_l167_167178


namespace average_of_consecutive_sequences_l167_167235

theorem average_of_consecutive_sequences (a b : ℕ) (h : b = (a + (a+1) + (a+2) + (a+3) + (a+4)) / 5) : 
    ((b + (b+1) + (b+2) + (b+3) + (b+4)) / 5) = a + 4 :=
by
  sorry

end average_of_consecutive_sequences_l167_167235


namespace donny_spent_on_thursday_l167_167719

theorem donny_spent_on_thursday :
  let savings_monday : ℤ := 15,
      savings_tuesday : ℤ := 28,
      savings_wednesday : ℤ := 13,
      total_savings : ℤ := savings_monday + savings_tuesday + savings_wednesday,
      amount_spent_thursday : ℤ := total_savings / 2
  in
  amount_spent_thursday = 28 :=
by
  sorry

end donny_spent_on_thursday_l167_167719


namespace room_height_l167_167856

-- Define the conditions
def total_curtain_length : ℕ := 101
def extra_material : ℕ := 5

-- Define the statement to be proven
theorem room_height : total_curtain_length - extra_material = 96 :=
by
  sorry

end room_height_l167_167856


namespace min_period_k_l167_167634

def has_period {α : Type*} [HasZero α] (r : α) (n : ℕ) : Prop :=
  -- A function definition to check if 'r' has a repeating decimal period of length 'n'
  sorry

theorem min_period_k (a b : ℚ) (h₁ : has_period a 30) (h₂ : has_period b 30) (h₃ : has_period (a - b) 15) :
  ∃ (k : ℕ), k = 6 ∧ has_period (a + k * b) 15 :=
begin
  sorry
end

end min_period_k_l167_167634


namespace max_area_of_fenced_rectangle_l167_167070

theorem max_area_of_fenced_rectangle (x y : ℝ) (h : x + y = 30) : x * y ≤ 225 :=
by {
  sorry
}

end max_area_of_fenced_rectangle_l167_167070


namespace total_crackers_l167_167226

-- Define the conditions
def boxes_Darren := 4
def crackers_per_box := 24
def boxes_Calvin := 2 * boxes_Darren - 1

-- Define the mathematical proof problem
theorem total_crackers : 
  let total_Darren := boxes_Darren * crackers_per_box
  let total_Calvin := boxes_Calvin * crackers_per_box
  total_Darren + total_Calvin = 264 :=
by
  sorry

end total_crackers_l167_167226


namespace young_fish_per_pregnant_fish_l167_167137

-- Definitions based on conditions
def tanks := 3
def fish_per_tank := 4
def total_young_fish := 240

-- Calculations based on conditions
def total_pregnant_fish := tanks * fish_per_tank

-- The proof statement
theorem young_fish_per_pregnant_fish : total_young_fish / total_pregnant_fish = 20 := by
  sorry

end young_fish_per_pregnant_fish_l167_167137


namespace problem_condition_holds_l167_167672

theorem problem_condition_holds (x y : ℝ) (h₁ : x + 0.35 * y - (x + y) = 200) : y = -307.69 :=
sorry

end problem_condition_holds_l167_167672


namespace cards_per_pack_l167_167638

-- Definitions from the problem conditions
def packs := 60
def cards_per_page := 10
def pages_needed := 42

-- Theorem statement for the mathematically equivalent proof problem
theorem cards_per_pack : (pages_needed * cards_per_page) / packs = 7 :=
by sorry

end cards_per_pack_l167_167638


namespace find_f_2015_l167_167239

variables (f : ℝ → ℝ)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 6) = f x + f 3

theorem find_f_2015 (h1 : is_even_function f) (h2 : satisfies_condition f) (h3 : f 1 = 2) : f 2015 = 2 :=
by
  sorry

end find_f_2015_l167_167239


namespace probability_heads_exactly_9_of_12_l167_167466

noncomputable def bin_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_heads_9_of_12_flips : ℚ :=
  (bin_coeff 12 9 : ℚ) / (2 ^ 12)

theorem probability_heads_exactly_9_of_12 :
  probability_heads_9_of_12_flips = 220 / 4096 :=
by
  sorry

end probability_heads_exactly_9_of_12_l167_167466


namespace projection_matrix_correct_l167_167130

noncomputable def Q_matrix : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![1/9, 5/9, 8/9], 
    ![7/9, 8/9, 2/9], 
    ![4/9, 2/9, 5/9]]

theorem projection_matrix_correct : 
  ∀ u : Fin 3 → ℝ,
  let n := ![2, 1, -2] in
  let Q : Matrix (Fin 3) (Fin 3) ℝ := Q_matrix in
  (Q.mulVec u) = u - ((((u ⬝ n) / (n ⬝ n)) • n)) :=
sorry

end projection_matrix_correct_l167_167130


namespace product_closest_to_106_l167_167163

theorem product_closest_to_106 :
  let product := (2.1 : ℝ) * (50.8 - 0.45)
  abs (product - 106) < abs (product - 105) ∧
  abs (product - 106) < abs (product - 107) ∧
  abs (product - 106) < abs (product - 108) ∧
  abs (product - 106) < abs (product - 110) :=
by
  sorry

end product_closest_to_106_l167_167163


namespace arithmetic_sequence_common_difference_l167_167916

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) 
    (h1 : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 81)
    (h2 : a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 171) : 
    ∃ d, d = 10 := 
by 
  sorry

end arithmetic_sequence_common_difference_l167_167916


namespace smallest_k_for_min_period_15_l167_167632

/-- Rational number with minimal period -/
def is_minimal_period (r : ℚ) (n : ℕ) : Prop :=
  ∃ m : ℤ, r = m / (10^n - 1)

variables (a b : ℚ)

-- Conditions for a and b
axiom ha : is_minimal_period a 30
axiom hb : is_minimal_period b 30

-- Condition for a - b
axiom hab_min_period : is_minimal_period (a - b) 15

-- Conclusion
theorem smallest_k_for_min_period_15 : ∃ k : ℕ, k = 6 ∧ is_minimal_period (a + k * b) 15 :=
by sorry

end smallest_k_for_min_period_15_l167_167632


namespace std_dev_of_normal_distribution_l167_167285

theorem std_dev_of_normal_distribution (μ σ : ℝ) (h1: μ = 14.5) (h2: μ - 2 * σ = 11.5) : σ = 1.5 := 
by 
  sorry

end std_dev_of_normal_distribution_l167_167285


namespace probability_heads_exactly_9_of_12_l167_167463

noncomputable def bin_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_heads_9_of_12_flips : ℚ :=
  (bin_coeff 12 9 : ℚ) / (2 ^ 12)

theorem probability_heads_exactly_9_of_12 :
  probability_heads_9_of_12_flips = 220 / 4096 :=
by
  sorry

end probability_heads_exactly_9_of_12_l167_167463


namespace volume_of_prism_l167_167284

theorem volume_of_prism
  (a b c : ℝ)
  (h1 : a * b = 30)
  (h2 : a * c = 40)
  (h3 : b * c = 60) :
  a * b * c = 120 * Real.sqrt 5 := 
by
  sorry

end volume_of_prism_l167_167284


namespace number_of_green_eyes_l167_167006

-- Definitions based on conditions
def total_people : Nat := 100
def blue_eyes : Nat := 19
def brown_eyes : Nat := total_people / 2
def black_eyes : Nat := total_people / 4

-- Theorem stating the main question and its answer
theorem number_of_green_eyes : 
  (total_people - (blue_eyes + brown_eyes + black_eyes)) = 6 := by
  sorry

end number_of_green_eyes_l167_167006


namespace find_unit_prices_l167_167725

variable (x : ℝ)

def typeB_unit_price (priceB : ℝ) : Prop :=
  priceB = 15

def typeA_unit_price (priceA : ℝ) : Prop :=
  priceA = 40

def budget_condition : Prop :=
  900 / x = 3 * (800 / (x + 25))

theorem find_unit_prices (h : budget_condition x) :
  typeB_unit_price x ∧ typeA_unit_price (x + 25) :=
sorry

end find_unit_prices_l167_167725


namespace jade_living_expenses_l167_167919

-- Definitions from the conditions
variable (income : ℝ) (insurance_fraction : ℝ) (savings : ℝ) (P : ℝ)

-- Constants from the given problem
noncomputable def jadeIncome : income = 1600 := by sorry
noncomputable def jadeInsuranceFraction : insurance_fraction = 1 / 5 := by sorry
noncomputable def jadeSavings : savings = 80 := by sorry

-- The proof problem statement
theorem jade_living_expenses :
    (P * 1600 + (1 / 5) * 1600 + 80 = 1600) → P = 3 / 4 := by
    intros h
    sorry

end jade_living_expenses_l167_167919


namespace train_length_l167_167667

theorem train_length (L : ℕ) :
  (L + 350) / 15 = (L + 500) / 20 → L = 100 := 
by
  intro h
  sorry

end train_length_l167_167667


namespace infinite_pairs_exists_l167_167935

noncomputable def exists_infinite_pairs : Prop :=
  ∃ (a b : ℕ), (a + b ∣ a * b + 1) ∧ (a - b ∣ a * b - 1) ∧ b > 1 ∧ a > b * Real.sqrt 3 - 1

theorem infinite_pairs_exists : ∃ (count : ℕ) (a b : ℕ), ∀ n < count, exists_infinite_pairs :=
sorry

end infinite_pairs_exists_l167_167935


namespace triangles_satisfying_equation_l167_167522

theorem triangles_satisfying_equation (a b c : ℝ) (h₂ : a + b > c) (h₃ : a + c > b) (h₄ : b + c > a) :
  (c ^ 2 - a ^ 2) / b + (b ^ 2 - c ^ 2) / a = b - a →
  (a = b ∨ c ^ 2 = a ^ 2 + b ^ 2) := 
sorry

end triangles_satisfying_equation_l167_167522


namespace determine_m_l167_167732

noncomputable def function_f (m : ℝ) (x : ℝ) : ℝ := m * x - |x + 1|

def exists_constant_interval (a b c m : ℝ) : Prop :=
  a < b ∧ ∀ x, a ≤ x ∧ x ≤ b → function_f m x = c

theorem determine_m (m : ℝ) (a b c : ℝ) :
  (a < b ∧ a ≥ -2 ∧ b ≥ -2 ∧ (∀ x, a ≤ x ∧ x ≤ b → function_f m x = c)) →
  m = 1 ∨ m = -1 :=
sorry

end determine_m_l167_167732


namespace starting_time_calculation_l167_167435

def glowing_light_start_time (max_glows : ℚ) (interval : ℚ) (end_time : Time) : Time :=
  let total_glows := max_glows.floor
  let total_seconds := total_glows * interval
  let total_hours := (total_seconds / 3600).toInt
  let remaining_seconds := total_seconds % 3600
  let total_minutes := (remaining_seconds / 60).toInt
  let remaining_final_seconds := remaining_seconds % 60
  let end_hours := end_time.hour - total_hours
  let end_minutes := (end_time.min - total_minutes + 60) % 60
  let end_seconds := end_time.sec - remaining_final_seconds
  Time.mk end_hours end_minutes end_seconds

theorem starting_time_calculation :
  (glowing_light_start_time 236.61904761904762 21 (Time.mk 3 20 47)) = Time.mk 1 58 11 :=
by
  sorry

end starting_time_calculation_l167_167435


namespace probability_heads_in_9_of_12_flips_l167_167499

theorem probability_heads_in_9_of_12_flips :
  let total_flips := 12 in
  let required_heads := 9 in
  let total_outcomes := 2^total_flips in
  let specific_outcomes := Nat.choose total_flips required_heads in
  (specific_outcomes : ℚ) / total_outcomes = 220 / 4096 := 
by 
  sorry

end probability_heads_in_9_of_12_flips_l167_167499


namespace students_attend_Purum_Elementary_School_l167_167295
open Nat

theorem students_attend_Purum_Elementary_School (P N : ℕ) 
  (h1 : P + N = 41) (h2 : P = N + 3) : P = 22 :=
sorry

end students_attend_Purum_Elementary_School_l167_167295


namespace grocery_store_spending_l167_167129

/-- Lenny has $84 initially. He spent $24 on video games and has $39 left.
We need to prove that he spent $21 at the grocery store. --/
theorem grocery_store_spending (initial_amount spent_on_video_games amount_left after_games_left : ℕ) 
    (h1 : initial_amount = 84)
    (h2 : spent_on_video_games = 24)
    (h3 : amount_left = 39)
    (h4 : after_games_left = initial_amount - spent_on_video_games) 
    : after_games_left - amount_left = 21 := 
sorry

end grocery_store_spending_l167_167129


namespace fair_die_proba_l167_167958
noncomputable def probability_of_six : ℚ := 1 / 6

theorem fair_die_proba : 
  (1 / 6 : ℚ) = probability_of_six :=
by
  sorry

end fair_die_proba_l167_167958


namespace woman_waits_for_man_l167_167994

noncomputable def man_speed := 5 / 60 -- miles per minute
noncomputable def woman_speed := 15 / 60 -- miles per minute
noncomputable def passed_time := 2 -- minutes

noncomputable def catch_up_time (man_speed woman_speed : ℝ) (passed_time : ℝ) : ℝ :=
  (woman_speed * passed_time) / man_speed

theorem woman_waits_for_man
  (man_speed woman_speed : ℝ)
  (passed_time : ℝ)
  (h_man_speed : man_speed = 5 / 60)
  (h_woman_speed : woman_speed = 15 / 60)
  (h_passed_time : passed_time = 2) :
  catch_up_time man_speed woman_speed passed_time = 6 := 
by
  -- actual proof skipped
  sorry

end woman_waits_for_man_l167_167994


namespace angle_of_rotation_l167_167700

-- Definitions for the given conditions
def radius_large := 9 -- cm
def radius_medium := 3 -- cm
def radius_small := 1 -- cm
def speed := 1 -- cm/s

-- Definition of the angles calculations
noncomputable def rotations_per_revolution (R1 R2 : ℝ) : ℝ := R1 / R2
noncomputable def total_rotations (R1 R2 R3 : ℝ) : ℝ := 
  let rotations_medium := rotations_per_revolution R1 R2
  let net_rotations_medium := rotations_medium - 1
  net_rotations_medium * rotations_per_revolution R2 R3 + 1

-- Assertion to prove
theorem angle_of_rotation : 
  total_rotations radius_large radius_medium radius_small * 360 = 2520 :=
by 
  simp [total_rotations, rotations_per_revolution]
  exact sorry -- proof placeholder

end angle_of_rotation_l167_167700


namespace min_value_expression_l167_167620

theorem min_value_expression (a b t : ℝ) (h : a + b = t) : 
  ∃ c : ℝ, c = ((a^2 + 1)^2 + (b^2 + 1)^2) → c = (t^4 + 8 * t^2 + 16) / 8 :=
by
  sorry

end min_value_expression_l167_167620


namespace A_squared_plus_B_squared_eq_one_l167_167413

theorem A_squared_plus_B_squared_eq_one
  (A B : ℝ) (h1 : A ≠ B)
  (h2 : ∀ x : ℝ, (A * (B * x ^ 2 + A) ^ 2 + B - (B * (A * x ^ 2 + B) ^ 2 + A)) = B ^ 2 - A ^ 2) :
  A ^ 2 + B ^ 2 = 1 :=
sorry

end A_squared_plus_B_squared_eq_one_l167_167413


namespace cos_product_triangle_l167_167616

theorem cos_product_triangle (A B C : ℝ) (h : A + B + C = π) (hA : A > 0) (hB : B > 0) (hC : C > 0) : 
  Real.cos A * Real.cos B * Real.cos C ≤ 1 / 8 := 
sorry

end cos_product_triangle_l167_167616


namespace power_minus_self_even_l167_167787

theorem power_minus_self_even (a n : ℕ) (ha : 0 < a) (hn : 0 < n) : Even (a^n - a) := by
  sorry

end power_minus_self_even_l167_167787


namespace linda_total_profit_is_50_l167_167926

def total_loaves : ℕ := 60
def loaves_sold_morning (total_loaves : ℕ) : ℕ := total_loaves / 3
def loaves_sold_afternoon (loaves_left_morning : ℕ) : ℕ := loaves_left_morning / 2
def loaves_sold_evening (loaves_left_afternoon : ℕ) : ℕ := loaves_left_afternoon

def price_per_loaf_morning : ℕ := 3
def price_per_loaf_afternoon : ℕ := 150 / 100 -- Representing $1.50 as 150 cents to use integer arithmetic
def price_per_loaf_evening : ℕ := 1

def cost_per_loaf : ℕ := 1

def calculate_profit (total_loaves loaves_sold_morning loaves_sold_afternoon loaves_sold_evening price_per_loaf_morning price_per_loaf_afternoon price_per_loaf_evening cost_per_loaf : ℕ) : ℕ := 
  let revenue_morning := loaves_sold_morning * price_per_loaf_morning
  let loaves_left_morning := total_loaves - loaves_sold_morning
  let revenue_afternoon := loaves_sold_afternoon * price_per_loaf_afternoon
  let loaves_left_afternoon := loaves_left_morning - loaves_sold_afternoon
  let revenue_evening := loaves_sold_evening * price_per_loaf_evening
  let total_revenue := revenue_morning + revenue_afternoon + revenue_evening
  let total_cost := total_loaves * cost_per_loaf
  total_revenue - total_cost

theorem linda_total_profit_is_50 : calculate_profit total_loaves (loaves_sold_morning total_loaves) (loaves_sold_afternoon (total_loaves - loaves_sold_morning total_loaves)) (total_loaves - loaves_sold_morning total_loaves - loaves_sold_afternoon (total_loaves - loaves_sold_morning total_loaves)) price_per_loaf_morning price_per_loaf_afternoon price_per_loaf_evening cost_per_loaf = 50 := 
  by 
    sorry

end linda_total_profit_is_50_l167_167926


namespace max_area_of_rectangular_pen_l167_167076

-- Define the perimeter and derive the formula for the area
def perimeter := 60
def half_perimeter := perimeter / 2
def area (x : ℝ) := x * (half_perimeter - x)

-- Statement of the problem: prove the maximum area is 225 square feet
theorem max_area_of_rectangular_pen : ∃ x : ℝ, 0 ≤ x ∧ x ≤ half_perimeter ∧ area x = 225 := 
sorry

end max_area_of_rectangular_pen_l167_167076


namespace ratio_first_to_second_l167_167166

theorem ratio_first_to_second (S F T : ℕ) 
  (hS : S = 60)
  (hT : T = F / 3)
  (hSum : F + S + T = 220) :
  F / S = 2 :=
by
  sorry

end ratio_first_to_second_l167_167166


namespace probability_heads_9_of_12_l167_167509

theorem probability_heads_9_of_12 :
  let total_outcomes := 2^12 in
  let favorable_outcomes := Nat.choose 12 9 in
  let probability := favorable_outcomes / total_outcomes.toRat in
  probability = 220 / 4096 := 
by
  sorry

end probability_heads_9_of_12_l167_167509


namespace int_solutions_eq_count_int_values_b_l167_167055

theorem int_solutions_eq (b : ℤ) : 
  ∃! x : ℤ, ∃! y : ℤ, (x + y = -b) ∧ (x * y = 12 * b) \/
  (x + y = -b) ∧ (x * y = 12 * b) :=
begin
  -- Assume roots p, q exist
  -- Use Vieta's formulas: p + q = -b, pq = 12b
  -- Transform the equation using SFFT
  sorry
end

theorem count_int_values_b :
  set_finite {b : ℤ | ∃! x : ℤ, ∃! y : ℤ, 
    (x + y = -b) ∧ (x * y = 12 * b ) \/ 
    (x + y = -b) ∧ (x * y = 12 * b)} ∧
  fintype.card {b : ℤ | ∃! x : ℤ, ∃! y : ℤ,
    (x + y = -b) ∧ (x * y = 12 * b ) \/ 
    (x + y = -b) ∧ (x * y = 12 * b)} = 16 :=
begin
  sorry
end

end int_solutions_eq_count_int_values_b_l167_167055


namespace total_length_of_fence_l167_167197

theorem total_length_of_fence
  (x : ℝ)
  (h1 : (2 : ℝ) * x ^ 2 = 200) :
  (2 * x + 2 * x) = 40 :=
by
sorry

end total_length_of_fence_l167_167197


namespace variance_of_sample_l167_167105

theorem variance_of_sample
  (x : ℝ)
  (h : (2 + 3 + x + 6 + 8) / 5 = 5) : 
  (1 / 5) * ((2 - 5) ^ 2 + (3 - 5) ^ 2 + (x - 5) ^ 2 + (6 - 5) ^ 2 + (8 - 5) ^ 2) = 24 / 5 :=
by
  sorry

end variance_of_sample_l167_167105


namespace probability_of_9_heads_in_12_flips_l167_167505

theorem probability_of_9_heads_in_12_flips :
  (nat.choose 12 9) * (1 / (2 : ℝ)^12) = 55 / 1024 := 
by
  sorry

end probability_of_9_heads_in_12_flips_l167_167505


namespace smallest_common_multiple_of_10_11_18_l167_167731

theorem smallest_common_multiple_of_10_11_18 : 
  ∃ (n : ℕ), (n % 10 = 0) ∧ (n % 11 = 0) ∧ (n % 18 = 0) ∧ (n = 990) :=
by
  sorry

end smallest_common_multiple_of_10_11_18_l167_167731


namespace min_max_values_l167_167433

noncomputable def f (x : ℝ) : ℝ := 1 + 3 * x - x^3

theorem min_max_values : 
  (∃ x : ℝ, f x = -1) ∧ (∃ x : ℝ, f x = 3) :=
by
  sorry

end min_max_values_l167_167433


namespace inequality_solution_l167_167939

theorem inequality_solution (x : ℝ) :
  -2 < (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) ∧
  (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) < 2 ↔
  4 - Real.sqrt 21 < x ∧ x < 4 + Real.sqrt 21 :=
sorry

end inequality_solution_l167_167939


namespace car_speed_l167_167681

-- Define the given conditions
def distance := 800 -- in kilometers
def time := 5 -- in hours

-- Define the speed calculation
def speed (d : ℕ) (t : ℕ) : ℕ := d / t

-- State the theorem to be proved
theorem car_speed : speed distance time = 160 := by
  -- proof would go here
  sorry

end car_speed_l167_167681


namespace sum_of_fourth_powers_l167_167063

theorem sum_of_fourth_powers (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 4) : 
  a^4 + b^4 + c^4 = 8 := 
by 
  sorry

end sum_of_fourth_powers_l167_167063


namespace area_of_OPF_eq_sqrt_2_div_2_l167_167895

noncomputable def area_of_triangle_OPF : ℝ :=
  let O : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (1, 0)
  let P : ℝ × ℝ := (0.5, Real.sqrt 2) -- We assume P is (1/2, sqrt(2))
  let P1 : ℝ × ℝ := (0.5, -Real.sqrt 2) -- We also define the other point P1
  if (dist O P = dist P F) ∨ (dist O P1 = dist P1 F) then
    let base := dist O F
    let height := Real.sqrt 2
    (1 / 2) * base * height
  else
    0

theorem area_of_OPF_eq_sqrt_2_div_2 : 
  let O : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (1, 0)
  let P : ℝ × ℝ := (0.5, Real.sqrt 2) -- We assume P is (1/2, sqrt(2))
  let P1 : ℝ × ℝ := (0.5, -Real.sqrt 2) -- We also define the other point P1
  (dist O P = dist P F) ∨ (dist O P1 = dist P1 F) →
  let base := dist O F
  let height := Real.sqrt 2
  area_of_triangle_OPF = Real.sqrt 2 / 2 := 
by 
  sorry

end area_of_OPF_eq_sqrt_2_div_2_l167_167895


namespace max_rectangle_area_l167_167087

theorem max_rectangle_area (P : ℝ) (hP : P = 60) : ∃ A : ℝ, A = 225 :=
by 
  sorry

end max_rectangle_area_l167_167087


namespace difference_in_squares_l167_167320

noncomputable def radius_of_circle (x y h R : ℝ) : Prop :=
  5 * x^2 - 4 * x * h + h^2 = R^2 ∧ 5 * y^2 + 4 * y * h + h^2 = R^2

theorem difference_in_squares (x y h R : ℝ) (h_radius : radius_of_circle x y h R) :
  2 * x - 2 * y = (8/5 : ℝ) * h :=
by
  sorry

end difference_in_squares_l167_167320


namespace valid_fractions_l167_167230

theorem valid_fractions :
  ∃ (x y z : ℕ), (1 ≤ x ∧ x ≤ 9) ∧ (1 ≤ y ∧ y ≤ 9) ∧ (1 ≤ z ∧ z ≤ 9) ∧
  (10 * x + y) % (10 * y + z) = 0 ∧ (10 * x + y) / (10 * y + z) = x / z :=
sorry

end valid_fractions_l167_167230


namespace find_analytical_expression_of_f_l167_167250

-- Define the function f and the condition it needs to satisfy
variable (f : ℝ → ℝ)
variable (hf : ∀ x : ℝ, x ≠ 0 → f (1 / x) = 1 / (x + 1))

-- State the objective to prove
theorem find_analytical_expression_of_f : 
  ∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 → f x = x / (1 + x) := by
  sorry

end find_analytical_expression_of_f_l167_167250


namespace calculate_expression_l167_167220

theorem calculate_expression : (8^5 / 8^2) * 3^6 = 373248 := by
  sorry

end calculate_expression_l167_167220


namespace makes_at_least_one_shot_l167_167019
noncomputable section

/-- The probability of making the free throw. -/
def free_throw_make_prob : ℚ := 4/5

/-- The probability of making the high school 3-pointer. -/
def high_school_make_prob : ℚ := 1/2

/-- The probability of making the professional 3-pointer. -/
def pro_make_prob : ℚ := 1/3

/-- The probability of making at least one of the three shots. -/
theorem makes_at_least_one_shot :
  (1 - ((1 - free_throw_make_prob) * (1 - high_school_make_prob) * (1 - pro_make_prob))) = 14 / 15 :=
by
  sorry

end makes_at_least_one_shot_l167_167019


namespace purchasing_plans_count_l167_167338

theorem purchasing_plans_count :
  (∃ (x y : ℕ), 15 * x + 20 * y = 360) ∧ ∀ (x y : ℕ), 15 * x + 20 * y = 360 → (x % 4 = 0) ∧ (y = 18 - (3 / 4) * x) := sorry

end purchasing_plans_count_l167_167338


namespace max_rectangle_area_l167_167084

theorem max_rectangle_area (P : ℝ) (hP : P = 60) : ∃ A : ℝ, A = 225 :=
by 
  sorry

end max_rectangle_area_l167_167084


namespace number_of_strings_is_multiple_of_3_l167_167286

theorem number_of_strings_is_multiple_of_3 (N : ℕ) :
  (∀ (avg_total avg_one_third avg_two_third : ℚ), 
    avg_total = 80 ∧ avg_one_third = 70 ∧ avg_two_third = 85 →
    (∃ k : ℕ, N = 3 * k)) :=
by
  intros avg_total avg_one_third avg_two_third h
  sorry

end number_of_strings_is_multiple_of_3_l167_167286


namespace sin_neg_45_l167_167709

theorem sin_neg_45 :
  Real.sin (-45 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
sorry

end sin_neg_45_l167_167709


namespace a_2017_value_l167_167952

theorem a_2017_value (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n : ℕ, S (n + 1) = 2 * (n + 1) - 1) :
  a 2017 = 2 :=
by
  sorry

end a_2017_value_l167_167952


namespace ratio_of_dolls_l167_167231

-- Definitions used in Lean 4 statement directly appear in the conditions
variable (I : ℕ) -- the number of dolls Ivy has
variable (Dina_dolls : ℕ := 60) -- Dina has 60 dolls
variable (Ivy_collectors : ℕ := 20) -- Ivy has 20 collector edition dolls

-- Condition based on given problem
axiom Ivy_collectors_condition : (2 / 3 : ℚ) * I = 20

-- Lean 4 statement for the proof problem
theorem ratio_of_dolls (h : 3 * Ivy_collectors = 2 * I) : Dina_dolls / I = 2 := by
  sorry

end ratio_of_dolls_l167_167231


namespace isosceles_triangle_angles_l167_167162

theorem isosceles_triangle_angles (α β γ : ℝ) 
  (h1 : α = 50)
  (h2 : α + β + γ = 180)
  (isosceles : (α = β ∨ α = γ ∨ β = γ)) :
  (β = 50 ∧ γ = 80) ∨ (γ = 50 ∧ β = 80) :=
by
  sorry

end isosceles_triangle_angles_l167_167162


namespace find_m_range_l167_167896

variable {x y m : ℝ}

theorem find_m_range (h1 : x + 2 * y = m + 4) (h2 : 2 * x + y = 2 * m - 1)
    (h3 : x + y < 2) (h4 : x - y < 4) : m < 1 := by
  sorry

end find_m_range_l167_167896


namespace boat_travel_distance_downstream_l167_167838

-- Definitions of the given conditions
def boatSpeedStillWater : ℕ := 10 -- km/hr
def streamSpeed : ℕ := 8 -- km/hr
def timeDownstream : ℕ := 3 -- hours

-- Effective speed downstream
def effectiveSpeedDownstream : ℕ := boatSpeedStillWater + streamSpeed

-- Goal: Distance traveled downstream equals 54 km
theorem boat_travel_distance_downstream :
  effectiveSpeedDownstream * timeDownstream = 54 := 
by
  -- Since only the statement is needed, we use sorry to indicate the proof is skipped
  sorry

end boat_travel_distance_downstream_l167_167838


namespace bus_sarah_probability_l167_167680

-- Define the probability of Sarah arriving while the bus is still there
theorem bus_sarah_probability :
  let total_minutes := 60
  let bus_waiting_time := 15
  let total_area := (total_minutes * total_minutes : ℕ)
  let triangle_area := (1 / 2 : ℝ) * 45 * 15
  let rectangle_area := 15 * 15
  let shaded_area := triangle_area + rectangle_area
  (shaded_area / total_area : ℝ) = (5 / 32 : ℝ) :=
by
  sorry

end bus_sarah_probability_l167_167680


namespace four_digit_numbers_with_5_or_7_l167_167587

theorem four_digit_numbers_with_5_or_7 : 
  let total_four_digit_numbers := 9000
  let no_5_or_7_first_digit := 7
  let no_5_or_7_other_digits := 8
  let four_digit_numbers_no_5_or_7 := no_5_or_7_first_digit * no_5_or_7_other_digits * no_5_or_7_other_digits * no_5_or_7_other_digits
  total_four_digit_numbers - four_digit_numbers_no_5_or_7 = 5416 :=
by
  let total_four_digit_numbers := 9000
  let no_5_or_7_first_digit := 7
  let no_5_or_7_other_digits := 8
  let four_digit_numbers_no_5_or_7 := no_5_or_7_first_digit * no_5_or_7_other_digits * no_5_or_7_other_digits * no_5_or_7_other_digits
  show total_four_digit_numbers - four_digit_numbers_no_5_or_7 = 5416 from sorry

end four_digit_numbers_with_5_or_7_l167_167587


namespace find_missing_number_l167_167261

def average (l : List ℕ) : ℚ := l.sum / l.length

theorem find_missing_number : 
  ∃ x : ℕ, 
    average [744, 745, 747, 748, 749, 752, 752, 753, 755, x] = 750 :=
sorry

end find_missing_number_l167_167261


namespace number_of_yellow_parrots_l167_167931

theorem number_of_yellow_parrots (total_parrots : ℕ) (red_fraction : ℚ) 
  (h_total_parrots : total_parrots = 108) 
  (h_red_fraction : red_fraction = 5 / 6) : 
  ∃ (yellow_parrots : ℕ), yellow_parrots = total_parrots * (1 - red_fraction) ∧ yellow_parrots = 18 := 
by
  sorry

end number_of_yellow_parrots_l167_167931


namespace ming_dynasty_wine_problem_l167_167914

theorem ming_dynasty_wine_problem :
  ∃ x y : ℝ, x + y = 19 ∧ 3 * x + (1 / 3) * y = 33 :=
by {
  -- Define the existence of variables x and y satisfying the conditions
  existsi (x : ℝ),
  existsi (y : ℝ),
  -- Conditions are given as premises to be satisfied
  split,
  -- First equation: x + y = 19
  exact x + y = 19,
  -- Second equation: 3x + (1/3)y = 33
  exact 3 * x + (1 / 3) * y = 33,
  -- Add placeholder to indicate where the actual proof would go
  sorry
}

end ming_dynasty_wine_problem_l167_167914


namespace sophie_total_spend_l167_167794

-- Definitions based on conditions
def cost_cupcakes : ℕ := 5 * 2
def cost_doughnuts : ℕ := 6 * 1
def cost_apple_pie : ℕ := 4 * 2
def cost_cookies : ℕ := 15 * 6 / 10 -- since 0.60 = 6/10

-- Total cost
def total_cost : ℕ := cost_cupcakes + cost_doughnuts + cost_apple_pie + cost_cookies

-- Prove the total cost
theorem sophie_total_spend : total_cost = 33 := by
  sorry

end sophie_total_spend_l167_167794


namespace trig_identity_l167_167530

theorem trig_identity :
  2 * Real.sin (Real.pi / 6) - Real.cos (Real.pi / 4)^2 + Real.cos (Real.pi / 3) = 1 :=
by
  sorry

end trig_identity_l167_167530


namespace gcd_50403_40302_l167_167348

theorem gcd_50403_40302 : Nat.gcd 50403 40302 = 1 :=
by
  sorry

end gcd_50403_40302_l167_167348


namespace number_of_distinct_paths_l167_167121

theorem number_of_distinct_paths (segments steps_per_segment : ℕ) (h_segments : segments = 15) (h_steps : steps_per_segment = 6) :
    ∑ i in Finset.range (segments + 1), Nat.fib (steps_per_segment + 1) = 195 :=
by
  have h : segments * Nat.fib (steps_per_segment + 1) = 15 * Nat.fib 7 := by
    rw [h_segments, h_steps]

  have fib_value : Nat.fib 7 = 13 := by
    rw [Nat.fib_succ_succ, Nat.fib_succ, Nat.fib_succ, Nat.fib_succ, Nat.fib_succ, Nat.fib_succ, Nat.fib_one, Nat.fib_zero]
    norm_num

  rw [fib_value] at h
  exact h
  sorry

end number_of_distinct_paths_l167_167121


namespace asymptotes_and_eccentricity_of_hyperbola_l167_167159

noncomputable def hyperbola_asymptotes_and_eccentricity : Prop :=
  let a := 1
  let b := Real.sqrt 2
  let c := Real.sqrt 3
  ∀ (x y : ℝ), x^2 - (y^2 / 2) = 1 →
    ((y = 2 * x ∨ y = -2 * x) ∧ Real.sqrt (1 + (b^2 / a^2)) = c)

theorem asymptotes_and_eccentricity_of_hyperbola :
  hyperbola_asymptotes_and_eccentricity :=
by
  sorry

end asymptotes_and_eccentricity_of_hyperbola_l167_167159


namespace square_projection_exists_l167_167296

structure Point :=
(x y : Real)

structure Line :=
(a b c : Real) -- Line equation ax + by + c = 0

def is_on_line (P : Point) (l : Line) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

theorem square_projection_exists (P : Point) (l : Line) :
  ∃ (A B C D : Point), 
  is_on_line A l ∧ 
  is_on_line B l ∧
  (A.x + B.x) / 2 = P.x ∧ 
  (A.y + B.y) / 2 = P.y ∧ 
  (A.x = B.x ∨ A.y = B.y) ∧ -- assuming one of the sides lies along the line
  (C.x + D.x) / 2 = P.x ∧ 
  (C.y + D.y) / 2 = P.y ∧ 
  C ≠ A ∧ C ≠ B ∧ D ≠ A ∧ D ≠ B :=
sorry

end square_projection_exists_l167_167296


namespace abc_le_one_eighth_l167_167408

theorem abc_le_one_eighth (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a / (1 + a) + b / (1 + b) + c / (1 + c) = 1) : 
  a * b * c ≤ 1 / 8 := 
by
  sorry

end abc_le_one_eighth_l167_167408


namespace max_area_of_rectangular_pen_l167_167083

theorem max_area_of_rectangular_pen (P : ℕ) (hP : P = 60) : 
  ∃ A : ℕ, A = 225 ∧ (∀ x y : ℕ, 2 * (x + y) = P → x * y ≤ A) := 
by
  sorry

end max_area_of_rectangular_pen_l167_167083


namespace tan_pink_violet_probability_l167_167836

noncomputable def probability_tan_pink_violet_consecutive_order : ℚ :=
  let num_ways := (Nat.factorial 4) * (Nat.factorial 3) * (Nat.factorial 5)
  let total_ways := Nat.factorial 12
  num_ways / total_ways

theorem tan_pink_violet_probability :
  probability_tan_pink_violet_consecutive_order = 1 / 27720 := by
  sorry

end tan_pink_violet_probability_l167_167836


namespace yoongi_more_points_l167_167183

def yoongiPoints : ℕ := 4
def jungkookPoints : ℕ := 6 - 3

theorem yoongi_more_points : yoongiPoints > jungkookPoints := by
  sorry

end yoongi_more_points_l167_167183


namespace probability_exactly_9_heads_in_12_flips_l167_167458

noncomputable theory

open BigOperators

-- Define a function for combinations
def C(n k : ℕ) : ℕ := if h : k ≤ n then (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k))) else 0

-- The main Lean 4 statement
theorem probability_exactly_9_heads_in_12_flips :
  let total_outcomes := 2 ^ 12 in
  let favorable_outcomes := C 12 9 in
  favorable_outcomes.to_float / total_outcomes.to_float = 220 / 4096 :=
by
  let total_outcomes := 2 ^ 12
  let favorable_outcomes := C 12 9
  sorry

end probability_exactly_9_heads_in_12_flips_l167_167458


namespace fraction_of_paint_used_l167_167126

theorem fraction_of_paint_used 
  (total_paint : ℕ)
  (paint_used_first_week : ℚ)
  (total_paint_used : ℕ)
  (paint_fraction_first_week : ℚ)
  (remaining_paint : ℚ)
  (paint_used_second_week : ℚ)
  (paint_fraction_second_week : ℚ)
  (h1 : total_paint = 360)
  (h2 : paint_fraction_first_week = 2/3)
  (h3 : paint_used_first_week = paint_fraction_first_week * total_paint)
  (h4 : remaining_paint = total_paint - paint_used_first_week)
  (h5 : remaining_paint = 120)
  (h6 : total_paint_used = 264)
  (h7 : paint_used_second_week = total_paint_used - paint_used_first_week)
  (h8 : paint_fraction_second_week = paint_used_second_week / remaining_paint):
  paint_fraction_second_week = 1/5 := 
by 
  sorry

end fraction_of_paint_used_l167_167126


namespace new_students_count_l167_167945

theorem new_students_count (x : ℕ) (avg_age_group new_avg_age avg_new_students : ℕ)
  (h1 : avg_age_group = 14) (h2 : new_avg_age = 15) (h3 : avg_new_students = 17)
  (initial_students : ℕ) (initial_avg_age : ℕ)
  (h4 : initial_students = 10) (h5 : initial_avg_age = initial_students * avg_age_group)
  (h6 : new_avg_age * (initial_students + x) = initial_avg_age + (x * avg_new_students)) :
  x = 5 := 
by
  sorry

end new_students_count_l167_167945


namespace probability_of_9_heads_in_12_l167_167477

def coin_flip_probability_9_heads_in_12_flips : Prop :=
  let total_outcomes := 2^12
  let success_outcomes := Nat.choose 12 9
  success_outcomes / total_outcomes = 220 / 4096

theorem probability_of_9_heads_in_12 : coin_flip_probability_9_heads_in_12_flips :=
  sorry

end probability_of_9_heads_in_12_l167_167477


namespace num_four_digit_with_5_or_7_l167_167592

def num_four_digit_without_5_or_7 : Nat := 7 * 8 * 8 * 8

def num_four_digit_total : Nat := 9999 - 1000 + 1

theorem num_four_digit_with_5_or_7 : 
  num_four_digit_total - num_four_digit_without_5_or_7 = 5416 :=
by
  sorry

end num_four_digit_with_5_or_7_l167_167592


namespace max_area_of_fenced_rectangle_l167_167067

theorem max_area_of_fenced_rectangle (x y : ℝ) (h : x + y = 30) : x * y ≤ 225 :=
by {
  sorry
}

end max_area_of_fenced_rectangle_l167_167067


namespace units_digit_lucas_L10_is_4_l167_167155

def lucas : ℕ → ℕ 
  | 0 => 2
  | 1 => 1
  | n + 2 => lucas (n + 1) + lucas n

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_lucas_L10_is_4 : units_digit (lucas (lucas 10)) = 4 := 
  sorry

end units_digit_lucas_L10_is_4_l167_167155


namespace food_drive_ratio_l167_167540

/-- Mark brings in 4 times as many cans as Jaydon,
Jaydon brings in 5 more cans than a certain multiple of the amount of cans that Rachel brought in,
There are 135 cans total, and Mark brought in 100 cans.
Prove that the ratio of the number of cans Jaydon brought in to the number of cans Rachel brought in is 5:2. -/
theorem food_drive_ratio (J R : ℕ) (k : ℕ)
  (h1 : 4 * J = 100)
  (h2 : J = k * R + 5)
  (h3 : 100 + J + R = 135) :
  J / Nat.gcd J R = 5 ∧ R / Nat.gcd J R = 2 := by
  sorry

end food_drive_ratio_l167_167540


namespace circus_accommodation_l167_167298

theorem circus_accommodation : 246 * 4 = 984 := by
  sorry

end circus_accommodation_l167_167298


namespace dog_count_l167_167910

theorem dog_count 
  (long_furred : ℕ) 
  (brown : ℕ) 
  (neither : ℕ) 
  (long_furred_brown : ℕ) 
  (total : ℕ) 
  (h1 : long_furred = 29) 
  (h2 : brown = 17) 
  (h3 : neither = 8) 
  (h4 : long_furred_brown = 9)
  (h5 : total = long_furred + brown - long_furred_brown + neither) : 
  total = 45 :=
by 
  sorry

end dog_count_l167_167910


namespace largest_tile_side_length_l167_167789

theorem largest_tile_side_length (w h : ℕ) (hw : w = 17) (hh : h = 23) : Nat.gcd w h = 1 := by
  -- Proof goes here
  sorry

end largest_tile_side_length_l167_167789


namespace sum_pattern_l167_167151

theorem sum_pattern (a b : ℕ) : (6 + 7 = 13) ∧ (8 + 9 = 17) ∧ (5 + 6 = 11) ∧ (7 + 8 = 15) ∧ (3 + 3 = 6) → (6 + 7 = 12) :=
by
  sorry

end sum_pattern_l167_167151


namespace average_selections_per_car_l167_167328

-- Definitions based on conditions
def num_cars : ℕ := 12
def num_clients : ℕ := 9
def selections_per_client : ℕ := 4

-- Theorem to prove
theorem average_selections_per_car :
  (num_clients * selections_per_client) / num_cars = 3 :=
by
  -- Placeholder for the proof
  sorry

end average_selections_per_car_l167_167328


namespace abc_le_one_eighth_l167_167405

theorem abc_le_one_eighth (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h : a / (1 + a) + b / (1 + b) + c / (1 + c) = 1) : a * b * c ≤ 1 / 8 :=
by
  sorry

end abc_le_one_eighth_l167_167405


namespace range_of_a_l167_167740

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 + (a - 1) * x + 1 / 4 > 0) ↔ (Real.sqrt 5 - 3) / 2 < a ∧ a < (3 + Real.sqrt 5) / 2 :=
by
  sorry

end range_of_a_l167_167740


namespace original_paint_intensity_l167_167986

theorem original_paint_intensity 
  (P : ℝ)
  (H1 : 0 ≤ P ∧ P ≤ 100)
  (H2 : ∀ (unit : ℝ), unit = 100)
  (H3 : ∀ (replaced_fraction : ℝ), replaced_fraction = 1.5)
  (H4 : ∀ (new_intensity : ℝ), new_intensity = 30)
  (H5 : ∀ (solution_intensity : ℝ), solution_intensity = 0.25) :
  P = 15 := 
by
  sorry

end original_paint_intensity_l167_167986


namespace probability_heads_in_9_of_12_flips_l167_167501

theorem probability_heads_in_9_of_12_flips :
  let total_flips := 12 in
  let required_heads := 9 in
  let total_outcomes := 2^total_flips in
  let specific_outcomes := Nat.choose total_flips required_heads in
  (specific_outcomes : ℚ) / total_outcomes = 220 / 4096 := 
by 
  sorry

end probability_heads_in_9_of_12_flips_l167_167501


namespace value_of_y_l167_167602

theorem value_of_y (y: ℚ) (h: (2 / 5 - 1 / 7) = 14 / y): y = 490 / 9 :=
by
  sorry

end value_of_y_l167_167602


namespace bus_stops_for_28_minutes_per_hour_l167_167728

-- Definitions based on the conditions
def without_stoppages_speed : ℕ := 75
def with_stoppages_speed : ℕ := 40
def speed_difference : ℕ := without_stoppages_speed - with_stoppages_speed

-- Theorem statement
theorem bus_stops_for_28_minutes_per_hour : 
  ∀ (T : ℕ), (T = (speed_difference*60)/(without_stoppages_speed))  → 
  T = 28 := 
by
  sorry

end bus_stops_for_28_minutes_per_hour_l167_167728


namespace probability_heads_in_nine_of_twelve_flips_l167_167481

theorem probability_heads_in_nine_of_twelve_flips : 
  (nat.choose 12 9) / (2^12) = 55 / 1024 := 
by
  sorry

end probability_heads_in_nine_of_twelve_flips_l167_167481


namespace expression_of_y_l167_167238

theorem expression_of_y (x y : ℝ) (h : x - y / 2 = 1) : y = 2 * x - 2 :=
sorry

end expression_of_y_l167_167238


namespace vertex_of_parabola_l167_167577

theorem vertex_of_parabola (a : ℝ) :
  (∃ (k : ℝ), ∀ x : ℝ, y = -4*x - 1 → x = 2 ∧ (a - 4) = -4 * 2 - 1) → 
  (2, -9) = (2, a - 4) → a = -5 :=
by
  sorry

end vertex_of_parabola_l167_167577


namespace martha_total_cost_l167_167661

def weight_cheese : ℝ := 1.5
def weight_meat : ℝ := 0.55    -- converting grams to kg
def weight_pasta : ℝ := 0.28   -- converting grams to kg
def weight_tomatoes : ℝ := 2.2

def price_cheese_per_kg : ℝ := 6.30
def price_meat_per_kg : ℝ := 8.55
def price_pasta_per_kg : ℝ := 2.40
def price_tomatoes_per_kg : ℝ := 1.79

def tax_cheese : ℝ := 0.07
def tax_meat : ℝ := 0.06
def tax_pasta : ℝ := 0.08
def tax_tomatoes : ℝ := 0.05

def total_cost : ℝ :=
  let cost_cheese := weight_cheese * price_cheese_per_kg * (1 + tax_cheese)
  let cost_meat := weight_meat * price_meat_per_kg * (1 + tax_meat)
  let cost_pasta := weight_pasta * price_pasta_per_kg * (1 + tax_pasta)
  let cost_tomatoes := weight_tomatoes * price_tomatoes_per_kg * (1 + tax_tomatoes)
  cost_cheese + cost_meat + cost_pasta + cost_tomatoes

theorem martha_total_cost : total_cost = 19.9568 := by
  sorry

end martha_total_cost_l167_167661


namespace exponent_sum_l167_167755

variables (a : ℝ) (m n : ℝ)

theorem exponent_sum (h1 : a^m = 3) (h2 : a^n = 2) : a^(m + n) = 6 :=
by
  sorry

end exponent_sum_l167_167755


namespace sum_of_transformed_parabolas_is_non_horizontal_line_l167_167325

theorem sum_of_transformed_parabolas_is_non_horizontal_line
    (a b c : ℝ)
    (f g : ℝ → ℝ)
    (hf : ∀ x, f x = a * (x - 8)^2 + b * (x - 8) + c)
    (hg : ∀ x, g x = -a * (x + 8)^2 - b * (x + 8) - (c - 3)) :
    ∃ m q : ℝ, ∀ x : ℝ, (f x + g x) = m * x + q ∧ m ≠ 0 :=
by sorry

end sum_of_transformed_parabolas_is_non_horizontal_line_l167_167325


namespace difference_q_r_share_l167_167668

theorem difference_q_r_share (x : ℝ) (h1 : 7 * x - 3 * x = 2800) :
  12 * x - 7 * x = 3500 :=
by
  sorry

end difference_q_r_share_l167_167668


namespace max_area_of_rectangular_pen_l167_167074

-- Define the perimeter and derive the formula for the area
def perimeter := 60
def half_perimeter := perimeter / 2
def area (x : ℝ) := x * (half_perimeter - x)

-- Statement of the problem: prove the maximum area is 225 square feet
theorem max_area_of_rectangular_pen : ∃ x : ℝ, 0 ≤ x ∧ x ≤ half_perimeter ∧ area x = 225 := 
sorry

end max_area_of_rectangular_pen_l167_167074


namespace negation_of_universal_proposition_l167_167885

variable {R : Type*} [LinearOrderedField R]
variable (f : R → R)

theorem negation_of_universal_proposition :
  (∀ x1 x2 : R, (f x2 - f x1) * (x2 - x1) ≥ 0) →
  ∃ x1 x2 : R, (f x2 - f x1) * (x2 - x1) < 0 :=
sorry

end negation_of_universal_proposition_l167_167885


namespace initial_percentage_of_alcohol_l167_167834

theorem initial_percentage_of_alcohol :
  ∃ P : ℝ, (P / 100 * 11) = (33 / 100 * 14) :=
by
  use 42
  sorry

end initial_percentage_of_alcohol_l167_167834


namespace probability_different_cars_l167_167292

theorem probability_different_cars : 
  let cars := {A, B, C} in
  let choices := cars × cars in
  let different_choices := { (a, b) | (a, b) ∈ choices ∧ a ≠ b } in
  (Finset.card different_choices : ℚ) / (Finset.card choices : ℚ) = 2 / 3 :=
by
  sorry

end probability_different_cars_l167_167292


namespace probability_heads_9_of_12_is_correct_l167_167456

-- Defining the probability of getting exactly 9 heads in 12 flips of a fair coin
noncomputable def probability_heads_9_of_12 := (55 : ℚ) / 1024

-- Theorem statement
theorem probability_heads_9_of_12_is_correct :
  probability_heads_9_of_12 = (55 : ℚ) / 1024 := 
  by sorry

end probability_heads_9_of_12_is_correct_l167_167456


namespace line_through_point_and_parallel_l167_167045

def point_A : ℝ × ℝ × ℝ := (-2, 3, 1)

def plane1 (x y z : ℝ) := x - 2*y - z - 2 = 0
def plane2 (x y z : ℝ) := 2*x + 3*y - z + 1 = 0

theorem line_through_point_and_parallel (x y z t : ℝ) :
  ∃ t, 
    x = 5 * t - 2 ∧
    y = -t + 3 ∧
    z = 7 * t + 1 :=
sorry

end line_through_point_and_parallel_l167_167045


namespace AC_eq_200_l167_167336

theorem AC_eq_200 (A B C : ℕ) (h1 : A + B + C = 500) (h2 : B + C = 330) (h3 : C = 30) : A + C = 200 := by
  sorry

end AC_eq_200_l167_167336


namespace find_x_l167_167033

theorem find_x (x : ℝ) (h : 15 * x + 16 * x + 19 * x + 11 = 161) : x = 3 :=
sorry

end find_x_l167_167033


namespace neg_p_range_of_x_neg_q_sufficient_not_necessary_for_neg_p_l167_167886

def p (x : ℝ) : Prop := (x^2 - x - 2) ≤ 0
def q (x m : ℝ) : Prop := (x^2 - x - m^2 - m) ≤ 0

theorem neg_p_range_of_x (x : ℝ) : ¬ p x → x > 2 ∨ x < -1 :=
by
-- proof steps here
sorry

theorem neg_q_sufficient_not_necessary_for_neg_p (m : ℝ) : 
  (∀ x, ¬ q x m → ¬ p x) ∧ (∃ x, p x → ¬ q x m) → m > 1 ∨ m < -2 :=
by
-- proof steps here
sorry

end neg_p_range_of_x_neg_q_sufficient_not_necessary_for_neg_p_l167_167886


namespace johns_raise_percentage_increase_l167_167670

def initial_earnings : ℚ := 65
def new_earnings : ℚ := 70
def percentage_increase (initial new : ℚ) : ℚ := ((new - initial) / initial) * 100

theorem johns_raise_percentage_increase : percentage_increase initial_earnings new_earnings = 7.692307692 :=
by
  sorry

end johns_raise_percentage_increase_l167_167670


namespace quadratic_root_is_zero_then_m_neg_one_l167_167246

theorem quadratic_root_is_zero_then_m_neg_one (m : ℝ) (h_eq : (m-1) * 0^2 + 2 * 0 + m^2 - 1 = 0) : m = -1 := by
  sorry

end quadratic_root_is_zero_then_m_neg_one_l167_167246


namespace mila_total_distance_l167_167279

/-- Mila's car consumes a gallon of gas every 40 miles, her full gas tank holds 16 gallons, starting with a full tank, she drove 400 miles, then refueled with 10 gallons, 
and upon arriving at her destination her gas tank was a third full.
Prove that the total distance Mila drove that day is 826 miles. -/
theorem mila_total_distance (consumption_per_mile : ℝ) (tank_capacity : ℝ) (initial_drive : ℝ) (refuel_amount : ℝ) (final_fraction : ℝ)
  (consumption_per_mile_def : consumption_per_mile = 1 / 40)
  (tank_capacity_def : tank_capacity = 16)
  (initial_drive_def : initial_drive = 400)
  (refuel_amount_def : refuel_amount = 10)
  (final_fraction_def : final_fraction = 1 / 3) :
  ∃ total_distance : ℝ, total_distance = 826 :=
by
  sorry

end mila_total_distance_l167_167279


namespace probability_nine_heads_l167_167472

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

def flips : ℕ := 12
def heads : ℕ := 9
def total_outcomes : ℕ := 2^flips
def favorable_outcomes : ℕ := binom flips heads
def probability : Rat := favorable_outcomes / total_outcomes.toRat

theorem probability_nine_heads :
  probability = (220/4096) := by
  sorry

end probability_nine_heads_l167_167472


namespace p_correct_l167_167359

noncomputable def p : ℝ → ℝ := sorry

axiom p_at_3 : p 3 = 10

axiom p_condition (x y : ℝ) : p x * p y = p x + p y + p (x * y) - 2

theorem p_correct : ∀ x, p x = x^2 + 1 :=
sorry

end p_correct_l167_167359


namespace pieces_per_box_l167_167703

theorem pieces_per_box (boxes : ℕ) (total_pieces : ℕ) (h_boxes : boxes = 7) (h_total : total_pieces = 21) : 
  total_pieces / boxes = 3 :=
by
  sorry

end pieces_per_box_l167_167703


namespace probability_nine_heads_l167_167471

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

def flips : ℕ := 12
def heads : ℕ := 9
def total_outcomes : ℕ := 2^flips
def favorable_outcomes : ℕ := binom flips heads
def probability : Rat := favorable_outcomes / total_outcomes.toRat

theorem probability_nine_heads :
  probability = (220/4096) := by
  sorry

end probability_nine_heads_l167_167471


namespace centroid_quad_area_correct_l167_167153

noncomputable def centroid_quadrilateral_area (E F G H Q : ℝ × ℝ) (side_length : ℝ) (EQ FQ : ℝ) : ℝ :=
  if h : side_length = 40 ∧ EQ = 15 ∧ FQ = 35 then
    12800 / 9
  else
    sorry

theorem centroid_quad_area_correct (E F G H Q : ℝ × ℝ) (side_length EQ FQ : ℝ) 
  (h : side_length = 40 ∧ EQ = 15 ∧ FQ = 35) :
  centroid_quadrilateral_area E F G H Q side_length EQ FQ = 12800 / 9 :=
sorry

end centroid_quad_area_correct_l167_167153


namespace find_f_three_l167_167734

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (a b : ℝ) : f (a + b) = f a + f b
axiom f_two : f 2 = 3

theorem find_f_three : f 3 = 9 / 2 :=
by
  sorry

end find_f_three_l167_167734


namespace solve_inequality_l167_167937

def polynomial_fraction (x : ℝ) : ℝ :=
  (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5)

theorem solve_inequality (x : ℝ) :
  -2 < polynomial_fraction x ∧ polynomial_fraction x < 2 ↔ 11.57 < x :=
sorry

end solve_inequality_l167_167937


namespace initial_mean_l167_167949

theorem initial_mean (M : ℝ) (h1 : 50 * (36.5 : ℝ) - 23 = 50 * (36.04 : ℝ) + 23)
: M = 36.04 :=
by
  sorry

end initial_mean_l167_167949


namespace line_parallel_not_passing_through_point_l167_167738

noncomputable def point_outside_line (A B C x0 y0 : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (A * x0 + B * y0 + C = k)

theorem line_parallel_not_passing_through_point 
  (A B C x0 y0 : ℝ) (h : point_outside_line A B C x0 y0) :
  ∃ k : ℝ, k ≠ 0 ∧ (∀ x y : ℝ, Ax + By + C + k = 0 → Ax_0 + By_0 + C + k ≠ 0) :=
sorry

end line_parallel_not_passing_through_point_l167_167738


namespace proposition_D_is_true_l167_167041

-- Define the propositions
def proposition_A : Prop := ∃ x₀ : ℝ, Real.exp x₀ ≤ 0
def proposition_B : Prop := ∀ x : ℝ, 2^x > x^2
def proposition_C : Prop := ∀ a b : ℝ, (a + b = 0 ↔ a / b = -1)
def proposition_D : Prop := ∀ a b : ℝ, (a > 1 ∧ b > 1) → a * b > 1

-- Problem statement: Proposition D is true
theorem proposition_D_is_true : proposition_D := 
by sorry

end proposition_D_is_true_l167_167041


namespace right_triangle_sides_l167_167650

theorem right_triangle_sides (a b c : ℝ) (h : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c = 60 → h = 12 → a^2 + b^2 = c^2 → a * b = 12 * c → 
  (a = 15 ∧ b = 20 ∧ c = 25) ∨ (a = 20 ∧ b = 15 ∧ c = 25) :=
by sorry

end right_triangle_sides_l167_167650


namespace table_tennis_possible_outcomes_l167_167304

-- Two people are playing a table tennis match. The first to win 3 games wins the match.
-- The match continues until a winner is determined.
-- Considering all possible outcomes (different numbers of wins and losses for each player are considered different outcomes),
-- prove that there are a total of 30 possible outcomes.

theorem table_tennis_possible_outcomes : 
  ∃ total_outcomes : ℕ, total_outcomes = 30 := 
by
  -- We need to prove that the total number of possible outcomes is 30
  sorry

end table_tennis_possible_outcomes_l167_167304


namespace probability_heads_9_of_12_is_correct_l167_167457

-- Defining the probability of getting exactly 9 heads in 12 flips of a fair coin
noncomputable def probability_heads_9_of_12 := (55 : ℚ) / 1024

-- Theorem statement
theorem probability_heads_9_of_12_is_correct :
  probability_heads_9_of_12 = (55 : ℚ) / 1024 := 
  by sorry

end probability_heads_9_of_12_is_correct_l167_167457


namespace certain_number_eq_1000_l167_167029

theorem certain_number_eq_1000 (x : ℝ) (h : 3500 - x / 20.50 = 3451.2195121951218) : x = 1000 := 
by
  sorry

end certain_number_eq_1000_l167_167029


namespace table_height_l167_167845

theorem table_height (l w h : ℝ) (h1 : l + h - w = 38) (h2 : w + h - l = 34) : h = 36 :=
by
  sorry

end table_height_l167_167845


namespace store_profit_is_33_percent_l167_167541

noncomputable def store_profit (C : ℝ) : ℝ :=
  let initial_markup := 1.20 * C
  let new_year_markup := initial_markup + 0.25 * initial_markup
  let february_discount := new_year_markup * 0.92
  let shipping_cost := C * 1.05
  (february_discount - shipping_cost)

theorem store_profit_is_33_percent (C : ℝ) : store_profit C = 0.33 * C :=
by
  sorry

end store_profit_is_33_percent_l167_167541


namespace days_vacuuming_l167_167551

theorem days_vacuuming (V : ℕ) (h1 : ∀ V, 130 = 30 * V + 40) : V = 3 :=
by
    have eq1 : 130 = 30 * V + 40 := h1 V
    sorry

end days_vacuuming_l167_167551


namespace total_books_proof_l167_167870

-- Define the number of books Lily finished last month.
def books_last_month : ℕ := 4

-- Define the number of books Lily wants to finish this month.
def books_this_month : ℕ := books_last_month * 2

-- Define the total number of books Lily will finish in two months.
def total_books_two_months : ℕ := books_last_month + books_this_month

-- Theorem to prove the total number of books Lily will finish in two months is 12.
theorem total_books_proof : total_books_two_months = 12 := by
  -- Here would be the proof steps.
  sorry

end total_books_proof_l167_167870


namespace smaller_base_length_trapezoid_l167_167161

variable (p q a b : ℝ)
variable (h : p < q)
variable (angle_ratio : ∃ α, ((2 * α) : ℝ) = α + (α : ℝ))

theorem smaller_base_length_trapezoid :
  b = (p^2 + a * p - q^2) / p :=
sorry

end smaller_base_length_trapezoid_l167_167161


namespace student_correct_answers_l167_167396

theorem student_correct_answers (C W : ℕ) 
  (h1 : 4 * C - W = 130) 
  (h2 : C + W = 80) : 
  C = 42 := by
  sorry

end student_correct_answers_l167_167396


namespace Mason_fathers_age_indeterminate_l167_167419

theorem Mason_fathers_age_indeterminate
  (Mason_age : ℕ) (Sydney_age Mason_father_age D : ℕ)
  (hM : Mason_age = 20)
  (hS_M : Mason_age = Sydney_age / 3)
  (hS_F : Mason_father_age - D = Sydney_age) :
  ¬ ∃ F, Mason_father_age = F :=
by {
  sorry
}

end Mason_fathers_age_indeterminate_l167_167419


namespace john_weekly_earnings_l167_167270

/-- John takes 3 days off of streaming per week. 
    John streams for 4 hours at a time on the days he does stream.
    John makes $10 an hour.
    Prove that John makes $160 a week. -/

theorem john_weekly_earnings (days_off : ℕ) (hours_per_day : ℕ) (wage_per_hour : ℕ) 
  (h_days_off : days_off = 3) (h_hours_per_day : hours_per_day = 4) 
  (h_wage_per_hour : wage_per_hour = 10) : 
  7 - days_off * hours_per_day * wage_per_hour = 160 := by
  sorry

end john_weekly_earnings_l167_167270


namespace batsman_sixes_l167_167032

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

end batsman_sixes_l167_167032


namespace negation_proof_converse_proof_l167_167802

-- Define the proposition
def prop_last_digit_zero_or_five (n : ℤ) : Prop := (n % 10 = 0) ∨ (n % 10 = 5)
def divisible_by_five (n : ℤ) : Prop := ∃ k : ℤ, n = 5 * k

-- Negation of the proposition
def negation_prop : Prop :=
  ∃ n : ℤ, prop_last_digit_zero_or_five n ∧ ¬ divisible_by_five n

-- Converse of the proposition
def converse_prop : Prop :=
  ∀ n : ℤ, ¬ prop_last_digit_zero_or_five n → ¬ divisible_by_five n

theorem negation_proof : negation_prop :=
  sorry  -- to be proved

theorem converse_proof : converse_prop :=
  sorry  -- to be proved

end negation_proof_converse_proof_l167_167802


namespace find_n_from_sequence_l167_167744

theorem find_n_from_sequence (a : ℕ → ℝ) (h₁ : ∀ n : ℕ, a n = (1 / (Real.sqrt n + Real.sqrt (n + 1))))
  (h₂ : ∃ n : ℕ, a n + a (n + 1) = Real.sqrt 11 - 3) : 9 ∈ {n | a n + a (n + 1) = Real.sqrt 11 - 3} :=
by
  sorry

end find_n_from_sequence_l167_167744


namespace percentage_increase_l167_167909

theorem percentage_increase (Z Y X : ℝ) (h1 : Y = 1.20 * Z) (h2 : Z = 250) (h3 : X + Y + Z = 925) :
  ((X - Y) / Y) * 100 = 25 :=
by
  sorry

end percentage_increase_l167_167909


namespace square_free_odd_integers_count_l167_167901

/-- Define the set of odd integers greater than 1 and less than 200 -/
def odd_integers := {n : ℕ | n > 1 ∧ n < 200 ∧ n % 2 = 1}

/-- Define a square-free predicate -/
def square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m * m ∣ n → false

/-- Define the set of square-free odd integers greater than 1 and less than 200 -/
def square_free_odd_integers := {n : ℕ | n ∈ odd_integers ∧ square_free n}

/-- The number of square-free odd integers between 1 and 200 is 79 -/
theorem square_free_odd_integers_count : 
  set.finite square_free_odd_integers ∧ set.card square_free_odd_integers = 79 :=
begin
  sorry
end

end square_free_odd_integers_count_l167_167901


namespace initial_oranges_in_bowl_l167_167534

theorem initial_oranges_in_bowl (A O : ℕ) (R : ℚ) (h1 : A = 14) (h2 : R = 0.7) 
    (h3 : R * (A + O - 15) = A) : O = 21 := 
by 
  sorry

end initial_oranges_in_bowl_l167_167534


namespace length_of_train_l167_167547

theorem length_of_train (speed_kmph : ℕ) (bridge_length_m : ℕ) (crossing_time_s : ℕ) 
  (h1 : speed_kmph = 45) (h2 : bridge_length_m = 220) (h3 : crossing_time_s = 30) :
  ∃ train_length_m : ℕ, train_length_m = 155 :=
by
  sorry

end length_of_train_l167_167547


namespace min_selections_l167_167751

theorem min_selections (p : ℝ) (n : ℕ) (P_B : ℝ) (h_p : p = 0.5) (h_P_B : P_B = 0.9) : 
  1 - p^n ≥ P_B ↔ n ≥ 4 :=
by
  rw [h_p, h_P_B]
  sorry

end min_selections_l167_167751


namespace octagon_mass_is_19kg_l167_167367

-- Define the parameters given in the problem
def side_length_square_sheet := 1  -- side length in meters
def thickness_sheet := 0.3  -- thickness in cm (3 mm)
def density_steel := 7.8  -- density in g/cm³

-- Given the geometric transformations and constants, prove the mass of the octagon
theorem octagon_mass_is_19kg :
  ∃ mass : ℝ, (mass = 19) :=
by
  -- Placeholder for the proof.
  -- The detailed steps would include geometrical transformations and volume calculations,
  -- which have been rigorously defined in the problem and derived in the solution.
  sorry

end octagon_mass_is_19kg_l167_167367


namespace max_area_of_fenced_rectangle_l167_167066

theorem max_area_of_fenced_rectangle (x y : ℝ) (h : x + y = 30) : x * y ≤ 225 :=
by {
  sorry
}

end max_area_of_fenced_rectangle_l167_167066


namespace cos_value_l167_167887

-- Given condition
axiom sin_condition (α : ℝ) : Real.sin (Real.pi / 6 + α) = 2 / 3

-- The theorem we need to prove
theorem cos_value (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = 2 / 3) : 
  Real.cos (Real.pi / 3 - α) = 2 / 3 := 
by 
  sorry

end cos_value_l167_167887


namespace frequency_count_l167_167844

theorem frequency_count (n : ℕ) (f : ℝ) (h1 : n = 1000) (h2 : f = 0.4) : n * f = 400 := by
  sorry

end frequency_count_l167_167844


namespace difference_of_digits_is_six_l167_167655

theorem difference_of_digits_is_six (a b : ℕ) (h_sum : a + b = 10) (h_number : 10 * a + b = 82) : a - b = 6 :=
sorry

end difference_of_digits_is_six_l167_167655


namespace arithmetic_sequence_geometric_ratio_l167_167891

theorem arithmetic_sequence_geometric_ratio
  (a : ℕ → ℚ) (d : ℚ)
  (h_arith : ∀ n : ℕ, a (n+1) = a n + d)
  (h_nonzero_d : d ≠ 0)
  (h_geo : (a 2) * (a 9) = (a 3) ^ 2)
  : (a 4 + a 5 + a 6) / (a 2 + a 3 + a 4) = (8 / 3) :=
by
  sorry

end arithmetic_sequence_geometric_ratio_l167_167891


namespace ratio_of_puzzle_times_l167_167301

def total_time := 70
def warmup_time := 10
def remaining_puzzles := 60 / 2

theorem ratio_of_puzzle_times : (remaining_puzzles / warmup_time) = 3 := by
  -- Given Conditions
  have H1 : 70 = 10 + 2 * (60 / 2) := by sorry
  -- Simplification and Calculation
  have H2 : (remaining_puzzles = 30) := by sorry
  -- Ratio Calculation
  have ratio_calculation: (30 / 10) = 3 := by sorry
  exact ratio_calculation

end ratio_of_puzzle_times_l167_167301


namespace decagon_triangle_probability_l167_167983

theorem decagon_triangle_probability : 
  let total_vertices := 10
  let total_triangles := Nat.choose total_vertices 3
  let favorable_triangles := 10
  (total_triangles > 0) → 
  (favorable_triangles / total_triangles : ℚ) = 1 / 12 :=
by
  sorry

end decagon_triangle_probability_l167_167983


namespace convex_polygon_diagonals_30_sides_l167_167344

theorem convex_polygon_diagonals_30_sides :
  ∀ (n : ℕ), n = 30 → ∀ (sides : ℕ), sides = n →
  let total_segments := (n * (n - 1)) / 2 in
  let diagonals := total_segments - n in
  diagonals = 405 :=
by
  intro n hn sides hs
  simp only [hn, hs]
  let total_segments := (30 * 29) / 2
  have h_total_segments : total_segments = 435 := by sorry
  let diagonals := total_segments - 30
  have h_diagonals : diagonals = 405 := by sorry
  exact h_diagonals

end convex_polygon_diagonals_30_sides_l167_167344


namespace gcd_two_powers_l167_167815

def m : ℕ := 2 ^ 1998 - 1
def n : ℕ := 2 ^ 1989 - 1

theorem gcd_two_powers :
  Nat.gcd (2 ^ 1998 - 1) (2 ^ 1989 - 1) = 511 := 
sorry

end gcd_two_powers_l167_167815


namespace smallest_p_l167_167816

theorem smallest_p (n p : ℕ) (h1 : n % 2 = 1) (h2 : n % 7 = 5) (h3 : (n + p) % 10 = 0) : p = 1 := 
sorry

end smallest_p_l167_167816


namespace probability_heads_in_nine_of_twelve_flips_l167_167482

theorem probability_heads_in_nine_of_twelve_flips : 
  (nat.choose 12 9) / (2^12) = 55 / 1024 := 
by
  sorry

end probability_heads_in_nine_of_twelve_flips_l167_167482


namespace snake_body_length_l167_167331

theorem snake_body_length (L : ℝ) (H : ℝ) (h1 : H = L / 10) (h2 : L = 10) : L - H = 9 :=
by
  sorry

end snake_body_length_l167_167331


namespace tommy_needs_to_save_l167_167810

theorem tommy_needs_to_save (books : ℕ) (cost_per_book : ℕ) (money_he_has : ℕ) 
  (total_cost : ℕ) (money_needed : ℕ) 
  (h1 : books = 8)
  (h2 : cost_per_book = 5)
  (h3 : money_he_has = 13)
  (h4 : total_cost = books * cost_per_book) :
  money_needed = total_cost - money_he_has ∧ money_needed = 27 :=
by 
  sorry

end tommy_needs_to_save_l167_167810


namespace count_ways_to_sum_2020_as_1s_and_2s_l167_167228

theorem count_ways_to_sum_2020_as_1s_and_2s : ∃ n, (∀ x y : ℕ, 4 * x + 5 * y = 2020 → x + y = n) → n = 102 :=
by
-- Mathematics proof needed.
sorry

end count_ways_to_sum_2020_as_1s_and_2s_l167_167228


namespace at_most_one_solution_l167_167415

theorem at_most_one_solution (a b c : ℝ) (hapos : 0 < a) (hbpos : 0 < b) (hcpos : 0 < c) :
  ∃! x : ℝ, a * x + b * ⌊x⌋ - c = 0 :=
sorry

end at_most_one_solution_l167_167415


namespace probability_exactly_9_heads_in_12_flips_l167_167459

noncomputable theory

open BigOperators

-- Define a function for combinations
def C(n k : ℕ) : ℕ := if h : k ≤ n then (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k))) else 0

-- The main Lean 4 statement
theorem probability_exactly_9_heads_in_12_flips :
  let total_outcomes := 2 ^ 12 in
  let favorable_outcomes := C 12 9 in
  favorable_outcomes.to_float / total_outcomes.to_float = 220 / 4096 :=
by
  let total_outcomes := 2 ^ 12
  let favorable_outcomes := C 12 9
  sorry

end probability_exactly_9_heads_in_12_flips_l167_167459


namespace solve_for_x_l167_167760

theorem solve_for_x (x : ℝ) (h : 5 * x + 3 = 10 * x - 17) : x = 4 := 
by sorry

end solve_for_x_l167_167760


namespace joan_games_last_year_l167_167921

theorem joan_games_last_year (games_this_year : ℕ) (total_games : ℕ) (games_last_year : ℕ) 
  (h1 : games_this_year = 4) 
  (h2 : total_games = 9) 
  (h3 : total_games = games_this_year + games_last_year) : 
  games_last_year = 5 := 
by
  sorry

end joan_games_last_year_l167_167921


namespace donny_spending_l167_167722

theorem donny_spending :
  (15 + 28 + 13) / 2 = 28 :=
by
  sorry

end donny_spending_l167_167722


namespace gcf_60_90_150_l167_167961

theorem gcf_60_90_150 : Nat.gcd (Nat.gcd 60 90) 150 = 30 :=
by
  sorry

end gcf_60_90_150_l167_167961


namespace cyclic_sum_inequality_l167_167242

theorem cyclic_sum_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  ( ( (a - b) * (a - c) / (a + b + c) ) + 
    ( (b - c) * (b - d) / (b + c + d) ) + 
    ( (c - d) * (c - a) / (c + d + a) ) + 
    ( (d - a) * (d - b) / (d + a + b) ) ) ≥ 0 := 
by
  sorry

end cyclic_sum_inequality_l167_167242


namespace odd_square_free_count_l167_167900

theorem odd_square_free_count : 
  ∃ n : ℕ, n = 80 ∧ ∀ k : ℕ, (k > 1 ∧ k < 200 ∧ k % 2 = 1) → 
    (¬ ∃ a : ℕ, a > 1 ∧ a * a ∣ k) → k ∈ (1 :: List.range (200 // 2)).filter (λ x, x % 2 = 1) :=
by
  sorry

end odd_square_free_count_l167_167900


namespace semicircle_area_l167_167985

theorem semicircle_area (x : ℝ) (y : ℝ) (r : ℝ) (h1 : x = 1) (h2 : y = 3) (h3 : x^2 + y^2 = (2*r)^2) :
  (1/2) * π * r^2 = (13 * π) / 8 :=
by
  sorry

end semicircle_area_l167_167985


namespace probability_of_9_heads_in_12_flips_l167_167485

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| _, 0       => 1
| 0, _       => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^12

-- Define the number of ways to get exactly 9 heads in 12 flips
def favorable_outcomes : ℕ := binom 12 9

-- Define the probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 220/4096
theorem probability_of_9_heads_in_12_flips : probability = 220 / 4096 := by
  sorry

end probability_of_9_heads_in_12_flips_l167_167485


namespace max_area_of_rectangular_pen_l167_167078

theorem max_area_of_rectangular_pen (P : ℕ) (hP : P = 60) : 
  ∃ A : ℕ, A = 225 ∧ (∀ x y : ℕ, 2 * (x + y) = P → x * y ≤ A) := 
by
  sorry

end max_area_of_rectangular_pen_l167_167078


namespace quadratic_solution_interval_l167_167763

theorem quadratic_solution_interval (a : ℝ) :
  (∃! x : ℝ, 0 < x ∧ x < 1 ∧ 2 * a * x^2 - x - 1 = 0) → a > 1 :=
by
  sorry

end quadratic_solution_interval_l167_167763


namespace total_number_of_numbers_l167_167119

-- Definitions using the conditions from the problem
def sum_of_first_4_numbers : ℕ := 4 * 4
def sum_of_last_4_numbers : ℕ := 4 * 4
def average_of_all_numbers (n : ℕ) : ℕ := 3 * n
def fourth_number : ℕ := 11
def total_sum_of_numbers : ℕ := sum_of_first_4_numbers + sum_of_last_4_numbers - fourth_number

-- Theorem stating the problem
theorem total_number_of_numbers (n : ℕ) : total_sum_of_numbers = average_of_all_numbers n → n = 7 :=
by {
  sorry
}

end total_number_of_numbers_l167_167119


namespace remainder_x_150_div_x_plus_1_pow_4_l167_167564

theorem remainder_x_150_div_x_plus_1_pow_4 :
  ∀ x : ℤ, Polynomial.x ^ 150 % (Polynomial.x + 1) ^ 4 = 551300 * Polynomial.x ^ 3 + 277161 * Polynomial.x ^ 2 + 736434 * Polynomial.x - 663863 :=
by
  intro x
  sorry

end remainder_x_150_div_x_plus_1_pow_4_l167_167564


namespace find_a5_l167_167882

variable {α : Type*} [Field α]

def geometric_seq (a : α) (q : α) (n : ℕ) : α := a * q ^ (n - 1)

theorem find_a5 (a q : α) 
  (h1 : geometric_seq a q 2 = 4)
  (h2 : geometric_seq a q 6 * geometric_seq a q 7 = 16 * geometric_seq a q 9) :
  geometric_seq a q 5 = 32 ∨ geometric_seq a q 5 = -32 :=
by
  -- Proof is omitted as per instructions
  sorry

end find_a5_l167_167882


namespace machinery_spent_correct_l167_167777

def raw_materials : ℝ := 3000
def total_amount : ℝ := 5714.29
def cash (total : ℝ) : ℝ := 0.30 * total
def machinery_spent (total : ℝ) (raw : ℝ) : ℝ := total - raw - cash total

theorem machinery_spent_correct :
  machinery_spent total_amount raw_materials = 1000 := 
  by
    sorry

end machinery_spent_correct_l167_167777


namespace zoey_preparation_months_l167_167194
open Nat

-- Define months as integers assuming 1 = January, 5 = May, 9 = September, etc.
def month_start : ℕ := 5 -- May
def month_exam : ℕ := 9 -- September

-- The function to calculate the number of months of preparation excluding the exam month.
def months_of_preparation (start : ℕ) (exam : ℕ) : ℕ := (exam - start)

theorem zoey_preparation_months :
  months_of_preparation month_start month_exam = 4 := by
  sorry

end zoey_preparation_months_l167_167194


namespace trapezoid_area_ratio_l167_167548

theorem trapezoid_area_ratio (b h x : ℝ) 
  (base_relation : b + 150 = x)
  (area_ratio : (3 / 7) * h * (b + 75) = (1 / 2) * h * (b + x))
  (mid_segment : x = b + 150) 
  : ⌊x^3 / 1000⌋ = 142 :=
by
  sorry

end trapezoid_area_ratio_l167_167548


namespace percentage_with_diploma_l167_167263

-- Define the percentages as variables for clarity
def low_income_perc := 0.25
def lower_middle_income_perc := 0.35
def upper_middle_income_perc := 0.25
def high_income_perc := 0.15

def low_income_diploma := 0.05
def lower_middle_income_diploma := 0.35
def upper_middle_income_diploma := 0.60
def high_income_diploma := 0.80

theorem percentage_with_diploma :
  (low_income_perc * low_income_diploma +
   lower_middle_income_perc * lower_middle_income_diploma +
   upper_middle_income_perc * upper_middle_income_diploma +
   high_income_perc * high_income_diploma) = 0.405 :=
by sorry

end percentage_with_diploma_l167_167263


namespace selling_price_correct_l167_167397

-- Define the parameters
def stamp_duty_rate : ℝ := 0.002
def commission_rate : ℝ := 0.0035
def bought_shares : ℝ := 3000
def buying_price_per_share : ℝ := 12
def profit : ℝ := 5967

-- Define the selling price per share
noncomputable def selling_price_per_share (x : ℝ) : ℝ :=
  bought_shares * x - bought_shares * buying_price_per_share -
  bought_shares * x * (stamp_duty_rate + commission_rate) - 
  bought_shares * buying_price_per_share * (stamp_duty_rate + commission_rate)

-- The target selling price per share
def target_selling_price_per_share : ℝ := 14.14

-- Statement of the problem
theorem selling_price_correct (x : ℝ) : selling_price_per_share x = profit → x = target_selling_price_per_share := by
  sorry

end selling_price_correct_l167_167397


namespace solution_set_l167_167382

open Real

noncomputable def f : ℝ → ℝ := sorry -- The function f is abstractly defined
axiom f_point : f 1 = 0 -- f passes through (1, 0)
axiom f_deriv_pos : ∀ (x : ℝ), x > 0 → x * (deriv f x) > 1 -- xf'(x) > 1 for x > 0

theorem solution_set (x : ℝ) : f x ≤ log x ↔ 0 < x ∧ x ≤ 1 :=
by
  sorry

end solution_set_l167_167382


namespace probability_heads_9_of_12_is_correct_l167_167455

-- Defining the probability of getting exactly 9 heads in 12 flips of a fair coin
noncomputable def probability_heads_9_of_12 := (55 : ℚ) / 1024

-- Theorem statement
theorem probability_heads_9_of_12_is_correct :
  probability_heads_9_of_12 = (55 : ℚ) / 1024 := 
  by sorry

end probability_heads_9_of_12_is_correct_l167_167455


namespace four_digit_numbers_with_5_or_7_l167_167586

theorem four_digit_numbers_with_5_or_7 : 
  let total_four_digit_numbers := 9000
  let no_5_or_7_first_digit := 7
  let no_5_or_7_other_digits := 8
  let four_digit_numbers_no_5_or_7 := no_5_or_7_first_digit * no_5_or_7_other_digits * no_5_or_7_other_digits * no_5_or_7_other_digits
  total_four_digit_numbers - four_digit_numbers_no_5_or_7 = 5416 :=
by
  let total_four_digit_numbers := 9000
  let no_5_or_7_first_digit := 7
  let no_5_or_7_other_digits := 8
  let four_digit_numbers_no_5_or_7 := no_5_or_7_first_digit * no_5_or_7_other_digits * no_5_or_7_other_digits * no_5_or_7_other_digits
  show total_four_digit_numbers - four_digit_numbers_no_5_or_7 = 5416 from sorry

end four_digit_numbers_with_5_or_7_l167_167586


namespace lcm_of_4_5_6_9_is_180_l167_167997

theorem lcm_of_4_5_6_9_is_180 : Nat.lcm (Nat.lcm 4 5) (Nat.lcm 6 9) = 180 :=
by
  sorry

end lcm_of_4_5_6_9_is_180_l167_167997


namespace gadgets_selling_prices_and_total_amount_l167_167209

def cost_price_mobile : ℕ := 16000
def cost_price_laptop : ℕ := 25000
def cost_price_camera : ℕ := 18000

def loss_percentage_mobile : ℕ := 20
def gain_percentage_laptop : ℕ := 15
def loss_percentage_camera : ℕ := 10

def selling_price_mobile : ℕ := cost_price_mobile - (cost_price_mobile * loss_percentage_mobile / 100)
def selling_price_laptop : ℕ := cost_price_laptop + (cost_price_laptop * gain_percentage_laptop / 100)
def selling_price_camera : ℕ := cost_price_camera - (cost_price_camera * loss_percentage_camera / 100)

def total_amount_received : ℕ := selling_price_mobile + selling_price_laptop + selling_price_camera

theorem gadgets_selling_prices_and_total_amount :
  selling_price_mobile = 12800 ∧
  selling_price_laptop = 28750 ∧
  selling_price_camera = 16200 ∧
  total_amount_received = 57750 := by
  sorry

end gadgets_selling_prices_and_total_amount_l167_167209


namespace total_books_l167_167873

theorem total_books (books_last_month : ℕ) (goal_factor : ℕ) (books_this_month : ℕ) (total_books : ℕ) 
  (h1 : books_last_month = 4) 
  (h2 : goal_factor = 2) 
  (h3 : books_this_month = goal_factor * books_last_month) 
  (h4 : total_books = books_last_month + books_this_month) 
  : total_books = 12 := 
by
  sorry

end total_books_l167_167873


namespace max_rectangle_area_l167_167095

-- Define the perimeter and side lengths of the rectangle
def perimeter := 60
def L (x : ℝ) := x
def W (x : ℝ) := 30 - x

-- Define the area of the rectangle
def area (x : ℝ) := L x * W x

-- State the theorem to prove the maximum area is 225 square feet
theorem max_rectangle_area : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 30 ∧ area x = 225 :=
by
  sorry

end max_rectangle_area_l167_167095


namespace range_of_t_l167_167714

noncomputable def f : ℝ → ℝ := sorry

axiom f_symmetric (x : ℝ) : f (x - 3) = f (-x - 3)
axiom f_ln_definition (x : ℝ) (h : x ≤ -3) : f x = Real.log (-x)

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f (Real.sin x - t) > f (3 * Real.sin x - 1)) ↔ (t < -1 ∨ t > 9) := sorry

end range_of_t_l167_167714


namespace bank_transfer_amount_l167_167424

/-- Paul made two bank transfers. A service charge of 2% was added to each transaction.
The second transaction was reversed without the service charge. His account balance is now $307 if 
it was $400 before he made any transfers. Prove that the amount of the first bank transfer was 
$91.18. -/
theorem bank_transfer_amount (x : ℝ) (initial_balance final_balance : ℝ) (service_charge_rate : ℝ) 
  (second_transaction_reversed : Prop)
  (h_initial : initial_balance = 400)
  (h_final : final_balance = 307)
  (h_charge : service_charge_rate = 0.02)
  (h_reversal : second_transaction_reversed):
  initial_balance - (1 + service_charge_rate) * x = final_balance ↔
  x = 91.18 := 
by
  sorry

end bank_transfer_amount_l167_167424


namespace probability_heads_9_of_12_l167_167510

theorem probability_heads_9_of_12 :
  let total_outcomes := 2^12 in
  let favorable_outcomes := Nat.choose 12 9 in
  let probability := favorable_outcomes / total_outcomes.toRat in
  probability = 220 / 4096 := 
by
  sorry

end probability_heads_9_of_12_l167_167510


namespace remainder_of_product_mod_seven_l167_167563

-- Definitions derived from the conditions
def seq : List ℕ := [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

-- The main statement to prove
theorem remainder_of_product_mod_seven : 
  (seq.foldl (λ acc x => acc * x) 1) % 7 = 0 := by
  sorry

end remainder_of_product_mod_seven_l167_167563


namespace car_gas_tank_capacity_l167_167702

theorem car_gas_tank_capacity
  (initial_mileage : ℕ)
  (final_mileage : ℕ)
  (miles_per_gallon : ℕ)
  (tank_fills : ℕ)
  (usage : initial_mileage = 1728)
  (usage_final : final_mileage = 2928)
  (car_efficiency : miles_per_gallon = 30)
  (fills : tank_fills = 2):
  (final_mileage - initial_mileage) / miles_per_gallon / tank_fills = 20 :=
by
  sorry

end car_gas_tank_capacity_l167_167702


namespace initial_oranges_correct_l167_167795

-- Define constants for the conditions
def oranges_shared : ℕ := 4
def oranges_left : ℕ := 42

-- Define the initial number of oranges
def initial_oranges : ℕ := oranges_left + oranges_shared

-- The theorem to prove
theorem initial_oranges_correct : initial_oranges = 46 :=
by 
  sorry  -- Proof to be provided

end initial_oranges_correct_l167_167795


namespace Lauryn_employs_80_men_l167_167403

theorem Lauryn_employs_80_men (W M : ℕ) 
  (h1 : M = W - 20) 
  (h2 : M + W = 180) : 
  M = 80 := 
by 
  sorry

end Lauryn_employs_80_men_l167_167403


namespace satisfies_differential_equation_l167_167832

noncomputable def y (x : ℝ) : ℝ := (Real.sin x) / x

theorem satisfies_differential_equation (x : ℝ) (hx : x ≠ 0) : 
  x * (deriv (fun x => (Real.sin x) / x) x) + (Real.sin x) / x = Real.cos x := 
by
  -- the proof goes here
  sorry

end satisfies_differential_equation_l167_167832


namespace strap_pieces_l167_167955

/-
  Given the conditions:
  1. The sum of the lengths of the two straps is 64 cm.
  2. The longer strap is 48 cm longer than the shorter strap.
  
  Prove that the number of pieces of strap that equal the length of the shorter strap 
  that can be cut from the longer strap is 7.
-/

theorem strap_pieces (S L : ℕ) (h1 : S + L = 64) (h2 : L = S + 48) :
  L / S = 7 :=
by
  sorry

end strap_pieces_l167_167955


namespace total_earnings_l167_167847

theorem total_earnings (L A J M : ℝ) 
  (hL : L = 2000) 
  (hA : A = 0.70 * L) 
  (hJ : J = 1.50 * A) 
  (hM : M = 0.40 * J) 
  : L + A + J + M = 6340 := 
  by 
    sorry

end total_earnings_l167_167847


namespace ratio_x_y_l167_167717

theorem ratio_x_y (x y : ℚ) (h : (14 * x - 5 * y) / (17 * x - 3 * y) = 2 / 3) : x / y = 1 / 23 := by
  sorry

end ratio_x_y_l167_167717


namespace parabola_intersection_difference_l167_167803

noncomputable def parabola1 (x : ℝ) := 3 * x^2 - 6 * x + 6
noncomputable def parabola2 (x : ℝ) := -2 * x^2 + 2 * x + 6

theorem parabola_intersection_difference :
  let a := 0
  let c := 8 / 5
  c - a = 8 / 5 := by
  sorry

end parabola_intersection_difference_l167_167803


namespace find_n_l167_167340

noncomputable
def equilateral_triangle_area_ratio (n : ℕ) (h : n > 4) : Prop :=
  let ratio := (2 : ℚ) / (n - 2 : ℚ)
  let area_PQR := (1 / 7 : ℚ)
  let menelaus_ap_pd := (n * (n - 2) : ℚ) / 4
  let area_triangle_ABP := (2 * (n - 2) : ℚ) / (n * (n - 2) + 4)
  let area_sum := 3 * area_triangle_ABP
  (area_sum * 7 = 6 * (n * (n - 2) + 4))

theorem find_n (n : ℕ) (h : n > 4) : 
  (equilateral_triangle_area_ratio n h) → n = 6 := sorry

end find_n_l167_167340


namespace length_of_nylon_cord_l167_167205

-- Definitions based on the conditions
def tree : ℝ := 0 -- Tree as the center point (assuming a 0 for simplicity)
def distance_ran : ℝ := 30 -- Dog ran approximately 30 feet

-- The theorem to prove
theorem length_of_nylon_cord : (distance_ran / 2) = 15 := by
  -- Assuming the dog ran along the diameter of the circle
  -- and the length of the cord is the radius of that circle.
  sorry

end length_of_nylon_cord_l167_167205


namespace abc_le_one_eighth_l167_167407

theorem abc_le_one_eighth (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a / (1 + a) + b / (1 + b) + c / (1 + c) = 1) : 
  a * b * c ≤ 1 / 8 := 
by
  sorry

end abc_le_one_eighth_l167_167407


namespace unique_integer_solution_l167_167355

theorem unique_integer_solution (x y : ℤ) : 
  x^4 + y^4 = 3 * x^3 * y → x = 0 ∧ y = 0 :=
by
  -- This is where the proof would go
  sorry

end unique_integer_solution_l167_167355


namespace non_neg_int_solutions_l167_167232

theorem non_neg_int_solutions :
  (∀ m n k : ℕ, 2 * m + 3 * n = k ^ 2 →
    (m = 0 ∧ n = 1 ∧ k = 2) ∨
    (m = 3 ∧ n = 0 ∧ k = 3) ∨
    (m = 4 ∧ n = 2 ∧ k = 5)) :=
by
  intro m n k h
  -- outline proof steps here
  sorry

end non_neg_int_solutions_l167_167232


namespace trader_goal_l167_167217

theorem trader_goal 
  (profit : ℕ)
  (half_profit : ℕ)
  (donation : ℕ)
  (total_funds : ℕ)
  (made_above_goal : ℕ)
  (goal : ℕ)
  (h1 : profit = 960)
  (h2 : half_profit = profit / 2)
  (h3 : donation = 310)
  (h4 : total_funds = half_profit + donation)
  (h5 : made_above_goal = 180)
  (h6 : goal = total_funds - made_above_goal) :
  goal = 610 :=
by 
  sorry

end trader_goal_l167_167217


namespace inequality_system_solution_l167_167283

theorem inequality_system_solution (x : ℝ) :
  (3 * x - 1 > 2 * (x + 1) ∧ (x + 2) / 3 > x - 2) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end inequality_system_solution_l167_167283


namespace consecutive_numbers_product_l167_167805

theorem consecutive_numbers_product : 
  ∃ n : ℕ, (n + n + 1 = 11) ∧ (n * (n + 1) * (n + 2) = 210) :=
sorry

end consecutive_numbers_product_l167_167805


namespace meaningful_expression_range_l167_167605

theorem meaningful_expression_range (x : ℝ) (h : 1 - x > 0) : x < 1 := sorry

end meaningful_expression_range_l167_167605


namespace permutations_behind_Alice_l167_167290

theorem permutations_behind_Alice (n : ℕ) (h : n = 7) : 
  (Nat.factorial n) = 5040 :=
by
  rw [h]
  rw [Nat.factorial]
  sorry

end permutations_behind_Alice_l167_167290


namespace tart_fill_l167_167693

theorem tart_fill (cherries blueberries total : ℚ) (h_cherries : cherries = 0.08) (h_blueberries : blueberries = 0.75) (h_total : total = 0.91) :
  total - (cherries + blueberries) = 0.08 :=
by
  sorry

end tart_fill_l167_167693


namespace relationship_among_a_b_c_l167_167369

noncomputable def a : ℝ := Real.log 0.3 / Real.log 2
noncomputable def b : ℝ := Real.exp (0.1 * Real.log 2)
noncomputable def c : ℝ := Real.exp (1.3 * Real.log 0.2)

theorem relationship_among_a_b_c : a < c ∧ c < b :=
by
  have a_neg : a < 0 :=
    by sorry
  have b_pos : b > 1 :=
    by sorry
  have c_pos : c < 1 :=
    by sorry
  sorry

end relationship_among_a_b_c_l167_167369


namespace total_crackers_l167_167225

-- Definitions based on conditions
def boxes_darren_bought : ℕ := 4
def crackers_per_box : ℕ := 24
def boxes_calvin_bought : ℕ := (2 * boxes_darren_bought) - 1

-- The statement to prove
theorem total_crackers (boxes_darren_bought = 4) (crackers_per_box = 24) : 
  (boxes_darren_bought * crackers_per_box) + (boxes_calvin_bought * crackers_per_box) = 264 := 
by 
  sorry

end total_crackers_l167_167225


namespace animal_lifespan_probability_l167_167992

theorem animal_lifespan_probability
    (P_B : ℝ) (hP_B : P_B = 0.8)
    (P_A : ℝ) (hP_A : P_A = 0.4)
    : (P_A / P_B = 0.5) :=
by
    sorry

end animal_lifespan_probability_l167_167992


namespace find_g4_l167_167800

variables (g : ℝ → ℝ)

-- Given conditions
axiom condition1 : ∀ x : ℝ, g x + 3 * g (2 - x) = 2 * x^2 + x - 1
axiom condition2 : g 4 + 3 * g (-2) = 35
axiom condition3 : g (-2) + 3 * g 4 = 5

theorem find_g4 : g 4 = -5 / 2 :=
by
  sorry

end find_g4_l167_167800


namespace calculate_f_of_f_of_f_l167_167381

def f (x : ℤ) : ℤ := 5 * x - 4

theorem calculate_f_of_f_of_f (h : f (f (f 3)) = 251) : f (f (f 3)) = 251 := 
by sorry

end calculate_f_of_f_of_f_l167_167381


namespace probability_heads_9_of_12_is_correct_l167_167454

-- Defining the probability of getting exactly 9 heads in 12 flips of a fair coin
noncomputable def probability_heads_9_of_12 := (55 : ℚ) / 1024

-- Theorem statement
theorem probability_heads_9_of_12_is_correct :
  probability_heads_9_of_12 = (55 : ℚ) / 1024 := 
  by sorry

end probability_heads_9_of_12_is_correct_l167_167454


namespace total_fertilizer_used_l167_167207

def daily_fertilizer := 3
def num_days := 12
def extra_final_day := 6

theorem total_fertilizer_used : 
    (daily_fertilizer * num_days + (daily_fertilizer + extra_final_day)) = 45 :=
by
  sorry

end total_fertilizer_used_l167_167207


namespace four_digit_numbers_with_5_or_7_l167_167588

theorem four_digit_numbers_with_5_or_7 : 
  let total_four_digit_numbers := 9000
  let no_5_or_7_first_digit := 7
  let no_5_or_7_other_digits := 8
  let four_digit_numbers_no_5_or_7 := no_5_or_7_first_digit * no_5_or_7_other_digits * no_5_or_7_other_digits * no_5_or_7_other_digits
  total_four_digit_numbers - four_digit_numbers_no_5_or_7 = 5416 :=
by
  let total_four_digit_numbers := 9000
  let no_5_or_7_first_digit := 7
  let no_5_or_7_other_digits := 8
  let four_digit_numbers_no_5_or_7 := no_5_or_7_first_digit * no_5_or_7_other_digits * no_5_or_7_other_digits * no_5_or_7_other_digits
  show total_four_digit_numbers - four_digit_numbers_no_5_or_7 = 5416 from sorry

end four_digit_numbers_with_5_or_7_l167_167588


namespace find_m_l167_167745

theorem find_m {m : ℕ} (h1 : Even (m^2 - 2 * m - 3)) (h2 : m^2 - 2 * m - 3 < 0) : m = 1 :=
sorry

end find_m_l167_167745


namespace find_cost_price_l167_167977

variable (C : ℝ)

def profit_10_percent_selling_price := 1.10 * C

def profit_15_percent_with_150_more := 1.10 * C + 150

def profit_15_percent_selling_price := 1.15 * C

theorem find_cost_price
  (h : profit_15_percent_with_150_more C = profit_15_percent_selling_price C) :
  C = 3000 :=
by
  sorry

end find_cost_price_l167_167977


namespace count_four_digit_numbers_with_5_or_7_l167_167584

def num_four_digit_numbers : Nat := 9000
def exclude_first_digit : Finset Nat := {1, 2, 3, 4, 6, 8, 9}  -- 7 options
def exclude_other_digits : Finset Nat := {0, 1, 2, 3, 4, 6, 8, 9}  -- 8 options
def excluded_numbers_count : Nat := exclude_first_digit.card * exclude_other_digits.card ^ 3  -- 3584
def included_numbers_count : Nat := num_four_digit_numbers - excluded_numbers_count  -- 5416

theorem count_four_digit_numbers_with_5_or_7 :
  included_numbers_count = 5416 :=
by
  sorry

end count_four_digit_numbers_with_5_or_7_l167_167584


namespace number_is_three_l167_167030

theorem number_is_three (n : ℝ) (h : 4 * n - 7 = 5) : n = 3 :=
by sorry

end number_is_three_l167_167030


namespace number_of_square_free_odds_l167_167903

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m * m ∣ n → false

theorem number_of_square_free_odds (n : ℕ) (h1 : 1 < n) (h2 : n < 200) (h3 : n % 2 = 1) :
  (is_square_free n) ↔ (n = 79) := by
  sorry

end number_of_square_free_odds_l167_167903


namespace minimize_acme_cost_l167_167218

theorem minimize_acme_cost (x : ℕ) : 75 + 12 * x < 16 * x → x = 19 :=
by
  intro h
  sorry

end minimize_acme_cost_l167_167218


namespace correct_calculation_l167_167180

theorem correct_calculation :
    (1 + Real.sqrt 2)^2 = 3 + 2 * Real.sqrt 2 :=
sorry

end correct_calculation_l167_167180


namespace students_in_class_l167_167766

theorem students_in_class (S : ℕ)
  (h₁ : S / 2 + 2 * S / 5 - S / 10 = 4 * S / 5)
  (h₂ : S / 5 = 4) :
  S = 20 :=
sorry

end students_in_class_l167_167766


namespace cost_of_french_bread_is_correct_l167_167427

noncomputable def cost_of_sandwiches := 2 * 7.75
noncomputable def cost_of_salami := 4.00
noncomputable def cost_of_brie := 3 * cost_of_salami
noncomputable def cost_of_olives := 10.00 * (1/4)
noncomputable def cost_of_feta := 8.00 * (1/2)
noncomputable def total_cost_of_items := cost_of_sandwiches + cost_of_salami + cost_of_brie + cost_of_olives + cost_of_feta
noncomputable def total_spent := 40.00
noncomputable def cost_of_french_bread := total_spent - total_cost_of_items

theorem cost_of_french_bread_is_correct :
  cost_of_french_bread = 2.00 :=
by
  sorry

end cost_of_french_bread_is_correct_l167_167427


namespace evaluate_power_l167_167109

theorem evaluate_power (n : ℕ) (h : 3^(2 * n) = 81) : 9^(n + 1) = 729 :=
by sorry

end evaluate_power_l167_167109


namespace intersection_of_M_and_N_l167_167746

-- Define the sets M and N
def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | x^2 - 2*x < 0}

-- The proof statement
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 1 < x ∧ x < 2} := 
  sorry

end intersection_of_M_and_N_l167_167746


namespace no_complete_divisibility_l167_167436

-- Definition of non-divisibility
def not_divides (m n : ℕ) := ¬ (m ∣ n)

theorem no_complete_divisibility (a b c d : ℕ) (h : a * d - b * c > 1) : 
  not_divides (a * d - b * c) a ∨ not_divides (a * d - b * c) b ∨ not_divides (a * d - b * c) c ∨ not_divides (a * d - b * c) d :=
by 
  sorry

end no_complete_divisibility_l167_167436


namespace linear_function_decreasing_y_l167_167288

theorem linear_function_decreasing_y (x1 y1 y2 : ℝ) :
  y1 = -2 * x1 - 7 → y2 = -2 * (x1 - 1) - 7 → y1 < y2 := by
  intros h1 h2
  sorry

end linear_function_decreasing_y_l167_167288


namespace abs_ineq_one_abs_ineq_two_l167_167282

-- First proof problem: |x-1| + |x+3| < 6 implies -4 < x < 2
theorem abs_ineq_one (x : ℝ) : |x - 1| + |x + 3| < 6 → -4 < x ∧ x < 2 :=
by
  sorry

-- Second proof problem: 1 < |3x-2| < 4 implies -2/3 ≤ x < 1/3 or 1 < x ≤ 2
theorem abs_ineq_two (x : ℝ) : 1 < |3 * x - 2| ∧ |3 * x - 2| < 4 → (-2/3) ≤ x ∧ x < (1/3) ∨ 1 < x ∧ x ≤ 2 :=
by
  sorry

end abs_ineq_one_abs_ineq_two_l167_167282


namespace proof_problem_l167_167759

theorem proof_problem :
  ∀ (X : ℝ), 213 * 16 = 3408 → (213 * 16) + (1.6 * 2.13) = X → X - (5 / 2) * 1.25 = 3408.283 :=
by
  intros X h1 h2
  sorry

end proof_problem_l167_167759


namespace probability_of_Ace_and_King_l167_167811

noncomputable def P_Ace : ℚ := 4 / 52
noncomputable def P_King_given_Ace : ℚ := 4 / 51
noncomputable def P_Ace_and_King : ℚ := P_Ace * P_King_given_Ace

theorem probability_of_Ace_and_King :
  P_Ace_and_King = 4 / 663 :=
by
  -- Directly stating the equivalence, proof to be added
  sorry

end probability_of_Ace_and_King_l167_167811


namespace functional_relationship_maximizing_profit_minimum_boxes_for_2000_profit_l167_167990

-- Definitions for the problem conditions
def cost_per_box : ℝ := 20
def min_selling_price : ℝ := 25
def max_selling_price : ℝ := 38
def base_sales_volume : ℝ := 250
def price_decrease_effect : ℝ := 10
def profit_requirement : ℝ := 2000

-- Given the initial conditions
noncomputable def sales_volume (x : ℝ) : ℝ := base_sales_volume - price_decrease_effect * (x - min_selling_price)

-- Target problem statement
-- Part 1: Functional relationship between y and x
theorem functional_relationship (x : ℝ) : sales_volume x = -10 * x + 500 := by
sorry

-- Part 2: Maximizing profit
noncomputable def profit (x : ℝ) : ℝ := (x - cost_per_box) * sales_volume x

theorem maximizing_profit : ∃ (x : ℝ), x = 35 ∧ profit x = 2250 := by
sorry

-- Part 3: Minimum number of boxes to sell for at least 2000 yuan profit
theorem minimum_boxes_for_2000_profit (x : ℝ) : x ≤ max_selling_price → profit x ≥ profit_requirement → sales_volume x ≥ 120 := by
sorry

end functional_relationship_maximizing_profit_minimum_boxes_for_2000_profit_l167_167990


namespace A_times_B_correct_l167_167410

noncomputable def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
noncomputable def B : Set ℝ := {y | y > 1}
noncomputable def A_times_B : Set ℝ := {x | (x ∈ A ∪ B) ∧ ¬(x ∈ A ∩ B)}

theorem A_times_B_correct : A_times_B = {x | (0 ≤ x ∧ x ≤ 1) ∨ x > 2} := 
sorry

end A_times_B_correct_l167_167410


namespace sales_volume_function_max_profit_min_boxes_for_2000_profit_l167_167989

-- Definitions and conditions
def cost_per_box : ℝ := 20
def min_selling_price : ℝ := 25
def init_boxes_sold : ℝ := 250
def price_increase_effect : ℝ := 10
def max_selling_price : ℝ := 38

-- Question 1: Find functional relationship between daily sales volume y and selling price per box x
theorem sales_volume_function (x : ℝ) (hx : x ≥ min_selling_price) :
  ∃ y, y = -10 * x + 500 := by
  sorry

-- Question 2: Find the price per box to maximize daily sales profit and the maximum profit
theorem max_profit (x : ℝ) (hx : x = 35) :
  ∃ P, P = -10 * (x-20) * (x) + 500 * (x-20) := by
  sorry

-- Question 3: Determine min boxes sold to make at least 2000 yuan given price does not exceed 38 yuan
theorem min_boxes_for_2000_profit (x : ℝ) (hx : min_selling_price ≤ x ∧ x ≤ max_selling_price ∧ 
                             -10 * (x-20) * (-10 * x + 500) ≥ 2000) :
  ∃ y, y = -10 * x + 500 ∧ y ≥ 120 := by
  sorry

end sales_volume_function_max_profit_min_boxes_for_2000_profit_l167_167989


namespace max_area_of_rectangle_with_perimeter_60_l167_167098

theorem max_area_of_rectangle_with_perimeter_60 :
  ∃ (x y : ℝ), 2 * (x + y) = 60 ∧ x * y = 225 :=
by
  sorry

end max_area_of_rectangle_with_perimeter_60_l167_167098


namespace max_days_for_process_C_l167_167035

/- 
  A project consists of four processes: A, B, C, and D, which require 2, 5, x, and 4 days to complete, respectively.
  The following conditions are given:
  - A and B can start at the same time.
  - C can start after A is completed.
  - D can start after both B and C are completed.
  - The total duration of the project is 9 days.
  We need to prove that the maximum number of days required to complete process C is 3.
-/
theorem max_days_for_process_C
  (A B C D : ℕ)
  (hA : A = 2)
  (hB : B = 5)
  (hD : D = 4)
  (total_duration : ℕ)
  (h_total : total_duration = 9)
  (h_condition1 : A + C + D = total_duration) : 
  C = 3 :=
by
  rw [hA, hD, h_total] at h_condition1
  linarith

#check max_days_for_process_C

end max_days_for_process_C_l167_167035


namespace shirts_sold_correct_l167_167790

-- Define the conditions
def shoes_sold := 6
def cost_per_shoe := 3
def earnings_per_person := 27
def total_earnings := 2 * earnings_per_person
def earnings_from_shoes := shoes_sold * cost_per_shoe
def cost_per_shirt := 2
def earnings_from_shirts := total_earnings - earnings_from_shoes

-- Define the total number of shirts sold and the target value to prove
def shirts_sold : Nat := earnings_from_shirts / cost_per_shirt

-- Prove that shirts_sold is 18
theorem shirts_sold_correct : shirts_sold = 18 := by
  sorry

end shirts_sold_correct_l167_167790


namespace playground_area_l167_167000

theorem playground_area (l w : ℝ) 
  (h_perimeter : 2 * l + 2 * w = 100)
  (h_length : l = 3 * w) : l * w = 468.75 :=
by
  sorry

end playground_area_l167_167000


namespace correct_propositions_l167_167103

-- Definitions according to the given conditions
def generatrix_cylinder (p1 p2 : Point) (c : Cylinder) : Prop :=
  -- Check if the line between points on upper and lower base is a generatrix
  sorry

def generatrix_cone (v : Point) (p : Point) (c : Cone) : Prop :=
  -- Check if the line from the vertex to a base point is a generatrix
  sorry

def generatrix_frustum (p1 p2 : Point) (f : Frustum) : Prop :=
  -- Check if the line between points on upper and lower base is a generatrix
  sorry

def parallel_generatrices_cylinder (gen1 gen2 : Line) (c : Cylinder) : Prop :=
  -- Check if two generatrices of the cylinder are parallel
  sorry

-- The theorem stating propositions ② and ④ are correct
theorem correct_propositions :
  generatrix_cone vertex point cone ∧
  parallel_generatrices_cylinder gen1 gen2 cylinder :=
by
  sorry

end correct_propositions_l167_167103


namespace four_digit_integers_correct_five_digit_integers_correct_l167_167452

-- Definition for the four-digit integers problem
def num_four_digit_integers := ∃ digits : Finset (Fin 5), 4 * 24 = 96

theorem four_digit_integers_correct : num_four_digit_integers := 
by
  sorry

-- Definition for the five-digit integers problem without repetition and greater than 21000
def num_five_digit_integers := ∃ digits : Finset (Fin 5), 48 + 18 = 66

theorem five_digit_integers_correct : num_five_digit_integers := 
by
  sorry

end four_digit_integers_correct_five_digit_integers_correct_l167_167452


namespace crates_of_mangoes_sold_l167_167401

def total_crates_sold := 50
def crates_grapes_sold := 13
def crates_passion_fruits_sold := 17

theorem crates_of_mangoes_sold : 
  (total_crates_sold - (crates_grapes_sold + crates_passion_fruits_sold) = 20) :=
by 
  sorry

end crates_of_mangoes_sold_l167_167401


namespace bob_total_profit_l167_167704

-- Define the given inputs
def n_dogs : ℕ := 2
def c_dog : ℝ := 250.00
def n_puppies : ℕ := 6
def c_food_vac : ℝ := 500.00
def c_ad : ℝ := 150.00
def p_puppy : ℝ := 350.00

-- The statement to prove
theorem bob_total_profit : 
  (n_puppies * p_puppy - (n_dogs * c_dog + c_food_vac + c_ad)) = 950.00 :=
by
  sorry

end bob_total_profit_l167_167704


namespace extreme_values_range_of_a_inequality_of_zeros_l167_167743

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -2 * (Real.log x) - a / (x ^ 2) + 1

theorem extreme_values (a : ℝ) (h : a = 1) :
  (∀ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x a ≤ 0) ∧
  (∃ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x a = 0) ∧
  (∀ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x a ≥ -3 + 2 * (Real.log 2)) ∧
  (∃ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x a = -3 + 2 * (Real.log 2)) :=
sorry

theorem range_of_a :
  (∀ a : ℝ, (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ 0 < a ∧ a < 1) :=
sorry

theorem inequality_of_zeros (a : ℝ) (h : 0 < a) (h1 : a < 1) (x1 x2 : ℝ) (hx1 : f x1 a = 0) (hx2 : f x2 a = 0) (hx1x2 : x1 ≠ x2) :
  1 / (x1 ^ 2) + 1 / (x2 ^ 2) > 2 / a :=
sorry

end extreme_values_range_of_a_inequality_of_zeros_l167_167743


namespace four_digit_integers_with_5_or_7_l167_167599

theorem four_digit_integers_with_5_or_7 : 
  let total := 9000 
  let valid_first_digits := [1, 2, 3, 4, 6, 8, 9]
  let valid_other_digits := [0, 1, 2, 3, 4, 6, 8, 9]
  let count_without_5_or_7 := 
      (list.length valid_first_digits) * 
      (list.length valid_other_digits) *
      (list.length valid_other_digits) *
      (list.length valid_other_digits)
  total - count_without_5_or_7 = 5416 := by
    let total := 9000
    let valid_first_digits := [1, 2, 3, 4, 6, 8, 9]
    let valid_other_digits := [0, 1, 2, 3, 4, 6, 8, 9]
    let count_without_5_or_7 := 
        (list.length valid_first_digits) * 
        (list.length valid_other_digits) *
        (list.length valid_other_digits) *
        (list.length valid_other_digits)
    show total - count_without_5_or_7 = 5416, from sorry

end four_digit_integers_with_5_or_7_l167_167599


namespace triangle_side_length_l167_167011

theorem triangle_side_length {x : ℝ} (h1 : 6 + x + x = 20) : x = 7 :=
by 
  sorry

end triangle_side_length_l167_167011


namespace union_sets_l167_167383

noncomputable def M : Set ℤ := {1, 2, 3}
noncomputable def N : Set ℤ := {x | (x + 1) * (x - 2) < 0}

theorem union_sets : M ∪ N = {0, 1, 2, 3} := by
  sorry

end union_sets_l167_167383


namespace units_digit_6_pow_6_l167_167177

theorem units_digit_6_pow_6 : (6 ^ 6) % 10 = 6 := 
by {
  sorry
}

end units_digit_6_pow_6_l167_167177


namespace M_inter_N_l167_167107

def M : Set ℝ := {x | x^2 - x = 0}
def N : Set ℝ := {-1, 0}

theorem M_inter_N :
  M ∩ N = {0} :=
by
  sorry

end M_inter_N_l167_167107


namespace max_rectangle_area_l167_167086

theorem max_rectangle_area (P : ℝ) (hP : P = 60) : ∃ A : ℝ, A = 225 :=
by 
  sorry

end max_rectangle_area_l167_167086


namespace comparison_arctan_l167_167923

theorem comparison_arctan (a b c : ℝ) (h : Real.arctan a + Real.arctan b + Real.arctan c + Real.pi / 2 = 0) :
  (a * b + b * c + c * a = 1) ∧ (a + b + c < a * b * c) :=
by
  sorry

end comparison_arctan_l167_167923


namespace yuri_total_puppies_l167_167185

-- Conditions
def first_week_puppies := 20
def second_week_puppies : ℕ := (2 / 5) * first_week_puppies
def third_week_puppies : ℕ := 2 * second_week_puppies
def fourth_week_puppies := first_week_puppies + 10

-- Total puppies
def total_puppies : ℕ := first_week_puppies + second_week_puppies + third_week_puppies + fourth_week_puppies

-- Theorem to prove
theorem yuri_total_puppies : total_puppies = 74 :=
by sorry

end yuri_total_puppies_l167_167185


namespace inscribed_sphere_radius_l167_167216

theorem inscribed_sphere_radius (b d : ℝ) : 
  (b * Real.sqrt d - b = 15 * (Real.sqrt 5 - 1) / 4) → 
  b + d = 11.75 :=
by
  intro h
  sorry

end inscribed_sphere_radius_l167_167216


namespace edric_monthly_salary_l167_167727

theorem edric_monthly_salary 
  (hours_per_day : ℝ)
  (days_per_week : ℝ)
  (weeks_per_month : ℝ)
  (hourly_rate : ℝ) :
  hours_per_day = 8 ∧ days_per_week = 6 ∧ weeks_per_month = 4.33 ∧ hourly_rate = 3 →
  (hours_per_day * days_per_week * weeks_per_month * hourly_rate) = 623.52 :=
by
  intros h
  sorry

end edric_monthly_salary_l167_167727


namespace machine_p_vs_machine_q_l167_167623

variable (MachineA_rate MachineQ_rate MachineP_rate : ℝ)
variable (Total_sprockets : ℝ := 550)
variable (Production_rate_A : ℝ := 5)
variable (Production_rate_Q : ℝ := MachineA_rate + 0.1 * MachineA_rate)
variable (Time_Q : ℝ := Total_sprockets / Production_rate_Q)
variable (Time_P : ℝ)
variable (Difference : ℝ)

noncomputable def production_times_difference (MachineA_rate MachineQ_rate MachineP_rate : ℝ) : ℝ :=
  let Production_rate_Q := MachineA_rate + 0.1 * MachineA_rate
  let Time_Q := Total_sprockets / Production_rate_Q
  let Difference := Time_P - Time_Q
  Difference

theorem machine_p_vs_machine_q : 
  Production_rate_A = 5 → 
  Total_sprockets = 550 →
  Production_rate_Q = 5.5 →
  Time_Q = 100 →
  MachineP_rate = MachineP_rate →
  Time_P = Time_P →
  Difference = (Time_P - Time_Q) :=
by
  intros
  sorry

end machine_p_vs_machine_q_l167_167623


namespace proof_true_proposition_l167_167248

open Classical

def P : Prop := ∀ x : ℝ, x^2 ≥ 0
def Q : Prop := ∃ x : ℚ, x^2 ≠ 3
def true_proposition (p q : Prop) := p ∨ ¬q

theorem proof_true_proposition : P ∧ ¬Q → true_proposition P Q :=
by
  intro h
  sorry

end proof_true_proposition_l167_167248


namespace max_rectangle_area_l167_167090

-- Define the perimeter and side lengths of the rectangle
def perimeter := 60
def L (x : ℝ) := x
def W (x : ℝ) := 30 - x

-- Define the area of the rectangle
def area (x : ℝ) := L x * W x

-- State the theorem to prove the maximum area is 225 square feet
theorem max_rectangle_area : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 30 ∧ area x = 225 :=
by
  sorry

end max_rectangle_area_l167_167090


namespace polynomial_not_33_l167_167934

theorem polynomial_not_33 (x y : ℤ) : x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 := 
sorry

end polynomial_not_33_l167_167934


namespace Donny_spends_28_on_Thursday_l167_167723

theorem Donny_spends_28_on_Thursday :
  let monday_savings := 15
  let tuesday_savings := 28
  let wednesday_savings := 13
  let total_savings := monday_savings + tuesday_savings + wednesday_savings
  let thursday_spending := total_savings / 2
  thursday_spending = 28 :=
by 
  let monday_savings := 15
  let tuesday_savings := 28
  let wednesday_savings := 13
  let total_savings := monday_savings + tuesday_savings + wednesday_savings
  let thursday_spending := total_savings / 2
  sorry

end Donny_spends_28_on_Thursday_l167_167723


namespace farmer_john_pairs_l167_167859

noncomputable def farmer_john_animals_pairing :
    Nat := 
  let cows := 5
  let pigs := 4
  let horses := 7
  let num_ways_cow_pig_pair := cows * pigs
  let num_ways_horses_remaining := Nat.factorial horses
  num_ways_cow_pig_pair * num_ways_horses_remaining

theorem farmer_john_pairs : farmer_john_animals_pairing = 100800 := 
by
  sorry

end farmer_john_pairs_l167_167859


namespace max_area_of_rectangular_pen_l167_167073

-- Define the perimeter and derive the formula for the area
def perimeter := 60
def half_perimeter := perimeter / 2
def area (x : ℝ) := x * (half_perimeter - x)

-- Statement of the problem: prove the maximum area is 225 square feet
theorem max_area_of_rectangular_pen : ∃ x : ℝ, 0 ≤ x ∧ x ≤ half_perimeter ∧ area x = 225 := 
sorry

end max_area_of_rectangular_pen_l167_167073


namespace value_of_b_minus_a_l167_167024

variable (a b : ℕ)

theorem value_of_b_minus_a 
  (h1 : b = 10)
  (h2 : a * b = 2 * (a + b) + 12) : b - a = 6 :=
by sorry

end value_of_b_minus_a_l167_167024


namespace marta_should_buy_84_ounces_l167_167928

/-- Definition of the problem's constants and assumptions --/
def apple_weight : ℕ := 4
def orange_weight : ℕ := 3
def bag_capacity : ℕ := 49
def num_bags : ℕ := 3

-- Marta wants to put the same number of apples and oranges in each bag
def equal_fruit (A O : ℕ) := A = O

-- Each bag should hold up to 49 ounces of fruit
def bag_limit (n : ℕ) := 4 * n + 3 * n ≤ 49

-- Marta's total apple weight based on the number of apples per bag and number of bags
def total_apple_weight (A : ℕ) : ℕ := (A * 3 * 4)

/-- Statement of the proof problem: 
Marta should buy 84 ounces of apples --/
theorem marta_should_buy_84_ounces : total_apple_weight 7 = 84 :=
by
  sorry

end marta_should_buy_84_ounces_l167_167928


namespace complex_in_fourth_quadrant_l167_167060

theorem complex_in_fourth_quadrant (m : ℝ) :
  (m^2 - 8*m + 15 > 0) ∧ (m^2 - 5*m - 14 < 0) →
  (-2 < m ∧ m < 3) ∨ (5 < m ∧ m < 7) :=
sorry

end complex_in_fourth_quadrant_l167_167060


namespace correct_substitution_l167_167310

theorem correct_substitution (x y : ℝ) 
  (h1 : y = 1 - x) 
  (h2 : x - 2 * y = 4) : x - 2 + 2 * x = 4 :=
by
  sorry

end correct_substitution_l167_167310


namespace motorcycles_meet_after_54_minutes_l167_167827

noncomputable def motorcycles_meet_time : ℕ := sorry

theorem motorcycles_meet_after_54_minutes :
  motorcycles_meet_time = 54 := sorry

end motorcycles_meet_after_54_minutes_l167_167827


namespace greatest_divisor_consistent_remainder_l167_167354

noncomputable def gcd_of_differences : ℕ :=
  Nat.gcd (Nat.gcd 1050 28770) 71670

theorem greatest_divisor_consistent_remainder :
  gcd_of_differences = 30 :=
by
  -- The proof can be filled in here...
  sorry

end greatest_divisor_consistent_remainder_l167_167354


namespace percentage_salt_l167_167333

-- Variables
variables {S1 S2 R : ℝ}

-- Conditions
def first_solution := S1
def second_solution := (25 / 100) * 19.000000000000007
def resulting_solution := 16

theorem percentage_salt (S1 S2 : ℝ) (H1: S2 = 19.000000000000007) 
(H2: (75 / 100) * S1 + (25 / 100) * S2 = 16) : 
S1 = 15 :=
by
    rw [H1] at H2
    sorry

end percentage_salt_l167_167333


namespace no_such_natural_number_exists_l167_167047

theorem no_such_natural_number_exists :
  ¬ ∃ (n : ℕ), (∃ (m k : ℤ), 2 * n - 5 = 9 * m ∧ n - 2 = 15 * k) :=
by
  sorry

end no_such_natural_number_exists_l167_167047


namespace cost_of_insulation_l167_167037

def rectangular_tank_dimension_l : ℕ := 6
def rectangular_tank_dimension_w : ℕ := 3
def rectangular_tank_dimension_h : ℕ := 2
def total_cost : ℕ := 1440

def surface_area (l w h : ℕ) : ℕ := 2 * (l * w + l * h + w * h)

def cost_per_square_foot (total_cost surface_area : ℕ) : ℕ := total_cost / surface_area

theorem cost_of_insulation : 
  cost_per_square_foot total_cost (surface_area rectangular_tank_dimension_l rectangular_tank_dimension_w rectangular_tank_dimension_h) = 20 :=
by
  sorry

end cost_of_insulation_l167_167037


namespace congruence_solution_exists_l167_167924

theorem congruence_solution_exists {p n a : ℕ} (hp : Prime p) (hn : n % p ≠ 0) (ha : a % p ≠ 0)
  (hx : ∃ x : ℕ, x^n % p = a % p) :
  ∀ r : ℕ, ∃ x : ℕ, x^n % (p^(r + 1)) = a % (p^(r + 1)) :=
by
  intros r
  sorry

end congruence_solution_exists_l167_167924


namespace octagon_mass_is_19kg_l167_167366

-- Define the parameters given in the problem
def side_length_square_sheet := 1  -- side length in meters
def thickness_sheet := 0.3  -- thickness in cm (3 mm)
def density_steel := 7.8  -- density in g/cm³

-- Given the geometric transformations and constants, prove the mass of the octagon
theorem octagon_mass_is_19kg :
  ∃ mass : ℝ, (mass = 19) :=
by
  -- Placeholder for the proof.
  -- The detailed steps would include geometrical transformations and volume calculations,
  -- which have been rigorously defined in the problem and derived in the solution.
  sorry

end octagon_mass_is_19kg_l167_167366


namespace smallest_k_l167_167637

theorem smallest_k (a b : ℚ) (h_a_period : ∀ n, a ≠ (10^30 - 1) * n)
  (h_b_period : ∀ n, b ≠ (10^30 - 1) * n)
  (h_diff_period : ∀ n, a - b ≠ (10^15 - 1) * n) :
  ∃ k : ℕ, k = 6 ∧ (a + (k:ℚ) * b) ≠ (10^15 - 1) :=
sorry

end smallest_k_l167_167637


namespace wider_can_radius_l167_167173

theorem wider_can_radius (h : ℝ) : 
  (∃ r : ℝ, ∀ V : ℝ, V = π * 8^2 * 2 * h → V = π * r^2 * h → r = 8 * Real.sqrt 2) :=
by 
  sorry

end wider_can_radius_l167_167173


namespace fractional_part_exceeds_bound_l167_167922

noncomputable def x (a b : ℕ) : ℝ := Real.sqrt a + Real.sqrt b

theorem fractional_part_exceeds_bound
  (a b : ℕ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hx_not_int : ¬ (∃ n : ℤ, x a b = n))
  (hx_lt : x a b < 1976) :
    x a b % 1 > 3.24e-11 :=
sorry

end fractional_part_exceeds_bound_l167_167922


namespace tangent_line_value_of_a_l167_167113

theorem tangent_line_value_of_a (a : ℝ) :
  (∃ (m : ℝ), (2 * m - 1 = a * m + Real.log m) ∧ (a + 1 / m = 2)) → a = 1 :=
by 
sorry

end tangent_line_value_of_a_l167_167113


namespace ratio_second_to_first_l167_167421

noncomputable def ratio_of_second_to_first (x y z : ℕ) (k : ℕ) : ℕ := sorry

theorem ratio_second_to_first
    (x y z : ℕ)
    (h1 : z = 2 * y)
    (h2 : y = k * x)
    (h3 : (x + y + z) / 3 = 78)
    (h4 : x = 18)
    (k_val : k = 4):
  ratio_of_second_to_first x y z k = 4 := sorry

end ratio_second_to_first_l167_167421


namespace distance_and_speed_l167_167659

-- Define the conditions given in the problem
def first_car_speed (y : ℕ) := y + 4
def second_car_speed (y : ℕ) := y
def third_car_speed (y : ℕ) := y - 6

def time_relation1 (x : ℕ) (y : ℕ) :=
  x / (first_car_speed y) = x / (second_car_speed y) - 3 / 60

def time_relation2 (x : ℕ) (y : ℕ) :=
  x / (second_car_speed y) = x / (third_car_speed y) - 5 / 60 

-- State the theorem to prove both the distance and the speed of the second car
theorem distance_and_speed : ∃ (x y : ℕ), 
  time_relation1 x y ∧ 
  time_relation2 x y ∧ 
  x = 120 ∧ 
  y = 96 :=
by
  sorry

end distance_and_speed_l167_167659


namespace decagon_triangle_probability_l167_167982

theorem decagon_triangle_probability :
  let n := 10 in
  let total_ways := Nat.choose n 3 in
  let favorable_ways := n in
  (favorable_ways / total_ways : ℚ) = 1 / 12 :=
by
  sorry

end decagon_triangle_probability_l167_167982


namespace count_four_digit_integers_with_5_or_7_l167_167591

theorem count_four_digit_integers_with_5_or_7 : 
  (set.range (λ n: ℕ, n >= 1000 ∧ n <= 9999) 
   \ { n | ∀ d ∈ toDigits n, d ≠ 5 ∧ d ≠ 7 }).card = 5416 :=
by
  sorry

end count_four_digit_integers_with_5_or_7_l167_167591


namespace problem_10_order_l167_167316

theorem problem_10_order (a b c : ℝ) (h1 : a = Real.sin (17 * Real.pi / 180) * Real.cos (45 * Real.pi / 180) + Real.cos (17 * Real.pi / 180) * Real.sin (45 * Real.pi / 180))
    (h2 : b = 2 * (Real.cos (13 * Real.pi / 180))^2 - 1)
    (h3 : c = Real.sqrt 3 / 2) :
    c < a ∧ a < b :=
sorry

end problem_10_order_l167_167316


namespace max_sum_of_triplet_product_60_l167_167438

theorem max_sum_of_triplet_product_60 : 
  ∃ a b c : ℕ, a * b * c = 60 ∧ a + b + c = 62 :=
sorry

end max_sum_of_triplet_product_60_l167_167438


namespace batsman_average_19th_inning_l167_167195

theorem batsman_average_19th_inning (initial_avg : ℝ) 
    (scored_19th_inning : ℝ) 
    (new_avg : ℝ) 
    (h1 : scored_19th_inning = 100) 
    (h2 : new_avg = initial_avg + 2)
    (h3 : new_avg = (18 * initial_avg + 100) / 19) :
    new_avg = 64 :=
by
  have h4 : initial_avg = 62 := by
    sorry
  sorry

end batsman_average_19th_inning_l167_167195


namespace scientific_notation_of_twenty_million_l167_167428

-- Define the number 20 million
def twenty_million : ℂ :=
  20000000

-- Define the scientific notation to be proved correct
def scientific_notation : ℂ :=
  2 * 10 ^ 7

-- The theorem to prove the equivalence
theorem scientific_notation_of_twenty_million : twenty_million = scientific_notation :=
  sorry

end scientific_notation_of_twenty_million_l167_167428


namespace quadratic_distinct_real_roots_l167_167606

theorem quadratic_distinct_real_roots (k : ℝ) : k < 1 / 2 ∧ k ≠ 0 ↔ (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (k * x1^2 - 2 * x1 + 2 = 0) ∧ (k * x2^2 - 2 * x2 + 2 = 0)) := 
by 
  sorry

end quadratic_distinct_real_roots_l167_167606


namespace min_ab_l167_167114

theorem min_ab (a b : ℝ) (h_cond1 : a > 0) (h_cond2 : b > 0)
  (h_eq : a * b = a + b + 3) : a * b = 9 :=
sorry

end min_ab_l167_167114


namespace find_principal_amount_l167_167804

theorem find_principal_amount :
  ∃ P : ℝ, P * (1 + 0.05) ^ 4 = 9724.05 ∧ P = 8000 :=
by
  sorry

end find_principal_amount_l167_167804


namespace max_rectangle_area_l167_167094

-- Define the perimeter and side lengths of the rectangle
def perimeter := 60
def L (x : ℝ) := x
def W (x : ℝ) := 30 - x

-- Define the area of the rectangle
def area (x : ℝ) := L x * W x

-- State the theorem to prove the maximum area is 225 square feet
theorem max_rectangle_area : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 30 ∧ area x = 225 :=
by
  sorry

end max_rectangle_area_l167_167094


namespace exponential_quotient_l167_167065

variable {x a b : ℝ}

theorem exponential_quotient (h1 : x^a = 3) (h2 : x^b = 5) : x^(a-b) = 3 / 5 :=
sorry

end exponential_quotient_l167_167065


namespace ratio_of_lengths_l167_167685

noncomputable def total_fence_length : ℝ := 640
noncomputable def short_side_length : ℝ := 80

theorem ratio_of_lengths (L S : ℝ) (h1 : 2 * L + 2 * S = total_fence_length) (h2 : S = short_side_length) :
  L / S = 3 :=
by {
  sorry
}

end ratio_of_lengths_l167_167685


namespace zeros_in_square_of_999_999_999_l167_167701

noncomputable def number_of_zeros_in_square (n : ℕ) : ℕ :=
  if n ≥ 1 then n - 1 else 0

theorem zeros_in_square_of_999_999_999 :
  number_of_zeros_in_square 9 = 8 :=
sorry

end zeros_in_square_of_999_999_999_l167_167701


namespace quadrants_cos_sin_identity_l167_167247

theorem quadrants_cos_sin_identity (α : ℝ) 
  (h1 : π < α ∧ α < 2 * π)  -- α in the fourth quadrant
  (h2 : Real.cos α = 3 / 5) :
  (1 + Real.sqrt 2 * Real.cos (2 * α - π / 4)) / 
  (Real.sin (α + π / 2)) = -2 / 5 :=
by
  sorry

end quadrants_cos_sin_identity_l167_167247


namespace min_a2_k2b2_l167_167275

variable (a b t k : ℝ)
variable (hk : 0 < k)
variable (h : a + k * b = t)

theorem min_a2_k2b2 (a b t k : ℝ) (hk : 0 < k) (h : a + k * b = t) :
  a^2 + (k * b)^2 ≥ (1 + k^2) * (t^2) / ((1 + k)^2) :=
sorry

end min_a2_k2b2_l167_167275


namespace arithmetic_mean_of_geometric_sequence_l167_167345

theorem arithmetic_mean_of_geometric_sequence (a r : ℕ) (h_a : a = 4) (h_r : r = 3) :
    ((a) + (a * r) + (a * r^2)) / 3 = (52 / 3) :=
by
  sorry

end arithmetic_mean_of_geometric_sequence_l167_167345


namespace bad_oranges_l167_167537

theorem bad_oranges (total_oranges : ℕ) (students : ℕ) (less_oranges_per_student : ℕ)
  (initial_oranges_per_student now_oranges_per_student shared_oranges now_total_oranges bad_oranges : ℕ) :
  total_oranges = 108 →
  students = 12 →
  less_oranges_per_student = 3 →
  initial_oranges_per_student = total_oranges / students →
  now_oranges_per_student = initial_oranges_per_student - less_oranges_per_student →
  shared_oranges = students * now_oranges_per_student →
  now_total_oranges = 72 →
  bad_oranges = total_oranges - now_total_oranges →
  bad_oranges = 36 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end bad_oranges_l167_167537


namespace age_of_replaced_person_is_46_l167_167158

variable (age_of_replaced_person : ℕ)
variable (new_person_age : ℕ := 16)
variable (decrease_in_age_per_person : ℕ := 3)
variable (number_of_people : ℕ := 10)

theorem age_of_replaced_person_is_46 :
  age_of_replaced_person - new_person_age = decrease_in_age_per_person * number_of_people → 
  age_of_replaced_person = 46 :=
by
  sorry

end age_of_replaced_person_is_46_l167_167158


namespace iced_tea_cost_is_correct_l167_167707

noncomputable def iced_tea_cost (cost_cappuccino cost_latte cost_espresso : ℝ) (num_cappuccino num_iced_tea num_latte num_espresso : ℕ) (bill_amount change_amount : ℝ) : ℝ :=
  let total_cappuccino_cost := cost_cappuccino * num_cappuccino
  let total_latte_cost := cost_latte * num_latte
  let total_espresso_cost := cost_espresso * num_espresso
  let total_spent := bill_amount - change_amount
  let total_other_cost := total_cappuccino_cost + total_latte_cost + total_espresso_cost
  let total_iced_tea_cost := total_spent - total_other_cost
  total_iced_tea_cost / num_iced_tea

theorem iced_tea_cost_is_correct:
  iced_tea_cost 2 1.5 1 3 2 2 2 20 3 = 3 :=
by
  sorry

end iced_tea_cost_is_correct_l167_167707


namespace polygon_diagonals_30_l167_167343

-- Define the properties and conditions of the problem
def sides := 30

-- Define the number of diagonals calculation function
def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- The proof statement to check the number of diagonals in a 30-sided convex polygon
theorem polygon_diagonals_30 : num_diagonals sides = 375 := by
  sorry

end polygon_diagonals_30_l167_167343


namespace probability_of_two_red_two_green_l167_167202

def red_balls : ℕ := 10
def green_balls : ℕ := 8
def total_balls : ℕ := red_balls + green_balls
def drawn_balls : ℕ := 4

def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def prob_two_red_two_green : ℚ :=
  (combination red_balls 2 * combination green_balls 2 : ℚ) / combination total_balls drawn_balls

theorem probability_of_two_red_two_green :
  prob_two_red_two_green = 7 / 17 := 
sorry

end probability_of_two_red_two_green_l167_167202


namespace abc_less_than_one_l167_167281

variables {a b c : ℝ}

theorem abc_less_than_one (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1: a^2 < b) (h2: b^2 < c) (h3: c^2 < a) : a < 1 ∧ b < 1 ∧ c < 1 := by
  sorry

end abc_less_than_one_l167_167281


namespace circle_center_l167_167561

theorem circle_center (x y : ℝ) (h : x^2 - 4 * x + y^2 - 6 * y - 12 = 0) : (x, y) = (2, 3) :=
sorry

end circle_center_l167_167561


namespace max_area_of_rectangle_with_perimeter_60_l167_167097

theorem max_area_of_rectangle_with_perimeter_60 :
  ∃ (x y : ℝ), 2 * (x + y) = 60 ∧ x * y = 225 :=
by
  sorry

end max_area_of_rectangle_with_perimeter_60_l167_167097


namespace probability_exactly_9_heads_in_12_flips_l167_167460

noncomputable theory

open BigOperators

-- Define a function for combinations
def C(n k : ℕ) : ℕ := if h : k ≤ n then (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k))) else 0

-- The main Lean 4 statement
theorem probability_exactly_9_heads_in_12_flips :
  let total_outcomes := 2 ^ 12 in
  let favorable_outcomes := C 12 9 in
  favorable_outcomes.to_float / total_outcomes.to_float = 220 / 4096 :=
by
  let total_outcomes := 2 ^ 12
  let favorable_outcomes := C 12 9
  sorry

end probability_exactly_9_heads_in_12_flips_l167_167460


namespace circle_equation_slope_intercept_l167_167880
noncomputable theory

open_locale real

-- Definitions of points A and B
def A := (0 : ℝ, 2 : ℝ)
def B := (2 : ℝ, -2 : ℝ)

-- Condition: The center of the circle lies on the line x - y + 1 = 0.
def center_condition (center : ℝ × ℝ) : Prop := 
  center.fst - center.snd + 1 = 0

-- Question I: Find the standard equation of circle C.
theorem circle_equation (t : ℝ) (center := (t, t + 1)) (r := sqrt ((-3) ^ 2 + (-2 - 2) ^ 2)) 
  (h_center : center_condition center) (h_center_eq : center = (-3, -2)): 
  (r ^ 2 = 25) → 
  ((∀ (x y : ℝ), (x + 3) ^ 2 + (y + 2) ^ 2) = 25) :=
sorry

-- Question II: Find the slope-intercept equation of line m.
theorem slope_intercept (k : ℝ) (intercept := (43 : ℝ) / 12)
  (h1 : k = 5 / 12) :
  ((∀ (x y : ℝ), y = k * x + intercept)) :=
sorry

end circle_equation_slope_intercept_l167_167880


namespace part_a_l167_167976

theorem part_a (a b c : ℕ) (h : (a + b + c) % 6 = 0) : (a^5 + b^3 + c) % 6 = 0 := 
by sorry

end part_a_l167_167976


namespace range_of_varphi_l167_167104

noncomputable def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) : ℝ := 2 * Real.sin (ω * x + ϕ) + 1

theorem range_of_varphi (ω ϕ : ℝ) (h_ω_pos : ω > 0) (h_ϕ_bound : |ϕ| ≤ (Real.pi) / 2)
  (h_intersection : (∀ x, f x ω ϕ = -1 → (∃ k : ℤ, x = (k * Real.pi) / ω)))
  (h_f_gt_1 : (∀ x, -Real.pi / 12 < x ∧ x < Real.pi / 3 → f x ω ϕ > 1)) :
  ω = 2 → (Real.pi / 6 ≤ ϕ) ∧ (ϕ ≤ Real.pi / 3) :=
by
  sorry

end range_of_varphi_l167_167104


namespace problem_value_l167_167018

theorem problem_value :
  1 - (-2) - 3 - (-4) - 5 - (-6) = 5 :=
by sorry

end problem_value_l167_167018


namespace train_distance_after_braking_l167_167999

theorem train_distance_after_braking : 
  (∃ t : ℝ, (27 * t - 0.45 * t^2 = 0) ∧ (∀ s : ℝ, s = 27 * t - 0.45 * t^2) ∧ s = 405) :=
sorry

end train_distance_after_braking_l167_167999


namespace num_four_digit_with_5_or_7_l167_167594

def num_four_digit_without_5_or_7 : Nat := 7 * 8 * 8 * 8

def num_four_digit_total : Nat := 9999 - 1000 + 1

theorem num_four_digit_with_5_or_7 : 
  num_four_digit_total - num_four_digit_without_5_or_7 = 5416 :=
by
  sorry

end num_four_digit_with_5_or_7_l167_167594


namespace sugar_amount_l167_167199

theorem sugar_amount (S F B : ℝ) 
    (h_ratio1 : S = F) 
    (h_ratio2 : F = 10 * B) 
    (h_ratio3 : F / (B + 60) = 8) : S = 2400 := 
by
  sorry

end sugar_amount_l167_167199


namespace splay_sequence_problem_l167_167950

def is_relatively_prime (m n : ℕ) : Prop :=
  Nat.gcd m n = 1

def set_splay_sequences : Set (List ℝ) :=
  {C | ∀ c ∈ C, 0 < c ∧ c < 1 }

def power (C : List ℝ) : ℝ :=
  C.foldr (· * ·) 1  -- product of the elements in the list

def sum_powers (S : Set (List ℝ)) : ℚ :=
  ⟨S.toList.map power).sum, 1⟩ -- Q it to rational as \frac{sum}{1}

theorem splay_sequence_problem :
  ∃ (m n : ℕ), is_relatively_prime m n ∧ sum_powers set_splay_sequences = ⟨m, n⟩ ∧ 100 * m + n = 4817 :=
by
  sorry

end splay_sequence_problem_l167_167950


namespace region_area_l167_167273

variable {θ : Real}
variable {R : ℝ}
variable {r : ℝ}

noncomputable def largestPossibleArea (θ : Real) (hθ1 : θ > Real.pi / 2) (hθ2 : θ < Real.pi) (hθ3 : Real.sin θ = 3 / 5) : ℝ :=
  let d : ℝ := 1
  let area : ℝ := Real.pi * (R^2 - r^2)
  have hr : R - r = d := by sorry
  have hP : R^2 - r^2 = (d/2)^2 := by sorry
  let rSquaredTerm := (d/2)^2
  have area_eq : area = Real.pi * rSquaredTerm := by sorry
  have final_area : area = Real.pi / 4 := by
    rw [area_eq, rSquaredTerm]
    simp
  final_area

theorem region_area (θ : Real) (hθ1 : θ > Real.pi / 2) (hθ2 : θ < Real.pi) (hθ3 : Real.sin θ = 3 / 5) : largestPossibleArea θ hθ1 hθ2 hθ3 = Real.pi / 4 := by
  sorry

end region_area_l167_167273


namespace complex_z_solution_l167_167377

theorem complex_z_solution (z : ℂ) (i : ℂ) (h : i * z = 1 - i) (hi : i * i = -1) : z = -1 - i :=
by sorry

end complex_z_solution_l167_167377


namespace first_group_men_l167_167260

theorem first_group_men (M : ℕ) (h : M * 15 = 25 * 24) : M = 40 := sorry

end first_group_men_l167_167260


namespace total_length_of_rubber_pen_pencil_l167_167995

variable (rubber pen pencil : ℕ)

theorem total_length_of_rubber_pen_pencil 
  (h1 : pen = rubber + 3)
  (h2 : pen = pencil - 2)
  (h3 : pencil = 12) : rubber + pen + pencil = 29 := by
  sorry

end total_length_of_rubber_pen_pencil_l167_167995


namespace alice_sold_20_pears_l167_167695

-- Definitions (Conditions)
def canned_more_than_poached (C P : ℝ) : Prop := C = P + 0.2 * P
def poached_less_than_sold (P S : ℝ) : Prop := P = 0.5 * S
def total_pears (S C P : ℝ) : Prop := S + C + P = 42

-- Theorem statement
theorem alice_sold_20_pears (S C P : ℝ) (h1 : canned_more_than_poached C P) (h2 : poached_less_than_sold P S) (h3 : total_pears S C P) : S = 20 :=
by 
  -- This is where the proof would go, but for now, we use sorry to signify it's omitted.
  sorry

end alice_sold_20_pears_l167_167695


namespace sets_are_equal_l167_167640

def setA : Set ℤ := {a | ∃ m n l : ℤ, a = 12 * m + 8 * n + 4 * l}
def setB : Set ℤ := {b | ∃ p q r : ℤ, b = 20 * p + 16 * q + 12 * r}

theorem sets_are_equal : setA = setB := sorry

end sets_are_equal_l167_167640


namespace bryan_bought_4_pairs_of_pants_l167_167850

def number_of_tshirts : Nat := 5
def total_cost : Nat := 1500
def cost_per_tshirt : Nat := 100
def cost_per_pants : Nat := 250

theorem bryan_bought_4_pairs_of_pants : (total_cost - number_of_tshirts * cost_per_tshirt) / cost_per_pants = 4 := by
  sorry

end bryan_bought_4_pairs_of_pants_l167_167850


namespace correct_result_l167_167557

-- Given condition
def mistaken_calculation (x : ℤ) : Prop :=
  x / 3 = 45

-- Proposition to prove the correct result
theorem correct_result (x : ℤ) (h : mistaken_calculation x) : 3 * x = 405 := by
  -- Here we can solve the proof later
  sorry

end correct_result_l167_167557


namespace country_albums_count_l167_167523

-- Definitions based on conditions
def pop_albums : Nat := 8
def songs_per_album : Nat := 7
def total_songs : Nat := 70

-- Theorem to prove the number of country albums
theorem country_albums_count : (total_songs - pop_albums * songs_per_album) / songs_per_album = 2 := by
  sorry

end country_albums_count_l167_167523


namespace problem_solution_l167_167276

noncomputable def a : ℝ := Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 10
noncomputable def b : ℝ := -Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 10
noncomputable def c : ℝ := Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 10
noncomputable def d : ℝ := -Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 10

theorem problem_solution : ((1 / a) + (1 / b) + (1 / c) + (1 / d))^2 = 0 :=
by
  sorry

end problem_solution_l167_167276


namespace max_val_xa_minus_2x_max_val_ab_plus_bc_plus_ac_l167_167980

-- First proof problem
theorem max_val_xa_minus_2x (x a : ℝ) (h1 : 0 < x) (h2 : 2 * x < a) :
  ∃ y, (y = x * (a - 2 * x)) ∧ y ≤ a^2 / 8 :=
sorry

-- Second proof problem
theorem max_val_ab_plus_bc_plus_ac (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + b^2 + c^2 = 4) :
  ab + bc + ac ≤ 4 :=
sorry

end max_val_xa_minus_2x_max_val_ab_plus_bc_plus_ac_l167_167980


namespace calculate_hidden_dots_l167_167875

def sum_faces_of_die : ℕ := 1 + 2 + 3 + 4 + 5 + 6

def number_of_dice : ℕ := 4
def total_sum_of_dots : ℕ := number_of_dice * sum_faces_of_die

def visible_faces : List (ℕ × String) :=
  [(1, "red"), (1, "none"), (2, "none"), (2, "blue"),
   (3, "none"), (4, "none"), (5, "none"), (6, "none")]

def adjust_face_value (value : ℕ) (color : String) : ℕ :=
  match color with
  | "red" => 2 * value
  | "blue" => 2 * value
  | _ => value

def visible_sum : ℕ :=
  visible_faces.foldl (fun acc (face) => acc + adjust_face_value face.1 face.2) 0

theorem calculate_hidden_dots :
  (total_sum_of_dots - visible_sum) = 57 :=
sorry

end calculate_hidden_dots_l167_167875


namespace average_weight_of_boys_l167_167023

theorem average_weight_of_boys 
  (n1 n2 : ℕ) 
  (w1 w2 : ℝ) 
  (h1 : n1 = 22) 
  (h2 : n2 = 8) 
  (h3 : w1 = 50.25) 
  (h4 : w2 = 45.15) : 
  (n1 * w1 + n2 * w2) / (n1 + n2) = 48.89 :=
by
  sorry

end average_weight_of_boys_l167_167023


namespace octagon_mass_l167_167365

theorem octagon_mass :
  let side_length := 1 -- side length of the original square (meters)
  let thickness := 0.3 -- thickness of the sheet (cm)
  let density := 7.8 -- density of steel (g/cm^3)
  let x := 50 * (2 - Real.sqrt 2) -- side length of the triangles (cm)
  let octagon_area := 20000 * (Real.sqrt 2 - 1) -- area of the octagon (cm^2)
  let volume := octagon_area * thickness -- volume of the octagon (cm^3)
  let mass := volume * density / 1000 -- mass of the octagon (kg), converted from g to kg
  mass = 19 :=
by
  sorry

end octagon_mass_l167_167365


namespace sum_digits_18_to_21_l167_167004

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_digits_18_to_21 :
  sum_of_digits 18 + sum_of_digits 19 + sum_of_digits 20 + sum_of_digits 21 = 24 :=
by
  sorry

end sum_digits_18_to_21_l167_167004


namespace second_horse_revolutions_l167_167539

theorem second_horse_revolutions (r1 r2 d1: ℝ) (n1 n2: ℕ) 
  (h1: r1 = 30) (h2: d1 = 36) (h3: r2 = 10) 
  (h4: 2 * Real.pi * r1 * d1 = 2 * Real.pi * r2 * n2) : 
  n2 = 108 := 
by
   sorry

end second_horse_revolutions_l167_167539


namespace not_perfect_square_2023_l167_167666

theorem not_perfect_square_2023 : ¬ (∃ x : ℤ, x^2 = 5^2023) := 
sorry

end not_perfect_square_2023_l167_167666


namespace general_term_sum_first_n_terms_l167_167371

variable {a : ℕ → ℝ} {b : ℕ → ℝ} {S : ℕ → ℝ}
variable (d : ℝ) (h1 : d ≠ 0)
variable (a10 : a 10 = 19)
variable (geo_seq : ∀ {x y z}, x * z = y ^ 2 → x = 1 → y = a 2 → z = a 5)
variable (arith_seq : ∀ n, a n = a 1 + (n - 1) * d)

-- General term of the arithmetic sequence
theorem general_term (a_1 : ℝ) (h1 : a 1 = a_1) : a n = 2 * n - 1 :=
sorry

-- Sum of the first n terms of the sequence b_n
theorem sum_first_n_terms (n : ℕ) : S n = (2 * n - 3) * 2^(n + 1) + 6 :=
sorry

end general_term_sum_first_n_terms_l167_167371


namespace complement_intersection_l167_167897

def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {0, 2}

theorem complement_intersection :
  (U \ A) ∩ B = {0} :=
  by
    sorry

end complement_intersection_l167_167897


namespace probability_9_heads_12_flips_l167_167516

noncomputable def probability_of_9_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_9_heads_12_flips : probability_of_9_heads 12 9 (1 / 2) = 55 / 1024 := by
  sorry

end probability_9_heads_12_flips_l167_167516


namespace Agnes_birth_year_l167_167830

theorem Agnes_birth_year (x y : ℕ) (h1 : 0 ≤ x ∧ x ≤ 9) (h2 : 0 ≤ y ∧ y ≤ 9)
  (h3 : (11 * x + 2 * y + x * y = 92)) : 1948 = 1900 + (10 * x + y) :=
sorry

end Agnes_birth_year_l167_167830


namespace tan_problem_l167_167363

noncomputable def problem : ℝ :=
  (Real.tan (20 * Real.pi / 180) + Real.tan (40 * Real.pi / 180) + Real.tan (120 * Real.pi / 180)) / 
  (Real.tan (20 * Real.pi / 180) * Real.tan (40 * Real.pi / 180))

theorem tan_problem : problem = -Real.sqrt 3 := by
  sorry

end tan_problem_l167_167363


namespace set_intersection_l167_167579

open Set

def U := {x : ℝ | True}
def A := {x : ℝ | x^2 - 2 * x < 0}
def B := {x : ℝ | x - 1 ≥ 0}
def complement (U B : Set ℝ) := {x : ℝ | x ∉ B}
def intersection (A B : Set ℝ) := {x : ℝ | x ∈ A ∧ x ∈ B}

theorem set_intersection :
  intersection A (complement U B) = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end set_intersection_l167_167579


namespace matrix_vector_multiplication_correct_l167_167046

noncomputable def mat : Matrix (Fin 2) (Fin 2) ℤ := ![![3, -2], ![1, 5]]
noncomputable def vec : Fin 2 → ℤ := ![-1, 2]
noncomputable def result : Fin 2 → ℤ := ![-7, 9]

theorem matrix_vector_multiplication_correct :
  (Matrix.mulVec mat vec) = result :=
by
  sorry

end matrix_vector_multiplication_correct_l167_167046


namespace wheat_field_problem_l167_167139

def equations (x F : ℕ) :=
  (6 * x - 300 = F) ∧ (5 * x + 200 = F)

theorem wheat_field_problem :
  ∃ (x F : ℕ), equations x F ∧ x = 500 ∧ F = 2700 :=
by
  sorry

end wheat_field_problem_l167_167139


namespace hyperbola_A_asymptote_l167_167040

-- Define the hyperbola and asymptote conditions
def hyperbola_A (x y : ℝ) : Prop := x^2 - (y^2 / 4) = 1
def asymptote_eq (y x : ℝ) : Prop := y = 2 * x ∨ y = -2 * x

-- Statement of the proof problem in Lean 4
theorem hyperbola_A_asymptote :
  ∀ (x y : ℝ), hyperbola_A x y → asymptote_eq y x :=
sorry

end hyperbola_A_asymptote_l167_167040


namespace total_books_proof_l167_167871

-- Define the number of books Lily finished last month.
def books_last_month : ℕ := 4

-- Define the number of books Lily wants to finish this month.
def books_this_month : ℕ := books_last_month * 2

-- Define the total number of books Lily will finish in two months.
def total_books_two_months : ℕ := books_last_month + books_this_month

-- Theorem to prove the total number of books Lily will finish in two months is 12.
theorem total_books_proof : total_books_two_months = 12 := by
  -- Here would be the proof steps.
  sorry

end total_books_proof_l167_167871


namespace ratio_of_parallel_vectors_l167_167748

theorem ratio_of_parallel_vectors (m n : ℝ) 
  (h1 : ∃ k : ℝ, (m, 1, 3) = (k * 2, k * n, k)) : (m / n) = 18 :=
by
  sorry

end ratio_of_parallel_vectors_l167_167748


namespace range_of_m_l167_167251

theorem range_of_m (m : ℝ) (x y : ℝ)
  (h1 : x + y - 3 * m = 0)
  (h2 : 2 * x - y + 2 * m - 1 = 0)
  (h3 : x > 0)
  (h4 : y < 0) : 
  -1 < m ∧ m < 1/8 := 
sorry

end range_of_m_l167_167251


namespace max_rectangle_area_l167_167089

theorem max_rectangle_area (P : ℝ) (hP : P = 60) : ∃ A : ℝ, A = 225 :=
by 
  sorry

end max_rectangle_area_l167_167089


namespace no_integer_solutions_l167_167531

theorem no_integer_solutions (x y z : ℤ) (h : 2 * x^4 + 2 * x^2 * y^2 + y^4 = z^2) (hx : x ≠ 0) : false :=
sorry

end no_integer_solutions_l167_167531


namespace maximum_value_a_over_b_plus_c_l167_167779

open Real

noncomputable def max_frac_a_over_b_plus_c (a b c : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c) (h_eq: a * (a + b + c) = b * c) : ℝ :=
  if (b = c) then (Real.sqrt 2 - 1) / 2 else -1 -- placeholder for irrelevant case

theorem maximum_value_a_over_b_plus_c 
  (a b c : ℝ) 
  (h_pos: 0 < a ∧ 0 < b ∧ 0 < c)
  (h_eq: a * (a + b + c) = b * c) :
  max_frac_a_over_b_plus_c a b c h_pos h_eq = (Real.sqrt 2 - 1) / 2 :=
sorry

end maximum_value_a_over_b_plus_c_l167_167779


namespace emery_reading_days_l167_167560

theorem emery_reading_days (S E : ℕ) (h1 : E = S / 5) (h2 : (E + S) / 2 = 60) : E = 20 := by
  sorry

end emery_reading_days_l167_167560


namespace total_questions_to_review_is_1750_l167_167038

-- Define the relevant conditions
def num_classes := 5
def students_per_class := 35
def questions_per_exam := 10

-- The total number of questions to be reviewed by Professor Oscar
def total_questions : Nat := num_classes * students_per_class * questions_per_exam

-- The theorem stating the equivalent proof problem
theorem total_questions_to_review_is_1750 : total_questions = 1750 := by
  -- proof steps are skipped here 
  sorry

end total_questions_to_review_is_1750_l167_167038


namespace polynomial_condition_satisfied_l167_167361

-- Definitions as per conditions:
def p (x : ℝ) : ℝ := x^2 + 1

-- Conditions:
axiom cond1 : p 3 = 10
axiom cond2 : ∀ (x y : ℝ), p x * p y = p x + p y + p (x * y) - 2

-- Theorem to prove:
theorem polynomial_condition_satisfied : (p 3 = 10) ∧ (∀ (x y : ℝ), p x * p y = p x + p y + p (x * y) - 2) :=
by
  apply And.intro cond1
  apply cond2

end polynomial_condition_satisfied_l167_167361


namespace multiply_polynomials_l167_167628

def polynomial_multiplication (x : ℝ) : Prop :=
  (x^4 + 24*x^2 + 576) * (x^2 - 24) = x^6 - 13824

theorem multiply_polynomials (x : ℝ) : polynomial_multiplication x :=
by
  sorry

end multiply_polynomials_l167_167628


namespace inverse_function_passes_through_point_l167_167907

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 1)

theorem inverse_function_passes_through_point {a : ℝ} (h1 : 0 < a) (h2 : a ≠ 1) (h3 : f a (-1) = 1) :
  f a⁻¹ 1 = -1 :=
sorry

end inverse_function_passes_through_point_l167_167907


namespace lines_intersect_l167_167862

theorem lines_intersect :
  ∃ x y : ℚ, 
  8 * x - 5 * y = 40 ∧ 
  6 * x - y = -5 ∧ 
  x = 15 / 38 ∧ 
  y = 140 / 19 :=
by { sorry }

end lines_intersect_l167_167862


namespace fraction_relationships_l167_167906

variables (a b c d : ℚ)

theorem fraction_relationships (h1 : a / b = 3) (h2 : b / c = 2 / 3) (h3 : c / d = 5) :
  d / a = 1 / 10 :=
by
  sorry

end fraction_relationships_l167_167906


namespace second_smallest_palindromic_prime_l167_167437

-- Three digit number definition
def three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Palindromic number definition
def is_palindromic (n : ℕ) : Prop := 
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let ones := n % 10
  hundreds = ones 

-- Prime number definition
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Second-smallest three-digit palindromic prime
theorem second_smallest_palindromic_prime :
  ∃ n : ℕ, three_digit_number n ∧ is_palindromic n ∧ is_prime n ∧ 
  ∃ m : ℕ, three_digit_number m ∧ is_palindromic m ∧ is_prime m ∧ m > 101 ∧ m < n ∧ 
  n = 131 := 
by
  sorry

end second_smallest_palindromic_prime_l167_167437


namespace remainder_of_sum_division_l167_167138

theorem remainder_of_sum_division (x y : ℕ) (k m : ℕ) 
  (hx : x = 90 * k + 75) (hy : y = 120 * m + 115) :
  (x + y) % 30 = 10 :=
by sorry

end remainder_of_sum_division_l167_167138


namespace scramble_language_words_count_l167_167948

theorem scramble_language_words_count :
  let total_words (n : ℕ) := 25 ^ n
  let words_without_B (n : ℕ) := 24 ^ n
  let words_with_B (n : ℕ) := total_words n - words_without_B n
  words_with_B 1 + words_with_B 2 + words_with_B 3 + words_with_B 4 + words_with_B 5 = 1863701 :=
by
  sorry

end scramble_language_words_count_l167_167948


namespace sales_volume_function_max_profit_min_boxes_for_2000_profit_l167_167988

-- Definitions and conditions
def cost_per_box : ℝ := 20
def min_selling_price : ℝ := 25
def init_boxes_sold : ℝ := 250
def price_increase_effect : ℝ := 10
def max_selling_price : ℝ := 38

-- Question 1: Find functional relationship between daily sales volume y and selling price per box x
theorem sales_volume_function (x : ℝ) (hx : x ≥ min_selling_price) :
  ∃ y, y = -10 * x + 500 := by
  sorry

-- Question 2: Find the price per box to maximize daily sales profit and the maximum profit
theorem max_profit (x : ℝ) (hx : x = 35) :
  ∃ P, P = -10 * (x-20) * (x) + 500 * (x-20) := by
  sorry

-- Question 3: Determine min boxes sold to make at least 2000 yuan given price does not exceed 38 yuan
theorem min_boxes_for_2000_profit (x : ℝ) (hx : min_selling_price ≤ x ∧ x ≤ max_selling_price ∧ 
                             -10 * (x-20) * (-10 * x + 500) ≥ 2000) :
  ∃ y, y = -10 * x + 500 ∧ y ≥ 120 := by
  sorry

end sales_volume_function_max_profit_min_boxes_for_2000_profit_l167_167988


namespace line_equation_l167_167021

theorem line_equation (m n : ℝ) (p : ℝ) (h : p = 3) :
  ∃ b : ℝ, ∀ x y : ℝ, (y = n + 21) → (x = m + 3) → y = 7 * x + b ∧ b = n - 7 * m :=
by sorry

end line_equation_l167_167021


namespace quadratic_root_is_zero_then_m_neg_one_l167_167245

theorem quadratic_root_is_zero_then_m_neg_one (m : ℝ) (h_eq : (m-1) * 0^2 + 2 * 0 + m^2 - 1 = 0) : m = -1 := by
  sorry

end quadratic_root_is_zero_then_m_neg_one_l167_167245


namespace circle_radius_range_l167_167771

theorem circle_radius_range (r : ℝ) : 
  (∃ P₁ P₂ : ℝ × ℝ, (P₁.2 = 1 ∨ P₁.2 = -1) ∧ (P₂.2 = 1 ∨ P₂.2 = -1) ∧ 
  (P₁.1 - 3) ^ 2 + (P₁.2 + 5) ^ 2 = r^2 ∧ (P₂.1 - 3) ^ 2 + (P₂.2 + 5) ^ 2 = r^2) → (4 < r ∧ r < 6) :=
by
  sorry

end circle_radius_range_l167_167771


namespace volume_after_increase_l167_167214

variable (l w h : ℕ)
variable (V S E : ℕ)

noncomputable def original_volume : ℕ := l * w * h
noncomputable def surface_sum : ℕ := (l * w) + (w * h) + (h * l)
noncomputable def edge_sum : ℕ := l + w + h

theorem volume_after_increase (h_volume : original_volume l w h = 5400)
  (h_surface : surface_sum l w h = 1176)
  (h_edge : edge_sum l w h = 60) : 
  (l + 1) * (w + 1) * (h + 1) = 6637 := sorry

end volume_after_increase_l167_167214


namespace range_of_a_l167_167379

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 < x → (a / x - 4 / x^2 < 1)) → a < 4 := 
by
  sorry

end range_of_a_l167_167379


namespace problem1_problem2_l167_167411

open Real

variables {α β γ : ℝ}

theorem problem1 (α β : ℝ) :
  abs (cos (α + β)) ≤ abs (cos α) + abs (sin β) ∧
  abs (sin (α + β)) ≤ abs (cos α) + abs (cos β) :=
sorry

theorem problem2 (h : α + β + γ = 0) :
  abs (cos α) + abs (cos β) + abs (cos γ) ≥ 1 :=
sorry

end problem1_problem2_l167_167411


namespace probability_heads_in_nine_of_twelve_flips_l167_167479

theorem probability_heads_in_nine_of_twelve_flips : 
  (nat.choose 12 9) / (2^12) = 55 / 1024 := 
by
  sorry

end probability_heads_in_nine_of_twelve_flips_l167_167479


namespace probability_of_at_least_one_white_is_D_l167_167566

-- Define the bag and the events

noncomputable def bag := ['red, 'red, 'red, 'white, 'white]

-- Define the event of drawing 3 balls
def draw_3_balls (s : List (String × ℕ)) : Set (List String) := 
  {l | l.length = 3 ∧ ∀ x, x ∈ l → (s.count x) ≥ (l.count x)}

-- Define the total possible combinations
def total_combinations := @Set.univ (List String)

-- Define the event that we do not draw 3 red balls
def not_all_red : Set (List String) := {l | ¬(l.count 'red = 3)}

-- Probability that at least one of 3 drawn balls is white
def probability_at_least_one_white : ℝ := 
  1 - (draw_3_balls total_combinations ∩ not_all_red).size.val.fst / total_combinations.size.val.fst

theorem probability_of_at_least_one_white_is_D :
  probability_at_least_one_white = sorry := sorry

end probability_of_at_least_one_white_is_D_l167_167566


namespace probability_heads_9_of_12_l167_167511

theorem probability_heads_9_of_12 :
  let total_outcomes := 2^12 in
  let favorable_outcomes := Nat.choose 12 9 in
  let probability := favorable_outcomes / total_outcomes.toRat in
  probability = 220 / 4096 := 
by
  sorry

end probability_heads_9_of_12_l167_167511


namespace solve_quintic_equation_l167_167641

theorem solve_quintic_equation :
  {x : ℝ | x * (x - 3)^2 * (5 + x) * (x^2 - 1) = 0} = {0, 3, -5, 1, -1} :=
by
  sorry

end solve_quintic_equation_l167_167641


namespace probability_of_9_heads_in_12_flips_l167_167506

theorem probability_of_9_heads_in_12_flips :
  (nat.choose 12 9) * (1 / (2 : ℝ)^12) = 55 / 1024 := 
by
  sorry

end probability_of_9_heads_in_12_flips_l167_167506


namespace sum_of_cubes_of_consecutive_integers_l167_167808

-- Define the given condition
def sum_of_squares_of_consecutive_integers (n : ℕ) : Prop :=
  (n - 1)^2 + n^2 + (n + 1)^2 = 7805

-- Define the statement we want to prove
theorem sum_of_cubes_of_consecutive_integers (n : ℕ) (h : sum_of_squares_of_consecutive_integers n) : 
  (n - 1)^3 + n^3 + (n + 1)^3 = 398259 :=
by
  sorry

end sum_of_cubes_of_consecutive_integers_l167_167808


namespace five_ones_make_100_l167_167973

noncomputable def concatenate (a b c : Nat) : Nat :=
  a * 100 + b * 10 + c

theorem five_ones_make_100 :
  let one := 1
  let x := concatenate one one one -- 111
  let y := concatenate one one 0 / 10 -- 11, concatenation of 1 and 1 treated as 110, divided by 10
  x - y = 100 :=
by
  sorry

end five_ones_make_100_l167_167973


namespace speed_ratio_l167_167822

theorem speed_ratio (L tA tB : ℝ) (R : ℝ) (h1: A_speed = R * B_speed) 
  (h2: head_start = 0.35 * L) (h3: finish_margin = 0.25 * L)
  (h4: A_distance = L + head_start) (h5: B_distance = L)
  (h6: A_finish = A_distance / A_speed)
  (h7: B_finish = B_distance / B_speed)
  (h8: B_finish_time = A_finish + finish_margin / B_speed)
  : R = 1.08 :=
by
  sorry

end speed_ratio_l167_167822


namespace tan_angle_PAB_correct_l167_167631

noncomputable def tan_angle_PAB (AB BC CA : ℝ) (P inside ABC : Prop) (PAB_angle_eq_PBC_angle_eq_PCA_angle : Prop) : ℝ :=
  180 / 329

theorem tan_angle_PAB_correct :
  ∀ (AB BC CA : ℝ)
    (P_inside_ABC : Prop)
    (PAB_angle_eq_PBC_angle_eq_PCA_angle : Prop),
    AB = 12 → BC = 15 → CA = 17 →
    (tan_angle_PAB AB BC CA P_inside_ABC PAB_angle_eq_PBC_angle_eq_PCA_angle) = 180 / 329 :=
by
  intros
  sorry

end tan_angle_PAB_correct_l167_167631


namespace probability_of_9_heads_in_12_flips_l167_167487

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| _, 0       => 1
| 0, _       => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^12

-- Define the number of ways to get exactly 9 heads in 12 flips
def favorable_outcomes : ℕ := binom 12 9

-- Define the probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 220/4096
theorem probability_of_9_heads_in_12_flips : probability = 220 / 4096 := by
  sorry

end probability_of_9_heads_in_12_flips_l167_167487


namespace probability_heads_in_nine_of_twelve_flips_l167_167478

theorem probability_heads_in_nine_of_twelve_flips : 
  (nat.choose 12 9) / (2^12) = 55 / 1024 := 
by
  sorry

end probability_heads_in_nine_of_twelve_flips_l167_167478


namespace triangle_third_side_l167_167812

theorem triangle_third_side (AB AC AD : ℝ) (hAB : AB = 25) (hAC : AC = 30) (hAD : AD = 24) :
  ∃ BC : ℝ, (BC = 25 ∨ BC = 11) :=
by
  sorry

end triangle_third_side_l167_167812


namespace donny_spending_l167_167721

theorem donny_spending :
  (15 + 28 + 13) / 2 = 28 :=
by
  sorry

end donny_spending_l167_167721


namespace f_2_eq_1_l167_167253

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 - b * x + 1

theorem f_2_eq_1 (a b : ℝ) (h : f a b (-2) = 1) : f a b 2 = 1 :=
by {
  sorry 
}

end f_2_eq_1_l167_167253


namespace people_came_in_first_hour_l167_167684
-- Import the entirety of the necessary library

-- Lean 4 statement for the given problem
theorem people_came_in_first_hour (X : ℕ) (net_change_first_hour : ℕ) (net_change_second_hour : ℕ) (people_after_2_hours : ℕ) : 
    (net_change_first_hour = X - 27) → 
    (net_change_second_hour = 18 - 9) →
    (people_after_2_hours = 76) → 
    (X - 27 + 9 = 76) → 
    X = 94 :=
by 
    intros h1 h2 h3 h4 
    sorry -- Proof is not required by instructions

end people_came_in_first_hour_l167_167684


namespace smallest_number_exceeding_triangle_perimeter_l167_167309

theorem smallest_number_exceeding_triangle_perimeter (a b : ℕ) (a_eq_7 : a = 7) (b_eq_21 : b = 21) :
  ∃ P : ℕ, (∀ c : ℝ, 14 < c ∧ c < 28 → a + b + c < P) ∧ P = 56 := by
  sorry

end smallest_number_exceeding_triangle_perimeter_l167_167309


namespace tan_half_theta_l167_167878

theorem tan_half_theta (θ : ℝ) (h1 : Real.sin θ = -3 / 5) (h2 : 3 * Real.pi < θ ∧ θ < 7 / 2 * Real.pi) :
  Real.tan (θ / 2) = -3 :=
sorry

end tan_half_theta_l167_167878


namespace runway_trip_time_l167_167646

-- Define the conditions
def num_models := 6
def num_bathing_suit_outfits := 2
def num_evening_wear_outfits := 3
def total_time_minutes := 60

-- Calculate the total number of outfits per model
def total_outfits_per_model := num_bathing_suit_outfits + num_evening_wear_outfits

-- Calculate the total number of runway trips
def total_runway_trips := num_models * total_outfits_per_model

-- State the goal: Time per runway trip
def time_per_runway_trip := total_time_minutes / total_runway_trips

theorem runway_trip_time : time_per_runway_trip = 2 := by
  sorry

end runway_trip_time_l167_167646


namespace probability_exactly_9_heads_l167_167495

theorem probability_exactly_9_heads :
  (Nat.choose 12 9) / (2 ^ 12 : ℚ) = 220 / 4096 := by
  sorry

end probability_exactly_9_heads_l167_167495


namespace towel_bleach_volume_decrease_l167_167998

theorem towel_bleach_volume_decrease :
  ∀ (L B T : ℝ) (L' B' T' : ℝ),
  (L' = L * 0.75) →
  (B' = B * 0.70) →
  (T' = T * 0.90) →
  (L * B * T = 1000000) →
  ((L * B * T - L' * B' * T') / (L * B * T) * 100) = 52.75 :=
by
  intros L B T L' B' T' hL' hB' hT' hV
  sorry

end towel_bleach_volume_decrease_l167_167998


namespace product_form_l167_167265

theorem product_form (b a : ℤ) :
  (10 * b + a) * (10 * b + 10 - a) = 100 * b * (b + 1) + a * (10 - a) := 
sorry

end product_form_l167_167265


namespace snake_body_length_l167_167330

theorem snake_body_length (l h : ℝ) (h_head: h = l / 10) (h_length: l = 10) : l - h = 9 := 
by 
  rw [h_length, h_head] 
  norm_num
  sorry

end snake_body_length_l167_167330


namespace max_area_of_rectangular_pen_l167_167072

-- Define the perimeter and derive the formula for the area
def perimeter := 60
def half_perimeter := perimeter / 2
def area (x : ℝ) := x * (half_perimeter - x)

-- Statement of the problem: prove the maximum area is 225 square feet
theorem max_area_of_rectangular_pen : ∃ x : ℝ, 0 ≤ x ∧ x ≤ half_perimeter ∧ area x = 225 := 
sorry

end max_area_of_rectangular_pen_l167_167072


namespace sum_of_coefficients_l167_167656

def polynomial (x y : ℕ) : ℕ := (x^2 - 3*x*y + y^2)^8

theorem sum_of_coefficients : polynomial 1 1 = 1 :=
sorry

end sum_of_coefficients_l167_167656


namespace triangle_problem_l167_167391

open Real

theorem triangle_problem (a b S : ℝ) (A B : ℝ) (hA_cos : cos A = (sqrt 6) / 3) (hA_val : a = 3) (hB_val : B = A + π / 2):
  b = 3 * sqrt 2 ∧
  S = (3 * sqrt 2) / 2 :=
by
  sorry

end triangle_problem_l167_167391


namespace committee_probability_l167_167644

theorem committee_probability :
  let total_members := 24
  let boys := 12
  let girls := 12
  let committee_size := 5
  let total_committees := Nat.choose total_members committee_size
  let all_boys_girls_committees := 2 * Nat.choose boys committee_size
  let mixed_committees := total_committees - all_boys_girls_committees
  let probability := (mixed_committees : ℚ) / total_committees
  probability = 455 / 472 :=
by
  sorry

end committee_probability_l167_167644


namespace find_m_l167_167134

/-
Define the ellipse equation
-/
def ellipse_eqn (x y : ℝ) : Prop :=
  (x^2 / 9) + (y^2) = 1

/-
Define the region R
-/
def region_R (x y : ℝ) : Prop :=
  (x ≥ 0) ∧ (y ≥ 0) ∧ (2*y = x) ∧ ellipse_eqn x y

/-
Define the region R'
-/
def region_R' (x y m : ℝ) : Prop :=
  (x ≥ 0) ∧ (y ≥ 0) ∧ (y = m*x) ∧ ellipse_eqn x y

/-
The statement we want to prove
-/
theorem find_m (m : ℝ) : (∃ (x y : ℝ), region_R x y) ∧ (∃ (x y : ℝ), region_R' x y m) →
(m = (2 : ℝ) / 9) := 
sorry

end find_m_l167_167134


namespace daily_profit_9080_l167_167211

theorem daily_profit_9080 (num_employees : Nat) (shirts_per_employee_per_day : Nat) (hours_per_shift : Nat) (wage_per_hour : Nat) (bonus_per_shirt : Nat) (shirt_sale_price : Nat) (nonemployee_expenses : Nat) :
  num_employees = 20 →
  shirts_per_employee_per_day = 20 →
  hours_per_shift = 8 →
  wage_per_hour = 12 →
  bonus_per_shirt = 5 →
  shirt_sale_price = 35 →
  nonemployee_expenses = 1000 →
  (num_employees * shirts_per_employee_per_day * shirt_sale_price) - ((num_employees * (hours_per_shift * wage_per_hour + shirts_per_employee_per_day * bonus_per_shirt)) + nonemployee_expenses) = 9080 := 
by
  intros
  sorry

end daily_profit_9080_l167_167211


namespace number_of_boxes_l167_167258

def magazines : ℕ := 63
def magazines_per_box : ℕ := 9

theorem number_of_boxes : magazines / magazines_per_box = 7 :=
by 
  sorry

end number_of_boxes_l167_167258


namespace skittles_taken_away_l167_167043

def initial_skittles : ℕ := 25
def remaining_skittles : ℕ := 18

theorem skittles_taken_away (C_initial C_remaining : ℕ) (h1 : C_initial = 25) (h2 : C_remaining = 18) :
  (C_initial - C_remaining = 7) :=
by
  sorry

end skittles_taken_away_l167_167043


namespace vivian_mail_in_august_l167_167015

-- Conditions
def april_mail : ℕ := 5
def may_mail : ℕ := 2 * april_mail
def june_mail : ℕ := 2 * may_mail
def july_mail : ℕ := 2 * june_mail

-- Question: Prove that Vivian will send 80 pieces of mail in August.
theorem vivian_mail_in_august : 2 * july_mail = 80 :=
by
  -- Sorry to skip the proof
  sorry

end vivian_mail_in_august_l167_167015


namespace regular_price_of_Pony_jeans_l167_167876

theorem regular_price_of_Pony_jeans 
(Fox_price : ℝ) 
(Pony_price : ℝ) 
(savings : ℝ) 
(Fox_discount_rate : ℝ) 
(Pony_discount_rate : ℝ)
(h1 : Fox_price = 15)
(h2 : savings = 8.91)
(h3 : Fox_discount_rate + Pony_discount_rate = 0.22)
(h4 : Pony_discount_rate = 0.10999999999999996) : Pony_price = 18 := 
sorry

end regular_price_of_Pony_jeans_l167_167876


namespace numbers_unchanged_by_powers_of_n_l167_167818

-- Definitions and conditions
def unchanged_when_raised (x : ℂ) (n : ℕ) : Prop :=
  x^n = x

def modulus_one (z : ℂ) : Prop :=
  Complex.abs z = 1

-- Proof statements
theorem numbers_unchanged_by_powers_of_n :
  (∀ x : ℂ, (∀ n : ℕ, n > 0 → unchanged_when_raised x n → x = 0 ∨ x = 1)) ∧
  (∀ z : ℂ, modulus_one z → (∀ n : ℕ, n > 0 → Complex.abs (z^n) = 1)) :=
by
  sorry

end numbers_unchanged_by_powers_of_n_l167_167818


namespace gcd_60_90_150_l167_167962

theorem gcd_60_90_150 : Nat.gcd (Nat.gcd 60 90) 150 = 30 := 
by
  sorry

end gcd_60_90_150_l167_167962


namespace evaluate_expression_l167_167858

theorem evaluate_expression : 
  (10^8 / (2.5 * 10^5) * 3) = 1200 :=
by
  sorry

end evaluate_expression_l167_167858


namespace combined_mean_of_scores_l167_167627

theorem combined_mean_of_scores (f s : ℕ) (mean_1 mean_2 : ℕ) (ratio : f = (2 * s) / 3) 
  (hmean1 : mean_1 = 90) (hmean2 : mean_2 = 75) :
  (135 * s) / ((2 * s) / 3 + s) = 81 := 
by
  sorry

end combined_mean_of_scores_l167_167627


namespace sum_2010_3_array_remainder_l167_167677
 
theorem sum_2010_3_array_remainder :
  let p := 2010
  let q := 3
  let sum_array := (∑' r : ℕ, ∑' c : ℕ, (1 / (2 * p)^r) * (1 / q^c))
  let fraction := 4020 / 4019 * 3 / 2
  let (m, n) := (6030, 4019) in
  m + n = 10049 → 10049 % 2010 = 1009 :=
by
  let sum_geom_r := (∑' r : ℕ, 1 / (2 * 2010)^r)
  let sum_geom_c := (∑' c : ℕ, 1 / 3^c)
  have sum_geom_r_correct : sum_geom_r = 1 / (1 - 1 / (2 * 2010)) := sorry
  have sum_geom_c_correct : sum_geom_c = 1 / (1 - 1 / 3) := sorry
  have sum_array_value : (∑' r, ∑' c, 1 / (2 * 2010)^r * 1 / 3^c) = fraction := sorry
  have fraction_eq : fraction = 6030 / 4019 := sorry
  have m_prime : Nat.gcd 6030 4019 = 1 := by norm_num
  have mod_eq : 10049 % 2010 = 1009 := by norm_num
  exact mod_eq

end sum_2010_3_array_remainder_l167_167677


namespace number_of_pages_l167_167182

-- Define the conditions
def rate_of_printer_A (P : ℕ) : ℕ := P / 60
def rate_of_printer_B (P : ℕ) : ℕ := (P / 60) + 6

-- Define the combined rate condition
def combined_rate (P : ℕ) (R_A R_B : ℕ) : Prop := (R_A + R_B) = P / 24

-- The main theorem to prove
theorem number_of_pages :
  ∃ (P : ℕ), combined_rate P (rate_of_printer_A P) (rate_of_printer_B P) ∧ P = 720 := by
  sorry

end number_of_pages_l167_167182


namespace second_tap_fills_in_15_hours_l167_167013

theorem second_tap_fills_in_15_hours 
  (r1 r3 : ℝ) 
  (x : ℝ) 
  (H1 : r1 = 1 / 10) 
  (H2 : r3 = 1 / 6) 
  (H3 : r1 + 1 / x + r3 = 1 / 3) : 
  x = 15 :=
sorry

end second_tap_fills_in_15_hours_l167_167013


namespace S_calculation_T_calculation_l167_167347

def S (a b : ℕ) : ℕ := 4 * a + 6 * b
def T (a b : ℕ) : ℕ := 5 * a + 3 * b

theorem S_calculation : S 6 3 = 42 :=
by sorry

theorem T_calculation : T 6 3 = 39 :=
by sorry

end S_calculation_T_calculation_l167_167347


namespace at_least_one_two_prob_l167_167303

-- Definitions and conditions corresponding to the problem
def total_outcomes (n : ℕ) : ℕ := n * n
def no_twos_outcomes (n : ℕ) : ℕ := (n - 1) * (n - 1)

-- The probability calculation
def probability_at_least_one_two (n : ℕ) : ℚ := 
  let tot_outs := total_outcomes n
  let no_twos := no_twos_outcomes n
  (tot_outs - no_twos : ℚ) / tot_outs

-- Our main theorem to be proved
theorem at_least_one_two_prob : 
  probability_at_least_one_two 6 = 11 / 36 := 
by
  sorry

end at_least_one_two_prob_l167_167303


namespace horizontal_distance_P_Q_l167_167318

-- Definitions for the given conditions
def curve (x : ℝ) : ℝ := x^2 + 2 * x - 3

-- Define the points P and Q on the curve
def P_x : Set ℝ := {x | curve x = 8}
def Q_x : Set ℝ := {x | curve x = -1}

-- State the theorem to prove horizontal distance is 3sqrt3
theorem horizontal_distance_P_Q : 
  ∃ (x₁ x₂ : ℝ), x₁ ∈ P_x ∧ x₂ ∈ Q_x ∧ |x₁ - x₂| = 3 * Real.sqrt 3 :=
sorry

end horizontal_distance_P_Q_l167_167318


namespace b_minus_a_eq_two_l167_167951

theorem b_minus_a_eq_two (a b : ℤ) (h1 : b = 7) (h2 : a * b = 2 * (a + b) + 11) : b - a = 2 :=
by
  sorry

end b_minus_a_eq_two_l167_167951


namespace find_number_l167_167294

-- Given conditions:
def sum_and_square (n : ℕ) : Prop := n^2 + n = 252
def is_factor (n d : ℕ) : Prop := d % n = 0

-- Equivalent proof problem statement
theorem find_number : ∃ n : ℕ, sum_and_square n ∧ is_factor n 180 ∧ n > 0 ∧ n = 14 :=
by
  sorry

end find_number_l167_167294


namespace solve_for_m_l167_167179

theorem solve_for_m :
  (∀ (m : ℕ), 
   ((1:ℚ)^(m+1) / 5^(m+1) * 1^18 / 4^18 = 1 / (2 * 10^35)) → m = 34) := 
by apply sorry

end solve_for_m_l167_167179


namespace difference_between_relations_l167_167176

-- Definitions based on conditions
def functional_relationship 
  (f : α → β) (x : α) (y : β) : Prop :=
  f x = y

def correlation_relationship (X Y : Type) : Prop :=
  ∃ (X_rand : X → ℝ) (Y_rand : Y → ℝ), 
    ∀ (x : X), ∃ (y : Y), X_rand x ≠ Y_rand y

-- Theorem stating the problem
theorem difference_between_relations :
  (∀ (f : α → β) (x : α) (y : β), functional_relationship f x y) ∧ 
  (∀ (X Y : Type), correlation_relationship X Y) :=
sorry

end difference_between_relations_l167_167176


namespace find_number_l167_167981

theorem find_number (number : ℝ) (h : 0.75 / 100 * number = 0.06) : number = 8 := 
by
  sorry

end find_number_l167_167981


namespace probability_heads_9_of_12_is_correct_l167_167453

-- Defining the probability of getting exactly 9 heads in 12 flips of a fair coin
noncomputable def probability_heads_9_of_12 := (55 : ℚ) / 1024

-- Theorem statement
theorem probability_heads_9_of_12_is_correct :
  probability_heads_9_of_12 = (55 : ℚ) / 1024 := 
  by sorry

end probability_heads_9_of_12_is_correct_l167_167453


namespace max_abs_sum_l167_167267

theorem max_abs_sum (a b c : ℝ) (h : ∀ x, -1 ≤ x ∧ x ≤ 1 → |a * x^2 + b * x + c| ≤ 1) :
  |a| + |b| + |c| ≤ 3 :=
sorry

end max_abs_sum_l167_167267


namespace leftover_coverage_l167_167349

theorem leftover_coverage 
  (coverage_per_bag : ℕ)
  (length : ℕ)
  (width : ℕ)
  (num_bags : ℕ) :
  coverage_per_bag = 250 →
  length = 22 →
  width = 36 →
  num_bags = 4 →
  let lawn_area := length * width,
      total_coverage := coverage_per_bag * num_bags,
      leftover_coverage := total_coverage - lawn_area
  in leftover_coverage = 208 := 
by
  intros h1 h2 h3 h4
  let lawn_area := 22 * 36
  let total_coverage := 250 * 4
  let leftover_coverage := total_coverage - lawn_area
  have : lawn_area = 792 := by norm_num
  have : total_coverage = 1000 := by norm_num
  have : leftover_coverage = total_coverage - lawn_area := rfl
  show leftover_coverage = 208, from by
    rw [this, this, this]
    norm_num
  sorry

end leftover_coverage_l167_167349


namespace gina_total_cost_l167_167733

-- Define the constants based on the conditions
def total_credits : ℕ := 18
def reg_credits : ℕ := 12
def reg_cost_per_credit : ℕ := 450
def lab_credits : ℕ := 6
def lab_cost_per_credit : ℕ := 550
def num_textbooks : ℕ := 3
def textbook_cost : ℕ := 150
def num_online_resources : ℕ := 4
def online_resource_cost : ℕ := 95
def facilities_fee : ℕ := 200
def lab_fee_per_credit : ℕ := 75

-- Calculating the total cost
noncomputable def total_cost : ℕ :=
  (reg_credits * reg_cost_per_credit) +
  (lab_credits * lab_cost_per_credit) +
  (num_textbooks * textbook_cost) +
  (num_online_resources * online_resource_cost) +
  facilities_fee +
  (lab_credits * lab_fee_per_credit)

-- The proof problem to show that the total cost is 10180
theorem gina_total_cost : total_cost = 10180 := by
  sorry

end gina_total_cost_l167_167733


namespace value_of_x_plus_y_l167_167797

theorem value_of_x_plus_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x / 3 = y^2) (h2 : x / 9 = 9 * y) : 
  x + y = 2214 :=
sorry

end value_of_x_plus_y_l167_167797


namespace inverse_proportion_value_of_m_l167_167757

theorem inverse_proportion_value_of_m (m : ℤ) (x : ℝ) (y : ℝ) : 
  y = (m - 2) * x ^ (m^2 - 5) → (m = -2) := 
by
  sorry

end inverse_proportion_value_of_m_l167_167757


namespace eighth_graders_taller_rows_remain_ordered_l167_167833

-- Part (a)

theorem eighth_graders_taller {n : ℕ} (h8 : Fin n → ℚ) (h7 : Fin n → ℚ)
  (ordered8 : ∀ i j : Fin n, i ≤ j → h8 i ≤ h8 j)
  (ordered7 : ∀ i j : Fin n, i ≤ j → h7 i ≤ h7 j)
  (initial_condition : ∀ i : Fin n, h8 i > h7 i) :
  ∀ i : Fin n, h8 i > h7 i :=
sorry

-- Part (b)

theorem rows_remain_ordered {m n : ℕ} (h : Fin m → Fin n → ℚ)
  (row_ordered : ∀ i : Fin m, ∀ j k : Fin n, j ≤ k → h i j ≤ h i k)
  (column_ordered_after : ∀ j : Fin n, ∀ i k : Fin m, i ≤ k → h i j ≤ h k j) :
  ∀ i : Fin m, ∀ j k : Fin n, j ≤ k → h i j ≤ h i k :=
sorry

end eighth_graders_taller_rows_remain_ordered_l167_167833


namespace number_of_street_trees_l167_167311

-- Definitions from conditions
def road_length : ℕ := 1500
def interval_distance : ℕ := 25

-- The statement to prove
theorem number_of_street_trees : (road_length / interval_distance) + 1 = 61 := 
by
  unfold road_length
  unfold interval_distance
  sorry

end number_of_street_trees_l167_167311


namespace probability_three_draws_exceed_ten_l167_167319

open Finset

-- Definitions based on problem conditions
def chips := range 1 9 -- Chips numbered from 1 to 8

/-- Defining pairs that sum to 8, exclude (4, 4) because chips are unique -/
def valid_pairs : Finset (ℕ × ℕ) := 
  {(1, 7), (2, 6), (3, 5), (5, 3), (6, 2), (7, 1)}.toFinset

/-- Calculate the total probability of valid pairs which sum to 8 -/
def valid_pairs_probability : ℚ := valid_pairs.card / (8 * 7)

-- Defining the probability of third chip drawn exceeding 10
def third_chip_probability : ℚ := 4 / 6

-- Combined final probability
def final_probability : ℚ := valid_pairs_probability * third_chip_probability

theorem probability_three_draws_exceed_ten : 
  final_probability = 1 / 14 :=
by sorry

end probability_three_draws_exceed_ten_l167_167319


namespace probability_of_9_heads_in_12_l167_167474

def coin_flip_probability_9_heads_in_12_flips : Prop :=
  let total_outcomes := 2^12
  let success_outcomes := Nat.choose 12 9
  success_outcomes / total_outcomes = 220 / 4096

theorem probability_of_9_heads_in_12 : coin_flip_probability_9_heads_in_12_flips :=
  sorry

end probability_of_9_heads_in_12_l167_167474


namespace lowest_score_dropped_l167_167669

-- Conditions definitions
def total_sum_of_scores (A B C D : ℕ) := A + B + C + D = 240
def total_sum_after_dropping_lowest (A B C : ℕ) := A + B + C = 195

-- Theorem statement
theorem lowest_score_dropped (A B C D : ℕ) (h1 : total_sum_of_scores A B C D) (h2 : total_sum_after_dropping_lowest A B C) : D = 45 := 
sorry

end lowest_score_dropped_l167_167669


namespace inequality_proof_l167_167978

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  a + b + c + d + 8 / (a*b + b*c + c*d + d*a) ≥ 6 := 
by
  sorry

end inequality_proof_l167_167978


namespace simplify_fraction_l167_167146

theorem simplify_fraction :
  (75 : ℚ) / (225 : ℚ) = 1 / 3 := by
  sorry

end simplify_fraction_l167_167146


namespace integer_values_of_b_l167_167058

theorem integer_values_of_b (b : ℤ) : 
  (∃ x y : ℤ, x^2 + b * x + 12 * b = 0 ∧ y^2 + b * y + 12 * b = 0 ∧ x ≠ y) → 
  ∃ S : finset ℤ, S.card = 8 ∧ ∀ c ∈ S, ∃ x : ℤ, x^2 + c * x + 12 * c = 0 :=
sorry

end integer_values_of_b_l167_167058


namespace range_of_m_given_q_range_of_m_given_p_or_q_and_not_p_and_q_l167_167739

def quadratic_has_two_distinct_positive_roots (m : ℝ) : Prop :=
  4 * m^2 - 4 * (m + 2) > 0 ∧ -2 * m > 0 ∧ m + 2 > 0

def hyperbola_with_foci_on_y_axis (m : ℝ) : Prop :=
  m + 3 < 0 ∧ 1 - 2 * m > 0

theorem range_of_m_given_q (m : ℝ) :
  hyperbola_with_foci_on_y_axis m → m < -3 :=
by
  sorry

theorem range_of_m_given_p_or_q_and_not_p_and_q (m : ℝ) :
  (quadratic_has_two_distinct_positive_roots m ∨ hyperbola_with_foci_on_y_axis m) ∧
  ¬(quadratic_has_two_distinct_positive_roots m ∧ hyperbola_with_foci_on_y_axis m) →
  (-2 < m ∧ m < -1) ∨ m < -3 :=
by
  sorry

end range_of_m_given_q_range_of_m_given_p_or_q_and_not_p_and_q_l167_167739


namespace total_age_10_years_from_now_is_75_l167_167956

-- Define the conditions
def eldest_age_now : ℕ := 20
def age_difference : ℕ := 5

-- Define the ages of the siblings 10 years from now
def eldest_age_10_years_from_now : ℕ := eldest_age_now + 10
def second_age_10_years_from_now : ℕ := (eldest_age_now - age_difference) + 10
def third_age_10_years_from_now : ℕ := (eldest_age_now - 2 * age_difference) + 10

-- Define the total age of the siblings 10 years from now
def total_age_10_years_from_now : ℕ := 
  eldest_age_10_years_from_now + 
  second_age_10_years_from_now + 
  third_age_10_years_from_now

-- The theorem statement
theorem total_age_10_years_from_now_is_75 : total_age_10_years_from_now = 75 := 
  by sorry

end total_age_10_years_from_now_is_75_l167_167956


namespace probability_of_green_is_7_over_50_l167_167535

-- We have tiles numbered from 1 to 100
def total_tiles : ℕ := 100

-- A tile is green if it is congruent to 3 modulo 7
def is_green (n : ℕ) : Prop := n % 7 = 3

-- The number of green tiles
def green_tiles : ℕ := (List.range total_tiles).filter is_green).length

-- The probability of selecting a green tile
def probability_of_green : ℚ := green_tiles / total_tiles

-- The proof statement
theorem probability_of_green_is_7_over_50 : probability_of_green = 7 / 50 := by
  sorry

end probability_of_green_is_7_over_50_l167_167535


namespace parabola_focus_coordinates_l167_167578

-- Define the given conditions
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2
def passes_through (a : ℝ) (p : ℝ × ℝ) : Prop := p.snd = parabola a p.fst

-- Main theorem: Prove the coordinates of the focus
theorem parabola_focus_coordinates (a : ℝ) (h : passes_through a (1, 4)) (ha : a = 4) : (0, 1 / 16) = (0, 1 / (4 * a)) :=
by
  rw [ha] -- substitute the value of a
  simp -- simplify the expression
  sorry

end parabola_focus_coordinates_l167_167578


namespace same_face_probability_correct_l167_167259

-- Define the number of sides on the dice
def sides_20 := 20
def sides_16 := 16

-- Define the number of colored sides for each dice category
def maroon_20 := 5
def teal_20 := 8
def cyan_20 := 6
def sparkly_20 := 1

def maroon_16 := 4
def teal_16 := 6
def cyan_16 := 5
def sparkly_16 := 1

-- Define the probabilities of each color matching
def prob_maroon : ℚ := (maroon_20 / sides_20) * (maroon_16 / sides_16)
def prob_teal : ℚ := (teal_20 / sides_20) * (teal_16 / sides_16)
def prob_cyan : ℚ := (cyan_20 / sides_20) * (cyan_16 / sides_16)
def prob_sparkly : ℚ := (sparkly_20 / sides_20) * (sparkly_16 / sides_16)

-- Define the total probability of same face
def prob_same_face := prob_maroon + prob_teal + prob_cyan + prob_sparkly

-- The theorem we need to prove
theorem same_face_probability_correct : 
  prob_same_face = 99 / 320 :=
by
  sorry

end same_face_probability_correct_l167_167259


namespace second_number_is_72_l167_167025

theorem second_number_is_72 
  (sum_eq_264 : ∀ (x : ℝ), 2 * x + x + (2 / 3) * x = 264) 
  (first_eq_2_second : ∀ (x : ℝ), first = 2 * x)
  (third_eq_1_3_first : ∀ (first : ℝ), third = 1 / 3 * first) :
  second = 72 :=
by
  sorry

end second_number_is_72_l167_167025


namespace max_cities_visited_l167_167929

theorem max_cities_visited (n k : ℕ) : ∃ t, t = n - k :=
by
  sorry

end max_cities_visited_l167_167929


namespace arithmetic_expression_l167_167852

theorem arithmetic_expression : (4 + 6 + 4) / 3 - 4 / 3 = 10 / 3 := by
  sorry

end arithmetic_expression_l167_167852


namespace black_equals_sum_of_white_l167_167417

theorem black_equals_sum_of_white :
  ∃ (a b c d : ℤ) (a_neq_zero : a ≠ 0) (b_neq_zero : b ≠ 0) (c_neq_zero : c ≠ 0) (d_neq_zero : d ≠ 0),
    (c + d * Real.sqrt 7 = (Real.sqrt (a + b * Real.sqrt 2) + Real.sqrt (a - b * Real.sqrt 2))^2) :=
by
  sorry

end black_equals_sum_of_white_l167_167417


namespace sin_double_angle_l167_167568

theorem sin_double_angle (x : ℝ) (h : Real.tan x = 1 / 3) : Real.sin (2 * x) = 3 / 5 := 
by 
  sorry

end sin_double_angle_l167_167568


namespace total_length_l167_167996

def length_pencil : ℕ := 12
def length_pen : ℕ := length_pencil - 2
def length_rubber : ℕ := length_pen - 3

theorem total_length : length_pencil + length_pen + length_rubber = 29 := by
  simp [length_pencil, length_pen, length_rubber]
  sorry

end total_length_l167_167996


namespace product_of_roots_is_four_thirds_l167_167132

theorem product_of_roots_is_four_thirds :
  (∀ p q r s : ℚ, (∃ a b c: ℚ, (3 * a^3 - 9 * a^2 + 5 * a - 4 = 0 ∧
                                   3 * b^3 - 9 * b^2 + 5 * b - 4 = 0 ∧
                                   3 * c^3 - 9 * c^2 + 5 * c - 4 = 0)) → 
  - s / p = (4 : ℚ) / 3) := sorry

end product_of_roots_is_four_thirds_l167_167132


namespace max_area_of_rectangle_with_perimeter_60_l167_167096

theorem max_area_of_rectangle_with_perimeter_60 :
  ∃ (x y : ℝ), 2 * (x + y) = 60 ∧ x * y = 225 :=
by
  sorry

end max_area_of_rectangle_with_perimeter_60_l167_167096


namespace suzie_reads_pages_hour_l167_167278

-- Declaration of the variables and conditions
variables (S : ℕ) -- S is the number of pages Suzie reads in an hour
variables (L : ℕ) -- L is the number of pages Liza reads in an hour

-- Conditions given in the problem
def reads_per_hour_Liza : L = 20 := sorry
def reads_more_pages : L * 3 = S * 3 + 15 := sorry

-- The statement we want to prove:
theorem suzie_reads_pages_hour : S = 15 :=
by
  -- Proof steps needed here (omitted due to the instruction)
  sorry

end suzie_reads_pages_hour_l167_167278


namespace digits_with_five_or_seven_is_5416_l167_167596

/-- The total number of four-digit positive integers. -/
def total_four_digit_integers : ℕ := 9000

/-- The number of four-digit integers without the digits 5 or 7. -/
def no_five_seven_integers : ℕ := 3584

/-- The number of four-digit positive integers that have at least one digit that is a 5 or a 7. -/
def digits_with_five_or_seven : ℕ :=
  total_four_digit_integers - no_five_seven_integers

/-- Proof that the number of four-digit integers with at least one digit that is a 5 or 7 is 5416. -/
theorem digits_with_five_or_seven_is_5416 :
  digits_with_five_or_seven = 5416 :=
by
  sorry

end digits_with_five_or_seven_is_5416_l167_167596


namespace probability_heads_9_of_12_flips_l167_167489

theorem probability_heads_9_of_12_flips : (nat.choose 12 9) / (2^12) = 55 / 1024 :=
by
  sorry

end probability_heads_9_of_12_flips_l167_167489


namespace nap_time_is_correct_l167_167552

-- Define the total trip time and the hours spent on each activity
def total_trip_time : ℝ := 15
def reading_time : ℝ := 2
def eating_time : ℝ := 1
def movies_time : ℝ := 3
def chatting_time : ℝ := 1
def browsing_time : ℝ := 0.75
def waiting_time : ℝ := 0.5
def working_time : ℝ := 2

-- Define the total activity time
def total_activity_time : ℝ := reading_time + eating_time + movies_time + chatting_time + browsing_time + waiting_time + working_time

-- Define the nap time as the difference between total trip time and total activity time
def nap_time : ℝ := total_trip_time - total_activity_time

-- Prove that the nap time is 4.75 hours
theorem nap_time_is_correct : nap_time = 4.75 :=
by
  -- Calculation hint, can be ignored
  -- nap_time = 15 - (2 + 1 + 3 + 1 + 0.75 + 0.5 + 2) = 15 - 10.25 = 4.75
  sorry

end nap_time_is_correct_l167_167552


namespace arithmetic_series_sum_l167_167965

theorem arithmetic_series_sum :
  let a := 18
  let d := 4
  let l := 58
  let n := (l - a) / d + 1
  let sum := n * (a + l) / 2
  sum = 418 := by {
  let a := 18
  let d := 4
  let l := 58
  let n := (l - a) / d + 1
  let sum := n * (a + l) / 2
  have h₁ : n = 11 := by sorry
  have h₂ : sum = 418 := by sorry
  exact h₂
}

end arithmetic_series_sum_l167_167965


namespace exists_a_max_value_of_four_l167_167718

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.cos x)^2 + 2 * a * Real.sin x + 3 * a - 1

theorem exists_a_max_value_of_four :
  ∃ a : ℝ, (a = 1) ∧ ∀ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2), f a x ≤ 4 := 
sorry

end exists_a_max_value_of_four_l167_167718


namespace probability_of_9_heads_in_12_l167_167473

def coin_flip_probability_9_heads_in_12_flips : Prop :=
  let total_outcomes := 2^12
  let success_outcomes := Nat.choose 12 9
  success_outcomes / total_outcomes = 220 / 4096

theorem probability_of_9_heads_in_12 : coin_flip_probability_9_heads_in_12_flips :=
  sorry

end probability_of_9_heads_in_12_l167_167473


namespace saved_percent_l167_167200

-- Definitions for conditions:
def last_year_saved (S : ℝ) : ℝ := 0.10 * S
def this_year_salary (S : ℝ) : ℝ := 1.10 * S
def this_year_saved (S : ℝ) : ℝ := 0.06 * (1.10 * S)

-- Given conditions and proof goal:
theorem saved_percent (S : ℝ) (hl_last_year_saved : last_year_saved S = 0.10 * S)
  (hl_this_year_salary : this_year_salary S = 1.10 * S)
  (hl_this_year_saved : this_year_saved S = 0.066 * S) :
  (this_year_saved S / last_year_saved S) * 100 = 66 :=
by
  sorry

end saved_percent_l167_167200


namespace base12_remainder_l167_167521

def base12_to_base10 (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) : ℕ :=
  a * 12^3 + b * 12^2 + c * 12^1 + d * 12^0

theorem base12_remainder (a b c d : ℕ) 
  (h1531 : base12_to_base10 a b c d = 1 * 12^3 + 5 * 12^2 + 3 * 12^1 + 1 * 12^0):
  (base12_to_base10 a b c d) % 8 = 5 :=
by
  unfold base12_to_base10 at h1531
  sorry

end base12_remainder_l167_167521


namespace total_erasers_l167_167268

def cases : ℕ := 7
def boxes_per_case : ℕ := 12
def erasers_per_box : ℕ := 25

theorem total_erasers : cases * boxes_per_case * erasers_per_box = 2100 := by
  sorry

end total_erasers_l167_167268


namespace arithmetic_geometric_sequence_l167_167157

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (q : ℝ) 
    (h1 : a 1 = 3)
    (h2 : a 1 + a 3 + a 5 = 21)
    (h3 : ∀ n, a (n + 1) = a n * q) :
  a 2 * a 4 = 36 := 
sorry

end arithmetic_geometric_sequence_l167_167157


namespace part_one_part_two_l167_167570

-- Part (1)
theorem part_one (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a^2 + 3 * b^2 ≥ 2 * b * (a + b) :=
sorry

-- Part (2)
theorem part_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) : 
  2 * a + b = 8 :=
sorry

end part_one_part_two_l167_167570


namespace max_value_inequality_l167_167416

theorem max_value_inequality : 
  ∀ (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (n : ℕ) (m : ℝ),
  (∀ n, S_n n = (n * a_n 1 + (1 / 2) * n * (n - 1) * d) ∧
  (∀ n, a_n n ^ 2 + (S_n n ^ 2 / n ^ 2) >= m * (a_n 1) ^ 2)) → 
  m ≤ 1 / 5 := 
sorry

end max_value_inequality_l167_167416


namespace absolute_value_of_slope_l167_167012

noncomputable def circle_center1 : ℝ × ℝ := (14, 92)
noncomputable def circle_center2 : ℝ × ℝ := (17, 76)
noncomputable def circle_center3 : ℝ × ℝ := (19, 84)
noncomputable def radius : ℝ := 3
noncomputable def point_on_line : ℝ × ℝ := (17, 76)

theorem absolute_value_of_slope :
  ∃ m : ℝ, ∀ line : ℝ × ℝ → Prop,
    (line point_on_line) ∧ 
    (∀ p, (line p) → true) → 
    abs m = 24 := 
  sorry

end absolute_value_of_slope_l167_167012


namespace tilly_bag_cost_l167_167449

noncomputable def cost_per_bag (n s P τ F : ℕ) : ℕ :=
  let revenue := n * s
  let total_sales_tax := n * (s * τ / 100)
  let total_additional_expenses := total_sales_tax + F
  (revenue - (P + total_additional_expenses)) / n

theorem tilly_bag_cost :
  let n := 100
  let s := 10
  let P := 300
  let τ := 5
  let F := 50
  cost_per_bag n s P τ F = 6 :=
  by
    let n := 100
    let s := 10
    let P := 300
    let τ := 5
    let F := 50
    have : cost_per_bag n s P τ F = 6 := sorry
    exact this

end tilly_bag_cost_l167_167449


namespace inequality_2_pow_n_gt_n_sq_for_n_5_l167_167662

theorem inequality_2_pow_n_gt_n_sq_for_n_5 : 2^5 > 5^2 := 
by {
    sorry -- Placeholder for the proof
}

end inequality_2_pow_n_gt_n_sq_for_n_5_l167_167662


namespace solve_equation_l167_167150

theorem solve_equation (x : ℝ) (h : x ≠ 2) : -x^2 = (4 * x + 2) / (x - 2) ↔ x = -2 :=
by sorry

end solve_equation_l167_167150


namespace smaller_of_two_digit_numbers_with_product_2210_l167_167002

theorem smaller_of_two_digit_numbers_with_product_2210 :
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 2210 ∧ a ≤ b ∧ a = 26 :=
by
  sorry

end smaller_of_two_digit_numbers_with_product_2210_l167_167002


namespace rhombus_diagonals_ratio_l167_167215

theorem rhombus_diagonals_ratio (a b d1 d2 : ℝ) 
  (h1: a > 0) (h2: b > 0)
  (h3: d1 = 2 * (a / Real.cos θ))
  (h4: d2 = 2 * (b / Real.cos θ)) :
  d1 / d2 = a / b := 
sorry

end rhombus_diagonals_ratio_l167_167215


namespace determine_m_range_l167_167241

-- Define propositions P and Q
def P (t : ℝ) : Prop := ∃ x y : ℝ, (x^2 / (t + 2) + y^2 / (t - 10) = 1)
def Q (t m : ℝ) : Prop := 1 - m < t ∧ t < 1 + m ∧ m > 0

-- Define negation of propositions
def notP (t : ℝ) : Prop := ∀ x y : ℝ, (x^2 / (t + 2) + y^2 / (t - 10) ≠ 1)
def notQ (t m : ℝ) : Prop := ¬ (1 - m < t ∧ t < 1 + m)

-- Main problem: Determine the range of m where notP -> notQ is a sufficient but not necessary condition
theorem determine_m_range {m : ℝ} : (∃ t : ℝ, notP t → notQ t m) ↔ (0 < m ∧ m ≤ 3) := by
  sorry

end determine_m_range_l167_167241


namespace tim_kittens_count_l167_167300

def initial_kittens : Nat := 6
def kittens_given_to_jessica : Nat := 3
def kittens_received_from_sara : Nat := 9

theorem tim_kittens_count : initial_kittens - kittens_given_to_jessica + kittens_received_from_sara = 12 :=
by
  sorry

end tim_kittens_count_l167_167300


namespace binomial_last_three_terms_sum_l167_167575

theorem binomial_last_three_terms_sum (n : ℕ) :
  (1 + n + (n * (n - 1)) / 2 = 79) → n = 12 :=
by
  sorry

end binomial_last_three_terms_sum_l167_167575


namespace probability_of_heads_on_999th_toss_l167_167959

theorem probability_of_heads_on_999th_toss (fair_coin : Bool → ℝ) :
  (∀ (i : ℕ), fair_coin true = 1 / 2 ∧ fair_coin false = 1 / 2) →
  fair_coin true = 1 / 2 :=
by
  sorry

end probability_of_heads_on_999th_toss_l167_167959


namespace find_tangent_line_at_neg1_l167_167861

noncomputable def tangent_line (x : ℝ) : ℝ := 2 * x^2 + 3

theorem find_tangent_line_at_neg1 :
  let x := -1
  let m := 4 * x
  let y := 2 * x^2 + 3
  let tangent := y + m * (x - x)
  tangent = -4 * x + 1 :=
by
  sorry

end find_tangent_line_at_neg1_l167_167861


namespace find_e_l167_167761

theorem find_e (a e : ℕ) (h1: a = 105) (h2: a ^ 3 = 21 * 25 * 45 * e) : e = 49 :=
sorry

end find_e_l167_167761


namespace arithmetic_sequence_sum_l167_167968

theorem arithmetic_sequence_sum :
  ∀ (x y : ℤ), (∃ (n m : ℕ), (3 + n * 6 = x) ∧ (3 + m * 6 = y) ∧ x + 6 = y ∧ y + 6 = 33) → x + y = 60 :=
by
  intro x y h
  obtain ⟨n, m, hn, hm, hx, hy⟩ := h
  exact sorry

end arithmetic_sequence_sum_l167_167968


namespace apple_selling_price_l167_167327

theorem apple_selling_price (CP SP Loss : ℝ) (h₀ : CP = 18) (h₁ : Loss = (1/6) * CP) (h₂ : SP = CP - Loss) : SP = 15 :=
  sorry

end apple_selling_price_l167_167327


namespace range_of_a_l167_167255

theorem range_of_a (a : ℝ) : 
  (¬ ∀ x : ℝ, x^2 + a * x + 1 ≥ 0) ↔ (a < -2 ∨ a > 2) :=
by
  sorry

end range_of_a_l167_167255


namespace eastville_to_westpath_travel_time_l167_167801

theorem eastville_to_westpath_travel_time :
  ∀ (d t₁ t₂ : ℝ) (s₁ s₂ : ℝ), 
  t₁ = 6 → s₁ = 80 → s₂ = 50 → d = s₁ * t₁ → t₂ = d / s₂ → t₂ = 9.6 := 
by
  intros d t₁ t₂ s₁ s₂ ht₁ hs₁ hs₂ hd ht₂
  sorry

end eastville_to_westpath_travel_time_l167_167801


namespace eggs_per_basket_l167_167418

-- Lucas places a total of 30 blue Easter eggs in several yellow baskets
-- Lucas places a total of 42 green Easter eggs in some purple baskets
-- Each basket contains the same number of eggs
-- There are at least 5 eggs in each basket

theorem eggs_per_basket (n : ℕ) (h1 : n ∣ 30) (h2 : n ∣ 42) (h3 : n ≥ 5) : n = 6 :=
by
  sorry

end eggs_per_basket_l167_167418


namespace p_correct_l167_167358

noncomputable def p : ℝ → ℝ := sorry

axiom p_at_3 : p 3 = 10

axiom p_condition (x y : ℝ) : p x * p y = p x + p y + p (x * y) - 2

theorem p_correct : ∀ x, p x = x^2 + 1 :=
sorry

end p_correct_l167_167358


namespace range_of_y_for_x_gt_2_l167_167615

theorem range_of_y_for_x_gt_2 :
  (∀ x : ℝ, x > 2 → 0 < 2 / x ∧ 2 / x < 1) :=
by 
  -- Proof is omitted
  sorry

end range_of_y_for_x_gt_2_l167_167615


namespace find_triplets_find_triplets_non_negative_l167_167729

theorem find_triplets :
  ∀ (x y z : ℕ), (x > 0) ∧ (y > 0) ∧ (z > 0) →
    x^2 + y^2 + 1 = 2^z →
    (x = 1 ∧ y = 0 ∧ z = 1) ∨ (x = 0 ∧ y = 1 ∧ z = 1) :=
by
  sorry

theorem find_triplets_non_negative :
  ∀ (x y z : ℕ), x^2 + y^2 + 1 = 2^z →
    (x = 1 ∧ y = 0 ∧ z = 1) ∨ (x = 0 ∧ y = 1 ∧ z = 1) ∨ (x = 0 ∧ y = 0 ∧ z = 0) :=
by
  sorry

end find_triplets_find_triplets_non_negative_l167_167729


namespace parabola_solution_l167_167213

theorem parabola_solution (a b : ℝ) : 
  (∃ y : ℝ, y = 2^2 + 2 * a + b ∧ y = 20) ∧ 
  (∃ y : ℝ, y = (-2)^2 + (-2) * a + b ∧ y = 0) ∧ 
  b = (0^2 + 0 * a + b) → 
  a = 5 ∧ b = 6 := 
by {
  sorry
}

end parabola_solution_l167_167213


namespace cricket_team_throwers_l167_167422

def cricket_equation (T N : ℕ) := 
  (2 * N / 3 = 51 - T) ∧ (T + N = 58)

theorem cricket_team_throwers : 
  ∃ T : ℕ, ∃ N : ℕ, cricket_equation T N ∧ T = 37 :=
by
  sorry

end cricket_team_throwers_l167_167422


namespace contribution_per_person_correct_l167_167184

-- Definitions from conditions
def total_fundraising_goal : ℕ := 2400
def number_of_participants : ℕ := 8
def administrative_fee_per_person : ℕ := 20

-- Desired answer
def total_contribution_per_person : ℕ := total_fundraising_goal / number_of_participants + administrative_fee_per_person

-- Proof statement
theorem contribution_per_person_correct :
  total_contribution_per_person = 320 :=
by
  sorry  -- Proof to be provided

end contribution_per_person_correct_l167_167184


namespace part_whole_ratio_l167_167324

theorem part_whole_ratio (N x : ℕ) (hN : N = 160) (hx : x + 4 = N / 4 - 4) :
  x / N = 1 / 5 :=
  sorry

end part_whole_ratio_l167_167324


namespace angle_Q_measure_in_triangle_PQR_l167_167768

theorem angle_Q_measure_in_triangle_PQR (angle_R angle_Q angle_P : ℝ) (h1 : angle_P = 3 * angle_R) (h2 : angle_Q = angle_R) (h3 : angle_R + angle_Q + angle_P = 180) : angle_Q = 36 :=
by {
  -- Placeholder for the proof, which is not required as per the instructions
  sorry
}

end angle_Q_measure_in_triangle_PQR_l167_167768


namespace sqrt_21_between_4_and_5_l167_167049

theorem sqrt_21_between_4_and_5 : 4 < Real.sqrt 21 ∧ Real.sqrt 21 < 5 := 
by 
  sorry

end sqrt_21_between_4_and_5_l167_167049


namespace area_of_octagon_l167_167737

-- Define the basic geometric elements and properties
variables {A B C D E F G H : Type}
variables (isRectangle : BDEF A B C D E F G H)
variables (AB BC : ℝ) (hAB : AB = 2) (hBC : BC = 2)
variables (isRightIsosceles : ABC A B C D E F G H)

-- Assumptions and known facts
def BDEF_is_rectangle : Prop := isRectangle
def AB_eq_2 : AB = 2 := hAB
def BC_eq_2 : BC = 2 := hBC
def ABC_is_right_isosceles : Prop := isRightIsosceles

-- Statement of the problem to be proved
theorem area_of_octagon : (exists (area : ℝ), area = 8 * Real.sqrt 2) :=
by {
  -- The proof details will go here, which we skip for now
  sorry
}

end area_of_octagon_l167_167737


namespace slope_angle_y_eq_neg1_l167_167293

theorem slope_angle_y_eq_neg1 : (∃ line : ℝ → ℝ, ∀ y: ℝ, line y = -1 → ∃ θ : ℝ, θ = 0) :=
by
  -- Sorry is used to skip the proof.
  sorry

end slope_angle_y_eq_neg1_l167_167293


namespace kareem_largest_l167_167272

def jose_final : ℕ :=
  let start := 15
  let minus_two := start - 2
  let triple := minus_two * 3
  triple + 5

def thuy_final : ℕ :=
  let start := 15
  let triple := start * 3
  let minus_two := triple - 2
  minus_two + 5

def kareem_final : ℕ :=
  let start := 15
  let minus_two := start - 2
  let add_five := minus_two + 5
  add_five * 3

theorem kareem_largest : kareem_final > jose_final ∧ kareem_final > thuy_final := by
  sorry

end kareem_largest_l167_167272


namespace angle_y_equals_90_l167_167357

/-- In a geometric configuration, if ∠CBD = 120° and ∠ABE = 30°, 
    then the measure of angle y is 90°. -/
theorem angle_y_equals_90 (angle_CBD angle_ABE : ℝ) 
  (h1 : angle_CBD = 120) 
  (h2 : angle_ABE = 30) : 
  ∃ y : ℝ, y = 90 := 
by
  sorry

end angle_y_equals_90_l167_167357


namespace necessary_but_not_sufficient_condition_is_purely_imaginary_l167_167374

noncomputable def is_purely_imaginary (z : ℂ) : Prop :=
  ∃ (b : ℝ), z = ⟨0, b⟩

theorem necessary_but_not_sufficient_condition_is_purely_imaginary (a b : ℝ) (h_imaginary : is_purely_imaginary (⟨a, b⟩)) : 
  (a = 0) ∧ (b ≠ 0) :=
by
  sorry

end necessary_but_not_sufficient_condition_is_purely_imaginary_l167_167374


namespace distinct_integer_values_b_for_quadratic_l167_167056

theorem distinct_integer_values_b_for_quadratic :
  ∃ (S : Finset ℤ), ∀ b ∈ S, (∃ x y : ℤ, x^2 + b * x + 12 * b = 0 ∧ y^2 + b * y + 12 * b = 0) ∧ S.card = 16 :=
sorry

end distinct_integer_values_b_for_quadratic_l167_167056


namespace smallest_number_among_bases_l167_167550

theorem smallest_number_among_bases : 
  let n1 := 8 * 9 + 5,
      n2 := 2 * 6^2 + 1 * 6,
      n3 := 1 * 4^3,
      n4 := 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2 + 1 in
  n4 < n3 ∧ n4 < n1 ∧ n4 < n2 :=
by
  let n1 := 8 * 9 + 5,
      n2 := 2 * 6^2 + 1 * 6,
      n3 := 1 * 4^3,
      n4 := 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2 + 1
  sorry

end smallest_number_among_bases_l167_167550


namespace solve_system_eqs_l167_167352

theorem solve_system_eqs (x y : ℝ) (h1 : (1 + x) * (1 + x^2) * (1 + x^4) = 1 + y^7)
                            (h2 : (1 + y) * (1 + y^2) * (1 + y^4) = 1 + x^7) :
  (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = -1) :=
sorry

end solve_system_eqs_l167_167352


namespace jerry_won_games_l167_167618

theorem jerry_won_games 
  (T : ℕ) (K D J : ℕ) 
  (h1 : T = 32) 
  (h2 : K = D + 5) 
  (h3 : D = J + 3) : 
  J = 7 := 
sorry

end jerry_won_games_l167_167618


namespace count_four_digit_integers_with_5_or_7_l167_167590

theorem count_four_digit_integers_with_5_or_7 : 
  (set.range (λ n: ℕ, n >= 1000 ∧ n <= 9999) 
   \ { n | ∀ d ∈ toDigits n, d ≠ 5 ∧ d ≠ 7 }).card = 5416 :=
by
  sorry

end count_four_digit_integers_with_5_or_7_l167_167590


namespace exists_divisible_sk_l167_167409

noncomputable def sequence_of_integers (c : ℕ) (a : ℕ → ℕ) :=
  ∀ n : ℕ, 0 < n → a n < a (n + 1) ∧ a (n + 1) < a n + c

noncomputable def infinite_string (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (10 ^ n) * (a (n + 1)) + a n

noncomputable def sk (s : ℕ) (k : ℕ) : ℕ :=
  (s % (10 ^ k))

theorem exists_divisible_sk (a : ℕ → ℕ) (c m : ℕ)
  (h : sequence_of_integers c a) :
  ∀ m : ℕ, ∃ k : ℕ, m > 0 → (sk (infinite_string a k) k) % m = 0 := by
  sorry

end exists_divisible_sk_l167_167409


namespace sum_of_eight_fib_not_in_sequence_l167_167688

theorem sum_of_eight_fib_not_in_sequence (k n : ℕ) :
  (∀ i, i ≤ 7 → fib (k + 1 + i) ≠ fib n) :=
by sorry

end sum_of_eight_fib_not_in_sequence_l167_167688


namespace smallest_k_l167_167636

theorem smallest_k (a b : ℚ) (h_a_period : ∀ n, a ≠ (10^30 - 1) * n)
  (h_b_period : ∀ n, b ≠ (10^30 - 1) * n)
  (h_diff_period : ∀ n, a - b ≠ (10^15 - 1) * n) :
  ∃ k : ℕ, k = 6 ∧ (a + (k:ℚ) * b) ≠ (10^15 - 1) :=
sorry

end smallest_k_l167_167636


namespace area_of_ADFE_l167_167384

namespace Geometry

open Classical

noncomputable def area_triangle (A B C : Type) [Field A] (area_DBF area_BFC area_FCE : A) : A :=
  let total_area := area_DBF + area_BFC + area_FCE
  let area := (105 : A) / 4
  total_area + area

theorem area_of_ADFE (A B C D E F : Type) [Field A] 
  (area_DBF : A) (area_BFC : A) (area_FCE : A) : 
  area_DBF = 4 → area_BFC = 6 → area_FCE = 5 → 
  area_triangle A B C area_DBF area_BFC area_FCE = (15 : A) + (105 : A) / 4 := 
by 
  intros 
  sorry

end area_of_ADFE_l167_167384


namespace tangent_line_at_pi_over_4_l167_167233

noncomputable def tangent_eq (x y : ℝ) : Prop :=
  y = 2 * x * Real.tan x

noncomputable def tangent_line_eq (x y : ℝ) : Prop :=
  (2 + Real.pi) * x - y - (Real.pi^2 / 4) = 0

theorem tangent_line_at_pi_over_4 :
  tangent_eq (Real.pi / 4) (Real.pi / 2) →
  tangent_line_eq (Real.pi / 4) (Real.pi / 2) :=
by
  sorry

end tangent_line_at_pi_over_4_l167_167233


namespace snake_body_length_l167_167332

theorem snake_body_length (L : ℝ) (H : ℝ) (h1 : H = L / 10) (h2 : L = 10) : L - H = 9 :=
by
  sorry

end snake_body_length_l167_167332


namespace probability_of_same_color_is_34_over_105_l167_167752

-- Define the number of each color of plates
def num_red_plates : ℕ := 7
def num_blue_plates : ℕ := 5
def num_yellow_plates : ℕ := 3

-- Define the total number of plates
def total_plates : ℕ := num_red_plates + num_blue_plates + num_yellow_plates

-- Define the total number of ways to choose 2 plates from the total plates
def total_ways_to_choose_2_plates : ℕ := Nat.choose total_plates 2

-- Define the number of ways to choose 2 red plates, 2 blue plates, and 2 yellow plates
def ways_to_choose_2_red_plates : ℕ := Nat.choose num_red_plates 2
def ways_to_choose_2_blue_plates : ℕ := Nat.choose num_blue_plates 2
def ways_to_choose_2_yellow_plates : ℕ := Nat.choose num_yellow_plates 2

-- Define the total number of favorable outcomes (same color plates)
def favorable_outcomes : ℕ :=
  ways_to_choose_2_red_plates + ways_to_choose_2_blue_plates + ways_to_choose_2_yellow_plates

-- Prove that the probability is 34/105
theorem probability_of_same_color_is_34_over_105 :
  (favorable_outcomes : ℚ) / (total_ways_to_choose_2_plates : ℚ) = 34 / 105 := by
  sorry

end probability_of_same_color_is_34_over_105_l167_167752


namespace twenty_million_in_scientific_notation_l167_167430

/-- Prove that 20 million in scientific notation is 2 * 10^7 --/
theorem twenty_million_in_scientific_notation : 20000000 = 2 * 10^7 :=
by
  sorry

end twenty_million_in_scientific_notation_l167_167430


namespace green_eyes_count_l167_167007

theorem green_eyes_count (total_people : ℕ) (blue_eyes : ℕ) (brown_eyes : ℕ) (black_eyes : ℕ) (green_eyes : ℕ) :
  total_people = 100 → 
  blue_eyes = 19 → 
  brown_eyes = total_people / 2 → 
  black_eyes = total_people / 4 → 
  green_eyes = total_people - (blue_eyes + brown_eyes + black_eyes) → 
  green_eyes = 6 := 
by 
  intros h_total h_blue h_brown h_black h_green 
  rw [h_total, h_blue, h_brown, h_black] at h_green 
  exact h_green.symm

end green_eyes_count_l167_167007


namespace sin_C_of_arithmetic_sequence_l167_167123

theorem sin_C_of_arithmetic_sequence 
  (A B C : ℝ) 
  (h1 : A + C = 2 * B) 
  (h2 : A + B + C = Real.pi) 
  (h3 : Real.cos A = 2 / 3) 
  : Real.sin C = (Real.sqrt 5 + 2 * Real.sqrt 3) / 6 :=
sorry

end sin_C_of_arithmetic_sequence_l167_167123


namespace probability_9_heads_12_flips_l167_167517

noncomputable def probability_of_9_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_9_heads_12_flips : probability_of_9_heads 12 9 (1 / 2) = 55 / 1024 := by
  sorry

end probability_9_heads_12_flips_l167_167517


namespace florist_has_56_roses_l167_167206

def initial_roses := 50
def roses_sold := 15
def roses_picked := 21

theorem florist_has_56_roses (r0 rs rp : ℕ) (h1 : r0 = initial_roses) (h2 : rs = roses_sold) (h3 : rp = roses_picked) : 
  r0 - rs + rp = 56 :=
by sorry

end florist_has_56_roses_l167_167206


namespace evaluate_expression_l167_167853

theorem evaluate_expression : 
  -((5: ℤ) ^ 2) - (-(3: ℤ) ^ 3) * ((2: ℚ) / 9) - 9 * |((-(2: ℚ)) / 3)| = -25 := by
  sorry

end evaluate_expression_l167_167853


namespace total_volume_of_all_cubes_l167_167222

def cube_volume (side_length : ℕ) : ℕ := side_length ^ 3

def total_volume (count : ℕ) (side_length : ℕ) : ℕ := count * (cube_volume side_length)

theorem total_volume_of_all_cubes :
  total_volume 4 3 + total_volume 3 4 = 300 :=
by
  sorry

end total_volume_of_all_cubes_l167_167222


namespace MrsHiltReadTotalChapters_l167_167315

-- Define the number of books and chapters per book
def numberOfBooks : ℕ := 4
def chaptersPerBook : ℕ := 17

-- Define the total number of chapters Mrs. Hilt read
def totalChapters (books : ℕ) (chapters : ℕ) : ℕ := books * chapters

-- The main statement to be proved
theorem MrsHiltReadTotalChapters : totalChapters numberOfBooks chaptersPerBook = 68 := by
  sorry

end MrsHiltReadTotalChapters_l167_167315


namespace rectangle_problem_l167_167735

theorem rectangle_problem (x : ℝ) (h1 : 4 * x = l) (h2 : x + 7 = w) (h3 : l * w = 2 * (2 * l + 2 * w)) : x = 1 := 
by {
  sorry
}

end rectangle_problem_l167_167735


namespace money_left_l167_167210

theorem money_left 
  (salary : ℝ)
  (spent_on_food : ℝ)
  (spent_on_rent : ℝ)
  (spent_on_clothes : ℝ)
  (total_spent : ℝ)
  (money_left : ℝ)
  (h_salary : salary = 170000)
  (h_food : spent_on_food = salary * (1 / 5))
  (h_rent : spent_on_rent = salary * (1 / 10))
  (h_clothes : spent_on_clothes = salary * (3 / 5))
  (h_total_spent : total_spent = spent_on_food + spent_on_rent + spent_on_clothes)
  (h_money_left : money_left = salary - total_spent) :
  money_left = 17000 :=
by
  sorry

end money_left_l167_167210


namespace max_rectangle_area_l167_167088

theorem max_rectangle_area (P : ℝ) (hP : P = 60) : ∃ A : ℝ, A = 225 :=
by 
  sorry

end max_rectangle_area_l167_167088


namespace find_m_l167_167243

theorem find_m (m : ℝ) :
  (∃ x : ℝ, (m - 1) * x^2 + 2 * x + m^2 - 1 = 0) ∧ (m - 1 ≠ 0) → m = -1 :=
by
  sorry

end find_m_l167_167243


namespace sum_in_base_4_l167_167846

theorem sum_in_base_4 : 
  let n1 := 2
  let n2 := 23
  let n3 := 132
  let n4 := 1320
  let sum := 20200
  n1 + n2 + n3 + n4 = sum := 
by
  sorry

end sum_in_base_4_l167_167846


namespace octagon_mass_l167_167364

theorem octagon_mass :
  let side_length := 1 -- side length of the original square (meters)
  let thickness := 0.3 -- thickness of the sheet (cm)
  let density := 7.8 -- density of steel (g/cm^3)
  let x := 50 * (2 - Real.sqrt 2) -- side length of the triangles (cm)
  let octagon_area := 20000 * (Real.sqrt 2 - 1) -- area of the octagon (cm^2)
  let volume := octagon_area * thickness -- volume of the octagon (cm^3)
  let mass := volume * density / 1000 -- mass of the octagon (kg), converted from g to kg
  mass = 19 :=
by
  sorry

end octagon_mass_l167_167364


namespace determinant_of_roots_l167_167780

noncomputable def determinant_expr (a b c d s p q r : ℝ) : ℝ :=
  by sorry

theorem determinant_of_roots (a b c d s p q r : ℝ)
    (h1 : a + b + c + d = -s)
    (h2 : abcd = r)
    (h3 : abc + abd + acd + bcd = -q)
    (h4 : ab + ac + bc = p) :
    determinant_expr a b c d s p q r = r - q + pq + p :=
  by sorry

end determinant_of_roots_l167_167780


namespace probability_heads_9_of_12_flips_l167_167492

theorem probability_heads_9_of_12_flips : (nat.choose 12 9) / (2^12) = 55 / 1024 :=
by
  sorry

end probability_heads_9_of_12_flips_l167_167492


namespace probability_9_heads_12_flips_l167_167514

noncomputable def probability_of_9_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_9_heads_12_flips : probability_of_9_heads 12 9 (1 / 2) = 55 / 1024 := by
  sorry

end probability_9_heads_12_flips_l167_167514


namespace probability_nine_heads_l167_167469

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

def flips : ℕ := 12
def heads : ℕ := 9
def total_outcomes : ℕ := 2^flips
def favorable_outcomes : ℕ := binom flips heads
def probability : Rat := favorable_outcomes / total_outcomes.toRat

theorem probability_nine_heads :
  probability = (220/4096) := by
  sorry

end probability_nine_heads_l167_167469


namespace product_closest_value_l167_167706

-- Define the constants used in the problem
def a : ℝ := 2.5
def b : ℝ := 53.6
def c : ℝ := 0.4

-- Define the expression and the expected correct answer
def expression : ℝ := a * (b - c)
def correct_answer : ℝ := 133

-- State the theorem that the expression evaluates to the correct answer
theorem product_closest_value : expression = correct_answer :=
by
  sorry

end product_closest_value_l167_167706


namespace prove_trig_inequality_l167_167164

noncomputable def trig_inequality : Prop :=
  (0 < 1 / 2) ∧ (1 / 2 < Real.pi / 6) ∧
  (∀ x y : ℝ, (0 < x) ∧ (x < y) ∧ (y < Real.pi / 6) → Real.sin x < Real.sin y) ∧
  (∀ x y : ℝ, (0 < x) ∧ (x < y) ∧ (y < Real.pi / 6) → Real.cos x > Real.cos y) →
  (Real.cos (1 / 2) > Real.tan (1 / 2) ∧ Real.tan (1 / 2) > Real.sin (1 / 2))

theorem prove_trig_inequality : trig_inequality :=
by
  sorry

end prove_trig_inequality_l167_167164


namespace probability_nine_heads_l167_167468

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

def flips : ℕ := 12
def heads : ℕ := 9
def total_outcomes : ℕ := 2^flips
def favorable_outcomes : ℕ := binom flips heads
def probability : Rat := favorable_outcomes / total_outcomes.toRat

theorem probability_nine_heads :
  probability = (220/4096) := by
  sorry

end probability_nine_heads_l167_167468


namespace yuri_total_puppies_l167_167187

-- Conditions
def first_week_puppies := 20
def second_week_puppies : ℕ := (2 / 5) * first_week_puppies
def third_week_puppies : ℕ := 2 * second_week_puppies
def fourth_week_puppies := first_week_puppies + 10

-- Total puppies
def total_puppies : ℕ := first_week_puppies + second_week_puppies + third_week_puppies + fourth_week_puppies

-- Theorem to prove
theorem yuri_total_puppies : total_puppies = 74 :=
by sorry

end yuri_total_puppies_l167_167187


namespace find_coordinates_of_P_l167_167898

noncomputable def pointP_minimizes_dot_product : Prop :=
  let OA := (2, 2)
  let OB := (4, 1)
  let AP x := (x - 2, -2)
  let BP x := (x - 4, -1)
  let dot_product x := (AP x).1 * (BP x).1 + (AP x).2 * (BP x).2
  ∃ x, (dot_product x = (x - 3) ^ 2 + 1) ∧ (∀ y, dot_product y ≥ dot_product x) ∧ (x = 3)

theorem find_coordinates_of_P : pointP_minimizes_dot_product :=
  sorry

end find_coordinates_of_P_l167_167898


namespace number_of_days_worked_l167_167877

-- Define the conditions
def hours_per_day := 8
def total_hours := 32

-- Define the proof statement
theorem number_of_days_worked : total_hours / hours_per_day = 4 :=
by
  -- Placeholder for the proof
  sorry

end number_of_days_worked_l167_167877


namespace inequality_solution_interval_l167_167941

noncomputable def solve_inequality (x : ℝ) : Prop :=
  -2 < (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) ∧
  (x^2 - 4 * x + 5) ≠ 0 ∧
  (3 * x^2 - 24 * x + 25) / (x^2 - 4 * x + 5) > 0 ∧
  (- x^2 - 8 * x + 5) / (x^2 - 4 * x + 5) < 0

theorem inequality_solution_interval (x : ℝ) :
  solve_inequality x :=
sorry

end inequality_solution_interval_l167_167941


namespace range_of_m_for_p_range_of_m_for_p_and_not_q_l167_167883

-- Definitions of the propositions p and q
def prop_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * m * x - 3 * m > 0

def prop_q (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 4 * m * x + 1 < 0

-- The Lean theorem to prove the ranges of m based on the conditions
theorem range_of_m_for_p :
  ∀ m : ℝ, prop_p m → -3 < m ∧ m < 0 := sorry

theorem range_of_m_for_p_and_not_q :
  ∀ m : ℝ, prop_p m → ¬ prop_q m → -1/2 ≤ m ∧ m < 0 := sorry

end range_of_m_for_p_range_of_m_for_p_and_not_q_l167_167883


namespace greatest_x_for_4x_in_factorial_21_l167_167673

-- Definition and theorem to state the problem mathematically
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_x_for_4x_in_factorial_21 : ∃ x : ℕ, (4^x ∣ factorial 21) ∧ ∀ y : ℕ, (4^y ∣ factorial 21) → y ≤ 9 :=
by
  sorry

end greatest_x_for_4x_in_factorial_21_l167_167673


namespace average_time_per_leg_l167_167026

-- Conditions
def time_y : ℕ := 58
def time_z : ℕ := 26
def total_time : ℕ := time_y + time_z
def number_of_legs : ℕ := 2

-- Theorem stating the average time per leg
theorem average_time_per_leg : total_time / number_of_legs = 42 := by
  sorry

end average_time_per_leg_l167_167026


namespace largest_possible_package_size_l167_167783

theorem largest_possible_package_size :
  ∃ (p : ℕ), Nat.gcd 60 36 = p ∧ p = 12 :=
by
  use 12
  sorry -- The proof is skipped as per instructions

end largest_possible_package_size_l167_167783


namespace probability_heads_9_of_12_flips_l167_167488

theorem probability_heads_9_of_12_flips : (nat.choose 12 9) / (2^12) = 55 / 1024 :=
by
  sorry

end probability_heads_9_of_12_flips_l167_167488


namespace satisfies_diff_eq_l167_167792

def y (x c : ℝ) := x * (c - real.log x)

theorem satisfies_diff_eq (x c : ℝ) (h₁ : x > 0) :
  let y := y x c,
      dy := deriv (λ x, y x c)
  in (x - y) + x * dy = 0 := sorry

end satisfies_diff_eq_l167_167792


namespace neg_of_proposition_l167_167256

variable (a : ℝ)

def proposition := ∀ x : ℝ, 0 < a^x

theorem neg_of_proposition (h₀ : 0 < a) (h₁ : a ≠ 1) : ¬proposition a ↔ ∃ x : ℝ, a^x ≤ 0 :=
by
  sorry

end neg_of_proposition_l167_167256


namespace largest_value_among_expressions_l167_167373

theorem largest_value_among_expressions 
  (a1 a2 b1 b2 : ℝ)
  (h1 : 0 < a1) (h2 : a1 < a2) (h3 : a2 < 1)
  (h4 : 0 < b1) (h5 : b1 < b2) (h6 : b2 < 1)
  (ha : a1 + a2 = 1) (hb : b1 + b2 = 1) :
  a1 * b1 + a2 * b2 > a1 * a2 + b1 * b2 ∧ 
  a1 * b1 + a2 * b2 > a1 * b2 + a2 * b1 := 
sorry

end largest_value_among_expressions_l167_167373


namespace ms_tom_investment_l167_167824

def invested_amounts (X Y : ℝ) : Prop :=
  X + Y = 100000 ∧ 0.17 * Y = 0.23 * X + 200 

theorem ms_tom_investment (X Y : ℝ) (h : invested_amounts X Y) : X = 42000 :=
by
  sorry

end ms_tom_investment_l167_167824


namespace max_rectangle_area_l167_167093

-- Define the perimeter and side lengths of the rectangle
def perimeter := 60
def L (x : ℝ) := x
def W (x : ℝ) := 30 - x

-- Define the area of the rectangle
def area (x : ℝ) := L x * W x

-- State the theorem to prove the maximum area is 225 square feet
theorem max_rectangle_area : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 30 ∧ area x = 225 :=
by
  sorry

end max_rectangle_area_l167_167093


namespace polynomial_condition_satisfied_l167_167360

-- Definitions as per conditions:
def p (x : ℝ) : ℝ := x^2 + 1

-- Conditions:
axiom cond1 : p 3 = 10
axiom cond2 : ∀ (x y : ℝ), p x * p y = p x + p y + p (x * y) - 2

-- Theorem to prove:
theorem polynomial_condition_satisfied : (p 3 = 10) ∧ (∀ (x y : ℝ), p x * p y = p x + p y + p (x * y) - 2) :=
by
  apply And.intro cond1
  apply cond2

end polynomial_condition_satisfied_l167_167360


namespace sum_four_digit_integers_ending_in_zero_l167_167664

def arithmetic_series_sum (a l d : ℕ) : ℕ := 
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem sum_four_digit_integers_ending_in_zero : 
  arithmetic_series_sum 1000 9990 10 = 4945500 :=
by
  sorry

end sum_four_digit_integers_ending_in_zero_l167_167664


namespace geometric_sequence_sum_six_l167_167888

theorem geometric_sequence_sum_six (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) 
  (h1 : 0 < q)
  (h2 : a 1 = 1)
  (h3 : a 3 * a 5 = 64)
  (h4 : ∀ n, a n = a 1 * q^(n-1))
  (h5 : ∀ n, S n = (a 1 * (1 - q^n)) / (1 - q)) :
  S 6 = 63 := 
sorry

end geometric_sequence_sum_six_l167_167888


namespace find_a_l167_167237

theorem find_a (a b : ℝ) (r s t : ℝ) 
  (h_poly : 7 * r^3 + 3 * a * r^2 + 6 * b * r + 2 * a = 0)
  (h_roots : r ≠ s ∧ s ≠ t ∧ r ≠ t)
  (h_pos_roots : r > 0 ∧ s > 0 ∧ t > 0)
  (h_log_sum : Real.log 243 = 5 * Real.log 3)
  (h_sum_squares : r^2 + s^2 + t^2 = 49) :
  a = -850.5 :=
by
  sorry

end find_a_l167_167237


namespace find_length_QS_l167_167796

theorem find_length_QS 
  (cosR : ℝ) (RS : ℝ) (QR : ℝ) (QS : ℝ)
  (h1 : cosR = 3 / 5)
  (h2 : RS = 10)
  (h3 : cosR = QR / RS) :
  QS = 8 :=
by
  sorry

end find_length_QS_l167_167796


namespace smallest_possible_value_of_other_integer_l167_167665

theorem smallest_possible_value_of_other_integer 
  (n : ℕ) (hn_pos : 0 < n) (h_eq : (Nat.lcm 75 n) / (Nat.gcd 75 n) = 45) : n = 135 :=
by sorry

end smallest_possible_value_of_other_integer_l167_167665


namespace cosine_identity_l167_167905

theorem cosine_identity (α : ℝ) (h : Real.sin α = 1 / 3) : Real.cos (π / 2 + α) = -1 / 3 := by
  sorry

end cosine_identity_l167_167905


namespace number_of_minibusses_l167_167917

def total_students := 156
def students_per_van := 10
def students_per_minibus := 24
def number_of_vans := 6

theorem number_of_minibusses : (total_students - number_of_vans * students_per_van) / students_per_minibus = 4 :=
by
  sorry

end number_of_minibusses_l167_167917


namespace band_to_orchestra_ratio_is_two_l167_167168

noncomputable def ratio_of_band_to_orchestra : ℤ :=
  let orchestra_students := 20
  let choir_students := 28
  let total_students := 88
  let band_students := (total_students - orchestra_students - choir_students)
  band_students / orchestra_students

theorem band_to_orchestra_ratio_is_two :
  let orchestra_students := 20
  let choir_students := 28
  let total_students := 88
  ratio_of_band_to_orchestra = 2 := by
  sorry

end band_to_orchestra_ratio_is_two_l167_167168


namespace inequality_solution_l167_167938

theorem inequality_solution (x : ℝ) :
  -2 < (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) ∧
  (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) < 2 ↔
  4 - Real.sqrt 21 < x ∧ x < 4 + Real.sqrt 21 :=
sorry

end inequality_solution_l167_167938


namespace snake_body_length_l167_167329

theorem snake_body_length (l h : ℝ) (h_head: h = l / 10) (h_length: l = 10) : l - h = 9 := 
by 
  rw [h_length, h_head] 
  norm_num
  sorry

end snake_body_length_l167_167329


namespace quadratic_no_real_roots_l167_167972

theorem quadratic_no_real_roots :
  ∀ (a b c : ℝ), (a = 1 ∧ b = 1 ∧ c = 2) → (b^2 - 4 * a * c < 0) := by
  intros a b c H
  cases H with Ha Hac
  cases Hac with Hb Hc
  rw [Ha, Hb, Hc]
  simp
  linarith

end quadratic_no_real_roots_l167_167972


namespace num_four_digit_with_5_or_7_l167_167593

def num_four_digit_without_5_or_7 : Nat := 7 * 8 * 8 * 8

def num_four_digit_total : Nat := 9999 - 1000 + 1

theorem num_four_digit_with_5_or_7 : 
  num_four_digit_total - num_four_digit_without_5_or_7 = 5416 :=
by
  sorry

end num_four_digit_with_5_or_7_l167_167593


namespace probability_exactly_9_heads_in_12_flips_l167_167461

noncomputable theory

open BigOperators

-- Define a function for combinations
def C(n k : ℕ) : ℕ := if h : k ≤ n then (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k))) else 0

-- The main Lean 4 statement
theorem probability_exactly_9_heads_in_12_flips :
  let total_outcomes := 2 ^ 12 in
  let favorable_outcomes := C 12 9 in
  favorable_outcomes.to_float / total_outcomes.to_float = 220 / 4096 :=
by
  let total_outcomes := 2 ^ 12
  let favorable_outcomes := C 12 9
  sorry

end probability_exactly_9_heads_in_12_flips_l167_167461


namespace first_person_work_days_l167_167152

theorem first_person_work_days (x : ℝ) (h1 : 0 < x) :
  (1/x + 1/40 = 1/15) → x = 24 :=
by
  intro h
  sorry

end first_person_work_days_l167_167152


namespace negation_of_existential_statement_l167_167649

variable (A : Set ℝ)

theorem negation_of_existential_statement :
  ¬(∃ x ∈ A, x^2 - 2 * x - 3 > 0) ↔ ∀ x ∈ A, x^2 - 2 * x - 3 ≤ 0 := by
  sorry

end negation_of_existential_statement_l167_167649


namespace cubic_difference_l167_167249

theorem cubic_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 59) : a^3 - b^3 = 448 :=
by
  sorry

end cubic_difference_l167_167249


namespace math_problem_l167_167710

noncomputable def f (x : ℝ) := (x^9 - 27*x^6 + 216*x^3 - 512) / (x^3 - 8)

theorem math_problem : f 6 = 43264 := by
  sorry

end math_problem_l167_167710


namespace ellipse_foci_y_axis_range_l167_167817

noncomputable def is_ellipse_with_foci_on_y_axis (k : ℝ) : Prop :=
  (k > 5) ∧ (k < 10) ∧ (10 - k > k - 5)

theorem ellipse_foci_y_axis_range (k : ℝ) :
  is_ellipse_with_foci_on_y_axis k ↔ 5 < k ∧ k < 7.5 := 
by
  sorry

end ellipse_foci_y_axis_range_l167_167817


namespace find_a_value_l167_167160

-- Define the problem conditions
theorem find_a_value (a : ℝ) :
  let x_values := [0, 1, 3, 4]
  let y_values := [a, 4.3, 4.8, 6.7]
  let mean_x := (0 + 1 + 3 + 4) / 4
  let mean_y := (a + 4.3 + 4.8 + 6.7) / 4
  (mean_y = 0.95 * mean_x + 2.6) → a = 2.2 :=
by
  -- Let bindings are for convenience to follow the problem statement
  let x_values := [0, 1, 3, 4]
  let y_values := [a, 4.3, 4.8, 6.7]
  let mean_x := (0 + 1 + 3 + 4) / 4
  let mean_y := (a + 4.3 + 4.8 + 6.7) / 4
  intro h
  sorry

end find_a_value_l167_167160


namespace string_cuts_l167_167448

theorem string_cuts (L S : ℕ) (h_diff : L - S = 48) (h_sum : L + S = 64) : 
  (L / S) = 7 :=
by
  sorry

end string_cuts_l167_167448


namespace max_area_of_rectangle_with_perimeter_60_l167_167100

theorem max_area_of_rectangle_with_perimeter_60 :
  ∃ (x y : ℝ), 2 * (x + y) = 60 ∧ x * y = 225 :=
by
  sorry

end max_area_of_rectangle_with_perimeter_60_l167_167100


namespace quadratic_roots_quadratic_roots_one_quadratic_roots_two_l167_167140

open scoped Classical

variables {p : Type*} [Field p] {a b c x : p}

theorem quadratic_roots (h_a : a ≠ 0) :
  (¬ ∃ y : p, y^2 = b^2 - 4 * a * c) → ∀ x : p, ¬ a * x^2 + b * x + c = 0 :=
by sorry

theorem quadratic_roots_one (h_a : a ≠ 0) :
  (b^2 - 4 * a * c = 0) → ∃ x : p, a * x^2 + b * x + c = 0 ∧ ∀ y : p, a * y^2 + b * y + c = 0 → y = x :=
by sorry

theorem quadratic_roots_two (h_a : a ≠ 0) :
  (∃ y : p, y^2 = b^2 - 4 * a * c) ∧ (b^2 - 4 * a * c ≠ 0) → ∃ x1 x2 : p, x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 :=
by sorry

end quadratic_roots_quadratic_roots_one_quadratic_roots_two_l167_167140


namespace circle_radius_zero_l167_167362

theorem circle_radius_zero (x y : ℝ) : 2*x^2 - 8*x + 2*y^2 + 4*y + 10 = 0 → (x - 2)^2 + (y + 1)^2 = 0 :=
by
  intro h
  sorry

end circle_radius_zero_l167_167362


namespace integer_solution_a_l167_167863

theorem integer_solution_a (a : ℤ) : 
  (∃ k : ℤ, 2 * a^2 = 7 * k + 2) ↔ (∃ ℓ : ℤ, a = 7 * ℓ + 1 ∨ a = 7 * ℓ - 1) :=
by
  sorry

end integer_solution_a_l167_167863


namespace total_water_intake_l167_167005

def theo_weekday := 8
def mason_weekday := 7
def roxy_weekday := 9
def zara_weekday := 10
def lily_weekday := 6

def theo_weekend := 10
def mason_weekend := 8
def roxy_weekend := 11
def zara_weekend := 12
def lily_weekend := 7

def total_cups_in_week (weekday_cups weekend_cups : ℕ) : ℕ :=
  5 * weekday_cups + 2 * weekend_cups

theorem total_water_intake :
  total_cups_in_week theo_weekday theo_weekend +
  total_cups_in_week mason_weekday mason_weekend +
  total_cups_in_week roxy_weekday roxy_weekend +
  total_cups_in_week zara_weekday zara_weekend +
  total_cups_in_week lily_weekday lily_weekend = 296 :=
by sorry

end total_water_intake_l167_167005


namespace circle_center_line_condition_l167_167380

theorem circle_center_line_condition (a : ℝ) :
    (∀ x y : ℝ, x^2 + y^2 - 2 * a * x + 4 * y - 6 = 0 → (a, -2) = (x, y) → x + 2 * y + 1 = 0) → a = 3 :=
by
  sorry

end circle_center_line_condition_l167_167380


namespace solution_set_inequality_l167_167893

open Real

theorem solution_set_inequality (f : ℝ → ℝ) (h1 : f e = 0) (h2 : ∀ x > 0, x * deriv f x < 2) :
    ∀ x, 0 < x → x ≤ e → f x + 2 ≥ 2 * log x :=
by
  sorry

end solution_set_inequality_l167_167893


namespace cheddar_cheese_slices_l167_167786

-- Define the conditions
def cheddar_slices (C : ℕ) := ∃ (packages : ℕ), packages * C = 84
def swiss_slices := 28
def randy_bought_same_slices (C : ℕ) := swiss_slices = 28 ∧ 84 = 84

-- Lean theorem statement to prove the number of slices per package of cheddar cheese equals 28.
theorem cheddar_cheese_slices {C : ℕ} (h1 : cheddar_slices C) (h2 : randy_bought_same_slices C) : C = 28 :=
sorry

end cheddar_cheese_slices_l167_167786


namespace jack_pages_l167_167125

theorem jack_pages (pages_per_booklet : ℕ) (num_booklets : ℕ) (h1 : pages_per_booklet = 9) (h2 : num_booklets = 49) : num_booklets * pages_per_booklet = 441 :=
by {
  sorry
}

end jack_pages_l167_167125


namespace largest_polygon_is_E_l167_167698

def area (num_unit_squares num_right_triangles num_half_squares: ℕ) : ℚ :=
  num_unit_squares + num_right_triangles * 0.5 + num_half_squares * 0.25

def polygon_A_area := area 3 2 0
def polygon_B_area := area 4 1 0
def polygon_C_area := area 2 4 2
def polygon_D_area := area 5 0 0
def polygon_E_area := area 3 3 4

theorem largest_polygon_is_E :
  polygon_E_area > polygon_A_area ∧ 
  polygon_E_area > polygon_B_area ∧ 
  polygon_E_area > polygon_C_area ∧ 
  polygon_E_area > polygon_D_area :=
by
  sorry

end largest_polygon_is_E_l167_167698


namespace polygon_coloring_l167_167912

open Nat

theorem polygon_coloring
  (n m : ℕ)
  (hn : n ≥ 3) (hm : 2 ≤ m ≤ n) :
  (choose n m) * (2^m + (-1)^m * 2) = 
  sorry

end polygon_coloring_l167_167912


namespace a_b_sum_of_powers_l167_167420

variable (a b : ℝ)

-- Conditions
def condition1 := a + b = 1
def condition2 := a^2 + b^2 = 3
def condition3 := a^3 + b^3 = 4
def condition4 := a^4 + b^4 = 7
def condition5 := a^5 + b^5 = 11

-- Theorem statement
theorem a_b_sum_of_powers (h1 : condition1 a b) (h2 : condition2 a b) (h3 : condition3 a b) 
  (h4 : condition4 a b) (h5 : condition5 a b) : a^10 + b^10 = 123 :=
sorry

end a_b_sum_of_powers_l167_167420


namespace even_function_of_shift_sine_l167_167567

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := (x - 6)^2 * Real.sin (ω * x)

theorem even_function_of_shift_sine :
  ∃ ω : ℝ, (∀ x : ℝ, f x ω = f (-x) ω) → ω = π / 4 :=
by
  sorry

end even_function_of_shift_sine_l167_167567


namespace max_area_of_rectangular_pen_l167_167081

theorem max_area_of_rectangular_pen (P : ℕ) (hP : P = 60) : 
  ∃ A : ℕ, A = 225 ∧ (∀ x y : ℕ, 2 * (x + y) = P → x * y ≤ A) := 
by
  sorry

end max_area_of_rectangular_pen_l167_167081


namespace david_marks_in_physics_l167_167227

theorem david_marks_in_physics : 
  ∀ (P : ℝ), 
  let english := 72 
  let mathematics := 60 
  let chemistry := 62 
  let biology := 84 
  let average_marks := 62.6 
  let num_subjects := 5 
  let total_marks := average_marks * num_subjects 
  let known_marks := english + mathematics + chemistry + biology 
  total_marks - known_marks = P → P = 35 :=
by
  sorry

end david_marks_in_physics_l167_167227


namespace probability_of_queen_after_first_queen_l167_167691

-- Define the standard deck
def standard_deck : Finset (Fin 54) := Finset.univ

-- Define the event of drawing the first queen
def first_queen (deck : Finset (Fin 54)) : Prop := -- placeholder defining first queen draw
  sorry

-- Define the event of drawing a queen immediately after the first queen
def queen_after_first_queen (deck : Finset (Fin 54)) : Prop :=
  sorry

-- Define the probability of an event given a condition
noncomputable def probability (event : Prop) (condition : Prop) : ℚ :=
  sorry

-- Main theorem statement
theorem probability_of_queen_after_first_queen : probability 
  (queen_after_first_queen standard_deck) (first_queen standard_deck) = 2/27 :=
sorry

end probability_of_queen_after_first_queen_l167_167691


namespace days_at_sister_l167_167617

def total_days_vacation : ℕ := 21
def days_plane : ℕ := 2
def days_grandparents : ℕ := 5
def days_train : ℕ := 1
def days_brother : ℕ := 5
def days_car_to_sister : ℕ := 1
def days_bus_to_sister : ℕ := 1
def extra_days_due_to_time_zones : ℕ := 1
def days_bus_back : ℕ := 1
def days_car_back : ℕ := 1

theorem days_at_sister : 
  total_days_vacation - (days_plane + days_grandparents + days_train + days_brother + days_car_to_sister + days_bus_to_sister + extra_days_due_to_time_zones + days_bus_back + days_car_back) = 3 :=
by
  sorry

end days_at_sister_l167_167617


namespace shortest_total_distance_piglet_by_noon_l167_167821

-- Define the distances
def distance_fs : ℕ := 1300  -- Distance through the forest (Piglet to Winnie-the-Pooh)
def distance_pr : ℕ := 600   -- Distance (Piglet to Rabbit)
def distance_rw : ℕ := 500   -- Distance (Rabbit to Winnie-the-Pooh)

-- Define the total distance via Rabbit and via forest
def total_distance_rabbit_path : ℕ := distance_pr + distance_rw + distance_rw
def total_distance_forest_path : ℕ := distance_fs + distance_rw

-- Prove that shortest distance Piglet covers by noon
theorem shortest_total_distance_piglet_by_noon : 
  min (total_distance_forest_path) (total_distance_rabbit_path) = 1600 := by
  sorry

end shortest_total_distance_piglet_by_noon_l167_167821


namespace wake_up_time_l167_167642

-- Definition of the conversion ratio from normal minutes to metric minutes
def conversion_ratio := 36 / 25

-- Definition of normal minutes in a full day
def normal_minutes_in_day := 24 * 60

-- Definition of metric minutes in a full day
def metric_minutes_in_day := 10 * 100

-- Definition to convert normal time (6:36 AM) to normal minutes
def normal_minutes_from_midnight (h m : ℕ) := h * 60 + m

-- Converting normal minutes to metric minutes using the conversion ratio
def metric_minutes (normal_mins : ℕ) := (normal_mins / 36) * 25

-- Definition of the final metric time 2:75
def metric_time := (2 * 100 + 75)

-- Proving the final answer is 275
theorem wake_up_time : 100 * 2 + 10 * 7 + 5 = 275 := by
  sorry

end wake_up_time_l167_167642


namespace find_number_l167_167337

theorem find_number :
  ∃ x : ℤ, 27 * (x + 143) = 9693 ∧ x = 216 :=
by 
  existsi 216
  sorry

end find_number_l167_167337


namespace inverse_proportion_m_value_l167_167758

theorem inverse_proportion_m_value (m : ℝ) (x : ℝ) (h : y = (m - 2) * x ^ (m^2 - 5)) : 
  y is inverse_proportional_function → m = -2 :=
sorry

end inverse_proportion_m_value_l167_167758


namespace twenty_million_in_scientific_notation_l167_167431

/-- Prove that 20 million in scientific notation is 2 * 10^7 --/
theorem twenty_million_in_scientific_notation : 20000000 = 2 * 10^7 :=
by
  sorry

end twenty_million_in_scientific_notation_l167_167431


namespace betty_age_l167_167039

-- Define the constants and conditions
variables (A M B : ℕ)
variables (h1 : A = 2 * M) (h2 : A = 4 * B) (h3 : M = A - 8)

-- Define the theorem to prove Betty's age
theorem betty_age : B = 4 :=
by sorry

end betty_age_l167_167039


namespace part1_l167_167715

def is_Xn_function (n : ℝ) (f : ℝ → ℝ) : Prop :=
  ∃ x1 x2, x1 ≠ x2 ∧ f x1 = f x2 ∧ x1 + x2 = 2 * n

theorem part1 : is_Xn_function 0 (fun x => abs x) ∧ is_Xn_function (1/2) (fun x => x^2 - x) :=
by
  sorry

end part1_l167_167715


namespace lilly_daily_savings_l167_167927

-- Conditions
def days_until_birthday : ℕ := 22
def flowers_to_buy : ℕ := 11
def cost_per_flower : ℕ := 4

-- Definition we want to prove
def total_cost : ℕ := flowers_to_buy * cost_per_flower
def daily_savings : ℕ := total_cost / days_until_birthday

theorem lilly_daily_savings : daily_savings = 2 := by
  sorry

end lilly_daily_savings_l167_167927


namespace log_expression_value_l167_167711

theorem log_expression_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (ha1 : a ≠ 1) (hb1 : b ≠ 1) : 
  ((Real.log b / Real.log a) * (Real.log a / Real.log b))^2 = 1 := 
by 
  sorry

end log_expression_value_l167_167711


namespace probability_exactly_9_heads_l167_167493

theorem probability_exactly_9_heads :
  (Nat.choose 12 9) / (2 ^ 12 : ℚ) = 220 / 4096 := by
  sorry

end probability_exactly_9_heads_l167_167493


namespace goods_train_length_l167_167993

noncomputable def speed_kmh : ℕ := 72  -- Speed of the goods train in km/hr
noncomputable def platform_length : ℕ := 280  -- Length of the platform in meters
noncomputable def time_seconds : ℕ := 26  -- Time taken to cross the platform in seconds
noncomputable def speed_mps : ℤ := speed_kmh * 1000 / 3600 -- Speed of the goods train in meters/second

theorem goods_train_length : 20 * time_seconds = 280 + 240 :=
by
  sorry

end goods_train_length_l167_167993


namespace soccer_tournament_matches_l167_167643

theorem soccer_tournament_matches (x : ℕ) (h : 1 ≤ x) : (1 / 2 : ℝ) * x * (x - 1) = 45 := sorry

end soccer_tournament_matches_l167_167643


namespace solution_set_inequality_l167_167441

theorem solution_set_inequality (x : ℝ) : 
  x * (x - 1) ≥ x ↔ x ≤ 0 ∨ x ≥ 2 := 
sorry

end solution_set_inequality_l167_167441


namespace minimum_n_for_80_intersections_l167_167947

-- Define what an n-sided polygon is and define the intersection condition
def n_sided_polygon (n : ℕ) : Type := sorry -- definition of n-sided polygon

-- Define the condition when boundaries of two polygons intersect at exactly 80 points
def boundaries_intersect_at (P Q : n_sided_polygon n) (k : ℕ) : Prop := sorry -- definition of boundaries intersecting at exactly k points

theorem minimum_n_for_80_intersections (n : ℕ) :
  (∃ (P Q : n_sided_polygon n), boundaries_intersect_at P Q 80) → (n ≥ 10) :=
sorry

end minimum_n_for_80_intersections_l167_167947


namespace range_of_a_l167_167115

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬(x^2 - 2*x + 3 ≤ a^2 - 2*a - 1))
  ↔ (-1 < a ∧ a < 3) :=
by sorry

end range_of_a_l167_167115


namespace probability_of_9_heads_in_12_flips_l167_167507

theorem probability_of_9_heads_in_12_flips :
  (nat.choose 12 9) * (1 / (2 : ℝ)^12) = 55 / 1024 := 
by
  sorry

end probability_of_9_heads_in_12_flips_l167_167507


namespace max_marks_for_test_l167_167975

theorem max_marks_for_test (M : ℝ) (h1: (0.30 * M) = 180) : M = 600 :=
by
  sorry

end max_marks_for_test_l167_167975


namespace basketball_team_points_l167_167651

variable (a b x : ℕ)

theorem basketball_team_points (h1 : 2 * a = 3 * b) 
                             (h2 : x = a + 1)
                             (h3 : 2 * a + 3 * b + x = 61) : 
    x = 13 :=
by {
  sorry
}

end basketball_team_points_l167_167651


namespace simplify_expression_l167_167145

theorem simplify_expression (p : ℝ) : ((7 * p - 3) - 3 * p * 2) * 2 + (5 - 2 / 2) * (8 * p - 12) = 34 * p - 54 := by
  sorry

end simplify_expression_l167_167145


namespace triangle_is_right_l167_167389

-- Define the side lengths of the triangle
def a : ℕ := 3
def b : ℕ := 4
def c : ℕ := 5

-- Define a predicate to check if a triangle is right using Pythagorean theorem
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- The proof problem statement
theorem triangle_is_right : is_right_triangle a b c :=
sorry

end triangle_is_right_l167_167389


namespace find_f_2010_l167_167133

noncomputable def f (a b α β : ℝ) (x : ℝ) :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem find_f_2010 {a b α β : ℝ} (h : f a b α β 2009 = 5) : f a b α β 2010 = 3 :=
sorry

end find_f_2010_l167_167133


namespace unique_roots_of_system_l167_167857

theorem unique_roots_of_system {x y z : ℂ} 
  (h1 : x + y + z = 3) 
  (h2 : x^2 + y^2 + z^2 = 3) 
  (h3 : x^3 + y^3 + z^3 = 3) : 
  (x = 1 ∧ y = 1 ∧ z = 1) :=
sorry

end unique_roots_of_system_l167_167857


namespace geometric_series_sum_l167_167964

theorem geometric_series_sum : 
  let a := 1 
  let r := 2 
  let n := 11 
  let S_n := (a * (1 - r^n)) / (1 - r)
  S_n = 2047 := by
  -- The proof steps would normally go here.
  sorry

end geometric_series_sum_l167_167964


namespace book_area_correct_l167_167434

/-- Converts inches to centimeters -/
def inch_to_cm (inches : ℚ) : ℚ :=
  inches * 2.54

/-- The length of the book given a parameter x -/
def book_length (x : ℚ) : ℚ :=
  3 * x - 4

/-- The width of the book in inches -/
def book_width_in_inches : ℚ :=
  5 / 2

/-- The width of the book in centimeters -/
def book_width : ℚ :=
  inch_to_cm book_width_in_inches

/-- The area of the book given a parameter x -/
def book_area (x : ℚ) : ℚ :=
  book_length x * book_width

/-- Proof that the area of the book with x = 5 is 69.85 cm² -/
theorem book_area_correct : book_area 5 = 69.85 := by
  sorry

end book_area_correct_l167_167434


namespace probability_heads_in_9_of_12_flips_l167_167502

theorem probability_heads_in_9_of_12_flips :
  let total_flips := 12 in
  let required_heads := 9 in
  let total_outcomes := 2^total_flips in
  let specific_outcomes := Nat.choose total_flips required_heads in
  (specific_outcomes : ℚ) / total_outcomes = 220 / 4096 := 
by 
  sorry

end probability_heads_in_9_of_12_flips_l167_167502


namespace probability_of_moving_twice_vs_once_l167_167767

theorem probability_of_moving_twice_vs_once :
  let p := 1 / 4
  let q := 3 / 4
  let move_once := (4 * (p) * (q ^ 3))
  let move_twice := (6 * (p ^ 2) * (q ^ 2))
  move_twice / move_once = 1 / 2 :=
sorry

end probability_of_moving_twice_vs_once_l167_167767


namespace count_four_digit_numbers_with_5_or_7_l167_167585

def num_four_digit_numbers : Nat := 9000
def exclude_first_digit : Finset Nat := {1, 2, 3, 4, 6, 8, 9}  -- 7 options
def exclude_other_digits : Finset Nat := {0, 1, 2, 3, 4, 6, 8, 9}  -- 8 options
def excluded_numbers_count : Nat := exclude_first_digit.card * exclude_other_digits.card ^ 3  -- 3584
def included_numbers_count : Nat := num_four_digit_numbers - excluded_numbers_count  -- 5416

theorem count_four_digit_numbers_with_5_or_7 :
  included_numbers_count = 5416 :=
by
  sorry

end count_four_digit_numbers_with_5_or_7_l167_167585


namespace arun_completes_work_alone_in_70_days_l167_167849

def arun_days (A : ℕ) : Prop :=
  ∃ T : ℕ, (A > 0) ∧ (T > 0) ∧ 
           (∀ (work_done_by_arun_in_1_day work_done_by_tarun_in_1_day : ℝ),
            work_done_by_arun_in_1_day = 1 / A ∧
            work_done_by_tarun_in_1_day = 1 / T ∧
            (work_done_by_arun_in_1_day + work_done_by_tarun_in_1_day = 1 / 10) ∧
            (4 * (work_done_by_arun_in_1_day + work_done_by_tarun_in_1_day) = 4 / 10) ∧
            (42 * work_done_by_arun_in_1_day = 6 / 10) )

theorem arun_completes_work_alone_in_70_days : arun_days 70 :=
  sorry

end arun_completes_work_alone_in_70_days_l167_167849


namespace triangle_side_c_l167_167765

variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to the respective angles

-- Conditions given
variable (h1 : Real.tan A = 2 * Real.tan B)
variable (h2 : a^2 - b^2 = (1 / 3) * c)

-- The proof problem
theorem triangle_side_c (h1 : Real.tan A = 2 * Real.tan B) (h2 : a^2 - b^2 = (1 / 3) * c) : c = 1 :=
by sorry

end triangle_side_c_l167_167765


namespace exists_acute_triangle_side_lengths_l167_167619

-- Define the real numbers d_1, d_2, ..., d_12 in the interval (1, 12).
noncomputable def real_numbers_in_interval (d : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 12 → 1 < d n ∧ d n < 12

-- Define the condition for d_i, d_j, d_k to form an acute triangle
def forms_acuse_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

-- The main theorem statement
theorem exists_acute_triangle_side_lengths (d : ℕ → ℝ) (h : real_numbers_in_interval d) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ forms_acuse_triangle (d i) (d j) (d k) :=
sorry

end exists_acute_triangle_side_lengths_l167_167619


namespace faculty_reduction_l167_167524

theorem faculty_reduction (x : ℝ) (h1 : 0.75 * x = 195) : x = 260 :=
by sorry

end faculty_reduction_l167_167524


namespace probability_heads_exactly_9_of_12_l167_167467

noncomputable def bin_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_heads_9_of_12_flips : ℚ :=
  (bin_coeff 12 9 : ℚ) / (2 ^ 12)

theorem probability_heads_exactly_9_of_12 :
  probability_heads_9_of_12_flips = 220 / 4096 :=
by
  sorry

end probability_heads_exactly_9_of_12_l167_167467


namespace solution_correctness_l167_167404

noncomputable def problem_statement : ℕ :=
  let conditions := {z : ℂ // |z| = 1 ∧ (z ^ 720 - z ^ 120).im = 0}
  let N := conditions.count_fun (λ z, 1)
  N % 1000

theorem solution_correctness : problem_statement = 440 :=
by {
  -- proof omitted
  sorry
}

end solution_correctness_l167_167404


namespace most_cost_effective_way_cost_is_860_l167_167444

-- Definitions based on the problem conditions
def adult_cost := 150
def child_cost := 60
def group_cost_per_person := 100
def group_min_size := 5

-- Number of adults and children
def num_adults := 4
def num_children := 7

-- Calculate the total cost for the most cost-effective way
noncomputable def most_cost_effective_way_cost :=
  let group_tickets_count := 5  -- 4 adults + 1 child
  let remaining_children := num_children - 1
  group_tickets_count * group_cost_per_person + remaining_children * child_cost

-- Theorem to state the cost for the most cost-effective way
theorem most_cost_effective_way_cost_is_860 : most_cost_effective_way_cost = 860 := by
  sorry

end most_cost_effective_way_cost_is_860_l167_167444


namespace integer_values_of_b_for_quadratic_eqn_l167_167057

noncomputable def number_of_integer_values_of_b : ℕ := 16

theorem integer_values_of_b_for_quadratic_eqn :
  ∃(b : ℤ) (k ≥ 0), ∀m n : ℤ, (m + n = -b ∧ m * n = 12 * b) → (m + 12) * (n + 12) = 144 → k = number_of_integer_values_of_b := sorry

end integer_values_of_b_for_quadratic_eqn_l167_167057


namespace arrange_balls_l167_167450

/-- Given 4 yellow balls and 3 red balls, we want to prove that there are 35 different ways to arrange these balls in a row. -/
theorem arrange_balls : (Nat.choose 7 4) = 35 := by
  sorry

end arrange_balls_l167_167450


namespace solution_set_inequality_l167_167864

theorem solution_set_inequality (x : ℝ) :
  (x^2 - 4) * (x - 6)^2 ≤ 0 ↔ (-2 ≤ x ∧ x ≤ 2) ∨ (x = 6) := by
  sorry

end solution_set_inequality_l167_167864


namespace george_max_pencils_l167_167062

-- Define the conditions for the problem
def total_money : ℝ := 9.30
def pencil_cost : ℝ := 1.05
def discount_rate : ℝ := 0.10

-- Define the final statement to prove
theorem george_max_pencils (n : ℕ) :
  (n ≤ 8 ∧ pencil_cost * n ≤ total_money) ∨ 
  (n > 8 ∧ pencil_cost * (1 - discount_rate) * n ≤ total_money) →
  n ≤ 9 :=
by
  sorry

end george_max_pencils_l167_167062


namespace gcd_60_90_150_l167_167963

theorem gcd_60_90_150 : Nat.gcd (Nat.gcd 60 90) 150 = 30 := 
by
  sorry

end gcd_60_90_150_l167_167963


namespace num_different_configurations_of_lights_l167_167658

-- Definition of initial conditions
def num_rows : Nat := 6
def num_columns : Nat := 6
def possible_switch_states (n : Nat) : Nat := 2^n

-- Problem statement to be verified
theorem num_different_configurations_of_lights :
  let num_configurations := (possible_switch_states num_rows - 1) * (possible_switch_states num_columns - 1) + 1
  num_configurations = 3970 :=
by
  sorry

end num_different_configurations_of_lights_l167_167658


namespace sum_digits_of_3n_l167_167953

noncomputable def sum_digits (n : ℕ) : ℕ :=
sorry  -- Placeholder for a proper implementation of sum_digits

theorem sum_digits_of_3n (n : ℕ) 
  (h1 : sum_digits n = 100) 
  (h2 : sum_digits (44 * n) = 800) : 
  sum_digits (3 * n) = 300 := 
by
  sorry

end sum_digits_of_3n_l167_167953


namespace find_fourth_digit_l167_167212

theorem find_fourth_digit (a b c d : ℕ) (h : 0 ≤ a ∧ a < 8 ∧ 0 ≤ b ∧ b < 8 ∧ 0 ≤ c ∧ c < 8 ∧ 0 ≤ d ∧ d < 8)
  (h_eq : 511 * a + 54 * b - 92 * c - 999 * d = 0) : d = 6 :=
by
  sorry

end find_fourth_digit_l167_167212


namespace sequence_1005th_term_l167_167223

-- Definitions based on conditions
def first_term : ℚ := sorry
def second_term : ℚ := 10
def third_term : ℚ := 4 * first_term - (1:ℚ)
def fourth_term : ℚ := 4 * first_term + (1:ℚ)

-- Common difference
def common_difference : ℚ := (fourth_term - third_term)

-- Arithmetic sequence term calculation
def nth_term (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n-1) * d

-- Theorem statement
theorem sequence_1005th_term : nth_term first_term common_difference 1005 = 5480 := sorry

end sequence_1005th_term_l167_167223


namespace probability_exactly_9_heads_l167_167494

theorem probability_exactly_9_heads :
  (Nat.choose 12 9) / (2 ^ 12 : ℚ) = 220 / 4096 := by
  sorry

end probability_exactly_9_heads_l167_167494


namespace initial_money_l167_167034

theorem initial_money (B S G M : ℕ) 
  (hB : B = 8) 
  (hS : S = 2 * B) 
  (hG : G = 3 * S) 
  (change : ℕ) 
  (h_change : change = 28)
  (h_total : B + S + G + change = M) : 
  M = 100 := 
by 
  sorry

end initial_money_l167_167034


namespace probability_9_heads_12_flips_l167_167513

noncomputable def probability_of_9_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_9_heads_12_flips : probability_of_9_heads 12 9 (1 / 2) = 55 / 1024 := by
  sorry

end probability_9_heads_12_flips_l167_167513


namespace number_of_schools_l167_167264

-- Define the conditions
def is_median (a : ℕ) (n : ℕ) : Prop := 2 * a - 1 = n
def high_team_score (a b c : ℕ) : Prop := a > b ∧ a > c
def ranks (b c : ℕ) : Prop := b = 39 ∧ c = 67

-- Define the main problem
theorem number_of_schools (a n b c : ℕ) :
  is_median a n →
  high_team_score a b c →
  ranks b c →
  34 ≤ a ∧ a < 39 →
  2 * a ≡ 1 [MOD 3] →
  (n = 67 → a = 35) →
  (∀ m : ℕ, n = 3 * m + 1) →
  m = 23 :=
by
  sorry

end number_of_schools_l167_167264


namespace solve_system_eqns_l167_167713

theorem solve_system_eqns:
  ∃ x y : ℚ, (3 * x - 2 * y = 12) ∧ (9 * y - 6 * x = -18) ∧ (x = 24/5) ∧ (y = 6/5) := 
by
  use 24/5, 6/5
  simp
  split; norm_num; sorry

end solve_system_eqns_l167_167713


namespace set_intersections_l167_167108

open Set Nat

def I : Set ℕ := univ

def A : Set ℕ := { x | ∃ n, x = 3 * n ∧ ∃ k, n = 2 * k }

def B : Set ℕ := { y | ∃ m, y = m ∧ 24 % m = 0 }

theorem set_intersections :
  A ∩ B = {6, 12, 24} ∧ (I \ A) ∩ B = {1, 2, 3, 4, 8} :=
by
  sorry

end set_intersections_l167_167108


namespace beads_per_necklace_l167_167128

-- Definitions based on conditions
def total_beads_used (N : ℕ) : ℕ :=
  10 * N + 2 * N + 50 + 35

-- Main theorem to prove the number of beads needed for one beaded necklace
theorem beads_per_necklace (N : ℕ) (h : total_beads_used N = 325) : N = 20 :=
by
  sorry

end beads_per_necklace_l167_167128


namespace find_m_l167_167244

theorem find_m (m : ℝ) :
  (∃ x : ℝ, (m - 1) * x^2 + 2 * x + m^2 - 1 = 0) ∧ (m - 1 ≠ 0) → m = -1 :=
by
  sorry

end find_m_l167_167244


namespace number_of_lines_with_angle_greater_than_30_degrees_l167_167280

-- Definition of next integer points on the curve
def next_integer_points (x y : ℝ) : Prop :=
  ∃ (k : ℤ), x = k ∧ y = real.sqrt (9 - x^2)

-- Check if a point is on the curve
def on_curve (x y : ℝ) : Prop := y = real.sqrt (9 - x^2)

-- Calculate slope between two points
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  if p2.1 - p1.1 = 0 then 0 else (p2.2 - p1.2) / (p2.1 - p1.1)

-- Angles between points
def angle_greater_than_30_degrees (p1 p2 : ℝ × ℝ) : Prop :=
  abs (slope p1 p2) > real.tan (real.pi / 6)

theorem number_of_lines_with_angle_greater_than_30_degrees :
  ∃ (l : ℕ), l = 2 ∧
    ∀ (p1 p2 : ℝ × ℝ),
      (on_curve p1.1 p1.2) ∧ (on_curve p2.1 p2.2) ∧
      (int.floor p1.1 = p1.1) ∧ (int.floor p2.1 = p2.1) ∧
      (int.floor p1.2 = p1.2) ∧ (int.floor p2.2 = p2.2) →
      angle_greater_than_30_degrees p1 p2 :=
begin
  sorry
end

end number_of_lines_with_angle_greater_than_30_degrees_l167_167280


namespace parabola_intersection_points_l167_167346

open Finset

-- Defining the conditions given in the problem
def focus := (0 : ℝ, 0 : ℝ)

def a_values := {-3, -2, -1, 0, 1, 2, 3}
def b_values := {-4, -3, -2, -1, 1, 2, 3, 4}

def directrices (a b : ℤ) := { p | ∃ x : ℝ, p = (x, ((a : ℝ) * x + (b : ℝ))) }

-- Non-intersecting parallel directrices count
def non_intersecting_parallel_count := 84

/-- Proving the number of intersection points for the given parabolas conditions. -/
theorem parabola_intersection_points : 
  2 * (choose 40 2 - non_intersecting_parallel_count) = 1392 := 
by 
  sorry

end parabola_intersection_points_l167_167346


namespace total_puppies_is_74_l167_167188

-- Define the number of puppies adopted each week based on the given conditions
def number_of_puppies_first_week : Nat := 20
def number_of_puppies_second_week : Nat := 2 * number_of_puppies_first_week / 5
def number_of_puppies_third_week : Nat := 2 * number_of_puppies_second_week
def number_of_puppies_fourth_week : Nat := number_of_puppies_first_week + 10

-- Define the total number of puppies
def total_number_of_puppies : Nat :=
  number_of_puppies_first_week + number_of_puppies_second_week + number_of_puppies_third_week + number_of_puppies_fourth_week

-- Proof statement: Prove that the total number of puppies is 74
theorem total_puppies_is_74 : total_number_of_puppies = 74 := by
  sorry

end total_puppies_is_74_l167_167188


namespace digit_57_of_one_over_seventeen_is_2_l167_167518

def decimal_rep_of_one_over_seventeen : ℕ → ℕ :=
λ n, ([0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7, 6, 4, 7].cycle.take n).get n

theorem digit_57_of_one_over_seventeen_is_2 : decimal_rep_of_one_over_seventeen 57 = 2 :=
sorry

end digit_57_of_one_over_seventeen_is_2_l167_167518


namespace total_cost_750_candies_l167_167203

def candy_cost (candies : ℕ) (cost_per_box : ℕ) (candies_per_box : ℕ) (discount_threshold : ℕ) (discount_rate : ℝ) : ℝ :=
  let boxes := candies / candies_per_box
  let total_cost := boxes * cost_per_box
  if candies > discount_threshold then
    (1 - discount_rate) * total_cost
  else
    total_cost

theorem total_cost_750_candies :
  candy_cost 750 8 30 500 0.1 = 180 :=
by sorry

end total_cost_750_candies_l167_167203


namespace necessary_but_not_sufficient_condition_l167_167322

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (∀ a, a > 2 → a ∈ set.Ici 2) ∧ ¬(∃ a, a ∈ set.Ici 2 ∧ a ≤ 2) :=
by
  sorry

end necessary_but_not_sufficient_condition_l167_167322


namespace evaluate_statements_l167_167699

-- Defining what it means for angles to be vertical
def vertical_angles (α β : ℝ) : Prop := α = β

-- Defining what complementary angles are
def complementary (α β : ℝ) : Prop := α + β = 90

-- Defining what supplementary angles are
def supplementary (α β : ℝ) : Prop := α + β = 180

-- Define the geometric properties for perpendicular and parallel lines
def unique_perpendicular_through_point (l : ℝ → ℝ) (p : ℝ × ℝ): Prop :=
  ∃! m, ∀ x, m * x + p.2 = l x

def unique_parallel_through_point (l : ℝ → ℝ) (p : ℝ × ℝ): Prop :=
  ∃! m, ∀ x, (l x ≠ m * x + p.2) ∧ (∀ y, y ≠ p.2 → l y ≠ m * y)

theorem evaluate_statements :
  (¬ ∃ α β, α = β ∧ vertical_angles α β) ∧
  (¬ ∃ α β, supplementary α β ∧ complementary α β) ∧
  ∃ l p, unique_perpendicular_through_point l p ∧
  ∃ l p, unique_parallel_through_point l p →
  2 = 2
  :=
by
  sorry  -- Proof is omitted

end evaluate_statements_l167_167699


namespace rose_flyers_l167_167918

theorem rose_flyers (total_flyers made: ℕ) (flyers_jack: ℕ) (flyers_left: ℕ) 
(h1 : total_flyers = 1236)
(h2 : flyers_jack = 120)
(h3 : flyers_left = 796)
: total_flyers - flyers_jack - flyers_left = 320 :=
by
  sorry

end rose_flyers_l167_167918


namespace solve_linear_system_l167_167442

theorem solve_linear_system :
  ∃ (x y : ℝ), (x + 3 * y = -1) ∧ (2 * x + y = 3) ∧ (x = 2) ∧ (y = -1) :=
  sorry

end solve_linear_system_l167_167442


namespace parallel_lines_l167_167112

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, ax + 2 * y + 3 * a = 0 → 3 * x + (a - 1) * y = a - 7) → a = 3 :=
by sorry

end parallel_lines_l167_167112


namespace time_difference_alice_bob_l167_167694

theorem time_difference_alice_bob
  (alice_speed : ℕ) (bob_speed : ℕ) (distance : ℕ)
  (h_alice_speed : alice_speed = 7)
  (h_bob_speed : bob_speed = 9)
  (h_distance : distance = 12) :
  (bob_speed * distance - alice_speed * distance) = 24 :=
by
  sorry

end time_difference_alice_bob_l167_167694


namespace more_perfect_squares_with_7_digit_17th_l167_167181

noncomputable def seventeenth_digit (n : ℕ) : ℕ :=
  (n / 10^16) % 10

theorem more_perfect_squares_with_7_digit_17th
  (h_bound : ∀ n, n < 10^10 → (n * n) < 10^20)
  (h_representation : ∀ m, m < 10^20 → ∃ n, n < 10^10 ∧ m = n * n) :
  (∃ majority_digit_7 : ℕ,
    (∃ majority_digit_8 : ℕ,
      ∀ n, seventeenth_digit (n * n) = 7 → majority_digit_7 > majority_digit_8)
  ) :=
sorry

end more_perfect_squares_with_7_digit_17th_l167_167181


namespace yuri_total_puppies_l167_167186

-- Conditions
def first_week_puppies := 20
def second_week_puppies : ℕ := (2 / 5) * first_week_puppies
def third_week_puppies : ℕ := 2 * second_week_puppies
def fourth_week_puppies := first_week_puppies + 10

-- Total puppies
def total_puppies : ℕ := first_week_puppies + second_week_puppies + third_week_puppies + fourth_week_puppies

-- Theorem to prove
theorem yuri_total_puppies : total_puppies = 74 :=
by sorry

end yuri_total_puppies_l167_167186


namespace lisa_quiz_goal_l167_167622

theorem lisa_quiz_goal (total_quizzes earned_A_on_first earned_A_goal remaining_quizzes additional_A_needed max_quizzes_below_A : ℕ)
  (h1 : total_quizzes = 60)
  (h2 : earned_A_on_first = 30)
  (h3 : earned_A_goal = total_quizzes * 85 / 100)
  (h4 : remaining_quizzes = total_quizzes - 40)
  (h5 : additional_A_needed = earned_A_goal - earned_A_on_first)
  (h6 : max_quizzes_below_A = remaining_quizzes - additional_A_needed):
  max_quizzes_below_A = 0 :=
by sorry

end lisa_quiz_goal_l167_167622


namespace find_number_l167_167289

theorem find_number (n : ℕ) : gcd 30 n = 10 ∧ 70 ≤ n ∧ n ≤ 80 ∧ 200 ≤ lcm 30 n ∧ lcm 30 n ≤ 300 → (n = 70 ∨ n = 80) :=
sorry

end find_number_l167_167289


namespace probability_exactly_9_heads_in_12_flips_l167_167462

noncomputable theory

open BigOperators

-- Define a function for combinations
def C(n k : ℕ) : ℕ := if h : k ≤ n then (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k))) else 0

-- The main Lean 4 statement
theorem probability_exactly_9_heads_in_12_flips :
  let total_outcomes := 2 ^ 12 in
  let favorable_outcomes := C 12 9 in
  favorable_outcomes.to_float / total_outcomes.to_float = 220 / 4096 :=
by
  let total_outcomes := 2 ^ 12
  let favorable_outcomes := C 12 9
  sorry

end probability_exactly_9_heads_in_12_flips_l167_167462


namespace probability_of_region_C_l167_167837

theorem probability_of_region_C (pA pB pC : ℚ) 
  (h1 : pA = 1/2) 
  (h2 : pB = 1/5) 
  (h3 : pA + pB + pC = 1) : 
  pC = 3/10 := 
sorry

end probability_of_region_C_l167_167837


namespace tangent_lengths_l167_167302

noncomputable def internal_tangent_length (r1 r2 d : ℝ) : ℝ :=
  Real.sqrt (d^2 - (r1 + r2)^2)

noncomputable def external_tangent_length (r1 r2 d : ℝ) : ℝ :=
  Real.sqrt (d^2 - (r1 - r2)^2)

theorem tangent_lengths (r1 r2 d : ℝ) (h_r1 : r1 = 8) (h_r2 : r2 = 10) (h_d : d = 50) :
  internal_tangent_length r1 r2 d = 46.67 ∧ external_tangent_length r1 r2 d = 49.96 :=
by
  sorry

end tangent_lengths_l167_167302


namespace exists_100_digit_number_divisible_by_2_pow_100_l167_167144

theorem exists_100_digit_number_divisible_by_2_pow_100 :
  ∃ (n : ℕ), nat.digits 10 n = list.replicate 100 1 ∨ list.replicate 100 2 →
  n % 2^100 = 0 :=
sorry

end exists_100_digit_number_divisible_by_2_pow_100_l167_167144


namespace arithmetic_sequence_sum_l167_167967

theorem arithmetic_sequence_sum (x y : ℕ)
  (h₁ : ∃ d, 9 = 3 + d)  -- Common difference exists, d = 6
  (h₂ : ∃ n, 15 = 3 + n * 6)  -- Arithmetic sequence term verification
  (h₃ : y = 33 - 6)
  (h₄ : x = 27 - 6) : x + y = 48 :=
sorry

end arithmetic_sequence_sum_l167_167967


namespace total_students_is_100_l167_167395

-- Definitions of the conditions
def largest_class_students : Nat := 24
def decrement : Nat := 2

-- Let n be the number of classes, which is given by 5
def num_classes : Nat := 5

-- The number of students in each class
def students_in_class (n : Nat) : Nat := 
  if n = 1 then largest_class_students
  else largest_class_students - decrement * (n - 1)

-- Total number of students in the school
def total_students : Nat :=
  List.sum (List.map students_in_class (List.range num_classes))

-- Theorem to prove that total_students equals 100
theorem total_students_is_100 : total_students = 100 := by
  sorry

end total_students_is_100_l167_167395


namespace inequality_holds_for_a_in_interval_l167_167565

theorem inequality_holds_for_a_in_interval:
  (∀ x y : ℝ, 
     2 ≤ x ∧ x ≤ 3 ∧ 3 ≤ y ∧ y ≤ 4 → (3*x - 2*y - a) * (3*x - 2*y - a^2) ≤ 0) ↔ a ∈ Set.Iic (-4) :=
by
  sorry

end inequality_holds_for_a_in_interval_l167_167565


namespace courier_problem_l167_167840

variable (x : ℝ) -- Let x represent the specified time in minutes
variable (d : ℝ) -- Let d represent the total distance traveled in km

theorem courier_problem
  (h1 : 1.2 * (x - 10) = d)
  (h2 : 0.8 * (x + 5) = d) :
  x = 40 ∧ d = 36 :=
by
  -- This theorem statement encapsulates the conditions and the answer.
  sorry

end courier_problem_l167_167840


namespace max_rectangle_area_l167_167092

-- Define the perimeter and side lengths of the rectangle
def perimeter := 60
def L (x : ℝ) := x
def W (x : ℝ) := 30 - x

-- Define the area of the rectangle
def area (x : ℝ) := L x * W x

-- State the theorem to prove the maximum area is 225 square feet
theorem max_rectangle_area : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 30 ∧ area x = 225 :=
by
  sorry

end max_rectangle_area_l167_167092


namespace arithmetic_sequence_sum_l167_167969

theorem arithmetic_sequence_sum :
  ∀ (x y : ℤ), (∃ (n m : ℕ), (3 + n * 6 = x) ∧ (3 + m * 6 = y) ∧ x + 6 = y ∧ y + 6 = 33) → x + y = 60 :=
by
  intro x y h
  obtain ⟨n, m, hn, hm, hx, hy⟩ := h
  exact sorry

end arithmetic_sequence_sum_l167_167969


namespace arithmetic_sequence_sum_l167_167966

theorem arithmetic_sequence_sum (x y : ℕ)
  (h₁ : ∃ d, 9 = 3 + d)  -- Common difference exists, d = 6
  (h₂ : ∃ n, 15 = 3 + n * 6)  -- Arithmetic sequence term verification
  (h₃ : y = 33 - 6)
  (h₄ : x = 27 - 6) : x + y = 48 :=
sorry

end arithmetic_sequence_sum_l167_167966


namespace necessary_but_not_sufficient_l167_167741

variable {I : Set ℝ} (f : ℝ → ℝ) (M : ℝ)

theorem necessary_but_not_sufficient :
  (∀ x ∈ I, f x ≤ M) ↔
  (∀ x ∈ I, f x ≤ M ∧ (∃ x ∈ I, f x = M) → M = M ∧ ∃ x ∈ I, f x = M) :=
by
  sorry

end necessary_but_not_sufficient_l167_167741


namespace students_at_end_l167_167314

def initial_students := 11
def students_left := 6
def new_students := 42

theorem students_at_end (init : ℕ := initial_students) (left : ℕ := students_left) (new : ℕ := new_students) :
    (init - left + new) = 47 := 
by
  sorry

end students_at_end_l167_167314


namespace students_play_neither_l167_167392

-- Define the conditions
def total_students : ℕ := 39
def football_players : ℕ := 26
def long_tennis_players : ℕ := 20
def both_players : ℕ := 17

-- Define a theorem that states the equivalent proof problem
theorem students_play_neither : 
  total_students - (football_players + long_tennis_players - both_players) = 10 := by
  sorry

end students_play_neither_l167_167392


namespace max_product_of_slopes_l167_167014

theorem max_product_of_slopes 
  (m₁ m₂ : ℝ)
  (h₁ : m₂ = 3 * m₁)
  (h₂ : abs ((m₂ - m₁) / (1 + m₁ * m₂)) = Real.sqrt 3) :
  m₁ * m₂ ≤ 2 :=
sorry

end max_product_of_slopes_l167_167014


namespace probability_two_white_balls_same_color_l167_167445

theorem probability_two_white_balls_same_color :
  let num_white := 3
  let num_black := 2
  let total_combinations_white := num_white.choose 2
  let total_combinations_black := num_black.choose 2
  let total_combinations_same_color := total_combinations_white + total_combinations_black
  (total_combinations_white + total_combinations_black > 0) →
  (total_combinations_white / total_combinations_same_color) = (3 / 4) :=
by
  let num_white := 3
  let num_black := 2
  let total_combinations_white := num_white.choose 2
  let total_combinations_black := num_black.choose 2
  let total_combinations_same_color := total_combinations_white + total_combinations_black
  intro h
  sorry

end probability_two_white_balls_same_color_l167_167445


namespace translated_function_is_correct_l167_167172

-- Define the original function
def f (x : ℝ) : ℝ := (x - 2) ^ 2 + 2

-- Define the translated function after moving 1 unit to the left
def g (x : ℝ) : ℝ := f (x + 1)

-- Define the final function after moving 1 unit upward
def h (x : ℝ) : ℝ := g x + 1

-- The statement to be proved
theorem translated_function_is_correct :
  ∀ x : ℝ, h x = (x - 1) ^ 2 + 3 :=
by
  -- Proof goes here
  sorry

end translated_function_is_correct_l167_167172


namespace inequality_solution_interval_l167_167940

noncomputable def solve_inequality (x : ℝ) : Prop :=
  -2 < (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) ∧
  (x^2 - 4 * x + 5) ≠ 0 ∧
  (3 * x^2 - 24 * x + 25) / (x^2 - 4 * x + 5) > 0 ∧
  (- x^2 - 8 * x + 5) / (x^2 - 4 * x + 5) < 0

theorem inequality_solution_interval (x : ℝ) :
  solve_inequality x :=
sorry

end inequality_solution_interval_l167_167940


namespace probability_exactly_9_heads_l167_167497

theorem probability_exactly_9_heads :
  (Nat.choose 12 9) / (2 ^ 12 : ℚ) = 220 / 4096 := by
  sorry

end probability_exactly_9_heads_l167_167497


namespace ratio_c_b_l167_167102

theorem ratio_c_b (x y a b c : ℝ) (h1 : x ≥ 1) (h2 : x + y ≤ 4) (h3 : a * x + b * y + c ≤ 0) 
    (h_max : ∀ x y, (x,y) = (2, 2) → 2 * x + y = 6) (h_min : ∀ x y, (x,y) = (1, -1) → 2 * x + y = 1) (h_b : b ≠ 0) :
    c / b = 4 := sorry

end ratio_c_b_l167_167102


namespace find_angle_A_find_AB_l167_167609

theorem find_angle_A (A B C : ℝ) (h1 : 2 * Real.sin B * Real.cos A = Real.sin (A + C)) (h2 : A + B + C = Real.pi) :
  A = Real.pi / 3 := by
  sorry

theorem find_AB (A B C : ℝ) (AB BC AC : ℝ) (h1 : 2 * Real.sin B * Real.cos A = Real.sin (A + C))
  (h2 : BC = 2) (h3 : 1 / 2 * AB * AC * Real.sin (Real.pi / 3) = Real.sqrt 3)
  (h4 : A = Real.pi / 3) :
  AB = 2 := by
  sorry

end find_angle_A_find_AB_l167_167609


namespace mans_rate_in_still_water_l167_167974

theorem mans_rate_in_still_water
  (V_m V_s : ℝ)
  (h1 : V_m + V_s = 20)
  (h2 : V_m - V_s = 4) :
  V_m = 12 :=
by
  sorry

end mans_rate_in_still_water_l167_167974


namespace total_cost_correct_l167_167828

def sandwich_cost : ℝ := 2.44
def soda_cost : ℝ := 0.87
def num_sandwiches : ℕ := 2
def num_sodas : ℕ := 4

theorem total_cost_correct :
  (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost) = 8.36 := by
  sorry

end total_cost_correct_l167_167828


namespace sum_of_squares_l167_167156

theorem sum_of_squares (x y z : ℝ)
  (h1 : (x + y + z) / 3 = 10)
  (h2 : (xyz)^(1/3) = 6)
  (h3 : 3 / ((1/x) + (1/y) + (1/z)) = 4) : 
  x^2 + y^2 + z^2 = 576 := 
by
  sorry

end sum_of_squares_l167_167156


namespace num_bicycles_eq_20_l167_167447

-- Definitions based on conditions
def num_cars : ℕ := 10
def num_motorcycles : ℕ := 5
def total_wheels : ℕ := 90
def wheels_per_bicycle : ℕ := 2
def wheels_per_car : ℕ := 4
def wheels_per_motorcycle : ℕ := 2

-- Statement to prove
theorem num_bicycles_eq_20 (B : ℕ) 
  (h_wheels_from_bicycles : wheels_per_bicycle * B = 2 * B)
  (h_wheels_from_cars : num_cars * wheels_per_car = 40)
  (h_wheels_from_motorcycles : num_motorcycles * wheels_per_motorcycle = 10)
  (h_total_wheels : wheels_per_bicycle * B + 40 + 10 = total_wheels) :
  B = 20 :=
sorry

end num_bicycles_eq_20_l167_167447


namespace ferry_journey_time_difference_l167_167368

/-
  Problem statement:
  Prove that the journey of ferry Q is 1 hour longer than the journey of ferry P,
  given the following conditions:
  1. Ferry P travels for 3 hours at 6 kilometers per hour.
  2. Ferry Q takes a route that is two times longer than ferry P.
  3. Ferry P is slower than ferry Q by 3 kilometers per hour.
-/

theorem ferry_journey_time_difference :
  let speed_P := 6
  let time_P := 3
  let distance_P := speed_P * time_P
  let distance_Q := 2 * distance_P
  let speed_diff := 3
  let speed_Q := speed_P + speed_diff
  let time_Q := distance_Q / speed_Q
  time_Q - time_P = 1 :=
by
  sorry

end ferry_journey_time_difference_l167_167368


namespace gwen_more_money_from_mom_l167_167866

def dollars_received_from_mom : ℕ := 7
def dollars_received_from_dad : ℕ := 5

theorem gwen_more_money_from_mom :
  dollars_received_from_mom - dollars_received_from_dad = 2 :=
by
  sorry

end gwen_more_money_from_mom_l167_167866


namespace fixed_point_at_5_75_l167_167236

-- Defining the function
def quadratic_function (k : ℝ) (x : ℝ) : ℝ := 3 * x^2 + k * x - 5 * k

-- Stating the theorem that the graph passes through the fixed point (5, 75)
theorem fixed_point_at_5_75 (k : ℝ) : quadratic_function k 5 = 75 := by
  sorry

end fixed_point_at_5_75_l167_167236


namespace most_likely_number_of_cars_l167_167663

theorem most_likely_number_of_cars 
    (cars_in_first_10_seconds : ℕ := 6) 
    (time_for_first_10_seconds : ℕ := 10) 
    (total_time_seconds : ℕ := 165) 
    (constant_speed : Prop := true) : 
    ∃ (num_cars : ℕ), num_cars = 100 :=
by
  sorry

end most_likely_number_of_cars_l167_167663


namespace hyperbola_condition_l167_167443

theorem hyperbola_condition (k : ℝ) : 
  (-1 < k ∧ k < 1) ↔ (∃ x y : ℝ, (x^2 / (k-1) + y^2 / (k+1)) = 1) := 
sorry

end hyperbola_condition_l167_167443


namespace parallelogram_angle_A_l167_167614

theorem parallelogram_angle_A 
  (A B : ℝ) (h1 : A + B = 180) (h2 : A - B = 40) :
  A = 110 :=
by sorry

end parallelogram_angle_A_l167_167614


namespace max_area_of_rectangular_pen_l167_167075

-- Define the perimeter and derive the formula for the area
def perimeter := 60
def half_perimeter := perimeter / 2
def area (x : ℝ) := x * (half_perimeter - x)

-- Statement of the problem: prove the maximum area is 225 square feet
theorem max_area_of_rectangular_pen : ∃ x : ℝ, 0 ≤ x ∧ x ≤ half_perimeter ∧ area x = 225 := 
sorry

end max_area_of_rectangular_pen_l167_167075


namespace profit_percentage_is_22_percent_l167_167141

-- Define the given conditions
def scooter_cost (C : ℝ) := C
def repair_cost (C : ℝ) := 0.10 * C
def repair_cost_value := 500
def profit := 1100

-- Let's state the main theorem
theorem profit_percentage_is_22_percent (C : ℝ) 
  (h1 : repair_cost C = repair_cost_value)
  (h2 : profit = 1100) : 
  (profit / C) * 100 = 22 :=
by
  sorry

end profit_percentage_is_22_percent_l167_167141


namespace James_beat_old_record_by_296_points_l167_167269

def touchdowns_per_game := 4
def points_per_touchdown := 6
def number_of_games := 15
def two_point_conversions := 6
def points_per_two_point_conversion := 2
def field_goals := 8
def points_per_field_goal := 3
def extra_point_attempts := 20
def points_per_extra_point := 1
def consecutive_touchdowns := 3
def games_with_consecutive_touchdowns := 5
def bonus_multiplier := 2
def old_record := 300

def James_points : ℕ :=
  (touchdowns_per_game * number_of_games * points_per_touchdown) + 
  ((consecutive_touchdowns * games_with_consecutive_touchdowns) * points_per_touchdown * bonus_multiplier) +
  (two_point_conversions * points_per_two_point_conversion) +
  (field_goals * points_per_field_goal) +
  (extra_point_attempts * points_per_extra_point)

def points_above_old_record := James_points - old_record

theorem James_beat_old_record_by_296_points : points_above_old_record = 296 := by
  -- here would be the proof
  sorry

end James_beat_old_record_by_296_points_l167_167269


namespace max_value_of_m_l167_167604

open Real

theorem max_value_of_m : (∀ (x₁ x₂ : ℝ), x₁ < x₂ → x₂ < 0 → (x₂ * exp x₁ - x₁ * exp x₂) / (exp x₂ - exp x₁) > 1) → (∀ m, m ≥ 0 → ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ x₁ < m ∧ x₂ < m) :=
begin
  sorry
end

end max_value_of_m_l167_167604


namespace second_part_of_ratio_l167_167987

theorem second_part_of_ratio (h_ratio : ∀ (x : ℝ), 25 = 0.5 * (25 + x)) : ∃ x : ℝ, x = 25 :=
by
  sorry

end second_part_of_ratio_l167_167987


namespace ellipse_condition_range_k_l167_167742

theorem ellipse_condition_range_k (k : ℝ) : 
  (2 - k > 0) ∧ (3 + k > 0) ∧ (2 - k ≠ 3 + k) → -3 < k ∧ k < 2 := 
by 
  sorry

end ellipse_condition_range_k_l167_167742


namespace max_value_x_plus_y_l167_167925

theorem max_value_x_plus_y : ∀ (x y : ℝ), 
  (5 * x + 3 * y ≤ 9) → 
  (3 * x + 5 * y ≤ 11) → 
  x + y ≤ 32 / 17 :=
by
  intros x y h1 h2
  -- proof steps go here
  sorry

end max_value_x_plus_y_l167_167925


namespace quadratic_has_two_zeros_l167_167868

theorem quadratic_has_two_zeros {a b c : ℝ} (h : a * c < 0) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 :=
by
  sorry

end quadratic_has_two_zeros_l167_167868


namespace difference_of_digits_l167_167526

theorem difference_of_digits (p q : ℕ) (h1 : ∀ n, n < 100 → n ≥ 10 → ∀ m, m < 100 → m ≥ 10 → 9 * (p - q) = 9) : 
  p - q = 1 :=
sorry

end difference_of_digits_l167_167526


namespace total_puppies_is_74_l167_167190

-- Define the number of puppies adopted each week based on the given conditions
def number_of_puppies_first_week : Nat := 20
def number_of_puppies_second_week : Nat := 2 * number_of_puppies_first_week / 5
def number_of_puppies_third_week : Nat := 2 * number_of_puppies_second_week
def number_of_puppies_fourth_week : Nat := number_of_puppies_first_week + 10

-- Define the total number of puppies
def total_number_of_puppies : Nat :=
  number_of_puppies_first_week + number_of_puppies_second_week + number_of_puppies_third_week + number_of_puppies_fourth_week

-- Proof statement: Prove that the total number of puppies is 74
theorem total_puppies_is_74 : total_number_of_puppies = 74 := by
  sorry

end total_puppies_is_74_l167_167190


namespace smallest_number_is_111111_2_l167_167549

def base9_to_decimal (n : Nat) : Nat :=
  (n / 10) * 9 + (n % 10)

def base6_to_decimal (n : Nat) : Nat :=
  (n / 100) * 36 + ((n % 100) / 10) * 6 + (n % 10)

def base4_to_decimal (n : Nat) : Nat :=
  (n / 1000) * 64

def base2_to_decimal (n : Nat) : Nat :=
  (n / 100000) * 32 + ((n % 100000) / 10000) * 16 + ((n % 10000) / 1000) * 8 + ((n % 1000) / 100) * 4 + ((n % 100) / 10) * 2 + (n % 10)

theorem smallest_number_is_111111_2 :
  let n1 := base9_to_decimal 85
  let n2 := base6_to_decimal 210
  let n3 := base4_to_decimal 1000
  let n4 := base2_to_decimal 111111
  n4 < n1 ∧ n4 < n2 ∧ n4 < n3 := by
    sorry

end smallest_number_is_111111_2_l167_167549


namespace row_even_col_odd_contradiction_row_odd_col_even_contradiction_l167_167546

theorem row_even_col_odd_contradiction : 
  ¬ (∃ (M : Matrix (Fin 20) (Fin 15) ℕ), 
      (∀ r : Fin 20, ∃ i : Fin 15, M r i = 2) ∧ 
      (∀ c : Fin 15, ∀ j : Fin 20, M j c = 5)) := 
sorry

theorem row_odd_col_even_contradiction : 
  ¬ (∃ (M : Matrix (Fin 20) (Fin 15) ℕ), 
      (∀ r : Fin 20, ∀ i : Fin 15, M r i = 5) ∧ 
      (∀ c : Fin 15, ∃ j : Fin 20, M j c = 2)) := 
sorry

end row_even_col_odd_contradiction_row_odd_col_even_contradiction_l167_167546


namespace max_rectangle_area_l167_167091

-- Define the perimeter and side lengths of the rectangle
def perimeter := 60
def L (x : ℝ) := x
def W (x : ℝ) := 30 - x

-- Define the area of the rectangle
def area (x : ℝ) := L x * W x

-- State the theorem to prove the maximum area is 225 square feet
theorem max_rectangle_area : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 30 ∧ area x = 225 :=
by
  sorry

end max_rectangle_area_l167_167091


namespace pages_left_to_be_read_l167_167169

def total_pages : ℕ := 381
def pages_read : ℕ := 149
def pages_per_day : ℕ := 20
def days_in_week : ℕ := 7

theorem pages_left_to_be_read :
  total_pages - pages_read - (pages_per_day * days_in_week) = 92 := by
  sorry

end pages_left_to_be_read_l167_167169


namespace consecutive_numbers_expression_l167_167829

theorem consecutive_numbers_expression (x y z : ℤ) (h1 : x = y + 1) (h2 : z = y - 1) (h3 : z = 2) :
  2 * x + 3 * y + 3 * z = 8 * y - 1 :=
by
  -- substitute the conditions and simplify
  sorry

end consecutive_numbers_expression_l167_167829


namespace four_digit_integers_with_5_or_7_l167_167598

theorem four_digit_integers_with_5_or_7 : 
  let total := 9000 
  let valid_first_digits := [1, 2, 3, 4, 6, 8, 9]
  let valid_other_digits := [0, 1, 2, 3, 4, 6, 8, 9]
  let count_without_5_or_7 := 
      (list.length valid_first_digits) * 
      (list.length valid_other_digits) *
      (list.length valid_other_digits) *
      (list.length valid_other_digits)
  total - count_without_5_or_7 = 5416 := by
    let total := 9000
    let valid_first_digits := [1, 2, 3, 4, 6, 8, 9]
    let valid_other_digits := [0, 1, 2, 3, 4, 6, 8, 9]
    let count_without_5_or_7 := 
        (list.length valid_first_digits) * 
        (list.length valid_other_digits) *
        (list.length valid_other_digits) *
        (list.length valid_other_digits)
    show total - count_without_5_or_7 = 5416, from sorry

end four_digit_integers_with_5_or_7_l167_167598


namespace power_of_7_mod_10_l167_167017

theorem power_of_7_mod_10 (k : ℕ) (h : 7^4 ≡ 1 [MOD 10]) : 7^150 ≡ 9 [MOD 10] :=
sorry

end power_of_7_mod_10_l167_167017


namespace max_area_of_fenced_rectangle_l167_167068

theorem max_area_of_fenced_rectangle (x y : ℝ) (h : x + y = 30) : x * y ≤ 225 :=
by {
  sorry
}

end max_area_of_fenced_rectangle_l167_167068


namespace multiple_of_first_number_is_eight_l167_167630

theorem multiple_of_first_number_is_eight 
  (a b c k : ℤ)
  (h1 : a = 7) 
  (h2 : b = a + 2) 
  (h3 : c = b + 2) 
  (h4 : 7 * k = 3 * c + (2 * b + 5)) : 
  k = 8 :=
by
  sorry

end multiple_of_first_number_is_eight_l167_167630


namespace pyramid_bottom_right_value_l167_167432

theorem pyramid_bottom_right_value (a x y z b : ℕ) (h1 : 18 = (21 + x) / 2)
  (h2 : 14 = (21 + y) / 2) (h3 : 16 = (15 + z) / 2) (h4 : b = (21 + y) / 2) :
  a = 6 := 
sorry

end pyramid_bottom_right_value_l167_167432


namespace yangyang_helps_mom_for_5_days_l167_167654

-- Defining the conditions
def quantity_of_rice_in_warehouses_are_same : Prop := sorry
def dad_transports_all_rice_in : ℕ := 10
def mom_transports_all_rice_in : ℕ := 12
def yangyang_transports_all_rice_in : ℕ := 15
def dad_and_mom_start_at_same_time : Prop := sorry
def yangyang_helps_mom_then_dad : Prop := sorry
def finish_transporting_at_same_time : Prop := sorry

-- The theorem to prove
theorem yangyang_helps_mom_for_5_days (h1 : quantity_of_rice_in_warehouses_are_same) 
    (h2 : dad_and_mom_start_at_same_time) 
    (h3 : yangyang_helps_mom_then_dad) 
    (h4 : finish_transporting_at_same_time) : 
    yangyang_helps_mom_then_dad :=
sorry

end yangyang_helps_mom_for_5_days_l167_167654


namespace least_number_of_apples_l167_167624

theorem least_number_of_apples (b : ℕ) : (b % 3 = 2) → (b % 4 = 3) → (b % 5 = 1) → b = 11 :=
by
  intros h1 h2 h3
  sorry

end least_number_of_apples_l167_167624


namespace square_free_odd_integers_count_l167_167904

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m * m ∣ n → m = 1

def count_square_free_odd_integers (lower upper : ℕ) : ℕ :=
  (List.range' (lower + 1) (upper - lower - 1)).filter (λ n, n % 2 = 1 ∧ is_square_free n).length

theorem square_free_odd_integers_count : count_square_free_odd_integers 1 200 = 79 := 
by
  unfold count_square_free_odd_integers
  unfold is_square_free
  sorry

end square_free_odd_integers_count_l167_167904


namespace dyck_paths_length_2n_l167_167533

-- Define Dyck Path of length 2n
structure DyckPath (n : Nat) := 
(start : Fin (2*n+1) × Fin n) 
(length : Nat)
(directions : List (Bool))
(never_below_x_axis : ∀ i < length, (steps i).snd ≥ 0)

def count_dyck_paths (n : Nat) : Nat := 
∑ k in (Finset.range n), (Catalan k) * (Catalan (n - k - 1))

theorem dyck_paths_length_2n (n : Nat) : 
∑ k in (Finset.range n), Catalan k * Catalan (n - k - 1) = Nat.choose (2 * n)  n - Nat.choose (2*n) (n+1) :=
by
  sorry

end dyck_paths_length_2n_l167_167533


namespace students_in_section_B_l167_167167

variable (x : ℕ)

/-- There are 30 students in section A and the number of students in section B is x. The 
    average weight of section A is 40 kg, and the average weight of section B is 35 kg. 
    The average weight of the whole class is 38 kg. Prove that the number of students in
    section B is 20. -/
theorem students_in_section_B (h : 30 * 40 + x * 35 = 38 * (30 + x)) : x = 20 :=
  sorry

end students_in_section_B_l167_167167


namespace smallest_number_of_cubes_l167_167679

def box_length : ℕ := 49
def box_width : ℕ := 42
def box_depth : ℕ := 14
def gcd_box_dimensions : ℕ := Nat.gcd (Nat.gcd box_length box_width) box_depth

theorem smallest_number_of_cubes :
  (box_length / gcd_box_dimensions) *
  (box_width / gcd_box_dimensions) *
  (box_depth / gcd_box_dimensions) = 84 := by
  sorry

end smallest_number_of_cubes_l167_167679


namespace total_crayons_l167_167933

noncomputable def original_crayons : ℝ := 479.0
noncomputable def additional_crayons : ℝ := 134.0

theorem total_crayons : original_crayons + additional_crayons = 613.0 := by
  sorry

end total_crayons_l167_167933


namespace possible_atomic_numbers_l167_167653

/-
Given the following conditions:
1. An element X is from Group IIA and exhibits a +2 charge.
2. An element Y is from Group VIIA and exhibits a -1 charge.
Prove that the possible atomic numbers for elements X and Y that can form an ionic compound with the formula XY₂ are 12 for X and 9 for Y.
-/

structure Element :=
  (atomic_number : Nat)
  (group : Nat)
  (charge : Int)

def GroupIIACharge := 2
def GroupVIIACharge := -1

axiom X : Element
axiom Y : Element

theorem possible_atomic_numbers (X_group_IIA : X.group = 2)
                                (X_charge : X.charge = GroupIIACharge)
                                (Y_group_VIIA : Y.group = 7)
                                (Y_charge : Y.charge = GroupVIIACharge) :
  (X.atomic_number = 12) ∧ (Y.atomic_number = 9) :=
sorry

end possible_atomic_numbers_l167_167653


namespace average_people_moving_l167_167399

theorem average_people_moving (days : ℕ) (total_people : ℕ) 
    (h_days : days = 5) (h_total_people : total_people = 3500) : 
    (total_people / days) = 700 :=
by
  sorry

end average_people_moving_l167_167399


namespace original_price_l167_167312

theorem original_price (P : ℝ) (h : 0.75 * (0.75 * P) = 17) : P = 30.22 :=
by
  sorry

end original_price_l167_167312


namespace prove_ellipse_and_sum_constant_l167_167400

-- Define the ellipse properties
def ellipse_center_origin (a b : ℝ) : Prop :=
  a = 4 ∧ b^2 = 12

-- Standard equation of the ellipse
def ellipse_standard_eqn (x y : ℝ) : Prop :=
  (x^2 / 16) + (y^2 / 12) = 1

-- Define the conditions for m and n given point M(1, 3)
def condition_m_n (m n : ℝ) (x0 : ℝ) : Prop :=
  (9 * m^2 + 96 * m + 48 - (13/4) * x0^2 = 0) ∧ (9 * n^2 + 96 * n + 48 - (13/4) * x0^2 = 0)

-- Prove the standard equation of the ellipse and m+n constant properties
theorem prove_ellipse_and_sum_constant (a b x y m n x0 : ℝ) 
  (h1 : ellipse_center_origin a b)
  (h2 : ellipse_standard_eqn x y)
  (h3 : condition_m_n m n x0) :
  m + n = -32/3 := 
sorry

end prove_ellipse_and_sum_constant_l167_167400


namespace area_of_triangle_l167_167274

open Matrix

def a : Matrix (Fin 2) (Fin 1) ℤ := ![![4], ![-1]]
def b : Matrix (Fin 2) (Fin 1) ℤ := ![![3], ![5]]

theorem area_of_triangle : (abs (a 0 0 * b 1 0 - a 1 0 * b 0 0) : ℚ) / 2 = 23 / 2 :=
by
  -- To be proved (using :ℚ for the cast to rational for division)
  sorry

end area_of_triangle_l167_167274


namespace solve_fiftieth_term_l167_167440

variable (a₇ a₂₁ : ℤ) (d : ℚ)

-- The conditions stated in the problem
def seventh_term : a₇ = 10 := by sorry
def twenty_first_term : a₂₁ = 34 := by sorry

-- The fifty term calculation assuming the common difference d
def fiftieth_term_is_fraction (d : ℚ) : ℚ := 10 + 43 * d

-- Translate the condition a₂₁ = a₇ + 14 * d
theorem solve_fiftieth_term : a₂₁ = a₇ + 14 * d → 
                              fiftieth_term_is_fraction d = 682 / 7 := by sorry


end solve_fiftieth_term_l167_167440


namespace cost_of_fencing_irregular_pentagon_l167_167353

noncomputable def total_cost_fencing (AB BC CD DE AE : ℝ) (cost_per_meter : ℝ) : ℝ := 
  (AB + BC + CD + DE + AE) * cost_per_meter

theorem cost_of_fencing_irregular_pentagon :
  total_cost_fencing 20 25 30 35 40 2 = 300 := 
by
  sorry

end cost_of_fencing_irregular_pentagon_l167_167353


namespace only_one_statement_is_true_l167_167629

theorem only_one_statement_is_true (A B C D E: Prop)
  (hA : A ↔ B)
  (hB : B ↔ ¬ E)
  (hC : C ↔ (A ∧ B ∧ C ∧ D ∧ E))
  (hD : D ↔ ¬ (A ∨ B ∨ C ∨ D ∨ E))
  (hE : E ↔ ¬ A)
  (h_unique : ∃! x, x = A ∨ x = B ∨ x = C ∨ x = D ∨ x = E ∧ x = True) : E :=
by
  sorry

end only_one_statement_is_true_l167_167629


namespace scientific_notation_of_twenty_million_l167_167429

-- Define the number 20 million
def twenty_million : ℂ :=
  20000000

-- Define the scientific notation to be proved correct
def scientific_notation : ℂ :=
  2 * 10 ^ 7

-- The theorem to prove the equivalence
theorem scientific_notation_of_twenty_million : twenty_million = scientific_notation :=
  sorry

end scientific_notation_of_twenty_million_l167_167429


namespace rita_needs_9_months_l167_167117

def total_required_hours : ℕ := 4000
def backstroke_hours : ℕ := 100
def breaststroke_hours : ℕ := 40
def butterfly_hours : ℕ := 320
def monthly_practice_hours : ℕ := 400

def hours_already_completed : ℕ := backstroke_hours + breaststroke_hours + butterfly_hours
def remaining_hours : ℕ := total_required_hours - hours_already_completed
def months_needed : ℕ := (remaining_hours + monthly_practice_hours - 1) / monthly_practice_hours -- Ceiling division

theorem rita_needs_9_months :
  months_needed = 9 := by
  sorry

end rita_needs_9_months_l167_167117


namespace find_expression_value_l167_167736

theorem find_expression_value 
  (x y : ℝ) 
  (h1 : 4 * x + y = 10) 
  (h2 : x + 4 * y = 18) : 
  16 * x^2 + 24 * x * y + 16 * y^2 = 424 := 
by 
  sorry

end find_expression_value_l167_167736


namespace probability_heads_9_of_12_l167_167508

theorem probability_heads_9_of_12 :
  let total_outcomes := 2^12 in
  let favorable_outcomes := Nat.choose 12 9 in
  let probability := favorable_outcomes / total_outcomes.toRat in
  probability = 220 / 4096 := 
by
  sorry

end probability_heads_9_of_12_l167_167508


namespace probability_of_9_heads_in_12_l167_167476

def coin_flip_probability_9_heads_in_12_flips : Prop :=
  let total_outcomes := 2^12
  let success_outcomes := Nat.choose 12 9
  success_outcomes / total_outcomes = 220 / 4096

theorem probability_of_9_heads_in_12 : coin_flip_probability_9_heads_in_12_flips :=
  sorry

end probability_of_9_heads_in_12_l167_167476


namespace log_neg_inequality_l167_167762

theorem log_neg_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  Real.log (-a) > Real.log (-b) := 
sorry

end log_neg_inequality_l167_167762


namespace additional_increment_charge_cents_l167_167682

-- Conditions as definitions
def first_increment_charge_cents : ℝ := 3.10
def total_charge_8_minutes_cents : ℝ := 18.70
def total_minutes : ℝ := 8
def increments_per_minute : ℝ := 5
def total_increments : ℝ := total_minutes * increments_per_minute
def remaining_increments : ℝ := total_increments - 1
def remaining_charge_cents : ℝ := total_charge_8_minutes_cents - first_increment_charge_cents

-- Proof problem: What is the charge for each additional 1/5 of a minute?
theorem additional_increment_charge_cents : remaining_charge_cents / remaining_increments = 0.40 := by
  sorry

end additional_increment_charge_cents_l167_167682


namespace cube_surface_area_of_same_volume_as_prism_l167_167686

theorem cube_surface_area_of_same_volume_as_prism :
  let prism_length := 10
  let prism_width := 5
  let prism_height := 24
  let prism_volume := prism_length * prism_width * prism_height
  let cube_edge := (prism_volume : ℝ)^(1/3)
  let cube_surface_area := 6 * cube_edge^2
  cube_surface_area = 677.76 := by
  sorry

end cube_surface_area_of_same_volume_as_prism_l167_167686


namespace probability_nine_heads_l167_167470

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

def flips : ℕ := 12
def heads : ℕ := 9
def total_outcomes : ℕ := 2^flips
def favorable_outcomes : ℕ := binom flips heads
def probability : Rat := favorable_outcomes / total_outcomes.toRat

theorem probability_nine_heads :
  probability = (220/4096) := by
  sorry

end probability_nine_heads_l167_167470


namespace upper_limit_of_x_l167_167608

theorem upper_limit_of_x 
  {x : ℤ} 
  (h1 : 0 < x) 
  (h2 : x < 15) 
  (h3 : -1 < x) 
  (h4 : x < 5) 
  (h5 : 0 < x) 
  (h6 : x < 3) 
  (h7 : x + 2 < 4) 
  (h8 : x = 1) : 
  0 < x ∧ x < 2 := 
by 
  sorry

end upper_limit_of_x_l167_167608


namespace median_is_70_74_l167_167712

-- Define the histogram data as given
def histogram : List (ℕ × ℕ) :=
  [(85, 5), (80, 15), (75, 18), (70, 22), (65, 20), (60, 10), (55, 10)]

-- Function to calculate the cumulative sum at each interval
def cumulativeSum (hist : List (ℕ × ℕ)) : List (ℕ × ℕ) :=
  hist.scanl (λ acc pair => (pair.1, acc.2 + pair.2)) (0, 0)

-- Function to find the interval where the median lies
def medianInterval (hist : List (ℕ × ℕ)) : ℕ :=
  let cumSum := cumulativeSum hist
  -- The median is the 50th and 51st scores
  let medianPos := 50
  -- Find the interval that contains the median position
  List.find? (λ pair => medianPos ≤ pair.2) cumSum |>.getD (0, 0) |>.1

-- The theorem stating that the median interval is 70-74
theorem median_is_70_74 : medianInterval histogram = 70 :=
  by sorry

end median_is_70_74_l167_167712


namespace initial_water_percentage_l167_167678

noncomputable def S : ℝ := 4.0
noncomputable def V_initial : ℝ := 440
noncomputable def V_final : ℝ := 460
noncomputable def sugar_added : ℝ := 3.2
noncomputable def water_added : ℝ := 10
noncomputable def kola_added : ℝ := 6.8
noncomputable def kola_percentage : ℝ := 8.0 / 100.0
noncomputable def final_sugar_percentage : ℝ := 4.521739130434784 / 100.0

theorem initial_water_percentage : 
  ∀ (W S : ℝ),
  V_initial * (S / 100) + sugar_added = final_sugar_percentage * V_final →
  (W + 8.0 + S) = 100.0 →
  W = 88.0
:=
by
  intros W S h1 h2
  sorry

end initial_water_percentage_l167_167678


namespace solveEquation_l167_167971

noncomputable def findNonZeroSolution (z : ℝ) : Prop :=
  (5 * z) ^ 10 = (20 * z) ^ 5 ∧ z ≠ 0

theorem solveEquation : ∃ z : ℝ, findNonZeroSolution z ∧ z = 4 / 5 := by
  exists 4 / 5
  simp [findNonZeroSolution]
  sorry

end solveEquation_l167_167971


namespace find_n_l167_167262

theorem find_n
  (c d : ℝ)
  (H1 : 450 * c + 300 * d = 300 * c + 375 * d)
  (H2 : ∃ t1 t2 t3 : ℝ, t1 = 4 ∧ t2 = 1 ∧ t3 = n ∧ 75 * 4 * (c + d) = 900 * c + t3 * d)
  : n = 600 / 7 :=
by
  sorry

end find_n_l167_167262


namespace total_age_of_siblings_in_10_years_l167_167957

theorem total_age_of_siblings_in_10_years (age_eldest : ℕ) (gap : ℕ) (h1 : age_eldest = 20) (h2 : gap = 5) :
  let age_second := age_eldest - gap,
      age_youngest := age_second - gap in
  age_eldest + 10 + (age_second + 10) + (age_youngest + 10) = 75 :=
by
  sorry

end total_age_of_siblings_in_10_years_l167_167957


namespace probability_of_9_heads_in_12_flips_l167_167484

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| _, 0       => 1
| 0, _       => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^12

-- Define the number of ways to get exactly 9 heads in 12 flips
def favorable_outcomes : ℕ := binom 12 9

-- Define the probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 220/4096
theorem probability_of_9_heads_in_12_flips : probability = 220 / 4096 := by
  sorry

end probability_of_9_heads_in_12_flips_l167_167484


namespace abc_le_one_eighth_l167_167406

theorem abc_le_one_eighth (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h : a / (1 + a) + b / (1 + b) + c / (1 + c) = 1) : a * b * c ≤ 1 / 8 :=
by
  sorry

end abc_le_one_eighth_l167_167406


namespace ln_X_distribution_equality_l167_167621

noncomputable def gamma_const : ℝ := 0.5772 -- Euler-Mascheroni constant approximation

theorem ln_X_distribution_equality (α : ℝ) (X : ℝ) (Y : ℕ → ℝ) 
  (hX : X ~ Probability.distrib.gamma α 1) 
  (hY : ∀ n, Y n ~ Probability.distrib.exponential 1)
  (h_representation : ∀ n, 
    log X = (∑ i in Finset.range n, 
    log (X + (Finset.range (i - 1)).sum (λ k, Y k)) - 
    log (X + (Finset.range i).sum (λ k, Y k))) + 
    log (X + (Finset.range n).sum (λ i, Y i))) : 
  log X ~ -(gamma_const) + 
    ∑ n in Finset.range_natural, (1 / (n + 1) - Y n / (n + α)) :=
sorry

end ln_X_distribution_equality_l167_167621


namespace rectangular_solid_surface_area_l167_167559

noncomputable def is_prime (n : ℕ) : Prop := sorry

theorem rectangular_solid_surface_area (l w h : ℕ) (hl : is_prime l) (hw : is_prime w) (hh : is_prime h) (volume_eq_437 : l * w * h = 437) :
  2 * (l * w + w * h + h * l) = 958 :=
sorry

end rectangular_solid_surface_area_l167_167559


namespace number_of_boys_l167_167930

-- Definitions for the given conditions
def total_children := 60
def happy_children := 30
def sad_children := 10
def neither_happy_nor_sad_children := 20
def total_girls := 41
def happy_boys := 6
def sad_girls := 4
def neither_happy_nor_sad_boys := 7

-- Define the total number of boys
def total_boys := total_children - total_girls

-- Proof statement
theorem number_of_boys : total_boys = 19 :=
  by
    sorry

end number_of_boys_l167_167930


namespace paid_amount_divisible_by_11_l167_167118

-- Define the original bill amount and the increased bill amount
def original_bill (x : ℕ) : ℕ := x
def paid_amount (x : ℕ) : ℕ := (11 * x) / 10

-- Theorem: The paid amount is divisible by 11
theorem paid_amount_divisible_by_11 (x : ℕ) (h : x % 10 = 0) : paid_amount x % 11 = 0 :=
by
  sorry

end paid_amount_divisible_by_11_l167_167118


namespace functional_relationship_maximizing_profit_minimum_boxes_for_2000_profit_l167_167991

-- Definitions for the problem conditions
def cost_per_box : ℝ := 20
def min_selling_price : ℝ := 25
def max_selling_price : ℝ := 38
def base_sales_volume : ℝ := 250
def price_decrease_effect : ℝ := 10
def profit_requirement : ℝ := 2000

-- Given the initial conditions
noncomputable def sales_volume (x : ℝ) : ℝ := base_sales_volume - price_decrease_effect * (x - min_selling_price)

-- Target problem statement
-- Part 1: Functional relationship between y and x
theorem functional_relationship (x : ℝ) : sales_volume x = -10 * x + 500 := by
sorry

-- Part 2: Maximizing profit
noncomputable def profit (x : ℝ) : ℝ := (x - cost_per_box) * sales_volume x

theorem maximizing_profit : ∃ (x : ℝ), x = 35 ∧ profit x = 2250 := by
sorry

-- Part 3: Minimum number of boxes to sell for at least 2000 yuan profit
theorem minimum_boxes_for_2000_profit (x : ℝ) : x ≤ max_selling_price → profit x ≥ profit_requirement → sales_volume x ≥ 120 := by
sorry

end functional_relationship_maximizing_profit_minimum_boxes_for_2000_profit_l167_167991


namespace triangle_area_is_32_5_l167_167814

-- Define points A, B, and C
def A : ℝ × ℝ := (-3, 4)
def B : ℝ × ℝ := (1, 7)
def C : ℝ × ℝ := (4, -1)

-- Calculate the area directly using the determinant method for the area of a triangle given by coordinates
def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (
    A.1 * (B.2 - C.2) +
    B.1 * (C.2 - A.2) +
    C.1 * (A.2 - B.2)
  )

-- Define the statement to be proved
theorem triangle_area_is_32_5 : area_triangle A B C = 32.5 := 
  by
  -- proof to be filled in
  sorry

end triangle_area_is_32_5_l167_167814


namespace ellipse_equation_line_AC_l167_167287

noncomputable def ellipse_eq (x y a b : ℝ) : Prop := 
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def foci_distance (a c : ℝ) : Prop := 
  a - c = 1 ∧ a + c = 3

noncomputable def b_value (a c b : ℝ) : Prop :=
  b = Real.sqrt (a^2 - c^2)

noncomputable def rhombus_on_line (m : ℝ) : Prop := 
  7 * (2 * m / 7) + 1 - 7 * (3 * m / 7) = 0

theorem ellipse_equation (a b c : ℝ) (h1 : foci_distance a c) (h2 : b_value a c b) :
  ellipse_eq x y a b :=
sorry

theorem line_AC (a b c x y x1 y1 x2 y2 : ℝ) 
  (h1 : ellipse_eq x1 y1 a b)
  (h2 : ellipse_eq x2 y2 a b)
  (h3 : 7 * x1 - 7 * y1 + 1 = 0)
  (h4 : 7 * x2 - 7 * y2 + 1 = 0)
  (h5 : rhombus_on_line y) :
  x + y + 1 = 0 :=
sorry

end ellipse_equation_line_AC_l167_167287


namespace g_at_neg_1001_l167_167154

-- Defining the function g and the conditions
def g (x : ℝ) : ℝ := 2.5 * x - 0.5

-- Defining the main theorem to be proved
theorem g_at_neg_1001 : g (-1001) = -2503 := by
  sorry

end g_at_neg_1001_l167_167154


namespace fraction_addition_simplification_l167_167851

theorem fraction_addition_simplification :
  (2 / 5 : ℚ) + (3 / 15) = 3 / 5 :=
by
  sorry

end fraction_addition_simplification_l167_167851


namespace eight_sided_dice_probability_l167_167048

noncomputable def probability_even_eq_odd (n : ℕ) : ℚ :=
  if h : n % 2 = 0 then
    let k := n / 2 in
    (nat.choose n k : ℚ) * (1/2)^n
  else 0

theorem eight_sided_dice_probability :
  probability_even_eq_odd 8 = 35/128 :=
by trivial

end eight_sided_dice_probability_l167_167048


namespace four_digit_integers_with_5_or_7_l167_167600

theorem four_digit_integers_with_5_or_7 : 
  let total := 9000 
  let valid_first_digits := [1, 2, 3, 4, 6, 8, 9]
  let valid_other_digits := [0, 1, 2, 3, 4, 6, 8, 9]
  let count_without_5_or_7 := 
      (list.length valid_first_digits) * 
      (list.length valid_other_digits) *
      (list.length valid_other_digits) *
      (list.length valid_other_digits)
  total - count_without_5_or_7 = 5416 := by
    let total := 9000
    let valid_first_digits := [1, 2, 3, 4, 6, 8, 9]
    let valid_other_digits := [0, 1, 2, 3, 4, 6, 8, 9]
    let count_without_5_or_7 := 
        (list.length valid_first_digits) * 
        (list.length valid_other_digits) *
        (list.length valid_other_digits) *
        (list.length valid_other_digits)
    show total - count_without_5_or_7 = 5416, from sorry

end four_digit_integers_with_5_or_7_l167_167600


namespace quadratic_root_property_l167_167375

theorem quadratic_root_property (m n : ℝ)
  (hmn : m^2 + m - 2021 = 0)
  (hn : n^2 + n - 2021 = 0) :
  m^2 + 2 * m + n = 2020 :=
by sorry

end quadratic_root_property_l167_167375


namespace proportion_of_segments_l167_167376

theorem proportion_of_segments
  (a b c d : ℝ)
  (h1 : b = 3)
  (h2 : c = 4)
  (h3 : d = 6)
  (h4 : a / b = c / d) :
  a = 2 :=
by
  sorry

end proportion_of_segments_l167_167376


namespace remainder_of_x50_div_x_minus_1_cubed_l167_167730

theorem remainder_of_x50_div_x_minus_1_cubed :
  (x : ℝ) → (x ^ 50) % ((x - 1) ^ 3) = 1225 * x ^ 2 - 2400 * x + 1176 := 
by
  sorry

end remainder_of_x50_div_x_minus_1_cubed_l167_167730


namespace relationship_among_abc_l167_167781

noncomputable def a : ℝ := (1/4)^(1/2)
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := (1/3)^(1/2)

theorem relationship_among_abc : b > c ∧ c > a :=
by
  -- Proof will go here
  sorry

end relationship_among_abc_l167_167781


namespace dilation_image_l167_167647

theorem dilation_image 
  (z z₀ : ℂ) (k : ℝ) 
  (hz : z = -2 + i) 
  (hz₀ : z₀ = 1 - 3 * I) 
  (hk : k = 3) : 
  (k * (z - z₀) + z₀) = (-8 + 9 * I) := 
by 
  rw [hz, hz₀, hk]
  -- Sorry means here we didn't write the complete proof, we assume it is correct.
  sorry

end dilation_image_l167_167647


namespace find_replaced_weight_l167_167946

-- Define the conditions and the hypothesis
def replaced_weight (W : ℝ) : Prop :=
  let avg_increase := 2.5
  let num_persons := 8
  let new_weight := 85
  (new_weight - W) = num_persons * avg_increase

-- Define the statement we aim to prove
theorem find_replaced_weight : replaced_weight 65 :=
by
  -- proof goes here
  sorry

end find_replaced_weight_l167_167946


namespace semi_circle_radius_l167_167825

theorem semi_circle_radius (π : ℝ) (hπ : Real.pi = π) (P : ℝ) (hP : P = 180) : 
  ∃ r : ℝ, r = 180 / (π + 2) :=
by
  sorry

end semi_circle_radius_l167_167825


namespace pages_left_to_be_read_l167_167170

def total_pages : ℕ := 381
def pages_read : ℕ := 149
def pages_per_day : ℕ := 20
def days_in_week : ℕ := 7

theorem pages_left_to_be_read :
  total_pages - pages_read - (pages_per_day * days_in_week) = 92 := by
  sorry

end pages_left_to_be_read_l167_167170


namespace circle_diameter_C_l167_167854

theorem circle_diameter_C {D C : ℝ} (hD : D = 20) (h_ratio : (π * (D/2)^2 - π * (C/2)^2) / (π * (C/2)^2) = 4) : C = 4 * Real.sqrt 5 := 
sorry

end circle_diameter_C_l167_167854


namespace probability_heads_9_of_12_flips_l167_167491

theorem probability_heads_9_of_12_flips : (nat.choose 12 9) / (2^12) = 55 / 1024 :=
by
  sorry

end probability_heads_9_of_12_flips_l167_167491


namespace two_digit_numbers_condition_l167_167305

theorem two_digit_numbers_condition :
  ∃ (x y : ℕ), x > y ∧ x < 100 ∧ y < 100 ∧ x - y = 56 ∧ (x^2 % 100) = (y^2 % 100) ∧
  ((x = 78 ∧ y = 22) ∨ (x = 22 ∧ y = 78)) :=
by sorry

end two_digit_numbers_condition_l167_167305


namespace average_of_consecutive_integers_l167_167234

theorem average_of_consecutive_integers (a b : ℤ) (h1 : b = (a + (a+1) + (a+2) + (a+3) + (a+4)) / 5) :
  let b := a + 2 in
  let avg_of_b := ((a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6)) / 5 in
  avg_of_b = a + 4 :=
by
  sorry

end average_of_consecutive_integers_l167_167234


namespace max_area_of_fenced_rectangle_l167_167069

theorem max_area_of_fenced_rectangle (x y : ℝ) (h : x + y = 30) : x * y ≤ 225 :=
by {
  sorry
}

end max_area_of_fenced_rectangle_l167_167069


namespace hours_spent_writing_l167_167774

-- Define the rates at which Jacob and Nathan write
def Nathan_rate : ℕ := 25        -- Nathan writes 25 letters per hour
def Jacob_rate : ℕ := 2 * Nathan_rate  -- Jacob writes twice as fast as Nathan

-- Define the combined rate
def combined_rate : ℕ := Nathan_rate + Jacob_rate

-- Define the total letters written and the hours spent
def total_letters : ℕ := 750
def hours_spent : ℕ := total_letters / combined_rate

-- The theorem to prove
theorem hours_spent_writing : hours_spent = 10 :=
by 
  -- Placeholder for the proof
  sorry

end hours_spent_writing_l167_167774


namespace probability_class_4_drawn_first_second_l167_167538

noncomputable def P_1 : ℝ := 1 / 10
noncomputable def P_2 : ℝ := 9 / 100

theorem probability_class_4_drawn_first_second :
  P_1 = 1 / 10 ∧ P_2 = 9 / 100 := by
  sorry

end probability_class_4_drawn_first_second_l167_167538


namespace call_processing_ratio_l167_167839

variables (A B C : ℝ)
variable (total_calls : ℝ)
variable (calls_processed_by_A_per_member calls_processed_by_B_per_member : ℝ)

-- Given conditions
def team_A_agents_ratio : Prop := A = (5 / 8) * B
def team_B_calls_ratio : Prop := calls_processed_by_B_per_member * B = (4 / 7) * total_calls
def team_A_calls_ratio : Prop := calls_processed_by_A_per_member * A = (3 / 7) * total_calls

-- Proving the ratio of calls processed by each member
theorem call_processing_ratio
    (hA : team_A_agents_ratio A B)
    (hB_calls : team_B_calls_ratio B total_calls calls_processed_by_B_per_member)
    (hA_calls : team_A_calls_ratio A total_calls calls_processed_by_A_per_member) :
  calls_processed_by_A_per_member / calls_processed_by_B_per_member = 6 / 5 :=
by
  sorry

end call_processing_ratio_l167_167839


namespace average_speed_l167_167791

theorem average_speed (d1 d2 t1 t2 : ℝ) (h1 : d1 = 50) (h2 : d2 = 20) (h3 : t1 = 50 / 20) (h4 : t2 = 20 / 40) :
  ((d1 + d2) / (t1 + t2)) = 23.33 := 
  sorry

end average_speed_l167_167791


namespace exponent_product_l167_167064

variables {a m n : ℝ}

theorem exponent_product (h1 : a^m = 2) (h2 : a^n = 8) : a^m * a^n = 16 :=
by
  sorry

end exponent_product_l167_167064


namespace total_puppies_adopted_l167_167193

-- Define the number of puppies adopted each week
def first_week_puppies : ℕ := 20
def second_week_puppies : ℕ := (2 / 5) * first_week_puppies
def third_week_puppies : ℕ := 2 * second_week_puppies
def fourth_week_puppies : ℕ := 10 + first_week_puppies

-- Prove that the total number of puppies adopted over the month is 74
theorem total_puppies_adopted : 
  first_week_puppies + second_week_puppies + third_week_puppies + fourth_week_puppies = 74 := by
  sorry

end total_puppies_adopted_l167_167193


namespace max_area_of_rectangle_with_perimeter_60_l167_167101

theorem max_area_of_rectangle_with_perimeter_60 :
  ∃ (x y : ℝ), 2 * (x + y) = 60 ∧ x * y = 225 :=
by
  sorry

end max_area_of_rectangle_with_perimeter_60_l167_167101


namespace combined_salaries_of_ABCD_l167_167003

theorem combined_salaries_of_ABCD 
  (A B C D E : ℝ)
  (h1 : E = 9000)
  (h2 : (A + B + C + D + E) / 5 = 8600) :
  A + B + C + D = 34000 := 
sorry

end combined_salaries_of_ABCD_l167_167003


namespace range_of_x_l167_167052

theorem range_of_x (x : ℝ) (h1 : 2 ≤ |x - 5|) (h2 : |x - 5| ≤ 10) (h3 : 0 < x) : 
  (0 < x ∧ x ≤ 3) ∨ (7 ≤ x ∧ x ≤ 15) := 
sorry

end range_of_x_l167_167052


namespace min_period_k_l167_167635

def has_period {α : Type*} [HasZero α] (r : α) (n : ℕ) : Prop :=
  -- A function definition to check if 'r' has a repeating decimal period of length 'n'
  sorry

theorem min_period_k (a b : ℚ) (h₁ : has_period a 30) (h₂ : has_period b 30) (h₃ : has_period (a - b) 15) :
  ∃ (k : ℕ), k = 6 ∧ has_period (a + k * b) 15 :=
begin
  sorry
end

end min_period_k_l167_167635


namespace avg_calculation_l167_167944

-- Define averages
def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem avg_calculation : avg3 (avg3 2 2 0) (avg2 0 2) 0 = 7 / 9 :=
  by
    sorry

end avg_calculation_l167_167944


namespace rectangle_area_l167_167542

variable (w l A P : ℝ)
variable (h1 : l = w + 6)
variable (h2 : A = w * l)
variable (h3 : P = 2 * (w + l))
variable (h4 : A = 2 * P)
variable (h5 : w = 3)

theorem rectangle_area
  (w l A P : ℝ)
  (h1 : l = w + 6)
  (h2 : A = w * l)
  (h3 : P = 2 * (w + l))
  (h4 : A = 2 * P)
  (h5 : w = 3) :
  A = 27 := 
sorry

end rectangle_area_l167_167542


namespace rectangle_width_l167_167036

theorem rectangle_width
  (l w : ℕ)
  (h1 : l * w = 1638)
  (h2 : 10 * l = 390) :
  w = 42 :=
by
  sorry

end rectangle_width_l167_167036


namespace evaluate_expression_l167_167556

theorem evaluate_expression :
  -5^2 + 2 * (-3)^2 - (-8) / (-1 + 1/3) = -13 :=
by 
  sorry

end evaluate_expression_l167_167556


namespace c_work_rate_l167_167823

variable {W : ℝ} -- Denoting the work by W
variable {a_rate : ℝ} -- Work rate of a
variable {b_rate : ℝ} -- Work rate of b
variable {c_rate : ℝ} -- Work rate of c
variable {combined_rate : ℝ} -- Combined work rate of a, b, and c

theorem c_work_rate (W a_rate b_rate c_rate combined_rate : ℝ)
  (h1 : a_rate = W / 12)
  (h2 : b_rate = W / 24)
  (h3 : combined_rate = W / 4)
  (h4 : combined_rate = a_rate + b_rate + c_rate) :
  c_rate = W / 4.5 :=
by
  sorry

end c_work_rate_l167_167823


namespace expression_eval_l167_167613

theorem expression_eval (a b c d : ℝ) :
  a * b + c - d = a * (b + c - d) :=
sorry

end expression_eval_l167_167613


namespace inverse_function_point_l167_167754

theorem inverse_function_point (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  ∃ (A : ℝ × ℝ), A = (2, 3) ∧ ∀ y, (∀ x, y = a^(x-3) + 1) → (2, 3) ∈ {(y, x) | y = a^(x-3) + 1} :=
by
  sorry

end inverse_function_point_l167_167754


namespace Isabel_total_problems_l167_167266

theorem Isabel_total_problems :
  let math_pages := 2
  let reading_pages := 4
  let science_pages := 3
  let history_pages := 1
  let problems_per_math_page := 5
  let problems_per_reading_page := 5
  let problems_per_science_page := 7
  let problems_per_history_page := 10
  let total_math_problems := math_pages * problems_per_math_page
  let total_reading_problems := reading_pages * problems_per_reading_page
  let total_science_problems := science_pages * problems_per_science_page
  let total_history_problems := history_pages * problems_per_history_page
  let total_problems := total_math_problems + total_reading_problems + total_science_problems + total_history_problems
  total_problems = 61 := by
  sorry

end Isabel_total_problems_l167_167266


namespace problem1_problem2_problem3_l167_167042

-- Problem (1)
theorem problem1 : -36 * (5 / 4 - 5 / 6 - 11 / 12) = 18 := by
  sorry

-- Problem (2)
theorem problem2 : (-2) ^ 2 - 3 * (-1) ^ 3 + 0 * (-2) ^ 3 = 7 := by
  sorry

-- Problem (3)
theorem problem3 (x : ℚ) (y : ℚ) (h1 : x = -2) (h2 : y = 1 / 2) : 
    (3 / 2) * x^2 * y + x * y^2 = 5 / 2 := by
  sorry

end problem1_problem2_problem3_l167_167042


namespace greatest_pq_plus_r_l167_167562

theorem greatest_pq_plus_r (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) 
  (h : p * q + q * r + r * p = 2016) : 
  pq + r ≤ 1008 :=
sorry

end greatest_pq_plus_r_l167_167562


namespace find_s_l167_167778

-- Define the roots of the quadratic equation
variables (a b n r s : ℝ)

-- Conditions from Vieta's formulas
def condition1 : Prop := a + b = n
def condition2 : Prop := a * b = 3

-- Roots of the second quadratic equation
def condition3 : Prop := (a + 1 / b) * (b + 1 / a) = s

-- The theorem statement
theorem find_s
  (h1 : condition1 a b n)
  (h2 : condition2 a b)
  (h3 : condition3 a b s) :
  s = 16 / 3 :=
by
  sorry

end find_s_l167_167778


namespace real_numbers_satisfy_relation_l167_167050

theorem real_numbers_satisfy_relation (a b : ℝ) :
  2 * (a^2 + 1) * (b^2 + 1) = (a + 1) * (b + 1) * (a * b + 1) → 
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = -1) :=
by
  sorry

end real_numbers_satisfy_relation_l167_167050


namespace cookie_sheet_perimeter_l167_167317

theorem cookie_sheet_perimeter :
  let width_in_inches := 15.2
  let length_in_inches := 3.7
  let conversion_factor := 2.54
  let width_in_cm := width_in_inches * conversion_factor
  let length_in_cm := length_in_inches * conversion_factor
  2 * (width_in_cm + length_in_cm) = 96.012 :=
by
  sorry

end cookie_sheet_perimeter_l167_167317


namespace relationship_among_a_b_c_l167_167782

noncomputable def a : ℝ := Real.log (Real.tan (70 * Real.pi / 180)) / Real.log (1 / 2)
noncomputable def b : ℝ := Real.log (Real.sin (25 * Real.pi / 180)) / Real.log (1 / 2)
noncomputable def c : ℝ := (1 / 2) ^ Real.cos (25 * Real.pi / 180)

theorem relationship_among_a_b_c : a < c ∧ c < b :=
by
  -- proofs would go here
  sorry

end relationship_among_a_b_c_l167_167782


namespace find_a1_l167_167544

theorem find_a1 (a : ℕ → ℝ) (h_rec : ∀ n, a (n + 1) = a n / (2 * a n + 1)) (h_init : a 3 = 1 / 5) : a 1 = 1 := by
  sorry

end find_a1_l167_167544


namespace problem1_problem2_l167_167675

-- Problem (I)
theorem problem1 (α : ℝ) (h1 : Real.tan α = 3) :
  (4 * Real.sin (Real.pi - α) - 2 * Real.cos (-α)) / (3 * Real.cos (Real.pi / 2 - α) - 5 * Real.cos (Real.pi + α)) = 5 / 7 := by
sorry

-- Problem (II)
theorem problem2 (x : ℝ) (h2 : Real.sin x + Real.cos x = 1 / 5) (h3 : 0 < x ∧ x < Real.pi) :
  Real.sin x = 4 / 5 ∧ Real.cos x = -3 / 5 := by
sorry

end problem1_problem2_l167_167675


namespace carter_drum_stick_sets_l167_167708

theorem carter_drum_stick_sets (sets_per_show sets_tossed_per_show nights : ℕ) :
  sets_per_show = 5 →
  sets_tossed_per_show = 6 →
  nights = 30 →
  (sets_per_show + sets_tossed_per_show) * nights = 330 := by
  intros
  sorry

end carter_drum_stick_sets_l167_167708


namespace point_p_final_position_l167_167785

theorem point_p_final_position :
  let P_start := -2
  let P_right := P_start + 5
  let P_final := P_right - 4
  P_final = -1 :=
by
  sorry

end point_p_final_position_l167_167785


namespace coeff_of_x6_in_expansion_l167_167229

theorem coeff_of_x6_in_expansion : 
  (∃ (c : ℤ), c = (finset.range 7).sum (λ r, (((-1)^r) * (nat.choose 6 r) * (1 : ℤ) ^ (6 - r))) ∧ c = -20) :=
begin
  sorry
end

end coeff_of_x6_in_expansion_l167_167229


namespace prob_X_eq_Y_l167_167696

theorem prob_X_eq_Y : 
  ∀ (x y : ℝ), -12 * Real.pi ≤ x ∧ x ≤ 12 * Real.pi ∧ -12 * Real.pi ≤ y ∧ y ≤ 12 * Real.pi ∧ cos (cos x) = cos (cos y) → 
    probability (X = Y) = 25/169 :=
by 
  sorry

end prob_X_eq_Y_l167_167696


namespace digits_with_five_or_seven_is_5416_l167_167595

/-- The total number of four-digit positive integers. -/
def total_four_digit_integers : ℕ := 9000

/-- The number of four-digit integers without the digits 5 or 7. -/
def no_five_seven_integers : ℕ := 3584

/-- The number of four-digit positive integers that have at least one digit that is a 5 or a 7. -/
def digits_with_five_or_seven : ℕ :=
  total_four_digit_integers - no_five_seven_integers

/-- Proof that the number of four-digit integers with at least one digit that is a 5 or 7 is 5416. -/
theorem digits_with_five_or_seven_is_5416 :
  digits_with_five_or_seven = 5416 :=
by
  sorry

end digits_with_five_or_seven_is_5416_l167_167595


namespace black_ink_cost_l167_167142

theorem black_ink_cost (B : ℕ) 
  (h1 : 2 * B + 3 * 15 + 2 * 13 = 50 + 43) : B = 11 :=
by
  sorry

end black_ink_cost_l167_167142


namespace sum_of_squares_of_rates_l167_167726

variable (b j s : ℕ)

theorem sum_of_squares_of_rates
  (h1 : 3 * b + 2 * j + 3 * s = 82)
  (h2 : 5 * b + 3 * j + 2 * s = 99) :
  b^2 + j^2 + s^2 = 314 := by
  sorry

end sum_of_squares_of_rates_l167_167726


namespace Donny_spends_28_on_Thursday_l167_167724

theorem Donny_spends_28_on_Thursday :
  let monday_savings := 15
  let tuesday_savings := 28
  let wednesday_savings := 13
  let total_savings := monday_savings + tuesday_savings + wednesday_savings
  let thursday_spending := total_savings / 2
  thursday_spending = 28 :=
by 
  let monday_savings := 15
  let tuesday_savings := 28
  let wednesday_savings := 13
  let total_savings := monday_savings + tuesday_savings + wednesday_savings
  let thursday_spending := total_savings / 2
  sorry

end Donny_spends_28_on_Thursday_l167_167724


namespace arithmetic_seq_max_sum_l167_167372

noncomputable def max_arith_seq_sum_lemma (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * a1 + (n * (n - 1) / 2) * d

theorem arithmetic_seq_max_sum :
  ∀ (a1 d : ℤ),
    (3 * a1 + 6 * d = 9) →
    (a1 + 5 * d = -9) →
    max_arith_seq_sum_lemma a1 d 3 = 21 :=
by
  sorry

end arithmetic_seq_max_sum_l167_167372


namespace distance_y_axis_l167_167913

def point_M (m : ℝ) : ℝ × ℝ := (2 - m, 1 + 2 * m)

theorem distance_y_axis :
  ∀ m : ℝ, abs (2 - m) = 2 → (point_M m = (2, 1)) ∨ (point_M m = (-2, 9)) :=
by
  sorry

end distance_y_axis_l167_167913


namespace line_equation_passing_through_points_l167_167257

theorem line_equation_passing_through_points 
  (a₁ b₁ a₂ b₂ : ℝ)
  (h1 : 2 * a₁ + 3 * b₁ + 1 = 0)
  (h2 : 2 * a₂ + 3 * b₂ + 1 = 0)
  (h3 : ∀ (x y : ℝ), (x, y) = (2, 3) → a₁ * x + b₁ * y + 1 = 0 ∧ a₂ * x + b₂ * y + 1 = 0) :
  (∀ (x y : ℝ), (2 * x + 3 * y + 1 = 0) ↔ 
                (a₁ = x ∧ b₁ = y) ∨ (a₂ = x ∧ b₂ = y)) :=
by
  sorry

end line_equation_passing_through_points_l167_167257


namespace smallest_k_for_min_period_15_l167_167633

/-- Rational number with minimal period -/
def is_minimal_period (r : ℚ) (n : ℕ) : Prop :=
  ∃ m : ℤ, r = m / (10^n - 1)

variables (a b : ℚ)

-- Conditions for a and b
axiom ha : is_minimal_period a 30
axiom hb : is_minimal_period b 30

-- Condition for a - b
axiom hab_min_period : is_minimal_period (a - b) 15

-- Conclusion
theorem smallest_k_for_min_period_15 : ∃ k : ℕ, k = 6 ∧ is_minimal_period (a + k * b) 15 :=
by sorry

end smallest_k_for_min_period_15_l167_167633


namespace part_1_part_2_part_3_l167_167881

variable {f : ℝ → ℝ}

axiom C1 : ∀ x y : ℝ, f (x + y) = f x + f y
axiom C2 : ∀ x : ℝ, x > 0 → f x < 0
axiom C3 : f 3 = -4

theorem part_1 : f 0 = 0 :=
by
  sorry

theorem part_2 : ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

theorem part_3 : ∀ x : ℝ, -9 ≤ x ∧ x ≤ 9 → f x ≤ 12 ∧ f x ≥ -12 :=
by
  sorry

end part_1_part_2_part_3_l167_167881


namespace subtraction_from_double_result_l167_167028

theorem subtraction_from_double_result (x : ℕ) (h : 2 * x = 18) : x - 4 = 5 :=
sorry

end subtraction_from_double_result_l167_167028


namespace second_negative_integer_l167_167603

theorem second_negative_integer (n : ℤ) (h : -11 * n + 5 = 93) : n = -8 :=
by
  sorry

end second_negative_integer_l167_167603


namespace geometric_sequence_a2_l167_167908

theorem geometric_sequence_a2 (a1 a2 a3 : ℝ) (h1 : 1 * (1/a1) = a1)
  (h2 : a1 * (1/a2) = a2) (h3 : a2 * (1/a3) = a3) (h4 : a3 * (1/4) = 4)
  (h5 : a2 > 0) : a2 = 2 := sorry

end geometric_sequence_a2_l167_167908


namespace probability_heads_exactly_9_of_12_l167_167465

noncomputable def bin_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_heads_9_of_12_flips : ℚ :=
  (bin_coeff 12 9 : ℚ) / (2 ^ 12)

theorem probability_heads_exactly_9_of_12 :
  probability_heads_9_of_12_flips = 220 / 4096 :=
by
  sorry

end probability_heads_exactly_9_of_12_l167_167465


namespace neg_sin_leq_1_l167_167574

theorem neg_sin_leq_1 :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x : ℝ, Real.sin x > 1 :=
by
  sorry

end neg_sin_leq_1_l167_167574


namespace square_side_length_l167_167799

theorem square_side_length (d : ℝ) (h : d = 2 * Real.sqrt 2) : ∃ s : ℝ, s = 2 :=
by 
  sorry

end square_side_length_l167_167799


namespace inequality_proof_l167_167569

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : b + d < a + c :=
sorry

end inequality_proof_l167_167569


namespace frisbee_price_l167_167545

theorem frisbee_price 
  (total_frisbees : ℕ)
  (frisbees_at_3 : ℕ)
  (price_x_frisbees : ℕ)
  (total_revenue : ℕ) 
  (min_frisbees_at_x : ℕ)
  (price_at_3 : ℕ) 
  (n_min_at_x : ℕ)
  (h1 : total_frisbees = 60)
  (h2 : price_at_3 = 3)
  (h3 : total_revenue = 200)
  (h4 : n_min_at_x = 20)
  (h5 : min_frisbees_at_x >= n_min_at_x)
  : price_x_frisbees = 4 :=
by
  sorry

end frisbee_price_l167_167545


namespace inequality_solution_set_l167_167806

theorem inequality_solution_set (x : ℝ) :
  ∀ x, 
  (x^2 * (x + 1) / (-x^2 - 5 * x + 6) <= 0) ↔ (-6 < x ∧ x <= -1) ∨ (x = 0) ∨ (1 < x) :=
by
  sorry

end inequality_solution_set_l167_167806


namespace probability_of_9_heads_in_12_flips_l167_167504

theorem probability_of_9_heads_in_12_flips :
  (nat.choose 12 9) * (1 / (2 : ℝ)^12) = 55 / 1024 := 
by
  sorry

end probability_of_9_heads_in_12_flips_l167_167504


namespace range_of_a_l167_167894

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (h₁ : ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ ≥ f x₂)
                    (h₂ : -2 ≤ a + 1 ∧ a + 1 ≤ 4)
                    (h₃ : -2 ≤ 2 * a ∧ 2 * a ≤ 4)
                    (h₄ : f (a + 1) > f (2 * a)) : 1 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l167_167894


namespace complex_number_solution_l167_167890

theorem complex_number_solution (i : ℂ) (h : i^2 = -1) : (5 / (2 - i) - i = 2) :=
  sorry

end complex_number_solution_l167_167890


namespace cat_total_birds_caught_l167_167536

theorem cat_total_birds_caught (day_birds night_birds : ℕ) 
  (h1 : day_birds = 8) 
  (h2 : night_birds = 2 * day_birds) :
  day_birds + night_birds = 24 :=
sorry

end cat_total_birds_caught_l167_167536


namespace sequence_property_l167_167855

theorem sequence_property
  (b : ℝ) (h₀ : b > 0)
  (u : ℕ → ℝ)
  (h₁ : u 1 = b)
  (h₂ : ∀ n ≥ 1, u (n + 1) = 1 / (2 - u n)) :
  u 10 = (4 * b - 3) / (6 * b - 5) :=
by
  sorry

end sequence_property_l167_167855


namespace cost_of_calf_l167_167208

theorem cost_of_calf (C : ℝ) (total_cost : ℝ) (cow_to_calf_ratio : ℝ) :
  total_cost = 990 ∧ cow_to_calf_ratio = 8 ∧ total_cost = C + 8 * C → C = 110 := by
  sorry

end cost_of_calf_l167_167208


namespace afternoon_to_morning_ratio_l167_167543

theorem afternoon_to_morning_ratio
  (A : ℕ) (M : ℕ)
  (h1 : A = 340)
  (h2 : A + M = 510) :
  A / M = 2 :=
by
  sorry

end afternoon_to_morning_ratio_l167_167543


namespace height_min_surface_area_l167_167277

def height_of_box (x : ℝ) : ℝ := x + 4

def surface_area (x : ℝ) : ℝ := 2 * x^2 + 4 * x * (x + 4)

theorem height_min_surface_area :
  ∀ x : ℝ, surface_area x ≥ 150 → x ≥ 5 → height_of_box x = 9 :=
by
  intros x h1 h2
  sorry

end height_min_surface_area_l167_167277


namespace net_hourly_rate_correct_l167_167841

noncomputable def net_hourly_rate
    (hours : ℕ) 
    (speed : ℕ) 
    (fuel_efficiency : ℕ) 
    (earnings_per_mile : ℝ) 
    (cost_per_gallon : ℝ) 
    (distance := speed * hours) 
    (gasoline_used := distance / fuel_efficiency) 
    (earnings := earnings_per_mile * distance) 
    (cost_of_gasoline := cost_per_gallon * gasoline_used) 
    (net_earnings := earnings - cost_of_gasoline) : ℝ :=
  net_earnings / hours

theorem net_hourly_rate_correct : 
  net_hourly_rate 3 45 25 0.6 1.8 = 23.76 := 
by 
  unfold net_hourly_rate
  norm_num
  sorry

end net_hourly_rate_correct_l167_167841


namespace range_of_a_l167_167607

theorem range_of_a {a : ℝ} (h1 : ∀ x : ℝ, x - a ≥ 0 → 2 * x - 10 < 0) :
  3 < a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_l167_167607


namespace quadrants_containing_points_l167_167165

theorem quadrants_containing_points (x y : ℝ) :
  (y > x + 1) → (y > 3 - 2 * x) → 
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
sorry

end quadrants_containing_points_l167_167165


namespace paths_from_A_to_B_via_C_l167_167423

open Classical

-- Definitions based on conditions
variables (lattice : Type) [PartialOrder lattice]
variables (A B C : lattice)
variables (first_red first_blue second_red second_blue first_green second_green orange : lattice)

-- Conditions encoded as hypotheses
def direction_changes : Prop :=
  -- Arrow from first green to orange is now one way from orange to green
  ∀ x : lattice, x = first_green → orange < x ∧ ¬ (x < orange) ∧
  -- Additional stop at point C located directly after the first blue arrows
  (C < first_blue ∨ first_blue < C)

-- Now stating the proof problem
theorem paths_from_A_to_B_via_C :
  direction_changes lattice first_green orange first_blue C →
  -- Total number of paths from A to B via C is 12
  (2 + 2) * 3 * 1 = 12 :=
by
  sorry

end paths_from_A_to_B_via_C_l167_167423


namespace num_integers_achievable_le_2014_l167_167175

def floor_div (x : ℤ) : ℤ := x / 2

def button1 (x : ℤ) : ℤ := floor_div x

def button2 (x : ℤ) : ℤ := 4 * x + 1

def num_valid_sequences (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 2
  else num_valid_sequences (n - 1) + num_valid_sequences (n - 2)

theorem num_integers_achievable_le_2014 :
  num_valid_sequences 11 = 233 :=
  by
    -- Proof starts here
    sorry

end num_integers_achievable_le_2014_l167_167175


namespace teacher_engineer_ratio_l167_167645

-- Define the context with the given conditions
variable (t e : ℕ)

-- Conditions
def avg_age (t e : ℕ) : Prop := (40 * t + 55 * e) / (t + e) = 45

-- The statement to be proved
theorem teacher_engineer_ratio
  (h : avg_age t e) :
  t / e = 2 := sorry

end teacher_engineer_ratio_l167_167645


namespace total_puppies_adopted_l167_167192

-- Define the number of puppies adopted each week
def first_week_puppies : ℕ := 20
def second_week_puppies : ℕ := (2 / 5) * first_week_puppies
def third_week_puppies : ℕ := 2 * second_week_puppies
def fourth_week_puppies : ℕ := 10 + first_week_puppies

-- Prove that the total number of puppies adopted over the month is 74
theorem total_puppies_adopted : 
  first_week_puppies + second_week_puppies + third_week_puppies + fourth_week_puppies = 74 := by
  sorry

end total_puppies_adopted_l167_167192


namespace max_rectangle_area_l167_167085

theorem max_rectangle_area (P : ℝ) (hP : P = 60) : ∃ A : ℝ, A = 225 :=
by 
  sorry

end max_rectangle_area_l167_167085


namespace molly_bike_miles_l167_167625

def total_miles_ridden (daily_miles years_riding days_per_year : ℕ) : ℕ :=
  daily_miles * years_riding * days_per_year

theorem molly_bike_miles :
  total_miles_ridden 3 3 365 = 3285 :=
by
  -- The definition and theorem are provided; the implementation will be done by the prover.
  sorry

end molly_bike_miles_l167_167625


namespace total_people_waiting_in_line_l167_167648

-- Conditions
def people_fitting_in_ferris_wheel : ℕ := 56
def people_not_getting_on : ℕ := 36

-- Definition: Number of people waiting in line
def number_of_people_waiting_in_line : ℕ := people_fitting_in_ferris_wheel + people_not_getting_on

-- Theorem to prove
theorem total_people_waiting_in_line : number_of_people_waiting_in_line = 92 := by
  -- This is a placeholder for the actual proof
  sorry

end total_people_waiting_in_line_l167_167648


namespace exist_indices_inequalities_l167_167831

open Nat

theorem exist_indices_inequalities (a b c : ℕ → ℕ) :
  ∃ p q : ℕ, a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
by
  -- The proof is to be written here
  sorry

end exist_indices_inequalities_l167_167831


namespace probability_of_9_heads_in_12_flips_l167_167503

theorem probability_of_9_heads_in_12_flips :
  (nat.choose 12 9) * (1 / (2 : ℝ)^12) = 55 / 1024 := 
by
  sorry

end probability_of_9_heads_in_12_flips_l167_167503


namespace student_losses_one_mark_l167_167770

def number_of_marks_lost_per_wrong_answer (correct_ans marks_attempted total_questions total_marks correct_questions : ℤ) : ℤ :=
  (correct_ans * correct_questions - total_marks) / (total_questions - correct_questions)

theorem student_losses_one_mark
  (correct_ans : ℤ)
  (marks_attempted : ℤ)
  (total_questions : ℤ)
  (total_marks : ℤ)
  (correct_questions : ℤ)
  (total_wrong : ℤ):
  correct_ans = 4 →
  total_questions = 80 →
  total_marks = 120 →
  correct_questions = 40 →
  total_wrong = total_questions - correct_questions →
  number_of_marks_lost_per_wrong_answer correct_ans marks_attempted total_questions total_marks correct_questions = 1 :=
by
  sorry

end student_losses_one_mark_l167_167770


namespace tan_angle_addition_l167_167110

theorem tan_angle_addition (x : ℝ) (h : Real.tan x = 2) : Real.tan (x + Real.pi / 3) = (5 * Real.sqrt 3 + 8) / -11 := by
  sorry

end tan_angle_addition_l167_167110


namespace max_area_of_fenced_rectangle_l167_167071

theorem max_area_of_fenced_rectangle (x y : ℝ) (h : x + y = 30) : x * y ≤ 225 :=
by {
  sorry
}

end max_area_of_fenced_rectangle_l167_167071


namespace minimum_value_condition_l167_167979

def f (a x : ℝ) : ℝ := -x^3 + 0.5 * (a + 3) * x^2 - a * x - 1

theorem minimum_value_condition (a : ℝ) (h : a ≥ 3) : 
  (∃ x₀ : ℝ, f a x₀ < f a 1) ∨ (f a 1 > f a ((a/3))) := 
sorry

end minimum_value_condition_l167_167979


namespace number_of_pages_in_contract_l167_167425

theorem number_of_pages_in_contract (total_pages_copied : ℕ) (copies_per_person : ℕ) (number_of_people : ℕ)
  (h1 : total_pages_copied = 360) (h2 : copies_per_person = 2) (h3 : number_of_people = 9) :
  total_pages_copied / (copies_per_person * number_of_people) = 20 :=
by
  sorry

end number_of_pages_in_contract_l167_167425


namespace Claire_photos_is_5_l167_167136

variable (Claire_photos : ℕ)
variable (Lisa_photos : ℕ := 3 * Claire_photos)
variable (Robert_photos : ℕ := Claire_photos + 10)

theorem Claire_photos_is_5
  (h1 : Lisa_photos = Robert_photos) :
  Claire_photos = 5 :=
by
  sorry

end Claire_photos_is_5_l167_167136


namespace new_average_is_10_5_l167_167022

-- define the conditions
def average_of_eight_numbers (numbers : List ℝ) : Prop :=
  numbers.length = 8 ∧ (numbers.sum / 8) = 8

def add_four_to_five_numbers (numbers : List ℝ) (new_numbers : List ℝ) : Prop :=
  new_numbers = (numbers.take 5).map (λ x => x + 4) ++ numbers.drop 5

-- state the theorem
theorem new_average_is_10_5 (numbers new_numbers : List ℝ) 
  (h1 : average_of_eight_numbers numbers)
  (h2 : add_four_to_five_numbers numbers new_numbers) :
  (new_numbers.sum / 8) = 10.5 := 
by 
  sorry

end new_average_is_10_5_l167_167022


namespace wendy_makeup_time_l167_167813

theorem wendy_makeup_time :
  ∀ (num_products wait_time total_time makeup_time : ℕ),
    num_products = 5 →
    wait_time = 5 →
    total_time = 55 →
    makeup_time = total_time - (num_products - 1) * wait_time →
    makeup_time = 35 :=
by
  intro num_products wait_time total_time makeup_time h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end wendy_makeup_time_l167_167813


namespace hayley_friends_l167_167385

theorem hayley_friends (total_stickers : ℕ) (stickers_per_friend : ℕ) (h1 : total_stickers = 72) (h2 : stickers_per_friend = 8) : (total_stickers / stickers_per_friend) = 9 :=
by
  sorry

end hayley_friends_l167_167385


namespace max_xy_is_2_min_y_over_x_plus_4_over_y_is_4_l167_167756

noncomputable def max_xy (x y : ℝ) : ℝ :=
if h : x > 0 ∧ y > 0 ∧ x + 2 * y = 4 then x * y else 0

noncomputable def min_y_over_x_plus_4_over_y (x y : ℝ) : ℝ :=
if h : x > 0 ∧ y > 0 ∧ x + 2 * y = 4 then y / x + 4 / y else 0

theorem max_xy_is_2 : ∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 4 → max_xy x y = 2 :=
by
  intros x y hx hy hxy
  sorry

theorem min_y_over_x_plus_4_over_y_is_4 : ∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 4 → min_y_over_x_plus_4_over_y x y = 4 :=
by
  intros x y hx hy hxy
  sorry

end max_xy_is_2_min_y_over_x_plus_4_over_y_is_4_l167_167756


namespace manager_decision_correct_l167_167683

theorem manager_decision_correct (x : ℝ) (profit : ℝ) 
  (h_condition1 : ∀ (x : ℝ), profit = (2 * x + 20) * (40 - x)) 
  (h_condition2 : 0 ≤ x ∧ x ≤ 40)
  (h_price_reduction : x = 15) :
  profit = 1250 :=
by
  sorry

end manager_decision_correct_l167_167683


namespace circumcircle_eq_of_triangle_vertices_l167_167378

theorem circumcircle_eq_of_triangle_vertices (A B C: ℝ × ℝ) (hA : A = (0, 4)) (hB : B = (0, 0)) (hC : C = (3, 0)) :
  ∃ D E F : ℝ,
    x^2 + y^2 + D*x + E*y + F = 0 ∧
    (x - 3/2)^2 + (y - 2)^2 = 25/4 :=
by 
  sorry

end circumcircle_eq_of_triangle_vertices_l167_167378


namespace max_area_of_rectangular_pen_l167_167080

theorem max_area_of_rectangular_pen (P : ℕ) (hP : P = 60) : 
  ∃ A : ℕ, A = 225 ∧ (∀ x y : ℕ, 2 * (x + y) = P → x * y ≤ A) := 
by
  sorry

end max_area_of_rectangular_pen_l167_167080


namespace count_four_digit_numbers_with_5_or_7_l167_167583

def num_four_digit_numbers : Nat := 9000
def exclude_first_digit : Finset Nat := {1, 2, 3, 4, 6, 8, 9}  -- 7 options
def exclude_other_digits : Finset Nat := {0, 1, 2, 3, 4, 6, 8, 9}  -- 8 options
def excluded_numbers_count : Nat := exclude_first_digit.card * exclude_other_digits.card ^ 3  -- 3584
def included_numbers_count : Nat := num_four_digit_numbers - excluded_numbers_count  -- 5416

theorem count_four_digit_numbers_with_5_or_7 :
  included_numbers_count = 5416 :=
by
  sorry

end count_four_digit_numbers_with_5_or_7_l167_167583


namespace marbles_per_customer_l167_167626

theorem marbles_per_customer
  (initial_marbles remaining_marbles customers marbles_per_customer : ℕ)
  (h1 : initial_marbles = 400)
  (h2 : remaining_marbles = 100)
  (h3 : customers = 20)
  (h4 : initial_marbles - remaining_marbles = customers * marbles_per_customer) :
  marbles_per_customer = 15 :=
by
  sorry

end marbles_per_customer_l167_167626


namespace king_luis_courtiers_are_odd_l167_167127

theorem king_luis_courtiers_are_odd (n : ℕ) 
  (h : ∀ i : ℕ, i < n → ∃ j : ℕ, j < n ∧ i ≠ j) : 
  ¬ Even n := 
sorry

end king_luis_courtiers_are_odd_l167_167127


namespace basketball_team_points_l167_167652

variable (a b x : ℕ)

theorem basketball_team_points (h1 : 2 * a = 3 * b) 
                             (h2 : x = a + 1)
                             (h3 : 2 * a + 3 * b + x = 61) : 
    x = 13 :=
by {
  sorry
}

end basketball_team_points_l167_167652


namespace playground_area_l167_167001

theorem playground_area (l w : ℝ) 
  (h_perimeter : 2 * l + 2 * w = 100)
  (h_length : l = 3 * w) : l * w = 468.75 :=
by
  sorry

end playground_area_l167_167001


namespace no_such_triples_l167_167860

noncomputable def no_triple_satisfy (a b c : ℤ) : Prop :=
  ∀ (x1 x2 x3 : ℤ), 
    x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    Int.gcd x1 x2 = 1 ∧ Int.gcd x2 x3 = 1 ∧ Int.gcd x1 x3 = 1 ∧
    (x1^3 - a^2 * x1^2 + b^2 * x1 - a * b + 3 * c = 0) ∧ 
    (x2^3 - a^2 * x2^2 + b^2 * x2 - a * b + 3 * c = 0) ∧ 
    (x3^3 - a^2 * x3^2 + b^2 * x3 - a * b + 3 * c = 0) →
    False

theorem no_such_triples : ∀ (a b c : ℤ), no_triple_satisfy a b c :=
by
  intros
  sorry

end no_such_triples_l167_167860


namespace _l167_167788

noncomputable theory
open Classical

/-- Statement of the theorem in Lean 4 -/
def prob_sine_floor_eq (x y : ℝ) (hx : 0 < x ∧ x < π / 2) (hy : 0 < y ∧ y < π / 2) :
  Pr(\(floor (sin x) = floor (sin y)\)) = 1 := sorry

end _l167_167788


namespace donny_spent_on_thursday_l167_167720

theorem donny_spent_on_thursday :
  let savings_monday : ℤ := 15,
      savings_tuesday : ℤ := 28,
      savings_wednesday : ℤ := 13,
      total_savings : ℤ := savings_monday + savings_tuesday + savings_wednesday,
      amount_spent_thursday : ℤ := total_savings / 2
  in
  amount_spent_thursday = 28 :=
by
  sorry

end donny_spent_on_thursday_l167_167720


namespace problem1_l167_167555

theorem problem1 :
  (-1 : ℤ)^2024 - (-1 : ℤ)^2023 = 2 := by
  sorry

end problem1_l167_167555


namespace probability_heads_9_of_12_flips_l167_167490

theorem probability_heads_9_of_12_flips : (nat.choose 12 9) / (2^12) = 55 / 1024 :=
by
  sorry

end probability_heads_9_of_12_flips_l167_167490


namespace probability_heads_in_nine_of_twelve_flips_l167_167480

theorem probability_heads_in_nine_of_twelve_flips : 
  (nat.choose 12 9) / (2^12) = 55 / 1024 := 
by
  sorry

end probability_heads_in_nine_of_twelve_flips_l167_167480


namespace probability_heads_in_9_of_12_flips_l167_167498

theorem probability_heads_in_9_of_12_flips :
  let total_flips := 12 in
  let required_heads := 9 in
  let total_outcomes := 2^total_flips in
  let specific_outcomes := Nat.choose total_flips required_heads in
  (specific_outcomes : ℚ) / total_outcomes = 220 / 4096 := 
by 
  sorry

end probability_heads_in_9_of_12_flips_l167_167498


namespace last_two_digits_of_sum_l167_167553

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_of_sum :
  last_two_digits (factorial 4 + factorial 5 + factorial 6 + factorial 7 + factorial 8 + factorial 9) = 4 :=
by
  sorry

end last_two_digits_of_sum_l167_167553


namespace auditorium_seats_l167_167769

variable (S : ℕ)

theorem auditorium_seats (h1 : 2 * S / 5 + S / 10 + 250 = S) : S = 500 :=
by
  sorry

end auditorium_seats_l167_167769


namespace num_both_sports_l167_167393

def num_people := 310
def num_tennis := 138
def num_baseball := 255
def num_no_sport := 11

theorem num_both_sports : (num_tennis + num_baseball - (num_people - num_no_sport)) = 94 :=
by 
-- leave the proof out for now
sorry

end num_both_sports_l167_167393


namespace food_per_puppy_meal_l167_167776

-- Definitions for conditions
def mom_daily_food : ℝ := 1.5 * 3
def num_puppies : ℕ := 5
def total_food_needed : ℝ := 57
def num_days : ℕ := 6

-- Total food for the mom dog over the given period
def total_mom_food : ℝ := mom_daily_food * num_days

-- Total food for the puppies over the given period
def total_puppy_food : ℝ := total_food_needed - total_mom_food

-- Total number of puppy meals over the given period
def total_puppy_meals : ℕ := (num_puppies * 2) * num_days

theorem food_per_puppy_meal :
  total_puppy_food / total_puppy_meals = 0.5 :=
  sorry

end food_per_puppy_meal_l167_167776


namespace sum_of_products_of_two_at_a_time_l167_167807

-- Given conditions
variables (a b c : ℝ)
axiom sum_of_squares : a^2 + b^2 + c^2 = 252
axiom sum_of_numbers : a + b + c = 22

-- The goal
theorem sum_of_products_of_two_at_a_time : a * b + b * c + c * a = 116 :=
sorry

end sum_of_products_of_two_at_a_time_l167_167807


namespace range_of_m_l167_167879

theorem range_of_m (m : ℝ) :
  (¬(∀ x : ℝ, x^2 + m * x + 1 = 0 → x ≠ 0) ∧ ∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 ≠ 0) → (1 < m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l167_167879


namespace smallest_consecutive_natural_number_sum_l167_167010

theorem smallest_consecutive_natural_number_sum (a n : ℕ) (hn : n > 1) (h : n * a + (n * (n - 1)) / 2 = 2016) :
  ∃ a, a = 1 :=
by
  sorry

end smallest_consecutive_natural_number_sum_l167_167010


namespace max_area_of_rectangular_pen_l167_167082

theorem max_area_of_rectangular_pen (P : ℕ) (hP : P = 60) : 
  ∃ A : ℕ, A = 225 ∧ (∀ x y : ℕ, 2 * (x + y) = P → x * y ≤ A) := 
by
  sorry

end max_area_of_rectangular_pen_l167_167082


namespace fraction_of_women_married_l167_167020

theorem fraction_of_women_married (total : ℕ) (women men married: ℕ) (h1 : total = women + men)
(h2 : women = 76 * total / 100) (h3 : married = 60 * total / 100) (h4 : 2 * (men - married) = 3 * men):
 (married - (total - women - married) * 1 / 3) = 13 * women / 19 :=
sorry

end fraction_of_women_married_l167_167020


namespace period_is_3_years_l167_167842

def gain_of_B_per_annum (principal : ℕ) (rate_A rate_B : ℚ) : ℚ := 
  (rate_B - rate_A) * principal

def period (principal : ℕ) (rate_A rate_B : ℚ) (total_gain : ℚ) : ℚ := 
  total_gain / gain_of_B_per_annum principal rate_A rate_B

theorem period_is_3_years :
  period 1500 (10 / 100) (11.5 / 100) 67.5 = 3 :=
by
  sorry

end period_is_3_years_l167_167842


namespace bridge_length_is_correct_l167_167671

noncomputable def length_of_bridge (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time_seconds : ℝ) : ℝ :=
  let speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := speed_mps * crossing_time_seconds
  total_distance - train_length

theorem bridge_length_is_correct :
  length_of_bridge 200 (60) 45 = 550.15 :=
by
  sorry

end bridge_length_is_correct_l167_167671


namespace evaluate_expression_l167_167307

theorem evaluate_expression : 4 * 5 - 3 + 2^3 - 3 * 2 = 19 := by
  sorry

end evaluate_expression_l167_167307


namespace n_not_2_7_l167_167106

open Set

variable (M N : Set ℕ)

-- Define the given set M
def M_def : Prop := M = {1, 4, 7}

-- Define the condition M ∪ N = M
def union_condition : Prop := M ∪ N = M

-- The main statement to be proved
theorem n_not_2_7 (M_def : M = {1, 4, 7}) (union_condition : M ∪ N = M) : N ≠ {2, 7} :=
  sorry

end n_not_2_7_l167_167106


namespace solve_for_x_l167_167147

theorem solve_for_x (x : ℕ) : 100^3 = 10^x → x = 6 := by
  sorry

end solve_for_x_l167_167147


namespace exists_n_consecutive_numbers_l167_167558

theorem exists_n_consecutive_numbers:
  ∃ n : ℕ, n % 5 = 0 ∧ (n + 1) % 4 = 0 ∧ (n + 2) % 3 = 0 := sorry

end exists_n_consecutive_numbers_l167_167558


namespace intersection_P_Q_equals_P_l167_167224

def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := { y | ∃ x ∈ Set.univ, y = Real.cos x }

theorem intersection_P_Q_equals_P : P ∩ Q = P := by
  sorry

end intersection_P_Q_equals_P_l167_167224


namespace sum_ef_l167_167135

variables (a b c d e f : ℝ)

-- Definitions based on conditions
def avg_ab : Prop := (a + b) / 2 = 5.2
def avg_cd : Prop := (c + d) / 2 = 5.8
def overall_avg : Prop := (a + b + c + d + e + f) / 6 = 5.4

-- Main theorem to prove
theorem sum_ef (h1 : avg_ab a b) (h2 : avg_cd c d) (h3 : overall_avg a b c d e f) : e + f = 10.4 :=
sorry

end sum_ef_l167_167135


namespace allyn_total_expense_in_june_l167_167689

/-- We have a house with 40 bulbs, each using 60 watts of power daily.
Allyn pays 0.20 dollars per watt used. June has 30 days.
We need to calculate Allyn's total monthly expense on electricity in June,
which should be \$14400. -/
theorem allyn_total_expense_in_june
    (daily_watt_per_bulb : ℕ := 60)
    (num_bulbs : ℕ := 40)
    (cost_per_watt : ℝ := 0.20)
    (days_in_june : ℕ := 30)
    : num_bulbs * daily_watt_per_bulb * days_in_june * cost_per_watt = 14400 := 
by
  sorry

end allyn_total_expense_in_june_l167_167689


namespace person_age_in_1954_l167_167942

theorem person_age_in_1954 
  (x : ℤ)
  (cond1 : ∃ k1 : ℤ, 7 * x = 13 * k1 + 11)
  (cond2 : ∃ k2 : ℤ, 13 * x = 11 * k2 + 7)
  (input_year : ℤ) :
  input_year = 1954 → x = 1868 → input_year - x = 86 :=
by
  sorry

end person_age_in_1954_l167_167942


namespace stratified_sampling_male_athletes_l167_167219

theorem stratified_sampling_male_athletes : 
  ∀ (total_males total_females total_to_sample : ℕ), 
    total_males = 20 → 
    total_females = 10 → 
    total_to_sample = 6 → 
    20 * (total_to_sample / (total_males + total_females)) = 4 :=
by
  intros total_males total_females total_to_sample h_males h_females h_sample
  rw [h_males, h_females, h_sample]
  sorry

end stratified_sampling_male_athletes_l167_167219


namespace sum_first_twelve_terms_of_arithmetic_sequence_l167_167554

theorem sum_first_twelve_terms_of_arithmetic_sequence :
    let a1 := -3
    let a12 := 48
    let n := 12
    let Sn := (n * (a1 + a12)) / 2
    Sn = 270 := 
by
  sorry

end sum_first_twelve_terms_of_arithmetic_sequence_l167_167554


namespace largest_gcd_of_sum_1729_l167_167954

theorem largest_gcd_of_sum_1729 (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1729) :
  ∃ g, g = Nat.gcd x y ∧ g = 247 := sorry

end largest_gcd_of_sum_1729_l167_167954


namespace fraction_zero_l167_167520

theorem fraction_zero (x : ℝ) (h₁ : 2 * x = 0) (h₂ : x + 2 ≠ 0) : (2 * x) / (x + 2) = 0 :=
by {
  sorry
}

end fraction_zero_l167_167520


namespace explicit_formula_for_sequence_l167_167370

theorem explicit_formula_for_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (hSn : ∀ n, S n = 2 * a n + 1) : 
  ∀ n, a n = (-2) ^ (n - 1) := 
by
  sorry

end explicit_formula_for_sequence_l167_167370


namespace trapezoid_area_l167_167519

noncomputable def area_trapezoid : ℝ :=
  let x1 := 10
  let x2 := -10
  let y1 := 10
  let h := 10
  let a := 20  -- length of top side at y = 10
  let b := 10  -- length of lower side
  (a + b) * h / 2

theorem trapezoid_area : area_trapezoid = 150 := by
  sorry

end trapezoid_area_l167_167519


namespace max_area_of_rectangle_with_perimeter_60_l167_167099

theorem max_area_of_rectangle_with_perimeter_60 :
  ∃ (x y : ℝ), 2 * (x + y) = 60 ∧ x * y = 225 :=
by
  sorry

end max_area_of_rectangle_with_perimeter_60_l167_167099


namespace quadratic_completion_l167_167793

theorem quadratic_completion 
    (x : ℝ) 
    (h : 16*x^2 - 32*x - 512 = 0) : 
    ∃ r s : ℝ, (x + r)^2 = s ∧ s = 33 :=
by sorry

end quadratic_completion_l167_167793


namespace max_area_of_rectangular_pen_l167_167077

-- Define the perimeter and derive the formula for the area
def perimeter := 60
def half_perimeter := perimeter / 2
def area (x : ℝ) := x * (half_perimeter - x)

-- Statement of the problem: prove the maximum area is 225 square feet
theorem max_area_of_rectangular_pen : ∃ x : ℝ, 0 ≤ x ∧ x ≤ half_perimeter ∧ area x = 225 := 
sorry

end max_area_of_rectangular_pen_l167_167077


namespace solve_inequality_l167_167936

def polynomial_fraction (x : ℝ) : ℝ :=
  (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5)

theorem solve_inequality (x : ℝ) :
  -2 < polynomial_fraction x ∧ polynomial_fraction x < 2 ↔ 11.57 < x :=
sorry

end solve_inequality_l167_167936


namespace heejin_most_balls_is_volleyballs_l167_167750

def heejin_basketballs : ℕ := 3
def heejin_volleyballs : ℕ := 5
def heejin_baseballs : ℕ := 1

theorem heejin_most_balls_is_volleyballs :
  heejin_volleyballs > heejin_basketballs ∧ heejin_volleyballs > heejin_baseballs :=
by
  sorry

end heejin_most_balls_is_volleyballs_l167_167750


namespace probability_9_heads_12_flips_l167_167515

noncomputable def probability_of_9_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_9_heads_12_flips : probability_of_9_heads 12 9 (1 / 2) = 55 / 1024 := by
  sorry

end probability_9_heads_12_flips_l167_167515


namespace hair_cut_off_length_l167_167773

def initial_hair_length : ℕ := 18
def hair_length_after_haircut : ℕ := 9

theorem hair_cut_off_length :
  initial_hair_length - hair_length_after_haircut = 9 :=
sorry

end hair_cut_off_length_l167_167773


namespace total_books_proof_l167_167869

-- Define the number of books Lily finished last month.
def books_last_month : ℕ := 4

-- Define the number of books Lily wants to finish this month.
def books_this_month : ℕ := books_last_month * 2

-- Define the total number of books Lily will finish in two months.
def total_books_two_months : ℕ := books_last_month + books_this_month

-- Theorem to prove the total number of books Lily will finish in two months is 12.
theorem total_books_proof : total_books_two_months = 12 := by
  -- Here would be the proof steps.
  sorry

end total_books_proof_l167_167869


namespace algebra_books_cannot_be_determined_uniquely_l167_167843

theorem algebra_books_cannot_be_determined_uniquely (A H S M E : ℕ) (pos_A : A > 0) (pos_H : H > 0) (pos_S : S > 0) 
  (pos_M : M > 0) (pos_E : E > 0) (distinct : A ≠ H ∧ A ≠ S ∧ A ≠ M ∧ A ≠ E ∧ H ≠ S ∧ H ≠ M ∧ H ≠ E ∧ S ≠ M ∧ S ≠ E ∧ M ≠ E) 
  (cond1: S < A) (cond2: M > H) (cond3: A + 2 * H = S + 2 * M) : 
  E = 0 :=
sorry

end algebra_books_cannot_be_determined_uniquely_l167_167843


namespace find_seventh_term_l167_167412

variables {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}

-- Define arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Define sum of the first n terms of the sequence
def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * a 0) + (d * (n * (n - 1)) / 2)

-- Now state the theorem
theorem find_seventh_term
  (h_arith_seq : arithmetic_sequence a d)
  (h_nonzero_d : d ≠ 0)
  (h_sum_five : S 5 = 5)
  (h_squares_eq : a 0 ^ 2 + a 1 ^ 2 = a 2 ^ 2 + a 3 ^ 2) :
  a 6 = 9 :=
sorry

end find_seventh_term_l167_167412


namespace ratio_of_incomes_l167_167439

variable {I1 I2 E1 E2 S1 S2 : ℝ}

theorem ratio_of_incomes
  (h1 : I1 = 4000)
  (h2 : E1 / E2 = 3 / 2)
  (h3 : S1 = 1600)
  (h4 : S2 = 1600)
  (h5 : S1 = I1 - E1)
  (h6 : S2 = I2 - E2) :
  I1 / I2 = 5 / 4 :=
by
  sorry

end ratio_of_incomes_l167_167439


namespace problem_statement_l167_167027

theorem problem_statement (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (1 / Real.sqrt (2011 + Real.sqrt (2011^2 - 1)) = Real.sqrt m - Real.sqrt n) →
  m + n = 2011 :=
sorry

end problem_statement_l167_167027


namespace balls_in_boxes_l167_167753

theorem balls_in_boxes : 
  (∃ f : Fin 7 → Fin 7, (∃ (I : Finset (Fin 7)), I.card = 3 ∧ ∀ i ∈ I, f i = i) ∧ 
  (∃ (J : Finset (Fin 7)), J.card = 4 ∧ J ∩ I = ∅ ∧ 
  ∀ j ∈ J, f j ≠ j)) →
  (∃ n : ℕ, n = 315) :=
begin
  sorry -- proof goes here
end

end balls_in_boxes_l167_167753


namespace find_passing_marks_l167_167196

-- Defining the conditions as Lean statements
def condition1 (T P : ℝ) : Prop := 0.30 * T = P - 50
def condition2 (T P : ℝ) : Prop := 0.45 * T = P + 25

-- The theorem to prove
theorem find_passing_marks (T P : ℝ) (h1 : condition1 T P) (h2 : condition2 T P) : P = 200 :=
by
  -- Placeholder proof
  sorry

end find_passing_marks_l167_167196


namespace relationship_among_a_b_c_l167_167889

noncomputable def a : ℝ := Real.logb 0.5 0.2
noncomputable def b : ℝ := Real.logb 2 0.2
noncomputable def c : ℝ := Real.exp (0.2 * Real.log 2)

theorem relationship_among_a_b_c : b < c ∧ c < a :=
by
  sorry

end relationship_among_a_b_c_l167_167889


namespace south_walk_correct_representation_l167_167116

theorem south_walk_correct_representation {north south : ℤ} (h_north : north = 3) (h_representation : south = -north) : south = -5 :=
by
  have h1 : -north = -3 := by rw [h_north]
  have h2 : -3 = -5 := by sorry
  rw [h_representation, h1]
  exact h2

end south_walk_correct_representation_l167_167116


namespace max_area_of_rectangular_pen_l167_167079

theorem max_area_of_rectangular_pen (P : ℕ) (hP : P = 60) : 
  ∃ A : ℕ, A = 225 ∧ (∀ x y : ℕ, 2 * (x + y) = P → x * y ≤ A) := 
by
  sorry

end max_area_of_rectangular_pen_l167_167079


namespace centroid_traces_ellipse_l167_167580

noncomputable def fixed_base_triangle (A B : ℝ × ℝ) (d : ℝ) : Prop :=
(A.1 = 0 ∧ A.2 = 0) ∧ (B.1 = d ∧ B.2 = 0)

noncomputable def vertex_moving_on_semicircle (A B C : ℝ × ℝ) : Prop :=
(C.1 - (A.1 + B.1) / 2)^2 + C.2^2 = ((B.1 - A.1) / 2)^2 ∧ C.2 ≥ 0

noncomputable def is_centroid (A B C G : ℝ × ℝ) : Prop :=
G.1 = (A.1 + B.1 + C.1) / 3 ∧ G.2 = (A.2 + B.2 + C.2) / 3

theorem centroid_traces_ellipse
  (A B C G : ℝ × ℝ) (d : ℝ) 
  (h1 : fixed_base_triangle A B d) 
  (h2 : vertex_moving_on_semicircle A B C)
  (h3 : is_centroid A B C G) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (G.1^2 / a^2 + G.2^2 / b^2 = 1) := 
sorry

end centroid_traces_ellipse_l167_167580


namespace solve_equation_solutions_count_l167_167149

open Real

theorem solve_equation_solutions_count :
  (∃ (x_plural : List ℝ), (∀ x ∈ x_plural, 2 * sqrt 2 * (sin (π * x / 4)) ^ 3 = cos (π / 4 * (1 - x)) ∧ 0 ≤ x ∧ x ≤ 2020) ∧ x_plural.length = 505) :=
sorry

end solve_equation_solutions_count_l167_167149


namespace total_books_l167_167874

theorem total_books (books_last_month : ℕ) (goal_factor : ℕ) (books_this_month : ℕ) (total_books : ℕ) 
  (h1 : books_last_month = 4) 
  (h2 : goal_factor = 2) 
  (h3 : books_this_month = goal_factor * books_last_month) 
  (h4 : total_books = books_last_month + books_this_month) 
  : total_books = 12 := 
by
  sorry

end total_books_l167_167874


namespace fixed_point_of_line_l167_167054

theorem fixed_point_of_line (m : ℝ) : 
  (m - 1) * (7 / 2) - (m + 3) * (5 / 2) - (m - 11) = 0 :=
by
  sorry

end fixed_point_of_line_l167_167054


namespace possible_birches_l167_167676

theorem possible_birches (N B L : ℕ) (hN : N = 130) (h_sum : B + L = 130)
  (h_linden_false : ∀ l, l < L → (∀ b, b < B → b + l < N → b < B → False))
  (h_birch_false : ∃ b, b < B ∧ (∀ l, l < L → l + b < N → l + b = 2 * B))
  : B = 87 :=
sorry

end possible_birches_l167_167676


namespace derivative_f_l167_167532

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) * (Real.exp (Real.sin x))

theorem derivative_f (x : ℝ) : deriv f x = ((Real.cos x)^2 - Real.sin x) * (Real.exp (Real.sin x)) :=
by
  sorry

end derivative_f_l167_167532


namespace final_quantity_of_milk_l167_167692

-- Define initial conditions
def initial_volume : ℝ := 60
def removed_volume : ℝ := 9

-- Given the initial conditions, calculate the quantity of milk left after two dilutions
theorem final_quantity_of_milk :
  let first_removal_ratio := initial_volume - removed_volume / initial_volume
  let first_milk_volume := initial_volume * (first_removal_ratio)
  let second_removal_ratio := first_milk_volume / initial_volume
  let second_milk_volume := first_milk_volume * (second_removal_ratio)
  second_milk_volume = 43.35 :=
by
  sorry

end final_quantity_of_milk_l167_167692


namespace product_pass_rate_l167_167326

variable {a b : ℝ} (h_a : 0 ≤ a ∧ a ≤ 1) (h_b : 0 ≤ b ∧ b ≤ 1) (h_indep : true)

theorem product_pass_rate : (1 - a) * (1 - b) = 
((1 - a) * (1 - b)) :=
by
  sorry

end product_pass_rate_l167_167326


namespace ming_dynasty_wine_problem_l167_167915

theorem ming_dynasty_wine_problem (x y : ℕ) (h1 : x + y = 19) (h2 : 3 * x + y / 3 = 33 ) : 
  (x = 10 ∧ y = 9) :=
by {
  sorry
}

end ming_dynasty_wine_problem_l167_167915


namespace second_car_mileage_l167_167848

theorem second_car_mileage (x : ℝ) : 
  (150 / 50) + (150 / x) + (150 / 15) = 56 / 2 → x = 10 :=
by
  intro h
  sorry

end second_car_mileage_l167_167848


namespace election_total_votes_l167_167612

theorem election_total_votes
  (V : ℝ)
  (h1 : 0 ≤ V) 
  (h_majority : 0.70 * V - 0.30 * V = 182) :
  V = 455 := 
by 
  sorry

end election_total_votes_l167_167612


namespace original_weight_of_apples_l167_167657

theorem original_weight_of_apples (x : ℕ) (h1 : 5 * (x - 30) = 2 * x) : x = 50 :=
by
  sorry

end original_weight_of_apples_l167_167657


namespace range_m_l167_167059

variable {x m : ℝ}

theorem range_m (h1 : m / (1 - x) - 2 / (x - 1) = 1) (h2 : x ≥ 0) (h3 : x ≠ 1) : m ≤ -1 ∧ m ≠ -2 := 
sorry

end range_m_l167_167059


namespace domain_of_function_l167_167122

noncomputable def is_defined (x : ℝ) : Prop :=
  (x + 4 ≥ 0) ∧ (x ≠ 0)

theorem domain_of_function :
  ∀ x : ℝ, is_defined x ↔ x ≥ -4 ∧ x ≠ 0 :=
by
  sorry

end domain_of_function_l167_167122


namespace panthers_second_half_points_l167_167611

theorem panthers_second_half_points (C1 P1 C2 P2 : ℕ) 
  (h1 : C1 + P1 = 38) 
  (h2 : C1 = P1 + 16) 
  (h3 : C1 + C2 + P1 + P2 = 58) 
  (h4 : C1 + C2 = P1 + P2 + 22) : 
  P2 = 7 :=
by 
  -- Definitions and substitutions are skipped here
  sorry

end panthers_second_half_points_l167_167611


namespace total_puppies_adopted_l167_167191

-- Define the number of puppies adopted each week
def first_week_puppies : ℕ := 20
def second_week_puppies : ℕ := (2 / 5) * first_week_puppies
def third_week_puppies : ℕ := 2 * second_week_puppies
def fourth_week_puppies : ℕ := 10 + first_week_puppies

-- Prove that the total number of puppies adopted over the month is 74
theorem total_puppies_adopted : 
  first_week_puppies + second_week_puppies + third_week_puppies + fourth_week_puppies = 74 := by
  sorry

end total_puppies_adopted_l167_167191


namespace P_has_no_negative_roots_but_at_least_one_positive_root_l167_167716

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := x^6 - 4*x^5 - 9*x^3 + 2*x + 9

-- Statement of the problem
theorem P_has_no_negative_roots_but_at_least_one_positive_root :
  (∀ x : ℝ, x < 0 → P x ≠ 0 ∧ P x > 0) ∧ (∃ x : ℝ, x > 0 ∧ P x = 0) :=
by
  sorry

end P_has_no_negative_roots_but_at_least_one_positive_root_l167_167716


namespace g_eq_one_l167_167414

theorem g_eq_one (g : ℝ → ℝ) 
  (h1 : ∀ (x y : ℝ), g (x - y) = g x * g y) 
  (h2 : ∀ (x : ℝ), g x ≠ 0) : 
  g 5 = 1 :=
by
  sorry

end g_eq_one_l167_167414


namespace all_propositions_correct_l167_167581

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

theorem all_propositions_correct (m n : ℝ) (a b : α) (h1 : m ≠ 0) (h2 : a ≠ 0) : 
  (∀ (m : ℝ) (a b : α), m • (a - b) = m • a - m • b) ∧
  (∀ (m n : ℝ) (a : α), (m - n) • a = m • a - n • a) ∧
  (∀ (m : ℝ) (a b : α), m • a = m • b → a = b) ∧
  (∀ (m n : ℝ) (a : α), m • a = n • a → m = n) :=
by {
  sorry
}

end all_propositions_correct_l167_167581


namespace lineD_intersects_line1_l167_167697

-- Define the lines based on the conditions
def line1 (x y : ℝ) := x + y - 1 = 0
def lineA (x y : ℝ) := 2 * x + 2 * y = 6
def lineB (x y : ℝ) := x + y = 0
def lineC (x y : ℝ) := y = -x - 3
def lineD (x y : ℝ) := y = x - 1

-- Define the statement that line D intersects with line1
theorem lineD_intersects_line1 : ∃ (x y : ℝ), line1 x y ∧ lineD x y :=
by
  sorry

end lineD_intersects_line1_l167_167697


namespace find_f_inv_64_l167_167387

noncomputable def f : ℝ → ℝ :=
  sorry  -- We don't know the exact form of f.

axiom f_property_1 : f 5 = 2

axiom f_property_2 : ∀ x : ℝ, f (2 * x) = 2 * f x

def f_inv (y : ℝ) : ℝ :=
  sorry  -- We define the inverse function in terms of y.

theorem find_f_inv_64 : f_inv 64 = 160 :=
by {
  -- Main statement to be proved.
  sorry
}

end find_f_inv_64_l167_167387


namespace fraction_of_students_with_buddy_l167_167911

variables (f e : ℕ)
-- Given:
axiom H1 : e / 4 = f / 3

-- Prove:
theorem fraction_of_students_with_buddy : 
  (e / 4 + f / 3) / (e + f) = 2 / 7 :=
by
  sorry

end fraction_of_students_with_buddy_l167_167911


namespace square_free_odd_integers_count_l167_167902

theorem square_free_odd_integers_count :
  let positiveOddIntegers := {n : ℕ | 1 < n ∧ n < 200 ∧ n % 2 = 1}
  let squareFree := λ x : ℕ, ∀ m : ℕ, m * m ∣ x → m = 1
  (∃ S : Finset ℕ, S.card = 82 ∧ ∀ n ∈ S, n ∈ positiveOddIntegers ∧ squareFree n) :=
sorry

end square_free_odd_integers_count_l167_167902


namespace tom_jerry_age_ratio_l167_167299

-- Definitions representing the conditions in the problem
variable (t j x : ℕ)

-- Condition 1: Three years ago, Tom was three times as old as Jerry
def condition1 : Prop := t - 3 = 3 * (j - 3)

-- Condition 2: Four years before that, Tom was five times as old as Jerry
def condition2 : Prop := t - 7 = 5 * (j - 7)

-- Question: In how many years will the ratio of their ages be 3:2,
-- asserting that the answer is 21
def ageRatioInYears : Prop := (t + x) / (j + x) = 3 / 2 → x = 21

-- The proposition we need to prove
theorem tom_jerry_age_ratio (h1 : condition1 t j) (h2 : condition2 t j) : ageRatioInYears t j x := 
  sorry
  
end tom_jerry_age_ratio_l167_167299


namespace sum_of_coefficients_is_37_l167_167051

open Polynomial

noncomputable def polynomial := 
  -3 * (X ^ 8 - 2 * X ^ 5 + 4 * X ^ 3 - 6) +
  5 * (2 * X ^ 4 + 3 * X ^ 2 - X) -
  2 * (3 * X ^ 6 - 7)

theorem sum_of_coefficients_is_37 : (polynomial.eval 1 polynomial) = 37 :=
by
  sorry

end sum_of_coefficients_is_37_l167_167051


namespace gcf_60_90_150_l167_167960

theorem gcf_60_90_150 : Nat.gcd (Nat.gcd 60 90) 150 = 30 :=
by
  sorry

end gcf_60_90_150_l167_167960


namespace prove_n_prime_l167_167867

theorem prove_n_prime (n : ℕ) (p : ℕ) (k : ℕ) (hp : Prime p) (h1 : n > 0) (h2 : 3^n - 2^n = p^k) : Prime n :=
by {
  sorry
}

end prove_n_prime_l167_167867


namespace remaining_bottles_l167_167525

variable (s : ℕ) (b : ℕ) (ps : ℚ) (pb : ℚ)

theorem remaining_bottles (h1 : s = 6000) (h2 : b = 14000) (h3 : ps = 0.20) (h4 : pb = 0.23) : 
  s - Nat.floor (ps * s) + b - Nat.floor (pb * b) = 15580 :=
by
  sorry

end remaining_bottles_l167_167525


namespace calculate_new_average_weight_l167_167446

noncomputable def new_average_weight (original_team_weight : ℕ) (num_original_players : ℕ) 
 (new_player1_weight : ℕ) (new_player2_weight : ℕ) (num_new_players : ℕ) : ℕ :=
 (original_team_weight + new_player1_weight + new_player2_weight) / (num_original_players + num_new_players)

theorem calculate_new_average_weight : 
  new_average_weight 847 7 110 60 2 = 113 := 
by 
sorry

end calculate_new_average_weight_l167_167446


namespace area_sum_four_smaller_circles_equals_area_of_large_circle_l167_167394

theorem area_sum_four_smaller_circles_equals_area_of_large_circle (R : ℝ) :
  let radius_large := R
  let radius_small := R / 2
  let area_large := π * radius_large^2
  let area_small := π * radius_small^2
  let total_area_small := 4 * area_small
  area_large = total_area_small :=
by
  sorry

end area_sum_four_smaller_circles_equals_area_of_large_circle_l167_167394


namespace projection_onto_plane_l167_167131

open Matrix Vec

def n : Vec ℝ 3 := ⟨2, 1, -2⟩

def Q : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![5/9, -2/9, 4/9],
    ![-2/9, 8/9, 2/9],
    ![4/9, 2/9, 5/9]
  ]

theorem projection_onto_plane (u : Vec ℝ 3) :
  ((Q ⬝ u) : Three) = proj u :=
sorry

end projection_onto_plane_l167_167131


namespace third_candidate_more_votes_than_john_l167_167984

-- Define the given conditions
def total_votes : ℕ := 1150
def john_votes : ℕ := 150
def remaining_votes : ℕ := total_votes - john_votes
def james_votes : ℕ := (7 * remaining_votes) / 10
def john_and_james_votes : ℕ := john_votes + james_votes
def third_candidate_votes : ℕ := total_votes - john_and_james_votes

-- Stating the problem to prove
theorem third_candidate_more_votes_than_john : third_candidate_votes - john_votes = 150 := 
by
  sorry

end third_candidate_more_votes_than_john_l167_167984


namespace negation_of_proposition_l167_167884

theorem negation_of_proposition (p : ∀ x : ℝ, -x^2 + 4 * x + 3 > 0) :
  (∃ x : ℝ, -x^2 + 4 * x + 3 ≤ 0) :=
sorry

end negation_of_proposition_l167_167884


namespace probability_exactly_9_heads_l167_167496

theorem probability_exactly_9_heads :
  (Nat.choose 12 9) / (2 ^ 12 : ℚ) = 220 / 4096 := by
  sorry

end probability_exactly_9_heads_l167_167496


namespace theresa_crayons_count_l167_167297

noncomputable def crayons_teresa (initial_teresa_crayons : Nat) 
                                 (initial_janice_crayons : Nat) 
                                 (shared_with_nancy : Nat)
                                 (given_to_mark : Nat)
                                 (received_from_nancy : Nat) : Nat := 
  initial_teresa_crayons + received_from_nancy

theorem theresa_crayons_count : crayons_teresa 32 12 (12 / 2) 3 8 = 40 := by
  -- Given: Theresa initially has 32 crayons.
  -- Janice initially has 12 crayons.
  -- Janice shares half of her crayons with Nancy: 12 / 2 = 6 crayons.
  -- Janice gives 3 crayons to Mark.
  -- Theresa receives 8 crayons from Nancy.
  -- Therefore: Theresa will have 32 + 8 = 40 crayons.
  sorry

end theresa_crayons_count_l167_167297


namespace probability_heads_9_of_12_l167_167512

theorem probability_heads_9_of_12 :
  let total_outcomes := 2^12 in
  let favorable_outcomes := Nat.choose 12 9 in
  let probability := favorable_outcomes / total_outcomes.toRat in
  probability = 220 / 4096 := 
by
  sorry

end probability_heads_9_of_12_l167_167512


namespace Greg_PPO_Obtained_90_Percent_l167_167749

theorem Greg_PPO_Obtained_90_Percent :
  let max_procgen_reward := 240
  let max_coinrun_reward := max_procgen_reward / 2
  let greg_reward := 108
  (greg_reward / max_coinrun_reward * 100) = 90 := by
  sorry

end Greg_PPO_Obtained_90_Percent_l167_167749


namespace intersection_M_N_l167_167747

def M : Set ℝ := { x | -1 ≤ x ∧ x ≤ 10 }
def N : Set ℝ := { x | x > 7 ∨ x < 1 }
def MN_intersection : Set ℝ := { x | (-1 ≤ x ∧ x < 1) ∨ (7 < x ∧ x ≤ 10) }

theorem intersection_M_N :
  M ∩ N = MN_intersection :=
by
  sorry

end intersection_M_N_l167_167747


namespace max_roses_purchase_l167_167529

/--
Given three purchasing options for roses:
1. Individual roses cost $5.30 each.
2. One dozen (12) roses cost $36.
3. Two dozen (24) roses cost $50.
Given a total budget of $680, prove that the maximum number of roses that can be purchased is 317.
-/
noncomputable def max_roses : ℝ := 317

/--
Prove that given the purchasing options and the budget, the maximum number of roses that can be purchased is 317.
-/
theorem max_roses_purchase (individual_cost dozen_cost two_dozen_cost budget : ℝ) 
  (h1 : individual_cost = 5.30) 
  (h2 : dozen_cost = 36) 
  (h3 : two_dozen_cost = 50) 
  (h4 : budget = 680) : 
  max_roses = 317 := 
sorry

end max_roses_purchase_l167_167529


namespace part1_part2_part3_l167_167174

-- Part (1)
theorem part1 (m n : ℤ) (h1 : m - n = -1) : 2 * (m - n)^2 + 18 = 20 := 
sorry

-- Part (2)
theorem part2 (m n : ℤ) (h2 : m^2 + 2 * m * n = 10) (h3 : n^2 + 3 * m * n = 6) : 2 * m^2 + n^2 + 7 * m * n = 26 :=
sorry

-- Part (3)
theorem part3 (a b c m x : ℤ) (h4: ax^5 + bx^3 + cx - 5 = m) (h5: x = -1) : ax^5 + bx^3 + cx - 5 = -m - 10 :=
sorry

end part1_part2_part3_l167_167174


namespace volleyball_team_selection_l167_167932

noncomputable def numberOfWaysToChooseStarters : ℕ :=
  (Nat.choose 13 4 * 3) + (Nat.choose 14 4 * 1)

theorem volleyball_team_selection :
  numberOfWaysToChooseStarters = 3146 := by
  sorry

end volleyball_team_selection_l167_167932


namespace total_cement_used_l167_167639

-- Define the amounts of cement used for Lexi's street and Tess's street
def cement_used_lexis_street : ℝ := 10
def cement_used_tess_street : ℝ := 5.1

-- Prove that the total amount of cement used is 15.1 tons
theorem total_cement_used : cement_used_lexis_street + cement_used_tess_street = 15.1 := sorry

end total_cement_used_l167_167639


namespace geometric_sequence_sum_l167_167572

variable (a : ℕ → ℝ)
variable (q : ℝ)

-- Conditions
def is_geometric_sequence := ∀ n : ℕ, a (n + 1) = a n * q
def cond1 := a 0 + a 1 = 3
def cond2 := a 2 + a 3 = 12
def cond3 := is_geometric_sequence a

theorem geometric_sequence_sum :
  cond1 a →
  cond2 a →
  cond3 a q →
  a 4 + a 5 = 48 :=
by
  intro h1 h2 h3
  sorry

end geometric_sequence_sum_l167_167572


namespace leftover_coverage_l167_167350

variable (bagCoverage lawnLength lawnWidth bagsPurchased : ℕ)

def area_of_lawn (length width : ℕ) : ℕ :=
  length * width

def total_coverage (bagCoverage bags : ℕ) : ℕ :=
  bags * bagCoverage

theorem leftover_coverage :
  let lawnLength := 22
  let lawnWidth := 36
  let bagCoverage := 250
  let bagsPurchased := 4
  let lawnArea := area_of_lawn lawnLength lawnWidth
  let totalSeedCoverage := total_coverage bagCoverage bagsPurchased
  totalSeedCoverage - lawnArea = 208 := by
  sorry

end leftover_coverage_l167_167350


namespace johns_weekly_earnings_l167_167271

-- Define conditions
def days_off_per_week : ℕ := 3
def streaming_hours_per_day : ℕ := 4
def earnings_per_hour : ℕ := 10

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the question as a theorem
theorem johns_weekly_earnings :
  let days_streaming_per_week := days_in_week - days_off_per_week,
      weekly_streaming_hours := days_streaming_per_week * streaming_hours_per_day,
      weekly_earnings := weekly_streaming_hours * earnings_per_hour
  in weekly_earnings = 160 := by
  -- Proof is omitted with 'sorry'
  sorry

end johns_weekly_earnings_l167_167271


namespace find_x_l167_167171

theorem find_x (x : ℕ) (h_odd : x % 2 = 1) (h_pos : 0 < x) :
  (∃ l : List ℕ, l.length = 8 ∧ (∀ n ∈ l, n < 80 ∧ n % 2 = 1) ∧ l.Nodup = true ∧
  (∀ k m, k > 0 → m % 2 = 1 → k * x * m ∈ l)) → x = 5 := by
  sorry

end find_x_l167_167171


namespace soccer_league_teams_l167_167809

theorem soccer_league_teams (n : ℕ) (h : n * (n - 1) / 2 = 45) : n = 10 :=
by
  sorry

end soccer_league_teams_l167_167809


namespace cos_x_plus_2y_is_one_l167_167571

theorem cos_x_plus_2y_is_one
    (x y : ℝ) (a : ℝ) 
    (hx : x ∈ Set.Icc (-Real.pi) Real.pi)
    (hy : y ∈ Set.Icc (-Real.pi) Real.pi)
    (h_eq : 2 * a = x ^ 3 + Real.sin x ∧ 2 * a = (-2 * y) ^ 3 - Real.sin (-2 * y)) :
    Real.cos (x + 2 * y) = 1 := 
sorry

end cos_x_plus_2y_is_one_l167_167571


namespace probability_of_9_heads_in_12_flips_l167_167486

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| _, 0       => 1
| 0, _       => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^12

-- Define the number of ways to get exactly 9 heads in 12 flips
def favorable_outcomes : ℕ := binom 12 9

-- Define the probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 220/4096
theorem probability_of_9_heads_in_12_flips : probability = 220 / 4096 := by
  sorry

end probability_of_9_heads_in_12_flips_l167_167486


namespace tim_age_difference_l167_167775

theorem tim_age_difference (j_turned_23_j_turned_35 : ∃ (j_age_when_james_23 : ℕ) (john_age_when_james_23 : ℕ), 
                                          j_age_when_james_23 = 23 ∧ john_age_when_james_23 = 35)
                           (tim_age : ℕ) (tim_age_eq : tim_age = 79)
                           (tim_age_twice_john_age_less_X : ∃ (X : ℕ) (john_age : ℕ), tim_age = 2 * john_age - X) :
  ∃ (X : ℕ), X = 15 :=
by
  sorry

end tim_age_difference_l167_167775


namespace length_of_train_75_l167_167527

variable (L : ℝ) -- Length of the train in meters

-- Condition 1: The train crosses a bridge of length 150 m in 7.5 seconds
def crosses_bridge (L: ℝ) : Prop := (L + 150) / 7.5 = L / 2.5

-- Condition 2: The train crosses a lamp post in 2.5 seconds
def crosses_lamp (L: ℝ) : Prop := L / 2.5 = L / 2.5

theorem length_of_train_75 (L : ℝ) (h1 : crosses_bridge L) (h2 : crosses_lamp L) : L = 75 := 
by 
  sorry

end length_of_train_75_l167_167527


namespace area_unpainted_region_l167_167451

theorem area_unpainted_region
  (width_board_1 : ℝ)
  (width_board_2 : ℝ)
  (cross_angle_degrees : ℝ)
  (unpainted_area : ℝ)
  (h1 : width_board_1 = 5)
  (h2 : width_board_2 = 7)
  (h3 : cross_angle_degrees = 45)
  (h4 : unpainted_area = (49 * Real.sqrt 2) / 2) : 
  unpainted_area = (width_board_2 * ((width_board_1 * Real.sqrt 2) / 2)) / 2 :=
sorry

end area_unpainted_region_l167_167451


namespace correct_operation_l167_167820

variables (a b : ℝ)

theorem correct_operation : 5 * a * b - 3 * a * b = 2 * a * b :=
by sorry

end correct_operation_l167_167820


namespace largest_four_digit_by_two_moves_l167_167313

def moves (n : Nat) (d1 d2 d3 d4 : Nat) : Prop :=
  ∃ x y : ℕ, d1 = x → d2 = y → n = 1405 → (x ≤ 2 ∧ y ≤ 2)

theorem largest_four_digit_by_two_moves :
  ∃ n : ℕ, moves 1405 1 4 0 5 ∧ n = 7705 :=
by
  sorry

end largest_four_digit_by_two_moves_l167_167313


namespace isosceles_triangle_l167_167764

theorem isosceles_triangle (a b c : ℝ) (h : a^2 * (b - c) + b^2 * (c - a) + c^2 * (a - b) = 0) : 
  a = b ∨ b = c ∨ c = a :=
sorry

end isosceles_triangle_l167_167764
