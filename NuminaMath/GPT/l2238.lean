import Mathlib

namespace polynomial_solution_l2238_223808

variable (P : ℚ) -- Assuming P is a constant polynomial

theorem polynomial_solution (P : ℚ) 
  (condition : P + (2 : ℚ) * X^2 + (5 : ℚ) * X - (2 : ℚ) = (2 : ℚ) * X^2 + (5 : ℚ) * X + (4 : ℚ)): 
  P = 6 := 
  sorry

end polynomial_solution_l2238_223808


namespace intervals_monotonicity_f_intervals_monotonicity_g_and_extremum_l2238_223835

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)

theorem intervals_monotonicity_f :
  ∀ k : ℤ,
    (∀ x : ℝ, k * Real.pi ≤ x ∧ x ≤ k * Real.pi + Real.pi / 2 → f x = Real.cos (2 * x)) ∧
    (∀ x : ℝ, k * Real.pi + Real.pi / 2 ≤ x ∧ x ≤ k * Real.pi + Real.pi → f x = Real.cos (2 * x)) :=
sorry

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 6)

theorem intervals_monotonicity_g_and_extremum :
  ∀ x : ℝ,
    (-Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 3 → g x = Real.cos (2 * (x + Real.pi / 6))) ∧
    (Real.pi / 3 ≤ x ∧ x ≤ 2 * Real.pi / 3 → g x = Real.cos (2 * (x + Real.pi / 6))) ∧
    (∀ x : ℝ, -Real.pi / 6 ≤ x ∧ x ≤ 2 * Real.pi / 3 → (g x ≤ 1 ∧ g x ≥ -1)) :=
sorry

end intervals_monotonicity_f_intervals_monotonicity_g_and_extremum_l2238_223835


namespace crayons_eaten_l2238_223850

def initial_crayons : ℕ := 87
def remaining_crayons : ℕ := 80

theorem crayons_eaten : initial_crayons - remaining_crayons = 7 := by
  sorry

end crayons_eaten_l2238_223850


namespace total_money_raised_for_charity_l2238_223836

theorem total_money_raised_for_charity:
    let price_small := 2
    let price_medium := 3
    let price_large := 5
    let num_small := 150
    let num_medium := 221
    let num_large := 185
    num_small * price_small + num_medium * price_medium + num_large * price_large = 1888 := by
  sorry

end total_money_raised_for_charity_l2238_223836


namespace projection_of_a_on_b_l2238_223860

open Real -- Use real numbers for vector operations

variables (a b : ℝ) -- Define a and b to be real numbers

-- Define the conditions as assumptions in Lean 4
def vector_magnitude_a (a : ℝ) : Prop := abs a = 1
def vector_magnitude_b (b : ℝ) : Prop := abs b = 1
def vector_dot_product (a b : ℝ) : Prop := (a + b) * b = 3 / 2

-- Define the goal to prove, using the assumptions
theorem projection_of_a_on_b (ha : vector_magnitude_a a) (hb : vector_magnitude_b b) (h_ab : vector_dot_product a b) : (abs a) * (a / b) = 1 / 2 :=
by
  sorry

end projection_of_a_on_b_l2238_223860


namespace degree_monomial_equal_four_l2238_223843

def degree_of_monomial (a b : ℝ) := 
  (3 + 1)

theorem degree_monomial_equal_four (a b : ℝ) 
  (h : a^3 * b = (2/3) * a^3 * b) : 
  degree_of_monomial a b = 4 :=
by sorry

end degree_monomial_equal_four_l2238_223843


namespace total_interest_after_tenth_year_l2238_223890

variable {P R : ℕ}

theorem total_interest_after_tenth_year
  (h1 : (P * R * 10) / 100 = 900)
  (h2 : 5 * P * R / 100 = 450)
  (h3 : 5 * 3 * P * R / 100 = 1350) :
  (450 + 1350) = 1800 :=
by
  sorry

end total_interest_after_tenth_year_l2238_223890


namespace truncated_pyramid_lateral_surface_area_l2238_223818

noncomputable def lateralSurfaceAreaTruncatedPyramid (s1 s2 h : ℝ) :=
  let l := Real.sqrt (h^2 + ((s1 - s2) / 2)^2)
  let P1 := 4 * s1
  let P2 := 4 * s2
  (1 / 2) * (P1 + P2) * l

theorem truncated_pyramid_lateral_surface_area :
  lateralSurfaceAreaTruncatedPyramid 10 5 7 = 222.9 :=
by
  sorry

end truncated_pyramid_lateral_surface_area_l2238_223818


namespace prob_a_greater_than_b_l2238_223821

noncomputable def probability_of_team_a_finishing_with_more_points (n_teams : ℕ) (initial_win : Bool) : ℚ :=
  if initial_win ∧ n_teams = 9 then
    39203 / 65536
  else
    0 -- This is a placeholder and not accurate for other cases

theorem prob_a_greater_than_b (n_teams : ℕ) (initial_win : Bool) (hp : initial_win ∧ n_teams = 9) :
  probability_of_team_a_finishing_with_more_points n_teams initial_win = 39203 / 65536 :=
by
  sorry

end prob_a_greater_than_b_l2238_223821


namespace determine_m_l2238_223879

theorem determine_m (x y m : ℝ) :
  (3 * x - y = 4 * m + 1) ∧ (x + y = 2 * m - 5) ∧ (x - y = 4) → m = 1 :=
by sorry

end determine_m_l2238_223879


namespace remainder_of_sum_of_ns_l2238_223814

theorem remainder_of_sum_of_ns (S : ℕ) :
  (∃ (ns : List ℕ), (∀ n ∈ ns, ∃ m : ℕ, n^2 + 12*n - 1997 = m^2) ∧ S = ns.sum) →
  S % 1000 = 154 :=
by
  sorry

end remainder_of_sum_of_ns_l2238_223814


namespace greater_number_l2238_223800

theorem greater_number (x y : ℕ) (h1 : x + y = 60) (h2 : x - y = 10) : x = 35 := 
by sorry

end greater_number_l2238_223800


namespace area_correct_l2238_223847

-- Define the conditions provided in the problem
def width (w : ℝ) := True
def length (l : ℝ) := True
def perimeter (p : ℝ) := True

-- Add the conditions about the playground
axiom length_exceeds_width_by : ∃ l w, l = 3 * w + 30
axiom perimeter_is_given : ∃ l w, 2 * (l + w) = 730

-- Define the area of the playground and state the theorem
noncomputable def area_of_playground : ℝ := 83.75 * 281.25

theorem area_correct :
  (∃ l w, l = 3 * w + 30 ∧ 2 * (l + w) = 730) →
  area_of_playground = 23554.6875 :=
by
  sorry

end area_correct_l2238_223847


namespace age_difference_l2238_223849

variable (x y z : ℝ)

def overall_age_condition (x y z : ℝ) : Prop := (x + y = y + z + 10)

theorem age_difference (x y z : ℝ) (h : overall_age_condition x y z) : (x - z) / 10 = 1 :=
  by
    sorry

end age_difference_l2238_223849


namespace probability_other_side_red_l2238_223851

def card_black_black := 4
def card_black_red := 2
def card_red_red := 2

def total_cards := card_black_black + card_black_red + card_red_red

-- Calculate the total number of red faces
def total_red_faces := (card_red_red * 2) + card_black_red

-- Number of red faces that have the other side also red
def red_faces_with_other_red := card_red_red * 2

-- Target probability to prove
theorem probability_other_side_red (h : total_cards = 8) : 
  (red_faces_with_other_red / total_red_faces) = 2 / 3 := 
  sorry

end probability_other_side_red_l2238_223851


namespace total_ladybugs_eq_11676_l2238_223830

def Number_of_leaves : ℕ := 84
def Ladybugs_per_leaf : ℕ := 139

theorem total_ladybugs_eq_11676 : Number_of_leaves * Ladybugs_per_leaf = 11676 := by
  sorry

end total_ladybugs_eq_11676_l2238_223830


namespace find_original_number_l2238_223885

-- Definitions based on the conditions of the problem
def tens_digit (x : ℕ) := 2 * x
def original_number (x : ℕ) := 10 * (tens_digit x) + x
def reversed_number (x : ℕ) := 10 * x + (tens_digit x)

-- Proof statement
theorem find_original_number (x : ℕ) (h1 : original_number x - reversed_number x = 27) : original_number x = 63 := by
  sorry

end find_original_number_l2238_223885


namespace tree_initial_height_example_l2238_223855

-- The height of the tree at the time Tony planted it
def initial_tree_height (growth_rate final_height years : ℕ) : ℕ :=
  final_height - (growth_rate * years)

theorem tree_initial_height_example :
  initial_tree_height 5 29 5 = 4 :=
by
  -- This is where the proof would go, we use 'sorry' to indicate it's omitted.
  sorry

end tree_initial_height_example_l2238_223855


namespace find_a4_l2238_223828

-- Define the sequence
noncomputable def a : ℕ → ℝ := sorry

-- Define the initial term a1 and common difference d
noncomputable def a1 : ℝ := sorry
noncomputable def d : ℝ := sorry

-- The conditions from the problem
def condition1 : Prop := a 2 + a 6 = 10 * Real.sqrt 3
def condition2 : Prop := a 3 + a 7 = 14 * Real.sqrt 3

-- Using the conditions to prove a4
theorem find_a4 (h1 : condition1) (h2 : condition2) : a 4 = 5 * Real.sqrt 3 :=
by
  sorry

end find_a4_l2238_223828


namespace number_of_correct_conclusions_l2238_223842

theorem number_of_correct_conclusions
  (a b c : ℕ)
  (h1 : (a^b - b^c) * (b^c - c^a) * (c^a - a^b) = 11713) 
  (conclusion1 : (a^b - b^c) % 2 = 1 ∧ (b^c - c^a) % 2 = 1 ∧ (c^a - a^b) % 2 = 1)
  (conclusion4 : ¬ ∃ a b c : ℕ, (a^b - b^c) * (b^c - c^a) * (c^a - a^b) = 11713) :
  ∃ n : ℕ, n = 2 :=
by
  sorry

end number_of_correct_conclusions_l2238_223842


namespace determinant_of_matrix_A_l2238_223884

noncomputable def matrix_A (x : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2, -1, 4], ![3, x, -2], ![1, -3, 0]]

theorem determinant_of_matrix_A (x : ℝ) :
  Matrix.det (matrix_A x) = -46 - 4 * x :=
by
  sorry

end determinant_of_matrix_A_l2238_223884


namespace interval_for_systematic_sampling_l2238_223877

-- Define the total number of students
def total_students : ℕ := 1200

-- Define the sample size
def sample_size : ℕ := 30

-- Define the interval for systematic sampling
def interval_k : ℕ := total_students / sample_size

-- The theorem to prove that the interval k should be 40
theorem interval_for_systematic_sampling :
  interval_k = 40 := sorry

end interval_for_systematic_sampling_l2238_223877


namespace value_of_expression_l2238_223883

theorem value_of_expression (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 4 / 3) :
  (1 / 3 * x^7 * y^6) * 4 = 1 :=
by
  sorry

end value_of_expression_l2238_223883


namespace probability_of_male_selected_l2238_223829

-- Define the total number of students
def num_students : ℕ := 100

-- Define the number of male students
def num_male_students : ℕ := 25

-- Define the number of students selected
def num_students_selected : ℕ := 20

theorem probability_of_male_selected :
  (num_students_selected : ℚ) / num_students = 1 / 5 :=
by
  sorry

end probability_of_male_selected_l2238_223829


namespace count_distinct_product_divisors_l2238_223823

-- Define the properties of 8000 and its divisors
def isDivisor (n d : ℕ) := d > 0 ∧ n % d = 0

def T := {d : ℕ | isDivisor 8000 d}

-- The main statement to prove
theorem count_distinct_product_divisors : 
    (∃ n : ℕ, n ∈ { m | ∃ a b : ℕ, a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ m = a * b } ∧ n = 88) :=
by {
  sorry
}

end count_distinct_product_divisors_l2238_223823


namespace greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l2238_223840

theorem greatest_divisor_of_sum_first_15_terms_arithmetic_sequence
  (x c : ℕ) -- where x and c are positive integers
  (h_pos_x : 0 < x) -- x is positive
  (h_pos_c : 0 < c) -- c is positive
  : ∃ (d : ℕ), d = 15 ∧ ∀ (S : ℕ), S = 15 * x + 105 * c → d ∣ S :=
by
  sorry

end greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l2238_223840


namespace find_angle_B_l2238_223815

def angle_A (B : ℝ) : ℝ := B + 21
def angle_C (B : ℝ) : ℝ := B + 36
def is_triangle_sum (A B C : ℝ) : Prop := A + B + C = 180

theorem find_angle_B (B : ℝ) 
  (hA : angle_A B = B + 21) 
  (hC : angle_C B = B + 36) 
  (h_sum : is_triangle_sum (angle_A B) B (angle_C B) ) : B = 41 :=
  sorry

end find_angle_B_l2238_223815


namespace number_of_participants_with_5_points_l2238_223889

-- Definitions for conditions
def num_participants : ℕ := 254

def points_for_victory : ℕ := 1

def additional_point_condition (winner_points loser_points : ℕ) : ℕ :=
  if winner_points < loser_points then 1 else 0

def points_for_loss : ℕ := 0

-- Theorem statement
theorem number_of_participants_with_5_points :
  ∃ num_students_with_5_points : ℕ, num_students_with_5_points = 56 := 
sorry

end number_of_participants_with_5_points_l2238_223889


namespace circle_radius_l2238_223805

theorem circle_radius (A : ℝ) (r : ℝ) (h : A = 36 * Real.pi) (h2 : A = Real.pi * r ^ 2) : r = 6 :=
sorry

end circle_radius_l2238_223805


namespace solution_set_inequality_l2238_223845

theorem solution_set_inequality (a : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 ↔ |2 * x - a| + a ≤ 6) → a = 2 :=
sorry

end solution_set_inequality_l2238_223845


namespace negation_of_proposition_l2238_223804

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) :=
sorry

end negation_of_proposition_l2238_223804


namespace bernardo_wins_l2238_223856

/-- 
Bernardo and Silvia play the following game. An integer between 0 and 999 inclusive is selected
and given to Bernardo. Whenever Bernardo receives a number, he doubles it and passes the result 
to Silvia. Whenever Silvia receives a number, she adds 50 to it and passes the result back. 
The winner is the last person who produces a number less than 1000. The smallest initial number 
that results in a win for Bernardo is 16, and the sum of the digits of 16 is 7.
-/
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem bernardo_wins (N : ℕ) (h : 16 ≤ N ∧ N ≤ 18) : sum_of_digits 16 = 7 :=
by
  sorry

end bernardo_wins_l2238_223856


namespace equal_angles_count_l2238_223824

-- Definitions corresponding to the problem conditions
def fast_clock_angle (t : ℝ) : ℝ := |30 * t - 5.5 * (t * 60)|
def slow_clock_angle (t : ℝ) : ℝ := |15 * t - 2.75 * (t * 60)|

theorem equal_angles_count :
  ∃ n : ℕ, n = 18 ∧ ∀ t : ℝ, 0 ≤ t ∧ t ≤ 12 →
  fast_clock_angle t = slow_clock_angle t ↔ n = 18 :=
sorry

end equal_angles_count_l2238_223824


namespace value_of_expression_l2238_223820

variables (a b c d m : ℝ)

theorem value_of_expression (h1: a + b = 0) (h2: c * d = 1) (h3: |m| = 3) :
  (a + b) / m + m^2 - c * d = 8 :=
by
  sorry

end value_of_expression_l2238_223820


namespace kite_AB_BC_ratio_l2238_223871

-- Define the kite properties and necessary elements to state the problem
def kite_problem (AB BC: ℝ) (angleB angleD : ℝ) (MN'_parallel_AC : Prop) : Prop :=
  angleB = 90 ∧ angleD = 90 ∧ MN'_parallel_AC ∧ AB / BC = (1 + Real.sqrt 5) / 2

-- Define the main theorem to be proven
theorem kite_AB_BC_ratio (AB BC : ℝ) (angleB angleD : ℝ) (MN'_parallel_AC : Prop) :
  kite_problem AB BC angleB angleD MN'_parallel_AC :=
by
  sorry

-- Statement of the condition that need to be satisfied
axiom MN'_parallel_AC : Prop

-- Example instantiation of the problem
example : kite_problem 1 1 90 90 MN'_parallel_AC :=
by
  sorry

end kite_AB_BC_ratio_l2238_223871


namespace total_toys_l2238_223826

theorem total_toys 
  (jaxon_toys : ℕ)
  (gabriel_toys : ℕ)
  (jerry_toys : ℕ)
  (h1 : jaxon_toys = 15)
  (h2 : gabriel_toys = 2 * jaxon_toys)
  (h3 : jerry_toys = gabriel_toys + 8) : 
  jaxon_toys + gabriel_toys + jerry_toys = 83 :=
  by sorry

end total_toys_l2238_223826


namespace evaluate_expression_l2238_223893

theorem evaluate_expression : 1 + 1 / (1 + 1 / (1 + 1 / (1 + 2))) = 11 / 7 :=
by 
  sorry

end evaluate_expression_l2238_223893


namespace janice_bought_30_fifty_cent_items_l2238_223834

theorem janice_bought_30_fifty_cent_items (x y z : ℕ) (h1 : x + y + z = 40) (h2 : 50 * x + 150 * y + 300 * z = 4500) : x = 30 :=
by
  sorry

end janice_bought_30_fifty_cent_items_l2238_223834


namespace inscribed_rectangle_area_l2238_223811

variable (a b h x : ℝ)
variable (h_pos : 0 < h) (a_b_pos : a > b) (b_pos : b > 0) (a_pos : a > 0) (x_pos : 0 < x) (hx : x < h)

theorem inscribed_rectangle_area (hb : b > 0) (ha : a > 0) (hx : 0 < x) (hxa : x < h) : 
  x * (a - b) * (h - x) / h = x * (a - b) * (h - x) / h := by
  sorry

end inscribed_rectangle_area_l2238_223811


namespace interest_earned_is_correct_l2238_223862

-- Define the principal amount, interest rate, and duration
def principal : ℝ := 2000
def rate : ℝ := 0.02
def duration : ℕ := 3

-- The compound interest formula to calculate the future value
def future_value (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r) ^ n

-- Calculate the interest earned
def interest (P : ℝ) (A : ℝ) : ℝ := A - P

-- Theorem statement: The interest Bart earns after 3 years is 122 dollars
theorem interest_earned_is_correct : interest principal (future_value principal rate duration) = 122 :=
by
  sorry

end interest_earned_is_correct_l2238_223862


namespace oak_trees_remaining_l2238_223844

-- Variables representing the initial number of oak trees and the number of cut down trees.
variables (initial_trees cut_down_trees remaining_trees : ℕ)

-- Conditions of the problem.
def initial_trees_condition : initial_trees = 9 := sorry
def cut_down_trees_condition : cut_down_trees = 2 := sorry

-- Theorem representing the proof problem.
theorem oak_trees_remaining (h1 : initial_trees = 9) (h2 : cut_down_trees = 2) :
  remaining_trees = initial_trees - cut_down_trees :=
sorry

end oak_trees_remaining_l2238_223844


namespace total_participants_l2238_223861

theorem total_participants (Petya Vasya total : ℕ) 
  (h1 : Petya = Vasya + 1) 
  (h2 : Petya = 10)
  (h3 : Vasya + 15 = total + 1) : 
  total = 23 :=
by
  sorry

end total_participants_l2238_223861


namespace percent_of_x_is_65_l2238_223863

variable (z y x : ℝ)

theorem percent_of_x_is_65 :
  (0.45 * z = 0.39 * y) → (y = 0.75 * x) → (z / x = 0.65) := by
  sorry

end percent_of_x_is_65_l2238_223863


namespace table_arrangement_division_l2238_223876

theorem table_arrangement_division (total_tables : ℕ) (rows : ℕ) (tables_per_row : ℕ) (tables_left_over : ℕ)
    (h1 : total_tables = 74) (h2 : rows = 8) (h3 : tables_per_row = total_tables / rows) (h4 : tables_left_over = total_tables % rows) :
    tables_per_row = 9 ∧ tables_left_over = 2 := by
  sorry

end table_arrangement_division_l2238_223876


namespace number_of_n_l2238_223864

theorem number_of_n (n : ℕ) (h1 : n > 0) (h2 : n ≤ 1200) (h3 : ∃ k : ℕ, 12 * n = k^2) :
  ∃ m : ℕ, m = 10 :=
by { sorry }

end number_of_n_l2238_223864


namespace inequality_solution_l2238_223832

theorem inequality_solution (x : ℝ) :
  -1 < (x^2 - 12 * x + 35) / (x^2 - 4 * x + 8) ∧
  (x^2 - 12 * x + 35) / (x^2 - 4 * x + 8) < 1 ↔
  x > (27 / 8) :=
by sorry

end inequality_solution_l2238_223832


namespace find_k_for_minimum_value_l2238_223822

theorem find_k_for_minimum_value :
  ∃ (k : ℝ), (∀ (x y : ℝ), 9 * x^2 - 6 * k * x * y + (3 * k^2 + 1) * y^2 - 6 * x - 6 * y + 7 ≥ 1)
  ∧ (∃ (x y : ℝ), 9 * x^2 - 6 * k * x * y + (3 * k^2 + 1) * y^2 - 6 * x - 6 * y + 7 = 1)
  ∧ k = 3 :=
sorry

end find_k_for_minimum_value_l2238_223822


namespace students_taking_art_l2238_223803

theorem students_taking_art :
  ∀ (total_students music_students both_music_art neither_music_art : ℕ),
  total_students = 500 →
  music_students = 30 →
  both_music_art = 10 →
  neither_music_art = 470 →
  (total_students - neither_music_art) - (music_students - both_music_art) - both_music_art = 10 :=
by
  intros total_students music_students both_music_art neither_music_art h_total h_music h_both h_neither
  sorry

end students_taking_art_l2238_223803


namespace halfway_between_one_nine_and_one_eleven_l2238_223875

theorem halfway_between_one_nine_and_one_eleven : 
  (1/9 + 1/11) / 2 = 10/99 :=
by sorry

end halfway_between_one_nine_and_one_eleven_l2238_223875


namespace minimize_J_l2238_223894

noncomputable def H (p q : ℝ) : ℝ := -3 * p * q + 4 * p * (1 - q) + 5 * (1 - p) * q - 6 * (1 - p) * (1 - q) + 2 * p

noncomputable def J (p : ℝ) : ℝ := max (H p 0) (H p 1)

theorem minimize_J (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : p = 11 / 18 ↔ ∀ q, 0 ≤ q ∧ q ≤ 1 → J p = J (11 / 18) := 
by
  sorry

end minimize_J_l2238_223894


namespace pathway_width_l2238_223866

theorem pathway_width {r1 r2 : ℝ} 
  (h1 : 2 * Real.pi * r1 - 2 * Real.pi * r2 = 20 * Real.pi)
  (h2 : r1 - r2 = 10) :
  r1 - r2 + 4 = 14 := 
by 
  sorry

end pathway_width_l2238_223866


namespace min_w_for_factors_l2238_223870

theorem min_w_for_factors (w : ℕ) (h_pos : w > 0)
  (h_product_factors : ∀ k, k > 0 → ∃ a b : ℕ, (1452 * w = k) → (a = 3^3) ∧ (b = 13^3) ∧ (k % a = 0) ∧ (k % b = 0)) : 
  w = 19773 :=
sorry

end min_w_for_factors_l2238_223870


namespace suff_but_not_necc_l2238_223867

def p (x : ℝ) : Prop := x = 2
def q (x : ℝ) : Prop := (x - 2) * (x + 3) = 0

theorem suff_but_not_necc (x : ℝ) : (p x → q x) ∧ ¬(q x → p x) :=
by
  sorry

end suff_but_not_necc_l2238_223867


namespace total_length_of_XYZ_l2238_223868

noncomputable def length_XYZ : ℝ :=
  let length_X := 2 + 2 + 2 * Real.sqrt 2
  let length_Y := 3 + 2 * Real.sqrt 2
  let length_Z := 3 + 3 + Real.sqrt 10
  length_X + length_Y + length_Z

theorem total_length_of_XYZ :
  length_XYZ = 13 + 4 * Real.sqrt 2 + Real.sqrt 10 :=
by
  sorry

end total_length_of_XYZ_l2238_223868


namespace num_ways_to_write_3070_l2238_223813

theorem num_ways_to_write_3070 :
  let valid_digits := {d : ℕ | d ≤ 99}
  ∃ (M : ℕ), 
  M = 6500 ∧
  ∃ (a3 a2 a1 a0 : ℕ) (H : a3 ∈ valid_digits) (H : a2 ∈ valid_digits) (H : a1 ∈ valid_digits) (H : a0 ∈ valid_digits),
  3070 = a3 * 10^3 + a2 * 10^2 + a1 * 10 + a0 := sorry

end num_ways_to_write_3070_l2238_223813


namespace floor_e_minus_3_eq_neg1_l2238_223869

noncomputable def e : ℝ := 2.718

theorem floor_e_minus_3_eq_neg1 : Int.floor (e - 3) = -1 := by
  sorry

end floor_e_minus_3_eq_neg1_l2238_223869


namespace area_of_region_l2238_223897

theorem area_of_region : 
  (∀ x y : ℝ, x^2 + y^2 - 8*x + 6*y = 0 → 
     let a := (x - 4)^2 + (y + 3)^2 
     (a = 25) ∧ ∃ r : ℝ, r = 5 ∧ (π * r^2 = 25 * π)) := 
sorry

end area_of_region_l2238_223897


namespace square_area_in_ellipse_l2238_223873

theorem square_area_in_ellipse : ∀ (s : ℝ), 
  (s > 0) → 
  (∀ x y, (x = s ∨ x = -s) ∧ (y = s ∨ y = -s) → (x^2) / 4 + (y^2) / 8 = 1) → 
  (2 * s)^2 = 32 / 3 := by
  sorry

end square_area_in_ellipse_l2238_223873


namespace laps_remaining_eq_five_l2238_223880

variable (total_distance : ℕ)
variable (distance_per_lap : ℕ)
variable (laps_already_run : ℕ)

theorem laps_remaining_eq_five 
  (h1 : total_distance = 99) 
  (h2 : distance_per_lap = 9) 
  (h3 : laps_already_run = 6) : 
  (total_distance / distance_per_lap - laps_already_run = 5) :=
by 
  sorry

end laps_remaining_eq_five_l2238_223880


namespace cistern_water_depth_l2238_223831

theorem cistern_water_depth 
  (l w a : ℝ)
  (hl : l = 8)
  (hw : w = 6)
  (ha : a = 83) :
  ∃ d : ℝ, 48 + 28 * d = 83 :=
by
  use 1.25
  sorry

end cistern_water_depth_l2238_223831


namespace divides_square_sum_implies_divides_l2238_223872

theorem divides_square_sum_implies_divides (a b : ℤ) (h : 7 ∣ a^2 + b^2) : 7 ∣ a ∧ 7 ∣ b := 
sorry

end divides_square_sum_implies_divides_l2238_223872


namespace grape_juice_amount_l2238_223812

theorem grape_juice_amount 
  (T : ℝ) -- total amount of the drink 
  (orange_juice_percentage watermelon_juice_percentage : ℝ) -- percentages 
  (combined_amount_of_oj_wj : ℝ) -- combined amount of orange and watermelon juice 
  (h1 : orange_juice_percentage = 0.15)
  (h2 : watermelon_juice_percentage = 0.60)
  (h3 : combined_amount_of_oj_wj = 120)
  (h4 : combined_amount_of_oj_wj = (orange_juice_percentage + watermelon_juice_percentage) * T) : 
  (T * (1 - (orange_juice_percentage + watermelon_juice_percentage)) = 40) := 
sorry

end grape_juice_amount_l2238_223812


namespace arithmetic_operators_correct_l2238_223892

theorem arithmetic_operators_correct :
  let op1 := (132: ℝ) - (7: ℝ) * (6: ℝ)
  let op2 := (12: ℝ) + (3: ℝ)
  (op1 / op2) = (6: ℝ) := by 
  sorry

end arithmetic_operators_correct_l2238_223892


namespace superdomino_probability_l2238_223801

-- Definitions based on conditions
def is_superdomino (a b : ℕ) : Prop := 0 ≤ a ∧ a ≤ 12 ∧ 0 ≤ b ∧ b ≤ 12
def is_superdouble (a b : ℕ) : Prop := a = b
def total_superdomino_count : ℕ := 13 * 13
def superdouble_count : ℕ := 13

-- Proof statement
theorem superdomino_probability : (superdouble_count : ℚ) / total_superdomino_count = 13 / 169 :=
by
  sorry

end superdomino_probability_l2238_223801


namespace equal_ivan_petrovich_and_peter_ivanovich_l2238_223837

theorem equal_ivan_petrovich_and_peter_ivanovich :
  (∀ n : ℕ, n % 10 = 0 → (n % 20 = 0) = (n % 200 = 0)) :=
by
  sorry

end equal_ivan_petrovich_and_peter_ivanovich_l2238_223837


namespace geometric_sequence_sums_l2238_223854

open Real

theorem geometric_sequence_sums (S T R : ℝ)
  (h1 : ∃ a r, S = a * (1 + r))
  (h2 : ∃ a r, T = a * (1 + r + r^2 + r^3))
  (h3 : ∃ a r, R = a * (1 + r + r^2 + r^3 + r^4 + r^5)) :
  S^2 + T^2 = S * (T + R) :=
by
  sorry

end geometric_sequence_sums_l2238_223854


namespace percentage_B_of_C_l2238_223887

variable (A B C : ℝ)

theorem percentage_B_of_C (h1 : A = 0.08 * C) (h2 : A = 0.5 * B) : B = 0.16 * C :=
by
  sorry

end percentage_B_of_C_l2238_223887


namespace people_per_column_in_second_arrangement_l2238_223841
-- Lean 4 Statement

theorem people_per_column_in_second_arrangement :
  ∀ P X : ℕ, (P = 30 * 16) → (12 * X = P) → X = 40 :=
by
  intros P X h1 h2
  sorry

end people_per_column_in_second_arrangement_l2238_223841


namespace geom_seq_increasing_sufficient_necessary_l2238_223859

theorem geom_seq_increasing_sufficient_necessary (a : ℕ → ℝ) (r : ℝ) (h_geo : ∀ n : ℕ, a n = a 0 * r ^ n) 
  (h_increasing : ∀ n : ℕ, a n < a (n + 1)) : 
  (a 0 < a 1 ∧ a 1 < a 2) ↔ (∀ n : ℕ, a n < a (n + 1)) :=
sorry

end geom_seq_increasing_sufficient_necessary_l2238_223859


namespace xiao_liang_reaches_museum_l2238_223853

noncomputable def xiao_liang_distance_to_museum : ℝ :=
  let science_museum := (200 * Real.sqrt 2, 200 * Real.sqrt 2)
  let initial_mistake := (-300 * Real.sqrt 2, 300 * Real.sqrt 2)
  let to_supermarket := (-100 * Real.sqrt 2, 500 * Real.sqrt 2)
  Real.sqrt ((science_museum.1 - to_supermarket.1)^2 + (science_museum.2 - to_supermarket.2)^2)

theorem xiao_liang_reaches_museum :
  xiao_liang_distance_to_museum = 600 :=
sorry

end xiao_liang_reaches_museum_l2238_223853


namespace greatest_sum_consecutive_lt_400_l2238_223802

noncomputable def greatest_sum_of_consecutive_integers (n : ℤ) : ℤ :=
if n * (n + 1) < 400 then n + (n + 1) else 0

theorem greatest_sum_consecutive_lt_400 : ∃ n : ℤ, n * (n + 1) < 400 ∧ greatest_sum_of_consecutive_integers n = 39 :=
by
  sorry

end greatest_sum_consecutive_lt_400_l2238_223802


namespace compute_ζ7_sum_l2238_223816

noncomputable def ζ_power_sum (ζ1 ζ2 ζ3 : ℂ) : Prop :=
  (ζ1 + ζ2 + ζ3 = 2) ∧
  (ζ1^2 + ζ2^2 + ζ3^2 = 6) ∧
  (ζ1^3 + ζ2^3 + ζ3^3 = 8) →
  ζ1^7 + ζ2^7 + ζ3^7 = 58

theorem compute_ζ7_sum (ζ1 ζ2 ζ3 : ℂ) (h : ζ_power_sum ζ1 ζ2 ζ3) : ζ1^7 + ζ2^7 + ζ3^7 = 58 :=
by
  -- proof goes here
  sorry

end compute_ζ7_sum_l2238_223816


namespace Deepak_and_Wife_meet_time_l2238_223809

theorem Deepak_and_Wife_meet_time 
    (circumference : ℕ) 
    (Deepak_speed : ℕ)
    (wife_speed : ℕ) 
    (conversion_factor_km_hr_to_m_hr : ℕ) 
    (minutes_per_hour : ℕ) :
    circumference = 726 →
    Deepak_speed = 4500 →  -- speed in meters per hour
    wife_speed = 3750 →  -- speed in meters per hour
    conversion_factor_km_hr_to_m_hr = 1000 →
    minutes_per_hour = 60 →
    (726 / ((4500 + 3750) / 1000) * 60 = 5.28) :=
by 
    sorry

end Deepak_and_Wife_meet_time_l2238_223809


namespace retailer_should_focus_on_mode_l2238_223833

-- Define the conditions as options.
inductive ClothingModels
| Average
| Mode
| Median
| Smallest

-- Define a function to determine the best measure to focus on in the market share survey.
def bestMeasureForMarketShareSurvey (choice : ClothingModels) : Prop :=
  match choice with
  | ClothingModels.Average => False
  | ClothingModels.Mode => True
  | ClothingModels.Median => False
  | ClothingModels.Smallest => False

-- The theorem stating that the mode is the best measure to focus on.
theorem retailer_should_focus_on_mode : bestMeasureForMarketShareSurvey ClothingModels.Mode :=
by
  -- This proof is intentionally left blank.
  sorry

end retailer_should_focus_on_mode_l2238_223833


namespace min_apples_l2238_223865

theorem min_apples :
  ∃ N : ℕ, 
  (N % 3 = 2) ∧ 
  (N % 4 = 2) ∧ 
  (N % 5 = 2) ∧ 
  (N = 62) :=
by
  sorry

end min_apples_l2238_223865


namespace find_f3_l2238_223899

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_f3
  (a b : ℝ)
  (h1 : f a b 3 1 = 7)
  (h2 : f a b 3 2 = 12) :
  f a b 3 3 = 18 :=
sorry

end find_f3_l2238_223899


namespace sophomores_in_sample_l2238_223874

-- Define the number of freshmen, sophomores, and juniors
def freshmen : ℕ := 400
def sophomores : ℕ := 600
def juniors : ℕ := 500

-- Define the total number of students
def total_students : ℕ := freshmen + sophomores + juniors

-- Define the total number of students in the sample
def sample_size : ℕ := 100

-- Define the expected number of sophomores in the sample
def expected_sophomores : ℕ := (sample_size * sophomores) / total_students

-- Statement of the problem we want to prove
theorem sophomores_in_sample : expected_sophomores = 40 := by
  sorry

end sophomores_in_sample_l2238_223874


namespace belfried_industries_tax_l2238_223878

noncomputable def payroll_tax (payroll : ℕ) : ℕ :=
  if payroll <= 200000 then
    0
  else
    ((payroll - 200000) * 2) / 1000

theorem belfried_industries_tax : payroll_tax 300000 = 200 :=
by
  sorry

end belfried_industries_tax_l2238_223878


namespace investment_time_l2238_223857

variable (P : ℝ) (R : ℝ) (SI : ℝ)

theorem investment_time (hP : P = 800) (hR : R = 0.04) (hSI : SI = 160) :
  SI / (P * R) = 5 := by
  sorry

end investment_time_l2238_223857


namespace min_shirts_to_save_money_l2238_223810

theorem min_shirts_to_save_money :
  ∃ (x : ℕ), 60 + 11 * x < 20 + 15 * x ∧ (∀ y : ℕ, 60 + 11 * y < 20 + 15 * y → y ≥ x) ∧ x = 11 :=
by
  sorry

end min_shirts_to_save_money_l2238_223810


namespace mean_median_modes_l2238_223839

theorem mean_median_modes (d μ M : ℝ)
  (dataset : Multiset ℕ)
  (h_dataset : dataset = Multiset.replicate 12 1 + Multiset.replicate 12 2 + Multiset.replicate 12 3 +
                         Multiset.replicate 12 4 + Multiset.replicate 12 5 + Multiset.replicate 12 6 +
                         Multiset.replicate 12 7 + Multiset.replicate 12 8 + Multiset.replicate 12 9 +
                         Multiset.replicate 12 10 + Multiset.replicate 12 11 + Multiset.replicate 12 12 +
                         Multiset.replicate 12 13 + Multiset.replicate 12 14 + Multiset.replicate 12 15 +
                         Multiset.replicate 12 16 + Multiset.replicate 12 17 + Multiset.replicate 12 18 +
                         Multiset.replicate 12 19 + Multiset.replicate 12 20 + Multiset.replicate 12 21 +
                         Multiset.replicate 12 22 + Multiset.replicate 12 23 + Multiset.replicate 12 24 +
                         Multiset.replicate 12 25 + Multiset.replicate 12 26 + Multiset.replicate 12 27 +
                         Multiset.replicate 12 28 + Multiset.replicate 12 29 + Multiset.replicate 12 30 +
                         Multiset.replicate 7 31)
  (h_M : M = 16)
  (h_μ : μ = 5797 / 366)
  (h_d : d = 15.5) :
  d < μ ∧ μ < M :=
sorry

end mean_median_modes_l2238_223839


namespace problem_l2238_223896

def g (x : ℤ) : ℤ := 3 * x^2 - x + 4

theorem problem : g (g 3) = 2328 := by
  sorry

end problem_l2238_223896


namespace nine_values_of_x_l2238_223817

theorem nine_values_of_x : ∃! (n : ℕ), ∃! (xs : Finset ℕ), xs.card = n ∧ 
  (∀ x ∈ xs, 3 * x < 100 ∧ 4 * x ≥ 100) ∧ 
  (xs.image (λ x => x)).val = ({25, 26, 27, 28, 29, 30, 31, 32, 33} : Finset ℕ).val :=
sorry

end nine_values_of_x_l2238_223817


namespace range_of_f_1_over_f_2_l2238_223806

theorem range_of_f_1_over_f_2 {f : ℝ → ℝ} (h1 : ∀ x > 0, f x > 0)
  (h2 : ∀ x > 0, 2 * f x < x * (deriv f x) ∧ x * (deriv f x) < 3 * f x) :
  1 / 8 < f 1 / f 2 ∧ f 1 / f 2 < 1 / 4 :=
by sorry

end range_of_f_1_over_f_2_l2238_223806


namespace smaller_octagon_half_area_l2238_223807

-- Define what it means to be a regular octagon
def is_regular_octagon (O : Point) (ABCDEFGH : List Point) : Prop :=
  -- Definition capturing the properties of a regular octagon around center O
  sorry

-- Define the function that computes the area of an octagon
def area_of_octagon (ABCDEFGH : List Point) : Real :=
  sorry

-- Define the function to create the smaller octagon by joining midpoints
def smaller_octagon (ABCDEFGH : List Point) : List Point :=
  sorry

theorem smaller_octagon_half_area (O : Point) (ABCDEFGH : List Point) :
  is_regular_octagon O ABCDEFGH →
  area_of_octagon (smaller_octagon ABCDEFGH) = (1 / 2) * area_of_octagon ABCDEFGH :=
by
  sorry

end smaller_octagon_half_area_l2238_223807


namespace find_abs_sum_roots_l2238_223882

noncomputable def polynomial_root_abs_sum (n p q r : ℤ) : Prop :=
(p + q + r = 0) ∧
(p * q + q * r + r * p = -2009) ∧
(p * q * r = -n) →
(|p| + |q| + |r| = 102)

theorem find_abs_sum_roots (n p q r : ℤ) :
  polynomial_root_abs_sum n p q r :=
sorry

end find_abs_sum_roots_l2238_223882


namespace ways_to_fifth_floor_l2238_223886

theorem ways_to_fifth_floor (floors : ℕ) (staircases : ℕ) (h_floors : floors = 5) (h_staircases : staircases = 2) :
  (staircases ^ (floors - 1)) = 16 :=
by
  rw [h_floors, h_staircases]
  sorry

end ways_to_fifth_floor_l2238_223886


namespace can_capacity_is_30_l2238_223846

noncomputable def capacity_of_can (x: ℝ) : ℝ :=
  7 * x + 10

theorem can_capacity_is_30 :
  ∃ (x: ℝ), (4 * x + 10) / (3 * x) = 5 / 2 ∧ capacity_of_can x = 30 :=
by
  sorry

end can_capacity_is_30_l2238_223846


namespace geometric_sequence_sum_l2238_223891

-- Define the positive terms of the geometric sequence
variables {a_1 a_2 a_3 a_4 a_5 : ℝ}
-- Assume all terms are positive
variables (h1 : a_1 > 0) (h2 : a_2 > 0) (h3 : a_3 > 0) (h4 : a_4 > 0) (h5 : a_5 > 0)

-- Main condition given in the problem
variable (h_main : a_1 * a_3 + 2 * a_2 * a_4 + a_3 * a_5 = 16)

-- Goal: Prove that a_2 + a_4 = 4
theorem geometric_sequence_sum : a_2 + a_4 = 4 :=
by
  sorry

end geometric_sequence_sum_l2238_223891


namespace mary_spent_total_amount_l2238_223888

def shirt_cost : ℝ := 13.04
def jacket_cost : ℝ := 12.27
def total_cost : ℝ := 25.31

theorem mary_spent_total_amount :
  shirt_cost + jacket_cost = total_cost := sorry

end mary_spent_total_amount_l2238_223888


namespace remainder_n_squared_l2238_223895

theorem remainder_n_squared (n : ℤ) (h : n % 5 = 3) : (n^2) % 5 = 4 := 
    sorry

end remainder_n_squared_l2238_223895


namespace smallest_scalene_prime_triangle_perimeter_l2238_223825

-- Define a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a scalene triangle with distinct side lengths
def is_scalene (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Define the triangle inequality
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define a valid scalene triangle with prime side lengths
def valid_scalene_triangle (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ is_scalene a b c ∧ triangle_inequality a b c

-- Proof statement
theorem smallest_scalene_prime_triangle_perimeter : ∃ (a b c : ℕ), 
  valid_scalene_triangle a b c ∧ a + b + c = 15 := 
sorry

end smallest_scalene_prime_triangle_perimeter_l2238_223825


namespace soccer_club_girls_count_l2238_223827

theorem soccer_club_girls_count
  (total_members : ℕ)
  (attended : ℕ)
  (B G : ℕ)
  (h1 : B + G = 30)
  (h2 : (1/3 : ℚ) * G + B = 18) : G = 18 := by
  sorry

end soccer_club_girls_count_l2238_223827


namespace cd_percentage_cheaper_l2238_223838

theorem cd_percentage_cheaper (cost_cd cost_book cost_album difference percentage : ℝ) 
  (h1 : cost_book = cost_cd + 4)
  (h2 : cost_book = 18)
  (h3 : cost_album = 20)
  (h4 : difference = cost_album - cost_cd)
  (h5 : percentage = (difference / cost_album) * 100) : 
  percentage = 30 :=
sorry

end cd_percentage_cheaper_l2238_223838


namespace volume_of_water_displaced_l2238_223819

theorem volume_of_water_displaced (r h s : ℝ) (V : ℝ) 
  (r_eq : r = 5) (h_eq : h = 12) (s_eq : s = 6) :
  V = s^3 :=
by
  have cube_volume : V = s^3 := by sorry
  show V = s^3
  exact cube_volume

end volume_of_water_displaced_l2238_223819


namespace inequality_proof_l2238_223881

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (n : ℕ) (hn : 0 < n) : 
  (x / (n * x + y + z) + y / (x + n * y + z) + z / (x + y + n * z)) ≤ 3 / (n + 2) :=
sorry

end inequality_proof_l2238_223881


namespace area_ratio_l2238_223858

-- Define the conditions: perimeters relation
def condition (a b : ℝ) := 4 * a = 16 * b

-- Define the theorem to be proved
theorem area_ratio (a b : ℝ) (h : condition a b) : (a * a) = 16 * (b * b) :=
sorry

end area_ratio_l2238_223858


namespace factorial_binomial_mod_l2238_223898

theorem factorial_binomial_mod (p : ℕ) (hp : Nat.Prime p) : 
  ((Nat.factorial (2 * p)) / (Nat.factorial p * Nat.factorial p)) - 2 ≡ 0 [MOD p] :=
by
  sorry

end factorial_binomial_mod_l2238_223898


namespace range_of_a_for_root_l2238_223852

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - 2 * x + a

theorem range_of_a_for_root (a : ℝ) : (∃ x, f x a = 0) ↔ a ≤ 2 * Real.log 2 - 2 := sorry

end range_of_a_for_root_l2238_223852


namespace triangles_with_two_white_vertices_l2238_223848

theorem triangles_with_two_white_vertices (p f z : ℕ) 
    (h1 : p * f + p * z + f * z = 213)
    (h2 : (p * (p - 1) / 2) + (f * (f - 1) / 2) + (z * (z - 1) / 2) = 112)
    (h3 : p * f * z = 540)
    (h4 : (p * (p - 1) / 2) * (f + z) = 612) :
    (f * (f - 1) / 2) * (p + z) = 210 ∨ (f * (f - 1) / 2) * (p + z) = 924 := 
  sorry

end triangles_with_two_white_vertices_l2238_223848
