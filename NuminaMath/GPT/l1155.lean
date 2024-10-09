import Mathlib

namespace cost_of_paving_l1155_115509

-- Definitions based on the given conditions
def length : ℝ := 6.5
def width : ℝ := 2.75
def rate : ℝ := 600

-- Theorem statement to prove the cost of paving
theorem cost_of_paving : length * width * rate = 10725 := by
  -- Calculation steps would go here, but we omit them with sorry
  sorry

end cost_of_paving_l1155_115509


namespace prob_8th_roll_last_l1155_115591

-- Define the conditions as functions or constants
def prob_diff_rolls : ℚ := 5/6
def prob_same_roll : ℚ := 1/6

-- Define the theorem stating the probability of the 8th roll being the last roll
theorem prob_8th_roll_last : (1 : ℚ) * prob_diff_rolls^6 * prob_same_roll = 15625 / 279936 := 
sorry

end prob_8th_roll_last_l1155_115591


namespace debate_team_selections_l1155_115584

theorem debate_team_selections
  (A_selected C_selected B_selected E_selected : Prop)
  (h1: A_selected ∨ C_selected)
  (h2: B_selected ∨ E_selected)
  (h3: ¬ (B_selected ∧ E_selected) ∧ ¬ (C_selected ∧ E_selected))
  (not_B_selected : ¬ B_selected) :
  A_selected ∧ E_selected :=
by
  sorry

end debate_team_selections_l1155_115584


namespace num_A_is_9_l1155_115561

-- Define the total number of animals
def total_animals : ℕ := 17

-- Define the number of animal B
def num_B : ℕ := 8

-- Define the number of animal A
def num_A : ℕ := total_animals - num_B

-- Statement to prove
theorem num_A_is_9 : num_A = 9 :=
by
  sorry

end num_A_is_9_l1155_115561


namespace sum_of_equal_numbers_l1155_115593

theorem sum_of_equal_numbers (a b : ℝ) (h1 : (12 + 25 + 18 + a + b) / 5 = 20) (h2 : a = b) : a + b = 45 :=
sorry

end sum_of_equal_numbers_l1155_115593


namespace shirt_discount_l1155_115520

theorem shirt_discount (original_price discounted_price : ℕ) 
  (h1 : original_price = 22) 
  (h2 : discounted_price = 16) : 
  original_price - discounted_price = 6 := 
by
  sorry

end shirt_discount_l1155_115520


namespace problem1_problem2_l1155_115583

theorem problem1 : 12 - (-18) + (-7) + (-15) = 8 :=
by sorry

theorem problem2 : (-1)^7 * 2 + (-3)^2 / 9 = -1 :=
by sorry

end problem1_problem2_l1155_115583


namespace total_pizzas_served_l1155_115516

def lunch_pizzas : ℚ := 12.5
def dinner_pizzas : ℚ := 8.25

theorem total_pizzas_served : lunch_pizzas + dinner_pizzas = 20.75 := by
  sorry

end total_pizzas_served_l1155_115516


namespace negation_of_universal_proposition_l1155_115590

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ ∃ x : ℝ, |x| + x^2 < 0 :=
by
  sorry

end negation_of_universal_proposition_l1155_115590


namespace sum_of_three_consecutive_cubes_divisible_by_9_l1155_115512

theorem sum_of_three_consecutive_cubes_divisible_by_9 (n : ℕ) : 
  (n^3 + (n + 1)^3 + (n + 2)^3) % 9 = 0 := 
by
  sorry

end sum_of_three_consecutive_cubes_divisible_by_9_l1155_115512


namespace sector_area_l1155_115529

theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (r : ℝ) (area : ℝ) : 
  arc_length = π / 3 ∧ central_angle = π / 6 → arc_length = central_angle * r → area = 1 / 2 * central_angle * r^2 → area = π / 3 :=
by
  sorry

end sector_area_l1155_115529


namespace technicians_count_l1155_115523

theorem technicians_count {T R : ℕ} (h1 : T + R = 12) (h2 : 2 * T + R = 18) : T = 6 :=
sorry

end technicians_count_l1155_115523


namespace sum_squares_mod_divisor_l1155_115515

-- Define the sum of the squares from 1 to 10
def sum_squares := (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 + 8^2 + 9^2 + 10^2)

-- Define the divisor
def divisor := 11

-- Prove that the remainder of sum_squares when divided by divisor is 0
theorem sum_squares_mod_divisor : sum_squares % divisor = 0 :=
by
  sorry

end sum_squares_mod_divisor_l1155_115515


namespace range_of_a_l1155_115580

open Real

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x ^ 2 + a * x - 1 < 0) ↔ -4 < a ∧ a ≤ 0 := 
by
  sorry

end range_of_a_l1155_115580


namespace find_f_8_l1155_115534

def f (n : ℕ) : ℕ := n^2 - 3 * n + 20

theorem find_f_8 : f 8 = 60 := 
by 
sorry

end find_f_8_l1155_115534


namespace value_of_double_operation_l1155_115539

def op1 (x : ℝ) : ℝ := 9 - x
def op2 (x : ℝ) : ℝ := x - 9

theorem value_of_double_operation :
  op2 (op1 10) = -10 := 
by 
  sorry

end value_of_double_operation_l1155_115539


namespace combined_height_difference_is_correct_l1155_115564

-- Define the initial conditions
def uncle_height : ℕ := 72
def james_initial_height : ℕ := (2 * uncle_height) / 3
def sarah_initial_height : ℕ := (3 * james_initial_height) / 4

-- Define the growth spurts
def james_growth_spurt : ℕ := 10
def sarah_growth_spurt : ℕ := 12

-- Define their heights after growth spurts
def james_final_height : ℕ := james_initial_height + james_growth_spurt
def sarah_final_height : ℕ := sarah_initial_height + sarah_growth_spurt

-- Define the combined height of James and Sarah after growth spurts
def combined_height : ℕ := james_final_height + sarah_final_height

-- Define the combined height difference between uncle and both James and Sarah now
def combined_height_difference : ℕ := combined_height - uncle_height

-- Lean statement to prove the combined height difference
theorem combined_height_difference_is_correct : combined_height_difference = 34 := by
  -- proof omitted
  sorry

end combined_height_difference_is_correct_l1155_115564


namespace molecular_weight_C4H10_l1155_115555

theorem molecular_weight_C4H10 (molecular_weight_six_moles : ℝ) (h : molecular_weight_six_moles = 390) :
  molecular_weight_six_moles / 6 = 65 :=
by
  -- proof to be filled in here
  sorry

end molecular_weight_C4H10_l1155_115555


namespace problem_l1155_115510

variable (a b c : ℝ)

def a_def : a = Real.log (1 / 2) := sorry
def b_def : b = Real.exp (1 / Real.exp 1) := sorry
def c_def : c = Real.exp (-2) := sorry

theorem problem (ha : a = Real.log (1 / 2)) 
               (hb : b = Real.exp (1 / Real.exp 1)) 
               (hc : c = Real.exp (-2)) : 
               a < c ∧ c < b := 
by
  rw [ha, hb, hc]
  sorry

end problem_l1155_115510


namespace projectile_height_reach_l1155_115586

theorem projectile_height_reach (t : ℝ) (h : -16 * t^2 + 64 * t = 25) : t = 3.6 :=
by
  sorry

end projectile_height_reach_l1155_115586


namespace solve_inequality_I_solve_inequality_II_l1155_115504

def f (x : ℝ) : ℝ := |x - 1| - |2 * x + 3|

theorem solve_inequality_I (x : ℝ) : f x > 2 ↔ -2 < x ∧ x < -4 / 3 :=
by sorry

theorem solve_inequality_II (a : ℝ) : ∀ x, f x ≤ (3 / 2) * a^2 - a ↔ a ≥ 5 / 3 :=
by sorry

end solve_inequality_I_solve_inequality_II_l1155_115504


namespace cookies_in_jar_l1155_115517

theorem cookies_in_jar (C : ℕ) (h : C - 1 = (C + 5) / 2) : C = 7 :=
by
  -- Proof goes here
  sorry

end cookies_in_jar_l1155_115517


namespace joe_lowest_dropped_score_l1155_115522

theorem joe_lowest_dropped_score (A B C D : ℕ) 
  (hmean_before : (A + B + C + D) / 4 = 35)
  (hmean_after : (A + B + C) / 3 = 40)
  (hdrop : D = min A (min B (min C D))) :
  D = 20 :=
by sorry

end joe_lowest_dropped_score_l1155_115522


namespace percentage_increase_is_50_l1155_115518

-- Definition of the given values
def original_time : ℕ := 30
def new_time : ℕ := 45

-- Assertion stating that the percentage increase is 50%
theorem percentage_increase_is_50 :
  (new_time - original_time) * 100 / original_time = 50 := 
sorry

end percentage_increase_is_50_l1155_115518


namespace negation_of_neither_even_l1155_115588

variable (a b : Nat)

def is_even (n : Nat) : Prop :=
  n % 2 = 0

theorem negation_of_neither_even 
  (H : ¬ (¬ is_even a ∧ ¬ is_even b)) : is_even a ∨ is_even b :=
sorry

end negation_of_neither_even_l1155_115588


namespace fish_swim_eastward_l1155_115563

-- Define the conditions
variables (E : ℕ)
variable (total_fish_left : ℕ := 2870)
variable (fish_westward : ℕ := 1800)
variable (fish_north : ℕ := 500)
variable (fishwestward_not_caught : ℕ := fish_westward / 4)
variable (fishnorth_not_caught : ℕ := fish_north)
variable (fish_tobe_left_after_caught : ℕ := total_fish_left - fishwestward_not_caught - fishnorth_not_caught)

-- Define the theorem to prove
theorem fish_swim_eastward (h : 3 / 5 * E = fish_tobe_left_after_caught) : E = 3200 := 
by
  sorry

end fish_swim_eastward_l1155_115563


namespace surface_area_of_solid_l1155_115573

-- Define a unit cube and the number of cubes
def unitCube : Type := { faces : ℕ // faces = 6 }
def numCubes : ℕ := 10

-- Define the surface area contribution from different orientations
def surfaceAreaFacingUs (cubes : ℕ) : ℕ := 2 * cubes -- faces towards and away
def verticalSidesArea (heightCubes : ℕ) : ℕ := 2 * heightCubes -- left and right vertical sides
def horizontalSidesArea (widthCubes : ℕ) : ℕ := 2 * widthCubes -- top and bottom horizontal sides

-- Define the surface area for the given configuration of 10 cubes
def totalSurfaceArea (cubes : ℕ) (height : ℕ) (width : ℕ) : ℕ :=
  (surfaceAreaFacingUs cubes) + (verticalSidesArea height) + (horizontalSidesArea width)

-- Assumptions based on problem description
def heightCubes : ℕ := 3
def widthCubes : ℕ := 4

-- The theorem we want to prove
theorem surface_area_of_solid : totalSurfaceArea numCubes heightCubes widthCubes = 34 := by
  sorry

end surface_area_of_solid_l1155_115573


namespace opposite_neg_one_half_l1155_115535

def opposite (x : ℚ) : ℚ := -x

theorem opposite_neg_one_half :
  opposite (- 1 / 2) = 1 / 2 := by
  sorry

end opposite_neg_one_half_l1155_115535


namespace part_a_part_b_l1155_115569

theorem part_a (a : ℕ) : ¬ (∃ k : ℕ, k^2 = ( ((a^2 - 3)^3 + 1)^a - 1)) :=
sorry

theorem part_b (a : ℕ) : ¬ (∃ k : ℕ, k^2 = ( ((a^2 - 3)^3 + 1)^(a + 1) - 1)) :=
sorry

end part_a_part_b_l1155_115569


namespace jordan_time_for_7_miles_l1155_115574

noncomputable def time_for_7_miles (jordan_miles : ℕ) (jordan_time : ℤ) : ℤ :=
  jordan_miles * jordan_time 

theorem jordan_time_for_7_miles :
  ∃ jordan_time : ℤ, (time_for_7_miles 7 (16 / 3)) = 112 / 3 :=
by
  sorry

end jordan_time_for_7_miles_l1155_115574


namespace carla_wins_one_game_l1155_115597

/-
We are given the conditions:
Alice, Bob, and Carla each play each other twice in a round-robin format.
Alice won 5 games and lost 3 games.
Bob won 6 games and lost 2 games.
Carla lost 5 games.
We need to prove that Carla won 1 game.
-/

theorem carla_wins_one_game (games_per_match : Nat) 
                            (total_players : Nat)
                            (alice_wins : Nat) 
                            (alice_losses : Nat) 
                            (bob_wins : Nat) 
                            (bob_losses : Nat) 
                            (carla_losses : Nat) :
  (games_per_match = 2) → 
  (total_players = 3) → 
  (alice_wins = 5) → 
  (alice_losses = 3) → 
  (bob_wins = 6) → 
  (bob_losses = 2) → 
  (carla_losses = 5) → 
  ∃ (carla_wins : Nat), 
  carla_wins = 1 := 
by
  intros 
    games_match_eq total_players_eq 
    alice_wins_eq alice_losses_eq 
    bob_wins_eq bob_losses_eq 
    carla_losses_eq
  sorry

end carla_wins_one_game_l1155_115597


namespace horizontal_asymptote_condition_l1155_115559

open Polynomial

def polynomial_deg_with_horiz_asymp (p : Polynomial ℝ) : Prop :=
  degree p ≤ 4

theorem horizontal_asymptote_condition (p : Polynomial ℝ) :
  polynomial_deg_with_horiz_asymp p :=
sorry

end horizontal_asymptote_condition_l1155_115559


namespace madeline_flower_count_l1155_115562

theorem madeline_flower_count 
    (r w : ℕ) 
    (b_percent : ℝ) 
    (total : ℕ) 
    (h_r : r = 4)
    (h_w : w = 2)
    (h_b_percent : b_percent = 0.40)
    (h_total : r + w + (b_percent * total) = total) : 
    total = 10 :=
by 
    sorry

end madeline_flower_count_l1155_115562


namespace area_of_square_is_25_l1155_115587

-- Define side length of the square
def sideLength : ℝ := 5

-- Define the area of the square
def area_of_square (side : ℝ) : ℝ := side * side

-- Prove the area of the square with side length 5 is 25 square meters
theorem area_of_square_is_25 : area_of_square sideLength = 25 := by
  sorry

end area_of_square_is_25_l1155_115587


namespace gold_silver_weight_problem_l1155_115527

theorem gold_silver_weight_problem (x y : ℕ) (h1 : 9 * x = 11 * y) (h2 : (10 * y + x) - (8 * x + y) = 13) :
  9 * x = 11 * y ∧ (10 * y + x) - (8 * x + y) = 13 :=
by
  refine ⟨h1, h2⟩

end gold_silver_weight_problem_l1155_115527


namespace inverse_value_l1155_115596

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 25 / (7 + 2 * x)

-- Define the goal of the proof
theorem inverse_value {g : ℝ → ℝ}
  (h : ∀ y, g (g⁻¹ y) = y) :
  ((g⁻¹ 5)⁻¹) = -1 :=
by
  sorry

end inverse_value_l1155_115596


namespace roots_polynomial_sum_l1155_115546

theorem roots_polynomial_sum (p q : ℂ) (hp : p^2 - 6 * p + 10 = 0) (hq : q^2 - 6 * q + 10 = 0) :
  p^4 + p^5 * q^3 + p^3 * q^5 + q^4 = 16056 := by
  sorry

end roots_polynomial_sum_l1155_115546


namespace john_unanswered_questions_l1155_115537

theorem john_unanswered_questions
  (x y z : ℕ)
  (h1 : 5 * x + 2 * z = 93)
  (h2 : 4 * x - y = 54)
  (h3 : x + y + z = 30) : z = 9 :=
sorry

end john_unanswered_questions_l1155_115537


namespace min_sum_ab_72_l1155_115594

theorem min_sum_ab_72 (a b : ℤ) (h : a * b = 72) : a + b ≥ -17 := sorry

end min_sum_ab_72_l1155_115594


namespace not_quadratic_eq3_l1155_115558

-- Define the equations as functions or premises
def eq1 (x : ℝ) := 9 * x^2 = 7 * x
def eq2 (y : ℝ) := abs (y^2) = 8
def eq3 (y : ℝ) := 3 * y * (y - 1) = y * (3 * y + 1)
def eq4 (x : ℝ) := abs 2 * (x^2 + 1) = abs 10

-- Define what it means to be a quadratic equation
def is_quadratic (eq : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, eq x = (a * x^2 + b * x + c = 0)

-- Prove that eq3 is not a quadratic equation
theorem not_quadratic_eq3 : ¬ is_quadratic eq3 :=
sorry

end not_quadratic_eq3_l1155_115558


namespace cost_of_concessions_l1155_115500

theorem cost_of_concessions (total_cost : ℕ) (adult_ticket_cost : ℕ) (child_ticket_cost : ℕ) (num_adults : ℕ) (num_children : ℕ) :
  total_cost = 76 →
  adult_ticket_cost = 10 →
  child_ticket_cost = 7 →
  num_adults = 5 →
  num_children = 2 →
  total_cost - (num_adults * adult_ticket_cost + num_children * child_ticket_cost) = 12 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end cost_of_concessions_l1155_115500


namespace find_ellipse_equation_l1155_115585

-- Definitions based on conditions
def ellipse_centered_at_origin (x y : ℝ) (m n : ℝ) := m * x ^ 2 + n * y ^ 2 = 1

def passes_through_points_A_and_B (m n : ℝ) := 
  (ellipse_centered_at_origin 0 (-2) m n) ∧ (ellipse_centered_at_origin (3 / 2) (-1) m n)

-- Statement to be proved
theorem find_ellipse_equation : 
  ∃ (m n : ℝ), (m > 0) ∧ (n > 0) ∧ (m ≠ n) ∧ 
  passes_through_points_A_and_B m n ∧ 
  m = 1 / 3 ∧ n = 1 / 4 :=
by sorry

end find_ellipse_equation_l1155_115585


namespace greatest_odd_factors_l1155_115513

theorem greatest_odd_factors (n : ℕ) (h1 : n < 1000) (h2 : ∀ k : ℕ, k * k = n → (k < 32)) :
  n = 31 * 31 :=
by
  sorry

end greatest_odd_factors_l1155_115513


namespace solution_set_inequality_l1155_115506

def f (a : ℝ) (x : ℝ) : ℝ := a * x + 2

theorem solution_set_inequality (a x : ℝ) (h : Set.Ioo (-1 : ℝ) (2 : ℝ) = {x | |f a x| < 6}) : 
    {x | f a x ≤ 1} = {x | x ≥ 1 / 4} :=
sorry

end solution_set_inequality_l1155_115506


namespace solution_set_ineq_l1155_115521

theorem solution_set_ineq (x : ℝ) : (1 / x > 1) ↔ (0 < x ∧ x < 1) :=
by
  sorry

end solution_set_ineq_l1155_115521


namespace range_of_a_l1155_115554

def inequality_system_has_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, (x + a ≥ 0) ∧ (1 - 2 * x > x - 2)

theorem range_of_a (a : ℝ) : inequality_system_has_solution a ↔ a > -1 :=
by
  sorry

end range_of_a_l1155_115554


namespace min_value_a_b_l1155_115578

theorem min_value_a_b (x y a b : ℝ) (h1 : 2 * x - y + 2 ≥ 0) (h2 : 8 * x - y - 4 ≤ 0) 
  (h3 : x ≥ 0) (h4 : y ≥ 0) (h5 : a > 0) (h6 : b > 0) (h7 : a * x + y = 8) : 
  a + b ≥ 4 :=
sorry

end min_value_a_b_l1155_115578


namespace johnny_marbles_l1155_115549

def num_ways_to_choose_marbles (n k : ℕ) : ℕ := Nat.choose n k

theorem johnny_marbles :
  num_ways_to_choose_marbles 7 3 = 35 :=
by
  sorry

end johnny_marbles_l1155_115549


namespace sum_of_distinct_prime_factors_l1155_115507

-- Definition of the expression
def expression : ℤ := 7^4 - 7^2

-- Statement of the theorem
theorem sum_of_distinct_prime_factors : 
  Nat.sum (List.eraseDup (Nat.factors expression.natAbs)) = 12 := 
by 
  sorry

end sum_of_distinct_prime_factors_l1155_115507


namespace smallest_of_product_and_sum_l1155_115589

theorem smallest_of_product_and_sum (a b c : ℤ) 
  (h1 : a * b * c = 32) 
  (h2 : a + b + c = 3) : 
  a = -4 ∨ b = -4 ∨ c = -4 :=
sorry

end smallest_of_product_and_sum_l1155_115589


namespace tank_capacity_l1155_115503

theorem tank_capacity (C : ℝ) (h₁ : 3/4 * C + 7 = 7/8 * C) : C = 56 :=
by
  sorry

end tank_capacity_l1155_115503


namespace no_integer_roots_of_polynomial_l1155_115505

theorem no_integer_roots_of_polynomial :
  ¬ ∃ x : ℤ, x^3 - 4 * x^2 - 14 * x + 28 = 0 :=
by
  sorry

end no_integer_roots_of_polynomial_l1155_115505


namespace determine_a_l1155_115530

theorem determine_a :
  ∃ (a b c d : ℕ), 
  (18 ^ a) * (9 ^ (4 * a - 1)) * (27 ^ c) = (2 ^ 6) * (3 ^ b) * (7 ^ d) ∧ 
  a * c = 4 / (2 * b + d) ∧ 
  b^2 - 4 * a * c = d ∧ 
  a = 6 := 
by
  sorry

end determine_a_l1155_115530


namespace no_positive_integer_solutions_l1155_115595

theorem no_positive_integer_solutions :
  ¬ ∃ (x1 x2 : ℕ), 903 * x1 + 731 * x2 = 1106 := by
  sorry

end no_positive_integer_solutions_l1155_115595


namespace Bethany_total_riding_hours_l1155_115525

-- Define daily riding hours
def Monday_hours : Nat := 1
def Wednesday_hours : Nat := 1
def Friday_hours : Nat := 1
def Tuesday_hours : Nat := 1 / 2
def Thursday_hours : Nat := 1 / 2
def Saturday_hours : Nat := 2

-- Define total weekly hours
def weekly_hours : Nat :=
  Monday_hours + Wednesday_hours + Friday_hours + (Tuesday_hours + Thursday_hours) + Saturday_hours

-- Definition to account for the 2-week period
def total_hours (weeks : Nat) : Nat := weeks * weekly_hours

-- Prove that Bethany rode 12 hours over 2 weeks
theorem Bethany_total_riding_hours : total_hours 2 = 12 := by
  sorry

end Bethany_total_riding_hours_l1155_115525


namespace slope_interval_non_intersect_l1155_115565

noncomputable def parabola (x : ℝ) : ℝ := x^2 + 5

def Q : ℝ × ℝ := (10, 10)

theorem slope_interval_non_intersect (r s : ℝ) (h : ∀ m : ℝ,
  ¬∃ x : ℝ, parabola x = m * (x - 10) + 10 ↔ r < m ∧ m < s) :
  r + s = 40 :=
sorry

end slope_interval_non_intersect_l1155_115565


namespace simplify_fraction_l1155_115599

theorem simplify_fraction (x : ℝ) :
  ((x + 2) / 4) + ((3 - 4 * x) / 3) = (18 - 13 * x) / 12 := by
  sorry

end simplify_fraction_l1155_115599


namespace harry_books_l1155_115581

theorem harry_books : ∀ (H : ℝ), 
  (H + 2 * H + H / 2 = 175) → 
  H = 50 :=
by
  intros H h_sum
  sorry

end harry_books_l1155_115581


namespace patch_area_difference_l1155_115532

theorem patch_area_difference :
  let alan_length := 30
  let alan_width := 50
  let betty_length := 35
  let betty_width := 40
  let alan_area := alan_length * alan_width
  let betty_area := betty_length * betty_width
  let area_difference := alan_area - betty_area
  area_difference = 100 := 
by
  -- Definitions
  let alan_length := 30
  let alan_width := 50
  let betty_length := 35
  let betty_width := 40
  let alan_area := alan_length * alan_width
  let betty_area := betty_length * betty_width
  let area_difference := alan_area - betty_area
  -- Proof (intentionally left as sorry)
  -- Lean should be able to use the initial definitions to verify the theorem statement.
  sorry

end patch_area_difference_l1155_115532


namespace calculate_value_l1155_115592

theorem calculate_value : 2 * (75 * 1313 - 25 * 1313) = 131300 := 
by 
  sorry

end calculate_value_l1155_115592


namespace abs_fraction_lt_one_l1155_115533

theorem abs_fraction_lt_one (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) : 
  |(x - y) / (1 - x * y)| < 1 := 
sorry

end abs_fraction_lt_one_l1155_115533


namespace domain_of_f_intervals_of_monotonicity_extremal_values_l1155_115576

noncomputable def f (x : ℝ) := (1 / 2) * x ^ 2 - 5 * x + 4 * Real.log x 

theorem domain_of_f : ∀ x, 0 < x → f x = (1 / 2) * x ^ 2 - 5 * x + 4 * Real.log x :=
by
  intro x hx
  exact rfl

theorem intervals_of_monotonicity :
  (∀ x, 0 < x ∧ x < 1 → f x < f 1) ∧
  (∀ x, 1 < x ∧ x < 4 → f x > f 1 ∧ f x < f 4) ∧
  (∀ x, 4 < x → f x > f 4) :=
sorry

theorem extremal_values :
  (f 1 = - (9 / 2)) ∧ 
  (f 4 = -12 + 4 * Real.log 4) :=
sorry

end domain_of_f_intervals_of_monotonicity_extremal_values_l1155_115576


namespace angle_between_line_and_plane_l1155_115598

variables (α β : ℝ) -- angles in radians
-- Definitions to capture the provided conditions
def dihedral_angle (α : ℝ) : Prop := true -- The angle between the planes γ₁ and γ₂
def angle_with_edge (β : ℝ) : Prop := true -- The angle between line AB and edge l

-- The angle between line AB and the plane γ₂
theorem angle_between_line_and_plane (α β : ℝ) (h1 : dihedral_angle α) (h2 : angle_with_edge β) : 
  ∃ θ : ℝ, θ = Real.arcsin (Real.sin α * Real.sin β) :=
by
  sorry

end angle_between_line_and_plane_l1155_115598


namespace inequality_holds_for_all_xyz_in_unit_interval_l1155_115566

theorem inequality_holds_for_all_xyz_in_unit_interval :
  ∀ (x y z : ℝ), (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → (0 ≤ z ∧ z ≤ 1) → 
  (x / (y + z + 1) + y / (z + x + 1) + z / (x + y + 1) ≤ 1 - (1 - x) * (1 - y) * (1 - z)) :=
by
  intros x y z hx hy hz
  sorry

end inequality_holds_for_all_xyz_in_unit_interval_l1155_115566


namespace widgets_production_l1155_115528

variables (A B C : ℝ)
variables (P : ℝ)

-- Conditions provided
def condition1 : Prop := 7 * A + 11 * B = 305
def condition2 : Prop := 8 * A + 22 * C = P

-- The question we need to answer
def question : Prop :=
  ∃ Q : ℝ, Q = 8 * (A + B + C)

theorem widgets_production (h1 : condition1 A B) (h2 : condition2 A C P) :
  question A B C :=
sorry

end widgets_production_l1155_115528


namespace lines_skew_iff_a_ne_20_l1155_115567

variable {t u a : ℝ}
-- Definitions for the lines
def line1 (t : ℝ) (a : ℝ) := (2 + 3 * t, 3 + 4 * t, a + 5 * t)
def line2 (u : ℝ) := (3 + 6 * u, 2 + 5 * u, 1 + 2 * u)

-- Condition for lines to intersect
def lines_intersect (t u a : ℝ) :=
  2 + 3 * t = 3 + 6 * u ∧
  3 + 4 * t = 2 + 5 * u ∧
  a + 5 * t = 1 + 2 * u

-- The main theorem stating when lines are skew
theorem lines_skew_iff_a_ne_20 (a : ℝ) :
  (¬ ∃ t u : ℝ, lines_intersect t u a) ↔ a ≠ 20 := 
by 
  sorry

end lines_skew_iff_a_ne_20_l1155_115567


namespace right_triangle_area_perimeter_l1155_115557

theorem right_triangle_area_perimeter (a b : ℕ) (h₁ : a = 36) (h₂ : b = 48) : 
  (1/2) * (a * b) = 864 ∧ a + b + Nat.sqrt (a * a + b * b) = 144 := by
  sorry

end right_triangle_area_perimeter_l1155_115557


namespace area_of_triangle_l1155_115575

theorem area_of_triangle :
  let A := (10, 1)
  let B := (15, 8)
  let C := (10, 8)
  ∃ (area : ℝ), 
  area = 17.5 ∧ 
  area = 1 / 2 * (abs (B.1 - C.1)) * (abs (C.2 - A.2)) :=
by
  sorry

end area_of_triangle_l1155_115575


namespace number_multiplied_by_any_integer_results_in_itself_l1155_115514

theorem number_multiplied_by_any_integer_results_in_itself (N : ℤ) (h : ∀ (x : ℤ), N * x = N) : N = 0 :=
  sorry

end number_multiplied_by_any_integer_results_in_itself_l1155_115514


namespace geom_seq_sum_2016_2017_l1155_115540

noncomputable def geom_seq (n : ℕ) (a1 q : ℝ) : ℝ := a1 * q ^ (n - 1)

noncomputable def sum_geometric_seq (a1 q : ℝ) (n : ℕ) : ℝ :=
if q = 1 then
  a1 * n
else
  a1 * (1 - q ^ n) / (1 - q)

theorem geom_seq_sum_2016_2017 :
  (a1 = 2) →
  (geom_seq 2 a1 q + geom_seq 5 a1 q = 0) →
  sum_geometric_seq a1 q 2016 + sum_geometric_seq a1 q 2017 = 2 :=
by
  sorry

end geom_seq_sum_2016_2017_l1155_115540


namespace jars_needed_l1155_115543

-- Definitions based on the given conditions
def total_cherry_tomatoes : ℕ := 56
def cherry_tomatoes_per_jar : ℕ := 8

-- Lean theorem to prove the question
theorem jars_needed (total_cherry_tomatoes cherry_tomatoes_per_jar : ℕ) (h1 : total_cherry_tomatoes = 56) (h2 : cherry_tomatoes_per_jar = 8) : (total_cherry_tomatoes / cherry_tomatoes_per_jar) = 7 := by
  -- Proof omitted
  sorry

end jars_needed_l1155_115543


namespace final_volume_solution_l1155_115552

variables (V2 V12 V_final : ℝ)

-- Given conditions
def V2_percent_solution (V2 : ℝ) := true
def V12_percent_solution (V12 : ℝ) := V12 = 18
def mixture_equation (V2 V12 V_final : ℝ) := 0.02 * V2 + 0.12 * V12 = 0.05 * V_final
def total_volume (V2 V12 V_final : ℝ) := V_final = V2 + V12

theorem final_volume_solution (V2 V_final : ℝ) (hV2: V2_percent_solution V2)
    (hV12 : V12_percent_solution V12) (h_mix : mixture_equation V2 V12 V_final)
    (h_total : total_volume V2 V12 V_final) : V_final = 60 :=
sorry

end final_volume_solution_l1155_115552


namespace problem_expression_eq_zero_l1155_115582

variable {x y : ℝ}

theorem problem_expression_eq_zero (h : x * y ≠ 0) : 
    ( ( (x^2 - 1) / x ) * ( (y^2 - 1) / y ) ) - 
    ( ( (x^2 - 1) / y ) * ( (y^2 - 1) / x ) ) = 0 :=
by
  sorry

end problem_expression_eq_zero_l1155_115582


namespace misread_number_l1155_115550

theorem misread_number (X : ℕ) :
  (average_10_initial : ℕ) = 18 →
  (incorrect_read : ℕ) = 26 →
  (average_10_correct : ℕ) = 22 →
  (10 * 22 - 10 * 18 = X + 26 - 26) →
  X = 66 :=
by sorry

end misread_number_l1155_115550


namespace negation_of_even_sum_l1155_115531

variables (a b : Int)

def is_even (n : Int) : Prop := ∃ k : Int, n = 2 * k

theorem negation_of_even_sum (h : ¬(is_even a ∧ is_even b)) : ¬is_even (a + b) :=
sorry

end negation_of_even_sum_l1155_115531


namespace find_a_plus_b_l1155_115577

theorem find_a_plus_b (a b : ℕ) 
  (h1 : 2^(2 * a) + 2^b + 5 = k^2) : a + b = 4 ∨ a + b = 5 :=
sorry

end find_a_plus_b_l1155_115577


namespace betty_needs_more_money_l1155_115547

-- Define the variables and conditions
def wallet_cost : ℕ := 100
def parents_gift : ℕ := 15
def grandparents_gift : ℕ := parents_gift * 2
def initial_betty_savings : ℕ := wallet_cost / 2
def total_savings : ℕ := initial_betty_savings + parents_gift + grandparents_gift

-- Prove that Betty needs 5 more dollars to buy the wallet
theorem betty_needs_more_money : total_savings + 5 = wallet_cost :=
by
  sorry

end betty_needs_more_money_l1155_115547


namespace distinct_remainders_l1155_115571

theorem distinct_remainders (n : ℕ) (hn : 0 < n) : 
  ∀ (i j : ℕ), (i < n) → (j < n) → (2 * i + 1 ≠ 2 * j + 1) → 
  ((2 * i + 1) ^ (2 * i + 1) % 2^n ≠ (2 * j + 1) ^ (2 * j + 1) % 2^n) :=
by
  sorry

end distinct_remainders_l1155_115571


namespace grandfather_age_correct_l1155_115570

-- Let's define the conditions
def xiaowen_age : ℕ := 13
def grandfather_age : ℕ := 5 * xiaowen_age + 8

-- The statement to prove
theorem grandfather_age_correct : grandfather_age = 73 := by
  sorry

end grandfather_age_correct_l1155_115570


namespace find_fg_minus_gf_l1155_115524

def f (x : ℝ) : ℝ := 3 * x^2 + 4 * x - 5
def g (x : ℝ) : ℝ := 2 * x + 1

theorem find_fg_minus_gf (x : ℝ) : f (g x) - g (f x) = 6 * x^2 + 12 * x + 11 := 
by 
  sorry

end find_fg_minus_gf_l1155_115524


namespace union_sets_a_l1155_115502

theorem union_sets_a (P S : Set ℝ) (a : ℝ) :
  P = {1, 5, 10} →
  S = {1, 3, a^2 + 1} →
  S ∪ P = {1, 3, 5, 10} →
  a = 2 ∨ a = -2 ∨ a = 3 ∨ a = -3 :=
by
  intros hP hS hUnion 
  sorry

end union_sets_a_l1155_115502


namespace coeff_x3_in_expansion_l1155_115541

theorem coeff_x3_in_expansion : (Polynomial.coeff ((Polynomial.C 1 - Polynomial.C 2 * Polynomial.X)^6) 3) = -160 := 
by 
  sorry

end coeff_x3_in_expansion_l1155_115541


namespace problem_solution_l1155_115508

-- We assume x and y are real numbers.
variables (x y : ℝ)

-- Our conditions
def condition1 : Prop := |x| - x + y = 6
def condition2 : Prop := x + |y| + y = 8

-- The goal is to prove that x + y = 30 under the given conditions.
theorem problem_solution (hx : condition1 x y) (hy : condition2 x y) : x + y = 30 :=
sorry

end problem_solution_l1155_115508


namespace fourth_guard_distance_l1155_115501

theorem fourth_guard_distance (d1 d2 d3 : ℕ) (d4 : ℕ) (h1 : d1 + d2 + d3 + d4 = 1000) (h2 : d1 + d2 + d3 = 850) : d4 = 150 :=
sorry

end fourth_guard_distance_l1155_115501


namespace sqrt_nine_over_four_l1155_115511

theorem sqrt_nine_over_four (x : ℝ) : x = 3 / 2 ∨ x = - (3 / 2) ↔ x * x = 9 / 4 :=
by {
  sorry
}

end sqrt_nine_over_four_l1155_115511


namespace cos_double_angle_l1155_115572

variable {α : ℝ}

theorem cos_double_angle (h1 : (Real.tan α - (1 / Real.tan α) = 3 / 2)) (h2 : (α > π / 4) ∧ (α < π / 2)) :
  Real.cos (2 * α) = -3 / 5 := 
sorry

end cos_double_angle_l1155_115572


namespace maximize_S_n_at_24_l1155_115553

noncomputable def a_n (n : ℕ) : ℝ := 142 + (n - 1) * (-2)
noncomputable def b_n (n : ℕ) : ℝ := 142 + (n - 1) * (-6)
noncomputable def S_n (n : ℕ) : ℝ := (n / 2.0) * (2 * 142 + (n - 1) * (-6))

theorem maximize_S_n_at_24 : ∀ (n : ℕ), S_n n ≤ S_n 24 :=
by sorry

end maximize_S_n_at_24_l1155_115553


namespace distribution_of_books_l1155_115545

theorem distribution_of_books :
  let A := 2 -- number of identical art albums (type A)
  let B := 3 -- number of identical stamp albums (type B)
  let friends := 4 -- number of friends
  let total_ways := 5 -- total number of ways to distribute books 
  (A + B) = friends + 1 →
  total_ways = 5 := 
by
  intros A B friends total_ways h
  sorry

end distribution_of_books_l1155_115545


namespace sum_of_valid_single_digit_z_l1155_115519

theorem sum_of_valid_single_digit_z :
  let valid_z (z : ℕ) := z < 10 ∧ (16 + z) % 3 = 0
  let sum_z := (Finset.filter valid_z (Finset.range 10)).sum id
  sum_z = 15 :=
by
  -- Proof steps are omitted
  sorry

end sum_of_valid_single_digit_z_l1155_115519


namespace percentage_cut_l1155_115556

def original_budget : ℝ := 840
def cut_amount : ℝ := 588

theorem percentage_cut : (cut_amount / original_budget) * 100 = 70 :=
by
  sorry

end percentage_cut_l1155_115556


namespace minimum_value_expression_l1155_115526

theorem minimum_value_expression {a b c : ℤ} (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  4 * (a^2 + b^2 + c^2) - (a + b + c)^2 = 8 := 
sorry

end minimum_value_expression_l1155_115526


namespace correct_statement_D_l1155_115548

def is_correct_option (n : ℕ) := n = 4

theorem correct_statement_D : is_correct_option 4 :=
  sorry

end correct_statement_D_l1155_115548


namespace walking_speed_is_4_l1155_115551

def distance : ℝ := 20
def total_time : ℝ := 3.75
def running_distance : ℝ := 10
def running_speed : ℝ := 8
def walking_distance : ℝ := 10

theorem walking_speed_is_4 (W : ℝ) 
  (H1 : running_distance + walking_distance = distance)
  (H2 : running_speed > 0)
  (H3 : walking_distance > 0)
  (H4 : W > 0)
  (H5 : walking_distance / W + running_distance / running_speed = total_time) :
  W = 4 :=
by sorry

end walking_speed_is_4_l1155_115551


namespace loss_percentage_is_30_l1155_115560

theorem loss_percentage_is_30
  (cost_price : ℝ)
  (selling_price : ℝ)
  (h1 : cost_price = 1900)
  (h2 : selling_price = 1330) :
  (cost_price - selling_price) / cost_price * 100 = 30 :=
by
  -- This is a placeholder for the actual proof
  sorry

end loss_percentage_is_30_l1155_115560


namespace arithmetic_seq_term_298_eq_100_l1155_115568

-- Define the arithmetic sequence
def arithmetic_seq (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Define the specific sequence given in the problem
def a_n (n : ℕ) : ℕ := arithmetic_seq 1 3 n

-- State the theorem
theorem arithmetic_seq_term_298_eq_100 : a_n 100 = 298 :=
by
  -- Proof will be filled in
  sorry

end arithmetic_seq_term_298_eq_100_l1155_115568


namespace triangle_is_obtuse_l1155_115542

-- Define the sides of the triangle with the given ratio
def a (x : ℝ) := 3 * x
def b (x : ℝ) := 4 * x
def c (x : ℝ) := 6 * x

-- The theorem statement
theorem triangle_is_obtuse (x : ℝ) (hx : 0 < x) : 
  (a x)^2 + (b x)^2 < (c x)^2 :=
by
  sorry

end triangle_is_obtuse_l1155_115542


namespace alice_total_cost_usd_is_correct_l1155_115544

def tea_cost_yen : ℕ := 250
def sandwich_cost_yen : ℕ := 350
def conversion_rate : ℕ := 100
def total_cost_usd (tea_cost_yen sandwich_cost_yen conversion_rate : ℕ) : ℕ :=
  (tea_cost_yen + sandwich_cost_yen) / conversion_rate

theorem alice_total_cost_usd_is_correct :
  total_cost_usd tea_cost_yen sandwich_cost_yen conversion_rate = 6 := 
by
  sorry

end alice_total_cost_usd_is_correct_l1155_115544


namespace rate_of_mixed_oil_l1155_115579

/--
If 10 litres of an oil at Rs. 50 per litre is mixed with 5 litres of another oil at Rs. 68 per litre, 
8 litres of a third oil at Rs. 42 per litre, and 7 litres of a fourth oil at Rs. 62 per litre, 
then the rate of the mixed oil per litre is Rs. 53.67.
-/
theorem rate_of_mixed_oil :
  let cost1 := 10 * 50
  let cost2 := 5 * 68
  let cost3 := 8 * 42
  let cost4 := 7 * 62
  let total_cost := cost1 + cost2 + cost3 + cost4
  let total_volume := 10 + 5 + 8 + 7
  let rate_per_litre := total_cost / total_volume
  rate_per_litre = 53.67 :=
by
  intros
  sorry

end rate_of_mixed_oil_l1155_115579


namespace value_of_f_2011_l1155_115538

noncomputable def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 7

theorem value_of_f_2011 (a b c : ℝ) (h : f a b c (-2011) = -17) : f a b c 2011 = 31 :=
by {
  sorry
}

end value_of_f_2011_l1155_115538


namespace silvia_percentage_shorter_l1155_115536

theorem silvia_percentage_shorter :
  let j := (2 : ℝ) + 1
  let s := Real.sqrt ((2 : ℝ) ^ 2 + (1 : ℝ) ^ 2)
  (abs (( (j - s) / j) * 100 - 25) < 1) :=
by
  let j := (2 : ℝ) + 1
  let s := Real.sqrt ((2 : ℝ) ^ 2 + (1 : ℝ) ^ 2)
  show (abs (( (j - s) / j) * 100 - 25) < 1)
  sorry

end silvia_percentage_shorter_l1155_115536
