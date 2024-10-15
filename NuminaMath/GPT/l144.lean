import Mathlib

namespace NUMINAMATH_GPT_geom_seq_min_value_l144_14472

theorem geom_seq_min_value (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, 0 < a n)
  (h_geom : ∀ n, a n = a 1 * q ^ (n - 1))
  (h_condition : a 7 = a 6 + 2 * a 5)
  (h_mult : ∃ m n, m ≠ n ∧ a m * a n = 16 * (a 1) ^ 2) :
  ∃ (m n : ℕ), m ≠ n ∧ m + n = 6 ∧ (1 / m : ℝ) + (4 / n : ℝ) = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_min_value_l144_14472


namespace NUMINAMATH_GPT_exist_indices_inequalities_l144_14494

open Nat

theorem exist_indices_inequalities (a b c : ℕ → ℕ) :
  ∃ p q : ℕ, a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
by
  -- The proof is to be written here
  sorry

end NUMINAMATH_GPT_exist_indices_inequalities_l144_14494


namespace NUMINAMATH_GPT_polygon_interior_angle_l144_14482

theorem polygon_interior_angle (n : ℕ) (h : n ≥ 3) 
  (interior_angle : ∀ i, 1 ≤ i ∧ i ≤ n → interior_angle = 120) :
  n = 6 := by sorry

end NUMINAMATH_GPT_polygon_interior_angle_l144_14482


namespace NUMINAMATH_GPT_compound_interest_rate_l144_14447

open Real

theorem compound_interest_rate
  (P : ℝ) (A : ℝ) (t : ℝ) (r : ℝ)
  (h_inv : P = 8000)
  (h_time : t = 2)
  (h_maturity : A = 8820) :
  r = 0.05 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_rate_l144_14447


namespace NUMINAMATH_GPT_percentage_of_cars_in_accident_l144_14417

-- Define probabilities of each segment of the rally
def prob_fall_bridge := 1 / 5
def prob_off_turn := 3 / 10
def prob_crash_tunnel := 1 / 10
def prob_stuck_sand := 2 / 5

-- Define complement probabilities (successful completion)
def prob_success_bridge := 1 - prob_fall_bridge
def prob_success_turn := 1 - prob_off_turn
def prob_success_tunnel := 1 - prob_crash_tunnel
def prob_success_sand := 1 - prob_stuck_sand

-- Define overall success probability
def prob_success_total := prob_success_bridge * prob_success_turn * prob_success_tunnel * prob_success_sand

-- Define percentage function
def percentage (p: ℚ) : ℚ := p * 100

-- Prove the percentage of cars involved in accidents
theorem percentage_of_cars_in_accident : percentage (1 - prob_success_total) = 70 := by sorry

end NUMINAMATH_GPT_percentage_of_cars_in_accident_l144_14417


namespace NUMINAMATH_GPT_no_ordered_triples_l144_14492

noncomputable def no_solution (x y z : ℝ) : Prop :=
  x^2 - 3 * x * y + 2 * y^2 - z^2 = 31 ∧
  -x^2 + 6 * y * z + 2 * z^2 = 44 ∧
  x^2 + x * y + 8 * z^2 = 100

theorem no_ordered_triples : ¬ ∃ (x y z : ℝ), no_solution x y z := 
by 
  sorry

end NUMINAMATH_GPT_no_ordered_triples_l144_14492


namespace NUMINAMATH_GPT_distinct_integers_sum_to_32_l144_14498

theorem distinct_integers_sum_to_32 
  (p q r s t : ℤ)
  (h_diff : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t)
  (h_eq : (9 - p) * (9 - q) * (9 - r) * (9 - s) * (9 - t) = -120) : 
  p + q + r + s + t = 32 := 
by 
  sorry

end NUMINAMATH_GPT_distinct_integers_sum_to_32_l144_14498


namespace NUMINAMATH_GPT_solve_for_b_l144_14428

variable (a b c d m : ℝ)

theorem solve_for_b (h : m = cadb / (a - b)) : b = ma / (cad + m) :=
sorry

end NUMINAMATH_GPT_solve_for_b_l144_14428


namespace NUMINAMATH_GPT_minutes_per_mile_l144_14404

-- Define the total distance Peter needs to walk
def total_distance : ℝ := 2.5

-- Define the distance Peter has already walked
def walked_distance : ℝ := 1.0

-- Define the remaining time Peter needs to walk to reach the grocery store
def remaining_time : ℝ := 30.0

-- Define the remaining distance Peter needs to walk
def remaining_distance : ℝ := total_distance - walked_distance

-- The desired statement to prove: it takes Peter 20 minutes to walk one mile
theorem minutes_per_mile : remaining_distance / remaining_time = 1.0 / 20.0 := by
  sorry

end NUMINAMATH_GPT_minutes_per_mile_l144_14404


namespace NUMINAMATH_GPT_length_of_EC_l144_14411

theorem length_of_EC
  (AB CD AC : ℝ)
  (h1 : AB = 3 * CD)
  (h2 : AC = 15)
  (EC : ℝ)
  (h3 : AC = 4 * EC)
  : EC = 15 / 4 := 
sorry

end NUMINAMATH_GPT_length_of_EC_l144_14411


namespace NUMINAMATH_GPT_smallest_multiple_l144_14467

theorem smallest_multiple (b : ℕ) (h1 : b % 6 = 0) (h2 : b % 15 = 0) (h3 : ∀ n : ℕ, (n % 6 = 0 ∧ n % 15 = 0) → n ≥ b) : b = 30 :=
sorry

end NUMINAMATH_GPT_smallest_multiple_l144_14467


namespace NUMINAMATH_GPT_canadian_math_olympiad_1992_l144_14426

theorem canadian_math_olympiad_1992
    (n : ℤ) (a : ℕ → ℤ) (k : ℕ)
    (h1 : n ≥ a 1) 
    (h2 : ∀ i, 1 ≤ i → i ≤ k → a i > 0)
    (h3 : ∀ i j, 1 ≤ i → i ≤ k → 1 ≤ j → j ≤ k → n ≥ Int.lcm (a i) (a j))
    (h4 : ∀ i, 1 ≤ i → i < k → a i > a (i + 1)) :
  ∀ i, 1 ≤ i → i ≤ k → i * a i ≤ n :=
sorry

end NUMINAMATH_GPT_canadian_math_olympiad_1992_l144_14426


namespace NUMINAMATH_GPT_second_reduction_percentage_is_4_l144_14424

def original_price := 500
def first_reduction_percent := 5 / 100
def total_reduction := 44

def first_reduction := first_reduction_percent * original_price
def price_after_first_reduction := original_price - first_reduction
def second_reduction := total_reduction - first_reduction
def second_reduction_percent := (second_reduction / price_after_first_reduction) * 100

theorem second_reduction_percentage_is_4 :
  second_reduction_percent = 4 := by
  sorry

end NUMINAMATH_GPT_second_reduction_percentage_is_4_l144_14424


namespace NUMINAMATH_GPT_sqrt_a_minus_2_meaningful_l144_14468

theorem sqrt_a_minus_2_meaningful (a : ℝ) (h : 0 ≤ a - 2) : 2 ≤ a :=
by
  sorry

end NUMINAMATH_GPT_sqrt_a_minus_2_meaningful_l144_14468


namespace NUMINAMATH_GPT_mary_mileage_l144_14432

def base9_to_base10 : Nat :=
  let d0 := 6 * 9^0
  let d1 := 5 * 9^1
  let d2 := 9 * 9^2
  let d3 := 3 * 9^3
  d0 + d1 + d2 + d3 

theorem mary_mileage :
  base9_to_base10 = 2967 :=
by 
  -- Calculation steps are skipped using sorry
  sorry

end NUMINAMATH_GPT_mary_mileage_l144_14432


namespace NUMINAMATH_GPT_inequality_x_y_l144_14414

theorem inequality_x_y 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) : 
  (x / (x + 5 * y)) + (y / (y + 5 * x)) ≤ 1 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_x_y_l144_14414


namespace NUMINAMATH_GPT_find_R_when_S_eq_5_l144_14400

theorem find_R_when_S_eq_5
  (g : ℚ)
  (h1 : ∀ S, R = g * S^2 - 6)
  (h2 : R = 15 ∧ S = 3) :
  R = 157 / 3 := by
    sorry

end NUMINAMATH_GPT_find_R_when_S_eq_5_l144_14400


namespace NUMINAMATH_GPT_ai_eq_i_l144_14434

namespace Problem

def gcd (m n : ℕ) : ℕ := Nat.gcd m n

def sequence_satisfies (a : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i ≠ j → gcd (a i) (a j) = gcd i j

theorem ai_eq_i (a : ℕ → ℕ) (h : sequence_satisfies a) : ∀ i : ℕ, a i = i :=
by
  sorry

end Problem

end NUMINAMATH_GPT_ai_eq_i_l144_14434


namespace NUMINAMATH_GPT_unit_prices_min_number_of_A_l144_14485

theorem unit_prices (x y : ℝ)
  (h1 : 3 * x + 4 * y = 580)
  (h2 : 6 * x + 5 * y = 860) :
  x = 60 ∧ y = 100 :=
by
  sorry

theorem min_number_of_A (x y a : ℝ)
  (x_h : x = 60)
  (y_h : y = 100)
  (h1 : 3 * x + 4 * y = 580)
  (h2 : 6 * x + 5 * y = 860)
  (trash_can_condition : a + 200 - a = 200)
  (cost_condition : 60 * a + 100 * (200 - a) ≤ 15000) :
  a ≥ 125 :=
by
  sorry

end NUMINAMATH_GPT_unit_prices_min_number_of_A_l144_14485


namespace NUMINAMATH_GPT_wally_not_all_numbers_l144_14471

def next_wally_number (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    n / 2
  else
    (n + 1001) / 2

def eventually_print(n: ℕ) : Prop :=
  ∃ k: ℕ, (next_wally_number^[k]) 1 = n

theorem wally_not_all_numbers :
  ¬ ∀ n, n ≤ 100 → eventually_print n :=
by
  sorry

end NUMINAMATH_GPT_wally_not_all_numbers_l144_14471


namespace NUMINAMATH_GPT_even_num_students_count_l144_14403

-- Define the number of students in each school
def num_students_A : Nat := 786
def num_students_B : Nat := 777
def num_students_C : Nat := 762
def num_students_D : Nat := 819
def num_students_E : Nat := 493

-- Define a predicate to check if a number is even
def is_even (n : Nat) : Prop := n % 2 = 0

-- The theorem to state the problem
theorem even_num_students_count :
  (is_even num_students_A ∧ is_even num_students_C) ∧ ¬(is_even num_students_B ∧ is_even num_students_D ∧ is_even num_students_E) →
  2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_even_num_students_count_l144_14403


namespace NUMINAMATH_GPT_A_inter_B_eq_C_l144_14413

noncomputable def A : Set ℝ := { x | ∃ α β : ℤ, α ≥ 0 ∧ β ≥ 0 ∧ x = 2^α * 3^β }
def B : Set ℝ := { x | 1 ≤ x ∧ x ≤ 5 }
def C : Set ℝ := {1, 2, 3, 4}

theorem A_inter_B_eq_C : A ∩ B = C :=
by
  sorry

end NUMINAMATH_GPT_A_inter_B_eq_C_l144_14413


namespace NUMINAMATH_GPT_solution_set_of_inequality_l144_14487

open Real Set

noncomputable def f (x : ℝ) : ℝ := exp (-x) - exp x - 5 * x

theorem solution_set_of_inequality :
  { x : ℝ | f (x ^ 2) + f (-x - 6) < 0 } = Iio (-2) ∪ Ioi 3 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l144_14487


namespace NUMINAMATH_GPT_carlo_practice_difference_l144_14493

-- Definitions for given conditions
def monday_practice (T : ℕ) : ℕ := 2 * T
def tuesday_practice (T : ℕ) : ℕ := T
def wednesday_practice (thursday_minutes : ℕ) : ℕ := thursday_minutes + 5
def thursday_practice : ℕ := 50
def friday_practice : ℕ := 60
def total_weekly_practice : ℕ := 300

theorem carlo_practice_difference 
  (T : ℕ) 
  (Monday Tuesday Wednesday Thursday Friday : ℕ) 
  (H1 : Monday = monday_practice T)
  (H2 : Tuesday = tuesday_practice T)
  (H3 : Wednesday = wednesday_practice Thursday)
  (H4 : Thursday = thursday_practice)
  (H5 : Friday = friday_practice)
  (H6 : Monday + Tuesday + Wednesday + Thursday + Friday = total_weekly_practice) :
  (Wednesday - Tuesday = 10) :=
by 
  -- Use the provided conditions and derive the required result.
  sorry

end NUMINAMATH_GPT_carlo_practice_difference_l144_14493


namespace NUMINAMATH_GPT_proposition_A_proposition_B_proposition_C_proposition_D_l144_14430

-- Definitions and conditions for proposition A
def propA_conditions (a b : ℝ) : Prop :=
  a > b ∧ (1 / a) > (1 / b)

def propA (a b : ℝ) : Prop :=
  a * b < 0

-- Definitions and conditions for proposition B
def propB_conditions (a b : ℝ) : Prop :=
  a < b ∧ b < 0

def propB (a b : ℝ) : Prop :=
  a^2 < a * b ∧ a * b < b^2

-- Definitions and conditions for proposition C
def propC_conditions (c a b : ℝ) : Prop :=
  c > a ∧ a > b ∧ b > 0

def propC (c a b : ℝ) : Prop :=
  (a / (c - a)) < (b / (c - b))

-- Definitions and conditions for proposition D
def propD_conditions (a b c : ℝ) : Prop :=
  a > b ∧ b > c ∧ c > 0

def propD (a b c : ℝ) : Prop :=
  (a / b) > ((a + c) / (b + c))

-- The propositions
theorem proposition_A (a b : ℝ) (h : propA_conditions a b) : propA a b := 
sorry

theorem proposition_B (a b : ℝ) (h : propB_conditions a b) : ¬ propB a b :=
sorry

theorem proposition_C (c a b : ℝ) (h : propC_conditions c a b) : ¬ propC c a b :=
sorry

theorem proposition_D (a b c : ℝ) (h : propD_conditions a b c) : propD a b c :=
sorry

end NUMINAMATH_GPT_proposition_A_proposition_B_proposition_C_proposition_D_l144_14430


namespace NUMINAMATH_GPT_cube_volume_surface_area_l144_14476

variable (x : ℝ)

theorem cube_volume_surface_area (h1 : s^3 = 8 * x) (h2 : 6 * s^2 = 2 * x) : x = 1728 :=
by
  sorry

end NUMINAMATH_GPT_cube_volume_surface_area_l144_14476


namespace NUMINAMATH_GPT_total_different_books_l144_14409

def tony_books : ℕ := 23
def dean_books : ℕ := 12
def breanna_books : ℕ := 17
def tony_dean_shared_books : ℕ := 3
def all_three_shared_book : ℕ := 1

theorem total_different_books :
  tony_books + dean_books + breanna_books - tony_dean_shared_books - 2 * all_three_shared_book = 47 := 
by
  sorry 

end NUMINAMATH_GPT_total_different_books_l144_14409


namespace NUMINAMATH_GPT_cross_product_example_l144_14457

def vector_cross (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  (u.2.1 * v.2.2 - u.2.2 * v.2.1, 
   u.2.2 * v.1 - u.1 * v.2.2, 
   u.1 * v.1 - u.2.1 * v.1)
   
theorem cross_product_example : 
  vector_cross (4, 3, -7) (2, 0, 5) = (15, -34, -6) :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_cross_product_example_l144_14457


namespace NUMINAMATH_GPT_callie_caught_frogs_l144_14488

theorem callie_caught_frogs (A Q B C : ℝ) 
  (hA : A = 2)
  (hQ : Q = 2 * A)
  (hB : B = 3 * Q)
  (hC : C = (5 / 8) * B) : 
  C = 7.5 := by
  sorry

end NUMINAMATH_GPT_callie_caught_frogs_l144_14488


namespace NUMINAMATH_GPT_solve_for_x_l144_14455

theorem solve_for_x:
  ∃ x : ℚ, (2 / 3 - 1 / 4 = 1 / x) ∧ (x = 12 / 5) := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l144_14455


namespace NUMINAMATH_GPT_solve_system_l144_14483

noncomputable def solution1 (a b : ℝ) : ℝ × ℝ := 
  ((a + Real.sqrt (a^2 + 4 * b)) / 2, (-a + Real.sqrt (a^2 + 4 * b)) / 2)

noncomputable def solution2 (a b : ℝ) : ℝ × ℝ := 
  ((a - Real.sqrt (a^2 + 4 * b)) / 2, (-a - Real.sqrt (a^2 + 4 * b)) / 2)

theorem solve_system (a b x y : ℝ) : 
  (x - y = a ∧ x * y = b) ↔ ((x, y) = solution1 a b ∨ (x, y) = solution2 a b) := 
by sorry

end NUMINAMATH_GPT_solve_system_l144_14483


namespace NUMINAMATH_GPT_widget_cost_reduction_l144_14408

theorem widget_cost_reduction (W R : ℝ) (h1 : 6 * W = 36) (h2 : 8 * (W - R) = 36) : R = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_widget_cost_reduction_l144_14408


namespace NUMINAMATH_GPT_cost_keyboard_l144_14460

def num_keyboards : ℕ := 15
def num_printers : ℕ := 25
def total_cost : ℝ := 2050
def cost_printer : ℝ := 70
def total_cost_printers : ℝ := num_printers * cost_printer
def total_cost_keyboards : ℝ := total_cost - total_cost_printers

theorem cost_keyboard : total_cost_keyboards / num_keyboards = 20 := by
  sorry

end NUMINAMATH_GPT_cost_keyboard_l144_14460


namespace NUMINAMATH_GPT_determine_angle_C_in_DEF_l144_14431

def Triangle := Type

structure TriangleProps (T : Triangle) :=
  (right_angle : Prop)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (angle_C : ℝ)

axiom triangle_ABC : Triangle
axiom triangle_DEF : Triangle

axiom ABC_props : TriangleProps triangle_ABC
axiom DEF_props : TriangleProps triangle_DEF

noncomputable def similar (T1 T2 : Triangle) : Prop := sorry

theorem determine_angle_C_in_DEF
  (h1 : ABC_props.right_angle = true)
  (h2 : ABC_props.angle_A = 30)
  (h3 : DEF_props.right_angle = true)
  (h4 : DEF_props.angle_B = 60)
  (h5 : similar triangle_ABC triangle_DEF) :
  DEF_props.angle_C = 30 :=
sorry

end NUMINAMATH_GPT_determine_angle_C_in_DEF_l144_14431


namespace NUMINAMATH_GPT_students_multiple_activities_l144_14427

theorem students_multiple_activities (total_students only_debate only_singing only_dance no_activities students_more_than_one : ℕ) 
  (h1 : total_students = 55) 
  (h2 : only_debate = 10) 
  (h3 : only_singing = 18) 
  (h4 : only_dance = 8)
  (h5 : no_activities = 5)
  (h6 : students_more_than_one = total_students - (only_debate + only_singing + only_dance + no_activities)) :
  students_more_than_one = 14 := by
  sorry

end NUMINAMATH_GPT_students_multiple_activities_l144_14427


namespace NUMINAMATH_GPT_ball_bounce_height_l144_14461

theorem ball_bounce_height :
  ∃ k : ℕ, 2000 * (2 / 3 : ℝ) ^ k < 2 ∧ ∀ j : ℕ, j < k → 2000 * (2 / 3 : ℝ) ^ j ≥ 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_ball_bounce_height_l144_14461


namespace NUMINAMATH_GPT_valentines_given_l144_14459

theorem valentines_given (x y : ℕ) (h : x * y = x + y + 40) : x * y = 84 :=
by
  -- solving for x, y based on the factors of 41
  sorry

end NUMINAMATH_GPT_valentines_given_l144_14459


namespace NUMINAMATH_GPT_profit_percentage_l144_14486

theorem profit_percentage (CP SP : ℝ) (hCP : CP = 150) (hSP : SP = 216.67) :
  SP = 0.9 * LP ∧ LP = SP / 0.9 ∧ Profit = SP - CP ∧ Profit_Percentage = (Profit / CP) * 100 ∧ Profit_Percentage = 44.44 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_l144_14486


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_problem_4_l144_14435

theorem problem_1 : (1 * -2.48) + 4.33 + (-7.52) + (-4.33) = -10 := by
  sorry

theorem problem_2 : 2 * (23 / 6 : ℚ) + - (36 / 7 : ℚ) + - (13 / 6 : ℚ) + - (230 / 7 : ℚ) = -(36 + 1 / 3 : ℚ) := by
  sorry

theorem problem_3 : (4 / 5 : ℚ) - (5 / 6 : ℚ) - (3 / 5 : ℚ) + (1 / 6 : ℚ) = - (7 / 15 : ℚ) := by
  sorry

theorem problem_4 : (-1 ^ 4 : ℚ) - (1 / 6) * (2 - (-3) ^ 2) = 1 / 6 := by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_problem_4_l144_14435


namespace NUMINAMATH_GPT_variance_decreases_l144_14444

def scores_initial := [5, 9, 7, 10, 9] -- Initial 5 shot scores
def additional_shot := 8 -- Additional shot score

-- Given variance of initial scores
def variance_initial : ℝ := 3.2

-- Placeholder function to calculate variance of a list of scores
noncomputable def variance (scores : List ℝ) : ℝ := sorry

-- Definition of the new scores list
def scores_new := scores_initial ++ [additional_shot]

-- Define the proof problem
theorem variance_decreases :
  variance scores_new < variance_initial :=
sorry

end NUMINAMATH_GPT_variance_decreases_l144_14444


namespace NUMINAMATH_GPT_defective_items_count_l144_14445

variables 
  (total_items : ℕ)
  (total_video_games : ℕ)
  (total_DVDs : ℕ)
  (total_books : ℕ)
  (working_video_games : ℕ)
  (working_DVDs : ℕ)

theorem defective_items_count
  (h1 : total_items = 56)
  (h2 : total_video_games = 30)
  (h3 : total_DVDs = 15)
  (h4 : total_books = total_items - total_video_games - total_DVDs)
  (h5 : working_video_games = 20)
  (h6 : working_DVDs = 10)
  : (total_video_games - working_video_games) + (total_DVDs - working_DVDs) = 15 :=
sorry

end NUMINAMATH_GPT_defective_items_count_l144_14445


namespace NUMINAMATH_GPT_find_edge_value_l144_14452

theorem find_edge_value (a b c d e_1 e_2 e_3 e_4 : ℕ) 
  (h1 : e_1 = a + b)
  (h2 : e_2 = b + c)
  (h3 : e_3 = c + d)
  (h4 : e_4 = d + a)
  (h5 : e_1 = 8)
  (h6 : e_3 = 13)
  (h7 : e_1 + e_3 = a + b + c + d)
  : e_4 = 12 := 
by sorry

end NUMINAMATH_GPT_find_edge_value_l144_14452


namespace NUMINAMATH_GPT_negation_of_universal_l144_14497

open Classical

theorem negation_of_universal (P : ∀ x : ℤ, x^3 < 1) : ∃ x : ℤ, x^3 ≥ 1 :=
by sorry

end NUMINAMATH_GPT_negation_of_universal_l144_14497


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l144_14465

noncomputable def isArithmeticSeq (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_first_n (a : ℕ → ℤ) (n : ℕ) := (n + 1) * (a 0 + a n) / 2

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (h_legal_seq : isArithmeticSeq a) (h_sum : sum_first_n a 9 = 120) : 
  a 1 + a 8 = 24 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l144_14465


namespace NUMINAMATH_GPT_exists_d_for_m_divides_f_of_f_n_l144_14415

noncomputable def f : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => 23 * f (n + 1) + f n

theorem exists_d_for_m_divides_f_of_f_n (m : ℕ) : 
  ∃ (d : ℕ), ∀ (n : ℕ), m ∣ f (f n) ↔ d ∣ n := 
sorry

end NUMINAMATH_GPT_exists_d_for_m_divides_f_of_f_n_l144_14415


namespace NUMINAMATH_GPT_find_xy_sum_l144_14450

open Nat

theorem find_xy_sum (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x + y + x * y = 8) 
  (h2 : y + z + y * z = 15) 
  (h3 : z + x + z * x = 35) : 
  x + y + z + x * y = 15 := 
sorry

end NUMINAMATH_GPT_find_xy_sum_l144_14450


namespace NUMINAMATH_GPT_polynomial_sum_at_points_l144_14458

def P (x : ℝ) : ℝ := x^5 - 1.7 * x^3 + 2.5

theorem polynomial_sum_at_points :
  P 19.1 + P (-19.1) = 5 := by
  sorry

end NUMINAMATH_GPT_polynomial_sum_at_points_l144_14458


namespace NUMINAMATH_GPT_digit_of_fraction_l144_14479

theorem digit_of_fraction (n : ℕ) : (15 / 37 : ℝ) = 0.405 ∧ 415 % 3 = 1 → ∃ d : ℕ, d = 4 :=
by
  sorry

end NUMINAMATH_GPT_digit_of_fraction_l144_14479


namespace NUMINAMATH_GPT_max_value_and_period_of_g_value_of_expression_if_fx_eq_2f_l144_14402

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f' (x : ℝ) : ℝ := Real.cos x - Real.sin x
noncomputable def g (x : ℝ) : ℝ := f x * f' x - f x ^ 2

theorem max_value_and_period_of_g :
  ∃ (M : ℝ) (T : ℝ), (∀ x, g x ≤ M) ∧ (∀ x, g (x + T) = g x) ∧ M = 2 ∧ T = Real.pi :=
sorry

theorem value_of_expression_if_fx_eq_2f'x (x : ℝ) :
  f x = 2 * f' x → (1 + Real.sin x ^ 2) / (Real.cos x ^ 2 - Real.sin x * Real.cos x) = 11 / 6 :=
sorry

end NUMINAMATH_GPT_max_value_and_period_of_g_value_of_expression_if_fx_eq_2f_l144_14402


namespace NUMINAMATH_GPT_sum_of_three_numbers_l144_14448

-- Definitions for the conditions
def mean_condition_1 (x y z : ℤ) := (x + y + z) / 3 = x + 20
def mean_condition_2 (x y z : ℤ) := (x + y + z) / 3 = z - 18
def median_condition (y : ℤ) := y = 9

-- The Lean 4 statement to prove the sum of x, y, and z is 21
theorem sum_of_three_numbers (x y z : ℤ) 
  (h1 : mean_condition_1 x y z) 
  (h2 : mean_condition_2 x y z) 
  (h3 : median_condition y) : 
  x + y + z = 21 := 
  by 
    sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l144_14448


namespace NUMINAMATH_GPT_units_digit_of_power_l144_14446

theorem units_digit_of_power (base : ℕ) (exp : ℕ) (units_base : ℕ) (units_exp_mod : ℕ) :
  (base % 10 = units_base) → (exp % 2 = units_exp_mod) → (units_base = 9) → (units_exp_mod = 0) →
  (base ^ exp % 10 = 1) :=
by
  intros
  sorry

end NUMINAMATH_GPT_units_digit_of_power_l144_14446


namespace NUMINAMATH_GPT_circle_bisection_relation_l144_14410

theorem circle_bisection_relation (a b : ℝ) :
  (∀ x y : ℝ, (x - a)^2 + (y - b)^2 = b^2 + 1 → (x + 1)^2 + (y + 1)^2 = 4) ↔ 
  a^2 + 2 * a + 2 * b + 5 = 0 :=
by sorry

end NUMINAMATH_GPT_circle_bisection_relation_l144_14410


namespace NUMINAMATH_GPT_time_to_fill_cistern_proof_l144_14475

-- Define the filling rate F and emptying rate E
def filling_rate : ℚ := 1 / 3 -- cisterns per hour
def emptying_rate : ℚ := 1 / 6 -- cisterns per hour

-- Define the net rate as the difference between filling and emptying rates
def net_rate : ℚ := filling_rate - emptying_rate

-- Define the time to fill the cistern given the net rate
def time_to_fill_cistern (net_rate : ℚ) : ℚ := 1 / net_rate

-- The proof statement
theorem time_to_fill_cistern_proof : time_to_fill_cistern net_rate = 6 := 
by sorry

end NUMINAMATH_GPT_time_to_fill_cistern_proof_l144_14475


namespace NUMINAMATH_GPT_last_digit_of_S_l144_14480

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_of_S : last_digit (54 ^ 2020 + 28 ^ 2022) = 0 :=
by 
  -- The Lean proof steps would go here
  sorry

end NUMINAMATH_GPT_last_digit_of_S_l144_14480


namespace NUMINAMATH_GPT_machine_value_after_two_years_l144_14421

noncomputable def machine_market_value (initial_value : ℝ) (years : ℕ) (decrease_rate : ℝ) : ℝ :=
  initial_value * (1 - decrease_rate) ^ years

theorem machine_value_after_two_years :
  machine_market_value 8000 2 0.2 = 5120 := by
  sorry

end NUMINAMATH_GPT_machine_value_after_two_years_l144_14421


namespace NUMINAMATH_GPT_largest_d_l144_14453

variable (a b c d : ℝ)

theorem largest_d (h : a + 1 = b - 2 ∧ b - 2 = c + 3 ∧ c + 3 = d - 4) : 
  d >= a ∧ d >= b ∧ d >= c :=
by
  sorry

end NUMINAMATH_GPT_largest_d_l144_14453


namespace NUMINAMATH_GPT_rectangle_area_l144_14456

variable {x : ℝ} (h : x > 0)

theorem rectangle_area (W : ℝ) (L : ℝ) (hL : L = 3 * W) (h_diag : W^2 + L^2 = x^2) :
  (W * L) = (3 / 10) * x^2 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l144_14456


namespace NUMINAMATH_GPT_missing_digit_B_l144_14451

theorem missing_digit_B (B : ℕ) (h : 0 ≤ B ∧ B ≤ 9) (h_div : (100 + 10 * B + 3) % 13 = 0) : B = 4 := 
by
  sorry

end NUMINAMATH_GPT_missing_digit_B_l144_14451


namespace NUMINAMATH_GPT_eq_implies_sq_eq_l144_14412

theorem eq_implies_sq_eq (a b : ℝ) (h : a = b) : a^2 = b^2 :=
sorry

end NUMINAMATH_GPT_eq_implies_sq_eq_l144_14412


namespace NUMINAMATH_GPT_trigonometric_identity_l144_14464

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 1 / 2) :
  Real.sin (2 * θ) - 2 * Real.cos θ ^ 2 = -4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l144_14464


namespace NUMINAMATH_GPT_number_of_workers_l144_14441

theorem number_of_workers (N C : ℕ) 
  (h1 : N * C = 300000) 
  (h2 : N * (C + 50) = 325000) : 
  N = 500 :=
sorry

end NUMINAMATH_GPT_number_of_workers_l144_14441


namespace NUMINAMATH_GPT_tom_age_ratio_l144_14463

-- Define the conditions
variable (T N : ℕ) (ages_of_children_sum : ℕ)

-- Given conditions as definitions
def condition1 : Prop := T = ages_of_children_sum
def condition2 : Prop := (T - N) = 3 * (T - 4 * N)

-- The theorem statement to be proven
theorem tom_age_ratio : condition1 T ages_of_children_sum ∧ condition2 T N → T / N = 11 / 2 :=
by sorry

end NUMINAMATH_GPT_tom_age_ratio_l144_14463


namespace NUMINAMATH_GPT_students_in_favor_ABC_l144_14466

variables (U A B C : Finset ℕ)

-- Given conditions
axiom total_students : U.card = 300
axiom students_in_favor_A : A.card = 210
axiom students_in_favor_B : B.card = 190
axiom students_in_favor_C : C.card = 160
axiom students_against_all : (U \ (A ∪ B ∪ C)).card = 40

-- Proof goal
theorem students_in_favor_ABC : (A ∩ B ∩ C).card = 80 :=
by {
  sorry
}

end NUMINAMATH_GPT_students_in_favor_ABC_l144_14466


namespace NUMINAMATH_GPT_incorrect_inequality_exists_l144_14469

theorem incorrect_inequality_exists :
  ∃ (x y : ℝ), x < y ∧ x^2 ≥ y^2 :=
by {
  sorry
}

end NUMINAMATH_GPT_incorrect_inequality_exists_l144_14469


namespace NUMINAMATH_GPT_girls_collected_more_mushrooms_l144_14473

variables (N I A V : ℝ)

theorem girls_collected_more_mushrooms 
    (h1 : N > I) 
    (h2 : N > A) 
    (h3 : N > V) 
    (h4 : I ≤ N) 
    (h5 : I ≤ A) 
    (h6 : I ≤ V) 
    (h7 : A > V) : 
    N + I > A + V := 
by {
    sorry
}

end NUMINAMATH_GPT_girls_collected_more_mushrooms_l144_14473


namespace NUMINAMATH_GPT_solve_abs_ineq_l144_14481

theorem solve_abs_ineq (x : ℝ) (h : x > 0) : |4 * x - 5| < 8 ↔ 0 < x ∧ x < 13 / 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_abs_ineq_l144_14481


namespace NUMINAMATH_GPT_exists_rational_non_integer_linear_l144_14484

theorem exists_rational_non_integer_linear (k1 k2 : ℤ) : 
  ∃ (x y : ℚ), x ≠ ⌊x⌋ ∧ y ≠ ⌊y⌋ ∧ 
  19 * x + 8 * y = k1 ∧ 8 * x + 3 * y = k2 :=
sorry

end NUMINAMATH_GPT_exists_rational_non_integer_linear_l144_14484


namespace NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l144_14433

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ) (h_arith_seq : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h1 : a 7 - 2 * a 4 = -1)
  (h2 : a 3 = 0) :
  (a 2 - a 1) = - 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l144_14433


namespace NUMINAMATH_GPT_jellybean_probability_l144_14418

theorem jellybean_probability :
  let total_jellybeans := 15
  let green_jellybeans := 6
  let purple_jellybeans := 2
  let yellow_jellybeans := 7
  let total_picked := 4
  let total_ways := Nat.choose total_jellybeans total_picked
  let ways_to_pick_two_yellow := Nat.choose yellow_jellybeans 2
  let ways_to_pick_two_non_yellow := Nat.choose (total_jellybeans - yellow_jellybeans) 2
  let successful_outcomes := ways_to_pick_two_yellow * ways_to_pick_two_non_yellow
  let probability := successful_outcomes / total_ways
  probability = 4 / 9 := by
sorry

end NUMINAMATH_GPT_jellybean_probability_l144_14418


namespace NUMINAMATH_GPT_douglas_won_in_Y_l144_14420

theorem douglas_won_in_Y (percent_total_vote : ℕ) (percent_vote_X : ℕ) (ratio_XY : ℕ) (P : ℕ) :
  percent_total_vote = 54 →
  percent_vote_X = 62 →
  ratio_XY = 2 →
  P = 38 :=
by
  sorry

end NUMINAMATH_GPT_douglas_won_in_Y_l144_14420


namespace NUMINAMATH_GPT_quadratic_expression_min_value_l144_14429

noncomputable def min_value_quadratic_expression (x y z : ℝ) : ℝ :=
(x + 5) ^ 2 + (y - 1) ^ 2 + (z + 3) ^ 2

theorem quadratic_expression_min_value :
  ∃ x y z : ℝ, x - 2 * y + 2 * z = 5 ∧ min_value_quadratic_expression x y z = 36 :=
sorry

end NUMINAMATH_GPT_quadratic_expression_min_value_l144_14429


namespace NUMINAMATH_GPT_cos_sin_ratio_l144_14474

open Real

-- Given conditions
variables {α β : Real}
axiom tan_alpha_beta : tan (α + β) = 2 / 5
axiom tan_beta_pi_over_4 : tan (β - π / 4) = 1 / 4

-- Theorem to be proven
theorem cos_sin_ratio (hαβ : tan (α + β) = 2 / 5) (hβ : tan (β - π / 4) = 1 / 4) :
  (cos α + sin α) / (cos α - sin α) = 3 / 22 :=
sorry

end NUMINAMATH_GPT_cos_sin_ratio_l144_14474


namespace NUMINAMATH_GPT_probability_of_individual_selection_l144_14423

theorem probability_of_individual_selection (sample_size : ℕ) (population_size : ℕ)
  (h_sample : sample_size = 10) (h_population : population_size = 42) :
  (sample_size : ℚ) / (population_size : ℚ) = 5 / 21 := 
by {
  sorry
}

end NUMINAMATH_GPT_probability_of_individual_selection_l144_14423


namespace NUMINAMATH_GPT_grade12_students_selected_l144_14477

theorem grade12_students_selected 
    (N : ℕ) (n10 : ℕ) (n12 : ℕ) (k : ℕ) 
    (h1 : N = 1200)
    (h2 : n10 = 240)
    (h3 : 3 * N / (k + 5 + 3) = n12)
    (h4 : k * N / (k + 5 + 3) = n10) :
    n12 = 360 := 
by sorry

end NUMINAMATH_GPT_grade12_students_selected_l144_14477


namespace NUMINAMATH_GPT_initial_violet_balloons_l144_14449

-- Defining the conditions
def violet_balloons_given_by_tom : ℕ := 16
def violet_balloons_left_with_tom : ℕ := 14

-- The statement to prove
theorem initial_violet_balloons (initial_balloons : ℕ) :
  initial_balloons = violet_balloons_given_by_tom + violet_balloons_left_with_tom :=
sorry

end NUMINAMATH_GPT_initial_violet_balloons_l144_14449


namespace NUMINAMATH_GPT_amount_of_money_C_l144_14437

theorem amount_of_money_C (a b c d : ℤ) 
  (h1 : a + b + c + d = 600)
  (h2 : a + c = 200)
  (h3 : b + c = 350)
  (h4 : a + d = 300)
  (h5 : a ≥ 2 * b) : c = 150 := 
by
  sorry

end NUMINAMATH_GPT_amount_of_money_C_l144_14437


namespace NUMINAMATH_GPT_height_difference_l144_14491

def empireStateBuildingHeight : ℕ := 443
def petronasTowersHeight : ℕ := 452

theorem height_difference :
  petronasTowersHeight - empireStateBuildingHeight = 9 := 
sorry

end NUMINAMATH_GPT_height_difference_l144_14491


namespace NUMINAMATH_GPT_final_grade_calculation_l144_14490

theorem final_grade_calculation
  (exam_score homework_score class_participation_score : ℝ)
  (exam_weight homework_weight participation_weight : ℝ)
  (h_exam_score : exam_score = 90)
  (h_homework_score : homework_score = 85)
  (h_class_participation_score : class_participation_score = 80)
  (h_exam_weight : exam_weight = 3)
  (h_homework_weight : homework_weight = 2)
  (h_participation_weight : participation_weight = 5) :
  (exam_score * exam_weight + homework_score * homework_weight + class_participation_score * participation_weight) /
  (exam_weight + homework_weight + participation_weight) = 84 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_final_grade_calculation_l144_14490


namespace NUMINAMATH_GPT_additional_savings_l144_14416

-- Defining the conditions
def initial_price : ℝ := 50
def discount_one : ℝ := 6
def discount_percentage : ℝ := 0.15

-- Defining the final prices according to the two methods
def first_method : ℝ := (1 - discount_percentage) * (initial_price - discount_one)
def second_method : ℝ := (1 - discount_percentage) * initial_price - discount_one

-- Defining the savings for the two methods
def savings_first_method : ℝ := initial_price - first_method
def savings_second_method : ℝ := initial_price - second_method

-- Proving that the second method results in an additional 0.90 savings
theorem additional_savings : (savings_second_method - savings_first_method) = 0.90 :=
by
  sorry

end NUMINAMATH_GPT_additional_savings_l144_14416


namespace NUMINAMATH_GPT_g_2187_value_l144_14407

-- Define the function properties and the goal
theorem g_2187_value (g : ℕ → ℝ) (h : ∀ x y m : ℕ, x + y = 3^m → g x + g y = m^3) :
  g 2187 = 343 :=
sorry

end NUMINAMATH_GPT_g_2187_value_l144_14407


namespace NUMINAMATH_GPT_number_of_teams_l144_14419

-- Total number of players
def total_players : Nat := 12

-- Number of ways to choose one captain
def ways_to_choose_captain : Nat := total_players

-- Number of remaining players after choosing the captain
def remaining_players : Nat := total_players - 1

-- Number of players needed to form a team (excluding the captain)
def team_size : Nat := 5

-- Number of ways to choose 5 players from the remaining 11
def ways_to_choose_team (n k : Nat) : Nat := Nat.choose n k

-- Total number of different teams
def total_teams : Nat := ways_to_choose_captain * ways_to_choose_team remaining_players team_size

theorem number_of_teams : total_teams = 5544 := by
  sorry

end NUMINAMATH_GPT_number_of_teams_l144_14419


namespace NUMINAMATH_GPT_k_equals_10_l144_14401

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (a d : α) : ℕ → α
  | 0     => a
  | (n+1) => a + (n+1) * d

noncomputable def sum_of_first_n_terms (a d : α) (n : ℕ) : α :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem k_equals_10
  (a d : α)
  (h1 : sum_of_first_n_terms a d 9 = sum_of_first_n_terms a d 4)
  (h2 : arithmetic_sequence a d 4 + arithmetic_sequence a d 10 = 0) :
  k = 10 :=
sorry

end NUMINAMATH_GPT_k_equals_10_l144_14401


namespace NUMINAMATH_GPT_S_eq_T_l144_14439

-- Define the sets S and T
def S : Set ℤ := {x | ∃ n : ℕ, x = 3 * n + 1}
def T : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 2}

-- Prove that S = T
theorem S_eq_T : S = T := 
by {
  sorry
}

end NUMINAMATH_GPT_S_eq_T_l144_14439


namespace NUMINAMATH_GPT_max_area_cross_section_of_prism_l144_14462

noncomputable def prism_vertex_A : ℝ × ℝ × ℝ := (3, 0, 0)
noncomputable def prism_vertex_B : ℝ × ℝ × ℝ := (-3, 0, 0)
noncomputable def prism_vertex_C : ℝ × ℝ × ℝ := (0, 3 * Real.sqrt 3, 0)
noncomputable def plane_eq (x y z : ℝ) : ℝ := 2 * x - 3 * y + 6 * z

-- Statement
theorem max_area_cross_section_of_prism (h : ℝ) (A B C : ℝ × ℝ × ℝ)
  (plane : ℝ → ℝ → ℝ → ℝ) (cond_h : h = 5)
  (cond_A : A = prism_vertex_A) (cond_B : B = prism_vertex_B) 
  (cond_C : C = prism_vertex_C) (cond_plane : ∀ x y z, plane x y z = 2 * x - 3 * y + 6 * z - 30) : 
  ∃ cross_section : ℝ, cross_section = 0 :=
by
  sorry

end NUMINAMATH_GPT_max_area_cross_section_of_prism_l144_14462


namespace NUMINAMATH_GPT_ratio_copper_to_zinc_l144_14425

theorem ratio_copper_to_zinc (copper zinc : ℝ) (hc : copper = 24) (hz : zinc = 10.67) : (copper / zinc) = 2.25 :=
by
  rw [hc, hz]
  -- Add the arithmetic operation
  sorry

end NUMINAMATH_GPT_ratio_copper_to_zinc_l144_14425


namespace NUMINAMATH_GPT_find_radius_of_circle_l144_14443

noncomputable def central_angle := 150
noncomputable def arc_length := 5 * Real.pi
noncomputable def arc_length_formula (θ : ℝ) (r : ℝ) : ℝ :=
  (θ / 180) * Real.pi * r

theorem find_radius_of_circle :
  (∃ r : ℝ, arc_length_formula central_angle r = arc_length) ↔ 6 = 6 :=
by  
  sorry

end NUMINAMATH_GPT_find_radius_of_circle_l144_14443


namespace NUMINAMATH_GPT_sqrt_diff_ineq_sum_sq_gt_sum_prod_l144_14454

-- First proof problem: Prove that sqrt(11) - 2 * sqrt(3) > 3 - sqrt(10)
theorem sqrt_diff_ineq : (Real.sqrt 11 - 2 * Real.sqrt 3) > (3 - Real.sqrt 10) := sorry

-- Second proof problem: Prove that a^2 + b^2 + c^2 > ab + bc + ca given a, b, and c are real numbers that are not all equal
theorem sum_sq_gt_sum_prod (a b c : ℝ) (h : ¬ (a = b ∧ b = c ∧ a = c)) : a^2 + b^2 + c^2 > a * b + b * c + c * a := sorry

end NUMINAMATH_GPT_sqrt_diff_ineq_sum_sq_gt_sum_prod_l144_14454


namespace NUMINAMATH_GPT_problem1_problem2_l144_14442

def a (n : ℕ) : ℕ :=
  if n = 0 then 0  -- We add this case for Lean to handle zero index
  else if n = 1 then 2
  else 2^(n-1)

def S (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n + 1)) a

theorem problem1 (n : ℕ) :
  a n = 
  if n = 1 then 2
  else 2^(n-1) :=
sorry

theorem problem2 (n : ℕ) :
  S n = 2^n :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l144_14442


namespace NUMINAMATH_GPT_Peter_work_rate_l144_14489

theorem Peter_work_rate:
  ∀ (m p j : ℝ),
    (m + p + j) * 20 = 1 →
    (m + p + j) * 10 = 0.5 →
    (p + j) * 10 = 0.5 →
    j * 15 = 0.5 →
    p * 60 = 1 :=
by
  intros m p j h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_Peter_work_rate_l144_14489


namespace NUMINAMATH_GPT_find_Gary_gold_l144_14422

variable (G : ℕ) -- G represents the number of grams of gold Gary has.
variable (cost_Gary_gold_per_gram : ℕ) -- The cost per gram of Gary's gold.
variable (grams_Anna_gold : ℕ) -- The number of grams of gold Anna has.
variable (cost_Anna_gold_per_gram : ℕ) -- The cost per gram of Anna's gold.
variable (combined_cost : ℕ) -- The combined cost of both Gary's and Anna's gold.

theorem find_Gary_gold (h1 : cost_Gary_gold_per_gram = 15)
                       (h2 : grams_Anna_gold = 50)
                       (h3 : cost_Anna_gold_per_gram = 20)
                       (h4 : combined_cost = 1450)
                       (h5 : combined_cost = cost_Gary_gold_per_gram * G + grams_Anna_gold * cost_Anna_gold_per_gram) :
  G = 30 :=
by 
  sorry

end NUMINAMATH_GPT_find_Gary_gold_l144_14422


namespace NUMINAMATH_GPT_part_I_part_II_l144_14478

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x + a * x^2 - (2 * a + 1) * x

theorem part_I (a : ℝ) (ha : a = -2) : 
  (∃ x : ℝ, f a x = 1) ∧ ∀ x : ℝ, f a x ≤ 1 :=
by sorry

theorem part_II (a : ℝ) (ha : a < 1/2) :
  (∃ x : ℝ, 0 < x ∧ x < exp 1 ∧ f a x < 0) → a < (exp 1 - 1) / (exp 1 * (exp 1 - 2)) :=
by sorry

end NUMINAMATH_GPT_part_I_part_II_l144_14478


namespace NUMINAMATH_GPT_cube_surface_area_150_of_volume_125_l144_14495

def volume (s : ℝ) : ℝ := s^3

def surface_area (s : ℝ) : ℝ := 6 * s^2

theorem cube_surface_area_150_of_volume_125 :
  ∀ (s : ℝ), volume s = 125 → surface_area s = 150 :=
by 
  intros s hs
  sorry

end NUMINAMATH_GPT_cube_surface_area_150_of_volume_125_l144_14495


namespace NUMINAMATH_GPT_only_one_of_A_B_qualifies_at_least_one_qualifies_l144_14440

-- Define the probabilities
def P_A_written : ℚ := 2/3
def P_B_written : ℚ := 1/2
def P_C_written : ℚ := 3/4

def P_A_interview : ℚ := 1/2
def P_B_interview : ℚ := 2/3
def P_C_interview : ℚ := 1/3

-- Calculate the overall probabilities for each student qualifying
def P_A_qualifies : ℚ := P_A_written * P_A_interview
def P_B_qualifies : ℚ := P_B_written * P_B_interview
def P_C_qualifies : ℚ := P_C_written * P_C_interview

-- Part 1: Probability that only one of A or B qualifies
theorem only_one_of_A_B_qualifies :
  P_A_qualifies * (1 - P_B_qualifies) + (1 - P_A_qualifies) * P_B_qualifies = 4/9 :=
by sorry

-- Part 2: Probability that at least one of A, B, or C qualifies
theorem at_least_one_qualifies :
  1 - (1 - P_A_qualifies) * (1 - P_B_qualifies) * (1 - P_C_qualifies) = 2/3 :=
by sorry

end NUMINAMATH_GPT_only_one_of_A_B_qualifies_at_least_one_qualifies_l144_14440


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l144_14470

open Real

noncomputable def a_n : ℕ → ℝ := sorry -- to represent the arithmetic sequence

theorem arithmetic_sequence_sum :
  (∃ d : ℝ, ∀ (n : ℕ), a_n n = a_n 1 + (n - 1) * d) ∧
  (∃ a1 a2011 : ℝ, (a_n 1 = a1) ∧ (a_n 2011 = a2011) ∧ (a1 ^ 2 - 10 * a1 + 16 = 0) ∧ (a2011 ^ 2 - 10 * a2011 + 16 = 0)) →
  a_n 2 + a_n 1006 + a_n 2010 = 15 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l144_14470


namespace NUMINAMATH_GPT_dan_money_left_l144_14496

def money_left (initial_amount spent_on_candy spent_on_gum : ℝ) : ℝ :=
  initial_amount - (spent_on_candy + spent_on_gum)

theorem dan_money_left :
  money_left 3.75 1.25 0.80 = 1.70 :=
by
  sorry

end NUMINAMATH_GPT_dan_money_left_l144_14496


namespace NUMINAMATH_GPT_find_purple_balls_count_l144_14499

theorem find_purple_balls_count (k : ℕ) (h : ∃ k > 0, (21 - 3 * k) = (3 / 4) * (7 + k)) : k = 4 :=
sorry

end NUMINAMATH_GPT_find_purple_balls_count_l144_14499


namespace NUMINAMATH_GPT_range_u_inequality_le_range_k_squared_l144_14436

def D (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 = k}

theorem range_u (k : ℝ) (hk : k > 0) :
  ∀ (x1 x2 : ℝ), (x1, x2) ∈ D k → 0 < x1 * x2 ∧ x1 * x2 ≤ k^2 / 4 :=
sorry

theorem inequality_le (k : ℝ) (hk : k ≥ 1) :
  ∀ (x1 x2 : ℝ), (x1, x2) ∈ D k →
  (1 / x1 - x1) * (1 / x2 - x2) ≤ (k / 2 - 2 / k)^2 :=
sorry

theorem range_k_squared (k : ℝ) :
  (0 < k^2 ∧ k^2 ≤ 4 * Real.sqrt 5 - 8) ↔
  ∀ (x1 x2 : ℝ), (x1, x2) ∈ D k →
  (1 / x1 - x1) * (1 / x2 - x2) ≥ (k / 2 - 2 / k)^2 :=
sorry

end NUMINAMATH_GPT_range_u_inequality_le_range_k_squared_l144_14436


namespace NUMINAMATH_GPT_find_number_of_pens_l144_14438

-- Definitions based on the conditions in the problem
def total_utensils (P L : ℕ) : Prop := P + L = 108
def pencils_formula (P L : ℕ) : Prop := L = 5 * P + 12

-- The theorem we need to prove
theorem find_number_of_pens (P L : ℕ) (h1 : total_utensils P L) (h2 : pencils_formula P L) : P = 16 :=
by sorry

end NUMINAMATH_GPT_find_number_of_pens_l144_14438


namespace NUMINAMATH_GPT_julia_cookies_l144_14406

theorem julia_cookies (N : ℕ) 
  (h1 : N % 6 = 5) 
  (h2 : N % 8 = 7) 
  (h3 : N < 100) : 
  N = 17 ∨ N = 41 ∨ N = 65 ∨ N = 89 → 17 + 41 + 65 + 89 = 212 :=
sorry

end NUMINAMATH_GPT_julia_cookies_l144_14406


namespace NUMINAMATH_GPT_gary_money_left_l144_14405

variable (initialAmount : Nat)
variable (amountSpent : Nat)

theorem gary_money_left (h1 : initialAmount = 73) (h2 : amountSpent = 55) : initialAmount - amountSpent = 18 :=
by
  sorry

end NUMINAMATH_GPT_gary_money_left_l144_14405
