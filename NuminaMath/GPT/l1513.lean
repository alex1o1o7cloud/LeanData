import Mathlib

namespace NUMINAMATH_GPT_find_a5_geometric_sequence_l1513_151397

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∃ r > 0, ∀ n ≥ 1, a (n + 1) = r * a n

theorem find_a5_geometric_sequence :
  ∀ (a : ℕ → ℝ),
  geometric_sequence a ∧ 
  (∀ n, a n > 0) ∧ 
  (a 3 * a 11 = 16) 
  → a 5 = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a5_geometric_sequence_l1513_151397


namespace NUMINAMATH_GPT_evaluate_expression_l1513_151393

-- Define the condition b = 2
def b : ℕ := 2

-- Theorem statement
theorem evaluate_expression : (b^3 * b^4 = 128) := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1513_151393


namespace NUMINAMATH_GPT_no_third_quadrant_l1513_151337

def quadratic_no_real_roots (b : ℝ) : Prop :=
  16 - 4 * b < 0

def passes_through_third_quadrant (b : ℝ) : Prop :=
  ∃ x y : ℝ, y = -2 * x + b ∧ x < 0 ∧ y < 0

theorem no_third_quadrant (b : ℝ) (h : quadratic_no_real_roots b) : ¬ passes_through_third_quadrant b := 
by {
  sorry
}

end NUMINAMATH_GPT_no_third_quadrant_l1513_151337


namespace NUMINAMATH_GPT_last_person_teeth_removed_l1513_151303

-- Define the initial conditions
def total_teeth : ℕ := 32
def total_removed : ℕ := 40
def first_person_removed : ℕ := total_teeth * 1 / 4
def second_person_removed : ℕ := total_teeth * 3 / 8
def third_person_removed : ℕ := total_teeth * 1 / 2

-- Express the problem in Lean
theorem last_person_teeth_removed : 
  first_person_removed + second_person_removed + third_person_removed + last_person_removed = total_removed →
  last_person_removed = 4 := 
by
  sorry

end NUMINAMATH_GPT_last_person_teeth_removed_l1513_151303


namespace NUMINAMATH_GPT_dad_steps_90_l1513_151322

/-- 
  Given:
  - When Dad takes 3 steps, Masha takes 5 steps.
  - When Masha takes 3 steps, Yasha takes 5 steps.
  - Masha and Yasha together made a total of 400 steps.

  Prove: 
  The number of steps that Dad took is 90.
-/
theorem dad_steps_90 (total_steps: ℕ) (masha_to_dad_ratio: ℕ) (yasha_to_masha_ratio: ℕ) (steps_masha_yasha: ℕ) (h1: masha_to_dad_ratio = 5) (h2: yasha_to_masha_ratio = 5) (h3: steps_masha_yasha = 400) :
  total_steps = 90 :=
by
  sorry

end NUMINAMATH_GPT_dad_steps_90_l1513_151322


namespace NUMINAMATH_GPT_remainder_when_dividing_p_by_g_is_3_l1513_151311

noncomputable def p (x : ℤ) : ℤ := x^5 - 2 * x^3 + 4 * x^2 + x + 5
noncomputable def g (x : ℤ) : ℤ := x + 2

theorem remainder_when_dividing_p_by_g_is_3 : p (-2) = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_dividing_p_by_g_is_3_l1513_151311


namespace NUMINAMATH_GPT_largest_number_of_minerals_per_shelf_l1513_151354

theorem largest_number_of_minerals_per_shelf (d : ℕ) :
  d ∣ 924 ∧ d ∣ 1386 ∧ d ∣ 462 ↔ d = 462 :=
by
  sorry

end NUMINAMATH_GPT_largest_number_of_minerals_per_shelf_l1513_151354


namespace NUMINAMATH_GPT_remainder_relation_l1513_151347

theorem remainder_relation (P P' D R R' : ℕ) (hP : P > P') (h1 : P % D = R) (h2 : P' % D = R') :
  ∃ C : ℕ, ((P + C) * P') % D ≠ (P * P') % D ∧ ∃ C : ℕ, ((P + C) * P') % D = (P * P') % D :=
by sorry

end NUMINAMATH_GPT_remainder_relation_l1513_151347


namespace NUMINAMATH_GPT_power_mod_2040_l1513_151320

theorem power_mod_2040 : (6^2040) % 13 = 1 := by
  -- Skipping the proof as the problem only requires the statement
  sorry

end NUMINAMATH_GPT_power_mod_2040_l1513_151320


namespace NUMINAMATH_GPT_cos_360_eq_one_l1513_151339

theorem cos_360_eq_one : Real.cos (2 * Real.pi) = 1 :=
by sorry

end NUMINAMATH_GPT_cos_360_eq_one_l1513_151339


namespace NUMINAMATH_GPT_winner_for_2023_winner_for_2024_l1513_151341

-- Definitions for the game conditions
def barbara_moves : List ℕ := [3, 5]
def jenna_moves : List ℕ := [1, 4, 5]

-- Lean theorem statement proving the required answers
theorem winner_for_2023 (coins : ℕ) (barbara_moves : List ℕ) (jenna_moves : List ℕ) :
  coins = 2023 →  -- Specifying that the game starts with 2023 coins
  (∀n, n ∈ barbara_moves → n ≤ 2023) ∧ (∀n, n ∈ jenna_moves → n ≤ 2023) → 
  -- Specifying valid moves for both players
  ∃ winner : String, winner = "Barbara" := 
sorry

theorem winner_for_2024 (coins : ℕ) (barbara_moves : List ℕ) (jenna_moves : List ℕ) :
  coins = 2024 →  -- Specifying that the game starts with 2024 coins
  (∀n, n ∈ barbara_moves → n ≤ 2024) ∧ (∀n, n ∈ jenna_moves → n ≤ 2024) → 
  -- Specifying valid moves for both players
  ∃ winner : String, winner = "Whoever starts" :=
sorry

end NUMINAMATH_GPT_winner_for_2023_winner_for_2024_l1513_151341


namespace NUMINAMATH_GPT_exists_q_lt_1_l1513_151333

variable {a : ℕ → ℝ}

theorem exists_q_lt_1 (h_nonneg : ∀ n, 0 ≤ a n)
  (h_rec : ∀ k m, a (k + m) ≤ a (k + m + 1) + a k * a m)
  (h_large_n : ∃ n₀, ∀ n ≥ n₀, n * a n < 0.2499) :
  ∃ q, 0 < q ∧ q < 1 ∧ (∃ n₀, ∀ n ≥ n₀, a n < q ^ n) :=
by
  sorry

end NUMINAMATH_GPT_exists_q_lt_1_l1513_151333


namespace NUMINAMATH_GPT_y_relationship_l1513_151367

theorem y_relationship (x1 x2 x3 y1 y2 y3 : ℝ) 
  (hA : y1 = -7 * x1 + 14) 
  (hB : y2 = -7 * x2 + 14) 
  (hC : y3 = -7 * x3 + 14) 
  (hx : x1 > x3 ∧ x3 > x2) : y1 < y3 ∧ y3 < y2 :=
by
  sorry

end NUMINAMATH_GPT_y_relationship_l1513_151367


namespace NUMINAMATH_GPT_num_factors_1728_l1513_151305

open Nat

noncomputable def num_factors (n : ℕ) : ℕ :=
  (6 + 1) * (3 + 1)

theorem num_factors_1728 : 
  num_factors 1728 = 28 := by
  sorry

end NUMINAMATH_GPT_num_factors_1728_l1513_151305


namespace NUMINAMATH_GPT_graduates_distribution_l1513_151329

theorem graduates_distribution (n : ℕ) (k : ℕ)
    (h_n : n = 5) (h_k : k = 3)
    (h_dist : ∀ e : Fin k, ∃ g : Finset (Fin n), g.card ≥ 1) :
    ∃ d : ℕ, d = 150 :=
by
  have h_distribution := 150
  use h_distribution
  sorry

end NUMINAMATH_GPT_graduates_distribution_l1513_151329


namespace NUMINAMATH_GPT_shaded_area_possible_values_l1513_151396

variable (AB BC PQ SC : ℕ)

-- Conditions:
def dimensions_correct : Prop := AB * BC = 33 ∧ AB < 7 ∧ BC < 7
def length_constraint : Prop := PQ < SC

-- Theorem statement
theorem shaded_area_possible_values (h1 : dimensions_correct AB BC) (h2 : length_constraint PQ SC) :
  (AB = 3 ∧ BC = 11 ∧ (PQ = 1 ∧ SC = 6 ∧ (33 - 1 * 4 - 2 * 6 = 17) ∨
                      (33 - 2 * 3 - 1 * 6 = 21) ∨
                      (33 - 2 * 4 - 1 * 5 = 20))) ∨ 
  (AB = 11 ∧ BC = 3 ∧ (PQ = 1 ∧ SC = 6 ∧ (33 - 1 * 4 - 2 * 6 = 17))) :=
sorry

end NUMINAMATH_GPT_shaded_area_possible_values_l1513_151396


namespace NUMINAMATH_GPT_min_total_cost_minimize_cost_l1513_151349

theorem min_total_cost (x : ℝ) (h₀ : x > 0) :
  (900 / x * 3 + 3 * x) ≥ 180 :=
by sorry

theorem minimize_cost (x : ℝ) (h₀ : x > 0) :
  x = 30 ↔ (900 / x * 3 + 3 * x) = 180 :=
by sorry

end NUMINAMATH_GPT_min_total_cost_minimize_cost_l1513_151349


namespace NUMINAMATH_GPT_find_S3_l1513_151315

noncomputable def geometric_sum (n : ℕ) : ℕ := sorry  -- Placeholder for the sum function.

theorem find_S3 (S : ℕ → ℕ) (hS6 : S 6 = 30) (hS9 : S 9 = 70) : S 3 = 10 :=
by
  -- Establish the needed conditions and equation 
  have h : (S 6 - S 3) ^ 2 = (S 9 - S 6) * S 3 := sorry
  -- Substitute given S6 and S9 into the equation and solve
  exact sorry

end NUMINAMATH_GPT_find_S3_l1513_151315


namespace NUMINAMATH_GPT_lcm_20_45_36_l1513_151356

-- Definitions from the problem
def num1 : ℕ := 20
def num2 : ℕ := 45
def num3 : ℕ := 36

-- Statement of the proof problem
theorem lcm_20_45_36 : Nat.lcm (Nat.lcm num1 num2) num3 = 180 := by
  sorry

end NUMINAMATH_GPT_lcm_20_45_36_l1513_151356


namespace NUMINAMATH_GPT_minimum_workers_required_l1513_151300

theorem minimum_workers_required (total_days : ℕ) (days_elapsed : ℕ) (initial_workers : ℕ) (job_fraction_done : ℚ)
  (remaining_work_fraction : job_fraction_done < 1) 
  (worker_productivity_constant : Prop) : 
  total_days = 40 → days_elapsed = 10 → initial_workers = 10 → job_fraction_done = (1/4) →
  (total_days - days_elapsed) * initial_workers * job_fraction_done = (1 - job_fraction_done) →
  job_fraction_done = 1 → initial_workers = 10 :=
by
  intros;
  sorry

end NUMINAMATH_GPT_minimum_workers_required_l1513_151300


namespace NUMINAMATH_GPT_partial_fraction_sum_zero_l1513_151384

theorem partial_fraction_sum_zero (A B C D E : ℚ) :
  (∀ x : ℚ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5)) →
  A + B + C + D + E = 0 :=
by
  sorry

end NUMINAMATH_GPT_partial_fraction_sum_zero_l1513_151384


namespace NUMINAMATH_GPT_rectangle_area_l1513_151362

theorem rectangle_area (a w l : ℝ) (h_square_area : a = 36) 
    (h_rect_width : w * w = a) 
    (h_rect_length : l = 3 * w) : w * l = 108 := 
sorry

end NUMINAMATH_GPT_rectangle_area_l1513_151362


namespace NUMINAMATH_GPT_range_of_k_for_positivity_l1513_151340

theorem range_of_k_for_positivity (k x : ℝ) (h1 : -1 ≤ x) (h2 : x ≤ 2) :
  ((k - 2) * x + 2 * |k| - 1 > 0) → (k > 5 / 4) :=
sorry

end NUMINAMATH_GPT_range_of_k_for_positivity_l1513_151340


namespace NUMINAMATH_GPT_total_swordfish_caught_correct_l1513_151385

-- Define the parameters and the catch per trip
def shelly_catch_per_trip : ℕ := 5 - 2
def sam_catch_per_trip (shelly_catch : ℕ) : ℕ := shelly_catch - 1
def total_catch_per_trip (shelly_catch sam_catch : ℕ) : ℕ := shelly_catch + sam_catch
def total_trips : ℕ := 5

-- Define the total number of swordfish caught over the trips
def total_swordfish_caught : ℕ :=
  let shelly_catch := shelly_catch_per_trip
  let sam_catch := sam_catch_per_trip shelly_catch
  let total_catch := total_catch_per_trip shelly_catch sam_catch
  total_catch * total_trips

-- The theorem states that the total swordfish caught is 25
theorem total_swordfish_caught_correct : total_swordfish_caught = 25 := by
  sorry

end NUMINAMATH_GPT_total_swordfish_caught_correct_l1513_151385


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1513_151368

variable (a b : ℝ)

theorem sufficient_but_not_necessary_condition (h1 : a < b) : 
  ((a - b) * a^2 < 0) ↔ (a < b) :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1513_151368


namespace NUMINAMATH_GPT_larger_number_l1513_151323

theorem larger_number (a b : ℕ) (h1 : 5 * b = 7 * a) (h2 : b - a = 10) : b = 35 :=
sorry

end NUMINAMATH_GPT_larger_number_l1513_151323


namespace NUMINAMATH_GPT_base_of_minus4_pow3_l1513_151350

theorem base_of_minus4_pow3 : ∀ (x : ℤ) (n : ℤ), (x, n) = (-4, 3) → x = -4 :=
by intros x n h
   cases h
   rfl

end NUMINAMATH_GPT_base_of_minus4_pow3_l1513_151350


namespace NUMINAMATH_GPT_exists_separating_line_l1513_151365

noncomputable def f1 (x : ℝ) (a1 b1 c1 : ℝ) : ℝ := a1 * x^2 + b1 * x + c1
noncomputable def f2 (x : ℝ) (a2 b2 c2 : ℝ) : ℝ := a2 * x^2 + b2 * x + c2

theorem exists_separating_line (a1 b1 c1 a2 b2 c2 : ℝ) (h_intersect : ∀ x, f1 x a1 b1 c1 ≠ f2 x a2 b2 c2)
  (h_neg : a1 * a2 < 0) : ∃ α β : ℝ, ∀ x, f1 x a1 b1 c1 < α * x + β ∧ α * x + β < f2 x a2 b2 c2 :=
sorry

end NUMINAMATH_GPT_exists_separating_line_l1513_151365


namespace NUMINAMATH_GPT_smallest_number_of_cubes_l1513_151381

theorem smallest_number_of_cubes (l w d : ℕ) (hl : l = 36) (hw : w = 45) (hd : d = 18) : 
  ∃ n : ℕ, n = 40 ∧ (∃ s : ℕ, l % s = 0 ∧ w % s = 0 ∧ d % s = 0 ∧ (l / s) * (w / s) * (d / s) = n) := 
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_cubes_l1513_151381


namespace NUMINAMATH_GPT_hardcover_volumes_l1513_151345

theorem hardcover_volumes (h p : ℕ) (h_condition : h + p = 12) (cost_condition : 25 * h + 15 * p = 240) : h = 6 :=
by
  -- omitted proof steps for brevity
  sorry

end NUMINAMATH_GPT_hardcover_volumes_l1513_151345


namespace NUMINAMATH_GPT_physics_majors_consecutive_probability_l1513_151336

open Nat

-- Define the total number of seats and the specific majors
def totalSeats : ℕ := 10
def mathMajors : ℕ := 4
def physicsMajors : ℕ := 3
def chemistryMajors : ℕ := 2
def biologyMajors : ℕ := 1

-- Assuming a round table configuration
def probabilityPhysicsMajorsConsecutive : ℚ :=
  (3 * (Nat.factorial (totalSeats - physicsMajors))) / (Nat.factorial (totalSeats - 1))

-- Declare the theorem
theorem physics_majors_consecutive_probability : 
  probabilityPhysicsMajorsConsecutive = 1 / 24 :=
by
  sorry

end NUMINAMATH_GPT_physics_majors_consecutive_probability_l1513_151336


namespace NUMINAMATH_GPT_length_AB_l1513_151358

noncomputable def parabola_p := 3
def x1_x2_sum := 6

theorem length_AB (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (h1 : x1 + x2 = x1_x2_sum)
  (h2 : (y1^2 = 6 * x1) ∧ (y2^2 = 6 * x2))
  : abs (x1 + parabola_p / 2 - (x2 + parabola_p / 2)) = 9 := by
  sorry

end NUMINAMATH_GPT_length_AB_l1513_151358


namespace NUMINAMATH_GPT_bowling_tournament_l1513_151316

-- Definition of the problem conditions
def playoff (num_bowlers: Nat): Nat := 
  if num_bowlers < 5 then
    0
  else
    2^(num_bowlers - 1)

-- Theorem statement to prove
theorem bowling_tournament (num_bowlers: Nat) (h: num_bowlers = 5): playoff num_bowlers = 16 := by
  sorry

end NUMINAMATH_GPT_bowling_tournament_l1513_151316


namespace NUMINAMATH_GPT_solution_set_inequality_l1513_151360

theorem solution_set_inequality (a x : ℝ) :
  (12 * x^2 - a * x > a^2) ↔
  ((a > 0 ∧ (x < -a / 4 ∨ x > a / 3)) ∨
   (a = 0 ∧ x ≠ 0) ∨
   (a < 0 ∧ (x > -a / 4 ∨ x < a / 3))) :=
sorry


end NUMINAMATH_GPT_solution_set_inequality_l1513_151360


namespace NUMINAMATH_GPT_probability_two_hearts_is_one_seventeenth_l1513_151359

-- Define the problem parameters
def totalCards : ℕ := 52
def hearts : ℕ := 13
def drawCount : ℕ := 2

-- Define function to calculate combinations
def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Define the probability calculation
def probability_drawing_two_hearts : ℚ :=
  (combination hearts drawCount) / (combination totalCards drawCount)

-- State the theorem to be proved
theorem probability_two_hearts_is_one_seventeenth :
  probability_drawing_two_hearts = 1 / 17 :=
by
  -- Proof not required, so provide sorry
  sorry

end NUMINAMATH_GPT_probability_two_hearts_is_one_seventeenth_l1513_151359


namespace NUMINAMATH_GPT_whitney_spent_179_l1513_151373

def total_cost (books_whales books_fish magazines book_cost magazine_cost : ℕ) : ℕ :=
  (books_whales + books_fish) * book_cost + magazines * magazine_cost

theorem whitney_spent_179 :
  total_cost 9 7 3 11 1 = 179 :=
by
  sorry

end NUMINAMATH_GPT_whitney_spent_179_l1513_151373


namespace NUMINAMATH_GPT_molly_age_l1513_151372

theorem molly_age
  (S M : ℕ)
  (h1 : S / M = 4 / 3)
  (h2 : S + 6 = 30) :
  M = 18 :=
sorry

end NUMINAMATH_GPT_molly_age_l1513_151372


namespace NUMINAMATH_GPT_total_resistance_l1513_151312

theorem total_resistance (x y z : ℝ) (R_parallel r : ℝ)
    (hx : x = 3)
    (hy : y = 6)
    (hz : z = 4)
    (hR_parallel : 1 / R_parallel = 1 / x + 1 / y)
    (hr : r = R_parallel + z) :
    r = 6 := by
  sorry

end NUMINAMATH_GPT_total_resistance_l1513_151312


namespace NUMINAMATH_GPT_problem_solution_l1513_151308

theorem problem_solution :
  (- (5 : ℚ) / 12) ^ 2023 * (12 / 5) ^ 2023 = -1 := 
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1513_151308


namespace NUMINAMATH_GPT_solve_for_x_l1513_151376

noncomputable def n : ℝ := Real.sqrt (7^2 + 24^2)
noncomputable def d : ℝ := Real.sqrt (49 + 16)
noncomputable def x : ℝ := n / d

theorem solve_for_x : x = 5 * Real.sqrt 65 / 13 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1513_151376


namespace NUMINAMATH_GPT_horse_problem_l1513_151398

theorem horse_problem (x : ℕ) :
  150 * (x + 12) = 240 * x :=
sorry

end NUMINAMATH_GPT_horse_problem_l1513_151398


namespace NUMINAMATH_GPT_molecular_weight_correct_l1513_151328

-- Definition of atomic weights for the elements
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Number of atoms in Ascorbic acid (C6H8O6)
def count_C : ℕ := 6
def count_H : ℕ := 8
def count_O : ℕ := 6

-- Calculation of molecular weight
def molecular_weight_ascorbic_acid : ℝ :=
  (count_C * atomic_weight_C) +
  (count_H * atomic_weight_H) +
  (count_O * atomic_weight_O)

theorem molecular_weight_correct :
  molecular_weight_ascorbic_acid = 176.124 :=
by sorry


end NUMINAMATH_GPT_molecular_weight_correct_l1513_151328


namespace NUMINAMATH_GPT_clarissa_copies_needed_l1513_151325

-- Define the given conditions
def manuscript_pages : ℕ := 400
def cost_per_page : ℚ := 0.05
def cost_per_binding : ℚ := 5.00
def total_cost : ℚ := 250.00

-- Calculate the total cost for one manuscript
def cost_per_copy_and_bind : ℚ := cost_per_page * manuscript_pages + cost_per_binding

-- Define number of copies needed
def number_of_copies_needed : ℚ := total_cost / cost_per_copy_and_bind

-- Prove number of copies needed is 10
theorem clarissa_copies_needed : number_of_copies_needed = 10 := 
by 
  -- Implementing the proof steps would go here
  sorry

end NUMINAMATH_GPT_clarissa_copies_needed_l1513_151325


namespace NUMINAMATH_GPT_bob_expected_difference_l1513_151318

-- Required definitions and conditions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_three (n : ℕ) : Prop := n % 3 = 0

def probability_of_event_s : ℚ := 4 / 7
def probability_of_event_u : ℚ := 2 / 7
def probability_of_event_s_and_u : ℚ := 1 / 7
def number_of_days : ℕ := 365

noncomputable def expected_days_sweetened : ℚ :=
   (probability_of_event_s - (1 / 2) * probability_of_event_s_and_u) * number_of_days

noncomputable def expected_days_unsweetened : ℚ :=
   (probability_of_event_u - (1 / 2) * probability_of_event_s_and_u) * number_of_days

noncomputable def expected_difference : ℚ :=
   expected_days_sweetened - expected_days_unsweetened

theorem bob_expected_difference : expected_difference = 135.45 := sorry

end NUMINAMATH_GPT_bob_expected_difference_l1513_151318


namespace NUMINAMATH_GPT_arithmetic_sequence_geometric_l1513_151379

theorem arithmetic_sequence_geometric (a : ℕ → ℤ) (d : ℤ) (m n : ℕ)
  (h1 : ∀ n, a (n+1) = a 1 + n * d)
  (h2 : a 1 = 1)
  (h3 : (a 3 - 2)^2 = a 1 * a 5)
  (h_d_nonzero : d ≠ 0)
  (h_mn : m - n = 10) :
  a m - a n = 30 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_geometric_l1513_151379


namespace NUMINAMATH_GPT_domain_of_log_function_l1513_151363

open Real

noncomputable def domain_of_function : Set ℝ :=
  {x | x > 2 ∨ x < -1}

theorem domain_of_log_function :
  ∀ x : ℝ, (x^2 - x - 2 > 0) ↔ (x > 2 ∨ x < -1) :=
by
  intro x
  exact sorry

end NUMINAMATH_GPT_domain_of_log_function_l1513_151363


namespace NUMINAMATH_GPT_withdraw_representation_l1513_151334

-- Define the concept of depositing and withdrawing money.
def deposit (amount : ℕ) : ℤ := amount
def withdraw (amount : ℕ) : ℤ := - amount

-- Define the given condition: depositing $30,000 is represented as $+30,000.
def deposit_condition : deposit 30000 = 30000 := by rfl

-- The statement to be proved: withdrawing $40,000 is represented as $-40,000
theorem withdraw_representation (deposit_condition : deposit 30000 = 30000) : withdraw 40000 = -40000 :=
by
  sorry

end NUMINAMATH_GPT_withdraw_representation_l1513_151334


namespace NUMINAMATH_GPT_midpoint_distance_l1513_151386

theorem midpoint_distance (a b c d : ℝ) :
  let m := (a + c) / 2
  let n := (b + d) / 2
  let m' := m - 0.5
  let n' := n - 0.5
  dist (m, n) (m', n') = (Real.sqrt 2) / 2 := 
by 
  sorry

end NUMINAMATH_GPT_midpoint_distance_l1513_151386


namespace NUMINAMATH_GPT_local_minimum_at_neg_one_l1513_151338

noncomputable def f (x : ℝ) := x * Real.exp x

theorem local_minimum_at_neg_one : (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x + 1) < δ → f x > f (-1)) :=
sorry

end NUMINAMATH_GPT_local_minimum_at_neg_one_l1513_151338


namespace NUMINAMATH_GPT_age_difference_l1513_151307

variables (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 18) : C = A - 18 :=
sorry

end NUMINAMATH_GPT_age_difference_l1513_151307


namespace NUMINAMATH_GPT_speed_increase_71_6_percent_l1513_151353

theorem speed_increase_71_6_percent (S : ℝ) (hS : 0 < S) : 
    let S₁ := S * 1.30
    let S₂ := S₁ * 1.10
    let S₃ := S₂ * 1.20
    (S₃ - S) / S * 100 = 71.6 :=
by
  let S₁ := S * 1.30
  let S₂ := S₁ * 1.10
  let S₃ := S₂ * 1.20
  sorry

end NUMINAMATH_GPT_speed_increase_71_6_percent_l1513_151353


namespace NUMINAMATH_GPT_total_rooms_count_l1513_151309

noncomputable def apartment_area : ℕ := 160
noncomputable def living_room_area : ℕ := 60
noncomputable def other_room_area : ℕ := 20

theorem total_rooms_count (A : apartment_area = 160) (L : living_room_area = 60) (O : other_room_area = 20) :
  1 + (apartment_area - living_room_area) / other_room_area = 6 :=
by
  sorry

end NUMINAMATH_GPT_total_rooms_count_l1513_151309


namespace NUMINAMATH_GPT_original_price_vase_l1513_151366

-- Definitions based on the conditions and problem elements
def original_price (P : ℝ) : Prop :=
  0.825 * P = 165

-- Statement to prove equivalence
theorem original_price_vase : ∃ P : ℝ, original_price P ∧ P = 200 :=
  by
    sorry

end NUMINAMATH_GPT_original_price_vase_l1513_151366


namespace NUMINAMATH_GPT_annulus_area_l1513_151387

theorem annulus_area (r_inner r_outer : ℝ) (h_inner : r_inner = 8) (h_outer : r_outer = 2 * r_inner) :
  π * r_outer ^ 2 - π * r_inner ^ 2 = 192 * π :=
by
  sorry

end NUMINAMATH_GPT_annulus_area_l1513_151387


namespace NUMINAMATH_GPT_geometric_sequence_condition_l1513_151388

variable (a b c : ℝ)

-- Condition: For a, b, c to form a geometric sequence.
def is_geometric_sequence (a b c : ℝ) : Prop :=
  (b ≠ 0) ∧ (b^2 = a * c)

-- Given that a, b, c are real numbers
-- Prove that ac = b^2 is a necessary but not sufficient condition for a, b, c to form a geometric sequence.
theorem geometric_sequence_condition (a b c : ℝ) (h : a * c = b^2) :
  ¬ (∃ b : ℝ, b^2 = a * c → (is_geometric_sequence a b c)) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_condition_l1513_151388


namespace NUMINAMATH_GPT_albert_earnings_l1513_151314

theorem albert_earnings (E E_final : ℝ) : 
  (0.90 * (E * 1.14) = 678) → 
  (E_final = 0.90 * (E * 1.15 * 1.20)) → 
  E_final = 819.72 :=
by
  sorry

end NUMINAMATH_GPT_albert_earnings_l1513_151314


namespace NUMINAMATH_GPT_cylinder_height_relation_l1513_151369

variables (r1 h1 r2 h2 : ℝ)
variables (V1_eq_V2 : π * r1^2 * h1 = π * r2^2 * h2) (r2_eq_1_2_r1 : r2 = 1.2 * r1)

theorem cylinder_height_relation : h1 = 1.44 * h2 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_height_relation_l1513_151369


namespace NUMINAMATH_GPT_extreme_value_0_at_minus_1_l1513_151351

theorem extreme_value_0_at_minus_1 (m n : ℝ)
  (h1 : (-1) + 3 * m - n + m^2 = 0)
  (h2 : 3 - 6 * m + n = 0) :
  m + n = 11 :=
sorry

end NUMINAMATH_GPT_extreme_value_0_at_minus_1_l1513_151351


namespace NUMINAMATH_GPT_quadratic_expression_always_positive_l1513_151321

theorem quadratic_expression_always_positive (x y : ℝ) : 
  x^2 - 4 * x * y + 6 * y^2 - 4 * y + 3 > 0 :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_expression_always_positive_l1513_151321


namespace NUMINAMATH_GPT_roots_squared_sum_eq_13_l1513_151324

/-- Let p and q be the roots of the quadratic equation x^2 - 5x + 6 = 0. Then the value of p^2 + q^2 is 13. -/
theorem roots_squared_sum_eq_13 (p q : ℝ) (h₁ : p + q = 5) (h₂ : p * q = 6) : p^2 + q^2 = 13 :=
by
  sorry

end NUMINAMATH_GPT_roots_squared_sum_eq_13_l1513_151324


namespace NUMINAMATH_GPT_both_questions_correct_l1513_151330

def total_students := 100
def first_question_correct := 75
def second_question_correct := 30
def neither_question_correct := 20

theorem both_questions_correct :
  (first_question_correct + second_question_correct - (total_students - neither_question_correct)) = 25 :=
by
  sorry

end NUMINAMATH_GPT_both_questions_correct_l1513_151330


namespace NUMINAMATH_GPT_option_C_correct_l1513_151370

theorem option_C_correct (m n : ℝ) (h : m > n) : (1/5) * m > (1/5) * n := 
by
  sorry

end NUMINAMATH_GPT_option_C_correct_l1513_151370


namespace NUMINAMATH_GPT_no_difference_410_l1513_151332

theorem no_difference_410 (n : ℕ) (R L a : ℕ) (h1 : R + L = 300)
  (h2 : L = 300 - R)
  (h3 : a ≤ 2 * R)
  (h4 : n = L + a)  :
  ¬ (n = 410) :=
by
  sorry

end NUMINAMATH_GPT_no_difference_410_l1513_151332


namespace NUMINAMATH_GPT_arithmetic_geometric_inequality_l1513_151399

theorem arithmetic_geometric_inequality (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  (a + b) / 2 ≥ Real.sqrt (a * b) ∧ ((a + b) / 2 = Real.sqrt (a * b) ↔ a = b) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_inequality_l1513_151399


namespace NUMINAMATH_GPT_function_even_iff_a_eq_one_l1513_151319

theorem function_even_iff_a_eq_one (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x : ℝ, f x = a * (3^x) + 1/(3^x)) → 
  (∀ x : ℝ, f x = f (-x)) ↔ a = 1 :=
by
  sorry

end NUMINAMATH_GPT_function_even_iff_a_eq_one_l1513_151319


namespace NUMINAMATH_GPT_craig_apples_total_l1513_151327

-- Conditions
def initial_apples := 20.0
def additional_apples := 7.0

-- Question turned into a proof problem
theorem craig_apples_total : initial_apples + additional_apples = 27.0 :=
by
  sorry

end NUMINAMATH_GPT_craig_apples_total_l1513_151327


namespace NUMINAMATH_GPT_mass_percentage_O_in_Al2_CO3_3_l1513_151380

-- Define the atomic masses
def atomic_mass_Al : Float := 26.98
def atomic_mass_C : Float := 12.01
def atomic_mass_O : Float := 16.00

-- Define the formula of aluminum carbonate
def Al_count : Nat := 2
def C_count : Nat := 3
def O_count : Nat := 9

-- Define the molar mass calculation
def molar_mass_Al2_CO3_3 : Float :=
  (Al_count.toFloat * atomic_mass_Al) + 
  (C_count.toFloat * atomic_mass_C) + 
  (O_count.toFloat * atomic_mass_O)

-- Define the mass of oxygen in aluminum carbonate
def mass_O_in_Al2_CO3_3 : Float := O_count.toFloat * atomic_mass_O

-- Define the mass percentage of oxygen in aluminum carbonate
def mass_percentage_O : Float := (mass_O_in_Al2_CO3_3 / molar_mass_Al2_CO3_3) * 100

-- Proof statement
theorem mass_percentage_O_in_Al2_CO3_3 :
  mass_percentage_O = 61.54 := by
  sorry

end NUMINAMATH_GPT_mass_percentage_O_in_Al2_CO3_3_l1513_151380


namespace NUMINAMATH_GPT_amitabh_avg_expenditure_feb_to_jul_l1513_151383

variable (expenditure_avg_jan_to_jun expenditure_jan expenditure_jul : ℕ)

theorem amitabh_avg_expenditure_feb_to_jul (h1 : expenditure_avg_jan_to_jun = 4200) 
  (h2 : expenditure_jan = 1200) (h3 : expenditure_jul = 1500) :
  (expenditure_avg_jan_to_jun * 6 - expenditure_jan + expenditure_jul) / 6 = 4250 := by
  -- Using the given conditions
  sorry

end NUMINAMATH_GPT_amitabh_avg_expenditure_feb_to_jul_l1513_151383


namespace NUMINAMATH_GPT_sequence_a2002_l1513_151375

theorem sequence_a2002 :
  ∀ (a : ℕ → ℕ), (a 1 = 1) → (a 2 = 2) → 
  (∀ n, 2 ≤ n → a (n + 1) = 3 * a n - 2 * a (n - 1)) → 
  a 2002 = 2 ^ 2001 :=
by
  intros a ha1 ha2 hrecur
  sorry

end NUMINAMATH_GPT_sequence_a2002_l1513_151375


namespace NUMINAMATH_GPT_price_of_n_kilograms_l1513_151394

theorem price_of_n_kilograms (m n : ℕ) (hm : m ≠ 0) (h : 9 = m) : (9 * n) / m = (9 * n) / m :=
by
  sorry

end NUMINAMATH_GPT_price_of_n_kilograms_l1513_151394


namespace NUMINAMATH_GPT_library_shelves_l1513_151346

theorem library_shelves (S : ℕ) (h_books : 4305 + 11 = 4316) :
  4316 % S = 0 ↔ S = 11 :=
by 
  have h_total_books := h_books
  sorry

end NUMINAMATH_GPT_library_shelves_l1513_151346


namespace NUMINAMATH_GPT_simplify_expression_l1513_151304

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = x⁻¹ * y⁻¹ * z⁻¹ :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1513_151304


namespace NUMINAMATH_GPT_problem_proof_l1513_151395

theorem problem_proof (M N : ℕ) 
  (h1 : 4 * 63 = 7 * M) 
  (h2 : 4 * N = 7 * 84) : 
  M + N = 183 :=
sorry

end NUMINAMATH_GPT_problem_proof_l1513_151395


namespace NUMINAMATH_GPT_trajectory_of_center_l1513_151377

theorem trajectory_of_center :
  ∃ (x y : ℝ), (x + 1) ^ 2 + y ^ 2 = 49 / 4 ∧ (x - 1) ^ 2 + y ^ 2 = 1 / 4 ∧ ( ∀ P, (P = (x, y) → (P.1^2) / 4 + (P.2^2) / 3 = 1) ) := sorry

end NUMINAMATH_GPT_trajectory_of_center_l1513_151377


namespace NUMINAMATH_GPT_poly_value_at_two_l1513_151361

def f (x : ℝ) : ℝ := x^5 + 2 * x^4 + 3 * x^3 + 4 * x^2 + 5 * x + 6

theorem poly_value_at_two : f 2 = 216 :=
by
  unfold f
  norm_num
  sorry

end NUMINAMATH_GPT_poly_value_at_two_l1513_151361


namespace NUMINAMATH_GPT_polynomial_characterization_l1513_151313
open Polynomial

noncomputable def satisfies_functional_eq (P : Polynomial ℝ) :=
  ∀ (a b c : ℝ), 
  P.eval (a + b - 2*c) + P.eval (b + c - 2*a) + P.eval (c + a - 2*b) = 
  3 * P.eval (a - b) + 3 * P.eval (b - c) + 3 * P.eval (c - a)

theorem polynomial_characterization (P : Polynomial ℝ) :
  satisfies_functional_eq P ↔ 
  (∃ a b : ℝ, P = Polynomial.C a * Polynomial.X + Polynomial.C b) ∨
  (∃ a b : ℝ, P = Polynomial.C a * Polynomial.X^2 + Polynomial.C b * Polynomial.X) :=
sorry

end NUMINAMATH_GPT_polynomial_characterization_l1513_151313


namespace NUMINAMATH_GPT_sandra_fathers_contribution_ratio_l1513_151344

theorem sandra_fathers_contribution_ratio :
  let saved := 10
  let mother := 4
  let candy_cost := 0.5
  let jellybean_cost := 0.2
  let candies := 14
  let jellybeans := 20
  let remaining := 11
  let total_cost := candies * candy_cost + jellybeans * jellybean_cost
  let total_amount := total_cost + remaining
  let amount_without_father := saved + mother
  let father := total_amount - amount_without_father
  (father / mother) = 2 := by 
  sorry

end NUMINAMATH_GPT_sandra_fathers_contribution_ratio_l1513_151344


namespace NUMINAMATH_GPT_sum_of_a_and_b_l1513_151331

theorem sum_of_a_and_b (a b : ℝ) (h_neq : a ≠ b) (h_a : a * (a - 4) = 21) (h_b : b * (b - 4) = 21) :
  a + b = 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_a_and_b_l1513_151331


namespace NUMINAMATH_GPT_june_earnings_l1513_151355

theorem june_earnings
  (total_clovers : ℕ)
  (clover_3_petals_percentage : ℝ)
  (clover_2_petals_percentage : ℝ)
  (clover_4_petals_percentage : ℝ)
  (earnings_per_clover : ℝ) :
  total_clovers = 200 →
  clover_3_petals_percentage = 0.75 →
  clover_2_petals_percentage = 0.24 →
  clover_4_petals_percentage = 0.01 →
  earnings_per_clover = 1 →
  (total_clovers * earnings_per_clover) = 200 := by
  sorry

end NUMINAMATH_GPT_june_earnings_l1513_151355


namespace NUMINAMATH_GPT_solution_set_inequality_l1513_151352

open Real

theorem solution_set_inequality (k : ℤ) (x : ℝ) :
  (x ∈ Set.Ioo (-π/4 + k * π) (k * π)) ↔ cos (4 * x) - 2 * sin (2 * x) - sin (4 * x) - 1 > 0 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1513_151352


namespace NUMINAMATH_GPT_side_length_of_largest_square_l1513_151326

theorem side_length_of_largest_square (A_cross : ℝ) (s : ℝ)
  (h1 : A_cross = 810) : s = 36 :=
  have h_large_squares : 2 * (s / 2)^2 = s^2 / 2 := by sorry
  have h_small_squares : 2 * (s / 4)^2 = s^2 / 8 := by sorry
  have h_combined_area : s^2 / 2 + s^2 / 8 = 810 := by sorry
  have h_final : 5 * s^2 / 8 = 810 := by sorry
  have h_s2 : s^2 = 1296 := by sorry
  have h_s : s = 36 := by sorry
  h_s

end NUMINAMATH_GPT_side_length_of_largest_square_l1513_151326


namespace NUMINAMATH_GPT_Gyeongyeon_cookies_l1513_151342

def initial_cookies : ℕ := 20
def cookies_given : ℕ := 7
def cookies_received : ℕ := 5

def final_cookies (initial : ℕ) (given : ℕ) (received : ℕ) : ℕ :=
  initial - given + received

theorem Gyeongyeon_cookies :
  final_cookies initial_cookies cookies_given cookies_received = 18 :=
by
  sorry

end NUMINAMATH_GPT_Gyeongyeon_cookies_l1513_151342


namespace NUMINAMATH_GPT_second_container_clay_l1513_151391

theorem second_container_clay :
  let h1 := 3
  let w1 := 5
  let l1 := 7
  let clay1 := 105
  let h2 := 3 * h1
  let w2 := 2 * w1
  let l2 := l1
  let V1 := h1 * w1 * l1
  let V2 := h2 * w2 * l2
  V1 = clay1 →
  V2 = 6 * V1 →
  V2 = 630 :=
by
  intros
  sorry

end NUMINAMATH_GPT_second_container_clay_l1513_151391


namespace NUMINAMATH_GPT_find_k_common_term_l1513_151301

def sequence_a (k : ℕ) (n : ℕ) : ℕ :=
  if n = 1 then 1 
  else if n = 2 then k 
  else if n = 3 then 3*k - 3 
  else if n = 4 then 6*k - 8 
  else (n * (n-1) * (k-2)) / 2 + n

def is_fermat (x : ℕ) : Prop :=
  ∃ m : ℕ, x = 2^(2^m) + 1

theorem find_k_common_term (k : ℕ) :
  k > 2 → ∃ n m : ℕ, sequence_a k n = 2^(2^m) + 1 :=
by
  sorry

end NUMINAMATH_GPT_find_k_common_term_l1513_151301


namespace NUMINAMATH_GPT_segment_length_at_1_point_5_l1513_151378

-- Definitions for the conditions
def Point := ℝ × ℝ
def Triangle (A B C : Point) := ∃ a b c : ℝ, a = 4 ∧ b = 3 ∧ c = 5 ∧ (A = (0, 0)) ∧ (B = (4, 0)) ∧ (C = (0, 3)) ∧ (c^2 = a^2 + b^2)

noncomputable def length_l (x : ℝ) : ℝ := (4 * (abs ((3/4) * x + 3))) / 5

theorem segment_length_at_1_point_5 (A B C : Point) (h : Triangle A B C) : 
  length_l 1.5 = 3.3 := by 
  sorry

end NUMINAMATH_GPT_segment_length_at_1_point_5_l1513_151378


namespace NUMINAMATH_GPT_stadium_length_l1513_151335

theorem stadium_length
  (W : ℝ) (H : ℝ) (P : ℝ) (L : ℝ)
  (h1 : W = 18)
  (h2 : H = 16)
  (h3 : P = 34)
  (h4 : P^2 = L^2 + W^2 + H^2) :
  L = 24 :=
by
  sorry

end NUMINAMATH_GPT_stadium_length_l1513_151335


namespace NUMINAMATH_GPT_find_n_l1513_151374

theorem find_n (x n : ℝ) (h : x > 0) 
  (h_eq : x / 10 + x / n = 0.14000000000000002 * x) : 
  n = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1513_151374


namespace NUMINAMATH_GPT_part_a_l1513_151382

theorem part_a (a b c : Int) (h1 : a + b + c = 0) : 
  ¬(a ^ 1999 + b ^ 1999 + c ^ 1999 = 2) :=
by
  sorry

end NUMINAMATH_GPT_part_a_l1513_151382


namespace NUMINAMATH_GPT_race_time_difference_l1513_151392

-- Define Malcolm's speed, Joshua's speed, and the distance
def malcolm_speed := 6 -- minutes per mile
def joshua_speed := 7 -- minutes per mile
def race_distance := 15 -- miles

-- Statement of the theorem
theorem race_time_difference :
  (joshua_speed * race_distance) - (malcolm_speed * race_distance) = 15 :=
by sorry

end NUMINAMATH_GPT_race_time_difference_l1513_151392


namespace NUMINAMATH_GPT_smallest_x_such_that_sum_is_cubic_l1513_151302

/-- 
  Given a positive integer x, the sum of the sequence x, x+3, x+6, x+9, and x+12 should be a perfect cube.
  Prove that the smallest such x is 19.
-/
theorem smallest_x_such_that_sum_is_cubic : 
  ∃ (x : ℕ), 0 < x ∧ (∃ k : ℕ, 5 * x + 30 = k^3) ∧ ∀ y : ℕ, 0 < y → (∃ m : ℕ, 5 * y + 30 = m^3) → y ≥ x :=
sorry

end NUMINAMATH_GPT_smallest_x_such_that_sum_is_cubic_l1513_151302


namespace NUMINAMATH_GPT_victor_percentage_of_marks_l1513_151310

theorem victor_percentage_of_marks (marks_obtained max_marks : ℝ) (percentage : ℝ) 
  (h_marks_obtained : marks_obtained = 368) 
  (h_max_marks : max_marks = 400) 
  (h_percentage : percentage = (marks_obtained / max_marks) * 100) : 
  percentage = 92 := by
sorry

end NUMINAMATH_GPT_victor_percentage_of_marks_l1513_151310


namespace NUMINAMATH_GPT_other_root_of_quadratic_l1513_151364

theorem other_root_of_quadratic (m : ℝ) (h : ∀ x : ℝ, x^2 + m*x - 20 = 0 → (x = -4)) 
: ∃ t : ℝ, t = 5 := 
by
  existsi 5
  sorry

end NUMINAMATH_GPT_other_root_of_quadratic_l1513_151364


namespace NUMINAMATH_GPT_part1_part2_part3_l1513_151317

-- Part (1)
theorem part1 (m n : ℤ) (h1 : m - n = -1) : 2 * (m - n)^2 + 18 = 20 := 
sorry

-- Part (2)
theorem part2 (m n : ℤ) (h2 : m^2 + 2 * m * n = 10) (h3 : n^2 + 3 * m * n = 6) : 2 * m^2 + n^2 + 7 * m * n = 26 :=
sorry

-- Part (3)
theorem part3 (a b c m x : ℤ) (h4: ax^5 + bx^3 + cx - 5 = m) (h5: x = -1) : ax^5 + bx^3 + cx - 5 = -m - 10 :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l1513_151317


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l1513_151389

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 0, y := 2 }
def B : Point := { x := 6, y := 0 }
def C : Point := { x := 4, y := 7 }

def triangle_area (P1 P2 P3 : Point) : ℝ :=
  0.5 * abs (P1.x * (P2.y - P3.y) +
             P2.x * (P3.y - P1.y) +
             P3.x * (P1.y - P2.y))

theorem area_of_triangle_ABC : triangle_area A B C = 19 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l1513_151389


namespace NUMINAMATH_GPT_f_m_plus_1_positive_l1513_151306

def f (a x : ℝ) := x^2 + x + a

theorem f_m_plus_1_positive (a m : ℝ) (ha : a > 0) (hm : f a m < 0) : f a (m + 1) > 0 := 
  sorry

end NUMINAMATH_GPT_f_m_plus_1_positive_l1513_151306


namespace NUMINAMATH_GPT_weights_identical_l1513_151343

theorem weights_identical (w : Fin 13 → ℤ) 
  (h : ∀ i, ∃ (A B : Finset (Fin 13)), A.card = 6 ∧ B.card = 6 ∧ A ∪ B = Finset.univ.erase i ∧ (A.sum w) = (B.sum w)) :
  ∀ i j, w i = w j :=
by
  sorry

end NUMINAMATH_GPT_weights_identical_l1513_151343


namespace NUMINAMATH_GPT_find_x_l1513_151348

def vec (x y : ℝ) := (x, y)

def a := vec 1 (-4)
def b (x : ℝ) := vec (-1) x
def c (x : ℝ) := (a.1 + 3 * (b x).1, a.2 + 3 * (b x).2)

theorem find_x (x : ℝ) : a.1 * (c x).2 = (c x).1 * a.2 → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1513_151348


namespace NUMINAMATH_GPT_evaluate_expression_l1513_151357

theorem evaluate_expression :
  let a := 3^1005
  let b := 4^1006
  (a + b)^2 - (a - b)^2 = 160 * 10^1004 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1513_151357


namespace NUMINAMATH_GPT_evaluate_fraction_l1513_151390

noncomputable def evaluate_expression : ℚ := 
  1 / (2 - (1 / (2 - (1 / (2 - (1 / 3))))))
  
theorem evaluate_fraction :
  evaluate_expression = 5 / 7 :=
sorry

end NUMINAMATH_GPT_evaluate_fraction_l1513_151390


namespace NUMINAMATH_GPT_abs_neg_five_l1513_151371

theorem abs_neg_five : abs (-5) = 5 :=
by
  sorry

end NUMINAMATH_GPT_abs_neg_five_l1513_151371
