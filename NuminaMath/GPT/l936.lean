import Mathlib

namespace NUMINAMATH_GPT_smallest_10_digit_number_with_sum_81_l936_93696

def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.sum

theorem smallest_10_digit_number_with_sum_81 {n : Nat} :
  n ≥ 1000000000 ∧ n < 10000000000 ∧ sum_of_digits n ≥ 81 → 
  n = 1899999999 :=
sorry

end NUMINAMATH_GPT_smallest_10_digit_number_with_sum_81_l936_93696


namespace NUMINAMATH_GPT_symmetrical_implies_congruent_l936_93626

-- Define a structure to represent figures
structure Figure where
  segments : Set ℕ
  angles : Set ℕ

-- Define symmetry about a line
def is_symmetrical_about_line (f1 f2 : Figure) : Prop :=
  ∀ s ∈ f1.segments, s ∈ f2.segments ∧ ∀ a ∈ f1.angles, a ∈ f2.angles

-- Define congruent figures
def are_congruent (f1 f2 : Figure) : Prop :=
  f1.segments = f2.segments ∧ f1.angles = f2.angles

-- Lean 4 statement of the proof problem
theorem symmetrical_implies_congruent (f1 f2 : Figure) (h : is_symmetrical_about_line f1 f2) : are_congruent f1 f2 :=
by
  sorry

end NUMINAMATH_GPT_symmetrical_implies_congruent_l936_93626


namespace NUMINAMATH_GPT_common_chord_length_l936_93636

noncomputable def dist_to_line (P : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * P.1 + b * P.2 + c) / Real.sqrt (a^2 + b^2)

theorem common_chord_length
  (x y : ℝ)
  (h1 : (x-2)^2 + (y-1)^2 = 10)
  (h2 : (x+6)^2 + (y+3)^2 = 50) :
  (dist_to_line (2, 1) 2 1 0 = Real.sqrt 5) →
  2 * Real.sqrt 5 = 2 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_common_chord_length_l936_93636


namespace NUMINAMATH_GPT_range_of_m_l936_93620

-- Definitions from conditions
def p (m : ℝ) : Prop := (∃ x y : ℝ, 2 * x^2 / m + y^2 / (m - 1) = 1)
def q (m : ℝ) : Prop := ∃ x1 : ℝ, 8 * x1^2 - 8 * m * x1 + 7 * m - 6 = 0
def proposition (m : ℝ) : Prop := (p m ∨ q m) ∧ ¬ (p m ∧ q m)

-- Proof statement
theorem range_of_m (m : ℝ) (h : proposition m) : (m ≤ 1 ∨ (3 / 2 < m ∧ m < 2)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l936_93620


namespace NUMINAMATH_GPT_min_side_b_of_triangle_l936_93605

theorem min_side_b_of_triangle (A B C a b c : ℝ) 
  (h_arith_seq : 2 * B = A + C)
  (h_sum_angles : A + B + C = Real.pi)
  (h_sides_opposite : b^2 = a^2 + c^2 - 2 * a * c * Real.cos B)
  (h_given_eq : 3 * a * c + b^2 = 25) :
  b ≥ 5 / 2 :=
  sorry

end NUMINAMATH_GPT_min_side_b_of_triangle_l936_93605


namespace NUMINAMATH_GPT_tom_sleep_increase_l936_93679

theorem tom_sleep_increase :
  ∀ (initial_sleep : ℕ) (increase_by : ℚ), 
  initial_sleep = 6 → 
  increase_by = 1/3 → 
  initial_sleep + increase_by * initial_sleep = 8 :=
by 
  intro initial_sleep increase_by h1 h2
  simp [*, add_mul, mul_comm]
  sorry

end NUMINAMATH_GPT_tom_sleep_increase_l936_93679


namespace NUMINAMATH_GPT_yellow_paint_amount_l936_93691

theorem yellow_paint_amount (b y : ℕ) (h_ratio : y * 7 = 3 * b) (h_blue_amount : b = 21) : y = 9 :=
by
  sorry

end NUMINAMATH_GPT_yellow_paint_amount_l936_93691


namespace NUMINAMATH_GPT_henry_added_water_l936_93677

theorem henry_added_water (initial_fraction full_capacity final_fraction : ℝ) (h_initial_fraction : initial_fraction = 3/4) (h_full_capacity : full_capacity = 56) (h_final_fraction : final_fraction = 7/8) :
  final_fraction * full_capacity - initial_fraction * full_capacity = 7 :=
by
  sorry

end NUMINAMATH_GPT_henry_added_water_l936_93677


namespace NUMINAMATH_GPT_temperature_fifth_day_l936_93634

variable (T1 T2 T3 T4 T5 : ℝ)

-- Conditions
def condition1 : T1 + T2 + T3 + T4 = 4 * 58 := by sorry
def condition2 : T2 + T3 + T4 + T5 = 4 * 59 := by sorry
def condition3 : T5 = (8 / 7) * T1 := by sorry

-- The statement we need to prove
theorem temperature_fifth_day : T5 = 32 := by
  -- Using the provided conditions
  sorry

end NUMINAMATH_GPT_temperature_fifth_day_l936_93634


namespace NUMINAMATH_GPT_smallest_possible_b_l936_93621

-- Definitions of conditions
variables {a b c : ℤ}

-- Conditions expressed in Lean
def is_geometric_progression (a b c : ℤ) : Prop := b^2 = a * c
def is_arithmetic_progression (a b c : ℤ) : Prop := a + b = 2 * c

-- The theorem statement
theorem smallest_possible_b (a b c : ℤ) 
  (h1 : a < b) (h2 : b < c) 
  (hg : is_geometric_progression a b c) 
  (ha : is_arithmetic_progression a c b) : b = 2 := sorry

end NUMINAMATH_GPT_smallest_possible_b_l936_93621


namespace NUMINAMATH_GPT_range_of_a_l936_93608

theorem range_of_a (a x y : ℝ) (h1 : 77 * a = (2 * x + 2 * y) / 2) (h2 : Real.sqrt (abs a) = Real.sqrt (x * y)) :
  a ∈ Set.Iic (-4) ∪ Set.Ici 4 :=
sorry

end NUMINAMATH_GPT_range_of_a_l936_93608


namespace NUMINAMATH_GPT_cost_equivalence_at_325_l936_93640

def cost_plan1 (x : ℕ) : ℝ := 65 + 0.40 * x
def cost_plan2 (x : ℕ) : ℝ := 0.60 * x

theorem cost_equivalence_at_325 : cost_plan1 325 = cost_plan2 325 :=
by sorry

end NUMINAMATH_GPT_cost_equivalence_at_325_l936_93640


namespace NUMINAMATH_GPT_low_card_value_is_one_l936_93612

-- Definitions and setting up the conditions
def num_high_cards : ℕ := 26
def num_low_cards : ℕ := 26
def high_card_points : ℕ := 2
def draw_scenarios : ℕ := 4

-- The point value of a low card L
noncomputable def low_card_points : ℕ :=
  if num_high_cards = 26 ∧ num_low_cards = 26 ∧ high_card_points = 2
     ∧ draw_scenarios = 4
  then 1 else 0 

theorem low_card_value_is_one :
  low_card_points = 1 :=
by
  sorry

end NUMINAMATH_GPT_low_card_value_is_one_l936_93612


namespace NUMINAMATH_GPT_points_and_conditions_proof_l936_93646

noncomputable def points_and_conditions (x y : ℝ) : Prop := 
|x - 3| + |y + 5| = 0

noncomputable def min_AM_BM (m : ℝ) : Prop :=
|3 - m| + |-5 - m| = 7 / 4 * |8|

noncomputable def min_PA_PB (p : ℝ) : Prop :=
|p - 3| + |p + 5| = 8

noncomputable def min_PD_PO (p : ℝ) : Prop :=
|p + 1| - |p| = -1

noncomputable def range_of_a (a : ℝ) : Prop :=
a ∈ Set.Icc (-5) (-1)

theorem points_and_conditions_proof (x y : ℝ) (m p a : ℝ) :
  points_and_conditions x y → 
  x = 3 ∧ y = -5 ∧ 
  ((m = -8 ∨ m = 6) → min_AM_BM m) ∧ 
  (min_PA_PB p) ∧ 
  (min_PD_PO p) ∧ 
  (range_of_a a) :=
by 
  sorry

end NUMINAMATH_GPT_points_and_conditions_proof_l936_93646


namespace NUMINAMATH_GPT_le_condition_l936_93602

-- Given positive numbers a, b, c
variables {a b c : ℝ}
-- Assume positive values for the numbers
variables (ha : a > 0) (hb : b > 0) (hc : c > 0)
-- Given condition a² + b² - ab = c²
axiom condition : a^2 + b^2 - a*b = c^2

-- We need to prove (a - c)(b - c) ≤ 0
theorem le_condition : (a - c) * (b - c) ≤ 0 :=
sorry

end NUMINAMATH_GPT_le_condition_l936_93602


namespace NUMINAMATH_GPT_dot_product_parallel_vectors_is_minus_ten_l936_93632

-- Definitions from the conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -4)
def are_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2) ∨ v = (k * u.1, k * u.2)

theorem dot_product_parallel_vectors_is_minus_ten (x : ℝ) (h : are_parallel vector_a (vector_b x)) : (vector_a.1 * (vector_b x).1 + vector_a.2 * (vector_b x).2) = -10 :=
by
  sorry

end NUMINAMATH_GPT_dot_product_parallel_vectors_is_minus_ten_l936_93632


namespace NUMINAMATH_GPT_block_measure_is_40_l936_93693

def jony_walks (start_time : String) (start_block end_block stop_block : ℕ) (stop_time : String) (speed : ℕ) : ℕ :=
  let total_time := 40 -- walking time in minutes
  let total_distance := speed * total_time -- total distance walked in meters
  let blocks_forward := end_block - start_block -- blocks walked forward
  let blocks_backward := end_block - stop_block -- blocks walked backward
  let total_blocks := blocks_forward + blocks_backward -- total blocks walked
  total_distance / total_blocks

theorem block_measure_is_40 :
  jony_walks "07:00" 10 90 70 "07:40" 100 = 40 := by
  sorry

end NUMINAMATH_GPT_block_measure_is_40_l936_93693


namespace NUMINAMATH_GPT_probability_not_snowing_l936_93601

theorem probability_not_snowing (p_snow : ℚ) (h : p_snow = 5 / 8) : 1 - p_snow = 3 / 8 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_probability_not_snowing_l936_93601


namespace NUMINAMATH_GPT_competition_score_l936_93695

theorem competition_score (x : ℕ) (h : x ≥ 15) : 10 * x - 5 * (20 - x) > 120 := by
  sorry

end NUMINAMATH_GPT_competition_score_l936_93695


namespace NUMINAMATH_GPT_lowest_possible_price_l936_93662

theorem lowest_possible_price
  (manufacturer_suggested_price : ℝ := 45)
  (regular_discount_percentage : ℝ := 0.30)
  (sale_discount_percentage : ℝ := 0.20)
  (regular_discounted_price : ℝ := manufacturer_suggested_price * (1 - regular_discount_percentage))
  (final_price : ℝ := regular_discounted_price * (1 - sale_discount_percentage)) :
  final_price = 25.20 :=
by sorry

end NUMINAMATH_GPT_lowest_possible_price_l936_93662


namespace NUMINAMATH_GPT_minimum_small_bottles_l936_93631

-- Define the capacities of the bottles
def small_bottle_capacity : ℕ := 35
def large_bottle_capacity : ℕ := 500

-- Define the number of small bottles needed to fill a large bottle
def small_bottles_needed_to_fill_large : ℕ := 
  (large_bottle_capacity + small_bottle_capacity - 1) / small_bottle_capacity

-- Statement of the theorem
theorem minimum_small_bottles : small_bottles_needed_to_fill_large = 15 := by
  sorry

end NUMINAMATH_GPT_minimum_small_bottles_l936_93631


namespace NUMINAMATH_GPT_non_congruent_rectangles_l936_93617

theorem non_congruent_rectangles (h w : ℕ) (hp : 2 * (h + w) = 80) :
  ∃ n, n = 20 := by
  sorry

end NUMINAMATH_GPT_non_congruent_rectangles_l936_93617


namespace NUMINAMATH_GPT_max_value_expression_l936_93683

theorem max_value_expression (x y : ℝ) : 
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 2) ≤ Real.sqrt 29 :=
by
  exact sorry

end NUMINAMATH_GPT_max_value_expression_l936_93683


namespace NUMINAMATH_GPT_olympiad_scores_l936_93697

theorem olympiad_scores (scores : Fin 20 → ℕ) 
  (uniqueScores : ∀ i j, i ≠ j → scores i ≠ scores j)
  (less_than_sum_of_others : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i, scores i > 18 := 
by sorry

end NUMINAMATH_GPT_olympiad_scores_l936_93697


namespace NUMINAMATH_GPT_jay_savings_first_week_l936_93600

theorem jay_savings_first_week :
  ∀ (x : ℕ), (x + (x + 10) + (x + 20) + (x + 30) = 60) → x = 0 :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_jay_savings_first_week_l936_93600


namespace NUMINAMATH_GPT_stop_shooting_after_2nd_scoring_5_points_eq_l936_93609

/-
Define the conditions and problem statement in Lean:
- Each person can shoot up to 10 times.
- Student A's shooting probability for each shot is 2/3.
- If student A stops shooting at the nth consecutive shot, they score 12-n points.
- We need to prove the probability that student A stops shooting right after the 2nd shot and scores 5 points is 8/729.
-/
def student_shoot_probability (shots : List Bool) (p : ℚ) : ℚ :=
  shots.foldr (λ s acc => if s then p * acc else (1 - p) * acc) 1

def stop_shooting_probability : ℚ :=
  let shots : List Bool := [false, true, false, false, false, true, true] -- represents misses and hits
  student_shoot_probability shots (2/3)

theorem stop_shooting_after_2nd_scoring_5_points_eq :
  stop_shooting_probability = (8 / 729) :=
sorry

end NUMINAMATH_GPT_stop_shooting_after_2nd_scoring_5_points_eq_l936_93609


namespace NUMINAMATH_GPT_arithmetic_sequence_S6_by_S4_l936_93680

-- Define the arithmetic sequence and the sum function
def sum_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Given conditions
def S1 : ℕ := 1
def r (S2 S4 : ℕ) : Prop := S4 / S2 = 4

-- Proof statement
theorem arithmetic_sequence_S6_by_S4 :
  ∀ (a d : ℕ), 
  (sum_arithmetic_sequence a d 1 = S1) → (r (sum_arithmetic_sequence a d 2) (sum_arithmetic_sequence a d 4)) → 
  (sum_arithmetic_sequence a d 6 / sum_arithmetic_sequence a d 4 = 9 / 4) := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_S6_by_S4_l936_93680


namespace NUMINAMATH_GPT_security_deposit_correct_l936_93629

-- Definitions (Conditions)
def daily_rate : ℝ := 125
def pet_fee_per_dog : ℝ := 100
def number_of_dogs : ℕ := 2
def tourism_tax_rate : ℝ := 0.10
def service_fee_rate : ℝ := 0.20
def activity_cost_per_person : ℝ := 45
def number_of_activities_per_person : ℕ := 3
def number_of_people : ℕ := 2
def security_deposit_rate : ℝ := 0.50
def usd_to_euro_conversion_rate : ℝ := 0.83

-- Function to calculate total cost
def total_cost_in_euros : ℝ :=
  let rental_cost := daily_rate * 14
  let pet_cost := pet_fee_per_dog * number_of_dogs
  let tourism_tax := tourism_tax_rate * rental_cost
  let service_fee := service_fee_rate * rental_cost
  let cabin_total := rental_cost + pet_cost + tourism_tax + service_fee
  let activities_total := number_of_activities_per_person * activity_cost_per_person * number_of_people
  let total_cost := cabin_total + activities_total
  let security_deposit_usd := security_deposit_rate * total_cost
  security_deposit_usd * usd_to_euro_conversion_rate

-- Theorem to prove
theorem security_deposit_correct :
  total_cost_in_euros = 1139.18 := 
sorry

end NUMINAMATH_GPT_security_deposit_correct_l936_93629


namespace NUMINAMATH_GPT_solve_system_l936_93616

-- Define the conditions of the system of equations
def condition1 (x y : ℤ) := 4 * x - 3 * y = -13
def condition2 (x y : ℤ) := 5 * x + 3 * y = -14

-- Define the proof goal using the conditions
theorem solve_system : ∃ (x y : ℤ), condition1 x y ∧ condition2 x y ∧ x = -3 ∧ y = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l936_93616


namespace NUMINAMATH_GPT_solve_absolute_value_equation_l936_93658

theorem solve_absolute_value_equation (x : ℝ) : x^2 - 3 * |x| - 4 = 0 ↔ x = 4 ∨ x = -4 :=
by
  sorry

end NUMINAMATH_GPT_solve_absolute_value_equation_l936_93658


namespace NUMINAMATH_GPT_total_amount_after_refunds_and_discounts_l936_93681

-- Definitions
def individual_bookings : ℤ := 12000
def group_bookings_before_discount : ℤ := 16000
def discount_rate : ℕ := 10
def refund_individual_1 : ℤ := 500
def count_refund_individual_1 : ℕ := 3
def refund_individual_2 : ℤ := 300
def count_refund_individual_2 : ℕ := 2
def total_refund_group : ℤ := 800

-- Calculation proofs
theorem total_amount_after_refunds_and_discounts : 
(individual_bookings + (group_bookings_before_discount - (discount_rate * group_bookings_before_discount / 100))) - 
((count_refund_individual_1 * refund_individual_1) + (count_refund_individual_2 * refund_individual_2) + total_refund_group) = 23500 := by
    sorry

end NUMINAMATH_GPT_total_amount_after_refunds_and_discounts_l936_93681


namespace NUMINAMATH_GPT_total_outfits_l936_93669

-- Define the number of shirts, pants, ties (including no-tie option), and shoes as given in the conditions.
def num_shirts : ℕ := 5
def num_pants : ℕ := 4
def num_ties : ℕ := 6 -- 5 ties + 1 no-tie option
def num_shoes : ℕ := 2

-- Proof statement: The total number of different outfits is 240.
theorem total_outfits : num_shirts * num_pants * num_ties * num_shoes = 240 :=
by
  sorry

end NUMINAMATH_GPT_total_outfits_l936_93669


namespace NUMINAMATH_GPT_parabola_translation_l936_93653

theorem parabola_translation :
  ∀(x y : ℝ), y = - (1 / 3) * (x - 5) ^ 2 + 3 →
  ∃(x' y' : ℝ), y' = -(1/3) * x'^2 + 6 := by
  sorry

end NUMINAMATH_GPT_parabola_translation_l936_93653


namespace NUMINAMATH_GPT_inequality_preservation_l936_93638

theorem inequality_preservation (x y : ℝ) (h : x < y) : 2 * x < 2 * y :=
sorry

end NUMINAMATH_GPT_inequality_preservation_l936_93638


namespace NUMINAMATH_GPT_book_width_l936_93648

noncomputable def phi_conjugate : ℝ := (Real.sqrt 5 - 1) / 2

theorem book_width {w l : ℝ} (h_ratio : w / l = phi_conjugate) (h_length : l = 14) :
  w = 7 * Real.sqrt 5 - 7 :=
by
  sorry

end NUMINAMATH_GPT_book_width_l936_93648


namespace NUMINAMATH_GPT_probability_at_least_one_woman_selected_l936_93604

open Classical

noncomputable def probability_of_selecting_at_least_one_woman : ℚ :=
  1 - (10 / 15) * (9 / 14) * (8 / 13) * (7 / 12) * (6 / 11)

theorem probability_at_least_one_woman_selected :
  probability_of_selecting_at_least_one_woman = 917 / 1001 :=
sorry

end NUMINAMATH_GPT_probability_at_least_one_woman_selected_l936_93604


namespace NUMINAMATH_GPT_reflection_matrix_determine_l936_93692

theorem reflection_matrix_determine (a b : ℚ)
  (h1 : (a^2 - (3/4) * b) = 1)
  (h2 : (-(3/4) * b + (1/16)) = 1)
  (h3 : (a * b + (1/4) * b) = 0)
  (h4 : (-(3/4) * a - (3/16)) = 0) :
  (a, b) = (1/4, -5/4) := 
sorry

end NUMINAMATH_GPT_reflection_matrix_determine_l936_93692


namespace NUMINAMATH_GPT_number_of_subcommittees_l936_93652

theorem number_of_subcommittees :
  ∃ (k : ℕ), ∀ (num_people num_sub_subcommittees subcommittee_size : ℕ), 
  num_people = 360 → 
  num_sub_subcommittees = 3 → 
  subcommittee_size = 6 → 
  k = (num_people * num_sub_subcommittees) / subcommittee_size :=
sorry

end NUMINAMATH_GPT_number_of_subcommittees_l936_93652


namespace NUMINAMATH_GPT_Meghan_total_money_l936_93603

theorem Meghan_total_money (h100 : ℕ) (h50 : ℕ) (h10 : ℕ) : 
  h100 = 2 → h50 = 5 → h10 = 10 → 100 * h100 + 50 * h50 + 10 * h10 = 550 :=
by
  sorry

end NUMINAMATH_GPT_Meghan_total_money_l936_93603


namespace NUMINAMATH_GPT_find_a_20_l936_93650

variable {a : ℕ → ℝ}
variable {r : ℝ}

-- Definitions: The sequence is geometric: a_n = a_1 * r^(n-1)
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a n = a 1 * r^(n-1)

-- Conditions in the problem: a_10 and a_30 satisfy the quadratic equation
def satisfies_quadratic_roots (a10 a30 : ℝ) : Prop :=
  a10 + a30 = 11 ∧ a10 * a30 = 16

-- Question: Find a_20
theorem find_a_20 (h1 : is_geometric_sequence a r)
                  (h2 : satisfies_quadratic_roots (a 10) (a 30)) :
  a 20 = 4 :=
sorry

end NUMINAMATH_GPT_find_a_20_l936_93650


namespace NUMINAMATH_GPT_tom_initial_balloons_l936_93674

noncomputable def initial_balloons (x : ℕ) : ℕ :=
  if h₁ : x % 2 = 1 ∧ (x / 3) + 10 = 45 then x else 0

theorem tom_initial_balloons : initial_balloons 105 = 105 :=
by {
  -- Given x is an odd number and the equation (x / 3) + 10 = 45 holds, prove x = 105.
  -- These conditions follow from the problem statement directly.
  -- Proof is skipped.
  sorry
}

end NUMINAMATH_GPT_tom_initial_balloons_l936_93674


namespace NUMINAMATH_GPT_find_cos_value_l936_93633

theorem find_cos_value (α : Real) 
  (h : Real.cos (Real.pi / 8 - α) = 1 / 6) : 
  Real.cos (3 * Real.pi / 4 + 2 * α) = 17 / 18 :=
by
  sorry

end NUMINAMATH_GPT_find_cos_value_l936_93633


namespace NUMINAMATH_GPT_evie_gave_2_shells_to_brother_l936_93613

def daily_shells : ℕ := 10
def days : ℕ := 6
def remaining_shells : ℕ := 58

def total_shells : ℕ := daily_shells * days
def shells_given : ℕ := total_shells - remaining_shells

theorem evie_gave_2_shells_to_brother :
  shells_given = 2 :=
by
  sorry

end NUMINAMATH_GPT_evie_gave_2_shells_to_brother_l936_93613


namespace NUMINAMATH_GPT_stratified_sampling_first_grade_selection_l936_93660

theorem stratified_sampling_first_grade_selection
  (total_students : ℕ)
  (students_grade1 : ℕ)
  (sample_size : ℕ)
  (h_total : total_students = 2000)
  (h_grade1 : students_grade1 = 400)
  (h_sample : sample_size = 200) :
  sample_size * students_grade1 / total_students = 40 := by
  sorry

end NUMINAMATH_GPT_stratified_sampling_first_grade_selection_l936_93660


namespace NUMINAMATH_GPT_probability_of_odd_number_l936_93699

theorem probability_of_odd_number (wedge1 wedge2 wedge3 wedge4 wedge5 : ℝ)
  (h_wedge1_split : wedge1/3 = wedge2) 
  (h_wedge2_twice_wedge1 : wedge2 = 2 * (wedge1/3))
  (h_wedge3 : wedge3 = 1/4)
  (h_wedge5 : wedge5 = 1/4)
  (h_total : wedge1/3 + wedge2 + wedge3 + wedge4 + wedge5 = 1) :
  wedge1/3 + wedge3 + wedge5 = 7 / 12 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_odd_number_l936_93699


namespace NUMINAMATH_GPT_KimSweaterTotal_l936_93647

theorem KimSweaterTotal :
  let monday := 8
  let tuesday := monday + 2
  let wednesday := tuesday - 4
  let thursday := wednesday
  let friday := monday / 2
  monday + tuesday + wednesday + thursday + friday = 34 := by
  sorry

end NUMINAMATH_GPT_KimSweaterTotal_l936_93647


namespace NUMINAMATH_GPT_find_c_l936_93649

theorem find_c (c q : ℤ) (h : ∃ (a b : ℤ), (3*x^3 + c*x + 9 = (x^2 + q*x + 1) * (a*x + b))) : c = -24 :=
sorry

end NUMINAMATH_GPT_find_c_l936_93649


namespace NUMINAMATH_GPT_number_is_nine_l936_93698

theorem number_is_nine (x : ℤ) (h : 3 * (2 * x + 9) = 81) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_number_is_nine_l936_93698


namespace NUMINAMATH_GPT_line_intersects_circle_l936_93611

theorem line_intersects_circle 
  (k : ℝ)
  (x y : ℝ)
  (h_line : x = 0 ∨ y = -2)
  (h_circle : (x - 1)^2 + (y + 2)^2 = 16) :
  (-2 - -2)^2 < 16 := by
  sorry

end NUMINAMATH_GPT_line_intersects_circle_l936_93611


namespace NUMINAMATH_GPT_find_second_number_l936_93657

def problem (a b c d : ℚ) : Prop :=
  a + b + c + d = 280 ∧
  a = 2 * b ∧
  c = 2 / 3 * a ∧
  d = b + c

theorem find_second_number (a b c d : ℚ) (h : problem a b c d) : b = 52.5 :=
by
  -- Proof will go here.
  sorry

end NUMINAMATH_GPT_find_second_number_l936_93657


namespace NUMINAMATH_GPT_find_4a_3b_l936_93678

noncomputable def g (x : ℝ) : ℝ := 4 * x - 6

noncomputable def f_inv (x : ℝ) : ℝ := g x + 2

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b

theorem find_4a_3b (a b : ℝ) (h_inv : ∀ x : ℝ, f (f_inv x) a b = x) : 4 * a + 3 * b = 4 :=
by
  -- Proof skipped for now
  sorry

end NUMINAMATH_GPT_find_4a_3b_l936_93678


namespace NUMINAMATH_GPT_distance_le_radius_l936_93672

variable (L : Line) (O : Circle)
variable (d r : ℝ)

-- Condition: Line L intersects with circle O
def intersects (L : Line) (O : Circle) : Prop := sorry -- Sketch: define what it means for a line to intersect a circle

axiom intersection_condition : intersects L O

-- Problem: Prove that if a line L intersects a circle O, then the distance d from the center of the circle to the line is less than or equal to the radius r of the circle.
theorem distance_le_radius (L : Line) (O : Circle) (d r : ℝ) :
  intersects L O → d ≤ r := by
  sorry

end NUMINAMATH_GPT_distance_le_radius_l936_93672


namespace NUMINAMATH_GPT_xy_plus_y_square_l936_93639

theorem xy_plus_y_square {x y : ℝ} (h1 : x * y = 16) (h2 : x + y = 8) : x^2 + y^2 = 32 :=
sorry

end NUMINAMATH_GPT_xy_plus_y_square_l936_93639


namespace NUMINAMATH_GPT_percentage_good_oranges_tree_A_l936_93676

theorem percentage_good_oranges_tree_A
  (total_trees : ℕ)
  (trees_A : ℕ)
  (trees_B : ℕ)
  (total_good_oranges : ℕ)
  (oranges_A_per_month : ℕ) 
  (oranges_B_per_month : ℕ)
  (good_oranges_B_ratio : ℚ)
  (good_oranges_total_B : ℕ) 
  (good_oranges_total_A : ℕ)
  (good_oranges_total : ℕ)
  (x : ℚ) 
  (total_trees_eq : total_trees = 10)
  (tree_percentage_eq : trees_A = total_trees / 2 ∧ trees_B = total_trees / 2)
  (oranges_A_per_month_eq : oranges_A_per_month = 10)
  (oranges_B_per_month_eq : oranges_B_per_month = 15)
  (good_oranges_B_ratio_eq : good_oranges_B_ratio = 1/3)
  (good_oranges_total_eq : total_good_oranges = 55)
  (good_oranges_total_B_eq : good_oranges_total_B = trees_B * oranges_B_per_month * good_oranges_B_ratio)
  (good_oranges_total_A_eq : good_oranges_total_A = total_good_oranges - good_oranges_total_B):
  trees_A * oranges_A_per_month * x = good_oranges_total_A → 
  x = 0.6 := by
  sorry

end NUMINAMATH_GPT_percentage_good_oranges_tree_A_l936_93676


namespace NUMINAMATH_GPT_cost_per_person_l936_93619

theorem cost_per_person (total_cost : ℕ) (num_people : ℕ) (h1 : total_cost = 30000) (h2 : num_people = 300) : total_cost / num_people = 100 := by
  -- No proof provided, only the theorem statement
  sorry

end NUMINAMATH_GPT_cost_per_person_l936_93619


namespace NUMINAMATH_GPT_expand_polynomial_l936_93644

theorem expand_polynomial (x : ℝ) : (x - 3) * (4 * x + 12) = 4 * x ^ 2 - 36 := 
by {
  sorry
}

end NUMINAMATH_GPT_expand_polynomial_l936_93644


namespace NUMINAMATH_GPT_cricketer_hits_two_sixes_l936_93630

-- Definitions of the given conditions
def total_runs : ℕ := 132
def boundaries_count : ℕ := 12
def running_percent : ℚ := 54.54545454545454 / 100

-- Function to calculate runs made by running
def runs_by_running (total: ℕ) (percent: ℚ) : ℚ :=
  percent * total

-- Function to calculate runs made from boundaries
def runs_from_boundaries (count: ℕ) : ℕ :=
  count * 4

-- Function to calculate runs made from sixes
def runs_from_sixes (total: ℕ) (boundaries_runs: ℕ) (running_runs: ℚ) : ℚ :=
  total - boundaries_runs - running_runs

-- Function to calculate number of sixes hit
def number_of_sixes (sixes_runs: ℚ) : ℚ :=
  sixes_runs / 6

-- The proof statement for the cricketer hitting 2 sixes
theorem cricketer_hits_two_sixes:
  number_of_sixes (runs_from_sixes total_runs (runs_from_boundaries boundaries_count) (runs_by_running total_runs running_percent)) = 2 := by
  sorry

end NUMINAMATH_GPT_cricketer_hits_two_sixes_l936_93630


namespace NUMINAMATH_GPT_georgia_carnations_proof_l936_93607

-- Define the conditions
def carnation_cost : ℝ := 0.50
def dozen_cost : ℝ := 4.00
def friends_carnations : ℕ := 14
def total_spent : ℝ := 25.00

-- Define the answer
def teachers_dozen : ℕ := 4

-- Prove the main statement
theorem georgia_carnations_proof : 
  (total_spent - (friends_carnations * carnation_cost)) / dozen_cost = teachers_dozen :=
by
  sorry

end NUMINAMATH_GPT_georgia_carnations_proof_l936_93607


namespace NUMINAMATH_GPT_leak_rate_l936_93610

-- Definitions based on conditions
def initialWater : ℕ := 10   -- 10 cups
def finalWater : ℕ := 2      -- 2 cups
def firstThreeMilesWater : ℕ := 3 * 1    -- 1 cup per mile for first 3 miles
def lastMileWater : ℕ := 3               -- 3 cups during the last mile
def hikeDuration : ℕ := 2    -- 2 hours

-- Proving the leak rate
theorem leak_rate (drunkWater : ℕ) (leakedWater : ℕ) (leakRate : ℕ) :
  drunkWater = firstThreeMilesWater + lastMileWater ∧ 
  (initialWater - finalWater) = (drunkWater + leakedWater) ∧
  hikeDuration = 2 ∧ 
  leakRate = leakedWater / hikeDuration → leakRate = 1 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_leak_rate_l936_93610


namespace NUMINAMATH_GPT_sum_of_three_numbers_l936_93688

theorem sum_of_three_numbers (x y z : ℝ) (h₁ : x + y = 29) (h₂ : y + z = 46) (h₃ : z + x = 53) : x + y + z = 64 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l936_93688


namespace NUMINAMATH_GPT_distinct_real_numbers_sum_l936_93627

theorem distinct_real_numbers_sum:
  ∀ (p q r s : ℝ),
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
    (r + s = 12 * p) →
    (r * s = -13 * q) →
    (p + q = 12 * r) →
    (p * q = -13 * s) →
    p + q + r + s = 2028 :=
by
  intros p q r s h_distinct h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_distinct_real_numbers_sum_l936_93627


namespace NUMINAMATH_GPT_mariela_cards_received_l936_93675

theorem mariela_cards_received (cards_in_hospital : ℕ) (cards_at_home : ℕ) 
  (h1 : cards_in_hospital = 403) (h2 : cards_at_home = 287) : 
  cards_in_hospital + cards_at_home = 690 := 
by 
  sorry

end NUMINAMATH_GPT_mariela_cards_received_l936_93675


namespace NUMINAMATH_GPT_a2_eq_1_l936_93614

-- Define the geometric sequence and the conditions
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

-- Given conditions
variables (a : ℕ → ℝ) (q : ℝ)
axiom a1_eq_2 : a 1 = 2
axiom condition1 : geometric_sequence a q
axiom condition2 : 16 * a 3 * a 5 = 8 * a 4 - 1

-- Prove that a_2 = 1
theorem a2_eq_1 : a 2 = 1 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_a2_eq_1_l936_93614


namespace NUMINAMATH_GPT_gary_asparagus_l936_93635

/-- Formalization of the problem -/
theorem gary_asparagus (A : ℝ) (ha : 700 * 0.50 = 350) (hg : 40 * 2.50 = 100) (hw : 630 = 3 * A + 350 + 100) : A = 60 :=
by
  sorry

end NUMINAMATH_GPT_gary_asparagus_l936_93635


namespace NUMINAMATH_GPT_min_value_x_plus_y_l936_93686

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 2) : x + y = 8 :=
sorry

end NUMINAMATH_GPT_min_value_x_plus_y_l936_93686


namespace NUMINAMATH_GPT_scarlet_savings_l936_93625

theorem scarlet_savings : 
  let initial_savings := 80
  let cost_earrings := 23
  let cost_necklace := 48
  let total_spent := cost_earrings + cost_necklace
  initial_savings - total_spent = 9 := 
by 
  sorry

end NUMINAMATH_GPT_scarlet_savings_l936_93625


namespace NUMINAMATH_GPT_division_result_l936_93656

theorem division_result : (0.284973 / 29 = 0.009827) := 
by sorry

end NUMINAMATH_GPT_division_result_l936_93656


namespace NUMINAMATH_GPT_gcd_f_x_x_l936_93667

theorem gcd_f_x_x (x : ℕ) (h : ∃ k : ℕ, x = 35622 * k) :
  Nat.gcd ((3 * x + 4) * (5 * x + 6) * (11 * x + 9) * (x + 7)) x = 378 :=
by
  sorry

end NUMINAMATH_GPT_gcd_f_x_x_l936_93667


namespace NUMINAMATH_GPT_plaster_cost_correct_l936_93673

def length : ℝ := 25
def width : ℝ := 12
def depth : ℝ := 6
def cost_per_sq_meter : ℝ := 0.30

def area_longer_walls : ℝ := 2 * (length * depth)
def area_shorter_walls : ℝ := 2 * (width * depth)
def area_bottom : ℝ := length * width
def total_area : ℝ := area_longer_walls + area_shorter_walls + area_bottom

def calculated_cost : ℝ := total_area * cost_per_sq_meter
def correct_cost : ℝ := 223.2

theorem plaster_cost_correct : calculated_cost = correct_cost := by
  sorry

end NUMINAMATH_GPT_plaster_cost_correct_l936_93673


namespace NUMINAMATH_GPT_glasses_per_pitcher_l936_93682

theorem glasses_per_pitcher (t p g : ℕ) (ht : t = 54) (hp : p = 9) : g = t / p := by
  rw [ht, hp]
  norm_num
  sorry

end NUMINAMATH_GPT_glasses_per_pitcher_l936_93682


namespace NUMINAMATH_GPT_probability_is_4_over_5_l936_93618

variable (total_balls : ℕ) (red_balls : ℕ) (purple_balls : ℕ)
variable (total_balls_eq : total_balls = 60) (red_balls_eq : red_balls = 5) (purple_balls_eq : purple_balls = 7)

def probability_neither_red_nor_purple : ℚ :=
  let favorable_outcomes := total_balls - (red_balls + purple_balls)
  let total_outcomes := total_balls
  favorable_outcomes / total_outcomes

theorem probability_is_4_over_5 :
  probability_neither_red_nor_purple total_balls red_balls purple_balls = 4 / 5 :=
by
  have h1: total_balls = 60 := total_balls_eq
  have h2: red_balls = 5 := red_balls_eq
  have h3: purple_balls = 7 := purple_balls_eq
  sorry

end NUMINAMATH_GPT_probability_is_4_over_5_l936_93618


namespace NUMINAMATH_GPT_algebraic_expression_value_l936_93689

theorem algebraic_expression_value (x y : ℝ) (h : |x - 2| + (y + 3)^2 = 0) : (x + y)^2023 = -1 := by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l936_93689


namespace NUMINAMATH_GPT_compute_five_fold_application_l936_93661

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then -x^2 else x + 10

theorem compute_five_fold_application : f (f (f (f (f 2)))) = -16 :=
by
  sorry

end NUMINAMATH_GPT_compute_five_fold_application_l936_93661


namespace NUMINAMATH_GPT_intersection_is_target_set_l936_93623

-- Define sets A and B
def is_in_A (x : ℝ) : Prop := |x - 1| < 2
def is_in_B (x : ℝ) : Prop := x^2 < 4

-- Define the intersection A ∩ B
def is_in_intersection (x : ℝ) : Prop := is_in_A x ∧ is_in_B x

-- Define the target set
def is_in_target_set (x : ℝ) : Prop := -1 < x ∧ x < 2

-- Statement to prove
theorem intersection_is_target_set : 
  ∀ x : ℝ, is_in_intersection x ↔ is_in_target_set x := sorry

end NUMINAMATH_GPT_intersection_is_target_set_l936_93623


namespace NUMINAMATH_GPT_math_problem_l936_93643

noncomputable def log_8 := Real.log 8
noncomputable def log_27 := Real.log 27
noncomputable def expr := (9 : ℝ) ^ (log_8 / log_27) + (2 : ℝ) ^ (log_27 / log_8)

theorem math_problem : expr = 7 := by
  sorry

end NUMINAMATH_GPT_math_problem_l936_93643


namespace NUMINAMATH_GPT_infinite_sum_computation_l936_93651

theorem infinite_sum_computation : 
  ∑' n : ℕ, (3 * (n + 1) + 2) / (n * (n + 1) * (n + 3)) = 10 / 3 :=
by sorry

end NUMINAMATH_GPT_infinite_sum_computation_l936_93651


namespace NUMINAMATH_GPT_fred_balloon_count_l936_93641

def sally_balloons : ℕ := 6

def fred_balloons (sally_balloons : ℕ) := 3 * sally_balloons

theorem fred_balloon_count : fred_balloons sally_balloons = 18 := by
  sorry

end NUMINAMATH_GPT_fred_balloon_count_l936_93641


namespace NUMINAMATH_GPT_fish_in_pond_l936_93668

-- Conditions
variable (N : ℕ)
variable (h₁ : 80 * 80 = 2 * N)

-- Theorem to prove 
theorem fish_in_pond (h₁ : 80 * 80 = 2 * N) : N = 3200 := 
by 
  sorry

end NUMINAMATH_GPT_fish_in_pond_l936_93668


namespace NUMINAMATH_GPT_nathan_tokens_used_is_18_l936_93655

-- We define the conditions as variables and constants
variables (airHockeyGames basketballGames tokensPerGame : ℕ)

-- State the values for the conditions
def Nathan_plays : Prop :=
  airHockeyGames = 2 ∧ basketballGames = 4 ∧ tokensPerGame = 3

-- Calculate the total tokens used
def totalTokensUsed (airHockeyGames basketballGames tokensPerGame : ℕ) : ℕ :=
  (airHockeyGames * tokensPerGame) + (basketballGames * tokensPerGame)

-- Proof statement 
theorem nathan_tokens_used_is_18 : Nathan_plays airHockeyGames basketballGames tokensPerGame → totalTokensUsed airHockeyGames basketballGames tokensPerGame = 18 :=
by 
  sorry

end NUMINAMATH_GPT_nathan_tokens_used_is_18_l936_93655


namespace NUMINAMATH_GPT_poly_sum_correct_l936_93664

def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2
def s (x : ℝ) : ℝ := -4 * x^2 + 12 * x - 12

theorem poly_sum_correct : ∀ x : ℝ, p x + q x + r x = s x :=
by
  sorry

end NUMINAMATH_GPT_poly_sum_correct_l936_93664


namespace NUMINAMATH_GPT_sequence_geometric_sequence_general_term_l936_93684

theorem sequence_geometric (a : ℕ → ℕ) (h₁ : a 1 = 2) (h₂ : ∀ n, a (n + 1) = 2 * a n + 1) :
  ∃ r : ℕ, (a 1 + 1) = 3 ∧ (∀ n, (a (n + 1) + 1) = r * (a n + 1)) := by
  sorry

theorem sequence_general_term (a : ℕ → ℕ) (h₁ : a 1 = 2) (h₂ : ∀ n, a (n + 1) = 2 * a n + 1) :
  ∀ n, a n = 3 * 2^(n-1) - 1 := by
  sorry

end NUMINAMATH_GPT_sequence_geometric_sequence_general_term_l936_93684


namespace NUMINAMATH_GPT_heads_not_consecutive_probability_l936_93665

theorem heads_not_consecutive_probability :
  (∃ n m : ℕ, n = 2^4 ∧ m = 1 + Nat.choose 4 1 + Nat.choose 3 2 ∧ (m / n : ℚ) = 1 / 2) :=
by
  use 16     -- n
  use 8      -- m
  sorry

end NUMINAMATH_GPT_heads_not_consecutive_probability_l936_93665


namespace NUMINAMATH_GPT_contractor_original_days_l936_93606

noncomputable def original_days (total_laborers absent_laborers working_laborers days_worked : ℝ) : ℝ :=
  (working_laborers * days_worked) / (total_laborers - absent_laborers)

-- Our conditions:
def total_laborers : ℝ := 21.67
def absent_laborers : ℝ := 5
def working_laborers : ℝ := 16.67
def days_worked : ℝ := 13

-- Our main theorem:
theorem contractor_original_days :
  original_days total_laborers absent_laborers working_laborers days_worked = 10 := 
by
  sorry

end NUMINAMATH_GPT_contractor_original_days_l936_93606


namespace NUMINAMATH_GPT_find_b_l936_93670

noncomputable def Q (x : ℝ) (a b c : ℝ) := 3 * x ^ 3 + a * x ^ 2 + b * x + c

theorem find_b (a b c : ℝ) (h₀ : c = 6) 
  (h₁ : ∃ (r₁ r₂ r₃ : ℝ), Q r₁ a b c = 0 ∧ Q r₂ a b c = 0 ∧ Q r₃ a b c = 0 ∧ (r₁ + r₂ + r₃) / 3 = -(c / 3) ∧ r₁ * r₂ * r₃ = -(c / 3))
  (h₂ : 3 + a + b + c = -(c / 3)): 
  b = -29 :=
sorry

end NUMINAMATH_GPT_find_b_l936_93670


namespace NUMINAMATH_GPT_find_quadruples_l936_93654

theorem find_quadruples (a b p n : ℕ) (h_prime : Prime p) (h_eq : a^3 + b^3 = p^n) :
  ∃ k : ℕ, (a, b, p, n) = (2^k, 2^k, 2, 3*k + 1) ∨ 
           (a, b, p, n) = (3^k, 2 * 3^k, 3, 3*k + 2) ∨ 
           (a, b, p, n) = (2 * 3^k, 3^k, 3, 3*k + 2) :=
sorry

end NUMINAMATH_GPT_find_quadruples_l936_93654


namespace NUMINAMATH_GPT_Jessica_biking_speed_l936_93659

theorem Jessica_biking_speed
  (swim_distance swim_speed : ℝ)
  (run_distance run_speed : ℝ)
  (bike_distance total_time : ℝ)
  (h1 : swim_distance = 0.5)
  (h2 : swim_speed = 1)
  (h3 : run_distance = 5)
  (h4 : run_speed = 5)
  (h5 : bike_distance = 20)
  (h6 : total_time = 4) :
  bike_distance / (total_time - (swim_distance / swim_speed + run_distance / run_speed)) = 8 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_Jessica_biking_speed_l936_93659


namespace NUMINAMATH_GPT_square_diff_l936_93642

theorem square_diff (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by 
  sorry

end NUMINAMATH_GPT_square_diff_l936_93642


namespace NUMINAMATH_GPT_value_of_R_l936_93628

theorem value_of_R (R : ℝ) (hR_pos : 0 < R)
  (h_line : ∀ x y : ℝ, x + y = 2 * R)
  (h_circle : ∀ x y : ℝ, (x - 1)^2 + y^2 = R) :
  R = (3 + Real.sqrt 5) / 4 ∨ R = (3 - Real.sqrt 5) / 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_R_l936_93628


namespace NUMINAMATH_GPT_min_sum_of_product_2004_l936_93624

theorem min_sum_of_product_2004 (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
    (hxyz : x * y * z = 2004) : x + y + z ≥ 174 ∧ ∃ (a b c : ℕ), a * b * c = 2004 ∧ a + b + c = 174 :=
by sorry

end NUMINAMATH_GPT_min_sum_of_product_2004_l936_93624


namespace NUMINAMATH_GPT_john_total_cost_l936_93615

theorem john_total_cost :
  let computer_cost := 1500
  let peripherals_cost := computer_cost / 4
  let base_video_card_cost := 300
  let upgraded_video_card_cost := 2.5 * base_video_card_cost
  let video_card_discount := 0.12 * upgraded_video_card_cost
  let upgraded_video_card_final_cost := upgraded_video_card_cost - video_card_discount
  let foreign_monitor_cost_local := 200
  let exchange_rate := 1.25
  let foreign_monitor_cost_usd := foreign_monitor_cost_local / exchange_rate
  let peripherals_sales_tax := 0.05 * peripherals_cost
  let subtotal := computer_cost + peripherals_cost + upgraded_video_card_final_cost + peripherals_sales_tax
  let store_loyalty_discount := 0.07 * (computer_cost + peripherals_cost + upgraded_video_card_final_cost)
  let final_cost := subtotal - store_loyalty_discount + foreign_monitor_cost_usd
  final_cost = 2536.30 := sorry

end NUMINAMATH_GPT_john_total_cost_l936_93615


namespace NUMINAMATH_GPT_diff_of_squares_l936_93645

variable {x y : ℝ}

theorem diff_of_squares : (x + y) * (x - y) = x^2 - y^2 := 
sorry

end NUMINAMATH_GPT_diff_of_squares_l936_93645


namespace NUMINAMATH_GPT_other_pencil_length_l936_93694

-- Definitions based on the conditions identified in a)
def pencil1_length : Nat := 12
def total_length : Nat := 24

-- Problem: Prove that the length of the other pencil (pencil2) is 12 cubes.
theorem other_pencil_length : total_length - pencil1_length = 12 := by 
  sorry

end NUMINAMATH_GPT_other_pencil_length_l936_93694


namespace NUMINAMATH_GPT_gcd_105_88_l936_93687

-- Define the numbers as constants
def a : ℕ := 105
def b : ℕ := 88

-- State the theorem: gcd(a, b) = 1
theorem gcd_105_88 : Nat.gcd a b = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_105_88_l936_93687


namespace NUMINAMATH_GPT_bounded_regions_l936_93663

noncomputable def regions (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => regions n + n + 1

theorem bounded_regions (n : ℕ) :
  (regions n = n * (n + 1) / 2 + 1) := by
  sorry

end NUMINAMATH_GPT_bounded_regions_l936_93663


namespace NUMINAMATH_GPT_lilith_additional_fund_l936_93666

theorem lilith_additional_fund
  (num_water_bottles : ℕ)
  (original_price : ℝ)
  (reduced_price : ℝ)
  (expected_difference : ℝ)
  (h1 : num_water_bottles = 5 * 12)
  (h2 : original_price = 2)
  (h3 : reduced_price = 1.85)
  (h4 : expected_difference = 9) :
  (num_water_bottles * original_price) - (num_water_bottles * reduced_price) = expected_difference :=
by
  sorry

end NUMINAMATH_GPT_lilith_additional_fund_l936_93666


namespace NUMINAMATH_GPT_exterior_angle_octagon_degree_l936_93637

-- Conditions
def sum_of_exterior_angles (n : ℕ) : ℕ := 360
def number_of_sides_octagon : ℕ := 8

-- Question and correct answer
theorem exterior_angle_octagon_degree :
  (sum_of_exterior_angles 8) / number_of_sides_octagon = 45 :=
by
  sorry

end NUMINAMATH_GPT_exterior_angle_octagon_degree_l936_93637


namespace NUMINAMATH_GPT_f_f_4_eq_1_l936_93622

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 x

theorem f_f_4_eq_1 : f (f 4) = 1 := by
  sorry

end NUMINAMATH_GPT_f_f_4_eq_1_l936_93622


namespace NUMINAMATH_GPT_Balaganov_made_a_mistake_l936_93685

variable (n1 n2 n3 : ℕ) (x : ℝ)
variable (average : ℝ)

def total_salary (n1 n2 : ℕ) (x : ℝ) (n3 : ℕ) : ℝ := 27 * n1 + 35 * n2 + x * n3

def number_of_employees (n1 n2 n3 : ℕ) : ℕ := n1 + n2 + n3

noncomputable def calculated_average_salary (n1 n2 : ℕ) (x : ℝ) (n3 : ℕ) : ℝ :=
 total_salary n1 n2 x n3 / number_of_employees n1 n2 n3

theorem Balaganov_made_a_mistake (h₀ : n1 > n2) 
  (h₁ : calculated_average_salary n1 n2 x n3 = average) 
  (h₂ : 31 < average) : false :=
sorry

end NUMINAMATH_GPT_Balaganov_made_a_mistake_l936_93685


namespace NUMINAMATH_GPT_combined_afternoon_burning_rate_l936_93690

theorem combined_afternoon_burning_rate 
  (morning_period_hours : ℕ)
  (afternoon_period_hours : ℕ)
  (rate_A_morning : ℕ)
  (rate_B_morning : ℕ)
  (total_morning_burn : ℕ)
  (initial_wood : ℕ)
  (remaining_wood : ℕ) :
  morning_period_hours = 4 →
  afternoon_period_hours = 4 →
  rate_A_morning = 2 →
  rate_B_morning = 1 →
  total_morning_burn = 12 →
  initial_wood = 50 →
  remaining_wood = 6 →
  ((initial_wood - remaining_wood - total_morning_burn) / afternoon_period_hours) = 8 := 
by
  intros
  -- We would continue with a proof here
  sorry

end NUMINAMATH_GPT_combined_afternoon_burning_rate_l936_93690


namespace NUMINAMATH_GPT_Fred_hourly_rate_l936_93671

-- Define the conditions
def hours_worked : ℝ := 8
def total_earned : ℝ := 100

-- Assert the proof goal
theorem Fred_hourly_rate : total_earned / hours_worked = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_Fred_hourly_rate_l936_93671
