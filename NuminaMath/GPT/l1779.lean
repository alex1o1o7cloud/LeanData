import Mathlib

namespace NUMINAMATH_GPT_partI_solution_set_partII_range_of_m_l1779_177979

def f (x m : ℝ) : ℝ := |x - m| + |x + 6|

theorem partI_solution_set (x : ℝ) :
  ∀ (x : ℝ), f x 5 ≤ 12 ↔ (-13 / 2 ≤ x ∧ x ≤ 11 / 2) :=
by
  sorry

theorem partII_range_of_m (m : ℝ) :
  (∀ x : ℝ, f x m ≥ 7) ↔ (m ≤ -13 ∨ m ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_partI_solution_set_partII_range_of_m_l1779_177979


namespace NUMINAMATH_GPT_parabola_min_value_l1779_177903

theorem parabola_min_value (x : ℝ) : (∃ x, x^2 + 10 * x + 21 = -4) := sorry

end NUMINAMATH_GPT_parabola_min_value_l1779_177903


namespace NUMINAMATH_GPT_cubic_roots_sum_cube_l1779_177977

theorem cubic_roots_sum_cube (a b c : ℂ) (h : ∀x : ℂ, (x=a ∨ x=b ∨ x=c) → (x^3 - 2*x^2 + 3*x - 4 = 0)) : a^3 + b^3 + c^3 = 2 :=
sorry

end NUMINAMATH_GPT_cubic_roots_sum_cube_l1779_177977


namespace NUMINAMATH_GPT_find_two_irreducible_fractions_l1779_177917

theorem find_two_irreducible_fractions :
  ∃ (a b d1 d2 : ℕ), 
    (1 ≤ a) ∧ 
    (1 ≤ b) ∧ 
    (gcd a d1 = 1) ∧ 
    (gcd b d2 = 1) ∧ 
    (1 ≤ d1) ∧ 
    (d1 ≤ 100) ∧ 
    (1 ≤ d2) ∧ 
    (d2 ≤ 100) ∧ 
    (a / (d1 : ℚ) + b / (d2 : ℚ) = 86 / 111) := 
by {
  sorry
}

end NUMINAMATH_GPT_find_two_irreducible_fractions_l1779_177917


namespace NUMINAMATH_GPT_maximum_possible_value_of_expression_l1779_177956

theorem maximum_possible_value_of_expression :
  ∀ (a b c d : ℕ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  (a = 0 ∨ a = 1 ∨ a = 3 ∨ a = 4) ∧
  (b = 0 ∨ b = 1 ∨ b = 3 ∨ b = 4) ∧
  (c = 0 ∨ c = 1 ∨ c = 3 ∨ c = 4) ∧
  (d = 0 ∨ d = 1 ∨ d = 3 ∨ d = 4) ∧
  ¬ (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) →
  (c * a^b + d ≤ 196) :=
by sorry

end NUMINAMATH_GPT_maximum_possible_value_of_expression_l1779_177956


namespace NUMINAMATH_GPT_probability_of_same_color_is_34_over_105_l1779_177966

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

end NUMINAMATH_GPT_probability_of_same_color_is_34_over_105_l1779_177966


namespace NUMINAMATH_GPT_intersection_complement_l1779_177916

def U : Set ℝ := Set.univ
def A : Set ℝ := {y | y ≥ 0}
def B : Set ℝ := {x | x > 3}

theorem intersection_complement :
  A ∩ (U \ B) = {x | 0 ≤ x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_GPT_intersection_complement_l1779_177916


namespace NUMINAMATH_GPT_flowers_brought_at_dawn_l1779_177973

theorem flowers_brought_at_dawn (F : ℕ) 
  (h1 : (3 / 5) * F = 180)
  (h2 :  (2 / 5) * F + (F - (3 / 5) * F) = 180) : 
  F = 300 := 
by
  sorry

end NUMINAMATH_GPT_flowers_brought_at_dawn_l1779_177973


namespace NUMINAMATH_GPT_fg_of_3_l1779_177994

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 + 1
def g (x : ℝ) : ℝ := 3 * x - 2

-- State the theorem we want to prove
theorem fg_of_3 : f (g 3) = 344 := by
  sorry

end NUMINAMATH_GPT_fg_of_3_l1779_177994


namespace NUMINAMATH_GPT_gcf_75_100_l1779_177975

theorem gcf_75_100 : Nat.gcd 75 100 = 25 := by
  -- Prime factorization for reference:
  -- 75 = 3 * 5^2
  -- 100 = 2^2 * 5^2
  sorry

end NUMINAMATH_GPT_gcf_75_100_l1779_177975


namespace NUMINAMATH_GPT_find_a8_l1779_177962

variable (a : ℕ → ℤ)

axiom h1 : ∀ n : ℕ, 2 * a n + a (n + 1) = 0
axiom h2 : a 3 = -2

theorem find_a8 : a 8 = 64 := by
  sorry

end NUMINAMATH_GPT_find_a8_l1779_177962


namespace NUMINAMATH_GPT_time_to_ascend_non_working_escalator_l1779_177976

-- Define the variables as given in the conditions
def V := 1 / 60 -- Speed of the moving escalator in units per minute
def U := (1 / 24) - (1 / 60) -- Speed of Gavrila running relative to the escalator

-- Theorem stating that the time to ascend a non-working escalator is 40 seconds
theorem time_to_ascend_non_working_escalator : 
  (1 : ℚ) = U * (40 / 60) := 
by sorry

end NUMINAMATH_GPT_time_to_ascend_non_working_escalator_l1779_177976


namespace NUMINAMATH_GPT_polygons_sides_l1779_177930

def sum_of_angles (x y : ℕ) : ℕ :=
(x - 2) * 180 + (y - 2) * 180

def num_diagonals (x y : ℕ) : ℕ :=
x * (x - 3) / 2 + y * (y - 3) / 2

theorem polygons_sides (x y : ℕ) (hx : x * (x - 3) / 2 + y * (y - 3) / 2 - (x + y) = 99) 
(hs : sum_of_angles x y = 21 * (x + y + num_diagonals x y) - 39) :
x = 17 ∧ y = 3 ∨ x = 3 ∧ y = 17 :=
by
  sorry

end NUMINAMATH_GPT_polygons_sides_l1779_177930


namespace NUMINAMATH_GPT_find_certain_number_l1779_177972

theorem find_certain_number : ∃ x : ℕ, (((x - 50) / 4) * 3 + 28 = 73) → x = 110 :=
by
  sorry

end NUMINAMATH_GPT_find_certain_number_l1779_177972


namespace NUMINAMATH_GPT_bananas_in_collection_l1779_177941

theorem bananas_in_collection
  (groups : ℕ)
  (bananas_per_group : ℕ)
  (h1 : groups = 11)
  (h2 : bananas_per_group = 37) :
  (groups * bananas_per_group) = 407 :=
by sorry

end NUMINAMATH_GPT_bananas_in_collection_l1779_177941


namespace NUMINAMATH_GPT_overlap_length_l1779_177922

noncomputable def length_of_all_red_segments := 98 -- in cm
noncomputable def total_length := 83 -- in cm
noncomputable def number_of_overlaps := 6 -- count

theorem overlap_length :
  ∃ (x : ℝ), length_of_all_red_segments - total_length = number_of_overlaps * x ∧ x = 2.5 := by
  sorry

end NUMINAMATH_GPT_overlap_length_l1779_177922


namespace NUMINAMATH_GPT_remainder_division_l1779_177902

theorem remainder_division {N : ℤ} (k : ℤ) (h : N = 125 * k + 40) : N % 15 = 10 :=
sorry

end NUMINAMATH_GPT_remainder_division_l1779_177902


namespace NUMINAMATH_GPT_lower_seat_tickets_l1779_177908

theorem lower_seat_tickets (L U : ℕ) (h1 : L + U = 80) (h2 : 30 * L + 20 * U = 2100) : L = 50 :=
by
  sorry

end NUMINAMATH_GPT_lower_seat_tickets_l1779_177908


namespace NUMINAMATH_GPT_numberOfRealSolutions_l1779_177932

theorem numberOfRealSolutions :
  ∀ (x : ℝ), (-4*x + 12)^2 + 1 = (x - 1)^2 → (∃ a b : ℝ, (a ≠ b) ∧ (-4*a + 12)^2 + 1 = (a - 1)^2 ∧ (-4*b + 12)^2 + 1 = (b - 1)^2) := by
  sorry

end NUMINAMATH_GPT_numberOfRealSolutions_l1779_177932


namespace NUMINAMATH_GPT_probability_of_B_not_losing_is_70_l1779_177906

-- Define the probabilities as given in the conditions
def prob_A_winning : ℝ := 0.30
def prob_draw : ℝ := 0.50

-- Define the probability of B not losing
def prob_B_not_losing : ℝ := 0.50 + (1 - prob_A_winning - prob_draw)

-- State the theorem
theorem probability_of_B_not_losing_is_70 :
  prob_B_not_losing = 0.70 := by
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_probability_of_B_not_losing_is_70_l1779_177906


namespace NUMINAMATH_GPT_find_m_l1779_177913

noncomputable def tangent_condition (m : ℝ) : Prop :=
  let d : ℝ := |2| / Real.sqrt (m^2 + 1)
  d = 1

theorem find_m (m : ℝ) : tangent_condition m ↔ m = Real.sqrt 3 ∨ m = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_find_m_l1779_177913


namespace NUMINAMATH_GPT_circle_tangent_line_l1779_177919

theorem circle_tangent_line (a : ℝ) : 
  ∃ (a : ℝ), a = 2 ∨ a = -8 := 
by 
  sorry

end NUMINAMATH_GPT_circle_tangent_line_l1779_177919


namespace NUMINAMATH_GPT_negation_of_universal_l1779_177940

theorem negation_of_universal :
  (¬ (∀ k : ℝ, ∃ x y : ℝ, x^2 + y^2 = 2 ∧ y = k * x + 1)) ↔ 
  (∃ k : ℝ, ¬ ∃ x y : ℝ, x^2 + y^2 = 2 ∧ y = k * x + 1) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_l1779_177940


namespace NUMINAMATH_GPT_total_spending_l1779_177921

theorem total_spending (Emma_spent : ℕ) (Elsa_spent : ℕ) (Elizabeth_spent : ℕ) : 
  Emma_spent = 58 →
  Elsa_spent = 2 * Emma_spent →
  Elizabeth_spent = 4 * Elsa_spent →
  Emma_spent + Elsa_spent + Elizabeth_spent = 638 := 
by
  intros h_Emma h_Elsa h_Elizabeth
  sorry

end NUMINAMATH_GPT_total_spending_l1779_177921


namespace NUMINAMATH_GPT_max_value_neg_domain_l1779_177987

theorem max_value_neg_domain (x : ℝ) (h : x < 0) : 
  ∃ y, y = 2 * x + 2 / x ∧ y ≤ -4 :=
sorry

end NUMINAMATH_GPT_max_value_neg_domain_l1779_177987


namespace NUMINAMATH_GPT_total_gallons_needed_l1779_177951

def gas_can_capacity : ℝ := 5.0
def number_of_cans : ℝ := 4.0
def total_gallons_of_gas : ℝ := gas_can_capacity * number_of_cans

theorem total_gallons_needed : total_gallons_of_gas = 20.0 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_gallons_needed_l1779_177951


namespace NUMINAMATH_GPT_value_of_a_l1779_177933

noncomputable def F (a : ℚ) (b : ℚ) (c : ℚ) : ℚ :=
  a * b^3 + c

theorem value_of_a :
  F a 2 3 = F a 3 4 → a = -1 / 19 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1779_177933


namespace NUMINAMATH_GPT_last_four_digits_of_5_pow_2013_l1779_177974

theorem last_four_digits_of_5_pow_2013 : (5 ^ 2013) % 10000 = 3125 :=
by
  sorry

end NUMINAMATH_GPT_last_four_digits_of_5_pow_2013_l1779_177974


namespace NUMINAMATH_GPT_muffin_to_banana_ratio_l1779_177960

variables (m b : ℝ) -- initial cost of a muffin and a banana

-- John's total cost for muffins and bananas
def johns_cost (m b : ℝ) : ℝ :=
  3 * m + 4 * b

-- Martha's total cost for muffins and bananas based on increased prices
def marthas_cost_increased (m b : ℝ) : ℝ :=
  5 * (1.2 * m) + 12 * (1.5 * b)

-- John's total cost times three
def marthas_cost_original_times_three (m b : ℝ) : ℝ :=
  3 * (johns_cost m b)

-- The theorem to prove
theorem muffin_to_banana_ratio
  (h3m4b_eq : johns_cost m b * 3 = marthas_cost_increased m b)
  (hm_eq_2b : m = 2 * b) :
  (1.2 * m) / (1.5 * b) = 4 / 5 := by
  sorry

end NUMINAMATH_GPT_muffin_to_banana_ratio_l1779_177960


namespace NUMINAMATH_GPT_calculate_expression_l1779_177946

theorem calculate_expression : (3.75 - 1.267 + 0.48 = 2.963) :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1779_177946


namespace NUMINAMATH_GPT_paper_cups_count_l1779_177911

variables (P C : ℝ) (x : ℕ)

theorem paper_cups_count :
  100 * P + x * C = 7.50 ∧ 20 * P + 40 * C = 1.50 → x = 200 :=
sorry

end NUMINAMATH_GPT_paper_cups_count_l1779_177911


namespace NUMINAMATH_GPT_min_red_beads_l1779_177934

-- Define the structure of the necklace and the conditions
structure Necklace where
  total_beads : ℕ
  blue_beads : ℕ
  red_beads : ℕ
  cyclic : Bool
  condition : ∀ (segment : List ℕ), segment.length = 8 → segment.count blue_beads ≥ 12 → segment.count red_beads ≥ 4

-- The given problem condition
def given_necklace : Necklace :=
  { total_beads := 50,
    blue_beads := 50,
    red_beads := 0,
    cyclic := true,
    condition := sorry }

-- The proof problem: Minimum number of red beads required
theorem min_red_beads (n : Necklace) : n.red_beads ≥ 29 :=
by { sorry }

end NUMINAMATH_GPT_min_red_beads_l1779_177934


namespace NUMINAMATH_GPT_remainder_n_pow_5_minus_n_mod_30_l1779_177937

theorem remainder_n_pow_5_minus_n_mod_30 (n : ℤ) : (n^5 - n) % 30 = 0 := 
by sorry

end NUMINAMATH_GPT_remainder_n_pow_5_minus_n_mod_30_l1779_177937


namespace NUMINAMATH_GPT_expected_difference_l1779_177938

def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7
def is_composite (n : ℕ) : Prop := n = 4 ∨ n = 6 ∨ n = 8

def roll_die : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def probability_eat_sweetened : ℚ := 4 / 7
def probability_eat_unsweetened : ℚ := 3 / 7
def days_in_leap_year : ℕ := 366

def expected_days_unsweetened : ℚ := probability_eat_unsweetened * days_in_leap_year
def expected_days_sweetened : ℚ := probability_eat_sweetened * days_in_leap_year

theorem expected_difference :
  expected_days_sweetened - expected_days_unsweetened = 52.28 := by
  sorry

end NUMINAMATH_GPT_expected_difference_l1779_177938


namespace NUMINAMATH_GPT_true_propositions_among_converse_inverse_contrapositive_l1779_177912

theorem true_propositions_among_converse_inverse_contrapositive
  (x : ℝ)
  (h1 : x^2 ≥ 1 → x ≥ 1) :
  (if x ≥ 1 then x^2 ≥ 1 else true) ∧ 
  (if x^2 < 1 then x < 1 else true) ∧ 
  (if x < 1 then x^2 < 1 else true) → 
  ∃ n, n = 2 :=
by sorry

end NUMINAMATH_GPT_true_propositions_among_converse_inverse_contrapositive_l1779_177912


namespace NUMINAMATH_GPT_difference_of_numbers_l1779_177924

variable (x y d : ℝ)

theorem difference_of_numbers
  (h1 : x + y = 5)
  (h2 : x - y = d)
  (h3 : x^2 - y^2 = 50) :
  d = 10 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_numbers_l1779_177924


namespace NUMINAMATH_GPT_minimum_value_inequality_l1779_177923

theorem minimum_value_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) : (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2 * a + b = 1 ∧ (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 → (2 / x + 1 / y) ≥ 9)) :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_minimum_value_inequality_l1779_177923


namespace NUMINAMATH_GPT_area_of_centroid_path_l1779_177920

theorem area_of_centroid_path (A B C O G : ℝ) (r : ℝ) (h1 : A ≠ B) 
  (h2 : 2 * r = 30) (h3 : ∀ C, C ≠ A ∧ C ≠ B ∧ dist O C = r) 
  (h4 : dist O G = r / 3) : 
  (π * (r / 3)^2 = 25 * π) :=
by 
  -- def AB := 2 * r -- given AB is a diameter of the circle
  -- def O := (A + B) / 2 -- center of the circle
  -- def G := (A + B + C) / 3 -- centroid of triangle ABC
  sorry

end NUMINAMATH_GPT_area_of_centroid_path_l1779_177920


namespace NUMINAMATH_GPT_bounds_on_xyz_l1779_177914

theorem bounds_on_xyz (a x y z : ℝ) (h1 : x + y + z = a)
                      (h2 : x^2 + y^2 + z^2 = (a^2) / 2)
                      (h3 : a > 0) (h4 : 0 < x) (h5 : 0 < y) (h6 : 0 < z) :
                      (0 < x ∧ x ≤ (2 / 3) * a) ∧ 
                      (0 < y ∧ y ≤ (2 / 3) * a) ∧ 
                      (0 < z ∧ z ≤ (2 / 3) * a) :=
sorry

end NUMINAMATH_GPT_bounds_on_xyz_l1779_177914


namespace NUMINAMATH_GPT_Cara_skate_distance_l1779_177904

-- Definitions corresponding to the conditions
def distance_CD : ℝ := 150
def speed_Cara : ℝ := 10
def speed_Dan : ℝ := 6
def angle_Cara_CD : ℝ := 45

-- main theorem based on the problem and given conditions
theorem Cara_skate_distance : ∃ t : ℝ, distance_CD = 150 ∧ speed_Cara = 10 ∧ speed_Dan = 6
                            ∧ angle_Cara_CD = 45 
                            ∧ 10 * t = 253.5 :=
by
  sorry

end NUMINAMATH_GPT_Cara_skate_distance_l1779_177904


namespace NUMINAMATH_GPT_find_two_digit_number_l1779_177964

theorem find_two_digit_number (n : ℕ) (h1 : 10 ≤ n ∧ n < 100)
  (h2 : n % 2 = 0)
  (h3 : (n + 1) % 3 = 0)
  (h4 : (n + 2) % 4 = 0)
  (h5 : (n + 3) % 5 = 0) : n = 62 :=
by
  sorry

end NUMINAMATH_GPT_find_two_digit_number_l1779_177964


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1779_177991

variable (a₁ d : ℝ)

def sum_odd := 5 * a₁ + 20 * d
def sum_even := 5 * a₁ + 25 * d

theorem arithmetic_sequence_common_difference 
  (h₁ : sum_odd a₁ d = 15) 
  (h₂ : sum_even a₁ d = 30) :
  d = 3 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1779_177991


namespace NUMINAMATH_GPT_graph_eq_pair_of_straight_lines_l1779_177997

theorem graph_eq_pair_of_straight_lines (x y : ℝ) :
  x^2 - 9*y^2 = 0 ↔ (x = 3*y ∨ x = -3*y) :=
by
  sorry

end NUMINAMATH_GPT_graph_eq_pair_of_straight_lines_l1779_177997


namespace NUMINAMATH_GPT_find_factor_l1779_177909

-- Define the conditions
def number : ℕ := 9
def expr1 (f : ℝ) : ℝ := (number + 2) * f
def expr2 : ℝ := 24 + number

-- The proof problem statement
theorem find_factor (f : ℝ) : expr1 f = expr2 → f = 3 := by
  sorry

end NUMINAMATH_GPT_find_factor_l1779_177909


namespace NUMINAMATH_GPT_rectangle_fitting_condition_l1779_177985

variables {a b c d : ℝ}

theorem rectangle_fitting_condition
  (h1: a < c ∧ c ≤ d ∧ d < b)
  (h2: a * b < c * d) :
  (b^2 - a^2)^2 ≤ (b*c - a*d)^2 + (b*d - a*c)^2 :=
sorry

end NUMINAMATH_GPT_rectangle_fitting_condition_l1779_177985


namespace NUMINAMATH_GPT_min_value_4a_plus_b_l1779_177992

theorem min_value_4a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1/a + 1/b = 1) : 4*a + b = 9 :=
sorry

end NUMINAMATH_GPT_min_value_4a_plus_b_l1779_177992


namespace NUMINAMATH_GPT_students_who_wanted_fruit_l1779_177968

theorem students_who_wanted_fruit (red_apples green_apples extra_apples ordered_apples served_apples students_wanted_fruit : ℕ)
    (h1 : red_apples = 43)
    (h2 : green_apples = 32)
    (h3 : extra_apples = 73)
    (h4 : ordered_apples = red_apples + green_apples)
    (h5 : served_apples = ordered_apples + extra_apples)
    (h6 : students_wanted_fruit = served_apples - ordered_apples) :
    students_wanted_fruit = 73 := 
by
    sorry

end NUMINAMATH_GPT_students_who_wanted_fruit_l1779_177968


namespace NUMINAMATH_GPT_difference_in_height_l1779_177983

-- Define the heights of the sandcastles
def h_J : ℚ := 3.6666666666666665
def h_S : ℚ := 2.3333333333333335

-- State the theorem
theorem difference_in_height :
  h_J - h_S = 1.333333333333333 := by
  sorry

end NUMINAMATH_GPT_difference_in_height_l1779_177983


namespace NUMINAMATH_GPT_no_valid_two_digit_factors_l1779_177953

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Main theorem to show: there are no valid two-digit factorizations of 1976
theorem no_valid_two_digit_factors : 
  ∃ (factors : ℕ → ℕ → Prop), (∀ (a b : ℕ), factors a b → (a * b = 1976) → (is_two_digit a) → (is_two_digit b)) → 
  ∃ (count : ℕ), count = 0 := 
sorry

end NUMINAMATH_GPT_no_valid_two_digit_factors_l1779_177953


namespace NUMINAMATH_GPT_count_three_digit_congruent_to_5_mod_7_l1779_177988

theorem count_three_digit_congruent_to_5_mod_7 : 
  (100 ≤ 7 * k + 5 ∧ 7 * k + 5 ≤ 999) → ∃ n : ℕ, n = 129 := sorry

end NUMINAMATH_GPT_count_three_digit_congruent_to_5_mod_7_l1779_177988


namespace NUMINAMATH_GPT_sequence_general_term_l1779_177925

theorem sequence_general_term (a : ℕ → ℕ) (n : ℕ) (h₁ : a 1 = 1) 
  (h₂ : ∀ n > 1, a n = 2 * a (n-1) + 1) : a n = 2^n - 1 :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l1779_177925


namespace NUMINAMATH_GPT_expressions_cannot_all_exceed_one_fourth_l1779_177965

theorem expressions_cannot_all_exceed_one_fourth (a b c : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) : 
  ¬ ((1 - a) * b > 1/4 ∧ (1 - b) * c > 1/4 ∧ (1 - c) * a > 1/4) := 
by
  sorry

end NUMINAMATH_GPT_expressions_cannot_all_exceed_one_fourth_l1779_177965


namespace NUMINAMATH_GPT_largest_prime_factor_l1779_177963

theorem largest_prime_factor (a b c d : ℕ) (ha : a = 20) (hb : b = 15) (hc : c = 10) (hd : d = 5) :
  ∃ p, Nat.Prime p ∧ p = 103 ∧ ∀ q, Nat.Prime q ∧ q ∣ (a^3 + b^4 - c^5 + d^6) → q ≤ p :=
by
  sorry

end NUMINAMATH_GPT_largest_prime_factor_l1779_177963


namespace NUMINAMATH_GPT_team_leader_prize_l1779_177935

theorem team_leader_prize 
    (number_of_students : ℕ := 10)
    (number_of_team_members : ℕ := 9)
    (team_member_prize : ℕ := 200)
    (additional_leader_prize : ℕ := 90)
    (total_prize : ℕ)
    (leader_prize : ℕ := total_prize - (number_of_team_members * team_member_prize))
    (average_prize : ℕ := (total_prize + additional_leader_prize) / number_of_students)
: leader_prize = 300 := 
by {
  sorry  -- Proof omitted
}

end NUMINAMATH_GPT_team_leader_prize_l1779_177935


namespace NUMINAMATH_GPT_tourist_groups_meet_l1779_177942

theorem tourist_groups_meet (x y : ℝ) (h1 : 4.5 * x + 2.5 * y = 30) (h2 : 3 * x + 5 * y = 30) : 
  x = 5 ∧ y = 3 := 
sorry

end NUMINAMATH_GPT_tourist_groups_meet_l1779_177942


namespace NUMINAMATH_GPT_doughnut_completion_time_l1779_177939

noncomputable def time_completion : Prop :=
  let start_time : ℕ := 7 * 60 -- 7:00 AM in minutes
  let quarter_complete_time : ℕ := 10 * 60 + 20 -- 10:20 AM in minutes
  let efficiency_decrease_time : ℕ := 12 * 60 -- 12:00 PM in minutes
  let one_quarter_duration : ℕ := quarter_complete_time - start_time
  let total_time_before_efficiency_decrease : ℕ := 5 * 60 -- from 7:00 AM to 12:00 PM is 5 hours
  let remaining_time_without_efficiency : ℕ := 4 * one_quarter_duration - total_time_before_efficiency_decrease
  let adjusted_remaining_time : ℕ := remaining_time_without_efficiency * 10 / 9 -- decrease by 10% efficiency
  let total_job_duration : ℕ := total_time_before_efficiency_decrease + adjusted_remaining_time
  let completion_time := efficiency_decrease_time + adjusted_remaining_time
  completion_time = 21 * 60 + 15 -- 9:15 PM in minutes

theorem doughnut_completion_time : time_completion :=
  by 
    sorry

end NUMINAMATH_GPT_doughnut_completion_time_l1779_177939


namespace NUMINAMATH_GPT_elevator_stop_time_l1779_177990

def time_to_reach_top (stories time_per_story : Nat) : Nat := stories * time_per_story

def total_time_with_stops (stories time_per_story stop_time : Nat) : Nat :=
  stories * time_per_story + (stories - 1) * stop_time

theorem elevator_stop_time (stories : Nat) (lola_time_per_story elevator_time_per_story total_elevator_time_to_top stop_time_per_floor : Nat)
  (lola_total_time : Nat) (is_slower : Bool)
  (h_lola: lola_total_time = time_to_reach_top stories lola_time_per_story)
  (h_slower: total_elevator_time_to_top = if is_slower then lola_total_time else 220)
  (h_no_stops: time_to_reach_top stories elevator_time_per_story + (stories - 1) * stop_time_per_floor = total_elevator_time_to_top) :
  stop_time_per_floor = 3 := 
  sorry

end NUMINAMATH_GPT_elevator_stop_time_l1779_177990


namespace NUMINAMATH_GPT_milk_needed_6_cookies_3_3_pints_l1779_177996

def gallon_to_quarts (g : ℚ) : ℚ := g * 4
def quarts_to_pints (q : ℚ) : ℚ := q * 2
def cookies_to_pints (p : ℚ) (c : ℚ) (n : ℚ) : ℚ := (p / c) * n
def measurement_error (p : ℚ) : ℚ := p * 1.1

theorem milk_needed_6_cookies_3_3_pints :
  (measurement_error (cookies_to_pints (quarts_to_pints (gallon_to_quarts 1.5)) 24 6) = 3.3) :=
by
  sorry

end NUMINAMATH_GPT_milk_needed_6_cookies_3_3_pints_l1779_177996


namespace NUMINAMATH_GPT_other_root_l1779_177993

theorem other_root (m : ℝ) : 
  (∀ x : ℝ, 3 * x^2 + m * x - 5 = 0 → (x = 1 ∨ x = -5 / 3)) :=
by {
  sorry
}

end NUMINAMATH_GPT_other_root_l1779_177993


namespace NUMINAMATH_GPT_debby_total_photos_l1779_177901

theorem debby_total_photos (friends_photos family_photos : ℕ) (h1 : friends_photos = 63) (h2 : family_photos = 23) : friends_photos + family_photos = 86 :=
by sorry

end NUMINAMATH_GPT_debby_total_photos_l1779_177901


namespace NUMINAMATH_GPT_greatest_QPN_value_l1779_177959

theorem greatest_QPN_value (N : ℕ) (Q P : ℕ) (QPN : ℕ) :
  (NN : ℕ) =
  10 * N + N ∧
  QPN = 100 * Q + 10 * P + N ∧
  N < 10 ∧ N ≥ 1 ∧
  NN * N = QPN ∧
  NN >= 10 ∧ NN < 100  -- Ensuring NN is a two-digit number
  → QPN <= 396 := sorry

end NUMINAMATH_GPT_greatest_QPN_value_l1779_177959


namespace NUMINAMATH_GPT_total_players_must_be_square_l1779_177999

variables (k m : ℕ)
def n : ℕ := k + m

theorem total_players_must_be_square (h: (k*(k-1) / 2) + (m*(m-1) / 2) = k * m) :
  ∃ (s : ℕ), n = s^2 :=
by sorry

end NUMINAMATH_GPT_total_players_must_be_square_l1779_177999


namespace NUMINAMATH_GPT_dismissed_cases_l1779_177978

theorem dismissed_cases (total_cases : Int) (X : Int)
  (total_cases_eq : total_cases = 17)
  (remaining_cases_eq : X = (2 * X / 3) + 1 + 4) :
  total_cases - X = 2 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_dismissed_cases_l1779_177978


namespace NUMINAMATH_GPT_correct_proposition_l1779_177982

theorem correct_proposition : 
  (¬ ∃ x_0 : ℝ, x_0^2 + 1 ≤ 2 * x_0) ↔ (∀ x : ℝ, x^2 + 1 > 2 * x) := 
sorry

end NUMINAMATH_GPT_correct_proposition_l1779_177982


namespace NUMINAMATH_GPT_not_necessarily_heavier_l1779_177958

/--
In a zoo, there are 10 elephants. It is known that if any four elephants stand on the left pan and any three on the right pan, the left pan will weigh more. If five elephants stand on the left pan and four on the right pan, the left pan does not necessarily weigh more.
-/
theorem not_necessarily_heavier (E : Fin 10 → ℝ) (H : ∀ (L : Finset (Fin 10)) (R : Finset (Fin 10)), L.card = 4 → R.card = 3 → L ≠ R → L.sum E > R.sum E) :
  ∃ (L' R' : Finset (Fin 10)), L'.card = 5 ∧ R'.card = 4 ∧ L'.sum E ≤ R'.sum E :=
by
  sorry

end NUMINAMATH_GPT_not_necessarily_heavier_l1779_177958


namespace NUMINAMATH_GPT_intersection_A_B_l1779_177929

def interval_A : Set ℝ := { x | x^2 - 3 * x - 4 < 0 }
def interval_B : Set ℝ := { x | x^2 - 4 * x + 3 > 0 }

theorem intersection_A_B :
  interval_A ∩ interval_B = { x | (-1 < x ∧ x < 1) ∨ (3 < x ∧ x < 4) } :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l1779_177929


namespace NUMINAMATH_GPT_fraction_exponentiation_l1779_177943

theorem fraction_exponentiation :
  (⟨1/3⟩ : ℝ) ^ 5 = (⟨1/243⟩ : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_fraction_exponentiation_l1779_177943


namespace NUMINAMATH_GPT_history_books_count_l1779_177949

-- Definitions based on conditions
def total_books : Nat := 100
def geography_books : Nat := 25
def math_books : Nat := 43

-- Problem statement: proving the number of history books
theorem history_books_count : total_books - geography_books - math_books = 32 := by
  sorry

end NUMINAMATH_GPT_history_books_count_l1779_177949


namespace NUMINAMATH_GPT_Seth_gave_to_his_mother_l1779_177952

variable (x : ℕ)

-- Define the conditions as per the problem statement
def initial_boxes := 9
def remaining_boxes_after_giving_to_mother := initial_boxes - x
def remaining_boxes_after_giving_half := remaining_boxes_after_giving_to_mother / 2

-- Specify the final condition
def final_boxes := 4

-- Form the main theorem
theorem Seth_gave_to_his_mother :
  final_boxes = remaining_boxes_after_giving_to_mother / 2 →
  initial_boxes - x = 8 :=
by sorry

end NUMINAMATH_GPT_Seth_gave_to_his_mother_l1779_177952


namespace NUMINAMATH_GPT_seq_an_identity_l1779_177944

theorem seq_an_identity (n : ℕ) (a : ℕ → ℕ) 
  (h₁ : a 1 = 1)
  (h₂ : ∀ n, a (n + 1) > a n)
  (h₃ : ∀ n, a (n + 1)^2 + a n^2 + 1 = 2 * (a (n + 1) * a n + a (n + 1) + a n)) 
  : a n = n^2 := sorry

end NUMINAMATH_GPT_seq_an_identity_l1779_177944


namespace NUMINAMATH_GPT_arctg_inequality_l1779_177970

theorem arctg_inequality (a b : ℝ) :
    |Real.arctan a - Real.arctan b| ≤ |b - a| := 
sorry

end NUMINAMATH_GPT_arctg_inequality_l1779_177970


namespace NUMINAMATH_GPT_problem_lean_l1779_177948

variable (α : ℝ)

-- Given condition
axiom given_cond : (1 + Real.sin α) * (1 - Real.cos α) = 1

-- Proof to be proven
theorem problem_lean : (1 - Real.sin α) * (1 + Real.cos α) = 1 - Real.sin (2 * α) := by
  sorry

end NUMINAMATH_GPT_problem_lean_l1779_177948


namespace NUMINAMATH_GPT_dryer_weight_l1779_177947

theorem dryer_weight 
(empty_truck_weight crates_soda_weight num_crates soda_weight_factor 
    fresh_produce_weight_factor num_dryers fully_loaded_truck_weight : ℕ) 

  (h1 : empty_truck_weight = 12000) 
  (h2 : crates_soda_weight = 50) 
  (h3 : num_crates = 20) 
  (h4 : soda_weight_factor = crates_soda_weight * num_crates) 
  (h5 : fresh_produce_weight_factor = 2 * soda_weight_factor) 
  (h6 : num_dryers = 3) 
  (h7 : fully_loaded_truck_weight = 24000) 

  : (fully_loaded_truck_weight - empty_truck_weight 
      - (soda_weight_factor + fresh_produce_weight_factor)) / num_dryers = 3000 := 
by sorry

end NUMINAMATH_GPT_dryer_weight_l1779_177947


namespace NUMINAMATH_GPT_bead_arrangement_probability_l1779_177907

def total_beads := 6
def red_beads := 2
def white_beads := 2
def blue_beads := 2

def total_arrangements : ℕ := Nat.factorial total_beads / (Nat.factorial red_beads * Nat.factorial white_beads * Nat.factorial blue_beads)

def valid_arrangements : ℕ := 6  -- Based on valid patterns RWBRWB, RWBWRB, and all other permutations for each starting color

def probability_valid := valid_arrangements / total_arrangements

theorem bead_arrangement_probability : probability_valid = 1 / 15 :=
  by
  -- The context and details of the solution steps are omitted as they are not included in the Lean theorem statement.
  -- This statement will skip the proof
  sorry

end NUMINAMATH_GPT_bead_arrangement_probability_l1779_177907


namespace NUMINAMATH_GPT_laura_owes_amount_l1779_177957

def principal : ℝ := 35
def rate : ℝ := 0.04
def time : ℝ := 1
def interest : ℝ := principal * rate * time
def total_amount : ℝ := principal + interest

theorem laura_owes_amount :
  total_amount = 36.40 := by
  sorry

end NUMINAMATH_GPT_laura_owes_amount_l1779_177957


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1779_177989

theorem sufficient_but_not_necessary_condition 
  (a : ℝ) 
  (h1 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x ^ 2 - a ≤ 0) : 
  a ≥ 5 :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1779_177989


namespace NUMINAMATH_GPT_factorize_expression_l1779_177927

theorem factorize_expression (a x : ℝ) :
  a * x^2 - 2 * a * x + a = a * (x - 1) ^ 2 := 
sorry

end NUMINAMATH_GPT_factorize_expression_l1779_177927


namespace NUMINAMATH_GPT_max_value_of_f_prime_div_f_l1779_177954

def f (x : ℝ) : ℝ := sorry

theorem max_value_of_f_prime_div_f (f : ℝ → ℝ) (h1 : ∀ x, deriv f x - f x = 2 * x * Real.exp x) (h2 : f 0 = 1) :
  ∀ x > 0, (deriv f x / f x) ≤ 2 :=
sorry

end NUMINAMATH_GPT_max_value_of_f_prime_div_f_l1779_177954


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1779_177918

open Set

variable {α : Type}

-- Definitions of the sets A and B
def A : Set ℤ := {-1, 0, 2, 3, 5}
def B : Set ℤ := {x | -1 < x ∧ x < 3}

-- Define the proof problem as a theorem
theorem intersection_of_A_and_B : A ∩ B = {0, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1779_177918


namespace NUMINAMATH_GPT_coffee_table_price_correct_l1779_177945

-- Conditions
def sofa_cost : ℕ := 1250
def armchair_cost_each : ℕ := 425
def num_armchairs : ℕ := 2
def total_invoice : ℕ := 2430

-- Question: What is the price of the coffee table?
def coffee_table_price : ℕ := total_invoice - (sofa_cost + num_armchairs * armchair_cost_each)

-- Proof statement (to be completed)
theorem coffee_table_price_correct : coffee_table_price = 330 := by
  sorry

end NUMINAMATH_GPT_coffee_table_price_correct_l1779_177945


namespace NUMINAMATH_GPT_x_intercept_of_line_l1779_177950

theorem x_intercept_of_line : ∃ x : ℚ, 3 * x + 5 * 0 = 20 ∧ (x, 0) = (20/3, 0) :=
by
  sorry

end NUMINAMATH_GPT_x_intercept_of_line_l1779_177950


namespace NUMINAMATH_GPT_intersection_of_sets_l1779_177900

def setA : Set ℝ := {x : ℝ | -3 ≤ x ∧ x < 4}
def setB : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}

theorem intersection_of_sets :
  setA ∩ setB = {x : ℝ | -2 ≤ x ∧ x < 4} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1779_177900


namespace NUMINAMATH_GPT_base_8_to_decimal_77_eq_63_l1779_177986

-- Define the problem in Lean 4
theorem base_8_to_decimal_77_eq_63 (k a1 a2 : ℕ) (h_k : k = 8) (h_a1 : a1 = 7) (h_a2 : a2 = 7) :
    a2 * k^1 + a1 * k^0 = 63 := 
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_base_8_to_decimal_77_eq_63_l1779_177986


namespace NUMINAMATH_GPT_tangent_line_equation_l1779_177936

-- Definitions used as conditions in the problem
def curve (x : ℝ) : ℝ := 2 * x - x^3
def point_of_tangency : ℝ × ℝ := (1, 1)

-- Lean 4 statement representing the proof problem
theorem tangent_line_equation :
  let x₀ := 1
  let y₀ := 1
  let m := deriv curve x₀
  m = -1 ∧ curve x₀ = y₀ →
  ∀ x y : ℝ, x + y - 2 = 0 → curve x₀ + m * (x - x₀) = y :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_tangent_line_equation_l1779_177936


namespace NUMINAMATH_GPT_integer_value_of_K_l1779_177931

theorem integer_value_of_K (K : ℤ) : 
  (1000 < K^4 ∧ K^4 < 5000) ∧ K > 1 → K = 6 ∨ K = 7 ∨ K = 8 :=
by sorry

end NUMINAMATH_GPT_integer_value_of_K_l1779_177931


namespace NUMINAMATH_GPT_calculate_gross_profit_l1779_177969

theorem calculate_gross_profit (sales_price : ℝ) (cost : ℝ) (gross_profit : ℝ) 
    (h1 : sales_price = 81)
    (h2 : gross_profit = 1.70 * cost)
    (h3 : sales_price = cost + gross_profit) : gross_profit = 51 :=
by
  sorry

end NUMINAMATH_GPT_calculate_gross_profit_l1779_177969


namespace NUMINAMATH_GPT_find_days_jane_indisposed_l1779_177926

-- Define the problem conditions
def John_rate := 1 / 20
def Jane_rate := 1 / 10
def together_rate := John_rate + Jane_rate
def total_task := 1
def total_days := 10

-- The time Jane was indisposed
def days_jane_indisposed (x : ℝ) : Prop :=
  (total_days - x) * together_rate + x * John_rate = total_task

-- Statement we want to prove
theorem find_days_jane_indisposed : ∃ x : ℝ, days_jane_indisposed x ∧ x = 5 :=
by 
  sorry

end NUMINAMATH_GPT_find_days_jane_indisposed_l1779_177926


namespace NUMINAMATH_GPT_binary_add_sub_l1779_177998

theorem binary_add_sub:
  let a := 0b10110
  let b := 0b1010
  let c := 0b11100
  let d := 0b1110
  a + b - c + d = 0b01110 := by
  sorry

end NUMINAMATH_GPT_binary_add_sub_l1779_177998


namespace NUMINAMATH_GPT_car_speed_l1779_177995

theorem car_speed (v t Δt : ℝ) (h1: 90 = v * t) (h2: 90 = (v + 30) * (t - Δt)) (h3: Δt = 0.5) : 
  ∃ v, 90 = v * t ∧ 90 = (v + 30) * (t - Δt) :=
by {
  sorry
}

end NUMINAMATH_GPT_car_speed_l1779_177995


namespace NUMINAMATH_GPT_Randy_bats_l1779_177971

theorem Randy_bats (bats gloves : ℕ) (h1 : gloves = 7 * bats + 1) (h2 : gloves = 29) : bats = 4 :=
by
  sorry

end NUMINAMATH_GPT_Randy_bats_l1779_177971


namespace NUMINAMATH_GPT_convert_3652_from_base7_to_base10_l1779_177910

def base7ToBase10(n : ℕ) := 
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  let d3 := (n / 1000) % 10
  d0 * (7^0) + d1 * (7^1) + d2 * (7^2) + d3 * (7^3)

theorem convert_3652_from_base7_to_base10 : base7ToBase10 3652 = 1360 :=
by
  sorry

end NUMINAMATH_GPT_convert_3652_from_base7_to_base10_l1779_177910


namespace NUMINAMATH_GPT_fraction_difference_l1779_177955

-- Definitions for the problem conditions
def repeatingDecimal72 := 8 / 11
def decimal72 := 18 / 25

-- Statement that needs to be proven
theorem fraction_difference : repeatingDecimal72 - decimal72 = 2 / 275 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_difference_l1779_177955


namespace NUMINAMATH_GPT_emmalyn_earnings_l1779_177961

theorem emmalyn_earnings
  (rate_per_meter : ℚ := 0.20)
  (number_of_fences : ℚ := 50)
  (length_per_fence : ℚ := 500) :
  rate_per_meter * (number_of_fences * length_per_fence) = 5000 := by
  sorry

end NUMINAMATH_GPT_emmalyn_earnings_l1779_177961


namespace NUMINAMATH_GPT_percent_decrease_computer_price_l1779_177967

theorem percent_decrease_computer_price (price_1990 price_2010 : ℝ) (h1 : price_1990 = 1200) (h2 : price_2010 = 600) :
  ((price_1990 - price_2010) / price_1990) * 100 = 50 := 
  sorry

end NUMINAMATH_GPT_percent_decrease_computer_price_l1779_177967


namespace NUMINAMATH_GPT_bus_initial_passengers_l1779_177915

theorem bus_initial_passengers (M W : ℕ) 
  (h1 : W = M / 2) 
  (h2 : M - 16 = W + 8) : 
  M + W = 72 :=
sorry

end NUMINAMATH_GPT_bus_initial_passengers_l1779_177915


namespace NUMINAMATH_GPT_greatest_temp_diff_on_tuesday_l1779_177928

def highest_temp_mon : ℝ := 5
def lowest_temp_mon : ℝ := 2
def highest_temp_tue : ℝ := 4
def lowest_temp_tue : ℝ := -1
def highest_temp_wed : ℝ := 0
def lowest_temp_wed : ℝ := -4

def temp_diff (highest lowest : ℝ) : ℝ :=
  highest - lowest

theorem greatest_temp_diff_on_tuesday : temp_diff highest_temp_tue lowest_temp_tue 
  > temp_diff highest_temp_mon lowest_temp_mon 
  ∧ temp_diff highest_temp_tue lowest_temp_tue 
  > temp_diff highest_temp_wed lowest_temp_wed := 
by
  sorry

end NUMINAMATH_GPT_greatest_temp_diff_on_tuesday_l1779_177928


namespace NUMINAMATH_GPT_power_equality_l1779_177905

theorem power_equality (x : ℕ) (h : (1 / 8) * (2^40) = 2^x) : x = 37 := by
  sorry

end NUMINAMATH_GPT_power_equality_l1779_177905


namespace NUMINAMATH_GPT_most_compliant_expression_l1779_177980

-- Define the expressions as algebraic terms.
def OptionA : String := "1(1/2)a"
def OptionB : String := "b/a"
def OptionC : String := "3a-1 个"
def OptionD : String := "a * 3"

-- Define a property that represents compliance with standard algebraic notation.
def is_compliant (expr : String) : Prop :=
  expr = OptionB

-- The theorem to prove.
theorem most_compliant_expression :
  is_compliant OptionB :=
by
  sorry

end NUMINAMATH_GPT_most_compliant_expression_l1779_177980


namespace NUMINAMATH_GPT_pencil_cost_l1779_177984

theorem pencil_cost (total_money : ℕ) (num_pencils : ℕ) (h1 : total_money = 50) (h2 : num_pencils = 10) :
    (total_money / num_pencils) = 5 :=
by
  sorry

end NUMINAMATH_GPT_pencil_cost_l1779_177984


namespace NUMINAMATH_GPT_slope_of_line_l1779_177981

theorem slope_of_line : 
  (∀ x y : ℝ, (y = (1/2) * x + 1) → ∃ m : ℝ, m = 1/2) :=
sorry

end NUMINAMATH_GPT_slope_of_line_l1779_177981
