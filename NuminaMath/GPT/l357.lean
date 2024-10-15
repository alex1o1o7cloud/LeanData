import Mathlib

namespace NUMINAMATH_GPT_coin_count_l357_35709

theorem coin_count (x y : ℕ) 
  (h1 : x + y = 12) 
  (h2 : 5 * x + 10 * y = 90) :
  x = 6 ∧ y = 6 := 
sorry

end NUMINAMATH_GPT_coin_count_l357_35709


namespace NUMINAMATH_GPT_claudia_coins_l357_35797

theorem claudia_coins (x y : ℕ) (h1 : x + y = 15) (h2 : 29 - x = 26) :
  y = 12 :=
by
  sorry

end NUMINAMATH_GPT_claudia_coins_l357_35797


namespace NUMINAMATH_GPT_fraction_increase_by_five_l357_35762

variable (x y : ℝ)

theorem fraction_increase_by_five :
  let f := fun x y => (x * y) / (2 * x - 3 * y)
  f (5 * x) (5 * y) = 5 * (f x y) :=
by
  sorry

end NUMINAMATH_GPT_fraction_increase_by_five_l357_35762


namespace NUMINAMATH_GPT_no_integer_solutions_l357_35755

theorem no_integer_solutions :
  ∀ x y : ℤ, x^3 + 4 * x^2 - 11 * x + 30 ≠ 8 * y^3 + 24 * y^2 + 18 * y + 7 :=
by sorry

end NUMINAMATH_GPT_no_integer_solutions_l357_35755


namespace NUMINAMATH_GPT_constant_term_in_expansion_l357_35771

-- Given conditions
def eq_half_n_minus_m_zero (n m : ℕ) : Prop := 1/2 * n = m
def eq_n_plus_m_ten (n m : ℕ) : Prop := n + m = 10
noncomputable def binom (n k : ℕ) : ℝ := Real.exp (Real.log (Nat.factorial n) - Real.log (Nat.factorial k) - Real.log (Nat.factorial (n - k)))

-- Main theorem
theorem constant_term_in_expansion : 
  ∃ (n m : ℕ), eq_half_n_minus_m_zero n m ∧ eq_n_plus_m_ten n m ∧ 
  binom 10 m * (3^4 : ℝ) = 17010 :=
by
  -- Definitions translation
  sorry

end NUMINAMATH_GPT_constant_term_in_expansion_l357_35771


namespace NUMINAMATH_GPT_simplify_polynomials_l357_35775

theorem simplify_polynomials :
  (3 * x^3 + 4 * x^2 + 6 * x - 5) - (2 * x^3 + 2 * x^2 + 3 * x - 8) = x^3 + 2 * x^2 + 3 * x + 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomials_l357_35775


namespace NUMINAMATH_GPT_water_needed_to_fill_glasses_l357_35747

theorem water_needed_to_fill_glasses :
  let glasses := 10
  let capacity_per_glass := 6
  let filled_fraction := 4 / 5
  let total_capacity := glasses * capacity_per_glass
  let total_water := glasses * (capacity_per_glass * filled_fraction)
  let water_needed := total_capacity - total_water
  water_needed = 12 :=
by
  sorry

end NUMINAMATH_GPT_water_needed_to_fill_glasses_l357_35747


namespace NUMINAMATH_GPT_value_of_a_l357_35738

theorem value_of_a (a : ℝ) (h1 : a < 0) (h2 : |a| = 3) : a = -3 := 
by
  sorry

end NUMINAMATH_GPT_value_of_a_l357_35738


namespace NUMINAMATH_GPT_pyarelal_loss_l357_35767

/-
Problem statement:
Given the following conditions:
1. Ashok's capital is 1/9 of Pyarelal's.
2. Ashok experienced a loss of 12% on his investment.
3. Pyarelal's loss was 9% of his investment.
4. Their total combined loss is Rs. 2,100.

Prove that the loss incurred by Pyarelal is Rs. 1,829.32.
-/

theorem pyarelal_loss (P : ℝ) (total_loss : ℝ) (ashok_ratio : ℝ) (ashok_loss_percent : ℝ) (pyarelal_loss_percent : ℝ)
  (h1 : ashok_ratio = (1 : ℝ) / 9)
  (h2 : ashok_loss_percent = 0.12)
  (h3 : pyarelal_loss_percent = 0.09)
  (h4 : total_loss = 2100)
  (h5 : total_loss = ashok_loss_percent * (P * ashok_ratio) + pyarelal_loss_percent * P) :
  pyarelal_loss_percent * P = 1829.32 :=
by
  sorry

end NUMINAMATH_GPT_pyarelal_loss_l357_35767


namespace NUMINAMATH_GPT_solve_abs_inequality_l357_35784

theorem solve_abs_inequality (x : ℝ) :
  (|x - 2| + |x - 4| > 6) ↔ (x < 0 ∨ 12 < x) :=
by
  sorry

end NUMINAMATH_GPT_solve_abs_inequality_l357_35784


namespace NUMINAMATH_GPT_jennifer_dogs_l357_35759

theorem jennifer_dogs (D : ℕ) (groom_time_per_dog : ℕ) (groom_days : ℕ) (total_groom_time : ℕ) :
  groom_time_per_dog = 20 →
  groom_days = 30 →
  total_groom_time = 1200 →
  groom_days * (groom_time_per_dog * D) = total_groom_time →
  D = 2 :=
by
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_jennifer_dogs_l357_35759


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l357_35736

theorem geometric_sequence_ratio (a : ℕ → ℤ) (q : ℤ) (n : ℕ) (i : ℕ → ℕ) (ε : ℕ → ℤ) :
  (∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → a k = a 1 * q ^ (k - 1)) ∧
  (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n → ε k * a (i k) = 0) ∧
  (∀ m, 1 ≤ i m ∧ i m ≤ n) → q = -1 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l357_35736


namespace NUMINAMATH_GPT_unique_arrangements_MOON_l357_35737

theorem unique_arrangements_MOON : 
  let M := 1
  let O := 2
  let N := 1
  let total_letters := 4
  (Nat.factorial total_letters / (Nat.factorial O)) = 12 :=
by
  sorry

end NUMINAMATH_GPT_unique_arrangements_MOON_l357_35737


namespace NUMINAMATH_GPT_number_of_students_l357_35740

theorem number_of_students (n : ℕ) (h1 : n < 40) (h2 : n % 7 = 3) (h3 : n % 6 = 1) : n = 31 := 
by
  sorry

end NUMINAMATH_GPT_number_of_students_l357_35740


namespace NUMINAMATH_GPT_sum_of_M_l357_35700

theorem sum_of_M (x y z w M : ℕ) (hxw : w = x + y + z) (hM : M = x * y * z * w) (hM_cond : M = 12 * (x + y + z + w)) :
  ∃ sum_M, sum_M = 2208 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_M_l357_35700


namespace NUMINAMATH_GPT_arithmetic_sequence_ratio_l357_35745

theorem arithmetic_sequence_ratio
  (d : ℕ) (h₀ : d ≠ 0)
  (a : ℕ → ℕ)
  (h₁ : ∀ n, a (n + 1) = a n + d)
  (h₂ : (a 3)^2 = (a 1) * (a 9)) :
  (a 1 + a 3 + a 6) / (a 2 + a 4 + a 10) = 5 / 8 :=
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_ratio_l357_35745


namespace NUMINAMATH_GPT_min_stool_height_l357_35796

/-
Alice needs to reach a ceiling fan switch located 15 centimeters below a 3-meter-tall ceiling.
Alice is 160 centimeters tall and can reach 50 centimeters above her head. She uses a stack of books
12 centimeters tall to assist her reach. We aim to show that the minimum height of the stool she needs is 63 centimeters.
-/

def ceiling_height_cm : ℕ := 300
def alice_height_cm : ℕ := 160
def reach_above_head_cm : ℕ := 50
def books_height_cm : ℕ := 12
def switch_below_ceiling_cm : ℕ := 15

def total_reach_with_books := alice_height_cm + reach_above_head_cm + books_height_cm
def switch_height_from_floor := ceiling_height_cm - switch_below_ceiling_cm

theorem min_stool_height : total_reach_with_books + 63 = switch_height_from_floor := by
  unfold total_reach_with_books switch_height_from_floor
  sorry

end NUMINAMATH_GPT_min_stool_height_l357_35796


namespace NUMINAMATH_GPT_triangle_is_isosceles_l357_35701

theorem triangle_is_isosceles 
  (A B C : ℝ)
  (h : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (h_triangle : A + B + C = π)
  (h_condition : (Real.sin B) * (Real.sin C) = (Real.cos (A / 2)) ^ 2) :
  (B = C) :=
sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l357_35701


namespace NUMINAMATH_GPT_yellow_yellow_pairs_l357_35776

variable (students_total : ℕ := 150)
variable (blue_students : ℕ := 65)
variable (yellow_students : ℕ := 85)
variable (total_pairs : ℕ := 75)
variable (blue_blue_pairs : ℕ := 30)

theorem yellow_yellow_pairs : 
  (yellow_students - (blue_students - blue_blue_pairs * 2)) / 2 = 40 :=
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_yellow_yellow_pairs_l357_35776


namespace NUMINAMATH_GPT_unique_ordered_triple_lcm_l357_35722

theorem unique_ordered_triple_lcm:
  ∃! (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c), 
    Nat.lcm a b = 2100 ∧ Nat.lcm b c = 3150 ∧ Nat.lcm c a = 4200 :=
by
  sorry

end NUMINAMATH_GPT_unique_ordered_triple_lcm_l357_35722


namespace NUMINAMATH_GPT_sum_digit_product_1001_to_2011_l357_35743

def digit_product (n : ℕ) : ℕ :=
  (n.digits 10).foldr (λ d acc => d * acc) 1

theorem sum_digit_product_1001_to_2011 :
  (Finset.range 1011).sum (λ k => digit_product (1001 + k)) = 91125 :=
by
  sorry

end NUMINAMATH_GPT_sum_digit_product_1001_to_2011_l357_35743


namespace NUMINAMATH_GPT_remainder_of_n_div_11_is_1_l357_35752

def A : ℕ := 20072009
def n : ℕ := 100 * A

theorem remainder_of_n_div_11_is_1 :
  (n % 11) = 1 :=
sorry

end NUMINAMATH_GPT_remainder_of_n_div_11_is_1_l357_35752


namespace NUMINAMATH_GPT_find_number_l357_35764

theorem find_number (x : ℝ) (h : x / 5 = 70 + x / 6) : x = 2100 := by
  sorry

end NUMINAMATH_GPT_find_number_l357_35764


namespace NUMINAMATH_GPT_jangshe_clothing_cost_l357_35717

theorem jangshe_clothing_cost
  (total_spent : ℝ)
  (untaxed_piece1 : ℝ)
  (untaxed_piece2 : ℝ)
  (total_pieces : ℕ)
  (remaining_pieces : ℕ)
  (remaining_pieces_price : ℝ)
  (sales_tax : ℝ)
  (price_multiple_of_five : ℝ) :
  total_spent = 610 ∧
  untaxed_piece1 = 49 ∧
  untaxed_piece2 = 81 ∧
  total_pieces = 7 ∧
  remaining_pieces = 5 ∧
  sales_tax = 0.10 ∧
  (∃ k : ℕ, remaining_pieces_price = k * 5) →
  remaining_pieces_price / remaining_pieces = 87 :=
by
  sorry

end NUMINAMATH_GPT_jangshe_clothing_cost_l357_35717


namespace NUMINAMATH_GPT_find_volume_of_sphere_l357_35703

noncomputable def volume_of_sphere (AB BC AA1 : ℝ) (hAB : AB = 2) (hBC : BC = 2) (hAA1 : AA1 = 2 * Real.sqrt 2) : ℝ :=
  let diagonal := Real.sqrt (AB^2 + BC^2 + AA1^2)
  let radius := diagonal / 2
  (4 * Real.pi * radius^3) / 3

theorem find_volume_of_sphere : volume_of_sphere 2 2 (2 * Real.sqrt 2) (by rfl) (by rfl) (by rfl) = (32 * Real.pi) / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_volume_of_sphere_l357_35703


namespace NUMINAMATH_GPT_zero_of_sum_of_squares_eq_zero_l357_35725

theorem zero_of_sum_of_squares_eq_zero (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
by
  sorry

end NUMINAMATH_GPT_zero_of_sum_of_squares_eq_zero_l357_35725


namespace NUMINAMATH_GPT_initial_weight_of_cheese_l357_35772

theorem initial_weight_of_cheese :
  let initial_weight : Nat := 850
  -- final state after 3 bites
  let final_weight1 : Nat := 25
  let final_weight2 : Nat := 25
  -- third state
  let third_weight1 : Nat := final_weight1 + final_weight2
  let third_weight2 : Nat := final_weight1
  -- second state
  let second_weight1 : Nat := third_weight1 + third_weight2
  let second_weight2 : Nat := third_weight1
  -- first state
  let first_weight1 : Nat := second_weight1 + second_weight2
  let first_weight2 : Nat := second_weight1
  -- initial state
  let initial_weight1 : Nat := first_weight1 + first_weight2
  let initial_weight2 : Nat := first_weight1
  initial_weight = initial_weight1 + initial_weight2 :=
by
  sorry

end NUMINAMATH_GPT_initial_weight_of_cheese_l357_35772


namespace NUMINAMATH_GPT_problem_lean_version_l357_35720

theorem problem_lean_version (n : ℕ) : 
  (n > 0) ∧ (6^n - 1 ∣ 7^n - 1) ↔ ∃ k : ℕ, n = 4 * k :=
by
  sorry

end NUMINAMATH_GPT_problem_lean_version_l357_35720


namespace NUMINAMATH_GPT_unique_real_solution_k_eq_35_over_4_l357_35749

theorem unique_real_solution_k_eq_35_over_4 :
  ∃ k : ℚ, (∀ x : ℝ, (x + 5) * (x + 3) = k + 3 * x) ↔ (k = 35 / 4) :=
by
  sorry

end NUMINAMATH_GPT_unique_real_solution_k_eq_35_over_4_l357_35749


namespace NUMINAMATH_GPT_area_of_equilateral_triangle_with_inscribed_circle_l357_35734

theorem area_of_equilateral_triangle_with_inscribed_circle 
  (r : ℝ) (A : ℝ) (area_circle_eq : A = 9 * Real.pi)
  (DEF_equilateral : ∀ {a b c : ℝ}, a = b ∧ b = c): 
  ∃ area_def : ℝ, area_def = 27 * Real.sqrt 3 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_area_of_equilateral_triangle_with_inscribed_circle_l357_35734


namespace NUMINAMATH_GPT_card_selection_ways_l357_35724

theorem card_selection_ways (deck_size : ℕ) (suits : ℕ) (cards_per_suit : ℕ) (total_cards_chosen : ℕ)
  (repeated_suit_count : ℕ) (distinct_suits_count : ℕ) (distinct_ranks_count : ℕ) 
  (correct_answer : ℕ) :
  deck_size = 52 ∧ suits = 4 ∧ cards_per_suit = 13 ∧ total_cards_chosen = 5 ∧ 
  repeated_suit_count = 2 ∧ distinct_suits_count = 3 ∧ distinct_ranks_count = 11 ∧ 
  correct_answer = 414384 :=
by 
  -- Sorry is used to skip actual proof steps, according to the instructions.
  sorry

end NUMINAMATH_GPT_card_selection_ways_l357_35724


namespace NUMINAMATH_GPT_find_values_of_a_b_l357_35706

variable (a b : ℤ)

def A : Set ℤ := {1, b, a + b}
def B : Set ℤ := {a - b, a * b}
def common_set : Set ℤ := {-1, 0}

theorem find_values_of_a_b (h : A a b ∩ B a b = common_set) : (a, b) = (-1, 0) := by
  sorry

end NUMINAMATH_GPT_find_values_of_a_b_l357_35706


namespace NUMINAMATH_GPT_leo_third_part_time_l357_35760

-- Definitions to represent the conditions
def total_time : ℕ := 120
def first_part_time : ℕ := 25
def second_part_time : ℕ := 2 * first_part_time

-- Proposition to prove
theorem leo_third_part_time :
  total_time - (first_part_time + second_part_time) = 45 :=
by
  sorry

end NUMINAMATH_GPT_leo_third_part_time_l357_35760


namespace NUMINAMATH_GPT_a6_value_l357_35778

theorem a6_value
  (a : ℕ → ℤ)
  (h1 : a 2 = 3)
  (h2 : a 4 = 15)
  (geo : ∃ q : ℤ, ∀ n : ℕ, n > 0 → a (n + 1) = q^n * (a 1 + 1) - 1):
  a 6 = 63 :=
by
  sorry

end NUMINAMATH_GPT_a6_value_l357_35778


namespace NUMINAMATH_GPT_probability_of_sum_leq_10_l357_35712

open Nat

-- Define the three dice roll outcomes
def dice_outcomes := {n : ℕ | 1 ≤ n ∧ n ≤ 6}

-- Define the total number of outcomes when rolling three dice
def total_outcomes : ℕ := 6 ^ 3

-- Count the number of valid outcomes where the sum of three dice is less than or equal to 10
def count_valid_outcomes : ℕ := 75  -- This is determined through combinatorial calculations or software

-- Define the desired probability
def desired_probability := (count_valid_outcomes : ℚ) / total_outcomes

-- Prove that the desired probability equals 25/72
theorem probability_of_sum_leq_10 :
  desired_probability = 25 / 72 :=
by sorry

end NUMINAMATH_GPT_probability_of_sum_leq_10_l357_35712


namespace NUMINAMATH_GPT_initial_bacteria_count_l357_35777

theorem initial_bacteria_count (doubling_interval : ℕ) (initial_count four_minutes_final_count : ℕ)
  (h1 : doubling_interval = 30)
  (h2 : four_minutes_final_count = 524288)
  (h3 : ∀ t : ℕ, initial_count * 2 ^ (t / doubling_interval) = four_minutes_final_count) :
  initial_count = 2048 :=
sorry

end NUMINAMATH_GPT_initial_bacteria_count_l357_35777


namespace NUMINAMATH_GPT_identify_stolen_bag_with_two_weighings_l357_35758

-- Definition of the weights of the nine bags
def weights : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Statement of the problem: Using two weighings on a balance scale without weights,
-- prove that it is possible to identify the specific bag from which the treasure was stolen.
theorem identify_stolen_bag_with_two_weighings (stolen_bag : {n // n < 9}) :
  ∃ (group1 group2 : List ℕ), group1 ≠ group2 ∧ (group1.sum = 11 ∨ group1.sum = 15) ∧ (group2.sum = 11 ∨ group2.sum = 15) →
  ∃ (b1 b2 b3 : ℕ), b1 ≠ b2 ∧ b1 ≠ b3 ∧ b2 ≠ b3 ∧ b1 + b2 + b3 = 6 ∧ (b1 + b2 = 11 ∨ b1 + b2 = 15) := sorry

end NUMINAMATH_GPT_identify_stolen_bag_with_two_weighings_l357_35758


namespace NUMINAMATH_GPT_rounding_effect_l357_35727

/-- Given positive integers x, y, and z, and rounding scenarios, the
  approximation of x/y - z is necessarily less than its exact value
  when z is rounded up and x and y are rounded down. -/
theorem rounding_effect (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
(RoundXDown RoundYDown RoundZUp : ℕ → ℕ) 
(HRoundXDown : ∀ a, RoundXDown a ≤ a)
(HRoundYDown : ∀ a, RoundYDown a ≤ a)
(HRoundZUp : ∀ a, a ≤ RoundZUp a) :
  (RoundXDown x) / (RoundYDown y) - (RoundZUp z) < x / y - z :=
sorry

end NUMINAMATH_GPT_rounding_effect_l357_35727


namespace NUMINAMATH_GPT_kitten_length_doubling_l357_35769

theorem kitten_length_doubling (initial_length : ℕ) (week2_length : ℕ) (current_length : ℕ) 
  (h1 : initial_length = 4) 
  (h2 : week2_length = 2 * initial_length) 
  (h3 : current_length = 2 * week2_length) : 
    current_length = 16 := 
by 
  sorry

end NUMINAMATH_GPT_kitten_length_doubling_l357_35769


namespace NUMINAMATH_GPT_mary_travel_time_l357_35715

noncomputable def ambulance_speed : ℝ := 60
noncomputable def don_speed : ℝ := 30
noncomputable def don_time : ℝ := 0.5

theorem mary_travel_time : (don_speed * don_time) / ambulance_speed * 60 = 15 := by
  sorry

end NUMINAMATH_GPT_mary_travel_time_l357_35715


namespace NUMINAMATH_GPT_semicircle_inequality_l357_35730

open Real

theorem semicircle_inequality {A B C D E : ℝ} (h : A^2 + B^2 + C^2 + D^2 + E^2 = 1):
  (A - B)^2 + (B - C)^2 + (C - D)^2 + (D - E)^2 + (A - B) * (B - C) * (C - D) + (B - C) * (C - D) * (D - E) < 4 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_semicircle_inequality_l357_35730


namespace NUMINAMATH_GPT_archer_scores_distribution_l357_35795

structure ArcherScores where
  hits_40 : ℕ
  hits_39 : ℕ
  hits_24 : ℕ
  hits_23 : ℕ
  hits_17 : ℕ
  hits_16 : ℕ
  total_score : ℕ

theorem archer_scores_distribution
  (dora : ArcherScores)
  (reggie : ArcherScores)
  (finch : ArcherScores)
  (h1 : dora.total_score = 120)
  (h2 : reggie.total_score = 110)
  (h3 : finch.total_score = 100)
  (h4 : dora.hits_40 + dora.hits_39 + dora.hits_24 + dora.hits_23 + dora.hits_17 + dora.hits_16 = 6)
  (h5 : reggie.hits_40 + reggie.hits_39 + reggie.hits_24 + reggie.hits_23 + reggie.hits_17 + reggie.hits_16 = 6)
  (h6 : finch.hits_40 + finch.hits_39 + finch.hits_24 + finch.hits_23 + finch.hits_17 + finch.hits_16 = 6)
  (h7 : 40 * dora.hits_40 + 39 * dora.hits_39 + 24 * dora.hits_24 + 23 * dora.hits_23 + 17 * dora.hits_17 + 16 * dora.hits_16 = 120)
  (h8 : 40 * reggie.hits_40 + 39 * reggie.hits_39 + 24 * reggie.hits_24 + 23 * reggie.hits_23 + 17 * reggie.hits_17 + 16 * reggie.hits_16 = 110)
  (h9 : 40 * finch.hits_40 + 39 * finch.hits_39 + 24 * finch.hits_24 + 23 * finch.hits_23 + 17 * finch.hits_17 + 16 * finch.hits_16 = 100)
  (h10 : dora.hits_40 = 1)
  (h11 : dora.hits_39 = 0)
  (h12 : dora.hits_24 = 0) :
  dora.hits_40 = 1 ∧ dora.hits_16 = 5 ∧ 
  reggie.hits_23 = 2 ∧ reggie.hits_16 = 4 ∧ 
  finch.hits_17 = 4 ∧ finch.hits_16 = 2 :=
sorry

end NUMINAMATH_GPT_archer_scores_distribution_l357_35795


namespace NUMINAMATH_GPT_collapsing_fraction_l357_35707

-- Define the total number of homes on Gotham St as a variable.
variable (T : ℕ)

/-- Fraction of homes on Gotham Street that are termite-ridden. -/
def fraction_termite_ridden (T : ℕ) : ℚ := 1 / 3

/-- Fraction of homes on Gotham Street that are termite-ridden but not collapsing. -/
def fraction_termite_not_collapsing (T : ℕ) : ℚ := 1 / 10

/-- Fraction of termite-ridden homes that are collapsing. -/
theorem collapsing_fraction :
  (fraction_termite_ridden T - fraction_termite_not_collapsing T) = 7 / 30 :=
by
  sorry

end NUMINAMATH_GPT_collapsing_fraction_l357_35707


namespace NUMINAMATH_GPT_find_n_l357_35726

theorem find_n (x n : ℤ) (k m : ℤ) (h1 : x = 82*k + 5) (h2 : x + n = 41*m + 22) : n = 5 := by
  sorry

end NUMINAMATH_GPT_find_n_l357_35726


namespace NUMINAMATH_GPT_reflection_about_x_axis_l357_35770

theorem reflection_about_x_axis (a : ℝ) : 
  (A : ℝ × ℝ) = (3, a) → (B : ℝ × ℝ) = (3, 4) → A = (3, -4) → a = -4 :=
by
  intros A_eq B_eq reflection_eq
  sorry

end NUMINAMATH_GPT_reflection_about_x_axis_l357_35770


namespace NUMINAMATH_GPT_yellow_tint_percentage_new_mixture_l357_35754

def original_volume : ℝ := 40
def yellow_tint_percentage : ℝ := 0.35
def additional_yellow_tint : ℝ := 10
def new_volume : ℝ := original_volume + additional_yellow_tint
def original_yellow_tint : ℝ := yellow_tint_percentage * original_volume
def new_yellow_tint : ℝ := original_yellow_tint + additional_yellow_tint

theorem yellow_tint_percentage_new_mixture : 
  (new_yellow_tint / new_volume) * 100 = 48 := 
by
  sorry

end NUMINAMATH_GPT_yellow_tint_percentage_new_mixture_l357_35754


namespace NUMINAMATH_GPT_largest_x_value_l357_35728

theorem largest_x_value : ∃ x : ℝ, (x / 7 + 3 / (7 * x) = 1) ∧ (∀ y : ℝ, (y / 7 + 3 / (7 * y) = 1) → y ≤ (7 + Real.sqrt 37) / 2) :=
by
  -- (Proof of the theorem is omitted for this task)
  sorry

end NUMINAMATH_GPT_largest_x_value_l357_35728


namespace NUMINAMATH_GPT_magic_square_sum_l357_35733

theorem magic_square_sum (S a b c d e : ℤ) (h1 : x + 15 + 100 = S)
                        (h2 : 23 + d + e = S)
                        (h3 : x + a + 23 = S)
                        (h4 : a = 92)
                        (h5 : 92 + b + d = x + 15 + 100)
                        (h6 : b = 0)
                        (h7 : d = 100) : x = 77 :=
by {
  sorry
}

end NUMINAMATH_GPT_magic_square_sum_l357_35733


namespace NUMINAMATH_GPT_point_on_xaxis_y_coord_zero_l357_35741

theorem point_on_xaxis_y_coord_zero (m : ℝ) (h : (3, m).snd = 0) : m = 0 :=
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_point_on_xaxis_y_coord_zero_l357_35741


namespace NUMINAMATH_GPT_sons_age_l357_35798

theorem sons_age (S M : ℕ) (h1 : M = 3 * S) (h2 : M + 12 = 2 * (S + 12)) : S = 12 :=
by 
  sorry

end NUMINAMATH_GPT_sons_age_l357_35798


namespace NUMINAMATH_GPT_ratio_of_r_to_pq_l357_35768

theorem ratio_of_r_to_pq (p q r : ℕ) (h₁ : p + q + r = 7000) (h₂ : r = 2800) :
  r / (p + q) = 2 / 3 :=
by sorry

end NUMINAMATH_GPT_ratio_of_r_to_pq_l357_35768


namespace NUMINAMATH_GPT_perpendicular_distance_is_8_cm_l357_35780

theorem perpendicular_distance_is_8_cm :
  ∀ (side_length distance_from_corner cut_angle : ℝ),
    side_length = 100 →
    distance_from_corner = 8 →
    cut_angle = 45 →
    (∃ h : ℝ, h = 8) :=
by
  intros side_length distance_from_corner cut_angle hms d8 a45
  sorry

end NUMINAMATH_GPT_perpendicular_distance_is_8_cm_l357_35780


namespace NUMINAMATH_GPT_mark_card_sum_l357_35735

/--
Mark has seven green cards numbered 1 through 7 and five red cards numbered 2 through 6.
He arranges the cards such that colors alternate and the sum of each pair of neighboring cards forms a prime.
Prove that the sum of the numbers on the last three cards in his stack is 16.
-/
theorem mark_card_sum {green_cards : Fin 7 → ℕ} {red_cards : Fin 5 → ℕ}
  (h_green_numbered : ∀ i, 1 ≤ green_cards i ∧ green_cards i ≤ 7)
  (h_red_numbered : ∀ i, 2 ≤ red_cards i ∧ red_cards i ≤ 6)
  (h_alternate : ∀ i, i < 6 → (∃ j k, green_cards j + red_cards k = prime) ∨ (red_cards j + green_cards k = prime)) :
  ∃ s, s = 16 := sorry

end NUMINAMATH_GPT_mark_card_sum_l357_35735


namespace NUMINAMATH_GPT_no_equilateral_triangle_on_grid_regular_tetrahedron_on_grid_l357_35766

-- Define the context for part (a)
theorem no_equilateral_triangle_on_grid (x1 y1 x2 y2 x3 y3 : ℤ) :
  ¬ (x1 = x2 ∧ y1 = y2) ∧ (x2 = x3 ∧ y2 = y3) ∧ (x3 = x1 ∧ y3 = y1) ∧ -- vertices must not be the same
  ((x2 - x1)^2 + (y2 - y1)^2 = (x3 - x2)^2 + (y3 - y2)^2) ∧ -- sides must be equal
  ((x3 - x1)^2 + (y3 - y1)^2 = (x2 - x1)^2 + (y2 - y1)^2) ->
  false := 
sorry

-- Define the context for part (b)
theorem regular_tetrahedron_on_grid (x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 : ℤ) :
  ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2 = (x3 - x2)^2 + (y3 - y2)^2 + (z3 - z2)^2) ∧ -- first condition: edge lengths equal
  ((x3 - x1)^2 + (y3 - y1)^2 + (z3 - z1)^2 = (x4 - x3)^2 + (y4 - y3)^2 + (z4 - z3)^2) ∧ -- second condition: edge lengths equal
  ((x4 - x1)^2 + (y4 - y1)^2 + (z4 - z1)^2 = (x2 - x4)^2 + (y2 - y4)^2 + (z2 - z4)^2) -> -- third condition: edge lengths equal
  true := 
sorry

end NUMINAMATH_GPT_no_equilateral_triangle_on_grid_regular_tetrahedron_on_grid_l357_35766


namespace NUMINAMATH_GPT_solve_exp_eq_l357_35774

theorem solve_exp_eq (x : ℝ) (h : Real.sqrt ((1 + Real.sqrt 2)^x) + Real.sqrt ((1 - Real.sqrt 2)^x) = 2) : 
  x = 0 := 
sorry

end NUMINAMATH_GPT_solve_exp_eq_l357_35774


namespace NUMINAMATH_GPT_isabella_non_yellow_houses_l357_35750

variable (Green Yellow Red Blue Pink : ℕ)

axiom h1 : 3 * Yellow = Green
axiom h2 : Red = Yellow + 40
axiom h3 : Green = 90
axiom h4 : Blue = (Green + Yellow) / 2
axiom h5 : Pink = (Red / 2) + 15

theorem isabella_non_yellow_houses : (Green + Red + Blue + Pink - Yellow) = 270 :=
by 
  sorry

end NUMINAMATH_GPT_isabella_non_yellow_houses_l357_35750


namespace NUMINAMATH_GPT_condition_2_3_implies_f_x1_greater_f_x2_l357_35751

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * Real.cos x

theorem condition_2_3_implies_f_x1_greater_f_x2 
(x1 x2 : ℝ) (h1 : -2 * Real.pi / 3 ≤ x1 ∧ x1 ≤ 2 * Real.pi / 3) 
(h2 : -2 * Real.pi / 3 ≤ x2 ∧ x2 ≤ 2 * Real.pi / 3) 
(hx1_sq_gt_x2_sq : x1^2 > x2^2) (hx1_gt_abs_x2 : x1 > |x2|) : 
  f x1 > f x2 := 
sorry

end NUMINAMATH_GPT_condition_2_3_implies_f_x1_greater_f_x2_l357_35751


namespace NUMINAMATH_GPT_relationship_between_x_plus_one_and_ex_l357_35708

theorem relationship_between_x_plus_one_and_ex (x : ℝ) : x + 1 ≤ Real.exp x :=
sorry

end NUMINAMATH_GPT_relationship_between_x_plus_one_and_ex_l357_35708


namespace NUMINAMATH_GPT_range_of_a_l357_35729

noncomputable def f : ℝ → ℝ := sorry

def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def is_monotone_on_nonneg (f : ℝ → ℝ) := ∀ ⦃x y : ℝ⦄, 0 ≤ x → 0 ≤ y → x < y → f x < f y

axiom even_f : is_even f
axiom monotone_f : is_monotone_on_nonneg f

theorem range_of_a (a : ℝ) (h : f a ≥ f 3) : a ≤ -3 ∨ a ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l357_35729


namespace NUMINAMATH_GPT_water_consumed_is_correct_l357_35787

def water_consumed (traveler_ounces : ℕ) (camel_multiplier : ℕ) (ounces_per_gallon : ℕ) : ℕ :=
  let camel_ounces := traveler_ounces * camel_multiplier
  let total_ounces := traveler_ounces + camel_ounces
  total_ounces / ounces_per_gallon

theorem water_consumed_is_correct :
  water_consumed 32 7 128 = 2 :=
by
  -- add proof here
  sorry

end NUMINAMATH_GPT_water_consumed_is_correct_l357_35787


namespace NUMINAMATH_GPT_ribbons_jane_uses_l357_35783

-- Given conditions
def dresses_sewn_first_period (dresses_per_day : ℕ) (days : ℕ) : ℕ :=
  dresses_per_day * days

def dresses_sewn_second_period (dresses_per_day : ℕ) (days : ℕ) : ℕ :=
  dresses_per_day * days

def total_dresses_sewn (dresses_first_period : ℕ) (dresses_second_period : ℕ) : ℕ :=
  dresses_first_period + dresses_second_period

def total_ribbons_used (total_dresses : ℕ) (ribbons_per_dress : ℕ) : ℕ :=
  total_dresses * ribbons_per_dress

-- Theorem to prove
theorem ribbons_jane_uses :
  total_ribbons_used (total_dresses_sewn (dresses_sewn_first_period 2 7) (dresses_sewn_second_period 3 2)) 2 = 40 :=
  sorry

end NUMINAMATH_GPT_ribbons_jane_uses_l357_35783


namespace NUMINAMATH_GPT_max_c_magnitude_l357_35799

variables {a b c : ℝ × ℝ}

-- Definitions of the given conditions
def unit_vector (v : ℝ × ℝ) : Prop := ‖v‖ = 1
def orthogonal (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0
def satisfied_c (c a b : ℝ × ℝ) : Prop := ‖c - (a + b)‖ = 2

-- Main theorem to prove
theorem max_c_magnitude (ha : unit_vector a) (hb : unit_vector b) (hab : orthogonal a b) (hc : satisfied_c c a b) : ‖c‖ ≤ 2 + Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_max_c_magnitude_l357_35799


namespace NUMINAMATH_GPT_least_subtracted_number_l357_35765

theorem least_subtracted_number (a b c d e : ℕ) 
  (h₁ : a = 2590) 
  (h₂ : b = 9) 
  (h₃ : c = 11) 
  (h₄ : d = 13) 
  (h₅ : e = 6) 
  : ∃ (x : ℕ), a - x % b = e ∧ a - x % c = e ∧ a - x % d = e := by
  sorry

end NUMINAMATH_GPT_least_subtracted_number_l357_35765


namespace NUMINAMATH_GPT_sample_size_survey_l357_35785

theorem sample_size_survey (students_selected : ℕ) (h : students_selected = 200) : students_selected = 200 :=
by
  assumption

end NUMINAMATH_GPT_sample_size_survey_l357_35785


namespace NUMINAMATH_GPT_jonathan_needs_more_money_l357_35794

def cost_dictionary : ℕ := 11
def cost_dinosaur_book : ℕ := 19
def cost_childrens_cookbook : ℕ := 7
def saved_money : ℕ := 8

def total_cost : ℕ := cost_dictionary + cost_dinosaur_book + cost_childrens_cookbook
def amount_needed : ℕ := total_cost - saved_money

theorem jonathan_needs_more_money : amount_needed = 29 := by
  have h1 : total_cost = 37 := by
    show 11 + 19 + 7 = 37
    sorry
  show 37 - 8 = 29
  sorry

end NUMINAMATH_GPT_jonathan_needs_more_money_l357_35794


namespace NUMINAMATH_GPT_eel_jellyfish_ratio_l357_35786

noncomputable def combined_cost : ℝ := 200
noncomputable def eel_cost : ℝ := 180
noncomputable def jellyfish_cost : ℝ := combined_cost - eel_cost

theorem eel_jellyfish_ratio : eel_cost / jellyfish_cost = 9 :=
by
  sorry

end NUMINAMATH_GPT_eel_jellyfish_ratio_l357_35786


namespace NUMINAMATH_GPT_f_sq_add_g_sq_eq_one_f_even_f_periodic_l357_35756

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom g_odd : ∀ x : ℝ, g (-x) = - g x
axiom f_0 : f 0 = 1
axiom f_eq : ∀ x y : ℝ, f (x - y) = f x * f y + g x * g y

theorem f_sq_add_g_sq_eq_one (x : ℝ) : f x ^ 2 + g x ^ 2 = 1 :=
sorry

theorem f_even : ∀ x : ℝ, f x = f (-x) :=
sorry

theorem f_periodic (a : ℝ) (ha : a ≠ 0) (hfa : f a = 1) : ∀ x : ℝ, f (x + a) = f x :=
sorry

end NUMINAMATH_GPT_f_sq_add_g_sq_eq_one_f_even_f_periodic_l357_35756


namespace NUMINAMATH_GPT_inequality_l357_35731
-- Import the necessary libraries from Mathlib

-- Define the theorem statement
theorem inequality (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := 
by
  sorry

end NUMINAMATH_GPT_inequality_l357_35731


namespace NUMINAMATH_GPT_words_per_page_l357_35788

theorem words_per_page (p : ℕ) (h1 : p ≤ 150) (h2 : 120 * p ≡ 172 [MOD 221]) : p = 114 := by
  sorry

end NUMINAMATH_GPT_words_per_page_l357_35788


namespace NUMINAMATH_GPT_intersecting_lines_l357_35702

theorem intersecting_lines (a b : ℝ) (h1 : 1 = 1 / 4 * 2 + a) (h2 : 2 = 1 / 4 * 1 + b) : 
  a + b = 9 / 4 := 
sorry

end NUMINAMATH_GPT_intersecting_lines_l357_35702


namespace NUMINAMATH_GPT_library_average_visitors_l357_35779

theorem library_average_visitors (V : ℝ) (h1 : (4 * 1000 + 26 * V = 750 * 30)) : V = 18500 / 26 := 
by 
  -- The actual proof is omitted and replaced by sorry.
  sorry

end NUMINAMATH_GPT_library_average_visitors_l357_35779


namespace NUMINAMATH_GPT_simplify_expression_l357_35714

theorem simplify_expression :
  (1 * 2 * a * 3 * a^2 * 4 * a^3 * 5 * a^4 * 6 * a^5) = 720 * a^15 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l357_35714


namespace NUMINAMATH_GPT_total_sheep_l357_35742

theorem total_sheep (n : ℕ) 
  (h1 : 3 ∣ n)
  (h2 : 5 ∣ n)
  (h3 : 6 ∣ n)
  (h4 : 8 ∣ n)
  (h5 : n * 7 / 40 = 12) : 
  n = 68 :=
by
  sorry

end NUMINAMATH_GPT_total_sheep_l357_35742


namespace NUMINAMATH_GPT_cinematic_academy_members_l357_35744

theorem cinematic_academy_members (h1 : ∀ x, x / 4 ≥ 196.25 → x ≥ 785) : 
  ∃ n : ℝ, 1 / 4 * n = 196.25 ∧ n = 785 :=
by
  sorry

end NUMINAMATH_GPT_cinematic_academy_members_l357_35744


namespace NUMINAMATH_GPT_election_candidate_a_votes_l357_35704

theorem election_candidate_a_votes :
  let total_votes : ℕ := 560000
  let invalid_percentage : ℚ := 15 / 100
  let candidate_a_percentage : ℚ := 70 / 100
  let total_valid_votes := total_votes * (1 - invalid_percentage)
  let candidate_a_votes := total_valid_votes * candidate_a_percentage
  candidate_a_votes = 333200 :=
by
  let total_votes : ℕ := 560000
  let invalid_percentage : ℚ := 15 / 100
  let candidate_a_percentage : ℚ := 70 / 100
  let total_valid_votes := total_votes * (1 - invalid_percentage)
  let candidate_a_votes := total_valid_votes * candidate_a_percentage
  show candidate_a_votes = 333200
  sorry

end NUMINAMATH_GPT_election_candidate_a_votes_l357_35704


namespace NUMINAMATH_GPT_normal_price_of_article_l357_35718

theorem normal_price_of_article 
  (final_price : ℝ)
  (discount1 : ℝ) 
  (discount2 : ℝ) 
  (P : ℝ)
  (h : final_price = 108) 
  (h1 : discount1 = 0.10) 
  (h2 : discount2 = 0.20)
  (h_eq : (1 - discount1) * (1 - discount2) * P = final_price) :
  P = 150 := by
  sorry

end NUMINAMATH_GPT_normal_price_of_article_l357_35718


namespace NUMINAMATH_GPT_people_own_pets_at_least_l357_35710

-- Definitions based on given conditions
def people_owning_only_dogs : ℕ := 15
def people_owning_only_cats : ℕ := 10
def people_owning_only_cats_and_dogs : ℕ := 5
def people_owning_cats_dogs_snakes : ℕ := 3
def total_snakes : ℕ := 59

-- Theorem statement to prove the total number of people owning pets
theorem people_own_pets_at_least : 
  people_owning_only_dogs + people_owning_only_cats + people_owning_only_cats_and_dogs + people_owning_cats_dogs_snakes ≥ 33 :=
by {
  -- Proof steps will go here
  sorry
}

end NUMINAMATH_GPT_people_own_pets_at_least_l357_35710


namespace NUMINAMATH_GPT_part1_part2_part3_l357_35773

def f (x : ℝ) : ℝ := abs (2 * x + 1) - abs (x - 2)
def g (x : ℝ) : ℝ := f x - abs (x - 2)

theorem part1 : ∀ x : ℝ, f x ≤ 8 ↔ (-11 ≤ x ∧ x ≤ 5) := by sorry

theorem part2 : ∃ x : ℝ, g x = 5 := by sorry

theorem part3 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 5) : 
  1 / a + 9 / b = 16 / 5 := by sorry

end NUMINAMATH_GPT_part1_part2_part3_l357_35773


namespace NUMINAMATH_GPT_value_of_expression_l357_35790

theorem value_of_expression : (15 + 5)^2 - (15^2 + 5^2) = 150 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l357_35790


namespace NUMINAMATH_GPT_planA_equals_planB_at_3_l357_35739

def planA_charge_for_first_9_minutes : ℝ := 0.24
def planA_charge (X: ℝ) (minutes: ℕ) : ℝ := if minutes <= 9 then X else X + 0.06 * (minutes - 9)
def planB_charge (minutes: ℕ) : ℝ := 0.08 * minutes

theorem planA_equals_planB_at_3 : planA_charge planA_charge_for_first_9_minutes 3 = planB_charge 3 :=
by sorry

end NUMINAMATH_GPT_planA_equals_planB_at_3_l357_35739


namespace NUMINAMATH_GPT_rectangle_area_l357_35761

theorem rectangle_area (r : ℝ) (L W : ℝ) (h₀ : r = 7) (h₁ : 2 * r = W) (h₂ : L / W = 3) : 
  L * W = 588 :=
by sorry

end NUMINAMATH_GPT_rectangle_area_l357_35761


namespace NUMINAMATH_GPT_inequality_proof_l357_35719

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) ≤ 1 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l357_35719


namespace NUMINAMATH_GPT_third_side_length_l357_35789

def is_odd (n : ℕ) := n % 2 = 1

theorem third_side_length (x : ℕ) (h1 : 2 + 5 > x) (h2 : x + 2 > 5) (h3 : is_odd x) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_third_side_length_l357_35789


namespace NUMINAMATH_GPT_find_expression_value_l357_35711

theorem find_expression_value (m: ℝ) (h: m^2 - 2 * m - 1 = 0) : 
  (m - 1)^2 - (m - 3) * (m + 3) - (m - 1) * (m - 3) = 6 := 
by 
  sorry

end NUMINAMATH_GPT_find_expression_value_l357_35711


namespace NUMINAMATH_GPT_seating_arrangements_l357_35713

-- Definitions for conditions
def num_parents : ℕ := 2
def num_children : ℕ := 3
def num_front_seats : ℕ := 2
def num_back_seats : ℕ := 3
def num_family_members : ℕ := num_parents + num_children

-- The statement we need to prove
theorem seating_arrangements : 
  (num_parents * -- choices for driver
  (num_family_members - 1) * -- choices for the front passenger
  (num_back_seats.factorial)) = 48 := -- arrangements for the back seats
by
  sorry

end NUMINAMATH_GPT_seating_arrangements_l357_35713


namespace NUMINAMATH_GPT_minimum_rubles_to_reverse_chips_l357_35782

theorem minimum_rubles_to_reverse_chips (n : ℕ) (h : n = 100)
  (adjacent_cost : ℕ → ℕ → ℕ)
  (free_cost : ℕ → ℕ → Prop)
  (reverse_cost : ℕ) :
  (∀ i j, i + 1 = j → adjacent_cost i j = 1) →
  (∀ i j, i + 5 = j → free_cost i j) →
  reverse_cost = 61 :=
by
  sorry

end NUMINAMATH_GPT_minimum_rubles_to_reverse_chips_l357_35782


namespace NUMINAMATH_GPT_man_l357_35705

-- Conditions
def speed_with_current : ℝ := 18
def speed_of_current : ℝ := 3.4

-- Problem statement
theorem man's_speed_against_current :
  (speed_with_current - speed_of_current - speed_of_current) = 11.2 := 
by
  sorry

end NUMINAMATH_GPT_man_l357_35705


namespace NUMINAMATH_GPT_proof_l357_35732

noncomputable def problem_statement (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (∀ x : ℝ, |x + a| + |x - b| + c ≥ 4)

theorem proof (a b c : ℝ) (h : problem_statement a b c) :
  a + b + c = 4 ∧ (∀ x : ℝ, 1 / a + 4 / b + 9 / c ≥ 9) :=
by
  sorry

end NUMINAMATH_GPT_proof_l357_35732


namespace NUMINAMATH_GPT_sum_of_solutions_l357_35723

theorem sum_of_solutions (x : ℝ) (h : x + 16 / x = 12) : x = 8 ∨ x = 4 → 8 + 4 = 12 := by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l357_35723


namespace NUMINAMATH_GPT_frequency_total_students_l357_35746

noncomputable def total_students (known : ℕ) (freq : ℝ) : ℝ :=
known / freq

theorem frequency_total_students (known : ℕ) (freq : ℝ) (h1 : known = 40) (h2 : freq = 0.8) :
  total_students known freq = 50 :=
by
  rw [total_students, h1, h2]
  norm_num

end NUMINAMATH_GPT_frequency_total_students_l357_35746


namespace NUMINAMATH_GPT_monotonically_increasing_interval_l357_35792

noncomputable def f (x : ℝ) : ℝ := 7 * Real.sin (x - Real.pi / 6)

theorem monotonically_increasing_interval :
  (∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 2 → f x < f y) :=
sorry

end NUMINAMATH_GPT_monotonically_increasing_interval_l357_35792


namespace NUMINAMATH_GPT_set_intersection_complement_l357_35757

open Set

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
noncomputable def M : Set ℕ := {2, 3, 4, 5}
noncomputable def N : Set ℕ := {1, 4, 5, 7}

theorem set_intersection_complement :
  M ∩ (U \ N) = {2, 3} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_complement_l357_35757


namespace NUMINAMATH_GPT_ann_total_fare_for_100_miles_l357_35763

-- Conditions
def base_fare : ℕ := 20
def fare_per_distance (distance : ℕ) : ℕ := 180 * distance / 80

-- Question: How much would Ann be charged if she traveled 100 miles?
def total_fare (distance : ℕ) : ℕ := (fare_per_distance distance) + base_fare

-- Prove that the total fare for 100 miles is 245 dollars
theorem ann_total_fare_for_100_miles : total_fare 100 = 245 :=
by
  -- Adding your proof here
  sorry

end NUMINAMATH_GPT_ann_total_fare_for_100_miles_l357_35763


namespace NUMINAMATH_GPT_green_yarn_length_l357_35791

/-- The length of the green piece of yarn given the red yarn is 8 cm more 
than three times the length of the green yarn and the total length 
for 2 pieces of yarn is 632 cm. -/
theorem green_yarn_length (G R : ℕ) 
  (h1 : R = 3 * G + 8)
  (h2 : G + R = 632) : 
  G = 156 := 
by
  sorry

end NUMINAMATH_GPT_green_yarn_length_l357_35791


namespace NUMINAMATH_GPT_infinite_squares_in_arithmetic_progression_l357_35781

theorem infinite_squares_in_arithmetic_progression
  (a d : ℕ) (hposd : 0 < d) (hpos : 0 < a) (k n : ℕ)
  (hk : a + k * d = n^2) :
  ∃ (t : ℕ), ∃ (m : ℕ), (a + (k + t) * d = m^2) := by
  sorry

end NUMINAMATH_GPT_infinite_squares_in_arithmetic_progression_l357_35781


namespace NUMINAMATH_GPT_problem_solution_l357_35793

theorem problem_solution (x : ℝ) (hx : x + 1/x = Real.sqrt 5) : x^11 - 7 * x^7 + x^3 = 0 := 
sorry

end NUMINAMATH_GPT_problem_solution_l357_35793


namespace NUMINAMATH_GPT_flower_seedlings_pots_l357_35753

theorem flower_seedlings_pots (x y z : ℕ) :
  (1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z) →
  (x + y + z = 16) →
  (2 * x + 4 * y + 10 * z = 50) →
  (x = 10 ∨ x = 13) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_flower_seedlings_pots_l357_35753


namespace NUMINAMATH_GPT_total_fencing_cost_l357_35721

theorem total_fencing_cost
  (park_is_square : true)
  (cost_per_side : ℕ)
  (h1 : cost_per_side = 43) :
  4 * cost_per_side = 172 :=
by
  sorry

end NUMINAMATH_GPT_total_fencing_cost_l357_35721


namespace NUMINAMATH_GPT_relationship_among_values_l357_35748

-- Assume there exists a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Condition 1: f is strictly increasing on (0, 3)
def increasing_on_0_to_3 : Prop :=
  ∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 3 → f x < f y

-- Condition 2: f(x + 3) is an even function
def even_function_shifted : Prop :=
  ∀ x : ℝ, f (x + 3) = f (-(x + 3))

-- The theorem we need to prove
theorem relationship_among_values 
  (h1 : increasing_on_0_to_3 f)
  (h2 : even_function_shifted f) :
  f (9/2) < f 2 ∧ f 2 < f (7/2) :=
sorry

end NUMINAMATH_GPT_relationship_among_values_l357_35748


namespace NUMINAMATH_GPT_opposite_of_negative_five_l357_35716

theorem opposite_of_negative_five : ∀ x : Int, -5 + x = 0 → x = 5 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_opposite_of_negative_five_l357_35716
