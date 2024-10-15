import Mathlib

namespace NUMINAMATH_GPT_value_of_expression_l1846_184669

def g (x : ℝ) (p q r s t : ℝ) : ℝ :=
  p * x^4 + q * x^3 + r * x^2 + s * x + t

theorem value_of_expression (p q r s t : ℝ) (h : g (-1) p q r s t = 4) :
  12 * p - 6 * q + 3 * r - 2 * s + t = 13 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l1846_184669


namespace NUMINAMATH_GPT_amazing_rectangle_area_unique_l1846_184684

def isAmazingRectangle (a b : ℕ) : Prop :=
  a = 2 * b ∧ a * b = 3 * (2 * (a + b))

theorem amazing_rectangle_area_unique :
  ∃ (a b : ℕ), isAmazingRectangle a b ∧ a * b = 162 :=
by
  sorry

end NUMINAMATH_GPT_amazing_rectangle_area_unique_l1846_184684


namespace NUMINAMATH_GPT_smallest_x_solution_l1846_184658

theorem smallest_x_solution :
  (∃ x : ℚ, abs (4 * x + 3) = 30 ∧ ∀ y : ℚ, abs (4 * y + 3) = 30 → x ≤ y) ↔ x = -33 / 4 := by
  sorry

end NUMINAMATH_GPT_smallest_x_solution_l1846_184658


namespace NUMINAMATH_GPT_probability_of_at_least_one_black_ball_l1846_184653

noncomputable def probability_at_least_one_black_ball := 
  let total_outcomes := Nat.choose 4 2
  let favorable_outcomes := (Nat.choose 2 1) * (Nat.choose 2 1) + (Nat.choose 2 2)
  favorable_outcomes / total_outcomes

theorem probability_of_at_least_one_black_ball :
  probability_at_least_one_black_ball = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_at_least_one_black_ball_l1846_184653


namespace NUMINAMATH_GPT_chad_sandwiches_l1846_184664

-- Definitions representing the conditions
def crackers_per_sleeve : ℕ := 28
def sleeves_per_box : ℕ := 4
def boxes : ℕ := 5
def nights : ℕ := 56
def crackers_per_sandwich : ℕ := 2

-- Definition representing the final question about the number of sandwiches
def sandwiches_per_night (crackers_per_sleeve sleeves_per_box boxes nights crackers_per_sandwich : ℕ) : ℕ :=
  (crackers_per_sleeve * sleeves_per_box * boxes) / nights / crackers_per_sandwich

-- The theorem that states Chad makes 5 sandwiches each night
theorem chad_sandwiches :
  sandwiches_per_night crackers_per_sleeve sleeves_per_box boxes nights crackers_per_sandwich = 5 :=
by
  -- Proof outline:
  -- crackers_per_sleeve * sleeves_per_box * boxes = 28 * 4 * 5 = 560
  -- 560 / nights = 560 / 56 = 10 crackers per night
  -- 10 / crackers_per_sandwich = 10 / 2 = 5 sandwiches per night
  sorry

end NUMINAMATH_GPT_chad_sandwiches_l1846_184664


namespace NUMINAMATH_GPT_smallest_value_n_l1846_184620

theorem smallest_value_n :
  ∃ (n : ℕ), n * 25 = Nat.lcm (Nat.lcm 10 18) 20 ∧ (∀ m, m * 25 = Nat.lcm (Nat.lcm 10 18) 20 → n ≤ m) := 
sorry

end NUMINAMATH_GPT_smallest_value_n_l1846_184620


namespace NUMINAMATH_GPT_complex_modulus_to_real_l1846_184623

theorem complex_modulus_to_real (a : ℝ) (h : (a + 1)^2 + (1 - a)^2 = 10) : a = 2 ∨ a = -2 :=
sorry

end NUMINAMATH_GPT_complex_modulus_to_real_l1846_184623


namespace NUMINAMATH_GPT_layla_goals_l1846_184641

variable (L K : ℕ)
variable (average_score : ℕ := 92)
variable (goals_difference : ℕ := 24)
variable (total_games : ℕ := 4)

theorem layla_goals :
  K = L - goals_difference →
  (L + K) = (average_score * total_games) →
  L = 196 :=
by
  sorry

end NUMINAMATH_GPT_layla_goals_l1846_184641


namespace NUMINAMATH_GPT_find_x_l1846_184661

variables (a b c d x : ℤ)

theorem find_x (h1 : a - b = c + d + 9) (h2 : a - c = 3) (h3 : a + b = c - d - x) : x = 3 :=
sorry

end NUMINAMATH_GPT_find_x_l1846_184661


namespace NUMINAMATH_GPT_largest_possible_b_l1846_184624

theorem largest_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b = 12 :=
sorry

end NUMINAMATH_GPT_largest_possible_b_l1846_184624


namespace NUMINAMATH_GPT_sin_cos_sum_eq_l1846_184615

theorem sin_cos_sum_eq :
  (Real.sin (20 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) +
   Real.sin (70 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)) = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_sin_cos_sum_eq_l1846_184615


namespace NUMINAMATH_GPT_complement_of_A_union_B_in_U_l1846_184693

def U : Set ℝ := { x | -5 < x ∧ x < 5 }
def A : Set ℝ := { x | x^2 - 4*x - 5 < 0 }
def B : Set ℝ := { x | -2 < x ∧ x < 4 }

theorem complement_of_A_union_B_in_U :
  (U \ (A ∪ B)) = { x | -5 < x ∧ x ≤ -2 } := by
  sorry

end NUMINAMATH_GPT_complement_of_A_union_B_in_U_l1846_184693


namespace NUMINAMATH_GPT_find_y_l1846_184625

theorem find_y (n x y : ℝ)
  (h1 : (100 + 200 + n + x) / 4 = 250)
  (h2 : (n + 150 + 100 + x + y) / 5 = 200) :
  y = 50 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1846_184625


namespace NUMINAMATH_GPT_weight_of_four_cakes_l1846_184637

variable (C B : ℕ)  -- We declare C and B as natural numbers representing the weights in grams.

def cake_bread_weight_conditions (C B : ℕ) : Prop :=
  (3 * C + 5 * B = 1100) ∧ (C = B + 100)

theorem weight_of_four_cakes (C B : ℕ) 
  (h : cake_bread_weight_conditions C B) : 
  4 * C = 800 := 
by 
  {sorry}

end NUMINAMATH_GPT_weight_of_four_cakes_l1846_184637


namespace NUMINAMATH_GPT_bobby_initial_candy_l1846_184677

theorem bobby_initial_candy (candy_ate_start candy_ate_more candy_left : ℕ)
  (h1 : candy_ate_start = 9) (h2 : candy_ate_more = 5) (h3 : candy_left = 8) :
  candy_ate_start + candy_ate_more + candy_left = 22 :=
by
  rw [h1, h2, h3]
  -- sorry


end NUMINAMATH_GPT_bobby_initial_candy_l1846_184677


namespace NUMINAMATH_GPT_modular_inverse_calculation_l1846_184647

theorem modular_inverse_calculation : 
  (3 * (49 : ℤ) + 12 * (40 : ℤ)) % 65 = 42 := 
by
  sorry

end NUMINAMATH_GPT_modular_inverse_calculation_l1846_184647


namespace NUMINAMATH_GPT_bob_total_questions_l1846_184606

theorem bob_total_questions (q1 q2 q3 : ℕ) : 
  q1 = 13 ∧ q2 = 2 * q1 ∧ q3 = 2 * q2 → q1 + q2 + q3 = 91 :=
by
  intros
  sorry

end NUMINAMATH_GPT_bob_total_questions_l1846_184606


namespace NUMINAMATH_GPT_problem_proof_l1846_184646

noncomputable def f : ℝ → ℝ := sorry

theorem problem_proof (h1 : ∀ x : ℝ, f (-x) = f x)
    (h2 : ∀ x y : ℝ, x < y ∧ y ≤ -1 → f x < f y) : 
    f 2 < f (-3 / 2) ∧ f (-3 / 2) < f (-1) :=
by
  sorry

end NUMINAMATH_GPT_problem_proof_l1846_184646


namespace NUMINAMATH_GPT_largest_number_in_box_l1846_184614

theorem largest_number_in_box
  (a : ℕ)
  (sum_eq_480 : a + (a + 1) + (a + 2) + (a + 10) + (a + 11) + (a + 12) = 480) :
  a + 12 = 86 :=
by
  sorry

end NUMINAMATH_GPT_largest_number_in_box_l1846_184614


namespace NUMINAMATH_GPT_last_four_digits_5_to_2019_l1846_184651

theorem last_four_digits_5_to_2019 :
  ∃ (x : ℕ), (5^2019) % 10000 = x ∧ x = 8125 :=
by
  sorry

end NUMINAMATH_GPT_last_four_digits_5_to_2019_l1846_184651


namespace NUMINAMATH_GPT_mixed_number_evaluation_l1846_184673

theorem mixed_number_evaluation :
  let a := (4 + 1 / 3 : ℚ)
  let b := (3 + 2 / 7 : ℚ)
  let c := (2 + 5 / 6 : ℚ)
  let d := (1 + 1 / 2 : ℚ)
  let e := (5 + 1 / 4 : ℚ)
  let f := (3 + 2 / 5 : ℚ)
  (a + b - c) * (d + e) / f = 9 + 198 / 317 :=
by {
  let a : ℚ := 4 + 1 / 3
  let b : ℚ := 3 + 2 / 7
  let c : ℚ := 2 + 5 / 6
  let d : ℚ := 1 + 1 / 2
  let e : ℚ := 5 + 1 / 4
  let f : ℚ := 3 + 2 / 5
  sorry
}

end NUMINAMATH_GPT_mixed_number_evaluation_l1846_184673


namespace NUMINAMATH_GPT_rectangle_length_reduction_30_percent_l1846_184692

variables (L W : ℝ) (x : ℝ)

theorem rectangle_length_reduction_30_percent
  (h : 1 = (1 - x / 100) * 1.4285714285714287) :
  x = 30 :=
sorry

end NUMINAMATH_GPT_rectangle_length_reduction_30_percent_l1846_184692


namespace NUMINAMATH_GPT_at_least_one_inequality_false_l1846_184666

open Classical

theorem at_least_one_inequality_false (a b c d : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d) :
  ¬ (a + b < c + d ∧ (a + b) * (c + d) < a * b + c * d ∧ (a + b) * c * d < a * b * (c + d)) :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_inequality_false_l1846_184666


namespace NUMINAMATH_GPT_first_present_cost_is_18_l1846_184672

-- Conditions as definitions
variables (x : ℕ)

-- Given conditions
def first_present_cost := x
def second_present_cost := x + 7
def third_present_cost := x - 11
def total_cost := first_present_cost x + second_present_cost x + third_present_cost x

-- Statement of the problem
theorem first_present_cost_is_18 (h : total_cost x = 50) : x = 18 :=
by {
  sorry  -- Proof omitted
}

end NUMINAMATH_GPT_first_present_cost_is_18_l1846_184672


namespace NUMINAMATH_GPT_checkerboard_no_identical_numbers_l1846_184630

theorem checkerboard_no_identical_numbers :
  ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 11 ∧ 1 ≤ j ∧ j ≤ 19 → 19 * (i - 1) + j = 11 * (j - 1) + i → false :=
by
  sorry

end NUMINAMATH_GPT_checkerboard_no_identical_numbers_l1846_184630


namespace NUMINAMATH_GPT_smallest_consecutive_divisible_by_17_l1846_184607

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_consecutive_divisible_by_17 :
  ∃ (n m : ℕ), 
    (m = n + 1) ∧
    sum_digits n % 17 = 0 ∧ 
    sum_digits m % 17 = 0 ∧ 
    n = 8899 ∧ 
    m = 8900 := 
by
  sorry

end NUMINAMATH_GPT_smallest_consecutive_divisible_by_17_l1846_184607


namespace NUMINAMATH_GPT_find_some_number_l1846_184645

theorem find_some_number (x some_number : ℝ) (h1 : (27 / 4) * x - some_number = 3 * x + 27) (h2 : x = 12) :
  some_number = 18 :=
by
  sorry

end NUMINAMATH_GPT_find_some_number_l1846_184645


namespace NUMINAMATH_GPT_find_digits_l1846_184638

-- Definitions, conditions and statement of the problem
def satisfies_condition (z : ℕ) (k : ℕ) (n : ℕ) : Prop :=
  n ≥ 1 ∧ (n^9 % 10^k) / 10^(k - 1) = z

theorem find_digits (z : ℕ) (k : ℕ) :
  k ≥ 1 →
  (z = 0 ∨ z = 1 ∨ z = 3 ∨ z = 7 ∨ z = 9) →
  ∃ n, satisfies_condition z k n := 
sorry

end NUMINAMATH_GPT_find_digits_l1846_184638


namespace NUMINAMATH_GPT_value_of_a_plus_d_l1846_184654

variable (a b c d : ℝ)

theorem value_of_a_plus_d
  (h1 : a + b = 4)
  (h2 : b + c = 5)
  (h3 : c + d = 3) :
  a + d = 1 :=
by
sorry

end NUMINAMATH_GPT_value_of_a_plus_d_l1846_184654


namespace NUMINAMATH_GPT_minimum_red_vertices_l1846_184635

theorem minimum_red_vertices (n : ℕ) (h : 0 < n) :
  ∃ R : ℕ, (∀ i j : ℕ, i < n ∧ j < n →
    (i + j) % 2 = 0 → true) ∧
    R = Int.ceil (n^2 / 2 : ℝ) :=
sorry

end NUMINAMATH_GPT_minimum_red_vertices_l1846_184635


namespace NUMINAMATH_GPT_find_counterfeit_coin_l1846_184611

-- Define the context of the problem
variables (coins : Fin 6 → ℝ) -- six coins represented as a function from Fin 6 to their weights
          (is_counterfeit : Fin 6 → Prop) -- a predicate indicating if the coin is counterfeit
          (real_weight : ℝ) -- the unknown weight of a real coin

-- Existence assertion for the counterfeit coin
axiom exists_counterfeit : ∃ x, is_counterfeit x

-- Define the total weights of coins 1&2 and 3&4
def weight_1_2 := coins 0 + coins 1
def weight_3_4 := coins 2 + coins 3

-- Statement of the problem
theorem find_counterfeit_coin :
  (weight_1_2 = weight_3_4 → (is_counterfeit 4 ∨ is_counterfeit 5)) ∧ 
  (weight_1_2 ≠ weight_3_4 → (is_counterfeit 0 ∨ is_counterfeit 1 ∨ is_counterfeit 2 ∨ is_counterfeit 3)) :=
sorry

end NUMINAMATH_GPT_find_counterfeit_coin_l1846_184611


namespace NUMINAMATH_GPT_minute_hand_rotation_l1846_184604

theorem minute_hand_rotation :
  (10 / 60) * (2 * Real.pi) = (- Real.pi / 3) :=
by
  sorry

end NUMINAMATH_GPT_minute_hand_rotation_l1846_184604


namespace NUMINAMATH_GPT_ellipse_range_l1846_184648

theorem ellipse_range (t : ℝ) (x y : ℝ) :
  (10 - t > 0) → (t - 4 > 0) → (10 - t ≠ t - 4) →
  (t ∈ (Set.Ioo 4 7 ∪ Set.Ioo 7 10)) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_ellipse_range_l1846_184648


namespace NUMINAMATH_GPT_Cody_book_series_total_count_l1846_184656

theorem Cody_book_series_total_count :
  ∀ (weeks: ℕ) (books_first_week: ℕ) (books_second_week: ℕ) (books_per_week_after: ℕ),
    weeks = 7 ∧ books_first_week = 6 ∧ books_second_week = 3 ∧ books_per_week_after = 9 →
    (books_first_week + books_second_week + (weeks - 2) * books_per_week_after) = 54 :=
by
  sorry

end NUMINAMATH_GPT_Cody_book_series_total_count_l1846_184656


namespace NUMINAMATH_GPT_certain_number_divisibility_l1846_184682

-- Define the conditions and the main problem statement
theorem certain_number_divisibility (n : ℕ) (h1 : 0 < n) (h2 : n < 11) (h3 : (18888 - n) % k = 0) (h4 : n = 1) : k = 11 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_divisibility_l1846_184682


namespace NUMINAMATH_GPT_solve_for_x_l1846_184696

theorem solve_for_x (x : ℤ) (h : 3 * x - 7 = 11) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1846_184696


namespace NUMINAMATH_GPT_hyperbola_center_l1846_184609

theorem hyperbola_center : ∃ c : ℝ × ℝ, c = (3, 5) ∧
  ∀ x y : ℝ, 9 * x ^ 2 - 54 * x - 36 * y ^ 2 + 360 * y - 891 = 0 → (c.1 = 3 ∧ c.2 = 5) :=
by
  use (3, 5)
  sorry

end NUMINAMATH_GPT_hyperbola_center_l1846_184609


namespace NUMINAMATH_GPT_options_equal_results_l1846_184633

theorem options_equal_results :
  (4^3 ≠ 3^4) ∧
  ((-5)^3 = (-5^3)) ∧
  ((-6)^2 ≠ -6^2) ∧
  ((- (5/2))^2 ≠ (- (2/5))^2) :=
by {
  sorry
}

end NUMINAMATH_GPT_options_equal_results_l1846_184633


namespace NUMINAMATH_GPT_part1_part2_l1846_184650

variable (a : ℝ)

-- Defining the set A
def setA (a : ℝ) : Set ℝ := { x : ℝ | (x - 2) * (x - 3 * a - 1) < 0 }

-- Part 1: For a = 2, setB should be {x | 2 < x < 7}
theorem part1 : setA 2 = { x : ℝ | 2 < x ∧ x < 7 } :=
by
  sorry

-- Part 2: If setA a = setB, then a = -1
theorem part2 (B : Set ℝ) (h : setA a = B) : a = -1 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1846_184650


namespace NUMINAMATH_GPT_max_axbycz_value_l1846_184689

theorem max_axbycz_value (a b c : ℝ) (x y z : ℝ) 
  (h_triangle: a + b > c ∧ b + c > a ∧ c + a > b)
  (h_positive: 0 < x ∧ 0 < y ∧ 0 < z)
  (h_sum : x + y + z = 1) : 
  a * x * y + b * y * z + c * z * x ≤ (a * b * c) / (2 * a * b + 2 * b * c + 2 * c * a - a^2 - b^2 - c^2) :=
  sorry

end NUMINAMATH_GPT_max_axbycz_value_l1846_184689


namespace NUMINAMATH_GPT_total_nails_to_cut_l1846_184652

theorem total_nails_to_cut :
  let dogs := 4 
  let legs_per_dog := 4
  let nails_per_dog_leg := 4
  let parrots := 8
  let legs_per_parrot := 2
  let nails_per_parrot_leg := 3
  let extra_nail := 1
  let total_dog_nails := dogs * legs_per_dog * nails_per_dog_leg
  let total_parrot_nails := (parrots * legs_per_parrot * nails_per_parrot_leg) + extra_nail
  total_dog_nails + total_parrot_nails = 113 :=
sorry

end NUMINAMATH_GPT_total_nails_to_cut_l1846_184652


namespace NUMINAMATH_GPT_simplify_fraction_l1846_184699

variable (x : ℕ)

theorem simplify_fraction (h : x = 3) : (x^10 + 15 * x^5 + 125) / (x^5 + 5) = 248 + 25 / 62 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1846_184699


namespace NUMINAMATH_GPT_not_possible_to_fill_6x6_with_1x4_l1846_184616

theorem not_possible_to_fill_6x6_with_1x4 :
  ¬ (∃ (a b : ℕ), a + 4 * b = 6 ∧ 4 * a + b = 6) :=
by
  -- Assuming a and b represent the number of 1x4 rectangles aligned horizontally and vertically respectively
  sorry

end NUMINAMATH_GPT_not_possible_to_fill_6x6_with_1x4_l1846_184616


namespace NUMINAMATH_GPT_total_games_won_l1846_184660

theorem total_games_won 
  (bulls_games : ℕ) (heat_games : ℕ) (knicks_games : ℕ)
  (bulls_condition : bulls_games = 70)
  (heat_condition : heat_games = bulls_games + 5)
  (knicks_condition : knicks_games = 2 * heat_games) :
  bulls_games + heat_games + knicks_games = 295 :=
by
  sorry

end NUMINAMATH_GPT_total_games_won_l1846_184660


namespace NUMINAMATH_GPT_general_term_arithmetic_sequence_l1846_184626

-- Define an arithmetic sequence with first term a1 and common ratio q
def arithmetic_sequence (a1 : ℤ) (q : ℤ) (n : ℕ) : ℤ :=
  a1 * q ^ (n - 1)

-- Theorem: given the conditions, prove that the general term is a1 * q^(n-1)
theorem general_term_arithmetic_sequence (a1 q : ℤ) (n : ℕ) :
  arithmetic_sequence a1 q n = a1 * q ^ (n - 1) :=
by
  sorry

end NUMINAMATH_GPT_general_term_arithmetic_sequence_l1846_184626


namespace NUMINAMATH_GPT_scarlet_savings_l1846_184643

noncomputable def remaining_savings (initial_savings earrings_cost necklace_cost bracelet_cost jewelry_set_cost jewelry_set_discount sales_tax_percentage : ℝ) : ℝ :=
  let total_item_cost := earrings_cost + necklace_cost + bracelet_cost
  let discounted_jewelry_set_cost := jewelry_set_cost * (1 - jewelry_set_discount / 100)
  let total_cost_before_tax := total_item_cost + discounted_jewelry_set_cost
  let total_sales_tax := total_cost_before_tax * (sales_tax_percentage / 100)
  let final_total_cost := total_cost_before_tax + total_sales_tax
  initial_savings - final_total_cost

theorem scarlet_savings : remaining_savings 200 23 48 35 80 25 5 = 25.70 :=
by
  sorry

end NUMINAMATH_GPT_scarlet_savings_l1846_184643


namespace NUMINAMATH_GPT_hexagon_area_correct_m_plus_n_l1846_184612

noncomputable def hexagon_area (b : ℝ) : ℝ :=
  let A := (0, 0)
  let B := (b, 3)
  let F := (-3 * (3 + b) / 2, 9)  -- derived from complex numbers and angle conversion
  let hexagon_height := 12  -- height difference between the y-coordinates
  let hexagon_base := 3 * (b + 3) / 2  -- distance between parallel lines AB and DE
  36 / 2 * (b + 3) + 6 * (6 + b * Real.sqrt 3)

theorem hexagon_area_correct (b : ℝ) :
  hexagon_area b = 72 * Real.sqrt 3 :=
sorry

theorem m_plus_n : 72 + 3 = 75 := rfl

end NUMINAMATH_GPT_hexagon_area_correct_m_plus_n_l1846_184612


namespace NUMINAMATH_GPT_average_age_before_new_students_joined_l1846_184603

/-
Problem: Given that the original strength of the class was 18, 
18 new students with an average age of 32 years joined the class, 
and the average age decreased by 4 years, prove that 
the average age of the class before the new students joined was 40 years.
-/

def original_strength := 18
def new_students := 18
def average_age_new_students := 32
def decrease_in_average_age := 4
def original_average_age := 40

theorem average_age_before_new_students_joined :
  (original_strength * original_average_age + new_students * average_age_new_students) / (original_strength + new_students) = original_average_age - decrease_in_average_age :=
by
  sorry

end NUMINAMATH_GPT_average_age_before_new_students_joined_l1846_184603


namespace NUMINAMATH_GPT_minimum_m_value_l1846_184632

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem minimum_m_value :
  (∃ m, ∀ (x1 x2 x3 : ℝ), 0 ≤ x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 ≤ Real.pi → |f x1 - f x2| + |f x2 - f x3| ≤ m) ∧
  ∀ m', (∀ (x1 x2 x3 : ℝ), 0 ≤ x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 ≤ Real.pi → |f x1 - f x2| + |f x2 - f x3| ≤ m') → 3 + Real.sqrt 3 / 2 ≤ m' :=
by
  sorry

end NUMINAMATH_GPT_minimum_m_value_l1846_184632


namespace NUMINAMATH_GPT_points_satisfy_equation_l1846_184642

theorem points_satisfy_equation :
  ∀ (x y : ℝ), x^2 - y^4 = Real.sqrt (18 * x - x^2 - 81) ↔ 
               (x = 9 ∧ y = Real.sqrt 3) ∨ (x = 9 ∧ y = -Real.sqrt 3) := 
by 
  intros x y 
  sorry

end NUMINAMATH_GPT_points_satisfy_equation_l1846_184642


namespace NUMINAMATH_GPT_total_cartons_packed_l1846_184631

-- Define the given conditions
def cans_per_carton : ℕ := 20
def cartons_loaded : ℕ := 40
def cans_left : ℕ := 200

-- Formalize the proof problem
theorem total_cartons_packed : cartons_loaded + (cans_left / cans_per_carton) = 50 := by
  sorry

end NUMINAMATH_GPT_total_cartons_packed_l1846_184631


namespace NUMINAMATH_GPT_trees_planted_l1846_184681

theorem trees_planted (interval trail_length : ℕ) (h1 : interval = 30) (h2 : trail_length = 1200) : 
  trail_length / interval = 40 :=
by
  sorry

end NUMINAMATH_GPT_trees_planted_l1846_184681


namespace NUMINAMATH_GPT_intersect_sets_l1846_184668

def M : Set ℝ := { x | x ≥ -1 }
def N : Set ℝ := { x | -2 < x ∧ x < 2 }

theorem intersect_sets :
  M ∩ N = { x | -1 ≤ x ∧ x < 2 } := by
  sorry

end NUMINAMATH_GPT_intersect_sets_l1846_184668


namespace NUMINAMATH_GPT_maria_score_l1846_184691

theorem maria_score (m j : ℕ) (h1 : m = j + 50) (h2 : (m + j) / 2 = 112) : m = 137 :=
by
  sorry

end NUMINAMATH_GPT_maria_score_l1846_184691


namespace NUMINAMATH_GPT_pokemon_cards_per_friend_l1846_184676

theorem pokemon_cards_per_friend (total_cards : ℕ) (num_friends : ℕ) 
  (hc : total_cards = 56) (hf : num_friends = 4) : (total_cards / num_friends) = 14 := 
by
  sorry

end NUMINAMATH_GPT_pokemon_cards_per_friend_l1846_184676


namespace NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l1846_184659

theorem quadratic_has_distinct_real_roots :
  let a := 2
  let b := 3
  let c := -4
  (b^2 - 4 * a * c) > 0 := by
  sorry

end NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l1846_184659


namespace NUMINAMATH_GPT_monkey_farm_l1846_184687

theorem monkey_farm (x y : ℕ) 
  (h1 : y = 14 * x + 48) 
  (h2 : y = 18 * x - 64) : 
  x = 28 ∧ y = 440 := 
by 
  sorry

end NUMINAMATH_GPT_monkey_farm_l1846_184687


namespace NUMINAMATH_GPT_quadratic_roots_l1846_184657

theorem quadratic_roots (m x1 x2 : ℝ) (h1 : x1 + x2 = 1) (h2 : x1*x1 + m*x1 + 2*m = 0) (h3 : x2*x2 + m*x2 + 2*m = 0) : x1 * x2 = -2 := 
by sorry

end NUMINAMATH_GPT_quadratic_roots_l1846_184657


namespace NUMINAMATH_GPT_acute_angle_sine_l1846_184655
--import Lean library

-- Define the problem conditions and statement
theorem acute_angle_sine (a : ℝ) (h1 : 0 < a) (h2 : a < π / 2) (h3 : Real.sin a = 0.6) :
  π / 6 < a ∧ a < π / 4 :=
by 
  sorry

end NUMINAMATH_GPT_acute_angle_sine_l1846_184655


namespace NUMINAMATH_GPT_integer_a_satisfies_equation_l1846_184683

theorem integer_a_satisfies_equation (a b c : ℤ) :
  (∃ b c : ℤ, (x - a) * (x - 5) + 2 = (x + b) * (x + c)) → 
    a = 2 :=
by
  intro h_eq
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_integer_a_satisfies_equation_l1846_184683


namespace NUMINAMATH_GPT_rational_solution_cos_eq_l1846_184688

theorem rational_solution_cos_eq {q : ℚ} (h0 : 0 < q) (h1 : q < 1) (heq : Real.cos (3 * Real.pi * q) + 2 * Real.cos (2 * Real.pi * q) = 0) : 
  q = 2 / 3 := 
sorry

end NUMINAMATH_GPT_rational_solution_cos_eq_l1846_184688


namespace NUMINAMATH_GPT_fraction_problem_l1846_184649

theorem fraction_problem (a b c d e: ℚ) (val: ℚ) (h_a: a = 1/4) (h_b: b = 1/3) 
  (h_c: c = 1/6) (h_d: d = 1/8) (h_val: val = 72) :
  (a * b * c * val + d) = 9 / 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_fraction_problem_l1846_184649


namespace NUMINAMATH_GPT_sum_first_five_terms_arithmetic_sequence_l1846_184636

theorem sum_first_five_terms_arithmetic_sequence (a d : ℤ)
  (h1 : a + 5 * d = 10)
  (h2 : a + 6 * d = 15)
  (h3 : a + 7 * d = 20) :
  5 * (2 * a + (5 - 1) * d) / 2 = -25 := by
  sorry

end NUMINAMATH_GPT_sum_first_five_terms_arithmetic_sequence_l1846_184636


namespace NUMINAMATH_GPT_common_points_intervals_l1846_184601

noncomputable def h (x : ℝ) : ℝ := (2 * Real.log x) / x

theorem common_points_intervals (a : ℝ) (h₀ : 1 < a) : 
  (∀ f g : ℝ → ℝ, (f x = a ^ x) → (g x = x ^ 2) → 
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ = g x₁ ∧ f x₂ = g x₂ ∧ f x₃ = g x₃) → 
  a < Real.exp (2 / Real.exp 1) :=
by
  sorry

end NUMINAMATH_GPT_common_points_intervals_l1846_184601


namespace NUMINAMATH_GPT_library_books_l1846_184608

/-- Last year, the school library purchased 50 new books. 
    This year, it purchased 3 times as many books. 
    If the library had 100 books before it purchased new books last year,
    prove that the library now has 300 books in total. -/
theorem library_books (initial_books : ℕ) (last_year_books : ℕ) (multiplier : ℕ)
  (h1 : initial_books = 100) (h2 : last_year_books = 50) (h3 : multiplier = 3) :
  initial_books + last_year_books + (multiplier * last_year_books) = 300 := 
sorry

end NUMINAMATH_GPT_library_books_l1846_184608


namespace NUMINAMATH_GPT_cube_square_third_smallest_prime_l1846_184674

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

def third_smallest_prime := 5

noncomputable def cube (n : ℕ) : ℕ := n * n * n

noncomputable def square (n : ℕ) : ℕ := n * n

theorem cube_square_third_smallest_prime : cube (square third_smallest_prime) = 15625 := by
  have h1 : is_prime 2 := by sorry
  have h2 : is_prime 3 := by sorry
  have h3 : is_prime 5 := by sorry
  sorry

end NUMINAMATH_GPT_cube_square_third_smallest_prime_l1846_184674


namespace NUMINAMATH_GPT_conclusion_A_conclusion_B_conclusion_C1_conclusion_C2_l1846_184662

variable {r a b x1 y1 x2 y2 : ℝ} -- variables used in the problem

-- conditions
def circle1 : x1^2 + y1^2 = r^2 := sorry -- Circle C1 equation
def circle2 : (x1 + a)^2 + (y1 + b)^2 = r^2 := sorry -- Circle C2 equation
def r_positive : r > 0 := sorry -- r > 0
def not_both_zero : ¬ (a = 0 ∧ b = 0) := sorry -- a, b are not both zero
def distinct_points : x1 ≠ x2 ∧ y1 ≠ y2 := sorry -- A(x1, y1) and B(x2, y2) are distinct

-- Proofs to be provided for each of the conclusions
theorem conclusion_A : 2 * a * x1 + 2 * b * y1 + a^2 + b^2 = 0 := sorry
theorem conclusion_B : a * (x1 - x2) + b * (y1 - y2) = 0 := sorry
theorem conclusion_C1 : x1 + x2 = -a := sorry
theorem conclusion_C2 : y1 + y2 = -b := sorry

end NUMINAMATH_GPT_conclusion_A_conclusion_B_conclusion_C1_conclusion_C2_l1846_184662


namespace NUMINAMATH_GPT_number_of_arrangements_l1846_184619

theorem number_of_arrangements (P : Fin 5 → Type) (youngest : Fin 5) 
  (h_in_not_first_last : ∀ (i : Fin 5), i ≠ 0 → i ≠ 4 → i ≠ youngest) : 
  ∃ n, n = 72 := 
by
  sorry

end NUMINAMATH_GPT_number_of_arrangements_l1846_184619


namespace NUMINAMATH_GPT_rounded_product_less_than_original_l1846_184686

theorem rounded_product_less_than_original
  (x y z : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) (hxy : x > 2 * y) :
  (x + z) * (y - z) < x * y :=
by
  sorry

end NUMINAMATH_GPT_rounded_product_less_than_original_l1846_184686


namespace NUMINAMATH_GPT_area_of_quadrilateral_is_195_l1846_184629

-- Definitions and conditions
def diagonal_length : ℝ := 26
def offset1 : ℝ := 9
def offset2 : ℝ := 6

-- Prove the area of the quadrilateral is 195 cm²
theorem area_of_quadrilateral_is_195 :
  1 / 2 * diagonal_length * offset1 + 1 / 2 * diagonal_length * offset2 = 195 := 
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_area_of_quadrilateral_is_195_l1846_184629


namespace NUMINAMATH_GPT_expression_value_l1846_184627

theorem expression_value : 4 * (8 - 2) ^ 2 - 6 = 138 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l1846_184627


namespace NUMINAMATH_GPT_find_length_of_second_train_l1846_184605

def length_of_second_train (L : ℚ) : Prop :=
  let length_first_train : ℚ := 300
  let speed_first_train : ℚ := 120 * 1000 / 3600
  let speed_second_train : ℚ := 80 * 1000 / 3600
  let crossing_time : ℚ := 9
  let relative_speed : ℚ := speed_first_train + speed_second_train
  let total_distance : ℚ := relative_speed * crossing_time
  total_distance = length_first_train + L

theorem find_length_of_second_train :
  ∃ (L : ℚ), length_of_second_train L ∧ L = 199.95 := 
by
  sorry

end NUMINAMATH_GPT_find_length_of_second_train_l1846_184605


namespace NUMINAMATH_GPT_ratio_of_scores_l1846_184690

theorem ratio_of_scores 
  (u v : ℝ) 
  (h1 : u > v) 
  (h2 : u - v = (u + v) / 2) 
  : v / u = 1 / 3 :=
sorry

end NUMINAMATH_GPT_ratio_of_scores_l1846_184690


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1846_184665

variable (a : ℚ)
variable (a_val : a = -1/2)

theorem simplify_and_evaluate : (4 - 3 * a) * (1 + 2 * a) - 3 * a * (1 - 2 * a) = 3 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1846_184665


namespace NUMINAMATH_GPT_tina_coins_after_five_hours_l1846_184602

theorem tina_coins_after_five_hours :
  let coins_in_first_hour := 20
  let coins_in_second_hour := 30
  let coins_in_third_hour := 30
  let coins_in_fourth_hour := 40
  let coins_taken_out_in_fifth_hour := 20
  let total_coins_after_five_hours := coins_in_first_hour + coins_in_second_hour + coins_in_third_hour + coins_in_fourth_hour - coins_taken_out_in_fifth_hour
  total_coins_after_five_hours = 100 :=
by {
  sorry
}

end NUMINAMATH_GPT_tina_coins_after_five_hours_l1846_184602


namespace NUMINAMATH_GPT_pencil_notebook_cost_l1846_184695

variable {p n : ℝ}

theorem pencil_notebook_cost (hp1 : 9 * p + 11 * n = 6.05) (hp2 : 6 * p + 4 * n = 2.68) :
  18 * p + 13 * n = 8.45 :=
sorry

end NUMINAMATH_GPT_pencil_notebook_cost_l1846_184695


namespace NUMINAMATH_GPT_solution_l1846_184639

variable (f g : ℝ → ℝ)

open Real

-- Define f(x) and g(x) as given in the problem
def isSolution (x : ℝ) : Prop :=
  f x + g x = sqrt ((1 + cos (2 * x)) / (1 - sin x)) ∧
  (∀ x, f (-x) = -f x) ∧
  (∀ x, g (-x) = g x)

-- The theorem we want to prove
theorem solution (x : ℝ) (hx : -π / 2 < x ∧ x < π / 2)
  (h : isSolution f g x) : (f x)^2 - (g x)^2 = -2 * cos x := 
sorry

end NUMINAMATH_GPT_solution_l1846_184639


namespace NUMINAMATH_GPT_find_Sn_find_Tn_l1846_184680

def Sn (n : ℕ) : ℕ := n^2 + n

def Tn (n : ℕ) : ℚ := (n : ℚ) / (n + 1)

section
variables {a₁ d : ℕ}

-- Given conditions
axiom S5 : 5 * a₁ + 10 * d = 30
axiom S10 : 10 * a₁ + 45 * d = 110

-- Problem statement 1
theorem find_Sn (n : ℕ) : Sn n = n^2 + n :=
sorry

-- Problem statement 2
theorem find_Tn (n : ℕ) : Tn n = (n : ℚ) / (n + 1) :=
sorry

end

end NUMINAMATH_GPT_find_Sn_find_Tn_l1846_184680


namespace NUMINAMATH_GPT_h_two_n_mul_h_2024_l1846_184618

variable {h : ℕ → ℝ}
variable {k : ℝ}
variable (n : ℕ) (k_ne_zero : k ≠ 0)

-- Condition 1: h(m + n) = h(m) * h(n)
axiom h_add_mul (m n : ℕ) : h (m + n) = h m * h n

-- Condition 2: h(2) = k
axiom h_two : h 2 = k

theorem h_two_n_mul_h_2024 : h (2 * n) * h 2024 = k^(n + 1012) := 
  sorry

end NUMINAMATH_GPT_h_two_n_mul_h_2024_l1846_184618


namespace NUMINAMATH_GPT_sin_sum_of_roots_l1846_184600

theorem sin_sum_of_roots (x1 x2 m : ℝ) (hx1 : 0 ≤ x1 ∧ x1 ≤ π) (hx2 : 0 ≤ x2 ∧ x2 ≤ π)
    (hroot1 : 2 * Real.sin x1 + Real.cos x1 = m) (hroot2 : 2 * Real.sin x2 + Real.cos x2 = m) :
    Real.sin (x1 + x2) = 4 / 5 := 
sorry

end NUMINAMATH_GPT_sin_sum_of_roots_l1846_184600


namespace NUMINAMATH_GPT_set_nonempty_iff_nonneg_l1846_184670

theorem set_nonempty_iff_nonneg (a : ℝ) :
  (∃ x : ℝ, x^2 ≤ a) ↔ a ≥ 0 :=
sorry

end NUMINAMATH_GPT_set_nonempty_iff_nonneg_l1846_184670


namespace NUMINAMATH_GPT_inequality_proof_l1846_184617

theorem inequality_proof (x y : ℝ) (h : 2 * y + 5 * x = 10) : (3 * x * y - x^2 - y^2 < 7) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1846_184617


namespace NUMINAMATH_GPT_smallest_x_solution_l1846_184671

theorem smallest_x_solution :
  (∃ x : ℝ, (3 * x^2 + 36 * x - 90 = 2 * x * (x + 16)) ∧ ∀ y : ℝ, (3 * y^2 + 36 * y - 90 = 2 * y * (y + 16)) → x ≤ y) ↔ x = -10 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_solution_l1846_184671


namespace NUMINAMATH_GPT_find_multiple_of_diff_l1846_184679

theorem find_multiple_of_diff (n sum diff remainder k : ℕ) 
  (hn : n = 220070) 
  (hs : sum = 555 + 445) 
  (hd : diff = 555 - 445)
  (hr : remainder = 70)
  (hmod : n % sum = remainder) 
  (hquot : n / sum = k) :
  ∃ k, k = 2 ∧ k * diff = n / sum := 
by 
  sorry

end NUMINAMATH_GPT_find_multiple_of_diff_l1846_184679


namespace NUMINAMATH_GPT_sum_2016_eq_1008_l1846_184613

-- Define the arithmetic sequence {a_n} and the sum of the first n terms S_n
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Given conditions
variable (h_arith_seq : ∀ n m, a (n+1) - a n = a (m+1) - a m)
variable (h_sum : ∀ n, S n = (n * (a 1 + a n)) / 2)

-- Additional conditions from the problem
variable (h_vector : a 4 + a 2013 = 1)

-- Goal: Prove that the sum of the first 2016 terms equals 1008
theorem sum_2016_eq_1008 : S 2016 = 1008 := by
  sorry

end NUMINAMATH_GPT_sum_2016_eq_1008_l1846_184613


namespace NUMINAMATH_GPT_inequality_proof_l1846_184663

theorem inequality_proof (a b c d : ℝ) : 
  (a^2 + b^2 + 1) * (c^2 + d^2 + 1) ≥ 2 * (a + c) * (b + d) :=
by sorry

end NUMINAMATH_GPT_inequality_proof_l1846_184663


namespace NUMINAMATH_GPT_product_odd_integers_lt_20_l1846_184694

/--
The product of all odd positive integers strictly less than 20 is a positive number ending with the digit 5.
-/
theorem product_odd_integers_lt_20 :
  let nums := [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
  let product := List.prod nums
  (product > 0) ∧ (product % 10 = 5) :=
by
  sorry

end NUMINAMATH_GPT_product_odd_integers_lt_20_l1846_184694


namespace NUMINAMATH_GPT_inequality_holds_l1846_184675

theorem inequality_holds (k : ℝ) : (∀ x : ℝ, x^2 + k * x + 1 > 0) ↔ (k > -2 ∧ k < 2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l1846_184675


namespace NUMINAMATH_GPT_total_homework_pages_l1846_184634

theorem total_homework_pages (R : ℕ) (H1 : R + 3 = 8) : R + (R + 3) = 13 :=
by sorry

end NUMINAMATH_GPT_total_homework_pages_l1846_184634


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1846_184697

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ)
  (h1 : ∀ n, S n = n * ((a 1 + a n) / 2))
  (h2 : S 9 = 27) :
  a 4 + a 6 = 6 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1846_184697


namespace NUMINAMATH_GPT_female_democrats_l1846_184667

theorem female_democrats (F M : ℕ) (h1 : F + M = 840) (h2 : F / 2 + M / 4 = 280) : F / 2 = 140 :=
by 
  sorry

end NUMINAMATH_GPT_female_democrats_l1846_184667


namespace NUMINAMATH_GPT_total_difference_in_cents_l1846_184644

variable (q : ℕ)

def charles_quarters := 6 * q + 2
def charles_dimes := 3 * q - 2

def richard_quarters := 2 * q + 10
def richard_dimes := 4 * q + 3

def cents_from_quarters (n : ℕ) : ℕ := 25 * n
def cents_from_dimes (n : ℕ) : ℕ := 10 * n

theorem total_difference_in_cents : 
  (cents_from_quarters (charles_quarters q) + cents_from_dimes (charles_dimes q)) - 
  (cents_from_quarters (richard_quarters q) + cents_from_dimes (richard_dimes q)) = 
  90 * q - 250 :=
by
  sorry

end NUMINAMATH_GPT_total_difference_in_cents_l1846_184644


namespace NUMINAMATH_GPT_Ellen_strawberries_used_l1846_184628

theorem Ellen_strawberries_used :
  let yogurt := 0.1
  let orange_juice := 0.2
  let total_ingredients := 0.5
  let strawberries := total_ingredients - (yogurt + orange_juice)
  strawberries = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_Ellen_strawberries_used_l1846_184628


namespace NUMINAMATH_GPT_contrapositive_example_l1846_184685

theorem contrapositive_example (x : ℝ) : (x > 1 → x^2 > 1) → (x^2 ≤ 1 → x ≤ 1) :=
sorry

end NUMINAMATH_GPT_contrapositive_example_l1846_184685


namespace NUMINAMATH_GPT_angle_quadrant_l1846_184610

theorem angle_quadrant 
  (θ : Real) 
  (h1 : Real.cos θ > 0) 
  (h2 : Real.sin (2 * θ) < 0) : 
  3 * π / 2 < θ ∧ θ < 2 * π := 
by
  sorry

end NUMINAMATH_GPT_angle_quadrant_l1846_184610


namespace NUMINAMATH_GPT_luxury_class_adults_l1846_184678

def total_passengers : ℕ := 300
def adult_percentage : ℝ := 0.70
def luxury_percentage : ℝ := 0.15

def total_adults (p : ℕ) : ℕ := (p * 70) / 100
def adults_in_luxury (a : ℕ) : ℕ := (a * 15) / 100

theorem luxury_class_adults :
  adults_in_luxury (total_adults total_passengers) = 31 :=
by
  sorry

end NUMINAMATH_GPT_luxury_class_adults_l1846_184678


namespace NUMINAMATH_GPT_find_amount_l1846_184622

def total_amount (A : ℝ) : Prop :=
  A / 20 = A / 25 + 100

theorem find_amount 
  (A : ℝ) 
  (h : total_amount A) : 
  A = 10000 := 
  sorry

end NUMINAMATH_GPT_find_amount_l1846_184622


namespace NUMINAMATH_GPT_grade_assignment_ways_l1846_184621

theorem grade_assignment_ways : (4 ^ 12) = 16777216 := by
  sorry

end NUMINAMATH_GPT_grade_assignment_ways_l1846_184621


namespace NUMINAMATH_GPT_bread_pieces_total_l1846_184698

def initial_slices : ℕ := 2
def pieces_per_slice (n : ℕ) : ℕ := n * 4

theorem bread_pieces_total : pieces_per_slice initial_slices = 8 :=
by
  sorry

end NUMINAMATH_GPT_bread_pieces_total_l1846_184698


namespace NUMINAMATH_GPT_initial_juggling_objects_l1846_184640

theorem initial_juggling_objects (x : ℕ) : (∀ i : ℕ, i = 5 → x + 2*i = 13) → x = 3 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_initial_juggling_objects_l1846_184640
