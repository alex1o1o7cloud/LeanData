import Mathlib

namespace NUMINAMATH_GPT_sqrt_meaningful_range_l623_62399

theorem sqrt_meaningful_range (x : ℝ) (h : 3 * x - 5 ≥ 0) : x ≥ 5 / 3 :=
sorry

end NUMINAMATH_GPT_sqrt_meaningful_range_l623_62399


namespace NUMINAMATH_GPT_price_of_sugar_and_salt_l623_62372

theorem price_of_sugar_and_salt:
  (∀ (sugar_price salt_price : ℝ), 2 * sugar_price + 5 * salt_price = 5.50 ∧ sugar_price = 1.50 →
  3 * sugar_price + salt_price = 5) := 
by 
  sorry

end NUMINAMATH_GPT_price_of_sugar_and_salt_l623_62372


namespace NUMINAMATH_GPT_arithmetic_sqrt_9_l623_62396

theorem arithmetic_sqrt_9 :
  (∃ y : ℝ, y * y = 9 ∧ y ≥ 0) → (∃ y : ℝ, y = 3) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sqrt_9_l623_62396


namespace NUMINAMATH_GPT_simple_interest_initial_amount_l623_62384

theorem simple_interest_initial_amount :
  ∃ P : ℝ, (P + P * 0.04 * 5 = 900) ∧ P = 750 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_initial_amount_l623_62384


namespace NUMINAMATH_GPT_sandy_correct_sums_l623_62308

/-- 
Sandy gets 3 marks for each correct sum and loses 2 marks for each incorrect sum.
Sandy attempts 50 sums and obtains 100 marks within a 45-minute time constraint.
If Sandy receives a 1-mark penalty for each sum not completed within the time limit,
prove that the number of correct sums Sandy got is 25.
-/
theorem sandy_correct_sums (c i : ℕ) (h1 : c + i = 50) (h2 : 3 * c - 2 * i - (50 - c) = 100) : c = 25 :=
by
  sorry

end NUMINAMATH_GPT_sandy_correct_sums_l623_62308


namespace NUMINAMATH_GPT_speed_of_stream_l623_62391

theorem speed_of_stream (v : ℝ) (h_still : ∀ (d : ℝ), d / (3 - v) = 2 * d / (3 + v)) : v = 1 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_stream_l623_62391


namespace NUMINAMATH_GPT_question1_question2_l623_62335

/-
In ΔABC, the sides opposite to angles A, B, and C are respectively a, b, and c.
It is given that b + c = 2 * a * cos B.

(1) Prove that A = 2B;
(2) If the area of ΔABC is S = a^2 / 4, find the magnitude of angle A.
-/

variables {A B C a b c : ℝ}
variables {S : ℝ}

-- Condition given in the problem
axiom h1 : b + c = 2 * a * Real.cos B
axiom h2 : 1 / 2 * b * c * Real.sin A = a^2 / 4

-- Question 1: Prove that A = 2 * B
theorem question1 (h1 : b + c = 2 * a * Real.cos B) : A = 2 * B := sorry

-- Question 2: Find the magnitude of angle A
theorem question2 (h2 : 1 / 2 * b * c * Real.sin A = a^2 / 4) : A = 90 ∨ A = 45 := sorry

end NUMINAMATH_GPT_question1_question2_l623_62335


namespace NUMINAMATH_GPT_smallest_sum_of_squares_l623_62358

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 221) : ∃ (x' y' : ℤ), x'^2 - y'^2 = 221 ∧ x'^2 + y'^2 ≤ x^2 + y^2 ∧ x'^2 + y'^2 = 229 :=
by
  -- Conditions and remaining goals to be proved
  sorry

end NUMINAMATH_GPT_smallest_sum_of_squares_l623_62358


namespace NUMINAMATH_GPT_locus_of_point_P_l623_62337

theorem locus_of_point_P (x y : ℝ) :
  let M := (-2, 0)
  let N := (2, 0)
  (x^2 + y^2 = 4 ∧ x ≠ 2 ∧ x ≠ -2) ↔ 
  ((x + 2)^2 + y^2 + (x - 2)^2 + y^2 = 16 ∧ x ≠ 2 ∧ x ≠ -2) :=
by
  sorry 

end NUMINAMATH_GPT_locus_of_point_P_l623_62337


namespace NUMINAMATH_GPT_GCF_of_48_180_98_l623_62347

theorem GCF_of_48_180_98 : Nat.gcd (Nat.gcd 48 180) 98 = 2 :=
by
  sorry

end NUMINAMATH_GPT_GCF_of_48_180_98_l623_62347


namespace NUMINAMATH_GPT_typing_page_percentage_l623_62359

/--
Given:
- Original sheet dimensions are 20 cm by 30 cm.
- Margins are 2 cm on each side (left and right), and 3 cm on the top and bottom.
Prove that the percentage of the page used by the typist is 64%.
-/
theorem typing_page_percentage (width height margin_lr margin_tb : ℝ)
  (h1 : width = 20) 
  (h2 : height = 30) 
  (h3 : margin_lr = 2) 
  (h4 : margin_tb = 3) : 
  (width - 2 * margin_lr) * (height - 2 * margin_tb) / (width * height) * 100 = 64 :=
by
  sorry

end NUMINAMATH_GPT_typing_page_percentage_l623_62359


namespace NUMINAMATH_GPT_experiment_variance_l623_62362

noncomputable def probability_of_success : ℚ := 5/9

noncomputable def variance_of_binomial (n : ℕ) (p : ℚ) : ℚ :=
  n * p * (1 - p)

def number_of_experiments : ℕ := 30

theorem experiment_variance :
  variance_of_binomial number_of_experiments probability_of_success = 200/27 :=
by
  sorry

end NUMINAMATH_GPT_experiment_variance_l623_62362


namespace NUMINAMATH_GPT_prank_combinations_l623_62355

-- Conditions stated as definitions
def monday_choices : ℕ := 1
def tuesday_choices : ℕ := 3
def wednesday_choices : ℕ := 5
def thursday_choices : ℕ := 6
def friday_choices : ℕ := 2

-- Theorem to prove
theorem prank_combinations :
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices = 180 :=
by
  sorry

end NUMINAMATH_GPT_prank_combinations_l623_62355


namespace NUMINAMATH_GPT_solution_1_solution_2_l623_62389

noncomputable def problem_1 : Real :=
  Real.log 25 + Real.log 2 * Real.log 50 + (Real.log 2)^2

noncomputable def problem_2 : Real :=
  (Real.logb 3 2 + Real.logb 9 2) * (Real.logb 4 3 + Real.logb 8 3)

theorem solution_1 : problem_1 = 2 := by
  sorry

theorem solution_2 : problem_2 = 5 / 4 := by
  sorry

end NUMINAMATH_GPT_solution_1_solution_2_l623_62389


namespace NUMINAMATH_GPT_area_ratio_correct_l623_62319

noncomputable def area_ratio_of_ABC_and_GHJ : ℝ :=
  let side_length_ABC := 12
  let BD := 5
  let CE := 5
  let AF := 8
  let area_ABC := (Real.sqrt 3 / 4) * side_length_ABC ^ 2
  (1 / 74338) * area_ABC / area_ABC

theorem area_ratio_correct : area_ratio_of_ABC_and_GHJ = 1 / 74338 := by
  sorry

end NUMINAMATH_GPT_area_ratio_correct_l623_62319


namespace NUMINAMATH_GPT_rational_relation_l623_62320

variable {a b : ℚ}

theorem rational_relation (h1 : a > 0) (h2 : b < 0) (h3 : |a| > |b|) : -a < -b ∧ -b < b ∧ b < a :=
by
  sorry

end NUMINAMATH_GPT_rational_relation_l623_62320


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l623_62351

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (∀ p : ℝ, (p = x1 ∨ p = x2) → (p ^ 2 + (4 * m + 1) * p + m = 0)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l623_62351


namespace NUMINAMATH_GPT_steak_chicken_ratio_l623_62383

variable (S C : ℕ)

theorem steak_chicken_ratio (h1 : S + C = 80) (h2 : 25 * S + 18 * C = 1860) : S = 3 * C :=
by
  sorry

end NUMINAMATH_GPT_steak_chicken_ratio_l623_62383


namespace NUMINAMATH_GPT_volume_ratio_of_cubes_l623_62366

theorem volume_ratio_of_cubes (s2 : ℝ) : 
  let s1 := s2 * (Real.sqrt 3)
  let V1 := s1^3
  let V2 := s2^3
  V1 / V2 = 3 * (Real.sqrt 3) :=
by
  admit -- si



end NUMINAMATH_GPT_volume_ratio_of_cubes_l623_62366


namespace NUMINAMATH_GPT_expected_value_of_winnings_is_4_l623_62303

noncomputable def expected_value_of_winnings : ℕ := 
  let outcomes := [7, 6, 5, 4, 4, 3, 2, 1]
  let total_winnings := outcomes.sum
  total_winnings / 8

theorem expected_value_of_winnings_is_4 :
  expected_value_of_winnings = 4 :=
by
  sorry

end NUMINAMATH_GPT_expected_value_of_winnings_is_4_l623_62303


namespace NUMINAMATH_GPT_find_x_l623_62328

def balanced (a b c d : ℝ) : Prop :=
  a + b + c + d = a^2 + b^2 + c^2 + d^2

theorem find_x (x : ℝ) : 
  (∀ a b c d : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ balanced a b c d → (x - a) * (x - b) * (x - c) * (x - d) ≥ 0) ↔ x ≥ 3 / 2 :=
sorry

end NUMINAMATH_GPT_find_x_l623_62328


namespace NUMINAMATH_GPT_students_behind_yoongi_l623_62330

theorem students_behind_yoongi (total_students : ℕ) (jungkook_position : ℕ) (yoongi_position : ℕ) (behind_students : ℕ)
  (h1 : total_students = 20)
  (h2 : jungkook_position = 3)
  (h3 : yoongi_position = jungkook_position + 1)
  (h4 : behind_students = total_students - yoongi_position) :
  behind_students = 16 :=
by
  sorry

end NUMINAMATH_GPT_students_behind_yoongi_l623_62330


namespace NUMINAMATH_GPT_fraction_meaningful_l623_62370

theorem fraction_meaningful (x : ℝ) : (x ≠ 1) ↔ ¬ (x - 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_l623_62370


namespace NUMINAMATH_GPT_candy_bar_cost_l623_62329

-- Define the conditions
def cost_gum_over_candy_bar (C G : ℝ) : Prop :=
  G = (1/2) * C

def total_cost (C G : ℝ) : Prop :=
  2 * G + 3 * C = 6

-- Define the proof problem
theorem candy_bar_cost (C G : ℝ) (h1 : cost_gum_over_candy_bar C G) (h2 : total_cost C G) : C = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_candy_bar_cost_l623_62329


namespace NUMINAMATH_GPT_solve_quadratic_eq_l623_62397

theorem solve_quadratic_eq (x : ℝ) : x^2 - x - 2 = 0 ↔ x = 2 ∨ x = -1 :=
by sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l623_62397


namespace NUMINAMATH_GPT_correct_relation_is_identity_l623_62326

theorem correct_relation_is_identity : 0 = 0 :=
by {
  -- Skipping proof steps as only statement is required
  sorry
}

end NUMINAMATH_GPT_correct_relation_is_identity_l623_62326


namespace NUMINAMATH_GPT_prob_not_lose_money_proof_min_purchase_price_proof_l623_62310

noncomputable def prob_not_lose_money : ℚ :=
  let pr_normal_rain := (2 : ℚ) / 3
  let pr_less_rain := (1 : ℚ) / 3
  let pr_price_6_normal := (1 : ℚ) / 4
  let pr_price_6_less := (2 : ℚ) / 3
  pr_normal_rain * pr_price_6_normal + pr_less_rain * pr_price_6_less

theorem prob_not_lose_money_proof : prob_not_lose_money = 7 / 18 := sorry

noncomputable def min_purchase_price : ℚ :=
  let old_exp_income := 500
  let new_yield := 2500
  let cost_increase := 1000
  (7000 + 1500 + cost_increase) / new_yield
  
theorem min_purchase_price_proof : min_purchase_price = 3.4 := sorry

end NUMINAMATH_GPT_prob_not_lose_money_proof_min_purchase_price_proof_l623_62310


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_product_l623_62314

theorem arithmetic_geometric_sequence_product :
  (∀ n : ℕ, ∃ d : ℝ, ∀ m : ℕ, a_n = a_1 + m * d) →
  (∀ n : ℕ, ∃ q : ℝ, ∀ m : ℕ, b_n = b_1 * q ^ m) →
  a_1 = 1 → a_2 = 2 →
  b_1 = 1 → b_2 = 2 →
  a_5 * b_5 = 80 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_product_l623_62314


namespace NUMINAMATH_GPT_digit_B_identification_l623_62317

theorem digit_B_identification (B : ℕ) 
  (hB_range : 0 ≤ B ∧ B < 10) 
  (h_units_digit : (5 * B % 10) = 5) 
  (h_product : (10 * B + 5) * (90 + B) = 9045) : 
  B = 9 :=
sorry

end NUMINAMATH_GPT_digit_B_identification_l623_62317


namespace NUMINAMATH_GPT_value_of_b_l623_62350

theorem value_of_b (y b : ℝ) (hy : y > 0) (h : (4 * y) / b + (3 * y) / 10 = 0.5 * y) : b = 20 :=
by
  -- Proof omitted for brevity
  sorry

end NUMINAMATH_GPT_value_of_b_l623_62350


namespace NUMINAMATH_GPT_exponent_equivalence_l623_62387

open Real

theorem exponent_equivalence (a : ℝ) (h : a > 0) : 
  (a^2 / (sqrt a * a^(2/3))) = a^(5/6) :=
  sorry

end NUMINAMATH_GPT_exponent_equivalence_l623_62387


namespace NUMINAMATH_GPT_min_frac_sum_l623_62322

theorem min_frac_sum (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_eq : 2 * a + b = 1) : 
  (3 / b + 2 / a) = 7 + 4 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_min_frac_sum_l623_62322


namespace NUMINAMATH_GPT_sticks_left_in_yard_l623_62321

def number_of_sticks_picked_up : Nat := 14
def difference_between_picked_and_left : Nat := 10

theorem sticks_left_in_yard 
  (picked_up : Nat := number_of_sticks_picked_up)
  (difference : Nat := difference_between_picked_and_left) 
  : Nat :=
  picked_up - difference

example : sticks_left_in_yard = 4 := by 
  sorry

end NUMINAMATH_GPT_sticks_left_in_yard_l623_62321


namespace NUMINAMATH_GPT_largest_positive_integer_divisible_l623_62346

theorem largest_positive_integer_divisible (n : ℕ) :
  (n + 20 ∣ n^3 - 100) ↔ n = 2080 :=
sorry

end NUMINAMATH_GPT_largest_positive_integer_divisible_l623_62346


namespace NUMINAMATH_GPT_find_middle_number_l623_62301

theorem find_middle_number (a : Fin 11 → ℝ)
  (h1 : ∀ i : Fin 9, a i + a (⟨i.1 + 1, by linarith [i.2]⟩) + a (⟨i.1 + 2, by linarith [i.2]⟩) = 18)
  (h2 : (Finset.univ.sum a) = 64) :
  a 5 = 8 := 
by
  sorry

end NUMINAMATH_GPT_find_middle_number_l623_62301


namespace NUMINAMATH_GPT_cost_of_jam_l623_62393

theorem cost_of_jam (N B J H : ℕ) (h : N > 1) (cost_eq : N * (6 * B + 7 * J + 4 * H) = 462) : 7 * J * N = 462 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_jam_l623_62393


namespace NUMINAMATH_GPT_sum_of_roots_eq_three_l623_62345

theorem sum_of_roots_eq_three (x1 x2 : ℝ) 
  (h1 : x1^2 - 3*x1 + 2 = 0)
  (h2 : x2^2 - 3*x2 + 2 = 0) 
  (h3 : x1 ≠ x2) : 
  x1 + x2 = 3 := 
sorry

end NUMINAMATH_GPT_sum_of_roots_eq_three_l623_62345


namespace NUMINAMATH_GPT_total_cups_l623_62313

theorem total_cups (t1 t2 : ℕ) (h1 : t2 = 240) (h2 : t2 = t1 - 20) : t1 + t2 = 500 := by
  sorry

end NUMINAMATH_GPT_total_cups_l623_62313


namespace NUMINAMATH_GPT_smallest_four_digit_divisible_by_53_l623_62309

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_divisible_by_53_l623_62309


namespace NUMINAMATH_GPT_brick_length_l623_62371

theorem brick_length (L : ℝ) :
  (∀ (V_wall V_brick : ℝ),
    V_wall = 29 * 100 * 2 * 100 * 0.75 * 100 ∧
    V_wall = 29000 * V_brick ∧
    V_brick = L * 10 * 7.5) →
  L = 20 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_brick_length_l623_62371


namespace NUMINAMATH_GPT_train_length_l623_62380

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (speed_conversion : speed_kmh = 40) 
  (time_condition : time_s = 27) : 
  (speed_kmh * 1000 / 3600 * time_s = 300) := 
by
  sorry

end NUMINAMATH_GPT_train_length_l623_62380


namespace NUMINAMATH_GPT_find_n_l623_62327

theorem find_n (x : ℝ) (h1 : x = 4.0) (h2 : 3 * x + n = 48) : n = 36 := by
  sorry

end NUMINAMATH_GPT_find_n_l623_62327


namespace NUMINAMATH_GPT_clothing_discount_l623_62395

theorem clothing_discount (P : ℝ) :
  let first_sale_price := (4 / 5) * P
  let second_sale_price := first_sale_price * 0.60
  second_sale_price = (12 / 25) * P :=
by
  sorry

end NUMINAMATH_GPT_clothing_discount_l623_62395


namespace NUMINAMATH_GPT_ratio_accepted_rejected_l623_62312

-- Definitions for the conditions given
def eggs_per_day : ℕ := 400
def ratio_accepted_to_rejected : ℕ × ℕ := (96, 4)
def additional_accepted_eggs : ℕ := 12

/-- The ratio of accepted eggs to rejected eggs on that particular day is 99:1. -/
theorem ratio_accepted_rejected (a r : ℕ) (h1 : ratio_accepted_to_rejected = (a, r)) 
  (h2 : (a + r) * (eggs_per_day / (a + r)) = eggs_per_day) 
  (h3 : additional_accepted_eggs = 12) :
  (a + additional_accepted_eggs) / r = 99 :=
  sorry

end NUMINAMATH_GPT_ratio_accepted_rejected_l623_62312


namespace NUMINAMATH_GPT_box_growth_factor_l623_62352

/-
Problem: When a large box in the shape of a cuboid measuring 6 centimeters (cm) wide,
4 centimeters (cm) long, and 1 centimeters (cm) high became larger into a volume of
30 centimeters (cm) wide, 20 centimeters (cm) long, and 5 centimeters (cm) high,
find how many times it has grown.
-/

def original_box_volume (w l h : ℕ) : ℕ := w * l * h
def larger_box_volume (w l h : ℕ) : ℕ := w * l * h

theorem box_growth_factor :
  original_box_volume 6 4 1 * 125 = larger_box_volume 30 20 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_box_growth_factor_l623_62352


namespace NUMINAMATH_GPT_harry_travel_time_l623_62382

variables (bus_time1 bus_time2 : ℕ) (walk_ratio : ℕ)
-- Conditions based on the problem
-- Harry has already been sat on the bus for 15 minutes.
def part1_time : ℕ := 15

-- and he knows the rest of the journey will take another 25 minutes.
def part2_time : ℕ := 25

-- The total bus journey time
def total_bus_time : ℕ := part1_time + part2_time

-- The walk from the bus stop to his house will take half the amount of time the bus journey took.
def walk_time : ℕ := total_bus_time / 2

-- Total travel time
def total_travel_time : ℕ := total_bus_time + walk_time

-- Rewrite the proof problem statement
theorem harry_travel_time : total_travel_time = 60 := by
  sorry

end NUMINAMATH_GPT_harry_travel_time_l623_62382


namespace NUMINAMATH_GPT_roses_in_vase_l623_62302

/-- There were initially 16 roses and 3 orchids in the vase.
    Jessica cut 8 roses and 8 orchids from her garden.
    There are now 7 orchids in the vase.
    Prove that the number of roses in the vase now is 24. -/
theorem roses_in_vase
  (initial_roses initial_orchids : ℕ)
  (cut_roses cut_orchids remaining_orchids final_roses : ℕ)
  (h_initial: initial_roses = 16)
  (h_initial_orchids: initial_orchids = 3)
  (h_cut: cut_roses = 8 ∧ cut_orchids = 8)
  (h_remaining_orchids: remaining_orchids = 7)
  (h_orchids_relation: initial_orchids + cut_orchids = remaining_orchids + cut_orchids - 4)
  : final_roses = initial_roses + cut_roses := by
  sorry

end NUMINAMATH_GPT_roses_in_vase_l623_62302


namespace NUMINAMATH_GPT_problem_solution_l623_62386

noncomputable def f (x : ℝ) : ℝ := if 0 ≤ x ∧ x ≤ 1 then x^2 else sorry

lemma f_odd (x : ℝ) : f (-x) = -f x := sorry

lemma f_xplus1_even (x : ℝ) : f (x + 1) = f (-x + 1) := sorry

theorem problem_solution : f 2015 = -1 := 
by 
  sorry

end NUMINAMATH_GPT_problem_solution_l623_62386


namespace NUMINAMATH_GPT_harry_apples_l623_62357

theorem harry_apples (martha_apples : ℕ) (tim_apples : ℕ) (harry_apples : ℕ)
  (h1 : martha_apples = 68)
  (h2 : tim_apples = martha_apples - 30)
  (h3 : harry_apples = tim_apples / 2) :
  harry_apples = 19 := 
by sorry

end NUMINAMATH_GPT_harry_apples_l623_62357


namespace NUMINAMATH_GPT_arithmetic_series_sum_after_multiplication_l623_62368

theorem arithmetic_series_sum_after_multiplication :
  let s : List ℕ := [110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
  3 * s.sum = 3435 := by
  sorry

end NUMINAMATH_GPT_arithmetic_series_sum_after_multiplication_l623_62368


namespace NUMINAMATH_GPT_combined_tax_rate_l623_62390

theorem combined_tax_rate (Mork_income Mindy_income : ℝ) (h1 : Mindy_income = 4 * Mork_income) :
  let Mork_tax := 0.45 * Mork_income;
  let Mindy_tax := 0.15 * Mindy_income;
  let combined_tax := Mork_tax + Mindy_tax;
  let combined_income := Mork_income + Mindy_income;
  combined_tax / combined_income * 100 = 21 := 
by
  sorry

end NUMINAMATH_GPT_combined_tax_rate_l623_62390


namespace NUMINAMATH_GPT_value_of_b_l623_62304

variable (a b c y1 y2 : ℝ)

def equation1 := (y1 = 4 * a + 2 * b + c)
def equation2 := (y2 = 4 * a - 2 * b + c)
def difference := (y1 - y2 = 8)

theorem value_of_b 
  (h1 : equation1 a b c y1)
  (h2 : equation2 a b c y2)
  (h3 : difference y1 y2) : 
  b = 2 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_b_l623_62304


namespace NUMINAMATH_GPT_sequence_sixth_term_is_364_l623_62367

theorem sequence_sixth_term_is_364 (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 2 = 7) (h3 : a 3 = 20)
  (h4 : ∀ n, a (n + 1) = 1 / 3 * (a n + a (n + 2))) :
  a 6 = 364 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_sequence_sixth_term_is_364_l623_62367


namespace NUMINAMATH_GPT_range_of_m_l623_62307

theorem range_of_m (x y : ℝ) (m : ℝ) (hx : x > 0) (hy : y > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → (2 * y / x + 8 * x / y > m^2 + 2 * m)) → -4 < m ∧ m < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l623_62307


namespace NUMINAMATH_GPT_folded_segment_square_length_eq_225_div_4_l623_62300

noncomputable def square_of_fold_length : ℝ :=
  let side_length := 15
  let distance_from_B := 5
  (side_length ^ 2 - distance_from_B * (2 * side_length - distance_from_B)) / 4

theorem folded_segment_square_length_eq_225_div_4 :
  square_of_fold_length = 225 / 4 :=
by
  sorry

end NUMINAMATH_GPT_folded_segment_square_length_eq_225_div_4_l623_62300


namespace NUMINAMATH_GPT_document_total_characters_l623_62344

theorem document_total_characters (T : ℕ) : 
  (∃ (t_1 t_2 t_3 : ℕ) (v_A v_B : ℕ),
      v_A = 100 ∧ v_B = 200 ∧
      t_1 = T / 600 ∧
      v_A * t_1 = T / 6 ∧
      v_B * t_1 = T / 3 ∧
      v_A * 3 * 5 = 1500 ∧
      t_2 = (T / 2 - 1500) / 500 ∧
      (v_A * 3 * t_2 + 1500 + v_A * t_1 = v_B * t_1 + v_B * t_2) ∧
      (v_A * 3 * (T - 3000) / 1000 + 1500 + v_A * T / 6 =
       v_B * 2 * (T - 3000) / 10 + v_B * T / 3)) →
  T = 18000 := by
  sorry

end NUMINAMATH_GPT_document_total_characters_l623_62344


namespace NUMINAMATH_GPT_arithmetic_mean_of_18_27_45_l623_62356

theorem arithmetic_mean_of_18_27_45 : (18 + 27 + 45) / 3 = 30 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_18_27_45_l623_62356


namespace NUMINAMATH_GPT_ratio_female_to_male_l623_62331

namespace DeltaSportsClub

variables (f m : ℕ) -- number of female and male members
-- Sum of ages of female and male members respectively
def sum_ages_females := 35 * f
def sum_ages_males := 30 * m
-- Total sum of ages
def total_sum_ages := sum_ages_females f + sum_ages_males m
-- Total number of members
def total_members := f + m

-- Given condition on the average age of all members
def average_age_condition := (total_sum_ages f m) / (total_members f m) = 32

-- The target theorem to prove the ratio of female to male members
theorem ratio_female_to_male (h : average_age_condition f m) : f/m = 2/3 :=
by sorry

end DeltaSportsClub

end NUMINAMATH_GPT_ratio_female_to_male_l623_62331


namespace NUMINAMATH_GPT_product_of_0_5_and_0_8_l623_62343

theorem product_of_0_5_and_0_8 : (0.5 * 0.8) = 0.4 := by
  sorry

end NUMINAMATH_GPT_product_of_0_5_and_0_8_l623_62343


namespace NUMINAMATH_GPT_prop_range_a_l623_62361

theorem prop_range_a (a : ℝ) 
  (p : ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 4 → x^2 ≥ a)
  (q : ∃ (x : ℝ), x^2 + 2 * a * x + (2 - a) = 0)
  : a = 1 ∨ a ≤ -2 :=
sorry

end NUMINAMATH_GPT_prop_range_a_l623_62361


namespace NUMINAMATH_GPT_arithmetic_sequence_length_l623_62364

theorem arithmetic_sequence_length : 
  let a := 11
  let d := 5
  let l := 101
  ∃ n : ℕ, a + (n-1) * d = l ∧ n = 19 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_length_l623_62364


namespace NUMINAMATH_GPT_intersection_of_sets_l623_62373

variable (M : Set ℤ) (N : Set ℤ)

theorem intersection_of_sets :
  M = {-2, -1, 0, 1, 2} →
  N = {x | x ≥ 3 ∨ x ≤ -2} →
  M ∩ N = {-2} :=
by
  intros hM hN
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l623_62373


namespace NUMINAMATH_GPT_bus_stops_duration_per_hour_l623_62336

def speed_without_stoppages : ℝ := 90
def speed_with_stoppages : ℝ := 84
def distance_covered_lost := speed_without_stoppages - speed_with_stoppages

theorem bus_stops_duration_per_hour :
  distance_covered_lost / speed_without_stoppages * 60 = 4 :=
by
  sorry

end NUMINAMATH_GPT_bus_stops_duration_per_hour_l623_62336


namespace NUMINAMATH_GPT_small_trucks_needed_l623_62388

-- Defining the problem's conditions
def total_flour : ℝ := 500
def large_truck_capacity : ℝ := 9.6
def num_large_trucks : ℝ := 40
def small_truck_capacity : ℝ := 4

-- Theorem statement to find the number of small trucks needed
theorem small_trucks_needed : (total_flour - (num_large_trucks * large_truck_capacity)) / small_truck_capacity = (500 - (40 * 9.6)) / 4 :=
by
  sorry

end NUMINAMATH_GPT_small_trucks_needed_l623_62388


namespace NUMINAMATH_GPT_at_least_one_boy_selected_l623_62379

-- Define the number of boys and girls
def boys : ℕ := 6
def girls : ℕ := 2

-- Define the total group and the total selected
def total_people : ℕ := boys + girls
def selected_people : ℕ := 3

-- Statement: In any selection of 3 people from the group, the selection contains at least one boy
theorem at_least_one_boy_selected :
  ∀ (selection : Finset ℕ), selection.card = selected_people → selection.card > girls :=
sorry

end NUMINAMATH_GPT_at_least_one_boy_selected_l623_62379


namespace NUMINAMATH_GPT_rectangular_prism_volume_l623_62369

theorem rectangular_prism_volume (a b c : ℝ) (h1 : a * b = 15) (h2 : b * c = 10) (h3 : a * c = 6) (h4 : c^2 = a^2 + b^2) : 
  a * b * c = 30 := 
sorry

end NUMINAMATH_GPT_rectangular_prism_volume_l623_62369


namespace NUMINAMATH_GPT_seq_arithmetic_l623_62340

noncomputable def f (x n : ℝ) : ℝ := (x - 1)^2 + n

def a_n (n : ℝ) : ℝ := n
def b_n (n : ℝ) : ℝ := n + 4
def c_n (n : ℝ) : ℝ := (b_n n)^2 - (a_n n) * (b_n n)

theorem seq_arithmetic (n : ℕ) (hn : 0 < n) :
  ∃ d, d ≠ 0 ∧ ∀ n, c_n (↑n : ℝ) = c_n (↑n + 1 : ℝ) - d := 
sorry

end NUMINAMATH_GPT_seq_arithmetic_l623_62340


namespace NUMINAMATH_GPT_find_f_3_l623_62339

def f (x : ℝ) : ℝ := x + 3  -- define the function as per the condition

theorem find_f_3 : f (3) = 7 := by
  sorry

end NUMINAMATH_GPT_find_f_3_l623_62339


namespace NUMINAMATH_GPT_frustum_lateral_surface_area_l623_62394

/-- A frustum of a right circular cone has the following properties:
  * Lower base radius r1 = 8 inches
  * Upper base radius r2 = 2 inches
  * Height h = 6 inches
  The lateral surface area of such a frustum is 60 * √2 * π square inches.
-/
theorem frustum_lateral_surface_area : 
  let r1 := 8 
  let r2 := 2 
  let h := 6 
  let s := Real.sqrt (h^2 + (r1 - r2)^2)
  A = π * (r1 + r2) * s :=
  sorry

end NUMINAMATH_GPT_frustum_lateral_surface_area_l623_62394


namespace NUMINAMATH_GPT_geometric_series_ratio_l623_62333

theorem geometric_series_ratio (a_1 a_2 S q : ℝ) (hq : |q| < 1)
  (hS : S = a_1 / (1 - q))
  (ha2 : a_2 = a_1 * q) :
  S / (S - a_1) = a_1 / a_2 := 
sorry

end NUMINAMATH_GPT_geometric_series_ratio_l623_62333


namespace NUMINAMATH_GPT_ratio_of_buyers_l623_62374

theorem ratio_of_buyers (B Y T : ℕ) (hB : B = 50) 
  (hT : T = Y + 40) (hTotal : B + Y + T = 140) : 
  (Y : ℚ) / B = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_buyers_l623_62374


namespace NUMINAMATH_GPT_not_pass_first_quadrant_l623_62332

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  (1/5)^(x + 1) + m

theorem not_pass_first_quadrant (m : ℝ) : 
  (∀ x : ℝ, (1/5)^(x+1) + m ≤ 0) ↔ m ≤ -(1/5) :=
  by
  sorry

end NUMINAMATH_GPT_not_pass_first_quadrant_l623_62332


namespace NUMINAMATH_GPT_fewest_seats_to_be_occupied_l623_62381

theorem fewest_seats_to_be_occupied (n : ℕ) (h : n = 120) : ∃ m, m = 40 ∧
  ∀ a b, a + b = n → a ≥ m → ∀ x, (x > 0 ∧ x ≤ n) → (x > 1 → a = m → a + (b / 2) ≥ n / 3) :=
sorry

end NUMINAMATH_GPT_fewest_seats_to_be_occupied_l623_62381


namespace NUMINAMATH_GPT_sqrt_simplify_l623_62324

theorem sqrt_simplify (a b x : ℝ) (h : a < b) (hx1 : x + b ≥ 0) (hx2 : x + a ≤ 0) :
  Real.sqrt (-(x + a)^3 * (x + b)) = -(x + a) * (Real.sqrt (-(x + a) * (x + b))) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_simplify_l623_62324


namespace NUMINAMATH_GPT_factor_polynomial_l623_62316

theorem factor_polynomial (x y z : ℝ) : 
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) = 
  (x - y) * (y - z) * (z - x) * (-(x * y + x * z + y * z)) :=
by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l623_62316


namespace NUMINAMATH_GPT_quadratic_not_factored_l623_62392

theorem quadratic_not_factored
  (a b c : ℕ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (p : ℕ)
  (h_prime_p : Nat.Prime p)
  (h_p : a * 1991^2 + b * 1991 + c = p) :
  ¬ (∃ d₁ d₂ e₁ e₂ : ℤ, a = d₁ * d₂ ∧ b = d₁ * e₂ + d₂ * e₁ ∧ c = e₁ * e₂) :=
sorry

end NUMINAMATH_GPT_quadratic_not_factored_l623_62392


namespace NUMINAMATH_GPT_minimum_colors_needed_l623_62365

def paint_fence_colors (B : ℕ → ℕ) : Prop :=
  (∀ i : ℕ, B i ≠ B (i + 2)) ∧
  (∀ i : ℕ, B i ≠ B (i + 3)) ∧
  (∀ i : ℕ, B i ≠ B (i + 5))

theorem minimum_colors_needed : ∃ (c : ℕ), 
  (∀ B : ℕ → ℕ, paint_fence_colors B → c ≥ 3) ∧
  (∃ B : ℕ → ℕ, paint_fence_colors B ∧ c = 3) :=
sorry

end NUMINAMATH_GPT_minimum_colors_needed_l623_62365


namespace NUMINAMATH_GPT_expression_evaluation_l623_62376

theorem expression_evaluation : -20 + 8 * (5 ^ 2 - 3) = 156 := by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l623_62376


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l623_62385

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, S n = a 0 * (1 - q ^ n) / (1 - q))  -- Sum of geometric series
  (h2 : a 3 = S 3 + 1) : q = 3 :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l623_62385


namespace NUMINAMATH_GPT_mike_investment_l623_62315

-- Define the given conditions and the conclusion we want to prove
theorem mike_investment (profit : ℝ) (mary_investment : ℝ) (mike_gets_more : ℝ) (total_profit_made : ℝ) :
  profit = 7500 → 
  mary_investment = 600 →
  mike_gets_more = 1000 →
  total_profit_made = 7500 →
  ∃ (mike_investment : ℝ), 
  ((1 / 3) * profit / 2 + (mary_investment / (mary_investment + mike_investment)) * ((2 / 3) * profit) = 
  (1 / 3) * profit / 2 + (mike_investment / (mary_investment + mike_investment)) * ((2 / 3) * profit) + mike_gets_more) →
  mike_investment = 400 :=
sorry

end NUMINAMATH_GPT_mike_investment_l623_62315


namespace NUMINAMATH_GPT_dot_product_of_PA_PB_l623_62378

theorem dot_product_of_PA_PB
  (A B P: ℝ × ℝ)
  (h_circle : ∀ (x y : ℝ), x ^ 2 + y ^ 2 + 4 * x - 5 = 0 → (x, y) = A ∨ (x, y) = B)
  (h_midpoint : (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 1)
  (h_x_axis_intersect : P.2 = 0 ∧ (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = -5) :
  (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = -5 :=
sorry

end NUMINAMATH_GPT_dot_product_of_PA_PB_l623_62378


namespace NUMINAMATH_GPT_child_tickets_sold_l623_62377

theorem child_tickets_sold (A C : ℕ) (h1 : A + C = 130) (h2 : 12 * A + 4 * C = 840) : C = 90 := by
  sorry

end NUMINAMATH_GPT_child_tickets_sold_l623_62377


namespace NUMINAMATH_GPT_remaining_credit_l623_62323

noncomputable def initial_balance : ℝ := 30
noncomputable def call_rate : ℝ := 0.16
noncomputable def call_duration : ℝ := 22

theorem remaining_credit : initial_balance - (call_rate * call_duration) = 26.48 :=
by
  -- Definitions for readability
  let total_cost := call_rate * call_duration
  let remaining_balance := initial_balance - total_cost
  have h : total_cost = 3.52 := sorry
  have h₂ : remaining_balance = 26.48 := sorry
  exact h₂

end NUMINAMATH_GPT_remaining_credit_l623_62323


namespace NUMINAMATH_GPT_point_in_third_quadrant_l623_62325

open Complex

-- Define that i is the imaginary unit
def imaginary_unit : ℂ := Complex.I

-- Define the condition i * z = 1 - 2i
def condition (z : ℂ) : Prop := imaginary_unit * z = (1 : ℂ) - 2 * imaginary_unit

-- Prove that the point corresponding to the complex number z is located in the third quadrant
theorem point_in_third_quadrant (z : ℂ) (h : condition z) : z.re < 0 ∧ z.im < 0 := sorry

end NUMINAMATH_GPT_point_in_third_quadrant_l623_62325


namespace NUMINAMATH_GPT_equivalent_proposition_l623_62354

theorem equivalent_proposition (H : Prop) (P : Prop) (Q : Prop) (hpq : H → P → ¬ Q) : (H → ¬ Q → ¬ P) :=
by
  intro h nq np
  sorry

end NUMINAMATH_GPT_equivalent_proposition_l623_62354


namespace NUMINAMATH_GPT_triangle_perimeter_ratio_l623_62375

theorem triangle_perimeter_ratio : 
  let side := 10
  let hypotenuse := Real.sqrt (side^2 + (side / 2) ^ 2)
  let triangle_perimeter := side + (side / 2) + hypotenuse
  let square_perimeter := 4 * side
  (triangle_perimeter / square_perimeter) = (15 + Real.sqrt 125) / 40 := 
by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_ratio_l623_62375


namespace NUMINAMATH_GPT_solve_grape_rate_l623_62342

noncomputable def grape_rate (G : ℝ) : Prop :=
  11 * G + 7 * 50 = 1428

theorem solve_grape_rate : ∃ G : ℝ, grape_rate G ∧ G = 98 :=
by
  exists 98
  sorry

end NUMINAMATH_GPT_solve_grape_rate_l623_62342


namespace NUMINAMATH_GPT_original_number_of_men_l623_62349

theorem original_number_of_men 
  (x : ℕ)
  (H : 15 * 18 * x = 15 * 18 * (x - 8) + 8 * 15 * 18)
  (h_pos : x > 8) :
  x = 40 :=
sorry

end NUMINAMATH_GPT_original_number_of_men_l623_62349


namespace NUMINAMATH_GPT_initial_population_l623_62311

/-- The population of a town decreases annually at the rate of 20% p.a.
    Given that the population of the town after 2 years is 19200,
    prove that the initial population of the town was 30,000. -/
theorem initial_population (P : ℝ) (h : 0.64 * P = 19200) : P = 30000 :=
sorry

end NUMINAMATH_GPT_initial_population_l623_62311


namespace NUMINAMATH_GPT_geometric_series_common_ratio_l623_62348

theorem geometric_series_common_ratio (r : ℚ) : 
  (∃ (a : ℚ), a = 4 / 7 ∧ a * r = 16 / 21) → r = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_common_ratio_l623_62348


namespace NUMINAMATH_GPT_minimum_soldiers_to_add_l623_62341

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  (N + 82) % 84 = 0 :=
by
  sorry

end NUMINAMATH_GPT_minimum_soldiers_to_add_l623_62341


namespace NUMINAMATH_GPT_prime_p_q_r_condition_l623_62305

theorem prime_p_q_r_condition (p q r : ℕ) (hp : Nat.Prime p) (hq_pos : 0 < q) (hr_pos : 0 < r)
    (hp_not_dvd_q : ¬ (p ∣ q)) (h3_not_dvd_q : ¬ (3 ∣ q)) (eqn : p^3 = r^3 - q^2) : 
    p = 7 := sorry

end NUMINAMATH_GPT_prime_p_q_r_condition_l623_62305


namespace NUMINAMATH_GPT_quadratic_expression_odd_quadratic_expression_not_square_l623_62363

theorem quadratic_expression_odd (n : ℕ) : 
  (n^2 + n + 1) % 2 = 1 := 
by sorry

theorem quadratic_expression_not_square (n : ℕ) : 
  ¬ ∃ (m : ℕ), m^2 = n^2 + n + 1 := 
by sorry

end NUMINAMATH_GPT_quadratic_expression_odd_quadratic_expression_not_square_l623_62363


namespace NUMINAMATH_GPT_frood_points_smallest_frood_points_l623_62398

theorem frood_points (n : ℕ) (h : n > 9) : (n * (n + 1)) / 2 > 5 * n :=
by {
  sorry
}

noncomputable def smallest_n : ℕ := 10

theorem smallest_frood_points (m : ℕ) (h : (m * (m + 1)) / 2 > 5 * m) : 10 ≤ m :=
by {
  sorry
}

end NUMINAMATH_GPT_frood_points_smallest_frood_points_l623_62398


namespace NUMINAMATH_GPT_speed_of_man_in_still_water_l623_62334

def upstream_speed : ℝ := 32
def downstream_speed : ℝ := 48

theorem speed_of_man_in_still_water :
  (upstream_speed + (downstream_speed - upstream_speed) / 2) = 40 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_man_in_still_water_l623_62334


namespace NUMINAMATH_GPT_problem_statement_l623_62360

/-- A predicate that checks if the numbers from 1 to 2n can be split into two groups 
    such that the sum of the product of the elements of each group is divisible by 2n - 1. -/
def valid_split (n : ℕ) : Prop :=
  ∃ (a b : Finset ℕ), 
  a ∪ b = Finset.range (2 * n) ∧
  a ∩ b = ∅ ∧
  (2 * n) ∣ (a.prod id + b.prod id - 1)

theorem problem_statement : 
  ∀ n : ℕ, n > 0 → valid_split n ↔ (n = 1 ∨ ∃ a : ℕ, n = 2^a ∧ a ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l623_62360


namespace NUMINAMATH_GPT_convex_polygon_num_sides_l623_62353

theorem convex_polygon_num_sides (n : ℕ) 
  (h1 : ∀ (i : ℕ), i < n → 120 + i * 5 < 180) 
  (h2 : (n - 2) * 180 = n * (240 + (n - 1) * 5) / 2) : 
  n = 9 :=
sorry

end NUMINAMATH_GPT_convex_polygon_num_sides_l623_62353


namespace NUMINAMATH_GPT_Katya_possible_numbers_l623_62306

def divisible_by (n m : ℕ) : Prop := m % n = 0

def possible_numbers (n : ℕ) : Prop :=
  let condition1 := divisible_by 7 n  -- Alyona's condition
  let condition2 := divisible_by 5 n  -- Lena's condition
  let condition3 := n < 9             -- Rita's condition
  (condition1 ∨ condition2) ∧ condition3 ∧ 
  ((condition1 ∧ condition3 ∧ ¬condition2) ∨ (condition2 ∧ condition3 ∧ ¬condition1))

theorem Katya_possible_numbers :
  ∀ n : ℕ, 
    (possible_numbers n) ↔ (n = 5 ∨ n = 7) :=
sorry

end NUMINAMATH_GPT_Katya_possible_numbers_l623_62306


namespace NUMINAMATH_GPT_ball_bounce_height_l623_62338

theorem ball_bounce_height :
  ∃ k : ℕ, k = 4 ∧ 45 * (1 / 3 : ℝ) ^ k < 2 :=
by 
  use 4
  sorry

end NUMINAMATH_GPT_ball_bounce_height_l623_62338


namespace NUMINAMATH_GPT_cosine_identity_l623_62318

theorem cosine_identity
  (α : ℝ)
  (h : Real.sin (π / 6 + α) = (Real.sqrt 3) / 3) :
  Real.cos (π / 3 - α) = (Real.sqrt 3) / 2 :=
by
  sorry

end NUMINAMATH_GPT_cosine_identity_l623_62318
