import Mathlib

namespace NUMINAMATH_GPT_math_problem_l1776_177670

theorem math_problem (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^3 + y^3 = x - y) : x^2 + 4 * y^2 < 1 := 
sorry

end NUMINAMATH_GPT_math_problem_l1776_177670


namespace NUMINAMATH_GPT_find_f_1991_l1776_177612

namespace FunctionProof

-- Defining the given conditions as statements in Lean
def func_f (f : ℤ → ℤ) : Prop :=
  ∀ m n : ℤ, f (m + f (f n)) = -f (f (m + 1)) - n

def poly_g (f g : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, g n = g (f n)

-- Statement of the problem
theorem find_f_1991 
  (f g : ℤ → ℤ)
  (Hf : func_f f)
  (Hg : poly_g f g) :
  f 1991 = -1992 := 
sorry

end FunctionProof

end NUMINAMATH_GPT_find_f_1991_l1776_177612


namespace NUMINAMATH_GPT_crows_and_trees_l1776_177699

theorem crows_and_trees : ∃ (x y : ℕ), 3 * y + 5 = x ∧ 5 * (y - 1) = x ∧ x = 20 ∧ y = 5 :=
by
  sorry

end NUMINAMATH_GPT_crows_and_trees_l1776_177699


namespace NUMINAMATH_GPT_probability_neither_defective_l1776_177637

def total_pens : ℕ := 8
def defective_pens : ℕ := 3
def non_defective_pens : ℕ := total_pens - defective_pens
def draw_count : ℕ := 2

def probability_of_non_defective (total : ℕ) (defective : ℕ) (draws : ℕ) : ℚ :=
  let non_defective := total - defective
  (non_defective / total) * ((non_defective - 1) / (total - 1))

theorem probability_neither_defective :
  probability_of_non_defective total_pens defective_pens draw_count = 5 / 14 :=
by sorry

end NUMINAMATH_GPT_probability_neither_defective_l1776_177637


namespace NUMINAMATH_GPT_largest_base_6_five_digits_l1776_177627

-- Define the base-6 number 55555 in base 10
def base_6_to_base_10 (n : Nat) : Nat :=
  let d0 := (n % 10)
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  let d3 := (n / 1000) % 10
  let d4 := (n / 10000) % 10
  d4 * 6^4 + d3 * 6^3 + d2 * 6^2 + d1 * 6^1 + d0 * 6^0

theorem largest_base_6_five_digits : base_6_to_base_10 55555 = 7775 := by
  sorry

end NUMINAMATH_GPT_largest_base_6_five_digits_l1776_177627


namespace NUMINAMATH_GPT_second_part_shorter_l1776_177622

def length_wire : ℕ := 180
def length_part1 : ℕ := 106
def length_part2 : ℕ := length_wire - length_part1
def length_difference : ℕ := length_part1 - length_part2

theorem second_part_shorter :
  length_difference = 32 :=
by
  sorry

end NUMINAMATH_GPT_second_part_shorter_l1776_177622


namespace NUMINAMATH_GPT_intersection_complement_l1776_177678

-- Definitions and conditions
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 2, 3}
def B : Set ℕ := {1, 3, 4}
def C_U (B : Set ℕ) : Set ℕ := {x ∈ U | x ∉ B}

-- Theorem statement
theorem intersection_complement :
  (C_U B) ∩ A = {0, 2} := 
by
  -- Proof is not required, so we use sorry
  sorry

end NUMINAMATH_GPT_intersection_complement_l1776_177678


namespace NUMINAMATH_GPT_Tim_weekly_earnings_l1776_177654

def number_of_tasks_per_day : ℕ := 100
def pay_per_task : ℝ := 1.2
def working_days_per_week : ℕ := 6

theorem Tim_weekly_earnings :
  (number_of_tasks_per_day * pay_per_task) * working_days_per_week = 720 := by
  sorry

end NUMINAMATH_GPT_Tim_weekly_earnings_l1776_177654


namespace NUMINAMATH_GPT_third_neigh_uses_100_more_l1776_177677

def total_water : Nat := 1200
def first_neigh_usage : Nat := 150
def second_neigh_usage : Nat := 2 * first_neigh_usage
def fourth_neigh_remaining : Nat := 350

def third_neigh_usage := total_water - (first_neigh_usage + second_neigh_usage + fourth_neigh_remaining)
def diff_third_second := third_neigh_usage - second_neigh_usage

theorem third_neigh_uses_100_more :
  diff_third_second = 100 := by
  sorry

end NUMINAMATH_GPT_third_neigh_uses_100_more_l1776_177677


namespace NUMINAMATH_GPT_total_blood_cells_correct_l1776_177697

def first_sample : ℕ := 4221
def second_sample : ℕ := 3120
def total_blood_cells : ℕ := first_sample + second_sample

theorem total_blood_cells_correct : total_blood_cells = 7341 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_blood_cells_correct_l1776_177697


namespace NUMINAMATH_GPT_rightmost_three_digits_of_7_pow_1987_l1776_177635

theorem rightmost_three_digits_of_7_pow_1987 :
  (7^1987 : ℕ) % 1000 = 643 := 
by 
  sorry

end NUMINAMATH_GPT_rightmost_three_digits_of_7_pow_1987_l1776_177635


namespace NUMINAMATH_GPT_extremum_value_of_a_g_monotonicity_l1776_177621

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 3 + x ^ 2

theorem extremum_value_of_a (a : ℝ) (h : (3 * a * (-4 / 3) ^ 2 + 2 * (-4 / 3) = 0)) : a = 1 / 2 :=
by
  -- We need to prove that a = 1 / 2 given the extremum condition.
  sorry

noncomputable def g (x : ℝ) : ℝ := (1 / 2 * x ^ 3 + x ^ 2) * Real.exp x

theorem g_monotonicity :
  (∀ x < -4, deriv g x < 0) ∧
  (∀ x, -4 < x ∧ x < -1 → deriv g x > 0) ∧
  (∀ x, -1 < x ∧ x < 0 → deriv g x < 0) ∧
  (∀ x > 0, deriv g x > 0) :=
by
  -- We need to prove the monotonicity of the function g in the specified intervals.
  sorry

end NUMINAMATH_GPT_extremum_value_of_a_g_monotonicity_l1776_177621


namespace NUMINAMATH_GPT_intersection_M_N_l1776_177624

def M : Set ℤ := {-1, 0, 1, 2}
def N : Set ℤ := {x | |x| > 1}

theorem intersection_M_N : M ∩ N = {2} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1776_177624


namespace NUMINAMATH_GPT_simplification_problem_l1776_177664

theorem simplification_problem :
  (3^2015 - 3^2013 + 3^2011) / (3^2015 + 3^2013 - 3^2011) = 73 / 89 :=
  sorry

end NUMINAMATH_GPT_simplification_problem_l1776_177664


namespace NUMINAMATH_GPT_quadratic_real_roots_iff_l1776_177692

-- Define the statement of the problem in Lean
theorem quadratic_real_roots_iff (m : ℝ) :
  (∃ x : ℂ, m * x^2 + 2 * x - 1 = 0) ↔ (m ≥ -1 ∧ m ≠ 0) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_iff_l1776_177692


namespace NUMINAMATH_GPT_value_of_a_l1776_177630

-- Define the quadratic function f
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * x + b

-- Define the condition f(1) = f(2)
def condition (a b : ℝ) : Prop := f 1 a b = f 2 a b

-- The proof problem statement
theorem value_of_a (a b : ℝ) (h : condition a b) : a = -3 :=
by sorry

end NUMINAMATH_GPT_value_of_a_l1776_177630


namespace NUMINAMATH_GPT_interchangeable_statements_l1776_177608

-- Modeled conditions and relationships
def perpendicular (l p: Type) : Prop := sorry -- Definition of perpendicularity between a line and a plane
def parallel (a b: Type) : Prop := sorry -- Definition of parallelism between two objects (lines or planes)

-- Original Statements
def statement_1 := ∀ (l₁ l₂ p: Type), (perpendicular l₁ p) ∧ (perpendicular l₂ p) → parallel l₁ l₂
def statement_2 := ∀ (p₁ p₂ p: Type), (perpendicular p₁ p) ∧ (perpendicular p₂ p) → parallel p₁ p₂
def statement_3 := ∀ (l₁ l₂ l: Type), (parallel l₁ l) ∧ (parallel l₂ l) → parallel l₁ l₂
def statement_4 := ∀ (l₁ l₂ p: Type), (parallel l₁ p) ∧ (parallel l₂ p) → parallel l₁ l₂

-- Swapped Statements
def swapped_1 := ∀ (p₁ p₂ l: Type), (perpendicular p₁ l) ∧ (perpendicular p₂ l) → parallel p₁ p₂
def swapped_2 := ∀ (l₁ l₂ l: Type), (perpendicular l₁ l) ∧ (perpendicular l₂ l) → parallel l₁ l₂
def swapped_3 := ∀ (p₁ p₂ p: Type), (parallel p₁ p) ∧ (parallel p₂ p) → parallel p₁ p₂
def swapped_4 := ∀ (p₁ p₂ l: Type), (parallel p₁ l) ∧ (parallel p₂ l) → parallel p₁ p₂

-- Proof Problem: Verify which statements are interchangeable
theorem interchangeable_statements :
  (statement_1 ↔ swapped_1) ∧
  (statement_2 ↔ swapped_2) ∧
  (statement_3 ↔ swapped_3) ∧
  (statement_4 ↔ swapped_4) :=
sorry

end NUMINAMATH_GPT_interchangeable_statements_l1776_177608


namespace NUMINAMATH_GPT_moles_of_HC2H3O2_needed_l1776_177645

theorem moles_of_HC2H3O2_needed :
  (∀ (HC2H3O2 NaHCO3 H2O : ℕ), 
    (HC2H3O2 + NaHCO3 = NaC2H3O2 + H2O + CO2) → 
    (H2O = 3) → 
    (NaHCO3 = 3) → 
    HC2H3O2 = 3) :=
by
  intros HC2H3O2 NaHCO3 H2O h_eq h_H2O h_NaHCO3
  -- Hint: You can use the balanced chemical equation to derive that HC2H3O2 must be 3
  sorry

end NUMINAMATH_GPT_moles_of_HC2H3O2_needed_l1776_177645


namespace NUMINAMATH_GPT_evaluate_expression_at_minus_one_l1776_177609

theorem evaluate_expression_at_minus_one :
  ((-1 + 1) * (-1 - 2) + 2 * (-1 + 4) * (-1 - 4)) = -30 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_minus_one_l1776_177609


namespace NUMINAMATH_GPT_correct_combined_average_l1776_177605

noncomputable def average_marks : ℝ :=
  let num_students : ℕ := 100
  let avg_math_marks : ℝ := 85
  let avg_science_marks : ℝ := 89
  let incorrect_math_marks : List ℝ := [76, 80, 95, 70, 90]
  let correct_math_marks : List ℝ := [86, 70, 75, 90, 100]
  let incorrect_science_marks : List ℝ := [105, 60, 80, 92, 78]
  let correct_science_marks : List ℝ := [95, 70, 90, 82, 88]

  let total_incorrect_math := incorrect_math_marks.sum
  let total_correct_math := correct_math_marks.sum
  let diff_math := total_correct_math - total_incorrect_math

  let total_incorrect_science := incorrect_science_marks.sum
  let total_correct_science := correct_science_marks.sum
  let diff_science := total_correct_science - total_incorrect_science

  let incorrect_total_math := avg_math_marks * num_students
  let correct_total_math := incorrect_total_math + diff_math

  let incorrect_total_science := avg_science_marks * num_students
  let correct_total_science := incorrect_total_science + diff_science

  let combined_total := correct_total_math + correct_total_science
  combined_total / (num_students * 2)

theorem correct_combined_average :
  average_marks = 87.1 :=
by
  sorry

end NUMINAMATH_GPT_correct_combined_average_l1776_177605


namespace NUMINAMATH_GPT_realize_ancient_dreams_only_C_l1776_177659

-- Define the available options
inductive Options
| A : Options
| B : Options
| C : Options
| D : Options

-- Define the ancient dreams condition
def realize_ancient_dreams (o : Options) : Prop :=
  o = Options.C

-- The theorem states that only Geographic Information Technology (option C) can realize the ancient dreams
theorem realize_ancient_dreams_only_C :
  realize_ancient_dreams Options.C :=
by
  -- skip the exact proof
  sorry

end NUMINAMATH_GPT_realize_ancient_dreams_only_C_l1776_177659


namespace NUMINAMATH_GPT_quadratic_intersects_x_axis_only_once_l1776_177667

theorem quadratic_intersects_x_axis_only_once (a : ℝ) :
  (∀ x : ℝ, (a * x^2 - a * x + 3 * x + 1 = 0) → a = 1 ∨ a = 9) :=
sorry

end NUMINAMATH_GPT_quadratic_intersects_x_axis_only_once_l1776_177667


namespace NUMINAMATH_GPT_count_elements_in_A_l1776_177685

variables (a b : ℕ)

def condition1 : Prop := a = 3 * b / 2
def condition2 : Prop := a + b - 1200 = 4500

theorem count_elements_in_A (h1 : condition1 a b) (h2 : condition2 a b) : a = 3420 :=
by sorry

end NUMINAMATH_GPT_count_elements_in_A_l1776_177685


namespace NUMINAMATH_GPT_average_weight_of_dogs_is_5_l1776_177632

def weight_of_brown_dog (B : ℝ) : ℝ := B
def weight_of_black_dog (B : ℝ) : ℝ := B + 1
def weight_of_white_dog (B : ℝ) : ℝ := 2 * B
def weight_of_grey_dog (B : ℝ) : ℝ := B - 1

theorem average_weight_of_dogs_is_5 (B : ℝ) (h : (weight_of_brown_dog B + weight_of_black_dog B + weight_of_white_dog B + weight_of_grey_dog B) / 4 = 5) :
  5 = 5 :=
by sorry

end NUMINAMATH_GPT_average_weight_of_dogs_is_5_l1776_177632


namespace NUMINAMATH_GPT_trigonometric_identity_proof_l1776_177665

noncomputable def m : ℝ := 2 * Real.sin (Real.pi / 10)
noncomputable def n : ℝ := 4 - m^2

theorem trigonometric_identity_proof :
  (m = 2 * Real.sin (Real.pi / 10)) →
  (m^2 + n = 4) →
  (m * Real.sqrt n) / (2 * Real.cos (3 * Real.pi / 20)^2 - 1) = 2 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_trigonometric_identity_proof_l1776_177665


namespace NUMINAMATH_GPT_part1_part2_l1776_177623

-- Define the conditions that translate the quadratic equation having distinct real roots
def discriminant_condition (m : ℝ) : Prop :=
  let a := 1
  let b := -4
  let c := 3 - 2 * m
  b ^ 2 - 4 * a * c > 0

-- Define the root condition from Vieta's formulas and the additional given condition
def additional_condition (m : ℝ) : Prop :=
  let x1_plus_x2 := 4
  let x1_times_x2 := 3 - 2 * m
  x1_times_x2 + x1_plus_x2 - m^2 = 4

-- Prove the range of m for part 1
theorem part1 (m : ℝ) : discriminant_condition m → m ≥ -1/2 := by
  sorry

-- Prove the value of m for part 2 with the range condition
theorem part2 (m : ℝ) : discriminant_condition m → additional_condition m → m = 1 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1776_177623


namespace NUMINAMATH_GPT_consecutive_odd_numbers_square_difference_l1776_177600

theorem consecutive_odd_numbers_square_difference (a b : ℤ) :
  (a - b = 2 ∨ b - a = 2) → (a^2 - b^2 = 2000) → (a = 501 ∧ b = 499 ∨ a = -501 ∧ b = -499) :=
by 
  intros h1 h2
  sorry

end NUMINAMATH_GPT_consecutive_odd_numbers_square_difference_l1776_177600


namespace NUMINAMATH_GPT_grape_juice_problem_l1776_177681

noncomputable def grape_juice_amount (initial_mixture_volume : ℕ) (initial_concentration : ℝ) (final_concentration : ℝ) : ℝ :=
  let initial_grape_juice := initial_mixture_volume * initial_concentration
  let total_volume := initial_mixture_volume + final_concentration * (final_concentration - initial_grape_juice) / (1 - final_concentration) -- Total volume after adding x gallons
  let added_grape_juice := total_volume - initial_mixture_volume -- x gallons added
  added_grape_juice

theorem grape_juice_problem :
  grape_juice_amount 40 0.20 0.36 = 10 := 
by
  sorry

end NUMINAMATH_GPT_grape_juice_problem_l1776_177681


namespace NUMINAMATH_GPT_length_of_segment_AB_l1776_177662

variables (h : ℝ) (AB CD : ℝ)

-- Defining the conditions
def condition_one : Prop := (AB / CD = 5 / 2)
def condition_two : Prop := (AB + CD = 280)

-- The theorem to prove
theorem length_of_segment_AB (h : ℝ) (AB CD : ℝ) :
  condition_one AB CD ∧ condition_two AB CD → AB = 200 :=
by
  sorry

end NUMINAMATH_GPT_length_of_segment_AB_l1776_177662


namespace NUMINAMATH_GPT_incorrect_statement_l1776_177666

theorem incorrect_statement : 
  ¬(∀ (p q : Prop), (¬p ∧ ¬q) → (¬p ∧ ¬q)) := 
    sorry

end NUMINAMATH_GPT_incorrect_statement_l1776_177666


namespace NUMINAMATH_GPT_train_speed_l1776_177655

theorem train_speed (train_length bridge_length : ℕ) (time : ℝ)
  (h_train_length : train_length = 110)
  (h_bridge_length : bridge_length = 290)
  (h_time : time = 23.998080153587715) :
  (train_length + bridge_length) / time * 3.6 = 60 := 
by
  rw [h_train_length, h_bridge_length, h_time]
  sorry

end NUMINAMATH_GPT_train_speed_l1776_177655


namespace NUMINAMATH_GPT_tangent_line_parallel_to_given_line_l1776_177682

theorem tangent_line_parallel_to_given_line 
  (x : ℝ) (y : ℝ) (tangent_line : ℝ → ℝ) :
  (tangent_line y = x^2 - 1) → 
  (tangent_line = 4) → 
  (4 * x - y - 5 = 0) :=
by 
  sorry

end NUMINAMATH_GPT_tangent_line_parallel_to_given_line_l1776_177682


namespace NUMINAMATH_GPT_gcd_779_209_589_l1776_177663

theorem gcd_779_209_589 : Int.gcd (Int.gcd 779 209) 589 = 19 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_779_209_589_l1776_177663


namespace NUMINAMATH_GPT_find_additional_fuel_per_person_l1776_177689

def num_passengers : ℕ := 30
def num_crew : ℕ := 5
def num_people : ℕ := num_passengers + num_crew
def num_bags_per_person : ℕ := 2
def num_bags : ℕ := num_people * num_bags_per_person
def fuel_empty_plane : ℕ := 20
def fuel_per_bag : ℕ := 2
def total_trip_fuel : ℕ := 106000
def trip_distance : ℕ := 400
def fuel_per_mile : ℕ := total_trip_fuel / trip_distance

def additional_fuel_per_person (x : ℕ) : Prop :=
  fuel_empty_plane + num_people * x + num_bags * fuel_per_bag = fuel_per_mile

theorem find_additional_fuel_per_person : additional_fuel_per_person 3 :=
  sorry

end NUMINAMATH_GPT_find_additional_fuel_per_person_l1776_177689


namespace NUMINAMATH_GPT_point_P_quadrant_l1776_177695

theorem point_P_quadrant 
  (h1 : Real.sin (θ / 2) = 3 / 5) 
  (h2 : Real.cos (θ / 2) = -4 / 5) : 
  (0 < Real.cos θ) ∧ (Real.sin θ < 0) :=
by
  sorry

end NUMINAMATH_GPT_point_P_quadrant_l1776_177695


namespace NUMINAMATH_GPT_both_not_divisible_by_7_l1776_177688

theorem both_not_divisible_by_7 {a b : ℝ} (h : ¬ (∃ k : ℤ, ab = 7 * k)) : ¬ (∃ m : ℤ, a = 7 * m) ∧ ¬ (∃ n : ℤ, b = 7 * n) :=
sorry

end NUMINAMATH_GPT_both_not_divisible_by_7_l1776_177688


namespace NUMINAMATH_GPT_tailor_trim_amount_l1776_177679

variable (x : ℝ)

def original_side : ℝ := 22
def trimmed_side : ℝ := original_side - x
def fixed_trimmed_side : ℝ := original_side - 5
def remaining_area : ℝ := 120

theorem tailor_trim_amount :
  (original_side - x) * 17 = remaining_area → x = 15 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_tailor_trim_amount_l1776_177679


namespace NUMINAMATH_GPT_shenille_scores_points_l1776_177690

theorem shenille_scores_points :
  ∀ (x y : ℕ), (x + y = 45) → (x = 2 * y) → 
  (25/100 * x + 40/100 * y) * 3 + (40/100 * y) * 2 = 33 :=
by 
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_shenille_scores_points_l1776_177690


namespace NUMINAMATH_GPT_pirate_total_dollar_amount_l1776_177603

def base_5_to_base_10 (digits : List ℕ) (base : ℕ) : ℕ :=
  digits.reverse.enum.map (λ ⟨p, d⟩ => d * base^p) |>.sum

def jewelry_base5 := [3, 1, 2, 4]
def gold_coins_base5 := [3, 1, 2, 2]
def alcohol_base5 := [1, 2, 4]

def jewelry_base10 := base_5_to_base_10 jewelry_base5 5
def gold_coins_base10 := base_5_to_base_10 gold_coins_base5 5
def alcohol_base10 := base_5_to_base_10 alcohol_base5 5

def total_base10 := jewelry_base10 + gold_coins_base10 + alcohol_base10

theorem pirate_total_dollar_amount :
  total_base10 = 865 :=
by
  unfold total_base10 jewelry_base10 gold_coins_base10 alcohol_base10 base_5_to_base_10
  simp
  sorry

end NUMINAMATH_GPT_pirate_total_dollar_amount_l1776_177603


namespace NUMINAMATH_GPT_longest_chord_length_of_circle_l1776_177602

theorem longest_chord_length_of_circle (r : ℝ) (h : r = 5) : ∃ d, d = 10 :=
by
  sorry

end NUMINAMATH_GPT_longest_chord_length_of_circle_l1776_177602


namespace NUMINAMATH_GPT_cube_sufficient_but_not_necessary_l1776_177619

theorem cube_sufficient_but_not_necessary (x : ℝ) : (x^3 > 27 → |x| > 3) ∧ (¬(|x| > 3 → x^3 > 27)) :=
by
  sorry

end NUMINAMATH_GPT_cube_sufficient_but_not_necessary_l1776_177619


namespace NUMINAMATH_GPT_oranges_in_first_bucket_l1776_177618

theorem oranges_in_first_bucket
  (x : ℕ) -- number of oranges in the first bucket
  (h1 : ∃ n, n = x) -- condition: There are some oranges in the first bucket
  (h2 : ∃ y, y = x + 17) -- condition: The second bucket has 17 more oranges than the first bucket
  (h3 : ∃ z, z = x + 6) -- condition: The third bucket has 11 fewer oranges than the second bucket
  (h4 : x + (x + 17) + (x + 6) = 89) -- condition: There are 89 oranges in all the buckets
  : x = 22 := -- conclusion: number of oranges in the first bucket is 22
sorry

end NUMINAMATH_GPT_oranges_in_first_bucket_l1776_177618


namespace NUMINAMATH_GPT_product_of_averages_is_125000_l1776_177674

-- Define the problem from step a
def sum_1_to_99 : ℕ := (99 * (1 + 99)) / 2
def average_of_group (x : ℕ) : Prop := 3 * 33 * x = sum_1_to_99

-- Define the goal to prove
theorem product_of_averages_is_125000 (x : ℕ) (h : average_of_group x) : x^3 = 125000 :=
by
  sorry

end NUMINAMATH_GPT_product_of_averages_is_125000_l1776_177674


namespace NUMINAMATH_GPT_sum_of_trinomials_1_l1776_177634

theorem sum_of_trinomials_1 (p q : ℝ) :
  (p + q = 0 ∨ p + q = 8) →
  (2 * (1 : ℝ)^2 + (p + q) * 1 + (p + q) = 2 ∨ 2 * (1 : ℝ)^2 + (p + q) * 1 + (p + q) = 18) :=
by sorry

end NUMINAMATH_GPT_sum_of_trinomials_1_l1776_177634


namespace NUMINAMATH_GPT_acute_angles_complementary_l1776_177691

-- Given conditions
variables (α β : ℝ)
variables (α_acute : 0 < α ∧ α < π / 2) (β_acute : 0 < β ∧ β < π / 2)
variables (h : (sin α) ^ 2 + (sin β) ^ 2 = sin (α + β))

-- Statement we want to prove
theorem acute_angles_complementary : α + β = π / 2 :=
  sorry

end NUMINAMATH_GPT_acute_angles_complementary_l1776_177691


namespace NUMINAMATH_GPT_students_received_B_l1776_177629

theorem students_received_B (x : ℕ) 
  (h1 : (0.8 * x : ℝ) + x + (1.2 * x : ℝ) = 28) : 
  x = 9 := 
by
  sorry

end NUMINAMATH_GPT_students_received_B_l1776_177629


namespace NUMINAMATH_GPT_find_c_l1776_177640

variable {a b c : ℝ} 
variable (h_perpendicular : (a / 3) * (-3 / b) = -1)
variable (h_intersect1 : 2 * a + 9 = c)
variable (h_intersect2 : 6 - 3 * b = -c)
variable (h_ab_equal : a = b)

theorem find_c : c = 39 := 
by
  sorry

end NUMINAMATH_GPT_find_c_l1776_177640


namespace NUMINAMATH_GPT_initial_milk_water_ratio_l1776_177652

theorem initial_milk_water_ratio
  (M W : ℕ)
  (h1 : M + W = 40000)
  (h2 : (M : ℚ) / (W + 1600) = 3 / 1) :
  (M : ℚ) / W = 3.55 :=
by
  sorry

end NUMINAMATH_GPT_initial_milk_water_ratio_l1776_177652


namespace NUMINAMATH_GPT_value_of_3a_plus_6b_l1776_177604

theorem value_of_3a_plus_6b (a b : ℝ) (h : a + 2 * b = 1) : 3 * a + 6 * b = 3 :=
sorry

end NUMINAMATH_GPT_value_of_3a_plus_6b_l1776_177604


namespace NUMINAMATH_GPT_difference_between_a_b_l1776_177684

theorem difference_between_a_b (a b : ℝ) (d : ℝ) : 
  (a - b = d) → (a ^ 2 + b ^ 2 = 150) → (a * b = 25) → d = 10 :=
by
  sorry

end NUMINAMATH_GPT_difference_between_a_b_l1776_177684


namespace NUMINAMATH_GPT_product_of_fractions_l1776_177642

open BigOperators

theorem product_of_fractions :
  (∏ n in Finset.range 9, (n + 2)^3 - 1) / (∏ n in Finset.range 9, (n + 2)^3 + 1) = 74 / 55 :=
by
  sorry

end NUMINAMATH_GPT_product_of_fractions_l1776_177642


namespace NUMINAMATH_GPT_dimes_paid_l1776_177676

theorem dimes_paid (cost_in_dollars : ℕ) (dollars_to_dimes : ℕ) (h₁ : cost_in_dollars = 5) (h₂ : dollars_to_dimes = 10) :
  cost_in_dollars * dollars_to_dimes = 50 :=
by
  sorry

end NUMINAMATH_GPT_dimes_paid_l1776_177676


namespace NUMINAMATH_GPT_melissa_bonus_points_l1776_177653

/-- Given that Melissa scored 109 points per game and a total of 15089 points in 79 games,
    prove that she got 82 bonus points per game. -/
theorem melissa_bonus_points (points_per_game : ℕ) (total_points : ℕ) (num_games : ℕ)
  (H1 : points_per_game = 109)
  (H2 : total_points = 15089)
  (H3 : num_games = 79) : 
  (total_points - points_per_game * num_games) / num_games = 82 := by
  sorry

end NUMINAMATH_GPT_melissa_bonus_points_l1776_177653


namespace NUMINAMATH_GPT_upstream_distance_l1776_177672

theorem upstream_distance (v : ℝ) 
  (H1 : ∀ d : ℝ, (10 + v) * 2 = 28) 
  (H2 : (10 - v) * 2 = d) : d = 12 := by
  sorry

end NUMINAMATH_GPT_upstream_distance_l1776_177672


namespace NUMINAMATH_GPT_find_a_extreme_value_l1776_177668

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (x + 1) - x - a * x

theorem find_a_extreme_value :
  (∃ a : ℝ, ∀ x, f x a = Real.log (x + 1) - x - a * x ∧ (∃ m : ℝ, ∀ y : ℝ, f y a ≤ m)) ↔ a = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_extreme_value_l1776_177668


namespace NUMINAMATH_GPT_kevin_bucket_size_l1776_177650

def rate_of_leakage (r : ℝ) : Prop := r = 1.5
def time_away (t : ℝ) : Prop := t = 12
def bucket_size (b : ℝ) (r t : ℝ) : Prop := b = 2 * r * t

theorem kevin_bucket_size
  (r t b : ℝ)
  (H1 : rate_of_leakage r)
  (H2 : time_away t) :
  bucket_size b r t :=
by
  simp [rate_of_leakage, time_away, bucket_size] at *
  sorry

end NUMINAMATH_GPT_kevin_bucket_size_l1776_177650


namespace NUMINAMATH_GPT_smallest_c_d_sum_l1776_177628

theorem smallest_c_d_sum : ∃ (c d : ℕ), 2^12 * 7^6 = c^d ∧  (∀ (c' d' : ℕ), 2^12 * 7^6 = c'^d'  → (c + d) ≤ (c' + d')) ∧ c + d = 21954 := by
  sorry

end NUMINAMATH_GPT_smallest_c_d_sum_l1776_177628


namespace NUMINAMATH_GPT_distance_A_B_l1776_177683

variable (x : ℚ)

def pointA := x
def pointB := 1
def pointC := -1

theorem distance_A_B : |pointA x - pointB| = |x - 1| := by
  sorry

end NUMINAMATH_GPT_distance_A_B_l1776_177683


namespace NUMINAMATH_GPT_pythagorean_theorem_l1776_177638

theorem pythagorean_theorem (a b c : ℝ) (h : a^2 + b^2 = c^2) : c^2 = a^2 + b^2 :=
sorry

end NUMINAMATH_GPT_pythagorean_theorem_l1776_177638


namespace NUMINAMATH_GPT_worker_saves_one_third_l1776_177693

variable {P : ℝ} 
variable {f : ℝ}

theorem worker_saves_one_third (h : P ≠ 0) (h_eq : 12 * f * P = 6 * (1 - f) * P) : 
  f = 1 / 3 :=
sorry

end NUMINAMATH_GPT_worker_saves_one_third_l1776_177693


namespace NUMINAMATH_GPT_histogram_groups_l1776_177606

theorem histogram_groups 
  (max_height : ℕ)
  (min_height : ℕ)
  (class_interval : ℕ)
  (h_max : max_height = 176)
  (h_min : min_height = 136)
  (h_interval : class_interval = 6) :
  Nat.ceil ((max_height - min_height) / class_interval) = 7 :=
by
  sorry

end NUMINAMATH_GPT_histogram_groups_l1776_177606


namespace NUMINAMATH_GPT_dan_marbles_l1776_177698

theorem dan_marbles (original_marbles : ℕ) (given_marbles : ℕ) (remaining_marbles : ℕ) : 
  original_marbles = 64 ∧ given_marbles = 14 → remaining_marbles = 50 := 
by 
  sorry

end NUMINAMATH_GPT_dan_marbles_l1776_177698


namespace NUMINAMATH_GPT_value_of_a_m_minus_3n_l1776_177607

theorem value_of_a_m_minus_3n (a : ℝ) (m n : ℝ) (h1 : a^m = 8) (h2 : a^n = 2) : a^(m - 3 * n) = 1 :=
sorry

end NUMINAMATH_GPT_value_of_a_m_minus_3n_l1776_177607


namespace NUMINAMATH_GPT_degree_to_radian_l1776_177601

theorem degree_to_radian : (855 : ℝ) * (Real.pi / 180) = (59 / 12) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_degree_to_radian_l1776_177601


namespace NUMINAMATH_GPT_min_value_exp_sum_eq_4sqrt2_l1776_177617

theorem min_value_exp_sum_eq_4sqrt2 {a b : ℝ} (h : a + b = 3) : 2^a + 2^b ≥ 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_exp_sum_eq_4sqrt2_l1776_177617


namespace NUMINAMATH_GPT_interval_length_of_solutions_l1776_177694

theorem interval_length_of_solutions (a b : ℝ) :
  (∃ x : ℝ, a ≤ 3*x + 6 ∧ 3*x + 6 ≤ b) ∧ (∃ (l : ℝ), l = (b - a) / 3 ∧ l = 15) → b - a = 45 :=
by sorry

end NUMINAMATH_GPT_interval_length_of_solutions_l1776_177694


namespace NUMINAMATH_GPT_tea_bags_count_l1776_177610

theorem tea_bags_count (n : ℕ) 
  (h1 : 2 * n ≤ 41) 
  (h2 : 41 ≤ 3 * n) 
  (h3 : 2 * n ≤ 58) 
  (h4 : 58 ≤ 3 * n) : 
  n = 20 := by
  sorry

end NUMINAMATH_GPT_tea_bags_count_l1776_177610


namespace NUMINAMATH_GPT_range_of_x_plus_y_l1776_177614

theorem range_of_x_plus_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^2 + 2 * x * y + 4 * y^2 = 1) : 0 < x + y ∧ x + y < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_plus_y_l1776_177614


namespace NUMINAMATH_GPT_deer_meat_distribution_l1776_177680

theorem deer_meat_distribution :
  ∃ (a1 a2 a3 a4 a5 : ℕ), a1 > a2 ∧ a2 > a3 ∧ a3 > a4 ∧ a4 > a5 ∧
  (a1 + a2 + a3 + a4 + a5 = 500) ∧
  (a2 + a3 + a4 = 300) :=
sorry

end NUMINAMATH_GPT_deer_meat_distribution_l1776_177680


namespace NUMINAMATH_GPT_units_digit_div_product_l1776_177657

theorem units_digit_div_product :
  (30 * 31 * 32 * 33 * 34 * 35) / 14000 % 10 = 2 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_div_product_l1776_177657


namespace NUMINAMATH_GPT_Rachel_total_earnings_l1776_177647

-- Define the constants for the conditions
def hourly_wage : ℝ := 12
def people_served : ℕ := 20
def tip_per_person : ℝ := 1.25

-- Define the problem
def total_money_made : ℝ := hourly_wage + (people_served * tip_per_person)

-- State the theorem to be proved
theorem Rachel_total_earnings : total_money_made = 37 := by
  sorry

end NUMINAMATH_GPT_Rachel_total_earnings_l1776_177647


namespace NUMINAMATH_GPT_mod_inverse_5_221_l1776_177656

theorem mod_inverse_5_221 : ∃ x : ℤ, 0 ≤ x ∧ x < 221 ∧ (5 * x) % 221 = 1 % 221 :=
by
  use 177
  sorry

end NUMINAMATH_GPT_mod_inverse_5_221_l1776_177656


namespace NUMINAMATH_GPT_prove_a_21022_le_1_l1776_177643

-- Define the sequence a_n
variable (a : ℕ → ℝ)

-- Conditions for the sequence
axiom seq_condition {n : ℕ} (hn : n ≥ 1) :
  (a (n + 1))^2 + a n * a (n + 2) ≤ a n + a (n + 2)

-- Positive real numbers condition
axiom seq_positive {n : ℕ} (hn : n ≥ 1) :
  a n > 0

-- The main theorem to prove
theorem prove_a_21022_le_1 :
  a 21022 ≤ 1 :=
sorry

end NUMINAMATH_GPT_prove_a_21022_le_1_l1776_177643


namespace NUMINAMATH_GPT_sum_of_squares_of_four_consecutive_even_numbers_eq_344_l1776_177661

theorem sum_of_squares_of_four_consecutive_even_numbers_eq_344 (n : ℤ) 
  (h : n + (n + 2) + (n + 4) + (n + 6) = 36) : 
  n^2 + (n + 2)^2 + (n + 4)^2 + (n + 6)^2 = 344 :=
by sorry

end NUMINAMATH_GPT_sum_of_squares_of_four_consecutive_even_numbers_eq_344_l1776_177661


namespace NUMINAMATH_GPT_total_jellybeans_needed_l1776_177671

def large_glass_jellybeans : ℕ := 50
def small_glass_jellybeans : ℕ := large_glass_jellybeans / 2
def num_large_glasses : ℕ := 5
def num_small_glasses : ℕ := 3

theorem total_jellybeans_needed : 
  (num_large_glasses * large_glass_jellybeans) + (num_small_glasses * small_glass_jellybeans) = 325 := 
by
  sorry

end NUMINAMATH_GPT_total_jellybeans_needed_l1776_177671


namespace NUMINAMATH_GPT_eric_time_ratio_l1776_177615

-- Defining the problem context
def eric_runs : ℕ := 20
def eric_jogs : ℕ := 10
def eric_return_time : ℕ := 90

-- The ratio is represented as a fraction
def ratio (a b : ℕ) := a / b

-- Stating the theorem
theorem eric_time_ratio :
  ratio eric_return_time (eric_runs + eric_jogs) = 3 :=
by
  sorry

end NUMINAMATH_GPT_eric_time_ratio_l1776_177615


namespace NUMINAMATH_GPT_calc_g_inv_sum_l1776_177620

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 3 * x - x * x

noncomputable def g_inv (y : ℝ) : ℝ := 
  if y = -4 then 4
  else if y = 0 then 3
  else if y = 4 then -1
  else 0

theorem calc_g_inv_sum : g_inv (-4) + g_inv 0 + g_inv 4 = 6 :=
by
  sorry

end NUMINAMATH_GPT_calc_g_inv_sum_l1776_177620


namespace NUMINAMATH_GPT_range_of_m_l1776_177669

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, (x - (m^2 - 2 * m + 4) * y - 6 > 0) ↔ (x, y) ≠ (-1, -1)) →
  -1 ≤ m ∧ m ≤ 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_m_l1776_177669


namespace NUMINAMATH_GPT_average_time_relay_race_l1776_177649

theorem average_time_relay_race :
  let dawson_time := 38
  let henry_time := 7
  let total_legs := 2
  (dawson_time + henry_time) / total_legs = 22.5 :=
by
  sorry

end NUMINAMATH_GPT_average_time_relay_race_l1776_177649


namespace NUMINAMATH_GPT_find_number_l1776_177673

theorem find_number (X : ℝ) (h : 0.8 * X = 0.7 * 60.00000000000001 + 30) : X = 90.00000000000001 :=
sorry

end NUMINAMATH_GPT_find_number_l1776_177673


namespace NUMINAMATH_GPT_christopher_strolled_5_miles_l1776_177660

theorem christopher_strolled_5_miles (s t : ℝ) (hs : s = 4) (ht : t = 1.25) : s * t = 5 :=
by
  rw [hs, ht]
  norm_num

end NUMINAMATH_GPT_christopher_strolled_5_miles_l1776_177660


namespace NUMINAMATH_GPT_state_B_more_candidates_l1776_177687

theorem state_B_more_candidates (appeared : ℕ) (selected_A_pct selected_B_pct : ℝ)
  (h1 : appeared = 8000)
  (h2 : selected_A_pct = 0.06)
  (h3 : selected_B_pct = 0.07) :
  (selected_B_pct * appeared - selected_A_pct * appeared = 80) :=
by
  sorry

end NUMINAMATH_GPT_state_B_more_candidates_l1776_177687


namespace NUMINAMATH_GPT_correct_avg_and_mode_l1776_177633

-- Define the conditions and correct answers
def avgIncorrect : ℚ := 13.5
def medianIncorrect : ℚ := 12
def modeCorrect : ℚ := 16
def totalNumbers : ℕ := 25
def incorrectNums : List ℚ := [33.5, 47.75, 58.5, 19/2]
def correctNums : List ℚ := [43.5, 56.25, 68.5, 21/2]

noncomputable def correctSum : ℚ := (avgIncorrect * totalNumbers) + (correctNums.sum - incorrectNums.sum)
noncomputable def correctAvg : ℚ := correctSum / totalNumbers

theorem correct_avg_and_mode :
  correctAvg = 367 / 25 ∧ modeCorrect = 16 :=
by
  sorry

end NUMINAMATH_GPT_correct_avg_and_mode_l1776_177633


namespace NUMINAMATH_GPT_cereal_difference_l1776_177611

-- Variables to represent the amounts of cereal in each box
variable (A B C : ℕ)

-- Define the conditions given in the problem
def problem_conditions : Prop :=
  A = 14 ∧
  B = A / 2 ∧
  A + B + C = 33

-- Prove the desired conclusion under these conditions
theorem cereal_difference
  (h : problem_conditions A B C) :
  C - B = 5 :=
sorry

end NUMINAMATH_GPT_cereal_difference_l1776_177611


namespace NUMINAMATH_GPT_mary_time_l1776_177696

-- Define the main entities for the problem
variables (mary_days : ℕ) (rosy_days : ℕ)
variable (rosy_efficiency_factor : ℝ) -- Rosy's efficiency factor compared to Mary

-- Given conditions
def rosy_efficient := rosy_efficiency_factor = 1.4
def rosy_time := rosy_days = 20

-- Problem Statement
theorem mary_time (h1 : rosy_efficient rosy_efficiency_factor) (h2 : rosy_time rosy_days) : mary_days = 28 :=
by
  sorry

end NUMINAMATH_GPT_mary_time_l1776_177696


namespace NUMINAMATH_GPT_quadratic_roots_range_quadratic_root_condition_l1776_177616

-- Problem 1: Prove that the range of real number \(k\) for which the quadratic 
-- equation \(x^{2} + (2k + 1)x + k^{2} + 1 = 0\) has two distinct real roots is \(k > \frac{3}{4}\). 
theorem quadratic_roots_range (k : ℝ) : 
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x^2 + (2*k+1)*x + k^2 + 1 = 0) ↔ (k > 3/4) := 
sorry

-- Problem 2: Given \(k > \frac{3}{4}\), prove that if the roots \(x₁\) and \(x₂\) of 
-- the equation satisfy \( |x₁| + |x₂| = x₁ \cdot x₂ \), then \( k = 2 \).
theorem quadratic_root_condition (k : ℝ) 
    (hk : k > 3 / 4)
    (x₁ x₂ : ℝ)
    (h₁ : x₁^2 + (2*k+1)*x₁ + k^2 + 1 = 0)
    (h₂ : x₂^2 + (2*k+1)*x₂ + k^2 + 1 = 0)
    (h3 : |x₁| + |x₂| = x₁ * x₂) : 
    k = 2 := 
sorry

end NUMINAMATH_GPT_quadratic_roots_range_quadratic_root_condition_l1776_177616


namespace NUMINAMATH_GPT_kelsey_total_distance_l1776_177658

-- Define the constants and variables involved
def total_distance (total_time : ℕ) (speed1 speed2 half_dist1 half_dist2 : ℕ) : ℕ :=
  let T1 := half_dist1 / speed1
  let T2 := half_dist2 / speed2
  let T := T1 + T2
  total_time

-- Prove the equivalency given the conditions
theorem kelsey_total_distance (total_time : ℕ) (speed1 speed2 : ℕ) : 
  (total_time = 10) ∧ (speed1 = 25) ∧ (speed2 = 40)  →
  ∃ D, D = 307 ∧ (10 = D / 50 + D / 80) :=
by 
  intro h
  have h_total_time := h.1
  have h_speed1 := h.2.1
  have h_speed2 := h.2.2
  -- Need to prove the statement using provided conditions
  let D := 307
  sorry

end NUMINAMATH_GPT_kelsey_total_distance_l1776_177658


namespace NUMINAMATH_GPT_steven_total_seeds_l1776_177631

-- Definitions based on the conditions
def apple_seed_count := 6
def pear_seed_count := 2
def grape_seed_count := 3

def apples_set_aside := 4
def pears_set_aside := 3
def grapes_set_aside := 9

def additional_seeds_needed := 3

-- The total seeds Steven already has
def total_seeds_from_fruits : ℕ :=
  apples_set_aside * apple_seed_count +
  pears_set_aside * pear_seed_count +
  grapes_set_aside * grape_seed_count

-- The total number of seeds Steven needs to collect, as given by the problem's solution
def total_seeds_needed : ℕ :=
  total_seeds_from_fruits + additional_seeds_needed

-- The actual proof statement
theorem steven_total_seeds : total_seeds_needed = 60 :=
  by
    sorry

end NUMINAMATH_GPT_steven_total_seeds_l1776_177631


namespace NUMINAMATH_GPT_sum_of_distinct_squares_l1776_177613

theorem sum_of_distinct_squares:
  ∀ (a b c : ℕ),
  a + b + c = 23 ∧ Nat.gcd a b + Nat.gcd b c + Nat.gcd c a = 9 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 + c^2 = 179 ∨ a^2 + b^2 + c^2 = 259 →
  a^2 + b^2 + c^2 = 438 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_distinct_squares_l1776_177613


namespace NUMINAMATH_GPT_unw_touchable_area_l1776_177625

-- Define the conditions
def ball_radius : ℝ := 1
def container_edge_length : ℝ := 5

-- Define the surface area that the ball can never touch
theorem unw_touchable_area : (ball_radius = 1) ∧ (container_edge_length = 5) → 
  let total_unreachable_area := 120
  let overlapping_area := 24
  let unreachable_area := total_unreachable_area - overlapping_area
  unreachable_area = 96 :=
by
  intros
  sorry

end NUMINAMATH_GPT_unw_touchable_area_l1776_177625


namespace NUMINAMATH_GPT_sum_of_midpoints_y_coordinates_l1776_177644

theorem sum_of_midpoints_y_coordinates (d e f : ℝ) (h : d + e + f = 15) : 
  (d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 15 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_midpoints_y_coordinates_l1776_177644


namespace NUMINAMATH_GPT_rectangular_field_area_l1776_177675

-- Given a rectangle with one side 4 meters and diagonal 5 meters, prove that its area is 12 square meters.
theorem rectangular_field_area
  (w l d : ℝ)
  (h_w : w = 4)
  (h_d : d = 5)
  (h_pythagoras : w^2 + l^2 = d^2) :
  w * l = 12 := 
by
  sorry

end NUMINAMATH_GPT_rectangular_field_area_l1776_177675


namespace NUMINAMATH_GPT_apples_per_basket_holds_15_l1776_177641

-- Conditions as Definitions
def trees := 10
def total_apples := 3000
def baskets_per_tree := 20

-- Definition for apples per tree (from the given total apples and number of trees)
def apples_per_tree : ℕ := total_apples / trees

-- Definition for apples per basket (from apples per tree and baskets per tree)
def apples_per_basket : ℕ := apples_per_tree / baskets_per_tree

-- The statement to prove the equivalent mathematical problem
theorem apples_per_basket_holds_15 
  (H1 : trees = 10)
  (H2 : total_apples = 3000)
  (H3 : baskets_per_tree = 20) :
  apples_per_basket = 15 :=
by 
  sorry

end NUMINAMATH_GPT_apples_per_basket_holds_15_l1776_177641


namespace NUMINAMATH_GPT_problem_statement_l1776_177648

def is_ideal_circle (circle : ℝ × ℝ → ℝ) (l : ℝ × ℝ → ℝ) : Prop :=
  ∃ P Q : ℝ × ℝ, (circle P = 0 ∧ circle Q = 0) ∧ (abs (l P) = 1 ∧ abs (l Q) = 1)

noncomputable def line_l (p : ℝ × ℝ) : ℝ := 3 * p.1 + 4 * p.2 - 12

noncomputable def circle_D (p : ℝ × ℝ) : ℝ := (p.1 - 4) ^ 2 + (p.2 - 4) ^ 2 - 16

theorem problem_statement : is_ideal_circle circle_D line_l :=
sorry  -- The proof would go here

end NUMINAMATH_GPT_problem_statement_l1776_177648


namespace NUMINAMATH_GPT_slices_left_for_lunch_tomorrow_l1776_177639

-- Definitions according to conditions
def initial_slices : ℕ := 12
def slices_eaten_for_lunch := initial_slices / 2
def remaining_slices_after_lunch := initial_slices - slices_eaten_for_lunch
def slices_eaten_for_dinner := 1 / 3 * remaining_slices_after_lunch
def remaining_slices_after_dinner := remaining_slices_after_lunch - slices_eaten_for_dinner
def slices_shared_with_friend := 1 / 4 * remaining_slices_after_dinner
def remaining_slices_after_sharing := remaining_slices_after_dinner - slices_shared_with_friend
def slices_eaten_by_sibling := if (1 / 5 * remaining_slices_after_sharing < 1) then 0 else 1 / 5 * remaining_slices_after_sharing
def remaining_slices_after_sibling := remaining_slices_after_sharing - slices_eaten_by_sibling

-- Lean statement of the proof problem
theorem slices_left_for_lunch_tomorrow : remaining_slices_after_sibling = 3 := by
  sorry

end NUMINAMATH_GPT_slices_left_for_lunch_tomorrow_l1776_177639


namespace NUMINAMATH_GPT_john_money_left_l1776_177686

theorem john_money_left 
  (start_amount : ℝ := 100) 
  (price_roast : ℝ := 17)
  (price_vegetables : ℝ := 11)
  (price_wine : ℝ := 12)
  (price_dessert : ℝ := 8)
  (price_bread : ℝ := 4)
  (price_milk : ℝ := 2)
  (discount_rate : ℝ := 0.15)
  (tax_rate : ℝ := 0.05)
  (total_cost := price_roast + price_vegetables + price_wine + price_dessert + price_bread + price_milk)
  (discount_amount := discount_rate * total_cost)
  (discounted_total := total_cost - discount_amount)
  (tax_amount := tax_rate * discounted_total)
  (final_amount := discounted_total + tax_amount)
  : start_amount - final_amount = 51.80 := sorry

end NUMINAMATH_GPT_john_money_left_l1776_177686


namespace NUMINAMATH_GPT_find_k_l1776_177651

noncomputable def series (k : ℝ) : ℝ := ∑' n, (7 * n - 2) / k^n

theorem find_k (k : ℝ) (h₁ : 1 < k) (h₂ : series k = 17 / 2) : k = 17 / 7 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1776_177651


namespace NUMINAMATH_GPT_weekly_earnings_before_rent_l1776_177626

theorem weekly_earnings_before_rent (EarningsAfterRent : ℝ) (weeks : ℕ) (rentPerWeek : ℝ) :
  EarningsAfterRent = 93899 → weeks = 233 → rentPerWeek = 49 →
  ((EarningsAfterRent + rentPerWeek * weeks) / weeks) = 451.99 :=
by
  intros H1 H2 H3
  -- convert the assumptions to the required form
  rw [H1, H2, H3]
  -- provide the objective statement
  change ((93899 + 49 * 233) / 233) = 451.99
  -- leave the final proof details as a sorry for now
  sorry

end NUMINAMATH_GPT_weekly_earnings_before_rent_l1776_177626


namespace NUMINAMATH_GPT_resulting_polygon_sides_l1776_177646

/-
Problem statement: 

Construct a regular pentagon on one side of a regular heptagon.
On one non-adjacent side of the pentagon, construct a regular hexagon.
On a non-adjacent side of the hexagon, construct an octagon.
Continue to construct regular polygons in the same way, until you construct a nonagon.
How many sides does the resulting polygon have?

Given facts:
1. Start with a heptagon (7 sides).
2. Construct a pentagon (5 sides) on one side of the heptagon.
3. Construct a hexagon (6 sides) on a non-adjacent side of the pentagon.
4. Construct an octagon (8 sides) on a non-adjacent side of the hexagon.
5. Construct a nonagon (9 sides) on a non-adjacent side of the octagon.
-/

def heptagon_sides : ℕ := 7
def pentagon_sides : ℕ := 5
def hexagon_sides : ℕ := 6
def octagon_sides : ℕ := 8
def nonagon_sides : ℕ := 9

theorem resulting_polygon_sides : 
  (heptagon_sides + nonagon_sides - 2 * 1) + (pentagon_sides + hexagon_sides + octagon_sides - 3 * 2) = 27 := by
  sorry

end NUMINAMATH_GPT_resulting_polygon_sides_l1776_177646


namespace NUMINAMATH_GPT_initial_volume_solution_l1776_177636

variable (V : ℝ)

theorem initial_volume_solution
  (h1 : 0.35 * V + 1.8 = 0.50 * (V + 1.8)) :
  V = 6 :=
by
  sorry

end NUMINAMATH_GPT_initial_volume_solution_l1776_177636
