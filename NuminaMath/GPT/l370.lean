import Mathlib

namespace incorrect_statement_is_B_l370_37012

-- Define the conditions
def genotype_AaBb_meiosis_results (sperm_genotypes : List String) : Prop :=
  sperm_genotypes = ["AB", "Ab", "aB", "ab"]

def spermatogonial_cell_AaXbY (malformed_sperm_genotype : String) (other_sperm_genotypes : List String) : Prop :=
  malformed_sperm_genotype = "AAaY" ∧ other_sperm_genotypes = ["aY", "X^b", "X^b"]

def spermatogonial_secondary_spermatocyte_Y_chromosomes (contains_two_Y : Bool) : Prop :=
  ¬ contains_two_Y

def female_animal_meiosis (primary_oocyte_alleles : Nat) (max_oocyte_b_alleles : Nat) : Prop :=
  primary_oocyte_alleles = 10 ∧ max_oocyte_b_alleles ≤ 5

-- The main statement that needs to be proved
theorem incorrect_statement_is_B :
  ∃ (sperm_genotypes : List String) 
    (malformed_sperm_genotype : String) 
    (other_sperm_genotypes : List String) 
    (contains_two_Y : Bool) 
    (primary_oocyte_alleles max_oocyte_b_alleles : Nat),
    genotype_AaBb_meiosis_results sperm_genotypes ∧ 
    spermatogonial_cell_AaXbY malformed_sperm_genotype other_sperm_genotypes ∧ 
    spermatogonial_secondary_spermatocyte_Y_chromosomes contains_two_Y ∧ 
    female_animal_meiosis primary_oocyte_alleles max_oocyte_b_alleles 
    ∧ (malformed_sperm_genotype = "AAaY" → false) := 
sorry

end incorrect_statement_is_B_l370_37012


namespace cheryl_more_points_l370_37081

-- Define the number of each type of eggs each child found
def kevin_small_eggs : Nat := 5
def kevin_large_eggs : Nat := 3

def bonnie_small_eggs : Nat := 13
def bonnie_medium_eggs : Nat := 7
def bonnie_large_eggs : Nat := 2

def george_small_eggs : Nat := 9
def george_medium_eggs : Nat := 6
def george_large_eggs : Nat := 1

def cheryl_small_eggs : Nat := 56
def cheryl_medium_eggs : Nat := 30
def cheryl_large_eggs : Nat := 15

-- Define the points for each type of egg
def small_egg_points : Nat := 1
def medium_egg_points : Nat := 3
def large_egg_points : Nat := 5

-- Calculate the total points for each child
def kevin_points : Nat := kevin_small_eggs * small_egg_points + kevin_large_eggs * large_egg_points
def bonnie_points : Nat := bonnie_small_eggs * small_egg_points + bonnie_medium_eggs * medium_egg_points + bonnie_large_eggs * large_egg_points
def george_points : Nat := george_small_eggs * small_egg_points + george_medium_eggs * medium_egg_points + george_large_eggs * large_egg_points
def cheryl_points : Nat := cheryl_small_eggs * small_egg_points + cheryl_medium_eggs * medium_egg_points + cheryl_large_eggs * large_egg_points

-- Statement of the proof problem
theorem cheryl_more_points : cheryl_points - (kevin_points + bonnie_points + george_points) = 125 :=
by
  -- Here would go the proof steps
  sorry

end cheryl_more_points_l370_37081


namespace combined_average_speed_l370_37013

-- Definitions based on conditions
def distance_A : ℕ := 250
def time_A : ℕ := 4

def distance_B : ℕ := 480
def time_B : ℕ := 6

def distance_C : ℕ := 390
def time_C : ℕ := 5

def total_distance : ℕ := distance_A + distance_B + distance_C
def total_time : ℕ := time_A + time_B + time_C

-- Prove combined average speed
theorem combined_average_speed : (total_distance : ℚ) / (total_time : ℚ) = 74.67 :=
  by
    sorry

end combined_average_speed_l370_37013


namespace pine_sample_count_l370_37055

variable (total_saplings : ℕ)
variable (pine_saplings : ℕ)
variable (sample_size : ℕ)

theorem pine_sample_count (h1 : total_saplings = 30000) (h2 : pine_saplings = 4000) (h3 : sample_size = 150) :
  pine_saplings * sample_size / total_saplings = 20 := 
sorry

end pine_sample_count_l370_37055


namespace value_of_f_sum_l370_37093

variable (a b c m : ℝ)

def f (x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2

theorem value_of_f_sum :
  f a b c 5 + f a b c (-5) = 4 :=
by
  sorry

end value_of_f_sum_l370_37093


namespace polar_to_rectangular_coordinates_l370_37078

theorem polar_to_rectangular_coordinates :
  let r := 2
  let θ := Real.pi / 3
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (x, y) = (1, Real.sqrt 3) :=
by
  sorry

end polar_to_rectangular_coordinates_l370_37078


namespace smaller_denom_is_five_l370_37007

-- Define the conditions
def num_smaller_bills : ℕ := 4
def num_ten_dollar_bills : ℕ := 8
def total_bills : ℕ := num_smaller_bills + num_ten_dollar_bills
def ten_dollar_bill_value : ℕ := 10
def total_value : ℕ := 100

-- Define the smaller denomination value
def value_smaller_denom (x : ℕ) : Prop :=
  num_smaller_bills * x + num_ten_dollar_bills * ten_dollar_bill_value = total_value

-- Prove that the value of the smaller denomination bill is 5
theorem smaller_denom_is_five : value_smaller_denom 5 :=
by
  sorry

end smaller_denom_is_five_l370_37007


namespace chris_dana_shared_rest_days_l370_37042

/-- Chris's and Dana's working schedules -/
structure work_schedule where
  work_days : ℕ
  rest_days : ℕ

/-- Define Chris's and Dana's schedules -/
def Chris_schedule : work_schedule := { work_days := 5, rest_days := 2 }
def Dana_schedule : work_schedule := { work_days := 6, rest_days := 1 }

/-- Number of days to consider -/
def total_days : ℕ := 1200

/-- Combinatorial function to calculate the number of coinciding rest-days -/
noncomputable def coinciding_rest_days (schedule1 schedule2 : work_schedule) (days : ℕ) : ℕ :=
  (days / (Nat.lcm (schedule1.work_days + schedule1.rest_days) (schedule2.work_days + schedule2.rest_days)))

/-- The proof problem statement -/
theorem chris_dana_shared_rest_days : 
coinciding_rest_days Chris_schedule Dana_schedule total_days = 171 :=
by sorry

end chris_dana_shared_rest_days_l370_37042


namespace solve_complex_eq_l370_37015

theorem solve_complex_eq (z : ℂ) (h : z^2 = -100 - 64 * I) : z = 3.06 - 10.46 * I ∨ z = -3.06 + 10.46 * I :=
by
  sorry

end solve_complex_eq_l370_37015


namespace isosceles_triangle_largest_angle_l370_37009

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h_isosceles : A = B) (h_angles : A = 60 ∧ B = 60) :
  max A (max B C) = 60 :=
by
  sorry

end isosceles_triangle_largest_angle_l370_37009


namespace nailcutter_sound_count_l370_37005

-- Definitions based on conditions
def nails_per_person : ℕ := 20
def number_of_customers : ℕ := 3
def sound_per_nail : ℕ := 1

-- The statement to prove 
theorem nailcutter_sound_count :
  (nails_per_person * number_of_customers * sound_per_nail) = 60 := by
  sorry

end nailcutter_sound_count_l370_37005


namespace simplify_expression_l370_37026

theorem simplify_expression (x y : ℤ) (h1 : x = 1) (h2 : y = -2) :
  2 * x ^ 2 - (3 * (-5 / 3 * x ^ 2 + 2 / 3 * x * y) - (x * y - 3 * x ^ 2)) + 2 * x * y = 2 :=
by {
  sorry
}

end simplify_expression_l370_37026


namespace tangent_line_at_x_5_l370_37086

noncomputable def f : ℝ → ℝ := sorry

theorem tangent_line_at_x_5 :
  (∀ x, f x = -x + 8 → f 5 + deriv f 5 = 2) := sorry

end tangent_line_at_x_5_l370_37086


namespace max_pages_copied_l370_37019

-- Definitions based on conditions
def cents_per_page := 7 / 4
def budget_cents := 1500

-- The theorem to prove
theorem max_pages_copied (c : ℝ) (budget : ℝ) (h₁ : c = cents_per_page) (h₂ : budget = budget_cents) : 
  ⌊(budget / c)⌋ = 857 :=
sorry

end max_pages_copied_l370_37019


namespace tv_selection_l370_37047

theorem tv_selection (A B : ℕ) (hA : A = 4) (hB : B = 5) : 
  ∃ n, n = 3 ∧ (∃ k, k = 70 ∧ 
    (n = 1 ∧ k = A * (B * (B - 1) / 2) + A * (A - 1) / 2 * B)) :=
sorry

end tv_selection_l370_37047


namespace monotonically_increasing_condition_l370_37022

theorem monotonically_increasing_condition 
  (a b c d : ℝ) (h : 0 < a) :
  (∀ x : ℝ, 0 ≤ 3 * a * x ^ 2 + 2 * b * x + c) ↔ (b^2 - 3 * a * c ≤ 0) :=
by {
  sorry
}

end monotonically_increasing_condition_l370_37022


namespace fabric_woven_in_30_days_l370_37006

theorem fabric_woven_in_30_days :
  let a1 := 5
  let d := 16 / 29
  (30 * a1 + (30 * (30 - 1) / 2) * d) = 390 :=
by
  let a1 := 5
  let d := 16 / 29
  sorry

end fabric_woven_in_30_days_l370_37006


namespace Petya_wrong_example_l370_37034

def a := 8
def b := 128

theorem Petya_wrong_example : (a^7 ∣ b^3) ∧ ¬ (a^2 ∣ b) :=
by {
  -- Prove the divisibility conditions and the counterexample
  sorry
}

end Petya_wrong_example_l370_37034


namespace cyclic_sum_inequality_l370_37048

open Real

theorem cyclic_sum_inequality
  (a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / c + b^2 / a + c^2 / b) + (b^2 / c + c^2 / a + a^2 / b) + (c^2 / a + a^2 / b + b^2 / c) + 
  7 * (a + b + c) 
  ≥ ((a + b + c)^3) / (a * b + b * c + c * a) + (2 * (a * b + b * c + c * a)^2) / (a * b * c) := 
sorry

end cyclic_sum_inequality_l370_37048


namespace area_of_triangle_BCD_l370_37028

-- Define the points A, B, C, D
variables {A B C D : Type} 

-- Define the lengths of segments AC and CD
variables (AC CD : ℝ)
-- Define the area of triangle ABC
variables (area_ABC : ℝ)

-- Define height h
variables (h : ℝ)

-- Initial conditions
axiom length_AC : AC = 9
axiom length_CD : CD = 39
axiom area_ABC_is_36 : area_ABC = 36
axiom height_is_8 : h = (2 * area_ABC) / AC

-- Define the area of triangle BCD
def area_BCD (CD h : ℝ) : ℝ := 0.5 * CD * h

-- The theorem that we want to prove
theorem area_of_triangle_BCD : area_BCD 39 8 = 156 :=
by
  sorry

end area_of_triangle_BCD_l370_37028


namespace carB_speed_l370_37071

variable (distance : ℝ) (time : ℝ) (ratio : ℝ) (speedB : ℝ)

theorem carB_speed (h1 : distance = 240) (h2 : time = 1.5) (h3 : ratio = 3 / 5) 
(h4 : (speedB + ratio * speedB) * time = distance) : speedB = 100 := 
by 
  sorry

end carB_speed_l370_37071


namespace continuous_function_fixed_point_l370_37038

variable (f : ℝ → ℝ)
variable (h_cont : Continuous f)
variable (h_comp : ∀ x : ℝ, ∃ n : ℕ, n > 0 ∧ (f^[n] x = 1))

theorem continuous_function_fixed_point : f 1 = 1 := 
by
  sorry

end continuous_function_fixed_point_l370_37038


namespace negation_of_universal_proposition_l370_37067

variable {R : Type*} [LinearOrderedField R]
variable (f : R → R)

theorem negation_of_universal_proposition :
  (∀ x1 x2 : R, (f x2 - f x1) * (x2 - x1) ≥ 0) →
  ∃ x1 x2 : R, (f x2 - f x1) * (x2 - x1) < 0 :=
sorry

end negation_of_universal_proposition_l370_37067


namespace probability_of_rolling_four_threes_l370_37046
open BigOperators

def probability_four_threes (n : ℕ) (k : ℕ) (p : ℚ) (q : ℚ) : ℚ := 
  (n.choose k) * (p ^ k) * (q ^ (n - k))

theorem probability_of_rolling_four_threes : 
  probability_four_threes 5 4 (1 / 10) (9 / 10) = 9 / 20000 := 
by 
  sorry

end probability_of_rolling_four_threes_l370_37046


namespace women_more_than_men_l370_37014

def men (W : ℕ) : ℕ := (5 * W) / 11

theorem women_more_than_men (M W : ℕ) (h1 : M + W = 16) (h2 : M = (5 * W) / 11) : W - M = 6 :=
by
  sorry

end women_more_than_men_l370_37014


namespace no_rational_satisfies_l370_37083

theorem no_rational_satisfies (a b c d : ℚ) : ¬ ((a + b * Real.sqrt 3)^4 + (c + d * Real.sqrt 3)^4 = 1 + Real.sqrt 3) :=
sorry

end no_rational_satisfies_l370_37083


namespace work_problem_l370_37066

theorem work_problem (W : ℕ) (h1: ∀ w, w = W → (24 * w + 1 = 73)) : W = 3 :=
by {
  -- Insert proof here
  sorry
}

end work_problem_l370_37066


namespace lucy_total_cost_for_lamp_and_table_l370_37075

noncomputable def original_price_lamp : ℝ := 200 / 1.2

noncomputable def table_price : ℝ := 2 * original_price_lamp

noncomputable def total_cost_paid (lamp_cost discounted_price table_price: ℝ) :=
  lamp_cost + table_price

theorem lucy_total_cost_for_lamp_and_table :
  total_cost_paid 20 (original_price_lamp * 0.6) table_price = 353.34 :=
by
  let lamp_original_price := original_price_lamp
  have h1 : original_price_lamp * (0.6 * (1 / 5)) = 20 := by sorry
  have h2 : table_price = 2 * original_price_lamp := by sorry
  have h3 : total_cost_paid 20 (original_price_lamp * 0.6) table_price = 20 + table_price := by sorry
  have h4 : table_price = 2 * (200 / 1.2) := by sorry
  have h5 : 20 + table_price = 353.34 := by sorry
  exact h5

end lucy_total_cost_for_lamp_and_table_l370_37075


namespace translation_vector_condition_l370_37074

theorem translation_vector_condition (m n : ℝ) :
  (∀ x : ℝ, 2 * (x - m) + n = 2 * x + 5) → n = 2 * m + 5 :=
by
  intro h
  -- proof can be filled here
  sorry

end translation_vector_condition_l370_37074


namespace part1_part2_1_part2_2_l370_37040

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 - x * Real.log x

theorem part1 (a : ℝ) :
  (∀ x : ℝ, x > 0 → (2 * a * x - Real.log x - 1) ≥ 0) ↔ a ≥ 0.5 := 
sorry

theorem part2_1 (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 < x2 ∧ f a x1 = x1 ∧ f a x2 = x2) :
  0 < a ∧ a < 1 := 
sorry

theorem part2_2 (a x1 x2 : ℝ) (h1 : x1 < x2) (h2 : f a x1 = x1) (h3 : f a x2 = x2) (h4 : x2 ≥ 3 * x1) :
  x1 * x2 ≥ 9 / Real.exp 2 := 
sorry

end part1_part2_1_part2_2_l370_37040


namespace sum_base_49_l370_37025

-- Definitions of base b numbers and their base 10 conversion
def num_14_in_base (b : ℕ) : ℕ := b + 4
def num_17_in_base (b : ℕ) : ℕ := b + 7
def num_18_in_base (b : ℕ) : ℕ := b + 8
def num_6274_in_base (b : ℕ) : ℕ := 6 * b^3 + 2 * b^2 + 7 * b + 4

-- The question: Compute 14 + 17 + 18 in base b
def sum_in_base (b : ℕ) : ℕ := 14 + 17 + 18

-- The main statement to prove
theorem sum_base_49 (b : ℕ) (h : (num_14_in_base b) * (num_17_in_base b) * (num_18_in_base b) = num_6274_in_base (b)) :
  sum_in_base b = 49 :=
by sorry

end sum_base_49_l370_37025


namespace range_of_x_for_obtuse_angle_l370_37057

def vectors_are_obtuse (a b : ℝ × ℝ) : Prop :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  dot_product < 0

theorem range_of_x_for_obtuse_angle :
  ∀ (x : ℝ), vectors_are_obtuse (1, 3) (x, -1) ↔ (x < -1/3 ∨ (-1/3 < x ∧ x < 3)) :=
by
  sorry

end range_of_x_for_obtuse_angle_l370_37057


namespace quadratic_has_two_distinct_real_roots_l370_37064

theorem quadratic_has_two_distinct_real_roots (k : ℝ) :
  let a := 1
  let b := -(k + 3)
  let c := 2 * k + 1
  let Δ := b^2 - 4 * a * c
  Δ > 0 :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l370_37064


namespace parallel_line_slope_l370_37063

theorem parallel_line_slope (x y : ℝ) :
  ∃ m b : ℝ, (3 * x - 6 * y = 21) → ∀ (x₁ y₁ : ℝ), (3 * x₁ - 6 * y₁ = 21) → m = 1 / 2 :=
by
  sorry

end parallel_line_slope_l370_37063


namespace intersection_x_value_l370_37052

/-- Prove that the x-value at the point of intersection of the lines
    y = 5x - 28 and 3x + y = 120 is 18.5 -/
theorem intersection_x_value :
  ∃ x y : ℝ, (y = 5 * x - 28) ∧ (3 * x + y = 120) ∧ (x = 18.5) :=
by
  sorry

end intersection_x_value_l370_37052


namespace problem1_problem2_problem3_l370_37002

-- Problem 1
theorem problem1 :
  1 - 1^2022 + ((-1/2)^2) * (-2)^3 * (-2)^2 - |Real.pi - 3.14|^0 = -10 :=
by sorry

-- Problem 2
variables (a b : ℝ)

theorem problem2 :
  a^3 * (-b^3)^2 + (-2 * a * b)^3 = a^3 * b^6 - 8 * a^3 * b^3 :=
by sorry

-- Problem 3
theorem problem3 (a b : ℝ) :
  (2 * a^3 * b^2 - 3 * a^2 * b - 4 * a) * 2 * b = 4 * a^3 * b^3 - 6 * a^2 * b^2 - 8 * a * b :=
by sorry

end problem1_problem2_problem3_l370_37002


namespace min_output_to_avoid_losses_l370_37060

theorem min_output_to_avoid_losses (x : ℝ) (y : ℝ) (h : y = 0.1 * x - 150) : y ≥ 0 → x ≥ 1500 :=
sorry

end min_output_to_avoid_losses_l370_37060


namespace sixth_term_sequence_l370_37003

theorem sixth_term_sequence (a : ℕ → ℕ) (h₁ : a 0 = 3) (h₂ : ∀ n, a (n + 1) = (a n)^2) : 
  a 5 = 1853020188851841 := 
by {
  sorry
}

end sixth_term_sequence_l370_37003


namespace area_of_sector_l370_37036

-- Given conditions
def central_angle : ℝ := 2
def perimeter : ℝ := 8

-- Define variables and expressions
variable (r l : ℝ)

-- Equations based on the conditions
def eq1 := l + 2 * r = perimeter
def eq2 := l = central_angle * r

-- Assertion of the correct answer
theorem area_of_sector : ∃ r l : ℝ, eq1 r l ∧ eq2 r l ∧ (1 / 2 * l * r = 4) := by
  sorry

end area_of_sector_l370_37036


namespace sqrt_expression_eq_two_l370_37030

theorem sqrt_expression_eq_two : 
  (Real.sqrt 3) * (Real.sqrt 3 - 1 / (Real.sqrt 3)) = 2 := 
  sorry

end sqrt_expression_eq_two_l370_37030


namespace part_I_part_II_l370_37050

open Real

noncomputable def f (x : ℝ) : ℝ := abs (2 * x + 1) - abs (x - 4)

theorem part_I (x : ℝ) : f x > 0 ↔ (x > 1 ∨ x < -5) := 
sorry

theorem part_II (m : ℝ) : (∀ x : ℝ, f x + 3 * abs (x - 4) > m) ↔ (m < 9) :=
sorry

end part_I_part_II_l370_37050


namespace algebraic_expression_value_l370_37058

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y = -4) :
  (2 * y - x) ^ 2 - 2 * x + 4 * y - 1 = 23 :=
by
  sorry

end algebraic_expression_value_l370_37058


namespace meet_at_centroid_l370_37041

-- Definitions of positions
def Harry : ℝ × ℝ := (10, -3)
def Sandy : ℝ × ℝ := (2, 7)
def Ron : ℝ × ℝ := (6, 1)

-- Mathematical proof problem statement
theorem meet_at_centroid : 
    (Harry.1 + Sandy.1 + Ron.1) / 3 = 6 ∧ (Harry.2 + Sandy.2 + Ron.2) / 3 = 5 / 3 := 
by
  sorry

end meet_at_centroid_l370_37041


namespace find_current_l370_37090

noncomputable def V : ℂ := 2 + 3 * Complex.I
noncomputable def Z : ℂ := 2 - 2 * Complex.I

theorem find_current : (V / Z) = (-1 / 4 : ℂ) + (5 / 4 : ℂ) * Complex.I := by
  sorry

end find_current_l370_37090


namespace find_two_digit_number_l370_37096

theorem find_two_digit_number (n : ℕ) (h1 : n % 9 = 7) (h2 : n % 7 = 5) (h3 : n % 3 = 1) (h4 : 10 ≤ n) (h5 : n < 100) : n = 61 := 
by
  sorry

end find_two_digit_number_l370_37096


namespace perpendicular_lines_k_value_l370_37027

theorem perpendicular_lines_k_value (k : ℝ) :
  (∀ x y : ℝ, k * x - y - 3 = 0 → x + (2 * k + 3) * y - 2 = 0) →
  k = -3 :=
by
  sorry

end perpendicular_lines_k_value_l370_37027


namespace probability_of_same_color_l370_37091

-- Defining the given conditions
def green_balls := 6
def red_balls := 4
def total_balls := green_balls + red_balls

def probability_same_color : ℚ :=
  let prob_green := (green_balls / total_balls) * (green_balls / total_balls)
  let prob_red := (red_balls / total_balls) * (red_balls / total_balls)
  prob_green + prob_red

-- Statement of the problem rewritten in Lean 4
theorem probability_of_same_color :
  probability_same_color = 13 / 25 :=
by
  sorry

end probability_of_same_color_l370_37091


namespace elmo_to_laura_books_ratio_l370_37079

-- Definitions of the conditions given in the problem
def ElmoBooks : ℕ := 24
def StuBooks : ℕ := 4
def LauraBooks : ℕ := 2 * StuBooks

-- Ratio calculation and proof of the ratio being 3:1
theorem elmo_to_laura_books_ratio : (ElmoBooks : ℚ) / (LauraBooks : ℚ) = 3 / 1 := by
  sorry

end elmo_to_laura_books_ratio_l370_37079


namespace find_speeds_l370_37062

noncomputable def speed_proof_problem (x y: ℝ) : Prop :=
  let distance_AB := 40
  let time_cyclist_start := 7 + 20 / 60
  let time_pedestrian_start := 4
  let time_cyclist_to_catch_up := (distance_AB / 2 - 10 / 3 * x) / (y - x)
  let time_pedestrian_meet := 10 / 3 + time_cyclist_to_catch_up + 1
  let time_second_cyclist_start := 8.5
  let dist_cyclist := y * (time_second_cyclist_start - time_pedestrian_start)
  let dist_pedestrian := x * time_pedestrian_meet 
  (x = 5 ∧ y = 30) ∧
  (time_cyclist_start - time_pedestrian_start = 10 / 3) ∧
  (dist_pedestrian + time_cyclist_to_catch_up * x = distance_AB / 2) ∧
  (dist_pedestrian + y * 1 = 40)

theorem find_speeds (x y: ℝ) :
  speed_proof_problem x y :=
sorry

end find_speeds_l370_37062


namespace initially_planned_days_l370_37011

theorem initially_planned_days (D : ℕ) (h1 : 6 * 3 + 10 * 3 = 6 * D) : D = 8 := by
  sorry

end initially_planned_days_l370_37011


namespace randy_total_trees_l370_37031

theorem randy_total_trees (mango_trees : ℕ) (coconut_trees : ℕ) 
  (h1 : mango_trees = 60) 
  (h2 : coconut_trees = (mango_trees / 2) - 5) : 
  mango_trees + coconut_trees = 85 :=
by
  sorry

end randy_total_trees_l370_37031


namespace cos_alpha_plus_pi_over_4_l370_37017

theorem cos_alpha_plus_pi_over_4
  (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.cos (α + β) = 3 / 5)
  (h4 : Real.sin (β - π / 4) = 5 / 13) : 
  Real.cos (α + π / 4) = 56 / 65 :=
by
  sorry 

end cos_alpha_plus_pi_over_4_l370_37017


namespace randy_trip_length_l370_37032

theorem randy_trip_length (x : ℝ) (h : x / 2 + 30 + x / 4 = x) : x = 120 :=
by
  sorry

end randy_trip_length_l370_37032


namespace complements_intersection_l370_37001

open Set

noncomputable def U : Set ℕ := { x | x ≤ 5 }
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 4}

theorem complements_intersection :
  (U \ A) ∩ (U \ B) = {0, 5} :=
by
  sorry

end complements_intersection_l370_37001


namespace Malcom_has_more_cards_l370_37018

-- Define the number of cards Brandon has
def Brandon_cards : ℕ := 20

-- Define the number of cards Malcom has initially, to be found
def Malcom_initial_cards (n : ℕ) := n

-- Define the given condition: Malcom has 14 cards left after giving away half of his cards
def Malcom_half_condition (n : ℕ) := n / 2 = 14

-- Prove that Malcom had 8 more cards than Brandon initially
theorem Malcom_has_more_cards (n : ℕ) (h : Malcom_half_condition n) :
  Malcom_initial_cards n - Brandon_cards = 8 :=
by
  sorry

end Malcom_has_more_cards_l370_37018


namespace neither_long_furred_nor_brown_dogs_is_8_l370_37085

def total_dogs : ℕ := 45
def long_furred_dogs : ℕ := 29
def brown_dogs : ℕ := 17
def long_furred_and_brown_dogs : ℕ := 9

def neither_long_furred_nor_brown_dogs : ℕ :=
  total_dogs - (long_furred_dogs + brown_dogs - long_furred_and_brown_dogs)

theorem neither_long_furred_nor_brown_dogs_is_8 :
  neither_long_furred_nor_brown_dogs = 8 := 
by 
  -- Here we can use substitution and calculation steps used in the solution
  sorry

end neither_long_furred_nor_brown_dogs_is_8_l370_37085


namespace distance_between_cities_l370_37088

variable (D : ℝ) -- D is the distance between City A and City B
variable (time_AB : ℝ) -- Time from City A to City B
variable (time_BA : ℝ) -- Time from City B to City A
variable (saved_time : ℝ) -- Time saved per trip
variable (avg_speed : ℝ) -- Average speed for the round trip with saved time

theorem distance_between_cities :
  time_AB = 6 → time_BA = 4.5 → saved_time = 0.5 → avg_speed = 90 →
  D = 427.5 :=
by
  sorry

end distance_between_cities_l370_37088


namespace right_to_left_evaluation_l370_37054

variable (a b c d : ℝ)

theorem right_to_left_evaluation :
  a / b - c + d = a / (b - c - d) :=
sorry

end right_to_left_evaluation_l370_37054


namespace smallest_solution_fraction_eq_l370_37072

theorem smallest_solution_fraction_eq (x : ℝ) (h : x ≠ 3) :
    3 * x / (x - 3) + (3 * x^2 - 27) / x = 16 ↔ x = (2 - Real.sqrt 31) / 3 := 
sorry

end smallest_solution_fraction_eq_l370_37072


namespace find_a_l370_37010

theorem find_a (a : ℝ) :
  let A := {5}
  let B := { x : ℝ | a * x - 1 = 0 }
  A ∩ B = B ↔ (a = 0 ∨ a = 1 / 5) :=
by
  sorry

end find_a_l370_37010


namespace geometric_progression_condition_l370_37065

variables (a b c : ℝ) (k n p : ℕ)

theorem geometric_progression_condition :
  (a / b) ^ (k - p) = (a / c) ^ (k - n) :=
sorry

end geometric_progression_condition_l370_37065


namespace quadratic_positive_intervals_l370_37099

-- Problem setup
def quadratic (x : ℝ) : ℝ := x^2 - x - 6

-- Define the roots of the quadratic function
def is_root (a b : ℝ) (f : ℝ → ℝ) := f a = 0 ∧ f b = 0

-- Proving the intervals where the quadratic function is greater than 0
theorem quadratic_positive_intervals :
  is_root (-2) 3 quadratic →
  { x : ℝ | quadratic x > 0 } = { x : ℝ | x < -2 } ∪ { x : ℝ | x > 3 } :=
by
  sorry

end quadratic_positive_intervals_l370_37099


namespace sum_is_zero_l370_37029

variable (a b c x y : ℝ)

theorem sum_is_zero (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
(h4 : a^3 + a * x + y = 0)
(h5 : b^3 + b * x + y = 0)
(h6 : c^3 + c * x + y = 0) : a + b + c = 0 :=
sorry

end sum_is_zero_l370_37029


namespace cost_of_shirt_l370_37077

theorem cost_of_shirt (J S : ℝ) (h1 : 3 * J + 2 * S = 69) (h2 : 2 * J + 3 * S = 71) : S = 15 :=
by
  sorry

end cost_of_shirt_l370_37077


namespace computer_multiplications_l370_37094

def rate : ℕ := 15000
def time : ℕ := 2 * 3600
def expected_multiplications : ℕ := 108000000

theorem computer_multiplications : rate * time = expected_multiplications := by
  sorry

end computer_multiplications_l370_37094


namespace find_direction_vector_l370_37098

def line_parametrization (v d : ℝ × ℝ) (t x y : ℝ) : ℝ × ℝ :=
  (v.fst + t * d.fst, v.snd + t * d.snd)

theorem find_direction_vector : 
  ∀ d: ℝ × ℝ, ∀ t: ℝ,
    ∀ (v : ℝ × ℝ) (x y : ℝ), 
    v = (-3, -1) → 
    y = (2 * x + 3) / 5 →
    x + 3 ≤ 0 →
    dist (line_parametrization v d t x y) (-3, -1) = t →
    d = (5/2, 1) :=
by
  intros d t v x y hv hy hcond hdist
  sorry

end find_direction_vector_l370_37098


namespace number_of_true_propositions_l370_37087

noncomputable def proposition1 : Prop := ∀ (x : ℝ), x^2 - 3 * x + 2 > 0
noncomputable def proposition2 : Prop := ∃ (x : ℚ), x^2 = 2
noncomputable def proposition3 : Prop := ∃ (x : ℝ), x^2 - 1 = 0
noncomputable def proposition4 : Prop := ∀ (x : ℝ), 4 * x^2 > 2 * x - 1 + 3 * x^2

theorem number_of_true_propositions : (¬ proposition1 ∧ ¬ proposition2 ∧ proposition3 ∧ ¬ proposition4) → 1 = 1 :=
by
  intros
  sorry

end number_of_true_propositions_l370_37087


namespace Q_investment_time_l370_37073

theorem Q_investment_time  
  (P Q x t : ℝ)
  (h_ratio_investments : P = 7 * x ∧ Q = 5 * x)
  (h_ratio_profits : (7 * x * 10) / (5 * x * t) = 7 / 10) :
  t = 20 :=
by {
  sorry
}

end Q_investment_time_l370_37073


namespace xiaomin_house_position_l370_37068

-- Define the initial position of the school at the origin
def school_pos : ℝ × ℝ := (0, 0)

-- Define the movement east and south from the school's position
def xiaomin_house_pos (east_distance south_distance : ℝ) : ℝ × ℝ :=
  (school_pos.1 + east_distance, school_pos.2 - south_distance)

-- The given conditions
def east_distance := 200
def south_distance := 150

-- The theorem stating Xiaomin's house position
theorem xiaomin_house_position :
  xiaomin_house_pos east_distance south_distance = (200, -150) :=
by
  -- Skipping the proof steps
  sorry

end xiaomin_house_position_l370_37068


namespace find_y_when_x_is_8_l370_37004

theorem find_y_when_x_is_8 (x y : ℕ) (k : ℕ) (h1 : x + y = 36) (h2 : x - y = 12) (h3 : x * y = k) (h4 : k = 288) : y = 36 :=
by
  -- Given the conditions
  sorry

end find_y_when_x_is_8_l370_37004


namespace total_selling_price_is_18000_l370_37045

def cost_price_per_meter : ℕ := 50
def loss_per_meter : ℕ := 5
def meters_sold : ℕ := 400

def selling_price_per_meter := cost_price_per_meter - loss_per_meter

def total_selling_price := selling_price_per_meter * meters_sold

theorem total_selling_price_is_18000 :
  total_selling_price = 18000 :=
sorry

end total_selling_price_is_18000_l370_37045


namespace prob_both_even_correct_l370_37020

-- Define the dice and verify their properties
def die1 := {n : ℕ // n ≥ 1 ∧ n ≤ 6}
def die2 := {n : ℕ // n ≥ 1 ∧ n ≤ 7}

-- Define the sets of even numbers for both dice
def even_die1 (n : die1) : Prop := n.1 % 2 = 0
def even_die2 (n : die2) : Prop := n.1 % 2 = 0

-- Define the probabilities of rolling an even number on each die
def prob_even_die1 := 3 / 6
def prob_even_die2 := 3 / 7

-- Calculate the combined probability
def prob_both_even := prob_even_die1 * prob_even_die2

-- The theorem stating the probability of both dice rolling even is 3/14
theorem prob_both_even_correct : prob_both_even = 3 / 14 :=
by
  -- Proof is omitted
  sorry

end prob_both_even_correct_l370_37020


namespace smallest_fraction_l370_37092

theorem smallest_fraction (f1 f2 f3 f4 f5 : ℚ) (h1 : f1 = 2 / 3) (h2 : f2 = 3 / 4) (h3 : f3 = 5 / 6) 
  (h4 : f4 = 5 / 8) (h5 : f5 = 11 / 12) : f4 = 5 / 8 ∧ f4 < f1 ∧ f4 < f2 ∧ f4 < f3 ∧ f4 < f5 := 
by 
  sorry

end smallest_fraction_l370_37092


namespace value_of_f_ln6_l370_37056

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then x + Real.exp x else -(x + Real.exp (-x))

theorem value_of_f_ln6 : (f (Real.log 6)) = Real.log 6 - (1/6) :=
by
  sorry

end value_of_f_ln6_l370_37056


namespace cos_B_value_l370_37049

-- Define the sides of the triangle
def AB : ℝ := 8
def AC : ℝ := 10
def right_angle_at_A : Prop := true

-- Define the cosine function within the context of the given triangle
noncomputable def cos_B : ℝ := AB / AC

-- The proof statement asserting the condition
theorem cos_B_value : cos_B = 4 / 5 :=
by
  -- Given conditions
  have h1 : AB = 8 := rfl
  have h2 : AC = 10 := rfl
  -- Direct computation
  sorry

end cos_B_value_l370_37049


namespace susan_avg_speed_l370_37069

variable (d1 d2 : ℕ) (s1 s2 : ℕ)

def time (d s : ℕ) : ℚ := d / s

theorem susan_avg_speed 
  (h1 : d1 = 40) 
  (h2 : s1 = 30) 
  (h3 : d2 = 40) 
  (h4 : s2 = 15) : 
  (d1 + d2) / (time d1 s1 + time d2 s2) = 20 := 
by 
  -- Sorry to skip the proof.
  sorry

end susan_avg_speed_l370_37069


namespace abs_inequality_m_eq_neg4_l370_37082

theorem abs_inequality_m_eq_neg4 (m : ℝ) : (∀ x : ℝ, |2 * x - m| ≤ |3 * x + 6|) ↔ (m = -4) :=
by
  sorry

end abs_inequality_m_eq_neg4_l370_37082


namespace remaining_people_statement_l370_37051

-- Definitions of conditions
def number_of_people : Nat := 10
def number_of_knights (K : Nat) : Prop := K ≤ number_of_people
def number_of_liars (L : Nat) : Prop := L ≤ number_of_people
def statement (s : String) : Prop := s = "There are more liars" ∨ s = "There are equal numbers"

-- Main theorem
theorem remaining_people_statement (K L : Nat) (h_total : K + L = number_of_people) 
  (h_knights_behavior : ∀ k, k < K → statement "There are equal numbers") 
  (h_liars_behavior : ∀ l, l < L → statement "There are more liars") :
  K = 5 → L = 5 → ∀ i, i < number_of_people → (i < 5 → statement "There are more liars") ∧ (i >= 5 → statement "There are equal numbers") := 
by
  sorry

end remaining_people_statement_l370_37051


namespace functional_equation_solution_l370_37080

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (y * f (x + y) + f x) = 4 * x + 2 * y * f (x + y)) →
  (∀ x : ℝ, f x = 2 * x) :=
sorry

end functional_equation_solution_l370_37080


namespace total_area_painted_correct_l370_37000

-- Defining the properties of the shed
def shed_w := 12  -- width in yards
def shed_l := 15  -- length in yards
def shed_h := 7   -- height in yards

-- Calculating area to be painted
def wall_area_1 := 2 * (shed_w * shed_h)
def wall_area_2 := 2 * (shed_l * shed_h)
def floor_ceiling_area := 2 * (shed_w * shed_l)
def total_painted_area := wall_area_1 + wall_area_2 + floor_ceiling_area

-- The theorem to be proved
theorem total_area_painted_correct :
  total_painted_area = 738 := by
  sorry

end total_area_painted_correct_l370_37000


namespace middle_digit_base_5_reversed_in_base_8_l370_37037

theorem middle_digit_base_5_reversed_in_base_8 (a b c : ℕ) (h₁ : 0 ≤ a ∧ a ≤ 4) (h₂ : 0 ≤ b ∧ b ≤ 4) 
  (h₃ : 0 ≤ c ∧ c ≤ 4) (h₄ : 25 * a + 5 * b + c = 64 * c + 8 * b + a) : b = 3 := 
by 
  sorry

end middle_digit_base_5_reversed_in_base_8_l370_37037


namespace find_eccentricity_l370_37023

variables {a b x_N x_M : ℝ}
variable {e : ℝ}

-- Conditions
def line_passes_through_N (x_N : ℝ) (x_M : ℝ) : Prop :=
x_N ≠ 0 ∧ x_N = 4 * x_M

def hyperbola (x y a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def midpoint_x_M (x_M : ℝ) : Prop :=
∃ (x1 x2 y1 y2 : ℝ), (x1 + x2) / 2 = x_M

-- Proof Problem
theorem find_eccentricity
  (hN : line_passes_through_N x_N x_M)
  (hC : hyperbola x_N 0 a b)
  (hM : midpoint_x_M x_M) :
  e = 2 :=
sorry

end find_eccentricity_l370_37023


namespace solve_inequality_l370_37021

-- Define the inequality
def inequality (a x : ℝ) : Prop := a * x^2 - (a + 2) * x + 2 < 0

-- Prove the solution sets for different values of a
theorem solve_inequality :
  ∀ (a : ℝ),
    (a = -1 → {x : ℝ | inequality a x} = {x | x < -2 ∨ x > 1}) ∧
    (a = 0 → {x : ℝ | inequality a x} = {x | x > 1}) ∧
    (a < 0 → {x : ℝ | inequality a x} = {x | x < 2 / a ∨ x > 1}) ∧
    (0 < a ∧ a < 2 → {x : ℝ | inequality a x} = {x | 1 < x ∧ x < 2 / a}) ∧
    (a = 2 → {x : ℝ | inequality a x} = ∅) ∧
    (a > 2 → {x : ℝ | inequality a x} = {x | 2 / a < x ∧ x < 1}) :=
by sorry

end solve_inequality_l370_37021


namespace divisibility_by_9_l370_37053

theorem divisibility_by_9 (x y z : ℕ) (h1 : 9 ≤ x ∧ x ≤ 9) (h2 : 0 ≤ y ∧ y ≤ 9) (h3 : 0 ≤ z ∧ z ≤ 9) :
  (100 * x + 10 * y + z) % 9 = 0 ↔ (x + y + z) % 9 = 0 := by
  sorry

end divisibility_by_9_l370_37053


namespace sum_of_first_17_terms_arithmetic_sequence_l370_37044

-- Define what it means for a sequence to be arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n / 2 * (a 1 + a n)

theorem sum_of_first_17_terms_arithmetic_sequence
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_cond : a 3 + a 9 + a 15 = 9) :
  sum_of_first_n_terms a 17 = 51 :=
sorry

end sum_of_first_17_terms_arithmetic_sequence_l370_37044


namespace fraction_of_capital_subscribed_l370_37024

theorem fraction_of_capital_subscribed (T : ℝ) (x : ℝ) :
  let B_capital := (1 / 4) * T
  let C_capital := (1 / 5) * T
  let Total_profit := 2445
  let A_profit := 815
  A_profit / Total_profit = x → x = 1 / 3 :=
by
  sorry

end fraction_of_capital_subscribed_l370_37024


namespace inequality_abc_l370_37084

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (b * c / a) + (a * c / b) + (a * b / c) ≥ a + b + c := 
  sorry

end inequality_abc_l370_37084


namespace monomial_degree_and_coefficient_l370_37035

theorem monomial_degree_and_coefficient (a b : ℤ) (h1 : -a = 7) (h2 : 1 + b = 4) : a + b = -4 :=
by
  sorry

end monomial_degree_and_coefficient_l370_37035


namespace car_race_probability_l370_37097

theorem car_race_probability :
  let pX := 1/8
  let pY := 1/12
  let pZ := 1/6
  pX + pY + pZ = 3/8 :=
by
  sorry

end car_race_probability_l370_37097


namespace number_of_5_letter_words_with_at_least_one_vowel_l370_37008

theorem number_of_5_letter_words_with_at_least_one_vowel :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F', 'G']
  let vowels := ['A', 'E']
  ∃ n : ℕ, n = 7^5 - 5^5 ∧ n = 13682 :=
by
  sorry

end number_of_5_letter_words_with_at_least_one_vowel_l370_37008


namespace cost_of_camel_l370_37033

variables (C H O E G Z L : ℕ)

theorem cost_of_camel :
  (10 * C = 24 * H) →
  (16 * H = 4 * O) →
  (6 * O = 4 * E) →
  (3 * E = 5 * G) →
  (8 * G = 12 * Z) →
  (20 * Z = 7 * L) →
  (10 * E = 120000) →
  C = 4800 :=
by
  sorry

end cost_of_camel_l370_37033


namespace a1_geq_2_pow_k_l370_37061

-- Definitions of the problem conditions in Lean 4
def conditions (a : ℕ → ℕ) (n k : ℕ) : Prop :=
  (∀ i, 1 ≤ i ∧ i ≤ n → a i < 2 * n) ∧
  (∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n ∧ i ≠ j → ¬(a i ∣ a j)) ∧
  (3^k < 2 * n ∧ 2 * n < 3^(k+1))

-- The main theorem to be proven
theorem a1_geq_2_pow_k (a : ℕ → ℕ) (n k : ℕ) (h : conditions a n k) : 
  a 1 ≥ 2^k :=
sorry

end a1_geq_2_pow_k_l370_37061


namespace book_price_percentage_change_l370_37089

theorem book_price_percentage_change (P : ℝ) (x : ℝ) (h : P * (1 - (x / 100) ^ 2) = 0.90 * P) : x = 32 := by
sorry

end book_price_percentage_change_l370_37089


namespace employee_payments_l370_37016

noncomputable def amount_paid_to_Y : ℝ := 934 / 3
noncomputable def amount_paid_to_X : ℝ := 1.20 * amount_paid_to_Y
noncomputable def amount_paid_to_Z : ℝ := 0.80 * amount_paid_to_Y

theorem employee_payments :
  amount_paid_to_X + amount_paid_to_Y + amount_paid_to_Z = 934 :=
by
  sorry

end employee_payments_l370_37016


namespace complement_intersect_l370_37039

def U : Set ℤ := {-3, -2, -1, 0, 1, 2, 3}
def A : Set ℤ := {x | x^2 - 1 ≤ 0}
def B : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
def C : Set ℤ := {x | x ∉ A ∧ x ∈ U} -- complement of A in U

theorem complement_intersect (U A B : Set ℤ) :
  (C ∩ B) = {2, 3} :=
by
  sorry

end complement_intersect_l370_37039


namespace fish_upstream_speed_l370_37070

def Vs : ℝ := 45
def Vdownstream : ℝ := 55

def Vupstream (Vs Vw : ℝ) : ℝ := Vs - Vw
def Vstream (Vs Vdownstream : ℝ) : ℝ := Vdownstream - Vs

theorem fish_upstream_speed :
  Vupstream Vs (Vstream Vs Vdownstream) = 35 := by
  sorry

end fish_upstream_speed_l370_37070


namespace euler_polyhedron_problem_l370_37043

theorem euler_polyhedron_problem : 
  ( ∀ (V E F T S : ℕ), F = 42 → (T = 2 ∧ S = 3) → V - E + F = 2 → 100 * S + 10 * T + V = 337 ) := 
by sorry

end euler_polyhedron_problem_l370_37043


namespace women_doubles_tournament_handshakes_l370_37059

theorem women_doubles_tournament_handshakes :
  ∀ (teams : List (List Prop)), List.length teams = 4 → (∀ t ∈ teams, List.length t = 2) →
  (∃ (handshakes : ℕ), handshakes = 24) :=
by
  intro teams h1 h2
  -- Assume teams are disjoint and participants shake hands meeting problem conditions
  -- The lean proof will follow the logical structure used for the mathematical solution
  -- We'll now formalize the conditions and the handshake calculation
  sorry

end women_doubles_tournament_handshakes_l370_37059


namespace smallest_value_of_a_squared_plus_b_l370_37095

theorem smallest_value_of_a_squared_plus_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
    (h3 : ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → a * x^3 + b * y^2 ≥ x * y - 1) :
    a^2 + b = 2 / (3 * Real.sqrt 3) :=
by
  sorry

end smallest_value_of_a_squared_plus_b_l370_37095


namespace find_smaller_number_l370_37076

theorem find_smaller_number (a b : ℤ) (h₁ : a + b = 8) (h₂ : a - b = 4) : b = 2 :=
by
  sorry

end find_smaller_number_l370_37076
