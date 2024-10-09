import Mathlib

namespace insurance_percentage_l414_41454

noncomputable def total_pills_per_year : ℕ := 2 * 365

noncomputable def cost_per_pill : ℕ := 5

noncomputable def total_medication_cost_per_year : ℕ := total_pills_per_year * cost_per_pill

noncomputable def doctor_visits_per_year : ℕ := 2

noncomputable def cost_per_doctor_visit : ℕ := 400

noncomputable def total_doctor_cost_per_year : ℕ := doctor_visits_per_year * cost_per_doctor_visit

noncomputable def total_yearly_cost_without_insurance : ℕ := total_medication_cost_per_year + total_doctor_cost_per_year

noncomputable def total_payment_per_year : ℕ := 1530

noncomputable def insurance_coverage_per_year : ℕ := total_yearly_cost_without_insurance - total_payment_per_year

theorem insurance_percentage:
  (insurance_coverage_per_year * 100) / total_medication_cost_per_year = 80 :=
by sorry

end insurance_percentage_l414_41454


namespace probability_even_first_odd_second_l414_41490

-- Definitions based on the conditions
def die_sides : Finset ℕ := {1, 2, 3, 4, 5, 6}
def even_numbers : Finset ℕ := {2, 4, 6}
def odd_numbers : Finset ℕ := {1, 3, 5}

-- Probability calculations
def prob_even := (even_numbers.card : ℚ) / (die_sides.card : ℚ)
def prob_odd := (odd_numbers.card : ℚ) / (die_sides.card : ℚ)

-- Proof statement
theorem probability_even_first_odd_second :
  prob_even * prob_odd = 1 / 4 :=
by
  sorry

end probability_even_first_odd_second_l414_41490


namespace find_quadruples_l414_41411

theorem find_quadruples (x y z n : ℕ) : 
  x^2 + y^2 + z^2 + 1 = 2^n → 
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ n = 0) ∨ 
  (x = 1 ∧ y = 0 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 1 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 0 ∧ z = 1 ∧ n = 1) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 1 ∧ n = 2) :=
by
  sorry

end find_quadruples_l414_41411


namespace num_parallelogram_even_l414_41470

-- Define the conditions of the problem in Lean
def isosceles_right_triangle (base_length : ℕ) := 
  base_length = 2

def square (side_length : ℕ) := 
  side_length = 1

def parallelogram (sides_length : ℕ) (diagonals_length : ℕ) := 
  sides_length = 1 ∧ diagonals_length = 1

-- Main statement to prove
theorem num_parallelogram_even (num_triangles num_squares num_parallelograms : ℕ)
  (Htriangle : ∀ t, t < num_triangles → isosceles_right_triangle 2)
  (Hsquare : ∀ s, s < num_squares → square 1)
  (Hparallelogram : ∀ p, p < num_parallelograms → parallelogram 1 1) :
  num_parallelograms % 2 = 0 := 
sorry

end num_parallelogram_even_l414_41470


namespace perimeter_of_square_l414_41469

-- Definitions based on problem conditions
def is_square_divided_into_four_congruent_rectangles (s : ℝ) (rect_perimeter : ℝ) : Prop :=
  rect_perimeter = 30 ∧ s > 0

-- Statement of the theorem to be proved
theorem perimeter_of_square (s : ℝ) (rect_perimeter : ℝ) (h : is_square_divided_into_four_congruent_rectangles s rect_perimeter) :
  4 * s = 48 :=
by sorry

end perimeter_of_square_l414_41469


namespace total_interest_is_350_l414_41417

-- Define the principal amounts, rates, and time
def principal1 : ℝ := 1000
def rate1 : ℝ := 0.03
def principal2 : ℝ := 1200
def rate2 : ℝ := 0.05
def time : ℝ := 3.888888888888889

-- Calculate the interest for one year for each loan
def interest_per_year1 : ℝ := principal1 * rate1
def interest_per_year2 : ℝ := principal2 * rate2

-- Calculate the total interest for the time period for each loan
def total_interest1 : ℝ := interest_per_year1 * time
def total_interest2 : ℝ := interest_per_year2 * time

-- Finally, calculate the total interest amount
def total_interest_amount : ℝ := total_interest1 + total_interest2

-- The proof problem: Prove that total_interest_amount == 350 Rs
theorem total_interest_is_350 : total_interest_amount = 350 := by
  sorry

end total_interest_is_350_l414_41417


namespace range_of_a_l414_41461

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬((a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0)) → (-2 < a ∧ a ≤ 6/5) :=
by
  sorry

end range_of_a_l414_41461


namespace jogging_track_circumference_l414_41432

/-- 
Given:
- Deepak's speed = 20 km/hr
- His wife's speed = 12 km/hr
- They meet for the first time in 32 minutes

Then:
The circumference of the jogging track is 17.0667 km.
-/
theorem jogging_track_circumference (deepak_speed : ℝ) (wife_speed : ℝ) (meet_time : ℝ)
  (h1 : deepak_speed = 20)
  (h2 : wife_speed = 12)
  (h3 : meet_time = (32 / 60) ) : 
  ∃ circumference : ℝ, circumference = 17.0667 :=
by
  sorry

end jogging_track_circumference_l414_41432


namespace probability_correct_arrangement_l414_41442

-- Definitions for conditions
def characters := {c : String | c = "医" ∨ c = "国"}

def valid_arrangements : Finset (List String) := 
    {["医", "医", "国"], ["医", "国", "医"], ["国", "医", "医"]}

def correct_arrangements : Finset (List String) := 
    {["医", "医", "国"], ["医", "国", "医"]}

-- Theorem statement
theorem probability_correct_arrangement :
  (correct_arrangements.card : ℚ) / (valid_arrangements.card : ℚ) = 1 / 3 :=
by
  sorry

end probability_correct_arrangement_l414_41442


namespace triangle_area_correct_l414_41420

def vector_2d (x y : ℝ) : ℝ × ℝ := (x, y)

def area_of_triangle (a b : ℝ × ℝ) : ℝ :=
  0.5 * abs (a.1 * b.2 - a.2 * b.1)

def a : ℝ × ℝ := vector_2d 3 2
def b : ℝ × ℝ := vector_2d 1 5

theorem triangle_area_correct : area_of_triangle a b = 6.5 :=
by
  sorry

end triangle_area_correct_l414_41420


namespace find_hyperbola_m_l414_41485

theorem find_hyperbola_m (m : ℝ) (h : m > 0) : 
  (∀ x y : ℝ, (x^2 / m - y^2 / 3 = 1 → y = 1 / 2 * x)) → m = 12 :=
by
  intros
  sorry

end find_hyperbola_m_l414_41485


namespace atomic_weight_Oxygen_l414_41494

theorem atomic_weight_Oxygen :
  ∀ (Ba_atomic_weight S_atomic_weight : ℝ),
    (Ba_atomic_weight = 137.33) →
    (S_atomic_weight = 32.07) →
    (Ba_atomic_weight + S_atomic_weight + 4 * 15.9 = 233) →
    15.9 = 233 - 137.33 - 32.07 / 4 := 
by
  intros Ba_atomic_weight S_atomic_weight hBa hS hm
  sorry

end atomic_weight_Oxygen_l414_41494


namespace find_DG_l414_41433

theorem find_DG (a b S k l DG BC : ℕ) (h1: S = 17 * (a + b)) (h2: S % a = 0) (h3: S % b = 0) (h4: a = S / k) (h5: b = S / l) (h6: BC = 17) (h7: (k - 17) * (l - 17) = 289) : DG = 306 :=
by
  sorry

end find_DG_l414_41433


namespace sqrt_fraction_simplification_l414_41498

theorem sqrt_fraction_simplification :
  (Real.sqrt ((25 / 49) - (16 / 81)) = (Real.sqrt 1241) / 63) := by
  sorry

end sqrt_fraction_simplification_l414_41498


namespace tangent_line_at_point_l414_41467

noncomputable def tangent_line_equation (x y : ℝ) : Prop :=
x + 4 * y - 3 = 0

theorem tangent_line_at_point (x y : ℝ) (h₁ : y = 1 / x^2) (h₂ : x = 2) (h₃ : y = 1/4) :
  tangent_line_equation x y :=
by 
  sorry

end tangent_line_at_point_l414_41467


namespace sum_repeating_decimals_as_fraction_l414_41475

-- Definitions for repeating decimals
def rep2 : ℝ := 0.2222
def rep02 : ℝ := 0.0202
def rep0002 : ℝ := 0.00020002

-- Prove the sum of the repeating decimals is equal to the given fraction
theorem sum_repeating_decimals_as_fraction :
  rep2 + rep02 + rep0002 = (2224 / 9999 : ℝ) :=
sorry

end sum_repeating_decimals_as_fraction_l414_41475


namespace initial_average_marks_is_90_l414_41474

def incorrect_average_marks (A : ℝ) : Prop :=
  let wrong_sum := 10 * A
  let correct_sum := 10 * 95
  wrong_sum + 50 = correct_sum

theorem initial_average_marks_is_90 : ∃ A : ℝ, incorrect_average_marks A ∧ A = 90 :=
by
  use 90
  unfold incorrect_average_marks
  simp
  sorry

end initial_average_marks_is_90_l414_41474


namespace evaluate_expression_l414_41487

theorem evaluate_expression : 
  (4 * 6 / (12 * 16)) * (8 * 12 * 16 / (4 * 6 * 8)) = 1 :=
by
  sorry

end evaluate_expression_l414_41487


namespace negation_of_forall_exp_gt_zero_l414_41426

open Real

theorem negation_of_forall_exp_gt_zero : 
  (¬ (∀ x : ℝ, exp x > 0)) ↔ (∃ x : ℝ, exp x ≤ 0) :=
by
  sorry

end negation_of_forall_exp_gt_zero_l414_41426


namespace remaining_value_subtract_70_percent_from_4500_l414_41468

theorem remaining_value_subtract_70_percent_from_4500 (num : ℝ) 
  (h : 0.36 * num = 2376) : 4500 - 0.70 * num = -120 :=
by
  sorry

end remaining_value_subtract_70_percent_from_4500_l414_41468


namespace pencil_case_solution_part1_pencil_case_solution_part2_1_pencil_case_solution_part2_2_l414_41452

section pencil_case_problem

variables (x m : ℕ)

-- Part 1: The cost prices of each $A$ type and $B$ type pencil cases.
def cost_price_A (x : ℕ) : Prop := 
  (800 : ℝ) / x = (1000 : ℝ) / (x + 2)

-- Part 2.1: Maximum quantity of $B$ type pencil cases.
def max_quantity_B (m : ℕ) : Prop := 
  3 * m - 50 + m ≤ 910

-- Part 2.2: Number of different scenarios for purchasing the pencil cases.
def profit_condition (m : ℕ) : Prop := 
  4 * (3 * m - 50) + 5 * m > 3795

theorem pencil_case_solution_part1 (hA : cost_price_A x) : 
  x = 8 := 
sorry

theorem pencil_case_solution_part2_1 (hB : max_quantity_B m) : 
  m ≤ 240 := 
sorry

theorem pencil_case_solution_part2_2 (hB : max_quantity_B m) (hp : profit_condition m) : 
  236 ≤ m ∧ m ≤ 240 := 
sorry

end pencil_case_problem

end pencil_case_solution_part1_pencil_case_solution_part2_1_pencil_case_solution_part2_2_l414_41452


namespace yellow_area_is_1_5625_percent_l414_41418

def square_flag_area (s : ℝ) : ℝ := s ^ 2

def cross_yellow_occupies_25_percent (s : ℝ) (w : ℝ) : Prop :=
  4 * w * s - 4 * w ^ 2 = 0.25 * s ^ 2

def yellow_area (s w : ℝ) : ℝ := 4 * w ^ 2

def percent_of_flag_area_is_yellow (s w : ℝ) : Prop :=
  yellow_area s w = 0.015625 * s ^ 2

theorem yellow_area_is_1_5625_percent (s w : ℝ) (h1: cross_yellow_occupies_25_percent s w) : 
  percent_of_flag_area_is_yellow s w :=
by sorry

end yellow_area_is_1_5625_percent_l414_41418


namespace range_of_b_l414_41422

theorem range_of_b :
  (∀ b, (∀ x : ℝ, x ≥ 1 → Real.log (2^x - b) ≥ 0) → b ≤ 1) :=
sorry

end range_of_b_l414_41422


namespace cos_alpha_eq_l414_41400

open Real

-- Define the angles and their conditions
variables (α β : ℝ)

-- Hypothesis and initial conditions
axiom ha1 : 0 < α ∧ α < π
axiom ha2 : 0 < β ∧ β < π
axiom h_cos_beta : cos β = -5 / 13
axiom h_sin_alpha_plus_beta : sin (α + β) = 3 / 5

-- The main theorem to prove
theorem cos_alpha_eq : cos α = 56 / 65 := sorry

end cos_alpha_eq_l414_41400


namespace roots_of_quadratic_l414_41409

theorem roots_of_quadratic (x : ℝ) (h : x^2 = x) : x = 0 ∨ x = 1 :=
by
  sorry

end roots_of_quadratic_l414_41409


namespace cos_C_value_triangle_perimeter_l414_41408

variables (A B C a b c : ℝ)
variables (cos_B : ℝ) (A_eq_2B : A = 2 * B) (cos_B_val : cos_B = 2 / 3)
variables (dot_product_88 : a * b * (Real.cos C) = 88)

theorem cos_C_value (A B : ℝ) (a b : ℝ) (cos_B : ℝ) (cos_C : ℝ) (dot_product_88 : a * b * cos_C = 88) :
  A = 2 * B →
  cos_B = 2 / 3 →
  cos_C = 22 / 27 :=
sorry

theorem triangle_perimeter (A B C a b c : ℝ) (cos_B : ℝ)
  (A_eq_2B : A = 2 * B) (cos_B_val : cos_B = 2 / 3) (dot_product_88 : a * b * (Real.cos C) = 88)
  (a_val : a = 12) (b_val : b = 9) (c_val : c = 7) :
  a + b + c = 28 :=
sorry

end cos_C_value_triangle_perimeter_l414_41408


namespace total_selling_price_correct_l414_41460

-- Define the conditions
def metres_of_cloth : ℕ := 500
def loss_per_metre : ℕ := 5
def cost_price_per_metre : ℕ := 41
def selling_price_per_metre : ℕ := cost_price_per_metre - loss_per_metre
def expected_total_selling_price : ℕ := 18000

-- Define the theorem
theorem total_selling_price_correct : 
  selling_price_per_metre * metres_of_cloth = expected_total_selling_price := 
by
  sorry

end total_selling_price_correct_l414_41460


namespace percentage_increase_l414_41499

variable (m y : ℝ)

theorem percentage_increase (h : x = y + (m / 100) * y) : x = ((100 + m) / 100) * y := by
  sorry

end percentage_increase_l414_41499


namespace symmetric_point_l414_41451

theorem symmetric_point (P Q : ℝ × ℝ)
  (l : ℝ → ℝ)
  (P_coords : P = (-1, 2))
  (l_eq : ∀ x, l x = x - 1) :
  Q = (3, -2) :=
by
  sorry

end symmetric_point_l414_41451


namespace symmetric_function_exists_l414_41448

-- Define the main sets A and B with given cardinalities
def A := { n : ℕ // n < 2011^2 }
def B := { n : ℕ // n < 2010 }

-- The main theorem to prove
theorem symmetric_function_exists :
  ∃ (f : A × A → B), 
  (∀ x y, f (x, y) = f (y, x)) ∧ 
  (∀ g : A → B, ∃ (a1 a2 : A), g a1 = f (a1, a2) ∧ g a2 = f (a1, a2) ∧ a1 ≠ a2) :=
sorry

end symmetric_function_exists_l414_41448


namespace solution_set_of_inequality_l414_41413

theorem solution_set_of_inequality :
  {x : ℝ | 3 * x ^ 2 - 7 * x - 10 ≥ 0} = {x : ℝ | x ≥ (10 / 3) ∨ x ≤ -1} :=
sorry

end solution_set_of_inequality_l414_41413


namespace gallons_added_in_fourth_hour_l414_41401

-- Defining the conditions
def initial_volume : ℕ := 40
def loss_rate_per_hour : ℕ := 2
def add_in_third_hour : ℕ := 1
def remaining_after_fourth_hour : ℕ := 36

-- Prove the problem statement
theorem gallons_added_in_fourth_hour :
  ∃ (x : ℕ), initial_volume - 2 * 4 + 1 - loss_rate_per_hour + x = remaining_after_fourth_hour :=
sorry

end gallons_added_in_fourth_hour_l414_41401


namespace sqrt_fraction_sum_l414_41414

theorem sqrt_fraction_sum : 
    Real.sqrt ((1 / 25) + (1 / 36)) = (Real.sqrt 61) / 30 := 
by
  sorry

end sqrt_fraction_sum_l414_41414


namespace intersection_of_A_and_B_l414_41457

open Set

def A : Set ℤ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℤ := {-1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_A_and_B_l414_41457


namespace determine_abc_l414_41479

theorem determine_abc (a b c : ℕ) (h1 : a * b * c = 2^4 * 3^2 * 5^3) 
  (h2 : gcd a b = 15) (h3 : gcd a c = 5) (h4 : gcd b c = 20) : 
  a = 15 ∧ b = 60 ∧ c = 20 :=
by
  sorry

end determine_abc_l414_41479


namespace find_f_2012_l414_41484

variable (f : ℕ → ℝ)

axiom f_one : f 1 = 3997
axiom recurrence : ∀ x, f x - f (x + 1) = 1

theorem find_f_2012 : f 2012 = 1986 :=
by
  -- Skipping proof
  sorry

end find_f_2012_l414_41484


namespace positive_integer_pairs_l414_41406

theorem positive_integer_pairs (m n : ℕ) (p : ℕ) (hp_prime : Prime p) (h_diff : m - n = p) (h_square : ∃ k : ℕ, m * n = k^2) :
  ∃ p' : ℕ, (Prime p') ∧ m = (p' + 1) / 2 ^ 2 ∧ n = (p' - 1) / 2 ^ 2 :=
sorry

end positive_integer_pairs_l414_41406


namespace third_angle_of_triangle_l414_41440

theorem third_angle_of_triangle (a b : ℝ) (h₁ : a = 25) (h₂ : b = 70) : 180 - a - b = 85 := 
by
  sorry

end third_angle_of_triangle_l414_41440


namespace automobile_travel_distance_l414_41492

variable (a r : ℝ)

theorem automobile_travel_distance (h : r ≠ 0) :
  (a / 4) * (240 / 1) * (1 / (3 * r)) = (20 * a) / r := 
by
  sorry

end automobile_travel_distance_l414_41492


namespace johnny_guitar_practice_l414_41493

theorem johnny_guitar_practice :
  ∃ x : ℕ, (∃ d : ℕ, d = 20 ∧ ∀ n : ℕ, (n = x - d ∧ n = x / 2)) ∧ (x + 80 = 3 * x) :=
by
  sorry

end johnny_guitar_practice_l414_41493


namespace compare_fractions_l414_41482

theorem compare_fractions (x y : ℝ) (n : ℕ) (h1 : 0 < x) (h2 : x < 1) (h3 : 0 < y) (h4 : y < 1) (h5 : 0 < n) :
  (x^n / (1 - x^2) + y^n / (1 - y^2)) ≥ ((x^n + y^n) / (1 - x * y)) :=
by sorry

end compare_fractions_l414_41482


namespace sum_of_values_of_n_l414_41445

theorem sum_of_values_of_n (n₁ n₂ : ℚ) (h1 : 3 * n₁ - 8 = 5) (h2 : 3 * n₂ - 8 = -5) : n₁ + n₂ = 16 / 3 := 
by {
  -- Use the provided conditions to solve the problem
  sorry 
}

end sum_of_values_of_n_l414_41445


namespace ratio_one_six_to_five_eighths_l414_41437

theorem ratio_one_six_to_five_eighths : (1 / 6) / (5 / 8) = 4 / 15 := by
  sorry

end ratio_one_six_to_five_eighths_l414_41437


namespace maximum_omega_l414_41407

noncomputable def f (omega varphi : ℝ) (x : ℝ) : ℝ :=
  Real.cos (omega * x + varphi)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = -f (-x)

def is_monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a < x → x < y → y < b → f y ≤ f x

theorem maximum_omega (omega varphi : ℝ)
    (h0 : omega > 0)
    (h1 : 0 < varphi ∧ varphi < π)
    (h2 : is_odd_function (f omega varphi))
    (h3 : is_monotonically_decreasing (f omega varphi) (-π/3) (π/6)) :
  omega ≤ 3/2 :=
sorry

end maximum_omega_l414_41407


namespace side_length_square_l414_41425

theorem side_length_square (s : ℝ) (h1 : ∃ (s : ℝ), (s > 0)) (h2 : 6 * s^2 = 3456) : s = 24 :=
sorry

end side_length_square_l414_41425


namespace arithmetic_first_term_l414_41489

theorem arithmetic_first_term (a : ℕ) (d : ℕ) (T : ℕ → ℕ) (k : ℕ) :
  (∀ n : ℕ, T n = n * (2 * a + (n - 1) * d) / 2) →
  (∀ n : ℕ, T (4 * n) / T n = k) →
  d = 5 →
  k = 16 →
  a = 3 := 
by
  sorry

end arithmetic_first_term_l414_41489


namespace ordered_pairs_satisfy_conditions_l414_41486

theorem ordered_pairs_satisfy_conditions :
  ∀ (a b : ℕ), 0 < a → 0 < b → (a^2 + b^2 + 25 = 15 * a * b) → Nat.Prime (a^2 + a * b + b^2) →
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) :=
by
  intros a b ha hb h1 h2
  sorry

end ordered_pairs_satisfy_conditions_l414_41486


namespace eight_in_C_l414_41446

def C : Set ℕ := {x | 1 ≤ x ∧ x ≤ 10}

theorem eight_in_C : 8 ∈ C :=
by {
  sorry
}

end eight_in_C_l414_41446


namespace initial_water_percentage_l414_41476

variable (W : ℝ) -- Initial percentage of water in the milk

theorem initial_water_percentage 
  (final_water_content : ℝ := 2) 
  (pure_milk_added : ℝ := 15) 
  (initial_milk_volume : ℝ := 10)
  (final_mixture_volume : ℝ := initial_milk_volume + pure_milk_added)
  (water_equation : W / 100 * initial_milk_volume = final_water_content / 100 * final_mixture_volume) 
  : W = 5 :=
by
  sorry

end initial_water_percentage_l414_41476


namespace right_triangle_side_length_l414_41423

theorem right_triangle_side_length (c a b : ℕ) (hc : c = 13) (ha : a = 12) (hypotenuse_eq : c ^ 2 = a ^ 2 + b ^ 2) : b = 5 :=
sorry

end right_triangle_side_length_l414_41423


namespace remainder_sum_mod_53_l414_41421

theorem remainder_sum_mod_53 (a b c d : ℕ)
  (h1 : a % 53 = 31)
  (h2 : b % 53 = 45)
  (h3 : c % 53 = 17)
  (h4 : d % 53 = 6) :
  (a + b + c + d) % 53 = 46 := 
sorry

end remainder_sum_mod_53_l414_41421


namespace num_signs_in_sign_language_l414_41419

theorem num_signs_in_sign_language (n : ℕ) (h : n^2 - (n - 2)^2 = 888) : n = 223 := 
sorry

end num_signs_in_sign_language_l414_41419


namespace blueberries_per_basket_l414_41488

-- Definitions based on the conditions
def total_blueberries : ℕ := 200
def total_baskets : ℕ := 10

-- Statement to be proven
theorem blueberries_per_basket : total_blueberries / total_baskets = 20 := 
by
  sorry

end blueberries_per_basket_l414_41488


namespace extreme_value_f_max_b_a_plus_1_l414_41478

noncomputable def f (x : ℝ) := Real.exp x - x + (1/2)*x^2

noncomputable def g (x : ℝ) (a b : ℝ) := (1/2)*x^2 + a*x + b

theorem extreme_value_f :
  ∃ x, deriv f x = 0 ∧ f x = 3 / 2 :=
sorry

theorem max_b_a_plus_1 (a : ℝ) (b : ℝ) :
  (∀ x, f x ≥ g x a b) → b * (a+1) ≤ (a+1)^2 - (a+1)^2 * Real.log (a+1) :=
sorry

end extreme_value_f_max_b_a_plus_1_l414_41478


namespace problem_statement_l414_41416

noncomputable def given_expression (x y z : ℝ) : ℝ :=
  (45 + (23 / 89) * Real.sin x) * (4 * y^2 - 7 * z^3)

theorem problem_statement : given_expression (Real.pi / 6) 3 (-2) = 4186 := by
  sorry

end problem_statement_l414_41416


namespace sales_tax_difference_l414_41455

theorem sales_tax_difference:
  let original_price := 50 
  let discount_rate := 0.10 
  let sales_tax_rate_1 := 0.08
  let sales_tax_rate_2 := 0.075 
  let discounted_price := original_price * (1 - discount_rate) 
  let sales_tax_1 := discounted_price * sales_tax_rate_1 
  let sales_tax_2 := discounted_price * sales_tax_rate_2 
  sales_tax_1 - sales_tax_2 = 0.225 := by
  sorry

end sales_tax_difference_l414_41455


namespace probability_of_at_least_ten_heads_in_twelve_given_first_two_heads_l414_41438

-- Define a fair coin
inductive Coin
| Heads
| Tails

def fair_coin : List Coin := [Coin.Heads, Coin.Tails]

-- Define a function to calculate the binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  Nat.descFactorial n k / k.factorial

-- Define a function to calculate the probability of at least 8 heads in 10 flips
def prob_at_least_eight_heads_in_ten : ℚ :=
  (binomial 10 8 + binomial 10 9 + binomial 10 10) / (2 ^ 10)

-- Define our theorem statement
theorem probability_of_at_least_ten_heads_in_twelve_given_first_two_heads :
    (prob_at_least_eight_heads_in_ten = 7 / 128) :=
  by
    -- The proof steps can be written here later
    sorry

end probability_of_at_least_ten_heads_in_twelve_given_first_two_heads_l414_41438


namespace ralph_socks_problem_l414_41415

theorem ralph_socks_problem :
  ∃ x y z : ℕ, x + y + z = 10 ∧ x + 2 * y + 4 * z = 30 ∧ 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z ∧ x = 2 :=
by
  sorry

end ralph_socks_problem_l414_41415


namespace rita_needs_9_months_l414_41466

def total_required_hours : ℕ := 4000
def backstroke_hours : ℕ := 100
def breaststroke_hours : ℕ := 40
def butterfly_hours : ℕ := 320
def monthly_practice_hours : ℕ := 400

def hours_already_completed : ℕ := backstroke_hours + breaststroke_hours + butterfly_hours
def remaining_hours : ℕ := total_required_hours - hours_already_completed
def months_needed : ℕ := (remaining_hours + monthly_practice_hours - 1) / monthly_practice_hours -- Ceiling division

theorem rita_needs_9_months :
  months_needed = 9 := by
  sorry

end rita_needs_9_months_l414_41466


namespace rides_on_roller_coaster_l414_41404

-- Definitions based on the conditions given.
def roller_coaster_cost : ℕ := 17
def total_tickets : ℕ := 255
def tickets_spent_on_other_activities : ℕ := 78

-- The proof statement.
theorem rides_on_roller_coaster : (total_tickets - tickets_spent_on_other_activities) / roller_coaster_cost = 10 :=
by 
  sorry

end rides_on_roller_coaster_l414_41404


namespace contrapositive_proposition_l414_41431

theorem contrapositive_proposition (α : ℝ) :
  (α = π / 4 → Real.tan α = 1) ↔ (Real.tan α ≠ 1 → α ≠ π / 4) :=
by
  sorry

end contrapositive_proposition_l414_41431


namespace total_weight_of_bars_l414_41412

-- Definitions for weights of each gold bar
variables (C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 : ℝ)
variables (W1 W2 W3 W4 W5 W6 W7 W8 : ℝ)

-- Definitions for the weighings
axiom weight_C1_C2 : W1 = C1 + C2
axiom weight_C1_C3 : W2 = C1 + C3
axiom weight_C2_C3 : W3 = C2 + C3
axiom weight_C4_C5 : W4 = C4 + C5
axiom weight_C6_C7 : W5 = C6 + C7
axiom weight_C8_C9 : W6 = C8 + C9
axiom weight_C10_C11 : W7 = C10 + C11
axiom weight_C12_C13 : W8 = C12 + C13

-- Prove the total weight of all gold bars
theorem total_weight_of_bars :
  (C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8 + C9 + C10 + C11 + C12 + C13)
  = (W1 + W2 + W3) / 2 + W4 + W5 + W6 + W7 + W8 :=
by sorry

end total_weight_of_bars_l414_41412


namespace number_of_square_integers_l414_41447

theorem number_of_square_integers (n : ℤ) (h1 : (0 ≤ n) ∧ (n < 30)) :
  (∃ (k : ℕ), n = 0 ∨ n = 15 ∨ n = 24 → ∃ (k : ℕ), n / (30 - n) = k^2) :=
by
  sorry

end number_of_square_integers_l414_41447


namespace solve_system_of_equations_l414_41496

theorem solve_system_of_equations (a b : ℝ) (h1 : a^2 ≠ 1) (h2 : b^2 ≠ 1) (h3 : a ≠ b) : 
  (∃ x y : ℝ, 
    (x - y) / (1 - x * y) = 2 * a / (1 + a^2) ∧ (x + y) / (1 + x * y) = 2 * b / (1 + b^2) ∧
    ((x = (a * b + 1) / (a + b) ∧ y = (a * b - 1) / (a - b)) ∨ 
     (x = (a + b) / (a * b + 1) ∧ y = (a - b) / (a * b - 1)))) :=
by
  sorry

end solve_system_of_equations_l414_41496


namespace find_point_B_l414_41449

def line_segment_parallel_to_x_axis (A B : (ℝ × ℝ)) : Prop :=
  A.snd = B.snd

def length_3 (A B : (ℝ × ℝ)) : Prop :=
  abs (A.fst - B.fst) = 3

theorem find_point_B (A B : (ℝ × ℝ))
  (h₁ : A = (3, 2))
  (h₂ : line_segment_parallel_to_x_axis A B)
  (h₃ : length_3 A B) :
  B = (0, 2) ∨ B = (6, 2) :=
sorry

end find_point_B_l414_41449


namespace remainder_equality_l414_41473

theorem remainder_equality 
  (Q Q' S S' E s s' : ℕ) 
  (Q_gt_Q' : Q > Q')
  (h1 : Q % E = S)
  (h2 : Q' % E = S')
  (h3 : (Q^2 * Q') % E = s)
  (h4 : (S^2 * S') % E = s') :
  s = s' :=
sorry

end remainder_equality_l414_41473


namespace find_window_width_on_second_wall_l414_41403

noncomputable def total_wall_area (width length height: ℝ) : ℝ :=
  4 * width * height

noncomputable def doorway_area (width height : ℝ) : ℝ :=
  width * height

noncomputable def window_area (width height : ℝ) : ℝ :=
  width * height

theorem find_window_width_on_second_wall :
  let room_width := 20
  let room_length := 20
  let room_height := 8
  let first_doorway_width := 3
  let first_doorway_height := 7
  let second_doorway_width := 5
  let second_doorway_height := 7
  let window_height := 4
  let area_to_paint := 560
  let total_area := total_wall_area room_width room_length room_height
  let first_doorway := doorway_area first_doorway_width first_doorway_height
  let second_doorway := doorway_area second_doorway_width second_doorway_height
  total_area - first_doorway - second_doorway - window_area w window_height = area_to_paint
  → w = 6 :=
by
  let room_width := 20
  let room_length := 20
  let room_height := 8
  let first_doorway_width := 3
  let first_doorway_height := 7
  let second_doorway_width := 5
  let second_doorway_height := 7
  let window_height := 4
  let area_to_paint := 560
  let total_area := total_wall_area room_width room_length room_height
  let first_doorway := doorway_area first_doorway_width first_doorway_height
  let second_doorway := doorway_area second_doorway_width second_doorway_height
  sorry

end find_window_width_on_second_wall_l414_41403


namespace area_of_shaded_region_l414_41434

theorem area_of_shaded_region : 
  let side_length := 4
  let radius := side_length / 2 
  let area_of_square := side_length * side_length 
  let area_of_one_quarter_circle := (pi * radius * radius) / 4
  let total_area_of_quarter_circles := 4 * area_of_one_quarter_circle 
  let area_of_shaded_region := area_of_square - total_area_of_quarter_circles 
  area_of_shaded_region = 16 - 4 * pi :=
by
  let side_length := 4
  let radius := side_length / 2
  let area_of_square := side_length * side_length
  let area_of_one_quarter_circle := (pi * radius * radius) / 4
  let total_area_of_quarter_circles := 4 * area_of_one_quarter_circle
  let area_of_shaded_region := area_of_square - total_area_of_quarter_circles
  sorry

end area_of_shaded_region_l414_41434


namespace monica_read_books_l414_41456

theorem monica_read_books (x : ℕ) 
    (h1 : 2 * (2 * x) + 5 = 69) : 
    x = 16 :=
by 
  sorry

end monica_read_books_l414_41456


namespace total_distance_correct_l414_41463

noncomputable def total_distance_covered (rA rB rC : ℝ) (revA revB revC : ℕ) : ℝ :=
  let pi := Real.pi
  let circumference (r : ℝ) := 2 * pi * r
  let distance (r : ℝ) (rev : ℕ) := circumference r * rev
  distance rA revA + distance rB revB + distance rC revC

theorem total_distance_correct :
  total_distance_covered 22.4 35.7 55.9 600 450 375 = 316015.4 :=
by
  sorry

end total_distance_correct_l414_41463


namespace find_a_l414_41459

theorem find_a (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_asymptote : ∀ x : ℝ, x = π/2 ∨ x = 3*π/2 ∨ x = -π/2 ∨ x = -3*π/2 → b*x = π/2 ∨ b*x = 3*π/2 ∨ b*x = -π/2 ∨ b*x = -3*π/2)
  (h_amplitude : ∀ x : ℝ, |a * (1 / Real.cos (b * x))| ≤ 3): 
  a = 3 := 
sorry

end find_a_l414_41459


namespace janous_inequality_l414_41450

theorem janous_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 :=
sorry

end janous_inequality_l414_41450


namespace find_eccentricity_l414_41472

variable (a b : ℝ) (ha : a > 0) (hb : b > 0)
variable (asymp_cond : b / a = 1 / 2)

theorem find_eccentricity : ∃ e : ℝ, e = Real.sqrt 5 / 2 :=
by
  let c := Real.sqrt ((a^2 + b^2) / 4)
  let e := c / a
  use e
  sorry

end find_eccentricity_l414_41472


namespace volume_of_rectangular_solid_l414_41405

variable {x y z : ℝ}
variable (hx : x * y = 3) (hy : x * z = 5) (hz : y * z = 15)

theorem volume_of_rectangular_solid : x * y * z = 15 :=
by sorry

end volume_of_rectangular_solid_l414_41405


namespace boys_and_girls_arrangement_l414_41430

theorem boys_and_girls_arrangement : 
  ∃ (arrangements : ℕ), arrangements = 48 :=
  sorry

end boys_and_girls_arrangement_l414_41430


namespace find_c_l414_41402

-- Definitions for the conditions
def line1 (x y : ℝ) : Prop := 4 * y + 2 * x + 6 = 0
def line2 (x y : ℝ) (c : ℝ) : Prop := 5 * y + c * x + 4 = 0
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Main theorem
theorem find_c (c : ℝ) : 
  (∀ x y : ℝ, line1 x y → y = -1/2 * x - 3/2) ∧ 
  (∀ x y : ℝ, line2 x y c → y = -c/5 * x - 4/5) ∧ 
  perpendicular (-1/2) (-c/5) → 
  c = -10 := by
  sorry

end find_c_l414_41402


namespace find_d_l414_41435

-- Definitions of the conditions
variables (r s t u d : ℤ)

-- Assume r, s, t, and u are positive integers
axiom r_pos : r > 0
axiom s_pos : s > 0
axiom t_pos : t > 0
axiom u_pos : u > 0

-- Given conditions
axiom h1 : r ^ 5 = s ^ 4
axiom h2 : t ^ 3 = u ^ 2
axiom h3 : t - r = 19
axiom h4 : d = u - s

-- Proof statement
theorem find_d : d = 757 :=
by sorry

end find_d_l414_41435


namespace inequality_property_l414_41480

theorem inequality_property (a b : ℝ) (h : a > b) : -5 * a < -5 * b := sorry

end inequality_property_l414_41480


namespace range_of_m_l414_41458

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem range_of_m:
  ∀ m : ℝ, 
  (∀ x, 0 ≤ x ∧ x ≤ m → f x ≤ -3) ∧ 
  (∃ x, 0 ≤ x ∧ x ≤ m ∧ f x = -4) → 
  1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_l414_41458


namespace solve_for_x_minus_y_l414_41453

theorem solve_for_x_minus_y (x y : ℝ) 
  (h1 : 3 * x - 5 * y = 5)
  (h2 : x / (x + y) = 5 / 7) : 
  x - y = 3 := 
by 
  -- Proof would go here
  sorry

end solve_for_x_minus_y_l414_41453


namespace evaluate_x_l414_41491

variable {R : Type*} [LinearOrderedField R]

theorem evaluate_x (m n k x : R) (hm : m ≠ 0) (hn : n ≠ 0) (h : m ≠ n) (h_eq : (x + m)^2 - (x + n)^2 = k * (m - n)^2) :
  x = ((k - 1) * (m + n) - 2 * k * n) / 2 :=
by
  sorry

end evaluate_x_l414_41491


namespace true_propositions_count_l414_41464

theorem true_propositions_count (b : ℤ) :
  (b = 3 → b^2 = 9) → 
  (∃! p : Prop, p = (b^2 ≠ 9 → b ≠ 3) ∨ p = (b ≠ 3 → b^2 ≠ 9) ∨ p = (b^2 = 9 → b = 3) ∧ (p = (b^2 ≠ 9 → b ≠ 3))) :=
sorry

end true_propositions_count_l414_41464


namespace reflection_line_sum_l414_41427

theorem reflection_line_sum (m b : ℝ) :
  (∀ (x y x' y' : ℝ), (x, y) = (2, 5) → (x', y') = (6, 1) →
  y' = m * x' + b ∧ y = m * x + b) → 
  m + b = 0 :=
sorry

end reflection_line_sum_l414_41427


namespace calculate_expression_l414_41441

theorem calculate_expression : 7 + 15 / 3 - 5 * 2 = 2 :=
by sorry

end calculate_expression_l414_41441


namespace sum_of_a_b_l414_41429

theorem sum_of_a_b (a b : ℝ) (h1 : a - b = 1) (h2 : a^2 + b^2 = 25) : a + b = 7 ∨ a + b = -7 := 
by 
  sorry

end sum_of_a_b_l414_41429


namespace positive_difference_perimeters_l414_41495

def perimeter_rectangle (length : ℕ) (width : ℕ) : ℕ :=
  2 * (length + width)

def perimeter_cross_shape : ℕ := 
  let top_and_bottom := 3 + 3 -- top and bottom edges
  let left_and_right := 3 + 3 -- left and right edges
  let internal_subtraction := 4
  top_and_bottom + left_and_right - internal_subtraction

theorem positive_difference_perimeters :
  let length := 4
  let width := 3
  perimeter_rectangle length width - perimeter_cross_shape = 6 :=
by
  let length := 4
  let width := 3
  sorry

end positive_difference_perimeters_l414_41495


namespace cos_six_arccos_two_fifths_l414_41462

noncomputable def arccos (x : ℝ) : ℝ := Real.arccos x
noncomputable def cos (x : ℝ) : ℝ := Real.cos x

theorem cos_six_arccos_two_fifths : cos (6 * arccos (2 / 5)) = 12223 / 15625 := 
by
  sorry

end cos_six_arccos_two_fifths_l414_41462


namespace solve_equation_l414_41443

theorem solve_equation (x y z : ℤ) (h : 19 * (x + y) + z = 19 * (-x + y) - 21) (hx : x = 1) : z = -59 := by
  sorry

end solve_equation_l414_41443


namespace tee_shirts_with_60_feet_of_material_l414_41483

def tee_shirts (f t : ℕ) : ℕ := t / f

theorem tee_shirts_with_60_feet_of_material :
  tee_shirts 4 60 = 15 :=
by
  sorry

end tee_shirts_with_60_feet_of_material_l414_41483


namespace value_of_expression_l414_41471

variables {A B C : ℚ}

def conditions (A B C : ℚ) : Prop := A / B = 3 / 2 ∧ B / C = 2 / 5

theorem value_of_expression (h : conditions A B C) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
sorry

end value_of_expression_l414_41471


namespace num_chairs_l414_41444

variable (C : Nat)
variable (tables_sticks : Nat := 6 * 9)
variable (stools_sticks : Nat := 4 * 2)
variable (total_sticks_needed : Nat := 34 * 5)
variable (total_sticks_chairs : Nat := 6 * C)

theorem num_chairs (h : total_sticks_chairs + tables_sticks + stools_sticks = total_sticks_needed) : C = 18 := 
by sorry

end num_chairs_l414_41444


namespace parabola_intersects_x_axis_two_points_l414_41424

theorem parabola_intersects_x_axis_two_points (m : ℝ) : 
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ mx^2 + (m-3)*x - 1 = 0 :=
by
  sorry

end parabola_intersects_x_axis_two_points_l414_41424


namespace polynomial_remainder_l414_41465

theorem polynomial_remainder (x : ℤ) :
  let poly := x^5 + 3*x^3 + 1
  let divisor := (x + 1)^2
  let remainder := 5*x + 9
  ∃ q : ℤ, poly = divisor * q + remainder := by
  sorry

end polynomial_remainder_l414_41465


namespace jesses_room_length_l414_41428

theorem jesses_room_length 
  (width : ℝ)
  (tile_area : ℝ)
  (num_tiles : ℕ)
  (total_area : ℝ := num_tiles * tile_area) 
  (room_length : ℝ := total_area / width)
  (hw : width = 12)
  (hta : tile_area = 4)
  (hnt : num_tiles = 6) :
  room_length = 2 :=
by
  -- proof omitted
  sorry

end jesses_room_length_l414_41428


namespace rosie_pies_l414_41481

def number_of_pies (apples : ℕ) : ℕ := sorry

theorem rosie_pies (h : number_of_pies 9 = 2) : number_of_pies 27 = 6 :=
by sorry

end rosie_pies_l414_41481


namespace sum_of_numbers_in_ratio_l414_41410

theorem sum_of_numbers_in_ratio (x : ℝ) (h1 : 8 * x - 3 * x = 20) : 3 * x + 8 * x = 44 :=
by
  sorry

end sum_of_numbers_in_ratio_l414_41410


namespace sum_of_fraction_components_l414_41439

def repeating_decimal_to_fraction (a b : ℕ) := a = 35 ∧ b = 99 ∧ Nat.gcd a b = 1

theorem sum_of_fraction_components :
  ∃ a b : ℕ, repeating_decimal_to_fraction a b ∧ a + b = 134 :=
by {
  sorry
}

end sum_of_fraction_components_l414_41439


namespace wastewater_volume_2013_l414_41477

variable (x_2013 x_2014 : ℝ)
variable (condition1 : x_2014 = 38000)
variable (condition2 : x_2014 = 1.6 * x_2013)

theorem wastewater_volume_2013 : x_2013 = 23750 := by
  sorry

end wastewater_volume_2013_l414_41477


namespace value_of_f_at_5_l414_41436

def f (x : ℝ) : ℝ := 4 * x + 2

theorem value_of_f_at_5 : f 5 = 22 :=
by
  sorry

end value_of_f_at_5_l414_41436


namespace edward_money_l414_41497

theorem edward_money (X : ℝ) (H1 : X - 130 - 0.25 * (X - 130) = 270) : X = 490 :=
by
  sorry

end edward_money_l414_41497
