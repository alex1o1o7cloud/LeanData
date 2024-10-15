import Mathlib

namespace NUMINAMATH_GPT_interest_calculation_l2186_218649

/-- Define the initial deposit in thousands of yuan (50,000 yuan = 5 x 10,000 yuan) -/
def principal : ℕ := 5

/-- Define the annual interest rate as a percentage in decimal form -/
def annual_interest_rate : ℝ := 0.04

/-- Define the number of years for the deposit -/
def years : ℕ := 3

/-- Calculate the total amount after 3 years using compound interest -/
def total_amount_after_3_years : ℝ :=
  principal * (1 + annual_interest_rate) ^ years

/-- Calculate the interest earned after 3 years -/
def interest_earned : ℝ :=
  total_amount_after_3_years - principal

theorem interest_calculation :
  interest_earned = 5 * (1 + 0.04) ^ 3 - 5 :=
by 
  sorry

end NUMINAMATH_GPT_interest_calculation_l2186_218649


namespace NUMINAMATH_GPT_distance_between_lines_is_sqrt2_l2186_218600

noncomputable def distance_between_parallel_lines (a b c1 c2 : ℝ) : ℝ :=
  |c1 - c2| / Real.sqrt (a^2 + b^2)

theorem distance_between_lines_is_sqrt2 :
  distance_between_parallel_lines 1 1 (-1) 1 = Real.sqrt 2 := 
by 
  sorry

end NUMINAMATH_GPT_distance_between_lines_is_sqrt2_l2186_218600


namespace NUMINAMATH_GPT_max_value_of_z_l2186_218674

open Real

theorem max_value_of_z (x y : ℝ) (h₁ : x + y ≥ 1) (h₂ : 2 * x - y ≤ 0) (h₃ : 3 * x - 2 * y + 2 ≥ 0) : 
  ∃ x y, 3 * x - y = 2 :=
sorry

end NUMINAMATH_GPT_max_value_of_z_l2186_218674


namespace NUMINAMATH_GPT_james_marbles_left_l2186_218662

def marbles_remain (total_marbles : ℕ) (bags : ℕ) (given_away : ℕ) : ℕ :=
  (total_marbles / bags) * (bags - given_away)

theorem james_marbles_left :
  marbles_remain 28 4 1 = 21 := 
by
  sorry

end NUMINAMATH_GPT_james_marbles_left_l2186_218662


namespace NUMINAMATH_GPT_smallest_positive_number_is_option_B_l2186_218647

theorem smallest_positive_number_is_option_B :
  let A := 8 - 2 * Real.sqrt 17
  let B := 2 * Real.sqrt 17 - 8
  let C := 25 - 7 * Real.sqrt 5
  let D := 40 - 9 * Real.sqrt 2
  let E := 9 * Real.sqrt 2 - 40
  0 < B ∧ (A ≤ 0 ∨ B < A) ∧ (C ≤ 0 ∨ B < C) ∧ (D ≤ 0 ∨ B < D) ∧ (E ≤ 0 ∨ B < E) :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_number_is_option_B_l2186_218647


namespace NUMINAMATH_GPT_main_theorem_l2186_218696

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem main_theorem :
  (∀ x : ℝ, f (x + 5/2) + f x = 2) ∧
  (∀ x : ℝ, f (1 + 2*x) = f (1 - 2*x)) ∧
  (∀ x : ℝ, g (x + 2) = g (x - 2)) ∧
  (∀ x : ℝ, g (-x + 1) - 1 = -g (x + 1) + 1) ∧
  (∀ x : ℝ, 0 ≤ x → x ≤ 2 → f x + g x = 3^x + x^3) →
  f 2022 * g 2022 = 72 :=
sorry

end NUMINAMATH_GPT_main_theorem_l2186_218696


namespace NUMINAMATH_GPT_prime_factors_power_l2186_218660

-- Given conditions
def a_b_c_factors (a b c : ℕ) : Prop :=
  (∀ x, x = a ∨ x = b ∨ x = c → Prime x) ∧
  a < b ∧ b < c ∧ a * b * c ∣ 1998

-- Proof problem
theorem prime_factors_power (a b c : ℕ) (h : a_b_c_factors a b c) : (b + c) ^ a = 1600 := 
sorry

end NUMINAMATH_GPT_prime_factors_power_l2186_218660


namespace NUMINAMATH_GPT_find_value_of_k_l2186_218636

theorem find_value_of_k (k x : ℝ) 
  (h : 1 / (4 - x ^ 2) + 2 = k / (x - 2)) : 
  k = -1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_k_l2186_218636


namespace NUMINAMATH_GPT_problem_statement_l2186_218664

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (Real.sin x)^2 - Real.tan x else Real.exp (-2 * x)

theorem problem_statement : f (f (-25 * Real.pi / 4)) = Real.exp (-3) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2186_218664


namespace NUMINAMATH_GPT_man_saves_percentage_of_salary_l2186_218615

variable (S : ℝ) (P : ℝ) (S_s : ℝ)

def problem_statement (S : ℝ) (S_s : ℝ) (P : ℝ) : Prop :=
  S_s = S - 1.2 * (S - (P / 100) * S)

theorem man_saves_percentage_of_salary
  (h1 : S = 6250)
  (h2 : S_s = 250) :
  problem_statement S S_s 20 :=
by
  sorry

end NUMINAMATH_GPT_man_saves_percentage_of_salary_l2186_218615


namespace NUMINAMATH_GPT_negation_of_universal_prop_l2186_218632

theorem negation_of_universal_prop :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 > 0 := 
by 
  sorry

end NUMINAMATH_GPT_negation_of_universal_prop_l2186_218632


namespace NUMINAMATH_GPT_prime_square_pairs_l2186_218692

theorem prime_square_pairs (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
    ∃ n : Nat, p^2 + 5 * p * q + 4 * q^2 = n^2 ↔ (p = 13 ∧ q = 3) ∨ (p = 7 ∧ q = 5) ∨ (p = 5 ∧ q = 11) ∨ (p = 3 ∧ q = 13) ∨ (p = 5 ∧ q = 7) ∨ (p = 11 ∧ q = 5) :=
by
  sorry

end NUMINAMATH_GPT_prime_square_pairs_l2186_218692


namespace NUMINAMATH_GPT_quadratic_two_equal_real_roots_c_l2186_218679

theorem quadratic_two_equal_real_roots_c (c : ℝ) : 
  (∃ x : ℝ, (2*x^2 - x + c = 0) ∧ (∃ y : ℝ, y ≠ x ∧ 2*y^2 - y + c = 0)) →
  c = 1/8 :=
sorry

end NUMINAMATH_GPT_quadratic_two_equal_real_roots_c_l2186_218679


namespace NUMINAMATH_GPT_sqrt_cubic_sqrt_decimal_l2186_218609

theorem sqrt_cubic_sqrt_decimal : 
  (Real.sqrt (0.0036 : ℝ))^(1/3) = 0.3912 :=
sorry

end NUMINAMATH_GPT_sqrt_cubic_sqrt_decimal_l2186_218609


namespace NUMINAMATH_GPT_matrix_self_inverse_pairs_l2186_218697

theorem matrix_self_inverse_pairs :
  ∃ p : Finset (ℝ × ℝ), (∀ a d, (a, d) ∈ p ↔ (∃ (m : Matrix (Fin 2) (Fin 2) ℝ), 
    m = !![a, 4; -9, d] ∧ m * m = 1)) ∧ p.card = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_matrix_self_inverse_pairs_l2186_218697


namespace NUMINAMATH_GPT_probability_blue_face_up_l2186_218625

-- Definitions of the conditions
def dodecahedron_faces : ℕ := 12
def blue_faces : ℕ := 10
def red_faces : ℕ := 2

-- Expected probability
def probability_blue_face : ℚ := 5 / 6

-- Theorem to prove the probability of rolling a blue face on a dodecahedron
theorem probability_blue_face_up (total_faces blue_count red_count : ℕ)
    (h1 : total_faces = dodecahedron_faces)
    (h2 : blue_count = blue_faces)
    (h3 : red_count = red_faces) :
  blue_count / total_faces = probability_blue_face :=
by sorry

end NUMINAMATH_GPT_probability_blue_face_up_l2186_218625


namespace NUMINAMATH_GPT_trisha_initial_money_l2186_218663

-- Definitions based on conditions
def spent_on_meat : ℕ := 17
def spent_on_chicken : ℕ := 22
def spent_on_veggies : ℕ := 43
def spent_on_eggs : ℕ := 5
def spent_on_dog_food : ℕ := 45
def spent_on_cat_food : ℕ := 18
def money_left : ℕ := 35

-- Total amount spent
def total_spent : ℕ :=
  spent_on_meat + spent_on_chicken + spent_on_veggies + spent_on_eggs + spent_on_dog_food + spent_on_cat_food

-- The target amount she brought with her at the beginning
def total_money_brought : ℕ :=
  total_spent + money_left

-- The theorem to be proved
theorem trisha_initial_money :
  total_money_brought = 185 :=
by
  sorry

end NUMINAMATH_GPT_trisha_initial_money_l2186_218663


namespace NUMINAMATH_GPT_larger_angle_of_nonagon_l2186_218644

theorem larger_angle_of_nonagon : 
  ∀ (n : ℕ) (x : ℝ), 
  n = 9 → 
  (∃ a b : ℕ, a + b = n ∧ a * x + b * (3 * x) = 180 * (n - 2)) → 
  3 * (180 * (n - 2) / 15) = 252 :=
by
  sorry

end NUMINAMATH_GPT_larger_angle_of_nonagon_l2186_218644


namespace NUMINAMATH_GPT_solve_for_x_l2186_218682

noncomputable def x_solution (x : ℚ) : Prop :=
  x > 1 ∧ 3 * x^2 + 11 * x - 20 = 0

theorem solve_for_x :
  ∃ x : ℚ, x_solution x ∧ x = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2186_218682


namespace NUMINAMATH_GPT_ryan_bread_slices_l2186_218651

theorem ryan_bread_slices 
  (num_pb_people : ℕ)
  (pb_sandwiches_per_person : ℕ)
  (num_tuna_people : ℕ)
  (tuna_sandwiches_per_person : ℕ)
  (num_turkey_people : ℕ)
  (turkey_sandwiches_per_person : ℕ)
  (slices_per_pb_sandwich : ℕ)
  (slices_per_tuna_sandwich : ℕ)
  (slices_per_turkey_sandwich : ℝ)
  (h1 : num_pb_people = 4)
  (h2 : pb_sandwiches_per_person = 2)
  (h3 : num_tuna_people = 3)
  (h4 : tuna_sandwiches_per_person = 3)
  (h5 : num_turkey_people = 2)
  (h6 : turkey_sandwiches_per_person = 1)
  (h7 : slices_per_pb_sandwich = 2)
  (h8 : slices_per_tuna_sandwich = 3)
  (h9 : slices_per_turkey_sandwich = 1.5) : 
  (num_pb_people * pb_sandwiches_per_person * slices_per_pb_sandwich 
  + num_tuna_people * tuna_sandwiches_per_person * slices_per_tuna_sandwich 
  + (num_turkey_people * turkey_sandwiches_per_person : ℝ) * slices_per_turkey_sandwich) = 46 :=
by
  sorry

end NUMINAMATH_GPT_ryan_bread_slices_l2186_218651


namespace NUMINAMATH_GPT_ratio_rounded_to_nearest_tenth_l2186_218653

theorem ratio_rounded_to_nearest_tenth : 
  (Float.round (11 / 16 : Float) * 10) / 10 = 0.7 :=
by
  -- sorry is used because the proof steps are not required in this task.
  sorry

end NUMINAMATH_GPT_ratio_rounded_to_nearest_tenth_l2186_218653


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2186_218612

def A : Set ℤ := {-3, -1, 2, 6}
def B : Set ℤ := {x | x > 0}

theorem intersection_of_A_and_B : A ∩ B = {2, 6} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2186_218612


namespace NUMINAMATH_GPT_smallest_multiple_14_15_16_l2186_218601

theorem smallest_multiple_14_15_16 : 
  Nat.lcm (Nat.lcm 14 15) 16 = 1680 := by
  sorry

end NUMINAMATH_GPT_smallest_multiple_14_15_16_l2186_218601


namespace NUMINAMATH_GPT_teacher_age_l2186_218690

theorem teacher_age (avg_student_age : ℕ) (num_students : ℕ) (new_avg_age : ℕ) (num_total : ℕ) (total_student_age : ℕ) (total_age_with_teacher : ℕ) :
  avg_student_age = 22 → 
  num_students = 23 → 
  new_avg_age = 23 → 
  num_total = 24 → 
  total_student_age = avg_student_age * num_students → 
  total_age_with_teacher = new_avg_age * num_total → 
  total_age_with_teacher - total_student_age = 46 :=
by
  intros
  sorry

end NUMINAMATH_GPT_teacher_age_l2186_218690


namespace NUMINAMATH_GPT_geometric_mean_of_roots_l2186_218646

theorem geometric_mean_of_roots (x : ℝ) (h : x^2 = (Real.sqrt 2 + 1) * (Real.sqrt 2 - 1)) : x = 1 ∨ x = -1 := 
by
  sorry

end NUMINAMATH_GPT_geometric_mean_of_roots_l2186_218646


namespace NUMINAMATH_GPT_framed_painting_ratio_l2186_218619

/-- A rectangular painting measuring 20" by 30" is to be framed, with the longer dimension vertical.
The width of the frame at the top and bottom is three times the width of the frame on the sides.
Given that the total area of the frame equals the area of the painting, the ratio of the smaller to the 
larger dimension of the framed painting is 4:7. -/
theorem framed_painting_ratio : 
  ∀ (w h : ℝ) (side_frame_width : ℝ), 
    w = 20 ∧ h = 30 ∧ 3 * side_frame_width * (2 * (w + 2 * side_frame_width) + 2 * (h + 6 * side_frame_width) - w * h) = w * h 
    → side_frame_width = 2 
    → (w + 2 * side_frame_width) / (h + 6 * side_frame_width) = 4 / 7 :=
sorry

end NUMINAMATH_GPT_framed_painting_ratio_l2186_218619


namespace NUMINAMATH_GPT_number_of_classmates_ate_cake_l2186_218616

theorem number_of_classmates_ate_cake (n : ℕ) 
  (Alyosha : ℝ) (Alena : ℝ) 
  (H1 : Alyosha = 1 / 11) 
  (H2 : Alena = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
sorry

end NUMINAMATH_GPT_number_of_classmates_ate_cake_l2186_218616


namespace NUMINAMATH_GPT_hannah_mugs_problem_l2186_218665

theorem hannah_mugs_problem :
  ∀ (total_mugs red_mugs yellow_mugs blue_mugs : ℕ),
    total_mugs = 40 →
    yellow_mugs = 12 →
    red_mugs * 2 = yellow_mugs →
    blue_mugs = 3 * red_mugs →
    total_mugs - (red_mugs + yellow_mugs + blue_mugs) = 4 :=
by
  intros total_mugs red_mugs yellow_mugs blue_mugs Htotal Hyellow Hred Hblue
  sorry

end NUMINAMATH_GPT_hannah_mugs_problem_l2186_218665


namespace NUMINAMATH_GPT_estimate_height_of_student_l2186_218607

theorem estimate_height_of_student
  (x_values : List ℝ)
  (y_values : List ℝ)
  (h_sum_x : x_values.sum = 225)
  (h_sum_y : y_values.sum = 1600)
  (h_length : x_values.length = 10 ∧ y_values.length = 10)
  (b : ℝ := 4) :
  ∃ a : ℝ, ∀ x : ℝ, x = 24 → (b * x + a = 166) :=
by
  have avg_x := (225 / 10 : ℝ)
  have avg_y := (1600 / 10 : ℝ)
  have a := avg_y - b * avg_x
  use a
  intro x h
  rw [h]
  sorry

end NUMINAMATH_GPT_estimate_height_of_student_l2186_218607


namespace NUMINAMATH_GPT_remainder_of_N_mod_45_l2186_218614

def concatenated_num_from_1_to_52 : ℕ := 
  -- This represents the concatenated number from 1 to 52.
  -- We define here in Lean as a placeholder 
  -- since Lean cannot concatenate numbers directly.
  12345678910111213141516171819202122232425262728293031323334353637383940414243444546474849505152

theorem remainder_of_N_mod_45 : 
  concatenated_num_from_1_to_52 % 45 = 37 := 
sorry

end NUMINAMATH_GPT_remainder_of_N_mod_45_l2186_218614


namespace NUMINAMATH_GPT_cows_with_no_spots_l2186_218606

-- Definitions of conditions
def total_cows : Nat := 140
def cows_with_red_spot : Nat := (40 * total_cows) / 100
def cows_without_red_spot : Nat := total_cows - cows_with_red_spot
def cows_with_blue_spot : Nat := (25 * cows_without_red_spot) / 100

-- Theorem statement asserting the number of cows with no spots
theorem cows_with_no_spots : (total_cows - cows_with_red_spot - cows_with_blue_spot) = 63 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_cows_with_no_spots_l2186_218606


namespace NUMINAMATH_GPT_remainder_of_polynomial_l2186_218673

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x + 4

-- Define the main theorem stating the remainder when f(x) is divided by (x - 1) is 6
theorem remainder_of_polynomial : f 1 = 6 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_of_polynomial_l2186_218673


namespace NUMINAMATH_GPT_friends_contribution_l2186_218626

theorem friends_contribution (x : ℝ) 
  (h1 : 4 * (x - 5) = 0.75 * 4 * x) : 
  0.75 * 4 * x = 60 :=
by 
  sorry

end NUMINAMATH_GPT_friends_contribution_l2186_218626


namespace NUMINAMATH_GPT_circle_problem_is_solved_l2186_218672

def circle_problem_pqr : ℕ :=
  let n := 3 / 2;
  let p := 3;
  let q := 1;
  let r := 4;
  p + q + r

theorem circle_problem_is_solved : circle_problem_pqr = 8 :=
by {
  -- Additional context of conditions can be added here if necessary
  sorry
}

end NUMINAMATH_GPT_circle_problem_is_solved_l2186_218672


namespace NUMINAMATH_GPT_evaluate_expression_at_2_l2186_218605

def f (x : ℚ) : ℚ := (4 * x^2 + 6 * x + 9) / (x^2 - 2 * x + 5)
def g (x : ℚ) : ℚ := 2 * x - 3

theorem evaluate_expression_at_2 : f (g 2) + g (f 2) = 331 / 20 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_2_l2186_218605


namespace NUMINAMATH_GPT_math_problem_l2186_218676

def Q (f : ℝ → ℝ) : Prop :=
  (∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → x + y ≠ 0 → f (1 / (x + y)) = f (1 / x) + f (1 / y))
  ∧ (∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → x + y ≠ 0 → (x + y) * f (x + y) = x * y * f x * f y)
  ∧ f 1 = 1

theorem math_problem (f : ℝ → ℝ) : Q f → (∀ (x : ℝ), x ≠ 0 → f x = 1 / x) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_math_problem_l2186_218676


namespace NUMINAMATH_GPT_sozopolian_ineq_find_p_l2186_218691

noncomputable def is_sozopolian (p a b c : ℕ) : Prop :=
  p % 2 = 1 ∧
  Nat.Prime p ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  (a * b + 1) % p = 0 ∧
  (b * c + 1) % p = 0 ∧
  (c * a + 1) % p = 0

theorem sozopolian_ineq (p a b c : ℕ) (hp : is_sozopolian p a b c) :
  p + 2 ≤ (a + b + c) / 3 :=
sorry

theorem find_p (p : ℕ) :
  (∃ a b c : ℕ, is_sozopolian p a b c ∧ (a + b + c) / 3 = p + 2) ↔ p = 5 :=
sorry

end NUMINAMATH_GPT_sozopolian_ineq_find_p_l2186_218691


namespace NUMINAMATH_GPT_sum_of_digits_l2186_218657

theorem sum_of_digits (A T M : ℕ) (h1 : T = A + 3) (h2 : M = 3)
    (h3 : (∃ k : ℕ, T = k^2 * M) ∧ (∃ l : ℕ, T = 33)) : 
    ∃ x : ℕ, ∃ dsum : ℕ, (A + x) % (M + x) = 0 ∧ dsum = 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_l2186_218657


namespace NUMINAMATH_GPT_num_type_A_cubes_internal_diagonal_l2186_218694

theorem num_type_A_cubes_internal_diagonal :
  let L := 120
  let W := 350
  let H := 400
  -- Total cubes traversed calculation
  let GCD := Nat.gcd
  let total_cubes_traversed := L + W + H - (GCD L W + GCD W H + GCD H L) + GCD L (GCD W H)
  -- Type A cubes calculation
  total_cubes_traversed / 2 = 390 := by sorry

end NUMINAMATH_GPT_num_type_A_cubes_internal_diagonal_l2186_218694


namespace NUMINAMATH_GPT_claire_photos_eq_10_l2186_218643

variable (C L R : Nat)

theorem claire_photos_eq_10
  (h1: L = 3 * C)
  (h2: R = C + 20)
  (h3: L = R)
  : C = 10 := by
  sorry

end NUMINAMATH_GPT_claire_photos_eq_10_l2186_218643


namespace NUMINAMATH_GPT_compute_pairs_a_b_l2186_218667

noncomputable def f (x a b : ℝ) : ℝ := (x + a) / (x + b)

theorem compute_pairs_a_b (a b : ℝ) (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ -b) :
  ((∀ x, f (f x a b) a b = -1 / x) ↔ (a = -1 ∧ b = 1)) :=
sorry

end NUMINAMATH_GPT_compute_pairs_a_b_l2186_218667


namespace NUMINAMATH_GPT_clothing_probability_l2186_218608

/-- I have a drawer with 6 shirts, 8 pairs of shorts, 7 pairs of socks, and 3 jackets in it.
    If I reach in and randomly remove four articles of clothing, what is the probability that 
    I get one shirt, one pair of shorts, one pair of socks, and one jacket? -/
theorem clothing_probability :
  let total_articles := 6 + 8 + 7 + 3
  let total_combinations := Nat.choose total_articles 4
  let favorable_combinations := 6 * 8 * 7 * 3
  (favorable_combinations : ℚ) / total_combinations = 144 / 1815 :=
by
  let total_articles := 6 + 8 + 7 + 3
  let total_combinations := Nat.choose total_articles 4
  let favorable_combinations := 6 * 8 * 7 * 3
  suffices (favorable_combinations : ℚ) / total_combinations = 144 / 1815
  by
    sorry
  sorry

end NUMINAMATH_GPT_clothing_probability_l2186_218608


namespace NUMINAMATH_GPT_tailor_time_calculation_l2186_218620

-- Define the basic quantities and their relationships
def time_ratio_shirt : ℕ := 1
def time_ratio_pants : ℕ := 2
def time_ratio_jacket : ℕ := 3

-- Given conditions
def shirts_made := 2
def pants_made := 3
def jackets_made := 4
def total_time_initial : ℝ := 10

-- Unknown time per shirt
noncomputable def time_per_shirt := total_time_initial / (shirts_made * time_ratio_shirt 
  + pants_made * time_ratio_pants 
  + jackets_made * time_ratio_jacket)

-- Future quantities
def future_shirts := 14
def future_pants := 10
def future_jackets := 2

-- Calculate the future total time required
noncomputable def future_time_required := (future_shirts * time_ratio_shirt 
  + future_pants * time_ratio_pants 
  + future_jackets * time_ratio_jacket) * time_per_shirt

-- State the theorem to prove
theorem tailor_time_calculation : future_time_required = 20 := by
  sorry

end NUMINAMATH_GPT_tailor_time_calculation_l2186_218620


namespace NUMINAMATH_GPT_average_marks_math_chem_l2186_218603

-- Definitions to capture the conditions
variables (M P C : ℕ)
variable (cond1 : M + P = 32)
variable (cond2 : C = P + 20)

-- The theorem to prove
theorem average_marks_math_chem (M P C : ℕ) 
  (cond1 : M + P = 32) 
  (cond2 : C = P + 20) : 
  (M + C) / 2 = 26 := 
sorry

end NUMINAMATH_GPT_average_marks_math_chem_l2186_218603


namespace NUMINAMATH_GPT_polynomial_quotient_correct_l2186_218656

noncomputable def polynomial_division_quotient : Polynomial ℝ :=
  (Polynomial.C 1 * Polynomial.X^6 + Polynomial.C 2 * Polynomial.X^2 + Polynomial.C 8) / (Polynomial.X - Polynomial.C 1)

-- Math proof statement
theorem polynomial_quotient_correct :
  polynomial_division_quotient = Polynomial.C 1 * Polynomial.X^5 + Polynomial.C 1 * Polynomial.X^4 
                                 + Polynomial.C 1 * Polynomial.X^3 + Polynomial.C 1 * Polynomial.X^2 
                                 + Polynomial.C 3 * Polynomial.X + Polynomial.C 3 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_quotient_correct_l2186_218656


namespace NUMINAMATH_GPT_max_b_lattice_free_line_l2186_218629

theorem max_b_lattice_free_line : 
  ∃ b : ℚ, (∀ (m : ℚ), (1 / 3) < m ∧ m < b → 
  ∀ x : ℤ, 0 < x ∧ x ≤ 150 → ¬ (∃ y : ℤ, y = m * x + 4)) ∧ 
  b = 50 / 147 :=
sorry

end NUMINAMATH_GPT_max_b_lattice_free_line_l2186_218629


namespace NUMINAMATH_GPT_part_a_l2186_218639

theorem part_a (b c: ℤ) : ∃ (n : ℕ) (a : ℕ → ℤ), 
  (a 0 = b) ∧ (a n = c) ∧ (∀ i, 1 ≤ i ∧ i ≤ n → |a i - a (i - 1)| = i^2) :=
sorry

end NUMINAMATH_GPT_part_a_l2186_218639


namespace NUMINAMATH_GPT_josephs_total_cards_l2186_218695

def number_of_decks : ℕ := 4
def cards_per_deck : ℕ := 52
def total_cards : ℕ := number_of_decks * cards_per_deck

theorem josephs_total_cards : total_cards = 208 := by
  sorry

end NUMINAMATH_GPT_josephs_total_cards_l2186_218695


namespace NUMINAMATH_GPT_area_of_circle_diameter_7_5_l2186_218610

theorem area_of_circle_diameter_7_5 :
  ∃ (A : ℝ), (A = 14.0625 * Real.pi) ↔ (∃ (d : ℝ), d = 7.5 ∧ A = Real.pi * (d / 2) ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_area_of_circle_diameter_7_5_l2186_218610


namespace NUMINAMATH_GPT_simplify_sqrt_25000_l2186_218633

theorem simplify_sqrt_25000 : Real.sqrt 25000 = 50 * Real.sqrt 10 := 
by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_25000_l2186_218633


namespace NUMINAMATH_GPT_arithmetic_expression_l2186_218637

theorem arithmetic_expression : 8 / 4 + 5 * 2 ^ 2 - (3 + 7) = 12 := by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_l2186_218637


namespace NUMINAMATH_GPT_smallest_number_divisible_by_20_and_36_is_180_l2186_218678

theorem smallest_number_divisible_by_20_and_36_is_180 :
  ∃ x, (x % 20 = 0) ∧ (x % 36 = 0) ∧ (∀ y, (y % 20 = 0) ∧ (y % 36 = 0) → x ≤ y) ∧ x = 180 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_divisible_by_20_and_36_is_180_l2186_218678


namespace NUMINAMATH_GPT_initial_number_of_machines_l2186_218638

theorem initial_number_of_machines
  (x : ℕ)
  (h1 : x * 270 = 1080)
  (h2 : 20 * 3600 = 144000)
  (h3 : ∀ y, (20 * y * 4 = 3600) → y = 45) :
  x = 6 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_of_machines_l2186_218638


namespace NUMINAMATH_GPT_min_value_expression_l2186_218661

theorem min_value_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 48) :
  x^2 + 6 * x * y + 9 * y^2 + 4 * z^2 ≥ 128 := 
sorry

end NUMINAMATH_GPT_min_value_expression_l2186_218661


namespace NUMINAMATH_GPT_expected_BBR_sequences_l2186_218635

theorem expected_BBR_sequences :
  let total_cards := 52
  let black_cards := 26
  let red_cards := 26
  let probability_of_next_black := (25 / 51)
  let probability_of_third_red := (26 / 50)
  let probability_of_BBR := probability_of_next_black * probability_of_third_red
  let possible_start_positions := 26
  let expected_BBR := possible_start_positions * probability_of_BBR
  expected_BBR = (338 / 51) :=
by
  sorry

end NUMINAMATH_GPT_expected_BBR_sequences_l2186_218635


namespace NUMINAMATH_GPT_identify_quadratic_equation_l2186_218602

theorem identify_quadratic_equation :
  (¬(∃ x y : ℝ, x^2 - 2*x*y + y^2 = 0) ∧  -- Condition A is not a quadratic equation
   ¬(∃ x : ℝ, x*(x + 3) = x^2 - 1) ∧      -- Condition B is not a quadratic equation
   (∃ x : ℝ, x^2 - 2*x - 3 = 0) ∧         -- Condition C is a quadratic equation
   ¬(∃ x : ℝ, x + (1/x) = 0)) →           -- Condition D is not a quadratic equation
  (true) := sorry

end NUMINAMATH_GPT_identify_quadratic_equation_l2186_218602


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l2186_218623

theorem isosceles_triangle_base_length (b : ℕ) (h₁ : 6 + 6 + b = 20) : b = 8 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l2186_218623


namespace NUMINAMATH_GPT_three_digit_numbers_count_l2186_218666

def number_of_3_digit_numbers : ℕ := 
  let without_zero := 2 * Nat.choose 9 3
  let with_zero := Nat.choose 9 2
  without_zero + with_zero

theorem three_digit_numbers_count : number_of_3_digit_numbers = 204 := by
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_three_digit_numbers_count_l2186_218666


namespace NUMINAMATH_GPT_greatest_possible_d_l2186_218634

noncomputable def point_2d_units_away_origin (d : ℝ) : Prop :=
  2 * d = Real.sqrt ((4 * Real.sqrt 3)^2 + (d + 5)^2)

theorem greatest_possible_d : 
  ∃ d : ℝ, point_2d_units_away_origin d ∧ d = (5 + Real.sqrt 244) / 3 :=
sorry

end NUMINAMATH_GPT_greatest_possible_d_l2186_218634


namespace NUMINAMATH_GPT_simplify_fraction_l2186_218686

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : (x^2 / (x - 1)) - (1 / (x - 1)) = x + 1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2186_218686


namespace NUMINAMATH_GPT_cubic_yards_to_cubic_feet_l2186_218693

theorem cubic_yards_to_cubic_feet :
  (1 : ℝ) * 3^3 * 5 = 135 := by
sorry

end NUMINAMATH_GPT_cubic_yards_to_cubic_feet_l2186_218693


namespace NUMINAMATH_GPT_volume_remaining_cube_l2186_218669

theorem volume_remaining_cube (a : ℝ) (original_volume vertex_cube_volume : ℝ) (number_of_vertices : ℕ) :
  original_volume = a^3 → 
  vertex_cube_volume = 1 → 
  number_of_vertices = 8 → 
  a = 3 →
  original_volume - (number_of_vertices * vertex_cube_volume) = 19 := 
by
  sorry

end NUMINAMATH_GPT_volume_remaining_cube_l2186_218669


namespace NUMINAMATH_GPT_b_range_given_conditions_l2186_218641

theorem b_range_given_conditions 
    (b c : ℝ)
    (roots_in_interval : ∀ x, x^2 + b * x + c = 0 → -1 ≤ x ∧ x ≤ 1)
    (ineq : 0 ≤ 3 * b + c ∧ 3 * b + c ≤ 3) :
    0 ≤ b ∧ b ≤ 2 :=
sorry

end NUMINAMATH_GPT_b_range_given_conditions_l2186_218641


namespace NUMINAMATH_GPT_system_of_equations_solution_l2186_218671

theorem system_of_equations_solution :
  ∃ x y : ℚ, x = 2 * y ∧ 2 * x - y = 5 ∧ x = 10 / 3 ∧ y = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l2186_218671


namespace NUMINAMATH_GPT_hundreds_digit_of_8_pow_2048_l2186_218684

theorem hundreds_digit_of_8_pow_2048 : 
  (8^2048 % 1000) / 100 = 0 := 
by
  sorry

end NUMINAMATH_GPT_hundreds_digit_of_8_pow_2048_l2186_218684


namespace NUMINAMATH_GPT_express_y_in_terms_of_x_l2186_218627

theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x + 3 * y = 4) : y = (4 - 2 * x) / 3 := 
by
  sorry

end NUMINAMATH_GPT_express_y_in_terms_of_x_l2186_218627


namespace NUMINAMATH_GPT_days_B_can_finish_alone_l2186_218618

theorem days_B_can_finish_alone (x : ℚ) : 
  (1 / 3 : ℚ) + (1 / x) = (1 / 2 : ℚ) → x = 6 := 
by
  sorry

end NUMINAMATH_GPT_days_B_can_finish_alone_l2186_218618


namespace NUMINAMATH_GPT_Emily_beads_l2186_218611

-- Define the conditions and question
theorem Emily_beads (n k : ℕ) (h1 : k = 4) (h2 : n = 5) : n * k = 20 := by
  -- Sorry: this is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_Emily_beads_l2186_218611


namespace NUMINAMATH_GPT_break_even_production_volume_l2186_218613

theorem break_even_production_volume :
  ∃ Q : ℝ, 300 = 100 + 100000 / Q ∧ Q = 500 :=
by
  use 500
  sorry

end NUMINAMATH_GPT_break_even_production_volume_l2186_218613


namespace NUMINAMATH_GPT_factorization_of_polynomial_l2186_218680

theorem factorization_of_polynomial (x : ℝ) : 16 * x ^ 2 - 40 * x + 25 = (4 * x - 5) ^ 2 := by 
  sorry

end NUMINAMATH_GPT_factorization_of_polynomial_l2186_218680


namespace NUMINAMATH_GPT_expression_for_f_general_formula_a_n_sum_S_n_l2186_218688

-- Definitions for conditions
def f (x : ℝ) : ℝ := x^2 + x

-- Given conditions
axiom f_zero : f 0 = 0
axiom f_recurrence : ∀ x : ℝ, f (x + 1) - f x = x + 1

-- Statements to prove
theorem expression_for_f (x : ℝ) : f x = x^2 + x := 
sorry

theorem general_formula_a_n (t : ℝ) (n : ℕ) (H : 0 < t) : 
    ∃ a_n : ℕ → ℝ, a_n n = t^n := 
sorry

theorem sum_S_n (t : ℝ) (n : ℕ) (H : 0 < t) :
    ∃ S_n : ℕ → ℝ, (S_n n = if t = 1 then ↑n else (t * (t^n - 1)) / (t - 1)) := 
sorry

end NUMINAMATH_GPT_expression_for_f_general_formula_a_n_sum_S_n_l2186_218688


namespace NUMINAMATH_GPT_find_square_side_length_l2186_218685

noncomputable def side_length_PQRS (x : ℝ) : Prop :=
  let PT := 1
  let QU := 2
  let RV := 3
  let SW := 4
  let PQRS_area := x^2
  let TUVW_area := 1 / 2 * x^2
  let triangle_area (base height : ℝ) : ℝ := 1 / 2 * base * height
  PQRS_area = x^2 ∧ TUVW_area = 1 / 2 * x^2 ∧
  triangle_area 1 (x - 4) + (x - 1) + 
  triangle_area 3 (x - 2) + 2 * (x - 3) = 1 / 2 * x^2

theorem find_square_side_length : ∃ x : ℝ, side_length_PQRS x ∧ x = 6 := 
  sorry

end NUMINAMATH_GPT_find_square_side_length_l2186_218685


namespace NUMINAMATH_GPT_liams_numbers_l2186_218648

theorem liams_numbers (x y : ℤ) 
  (h1 : 3 * x + 2 * y = 75)
  (h2 : x = 15)
  (h3 : ∃ k : ℕ, x * y = 5 * k) : 
  y = 15 := 
by
  sorry

end NUMINAMATH_GPT_liams_numbers_l2186_218648


namespace NUMINAMATH_GPT_no_rational_points_on_sqrt3_circle_l2186_218654

theorem no_rational_points_on_sqrt3_circle (x y : ℚ) : x^2 + y^2 ≠ 3 :=
sorry

end NUMINAMATH_GPT_no_rational_points_on_sqrt3_circle_l2186_218654


namespace NUMINAMATH_GPT_range_of_a_l2186_218659

theorem range_of_a 
  (a b x1 x2 x3 x4 : ℝ)
  (h1 : a ≠ 0)
  (h2 : a^2 ≠ 0)
  (hx1 : a * x1^2 + b * x1 + 1 = 0) 
  (hx2 : a * x2^2 + b * x2 + 1 = 0) 
  (hx3 : a^2 * x3^2 + b * x3 + 1 = 0) 
  (hx4 : a^2 * x4^2 + b * x4 + 1 = 0)
  (h_order : x3 < x1 ∧ x1 < x2 ∧ x2 < x4) :
  0 < a ∧ a < 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2186_218659


namespace NUMINAMATH_GPT_common_difference_l2186_218628

def arith_seq_common_difference (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference {a : ℕ → ℤ} (h₁ : a 5 = 3) (h₂ : a 6 = -2) : arith_seq_common_difference a (-5) :=
by
  intros n
  cases n with
  | zero => sorry -- base case: a 1 = a 0 + (-5), requires additional initial condition
  | succ n' => sorry -- inductive step

end NUMINAMATH_GPT_common_difference_l2186_218628


namespace NUMINAMATH_GPT_yaya_bike_walk_l2186_218670

theorem yaya_bike_walk (x y : ℝ) : 
  (x + y = 1.5 ∧ 15 * x + 5 * y = 20) ↔ (x + y = 1.5 ∧ 15 * x + 5 * y = 20) :=
by 
  sorry

end NUMINAMATH_GPT_yaya_bike_walk_l2186_218670


namespace NUMINAMATH_GPT_fruits_in_good_condition_l2186_218683

def percentage_good_fruits (num_oranges num_bananas pct_rotten_oranges pct_rotten_bananas : ℕ) : ℚ :=
  let total_fruits := num_oranges + num_bananas
  let rotten_oranges := (pct_rotten_oranges * num_oranges) / 100
  let rotten_bananas := (pct_rotten_bananas * num_bananas) / 100
  let good_fruits := total_fruits - (rotten_oranges + rotten_bananas)
  (good_fruits * 100) / total_fruits

theorem fruits_in_good_condition :
  percentage_good_fruits 600 400 15 8 = 87.8 := sorry

end NUMINAMATH_GPT_fruits_in_good_condition_l2186_218683


namespace NUMINAMATH_GPT_speed_conversion_l2186_218658

-- Define the given condition
def kmph_to_mps (v : ℕ) : ℕ := v * 5 / 18

-- Speed in kmph
def speed_kmph : ℕ := 216

-- The proof statement
theorem speed_conversion : kmph_to_mps speed_kmph = 60 :=
by
  sorry

end NUMINAMATH_GPT_speed_conversion_l2186_218658


namespace NUMINAMATH_GPT_original_area_of_triangle_l2186_218624

theorem original_area_of_triangle (A : ℝ) (h1 : 4 * A * 16 = 64) : A = 4 :=
by
  sorry

end NUMINAMATH_GPT_original_area_of_triangle_l2186_218624


namespace NUMINAMATH_GPT_clock_angle_5_30_l2186_218604

theorem clock_angle_5_30 (h_degree : ℕ → ℝ) (m_degree : ℕ → ℝ) (hours_pos : ℕ → ℝ) :
  (h_degree 12 = 360) →
  (m_degree 60 = 360) →
  (hours_pos 5 + h_degree 1 - (m_degree 30 / 2) = 165) →
  (m_degree 30 = 180) →
  ∃ θ : ℝ, θ = abs (m_degree 30 - (hours_pos 5 + h_degree 1 - (m_degree 30 / 2))) ∧ θ = 15 :=
by
  sorry

end NUMINAMATH_GPT_clock_angle_5_30_l2186_218604


namespace NUMINAMATH_GPT_no_valid_arrangement_in_7x7_grid_l2186_218689

theorem no_valid_arrangement_in_7x7_grid :
  ¬ (∃ (f : Fin 7 → Fin 7 → ℕ),
    (∀ (i j : Fin 6),
      (f i j + f i (j + 1) + f (i + 1) j + f (i + 1) (j + 1)) % 2 = 1) ∧
    (∀ (i j : Fin 5),
      (f i j + f i (j + 1) + f i (j + 2) + f (i + 1) j + f (i + 1) (j + 1) + f (i + 1) (j + 2) +
       f (i + 2) j + f (i + 2) (j + 1) + f (i + 2) (j + 2)) % 2 = 1)) := by
  sorry

end NUMINAMATH_GPT_no_valid_arrangement_in_7x7_grid_l2186_218689


namespace NUMINAMATH_GPT_parabola_property_l2186_218677

-- Define the conditions of the problem in Lean
variable (a b : ℝ)
variable (h1 : (a, b) ∈ {p : ℝ × ℝ | p.1^2 = 20 * p.2}) -- P lies on the parabola x^2 = 20y
variable (h2 : dist (a, b) (0, 5) = 25) -- Distance from P to focus F

theorem parabola_property : |a * b| = 400 := by
  sorry

end NUMINAMATH_GPT_parabola_property_l2186_218677


namespace NUMINAMATH_GPT_contribution_is_6_l2186_218630

-- Defining the earnings of each friend
def earning_1 : ℕ := 18
def earning_2 : ℕ := 22
def earning_3 : ℕ := 30
def earning_4 : ℕ := 35
def earning_5 : ℕ := 45

-- Defining the modified contribution for the highest earner
def modified_earning_5 : ℕ := 40

-- Calculate the total adjusted earnings
def total_earnings : ℕ := earning_1 + earning_2 + earning_3 + earning_4 + modified_earning_5

-- Calculate the equal share each friend should receive
def equal_share : ℕ := total_earnings / 5

-- Calculate the contribution needed from the friend who earned $35 to match the equal share
def contribution_from_earning_4 : ℕ := earning_4 - equal_share

-- Stating the proof problem
theorem contribution_is_6 : contribution_from_earning_4 = 6 := by
  sorry

end NUMINAMATH_GPT_contribution_is_6_l2186_218630


namespace NUMINAMATH_GPT_altitude_segment_product_eq_half_side_diff_square_l2186_218652

noncomputable def altitude_product (a b c t m m_1: ℝ) :=
  m * m_1 = (b^2 + c^2 - a^2) / 2

theorem altitude_segment_product_eq_half_side_diff_square {a b c t m m_1: ℝ}
  (hm : m = 2 * t / a)
  (hm_1 : m_1 = a * (b^2 + c^2 - a^2) / (4 * t)) :
  altitude_product a b c t m m_1 :=
by sorry

end NUMINAMATH_GPT_altitude_segment_product_eq_half_side_diff_square_l2186_218652


namespace NUMINAMATH_GPT_find_E_l2186_218655

variables (E F G H : ℕ)

noncomputable def conditions := 
  (E * F = 120) ∧ 
  (G * H = 120) ∧ 
  (E - F = G + H - 2) ∧ 
  (E ≠ F) ∧
  (E ≠ G) ∧ 
  (E ≠ H) ∧
  (F ≠ G) ∧
  (F ≠ H) ∧
  (G ≠ H)

theorem find_E (E F G H : ℕ) (h : conditions E F G H) : E = 30 :=
sorry

end NUMINAMATH_GPT_find_E_l2186_218655


namespace NUMINAMATH_GPT_imaginary_unit_problem_l2186_218642

theorem imaginary_unit_problem (i : ℂ) (h : i^2 = -1) :
  ( (1 + i) / i )^2014 = 2^(1007 : ℤ) * i :=
by sorry

end NUMINAMATH_GPT_imaginary_unit_problem_l2186_218642


namespace NUMINAMATH_GPT_solve_for_r_l2186_218699

variable (k r : ℝ)

theorem solve_for_r (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = 2 := sorry

end NUMINAMATH_GPT_solve_for_r_l2186_218699


namespace NUMINAMATH_GPT_inscribed_square_area_l2186_218631

theorem inscribed_square_area :
  (∃ (t : ℝ), (2*t)^2 = 4 * (t^2) ∧ ∀ (x y : ℝ), (x = t ∧ y = t ∨ x = -t ∧ y = t ∨ x = t ∧ y = -t ∨ x = -t ∧ y = -t) 
  → (x^2 / 4 + y^2 / 8 = 1) ) 
  → (∃ (a : ℝ), a = 32 / 3) := 
by
  sorry

end NUMINAMATH_GPT_inscribed_square_area_l2186_218631


namespace NUMINAMATH_GPT_cake_recipe_l2186_218617

theorem cake_recipe (flour : ℕ) (milk_per_200ml : ℕ) (egg_per_200ml : ℕ) (total_flour : ℕ)
  (h1 : milk_per_200ml = 60)
  (h2 : egg_per_200ml = 1)
  (h3 : total_flour = 800) :
  (total_flour / 200 * milk_per_200ml = 240) ∧ (total_flour / 200 * egg_per_200ml = 4) :=
by
  sorry

end NUMINAMATH_GPT_cake_recipe_l2186_218617


namespace NUMINAMATH_GPT_solution_set_inequality_l2186_218640

theorem solution_set_inequality (x : ℝ) (h1 : x < -3) (h2 : x < 2) : x < -3 :=
by
  exact h1

end NUMINAMATH_GPT_solution_set_inequality_l2186_218640


namespace NUMINAMATH_GPT_bus_speed_excluding_stoppages_l2186_218621

theorem bus_speed_excluding_stoppages (v : ℕ): (45 : ℝ) = (5 / 6 * v) → v = 54 :=
by
  sorry

end NUMINAMATH_GPT_bus_speed_excluding_stoppages_l2186_218621


namespace NUMINAMATH_GPT_james_paid_with_l2186_218681

variable (candy_packs : ℕ) (cost_per_pack : ℕ) (change_received : ℕ)

theorem james_paid_with (h1 : candy_packs = 3) (h2 : cost_per_pack = 3) (h3 : change_received = 11) :
  let total_cost := candy_packs * cost_per_pack
  let amount_paid := total_cost + change_received
  amount_paid = 20 :=
by
  sorry

end NUMINAMATH_GPT_james_paid_with_l2186_218681


namespace NUMINAMATH_GPT_range_of_m_l2186_218675

variable (m : ℝ)

def p : Prop := (m^2 - 4 > 0) ∧ (m > 0)
def q : Prop := 16 * (m - 2)^2 - 16 < 0

theorem range_of_m :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → (1 < m ∧ m ≤ 2) ∨ (3 ≤ m) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_m_l2186_218675


namespace NUMINAMATH_GPT_find_a_l2186_218650

def star (a b : ℝ) : ℝ := 3 * a - 2 * b^2

theorem find_a (a : ℝ) (h : star a 4 = 17) : a = 49 / 3 :=
by sorry

end NUMINAMATH_GPT_find_a_l2186_218650


namespace NUMINAMATH_GPT_validate_triangle_count_l2186_218622

noncomputable def count_valid_triangles : ℕ :=
  let total_points := 25
  let total_triples := (Nat.choose total_points 3)
  let collinear_rows := 5 * (Nat.choose 5 3)
  let collinear_columns := 5 * (Nat.choose 5 3)
  let main_diagonals := 2 * (Nat.choose 5 3)
  let secondary_diagonals := 8 * (Nat.choose 4 3)
  let invalid_triangles := collinear_rows + collinear_columns + main_diagonals + secondary_diagonals
  total_triples - invalid_triangles

theorem validate_triangle_count : count_valid_triangles = 2148 :=
by
  sorry

end NUMINAMATH_GPT_validate_triangle_count_l2186_218622


namespace NUMINAMATH_GPT_other_root_is_neg_2_l2186_218645

theorem other_root_is_neg_2 (k : ℝ) (h : Polynomial.eval 0 (Polynomial.C k + Polynomial.X * 2 + Polynomial.X ^ 2) = 0) : 
  ∃ t : ℝ, (Polynomial.eval t (Polynomial.C k + Polynomial.X * 2 + Polynomial.X ^ 2) = 0) ∧ t = -2 :=
by
  sorry

end NUMINAMATH_GPT_other_root_is_neg_2_l2186_218645


namespace NUMINAMATH_GPT_triangle_area_l2186_218668

theorem triangle_area (base height : ℕ) (h_base : base = 10) (h_height : height = 5) :
  (base * height) / 2 = 25 := by
  -- Proof is not required as per instructions.
  sorry

end NUMINAMATH_GPT_triangle_area_l2186_218668


namespace NUMINAMATH_GPT_regular_hexagon_has_greatest_lines_of_symmetry_l2186_218698

-- Definitions for the various shapes and their lines of symmetry.
def regular_pentagon_lines_of_symmetry : ℕ := 5
def parallelogram_lines_of_symmetry : ℕ := 0
def oval_ellipse_lines_of_symmetry : ℕ := 2
def right_triangle_lines_of_symmetry : ℕ := 0
def regular_hexagon_lines_of_symmetry : ℕ := 6

-- Theorem stating that the regular hexagon has the greatest number of lines of symmetry.
theorem regular_hexagon_has_greatest_lines_of_symmetry :
  regular_hexagon_lines_of_symmetry > regular_pentagon_lines_of_symmetry ∧
  regular_hexagon_lines_of_symmetry > parallelogram_lines_of_symmetry ∧
  regular_hexagon_lines_of_symmetry > oval_ellipse_lines_of_symmetry ∧
  regular_hexagon_lines_of_symmetry > right_triangle_lines_of_symmetry :=
by
  sorry

end NUMINAMATH_GPT_regular_hexagon_has_greatest_lines_of_symmetry_l2186_218698


namespace NUMINAMATH_GPT_exponential_function_f1_l2186_218687

theorem exponential_function_f1 (a : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) (h3 : a^3 = 8) : a^1 = 2 := by
  sorry

end NUMINAMATH_GPT_exponential_function_f1_l2186_218687
