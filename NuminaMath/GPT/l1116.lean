import Mathlib

namespace count_ways_to_sum_2020_as_1s_and_2s_l1116_111681

theorem count_ways_to_sum_2020_as_1s_and_2s : ∃ n, (∀ x y : ℕ, 4 * x + 5 * y = 2020 → x + y = n) → n = 102 :=
by
-- Mathematics proof needed.
sorry

end count_ways_to_sum_2020_as_1s_and_2s_l1116_111681


namespace intersect_circle_line_l1116_111666

theorem intersect_circle_line (k m : ℝ) : 
  (∃ (x y : ℝ), y = k * x + 2 * k ∧ x^2 + y^2 + m * x + 4 = 0) → m > 4 :=
by
  -- This statement follows from the conditions given in the problem
  -- You can use implicit for pure documentation
  -- We include a sorry here to skip the proof
  sorry

end intersect_circle_line_l1116_111666


namespace compare_f_values_l1116_111668

noncomputable def f (x : Real) : Real := 
  Real.cos x + 2 * x * (1 / 2)  -- given f''(pi/6) = 1/2

theorem compare_f_values :
  f (-Real.pi / 3) < f (Real.pi / 3) :=
by
  sorry

end compare_f_values_l1116_111668


namespace chess_tournament_participants_l1116_111650

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 171) : n = 19 := 
by sorry

end chess_tournament_participants_l1116_111650


namespace percent_not_red_balls_l1116_111698

theorem percent_not_red_balls (percent_cubes percent_red_balls : ℝ) 
  (h1 : percent_cubes = 0.3) (h2 : percent_red_balls = 0.25) : 
  (1 - percent_red_balls) * (1 - percent_cubes) = 0.525 :=
by
  sorry

end percent_not_red_balls_l1116_111698


namespace figure_perimeter_l1116_111602

-- Define the side length of the square and the triangles.
def square_side_length : ℕ := 3
def triangle_side_length : ℕ := 2

-- Calculate the perimeter of the figure
def perimeter (a b : ℕ) : ℕ := 2 * a + 2 * b

-- Statement to prove
theorem figure_perimeter : perimeter square_side_length triangle_side_length = 10 := 
by 
  -- "sorry" denotes that the proof is omitted.
  sorry

end figure_perimeter_l1116_111602


namespace ellipse_solution_length_AB_l1116_111674

noncomputable def ellipse_equation (a b : ℝ) (e : ℝ) (minor_axis : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ e = 3 / 4 ∧ 2 * b = minor_axis ∧ minor_axis = 2 * Real.sqrt 7

theorem ellipse_solution (a b : ℝ) (e : ℝ) (minor_axis : ℝ) :
  ellipse_equation a b e minor_axis →
  (a^2 = 16 ∧ b^2 = 7 ∧ (1 / a^2) = 1 / 16 ∧ (1 / b^2) = 1 / 7) :=
by 
  intros h
  sorry

noncomputable def area_ratio (S1 S2 : ℝ) : Prop :=
  S1 / S2 = 9 / 13

theorem length_AB (S1 S2 : ℝ) :
  area_ratio S1 S2 →
  |S1 / S2| = |(9 * Real.sqrt 105) / 26| :=
by
  intros h
  sorry

end ellipse_solution_length_AB_l1116_111674


namespace f_properties_l1116_111641

variable (f : ℝ → ℝ)
variable (f_pos : ∀ x : ℝ, f x > 0)
variable (f_eq : ∀ a b : ℝ, f a * f b = f (a + b))

theorem f_properties :
  (f 0 = 1) ∧
  (∀ a : ℝ, f (-a) = 1 / f a) ∧
  (∀ a : ℝ, f a = (f (3 * a))^(1/3)) :=
by {
  sorry
}

end f_properties_l1116_111641


namespace fill_entire_bucket_l1116_111607

theorem fill_entire_bucket (h : (2/3 : ℝ) * t = 2) : t = 3 :=
sorry

end fill_entire_bucket_l1116_111607


namespace negation_of_exists_lt_l1116_111605

theorem negation_of_exists_lt :
  (¬ ∃ x : ℝ, x^2 + 2 * x + 3 < 0) = (∀ x : ℝ, x^2 + 2 * x + 3 ≥ 0) :=
by sorry

end negation_of_exists_lt_l1116_111605


namespace polynomial_no_negative_roots_l1116_111682

theorem polynomial_no_negative_roots (x : ℝ) (h : x < 0) : x^4 - 4 * x^3 - 6 * x^2 - 3 * x + 9 ≠ 0 := 
by 
  sorry

end polynomial_no_negative_roots_l1116_111682


namespace find_x_l1116_111617

theorem find_x (b x : ℝ) (hb : b > 1) (hx : x > 0)
  (h : (3 * x)^(Real.log 3 / Real.log b) - (5 * x)^(Real.log 5 / Real.log b) = 0) :
  x = 1 / 5^(Real.log b / Real.log (5 / 3)) :=
by
  sorry

end find_x_l1116_111617


namespace smallest_k_l1116_111695

theorem smallest_k (k : ℕ) (h₁ : k > 1) (h₂ : k % 17 = 1) (h₃ : k % 6 = 1) (h₄ : k % 2 = 1) : k = 103 :=
by sorry

end smallest_k_l1116_111695


namespace addition_correct_l1116_111606

theorem addition_correct :
  1357 + 2468 + 3579 + 4680 + 5791 = 17875 := 
by
  sorry

end addition_correct_l1116_111606


namespace greatest_power_of_2_factor_l1116_111667

theorem greatest_power_of_2_factor
    : ∃ k : ℕ, (2^k) ∣ (10^1503 - 4^752) ∧ ∀ m : ℕ, (2^(m+1)) ∣ (10^1503 - 4^752) → m < k :=
by
    sorry

end greatest_power_of_2_factor_l1116_111667


namespace circle_properties_l1116_111644

theorem circle_properties (D r C A : ℝ) (h1 : D = 15)
  (h2 : r = 7.5)
  (h3 : C = 15 * Real.pi)
  (h4 : A = 56.25 * Real.pi) :
  (9 ^ 2 + 12 ^ 2 = D ^ 2) ∧ (D = 2 * r) ∧ (C = Real.pi * D) ∧ (A = Real.pi * r ^ 2) :=
by
  sorry

end circle_properties_l1116_111644


namespace power_of_power_rule_l1116_111643

theorem power_of_power_rule (h : 128 = 2^7) : (128: ℝ)^(4/7) = 16 := by
  sorry

end power_of_power_rule_l1116_111643


namespace pieces_brought_to_school_on_friday_l1116_111678

def pieces_of_fruit_mark_had := 10
def pieces_eaten_first_four_days := 5
def pieces_kept_for_next_week := 2

theorem pieces_brought_to_school_on_friday :
  pieces_of_fruit_mark_had - pieces_eaten_first_four_days - pieces_kept_for_next_week = 3 :=
by
  sorry

end pieces_brought_to_school_on_friday_l1116_111678


namespace average_of_solutions_l1116_111684

theorem average_of_solutions (a b : ℝ) (h : ∃ x1 x2 : ℝ, a * x1 ^ 2 + 3 * a * x1 + b = 0 ∧ a * x2 ^ 2 + 3 * a * x2 + b = 0) :
  ((-3 : ℝ) / 2) = - 3 / 2 :=
by sorry

end average_of_solutions_l1116_111684


namespace find_integer_closest_expression_l1116_111658

theorem find_integer_closest_expression :
  let a := (7 + Real.sqrt 48) ^ 2023
  let b := (7 - Real.sqrt 48) ^ 2023
  ((a + b) ^ 2 - (a - b) ^ 2) = 4 :=
by
  sorry

end find_integer_closest_expression_l1116_111658


namespace part1_part2_l1116_111619

def f (x : ℝ) (a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2*a + 1)

theorem part1 (x : ℝ) : f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 :=
by sorry

theorem part2 (a : ℝ) : (∀ x, f x a ≥ 4) ↔ (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l1116_111619


namespace first_person_job_completion_time_l1116_111622

noncomputable def job_completion_time :=
  let A := 1 - (1/5)
  let C := 1/8
  let combined_rate := A + C
  have h1 : combined_rate = 0.325 := by
    sorry
  have h2 : A ≠ 0 := by
    sorry
  (1 / A : ℝ)
  
theorem first_person_job_completion_time :
  job_completion_time = 1.25 :=
by
  sorry

end first_person_job_completion_time_l1116_111622


namespace cost_price_of_item_l1116_111625

theorem cost_price_of_item 
  (retail_price : ℝ) (reduction_percentage : ℝ) 
  (additional_discount : ℝ) (profit_percentage : ℝ) 
  (selling_price : ℝ) 
  (h1 : retail_price = 900)
  (h2 : reduction_percentage = 0.1)
  (h3 : additional_discount = 48)
  (h4 : profit_percentage = 0.2)
  (h5 : selling_price = 762) :
  ∃ x : ℝ, selling_price = 1.2 * x ∧ x = 635 := 
by {
  sorry
}

end cost_price_of_item_l1116_111625


namespace parallel_lines_m_value_l1116_111631

/-- Given two lines l_1: (3 + m) * x + 4 * y = 5 - 3 * m, and l_2: 2 * x + (5 + m) * y = 8,
the value of m for which l_1 is parallel to l_2 is -7. -/
theorem parallel_lines_m_value (m : ℝ) :
  (∀ x y : ℝ, (3 + m) * x + 4 * y = 5 - 3 * m) →
  (∀ x y : ℝ, 2 * x + (5 + m) * y = 8) →
  m = -7 :=
sorry

end parallel_lines_m_value_l1116_111631


namespace intensity_on_Thursday_l1116_111659

-- Step a) - Definitions from Conditions
def inversely_proportional (i b k : ℕ) : Prop := i * b = k

-- Translation of the proof problem
theorem intensity_on_Thursday (k b : ℕ) (h₁ : k = 24) (h₂ : b = 3) : ∃ i, inversely_proportional i b k ∧ i = 8 := 
by
  sorry

end intensity_on_Thursday_l1116_111659


namespace part_a_part_b_case1_part_b_case2_l1116_111635

theorem part_a (x1 x2 p : ℝ) 
  (h1 : 4 * (x1 ^ 2) + x1 + 4 * p = 0)
  (h2 : 4 * (x2 ^ 2) + x2 + 4 * p = 0) 
  (h3 : x1 / x2 + x2 / x1 = -9 / 4) : 
  p = -1 / 23 :=
sorry

theorem part_b_case1 (x1 x2 p : ℝ) 
  (h1 : 4 * (x1 ^ 2) + x1 + 4 * p = 0)
  (h2 : 4 * (x2 ^ 2) + x2 + 4 * p = 0) 
  (h3 : x2 = x1^2 - 1) : 
  p = -3 / 8 :=
sorry

theorem part_b_case2 (x1 x2 p : ℝ) 
  (h1 : 4 * (x1 ^ 2) + x1 + 4 * p = 0)
  (h2 : 4 * (x2 ^ 2) + x2 + 4 * p = 0) 
  (h3 : x2 = x1^2 - 1) : 
  p = -15 / 8 :=
sorry

end part_a_part_b_case1_part_b_case2_l1116_111635


namespace merchant_problem_l1116_111613

theorem merchant_problem (P C : ℝ) (h1 : P + C = 60) (h2 : 2.40 * P + 6.00 * C = 180) : C = 10 := 
by
  -- Proof goes here
  sorry

end merchant_problem_l1116_111613


namespace misha_grade_students_l1116_111600

theorem misha_grade_students (n : ℕ) (h1 : n = 75) (h2 : n = 75) : 2 * n - 1 = 149 := 
by
  sorry

end misha_grade_students_l1116_111600


namespace functions_with_inverses_l1116_111663

-- Definitions for the conditions
def passes_Horizontal_Line_Test_A : Prop := false
def passes_Horizontal_Line_Test_B : Prop := true
def passes_Horizontal_Line_Test_C : Prop := true
def passes_Horizontal_Line_Test_D : Prop := false
def passes_Horizontal_Line_Test_E : Prop := false

-- Proof statement
theorem functions_with_inverses :
  (passes_Horizontal_Line_Test_A = false) ∧
  (passes_Horizontal_Line_Test_B = true) ∧
  (passes_Horizontal_Line_Test_C = true) ∧
  (passes_Horizontal_Line_Test_D = false) ∧
  (passes_Horizontal_Line_Test_E = false) →
  ([B, C] = which_functions_have_inverses) :=
sorry

end functions_with_inverses_l1116_111663


namespace solve_inequality_l1116_111679

theorem solve_inequality :
  {x : ℝ | (3 * x + 1) * (2 * x - 1) < 0} = {x : ℝ | -1 / 3 < x ∧ x < 1 / 2} :=
  sorry

end solve_inequality_l1116_111679


namespace ab_over_a_minus_b_l1116_111687

theorem ab_over_a_minus_b (a b : ℝ) (h : (1 / a) - (1 / b) = 1 / 3) : (a * b) / (a - b) = -3 := by
  sorry

end ab_over_a_minus_b_l1116_111687


namespace jacob_initial_fish_count_l1116_111648

theorem jacob_initial_fish_count : 
  ∃ J : ℕ, 
    (∀ A : ℕ, A = 7 * J) → 
    (A' = A - 23) → 
    (J + 26 = A' + 1) → 
    J = 8 := 
by 
  sorry

end jacob_initial_fish_count_l1116_111648


namespace andrew_current_age_l1116_111633

-- Definitions based on conditions.
def initial_age := 11  -- Andrew started donating at age 11
def donation_per_year := 7  -- Andrew donates 7k each year on his birthday
def total_donation := 133  -- Andrew has donated a total of 133k till now

-- The theorem stating the problem and the conclusion.
theorem andrew_current_age : 
  ∃ (A : ℕ), donation_per_year * (A - initial_age) = total_donation :=
by {
  sorry
}

end andrew_current_age_l1116_111633


namespace discount_given_l1116_111672

variables (initial_money : ℕ) (extra_fraction : ℕ) (additional_money_needed : ℕ)
variables (total_with_discount : ℕ) (discount_amount : ℕ)

def total_without_discount (initial_money : ℕ) (extra_fraction : ℕ) : ℕ :=
  initial_money + extra_fraction

def discount (initial_money : ℕ) (total_without_discount : ℕ) (total_with_discount : ℕ) : ℕ :=
  total_without_discount - total_with_discount

def discount_percentage (discount_amount : ℕ) (total_without_discount : ℕ) : ℚ :=
  (discount_amount : ℚ) / (total_without_discount : ℚ) * 100

theorem discount_given 
  (initial_money : ℕ := 500)
  (extra_fraction : ℕ := 200)
  (additional_money_needed : ℕ := 95)
  (total_without_discount₀ : ℕ := total_without_discount initial_money extra_fraction)
  (total_with_discount₀ : ℕ := initial_money + additional_money_needed)
  (discount_amount₀ : ℕ := discount initial_money total_without_discount₀ total_with_discount₀)
  : discount_percentage discount_amount₀ total_without_discount₀ = 15 :=
by sorry

end discount_given_l1116_111672


namespace x_squared_minus_y_squared_l1116_111651

theorem x_squared_minus_y_squared (x y : ℚ) (h1 : x + y = 2 / 5) (h2 : x - y = 1 / 10) : x ^ 2 - y ^ 2 = 1 / 25 :=
by
  sorry

end x_squared_minus_y_squared_l1116_111651


namespace minimum_a_l1116_111642

open Real

theorem minimum_a (a : ℝ) : (∀ (x y : ℝ), x > 0 → y > 0 → (1 / x + a / y) ≥ (16 / (x + y))) → a ≥ 9 := by
sorry

end minimum_a_l1116_111642


namespace calc_expr_l1116_111634

theorem calc_expr : 
  (-1: ℝ)^4 - 2 * Real.tan (Real.pi / 3) + (Real.sqrt 3 - Real.sqrt 2)^0 + Real.sqrt 12 = 2 := 
by
  sorry

end calc_expr_l1116_111634


namespace min_value_p_plus_q_l1116_111601

theorem min_value_p_plus_q (p q : ℕ) (hp : 1 < p) (hq : 1 < q) 
  (h : 17 * (p + 1) = 20 * (q + 1)) : p + q = 37 :=
sorry

end min_value_p_plus_q_l1116_111601


namespace hyperbola_eccentricity_l1116_111661

theorem hyperbola_eccentricity (a : ℝ) (h : 0 < a) (h_asymptote : Real.tan (Real.pi / 6) = 1 / a) : 
  (a = Real.sqrt 3) → 
  (∃ e : ℝ, e = (2 * Real.sqrt 3) / 3) :=
by
  intros
  sorry

end hyperbola_eccentricity_l1116_111661


namespace red_tile_probability_l1116_111671

def is_red_tile (n : ℕ) : Prop := n % 7 = 3

noncomputable def red_tiles_count : ℕ :=
  Nat.card {n : ℕ | n ≤ 70 ∧ is_red_tile n}

noncomputable def total_tiles_count : ℕ := 70

theorem red_tile_probability :
  (red_tiles_count : ℤ) / (total_tiles_count : ℤ) = (1 : ℤ) / 7 :=
sorry

end red_tile_probability_l1116_111671


namespace geometric_sequence_problem_l1116_111677

noncomputable def geom_sequence (a r : ℝ) (n : ℕ) : ℝ := a * r ^ n

theorem geometric_sequence_problem (a r : ℝ) (a4 a8 a6 a10 : ℝ) :
  a4 = geom_sequence a r 4 →
  a8 = geom_sequence a r 8 →
  a6 = geom_sequence a r 6 →
  a10 = geom_sequence a r 10 →
  a4 + a8 = -2 →
  a4^2 + 2 * a6^2 + a6 * a10 = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end geometric_sequence_problem_l1116_111677


namespace points_subtracted_per_wrong_answer_l1116_111675

theorem points_subtracted_per_wrong_answer 
  (total_problems : ℕ) 
  (wrong_answers : ℕ) 
  (score : ℕ) 
  (points_per_right_answer : ℕ) 
  (correct_answers : ℕ)
  (subtracted_points : ℕ) 
  (expected_points : ℕ) 
  (points_subtracted : ℕ) :
  total_problems = 25 → 
  wrong_answers = 3 → 
  score = 85 → 
  points_per_right_answer = 4 → 
  correct_answers = total_problems - wrong_answers → 
  expected_points = correct_answers * points_per_right_answer → 
  subtracted_points = expected_points - score → 
  points_subtracted = subtracted_points / wrong_answers → 
  points_subtracted = 1 := 
by
  intros;
  sorry

end points_subtracted_per_wrong_answer_l1116_111675


namespace mean_temperature_l1116_111653

def temperatures : List ℝ := [-6.5, -2, -3.5, -1, 0.5, 4, 1.5]

theorem mean_temperature : (temperatures.sum / temperatures.length) = -1 := by
  sorry

end mean_temperature_l1116_111653


namespace find_B_l1116_111637

theorem find_B (A C B : ℕ) (hA : A = 520) (hC : C = A + 204) (hCB : C = B + 179) : B = 545 :=
by
  sorry

end find_B_l1116_111637


namespace gcd_of_power_of_two_plus_one_l1116_111621

theorem gcd_of_power_of_two_plus_one (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : m ≠ n) : 
  Nat.gcd (2^(2^m) + 1) (2^(2^n) + 1) = 1 := 
sorry

end gcd_of_power_of_two_plus_one_l1116_111621


namespace avg_difference_even_avg_difference_odd_l1116_111693

noncomputable def avg (seq : List ℕ) : ℚ := (seq.sum : ℚ) / seq.length

def even_ints_20_to_60 := [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]
def even_ints_10_to_140 := [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140]

def odd_ints_21_to_59 := [21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59]
def odd_ints_11_to_139 := [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137, 139]

theorem avg_difference_even :
  avg even_ints_20_to_60 - avg even_ints_10_to_140 = -35 := sorry

theorem avg_difference_odd :
  avg odd_ints_21_to_59 - avg odd_ints_11_to_139 = -35 := sorry

end avg_difference_even_avg_difference_odd_l1116_111693


namespace beef_weight_loss_l1116_111611

theorem beef_weight_loss (weight_before weight_after: ℕ) 
                         (h1: weight_before = 400) 
                         (h2: weight_after = 240) : 
                         ((weight_before - weight_after) * 100 / weight_before = 40) :=
by 
  sorry

end beef_weight_loss_l1116_111611


namespace find_f_comp_f_l1116_111604

def f (x : ℚ) : ℚ :=
  if x ≤ 1 then x + 1 else -x + 3

theorem find_f_comp_f (h : f (f (5/2)) = 3/2) :
  f (f (5/2)) = 3/2 := by
  sorry

end find_f_comp_f_l1116_111604


namespace inequality_condition_l1116_111694

noncomputable def inequality_holds_for_all (a b c : ℝ) : Prop :=
  ∀ (x : ℝ), a * Real.sin x + b * Real.cos x + c > 0

theorem inequality_condition (a b c : ℝ) :
  inequality_holds_for_all a b c ↔ Real.sqrt (a^2 + b^2) < c :=
by sorry

end inequality_condition_l1116_111694


namespace clipping_per_friend_l1116_111676

def GluePerClipping : Nat := 6
def TotalGlue : Nat := 126
def TotalFriends : Nat := 7

theorem clipping_per_friend :
  (TotalGlue / GluePerClipping) / TotalFriends = 3 := by
  sorry

end clipping_per_friend_l1116_111676


namespace Jackie_apples_count_l1116_111652

variable (Adam_apples Jackie_apples : ℕ)
variable (h1 : Adam_apples = 10)
variable (h2 : Adam_apples = Jackie_apples + 8)

theorem Jackie_apples_count : Jackie_apples = 2 := by
  sorry

end Jackie_apples_count_l1116_111652


namespace parabola_y_intercepts_l1116_111665

theorem parabola_y_intercepts : ∃ y1 y2 : ℝ, (3 * y1^2 - 6 * y1 + 2 = 0) ∧ (3 * y2^2 - 6 * y2 + 2 = 0) ∧ (y1 ≠ y2) :=
by 
  sorry

end parabola_y_intercepts_l1116_111665


namespace resulting_total_mass_l1116_111696

-- Define initial conditions
def initial_total_mass : ℝ := 12
def initial_white_paint_mass : ℝ := 0.8 * initial_total_mass
def initial_black_paint_mass : ℝ := initial_total_mass - initial_white_paint_mass

-- Required condition for the new mixture
def final_white_paint_percentage : ℝ := 0.9

-- Prove that the resulting total mass of paint is 24 kg
theorem resulting_total_mass (x : ℝ) (h1 : initial_total_mass = 12) 
                            (h2 : initial_white_paint_mass = 0.8 * initial_total_mass)
                            (h3 : initial_black_paint_mass = initial_total_mass - initial_white_paint_mass)
                            (h4 : final_white_paint_percentage = 0.9) 
                            (h5 : (initial_white_paint_mass + x) / (initial_total_mass + x) = final_white_paint_percentage) : 
                            initial_total_mass + x = 24 :=
by 
  -- Temporarily assume the proof without detailing the solution steps
  sorry

end resulting_total_mass_l1116_111696


namespace solve_for_x_l1116_111697

theorem solve_for_x (x : ℚ) (h : 10 * x = x + 20) : x = 20 / 9 :=
  sorry

end solve_for_x_l1116_111697


namespace math_problem_l1116_111660

-- Definitions for increasing function and periodic function
def increasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f x ≤ f y
def periodic (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x : ℝ, f (x + T) = f x

-- The main theorem statement
theorem math_problem (f g h : ℝ → ℝ) (T : ℝ) :
  (∀ x y : ℝ, x < y → f x + g x ≤ f y + g y) ∧ (∀ x y : ℝ, x < y → f x + h x ≤ f y + h y) ∧ (∀ x y : ℝ, x < y → g x + h x ≤ g y + h y) → 
  ¬(increasing g) ∧
  (∀ x : ℝ, f (x + T) + g (x + T) = f x + g x ∧ f (x + T) + h (x + T) = f x + h x ∧ g (x + T) + h (x + T) = g x + h x) → 
  increasing f ∧ increasing g ∧ increasing h :=
sorry

end math_problem_l1116_111660


namespace log_sum_l1116_111632

open Real

theorem log_sum : log 2 + log 5 = 1 :=
sorry

end log_sum_l1116_111632


namespace teachers_no_conditions_percentage_l1116_111645

theorem teachers_no_conditions_percentage :
  let total_teachers := 150
  let high_blood_pressure := 90
  let heart_trouble := 60
  let both_hbp_ht := 30
  let diabetes := 10
  let both_diabetes_ht := 5
  let both_diabetes_hbp := 8
  let all_three := 3

  let only_hbp := high_blood_pressure - both_hbp_ht - both_diabetes_hbp - all_three
  let only_ht := heart_trouble - both_hbp_ht - both_diabetes_ht - all_three
  let only_diabetes := diabetes - both_diabetes_hbp - both_diabetes_ht - all_three
  let both_hbp_ht_only := both_hbp_ht - all_three
  let both_hbp_diabetes_only := both_diabetes_hbp - all_three
  let both_ht_diabetes_only := both_diabetes_ht - all_three
  let any_condition := only_hbp + only_ht + only_diabetes + both_hbp_ht_only + both_hbp_diabetes_only + both_ht_diabetes_only + all_three
  let no_conditions := total_teachers - any_condition

  (no_conditions / total_teachers * 100) = 28 :=
by
  sorry

end teachers_no_conditions_percentage_l1116_111645


namespace number_of_jars_pasta_sauce_l1116_111629

-- Conditions
def pasta_cost_per_kg := 1.5
def pasta_weight_kg := 2.0
def ground_beef_cost_per_kg := 8.0
def ground_beef_weight_kg := 1.0 / 4.0
def quesadilla_cost := 6.0
def jar_sauce_cost := 2.0
def total_money := 15.0

-- Helper definitions for total costs
def pasta_total_cost := pasta_weight_kg * pasta_cost_per_kg
def ground_beef_total_cost := ground_beef_weight_kg * ground_beef_cost_per_kg
def other_total_cost := quesadilla_cost + pasta_total_cost + ground_beef_total_cost
def remaining_money := total_money - other_total_cost

-- Proof statement
theorem number_of_jars_pasta_sauce :
  (remaining_money / jar_sauce_cost) = 2 := by
  sorry

end number_of_jars_pasta_sauce_l1116_111629


namespace find_real_pairs_l1116_111620

theorem find_real_pairs (x y : ℝ) (h : 2 * x / (1 + x^2) = (1 + y^2) / (2 * y)) : 
  (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) :=
sorry

end find_real_pairs_l1116_111620


namespace find_values_l1116_111628

theorem find_values (a b : ℕ) (h1 : (a + 1) % b = 0) (h2 : a = 2 * b + 5) (h3 : Nat.Prime (a + 7 * b)) : (a = 9 ∧ b = 2) ∨ (a = 17 ∧ b = 6) :=
sorry

end find_values_l1116_111628


namespace evaluate_expression_l1116_111609

theorem evaluate_expression (x y : ℕ) (h1 : x = 2) (h2 : y = 3) : 3 * x^y + 4 * y^x = 60 :=
by
  rw [h1, h2]
  sorry

end evaluate_expression_l1116_111609


namespace valid_numbers_l1116_111603

noncomputable def is_valid_number (a : ℕ) : Prop :=
  ∃ b c d x y : ℕ, 
    a = b * c + d ∧
    a = 10 * x + y ∧
    x > 0 ∧ x ≤ 9 ∧ y ≤ 9 ∧
    10 * x + y = 4 * x + 4 * y

theorem valid_numbers : 
  ∃ a : ℕ, (a = 12 ∨ a = 24 ∨ a = 36 ∨ a = 48) ∧ is_valid_number a :=
by
  sorry

end valid_numbers_l1116_111603


namespace inequality_f_l1116_111639

noncomputable def f (x y z : ℝ) : ℝ :=
  x * y + y * z + z * x - 2 * x * y * z

theorem inequality_f (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 1) :
  0 ≤ f x y z ∧ f x y z ≤ 7 / 27 :=
  sorry

end inequality_f_l1116_111639


namespace cos_value_l1116_111670

theorem cos_value (α : ℝ) (h : Real.sin (π / 5 - α) = 1 / 4) : 
  Real.cos (2 * α + 3 * π / 5) = -7 / 8 := 
by
  sorry

end cos_value_l1116_111670


namespace factor_expression_l1116_111699

theorem factor_expression :
  (8 * x^6 + 36 * x^4 - 5) - (2 * x^6 - 6 * x^4 + 5) = 2 * (3 * x^6 + 21 * x^4 - 5) :=
by
  sorry

end factor_expression_l1116_111699


namespace solve_xyz_sum_l1116_111680

theorem solve_xyz_sum :
  ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ (x+y+z)^3 - x^3 - y^3 - z^3 = 378 ∧ x+y+z = 9 :=
by
  sorry

end solve_xyz_sum_l1116_111680


namespace determine_original_volume_of_tank_l1116_111683

noncomputable def salt_volume (x : ℝ) := 0.20 * x
noncomputable def new_volume_after_evaporation (x : ℝ) := (3 / 4) * x
noncomputable def new_volume_after_additions (x : ℝ) := (3 / 4) * x + 6 + 12
noncomputable def new_salt_after_addition (x : ℝ) := 0.20 * x + 12
noncomputable def resulting_salt_concentration (x : ℝ) := (0.20 * x + 12) / ((3 / 4) * x + 18)

theorem determine_original_volume_of_tank (x : ℝ) :
  resulting_salt_concentration x = 1 / 3 → x = 120 := 
by 
  sorry

end determine_original_volume_of_tank_l1116_111683


namespace cars_needed_to_double_earnings_l1116_111656

-- Define the conditions
def baseSalary : Int := 1000
def commissionPerCar : Int := 200
def januaryEarnings : Int := 1800

-- The proof goal
theorem cars_needed_to_double_earnings : 
  ∃ (carsSoldInFeb : Int), 
    1000 + commissionPerCar * carsSoldInFeb = 2 * januaryEarnings :=
by
  sorry

end cars_needed_to_double_earnings_l1116_111656


namespace xy_sum_l1116_111630

namespace ProofExample

variable (x y : ℚ)

def condition1 : Prop := (1 / x) + (1 / y) = 4
def condition2 : Prop := (1 / x) - (1 / y) = -6

theorem xy_sum : condition1 x y → condition2 x y → (x + y = -4 / 5) := by
  intros
  sorry

end ProofExample

end xy_sum_l1116_111630


namespace tank_overflows_after_24_minutes_l1116_111616

theorem tank_overflows_after_24_minutes 
  (rateA : ℝ) (rateB : ℝ) (t : ℝ) 
  (hA : rateA = 1) 
  (hB : rateB = 4) :
  t - 1/4 * rateB + t * rateA = 1 → t = 2/5 :=
by 
  intros h
  -- the proof steps go here
  sorry

end tank_overflows_after_24_minutes_l1116_111616


namespace geo_seq_fifth_term_l1116_111649

theorem geo_seq_fifth_term (a r : ℝ) (a_pos : 0 < a) (r_pos : 0 < r)
  (h3 : a * r^2 = 8) (h7 : a * r^6 = 18) : a * r^4 = 12 :=
sorry

end geo_seq_fifth_term_l1116_111649


namespace find_a_l1116_111618

theorem find_a (a b c : ℕ) (h1 : a ≥ b ∧ b ≥ c)  
  (h2 : (a:ℤ) ^ 2 - b ^ 2 - c ^ 2 + a * b = 2011) 
  (h3 : (a:ℤ) ^ 2 + 3 * b ^ 2 + 3 * c ^ 2 - 3 * a * b - 2 * a * c - 2 * b * c = -1997) : 
a = 253 := 
sorry

end find_a_l1116_111618


namespace heidi_paints_fraction_in_10_minutes_l1116_111640

variable (Heidi_paint_rate : ℕ → ℝ)
variable (t : ℕ)
variable (fraction : ℝ)

theorem heidi_paints_fraction_in_10_minutes 
  (h1 : Heidi_paint_rate 30 = 1) 
  (h2 : t = 10) 
  (h3 : fraction = 1 / 3) : 
  Heidi_paint_rate t = fraction := 
sorry

end heidi_paints_fraction_in_10_minutes_l1116_111640


namespace rod_length_l1116_111608

theorem rod_length (L : ℝ) (weight : ℝ → ℝ) (weight_6m : weight 6 = 14.04) (weight_L : weight L = 23.4) :
  L = 10 :=
by 
  sorry

end rod_length_l1116_111608


namespace min_value_of_a_plus_2b_l1116_111655

theorem min_value_of_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) : a + 2*b = 3 + 2*Real.sqrt 2 := 
sorry

end min_value_of_a_plus_2b_l1116_111655


namespace number_of_valid_x_l1116_111612

theorem number_of_valid_x (x : ℕ) : 
  ((x + 3) * (x - 3) * (x ^ 2 + 9) < 500) ∧ (x - 3 > 0) ↔ x = 4 :=
sorry

end number_of_valid_x_l1116_111612


namespace new_cost_after_decrease_l1116_111664

def actual_cost : ℝ := 2400
def decrease_percentage : ℝ := 0.50
def decreased_amount (cost percentage : ℝ) : ℝ := percentage * cost
def new_cost (cost decreased : ℝ) : ℝ := cost - decreased

theorem new_cost_after_decrease :
  new_cost actual_cost (decreased_amount actual_cost decrease_percentage) = 1200 :=
by sorry

end new_cost_after_decrease_l1116_111664


namespace inequality_holds_l1116_111615

variable {f : ℝ → ℝ}

-- Conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_monotonic_on_nonneg_interval (f : ℝ → ℝ) : Prop := ∀ x y, (0 ≤ x ∧ x < y ∧ y < 8) → f y ≤ f x

axiom condition1 : is_even f
axiom condition2 : is_monotonic_on_nonneg_interval f
axiom condition3 : f (-3) < f 2

-- The statement to be proven
theorem inequality_holds : f 5 < f (-3) ∧ f (-3) < f (-1) :=
by
  sorry

end inequality_holds_l1116_111615


namespace test_total_points_l1116_111691

def total_points (total_problems comp_problems : ℕ) (points_comp points_word : ℕ) : ℕ :=
  let word_problems := total_problems - comp_problems
  (comp_problems * points_comp) + (word_problems * points_word)

theorem test_total_points :
  total_points 30 20 3 5 = 110 := by
  sorry

end test_total_points_l1116_111691


namespace incorrect_conclusion_l1116_111623

noncomputable def data_set : List ℕ := [4, 1, 6, 2, 9, 5, 8]
def mean_x : ℝ := 2
def mean_y : ℝ := 20
def regression_eq (x : ℝ) : ℝ := 9.1 * x + 1.8
def chi_squared_value : ℝ := 9.632
def alpha : ℝ := 0.001
def critical_value : ℝ := 10.828

theorem incorrect_conclusion : ¬(chi_squared_value ≥ critical_value) := by
  -- Insert proof here
  sorry

end incorrect_conclusion_l1116_111623


namespace nautical_mile_to_land_mile_l1116_111662

theorem nautical_mile_to_land_mile 
    (speed_one_sail : ℕ := 25) 
    (speed_two_sails : ℕ := 50) 
    (travel_time_one_sail : ℕ := 4) 
    (travel_time_two_sails : ℕ := 4)
    (total_distance : ℕ := 345) : 
    ∃ (x : ℚ), x = 1.15 ∧ 
    total_distance = travel_time_one_sail * speed_one_sail * x +
                    travel_time_two_sails * speed_two_sails * x := 
by
  sorry

end nautical_mile_to_land_mile_l1116_111662


namespace interview_passing_probability_l1116_111669

def probability_of_passing_interview (p : ℝ) : ℝ :=
  p + (1 - p) * p + (1 - p) * (1 - p) * p

theorem interview_passing_probability : probability_of_passing_interview 0.7 = 0.973 :=
by
  -- proof steps to be filled
  sorry

end interview_passing_probability_l1116_111669


namespace no_solution_frac_eq_l1116_111614

theorem no_solution_frac_eq (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1) : 
  3 / x + 6 / (x - 1) - (x + 5) / (x * (x - 1)) ≠ 0 :=
by {
  sorry
}

end no_solution_frac_eq_l1116_111614


namespace gcd_of_45_135_225_is_45_l1116_111685

theorem gcd_of_45_135_225_is_45 : Nat.gcd (Nat.gcd 45 135) 225 = 45 :=
by
  sorry

end gcd_of_45_135_225_is_45_l1116_111685


namespace volume_hemisphere_from_sphere_l1116_111689

theorem volume_hemisphere_from_sphere (r : ℝ) (V_sphere : ℝ) (V_hemisphere : ℝ) 
  (h1 : V_sphere = 150 * Real.pi) 
  (h2 : V_sphere = (4 / 3) * Real.pi * r^3) : 
  V_hemisphere = 75 * Real.pi :=
by
  sorry

end volume_hemisphere_from_sphere_l1116_111689


namespace stella_profit_loss_l1116_111646

theorem stella_profit_loss :
  let dolls := 6
  let clocks := 4
  let glasses := 8
  let vases := 3
  let postcards := 10
  let dolls_price := 8
  let clocks_price := 25
  let glasses_price := 6
  let vases_price := 12
  let postcards_price := 3
  let cost := 250
  let clocks_discount_threshold := 2
  let clocks_discount := 10 / 100
  let glasses_bundle := 3
  let glasses_bundle_price := 2 * glasses_price
  let sales_tax_rate := 5 / 100
  let dolls_revenue := dolls * dolls_price
  let clocks_revenue_full := clocks * clocks_price
  let clocks_discounts_count := clocks / clocks_discount_threshold
  let clocks_discount_amount := clocks_discounts_count * clocks_discount * clocks_discount_threshold * clocks_price
  let clocks_revenue := clocks_revenue_full - clocks_discount_amount
  let glasses_discount_quantity := glasses / glasses_bundle
  let glasses_revenue := (glasses - glasses_discount_quantity) * glasses_price
  let vases_revenue := vases * vases_price
  let postcards_revenue := postcards * postcards_price
  let total_revenue_without_discounts := dolls_revenue + clocks_revenue_full + glasses_revenue + vases_revenue + postcards_revenue
  let total_revenue_with_discounts := dolls_revenue + clocks_revenue + glasses_revenue + vases_revenue + postcards_revenue
  let sales_tax := sales_tax_rate * total_revenue_with_discounts
  let profit := total_revenue_with_discounts - cost - sales_tax
  profit = -17.25 := by sorry

end stella_profit_loss_l1116_111646


namespace find_A_plus_B_l1116_111657

def isFourDigitNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def isOdd (n : ℕ) : Prop :=
  n % 2 = 1

def isMultipleOf5 (n : ℕ) : Prop :=
  n % 5 = 0

def countFourDigitOddNumbers : ℕ :=
  ((9 : ℕ) * 10 * 10 * 5)

def countFourDigitMultiplesOf5 : ℕ :=
  ((9 : ℕ) * 10 * 10 * 2)

theorem find_A_plus_B : countFourDigitOddNumbers + countFourDigitMultiplesOf5 = 6300 := by
  sorry

end find_A_plus_B_l1116_111657


namespace third_number_eq_l1116_111686

theorem third_number_eq :
  ∃ x : ℝ, (0.625 * 0.0729 * x) / (0.0017 * 0.025 * 8.1) = 382.5 ∧ x = 2.33075 := 
by
  sorry

end third_number_eq_l1116_111686


namespace variance_of_ξ_l1116_111627

noncomputable def probability_distribution (ξ : ℕ) : ℚ :=
  if ξ = 2 ∨ ξ = 4 ∨ ξ = 6 ∨ ξ = 8 ∨ ξ = 10 then 1/5 else 0

def expected_value (ξ_values : List ℕ) (prob : ℕ → ℚ) : ℚ :=
  ξ_values.map (λ ξ => ξ * prob ξ) |>.sum

def variance (ξ_values : List ℕ) (prob : ℕ → ℚ) (Eξ : ℚ) : ℚ :=
  ξ_values.map (λ ξ => prob ξ * (ξ - Eξ) ^ 2) |>.sum

theorem variance_of_ξ :
  let ξ_values := [2, 4, 6, 8, 10]
  let prob := probability_distribution
  let Eξ := expected_value ξ_values prob
  variance ξ_values prob Eξ = 8 :=
by
  -- Proof goes here
  sorry

end variance_of_ξ_l1116_111627


namespace six_digit_palindromes_count_l1116_111692

theorem six_digit_palindromes_count :
  (∃ (a b c d : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    0 ≤ d ∧ d ≤ 9) →
  9 * 10 * 10 * 10 = 9000 :=
by
  sorry

end six_digit_palindromes_count_l1116_111692


namespace volume_of_prism_l1116_111690

theorem volume_of_prism
  (a b c : ℝ)
  (h1 : a * b = 30)
  (h2 : a * c = 40)
  (h3 : b * c = 60) :
  a * b * c = 120 * Real.sqrt 5 := 
by
  sorry

end volume_of_prism_l1116_111690


namespace count_perfect_cubes_l1116_111624

theorem count_perfect_cubes (a b : ℤ) (h₁ : 100 < a) (h₂ : b < 1000) : 
  ∃ n m : ℤ, (n^3 > 100 ∧ m^3 < 1000) ∧ m - n + 1 = 5 := 
by
  -- The proof is omitted and replaced with sorry.
  sorry

end count_perfect_cubes_l1116_111624


namespace temperature_range_for_5_percent_deviation_l1116_111610

noncomputable def approx_formula (C : ℝ) : ℝ := 2 * C + 30
noncomputable def exact_formula (C : ℝ) : ℝ := (9/5 : ℝ) * C + 32
noncomputable def deviation (C : ℝ) : ℝ := approx_formula C - exact_formula C
noncomputable def percentage_deviation (C : ℝ) : ℝ := abs (deviation C / exact_formula C)

theorem temperature_range_for_5_percent_deviation :
  ∀ (C : ℝ), 1 + 11 / 29 ≤ C ∧ C ≤ 32 + 8 / 11 ↔ percentage_deviation C ≤ 0.05 := sorry

end temperature_range_for_5_percent_deviation_l1116_111610


namespace compute_factorial_expression_l1116_111626

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

theorem compute_factorial_expression :
  factorial 9 - factorial 8 - factorial 7 + factorial 6 = 318240 := by
  sorry

end compute_factorial_expression_l1116_111626


namespace line_passes_second_and_third_quadrants_l1116_111654

theorem line_passes_second_and_third_quadrants 
  (a b c p : ℝ)
  (h1 : a * b * c ≠ 0)
  (h2 : (a + b) / c = p)
  (h3 : (b + c) / a = p)
  (h4 : (c + a) / b = p) :
  ∀ (x y : ℝ), y = p * x + p → 
  ((x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) :=
sorry

end line_passes_second_and_third_quadrants_l1116_111654


namespace green_space_equation_l1116_111636

theorem green_space_equation (x : ℝ) (h_area : x * (x - 30) = 1000) :
  x * (x - 30) = 1000 := 
by
  exact h_area

end green_space_equation_l1116_111636


namespace expression_simplification_l1116_111647

open Real

theorem expression_simplification (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : 3*x + y / 3 ≠ 0) :
  (3*x + y/3)⁻¹ * ((3*x)⁻¹ + (y/3)⁻¹) = 1 / (3 * (x * y)) :=
by
  -- proof steps would go here
  sorry

end expression_simplification_l1116_111647


namespace dog_running_direction_undeterminable_l1116_111688

/-- Given the conditions:
 1. A dog is tied to a tree with a nylon cord of length 10 feet.
 2. The dog runs from one side of the tree to the opposite side with the cord fully extended.
 3. The dog runs approximately 30 feet.
 Prove that it is not possible to determine the specific starting direction of the dog.
-/
theorem dog_running_direction_undeterminable (r : ℝ) (full_length : r = 10) (distance_ran : ℝ) (approx_distance : distance_ran = 30) : (
  ∀ (d : ℝ), d < 2 * π * r → ¬∃ θ : ℝ, 0 ≤ θ ∧ θ < 2 * π ∧ (distance_ran = r * θ)
  ) :=
by
  sorry

end dog_running_direction_undeterminable_l1116_111688


namespace meal_cost_l1116_111673

variable (s c p : ℝ)

axiom cond1 : 5 * s + 8 * c + p = 5.00
axiom cond2 : 7 * s + 12 * c + p = 7.20
axiom cond3 : 4 * s + 6 * c + 2 * p = 6.00

theorem meal_cost : s + c + p = 1.90 :=
by
  sorry

end meal_cost_l1116_111673


namespace total_games_played_l1116_111638

-- Defining the conditions
def games_won : ℕ := 18
def games_lost : ℕ := games_won + 21

-- Problem statement
theorem total_games_played : games_won + games_lost = 57 := by
  sorry

end total_games_played_l1116_111638
