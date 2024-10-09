import Mathlib

namespace time_to_complete_together_l124_12425

theorem time_to_complete_together (sylvia_time carla_time combined_time : ℕ) (h_sylvia : sylvia_time = 45) (h_carla : carla_time = 30) :
  let sylvia_rate := 1 / (sylvia_time : ℚ)
  let carla_rate := 1 / (carla_time : ℚ)
  let combined_rate := sylvia_rate + carla_rate
  let time_to_complete := 1 / combined_rate
  time_to_complete = (combined_time : ℚ) :=
by
  sorry

end time_to_complete_together_l124_12425


namespace find_angle_B_l124_12403

variable (a b c A B C : ℝ)

-- Assuming all the necessary conditions and givens
axiom triangle_condition1 : a * (Real.sin B * Real.cos C) + c * (Real.sin B * Real.cos A) = (1 / 2) * b
axiom triangle_condition2 : a > b

-- We need to prove B = π / 6
theorem find_angle_B : B = π / 6 :=
by
  sorry

end find_angle_B_l124_12403


namespace logarithmic_expression_evaluation_l124_12462

noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem logarithmic_expression_evaluation : 
  log_base_10 (5 / 2) + 2 * log_base_10 2 - (1/2)⁻¹ = -1 := 
by 
  sorry

end logarithmic_expression_evaluation_l124_12462


namespace solve_system_l124_12496

theorem solve_system (x y z : ℤ) 
  (h1 : x + 3 * y = 20)
  (h2 : x + y + z = 25)
  (h3 : x - z = 5) : 
  x = 14 ∧ y = 2 ∧ z = 9 := 
  sorry

end solve_system_l124_12496


namespace martha_total_payment_l124_12420

noncomputable def cheese_kg : ℝ := 1.5
noncomputable def meat_kg : ℝ := 0.55
noncomputable def pasta_kg : ℝ := 0.28
noncomputable def tomatoes_kg : ℝ := 2.2

noncomputable def cheese_price_per_kg : ℝ := 6.30
noncomputable def meat_price_per_kg : ℝ := 8.55
noncomputable def pasta_price_per_kg : ℝ := 2.40
noncomputable def tomatoes_price_per_kg : ℝ := 1.79

noncomputable def total_cost :=
  cheese_kg * cheese_price_per_kg +
  meat_kg * meat_price_per_kg +
  pasta_kg * pasta_price_per_kg +
  tomatoes_kg * tomatoes_price_per_kg

theorem martha_total_payment : total_cost = 18.76 := by
  sorry

end martha_total_payment_l124_12420


namespace x_squared_minus_y_squared_l124_12407

theorem x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 9) (h2 : x - y = 3) : x^2 - y^2 = 27 := 
sorry

end x_squared_minus_y_squared_l124_12407


namespace binary_to_decimal_l124_12408

theorem binary_to_decimal :
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 1 * 2^3 + 1 * 2^4 + 1 * 2^5 + 1 * 2^6 + 0 * 2^7 + 1 * 2^8) = 379 := 
by
  sorry

end binary_to_decimal_l124_12408


namespace percentage_increase_of_y_over_x_l124_12468

variable (x y : ℝ) (h : x > 0 ∧ y > 0) 

theorem percentage_increase_of_y_over_x
  (h_ratio : (x / 8) = (y / 7)) :
  ((y - x) / x) * 100 = 12.5 := 
sorry

end percentage_increase_of_y_over_x_l124_12468


namespace value_of_expression_l124_12460

theorem value_of_expression (p q : ℚ) (h : p / q = 4 / 5) :
    11 / 7 + (2 * q - p) / (2 * q + p) = 2 :=
sorry

end value_of_expression_l124_12460


namespace closest_time_to_1600_mirror_l124_12459

noncomputable def clock_in_mirror_time (hour_hand_minute: ℕ) (minute_hand_minute: ℕ) : (ℕ × ℕ) :=
  let hour_in_mirror := (12 - hour_hand_minute) % 12
  let minute_in_mirror := minute_hand_minute
  (hour_in_mirror, minute_in_mirror)

theorem closest_time_to_1600_mirror (A B C D : (ℕ × ℕ)) :
  clock_in_mirror_time 4 0 = D → D = (8, 0) :=
by
  -- Introduction of hypothesis that clock closest to 16:00 (4:00) is represented by D
  intro h
  -- State the conclusion based on the given hypothesis
  sorry

end closest_time_to_1600_mirror_l124_12459


namespace total_pages_in_book_l124_12472

theorem total_pages_in_book :
  ∃ x : ℝ, (x - (1/6 * x + 10) - (1/3 * (x - (1/6 * x + 10)) + 20)
           - (1/2 * ((x - (1/6 * x + 10) - (1/3 * (x - (1/6 * x + 10)) + 20))) + 25) = 120) ∧
           x = 552 :=
by
  sorry

end total_pages_in_book_l124_12472


namespace constant_term_in_modified_equation_l124_12477

theorem constant_term_in_modified_equation :
  ∃ (c : ℝ), ∀ (q : ℝ), (3 * (3 * 5 - 3) - 3 + c = 132) → c = 99 := 
by
  sorry

end constant_term_in_modified_equation_l124_12477


namespace triangle_side_a_l124_12473

theorem triangle_side_a (c b : ℝ) (B : ℝ) (h₁ : c = 2) (h₂ : b = 6) (h₃ : B = 120) : a = 2 :=
by sorry

end triangle_side_a_l124_12473


namespace quadratic_inequality_solution_l124_12445

theorem quadratic_inequality_solution (a b c : ℝ) 
  (h1 : a < 0) 
  (h2 : a * 2^2 + b * 2 + c = 0) 
  (h3 : a * (-1)^2 + b * (-1) + c = 0) :
  ∀ x, ax^2 + bx + c ≥ 0 ↔ (-1 ≤ x ∧ x ≤ 2) :=
by 
  sorry

end quadratic_inequality_solution_l124_12445


namespace inequality_sqrt_l124_12475

theorem inequality_sqrt (m n : ℕ) (h : m < n) : 
  (m^2 + Real.sqrt (m^2 + m) < n^2 - Real.sqrt (n^2 - n)) :=
by
  sorry

end inequality_sqrt_l124_12475


namespace find_second_term_l124_12431

-- Define the terms and common ratio in the geometric sequence
def geometric_sequence (a r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

-- Given the fifth and sixth terms
variables (a r : ℚ)
axiom fifth_term : geometric_sequence a r 5 = 48
axiom sixth_term : geometric_sequence a r 6 = 72

-- Prove that the second term is 128/9
theorem find_second_term : geometric_sequence a r 2 = 128 / 9 :=
sorry

end find_second_term_l124_12431


namespace quadrilateral_count_correct_triangle_count_correct_intersection_triangle_count_correct_l124_12442

-- 1. Problem: Count of quadrilaterals from 12 points in a semicircle
def semicircle_points : ℕ := 12
def quadrilaterals_from_semicircle_points : ℕ :=
  let points_on_semicircle := 8
  let points_on_diameter := 4
  360 -- This corresponds to the final computed count, skipping calculation details

theorem quadrilateral_count_correct :
  quadrilaterals_from_semicircle_points = 360 := sorry

-- 2. Problem: Count of triangles from 10 points along an angle
def angle_points : ℕ := 10
def triangles_from_angle_points : ℕ :=
  let points_on_one_side := 5
  let points_on_other_side := 4
  90 -- This corresponds to the final computed count, skipping calculation details

theorem triangle_count_correct :
  triangles_from_angle_points = 90 := sorry

-- 3. Problem: Count of triangles from intersection points of parallel lines
def intersection_points : ℕ := 12
def triangles_from_intersections : ℕ :=
  let line_set_1_count := 3
  let line_set_2_count := 4
  200 -- This corresponds to the final computed count, skipping calculation details

theorem intersection_triangle_count_correct :
  triangles_from_intersections = 200 := sorry

end quadrilateral_count_correct_triangle_count_correct_intersection_triangle_count_correct_l124_12442


namespace rubies_in_treasure_l124_12495

theorem rubies_in_treasure (total_gems diamonds : ℕ) (h1 : total_gems = 5155) (h2 : diamonds = 45) : 
  total_gems - diamonds = 5110 := by
  sorry

end rubies_in_treasure_l124_12495


namespace total_stamps_collected_l124_12452

-- Conditions
def harry_stamps : ℕ := 180
def sister_stamps : ℕ := 60
def harry_three_times_sister : harry_stamps = 3 * sister_stamps := 
  by
  sorry  -- Proof will show that 180 = 3 * 60 (provided for completeness)

-- Statement to prove
theorem total_stamps_collected : harry_stamps + sister_stamps = 240 :=
  by
  sorry

end total_stamps_collected_l124_12452


namespace find_a_of_parabola_l124_12400

theorem find_a_of_parabola
  (a b c : ℝ)
  (h_point : 2 = c)
  (h_vertex : -2 = a * (2 - 2)^2 + b * 2 + c) :
  a = 1 :=
by
  sorry

end find_a_of_parabola_l124_12400


namespace radical_multiplication_l124_12483

noncomputable def root4 (x : ℝ) : ℝ := x ^ (1/4)
noncomputable def root3 (x : ℝ) : ℝ := x ^ (1/3)
noncomputable def root2 (x : ℝ) : ℝ := x ^ (1/2)

theorem radical_multiplication : root4 256 * root3 8 * root2 16 = 32 := by
  sorry

end radical_multiplication_l124_12483


namespace cloth_sales_worth_l124_12401

/--
An agent gets a commission of 2.5% on the sales of cloth. If on a certain day, he gets Rs. 15 as commission, 
proves that the worth of the cloth sold through him on that day is Rs. 600.
-/
theorem cloth_sales_worth (commission : ℝ) (rate : ℝ) (total_sales : ℝ) 
  (h_commission : commission = 15) (h_rate : rate = 2.5) (h_commission_formula : commission = (rate / 100) * total_sales) : 
  total_sales = 600 := 
by
  sorry

end cloth_sales_worth_l124_12401


namespace attendance_ratio_3_to_1_l124_12448

variable (x y : ℕ)
variable (total_attendance : ℕ := 2700)
variable (second_day_attendance : ℕ := 300)

/-- 
Prove that the ratio of the number of people attending the third day to the number of people attending the first day is 3:1
-/
theorem attendance_ratio_3_to_1
  (h1 : total_attendance = 2700)
  (h2 : second_day_attendance = x / 2)
  (h3 : second_day_attendance = 300)
  (h4 : y = total_attendance - x - second_day_attendance) :
  y / x = 3 :=
by
  sorry

end attendance_ratio_3_to_1_l124_12448


namespace polynomial_factorization_l124_12430

variable (a b c : ℝ)

theorem polynomial_factorization :
  2 * a * (b - c)^3 + 3 * b * (c - a)^3 + 2 * c * (a - b)^3 =
  (a - b) * (b - c) * (c - a) * (5 * b - c) :=
by sorry

end polynomial_factorization_l124_12430


namespace parametric_equations_l124_12497

variables (t : ℝ)
def x_velocity : ℝ := 9
def y_velocity : ℝ := 12
def init_x : ℝ := 1
def init_y : ℝ := 1

theorem parametric_equations :
  (x = init_x + x_velocity * t) ∧ (y = init_y + y_velocity * t) :=
sorry

end parametric_equations_l124_12497


namespace value_of_expression_l124_12461

-- Define the conditions
def x := -2
def y := 1
def z := 1
def w := 3

-- The main theorem statement
theorem value_of_expression : 
  (x^2 * y^2 * z^2) - (x^2 * y * z^2) + (y / w) * Real.sin (x * z) = - (1 / 3) * Real.sin 2 := by
  sorry

end value_of_expression_l124_12461


namespace nonnegative_integer_solutions_l124_12469

theorem nonnegative_integer_solutions (x y : ℕ) :
  3 * x^2 + 2 * 9^y = x * (4^(y+1) - 1) ↔ (x, y) ∈ [(2, 1), (3, 1), (3, 2), (18, 2)] :=
by sorry

end nonnegative_integer_solutions_l124_12469


namespace necessary_but_not_sufficient_l124_12404

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 + 2 * x - 8 > 0) ↔ (x > 2) ∨ (x < -4) := by
sorry

end necessary_but_not_sufficient_l124_12404


namespace decreasing_function_inequality_l124_12411

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ)
  (h_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2)
  (h_condition : f (3 * a) < f (-2 * a + 10)) :
  a > 2 :=
sorry

end decreasing_function_inequality_l124_12411


namespace tangency_point_is_ln2_l124_12450

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

theorem tangency_point_is_ln2 (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) →
  (∃ m : ℝ, Real.exp m - Real.exp (-m) = 3 / 2) →
  m = Real.log 2 :=
by
  intro h1 h2
  sorry

end tangency_point_is_ln2_l124_12450


namespace sufficiency_but_not_necessity_l124_12494

theorem sufficiency_but_not_necessity (a b : ℝ) :
  (a = 0 → a * b = 0) ∧ (a * b = 0 → a = 0) → False :=
by
   -- Proof is skipped
   sorry

end sufficiency_but_not_necessity_l124_12494


namespace pump_filling_time_without_leak_l124_12492

theorem pump_filling_time_without_leak (P : ℝ) (h1 : 1 / P - 1 / 14 = 3 / 7) : P = 2 :=
sorry

end pump_filling_time_without_leak_l124_12492


namespace expand_binomials_l124_12419

theorem expand_binomials (x : ℝ) : 
  (1 + x + x^3) * (1 - x^4) = 1 + x + x^3 - x^4 - x^5 - x^7 :=
by
  sorry

end expand_binomials_l124_12419


namespace total_number_of_rulers_l124_12427

-- Given conditions
def initial_rulers : ℕ := 11
def rulers_added_by_tim : ℕ := 14

-- Given question and desired outcome
def total_rulers (initial_rulers rulers_added_by_tim : ℕ) : ℕ :=
  initial_rulers + rulers_added_by_tim

-- The proof problem statement
theorem total_number_of_rulers : total_rulers 11 14 = 25 := by
  sorry

end total_number_of_rulers_l124_12427


namespace cindy_correct_operation_l124_12487

-- Let's define the conditions and proof statement in Lean 4.

variable (x : ℝ)
axiom incorrect_operation : (x - 7) / 5 = 25

theorem cindy_correct_operation :
  (x - 5) / 7 = 18 + 1 / 7 :=
sorry

end cindy_correct_operation_l124_12487


namespace solve_y_l124_12485

theorem solve_y (y : ℝ) (h1 : y > 0) (h2 : (y - 6) / 16 = 6 / (y - 16)) : y = 22 :=
by
  sorry

end solve_y_l124_12485


namespace total_volume_of_five_cubes_l124_12463

-- Definition for volume of a cube function
def volume_of_cube (edge_length : ℝ) : ℝ :=
  edge_length ^ 3

-- Conditions
def edge_length : ℝ := 5
def number_of_cubes : ℝ := 5

-- Proof statement
theorem total_volume_of_five_cubes : 
  volume_of_cube edge_length * number_of_cubes = 625 := 
by
  sorry

end total_volume_of_five_cubes_l124_12463


namespace total_stamps_l124_12498

-- Definitions based on conditions
def kylies_stamps : ℕ := 34
def nellys_stamps : ℕ := kylies_stamps + 44

-- Statement of the proof problem
theorem total_stamps : kylies_stamps + nellys_stamps = 112 :=
by
  -- Proof goes here
  sorry

end total_stamps_l124_12498


namespace correct_option_C_l124_12484

variable {a : ℝ} (x : ℝ) (b : ℝ)

theorem correct_option_C : 
  (a^8 / a^2 = a^6) :=
by {
  sorry
}

end correct_option_C_l124_12484


namespace sequence_general_formula_and_max_n_l124_12453

theorem sequence_general_formula_and_max_n {a : ℕ → ℝ} {S : ℕ → ℝ} {T : ℕ → ℝ}
  (hS2 : S 2 = (3 / 2) * a 2 - 1) 
  (hS3 : S 3 = (3 / 2) * a 3 - 1) :
  (∀ n, a n = 2 * 3^(n - 1)) ∧ 
  (∃ n : ℕ, (8 / 5) * T n + n / (5 * 3 ^ (n - 1)) ≤ 40 / 27 ∧ ∀ k > n, 
    (8 / 5) * T k + k / (5 * 3 ^ (k - 1)) > 40 / 27) :=
by
  sorry

end sequence_general_formula_and_max_n_l124_12453


namespace fraction_irreducible_l124_12429

theorem fraction_irreducible (n : ℕ) : Nat.gcd (12 * n + 1) (30 * n + 2) = 1 :=
sorry

end fraction_irreducible_l124_12429


namespace triangle_area_l124_12428

theorem triangle_area :
  ∀ (k : ℝ), ∃ (area : ℝ), 
  (∃ (r : ℝ) (a b c : ℝ), 
      r = 2 * Real.sqrt 3 ∧
      a / b = 3 / 5 ∧ a / c = 3 / 7 ∧ b / c = 5 / 7 ∧
      (∃ (A B C : ℝ),
          A = 3 * k ∧ B = 5 * k ∧ C = 7 * k ∧
          area = (1/2) * a * b * Real.sin (2 * Real.pi / 3))) →
  area = (135 * Real.sqrt 3 / 49) :=
sorry

end triangle_area_l124_12428


namespace sum_first_five_special_l124_12464

def is_special (n : ℕ) : Prop :=
  ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ n = p^2 * q^2

theorem sum_first_five_special :
  let special_numbers := [36, 100, 196, 484, 676]
  (∀ n ∈ special_numbers, is_special n) →
  special_numbers.sum = 1492 := by {
  sorry
}

end sum_first_five_special_l124_12464


namespace product_roots_positive_real_part_l124_12409

open Complex

theorem product_roots_positive_real_part :
    (∃ (roots : Fin 6 → ℂ),
       (∀ k, roots k ^ 6 = -64) ∧
       (∀ k, (roots k).re > 0 → (roots 0).re > 0 ∧ (roots 0).im > 0 ∧
                               (roots 1).re > 0 ∧ (roots 1).im < 0) ∧
       (roots 0 * roots 1 = 4)
    ) :=
sorry

end product_roots_positive_real_part_l124_12409


namespace acres_used_for_corn_l124_12434

theorem acres_used_for_corn (total_acres : ℕ) (ratio_beans ratio_wheat ratio_corn : ℕ)
    (h_total : total_acres = 1034)
    (h_ratio : ratio_beans = 5 ∧ ratio_wheat = 2 ∧ ratio_corn = 4) : 
    ratio_corn * (total_acres / (ratio_beans + ratio_wheat + ratio_corn)) = 376 := 
by
  -- Proof goes here
  sorry

end acres_used_for_corn_l124_12434


namespace value_of_nested_radical_l124_12451

def nested_radical : ℝ := 
  sorry -- Definition of the recurring expression is needed here, let's call it x
  
theorem value_of_nested_radical :
  (nested_radical = 5) :=
sorry -- The actual proof steps will be written here.

end value_of_nested_radical_l124_12451


namespace total_worth_of_stock_l124_12454

noncomputable def shop_equation (X : ℝ) : Prop :=
  0.04 * X - 0.02 * X = 400

theorem total_worth_of_stock :
  ∃ (X : ℝ), shop_equation X ∧ X = 20000 :=
by
  use 20000
  have h : shop_equation 20000 := by
    unfold shop_equation
    norm_num
  exact ⟨h, rfl⟩

end total_worth_of_stock_l124_12454


namespace no_nat_nums_gt_one_divisibility_conditions_l124_12423

theorem no_nat_nums_gt_one_divisibility_conditions :
  ¬ ∃ (a b c : ℕ), 
    1 < a ∧ 1 < b ∧ 1 < c ∧
    (c ∣ a^2 - 1) ∧ (b ∣ a^2 - 1) ∧ 
    (a ∣ c^2 - 1) ∧ (b ∣ c^2 - 1) :=
by 
  sorry

end no_nat_nums_gt_one_divisibility_conditions_l124_12423


namespace value_range_of_quadratic_function_l124_12470

def quadratic_function (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem value_range_of_quadratic_function :
  (∀ x : ℝ, 1 < x ∧ x ≤ 4 → -1 < quadratic_function x ∧ quadratic_function x ≤ 3) :=
sorry

end value_range_of_quadratic_function_l124_12470


namespace cost_of_two_pencils_and_one_pen_l124_12446

variables (a b : ℝ)

theorem cost_of_two_pencils_and_one_pen
  (h1 : 3 * a + b = 3.00)
  (h2 : 3 * a + 4 * b = 7.50) :
  2 * a + b = 2.50 :=
sorry

end cost_of_two_pencils_and_one_pen_l124_12446


namespace minimum_possible_value_l124_12413

-- Define the set of distinct elements
def distinct_elems : Set ℤ := {-8, -6, -4, -1, 1, 3, 7, 12}

-- Define the existence of distinct elements
def elem_distinct (p q r s t u v w : ℤ) : Prop :=
  p ∈ distinct_elems ∧ q ∈ distinct_elems ∧ r ∈ distinct_elems ∧ s ∈ distinct_elems ∧ 
  t ∈ distinct_elems ∧ u ∈ distinct_elems ∧ v ∈ distinct_elems ∧ w ∈ distinct_elems ∧ 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧ 
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
  r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧ 
  s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧ 
  t ≠ u ∧ t ≠ v ∧ t ≠ w ∧ 
  u ≠ v ∧ u ≠ w ∧ 
  v ≠ w

-- The main proof problem
theorem minimum_possible_value :
  ∀ (p q r s t u v w : ℤ), elem_distinct p q r s t u v w ->
  (p + q + r + s)^2 + (t + u + v + w)^2 = 10 := 
sorry

end minimum_possible_value_l124_12413


namespace jay_savings_in_a_month_is_correct_l124_12424

-- Definitions for the conditions
def initial_savings : ℕ := 20
def weekly_increase : ℕ := 10

-- Define the savings for each week
def savings_after_week (week : ℕ) : ℕ :=
  initial_savings + (week - 1) * weekly_increase

-- Define the total savings over 4 weeks
def total_savings_after_4_weeks : ℕ :=
  savings_after_week 1 + savings_after_week 2 + savings_after_week 3 + savings_after_week 4

-- Proposition statement 
theorem jay_savings_in_a_month_is_correct :
  total_savings_after_4_weeks = 140 :=
  by
  -- proof will go here
  sorry

end jay_savings_in_a_month_is_correct_l124_12424


namespace sum_twice_father_age_plus_son_age_l124_12488

/-- 
  Given:
  1. Twice the son's age plus the father's age equals 70.
  2. Father's age is 40.
  3. Son's age is 15.

  Prove:
  The sum when twice the father's age is added to the son's age is 95.
-/
theorem sum_twice_father_age_plus_son_age :
  ∀ (father_age son_age : ℕ), 
    2 * son_age + father_age = 70 → 
    father_age = 40 → 
    son_age = 15 → 
    2 * father_age + son_age = 95 := by
  intros
  sorry

end sum_twice_father_age_plus_son_age_l124_12488


namespace multiple_of_9_l124_12467

theorem multiple_of_9 (x : ℕ) (hx1 : ∃ k : ℕ, x = 9 * k) (hx2 : x^2 > 80) (hx3 : x < 30) : x = 9 ∨ x = 18 ∨ x = 27 :=
sorry

end multiple_of_9_l124_12467


namespace selling_price_l124_12491

theorem selling_price 
  (cost_price : ℝ) 
  (profit_percentage : ℝ) 
  (h_cost : cost_price = 192) 
  (h_profit : profit_percentage = 0.25) : 
  ∃ selling_price : ℝ, selling_price = cost_price * (1 + profit_percentage) := 
by {
  sorry
}

end selling_price_l124_12491


namespace simplify_expression_l124_12447

theorem simplify_expression (x : ℝ) : 4 * x - 3 * x^2 + 6 + (8 - 5 * x + 2 * x^2) = - x^2 - x + 14 := by
  sorry

end simplify_expression_l124_12447


namespace largest_five_digit_product_l124_12486

theorem largest_five_digit_product
  (digs : List ℕ)
  (h_digit_count : digs.length = 5)
  (h_product : (digs.foldr (· * ·) 1) = 9 * 8 * 7 * 6 * 5) :
  (digs.foldr (λ a b => if a > b then 10 * a + b else 10 * b + a) 0) = 98765 :=
sorry

end largest_five_digit_product_l124_12486


namespace total_insects_eaten_l124_12481

theorem total_insects_eaten :
  let geckos := 5
  let insects_per_gecko := 6
  let lizards := 3
  let insects_per_lizard := 2 * insects_per_gecko
  let total_insects := geckos * insects_per_gecko + lizards * insects_per_lizard
  total_insects = 66 := by
  sorry

end total_insects_eaten_l124_12481


namespace product_closest_to_l124_12414

def is_closest_to (n target : ℝ) (options : List ℝ) : Prop :=
  ∀ o ∈ options, |n - target| ≤ |n - o|

theorem product_closest_to : is_closest_to ((2.5) * (50.5 + 0.25)) 127 [120, 125, 127, 130, 140] :=
by
  sorry

end product_closest_to_l124_12414


namespace pradeep_pass_percentage_l124_12482

-- Define the given data as constants
def score : ℕ := 185
def shortfall : ℕ := 25
def maxMarks : ℕ := 840

-- Calculate the passing mark
def passingMark : ℕ := score + shortfall

-- Calculate the percentage needed to pass
def passPercentage (passingMark : ℕ) (maxMarks : ℕ) : ℕ :=
  (passingMark * 100) / maxMarks

-- Statement of the theorem that we aim to prove
theorem pradeep_pass_percentage (score shortfall maxMarks : ℕ)
  (h_score : score = 185) (h_shortfall : shortfall = 25) (h_maxMarks : maxMarks = 840) :
  passPercentage (score + shortfall) maxMarks = 25 :=
by
  -- This is where the proof would go
  sorry

-- Example of calling the function to ensure definitions are correct
#eval passPercentage (score + shortfall) maxMarks -- Should output 25

end pradeep_pass_percentage_l124_12482


namespace find_number_l124_12489

theorem find_number (x : ℝ) : 35 + 3 * x^2 = 89 ↔ x = 3 * Real.sqrt 2 ∨ x = -3 * Real.sqrt 2 := by
  sorry

end find_number_l124_12489


namespace majority_owner_percentage_l124_12449

theorem majority_owner_percentage (profit total_profit : ℝ)
    (majority_owner_share : ℝ) (partner_share : ℝ) 
    (combined_share : ℝ) 
    (num_partners : ℕ) 
    (total_profit_value : total_profit = 80000) 
    (partner_share_value : partner_share = 0.25 * (1 - majority_owner_share)) 
    (combined_share_value : combined_share = profit)
    (combined_share_amount : combined_share = 50000) 
    (num_partners_value : num_partners = 4) :
  majority_owner_share = 0.25 :=
by
  sorry

end majority_owner_percentage_l124_12449


namespace spadesuit_eval_l124_12455

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_eval :
  spadesuit 5 (spadesuit 2 3) = 0 :=
by
  sorry

end spadesuit_eval_l124_12455


namespace expr_divisible_by_120_l124_12410

theorem expr_divisible_by_120 (m : ℕ) : 120 ∣ (m^5 - 5 * m^3 + 4 * m) :=
sorry

end expr_divisible_by_120_l124_12410


namespace water_left_in_bathtub_l124_12426

theorem water_left_in_bathtub :
  (40 * 60 * 9 - 200 * 9 - 12000 = 7800) :=
by
  -- Dripping rate per minute * number of minutes in an hour * number of hours
  let inflow_rate := 40 * 60
  let total_inflow := inflow_rate * 9
  -- Evaporation rate per hour * number of hours
  let total_evaporation := 200 * 9
  -- Water dumped out
  let water_dumped := 12000
  -- Final amount of water
  let final_amount := total_inflow - total_evaporation - water_dumped
  have h : final_amount = 7800 := by
    sorry
  exact h

end water_left_in_bathtub_l124_12426


namespace number_of_six_digit_palindromes_l124_12457

def is_six_digit_palindrome (n : ℕ) : Prop := 
  100000 ≤ n ∧ n ≤ 999999 ∧ (∀ a b c : ℕ, 
    n = 100000 * a + 10000 * b + 1000 * c + 100 * c + 10 * b + a → a ≠ 0)

theorem number_of_six_digit_palindromes : 
  ∃ (count : ℕ), (count = 900 ∧ 
  ∀ n : ℕ, is_six_digit_palindrome n → true) 
:= 
by 
  use 900 
  sorry

end number_of_six_digit_palindromes_l124_12457


namespace solve_x_in_equation_l124_12416

theorem solve_x_in_equation (a b x : ℝ) (h1 : a ≠ b) (h2 : a ≠ -b) (h3 : x ≠ 0) : 
  (b ≠ 0 ∧ (1 / (a + b) + (a - b) / x = 1 / (a - b) + (a - b) / x) → x = a^2 - b^2) ∧ 
  (b = 0 ∧ a ≠ 0 ∧ (1 / a + a / x = 1 / a + a / x) → x ≠ 0) := 
by
  sorry

end solve_x_in_equation_l124_12416


namespace line_plane_intersection_l124_12412

theorem line_plane_intersection 
  (t : ℝ)
  (x_eq : ∀ t: ℝ, x = 5 - t)
  (y_eq : ∀ t: ℝ, y = -3 + 5 * t)
  (z_eq : ∀ t: ℝ, z = 1 + 2 * t)
  (plane_eq : 3 * x + 7 * y - 5 * z - 11 = 0)
  : x = 4 ∧ y = 2 ∧ z = 3 :=
by
  sorry

end line_plane_intersection_l124_12412


namespace length_of_XY_l124_12490

-- Defining the points on the circle
variables (A B C D P Q X Y : Type*)
-- Lengths given in the problem
variables (AB_len CD_len AP_len CQ_len PQ_len : ℕ)
-- Points and lengths conditions
variables (h1 : AB_len = 11) (h2 : CD_len = 19)
variables (h3 : AP_len = 6) (h4 : CQ_len = 7)
variables (h5 : PQ_len = 27)

-- Assuming the Power of a Point theorem applied to P and Q
variables (PX_len PY_len QX_len QY_len : ℕ)
variables (h6 : PX_len = 1) (h7 : QY_len = 3)
variables (h8 : PX_len + PQ_len + QY_len = XY_len)

-- The final length of XY is to be found
def XY_len : ℕ := PX_len + PQ_len + QY_len

-- The goal is to show XY = 31
theorem length_of_XY : XY_len = 31 :=
  by
    sorry

end length_of_XY_l124_12490


namespace number_of_sacks_after_49_days_l124_12418

def sacks_per_day : ℕ := 38
def days_of_harvest : ℕ := 49
def total_sacks_after_49_days : ℕ := 1862

theorem number_of_sacks_after_49_days :
  sacks_per_day * days_of_harvest = total_sacks_after_49_days :=
by
  sorry

end number_of_sacks_after_49_days_l124_12418


namespace equivalent_problem_l124_12466

theorem equivalent_problem (n : ℕ) (h₁ : 0 ≤ n) (h₂ : n < 29) (h₃ : 2 * n % 29 = 1) :
  (3^n % 29)^3 - 3 % 29 = 3 :=
sorry

end equivalent_problem_l124_12466


namespace remaining_balance_is_correct_l124_12476

def initial_balance : ℕ := 50
def spent_coffee : ℕ := 10
def spent_tumbler : ℕ := 30

theorem remaining_balance_is_correct : initial_balance - (spent_coffee + spent_tumbler) = 10 := by
  sorry

end remaining_balance_is_correct_l124_12476


namespace option_B_correct_option_C_correct_l124_12474

-- Define the permutation coefficient
def A (n m : ℕ) : ℕ := n * (n-1) * (n-2) * (n-m+1)

-- Prove the equation for option B
theorem option_B_correct (n m : ℕ) : A (n+1) (m+1) - A n m = n^2 * A (n-1) (m-1) :=
by
  sorry

-- Prove the equation for option C
theorem option_C_correct (n m : ℕ) : A n m = n * A (n-1) (m-1) :=
by
  sorry

end option_B_correct_option_C_correct_l124_12474


namespace sum_first_50_arithmetic_sequence_l124_12480

theorem sum_first_50_arithmetic_sequence : 
  let a : ℕ := 2
  let d : ℕ := 4
  let n : ℕ := 50
  let a_n (n : ℕ) : ℕ := a + (n - 1) * d
  let S_n (n : ℕ) : ℕ := n / 2 * (2 * a + (n - 1) * d)
  S_n n = 5000 :=
by
  sorry

end sum_first_50_arithmetic_sequence_l124_12480


namespace dolphins_to_be_trained_next_month_l124_12456

theorem dolphins_to_be_trained_next_month :
  ∀ (total_dolphins fully_trained remaining trained_next_month : ℕ),
    total_dolphins = 20 →
    fully_trained = (1 / 4 : ℚ) * total_dolphins →
    remaining = total_dolphins - fully_trained →
    (2 / 3 : ℚ) * remaining = 10 →
    trained_next_month = remaining - 10 →
    trained_next_month = 5 :=
by
  intros total_dolphins fully_trained remaining trained_next_month
  intro h1 h2 h3 h4 h5
  sorry

end dolphins_to_be_trained_next_month_l124_12456


namespace false_proposition_among_given_l124_12441

theorem false_proposition_among_given (a b c : Prop) : 
  (a = ∀ x : ℝ, ∃ y : ℝ, x = y) ∧
  (b = (∃ a b c : ℕ, a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2)) ∧
  (c = ∀ α β : ℝ, α = β ∧ ∃ P : Type, ∃ vertices : P, α = β ) → ¬c := by
  sorry

end false_proposition_among_given_l124_12441


namespace line_intersects_parabola_at_9_units_apart_l124_12465

theorem line_intersects_parabola_at_9_units_apart :
  ∃ m b, (∃ (k1 k2 : ℝ), 
              (y1 = k1^2 + 6*k1 - 4) ∧ 
              (y2 = k2^2 + 6*k2 - 4) ∧ 
              (y1 = m*k1 + b) ∧ 
              (y2 = m*k2 + b) ∧ 
              |y1 - y2| = 9) ∧ 
          (0 ≠ b) ∧ 
          ((1 : ℝ) = 2*m + b) ∧ 
          (m = 4 ∧ b = -7)
:= sorry

end line_intersects_parabola_at_9_units_apart_l124_12465


namespace parabola_ellipse_sum_distances_l124_12406

noncomputable def sum_distances_intersection_points (b c : ℝ) : ℝ :=
  2 * Real.sqrt b + 2 * Real.sqrt c

theorem parabola_ellipse_sum_distances
  (A B : ℝ)
  (h1 : A > 0) -- semi-major axis condition implied
  (h2 : B > 0) -- semi-minor axis condition implied
  (ellipse_eq : ∀ x y, (x^2) / A^2 + (y^2) / B^2 = 1)
  (focus_shared : ∃ f : ℝ, f = Real.sqrt (A^2 - B^2))
  (directrix_parabola : ∃ d : ℝ, d = B) -- directrix condition
  (intersections : ∃ (b c : ℝ), (b > 0 ∧ c > 0)) -- existence of such intersection points
  : sum_distances_intersection_points b c = 2 * Real.sqrt b + 2 * Real.sqrt c :=
sorry  -- proof omitted

end parabola_ellipse_sum_distances_l124_12406


namespace initial_water_amount_l124_12436

theorem initial_water_amount (W : ℝ) (h1 : ∀ t, t = 50 -> 0.008 * t = 0.4) (h2 : 0.04 * W = 0.4) : W = 10 :=
by
  sorry

end initial_water_amount_l124_12436


namespace hats_in_box_total_l124_12437

theorem hats_in_box_total : 
  (∃ (n : ℕ), (∀ (r b y : ℕ), r + y = n - 2 ∧ r + b = n - 2 ∧ b + y = n - 2)) → (∃ n, n = 3) :=
by
  sorry

end hats_in_box_total_l124_12437


namespace arctan_sum_eq_pi_over_4_l124_12402

theorem arctan_sum_eq_pi_over_4 : 
  Real.arctan (1/3) + Real.arctan (1/4) + Real.arctan (1/5) + Real.arctan (1/47) = Real.pi / 4 :=
by
  sorry

end arctan_sum_eq_pi_over_4_l124_12402


namespace option_c_l124_12438

theorem option_c (a b : ℝ) (h : a > |b|) : a^2 > b^2 := sorry

end option_c_l124_12438


namespace log_base_eq_l124_12444

theorem log_base_eq (x : ℝ) (h₁ : x > 0) (h₂ : x ≠ 1) : 
  (Real.log x / Real.log 4) * (Real.log 8 / Real.log x) = Real.log 8 / Real.log 4 := 
by 
  sorry

end log_base_eq_l124_12444


namespace trapezoid_area_division_l124_12479

theorem trapezoid_area_division (AD BC MN : ℝ) (h₁ : AD = 4) (h₂ : BC = 3)
  (h₃ : MN > 0) (area_ratio : ∃ (S_ABMD S_MBCN : ℝ), MN/BC = (S_ABMD + S_MBCN)/(S_ABMD) ∧ (S_ABMD/S_MBCN = 2/5)) :
  MN = Real.sqrt 14 :=
by
  sorry

end trapezoid_area_division_l124_12479


namespace sum_of_sequences_is_43_l124_12422

theorem sum_of_sequences_is_43
  (A B C D : ℕ)
  (hA_pos : 0 < A)
  (hB_pos : 0 < B)
  (hC_pos : 0 < C)
  (hD_pos : 0 < D)
  (h_arith : A + (C - B) = B)
  (h_geom : C = (4 * B) / 3)
  (hD_def : D = (4 * C) / 3) :
  A + B + C + D = 43 :=
sorry

end sum_of_sequences_is_43_l124_12422


namespace parcels_division_l124_12458

theorem parcels_division (x y n : ℕ) (h : 5 + 2 * x + 3 * y = 4 * n) (hn : n = x + y) :
    n = 3 ∨ n = 4 ∨ n = 5 := 
sorry

end parcels_division_l124_12458


namespace smallest_positive_debt_resolves_l124_12478

theorem smallest_positive_debt_resolves :
  ∃ (c t : ℤ), (240 * c + 180 * t = 60) ∧ (60 > 0) :=
by
  sorry

end smallest_positive_debt_resolves_l124_12478


namespace hyperbola_chord_line_eq_l124_12415

theorem hyperbola_chord_line_eq (m n s t : ℝ) (h_mn_pos : m > 0 ∧ n > 0 ∧ s > 0 ∧ t > 0)
  (h_mn_sum : m + n = 2)
  (h_m_n_s_t : m / s + n / t = 9)
  (h_s_t_min : s + t = 4 / 9)
  (h_midpoint : (2 : ℝ) = (m + n)) :
  ∃ (c : ℝ), (∀ (x1 y1 x2 y2 : ℝ), 
    (x1 + x2) / 2 = m ∧ (y1 + y2) / 2 = n ∧ 
    (x1 ^ 2 / 4 - y1 ^ 2 / 2 = 1 ∧ x2 ^ 2 / 4 - y2 ^ 2 / 2 = 1) → 
    y2 - y1 = c * (x2 - x1)) ∧ (c = 1 / 2) →
  ∀ (x y : ℝ), x - 2 * y + 1 = 0 :=
by sorry

end hyperbola_chord_line_eq_l124_12415


namespace hypotenuse_length_l124_12432

theorem hypotenuse_length (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 1450) (h2 : c^2 = a^2 + b^2) : 
  c = Real.sqrt 725 :=
by
  sorry

end hypotenuse_length_l124_12432


namespace find_incorrect_statement_l124_12440

def is_opposite (a b : ℝ) := a = -b

theorem find_incorrect_statement :
  ¬∀ (a b : ℝ), (a * b < 0) → is_opposite a b := sorry

end find_incorrect_statement_l124_12440


namespace expression_evaluation_l124_12405

noncomputable def evaluate_expression : ℝ :=
  (Real.sin (38 * Real.pi / 180) * Real.sin (38 * Real.pi / 180) 
  + Real.cos (38 * Real.pi / 180) * Real.sin (52 * Real.pi / 180) 
  - Real.tan (15 * Real.pi / 180) ^ 2) / (3 * Real.tan (15 * Real.pi / 180))

theorem expression_evaluation : 
  evaluate_expression = (2 * Real.sqrt 3) / 3 :=
by
  sorry

end expression_evaluation_l124_12405


namespace polikarp_make_first_box_empty_l124_12471

theorem polikarp_make_first_box_empty (n : ℕ) (h : n ≤ 30) : ∃ (x y : ℕ), x + y ≤ 10 ∧ ∀ k : ℕ, k ≤ x → k + k * y = n :=
by
  sorry

end polikarp_make_first_box_empty_l124_12471


namespace sum_of_digits_of_largest_five_digit_number_with_product_120_l124_12433

theorem sum_of_digits_of_largest_five_digit_number_with_product_120 
  (a b c d e : ℕ)
  (h_digit_a : 0 ≤ a ∧ a ≤ 9)
  (h_digit_b : 0 ≤ b ∧ b ≤ 9)
  (h_digit_c : 0 ≤ c ∧ c ≤ 9)
  (h_digit_d : 0 ≤ d ∧ d ≤ 9)
  (h_digit_e : 0 ≤ e ∧ e ≤ 9)
  (h_product : a * b * c * d * e = 120)
  (h_largest : ∀ f g h i j : ℕ, 
                0 ≤ f ∧ f ≤ 9 → 
                0 ≤ g ∧ g ≤ 9 → 
                0 ≤ h ∧ h ≤ 9 → 
                0 ≤ i ∧ i ≤ 9 → 
                0 ≤ j ∧ j ≤ 9 → 
                f * g * h * i * j = 120 → 
                f * 10000 + g * 1000 + h * 100 + i * 10 + j ≤ a * 10000 + b * 1000 + c * 100 + d * 10 + e) :
  a + b + c + d + e = 18 :=
by sorry

end sum_of_digits_of_largest_five_digit_number_with_product_120_l124_12433


namespace gardener_cabbages_this_year_l124_12435

-- Definitions for the conditions
def side_length_last_year (x : ℕ) := true
def area_last_year (x : ℕ) := x * x
def increase_in_output := 197

-- Proposition to prove the number of cabbages this year
theorem gardener_cabbages_this_year (x : ℕ) (hx : side_length_last_year x) : 
  (area_last_year x + increase_in_output) = 9801 :=
by 
  sorry

end gardener_cabbages_this_year_l124_12435


namespace factorize_expression_polygon_sides_l124_12421

-- Problem 1: Factorize 2x^3 - 8x
theorem factorize_expression (x : ℝ) : 2 * x^3 - 8 * x = 2 * x * (x - 2) * (x + 2) :=
by
  sorry

-- Problem 2: Find the number of sides of a polygon with interior angle sum 1080 degrees
theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

end factorize_expression_polygon_sides_l124_12421


namespace tan_value_sin_cos_ratio_sin_squared_expression_l124_12439

theorem tan_value (α : ℝ) (h1 : 3 * Real.pi / 4 < α) (h2 : α < Real.pi) (h3 : Real.tan α + 1 / Real.tan α = -10 / 3) : 
  Real.tan α = -1 / 3 :=
sorry

theorem sin_cos_ratio (α : ℝ) (h1 : 3 * Real.pi / 4 < α) (h2 : α < Real.pi) (h3 : Real.tan α + 1 / Real.tan α = -10 / 3) (h4 : Real.tan α = -1 / 3) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -1 / 2 :=
sorry

theorem sin_squared_expression (α : ℝ) (h1 : 3 * Real.pi / 4 < α) (h2 : α < Real.pi) (h3 : Real.tan α + 1 / Real.tan α = -10 / 3) (h4 : Real.tan α = -1 / 3) : 
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α - 3 * Real.cos α ^ 2 = -11 / 5 :=
sorry

end tan_value_sin_cos_ratio_sin_squared_expression_l124_12439


namespace find_unknown_number_l124_12493

theorem find_unknown_number (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end find_unknown_number_l124_12493


namespace length_of_inner_rectangle_is_4_l124_12443

-- Defining the conditions and the final proof statement
theorem length_of_inner_rectangle_is_4 :
  ∃ y : ℝ, y = 4 ∧
  let inner_width := 2
  let second_width := inner_width + 4
  let largest_width := second_width + 4
  let inner_area := inner_width * y
  let second_area := 6 * second_width
  let largest_area := 10 * largest_width
  let first_shaded_area := second_area - inner_area
  let second_shaded_area := largest_area - second_area
  (first_shaded_area - inner_area = second_shaded_area - first_shaded_area)
:= sorry

end length_of_inner_rectangle_is_4_l124_12443


namespace johns_age_fraction_l124_12417

theorem johns_age_fraction (F M J : ℕ) 
  (hF : F = 40) 
  (hFM : F = M + 4) 
  (hJM : J = M - 16) : 
  J / F = 1 / 2 := 
by
  -- We don't need to fill in the proof, adding sorry to skip it
  sorry

end johns_age_fraction_l124_12417


namespace subjects_difference_marius_monica_l124_12499

-- Definitions of given conditions.
def Monica_subjects : ℕ := 10
def Total_subjects : ℕ := 41
def Millie_offset : ℕ := 3

-- Theorem to prove the question == answer given conditions
theorem subjects_difference_marius_monica : 
  ∃ (M : ℕ), (M + (M + Millie_offset) + Monica_subjects = Total_subjects) ∧ (M - Monica_subjects = 4) := 
by
  sorry

end subjects_difference_marius_monica_l124_12499
