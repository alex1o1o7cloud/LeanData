import Mathlib

namespace NUMINAMATH_GPT_interest_second_month_l1606_160644

theorem interest_second_month {P r n : ℝ} (hP : P = 200) (hr : r = 0.10) (hn : n = 12) :
  (P * (1 + r / n) ^ (n * (1/12)) - P) * r / n = 1.68 :=
by
  sorry

end NUMINAMATH_GPT_interest_second_month_l1606_160644


namespace NUMINAMATH_GPT_pen_cost_difference_l1606_160622

theorem pen_cost_difference :
  ∀ (P : ℕ), (P + 2 = 13) → (P - 2 = 9) :=
by
  intro P
  intro h
  sorry

end NUMINAMATH_GPT_pen_cost_difference_l1606_160622


namespace NUMINAMATH_GPT_right_triangle_min_hypotenuse_right_triangle_min_hypotenuse_achieved_l1606_160687

theorem right_triangle_min_hypotenuse (a b c : ℝ) (h_right : (a^2 + b^2 = c^2)) (h_perimeter : (a + b + c = 8)) : c ≥ 4 * Real.sqrt 2 := by
  sorry

theorem right_triangle_min_hypotenuse_achieved (a b c : ℝ) (h_right : (a^2 + b^2 = c^2)) (h_perimeter : (a + b + c = 8)) (h_isosceles : a = b) : c = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_right_triangle_min_hypotenuse_right_triangle_min_hypotenuse_achieved_l1606_160687


namespace NUMINAMATH_GPT_fraction_irreducible_l1606_160677

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
by {
    sorry
}

end NUMINAMATH_GPT_fraction_irreducible_l1606_160677


namespace NUMINAMATH_GPT_ratio_female_to_male_l1606_160659

theorem ratio_female_to_male (total_members : ℕ) (female_members : ℕ) (male_members : ℕ) 
  (h1 : total_members = 18) (h2 : female_members = 12) (h3 : male_members = total_members - female_members) : 
  (female_members : ℚ) / (male_members : ℚ) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_female_to_male_l1606_160659


namespace NUMINAMATH_GPT_arithmetic_sequence_a2_a8_l1606_160649

variable {a : ℕ → ℝ}

-- given condition
axiom h1 : a 4 + a 5 + a 6 = 450

-- problem statement
theorem arithmetic_sequence_a2_a8 : a 2 + a 8 = 300 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a2_a8_l1606_160649


namespace NUMINAMATH_GPT_probability_N_lt_L_is_zero_l1606_160641

variable (M N L O : ℝ)
variable (hNM : N < M)
variable (hLO : L > O)

theorem probability_N_lt_L_is_zero : 
  (∃ (permutations : List (ℝ → ℝ)), 
  (∀ perm : ℝ → ℝ, perm ∈ permutations → N < M ∧ L > O) ∧ 
  ∀ perm : ℝ → ℝ, N > L) → false :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_N_lt_L_is_zero_l1606_160641


namespace NUMINAMATH_GPT_problem_statement_l1606_160638

theorem problem_statement (f : ℝ → ℝ) (hf_odd : ∀ x, f (-x) = - f x)
  (hf_deriv : ∀ x < 0, 2 * f x + x * deriv f x < 0) :
  f 1 < 2016 * f (Real.sqrt 2016) ∧ 2016 * f (Real.sqrt 2016) < 2017 * f (Real.sqrt 2017) := 
  sorry

end NUMINAMATH_GPT_problem_statement_l1606_160638


namespace NUMINAMATH_GPT_perfect_square_expression_l1606_160610

theorem perfect_square_expression (p : ℝ) : 
  (12.86^2 + 12.86 * p + 0.14^2) = (12.86 + 0.14)^2 → p = 0.28 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_expression_l1606_160610


namespace NUMINAMATH_GPT_nth_equation_pattern_l1606_160611

theorem nth_equation_pattern (n : ℕ) : 
  (List.range' n (2 * n - 1)).sum = (2 * n - 1) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_nth_equation_pattern_l1606_160611


namespace NUMINAMATH_GPT_abs_neg_sub_three_eq_zero_l1606_160672

theorem abs_neg_sub_three_eq_zero : |(-3 : ℤ)| - 3 = 0 :=
by sorry

end NUMINAMATH_GPT_abs_neg_sub_three_eq_zero_l1606_160672


namespace NUMINAMATH_GPT_expand_polynomial_l1606_160656

theorem expand_polynomial : 
  (∀ (x : ℝ), (5 * x^3 + 7) * (3 * x + 4) = 15 * x^4 + 20 * x^3 + 21 * x + 28) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_expand_polynomial_l1606_160656


namespace NUMINAMATH_GPT_option_d_is_quadratic_equation_l1606_160616

theorem option_d_is_quadratic_equation (x y : ℝ) : 
  (x^2 + x - 4 = 0) ↔ (x^2 + x = 4) := 
by
  sorry

end NUMINAMATH_GPT_option_d_is_quadratic_equation_l1606_160616


namespace NUMINAMATH_GPT_decimal_to_fraction_l1606_160618

theorem decimal_to_fraction :
  (368 / 100 : ℚ) = (92 / 25 : ℚ) := by
  sorry

end NUMINAMATH_GPT_decimal_to_fraction_l1606_160618


namespace NUMINAMATH_GPT_sum_of_numbers_l1606_160632

theorem sum_of_numbers :
  15.58 + 21.32 + 642.51 + 51.51 = 730.92 := 
  by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l1606_160632


namespace NUMINAMATH_GPT_range_of_a_l1606_160673

theorem range_of_a (a : ℝ) :
  (∀ p : ℝ × ℝ, (p.1 - 2 * a) ^ 2 + (p.2 - (a + 3)) ^ 2 = 4 → p.1 ^ 2 + p.2 ^ 2 = 1) →
  -1 < a ∧ a < 0 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1606_160673


namespace NUMINAMATH_GPT_remaining_movies_l1606_160645

-- Definitions based on the problem's conditions
def total_movies : ℕ := 8
def watched_movies : ℕ := 4

-- Theorem statement to prove that you still have 4 movies left to watch
theorem remaining_movies : total_movies - watched_movies = 4 :=
by
  sorry

end NUMINAMATH_GPT_remaining_movies_l1606_160645


namespace NUMINAMATH_GPT_range_of_m_l1606_160612

theorem range_of_m (f : ℝ → ℝ) (h_decreasing : ∀ x y, x < y → f y ≤ f x) (m : ℝ) (h : f (m-1) > f (2*m-1)) : 0 < m :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1606_160612


namespace NUMINAMATH_GPT_identify_first_brother_l1606_160667

-- Definitions for conditions
inductive Brother
| Trulya : Brother
| Falsa : Brother

-- Extracting conditions into Lean 4 statements
def first_brother_says : String := "Both cards are of the purplish suit."
def second_brother_says : String := "This is not true!"

axiom trulya_always_truthful : ∀ (b : Brother) (statement : String), b = Brother.Trulya ↔ (statement = first_brother_says ∨ statement = second_brother_says)
axiom falsa_always_lies : ∀ (b : Brother) (statement : String), b = Brother.Falsa ↔ ¬(statement = first_brother_says ∨ statement = second_brother_says)

-- Proof statement 
theorem identify_first_brother :
  ∃ (b : Brother), b = Brother.Trulya :=
sorry

end NUMINAMATH_GPT_identify_first_brother_l1606_160667


namespace NUMINAMATH_GPT_remainder_x_plus_3uy_div_y_l1606_160689

theorem remainder_x_plus_3uy_div_y (x y u v : ℕ) (hx : x = u * y + v) (hv_range : 0 ≤ v ∧ v < y) :
  (x + 3 * u * y) % y = v :=
by
  sorry

end NUMINAMATH_GPT_remainder_x_plus_3uy_div_y_l1606_160689


namespace NUMINAMATH_GPT_tank_emptying_time_l1606_160669

theorem tank_emptying_time (fill_without_leak fill_with_leak : ℝ) (h1 : fill_without_leak = 7) (h2 : fill_with_leak = 8) : 
  let R := 1 / fill_without_leak
  let L := R - 1 / fill_with_leak
  let emptying_time := 1 / L
  emptying_time = 56 :=
by
  sorry

end NUMINAMATH_GPT_tank_emptying_time_l1606_160669


namespace NUMINAMATH_GPT_value_of_expression_l1606_160697

theorem value_of_expression (a b : ℤ) (h : 2 * a - b = 10) : 2023 - 2 * a + b = 2013 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1606_160697


namespace NUMINAMATH_GPT_surface_area_of_sphere_l1606_160691

noncomputable def sphere_surface_area : ℝ :=
  let AB := 2
  let SA := 2
  let SB := 2
  let SC := 2
  let ABC_is_isosceles_right := true -- denotes the property
  let SABC_on_sphere := true -- denotes the property
  let R := (2 * Real.sqrt 3) / 3
  let surface_area := 4 * Real.pi * R^2
  surface_area

theorem surface_area_of_sphere : sphere_surface_area = (16 * Real.pi) / 3 := 
sorry

end NUMINAMATH_GPT_surface_area_of_sphere_l1606_160691


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1606_160607

theorem solution_set_of_inequality (x : ℝ) : x * (1 - x) > 0 ↔ 0 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1606_160607


namespace NUMINAMATH_GPT_sum_consecutive_integers_150_l1606_160676

theorem sum_consecutive_integers_150 (n : ℕ) (a : ℕ) (hn : n ≥ 3) (hdiv : 300 % n = 0) :
  n * (2 * a + n - 1) = 300 ↔ (a > 0) → n = 3 ∨ n = 5 ∨ n = 15 :=
by sorry

end NUMINAMATH_GPT_sum_consecutive_integers_150_l1606_160676


namespace NUMINAMATH_GPT_largest_initial_number_l1606_160605

theorem largest_initial_number : ∃ (n : ℕ), (∀ i, 1 ≤ i ∧ i ≤ 5 → ∃ a : ℕ, ¬ (n + (i - 1) * a = n + (i - 1) * a) ∧ n + (i - 1) * a = 100) ∧ (∀ m, m ≥ n → m = 89) := 
sorry

end NUMINAMATH_GPT_largest_initial_number_l1606_160605


namespace NUMINAMATH_GPT_expression_not_defined_at_x_eq_5_l1606_160636

theorem expression_not_defined_at_x_eq_5 :
  ∃ x : ℝ, x^3 - 15 * x^2 + 75 * x - 125 = 0 ↔ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_expression_not_defined_at_x_eq_5_l1606_160636


namespace NUMINAMATH_GPT_first_dog_walks_two_miles_per_day_l1606_160661

variable (x : ℝ)

theorem first_dog_walks_two_miles_per_day  
  (h1 : 7 * x + 56 = 70) : 
  x = 2 := 
by 
  sorry

end NUMINAMATH_GPT_first_dog_walks_two_miles_per_day_l1606_160661


namespace NUMINAMATH_GPT_range_of_m_l1606_160678

noncomputable def f (x : ℝ) : ℝ := sorry -- to be defined as an odd, decreasing function

theorem range_of_m 
  (hf_odd : ∀ x, f (-x) = -f x) -- f is odd
  (hf_decreasing : ∀ x y, x < y → f y < f x) -- f is strictly decreasing
  (h_condition : ∀ m, f (1 - m) + f (1 - m^2) < 0) :
  ∀ m, (0 < m ∧ m < 1) :=
sorry

end NUMINAMATH_GPT_range_of_m_l1606_160678


namespace NUMINAMATH_GPT_correct_factorization_l1606_160634

theorem correct_factorization (x m n a : ℝ) : 
  (¬ (x^2 + 2 * x + 1 = x * (x + 2) + 1)) ∧
  (¬ (m^2 - 2 * m * n + n^2 = (m + n)^2)) ∧
  (¬ (-a^4 + 16 = -(a^2 + 4) * (a^2 - 4))) ∧
  (x^3 - 4 * x = x * (x + 2) * (x - 2)) :=
by
  sorry

end NUMINAMATH_GPT_correct_factorization_l1606_160634


namespace NUMINAMATH_GPT_remove_wallpaper_time_l1606_160630

theorem remove_wallpaper_time 
    (total_walls : ℕ := 8)
    (remaining_walls : ℕ := 7)
    (time_for_remaining_walls : ℕ := 14) :
    time_for_remaining_walls / remaining_walls = 2 :=
by
sorry

end NUMINAMATH_GPT_remove_wallpaper_time_l1606_160630


namespace NUMINAMATH_GPT_tangent_line_x_squared_at_one_one_l1606_160670

open Real

theorem tangent_line_x_squared_at_one_one :
  ∀ (x y : ℝ), y = x^2 → (x, y) = (1, 1) → (2 * x - y - 1 = 0) :=
by
  intros x y h_curve h_point
  sorry

end NUMINAMATH_GPT_tangent_line_x_squared_at_one_one_l1606_160670


namespace NUMINAMATH_GPT_bread_slices_per_friend_l1606_160613

theorem bread_slices_per_friend :
  (∀ (slices_per_loaf friends loaves total_slices_per_friend : ℕ),
    slices_per_loaf = 15 →
    friends = 10 →
    loaves = 4 →
    total_slices_per_friend = slices_per_loaf * loaves / friends →
    total_slices_per_friend = 6) :=
by 
  intros slices_per_loaf friends loaves total_slices_per_friend h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_bread_slices_per_friend_l1606_160613


namespace NUMINAMATH_GPT_tan_alpha_eq_two_and_expression_value_sin_tan_simplify_l1606_160621

-- First problem: Given condition and expression to be proved equal to the correct answer.
theorem tan_alpha_eq_two_and_expression_value (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (2 * Real.pi - α) + Real.cos (Real.pi + α)) / 
  (Real.cos (α - Real.pi) - Real.cos (3 * Real.pi / 2 - α)) = -3 := sorry

-- Second problem: Given expression to be proved simplified to the correct answer.
theorem sin_tan_simplify :
  Real.sin (50 * Real.pi / 180) * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi/180)) = 1 := sorry

end NUMINAMATH_GPT_tan_alpha_eq_two_and_expression_value_sin_tan_simplify_l1606_160621


namespace NUMINAMATH_GPT_calculate_bubble_bath_needed_l1606_160629

theorem calculate_bubble_bath_needed :
  let double_suites_capacity := 5 * 4
  let rooms_for_couples_capacity := 13 * 2
  let single_rooms_capacity := 14 * 1
  let family_rooms_capacity := 3 * 6
  let total_guests := double_suites_capacity + rooms_for_couples_capacity + single_rooms_capacity + family_rooms_capacity
  let bubble_bath_per_guest := 25
  total_guests * bubble_bath_per_guest = 1950 := by
  let double_suites_capacity := 5 * 4
  let rooms_for_couples_capacity := 13 * 2
  let single_rooms_capacity := 14 * 1
  let family_rooms_capacity := 3 * 6
  let total_guests := double_suites_capacity + rooms_for_couples_capacity + single_rooms_capacity + family_rooms_capacity
  let bubble_bath_per_guest := 25
  sorry

end NUMINAMATH_GPT_calculate_bubble_bath_needed_l1606_160629


namespace NUMINAMATH_GPT_volume_of_sphere_in_cone_l1606_160653

/-- The volume of a sphere inscribed in a right circular cone with
a base diameter of 16 inches and a cross-section with a vertex angle of 45 degrees
is 4096 * sqrt 2 * π / 3 cubic inches. -/
theorem volume_of_sphere_in_cone :
  let d := 16 -- the diameter of the base of the cone in inches
  let angle := 45 -- the vertex angle of the cross-section triangle in degrees
  let r := 8 * Real.sqrt 2 -- the radius of the sphere in inches
  let V := 4 / 3 * Real.pi * r^3 -- the volume of the sphere in cubic inches
  V = 4096 * Real.sqrt 2 * Real.pi / 3 :=
by
  simp only [Real.sqrt]
  sorry -- proof goes here

end NUMINAMATH_GPT_volume_of_sphere_in_cone_l1606_160653


namespace NUMINAMATH_GPT_convex_quadrilateral_area_lt_a_sq_l1606_160602

theorem convex_quadrilateral_area_lt_a_sq {a x y z t : ℝ} (hx : x < a) (hy : y < a) (hz : z < a) (ht : t < a) :
  (∃ S : ℝ, S < a^2) :=
sorry

end NUMINAMATH_GPT_convex_quadrilateral_area_lt_a_sq_l1606_160602


namespace NUMINAMATH_GPT_range_of_a_l1606_160696

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) ↔ (0 ≤ a ∧ a ≤ 1) := 
sorry

end NUMINAMATH_GPT_range_of_a_l1606_160696


namespace NUMINAMATH_GPT_power_mod_remainder_l1606_160601

theorem power_mod_remainder (a b : ℕ) (h1 : a = 3) (h2 : b = 167) :
  (3^167) % 11 = 9 := by
  sorry

end NUMINAMATH_GPT_power_mod_remainder_l1606_160601


namespace NUMINAMATH_GPT_evaluate_poly_at_2_l1606_160637

def my_op (x y : ℕ) : ℕ := (x + 1) * (y + 1)
def star2 (x : ℕ) : ℕ := my_op x x

theorem evaluate_poly_at_2 :
  3 * (star2 2) - 2 * 2 + 1 = 24 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_poly_at_2_l1606_160637


namespace NUMINAMATH_GPT_remainder_when_divided_by_DE_l1606_160627

theorem remainder_when_divided_by_DE (P D E Q R M S C : ℕ) 
  (h1 : P = Q * D + R) 
  (h2 : Q = E * M + S) :
  (∃ quotient : ℕ, P = quotient * (D * E) + (S * D + R + C)) :=
by {
  sorry
}

end NUMINAMATH_GPT_remainder_when_divided_by_DE_l1606_160627


namespace NUMINAMATH_GPT_find_x_l1606_160648

theorem find_x (x : ℝ) (h : 0.75 * x + 2 = 8) : x = 8 :=
sorry

end NUMINAMATH_GPT_find_x_l1606_160648


namespace NUMINAMATH_GPT_evaluate_nested_square_root_l1606_160665

-- Define the condition
def pos_real_solution (x : ℝ) : Prop := x = Real.sqrt (18 + x)

-- State the theorem
theorem evaluate_nested_square_root :
  ∃ (x : ℝ), pos_real_solution x ∧ x = (1 + Real.sqrt 73) / 2 :=
sorry

end NUMINAMATH_GPT_evaluate_nested_square_root_l1606_160665


namespace NUMINAMATH_GPT_find_sum_l1606_160688

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (2 + x) + f (2 - x) = 0

theorem find_sum (f : ℝ → ℝ) (h_odd : odd_function f) (h_func : functional_equation f) (h_val : f 1 = 9) :
  f 2016 + f 2017 + f 2018 = 9 :=
  sorry

end NUMINAMATH_GPT_find_sum_l1606_160688


namespace NUMINAMATH_GPT_smallest_n_exists_l1606_160628

theorem smallest_n_exists (n : ℕ) (h : n ≥ 4) :
  (∃ (S : Finset ℤ), S.card = n ∧
    (∃ (a b c d : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
        (a + b - c - d) % 20 = 0))
  ↔ n = 9 := sorry

end NUMINAMATH_GPT_smallest_n_exists_l1606_160628


namespace NUMINAMATH_GPT_evaluate_expr_l1606_160662

theorem evaluate_expr : Int.ceil (5 / 4 : ℚ) + Int.floor (-5 / 4 : ℚ) = 0 := by
  sorry

end NUMINAMATH_GPT_evaluate_expr_l1606_160662


namespace NUMINAMATH_GPT_remaining_money_l1606_160684

def initial_amount : Float := 499.9999999999999

def spent_on_clothes (initial : Float) : Float :=
  (1/3) * initial

def remaining_after_clothes (initial : Float) : Float :=
  initial - spent_on_clothes initial

def spent_on_food (remaining_clothes : Float) : Float :=
  (1/5) * remaining_clothes

def remaining_after_food (remaining_clothes : Float) : Float :=
  remaining_clothes - spent_on_food remaining_clothes

def spent_on_travel (remaining_food : Float) : Float :=
  (1/4) * remaining_food

def remaining_after_travel (remaining_food : Float) : Float :=
  remaining_food - spent_on_travel remaining_food

theorem remaining_money :
  remaining_after_travel (remaining_after_food (remaining_after_clothes initial_amount)) = 199.99 :=
by
  sorry

end NUMINAMATH_GPT_remaining_money_l1606_160684


namespace NUMINAMATH_GPT_no_integer_solutions_l1606_160650

theorem no_integer_solutions (x y z : ℤ) (h1 : x > y) (h2 : y > z) : 
  x * (x - y) + y * (y - z) + z * (z - x) ≠ 3 := 
by
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l1606_160650


namespace NUMINAMATH_GPT_consecutive_odd_sum_l1606_160682

theorem consecutive_odd_sum (n : ℤ) (h : n + 2 = 9) : 
  let a := n
  let b := n + 2
  let c := n + 4
  (a + b + c) = a + 20 := by
  sorry

end NUMINAMATH_GPT_consecutive_odd_sum_l1606_160682


namespace NUMINAMATH_GPT_Lilith_caps_collection_l1606_160623

theorem Lilith_caps_collection
  (caps_per_month_first_year : ℕ)
  (caps_per_month_after_first_year : ℕ)
  (caps_received_each_christmas : ℕ)
  (caps_lost_per_year : ℕ)
  (total_caps_collected : ℕ)
  (first_year_caps : ℕ := caps_per_month_first_year * 12)
  (years_after_first_year : ℕ)
  (total_years : ℕ := years_after_first_year + 1)
  (caps_collected_after_first_year : ℕ := caps_per_month_after_first_year * 12 * years_after_first_year)
  (caps_received_total : ℕ := caps_received_each_christmas * total_years)
  (caps_lost_total : ℕ := caps_lost_per_year * total_years)
  (total_calculated_caps : ℕ := first_year_caps + caps_collected_after_first_year + caps_received_total - caps_lost_total) :
  total_caps_collected = 401 → total_years = 5 :=
by
  sorry

end NUMINAMATH_GPT_Lilith_caps_collection_l1606_160623


namespace NUMINAMATH_GPT_find_y_l1606_160699

theorem find_y (y : ℝ) (h : 3 * y / 4 = 15) : y = 20 :=
sorry

end NUMINAMATH_GPT_find_y_l1606_160699


namespace NUMINAMATH_GPT_billiard_ball_weight_l1606_160683

theorem billiard_ball_weight (w_box w_box_with_balls : ℝ) (h_w_box : w_box = 0.5) 
(h_w_box_with_balls : w_box_with_balls = 1.82) : 
    let total_weight_balls := w_box_with_balls - w_box;
    let weight_one_ball := total_weight_balls / 6;
    weight_one_ball = 0.22 :=
by
  sorry

end NUMINAMATH_GPT_billiard_ball_weight_l1606_160683


namespace NUMINAMATH_GPT_cos_of_angle_B_l1606_160614

theorem cos_of_angle_B (A B C a b c : Real) (h₁ : A - C = Real.pi / 2) (h₂ : 2 * b = a + c) 
  (h₃ : 2 * a * Real.sin A = 2 * b * Real.sin B) (h₄ : 2 * c * Real.sin C = 2 * b * Real.sin B) :
  Real.cos B = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_cos_of_angle_B_l1606_160614


namespace NUMINAMATH_GPT_common_root_value_l1606_160635

theorem common_root_value (a : ℝ) : 
  (∃ x : ℝ, x^2 + a * x + 8 = 0 ∧ x^2 + x + a = 0) ↔ a = -6 :=
sorry

end NUMINAMATH_GPT_common_root_value_l1606_160635


namespace NUMINAMATH_GPT_household_member_count_l1606_160642

variable (M : ℕ) -- the number of members in the household

-- Conditions
def slices_per_breakfast := 3
def slices_per_snack := 2
def slices_per_member_daily := slices_per_breakfast + slices_per_snack
def slices_per_loaf := 12
def loaves_last_days := 3
def loaves_given := 5
def total_slices := slices_per_loaf * loaves_given
def daily_consumption := total_slices / loaves_last_days

-- Proof statement
theorem household_member_count : daily_consumption = slices_per_member_daily * M → M = 4 :=
by
  sorry

end NUMINAMATH_GPT_household_member_count_l1606_160642


namespace NUMINAMATH_GPT_sqrt_inequality_l1606_160654

theorem sqrt_inequality (x : ℝ) (h : ∀ r : ℝ, r = 2 * x - 1 → r ≥ 0) : x ≥ 1 / 2 :=
sorry

end NUMINAMATH_GPT_sqrt_inequality_l1606_160654


namespace NUMINAMATH_GPT_ineq_condition_l1606_160631

theorem ineq_condition (a b : ℝ) : (a + 1 > b - 2) ↔ (a > b - 3 ∧ ¬(a > b)) :=
by
  sorry

end NUMINAMATH_GPT_ineq_condition_l1606_160631


namespace NUMINAMATH_GPT_greatest_third_side_l1606_160698

theorem greatest_third_side (a b c : ℝ) (h₀: a = 5) (h₁: b = 11) (h₂ : 6 < c ∧ c < 16) : c ≤ 15 :=
by
  -- assumption applying that c needs to be within 6 and 16
  have h₃ : 6 < c := h₂.1
  have h₄: c < 16 := h₂.2
  -- need to show greatest integer c is 15
  sorry

end NUMINAMATH_GPT_greatest_third_side_l1606_160698


namespace NUMINAMATH_GPT_two_digit_sabroso_numbers_l1606_160647

theorem two_digit_sabroso_numbers :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ ∃ a b : ℕ, n = 10 * a + b ∧ (n + (10 * b + a) = k^2)} =
  {29, 38, 47, 56, 65, 74, 83, 92} :=
sorry

end NUMINAMATH_GPT_two_digit_sabroso_numbers_l1606_160647


namespace NUMINAMATH_GPT_log_ratio_l1606_160639

theorem log_ratio : (Real.logb 2 16) / (Real.logb 2 4) = 2 := sorry

end NUMINAMATH_GPT_log_ratio_l1606_160639


namespace NUMINAMATH_GPT_eggplant_weight_l1606_160660

-- Define the conditions
def number_of_cucumbers : ℕ := 25
def weight_per_cucumber_basket : ℕ := 30
def number_of_eggplants : ℕ := 32
def total_weight : ℕ := 1870

-- Define the statement to be proved
theorem eggplant_weight :
  (total_weight - (number_of_cucumbers * weight_per_cucumber_basket)) / number_of_eggplants =
  (1870 - (25 * 30)) / 32 := 
by sorry

end NUMINAMATH_GPT_eggplant_weight_l1606_160660


namespace NUMINAMATH_GPT_find_omega_and_range_l1606_160608

noncomputable def f (ω : ℝ) (x : ℝ) := (Real.sin (ω * x))^2 + (Real.sqrt 3) * (Real.sin (ω * x)) * (Real.sin (ω * x + Real.pi / 2))

theorem find_omega_and_range :
  ∃ ω : ℝ, ω > 0 ∧ (∀ x, f ω x = (Real.sin (2 * ω * x - Real.pi / 6) + 1/2)) ∧
    (∀ x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 2),
      f 1 x ∈ Set.Icc ((1 - Real.sqrt 3) / 2) (3 / 2)) :=
by
  sorry

end NUMINAMATH_GPT_find_omega_and_range_l1606_160608


namespace NUMINAMATH_GPT_inscribed_squares_ratio_l1606_160651

theorem inscribed_squares_ratio (x y : ℝ) (h1 : ∃ (x : ℝ), x * (13 * 12 + 13 * 5 - 5 * 12) = 60) 
  (h2 : ∃ (y : ℝ), 30 * y = 13 ^ 2) :
  x / y = 1800 / 2863 := 
sorry

end NUMINAMATH_GPT_inscribed_squares_ratio_l1606_160651


namespace NUMINAMATH_GPT_polar_to_cartesian_max_and_min_x_plus_y_l1606_160620

-- Define the given polar equation and convert it to Cartesian equations
def polar_equation (rho θ : ℝ) : Prop :=
  rho^2 - 4 * (Real.sqrt 2) * rho * Real.cos (θ - Real.pi / 4) + 6 = 0

-- Cartesian equation derived from the polar equation
def cartesian_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 4 * y + 6 = 0

-- Prove equivalence of the given polar equation and its equivalent Cartesian form for all ρ and \theta
theorem polar_to_cartesian (rho θ : ℝ) : 
  (∃ (x y : ℝ), polar_equation rho θ ∧ x = rho * Real.cos θ ∧ y = rho * Real.sin θ ∧ cartesian_equation x y) :=
by
  sorry

-- Property of points (x, y) on the circle defined by the Cartesian equation
def lies_on_circle (x y : ℝ) : Prop :=
  cartesian_equation x y

-- Given a point (x, y) on the circle defined by cartesian_equation, show bounds for x + y
theorem max_and_min_x_plus_y (x y : ℝ) (h : lies_on_circle x y) : 
  2 ≤ x + y ∧ x + y ≤ 6 :=
by
  sorry

end NUMINAMATH_GPT_polar_to_cartesian_max_and_min_x_plus_y_l1606_160620


namespace NUMINAMATH_GPT_large_cube_surface_area_l1606_160619

-- Define given conditions
def small_cube_volume := 512 -- volume in cm^3
def num_small_cubes := 8

-- Define side length of small cube
def small_cube_side_length := (small_cube_volume : ℝ)^(1/3)

-- Define side length of large cube
def large_cube_side_length := 2 * small_cube_side_length

-- Surface area formula for a cube
def surface_area (side_length : ℝ) := 6 * side_length^2

-- Theorem: The surface area of the large cube is 1536 cm^2
theorem large_cube_surface_area :
  surface_area large_cube_side_length = 1536 :=
sorry

end NUMINAMATH_GPT_large_cube_surface_area_l1606_160619


namespace NUMINAMATH_GPT_sampling_methods_correct_l1606_160679

-- Assuming definitions for the populations for both surveys
structure CommunityHouseholds where
  high_income : Nat
  middle_income : Nat
  low_income : Nat

structure ArtisticStudents where
  total_students : Nat

-- Given conditions
def households_population : CommunityHouseholds := { high_income := 125, middle_income := 280, low_income := 95 }
def students_population : ArtisticStudents := { total_students := 15 }

-- Correct answer according to the conditions
def appropriate_sampling_methods (ch: CommunityHouseholds) (as: ArtisticStudents) : String :=
  if ch.high_income > 0 ∧ ch.middle_income > 0 ∧ ch.low_income > 0 ∧ as.total_students ≥ 3 then
    "B" -- ① Stratified sampling, ② Simple random sampling
  else
    "Invalid"

theorem sampling_methods_correct :
  appropriate_sampling_methods households_population students_population = "B" := by
  sorry

end NUMINAMATH_GPT_sampling_methods_correct_l1606_160679


namespace NUMINAMATH_GPT_correct_transformation_option_c_l1606_160624

theorem correct_transformation_option_c (x : ℝ) (h : (x / 2) - (x / 3) = 1) : 3 * x - 2 * x = 6 :=
by
  sorry

end NUMINAMATH_GPT_correct_transformation_option_c_l1606_160624


namespace NUMINAMATH_GPT_algae_colony_growth_l1606_160609

def initial_cells : ℕ := 5
def days : ℕ := 10
def tripling_period : ℕ := 3
def cell_growth_ratio : ℕ := 3

noncomputable def cells_after_n_days (init_cells : ℕ) (day_count : ℕ) (period : ℕ) (growth_ratio : ℕ) : ℕ :=
  let steps := day_count / period
  init_cells * growth_ratio^steps

theorem algae_colony_growth : cells_after_n_days initial_cells days tripling_period cell_growth_ratio = 135 :=
  by sorry

end NUMINAMATH_GPT_algae_colony_growth_l1606_160609


namespace NUMINAMATH_GPT_jury_selection_duration_is_two_l1606_160625

variable (jury_selection_days : ℕ) (trial_days : ℕ) (deliberation_days : ℕ)

axiom trial_lasts_four_times_jury_selection : trial_days = 4 * jury_selection_days
axiom deliberation_is_six_full_days : deliberation_days = (6 * 24) / 16
axiom john_spends_nineteen_days : jury_selection_days + trial_days + deliberation_days = 19

theorem jury_selection_duration_is_two : jury_selection_days = 2 :=
by
  sorry

end NUMINAMATH_GPT_jury_selection_duration_is_two_l1606_160625


namespace NUMINAMATH_GPT_plane_through_points_l1606_160681

-- Define the vectors as tuples of three integers
def point := (ℤ × ℤ × ℤ)

-- The given points
def p : point := (2, -1, 3)
def q : point := (4, -1, 5)
def r : point := (5, -3, 4)

-- A function to find the equation of the plane given three points
def plane_equation (p q r : point) : ℤ × ℤ × ℤ × ℤ :=
  let (px, py, pz) := p
  let (qx, qy, qz) := q
  let (rx, ry, rz) := r
  let a := (qy - py) * (rz - pz) - (qy - py) * (rz - pz)
  let b := (qx - px) * (rz - pz) - (qx - px) * (rz - pz)
  let c := (qx - px) * (ry - py) - (qx - px) * (ry - py)
  let d := -(a * px + b * py + c * pz)
  (a, b, c, d)

-- The proof statement
theorem plane_through_points : plane_equation (2, -1, 3) (4, -1, 5) (5, -3, 4) = (1, 2, -2, 6) :=
  by sorry

end NUMINAMATH_GPT_plane_through_points_l1606_160681


namespace NUMINAMATH_GPT_milkshake_hours_l1606_160600

theorem milkshake_hours (h : ℕ) : 
  (3 * h + 7 * h = 80) → h = 8 := 
by
  intro h_milkshake_eq
  sorry

end NUMINAMATH_GPT_milkshake_hours_l1606_160600


namespace NUMINAMATH_GPT_sum_of_monomials_same_type_l1606_160675

theorem sum_of_monomials_same_type 
  (x y : ℝ) 
  (m n : ℕ) 
  (h1 : m = 1) 
  (h2 : 3 = n + 1) : 
  (2 * x ^ m * y ^ 3) + (-5 * x * y ^ (n + 1)) = -3 * x * y ^ 3 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_monomials_same_type_l1606_160675


namespace NUMINAMATH_GPT_inequality_implies_strict_inequality_l1606_160657

theorem inequality_implies_strict_inequality (x y z : ℝ) (h : x^2 + x * y + x * z < 0) : y^2 > 4 * x * z :=
sorry

end NUMINAMATH_GPT_inequality_implies_strict_inequality_l1606_160657


namespace NUMINAMATH_GPT_tim_cantaloupes_l1606_160690

theorem tim_cantaloupes (fred_cantaloupes : ℕ) (total_cantaloupes : ℕ) (h1 : fred_cantaloupes = 38) (h2 : total_cantaloupes = 82) : total_cantaloupes - fred_cantaloupes = 44 :=
by {
  -- proof steps go here
  sorry
}

end NUMINAMATH_GPT_tim_cantaloupes_l1606_160690


namespace NUMINAMATH_GPT_triangle_area_l1606_160615

theorem triangle_area (a b : ℝ) (sinC sinA : ℝ) 
  (h1 : a = Real.sqrt 5) 
  (h2 : b = 3) 
  (h3 : sinC = 2 * sinA) : 
  ∃ (area : ℝ), area = 3 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_area_l1606_160615


namespace NUMINAMATH_GPT_total_wage_l1606_160671

theorem total_wage (work_days_A work_days_B : ℕ) (wage_A : ℕ) (total_wage : ℕ) 
  (h1 : work_days_A = 10) 
  (h2 : work_days_B = 15) 
  (h3 : wage_A = 1980)
  (h4 : (wage_A / (wage_A / (total_wage * 3 / 5))) = 3)
  : total_wage = 3300 :=
sorry

end NUMINAMATH_GPT_total_wage_l1606_160671


namespace NUMINAMATH_GPT_simplify_fraction_l1606_160603

theorem simplify_fraction (a b x : ℝ) (h₁ : x = a / b) (h₂ : a ≠ b) (h₃ : b ≠ 0) : 
  (2 * a + b) / (a - 2 * b) = (2 * x + 1) / (x - 2) :=
sorry

end NUMINAMATH_GPT_simplify_fraction_l1606_160603


namespace NUMINAMATH_GPT_min_vertical_segment_length_l1606_160626

def f (x : ℝ) : ℝ := |x - 1|
def g (x : ℝ) : ℝ := -x^2 - 4 * x - 3
def L (x : ℝ) : ℝ := f x - g x

theorem min_vertical_segment_length : ∃ (x : ℝ), L x = 10 :=
by
  sorry

end NUMINAMATH_GPT_min_vertical_segment_length_l1606_160626


namespace NUMINAMATH_GPT_remainder_of_2n_div_10_l1606_160693

theorem remainder_of_2n_div_10 (n : ℤ) (h : n % 20 = 11) : (2 * n) % 10 = 2 := by
  sorry

end NUMINAMATH_GPT_remainder_of_2n_div_10_l1606_160693


namespace NUMINAMATH_GPT_no_solution_exists_l1606_160686

theorem no_solution_exists : ¬ ∃ n : ℕ, 0 < n ∧ (2^n % 60 = 29 ∨ 2^n % 60 = 31) := 
by
  sorry

end NUMINAMATH_GPT_no_solution_exists_l1606_160686


namespace NUMINAMATH_GPT_jacket_final_price_l1606_160692

theorem jacket_final_price 
  (original_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) (final_discount : ℝ)
  (price_after_first : ℝ := original_price * (1 - first_discount))
  (price_after_second : ℝ := price_after_first * (1 - second_discount))
  (final_price : ℝ := price_after_second * (1 - final_discount)) :
  original_price = 250 ∧ first_discount = 0.4 ∧ second_discount = 0.3 ∧ final_discount = 0.1 →
  final_price = 94.5 := 
by 
  sorry

end NUMINAMATH_GPT_jacket_final_price_l1606_160692


namespace NUMINAMATH_GPT_value_of_mn_squared_l1606_160694

theorem value_of_mn_squared (m n : ℤ) (h1 : |m| = 4) (h2 : |n| = 3) (h3 : m - n < 0) : (m + n)^2 = 1 ∨ (m + n)^2 = 49 :=
by sorry

end NUMINAMATH_GPT_value_of_mn_squared_l1606_160694


namespace NUMINAMATH_GPT_median_of_trapezoid_l1606_160663

theorem median_of_trapezoid (h : ℝ) (x : ℝ) 
  (triangle_area_eq_trapezoid_area : (1 / 2) * 24 * h = ((x + (2 * x)) / 2) * h) : 
  ((x + (2 * x)) / 2) = 12 := by
  sorry

end NUMINAMATH_GPT_median_of_trapezoid_l1606_160663


namespace NUMINAMATH_GPT_scientific_notation_l1606_160633

theorem scientific_notation (a : ℝ) (n : ℤ) (h1 : 1 ≤ a ∧ a < 10) (h2 : 43050000 = a * 10^n) : a = 4.305 ∧ n = 7 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_l1606_160633


namespace NUMINAMATH_GPT_no_real_roots_range_k_l1606_160604

theorem no_real_roots_range_k (k : ℝ) : (x^2 - 2 * x - k = 0) ∧ (∀ x : ℝ, x^2 - 2 * x - k ≠ 0) → k < -1 := 
by
  sorry

end NUMINAMATH_GPT_no_real_roots_range_k_l1606_160604


namespace NUMINAMATH_GPT_number_of_days_A_to_finish_remaining_work_l1606_160646

theorem number_of_days_A_to_finish_remaining_work
  (A_days : ℕ) (B_days : ℕ) (B_work_days : ℕ) : 
  A_days = 9 → 
  B_days = 15 → 
  B_work_days = 10 → 
  ∃ d : ℕ, d = 3 :=
by 
  intros hA hB hBw
  sorry

end NUMINAMATH_GPT_number_of_days_A_to_finish_remaining_work_l1606_160646


namespace NUMINAMATH_GPT_flower_pots_on_path_count_l1606_160640

theorem flower_pots_on_path_count (L d : ℕ) (hL : L = 15) (hd : d = 3) : 
  (L / d) + 1 = 6 :=
by
  sorry

end NUMINAMATH_GPT_flower_pots_on_path_count_l1606_160640


namespace NUMINAMATH_GPT_jogged_time_l1606_160652

theorem jogged_time (J : ℕ) (W : ℕ) (r : ℚ) (h1 : r = 5 / 3) (h2 : W = 9) (h3 : r = J / W) : J = 15 := 
by
  sorry

end NUMINAMATH_GPT_jogged_time_l1606_160652


namespace NUMINAMATH_GPT_integer_values_of_b_for_polynomial_root_l1606_160685

theorem integer_values_of_b_for_polynomial_root
    (b : ℤ) :
    (∃ x : ℤ, x^3 + 6 * x^2 + b * x + 12 = 0) ↔
    b = -217 ∨ b = -74 ∨ b = -43 ∨ b = -31 ∨ b = -22 ∨ b = -19 ∨
    b = 19 ∨ b = 22 ∨ b = 31 ∨ b = 43 ∨ b = 74 ∨ b = 217 :=
    sorry

end NUMINAMATH_GPT_integer_values_of_b_for_polynomial_root_l1606_160685


namespace NUMINAMATH_GPT_rate_of_interest_l1606_160695

theorem rate_of_interest (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) (h : P > 0 ∧ T = 7 ∧ SI = P / 5 ∧ SI = (P * R * T) / 100) : 
  R = 20 / 7 := 
by
  sorry

end NUMINAMATH_GPT_rate_of_interest_l1606_160695


namespace NUMINAMATH_GPT_determine_days_l1606_160666

-- Define the problem
def team_repair_time (x y : ℕ) : Prop :=
  ((1 / (x:ℝ)) + (1 / (y:ℝ)) = 1 / 18) ∧ 
  ((2 / 3 * x + 1 / 3 * y = 40))

theorem determine_days : ∃ x y : ℕ, team_repair_time x y :=
by
    use 45
    use 30
    have h1: (1/(45:ℝ) + 1/(30:ℝ)) = 1/18 := by
        sorry
    have h2: (2/3*45 + 1/3*30 = 40) := by
        sorry 
    exact ⟨h1, h2⟩

end NUMINAMATH_GPT_determine_days_l1606_160666


namespace NUMINAMATH_GPT_product_of_possible_values_l1606_160655

theorem product_of_possible_values : 
  (∀ x : ℝ, |x - 5| - 4 = -1 → (x = 2 ∨ x = 8)) → (2 * 8) = 16 :=
by 
  sorry

end NUMINAMATH_GPT_product_of_possible_values_l1606_160655


namespace NUMINAMATH_GPT_max_parrots_l1606_160664

theorem max_parrots (x y z : ℕ) (h1 : y + z ≤ 9) (h2 : x + z ≤ 11) : x + y + z ≤ 19 :=
sorry

end NUMINAMATH_GPT_max_parrots_l1606_160664


namespace NUMINAMATH_GPT_inequality_proof_l1606_160606

theorem inequality_proof (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1/a + 1/b = 1) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n + 1) := by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1606_160606


namespace NUMINAMATH_GPT_none_of_these_l1606_160674

def table : List (ℕ × ℕ) := [(1, 5), (2, 15), (3, 33), (4, 61), (5, 101)]

def formula_A (x : ℕ) : ℕ := 2 * x^3 + 3 * x^2 - x + 1
def formula_B (x : ℕ) : ℕ := 3 * x^3 + x^2 + x + 1
def formula_C (x : ℕ) : ℕ := 2 * x^3 + x^2 + x + 1
def formula_D (x : ℕ) : ℕ := 2 * x^3 + x^2 + x - 1

theorem none_of_these :
  ¬ (∀ (x y : ℕ), (x, y) ∈ table → (y = formula_A x ∨ y = formula_B x ∨ y = formula_C x ∨ y = formula_D x)) :=
by {
  sorry
}

end NUMINAMATH_GPT_none_of_these_l1606_160674


namespace NUMINAMATH_GPT_circle_area_is_162_pi_l1606_160658

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

noncomputable def circle_area (radius : ℝ) : ℝ :=
  Real.pi * radius ^ 2

def R : ℝ × ℝ := (5, -2)
def S : ℝ × ℝ := (-4, 7)

theorem circle_area_is_162_pi :
  circle_area (distance R S) = 162 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_circle_area_is_162_pi_l1606_160658


namespace NUMINAMATH_GPT_account_balance_after_transfer_l1606_160643

def account_after_transfer (initial_balance transfer_amount : ℕ) : ℕ :=
  initial_balance - transfer_amount

theorem account_balance_after_transfer :
  account_after_transfer 27004 69 = 26935 :=
by
  sorry

end NUMINAMATH_GPT_account_balance_after_transfer_l1606_160643


namespace NUMINAMATH_GPT_probability_no_3x3_red_square_l1606_160668

def grid_probability (m n : ℕ) : Prop :=
  (gcd m n = 1) ∧ ((m : ℚ) / n = 170 / 171)

theorem probability_no_3x3_red_square (m n : ℕ) (h1 : grid_probability m n) : m + n = 341 :=
by
  sorry

end NUMINAMATH_GPT_probability_no_3x3_red_square_l1606_160668


namespace NUMINAMATH_GPT_amount_borrowed_l1606_160617

variable (P : ℝ)
variable (interest_paid : ℝ) -- Interest paid on borrowing
variable (interest_earned : ℝ) -- Interest earned on lending
variable (gain_per_year : ℝ)

variable (h1 : interest_paid = P * 4 * 2 / 100)
variable (h2 : interest_earned = P * 6 * 2 / 100)
variable (h3 : gain_per_year = 160)
variable (h4 : gain_per_year = (interest_earned - interest_paid) / 2)

theorem amount_borrowed : P = 8000 := by
  sorry

end NUMINAMATH_GPT_amount_borrowed_l1606_160617


namespace NUMINAMATH_GPT_total_heads_l1606_160680

theorem total_heads (D P : ℕ) (h1 : D = 9) (h2 : 4 * D + 2 * P = 42) : D + P = 12 :=
by
  sorry

end NUMINAMATH_GPT_total_heads_l1606_160680
