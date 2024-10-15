import Mathlib

namespace NUMINAMATH_GPT_compare_y_values_l1596_159612

-- Define the quadratic function y = x^2 + 2x + c
def quadratic (x : ℝ) (c : ℝ) : ℝ := x^2 + 2 * x + c

-- Points A, B, and C on the quadratic function
variables 
  (c : ℝ) 
  (y1 y2 y3 : ℝ) 
  (hA : y1 = quadratic (-3) c) 
  (hB : y2 = quadratic (-2) c) 
  (hC : y3 = quadratic 2 c)

theorem compare_y_values :
  y3 > y1 ∧ y1 > y2 :=
by sorry

end NUMINAMATH_GPT_compare_y_values_l1596_159612


namespace NUMINAMATH_GPT_lost_card_number_l1596_159669

theorem lost_card_number (n x : ℕ) (sum_n : ℕ) (h_sum : sum_n = n * (n + 1) / 2) (h_remaining_sum : sum_n - x = 101) : x = 4 :=
sorry

end NUMINAMATH_GPT_lost_card_number_l1596_159669


namespace NUMINAMATH_GPT_train_a_distance_traveled_l1596_159650

variable (distance : ℝ) (speedA : ℝ) (speedB : ℝ) (relative_speed : ℝ) (time_to_meet : ℝ) 

axiom condition1 : distance = 450
axiom condition2 : speedA = 50
axiom condition3 : speedB = 50
axiom condition4 : relative_speed = speedA + speedB
axiom condition5 : time_to_meet = distance / relative_speed

theorem train_a_distance_traveled : (50 * time_to_meet) = 225 := by
  sorry

end NUMINAMATH_GPT_train_a_distance_traveled_l1596_159650


namespace NUMINAMATH_GPT_cost_of_bananas_and_cantaloupe_l1596_159679

-- Define variables representing the prices
variables (a b c d : ℝ)

-- Define the given conditions as hypotheses
def conditions : Prop :=
  a + b + c + d = 33 ∧
  d = 3 * a ∧
  c = a + 2 * b

-- State the main theorem
theorem cost_of_bananas_and_cantaloupe (h : conditions a b c d) : b + c = 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_cost_of_bananas_and_cantaloupe_l1596_159679


namespace NUMINAMATH_GPT_stratified_sampling_school_C_l1596_159603

theorem stratified_sampling_school_C 
  (teachers_A : ℕ) 
  (teachers_B : ℕ) 
  (teachers_C : ℕ) 
  (total_teachers : ℕ)
  (total_drawn : ℕ)
  (hA : teachers_A = 180)
  (hB : teachers_B = 140)
  (hC : teachers_C = 160)
  (hTotal : total_teachers = teachers_A + teachers_B + teachers_C)
  (hDraw : total_drawn = 60) :
  (total_drawn * teachers_C / total_teachers) = 20 := 
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_school_C_l1596_159603


namespace NUMINAMATH_GPT_T_perimeter_is_20_l1596_159663

-- Define the perimeter of a rectangle given its length and width
def perimeter_rectangle (length width : ℝ) : ℝ :=
  2 * length + 2 * width

-- Given conditions
def rect1_length : ℝ := 1
def rect1_width : ℝ := 4
def rect2_length : ℝ := 2
def rect2_width : ℝ := 5
def overlap_height : ℝ := 1

-- Calculate the perimeter of each rectangle
def perimeter_rect1 : ℝ := perimeter_rectangle rect1_length rect1_width
def perimeter_rect2 : ℝ := perimeter_rectangle rect2_length rect2_width

-- Calculate the overlap adjustment
def overlap_adjustment : ℝ := 2 * overlap_height

-- The total perimeter of the T shape
def perimeter_T : ℝ := perimeter_rect1 + perimeter_rect2 - overlap_adjustment

-- The proof statement that we need to show
theorem T_perimeter_is_20 : perimeter_T = 20 := by
  sorry

end NUMINAMATH_GPT_T_perimeter_is_20_l1596_159663


namespace NUMINAMATH_GPT_triangle_equilateral_l1596_159674

noncomputable def point := (ℝ × ℝ)

noncomputable def D : point := (0, 0)
noncomputable def E : point := (2, 0)
noncomputable def F : point := (1, Real.sqrt 3)

noncomputable def dist (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def D' (l : ℝ) : point :=
  let ED := dist E D
  (D.1 + l * ED * (Real.sqrt 3), D.2 + l * ED)

noncomputable def E' (l : ℝ) : point :=
  let DF := dist D F
  (E.1 + l * DF * (Real.sqrt 3), E.2 + l * DF)

noncomputable def F' (l : ℝ) : point :=
  let DE := dist D E
  (F.1 - 2 * l * DE, F.2 + (Real.sqrt 3 - l * DE))

theorem triangle_equilateral (l : ℝ) (h : l = 1 / Real.sqrt 3) :
  let DD' := dist D (D' l)
  let EE' := dist E (E' l)
  let FF' := dist F (F' l)
  dist (D' l) (E' l) = dist (E' l) (F' l) ∧ dist (E' l) (F' l) = dist (F' l) (D' l) ∧ dist (F' l) (D' l) = dist (D' l) (E' l) := sorry

end NUMINAMATH_GPT_triangle_equilateral_l1596_159674


namespace NUMINAMATH_GPT_subset_123_12_false_l1596_159694

-- Definitions derived from conditions
def is_int (x : ℤ) := true
def subset_123_12 (A B : Set ℕ) := A = {1, 2, 3} ∧ B = {1, 2}
def intersection_empty {A B : Set ℕ} (hA : A = {1, 2}) (hB : B = ∅) := (A ∩ B = ∅)
def union_nat_real {A B : Set ℝ} (hA : Set.univ ⊆ A) (hB : Set.univ ⊆ B) := (A ∪ B)

-- The mathematically equivalent proof problem
theorem subset_123_12_false (A B : Set ℕ) (hA : A = {1, 2, 3}) (hB : B = {1, 2}):
  ¬ (A ⊆ B) :=
by
  sorry

end NUMINAMATH_GPT_subset_123_12_false_l1596_159694


namespace NUMINAMATH_GPT_solve_cubic_equation_l1596_159600

theorem solve_cubic_equation : ∀ x : ℝ, (x^3 - 5*x^2 + 6*x - 2 = 0) → (x = 2) :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_cubic_equation_l1596_159600


namespace NUMINAMATH_GPT_percentage_change_difference_l1596_159615

theorem percentage_change_difference (total_students : ℕ) (initial_enjoy : ℕ) (initial_not_enjoy : ℕ) (final_enjoy : ℕ) (final_not_enjoy : ℕ) :
  total_students = 100 →
  initial_enjoy = 40 →
  initial_not_enjoy = 60 →
  final_enjoy = 80 →
  final_not_enjoy = 20 →
  (40 ≤ y ∧ y ≤ 80) ∧ (40 - 40 = 0) ∧ (80 - 40 = 40) ∧ (80 - 40 = 40) :=
by
  sorry

end NUMINAMATH_GPT_percentage_change_difference_l1596_159615


namespace NUMINAMATH_GPT_right_triangle_perimeter_l1596_159685

theorem right_triangle_perimeter (area : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) 
  (h_area : area = 120)
  (h_a : a = 24)
  (h_area_eq : area = (1/2) * a * b)
  (h_c : c^2 = a^2 + b^2) :
  a + b + c = 60 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_perimeter_l1596_159685


namespace NUMINAMATH_GPT_interval_contains_root_l1596_159620

theorem interval_contains_root :
  (∃ c, (0 < c ∧ c < 1) ∧ (2^c + c - 2 = 0) ∧ 
        (∀ x1 x2, x1 < x2 → 2^x1 + x1 - 2 < 2^x2 + x2 - 2) ∧ 
        (0 < 1) ∧ 
        ((2^0 + 0 - 2) = -1) ∧ 
        ((2^1 + 1 - 2) = 1)) := 
by 
  sorry

end NUMINAMATH_GPT_interval_contains_root_l1596_159620


namespace NUMINAMATH_GPT_sugar_initial_weight_l1596_159662

theorem sugar_initial_weight (packs : ℕ) (pack_weight : ℕ) (leftover : ℕ) (used_percentage : ℝ)
  (h1 : packs = 30)
  (h2 : pack_weight = 350)
  (h3 : leftover = 50)
  (h4 : used_percentage = 0.60) : 
  (packs * pack_weight + leftover) = 10550 :=
by 
  sorry

end NUMINAMATH_GPT_sugar_initial_weight_l1596_159662


namespace NUMINAMATH_GPT_combined_weight_of_Meg_and_Chris_cats_l1596_159659

-- Definitions based on the conditions
def ratio (M A C : ℕ) : Prop := 13 * A = 21 * M ∧ 13 * C = 28 * M 
def half_anne (M A : ℕ) : Prop := M = 20 + A / 2
def total_weight (M A C T : ℕ) : Prop := T = M + A + C

-- Theorem statement
theorem combined_weight_of_Meg_and_Chris_cats (M A C T : ℕ) 
  (h1 : ratio M A C) 
  (h2 : half_anne M A) 
  (h3 : total_weight M A C T) : 
  M + C = 328 := 
sorry

end NUMINAMATH_GPT_combined_weight_of_Meg_and_Chris_cats_l1596_159659


namespace NUMINAMATH_GPT_neg_exists_equiv_forall_l1596_159643

theorem neg_exists_equiv_forall (p : ∃ n : ℕ, 2^n > 1000) :
  (¬ ∃ n : ℕ, 2^n > 1000) ↔ ∀ n : ℕ, 2^n ≤ 1000 := 
sorry

end NUMINAMATH_GPT_neg_exists_equiv_forall_l1596_159643


namespace NUMINAMATH_GPT_irreducible_fraction_eq_l1596_159687

theorem irreducible_fraction_eq (p q : ℕ) (h1 : p > 0) (h2 : q > 0) (h3 : Nat.gcd p q = 1) (h4 : q % 2 = 1) :
  ∃ n k : ℕ, n > 0 ∧ k > 0 ∧ (p : ℚ) / q = (n : ℚ) / (2 ^ k - 1) :=
by
  sorry

end NUMINAMATH_GPT_irreducible_fraction_eq_l1596_159687


namespace NUMINAMATH_GPT_find_x_when_parallel_l1596_159660

-- Given vectors
def a : ℝ × ℝ := (-2, 4)
def b (x : ℝ) : ℝ × ℝ := (x, -2)

-- Conditional statement: parallel vectors
def parallel_vectors (u v : ℝ × ℝ) : Prop := 
  u.1 * v.2 = u.2 * v.1

-- Proof statement
theorem find_x_when_parallel (x : ℝ) (h : parallel_vectors a (b x)) : x = 1 := 
  sorry

end NUMINAMATH_GPT_find_x_when_parallel_l1596_159660


namespace NUMINAMATH_GPT_find_working_hours_for_y_l1596_159682

theorem find_working_hours_for_y (Wx Wy Wz Ww : ℝ) (h1 : Wx = 1/8)
  (h2 : Wy + Wz = 1/6) (h3 : Wx + Wz = 1/4) (h4 : Wx + Wy + Ww = 1/5)
  (h5 : Wx + Ww + Wz = 1/3) : 1 / Wy = 24 :=
by
  -- Given the conditions
  -- Wx = 1/8
  -- Wy + Wz = 1/6
  -- Wx + Wz = 1/4
  -- Wx + Wy + Ww = 1/5
  -- Wx + Ww + Wz = 1/3
  -- We need to prove that 1 / Wy = 24
  sorry

end NUMINAMATH_GPT_find_working_hours_for_y_l1596_159682


namespace NUMINAMATH_GPT_eval_expression_l1596_159623

theorem eval_expression : (Real.sqrt (16 - 8 * Real.sqrt 6) + Real.sqrt (16 + 8 * Real.sqrt 6)) = 4 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l1596_159623


namespace NUMINAMATH_GPT_a_not_multiple_of_5_l1596_159664

theorem a_not_multiple_of_5 (a : ℤ) (h : a % 5 ≠ 0) : (a^4 + 4) % 5 = 0 :=
sorry

end NUMINAMATH_GPT_a_not_multiple_of_5_l1596_159664


namespace NUMINAMATH_GPT_symmetry_graph_l1596_159605

theorem symmetry_graph (θ:ℝ) (hθ: θ > 0):
  (∀ k: ℤ, 2 * (3 * Real.pi / 4) + (Real.pi / 3) - 2 * θ = k * Real.pi + Real.pi / 2) 
  → θ = Real.pi / 6 :=
by 
  sorry

end NUMINAMATH_GPT_symmetry_graph_l1596_159605


namespace NUMINAMATH_GPT_problem_statement_l1596_159606

noncomputable def f (x : ℝ) : ℝ := 3^x + 3^(-x)

noncomputable def g (x : ℝ) : ℝ := 3^x - 3^(-x)

theorem problem_statement : 
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, g (-x) = -g x) :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_statement_l1596_159606


namespace NUMINAMATH_GPT_pairs_divisible_by_4_l1596_159622

-- Define the set of valid pairs of digits from 00 to 99
def valid_pairs : List (Fin 100) := List.filter (λ n => n % 4 = 0) (List.range 100)

-- State the theorem
theorem pairs_divisible_by_4 : valid_pairs.length = 25 := by
  sorry

end NUMINAMATH_GPT_pairs_divisible_by_4_l1596_159622


namespace NUMINAMATH_GPT_distance_inequality_l1596_159657

theorem distance_inequality (a : ℝ) (h : |a - 1| < 3) : -2 < a ∧ a < 4 :=
sorry

end NUMINAMATH_GPT_distance_inequality_l1596_159657


namespace NUMINAMATH_GPT_least_number_with_remainders_l1596_159637

theorem least_number_with_remainders :
  ∃ x, (x ≡ 4 [MOD 5]) ∧ (x ≡ 4 [MOD 6]) ∧ (x ≡ 4 [MOD 9]) ∧ (x ≡ 4 [MOD 18]) ∧ x = 94 := 
by 
  sorry

end NUMINAMATH_GPT_least_number_with_remainders_l1596_159637


namespace NUMINAMATH_GPT_calculate_two_times_square_root_squared_l1596_159691

theorem calculate_two_times_square_root_squared : 2 * (Real.sqrt 50625) ^ 2 = 101250 := by
  sorry

end NUMINAMATH_GPT_calculate_two_times_square_root_squared_l1596_159691


namespace NUMINAMATH_GPT_max_value_of_f_f_is_increasing_on_intervals_l1596_159653

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.cos x ^ 2 + Real.sqrt 3 * Real.sin (2 * x)

theorem max_value_of_f :
  ∃ (k : ℤ), ∀ (x : ℝ), x = k * Real.pi + Real.pi / 6 → f x = 3 :=
sorry

theorem f_is_increasing_on_intervals :
  ∀ (k : ℤ), ∀ (x y : ℝ), k * Real.pi - Real.pi / 3 ≤ x →
                x ≤ y → y ≤ k * Real.pi + Real.pi / 6 →
                f x ≤ f y :=
sorry

end NUMINAMATH_GPT_max_value_of_f_f_is_increasing_on_intervals_l1596_159653


namespace NUMINAMATH_GPT_container_volume_ratio_l1596_159689

theorem container_volume_ratio
  (A B : ℚ)
  (H1 : 3/5 * A + 1/4 * B = 4/5 * B)
  (H2 : 3/5 * A = (4/5 * B - 1/4 * B)) :
  A / B = 11 / 12 :=
by
  sorry

end NUMINAMATH_GPT_container_volume_ratio_l1596_159689


namespace NUMINAMATH_GPT_inequality_one_inequality_system_l1596_159631

-- Definition for the first problem
theorem inequality_one (x : ℝ) : 3 * x > 2 * (1 - x) ↔ x > 2 / 5 :=
by
  sorry

-- Definitions for the second problem
theorem inequality_system (x : ℝ) : 
  (3 * x - 7) / 2 ≤ x - 2 ∧ 4 * (x - 1) > 4 ↔ 2 < x ∧ x ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_one_inequality_system_l1596_159631


namespace NUMINAMATH_GPT_smallest_positive_integer_n_l1596_159673

theorem smallest_positive_integer_n :
  ∃ n: ℕ, (n > 0) ∧ (∀ k: ℕ, 1 ≤ k ∧ k ≤ n → (∃ d: ℕ, d ∣ (n^2 - 2 * n) ∧ d ∣ k) ∧ (k ∣ (n^2 - 2 * n) → k = d)) ∧ n = 5 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_n_l1596_159673


namespace NUMINAMATH_GPT_ratio_of_areas_l1596_159658

theorem ratio_of_areas (T A B : ℝ) (hT : T = 900) (hB : B = 405) (hSum : A + B = T) :
  (A - B) / ((A + B) / 2) = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1596_159658


namespace NUMINAMATH_GPT_find_a_l1596_159608

variables (x y : ℝ) (a : ℝ)

-- Condition 1: Original profit equation
def original_profit := y - x = x * (a / 100)

-- Condition 2: New profit equation with 5% cost decrease
def new_profit := y - 0.95 * x = 0.95 * x * ((a + 15) / 100)

theorem find_a (h1 : original_profit x y a) (h2 : new_profit x y a) : a = 185 :=
sorry

end NUMINAMATH_GPT_find_a_l1596_159608


namespace NUMINAMATH_GPT_settle_debt_using_coins_l1596_159644

theorem settle_debt_using_coins :
  ∃ n m : ℕ, 49 * n - 99 * m = 1 :=
sorry

end NUMINAMATH_GPT_settle_debt_using_coins_l1596_159644


namespace NUMINAMATH_GPT_largest_determinable_1986_l1596_159696

-- Define main problem with conditions
def largest_determinable_cards (total : ℕ) (select : ℕ) : ℕ :=
  total - 27

-- Statement we need to prove
theorem largest_determinable_1986 :
  largest_determinable_cards 2013 10 = 1986 :=
by
  sorry

end NUMINAMATH_GPT_largest_determinable_1986_l1596_159696


namespace NUMINAMATH_GPT_sharp_triple_72_l1596_159699

-- Definition of the transformation function
def sharp (N : ℝ) : ℝ := 0.4 * N + 3

-- Theorem statement
theorem sharp_triple_72 : sharp (sharp (sharp 72)) = 9.288 := by
  sorry

end NUMINAMATH_GPT_sharp_triple_72_l1596_159699


namespace NUMINAMATH_GPT_students_divided_into_groups_l1596_159677

theorem students_divided_into_groups (total_students : ℕ) (not_picked : ℕ) (students_per_group : ℕ) (n_groups : ℕ) 
  (h1 : total_students = 64) 
  (h2 : not_picked = 36) 
  (h3 : students_per_group = 7) 
  (h4 : total_students - not_picked = 28) 
  (h5 : 28 / students_per_group = 4) :
  n_groups = 4 :=
by
  sorry

end NUMINAMATH_GPT_students_divided_into_groups_l1596_159677


namespace NUMINAMATH_GPT_no_integer_n_satisfies_conditions_l1596_159624

theorem no_integer_n_satisfies_conditions :
  ¬ ∃ n : ℕ, 0 < n ∧ 1000 ≤ n / 5 ∧ n / 5 ≤ 9999 ∧ 1000 ≤ 5 * n ∧ 5 * n ≤ 9999 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_n_satisfies_conditions_l1596_159624


namespace NUMINAMATH_GPT_eccentricity_of_hyperbola_l1596_159681

variables (a b c e : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : c = Real.sqrt (a^2 + b^2))
variable (h4 : 3 * -(a^2 / c) + c = a^2 * c / (b^2 - a^2) + c)
variable (h5 : e = c / a)

theorem eccentricity_of_hyperbola : e = Real.sqrt 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_eccentricity_of_hyperbola_l1596_159681


namespace NUMINAMATH_GPT_smallest_six_digit_divisible_by_111_l1596_159692

theorem smallest_six_digit_divisible_by_111 : ∃ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ n % 111 = 0 ∧ n = 100011 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_six_digit_divisible_by_111_l1596_159692


namespace NUMINAMATH_GPT_not_factorable_l1596_159625

-- Define the quartic polynomial P(x)
def P (x : ℤ) : ℤ := x^4 + 2 * x^2 + 2 * x + 2

-- Define the quadratic polynomials with integer coefficients
def Q₁ (a b x : ℤ) : ℤ := x^2 + a * x + b
def Q₂ (c d x : ℤ) : ℤ := x^2 + c * x + d

-- Define the condition for factorization, and the theorem to be proven
theorem not_factorable :
  ¬ ∃ (a b c d : ℤ), ∀ x : ℤ, P x = (Q₁ a b x) * (Q₂ c d x) := by
  sorry

end NUMINAMATH_GPT_not_factorable_l1596_159625


namespace NUMINAMATH_GPT_perpendicular_line_equation_l1596_159651

theorem perpendicular_line_equation :
  (∀ (x y : ℝ), 2 * x + 3 * y + 1 = 0 → x - 3 * y + 4 = 0 →
  ∃ (l : ℝ) (m : ℝ), m = 4 / 3 ∧ y = m * x + l → y = 4 / 3 * x + 1 / 9) 
  ∧ (∀ (x y : ℝ), 3 * x + 4 * y - 7 = 0 → -3 / 4 * 4 / 3 = -1) :=
by 
  sorry

end NUMINAMATH_GPT_perpendicular_line_equation_l1596_159651


namespace NUMINAMATH_GPT_simplify_complex_expression_l1596_159610

noncomputable def i : ℂ := Complex.I

theorem simplify_complex_expression : 
  (5 * (1 + i^3)) / ((2 + i) * (2 - i)) = 1 - i := by 
  sorry

end NUMINAMATH_GPT_simplify_complex_expression_l1596_159610


namespace NUMINAMATH_GPT_bc_ad_divisible_by_u_l1596_159668

theorem bc_ad_divisible_by_u 
  (a b c d u : ℤ) 
  (h1 : u ∣ a * c)
  (h2 : u ∣ b * c + a * d)
  (h3 : u ∣ b * d) : 
  u ∣ b * c ∧ u ∣ a * d :=
by
  sorry

end NUMINAMATH_GPT_bc_ad_divisible_by_u_l1596_159668


namespace NUMINAMATH_GPT_cashier_five_dollar_bills_l1596_159640

-- Define the conditions as a structure
structure CashierBills (x y : ℕ) : Prop :=
(total_bills : x + y = 126)
(total_value : 5 * x + 10 * y = 840)

-- State the theorem that we need to prove
theorem cashier_five_dollar_bills (x y : ℕ) (h : CashierBills x y) : x = 84 :=
sorry

end NUMINAMATH_GPT_cashier_five_dollar_bills_l1596_159640


namespace NUMINAMATH_GPT_arithmetic_mean_is_12_l1596_159670

/-- The arithmetic mean of the numbers 3, 11, 7, 9, 15, 13, 8, 19, 17, 21, 14, and 7 is equal to 12 -/
theorem arithmetic_mean_is_12 : 
  let numbers := [3, 11, 7, 9, 15, 13, 8, 19, 17, 21, 14, 7]
  let sum := numbers.foldl (· + ·) 0
  let count := numbers.length
  (sum / count) = 12 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_is_12_l1596_159670


namespace NUMINAMATH_GPT_largest_possible_red_socks_l1596_159688

theorem largest_possible_red_socks (t r g : ℕ) (h1 : t = r + g) (h2 : t ≤ 3000)
    (h3 : (r * (r - 1) + g * (g - 1)) * 5 = 3 * t * (t - 1)) :
    r ≤ 1199 :=
sorry

end NUMINAMATH_GPT_largest_possible_red_socks_l1596_159688


namespace NUMINAMATH_GPT_necessary_and_sufficient_for_perpendicular_l1596_159636

theorem necessary_and_sufficient_for_perpendicular (a : ℝ) :
  (a = -2) ↔ (∀ (x y : ℝ), x + 2 * y = 0 → ax + y = 1 → false) :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_for_perpendicular_l1596_159636


namespace NUMINAMATH_GPT_probability_not_both_ends_l1596_159618

theorem probability_not_both_ends :
  let total_arrangements := 120
  let both_ends_arrangements := 12
  let favorable_arrangements := total_arrangements - both_ends_arrangements
  let probability := favorable_arrangements / total_arrangements
  total_arrangements = 120 ∧ both_ends_arrangements = 12 ∧ favorable_arrangements = 108 ∧ probability = 0.9 :=
by
  sorry

end NUMINAMATH_GPT_probability_not_both_ends_l1596_159618


namespace NUMINAMATH_GPT_find_f_4_l1596_159652

noncomputable def f (x : ℕ) (a b c : ℕ) : ℕ := 2 * a * x + b * x + c

theorem find_f_4
  (a b c : ℕ)
  (f1 : f 1 a b c = 10)
  (f2 : f 2 a b c = 20) :
  f 4 a b c = 40 :=
sorry

end NUMINAMATH_GPT_find_f_4_l1596_159652


namespace NUMINAMATH_GPT_total_students_in_high_school_l1596_159683

-- Definitions based on the problem conditions
def freshman_students : ℕ := 400
def sample_students : ℕ := 45
def sophomore_sample_students : ℕ := 15
def senior_sample_students : ℕ := 10

-- The theorem to be proved
theorem total_students_in_high_school : (sample_students = 45) → (freshman_students = 400) → (sophomore_sample_students = 15) → (senior_sample_students = 10) → ∃ total_students : ℕ, total_students = 900 :=
by
  sorry

end NUMINAMATH_GPT_total_students_in_high_school_l1596_159683


namespace NUMINAMATH_GPT_jana_distance_travel_in_20_minutes_l1596_159628

theorem jana_distance_travel_in_20_minutes :
  ∀ (usual_pace half_pace double_pace : ℚ)
    (first_15_minutes_distance second_5_minutes_distance total_distance : ℚ),
  usual_pace = 1 / 30 →
  half_pace = usual_pace / 2 →
  double_pace = usual_pace * 2 →
  first_15_minutes_distance = 15 * half_pace →
  second_5_minutes_distance = 5 * double_pace →
  total_distance = first_15_minutes_distance + second_5_minutes_distance →
  total_distance = 7 / 12 := 
by
  intros
  sorry

end NUMINAMATH_GPT_jana_distance_travel_in_20_minutes_l1596_159628


namespace NUMINAMATH_GPT_ineq_medians_triangle_l1596_159684

theorem ineq_medians_triangle (a b c s_a s_b s_c : ℝ)
  (h_mediana : s_a = 1 / 2 * Real.sqrt (2 * b^2 + 2 * c^2 - a^2))
  (h_medianb : s_b = 1 / 2 * Real.sqrt (2 * a^2 + 2 * c^2 - b^2))
  (h_medianc : s_c = 1 / 2 * Real.sqrt (2 * a^2 + 2 * b^2 - c^2))
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a + b + c > s_a + s_b + s_c ∧ s_a + s_b + s_c > (3 / 4) * (a + b + c) := 
sorry

end NUMINAMATH_GPT_ineq_medians_triangle_l1596_159684


namespace NUMINAMATH_GPT_stratified_sampling_distribution_l1596_159654

/-- A high school has a total of 2700 students, among which there are 900 freshmen, 
1200 sophomores, and 600 juniors. Using stratified sampling, a sample of 135 students 
is drawn. Prove that the sample contains 45 freshmen, 60 sophomores, and 30 juniors --/
theorem stratified_sampling_distribution :
  let total_students := 2700
  let freshmen := 900
  let sophomores := 1200
  let juniors := 600
  let sample_size := 135
  (sample_size * freshmen / total_students = 45) ∧ 
  (sample_size * sophomores / total_students = 60) ∧ 
  (sample_size * juniors / total_students = 30) :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_distribution_l1596_159654


namespace NUMINAMATH_GPT_num_and_sum_of_divisors_of_36_l1596_159676

noncomputable def num_divisors_and_sum (n : ℕ) : ℕ × ℕ :=
  let divisors := (List.range (n + 1)).filter (λ x => n % x = 0)
  (divisors.length, divisors.sum)

theorem num_and_sum_of_divisors_of_36 : num_divisors_and_sum 36 = (9, 91) := by
  sorry

end NUMINAMATH_GPT_num_and_sum_of_divisors_of_36_l1596_159676


namespace NUMINAMATH_GPT_problem_statement_l1596_159630

-- Define a multiple of 6 and a multiple of 9
variables (a b : ℤ)
variable (ha : ∃ k, a = 6 * k)
variable (hb : ∃ k, b = 9 * k)

-- Prove that a + b is a multiple of 3
theorem problem_statement : 
  (∃ k, a + b = 3 * k) ∧ 
  ¬((∀ m n, a = 6 * m ∧ b = 9 * n → (a + b = odd))) ∧ 
  ¬(∃ k, a + b = 6 * k) ∧ 
  ¬(∃ k, a + b = 9 * k) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1596_159630


namespace NUMINAMATH_GPT_hours_worked_on_saturday_l1596_159667

-- Definitions from the problem conditions
def hourly_wage : ℝ := 15
def hours_friday : ℝ := 10
def hours_sunday : ℝ := 14
def total_earnings : ℝ := 450

-- Define number of hours worked on Saturday as a variable
variable (hours_saturday : ℝ)

-- Total earnings can be expressed as the sum of individual day earnings
def total_earnings_eq : Prop := 
  total_earnings = (hours_friday * hourly_wage) + (hours_sunday * hourly_wage) + (hours_saturday * hourly_wage)

-- Prove that the hours worked on Saturday is 6
theorem hours_worked_on_saturday :
  total_earnings_eq hours_saturday →
  hours_saturday = 6 := by
  sorry

end NUMINAMATH_GPT_hours_worked_on_saturday_l1596_159667


namespace NUMINAMATH_GPT_ivan_total_pay_l1596_159619

theorem ivan_total_pay (cost_per_card : ℕ) (number_of_cards : ℕ) (discount_per_card : ℕ) :
  cost_per_card = 12 → number_of_cards = 10 → discount_per_card = 2 →
  (number_of_cards * (cost_per_card - discount_per_card)) = 100 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_ivan_total_pay_l1596_159619


namespace NUMINAMATH_GPT_min_xy_min_x_plus_y_l1596_159695

theorem min_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : xy ≥ 64 := 
sorry

theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : x + y ≥ 18 := 
sorry

end NUMINAMATH_GPT_min_xy_min_x_plus_y_l1596_159695


namespace NUMINAMATH_GPT_function_single_intersection_l1596_159672

theorem function_single_intersection (a : ℝ) : 
  (∃ x : ℝ, ax^2 - x + 1 = 0 ∧ ∀ y : ℝ, (ax^2 - x + 1 = 0 → y = x)) ↔ (a = 0 ∨ a = 1/4) :=
sorry

end NUMINAMATH_GPT_function_single_intersection_l1596_159672


namespace NUMINAMATH_GPT_new_point_in_fourth_quadrant_l1596_159634

-- Define the initial point P with coordinates (-3, 2)
def P : ℝ × ℝ := (-3, 2)

-- Define the move operation: 4 units to the right and 6 units down
def move (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 4, p.2 - 6)

-- Define the new point after the move operation
def P' : ℝ × ℝ := move P

-- Prove that the new point P' is in the fourth quadrant
theorem new_point_in_fourth_quadrant (x y : ℝ) (h : P' = (x, y)) : x > 0 ∧ y < 0 :=
by
  sorry

end NUMINAMATH_GPT_new_point_in_fourth_quadrant_l1596_159634


namespace NUMINAMATH_GPT_boat_problem_l1596_159671

theorem boat_problem (x n : ℕ) (h1 : n = 7 * x + 5) (h2 : n = 8 * x - 2) :
  n = 54 ∧ x = 7 := by
sorry

end NUMINAMATH_GPT_boat_problem_l1596_159671


namespace NUMINAMATH_GPT_time_for_b_alone_l1596_159686

theorem time_for_b_alone (A B : ℝ) (h1 : A + B = 1 / 16) (h2 : A = 1 / 24) : B = 1 / 48 :=
by
  sorry

end NUMINAMATH_GPT_time_for_b_alone_l1596_159686


namespace NUMINAMATH_GPT_geo_seq_product_l1596_159655

theorem geo_seq_product (a : ℕ → ℝ) (r : ℝ) (h_pos : ∀ n, 0 < a n) 
  (h_geo : ∀ n, a (n + 1) = a n * r) 
  (h_roots : a 1 ^ 2 - 10 * a 1 + 16 = 0) 
  (h_root19 : a 19 ^ 2 - 10 * a 19 + 16 = 0) : 
  a 8 * a 10 * a 12 = 64 :=
by
  sorry

end NUMINAMATH_GPT_geo_seq_product_l1596_159655


namespace NUMINAMATH_GPT_b_amount_l1596_159656

-- Define the conditions
def total_amount (a b : ℝ) : Prop := a + b = 1210
def fraction_condition (a b : ℝ) : Prop := (1/3) * a = (1/4) * b

-- Define the main theorem to prove B's amount
theorem b_amount (a b : ℝ) (h1 : total_amount a b) (h2 : fraction_condition a b) : b = 691.43 :=
sorry

end NUMINAMATH_GPT_b_amount_l1596_159656


namespace NUMINAMATH_GPT_total_students_l1596_159666

-- Define n as total number of students
variable (n : ℕ)

-- Define conditions
variable (h1 : 550 ≤ n)
variable (h2 : (n / 10) + 10 ≤ n)

-- Define the proof statement
theorem total_students (h : (550 * 10 + 5) = n ∧ 
                        550 * 10 / n + 10 = 45 + n) : 
                        n = 1000 := by
  sorry

end NUMINAMATH_GPT_total_students_l1596_159666


namespace NUMINAMATH_GPT_car_speed_car_speed_correct_l1596_159602

theorem car_speed (d t s : ℝ) (hd : d = 810) (ht : t = 5) : s = d / t := 
by
  sorry

theorem car_speed_correct (d t : ℝ) (hd : d = 810) (ht : t = 5) : d / t = 162 :=
by
  sorry

end NUMINAMATH_GPT_car_speed_car_speed_correct_l1596_159602


namespace NUMINAMATH_GPT_sum_of_numbers_in_ratio_l1596_159639

theorem sum_of_numbers_in_ratio 
  (x : ℕ)
  (h : 5 * x = 560) : 
  2 * x + 3 * x + 4 * x + 5 * x = 1568 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_numbers_in_ratio_l1596_159639


namespace NUMINAMATH_GPT_redesigned_lock_additional_combinations_l1596_159632

-- Definitions for the problem conditions
def original_combinations : ℕ := Nat.choose 10 5
def total_new_combinations : ℕ := (Finset.range 10).sum (λ k => Nat.choose 10 (k + 1)) 
def additional_combinations := total_new_combinations - original_combinations - 2 -- subtract combinations for 0 and 10

-- Statement of the theorem
theorem redesigned_lock_additional_combinations : additional_combinations = 770 :=
by
  -- Proof omitted (insert 'sorry' to indicate incomplete proof state)
  sorry

end NUMINAMATH_GPT_redesigned_lock_additional_combinations_l1596_159632


namespace NUMINAMATH_GPT_inlet_pipe_rate_16_liters_per_minute_l1596_159626

noncomputable def rate_of_inlet_pipe : ℝ :=
  let capacity := 21600 -- litres
  let outlet_time_alone := 10 -- hours
  let outlet_time_with_inlet := 18 -- hours
  let outlet_rate := capacity / outlet_time_alone
  let combined_rate := capacity / outlet_time_with_inlet
  let inlet_rate := outlet_rate - combined_rate
  inlet_rate / 60 -- converting litres/hour to litres/min

theorem inlet_pipe_rate_16_liters_per_minute : rate_of_inlet_pipe = 16 :=
by
  sorry

end NUMINAMATH_GPT_inlet_pipe_rate_16_liters_per_minute_l1596_159626


namespace NUMINAMATH_GPT_total_number_of_students_l1596_159617

/-- The total number of high school students in the school given sampling constraints. -/
theorem total_number_of_students (F1 F2 F3 : ℕ) (sample_size : ℕ) (consistency_ratio : ℕ) :
  F2 = 300 ∧ sample_size = 45 ∧ (F1 / F3) = 2 ∧ 
  (20 + 10 + (sample_size - 30)) = sample_size → F1 + F2 + F3 = 900 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_students_l1596_159617


namespace NUMINAMATH_GPT_panthers_score_l1596_159645

theorem panthers_score (P : ℕ) (wildcats_score : ℕ := 36) (score_difference : ℕ := 19) (h : wildcats_score = P + score_difference) : P = 17 := by
  sorry

end NUMINAMATH_GPT_panthers_score_l1596_159645


namespace NUMINAMATH_GPT_hyperbola_eccentricity_squared_l1596_159611

/-- Given that F is the right focus of the hyperbola 
    \( C: \frac{x^2}{a^2} - \frac{y^2}{b^2} = 1 \) with \( a > 0 \) and \( b > 0 \), 
    a line perpendicular to the x-axis is drawn through point F, 
    intersecting one asymptote of the hyperbola at point M. 
    If \( |FM| = 2a \), denote the eccentricity of the hyperbola as \( e \). 
    Prove that \( e^2 = \frac{1 + \sqrt{17}}{2} \).
 -/
theorem hyperbola_eccentricity_squared (a b c : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3: c^2 = a^2 + b^2) (h4: b * c = 2 * a^2) : 
  (c / a)^2 = (1 + Real.sqrt 17) / 2 := 
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_squared_l1596_159611


namespace NUMINAMATH_GPT_carlton_school_earnings_l1596_159633

theorem carlton_school_earnings :
  let students_days_adams := 8 * 4
  let students_days_byron := 5 * 6
  let students_days_carlton := 6 * 10
  let total_wages := 1092
  students_days_adams + students_days_byron = 62 → 
  62 * (2 * x) + students_days_carlton * x = total_wages → 
  x = (total_wages : ℝ) / 184 → 
  (students_days_carlton : ℝ) * x = 356.09 := 
by
  intros _ _ _ 
  sorry

end NUMINAMATH_GPT_carlton_school_earnings_l1596_159633


namespace NUMINAMATH_GPT_tan_product_l1596_159647

open Real

theorem tan_product (x y : ℝ) 
(h1 : sin x * sin y = 24 / 65) 
(h2 : cos x * cos y = 48 / 65) :
tan x * tan y = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_tan_product_l1596_159647


namespace NUMINAMATH_GPT_solve_inequality_l1596_159604

noncomputable def log_b (b x : ℝ) := Real.log x / Real.log b

theorem solve_inequality (x : ℝ) (hx : x ≠ 0 ∧ 0 < x) :
  (64 + (log_b (1/5) (x^2))^3) / (log_b (1/5) (x^6) * log_b 5 (x^2) + 5 * log_b 5 (x^6) + 14 * log_b (1/5) (x^2) + 2) ≤ 0 ↔
  (x ∈ Set.Icc (-25 : ℝ) (- Real.sqrt 5)) ∨
  (x ∈ Set.Icc (- (Real.exp (Real.log 5 / 3))) 0) ∨
  (x ∈ Set.Icc 0 (Real.exp (Real.log 5 / 3))) ∨
  (x ∈ Set.Icc (Real.sqrt 5) 25) :=
by 
  sorry

end NUMINAMATH_GPT_solve_inequality_l1596_159604


namespace NUMINAMATH_GPT_min_folds_exceed_12mm_l1596_159641

theorem min_folds_exceed_12mm : ∃ n : ℕ, 0.1 * (2: ℝ)^n > 12 ∧ ∀ m < n, 0.1 * (2: ℝ)^m ≤ 12 := 
by
  sorry

end NUMINAMATH_GPT_min_folds_exceed_12mm_l1596_159641


namespace NUMINAMATH_GPT_bottle_caps_left_l1596_159675

theorem bottle_caps_left {init_caps given_away_rebecca given_away_michael left_caps : ℝ} 
  (h1 : init_caps = 143.6)
  (h2 : given_away_rebecca = 89.2)
  (h3 : given_away_michael = 16.7)
  (h4 : left_caps = init_caps - (given_away_rebecca + given_away_michael)) :
  left_caps = 37.7 := by
  sorry

end NUMINAMATH_GPT_bottle_caps_left_l1596_159675


namespace NUMINAMATH_GPT_angle_sum_x_y_l1596_159678

def angle_A := 36
def angle_B := 80
def angle_C := 24

def target_sum : ℕ := 140

theorem angle_sum_x_y (angle_A angle_B angle_C : ℕ) (x y : ℕ) : 
  angle_A = 36 → angle_B = 80 → angle_C = 24 → x + y = 140 := by 
  intros _ _ _
  sorry

end NUMINAMATH_GPT_angle_sum_x_y_l1596_159678


namespace NUMINAMATH_GPT_bernardo_winning_N_initial_bernardo_smallest_N_sum_of_digits_34_l1596_159646

def bernardo (x : ℕ) : ℕ := 2 * x
def silvia (x : ℕ) : ℕ := x + 30

theorem bernardo_winning_N_initial (N : ℕ) :
  (∃ k : ℕ, (bernardo $ silvia $ bernardo $ silvia $ bernardo $ silvia $ bernardo $ silvia N) = k
  ∧ 950 ≤ k ∧ k ≤ 999)
  → 34 ≤ N ∧ N ≤ 35 :=
by
  sorry

theorem bernardo_smallest_N (N : ℕ) (h : 34 ≤ N ∧ N ≤ 35) :
  (N = 34) :=
by
  sorry

theorem sum_of_digits_34 :
  (3 + 4 = 7) :=
by
  sorry

end NUMINAMATH_GPT_bernardo_winning_N_initial_bernardo_smallest_N_sum_of_digits_34_l1596_159646


namespace NUMINAMATH_GPT_positive_difference_abs_eq_l1596_159607

theorem positive_difference_abs_eq (x₁ x₂ : ℝ) (h₁ : x₁ - 3 = 15) (h₂ : x₂ - 3 = -15) : x₁ - x₂ = 30 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_abs_eq_l1596_159607


namespace NUMINAMATH_GPT_complement_of_A_with_respect_to_U_l1596_159613

open Set

def U : Set ℕ := {3, 4, 5, 6}
def A : Set ℕ := {3, 5}
def complement_U_A : Set ℕ := {4, 6}

theorem complement_of_A_with_respect_to_U :
  U \ A = complement_U_A := by
  sorry

end NUMINAMATH_GPT_complement_of_A_with_respect_to_U_l1596_159613


namespace NUMINAMATH_GPT_range_of_k_is_l1596_159621

noncomputable def range_of_k (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : Set ℝ :=
{k : ℝ | ∀ x : ℝ, a^x + 4 * a^(-x) - k > 0}

theorem range_of_k_is (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  range_of_k a h₁ h₂ = { k : ℝ | k < 4 ∧ k ≠ 3 } :=
sorry

end NUMINAMATH_GPT_range_of_k_is_l1596_159621


namespace NUMINAMATH_GPT_find_greatest_K_l1596_159693

theorem find_greatest_K {u v w K : ℝ} (hu : u > 0) (hv : v > 0) (hw : w > 0) (hu2_gt_4vw : u^2 > 4 * v * w) :
  (u^2 - 4 * v * w)^2 > K * (2 * v^2 - u * w) * (2 * w^2 - u * v) ↔ K ≤ 16 := 
sorry

end NUMINAMATH_GPT_find_greatest_K_l1596_159693


namespace NUMINAMATH_GPT_intersection_complement_l1596_159697

def A : Set ℝ := { x | x^2 ≤ 4 * x }
def B : Set ℝ := { x | ∃ y, y = Real.sqrt (x - 3) }

theorem intersection_complement (x : ℝ) : 
  x ∈ A ∩ (Set.univ \ B) ↔ x ∈ Set.Ico 0 3 := 
sorry

end NUMINAMATH_GPT_intersection_complement_l1596_159697


namespace NUMINAMATH_GPT_line_does_not_intersect_circle_l1596_159616

theorem line_does_not_intersect_circle (a : ℝ) : 
  (a > 1 ∨ a < -1) → ¬ ∃ (x y : ℝ), (x + y = a) ∧ (x^2 + y^2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_line_does_not_intersect_circle_l1596_159616


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_sqrt_28_l1596_159635

theorem sum_of_consecutive_integers_sqrt_28 (a b : ℤ) (h1 : b = a + 1) (h2 : a < Real.sqrt 28) (h3 : Real.sqrt 28 < b) : a + b = 11 :=
by 
    sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_sqrt_28_l1596_159635


namespace NUMINAMATH_GPT_arianna_sleep_hours_l1596_159665

-- Defining the given conditions
def total_hours_in_a_day : ℕ := 24
def hours_at_work : ℕ := 6
def hours_in_class : ℕ := 3
def hours_at_gym : ℕ := 2
def hours_on_chores : ℕ := 5

-- Formulating the total hours spent on activities
def total_hours_on_activities := hours_at_work + hours_in_class + hours_at_gym + hours_on_chores

-- Proving Arianna's sleep hours
theorem arianna_sleep_hours : total_hours_in_a_day - total_hours_on_activities = 8 :=
by
  -- Direct proof placeholder, to be filled in with actual proof steps or tactic
  sorry

end NUMINAMATH_GPT_arianna_sleep_hours_l1596_159665


namespace NUMINAMATH_GPT_additional_tickets_won_l1596_159698

-- Definitions from the problem
def initial_tickets : ℕ := 49
def spent_tickets : ℕ := 25
def final_tickets : ℕ := 30

-- The main statement we need to prove
theorem additional_tickets_won (initial_tickets : ℕ) (spent_tickets : ℕ) (final_tickets : ℕ) : 
  final_tickets - (initial_tickets - spent_tickets) = 6 :=
by
  sorry

end NUMINAMATH_GPT_additional_tickets_won_l1596_159698


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l1596_159649

def point : ℝ × ℝ := (1, -2)

def is_fourth_quadrant (p: ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : is_fourth_quadrant point :=
by
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l1596_159649


namespace NUMINAMATH_GPT_number_of_friends_l1596_159638

def total_gold := 100
def lost_gold := 20
def gold_per_friend := 20

theorem number_of_friends :
  (total_gold - lost_gold) / gold_per_friend = 4 := by
  sorry

end NUMINAMATH_GPT_number_of_friends_l1596_159638


namespace NUMINAMATH_GPT_rabbit_carrots_l1596_159629

theorem rabbit_carrots (r f : ℕ) (hr : 3 * r = 5 * f) (hf : f = r - 6) : 3 * r = 45 :=
by
  sorry

end NUMINAMATH_GPT_rabbit_carrots_l1596_159629


namespace NUMINAMATH_GPT_obtuse_dihedral_angles_l1596_159648

theorem obtuse_dihedral_angles (AOB BOC COA : ℝ) (h1 : AOB > 90) (h2 : BOC > 90) (h3 : COA > 90) :
  ∃ α β γ : ℝ, α > 90 ∧ β > 90 ∧ γ > 90 :=
sorry

end NUMINAMATH_GPT_obtuse_dihedral_angles_l1596_159648


namespace NUMINAMATH_GPT_points_per_draw_l1596_159627

-- Definitions based on conditions
def total_games : ℕ := 20
def wins : ℕ := 14
def losses : ℕ := 2
def total_points : ℕ := 46
def points_per_win : ℕ := 3
def points_per_loss : ℕ := 0

-- Calculation of the number of draws and points per draw
def draws : ℕ := total_games - wins - losses
def points_wins : ℕ := wins * points_per_win
def points_draws : ℕ := total_points - points_wins

-- Theorem statement
theorem points_per_draw : points_draws / draws = 1 := by
  sorry

end NUMINAMATH_GPT_points_per_draw_l1596_159627


namespace NUMINAMATH_GPT_find_a4_l1596_159680

variable {a_n : ℕ → ℕ}
variable {S : ℕ → ℕ}

def is_arithmetic_sequence (a_n : ℕ → ℕ) :=
  ∃ d : ℕ, ∀ n : ℕ, a_n (n + 1) = a_n n + d

def sum_first_n_terms (S : ℕ → ℕ) (a_n : ℕ → ℕ) :=
  ∀ n : ℕ, S n = (n * (a_n 1 + a_n n)) / 2

theorem find_a4 (h : S 7 = 35) (hs : sum_first_n_terms S a_n) (ha : is_arithmetic_sequence a_n) : a_n 4 = 5 := 
  by sorry

end NUMINAMATH_GPT_find_a4_l1596_159680


namespace NUMINAMATH_GPT_algebraic_expression_transformation_l1596_159661

theorem algebraic_expression_transformation (a b : ℝ) (h : ∀ x : ℝ, x^2 - 6*x + b = (x - a)^2 - 1) : b - a = 5 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_transformation_l1596_159661


namespace NUMINAMATH_GPT_transform_equation_l1596_159690

theorem transform_equation (x : ℝ) :
  x^2 + 4 * x + 1 = 0 → (x + 2)^2 = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_transform_equation_l1596_159690


namespace NUMINAMATH_GPT_half_AB_equals_l1596_159609

-- Define vectors OA and OB
def vector_OA : ℝ × ℝ := (3, 2)
def vector_OB : ℝ × ℝ := (4, 7)

-- Prove that (1 / 2) * (OB - OA) = (1 / 2, 5 / 2)
theorem half_AB_equals :
  (1 / 2 : ℝ) • ((vector_OB.1 - vector_OA.1), (vector_OB.2 - vector_OA.2)) = (1 / 2, 5 / 2) := 
  sorry

end NUMINAMATH_GPT_half_AB_equals_l1596_159609


namespace NUMINAMATH_GPT_complex_i_power_l1596_159614

theorem complex_i_power (i : ℂ) (h1 : i^2 = -1) (h2 : i^3 = -i) (h3 : i^4 = 1) : i^2015 = -i := 
by
  sorry

end NUMINAMATH_GPT_complex_i_power_l1596_159614


namespace NUMINAMATH_GPT_bud_age_uncle_age_relation_l1596_159601

variable (bud_age uncle_age : Nat)

theorem bud_age_uncle_age_relation (h : bud_age = 8) (h0 : bud_age = uncle_age / 3) : uncle_age = 24 := by
  sorry

end NUMINAMATH_GPT_bud_age_uncle_age_relation_l1596_159601


namespace NUMINAMATH_GPT_parallel_lines_slope_l1596_159642

theorem parallel_lines_slope (b : ℚ) :
  (∀ x y : ℚ, 3 * y + x - 1 = 0 → 2 * y + b * x - 4 = 0 ∨
    3 * y + x - 1 = 0 ∧ 2 * y + b * x - 4 = 0) →
  b = 2 / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_l1596_159642
