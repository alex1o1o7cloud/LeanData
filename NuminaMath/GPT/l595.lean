import Mathlib

namespace NUMINAMATH_GPT_laticia_total_pairs_l595_59573

-- Definitions of the conditions about the pairs of socks knitted each week

-- Number of pairs knitted in the first week
def pairs_week1 : ℕ := 12

-- Number of pairs knitted in the second week
def pairs_week2 : ℕ := pairs_week1 + 4

-- Number of pairs knitted in the third week
def pairs_week3 : ℕ := (pairs_week1 + pairs_week2) / 2

-- Number of pairs knitted in the fourth week
def pairs_week4 : ℕ := pairs_week3 - 3

-- Statement: Sum of pairs over the four weeks
theorem laticia_total_pairs :
  pairs_week1 + pairs_week2 + pairs_week3 + pairs_week4 = 53 := by
  sorry

end NUMINAMATH_GPT_laticia_total_pairs_l595_59573


namespace NUMINAMATH_GPT_even_integers_between_sqrt_10_and_sqrt_100_l595_59569

theorem even_integers_between_sqrt_10_and_sqrt_100 : 
  ∃ (n : ℕ), n = 4 ∧ (∀ (a : ℕ), (∃ k, (2 * k = a ∧ a > Real.sqrt 10 ∧ a < Real.sqrt 100)) ↔ 
  (a = 4 ∨ a = 6 ∨ a = 8 ∨ a = 10)) := 
by 
  sorry

end NUMINAMATH_GPT_even_integers_between_sqrt_10_and_sqrt_100_l595_59569


namespace NUMINAMATH_GPT_inverse_function_log3_l595_59597

noncomputable def f (x : ℝ) : ℝ := Real.logb 3 x

theorem inverse_function_log3 :
  ∀ x : ℝ, x > 0 →
  ∃ y : ℝ, f (3 ^ y) = y := 
sorry

end NUMINAMATH_GPT_inverse_function_log3_l595_59597


namespace NUMINAMATH_GPT_total_fruit_pieces_correct_l595_59583

/-
  Define the quantities of each type of fruit.
-/
def red_apples : Nat := 9
def green_apples : Nat := 4
def purple_grapes : Nat := 3
def yellow_bananas : Nat := 6
def orange_oranges : Nat := 2

/-
  The total number of fruit pieces in the basket.
-/
def total_fruit_pieces : Nat := red_apples + green_apples + purple_grapes + yellow_bananas + orange_oranges

/-
  Prove that the total number of fruit pieces is 24.
-/
theorem total_fruit_pieces_correct : total_fruit_pieces = 24 := by
  sorry

end NUMINAMATH_GPT_total_fruit_pieces_correct_l595_59583


namespace NUMINAMATH_GPT_final_number_is_odd_l595_59542

theorem final_number_is_odd : 
  ∃ (n : ℤ), n % 2 = 1 ∧ n ≥ 1 ∧ n < 1024 := sorry

end NUMINAMATH_GPT_final_number_is_odd_l595_59542


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l595_59535

theorem sufficient_not_necessary_condition (a : ℝ)
  : (∃ x : ℝ, a * x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, a ≥ 0 ∨ a * x^2 + x + 1 ≥ 0)
:= sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l595_59535


namespace NUMINAMATH_GPT_find_minimum_value_l595_59521

-- This definition captures the condition that a, b, c are positive real numbers
def pos_reals := { x : ℝ // 0 < x }

-- The main theorem statement
theorem find_minimum_value (a b c : pos_reals) :
  4 * (a.1 ^ 4) + 8 * (b.1 ^ 4) + 16 * (c.1 ^ 4) + 1 / (a.1 * b.1 * c.1) ≥ 10 :=
by
  -- This is where the proof will go
  sorry

end NUMINAMATH_GPT_find_minimum_value_l595_59521


namespace NUMINAMATH_GPT_range_of_a_l595_59534

-- Define the function f
def f (a x : ℝ) : ℝ := a * x ^ 2 + a * x - 1

-- State the problem
theorem range_of_a (a : ℝ) : (∀ x : ℝ, f a x < 0) ↔ -4 < a ∧ a ≤ 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_a_l595_59534


namespace NUMINAMATH_GPT_least_number_of_tiles_l595_59501

/-- A room of 544 cm long and 374 cm broad is to be paved with square tiles. 
    Prove that the least number of square tiles required to cover the floor is 176. -/
theorem least_number_of_tiles (length breadth : ℕ) (h1 : length = 544) (h2 : breadth = 374) :
  let gcd_length_breadth := Nat.gcd length breadth
  let num_tiles_length := length / gcd_length_breadth
  let num_tiles_breadth := breadth / gcd_length_breadth
  num_tiles_length * num_tiles_breadth = 176 :=
by
  sorry

end NUMINAMATH_GPT_least_number_of_tiles_l595_59501


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l595_59554

-- System (1)
theorem system1_solution {x y : ℝ} : 
  x + y = 3 → 
  x - y = 1 → 
  (x = 2 ∧ y = 1) :=
by
  intros h1 h2
  -- proof goes here
  sorry

-- System (2)
theorem system2_solution {x y : ℝ} :
  2 * x + y = 3 →
  x - 2 * y = 1 →
  (x = 7 / 5 ∧ y = 1 / 5) :=
by
  intros h1 h2
  -- proof goes here
  sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l595_59554


namespace NUMINAMATH_GPT_first_cube_weight_l595_59503

-- Given definitions of cubes and their relationships
def weight_of_cube (s : ℝ) (weight : ℝ) : Prop :=
  ∃ v : ℝ, v = s^3 ∧ weight = v

def cube_relationship (s1 s2 weight2 : ℝ) : Prop :=
  s2 = 2 * s1 ∧ weight2 = 32

-- The proof problem
theorem first_cube_weight (s1 s2 weight1 weight2 : ℝ) (h1 : cube_relationship s1 s2 weight2) : weight1 = 4 :=
  sorry

end NUMINAMATH_GPT_first_cube_weight_l595_59503


namespace NUMINAMATH_GPT_intersection_M_N_l595_59544

def set_M : Set ℝ := { x | x * (x - 1) ≤ 0 }
def set_N : Set ℝ := { x | x < 1 }

theorem intersection_M_N : set_M ∩ set_N = { x | 0 ≤ x ∧ x < 1 } := sorry

end NUMINAMATH_GPT_intersection_M_N_l595_59544


namespace NUMINAMATH_GPT_inequality_proof_l595_59545

variable (a : ℝ)

theorem inequality_proof (a : ℝ) : 
  (a^2 + a + 2) / (Real.sqrt (a^2 + a + 1)) ≥ 2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l595_59545


namespace NUMINAMATH_GPT_cos_frac_less_sin_frac_l595_59561

theorem cos_frac_less_sin_frac : 
  let a := Real.cos (3 / 2)
  let b := Real.sin (1 / 10)
  a < b :=
by
  let a := Real.cos (3 / 2)
  let b := Real.sin (1 / 10)
  sorry -- proof skipped

end NUMINAMATH_GPT_cos_frac_less_sin_frac_l595_59561


namespace NUMINAMATH_GPT_john_new_weekly_earnings_l595_59516

theorem john_new_weekly_earnings
  (original_earnings : ℕ)
  (percentage_increase : ℕ)
  (raise_amount : ℕ)
  (new_weekly_earnings : ℕ)
  (original_earnings_eq : original_earnings = 50)
  (percentage_increase_eq : percentage_increase = 40)
  (raise_amount_eq : raise_amount = original_earnings * percentage_increase / 100)
  (new_weekly_earnings_eq : new_weekly_earnings = original_earnings + raise_amount) :
  new_weekly_earnings = 70 := by
  sorry

end NUMINAMATH_GPT_john_new_weekly_earnings_l595_59516


namespace NUMINAMATH_GPT_number_of_girls_l595_59579

-- Definitions from the problem conditions
def ratio_girls_boys (g b : ℕ) : Prop := 4 * b = 3 * g
def total_students (g b : ℕ) : Prop := g + b = 56

-- The proof statement
theorem number_of_girls (g b k : ℕ) (hg : 4 * k = g) (hb : 3 * k = b) (hr : ratio_girls_boys g b) (ht : total_students g b) : g = 32 :=
by sorry

end NUMINAMATH_GPT_number_of_girls_l595_59579


namespace NUMINAMATH_GPT_part_a_area_of_square_l595_59563

theorem part_a_area_of_square {s : ℝ} (h : s = 9) : s ^ 2 = 81 := 
sorry

end NUMINAMATH_GPT_part_a_area_of_square_l595_59563


namespace NUMINAMATH_GPT_sum_coefficients_eq_neg_one_l595_59571

theorem sum_coefficients_eq_neg_one (a a1 a2 a3 a4 a5 : ℝ) :
  (∀ x y : ℝ, (x - 2 * y)^5 = a * x^5 + a1 * x^4 * y + a2 * x^3 * y^2 + a3 * x^2 * y^3 + a4 * x * y^4 + a5 * y^5) →
  a + a1 + a2 + a3 + a4 + a5 = -1 :=
by
  sorry

end NUMINAMATH_GPT_sum_coefficients_eq_neg_one_l595_59571


namespace NUMINAMATH_GPT_cupcakes_left_l595_59537

theorem cupcakes_left (initial_cupcakes : ℕ)
  (students_delmont : ℕ) (ms_delmont : ℕ)
  (students_donnelly : ℕ) (mrs_donnelly : ℕ)
  (school_nurse : ℕ) (school_principal : ℕ) (school_custodians : ℕ)
  (favorite_teachers : ℕ) (cupcakes_per_favorite_teacher : ℕ)
  (other_classmates : ℕ) :
  initial_cupcakes = 80 →
  students_delmont = 18 → ms_delmont = 1 →
  students_donnelly = 16 → mrs_donnelly = 1 →
  school_nurse = 1 → school_principal = 1 → school_custodians = 3 →
  favorite_teachers = 5 → cupcakes_per_favorite_teacher = 2 → 
  other_classmates = 10 →
  initial_cupcakes - (students_delmont + ms_delmont +
                      students_donnelly + mrs_donnelly +
                      school_nurse + school_principal + school_custodians +
                      favorite_teachers * cupcakes_per_favorite_teacher +
                      other_classmates) = 19 :=
by
  intros _ _ _ _ _ _ _ _ _ _ _
  sorry

end NUMINAMATH_GPT_cupcakes_left_l595_59537


namespace NUMINAMATH_GPT_speed_of_stream_l595_59507

theorem speed_of_stream
  (b s : ℝ)
  (H1 : 120 = 2 * (b + s))
  (H2 : 60 = 2 * (b - s)) :
  s = 15 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_stream_l595_59507


namespace NUMINAMATH_GPT_relationship_between_abc_l595_59596

noncomputable def a : Real := (2 / 5) ^ (3 / 5)
noncomputable def b : Real := (2 / 5) ^ (2 / 5)
noncomputable def c : Real := (3 / 5) ^ (3 / 5)

theorem relationship_between_abc : a < b ∧ b < c := by
  sorry

end NUMINAMATH_GPT_relationship_between_abc_l595_59596


namespace NUMINAMATH_GPT_remainder_when_13_add_x_div_31_eq_22_l595_59512

open BigOperators

theorem remainder_when_13_add_x_div_31_eq_22
  (x : ℕ) (hx : x > 0) (hmod : 7 * x ≡ 1 [MOD 31]) :
  (13 + x) % 31 = 22 := 
  sorry

end NUMINAMATH_GPT_remainder_when_13_add_x_div_31_eq_22_l595_59512


namespace NUMINAMATH_GPT_degree_le_three_l595_59508

theorem degree_le_three
  (d : ℕ)
  (P : Polynomial ℤ)
  (hdeg : P.degree = d)
  (hP : ∃ (S : Finset ℤ), (S.card ≥ d + 1) ∧ ∀ m ∈ S, |P.eval m| = 1) :
  d ≤ 3 := 
sorry

end NUMINAMATH_GPT_degree_le_three_l595_59508


namespace NUMINAMATH_GPT_odd_consecutive_nums_divisibility_l595_59598

theorem odd_consecutive_nums_divisibility (a b : ℕ) (h_consecutive : b = a + 2) (h_odd_a : a % 2 = 1) (h_odd_b : b % 2 = 1) : (a^b + b^a) % (a + b) = 0 := by
  sorry

end NUMINAMATH_GPT_odd_consecutive_nums_divisibility_l595_59598


namespace NUMINAMATH_GPT_positive_inequality_l595_59506

theorem positive_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a^2 + b^2) / (2 * a^5 * b^5) + 81 * (a^2 * b^2) / 4 + 9 * a * b > 18 := 
  sorry

end NUMINAMATH_GPT_positive_inequality_l595_59506


namespace NUMINAMATH_GPT_min_value_of_parabola_l595_59558

theorem min_value_of_parabola : ∃ x : ℝ, ∀ y : ℝ, y = 3 * x^2 - 18 * x + 244 → y = 217 := by
  sorry

end NUMINAMATH_GPT_min_value_of_parabola_l595_59558


namespace NUMINAMATH_GPT_fraction_area_of_triangles_l595_59510

theorem fraction_area_of_triangles 
  (base_PQR : ℝ) (height_PQR : ℝ)
  (base_XYZ : ℝ) (height_XYZ : ℝ)
  (h_base_PQR : base_PQR = 3)
  (h_height_PQR : height_PQR = 2)
  (h_base_XYZ : base_XYZ = 6)
  (h_height_XYZ : height_XYZ = 3) :
  (1/2 * base_PQR * height_PQR) / (1/2 * base_XYZ * height_XYZ) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_area_of_triangles_l595_59510


namespace NUMINAMATH_GPT_math_problem_l595_59536

-- Definitions for the conditions
def condition1 (a b c : ℝ) : Prop := a + b + c = 0
def condition2 (a b c : ℝ) : Prop := |a| > |b| ∧ |b| > |c|

-- Theorem statement
theorem math_problem (a b c : ℝ) (h1 : condition1 a b c) (h2 : condition2 a b c) : c > 0 ∧ a < 0 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l595_59536


namespace NUMINAMATH_GPT_ratio_of_shaded_area_l595_59574

-- Definitions
variable (S : Type) [Field S]
variable (square_area shaded_area : S) -- Areas of the square and the shaded regions.
variable (PX XQ : S) -- Lengths such that PX = 3 * XQ.

-- Conditions
axiom condition1 : PX = 3 * XQ
axiom condition2 : shaded_area / square_area = 0.375

-- Goal
theorem ratio_of_shaded_area (PX XQ square_area shaded_area : S) [Field S] 
  (condition1 : PX = 3 * XQ)
  (condition2 : shaded_area / square_area = 0.375) : shaded_area / square_area = 0.375 := 
  by
  sorry

end NUMINAMATH_GPT_ratio_of_shaded_area_l595_59574


namespace NUMINAMATH_GPT_cuboid_third_edge_l595_59562

theorem cuboid_third_edge (a b V h : ℝ) (ha : a = 4) (hb : b = 4) (hV : V = 96) (volume_formula : V = a * b * h) : h = 6 :=
by
  sorry

end NUMINAMATH_GPT_cuboid_third_edge_l595_59562


namespace NUMINAMATH_GPT_negation_proposition_l595_59578

variable (n : ℕ)
variable (n_positive : n > 0)
variable (f : ℕ → ℕ)
variable (H1 : ∀ n, n > 0 → (f n) > 0 ∧ (f n) ≤ n)

theorem negation_proposition :
  (∃ n_0, n_0 > 0 ∧ ((f n_0) ≤ 0 ∨ (f n_0) > n_0)) ↔ ¬(∀ n, n > 0 → (f n) >0 ∧ (f n) ≤ n) :=
by 
  sorry

end NUMINAMATH_GPT_negation_proposition_l595_59578


namespace NUMINAMATH_GPT_norm_of_w_l595_59530

variable (u v : EuclideanSpace ℝ (Fin 2)) 
variable (hu : ‖u‖ = 3) (hv : ‖v‖ = 5) 
variable (h_orthogonal : inner u v = 0)

theorem norm_of_w :
  ‖4 • u - 2 • v‖ = 2 * Real.sqrt 61 := by
  sorry

end NUMINAMATH_GPT_norm_of_w_l595_59530


namespace NUMINAMATH_GPT_sum_of_binary_digits_of_315_l595_59541

theorem sum_of_binary_digits_of_315 : 
    (Nat.digits 2 315).sum = 6 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_binary_digits_of_315_l595_59541


namespace NUMINAMATH_GPT_solve_for_y_l595_59570

theorem solve_for_y (x y : ℝ) : 3 * x + 5 * y = 10 → y = 2 - (3 / 5) * x :=
by 
  -- proof steps would be filled here
  sorry

end NUMINAMATH_GPT_solve_for_y_l595_59570


namespace NUMINAMATH_GPT_simplify_expression_l595_59588

variable (x y : ℝ)

theorem simplify_expression : 
  (3 * x^2 * y - 2 * x * y^2) - (x * y^2 - 2 * x^2 * y) - 2 * (-3 * x^2 * y - x * y^2) = 26 := by
  -- Given conditions
  let x := -1
  let y := 2
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_simplify_expression_l595_59588


namespace NUMINAMATH_GPT_addition_addends_l595_59522

theorem addition_addends (a b : ℕ) (c₁ c₂ : ℕ) (d : ℕ) : 
  a + b = c₁ ∧ a + (b - d) = c₂ ∧ d = 50 ∧ c₁ = 982 ∧ c₂ = 577 → 
  a = 450 ∧ b = 532 :=
by
  sorry

end NUMINAMATH_GPT_addition_addends_l595_59522


namespace NUMINAMATH_GPT_decompose_one_into_five_unit_fractions_l595_59555

theorem decompose_one_into_five_unit_fractions :
  1 = (1/2) + (1/3) + (1/7) + (1/43) + (1/1806) :=
by
  sorry

end NUMINAMATH_GPT_decompose_one_into_five_unit_fractions_l595_59555


namespace NUMINAMATH_GPT_tiffany_cans_at_end_of_week_l595_59559

theorem tiffany_cans_at_end_of_week:
  (4 + 2.5 - 1.25 + 0 + 3.75 - 1.5 + 0 = 7.5) :=
by
  sorry

end NUMINAMATH_GPT_tiffany_cans_at_end_of_week_l595_59559


namespace NUMINAMATH_GPT_min_moves_move_stack_from_A_to_F_l595_59513

theorem min_moves_move_stack_from_A_to_F : 
  ∀ (squares : Fin 6) (stack : Fin 15), 
  (∃ moves : Nat, 
    (moves >= 0) ∧ 
    (moves == 49) ∧
    ∀ (a b : Fin 6), 
        ∃ (piece_from : Fin 15) (piece_to : Fin 15), 
        ((piece_from > piece_to) → (a ≠ b)) ∧
        (a == 0) ∧ 
        (b == 5)) :=
sorry

end NUMINAMATH_GPT_min_moves_move_stack_from_A_to_F_l595_59513


namespace NUMINAMATH_GPT_smallest_positive_integer_cube_ends_544_l595_59550

theorem smallest_positive_integer_cube_ends_544 : ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 544 ∧ ∀ m : ℕ, m > 0 ∧ m^3 % 1000 = 544 → m ≥ n :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_cube_ends_544_l595_59550


namespace NUMINAMATH_GPT_purely_imaginary_necessary_not_sufficient_l595_59590

-- Definition of a purely imaginary number
def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem purely_imaginary_necessary_not_sufficient (a b : ℝ) :
  a = 0 → (z : ℂ) = ⟨a, b⟩ → is_purely_imaginary z ↔ (a = 0 ∧ b ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_purely_imaginary_necessary_not_sufficient_l595_59590


namespace NUMINAMATH_GPT_female_students_selected_l595_59546

theorem female_students_selected (males females : ℕ) (p : ℚ) (h_males : males = 28)
  (h_females : females = 21) (h_p : p = 1 / 7) : females * p = 3 := by 
  sorry

end NUMINAMATH_GPT_female_students_selected_l595_59546


namespace NUMINAMATH_GPT_A_and_B_work_together_for_49_days_l595_59511

variable (A B : ℝ)
variable (d : ℝ)
variable (fraction_left : ℝ)

def work_rate_A := 1 / 15
def work_rate_B := 1 / 20
def combined_work_rate := work_rate_A + work_rate_B

def fraction_work_completed (d : ℝ) := combined_work_rate * d

theorem A_and_B_work_together_for_49_days
    (A : ℝ := 1 / 15)
    (B : ℝ := 1 / 20)
    (fraction_left : ℝ := 0.18333333333333335) :
    (d : ℝ) → (fraction_work_completed d = 1 - fraction_left) →
    d = 49 :=
by
  sorry

end NUMINAMATH_GPT_A_and_B_work_together_for_49_days_l595_59511


namespace NUMINAMATH_GPT_range_of_a_l595_59552

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 1 ≤ x ∧ x ≤ a}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

-- The theorem we need to prove
theorem range_of_a {a : ℝ} (h : A a ⊆ B) : 1 ≤ a ∧ a < 5 := 
sorry

end NUMINAMATH_GPT_range_of_a_l595_59552


namespace NUMINAMATH_GPT_five_by_five_rectangles_l595_59584

def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem five_by_five_rectangles : (choose 5 2) * (choose 5 2) = 100 :=
by
  sorry

end NUMINAMATH_GPT_five_by_five_rectangles_l595_59584


namespace NUMINAMATH_GPT_no_such_function_exists_l595_59523

theorem no_such_function_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = f (n + 1) - f n :=
by
  sorry

end NUMINAMATH_GPT_no_such_function_exists_l595_59523


namespace NUMINAMATH_GPT_solve_equation_l595_59580

theorem solve_equation (x : ℝ) (h : (x - 60) / 3 = (4 - 3 * x) / 6) : x = 124 / 5 := by
  sorry

end NUMINAMATH_GPT_solve_equation_l595_59580


namespace NUMINAMATH_GPT_distance_to_school_l595_59533

variable (v d : ℝ) -- typical speed (v) and distance (d)

theorem distance_to_school :
  (30 / 60 : ℝ) = 1 / 2 ∧ -- 30 minutes is 1/2 hour
  (18 / 60 : ℝ) = 3 / 10 ∧ -- 18 minutes is 3/10 hour
  d = v * (1 / 2) ∧ -- distance for typical day
  d = (v + 12) * (3 / 10) -- distance for quieter day
  → d = 9 := sorry

end NUMINAMATH_GPT_distance_to_school_l595_59533


namespace NUMINAMATH_GPT_inequality_implication_l595_59527

theorem inequality_implication (x : ℝ) : 3 * x + 4 < 5 * x - 6 → x > 5 := 
by {
  sorry
}

end NUMINAMATH_GPT_inequality_implication_l595_59527


namespace NUMINAMATH_GPT_probability_divisible_by_three_l595_59575

noncomputable def prob_divisible_by_three : ℚ :=
  1 - (4/6)^6

theorem probability_divisible_by_three :
  prob_divisible_by_three = 665 / 729 :=
by
  sorry

end NUMINAMATH_GPT_probability_divisible_by_three_l595_59575


namespace NUMINAMATH_GPT_find_a_l595_59576

/-- The random variable ξ takes on all possible values 1, 2, 3, 4, 5,
and P(ξ = k) = a * k for k = 1, 2, 3, 4, 5. Given that the sum 
of probabilities for all possible outcomes of a discrete random
variable equals 1, find the value of a. -/
theorem find_a (a : ℝ) 
  (h : (a * 1) + (a * 2) + (a * 3) + (a * 4) + (a * 5) = 1) : 
  a = 1 / 15 :=
sorry

end NUMINAMATH_GPT_find_a_l595_59576


namespace NUMINAMATH_GPT_missing_number_in_proportion_l595_59581

/-- Given the proportion 2 : 5 = x : 3.333333333333333, prove that the missing number x is 1.3333333333333332 -/
theorem missing_number_in_proportion : ∃ x, (2 / 5 = x / 3.333333333333333) ∧ x = 1.3333333333333332 :=
  sorry

end NUMINAMATH_GPT_missing_number_in_proportion_l595_59581


namespace NUMINAMATH_GPT_ratio_of_engineers_to_designers_l595_59518

-- Definitions of the variables
variables (e d : ℕ)

-- Conditions:
-- 1. The average age of the group is 45
-- 2. The average age of engineers is 40
-- 3. The average age of designers is 55

theorem ratio_of_engineers_to_designers (h : (40 * e + 55 * d) / (e + d) = 45) : e / d = 2 :=
by
-- Placeholder for the proof
sorry

end NUMINAMATH_GPT_ratio_of_engineers_to_designers_l595_59518


namespace NUMINAMATH_GPT_abc_sum_eq_11_sqrt_6_l595_59553

variable {a b c : ℝ}

theorem abc_sum_eq_11_sqrt_6 : 
  0 < a → 0 < b → 0 < c → 
  a * b = 36 → 
  a * c = 72 → 
  b * c = 108 → 
  a + b + c = 11 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_GPT_abc_sum_eq_11_sqrt_6_l595_59553


namespace NUMINAMATH_GPT_inequality_solution_l595_59592

theorem inequality_solution (x : ℝ) : 
  (x-20) / (x+16) ≤ 0 ↔ -16 < x ∧ x ≤ 20 := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l595_59592


namespace NUMINAMATH_GPT_seatingArrangementsAreSix_l595_59504

-- Define the number of seating arrangements for 4 people around a round table
def numSeatingArrangements : ℕ :=
  3 * 2 * 1 -- Following the condition that the narrator's position is fixed

-- The main theorem stating the number of different seating arrangements
theorem seatingArrangementsAreSix : numSeatingArrangements = 6 :=
  by
    -- This is equivalent to following the explanation of solution which is just multiplying the numbers
    sorry

end NUMINAMATH_GPT_seatingArrangementsAreSix_l595_59504


namespace NUMINAMATH_GPT_carl_max_rocks_value_l595_59549

/-- 
Carl finds rocks of three different types:
  - 6-pound rocks worth $18 each.
  - 3-pound rocks worth $9 each.
  - 2-pound rocks worth $3 each.
There are at least 15 rocks available for each type.
Carl can carry at most 20 pounds.

Prove that the maximum value, in dollars, of the rocks Carl can carry out of the cave is $57.
-/
theorem carl_max_rocks_value : 
  (∃ x y z : ℕ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ 6 * x + 3 * y + 2 * z ≤ 20 ∧ 18 * x + 9 * y + 3 * z = 57) :=
sorry

end NUMINAMATH_GPT_carl_max_rocks_value_l595_59549


namespace NUMINAMATH_GPT_domain_ln_l595_59539

theorem domain_ln (x : ℝ) : x^2 - x - 2 > 0 ↔ (x < -1 ∨ x > 2) := by
  sorry

end NUMINAMATH_GPT_domain_ln_l595_59539


namespace NUMINAMATH_GPT_expr_value_l595_59520

variable (x y m n a : ℝ)
variable (hxy : x = -y) (hmn : m * n = 1) (ha : |a| = 3)

theorem expr_value : (a / (m * n) + 2018 * (x + y)) = a := sorry

end NUMINAMATH_GPT_expr_value_l595_59520


namespace NUMINAMATH_GPT_count_players_studying_chemistry_l595_59500

theorem count_players_studying_chemistry :
  ∀ 
    (total_players : ℕ)
    (math_players : ℕ)
    (physics_players : ℕ)
    (math_and_physics_players : ℕ)
    (all_three_subjects_players : ℕ),
    total_players = 18 →
    math_players = 10 →
    physics_players = 6 →
    math_and_physics_players = 3 →
    all_three_subjects_players = 2 →
    (total_players - (math_players + physics_players - math_and_physics_players)) + all_three_subjects_players = 7 :=
by
  intros total_players math_players physics_players math_and_physics_players all_three_subjects_players
  sorry

end NUMINAMATH_GPT_count_players_studying_chemistry_l595_59500


namespace NUMINAMATH_GPT_John_leftover_money_l595_59529

variables (q : ℝ)

def drinks_price (q : ℝ) : ℝ := 4 * q
def small_pizza_price (q : ℝ) : ℝ := q
def large_pizza_price (q : ℝ) : ℝ := 4 * q
def total_cost (q : ℝ) : ℝ := drinks_price q + small_pizza_price q + 2 * large_pizza_price q
def John_initial_money : ℝ := 50
def John_money_left (q : ℝ) : ℝ := John_initial_money - total_cost q

theorem John_leftover_money : John_money_left q = 50 - 13 * q :=
by
  sorry

end NUMINAMATH_GPT_John_leftover_money_l595_59529


namespace NUMINAMATH_GPT_minimum_trucks_on_lot_l595_59599

variable (total_trucks : ℕ) (rented_percentage : ℝ) (max_rented_trucks : ℕ)
variable (initial_trucks : total_trucks = 24) (returned_percentage : rented_percentage = 0.5) (max_trucks_rented : max_rented_trucks = 24)

theorem minimum_trucks_on_lot (total_trucks : ℕ) (rented_percentage : ℝ) (max_rented_trucks : ℕ)
  (initial_trucks : total_trucks = 24) (returned_percentage : rented_percentage = 0.5) (max_trucks_rented : max_rented_trucks = 24) :
  max_rented_trucks / 2 = 12 :=
by sorry

end NUMINAMATH_GPT_minimum_trucks_on_lot_l595_59599


namespace NUMINAMATH_GPT_express_y_in_terms_of_x_l595_59502

theorem express_y_in_terms_of_x (x y : ℝ) (h : y - 2 * x = 5) : y = 2 * x + 5 :=
by
  sorry

end NUMINAMATH_GPT_express_y_in_terms_of_x_l595_59502


namespace NUMINAMATH_GPT_todd_numbers_sum_eq_l595_59572

def sum_of_todd_numbers (n : ℕ) : ℕ :=
  sorry -- This would be the implementation of the sum based on provided problem conditions

theorem todd_numbers_sum_eq :
  sum_of_todd_numbers 5000 = 1250025 :=
sorry

end NUMINAMATH_GPT_todd_numbers_sum_eq_l595_59572


namespace NUMINAMATH_GPT_factory_produces_11250_products_l595_59585

noncomputable def total_products (refrigerators_per_hour coolers_per_hour hours_per_day days : ℕ) : ℕ :=
  (refrigerators_per_hour + coolers_per_hour) * (hours_per_day * days)

theorem factory_produces_11250_products :
  total_products 90 (90 + 70) 9 5 = 11250 := by
  sorry

end NUMINAMATH_GPT_factory_produces_11250_products_l595_59585


namespace NUMINAMATH_GPT_carl_insurance_payment_percentage_l595_59589

variable (property_damage : ℝ) (medical_bills : ℝ) 
          (total_cost : ℝ) (carl_payment : ℝ) (insurance_payment_percentage : ℝ)

theorem carl_insurance_payment_percentage :
  property_damage = 40000 ∧
  medical_bills = 70000 ∧
  total_cost = property_damage + medical_bills ∧
  carl_payment = 22000 ∧
  carl_payment = 0.20 * total_cost →
  insurance_payment_percentage = 100 - 20 :=
by
  sorry

end NUMINAMATH_GPT_carl_insurance_payment_percentage_l595_59589


namespace NUMINAMATH_GPT_solve_for_x_l595_59564

theorem solve_for_x : ∃ x : ℝ, 64 = 2 * (16 : ℝ)^(x - 2) ∧ x = 3.25 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l595_59564


namespace NUMINAMATH_GPT_julia_paint_area_l595_59547

noncomputable def area_to_paint (bedroom_length: ℕ) (bedroom_width: ℕ) (bedroom_height: ℕ) (non_paint_area: ℕ) (num_bedrooms: ℕ) : ℕ :=
  let wall_area_one_bedroom := 2 * (bedroom_length * bedroom_height) + 2 * (bedroom_width * bedroom_height)
  let paintable_area_one_bedroom := wall_area_one_bedroom - non_paint_area
  num_bedrooms * paintable_area_one_bedroom

theorem julia_paint_area :
  area_to_paint 14 11 9 70 4 = 1520 :=
by
  sorry

end NUMINAMATH_GPT_julia_paint_area_l595_59547


namespace NUMINAMATH_GPT_positive_number_property_l595_59557

theorem positive_number_property (x : ℝ) (h_pos : 0 < x) (h_eq : (x^2) / 100 = 9) : x = 30 :=
sorry

end NUMINAMATH_GPT_positive_number_property_l595_59557


namespace NUMINAMATH_GPT_number_of_correct_conclusions_l595_59586

-- Define the conditions as hypotheses
variable {a_n : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}
variable {a_1 : ℝ}
variable {n : ℕ}

-- Arithmetic sequence definition for a_n
def arithmetic_sequence (a_n : ℕ → ℝ) (a_1 d : ℝ) : Prop :=
  ∀ n, a_n n = a_1 + (n - 1) * d

-- Sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (S : ℕ → ℝ) (a_1 d : ℝ) : Prop :=
  ∀ n, S n = n / 2 * (2 * a_1 + (n - 1) * d)

-- Problem statement
theorem number_of_correct_conclusions 
  (h_seq : arithmetic_sequence a_n a_1 d)
  (h_sum : sum_arithmetic_sequence S a_1 d)
  (h1 : S 5 < S 6)
  (h2 : S 6 = S 7 ∧ S 7 > S 8) :
  ∃ n, n = 3 ∧ 
       (d < 0) ∧ 
       (a_n 7 = 0) ∧ 
       ¬(S 9 = S 5) ∧ 
       (S 6 = S 7 ∧ ∀ m, m > 7 → S m < S 6) := 
sorry

end NUMINAMATH_GPT_number_of_correct_conclusions_l595_59586


namespace NUMINAMATH_GPT_not_exists_odd_product_sum_l595_59551

theorem not_exists_odd_product_sum (a b : ℤ) : ¬ (a * b * (a + b) = 20182017) :=
sorry

end NUMINAMATH_GPT_not_exists_odd_product_sum_l595_59551


namespace NUMINAMATH_GPT_james_two_point_shots_l595_59514

-- Definitions based on conditions
def field_goals := 13
def field_goal_points := 3
def total_points := 79

-- Statement to be proven
theorem james_two_point_shots :
  ∃ x : ℕ, 79 = (field_goals * field_goal_points) + (2 * x) ∧ x = 20 :=
by
  sorry

end NUMINAMATH_GPT_james_two_point_shots_l595_59514


namespace NUMINAMATH_GPT_difference_of_squares_l595_59517

theorem difference_of_squares : 73^2 - 47^2 = 3120 :=
by sorry

end NUMINAMATH_GPT_difference_of_squares_l595_59517


namespace NUMINAMATH_GPT_ratio_of_pens_to_pencils_l595_59595

/-
The store ordered pens and pencils:
1. The number of pens was some multiple of the number of pencils plus 300.
2. The cost of a pen was $5.
3. The cost of a pencil was $4.
4. The store ordered 15 boxes, each having 80 pencils.
5. The store paid a total of $18,300 for the stationery.
Prove that the ratio of the number of pens to the number of pencils is 2.25.
-/

variables (e p k : ℕ)
variables (cost_pen : ℕ := 5) (cost_pencil : ℕ := 4) (total_cost : ℕ := 18300)

def number_of_pencils := 15 * 80

def number_of_pens := p -- to be defined in terms of e and k

def total_cost_pens := p * cost_pen
def total_cost_pencils := e * cost_pencil

theorem ratio_of_pens_to_pencils :
  p = k * e + 300 →
  e = 1200 →
  5 * p + 4 * e = 18300 →
  (p : ℚ) / e = 2.25 :=
by
  intros hp he htotal
  sorry

end NUMINAMATH_GPT_ratio_of_pens_to_pencils_l595_59595


namespace NUMINAMATH_GPT_distance_from_center_to_line_of_tangent_circle_l595_59548

theorem distance_from_center_to_line_of_tangent_circle 
  (a : ℝ) (ha : 0 < a) 
  (h_circle : (2 - a)^2 + (1 - a)^2 = a^2)
  (h_tangent : ∀ x y : ℝ, x = 0 ∨ y = 0): 
  (|2 * a - a - 3| / ((2:ℝ)^2 + (-1)^2).sqrt) = (2 * (5:ℝ).sqrt) / 5 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_center_to_line_of_tangent_circle_l595_59548


namespace NUMINAMATH_GPT_tangent_line_at_pi_l595_59543

theorem tangent_line_at_pi :
  ∀ f : ℝ → ℝ, (∀ x, f x = Real.sin x) → 
  ∀ x, x = Real.pi →
  ∀ y, (y = -x + Real.pi) ↔
        (∀ x, y = -x + Real.pi) := 
  sorry

end NUMINAMATH_GPT_tangent_line_at_pi_l595_59543


namespace NUMINAMATH_GPT_maximize_squares_l595_59509

theorem maximize_squares (a b : ℕ) (k : ℕ) :
  (a ≠ b) →
  ((∃ (k : ℤ), k ≠ 1 ∧ b = k^2) ↔ 
   (∃ (c₁ c₂ c₃ : ℕ), a * (b + 8) = c₁^2 ∧ b * (a + 8) = c₂^2 ∧ a * b = c₃^2 
     ∧ a = 1)) :=
by { sorry }

end NUMINAMATH_GPT_maximize_squares_l595_59509


namespace NUMINAMATH_GPT_arithmetic_sequence_inequality_l595_59540

variable {a : ℕ → ℝ}
variable {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_inequality 
  (h : is_arithmetic_sequence a d)
  (d_pos : d ≠ 0)
  (a_pos : ∀ n, a n > 0) :
  (a 1) * (a 8) < (a 4) * (a 5) := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_inequality_l595_59540


namespace NUMINAMATH_GPT_statues_ratio_l595_59519

theorem statues_ratio :
  let y1 := 4                  -- Number of statues after first year.
  let y2 := 4 * y1             -- Number of statues after second year.
  let y3 := (y2 + 12) - 3      -- Number of statues after third year.
  let y4 := 31                 -- Number of statues after fourth year.
  let added_fourth_year := y4 - y3  -- Statues added in the fourth year.
  let broken_third_year := 3        -- Statues broken in the third year.
  added_fourth_year / broken_third_year = 2 :=
by
  sorry

end NUMINAMATH_GPT_statues_ratio_l595_59519


namespace NUMINAMATH_GPT_person_y_speed_in_still_water_l595_59582

theorem person_y_speed_in_still_water 
    (speed_x_in_still_water : ℝ)
    (time_meeting_towards_each_other : ℝ)
    (time_catching_up_same_direction: ℝ)
    (distance_upstream_meeting: ℝ)
    (distance_downstream_meeting: ℝ)
    (total_distance: ℝ) :
    speed_x_in_still_water = 6 →
    time_meeting_towards_each_other = 4 →
    time_catching_up_same_direction = 16 →
    distance_upstream_meeting = 4 * (6 - distance_upstream_meeting) + 4 * (10 + distance_downstream_meeting) →
    distance_downstream_meeting = 4 * (6 + distance_upstream_meeting) →
    total_distance = 4 * (6 + 10) →
    ∃ (speed_y_in_still_water : ℝ), speed_y_in_still_water = 10 :=
by
  intros h_speed_x h_time_meeting h_time_catching h_distance_upstream h_distance_downstream h_total_distance
  sorry

end NUMINAMATH_GPT_person_y_speed_in_still_water_l595_59582


namespace NUMINAMATH_GPT_coefficient_x3_l595_59525

-- Define the binomial coefficient
def binomial_coefficient (n k : Nat) : Nat :=
  Nat.choose n k

noncomputable def coefficient_x3_term : Nat :=
  binomial_coefficient 25 3

theorem coefficient_x3 : coefficient_x3_term = 2300 :=
by
  unfold coefficient_x3_term
  unfold binomial_coefficient
  -- Here, one would normally provide the proof steps, but we're adding sorry to skip
  sorry

end NUMINAMATH_GPT_coefficient_x3_l595_59525


namespace NUMINAMATH_GPT_final_price_of_hat_is_correct_l595_59526

-- Definitions capturing the conditions.
def original_price : ℝ := 15
def first_discount_rate : ℝ := 0.20
def second_discount_rate : ℝ := 0.25

-- Calculations for the intermediate prices.
def price_after_first_discount : ℝ := original_price * (1 - first_discount_rate)
def final_price : ℝ := price_after_first_discount * (1 - second_discount_rate)

-- The theorem we need to prove.
theorem final_price_of_hat_is_correct : final_price = 9 := by
  sorry

end NUMINAMATH_GPT_final_price_of_hat_is_correct_l595_59526


namespace NUMINAMATH_GPT_total_expenditure_is_108_l595_59591

-- Define the costs of items and quantities purchased by Robert and Teddy
def cost_pizza := 10   -- cost of one box of pizza
def cost_soft_drink := 2  -- cost of one can of soft drink
def cost_hamburger := 3   -- cost of one hamburger

def qty_pizza_robert := 5     -- quantity of pizza boxes by Robert
def qty_soft_drink_robert := 10 -- quantity of soft drinks by Robert

def qty_hamburger_teddy := 6  -- quantity of hamburgers by Teddy
def qty_soft_drink_teddy := 10 -- quantity of soft drinks by Teddy

-- Calculate total expenditure for Robert and Teddy
def total_cost_robert := (qty_pizza_robert * cost_pizza) + (qty_soft_drink_robert * cost_soft_drink)
def total_cost_teddy := (qty_hamburger_teddy * cost_hamburger) + (qty_soft_drink_teddy * cost_soft_drink)

-- Total expenditure in all
def total_expenditure := total_cost_robert + total_cost_teddy

-- We formulate the theorem to prove that the total expenditure is $108
theorem total_expenditure_is_108 : total_expenditure = 108 :=
by 
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_total_expenditure_is_108_l595_59591


namespace NUMINAMATH_GPT_find_missing_number_l595_59531

theorem find_missing_number (x : ℕ) (h : 10010 - 12 * 3 * x = 9938) : x = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_missing_number_l595_59531


namespace NUMINAMATH_GPT_sum_of_a_b_l595_59538

-- Definitions for the given conditions
def geom_series_sum (a : ℤ) (n : ℕ) : ℤ := 2^n + a
def arith_series_sum (b : ℤ) (n : ℕ) : ℤ := n^2 - 2*n + b

-- Theorem statement
theorem sum_of_a_b (a b : ℤ) (h1 : ∀ n, geom_series_sum a n = 2^n + a)
  (h2 : ∀ n, arith_series_sum b n = n^2 - 2*n + b) :
  a + b = -1 :=
sorry

end NUMINAMATH_GPT_sum_of_a_b_l595_59538


namespace NUMINAMATH_GPT_number_of_pairs_l595_59532

theorem number_of_pairs (x y : ℤ) (hx : 1 ≤ x ∧ x ≤ 1000) (hy : 1 ≤ y ∧ y ≤ 1000) :
  (x^2 + y^2) % 7 = 0 → (∃ n : ℕ, n = 20164) :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_pairs_l595_59532


namespace NUMINAMATH_GPT_simplify_expression_l595_59528

def is_real (x : ℂ) : Prop := ∃ (y : ℝ), x = y

theorem simplify_expression 
  (x y c : ℝ) 
  (i : ℂ) 
  (hi : i^2 = -1) :
  (x + i*y + c)^2 = (x^2 + c^2 - y^2 + 2 * c * x + (2 * x * y + 2 * c * y) * i) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l595_59528


namespace NUMINAMATH_GPT_sunflower_height_A_l595_59594

-- Define the height of sunflowers from Packet B
def height_B : ℝ := 160

-- Define that Packet A sunflowers are 20% taller than Packet B sunflowers
def height_A : ℝ := 1.2 * height_B

-- State the theorem to show that height_A equals 192 inches
theorem sunflower_height_A : height_A = 192 := by
  sorry

end NUMINAMATH_GPT_sunflower_height_A_l595_59594


namespace NUMINAMATH_GPT_tom_sawyer_bible_l595_59593

def blue_tickets_needed (yellow: ℕ) (red: ℕ) (blue: ℕ): ℕ := 
  10 * 10 * 10 * yellow + 10 * 10 * red + blue

theorem tom_sawyer_bible (y r b : ℕ) (hc : y = 8 ∧ r = 3 ∧ b = 7):
  blue_tickets_needed 10 0 0 - blue_tickets_needed y r b = 163 :=
by 
  sorry

end NUMINAMATH_GPT_tom_sawyer_bible_l595_59593


namespace NUMINAMATH_GPT_pizza_toppings_l595_59515

theorem pizza_toppings :
  ∀ (F V T : ℕ), F = 4 → V = 16 → F * (1 + T) = V → T = 3 :=
by
  intros F V T hF hV h
  sorry

end NUMINAMATH_GPT_pizza_toppings_l595_59515


namespace NUMINAMATH_GPT_inequality_am_gm_l595_59524

theorem inequality_am_gm (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) :=
sorry

end NUMINAMATH_GPT_inequality_am_gm_l595_59524


namespace NUMINAMATH_GPT_count_valid_n_l595_59566

theorem count_valid_n :
  let n_values := [50, 550, 1050, 1550, 2050]
  ( ∀ n : ℤ, (50 * ((n + 500) / 50) - 500 = n) ∧ (Int.floor (Real.sqrt (2 * n : ℝ)) = (n + 500) / 50) → n ∈ n_values ) ∧
  ((∀ n : ℤ, ∃ k : ℤ, (n = 50 * k - 500) ∧ (k = Int.floor (Real.sqrt (2 * (50 * k - 500) : ℝ))) ∧ 0 < n ) → n_values.length = 5) :=
by
  sorry

end NUMINAMATH_GPT_count_valid_n_l595_59566


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_segment_ratio_l595_59565

theorem right_triangle_hypotenuse_segment_ratio
  (x : ℝ) (AB BC AC BD AD CD : ℝ)
  (h1 : AB = 4 * x) 
  (h2 : BC = 3 * x) 
  (h3 : AC = 5 * x) 
  (h4 : (BD ^ 2) = AD * CD) :
  (CD / AD) = (16 / 9) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_segment_ratio_l595_59565


namespace NUMINAMATH_GPT_four_number_theorem_l595_59505

theorem four_number_theorem (a b c d : ℕ) (H : a * b = c * d) (Ha : 0 < a) (Hb : 0 < b) (Hc : 0 < c) (Hd : 0 < d) : 
  ∃ (p q r s : ℕ), 0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < s ∧ a = p * q ∧ b = r * s ∧ c = p * s ∧ d = q * r :=
by
  sorry

end NUMINAMATH_GPT_four_number_theorem_l595_59505


namespace NUMINAMATH_GPT_matrix_vector_multiplication_correct_l595_59587

noncomputable def mat : Matrix (Fin 2) (Fin 2) ℤ := ![![3, -2], ![1, 5]]
noncomputable def vec : Fin 2 → ℤ := ![-1, 2]
noncomputable def result : Fin 2 → ℤ := ![-7, 9]

theorem matrix_vector_multiplication_correct :
  (Matrix.mulVec mat vec) = result :=
by
  sorry

end NUMINAMATH_GPT_matrix_vector_multiplication_correct_l595_59587


namespace NUMINAMATH_GPT_largest_possible_s_l595_59568

theorem largest_possible_s (r s : ℕ) (h1 : 3 ≤ s) (h2 : s ≤ r) (h3 : s < 122)
    (h4 : ∀ r s, (61 * (s - 2) * r = 60 * (r - 2) * s)) : s ≤ 121 :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_s_l595_59568


namespace NUMINAMATH_GPT_geo_seq_sum_eq_l595_59577

variable {a : ℕ → ℝ}

-- Conditions
def is_geo_seq (a : ℕ → ℝ) : Prop := ∃ r : ℝ, ∀ n : ℕ, a (n+1) = a n * r
def positive_seq (a : ℕ → ℝ) : Prop := ∀ n : ℕ, a n > 0
def specific_eq (a : ℕ → ℝ) : Prop := a 1 * a 3 + 2 * a 2 * a 4 + a 3 * a 5 = 25

theorem geo_seq_sum_eq (a : ℕ → ℝ) (h_geo : is_geo_seq a) (h_pos : positive_seq a) (h_eq : specific_eq a) : 
  a 2 + a 4 = 5 :=
by
  sorry

end NUMINAMATH_GPT_geo_seq_sum_eq_l595_59577


namespace NUMINAMATH_GPT_total_revenue_correct_l595_59556

-- Define the conditions
def original_price_sneakers : ℝ := 80
def discount_sneakers : ℝ := 0.25
def pairs_sold_sneakers : ℕ := 2

def original_price_sandals : ℝ := 60
def discount_sandals : ℝ := 0.35
def pairs_sold_sandals : ℕ := 4

def original_price_boots : ℝ := 120
def discount_boots : ℝ := 0.4
def pairs_sold_boots : ℕ := 11

-- Compute discounted prices
def discounted_price (original_price : ℝ) (discount : ℝ) : ℝ :=
  original_price - (original_price * discount)

-- Compute revenue from each type of shoe
def revenue (price : ℝ) (pairs_sold : ℕ) : ℝ :=
  price * (pairs_sold : ℝ)

open Real

-- Main statement to prove
theorem total_revenue_correct : 
  revenue (discounted_price original_price_sneakers discount_sneakers) pairs_sold_sneakers + 
  revenue (discounted_price original_price_sandals discount_sandals) pairs_sold_sandals + 
  revenue (discounted_price original_price_boots discount_boots) pairs_sold_boots = 1068 := 
by
  sorry

end NUMINAMATH_GPT_total_revenue_correct_l595_59556


namespace NUMINAMATH_GPT_difference_max_min_y_l595_59567

-- Define initial and final percentages of responses
def initial_yes : ℝ := 0.30
def initial_no : ℝ := 0.70
def final_yes : ℝ := 0.60
def final_no : ℝ := 0.40

-- Define the problem statement
theorem difference_max_min_y : 
  ∃ y_min y_max : ℝ, (initial_yes + initial_no = 1) ∧ (final_yes + final_no = 1) ∧
  (initial_yes + initial_no = final_yes + final_no) ∧ y_min ≤ y_max ∧ 
  y_max - y_min = 0.30 :=
sorry

end NUMINAMATH_GPT_difference_max_min_y_l595_59567


namespace NUMINAMATH_GPT_max_glows_in_time_range_l595_59560

theorem max_glows_in_time_range (start_time end_time : ℤ) (interval : ℤ) (h1 : start_time = 3600 + 3420 + 58) (h2 : end_time = 10800 + 1200 + 47) (h3 : interval = 21) :
  (end_time - start_time) / interval = 236 := 
  sorry

end NUMINAMATH_GPT_max_glows_in_time_range_l595_59560
