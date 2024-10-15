import Mathlib

namespace NUMINAMATH_GPT_georgie_guacamole_servings_l26_2651

-- Define the conditions
def avocados_needed_per_serving : Nat := 3
def initial_avocados : Nat := 5
def additional_avocados : Nat := 4

-- State the target number of servings Georgie can make
def total_avocados := initial_avocados + additional_avocados
def guacamole_servings := total_avocados / avocados_needed_per_serving

-- Lean 4 statement asserting the number of servings equals 3
theorem georgie_guacamole_servings : guacamole_servings = 3 := by
  sorry

end NUMINAMATH_GPT_georgie_guacamole_servings_l26_2651


namespace NUMINAMATH_GPT_hyperbola_same_foci_as_ellipse_eccentricity_two_l26_2647

theorem hyperbola_same_foci_as_ellipse_eccentricity_two
  (a b c e : ℝ)
  (ellipse_eq : ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (a = 5 ∧ b = 3 ∧ c = 4))
  (eccentricity_eq : e = 2) :
  ∃ x y : ℝ, (x^2 / (c / e)^2 - y^2 / (c^2 - (c / e)^2) = 1) ↔ (x^2 / 4 - y^2 / 12 = 1) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_same_foci_as_ellipse_eccentricity_two_l26_2647


namespace NUMINAMATH_GPT_set_intersection_complement_l26_2642

theorem set_intersection_complement (U M N : Set ℤ)
  (hU : U = {0, -1, -2, -3, -4})
  (hM : M = {0, -1, -2})
  (hN : N = {0, -3, -4}) :
  (U \ M) ∩ N = {-3, -4} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_complement_l26_2642


namespace NUMINAMATH_GPT_how_many_leaves_l26_2668

def ladybugs_per_leaf : ℕ := 139
def total_ladybugs : ℕ := 11676

theorem how_many_leaves : total_ladybugs / ladybugs_per_leaf = 84 :=
by
  sorry

end NUMINAMATH_GPT_how_many_leaves_l26_2668


namespace NUMINAMATH_GPT_polygon_sides_from_diagonals_l26_2692

theorem polygon_sides_from_diagonals (n D : ℕ) (h1 : D = 15) (h2 : D = n * (n - 3) / 2) : n = 8 :=
by
  -- skipping proof
  sorry

end NUMINAMATH_GPT_polygon_sides_from_diagonals_l26_2692


namespace NUMINAMATH_GPT_Marty_painting_combinations_l26_2635

theorem Marty_painting_combinations :
  let parts_of_room := 2
  let colors := 5
  let methods := 3
  (parts_of_room * colors * methods) = 30 := 
by
  let parts_of_room := 2
  let colors := 5
  let methods := 3
  show (parts_of_room * colors * methods) = 30
  sorry

end NUMINAMATH_GPT_Marty_painting_combinations_l26_2635


namespace NUMINAMATH_GPT_quadratic_factors_l26_2632

-- Define the quadratic polynomial
def quadratic (b c x : ℝ) : ℝ := x^2 + b * x + c

-- Define the roots
def root1 : ℝ := -2
def root2 : ℝ := 3

-- Theorem: If the quadratic equation has roots -2 and 3, then it factors as (x + 2)(x - 3)
theorem quadratic_factors (b c : ℝ) (h1 : quadratic b c root1 = 0) (h2 : quadratic b c root2 = 0) :
  ∀ x : ℝ, quadratic b c x = (x + 2) * (x - 3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_factors_l26_2632


namespace NUMINAMATH_GPT_find_x_l26_2607

theorem find_x (x : ℝ) : (x / (x + 2) + 3 / (x + 2) + 2 * x / (x + 2) = 4) → x = -5 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l26_2607


namespace NUMINAMATH_GPT_inequality_l26_2666

theorem inequality (a b c : ℝ) (h₀ : 0 < c) (h₁ : c < b) (h₂ : b < a) :
  a^4 * b + b^4 * c + c^4 * a > a * b^4 + b * c^4 + c * a^4 :=
by sorry

end NUMINAMATH_GPT_inequality_l26_2666


namespace NUMINAMATH_GPT_calculate_cubic_sum_roots_l26_2663

noncomputable def α := (27 : ℝ)^(1/3)
noncomputable def β := (64 : ℝ)^(1/3)
noncomputable def γ := (125 : ℝ)^(1/3)

theorem calculate_cubic_sum_roots (u v w : ℝ) :
  (u - α) * (u - β) * (u - γ) = 1/2 ∧
  (v - α) * (v - β) * (v - γ) = 1/2 ∧
  (w - α) * (w - β) * (w - γ) = 1/2 →
  u^3 + v^3 + w^3 = 217.5 :=
by
  sorry

end NUMINAMATH_GPT_calculate_cubic_sum_roots_l26_2663


namespace NUMINAMATH_GPT_scarlett_oil_amount_l26_2698

theorem scarlett_oil_amount (initial_oil add_oil : ℝ) (h1 : initial_oil = 0.17) (h2 : add_oil = 0.67) :
  initial_oil + add_oil = 0.84 :=
by
  rw [h1, h2]
  -- Proof step goes here
  sorry

end NUMINAMATH_GPT_scarlett_oil_amount_l26_2698


namespace NUMINAMATH_GPT_proof_problem_l26_2622

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set A as {y | y = 2^x, x ∈ ℝ}
def A : Set ℝ := {y | ∃ x : ℝ, y = 2^x}

-- Define the set B as {x ∈ ℤ | x^2 - 4 ≤ 0}
def B : Set ℤ := {x | x ∈ Set.Icc (-2 : ℤ) 2}

-- Define the complement of A relative to U (universal set)
def CU_A : Set ℝ := {x | x ≤ 0}

-- Define the proposition to be proved
theorem proof_problem :
  (CU_A ∩ (Set.image (coe : ℤ → ℝ) B)) = {-2.0, 1.0, 0.0} :=
by 
  sorry

end NUMINAMATH_GPT_proof_problem_l26_2622


namespace NUMINAMATH_GPT_bricks_needed_for_wall_l26_2609

noncomputable def brick_volume (length width height : ℝ) : ℝ :=
  length * width * height

noncomputable def wall_volume (length height thickness : ℝ) : ℝ :=
  length * height * thickness

theorem bricks_needed_for_wall :
  let length_wall := 800
  let height_wall := 600
  let thickness_wall := 22.5
  let length_brick := 100
  let width_brick := 11.25
  let height_brick := 6
  let vol_wall := wall_volume length_wall height_wall thickness_wall
  let vol_brick := brick_volume length_brick width_brick height_brick
  vol_wall / vol_brick = 1600 :=
by
  sorry

end NUMINAMATH_GPT_bricks_needed_for_wall_l26_2609


namespace NUMINAMATH_GPT_num_sets_M_l26_2699

theorem num_sets_M (M : Set ℕ) :
  {1, 2} ⊆ M ∧ M ⊆ {1, 2, 3, 4, 5, 6} → ∃ n : Nat, n = 16 :=
by
  sorry

end NUMINAMATH_GPT_num_sets_M_l26_2699


namespace NUMINAMATH_GPT_cube_paint_same_color_l26_2671

theorem cube_paint_same_color (colors : Fin 6) : ∃ ways : ℕ, ways = 6 :=
sorry

end NUMINAMATH_GPT_cube_paint_same_color_l26_2671


namespace NUMINAMATH_GPT_rectangle_area_l26_2614

theorem rectangle_area {AB AC BC : ℕ} (hAB : AB = 15) (hAC : AC = 17)
  (hRightTriangle : AC * AC = AB * AB + BC * BC) : AB * BC = 120 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l26_2614


namespace NUMINAMATH_GPT_desired_butterfat_percentage_l26_2689

theorem desired_butterfat_percentage (milk1 milk2 : ℝ) (butterfat1 butterfat2 : ℝ) :
  milk1 = 8 →
  butterfat1 = 0.10 →
  milk2 = 8 →
  butterfat2 = 0.30 →
  ((butterfat1 * milk1) + (butterfat2 * milk2)) / (milk1 + milk2) * 100 = 20 := 
by
  intros
  sorry

end NUMINAMATH_GPT_desired_butterfat_percentage_l26_2689


namespace NUMINAMATH_GPT_ratio_of_new_time_to_previous_time_l26_2634

-- Given conditions
def distance : ℕ := 288
def initial_time : ℕ := 6
def new_speed : ℕ := 32

-- Question: Prove the ratio of the new time to the previous time is 3:2
theorem ratio_of_new_time_to_previous_time :
  (distance / new_speed) / initial_time = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_new_time_to_previous_time_l26_2634


namespace NUMINAMATH_GPT_product_mod_eq_l26_2695

theorem product_mod_eq :
  (1497 * 2003) % 600 = 291 := 
sorry

end NUMINAMATH_GPT_product_mod_eq_l26_2695


namespace NUMINAMATH_GPT_height_of_isosceles_triangle_l26_2685

variable (s : ℝ) (h : ℝ) (A : ℝ)
variable (triangle : ∀ (s : ℝ) (h : ℝ), A = 0.5 * (2 * s) * h)
variable (rectangle : ∀ (s : ℝ), A = s^2)

theorem height_of_isosceles_triangle (s : ℝ) (h : ℝ) (A : ℝ) (triangle : ∀ (s : ℝ) (h : ℝ), A = 0.5 * (2 * s) * h)
  (rectangle : ∀ (s : ℝ), A = s^2) : h = s := by
  sorry

end NUMINAMATH_GPT_height_of_isosceles_triangle_l26_2685


namespace NUMINAMATH_GPT_min_value_fraction_l26_2667

theorem min_value_fraction (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) :
  (1 / a + 9 / b) ≥ 8 :=
by sorry

end NUMINAMATH_GPT_min_value_fraction_l26_2667


namespace NUMINAMATH_GPT_train_length_l26_2644

theorem train_length
  (V L : ℝ)
  (h1 : L = V * 18)
  (h2 : L + 350 = V * 39) :
  L = 300 := 
by
  sorry

end NUMINAMATH_GPT_train_length_l26_2644


namespace NUMINAMATH_GPT_problem1_statement_problem2_statement_l26_2620

-- Defining the sets A and B
def set_A (x : ℝ) := 2*x^2 - 7*x + 3 ≤ 0
def set_B (x a : ℝ) := x + a < 0

-- Problem 1: Intersection of A and B when a = -2
def question1 (x : ℝ) : Prop := set_A x ∧ set_B x (-2)

-- Problem 2: Range of a for A ∩ B = A
def question2 (a : ℝ) : Prop := ∀ x, set_A x → set_B x a

theorem problem1_statement :
  ∀ x, question1 x ↔ x >= 1/2 ∧ x < 2 :=
by sorry

theorem problem2_statement :
  ∀ a, (∀ x, set_A x → set_B x a) ↔ a < -3 :=
by sorry

end NUMINAMATH_GPT_problem1_statement_problem2_statement_l26_2620


namespace NUMINAMATH_GPT_age_sum_proof_l26_2605

noncomputable def leilei_age : ℝ := 30 -- Age of Leilei this year
noncomputable def feifei_age (R : ℝ) : ℝ := 1 / 2 * R + 12 -- Age of Feifei this year defined in terms of R

theorem age_sum_proof (R F : ℝ)
  (h1 : F = 1 / 2 * R + 12)
  (h2 : F + 1 = 2 * (R + 1) - 34) :
  R + F = 57 :=
by 
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_age_sum_proof_l26_2605


namespace NUMINAMATH_GPT_exists_strictly_increasing_sequence_l26_2664

open Nat

-- Definition of strictly increasing sequence of integers a
def strictly_increasing_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a n < a (n + 1)

-- Condition i): Every natural number can be written as the sum of two terms from the sequence
def condition_i (a : ℕ → ℕ) : Prop :=
  ∀ m : ℕ, ∃ i j : ℕ, m = a i + a j

-- Condition ii): For each positive integer n, a_n > n^2/16
def condition_ii (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → a n > n^2 / 16

-- The main theorem stating the existence of such a sequence
theorem exists_strictly_increasing_sequence :
  ∃ a : ℕ → ℕ, a 0 = 0 ∧ strictly_increasing_sequence a ∧ condition_i a ∧ condition_ii a :=
sorry

end NUMINAMATH_GPT_exists_strictly_increasing_sequence_l26_2664


namespace NUMINAMATH_GPT_steps_to_11th_floor_l26_2679

theorem steps_to_11th_floor 
  (steps_between_3_and_5 : ℕ) 
  (third_floor : ℕ := 3) 
  (fifth_floor : ℕ := 5) 
  (eleventh_floor : ℕ := 11) 
  (ground_floor : ℕ := 1) 
  (steps_per_floor : ℕ := steps_between_3_and_5 / (fifth_floor - third_floor)) :
  steps_between_3_and_5 = 42 →
  steps_between_3_and_5 / (fifth_floor - third_floor) = 21 →
  (eleventh_floor - ground_floor) = 10 →
  21 * 10 = 210 := 
by
  intros _ _ _
  exact rfl

end NUMINAMATH_GPT_steps_to_11th_floor_l26_2679


namespace NUMINAMATH_GPT_product_of_undefined_roots_l26_2697

theorem product_of_undefined_roots :
  let f (x : ℝ) := (x^2 - 4*x + 4) / (x^2 - 5*x + 6)
  ∀ x : ℝ, (x^2 - 5*x + 6 = 0) → x = 2 ∨ x = 3 →
  (x = 2 ∨ x = 3 → x1 = 2 ∧ x2 = 3 → x1 * x2 = 6) :=
by
  sorry

end NUMINAMATH_GPT_product_of_undefined_roots_l26_2697


namespace NUMINAMATH_GPT_digit_100th_is_4_digit_1000th_is_3_l26_2682

noncomputable section

def digit_100th_place : Nat :=
  4

def digit_1000th_place : Nat :=
  3

theorem digit_100th_is_4 (n : ℕ) (h1 : n ∈ {m | m = 100}) : digit_100th_place = 4 := by
  sorry

theorem digit_1000th_is_3 (n : ℕ) (h1 : n ∈ {m | m = 1000}) : digit_1000th_place = 3 := by
  sorry

end NUMINAMATH_GPT_digit_100th_is_4_digit_1000th_is_3_l26_2682


namespace NUMINAMATH_GPT_x_eq_1_sufficient_not_necessary_l26_2673

theorem x_eq_1_sufficient_not_necessary (x : ℝ) : 
    (x = 1 → (x^2 - 3 * x + 2 = 0)) ∧ ¬((x^2 - 3 * x + 2 = 0) → (x = 1)) := 
by
  sorry

end NUMINAMATH_GPT_x_eq_1_sufficient_not_necessary_l26_2673


namespace NUMINAMATH_GPT_intersection_complement_U_l26_2611

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

def B_complement_U : Set ℕ := U \ B

theorem intersection_complement_U (hU : U = {1, 3, 5, 7}) 
                                  (hA : A = {3, 5}) 
                                  (hB : B = {1, 3, 7}) : 
  A ∩ (B_complement_U U B) = {5} := by
  sorry

end NUMINAMATH_GPT_intersection_complement_U_l26_2611


namespace NUMINAMATH_GPT_percentage_discount_of_retail_price_l26_2624

theorem percentage_discount_of_retail_price {wp rp sp discount : ℝ} (h1 : wp = 99) (h2 : rp = 132) (h3 : sp = wp + 0.20 * wp) (h4 : discount = (rp - sp) / rp * 100) : discount = 10 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_discount_of_retail_price_l26_2624


namespace NUMINAMATH_GPT_intersection_M_N_l26_2675

-- Define the universe U
def U : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the set M based on the condition x^2 <= x
def M : Set ℤ := {x ∈ U | x^2 ≤ x}

-- Define the set N based on the condition x^3 - 3x^2 + 2x = 0
def N : Set ℤ := {x ∈ U | x^3 - 3*x^2 + 2*x = 0}

-- State the theorem to be proven
theorem intersection_M_N : M ∩ N = {0, 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l26_2675


namespace NUMINAMATH_GPT_initial_pieces_count_l26_2616

theorem initial_pieces_count (people : ℕ) (pieces_per_person : ℕ) (leftover_pieces : ℕ) :
  people = 6 → pieces_per_person = 7 → leftover_pieces = 3 → people * pieces_per_person + leftover_pieces = 45 :=
by
  intros h_people h_pieces_per_person h_leftover_pieces
  sorry

end NUMINAMATH_GPT_initial_pieces_count_l26_2616


namespace NUMINAMATH_GPT_find_z_value_l26_2659

-- We will define the variables and the given condition
variables {x y z : ℝ}

-- Translate the given condition into Lean
def given_condition (x y z : ℝ) : Prop := (1 / x^2 - 1 / y^2) = (1 / z)

-- State the theorem to prove
theorem find_z_value (x y z : ℝ) (h : given_condition x y z) : 
  z = (x^2 * y^2) / (y^2 - x^2) :=
sorry

end NUMINAMATH_GPT_find_z_value_l26_2659


namespace NUMINAMATH_GPT_tangent_line_parabola_l26_2638

theorem tangent_line_parabola (d : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + d → ∃! x y, y^2 = 12 * x) → d = 3 := 
by
  intro h
  -- Here, "h" would be our hypothesis where we assume the line is tangent to the parabola
  sorry

end NUMINAMATH_GPT_tangent_line_parabola_l26_2638


namespace NUMINAMATH_GPT_smallest_x_plus_y_l26_2678

theorem smallest_x_plus_y {x y : ℕ} (hxy : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y) 
(h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 12) : x + y = 49 :=
sorry

end NUMINAMATH_GPT_smallest_x_plus_y_l26_2678


namespace NUMINAMATH_GPT_find_a_l26_2684

open Real

-- Definition of regression line
def regression_line (x : ℝ) : ℝ := 12.6 * x + 0.6

-- Data points for x and y
def x_values : List ℝ := [2, 3, 3.5, 4.5, 7]
def y_values : List ℝ := [26, 38, 43, 60]

-- Proof statement
theorem find_a (a : ℝ) (hx : x_values = [2, 3, 3.5, 4.5, 7])
  (hy : y_values ++ [a] = [26, 38, 43, 60, a]) : a = 88 :=
  sorry

end NUMINAMATH_GPT_find_a_l26_2684


namespace NUMINAMATH_GPT_apples_and_pears_weight_l26_2628

theorem apples_and_pears_weight (apples pears : ℕ) 
    (h_apples : apples = 240) 
    (h_pears : pears = 3 * apples) : 
    apples + pears = 960 := 
  by
  sorry

end NUMINAMATH_GPT_apples_and_pears_weight_l26_2628


namespace NUMINAMATH_GPT_complex_round_quadrant_l26_2676

open Complex

theorem complex_round_quadrant (z : ℂ) (i : ℂ) (h : i = Complex.I) (h1 : z * i = 2 - i):
  z.re < 0 ∧ z.im < 0 := 
sorry

end NUMINAMATH_GPT_complex_round_quadrant_l26_2676


namespace NUMINAMATH_GPT_total_action_figures_l26_2618

def action_figures_per_shelf : ℕ := 11
def number_of_shelves : ℕ := 4

theorem total_action_figures : action_figures_per_shelf * number_of_shelves = 44 := by
  sorry

end NUMINAMATH_GPT_total_action_figures_l26_2618


namespace NUMINAMATH_GPT_joyce_apples_l26_2680

theorem joyce_apples : 
  ∀ (initial_apples given_apples remaining_apples : ℕ), 
    initial_apples = 75 → 
    given_apples = 52 → 
    remaining_apples = initial_apples - given_apples → 
    remaining_apples = 23 :=
by 
  intros initial_apples given_apples remaining_apples h_initial h_given h_remaining
  rw [h_initial, h_given] at h_remaining
  exact h_remaining

end NUMINAMATH_GPT_joyce_apples_l26_2680


namespace NUMINAMATH_GPT_count_valid_n_decomposition_l26_2656

theorem count_valid_n_decomposition : 
  ∃ (count : ℕ), count = 108 ∧ 
  ∀ (a b c n : ℕ), 
    8 * a + 88 * b + 888 * c = 8000 → 
    0 ≤ b ∧ b ≤ 90 → 
    0 ≤ c ∧ c ≤ 9 → 
    n = a + 2 * b + 3 * c → 
    n < 1000 :=
sorry

end NUMINAMATH_GPT_count_valid_n_decomposition_l26_2656


namespace NUMINAMATH_GPT_ellipse_equation_l26_2625

theorem ellipse_equation (a b : ℝ) (A : ℝ × ℝ)
  (hA : A = (-3, 1.75))
  (he : 0.75 = Real.sqrt (a^2 - b^2) / a) 
  (hcond : (Real.sqrt (a^2 - b^2) / a) = 0.75) :
  (16 = a^2) ∧ (7 = b^2) :=
by
  have h1 : A = (-3, 1.75) := hA
  have h2 : Real.sqrt (a^2 - b^2) / a = 0.75 := hcond
  sorry

end NUMINAMATH_GPT_ellipse_equation_l26_2625


namespace NUMINAMATH_GPT_abs_eq_five_l26_2626

theorem abs_eq_five (x : ℝ) : |x| = 5 → (x = 5 ∨ x = -5) :=
by
  sorry

end NUMINAMATH_GPT_abs_eq_five_l26_2626


namespace NUMINAMATH_GPT_gcd_fx_x_l26_2693

def f (x: ℕ) := (5 * x + 4) * (9 * x + 7) * (11 * x + 3) * (x + 12)

theorem gcd_fx_x (x: ℕ) (h: x % 54896 = 0) : Nat.gcd (f x) x = 112 :=
  sorry

end NUMINAMATH_GPT_gcd_fx_x_l26_2693


namespace NUMINAMATH_GPT_no_carry_consecutive_pairs_l26_2636

/-- Consider the range of integers {2000, 2001, ..., 3000}. 
    We determine that the number of pairs of consecutive integers in this range such that their addition requires no carrying is 729. -/
theorem no_carry_consecutive_pairs : 
  ∀ (n : ℕ), (2000 ≤ n ∧ n < 3000) ∧ ((n + 1) ≤ 3000) → 
  ∃ (count : ℕ), count = 729 := 
sorry

end NUMINAMATH_GPT_no_carry_consecutive_pairs_l26_2636


namespace NUMINAMATH_GPT_least_integer_solution_l26_2645

theorem least_integer_solution (x : ℤ) : (∀ y : ℤ, |2 * y + 9| <= 20 → x ≤ y) ↔ x = -14 := by
  sorry

end NUMINAMATH_GPT_least_integer_solution_l26_2645


namespace NUMINAMATH_GPT_problem_1_problem_2_l26_2690

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a|

theorem problem_1 (x : ℝ) : (∀ x, f x 4 < 8 - |x - 1|) → x ∈ Set.Ioo (-1 : ℝ) (13 / 3) :=
by sorry

theorem problem_2 (a : ℝ) : (∃ x, f x a > 8 + |2 * x - 1|) → a > 9 ∨ a < -7 :=
by sorry

end NUMINAMATH_GPT_problem_1_problem_2_l26_2690


namespace NUMINAMATH_GPT_union_sets_l26_2683

def setA : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def setB : Set ℝ := { x | 0 < x ∧ x < 3 }

theorem union_sets :
  (setA ∪ setB) = { x | -1 ≤ x ∧ x < 3 } :=
sorry

end NUMINAMATH_GPT_union_sets_l26_2683


namespace NUMINAMATH_GPT_petya_coin_difference_20_l26_2660

-- Definitions for the problem conditions
variables (n k : ℕ) -- n: number of 5-ruble coins Petya has, k: number of 2-ruble coins Petya has

-- Condition: Petya has 60 rubles more than Vanya
def petya_has_60_more (n k : ℕ) : Prop := (5 * n + 2 * k = 5 * k + 2 * n + 60)

-- Theorem to prove Petya has 20 more 5-ruble coins than 2-ruble coins
theorem petya_coin_difference_20 (n k : ℕ) (h : petya_has_60_more n k) : n - k = 20 :=
sorry

end NUMINAMATH_GPT_petya_coin_difference_20_l26_2660


namespace NUMINAMATH_GPT_sqrt_of_0_09_l26_2687

theorem sqrt_of_0_09 : Real.sqrt 0.09 = 0.3 :=
by
  -- Mathematical problem restates that the square root of 0.09 equals 0.3
  sorry

end NUMINAMATH_GPT_sqrt_of_0_09_l26_2687


namespace NUMINAMATH_GPT_capacity_of_other_bottle_l26_2694

theorem capacity_of_other_bottle (C : ℝ) :
  (∀ (total_milk c1 c2 : ℝ), total_milk = 8 ∧ c1 = 5.333333333333333 ∧ c2 = C ∧ 
  (c1 / 8 = (c2 / C))) → C = 4 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_capacity_of_other_bottle_l26_2694


namespace NUMINAMATH_GPT_more_than_half_remains_l26_2617

def cubic_block := { n : ℕ // n > 0 }

noncomputable def total_cubes (b : cubic_block) : ℕ := b.val ^ 3

noncomputable def outer_layer_cubes (b : cubic_block) : ℕ := 6 * (b.val ^ 2) - 12 * b.val + 8

noncomputable def remaining_cubes (b : cubic_block) : ℕ := total_cubes b - outer_layer_cubes b

theorem more_than_half_remains (b : cubic_block) (h : b.val = 10) : remaining_cubes b > total_cubes b / 2 :=
by
  sorry

end NUMINAMATH_GPT_more_than_half_remains_l26_2617


namespace NUMINAMATH_GPT_matrix_multiplication_correct_l26_2677

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 1], ![4, -2]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![5, -3], ![2, 6]]
def C : Matrix (Fin 2) (Fin 2) ℤ := ![![17, -3], ![16, -24]]

theorem matrix_multiplication_correct : A * B = C := by 
  sorry

end NUMINAMATH_GPT_matrix_multiplication_correct_l26_2677


namespace NUMINAMATH_GPT_lucy_flour_used_l26_2610

theorem lucy_flour_used
  (initial_flour : ℕ := 500)
  (final_flour : ℕ := 130)
  (flour_needed_to_buy : ℤ := 370)
  (used_flour : ℕ) :
  initial_flour - used_flour = 2 * final_flour → used_flour = 240 :=
by
  sorry

end NUMINAMATH_GPT_lucy_flour_used_l26_2610


namespace NUMINAMATH_GPT_pqrs_l26_2603

theorem pqrs(p q r s t u : ℤ) :
  (729 * (x : ℤ) * x * x + 64 = (p * x * x + q * x + r) * (s * x * x + t * x + u)) →
  p = 9 → q = 4 → r = 0 → s = 81 → t = -36 → u = 16 →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 8210 := by
  intros h1 hp hq hr hs ht hu
  sorry

end NUMINAMATH_GPT_pqrs_l26_2603


namespace NUMINAMATH_GPT_cost_to_fill_half_of_CanB_l26_2633

theorem cost_to_fill_half_of_CanB (r h : ℝ) (C_cost : ℝ) (VC VB : ℝ) 
(h1 : VC = 2 * VB) 
(h2 : VB = Real.pi * r^2 * h) 
(h3 : VC = Real.pi * (2 * r)^2 * (h / 2)) 
(h4 : C_cost = 16):
  C_cost / 4 = 4 :=
by
  sorry

end NUMINAMATH_GPT_cost_to_fill_half_of_CanB_l26_2633


namespace NUMINAMATH_GPT_convert_to_base_k_l26_2646

noncomputable def base_k_eq (k : ℕ) : Prop :=
  4 * k + 4 = 36

theorem convert_to_base_k :
  ∃ k : ℕ, base_k_eq k ∧ (67 / k^2 % k^2 % k = 1 ∧ 67 / k % k = 0 ∧ 67 % k = 3) :=
sorry

end NUMINAMATH_GPT_convert_to_base_k_l26_2646


namespace NUMINAMATH_GPT_slope_product_is_neg_one_l26_2613

noncomputable def slope_product (m n : ℝ) : ℝ := m * n

theorem slope_product_is_neg_one 
  (m n : ℝ)
  (eqn1 : ∀ x, ∃ y, y = m * x)
  (eqn2 : ∀ x, ∃ y, y = n * x)
  (angle : ∃ θ1 θ2 : ℝ, θ1 = θ2 + π / 4)
  (neg_reciprocal : m = -1 / n):
  slope_product m n = -1 := 
sorry

end NUMINAMATH_GPT_slope_product_is_neg_one_l26_2613


namespace NUMINAMATH_GPT_meeting_success_probability_l26_2629

noncomputable def meeting_probability : ℝ :=
  let totalVolume := 1.5 ^ 3
  let z_gt_x_y := (1.5 * 1.5 * 1.5) / 3
  let assistants_leave := 2 * ((1.5 * 0.5 / 2) / 3 * 0.5)
  let effectiveVolume := z_gt_x_y - assistants_leave
  let probability := effectiveVolume / totalVolume
  probability

theorem meeting_success_probability :
  meeting_probability = 8 / 27 := by
  sorry

end NUMINAMATH_GPT_meeting_success_probability_l26_2629


namespace NUMINAMATH_GPT_product_mb_gt_one_l26_2641

theorem product_mb_gt_one (m b : ℝ) (hm : m = 3 / 4) (hb : b = 2) : m * b = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_product_mb_gt_one_l26_2641


namespace NUMINAMATH_GPT_sequence_is_increasing_l26_2640

-- Define the sequence recurrence property
def sequence_condition (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = 3

-- The theorem statement
theorem sequence_is_increasing (a : ℕ → ℤ) (h : sequence_condition a) : 
  ∀ n : ℕ, a n < a (n + 1) :=
by
  unfold sequence_condition at h
  intro n
  specialize h n
  sorry

end NUMINAMATH_GPT_sequence_is_increasing_l26_2640


namespace NUMINAMATH_GPT_simplify_expression_l26_2619

variable (a b : ℝ)

theorem simplify_expression (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b) :
  (3 * (a^2 + a * b + b^2) / (4 * (a + b))) * (2 * (a^2 - b^2) / (9 * (a^3 - b^3))) = 
  1 / 6 := 
by
  -- Placeholder for proof steps
  sorry

end NUMINAMATH_GPT_simplify_expression_l26_2619


namespace NUMINAMATH_GPT_shortest_side_of_triangle_l26_2627

noncomputable def triangle_shortest_side (AB : ℝ) (AD : ℝ) (DB : ℝ) (radius : ℝ) : ℝ :=
  let x := 6
  let y := 5
  2 * y

theorem shortest_side_of_triangle :
  let AB := 16
  let AD := 7
  let DB := 9
  let radius := 5
  AB = AD + DB →
  (AD = 7) ∧ (DB = 9) ∧ (radius = 5) →
  triangle_shortest_side AB AD DB radius = 10 :=
by
  intros h1 h2
  -- proof goes here
  sorry

end NUMINAMATH_GPT_shortest_side_of_triangle_l26_2627


namespace NUMINAMATH_GPT_part_I_part_II_part_III_l26_2612

noncomputable def f (x : ℝ) := x / (x^2 - 1)

-- (I) Prove that f(2) = 2/3.
theorem part_I : f 2 = 2 / 3 :=
by sorry

-- (II) Prove that f(x) is decreasing on the interval (-1, 1).
theorem part_II : ∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → f x1 > f x2 :=
by sorry

-- (III) Prove that f(x) is an odd function.
theorem part_III : ∀ x : ℝ, f (-x) = -f x :=
by sorry

end NUMINAMATH_GPT_part_I_part_II_part_III_l26_2612


namespace NUMINAMATH_GPT_smallest_n_sqrt_12n_integer_l26_2601

theorem smallest_n_sqrt_12n_integer : ∃ n : ℕ, (n > 0) ∧ (∃ k : ℕ, 12 * n = k^2) ∧ n = 3 := by
  sorry

end NUMINAMATH_GPT_smallest_n_sqrt_12n_integer_l26_2601


namespace NUMINAMATH_GPT_find_x_l26_2654

theorem find_x :
    ∃ x : ℚ, (1/7 + 7/x = 15/x + 1/15) ∧ x = 105 := by
  sorry

end NUMINAMATH_GPT_find_x_l26_2654


namespace NUMINAMATH_GPT_pirates_divide_coins_l26_2649

theorem pirates_divide_coins (N : ℕ) (hN : 220 ≤ N ∧ N ≤ 300) :
  ∃ n : ℕ, 
    (N - 2 - (N - 2) / 3 - 2 - (2 * ((N - 2) / 3 - (2 * ((N - 2) / 3) / 3)) / 3) - 
    2 - (2 * (((N - 2) / 3 - (2 * ((N - 2) / 3) / 3)) / 3)) / 3) / 3 = 84 := 
sorry

end NUMINAMATH_GPT_pirates_divide_coins_l26_2649


namespace NUMINAMATH_GPT_negation_of_exists_proposition_l26_2655

theorem negation_of_exists_proposition :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) :=
sorry

end NUMINAMATH_GPT_negation_of_exists_proposition_l26_2655


namespace NUMINAMATH_GPT_range_of_a_l26_2688

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > a then x + 2 else x^2 + 5 * x + 2

noncomputable def g (x : ℝ) (a : ℝ) : ℝ :=
f x a - 2 * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, g x a = 0 → (x = 2 ∨ x = -1 ∨ x = -2)) ↔ (-1 ≤ a ∧ a < 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l26_2688


namespace NUMINAMATH_GPT_trigonometric_identity_l26_2658

variable {a b c A B C : ℝ}

theorem trigonometric_identity (h1 : 2 * c^2 - 2 * a^2 = b^2) 
  (cos_A : ℝ) (cos_C : ℝ) 
  (h_cos_A : cos_A = (b^2 + c^2 - a^2) / (2 * b * c))
  (h_cos_C : cos_C = (a^2 + b^2 - c^2) / (2 * a * b)) :
  2 * c * cos_A - 2 * a * cos_C = b := 
sorry

end NUMINAMATH_GPT_trigonometric_identity_l26_2658


namespace NUMINAMATH_GPT_rowing_downstream_speed_l26_2648

-- Define the given conditions
def V_u : ℝ := 60  -- speed upstream in kmph
def V_s : ℝ := 75  -- speed in still water in kmph

-- Define the problem statement
theorem rowing_downstream_speed : ∃ (V_d : ℝ), V_s = (V_u + V_d) / 2 ∧ V_d = 90 :=
by
  sorry

end NUMINAMATH_GPT_rowing_downstream_speed_l26_2648


namespace NUMINAMATH_GPT_cost_of_drill_bits_l26_2631

theorem cost_of_drill_bits (x : ℝ) (h1 : 5 * x + 0.10 * (5 * x) = 33) : x = 6 :=
sorry

end NUMINAMATH_GPT_cost_of_drill_bits_l26_2631


namespace NUMINAMATH_GPT_function_equation_l26_2669

noncomputable def f (n : ℕ) : ℕ := sorry

theorem function_equation (h : ∀ m n : ℕ, m > 0 → n > 0 →
  f (f (f m) + 2 * f (f n)) = m^2 + 2 * n^2) : 
  ∀ n : ℕ, n > 0 → f n = n := 
sorry

end NUMINAMATH_GPT_function_equation_l26_2669


namespace NUMINAMATH_GPT_total_handshakes_l26_2662

-- Define the groups and their properties
def GroupA := 30
def GroupB := 15
def GroupC := 5
def KnowEachOtherA := true -- All 30 people in Group A know each other
def KnowFromB := 10 -- Each person in Group B knows 10 people from Group A
def KnowNoOneC := true -- Each person in Group C knows no one

-- Define the number of handshakes based on the conditions
def handshakes_between_A_and_B : Nat := GroupB * (GroupA - KnowFromB)
def handshakes_between_B_and_C : Nat := GroupB * GroupC
def handshakes_within_C : Nat := (GroupC * (GroupC - 1)) / 2
def handshakes_between_A_and_C : Nat := GroupA * GroupC

-- Prove the total number of handshakes
theorem total_handshakes : 
  handshakes_between_A_and_B +
  handshakes_between_B_and_C +
  handshakes_within_C +
  handshakes_between_A_and_C = 535 :=
by sorry

end NUMINAMATH_GPT_total_handshakes_l26_2662


namespace NUMINAMATH_GPT_values_of_j_for_exactly_one_real_solution_l26_2686

open Real

theorem values_of_j_for_exactly_one_real_solution :
  ∀ j : ℝ, (∀ x : ℝ, (3 * x + 4) * (x - 6) = -51 + j * x) → (j = 0 ∨ j = -36) := by
sorry

end NUMINAMATH_GPT_values_of_j_for_exactly_one_real_solution_l26_2686


namespace NUMINAMATH_GPT_perfect_square_trinomial_solution_l26_2681

theorem perfect_square_trinomial_solution (m : ℝ) :
  (∃ a : ℝ, (∀ x : ℝ, x^2 - 2*(m+3)*x + 9 = (x - a)^2))
  → m = 0 ∨ m = -6 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_solution_l26_2681


namespace NUMINAMATH_GPT_line_through_midpoint_bisects_chord_eqn_l26_2602

theorem line_through_midpoint_bisects_chord_eqn :
  ∀ (x y : ℝ), (x^2 - 4*y^2 = 4) ∧ (∃ x1 y1 x2 y2 : ℝ, 
    (x1^2 - 4 * y1^2 = 4) ∧ (x2^2 - 4 * y2^2 = 4) ∧ 
    (x1 + x2) / 2 = 3 ∧ (y1 + y2) / 2 = -1) → 
    3 * x + 4 * y - 5 = 0 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_line_through_midpoint_bisects_chord_eqn_l26_2602


namespace NUMINAMATH_GPT_half_difference_donation_l26_2608

def margoDonation : ℝ := 4300
def julieDonation : ℝ := 4700

theorem half_difference_donation : (julieDonation - margoDonation) / 2 = 200 := by
  sorry

end NUMINAMATH_GPT_half_difference_donation_l26_2608


namespace NUMINAMATH_GPT_group4_equations_groupN_equations_find_k_pos_l26_2665

-- Conditions from the problem
def group1_fractions := (1 : ℚ) / 1 + (1 : ℚ) / 3 = 4 / 3
def group1_pythagorean := 4^2 + 3^2 = 5^2

def group2_fractions := (1 : ℚ) / 3 + (1 : ℚ) / 5 = 8 / 15
def group2_pythagorean := 8^2 + 15^2 = 17^2

def group3_fractions := (1 : ℚ) / 5 + (1 : ℚ) / 7 = 12 / 35
def group3_pythagorean := 12^2 + 35^2 = 37^2

-- Proof Statements
theorem group4_equations :
  ((1 : ℚ) / 7 + (1 : ℚ) / 9 = 16 / 63) ∧ (16^2 + 63^2 = 65^2) := 
  sorry

theorem groupN_equations (n : ℕ) :
  ((1 : ℚ) / (2 * n - 1) + (1 : ℚ) / (2 * n + 1) = 4 * n / (4 * n^2 - 1)) ∧
  ((4 * n)^2 + (4 * n^2 - 1)^2 = (4 * n^2 + 1)^2) :=
  sorry

theorem find_k_pos (k : ℕ) : 
  k^2 + 9603^2 = 9605^2 → k = 196 := 
  sorry

end NUMINAMATH_GPT_group4_equations_groupN_equations_find_k_pos_l26_2665


namespace NUMINAMATH_GPT_intersection_P_Q_l26_2630

def set_P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def set_Q : Set ℝ := {x | (x - 1) ^ 2 ≤ 4}

theorem intersection_P_Q :
  {x | x ∈ set_P ∧ x ∈ set_Q} = {x : ℝ | 1 ≤ x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_GPT_intersection_P_Q_l26_2630


namespace NUMINAMATH_GPT_smallest_xyz_sum_l26_2672

theorem smallest_xyz_sum (x y z : ℕ) (h1 : (x + y) * (y + z) = 2016) (h2 : (x + y) * (z + x) = 1080) :
  x > 0 → y > 0 → z > 0 → x + y + z = 61 :=
  sorry

end NUMINAMATH_GPT_smallest_xyz_sum_l26_2672


namespace NUMINAMATH_GPT_distance_on_map_is_correct_l26_2657

-- Define the parameters
def time_hours : ℝ := 1.5
def speed_mph : ℝ := 60
def map_scale_inches_per_mile : ℝ := 0.05555555555555555

-- Define the computation of actual distance and distance on the map
def actual_distance_miles : ℝ := speed_mph * time_hours
def distance_on_map_inches : ℝ := actual_distance_miles * map_scale_inches_per_mile

-- Theorem statement
theorem distance_on_map_is_correct :
  distance_on_map_inches = 5 :=
by 
  sorry

end NUMINAMATH_GPT_distance_on_map_is_correct_l26_2657


namespace NUMINAMATH_GPT_ratio_brothers_sisters_boys_ratio_brothers_sisters_girls_l26_2637

variables (x y k t : ℕ)

theorem ratio_brothers_sisters_boys (h1 : (x+1) / y = k) (h2 : x / (y+1) = t) :
  (x / (y+1)) = t := 
by simp [h2]

theorem ratio_brothers_sisters_girls (h1 : (x+1) / y = k) (h2 : x / (y+1) = t) :
  ((x+1) / y) = k := 
by simp [h1]

#check ratio_brothers_sisters_boys    -- Just for verification
#check ratio_brothers_sisters_girls   -- Just for verification

end NUMINAMATH_GPT_ratio_brothers_sisters_boys_ratio_brothers_sisters_girls_l26_2637


namespace NUMINAMATH_GPT_dealer_gross_profit_l26_2661

variable (purchase_price : ℝ) (markup_rate : ℝ) (gross_profit : ℝ)

def desk_problem (purchase_price : ℝ) (markup_rate : ℝ) (gross_profit : ℝ) : Prop :=
  ∀ (S : ℝ), S = purchase_price + markup_rate * S → gross_profit = S - purchase_price

theorem dealer_gross_profit : desk_problem 150 0.5 150 :=
by 
  sorry

end NUMINAMATH_GPT_dealer_gross_profit_l26_2661


namespace NUMINAMATH_GPT_find_x_l26_2623

theorem find_x (x y : ℝ) (h1 : y = 1 / (2 * x + 2)) (h2 : y = 2) : x = -3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l26_2623


namespace NUMINAMATH_GPT_more_than_3000_students_l26_2652

-- Define the conditions
def students_know_secret (n : ℕ) : ℕ :=
  3 ^ (n - 1)

-- Define the statement to prove
theorem more_than_3000_students : ∃ n : ℕ, students_know_secret n > 3000 ∧ n = 9 := by
  sorry

end NUMINAMATH_GPT_more_than_3000_students_l26_2652


namespace NUMINAMATH_GPT_painting_two_sides_time_l26_2696

-- Definitions for the conditions
def time_to_paint_one_side_per_board : Nat := 1
def drying_time_per_board : Nat := 5

-- Definitions for the problem
def total_boards : Nat := 6

-- Main theorem statement
theorem painting_two_sides_time :
  (total_boards * time_to_paint_one_side_per_board) + drying_time_per_board + (total_boards * time_to_paint_one_side_per_board) = 12 :=
sorry

end NUMINAMATH_GPT_painting_two_sides_time_l26_2696


namespace NUMINAMATH_GPT_hyperbola_is_given_equation_l26_2604

noncomputable def hyperbola_equation : Prop :=
  ∃ a b : ℝ, 
    (a > 0 ∧ b > 0) ∧ 
    (4^2 = a^2 + b^2) ∧ 
    (a = b) ∧ 
    (∀ x y : ℝ, (x^2 / 8 - y^2 / 8 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1))

theorem hyperbola_is_given_equation : hyperbola_equation :=
sorry

end NUMINAMATH_GPT_hyperbola_is_given_equation_l26_2604


namespace NUMINAMATH_GPT_john_total_feet_climbed_l26_2691

def first_stair_steps : ℕ := 20
def second_stair_steps : ℕ := 2 * first_stair_steps
def third_stair_steps : ℕ := second_stair_steps - 10
def step_height : ℝ := 0.5

theorem john_total_feet_climbed : 
  (first_stair_steps + second_stair_steps + third_stair_steps) * step_height = 45 :=
by
  sorry

end NUMINAMATH_GPT_john_total_feet_climbed_l26_2691


namespace NUMINAMATH_GPT_number_of_people_per_van_l26_2621

theorem number_of_people_per_van (num_students : ℕ) (num_adults : ℕ) (num_vans : ℕ) (total_people : ℕ) (people_per_van : ℕ) :
  num_students = 40 →
  num_adults = 14 →
  num_vans = 6 →
  total_people = num_students + num_adults →
  people_per_van = total_people / num_vans →
  people_per_van = 9 :=
by
  intros h_students h_adults h_vans h_total h_div
  sorry

end NUMINAMATH_GPT_number_of_people_per_van_l26_2621


namespace NUMINAMATH_GPT_find_triangle_sides_l26_2600

theorem find_triangle_sides (x y : ℕ) : 
  (x * y = 200) ∧ (x + 2 * y = 50) → ((x = 40 ∧ y = 5) ∨ (x = 10 ∧ y = 20)) := 
by
  intro h
  sorry

end NUMINAMATH_GPT_find_triangle_sides_l26_2600


namespace NUMINAMATH_GPT_farmer_feed_full_price_l26_2606

theorem farmer_feed_full_price
  (total_spent : ℕ)
  (chicken_feed_discount_percent : ℕ)
  (chicken_feed_percent : ℕ)
  (goat_feed_percent : ℕ)
  (total_spent_val : total_spent = 35)
  (chicken_feed_discount_percent_val : chicken_feed_discount_percent = 50)
  (chicken_feed_percent_val : chicken_feed_percent = 40)
  (goat_feed_percent_val : goat_feed_percent = 60) :
  (total_spent * chicken_feed_percent / 100 * 2) + (total_spent * goat_feed_percent / 100) = 49 := 
by
  -- Placeholder for proof.
  sorry

end NUMINAMATH_GPT_farmer_feed_full_price_l26_2606


namespace NUMINAMATH_GPT_sum_of_squares_of_coeffs_l26_2639

   theorem sum_of_squares_of_coeffs :
     let expr := 3 * (X^3 - 4 * X^2 + X) - 5 * (X^3 + 2 * X^2 - 5 * X + 3)
     let simplified_expr := -2 * X^3 - 22 * X^2 + 28 * X - 15
     let coefficients := [-2, -22, 28, -15]
     (coefficients.map (λ a => a^2)).sum = 1497 := 
   by 
     -- expending, simplifying and summing up the coefficients 
     sorry
   
end NUMINAMATH_GPT_sum_of_squares_of_coeffs_l26_2639


namespace NUMINAMATH_GPT_sand_exchange_impossible_to_achieve_l26_2643

-- Let G and P be the initial weights of gold and platinum sand, respectively
def initial_G : ℕ := 1 -- 1 kg
def initial_P : ℕ := 1 -- 1 kg

-- Initial values for g and p
def initial_g : ℕ := 1001
def initial_p : ℕ := 1001

-- Daily reduction of either g or p
axiom decrease_g_or_p (g p : ℕ) : g > 1 ∨ p > 1 → (g = g - 1 ∨ p = p - 1) ∧ (g ≥ 1) ∧ (p ≥ 1)

-- Final condition: after 2000 days, g and p both equal to 1
axiom final_g_p_after_2000_days : ∀ (g p : ℕ), (g = initial_g - 2000) ∧ (p = initial_p - 2000) → g = 1 ∧ p = 1

-- State of the system, defined as S = G * p + P * g
def S (G P g p : ℕ) : ℕ := G * p + P * g

-- Prove that after 2000 days, the banker cannot have at least 2 kg of each type of sand
theorem sand_exchange_impossible_to_achieve (G P g p : ℕ) (h : G = initial_G) (h1 : P = initial_P) 
  (h2 : g = initial_g) (h3 : p = initial_p) : 
  ∀ (d : ℕ), (d = 2000) → (g = 1) ∧ (p = 1) 
    → (S G P g p < 4) :=
by
  sorry

end NUMINAMATH_GPT_sand_exchange_impossible_to_achieve_l26_2643


namespace NUMINAMATH_GPT_estimate_passed_students_l26_2615

-- Definitions for the given conditions
def total_papers_in_city : ℕ := 5000
def papers_selected : ℕ := 400
def papers_passed : ℕ := 360

-- The theorem stating the problem in Lean
theorem estimate_passed_students : 
    (5000:ℕ) * ((360:ℕ) / (400:ℕ)) = (4500:ℕ) :=
by
  -- Providing a trivial sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_estimate_passed_students_l26_2615


namespace NUMINAMATH_GPT_rigid_motion_pattern_l26_2653

-- Define the types of transformations
inductive Transformation
| rotation : ℝ → Transformation -- rotation by an angle
| translation : ℝ → Transformation -- translation by a distance
| reflection_across_m : Transformation -- reflection across line m
| reflection_perpendicular_to_m : ℝ → Transformation -- reflective across line perpendicular to m at a point

-- Define the problem statement conditions
def pattern_alternates (line_m : ℝ → ℝ) : Prop := sorry -- This should define the alternating pattern of equilateral triangles and squares along line m

-- Problem statement in Lean
theorem rigid_motion_pattern (line_m : ℝ → ℝ) (p : Transformation → Prop)
    (h1 : p (Transformation.rotation 180)) -- 180-degree rotation is a valid transformation for the pattern
    (h2 : ∀ d, p (Transformation.translation d)) -- any translation by pattern unit length is a valid transformation
    (h3 : p Transformation.reflection_across_m) -- reflection across line m is a valid transformation
    (h4 : ∀ x, p (Transformation.reflection_perpendicular_to_m x)) -- reflection across any perpendicular line is a valid transformation
    : ∃ t : Finset Transformation, t.card = 4 ∧ ∀ t_val, t_val ∈ t → p t_val ∧ t_val ≠ Transformation.rotation 0 := 
sorry

end NUMINAMATH_GPT_rigid_motion_pattern_l26_2653


namespace NUMINAMATH_GPT_sin_double_angle_condition_l26_2670

theorem sin_double_angle_condition (θ : ℝ) (h : Real.sin θ + Real.cos θ = 1 / 3) : Real.sin (2 * θ) = -8 / 9 := 
sorry

end NUMINAMATH_GPT_sin_double_angle_condition_l26_2670


namespace NUMINAMATH_GPT_diagonal_angle_with_plane_l26_2650

theorem diagonal_angle_with_plane (α : ℝ) {a : ℝ} 
  (h_square: a > 0)
  (θ : ℝ := Real.arcsin ((Real.sin α) / Real.sqrt 2)): 
  ∃ (β : ℝ), β = θ :=
sorry

end NUMINAMATH_GPT_diagonal_angle_with_plane_l26_2650


namespace NUMINAMATH_GPT_range_of_a_for_min_value_at_x_eq_1_l26_2674

noncomputable def f (a x : ℝ) : ℝ := a*x^3 + (a-1)*x^2 - x + 2

theorem range_of_a_for_min_value_at_x_eq_1 :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f a 1 ≤ f a x) → a ≤ 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_min_value_at_x_eq_1_l26_2674
