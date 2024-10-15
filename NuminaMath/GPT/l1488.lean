import Mathlib

namespace NUMINAMATH_GPT_brother_age_in_5_years_l1488_148847

noncomputable def Nick : ℕ := 13
noncomputable def Sister : ℕ := Nick + 6
noncomputable def CombinedAge : ℕ := Nick + Sister
noncomputable def Brother : ℕ := CombinedAge / 2

theorem brother_age_in_5_years : Brother + 5 = 21 := by
  sorry

end NUMINAMATH_GPT_brother_age_in_5_years_l1488_148847


namespace NUMINAMATH_GPT_rowing_time_ratio_l1488_148896

def V_b : ℕ := 57
def V_s : ℕ := 19
def V_up : ℕ := V_b - V_s
def V_down : ℕ := V_b + V_s

theorem rowing_time_ratio :
  ∀ (T_up T_down : ℕ), V_up * T_up = V_down * T_down → T_up = 2 * T_down :=
by
  intros T_up T_down h
  sorry

end NUMINAMATH_GPT_rowing_time_ratio_l1488_148896


namespace NUMINAMATH_GPT_remaining_area_after_cut_l1488_148869

theorem remaining_area_after_cut
  (cell_side_length : ℝ)
  (grid_side_length : ℕ)
  (total_area : ℝ)
  (removed_area : ℝ)
  (hyp1 : cell_side_length = 1)
  (hyp2 : grid_side_length = 6)
  (hyp3 : total_area = (grid_side_length * grid_side_length) * cell_side_length * cell_side_length) 
  (hyp4 : removed_area = 9) :
  total_area - removed_area = 27 := by
  sorry

end NUMINAMATH_GPT_remaining_area_after_cut_l1488_148869


namespace NUMINAMATH_GPT_no_n_ge_1_such_that_sum_is_perfect_square_l1488_148835

theorem no_n_ge_1_such_that_sum_is_perfect_square :
  ¬ ∃ n : ℕ, n ≥ 1 ∧ ∃ k : ℕ, 2^n + 12^n + 2014^n = k^2 :=
by
  sorry

end NUMINAMATH_GPT_no_n_ge_1_such_that_sum_is_perfect_square_l1488_148835


namespace NUMINAMATH_GPT_day_of_week_after_10_pow_90_days_l1488_148811

theorem day_of_week_after_10_pow_90_days :
  let initial_day := "Friday"
  ∃ day_after_10_pow_90 : String,
  day_after_10_pow_90 = "Saturday" :=
by
  sorry

end NUMINAMATH_GPT_day_of_week_after_10_pow_90_days_l1488_148811


namespace NUMINAMATH_GPT_lcm_4_6_9_l1488_148810

/-- The least common multiple (LCM) of 4, 6, and 9 is 36 -/
theorem lcm_4_6_9 : Nat.lcm (Nat.lcm 4 6) 9 = 36 :=
by
  -- sorry replaces the actual proof steps
  sorry

end NUMINAMATH_GPT_lcm_4_6_9_l1488_148810


namespace NUMINAMATH_GPT_minimum_value_l1488_148876

theorem minimum_value (x y z : ℝ) (h : x + 2 * y + z = 1) : x^2 + 4 * y^2 + z^2 ≥ 1 / 3 :=
sorry

end NUMINAMATH_GPT_minimum_value_l1488_148876


namespace NUMINAMATH_GPT_line_through_fixed_point_l1488_148818

theorem line_through_fixed_point (a : ℝ) :
  ∃ P : ℝ × ℝ, (P = (1, 2)) ∧ (∀ x y, a * x + y - a - 2 = 0 → P = (x, y)) ∧
  ((∃ a, x + y = a ∧ x = 1 ∧ y = 2) → (a = 3)) :=
by
  sorry

end NUMINAMATH_GPT_line_through_fixed_point_l1488_148818


namespace NUMINAMATH_GPT_translation_preserves_coordinates_l1488_148889

-- Given coordinates of point P
def point_P : (Int × Int) := (-2, 3)

-- Translating point P 3 units in the positive direction of the x-axis
def translate_x (p : Int × Int) (dx : Int) : (Int × Int) := 
  (p.1 + dx, p.2)

-- Translating point P 2 units in the negative direction of the y-axis
def translate_y (p : Int × Int) (dy : Int) : (Int × Int) := 
  (p.1, p.2 - dy)

-- Final coordinates after both translations
def final_coordinates (p : Int × Int) (dx dy : Int) : (Int × Int) := 
  translate_y (translate_x p dx) dy

theorem translation_preserves_coordinates :
  final_coordinates point_P 3 2 = (1, 1) :=
by
  sorry

end NUMINAMATH_GPT_translation_preserves_coordinates_l1488_148889


namespace NUMINAMATH_GPT_rational_zero_quadratic_roots_l1488_148826

-- Part 1
theorem rational_zero (a b : ℚ) (h : a + b * Real.sqrt 5 = 0) : a = 0 ∧ b = 0 :=
sorry

-- Part 2
theorem quadratic_roots (k : ℝ) (h : k ≠ 0) (x1 x2 : ℝ)
  (h1 : 4 * k * x1^2 - 4 * k * x1 + k + 1 = 0)
  (h2 : 4 * k * x2^2 - 4 * k * x2 + k + 1 = 0)
  (h3 : x1 ≠ x2) 
  (h4 : x1^2 + x2^2 - 2 * x1 * x2 = 0.5) : k = -2 :=
sorry

end NUMINAMATH_GPT_rational_zero_quadratic_roots_l1488_148826


namespace NUMINAMATH_GPT_tim_has_33_books_l1488_148894

-- Define the conditions
def b := 24   -- Benny's initial books
def s := 10   -- Books given to Sandy
def total_books : Nat := 47  -- Total books

-- Define the remaining books after Benny gives to Sandy
def remaining_b : Nat := b - s

-- Define Tim's books
def tim_books : Nat := total_books - remaining_b

-- Prove that Tim has 33 books
theorem tim_has_33_books : tim_books = 33 := by
  -- This is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_tim_has_33_books_l1488_148894


namespace NUMINAMATH_GPT_square_plot_area_l1488_148841

theorem square_plot_area (cost_per_foot : ℕ) (total_cost : ℕ) (P : ℕ) :
  cost_per_foot = 54 →
  total_cost = 3672 →
  P = 4 * (total_cost / (4 * cost_per_foot)) →
  (total_cost / (4 * cost_per_foot)) ^ 2 = 289 :=
by
  intros h_cost_per_foot h_total_cost h_perimeter
  sorry

end NUMINAMATH_GPT_square_plot_area_l1488_148841


namespace NUMINAMATH_GPT_exists_f_ff_eq_square_l1488_148840

open Nat

theorem exists_f_ff_eq_square : ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n ^ 2 :=
by
  -- proof to be provided
  sorry

end NUMINAMATH_GPT_exists_f_ff_eq_square_l1488_148840


namespace NUMINAMATH_GPT_breadth_of_added_rectangle_l1488_148816

theorem breadth_of_added_rectangle 
  (s : ℝ) (b : ℝ) 
  (h_square_side : s = 8) 
  (h_perimeter_new_rectangle : 2 * s + 2 * (s + b) = 40) : 
  b = 4 :=
by
  sorry

end NUMINAMATH_GPT_breadth_of_added_rectangle_l1488_148816


namespace NUMINAMATH_GPT_wrench_weight_relation_l1488_148825

variables (h w : ℕ)

theorem wrench_weight_relation (h w : ℕ) 
  (cond : 2 * h + 2 * w = (1 / 3) * (8 * h + 5 * w)) : w = 2 * h := 
by sorry

end NUMINAMATH_GPT_wrench_weight_relation_l1488_148825


namespace NUMINAMATH_GPT_mix_solutions_l1488_148877

theorem mix_solutions {x : ℝ} (h : 0.60 * x + 0.75 * (20 - x) = 0.72 * 20) : x = 4 :=
by
-- skipping the proof with sorry
sorry

end NUMINAMATH_GPT_mix_solutions_l1488_148877


namespace NUMINAMATH_GPT_maple_trees_cut_down_l1488_148891

-- Define the initial number of maple trees.
def initial_maple_trees : ℝ := 9.0

-- Define the final number of maple trees after cutting.
def final_maple_trees : ℝ := 7.0

-- Define the number of maple trees cut down.
def cut_down_maple_trees : ℝ := initial_maple_trees - final_maple_trees

-- Prove that the number of cut down maple trees is 2.
theorem maple_trees_cut_down : cut_down_maple_trees = 2 := by
  sorry

end NUMINAMATH_GPT_maple_trees_cut_down_l1488_148891


namespace NUMINAMATH_GPT_number_of_long_sleeved_jerseys_l1488_148855

def cost_per_long_sleeved := 15
def cost_per_striped := 10
def num_striped_jerseys := 2
def total_spent := 80

theorem number_of_long_sleeved_jerseys (x : ℕ) :
  total_spent = cost_per_long_sleeved * x + cost_per_striped * num_striped_jerseys →
  x = 4 := by
  sorry

end NUMINAMATH_GPT_number_of_long_sleeved_jerseys_l1488_148855


namespace NUMINAMATH_GPT_max_omega_is_2_l1488_148863

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 6)

theorem max_omega_is_2 {ω : ℝ} (h₀ : ω > 0) (h₁ : MonotoneOn (f ω) (Set.Icc (-Real.pi / 6) (Real.pi / 6))) :
  ω ≤ 2 :=
sorry

end NUMINAMATH_GPT_max_omega_is_2_l1488_148863


namespace NUMINAMATH_GPT_fraction_to_decimal_l1488_148807

theorem fraction_to_decimal : (7 : ℝ) / 250 = 0.028 := 
sorry

end NUMINAMATH_GPT_fraction_to_decimal_l1488_148807


namespace NUMINAMATH_GPT_locus_of_center_l1488_148823

-- Define point A
def PointA : ℝ × ℝ := (-2, 0)

-- Define the tangent line
def TangentLine : ℝ := 2

-- The condition to prove the locus equation
theorem locus_of_center (x₀ y₀ : ℝ) :
  (∃ r : ℝ, abs (x₀ - TangentLine) = r ∧ (x₀ + 2)^2 + y₀^2 = r^2) →
  y₀^2 = -8 * x₀ := by
  sorry

end NUMINAMATH_GPT_locus_of_center_l1488_148823


namespace NUMINAMATH_GPT_dave_spent_102_dollars_l1488_148800

noncomputable def total_cost (books_animals books_space books_trains cost_per_book : ℕ) : ℕ :=
  (books_animals + books_space + books_trains) * cost_per_book

theorem dave_spent_102_dollars :
  total_cost 8 6 3 6 = 102 := by
  sorry

end NUMINAMATH_GPT_dave_spent_102_dollars_l1488_148800


namespace NUMINAMATH_GPT_min_value_sin_cos_l1488_148859

open Real

theorem min_value_sin_cos : ∃ x : ℝ, sin x * cos x = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_min_value_sin_cos_l1488_148859


namespace NUMINAMATH_GPT_multiplication_digit_sum_l1488_148831

theorem multiplication_digit_sum :
  let a := 879
  let b := 492
  let product := a * b
  let sum_of_digits := (4 + 3 + 2 + 4 + 6 + 8)
  product = 432468 ∧ sum_of_digits = 27 := by
  -- Step 1: Set up the given numbers
  let a := 879
  let b := 492

  -- Step 2: Calculate the product
  let product := a * b
  have product_eq : product = 432468 := by
    sorry

  -- Step 3: Sum the digits of the product
  let sum_of_digits := (4 + 3 + 2 + 4 + 6 + 8)
  have sum_of_digits_eq : sum_of_digits = 27 := by
    sorry

  -- Conclusion
  exact ⟨product_eq, sum_of_digits_eq⟩

end NUMINAMATH_GPT_multiplication_digit_sum_l1488_148831


namespace NUMINAMATH_GPT_product_f_g_l1488_148850

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x * (x + 1))
noncomputable def g (x : ℝ) : ℝ := 1 / Real.sqrt x

theorem product_f_g (x : ℝ) (hx : 0 < x) : f x * g x = Real.sqrt (x + 1) := 
by 
  sorry

end NUMINAMATH_GPT_product_f_g_l1488_148850


namespace NUMINAMATH_GPT_cost_per_blue_shirt_l1488_148886

theorem cost_per_blue_shirt :
  let pto_spent := 2317
  let num_kindergarten := 101
  let cost_orange := 5.80
  let total_orange := num_kindergarten * cost_orange

  let num_first_grade := 113
  let cost_yellow := 5
  let total_yellow := num_first_grade * cost_yellow

  let num_third_grade := 108
  let cost_green := 5.25
  let total_green := num_third_grade * cost_green

  let total_other_shirts := total_orange + total_yellow + total_green
  let pto_spent_on_blue := pto_spent - total_other_shirts

  let num_second_grade := 107
  let cost_per_blue_shirt := pto_spent_on_blue / num_second_grade

  cost_per_blue_shirt = 5.60 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_blue_shirt_l1488_148886


namespace NUMINAMATH_GPT_part_I_part_II_l1488_148858

noncomputable def f (x : ℝ) (a : ℝ) := x - (2 * a - 1) / x - 2 * a * Real.log x

theorem part_I (a : ℝ) (h : a = 3 / 2) : 
  (∀ x, 0 < x ∧ x < 1 → f x a < 0) ∧ (∀ x, 1 < x ∧ x < 2 → f x a > 0) ∧ (∀ x, 2 < x → f x a < 0) := sorry

theorem part_II (a : ℝ) : (∀ x, 1 ≤ x → f x a ≥ 0) → a ≤ 1 := sorry

end NUMINAMATH_GPT_part_I_part_II_l1488_148858


namespace NUMINAMATH_GPT_describe_T_l1488_148843

def T : Set (ℝ × ℝ) := 
  { p | ∃ x y : ℝ, p = (x, y) ∧ (
      (5 = x + 3 ∧ y - 6 ≤ 5) ∨
      (5 = y - 6 ∧ x + 3 ≤ 5) ∨
      (x + 3 = y - 6 ∧ x + 3 ≤ 5 ∧ y - 6 ≤ 5)
  )}

theorem describe_T : T = { p | ∃ x y : ℝ, p = (2, y) ∧ y ≤ 11 ∨
                                      p = (x, 11) ∧ x ≤ 2 ∨
                                      p = (x, x + 9) ∧ x ≤ 2 ∧ x + 9 ≤ 11 } :=
by
  sorry

end NUMINAMATH_GPT_describe_T_l1488_148843


namespace NUMINAMATH_GPT_general_term_formula_l1488_148867

theorem general_term_formula (f : ℕ → ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ) :
  (∀ x, f x = 1 - 2^x) →
  (∀ n, f n = S n) →
  (∀ n, S n = 1 - 2^n) →
  (∀ n, n = 1 → a n = S 1) →
  (∀ n, n ≥ 2 → a n = S n - S (n-1)) →
  (∀ n, a n = -2^(n-1)) :=
by
  sorry

end NUMINAMATH_GPT_general_term_formula_l1488_148867


namespace NUMINAMATH_GPT_find_linear_function_b_l1488_148871

theorem find_linear_function_b (b : ℝ) :
  (∃ b, (∀ x y, y = 2 * x + b - 2 → (x = -1 ∧ y = 0)) → b = 4) :=
sorry

end NUMINAMATH_GPT_find_linear_function_b_l1488_148871


namespace NUMINAMATH_GPT_least_blue_eyes_and_snack_l1488_148812

variable (total_students blue_eyes students_with_snack : ℕ)

theorem least_blue_eyes_and_snack (h1 : total_students = 35) 
                                 (h2 : blue_eyes = 14) 
                                 (h3 : students_with_snack = 22) :
  ∃ n, n = 1 ∧ 
        ∀ k, (k < n → 
                 ∃ no_snack_no_blue : ℕ, no_snack_no_blue = total_students - students_with_snack ∧
                      no_snack_no_blue = blue_eyes - k) := 
by
  sorry

end NUMINAMATH_GPT_least_blue_eyes_and_snack_l1488_148812


namespace NUMINAMATH_GPT_daryl_age_l1488_148878

theorem daryl_age (d j : ℕ) 
  (h1 : d - 4 = 3 * (j - 4)) 
  (h2 : d + 5 = 2 * (j + 5)) :
  d = 31 :=
by sorry

end NUMINAMATH_GPT_daryl_age_l1488_148878


namespace NUMINAMATH_GPT_smallest_y_value_in_set_l1488_148830

theorem smallest_y_value_in_set : ∀ y : ℕ, (0 < y) ∧ (y + 4 ≤ 8) → y = 4 :=
by
  intros y h
  have h1 : y + 4 ≤ 8 := h.2
  have h2 : 0 < y := h.1
  sorry

end NUMINAMATH_GPT_smallest_y_value_in_set_l1488_148830


namespace NUMINAMATH_GPT_matrix_pow_101_l1488_148822

noncomputable def matrixA : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![0, 0, 1],
    ![1, 0, 0],
    ![0, 1, 0]
  ]

theorem matrix_pow_101 :
  matrixA ^ 101 =
  ![
    ![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]
  ] :=
sorry

end NUMINAMATH_GPT_matrix_pow_101_l1488_148822


namespace NUMINAMATH_GPT_sector_perimeter_l1488_148829

theorem sector_perimeter (R : ℝ) (α : ℝ) (A : ℝ) (P : ℝ) : 
  A = (1 / 2) * R^2 * α → 
  α = 4 → 
  A = 2 → 
  P = 2 * R + R * α → 
  P = 6 := 
by
  intros hArea hAlpha hA hP
  sorry

end NUMINAMATH_GPT_sector_perimeter_l1488_148829


namespace NUMINAMATH_GPT_cuboid_cutout_l1488_148839

theorem cuboid_cutout (x y : ℕ) (h1 : x * y = 36) (h2 : 0 < x) (h3 : x < 4) (h4 : 0 < y) (h5 : y < 15) :
  x + y = 15 :=
sorry

end NUMINAMATH_GPT_cuboid_cutout_l1488_148839


namespace NUMINAMATH_GPT_first_chapter_pages_calculation_l1488_148813

-- Define the constants and conditions
def second_chapter_pages : ℕ := 11
def first_chapter_pages_more : ℕ := 37

-- Main proof problem
theorem first_chapter_pages_calculation : first_chapter_pages_more + second_chapter_pages = 48 := by
  sorry

end NUMINAMATH_GPT_first_chapter_pages_calculation_l1488_148813


namespace NUMINAMATH_GPT_sum_of_m_and_n_l1488_148828

theorem sum_of_m_and_n (m n : ℝ) (h : m^2 + n^2 - 6 * m + 10 * n + 34 = 0) : m + n = -2 := 
sorry

end NUMINAMATH_GPT_sum_of_m_and_n_l1488_148828


namespace NUMINAMATH_GPT_negation_proposition_l1488_148820

theorem negation_proposition : 
  (¬ ∃ x_0 : ℝ, 2 * x_0 - 3 > 1) ↔ (∀ x : ℝ, 2 * x - 3 ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l1488_148820


namespace NUMINAMATH_GPT_yeast_cells_at_2_20_pm_l1488_148817

noncomputable def yeast_population (initial : Nat) (rate : Nat) (intervals : Nat) : Nat :=
  initial * rate ^ intervals

theorem yeast_cells_at_2_20_pm :
  let initial_population := 30
  let triple_rate := 3
  let intervals := 5 -- 20 minutes / 4 minutes per interval
  yeast_population initial_population triple_rate intervals = 7290 :=
by
  let initial_population := 30
  let triple_rate := 3
  let intervals := 5
  show yeast_population initial_population triple_rate intervals = 7290
  sorry

end NUMINAMATH_GPT_yeast_cells_at_2_20_pm_l1488_148817


namespace NUMINAMATH_GPT_rhombus_diagonal_l1488_148853

theorem rhombus_diagonal (a b : ℝ) (area_triangle : ℝ) (d1 d2 : ℝ)
  (h1 : 2 * area_triangle = a * b)
  (h2 : area_triangle = 75)
  (h3 : a = 20) :
  b = 15 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_l1488_148853


namespace NUMINAMATH_GPT_smallest_square_side_lengths_l1488_148870

theorem smallest_square_side_lengths (x : ℕ) 
    (h₁ : ∀ (y : ℕ), y = x + 8) 
    (h₂ : ∀ (z : ℕ), z = 50) 
    (h₃ : ∀ (QS PS RT QT : ℕ), QS = 8 ∧ PS = x ∧ RT = 42 - x ∧ QT = x + 8 ∧ (8 / x) = ((42 - x) / (x + 8))) : 
  x = 2 ∨ x = 32 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_square_side_lengths_l1488_148870


namespace NUMINAMATH_GPT_find_f_at_2_l1488_148801

variable {R : Type} [Ring R]

def f (a b x : R) : R := a * x ^ 3 + b * x - 3

theorem find_f_at_2 (a b : R) (h : f a b (-2) = 7) : f a b 2 = -13 := 
by 
  have h₁ : f a b (-2) + f a b 2 = -6 := sorry
  have h₂ : f a b 2 = -6 - f a b (-2) := sorry
  rw [h₂, h]
  norm_num

end NUMINAMATH_GPT_find_f_at_2_l1488_148801


namespace NUMINAMATH_GPT_area_between_tangent_circles_l1488_148880

theorem area_between_tangent_circles (r : ℝ) (h_r : r > 0) :
  let area_trapezoid := 4 * r^2 * Real.sqrt 3
  let area_sector1 := π * r^2 / 3
  let area_sector2 := 3 * π * r^2 / 2
  area_trapezoid - (area_sector1 + area_sector2) = r^2 * (24 * Real.sqrt 3 - 11 * π) / 6 := by
  sorry

end NUMINAMATH_GPT_area_between_tangent_circles_l1488_148880


namespace NUMINAMATH_GPT_freight_train_distance_l1488_148809

variable (travel_rate : ℕ) (initial_distance : ℕ) (time_minutes : ℕ) 

def total_distance_traveled (travel_rate : ℕ) (initial_distance : ℕ) (time_minutes : ℕ) : ℕ :=
  let traveled_distance := (time_minutes / travel_rate) 
  traveled_distance + initial_distance

theorem freight_train_distance :
  total_distance_traveled 2 5 90 = 50 :=
by
  sorry

end NUMINAMATH_GPT_freight_train_distance_l1488_148809


namespace NUMINAMATH_GPT_mandy_toys_count_l1488_148866

theorem mandy_toys_count (M A Am P : ℕ) 
    (h1 : A = 3 * M) 
    (h2 : A = Am - 2) 
    (h3 : A = P / 2) 
    (h4 : M + A + Am + P = 278) : 
    M = 21 := 
by
  sorry

end NUMINAMATH_GPT_mandy_toys_count_l1488_148866


namespace NUMINAMATH_GPT_number_of_nickels_is_three_l1488_148815

def coin_problem : Prop :=
  ∃ p n d q : ℕ,
    p + n + d + q = 12 ∧
    p + 5 * n + 10 * d + 25 * q = 128 ∧
    p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 1 ∧
    q = 2 * d ∧
    n = 3

theorem number_of_nickels_is_three : coin_problem := 
by 
  sorry

end NUMINAMATH_GPT_number_of_nickels_is_three_l1488_148815


namespace NUMINAMATH_GPT_circle_area_l1488_148832

theorem circle_area (C : ℝ) (hC : C = 24) : ∃ (A : ℝ), A = 144 / π :=
by
  sorry

end NUMINAMATH_GPT_circle_area_l1488_148832


namespace NUMINAMATH_GPT_max_banner_area_l1488_148890

theorem max_banner_area (x y : ℕ) (cost_constraint : 330 * x + 450 * y ≤ 10000) : x * y ≤ 165 :=
by
  sorry

end NUMINAMATH_GPT_max_banner_area_l1488_148890


namespace NUMINAMATH_GPT_lcm_of_12_15_18_is_180_l1488_148887

theorem lcm_of_12_15_18_is_180 :
  Nat.lcm 12 (Nat.lcm 15 18) = 180 := by
  sorry

end NUMINAMATH_GPT_lcm_of_12_15_18_is_180_l1488_148887


namespace NUMINAMATH_GPT_triangle_base_length_l1488_148865

theorem triangle_base_length (A h b : ℝ) 
  (h1 : A = 30) 
  (h2 : h = 5) 
  (h3 : A = (b * h) / 2) : 
  b = 12 :=
by
  sorry

end NUMINAMATH_GPT_triangle_base_length_l1488_148865


namespace NUMINAMATH_GPT_brad_books_this_month_l1488_148844

-- Define the number of books William read last month
def william_books_last_month : ℕ := 6

-- Define the number of books Brad read last month
def brad_books_last_month : ℕ := 3 * william_books_last_month

-- Define the number of books Brad read this month as a variable
variable (B : ℕ)

-- Define the total number of books William read over the two months
def total_william_books (B : ℕ) : ℕ := william_books_last_month + 2 * B

-- Define the total number of books Brad read over the two months
def total_brad_books (B : ℕ) : ℕ := brad_books_last_month + B

-- State the condition that William read 4 more books than Brad
def william_read_more_books_condition (B : ℕ) : Prop := total_william_books B = total_brad_books B + 4

-- State the theorem to be proven
theorem brad_books_this_month (B : ℕ) : william_read_more_books_condition B → B = 16 :=
by
  sorry

end NUMINAMATH_GPT_brad_books_this_month_l1488_148844


namespace NUMINAMATH_GPT_fraction_comparison_l1488_148873

theorem fraction_comparison (a b : ℝ) (h : a > b ∧ b > 0) : 
  (a / b) > (a + 1) / (b + 1) :=
by
  sorry

end NUMINAMATH_GPT_fraction_comparison_l1488_148873


namespace NUMINAMATH_GPT_marie_erasers_l1488_148856

-- Define the initial conditions
def initial_erasers : ℝ := 95.0
def additional_erasers : ℝ := 42.0

-- Define the target final erasers count
def final_erasers : ℝ := 137.0

-- The theorem we need to prove
theorem marie_erasers :
  initial_erasers + additional_erasers = final_erasers := by
  sorry

end NUMINAMATH_GPT_marie_erasers_l1488_148856


namespace NUMINAMATH_GPT_subtract_fifteen_result_l1488_148849

theorem subtract_fifteen_result (x : ℕ) (h : x / 10 = 6) : x - 15 = 45 :=
by
  sorry

end NUMINAMATH_GPT_subtract_fifteen_result_l1488_148849


namespace NUMINAMATH_GPT_roots_greater_than_one_implies_s_greater_than_zero_l1488_148838

theorem roots_greater_than_one_implies_s_greater_than_zero
  (b c : ℝ)
  (h : ∃ α β : ℝ, α > 0 ∧ β > 0 ∧ (1 + α) + (1 + β) = -b ∧ (1 + α) * (1 + β) = c) :
  b + c + 1 > 0 :=
sorry

end NUMINAMATH_GPT_roots_greater_than_one_implies_s_greater_than_zero_l1488_148838


namespace NUMINAMATH_GPT_question1_l1488_148808

def sequence1 (a : ℕ → ℕ) : Prop :=
   a 1 = 1 ∧ ∀ n, n ≥ 2 → a n = 3 * a (n - 1) + 1

noncomputable def a_n1 (n : ℕ) : ℕ := (3^n - 1) / 2

theorem question1 (a : ℕ → ℕ) (n : ℕ) : sequence1 a → a n = a_n1 n :=
by
  sorry

end NUMINAMATH_GPT_question1_l1488_148808


namespace NUMINAMATH_GPT_machines_working_together_l1488_148821

theorem machines_working_together (x : ℝ) :
  let R_time := x + 4
  let Q_time := x + 9
  let P_time := x + 12
  (1 / P_time + 1 / Q_time + 1 / R_time) = 1 / x ↔ x = 1 := 
by
  sorry

end NUMINAMATH_GPT_machines_working_together_l1488_148821


namespace NUMINAMATH_GPT_product_of_intersection_points_l1488_148803

-- Define the two circles in the plane
def circle1 (x y : ℝ) : Prop := x^2 - 4*x + y^2 - 8*y + 16 = 0
def circle2 (x y : ℝ) : Prop := x^2 - 6*x + y^2 - 8*y + 21 = 0

-- Define the intersection points property
def are_intersection_points (x y : ℝ) : Prop := circle1 x y ∧ circle2 x y

-- The theorem to be proved
theorem product_of_intersection_points : ∃ x y : ℝ, are_intersection_points x y ∧ x * y = 12 := 
by
  sorry

end NUMINAMATH_GPT_product_of_intersection_points_l1488_148803


namespace NUMINAMATH_GPT_minimum_value_of_y_l1488_148834

theorem minimum_value_of_y (x : ℝ) (h : x > 2) : 
  ∃ y, y = x + 4 / (x - 2) ∧ y = 6 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_y_l1488_148834


namespace NUMINAMATH_GPT_find_x_l1488_148895

theorem find_x (x y : ℝ) (h1 : x / y = 12 / 5) (h2 : y = 25) : x = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1488_148895


namespace NUMINAMATH_GPT_fraction_habitable_surface_l1488_148846

def fraction_exposed_land : ℚ := 3 / 8
def fraction_inhabitable_land : ℚ := 2 / 3

theorem fraction_habitable_surface :
  fraction_exposed_land * fraction_inhabitable_land = 1 / 4 := by
    -- proof steps omitted
    sorry

end NUMINAMATH_GPT_fraction_habitable_surface_l1488_148846


namespace NUMINAMATH_GPT_circle_a_center_radius_circle_b_center_radius_circle_c_center_radius_l1488_148837

-- Part (a): Prove the center and radius for the given circle equation: (x-3)^2 + (y+2)^2 = 16
theorem circle_a_center_radius :
  (∃ (a b : ℤ) (R : ℕ), (∀ (x y : ℝ), (x - 3) ^ 2 + (y + 2) ^ 2 = 16 ↔ (x - a) ^ 2 + (y - b) ^ 2 = R^2) ∧ a = 3 ∧ b = -2 ∧ R = 4) :=
by {
  sorry
}

-- Part (b): Prove the center and radius for the given circle equation: x^2 + y^2 - 2(x - 3y) - 15 = 0
theorem circle_b_center_radius :
  (∃ (a b : ℤ) (R : ℕ), (∀ (x y : ℝ), x^2 + y^2 - 2 * (x - 3 * y) - 15 = 0 ↔ (x - a) ^ 2 + (y - b) ^ 2 = R^2) ∧ a = 1 ∧ b = -3 ∧ R = 5) :=
by {
  sorry
}

-- Part (c): Prove the center and radius for the given circle equation: x^2 + y^2 = x + y + 1/2
theorem circle_c_center_radius :
  (∃ (a b : ℚ) (R : ℚ), (∀ (x y : ℚ), x^2 + y^2 = x + y + 1/2 ↔ (x - a) ^ 2 + (y - b) ^ 2 = R^2) ∧ a = 1/2 ∧ b = 1/2 ∧ R = 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_circle_a_center_radius_circle_b_center_radius_circle_c_center_radius_l1488_148837


namespace NUMINAMATH_GPT_second_pipe_fill_time_l1488_148806

theorem second_pipe_fill_time (x : ℝ) :
  let rate1 := 1 / 8
  let rate2 := 1 / x
  let combined_rate := 1 / 4.8
  rate1 + rate2 = combined_rate → x = 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_second_pipe_fill_time_l1488_148806


namespace NUMINAMATH_GPT_concentration_of_acid_in_third_flask_is_correct_l1488_148892

noncomputable def concentration_of_acid_in_third_flask
  (acid_flask1 : ℕ) (acid_flask2 : ℕ) (acid_flask3 : ℕ) 
  (water_first_to_first_flask : ℕ) (water_second_to_second_flask : Rat) :
  Rat :=
  let total_water := water_first_to_first_flask + water_second_to_second_flask
  let concentration := (acid_flask3 : Rat) / (acid_flask3 + total_water) * 100
  concentration

theorem concentration_of_acid_in_third_flask_is_correct :
  concentration_of_acid_in_third_flask 10 20 30 190 (460/7) = 10.5 :=
  sorry

end NUMINAMATH_GPT_concentration_of_acid_in_third_flask_is_correct_l1488_148892


namespace NUMINAMATH_GPT_JameMade112kProfit_l1488_148897

def JameProfitProblem : Prop :=
  let initial_purchase_cost := 40000
  let feeding_cost_rate := 0.2
  let num_cattle := 100
  let weight_per_cattle := 1000
  let sell_price_per_pound := 2
  let additional_feeding_cost := initial_purchase_cost * feeding_cost_rate
  let total_feeding_cost := initial_purchase_cost + additional_feeding_cost
  let total_purchase_and_feeding_cost := initial_purchase_cost + total_feeding_cost
  let total_revenue := num_cattle * weight_per_cattle * sell_price_per_pound
  let profit := total_revenue - total_purchase_and_feeding_cost
  profit = 112000

theorem JameMade112kProfit :
  JameProfitProblem :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_JameMade112kProfit_l1488_148897


namespace NUMINAMATH_GPT_shirt_original_price_l1488_148819

theorem shirt_original_price {P : ℝ} :
  (P * 0.80045740423098913 * 0.8745 = 105) → P = 150 :=
by sorry

end NUMINAMATH_GPT_shirt_original_price_l1488_148819


namespace NUMINAMATH_GPT_positive_number_satisfying_condition_l1488_148802

theorem positive_number_satisfying_condition :
  ∃ x : ℝ, x > 0 ∧ x^2 = 64 ∧ x = 8 := by sorry

end NUMINAMATH_GPT_positive_number_satisfying_condition_l1488_148802


namespace NUMINAMATH_GPT_value_of_a_minus_2_b_minus_2_l1488_148881

theorem value_of_a_minus_2_b_minus_2 :
  ∀ (a b : ℝ), (a + b = -4/3 ∧ a * b = -7/3) → ((a - 2) * (b - 2) = 0) := by
  sorry

end NUMINAMATH_GPT_value_of_a_minus_2_b_minus_2_l1488_148881


namespace NUMINAMATH_GPT_mod_equiv_1_l1488_148848

theorem mod_equiv_1 : (179 * 933 / 7) % 50 = 1 := by
  sorry

end NUMINAMATH_GPT_mod_equiv_1_l1488_148848


namespace NUMINAMATH_GPT_problem_1_problem_2_l1488_148814

theorem problem_1 (P_A P_B P_notA P_notB : ℚ) (hA: P_A = 1/2) (hB: P_B = 2/5) (hNotA: P_notA = 1/2) (hNotB: P_notB = 3/5) : 
  P_A * P_notB + P_B * P_notA = 1/2 := 
by 
  rw [hA, hB, hNotA, hNotB]
  -- exact calculations here
  sorry

theorem problem_2 (P_A P_B : ℚ) (hA: P_A = 1/2) (hB: P_B = 2/5) :
  (1 - (P_A * P_A * (1 - P_B) * (1 - P_B))) = 91/100 := 
by 
  rw [hA, hB]
  -- exact calculations here
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1488_148814


namespace NUMINAMATH_GPT_problem1_problem2_l1488_148824

-- Proof Problem 1
theorem problem1 (x : ℝ) : -x^2 + 4 * x + 5 < 0 ↔ x < -1 ∨ x > 5 :=
by sorry

-- Proof Problem 2
theorem problem2 (x a : ℝ) :
  if a = -1 then (x^2 + (1 - a) * x - a < 0 ↔ false) else
  if a > -1 then (x^2 + (1 - a) * x - a < 0 ↔ -1 < x ∧ x < a) else
  (x^2 + (1 - a) * x - a < 0 ↔ a < x ∧ x < -1) :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1488_148824


namespace NUMINAMATH_GPT_range_of_a_l1488_148827

theorem range_of_a (a : ℝ) 
  (h : ∀ (f : ℝ → ℝ), 
    (∀ x ≤ a, f x = -x^2 - 2*x) ∧ 
    (∀ x > a, f x = -x) ∧ 
    ¬ ∃ M, ∀ x, f x ≤ M) : 
  a < -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1488_148827


namespace NUMINAMATH_GPT_find_speed_of_P_l1488_148899

noncomputable def walking_speeds (v_P v_Q : ℝ) : Prop :=
  let distance_XY := 90
  let distance_meet_from_Y := 15
  let distance_P := distance_XY - distance_meet_from_Y
  let distance_Q := distance_XY + distance_meet_from_Y
  (v_Q = v_P + 3) ∧
  (distance_P / v_P = distance_Q / v_Q)

theorem find_speed_of_P : ∃ v_P : ℝ, walking_speeds v_P (v_P + 3) ∧ v_P = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_find_speed_of_P_l1488_148899


namespace NUMINAMATH_GPT_sum_of_integers_l1488_148833

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 14) (h3 : x * y = 180) :
  x + y = 2 * Int.sqrt 229 :=
sorry

end NUMINAMATH_GPT_sum_of_integers_l1488_148833


namespace NUMINAMATH_GPT_frederick_final_amount_l1488_148898

-- Definitions of conditions
def P : ℝ := 2000
def r : ℝ := 0.05
def n : ℕ := 18

-- Define the compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

-- Theorem stating the question's answer
theorem frederick_final_amount : compound_interest P r n = 4813.24 :=
by
  sorry

end NUMINAMATH_GPT_frederick_final_amount_l1488_148898


namespace NUMINAMATH_GPT_recurring_decimal_to_fraction_l1488_148885

theorem recurring_decimal_to_fraction : (56 : ℚ) / 99 = 0.56 :=
by
  -- Problem statement and conditions are set, proof needs to be filled in
  sorry

end NUMINAMATH_GPT_recurring_decimal_to_fraction_l1488_148885


namespace NUMINAMATH_GPT_ratio_of_james_to_jacob_l1488_148893

noncomputable def MarkJumpHeight : ℕ := 6
noncomputable def LisaJumpHeight : ℕ := 2 * MarkJumpHeight
noncomputable def JacobJumpHeight : ℕ := 2 * LisaJumpHeight
noncomputable def JamesJumpHeight : ℕ := 16

theorem ratio_of_james_to_jacob : (JamesJumpHeight : ℚ) / (JacobJumpHeight : ℚ) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_james_to_jacob_l1488_148893


namespace NUMINAMATH_GPT_evaluate_f_l1488_148845

def f (n : ℕ) : ℕ :=
  if n < 4 then n^2 - 1 else 3*n - 2

theorem evaluate_f (h : f (f (f 2)) = 22) : f (f (f 2)) = 22 :=
by
  -- we state the final result directly
  sorry

end NUMINAMATH_GPT_evaluate_f_l1488_148845


namespace NUMINAMATH_GPT_convert_speed_l1488_148805

theorem convert_speed (v_kmph : ℝ) (conversion_factor : ℝ) : 
  v_kmph = 252 → conversion_factor = 0.277778 → v_kmph * conversion_factor = 70 := by
  intros h1 h2
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_convert_speed_l1488_148805


namespace NUMINAMATH_GPT_apples_taken_from_each_basket_l1488_148872

theorem apples_taken_from_each_basket (total_apples : ℕ) (baskets : ℕ) (remaining_apples_per_basket : ℕ) 
(h1 : total_apples = 64) (h2 : baskets = 4) (h3 : remaining_apples_per_basket = 13) : 
(total_apples - (remaining_apples_per_basket * baskets)) / baskets = 3 :=
sorry

end NUMINAMATH_GPT_apples_taken_from_each_basket_l1488_148872


namespace NUMINAMATH_GPT_percent_increase_correct_l1488_148854

variable (p_initial p_final : ℝ)

theorem percent_increase_correct : p_initial = 25 → p_final = 28 → (p_final - p_initial) / p_initial * 100 = 12 := by
  intros h_initial h_final
  sorry

end NUMINAMATH_GPT_percent_increase_correct_l1488_148854


namespace NUMINAMATH_GPT_twenty_mul_b_sub_a_not_integer_l1488_148874

theorem twenty_mul_b_sub_a_not_integer {a b : ℝ} (hneq : a ≠ b) (hno_roots : ∀ x : ℝ,
  (x^2 + 20 * a * x + 10 * b) * (x^2 + 20 * b * x + 10 * a) ≠ 0) :
  ¬ ∃ n : ℤ, 20 * (b - a) = n :=
sorry

end NUMINAMATH_GPT_twenty_mul_b_sub_a_not_integer_l1488_148874


namespace NUMINAMATH_GPT_find_number_l1488_148852

-- Definitions based on conditions
def sum : ℕ := 555 + 445
def difference : ℕ := 555 - 445
def quotient : ℕ := 2 * difference
def remainder : ℕ := 70
def divisor : ℕ := sum

-- Statement to be proved
theorem find_number : (divisor * quotient + remainder) = 220070 := by
  sorry

end NUMINAMATH_GPT_find_number_l1488_148852


namespace NUMINAMATH_GPT_johns_last_month_savings_l1488_148875

theorem johns_last_month_savings (earnings rent dishwasher left_over : ℝ) 
  (h1 : rent = 0.40 * earnings) 
  (h2 : dishwasher = 0.70 * rent) 
  (h3 : left_over = earnings - rent - dishwasher) :
  left_over = 0.32 * earnings :=
by 
  sorry

end NUMINAMATH_GPT_johns_last_month_savings_l1488_148875


namespace NUMINAMATH_GPT_number_of_nonsimilar_triangles_l1488_148888
-- Import the necessary library

-- Define the problem conditions
def angles_in_arithmetic_progression (a d : ℕ) : Prop :=
  0 < d ∧ d < 30 ∧ 
  (a - d > 0) ∧ (a + d < 180) ∧ -- Ensures positive and valid angles
  (a - d) + a + (a + d) = 180  -- Triangle sum property

-- Declare the theorem
theorem number_of_nonsimilar_triangles : 
  ∃ n : ℕ, n = 29 ∧ ∀ (a d : ℕ), angles_in_arithmetic_progression a d → d < 30 → a = 60 :=
sorry

end NUMINAMATH_GPT_number_of_nonsimilar_triangles_l1488_148888


namespace NUMINAMATH_GPT_inequality_one_inequality_two_l1488_148882

variable (a b c : ℝ)

-- Conditions given in the problem
axiom positive_a : 0 < a
axiom positive_b : 0 < b
axiom positive_c : 0 < c
axiom sum_eq_one : a + b + c = 1

-- Statements to prove
theorem inequality_one : ab + bc + ac ≤ 1 / 3 :=
sorry

theorem inequality_two : a^2 / b + b^2 / c + c^2 / a ≥ 1 :=
sorry

end NUMINAMATH_GPT_inequality_one_inequality_two_l1488_148882


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l1488_148864

variable (x : ℝ) -- speed of the boat in still water in km/hr
variable (current_rate : ℝ := 4) -- rate of the current in km/hr
variable (downstream_distance : ℝ := 4.8) -- distance traveled downstream in km
variable (downstream_time : ℝ := 18 / 60) -- time traveled downstream in hours

-- The main theorem stating that the speed of the boat in still water is 12 km/hr
theorem boat_speed_in_still_water : x = 12 :=
by
  -- Express the downstream speed and time relation
  have downstream_speed := x + current_rate
  have distance_relation := downstream_distance = downstream_speed * downstream_time
  -- Simplify and solve for x
  simp at distance_relation
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l1488_148864


namespace NUMINAMATH_GPT_product_of_a_and_b_l1488_148804

variable (a b : ℕ)

-- Conditions
def LCM(a b : ℕ) : ℕ := Nat.lcm a b
def HCF(a b : ℕ) : ℕ := Nat.gcd a b

-- Assertion: product of a and b
theorem product_of_a_and_b (h_lcm: LCM a b = 72) (h_hcf: HCF a b = 6) : a * b = 432 := by
  sorry

end NUMINAMATH_GPT_product_of_a_and_b_l1488_148804


namespace NUMINAMATH_GPT_initial_volume_salt_solution_l1488_148883

theorem initial_volume_salt_solution (V : ℝ) (V1 : ℝ) (V2 : ℝ) : 
  V1 = 0.20 * V → 
  V2 = 30 →
  V1 = 0.15 * (V + V2) →
  V = 90 := 
by 
  sorry

end NUMINAMATH_GPT_initial_volume_salt_solution_l1488_148883


namespace NUMINAMATH_GPT_find_f11_l1488_148836

-- Define the odd function properties
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define the functional equation property
def functional_eqn (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = -f x

-- Define the specific values of the function on (0,2)
def specific_values (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x^2

-- The main theorem that needs to be proved
theorem find_f11 (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : functional_eqn f) (h3 : specific_values f) : 
  f 11 = -2 :=
sorry

end NUMINAMATH_GPT_find_f11_l1488_148836


namespace NUMINAMATH_GPT_sequence_sum_problem_l1488_148842

theorem sequence_sum_problem :
  let seq := [72, 76, 80, 84, 88, 92, 96, 100, 104, 108]
  3 * (seq.sum) = 2700 :=
by
  sorry

end NUMINAMATH_GPT_sequence_sum_problem_l1488_148842


namespace NUMINAMATH_GPT_range_of_k_l1488_148879

theorem range_of_k (k : ℝ) (h : k ≠ 0) : (k^2 - 6 * k + 8 ≥ 0) ↔ (k ≥ 4 ∨ k ≤ 2) := 
by sorry

end NUMINAMATH_GPT_range_of_k_l1488_148879


namespace NUMINAMATH_GPT_Harriet_sibling_product_l1488_148861

-- Definition of the family structure
def Harry : Prop := 
  let sisters := 4
  let brothers := 4
  true

-- Harriet being one of Harry's sisters and calculating her siblings
def Harriet : Prop :=
  let S := 4 - 1 -- Number of Harriet's sisters
  let B := 4 -- Number of Harriet's brothers
  S * B = 12

theorem Harriet_sibling_product : Harry → Harriet := by
  intro h
  let S := 3
  let B := 4
  have : S * B = 12 := by norm_num
  exact this

end NUMINAMATH_GPT_Harriet_sibling_product_l1488_148861


namespace NUMINAMATH_GPT_ellipse_equation_y_intercept_range_l1488_148868

noncomputable def a := 2 * Real.sqrt 2
noncomputable def b := Real.sqrt 2
noncomputable def e := Real.sqrt 3 / 2
noncomputable def c := Real.sqrt 6
def M : ℝ × ℝ := (2, 1)

-- Condition: The ellipse equation form
def ellipse (x y : ℝ) : Prop := (x^2) / (a^2) + (y^2) / (b^2) = 1

-- Question 1: Proof that the ellipse equation is as given
theorem ellipse_equation :
  ellipse x y ↔ (x^2) / 8 + (y^2) / 2 = 1 := sorry

-- Condition: Line l is parallel to OM
def slope_OM := 1 / 2
def line_l (m x y : ℝ) : Prop := y = slope_OM * x + m

-- Question 2: Proof of the range for y-intercept m given the conditions
theorem y_intercept_range (m : ℝ) :
  (-Real.sqrt 2 < m ∧ m < 0 ∨ 0 < m ∧ m < Real.sqrt 2) ↔
  ∃ x1 y1 x2 y2,
    line_l m x1 y1 ∧ 
    line_l m x2 y2 ∧ 
    x1 ≠ x2 ∧ 
    y1 ≠ y2 ∧
    x1 * x2 + y1 * y2 < 0 := sorry

end NUMINAMATH_GPT_ellipse_equation_y_intercept_range_l1488_148868


namespace NUMINAMATH_GPT_dave_initial_boxes_l1488_148862

def pieces_per_box : ℕ := 3
def boxes_given_away : ℕ := 5
def pieces_left : ℕ := 21
def total_pieces_given_away := boxes_given_away * pieces_per_box
def total_pieces_initially := total_pieces_given_away + pieces_left

theorem dave_initial_boxes : total_pieces_initially / pieces_per_box = 12 := by
  sorry

end NUMINAMATH_GPT_dave_initial_boxes_l1488_148862


namespace NUMINAMATH_GPT_ab_bc_ca_negative_l1488_148857

theorem ab_bc_ca_negative (a b c : ℝ) (h₁ : a + b + c = 0) (h₂ : abc > 0) : ab + bc + ca < 0 :=
sorry

end NUMINAMATH_GPT_ab_bc_ca_negative_l1488_148857


namespace NUMINAMATH_GPT_normal_intersects_at_l1488_148851

def parabola (x : ℝ) : ℝ := x^2

def slope_of_tangent (x : ℝ) : ℝ := 2 * x

-- C = (2, 4) is a point on the parabola
def C : ℝ × ℝ := (2, parabola 2)

-- Normal to the parabola at C intersects again at point D
-- Prove that D = (-9/4, 81/16)
theorem normal_intersects_at (D : ℝ × ℝ) :
  D = (-9/4, 81/16) :=
sorry

end NUMINAMATH_GPT_normal_intersects_at_l1488_148851


namespace NUMINAMATH_GPT_proof_problem_l1488_148860

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 4)

theorem proof_problem :
  (∃ A ω φ, (A = 2) ∧ (ω = 2) ∧ (φ = Real.pi / 4) ∧
  f (3 * Real.pi / 8) = 0 ∧
  f (Real.pi / 8) = 2 ∧
  (∀ x, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 → f x ≤ 2) ∧
  (∀ x, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 → f x ≥ -Real.sqrt 2) ∧
  f (-Real.pi / 4) = -Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_proof_problem_l1488_148860


namespace NUMINAMATH_GPT_electricity_cost_per_kWh_is_14_cents_l1488_148884

-- Define the conditions
def powerUsagePerHour : ℕ := 125 -- watts
def dailyUsageHours : ℕ := 4 -- hours
def weeklyCostInCents : ℕ := 49 -- cents
def daysInWeek : ℕ := 7 -- days
def wattsToKilowattsFactor : ℕ := 1000 -- conversion factor

-- Define a function to calculate the cost per kWh
def costPerKwh (powerUsagePerHour : ℕ) (dailyUsageHours : ℕ) (weeklyCostInCents : ℕ) (daysInWeek : ℕ) (wattsToKilowattsFactor : ℕ) : ℕ :=
  let dailyConsumption := powerUsagePerHour * dailyUsageHours
  let weeklyConsumption := dailyConsumption * daysInWeek
  let weeklyConsumptionInKwh := weeklyConsumption / wattsToKilowattsFactor
  weeklyCostInCents / weeklyConsumptionInKwh

-- State the theorem
theorem electricity_cost_per_kWh_is_14_cents :
  costPerKwh powerUsagePerHour dailyUsageHours weeklyCostInCents daysInWeek wattsToKilowattsFactor = 14 :=
by
  sorry

end NUMINAMATH_GPT_electricity_cost_per_kWh_is_14_cents_l1488_148884
