import Mathlib

namespace NUMINAMATH_GPT_pond_ratios_l1965_196521

theorem pond_ratios (T A : ℕ) (h1 : T = 48) (h2 : A = 32) : A / (T - A) = 2 :=
by
  sorry

end NUMINAMATH_GPT_pond_ratios_l1965_196521


namespace NUMINAMATH_GPT_inequality_solution_set_l1965_196539

theorem inequality_solution_set (a c x : ℝ) 
  (h1 : -1/3 < x ∧ x < 1/2 → 0 < a * x^2 + 2 * x + c) :
  -2 < x ∧ x < 3 ↔ -c * x^2 + 2 * x - a > 0 :=
by sorry

end NUMINAMATH_GPT_inequality_solution_set_l1965_196539


namespace NUMINAMATH_GPT_comparison_of_a_b_c_l1965_196536

theorem comparison_of_a_b_c (a b c : ℝ) (h_a : a = Real.log 2) (h_b : b = 5^(-1/2 : ℝ)) (h_c : c = Real.sin (Real.pi / 6)) : 
  b < c ∧ c < a :=
by
  sorry

end NUMINAMATH_GPT_comparison_of_a_b_c_l1965_196536


namespace NUMINAMATH_GPT_decimal_properties_l1965_196559

theorem decimal_properties :
  (3.00 : ℝ) = (3 : ℝ) :=
by sorry

end NUMINAMATH_GPT_decimal_properties_l1965_196559


namespace NUMINAMATH_GPT_value_of_e_is_91_l1965_196518

noncomputable def value_of_e (a b c d e : ℤ) (k : ℤ) : Prop :=
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧ e % 2 = 1 ∧
  b = a + 2 * k ∧ c = a + 4 * k ∧ d = a + 6 * k ∧ e = a + 8 * k ∧
  a + c = 146 ∧ k > 0 ∧ 2 * k ≥ 4 ∧ k ≠ 2

theorem value_of_e_is_91 (a b c d e k : ℤ) (h : value_of_e a b c d e k) : e = 91 :=
  sorry

end NUMINAMATH_GPT_value_of_e_is_91_l1965_196518


namespace NUMINAMATH_GPT_square_area_l1965_196552

noncomputable def line_lies_on_square_side (a b : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), A = (a, a + 4) ∧ B = (b, b + 4)

noncomputable def points_on_parabola (x y : ℝ) : Prop :=
  ∃ (C D : ℝ × ℝ), C = (y^2, y) ∧ D = (x^2, x)

theorem square_area (a b : ℝ) (x y : ℝ)
  (h1 : line_lies_on_square_side a b)
  (h2 : points_on_parabola x y) :
  ∃ (s : ℝ), s^2 = (boxed_solution) :=
sorry

end NUMINAMATH_GPT_square_area_l1965_196552


namespace NUMINAMATH_GPT_find_x_perpendicular_l1965_196570

-- Definitions used in the conditions
def a : ℝ × ℝ := (3, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

-- Condition: vectors a and b are perpendicular
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

-- The theorem we want to prove
theorem find_x_perpendicular : ∀ x : ℝ, perpendicular a (b x) → x = -8 / 3 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_find_x_perpendicular_l1965_196570


namespace NUMINAMATH_GPT_derivative_of_y_is_correct_l1965_196591

noncomputable def y (x : ℝ) := x^2 * Real.sin x

theorem derivative_of_y_is_correct : (deriv y x = 2 * x * Real.sin x + x^2 * Real.cos x) :=
by
  sorry

end NUMINAMATH_GPT_derivative_of_y_is_correct_l1965_196591


namespace NUMINAMATH_GPT_inequality_abc_l1965_196557

theorem inequality_abc (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) :
  (a + b) * (b + c) * (c + a) ≥ 4 * (a + b + c - 1) :=
sorry

end NUMINAMATH_GPT_inequality_abc_l1965_196557


namespace NUMINAMATH_GPT_blocks_differs_in_exactly_two_ways_correct_l1965_196589

structure Block where
  material : Bool       -- material: false for plastic, true for wood
  size : Fin 3          -- sizes: 0 for small, 1 for medium, 2 for large
  color : Fin 4         -- colors: 0 for blue, 1 for green, 2 for red, 3 for yellow
  shape : Fin 4         -- shapes: 0 for circle, 1 for hexagon, 2 for square, 3 for triangle
  finish : Bool         -- finish: false for glossy, true for matte

def originalBlock : Block :=
  { material := false, size := 1, color := 2, shape := 0, finish := false }

def differsInExactlyTwoWays (b1 b2 : Block) : Bool :=
  (if b1.material ≠ b2.material then 1 else 0) +
  (if b1.size ≠ b2.size then 1 else 0) +
  (if b1.color ≠ b2.color then 1 else 0) +
  (if b1.shape ≠ b2.shape then 1 else 0) +
  (if b1.finish ≠ b2.finish then 1 else 0) == 2

def countBlocksDifferingInTwoWays : Nat :=
  let allBlocks := List.product
                  (List.product
                    (List.product
                      (List.product
                        [false, true]
                        ([0, 1, 2] : List (Fin 3)))
                      ([0, 1, 2, 3] : List (Fin 4)))
                    ([0, 1, 2, 3] : List (Fin 4)))
                  [false, true]
  (allBlocks.filter
    (λ b => differsInExactlyTwoWays originalBlock
                { material := b.1.1.1.1, size := b.1.1.1.2, color := b.1.1.2, shape := b.1.2, finish := b.2 })).length

theorem blocks_differs_in_exactly_two_ways_correct :
  countBlocksDifferingInTwoWays = 51 :=
  by
    sorry

end NUMINAMATH_GPT_blocks_differs_in_exactly_two_ways_correct_l1965_196589


namespace NUMINAMATH_GPT_Z_4_3_eq_neg11_l1965_196526

def Z (a b : ℤ) : ℤ := a^2 - 3 * a * b + b^2

theorem Z_4_3_eq_neg11 : Z 4 3 = -11 := 
by
  sorry

end NUMINAMATH_GPT_Z_4_3_eq_neg11_l1965_196526


namespace NUMINAMATH_GPT_arithmetic_sequence_common_diff_l1965_196515

noncomputable def variance (s : List ℝ) : ℝ :=
  let mean := (s.sum) / (s.length : ℝ)
  (s.map (λ x => (x - mean) ^ 2)).sum / (s.length : ℝ)

theorem arithmetic_sequence_common_diff (a1 a2 a3 a4 a5 a6 a7 d : ℝ) 
(h_seq : a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d ∧ a5 = a1 + 4 * d ∧ a6 = a1 + 5 * d ∧ a7 = a1 + 6 * d)
(h_var : variance [a1, a2, a3, a4, a5, a6, a7] = 1) : 
d = 1 / 2 ∨ d = -1 / 2 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_diff_l1965_196515


namespace NUMINAMATH_GPT_fence_width_l1965_196592

theorem fence_width (L W : ℝ) 
  (circumference_eq : 2 * (L + W) = 30)
  (width_eq : W = 2 * L) : 
  W = 10 :=
by 
  sorry

end NUMINAMATH_GPT_fence_width_l1965_196592


namespace NUMINAMATH_GPT_BD_distance_16_l1965_196513

noncomputable def distanceBD (DA AB : ℝ) (angleBDA : ℝ) : ℝ :=
  (DA^2 + AB^2 - 2 * DA * AB * Real.cos angleBDA).sqrt

theorem BD_distance_16 :
  distanceBD 10 14 (60 * Real.pi / 180) = 16 := by
  sorry

end NUMINAMATH_GPT_BD_distance_16_l1965_196513


namespace NUMINAMATH_GPT_roots_of_cubic_l1965_196527

theorem roots_of_cubic (a b c d r s t : ℝ) 
  (h1 : r + s + t = -b / a)
  (h2 : r * s + r * t + s * t = c / a)
  (h3 : r * s * t = -d / a) :
  1 / (r ^ 2) + 1 / (s ^ 2) + 1 / (t ^ 2) = (c ^ 2 - 2 * b * d) / (d ^ 2) := 
sorry

end NUMINAMATH_GPT_roots_of_cubic_l1965_196527


namespace NUMINAMATH_GPT_find_QS_l1965_196588

theorem find_QS (cosR : ℝ) (RS QR QS : ℝ) (h1 : cosR = 3 / 5) (h2 : RS = 10) (h3 : cosR = QR / RS) (h4: QR ^ 2 + QS ^ 2 = RS ^ 2) : QS = 8 :=
by 
  sorry

end NUMINAMATH_GPT_find_QS_l1965_196588


namespace NUMINAMATH_GPT_find_smallest_k_satisfying_cos_square_l1965_196575

theorem find_smallest_k_satisfying_cos_square (k : ℕ) (h : ∃ n : ℕ, k^2 = 180 * n - 64):
  k = 48 ∨ k = 53 :=
by sorry

end NUMINAMATH_GPT_find_smallest_k_satisfying_cos_square_l1965_196575


namespace NUMINAMATH_GPT_product_of_solutions_eq_zero_l1965_196516

theorem product_of_solutions_eq_zero :
  (∀ x : ℝ, (3 * x + 5) / (6 * x + 5) = (5 * x + 4) / (9 * x + 4) → (x = 0 ∨ x = 8 / 3)) →
  0 * (8 / 3) = 0 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_product_of_solutions_eq_zero_l1965_196516


namespace NUMINAMATH_GPT_more_orange_pages_read_l1965_196582

-- Define the conditions
def purple_pages_per_book : Nat := 230
def orange_pages_per_book : Nat := 510
def purple_books_read : Nat := 5
def orange_books_read : Nat := 4

-- Calculate the total pages read from purple and orange books respectively
def total_purple_pages_read : Nat := purple_pages_per_book * purple_books_read
def total_orange_pages_read : Nat := orange_pages_per_book * orange_books_read

-- State the theorem to be proved
theorem more_orange_pages_read : total_orange_pages_read - total_purple_pages_read = 890 :=
by
  -- This is where the proof steps would go, but we'll leave it as sorry to indicate the proof is not provided
  sorry

end NUMINAMATH_GPT_more_orange_pages_read_l1965_196582


namespace NUMINAMATH_GPT_min_value_expression_l1965_196547

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + y) * (1 / x + 4 / y) ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1965_196547


namespace NUMINAMATH_GPT_cylinder_volume_l1965_196560

theorem cylinder_volume (r h : ℝ) (radius_is_2 : r = 2) (height_is_3 : h = 3) :
  π * r^2 * h = 12 * π :=
by
  rw [radius_is_2, height_is_3]
  sorry

end NUMINAMATH_GPT_cylinder_volume_l1965_196560


namespace NUMINAMATH_GPT_average_coins_per_day_l1965_196555

theorem average_coins_per_day :
  let a := 10
  let d := 10
  let n := 7
  let extra := 20
  let total_coins := a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) + (a + 5 * d) + (a + 6 * d + extra)
  total_coins = 300 →
  total_coins / n = 300 / 7 :=
by
  sorry

end NUMINAMATH_GPT_average_coins_per_day_l1965_196555


namespace NUMINAMATH_GPT_apple_trees_count_l1965_196567

-- Conditions
def num_peach_trees : ℕ := 45
def kg_per_peach_tree : ℕ := 65
def total_mass_fruit : ℕ := 7425
def kg_per_apple_tree : ℕ := 150
variable (A : ℕ)

-- Proof goal
theorem apple_trees_count (h : A * kg_per_apple_tree + num_peach_trees * kg_per_peach_tree = total_mass_fruit) : A = 30 := 
sorry

end NUMINAMATH_GPT_apple_trees_count_l1965_196567


namespace NUMINAMATH_GPT_cloves_used_for_roast_chicken_l1965_196549

section
variable (total_cloves : ℕ)
variable (remaining_cloves : ℕ)

theorem cloves_used_for_roast_chicken (h1 : total_cloves = 93) (h2 : remaining_cloves = 7) : total_cloves - remaining_cloves = 86 := 
by 
  have h : total_cloves - remaining_cloves = 93 - 7 := by rw [h1, h2]
  exact h
-- sorry
end

end NUMINAMATH_GPT_cloves_used_for_roast_chicken_l1965_196549


namespace NUMINAMATH_GPT_power_mod_equiv_l1965_196585

-- Define the main theorem
theorem power_mod_equiv {a n k : ℕ} (h₁ : a ≥ 2) (h₂ : n ≥ 1) :
  (a^k ≡ 1 [MOD (a^n - 1)]) ↔ (k % n = 0) :=
by sorry

end NUMINAMATH_GPT_power_mod_equiv_l1965_196585


namespace NUMINAMATH_GPT_sum_of_three_largest_of_consecutive_numbers_l1965_196587

theorem sum_of_three_largest_of_consecutive_numbers (n : ℕ) :
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  -- Given the conditions and expected result, we can break down the proof as follows:
  intros h1
  sorry

end NUMINAMATH_GPT_sum_of_three_largest_of_consecutive_numbers_l1965_196587


namespace NUMINAMATH_GPT_M_empty_iff_k_range_M_interval_iff_k_range_l1965_196531

-- Part 1
theorem M_empty_iff_k_range (k : ℝ) :
  (∀ x : ℝ, (k^2 + 2 * k - 3) * x^2 + (k + 3) * x - 1 ≤ 0) ↔ -3 ≤ k ∧ k ≤ 1 / 5 := sorry

-- Part 2
theorem M_interval_iff_k_range (k a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_ab : a < b) :
  (∀ x : ℝ, (k^2 + 2 * k - 3) * x^2 + (k + 3) * x - 1 > 0 ↔ a < x ∧ x < b) ↔ 1 / 5 < k ∧ k < 1 := sorry

end NUMINAMATH_GPT_M_empty_iff_k_range_M_interval_iff_k_range_l1965_196531


namespace NUMINAMATH_GPT_min_sum_of_dimensions_l1965_196538

theorem min_sum_of_dimensions 
  (a b c : ℕ) 
  (h_pos : a > 0) 
  (h_pos_2 : b > 0) 
  (h_pos_3 : c > 0) 
  (h_even : a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) 
  (h_vol : a * b * c = 1806) 
  : a + b + c = 56 :=
sorry

end NUMINAMATH_GPT_min_sum_of_dimensions_l1965_196538


namespace NUMINAMATH_GPT_pentagon_interior_angles_l1965_196533

theorem pentagon_interior_angles
  (x y : ℝ)
  (H_eq_triangle : ∀ (angle : ℝ), angle = 60)
  (H_rect_QT : ∀ (angle : ℝ), angle = 90)
  (sum_interior_angles_pentagon : ∀ (n : ℕ), (n - 2) * 180 = 3 * 180) :
  x + y = 60 :=
by
  sorry

end NUMINAMATH_GPT_pentagon_interior_angles_l1965_196533


namespace NUMINAMATH_GPT_find_f_zero_l1965_196571

variable (f : ℝ → ℝ)
variable (hf : ∀ x y : ℝ, f (x + y) = f x + f y + 1 / 2)

theorem find_f_zero : f 0 = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_zero_l1965_196571


namespace NUMINAMATH_GPT_solve_system_l1965_196528

theorem solve_system :
  ∃ x y : ℝ, (x^2 - 9 * y^2 = 0 ∧ 2 * x - 3 * y = 6) ∧ (x = 6 ∧ y = 2) ∨ (x = 2 ∧ y = -2 / 3) :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_solve_system_l1965_196528


namespace NUMINAMATH_GPT_range_of_y_given_x_l1965_196542

theorem range_of_y_given_x (x : ℝ) (h₁ : x > 3) : 0 < (6 / x) ∧ (6 / x) < 2 :=
by 
  sorry

end NUMINAMATH_GPT_range_of_y_given_x_l1965_196542


namespace NUMINAMATH_GPT_jake_time_to_row_lake_l1965_196584

noncomputable def time_to_row_lake (side_length miles_per_side : ℝ) (swim_time_per_mile minutes_per_mile : ℝ) : ℝ :=
  let swim_speed := 60 / swim_time_per_mile -- miles per hour
  let row_speed := 2 * swim_speed          -- miles per hour
  let total_distance := 4 * side_length    -- miles
  total_distance / row_speed               -- hours

theorem jake_time_to_row_lake :
  time_to_row_lake 15 20 = 10 := sorry

end NUMINAMATH_GPT_jake_time_to_row_lake_l1965_196584


namespace NUMINAMATH_GPT_minimum_value_of_expression_l1965_196594

theorem minimum_value_of_expression (a b : ℝ) (h1 : a + 2 * b = 1) (h2 : a * b > 0) : 
  (1 / a) + (2 / b) = 5 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l1965_196594


namespace NUMINAMATH_GPT_max_difference_intersection_ycoords_l1965_196558

theorem max_difference_intersection_ycoords :
  let f₁ (x : ℝ) := 5 - 2 * x^2 + x^3
  let f₂ (x : ℝ) := 1 + x^2 + x^3
  let x1 := (2 : ℝ) / Real.sqrt 3
  let x2 := - (2 : ℝ) / Real.sqrt 3
  let y1 := f₁ x1
  let y2 := f₂ x2
  (f₁ = f₂)
  → abs (y1 - y2) = (16 * Real.sqrt 3 / 9) :=
by
  sorry

end NUMINAMATH_GPT_max_difference_intersection_ycoords_l1965_196558


namespace NUMINAMATH_GPT_peter_candles_l1965_196564

theorem peter_candles (candles_rupert : ℕ) (ratio : ℝ) (candles_peter : ℕ) 
  (h1 : ratio = 3.5) (h2 : candles_rupert = 35) (h3 : candles_peter = candles_rupert / ratio) : 
  candles_peter = 10 := 
sorry

end NUMINAMATH_GPT_peter_candles_l1965_196564


namespace NUMINAMATH_GPT_fraction_of_second_year_students_l1965_196598

-- Define the fractions of first-year and second-year students
variables (F S f s: ℝ)

-- Conditions
axiom h1 : F + S = 1
axiom h2 : f = (1 / 5) * F
axiom h3 : s = 4 * f
axiom h4 : S - s = 0.2

-- The theorem statement to prove the fraction of second-year students is 2 / 3
theorem fraction_of_second_year_students (F S f s: ℝ) 
    (h1: F + S = 1) 
    (h2: f = (1 / 5) * F) 
    (h3: s = 4 * f) 
    (h4: S - s = 0.2) : 
    S = 2 / 3 :=
by 
    sorry

end NUMINAMATH_GPT_fraction_of_second_year_students_l1965_196598


namespace NUMINAMATH_GPT_maximum_k_value_l1965_196505

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / (x - 1)
noncomputable def g (x : ℝ) (k : ℕ) : ℝ := k / x

theorem maximum_k_value (c : ℝ) (h_c : c > 1) : 
  (∃ a b : ℝ, 0 < a ∧ a < b ∧ b < c ∧ f c = f a ∧ f a = g b 3) ∧ 
  (∀ k : ℕ, k > 3 → ¬ ∃ a b : ℝ, 0 < a ∧ a < b ∧ b < c ∧ f c = f a ∧ f a = g b k) :=
sorry

end NUMINAMATH_GPT_maximum_k_value_l1965_196505


namespace NUMINAMATH_GPT_greatest_integer_radius_l1965_196522

theorem greatest_integer_radius (r : ℕ) :
  (π * (r: ℝ)^2 < 30 * π) ∧ (2 * π * (r: ℝ) > 10 * π) → r = 5 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_radius_l1965_196522


namespace NUMINAMATH_GPT_sphere_radius_l1965_196597

/-- Given the curved surface area (CSA) of a sphere and its formula, 
    prove that the radius of the sphere is 4 cm.
    Conditions:
    - CSA = 4πr²
    - Curved surface area is 64π cm²
-/
theorem sphere_radius (r : ℝ) (h : 4 * Real.pi * r^2 = 64 * Real.pi) : r = 4 := by
  sorry

end NUMINAMATH_GPT_sphere_radius_l1965_196597


namespace NUMINAMATH_GPT_minimum_value_of_expression_l1965_196502

theorem minimum_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : (x / y + y / z + z / x) + (y / x + z / y + x / z) = 10) :
  ∃ P, (P = (x / y + y / z + z / x) * (y / x + z / y + x / z)) ∧ P = 25 := 
by sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l1965_196502


namespace NUMINAMATH_GPT_calc_mixed_number_expr_l1965_196519

theorem calc_mixed_number_expr :
  53 * (3 + 1 / 4 - (3 + 3 / 4)) / (1 + 2 / 3 + (2 + 2 / 5)) = -6 - 57 / 122 := 
by
  sorry

end NUMINAMATH_GPT_calc_mixed_number_expr_l1965_196519


namespace NUMINAMATH_GPT_solve_equation_l1965_196590

theorem solve_equation : ∃ x : ℝ, (2 * x - 1) / 3 - (x - 2) / 6 = 2 ∧ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1965_196590


namespace NUMINAMATH_GPT_reciprocal_of_2022_l1965_196540

theorem reciprocal_of_2022 : 1 / 2022 = (1 : ℝ) / 2022 :=
sorry

end NUMINAMATH_GPT_reciprocal_of_2022_l1965_196540


namespace NUMINAMATH_GPT_calc_sqrt_expr_l1965_196595

theorem calc_sqrt_expr : (Real.sqrt 2 + 1) ^ 2 - Real.sqrt 18 + 2 * Real.sqrt (1 / 2) = 3 := by
  sorry

end NUMINAMATH_GPT_calc_sqrt_expr_l1965_196595


namespace NUMINAMATH_GPT_perp_lines_solution_l1965_196507

theorem perp_lines_solution (a : ℝ) :
  ((a+2) * (a-1) + (1-a) * (2*a + 3) = 0) → (a = 1 ∨ a = -1) :=
by
  sorry

end NUMINAMATH_GPT_perp_lines_solution_l1965_196507


namespace NUMINAMATH_GPT_area_of_white_square_l1965_196593

theorem area_of_white_square
  (face_area : ℕ)
  (total_surface_area : ℕ)
  (blue_paint_area : ℕ)
  (faces : ℕ)
  (area_of_white_square : ℕ) :
  face_area = 12 * 12 →
  total_surface_area = 6 * face_area →
  blue_paint_area = 432 →
  faces = 6 →
  area_of_white_square = face_area - (blue_paint_area / faces) →
  area_of_white_square = 72 :=
by
  sorry

end NUMINAMATH_GPT_area_of_white_square_l1965_196593


namespace NUMINAMATH_GPT_max_candies_theorem_l1965_196561

-- Defining constants: the number of students and the total number of candies.
def n : ℕ := 40
def T : ℕ := 200

-- Defining the condition that each student takes at least 2 candies.
def min_candies_per_student : ℕ := 2

-- Calculating the minimum total number of candies taken by 39 students.
def min_total_for_39_students := min_candies_per_student * (n - 1)

-- The maximum number of candies one student can take.
def max_candies_one_student_can_take := T - min_total_for_39_students

-- The statement to prove.
theorem max_candies_theorem : max_candies_one_student_can_take = 122 :=
by
  sorry

end NUMINAMATH_GPT_max_candies_theorem_l1965_196561


namespace NUMINAMATH_GPT_grid_with_value_exists_possible_values_smallest_possible_value_l1965_196537

open Nat

def isGridValuesP (P : ℕ) (a b c d e f g h i : ℕ) : Prop :=
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i) ∧
  (b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
   d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
   e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
   f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
   g ≠ h ∧ g ≠ i ∧
   h ≠ i) ∧
  (a * b * c = P) ∧ (d * e * f = P) ∧
  (g * h * i = P) ∧ (a * d * g = P) ∧
  (b * e * h = P) ∧ (c * f * i = P)

theorem grid_with_value_exists (P : ℕ) :
  ∃ a b c d e f g h i : ℕ, isGridValuesP P a b c d e f g h i :=
sorry

theorem possible_values (P : ℕ) :
  P ∈ [1992, 1995] ↔ 
  ∃ a b c d e f g h i : ℕ, isGridValuesP P a b c d e f g h i :=
sorry

theorem smallest_possible_value : 
  ∃ P a b c d e f g h i : ℕ, isGridValuesP P a b c d e f g h i ∧ 
  ∀ Q, (∃ w x y z u v s t q : ℕ, isGridValuesP Q w x y z u v s t q) → Q ≥ 120 :=
sorry

end NUMINAMATH_GPT_grid_with_value_exists_possible_values_smallest_possible_value_l1965_196537


namespace NUMINAMATH_GPT_max_value_of_a2b3c2_l1965_196572

theorem max_value_of_a2b3c2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + b + c = 1) :
  a^2 * b^3 * c^2 ≤ 81 / 262144 :=
sorry

end NUMINAMATH_GPT_max_value_of_a2b3c2_l1965_196572


namespace NUMINAMATH_GPT_determine_vector_p_l1965_196544

structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def vector_operation (m p : Vector2D) : Vector2D :=
  Vector2D.mk (m.x * p.x + m.y * p.y) (m.x * p.y + m.y * p.x)

theorem determine_vector_p (p : Vector2D) : 
  (∀ (m : Vector2D), vector_operation m p = m) → p = Vector2D.mk 1 0 :=
by
  sorry

end NUMINAMATH_GPT_determine_vector_p_l1965_196544


namespace NUMINAMATH_GPT_number_of_customers_l1965_196500

theorem number_of_customers 
    (boxes_opened : ℕ) 
    (samples_per_box : ℕ) 
    (samples_left_over : ℕ) 
    (samples_limit_per_person : ℕ)
    (h1 : boxes_opened = 12)
    (h2 : samples_per_box = 20)
    (h3 : samples_left_over = 5)
    (h4 : samples_limit_per_person = 1) : 
    ∃ customers : ℕ, customers = (boxes_opened * samples_per_box) - samples_left_over ∧ customers = 235 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_customers_l1965_196500


namespace NUMINAMATH_GPT_postage_problem_l1965_196511

noncomputable def sum_all_positive_integers (n1 n2 : ℕ) : ℕ :=
  n1 + n2

theorem postage_problem : sum_all_positive_integers 21 22 = 43 :=
by
  have h1 : ∀ x y z : ℕ, 7 * x + 21 * y + 23 * z ≠ 120 := sorry
  have h2 : ∀ x y z : ℕ, 7 * x + 22 * y + 24 * z ≠ 120 := sorry
  exact rfl

end NUMINAMATH_GPT_postage_problem_l1965_196511


namespace NUMINAMATH_GPT_time_to_run_home_l1965_196514

-- Define the conditions
def blocks_run_per_time : ℚ := 2 -- Justin runs 2 blocks
def time_per_blocks : ℚ := 1.5 -- in 1.5 minutes
def blocks_to_home : ℚ := 8 -- Justin is 8 blocks from home

-- Define the theorem to prove the time taken for Justin to run home
theorem time_to_run_home : (blocks_to_home / blocks_run_per_time) * time_per_blocks = 6 :=
by
  sorry

end NUMINAMATH_GPT_time_to_run_home_l1965_196514


namespace NUMINAMATH_GPT_max_shortest_side_decagon_inscribed_circle_l1965_196524

noncomputable def shortest_side_decagon : ℝ :=
  2 * Real.sin (36 * Real.pi / 180 / 2)

theorem max_shortest_side_decagon_inscribed_circle :
  shortest_side_decagon = (Real.sqrt 5 - 1) / 2 :=
by {
  -- Proof details here
  sorry
}

end NUMINAMATH_GPT_max_shortest_side_decagon_inscribed_circle_l1965_196524


namespace NUMINAMATH_GPT_initial_water_amount_l1965_196501

theorem initial_water_amount (W : ℝ) (h1 : 0.006 * 50 = 0.03 * W) : W = 10 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_initial_water_amount_l1965_196501


namespace NUMINAMATH_GPT_ratio_of_ages_l1965_196599

theorem ratio_of_ages (S F : Nat) 
  (h1 : F = 3 * S) 
  (h2 : (S + 6) + (F + 6) = 156) : 
  (F + 6) / (S + 6) = 19 / 7 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_ages_l1965_196599


namespace NUMINAMATH_GPT_largest_possible_d_l1965_196569

theorem largest_possible_d (a b c d : ℝ) (h1 : a + b + c + d = 10) (h2 : ab + ac + ad + bc + bd + cd = 20) : 
    d ≤ (5 + Real.sqrt 105) / 2 :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_d_l1965_196569


namespace NUMINAMATH_GPT_alt_rep_of_set_l1965_196566

def NatPos (x : ℕ) := x > 0

theorem alt_rep_of_set : {x : ℕ | NatPos x ∧ x - 3 < 2} = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_GPT_alt_rep_of_set_l1965_196566


namespace NUMINAMATH_GPT_jessica_total_spent_l1965_196535

noncomputable def catToyCost : ℝ := 10.22
noncomputable def cageCost : ℝ := 11.73
noncomputable def totalCost : ℝ := 21.95

theorem jessica_total_spent :
  catToyCost + cageCost = totalCost :=
sorry

end NUMINAMATH_GPT_jessica_total_spent_l1965_196535


namespace NUMINAMATH_GPT_total_cost_textbooks_l1965_196512

theorem total_cost_textbooks :
  let sale_books := 5 * 10
  let online_books := 40
  let bookstore_books := 3 * 40
  sale_books + online_books + bookstore_books = 210 :=
by
  let sale_books := 5 * 10
  let online_books := 40
  let bookstore_books := 3 * 40
  show sale_books + online_books + bookstore_books = 210
  sorry

end NUMINAMATH_GPT_total_cost_textbooks_l1965_196512


namespace NUMINAMATH_GPT_polynomial_satisfies_condition_l1965_196520

-- Define P as a real polynomial
def P (a : ℝ) (X : ℝ) : ℝ := a * X

-- Define a statement that needs to be proven
theorem polynomial_satisfies_condition (P : ℝ → ℝ) :
  (∀ X : ℝ, P (2 * X) = 2 * P X) ↔ ∃ a : ℝ, ∀ X : ℝ, P X = a * X :=
by
  sorry

end NUMINAMATH_GPT_polynomial_satisfies_condition_l1965_196520


namespace NUMINAMATH_GPT_point_side_opposite_l1965_196509

def equation_lhs (x y : ℝ) : ℝ := 2 * y - 6 * x + 1

theorem point_side_opposite : 
  (equation_lhs 0 0 * equation_lhs 2 1 < 0) := 
by 
   sorry

end NUMINAMATH_GPT_point_side_opposite_l1965_196509


namespace NUMINAMATH_GPT_inconsistent_intercepts_l1965_196596

-- Define the ellipse equation
def ellipse (x y : ℝ) (m : ℝ) : Prop :=
  x^2 / m + y^2 / 4 = 1

-- Define the line equations
def line1 (x k : ℝ) : ℝ := k * x + 1
def line2 (x : ℝ) (k : ℝ) : ℝ := - k * x - 2

-- Disc calculation for line1
def disc1 (m k : ℝ) : ℝ :=
  let a := 4 + m * k^2
  let b := 2 * m * k
  let c := -3 * m
  b^2 - 4 * a * c

-- Disc calculation for line2
def disc2 (m k : ℝ) : ℝ :=
  let bb := 4 * m * k
  bb^2

-- Statement of the problem
theorem inconsistent_intercepts (m k : ℝ) (hm_pos : 0 < m) :
  disc1 m k ≠ disc2 m k :=
by
  sorry

end NUMINAMATH_GPT_inconsistent_intercepts_l1965_196596


namespace NUMINAMATH_GPT_coords_A_l1965_196586

def A : ℝ × ℝ := (1, -2)

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def move_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 + d)

def A' : ℝ × ℝ := reflect_y_axis A

def A'' : ℝ × ℝ := move_up A' 3

theorem coords_A'' : A'' = (-1, 1) := by
  sorry

end NUMINAMATH_GPT_coords_A_l1965_196586


namespace NUMINAMATH_GPT_first_number_in_set_l1965_196532

theorem first_number_in_set (x : ℝ)
  (h : (x + 40 + 60) / 3 = (10 + 80 + 15) / 3 + 5) :
  x = 20 := by
  sorry

end NUMINAMATH_GPT_first_number_in_set_l1965_196532


namespace NUMINAMATH_GPT_find_number_l1965_196556

theorem find_number (N: ℕ): (N % 131 = 112) ∧ (N % 132 = 98) → 1000 ≤ N ∧ N ≤ 9999 ∧ N = 1946 :=
sorry

end NUMINAMATH_GPT_find_number_l1965_196556


namespace NUMINAMATH_GPT_added_number_is_four_l1965_196506

theorem added_number_is_four :
  ∃ x y, 2 * x < 3 * x ∧ (3 * x - 2 * x = 8) ∧ 
         ((2 * x + y) * 7 = 5 * (3 * x + y)) ∧ y = 4 :=
  sorry

end NUMINAMATH_GPT_added_number_is_four_l1965_196506


namespace NUMINAMATH_GPT_king_arthur_round_table_seats_l1965_196510

theorem king_arthur_round_table_seats (n : ℕ) (h₁ : n > 1) (h₂ : 10 < 29) (h₃ : (29 - 10) * 2 = n - 2) : 
  n = 38 := 
by
  sorry

end NUMINAMATH_GPT_king_arthur_round_table_seats_l1965_196510


namespace NUMINAMATH_GPT_point_reflection_example_l1965_196503

def point := ℝ × ℝ

def reflect_x_axis (p : point) : point := (p.1, -p.2)

theorem point_reflection_example : reflect_x_axis (1, -2) = (1, 2) := sorry

end NUMINAMATH_GPT_point_reflection_example_l1965_196503


namespace NUMINAMATH_GPT_revenue_increase_20_percent_l1965_196504

variable (P Q : ℝ)

def original_revenue (P Q : ℝ) : ℝ := P * Q
def new_price (P : ℝ) : ℝ := P * 1.5
def new_quantity (Q : ℝ) : ℝ := Q * 0.8
def new_revenue (P Q : ℝ) : ℝ := (new_price P) * (new_quantity Q)

theorem revenue_increase_20_percent (P Q : ℝ) : 
  (new_revenue P Q) = 1.2 * (original_revenue P Q) := by
  sorry

end NUMINAMATH_GPT_revenue_increase_20_percent_l1965_196504


namespace NUMINAMATH_GPT_pair_with_15_l1965_196554

theorem pair_with_15 (s : List ℕ) (h : s = [49, 29, 9, 40, 22, 15, 53, 33, 13, 47]) :
  ∃ (t : List (ℕ × ℕ)), (∀ (x y : ℕ), (x, y) ∈ t → x + y = 62) ∧ (15, 47) ∈ t := by
  sorry

end NUMINAMATH_GPT_pair_with_15_l1965_196554


namespace NUMINAMATH_GPT_roots_of_quadratic_eq_l1965_196530

theorem roots_of_quadratic_eq : 
    ∃ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) ∧ (x1 ≠ x2) ∧ (x1 + x2) / (x1 * x2) = -1 / 4 := 
sorry

end NUMINAMATH_GPT_roots_of_quadratic_eq_l1965_196530


namespace NUMINAMATH_GPT_max_lattice_points_in_unit_circle_l1965_196581

-- Define a point with integer coordinates
structure LatticePoint :=
  (x : ℤ)
  (y : ℤ)

-- Define the condition for a lattice point to be strictly inside a given circle
def strictly_inside_circle (p : LatticePoint) (center : Prod ℤ ℤ) (r : ℝ) : Prop :=
  let dx := (p.x - center.fst : ℝ)
  let dy := (p.y - center.snd : ℝ)
  dx^2 + dy^2 < r^2

-- Define the problem statement
theorem max_lattice_points_in_unit_circle : ∀ (center : Prod ℤ ℤ) (r : ℝ),
  r = 1 → 
  ∃ (ps : Finset LatticePoint), 
    (∀ p ∈ ps, strictly_inside_circle p center r) ∧ 
    ps.card = 4 :=
by
  sorry

end NUMINAMATH_GPT_max_lattice_points_in_unit_circle_l1965_196581


namespace NUMINAMATH_GPT_balls_distribution_l1965_196546

def balls_into_boxes : Nat := 6
def boxes : Nat := 3
def at_least_one_in_first (n m : Nat) : ℕ := sorry -- Use a function with appropriate constraints to ensure at least 1 ball is in the first box

theorem balls_distribution (n m : Nat) (h: n = 6) (h2: m = 3) :
  at_least_one_in_first n m = 665 :=
by
  sorry

end NUMINAMATH_GPT_balls_distribution_l1965_196546


namespace NUMINAMATH_GPT_ribbons_at_start_l1965_196534

theorem ribbons_at_start (morning_ribbons : ℕ) (afternoon_ribbons : ℕ) (left_ribbons : ℕ)
  (h_morning : morning_ribbons = 14) (h_afternoon : afternoon_ribbons = 16) (h_left : left_ribbons = 8) :
  morning_ribbons + afternoon_ribbons + left_ribbons = 38 :=
by
  sorry

end NUMINAMATH_GPT_ribbons_at_start_l1965_196534


namespace NUMINAMATH_GPT_angle_conversion_l1965_196563

/--
 Given an angle in degrees, express it in degrees, minutes, and seconds.
 Theorem: 20.23 degrees can be converted to 20 degrees, 13 minutes, and 48 seconds.
-/
theorem angle_conversion : (20.23:ℝ) = 20 + (13/60 : ℝ) + (48/3600 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_angle_conversion_l1965_196563


namespace NUMINAMATH_GPT_blister_slowdown_l1965_196541

theorem blister_slowdown
    (old_speed new_speed time : ℕ) (new_speed_initial : ℕ) (blister_freq : ℕ)
    (distance_old : ℕ) (blister_per_hour_slowdown : ℝ):
    -- Given conditions
    old_speed = 6 →
    new_speed = 11 →
    new_speed_initial = 11 →
    time = 4 →
    blister_freq = 2 →
    distance_old = old_speed * time →
    -- Prove that each blister slows Candace down by 10 miles per hour
    blister_per_hour_slowdown = 10 :=
  by
    sorry

end NUMINAMATH_GPT_blister_slowdown_l1965_196541


namespace NUMINAMATH_GPT_div_m_by_18_equals_500_l1965_196543

-- Define the conditions
noncomputable def m : ℕ := 9000 -- 'm' is given as 9000 since it fulfills all conditions described
def is_multiple_of_18 (n : ℕ) : Prop := n % 18 = 0
def all_digits_9_or_0 (n : ℕ) : Prop := ∀ (d : ℕ), (∃ (k : ℕ), n = 10^k * d) → (d = 0 ∨ d = 9)

-- Define the proof problem statement
theorem div_m_by_18_equals_500 
  (h1 : is_multiple_of_18 m) 
  (h2 : all_digits_9_or_0 m) 
  (h3 : ∀ n, is_multiple_of_18 n ∧ all_digits_9_or_0 n → n ≤ m) : 
  m / 18 = 500 :=
sorry

end NUMINAMATH_GPT_div_m_by_18_equals_500_l1965_196543


namespace NUMINAMATH_GPT_no_valid_n_l1965_196508

theorem no_valid_n (n : ℕ) : (100 ≤ n / 4 ∧ n / 4 ≤ 999) → (100 ≤ 4 * n ∧ 4 * n ≤ 999) → false :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_no_valid_n_l1965_196508


namespace NUMINAMATH_GPT_min_heaviest_weight_l1965_196577

theorem min_heaviest_weight : 
  ∃ (w : ℕ), ∀ (weights : Fin 8 → ℕ),
    (∀ i j, i ≠ j → weights i ≠ weights j) ∧
    (∀ (a b c d : Fin 8),
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → 
      (weights a + weights b) ≠ (weights c + weights d) ∧ 
      max (max (weights a) (weights b)) (max (weights c) (weights d)) >= w) 
  → w = 34 := 
by
  sorry

end NUMINAMATH_GPT_min_heaviest_weight_l1965_196577


namespace NUMINAMATH_GPT_smallest_n_equal_sums_l1965_196580

def sum_first_n_arithmetic (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem smallest_n_equal_sums : ∀ (n : ℕ), 
  sum_first_n_arithmetic 7 4 n = sum_first_n_arithmetic 15 3 n → n ≠ 0 → n = 7 := by
  intros n h1 h2
  sorry

end NUMINAMATH_GPT_smallest_n_equal_sums_l1965_196580


namespace NUMINAMATH_GPT_correct_option_is_D_l1965_196579

-- Define the expressions to be checked
def exprA (x : ℝ) := 3 * x + 2 * x = 5 * x^2
def exprB (x : ℝ) := -2 * x^2 * x^3 = 2 * x^5
def exprC (x y : ℝ) := (y + 3 * x) * (3 * x - y) = y^2 - 9 * x^2
def exprD (x y : ℝ) := (-2 * x^2 * y)^3 = -8 * x^6 * y^3

theorem correct_option_is_D (x y : ℝ) : 
  ¬ exprA x ∧ 
  ¬ exprB x ∧ 
  ¬ exprC x y ∧ 
  exprD x y := by
  -- The proof would be provided here
  sorry

end NUMINAMATH_GPT_correct_option_is_D_l1965_196579


namespace NUMINAMATH_GPT_smallest_four_digit_multiple_of_18_l1965_196551

-- Define the concept of a four-digit number
def four_digit (N : ℕ) : Prop := 1000 ≤ N ∧ N < 10000

-- Define the concept of a multiple of 18
def multiple_of_18 (N : ℕ) : Prop := ∃ k : ℕ, N = 18 * k

-- Define the combined condition for N being a four-digit multiple of 18
def four_digit_multiple_of_18 (N : ℕ) : Prop := four_digit N ∧ multiple_of_18 N

-- State that 1008 is the smallest such number
theorem smallest_four_digit_multiple_of_18 : ∀ N : ℕ, four_digit_multiple_of_18 N → 1008 ≤ N := 
by
  intros N H
  sorry

end NUMINAMATH_GPT_smallest_four_digit_multiple_of_18_l1965_196551


namespace NUMINAMATH_GPT_sin_14pi_div_3_eq_sqrt3_div_2_l1965_196523

theorem sin_14pi_div_3_eq_sqrt3_div_2 : Real.sin (14 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_14pi_div_3_eq_sqrt3_div_2_l1965_196523


namespace NUMINAMATH_GPT_mashed_potatoes_count_l1965_196576

theorem mashed_potatoes_count :
  ∀ (b s : ℕ), b = 489 → b = s + 10 → s = 479 :=
by
  intros b s h₁ h₂
  sorry

end NUMINAMATH_GPT_mashed_potatoes_count_l1965_196576


namespace NUMINAMATH_GPT_mike_total_cards_l1965_196568

-- Given conditions
def mike_original_cards : ℕ := 87
def sam_given_cards : ℕ := 13

-- Question equivalence in Lean: Prove that Mike has 100 baseball cards now
theorem mike_total_cards : mike_original_cards + sam_given_cards = 100 :=
by 
  sorry

end NUMINAMATH_GPT_mike_total_cards_l1965_196568


namespace NUMINAMATH_GPT_sum_f_neg12_to_13_l1965_196573

noncomputable def f (x : ℝ) := 1 / (3^x + Real.sqrt 3)

theorem sum_f_neg12_to_13 : 
  (f (-12) + f (-11) + f (-10) + f (-9) + f (-8) + f (-7) + f (-6)
  + f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f 0
  + f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10
  + f 11 + f 12 + f 13) = (13 * Real.sqrt 3 / 3) :=
sorry

end NUMINAMATH_GPT_sum_f_neg12_to_13_l1965_196573


namespace NUMINAMATH_GPT_value_of_e_l1965_196525

theorem value_of_e (a : ℕ) (e : ℕ) 
  (h1 : a = 105) 
  (h2 : a^3 = 21 * 25 * 45 * e) : 
  e = 49 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_e_l1965_196525


namespace NUMINAMATH_GPT_first_range_is_30_l1965_196545

theorem first_range_is_30 
  (R2 R3 : ℕ)
  (h1 : R2 = 26)
  (h2 : R3 = 32)
  (h3 : min 26 (min 30 32) = 30) : 
  ∃ R1 : ℕ, R1 = 30 :=
  sorry

end NUMINAMATH_GPT_first_range_is_30_l1965_196545


namespace NUMINAMATH_GPT_consecutive_even_numbers_l1965_196583

theorem consecutive_even_numbers (n m : ℕ) (h : 52 * (2 * n - 1) = 100 * n) : n = 13 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_even_numbers_l1965_196583


namespace NUMINAMATH_GPT_fred_now_has_l1965_196548

-- Definitions based on conditions
def original_cards : ℕ := 40
def purchased_cards : ℕ := 22

-- Theorem to prove the number of cards Fred has now
theorem fred_now_has (original_cards : ℕ) (purchased_cards : ℕ) : original_cards - purchased_cards = 18 :=
by
  sorry

end NUMINAMATH_GPT_fred_now_has_l1965_196548


namespace NUMINAMATH_GPT_minimal_area_circle_equation_circle_equation_center_on_line_l1965_196553

-- Question (1): Prove the equation of the circle with minimal area
theorem minimal_area_circle_equation :
  (∃ (C : ℝ × ℝ) (r : ℝ), (r > 0) ∧ 
  C = (0, -4) ∧ r = Real.sqrt 5 ∧ 
  ∀ (P : ℝ × ℝ), (P = (2, -3) ∨ P = (-2, -5)) → P.1 ^ 2 + (P.2 + 4) ^ 2 = 5) :=
sorry

-- Question (2): Prove the equation of a circle with the center on a specific line
theorem circle_equation_center_on_line :
  (∃ (C : ℝ × ℝ) (r : ℝ), (r > 0) ∧ 
  (C.1 - 2 * C.2 - 3 = 0) ∧
  C = (-1, -2) ∧ r = Real.sqrt 10 ∧ 
  ∀ (P : ℝ × ℝ), (P = (2, -3) ∨ P = (-2, -5)) → (P.1 + 1) ^ 2 + (P.2 + 2) ^ 2 = 10) :=
sorry

end NUMINAMATH_GPT_minimal_area_circle_equation_circle_equation_center_on_line_l1965_196553


namespace NUMINAMATH_GPT_females_over_30_prefer_webstream_l1965_196578

-- Define the total number of survey participants
def total_participants : ℕ := 420

-- Define the number of participants who prefer WebStream
def prefer_webstream : ℕ := 200

-- Define the number of participants who do not prefer WebStream
def not_prefer_webstream : ℕ := 220

-- Define the number of males who prefer WebStream
def males_prefer : ℕ := 80

-- Define the number of females under 30 who do not prefer WebStream
def females_under_30_not_prefer : ℕ := 90

-- Define the number of females over 30 who do not prefer WebStream
def females_over_30_not_prefer : ℕ := 70

-- Define the total number of females under 30 who do not prefer WebStream
def females_not_prefer : ℕ := females_under_30_not_prefer + females_over_30_not_prefer

-- Define the total number of participants who do not prefer WebStream
def total_not_prefer : ℕ := 220

-- Define the number of males who do not prefer WebStream
def males_not_prefer : ℕ := total_not_prefer - females_not_prefer

-- Define the number of females who prefer WebStream
def females_prefer : ℕ := prefer_webstream - males_prefer

-- Define the total number of females under 30 who prefer WebStream
def females_under_30_prefer : ℕ := total_participants - prefer_webstream - females_under_30_not_prefer

-- Define the remaining females over 30 who prefer WebStream
def females_over_30_prefer : ℕ := females_prefer - females_under_30_prefer

-- The Lean statement to prove
theorem females_over_30_prefer_webstream : females_over_30_prefer = 110 := by
  sorry

end NUMINAMATH_GPT_females_over_30_prefer_webstream_l1965_196578


namespace NUMINAMATH_GPT_division_identity_l1965_196517

theorem division_identity (h : 6 / 3 = 2) : 72 / (6 / 3) = 36 := by
  sorry

end NUMINAMATH_GPT_division_identity_l1965_196517


namespace NUMINAMATH_GPT_directrix_of_parabola_l1965_196574

theorem directrix_of_parabola : 
  (∀ (y x: ℝ), y^2 = 12 * x → x = -3) :=
sorry

end NUMINAMATH_GPT_directrix_of_parabola_l1965_196574


namespace NUMINAMATH_GPT_dolphins_score_l1965_196550

theorem dolphins_score (S D : ℕ) (h1 : S + D = 48) (h2 : S = D + 20) : D = 14 := by
    sorry

end NUMINAMATH_GPT_dolphins_score_l1965_196550


namespace NUMINAMATH_GPT_ian_money_left_l1965_196562

theorem ian_money_left
  (hours_worked : ℕ)
  (hourly_rate : ℕ)
  (spending_percentage : ℚ)
  (total_earnings : ℕ)
  (amount_spent : ℕ)
  (amount_left : ℕ)
  (h_worked : hours_worked = 8)
  (h_rate : hourly_rate = 18)
  (h_spending : spending_percentage = 0.5)
  (h_earnings : total_earnings = hours_worked * hourly_rate)
  (h_spent : amount_spent = total_earnings * spending_percentage)
  (h_left : amount_left = total_earnings - amount_spent) :
  amount_left = 72 := 
  sorry

end NUMINAMATH_GPT_ian_money_left_l1965_196562


namespace NUMINAMATH_GPT_m_range_l1965_196529

variable (a1 b1 : ℝ)

def arithmetic_sequence (n : ℕ) : ℝ := a1 + 2 * (n - 1)
def geometric_sequence (n : ℕ) : ℝ := b1 * 2^(n - 1)

def a2_condition : Prop := arithmetic_sequence a1 2 + geometric_sequence b1 2 < -2
def a1_b1_condition : Prop := a1 + b1 > 0

theorem m_range : a1_b1_condition a1 b1 ∧ a2_condition a1 b1 → 
  let a4 := arithmetic_sequence a1 4 
  let b3 := geometric_sequence b1 3 
  let m := a4 + b3 
  m < 0 := 
by
  sorry

end NUMINAMATH_GPT_m_range_l1965_196529


namespace NUMINAMATH_GPT_value_of_f_at_2_l1965_196565

def f (x : ℝ) : ℝ := x^3 - x^2 - 1

theorem value_of_f_at_2 : f 2 = 3 := by
  sorry

end NUMINAMATH_GPT_value_of_f_at_2_l1965_196565
