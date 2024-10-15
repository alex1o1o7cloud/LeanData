import Mathlib

namespace NUMINAMATH_GPT_area_of_triangle_is_right_angled_l472_47254

noncomputable def vector_a : ℝ × ℝ := (3, 4)
noncomputable def vector_b : ℝ × ℝ := (-4, 3)

theorem area_of_triangle_is_right_angled (h1 : vector_a = (3, 4)) (h2 : vector_b = (-4, 3)) : 
  let det := vector_a.1 * vector_b.2 - vector_a.2 * vector_b.1
  (1 / 2) * abs det = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_is_right_angled_l472_47254


namespace NUMINAMATH_GPT_streetlights_each_square_l472_47297

-- Define the conditions
def total_streetlights : Nat := 200
def total_squares : Nat := 15
def unused_streetlights : Nat := 20

-- State the question mathematically
def streetlights_installed := total_streetlights - unused_streetlights
def streetlights_per_square := streetlights_installed / total_squares

-- The theorem we need to prove
theorem streetlights_each_square : streetlights_per_square = 12 := sorry

end NUMINAMATH_GPT_streetlights_each_square_l472_47297


namespace NUMINAMATH_GPT_diagonal_of_square_l472_47235

theorem diagonal_of_square (s d : ℝ) (h_perimeter : 4 * s = 40) : d = 10 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_diagonal_of_square_l472_47235


namespace NUMINAMATH_GPT_x_squared_plus_y_squared_l472_47262

theorem x_squared_plus_y_squared (x y : ℝ) 
   (h1 : (x + y)^2 = 49) 
   (h2 : x * y = 8) 
   : x^2 + y^2 = 33 := 
by
  sorry

end NUMINAMATH_GPT_x_squared_plus_y_squared_l472_47262


namespace NUMINAMATH_GPT_min_k_l472_47277

noncomputable 
def f (k : ℕ) (x : ℝ) : ℝ := 
  (Real.sin (k * x / 10)) ^ 4 + (Real.cos (k * x / 10)) ^ 4

theorem min_k (k : ℕ) 
    (h : (∀ a : ℝ, {y | ∃ x : ℝ, a < x ∧ x < a+1 ∧ y = f k x} = 
                  {y | ∃ x : ℝ, y = f k x})) 
    : k ≥ 16 :=
by
  sorry

end NUMINAMATH_GPT_min_k_l472_47277


namespace NUMINAMATH_GPT_exercise_books_purchasing_methods_l472_47250

theorem exercise_books_purchasing_methods :
  ∃ (ways : ℕ), ways = 5 ∧
  (∃ (x y z : ℕ), 2 * x + 5 * y + 11 * z = 40 ∧ x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1) ∧
  (∀ (x₁ y₁ z₁ x₂ y₂ z₂ : ℕ),
    2 * x₁ + 5 * y₁ + 11 * z₂ = 40 ∧ x₁ ≥ 1 ∧ y₁ ≥ 1 ∧ z₁ ≥ 1 →
    2 * x₂ + 5 * y₂ + 11 * z₂ = 40 ∧ x₂ ≥ 1 ∧ y₂ ≥ 1 ∧ z₂ ≥ 1 →
    (x₁, y₁, z₁) = (x₂, y₂, z₂)) := sorry

end NUMINAMATH_GPT_exercise_books_purchasing_methods_l472_47250


namespace NUMINAMATH_GPT_alpha_range_in_first_quadrant_l472_47281

open Real

theorem alpha_range_in_first_quadrant (k : ℤ) (α : ℝ) 
  (h1 : cos α ≤ sin α) : 
  (2 * k * π + π / 4) ≤ α ∧ α < (2 * k * π + π / 2) :=
sorry

end NUMINAMATH_GPT_alpha_range_in_first_quadrant_l472_47281


namespace NUMINAMATH_GPT_find_C_l472_47270

theorem find_C (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 350) : C = 50 := 
by
  sorry

end NUMINAMATH_GPT_find_C_l472_47270


namespace NUMINAMATH_GPT_geometric_progression_coincides_arithmetic_l472_47285

variables (a d q : ℝ)
variables (ap : ℕ → ℝ) (gp : ℕ → ℝ)

-- Define the N-th term of the arithmetic progression
def nth_term_ap (n : ℕ) : ℝ := a + n * d

-- Define the N-th term of the geometric progression
def nth_term_gp (n : ℕ) : ℝ := a * q^n

theorem geometric_progression_coincides_arithmetic :
  gp 3 = ap 10 →
  gp 4 = ap 74 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_geometric_progression_coincides_arithmetic_l472_47285


namespace NUMINAMATH_GPT_geometric_sequence_third_term_l472_47243

theorem geometric_sequence_third_term (a : ℕ → ℕ) (x : ℕ) (h1 : a 1 = 4) (h2 : a 2 = 6) (h3 : a 3 = x) (h_geom : ∀ n, a (n + 1) = a n * r) :
  x = 9 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_third_term_l472_47243


namespace NUMINAMATH_GPT_one_cow_one_bag_l472_47231

def husk_eating (C B D : ℕ) : Prop :=
  C * D / B = D

theorem one_cow_one_bag (C B D n : ℕ) (h : husk_eating C B D) (hC : C = 46) (hB : B = 46) (hD : D = 46) : n = D :=
by
  rw [hC, hB, hD] at h
  sorry

end NUMINAMATH_GPT_one_cow_one_bag_l472_47231


namespace NUMINAMATH_GPT_polynomial_sum_l472_47223

def f (x : ℝ) : ℝ := -4 * x^3 + 2 * x^2 - 5 * x - 7
def g (x : ℝ) : ℝ := 6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := -x^3 + 3 * x^2 + 2 * x + 8

theorem polynomial_sum (x : ℝ) : f x + g x + h x = -5 * x^3 + 11 * x^2 + x - 8 :=
  sorry

end NUMINAMATH_GPT_polynomial_sum_l472_47223


namespace NUMINAMATH_GPT_total_students_l472_47212

theorem total_students (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) (hshake : (2 * m * n - m - n) = 252) : m * n = 72 :=
  sorry

end NUMINAMATH_GPT_total_students_l472_47212


namespace NUMINAMATH_GPT_largest_4_digit_divisible_by_88_and_prime_gt_100_l472_47209

noncomputable def is_4_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

noncomputable def is_divisible_by (n d : ℕ) : Prop :=
  d ∣ n

noncomputable def is_prime (p : ℕ) : Prop :=
  Nat.Prime p

noncomputable def lcm (a b : ℕ) : ℕ :=
  Nat.lcm a b

theorem largest_4_digit_divisible_by_88_and_prime_gt_100 (p : ℕ) (hp : is_prime p) (h1 : 100 < p):
  ∃ n, is_4_digit n ∧ is_divisible_by n 88 ∧ is_divisible_by n p ∧
       (∀ m, is_4_digit m ∧ is_divisible_by m 88 ∧ is_divisible_by m p → m ≤ n) :=
sorry

end NUMINAMATH_GPT_largest_4_digit_divisible_by_88_and_prime_gt_100_l472_47209


namespace NUMINAMATH_GPT_exists_permutation_with_large_neighbor_difference_l472_47294

theorem exists_permutation_with_large_neighbor_difference :
  ∃ (σ : Fin 100 → Fin 100), 
    (∀ (i : Fin 99), (|σ i.succ - σ i| ≥ 50)) :=
sorry

end NUMINAMATH_GPT_exists_permutation_with_large_neighbor_difference_l472_47294


namespace NUMINAMATH_GPT_petya_no_win_implies_draw_or_lost_l472_47267

noncomputable def petya_cannot_win (n : ℕ) (h : n ≥ 3) : Prop :=
  ∀ (Petya_strategy Vasya_strategy : ℕ → ℕ → ℕ),
    ∃ m : ℕ, Petya_strategy m ≠ Vasya_strategy m

theorem petya_no_win_implies_draw_or_lost (n : ℕ) (h : n ≥ 3) :
  ¬ ∃ Petya_strategy Vasya_strategy : ℕ → ℕ → ℕ, 
    (∀ m : ℕ, Petya_strategy m = Vasya_strategy m) :=
by {
  sorry
}

end NUMINAMATH_GPT_petya_no_win_implies_draw_or_lost_l472_47267


namespace NUMINAMATH_GPT_production_equipment_B_l472_47214

theorem production_equipment_B :
  ∃ (X Y : ℕ), X + Y = 4800 ∧ (50 / 80 = 5 / 8) ∧ (X / 4800 = 5 / 8) ∧ Y = 1800 :=
by
  sorry

end NUMINAMATH_GPT_production_equipment_B_l472_47214


namespace NUMINAMATH_GPT_value_of_f_5_l472_47276

variable (a b c m : ℝ)

-- Conditions: definition of f and given value of f(-5)
def f (x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2
axiom H1 : f a b c (-5) = m

-- Question: Prove that f(5) = -m + 4
theorem value_of_f_5 : f a b c 5 = -m + 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_5_l472_47276


namespace NUMINAMATH_GPT_glasses_displayed_is_correct_l472_47249

-- Definitions from the problem conditions
def tall_cupboard_capacity : Nat := 20
def wide_cupboard_capacity : Nat := 2 * tall_cupboard_capacity
def per_shelf_narrow_cupboard : Nat := 15 / 3
def usable_narrow_cupboard_capacity : Nat := 2 * per_shelf_narrow_cupboard

-- Theorem to prove that the total number of glasses displayed is 70
theorem glasses_displayed_is_correct :
  (tall_cupboard_capacity + wide_cupboard_capacity + usable_narrow_cupboard_capacity) = 70 :=
by
  sorry

end NUMINAMATH_GPT_glasses_displayed_is_correct_l472_47249


namespace NUMINAMATH_GPT_camila_bikes_more_l472_47200

-- Definitions based on conditions
def camila_speed : ℝ := 15
def daniel_speed_initial : ℝ := 15
def daniel_speed_after_3hours : ℝ := 10
def biking_time : ℝ := 6
def time_before_decrease : ℝ := 3
def time_after_decrease : ℝ := biking_time - time_before_decrease

def distance_camila := camila_speed * biking_time
def distance_daniel := (daniel_speed_initial * time_before_decrease) + (daniel_speed_after_3hours * time_after_decrease)

-- The statement to prove: Camila has biked 15 more miles than Daniel
theorem camila_bikes_more : distance_camila - distance_daniel = 15 := 
by
  sorry

end NUMINAMATH_GPT_camila_bikes_more_l472_47200


namespace NUMINAMATH_GPT_number_of_free_ranging_chickens_is_105_l472_47291

namespace ChickenProblem

-- Conditions as definitions
def coop_chickens : ℕ := 14
def run_chickens : ℕ := 2 * coop_chickens
def free_ranging_chickens : ℕ := 2 * run_chickens - 4
def total_coop_run_chickens : ℕ := coop_chickens + run_chickens

-- The ratio condition
def ratio_condition : Prop :=
  (coop_chickens + run_chickens) * 5 = free_ranging_chickens * 2

-- Proof Statement
theorem number_of_free_ranging_chickens_is_105 :
  free_ranging_chickens = 105 :=
by {
  sorry
}

end ChickenProblem

end NUMINAMATH_GPT_number_of_free_ranging_chickens_is_105_l472_47291


namespace NUMINAMATH_GPT_thickness_of_stack_l472_47236

theorem thickness_of_stack (books : ℕ) (avg_pages_per_book : ℕ) (pages_per_inch : ℕ) (total_pages : ℕ) (thick_in_inches : ℕ)
    (h1 : books = 6)
    (h2 : avg_pages_per_book = 160)
    (h3 : pages_per_inch = 80)
    (h4 : total_pages = books * avg_pages_per_book)
    (h5 : thick_in_inches = total_pages / pages_per_inch) :
    thick_in_inches = 12 :=
by {
    -- statement without proof
    sorry
}

end NUMINAMATH_GPT_thickness_of_stack_l472_47236


namespace NUMINAMATH_GPT_lcm_of_two_numbers_l472_47264

theorem lcm_of_two_numbers (A B : ℕ) (h1 : A * B = 62216) (h2 : Nat.gcd A B = 22) :
  Nat.lcm A B = 2828 :=
by
  sorry

end NUMINAMATH_GPT_lcm_of_two_numbers_l472_47264


namespace NUMINAMATH_GPT_mike_salary_calculation_l472_47256

theorem mike_salary_calculation
  (F : ℝ) (M : ℝ) (new_M : ℝ) (x : ℝ)
  (F_eq : F = 1000)
  (M_eq : M = x * F)
  (increase_eq : new_M = 1.40 * M)
  (new_M_val : new_M = 15400) :
  M = 11000 ∧ x = 11 :=
by
  sorry

end NUMINAMATH_GPT_mike_salary_calculation_l472_47256


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l472_47293

theorem hyperbola_eccentricity (a b : ℝ) (h : a^2 = 4 ∧ b^2 = 3) :
    let c := Real.sqrt (a^2 + b^2)
    let e := c / a
    e = Real.sqrt 7 / 2 :=
    by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l472_47293


namespace NUMINAMATH_GPT_sum_of_five_integers_l472_47296

-- Definitions of the five integers based on the conditions given in the problem
def a := 12345
def b := 23451
def c := 34512
def d := 45123
def e := 51234

-- Statement of the proof problem
theorem sum_of_five_integers :
  a + b + c + d + e = 166665 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_sum_of_five_integers_l472_47296


namespace NUMINAMATH_GPT_workers_distribution_l472_47261

theorem workers_distribution (x y : ℕ) (h1 : x + y = 32) (h2 : 2 * 5 * x = 6 * y) : 
  (∃ x y : ℕ, x + y = 32 ∧ 2 * 5 * x = 6 * y) :=
sorry

end NUMINAMATH_GPT_workers_distribution_l472_47261


namespace NUMINAMATH_GPT_find_x_l472_47299

-- Definitions based directly on conditions
def vec_a : ℝ × ℝ := (2, 4)
def vec_b (x : ℝ) : ℝ × ℝ := (x, 3)
def vec_c (x : ℝ) : ℝ × ℝ := (2 - x, 1)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The mathematically equivalent proof problem statement
theorem find_x (x : ℝ) : dot_product (vec_c x) (vec_b x) = 0 → (x = -1 ∨ x = 3) :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_find_x_l472_47299


namespace NUMINAMATH_GPT_y_intercept_of_line_l472_47288

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : y = 4 :=
by sorry

end NUMINAMATH_GPT_y_intercept_of_line_l472_47288


namespace NUMINAMATH_GPT_second_more_than_third_l472_47213

def firstChapterPages : ℕ := 35
def secondChapterPages : ℕ := 18
def thirdChapterPages : ℕ := 3

theorem second_more_than_third : secondChapterPages - thirdChapterPages = 15 := by
  sorry

end NUMINAMATH_GPT_second_more_than_third_l472_47213


namespace NUMINAMATH_GPT_upper_limit_of_prime_range_l472_47274

theorem upper_limit_of_prime_range : 
  ∃ x : ℝ, (26 / 3 < 11) ∧ (11 < x) ∧ (x < 17) :=
by
  sorry

end NUMINAMATH_GPT_upper_limit_of_prime_range_l472_47274


namespace NUMINAMATH_GPT_calculate_expression_l472_47234

theorem calculate_expression :
  (6 * 5 * 4 * 3 * 2 * 1 - 5 * 4 * 3 * 2 * 1) / (4 * 3 * 2 * 1) = 25 := 
by sorry

end NUMINAMATH_GPT_calculate_expression_l472_47234


namespace NUMINAMATH_GPT_cupric_cyanide_formed_l472_47211

-- Definition of the problem
def formonitrile : ℕ := 6
def copper_sulfate : ℕ := 3
def sulfuric_acid : ℕ := 3

-- Stoichiometry from the balanced equation
def stoichiometry (hcn mol_multiplier: ℕ): ℕ := 
  (hcn / mol_multiplier)

theorem cupric_cyanide_formed :
  stoichiometry formonitrile 2 = 3 := 
sorry

end NUMINAMATH_GPT_cupric_cyanide_formed_l472_47211


namespace NUMINAMATH_GPT_cost_of_gravelling_path_eq_630_l472_47221

-- Define the dimensions of the grassy plot.
def length_grassy_plot : ℝ := 110
def width_grassy_plot : ℝ := 65

-- Define the width of the gravel path.
def width_gravel_path : ℝ := 2.5

-- Define the cost of gravelling per square meter in INR.
def cost_per_sqm : ℝ := 0.70

-- Compute the dimensions of the plot including the gravel path.
def length_including_path := length_grassy_plot + 2 * width_gravel_path
def width_including_path := width_grassy_plot + 2 * width_gravel_path

-- Compute the area of the plot including the gravel path.
def area_including_path := length_including_path * width_including_path

-- Compute the area of the grassy plot without the gravel path.
def area_grassy_plot := length_grassy_plot * width_grassy_plot

-- Compute the area of the gravel path alone.
def area_gravel_path := area_including_path - area_grassy_plot

-- Compute the total cost of gravelling the path.
def total_cost := area_gravel_path * cost_per_sqm

-- The theorem stating the cost of gravelling the path.
theorem cost_of_gravelling_path_eq_630 : total_cost = 630 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cost_of_gravelling_path_eq_630_l472_47221


namespace NUMINAMATH_GPT_sum_of_interior_angles_quadrilateral_l472_47298

-- Define the function for the sum of the interior angles
def sum_of_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

-- Theorem that the sum of the interior angles of a quadrilateral is 360 degrees
theorem sum_of_interior_angles_quadrilateral : sum_of_interior_angles 4 = 360 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_quadrilateral_l472_47298


namespace NUMINAMATH_GPT_solve_medium_apple_cost_l472_47233

def cost_small_apple : ℝ := 1.5
def cost_big_apple : ℝ := 3.0
def num_small_apples : ℕ := 6
def num_medium_apples : ℕ := 6
def num_big_apples : ℕ := 8
def total_cost : ℝ := 45

noncomputable def cost_medium_apple (M : ℝ) : Prop :=
  (6 * cost_small_apple) + (6 * M) + (8 * cost_big_apple) = total_cost

theorem solve_medium_apple_cost : ∃ M : ℝ, cost_medium_apple M ∧ M = 2 := by
  sorry

end NUMINAMATH_GPT_solve_medium_apple_cost_l472_47233


namespace NUMINAMATH_GPT_intersection_range_l472_47246

theorem intersection_range (k : ℝ) :
  (∃ x y : ℝ, y = k * x + k + 2 ∧ y = -2 * x + 4 ∧ x > 0 ∧ y > 0) ↔ -2/3 < k ∧ k < 2 :=
by
  sorry

end NUMINAMATH_GPT_intersection_range_l472_47246


namespace NUMINAMATH_GPT_if_a_eq_b_then_a_squared_eq_b_squared_l472_47222

theorem if_a_eq_b_then_a_squared_eq_b_squared (a b : ℝ) (h : a = b) : a^2 = b^2 :=
sorry

end NUMINAMATH_GPT_if_a_eq_b_then_a_squared_eq_b_squared_l472_47222


namespace NUMINAMATH_GPT_karlsson_weight_l472_47287

variable {F K M : ℕ}

theorem karlsson_weight (h1 : F + K = M + 120) (h2 : K + M = F + 60) : K = 90 := by
  sorry

end NUMINAMATH_GPT_karlsson_weight_l472_47287


namespace NUMINAMATH_GPT_right_angled_triangles_with_cathetus_2021_l472_47244

theorem right_angled_triangles_with_cathetus_2021 :
  ∃ n : Nat, n = 4 ∧ ∀ (a b c : ℕ), ((a = 2021 ∧ a * a + b * b = c * c) ↔ (a = 2021 ∧ 
    ∃ m n, (m > n ∧ m > 0 ∧ n > 0 ∧ 2021 = m^2 - n^2 ∧ b = 2 * m * n ∧ c = m^2 + n^2))) :=
sorry

end NUMINAMATH_GPT_right_angled_triangles_with_cathetus_2021_l472_47244


namespace NUMINAMATH_GPT_range_of_a_l472_47265

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 + (a - 1) * x + 1 / 4 > 0) ↔ (Real.sqrt 5 - 3) / 2 < a ∧ a < (3 + Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l472_47265


namespace NUMINAMATH_GPT_expand_expression_l472_47207

variable (x y z : ℕ)

theorem expand_expression (x y z: ℕ) : 
  (x + 10) * (3 * y + 5 * z + 15) = 3 * x * y + 5 * x * z + 15 * x + 30 * y + 50 * z + 150 :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l472_47207


namespace NUMINAMATH_GPT_symmetric_line_equation_y_axis_l472_47260

theorem symmetric_line_equation_y_axis (x y : ℝ) : 
  (∃ m n : ℝ, (y = 3 * x + 1) ∧ (x + m = 0) ∧ (y = n) ∧ (n = 3 * m + 1)) → 
  y = -3 * x + 1 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_line_equation_y_axis_l472_47260


namespace NUMINAMATH_GPT_sqrt_74_between_8_and_9_product_of_consecutive_integers_l472_47242

theorem sqrt_74_between_8_and_9 : 8 < Real.sqrt 74 ∧ Real.sqrt 74 < 9 := sorry

theorem product_of_consecutive_integers (h : 8 < Real.sqrt 74 ∧ Real.sqrt 74 < 9) : 8 * 9 = 72 := by
  have h1 : 8 < Real.sqrt 74 := And.left h
  have h2 : Real.sqrt 74 < 9 := And.right h
  calc
    8 * 9 = 72 := by norm_num

end NUMINAMATH_GPT_sqrt_74_between_8_and_9_product_of_consecutive_integers_l472_47242


namespace NUMINAMATH_GPT_monomial_2015_l472_47289

def a (n : ℕ) : ℤ := (-1 : ℤ)^n * (2 * n - 1)

theorem monomial_2015 :
  a 2015 * (x : ℤ) ^ 2015 = -4029 * (x : ℤ) ^ 2015 :=
by
  sorry

end NUMINAMATH_GPT_monomial_2015_l472_47289


namespace NUMINAMATH_GPT_matrix_equation_l472_47219

-- Definitions from conditions
def N : Matrix (Fin 2) (Fin 2) ℤ := ![![ -1, 4], ![ -6, 3]]
def I : Matrix (Fin 2) (Fin 2) ℤ := 1  -- Identity matrix

-- Given calculation of N^2
def N_squared : Matrix (Fin 2) (Fin 2) ℤ := ![![ -23, 8], ![ -12, -15]]

-- Goal: prove that N^2 = r*N + s*I for r = 2 and s = -21
theorem matrix_equation (r s : ℤ) (h_r : r = 2) (h_s : s = -21) : N_squared = r • N + s • I := by
  sorry

end NUMINAMATH_GPT_matrix_equation_l472_47219


namespace NUMINAMATH_GPT_sum_of_operations_l472_47203

noncomputable def triangle (a b c : ℕ) : ℕ :=
  a + 2 * b - c

theorem sum_of_operations :
  triangle 3 5 7 + triangle 6 1 8 = 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_operations_l472_47203


namespace NUMINAMATH_GPT_height_of_removed_player_l472_47217

theorem height_of_removed_player (S : ℕ) (x : ℕ) (total_height_11 : S + x = 182 * 11)
  (average_height_10 : S = 181 * 10): x = 192 :=
by
  sorry

end NUMINAMATH_GPT_height_of_removed_player_l472_47217


namespace NUMINAMATH_GPT_factorial_expression_equiv_l472_47226

theorem factorial_expression_equiv :
  6 * Nat.factorial 6 + 5 * Nat.factorial 5 + 3 * Nat.factorial 4 + Nat.factorial 4 = 1416 := 
sorry

end NUMINAMATH_GPT_factorial_expression_equiv_l472_47226


namespace NUMINAMATH_GPT_ben_has_20_mms_l472_47273

theorem ben_has_20_mms (B_candies Ben_candies : ℕ) 
  (h1 : B_candies = 50) 
  (h2 : B_candies = Ben_candies + 30) : 
  Ben_candies = 20 := 
by
  sorry

end NUMINAMATH_GPT_ben_has_20_mms_l472_47273


namespace NUMINAMATH_GPT_range_of_m_l472_47232

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x < m - 1 ∨ x > m + 1) → x^2 - 2 * x - 3 > 0) → (0 ≤ m ∧ m ≤ 2) := 
sorry

end NUMINAMATH_GPT_range_of_m_l472_47232


namespace NUMINAMATH_GPT_function_odd_domain_of_f_range_of_f_l472_47253

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x^2 + 1) + x - 1) / (Real.sqrt (x^2 + 1) + x + 1)

theorem function_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  intro x
  sorry

theorem domain_of_f : ∀ x : ℝ, true :=
by
  intro x
  trivial

theorem range_of_f : ∀ y : ℝ, y ∈ Set.Ioo (-1 : ℝ) 1 :=
by
  intro y
  sorry

end NUMINAMATH_GPT_function_odd_domain_of_f_range_of_f_l472_47253


namespace NUMINAMATH_GPT_find_n_l472_47245

theorem find_n (a n : ℕ) 
  (h1 : a^2 % n = 8) 
  (h2 : a^3 % n = 25) 
  (h3 : n > 25) : 
  n = 113 := 
sorry

end NUMINAMATH_GPT_find_n_l472_47245


namespace NUMINAMATH_GPT_goods_train_passes_man_in_10_seconds_l472_47280

def goods_train_pass_time (man_speed_kmph goods_speed_kmph goods_length_m : ℕ) : ℕ :=
  let relative_speed_mps := (man_speed_kmph + goods_speed_kmph) * 1000 / 3600
  goods_length_m / relative_speed_mps

theorem goods_train_passes_man_in_10_seconds :
  goods_train_pass_time 55 60 320 = 10 := sorry

end NUMINAMATH_GPT_goods_train_passes_man_in_10_seconds_l472_47280


namespace NUMINAMATH_GPT_two_mul_seven_pow_n_plus_one_divisible_by_three_l472_47271

-- Definition of natural numbers
variable (n : ℕ)

-- Statement of the problem in Lean
theorem two_mul_seven_pow_n_plus_one_divisible_by_three (n : ℕ) : 3 ∣ (2 * 7^n + 1) := 
sorry

end NUMINAMATH_GPT_two_mul_seven_pow_n_plus_one_divisible_by_three_l472_47271


namespace NUMINAMATH_GPT_cricket_average_increase_l472_47202

theorem cricket_average_increase (runs_mean : ℕ) (innings : ℕ) (runs : ℕ) (new_runs : ℕ) (x : ℕ) :
  runs_mean = 35 → innings = 10 → runs = 79 → (total_runs : ℕ) = runs_mean * innings → 
  (new_total : ℕ) = total_runs + runs → (new_mean : ℕ) = new_total / (innings + 1) ∧ new_mean = runs_mean + x → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_cricket_average_increase_l472_47202


namespace NUMINAMATH_GPT_exists_multiple_of_prime_with_all_nines_digits_l472_47286

theorem exists_multiple_of_prime_with_all_nines_digits (p : ℕ) (hp_prime : Nat.Prime p) (h2 : p ≠ 2) (h5 : p ≠ 5) :
  ∃ n : ℕ, (∀ d ∈ (n.digits 10), d = 9) ∧ p ∣ n :=
by
  sorry

end NUMINAMATH_GPT_exists_multiple_of_prime_with_all_nines_digits_l472_47286


namespace NUMINAMATH_GPT_inequality_not_always_correct_l472_47204

variables (x y z : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x > y) (h₄ : z > 0)

theorem inequality_not_always_correct :
  ¬ ∀ z > 0, (xz^2 / z > yz^2 / z) :=
sorry

end NUMINAMATH_GPT_inequality_not_always_correct_l472_47204


namespace NUMINAMATH_GPT_products_not_all_greater_than_one_quarter_l472_47201

theorem products_not_all_greater_than_one_quarter
  (a b c : ℝ)
  (ha : 0 < a ∧ a < 1)
  (hb : 0 < b ∧ b < 1)
  (hc : 0 < c ∧ c < 1) :
  ¬ ((1 - a) * b > 1 / 4 ∧ (1 - b) * c > 1 / 4 ∧ (1 - c) * a > 1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_products_not_all_greater_than_one_quarter_l472_47201


namespace NUMINAMATH_GPT_number_of_ways_to_choose_a_pair_of_socks_l472_47284

-- Define the number of socks of each color
def white_socks := 5
def brown_socks := 5
def blue_socks := 5
def green_socks := 5

-- Define the total number of socks
def total_socks := white_socks + brown_socks + blue_socks + green_socks

-- Define the number of ways to choose 2 blue socks from 5 blue socks
def num_ways_choose_two_blue_socks : ℕ := Nat.choose blue_socks 2

-- The proof statement
theorem number_of_ways_to_choose_a_pair_of_socks :
  num_ways_choose_two_blue_socks = 10 :=
sorry

end NUMINAMATH_GPT_number_of_ways_to_choose_a_pair_of_socks_l472_47284


namespace NUMINAMATH_GPT_ratio_of_boys_to_total_l472_47272

theorem ratio_of_boys_to_total (p_b p_g : ℝ) (h1 : p_b + p_g = 1) (h2 : p_b = (2 / 3) * p_g) :
  p_b = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_boys_to_total_l472_47272


namespace NUMINAMATH_GPT_sphere_radius_ratio_l472_47225

theorem sphere_radius_ratio (R1 R2 : ℝ) (m n : ℝ) (hm : 1 < m) (hn : 1 < n) 
  (h_ratio1 : (2 * π * R1 * ((2 * R1) / (m + 1))) / (4 * π * R1 * R1) = 1 / (m + 1))
  (h_ratio2 : (2 * π * R2 * ((2 * R2) / (n + 1))) / (4 * π * R2 * R2) = 1 / (n + 1)): 
  R2 / R1 = ((m - 1) * (n + 1)) / ((m + 1) * (n - 1)) := 
by
  sorry

end NUMINAMATH_GPT_sphere_radius_ratio_l472_47225


namespace NUMINAMATH_GPT_ratio_of_chords_l472_47218

theorem ratio_of_chords 
  (E F G H Q : Type)
  (EQ GQ FQ HQ : ℝ)
  (h1 : EQ = 4)
  (h2 : GQ = 10)
  (h3 : EQ * FQ = GQ * HQ) : 
  FQ / HQ = 5 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_chords_l472_47218


namespace NUMINAMATH_GPT_not_in_second_column_l472_47229

theorem not_in_second_column : ¬∃ (n : ℕ), (1 ≤ n ∧ n ≤ 400) ∧ 3 * n + 1 = 131 :=
by sorry

end NUMINAMATH_GPT_not_in_second_column_l472_47229


namespace NUMINAMATH_GPT_find_a_b_transform_line_l472_47241

theorem find_a_b_transform_line (a b : ℝ) (hA : Matrix (Fin 2) (Fin 2) ℝ := ![![-1, a], ![b, 3]]) :
  (∀ x y : ℝ, (2 * (-(x) + a*y) - (b*x + 3*y) - 3 = 0) → (2*x - y - 3 = 0)) →
  a = 1 ∧ b = -4 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a_b_transform_line_l472_47241


namespace NUMINAMATH_GPT_minute_hand_travel_distance_l472_47275

theorem minute_hand_travel_distance :
  ∀ (r : ℝ), r = 8 → (45 / 60) * (2 * Real.pi * r) = 12 * Real.pi :=
by
  intros r r_eq
  sorry

end NUMINAMATH_GPT_minute_hand_travel_distance_l472_47275


namespace NUMINAMATH_GPT_probability_C_l472_47215

variable (pA pB pD pC : ℚ)
variable (hA : pA = 1 / 4)
variable (hB : pB = 1 / 3)
variable (hD : pD = 1 / 6)
variable (total_prob : pA + pB + pD + pC = 1)

theorem probability_C (hA : pA = 1 / 4) (hB : pB = 1 / 3) (hD : pD = 1 / 6) (total_prob : pA + pB + pD + pC = 1) : pC = 1 / 4 :=
sorry

end NUMINAMATH_GPT_probability_C_l472_47215


namespace NUMINAMATH_GPT_derivative_of_x_ln_x_l472_47240

noncomputable
def x_ln_x (x : ℝ) : ℝ := x * Real.log x

theorem derivative_of_x_ln_x (x : ℝ) (hx : x > 0) :
  deriv (x_ln_x) x = 1 + Real.log x :=
by
  -- Proof body, with necessary assumptions and justifications
  sorry

end NUMINAMATH_GPT_derivative_of_x_ln_x_l472_47240


namespace NUMINAMATH_GPT_urn_problem_l472_47237

noncomputable def count_balls (initial_white : ℕ) (initial_black : ℕ) (operations : ℕ) : ℕ :=
initial_white + initial_black + operations

noncomputable def urn_probability (initial_white : ℕ) (initial_black : ℕ) (operations : ℕ) (final_white : ℕ) (final_black : ℕ) : ℚ :=
if final_white + final_black = count_balls initial_white initial_black operations &&
   final_white = (initial_white + (operations - (final_black - initial_black))) &&
   (final_white + final_black) = 8 then 3 / 5 else 0

theorem urn_problem :
  let initial_white := 2
  let initial_black := 1
  let operations := 4
  let final_white := 4
  let final_black := 4
  count_balls initial_white initial_black operations = 8 ∧ urn_probability initial_white initial_black operations final_white final_black = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_urn_problem_l472_47237


namespace NUMINAMATH_GPT_train_length_l472_47224

/-
  Given:
  - Speed of the train is 78 km/h
  - Time to pass an electric pole is 5.0769230769230775 seconds
  We need to prove that the length of the train is 110 meters.
-/

def speed_kmph : ℝ := 78
def time_seconds : ℝ := 5.0769230769230775
def expected_length_meters : ℝ := 110

theorem train_length :
  (speed_kmph * 1000 / 3600) * time_seconds = expected_length_meters :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_train_length_l472_47224


namespace NUMINAMATH_GPT_describes_random_event_proof_l472_47247

def describes_random_event (phrase : String) : Prop :=
  match phrase with
  | "Winter turns into spring"  => False
  | "Fishing for the moon in the water" => False
  | "Seeking fish on a tree" => False
  | "Meeting unexpectedly" => True
  | _ => False

theorem describes_random_event_proof : describes_random_event "Meeting unexpectedly" = True :=
by
  sorry

end NUMINAMATH_GPT_describes_random_event_proof_l472_47247


namespace NUMINAMATH_GPT_solution_set_of_inequality_l472_47205

open Set

theorem solution_set_of_inequality :
  {x : ℝ | (x ≠ -2) ∧ (x ≠ -8) ∧ (2 / (x + 2) + 4 / (x + 8) ≥ 4 / 5)} =
  {x : ℝ | (-8 < x ∧ x < -2) ∨ (-2 < x ∧ x ≤ 4)} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l472_47205


namespace NUMINAMATH_GPT_tom_cheaper_than_jane_l472_47279

-- Define constants for Store A
def store_a_full_price : ℝ := 125
def store_a_discount_one : ℝ := 0.08
def store_a_discount_two : ℝ := 0.12
def store_a_tax : ℝ := 0.07

-- Define constants for Store B
def store_b_full_price : ℝ := 130
def store_b_discount_one : ℝ := 0.10
def store_b_discount_three : ℝ := 0.15
def store_b_tax : ℝ := 0.05

-- Define the number of smartphones bought by Tom and Jane
def tom_quantity : ℕ := 2
def jane_quantity : ℕ := 3

-- Define the final amount Tom pays
def final_amount_tom : ℝ :=
  let full_price := tom_quantity * store_a_full_price
  let discount := store_a_discount_two * full_price
  let discounted_price := full_price - discount
  let tax := store_a_tax * discounted_price
  discounted_price + tax

-- Define the final amount Jane pays
def final_amount_jane : ℝ :=
  let full_price := jane_quantity * store_b_full_price
  let discount := store_b_discount_three * full_price
  let discounted_price := full_price - discount
  let tax := store_b_tax * discounted_price
  discounted_price + tax

-- Prove that Tom's total cost is $112.68 cheaper than Jane's total cost
theorem tom_cheaper_than_jane : final_amount_jane - final_amount_tom = 112.68 :=
by
  have tom := final_amount_tom
  have jane := final_amount_jane
  sorry

end NUMINAMATH_GPT_tom_cheaper_than_jane_l472_47279


namespace NUMINAMATH_GPT_line_symmetric_about_y_eq_x_l472_47227

-- Define the line equation types and the condition for symmetry
def line_equation (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

-- Conditions given
variable (a b c : ℝ)
variable (h_ab_pos : a * b > 0)

-- Definition of the problem in Lean
theorem line_symmetric_about_y_eq_x (h_bisector : ∀ x y : ℝ, line_equation a b c x y ↔ line_equation b a c y x) : 
  ∀ x y : ℝ, line_equation b a c x y := by
  sorry

end NUMINAMATH_GPT_line_symmetric_about_y_eq_x_l472_47227


namespace NUMINAMATH_GPT_breadthOfRectangularPart_l472_47259

variable (b l : ℝ)

def rectangularAreaProblem : Prop :=
  (l * b + (1 / 12) * b * l = 24 * b) ∧ (l - b = 10)

theorem breadthOfRectangularPart :
  rectangularAreaProblem b l → b = 12.15 :=
by
  intros
  sorry

end NUMINAMATH_GPT_breadthOfRectangularPart_l472_47259


namespace NUMINAMATH_GPT_original_sum_of_money_l472_47220

theorem original_sum_of_money (P R : ℝ) 
  (h1 : 720 = P + (P * R * 2) / 100) 
  (h2 : 1020 = P + (P * R * 7) / 100) : 
  P = 600 := 
by sorry

end NUMINAMATH_GPT_original_sum_of_money_l472_47220


namespace NUMINAMATH_GPT_line_always_passes_fixed_point_l472_47208

theorem line_always_passes_fixed_point:
  ∀ a x y, x = 5 → y = -3 → (a * x + (2 * a - 1) * y + a - 3 = 0) :=
by
  intros a x y h1 h2
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_line_always_passes_fixed_point_l472_47208


namespace NUMINAMATH_GPT_ratio_adult_women_to_men_event_l472_47257

theorem ratio_adult_women_to_men_event :
  ∀ (total_members men_ratio women_ratio children : ℕ), 
  total_members = 2000 →
  men_ratio = 30 →
  children = 200 →
  women_ratio = men_ratio →
  women_ratio / men_ratio = 1 / 1 := 
by
  intros total_members men_ratio women_ratio children
  sorry

end NUMINAMATH_GPT_ratio_adult_women_to_men_event_l472_47257


namespace NUMINAMATH_GPT_smallest_value_not_defined_l472_47238

noncomputable def smallest_undefined_x : ℝ :=
  let a := 6
  let b := -37
  let c := 5
  let discriminant := b * b - 4 * a * c
  let sqrt_discriminant := Real.sqrt discriminant
  let x1 := (-b + sqrt_discriminant) / (2 * a)
  let x2 := (-b - sqrt_discriminant) / (2 * a)
  if x1 < x2 then x1 else x2

theorem smallest_value_not_defined :
  smallest_undefined_x = 0.1383 :=
by sorry

end NUMINAMATH_GPT_smallest_value_not_defined_l472_47238


namespace NUMINAMATH_GPT_power_equation_l472_47248

theorem power_equation (x a b : ℝ) (ha : 3^x = a) (hb : 5^x = b) : 45^x = a^2 * b :=
sorry

end NUMINAMATH_GPT_power_equation_l472_47248


namespace NUMINAMATH_GPT_maximum_special_points_l472_47282

theorem maximum_special_points (n : ℕ) (h : n = 11) : 
  ∃ p : ℕ, p = 91 := 
sorry

end NUMINAMATH_GPT_maximum_special_points_l472_47282


namespace NUMINAMATH_GPT_find_y_l472_47206

theorem find_y (x y : ℤ)
  (h1 : (100 + 200300 + x) / 3 = 250)
  (h2 : (300 + 150100 + x + y) / 4 = 200) :
  y = -4250 :=
sorry

end NUMINAMATH_GPT_find_y_l472_47206


namespace NUMINAMATH_GPT_cafeteria_extra_fruits_l472_47295

theorem cafeteria_extra_fruits (red_apples green_apples bananas oranges students : ℕ) (fruits_per_student : ℕ)
  (h1 : red_apples = 33)
  (h2 : green_apples = 23)
  (h3 : bananas = 17)
  (h4 : oranges = 12)
  (h5 : students = 21)
  (h6 : fruits_per_student = 2) :
  (red_apples + green_apples + bananas + oranges) - (students * fruits_per_student) = 43 :=
by
  sorry

end NUMINAMATH_GPT_cafeteria_extra_fruits_l472_47295


namespace NUMINAMATH_GPT_max_sqrt_expr_l472_47216

variable {x y z : ℝ}

noncomputable def f (x y z : ℝ) : ℝ := Real.sqrt x + Real.sqrt (2 * y) + Real.sqrt (3 * z)

theorem max_sqrt_expr (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) : 
  f x y z ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_max_sqrt_expr_l472_47216


namespace NUMINAMATH_GPT_asymptotes_and_foci_of_hyperbola_l472_47263

def hyperbola (x y : ℝ) : Prop := x^2 / 144 - y^2 / 81 = 1

theorem asymptotes_and_foci_of_hyperbola :
  (∀ x y : ℝ, hyperbola x y → y = x * (3 / 4) ∨ y = x * -(3 / 4)) ∧
  (∃ x y : ℝ, (x, y) = (15, 0) ∨ (x, y) = (-15, 0)) :=
by {
  -- prove these conditions here
  sorry 
}

end NUMINAMATH_GPT_asymptotes_and_foci_of_hyperbola_l472_47263


namespace NUMINAMATH_GPT_geometric_seq_common_ratio_l472_47283

theorem geometric_seq_common_ratio (a_n : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) 
  (hS3 : S 3 = a_n 1 * (1 - q ^ 3) / (1 - q))
  (hS2 : S 2 = a_n 1 * (1 - q ^ 2) / (1 - q))
  (h : S 3 + 3 * S 2 = 0) 
  (hq_not_one : q ≠ 1) :
  q = -2 :=
by sorry

end NUMINAMATH_GPT_geometric_seq_common_ratio_l472_47283


namespace NUMINAMATH_GPT_exists_points_with_small_distance_l472_47230

theorem exists_points_with_small_distance :
  ∃ A B : ℝ × ℝ, (A.2 = A.1^4) ∧ (B.2 = B.1^4 + B.1^2 + B.1 + 1) ∧ 
  (dist A B < 1 / 100) :=
by
  sorry

end NUMINAMATH_GPT_exists_points_with_small_distance_l472_47230


namespace NUMINAMATH_GPT_k_is_odd_l472_47252

theorem k_is_odd (m n k : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_pos_k : 0 < k) (h : 3 * m * k = (m + 3)^n + 1) : Odd k :=
by {
  sorry
}

end NUMINAMATH_GPT_k_is_odd_l472_47252


namespace NUMINAMATH_GPT_largest_among_five_numbers_l472_47269

theorem largest_among_five_numbers :
  let A := 98765 + (1 / 4321)
  let B := 98765 - (1 / 4321)
  let C := 98765 * (1 / 4321)
  let D := 98765 / (1 / 4321)
  let E := 98765.4321
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  let A := 98765 + (1 / 4321)
  let B := 98765 - (1 / 4321)
  let C := 98765 * (1 / 4321)
  let D := 98765 / (1 / 4321)
  let E := 98765.4321
  sorry

end NUMINAMATH_GPT_largest_among_five_numbers_l472_47269


namespace NUMINAMATH_GPT_number_of_men_in_first_group_l472_47255

-- Definitions based on the conditions provided
def work_done (men : ℕ) (days : ℕ) (work_rate : ℝ) : ℝ :=
  men * days * work_rate

-- Given conditions
def condition1 (M : ℕ) : Prop :=
  ∃ work_rate : ℝ, work_done M 12 work_rate = 66

def condition2 : Prop :=
  ∃ work_rate : ℝ, work_done 86 8 work_rate = 189.2

-- Proof goal
theorem number_of_men_in_first_group : 
  ∀ M : ℕ, condition1 M → condition2 → M = 57 := by
  sorry

end NUMINAMATH_GPT_number_of_men_in_first_group_l472_47255


namespace NUMINAMATH_GPT_squirrel_rise_per_circuit_l472_47228

theorem squirrel_rise_per_circuit
  (h_post_height : ℕ := 12)
  (h_circumference : ℕ := 3)
  (h_travel_distance : ℕ := 9) :
  (h_post_height / (h_travel_distance / h_circumference) = 4) :=
  sorry

end NUMINAMATH_GPT_squirrel_rise_per_circuit_l472_47228


namespace NUMINAMATH_GPT_largest_four_digit_mod_5_l472_47251

theorem largest_four_digit_mod_5 : ∃ (n : ℤ), n % 5 = 3 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℤ, m % 5 = 3 ∧ 1000 ≤ m ∧ m ≤ 9999 → m ≤ n :=
sorry

end NUMINAMATH_GPT_largest_four_digit_mod_5_l472_47251


namespace NUMINAMATH_GPT_total_birdseed_amount_l472_47268

-- Define the birdseed amounts in the boxes
def box1_amount : ℕ := 250
def box2_amount : ℕ := 275
def box3_amount : ℕ := 225
def box4_amount : ℕ := 300
def box5_amount : ℕ := 275
def box6_amount : ℕ := 200
def box7_amount : ℕ := 150
def box8_amount : ℕ := 180

-- Define the weekly consumption of each bird
def parrot_consumption : ℕ := 100
def cockatiel_consumption : ℕ := 50
def canary_consumption : ℕ := 25

-- Define a theorem to calculate the total birdseed that Leah has
theorem total_birdseed_amount : box1_amount + box2_amount + box3_amount + box4_amount + box5_amount + box6_amount + box7_amount + box8_amount = 1855 :=
by
  sorry

end NUMINAMATH_GPT_total_birdseed_amount_l472_47268


namespace NUMINAMATH_GPT_time_to_park_l472_47290

-- distance from house to market in miles
def d_market : ℝ := 5

-- distance from house to park in miles
def d_park : ℝ := 3

-- time to market in minutes
def t_market : ℝ := 30

-- assuming constant speed, calculate time to park
theorem time_to_park : (3 / 5) * 30 = 18 := by
  sorry

end NUMINAMATH_GPT_time_to_park_l472_47290


namespace NUMINAMATH_GPT_trapezoid_cd_length_l472_47239

noncomputable def proof_cd_length (AD BC CD : ℝ) (BD : ℝ) (angle_DBA angle_BDC : ℝ) (ratio_BC_AD : ℝ) : Prop :=
  AD > 0 ∧ BC > 0 ∧
  BD = 1 ∧
  angle_DBA = 23 ∧
  angle_BDC = 46 ∧
  ratio_BC_AD = 9 / 5 ∧
  AD / BC = 5 / 9 ∧
  CD = 4 / 5

theorem trapezoid_cd_length
  (AD BC CD : ℝ)
  (BD : ℝ := 1)
  (angle_DBA : ℝ := 23)
  (angle_BDC : ℝ := 46)
  (ratio_BC_AD : ℝ := 9 / 5)
  (h_conditions : proof_cd_length AD BC CD BD angle_DBA angle_BDC ratio_BC_AD) : CD = 4 / 5 :=
sorry

end NUMINAMATH_GPT_trapezoid_cd_length_l472_47239


namespace NUMINAMATH_GPT_find_c_l472_47278

theorem find_c (x y c : ℝ) (h1 : 7^(3 * x - 1) * 3^(4 * y - 3) = c^x * 27^y)
  (h2 : x + y = 4) : c = 49 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l472_47278


namespace NUMINAMATH_GPT_percentage_decrease_l472_47258

theorem percentage_decrease (original_price new_price : ℝ) (h₁ : original_price = 700) (h₂ : new_price = 532) : 
  ((original_price - new_price) / original_price) * 100 = 24 := by
  sorry

end NUMINAMATH_GPT_percentage_decrease_l472_47258


namespace NUMINAMATH_GPT_min_value_at_2_l472_47266

noncomputable def f (x : ℝ) : ℝ := (2 / (x^2)) + Real.log x

theorem min_value_at_2 : (∀ x ∈ Set.Ioi (0 : ℝ), f x ≥ f 2) ∧ (∃ x ∈ Set.Ioi (0 : ℝ), f x = f 2) :=
by
  sorry

end NUMINAMATH_GPT_min_value_at_2_l472_47266


namespace NUMINAMATH_GPT_jackson_fishes_per_day_l472_47210

def total_fishes : ℕ := 90
def jonah_per_day : ℕ := 4
def george_per_day : ℕ := 8
def competition_days : ℕ := 5

def jackson_per_day (J : ℕ) : Prop :=
  (total_fishes - (jonah_per_day * competition_days + george_per_day * competition_days)) / competition_days = J

theorem jackson_fishes_per_day : jackson_per_day 6 :=
  by
    sorry

end NUMINAMATH_GPT_jackson_fishes_per_day_l472_47210


namespace NUMINAMATH_GPT_weighted_valid_votes_l472_47292

theorem weighted_valid_votes :
  let total_votes := 10000
  let invalid_vote_rate := 0.25
  let valid_votes := total_votes * (1 - invalid_vote_rate)
  let v_b := (valid_votes - 2 * (valid_votes * 0.15 + valid_votes * 0.07) + valid_votes * 0.05) / 4
  let v_a := v_b + valid_votes * 0.15
  let v_c := v_a + valid_votes * 0.07
  let v_d := v_b - valid_votes * 0.05
  let weighted_votes_A := v_a * 3.0
  let weighted_votes_B := v_b * 2.5
  let weighted_votes_C := v_c * 2.75
  let weighted_votes_D := v_d * 2.25
  weighted_votes_A = 7200 ∧
  weighted_votes_B = 3187.5 ∧
  weighted_votes_C = 8043.75 ∧
  weighted_votes_D = 2025 :=
by
  sorry

end NUMINAMATH_GPT_weighted_valid_votes_l472_47292
