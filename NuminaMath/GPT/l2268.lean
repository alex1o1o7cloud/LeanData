import Mathlib

namespace NUMINAMATH_GPT_quadratic_root_property_l2268_226898

theorem quadratic_root_property (a x1 x2 : ℝ) 
  (h_eq : ∀ x, a * x^2 - (3 * a + 1) * x + 2 * (a + 1) = 0)
  (h_distinct : x1 ≠ x2)
  (h_relation : x1 - x1 * x2 + x2 = 1 - a) : a = -1 :=
sorry

end NUMINAMATH_GPT_quadratic_root_property_l2268_226898


namespace NUMINAMATH_GPT_power_function_alpha_l2268_226814

theorem power_function_alpha (α : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x^α) (point_condition : f 8 = 2) : 
  α = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_power_function_alpha_l2268_226814


namespace NUMINAMATH_GPT_floor_of_neg_seven_fourths_l2268_226820

theorem floor_of_neg_seven_fourths : 
  Int.floor (-7 / 4 : ℚ) = -2 := 
by sorry

end NUMINAMATH_GPT_floor_of_neg_seven_fourths_l2268_226820


namespace NUMINAMATH_GPT_largest_mersenne_prime_less_than_500_l2268_226817

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_mersenne_prime (p : ℕ) : Prop := ∃ n : ℕ, is_prime n ∧ p = 2^n - 1

theorem largest_mersenne_prime_less_than_500 : ∃ p : ℕ, is_mersenne_prime p ∧ p < 500 ∧ ∀ q : ℕ, is_mersenne_prime q ∧ q < 500 → q ≤ p :=
sorry

end NUMINAMATH_GPT_largest_mersenne_prime_less_than_500_l2268_226817


namespace NUMINAMATH_GPT_coordinates_of_B_l2268_226888

structure Point where
  x : Float
  y : Float

def symmetricWithRespectToY (A B : Point) : Prop :=
  B.x = -A.x ∧ B.y = A.y

theorem coordinates_of_B (A B : Point) 
  (hA : A.x = 2 ∧ A.y = -5)
  (h_sym : symmetricWithRespectToY A B) :
  B.x = -2 ∧ B.y = -5 :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_B_l2268_226888


namespace NUMINAMATH_GPT_fraction_simplification_l2268_226833

variable (a b x : ℝ)
variable (h1 : x = a / b)
variable (h2 : a ≠ b)
variable (h3 : b ≠ 0)
variable (h4 : a = b * x ^ 2)

theorem fraction_simplification : (a + b) / (a - b) = (x ^ 2 + 1) / (x ^ 2 - 1) := by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l2268_226833


namespace NUMINAMATH_GPT_math_problem_l2268_226882

theorem math_problem : (3 ^ 456) + (9 ^ 5 / 9 ^ 3) = 82 := 
by 
  sorry

end NUMINAMATH_GPT_math_problem_l2268_226882


namespace NUMINAMATH_GPT_max_sum_value_l2268_226806

noncomputable def max_sum (x y : ℝ) (h : 3 * (x^2 + y^2) = x - y) : ℝ :=
  x + y

theorem max_sum_value :
  ∃ x y : ℝ, ∃ h : 3 * (x^2 + y^2) = x - y, max_sum x y h = 1/3 :=
sorry

end NUMINAMATH_GPT_max_sum_value_l2268_226806


namespace NUMINAMATH_GPT_olympiad_problem_l2268_226895

variable (a b c d : ℕ)
variable (N : ℕ := a + b + c + d)

theorem olympiad_problem
  (h1 : (a + d) / (N:ℚ) = 0.5)
  (h2 : (b + d) / (N:ℚ) = 0.6)
  (h3 : (c + d) / (N:ℚ) = 0.7)
  : (d : ℚ) / N * 100 = 40 := by
  sorry

end NUMINAMATH_GPT_olympiad_problem_l2268_226895


namespace NUMINAMATH_GPT_f1_g1_eq_one_l2268_226879

-- Definitions of even and odd functions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def odd_function (g : ℝ → ℝ) : Prop := ∀ x, g x = -g (-x)

-- Given statement to be proved
theorem f1_g1_eq_one (f g : ℝ → ℝ) (h_even : even_function f) (h_odd : odd_function g)
    (h_diff : ∀ x, f x - g x = x^3 + x^2 + 1) : f 1 + g 1 = 1 :=
  sorry

end NUMINAMATH_GPT_f1_g1_eq_one_l2268_226879


namespace NUMINAMATH_GPT_initial_distances_l2268_226841

theorem initial_distances (x y : ℝ) 
  (h1: x^2 + y^2 = 400)
  (h2: (x - 6)^2 + (y - 8)^2 = 100) : 
  x = 12 ∧ y = 16 := 
by 
  sorry

end NUMINAMATH_GPT_initial_distances_l2268_226841


namespace NUMINAMATH_GPT_solve_for_y_l2268_226846

variable (x y z : ℝ)

theorem solve_for_y (h : 3 * x + 3 * y + 3 * z + 11 = 143) : y = 44 - x - z :=
by 
  sorry

end NUMINAMATH_GPT_solve_for_y_l2268_226846


namespace NUMINAMATH_GPT_initial_average_mark_l2268_226815

-- Define the initial conditions
def num_students : ℕ := 9
def excluded_students_avg : ℕ := 44
def remaining_students_avg : ℕ := 80

-- Define the variables for total marks we calculated in the solution
def total_marks_initial := num_students * (num_students * excluded_students_avg / 5 + remaining_students_avg / (num_students - 5) * (num_students - 5))

-- The theorem we need to prove:
theorem initial_average_mark :
  (num_students * (excluded_students_avg * 5 + remaining_students_avg * (num_students - 5))) / num_students = 60 := 
  by
  -- step-by-step solution proof could go here, but we use sorry as placeholder
  sorry

end NUMINAMATH_GPT_initial_average_mark_l2268_226815


namespace NUMINAMATH_GPT_geom_seq_a6_value_l2268_226863

variable {α : Type _} [LinearOrderedField α]

theorem geom_seq_a6_value (a : ℕ → α) (q : α) 
(h_geom : ∀ n, a (n + 1) = a n * q)
(h_cond : a 4 + a 8 = π) : 
a 6 * (a 2 + 2 * a 6 + a 10) = π^2 := by
  sorry

end NUMINAMATH_GPT_geom_seq_a6_value_l2268_226863


namespace NUMINAMATH_GPT_find_n_tan_l2268_226800

theorem find_n_tan (n : ℤ) (hn : -90 < n ∧ n < 90) (htan : Real.tan (n * Real.pi / 180) = Real.tan (312 * Real.pi / 180)) : 
  n = -48 := 
sorry

end NUMINAMATH_GPT_find_n_tan_l2268_226800


namespace NUMINAMATH_GPT_OJ_perpendicular_PQ_l2268_226805

noncomputable def quadrilateral (A B C D : Point) : Prop := sorry

noncomputable def inscribed (A B C D : Point) : Prop := sorry

noncomputable def circumscribed (A B C D : Point) : Prop := sorry

noncomputable def no_diameter (A B C D : Point) : Prop := sorry

noncomputable def intersection_of_external_bisectors (A B C D : Point) (P : Point) : Prop := sorry

noncomputable def incenter (A B C D J : Point) : Prop := sorry

noncomputable def circumcenter (A B C D O : Point) : Prop := sorry

noncomputable def PQ_perpendicular (O J P Q : Point) : Prop := sorry

theorem OJ_perpendicular_PQ (A B C D P Q J O : Point) :
  quadrilateral A B C D →
  inscribed A B C D →
  circumscribed A B C D →
  no_diameter A B C D →
  intersection_of_external_bisectors A B C D P →
  intersection_of_external_bisectors C D A B Q →
  incenter A B C D J →
  circumcenter A B C D O →
  PQ_perpendicular O J P Q :=
sorry

end NUMINAMATH_GPT_OJ_perpendicular_PQ_l2268_226805


namespace NUMINAMATH_GPT_find_a_evaluate_expr_l2268_226843

-- Given polynomials A and B
def A (a x y : ℝ) : ℝ := a * x^2 + 3 * x * y + 2 * |a| * x
def B (x y : ℝ) : ℝ := 2 * x^2 + 6 * x * y + 4 * x + y + 1

-- Statement part (1)
theorem find_a (a : ℝ) (x y : ℝ) (h : (2 * A a x y - B x y) = (2 * a - 2) * x^2 + (4 * |a| - 4) * x - y - 1) : a = -1 := 
  sorry

-- Expression for part (2)
def expr (a : ℝ) : ℝ := 3 * (-3 * a^2 - 2 * a) - (a^2 - 2 * (5 * a - 4 * a^2 + 1) - 2 * a)

-- Statement part (2)
theorem evaluate_expr : expr (-1) = -22 := 
  sorry

end NUMINAMATH_GPT_find_a_evaluate_expr_l2268_226843


namespace NUMINAMATH_GPT_alpha_beta_value_l2268_226872

variable (α β : ℝ)

def quadratic (x : ℝ) := x^2 + 2 * x - 2005

axiom roots_quadratic_eq : quadratic α = 0 ∧ quadratic β = 0

theorem alpha_beta_value :
  α^2 + 3 * α + β = 2003 :=
by sorry

end NUMINAMATH_GPT_alpha_beta_value_l2268_226872


namespace NUMINAMATH_GPT_time_to_drain_tank_due_to_leak_l2268_226865

noncomputable def timeToDrain (P L : ℝ) : ℝ := (1 : ℝ) / L

theorem time_to_drain_tank_due_to_leak (P L : ℝ)
  (hP : P = 0.5)
  (hL : P - L = 5/11) :
  timeToDrain P L = 22 :=
by
  -- to state what needs to be proved here
  sorry

end NUMINAMATH_GPT_time_to_drain_tank_due_to_leak_l2268_226865


namespace NUMINAMATH_GPT_circle_area_from_circumference_l2268_226868

theorem circle_area_from_circumference (C : ℝ) (A : ℝ) (hC : C = 36) (hCircumference : ∀ r, C = 2 * Real.pi * r) (hAreaFormula : ∀ r, A = Real.pi * r^2) :
  A = 324 / Real.pi :=
by
  sorry

end NUMINAMATH_GPT_circle_area_from_circumference_l2268_226868


namespace NUMINAMATH_GPT_find_x_l2268_226897

theorem find_x (x : ℝ) : (x / 4 * 5 + 10 - 12 = 48) → (x = 40) :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2268_226897


namespace NUMINAMATH_GPT_students_in_circle_l2268_226887

theorem students_in_circle (n : ℕ) (h1 : n > 6) (h2 : n > 16) (h3 : n / 2 = 10) : n + 2 = 22 := by
  sorry

end NUMINAMATH_GPT_students_in_circle_l2268_226887


namespace NUMINAMATH_GPT_rachel_plant_placement_l2268_226855

def num_ways_to_place_plants : ℕ :=
  let plants := ["basil", "basil", "aloe", "cactus"]
  let lamps := ["white", "white", "red", "red"]
  -- we need to compute the number of ways to place 4 plants under 4 lamps
  22

theorem rachel_plant_placement :
  num_ways_to_place_plants = 22 :=
by
  -- Proof omitted for brevity
  sorry

end NUMINAMATH_GPT_rachel_plant_placement_l2268_226855


namespace NUMINAMATH_GPT_blue_tint_percentage_in_new_mixture_l2268_226826

-- Define the conditions given in the problem
def original_volume : ℝ := 40
def blue_tint_percentage : ℝ := 0.20
def added_blue_tint_volume : ℝ := 8

-- Calculate the original blue tint volume
def original_blue_tint_volume := blue_tint_percentage * original_volume

-- Calculate the new blue tint volume after adding more blue tint
def new_blue_tint_volume := original_blue_tint_volume + added_blue_tint_volume

-- Calculate the new total volume of the mixture
def new_total_volume := original_volume + added_blue_tint_volume

-- Define the expected result in percentage
def expected_blue_tint_percentage : ℝ := 33.3333

-- Statement to prove
theorem blue_tint_percentage_in_new_mixture :
  (new_blue_tint_volume / new_total_volume) * 100 = expected_blue_tint_percentage :=
sorry

end NUMINAMATH_GPT_blue_tint_percentage_in_new_mixture_l2268_226826


namespace NUMINAMATH_GPT_negation_proof_l2268_226838

theorem negation_proof :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_proof_l2268_226838


namespace NUMINAMATH_GPT_polygon_sides_eq_four_l2268_226812

theorem polygon_sides_eq_four (n : ℕ)
  (h_interior : (n - 2) * 180 = 360)
  (h_exterior : ∀ (m : ℕ), m = n -> 360 = 360) :
  n = 4 :=
sorry

end NUMINAMATH_GPT_polygon_sides_eq_four_l2268_226812


namespace NUMINAMATH_GPT_fettuccine_to_penne_ratio_l2268_226876

theorem fettuccine_to_penne_ratio
  (num_surveyed : ℕ)
  (num_spaghetti : ℕ)
  (num_ravioli : ℕ)
  (num_fettuccine : ℕ)
  (num_penne : ℕ)
  (h_surveyed : num_surveyed = 800)
  (h_spaghetti : num_spaghetti = 300)
  (h_ravioli : num_ravioli = 200)
  (h_fettuccine : num_fettuccine = 150)
  (h_penne : num_penne = 150) :
  num_fettuccine / num_penne = 1 :=
by
  sorry

end NUMINAMATH_GPT_fettuccine_to_penne_ratio_l2268_226876


namespace NUMINAMATH_GPT_money_left_after_transactions_l2268_226844

-- Define the coin values and quantities
def dimes := 50
def quarters := 24
def nickels := 40
def pennies := 75

-- Define the item costs
def candy_bar_cost := 6 * 10 + 4 * 5 + 5
def lollipop_cost := 25 + 2 * 10 + 10 - 5 
def bag_of_chips_cost := 2 * 25 + 3 * 10 + 15
def bottle_of_soda_cost := 25 + 6 * 10 + 5 * 5 + 20 - 5

-- Define the number of items bought
def num_candy_bars := 6
def num_lollipops := 3
def num_bags_of_chips := 4
def num_bottles_of_soda := 2

-- Define the initial total money
def total_money := (dimes * 10) + (quarters * 25) + (nickels * 5) + (pennies)

-- Calculate the total cost of items
def total_cost := num_candy_bars * candy_bar_cost + num_lollipops * lollipop_cost + num_bags_of_chips * bag_of_chips_cost + num_bottles_of_soda * bottle_of_soda_cost

-- Calculate the money left after transactions
def money_left := total_money - total_cost

-- Theorem statement to prove
theorem money_left_after_transactions : money_left = 85 := by
  sorry

end NUMINAMATH_GPT_money_left_after_transactions_l2268_226844


namespace NUMINAMATH_GPT_sum_of_exponents_sqrt_l2268_226870

theorem sum_of_exponents_sqrt (a b c : ℕ) : 2 + 4 + 6 = 12 := by
  sorry

end NUMINAMATH_GPT_sum_of_exponents_sqrt_l2268_226870


namespace NUMINAMATH_GPT_quadratic_function_range_l2268_226881

-- Define the quadratic function and the domain
def quadratic_function (x : ℝ) : ℝ := -(x - 2)^2 + 1

-- State the proof problem
theorem quadratic_function_range : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 5 → -8 ≤ quadratic_function x ∧ quadratic_function x ≤ 1 := 
by 
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_quadratic_function_range_l2268_226881


namespace NUMINAMATH_GPT_paving_stone_width_l2268_226860

theorem paving_stone_width :
  let courtyard_length := 70
  let courtyard_width := 16.5
  let num_paving_stones := 231
  let paving_stone_length := 2.5
  let courtyard_area := courtyard_length * courtyard_width
  let total_area_covered := courtyard_area
  let paving_stone_width := total_area_covered / (paving_stone_length * num_paving_stones)
  paving_stone_width = 2 :=
by
  sorry

end NUMINAMATH_GPT_paving_stone_width_l2268_226860


namespace NUMINAMATH_GPT_find_x_y_z_l2268_226827

theorem find_x_y_z (x y z : ℝ) (h1 : 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z) (h2 : x * y * z = 10)
  (h3 : x ^ Real.log x * y ^ Real.log y * z ^ Real.log z = 10) :
  (x = 1 ∧ y = 1 ∧ z = 10) ∨ (x = 10 ∧ y = 1 ∧ z = 1) ∨ (x = 1 ∧ y = 10 ∧ z = 1) :=
sorry

end NUMINAMATH_GPT_find_x_y_z_l2268_226827


namespace NUMINAMATH_GPT_analyze_properties_l2268_226883

noncomputable def eq_condition (x a : ℝ) : Prop :=
x ≠ 0 ∧ a = (x - 1) / (x^2)

noncomputable def first_condition (x a : ℝ) : Prop :=
x⁻¹ + a * x = 1

noncomputable def second_condition (x a : ℝ) : Prop :=
x⁻¹ + a * x > 1

noncomputable def third_condition (x a : ℝ) : Prop :=
x⁻¹ + a * x < 1

theorem analyze_properties (x a : ℝ) (h1 : eq_condition x a):
(first_condition x a) ∧ ¬(second_condition x a) ∧ ¬(third_condition x a) :=
by
  sorry

end NUMINAMATH_GPT_analyze_properties_l2268_226883


namespace NUMINAMATH_GPT_all_equal_l2268_226866

theorem all_equal (a : Fin 100 → ℝ) 
  (h1 : a 0 - 3 * a 1 + 2 * a 2 ≥ 0)
  (h2 : a 1 - 3 * a 2 + 2 * a 3 ≥ 0)
  (h3 : a 2 - 3 * a 3 + 2 * a 4 ≥ 0)
  -- ...
  (h99: a 98 - 3 * a 99 + 2 * a 0 ≥ 0)
  (h100: a 99 - 3 * a 0 + 2 * a 1 ≥ 0) : 
    ∀ i : Fin 100, a i = a 0 := 
by 
  sorry

end NUMINAMATH_GPT_all_equal_l2268_226866


namespace NUMINAMATH_GPT_wendy_makeup_time_l2268_226899

theorem wendy_makeup_time :
  ∀ (num_products wait_time total_time makeup_time : ℕ),
    num_products = 5 →
    wait_time = 5 →
    total_time = 55 →
    makeup_time = total_time - (num_products - 1) * wait_time →
    makeup_time = 35 :=
by
  intro num_products wait_time total_time makeup_time h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end NUMINAMATH_GPT_wendy_makeup_time_l2268_226899


namespace NUMINAMATH_GPT_tablecloth_diameter_l2268_226831

theorem tablecloth_diameter (r : ℝ) (h : r = 5) : 2 * r = 10 :=
by
  simp [h]
  sorry

end NUMINAMATH_GPT_tablecloth_diameter_l2268_226831


namespace NUMINAMATH_GPT_inequality_preservation_l2268_226837

theorem inequality_preservation (a b : ℝ) (h : a > b) : (1/3 : ℝ) * a - 1 > (1/3 : ℝ) * b - 1 := 
by sorry

end NUMINAMATH_GPT_inequality_preservation_l2268_226837


namespace NUMINAMATH_GPT_transformed_quadratic_roots_l2268_226880

-- Definitions of the conditions
def quadratic_roots (a b : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + b * x + 3 = 0 → (x = -2) ∨ (x = 3)

-- Statement of the theorem
theorem transformed_quadratic_roots (a b : ℝ) :
  quadratic_roots a b →
  ∀ x : ℝ, a * (x + 2)^2 + b * (x + 2) + 3 = 0 → (x = -4) ∨ (x = 1) :=
sorry

end NUMINAMATH_GPT_transformed_quadratic_roots_l2268_226880


namespace NUMINAMATH_GPT_sum_of_midpoints_l2268_226839

theorem sum_of_midpoints (d e f : ℝ) (h : d + e + f = 15) :
  (d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 15 :=
by sorry

end NUMINAMATH_GPT_sum_of_midpoints_l2268_226839


namespace NUMINAMATH_GPT_optionD_is_not_linear_system_l2268_226891

-- Define the equations for each option
def eqA1 (x y : ℝ) : Prop := 3 * x + 2 * y = 10
def eqA2 (x y : ℝ) : Prop := 2 * x - 3 * y = 5

def eqB1 (x y : ℝ) : Prop := 3 * x + 5 * y = 1
def eqB2 (x y : ℝ) : Prop := 2 * x - y = 4

def eqC1 (x y : ℝ) : Prop := x + 5 * y = 1
def eqC2 (x y : ℝ) : Prop := x - 5 * y = 2

def eqD1 (x y : ℝ) : Prop := x - y = 1
def eqD2 (x y : ℝ) : Prop := y + 1 / x = 3

-- Define the property of a linear equation
def is_linear (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, eq x y → a * x + b * y = c

-- State the theorem
theorem optionD_is_not_linear_system : ¬ (is_linear eqD1 ∧ is_linear eqD2) :=
by
  sorry

end NUMINAMATH_GPT_optionD_is_not_linear_system_l2268_226891


namespace NUMINAMATH_GPT_find_smallest_b_l2268_226801

theorem find_smallest_b :
  ∃ b : ℕ, 
    (∀ r s : ℤ, r * s = 3960 → r + s ≠ b ∨ r + s > 0) ∧ 
    (∀ r s : ℤ, r * s = 3960 → (r + s < b → r + s ≤ 0)) ∧ 
    b = 126 :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_b_l2268_226801


namespace NUMINAMATH_GPT_rate_of_first_car_l2268_226857

theorem rate_of_first_car
  (r : ℕ) (h1 : 3 * r + 30 = 180) : r = 50 :=
sorry

end NUMINAMATH_GPT_rate_of_first_car_l2268_226857


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_problem_l2268_226862

theorem arithmetic_geometric_sequence_problem 
  (a : ℕ → ℚ)
  (b : ℕ → ℚ)
  (q : ℚ)
  (h1 : ∀ n m : ℕ, a (n + m) = a n * (q ^ m))
  (h2 : a 2 * a 3 * a 4 = 27 / 64)
  (h3 : q = 2)
  (h4 : ∃ d : ℚ, ∀ n : ℕ, b (n + 1) = b n + d)
  (h5 : b 7 = a 5) : 
  b 3 + b 11 = 6 := 
sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_problem_l2268_226862


namespace NUMINAMATH_GPT_train_length_l2268_226848

theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 36 * 1000 / 3600) (h2 : time = 14.998800095992321) :
  speed * time = 149.99 :=
by {
  sorry
}

end NUMINAMATH_GPT_train_length_l2268_226848


namespace NUMINAMATH_GPT_factor_expression_l2268_226804

theorem factor_expression (a b c : ℝ) : 
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 = (a - b) * (b - c) * (c - a) * (a * b^2 + a * c^2) :=
by 
  sorry

end NUMINAMATH_GPT_factor_expression_l2268_226804


namespace NUMINAMATH_GPT_part_A_part_C_part_D_l2268_226875

noncomputable def f : ℝ → ℝ := sorry -- define f with given properties

-- Given conditions
axiom mono_incr_on_neg1_0 : ∀ x y : ℝ, -1 ≤ x → x ≤ 0 → -1 ≤ y → y ≤ 0 → x < y → f x < f y
axiom symmetry_about_1 : ∀ x : ℝ, f (1 + x) = f (1 - x)
axiom symmetry_about_2_0 : ∀ x : ℝ, f (2 + x) = -f (2 - x)

-- Prove the statements
theorem part_A : f 0 = f (-2) := sorry
theorem part_C : ∀ x y : ℝ, 2 < x → x < 3 → 2 < y → y < 3 → x < y → f x > f y := sorry
theorem part_D : f 2021 > f 2022 ∧ f 2022 > f 2023 := sorry

end NUMINAMATH_GPT_part_A_part_C_part_D_l2268_226875


namespace NUMINAMATH_GPT_megan_initial_cupcakes_l2268_226884

noncomputable def initial_cupcakes (packages : Nat) (cupcakes_per_package : Nat) (cupcakes_eaten : Nat) : Nat :=
  packages * cupcakes_per_package + cupcakes_eaten

theorem megan_initial_cupcakes (packages : Nat) (cupcakes_per_package : Nat) (cupcakes_eaten : Nat) :
  packages = 4 → cupcakes_per_package = 7 → cupcakes_eaten = 43 →
  initial_cupcakes packages cupcakes_per_package cupcakes_eaten = 71 :=
by
  intros
  simp [initial_cupcakes]
  sorry

end NUMINAMATH_GPT_megan_initial_cupcakes_l2268_226884


namespace NUMINAMATH_GPT_max_take_home_pay_income_l2268_226858

theorem max_take_home_pay_income (x : ℤ) : 
  (1000 * 2 * 50) - 20 * 50^2 = 100000 := 
by 
  sorry

end NUMINAMATH_GPT_max_take_home_pay_income_l2268_226858


namespace NUMINAMATH_GPT_statement_A_statement_E_statement_B_false_statement_C_false_statement_D_false_l2268_226802

theorem statement_A : ∃ n : ℤ, 20 = 4 * n := by 
  sorry

theorem statement_E : ∃ n : ℤ, 180 = 9 * n := by 
  sorry

theorem statement_B_false : ¬ (19 ∣ 57) := by 
  sorry

theorem statement_C_false : 30 ∣ 90 := by 
  sorry

theorem statement_D_false : 17 ∣ 51 := by 
  sorry

end NUMINAMATH_GPT_statement_A_statement_E_statement_B_false_statement_C_false_statement_D_false_l2268_226802


namespace NUMINAMATH_GPT_product_of_integers_l2268_226853

-- Define the conditions as variables in Lean
variables {x y : ℤ}

-- State the main theorem/proof
theorem product_of_integers (h1 : x + y = 8) (h2 : x^2 + y^2 = 34) : x * y = 15 := by
  sorry

end NUMINAMATH_GPT_product_of_integers_l2268_226853


namespace NUMINAMATH_GPT_tree_current_height_l2268_226825

theorem tree_current_height 
  (growth_rate_per_week : ℕ)
  (weeks_per_month : ℕ)
  (total_height_after_4_months : ℕ) 
  (growth_rate_per_week_eq : growth_rate_per_week = 2)
  (weeks_per_month_eq : weeks_per_month = 4)
  (total_height_after_4_months_eq : total_height_after_4_months = 42) : 
  (∃ (current_height : ℕ), current_height = 10) :=
by
  sorry

end NUMINAMATH_GPT_tree_current_height_l2268_226825


namespace NUMINAMATH_GPT_tan_double_angle_l2268_226893

variable {α β : ℝ}

theorem tan_double_angle (h1 : Real.tan (α + β) = 3) (h2 : Real.tan (α - β) = 2) : Real.tan (2 * α) = -1 := by
  sorry

end NUMINAMATH_GPT_tan_double_angle_l2268_226893


namespace NUMINAMATH_GPT_length_of_platform_is_correct_l2268_226830

noncomputable def length_of_platform : ℝ :=
  let train_length := 200 -- in meters
  let train_speed := 80 * 1000 / 3600 -- kmph to m/s
  let crossing_time := 22 -- in seconds
  (train_speed * crossing_time) - train_length

theorem length_of_platform_is_correct :
  length_of_platform = 2600 / 9 :=
by 
  -- proof would go here
  sorry

end NUMINAMATH_GPT_length_of_platform_is_correct_l2268_226830


namespace NUMINAMATH_GPT_harry_carries_buckets_rounds_l2268_226811

noncomputable def george_rate := 2
noncomputable def total_buckets := 110
noncomputable def total_rounds := 22
noncomputable def harry_buckets_each_round := 3

theorem harry_carries_buckets_rounds :
  (george_rate * total_rounds + harry_buckets_each_round * total_rounds = total_buckets) :=
by sorry

end NUMINAMATH_GPT_harry_carries_buckets_rounds_l2268_226811


namespace NUMINAMATH_GPT_hex_B3F_to_decimal_l2268_226886

-- Define the hexadecimal values of B, 3, F
def hex_B : ℕ := 11
def hex_3 : ℕ := 3
def hex_F : ℕ := 15

-- Prove the conversion of B3F_{16} to a base 10 integer equals 2879
theorem hex_B3F_to_decimal : (hex_B * 16^2 + hex_3 * 16^1 + hex_F * 16^0) = 2879 := 
by 
  -- calculation details skipped
  sorry

end NUMINAMATH_GPT_hex_B3F_to_decimal_l2268_226886


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2268_226869

def setA : Set ℝ := {x : ℝ | |x| > 1}
def setB : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

theorem intersection_of_A_and_B : setA ∩ setB = {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2268_226869


namespace NUMINAMATH_GPT_find_q_zero_l2268_226829

theorem find_q_zero
  (p q r : ℝ → ℝ)  -- Define p, q, r as functions from ℝ to ℝ (since they are polynomials)
  (h1 : ∀ x, r x = p x * q x + 2)  -- Condition 1: r(x) = p(x) * q(x) + 2
  (h2 : p 0 = 6)                   -- Condition 2: constant term of p(x) is 6
  (h3 : r 0 = 5)                   -- Condition 3: constant term of r(x) is 5
  : q 0 = 1 / 2 :=                 -- Conclusion: q(0) = 1/2
sorry

end NUMINAMATH_GPT_find_q_zero_l2268_226829


namespace NUMINAMATH_GPT_gcd_lcm_sum_l2268_226819

theorem gcd_lcm_sum (a b : ℕ) (ha : a = 45) (hb : b = 4050) :
  Nat.gcd a b + Nat.lcm a b = 4095 := by
  sorry

end NUMINAMATH_GPT_gcd_lcm_sum_l2268_226819


namespace NUMINAMATH_GPT_add_second_largest_to_sum_l2268_226828

def is_valid_digit (d : ℕ) : Prop := d = 2 ∨ d = 5 ∨ d = 8

def form_number (d1 d2 d3 : ℕ) : ℕ := 100 * d1 + 10 * d2 + d3

def largest_number : ℕ := form_number 8 5 2
def smallest_number : ℕ := form_number 2 5 8
def second_largest_number : ℕ := form_number 8 2 5

theorem add_second_largest_to_sum : 
  second_largest_number + (largest_number + smallest_number) = 1935 := 
  sorry

end NUMINAMATH_GPT_add_second_largest_to_sum_l2268_226828


namespace NUMINAMATH_GPT_value_of_a_l2268_226813

theorem value_of_a (a x : ℝ) (h : (3 * x^2 + 2 * a * x = 0) → (x^3 + a * x^2 - (4 / 3) * a = 0)) :
  a = 0 ∨ a = 3 ∨ a = -3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l2268_226813


namespace NUMINAMATH_GPT_complete_square_l2268_226803

theorem complete_square {x : ℝ} :
  x^2 - 6 * x - 8 = 0 ↔ (x - 3)^2 = 17 :=
sorry

end NUMINAMATH_GPT_complete_square_l2268_226803


namespace NUMINAMATH_GPT_find_angle_A_l2268_226892

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) 
  (h1 : (Real.sin A + Real.sin B) * (a - b) = (Real.sin C - Real.sin B) * c) :
  A = Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_find_angle_A_l2268_226892


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_x_lt_3_not_necessary_condition_for_x_lt_3_l2268_226871

theorem sufficient_but_not_necessary_condition_for_x_lt_3 (x : ℝ) : |x - 1| < 2 → x < 3 :=
by {
  sorry
}

theorem not_necessary_condition_for_x_lt_3 (x : ℝ) : (x < 3) → ¬(-1 < x ∧ x < 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_x_lt_3_not_necessary_condition_for_x_lt_3_l2268_226871


namespace NUMINAMATH_GPT_value_of_a_l2268_226856

noncomputable def f (a : ℝ) (x : ℝ) := (x-1)*(x^2 - 3*x + a)

-- Define the condition that 1 is not a critical point
def not_critical (a : ℝ) : Prop := f a 1 ≠ 0

theorem value_of_a (a : ℝ) (h : not_critical a) : a = 2 := 
sorry

end NUMINAMATH_GPT_value_of_a_l2268_226856


namespace NUMINAMATH_GPT_sunglasses_price_l2268_226890

theorem sunglasses_price (P : ℝ) 
  (buy_cost_per_pair : ℝ := 26) 
  (pairs_sold : ℝ := 10) 
  (sign_cost : ℝ := 20) :
  (pairs_sold * P - pairs_sold * buy_cost_per_pair) / 2 = sign_cost →
  P = 30 := 
by
  sorry

end NUMINAMATH_GPT_sunglasses_price_l2268_226890


namespace NUMINAMATH_GPT_general_term_formula_l2268_226847

variable {a : ℕ → ℝ} -- Define the sequence as a function ℕ → ℝ

-- Conditions
axiom geom_seq (n : ℕ) (h : n ≥ 2): a (n + 1) = a 2 * (2 : ℝ) ^ (n - 1)
axiom a2_eq_2 : a 2 = 2
axiom a3_a4_cond : 2 * a 3 + a 4 = 16

theorem general_term_formula (n : ℕ) : a n = 2 ^ (n - 1) := by
  sorry -- Proof is not required

end NUMINAMATH_GPT_general_term_formula_l2268_226847


namespace NUMINAMATH_GPT_incorrect_relationship_f_pi4_f_pi_l2268_226852

open Real

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom f_derivative_exists : ∀ x : ℝ, DifferentiableAt ℝ f x
axiom f_derivative_lt_sin2x : ∀ x : ℝ, 0 < x → deriv f x < (sin x) ^ 2
axiom f_symmetric_property : ∀ x : ℝ, f (-x) + f x = 2 * (sin x) ^ 2

theorem incorrect_relationship_f_pi4_f_pi : ¬ (f (π / 4) < f π) :=
by sorry

end NUMINAMATH_GPT_incorrect_relationship_f_pi4_f_pi_l2268_226852


namespace NUMINAMATH_GPT_lucky_sum_mod_1000_l2268_226809

def is_lucky (n : ℕ) : Prop := ∀ d ∈ n.digits 10, d = 7

def first_twenty_lucky_numbers : List ℕ :=
  [7, 77] ++ List.replicate 18 777

theorem lucky_sum_mod_1000 :
  (first_twenty_lucky_numbers.sum % 1000) = 70 := 
sorry

end NUMINAMATH_GPT_lucky_sum_mod_1000_l2268_226809


namespace NUMINAMATH_GPT_highest_probability_ksi_expected_value_ksi_equals_l2268_226836

noncomputable def probability_ksi_equals (k : ℕ) : ℚ :=
  match k with
  | 2 => 9 / 64
  | 3 => 18 / 64
  | 4 => 21 / 64
  | 5 => 12 / 64
  | 6 => 4 / 64
  | _ => 0

noncomputable def expected_value_ksi : ℚ :=
  2 * (9 / 64) + 3 * (18 / 64) + 4 * (21 / 64) + 5 * (12 / 64) + 6 * (4 / 64)

theorem highest_probability_ksi :
  ∃ k : ℕ, (∀ m : ℕ, probability_ksi_equals k ≥ probability_ksi_equals m) ∧ k = 4 :=
by
  sorry

theorem expected_value_ksi_equals :
  expected_value_ksi = 15 / 4 :=
by
  sorry

end NUMINAMATH_GPT_highest_probability_ksi_expected_value_ksi_equals_l2268_226836


namespace NUMINAMATH_GPT_rational_coefficients_terms_count_l2268_226894

theorem rational_coefficients_terms_count : 
  (∃ s : Finset ℕ, ∀ k ∈ s, k % 20 = 0 ∧ k ≤ 725 ∧ s.card = 37) :=
by
  -- Translates to finding the set of all k satisfying the condition and 
  -- ensuring it has a cardinality of 37.
  sorry

end NUMINAMATH_GPT_rational_coefficients_terms_count_l2268_226894


namespace NUMINAMATH_GPT_perimeter_of_square_fence_l2268_226818

theorem perimeter_of_square_fence :
  ∀ (n : ℕ) (post_gap post_width : ℝ), 
  4 * n - 4 = 24 →
  post_gap = 6 →
  post_width = 5 / 12 →
  4 * ((n - 1) * post_gap + n * post_width) = 156 :=
by
  intros n post_gap post_width h1 h2 h3
  sorry

end NUMINAMATH_GPT_perimeter_of_square_fence_l2268_226818


namespace NUMINAMATH_GPT_half_hour_half_circle_half_hour_statement_is_true_l2268_226849

-- Definitions based on conditions
def half_circle_divisions : ℕ := 30
def small_divisions_per_minute : ℕ := 1
def total_small_divisions : ℕ := 60
def minutes_per_circle : ℕ := 60

-- Relation of small divisions and time taken
def time_taken_for_small_divisions (divs : ℕ) : ℕ := divs * small_divisions_per_minute

-- Theorem to prove the statement
theorem half_hour_half_circle : time_taken_for_small_divisions half_circle_divisions = 30 :=
by
  -- Given half circle covers 30 small divisions
  -- Each small division represents 1 minute
  -- Therefore, time taken for 30 divisions should be 30 minutes
  exact rfl

-- The final statement proving the truth of the condition
theorem half_hour_statement_is_true : 
  (time_taken_for_small_divisions half_circle_divisions = 30) → True :=
by
  intro h
  trivial

end NUMINAMATH_GPT_half_hour_half_circle_half_hour_statement_is_true_l2268_226849


namespace NUMINAMATH_GPT_min_bounces_l2268_226807

theorem min_bounces
  (h₀ : ℝ := 160)  -- initial height
  (r : ℝ := 3/4)  -- bounce ratio
  (final_h : ℝ := 20)  -- desired height
  (b : ℕ)  -- number of bounces
  : ∃ b, (h₀ * (r ^ b) < final_h ∧ ∀ b', b' < b → ¬(h₀ * (r ^ b') < final_h)) :=
sorry

end NUMINAMATH_GPT_min_bounces_l2268_226807


namespace NUMINAMATH_GPT_asparagus_cost_correct_l2268_226842

def cost_asparagus (total_start: Int) (total_left: Int) (cost_bananas: Int) (cost_pears: Int) (cost_chicken: Int) : Int := 
  total_start - total_left - cost_bananas - cost_pears - cost_chicken

theorem asparagus_cost_correct :
  cost_asparagus 55 28 8 2 11 = 6 :=
by
  sorry

end NUMINAMATH_GPT_asparagus_cost_correct_l2268_226842


namespace NUMINAMATH_GPT_quadratic_roots_value_l2268_226896

theorem quadratic_roots_value (d : ℝ) 
  (h : ∀ x : ℝ, x^2 + 7 * x + d = 0 → x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) : 
  d = 9.8 :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_roots_value_l2268_226896


namespace NUMINAMATH_GPT_integer_solutions_abs_inequality_l2268_226873

-- Define the condition as a predicate
def abs_inequality_condition (x : ℝ) : Prop := |x - 4| ≤ 3

-- State the proposition
theorem integer_solutions_abs_inequality : ∃ (n : ℕ), n = 7 ∧ ∀ (x : ℤ), abs_inequality_condition x → (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7) :=
sorry

end NUMINAMATH_GPT_integer_solutions_abs_inequality_l2268_226873


namespace NUMINAMATH_GPT_largest_possible_b_l2268_226832

theorem largest_possible_b (a b c : ℤ) (h1 : a > b) (h2 : b > c) (h3 : c > 2) (h4 : a * b * c = 360) : b = 10 :=
sorry

end NUMINAMATH_GPT_largest_possible_b_l2268_226832


namespace NUMINAMATH_GPT_distance_from_hut_to_station_l2268_226845

variable (t s : ℝ)

theorem distance_from_hut_to_station
  (h1 : s / 4 = t + 3 / 4)
  (h2 : s / 6 = t - 1 / 2) :
  s = 15 := by
  sorry

end NUMINAMATH_GPT_distance_from_hut_to_station_l2268_226845


namespace NUMINAMATH_GPT_sum_divisibility_l2268_226822

theorem sum_divisibility (a b : ℤ) (h : 6 * a + 11 * b ≡ 0 [ZMOD 31]) : a + 7 * b ≡ 0 [ZMOD 31] :=
sorry

end NUMINAMATH_GPT_sum_divisibility_l2268_226822


namespace NUMINAMATH_GPT_faster_speed_l2268_226808

theorem faster_speed (Speed1 : ℝ) (ExtraDistance : ℝ) (ActualDistance : ℝ) (v : ℝ) : 
  Speed1 = 10 ∧ ExtraDistance = 31 ∧ ActualDistance = 20.67 ∧ 
  (ActualDistance / Speed1 = (ActualDistance + ExtraDistance) / v) → 
  v = 25 :=
by
  sorry

end NUMINAMATH_GPT_faster_speed_l2268_226808


namespace NUMINAMATH_GPT_stratified_sampling_correct_l2268_226823

-- Definitions based on the conditions
def total_employees : ℕ := 300
def over_40 : ℕ := 50
def between_30_and_40 : ℕ := 150
def under_30 : ℕ := 100
def sample_size : ℕ := 30
def stratified_ratio : ℕ := 1 / 10  -- sample_size / total_employees

-- Function to compute the number of individuals sampled from each age group
def sampled_from_age_group (group_size : ℕ) : ℕ :=
  group_size * stratified_ratio

-- Mathematical properties to be proved
theorem stratified_sampling_correct :
  sampled_from_age_group over_40 = 5 ∧ 
  sampled_from_age_group between_30_and_40 = 15 ∧ 
  sampled_from_age_group under_30 = 10 := by
  sorry

end NUMINAMATH_GPT_stratified_sampling_correct_l2268_226823


namespace NUMINAMATH_GPT_isabella_purchases_l2268_226864

def isabella_items_total (alexis_pants alexis_dresses isabella_pants isabella_dresses : ℕ) : ℕ :=
  isabella_pants + isabella_dresses

theorem isabella_purchases
  (alexis_pants : ℕ) (alexis_dresses : ℕ)
  (h_pants : alexis_pants = 21)
  (h_dresses : alexis_dresses = 18)
  (h_ratio : ∀ (x : ℕ), alexis_pants = 3 * x → alexis_dresses = 3 * x):
  isabella_items_total (21 / 3) (18 / 3) = 13 :=
by
  sorry

end NUMINAMATH_GPT_isabella_purchases_l2268_226864


namespace NUMINAMATH_GPT_speed_of_current_l2268_226859

variable (m c : ℝ)

theorem speed_of_current (h1 : m + c = 15) (h2 : m - c = 10) : c = 2.5 :=
sorry

end NUMINAMATH_GPT_speed_of_current_l2268_226859


namespace NUMINAMATH_GPT_only_natural_number_dividing_power_diff_l2268_226810

theorem only_natural_number_dividing_power_diff (n : ℕ) (h : n ∣ (2^n - 1)) : n = 1 :=
by
  sorry

end NUMINAMATH_GPT_only_natural_number_dividing_power_diff_l2268_226810


namespace NUMINAMATH_GPT_divide_90_into_two_parts_l2268_226821

theorem divide_90_into_two_parts (x y : ℝ) (h : x + y = 90) 
  (cond : 0.4 * x = 0.3 * y + 15) : x = 60 ∨ y = 60 := 
by
  sorry

end NUMINAMATH_GPT_divide_90_into_two_parts_l2268_226821


namespace NUMINAMATH_GPT_original_price_of_cycle_l2268_226816

variable (P : ℝ)

theorem original_price_of_cycle (h : 0.92 * P = 1610) : P = 1750 :=
sorry

end NUMINAMATH_GPT_original_price_of_cycle_l2268_226816


namespace NUMINAMATH_GPT_staircase_steps_l2268_226885

theorem staircase_steps (x : ℕ) :
  x % 2 = 1 ∧
  x % 3 = 2 ∧
  x % 4 = 3 ∧
  x % 5 = 4 ∧
  x % 6 = 5 ∧
  x % 7 = 0 → 
  x ≡ 119 [MOD 420] :=
by
  sorry

end NUMINAMATH_GPT_staircase_steps_l2268_226885


namespace NUMINAMATH_GPT_rectangular_solid_edges_sum_l2268_226824

theorem rectangular_solid_edges_sum
  (b s : ℝ)
  (h_vol : (b / s) * b * (b * s) = 432)
  (h_sa : 2 * ((b ^ 2 / s) + b ^ 2 * s + b ^ 2) = 432)
  (h_gp : 0 < s ∧ s ≠ 1) :
  4 * (b / s + b + b * s) = 144 := 
by
  sorry

end NUMINAMATH_GPT_rectangular_solid_edges_sum_l2268_226824


namespace NUMINAMATH_GPT_domain_of_f_decreasing_on_interval_range_of_f_l2268_226854

noncomputable def f (x : ℝ) : ℝ := Real.log (3 + 2 * x - x^2) / Real.log 2

theorem domain_of_f :
  ∀ x : ℝ, (3 + 2 * x - x^2 > 0) ↔ (-1 < x ∧ x < 3) :=
by
  sorry

theorem decreasing_on_interval :
  ∀ (x₁ x₂ : ℝ), (1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3) →
  f x₂ < f x₁ :=
by
  sorry

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, -1 < x ∧ x < 3 ∧ y = f x) ↔ y ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_decreasing_on_interval_range_of_f_l2268_226854


namespace NUMINAMATH_GPT_number_of_dvds_remaining_l2268_226874

def initial_dvds : ℕ := 850

def week1_rented : ℕ := (initial_dvds * 25) / 100
def week1_sold : ℕ := 15
def remaining_after_week1 : ℕ := initial_dvds - week1_rented - week1_sold

def week2_rented : ℕ := (remaining_after_week1 * 35) / 100
def week2_sold : ℕ := 25
def remaining_after_week2 : ℕ := remaining_after_week1 - week2_rented - week2_sold

def week3_rented : ℕ := (remaining_after_week2 * 50) / 100
def week3_sold : ℕ := (remaining_after_week2 - week3_rented) * 5 / 100
def remaining_after_week3 : ℕ := remaining_after_week2 - week3_rented - week3_sold

theorem number_of_dvds_remaining : remaining_after_week3 = 181 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_number_of_dvds_remaining_l2268_226874


namespace NUMINAMATH_GPT_unique_solution_l2268_226878

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem unique_solution (x y : ℕ) :
  is_prime x →
  is_odd y →
  x^2 + y = 2007 →
  (x = 2 ∧ y = 2003) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_l2268_226878


namespace NUMINAMATH_GPT_compute_expression_l2268_226851

theorem compute_expression : 7^2 - 5 * 6 + 6^2 = 55 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l2268_226851


namespace NUMINAMATH_GPT_intersection_A_B_l2268_226840

open Set

def setA : Set ℕ := {x | x - 4 < 0}
def setB : Set ℕ := {0, 1, 3, 4}

theorem intersection_A_B : setA ∩ setB = {0, 1, 3} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2268_226840


namespace NUMINAMATH_GPT_unoccupied_volume_of_tank_l2268_226867

theorem unoccupied_volume_of_tank (length width height : ℝ) (num_marbles : ℕ) (marble_radius : ℝ) (fill_fraction : ℝ) :
    length = 12 → width = 12 → height = 15 → num_marbles = 5 → marble_radius = 1.5 → fill_fraction = 1/3 →
    (length * width * height * (1 - fill_fraction) - num_marbles * (4 / 3 * Real.pi * marble_radius^3) = 1440 - 22.5 * Real.pi) :=
by
  intros
  sorry

end NUMINAMATH_GPT_unoccupied_volume_of_tank_l2268_226867


namespace NUMINAMATH_GPT_shelves_per_case_l2268_226835

noncomputable section

-- Define the total number of ridges
def total_ridges : ℕ := 8640

-- Define the number of ridges per record
def ridges_per_record : ℕ := 60

-- Define the number of records per shelf when the shelf is 60% full
def records_per_shelf : ℕ := (60 * 20) / 100

-- Define the number of ridges per shelf
def ridges_per_shelf : ℕ := records_per_shelf * ridges_per_record

-- Given 4 cases, we need to determine the number of shelves per case
theorem shelves_per_case (cases shelves : ℕ) (h₁ : cases = 4) (h₂ : shelves * ridges_per_shelf = total_ridges) :
  shelves / cases = 3 := by
  sorry

end NUMINAMATH_GPT_shelves_per_case_l2268_226835


namespace NUMINAMATH_GPT_division_problem_l2268_226889

theorem division_problem (n : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h_div : divisor = 12) (h_quo : quotient = 9) (h_rem : remainder = 1) 
  (h_eq: n = divisor * quotient + remainder) : n = 109 :=
by
  sorry

end NUMINAMATH_GPT_division_problem_l2268_226889


namespace NUMINAMATH_GPT_bridge_length_sufficient_l2268_226861

structure Train :=
  (length : ℕ) -- length of the train in meters
  (speed : ℚ) -- speed of the train in km/hr

def speed_in_m_per_s (speed_in_km_per_hr : ℚ) : ℚ :=
  speed_in_km_per_hr * 1000 / 3600

noncomputable def length_of_bridge (train1 train2 : Train) : ℚ :=
  let train1_speed_m_per_s := speed_in_m_per_s train1.speed
  let train2_speed_m_per_s := speed_in_m_per_s train2.speed
  let relative_speed := train1_speed_m_per_s + train2_speed_m_per_s
  let total_length := train1.length + train2.length
  let time_to_pass := total_length / relative_speed
  let distance_train1 := train1_speed_m_per_s * time_to_pass
  let distance_train2 := train2_speed_m_per_s * time_to_pass
  distance_train1 + distance_train2

theorem bridge_length_sufficient (train1 train2 : Train) (h1 : train1.length = 200) (h2 : train1.speed = 60) (h3 : train2.length = 150) (h4 : train2.speed = 45) :
  length_of_bridge train1 train2 ≥ 350.04 :=
  by
  sorry

end NUMINAMATH_GPT_bridge_length_sufficient_l2268_226861


namespace NUMINAMATH_GPT_power_mod_444_444_l2268_226834

open Nat

theorem power_mod_444_444 : (444:ℕ)^444 % 13 = 1 := by
  sorry

end NUMINAMATH_GPT_power_mod_444_444_l2268_226834


namespace NUMINAMATH_GPT_a_n_formula_b_n_formula_S_n_formula_l2268_226850

noncomputable def a_n (n : ℕ) : ℕ := 3 * n
noncomputable def b_n (n : ℕ) : ℕ := 2^(n-1) + 3 * n
noncomputable def S_n (n : ℕ) : ℕ := 2^n - 1 + (3 * n^2 + 3 * n) / 2

theorem a_n_formula (n : ℕ) : a_n n = 3 * n := by
  unfold a_n
  rfl

theorem b_n_formula (n : ℕ) : b_n n = 2^(n-1) + 3 * n := by
  unfold b_n
  rfl

theorem S_n_formula (n : ℕ) : S_n n = 2^n - 1 + (3 * n^2 + 3 * n) / 2 := by
  unfold S_n
  rfl

end NUMINAMATH_GPT_a_n_formula_b_n_formula_S_n_formula_l2268_226850


namespace NUMINAMATH_GPT_find_n_l2268_226877

theorem find_n (n : ℕ) (h : n * n.factorial + 2 * n.factorial = 5040) : n = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_n_l2268_226877
