import Mathlib

namespace NUMINAMATH_CALUDE_pure_imaginary_product_l2904_290437

def complex (a b : ℝ) : ℂ := Complex.mk a b

theorem pure_imaginary_product (m : ℝ) : 
  let z₁ : ℂ := complex 3 2
  let z₂ : ℂ := complex 1 m
  (z₁ * z₂).re = 0 → m = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l2904_290437


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l2904_290410

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 5; -2, 9]

def inverse_A : Matrix (Fin 2) (Fin 2) ℚ := !![9/46, -5/46; 1/23, 2/23]

theorem matrix_inverse_proof :
  (Matrix.det A ≠ 0 ∧ A * inverse_A = 1 ∧ inverse_A * A = 1) ∨
  (Matrix.det A = 0 ∧ inverse_A = 0) := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l2904_290410


namespace NUMINAMATH_CALUDE_power_multiplication_calculate_expression_l2904_290472

theorem power_multiplication (a : ℕ) (m n : ℕ) :
  a * (a ^ n) = a ^ (n + 1) :=
by
  sorry

theorem calculate_expression : 
  3000 * (3000 ^ 1500) = 3000 ^ 1501 :=
by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_calculate_expression_l2904_290472


namespace NUMINAMATH_CALUDE_problem_statement_l2904_290419

theorem problem_statement (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) (hab : a + b = 1) :
  (∃ (min : ℝ), min = 2/3 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → a^2 + 2*b^2 ≥ min) ∧
  (a*x + b*y) * (a*y + b*x) ≥ x*y := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2904_290419


namespace NUMINAMATH_CALUDE_union_M_complement_N_equals_U_l2904_290452

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M as the domain of ln(1-x)
def M : Set ℝ := {x | x < 1}

-- Define set N as {x | x²-x < 0}
def N : Set ℝ := {x | x^2 - x < 0}

-- Theorem statement
theorem union_M_complement_N_equals_U : M ∪ (U \ N) = U := by sorry

end NUMINAMATH_CALUDE_union_M_complement_N_equals_U_l2904_290452


namespace NUMINAMATH_CALUDE_sphere_diameter_count_l2904_290491

theorem sphere_diameter_count (total_points : ℕ) (surface_percentage : ℚ) 
  (h1 : total_points = 39)
  (h2 : surface_percentage ≤ 72/100)
  : ∃ (surface_points : ℕ), 
    surface_points ≤ ⌊(surface_percentage * total_points)⌋ ∧ 
    (surface_points.choose 2) = 378 := by
  sorry

end NUMINAMATH_CALUDE_sphere_diameter_count_l2904_290491


namespace NUMINAMATH_CALUDE_circle_condition_l2904_290439

/-- 
Theorem: The equation x^2 + y^2 + x + 2my + m = 0 represents a circle if and only if m ≠ 1/2.
-/
theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + x + 2*m*y + m = 0) ↔ m ≠ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l2904_290439


namespace NUMINAMATH_CALUDE_train_length_proof_l2904_290425

theorem train_length_proof (bridge_length : ℝ) (bridge_time : ℝ) (post_time : ℝ) 
  (h1 : bridge_length = 800)
  (h2 : bridge_time = 45)
  (h3 : post_time = 15) :
  ∃ train_length : ℝ, train_length = 400 ∧ 
  train_length / post_time = (train_length + bridge_length) / bridge_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_proof_l2904_290425


namespace NUMINAMATH_CALUDE_B_power_6_l2904_290451

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, -3; 4, 5]

theorem B_power_6 : 
  B^6 = 1715 • B - 16184 • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_B_power_6_l2904_290451


namespace NUMINAMATH_CALUDE_initial_sets_count_l2904_290499

/-- The number of letters available (A through J) -/
def num_letters : ℕ := 10

/-- The number of letters in each set of initials -/
def set_size : ℕ := 3

/-- The number of arrangements for three letters where two are identical -/
def repeated_letter_arrangements : ℕ := 3

/-- The number of different three-letter sets of initials possible using letters A through J, 
    where one letter can appear twice and the third must be different -/
theorem initial_sets_count : 
  num_letters * (num_letters - 1) * repeated_letter_arrangements = 270 := by
  sorry

end NUMINAMATH_CALUDE_initial_sets_count_l2904_290499


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2904_290492

def P : Set ℝ := {1, 3, 5, 7}
def Q : Set ℝ := {x | 2 * x - 1 > 5}

theorem intersection_of_P_and_Q : P ∩ Q = {5, 7} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2904_290492


namespace NUMINAMATH_CALUDE_locus_of_point_in_cube_l2904_290478

/-- The locus of a point M in a unit cube, where the sum of squares of distances 
    from M to the faces of the cube is constant, is a sphere centered at (1/2, 1/2, 1/2). -/
theorem locus_of_point_in_cube (x y z : ℝ) (k : ℝ) : 
  (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) →
  x^2 + (1 - x)^2 + y^2 + (1 - y)^2 + z^2 + (1 - z)^2 = k →
  ∃ r : ℝ, (x - 1/2)^2 + (y - 1/2)^2 + (z - 1/2)^2 = r^2 :=
by sorry


end NUMINAMATH_CALUDE_locus_of_point_in_cube_l2904_290478


namespace NUMINAMATH_CALUDE_horatio_sonnets_count_l2904_290400

/-- Represents the number of sonnets Horatio wrote -/
def total_sonnets : ℕ := 12

/-- Represents the number of lines in each sonnet -/
def lines_per_sonnet : ℕ := 14

/-- Represents the number of sonnets Horatio's lady fair heard -/
def sonnets_heard : ℕ := 7

/-- Represents the number of romantic lines that were never heard -/
def unheard_lines : ℕ := 70

/-- Theorem stating that the total number of sonnets Horatio wrote is correct -/
theorem horatio_sonnets_count :
  total_sonnets = sonnets_heard + (unheard_lines / lines_per_sonnet) := by
  sorry

end NUMINAMATH_CALUDE_horatio_sonnets_count_l2904_290400


namespace NUMINAMATH_CALUDE_print_shop_charge_difference_l2904_290434

/-- The charge per color copy at print shop X -/
def charge_x : ℚ := 125/100

/-- The charge per color copy at print shop Y -/
def charge_y : ℚ := 275/100

/-- The number of color copies -/
def num_copies : ℕ := 40

/-- The difference in charges between print shop Y and X for num_copies color copies -/
def charge_difference : ℚ := num_copies * charge_y - num_copies * charge_x

theorem print_shop_charge_difference : charge_difference = 60 := by
  sorry

end NUMINAMATH_CALUDE_print_shop_charge_difference_l2904_290434


namespace NUMINAMATH_CALUDE_f_monotone_increasing_l2904_290405

def f (x : ℝ) : ℝ := 3 * x + 1

theorem f_monotone_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l2904_290405


namespace NUMINAMATH_CALUDE_last_digit_sum_l2904_290464

theorem last_digit_sum (n : ℕ) : (3^1991 + 1991^3) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_sum_l2904_290464


namespace NUMINAMATH_CALUDE_good_set_properties_l2904_290417

def GoodSet (s : Set ℝ) : Prop :=
  ∀ a ∈ s, (8 - a) ∈ s

theorem good_set_properties :
  (¬ GoodSet {1, 2}) ∧
  (GoodSet {1, 4, 7}) ∧
  (GoodSet {4}) ∧
  (GoodSet {3, 4, 5}) ∧
  (GoodSet {2, 6}) ∧
  (GoodSet {1, 2, 4, 6, 7}) ∧
  (GoodSet {0, 8}) :=
by sorry

end NUMINAMATH_CALUDE_good_set_properties_l2904_290417


namespace NUMINAMATH_CALUDE_third_card_value_l2904_290428

theorem third_card_value (a b c : ℕ) 
  (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h2 : 1 ≤ a ∧ a ≤ 13)
  (h3 : 1 ≤ b ∧ b ≤ 13)
  (h4 : 1 ≤ c ∧ c ≤ 13)
  (h5 : a + b = 25)
  (h6 : b + c = 13) :
  c = 1 := by
sorry

end NUMINAMATH_CALUDE_third_card_value_l2904_290428


namespace NUMINAMATH_CALUDE_melanie_dimes_and_choiceland_coins_l2904_290479

/-- Proves the number of dimes Melanie has and their value in ChoiceLand coins -/
theorem melanie_dimes_and_choiceland_coins 
  (initial_dimes : ℕ) 
  (dad_dimes : ℕ) 
  (mom_dimes : ℕ) 
  (exchange_rate : ℚ) 
  (h1 : initial_dimes = 7)
  (h2 : dad_dimes = 8)
  (h3 : mom_dimes = 4)
  (h4 : exchange_rate = 5/2) : 
  (initial_dimes + dad_dimes + mom_dimes = 19) ∧ 
  ((initial_dimes + dad_dimes + mom_dimes : ℚ) * exchange_rate = 95/2) := by
sorry

end NUMINAMATH_CALUDE_melanie_dimes_and_choiceland_coins_l2904_290479


namespace NUMINAMATH_CALUDE_square_value_l2904_290438

theorem square_value (a b : ℝ) (h : ∃ square, square * (3 * a * b) = 3 * a^2 * b) : 
  ∃ square, square = a := by sorry

end NUMINAMATH_CALUDE_square_value_l2904_290438


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_first_ten_terms_sum_l2904_290454

theorem arithmetic_sequence_sum : ℤ → ℤ → ℕ → ℤ
  | a, l, n => n * (a + l) / 2

theorem first_ten_terms_sum (a l : ℤ) (n : ℕ) (h1 : a = -5) (h2 : l = 40) (h3 : n = 10) :
  arithmetic_sequence_sum a l n = 175 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_first_ten_terms_sum_l2904_290454


namespace NUMINAMATH_CALUDE_johns_age_l2904_290462

theorem johns_age (john_age dad_age : ℕ) 
  (h1 : john_age = dad_age - 24)
  (h2 : john_age + dad_age = 68) : 
  john_age = 22 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_l2904_290462


namespace NUMINAMATH_CALUDE_rectangle_x_value_l2904_290489

/-- A rectangular figure with specified segment lengths -/
structure RectangularFigure where
  top_segment1 : ℝ
  top_segment2 : ℝ
  top_segment3 : ℝ
  bottom_segment1 : ℝ
  bottom_segment2 : ℝ
  bottom_segment3 : ℝ

/-- The property that the total length of top and bottom sides are equal -/
def is_valid_rectangle (r : RectangularFigure) : Prop :=
  r.top_segment1 + r.top_segment2 + r.top_segment3 = r.bottom_segment1 + r.bottom_segment2 + r.bottom_segment3

/-- The theorem stating that X must be 6 for the given rectangular figure -/
theorem rectangle_x_value :
  ∀ (x : ℝ),
  is_valid_rectangle ⟨3, 2, x, 4, 2, 5⟩ → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_x_value_l2904_290489


namespace NUMINAMATH_CALUDE_simple_sampling_methods_correct_l2904_290497

/-- The set of methods for implementing simple sampling -/
def SimpleSamplingMethods : Set String :=
  {"Lottery method", "Random number table method"}

/-- Theorem stating that the set of methods for implementing simple sampling
    contains exactly the lottery method and random number table method -/
theorem simple_sampling_methods_correct :
  SimpleSamplingMethods = {"Lottery method", "Random number table method"} := by
  sorry

end NUMINAMATH_CALUDE_simple_sampling_methods_correct_l2904_290497


namespace NUMINAMATH_CALUDE_unknown_number_proof_l2904_290433

theorem unknown_number_proof (x : ℝ) : 
  (10 + 30 + 50) / 3 = (20 + x + 6) / 3 + 8 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l2904_290433


namespace NUMINAMATH_CALUDE_chromium_percentage_proof_l2904_290480

/-- The percentage of chromium in the first alloy -/
def chromium_percentage_1 : ℝ := 10

/-- The percentage of chromium in the second alloy -/
def chromium_percentage_2 : ℝ := 8

/-- The weight of the first alloy in kg -/
def weight_1 : ℝ := 15

/-- The weight of the second alloy in kg -/
def weight_2 : ℝ := 35

/-- The percentage of chromium in the new alloy -/
def chromium_percentage_new : ℝ := 8.6

/-- The total weight of the new alloy in kg -/
def total_weight : ℝ := weight_1 + weight_2

theorem chromium_percentage_proof :
  (chromium_percentage_1 / 100) * weight_1 + (chromium_percentage_2 / 100) * weight_2 =
  (chromium_percentage_new / 100) * total_weight :=
by sorry

end NUMINAMATH_CALUDE_chromium_percentage_proof_l2904_290480


namespace NUMINAMATH_CALUDE_right_triangle_sides_l2904_290404

theorem right_triangle_sides : ∀ (a b c : ℝ), 
  a > 0 → b > 0 → c > 0 →
  a = 5 → b = 12 → c = 13 →
  a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l2904_290404


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2904_290493

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2904_290493


namespace NUMINAMATH_CALUDE_maximize_x_cubed_y_fourth_l2904_290485

theorem maximize_x_cubed_y_fourth (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2 * y = 35) :
  x^3 * y^4 ≤ 21^3 * 7^4 ∧ 
  (x^3 * y^4 = 21^3 * 7^4 ↔ x = 21 ∧ y = 7) :=
sorry

end NUMINAMATH_CALUDE_maximize_x_cubed_y_fourth_l2904_290485


namespace NUMINAMATH_CALUDE_identity_function_unique_l2904_290494

def C : ℕ := 2022^2022

theorem identity_function_unique :
  ∀ f : ℕ → ℕ,
  (∀ x y : ℕ, x > 0 → y > 0 → 
    ∃ k : ℕ, k > 0 ∧ k ≤ C ∧ f (x + y) = f x + k * f y) →
  f = id :=
by sorry

end NUMINAMATH_CALUDE_identity_function_unique_l2904_290494


namespace NUMINAMATH_CALUDE_angle_B_measure_l2904_290498

/-- In a triangle ABC, given that the measures of angles A, B, C form a geometric progression
    and b^2 - a^2 = a*c, prove that the measure of angle B is 2π/7 -/
theorem angle_B_measure (A B C : ℝ) (a b c : ℝ) :
  A > 0 → B > 0 → C > 0 →
  a > 0 → b > 0 → c > 0 →
  A + B + C = π →
  ∃ (q : ℝ), q > 0 ∧ B = q * A ∧ C = q * B →
  b^2 - a^2 = a * c →
  B = 2 * π / 7 := by
sorry

end NUMINAMATH_CALUDE_angle_B_measure_l2904_290498


namespace NUMINAMATH_CALUDE_hours_per_day_is_five_l2904_290422

/-- The number of hours worked per day by the first group of women -/
def hours_per_day : ℝ := 5

/-- The number of women in the first group -/
def women_group1 : ℕ := 6

/-- The number of days worked by the first group -/
def days_group1 : ℕ := 8

/-- The units of work completed by the first group -/
def work_units_group1 : ℕ := 75

/-- The number of women in the second group -/
def women_group2 : ℕ := 4

/-- The number of days worked by the second group -/
def days_group2 : ℕ := 3

/-- The units of work completed by the second group -/
def work_units_group2 : ℕ := 30

/-- The number of hours worked per day by the second group -/
def hours_per_day_group2 : ℕ := 8

/-- The proposition that the amount of work done is proportional to the number of woman-hours worked -/
axiom work_proportional_to_hours : 
  (women_group1 * days_group1 * hours_per_day) / work_units_group1 = 
  (women_group2 * days_group2 * hours_per_day_group2) / work_units_group2

theorem hours_per_day_is_five : hours_per_day = 5 := by
  sorry

end NUMINAMATH_CALUDE_hours_per_day_is_five_l2904_290422


namespace NUMINAMATH_CALUDE_square_triangle_equal_area_l2904_290468

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (x : ℝ) : 
  square_perimeter = 64 →
  triangle_height = 36 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * x * triangle_height →
  x = 128 / 9 := by
sorry

end NUMINAMATH_CALUDE_square_triangle_equal_area_l2904_290468


namespace NUMINAMATH_CALUDE_equivalent_operations_l2904_290453

theorem equivalent_operations (x : ℝ) : x * (4/5) / (2/7) = x * (7/5) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_operations_l2904_290453


namespace NUMINAMATH_CALUDE_largest_sum_is_994_l2904_290456

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the sum of XXX + YY + Z -/
def sum (X Y Z : Digit) : ℕ := 111 * X.val + 11 * Y.val + Z.val

theorem largest_sum_is_994 :
  ∃ (X Y Z : Digit), X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z ∧
    sum X Y Z ≤ 999 ∧
    (∀ (A B C : Digit), A ≠ B ∧ A ≠ C ∧ B ≠ C → sum A B C ≤ sum X Y Z) ∧
    sum X Y Z = 994 ∧
    X = Y ∧ Y ≠ Z :=
by sorry

end NUMINAMATH_CALUDE_largest_sum_is_994_l2904_290456


namespace NUMINAMATH_CALUDE_division_problem_l2904_290496

theorem division_problem (x : ℤ) : (64 / x = 4) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2904_290496


namespace NUMINAMATH_CALUDE_number_of_refills_l2904_290488

def total_spent : ℕ := 63
def cost_per_refill : ℕ := 21

theorem number_of_refills : total_spent / cost_per_refill = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_of_refills_l2904_290488


namespace NUMINAMATH_CALUDE_no_single_digit_divisor_l2904_290407

theorem no_single_digit_divisor (n : ℤ) (d : ℤ) :
  1 < d → d < 10 → ¬(∃ k : ℤ, 2 * n^2 - 31 = d * k) := by
  sorry

end NUMINAMATH_CALUDE_no_single_digit_divisor_l2904_290407


namespace NUMINAMATH_CALUDE_mod_twelve_six_eight_l2904_290447

theorem mod_twelve_six_eight (m : ℕ) : 12^6 ≡ m [ZMOD 8] → 0 ≤ m → m < 8 → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_mod_twelve_six_eight_l2904_290447


namespace NUMINAMATH_CALUDE_least_frood_count_l2904_290418

/-- The function representing points earned by dropping n froods -/
def drop_points (n : ℕ) : ℚ := n * (n + 1) / 2

/-- The function representing points earned by eating n froods -/
def eat_points (n : ℕ) : ℚ := 20 * n

/-- The theorem stating that 40 is the least positive integer for which
    dropping froods earns more points than eating them -/
theorem least_frood_count : ∀ n : ℕ, n > 0 → (drop_points n > eat_points n ↔ n ≥ 40) := by
  sorry

end NUMINAMATH_CALUDE_least_frood_count_l2904_290418


namespace NUMINAMATH_CALUDE_abs_sum_lower_bound_l2904_290416

theorem abs_sum_lower_bound :
  (∀ x : ℝ, |x - 1| + |x + 2| ≥ 3) ∧
  (∀ ε > 0, ∃ x : ℝ, |x - 1| + |x + 2| < 3 + ε) :=
by sorry

end NUMINAMATH_CALUDE_abs_sum_lower_bound_l2904_290416


namespace NUMINAMATH_CALUDE_salary_change_percentage_salary_loss_percentage_l2904_290470

theorem salary_change_percentage (original : ℝ) (original_positive : 0 < original) :
  let decreased := original * (1 - 0.6)
  let increased := decreased * (1 + 0.6)
  increased = original * 0.64 :=
by
  sorry

theorem salary_loss_percentage (original : ℝ) (original_positive : 0 < original) :
  let decreased := original * (1 - 0.6)
  let increased := decreased * (1 + 0.6)
  (original - increased) / original = 0.36 :=
by
  sorry

end NUMINAMATH_CALUDE_salary_change_percentage_salary_loss_percentage_l2904_290470


namespace NUMINAMATH_CALUDE_triangle_count_is_sixteen_l2904_290457

/-- Represents a rectangle with diagonals and internal rectangle --/
structure ConfiguredRectangle where
  vertices : Fin 4 → Point
  diagonals : List (Point × Point)
  midpoints : Fin 4 → Point
  internal_rectangle : List (Point × Point)

/-- Counts the number of triangles in the configured rectangle --/
def count_triangles (rect : ConfiguredRectangle) : ℕ :=
  sorry

/-- Theorem stating that the number of triangles is 16 --/
theorem triangle_count_is_sixteen (rect : ConfiguredRectangle) : 
  count_triangles rect = 16 :=
sorry

end NUMINAMATH_CALUDE_triangle_count_is_sixteen_l2904_290457


namespace NUMINAMATH_CALUDE_paper_side_length_l2904_290473

theorem paper_side_length (cube_side: ℝ) (num_pieces: ℕ) (paper_side: ℝ)
  (h1: cube_side = 12)
  (h2: num_pieces = 54)
  (h3: (6 * cube_side^2) = (num_pieces * paper_side^2)) :
  paper_side = 4 := by
  sorry

end NUMINAMATH_CALUDE_paper_side_length_l2904_290473


namespace NUMINAMATH_CALUDE_original_pencils_count_l2904_290475

/-- The number of pencils Mike placed in the drawer -/
def pencils_added : ℕ := 30

/-- The total number of pencils now in the drawer -/
def total_pencils : ℕ := 71

/-- The original number of pencils in the drawer -/
def original_pencils : ℕ := total_pencils - pencils_added

theorem original_pencils_count : original_pencils = 41 := by
  sorry

end NUMINAMATH_CALUDE_original_pencils_count_l2904_290475


namespace NUMINAMATH_CALUDE_extremum_implies_a_in_open_interval_l2904_290465

open Set
open Function
open Real

/-- A function f has exactly one extremum point in an interval (a, b) if there exists
    exactly one point c in (a, b) where f'(c) = 0. -/
def has_exactly_one_extremum (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! c, a < c ∧ c < b ∧ deriv f c = 0

/-- The cubic function f(x) = x^3 + x^2 - ax - 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + x^2 - a*x - 4

theorem extremum_implies_a_in_open_interval :
  ∀ a : ℝ, has_exactly_one_extremum (f a) (-1) 1 → a ∈ Ioo 1 5 :=
sorry

end NUMINAMATH_CALUDE_extremum_implies_a_in_open_interval_l2904_290465


namespace NUMINAMATH_CALUDE_f_less_than_4_iff_in_M_abs_sum_less_than_abs_product_plus_4_l2904_290412

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Define the set M
def M : Set ℝ := Set.Ioo (-2) 2

-- Statement 1
theorem f_less_than_4_iff_in_M : ∀ x : ℝ, f x < 4 ↔ x ∈ M := by sorry

-- Statement 2
theorem abs_sum_less_than_abs_product_plus_4 : 
  ∀ x y : ℝ, x ∈ M → y ∈ M → |x + y| < |x * y / 2 + 2| := by sorry

end NUMINAMATH_CALUDE_f_less_than_4_iff_in_M_abs_sum_less_than_abs_product_plus_4_l2904_290412


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l2904_290482

theorem bernoulli_inequality (c x : ℝ) (p : ℤ) 
  (hc : c > 0) (hp : p > 1) (hx1 : x > -1) (hx2 : x ≠ 0) : 
  (1 + x)^p > 1 + p * x := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l2904_290482


namespace NUMINAMATH_CALUDE_baseball_hits_percentage_l2904_290460

/-- 
Given a baseball player's hit statistics for a season:
- Total hits: 50
- Home runs: 2
- Triples: 3
- Doubles: 8

This theorem proves that the percentage of hits that were singles is 74%.
-/
theorem baseball_hits_percentage (total_hits home_runs triples doubles : ℕ) 
  (h1 : total_hits = 50)
  (h2 : home_runs = 2)
  (h3 : triples = 3)
  (h4 : doubles = 8) :
  (total_hits - (home_runs + triples + doubles)) / total_hits * 100 = 74 := by
  sorry

#eval (50 - (2 + 3 + 8)) / 50 * 100  -- Should output 74

end NUMINAMATH_CALUDE_baseball_hits_percentage_l2904_290460


namespace NUMINAMATH_CALUDE_not_isosceles_l2904_290436

/-- A set of three distinct real numbers that can form the sides of a triangle -/
structure TriangleSet where
  a : ℝ
  b : ℝ
  c : ℝ
  distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The triangle formed by a TriangleSet cannot be isosceles -/
theorem not_isosceles (S : TriangleSet) : ¬(S.a = S.b ∨ S.b = S.c ∨ S.c = S.a) :=
sorry

end NUMINAMATH_CALUDE_not_isosceles_l2904_290436


namespace NUMINAMATH_CALUDE_opposite_is_five_l2904_290458

theorem opposite_is_five (x : ℝ) : -x = 5 → x = -5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_is_five_l2904_290458


namespace NUMINAMATH_CALUDE_room_population_lower_limit_l2904_290440

theorem room_population_lower_limit (total : ℕ) (under_21 : ℕ) (over_65 : ℕ) : 
  under_21 = 30 →
  under_21 = (3 : ℚ) / 7 * total →
  over_65 = (5 : ℚ) / 10 * total →
  ∃ (upper : ℕ), total ∈ Set.Icc total upper →
  70 ≤ total :=
by sorry

end NUMINAMATH_CALUDE_room_population_lower_limit_l2904_290440


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_zero_l2904_290481

-- Define the functions f and g
def f (A B x : ℝ) : ℝ := A * x^2 + B * x + 1
def g (A B x : ℝ) : ℝ := B * x^2 + A * x + 1

-- State the theorem
theorem sum_of_coefficients_is_zero (A B : ℝ) :
  A ≠ B →
  (∀ x, f A B (g A B x) - g A B (f A B x) = x^4 + 5*x^3 + x^2 - 4*x) →
  A + B = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_zero_l2904_290481


namespace NUMINAMATH_CALUDE_function_property_implies_odd_l2904_290427

theorem function_property_implies_odd (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f y = f (x + y)) : 
  ∀ x : ℝ, f (-x) = -f x := by
sorry

end NUMINAMATH_CALUDE_function_property_implies_odd_l2904_290427


namespace NUMINAMATH_CALUDE_decimal_multiplication_l2904_290423

theorem decimal_multiplication (h : 28 * 15 = 420) :
  (2.8 * 1.5 = 4.2) ∧ (0.28 * 1.5 = 42) ∧ (0.028 * 0.15 = 0.0042) := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l2904_290423


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2904_290443

/-- The standard equation of a hyperbola with one focus at (2,0) and an asymptote
    with inclination angle of 60° is x^2 - (y^2/3) = 1 -/
theorem hyperbola_equation (C : Set (ℝ × ℝ)) (F : ℝ × ℝ) (θ : ℝ) :
  F = (2, 0) →
  θ = π/3 →
  (∃ (a b : ℝ), ∀ (x y : ℝ),
    (x, y) ∈ C ↔ x^2 / a^2 - y^2 / b^2 = 1 ∧
    b / a = Real.sqrt 3 ∧
    2^2 = a^2 + b^2) →
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 - y^2 / 3 = 1) :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_equation_l2904_290443


namespace NUMINAMATH_CALUDE_not_p_false_range_p_necessary_not_sufficient_range_l2904_290401

-- Define the function f
def f (x a : ℝ) : ℝ := x^2 - 2*x + a^2 + 3*a - 3

-- Define proposition p
def p (a : ℝ) : Prop := ∃ x, f x a < 0

-- Define proposition r
def r (a x : ℝ) : Prop := 1 - a ≤ x ∧ x ≤ 1 + a

-- Theorem for part (1)
theorem not_p_false_range (a : ℝ) : 
  ¬(¬(p a)) → a ∈ Set.Ioo (-4 : ℝ) 1 :=
sorry

-- Theorem for part (2)
theorem p_necessary_not_sufficient_range (a : ℝ) :
  (∀ x, r a x → p a) ∧ (∃ x, p a ∧ ¬r a x) → a ∈ Set.Ici 5 :=
sorry

end NUMINAMATH_CALUDE_not_p_false_range_p_necessary_not_sufficient_range_l2904_290401


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2904_290430

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 7) = 9 → x = 74 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2904_290430


namespace NUMINAMATH_CALUDE_policy_support_percentage_l2904_290421

theorem policy_support_percentage
  (total_population : ℕ)
  (men_count : ℕ)
  (women_count : ℕ)
  (men_support_rate : ℚ)
  (women_support_rate : ℚ)
  (h1 : total_population = men_count + women_count)
  (h2 : total_population = 1000)
  (h3 : men_count = 200)
  (h4 : women_count = 800)
  (h5 : men_support_rate = 70 / 100)
  (h6 : women_support_rate = 75 / 100)
  : (men_count * men_support_rate + women_count * women_support_rate) / total_population = 74 / 100 := by
  sorry

end NUMINAMATH_CALUDE_policy_support_percentage_l2904_290421


namespace NUMINAMATH_CALUDE_garden_carnations_percentage_l2904_290495

theorem garden_carnations_percentage 
  (total : ℕ) 
  (pink : ℕ) 
  (white : ℕ) 
  (pink_roses : ℕ) 
  (red_carnations : ℕ) 
  (h_pink : pink = 3 * total / 5)
  (h_white : white = total / 5)
  (h_pink_roses : pink_roses = pink / 2)
  (h_red_carnations : red_carnations = (total - pink - white) / 2) :
  (pink - pink_roses + red_carnations + white) * 100 = 60 * total :=
sorry

end NUMINAMATH_CALUDE_garden_carnations_percentage_l2904_290495


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2904_290424

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2904_290424


namespace NUMINAMATH_CALUDE_function_lower_bound_l2904_290431

theorem function_lower_bound (a : ℝ) (ha : a > 0) :
  ∀ x : ℝ, a * (Real.exp x + a) - x > 2 * Real.log a + 3/2 := by
  sorry

end NUMINAMATH_CALUDE_function_lower_bound_l2904_290431


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l2904_290442

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l2904_290442


namespace NUMINAMATH_CALUDE_max_sum_of_square_roots_l2904_290490

theorem max_sum_of_square_roots (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 7) :
  Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2) ≤ 9 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_square_roots_l2904_290490


namespace NUMINAMATH_CALUDE_scientific_notation_of_595_5_billion_yuan_l2904_290435

def billion : ℝ := 1000000000

theorem scientific_notation_of_595_5_billion_yuan :
  ∃ (a : ℝ) (n : ℤ), 
    595.5 * billion = a * (10 : ℝ) ^ n ∧ 
    1 ≤ |a| ∧ 
    |a| < 10 ∧
    a = 5.955 ∧
    n = 11 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_595_5_billion_yuan_l2904_290435


namespace NUMINAMATH_CALUDE_sell_all_cars_in_five_months_l2904_290471

/-- Calculates the number of months needed to sell all cars -/
def months_to_sell_cars (total_cars : ℕ) (num_salespeople : ℕ) (cars_per_salesperson_per_month : ℕ) : ℕ :=
  total_cars / (num_salespeople * cars_per_salesperson_per_month)

/-- Proves that it takes 5 months to sell all cars under given conditions -/
theorem sell_all_cars_in_five_months : 
  months_to_sell_cars 500 10 10 = 5 := by
  sorry

#eval months_to_sell_cars 500 10 10

end NUMINAMATH_CALUDE_sell_all_cars_in_five_months_l2904_290471


namespace NUMINAMATH_CALUDE_f_intersects_x_axis_min_distance_between_roots_range_of_a_l2904_290484

/-- The quadratic function f(x) = x^2 - 2ax - 2(a + 1) -/
def f (a x : ℝ) : ℝ := x^2 - 2*a*x - 2*(a + 1)

theorem f_intersects_x_axis (a : ℝ) : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 := by
  sorry

theorem min_distance_between_roots (a : ℝ) :
  ∃ x₁ x₂ : ℝ, f a x₁ = 0 ∧ f a x₂ = 0 ∧ |x₁ - x₂| ≥ 2 ∧ (∀ y₁ y₂ : ℝ, f a y₁ = 0 → f a y₂ = 0 → |y₁ - y₂| ≥ 2) := by
  sorry

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > -1 → f a x + 3 ≥ 0) → a ≤ Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_intersects_x_axis_min_distance_between_roots_range_of_a_l2904_290484


namespace NUMINAMATH_CALUDE_map_scale_l2904_290483

/-- Given a map where 15 cm represents 90 km, prove that 20 cm represents 120 km -/
theorem map_scale (map_cm : ℝ) (real_km : ℝ) (h : map_cm / 15 = real_km / 90) :
  (20 * real_km) / map_cm = 120 :=
sorry

end NUMINAMATH_CALUDE_map_scale_l2904_290483


namespace NUMINAMATH_CALUDE_mary_earnings_l2904_290469

/-- Mary's earnings problem -/
theorem mary_earnings (earnings_per_home : ℕ) (homes_cleaned : ℕ) : 
  earnings_per_home = 46 → homes_cleaned = 6 → earnings_per_home * homes_cleaned = 276 := by
  sorry

end NUMINAMATH_CALUDE_mary_earnings_l2904_290469


namespace NUMINAMATH_CALUDE_stock_price_increase_l2904_290486

theorem stock_price_increase (x : ℝ) : 
  (1 + x / 100) * (1 - 25 / 100) * (1 + 30 / 100) = 117 / 100 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_increase_l2904_290486


namespace NUMINAMATH_CALUDE_gas_purchase_cost_l2904_290477

/-- Calculates the total cost of gas purchases given a price rollback and two separate purchases. -/
theorem gas_purchase_cost 
  (rollback : ℝ) 
  (initial_price : ℝ) 
  (liters_today : ℝ) 
  (liters_friday : ℝ) 
  (h1 : rollback = 0.4) 
  (h2 : initial_price = 1.4) 
  (h3 : liters_today = 10) 
  (h4 : liters_friday = 25) : 
  initial_price * liters_today + (initial_price - rollback) * liters_friday = 39 := by
sorry

end NUMINAMATH_CALUDE_gas_purchase_cost_l2904_290477


namespace NUMINAMATH_CALUDE_oliver_seashell_collection_l2904_290448

/-- Represents the number of seashells Oliver collected on a given day -/
structure DailyCollection where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- Calculates the total number of seashells Oliver has after Thursday -/
def totalAfterThursday (c : DailyCollection) : ℕ :=
  c.monday + c.tuesday / 2 + c.wednesday + 5

/-- Theorem stating that Oliver collected 71 seashells on Monday, Tuesday, and Wednesday -/
theorem oliver_seashell_collection (c : DailyCollection) :
  totalAfterThursday c = 76 →
  c.monday + c.tuesday + c.wednesday = 71 := by
  sorry

end NUMINAMATH_CALUDE_oliver_seashell_collection_l2904_290448


namespace NUMINAMATH_CALUDE_fireworks_count_l2904_290429

/-- The number of fireworks Henry and his friend have now -/
def total_fireworks (henry_new : ℕ) (friend_new : ℕ) (last_year : ℕ) : ℕ :=
  henry_new + friend_new + last_year

/-- Proof that Henry and his friend have 11 fireworks in total -/
theorem fireworks_count : total_fireworks 2 3 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_fireworks_count_l2904_290429


namespace NUMINAMATH_CALUDE_arithmetic_proof_l2904_290402

theorem arithmetic_proof : -3 + 15 - (-8) = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_proof_l2904_290402


namespace NUMINAMATH_CALUDE_complex_multiplication_sum_l2904_290432

theorem complex_multiplication_sum (z a b : ℂ) : 
  z = 3 + Complex.I ∧ Complex.I * z = a + b * Complex.I → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_sum_l2904_290432


namespace NUMINAMATH_CALUDE_polynomial_equality_l2904_290450

theorem polynomial_equality (x : ℝ) (h : ℝ → ℝ) :
  (8 * x^4 - 4 * x^2 + 2 + h x = 2 * x^3 - 6 * x + 4) →
  (h x = -8 * x^4 + 2 * x^3 + 4 * x^2 - 6 * x + 2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2904_290450


namespace NUMINAMATH_CALUDE_common_tangent_implies_a_value_l2904_290449

/-- Two curves with a common tangent line at their common point imply a specific value for a parameter -/
theorem common_tangent_implies_a_value (e : ℝ) (a s t : ℝ) : 
  (t = (1 / (2 * Real.exp 1)) * s^2) →  -- Point P(s,t) is on the first curve
  (t = a * Real.log s) →                -- Point P(s,t) is on the second curve
  ((s / Real.exp 1) = (a / s)) →        -- Slopes are equal at point P(s,t)
  (a = 1) := by
sorry

end NUMINAMATH_CALUDE_common_tangent_implies_a_value_l2904_290449


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2904_290463

theorem quadratic_inequality_solution_set (m : ℝ) :
  (∀ x : ℝ, m * x^2 - (1 - m) * x + m ≥ 0) ↔ m ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2904_290463


namespace NUMINAMATH_CALUDE_students_only_swimming_l2904_290420

/-- The number of students only participating in swimming in a sports day scenario --/
theorem students_only_swimming (total : ℕ) (swimming : ℕ) (track : ℕ) (ball : ℕ) 
  (swim_track : ℕ) (swim_ball : ℕ) : 
  total = 28 → 
  swimming = 15 → 
  track = 8 → 
  ball = 14 → 
  swim_track = 3 → 
  swim_ball = 3 → 
  swimming - (swim_track + swim_ball) = 9 := by
  sorry

#check students_only_swimming

end NUMINAMATH_CALUDE_students_only_swimming_l2904_290420


namespace NUMINAMATH_CALUDE_max_y_coordinate_ellipse_l2904_290446

theorem max_y_coordinate_ellipse :
  ∀ x y : ℝ, x^2/25 + (y-3)^2/25 = 0 → y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_y_coordinate_ellipse_l2904_290446


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l2904_290487

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  Real.sqrt ((c1.1 - c2.1)^2 + (c1.2 - c2.2)^2) = r1 + r2

/-- The equation of the first circle: x^2 + y^2 = 4 -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The equation of the second circle: x^2 + y^2 - 10x + 16 = 0 -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 10*x + 16 = 0

theorem circles_externally_tangent :
  externally_tangent (0, 0) (5, 0) 2 3 :=
by sorry

#check circles_externally_tangent

end NUMINAMATH_CALUDE_circles_externally_tangent_l2904_290487


namespace NUMINAMATH_CALUDE_log_equation_solution_l2904_290426

theorem log_equation_solution (b x : ℝ) 
  (h1 : b > 0) 
  (h2 : b ≠ 1) 
  (h3 : x ≠ 1) 
  (h4 : Real.log x / Real.log (b^3) + Real.log b / Real.log (x^3) = 1) : 
  x = b^((3 + Real.sqrt 5) / 2) ∨ x = b^((3 - Real.sqrt 5) / 2) :=
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2904_290426


namespace NUMINAMATH_CALUDE_vessel_width_calculation_l2904_290414

-- Define the given parameters
def cube_edge : ℝ := 15
def vessel_length : ℝ := 20
def water_rise : ℝ := 12.053571428571429

-- Define the theorem
theorem vessel_width_calculation (w : ℝ) :
  (cube_edge ^ 3 = vessel_length * w * water_rise) →
  w = 14 := by
  sorry

end NUMINAMATH_CALUDE_vessel_width_calculation_l2904_290414


namespace NUMINAMATH_CALUDE_problem_statement_l2904_290411

theorem problem_statement :
  (∃ x : ℝ, x^2 - x + 1 ≥ 0) ∧
  ¬(∀ a b : ℝ, a^2 < b^2 → a < b) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2904_290411


namespace NUMINAMATH_CALUDE_highest_divisible_digit_l2904_290467

theorem highest_divisible_digit : 
  ∃ (a : ℕ), a ≤ 9 ∧ 
  (43752 * 1000 + a * 100 + 539) % 8 = 0 ∧
  (43752 * 1000 + a * 100 + 539) % 9 = 0 ∧
  (43752 * 1000 + a * 100 + 539) % 12 = 0 ∧
  ∀ (b : ℕ), b > a → b ≤ 9 → 
    (43752 * 1000 + b * 100 + 539) % 8 ≠ 0 ∨
    (43752 * 1000 + b * 100 + 539) % 9 ≠ 0 ∨
    (43752 * 1000 + b * 100 + 539) % 12 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_highest_divisible_digit_l2904_290467


namespace NUMINAMATH_CALUDE_razorback_shop_profit_l2904_290474

/-- Calculates the total profit from selling various items in the Razorback shop -/
def total_profit (jersey_profit t_shirt_profit hoodie_profit hat_profit : ℕ)
                 (jerseys_sold t_shirts_sold hoodies_sold hats_sold : ℕ) : ℕ :=
  jersey_profit * jerseys_sold +
  t_shirt_profit * t_shirts_sold +
  hoodie_profit * hoodies_sold +
  hat_profit * hats_sold

/-- The total profit from the Razorback shop during the Arkansas and Texas Tech game -/
theorem razorback_shop_profit :
  total_profit 76 204 132 48 2 158 75 120 = 48044 := by
  sorry

end NUMINAMATH_CALUDE_razorback_shop_profit_l2904_290474


namespace NUMINAMATH_CALUDE_rectangle_fold_trapezoid_l2904_290466

/-- 
Given a rectangle with sides a and b, if folding it along its diagonal 
creates an isosceles trapezoid with three equal sides and the fourth side 
of length 10√3, then a = 15 and b = 5√3.
-/
theorem rectangle_fold_trapezoid (a b : ℝ) 
  (h_rect : a > 0 ∧ b > 0)
  (h_fold : ∃ (x y z : ℝ), x = y ∧ y = z ∧ 
    x^2 + y^2 = a^2 + b^2 ∧ 
    z^2 + (10 * Real.sqrt 3)^2 = a^2 + b^2) : 
  a = 15 ∧ b = 5 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_fold_trapezoid_l2904_290466


namespace NUMINAMATH_CALUDE_cappuccino_cost_l2904_290403

theorem cappuccino_cost (cappuccino_cost : ℝ) : 
  (3 : ℝ) * cappuccino_cost + 2 * 3 + 2 * 1.5 + 2 * 1 = 20 - 3 → 
  cappuccino_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_cappuccino_cost_l2904_290403


namespace NUMINAMATH_CALUDE_all_groups_have_access_l2904_290444

-- Define the type for house groups
inductive HouseGroup : Type
  | a | b | c | d | e | f | g | h | i | j | k | l | m

-- Define the adjacency relation
def adjacent : HouseGroup → HouseGroup → Prop
  | HouseGroup.a, HouseGroup.b => True
  | HouseGroup.a, HouseGroup.d => True
  | HouseGroup.b, HouseGroup.a => True
  | HouseGroup.b, HouseGroup.c => True
  | HouseGroup.b, HouseGroup.d => True
  | HouseGroup.c, HouseGroup.b => True
  | HouseGroup.d, HouseGroup.a => True
  | HouseGroup.d, HouseGroup.b => True
  | HouseGroup.d, HouseGroup.f => True
  | HouseGroup.d, HouseGroup.e => True
  | HouseGroup.e, HouseGroup.d => True
  | HouseGroup.e, HouseGroup.f => True
  | HouseGroup.e, HouseGroup.j => True
  | HouseGroup.e, HouseGroup.l => True
  | HouseGroup.f, HouseGroup.d => True
  | HouseGroup.f, HouseGroup.e => True
  | HouseGroup.f, HouseGroup.j => True
  | HouseGroup.f, HouseGroup.i => True
  | HouseGroup.f, HouseGroup.g => True
  | HouseGroup.g, HouseGroup.f => True
  | HouseGroup.g, HouseGroup.i => True
  | HouseGroup.g, HouseGroup.h => True
  | HouseGroup.h, HouseGroup.g => True
  | HouseGroup.h, HouseGroup.i => True
  | HouseGroup.i, HouseGroup.j => True
  | HouseGroup.i, HouseGroup.f => True
  | HouseGroup.i, HouseGroup.g => True
  | HouseGroup.i, HouseGroup.h => True
  | HouseGroup.j, HouseGroup.k => True
  | HouseGroup.j, HouseGroup.e => True
  | HouseGroup.j, HouseGroup.f => True
  | HouseGroup.j, HouseGroup.i => True
  | HouseGroup.k, HouseGroup.l => True
  | HouseGroup.k, HouseGroup.j => True
  | HouseGroup.l, HouseGroup.k => True
  | HouseGroup.l, HouseGroup.e => True
  | _, _ => False

-- Define the set of pharmacy locations
def pharmacyLocations : Set HouseGroup :=
  {HouseGroup.b, HouseGroup.i, HouseGroup.l, HouseGroup.m}

-- Define the property of having access to a pharmacy
def hasAccessToPharmacy (g : HouseGroup) : Prop :=
  g ∈ pharmacyLocations ∨ ∃ h ∈ pharmacyLocations, adjacent g h

-- Theorem statement
theorem all_groups_have_access :
  ∀ g : HouseGroup, hasAccessToPharmacy g :=
by sorry

end NUMINAMATH_CALUDE_all_groups_have_access_l2904_290444


namespace NUMINAMATH_CALUDE_unique_max_sum_pair_l2904_290445

theorem unique_max_sum_pair :
  ∃! (x y : ℕ), 
    (∃ (k : ℕ), 19 * x + 95 * y = k * k) ∧
    19 * x + 95 * y ≤ 1995 ∧
    (∀ (a b : ℕ), (∃ (m : ℕ), 19 * a + 95 * b = m * m) → 
      19 * a + 95 * b ≤ 1995 → 
      a + b ≤ x + y) :=
by sorry

end NUMINAMATH_CALUDE_unique_max_sum_pair_l2904_290445


namespace NUMINAMATH_CALUDE_donut_distribution_l2904_290415

/-- The number of ways to distribute n identical objects into k distinct boxes,
    with each box containing at least m objects. -/
def distributionWays (n k m : ℕ) : ℕ := sorry

/-- The theorem stating that there are 10 ways to distribute 10 donuts
    into 4 kinds with at least 2 of each kind. -/
theorem donut_distribution : distributionWays 10 4 2 = 10 := by sorry

end NUMINAMATH_CALUDE_donut_distribution_l2904_290415


namespace NUMINAMATH_CALUDE_book_price_problem_l2904_290408

theorem book_price_problem (n : ℕ) (a : ℕ → ℝ) :
  n = 41 ∧
  a 1 = 7 ∧
  (∀ i, 1 ≤ i ∧ i < n → a (i + 1) = a i + 3) ∧
  a n = a ((n + 1) / 2) + a (((n + 1) / 2) + 1) →
  a ((n + 1) / 2) = 67 := by
sorry

end NUMINAMATH_CALUDE_book_price_problem_l2904_290408


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2904_290459

theorem quadratic_inequality_solution_sets (a b : ℝ) :
  (∀ x : ℝ, ax^2 + b*x + 2 > 0 ↔ -1 < x ∧ x < 2) →
  (∀ x : ℝ, 2*x^2 + b*x + a > 0 ↔ x < -1 ∨ x > 1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2904_290459


namespace NUMINAMATH_CALUDE_weight_of_b_l2904_290476

-- Define variables for weights and heights
variable (W_a W_b W_c : ℚ)
variable (h_a h_b h_c : ℚ)

-- Define the conditions
def condition1 : Prop := (W_a + W_b + W_c) / 3 = 45
def condition2 : Prop := (W_a + W_b) / 2 = 40
def condition3 : Prop := (W_b + W_c) / 2 = 47
def condition4 : Prop := h_a + h_c = 2 * h_b
def condition5 : Prop := ∃ (n : ℤ), W_a + W_b + W_c = 2 * n + 1

-- Theorem statement
theorem weight_of_b 
  (h1 : condition1 W_a W_b W_c)
  (h2 : condition2 W_a W_b)
  (h3 : condition3 W_b W_c)
  (h4 : condition4 h_a h_b h_c)
  (h5 : condition5 W_a W_b W_c) :
  W_b = 39 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l2904_290476


namespace NUMINAMATH_CALUDE_tan_value_from_sin_cos_equation_l2904_290406

theorem tan_value_from_sin_cos_equation (α : Real) 
  (h : Real.sin α + Real.sqrt 2 * Real.cos α = Real.sqrt 3) : 
  Real.tan α = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_sin_cos_equation_l2904_290406


namespace NUMINAMATH_CALUDE_absolute_value_fraction_sum_l2904_290441

theorem absolute_value_fraction_sum (x y : ℝ) (h1 : x < y) (h2 : y < 0) :
  |x| / x + |x * y| / (x * y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_fraction_sum_l2904_290441


namespace NUMINAMATH_CALUDE_line_equation_proof_l2904_290455

/-- Given a line in the form ax + by + c = 0, prove it has slope -3 and x-intercept 2 -/
theorem line_equation_proof (a b c : ℝ) (h1 : a = 3) (h2 : b = 1) (h3 : c = -6) : 
  (∀ x y : ℝ, a*x + b*y + c = 0 ↔ y = -3*(x - 2)) := by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2904_290455


namespace NUMINAMATH_CALUDE_ethans_rowing_time_l2904_290461

/-- Proves that Ethan's rowing time is 25 minutes given the conditions -/
theorem ethans_rowing_time (total_time : ℕ) (ethan_time : ℕ) :
  total_time = 75 →
  total_time = ethan_time + 2 * ethan_time →
  ethan_time = 25 := by
  sorry

end NUMINAMATH_CALUDE_ethans_rowing_time_l2904_290461


namespace NUMINAMATH_CALUDE_math_majors_consecutive_probability_math_majors_consecutive_probability_proof_l2904_290409

/-- The probability of all math majors sitting consecutively around a circular table -/
theorem math_majors_consecutive_probability : ℚ :=
  let total_people : ℕ := 12
  let math_majors : ℕ := 5
  let physics_majors : ℕ := 4
  let biology_majors : ℕ := 3
  1 / 330

/-- Proof that the probability of all math majors sitting consecutively is 1/330 -/
theorem math_majors_consecutive_probability_proof :
  math_majors_consecutive_probability = 1 / 330 := by
  sorry

end NUMINAMATH_CALUDE_math_majors_consecutive_probability_math_majors_consecutive_probability_proof_l2904_290409


namespace NUMINAMATH_CALUDE_bruno_score_l2904_290413

/-- Given that Richard's score is 62 and Bruno's score is 14 points lower than Richard's,
    prove that Bruno's score is 48. -/
theorem bruno_score (richard_score : ℕ) (bruno_diff : ℕ) : 
  richard_score = 62 → 
  bruno_diff = 14 → 
  richard_score - bruno_diff = 48 := by
  sorry

end NUMINAMATH_CALUDE_bruno_score_l2904_290413
