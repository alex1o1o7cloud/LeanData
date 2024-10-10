import Mathlib

namespace find_c_l1373_137362

theorem find_c (a b c d e : ℝ) : 
  (a + b + c) / 3 = 16 →
  (c + d + e) / 3 = 26 →
  (a + b + c + d + e) / 5 = 20 →
  c = 26 := by
sorry

end find_c_l1373_137362


namespace angle_inclination_range_l1373_137386

-- Define the slope k
def k : ℝ := sorry

-- Define the angle of inclination α in radians
def α : ℝ := sorry

-- Define the relationship between k and α
axiom slope_angle_relation : k = Real.tan α

-- Define the range of k
axiom k_range : -1 ≤ k ∧ k < 1

-- Define the range of α (0 to π)
axiom α_range : 0 ≤ α ∧ α < Real.pi

-- Theorem to prove
theorem angle_inclination_range :
  (0 ≤ α ∧ α < Real.pi / 4) ∨ (3 * Real.pi / 4 ≤ α ∧ α < Real.pi) :=
sorry

end angle_inclination_range_l1373_137386


namespace a_properties_l1373_137355

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => 3 * a n + 2 * (Int.sqrt (2 * (a n)^2 - 1)).toNat

theorem a_properties :
  (∀ n : ℕ, a n > 0) ∧
  (∀ m : ℕ, ¬(2015 ∣ a m)) := by
  sorry

end a_properties_l1373_137355


namespace sum_of_c_values_l1373_137372

theorem sum_of_c_values : ∃ (S : Finset ℤ),
  (∀ c ∈ S, c ≤ 30 ∧ 
    ∃ (x y : ℚ), y = x^2 - 11*x - c ∧ 
    ∀ z : ℚ, z^2 - 11*z - c = 0 ↔ (z = x ∨ z = y)) ∧
  (∀ c : ℤ, c ≤ 30 → 
    (∃ (x y : ℚ), y = x^2 - 11*x - c ∧ 
    ∀ z : ℚ, z^2 - 11*z - c = 0 ↔ (z = x ∨ z = y)) → 
    c ∈ S) ∧
  (S.sum id = 38) :=
sorry

end sum_of_c_values_l1373_137372


namespace imaginary_part_of_complex_power_imaginary_part_of_specific_complex_l1373_137367

theorem imaginary_part_of_complex_power (r θ : ℝ) (n : ℕ) :
  let z := (r * (Complex.cos θ + Complex.I * Complex.sin θ)) ^ n
  Complex.im z = r^n * Real.sin (n * θ) := by sorry

theorem imaginary_part_of_specific_complex (π : ℝ) :
  let z := (2 * (Complex.cos (π/4) + Complex.I * Complex.sin (π/4))) ^ 5
  Complex.im z = -16 * Real.sqrt 2 := by sorry

end imaginary_part_of_complex_power_imaginary_part_of_specific_complex_l1373_137367


namespace quadratic_completion_square_l1373_137319

theorem quadratic_completion_square (x : ℝ) : 
  (2 * x^2 + 3 * x + 1 = 0) ↔ (2 * (x + 3/4)^2 - 1/8 = 0) := by
  sorry

end quadratic_completion_square_l1373_137319


namespace product_of_roots_l1373_137342

theorem product_of_roots (x : ℝ) : 
  (25 * x^2 + 60 * x - 350 = 0) → 
  ∃ r₁ r₂ : ℝ, (r₁ * r₂ = -14 ∧ 25 * r₁^2 + 60 * r₁ - 350 = 0 ∧ 25 * r₂^2 + 60 * r₂ - 350 = 0) :=
by sorry

end product_of_roots_l1373_137342


namespace range_of_c_over_a_l1373_137341

theorem range_of_c_over_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a > b) (h3 : b > c) :
  -2 < c / a ∧ c / a < -1/2 := by
  sorry

end range_of_c_over_a_l1373_137341


namespace max_sum_of_factors_l1373_137315

theorem max_sum_of_factors (heart club : ℕ) : 
  heart * club = 48 → 
  Even heart → 
  heart + club ≤ 26 :=
sorry

end max_sum_of_factors_l1373_137315


namespace shaded_squares_count_l1373_137332

/-- Represents the number of shaded squares in each column of the grid -/
def shaded_per_column : List Nat := [1, 3, 5, 4, 2, 0, 0, 0]

/-- The total number of squares in the grid -/
def total_squares : Nat := 30

/-- The number of columns in the grid -/
def num_columns : Nat := 8

theorem shaded_squares_count :
  (List.sum shaded_per_column = 15) ∧
  (List.sum shaded_per_column = total_squares / 2) ∧
  (List.length shaded_per_column = num_columns) := by
  sorry

end shaded_squares_count_l1373_137332


namespace sum_of_digits_of_X_squared_l1373_137388

-- Define the number with 8 repeated ones
def X : ℕ := 11111111

-- Define a function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_of_X_squared : sum_of_digits (X^2) = 64 := by sorry

end sum_of_digits_of_X_squared_l1373_137388


namespace sqrt_comparison_l1373_137352

theorem sqrt_comparison : 2 * Real.sqrt 7 < 3 * Real.sqrt 5 := by
  sorry

end sqrt_comparison_l1373_137352


namespace square_shape_side_length_l1373_137322

theorem square_shape_side_length (x : ℝ) :
  x > 0 →
  x - 3 > 0 →
  (x + (x - 1)) = ((x - 2) + (x - 3) + 4) →
  1 = x - 3 →
  4 = (x + (x - 1)) - (2 * x - 5) := by
sorry

end square_shape_side_length_l1373_137322


namespace fixed_point_of_exponential_function_l1373_137343

/-- The function f(x) = a^(x+1) - 2 has a fixed point at (-1, -1) for a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x + 1) - 2
  f (-1) = -1 ∧ ∀ x : ℝ, f x = x → x = -1 := by
  sorry

end fixed_point_of_exponential_function_l1373_137343


namespace cloud_height_above_lake_l1373_137374

/-- The height of a cloud above a lake surface, given observation conditions --/
theorem cloud_height_above_lake (h : ℝ) (elevation_angle depression_angle : ℝ) : 
  h = 10 → 
  elevation_angle = 30 * π / 180 →
  depression_angle = 45 * π / 180 →
  ∃ (cloud_height : ℝ), abs (cloud_height - 37.3) < 0.1 := by
  sorry

end cloud_height_above_lake_l1373_137374


namespace malingerers_exposed_l1373_137306

/-- Represents a five-digit number where each digit is represented by a letter --/
structure CryptarithmNumber where
  a : Nat
  b : Nat
  c : Nat
  h_a_digit : a < 10
  h_b_digit : b < 10
  h_c_digit : c < 10
  h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c

def draftees (n : CryptarithmNumber) : Nat :=
  10000 * n.a + 1000 * n.a + 100 * n.b + 10 * n.b + n.b

def malingerers (n : CryptarithmNumber) : Nat :=
  10000 * n.a + 1000 * n.b + 100 * n.c + 10 * n.c + n.c

theorem malingerers_exposed (n : CryptarithmNumber) :
  draftees n - 1 = malingerers n → malingerers n = 10999 := by
  sorry

#check malingerers_exposed

end malingerers_exposed_l1373_137306


namespace equation_solution_l1373_137346

theorem equation_solution :
  ∃! x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 4 / (x + 3) = 4) ∧ (x = -14) :=
by sorry

end equation_solution_l1373_137346


namespace arithmetic_progression_sum_l1373_137389

/-- An arithmetic progression with a_3 = 10 -/
def arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + (n - 1) * d ∧ a 3 = 10

/-- The sum of a_1, a_2, and a_6 in the arithmetic progression -/
def sum_terms (a : ℕ → ℝ) : ℝ := a 1 + a 2 + a 6

theorem arithmetic_progression_sum (a : ℕ → ℝ) :
  arithmetic_progression a → sum_terms a = 30 := by
  sorry

end arithmetic_progression_sum_l1373_137389


namespace geometric_sequence_property_l1373_137360

/-- Given a geometric sequence {a_n} where a_6 + a_8 = 4, 
    prove that a_8(a_4 + 2a_6 + a_8) = 16 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_sum : a 6 + a 8 = 4) : 
  a 8 * (a 4 + 2 * a 6 + a 8) = 16 := by
  sorry

end geometric_sequence_property_l1373_137360


namespace factor_expression_l1373_137312

theorem factor_expression (x : ℝ) : 4*x*(x-5) + 6*(x-5) = (4*x+6)*(x-5) := by
  sorry

end factor_expression_l1373_137312


namespace cos_300_degrees_l1373_137392

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end cos_300_degrees_l1373_137392


namespace smallest_b_value_l1373_137330

theorem smallest_b_value (a b : ℤ) (h1 : 29 < a ∧ a < 41) (h2 : b < 51) 
  (h3 : (40 : ℚ) / b - (30 : ℚ) / 50 = (2 : ℚ) / 5) : b ≥ 40 := by
  sorry

end smallest_b_value_l1373_137330


namespace no_valid_x_l1373_137369

theorem no_valid_x : ¬∃ (x : ℕ), x > 1 ∧ x ≠ 5 ∧ x ≠ 6 ∧ x ≠ 12 ∧ 
  184 % 5 = 4 ∧ 184 % 6 = 4 ∧ 184 % x = 4 ∧ 184 % 12 = 4 := by
  sorry

end no_valid_x_l1373_137369


namespace platform_height_is_44_l1373_137371

/-- Represents the dimensions of a rectangular brick -/
structure Brick where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the experimental setup -/
structure Setup where
  platform_height : ℝ
  brick : Brick
  r : ℝ
  s : ℝ

/-- The main theorem stating the height of the platform -/
theorem platform_height_is_44 (setup : Setup) :
  setup.brick.length + setup.platform_height - 2 * setup.brick.width = setup.r ∧
  setup.brick.width + setup.platform_height - setup.brick.length = setup.s ∧
  setup.platform_height = 2 * setup.brick.width ∧
  setup.r = 36 ∧
  setup.s = 30 →
  setup.platform_height = 44 := by
sorry


end platform_height_is_44_l1373_137371


namespace greatest_q_minus_r_l1373_137383

theorem greatest_q_minus_r : ∃ (q r : ℕ+), 
  839 = 19 * q + r ∧ 
  ∀ (q' r' : ℕ+), 839 = 19 * q' + r' → (q - r : ℤ) ≥ (q' - r' : ℤ) ∧
  (q - r : ℤ) = 41 := by
  sorry

end greatest_q_minus_r_l1373_137383


namespace ball_catching_circle_l1373_137339

theorem ball_catching_circle (n : ℕ) (skip : ℕ) (h1 : n = 50) (h2 : skip = 6) :
  ∃ (m : ℕ), m = 25 ∧ m = n - (n.lcm skip / skip) :=
sorry

end ball_catching_circle_l1373_137339


namespace kiwi_count_l1373_137307

theorem kiwi_count (initial_oranges : ℕ) (added_kiwis : ℕ) (orange_percentage : ℚ) : 
  initial_oranges = 24 →
  added_kiwis = 26 →
  orange_percentage = 30 / 100 →
  ∃ initial_kiwis : ℕ, 
    (initial_oranges : ℚ) = orange_percentage * ((initial_oranges : ℚ) + (initial_kiwis : ℚ) + (added_kiwis : ℚ)) →
    initial_kiwis = 30 :=
by sorry

end kiwi_count_l1373_137307


namespace p_and_not_q_is_true_l1373_137365

-- Define proposition p
def p : Prop := ∃ x : ℝ, x - 2 > Real.log x

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 > 0

-- Theorem to prove
theorem p_and_not_q_is_true : p ∧ ¬q := by
  sorry

end p_and_not_q_is_true_l1373_137365


namespace multiple_of_smaller_number_l1373_137357

theorem multiple_of_smaller_number 
  (L S m : ℝ) 
  (h1 : L = 33) 
  (h2 : L + S = 51) 
  (h3 : L = m * S - 3) : 
  m = 2 := by
  sorry

end multiple_of_smaller_number_l1373_137357


namespace investment_equalizes_profits_l1373_137391

/-- The investment amount in yuan that equalizes profits from two selling methods -/
def investment : ℝ := 20000

/-- The profit rate when selling at the beginning of the month -/
def early_profit_rate : ℝ := 0.15

/-- The profit rate for reinvestment -/
def reinvestment_profit_rate : ℝ := 0.10

/-- The profit rate when selling at the end of the month -/
def late_profit_rate : ℝ := 0.30

/-- The storage fee in yuan -/
def storage_fee : ℝ := 700

/-- Theorem stating that the investment amount equalizes profits from both selling methods -/
theorem investment_equalizes_profits :
  investment * (1 + early_profit_rate) * (1 + reinvestment_profit_rate) =
  investment * (1 + late_profit_rate) - storage_fee := by
  sorry

#eval investment -- Should output 20000

end investment_equalizes_profits_l1373_137391


namespace marble_problem_l1373_137366

theorem marble_problem (total : ℕ) (white : ℕ) (remaining : ℕ) : 
  total = 50 → 
  white = 20 → 
  remaining = 40 → 
  ∃ (red blue removed : ℕ),
    red = blue ∧ 
    total = white + red + blue ∧
    removed = total - remaining ∧
    removed = 2 * (white - blue) :=
by
  sorry

end marble_problem_l1373_137366


namespace fraction_multiplication_l1373_137335

theorem fraction_multiplication (x y : ℝ) (h : x + y ≠ 0) :
  (3*x * 3*y) / (3*x + 3*y) = 3 * (x*y / (x+y)) := by
  sorry

end fraction_multiplication_l1373_137335


namespace sin_cos_105_degrees_l1373_137364

theorem sin_cos_105_degrees : Real.sin (105 * π / 180) * Real.cos (105 * π / 180) = -1/4 := by
  sorry

end sin_cos_105_degrees_l1373_137364


namespace inscribed_cube_surface_area_l1373_137378

/-- Given a cube with a sphere inscribed within it, and another cube inscribed within that sphere,
    this theorem relates the surface area of the outer cube to the surface area of the inner cube. -/
theorem inscribed_cube_surface_area (outer_surface_area : ℝ) :
  outer_surface_area = 54 →
  ∃ (inner_surface_area : ℝ),
    inner_surface_area = 18 ∧
    (∃ (outer_side_length inner_side_length : ℝ),
      outer_surface_area = 6 * outer_side_length^2 ∧
      inner_surface_area = 6 * inner_side_length^2 ∧
      inner_side_length = outer_side_length / Real.sqrt 3) :=
by
  sorry


end inscribed_cube_surface_area_l1373_137378


namespace binomial_8_choose_5_l1373_137379

theorem binomial_8_choose_5 : Nat.choose 8 5 = 56 := by
  sorry

end binomial_8_choose_5_l1373_137379


namespace right_angled_triangle_m_values_l1373_137310

/-- Given three lines that form a right-angled triangle, prove the possible values of m -/
theorem right_angled_triangle_m_values :
  ∀ (m : ℝ),
  (∃ (x y : ℝ), 3*x + 2*y + 6 = 0 ∧ 2*x - 3*m^2*y + 18 = 0 ∧ 2*m*x - 3*y + 12 = 0) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (3*x₁ + 2*y₁ + 6 = 0 ∧ 2*x₁ - 3*m^2*y₁ + 18 = 0) ∧
    (3*x₂ + 2*y₂ + 6 = 0 ∧ 2*m*x₂ - 3*y₂ + 12 = 0) ∧
    ((3*2 + 2*(-3*m^2) = 0) ∨ (3*(2*m) + 2*(-3) = 0) ∨ (2*(-3*m^2) + (-3)*(2*m) = 0))) →
  m = 0 ∨ m = -1 ∨ m = -4/9 :=
sorry

end right_angled_triangle_m_values_l1373_137310


namespace quadratic_symmetry_l1373_137368

/-- A quadratic function with specific properties -/
def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x, p a b c x = a * x^2 + b * x + c) →   -- p is quadratic
  (p a b c 9 = 4) →                         -- p(9) = 4
  (∀ x, p a b c (18 - x) = p a b c x) →     -- axis of symmetry at x = 9
  (∃ n : ℤ, p a b c 0 = n) →                -- p(0) is an integer
  p a b c 18 = 1 :=                         -- prove p(18) = 1
by sorry

end quadratic_symmetry_l1373_137368


namespace distinct_prime_factors_count_l1373_137321

def product : ℕ := 91 * 92 * 93 * 94

theorem distinct_prime_factors_count :
  (Nat.factors product).toFinset.card = 7 := by sorry

end distinct_prime_factors_count_l1373_137321


namespace final_quiz_score_for_a_l1373_137309

def number_of_quizzes : ℕ := 4
def average_score : ℚ := 92 / 100
def required_average : ℚ := 90 / 100

theorem final_quiz_score_for_a (final_score : ℚ) :
  (number_of_quizzes * average_score + final_score) / (number_of_quizzes + 1) ≥ required_average →
  final_score ≥ 82 / 100 :=
by sorry

end final_quiz_score_for_a_l1373_137309


namespace waiter_customers_l1373_137376

theorem waiter_customers (initial : ℕ) (left : ℕ) (new : ℕ) : 
  initial = 14 → left = 3 → new = 39 → initial - left + new = 50 := by
  sorry

end waiter_customers_l1373_137376


namespace mistaken_addition_l1373_137390

theorem mistaken_addition (N : ℤ) : (41 - N = 12) → (41 + N = 70) := by
  sorry

end mistaken_addition_l1373_137390


namespace abc_inequality_l1373_137318

theorem abc_inequality (a b c : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc : c < 0)
  (ha' : a > -3) (hb' : b > -3) (hc' : c > -3) : 
  a * b * c > -27 := by
  sorry

end abc_inequality_l1373_137318


namespace modified_cube_edge_count_l1373_137396

/-- Represents a cube with smaller cubes removed from its corners -/
structure ModifiedCube where
  sideLength : ℕ
  smallCubeSize : ℕ
  largeCubeSize : ℕ

/-- Calculates the number of edges in the modified cube -/
def edgeCount (c : ModifiedCube) : ℕ :=
  sorry

/-- Theorem stating that a cube of side length 4 with specific corner removals has 48 edges -/
theorem modified_cube_edge_count :
  let c : ModifiedCube := ⟨4, 1, 2⟩
  edgeCount c = 48 := by sorry

end modified_cube_edge_count_l1373_137396


namespace count_numbers_with_remainder_l1373_137384

theorem count_numbers_with_remainder (n : ℕ) : 
  (Finset.filter (fun N : ℕ => N > 17 ∧ 2017 % N = 17) (Finset.range (2017 + 1))).card = 13 := by
  sorry

end count_numbers_with_remainder_l1373_137384


namespace fraction_to_decimal_l1373_137329

theorem fraction_to_decimal : (4 : ℚ) / 5 = (0.8 : ℚ) := by sorry

end fraction_to_decimal_l1373_137329


namespace data_comparison_l1373_137349

def set1 (x₁ x₂ x₃ x₄ x₅ : ℝ) := [x₁, x₂, x₃, x₄, x₅]
def set2 (x₁ x₂ x₃ x₄ x₅ : ℝ) := [2*x₁+3, 2*x₂+3, 2*x₃+3, 2*x₄+3, 2*x₅+3]

def standardDeviation (xs : List ℝ) : ℝ := sorry
def median (xs : List ℝ) : ℝ := sorry
def mean (xs : List ℝ) : ℝ := sorry

theorem data_comparison (x₁ x₂ x₃ x₄ x₅ : ℝ) :
  (standardDeviation (set2 x₁ x₂ x₃ x₄ x₅) ≠ standardDeviation (set1 x₁ x₂ x₃ x₄ x₅)) ∧
  (median (set2 x₁ x₂ x₃ x₄ x₅) ≠ median (set1 x₁ x₂ x₃ x₄ x₅)) ∧
  (mean (set2 x₁ x₂ x₃ x₄ x₅) ≠ mean (set1 x₁ x₂ x₃ x₄ x₅)) := by
  sorry

end data_comparison_l1373_137349


namespace range_of_a_for_two_roots_l1373_137399

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 1 then (1/4) * x + 1 else Real.log x

theorem range_of_a_for_two_roots :
  ∃ (a_min a_max : ℝ), a_min = (1/4) ∧ a_max = (1/Real.exp 1) ∧
  ∀ (a : ℝ), (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = a * x₁ ∧ f x₂ = a * x₂ ∧
              ∀ (x : ℝ), f x = a * x → (x = x₁ ∨ x = x₂)) ↔
              (a_min ≤ a ∧ a < a_max) :=
sorry

end range_of_a_for_two_roots_l1373_137399


namespace functional_equation_solution_l1373_137397

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x * y) + x^2 = x * g y + g x

theorem functional_equation_solution (g : ℝ → ℝ) 
  (h1 : FunctionalEquation g) 
  (h2 : g (-1) = 7) : 
  g (-1001) = 6006013 := by
  sorry

end functional_equation_solution_l1373_137397


namespace rectangle_dimension_difference_l1373_137348

theorem rectangle_dimension_difference (x y : ℝ) 
  (perimeter : x + y = 10)  -- Half of the perimeter is 10
  (diagonal : x^2 + y^2 = 100)  -- Diagonal squared is 100
  : x - y = 10 := by sorry

end rectangle_dimension_difference_l1373_137348


namespace line_and_circle_equations_l1373_137359

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Circle in 2D space represented by the equation (x-h)² + (y-k)² = r² -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Given two points, determine if a line passes through the first point and is perpendicular to the line connecting the two points -/
def isPerpendicular (p1 p2 : Point) (l : Line) : Prop :=
  -- Line passes through p1
  l.a * p1.x + l.b * p1.y + l.c = 0 ∧
  -- Line is perpendicular to the line connecting p1 and p2
  l.a * (p2.x - p1.x) + l.b * (p2.y - p1.y) = 0

/-- Given two points, determine if a circle has these points as the endpoints of its diameter -/
def isDiameter (p1 p2 : Point) (c : Circle) : Prop :=
  -- Center of the circle is the midpoint of p1 and p2
  c.h = (p1.x + p2.x) / 2 ∧
  c.k = (p1.y + p2.y) / 2 ∧
  -- Radius of the circle is half the distance between p1 and p2
  c.r^2 = ((p2.x - p1.x)^2 + (p2.y - p1.y)^2) / 4

theorem line_and_circle_equations (A B : Point) (l : Line) (C : Circle)
    (hA : A.x = -3 ∧ A.y = -1)
    (hB : B.x = 5 ∧ B.y = 5)
    (hl : l.a = 4 ∧ l.b = 3 ∧ l.c = 15)
    (hC : C.h = 1 ∧ C.k = 2 ∧ C.r = 5) :
    isPerpendicular A B l ∧ isDiameter A B C := by
  sorry

end line_and_circle_equations_l1373_137359


namespace arcsin_arccos_equation_solution_l1373_137358

theorem arcsin_arccos_equation_solution (x : ℝ) :
  Real.arcsin (3 * x) + Real.arccos (2 * x) = π / 4 →
  x = 1 / Real.sqrt (11 - 2 * Real.sqrt 2) ∨
  x = -1 / Real.sqrt (11 - 2 * Real.sqrt 2) :=
by sorry

end arcsin_arccos_equation_solution_l1373_137358


namespace cube_product_three_four_l1373_137345

theorem cube_product_three_four : (3 : ℕ)^3 * (4 : ℕ)^3 = 1728 := by
  sorry

end cube_product_three_four_l1373_137345


namespace expected_adjacent_pairs_l1373_137325

/-- The expected number of adjacent boy-girl pairs in a random permutation of boys and girls -/
theorem expected_adjacent_pairs (num_boys num_girls : ℕ) : 
  let total := num_boys + num_girls
  let prob_pair := (num_boys : ℚ) * num_girls / (total * (total - 1))
  let num_pairs := total - 1
  num_boys = 8 → num_girls = 12 → 2 * num_pairs * prob_pair = 912 / 95 := by
  sorry

end expected_adjacent_pairs_l1373_137325


namespace unique_six_digit_number_l1373_137377

theorem unique_six_digit_number : ∃! n : ℕ,
  (100000 ≤ n ∧ n < 1000000) ∧  -- 6-digit number
  (n % 10 = 2 ∧ n / 100000 = 2) ∧  -- begins and ends with 2
  (∃ k : ℕ, n = (2*k - 2) * (2*k) * (2*k + 2)) ∧  -- product of three consecutive even integers
  n = 287232 :=
by sorry

end unique_six_digit_number_l1373_137377


namespace binomial_coefficient_equality_l1373_137320

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 28 (3 * x) = Nat.choose 28 (x + 8)) ↔ (x = 4 ∨ x = 5) :=
sorry

end binomial_coefficient_equality_l1373_137320


namespace division_problem_l1373_137327

theorem division_problem (n : ℕ) : n % 12 = 1 ∧ n / 12 = 9 → n = 109 := by
  sorry

end division_problem_l1373_137327


namespace certain_number_proof_l1373_137394

theorem certain_number_proof : ∃ x : ℝ, x * 9 = 0.45 * 900 ∧ x = 45 := by
  sorry

end certain_number_proof_l1373_137394


namespace subset_condition_implies_m_range_l1373_137382

theorem subset_condition_implies_m_range (m : ℝ) : 
  (∀ x, -1 < x ∧ x < 2 → -1 < x ∧ x < m + 1) ∧ 
  (∃ y, -1 < y ∧ y < m + 1 ∧ ¬(-1 < y ∧ y < 2)) → 
  m > 1 := by sorry

end subset_condition_implies_m_range_l1373_137382


namespace kira_downloaded_songs_l1373_137344

/-- The size of each song in megabytes -/
def song_size : ℕ := 5

/-- The total size of new songs in megabytes -/
def total_new_size : ℕ := 140

/-- The number of songs downloaded later on that day -/
def songs_downloaded : ℕ := total_new_size / song_size

theorem kira_downloaded_songs :
  songs_downloaded = 28 := by sorry

end kira_downloaded_songs_l1373_137344


namespace fashion_design_not_in_digital_china_l1373_137333

-- Define the concept of a service area
def ServiceArea : Type := String

-- Define Digital China as a structure with a set of service areas
structure DigitalChina :=
  (services : Set ServiceArea)

-- Define known service areas
def environmentalMonitoring : ServiceArea := "Environmental Monitoring"
def publicSecurity : ServiceArea := "Public Security"
def financialInfo : ServiceArea := "Financial Information"
def fashionDesign : ServiceArea := "Fashion Design"

-- Theorem: Fashion design is not a service area of Digital China
theorem fashion_design_not_in_digital_china 
  (dc : DigitalChina) 
  (h1 : environmentalMonitoring ∈ dc.services)
  (h2 : publicSecurity ∈ dc.services)
  (h3 : financialInfo ∈ dc.services) :
  fashionDesign ∉ dc.services := by
  sorry


end fashion_design_not_in_digital_china_l1373_137333


namespace simplify_expression_l1373_137334

theorem simplify_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 * b) / (a^2 - a * b) * (a / b - b / a) = a + b := by
  sorry

end simplify_expression_l1373_137334


namespace bridge_length_proof_l1373_137351

/-- 
Given a train with length 120 meters crossing a bridge in 55 seconds at a speed of 39.27272727272727 m/s,
prove that the length of the bridge is 2040 meters.
-/
theorem bridge_length_proof (train_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  train_length = 120 →
  crossing_time = 55 →
  train_speed = 39.27272727272727 →
  train_speed * crossing_time - train_length = 2040 :=
by sorry

end bridge_length_proof_l1373_137351


namespace max_gcd_of_product_7200_l1373_137387

theorem max_gcd_of_product_7200 :
  ∃ (a b : ℕ), a * b = 7200 ∧
  ∀ (x y : ℕ), x * y = 7200 → Nat.gcd x y ≤ Nat.gcd a b ∧
  Nat.gcd a b = 60 := by
  sorry

end max_gcd_of_product_7200_l1373_137387


namespace work_left_after_collaboration_l1373_137395

theorem work_left_after_collaboration (days_a days_b collab_days : ℕ) 
  (ha : days_a = 15) (hb : days_b = 20) (hc : collab_days = 3) : 
  1 - (collab_days * (1 / days_a + 1 / days_b)) = 13 / 20 := by
  sorry

end work_left_after_collaboration_l1373_137395


namespace light_ray_reflection_l1373_137331

/-- A light ray reflection problem -/
theorem light_ray_reflection 
  (M : ℝ × ℝ) 
  (N : ℝ × ℝ) 
  (l : ℝ → ℝ → Prop) : 
  M = (2, 6) → 
  N = (-3, 4) → 
  (∀ x y, l x y ↔ x - y + 3 = 0) → 
  ∃ A B C : ℝ, 
    (∀ x y, A * x + B * y + C = 0 ↔ 
      (∃ K : ℝ × ℝ, 
        -- K is symmetric to M with respect to l
        (K.1 - M.1) / (K.2 - M.2) = -1 ∧ 
        l ((K.1 + M.1) / 2) ((K.2 + M.2) / 2) ∧
        -- N lies on the line through K
        (N.2 - K.2) / (N.1 - K.1) = (y - K.2) / (x - K.1))) ∧
    A = 1 ∧ B = -6 ∧ C = 27 :=
by sorry

end light_ray_reflection_l1373_137331


namespace factor_x6_minus_64_l1373_137303

theorem factor_x6_minus_64 (x : ℝ) : 
  x^6 - 64 = (x - 2) * (x + 2) * (x^4 + 4*x^2 + 16) := by
  sorry

end factor_x6_minus_64_l1373_137303


namespace equation_solution_l1373_137363

theorem equation_solution (a : ℝ) : 
  (∀ x, 3*x + |a - 2| = -3 ↔ 3*x + 4 = 0) → 
  ((a - 2)^2010 - 2*a + 1 = -4 ∨ (a - 2)^2010 - 2*a + 1 = 0) := by
sorry

end equation_solution_l1373_137363


namespace unique_1x5x_divisible_by_36_l1373_137393

def is_form_1x5x (n : ℕ) : Prop :=
  ∃ x : ℕ, x < 10 ∧ n = 1000 + 100 * x + 50 + x

theorem unique_1x5x_divisible_by_36 :
  ∃! n : ℕ, is_form_1x5x n ∧ n % 36 = 0 :=
sorry

end unique_1x5x_divisible_by_36_l1373_137393


namespace existence_of_square_with_no_visible_points_l1373_137336

/-- A point is visible from the origin if the greatest common divisor of its coordinates is 1 -/
def visible_from_origin (x y : ℤ) : Prop := Int.gcd x y = 1

/-- A point (x, y) is inside a square with bottom-left corner (a, b) and side length n if
    a < x < a + n and b < y < b + n -/
def inside_square (x y a b n : ℤ) : Prop :=
  a < x ∧ x < a + n ∧ b < y ∧ y < b + n

theorem existence_of_square_with_no_visible_points :
  ∀ n : ℕ, n > 0 → ∃ a b : ℤ,
    ∀ x y : ℤ, inside_square x y a b n → ¬(visible_from_origin x y) :=
sorry

end existence_of_square_with_no_visible_points_l1373_137336


namespace circle_angle_measure_l1373_137347

noncomputable def Circle := ℝ × ℝ → Prop

def diameter (c : Circle) (A B : ℝ × ℝ) : Prop := sorry

def parallel (A B C D : ℝ × ℝ) : Prop := sorry

def angle (A B C : ℝ × ℝ) : ℝ := sorry

theorem circle_angle_measure 
  (c : Circle) (A B C D E : ℝ × ℝ) :
  diameter c E B →
  parallel E B D C →
  parallel A B E C →
  angle A E B = (3/7) * Real.pi →
  angle A B E = (4/7) * Real.pi →
  angle B D C = (900/7) * (Real.pi/180) :=
by sorry

end circle_angle_measure_l1373_137347


namespace arithmetic_sequence_sum_l1373_137324

theorem arithmetic_sequence_sum (a₁ : ℚ) (aₙ : ℚ) (d : ℚ) (n : ℕ) :
  a₁ = 2/7 →
  aₙ = 20/7 →
  d = 2/7 →
  n * (a₁ + aₙ) / 2 = 110/7 :=
by sorry

end arithmetic_sequence_sum_l1373_137324


namespace discount_percentage_proof_l1373_137385

theorem discount_percentage_proof (jacket_price shirt_price : ℝ)
  (jacket_discount shirt_discount : ℝ) :
  jacket_price = 80 →
  shirt_price = 40 →
  jacket_discount = 0.4 →
  shirt_discount = 0.55 →
  (jacket_price * jacket_discount + shirt_price * shirt_discount) /
  (jacket_price + shirt_price) = 0.45 := by
  sorry

end discount_percentage_proof_l1373_137385


namespace adam_apples_l1373_137316

theorem adam_apples (x : ℕ) : 
  x + 3 * x + 12 * x = 240 → x = 15 := by
  sorry

end adam_apples_l1373_137316


namespace complex_equal_parts_l1373_137304

theorem complex_equal_parts (a : ℝ) :
  let z : ℂ := a - 2 * Complex.I
  z.re = z.im → a = -2 := by sorry

end complex_equal_parts_l1373_137304


namespace two_distinct_roots_implies_b_value_l1373_137337

-- Define the polynomial function
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - b*x^2 + 1

-- State the theorem
theorem two_distinct_roots_implies_b_value (b : ℝ) :
  (∃ (x y : ℝ), x ≠ y ∧ f b x = 0 ∧ f b y = 0 ∧ 
   ∀ (z : ℝ), f b z = 0 → (z = x ∨ z = y)) →
  b = (3/2) * Real.rpow 2 (1/3) :=
sorry

end two_distinct_roots_implies_b_value_l1373_137337


namespace systematic_sampling_tenth_group_l1373_137311

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (firstDraw : ℕ) (n : ℕ) : ℕ :=
  firstDraw + (totalStudents / sampleSize) * (n - 1)

/-- Theorem: In a systematic sampling of 1000 students into 100 groups,
    if the number drawn from the first group is 6,
    then the number drawn from the tenth group is 96. -/
theorem systematic_sampling_tenth_group :
  systematicSample 1000 100 6 10 = 96 := by
  sorry

end systematic_sampling_tenth_group_l1373_137311


namespace fraction_simplification_l1373_137302

theorem fraction_simplification :
  (3 + 9 - 27 + 81 + 243 - 729) / (9 + 27 - 81 + 243 + 729 - 2187) = 1 / 3 := by
  sorry

end fraction_simplification_l1373_137302


namespace complex_magnitude_power_eight_l1373_137398

theorem complex_magnitude_power_eight :
  Complex.abs ((5 / 13 : ℂ) + (12 / 13 : ℂ) * Complex.I) ^ 8 = 1 := by sorry

end complex_magnitude_power_eight_l1373_137398


namespace power_equation_solution_l1373_137350

theorem power_equation_solution (n : ℕ) : 
  2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^18 → n = 17 := by
sorry

end power_equation_solution_l1373_137350


namespace worker_arrival_time_l1373_137300

theorem worker_arrival_time (S : ℝ) (D : ℝ) (h1 : D = S * 36) (h2 : S > 0) :
  D / (3/4 * S) - 36 = 12 := by
  sorry

end worker_arrival_time_l1373_137300


namespace fourth_grade_blue_count_l1373_137326

/-- Represents the number of students in each grade and uniform color combination -/
structure StudentCount where
  third_red_blue : ℕ
  third_white : ℕ
  fourth_red : ℕ
  fourth_white : ℕ
  fourth_blue : ℕ
  fifth_red_blue : ℕ
  fifth_white : ℕ

/-- The theorem stating the number of 4th grade students wearing blue uniforms -/
theorem fourth_grade_blue_count (s : StudentCount) : s.fourth_blue = 213 :=
  by
  have total_participants : s.third_red_blue + s.third_white + s.fourth_red + s.fourth_white + s.fourth_blue + s.fifth_red_blue + s.fifth_white = 2013 := by sorry
  have fourth_grade_total : s.fourth_red + s.fourth_white + s.fourth_blue = 600 := by sorry
  have fifth_grade_total : s.fifth_red_blue + s.fifth_white = 800 := by sorry
  have total_white : s.third_white + s.fourth_white + s.fifth_white = 800 := by sorry
  have third_red_blue : s.third_red_blue = 200 := by sorry
  have fourth_red : s.fourth_red = 200 := by sorry
  have fifth_white : s.fifth_white = 200 := by sorry
  sorry

end fourth_grade_blue_count_l1373_137326


namespace probability_for_given_scenario_l1373_137308

/-- The probability that at least 4 people stay for the entire basketball game -/
def probability_at_least_4_stay (total_people : ℕ) (certain_stay : ℕ) (uncertain_stay : ℕ) 
  (prob_uncertain_stay : ℚ) : ℚ :=
  sorry

/-- Theorem stating the probability for the specific scenario -/
theorem probability_for_given_scenario : 
  probability_at_least_4_stay 8 3 5 (1/3) = 401/243 := by
  sorry

end probability_for_given_scenario_l1373_137308


namespace subtract_point_five_from_forty_three_point_two_l1373_137340

theorem subtract_point_five_from_forty_three_point_two :
  43.2 - 0.5 = 42.7 := by sorry

end subtract_point_five_from_forty_three_point_two_l1373_137340


namespace regular_polygon_interior_twice_exterior_has_six_sides_l1373_137328

/-- A regular polygon where the sum of interior angles is twice the sum of exterior angles has 6 sides. -/
theorem regular_polygon_interior_twice_exterior_has_six_sides :
  ∀ n : ℕ,
  n > 2 →
  (n - 2) * 180 = 2 * 360 →
  n = 6 :=
by sorry

end regular_polygon_interior_twice_exterior_has_six_sides_l1373_137328


namespace gcd_180_450_l1373_137323

theorem gcd_180_450 : Nat.gcd 180 450 = 90 := by
  sorry

end gcd_180_450_l1373_137323


namespace parabola_standard_equation_l1373_137338

/-- A parabola with focus at (2, 0) that opens to the right has the standard equation y² = 8x -/
theorem parabola_standard_equation (f : ℝ × ℝ) (opens_right : Bool) :
  f = (2, 0) → opens_right = true → ∃ (x y : ℝ), y^2 = 8*x := by
  sorry

end parabola_standard_equation_l1373_137338


namespace hyperbola_equation_l1373_137356

/-- Given a hyperbola with asymptote equations x ± 2y = 0 and focal length 10,
    prove that its equation is either x²/20 - y²/5 = 1 or y²/5 - x²/20 = 1 -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ k : ℝ, k * x + 2 * y = 0 ∧ k * x - 2 * y = 0) →
  (∃ c : ℝ, c^2 = 100) →
  (x^2 / 20 - y^2 / 5 = 1) ∨ (y^2 / 5 - x^2 / 20 = 1) :=
by sorry

end hyperbola_equation_l1373_137356


namespace cyclic_quadrilateral_intersection_product_l1373_137353

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary geometric relations
variable (is_convex_cyclic_quadrilateral : Point → Point → Point → Point → Prop)
variable (is_center_of_circumcircle : Point → Point → Point → Point → Point → Prop)
variable (is_on_circle : Point → Circle → Prop)
variable (circumcircle : Point → Point → Point → Circle)
variable (intersection_point : Circle → Circle → Point)

-- Define the distance function
variable (distance : Point → Point → ℝ)

theorem cyclic_quadrilateral_intersection_product
  (A B C D O Q : Point)
  (h1 : is_convex_cyclic_quadrilateral A B C D)
  (h2 : is_center_of_circumcircle O A B C D)
  (h3 : Q = intersection_point (circumcircle O A B) (circumcircle O C D))
  : distance Q A * distance Q B = distance Q C * distance Q D := by
  sorry

end cyclic_quadrilateral_intersection_product_l1373_137353


namespace different_color_probability_l1373_137373

def blue_chips : ℕ := 6
def red_chips : ℕ := 5
def yellow_chips : ℕ := 4
def green_chips : ℕ := 3

def total_chips : ℕ := blue_chips + red_chips + yellow_chips + green_chips

theorem different_color_probability :
  let p_different := (blue_chips * (total_chips - blue_chips) +
                      red_chips * (total_chips - red_chips) +
                      yellow_chips * (total_chips - yellow_chips) +
                      green_chips * (total_chips - green_chips)) /
                     (total_chips * total_chips)
  p_different = 119 / 162 := by
  sorry

end different_color_probability_l1373_137373


namespace arrangement_count_l1373_137380

/-- Represents the number of different books of each subject -/
structure BookCounts where
  math : Nat
  physics : Nat
  chemistry : Nat

/-- Calculates the number of arrangements given the book counts and constraints -/
def countArrangements (books : BookCounts) : Nat :=
  let totalBooks := books.math + books.physics + books.chemistry
  let mathUnit := 1  -- Treat math books as a single unit
  let nonMathBooks := books.physics + books.chemistry
  let totalUnits := mathUnit + nonMathBooks
  sorry

/-- The theorem to be proven -/
theorem arrangement_count :
  let books : BookCounts := { math := 3, physics := 2, chemistry := 1 }
  countArrangements books = 2592 := by
  sorry

end arrangement_count_l1373_137380


namespace july_birth_percentage_l1373_137305

def total_athletes : ℕ := 120
def july_athletes : ℕ := 18

def percentage_born_in_july : ℚ := july_athletes / total_athletes * 100

theorem july_birth_percentage :
  percentage_born_in_july = 15 := by
  sorry

end july_birth_percentage_l1373_137305


namespace wire_ratio_proof_l1373_137301

theorem wire_ratio_proof (total_length : ℝ) (shorter_length : ℝ) :
  total_length = 50 →
  shorter_length = 14.285714285714285 →
  let longer_length := total_length - shorter_length
  shorter_length / longer_length = 2 / 5 := by
sorry

end wire_ratio_proof_l1373_137301


namespace best_fit_line_slope_l1373_137381

/-- Represents a temperature measurement at a specific time -/
structure Measurement where
  time : ℝ
  temp : ℝ

/-- Given three equally spaced time measurements with corresponding temperatures,
    the slope of the best-fit line is (T₃ - T₁) / (t₃ - t₁) -/
theorem best_fit_line_slope (m₁ m₂ m₃ : Measurement) (h : ℝ) 
    (h1 : m₂.time = m₁.time + h)
    (h2 : m₃.time = m₁.time + 2 * h) :
  (m₃.temp - m₁.temp) / (m₃.time - m₁.time) =
    ((m₁.time - (m₁.time + h)) * (m₁.temp - (m₁.temp + m₂.temp + m₃.temp) / 3) +
     (m₂.time - (m₁.time + h)) * (m₂.temp - (m₁.temp + m₂.temp + m₃.temp) / 3) +
     (m₃.time - (m₁.time + h)) * (m₃.temp - (m₁.temp + m₂.temp + m₃.temp) / 3)) /
    ((m₁.time - (m₁.time + h))^2 + (m₂.time - (m₁.time + h))^2 + (m₃.time - (m₁.time + h))^2) :=
by sorry

end best_fit_line_slope_l1373_137381


namespace tangent_line_to_exponential_curve_l1373_137313

/-- Given that the line y = 1/m is tangent to the curve y = xe^x, prove that m = -e -/
theorem tangent_line_to_exponential_curve (m : ℝ) : 
  (∃ n : ℝ, n * Real.exp n = 1/m ∧ 
   ∀ x : ℝ, x * Real.exp x ≤ 1/m ∧ 
   (x * Real.exp x = 1/m → x = n)) → 
  m = -Real.exp 1 := by
sorry

end tangent_line_to_exponential_curve_l1373_137313


namespace two_times_first_exceeds_three_times_second_l1373_137354

theorem two_times_first_exceeds_three_times_second (x y : ℝ) : 
  x + y = 10 → x = 7 → y = 3 → 2 * x - 3 * y = 5 := by
  sorry

end two_times_first_exceeds_three_times_second_l1373_137354


namespace cow_count_is_twelve_l1373_137361

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs in the group -/
def totalLegs (count : AnimalCount) : ℕ :=
  2 * count.ducks + 4 * count.cows

/-- Calculates the total number of heads in the group -/
def totalHeads (count : AnimalCount) : ℕ :=
  count.ducks + count.cows

/-- Theorem stating that the number of cows is 12 -/
theorem cow_count_is_twelve :
  ∃ (count : AnimalCount), 
    totalLegs count = 2 * totalHeads count + 24 ∧ 
    count.cows = 12 := by
  sorry

end cow_count_is_twelve_l1373_137361


namespace right_triangle_side_length_l1373_137370

theorem right_triangle_side_length : ∃ (k : ℕ), 
  (5 * k : ℕ) > 0 ∧ 
  (12 * k : ℕ) > 0 ∧ 
  (13 * k : ℕ) > 0 ∧ 
  (5 * k)^2 + (12 * k)^2 = (13 * k)^2 ∧ 
  (13 * k = 91 ∨ 12 * k = 91 ∨ 5 * k = 91) :=
by
  sorry

end right_triangle_side_length_l1373_137370


namespace rectangular_field_width_l1373_137317

/-- Given a rectangular field with perimeter 240 meters and perimeter equal to 3 times its length, prove that its width is 40 meters. -/
theorem rectangular_field_width (length width : ℝ) : 
  (2 * length + 2 * width = 240) →  -- Perimeter formula
  (240 = 3 * length) →              -- Perimeter is 3 times length
  width = 40 := by
sorry

end rectangular_field_width_l1373_137317


namespace xyz_max_value_l1373_137375

theorem xyz_max_value (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x + y + z = 1 → 
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y → 
  x * y * z ≤ 3/125 := by
sorry

end xyz_max_value_l1373_137375


namespace greatest_two_digit_with_digit_product_12_l1373_137314

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

theorem greatest_two_digit_with_digit_product_12 :
  ∃ (n : ℕ), is_two_digit n ∧ digit_product n = 12 ∧
  ∀ (m : ℕ), is_two_digit m → digit_product m = 12 → m ≤ n :=
by
  use 62
  sorry

end greatest_two_digit_with_digit_product_12_l1373_137314
