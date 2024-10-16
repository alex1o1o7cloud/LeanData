import Mathlib

namespace NUMINAMATH_CALUDE_max_product_f_value_l1144_114422

-- Define the function f
def f (a b x : ℝ) : ℝ := 2 * a * x + b

-- State the theorem
theorem max_product_f_value :
  ∀ a b : ℝ,
    a > 0 →
    b > 0 →
    (∀ x : ℝ, x ∈ Set.Icc (-1/2) (1/2) → |f a b x| ≤ 2) →
    (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
      (∀ x : ℝ, x ∈ Set.Icc (-1/2) (1/2) → |f a' b' x| ≤ 2) → 
      a * b ≥ a' * b') →
    f a b 2017 = 4035 :=
by
  sorry


end NUMINAMATH_CALUDE_max_product_f_value_l1144_114422


namespace NUMINAMATH_CALUDE_quadratic_vertex_l1144_114489

/-- The quadratic function f(x) = (x-2)^2 - 1 -/
def f (x : ℝ) : ℝ := (x - 2)^2 - 1

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (2, -1)

/-- Theorem: The vertex of the quadratic function f(x) = (x-2)^2 - 1 is at the point (2, -1) -/
theorem quadratic_vertex : 
  (∀ x : ℝ, f x ≥ f (vertex.1)) ∧ vertex.2 = f (vertex.1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l1144_114489


namespace NUMINAMATH_CALUDE_five_ruble_coins_l1144_114480

/-- Represents the number of coins of each denomination -/
structure CoinCount where
  one : ℕ
  two : ℕ
  five : ℕ
  ten : ℕ

/-- The total number of coins -/
def total_coins : ℕ := 25

/-- The number of coins that are not of each denomination -/
def not_two_coins : ℕ := 19
def not_ten_coins : ℕ := 20
def not_one_coins : ℕ := 16

/-- Theorem stating the number of five-ruble coins -/
theorem five_ruble_coins (c : CoinCount) : c.five = 5 :=
  by
    have h1 : c.one + c.two + c.five + c.ten = total_coins := sorry
    have h2 : c.two = total_coins - not_two_coins := sorry
    have h3 : c.ten = total_coins - not_ten_coins := sorry
    have h4 : c.one = total_coins - not_one_coins := sorry
    sorry

end NUMINAMATH_CALUDE_five_ruble_coins_l1144_114480


namespace NUMINAMATH_CALUDE_range_of_m_l1144_114471

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |x - m| ≤ 2) → 
  -1 ≤ m ∧ m ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1144_114471


namespace NUMINAMATH_CALUDE_solve_cube_equation_l1144_114464

theorem solve_cube_equation : ∃ x : ℝ, (x - 5)^3 = -(1/27)⁻¹ ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_cube_equation_l1144_114464


namespace NUMINAMATH_CALUDE_ball_drawing_probabilities_l1144_114437

/-- The probability of drawing a red ball exactly on the fourth draw with replacement -/
def prob_red_fourth_with_replacement (total_balls : ℕ) (red_balls : ℕ) : ℚ :=
  (1 - red_balls / total_balls) ^ 3 * (red_balls / total_balls)

/-- The probability of drawing a red ball exactly on the fourth draw without replacement -/
def prob_red_fourth_without_replacement (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) : ℚ :=
  (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1)) *
  ((white_balls - 2) / (total_balls - 2)) * (red_balls / (total_balls - 3))

theorem ball_drawing_probabilities :
  let total_balls := 10
  let red_balls := 6
  let white_balls := 4
  prob_red_fourth_with_replacement total_balls red_balls = 24 / 625 ∧
  prob_red_fourth_without_replacement total_balls red_balls white_balls = 1 / 70 := by
  sorry

end NUMINAMATH_CALUDE_ball_drawing_probabilities_l1144_114437


namespace NUMINAMATH_CALUDE_no_real_solutions_l1144_114472

theorem no_real_solutions : ¬∃ (x y z : ℝ), (x + y = 3) ∧ (x*y - z^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1144_114472


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1144_114494

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 - 2^3 = 190 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1144_114494


namespace NUMINAMATH_CALUDE_yellow_pairs_l1144_114491

theorem yellow_pairs (total_students : ℕ) (blue_students : ℕ) (yellow_students : ℕ) 
  (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  total_students = 132 →
  blue_students = 57 →
  yellow_students = 75 →
  total_pairs = 66 →
  blue_blue_pairs = 23 →
  blue_students + yellow_students = total_students →
  2 * total_pairs = total_students →
  ∃ (yellow_yellow_pairs : ℕ),
    yellow_yellow_pairs = 32 ∧
    yellow_yellow_pairs + blue_blue_pairs + (total_pairs - yellow_yellow_pairs - blue_blue_pairs) = total_pairs :=
by sorry

end NUMINAMATH_CALUDE_yellow_pairs_l1144_114491


namespace NUMINAMATH_CALUDE_binary_10011_is_19_l1144_114451

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Define the binary number 10011
def binary_10011 : List Bool := [true, true, false, false, true]

-- Theorem statement
theorem binary_10011_is_19 : binary_to_decimal binary_10011 = 19 := by
  sorry

end NUMINAMATH_CALUDE_binary_10011_is_19_l1144_114451


namespace NUMINAMATH_CALUDE_five_letter_words_count_l1144_114448

/-- The number of vowels in the alphabet -/
def num_vowels : ℕ := 5

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of five-letter words where the first and last letters are the same vowel,
    and the remaining three letters can be any letters from the alphabet -/
def num_words : ℕ := num_vowels * alphabet_size^3

theorem five_letter_words_count : num_words = 87880 := by
  sorry

end NUMINAMATH_CALUDE_five_letter_words_count_l1144_114448


namespace NUMINAMATH_CALUDE_integer_count_equality_l1144_114495

theorem integer_count_equality : 
  ∃! (count : ℕ), count = 39999 ∧ 
  (∀ n : ℤ, (2 + ⌊(200 * n : ℚ) / 201⌋ = ⌈(198 * n : ℚ) / 199⌉) ↔ 
    (∃ k : ℤ, 0 ≤ k ∧ k < count ∧ n ≡ k [ZMOD 39999])) :=
by sorry

end NUMINAMATH_CALUDE_integer_count_equality_l1144_114495


namespace NUMINAMATH_CALUDE_fescue_percentage_in_Y_l1144_114467

/-- Represents a seed mixture --/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The combined mixture of X and Y --/
def CombinedMixture (X Y : SeedMixture) (xWeight : ℝ) : SeedMixture :=
  { ryegrass := xWeight * X.ryegrass + (1 - xWeight) * Y.ryegrass,
    bluegrass := xWeight * X.bluegrass + (1 - xWeight) * Y.bluegrass,
    fescue := xWeight * X.fescue + (1 - xWeight) * Y.fescue }

theorem fescue_percentage_in_Y
  (X : SeedMixture)
  (Y : SeedMixture)
  (h1 : X.ryegrass = 0.4)
  (h2 : X.bluegrass = 0.6)
  (h3 : Y.ryegrass = 0.25)
  (h4 : (CombinedMixture X Y (1/3)).ryegrass = 0.3)
  : Y.fescue = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_fescue_percentage_in_Y_l1144_114467


namespace NUMINAMATH_CALUDE_max_value_expression_l1144_114469

theorem max_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y * z * (x + y + z)^2) / ((x + y)^2 * (y + z)^2) ≤ (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l1144_114469


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_10_l1144_114427

/-- The area of a circle with diameter 10 centimeters is 25π square centimeters. -/
theorem circle_area_with_diameter_10 :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 25 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_10_l1144_114427


namespace NUMINAMATH_CALUDE_digit_value_in_different_bases_l1144_114446

theorem digit_value_in_different_bases :
  ∃ (d : ℕ), d < 7 ∧ d * 7 + 4 = d * 8 + 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_digit_value_in_different_bases_l1144_114446


namespace NUMINAMATH_CALUDE_max_expression_l1144_114483

theorem max_expression (x₁ x₂ y₁ y₂ : ℝ) 
  (hx : 0 < x₁ ∧ x₁ < x₂) 
  (hy : 0 < y₁ ∧ y₁ < y₂) 
  (hsum_x : x₁ + x₂ = 1) 
  (hsum_y : y₁ + y₂ = 1) : 
  x₁ * y₁ + x₂ * y₂ ≥ max (x₁ * x₂ + y₁ * y₂) (max (x₁ * y₂ + x₂ * y₁) (1/2)) :=
sorry

end NUMINAMATH_CALUDE_max_expression_l1144_114483


namespace NUMINAMATH_CALUDE_min_area_intersecting_hyperbolas_l1144_114426

/-- A set in ℝ² is convex if for any two points in the set, 
    the line segment connecting them is also in the set -/
def is_convex (S : Set (ℝ × ℝ)) : Prop :=
  ∀ (p q : ℝ × ℝ), p ∈ S → q ∈ S → 
    ∀ (t : ℝ), 0 ≤ t ∧ t ≤ 1 → (1 - t) • p + t • q ∈ S

/-- A set intersects a hyperbola if there exists a point in the set 
    that satisfies the hyperbola equation -/
def intersects_hyperbola (S : Set (ℝ × ℝ)) (k : ℝ) : Prop :=
  ∃ (x y : ℝ), (x, y) ∈ S ∧ x * y = k

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- The main theorem: The minimum area of a convex set intersecting 
    both branches of xy = 1 and xy = -1 is 4 -/
theorem min_area_intersecting_hyperbolas :
  (∃ (S : Set (ℝ × ℝ)), 
    is_convex S ∧ 
    intersects_hyperbola S 1 ∧ 
    intersects_hyperbola S (-1)) →
  (∀ (S : Set (ℝ × ℝ)), 
    is_convex S → 
    intersects_hyperbola S 1 → 
    intersects_hyperbola S (-1) → 
    area S ≥ 4) ∧
  (∃ (S : Set (ℝ × ℝ)), 
    is_convex S ∧ 
    intersects_hyperbola S 1 ∧ 
    intersects_hyperbola S (-1) ∧ 
    area S = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_area_intersecting_hyperbolas_l1144_114426


namespace NUMINAMATH_CALUDE_abs_negative_seventeen_l1144_114466

theorem abs_negative_seventeen : |(-17 : ℤ)| = 17 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_seventeen_l1144_114466


namespace NUMINAMATH_CALUDE_lending_period_is_one_year_l1144_114418

/-- Proves that the lending period is 1 year given the problem conditions --/
theorem lending_period_is_one_year 
  (principal : ℝ)
  (borrowing_rate : ℝ)
  (lending_rate : ℝ)
  (annual_gain : ℝ)
  (h1 : principal = 5000)
  (h2 : borrowing_rate = 0.04)
  (h3 : lending_rate = 0.05)
  (h4 : annual_gain = 50)
  : ∃ t : ℝ, t = 1 ∧ principal * lending_rate * t - principal * borrowing_rate * t = annual_gain :=
sorry

end NUMINAMATH_CALUDE_lending_period_is_one_year_l1144_114418


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1144_114457

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (2 * x + 1)^3 = a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀) →
  a₀ + a₁ + a₂ + a₃ = 27 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1144_114457


namespace NUMINAMATH_CALUDE_ceiling_times_self_equals_156_l1144_114436

theorem ceiling_times_self_equals_156 :
  ∃! (x : ℝ), ⌈x⌉ * x = 156 :=
by sorry

end NUMINAMATH_CALUDE_ceiling_times_self_equals_156_l1144_114436


namespace NUMINAMATH_CALUDE_function_and_composition_l1144_114429

def f (x : ℝ) (b c : ℝ) : ℝ := x^2 - b*x + c

theorem function_and_composition 
  (h1 : f 1 b c = 0) 
  (h2 : f 2 b c = -3) :
  (∀ x, f x b c = x^2 - 6*x + 5) ∧ 
  (∀ x > -1, f (1 / Real.sqrt (x + 1)) b c = 1 / (x + 1) - 6 / Real.sqrt (x + 1) + 5) := by
  sorry

end NUMINAMATH_CALUDE_function_and_composition_l1144_114429


namespace NUMINAMATH_CALUDE_simplify_expression_l1144_114496

theorem simplify_expression (a : ℝ) : 5*a + 2*a + 3*a - 2*a = 8*a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1144_114496


namespace NUMINAMATH_CALUDE_rectangle_configuration_l1144_114454

/-- The side length of square S2 in the given rectangle configuration. -/
def side_length_S2 : ℕ := 1300

/-- The side length of squares S1 and S3 in the given rectangle configuration. -/
def side_length_S1_S3 : ℕ := side_length_S2 + 50

/-- The width of the entire rectangle. -/
def total_width : ℕ := 4000

/-- The height of the entire rectangle. -/
def total_height : ℕ := 2500

/-- The theorem stating that the given configuration satisfies all conditions. -/
theorem rectangle_configuration :
  side_length_S1_S3 + side_length_S2 + side_length_S1_S3 = total_width ∧
  ∃ (r : ℕ), 2 * r + side_length_S2 = total_height :=
by sorry

end NUMINAMATH_CALUDE_rectangle_configuration_l1144_114454


namespace NUMINAMATH_CALUDE_polygon_diagonals_l1144_114434

theorem polygon_diagonals (n : ℕ+) : 
  (∃ n, n * (n - 3) / 2 = 2 ∨ n * (n - 3) / 2 = 54) ∧ 
  (∀ n, n * (n - 3) / 2 ≠ 21 ∧ n * (n - 3) / 2 ≠ 32 ∧ n * (n - 3) / 2 ≠ 63) :=
by sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l1144_114434


namespace NUMINAMATH_CALUDE_m_range_l1144_114444

-- Define the plane region
def plane_region (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |2 * p.1 + p.2 + m| < 3}

-- Theorem statement
theorem m_range (m : ℝ) :
  ((0, 0) ∈ plane_region m) ∧
  ((-1, 1) ∈ plane_region m) ↔
  -2 < m ∧ m < 3 :=
by
  sorry


end NUMINAMATH_CALUDE_m_range_l1144_114444


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_t_for_nonempty_solution_l1144_114497

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| - |2*x - 4|

-- Theorem 1: Solution set of f(x) > 2
theorem solution_set_f_greater_than_2 :
  {x : ℝ | f x > 2} = Set.Ioo (5/3) 3 := by sorry

-- Theorem 2: Range of t for non-empty solution set of f(x) > t^2 + 2t
theorem range_of_t_for_nonempty_solution :
  ∀ t : ℝ, (∃ x : ℝ, f x > t^2 + 2*t) ↔ t ∈ Set.Ioo (-3) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_t_for_nonempty_solution_l1144_114497


namespace NUMINAMATH_CALUDE_unique_intersection_iff_k_eq_22_div_3_l1144_114488

/-- The parabola function -/
def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 7

/-- The line function -/
def line (k : ℝ) : ℝ := k

/-- The condition for exactly one intersection point -/
def has_unique_intersection (k : ℝ) : Prop :=
  ∃! y, parabola y = line k

theorem unique_intersection_iff_k_eq_22_div_3 :
  ∀ k : ℝ, has_unique_intersection k ↔ k = 22 / 3 := by sorry

end NUMINAMATH_CALUDE_unique_intersection_iff_k_eq_22_div_3_l1144_114488


namespace NUMINAMATH_CALUDE_power_function_sum_range_l1144_114401

def f (x : ℝ) : ℝ := x^2

theorem power_function_sum_range 
  (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ ≥ x₂) 
  (h₂ : x₂ ≥ x₃) 
  (h₃ : x₁ + x₂ + x₃ = 1) 
  (h₄ : f x₁ + f x₂ + f x₃ = 1) :
  2/3 ≤ x₁ + x₂ ∧ x₁ + x₂ ≤ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_power_function_sum_range_l1144_114401


namespace NUMINAMATH_CALUDE_first_number_proof_l1144_114485

theorem first_number_proof (x y : ℝ) : 
  x + y = 10 → 2 * x = 3 * y + 5 → x = 7 := by sorry

end NUMINAMATH_CALUDE_first_number_proof_l1144_114485


namespace NUMINAMATH_CALUDE_cubic_root_sum_of_squares_reciprocal_l1144_114459

theorem cubic_root_sum_of_squares_reciprocal (a b c : ℝ) : 
  a^3 - 12*a^2 + 20*a - 3 = 0 → 
  b^3 - 12*b^2 + 20*b - 3 = 0 → 
  c^3 - 12*c^2 + 20*c - 3 = 0 → 
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 328/9 := by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_of_squares_reciprocal_l1144_114459


namespace NUMINAMATH_CALUDE_equality_of_arithmetic_progressions_l1144_114414

theorem equality_of_arithmetic_progressions (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : ∃ r : ℝ, b^2 - a^2 = r ∧ c^2 - b^2 = r ∧ d^2 - c^2 = r)
  (h2 : ∃ s : ℝ, 1/(a+b+d) - 1/(a+b+c) = s ∧ 
               1/(a+c+d) - 1/(a+b+d) = s ∧ 
               1/(b+c+d) - 1/(a+c+d) = s) :
  a = b ∧ b = c ∧ c = d := by
sorry

end NUMINAMATH_CALUDE_equality_of_arithmetic_progressions_l1144_114414


namespace NUMINAMATH_CALUDE_inequality_solution_l1144_114462

theorem inequality_solution (x : ℝ) (h : x ≠ 4) :
  (x^2 - 16) / (x - 4) ≤ 0 ↔ x ∈ Set.Iic (-4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1144_114462


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1144_114455

theorem inequality_solution_set (x : ℝ) : 
  (4 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 9) ↔ 
  (63 / 26 < x ∧ x ≤ 28 / 11) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1144_114455


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1144_114499

/-- 
Given a quadratic equation ax² + bx + c = 0, 
this theorem proves that for the specific equation x² - 3x - 2 = 0, 
the coefficients a, b, and c are 1, -3, and -2 respectively.
-/
theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), 
    (∀ x, a * x^2 + b * x + c = 0 ↔ x^2 - 3*x - 2 = 0) ∧ 
    a = 1 ∧ b = -3 ∧ c = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1144_114499


namespace NUMINAMATH_CALUDE_candy_given_to_haley_l1144_114408

theorem candy_given_to_haley (initial_candy : ℕ) (remaining_candy : ℕ) (candy_given : ℕ) :
  initial_candy = 15 →
  remaining_candy = 9 →
  candy_given = initial_candy - remaining_candy →
  candy_given = 6 := by
sorry

end NUMINAMATH_CALUDE_candy_given_to_haley_l1144_114408


namespace NUMINAMATH_CALUDE_complex_number_real_imag_equal_l1144_114403

theorem complex_number_real_imag_equal (b : ℝ) : 
  let z : ℂ := (1 - b * Complex.I) / (2 + Complex.I)
  z.re = z.im ↔ b = -3 := by sorry

end NUMINAMATH_CALUDE_complex_number_real_imag_equal_l1144_114403


namespace NUMINAMATH_CALUDE_art_probability_correct_l1144_114465

def art_arrangement_probability (total : ℕ) (escher : ℕ) (picasso : ℕ) : ℚ :=
  let other := total - escher - picasso
  let grouped_items := other + 2  -- other items + Escher block + Picasso block
  (grouped_items.factorial * escher.factorial * picasso.factorial : ℚ) / total.factorial

theorem art_probability_correct :
  art_arrangement_probability 12 4 3 = 1 / 660 := by
  sorry

end NUMINAMATH_CALUDE_art_probability_correct_l1144_114465


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1144_114470

theorem diophantine_equation_solution :
  ∃ (a b c d : ℕ), 5 * a + 6 * b + 7 * c + 11 * d = 1999 ∧
  a = 389 ∧ b = 2 ∧ c = 1 ∧ d = 3 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1144_114470


namespace NUMINAMATH_CALUDE_sphere_section_distance_l1144_114445

theorem sphere_section_distance (r : ℝ) (d : ℝ) (A : ℝ) :
  r = 2 →
  A = π →
  d = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_sphere_section_distance_l1144_114445


namespace NUMINAMATH_CALUDE_functional_inequality_solution_l1144_114477

/-- A function from positive reals to positive reals -/
def PositiveRealFunction := {f : ℝ → ℝ // ∀ x > 0, f x > 0}

/-- The functional inequality condition -/
def SatisfiesInequality (f : PositiveRealFunction) : Prop :=
  ∀ x y, x > 0 → y > 0 → f.val (x * y) ≤ (x * f.val y + y * f.val x) / 2

/-- The theorem statement -/
theorem functional_inequality_solution :
  ∀ f : PositiveRealFunction, SatisfiesInequality f →
  ∃ a : ℝ, a > 0 ∧ ∀ x > 0, f.val x = a * x :=
sorry

end NUMINAMATH_CALUDE_functional_inequality_solution_l1144_114477


namespace NUMINAMATH_CALUDE_relationship_abc_l1144_114435

theorem relationship_abc (a b c : ℝ) (ha : a = Real.log 3 / Real.log 0.5)
  (hb : b = Real.sqrt 2) (hc : c = Real.sqrt 0.5) : b > c ∧ c > a := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l1144_114435


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_13_l1144_114420

theorem least_three_digit_multiple_of_13 : ∃ (n : ℕ), n = 104 ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 13 ∣ m → n ≤ m) ∧ 
  100 ≤ n ∧ n < 1000 ∧ 13 ∣ n :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_13_l1144_114420


namespace NUMINAMATH_CALUDE_tangent_line_inclination_angle_range_l1144_114425

open Real Set

theorem tangent_line_inclination_angle_range :
  ∀ x : ℝ, 
  let P : ℝ × ℝ := (x, Real.sin x)
  let θ := Real.arctan (Real.cos x)
  θ ∈ Icc 0 (π/4) ∪ Ico (3*π/4) π := by
sorry

end NUMINAMATH_CALUDE_tangent_line_inclination_angle_range_l1144_114425


namespace NUMINAMATH_CALUDE_max_section_area_is_two_l1144_114413

/-- Represents a cone with its lateral surface unfolded into a sector -/
structure UnfoldedCone where
  radius : ℝ
  centralAngle : ℝ

/-- Calculates the maximum area of a section determined by two generatrices of the cone -/
def maxSectionArea (cone : UnfoldedCone) : ℝ :=
  sorry

/-- Theorem stating that for a cone with lateral surface unfolded into a sector
    with radius 2 and central angle 5π/3, the maximum section area is 2 -/
theorem max_section_area_is_two :
  let cone : UnfoldedCone := ⟨2, 5 * Real.pi / 3⟩
  maxSectionArea cone = 2 :=
sorry

end NUMINAMATH_CALUDE_max_section_area_is_two_l1144_114413


namespace NUMINAMATH_CALUDE_units_digit_153_base3_l1144_114411

/-- Converts a natural number to its base 3 representation -/
def toBase3 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 3) ((m % 3) :: acc)
    aux n []

/-- The units digit is the last digit in the list representation -/
def unitsDigit (digits : List ℕ) : ℕ :=
  match digits.reverse with
  | [] => 0
  | d :: _ => d

theorem units_digit_153_base3 :
  unitsDigit (toBase3 153) = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_153_base3_l1144_114411


namespace NUMINAMATH_CALUDE_division_remainder_l1144_114492

-- Define the dividend polynomial
def dividend (x : ℝ) : ℝ := 3*x^5 + 2*x^4 - 5*x^3 + 6*x - 8

-- Define the divisor polynomial
def divisor (x : ℝ) : ℝ := x^2 + 3*x + 2

-- Define the remainder polynomial
def remainder (x : ℝ) : ℝ := 34*x + 24

-- Theorem statement
theorem division_remainder :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, dividend x = divisor x * q x + remainder x :=
sorry

end NUMINAMATH_CALUDE_division_remainder_l1144_114492


namespace NUMINAMATH_CALUDE_count_six_digit_integers_l1144_114487

def digit_set : Multiset ℕ := {1, 1, 2, 3, 3, 3}

/-- The number of different positive, six-digit integers formed from the given digit set -/
def num_six_digit_integers : ℕ := sorry

theorem count_six_digit_integers : num_six_digit_integers = 60 := by sorry

end NUMINAMATH_CALUDE_count_six_digit_integers_l1144_114487


namespace NUMINAMATH_CALUDE_parentheses_number_l1144_114415

theorem parentheses_number (x : ℤ) : x - (-6) = 20 → x = 14 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_number_l1144_114415


namespace NUMINAMATH_CALUDE_solution_to_system_l1144_114478

theorem solution_to_system : ∃ (x y : ℚ), 
  (7 * x - 50 * y = 3) ∧ (3 * y - x = 5) ∧ 
  (x = -259/29) ∧ (y = -38/29) := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_l1144_114478


namespace NUMINAMATH_CALUDE_chocolate_bars_distribution_l1144_114419

theorem chocolate_bars_distribution (total_bars : ℕ) (num_people : ℕ) 
  (h1 : total_bars = 12) (h2 : num_people = 3) :
  2 * (total_bars / num_people) = 8 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_distribution_l1144_114419


namespace NUMINAMATH_CALUDE_petyas_torn_sheets_l1144_114461

/-- Represents a book with consecutively numbered pages -/
structure Book where
  firstTornPage : ℕ
  lastTornPage : ℕ

/-- Checks if two numbers have the same digits -/
def sameDigits (a b : ℕ) : Prop := sorry

/-- Calculates the number of sheets torn out from a book -/
def sheetsTornOut (book : Book) : ℕ := sorry

/-- Theorem stating the number of sheets torn out by Petya -/
theorem petyas_torn_sheets (book : Book) : 
  book.firstTornPage = 185 ∧ 
  sameDigits book.firstTornPage book.lastTornPage ∧
  book.lastTornPage > book.firstTornPage ∧
  Even book.lastTornPage →
  sheetsTornOut book = 167 := by
  sorry

end NUMINAMATH_CALUDE_petyas_torn_sheets_l1144_114461


namespace NUMINAMATH_CALUDE_expression_equality_l1144_114479

theorem expression_equality : (40 - (2040 - 210)) + (2040 - (210 - 40)) = 80 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1144_114479


namespace NUMINAMATH_CALUDE_outfits_count_l1144_114486

/-- The number of different outfits that can be formed by choosing one top and one pair of pants -/
def number_of_outfits (num_tops : ℕ) (num_pants : ℕ) : ℕ :=
  num_tops * num_pants

/-- Theorem stating that with 4 tops and 3 pants, the number of outfits is 12 -/
theorem outfits_count : number_of_outfits 4 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l1144_114486


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1144_114428

theorem max_value_of_expression (m n : ℝ) (hm : m > 0) (hn : n < 0) (h : 1/m + 1/n = 1) :
  ∃ (x : ℝ), ∀ (m' n' : ℝ), m' > 0 → n' < 0 → 1/m' + 1/n' = 1 → 4*m' + n' ≤ x ∧ 4*m + n = x :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1144_114428


namespace NUMINAMATH_CALUDE_sequence_properties_l1144_114439

-- Define the sequence a_n
def a : ℕ → ℤ
| n => if n ≤ 4 then n - 4 else 2^(n-5)

-- State the theorem
theorem sequence_properties :
  -- Conditions
  (a 2 = -2) ∧
  (a 7 = 4) ∧
  (∀ n ≤ 6, a (n+1) - a n = a (n+2) - a (n+1)) ∧
  (∀ n ≥ 5, (a (n+1))^2 = a n * a (n+2)) ∧
  -- Conclusions
  (∀ n, a n = if n ≤ 4 then n - 4 else 2^(n-5)) ∧
  (∀ m : ℕ, m > 0 → (a m + a (m+1) + a (m+2) = a m * a (m+1) * a (m+2) ↔ m = 1 ∨ m = 3)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1144_114439


namespace NUMINAMATH_CALUDE_faye_pencils_and_crayons_l1144_114438

/-- Given that Faye arranges her pencils and crayons in 11 rows,
    with 31 pencils and 27 crayons in each row,
    prove that she has 638 pencils and crayons in total. -/
theorem faye_pencils_and_crayons (rows : ℕ) (pencils_per_row : ℕ) (crayons_per_row : ℕ)
    (h1 : rows = 11)
    (h2 : pencils_per_row = 31)
    (h3 : crayons_per_row = 27) :
    rows * pencils_per_row + rows * crayons_per_row = 638 := by
  sorry

end NUMINAMATH_CALUDE_faye_pencils_and_crayons_l1144_114438


namespace NUMINAMATH_CALUDE_EQ_equals_15_l1144_114409

/-- Represents a trapezoid EFGH with a circle tangent to its sides -/
structure TrapezoidWithTangentCircle where
  -- Length of side EF
  EF : ℝ
  -- Length of side FG
  FG : ℝ
  -- Length of side GH
  GH : ℝ
  -- Length of side HE
  HE : ℝ
  -- Assumption that EF is parallel to GH
  EF_parallel_GH : True
  -- Assumption that there exists a circle with center Q on EF tangent to FG and HE
  circle_tangent : True

/-- Theorem stating that EQ = 15 in the given trapezoid configuration -/
theorem EQ_equals_15 (t : TrapezoidWithTangentCircle)
  (h1 : t.EF = 137)
  (h2 : t.FG = 75)
  (h3 : t.GH = 28)
  (h4 : t.HE = 105) :
  ∃ Q : ℝ, Q = 15 ∧ Q ≤ t.EF := by
  sorry

end NUMINAMATH_CALUDE_EQ_equals_15_l1144_114409


namespace NUMINAMATH_CALUDE_f_minimum_and_equal_values_l1144_114410

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x

noncomputable def h (t : ℝ) : ℝ :=
  if t ≤ -1 then t * Real.exp (t + 2)
  else if t ≤ 1 then -Real.exp 1
  else (t - 2) * Real.exp t

theorem f_minimum_and_equal_values :
  (∀ t : ℝ, ∀ x ∈ Set.Icc t (t + 2), f x ≥ h t) ∧
  (∀ α β : ℝ, α ≠ β → f α = f β → α + β < 2) := by sorry

end NUMINAMATH_CALUDE_f_minimum_and_equal_values_l1144_114410


namespace NUMINAMATH_CALUDE_unique_solution_l1144_114416

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: 36018 is the only positive integer m that satisfies 2001 * S(m) = m -/
theorem unique_solution :
  ∀ m : ℕ, m > 0 → (2001 * sumOfDigits m = m) ↔ m = 36018 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l1144_114416


namespace NUMINAMATH_CALUDE_water_tank_capacity_l1144_114460

theorem water_tank_capacity : ∀ C : ℝ, 
  (0.4 * C - 0.1 * C = 36) → C = 120 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l1144_114460


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_primes_l1144_114424

theorem smallest_five_digit_divisible_by_primes : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧
  (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ 2 ∣ m ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m ∧ 11 ∣ m → n ≤ m) ∧
  2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n ∧
  n = 11550 :=
by sorry

#check smallest_five_digit_divisible_by_primes

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_primes_l1144_114424


namespace NUMINAMATH_CALUDE_quadrant_crossing_linear_function_y_intercept_positive_l1144_114456

/-- A linear function passing through the first, second, and third quadrants -/
structure QuadrantCrossingLinearFunction where
  b : ℝ
  passes_first_quadrant : ∃ x y, x > 0 ∧ y > 0 ∧ y = x + b
  passes_second_quadrant : ∃ x y, x < 0 ∧ y > 0 ∧ y = x + b
  passes_third_quadrant : ∃ x y, x < 0 ∧ y < 0 ∧ y = x + b

/-- The y-intercept of a quadrant crossing linear function is positive -/
theorem quadrant_crossing_linear_function_y_intercept_positive
  (f : QuadrantCrossingLinearFunction) : f.b > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadrant_crossing_linear_function_y_intercept_positive_l1144_114456


namespace NUMINAMATH_CALUDE_race_distance_l1144_114442

/-- A race between two runners p and q, where p is faster but q gets a head start -/
structure Race where
  /-- The speed of runner q (in meters per second) -/
  q_speed : ℝ
  /-- The speed of runner p (in meters per second) -/
  p_speed : ℝ
  /-- The head start given to runner q (in meters) -/
  head_start : ℝ
  /-- The condition that p is 25% faster than q -/
  speed_ratio : p_speed = 1.25 * q_speed
  /-- The head start is 60 meters -/
  head_start_value : head_start = 60

/-- The theorem stating that if the race ends in a tie, p ran 300 meters -/
theorem race_distance (race : Race) : 
  (∃ t : ℝ, race.q_speed * t = race.p_speed * t - race.head_start) → 
  race.p_speed * (300 / race.p_speed) = 300 := by
  sorry

#check race_distance

end NUMINAMATH_CALUDE_race_distance_l1144_114442


namespace NUMINAMATH_CALUDE_toy_store_revenue_ratio_l1144_114453

theorem toy_store_revenue_ratio :
  ∀ (november december january : ℝ),
  january = (1 / 6) * november →
  december = 2.857142857142857 * (november + january) / 2 →
  november / december = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_toy_store_revenue_ratio_l1144_114453


namespace NUMINAMATH_CALUDE_theater_ticket_profit_l1144_114476

/-- Calculates the total profit from ticket sales given the ticket prices and quantities sold. -/
theorem theater_ticket_profit
  (adult_price : ℕ)
  (kid_price : ℕ)
  (total_tickets : ℕ)
  (kid_tickets : ℕ)
  (h1 : adult_price = 6)
  (h2 : kid_price = 2)
  (h3 : total_tickets = 175)
  (h4 : kid_tickets = 75) :
  (total_tickets - kid_tickets) * adult_price + kid_tickets * kid_price = 750 :=
by sorry


end NUMINAMATH_CALUDE_theater_ticket_profit_l1144_114476


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l1144_114484

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → y = 0 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l1144_114484


namespace NUMINAMATH_CALUDE_m_values_l1144_114423

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}

theorem m_values (m : ℝ) : (A ∪ B m = A) ↔ m ∈ ({0, -1/2, 1/3} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_m_values_l1144_114423


namespace NUMINAMATH_CALUDE_interval_length_theorem_l1144_114412

theorem interval_length_theorem (c d : ℝ) : 
  (∃ (x : ℝ), c ≤ 3 * x + 5 ∧ 3 * x + 5 ≤ d) → 
  ((d - 5) / 3 - (c - 5) / 3 = 15) → 
  d - c = 45 := by
sorry

end NUMINAMATH_CALUDE_interval_length_theorem_l1144_114412


namespace NUMINAMATH_CALUDE_quadrilateral_with_equal_incircle_radii_is_rhombus_l1144_114450

/-- A point in a 2D plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- The intersection point of the diagonals of a quadrilateral -/
def diagonalIntersection (q : Quadrilateral) : Point :=
  sorry

/-- The radius of the incircle of a triangle -/
def incircleRadius (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop :=
  sorry

/-- Checks if a quadrilateral is a rhombus -/
def isRhombus (q : Quadrilateral) : Prop :=
  sorry

theorem quadrilateral_with_equal_incircle_radii_is_rhombus
  (q : Quadrilateral)
  (h_convex : isConvex q)
  (O : Point)
  (h_O : O = diagonalIntersection q)
  (h_radii : incircleRadius q.A q.B O = incircleRadius q.B q.C O ∧
             incircleRadius q.B q.C O = incircleRadius q.C q.D O ∧
             incircleRadius q.C q.D O = incircleRadius q.D q.A O) :
  isRhombus q :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_with_equal_incircle_radii_is_rhombus_l1144_114450


namespace NUMINAMATH_CALUDE_set_operations_and_equality_l1144_114405

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}

theorem set_operations_and_equality :
  (∃ m : ℝ, 
    (A ∩ B m = {x | 3 ≤ x ∧ x ≤ 5} ∧
     (Set.univ \ A) ∪ B m = {x | x < 2 ∨ x ≥ 3})) ∧
  (∀ m : ℝ, A = B m ↔ 2 ≤ m ∧ m ≤ 3) := by sorry

end NUMINAMATH_CALUDE_set_operations_and_equality_l1144_114405


namespace NUMINAMATH_CALUDE_evaluate_expression_l1144_114441

theorem evaluate_expression : 6 - 8 * (9 - 4^2) / 2 = 34 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1144_114441


namespace NUMINAMATH_CALUDE_diamond_self_not_always_zero_l1144_114421

-- Define the diamond operator
def diamond (x y : ℝ) : ℝ := |x - 2*y|

-- Theorem stating that the statement "For all real x, x ◇ x = 0" is false
theorem diamond_self_not_always_zero : ¬ ∀ x : ℝ, diamond x x = 0 := by
  sorry

end NUMINAMATH_CALUDE_diamond_self_not_always_zero_l1144_114421


namespace NUMINAMATH_CALUDE_expression_equals_one_l1144_114474

theorem expression_equals_one : 
  (150^2 - 9^2) / (110^2 - 13^2) * ((110 - 13) * (110 + 13)) / ((150 - 9) * (150 + 9)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l1144_114474


namespace NUMINAMATH_CALUDE_combined_mean_of_sets_l1144_114493

theorem combined_mean_of_sets (set1_count : ℕ) (set1_mean : ℚ) (set2_count : ℕ) (set2_mean : ℚ) 
  (new_set1_count : ℕ) (new_set1_mean : ℚ) :
  set1_count = 7 →
  set1_mean = 15 →
  set2_count = 8 →
  set2_mean = 21 →
  new_set1_count = set1_count + 1 →
  new_set1_mean = 16 →
  let total_count := new_set1_count + set2_count
  let total_sum := new_set1_mean * new_set1_count + set2_mean * set2_count
  (total_sum / total_count : ℚ) = 37/2 := by
sorry

end NUMINAMATH_CALUDE_combined_mean_of_sets_l1144_114493


namespace NUMINAMATH_CALUDE_bake_sale_pastries_sold_l1144_114482

/-- Represents the number of pastries sold at a bake sale. -/
def pastries_sold (cupcakes cookies taken_home : ℕ) : ℕ :=
  cupcakes + cookies - taken_home

/-- Proves that the number of pastries sold is correct given the conditions. -/
theorem bake_sale_pastries_sold :
  pastries_sold 4 29 24 = 9 := by
  sorry

end NUMINAMATH_CALUDE_bake_sale_pastries_sold_l1144_114482


namespace NUMINAMATH_CALUDE_difference_max_min_change_l1144_114452

def initial_yes : ℝ := 40
def initial_no : ℝ := 30
def initial_maybe : ℝ := 30
def final_yes : ℝ := 60
def final_no : ℝ := 20
def final_maybe : ℝ := 20

def min_change : ℝ := 20
def max_change : ℝ := 40

theorem difference_max_min_change :
  max_change - min_change = 20 :=
sorry

end NUMINAMATH_CALUDE_difference_max_min_change_l1144_114452


namespace NUMINAMATH_CALUDE_unknown_number_in_set_l1144_114431

theorem unknown_number_in_set (x : ℝ) : 
  ((14 + 32 + 53) / 3 = (21 + x + 22) / 3 + 3) → x = 47 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_in_set_l1144_114431


namespace NUMINAMATH_CALUDE_a_share_in_profit_l1144_114443

/-- Given the investments of A, B, and C, and the total profit, prove A's share in the profit --/
theorem a_share_in_profit 
  (a_investment b_investment c_investment total_profit : ℕ) 
  (h1 : a_investment = 2400)
  (h2 : b_investment = 7200)
  (h3 : c_investment = 9600)
  (h4 : total_profit = 9000) :
  a_investment * total_profit / (a_investment + b_investment + c_investment) = 1125 := by
  sorry

end NUMINAMATH_CALUDE_a_share_in_profit_l1144_114443


namespace NUMINAMATH_CALUDE_sequence_inequality_l1144_114498

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℝ := 3 * n - 2 * n^2

/-- The n-th term of the sequence -/
def a (n : ℕ) : ℝ := S n - S (n - 1)

theorem sequence_inequality (n : ℕ) (h : n ≥ 2) :
  n * (a 1) > S n ∧ S n > n * (a n) := by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1144_114498


namespace NUMINAMATH_CALUDE_original_apple_price_l1144_114440

/-- The original price of apples per pound -/
def original_price : ℝ := sorry

/-- The price increase percentage -/
def price_increase : ℝ := 0.25

/-- The new price of apples per pound after the increase -/
def new_price : ℝ := original_price * (1 + price_increase)

/-- The total weight of apples bought -/
def total_weight : ℝ := 8

/-- The total cost of apples after the price increase -/
def total_cost : ℝ := 64

theorem original_apple_price :
  new_price * total_weight = total_cost →
  original_price = 6.40 := by sorry

end NUMINAMATH_CALUDE_original_apple_price_l1144_114440


namespace NUMINAMATH_CALUDE_correlation_relationships_l1144_114402

/-- Represents a relationship between variables -/
inductive Relationship
  | CubeVolumeEdge
  | PointOnCurve
  | AppleProductionClimate
  | TreeDiameterHeight

/-- Defines whether a relationship is a correlation relationship -/
def isCorrelationRelationship (r : Relationship) : Prop :=
  match r with
  | Relationship.AppleProductionClimate => True
  | Relationship.TreeDiameterHeight => True
  | _ => False

/-- Theorem stating that only AppleProductionClimate and TreeDiameterHeight are correlation relationships -/
theorem correlation_relationships :
  ∀ r : Relationship,
    isCorrelationRelationship r ↔ (r = Relationship.AppleProductionClimate ∨ r = Relationship.TreeDiameterHeight) :=
by sorry

end NUMINAMATH_CALUDE_correlation_relationships_l1144_114402


namespace NUMINAMATH_CALUDE_stadium_entry_fee_l1144_114449

/-- Proves that the entry fee per person is $20 given the stadium conditions --/
theorem stadium_entry_fee (capacity : ℕ) (occupancy_ratio : ℚ) (fee_difference : ℕ) :
  capacity = 2000 →
  occupancy_ratio = 3/4 →
  fee_difference = 10000 →
  ∃ (fee : ℚ), fee = 20 ∧
    (capacity : ℚ) * fee - (capacity : ℚ) * occupancy_ratio * fee = fee_difference :=
by sorry

end NUMINAMATH_CALUDE_stadium_entry_fee_l1144_114449


namespace NUMINAMATH_CALUDE_CH₄_has_most_atoms_l1144_114404

-- Define the molecules and their atom counts
def O₂_atoms : ℕ := 2
def NH₃_atoms : ℕ := 4
def CO_atoms : ℕ := 2
def CH₄_atoms : ℕ := 5

-- Define a function to compare atom counts
def has_more_atoms (a b : ℕ) : Prop := a > b

-- Theorem statement
theorem CH₄_has_most_atoms :
  has_more_atoms CH₄_atoms O₂_atoms ∧
  has_more_atoms CH₄_atoms NH₃_atoms ∧
  has_more_atoms CH₄_atoms CO_atoms :=
by sorry

end NUMINAMATH_CALUDE_CH₄_has_most_atoms_l1144_114404


namespace NUMINAMATH_CALUDE_sum_remainder_mod_seven_l1144_114475

theorem sum_remainder_mod_seven : (9^5 + 8^6 + 7^7) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_seven_l1144_114475


namespace NUMINAMATH_CALUDE_sugar_mixture_percentage_l1144_114406

/-- Given two solutions, where one fourth of the first solution is replaced by the second solution,
    resulting in a mixture that is 17% sugar, and the second solution is 38% sugar,
    prove that the first solution was 10% sugar. -/
theorem sugar_mixture_percentage (first_solution second_solution final_mixture : ℝ) 
    (h1 : 3/4 * first_solution + 1/4 * second_solution = final_mixture)
    (h2 : final_mixture = 17)
    (h3 : second_solution = 38) :
    first_solution = 10 := by
  sorry

end NUMINAMATH_CALUDE_sugar_mixture_percentage_l1144_114406


namespace NUMINAMATH_CALUDE_car_journey_speed_l1144_114463

/-- Proves that given specific conditions about a car's journey, 
    the speed for the remaining part of the trip is 60 mph. -/
theorem car_journey_speed (D : ℝ) (h1 : D > 0) : 
  let first_part_distance := 0.4 * D
  let first_part_speed := 40
  let total_average_speed := 50
  let remaining_part_distance := 0.6 * D
  let remaining_part_speed := 
    remaining_part_distance / 
    (D / total_average_speed - first_part_distance / first_part_speed)
  remaining_part_speed = 60 := by
  sorry


end NUMINAMATH_CALUDE_car_journey_speed_l1144_114463


namespace NUMINAMATH_CALUDE_total_pastries_is_97_l1144_114417

/-- Given the number of pastries for Grace, calculate the total number of pastries for Grace, Calvin, Phoebe, and Frank. -/
def totalPastries (grace : ℕ) : ℕ :=
  let calvin := grace - 5
  let phoebe := grace - 5
  let frank := calvin - 8
  grace + calvin + phoebe + frank

/-- Theorem stating that given Grace has 30 pastries, the total number of pastries for all four is 97. -/
theorem total_pastries_is_97 : totalPastries 30 = 97 := by
  sorry

#eval totalPastries 30

end NUMINAMATH_CALUDE_total_pastries_is_97_l1144_114417


namespace NUMINAMATH_CALUDE_one_third_of_cake_flour_one_third_of_cake_flour_mixed_number_l1144_114468

theorem one_third_of_cake_flour :
  let full_recipe : ℚ := 19 / 3
  let one_third_recipe : ℚ := full_recipe / 3
  one_third_recipe = 19 / 9 :=
by sorry

-- Convert to mixed number
theorem one_third_of_cake_flour_mixed_number :
  let full_recipe : ℚ := 19 / 3
  let one_third_recipe : ℚ := full_recipe / 3
  ∃ (whole : ℕ) (numerator : ℕ) (denominator : ℕ),
    one_third_recipe = whole + (numerator : ℚ) / denominator ∧
    whole = 2 ∧ numerator = 1 ∧ denominator = 9 :=
by sorry

end NUMINAMATH_CALUDE_one_third_of_cake_flour_one_third_of_cake_flour_mixed_number_l1144_114468


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l1144_114430

theorem line_segment_endpoint (x : ℝ) : 
  (∃ (y : ℝ), (x - 3)^2 + (y - (-1))^2 = 15^2 ∧ 
               y = 7 ∧ 
               (y - (-1)) / (x - 3) = 1) →
  (x - 3)^2 + 64 = 225 :=
by sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l1144_114430


namespace NUMINAMATH_CALUDE_sequence_strictly_increasing_l1144_114400

theorem sequence_strictly_increasing (n : ℕ) (h : n ≥ 14) : 
  let a : ℕ → ℤ := λ k => k^4 - 20*k^2 - 10*k + 1
  a n > a (n-1) := by sorry

end NUMINAMATH_CALUDE_sequence_strictly_increasing_l1144_114400


namespace NUMINAMATH_CALUDE_motion_equation_l1144_114490

/-- Given V = gt + V₀ and S = (1/2)gt² + V₀t + kt³, 
    prove that t = (2S(V-V₀)) / (V² - V₀² + 2k(V-V₀)²) -/
theorem motion_equation (g k V V₀ S t : ℝ) 
  (hV : V = g * t + V₀)
  (hS : S = (1/2) * g * t^2 + V₀ * t + k * t^3) :
  t = (2 * S * (V - V₀)) / (V^2 - V₀^2 + 2 * k * (V - V₀)^2) :=
by sorry

end NUMINAMATH_CALUDE_motion_equation_l1144_114490


namespace NUMINAMATH_CALUDE_markup_calculation_l1144_114473

theorem markup_calculation (purchase_price : ℝ) (overhead_percentage : ℝ) (net_profit : ℝ) : 
  purchase_price = 48 →
  overhead_percentage = 0.25 →
  net_profit = 12 →
  purchase_price + purchase_price * overhead_percentage + net_profit - purchase_price = 24 := by
sorry

end NUMINAMATH_CALUDE_markup_calculation_l1144_114473


namespace NUMINAMATH_CALUDE_g_properties_l1144_114458

-- Define g as a function from real numbers to real numbers
variable (g : ℝ → ℝ)

-- Define the properties of g
axiom g_positive : ∀ x, g x > 0
axiom g_sum_property : ∀ a b, g a + g b = g (a + b + 1)

-- State the theorem
theorem g_properties :
  (∃ k : ℝ, k > 0 ∧ g 0 = k) ∧
  (∃ a : ℝ, g (-a) ≠ 1 - g a) :=
sorry

end NUMINAMATH_CALUDE_g_properties_l1144_114458


namespace NUMINAMATH_CALUDE_intersection_equals_T_l1144_114433

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_equals_T : S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_intersection_equals_T_l1144_114433


namespace NUMINAMATH_CALUDE_dividend_calculation_l1144_114432

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 16) 
  (h2 : quotient = 8) 
  (h3 : remainder = 4) : 
  divisor * quotient + remainder = 132 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1144_114432


namespace NUMINAMATH_CALUDE_product_divisibility_probability_l1144_114407

/-- The number of dice rolled -/
def n : ℕ := 8

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The probability that a single die roll is even -/
def p_even : ℚ := 1/2

/-- The probability that a single die roll is divisible by 3 -/
def p_div3 : ℚ := 1/3

/-- The probability that the product of n dice rolls is divisible by both 4 and 3 -/
def prob_div_4_and_3 : ℚ := 1554975/1679616

theorem product_divisibility_probability :
  (1 - (1 - (1 - (1 - p_even)^n - n * (1 - p_even)^(n-1) * p_even))) *
  (1 - (1 - p_div3)^n) = prob_div_4_and_3 := by
  sorry

end NUMINAMATH_CALUDE_product_divisibility_probability_l1144_114407


namespace NUMINAMATH_CALUDE_number_of_lists_18_4_l1144_114447

/-- The number of elements in the set of balls -/
def n : ℕ := 18

/-- The number of draws -/
def k : ℕ := 4

/-- The number of possible lists when drawing k times with replacement from a set of n elements -/
def number_of_lists (n k : ℕ) : ℕ := n^k

/-- Theorem: The number of possible lists when drawing 4 times with replacement from a set of 18 elements is 104,976 -/
theorem number_of_lists_18_4 : number_of_lists n k = 104976 := by
  sorry

end NUMINAMATH_CALUDE_number_of_lists_18_4_l1144_114447


namespace NUMINAMATH_CALUDE_integer_roots_conditions_l1144_114481

theorem integer_roots_conditions (p q : ℤ) : 
  (∃ (a b c d : ℤ), (∀ x : ℤ, x^4 + 2*p*x^2 + q*x + p^2 - 36 = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) ∧
  (a + b + c + d = 0) ∧
  (a*b + a*c + a*d + b*c + b*d + c*d = 2*p) ∧
  (a*b*c*d = p^2 - 36)) →
  ∃ (x y z : ℕ), 18 = 2*x^2 + y^2 + z^2 ∧ 
  ((x = 0 ∧ y = 3 ∧ z = 3) ∨
   (x = 1 ∧ y = 4 ∧ z = 0) ∨
   (x = 1 ∧ y = 0 ∧ z = 4) ∨
   (x = 2 ∧ y = 3 ∧ z = 1) ∨
   (x = 2 ∧ y = 1 ∧ z = 3) ∨
   (x = 3 ∧ y = 0 ∧ z = 0)) :=
by sorry

end NUMINAMATH_CALUDE_integer_roots_conditions_l1144_114481
