import Mathlib

namespace NUMINAMATH_CALUDE_min_value_quadratic_l485_48577

theorem min_value_quadratic (x : ℝ) : 
  ∃ (m : ℝ), m = -5 ∧ ∀ x, x^2 + 2*x - 4 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l485_48577


namespace NUMINAMATH_CALUDE_union_A_B_when_a_is_one_range_of_a_for_necessary_not_sufficient_l485_48599

-- Define set A
def A (a : ℝ) : Set ℝ := {x | (x - a) / (x - 3*a) < 0}

-- Define set B
def B : Set ℝ := {x | x^2 - 5*x + 6 < 0}

-- Theorem for part (1)
theorem union_A_B_when_a_is_one : 
  A 1 ∪ B = {x | 1 < x ∧ x < 3} := by sorry

-- Theorem for part (2)
theorem range_of_a_for_necessary_not_sufficient : 
  {a : ℝ | ∀ x, x ∈ B → x ∈ A a ∧ ∃ y, y ∈ A a ∧ y ∉ B} = {a | 1 ≤ a ∧ a ≤ 2} := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_a_is_one_range_of_a_for_necessary_not_sufficient_l485_48599


namespace NUMINAMATH_CALUDE_batsman_innings_count_l485_48554

theorem batsman_innings_count
  (avg : ℝ)
  (score_diff : ℕ)
  (avg_excluding : ℝ)
  (highest_score : ℕ)
  (h_avg : avg = 60)
  (h_score_diff : score_diff = 150)
  (h_avg_excluding : avg_excluding = 58)
  (h_highest_score : highest_score = 179)
  : ∃ n : ℕ, n = 46 ∧ 
    avg * n = avg_excluding * (n - 2) + highest_score + (highest_score - score_diff) :=
by sorry

end NUMINAMATH_CALUDE_batsman_innings_count_l485_48554


namespace NUMINAMATH_CALUDE_complex_magnitude_power_l485_48516

theorem complex_magnitude_power : 
  Complex.abs ((2 : ℂ) + (2 * Complex.I * Real.sqrt 3)) ^ 6 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_power_l485_48516


namespace NUMINAMATH_CALUDE_books_left_to_read_l485_48567

theorem books_left_to_read (total_books read_books : ℕ) : 
  total_books = 13 → read_books = 9 → total_books - read_books = 4 := by
  sorry

end NUMINAMATH_CALUDE_books_left_to_read_l485_48567


namespace NUMINAMATH_CALUDE_twenty_one_in_fibonacci_l485_48596

def fibonacci : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

theorem twenty_one_in_fibonacci : ∃ n : ℕ, fibonacci n = 21 := by
  sorry

end NUMINAMATH_CALUDE_twenty_one_in_fibonacci_l485_48596


namespace NUMINAMATH_CALUDE_bernard_white_notebooks_l485_48588

/-- The number of white notebooks Bernard had -/
def white_notebooks : ℕ := sorry

/-- The number of red notebooks Bernard had -/
def red_notebooks : ℕ := 15

/-- The number of blue notebooks Bernard had -/
def blue_notebooks : ℕ := 17

/-- The number of notebooks Bernard gave to Tom -/
def notebooks_given : ℕ := 46

/-- The number of notebooks Bernard had left -/
def notebooks_left : ℕ := 5

/-- The total number of notebooks Bernard originally had -/
def total_notebooks : ℕ := notebooks_given + notebooks_left

theorem bernard_white_notebooks : 
  white_notebooks = total_notebooks - (red_notebooks + blue_notebooks) ∧ 
  white_notebooks = 19 := by sorry

end NUMINAMATH_CALUDE_bernard_white_notebooks_l485_48588


namespace NUMINAMATH_CALUDE_triangle_problem_l485_48576

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a * Real.cos B * Real.cos C + b * Real.cos A * Real.cos C = c / 2 →
  c = Real.sqrt 7 →
  a + b = 5 →
  C = π / 3 ∧ (1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l485_48576


namespace NUMINAMATH_CALUDE_johns_hat_cost_l485_48598

/-- The total cost of John's hats -/
def total_cost (weeks : ℕ) (odd_cost even_cost : ℕ) : ℕ :=
  let total_days := weeks * 7
  let odd_days := total_days / 2
  let even_days := total_days / 2
  odd_days * odd_cost + even_days * even_cost

/-- Theorem stating that the total cost of John's hats is $7350 -/
theorem johns_hat_cost :
  total_cost 20 45 60 = 7350 := by
  sorry

end NUMINAMATH_CALUDE_johns_hat_cost_l485_48598


namespace NUMINAMATH_CALUDE_cubic_polynomial_factor_l485_48549

/-- Given a cubic polynomial of the form 3x^3 - dx + 18 with a quadratic factor x^2 + qx + 2,
    prove that d = -6 -/
theorem cubic_polynomial_factor (d : ℝ) : 
  (∃ q : ℝ, ∃ m : ℝ, ∀ x : ℝ, 
    3 * x^3 - d * x + 18 = (x^2 + q * x + 2) * (m * x)) → 
  d = -6 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_factor_l485_48549


namespace NUMINAMATH_CALUDE_painting_class_combinations_l485_48591

theorem painting_class_combinations : Nat.choose 10 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_painting_class_combinations_l485_48591


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l485_48569

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 6 + a 8 = 10) 
  (h_a3 : a 3 = 1) : 
  a 11 = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l485_48569


namespace NUMINAMATH_CALUDE_custom_op_result_l485_48590

-- Define the custom operation
def custom_op (a b : ℚ) : ℚ := (a + b) / (a - b - 1)

-- State the theorem
theorem custom_op_result : custom_op (custom_op 7 5) 2 = 14 / 9 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_result_l485_48590


namespace NUMINAMATH_CALUDE_blue_paint_gallons_l485_48582

theorem blue_paint_gallons (total : ℕ) (white : ℕ) (blue : ℕ) :
  total = 6689 →
  white + blue = total →
  8 * white = 5 * blue →
  blue = 4116 := by
sorry

end NUMINAMATH_CALUDE_blue_paint_gallons_l485_48582


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_18_with_sqrt_between_26_and_26_2_l485_48512

theorem unique_integer_divisible_by_18_with_sqrt_between_26_and_26_2 :
  ∃! (N : ℕ), 
    N > 0 ∧ 
    N % 18 = 0 ∧ 
    (26 : ℝ) < Real.sqrt N ∧ 
    Real.sqrt N < 26.2 ∧ 
    N = 684 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_18_with_sqrt_between_26_and_26_2_l485_48512


namespace NUMINAMATH_CALUDE_probability_diamond_spade_heart_value_l485_48520

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of cards of each suit in a standard deck -/
def CardsPerSuit : ℕ := 13

/-- Probability of drawing a diamond, then a spade, then a heart from a standard deck -/
def probability_diamond_spade_heart : ℚ :=
  (CardsPerSuit / StandardDeck) *
  (CardsPerSuit / (StandardDeck - 1)) *
  (CardsPerSuit / (StandardDeck - 2))

/-- Theorem stating the probability of drawing a diamond, then a spade, then a heart -/
theorem probability_diamond_spade_heart_value :
  probability_diamond_spade_heart = 169 / 10200 := by
  sorry

end NUMINAMATH_CALUDE_probability_diamond_spade_heart_value_l485_48520


namespace NUMINAMATH_CALUDE_negation_equivalence_l485_48546

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x + |x| < 0) ↔ (∀ x : ℝ, x + |x| ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l485_48546


namespace NUMINAMATH_CALUDE_first_day_exceeding_500_l485_48501

def algae_population (n : ℕ) : ℕ := 5 * 3^n

theorem first_day_exceeding_500 :
  (∀ k : ℕ, k < 5 → algae_population k ≤ 500) ∧
  algae_population 5 > 500 :=
sorry

end NUMINAMATH_CALUDE_first_day_exceeding_500_l485_48501


namespace NUMINAMATH_CALUDE_geometric_arithmetic_mean_sum_l485_48556

theorem geometric_arithmetic_mean_sum (a b c x y : ℝ) 
  (h1 : b ^ 2 = a * c)  -- geometric sequence condition
  (h2 : x ≠ 0)
  (h3 : y ≠ 0)
  (h4 : 2 * x = a + b)  -- arithmetic mean condition
  (h5 : 2 * y = b + c)  -- arithmetic mean condition
  : a / x + c / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_mean_sum_l485_48556


namespace NUMINAMATH_CALUDE_simplified_A_value_l485_48518

theorem simplified_A_value (a : ℝ) : 
  let A := (a - 1) / (a + 2) * ((a^2 - 4) / (a^2 - 2*a + 1)) / (1 / (a - 1))
  (a^2 - a = 0) → A = -2 := by
  sorry

end NUMINAMATH_CALUDE_simplified_A_value_l485_48518


namespace NUMINAMATH_CALUDE_fraction_value_l485_48528

theorem fraction_value (x y : ℝ) (h1 : 4 < (2*x - 3*y) / (2*x + 3*y)) 
  (h2 : (2*x - 3*y) / (2*x + 3*y) < 8) (h3 : ∃ (n : ℤ), x/y = n) : 
  x/y = -2 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l485_48528


namespace NUMINAMATH_CALUDE_cube_and_square_root_problem_l485_48571

theorem cube_and_square_root_problem (a b : ℝ) :
  (2*b - 2*a)^(1/3) = -2 →
  (4*a + 3*b)^(1/2) = 3 →
  (a = 3 ∧ b = -1) ∧ ((5*a - b)^(1/2) = 4 ∨ (5*a - b)^(1/2) = -4) :=
by sorry

end NUMINAMATH_CALUDE_cube_and_square_root_problem_l485_48571


namespace NUMINAMATH_CALUDE_disjoint_quadratic_sets_l485_48566

theorem disjoint_quadratic_sets (A B : ℤ) : ∃ C : ℤ,
  ∀ x y : ℤ, x^2 + A*x + B ≠ 2*y^2 + 2*y + C :=
by sorry

end NUMINAMATH_CALUDE_disjoint_quadratic_sets_l485_48566


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_three_l485_48509

theorem three_digit_divisible_by_three :
  ∀ n : ℕ,
  (n ≥ 100 ∧ n < 1000) →  -- Three-digit number
  (n % 10 = 4) →  -- Units digit is 4
  (n / 100 = 4) →  -- Hundreds digit is 4
  (n % 3 = 0) →  -- Divisible by 3
  (n = 414 ∨ n = 444 ∨ n = 474) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_three_l485_48509


namespace NUMINAMATH_CALUDE_function_bounds_l485_48540

theorem function_bounds (k : ℕ) (f : ℕ → ℕ) 
  (h_increasing : ∀ m n, m < n → f m < f n)
  (h_composition : ∀ n, f (f n) = k * n) :
  ∀ n : ℕ, (2 * k : ℚ) / (k + 1 : ℚ) * n ≤ f n ∧ (f n : ℚ) ≤ (k + 1 : ℚ) / 2 * n :=
by sorry

end NUMINAMATH_CALUDE_function_bounds_l485_48540


namespace NUMINAMATH_CALUDE_simplify_expression_l485_48537

theorem simplify_expression (x : ℝ) : 120*x - 72*x + 15*x - 9*x = 54*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l485_48537


namespace NUMINAMATH_CALUDE_left_square_side_length_l485_48560

/-- Proves that given three squares with specific side length relationships, 
    the left square has a side length of 8 cm. -/
theorem left_square_side_length : 
  ∀ (left middle right : ℝ),
  left + middle + right = 52 →
  middle = left + 17 →
  right = middle - 6 →
  left = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_left_square_side_length_l485_48560


namespace NUMINAMATH_CALUDE_tangent_perpendicular_to_y_axis_l485_48575

/-- The curve function -/
def f (a : ℝ) (x : ℝ) : ℝ := x^4 + a*x^2 + 1

/-- The derivative of the curve function -/
def f_prime (a : ℝ) (x : ℝ) : ℝ := 4*x^3 + 2*a*x

theorem tangent_perpendicular_to_y_axis (a : ℝ) :
  (f a (-1) = a + 2) →
  (f_prime a (-1) = 0) →
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_to_y_axis_l485_48575


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l485_48583

/-- The hyperbola C: mx² + ny² = 1 -/
structure Hyperbola where
  m : ℝ
  n : ℝ
  h_mn : m * n < 0

/-- The circle x² + y² - 6x - 2y + 9 = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 6*p.1 - 2*p.2 + 9 = 0}

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The asymptotes of a hyperbola -/
def asymptotes (h : Hyperbola) : Set (Set (ℝ × ℝ)) := sorry

/-- Predicate to check if a line is tangent to a circle -/
def is_tangent_to (line : Set (ℝ × ℝ)) (circle : Set (ℝ × ℝ)) : Prop := sorry

theorem hyperbola_eccentricity (h : Hyperbola) :
  (∃ a ∈ asymptotes h, is_tangent_to a Circle) →
  (eccentricity h = 5/3 ∨ eccentricity h = 5/4) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l485_48583


namespace NUMINAMATH_CALUDE_polynomial_degree_l485_48579

/-- The degree of the polynomial resulting from the expansion of 
    (3x^5 + 2x^3 - x + 7)(4x^11 - 6x^8 + 5x^5 - 15) - (x^2 + 3)^8 is 16 -/
theorem polynomial_degree : ℕ := by
  sorry

end NUMINAMATH_CALUDE_polynomial_degree_l485_48579


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l485_48578

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁^2 + 4*x₁ = 5) ∧ 
  (x₂^2 + 4*x₂ = 5) ∧ 
  x₁ = 1 ∧ 
  x₂ = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l485_48578


namespace NUMINAMATH_CALUDE_cubic_equation_root_l485_48517

theorem cubic_equation_root (c d : ℚ) : 
  (∃ x : ℝ, x^3 + c*x^2 + d*x + 44 = 0 ∧ x = 1 - 3*Real.sqrt 5) → c = -3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l485_48517


namespace NUMINAMATH_CALUDE_kayla_apples_l485_48542

theorem kayla_apples (total : ℕ) (kayla kylie : ℕ) : 
  total = 200 →
  total = kayla + kylie →
  kayla = kylie / 4 →
  kayla = 40 := by
sorry

end NUMINAMATH_CALUDE_kayla_apples_l485_48542


namespace NUMINAMATH_CALUDE_sum_product_theorem_l485_48500

theorem sum_product_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 213) 
  (h2 : a + b + c = 15) : 
  a*b + b*c + c*a = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_product_theorem_l485_48500


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l485_48534

/-- The sequence defined by a₀ = a₁ = a₂ = 1 and a_{n+2} = (a_n * a_{n+1} + 1) / a_{n-1} for n ≥ 1 -/
def a : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 1
| (n + 3) => (a n * a (n + 1) + 1) / a (n - 1)

/-- The property that needs to be satisfied by the triples (a, b, c) -/
def satisfies_equation (a b c : ℕ) : Prop :=
  1 / a + 1 / b + 1 / c + 1 / (a * b * c) = 12 / (a + b + c)

theorem infinitely_many_solutions :
  ∀ N : ℕ, ∃ a b c : ℕ, a > N ∧ b > N ∧ c > N ∧ satisfies_equation a b c :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l485_48534


namespace NUMINAMATH_CALUDE_max_value_of_expression_l485_48525

def is_distinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

def is_nonzero_digit (n : ℕ) : Prop := 0 < n ∧ n ≤ 9

def expression (a b c : ℕ) : ℚ :=
  1 / (a + 2010 / (b + 1 / c))

theorem max_value_of_expression :
  ∀ a b c : ℕ,
    is_distinct a b c →
    is_nonzero_digit a →
    is_nonzero_digit b →
    is_nonzero_digit c →
    expression a b c ≤ 1 / 203 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l485_48525


namespace NUMINAMATH_CALUDE_intersection_implies_m_eq_neg_two_l485_48503

-- Define the sets M and N
def M (m : ℝ) : Set ℂ := {1, 2, (m^2 - 2*m - 5 : ℂ) + (m^2 + 5*m + 6 : ℂ)*Complex.I}
def N : Set ℂ := {3}

-- State the theorem
theorem intersection_implies_m_eq_neg_two (m : ℝ) : 
  (M m ∩ N).Nonempty → m = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_eq_neg_two_l485_48503


namespace NUMINAMATH_CALUDE_value_of_x_l485_48573

theorem value_of_x (x y z : ℚ) : 
  x = (1 / 3) * y → 
  y = (1 / 10) * z → 
  z = 100 → 
  x = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l485_48573


namespace NUMINAMATH_CALUDE_least_integer_square_72_more_than_double_l485_48544

theorem least_integer_square_72_more_than_double :
  ∃ (x : ℤ), x^2 = 2*x + 72 ∧ ∀ (y : ℤ), y^2 = 2*y + 72 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_integer_square_72_more_than_double_l485_48544


namespace NUMINAMATH_CALUDE_plane_curve_mass_approx_l485_48561

noncomputable def curve_mass (a b : Real) : Real :=
  ∫ x in a..b, (1 + x^2) * Real.sqrt (1 + (3 * x^2)^2)

theorem plane_curve_mass_approx : 
  ∃ ε > 0, abs (curve_mass 0 0.1 - 0.099985655) < ε :=
sorry

end NUMINAMATH_CALUDE_plane_curve_mass_approx_l485_48561


namespace NUMINAMATH_CALUDE_pushup_problem_l485_48508

/-- Given that David did 30 more push-ups than Zachary, Sarah completed twice as many push-ups as Zachary,
    and David did 37 push-ups, prove that Zachary and Sarah did 21 push-ups combined. -/
theorem pushup_problem (david zachary sarah : ℕ) : 
  david = zachary + 30 →
  sarah = 2 * zachary →
  david = 37 →
  zachary + sarah = 21 := by
  sorry

end NUMINAMATH_CALUDE_pushup_problem_l485_48508


namespace NUMINAMATH_CALUDE_expression_equality_l485_48565

theorem expression_equality (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 - 1) / x) * ((y^2 - 1) / y) - ((x^2 - 1) / y) * ((y^2 - 1) / x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l485_48565


namespace NUMINAMATH_CALUDE_exists_monochromatic_congruent_triangle_l485_48523

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring of the plane
def Coloring := Point → Color

-- Define a triangle
structure Triangle where
  a : Point
  b : Point
  c : Point

-- Define congruence for triangles
def Congruent (t1 t2 : Triangle) : Prop := sorry

-- Define the property of having all vertices of the same color
def SameColor (t : Triangle) (coloring : Coloring) : Prop :=
  coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c

-- The main theorem
theorem exists_monochromatic_congruent_triangle :
  ∃ (T : Triangle), ∀ (coloring : Coloring),
    ∃ (T' : Triangle), Congruent T T' ∧ SameColor T' coloring := by sorry

end NUMINAMATH_CALUDE_exists_monochromatic_congruent_triangle_l485_48523


namespace NUMINAMATH_CALUDE_heidi_to_danielle_ratio_l485_48557

/-- The number of rooms in Danielle's apartment -/
def danielles_rooms : ℕ := 6

/-- The number of rooms in Grant's apartment -/
def grants_rooms : ℕ := 2

/-- The ratio of Grant's rooms to Heidi's rooms -/
def grant_to_heidi_ratio : ℚ := 1 / 9

/-- The number of rooms in Heidi's apartment -/
def heidis_rooms : ℕ := grants_rooms * 9

theorem heidi_to_danielle_ratio : 
  (heidis_rooms : ℚ) / danielles_rooms = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_heidi_to_danielle_ratio_l485_48557


namespace NUMINAMATH_CALUDE_expression_evaluation_l485_48568

theorem expression_evaluation : 
  (0.82 : Real)^3 - (0.1 : Real)^3 / (0.82 : Real)^2 + 0.082 + (0.1 : Real)^2 = 0.641881 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l485_48568


namespace NUMINAMATH_CALUDE_four_digit_sum_l485_48543

theorem four_digit_sum (a b c d : Nat) : 
  a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →
  (∃ (x y z : Nat), x < 10 ∧ y < 10 ∧ z < 10 ∧
    ((1000 * a + 100 * b + 10 * c + d) + (100 * x + 10 * y + z) = 6031 ∨
     (1000 * a + 100 * b + 10 * c + d) + (100 * a + 10 * y + z) = 6031 ∨
     (1000 * a + 100 * b + 10 * c + d) + (100 * a + 10 * b + z) = 6031 ∨
     (1000 * a + 100 * b + 10 * c + d) + (100 * x + 10 * y + c) = 6031)) →
  a + b + c + d = 20 := by
sorry

end NUMINAMATH_CALUDE_four_digit_sum_l485_48543


namespace NUMINAMATH_CALUDE_bug_crawl_theorem_l485_48536

def bug_movements : List Int := [5, -3, 10, -8, -6, 12, -10]

theorem bug_crawl_theorem :
  (List.sum bug_movements = 0) ∧
  (List.sum (List.map Int.natAbs bug_movements) = 54) := by
  sorry

end NUMINAMATH_CALUDE_bug_crawl_theorem_l485_48536


namespace NUMINAMATH_CALUDE_strawberry_weight_difference_l485_48547

theorem strawberry_weight_difference (marco_weight dad_weight total_weight : ℕ) 
  (h1 : marco_weight = 30)
  (h2 : total_weight = 47)
  (h3 : total_weight = marco_weight + dad_weight) :
  marco_weight - dad_weight = 13 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_weight_difference_l485_48547


namespace NUMINAMATH_CALUDE_car_engine_part_cost_l485_48521

/-- Calculates the cost of a car engine part given labor and total cost information --/
theorem car_engine_part_cost
  (labor_rate : ℕ)
  (labor_hours : ℕ)
  (total_cost : ℕ)
  (h1 : labor_rate = 75)
  (h2 : labor_hours = 16)
  (h3 : total_cost = 2400) :
  total_cost - (labor_rate * labor_hours) = 1200 := by
  sorry

#check car_engine_part_cost

end NUMINAMATH_CALUDE_car_engine_part_cost_l485_48521


namespace NUMINAMATH_CALUDE_tetrahedron_surface_area_bound_l485_48538

/-- Tetrahedron with given edge lengths and surface area -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  S : ℝ
  edge_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f
  surface_positive : 0 < S

/-- The surface area of a tetrahedron is bounded by a function of its edge lengths -/
theorem tetrahedron_surface_area_bound (t : Tetrahedron) :
    t.S ≤ (Real.sqrt 3 / 6) * (t.a^2 + t.b^2 + t.c^2 + t.d^2 + t.e^2 + t.f^2) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_surface_area_bound_l485_48538


namespace NUMINAMATH_CALUDE_proportion_fourth_term_l485_48514

theorem proportion_fourth_term (x y : ℝ) : 
  (0.25 / x = 2 / y) → x = 0.75 → y = 6 := by sorry

end NUMINAMATH_CALUDE_proportion_fourth_term_l485_48514


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l485_48580

theorem simplest_quadratic_radical :
  let a := Real.sqrt 8
  let b := Real.sqrt 7
  let c := Real.sqrt 12
  let d := Real.sqrt (1/3)
  (∃ (x y : ℝ), a = x * Real.sqrt y ∧ x ≠ 1) ∧
  (∃ (x y : ℝ), c = x * Real.sqrt y ∧ x ≠ 1) ∧
  (∃ (x y : ℝ), d = x * Real.sqrt y ∧ x ≠ 1) ∧
  (∀ (x y : ℝ), b = x * Real.sqrt y → x = 1) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l485_48580


namespace NUMINAMATH_CALUDE_goldbach_multiplication_counterexample_l485_48515

theorem goldbach_multiplication_counterexample :
  ∃ p : ℕ, Prime p ∧ p > 5 ∧
  (∀ q r : ℕ, Prime q → Prime r → Odd q → Odd r → p ≠ q * r) ∧
  (∀ q : ℕ, Prime q → Odd q → p ≠ q^2) := by
  sorry

end NUMINAMATH_CALUDE_goldbach_multiplication_counterexample_l485_48515


namespace NUMINAMATH_CALUDE_joanna_reading_speed_l485_48572

theorem joanna_reading_speed :
  ∀ (total_pages : ℕ) (monday_hours tuesday_hours remaining_hours : ℝ) (pages_per_hour : ℝ),
    total_pages = 248 →
    monday_hours = 3 →
    tuesday_hours = 6.5 →
    remaining_hours = 6 →
    (monday_hours + tuesday_hours + remaining_hours) * pages_per_hour = total_pages →
    pages_per_hour = 16 := by
  sorry

end NUMINAMATH_CALUDE_joanna_reading_speed_l485_48572


namespace NUMINAMATH_CALUDE_plane_points_theorem_l485_48555

def connecting_lines (n : ℕ) : ℕ := n * (n - 1) / 2

theorem plane_points_theorem (n₁ n₂ : ℕ) : 
  (connecting_lines n₁ = connecting_lines n₂ + 27) →
  (connecting_lines n₁ + connecting_lines n₂ = 171) →
  (n₁ = 11 ∧ n₂ = 8) :=
by sorry

end NUMINAMATH_CALUDE_plane_points_theorem_l485_48555


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l485_48597

def A : Set ℝ := {x | -5 ≤ x ∧ x < 1}
def B : Set ℝ := {x | x ≤ 2}

theorem union_of_A_and_B : A ∪ B = {x | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l485_48597


namespace NUMINAMATH_CALUDE_eighth_of_2_36_equals_2_33_l485_48530

theorem eighth_of_2_36_equals_2_33 : ∃ y : ℕ, (1 / 8 : ℝ) * (2 ^ 36) = 2 ^ y → y = 33 := by
  sorry

end NUMINAMATH_CALUDE_eighth_of_2_36_equals_2_33_l485_48530


namespace NUMINAMATH_CALUDE_min_value_expression_l485_48552

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 6) :
  (9 / a + 16 / b + 25 / c) ≥ 18 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l485_48552


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l485_48511

theorem polynomial_evaluation :
  let f : ℝ → ℝ := λ x => 2*x^4 + 3*x^3 + x^2 + 2*x + 3
  f 2 = 67 := by
sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l485_48511


namespace NUMINAMATH_CALUDE_shortest_tangent_length_l485_48562

/-- Circle C3 centered at (8, 0) with radius 5 -/
def C3 (x y : ℝ) : Prop := (x - 8)^2 + y^2 = 25

/-- Circle C4 centered at (-10, 0) with radius 7 -/
def C4 (x y : ℝ) : Prop := (x + 10)^2 + y^2 = 49

/-- Point R on circle C3 -/
def R : ℝ × ℝ := sorry

/-- Point S on circle C4 -/
def S : ℝ × ℝ := sorry

/-- The shortest line segment RS is tangent to C3 at R and C4 at S -/
theorem shortest_tangent_length : 
  C3 R.1 R.2 ∧ C4 S.1 S.2 → 
  ∃ (R S : ℝ × ℝ), C3 R.1 R.2 ∧ C4 S.1 S.2 ∧ 
    Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_shortest_tangent_length_l485_48562


namespace NUMINAMATH_CALUDE_four_customers_no_change_l485_48535

/-- Represents the auto shop scenario -/
structure AutoShop where
  initial_cars : ℕ
  new_customers : ℕ
  tires_per_car : ℕ
  half_change_customers : ℕ
  tires_left : ℕ

/-- Calculates the number of customers who didn't want their tires changed -/
def customers_no_change (shop : AutoShop) : ℕ :=
  let total_cars := shop.initial_cars + shop.new_customers
  let total_tires_bought := total_cars * shop.tires_per_car
  let half_change_tires := shop.half_change_customers * (shop.tires_per_car / 2)
  let unused_tires := shop.tires_left - half_change_tires
  unused_tires / shop.tires_per_car

/-- Theorem stating that given the conditions, 4 customers decided not to change their tires -/
theorem four_customers_no_change (shop : AutoShop) 
  (h1 : shop.initial_cars = 4)
  (h2 : shop.new_customers = 6)
  (h3 : shop.tires_per_car = 4)
  (h4 : shop.half_change_customers = 2)
  (h5 : shop.tires_left = 20) :
  customers_no_change shop = 4 := by
  sorry

#eval customers_no_change { initial_cars := 4, new_customers := 6, tires_per_car := 4, half_change_customers := 2, tires_left := 20 }

end NUMINAMATH_CALUDE_four_customers_no_change_l485_48535


namespace NUMINAMATH_CALUDE_factor_expression_l485_48519

theorem factor_expression (x : ℝ) : 4 * x * (x + 1) + 9 * (x + 1) = (x + 1) * (4 * x + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l485_48519


namespace NUMINAMATH_CALUDE_custom_op_seven_three_l485_48513

-- Define the custom operation
def custom_op (a b : ℤ) : ℤ := 4*a + 5*b - a*b

-- Theorem statement
theorem custom_op_seven_three :
  custom_op 7 3 = 22 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_seven_three_l485_48513


namespace NUMINAMATH_CALUDE_factorization_proof_l485_48553

theorem factorization_proof (a m n : ℝ) : a * m^2 - 2 * a * m * n + a * n^2 = a * (m - n)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l485_48553


namespace NUMINAMATH_CALUDE_quadruple_theorem_l485_48507

def is_valid_quadruple (a b c d : ℝ) : Prop :=
  (a = b * c ∨ a = b * d ∨ a = c * d) ∧
  (b = a * c ∨ b = a * d ∨ b = c * d) ∧
  (c = a * b ∨ c = a * d ∨ c = b * d) ∧
  (d = a * b ∨ d = a * c ∨ d = b * c)

def is_solution_quadruple (a b c d : ℝ) : Prop :=
  (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
  ((a = 1 ∧ b = 1 ∧ c = -1 ∧ d = -1) ∨
   (a = 1 ∧ b = -1 ∧ c = 1 ∧ d = -1) ∨
   (a = 1 ∧ b = -1 ∧ c = -1 ∧ d = 1) ∨
   (a = -1 ∧ b = 1 ∧ c = 1 ∧ d = -1) ∨
   (a = -1 ∧ b = 1 ∧ c = -1 ∧ d = 1) ∨
   (a = -1 ∧ b = -1 ∧ c = 1 ∧ d = 1)) ∨
  ((a = 1 ∧ b = -1 ∧ c = -1 ∧ d = -1) ∨
   (a = -1 ∧ b = 1 ∧ c = -1 ∧ d = -1) ∨
   (a = -1 ∧ b = -1 ∧ c = 1 ∧ d = -1) ∨
   (a = -1 ∧ b = -1 ∧ c = -1 ∧ d = 1))

theorem quadruple_theorem (a b c d : ℝ) :
  is_valid_quadruple a b c d → is_solution_quadruple a b c d := by
  sorry

end NUMINAMATH_CALUDE_quadruple_theorem_l485_48507


namespace NUMINAMATH_CALUDE_erasers_ratio_l485_48558

def erasers_problem (hanna rachel tanya_red tanya_total : ℕ) : Prop :=
  hanna = 4 ∧
  tanya_total = 20 ∧
  hanna = 2 * rachel ∧
  rachel = tanya_red / 2 - 3 ∧
  tanya_red ≤ tanya_total

theorem erasers_ratio :
  ∀ hanna rachel tanya_red tanya_total,
    erasers_problem hanna rachel tanya_red tanya_total →
    (tanya_red : ℚ) / tanya_total = 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_erasers_ratio_l485_48558


namespace NUMINAMATH_CALUDE_surface_area_ratio_of_cubes_l485_48532

theorem surface_area_ratio_of_cubes (a b : ℝ) (h : a / b = 5) :
  (6 * a^2) / (6 * b^2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_ratio_of_cubes_l485_48532


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_equations_l485_48592

/-- The equations of an ellipse and a hyperbola with shared foci -/
theorem ellipse_hyperbola_equations :
  ∀ (a b m n : ℝ),
  a > b ∧ b > 0 ∧ m > 0 ∧ n > 0 →
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ x^2/m^2 - y^2/n^2 = 1) →
  (a^2 - b^2 = 4 ∧ m^2 + n^2 = 4) →
  (∃ k : ℝ, k > 0 ∧ ∀ x y : ℝ, y = k*x → x^2/m^2 - y^2/n^2 = 1) →
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    (x₁ - 2)^2 + y₁^2 = (x₁ + 2)^2 + y₁^2 ∧
    x₁^2/m^2 - y₁^2/n^2 = 1 ∧
    x₂^2/a^2 + y₂^2/b^2 = 1 ∧
    x₂^2/m^2 - y₂^2/n^2 = 1) →
  (∀ x y : ℝ, 11*x^2/60 + 11*y^2/16 = 1 ↔ x^2/a^2 + y^2/b^2 = 1) ∧
  (∀ x y : ℝ, 5*x^2/4 - 5*y^2/16 = 1 ↔ x^2/m^2 - y^2/n^2 = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_equations_l485_48592


namespace NUMINAMATH_CALUDE_copper_weights_problem_l485_48595

theorem copper_weights_problem :
  ∃ (x y z u : ℕ+),
    (x : ℤ) + y + z + u = 40 ∧
    ∀ W : ℤ, 1 ≤ W ∧ W ≤ 40 →
      ∃ (a b c d : ℤ),
        (a = -1 ∨ a = 0 ∨ a = 1) ∧
        (b = -1 ∨ b = 0 ∨ b = 1) ∧
        (c = -1 ∨ c = 0 ∨ c = 1) ∧
        (d = -1 ∨ d = 0 ∨ d = 1) ∧
        W = a * x + b * y + c * z + d * u :=
by sorry

end NUMINAMATH_CALUDE_copper_weights_problem_l485_48595


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l485_48506

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < p / q ∧ p / q < (5 : ℚ) / 8 ∧ 
  (∀ p' q' : ℕ+, (3 : ℚ) / 5 < p' / q' ∧ p' / q' < (5 : ℚ) / 8 → q ≤ q') →
  q - p = 5 := by sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l485_48506


namespace NUMINAMATH_CALUDE_m_minus_n_values_l485_48589

theorem m_minus_n_values (m n : ℤ) 
  (hm : |m| = 4)
  (hn : |n| = 6)
  (hmn : |m + n| = m + n) :
  m - n = -2 ∨ m - n = -10 := by
sorry

end NUMINAMATH_CALUDE_m_minus_n_values_l485_48589


namespace NUMINAMATH_CALUDE_unique_natural_number_solution_l485_48564

theorem unique_natural_number_solution (n p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → (1 : ℚ) / n = 1 / p + 1 / q + 1 / (p * q) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_natural_number_solution_l485_48564


namespace NUMINAMATH_CALUDE_ellipse_focal_property_l485_48539

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

/-- The left focus of the ellipse -/
def leftFocus : ℝ × ℝ := (-3, 0)

/-- The left directrix of the ellipse -/
def leftDirectrix (x : ℝ) : Prop := x = -25/3

/-- A line passing through the left focus -/
def lineThroughFocus (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 3)

/-- Point D is to the right of the left focus -/
def pointD (a θ : ℝ) : Prop := a > -3

/-- The circle with MN as diameter passes through F₁ -/
def circleCondition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ + 3) * (x₂ + 3) + y₁ * y₂ = 0

/-- The main theorem -/
theorem ellipse_focal_property (k a θ : ℝ) (x₁ y₁ x₂ y₂ xₘ xₙ yₘ yₙ : ℝ) :
  ellipse x₁ y₁ →
  ellipse x₂ y₂ →
  lineThroughFocus k x₁ y₁ →
  lineThroughFocus k x₂ y₂ →
  pointD a θ →
  leftDirectrix xₘ →
  leftDirectrix xₙ →
  circleCondition xₘ yₘ xₙ yₙ →
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_ellipse_focal_property_l485_48539


namespace NUMINAMATH_CALUDE_april_spending_l485_48586

def initial_savings : ℕ := 11000
def february_percentage : ℚ := 20 / 100
def march_percentage : ℚ := 40 / 100
def remaining_savings : ℕ := 2900

theorem april_spending :
  let february_spending := (february_percentage * initial_savings).floor
  let march_spending := (march_percentage * initial_savings).floor
  let total_spent := initial_savings - remaining_savings
  total_spent - february_spending - march_spending = 1500 := by sorry

end NUMINAMATH_CALUDE_april_spending_l485_48586


namespace NUMINAMATH_CALUDE_inequality_proof_l485_48505

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a / (b + 2*c + 3*d)) + (b / (c + 2*d + 3*a)) + (c / (d + 2*a + 3*b)) + (d / (a + 2*b + 3*c)) ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l485_48505


namespace NUMINAMATH_CALUDE_cos_A_minus_B_l485_48551

theorem cos_A_minus_B (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3/2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_A_minus_B_l485_48551


namespace NUMINAMATH_CALUDE_fraction_decomposition_l485_48570

theorem fraction_decomposition (x A B : ℚ) : 
  (7 * x - 15) / (3 * x^2 + 2 * x - 8) = A / (x + 2) + B / (3 * x - 4) → 
  A = 29 / 10 ∧ B = -17 / 10 := by
sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l485_48570


namespace NUMINAMATH_CALUDE_parallel_vectors_theta_l485_48524

theorem parallel_vectors_theta (θ : Real) 
  (h1 : θ > 0) (h2 : θ < Real.pi / 2)
  (a : Fin 2 → Real) (b : Fin 2 → Real)
  (ha : a = ![3/2, Real.sin θ])
  (hb : b = ![Real.cos θ, 1/3])
  (h_parallel : ∃ (k : Real), a = k • b) :
  θ = Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_theta_l485_48524


namespace NUMINAMATH_CALUDE_positive_integer_solutions_l485_48548

theorem positive_integer_solutions : 
  ∀ x y z : ℕ+, 
    (x + y = z ∧ x^2 * y = z^2 + 1) ↔ 
    ((x = 5 ∧ y = 2 ∧ z = 7) ∨ (x = 5 ∧ y = 3 ∧ z = 8)) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_l485_48548


namespace NUMINAMATH_CALUDE_hash_difference_l485_48502

def hash (x y : ℝ) : ℝ := x * y - 3 * x

theorem hash_difference : (hash 6 4) - (hash 4 6) = -6 := by
  sorry

end NUMINAMATH_CALUDE_hash_difference_l485_48502


namespace NUMINAMATH_CALUDE_senior_class_college_attendance_l485_48581

theorem senior_class_college_attendance 
  (num_boys : ℕ) 
  (num_girls : ℕ) 
  (percent_not_attending : ℝ) 
  (h1 : num_boys = 300)
  (h2 : num_girls = 240)
  (h3 : percent_not_attending = 0.3)
  : (((1 - percent_not_attending) * (num_boys + num_girls)) / (num_boys + num_girls)) * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_senior_class_college_attendance_l485_48581


namespace NUMINAMATH_CALUDE_johns_money_ratio_l485_48550

/-- The ratio of money John got from his grandma to his grandpa -/
theorem johns_money_ratio :
  ∀ (x : ℚ), 
  (30 : ℚ) + 30 * x = 120 →
  (30 * x) / 30 = 3 / 1 :=
by sorry

end NUMINAMATH_CALUDE_johns_money_ratio_l485_48550


namespace NUMINAMATH_CALUDE_abs_neg_three_l485_48531

theorem abs_neg_three : |(-3 : ℤ)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_l485_48531


namespace NUMINAMATH_CALUDE_souvenir_store_problem_l485_48584

/-- Souvenir store problem -/
theorem souvenir_store_problem 
  (total_souvenirs : ℕ)
  (cost_40A_30B cost_10A_50B : ℕ)
  (sell_price_A sell_price_B : ℕ)
  (m : ℚ)
  (h_total : total_souvenirs = 300)
  (h_cost1 : cost_40A_30B = 5000)
  (h_cost2 : cost_10A_50B = 3800)
  (h_sell_A : sell_price_A = 120)
  (h_sell_B : sell_price_B = 80)
  (h_m_range : 4 < m ∧ m < 8) :
  ∃ (cost_A cost_B max_profit : ℕ) (reduced_profit : ℚ),
    cost_A = 80 ∧ 
    cost_B = 60 ∧
    max_profit = 7500 ∧
    reduced_profit = 5720 ∧
    ∀ (a : ℕ), 
      a ≤ total_souvenirs →
      (total_souvenirs - a) ≥ 3 * a →
      (sell_price_A - cost_A) * a + (sell_price_B - cost_B) * (total_souvenirs - a) ≥ 7400 →
      (sell_price_A - cost_A) * a + (sell_price_B - cost_B) * (total_souvenirs - a) ≤ max_profit ∧
      ((sell_price_A - 5 * m - cost_A) * 70 + (sell_price_B - cost_B) * (total_souvenirs - 70) : ℚ) = reduced_profit →
      m = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_souvenir_store_problem_l485_48584


namespace NUMINAMATH_CALUDE_employee_pay_l485_48541

theorem employee_pay (total : ℝ) (x y : ℝ) (h1 : total = 572) (h2 : x + y = total) (h3 : x = 1.2 * y) : y = 260 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_l485_48541


namespace NUMINAMATH_CALUDE_smallest_divisible_by_12_l485_48533

def initial_sequence : List ℕ := [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81]

def append_number (seq : List ℕ) (n : ℕ) : List ℕ :=
  seq ++ [n]

def to_single_number (seq : List ℕ) : ℕ :=
  seq.foldl (λ acc x => acc * 10^(Nat.digits 10 x).length + x) 0

def is_divisible_by_12 (n : ℕ) : Prop :=
  n % 12 = 0

theorem smallest_divisible_by_12 :
  ∃ N : ℕ, N ≥ 82 ∧
    is_divisible_by_12 (to_single_number (append_number initial_sequence N)) ∧
    ∀ k : ℕ, 82 ≤ k ∧ k < N →
      ¬is_divisible_by_12 (to_single_number (append_number initial_sequence k)) ∧
    N = 84 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_12_l485_48533


namespace NUMINAMATH_CALUDE_sum_product_over_sum_squares_l485_48522

theorem sum_product_over_sum_squares (a b c : ℝ) (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h2 : a + b + c = 1) :
  (a * b + b * c + c * a) / (a^2 + b^2 + c^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_over_sum_squares_l485_48522


namespace NUMINAMATH_CALUDE_f_greater_g_iff_a_geq_half_l485_48559

noncomputable section

open Real

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a - log x

def g (x : ℝ) : ℝ := 1/x - Real.exp 1 / (Real.exp x)

-- State the theorem
theorem f_greater_g_iff_a_geq_half (a : ℝ) :
  (∀ x > 1, f a x > g x) ↔ a ≥ 1/2 := by sorry

end

end NUMINAMATH_CALUDE_f_greater_g_iff_a_geq_half_l485_48559


namespace NUMINAMATH_CALUDE_cos_2015_eq_neg_sin_55_l485_48510

theorem cos_2015_eq_neg_sin_55 (m : ℝ) (h : Real.sin (55 * π / 180) = m) :
  Real.cos (2015 * π / 180) = -m := by
  sorry

end NUMINAMATH_CALUDE_cos_2015_eq_neg_sin_55_l485_48510


namespace NUMINAMATH_CALUDE_at_least_one_passes_probability_l485_48526

/-- Probability of A answering a single question correctly -/
def prob_A : ℚ := 2/3

/-- Probability of B answering a single question correctly -/
def prob_B : ℚ := 1/2

/-- Number of questions in the test -/
def num_questions : ℕ := 3

/-- Number of correct answers required to pass -/
def pass_threshold : ℕ := 2

/-- Probability of at least one of A and B passing the test -/
def prob_at_least_one_passes : ℚ := 47/54

theorem at_least_one_passes_probability :
  prob_at_least_one_passes = 1 - (1 - (Nat.choose num_questions pass_threshold * prob_A^pass_threshold * (1-prob_A)^(num_questions-pass_threshold) + prob_A^num_questions)) *
                                 (1 - (Nat.choose num_questions pass_threshold * prob_B^pass_threshold * (1-prob_B)^(num_questions-pass_threshold) + prob_B^num_questions)) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_passes_probability_l485_48526


namespace NUMINAMATH_CALUDE_greater_number_problem_l485_48585

theorem greater_number_problem (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) (h3 : x > y) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_problem_l485_48585


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l485_48594

theorem cubic_roots_sum (p q r : ℝ) : 
  (3 * p^3 - 4 * p^2 + 200 * p - 5 = 0) →
  (3 * q^3 - 4 * q^2 + 200 * q - 5 = 0) →
  (3 * r^3 - 4 * r^2 + 200 * r - 5 = 0) →
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = 184/9 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l485_48594


namespace NUMINAMATH_CALUDE_clothing_store_gross_profit_l485_48563

-- Define the purchase price
def purchase_price : ℚ := 81

-- Define the initial markup percentage
def markup_percentage : ℚ := 1/4

-- Define the price decrease percentage
def price_decrease_percentage : ℚ := 1/5

-- Define the function to calculate the initial selling price
def initial_selling_price (purchase_price : ℚ) (markup_percentage : ℚ) : ℚ :=
  purchase_price / (1 - markup_percentage)

-- Define the function to calculate the new selling price after discount
def new_selling_price (initial_price : ℚ) (decrease_percentage : ℚ) : ℚ :=
  initial_price * (1 - decrease_percentage)

-- Define the function to calculate the gross profit
def gross_profit (new_price : ℚ) (purchase_price : ℚ) : ℚ :=
  new_price - purchase_price

-- Theorem statement
theorem clothing_store_gross_profit :
  let initial_price := initial_selling_price purchase_price markup_percentage
  let new_price := new_selling_price initial_price price_decrease_percentage
  gross_profit new_price purchase_price = 27/5 := by sorry

end NUMINAMATH_CALUDE_clothing_store_gross_profit_l485_48563


namespace NUMINAMATH_CALUDE_range_of_f_l485_48587

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Define the domain
def domain : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | -4 ≤ y ∧ y ≤ 5} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l485_48587


namespace NUMINAMATH_CALUDE_domain_of_f_with_restricted_range_l485_48593

def f (x : ℝ) : ℝ := x^2

def domain : Set ℝ := {-2, -1, 1, 2}
def range : Set ℝ := {1, 4}

theorem domain_of_f_with_restricted_range :
  ∀ y ∈ range, ∃ x ∈ domain, f x = y ∧
  ∀ x : ℝ, f x ∈ range → x ∈ domain :=
by
  sorry

end NUMINAMATH_CALUDE_domain_of_f_with_restricted_range_l485_48593


namespace NUMINAMATH_CALUDE_x_value_l485_48545

def M (x : ℝ) : Set ℝ := {2, 0, x}
def N : Set ℝ := {0, 1}

theorem x_value (h : N ⊆ M x) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l485_48545


namespace NUMINAMATH_CALUDE_gcd_210_294_l485_48504

theorem gcd_210_294 : Nat.gcd 210 294 = 42 := by
  sorry

end NUMINAMATH_CALUDE_gcd_210_294_l485_48504


namespace NUMINAMATH_CALUDE_factorization_equality_l485_48574

theorem factorization_equality (a : ℝ) : (a + 2) * (a - 2) - 3 * a = (a - 4) * (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l485_48574


namespace NUMINAMATH_CALUDE_count_valid_triangles_l485_48529

/-- A point in the 4x4 grid --/
structure GridPoint where
  x : Fin 4
  y : Fin 4

/-- A triangle formed by three grid points --/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- Function to check if three points are collinear --/
def collinear (p1 p2 p3 : GridPoint) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Function to check if a triangle has positive area --/
def positiveArea (t : GridTriangle) : Prop :=
  ¬collinear t.p1 t.p2 t.p3

/-- The set of all possible grid points --/
def allGridPoints : Finset GridPoint :=
  sorry

/-- The set of all possible triangles with positive area --/
def validTriangles : Finset GridTriangle :=
  sorry

/-- The main theorem --/
theorem count_valid_triangles :
  Finset.card validTriangles = 516 :=
sorry

end NUMINAMATH_CALUDE_count_valid_triangles_l485_48529


namespace NUMINAMATH_CALUDE_collinear_points_sum_l485_48527

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (a b c : ℝ × ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), b - a = t • (c - a) ∨ c - a = t • (b - a)

/-- The theorem states that if the given points are collinear, then p + q = 6. -/
theorem collinear_points_sum (p q : ℝ) :
  collinear (2, p, q) (p, 3, q) (p, q, 4) → p + q = 6 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l485_48527
