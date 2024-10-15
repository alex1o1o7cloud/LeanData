import Mathlib

namespace NUMINAMATH_CALUDE_fib_equation_solutions_l3836_383677

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The set of solutions to the Fibonacci equation -/
def fibSolutions : Set (ℕ × ℕ) :=
  {p : ℕ × ℕ | 5 * fib p.1 - 3 * fib p.2 = 1}

theorem fib_equation_solutions :
  fibSolutions = {(2, 3), (5, 8), (8, 13)} := by sorry

end NUMINAMATH_CALUDE_fib_equation_solutions_l3836_383677


namespace NUMINAMATH_CALUDE_cylinder_base_radius_l3836_383608

/-- Given a cylinder with generatrix length 3 cm and lateral area 12π cm², 
    prove that the radius of the base is 2 cm. -/
theorem cylinder_base_radius 
  (generatrix : ℝ) 
  (lateral_area : ℝ) 
  (h1 : generatrix = 3) 
  (h2 : lateral_area = 12 * Real.pi) : 
  lateral_area / (2 * Real.pi * generatrix) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_base_radius_l3836_383608


namespace NUMINAMATH_CALUDE_inverse_f_243_l3836_383619

def f (x : ℝ) : ℝ := sorry

theorem inverse_f_243 (h1 : f 5 = 3) (h2 : ∀ x, f (3 * x) = 3 * f x) : 
  f 405 = 243 := by sorry

end NUMINAMATH_CALUDE_inverse_f_243_l3836_383619


namespace NUMINAMATH_CALUDE_rachel_picture_book_shelves_l3836_383621

/-- Calculates the number of picture book shelves given the total number of books,
    number of mystery book shelves, and books per shelf. -/
def picture_book_shelves (total_books : ℕ) (mystery_shelves : ℕ) (books_per_shelf : ℕ) : ℕ :=
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf

/-- Proves that Rachel has 2 shelves of picture books given the problem conditions. -/
theorem rachel_picture_book_shelves :
  picture_book_shelves 72 6 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_rachel_picture_book_shelves_l3836_383621


namespace NUMINAMATH_CALUDE_optimal_ships_l3836_383633

/-- The maximum annual shipbuilding capacity -/
def max_capacity : ℕ := 20

/-- The output value function -/
def R (x : ℕ) : ℚ := 3700 * x + 45 * x^2 - 10 * x^3

/-- The cost function -/
def C (x : ℕ) : ℚ := 460 * x + 5000

/-- The profit function -/
def p (x : ℕ) : ℚ := R x - C x

/-- The marginal function of a function f -/
def M (f : ℕ → ℚ) (x : ℕ) : ℚ := f (x + 1) - f x

/-- The theorem stating the optimal number of ships to build -/
theorem optimal_ships : 
  ∃ (x : ℕ), x ≤ max_capacity ∧ x > 0 ∧
  ∀ (y : ℕ), y ≤ max_capacity ∧ y > 0 → p x ≥ p y ∧
  x = 12 :=
sorry

end NUMINAMATH_CALUDE_optimal_ships_l3836_383633


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3836_383634

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ ∀ x, ¬ P x :=
by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - x + 4 < 0) ↔ (∀ x : ℝ, x^2 - x + 4 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3836_383634


namespace NUMINAMATH_CALUDE_negation_of_statement_l3836_383644

theorem negation_of_statement :
  (¬ (∀ x : ℝ, (x = 0 ∨ x = 1) → x^2 - x = 0)) ↔
  (∀ x : ℝ, (x ≠ 0 ∧ x ≠ 1) → x^2 - x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_statement_l3836_383644


namespace NUMINAMATH_CALUDE_hyperbola_tangent_property_l3836_383606

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

-- Define the point P
def P : ℝ × ℝ := (2, 3)

-- Define the line AB
def line_AB (x y : ℝ) : Prop := 2*x - 3*y = 2

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define points A and B on the hyperbola
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the angle between two vectors
def angle (v₁ v₂ : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem hyperbola_tangent_property :
  ∀ (A B : ℝ × ℝ),
  hyperbola A.1 A.2 →
  hyperbola B.1 B.2 →
  A.1 < B.1 →
  line_AB A.1 A.2 →
  line_AB B.1 B.2 →
  line_AB P.1 P.2 →
  (∀ (x y : ℝ), hyperbola x y → line_AB x y → (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2)) →
  (line_AB A.1 A.2 ∧ line_AB B.1 B.2) ∧
  angle (A.1 - F₁.1, A.2 - F₁.2) (P.1 - F₁.1, P.2 - F₁.2) =
  angle (B.1 - F₂.1, B.2 - F₂.2) (P.1 - F₂.1, P.2 - F₂.2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_tangent_property_l3836_383606


namespace NUMINAMATH_CALUDE_reflection_line_sum_l3836_383687

/-- Given a line y = mx + b, if the reflection of point (1,2) across this line is (7,6), then m + b = 8.5 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), x = 7 ∧ y = 6 ∧ 
    (x - 1)^2 + (y - 2)^2 = (7 - x)^2 + (6 - y)^2 ∧
    (y - 2) = m * (x - 1) ∧
    y = m * x + b) →
  m + b = 8.5 := by sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l3836_383687


namespace NUMINAMATH_CALUDE_max_right_angles_is_14_l3836_383673

/-- A triangular prism -/
structure TriangularPrism :=
  (faces : Nat)
  (angles : Nat)
  (h_faces : faces = 5)
  (h_angles : angles = 18)

/-- The maximum number of right angles in a triangular prism -/
def max_right_angles (prism : TriangularPrism) : Nat := 14

/-- Theorem: The maximum number of right angles in a triangular prism is 14 -/
theorem max_right_angles_is_14 (prism : TriangularPrism) :
  max_right_angles prism = 14 := by sorry

end NUMINAMATH_CALUDE_max_right_angles_is_14_l3836_383673


namespace NUMINAMATH_CALUDE_divisors_of_5_pow_30_minus_1_l3836_383697

theorem divisors_of_5_pow_30_minus_1 :
  ∀ n : ℕ, 90 < n → n < 100 → (5^30 - 1) % n = 0 ↔ n = 91 ∨ n = 97 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_5_pow_30_minus_1_l3836_383697


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3836_383636

def repeating_decimal_6 : ℚ := 2/3
def repeating_decimal_3 : ℚ := 1/3

theorem sum_of_repeating_decimals : 
  repeating_decimal_6 + repeating_decimal_3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3836_383636


namespace NUMINAMATH_CALUDE_square_of_binomial_l3836_383688

theorem square_of_binomial (d : ℝ) : 
  (∃ a b : ℝ, ∀ x, 8*x^2 + 24*x + d = (a*x + b)^2) → d = 18 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_l3836_383688


namespace NUMINAMATH_CALUDE_world_cup_viewers_scientific_notation_l3836_383679

/-- Expresses a number in millions as scientific notation -/
def scientific_notation_millions (x : ℝ) : ℝ × ℤ :=
  (x, 7)

theorem world_cup_viewers_scientific_notation :
  scientific_notation_millions 70.62 = (7.062, 7) := by
  sorry

end NUMINAMATH_CALUDE_world_cup_viewers_scientific_notation_l3836_383679


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l3836_383685

/-- Proves that if a fruit seller sells 50% of his apples and is left with 5000 apples, 
    then he originally had 10000 apples. -/
theorem fruit_seller_apples (original : ℕ) (sold_percentage : ℚ) (remaining : ℕ) 
    (h1 : sold_percentage = 1/2)
    (h2 : remaining = 5000)
    (h3 : (1 - sold_percentage) * original = remaining) : 
  original = 10000 := by
  sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l3836_383685


namespace NUMINAMATH_CALUDE_distribute_10_8_l3836_383655

/-- The number of ways to distribute n indistinguishable objects into k distinguishable bins,
    with each bin receiving at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Theorem stating that distributing 10 objects into 8 bins, with each bin receiving at least one,
    results in 36 possible distributions. -/
theorem distribute_10_8 : distribute 10 8 = 36 := by
  sorry

end NUMINAMATH_CALUDE_distribute_10_8_l3836_383655


namespace NUMINAMATH_CALUDE_multi_digit_perfect_square_distinct_digits_l3836_383654

theorem multi_digit_perfect_square_distinct_digits :
  ∀ n : ℕ, n > 9 → (∃ m : ℕ, n = m^2) →
    ∃ d₁ d₂ : ℕ, d₁ ≠ d₂ ∧ d₁ < 10 ∧ d₂ < 10 ∧
    ∃ k : ℕ, n = d₁ + 10 * k ∧ ∃ l : ℕ, k = d₂ + 10 * l :=
by sorry

end NUMINAMATH_CALUDE_multi_digit_perfect_square_distinct_digits_l3836_383654


namespace NUMINAMATH_CALUDE_circle_triangle_area_l3836_383627

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def CircleTangentToLine (c : Circle) (l : Line) : Prop := sorry

def CirclesInternallyTangent (c1 c2 : Circle) : Prop := sorry

def CirclesExternallyTangent (c1 c2 : Circle) : Prop := sorry

def PointBetween (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

def AreaOfTriangle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem circle_triangle_area 
  (A B C : Circle)
  (m : Line)
  (A' B' C' : ℝ × ℝ) :
  A.radius = 3 →
  B.radius = 4 →
  C.radius = 5 →
  CircleTangentToLine A m →
  CircleTangentToLine B m →
  CircleTangentToLine C m →
  PointBetween A' B' C' →
  CirclesInternallyTangent A B →
  CirclesExternallyTangent B C →
  AreaOfTriangle A.center B.center C.center = 7 := by
  sorry

end NUMINAMATH_CALUDE_circle_triangle_area_l3836_383627


namespace NUMINAMATH_CALUDE_right_triangle_leg_sum_l3836_383622

theorem right_triangle_leg_sum : 
  ∀ (a b : ℕ), 
  (∃ k : ℕ, a = 2 * k ∧ b = 2 * k + 2) → -- legs are consecutive even whole numbers
  a^2 + b^2 = 50^2 → -- Pythagorean theorem with hypotenuse 50
  a + b = 80 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_sum_l3836_383622


namespace NUMINAMATH_CALUDE_smallest_A_for_divisibility_l3836_383671

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def six_digit_number (A : ℕ) : ℕ := 4 * 100000 + A * 10000 + 88851

theorem smallest_A_for_divisibility :
  ∀ A : ℕ, A ≥ 1 →
    (is_divisible_by_3 (six_digit_number A) → A ≥ 1) ∧
    is_divisible_by_3 (six_digit_number 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_A_for_divisibility_l3836_383671


namespace NUMINAMATH_CALUDE_cube_root_of_x_plus_3y_is_3_l3836_383676

theorem cube_root_of_x_plus_3y_is_3 (x y : ℝ) 
  (h : y = Real.sqrt (x - 3) + Real.sqrt (3 - x) + 8) : 
  (x + 3 * y) ^ (1/3 : ℝ) = 3 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_x_plus_3y_is_3_l3836_383676


namespace NUMINAMATH_CALUDE_sequence_sum_l3836_383678

theorem sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) : 
  a 1 = 1 → 
  (∀ n : ℕ, 2 * S n = a (n + 1) - 1) → 
  a 3 + a 4 + a 5 = 117 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l3836_383678


namespace NUMINAMATH_CALUDE_tangent_perpendicular_implies_negative_a_l3836_383642

theorem tangent_perpendicular_implies_negative_a (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (3 * a * x^2 + 1 / x = 0)) → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_implies_negative_a_l3836_383642


namespace NUMINAMATH_CALUDE_cubic_monotonic_and_odd_l3836_383609

def f (x : ℝ) : ℝ := x^3

theorem cubic_monotonic_and_odd :
  (∀ x y, x < y → f x < f y) ∧ 
  (∀ x, f (-x) = -f x) :=
sorry

end NUMINAMATH_CALUDE_cubic_monotonic_and_odd_l3836_383609


namespace NUMINAMATH_CALUDE_factorization_equality_l3836_383625

theorem factorization_equality (x y : ℝ) : x^2 - 1 + 2*x*y + y^2 = (x+y+1)*(x+y-1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3836_383625


namespace NUMINAMATH_CALUDE_first_candidate_percentage_l3836_383675

theorem first_candidate_percentage (total_votes : ℕ) (invalid_percent : ℚ) (second_candidate_votes : ℕ) :
  total_votes = 5500 →
  invalid_percent = 20/100 →
  second_candidate_votes = 1980 →
  let valid_votes := total_votes * (1 - invalid_percent)
  let first_candidate_votes := valid_votes - second_candidate_votes
  (first_candidate_votes : ℚ) / valid_votes * 100 = 55 := by
  sorry

end NUMINAMATH_CALUDE_first_candidate_percentage_l3836_383675


namespace NUMINAMATH_CALUDE_cannoneer_count_l3836_383689

theorem cannoneer_count (total : ℕ) (cannoneers : ℕ) (women : ℕ) (men : ℕ)
  (h1 : women = 2 * cannoneers)
  (h2 : men = 2 * women)
  (h3 : total = men + women)
  (h4 : total = 378) :
  cannoneers = 63 := by
sorry

end NUMINAMATH_CALUDE_cannoneer_count_l3836_383689


namespace NUMINAMATH_CALUDE_equation_proof_l3836_383650

theorem equation_proof : (5568 / 87)^(1/3) + (72 * 2)^(1/2) = (256)^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3836_383650


namespace NUMINAMATH_CALUDE_num_squares_6x6_l3836_383623

/-- A square on a grid --/
structure GridSquare where
  size : ℕ
  rotation : Bool  -- False for regular, True for diagonal

/-- The set of all possible non-congruent squares on a 6x6 grid --/
def squares_6x6 : Finset GridSquare := sorry

/-- The number of non-congruent squares on a 6x6 grid --/
theorem num_squares_6x6 : Finset.card squares_6x6 = 75 := by sorry

end NUMINAMATH_CALUDE_num_squares_6x6_l3836_383623


namespace NUMINAMATH_CALUDE_notebook_distribution_l3836_383611

theorem notebook_distribution (C : ℕ) (H : ℕ) : 
  (C * (C / 8) = 512) →
  (H / 8 = 16) →
  (H : ℚ) / C = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_notebook_distribution_l3836_383611


namespace NUMINAMATH_CALUDE_balloons_given_correct_fred_balloons_l3836_383632

/-- The number of balloons Fred gave to Sandy -/
def balloons_given (initial current : ℕ) : ℕ := initial - current

theorem balloons_given_correct (initial current : ℕ) (h : initial ≥ current) :
  balloons_given initial current = initial - current :=
by sorry

/-- Fred's scenario -/
theorem fred_balloons :
  let initial : ℕ := 709
  let current : ℕ := 488
  balloons_given initial current = 221 :=
by sorry

end NUMINAMATH_CALUDE_balloons_given_correct_fred_balloons_l3836_383632


namespace NUMINAMATH_CALUDE_not_property_P_if_cong_4_mod_9_l3836_383638

/-- Property P: An integer n has property P if there exist integers x, y, z 
    such that n = x³ + y³ + z³ - 3xyz -/
def has_property_P (n : ℤ) : Prop :=
  ∃ x y z : ℤ, n = x^3 + y^3 + z^3 - 3*x*y*z

/-- Theorem: If an integer n is congruent to 4 modulo 9, then it does not have property P -/
theorem not_property_P_if_cong_4_mod_9 (n : ℤ) (h : n % 9 = 4) : 
  ¬(has_property_P n) := by
  sorry

#check not_property_P_if_cong_4_mod_9

end NUMINAMATH_CALUDE_not_property_P_if_cong_4_mod_9_l3836_383638


namespace NUMINAMATH_CALUDE_cube_volume_increase_cube_volume_not_8_times_l3836_383683

theorem cube_volume_increase (edge : ℝ) (edge_positive : 0 < edge) : 
  (2 * edge)^3 = 27 * edge^3 := by sorry

theorem cube_volume_not_8_times (edge : ℝ) (edge_positive : 0 < edge) : 
  (2 * edge)^3 ≠ 8 * edge^3 := by sorry

end NUMINAMATH_CALUDE_cube_volume_increase_cube_volume_not_8_times_l3836_383683


namespace NUMINAMATH_CALUDE_unique_solution_system_l3836_383669

theorem unique_solution_system (x y : ℝ) :
  (x + y = (5 - x) + (5 - y)) ∧ (x - y = (x - 1) + (y - 1)) →
  x = 4 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3836_383669


namespace NUMINAMATH_CALUDE_janinas_pancakes_sales_l3836_383643

/-- The number of pancakes Janina must sell daily to cover her expenses -/
def pancakes_to_sell (daily_rent : ℕ) (daily_supplies : ℕ) (price_per_pancake : ℕ) : ℕ :=
  (daily_rent + daily_supplies) / price_per_pancake

/-- Theorem stating that Janina must sell 21 pancakes daily to cover her expenses -/
theorem janinas_pancakes_sales : pancakes_to_sell 30 12 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_janinas_pancakes_sales_l3836_383643


namespace NUMINAMATH_CALUDE_dispersion_measures_l3836_383626

-- Define a sample as a list of real numbers
def Sample := List ℝ

-- Define the statistics
def standardDeviation (s : Sample) : ℝ := sorry
def range (s : Sample) : ℝ := sorry
def mean (s : Sample) : ℝ := sorry
def median (s : Sample) : ℝ := sorry

-- Define a predicate for whether a statistic measures dispersion
def measuresDispersion (f : Sample → ℝ) : Prop := sorry

-- Theorem stating which statistics measure dispersion
theorem dispersion_measures (s : Sample) :
  measuresDispersion (standardDeviation) ∧
  measuresDispersion (range) ∧
  ¬measuresDispersion (mean) ∧
  ¬measuresDispersion (median) :=
sorry

end NUMINAMATH_CALUDE_dispersion_measures_l3836_383626


namespace NUMINAMATH_CALUDE_count_valid_pairs_l3836_383696

/-- The number of ordered pairs (m, n) of positive integers satisfying m ≥ n and m² - n² = 120 -/
def count_pairs : ℕ := 4

/-- Predicate for valid pairs -/
def is_valid_pair (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≥ n ∧ m^2 - n^2 = 120

theorem count_valid_pairs :
  (∃! (s : Finset (ℕ × ℕ)), s.card = count_pairs ∧ 
    ∀ p : ℕ × ℕ, p ∈ s ↔ is_valid_pair p.1 p.2) :=
sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l3836_383696


namespace NUMINAMATH_CALUDE_inverse_iff_horizontal_line_test_l3836_383601

-- Define a type for our functions
def Function := ℝ → ℝ

-- Define what it means for a function to have an inverse
def has_inverse (f : Function) : Prop :=
  ∃ g : Function, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- Define the horizontal line test
def passes_horizontal_line_test (f : Function) : Prop :=
  ∀ y : ℝ, ∀ x₁ x₂ : ℝ, f x₁ = y ∧ f x₂ = y → x₁ = x₂

-- State the theorem
theorem inverse_iff_horizontal_line_test (f : Function) :
  has_inverse f ↔ passes_horizontal_line_test f :=
sorry

end NUMINAMATH_CALUDE_inverse_iff_horizontal_line_test_l3836_383601


namespace NUMINAMATH_CALUDE_correct_calculation_l3836_383695

theorem correct_calculation (x : ℝ) : 
  (x / 2 + 45 = 85) → (2 * x - 45 = 115) := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3836_383695


namespace NUMINAMATH_CALUDE_largest_number_in_set_l3836_383639

theorem largest_number_in_set (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c ∧  -- a, b, c are in ascending order
  (a + b + c) / 3 = 6 ∧  -- mean is 6
  b = 6 ∧  -- median is 6
  a = 2  -- smallest number is 2
  → c = 10 := by sorry

end NUMINAMATH_CALUDE_largest_number_in_set_l3836_383639


namespace NUMINAMATH_CALUDE_baking_time_per_batch_l3836_383658

/-- Proves that the time to bake one batch of cupcakes is 20 minutes -/
theorem baking_time_per_batch (
  num_batches : ℕ)
  (icing_time_per_batch : ℕ)
  (total_time : ℕ)
  (h1 : num_batches = 4)
  (h2 : icing_time_per_batch = 30)
  (h3 : total_time = 200)
  : ∃ (baking_time_per_batch : ℕ),
    baking_time_per_batch * num_batches + icing_time_per_batch * num_batches = total_time ∧
    baking_time_per_batch = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_baking_time_per_batch_l3836_383658


namespace NUMINAMATH_CALUDE_integer_pairs_sum_product_l3836_383640

theorem integer_pairs_sum_product (m n : ℤ) : m + n + m * n = 6 ↔ (m = 0 ∧ n = 6) ∨ (m = 6 ∧ n = 0) := by
  sorry

end NUMINAMATH_CALUDE_integer_pairs_sum_product_l3836_383640


namespace NUMINAMATH_CALUDE_pet_store_gerbils_l3836_383617

theorem pet_store_gerbils : 
  ∀ (initial_gerbils sold_gerbils remaining_gerbils : ℕ),
  sold_gerbils = 69 →
  remaining_gerbils = 16 →
  initial_gerbils = sold_gerbils + remaining_gerbils →
  initial_gerbils = 85 := by
sorry

end NUMINAMATH_CALUDE_pet_store_gerbils_l3836_383617


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3836_383646

theorem sum_of_fractions : (3 : ℚ) / 4 + (6 : ℚ) / 9 = (17 : ℚ) / 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3836_383646


namespace NUMINAMATH_CALUDE_function_bounds_l3836_383607

/-- A strictly increasing function from ℕ to ℕ -/
def StrictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ m n, m < n → f m < f n

theorem function_bounds
  (k : ℕ)
  (f : ℕ → ℕ)
  (h_strict : StrictlyIncreasing f)
  (h_comp : ∀ n, f (f n) = k * n) :
  ∀ n, (2 * k : ℚ) / (k + 1) * n ≤ f n ∧ (f n : ℚ) ≤ (k + 1 : ℚ) / 2 * n :=
by sorry

end NUMINAMATH_CALUDE_function_bounds_l3836_383607


namespace NUMINAMATH_CALUDE_equation_equivalence_l3836_383656

theorem equation_equivalence (x : ℝ) : 
  (2*x + 1) / 3 - (5*x - 3) / 2 = 1 ↔ 2*(2*x + 1) - 3*(5*x - 3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3836_383656


namespace NUMINAMATH_CALUDE_min_sum_absolute_values_l3836_383613

theorem min_sum_absolute_values (x : ℝ) :
  ∃ (m : ℝ), m = -2 ∧ (∀ x, |x + 3| + |x + 5| + |x + 6| ≥ m) ∧ (∃ x, |x + 3| + |x + 5| + |x + 6| = m) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_absolute_values_l3836_383613


namespace NUMINAMATH_CALUDE_fourth_player_win_probability_prove_fourth_player_win_probability_l3836_383693

/-- The probability of the fourth player winning in a coin-flipping game -/
theorem fourth_player_win_probability : Real → Prop :=
  fun p =>
    -- Define the game setup
    let n_players : ℕ := 4
    let coin_prob : Real := 1 / 2
    -- Define the probability of the fourth player winning on their nth turn
    let prob_win_nth_turn : ℕ → Real := fun n => coin_prob ^ (n_players * n)
    -- Define the sum of the infinite geometric series
    let total_prob : Real := (prob_win_nth_turn 1) / (1 - prob_win_nth_turn 1)
    -- The theorem statement
    p = total_prob ∧ p = 1 / 31

/-- Proof of the theorem -/
theorem prove_fourth_player_win_probability : 
  ∃ p : Real, fourth_player_win_probability p :=
sorry

end NUMINAMATH_CALUDE_fourth_player_win_probability_prove_fourth_player_win_probability_l3836_383693


namespace NUMINAMATH_CALUDE_ellipse_focus_distance_l3836_383659

/-- An ellipse with equation x²/25 + y²/9 = 1 -/
structure Ellipse :=
  (x y : ℝ)
  (eq : x^2/25 + y^2/9 = 1)

/-- The distance from a point to a focus of the ellipse -/
def distance_to_focus (P : Ellipse) (focus : ℝ × ℝ) : ℝ :=
  sorry

theorem ellipse_focus_distance (P : Ellipse) (F1 F2 : ℝ × ℝ) :
  distance_to_focus P F1 = 3 →
  distance_to_focus P F2 = 7 :=
sorry

end NUMINAMATH_CALUDE_ellipse_focus_distance_l3836_383659


namespace NUMINAMATH_CALUDE_solution_difference_l3836_383629

theorem solution_difference (r s : ℝ) : 
  (((6 * r - 18) / (r^2 + 3*r - 18) = r + 3) ∧
   ((6 * s - 18) / (s^2 + 3*s - 18) = s + 3) ∧
   (r ≠ s) ∧
   (r > s)) → 
  (r - s = 11) := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l3836_383629


namespace NUMINAMATH_CALUDE_intersection_distance_and_difference_l3836_383663

theorem intersection_distance_and_difference : ∃ (x₁ x₂ : ℝ),
  (4 * x₁^2 + x₁ - 1 = 5) ∧
  (4 * x₂^2 + x₂ - 1 = 5) ∧
  (x₁ ≠ x₂) ∧
  (|x₁ - x₂| = Real.sqrt 97 / 4) ∧
  (97 - 4 = 93) := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_and_difference_l3836_383663


namespace NUMINAMATH_CALUDE_point_on_extension_line_l3836_383657

theorem point_on_extension_line (θ : ℝ) (M : ℝ × ℝ) :
  (∃ k : ℝ, k > 1 ∧ M = (k * Real.cos θ, k * Real.sin θ)) →
  (M.1^2 + M.2^2 = 4) →
  M = (-2 * Real.cos θ, -2 * Real.sin θ) := by
  sorry

end NUMINAMATH_CALUDE_point_on_extension_line_l3836_383657


namespace NUMINAMATH_CALUDE_tribe_leadership_count_l3836_383616

def tribe_size : ℕ := 15

def leadership_arrangements : ℕ :=
  tribe_size *
  (tribe_size - 1) *
  (tribe_size - 2) *
  (tribe_size - 3) *
  (tribe_size - 4) *
  (Nat.choose (tribe_size - 5) 2) *
  (Nat.choose (tribe_size - 7) 2)

theorem tribe_leadership_count :
  leadership_arrangements = 216216000 := by
  sorry

end NUMINAMATH_CALUDE_tribe_leadership_count_l3836_383616


namespace NUMINAMATH_CALUDE_age_ratio_problem_l3836_383618

/-- Proves that the ratio of b's age to c's age is 2:1 given the problem conditions -/
theorem age_ratio_problem (a b c : ℕ) : 
  a = b + 2 →  -- a is two years older than b
  a + b + c = 52 →  -- total of ages is 52
  b = 20 →  -- b is 20 years old
  b = 2 * c  -- ratio of b's age to c's age is 2:1
:= by sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l3836_383618


namespace NUMINAMATH_CALUDE_lunch_costs_more_than_breakfast_l3836_383600

/-- Represents the cost of Anna's meals -/
structure MealCosts where
  bagel : ℝ
  orange_juice : ℝ
  sandwich : ℝ
  milk : ℝ

/-- Calculates the difference between lunch and breakfast costs -/
def lunch_breakfast_difference (costs : MealCosts) : ℝ :=
  (costs.sandwich + costs.milk) - (costs.bagel + costs.orange_juice)

/-- Theorem stating the difference between lunch and breakfast costs -/
theorem lunch_costs_more_than_breakfast (costs : MealCosts) 
  (h1 : costs.bagel = 0.95)
  (h2 : costs.orange_juice = 0.85)
  (h3 : costs.sandwich = 4.65)
  (h4 : costs.milk = 1.15) :
  lunch_breakfast_difference costs = 4.00 := by
  sorry

end NUMINAMATH_CALUDE_lunch_costs_more_than_breakfast_l3836_383600


namespace NUMINAMATH_CALUDE_rectangle_perimeter_from_triangle_l3836_383660

/-- Given a triangle with sides 5, 12, and 13 units, and a rectangle with width 5 units
    and area equal to the triangle's area, the perimeter of the rectangle is 22 units. -/
theorem rectangle_perimeter_from_triangle : 
  ∀ (triangle_side1 triangle_side2 triangle_side3 rectangle_width : ℝ),
  triangle_side1 = 5 →
  triangle_side2 = 12 →
  triangle_side3 = 13 →
  rectangle_width = 5 →
  (1/2) * triangle_side1 * triangle_side2 = rectangle_width * (1/2 * triangle_side1 * triangle_side2 / rectangle_width) →
  2 * (rectangle_width + (1/2 * triangle_side1 * triangle_side2 / rectangle_width)) = 22 :=
by
  sorry


end NUMINAMATH_CALUDE_rectangle_perimeter_from_triangle_l3836_383660


namespace NUMINAMATH_CALUDE_reading_time_difference_l3836_383612

/-- Prove that the difference in reading time between two people is 360 minutes -/
theorem reading_time_difference 
  (xanthia_speed molly_speed : ℕ) 
  (book_length : ℕ) 
  (h1 : xanthia_speed = 120)
  (h2 : molly_speed = 40)
  (h3 : book_length = 360) :
  (book_length / molly_speed - book_length / xanthia_speed) * 60 = 360 := by
  sorry

#check reading_time_difference

end NUMINAMATH_CALUDE_reading_time_difference_l3836_383612


namespace NUMINAMATH_CALUDE_not_cube_of_integer_l3836_383694

theorem not_cube_of_integer : ¬ ∃ k : ℤ, (10^150 + 5 * 10^100 + 1 : ℤ) = k^3 := by
  sorry

end NUMINAMATH_CALUDE_not_cube_of_integer_l3836_383694


namespace NUMINAMATH_CALUDE_negation_of_universal_nonnegative_naturals_l3836_383686

theorem negation_of_universal_nonnegative_naturals :
  (¬ ∀ (x : ℕ), x ≥ 0) ↔ (∃ (x : ℕ), x < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_nonnegative_naturals_l3836_383686


namespace NUMINAMATH_CALUDE_green_apples_count_l3836_383630

theorem green_apples_count (total : ℕ) (red : ℕ) (yellow : ℕ) (h1 : total = 19) (h2 : red = 3) (h3 : yellow = 14) :
  total - (red + yellow) = 2 := by
  sorry

end NUMINAMATH_CALUDE_green_apples_count_l3836_383630


namespace NUMINAMATH_CALUDE_trig_identity_proof_l3836_383682

theorem trig_identity_proof : 
  2 * Real.cos (π / 6) - Real.tan (π / 3) + Real.sin (π / 4) * Real.cos (π / 4) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l3836_383682


namespace NUMINAMATH_CALUDE_composite_rectangle_area_l3836_383637

theorem composite_rectangle_area : 
  let rect1_area := 6 * 9
  let rect2_area := 4 * 6
  let rect3_area := 5 * 2
  rect1_area + rect2_area + rect3_area = 88 := by
  sorry

end NUMINAMATH_CALUDE_composite_rectangle_area_l3836_383637


namespace NUMINAMATH_CALUDE_abs_le_2_set_equality_l3836_383605

def set_of_integers_abs_le_2 : Set ℤ := {x | |x| ≤ 2}

theorem abs_le_2_set_equality : set_of_integers_abs_le_2 = {-2, -1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_abs_le_2_set_equality_l3836_383605


namespace NUMINAMATH_CALUDE_novelists_to_poets_ratio_l3836_383684

def total_people : ℕ := 24
def novelists : ℕ := 15

def poets : ℕ := total_people - novelists

theorem novelists_to_poets_ratio :
  (novelists : ℚ) / (poets : ℚ) = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_novelists_to_poets_ratio_l3836_383684


namespace NUMINAMATH_CALUDE_solution_set_when_m_neg_one_range_of_m_l3836_383680

-- Define the function f
def f (x m : ℝ) : ℝ := |x + m| + |2*x - 1|

-- Part I
theorem solution_set_when_m_neg_one :
  {x : ℝ | f x (-1) ≤ 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 4/3} := by sorry

-- Part II
def A (m : ℝ) : Set ℝ := {x : ℝ | f x m ≤ |2*x + 1|}

theorem range_of_m (h : Set.Icc (3/4 : ℝ) 2 ⊆ A m) :
  m ∈ Set.Icc (-11/4 : ℝ) 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_neg_one_range_of_m_l3836_383680


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3836_383604

theorem sum_of_roots_quadratic : ∃ (x₁ x₂ : ℤ),
  (x₁^2 = x₁ + 272) ∧ 
  (x₂^2 = x₂ + 272) ∧ 
  (∀ x : ℤ, x^2 = x + 272 → x = x₁ ∨ x = x₂) ∧
  (x₁ + x₂ = 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3836_383604


namespace NUMINAMATH_CALUDE_train_passing_time_l3836_383672

/-- The time taken for a train to pass a man moving in the opposite direction -/
theorem train_passing_time (train_speed : ℝ) (train_length : ℝ) (man_speed : ℝ) :
  train_speed = 60 →
  train_length = 110 →
  man_speed = 6 →
  ∃ t : ℝ, t > 0 ∧ t < 7 ∧
  t = train_length / ((train_speed + man_speed) * (1000 / 3600)) :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l3836_383672


namespace NUMINAMATH_CALUDE_trapezoid_bc_length_l3836_383645

/-- Trapezoid properties -/
structure Trapezoid :=
  (area : ℝ)
  (altitude : ℝ)
  (ab : ℝ)
  (cd : ℝ)

/-- Theorem: For a trapezoid with given properties, BC = 10 cm -/
theorem trapezoid_bc_length (t : Trapezoid) 
  (h1 : t.area = 200)
  (h2 : t.altitude = 10)
  (h3 : t.ab = 12)
  (h4 : t.cd = 22) :
  ∃ bc : ℝ, bc = 10 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_bc_length_l3836_383645


namespace NUMINAMATH_CALUDE_unique_solutions_l3836_383699

/-- A pair of positive integers (m, n) satisfies the given conditions if
    m^2 - 4n and n^2 - 4m are both perfect squares. -/
def satisfies_conditions (m n : ℕ+) : Prop :=
  ∃ k l : ℕ, (m : ℤ)^2 - 4*(n : ℤ) = (k : ℤ)^2 ∧ (n : ℤ)^2 - 4*(m : ℤ) = (l : ℤ)^2

/-- The theorem stating that the only pairs of positive integers (m, n) satisfying
    the conditions are (4, 4), (5, 6), and (6, 5). -/
theorem unique_solutions :
  ∀ m n : ℕ+, satisfies_conditions m n ↔ 
    ((m = 4 ∧ n = 4) ∨ (m = 5 ∧ n = 6) ∨ (m = 6 ∧ n = 5)) :=
by sorry

end NUMINAMATH_CALUDE_unique_solutions_l3836_383699


namespace NUMINAMATH_CALUDE_proof_arrangements_l3836_383641

/-- The number of letters in the word PROOF -/
def word_length : ℕ := 5

/-- The number of times the letter 'O' appears in PROOF -/
def o_count : ℕ := 2

/-- Formula for calculating the number of arrangements -/
def arrangements (n : ℕ) (k : ℕ) : ℕ := n.factorial / k.factorial

/-- Theorem stating that the number of unique arrangements of PROOF is 60 -/
theorem proof_arrangements : arrangements word_length o_count = 60 := by
  sorry

end NUMINAMATH_CALUDE_proof_arrangements_l3836_383641


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3836_383614

theorem cost_price_calculation (C : ℝ) : 0.18 * C - 0.09 * C = 72 → C = 800 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3836_383614


namespace NUMINAMATH_CALUDE_crop_allocation_theorem_l3836_383666

/-- Represents the yield function for crop A -/
def yield_A (x : ℝ) : ℝ := (2 + x) * (1.2 - 0.1 * x)

/-- Represents the maximum yield for crop A -/
def max_yield_A : ℝ := 4.9

/-- Represents the yield for crop B -/
def yield_B : ℝ := 10 * 0.5

/-- The total land area in square meters -/
def total_area : ℝ := 100

/-- The minimum required total yield in kg -/
def min_total_yield : ℝ := 496

theorem crop_allocation_theorem :
  ∃ (a : ℝ), a ≤ 40 ∧ a ≥ 0 ∧
  ∀ (x : ℝ), x ≤ 40 ∧ x ≥ 0 →
    max_yield_A * a + yield_B * (total_area - a) ≥ min_total_yield ∧
    (x > a → max_yield_A * x + yield_B * (total_area - x) < min_total_yield) :=
by sorry

end NUMINAMATH_CALUDE_crop_allocation_theorem_l3836_383666


namespace NUMINAMATH_CALUDE_no_partition_exists_l3836_383628

theorem no_partition_exists : ¬∃ (A B C : Set ℕ), 
  (A ∪ B ∪ C = Set.univ) ∧ 
  (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (C ∩ A = ∅) ∧
  (A ≠ ∅) ∧ (B ≠ ∅) ∧ (C ≠ ∅) ∧
  (∀ a b, a ∈ A → b ∈ B → a + b + 2008 ∈ C) ∧
  (∀ b c, b ∈ B → c ∈ C → b + c + 2008 ∈ A) ∧
  (∀ c a, c ∈ C → a ∈ A → c + a + 2008 ∈ B) := by
sorry

end NUMINAMATH_CALUDE_no_partition_exists_l3836_383628


namespace NUMINAMATH_CALUDE_greatest_x_value_l3836_383653

theorem greatest_x_value : 
  (∃ (x : ℤ), ∀ (y : ℤ), (2.13 * (10 : ℝ)^(y : ℝ) < 2100) → y ≤ x) ∧ 
  (2.13 * (10 : ℝ)^(2 : ℝ) < 2100) ∧ 
  (∀ (z : ℤ), z > 2 → 2.13 * (10 : ℝ)^(z : ℝ) ≥ 2100) :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l3836_383653


namespace NUMINAMATH_CALUDE_no_arithmetic_progression_l3836_383667

theorem no_arithmetic_progression (m : ℕ+) :
  ∃ σ : Fin (2^m.val) ↪ Fin (2^m.val),
    ∀ (i j k : Fin (2^m.val)), i < j → j < k →
      σ j - σ i ≠ σ k - σ j :=
by sorry

end NUMINAMATH_CALUDE_no_arithmetic_progression_l3836_383667


namespace NUMINAMATH_CALUDE_expression_simplification_l3836_383615

theorem expression_simplification (q : ℝ) : 
  ((7*q + 3) - 3*q*5)*4 + (5 - 2/4)*(8*q - 12) = 4*q - 42 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3836_383615


namespace NUMINAMATH_CALUDE_mean_temperature_l3836_383651

def temperatures : List ℤ := [-3, 0, 2, -1, 4, 5, 3]

theorem mean_temperature : 
  (List.sum temperatures) / (List.length temperatures) = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l3836_383651


namespace NUMINAMATH_CALUDE_binomial_12_3_l3836_383647

theorem binomial_12_3 : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_3_l3836_383647


namespace NUMINAMATH_CALUDE_absolute_value_symmetry_axis_of_symmetry_is_three_l3836_383635

/-- The axis of symmetry for the absolute value function y = |x-a| --/
def axisOfSymmetry (a : ℝ) : ℝ := a

/-- A function is symmetric about a vertical line if it remains unchanged when reflected about that line --/
def isSymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

theorem absolute_value_symmetry (a : ℝ) :
  isSymmetricAbout (fun x ↦ |x - a|) (axisOfSymmetry a) := by sorry

theorem axis_of_symmetry_is_three (a : ℝ) :
  axisOfSymmetry a = 3 → a = 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_symmetry_axis_of_symmetry_is_three_l3836_383635


namespace NUMINAMATH_CALUDE_sum_of_min_max_x_l3836_383681

theorem sum_of_min_max_x (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) :
  ∃ (m M : ℝ), (∀ x' y' z' : ℝ, x' + y' + z' = 5 → x'^2 + y'^2 + z'^2 = 11 → m ≤ x' ∧ x' ≤ M) ∧
                m + M = 8/3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_min_max_x_l3836_383681


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_one_third_l3836_383652

theorem reciprocal_of_repeating_decimal_one_third (x : ℚ) : 
  (x = 1/3) → (1/x = 3) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_one_third_l3836_383652


namespace NUMINAMATH_CALUDE_quadratic_function_max_min_l3836_383661

theorem quadratic_function_max_min (a b : ℝ) (h1 : a ≠ 0) : 
  let f : ℝ → ℝ := λ x ↦ a * x^2 - 2 * a * x + b
  (∃ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 1 2, f y ≤ f x) ∧
  (∃ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 1 2, f y ≥ f x) ∧
  (∃ x ∈ Set.Icc 1 2, f x = 0) ∧
  (∃ x ∈ Set.Icc 1 2, f x = -1) →
  ((a = 1 ∧ b = 0) ∨ (a = -1 ∧ b = -1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_max_min_l3836_383661


namespace NUMINAMATH_CALUDE_shaded_area_circles_l3836_383670

theorem shaded_area_circles (R : ℝ) (r : ℝ) : 
  R^2 * π = 100 * π →
  r = R / 2 →
  (2 / 3) * (π * R^2) + (1 / 3) * (π * r^2) = 75 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_circles_l3836_383670


namespace NUMINAMATH_CALUDE_women_reseating_l3836_383610

/-- The number of ways to reseat n women in a circle, where each woman can sit in her original seat,
    an adjacent seat, or two seats away. -/
def C : ℕ → ℕ
  | 0 => 0  -- Added for completeness
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | n + 4 => 2 * C (n + 3) + 2 * C (n + 2) + C (n + 1)

/-- The number of ways to reseat 9 women in a circle, where each woman can sit in her original seat,
    an adjacent seat, or two seats away, is equal to 3086. -/
theorem women_reseating : C 9 = 3086 := by
  sorry

end NUMINAMATH_CALUDE_women_reseating_l3836_383610


namespace NUMINAMATH_CALUDE_distinct_pairs_solution_l3836_383620

theorem distinct_pairs_solution (x y : ℝ) : 
  x ≠ y ∧ 
  x^100 - y^100 = 2^99 * (x - y) ∧ 
  x^200 - y^200 = 2^199 * (x - y) → 
  (x = 2 ∧ y = 0) ∨ (x = 0 ∧ y = 2) := by
sorry

end NUMINAMATH_CALUDE_distinct_pairs_solution_l3836_383620


namespace NUMINAMATH_CALUDE_inequality_solution_l3836_383602

theorem inequality_solution : 
  ∀ x y : ℤ, 
    (x - 3*y + 2 ≥ 1) → 
    (-x + 2*y + 1 ≥ 1) → 
    (x^2 / Real.sqrt (x - 3*y + 2 : ℝ) + y^2 / Real.sqrt (-x + 2*y + 1 : ℝ) ≥ y^2 + 2*x^2 - 2*x - 1) →
    ((x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = 1)) := by
  sorry

#check inequality_solution

end NUMINAMATH_CALUDE_inequality_solution_l3836_383602


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3836_383648

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (10 + n) = 8 → n = 54 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3836_383648


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l3836_383624

def S : Set ℝ := {x | 2 * x + 1 > 0}
def T : Set ℝ := {x | 3 * x - 5 < 0}

theorem set_intersection_theorem :
  S ∩ T = {x : ℝ | -1/2 < x ∧ x < 5/3} :=
by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l3836_383624


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3836_383690

theorem complex_fraction_simplification :
  (((1 : ℂ) + 2 * Complex.I) ^ 2) / ((3 : ℂ) - 4 * Complex.I) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3836_383690


namespace NUMINAMATH_CALUDE_cos_135_degrees_l3836_383698

theorem cos_135_degrees : Real.cos (135 * π / 180) = -1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l3836_383698


namespace NUMINAMATH_CALUDE_foil_covered_prism_width_l3836_383692

/-- The width of a foil-covered rectangular prism -/
theorem foil_covered_prism_width :
  ∀ (inner_length inner_width inner_height : ℝ),
    inner_length * inner_width * inner_height = 128 →
    inner_width = 2 * inner_length →
    inner_width = 2 * inner_height →
    ∃ (outer_width : ℝ),
      outer_width = 4 * (2 : ℝ)^(1/3) + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_foil_covered_prism_width_l3836_383692


namespace NUMINAMATH_CALUDE_f_is_decreasing_l3836_383631

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom not_identically_zero : ∃ x, f x ≠ 0
axiom additivity : ∀ x y, f (x + y) = f x + f y
axiom negative_for_positive : ∀ x, x > 0 → f x < 0

-- State the theorem
theorem f_is_decreasing : 
  (∀ x y, x < y → f x > f y) :=
sorry

end NUMINAMATH_CALUDE_f_is_decreasing_l3836_383631


namespace NUMINAMATH_CALUDE_simplify_expression_l3836_383665

theorem simplify_expression (z : ℝ) : (5 - 4 * z^2) - (7 - 6 * z + 3 * z^2) = -2 - 7 * z^2 + 6 * z := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3836_383665


namespace NUMINAMATH_CALUDE_sqrt_2_irrational_sqrt_2_only_irrational_in_set_l3836_383662

-- Define what it means for a real number to be rational
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define what it means for a real number to be irrational
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- State the theorem
theorem sqrt_2_irrational : IsIrrational (Real.sqrt 2) := by
  sorry

-- Define the set of numbers from the original problem
def problem_numbers : Set ℝ := {0, -1, Real.sqrt 2, 3.14}

-- State that √2 is the only irrational number in the set
theorem sqrt_2_only_irrational_in_set : 
  ∀ x ∈ problem_numbers, IsIrrational x ↔ x = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_irrational_sqrt_2_only_irrational_in_set_l3836_383662


namespace NUMINAMATH_CALUDE_diamond_fifteen_two_l3836_383668

-- Define the diamond operation
def diamond (a b : ℤ) : ℚ := a + a / (b + 1)

-- State the theorem
theorem diamond_fifteen_two : diamond 15 2 = 20 := by sorry

end NUMINAMATH_CALUDE_diamond_fifteen_two_l3836_383668


namespace NUMINAMATH_CALUDE_rectangle_side_length_l3836_383691

/-- Given three rectangles with equal areas and integer sides, where one side is 37, prove that a specific side length is 1406. -/
theorem rectangle_side_length (a b : ℕ) : 
  let S := 37 * (a + b)  -- Common area of the rectangles
  -- ABCD area
  S = 37 * (a + b) →
  -- DEFG area
  S = a * 1406 →
  -- CEIH area
  S = b * 38 →
  -- All sides are integers
  a > 0 → b > 0 →
  -- DG length
  1406 = 1406 := by sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l3836_383691


namespace NUMINAMATH_CALUDE_equation_solution_l3836_383603

theorem equation_solution : ∃! x : ℝ, (1 / (x - 1) = 3 / (2 * x - 3)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3836_383603


namespace NUMINAMATH_CALUDE_reflection_of_circle_center_l3836_383649

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem reflection_of_circle_center :
  let original_center : ℝ × ℝ := (-3, 7)
  let reflected_center : ℝ × ℝ := (-7, 3)
  reflect_about_y_eq_neg_x original_center = reflected_center := by
sorry

end NUMINAMATH_CALUDE_reflection_of_circle_center_l3836_383649


namespace NUMINAMATH_CALUDE_second_class_revenue_l3836_383664

/-- The amount collected from II class passengers given the passenger and fare ratios --/
theorem second_class_revenue (total_revenue : ℚ) 
  (h1 : total_revenue = 1325)
  (h2 : ∃ (x y : ℚ), x * y * 53 = total_revenue ∧ x > 0 ∧ y > 0) :
  ∃ (x y : ℚ), 50 * x * y = 1250 :=
sorry

end NUMINAMATH_CALUDE_second_class_revenue_l3836_383664


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l3836_383674

theorem absolute_value_equation_solutions :
  ∀ x : ℝ, (3 * x + 9 = |(-20 + 4 * x)|) ↔ (x = 29 ∨ x = 11/7) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l3836_383674
