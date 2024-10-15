import Mathlib

namespace NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l1578_157841

-- Define the cubic polynomial
def q (x : ℚ) : ℚ := 7/4 * x^3 - 19 * x^2 + 149/4 * x + 6

-- Theorem statement
theorem cubic_polynomial_satisfies_conditions :
  q 1 = -6 ∧ q 3 = -20 ∧ q 4 = -42 ∧ q 5 = -60 := by
  sorry


end NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l1578_157841


namespace NUMINAMATH_CALUDE_frog_climb_proof_l1578_157890

def well_depth : ℝ := 4

def climb_distances : List ℝ := [1.2, 1.4, 1.1, 1.2]
def slide_distances : List ℝ := [0.4, 0.5, 0.3, 0.2]

def net_distance_climbed : ℝ := 
  List.sum (List.zipWith (·-·) climb_distances slide_distances)

def total_distance_covered : ℝ := 
  List.sum climb_distances + List.sum slide_distances

def fifth_climb_distance : ℝ := 1.2

theorem frog_climb_proof :
  (well_depth - net_distance_climbed = 0.5) ∧
  (total_distance_covered = 6.3) ∧
  (net_distance_climbed + fifth_climb_distance > well_depth) := by
  sorry

end NUMINAMATH_CALUDE_frog_climb_proof_l1578_157890


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l1578_157889

/-- Given a train of length 1100 meters that crosses a tree in 110 seconds,
    prove that it takes 180 seconds to pass a platform of length 700 meters. -/
theorem train_platform_crossing_time :
  let train_length : ℝ := 1100
  let tree_crossing_time : ℝ := 110
  let platform_length : ℝ := 700
  let train_speed : ℝ := train_length / tree_crossing_time
  let total_distance : ℝ := train_length + platform_length
  total_distance / train_speed = 180 := by sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l1578_157889


namespace NUMINAMATH_CALUDE_lower_bound_of_fraction_l1578_157813

theorem lower_bound_of_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1 / (3 * a) + 3 / b ≥ 8 / 3 := by
sorry

end NUMINAMATH_CALUDE_lower_bound_of_fraction_l1578_157813


namespace NUMINAMATH_CALUDE_abc_product_l1578_157803

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 24 * Real.sqrt 3)
  (hac : a * c = 30 * Real.sqrt 3)
  (hbc : b * c = 40 * Real.sqrt 3) :
  a * b * c = 120 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l1578_157803


namespace NUMINAMATH_CALUDE_digit_count_problem_l1578_157824

theorem digit_count_problem (n : ℕ) 
  (h1 : (n : ℝ) * 500 = 14 * 390 + 6 * 756.67)
  (h2 : n > 0) : 
  n = 20 := by
  sorry

end NUMINAMATH_CALUDE_digit_count_problem_l1578_157824


namespace NUMINAMATH_CALUDE_sqrt_plus_inverse_geq_two_l1578_157883

theorem sqrt_plus_inverse_geq_two (x : ℝ) (hx : x > 0) : Real.sqrt x + 1 / Real.sqrt x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_plus_inverse_geq_two_l1578_157883


namespace NUMINAMATH_CALUDE_complex_number_location_l1578_157856

theorem complex_number_location :
  let z : ℂ := (3 + Complex.I) / (1 + Complex.I)
  (0 < z.re) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l1578_157856


namespace NUMINAMATH_CALUDE_binary_101_equals_5_l1578_157857

/- Define a function to convert binary to decimal -/
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/- Define the binary number 101 -/
def binary_101 : List Bool := [true, false, true]

/- Theorem statement -/
theorem binary_101_equals_5 :
  binary_to_decimal binary_101 = 5 := by sorry

end NUMINAMATH_CALUDE_binary_101_equals_5_l1578_157857


namespace NUMINAMATH_CALUDE_intersection_problem_l1578_157869

-- Define the functions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1
def g (c d : ℝ) (x : ℝ) : ℝ := x^2 + c * x + d

-- State the theorem
theorem intersection_problem (a b c d : ℝ) :
  (f a b 2 = 4) →  -- The graphs intersect at x = 2
  (g c d 2 = 4) →  -- The graphs intersect at x = 2
  (b + c = 1) →    -- Given condition
  (4 * a + d = 1)  -- What we want to prove
:= by sorry

end NUMINAMATH_CALUDE_intersection_problem_l1578_157869


namespace NUMINAMATH_CALUDE_exists_convex_quadrilateral_geometric_progression_l1578_157888

/-- A convex quadrilateral with sides a₁, a₂, a₃, a₄ and diagonals d₁, d₂ -/
structure ConvexQuadrilateral where
  a₁ : ℝ
  a₂ : ℝ
  a₃ : ℝ
  a₄ : ℝ
  d₁ : ℝ
  d₂ : ℝ
  a₁_pos : a₁ > 0
  a₂_pos : a₂ > 0
  a₃_pos : a₃ > 0
  a₄_pos : a₄ > 0
  d₁_pos : d₁ > 0
  d₂_pos : d₂ > 0
  convex : a₁ + a₂ + a₃ > a₄ ∧
           a₁ + a₂ + a₄ > a₃ ∧
           a₁ + a₃ + a₄ > a₂ ∧
           a₂ + a₃ + a₄ > a₁

/-- Predicate to check if a sequence forms a geometric progression -/
def IsGeometricProgression (seq : List ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ i : Fin (seq.length - 1), seq[i.val + 1] = seq[i.val] * r

/-- Theorem stating the existence of a convex quadrilateral with sides and diagonals
    forming a geometric progression -/
theorem exists_convex_quadrilateral_geometric_progression :
  ∃ q : ConvexQuadrilateral, IsGeometricProgression [q.a₁, q.a₂, q.a₃, q.a₄, q.d₁, q.d₂] :=
sorry

end NUMINAMATH_CALUDE_exists_convex_quadrilateral_geometric_progression_l1578_157888


namespace NUMINAMATH_CALUDE_complex_division_equality_l1578_157850

theorem complex_division_equality : (3 - I) / (1 + I) = 1 - 2*I := by sorry

end NUMINAMATH_CALUDE_complex_division_equality_l1578_157850


namespace NUMINAMATH_CALUDE_hexagon_diagonal_intersection_probability_l1578_157860

/-- A convex hexagon -/
structure ConvexHexagon where
  -- We don't need to define the structure of a hexagon for this problem

/-- A diagonal of a hexagon -/
structure Diagonal (h : ConvexHexagon) where
  -- We don't need to define the structure of a diagonal for this problem

/-- Predicate to check if two diagonals intersect inside the hexagon -/
def intersect_inside (h : ConvexHexagon) (d1 d2 : Diagonal h) : Prop :=
  sorry  -- Definition not provided, as it's not necessary for the statement

/-- The probability of two randomly chosen diagonals intersecting inside the hexagon -/
def intersection_probability (h : ConvexHexagon) : ℚ :=
  sorry  -- Definition not provided, as it's not necessary for the statement

/-- Theorem stating that the probability of two randomly chosen diagonals 
    intersecting inside a convex hexagon is 5/12 -/
theorem hexagon_diagonal_intersection_probability (h : ConvexHexagon) :
  intersection_probability h = 5 / 12 :=
sorry

end NUMINAMATH_CALUDE_hexagon_diagonal_intersection_probability_l1578_157860


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l1578_157855

theorem sum_of_x_and_y_is_two (x y : ℝ) 
  (hx : (x - 1)^3 + 1997*(x - 1) = -1)
  (hy : (y - 1)^3 + 1997*(y - 1) = 1) : 
  x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l1578_157855


namespace NUMINAMATH_CALUDE_five_letter_words_same_start_end_l1578_157840

theorem five_letter_words_same_start_end (alphabet_size : ℕ) (word_length : ℕ) : 
  alphabet_size = 26 → word_length = 5 → 
  (alphabet_size ^ (word_length - 2)) * alphabet_size = 456976 := by
  sorry

end NUMINAMATH_CALUDE_five_letter_words_same_start_end_l1578_157840


namespace NUMINAMATH_CALUDE_triangle_properties_l1578_157800

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  c = Real.sqrt 7 →
  4 * Real.sin (((A + B) / 2) : ℝ)^2 - Real.cos (2 * C) = 7/2 →
  C = π/3 ∧
  (∀ (a' b' : ℝ), (1/2) * a' * b' * Real.sin C ≤ (7 * Real.sqrt 3) / 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1578_157800


namespace NUMINAMATH_CALUDE_lent_sum_theorem_l1578_157843

/-- Represents the sum of money lent in two parts -/
structure LentSum where
  first_part : ℕ
  second_part : ℕ
  total : ℕ

/-- Calculates the interest on a principal amount for a given rate and time -/
def calculate_interest (principal : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  principal * rate * time

theorem lent_sum_theorem (s : LentSum) :
  s.second_part = 1672 →
  calculate_interest s.first_part 3 8 = calculate_interest s.second_part 5 3 →
  s.total = s.first_part + s.second_part →
  s.total = 2717 := by
    sorry

end NUMINAMATH_CALUDE_lent_sum_theorem_l1578_157843


namespace NUMINAMATH_CALUDE_absolute_value_of_five_minus_pi_plus_two_l1578_157826

theorem absolute_value_of_five_minus_pi_plus_two : |5 - Real.pi + 2| = 7 - Real.pi := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_five_minus_pi_plus_two_l1578_157826


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l1578_157898

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint where
  x : ℚ
  y : ℚ

/-- Definition of the first line: y = -3x -/
def line1 (x y : ℚ) : Prop := y = -3 * x

/-- Definition of the second line: y - 3 = 9x -/
def line2 (x y : ℚ) : Prop := y - 3 = 9 * x

/-- Theorem stating that (-1/4, 3/4) is the unique intersection point of the two lines -/
theorem intersection_point_of_lines :
  ∃! p : IntersectionPoint, line1 p.x p.y ∧ line2 p.x p.y ∧ p.x = -1/4 ∧ p.y = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l1578_157898


namespace NUMINAMATH_CALUDE_square_of_complex_number_l1578_157823

theorem square_of_complex_number :
  let z : ℂ := 5 - 3*I
  z^2 = 16 - 30*I := by sorry

end NUMINAMATH_CALUDE_square_of_complex_number_l1578_157823


namespace NUMINAMATH_CALUDE_spider_group_ratio_l1578_157849

/-- Represents a group of spiders -/
structure SpiderGroup where
  /-- Number of spiders in the group -/
  count : ℕ
  /-- Number of legs per spider -/
  legsPerSpider : ℕ
  /-- The group has more spiders than half the legs of a single spider -/
  more_than_half : count > legsPerSpider / 2
  /-- Total number of legs in the group -/
  totalLegs : ℕ
  /-- The total legs is the product of count and legs per spider -/
  total_legs_eq : totalLegs = count * legsPerSpider

/-- The theorem to be proved -/
theorem spider_group_ratio (g : SpiderGroup)
  (h1 : g.legsPerSpider = 8)
  (h2 : g.totalLegs = 112) :
  (g.count : ℚ) / (g.legsPerSpider / 2 : ℚ) = 7 / 2 :=
sorry

end NUMINAMATH_CALUDE_spider_group_ratio_l1578_157849


namespace NUMINAMATH_CALUDE_existence_of_h_l1578_157806

theorem existence_of_h : ∃ h : ℝ, ∀ n : ℕ, 
  ¬(⌊h * 1969^n⌋ ∣ ⌊h * 1969^(n-1)⌋) :=
sorry

end NUMINAMATH_CALUDE_existence_of_h_l1578_157806


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l1578_157853

theorem partial_fraction_decomposition_sum (A B C D E F : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -4 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l1578_157853


namespace NUMINAMATH_CALUDE_age_sum_problem_l1578_157835

theorem age_sum_problem (twin1_age twin2_age youngest_age : ℕ) :
  twin1_age = twin2_age →
  twin1_age > youngest_age →
  youngest_age < 10 →
  twin1_age * twin2_age * youngest_age = 72 →
  twin1_age + twin2_age + youngest_age = 14 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_problem_l1578_157835


namespace NUMINAMATH_CALUDE_x_power_125_minus_reciprocal_l1578_157851

theorem x_power_125_minus_reciprocal (x : ℝ) (h : x - 1/x = Real.sqrt 3) :
  x^125 - 1/x^125 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_x_power_125_minus_reciprocal_l1578_157851


namespace NUMINAMATH_CALUDE_cos_negative_300_degrees_l1578_157896

theorem cos_negative_300_degrees : Real.cos (-300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_300_degrees_l1578_157896


namespace NUMINAMATH_CALUDE_S_properties_l1578_157846

def S : Set ℤ := {x | ∃ n : ℤ, x = (n-1)^2 + n^2 + (n+1)^2}

theorem S_properties : 
  (∀ x ∈ S, ¬(3 ∣ x)) ∧ (∃ x ∈ S, 11 ∣ x) := by
  sorry

end NUMINAMATH_CALUDE_S_properties_l1578_157846


namespace NUMINAMATH_CALUDE_polynomial_equality_l1578_157870

theorem polynomial_equality (t s : ℚ) : 
  (∀ x : ℚ, (3*x^2 - 4*x + 9) * (5*x^2 + t*x + 12) = 15*x^4 + s*x^3 + 33*x^2 + 12*x + 108) 
  ↔ 
  (t = 37/5 ∧ s = 11/5) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1578_157870


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1578_157815

theorem quadratic_roots_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + 2 * x + 1 = 0 ∧ a * y^2 + 2 * y + 1 = 0) →
  a < 1 ∧ a ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1578_157815


namespace NUMINAMATH_CALUDE_bronze_ball_balance_l1578_157825

theorem bronze_ball_balance (a : Fin 10 → ℝ) : 
  ∃ (S : Finset (Fin 10)), 
    (S.sum (λ i => |a (i + 1) - a i|)) = 
    ((Finset.univ \ S).sum (λ i => |a (i + 1) - a i|)) := by
  sorry


end NUMINAMATH_CALUDE_bronze_ball_balance_l1578_157825


namespace NUMINAMATH_CALUDE_arbitrarily_large_N_exists_l1578_157859

def is_increasing_seq (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, x n < x (n + 1)

def limit_zero (x : ℕ → ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n / n| < ε

theorem arbitrarily_large_N_exists (x : ℕ → ℝ) 
  (h_pos : ∀ n, x n > 0)
  (h_inc : is_increasing_seq x)
  (h_lim : limit_zero x) :
  ∀ M : ℕ, ∃ N > M, ∀ i : ℕ, 1 ≤ i → i < N → x i + x (2*N - i) < 2 * x N :=
sorry

end NUMINAMATH_CALUDE_arbitrarily_large_N_exists_l1578_157859


namespace NUMINAMATH_CALUDE_revenue_in_scientific_notation_l1578_157805

/-- Represents the value of 1 billion in scientific notation -/
def billion : ℝ := 10^9

/-- The box office revenue in billions -/
def revenue : ℝ := 53.96

theorem revenue_in_scientific_notation :
  revenue * billion = 5.396 * 10^10 := by sorry

end NUMINAMATH_CALUDE_revenue_in_scientific_notation_l1578_157805


namespace NUMINAMATH_CALUDE_simplify_polynomial_expression_l1578_157867

/-- Given two polynomials A and B in x and y, prove that 2A - B simplifies to a specific form. -/
theorem simplify_polynomial_expression (x y : ℝ) :
  let A := 2 * x^2 + x * y - 3
  let B := -x^2 + 2 * x * y - 1
  2 * A - B = 5 * x^2 - 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_expression_l1578_157867


namespace NUMINAMATH_CALUDE_seokmin_school_cookies_l1578_157811

/-- The number of boxes of cookies needed for a given number of students -/
def cookies_boxes_needed (num_students : ℕ) (cookies_per_student : ℕ) (cookies_per_box : ℕ) : ℕ :=
  ((num_students * cookies_per_student + cookies_per_box - 1) / cookies_per_box : ℕ)

/-- Theorem stating the number of boxes needed for Seokmin's school -/
theorem seokmin_school_cookies :
  cookies_boxes_needed 134 7 28 = 34 := by
  sorry

end NUMINAMATH_CALUDE_seokmin_school_cookies_l1578_157811


namespace NUMINAMATH_CALUDE_average_of_middle_two_l1578_157897

theorem average_of_middle_two (total_avg : ℝ) (first_two_avg : ℝ) (last_two_avg : ℝ) :
  total_avg = 3.95 →
  first_two_avg = 3.4 →
  last_two_avg = 4.600000000000001 →
  (6 * total_avg - 2 * first_two_avg - 2 * last_two_avg) / 2 = 3.85 := by
  sorry

end NUMINAMATH_CALUDE_average_of_middle_two_l1578_157897


namespace NUMINAMATH_CALUDE_solution_count_l1578_157877

/-- For any integer k > 1, there exist at least 3k + 1 distinct triples of positive integers (m, n, r) 
    satisfying the equation mn + nr + mr = k(m + n + r). -/
theorem solution_count (k : ℕ) (h : k > 1) : 
  ∃ S : Finset (ℕ × ℕ × ℕ), 
    (∀ (m n r : ℕ), (m, n, r) ∈ S → m > 0 ∧ n > 0 ∧ r > 0) ∧ 
    (∀ (m n r : ℕ), (m, n, r) ∈ S → m * n + n * r + m * r = k * (m + n + r)) ∧
    S.card ≥ 3 * k + 1 :=
sorry

end NUMINAMATH_CALUDE_solution_count_l1578_157877


namespace NUMINAMATH_CALUDE_book_costs_proof_l1578_157831

theorem book_costs_proof (total_cost : ℝ) (book1 book2 book3 book4 book5 : ℝ) :
  total_cost = 24 ∧
  book1 = book2 + 2 ∧
  book3 = book1 + 4 ∧
  book4 = book3 - 3 ∧
  book5 = book2 ∧
  book1 ≠ book2 ∧ book1 ≠ book3 ∧ book1 ≠ book4 ∧ book1 ≠ book5 ∧
  book2 ≠ book3 ∧ book2 ≠ book4 ∧
  book3 ≠ book4 ∧ book3 ≠ book5 ∧
  book4 ≠ book5 →
  book1 = 4.6 ∧ book2 = 2.6 ∧ book3 = 8.6 ∧ book4 = 5.6 ∧ book5 = 2.6 ∧
  total_cost = book1 + book2 + book3 + book4 + book5 := by
sorry

end NUMINAMATH_CALUDE_book_costs_proof_l1578_157831


namespace NUMINAMATH_CALUDE_residue_calculation_l1578_157862

theorem residue_calculation : 196 * 18 - 21 * 9 + 5 ≡ 14 [ZMOD 18] := by
  sorry

end NUMINAMATH_CALUDE_residue_calculation_l1578_157862


namespace NUMINAMATH_CALUDE_negation_of_universal_positive_square_l1578_157874

theorem negation_of_universal_positive_square (P : ℝ → Prop) : 
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_positive_square_l1578_157874


namespace NUMINAMATH_CALUDE_twentieth_term_is_negative_49_l1578_157848

/-- An arithmetic sequence is defined by its first term and common difference. -/
structure ArithmeticSequence where
  firstTerm : ℤ
  commonDiff : ℤ

/-- The nth term of an arithmetic sequence. -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.firstTerm + (n - 1 : ℤ) * seq.commonDiff

/-- The theorem stating that the 20th term of the specific arithmetic sequence is -49. -/
theorem twentieth_term_is_negative_49 :
  let seq := ArithmeticSequence.mk 8 (-3)
  nthTerm seq 20 = -49 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_is_negative_49_l1578_157848


namespace NUMINAMATH_CALUDE_x_value_in_sequence_l1578_157827

def fibonacci_like_sequence (a b : ℤ) : ℕ → ℤ
  | 0 => a
  | 1 => b
  | n+2 => fibonacci_like_sequence a b n + fibonacci_like_sequence a b (n+1)

theorem x_value_in_sequence :
  ∃ (start : ℕ), 
    (fibonacci_like_sequence (-1) 2 (start + 2) = 3) ∧
    (fibonacci_like_sequence (-1) 2 (start + 3) = 5) ∧
    (fibonacci_like_sequence (-1) 2 (start + 4) = 8) ∧
    (fibonacci_like_sequence (-1) 2 (start + 5) = 13) ∧
    (fibonacci_like_sequence (-1) 2 (start + 6) = 21) ∧
    (fibonacci_like_sequence (-1) 2 (start + 7) = 34) ∧
    (fibonacci_like_sequence (-1) 2 (start + 8) = 55) := by
  sorry

end NUMINAMATH_CALUDE_x_value_in_sequence_l1578_157827


namespace NUMINAMATH_CALUDE_vicente_spent_25_l1578_157885

/-- The total amount Vicente spent on rice and meat -/
def total_spent (rice_kg : ℕ) (rice_price : ℚ) (meat_lb : ℕ) (meat_price : ℚ) : ℚ :=
  (rice_kg : ℚ) * rice_price + (meat_lb : ℚ) * meat_price

/-- Proof that Vicente spent $25 on his purchase -/
theorem vicente_spent_25 :
  total_spent 5 2 3 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_vicente_spent_25_l1578_157885


namespace NUMINAMATH_CALUDE_final_result_proof_l1578_157854

theorem final_result_proof (chosen_number : ℕ) (h : chosen_number = 1376) :
  (chosen_number / 8 : ℚ) - 160 = 12 := by
  sorry

end NUMINAMATH_CALUDE_final_result_proof_l1578_157854


namespace NUMINAMATH_CALUDE_unique_solution_l1578_157828

def is_valid_digit (d : ℕ) : Prop := d ≥ 1 ∧ d ≤ 8

def are_distinct (a b c d e f g h : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
  f ≠ g ∧ f ≠ h ∧
  g ≠ h

def four_digit_number (a b c d : ℕ) : ℕ := a * 1000 + b * 100 + c * 10 + d

theorem unique_solution (a b c d e f g h : ℕ) :
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ is_valid_digit d ∧
  is_valid_digit e ∧ is_valid_digit f ∧ is_valid_digit g ∧ is_valid_digit h ∧
  are_distinct a b c d e f g h ∧
  four_digit_number a b c d + e * f * g * h = 2011 →
  four_digit_number a b c d = 1563 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l1578_157828


namespace NUMINAMATH_CALUDE_not_both_bidirectional_l1578_157891

-- Define the two proof methods
inductive ProofMethod
| Synthetic
| Analytic

-- Define the reasoning direction
inductive ReasoningDirection
| CauseToEffect
| EffectToCause

-- Define the properties of the proof methods
def methodDirection (m : ProofMethod) : ReasoningDirection :=
  match m with
  | ProofMethod.Synthetic => ReasoningDirection.CauseToEffect
  | ProofMethod.Analytic => ReasoningDirection.EffectToCause

-- Theorem statement
theorem not_both_bidirectional :
  ¬(∀ (m : ProofMethod), 
    (methodDirection m = ReasoningDirection.CauseToEffect ∧
     methodDirection m = ReasoningDirection.EffectToCause)) :=
by
  sorry

end NUMINAMATH_CALUDE_not_both_bidirectional_l1578_157891


namespace NUMINAMATH_CALUDE_larger_quadrilateral_cyclic_l1578_157842

/-- A configuration of five cyclic quadrilaterals forming a larger quadrilateral -/
structure FiveQuadrilateralsConfig where
  /-- The angles of the five smaller quadrilaterals -/
  angles : Fin 10 → ℝ
  /-- Each smaller quadrilateral is cyclic -/
  cyclic_small : ∀ i : Fin 5, angles (2*i) + angles (2*i+1) = 180
  /-- The sum of angles around each internal vertex is 360° -/
  vertex_sum : angles 1 + angles 7 + angles 8 = 360 ∧ angles 3 + angles 5 + angles 9 = 360

/-- The theorem stating that the larger quadrilateral is cyclic -/
theorem larger_quadrilateral_cyclic (config : FiveQuadrilateralsConfig) :
  config.angles 0 + config.angles 2 + config.angles 4 + config.angles 6 = 180 :=
sorry

end NUMINAMATH_CALUDE_larger_quadrilateral_cyclic_l1578_157842


namespace NUMINAMATH_CALUDE_complex_product_l1578_157872

/-- Given complex numbers Q, E, and D, prove that their product is -25i. -/
theorem complex_product (Q E D : ℂ) : 
  Q = 3 + 4*I ∧ E = -I ∧ D = 3 - 4*I → Q * E * D = -25 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_product_l1578_157872


namespace NUMINAMATH_CALUDE_fifteen_points_max_planes_l1578_157804

/-- The maximum number of planes determined by n points in space,
    where no four points are coplanar. -/
def maxPlanes (n : ℕ) : ℕ := Nat.choose n 3

/-- The condition that no four points are coplanar is implicitly assumed
    in the definition of maxPlanes. -/
theorem fifteen_points_max_planes :
  maxPlanes 15 = 455 := by sorry

end NUMINAMATH_CALUDE_fifteen_points_max_planes_l1578_157804


namespace NUMINAMATH_CALUDE_max_mogs_bill_can_buy_l1578_157836

/-- The cost of one mag -/
def mag_cost : ℕ := 3

/-- The cost of one mig -/
def mig_cost : ℕ := 4

/-- The cost of one mog -/
def mog_cost : ℕ := 8

/-- The total amount Bill will spend -/
def total_spent : ℕ := 100

/-- Theorem stating the maximum number of mogs Bill can buy -/
theorem max_mogs_bill_can_buy :
  ∃ (mags migs mogs : ℕ),
    mags ≥ 1 ∧
    migs ≥ 1 ∧
    mogs ≥ 1 ∧
    mag_cost * mags + mig_cost * migs + mog_cost * mogs = total_spent ∧
    mogs = 10 ∧
    (∀ (m : ℕ), m > 10 →
      ¬∃ (x y : ℕ), x ≥ 1 ∧ y ≥ 1 ∧
        mag_cost * x + mig_cost * y + mog_cost * m = total_spent) :=
sorry

end NUMINAMATH_CALUDE_max_mogs_bill_can_buy_l1578_157836


namespace NUMINAMATH_CALUDE_weight_of_A_l1578_157830

/-- Given the weights of five people A, B, C, D, and E, prove that A weighs 87 kg -/
theorem weight_of_A (A B C D E : ℝ) : 
  ((A + B + C) / 3 = 60) →
  ((A + B + C + D) / 4 = 65) →
  (E = D + 3) →
  ((B + C + D + E) / 4 = 64) →
  A = 87 := by
sorry

end NUMINAMATH_CALUDE_weight_of_A_l1578_157830


namespace NUMINAMATH_CALUDE_percent_value_in_quarters_l1578_157838

theorem percent_value_in_quarters : 
  let num_dimes : ℕ := 40
  let num_quarters : ℕ := 30
  let dime_value : ℕ := 10
  let quarter_value : ℕ := 25
  let total_dimes_value : ℕ := num_dimes * dime_value
  let total_quarters_value : ℕ := num_quarters * quarter_value
  let total_value : ℕ := total_dimes_value + total_quarters_value
  (total_quarters_value : ℝ) / (total_value : ℝ) * 100 = 65.22 :=
by sorry

end NUMINAMATH_CALUDE_percent_value_in_quarters_l1578_157838


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l1578_157845

/-- Given that f(x) = x³(a⋅2^x - 2^(-x)) is an even function, prove that a = 1 --/
theorem even_function_implies_a_equals_one (a : ℝ) :
  (∀ x : ℝ, x^3 * (a * 2^x - 2^(-x)) = (-x)^3 * (a * 2^(-x) - 2^x)) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l1578_157845


namespace NUMINAMATH_CALUDE_total_spent_is_correct_l1578_157878

def batman_price : ℚ := 13.60
def superman_price : ℚ := 5.06
def batman_discount : ℚ := 0.10
def superman_discount : ℚ := 0.05
def sales_tax : ℚ := 0.08
def game1_price : ℚ := 7.25
def game2_price : ℚ := 12.50

def total_spent : ℚ :=
  let batman_discounted := batman_price * (1 - batman_discount)
  let superman_discounted := superman_price * (1 - superman_discount)
  let batman_with_tax := batman_discounted * (1 + sales_tax)
  let superman_with_tax := superman_discounted * (1 + sales_tax)
  batman_with_tax + superman_with_tax + game1_price + game2_price

theorem total_spent_is_correct : total_spent = 38.16 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_correct_l1578_157878


namespace NUMINAMATH_CALUDE_function_inequality_solution_set_l1578_157820

open Real

def isSolutionSet (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x, f x < exp x ↔ x ∈ S

theorem function_inequality_solution_set
  (f : ℝ → ℝ)
  (hf_diff : Differentiable ℝ f)
  (hf_deriv : ∀ x, deriv f x < f x)
  (hf_even : ∀ x, f (x + 2) = f (-x + 2))
  (hf_init : f 0 = exp 4) :
  isSolutionSet f (Set.Ici 4) :=
sorry

end NUMINAMATH_CALUDE_function_inequality_solution_set_l1578_157820


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l1578_157894

/-- Calculate the cost of plastering a tank's walls and bottom -/
def plasteringCost (length width depth : ℝ) (costPerSquareMeter : ℝ) : ℝ :=
  let wallArea := 2 * (length * depth + width * depth)
  let bottomArea := length * width
  let totalArea := wallArea + bottomArea
  totalArea * costPerSquareMeter

/-- Theorem: The cost of plastering a tank with given dimensions is 558 rupees -/
theorem tank_plastering_cost :
  plasteringCost 25 12 6 0.75 = 558 := by
  sorry

end NUMINAMATH_CALUDE_tank_plastering_cost_l1578_157894


namespace NUMINAMATH_CALUDE_odd_periodic_function_property_l1578_157847

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function has period T if f(x + T) = f(x) for all x -/
def HasPeriod (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem odd_periodic_function_property (f : ℝ → ℝ) 
    (h_odd : IsOdd f) 
    (h_period : HasPeriod f 5) 
    (h1 : f 1 = 1) 
    (h2 : f 2 = 2) : 
  f 3 - f 4 = -1 := by
sorry

end NUMINAMATH_CALUDE_odd_periodic_function_property_l1578_157847


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1578_157834

/-- Given an arithmetic sequence {a_n} where a_1 + a_5 + a_9 = 6, prove that a_2 + a_8 = 4 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence condition
  (a 1 + a 5 + a 9 = 6) →                           -- given condition
  (a 2 + a 8 = 4) :=                                -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1578_157834


namespace NUMINAMATH_CALUDE_distance_major_minor_endpoints_l1578_157829

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  4 * (x - 3)^2 + 16 * (y + 2)^2 = 64

-- Define the center of the ellipse
def center : ℝ × ℝ := (3, -2)

-- Define the semi-major and semi-minor axes
def a : ℝ := 4
def b : ℝ := 2

-- Define a point on the major axis
def point_on_major_axis (x y : ℝ) : Prop :=
  ellipse x y ∧ y = center.2

-- Define a point on the minor axis
def point_on_minor_axis (x y : ℝ) : Prop :=
  ellipse x y ∧ x = center.1

-- Theorem: The distance between an endpoint of the major axis
-- and an endpoint of the minor axis is 2√5
theorem distance_major_minor_endpoints :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    point_on_major_axis x₁ y₁ ∧
    point_on_minor_axis x₂ y₂ ∧
    ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 20 :=
sorry

end NUMINAMATH_CALUDE_distance_major_minor_endpoints_l1578_157829


namespace NUMINAMATH_CALUDE_female_democrats_count_l1578_157832

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) :
  total = 840 →
  female + male = total →
  (female / 2 : ℚ) + (male / 4 : ℚ) = total / 3 →
  female / 2 = 140 :=
by sorry

end NUMINAMATH_CALUDE_female_democrats_count_l1578_157832


namespace NUMINAMATH_CALUDE_hay_shortage_farmer_hay_shortage_l1578_157866

/-- Calculates the hay shortage for a farmer given specific conditions --/
theorem hay_shortage (original_harvest : ℕ) (original_acres : ℕ) (additional_acres : ℕ)
                     (num_horses : ℕ) (hay_per_horse : ℕ) (feeding_months : ℕ) : ℤ :=
  let total_acres := original_acres + additional_acres
  let total_harvest := (original_harvest / original_acres) * total_acres
  let daily_consumption := hay_per_horse * num_horses
  let monthly_consumption := daily_consumption * 30
  let total_consumption := monthly_consumption * feeding_months
  total_consumption - total_harvest

/-- Proves that the farmer will be short by 1896 bales of hay --/
theorem farmer_hay_shortage :
  hay_shortage 560 5 7 9 3 4 = 1896 := by
  sorry

end NUMINAMATH_CALUDE_hay_shortage_farmer_hay_shortage_l1578_157866


namespace NUMINAMATH_CALUDE_ratio_q_to_r_l1578_157876

/-- 
Given a total amount of 1210 divided among three persons p, q, and r,
where the ratio of p to q is 5:4 and r receives 400,
prove that the ratio of q to r is 9:10.
-/
theorem ratio_q_to_r (total : ℕ) (p q r : ℕ) (h1 : total = 1210) 
  (h2 : p + q + r = total) (h3 : 5 * q = 4 * p) (h4 : r = 400) :
  9 * r = 10 * q := by
  sorry


end NUMINAMATH_CALUDE_ratio_q_to_r_l1578_157876


namespace NUMINAMATH_CALUDE_harolds_class_size_l1578_157807

/-- Represents the number of apples Harold split among classmates -/
def total_apples : ℕ := 15

/-- Represents the number of apples each classmate received -/
def apples_per_classmate : ℕ := 5

/-- Theorem stating the number of people in Harold's class who received apples -/
theorem harolds_class_size : 
  total_apples / apples_per_classmate = 3 := by sorry

end NUMINAMATH_CALUDE_harolds_class_size_l1578_157807


namespace NUMINAMATH_CALUDE_ladder_in_alley_l1578_157810

/-- In a narrow alley, a ladder of length b is placed between two walls.
    When resting against one wall, it makes a 60° angle with the ground and reaches height s.
    When resting against the other wall, it makes a 70° angle with the ground and reaches height m.
    This theorem states that the width of the alley w is equal to m. -/
theorem ladder_in_alley (w b s m : ℝ) (h1 : 0 < w) (h2 : 0 < b) (h3 : 0 < s) (h4 : 0 < m)
  (h5 : w = b * Real.sin (60 * π / 180))
  (h6 : s = b * Real.sin (60 * π / 180))
  (h7 : w = b * Real.sin (70 * π / 180))
  (h8 : m = b * Real.sin (70 * π / 180)) :
  w = m :=
sorry

end NUMINAMATH_CALUDE_ladder_in_alley_l1578_157810


namespace NUMINAMATH_CALUDE_f_odd_and_periodic_l1578_157816

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom cond1 : ∀ x, f (10 + x) = f (10 - x)
axiom cond2 : ∀ x, f (20 - x) = -f (20 + x)

-- Define odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define periodic function
def is_periodic (f : ℝ → ℝ) : Prop := ∃ T > 0, ∀ x, f (x + T) = f x

-- Theorem statement
theorem f_odd_and_periodic : is_odd f ∧ is_periodic f := by sorry

end NUMINAMATH_CALUDE_f_odd_and_periodic_l1578_157816


namespace NUMINAMATH_CALUDE_jennifer_fish_count_l1578_157861

/-- The number of tanks Jennifer has already built -/
def built_tanks : ℕ := 3

/-- The number of fish each built tank can hold -/
def fish_per_built_tank : ℕ := 15

/-- The number of tanks Jennifer plans to build -/
def planned_tanks : ℕ := 3

/-- The number of fish each planned tank can hold -/
def fish_per_planned_tank : ℕ := 10

/-- The total number of fish Jennifer wants to house -/
def total_fish : ℕ := built_tanks * fish_per_built_tank + planned_tanks * fish_per_planned_tank

theorem jennifer_fish_count : total_fish = 75 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_fish_count_l1578_157861


namespace NUMINAMATH_CALUDE_fuel_change_calculation_l1578_157871

/-- Calculates the change received when fueling a vehicle --/
theorem fuel_change_calculation (tank_capacity : ℝ) (initial_fuel : ℝ) (fuel_cost : ℝ) (payment : ℝ) :
  tank_capacity = 150 →
  initial_fuel = 38 →
  fuel_cost = 3 →
  payment = 350 →
  payment - (tank_capacity - initial_fuel) * fuel_cost = 14 := by
  sorry

#check fuel_change_calculation

end NUMINAMATH_CALUDE_fuel_change_calculation_l1578_157871


namespace NUMINAMATH_CALUDE_tangent_line_at_one_a_range_when_f_negative_l1578_157864

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

theorem tangent_line_at_one (h : ℝ → ℝ := f 2) :
  ∃ (m b : ℝ), ∀ x y, y = m * (x - 1) + h 1 ↔ x + y + 1 = 0 :=
sorry

theorem a_range_when_f_negative (a : ℝ) :
  (∀ x > 0, f a x < 0) → a > Real.exp (-1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_a_range_when_f_negative_l1578_157864


namespace NUMINAMATH_CALUDE_lcm_ratio_sum_l1578_157863

theorem lcm_ratio_sum (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : 
  Nat.lcm a b = 54 → a * 3 = b * 2 → a + b = 45 := by
  sorry

end NUMINAMATH_CALUDE_lcm_ratio_sum_l1578_157863


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l1578_157868

/-- 
Given a quadratic expression 3x^2 + nx + 54, this theorem states that 163 is the largest 
value of n for which the expression can be factored as the product of two linear factors 
with integer coefficients.
-/
theorem largest_n_for_factorization : 
  ∀ n : ℤ, (∃ a b c d : ℤ, 3*x^2 + n*x + 54 = (a*x + b) * (c*x + d)) → n ≤ 163 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_factorization_l1578_157868


namespace NUMINAMATH_CALUDE_total_seeds_after_trading_is_2340_l1578_157858

/-- Represents the number of watermelon seeds each person has -/
structure SeedCount where
  bom : ℕ
  gwi : ℕ
  yeon : ℕ
  eun : ℕ

/-- Calculates the total number of seeds after trading -/
def totalSeedsAfterTrading (initial : SeedCount) : ℕ :=
  let bomAfter := initial.bom - 50
  let gwiAfter := initial.gwi + (initial.yeon * 20 / 100)
  let yeonAfter := initial.yeon - (initial.yeon * 20 / 100)
  let eunAfter := initial.eun + 50
  bomAfter + gwiAfter + yeonAfter + eunAfter

/-- Theorem stating that the total number of seeds after trading is 2340 -/
theorem total_seeds_after_trading_is_2340 (initial : SeedCount) 
  (h1 : initial.yeon = 3 * initial.gwi)
  (h2 : initial.gwi = initial.bom + 40)
  (h3 : initial.eun = 2 * initial.gwi)
  (h4 : initial.bom = 300) :
  totalSeedsAfterTrading initial = 2340 := by
  sorry

#eval totalSeedsAfterTrading { bom := 300, gwi := 340, yeon := 1020, eun := 680 }

end NUMINAMATH_CALUDE_total_seeds_after_trading_is_2340_l1578_157858


namespace NUMINAMATH_CALUDE_standard_pairs_parity_l1578_157875

/-- Represents a color on the chessboard -/
inductive Color
| Red
| Blue

/-- Represents a chessboard -/
structure Chessboard (m n : ℕ) where
  colors : Fin m → Fin n → Color
  m_ge_three : m ≥ 3
  n_ge_three : n ≥ 3

/-- Counts the number of blue squares on the boundary (excluding corners) -/
def countBlueBoundary (board : Chessboard m n) : ℕ :=
  sorry

/-- Counts the number of standard pairs on the chessboard -/
def countStandardPairs (board : Chessboard m n) : ℕ :=
  sorry

/-- Theorem stating that the parity of standard pairs is determined by the parity of blue boundary squares -/
theorem standard_pairs_parity (m n : ℕ) (board : Chessboard m n) :
  Even (countStandardPairs board) ↔ Even (countBlueBoundary board) :=
sorry

end NUMINAMATH_CALUDE_standard_pairs_parity_l1578_157875


namespace NUMINAMATH_CALUDE_ellipse_problem_l1578_157892

def given_ellipse (x y : ℝ) : Prop :=
  8 * x^2 / 81 + y^2 / 36 = 1

def reference_ellipse (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 / 4 = 1

def required_ellipse (x y : ℝ) : Prop :=
  x^2 / 15 + y^2 / 10 = 1

theorem ellipse_problem (x₀ : ℝ) (h1 : given_ellipse x₀ 2) (h2 : x₀ < 0) :
  x₀ = -3 ∧
  ∀ (x y : ℝ), (x = x₀ ∧ y = 2 → required_ellipse x y) ∧
  (∃ (c : ℝ), ∀ (x y : ℝ), reference_ellipse x y ↔ x^2 + y^2 = 9 + 4 - c^2 ∧ c^2 = 5) ∧
  (∃ (c : ℝ), ∀ (x y : ℝ), required_ellipse x y ↔ x^2 + y^2 = 15 + 10 - c^2 ∧ c^2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_problem_l1578_157892


namespace NUMINAMATH_CALUDE_divisible_by_seven_count_l1578_157893

theorem divisible_by_seven_count : 
  (∃! (s : Finset Nat), 
    (∀ k ∈ s, k < 100 ∧ k > 0) ∧ 
    (∀ k ∈ s, ∀ n : Nat, n > 0 → (2 * (3^(6*n)) + k * (2^(3*n+1)) - 1) % 7 = 0) ∧
    s.card = 14) := by sorry

end NUMINAMATH_CALUDE_divisible_by_seven_count_l1578_157893


namespace NUMINAMATH_CALUDE_min_marked_points_l1578_157895

/-- Represents a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A configuration of points in a plane -/
structure PointConfiguration where
  points : Finset Point
  unique_distances : ∀ p q r s : Point, p ∈ points → q ∈ points → r ∈ points → s ∈ points →
    p ≠ q → r ≠ s → (p.x - q.x)^2 + (p.y - q.y)^2 ≠ (r.x - s.x)^2 + (r.y - s.y)^2

/-- The set of points marked as closest to at least one other point -/
def marked_points (config : PointConfiguration) : Finset Point :=
  sorry

/-- The theorem stating the minimum number of marked points -/
theorem min_marked_points (config : PointConfiguration) :
  config.points.card = 2018 →
  (marked_points config).card ≥ 404 :=
sorry

end NUMINAMATH_CALUDE_min_marked_points_l1578_157895


namespace NUMINAMATH_CALUDE_pants_price_is_6_l1578_157844

-- Define variables for pants and shirt prices
variable (pants_price : ℝ)
variable (shirt_price : ℝ)

-- Define Peter's purchase
def peter_total : ℝ := 2 * pants_price + 5 * shirt_price

-- Define Jessica's purchase
def jessica_total : ℝ := 2 * shirt_price

-- Theorem stating the price of one pair of pants
theorem pants_price_is_6 
  (h1 : peter_total = 62)
  (h2 : jessica_total = 20) :
  pants_price = 6 := by
sorry

end NUMINAMATH_CALUDE_pants_price_is_6_l1578_157844


namespace NUMINAMATH_CALUDE_characterization_of_expressible_numbers_l1578_157837

theorem characterization_of_expressible_numbers (n : ℕ) :
  (∃ k : ℕ, n = k + 2 * Int.floor (Real.sqrt k) + 2) ↔
  (∀ y : ℕ, n ≠ y^2 ∧ n ≠ y^2 - 1) :=
sorry

end NUMINAMATH_CALUDE_characterization_of_expressible_numbers_l1578_157837


namespace NUMINAMATH_CALUDE_tourist_survival_l1578_157887

theorem tourist_survival (initial_tourists : ℕ) (eaten : ℕ) (poison_fraction : ℚ) (recovery_fraction : ℚ) : initial_tourists = 30 → eaten = 2 → poison_fraction = 1/2 → recovery_fraction = 1/7 → 
  let remaining_after_eaten := initial_tourists - eaten
  let poisoned := (remaining_after_eaten : ℚ) * poison_fraction
  let recovered := poisoned * recovery_fraction
  (remaining_after_eaten : ℚ) - poisoned + recovered = 16 := by
  sorry

end NUMINAMATH_CALUDE_tourist_survival_l1578_157887


namespace NUMINAMATH_CALUDE_system_solution_sum_reciprocals_l1578_157881

theorem system_solution_sum_reciprocals (x₀ y₀ : ℚ) :
  x₀ / 3 + y₀ / 5 = 1 ∧ x₀ / 5 + y₀ / 3 = 1 →
  1 / x₀ + 1 / y₀ = 16 / 15 := by
sorry

end NUMINAMATH_CALUDE_system_solution_sum_reciprocals_l1578_157881


namespace NUMINAMATH_CALUDE_b_not_unique_l1578_157822

-- Define the line equation
def line_equation (y : ℝ) : ℝ := 8 * y + 5

-- Define the points on the line
def point1 (m B : ℝ) : ℝ × ℝ := (m, B)
def point2 (m B : ℝ) : ℝ × ℝ := (m + 2, B + 0.25)

-- Theorem stating that B cannot be uniquely determined
theorem b_not_unique (m B : ℝ) : 
  line_equation B = (point1 m B).1 ∧ 
  line_equation (B + 0.25) = (point2 m B).1 → 
  ∃ (B' : ℝ), B' ≠ B ∧ 
    line_equation B' = (point1 m B').1 ∧ 
    line_equation (B' + 0.25) = (point2 m B').1 :=
by
  sorry


end NUMINAMATH_CALUDE_b_not_unique_l1578_157822


namespace NUMINAMATH_CALUDE_ballon_arrangements_l1578_157879

theorem ballon_arrangements :
  let total_letters : Nat := 6
  let repeated_letters : Nat := 2
  Nat.factorial total_letters / Nat.factorial repeated_letters = 360 := by
  sorry

end NUMINAMATH_CALUDE_ballon_arrangements_l1578_157879


namespace NUMINAMATH_CALUDE_entire_group_is_population_l1578_157821

/-- Represents a group of students who took a test -/
structure StudentGroup where
  size : ℕ
  scores : Finset ℝ
  h_size : scores.card = size

/-- Represents a sample extracted from a larger group -/
structure Sample (group : StudentGroup) where
  size : ℕ
  scores : Finset ℝ
  h_size : scores.card = size
  h_subset : scores ⊆ group.scores

/-- Definition of a population in statistical terms -/
def isPopulation (group : StudentGroup) : Prop :=
  ∀ (sample : Sample group), sample.scores ⊆ group.scores

/-- The theorem to be proved -/
theorem entire_group_is_population 
  (entireGroup : StudentGroup) 
  (sample : Sample entireGroup) 
  (h_entire_size : entireGroup.size = 5000) 
  (h_sample_size : sample.size = 200) : 
  isPopulation entireGroup := by
  sorry

end NUMINAMATH_CALUDE_entire_group_is_population_l1578_157821


namespace NUMINAMATH_CALUDE_book_pages_problem_l1578_157852

theorem book_pages_problem :
  ∃ (n k : ℕ), 
    n > 0 ∧ 
    k > 0 ∧ 
    k < n ∧ 
    n * (n + 1) / 2 - (2 * k + 1) = 4979 :=
sorry

end NUMINAMATH_CALUDE_book_pages_problem_l1578_157852


namespace NUMINAMATH_CALUDE_new_ratio_after_transaction_l1578_157802

/-- Represents the number of animals on the farm -/
structure FarmAnimals where
  horses : ℕ
  cows : ℕ

/-- Represents the transaction of selling horses and buying cows -/
def performTransaction (farm : FarmAnimals) : FarmAnimals :=
  { horses := farm.horses - 15, cows := farm.cows + 15 }

/-- Theorem stating the new ratio of horses to cows after the transaction -/
theorem new_ratio_after_transaction (initial : FarmAnimals)
    (h1 : initial.horses = 4 * initial.cows)
    (h2 : (performTransaction initial).horses = (performTransaction initial).cows + 60) :
    (performTransaction initial).horses / (performTransaction initial).cows = 7 / 3 := by
  sorry


end NUMINAMATH_CALUDE_new_ratio_after_transaction_l1578_157802


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1578_157809

def A : Set ℚ := {x | ∃ k : ℕ, x = 3 * k + 1}
def B : Set ℚ := {x | x ≤ 7}

theorem intersection_of_A_and_B : A ∩ B = {1, 4, 7} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1578_157809


namespace NUMINAMATH_CALUDE_arithmetic_sequence_divisibility_l1578_157899

theorem arithmetic_sequence_divisibility (a : ℕ) :
  ∃! k : Fin 7, ∃ n : ℕ, n = a + k * 30 ∧ n % 7 = 0 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_divisibility_l1578_157899


namespace NUMINAMATH_CALUDE_intersection_of_P_and_M_l1578_157839

def P : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 3}
def M : Set ℝ := {x : ℝ | |x| ≤ 3}

theorem intersection_of_P_and_M : P ∩ M = {x : ℝ | 0 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_M_l1578_157839


namespace NUMINAMATH_CALUDE_perimeter_area_sum_l1578_157814

/-- A parallelogram with vertices at (2,3), (2,8), (9,8), and (9,3) -/
structure Parallelogram where
  v1 : ℝ × ℝ := (2, 3)
  v2 : ℝ × ℝ := (2, 8)
  v3 : ℝ × ℝ := (9, 8)
  v4 : ℝ × ℝ := (9, 3)

/-- Calculate the perimeter of the parallelogram -/
def perimeter (p : Parallelogram) : ℝ :=
  2 * (abs (p.v3.1 - p.v1.1) + abs (p.v2.2 - p.v1.2))

/-- Calculate the area of the parallelogram -/
def area (p : Parallelogram) : ℝ :=
  abs (p.v3.1 - p.v1.1) * abs (p.v2.2 - p.v1.2)

/-- The sum of the perimeter and area of the parallelogram is 59 -/
theorem perimeter_area_sum (p : Parallelogram) : perimeter p + area p = 59 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_area_sum_l1578_157814


namespace NUMINAMATH_CALUDE_gas_cost_calculation_l1578_157801

/-- Calculates the total cost of filling up a car's gas tank multiple times with different gas prices -/
theorem gas_cost_calculation (tank_capacity : ℝ) (prices : List ℝ) :
  tank_capacity = 12 ∧ 
  prices = [3, 3.5, 4, 4.5] →
  (prices.map (· * tank_capacity)).sum = 180 := by
  sorry

#check gas_cost_calculation

end NUMINAMATH_CALUDE_gas_cost_calculation_l1578_157801


namespace NUMINAMATH_CALUDE_robot_max_score_l1578_157808

def initial_iq : ℕ := 25

def point_range : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 10}

def solve_problem (iq : ℕ) (points : ℕ) : Option ℕ :=
  if iq ≥ points then some (iq - points + 1) else none

def max_score (problems : List ℕ) : ℕ :=
  problems.foldl (λ acc p => acc + p) 0

theorem robot_max_score :
  ∃ (problems : List ℕ),
    (∀ p ∈ problems, p ∈ point_range) ∧
    (problems.foldl (λ iq p => (solve_problem iq p).getD 0) initial_iq ≠ 0) ∧
    (max_score problems = 31) ∧
    (∀ (other_problems : List ℕ),
      (∀ p ∈ other_problems, p ∈ point_range) →
      (other_problems.foldl (λ iq p => (solve_problem iq p).getD 0) initial_iq ≠ 0) →
      max_score other_problems ≤ 31) :=
by sorry

end NUMINAMATH_CALUDE_robot_max_score_l1578_157808


namespace NUMINAMATH_CALUDE_complex_number_properties_l1578_157884

def i : ℂ := Complex.I

theorem complex_number_properties (z : ℂ) (h : z * (2 - i) = i ^ 2020) :
  (Complex.im z = 1/5) ∧ (Complex.re z > 0 ∧ Complex.im z > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l1578_157884


namespace NUMINAMATH_CALUDE_real_part_of_w_cubed_l1578_157865

theorem real_part_of_w_cubed (w : ℂ) 
  (h1 : w.im > 0)
  (h2 : Complex.abs w = 5)
  (h3 : (w^2 - w) • (w^3 - w) = 0) :
  (w^3).re = -73 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_w_cubed_l1578_157865


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l1578_157880

/-- A 2-digit positive integer is represented by its tens and ones digits -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  is_valid : 1 ≤ tens ∧ tens ≤ 9 ∧ ones ≤ 9

/-- The value of a 2-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

theorem unique_two_digit_number :
  ∃! (c : TwoDigitNumber), 
    c.tens + c.ones = 10 ∧ 
    c.tens * c.ones = 25 ∧ 
    c.value = 55 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l1578_157880


namespace NUMINAMATH_CALUDE_king_crown_payment_l1578_157882

/-- Calculates the total amount paid for a crown, including tip -/
def totalAmountPaid (crownCost : ℝ) (tipRate : ℝ) : ℝ :=
  crownCost + (crownCost * tipRate)

/-- Theorem: The king pays $22,000 for a $20,000 crown with a 10% tip -/
theorem king_crown_payment :
  totalAmountPaid 20000 0.1 = 22000 := by
  sorry

end NUMINAMATH_CALUDE_king_crown_payment_l1578_157882


namespace NUMINAMATH_CALUDE_no_solution_implies_b_bounded_l1578_157817

theorem no_solution_implies_b_bounded (a b : ℝ) :
  (∀ x : ℝ, a * Real.cos x + b * Real.cos (3 * x) ≤ 1) →
  abs b ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_b_bounded_l1578_157817


namespace NUMINAMATH_CALUDE_distribute_men_and_women_l1578_157873

/- Define the number of men and women -/
def num_men : ℕ := 4
def num_women : ℕ := 5

/- Define the size of each group -/
def group_size : ℕ := 3

/- Define a function to calculate the number of ways to distribute people -/
def distribute_people (m : ℕ) (w : ℕ) : ℕ :=
  let ways_group1 := (m.choose 1) * (w.choose 2)
  let ways_group2 := ((m - 1).choose 1) * ((w - 2).choose 2)
  ways_group1 * ways_group2 / 2

/- Theorem statement -/
theorem distribute_men_and_women :
  distribute_people num_men num_women = 180 :=
by sorry

end NUMINAMATH_CALUDE_distribute_men_and_women_l1578_157873


namespace NUMINAMATH_CALUDE_percentage_of_part_to_whole_l1578_157886

theorem percentage_of_part_to_whole (total : ℝ) (part : ℝ) : 
  total > 0 → part ≥ 0 → part ≤ total → (part / total) * 100 = 25 → total = 400 ∧ part = 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_part_to_whole_l1578_157886


namespace NUMINAMATH_CALUDE_remainder_5_divisors_2002_l1578_157812

def divides_with_remainder_5 (d : ℕ) : Prop :=
  ∃ q : ℕ, 2007 = d * q + 5

def divisors_of_2002 : Set ℕ :=
  {d : ℕ | d > 0 ∧ 2002 % d = 0}

theorem remainder_5_divisors_2002 :
  {d : ℕ | divides_with_remainder_5 d} = {d ∈ divisors_of_2002 | d > 5} :=
by sorry

end NUMINAMATH_CALUDE_remainder_5_divisors_2002_l1578_157812


namespace NUMINAMATH_CALUDE_consecutive_odd_squares_difference_l1578_157833

theorem consecutive_odd_squares_difference (n : ℕ) : 
  ∃ k : ℤ, (2*n + 1)^2 - (2*n - 1)^2 = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_squares_difference_l1578_157833


namespace NUMINAMATH_CALUDE_alice_bushes_l1578_157819

/-- The number of bushes needed to cover three sides of a yard --/
def bushes_needed (side_length : ℕ) (sides : ℕ) (bush_width : ℕ) : ℕ :=
  (side_length * sides) / bush_width

/-- Theorem: Alice needs 24 bushes for her yard --/
theorem alice_bushes :
  bushes_needed 24 3 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_alice_bushes_l1578_157819


namespace NUMINAMATH_CALUDE_books_read_in_common_l1578_157818

theorem books_read_in_common (tony_books dean_books breanna_books total_books : ℕ) 
  (h1 : tony_books = 23)
  (h2 : dean_books = 12)
  (h3 : breanna_books = 17)
  (h4 : total_books = 47)
  (h5 : ∃ (common : ℕ), common > 0 ∧ common ≤ min tony_books dean_books)
  (h6 : ∃ (all_common : ℕ), all_common > 0 ∧ all_common ≤ min tony_books (min dean_books breanna_books)) :
  ∃ (x : ℕ), x = 3 ∧ 
    tony_books + dean_books + breanna_books - x - 1 = total_books :=
by sorry

end NUMINAMATH_CALUDE_books_read_in_common_l1578_157818
