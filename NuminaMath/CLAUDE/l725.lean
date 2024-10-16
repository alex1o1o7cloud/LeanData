import Mathlib

namespace NUMINAMATH_CALUDE_monet_paintings_consecutive_probability_l725_72521

/-- The probability of consecutive Monet paintings in a random arrangement -/
theorem monet_paintings_consecutive_probability 
  (total_pieces : ℕ) 
  (monet_paintings : ℕ) 
  (h1 : total_pieces = 12) 
  (h2 : monet_paintings = 4) :
  (monet_paintings.factorial * (total_pieces - monet_paintings + 1)) / total_pieces.factorial = 18 / 95 := by
  sorry

end NUMINAMATH_CALUDE_monet_paintings_consecutive_probability_l725_72521


namespace NUMINAMATH_CALUDE_count_squares_below_line_l725_72530

/-- The number of 1x1 squares in the first quadrant entirely below the line 6x + 143y = 858 -/
def squares_below_line : ℕ :=
  355

/-- The equation of the line -/
def line_equation (x y : ℚ) : Prop :=
  6 * x + 143 * y = 858

theorem count_squares_below_line :
  squares_below_line = 355 :=
sorry

end NUMINAMATH_CALUDE_count_squares_below_line_l725_72530


namespace NUMINAMATH_CALUDE_min_value_sqrt_a_plus_four_over_sqrt_a_plus_one_l725_72519

theorem min_value_sqrt_a_plus_four_over_sqrt_a_plus_one (a : ℝ) (ha : a > 0) :
  Real.sqrt a + 4 / (Real.sqrt a + 1) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_a_plus_four_over_sqrt_a_plus_one_l725_72519


namespace NUMINAMATH_CALUDE_cube_edge_length_l725_72508

/-- Given a cube with volume 3375 cm³, prove that the total length of its edges is 180 cm. -/
theorem cube_edge_length (V : ℝ) (h : V = 3375) : 
  12 * (V ^ (1/3 : ℝ)) = 180 :=
sorry

end NUMINAMATH_CALUDE_cube_edge_length_l725_72508


namespace NUMINAMATH_CALUDE_move_point_l725_72574

/-- Moving a point left decreases its x-coordinate --/
def move_left (x : ℝ) (units : ℝ) : ℝ := x - units

/-- Moving a point up increases its y-coordinate --/
def move_up (y : ℝ) (units : ℝ) : ℝ := y + units

/-- A 2D point --/
structure Point where
  x : ℝ
  y : ℝ

/-- The initial point P --/
def P : Point := ⟨-2, -3⟩

/-- Theorem: Moving P 1 unit left and 3 units up results in (-3, 0) --/
theorem move_point :
  let new_x := move_left P.x 1
  let new_y := move_up P.y 3
  (new_x, new_y) = (-3, 0) := by sorry

end NUMINAMATH_CALUDE_move_point_l725_72574


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l725_72584

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x ≥ 0 → x^2 + x - 1 > 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^2 + x - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l725_72584


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_l725_72571

theorem unique_number_with_three_prime_divisors (x : ℕ) (n : ℕ) :
  Odd n →
  x = 6^n + 1 →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧
    x = 11 * p * q ∧
    ∀ r : ℕ, Prime r → r ∣ x → (r = 11 ∨ r = p ∨ r = q)) →
  x = 7777 := by
sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_l725_72571


namespace NUMINAMATH_CALUDE_partner_contribution_b_contribution_is_31500_l725_72573

/-- Given a business partnership scenario, calculate partner B's capital contribution. -/
theorem partner_contribution (a_initial : ℕ) (a_months : ℕ) (b_months : ℕ) (profit_ratio_a : ℕ) (profit_ratio_b : ℕ) : ℕ :=
  let total_months := a_months
  let b_contribution := (a_initial * total_months * profit_ratio_b) / (b_months * profit_ratio_a)
  b_contribution

/-- Prove that B's contribution is 31500 rupees given the specific scenario. -/
theorem b_contribution_is_31500 :
  partner_contribution 3500 12 2 2 3 = 31500 := by
  sorry

end NUMINAMATH_CALUDE_partner_contribution_b_contribution_is_31500_l725_72573


namespace NUMINAMATH_CALUDE_square_64_implies_product_63_l725_72539

theorem square_64_implies_product_63 (m : ℝ) : (m + 2)^2 = 64 → (m + 1) * (m + 3) = 63 := by
  sorry

end NUMINAMATH_CALUDE_square_64_implies_product_63_l725_72539


namespace NUMINAMATH_CALUDE_total_amount_proof_l725_72585

theorem total_amount_proof (a b c : ℕ) : 
  a = 3 * b → 
  b = c + 25 → 
  b = 134 → 
  a + b + c = 645 := by
sorry

end NUMINAMATH_CALUDE_total_amount_proof_l725_72585


namespace NUMINAMATH_CALUDE_problem_solution_l725_72596

def sequence1 (n : ℕ) : ℤ := (-1)^n * (2*n - 1)
def sequence2 (n : ℕ) : ℤ := (-1)^n * (2*n - 1) - 2
def sequence3 (n : ℕ) : ℤ := 3 * (2*n - 1) * (-1)^(n+1)

theorem problem_solution :
  (sequence1 10 = 19) ∧
  (sequence2 15 = -31) ∧
  (∀ n : ℕ, sequence2 n + sequence2 (n+1) + sequence2 (n+2) ≠ 1001) ∧
  (∃! k : ℕ, k % 2 = 1 ∧ sequence1 k + sequence2 k + sequence3 k = 599 ∧ k = 301) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l725_72596


namespace NUMINAMATH_CALUDE_vkontakte_problem_l725_72561

-- Define predicates for each person being on VKontakte
variable (M I A P : Prop)

-- State the theorem
theorem vkontakte_problem :
  (M → (I ∧ A)) →  -- If M is on VKontakte, then both I and A are on VKontakte
  (A ↔ ¬P) →       -- Only one of A or P is on VKontakte
  (I ∨ M) →        -- At least one of I or M is on VKontakte
  (P ↔ I) →        -- P and I are either both on or both not on VKontakte
  (I ∧ P ∧ ¬M ∧ ¬A) -- Conclusion: I and P are on VKontakte, M and A are not
  := by sorry

end NUMINAMATH_CALUDE_vkontakte_problem_l725_72561


namespace NUMINAMATH_CALUDE_original_equals_scientific_l725_72527

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be converted -/
def original_number : ℕ := 28000

/-- The scientific notation representation of the original number -/
def scientific_representation : ScientificNotation :=
  { coefficient := 2.8
    exponent := 4
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent :=
by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l725_72527


namespace NUMINAMATH_CALUDE_polynomial_factorization_l725_72522

theorem polynomial_factorization (x : ℝ) :
  (x^4 - 4*x^2 + 1) * (x^4 + 3*x^2 + 1) + 10*x^4 = 
  (x + 1)^2 * (x - 1)^2 * (x^2 + x + 1) * (x^2 - x + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l725_72522


namespace NUMINAMATH_CALUDE_part1_solution_part2_solution_l725_72525

-- Define A_n (falling factorial)
def A (n : ℕ) : ℕ := n * (n - 1) * (n - 2) * (n - 3)

-- Define C_n (binomial coefficient)
def C (n : ℕ) : ℕ := n * (n - 1) * (n - 2) * (n - 3) * (n - 4) * (n - 5) / 720

-- Part 1: Prove that the only positive integer solution to A_{2n+1}^4 = 140A_n^3 is n = 3
theorem part1_solution : {n : ℕ | n > 0 ∧ A (2*n + 1)^4 = 140 * A n^3} = {3} := by sorry

-- Part 2: Prove that the positive integer solutions to A_N^4 ≥ 24C_n^6 where n ≥ 6 are n = 6, 7, 8, 9, 10
theorem part2_solution : {n : ℕ | n ≥ 6 ∧ A n^4 ≥ 24 * C n^6} = {6, 7, 8, 9, 10} := by sorry

end NUMINAMATH_CALUDE_part1_solution_part2_solution_l725_72525


namespace NUMINAMATH_CALUDE_compare_cubic_and_quadratic_diff_l725_72517

theorem compare_cubic_and_quadratic_diff (a b : ℝ) :
  (a ≥ b → a^3 - b^3 ≥ a*b^2 - a^2*b) ∧
  (a < b → a^3 - b^3 ≤ a*b^2 - a^2*b) :=
by sorry

end NUMINAMATH_CALUDE_compare_cubic_and_quadratic_diff_l725_72517


namespace NUMINAMATH_CALUDE_arrangements_theorem_l725_72506

def number_of_arrangements (n : ℕ) (red yellow blue : ℕ) : ℕ :=
  sorry

theorem arrangements_theorem :
  number_of_arrangements 5 2 2 1 = 48 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_theorem_l725_72506


namespace NUMINAMATH_CALUDE_linear_function_uniqueness_l725_72562

/-- A linear function f : ℝ → ℝ is increasing if for all x, y ∈ ℝ, x < y implies f x < f y -/
def IsIncreasingLinear (f : ℝ → ℝ) : Prop :=
  (∃ a b : ℝ, ∀ x, f x = a * x + b) ∧ (∀ x y, x < y → f x < f y)

/-- The main theorem -/
theorem linear_function_uniqueness (f : ℝ → ℝ) 
  (h_increasing : IsIncreasingLinear f)
  (h_composition : ∀ x, f (f x) = 4 * x + 3) :
  ∀ x, f x = 2 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_uniqueness_l725_72562


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l725_72555

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 + a 3 + a 5 = 7) →
  (a 5 + a 7 + a 9 = 28) →
  (a 9 + a 11 + a 13 = 112) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l725_72555


namespace NUMINAMATH_CALUDE_complex_root_product_simplification_l725_72500

theorem complex_root_product_simplification :
  2 * Real.sqrt 3 * 6 * (12 ^ (1/6 : ℝ)) * 3 * ((3/2) ^ (1/3 : ℝ)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_product_simplification_l725_72500


namespace NUMINAMATH_CALUDE_xyz_value_l725_72537

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) + x * y * z = 15) :
  x * y * z = 9 / 2 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l725_72537


namespace NUMINAMATH_CALUDE_range_of_a_l725_72559

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo 2 3, x^2 + 5 > a*x) = False → a ∈ Set.Ici (2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l725_72559


namespace NUMINAMATH_CALUDE_handshake_arrangements_mod_1000_l725_72588

/-- Represents the number of ways N people can shake hands with exactly two others each -/
def handshake_arrangements (N : ℕ) : ℕ :=
  sorry

/-- The number of ways 9 people can shake hands with exactly two others each -/
def N : ℕ := handshake_arrangements 9

/-- Theorem stating that the number of handshake arrangements for 9 people is congruent to 152 modulo 1000 -/
theorem handshake_arrangements_mod_1000 : N ≡ 152 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_handshake_arrangements_mod_1000_l725_72588


namespace NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l725_72532

/-- An arithmetic sequence with common difference d and first term 9d -/
def arithmetic_sequence (d : ℝ) (n : ℕ) : ℝ := 9 * d + (n - 1) * d

theorem arithmetic_sequence_geometric_mean (d : ℝ) (k : ℕ) 
  (h_d : d ≠ 0) :
  (arithmetic_sequence d k) ^ 2 = 
    (arithmetic_sequence d 1) * (arithmetic_sequence d (2 * k)) → 
  k = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l725_72532


namespace NUMINAMATH_CALUDE_tuesday_earnings_l725_72551

/-- Lauren's earnings from social media on Tuesday -/
def laurens_earnings (commercial_rate : ℚ) (subscription_rate : ℚ) 
  (commercial_views : ℕ) (subscriptions : ℕ) : ℚ :=
  commercial_rate * commercial_views + subscription_rate * subscriptions

/-- Theorem: Lauren's earnings on Tuesday -/
theorem tuesday_earnings : 
  laurens_earnings (1/2) 1 100 27 = 77 := by sorry

end NUMINAMATH_CALUDE_tuesday_earnings_l725_72551


namespace NUMINAMATH_CALUDE_polynomial_roots_sum_l725_72544

theorem polynomial_roots_sum (n : ℤ) (p q r : ℤ) : 
  (∀ x : ℝ, x^3 - 2009*x + n = 0 ↔ x = p ∨ x = q ∨ x = r) →
  abs p + abs q + abs r = 102 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_sum_l725_72544


namespace NUMINAMATH_CALUDE_kathys_candy_collection_l725_72512

theorem kathys_candy_collection (num_groups : ℕ) (candies_per_group : ℕ) 
  (h1 : num_groups = 10) (h2 : candies_per_group = 3) : 
  num_groups * candies_per_group = 30 := by
  sorry

end NUMINAMATH_CALUDE_kathys_candy_collection_l725_72512


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l725_72547

theorem triangle_max_perimeter :
  ∀ x : ℕ,
  x > 0 →
  x + 4*x > 20 →
  x + 20 > 4*x →
  4*x + 20 > x →
  (∀ y : ℕ, y > 0 → y + 4*y > 20 → y + 20 > 4*y → 4*y + 20 > y → x + 4*x + 20 ≥ y + 4*y + 20) →
  x + 4*x + 20 = 50 :=
by
  sorry

#check triangle_max_perimeter

end NUMINAMATH_CALUDE_triangle_max_perimeter_l725_72547


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l725_72597

def A : Set Int := {-2, -1, 0, 1, 2}
def B : Set Int := {1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l725_72597


namespace NUMINAMATH_CALUDE_largest_even_five_digit_number_with_square_and_cube_l725_72514

/-- A function that checks if a number is a perfect square --/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that checks if a number is a perfect cube --/
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m * m = n

/-- A function that returns the first three digits of a 5-digit number --/
def first_three_digits (n : ℕ) : ℕ :=
  n / 100

/-- A function that returns the last three digits of a 5-digit number --/
def last_three_digits (n : ℕ) : ℕ :=
  n % 1000

/-- Main theorem --/
theorem largest_even_five_digit_number_with_square_and_cube : 
  ∀ n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧  -- 5-digit number
    Even n ∧  -- even number
    is_perfect_square (first_three_digits n) ∧  -- first three digits form a perfect square
    is_perfect_cube (last_three_digits n)  -- last three digits form a perfect cube
    → n ≤ 62512 :=
by sorry

end NUMINAMATH_CALUDE_largest_even_five_digit_number_with_square_and_cube_l725_72514


namespace NUMINAMATH_CALUDE_min_additional_squares_for_axisymmetry_l725_72586

/-- Represents a rectangle with shaded squares -/
structure ShadedRectangle where
  width : ℕ
  height : ℕ
  shadedSquares : Finset (ℕ × ℕ)

/-- Checks if a ShadedRectangle is axisymmetric with two lines of symmetry -/
def isAxisymmetric (rect : ShadedRectangle) : Prop :=
  ∀ (x y : ℕ), x < rect.width ∧ y < rect.height →
    ((x, y) ∈ rect.shadedSquares ↔ (rect.width - 1 - x, y) ∈ rect.shadedSquares) ∧
    ((x, y) ∈ rect.shadedSquares ↔ (x, rect.height - 1 - y) ∈ rect.shadedSquares)

/-- The theorem to be proved -/
theorem min_additional_squares_for_axisymmetry 
  (rect : ShadedRectangle) 
  (h : rect.shadedSquares.card = 3) : 
  ∃ (additionalSquares : Finset (ℕ × ℕ)),
    additionalSquares.card = 6 ∧
    isAxisymmetric ⟨rect.width, rect.height, rect.shadedSquares ∪ additionalSquares⟩ ∧
    ∀ (smallerSet : Finset (ℕ × ℕ)), 
      smallerSet.card < 6 → 
      ¬isAxisymmetric ⟨rect.width, rect.height, rect.shadedSquares ∪ smallerSet⟩ :=
sorry

end NUMINAMATH_CALUDE_min_additional_squares_for_axisymmetry_l725_72586


namespace NUMINAMATH_CALUDE_complex_fraction_power_l725_72526

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_power (h : i * i = -1) : ((1 + i) / (1 - i)) ^ 2013 = i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_power_l725_72526


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l725_72505

theorem polynomial_identity_sum (a₀ a₁ a₂ a₃ : ℝ) 
  (h : ∀ x : ℝ, 1 + x + x^2 + x^3 = a₀ + a₁*(1-x) + a₂*(1-x)^2 + a₃*(1-x)^3) : 
  a₁ + a₂ + a₃ = -3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l725_72505


namespace NUMINAMATH_CALUDE_point_Q_y_coordinate_product_l725_72509

theorem point_Q_y_coordinate_product : ∀ (y₁ y₂ : ℝ),
  (∃ (Q : ℝ × ℝ), 
    Q.1 = 4 ∧ 
    ((Q.1 - 1)^2 + (Q.2 - (-3))^2) = 10^2 ∧
    (Q.2 = y₁ ∨ Q.2 = y₂) ∧
    y₁ ≠ y₂) →
  y₁ * y₂ = -82 := by
sorry

end NUMINAMATH_CALUDE_point_Q_y_coordinate_product_l725_72509


namespace NUMINAMATH_CALUDE_new_quadratic_equation_l725_72520

theorem new_quadratic_equation (a b c : ℝ) (h : b^2 - 4*a*c > 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let new_eq := fun y => a * y^2 + b * y + (c - a + Real.sqrt (b^2 - 4*a*c))
  (new_eq (x₁ - 1) = 0) ∧ (new_eq (x₂ + 1) = 0) := by
sorry

end NUMINAMATH_CALUDE_new_quadratic_equation_l725_72520


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l725_72567

theorem greatest_integer_difference (x y : ℤ) 
  (hx : 7 < x ∧ x < 9)
  (hy : 9 < y ∧ y < 15) :
  ∃ (d : ℤ), d = y - x ∧ d ≤ 6 ∧ ∀ (d' : ℤ), d' = y - x → d' ≤ d :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l725_72567


namespace NUMINAMATH_CALUDE_pyramid_circumscribed_equivalence_l725_72510

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a pyramid with n vertices -/
structure Pyramid (n : ℕ) where
  apex : Point3D
  base : Fin n → Point3D

/-- Predicate for the existence of a circumscribed sphere around a pyramid -/
def has_circumscribed_sphere (p : Pyramid n) : Prop := sorry

/-- Predicate for the existence of a circumscribed circle around the base of a pyramid -/
def has_circumscribed_circle_base (p : Pyramid n) : Prop := sorry

/-- Theorem stating the equivalence of circumscribed sphere and circle for a pyramid -/
theorem pyramid_circumscribed_equivalence (n : ℕ) (p : Pyramid n) :
  has_circumscribed_sphere p ↔ has_circumscribed_circle_base p := by sorry

end NUMINAMATH_CALUDE_pyramid_circumscribed_equivalence_l725_72510


namespace NUMINAMATH_CALUDE_box_volume_increase_l725_72557

theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 4320)
  (surface_area : 2 * (l * w + w * h + l * h) = 1704)
  (edge_sum : 4 * (l + w + h) = 208) :
  (l + 4) * (w + 4) * (h + 4) = 8624 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l725_72557


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l725_72548

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → 1 / a < 1 / b) ∧
  (∃ a b : ℝ, 1 / a < 1 / b ∧ ¬(a > b ∧ b > 0)) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l725_72548


namespace NUMINAMATH_CALUDE_pie_chart_best_for_part_whole_l725_72580

-- Define the types of statistical graphs
inductive StatisticalGraph
  | BarGraph
  | PieChart
  | LineGraph
  | FrequencyDistributionHistogram

-- Define a property for highlighting part-whole relationships
def highlightsPartWholeRelationship (graph : StatisticalGraph) : Prop :=
  match graph with
  | StatisticalGraph.PieChart => true
  | _ => false

-- Theorem statement
theorem pie_chart_best_for_part_whole : 
  ∀ (graph : StatisticalGraph), 
    highlightsPartWholeRelationship graph → graph = StatisticalGraph.PieChart :=
by
  sorry


end NUMINAMATH_CALUDE_pie_chart_best_for_part_whole_l725_72580


namespace NUMINAMATH_CALUDE_complex_equality_l725_72594

theorem complex_equality (a : ℝ) (z : ℂ) : 
  z = (a + 3*I) / (1 + 2*I) → z.re = z.im → a = -1 := by sorry

end NUMINAMATH_CALUDE_complex_equality_l725_72594


namespace NUMINAMATH_CALUDE_point_on_line_l725_72577

/-- A point (x, y) lies on a line passing through two points (x₁, y₁) and (x₂, y₂) if it satisfies the equation of the line. -/
def lies_on_line (x y x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y - y₁ = ((y₂ - y₁) / (x₂ - x₁)) * (x - x₁)

/-- The point (0,3) lies on the line passing through (-2,1) and (2,5). -/
theorem point_on_line : lies_on_line 0 3 (-2) 1 2 5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l725_72577


namespace NUMINAMATH_CALUDE_orthic_triangle_smallest_perimeter_l725_72587

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is acute-angled -/
def isAcuteAngled (t : Triangle) : Prop := sorry

/-- Calculates the perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := sorry

/-- Checks if a point is on a line segment between two other points -/
def isOnSegment (P : Point) (A : Point) (B : Point) : Prop := sorry

/-- Checks if a triangle is inscribed in another triangle -/
def isInscribed (inner outer : Triangle) : Prop := 
  isOnSegment inner.A outer.B outer.C ∧
  isOnSegment inner.B outer.A outer.C ∧
  isOnSegment inner.C outer.A outer.B

/-- Constructs the orthic triangle of a given triangle -/
def orthicTriangle (t : Triangle) : Triangle := sorry

/-- The main theorem: the orthic triangle has the smallest perimeter among all inscribed triangles -/
theorem orthic_triangle_smallest_perimeter (ABC : Triangle) 
  (h_acute : isAcuteAngled ABC) :
  let PQR := orthicTriangle ABC
  ∀ XYZ : Triangle, isInscribed XYZ ABC → perimeter PQR ≤ perimeter XYZ := by
  sorry

end NUMINAMATH_CALUDE_orthic_triangle_smallest_perimeter_l725_72587


namespace NUMINAMATH_CALUDE_integral_x_cubed_minus_reciprocal_x_fourth_l725_72590

theorem integral_x_cubed_minus_reciprocal_x_fourth (f : ℝ → ℝ) :
  (∀ x, f x = x^3 - 1/x^4) →
  ∫ x in (-1)..1, f x = 2/3 := by sorry

end NUMINAMATH_CALUDE_integral_x_cubed_minus_reciprocal_x_fourth_l725_72590


namespace NUMINAMATH_CALUDE_john_distance_l725_72595

/-- Calculates the total distance John travels given his speeds and running times -/
def total_distance (solo_speed : ℝ) (dog_speed : ℝ) (time_with_dog : ℝ) (time_solo : ℝ) : ℝ :=
  dog_speed * time_with_dog + solo_speed * time_solo

/-- Proves that John travels 5 miles given the specified conditions -/
theorem john_distance :
  let solo_speed : ℝ := 4
  let dog_speed : ℝ := 6
  let time_with_dog : ℝ := 0.5
  let time_solo : ℝ := 0.5
  total_distance solo_speed dog_speed time_with_dog time_solo = 5 := by
  sorry

#eval total_distance 4 6 0.5 0.5

end NUMINAMATH_CALUDE_john_distance_l725_72595


namespace NUMINAMATH_CALUDE_zane_bought_two_shirts_l725_72558

/-- Calculates the number of polo shirts bought given the discount percentage, regular price, and total amount paid. -/
def polo_shirts_bought (discount_percent : ℚ) (regular_price : ℚ) (total_paid : ℚ) : ℚ :=
  let discounted_price := regular_price * (1 - discount_percent)
  total_paid / discounted_price

/-- Proves that Zane bought 2 polo shirts given the specified conditions. -/
theorem zane_bought_two_shirts : 
  polo_shirts_bought (40/100) 50 60 = 2 := by
  sorry

#eval polo_shirts_bought (40/100) 50 60

end NUMINAMATH_CALUDE_zane_bought_two_shirts_l725_72558


namespace NUMINAMATH_CALUDE_roof_tiles_needed_l725_72511

def land_cost_per_sqm : ℕ := 50
def brick_cost_per_thousand : ℕ := 100
def roof_tile_cost : ℕ := 10
def land_area : ℕ := 2000
def brick_count : ℕ := 10000
def total_cost : ℕ := 106000

theorem roof_tiles_needed : ℕ := by
  -- The number of roof tiles needed is 500
  sorry

end NUMINAMATH_CALUDE_roof_tiles_needed_l725_72511


namespace NUMINAMATH_CALUDE_polynomial_roots_l725_72546

theorem polynomial_roots : ∃ (a b : ℝ), 
  (∀ x : ℝ, 6*x^4 + 25*x^3 - 59*x^2 + 28*x = 0 ↔ 
    x = 0 ∨ x = 1 ∨ x = (-31 + Real.sqrt 1633) / 12 ∨ x = (-31 - Real.sqrt 1633) / 12) ∧
  a = (-31 + Real.sqrt 1633) / 12 ∧
  b = (-31 - Real.sqrt 1633) / 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l725_72546


namespace NUMINAMATH_CALUDE_complex_equation_solution_l725_72531

theorem complex_equation_solution :
  ∀ z : ℂ, (1 + 2*I) * z = 1 - I → z = -1/5 - 3/5*I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l725_72531


namespace NUMINAMATH_CALUDE_conditional_probability_B_given_A_l725_72528

/-- The total number of zongzi -/
def total_zongzi : ℕ := 5

/-- The number of zongzi with pork filling -/
def pork_zongzi : ℕ := 2

/-- The number of zongzi with red bean paste filling -/
def red_bean_zongzi : ℕ := 3

/-- Event A: the two picked zongzi have the same filling -/
def event_A : Set (Fin total_zongzi × Fin total_zongzi) := sorry

/-- Event B: the two picked zongzi both have red bean paste filling -/
def event_B : Set (Fin total_zongzi × Fin total_zongzi) := sorry

/-- The probability measure on the sample space -/
def P : Set (Fin total_zongzi × Fin total_zongzi) → ℝ := sorry

theorem conditional_probability_B_given_A :
  P event_B / P event_A = 3/4 := by sorry

end NUMINAMATH_CALUDE_conditional_probability_B_given_A_l725_72528


namespace NUMINAMATH_CALUDE_paving_rate_calculation_l725_72550

/-- Given a room with length and width, and the total cost of paving,
    calculate the rate of paving per square meter. -/
theorem paving_rate_calculation (length width total_cost : ℝ) :
  length = 5.5 ∧ width = 3.75 ∧ total_cost = 20625 →
  total_cost / (length * width) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_paving_rate_calculation_l725_72550


namespace NUMINAMATH_CALUDE_log_equation_solution_l725_72598

theorem log_equation_solution (x : ℝ) :
  (Real.log x / Real.log 8 + Real.log (x^3) / Real.log 4 = 9) ↔ (x = 2^(54/11)) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l725_72598


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l725_72578

theorem complex_fraction_simplification :
  (5 + 3*Complex.I) / (2 + 3*Complex.I) = 19/13 - 9/13 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l725_72578


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l725_72501

-- Define an isosceles triangle
structure IsoscelesTriangle :=
  (vertex_angle : ℝ)
  (base_angle : ℝ)
  (is_isosceles : base_angle = base_angle)

-- Define the exterior angle
def exterior_angle (t : IsoscelesTriangle) : ℝ := 100

-- Theorem statement
theorem isosceles_triangle_vertex_angle 
  (t : IsoscelesTriangle) 
  (h_exterior : exterior_angle t = 100) :
  t.vertex_angle = 20 ∨ t.vertex_angle = 80 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l725_72501


namespace NUMINAMATH_CALUDE_fern_leaves_count_l725_72556

/-- The number of ferns Karen hangs -/
def num_ferns : ℕ := 6

/-- The number of fronds each fern has -/
def fronds_per_fern : ℕ := 7

/-- The number of leaves each frond has -/
def leaves_per_frond : ℕ := 30

/-- The total number of leaves on all ferns -/
def total_leaves : ℕ := num_ferns * fronds_per_fern * leaves_per_frond

theorem fern_leaves_count : total_leaves = 1260 := by
  sorry

end NUMINAMATH_CALUDE_fern_leaves_count_l725_72556


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l725_72507

theorem smallest_positive_integer_with_remainders : ∃! x : ℕ, 
  x > 0 ∧ 
  x % 5 = 4 ∧ 
  x % 7 = 6 ∧ 
  x % 8 = 7 ∧ 
  ∀ y : ℕ, y > 0 ∧ y % 5 = 4 ∧ y % 7 = 6 ∧ y % 8 = 7 → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l725_72507


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l725_72502

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := 16/21
  let a₃ : ℚ := 64/63
  let r : ℚ := a₂ / a₁
  r = 4/3 := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l725_72502


namespace NUMINAMATH_CALUDE_x_minus_q_equals_three_minus_two_q_l725_72576

theorem x_minus_q_equals_three_minus_two_q (x q : ℝ) 
  (h1 : |x - 3| = q) 
  (h2 : x < 3) : 
  x - q = 3 - 2*q := by
sorry

end NUMINAMATH_CALUDE_x_minus_q_equals_three_minus_two_q_l725_72576


namespace NUMINAMATH_CALUDE_four_vertices_unique_distances_five_vertices_not_unique_distances_l725_72524

/-- Represents a regular 13-sided polygon -/
structure RegularPolygon13 where
  vertices : Fin 13 → ℝ × ℝ

/-- Calculates the distance between two vertices in a regular 13-sided polygon -/
def distance (p : RegularPolygon13) (v1 v2 : Fin 13) : ℝ := sorry

/-- Checks if all pairwise distances in a set of vertices are unique -/
def all_distances_unique (p : RegularPolygon13) (vs : Finset (Fin 13)) : Prop := sorry

theorem four_vertices_unique_distances (p : RegularPolygon13) :
  ∃ (vs : Finset (Fin 13)), vs.card = 4 ∧ all_distances_unique p vs := sorry

theorem five_vertices_not_unique_distances (p : RegularPolygon13) :
  ¬∃ (vs : Finset (Fin 13)), vs.card = 5 ∧ all_distances_unique p vs := sorry

end NUMINAMATH_CALUDE_four_vertices_unique_distances_five_vertices_not_unique_distances_l725_72524


namespace NUMINAMATH_CALUDE_sum_of_digits_2017_power_l725_72582

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem stating that S(S(S(S(2017^2017)))) = 1 -/
theorem sum_of_digits_2017_power : S (S (S (S (2017^2017)))) = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_2017_power_l725_72582


namespace NUMINAMATH_CALUDE_equation_solution_l725_72538

theorem equation_solution (n : ℝ) : 
  (1 / (n + 1) + 2 / (n + 1) + n / (n + 1) + 1 / (n + 2) = 4) ↔ 
  (n = (-3 + Real.sqrt 6) / 3 ∨ n = (-3 - Real.sqrt 6) / 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l725_72538


namespace NUMINAMATH_CALUDE_percentage_change_condition_l725_72579

theorem percentage_change_condition
  (p q r M : ℝ)
  (hp : p > 0)
  (hq : 0 < q ∧ q < 100)
  (hr : 0 < r ∧ r < 100)
  (hM : M > 0) :
  M * (1 + p / 100) * (1 - q / 100) * (1 - r / 100) > M ↔
  p > (100 * (q + r)) / (100 - q - r) :=
by sorry

end NUMINAMATH_CALUDE_percentage_change_condition_l725_72579


namespace NUMINAMATH_CALUDE_original_price_calculation_l725_72534

/-- Given a discount percentage and a discounted price, calculates the original price. -/
def calculateOriginalPrice (discountPercentage : ℚ) (discountedPrice : ℚ) : ℚ :=
  discountedPrice / (1 - discountPercentage)

/-- Theorem: Given a 30% discount and a discounted price of 560, the original price is 800. -/
theorem original_price_calculation :
  calculateOriginalPrice (30 / 100) 560 = 800 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l725_72534


namespace NUMINAMATH_CALUDE_picture_area_l725_72568

/-- Given a sheet of paper with specified dimensions and margin, calculate the area of the picture --/
theorem picture_area (paper_width paper_length margin : ℝ) 
  (hw : paper_width = 8.5)
  (hl : paper_length = 10)
  (hm : margin = 1.5) : 
  (paper_width - 2 * margin) * (paper_length - 2 * margin) = 38.5 := by
  sorry

#check picture_area

end NUMINAMATH_CALUDE_picture_area_l725_72568


namespace NUMINAMATH_CALUDE_external_internal_triangles_form_parallelogram_l725_72536

-- Define the basic structures
structure Point :=
  (x y : ℝ)

structure Triangle :=
  (A B C : Point)

structure Quadrilateral :=
  (A B C D : Point)

-- Define the properties
def isEquilateral (t : Triangle) : Prop :=
  sorry

def areSimilar (t1 t2 : Triangle) : Prop :=
  sorry

def isParallelogram (q : Quadrilateral) : Prop :=
  sorry

def constructedExternally (base outer : Triangle) : Prop :=
  sorry

def constructedInternally (base inner : Triangle) : Prop :=
  sorry

-- State the theorem
theorem external_internal_triangles_form_parallelogram
  (ABC : Triangle)
  (AB₁C AC₁B BA₁C : Triangle)
  (ABB₁AC₁ : Quadrilateral) :
  isEquilateral AB₁C ∧
  isEquilateral AC₁B ∧
  areSimilar AB₁C ABC ∧
  areSimilar AC₁B ABC ∧
  constructedExternally ABC AB₁C ∧
  constructedExternally ABC AC₁B ∧
  constructedInternally ABC BA₁C ∧
  ABB₁AC₁.A = ABC.A ∧
  ABB₁AC₁.B = AB₁C.B ∧
  ABB₁AC₁.C = AC₁B.C ∧
  ABB₁AC₁.D = BA₁C.A →
  isParallelogram ABB₁AC₁ :=
sorry

end NUMINAMATH_CALUDE_external_internal_triangles_form_parallelogram_l725_72536


namespace NUMINAMATH_CALUDE_continuous_function_integrable_l725_72513

theorem continuous_function_integrable 
  {a b : ℝ} (f : ℝ → ℝ) (h : ContinuousOn f (Set.Icc a b)) : 
  IntervalIntegrable f volume a b :=
sorry

end NUMINAMATH_CALUDE_continuous_function_integrable_l725_72513


namespace NUMINAMATH_CALUDE_existence_of_x_l725_72583

theorem existence_of_x (a : Fin 1997 → ℕ)
  (h1 : ∀ i j : Fin 1997, i + j ≤ 1997 → a i + a j ≤ a (i + j))
  (h2 : ∀ i j : Fin 1997, i + j ≤ 1997 → a (i + j) ≤ a i + a j + 1) :
  ∃ x : ℝ, ∀ n : Fin 1997, a n = ⌊n * x⌋ := by
sorry

end NUMINAMATH_CALUDE_existence_of_x_l725_72583


namespace NUMINAMATH_CALUDE_problem_statement_l725_72570

theorem problem_statement (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x + 2) = (-5 * x^2 + 18 * x + 40) / (x - 3)) →
  P + Q = 44 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l725_72570


namespace NUMINAMATH_CALUDE_necklace_profit_l725_72581

/-- Calculate the profit from selling necklaces -/
theorem necklace_profit
  (charms_per_necklace : ℕ)
  (charm_cost : ℕ)
  (selling_price : ℕ)
  (necklaces_sold : ℕ)
  (h1 : charms_per_necklace = 10)
  (h2 : charm_cost = 15)
  (h3 : selling_price = 200)
  (h4 : necklaces_sold = 30) :
  (selling_price - charms_per_necklace * charm_cost) * necklaces_sold = 1500 :=
by sorry

end NUMINAMATH_CALUDE_necklace_profit_l725_72581


namespace NUMINAMATH_CALUDE_union_M_N_equals_N_l725_72563

open Set

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + x > 0}
def N : Set ℝ := {x | |x| > 2}

-- State the theorem
theorem union_M_N_equals_N : M ∪ N = N := by sorry

end NUMINAMATH_CALUDE_union_M_N_equals_N_l725_72563


namespace NUMINAMATH_CALUDE_tangent_line_of_odd_function_l725_72593

/-- Given function f(x) = (a-1)x^2 - a*sin(x) is odd, 
    prove that its tangent line at (0,0) is y = -x -/
theorem tangent_line_of_odd_function (a : ℝ) :
  (∀ x, ((a - 1) * x^2 - a * Real.sin x) = -((a - 1) * (-x)^2 - a * Real.sin (-x))) →
  (∃ f : ℝ → ℝ, (∀ x, f x = (a - 1) * x^2 - a * Real.sin x) ∧ 
    (∃ f' : ℝ → ℝ, (∀ x, HasDerivAt f (f' x) x) ∧ f' 0 = -1)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_of_odd_function_l725_72593


namespace NUMINAMATH_CALUDE_money_distribution_l725_72566

/-- Given three people A, B, and C with the following money distribution:
    - A, B, and C have Rs. 700 in total
    - A and C together have Rs. 300
    - C has Rs. 200
    Prove that B and C together have Rs. 600 -/
theorem money_distribution (total : ℕ) (ac_sum : ℕ) (c_money : ℕ) :
  total = 700 →
  ac_sum = 300 →
  c_money = 200 →
  ∃ (a b : ℕ), a + b + c_money = total ∧ a + c_money = ac_sum ∧ b + c_money = 600 :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l725_72566


namespace NUMINAMATH_CALUDE_zoo_trip_cost_l725_72533

def zoo_ticket_cost : ℝ := 5
def bus_fare_one_way : ℝ := 1.5
def num_people : ℕ := 2
def lunch_snacks_money : ℝ := 24

def total_zoo_tickets : ℝ := zoo_ticket_cost * num_people
def total_bus_fare : ℝ := bus_fare_one_way * num_people * 2

theorem zoo_trip_cost : total_zoo_tickets + total_bus_fare + lunch_snacks_money = 40 := by
  sorry

end NUMINAMATH_CALUDE_zoo_trip_cost_l725_72533


namespace NUMINAMATH_CALUDE_expansion_coefficient_l725_72529

/-- The coefficient of a^2 * b^3 * c^3 in the expansion of (a + b + c)^8 -/
def coefficient_a2b3c3 : ℕ :=
  Nat.choose 8 5 * Nat.choose 5 3

theorem expansion_coefficient :
  coefficient_a2b3c3 = 560 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l725_72529


namespace NUMINAMATH_CALUDE_unique_monotonic_involutive_function_l725_72515

-- Define the properties of the function
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

def Involutive (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (f x) = x

-- Theorem statement
theorem unique_monotonic_involutive_function :
  ∀ f : ℝ → ℝ, Monotonic f → Involutive f → ∀ x : ℝ, f x = x :=
by
  sorry


end NUMINAMATH_CALUDE_unique_monotonic_involutive_function_l725_72515


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l725_72599

/-- A geometric sequence of positive integers with first term 5 and fifth term 3125 has its fourth term equal to 625. -/
theorem geometric_sequence_fourth_term : ∀ (a : ℕ → ℕ),
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 5 →                            -- First term is 5
  a 5 = 3125 →                         -- Fifth term is 3125
  a 4 = 625 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l725_72599


namespace NUMINAMATH_CALUDE_initial_average_age_l725_72503

theorem initial_average_age (n : ℕ) (new_age : ℕ) (new_average : ℕ) 
  (h1 : n = 9)
  (h2 : new_age = 35)
  (h3 : new_average = 17) :
  ∃ initial_average : ℚ, 
    initial_average = 15 ∧ 
    (n : ℚ) * initial_average + new_age = ((n : ℚ) + 1) * new_average :=
by sorry

end NUMINAMATH_CALUDE_initial_average_age_l725_72503


namespace NUMINAMATH_CALUDE_binary_repr_24_l725_72523

/-- The binary representation of a natural number -/
def binary_repr (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Theorem: The binary representation of 24 is [false, false, false, true, true] -/
theorem binary_repr_24 : binary_repr 24 = [false, false, false, true, true] := by
  sorry

end NUMINAMATH_CALUDE_binary_repr_24_l725_72523


namespace NUMINAMATH_CALUDE_archibald_tennis_game_l725_72535

/-- Archibald's Tennis Game Problem -/
theorem archibald_tennis_game (games_won_by_archibald : ℕ) (percentage_won_by_archibald : ℚ) :
  games_won_by_archibald = 12 →
  percentage_won_by_archibald = 2/5 →
  ∃ (total_games : ℕ) (games_won_by_brother : ℕ),
    total_games = games_won_by_archibald + games_won_by_brother ∧
    (games_won_by_archibald : ℚ) / total_games = percentage_won_by_archibald ∧
    games_won_by_brother = 18 :=
by
  sorry

#check archibald_tennis_game

end NUMINAMATH_CALUDE_archibald_tennis_game_l725_72535


namespace NUMINAMATH_CALUDE_fruits_in_box_l725_72564

/-- The number of fruits in a box after adding persimmons and apples -/
theorem fruits_in_box (persimmons apples : ℕ) : persimmons = 2 → apples = 7 → persimmons + apples = 9 := by
  sorry

end NUMINAMATH_CALUDE_fruits_in_box_l725_72564


namespace NUMINAMATH_CALUDE_equal_digit_probability_l725_72575

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The number of sides on each die -/
def num_sides : ℕ := 20

/-- The number of one-digit numbers on each die -/
def one_digit_count : ℕ := 9

/-- The number of two-digit numbers on each die -/
def two_digit_count : ℕ := 11

/-- The probability of rolling an equal number of one-digit and two-digit numbers -/
def equal_digit_prob : ℚ := 539055 / 1600000

theorem equal_digit_probability :
  let p_one_digit : ℚ := one_digit_count / num_sides
  let p_two_digit : ℚ := two_digit_count / num_sides
  (num_dice.choose (num_dice / 2)) * p_one_digit ^ (num_dice / 2) * p_two_digit ^ (num_dice / 2) = equal_digit_prob := by
  sorry

end NUMINAMATH_CALUDE_equal_digit_probability_l725_72575


namespace NUMINAMATH_CALUDE_stratified_sampling_male_athletes_l725_72554

/-- Represents the total number of athletes -/
def total_athletes : ℕ := 30

/-- Represents the number of male athletes -/
def male_athletes : ℕ := 20

/-- Represents the number of female athletes -/
def female_athletes : ℕ := 10

/-- Represents the sample size -/
def sample_size : ℕ := 6

/-- Represents the number of male athletes to be sampled -/
def male_sample_size : ℚ := (male_athletes : ℚ) * (sample_size : ℚ) / (total_athletes : ℚ)

theorem stratified_sampling_male_athletes :
  male_sample_size = 4 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_male_athletes_l725_72554


namespace NUMINAMATH_CALUDE_cookie_value_approx_l725_72552

/-- Calculates the value of cookies left after a series of operations -/
def cookie_value (initial : ℝ) (eaten1 : ℝ) (received1 : ℝ) (eaten2 : ℝ) (received2 : ℝ) (share_fraction : ℝ) (cost_per_cookie : ℝ) : ℝ :=
  let remaining1 := initial - eaten1
  let after_first_gift := remaining1 + received1
  let after_second_eating := after_first_gift - eaten2
  let total_before_sharing := after_second_eating + received2
  let shared := total_before_sharing * share_fraction
  let final_count := total_before_sharing - shared
  final_count * cost_per_cookie

/-- The value of cookies left is approximately $1.73 -/
theorem cookie_value_approx :
  ∃ ε > 0, |cookie_value 7 2.5 4.2 1.3 3 (1/3) 0.25 - 1.73| < ε :=
sorry

end NUMINAMATH_CALUDE_cookie_value_approx_l725_72552


namespace NUMINAMATH_CALUDE_deposit_maturity_equation_l725_72545

/-- Represents the cash amount paid to the depositor upon maturity -/
def x : ℝ := sorry

/-- The initial deposit amount in yuan -/
def initial_deposit : ℝ := 5000

/-- The interest rate for one-year fixed deposits -/
def interest_rate : ℝ := 0.0306

/-- The interest tax rate -/
def tax_rate : ℝ := 0.20

theorem deposit_maturity_equation :
  x + initial_deposit * interest_rate * tax_rate = initial_deposit * (1 + interest_rate) :=
by sorry

end NUMINAMATH_CALUDE_deposit_maturity_equation_l725_72545


namespace NUMINAMATH_CALUDE_eraser_cost_proof_l725_72504

/-- Represents the price of a single pencil -/
def pencil_price : ℝ := 2

/-- Represents the price of a single eraser -/
def eraser_price : ℝ := 1

/-- The number of pencils sold -/
def pencils_sold : ℕ := 20

/-- The number of erasers sold -/
def erasers_sold : ℕ := pencils_sold * 2

/-- The total revenue from sales -/
def total_revenue : ℝ := 80

theorem eraser_cost_proof :
  (pencils_sold : ℝ) * pencil_price + (erasers_sold : ℝ) * eraser_price = total_revenue ∧
  2 * eraser_price = pencil_price ∧
  eraser_price = 1 :=
by sorry

end NUMINAMATH_CALUDE_eraser_cost_proof_l725_72504


namespace NUMINAMATH_CALUDE_factorial_10_mod_11_l725_72540

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_10_mod_11 : factorial 10 % 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_factorial_10_mod_11_l725_72540


namespace NUMINAMATH_CALUDE_quadratic_sum_l725_72518

theorem quadratic_sum (x y : ℝ) 
  (h1 : 4 * x + 2 * y = 12) 
  (h2 : 2 * x + 4 * y = 20) : 
  20 * x^2 + 24 * x * y + 20 * y^2 = 544 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l725_72518


namespace NUMINAMATH_CALUDE_imaginary_part_of_fraction_l725_72543

noncomputable def i : ℂ := Complex.I

theorem imaginary_part_of_fraction (z : ℂ) : z = 2016 / (1 + i) → Complex.im z = -1008 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_fraction_l725_72543


namespace NUMINAMATH_CALUDE_max_value_theorem_l725_72592

theorem max_value_theorem (x y z : ℝ) (h : 9*x^2 + 4*y^2 + 25*z^2 = 1) :
  8*x + 5*y + 15*z ≤ 28 / Real.sqrt 38 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l725_72592


namespace NUMINAMATH_CALUDE_fencing_cost_l725_72553

/-- The total cost of fencing a rectangular field with a square pond -/
theorem fencing_cost (field_area : ℝ) (outer_fence_cost : ℝ) (pond_fence_cost : ℝ) : 
  field_area = 10800 ∧ 
  outer_fence_cost = 1.5 ∧ 
  pond_fence_cost = 1 → 
  ∃ (short_side long_side pond_side : ℝ),
    short_side * long_side = field_area ∧
    long_side = (4/3) * short_side ∧
    pond_side = (1/6) * short_side ∧
    2 * (short_side + long_side) * outer_fence_cost + 4 * pond_side * pond_fence_cost = 690 :=
by sorry

end NUMINAMATH_CALUDE_fencing_cost_l725_72553


namespace NUMINAMATH_CALUDE_least_three_digit_7_heavy_l725_72549

def is_7_heavy (n : ℕ) : Prop := n % 7 ≥ 5

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_7_heavy : 
  (∀ n : ℕ, is_three_digit n → is_7_heavy n → 103 ≤ n) ∧ 
  is_three_digit 103 ∧ 
  is_7_heavy 103 :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_7_heavy_l725_72549


namespace NUMINAMATH_CALUDE_max_sum_semi_axes_l725_72565

/-- The maximum sum of semi-axes of an ellipse and hyperbola with the same foci -/
theorem max_sum_semi_axes (m n : ℝ) : 
  m > 0 → n > 0 → 
  (∃ x y : ℝ, x^2/25 + y^2/m^2 = 1) →
  (∃ x y : ℝ, x^2/7 - y^2/n^2 = 1) →
  (25 - m^2 = 7 + n^2) →
  (∀ m' n' : ℝ, m' > 0 → n' > 0 → 
    (∃ x y : ℝ, x^2/25 + y^2/m'^2 = 1) →
    (∃ x y : ℝ, x^2/7 - y^2/n'^2 = 1) →
    (25 - m'^2 = 7 + n'^2) →
    m + n ≥ m' + n') →
  m + n = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_semi_axes_l725_72565


namespace NUMINAMATH_CALUDE_reema_loan_interest_l725_72569

/-- Calculate simple interest given principal, rate, and time -/
def simple_interest (principal rate time : ℕ) : ℕ :=
  principal * rate * time / 100

theorem reema_loan_interest :
  let principal : ℕ := 1200
  let rate : ℕ := 6
  let time : ℕ := rate
  simple_interest principal rate time = 432 := by
  sorry

end NUMINAMATH_CALUDE_reema_loan_interest_l725_72569


namespace NUMINAMATH_CALUDE_smallest_divisible_number_l725_72589

theorem smallest_divisible_number (n : ℕ) : 
  n = 719 + 288721 → 
  (∀ m : ℕ, 719 < m ∧ m < n → ¬(618 ∣ m ∧ 3648 ∣ m ∧ 60 ∣ m)) ∧ 
  (618 ∣ n ∧ 3648 ∣ n ∧ 60 ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_number_l725_72589


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l725_72541

theorem square_plus_reciprocal_square (x : ℝ) (h : x ≠ 0) :
  x^4 + (1/x^4) = 47 → x^2 + (1/x^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l725_72541


namespace NUMINAMATH_CALUDE_solution_in_interval_l725_72516

-- Define the function f(x) = x^3 + x - 4
def f (x : ℝ) : ℝ := x^3 + x - 4

-- State the theorem
theorem solution_in_interval :
  ∃ x : ℝ, x ∈ Set.Ioo 1 (3/2) ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_in_interval_l725_72516


namespace NUMINAMATH_CALUDE_cone_volume_l725_72560

def slant_height : ℝ := 5
def base_radius : ℝ := 3

theorem cone_volume : 
  let height := Real.sqrt (slant_height^2 - base_radius^2)
  (1/3 : ℝ) * Real.pi * base_radius^2 * height = 12 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l725_72560


namespace NUMINAMATH_CALUDE_work_completion_time_l725_72542

/-- The number of days A takes to complete the work -/
def a_days : ℝ := 12

/-- B's efficiency compared to A -/
def b_efficiency : ℝ := 1.2

/-- The number of days B takes to complete the work -/
def b_days : ℝ := 10

theorem work_completion_time :
  a_days * b_efficiency = b_days := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l725_72542


namespace NUMINAMATH_CALUDE_breakfast_cost_theorem_l725_72572

/-- The cost of breakfast for Francis and Kiera -/
def breakfast_cost (muffin_price fruit_cup_price : ℕ) 
  (francis_muffins francis_fruit_cups : ℕ)
  (kiera_muffins kiera_fruit_cups : ℕ) : ℕ :=
  (francis_muffins + kiera_muffins) * muffin_price +
  (francis_fruit_cups + kiera_fruit_cups) * fruit_cup_price

/-- Theorem stating the total cost of Francis and Kiera's breakfast -/
theorem breakfast_cost_theorem : 
  breakfast_cost 2 3 2 2 2 1 = 17 := by
  sorry

end NUMINAMATH_CALUDE_breakfast_cost_theorem_l725_72572


namespace NUMINAMATH_CALUDE_concrete_components_correct_l725_72591

/-- Represents the ratio of cement, sand, and gravel in the concrete mixture -/
def concrete_ratio : Fin 3 → ℕ
  | 0 => 2  -- cement
  | 1 => 4  -- sand
  | 2 => 5  -- gravel

/-- The total amount of concrete needed in tons -/
def total_concrete : ℕ := 121

/-- Calculates the amount of a component needed based on its ratio and the total concrete amount -/
def component_amount (ratio : ℕ) (total_ratio : ℕ) (total_amount : ℕ) : ℕ :=
  (ratio * total_amount) / total_ratio

/-- Theorem stating the correct amounts of cement and gravel needed -/
theorem concrete_components_correct :
  let total_ratio := (concrete_ratio 0) + (concrete_ratio 1) + (concrete_ratio 2)
  component_amount (concrete_ratio 0) total_ratio total_concrete = 22 ∧
  component_amount (concrete_ratio 2) total_ratio total_concrete = 55 := by
  sorry


end NUMINAMATH_CALUDE_concrete_components_correct_l725_72591
