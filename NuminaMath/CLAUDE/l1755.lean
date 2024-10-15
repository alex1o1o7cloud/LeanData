import Mathlib

namespace NUMINAMATH_CALUDE_special_triangle_properties_l1755_175555

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Properties of the specific triangle in the problem -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.a + t.b + t.c = Real.sqrt 2 + 1 ∧
  Real.sin t.A + Real.sin t.B = Real.sqrt 2 * Real.sin t.C ∧
  (1/2) * t.a * t.b * Real.sin t.C = (1/5) * Real.sin t.C

theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  t.c = 1 ∧ Real.cos t.C = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_special_triangle_properties_l1755_175555


namespace NUMINAMATH_CALUDE_memory_card_cost_l1755_175549

/-- If three identical memory cards cost $45 in total, then eight of these memory cards will cost $120. -/
theorem memory_card_cost (cost_of_three : ℝ) : cost_of_three = 45 → 8 * (cost_of_three / 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_memory_card_cost_l1755_175549


namespace NUMINAMATH_CALUDE_ray_initial_cents_l1755_175519

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of cents Ray gives to Peter -/
def cents_to_peter : ℕ := 25

/-- The number of nickels Ray has left after giving away cents -/
def nickels_left : ℕ := 4

/-- The initial number of cents Ray had -/
def initial_cents : ℕ := 95

theorem ray_initial_cents :
  initial_cents = 
    cents_to_peter + 
    (2 * cents_to_peter) + 
    (nickels_left * nickel_value) :=
by sorry

end NUMINAMATH_CALUDE_ray_initial_cents_l1755_175519


namespace NUMINAMATH_CALUDE_max_square_plots_l1755_175595

/-- Represents the field dimensions and available fencing -/
structure FieldData where
  width : ℝ
  length : ℝ
  fence : ℝ

/-- Calculates the number of square plots given the number of plots along the width -/
def numPlots (n : ℕ) : ℕ := n * (2 * n)

/-- Calculates the length of fence used given the number of plots along the width -/
def fenceUsed (n : ℕ) : ℝ := 120 * n - 90

/-- The main theorem stating the maximum number of square plots -/
theorem max_square_plots (field : FieldData) 
    (h_width : field.width = 30)
    (h_length : field.length = 60)
    (h_fence : field.fence = 2268) : 
  (∃ (n : ℕ), numPlots n = 722 ∧ 
              fenceUsed n ≤ field.fence ∧ 
              ∀ (m : ℕ), m > n → fenceUsed m > field.fence) :=
sorry

end NUMINAMATH_CALUDE_max_square_plots_l1755_175595


namespace NUMINAMATH_CALUDE_existence_of_n_with_k_prime_factors_l1755_175542

theorem existence_of_n_with_k_prime_factors (k m : ℕ) (hk : k > 0) (hm : m > 0) (hm_odd : Odd m) :
  ∃ n : ℕ, n > 0 ∧ (∃ (S : Finset ℕ), S.card ≥ k ∧ ∀ p ∈ S, Prime p ∧ p ∣ (m^n + n^m)) :=
sorry

end NUMINAMATH_CALUDE_existence_of_n_with_k_prime_factors_l1755_175542


namespace NUMINAMATH_CALUDE_mike_catches_l1755_175561

/-- The number of times Joe caught the ball -/
def J : ℕ := 23

/-- The number of times Derek caught the ball -/
def D : ℕ := 2 * J - 4

/-- The number of times Tammy caught the ball -/
def T : ℕ := (D / 3) + 16

/-- The number of times Mike caught the ball -/
def M : ℕ := (2 * T * 120) / 100

theorem mike_catches : M = 72 := by
  sorry

end NUMINAMATH_CALUDE_mike_catches_l1755_175561


namespace NUMINAMATH_CALUDE_hyperbola_to_ellipse_l1755_175570

/-- Given a hyperbola with equation x²/4 - y²/12 = -1, 
    the ellipse with foci at the vertices of this hyperbola 
    has the equation x²/4 + y²/16 = 1 -/
theorem hyperbola_to_ellipse : 
  ∃ (h : Set (ℝ × ℝ)) (e : Set (ℝ × ℝ)),
    (h = {(x, y) | x^2/4 - y^2/12 = -1}) →
    (e = {(x, y) | x^2/4 + y^2/16 = 1}) →
    (∀ (fx fy : ℝ), (fx, fy) ∈ {v | v ∈ h ∧ (∀ (x y : ℝ), (x, y) ∈ h → x^2 + y^2 ≤ fx^2 + fy^2)} →
      (fx, fy) ∈ {f | f ∈ e ∧ (∀ (x y : ℝ), (x, y) ∈ e → (x - fx)^2 + (y - fy)^2 ≥ 
        (x + fx)^2 + (y + fy)^2)}) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_to_ellipse_l1755_175570


namespace NUMINAMATH_CALUDE_like_terms_sum_l1755_175507

theorem like_terms_sum (a b : ℤ) : 
  (a + 1 = 2) ∧ (b - 2 = 3) → a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_sum_l1755_175507


namespace NUMINAMATH_CALUDE_magnitude_of_vector_difference_l1755_175547

def vector_a : Fin 2 → ℝ := ![2, 1]
def vector_b : Fin 2 → ℝ := ![-2, 4]

theorem magnitude_of_vector_difference :
  Real.sqrt ((vector_a 0 - vector_b 0)^2 + (vector_a 1 - vector_b 1)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_difference_l1755_175547


namespace NUMINAMATH_CALUDE_planet_combinations_count_l1755_175530

/-- Represents the number of Earth-like planets -/
def earth_like_planets : ℕ := 5

/-- Represents the number of Mars-like planets -/
def mars_like_planets : ℕ := 6

/-- Represents the colonization units required for an Earth-like planet -/
def earth_like_units : ℕ := 2

/-- Represents the colonization units required for a Mars-like planet -/
def mars_like_units : ℕ := 1

/-- Represents the total available colonization units -/
def total_units : ℕ := 12

/-- Calculates the number of ways to choose planets given the constraints -/
def count_planet_combinations : ℕ :=
  (Nat.choose earth_like_planets 3 * Nat.choose mars_like_planets 6) +
  (Nat.choose earth_like_planets 4 * Nat.choose mars_like_planets 4) +
  (Nat.choose earth_like_planets 5 * Nat.choose mars_like_planets 2)

/-- Theorem stating that the number of planet combinations is 100 -/
theorem planet_combinations_count :
  count_planet_combinations = 100 := by sorry

end NUMINAMATH_CALUDE_planet_combinations_count_l1755_175530


namespace NUMINAMATH_CALUDE_sheridan_cats_l1755_175510

/-- The number of cats Mrs. Sheridan gave away -/
def cats_given_away : ℝ := 14.0

/-- The number of cats Mrs. Sheridan has left -/
def cats_left : ℕ := 3

/-- The initial number of cats Mrs. Sheridan had -/
def initial_cats : ℕ := 17

theorem sheridan_cats : ↑initial_cats = cats_given_away + cats_left := by sorry

end NUMINAMATH_CALUDE_sheridan_cats_l1755_175510


namespace NUMINAMATH_CALUDE_quadrilateral_area_not_integer_l1755_175521

theorem quadrilateral_area_not_integer (n : ℕ) : 
  ¬ (∃ (m : ℕ), m^2 = n * (n + 1) * (n + 2) * (n + 3)) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_not_integer_l1755_175521


namespace NUMINAMATH_CALUDE_tenth_minus_ninth_square_diff_l1755_175502

/-- The number of tiles in the nth square of the sequence -/
def tiles_in_square (n : ℕ) : ℕ := n^2

/-- The theorem stating the difference in tiles between the 10th and 9th squares -/
theorem tenth_minus_ninth_square_diff : tiles_in_square 10 - tiles_in_square 9 = 19 := by
  sorry

end NUMINAMATH_CALUDE_tenth_minus_ninth_square_diff_l1755_175502


namespace NUMINAMATH_CALUDE_original_group_size_l1755_175539

/-- Given a group of men working on a project, this theorem proves that the original number of men is 30, based on the given conditions. -/
theorem original_group_size (initial_days work_days : ℕ) (absent_men : ℕ) : 
  initial_days = 10 → 
  work_days = 12 → 
  absent_men = 5 → 
  ∃ (original_size : ℕ), 
    original_size * initial_days = (original_size - absent_men) * work_days ∧ 
    original_size = 30 := by
  sorry

end NUMINAMATH_CALUDE_original_group_size_l1755_175539


namespace NUMINAMATH_CALUDE_basketball_lineup_count_l1755_175518

theorem basketball_lineup_count :
  let total_players : ℕ := 20
  let lineup_size : ℕ := 5
  let specific_role : ℕ := 1
  let interchangeable : ℕ := 4
  total_players.choose specific_role * (total_players - specific_role).choose interchangeable = 77520 :=
by sorry

end NUMINAMATH_CALUDE_basketball_lineup_count_l1755_175518


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l1755_175597

theorem restaurant_bill_calculation (appetizer_cost : ℝ) (entree_cost : ℝ) (num_entrees : ℕ) (tip_percentage : ℝ) : 
  appetizer_cost = 10 ∧ 
  entree_cost = 20 ∧ 
  num_entrees = 4 ∧ 
  tip_percentage = 0.2 → 
  appetizer_cost + (entree_cost * num_entrees) + (appetizer_cost + entree_cost * num_entrees) * tip_percentage = 108 := by
  sorry

#check restaurant_bill_calculation

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l1755_175597


namespace NUMINAMATH_CALUDE_union_of_sets_l1755_175545

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 4}
  let B : Set ℕ := {2, 4, 6}
  A ∪ B = {1, 2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l1755_175545


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_sum_90_l1755_175582

theorem largest_of_five_consecutive_sum_90 :
  ∀ n : ℕ, (n + (n+1) + (n+2) + (n+3) + (n+4) = 90) → (n+4 = 20) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_of_five_consecutive_sum_90_l1755_175582


namespace NUMINAMATH_CALUDE_abs_z_eq_sqrt_10_div_2_l1755_175528

theorem abs_z_eq_sqrt_10_div_2 (z : ℂ) (h : (1 - Complex.I) * z = 1 + 2 * Complex.I) :
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_eq_sqrt_10_div_2_l1755_175528


namespace NUMINAMATH_CALUDE_f_minus_g_equals_one_l1755_175520

-- Define f and g as functions from ℝ to ℝ
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g x = -g (-x)

-- State the theorem
theorem f_minus_g_equals_one 
  (h_even : is_even f) 
  (h_odd : is_odd g) 
  (h_sum : ∀ x, f x + g x = x^3 + x^2 + 1) : 
  f 1 - g 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_f_minus_g_equals_one_l1755_175520


namespace NUMINAMATH_CALUDE_product_sum_coefficients_l1755_175529

theorem product_sum_coefficients :
  ∀ (A B C D : ℝ), 
  (∀ x : ℝ, (2 * x^2 - 3 * x + 5) * (5 - 3 * x) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 8 :=
by sorry

end NUMINAMATH_CALUDE_product_sum_coefficients_l1755_175529


namespace NUMINAMATH_CALUDE_calculator_time_saved_l1755_175546

/-- Proves that using a calculator saves 60 minutes for a 20-problem assignment -/
theorem calculator_time_saved 
  (time_with_calculator : ℕ) 
  (time_without_calculator : ℕ) 
  (num_problems : ℕ) 
  (h1 : time_with_calculator = 2)
  (h2 : time_without_calculator = 5)
  (h3 : num_problems = 20) :
  (time_without_calculator - time_with_calculator) * num_problems = 60 :=
by sorry

end NUMINAMATH_CALUDE_calculator_time_saved_l1755_175546


namespace NUMINAMATH_CALUDE_polynomial_functional_equation_l1755_175589

theorem polynomial_functional_equation (p : ℝ → ℝ) (c : ℝ) :
  (∀ x, p (p x) = x * p x + c * x^2) →
  ((p = id ∧ c = 0) ∨ (∀ x, p x = -x ∧ c = -2)) :=
sorry

end NUMINAMATH_CALUDE_polynomial_functional_equation_l1755_175589


namespace NUMINAMATH_CALUDE_bryan_tshirt_count_l1755_175534

def total_cost : ℕ := 1500
def tshirt_cost : ℕ := 100
def pants_cost : ℕ := 250
def pants_count : ℕ := 4

theorem bryan_tshirt_count :
  (total_cost - pants_count * pants_cost) / tshirt_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_bryan_tshirt_count_l1755_175534


namespace NUMINAMATH_CALUDE_solve_equation_l1755_175590

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((3 / x) + 3) = 2 / 3 → x = -27 / 23 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1755_175590


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1755_175543

theorem complex_equation_solution (x y : ℝ) :
  (x * Complex.I + 2 = y - Complex.I) → (x - y = -3) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1755_175543


namespace NUMINAMATH_CALUDE_sandy_book_purchase_l1755_175551

/-- The number of books Sandy bought from the first shop -/
def books_first_shop : ℕ := 65

/-- The amount Sandy spent at the first shop -/
def amount_first_shop : ℕ := 1380

/-- The number of books Sandy bought from the second shop -/
def books_second_shop : ℕ := 55

/-- The amount Sandy spent at the second shop -/
def amount_second_shop : ℕ := 900

/-- The average price per book -/
def average_price : ℕ := 19

theorem sandy_book_purchase :
  books_first_shop = 65 ∧
  (amount_first_shop + amount_second_shop : ℚ) / (books_first_shop + books_second_shop) = average_price := by
  sorry

end NUMINAMATH_CALUDE_sandy_book_purchase_l1755_175551


namespace NUMINAMATH_CALUDE_range_of_m_l1755_175598

-- Define the curve C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -Real.sqrt (4 - p.2^2)}

-- Define the line l
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 6}

-- Define the point A
def A (m : ℝ) : ℝ × ℝ := (m, 0)

-- Theorem statement
theorem range_of_m (m : ℝ) :
  (∃ P ∈ C, ∃ Q ∈ l, A m + P - (A m + Q) = (0, 0)) →
  m ∈ Set.Icc 2 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1755_175598


namespace NUMINAMATH_CALUDE_expansion_terms_count_l1755_175562

/-- The number of terms in the expansion of a product of two sums -/
def num_terms_in_expansion (m n : ℕ) : ℕ := m * n

/-- Theorem: The expansion of (a+b+c+d)(e+f+g+h+i) has 20 terms -/
theorem expansion_terms_count : num_terms_in_expansion 4 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_count_l1755_175562


namespace NUMINAMATH_CALUDE_characterize_valid_common_differences_l1755_175580

/-- A number is interesting if 2018 divides its number of positive divisors -/
def IsInteresting (n : ℕ) : Prop :=
  2018 ∣ (Nat.divisors n).card

/-- An arithmetic progression with first term a and common difference k -/
def ArithmeticProgression (a k : ℕ) : ℕ → ℕ :=
  fun i => a + i * k

/-- The property of k being a valid common difference for an infinite
    arithmetic progression of interesting numbers -/
def IsValidCommonDifference (k : ℕ) : Prop :=
  ∃ a : ℕ, ∀ i : ℕ, IsInteresting (ArithmeticProgression a k i)

/-- The main theorem characterizing valid common differences -/
theorem characterize_valid_common_differences :
  ∀ k : ℕ, k > 0 →
  (IsValidCommonDifference k ↔
    (∃ (m : ℕ) (p : ℕ), m > 0 ∧ Nat.Prime p ∧ k = m * p^1009) ∧
    k ≠ 2^2009) :=
  sorry

end NUMINAMATH_CALUDE_characterize_valid_common_differences_l1755_175580


namespace NUMINAMATH_CALUDE_pool_filling_trips_l1755_175532

/-- The number of trips required to fill the pool -/
def trips_to_fill_pool (caleb_gallons cynthia_gallons pool_capacity : ℕ) : ℕ :=
  (pool_capacity + caleb_gallons + cynthia_gallons - 1) / (caleb_gallons + cynthia_gallons)

/-- Theorem stating that it takes 7 trips to fill the pool -/
theorem pool_filling_trips :
  trips_to_fill_pool 7 8 105 = 7 := by sorry

end NUMINAMATH_CALUDE_pool_filling_trips_l1755_175532


namespace NUMINAMATH_CALUDE_tiles_difference_l1755_175569

/-- The side length of the nth square in the sequence -/
def side_length (n : ℕ) : ℕ := 2 * n - 1

/-- The number of tiles in the nth square -/
def tiles_in_square (n : ℕ) : ℕ := (side_length n) ^ 2

/-- The theorem stating the difference in tiles between the 10th and 9th squares -/
theorem tiles_difference : tiles_in_square 10 - tiles_in_square 9 = 72 := by
  sorry

end NUMINAMATH_CALUDE_tiles_difference_l1755_175569


namespace NUMINAMATH_CALUDE_nine_valid_numbers_l1755_175556

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≥ 0 ∧ units ≤ 9

/-- Reverses the digits of a two-digit number -/
def reverse (n : TwoDigitNumber) : TwoDigitNumber :=
  ⟨n.units, n.tens, by sorry⟩

/-- Converts a TwoDigitNumber to a natural number -/
def to_nat (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

/-- Checks if a natural number is a positive perfect square -/
def is_positive_perfect_square (n : Nat) : Prop :=
  ∃ m : Nat, m > 0 ∧ m * m = n

/-- The main theorem to prove -/
theorem nine_valid_numbers :
  ∃ (S : Finset TwoDigitNumber),
    S.card = 9 ∧
    (∀ n : TwoDigitNumber, n ∈ S ↔
      is_positive_perfect_square (to_nat n - to_nat (reverse n))) ∧
    (∀ n : TwoDigitNumber,
      is_positive_perfect_square (to_nat n - to_nat (reverse n)) →
      n ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_nine_valid_numbers_l1755_175556


namespace NUMINAMATH_CALUDE_key_arrangement_count_l1755_175522

/-- The number of ways to arrange n distinct objects in a circular permutation -/
def circularPermutations (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of boxes -/
def numBoxes : ℕ := 6

theorem key_arrangement_count :
  circularPermutations numBoxes = 120 :=
by sorry

end NUMINAMATH_CALUDE_key_arrangement_count_l1755_175522


namespace NUMINAMATH_CALUDE_apple_picking_problem_l1755_175553

theorem apple_picking_problem (maggie_apples layla_apples average_apples : ℕ) 
  (h1 : maggie_apples = 40)
  (h2 : layla_apples = 22)
  (h3 : average_apples = 30)
  (h4 : (maggie_apples + layla_apples + kelsey_apples) / 3 = average_apples) :
  kelsey_apples = 28 := by
  sorry

end NUMINAMATH_CALUDE_apple_picking_problem_l1755_175553


namespace NUMINAMATH_CALUDE_short_trees_planted_calculation_park_short_trees_planted_l1755_175523

/-- The number of short trees planted in a park -/
def short_trees_planted (initial_short_trees final_short_trees : ℕ) : ℕ :=
  final_short_trees - initial_short_trees

/-- Theorem stating that the number of short trees planted is the difference between the final and initial number of short trees -/
theorem short_trees_planted_calculation (initial_short_trees final_short_trees : ℕ) 
  (h : final_short_trees ≥ initial_short_trees) :
  short_trees_planted initial_short_trees final_short_trees = final_short_trees - initial_short_trees :=
by
  sorry

/-- The specific case for the park problem -/
theorem park_short_trees_planted :
  short_trees_planted 41 98 = 57 :=
by
  sorry

end NUMINAMATH_CALUDE_short_trees_planted_calculation_park_short_trees_planted_l1755_175523


namespace NUMINAMATH_CALUDE_single_point_conic_section_l1755_175538

theorem single_point_conic_section (d : ℝ) : 
  (∃! p : ℝ × ℝ, 3 * p.1^2 + p.2^2 + 6 * p.1 - 8 * p.2 + d = 0) → d = 19 := by
  sorry

end NUMINAMATH_CALUDE_single_point_conic_section_l1755_175538


namespace NUMINAMATH_CALUDE_parabola_cross_section_l1755_175596

/-- Represents a cone --/
structure Cone where
  vertex_angle : ℝ

/-- Represents a cross-section of a cone --/
structure CrossSection where
  angle_with_axis : ℝ

/-- Represents the type of curve formed by a cross-section --/
inductive CurveType
  | Circle
  | Ellipse
  | Hyperbola
  | Parabola

/-- Determines the curve type of a cross-section for a given cone --/
def cross_section_curve_type (cone : Cone) (cs : CrossSection) : CurveType :=
  sorry

/-- Theorem stating that for a cone with 90° vertex angle and 45° cross-section angle, 
    the resulting curve is a parabola --/
theorem parabola_cross_section 
  (cone : Cone) 
  (cs : CrossSection) 
  (h1 : cone.vertex_angle = 90) 
  (h2 : cs.angle_with_axis = 45) : 
  cross_section_curve_type cone cs = CurveType.Parabola := by
  sorry

end NUMINAMATH_CALUDE_parabola_cross_section_l1755_175596


namespace NUMINAMATH_CALUDE_regular_polygon_sides_is_ten_l1755_175505

/-- The number of sides of a regular polygon with an interior angle of 144 degrees -/
def regular_polygon_sides : ℕ := by
  -- Define the interior angle
  let interior_angle : ℝ := 144

  -- Define the function for the sum of interior angles of an n-sided polygon
  let sum_of_angles (n : ℕ) : ℝ := 180 * (n - 2)

  -- Define the equation: sum of angles equals n times the interior angle
  let sides_equation (n : ℕ) : Prop := sum_of_angles n = n * interior_angle

  -- The number of sides is the solution to this equation
  exact sorry

theorem regular_polygon_sides_is_ten : regular_polygon_sides = 10 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_is_ten_l1755_175505


namespace NUMINAMATH_CALUDE_three_digit_numbers_with_repeated_digits_l1755_175573

theorem three_digit_numbers_with_repeated_digits : 
  let total_three_digit_numbers := 999 - 100 + 1
  let distinct_digit_numbers := 9 * 9 * 8
  total_three_digit_numbers - distinct_digit_numbers = 252 := by
sorry

end NUMINAMATH_CALUDE_three_digit_numbers_with_repeated_digits_l1755_175573


namespace NUMINAMATH_CALUDE_f_positive_implies_a_range_l1755_175535

open Real

/-- The function f(x) defined in terms of parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 + (a - 1) * x + 1

/-- Theorem stating that if f(x) > 0 for all real x, then 1 ≤ a < 5 -/
theorem f_positive_implies_a_range (a : ℝ) :
  (∀ x : ℝ, f a x > 0) → 1 ≤ a ∧ a < 5 := by
  sorry

end NUMINAMATH_CALUDE_f_positive_implies_a_range_l1755_175535


namespace NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l1755_175579

def B_current_age : ℕ := 34
def A_current_age : ℕ := B_current_age + 4

def A_future_age : ℕ := A_current_age + 10
def B_past_age : ℕ := B_current_age - 10

theorem age_ratio_is_two_to_one :
  A_future_age / B_past_age = 2 ∧ A_future_age % B_past_age = 0 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l1755_175579


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1755_175574

theorem rationalize_denominator : 
  (Real.sqrt 18 - Real.sqrt 8) / (Real.sqrt 8 + Real.sqrt 2) = (1 + Real.sqrt 2) / 3 := by
sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1755_175574


namespace NUMINAMATH_CALUDE_april_days_l1755_175563

/-- Proves the number of days in April based on Hannah's strawberry harvesting scenario -/
theorem april_days (daily_harvest : ℕ) (given_away : ℕ) (stolen : ℕ) (final_count : ℕ) :
  daily_harvest = 5 →
  given_away = 20 →
  stolen = 30 →
  final_count = 100 →
  (final_count + given_away + stolen) / daily_harvest = 30 := by
  sorry

#check april_days

end NUMINAMATH_CALUDE_april_days_l1755_175563


namespace NUMINAMATH_CALUDE_sixth_term_value_l1755_175500

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the properties of a₄ and a₈
def roots_property (a : ℕ → ℝ) : Prop :=
  a 4 ^ 2 - 34 * a 4 + 64 = 0 ∧ a 8 ^ 2 - 34 * a 8 + 64 = 0

-- Theorem statement
theorem sixth_term_value (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : roots_property a) : 
  a 6 = 8 := by sorry

end NUMINAMATH_CALUDE_sixth_term_value_l1755_175500


namespace NUMINAMATH_CALUDE_point_on_line_l1755_175560

/-- A point (x, y) lies on a line passing through (x1, y1) and (x2, y2) if and only if
    (y - y1) / (x - x1) = (y2 - y1) / (x2 - x1) -/
def on_line (x y x1 y1 x2 y2 : ℚ) : Prop :=
  (y - y1) * (x2 - x1) = (y2 - y1) * (x - x1)

theorem point_on_line :
  on_line (-2/3) (-1) 2 1 10 7 := by sorry

end NUMINAMATH_CALUDE_point_on_line_l1755_175560


namespace NUMINAMATH_CALUDE_estimate_fish_population_l1755_175576

/-- Estimates the total number of fish in a pond using the mark-recapture method. -/
theorem estimate_fish_population (initially_marked : ℕ) (second_catch : ℕ) (marked_in_second : ℕ) :
  initially_marked = 120 →
  second_catch = 100 →
  marked_in_second = 10 →
  (initially_marked * second_catch) / marked_in_second = 1200 :=
by
  sorry

#check estimate_fish_population

end NUMINAMATH_CALUDE_estimate_fish_population_l1755_175576


namespace NUMINAMATH_CALUDE_assignments_for_40_points_l1755_175524

/-- Calculates the number of assignments needed for a given number of points and assignments per point -/
def assignmentsForPoints (points : ℕ) (assignmentsPerPoint : ℕ) : ℕ :=
  points * assignmentsPerPoint

/-- Calculates the total number of assignments needed for 40 homework points -/
def totalAssignments : ℕ :=
  assignmentsForPoints 7 3 +  -- First 7 points
  assignmentsForPoints 7 4 +  -- Next 7 points (8-14)
  assignmentsForPoints 7 5 +  -- Next 7 points (15-21)
  assignmentsForPoints 7 6 +  -- Next 7 points (22-28)
  assignmentsForPoints 7 7 +  -- Next 7 points (29-35)
  assignmentsForPoints 5 8    -- Last 5 points (36-40)

/-- The theorem stating that 215 assignments are needed for 40 homework points -/
theorem assignments_for_40_points : totalAssignments = 215 := by
  sorry


end NUMINAMATH_CALUDE_assignments_for_40_points_l1755_175524


namespace NUMINAMATH_CALUDE_sample_size_is_192_l1755_175508

/-- Represents the total population in the school survey --/
def total_population : ℕ := 2400

/-- Represents the number of female students in the school --/
def female_students : ℕ := 1000

/-- Represents the number of female students in the sample --/
def female_sample : ℕ := 80

/-- Calculates the sample size based on the given information --/
def sample_size : ℕ := (total_population * female_sample) / female_students

/-- Theorem stating that the sample size is 192 --/
theorem sample_size_is_192 : sample_size = 192 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_is_192_l1755_175508


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l1755_175583

theorem point_in_third_quadrant : ∃ (x y : ℝ), 
  x = Real.sin (2014 * π / 180) ∧ 
  y = Real.cos (2014 * π / 180) ∧ 
  x < 0 ∧ y < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l1755_175583


namespace NUMINAMATH_CALUDE_point_symmetry_l1755_175585

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry about x-axis
def symmetricAboutXAxis (p1 p2 : Point2D) : Prop :=
  p1.x = p2.x ∧ p1.y = -p2.y

-- Define symmetry about y-axis
def symmetricAboutYAxis (p1 p2 : Point2D) : Prop :=
  p1.x = -p2.x ∧ p1.y = p2.y

theorem point_symmetry (M P N : Point2D) :
  symmetricAboutXAxis M P →
  symmetricAboutYAxis N M →
  N = Point2D.mk 1 2 →
  P = Point2D.mk (-1) (-2) := by
  sorry

end NUMINAMATH_CALUDE_point_symmetry_l1755_175585


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l1755_175541

/-- Given a parabola x² = 2py where p > 0, with a point M(4, y₀) on the parabola,
    and the distance between M and the focus F being |MF| = 5/4 * y₀,
    prove that the coordinates of the focus F are (0, 1). -/
theorem parabola_focus_coordinates (p : ℝ) (y₀ : ℝ) (h_p : p > 0) :
  x^2 = 2*p*y →
  4^2 = 2*p*y₀ →
  (4^2 + (y₀ - p/2)^2)^(1/2) = 5/4 * y₀ →
  (0, 1) = (0, p/2) := by sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l1755_175541


namespace NUMINAMATH_CALUDE_no_prime_solution_l1755_175525

/-- Represents a number in base p notation -/
def BaseP (coeffs : List Nat) (p : Nat) : Nat :=
  coeffs.enum.foldl (fun acc (i, a) => acc + a * p^i) 0

theorem no_prime_solution :
  ¬∃ (p : Nat), 
    Nat.Prime p ∧ 
    (BaseP [7, 1, 0, 2] p + BaseP [2, 0, 4] p + BaseP [4, 1, 1] p + 
     BaseP [0, 3, 2] p + BaseP [7] p = 
     BaseP [1, 0, 3] p + BaseP [2, 7, 4] p + BaseP [3, 1, 5] p) :=
by sorry

#eval BaseP [7, 1, 0, 2] 10  -- Should output 2017
#eval BaseP [2, 0, 4] 10     -- Should output 402
#eval BaseP [4, 1, 1] 10     -- Should output 114
#eval BaseP [0, 3, 2] 10     -- Should output 230
#eval BaseP [7] 10           -- Should output 7
#eval BaseP [1, 0, 3] 10     -- Should output 301
#eval BaseP [2, 7, 4] 10     -- Should output 472
#eval BaseP [3, 1, 5] 10     -- Should output 503

end NUMINAMATH_CALUDE_no_prime_solution_l1755_175525


namespace NUMINAMATH_CALUDE_unique_n_for_equation_l1755_175552

theorem unique_n_for_equation : ∃! (n : ℕ+), 
  ∃ (x y : ℕ+), y^2 + x*y + 3*x = n*(x^2 + x*y + 3*y) := by
  sorry

end NUMINAMATH_CALUDE_unique_n_for_equation_l1755_175552


namespace NUMINAMATH_CALUDE_al_original_amount_l1755_175554

/-- Represents the investment scenario with Al, Betty, and Clare --/
structure Investment where
  al : ℝ
  betty : ℝ
  clare : ℝ

/-- The conditions of the investment problem --/
def validInvestment (inv : Investment) : Prop :=
  inv.al + inv.betty + inv.clare = 1200 ∧
  (inv.al - 200) + (3 * inv.betty) + (4 * inv.clare) = 1800

/-- The theorem stating Al's original investment amount --/
theorem al_original_amount :
  ∀ inv : Investment, validInvestment inv → inv.al = 860 := by
  sorry

end NUMINAMATH_CALUDE_al_original_amount_l1755_175554


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l1755_175531

def circle_radius : ℝ := 6

theorem rectangle_longer_side (rectangle_area rectangle_shorter_side rectangle_longer_side : ℝ) : 
  rectangle_area = 3 * (π * circle_radius^2) →
  rectangle_shorter_side = 2 * circle_radius →
  rectangle_area = rectangle_shorter_side * rectangle_longer_side →
  rectangle_longer_side = 9 * π := by
sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l1755_175531


namespace NUMINAMATH_CALUDE_mitch_weekend_hours_l1755_175592

/-- Represents Mitch's work schedule and earnings --/
structure MitchWork where
  weekdayHours : ℕ  -- Hours worked per weekday
  weekdayRate : ℕ   -- Hourly rate for weekdays in dollars
  totalWeeklyEarnings : ℕ  -- Total weekly earnings in dollars
  weekendRate : ℕ   -- Hourly rate for weekends in dollars

/-- Calculates the number of weekend hours Mitch works --/
def weekendHours (m : MitchWork) : ℕ :=
  let weekdayEarnings := m.weekdayHours * 5 * m.weekdayRate
  let weekendEarnings := m.totalWeeklyEarnings - weekdayEarnings
  weekendEarnings / m.weekendRate

/-- Theorem stating that Mitch works 6 hours on weekends --/
theorem mitch_weekend_hours :
  ∀ (m : MitchWork),
  m.weekdayHours = 5 ∧
  m.weekdayRate = 3 ∧
  m.totalWeeklyEarnings = 111 ∧
  m.weekendRate = 6 →
  weekendHours m = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_mitch_weekend_hours_l1755_175592


namespace NUMINAMATH_CALUDE_eventually_linear_closed_under_addition_l1755_175517

theorem eventually_linear_closed_under_addition (S : Set ℕ) 
  (h_closed : ∀ a b : ℕ, a ∈ S → b ∈ S → (a + b) ∈ S) :
  ∃ k N : ℕ, ∀ n : ℕ, n > N → (n ∈ S ↔ k ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_eventually_linear_closed_under_addition_l1755_175517


namespace NUMINAMATH_CALUDE_sector_central_angle_l1755_175593

/-- The central angle of a circular sector, given its radius and area -/
theorem sector_central_angle (r : ℝ) (area : ℝ) (h1 : r = 2) (h2 : area = 4) :
  (2 * area) / (r ^ 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1755_175593


namespace NUMINAMATH_CALUDE_volume_of_cut_cube_l1755_175540

/-- Represents a three-dimensional solid --/
structure Solid :=
  (volume : ℝ)

/-- Represents a cube --/
def Cube (edge_length : ℝ) : Solid :=
  { volume := edge_length ^ 3 }

/-- Represents the result of cutting parts off a cube --/
def CutCube (c : Solid) (cut_volume : ℝ) : Solid :=
  { volume := c.volume - cut_volume }

/-- Theorem stating that the volume of the resulting solid is 9 --/
theorem volume_of_cut_cube : 
  ∃ (cut_volume : ℝ), 
    (CutCube (Cube 3) cut_volume).volume = 9 :=
sorry

end NUMINAMATH_CALUDE_volume_of_cut_cube_l1755_175540


namespace NUMINAMATH_CALUDE_triangle_base_length_l1755_175537

theorem triangle_base_length 
  (area : ℝ) 
  (height : ℝ) 
  (h1 : area = 10) 
  (h2 : height = 5) : 
  area = (height * 4) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l1755_175537


namespace NUMINAMATH_CALUDE_least_number_divisible_by_multiple_l1755_175504

theorem least_number_divisible_by_multiple (n : ℕ) : n = 856 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k₁ k₂ k₃ k₄ : ℕ, 
    (m + 8 = 24 * k₁) ∧ 
    (m + 8 = 32 * k₂) ∧ 
    (m + 8 = 36 * k₃) ∧ 
    (m + 8 = 54 * k₄))) ∧ 
  (∃ k₁ k₂ k₃ k₄ : ℕ, 
    (n + 8 = 24 * k₁) ∧ 
    (n + 8 = 32 * k₂) ∧ 
    (n + 8 = 36 * k₃) ∧ 
    (n + 8 = 54 * k₄)) := by
  sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_multiple_l1755_175504


namespace NUMINAMATH_CALUDE_right_triangle_cos_c_l1755_175565

theorem right_triangle_cos_c (A B C : ℝ) (h1 : A + B + C = Real.pi) 
  (h2 : A = Real.pi / 2) (h3 : Real.sin B = 3 / 5) : Real.cos C = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cos_c_l1755_175565


namespace NUMINAMATH_CALUDE_derricks_yard_length_l1755_175533

theorem derricks_yard_length (derrick_length alex_length brianne_length : ℝ) : 
  alex_length = derrick_length / 2 →
  brianne_length = 6 * alex_length →
  brianne_length = 30 →
  derrick_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_derricks_yard_length_l1755_175533


namespace NUMINAMATH_CALUDE_tiling_iff_div_four_l1755_175511

/-- A T-tetromino is a shape that covers exactly 4 squares. -/
def TTetromino : Type := Unit

/-- A tiling of an n×n board with T-tetrominos. -/
def Tiling (n : ℕ) : Type := 
  {arrangement : Fin n → Fin n → Option TTetromino // 
    ∀ (i j : Fin n), ∃ (t : TTetromino), arrangement i j = some t}

/-- The main theorem: An n×n board can be tiled with T-tetrominos iff n is divisible by 4. -/
theorem tiling_iff_div_four (n : ℕ) : 
  (∃ (t : Tiling n), True) ↔ 4 ∣ n := by sorry

end NUMINAMATH_CALUDE_tiling_iff_div_four_l1755_175511


namespace NUMINAMATH_CALUDE_initial_overs_calculation_l1755_175599

theorem initial_overs_calculation (target : ℝ) (initial_rate : ℝ) (remaining_rate : ℝ) 
  (remaining_overs : ℝ) (h1 : target = 250) (h2 : initial_rate = 4.2) 
  (h3 : remaining_rate = 5.533333333333333) (h4 : remaining_overs = 30) :
  ∃ x : ℝ, x = 20 ∧ initial_rate * x + remaining_rate * remaining_overs = target :=
by
  sorry

end NUMINAMATH_CALUDE_initial_overs_calculation_l1755_175599


namespace NUMINAMATH_CALUDE_sin_identity_l1755_175581

theorem sin_identity (α : Real) (h : Real.sin (α - π/4) = 1/2) :
  Real.sin (5*π/4 - α) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_identity_l1755_175581


namespace NUMINAMATH_CALUDE_cubic_function_properties_l1755_175571

/-- The cubic function f(x) = x³ + ax² + bx + c -/
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_function_properties :
  ∃ (a b c : ℝ),
    (∀ x, f' a b x = 0 → x = -1 ∨ x = 3) ∧
    (f a b c (-1) = 7) ∧
    (∀ x, f a b c x ≤ 7) ∧
    (∀ x, f a b c x ≥ f a b c 3) ∧
    (a = -3) ∧
    (b = -9) ∧
    (c = 2) ∧
    (f a b c 3 = -25) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l1755_175571


namespace NUMINAMATH_CALUDE_sqrt_three_expression_l1755_175515

theorem sqrt_three_expression : Real.sqrt 3 * (Real.sqrt 3 - 1 / Real.sqrt 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_expression_l1755_175515


namespace NUMINAMATH_CALUDE_smallest_c_for_inequality_l1755_175527

theorem smallest_c_for_inequality : ∃ c : ℕ, c = 9 ∧ (∀ k : ℕ, 27 ^ k > 3 ^ 24 → k ≥ c) := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_for_inequality_l1755_175527


namespace NUMINAMATH_CALUDE_complex_sum_pure_imaginary_l1755_175514

theorem complex_sum_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := a + 2*I
  let z₂ : ℂ := 3 - 4*I
  (z₁ + z₂).re = 0 → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_pure_imaginary_l1755_175514


namespace NUMINAMATH_CALUDE_fourth_term_max_coefficient_l1755_175594

def has_max_fourth_term (n : ℕ) : Prop :=
  ∀ k, 0 ≤ k ∧ k ≤ n → Nat.choose n 3 ≥ Nat.choose n k

theorem fourth_term_max_coefficient (n : ℕ) :
  has_max_fourth_term n ↔ n = 5 ∨ n = 6 ∨ n = 7 := by sorry

end NUMINAMATH_CALUDE_fourth_term_max_coefficient_l1755_175594


namespace NUMINAMATH_CALUDE_parabola_segment_length_l1755_175559

/-- The length of a segment AB on a parabola y = 4x² -/
theorem parabola_segment_length :
  ∀ (x₁ x₂ y₁ y₂ : ℝ),
  y₁ = 4 * x₁^2 →
  y₂ = 4 * x₂^2 →
  ∃ (k : ℝ),
  y₁ = k * x₁ + 1/16 →
  y₂ = k * x₂ + 1/16 →
  y₁ + y₂ = 2 →
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = (17/8)^2 :=
by
  sorry


end NUMINAMATH_CALUDE_parabola_segment_length_l1755_175559


namespace NUMINAMATH_CALUDE_tan_sum_of_roots_l1755_175512

theorem tan_sum_of_roots (α β : ℝ) : 
  (∃ x y : ℝ, x^2 - 3 * Real.sqrt 3 * x + 4 = 0 ∧ 
              y^2 - 3 * Real.sqrt 3 * y + 4 = 0 ∧ 
              x = Real.tan α ∧ 
              y = Real.tan β) → 
  Real.tan (α + β) = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_sum_of_roots_l1755_175512


namespace NUMINAMATH_CALUDE_triangle_area_is_13_5_l1755_175536

/-- The area of a triangular region bounded by the two coordinate axes and the line 3x + y = 9 -/
def triangleArea : ℝ := 13.5

/-- The equation of the line bounding the triangular region -/
def lineEquation (x y : ℝ) : Prop := 3 * x + y = 9

/-- Theorem stating that the area of the triangular region is 13.5 square units -/
theorem triangle_area_is_13_5 :
  triangleArea = 13.5 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_13_5_l1755_175536


namespace NUMINAMATH_CALUDE_angle_relations_l1755_175558

def acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2

theorem angle_relations (α β : ℝ) 
  (h_acute_α : acute_angle α) 
  (h_acute_β : acute_angle β) 
  (h_sin_α : Real.sin α = 3/5) 
  (h_tan_diff : Real.tan (α - β) = 1/3) : 
  Real.tan β = 1/3 ∧ 
  Real.sin (2*α - β) = (13 * Real.sqrt 10) / 50 := by
sorry

end NUMINAMATH_CALUDE_angle_relations_l1755_175558


namespace NUMINAMATH_CALUDE_octal_subtraction_l1755_175516

/-- Convert a base-8 number to base-10 --/
def octal_to_decimal (n : ℕ) : ℕ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 * 64 + d2 * 8 + d3

/-- Convert a base-10 number to base-8 --/
def decimal_to_octal (n : ℕ) : ℕ :=
  let d1 := n / 64
  let d2 := (n / 8) % 8
  let d3 := n % 8
  d1 * 100 + d2 * 10 + d3

theorem octal_subtraction :
  decimal_to_octal (octal_to_decimal 526 - octal_to_decimal 321) = 205 := by
  sorry

end NUMINAMATH_CALUDE_octal_subtraction_l1755_175516


namespace NUMINAMATH_CALUDE_calculate_expression_l1755_175509

theorem calculate_expression : ((18^18 / 18^17)^3 * 8^3) / 2^9 = 5832 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1755_175509


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1755_175572

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ → ℝ × ℝ := λ x ↦ (-2, x)
  ∀ x : ℝ, are_parallel a (b x) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1755_175572


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1755_175513

/-- A polynomial is exactly divisible by (x-1)^3 if and only if its coefficients satisfy specific conditions -/
theorem polynomial_divisibility (a b c : ℤ) : 
  (∃ q : Polynomial ℤ, x^4 + a*x^2 + b*x + c = (x - 1)^3 * q) ↔ 
  (a = -6 ∧ b = 8 ∧ c = -3) :=
sorry


end NUMINAMATH_CALUDE_polynomial_divisibility_l1755_175513


namespace NUMINAMATH_CALUDE_grasshopper_jump_distance_l1755_175564

theorem grasshopper_jump_distance 
  (frog_distance : ℕ → ℕ → ℕ) 
  (mouse_distance : ℕ → ℕ → ℕ) 
  (grasshopper_distance : ℕ → ℕ) :
  (∀ g f, frog_distance g f = g + 32) →
  (∀ m f, mouse_distance m f = f - 26) →
  mouse_distance 31 (frog_distance (grasshopper_distance 31) 31) = 31 →
  grasshopper_distance 31 = 25 := by
sorry

end NUMINAMATH_CALUDE_grasshopper_jump_distance_l1755_175564


namespace NUMINAMATH_CALUDE_power_of_power_l1755_175501

theorem power_of_power (a : ℝ) : (a^5)^3 = a^15 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1755_175501


namespace NUMINAMATH_CALUDE_karens_order_cost_l1755_175588

/-- The cost of Karen's fast-food order -/
def fast_food_order_cost (burger_price sandwich_price smoothie_price : ℕ) 
  (burger_quantity sandwich_quantity smoothie_quantity : ℕ) : ℕ :=
  burger_price * burger_quantity + 
  sandwich_price * sandwich_quantity + 
  smoothie_price * smoothie_quantity

/-- Theorem stating that Karen's fast-food order costs $17 -/
theorem karens_order_cost : 
  fast_food_order_cost 5 4 4 1 1 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_karens_order_cost_l1755_175588


namespace NUMINAMATH_CALUDE_move_right_example_l1755_175526

/-- Represents a point in 2D Cartesian coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Moves a point horizontally by a given distance -/
def moveRight (p : Point) (distance : ℝ) : Point :=
  { x := p.x + distance, y := p.y }

/-- The theorem stating that moving (-1, 3) by 5 units right results in (4, 3) -/
theorem move_right_example :
  let initial := Point.mk (-1) 3
  let final := moveRight initial 5
  final = Point.mk 4 3 := by sorry

end NUMINAMATH_CALUDE_move_right_example_l1755_175526


namespace NUMINAMATH_CALUDE_sport_water_amount_l1755_175578

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (cornSyrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standardRatio : DrinkRatio :=
  { flavoring := 1,
    cornSyrup := 12,
    water := 30 }

/-- The sport formulation ratio -/
def sportRatio : DrinkRatio :=
  { flavoring := standardRatio.flavoring,
    cornSyrup := standardRatio.cornSyrup / 3,
    water := standardRatio.water * 2 }

/-- Amount of corn syrup in the sport formulation (in ounces) -/
def sportCornSyrup : ℚ := 6

/-- Theorem: The amount of water in the sport formulation is 90 ounces -/
theorem sport_water_amount : 
  (sportRatio.water / sportRatio.cornSyrup) * sportCornSyrup = 90 := by
  sorry

end NUMINAMATH_CALUDE_sport_water_amount_l1755_175578


namespace NUMINAMATH_CALUDE_amount_ratio_l1755_175548

theorem amount_ratio (total : ℕ) (r_amount : ℕ) : 
  total = 7000 →
  r_amount = 2800 →
  (r_amount : ℚ) / ((total - r_amount) : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_amount_ratio_l1755_175548


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_range_l1755_175506

theorem quadratic_real_solutions_range (m : ℝ) :
  (∃ x : ℝ, m * x^2 + 2 * x + 1 = 0) ∧ (m ≠ 0) ↔ m ≤ 1 ∧ m ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_range_l1755_175506


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1755_175503

theorem cubic_equation_roots (p : ℝ) : 
  (p = 6 ∨ p = -6) → 
  ∃ x y : ℝ, x ≠ y ∧ y - x = 1 ∧ 
  x^3 - 7*x + p = 0 ∧ 
  y^3 - 7*y + p = 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l1755_175503


namespace NUMINAMATH_CALUDE_sphere_to_hemisphere_volume_ratio_l1755_175566

/-- The ratio of the volume of a sphere to the volume of a hemisphere -/
theorem sphere_to_hemisphere_volume_ratio 
  (r : ℝ) -- radius of the sphere
  (k : ℝ) -- material density coefficient of the hemisphere
  (h : k = 2/3) -- given condition for k
  : (4/3 * Real.pi * r^3) / (k * 1/2 * 4/3 * Real.pi * (3*r)^3) = 2/27 := by
  sorry

end NUMINAMATH_CALUDE_sphere_to_hemisphere_volume_ratio_l1755_175566


namespace NUMINAMATH_CALUDE_cubic_sum_identity_l1755_175575

theorem cubic_sum_identity (y : ℝ) (h : y^3 + 1/y^3 = 110) : y + 1/y = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_identity_l1755_175575


namespace NUMINAMATH_CALUDE_f_properties_f_50_l1755_175584

/-- A cubic polynomial function satisfying specific conditions -/
def f (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating the properties of the function f -/
theorem f_properties :
  f 0 = 1 ∧
  f 1 = 5 ∧
  f 2 = 13 ∧
  f 3 = 25 :=
sorry

/-- Theorem proving the value of f(50) -/
theorem f_50 : f 50 = 62676 :=
sorry

end NUMINAMATH_CALUDE_f_properties_f_50_l1755_175584


namespace NUMINAMATH_CALUDE_regression_lines_common_point_l1755_175544

/-- Represents a regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Represents a dataset with means -/
structure Dataset where
  x_mean : ℝ
  y_mean : ℝ

/-- Checks if a point is on a regression line -/
def point_on_line (line : RegressionLine) (x y : ℝ) : Prop :=
  y = line.slope * x + line.intercept

/-- Theorem: Two regression lines with the same dataset means have a common point -/
theorem regression_lines_common_point 
  (line1 line2 : RegressionLine) (data : Dataset) :
  point_on_line line1 data.x_mean data.y_mean →
  point_on_line line2 data.x_mean data.y_mean →
  ∃ (x y : ℝ), point_on_line line1 x y ∧ point_on_line line2 x y :=
sorry

end NUMINAMATH_CALUDE_regression_lines_common_point_l1755_175544


namespace NUMINAMATH_CALUDE_p_half_q_age_years_ago_l1755_175587

/-- The number of years ago when p was half of q in age -/
def years_ago : ℕ := 12

/-- The present age of p -/
def p_age : ℕ := 18

/-- The present age of q -/
def q_age : ℕ := 24

/-- Theorem: Given the conditions, prove that p was half of q in age 12 years ago -/
theorem p_half_q_age_years_ago :
  (p_age : ℚ) / (q_age : ℚ) = 3 / 4 ∧
  p_age + q_age = 42 ∧
  (p_age - years_ago : ℚ) = (q_age - years_ago : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_p_half_q_age_years_ago_l1755_175587


namespace NUMINAMATH_CALUDE_water_balloon_ratio_l1755_175577

/-- The number of water balloons each person has -/
structure WaterBalloons where
  cynthia : ℕ
  randy : ℕ
  janice : ℕ

/-- The conditions of the problem -/
def problem_conditions (wb : WaterBalloons) : Prop :=
  wb.cynthia = 12 ∧ wb.janice = 6 ∧ wb.randy = wb.janice / 2

/-- The theorem stating the ratio of Cynthia's to Randy's water balloons -/
theorem water_balloon_ratio (wb : WaterBalloons) 
  (h : problem_conditions wb) : wb.cynthia / wb.randy = 4 := by
  sorry

end NUMINAMATH_CALUDE_water_balloon_ratio_l1755_175577


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l1755_175550

theorem largest_angle_in_triangle (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = 45 →           -- One angle is 45°
  5 * b = 4 * c →    -- The other two angles are in the ratio 4:5
  max a (max b c) = 75 -- The largest angle is 75°
  := by sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l1755_175550


namespace NUMINAMATH_CALUDE_repeating_decimal_proof_l1755_175591

def repeating_decimal : ℚ := 78 / 99

theorem repeating_decimal_proof :
  repeating_decimal = 26 / 33 ∧
  26 + 33 = 59 := by
  sorry

#eval (Nat.gcd 78 99)  -- Expected output: 3
#eval (78 / 3)         -- Expected output: 26
#eval (99 / 3)         -- Expected output: 33

end NUMINAMATH_CALUDE_repeating_decimal_proof_l1755_175591


namespace NUMINAMATH_CALUDE_conference_games_count_l1755_175557

/-- Calculates the number of games in a sports conference season. -/
def conference_games (total_teams : ℕ) (division_size : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let teams_per_division := total_teams / 2
  let intra_games := teams_per_division * (division_size - 1) * intra_division_games
  let inter_games := total_teams * division_size * inter_division_games
  (intra_games + inter_games) / 2

/-- Theorem stating the number of games in the specific conference setup. -/
theorem conference_games_count : 
  conference_games 16 8 3 2 = 296 := by
  sorry

end NUMINAMATH_CALUDE_conference_games_count_l1755_175557


namespace NUMINAMATH_CALUDE_puzzles_sum_is_five_l1755_175567

def alphabet_value (n : ℕ) : ℤ :=
  match n % 8 with
  | 0 => 0
  | 1 => -1
  | 2 => 2
  | 3 => -1
  | 4 => 0
  | 5 => 1
  | 6 => -2
  | 7 => 1
  | _ => 0 -- This case should never occur, but Lean requires it for completeness

def letter_position (c : Char) : ℕ :=
  match c with
  | 'p' => 16
  | 'u' => 21
  | 'z' => 26
  | 'l' => 12
  | 'e' => 5
  | 's' => 19
  | _ => 0 -- Default case for other characters

theorem puzzles_sum_is_five :
  (alphabet_value (letter_position 'p') +
   alphabet_value (letter_position 'u') +
   alphabet_value (letter_position 'z') +
   alphabet_value (letter_position 'z') +
   alphabet_value (letter_position 'l') +
   alphabet_value (letter_position 'e') +
   alphabet_value (letter_position 's')) = 5 := by
  sorry

end NUMINAMATH_CALUDE_puzzles_sum_is_five_l1755_175567


namespace NUMINAMATH_CALUDE_profit_difference_l1755_175586

def business_problem (a b c : ℕ) (b_profit : ℕ) : Prop :=
  let total_capital := a + b + c
  let a_ratio := a * b_profit * 3 / b
  let c_ratio := c * b_profit * 3 / b
  c_ratio - a_ratio = 760

theorem profit_difference :
  business_problem 8000 10000 12000 1900 :=
sorry

end NUMINAMATH_CALUDE_profit_difference_l1755_175586


namespace NUMINAMATH_CALUDE_sally_grew_five_onions_l1755_175568

/-- The number of onions grown by Sara -/
def sara_onions : ℕ := 4

/-- The number of onions grown by Fred -/
def fred_onions : ℕ := 9

/-- The total number of onions grown -/
def total_onions : ℕ := 18

/-- The number of onions grown by Sally -/
def sally_onions : ℕ := total_onions - (sara_onions + fred_onions)

theorem sally_grew_five_onions : sally_onions = 5 := by
  sorry

end NUMINAMATH_CALUDE_sally_grew_five_onions_l1755_175568
