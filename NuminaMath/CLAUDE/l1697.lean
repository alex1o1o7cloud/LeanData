import Mathlib

namespace NUMINAMATH_CALUDE_right_triangle_sides_l1697_169726

theorem right_triangle_sides (x Δ : ℝ) (hx : x > 0) (hΔ : Δ > 0) :
  (x + 2*Δ)^2 = x^2 + (x + Δ)^2 ↔ x = (Δ*(-1 + 2*Real.sqrt 7))/2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l1697_169726


namespace NUMINAMATH_CALUDE_y_coord_comparison_l1697_169724

/-- Given two points on a line, prove that the y-coordinate of the left point is greater than the y-coordinate of the right point. -/
theorem y_coord_comparison (y₁ y₂ : ℝ) : 
  ((-4 : ℝ), y₁) ∈ {(x, y) | y = -1/2 * x + 2} →
  ((2 : ℝ), y₂) ∈ {(x, y) | y = -1/2 * x + 2} →
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y_coord_comparison_l1697_169724


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1697_169787

theorem rationalize_denominator : 
  Real.sqrt (5 / (2 + Real.sqrt 2)) = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1697_169787


namespace NUMINAMATH_CALUDE_stating_distinguishable_triangles_count_l1697_169767

/-- Represents the number of available colors for the triangles -/
def num_colors : ℕ := 8

/-- Represents the number of small triangles needed to construct a large triangle -/
def triangles_per_large : ℕ := 4

/-- 
Calculates the number of distinguishable large equilateral triangles that can be constructed
given the number of available colors and the number of small triangles per large triangle.
-/
def count_distinguishable_triangles (colors : ℕ) (triangles : ℕ) : ℕ :=
  colors * (colors - 1) * (colors - 2) * (colors - 3)

/-- 
Theorem stating that the number of distinguishable large equilateral triangles
that can be constructed under the given conditions is 1680.
-/
theorem distinguishable_triangles_count :
  count_distinguishable_triangles num_colors triangles_per_large = 1680 := by
  sorry


end NUMINAMATH_CALUDE_stating_distinguishable_triangles_count_l1697_169767


namespace NUMINAMATH_CALUDE_savings_for_three_shirts_l1697_169755

/-- The cost of a single item -/
def itemCost : ℕ := 10

/-- The discount percentage for the second item -/
def secondItemDiscount : ℚ := 1/2

/-- The discount percentage for the third item -/
def thirdItemDiscount : ℚ := 3/5

/-- Calculate the savings for a given number of items -/
def calculateSavings (n : ℕ) : ℚ :=
  if n ≤ 1 then 0
  else if n = 2 then secondItemDiscount * itemCost
  else secondItemDiscount * itemCost + thirdItemDiscount * itemCost

theorem savings_for_three_shirts :
  calculateSavings 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_savings_for_three_shirts_l1697_169755


namespace NUMINAMATH_CALUDE_circle_symmetry_tangent_length_l1697_169780

/-- Circle C with equation x^2 + y^2 + 2x - 4y + 3 = 0 -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 3 = 0

/-- Line of symmetry with equation 2ax + by + 6 = 0 -/
def SymmetryLine (a b x y : ℝ) : Prop := 2*a*x + b*y + 6 = 0

/-- Point (a, b) lies on the symmetry line -/
def PointOnSymmetryLine (a b : ℝ) : Prop := 2*a*a + b*b + 6 = 0

/-- Minimum length of tangent line segment from (a, b) to the circle -/
def MinTangentLength (a b : ℝ) : ℝ := 4

theorem circle_symmetry_tangent_length 
  (a b : ℝ) 
  (h1 : PointOnSymmetryLine a b) :
  MinTangentLength a b = 4 := by sorry

end NUMINAMATH_CALUDE_circle_symmetry_tangent_length_l1697_169780


namespace NUMINAMATH_CALUDE_quadratic_nature_l1697_169792

/-- A quadratic function g(x) with the condition c = a + b^2 -/
def g (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + (a + b^2)

theorem quadratic_nature (a b : ℝ) :
  (a < 0 → ∃ x₀, ∀ x, g a b x ≤ g a b x₀) ∧
  (a > 0 → ∃ x₀, ∀ x, g a b x ≥ g a b x₀) :=
sorry

end NUMINAMATH_CALUDE_quadratic_nature_l1697_169792


namespace NUMINAMATH_CALUDE_min_value_fraction_l1697_169795

theorem min_value_fraction (x : ℝ) (h : x > 5) : 
  x^2 / (x - 5) ≥ 20 ∧ ∃ y > 5, y^2 / (y - 5) = 20 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1697_169795


namespace NUMINAMATH_CALUDE_odd_digits_base7_528_l1697_169772

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of odd digits in the base-7 representation of 528 (base 10) is 4 -/
theorem odd_digits_base7_528 : countOddDigits (toBase7 528) = 4 := by
  sorry

end NUMINAMATH_CALUDE_odd_digits_base7_528_l1697_169772


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_lower_bound_l1697_169756

theorem sum_of_reciprocals_lower_bound (a b : ℝ) (h1 : a * b > 0) (h2 : a + b = 1) : 
  1 / a + 1 / b ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_lower_bound_l1697_169756


namespace NUMINAMATH_CALUDE_grasshoppers_on_plant_count_l1697_169712

def total_grasshoppers : ℕ := 31
def baby_grasshoppers_dozens : ℕ := 2

def grasshoppers_on_plant : ℕ := total_grasshoppers - (baby_grasshoppers_dozens * 12)

theorem grasshoppers_on_plant_count : grasshoppers_on_plant = 7 := by
  sorry

end NUMINAMATH_CALUDE_grasshoppers_on_plant_count_l1697_169712


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1697_169764

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Area formula
  b^2 / (3 * Real.sin B) = (1/2) * a * c * Real.sin B →
  -- Given condition
  6 * Real.cos A * Real.cos C = 1 →
  -- Given side length
  b = 3 →
  -- Conclusion
  B = π/3 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1697_169764


namespace NUMINAMATH_CALUDE_smallest_multiple_of_42_and_56_not_18_l1697_169798

theorem smallest_multiple_of_42_and_56_not_18 : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → (42 ∣ m.val ∧ 56 ∣ m.val) → 18 ∣ m.val) ∧ 
  42 ∣ n.val ∧ 56 ∣ n.val ∧ ¬(18 ∣ n.val) ∧ n.val = 168 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_42_and_56_not_18_l1697_169798


namespace NUMINAMATH_CALUDE_sales_percentage_other_l1697_169783

theorem sales_percentage_other (total_percentage : ℝ) (markers_percentage : ℝ) (notebooks_percentage : ℝ)
  (h1 : total_percentage = 100)
  (h2 : markers_percentage = 42)
  (h3 : notebooks_percentage = 22) :
  total_percentage - markers_percentage - notebooks_percentage = 36 := by
sorry

end NUMINAMATH_CALUDE_sales_percentage_other_l1697_169783


namespace NUMINAMATH_CALUDE_investment_sum_l1697_169760

theorem investment_sum (raghu_investment : ℝ) : raghu_investment = 2400 →
  let trishul_investment := raghu_investment * 0.9
  let vishal_investment := trishul_investment * 1.1
  raghu_investment + trishul_investment + vishal_investment = 6936 := by
sorry

end NUMINAMATH_CALUDE_investment_sum_l1697_169760


namespace NUMINAMATH_CALUDE_jungsoos_number_is_420_75_l1697_169732

/-- Jinho's number is defined as the sum of 1 multiplied by 4, 0.1 multiplied by 2, and 0.001 multiplied by 7 -/
def jinhos_number : ℝ := 1 * 4 + 0.1 * 2 + 0.001 * 7

/-- Younghee's number is defined as 100 multiplied by Jinho's number -/
def younghees_number : ℝ := 100 * jinhos_number

/-- Jungsoo's number is defined as Younghee's number plus 0.05 -/
def jungsoos_number : ℝ := younghees_number + 0.05

/-- Theorem stating that Jungsoo's number equals 420.75 -/
theorem jungsoos_number_is_420_75 : jungsoos_number = 420.75 := by sorry

end NUMINAMATH_CALUDE_jungsoos_number_is_420_75_l1697_169732


namespace NUMINAMATH_CALUDE_fraction_simplification_l1697_169778

theorem fraction_simplification (x y z : ℝ) (h : x + y + z = 3) :
  (x*y + y*z + z*x) / (x^2 + y^2 + z^2) = (x*y + y*z + z*x) / (9 - 2*(x*y + y*z + z*x)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1697_169778


namespace NUMINAMATH_CALUDE_valid_two_digit_numbers_l1697_169786

def digits : Set Nat := {1, 2, 3}

def is_two_digit (n : Nat) : Prop :=
  10 ≤ n ∧ n < 100

def has_no_repeated_digits (n : Nat) : Prop :=
  let tens := n / 10
  let ones := n % 10
  tens ≠ ones

def is_valid_number (n : Nat) : Prop :=
  is_two_digit n ∧
  has_no_repeated_digits n ∧
  (n / 10 ∈ digits) ∧
  (n % 10 ∈ digits)

theorem valid_two_digit_numbers :
  {n : Nat | is_valid_number n} = {12, 13, 21, 23, 31, 32} := by
  sorry

end NUMINAMATH_CALUDE_valid_two_digit_numbers_l1697_169786


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1697_169736

theorem square_plus_reciprocal_square (x : ℝ) (h : x ≠ 0) :
  x^2 + 1/x^2 = 7 → x^4 + 1/x^4 = 47 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1697_169736


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l1697_169704

theorem gcd_of_specific_numbers : Nat.gcd 55555555 111111111 = 11111111 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l1697_169704


namespace NUMINAMATH_CALUDE_even_function_implies_a_eq_two_l1697_169766

/-- A function f is even if f(-x) = f(x) for all x in ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = (x+a)(x-2) -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ (x + a) * (x - 2)

/-- If f(x) = (x+a)(x-2) is an even function, then a = 2 -/
theorem even_function_implies_a_eq_two :
  ∀ a : ℝ, IsEven (f a) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_eq_two_l1697_169766


namespace NUMINAMATH_CALUDE_factorial_different_remainders_l1697_169707

theorem factorial_different_remainders (n : ℕ) : n ≥ 2 →
  (∀ i j, 1 ≤ i ∧ i < j ∧ j < n → Nat.factorial i % n ≠ Nat.factorial j % n) ↔ n = 2 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_factorial_different_remainders_l1697_169707


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1697_169752

/-- The number of games played in a chess tournament with n participants,
    where each participant plays exactly one game with each of the others. -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 19 participants, where each participant
    plays exactly one game with each of the remaining participants,
    the total number of games played is 171. -/
theorem chess_tournament_games :
  tournament_games 19 = 171 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l1697_169752


namespace NUMINAMATH_CALUDE_second_frog_hops_second_frog_hops_proof_l1697_169749

/-- Given three frogs hopping across a road, prove the number of hops taken by the second frog -/
theorem second_frog_hops : ℕ → ℕ → ℕ → Prop :=
  fun frog1 frog2 frog3 =>
    frog1 = 4 * frog2 ∧            -- First frog takes 4 times as many hops as the second
    frog2 = 2 * frog3 ∧            -- Second frog takes twice as many hops as the third
    frog1 + frog2 + frog3 = 99 →   -- Total hops is 99
    frog2 = 18                     -- Second frog takes 18 hops

theorem second_frog_hops_proof : ∃ (frog1 frog2 frog3 : ℕ), second_frog_hops frog1 frog2 frog3 := by
  sorry

end NUMINAMATH_CALUDE_second_frog_hops_second_frog_hops_proof_l1697_169749


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_range_is_zero_l1697_169754

def integers_range : List Int := List.range 11 |>.map (λ x => x - 5)

theorem arithmetic_mean_of_range_is_zero :
  (integers_range.sum : ℚ) / integers_range.length = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_range_is_zero_l1697_169754


namespace NUMINAMATH_CALUDE_rolling_cube_dot_path_length_l1697_169719

/-- The path length of a dot on a rolling cube -/
theorem rolling_cube_dot_path_length :
  let cube_side : ℝ := 2
  let dot_distance : ℝ := 2 / 3
  let path_length : ℝ := (4 * Real.pi * Real.sqrt 10) / 3
  cube_side > 0 ∧ 0 < dot_distance ∧ dot_distance < cube_side →
  path_length = 4 * (Real.pi * Real.sqrt (dot_distance^2 + cube_side^2)) / 2 :=
by sorry


end NUMINAMATH_CALUDE_rolling_cube_dot_path_length_l1697_169719


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1697_169761

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = Complex.I) : 
  z.im = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1697_169761


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1697_169751

theorem quadratic_equation_solution :
  ∃! y : ℝ, y^2 + 6*y + 8 = -(y + 2)*(y + 6) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1697_169751


namespace NUMINAMATH_CALUDE_binomial_inequalities_l1697_169753

theorem binomial_inequalities (n : ℕ) (h : n ≥ 2) :
  (2 : ℝ)^n < (Nat.choose (2*n) n : ℝ) ∧
  (Nat.choose (2*n) n : ℝ) < 4^n ∧
  (Nat.choose (2*n - 1) n : ℝ) < 4^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_binomial_inequalities_l1697_169753


namespace NUMINAMATH_CALUDE_monomial_exponents_l1697_169788

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def are_like_terms (m1 m2 : ℕ → ℕ) : Prop :=
  ∀ i, m1 i = m2 i

theorem monomial_exponents (a b : ℕ) :
  are_like_terms (fun i => if i = 0 then a + 1 else if i = 1 then 3 else 0)
                 (fun i => if i = 0 then 2 else if i = 1 then b else 0) →
  a = 1 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_monomial_exponents_l1697_169788


namespace NUMINAMATH_CALUDE_largest_k_for_g_range_three_l1697_169743

/-- The function g(x) = 2x^2 - 8x + k -/
def g (k : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 8 * x + k

/-- Theorem stating that 11 is the largest value of k such that 3 is in the range of g(x) -/
theorem largest_k_for_g_range_three :
  ∀ k : ℝ, (∃ x : ℝ, g k x = 3) ↔ k ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_largest_k_for_g_range_three_l1697_169743


namespace NUMINAMATH_CALUDE_interval_intersection_l1697_169769

theorem interval_intersection (x : ℝ) : 
  (2 < 3*x ∧ 3*x < 3) ∧ (2 < 4*x ∧ 4*x < 3) ↔ (2/3 < x ∧ x < 3/4) :=
by sorry

end NUMINAMATH_CALUDE_interval_intersection_l1697_169769


namespace NUMINAMATH_CALUDE_lines_intersect_at_point_l1697_169758

def line1 (t : ℚ) : ℚ × ℚ := (2 + 3*t, 2 - 4*t)
def line2 (u : ℚ) : ℚ × ℚ := (4 + 5*u, -8 + 3*u)

def intersection_point : ℚ × ℚ := (-123/141, 454/141)

theorem lines_intersect_at_point :
  ∃ (t u : ℚ), line1 t = line2 u ∧ line1 t = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_lines_intersect_at_point_l1697_169758


namespace NUMINAMATH_CALUDE_complex_magnitude_l1697_169728

theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1697_169728


namespace NUMINAMATH_CALUDE_dance_partners_exist_l1697_169765

variable {Boys Girls : Type}
variable (danced : Boys → Girls → Prop)

theorem dance_partners_exist
  (h1 : ∀ b : Boys, ∃ g : Girls, ¬danced b g)
  (h2 : ∀ g : Girls, ∃ b : Boys, danced b g) :
  ∃ (g g' : Boys) (f f' : Girls),
    danced g f ∧ ¬danced g f' ∧ danced g' f' ∧ ¬danced g' f :=
by sorry

end NUMINAMATH_CALUDE_dance_partners_exist_l1697_169765


namespace NUMINAMATH_CALUDE_no_real_solution_l1697_169727

theorem no_real_solution (a : ℝ) (ha : a > 0) (h : a^3 = 6*(a + 1)) :
  ∀ x : ℝ, x^2 + a*x + a^2 - 6 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_solution_l1697_169727


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_x_l1697_169781

theorem x_gt_one_sufficient_not_necessary_for_x_squared_gt_x :
  (∃ x : ℝ, x > 1 ∧ x^2 > x) ∧
  (∃ x : ℝ, x^2 > x ∧ ¬(x > 1)) ∧
  (∀ x : ℝ, x > 1 → x^2 > x) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_x_l1697_169781


namespace NUMINAMATH_CALUDE_square_circle_area_l1697_169713

theorem square_circle_area (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  let r := d / 2
  s^2 + π * r^2 = 144 + 72 * π := by sorry

end NUMINAMATH_CALUDE_square_circle_area_l1697_169713


namespace NUMINAMATH_CALUDE_max_value_implies_ratio_l1697_169750

/-- Given a function f(x) = x^3 + ax^2 + bx - a^2 - 7a that reaches its maximum value of 10 at x = 1,
    prove that a/b = -2/3 -/
theorem max_value_implies_ratio (a b : ℝ) :
  let f := fun (x : ℝ) => x^3 + a*x^2 + b*x - a^2 - 7*a
  (∀ x, f x ≤ f 1) ∧ (f 1 = 10) → a/b = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_ratio_l1697_169750


namespace NUMINAMATH_CALUDE_smallest_total_books_l1697_169716

/-- Represents the number of books in the library -/
structure LibraryBooks where
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Checks if the given LibraryBooks satisfies the required ratios -/
def satisfiesRatios (books : LibraryBooks) : Prop :=
  3 * books.chemistry = 2 * books.physics ∧ 
  4 * books.biology = 3 * books.chemistry

/-- Calculates the total number of books -/
def totalBooks (books : LibraryBooks) : ℕ :=
  books.physics + books.chemistry + books.biology

/-- Theorem: The smallest possible total number of books satisfying the conditions is 3003 -/
theorem smallest_total_books : 
  ∃ (books : LibraryBooks), 
    satisfiesRatios books ∧ 
    totalBooks books > 3000 ∧
    totalBooks books = 3003 ∧
    ∀ (other : LibraryBooks), 
      satisfiesRatios other → 
      totalBooks other > 3000 → 
      totalBooks other ≥ totalBooks books := by
  sorry

end NUMINAMATH_CALUDE_smallest_total_books_l1697_169716


namespace NUMINAMATH_CALUDE_extreme_points_cubic_l1697_169763

/-- Given a cubic function f(x) = x³ + ax² + bx with extreme points at x = -2 and x = 4,
    prove that a - b = 21. -/
theorem extreme_points_cubic (a b : ℝ) : 
  (∀ x : ℝ, (x = -2 ∨ x = 4) → (3 * x^2 + 2 * a * x + b = 0)) → 
  a - b = 21 := by
  sorry

end NUMINAMATH_CALUDE_extreme_points_cubic_l1697_169763


namespace NUMINAMATH_CALUDE_power_product_equals_sum_l1697_169715

theorem power_product_equals_sum (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_sum_l1697_169715


namespace NUMINAMATH_CALUDE_ellipse_inequality_l1697_169789

/-- An ellipse with equation ax^2 + by^2 = 1 and foci on the x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  is_ellipse : a > 0 ∧ b > 0
  foci_on_x_axis : True  -- We can't directly express this condition in Lean, so we use True as a placeholder

/-- Theorem: For an ellipse ax^2 + by^2 = 1 with foci on the x-axis, 0 < a < b -/
theorem ellipse_inequality (e : Ellipse) : 0 < e.a ∧ e.a < e.b := by
  sorry

end NUMINAMATH_CALUDE_ellipse_inequality_l1697_169789


namespace NUMINAMATH_CALUDE_ratio_problem_l1697_169722

theorem ratio_problem (x y : ℝ) (h : (3 * x^2 - y) / (x + y) = 1/2) :
  x / y = 3 / (6 * x - 1) := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l1697_169722


namespace NUMINAMATH_CALUDE_hyperbola_axis_ratio_l1697_169782

/-- Given a hyperbola with equation mx^2 + y^2 = 1, if its conjugate axis is twice the length
of its transverse axis, then m = -1/4 -/
theorem hyperbola_axis_ratio (m : ℝ) : 
  (∀ x y : ℝ, m * x^2 + y^2 = 1) →  -- Equation of the hyperbola
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧       -- Existence of positive a and b
    (∀ x y : ℝ, y^2 / b^2 - x^2 / a^2 = 1) ∧  -- Standard form of hyperbola
    2 * b = 2 * a) →                -- Conjugate axis is twice the transverse axis
  m = -1/4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_axis_ratio_l1697_169782


namespace NUMINAMATH_CALUDE_repeating_block_length_seven_thirteenths_l1697_169768

/-- The length of the smallest repeating block in the decimal expansion of 7/13 is 6. -/
theorem repeating_block_length_seven_thirteenths : 
  ∃ (d : ℕ) (n : ℕ), d = 6 ∧ 7 * (10^d - 1) = 13 * n :=
by sorry

end NUMINAMATH_CALUDE_repeating_block_length_seven_thirteenths_l1697_169768


namespace NUMINAMATH_CALUDE_linear_function_increasing_y_l1697_169702

theorem linear_function_increasing_y (k b y₁ y₂ : ℝ) :
  k > 0 →
  y₁ = k * (-1) - b →
  y₂ = k * 2 - b →
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_linear_function_increasing_y_l1697_169702


namespace NUMINAMATH_CALUDE_common_difference_is_five_l1697_169771

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_is_five 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum1 : a 2 + a 6 = 8) 
  (h_sum2 : a 3 + a 4 = 3) : 
  ∃ d : ℝ, d = 5 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_common_difference_is_five_l1697_169771


namespace NUMINAMATH_CALUDE_quadratic_root_coefficients_l1697_169710

theorem quadratic_root_coefficients :
  ∀ (r s : ℝ),
  (∃ x : ℂ, 3 * x^2 + r * x + s = 0 ∧ x = 2 + Complex.I * Real.sqrt 3) →
  r = -12 ∧ s = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_coefficients_l1697_169710


namespace NUMINAMATH_CALUDE_percentage_failed_english_l1697_169705

theorem percentage_failed_english (total : ℝ) (failed_hindi : ℝ) (failed_both : ℝ) (passed_both : ℝ) 
  (h_total : total = 100)
  (h_failed_hindi : failed_hindi = 32)
  (h_failed_both : failed_both = 12)
  (h_passed_both : passed_both = 24) :
  ∃ failed_english : ℝ, failed_english = 56 ∧ 
  total - (failed_hindi + failed_english - failed_both) = passed_both :=
by sorry

end NUMINAMATH_CALUDE_percentage_failed_english_l1697_169705


namespace NUMINAMATH_CALUDE_shortest_side_length_l1697_169723

/-- A triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of one side of the triangle -/
  side : ℝ
  /-- The length of the first segment of the side divided by the point of tangency -/
  segment1 : ℝ
  /-- The length of the second segment of the side divided by the point of tangency -/
  segment2 : ℝ
  /-- The condition that the segments add up to the side length -/
  side_condition : side = segment1 + segment2

/-- The theorem stating the length of the shortest side of the triangle -/
theorem shortest_side_length (t : InscribedCircleTriangle) 
  (h1 : t.r = 5)
  (h2 : t.segment1 = 7)
  (h3 : t.segment2 = 9) :
  ∃ (shortest_side : ℝ), 
    shortest_side = 10 ∧ 
    (∀ (other_side : ℝ), other_side = t.side ∨ other_side ≥ shortest_side) :=
sorry

end NUMINAMATH_CALUDE_shortest_side_length_l1697_169723


namespace NUMINAMATH_CALUDE_wildflower_color_difference_l1697_169748

/-- Given the following conditions about wildflowers:
  * The total number of wildflowers is 44
  * There are 13 yellow and white flowers
  * There are 17 red and yellow flowers
  * There are 14 red and white flowers

  Prove that the number of flowers containing red minus
  the number of flowers containing white equals 4.
-/
theorem wildflower_color_difference
  (total : ℕ)
  (yellow_white : ℕ)
  (red_yellow : ℕ)
  (red_white : ℕ)
  (h_total : total = 44)
  (h_yellow_white : yellow_white = 13)
  (h_red_yellow : red_yellow = 17)
  (h_red_white : red_white = 14) :
  (red_yellow + red_white) - (yellow_white + red_white) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_wildflower_color_difference_l1697_169748


namespace NUMINAMATH_CALUDE_beef_cost_theorem_l1697_169747

/-- Calculates the total cost of beef given the number of packs, pounds per pack, and price per pound. -/
def total_cost (num_packs : ℕ) (pounds_per_pack : ℕ) (price_per_pound : ℚ) : ℚ :=
  (num_packs * pounds_per_pack : ℚ) * price_per_pound

/-- Proves that given 5 packs of beef, 4 pounds per pack, and a price of $5.50 per pound, the total cost is $110. -/
theorem beef_cost_theorem :
  total_cost 5 4 (11/2) = 110 := by
  sorry

end NUMINAMATH_CALUDE_beef_cost_theorem_l1697_169747


namespace NUMINAMATH_CALUDE_inverse_proportionality_fraction_l1697_169742

theorem inverse_proportionality_fraction (k : ℝ) (x y : ℝ) (h : x > 0 ∧ y > 0) :
  (k = x * y) → (∃ c : ℝ, c > 0 ∧ y = c / x) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportionality_fraction_l1697_169742


namespace NUMINAMATH_CALUDE_range_of_a_l1697_169796

theorem range_of_a (a : ℝ) : 
  (∀ (x : ℝ), x > 3 → x > a) ↔ a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1697_169796


namespace NUMINAMATH_CALUDE_tan_x_equals_sqrt_three_l1697_169739

theorem tan_x_equals_sqrt_three (x : ℝ) 
  (h : Real.sin (x + Real.pi / 9) = Real.cos (x + Real.pi / 18) + Real.cos (x - Real.pi / 18)) : 
  Real.tan x = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_x_equals_sqrt_three_l1697_169739


namespace NUMINAMATH_CALUDE_motion_equation_l1697_169773

theorem motion_equation (g a V V₀ S t : ℝ) 
  (hV : V = (g + a) * t + V₀)
  (hS : S = (1/2) * (g + a) * t^2 + V₀ * t) :
  t = 2 * S / (V + V₀) := by
  sorry

end NUMINAMATH_CALUDE_motion_equation_l1697_169773


namespace NUMINAMATH_CALUDE_marge_personal_spending_l1697_169797

/-- Calculates Marge's personal spending amount after one year --/
def personal_spending_after_one_year (
  lottery_winnings : ℝ)
  (tax_rate : ℝ)
  (mortgage_rate : ℝ)
  (retirement_rate : ℝ)
  (retirement_interest : ℝ)
  (college_rate : ℝ)
  (savings : ℝ)
  (stock_investment_rate : ℝ)
  (stock_return : ℝ) : ℝ :=
  let after_tax := lottery_winnings * (1 - tax_rate)
  let after_mortgage := after_tax * (1 - mortgage_rate)
  let after_retirement := after_mortgage * (1 - retirement_rate)
  let after_college := after_retirement * (1 - college_rate)
  let retirement_growth := after_mortgage * retirement_rate * retirement_interest
  let stock_investment := savings * stock_investment_rate
  let stock_growth := stock_investment * stock_return
  after_college + (savings - stock_investment) + retirement_growth + stock_growth

/-- Theorem stating that Marge's personal spending after one year is $5,363 --/
theorem marge_personal_spending :
  personal_spending_after_one_year 50000 0.6 0.5 0.4 0.05 0.25 1500 0.6 0.07 = 5363 := by
  sorry

end NUMINAMATH_CALUDE_marge_personal_spending_l1697_169797


namespace NUMINAMATH_CALUDE_sqrt_x_plus_one_real_l1697_169718

theorem sqrt_x_plus_one_real (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_one_real_l1697_169718


namespace NUMINAMATH_CALUDE_quadratic_minimum_values_l1697_169738

/-- A quadratic function f(x) = mx^2 - 2mx + 2 with m ≠ 0 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * m * x + 2

/-- The theorem stating the conditions and conclusion -/
theorem quadratic_minimum_values (m : ℝ) :
  m ≠ 0 →
  (∀ x, -2 ≤ x → x < 2 → f m x ≥ -2) →
  (∃ x, -2 ≤ x ∧ x < 2 ∧ f m x = -2) →
  (m = 4 ∨ m = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_values_l1697_169738


namespace NUMINAMATH_CALUDE_green_eyed_brunettes_l1697_169746

theorem green_eyed_brunettes (total : ℕ) (blueEyedBlondes : ℕ) (brunettes : ℕ) (greenEyed : ℕ) :
  total = 60 →
  blueEyedBlondes = 20 →
  brunettes = 35 →
  greenEyed = 25 →
  ∃ (greenEyedBrunettes : ℕ),
    greenEyedBrunettes = 10 ∧
    greenEyedBrunettes ≤ brunettes ∧
    greenEyedBrunettes ≤ greenEyed ∧
    blueEyedBlondes + (brunettes - greenEyedBrunettes) + greenEyed = total :=
by
  sorry

end NUMINAMATH_CALUDE_green_eyed_brunettes_l1697_169746


namespace NUMINAMATH_CALUDE_total_marbles_count_l1697_169729

/-- Represents the number of marbles of each color in the container -/
structure MarbleCount where
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- The ratio of marbles in the container -/
def marbleRatio : MarbleCount := ⟨2, 3, 4⟩

/-- The actual number of yellow marbles in the container -/
def yellowMarbleCount : ℕ := 40

/-- Theorem stating that the total number of marbles is 90 -/
theorem total_marbles_count :
  let total := marbleRatio.red + marbleRatio.blue + marbleRatio.yellow
  (yellowMarbleCount * total) / marbleRatio.yellow = 90 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_count_l1697_169729


namespace NUMINAMATH_CALUDE_opposite_of_83_is_84_l1697_169711

/-- Represents a circle with 100 equally spaced points numbered from 1 to 100. -/
structure NumberedCircle where
  numbers : Fin 100 → Fin 100
  bijective : Function.Bijective numbers

/-- Checks if numbers less than k are evenly distributed on both sides of the diameter through k. -/
def evenlyDistributed (c : NumberedCircle) (k : Fin 100) : Prop :=
  ∀ m < k, (c.numbers m < k ∧ c.numbers (m + 50) ≥ k) ∨
           (c.numbers m ≥ k ∧ c.numbers (m + 50) < k)

/-- The main theorem stating that if numbers are evenly distributed for all k,
    then the number opposite to 83 is 84. -/
theorem opposite_of_83_is_84 (c : NumberedCircle) 
  (h : ∀ k, evenlyDistributed c k) : 
  c.numbers (Fin.ofNat 33) = Fin.ofNat 84 :=
sorry

end NUMINAMATH_CALUDE_opposite_of_83_is_84_l1697_169711


namespace NUMINAMATH_CALUDE_A_enumeration_l1697_169790

def A : Set ℤ := {y | ∃ x : ℕ, y = 6 / (x - 2) ∧ 6 % (x - 2) = 0}

theorem A_enumeration : A = {-3, -6, 6, 3, 2, 1} := by
  sorry

end NUMINAMATH_CALUDE_A_enumeration_l1697_169790


namespace NUMINAMATH_CALUDE_sin_10_50_70_equals_one_eighth_l1697_169703

theorem sin_10_50_70_equals_one_eighth :
  Real.sin (10 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_10_50_70_equals_one_eighth_l1697_169703


namespace NUMINAMATH_CALUDE_subset_implication_l1697_169776

theorem subset_implication (A B : Set ℕ) (a : ℕ) :
  A = {1, a} ∧ B = {1, 2, 3} →
  (a = 3 → A ⊆ B) := by
  sorry

end NUMINAMATH_CALUDE_subset_implication_l1697_169776


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1697_169785

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 14 → b = 48 → c^2 = a^2 + b^2 → c = 50 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1697_169785


namespace NUMINAMATH_CALUDE_icosahedron_coloring_count_l1697_169700

/-- The number of faces in a regular icosahedron -/
def num_faces : ℕ := 20

/-- The number of colors available -/
def num_colors : ℕ := 10

/-- The order of the rotational symmetry group of a regular icosahedron -/
def icosahedron_symmetry_order : ℕ := 60

/-- The number of rotations around an axis through opposite faces -/
def rotations_per_axis : ℕ := 5

theorem icosahedron_coloring_count :
  (Nat.factorial (num_colors - 1)) / rotations_per_axis =
  72576 := by sorry

end NUMINAMATH_CALUDE_icosahedron_coloring_count_l1697_169700


namespace NUMINAMATH_CALUDE_nails_per_plank_l1697_169793

/-- Given that John uses 11 nails in total, with 8 additional nails, and needs 1 plank,
    prove that each plank requires 3 nails to be secured. -/
theorem nails_per_plank (total_nails : ℕ) (additional_nails : ℕ) (num_planks : ℕ)
  (h1 : total_nails = 11)
  (h2 : additional_nails = 8)
  (h3 : num_planks = 1) :
  total_nails - additional_nails = 3 := by
sorry

end NUMINAMATH_CALUDE_nails_per_plank_l1697_169793


namespace NUMINAMATH_CALUDE_equation_holds_for_three_l1697_169734

-- Define the equation we want to prove
def equation (n : ℕ) : Prop :=
  (((17 * Real.sqrt 5 + 38) ^ (1 / n : ℝ)) + ((17 * Real.sqrt 5 - 38) ^ (1 / n : ℝ))) = 2 * Real.sqrt 5

-- Theorem statement
theorem equation_holds_for_three : 
  equation 3 := by sorry

end NUMINAMATH_CALUDE_equation_holds_for_three_l1697_169734


namespace NUMINAMATH_CALUDE_intersection_when_m_is_one_intersection_equals_A_iff_l1697_169720

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | (x - 2*m + 1)*(x - m + 2) < 0}
def B : Set ℝ := {x | 1 ≤ x + 1 ∧ x + 1 ≤ 4}

-- Theorem 1: When m = 1, A ∩ B = {x | 0 ≤ x < 1}
theorem intersection_when_m_is_one :
  A 1 ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

-- Theorem 2: A ∩ B = A if and only if m ∈ {-1, 2}
theorem intersection_equals_A_iff :
  ∀ m : ℝ, A m ∩ B = A m ↔ m = -1 ∨ m = 2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_one_intersection_equals_A_iff_l1697_169720


namespace NUMINAMATH_CALUDE_multiplication_error_l1697_169730

theorem multiplication_error (a b : ℕ) (h1 : 100 ≤ a ∧ a < 1000) (h2 : 100 ≤ b ∧ b < 1000) :
  (∃ n : ℕ, 10000 * a + b = n * (a * b)) → (∃ n : ℕ, 10000 * a + b = 73 * (a * b)) :=
by sorry

end NUMINAMATH_CALUDE_multiplication_error_l1697_169730


namespace NUMINAMATH_CALUDE_glass_mass_problem_l1697_169757

theorem glass_mass_problem (full_mass : ℝ) (half_removed_mass : ℝ) 
  (h1 : full_mass = 1000)
  (h2 : half_removed_mass = 700) : 
  full_mass - 2 * (full_mass - half_removed_mass) = 400 := by
  sorry

end NUMINAMATH_CALUDE_glass_mass_problem_l1697_169757


namespace NUMINAMATH_CALUDE_lab_budget_calculation_l1697_169714

theorem lab_budget_calculation (flask_cost safety_gear_cost test_tube_cost remaining_budget : ℝ) 
  (h1 : flask_cost = 150)
  (h2 : test_tube_cost = 2/3 * flask_cost)
  (h3 : safety_gear_cost = 1/2 * test_tube_cost)
  (h4 : remaining_budget = 25) :
  flask_cost + test_tube_cost + safety_gear_cost + remaining_budget = 325 := by
sorry

end NUMINAMATH_CALUDE_lab_budget_calculation_l1697_169714


namespace NUMINAMATH_CALUDE_smallest_t_value_l1697_169774

theorem smallest_t_value (u v w t : ℤ) : 
  (u^3 + v^3 + w^3 = t^3) →
  (u^3 < v^3) →
  (v^3 < w^3) →
  (w^3 < t^3) →
  (u^3 < 0) →
  (v^3 < 0) →
  (w^3 < 0) →
  (t^3 < 0) →
  (∃ k : ℤ, u = k - 1 ∧ v = k ∧ w = k + 1 ∧ t = k + 2) →
  (∀ s : ℤ, s < 0 ∧ (∃ x y z : ℤ, x^3 + y^3 + z^3 = s^3 ∧ 
    x^3 < y^3 ∧ y^3 < z^3 ∧ z^3 < s^3 ∧ 
    x^3 < 0 ∧ y^3 < 0 ∧ z^3 < 0 ∧ s^3 < 0 ∧
    (∃ j : ℤ, x = j - 1 ∧ y = j ∧ z = j + 1 ∧ s = j + 2)) → 
    8 ≤ |s|) →
  8 = |t| :=
sorry

end NUMINAMATH_CALUDE_smallest_t_value_l1697_169774


namespace NUMINAMATH_CALUDE_escalator_steps_l1697_169721

/-- The number of steps on an escalator between two floors -/
def N : ℕ := 47

/-- The number of steps Jack walks while on the moving escalator -/
def jack_steps : ℕ := 29

/-- The number of steps Jill walks while on the moving escalator -/
def jill_steps : ℕ := 11

/-- Jill's travel time is twice Jack's -/
def time_ratio : ℕ := 2

theorem escalator_steps :
  N - jill_steps = time_ratio * (N - jack_steps) :=
sorry

end NUMINAMATH_CALUDE_escalator_steps_l1697_169721


namespace NUMINAMATH_CALUDE_class_size_count_l1697_169717

def is_valid_class_size (n : ℕ) : Prop :=
  ∃ b g : ℕ, n = b + g ∧ n > 25 ∧ 2 < b ∧ b < 10 ∧ 14 < g ∧ g < 23

theorem class_size_count :
  ∃! (s : Finset ℕ), (∀ n, n ∈ s ↔ is_valid_class_size n) ∧ s.card = 6 := by
  sorry

end NUMINAMATH_CALUDE_class_size_count_l1697_169717


namespace NUMINAMATH_CALUDE_quadratic_sequence_proof_l1697_169741

/-- A quadratic function passing through the origin with given derivative -/
noncomputable def f (x : ℝ) : ℝ := sorry

/-- The sequence a_n -/
def a (n : ℕ+) : ℝ := sorry

/-- The sum of the first n terms of a_n -/
def S (n : ℕ+) : ℝ := sorry

/-- The sequence b_n -/
def b (n : ℕ+) : ℝ := sorry

/-- The sum of the first n terms of b_n -/
def T (n : ℕ+) : ℝ := sorry

theorem quadratic_sequence_proof 
  (h1 : f 0 = 0)
  (h2 : ∀ x, deriv f x = 6 * x - 2)
  (h3 : ∀ n : ℕ+, S n = f n) :
  (∀ n : ℕ+, a n = 6 * n - 5) ∧
  (∀ m : ℝ, (∀ n : ℕ+, T n ≥ m / 20) ↔ m ≤ 60 / 7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_sequence_proof_l1697_169741


namespace NUMINAMATH_CALUDE_tom_balloons_l1697_169777

theorem tom_balloons (initial : ℕ) (given : ℕ) (remaining : ℕ) : 
  initial = 30 → given = 16 → remaining = initial - given → remaining = 14 := by
  sorry

end NUMINAMATH_CALUDE_tom_balloons_l1697_169777


namespace NUMINAMATH_CALUDE_unique_representation_theorem_l1697_169735

-- Define a type for representing a person (boy or girl)
inductive Person : Type
| boy : Person
| girl : Person

-- Define a function to convert a natural number to a list of 5 binary digits
def toBinaryDigits (n : Nat) : List Bool :=
  List.reverse (List.take 5 (List.map (fun i => n / 2^i % 2 = 1) (List.range 5)))

-- Define a function to convert a list of binary digits to a list of persons
def binaryToPersons (bits : List Bool) : List Person :=
  List.map (fun b => if b then Person.boy else Person.girl) bits

-- Define a function to convert a list of persons back to a natural number
def personsToNumber (persons : List Person) : Nat :=
  List.foldl (fun acc p => 2 * acc + match p with
    | Person.boy => 1
    | Person.girl => 0) 0 persons

-- Theorem statement
theorem unique_representation_theorem (n : Nat) (h : n > 0 ∧ n ≤ 31) :
  ∃! (arrangement : List Person),
    arrangement.length = 5 ∧
    personsToNumber arrangement = n :=
  sorry

end NUMINAMATH_CALUDE_unique_representation_theorem_l1697_169735


namespace NUMINAMATH_CALUDE_solution_value_l1697_169701

/-- 
If (1, k) is a solution to the equation 2x + y = 6, then k = 4.
-/
theorem solution_value (k : ℝ) : (2 * 1 + k = 6) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1697_169701


namespace NUMINAMATH_CALUDE_bface_hex_to_decimal_l1697_169737

/-- Converts a single hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | _ => 0  -- Default case, should not be reached for valid hex digits

/-- Converts a hexadecimal string to its decimal value -/
def hexadecimal_to_decimal (s : String) : ℕ :=
  s.foldr (fun c acc => hex_to_dec c + 16 * acc) 0

theorem bface_hex_to_decimal :
  hexadecimal_to_decimal "BFACE" = 785102 := by
  sorry

end NUMINAMATH_CALUDE_bface_hex_to_decimal_l1697_169737


namespace NUMINAMATH_CALUDE_equation_solutions_l1697_169740

def satisfies_equation (a b c : ℤ) : Prop :=
  (abs (a + 3) : ℤ) + b^2 + 4*c^2 - 14*b - 12*c + 55 = 0

def solution_set : Set (ℤ × ℤ × ℤ) :=
  {(-2, 8, 2), (-2, 6, 2), (-4, 8, 2), (-4, 6, 2), (-1, 7, 2), (-1, 7, 1), (-5, 7, 2), (-5, 7, 1)}

theorem equation_solutions :
  ∀ (a b c : ℤ), satisfies_equation a b c ↔ (a, b, c) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1697_169740


namespace NUMINAMATH_CALUDE_quadratic_roots_real_for_pure_imaginary_k_l1697_169794

theorem quadratic_roots_real_for_pure_imaginary_k :
  ∀ (k : ℂ), (∃ (r : ℝ), k = r * I) →
  ∃ (z₁ z₂ : ℝ), (5 : ℂ) * (z₁ : ℂ)^2 + 7 * I * (z₁ : ℂ) - k = 0 ∧
                 (5 : ℂ) * (z₂ : ℂ)^2 + 7 * I * (z₂ : ℂ) - k = 0 :=
by sorry


end NUMINAMATH_CALUDE_quadratic_roots_real_for_pure_imaginary_k_l1697_169794


namespace NUMINAMATH_CALUDE_cupcakes_problem_l1697_169799

/-- Given the initial number of cupcakes, the number of cupcakes eaten, and the number of packages,
    calculate the number of cupcakes in each package. -/
def cupcakes_per_package (initial : ℕ) (eaten : ℕ) (packages : ℕ) : ℕ :=
  (initial - eaten) / packages

/-- Theorem: Given 38 initial cupcakes, 14 cupcakes eaten, and 3 packages made,
    the number of cupcakes in each package is 8. -/
theorem cupcakes_problem : cupcakes_per_package 38 14 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_problem_l1697_169799


namespace NUMINAMATH_CALUDE_gasoline_tank_capacity_l1697_169775

theorem gasoline_tank_capacity : ∃ (capacity : ℝ), 
  capacity > 0 ∧
  (3/4 * capacity - 18 = 1/3 * capacity) ∧
  capacity = 43.2 := by
  sorry

end NUMINAMATH_CALUDE_gasoline_tank_capacity_l1697_169775


namespace NUMINAMATH_CALUDE_car_speed_time_relation_l1697_169725

theorem car_speed_time_relation (distance : ℝ) (original_time : ℝ) (new_speed : ℝ) :
  distance = 540 ∧ original_time = 12 ∧ new_speed = 60 →
  (distance / new_speed) / original_time = 3 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_car_speed_time_relation_l1697_169725


namespace NUMINAMATH_CALUDE_nectar_water_percentage_l1697_169731

/-- Given that 1.7 kg of nectar yields 1 kg of honey, and the honey contains 15% water,
    prove that the nectar contains 50% water. -/
theorem nectar_water_percentage :
  ∀ (nectar_weight honey_weight : ℝ) 
    (honey_water_percentage nectar_water_percentage : ℝ),
  nectar_weight = 1.7 →
  honey_weight = 1 →
  honey_water_percentage = 15 →
  nectar_water_percentage = 
    (nectar_weight * honey_water_percentage / 100 + (nectar_weight - honey_weight)) / 
    nectar_weight * 100 →
  nectar_water_percentage = 50 := by
  sorry

end NUMINAMATH_CALUDE_nectar_water_percentage_l1697_169731


namespace NUMINAMATH_CALUDE_total_profit_calculation_l1697_169709

-- Define the investments and A's profit share
def investment_A : ℕ := 6300
def investment_B : ℕ := 4200
def investment_C : ℕ := 10500
def A_profit_share : ℕ := 4080

-- Define the total investment
def total_investment : ℕ := investment_A + investment_B + investment_C

-- Define A's investment ratio
def A_investment_ratio : ℚ := investment_A / total_investment

-- Theorem to prove
theorem total_profit_calculation : 
  ∃ (total_profit : ℕ), 
    (A_investment_ratio * total_profit = A_profit_share) ∧
    (total_profit = 13600) :=
by sorry

end NUMINAMATH_CALUDE_total_profit_calculation_l1697_169709


namespace NUMINAMATH_CALUDE_current_rate_l1697_169744

/-- Given a man's rowing speeds, calculate the rate of the current -/
theorem current_rate (downstream_speed upstream_speed still_water_speed : ℝ)
  (h1 : downstream_speed = 30)
  (h2 : upstream_speed = 10)
  (h3 : still_water_speed = 20) :
  downstream_speed - still_water_speed = 10 := by
  sorry

end NUMINAMATH_CALUDE_current_rate_l1697_169744


namespace NUMINAMATH_CALUDE_ab_value_l1697_169784

theorem ab_value (a b : ℝ) (h : |a + 1| + (b - 3)^2 = 0) : a^b = -1 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1697_169784


namespace NUMINAMATH_CALUDE_beta_values_l1697_169759

theorem beta_values (β : ℂ) (h1 : β ≠ 1) 
  (h2 : Complex.abs (β^3 - 1) = 3 * Complex.abs (β - 1))
  (h3 : Complex.abs (β^6 - 1) = 6 * Complex.abs (β - 1)) :
  β = Complex.I * 2 * Real.sqrt 2 ∨ β = Complex.I * (-2) * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_beta_values_l1697_169759


namespace NUMINAMATH_CALUDE_prob_same_color_correct_l1697_169791

def total_balls : ℕ := 16
def green_balls : ℕ := 8
def red_balls : ℕ := 5
def blue_balls : ℕ := 3

def prob_same_color : ℚ := 98 / 256

theorem prob_same_color_correct :
  (green_balls / total_balls)^2 + (red_balls / total_balls)^2 + (blue_balls / total_balls)^2 = prob_same_color :=
by sorry

end NUMINAMATH_CALUDE_prob_same_color_correct_l1697_169791


namespace NUMINAMATH_CALUDE_magical_stack_131_l1697_169762

/-- Definition of a magical stack -/
def is_magical_stack (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≤ n ∧ b > n ∧ b ≤ 2*n ∧
  (a = 2*a - 1 ∨ b = 2*(b - n))

/-- Theorem: A stack with 392 cards where card 131 retains its position is magical -/
theorem magical_stack_131 :
  ∃ (n : ℕ), 2*n = 392 ∧ is_magical_stack n ∧ 131 ≤ n ∧ 131 = 2*131 - 1 := by
  sorry

end NUMINAMATH_CALUDE_magical_stack_131_l1697_169762


namespace NUMINAMATH_CALUDE_square_field_area_l1697_169706

/-- The area of a square field given the time and speed of a horse running around it -/
theorem square_field_area (time : ℝ) (speed : ℝ) : 
  time = 10 → speed = 12 → (time * speed / 4) ^ 2 = 900 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l1697_169706


namespace NUMINAMATH_CALUDE_average_age_increase_l1697_169770

theorem average_age_increase (initial_count : Nat) (replaced_age1 replaced_age2 women_avg_age : ℕ) :
  initial_count = 9 →
  replaced_age1 = 36 →
  replaced_age2 = 32 →
  women_avg_age = 52 →
  (2 * women_avg_age - (replaced_age1 + replaced_age2)) / initial_count = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_average_age_increase_l1697_169770


namespace NUMINAMATH_CALUDE_simplify_expression_l1697_169779

theorem simplify_expression (a b c : ℝ) : 
  3*a - (4*a - 6*b - 3*c) - 5*(c - b) = -a + 11*b - 2*c := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1697_169779


namespace NUMINAMATH_CALUDE_triangle_side_length_l1697_169733

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = Real.sqrt 10 →
  c = 3 →
  Real.cos A = 1/4 →
  b^2 + c^2 - a^2 = 2 * b * c * Real.cos A →
  b = 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1697_169733


namespace NUMINAMATH_CALUDE_count_rectangles_with_cell_l1697_169745

/-- The number of rectangles containing a specific cell in a grid. -/
def num_rectangles (m n p q : ℕ) : ℕ :=
  p * q * (m - p + 1) * (n - q + 1)

/-- Theorem stating the number of rectangles containing a specific cell in a grid. -/
theorem count_rectangles_with_cell (m n p q : ℕ) 
  (hpm : p ≤ m) (hqn : q ≤ n) (hp : p > 0) (hq : q > 0) : 
  num_rectangles m n p q = p * q * (m - p + 1) * (n - q + 1) := by
  sorry

end NUMINAMATH_CALUDE_count_rectangles_with_cell_l1697_169745


namespace NUMINAMATH_CALUDE_lance_licks_l1697_169708

/-- The number of licks it takes Dan to get to the center of a lollipop -/
def dan_licks : ℕ := 58

/-- The number of licks it takes Michael to get to the center of a lollipop -/
def michael_licks : ℕ := 63

/-- The number of licks it takes Sam to get to the center of a lollipop -/
def sam_licks : ℕ := 70

/-- The number of licks it takes David to get to the center of a lollipop -/
def david_licks : ℕ := 70

/-- The average number of licks it takes for all 5 people to get to the center of a lollipop -/
def average_licks : ℕ := 60

/-- The number of people in the group -/
def num_people : ℕ := 5

/-- The theorem stating how many licks it takes Lance to get to the center of a lollipop -/
theorem lance_licks : 
  (num_people * average_licks) - (dan_licks + michael_licks + sam_licks + david_licks) = 39 := by
  sorry

end NUMINAMATH_CALUDE_lance_licks_l1697_169708
