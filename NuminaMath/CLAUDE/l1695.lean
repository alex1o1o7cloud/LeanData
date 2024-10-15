import Mathlib

namespace NUMINAMATH_CALUDE_exp_cos_inequality_l1695_169534

theorem exp_cos_inequality : 
  (Real.exp (Real.cos 1)) / (Real.cos 2 + 1) < 2 * Real.sqrt (Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_exp_cos_inequality_l1695_169534


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1695_169522

theorem hyperbola_equation (m a b : ℝ) :
  (∀ x y : ℝ, x^2 / (4 + m^2) + y^2 / m^2 = 1 → x^2 / a^2 - y^2 / b^2 = 1) →
  (a^2 + b^2 = 4) →
  (a^2 / b^2 = 4) →
  x^2 - y^2 / 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1695_169522


namespace NUMINAMATH_CALUDE_remainder_problem_l1695_169560

theorem remainder_problem (k : ℕ) 
  (h1 : k > 0)
  (h2 : k < 168)
  (h3 : k % 5 = 2)
  (h4 : k % 6 = 5)
  (h5 : k % 8 = 7)
  (h6 : k % 11 = 3) :
  k % 13 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1695_169560


namespace NUMINAMATH_CALUDE_factorial_trailing_zeros_l1695_169535

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem factorial_trailing_zeros : 
  trailing_zeros 238 = 57 ∧ trailing_zeros 238 - trailing_zeros 236 = 0 :=
by sorry

end NUMINAMATH_CALUDE_factorial_trailing_zeros_l1695_169535


namespace NUMINAMATH_CALUDE_school_seminar_cost_l1695_169585

/-- Calculates the total amount spent by a school for a teacher seminar with discounts and food allowance. -/
def total_seminar_cost (regular_fee : ℝ) (discount_percent : ℝ) (num_teachers : ℕ) (food_allowance : ℝ) : ℝ :=
  let discounted_fee := regular_fee * (1 - discount_percent)
  let total_seminar_fees := discounted_fee * num_teachers
  let total_food_allowance := food_allowance * num_teachers
  total_seminar_fees + total_food_allowance

/-- Theorem stating the total cost for the school's teacher seminar -/
theorem school_seminar_cost :
  total_seminar_cost 150 0.05 10 10 = 1525 :=
by sorry

end NUMINAMATH_CALUDE_school_seminar_cost_l1695_169585


namespace NUMINAMATH_CALUDE_identity_function_is_unique_solution_l1695_169508

theorem identity_function_is_unique_solution
  (f : ℕ → ℕ)
  (h : ∀ n : ℕ, f n + f (f n) + f (f (f n)) = 3 * n) :
  ∀ n : ℕ, f n = n :=
by sorry

end NUMINAMATH_CALUDE_identity_function_is_unique_solution_l1695_169508


namespace NUMINAMATH_CALUDE_probability_specific_individual_in_sample_l1695_169515

/-- The probability of selecting a specific individual in a simple random sample -/
theorem probability_specific_individual_in_sample 
  (population_size : ℕ) 
  (sample_size : ℕ) 
  (h1 : population_size = 10)
  (h2 : sample_size = 3)
  (h3 : sample_size < population_size) :
  (sample_size : ℚ) / population_size = 3 / 10 :=
by sorry

end NUMINAMATH_CALUDE_probability_specific_individual_in_sample_l1695_169515


namespace NUMINAMATH_CALUDE_max_value_sum_l1695_169537

/-- Given positive real numbers x, y, and z satisfying 4x^2 + 9y^2 + 16z^2 = 144,
    the maximum value N of the expression 3xz + 5yz + 8xy plus the sum of x, y, and z
    that produce this maximum is equal to 319. -/
theorem max_value_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 4*x^2 + 9*y^2 + 16*z^2 = 144) :
  ∃ (N x_N y_N z_N : ℝ),
    (∀ x' y' z' : ℝ, x' > 0 → y' > 0 → z' > 0 → 4*x'^2 + 9*y'^2 + 16*z'^2 = 144 →
      3*x'*z' + 5*y'*z' + 8*x'*y' ≤ N) ∧
    3*x_N*z_N + 5*y_N*z_N + 8*x_N*y_N = N ∧
    4*x_N^2 + 9*y_N^2 + 16*z_N^2 = 144 ∧
    N + x_N + y_N + z_N = 319 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_l1695_169537


namespace NUMINAMATH_CALUDE_number_difference_proof_l1695_169519

theorem number_difference_proof (x : ℝ) (h : x - (3/4) * x = 100) : (1/4) * x = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_proof_l1695_169519


namespace NUMINAMATH_CALUDE_book_price_theorem_l1695_169548

theorem book_price_theorem (suggested_retail_price : ℝ) 
  (h1 : suggested_retail_price > 0) : 
  let marked_price := 0.6 * suggested_retail_price
  let alice_paid := 0.75 * marked_price
  alice_paid / suggested_retail_price = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_book_price_theorem_l1695_169548


namespace NUMINAMATH_CALUDE_polynomial_sum_l1695_169552

-- Define the polynomial P
def P (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem polynomial_sum (a b c d : ℝ) :
  P a b c d 1 = 2000 →
  P a b c d 2 = 4000 →
  P a b c d 3 = 6000 →
  P a b c d 9 + P a b c d (-5) = 12704 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l1695_169552


namespace NUMINAMATH_CALUDE_max_value_on_circle_l1695_169577

theorem max_value_on_circle : 
  ∀ x y : ℝ, x^2 + y^2 - 6*x + 8 = 0 → x^2 + y^2 ≤ 16 ∧ ∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 6*x₀ + 8 = 0 ∧ x₀^2 + y₀^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l1695_169577


namespace NUMINAMATH_CALUDE_cubes_volume_ratio_l1695_169518

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the number of cubes that can fit along each dimension -/
def cubesFit (boxDim : ℕ) (cubeDim : ℕ) : ℕ :=
  boxDim / cubeDim

/-- Calculates the volume occupied by cubes in the box -/
def cubesVolume (box : BoxDimensions) (cubeDim : ℕ) : ℕ :=
  let l := cubesFit box.length cubeDim
  let w := cubesFit box.width cubeDim
  let h := cubesFit box.height cubeDim
  l * w * h * (cubeDim ^ 3)

/-- The main theorem to be proved -/
theorem cubes_volume_ratio (box : BoxDimensions) (cubeDim : ℕ) : 
  box.length = 8 → box.width = 6 → box.height = 12 → cubeDim = 4 →
  (cubesVolume box cubeDim : ℚ) / (boxVolume box : ℚ) = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_cubes_volume_ratio_l1695_169518


namespace NUMINAMATH_CALUDE_geometric_series_relation_l1695_169559

/-- Given two infinite geometric series with specific properties, prove that n = 6 -/
theorem geometric_series_relation (n : ℝ) : 
  let a₁ : ℝ := 15  -- First term of both series
  let b₁ : ℝ := 6   -- Second term of first series
  let b₂ : ℝ := 6 + n  -- Second term of second series
  let r₁ : ℝ := b₁ / a₁  -- Common ratio of first series
  let r₂ : ℝ := b₂ / a₁  -- Common ratio of second series
  let S₁ : ℝ := a₁ / (1 - r₁)  -- Sum of first series
  let S₂ : ℝ := a₁ / (1 - r₂)  -- Sum of second series
  S₂ = 3 * S₁ →  -- Condition: sum of second series is three times the sum of first series
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_relation_l1695_169559


namespace NUMINAMATH_CALUDE_candy_bar_difference_l1695_169527

theorem candy_bar_difference (lena kevin nicole : ℕ) : 
  lena = 16 →
  lena + 5 = 3 * kevin →
  kevin + 4 = nicole →
  lena - nicole = 5 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_difference_l1695_169527


namespace NUMINAMATH_CALUDE_ellipse_properties_l1695_169572

/-- An ellipse with center at the origin, foci on the x-axis, and max/min distances to focus 3 and 1 -/
structure Ellipse where
  center : ℝ × ℝ := (0, 0)
  foci_on_x_axis : Bool
  max_dist : ℝ := 3
  min_dist : ℝ := 1

/-- The standard equation of the ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- A line with equation y = x + m -/
structure Line where
  m : ℝ

/-- Predicate for line intersection with ellipse -/
def intersects (l : Line) (e : Ellipse) : Prop :=
  ∃ x y : ℝ, y = x + l.m ∧ standard_equation e x y

theorem ellipse_properties (e : Ellipse) :
  (∀ x y : ℝ, standard_equation e x y ↔ 
    x^2 / 4 + y^2 / 3 = 1) ∧
  (∀ l : Line, intersects l e ↔ 
    -Real.sqrt 7 ≤ l.m ∧ l.m ≤ Real.sqrt 7) := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1695_169572


namespace NUMINAMATH_CALUDE_second_error_greater_l1695_169568

/-- Given two measured lines with their lengths and errors, prove that the absolute error of the second measurement is greater than the first. -/
theorem second_error_greater (length1 length2 error1 error2 : ℝ) 
  (h1 : length1 = 50)
  (h2 : length2 = 200)
  (h3 : error1 = 0.05)
  (h4 : error2 = 0.4) : 
  error2 > error1 := by
  sorry

end NUMINAMATH_CALUDE_second_error_greater_l1695_169568


namespace NUMINAMATH_CALUDE_stratified_sample_size_l1695_169541

/-- Represents the total number of employees -/
def total_employees : ℕ := 750

/-- Represents the number of young employees -/
def young_employees : ℕ := 350

/-- Represents the number of middle-aged employees -/
def middle_aged_employees : ℕ := 250

/-- Represents the number of elderly employees -/
def elderly_employees : ℕ := 150

/-- Represents the number of young employees in the sample -/
def young_in_sample : ℕ := 7

/-- Theorem stating that the sample size is 15 given the conditions -/
theorem stratified_sample_size :
  ∃ (sample_size : ℕ),
    sample_size * young_employees = young_in_sample * total_employees ∧
    sample_size = 15 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l1695_169541


namespace NUMINAMATH_CALUDE_third_dimension_of_large_box_l1695_169599

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of small boxes that can fit into a larger box -/
def maxSmallBoxes (largeBox : BoxDimensions) (smallBox : BoxDimensions) : ℕ :=
  (largeBox.length / smallBox.length) * (largeBox.width / smallBox.width) * (largeBox.height / smallBox.height)

theorem third_dimension_of_large_box 
  (largeBox : BoxDimensions) 
  (smallBox : BoxDimensions) 
  (h : ℕ) :
  largeBox.length = 12 ∧ 
  largeBox.width = 14 ∧ 
  largeBox.height = h ∧
  smallBox.length = 3 ∧ 
  smallBox.width = 7 ∧ 
  smallBox.height = 2 ∧
  maxSmallBoxes largeBox smallBox = 64 →
  h = 16 :=
by sorry

end NUMINAMATH_CALUDE_third_dimension_of_large_box_l1695_169599


namespace NUMINAMATH_CALUDE_oblique_triangular_prism_volume_l1695_169596

/-- The volume of an oblique triangular prism with specific properties -/
theorem oblique_triangular_prism_volume (a : ℝ) (ha : a > 0) :
  let base_area := (a^2 * Real.sqrt 3) / 4
  let height := a * Real.sqrt 3 / 2
  base_area * height = (3 * a^3) / 8 := by sorry

end NUMINAMATH_CALUDE_oblique_triangular_prism_volume_l1695_169596


namespace NUMINAMATH_CALUDE_exists_n_where_B_less_than_A_l1695_169576

/-- Alphonse's jump function -/
def A (n : ℕ) : ℕ :=
  if n ≥ 8 then A (n - 8) + 1 else n

/-- Beryl's jump function -/
def B (n : ℕ) : ℕ :=
  if n ≥ 7 then B (n - 7) + 1 else n

/-- Theorem stating the existence of n > 200 where B(n) < A(n) -/
theorem exists_n_where_B_less_than_A :
  ∃ n : ℕ, n > 200 ∧ B n < A n :=
sorry

end NUMINAMATH_CALUDE_exists_n_where_B_less_than_A_l1695_169576


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1695_169581

universe u

def U : Set Nat := {0, 1, 2, 3, 4}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 4}

theorem complement_union_theorem :
  (Aᶜ ∩ U) ∪ B = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1695_169581


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1695_169592

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (72 - 18*x - x^2 = 0) → (∃ r s : ℝ, (72 - 18*r - r^2 = 0) ∧ (72 - 18*s - s^2 = 0) ∧ (r + s = 18)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1695_169592


namespace NUMINAMATH_CALUDE_knights_adjacent_probability_l1695_169532

def numKnights : ℕ := 20
def chosenKnights : ℕ := 4

def probability_no_adjacent (n k : ℕ) : ℚ :=
  (n - 3) * (n - 5) * (n - 7) * (n - 9) / (n.choose k)

theorem knights_adjacent_probability :
  ∃ (Q : ℚ), Q = 1 - probability_no_adjacent numKnights chosenKnights :=
sorry

end NUMINAMATH_CALUDE_knights_adjacent_probability_l1695_169532


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1695_169524

/-- A quadratic function g(x) = px^2 + qx + r -/
def g (p q r : ℝ) (x : ℝ) : ℝ := p * x^2 + q * x + r

/-- Theorem: If g(-2) = 0, g(3) = 0, and g(1) = 5, then q = 5/6 -/
theorem quadratic_coefficient (p q r : ℝ) :
  g p q r (-2) = 0 → g p q r 3 = 0 → g p q r 1 = 5 → q = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1695_169524


namespace NUMINAMATH_CALUDE_arthur_walked_seven_miles_l1695_169556

/-- The distance Arthur walked in miles -/
def arthur_distance (blocks_east blocks_north blocks_west : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_east + blocks_north + blocks_west : ℚ) * miles_per_block

/-- Theorem stating that Arthur walked 7 miles -/
theorem arthur_walked_seven_miles :
  arthur_distance 8 15 5 (1/4) = 7 := by
  sorry

end NUMINAMATH_CALUDE_arthur_walked_seven_miles_l1695_169556


namespace NUMINAMATH_CALUDE_weight_of_b_l1695_169569

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43) :
  b = 31 := by sorry

end NUMINAMATH_CALUDE_weight_of_b_l1695_169569


namespace NUMINAMATH_CALUDE_mrs_hilt_pecan_pies_l1695_169589

/-- The number of pecan pies Mrs. Hilt baked -/
def pecan_pies : ℕ := sorry

/-- The number of apple pies Mrs. Hilt baked -/
def apple_pies : ℕ := 14

/-- The number of rows in the pie arrangement -/
def rows : ℕ := 30

/-- The number of pies in each row -/
def pies_per_row : ℕ := 5

/-- The total number of pies -/
def total_pies : ℕ := rows * pies_per_row

theorem mrs_hilt_pecan_pies :
  pecan_pies = 136 :=
by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_pecan_pies_l1695_169589


namespace NUMINAMATH_CALUDE_telescope_visual_range_l1695_169579

theorem telescope_visual_range (original_range : ℝ) : 
  (original_range + 1.5 * original_range = 150) → original_range = 60 := by
  sorry

end NUMINAMATH_CALUDE_telescope_visual_range_l1695_169579


namespace NUMINAMATH_CALUDE_polynomial_equality_l1695_169551

theorem polynomial_equality (a b : ℝ) : 
  (∀ x : ℝ, (x + a) * (x - 2) = x^2 + b*x - 6) → (a = 3 ∧ b = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1695_169551


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1695_169586

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 = 3 →
  a 1 + a 6 = 12 →
  a 7 + a 8 + a 9 = 45 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1695_169586


namespace NUMINAMATH_CALUDE_first_shipment_cost_l1695_169593

/-- Represents the cost of a clothing shipment -/
def shipment_cost (num_sweaters num_jackets : ℕ) (sweater_price jacket_price : ℚ) : ℚ :=
  num_sweaters * sweater_price + num_jackets * jacket_price

theorem first_shipment_cost (sweater_price jacket_price : ℚ) :
  shipment_cost 5 15 sweater_price jacket_price = 550 →
  shipment_cost 10 20 sweater_price jacket_price = 1100 := by
  sorry

end NUMINAMATH_CALUDE_first_shipment_cost_l1695_169593


namespace NUMINAMATH_CALUDE_binomial_18_10_l1695_169531

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 45760 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_10_l1695_169531


namespace NUMINAMATH_CALUDE_triangle_angle_c_l1695_169574

theorem triangle_angle_c (A B C : ℝ) : 
  A + B + C = π →  -- Sum of angles in a triangle
  |2 * Real.sin A - 1| + |Real.sqrt 2 / 2 - Real.cos B| = 0 →
  C = 7 * π / 12  -- 105° in radians
:= by sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l1695_169574


namespace NUMINAMATH_CALUDE_paper_pack_sheets_l1695_169501

theorem paper_pack_sheets : ∃ (S P : ℕ), S = 115 ∧ S - P = 100 ∧ 5 * P + 35 = S := by
  sorry

end NUMINAMATH_CALUDE_paper_pack_sheets_l1695_169501


namespace NUMINAMATH_CALUDE_rectangle_ratio_extension_l1695_169525

theorem rectangle_ratio_extension (x : ℝ) :
  (2*x > 0) →
  (5*x > 0) →
  ((2*x + 9) / (5*x + 9) = 3/7) →
  ((2*x + 18) / (5*x + 18) = 5/11) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_ratio_extension_l1695_169525


namespace NUMINAMATH_CALUDE_smallest_radii_sum_squares_l1695_169533

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Check if four points lie on a circle -/
def onCircle (A B C D : Point) : Prop := sorry

/-- The theorem to be proved -/
theorem smallest_radii_sum_squares
  (A : Point) (B : Point) (C : Point) (D : Point)
  (h1 : A = ⟨0, 0⟩)
  (h2 : B = ⟨-1, -1⟩)
  (h3 : ∃ (x y : ℤ), x > y ∧ y > 0 ∧ C = ⟨x, y⟩ ∧ D = ⟨x + 1, y⟩)
  (h4 : onCircle A B C D) :
  ∃ (r₁ r₂ : ℝ), r₁ > 0 ∧ r₂ > r₁ ∧
    (∀ (r : ℝ), onCircle A B C D → r ≥ r₁) ∧
    (∀ (r : ℝ), onCircle A B C D ∧ r ≠ r₁ → r ≥ r₂) ∧
    r₁^2 + r₂^2 = 1381 := by
  sorry

end NUMINAMATH_CALUDE_smallest_radii_sum_squares_l1695_169533


namespace NUMINAMATH_CALUDE_sum_of_roots_l1695_169553

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 3*a^2 + 5*a - 17 = 0)
  (hb : b^3 - 3*b^2 + 5*b + 11 = 0) : 
  a + b = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1695_169553


namespace NUMINAMATH_CALUDE_range_of_a_l1695_169549

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem range_of_a (a : ℝ) (h : a ∈ A) : a ∈ Set.Icc (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1695_169549


namespace NUMINAMATH_CALUDE_marble_ratio_l1695_169591

theorem marble_ratio : 
  let total_marbles : ℕ := 63
  let red_marbles : ℕ := 38
  let green_marbles : ℕ := 4
  let dark_blue_marbles : ℕ := total_marbles - red_marbles - green_marbles
  (dark_blue_marbles : ℚ) / total_marbles = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_l1695_169591


namespace NUMINAMATH_CALUDE_fertilizer_production_equation_l1695_169594

/-- Given a fertilizer factory with:
  * Original production plan of x tons per day
  * New production of x + 3 tons per day
  * Time to produce 180 tons (new rate) = Time to produce 120 tons (original rate)
  Prove that the equation 120/x = 180/(x + 3) correctly represents the relationship
  between the original production rate x and the time taken to produce different
  quantities of fertilizer. -/
theorem fertilizer_production_equation (x : ℝ) (h : x > 0) :
  (120 : ℝ) / x = 180 / (x + 3) ↔
  (120 : ℝ) / x = (180 : ℝ) / (x + 3) :=
by sorry

end NUMINAMATH_CALUDE_fertilizer_production_equation_l1695_169594


namespace NUMINAMATH_CALUDE_wonderland_roads_l1695_169563

/-- The number of vertices in the complete graph -/
def n : ℕ := 5

/-- The number of edges shown on Alice's map -/
def shown_edges : ℕ := 7

/-- The total number of edges in a complete graph with n vertices -/
def total_edges (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of missing edges -/
def missing_edges : ℕ := total_edges n - shown_edges

theorem wonderland_roads :
  missing_edges = 3 := by sorry

end NUMINAMATH_CALUDE_wonderland_roads_l1695_169563


namespace NUMINAMATH_CALUDE_yellow_pairs_count_l1695_169564

theorem yellow_pairs_count (total_students : ℕ) (blue_students : ℕ) (yellow_students : ℕ) 
  (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  total_students = 156 →
  blue_students = 68 →
  yellow_students = 88 →
  total_pairs = 78 →
  blue_blue_pairs = 31 →
  total_students = blue_students + yellow_students →
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 41 ∧ 
    yellow_yellow_pairs + blue_blue_pairs + (blue_students - 2 * blue_blue_pairs) = total_pairs :=
by sorry

end NUMINAMATH_CALUDE_yellow_pairs_count_l1695_169564


namespace NUMINAMATH_CALUDE_inequality_proof_l1695_169514

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^2 + y^4 + z^6 ≥ x*y^2 + y^2*z^3 + x*z^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1695_169514


namespace NUMINAMATH_CALUDE_largest_whole_number_less_than_150_over_11_l1695_169573

theorem largest_whole_number_less_than_150_over_11 : 
  (∀ x : ℕ, x > 13 → 11 * x ≥ 150) ∧ (11 * 13 < 150) := by
  sorry

end NUMINAMATH_CALUDE_largest_whole_number_less_than_150_over_11_l1695_169573


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1695_169566

theorem inequality_equivalence (x : ℝ) : 3 * x + 2 < 10 - 2 * x ↔ x < 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1695_169566


namespace NUMINAMATH_CALUDE_two_x_eq_zero_is_linear_l1695_169502

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation 2x = 0 -/
def f (x : ℝ) : ℝ := 2 * x

/-- Theorem: The equation 2x = 0 is a linear equation -/
theorem two_x_eq_zero_is_linear : is_linear_equation f := by
  sorry


end NUMINAMATH_CALUDE_two_x_eq_zero_is_linear_l1695_169502


namespace NUMINAMATH_CALUDE_digit_sum_divisible_by_11_l1695_169562

/-- The digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Theorem: In any 39 successive natural numbers, at least one has a digit sum divisible by 11 -/
theorem digit_sum_divisible_by_11 (n : ℕ) : 
  ∃ k : ℕ, k ∈ Finset.range 39 ∧ (digitSum (n + k) % 11 = 0) := by sorry

end NUMINAMATH_CALUDE_digit_sum_divisible_by_11_l1695_169562


namespace NUMINAMATH_CALUDE_train_platform_ratio_l1695_169517

/-- Given a train passing a pole and a platform, prove the ratio of platform length to train length -/
theorem train_platform_ratio (l t v : ℝ) (h1 : l > 0) (h2 : t > 0) (h3 : v > 0) :
  let pole_time := t
  let platform_time := 3.5 * t
  let train_length := l
  let platform_length := v * platform_time - train_length
  platform_length / train_length = 2.5 := by sorry

end NUMINAMATH_CALUDE_train_platform_ratio_l1695_169517


namespace NUMINAMATH_CALUDE_stating_tournament_orderings_l1695_169547

/-- Represents the number of players in the tournament -/
def num_players : Nat := 6

/-- Represents the number of possible outcomes for each game -/
def outcomes_per_game : Nat := 2

/-- Calculates the number of possible orderings in the tournament -/
def num_orderings : Nat := outcomes_per_game ^ (num_players - 1)

/-- 
Theorem stating that the number of possible orderings in the tournament is 32
given the specified number of players and outcomes per game.
-/
theorem tournament_orderings :
  num_orderings = 32 :=
by sorry

end NUMINAMATH_CALUDE_stating_tournament_orderings_l1695_169547


namespace NUMINAMATH_CALUDE_smallest_factor_for_square_l1695_169520

theorem smallest_factor_for_square (a : ℕ) : 
  3150 = 2 * 3^2 * 5^2 * 7 → 
  (∀ k : ℕ, k > 0 ∧ k < 14 → ¬ ∃ m : ℕ, 3150 * k = m^2) ∧
  (∃ m : ℕ, 3150 * 14 = m^2) ∧
  (14 > 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_factor_for_square_l1695_169520


namespace NUMINAMATH_CALUDE_integral_proof_l1695_169516

open Real

noncomputable def f (x : ℝ) := (1/2) * log (abs (x^2 + 2 * sin x))

theorem integral_proof (x : ℝ) :
  deriv f x = (x + cos x) / (x^2 + 2 * sin x) :=
by sorry

end NUMINAMATH_CALUDE_integral_proof_l1695_169516


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1695_169590

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 1 + a 2 + a 3 = 7) →
  (a 2 + a 3 + a 4 = 14) →
  (a 4 + a 5 + a 6 = 56) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1695_169590


namespace NUMINAMATH_CALUDE_movie_theatre_attendance_l1695_169578

theorem movie_theatre_attendance (total_seats : ℕ) (adult_price child_price : ℚ) 
  (total_revenue : ℚ) (h_seats : total_seats = 250) (h_adult_price : adult_price = 6)
  (h_child_price : child_price = 4) (h_revenue : total_revenue = 1124) :
  ∃ (children : ℕ), children = 188 ∧ 
    (∃ (adults : ℕ), adults + children = total_seats ∧
      adult_price * adults + child_price * children = total_revenue) :=
by sorry

end NUMINAMATH_CALUDE_movie_theatre_attendance_l1695_169578


namespace NUMINAMATH_CALUDE_kate_candy_count_l1695_169507

/-- Given a distribution of candy among four children (Kate, Robert, Bill, and Mary),
    prove that Kate gets 4 pieces of candy. -/
theorem kate_candy_count (kate robert bill mary : ℕ)
  (total : kate + robert + bill + mary = 20)
  (robert_kate : robert = kate + 2)
  (bill_mary : bill = mary - 6)
  (mary_robert : mary = robert + 2)
  (kate_bill : kate = bill + 2) :
  kate = 4 := by
  sorry

end NUMINAMATH_CALUDE_kate_candy_count_l1695_169507


namespace NUMINAMATH_CALUDE_probability_between_C_and_E_l1695_169521

/-- Given points A, B, C, D, and E on a line segment AB, where AB = 4AD = 8BC = 2DE,
    the probability of a randomly selected point on AB being between C and E is 7/8. -/
theorem probability_between_C_and_E (A B C D E : ℝ) : 
  A < B ∧ A ≤ C ∧ C < D ∧ D < E ∧ E ≤ B ∧
  B - A = 4 * (D - A) ∧
  B - A = 8 * (C - B) ∧
  B - A = 2 * (E - D) →
  (E - C) / (B - A) = 7 / 8 := by
sorry

end NUMINAMATH_CALUDE_probability_between_C_and_E_l1695_169521


namespace NUMINAMATH_CALUDE_expression_equals_minus_15i_l1695_169504

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The complex number z -/
noncomputable def z : ℂ := (1 + i) / (1 - i)

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The expression to be evaluated -/
noncomputable def expression : ℂ := 
  binomial 8 1 + 
  binomial 8 2 * z + 
  binomial 8 3 * z^2 + 
  binomial 8 4 * z^3 + 
  binomial 8 5 * z^4 + 
  binomial 8 6 * z^5 + 
  binomial 8 7 * z^6 + 
  binomial 8 8 * z^7

theorem expression_equals_minus_15i : expression = -15 * i := by sorry

end NUMINAMATH_CALUDE_expression_equals_minus_15i_l1695_169504


namespace NUMINAMATH_CALUDE_unique_factorial_sum_l1695_169500

/-- factorial function -/
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Function to get the hundreds digit of a natural number -/
def hundreds_digit (n : ℕ) : ℕ := 
  (n / 100) % 10

/-- Function to get the tens digit of a natural number -/
def tens_digit (n : ℕ) : ℕ := 
  (n / 10) % 10

/-- Function to get the units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := 
  n % 10

/-- Theorem stating that 145 is the only three-digit number with 1 as its hundreds digit 
    that is equal to the sum of the factorials of its digits -/
theorem unique_factorial_sum : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ hundreds_digit n = 1 → 
  (n = factorial (hundreds_digit n) + factorial (tens_digit n) + factorial (units_digit n) ↔ n = 145) := by
  sorry

end NUMINAMATH_CALUDE_unique_factorial_sum_l1695_169500


namespace NUMINAMATH_CALUDE_mean_height_is_70_l1695_169570

def heights : List ℕ := [58, 59, 60, 62, 63, 65, 65, 68, 70, 71, 71, 72, 76, 76, 78, 79, 79]

theorem mean_height_is_70 : 
  (List.sum heights) / (heights.length : ℚ) = 70 := by
  sorry

end NUMINAMATH_CALUDE_mean_height_is_70_l1695_169570


namespace NUMINAMATH_CALUDE_complementary_angles_ratio_l1695_169529

theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  a = 4 * b →   -- ratio of angles is 4:1
  b = 18 :=     -- smaller angle is 18 degrees
by
  sorry

end NUMINAMATH_CALUDE_complementary_angles_ratio_l1695_169529


namespace NUMINAMATH_CALUDE_remainder_1743_base12_div_9_l1695_169580

/-- Converts a base-12 number to base-10 --/
def base12ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ i)) 0

/-- The base-12 representation of 1743 --/
def num1743Base12 : List Nat := [3, 4, 7, 1]

theorem remainder_1743_base12_div_9 :
  (base12ToBase10 num1743Base12) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1743_base12_div_9_l1695_169580


namespace NUMINAMATH_CALUDE_circle_equation_l1695_169503

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the line on which the center lies
def center_line (x y : ℝ) : Prop := y = -4 * x

-- Define the tangent point
def tangent_point : ℝ × ℝ := (3, -2)

-- State the theorem
theorem circle_equation 
  (C : Circle) 
  (h1 : line_l (tangent_point.1) (tangent_point.2))
  (h2 : center_line C.center.1 C.center.2)
  (h3 : ∃ (t : ℝ), C.center.1 + t * (tangent_point.1 - C.center.1) = tangent_point.1 ∧
                   C.center.2 + t * (tangent_point.2 - C.center.2) = tangent_point.2 ∧
                   t = 1) :
  ∀ (x y : ℝ), (x - 1)^2 + (y + 4)^2 = 8 ↔ 
    (x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l1695_169503


namespace NUMINAMATH_CALUDE_perpendicular_condition_parallel_condition_l1695_169595

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-1, 2]

-- Define the dot product of two 2D vectors
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Define perpendicularity of two 2D vectors
def perpendicular (v w : Fin 2 → ℝ) : Prop := dot_product v w = 0

-- Define parallelism of two 2D vectors
def parallel (v w : Fin 2 → ℝ) : Prop := ∃ (c : ℝ), ∀ (i : Fin 2), v i = c * w i

-- Define the vector operations
def add_vectors (v w : Fin 2 → ℝ) : Fin 2 → ℝ := λ i => v i + w i
def scale_vector (k : ℝ) (v : Fin 2 → ℝ) : Fin 2 → ℝ := λ i => k * v i

-- Theorem statements
theorem perpendicular_condition (k : ℝ) : 
  perpendicular (add_vectors (scale_vector k a) b) (add_vectors a (scale_vector (-3) b)) ↔ k = -3 := by sorry

theorem parallel_condition (k : ℝ) : 
  parallel (add_vectors (scale_vector k a) b) (add_vectors a (scale_vector (-3) b)) ↔ k = -1/3 := by sorry

end NUMINAMATH_CALUDE_perpendicular_condition_parallel_condition_l1695_169595


namespace NUMINAMATH_CALUDE_expression_evaluation_l1695_169597

theorem expression_evaluation :
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1695_169597


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1695_169526

theorem polynomial_divisibility (n : ℤ) : 
  ∃ k : ℤ, (n + 7)^2 - n^2 = 7 * k := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1695_169526


namespace NUMINAMATH_CALUDE_max_regular_lines_six_points_l1695_169539

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Enumeration of possible regular line types -/
inductive RegularLineType
  | Horizontal
  | Vertical
  | LeftDiagonal
  | RightDiagonal

/-- A regular line in the 2D plane -/
structure RegularLine where
  type : RegularLineType
  offset : ℝ

/-- Function to check if a point lies on a regular line -/
def pointOnRegularLine (p : Point2D) (l : RegularLine) : Prop :=
  match l.type with
  | RegularLineType.Horizontal => p.y = l.offset
  | RegularLineType.Vertical => p.x = l.offset
  | RegularLineType.LeftDiagonal => p.y - p.x = l.offset
  | RegularLineType.RightDiagonal => p.y + p.x = l.offset

/-- The main theorem stating the maximum number of regular lines -/
theorem max_regular_lines_six_points (points : Fin 6 → Point2D) :
  (∃ (lines : Finset RegularLine), 
    (∀ l ∈ lines, ∃ i j, i ≠ j ∧ pointOnRegularLine (points i) l ∧ pointOnRegularLine (points j) l) ∧
    lines.card = 11) ∧
  (∀ (lines : Finset RegularLine),
    (∀ l ∈ lines, ∃ i j, i ≠ j ∧ pointOnRegularLine (points i) l ∧ pointOnRegularLine (points j) l) →
    lines.card ≤ 11) :=
sorry

end NUMINAMATH_CALUDE_max_regular_lines_six_points_l1695_169539


namespace NUMINAMATH_CALUDE_rachel_essay_editing_time_l1695_169513

/-- Rachel's essay writing problem -/
theorem rachel_essay_editing_time 
  (writing_rate : ℕ → ℕ)  -- Function mapping pages to minutes
  (research_time : ℕ)     -- Time spent researching in minutes
  (total_pages : ℕ)       -- Total pages written
  (total_time : ℕ)        -- Total time spent on the essay in minutes
  (h1 : writing_rate 1 = 30)  -- Writing rate is 1 page per 30 minutes
  (h2 : research_time = 45)   -- 45 minutes spent researching
  (h3 : total_pages = 6)      -- 6 pages written in total
  (h4 : total_time = 5 * 60)  -- Total time is 5 hours (300 minutes)
  : total_time - (research_time + writing_rate total_pages) = 75 := by
  sorry

#check rachel_essay_editing_time

end NUMINAMATH_CALUDE_rachel_essay_editing_time_l1695_169513


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l1695_169567

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 18) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 342 := by sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l1695_169567


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1695_169538

theorem necessary_but_not_sufficient :
  (∀ a b : ℝ, a > b ∧ b > 0 → a / b > 1) ∧
  (∃ a b : ℝ, a / b > 1 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1695_169538


namespace NUMINAMATH_CALUDE_terrys_trip_distance_l1695_169557

/-- Proves that given the conditions of Terry's trip, the total distance driven is 780 miles. -/
theorem terrys_trip_distance :
  ∀ (scenic_road_mpg freeway_mpg : ℝ),
  freeway_mpg = scenic_road_mpg + 6.5 →
  (9 * scenic_road_mpg + 17 * freeway_mpg) / (9 + 17) = 30 →
  9 * scenic_road_mpg + 17 * freeway_mpg = 780 :=
by
  sorry

#check terrys_trip_distance

end NUMINAMATH_CALUDE_terrys_trip_distance_l1695_169557


namespace NUMINAMATH_CALUDE_circle_area_l1695_169554

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 6*x - 4*y = 3

-- Define the center and radius of the circle
def circle_center : ℝ × ℝ := (-3, 2)
def circle_radius : ℝ := 4

-- Theorem statement
theorem circle_area :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    (center = circle_center) ∧
    (radius = circle_radius) ∧
    (Real.pi * radius^2 = 16 * Real.pi) :=
sorry

end NUMINAMATH_CALUDE_circle_area_l1695_169554


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1695_169536

-- Problem 1
theorem problem_1 (x y : ℝ) :
  (x + y) * (x - 2*y) + (x - y)^2 + 3*x * 2*y = 2*x^2 + 3*x*y - y^2 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 1) :
  (x^2 - 4*x + 4) / (x^2 - x) / (x + 1 - 3 / (x - 1)) = (x - 2) / (x * (x + 2)) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1695_169536


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l1695_169565

theorem shaded_area_between_circles (r : Real) : 
  r > 0 → -- radius of smaller circle is positive
  (2 * r = 6) → -- diameter of smaller circle is 6 units
  π * (3 * r)^2 - π * r^2 = 72 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l1695_169565


namespace NUMINAMATH_CALUDE_peony_count_l1695_169575

theorem peony_count (n : ℕ) 
  (h1 : ∃ (s d t : ℕ), n = s + d + t ∧ t = s + 30)
  (h2 : ∃ (x : ℕ), s = 4 * x ∧ d = 2 * x ∧ t = 6 * x) : 
  n = 180 := by
sorry

end NUMINAMATH_CALUDE_peony_count_l1695_169575


namespace NUMINAMATH_CALUDE_subset_condition_l1695_169582

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 2 = 0}

-- State the theorem
theorem subset_condition (a : ℝ) : B a ⊆ A ↔ a = 0 ∨ a = 1 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l1695_169582


namespace NUMINAMATH_CALUDE_combined_painting_time_l1695_169523

/-- Given Shawn's and Karen's individual painting rates, calculate their combined time to paint one house -/
theorem combined_painting_time (shawn_rate karen_rate : ℝ) (h1 : shawn_rate = 1 / 18) (h2 : karen_rate = 1 / 12) :
  1 / (shawn_rate + karen_rate) = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_combined_painting_time_l1695_169523


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l1695_169505

theorem divisibility_equivalence (r : ℕ) (k : ℕ) :
  (∃ (m n : ℕ), m > 1 ∧ m % 2 = 1 ∧ k ∣ m^(2^r) - 1 ∧ m ∣ n^((m^(2^r) - 1)/k) + 1) ↔
  (2^(r+1) ∣ k) := by
sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l1695_169505


namespace NUMINAMATH_CALUDE_cost_of_shoes_l1695_169587

def monthly_allowance : ℕ := 5
def months_saved : ℕ := 3
def lawn_mowing_fee : ℕ := 15
def lawns_mowed : ℕ := 4
def driveway_shoveling_fee : ℕ := 7
def driveways_shoveled : ℕ := 5
def change_left : ℕ := 15

def total_saved : ℕ := 
  monthly_allowance * months_saved + 
  lawn_mowing_fee * lawns_mowed + 
  driveway_shoveling_fee * driveways_shoveled

theorem cost_of_shoes : 
  total_saved - change_left = 95 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_shoes_l1695_169587


namespace NUMINAMATH_CALUDE_unique_valid_n_l1695_169512

def is_valid (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 210 ∧ 
  (∀ k ∈ Finset.range 2013, (n + (k + 1).factorial) % 210 = 0)

theorem unique_valid_n : ∃! n : ℕ, is_valid n := by
  sorry

end NUMINAMATH_CALUDE_unique_valid_n_l1695_169512


namespace NUMINAMATH_CALUDE_greatest_x_with_lcm_l1695_169511

theorem greatest_x_with_lcm (x : ℕ) : 
  (∃ (lcm : ℕ), lcm = Nat.lcm x (Nat.lcm 15 21) ∧ lcm = 105) →
  x ≤ 105 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_x_with_lcm_l1695_169511


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_complement_B_union_A_a_range_for_C_subset_B_l1695_169543

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- State the theorems
theorem complement_intersection_A_B :
  (Set.univ : Set ℝ) \ (A ∩ B) = {x | x < 3 ∨ x ≥ 6} := by sorry

theorem complement_B_union_A :
  ((Set.univ : Set ℝ) \ B) ∪ A = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9} := by sorry

theorem a_range_for_C_subset_B :
  {a : ℝ | C a ⊆ B} = Set.Icc 2 8 := by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_complement_B_union_A_a_range_for_C_subset_B_l1695_169543


namespace NUMINAMATH_CALUDE_equation_solution_l1695_169544

theorem equation_solution : ∃ x : ℝ, x ≠ 0 ∧ 2 * ((1 / x) + (3 / x) / (6 / x)) - (1 / x) = 1.5 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1695_169544


namespace NUMINAMATH_CALUDE_greatest_number_of_baskets_l1695_169530

theorem greatest_number_of_baskets (oranges pears bananas : ℕ) 
  (h_oranges : oranges = 18) 
  (h_pears : pears = 27) 
  (h_bananas : bananas = 12) : 
  (Nat.gcd oranges (Nat.gcd pears bananas)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_greatest_number_of_baskets_l1695_169530


namespace NUMINAMATH_CALUDE_smallest_common_factor_l1695_169542

theorem smallest_common_factor (n : ℕ) : 
  (∀ m : ℕ, m < 5 → Nat.gcd (11 * m - 3) (8 * m + 4) = 1) ∧ 
  Nat.gcd (11 * 5 - 3) (8 * 5 + 4) > 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l1695_169542


namespace NUMINAMATH_CALUDE_blithe_lost_toys_l1695_169509

theorem blithe_lost_toys (initial_toys : ℕ) (found_toys : ℕ) (final_toys : ℕ) 
  (h1 : initial_toys = 40)
  (h2 : found_toys = 9)
  (h3 : final_toys = 43)
  : initial_toys - (final_toys - found_toys) = 9 := by
  sorry

end NUMINAMATH_CALUDE_blithe_lost_toys_l1695_169509


namespace NUMINAMATH_CALUDE_right_plus_acute_is_obtuse_quarter_circle_is_right_angle_l1695_169561

-- Define angles in degrees
def RightAngle : ℝ := 90
def FullCircle : ℝ := 360

-- Define angle types
def IsAcuteAngle (θ : ℝ) : Prop := 0 < θ ∧ θ < RightAngle
def IsObtuseAngle (θ : ℝ) : Prop := RightAngle < θ ∧ θ < 180

theorem right_plus_acute_is_obtuse (θ : ℝ) (h : IsAcuteAngle θ) :
  IsObtuseAngle (RightAngle + θ) := by sorry

theorem quarter_circle_is_right_angle :
  FullCircle / 4 = RightAngle := by sorry

end NUMINAMATH_CALUDE_right_plus_acute_is_obtuse_quarter_circle_is_right_angle_l1695_169561


namespace NUMINAMATH_CALUDE_product_of_roots_cubic_l1695_169588

theorem product_of_roots_cubic (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 - 15*x^2 + 75*x - 50
  ∃ a b c : ℝ, (∀ x, f x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧ a * b * c = 50 :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_cubic_l1695_169588


namespace NUMINAMATH_CALUDE_vector_AB_equals_zero_three_l1695_169558

-- Define points A and B
def A : ℝ × ℝ := (1, -1)
def B : ℝ × ℝ := (1, 2)

-- Define vector AB
def vectorAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Theorem statement
theorem vector_AB_equals_zero_three : vectorAB = (0, 3) := by
  sorry

end NUMINAMATH_CALUDE_vector_AB_equals_zero_three_l1695_169558


namespace NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l1695_169510

/-- The number of dots on each side of the square array -/
def n : ℕ := 5

/-- The number of different rectangles with sides parallel to the grid
    that can be formed by connecting four dots in an n×n square array of dots -/
def num_rectangles (n : ℕ) : ℕ :=
  (n.choose 2) * (n.choose 2)

/-- Theorem stating that the number of rectangles in a 5x5 grid is 100 -/
theorem rectangles_in_5x5_grid :
  num_rectangles n = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l1695_169510


namespace NUMINAMATH_CALUDE_combinations_equal_sixty_l1695_169528

/-- The number of paint colors available -/
def num_colors : ℕ := 5

/-- The number of painting methods available -/
def num_methods : ℕ := 4

/-- The number of pattern options available -/
def num_patterns : ℕ := 3

/-- The total number of unique combinations of color, method, and pattern -/
def total_combinations : ℕ := num_colors * num_methods * num_patterns

/-- Theorem stating that the total number of combinations is 60 -/
theorem combinations_equal_sixty : total_combinations = 60 := by
  sorry

end NUMINAMATH_CALUDE_combinations_equal_sixty_l1695_169528


namespace NUMINAMATH_CALUDE_x_value_l1695_169550

theorem x_value : ∃ x : ℚ, (3 * x + 5) / 5 = 17 ∧ x = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1695_169550


namespace NUMINAMATH_CALUDE_cab_travel_time_l1695_169540

/-- Proves that if a cab travels at 5/6 of its usual speed and arrives 6 minutes late, its usual travel time is 30 minutes. -/
theorem cab_travel_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) 
  (h2 : usual_time > 0) 
  (h3 : usual_speed * usual_time = (5/6 * usual_speed) * (usual_time + 1/10)) : 
  usual_time = 1/2 := by
sorry

end NUMINAMATH_CALUDE_cab_travel_time_l1695_169540


namespace NUMINAMATH_CALUDE_lcm_factor_proof_l1695_169545

theorem lcm_factor_proof (A B X : ℕ) : 
  A > 0 → B > 0 →
  Nat.gcd A B = 23 →
  A = 414 →
  ∃ (Y : ℕ), Nat.lcm A B = 23 * 13 * X ∧ Nat.lcm A B = 23 * 13 * Y →
  X = 18 := by sorry

end NUMINAMATH_CALUDE_lcm_factor_proof_l1695_169545


namespace NUMINAMATH_CALUDE_no_right_triangle_with_75_median_l1695_169555

theorem no_right_triangle_with_75_median (a b c : ℕ) : 
  (a * a + b * b = c * c) →  -- Pythagorean theorem
  (Nat.gcd a (Nat.gcd b c) = 1) →  -- (a, b, c) = 1
  ¬(((a * a + 4 * b * b : ℚ) / 4 = 15 * 15 / 4) ∨  -- median to leg
    (2 * a * a + 2 * b * b - c * c : ℚ) / 4 = 15 * 15 / 4)  -- median to hypotenuse
:= by sorry

end NUMINAMATH_CALUDE_no_right_triangle_with_75_median_l1695_169555


namespace NUMINAMATH_CALUDE_fudge_pan_dimension_l1695_169546

/-- Represents a rectangular pan of fudge --/
structure FudgePan where
  side1 : ℕ
  side2 : ℕ
  pieces : ℕ

/-- Theorem stating the relationship between pan dimensions and number of fudge pieces --/
theorem fudge_pan_dimension (pan : FudgePan) 
  (h1 : pan.side1 = 18)
  (h2 : pan.pieces = 522) :
  pan.side2 = 29 := by
  sorry

#check fudge_pan_dimension

end NUMINAMATH_CALUDE_fudge_pan_dimension_l1695_169546


namespace NUMINAMATH_CALUDE_hannah_cutting_speed_l1695_169571

/-- The number of strands Hannah can cut per minute -/
def hannah_strands_per_minute : ℕ := 8

/-- The total number of strands of duct tape -/
def total_strands : ℕ := 22

/-- The number of strands Hannah's son can cut per minute -/
def son_strands_per_minute : ℕ := 3

/-- The time it takes to cut all strands (in minutes) -/
def total_time : ℕ := 2

theorem hannah_cutting_speed :
  hannah_strands_per_minute = 8 ∧
  total_strands = 22 ∧
  son_strands_per_minute = 3 ∧
  total_time = 2 ∧
  total_time * (hannah_strands_per_minute + son_strands_per_minute) = total_strands :=
by sorry

end NUMINAMATH_CALUDE_hannah_cutting_speed_l1695_169571


namespace NUMINAMATH_CALUDE_expression_equality_l1695_169598

theorem expression_equality : 150 * (150 - 4) - (150 * 150 - 8 + 2^3) = -600 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1695_169598


namespace NUMINAMATH_CALUDE_convention_handshakes_theorem_l1695_169584

/-- The number of handshakes in a convention with multiple companies -/
def convention_handshakes (num_companies : ℕ) (representatives_per_company : ℕ) : ℕ :=
  let total_people := num_companies * representatives_per_company
  let handshakes_per_person := total_people - representatives_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a convention with 3 companies, each having 5 representatives,
    where each person shakes hands only once with every person except those
    from their own company, the total number of handshakes is 75. -/
theorem convention_handshakes_theorem :
  convention_handshakes 3 5 = 75 := by
  sorry

end NUMINAMATH_CALUDE_convention_handshakes_theorem_l1695_169584


namespace NUMINAMATH_CALUDE_surface_area_of_cut_solid_l1695_169506

/-- A right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  baseSideLength : ℝ

/-- Midpoints of edges in the prism -/
structure Midpoints where
  L : ℝ × ℝ × ℝ
  M : ℝ × ℝ × ℝ
  N : ℝ × ℝ × ℝ

/-- The solid formed by cutting the prism through midpoints -/
def CutSolid (p : RightPrism) (m : Midpoints) : ℝ × ℝ × ℝ × ℝ := sorry

/-- Calculate the surface area of the cut solid -/
def surfaceArea (solid : ℝ × ℝ × ℝ × ℝ) : ℝ := sorry

/-- Main theorem: The surface area of the cut solid -/
theorem surface_area_of_cut_solid (p : RightPrism) (m : Midpoints) :
  p.height = 20 ∧ p.baseSideLength = 10 →
  surfaceArea (CutSolid p m) = 50 + (25 * Real.sqrt 3) / 4 + (5 * Real.sqrt 118.75) / 2 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_cut_solid_l1695_169506


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l1695_169583

theorem smallest_solution_of_equation :
  ∀ x : ℚ, 3 * (8 * x^2 + 10 * x + 12) = x * (8 * x - 36) →
  x ≥ (-3 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l1695_169583
