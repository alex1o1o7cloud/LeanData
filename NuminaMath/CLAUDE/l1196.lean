import Mathlib

namespace NUMINAMATH_CALUDE_minimal_distance_point_l1196_119603

/-- The point that minimizes the sum of distances to two fixed points on a given line -/
theorem minimal_distance_point 
  (A B P : ℝ × ℝ) 
  (h_A : A = (-3, 1)) 
  (h_B : B = (5, -1)) 
  (h_P : P.2 = -2) : 
  (P = (3, -2)) ↔ 
  (∀ Q : ℝ × ℝ, Q.2 = -2 → 
    Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) + Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) ≤ 
    Real.sqrt ((Q.1 - A.1)^2 + (Q.2 - A.2)^2) + Real.sqrt ((Q.1 - B.1)^2 + (Q.2 - B.2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_minimal_distance_point_l1196_119603


namespace NUMINAMATH_CALUDE_oliver_fruit_consumption_l1196_119614

/-- The number of fruits Oliver consumed -/
def fruits_consumed (initial_cherries initial_strawberries initial_blueberries
                     remaining_cherries remaining_strawberries remaining_blueberries : ℝ) : ℝ :=
  (initial_cherries - remaining_cherries) +
  (initial_strawberries - remaining_strawberries) +
  (initial_blueberries - remaining_blueberries)

/-- Theorem stating that Oliver consumed 17.2 fruits in total -/
theorem oliver_fruit_consumption :
  fruits_consumed 16.5 10.7 20.2 6.3 8.4 15.5 = 17.2 := by
  sorry

end NUMINAMATH_CALUDE_oliver_fruit_consumption_l1196_119614


namespace NUMINAMATH_CALUDE_sufficient_condition_product_greater_than_one_l1196_119612

theorem sufficient_condition_product_greater_than_one :
  ∀ a b : ℝ, a > 1 → b > 1 → a * b > 1 := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_product_greater_than_one_l1196_119612


namespace NUMINAMATH_CALUDE_white_pairs_coincide_l1196_119694

/-- Represents the number of triangles of each color in each half of the figure -/
structure HalfFigure where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs when the figure is folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_white : ℕ
  white_white : ℕ

/-- The main theorem stating that 5 white pairs coincide -/
theorem white_pairs_coincide (half : HalfFigure) (pairs : CoincidingPairs) :
  half.red = 4 ∧ half.blue = 6 ∧ half.white = 10 ∧
  pairs.red_red = 3 ∧ pairs.blue_blue = 4 ∧ pairs.red_white = 3 →
  pairs.white_white = 5 := by
  sorry

end NUMINAMATH_CALUDE_white_pairs_coincide_l1196_119694


namespace NUMINAMATH_CALUDE_tan_theta_value_l1196_119642

theorem tan_theta_value (θ : Real) 
  (h1 : 2 * Real.sin θ + Real.cos θ = Real.sqrt 2 / 3)
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.tan θ = -(90 + 5 * Real.sqrt 86) / 168 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l1196_119642


namespace NUMINAMATH_CALUDE_discount_comparison_l1196_119646

def initial_amount : ℝ := 20000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def option1_discounts : List ℝ := [0.25, 0.15, 0.10]
def option2_discounts : List ℝ := [0.30, 0.10, 0.10]

def apply_successive_discounts (amount : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount amount

theorem discount_comparison :
  apply_successive_discounts initial_amount option1_discounts -
  apply_successive_discounts initial_amount option2_discounts = 135 :=
by sorry

end NUMINAMATH_CALUDE_discount_comparison_l1196_119646


namespace NUMINAMATH_CALUDE_unique_divisor_product_100_l1196_119613

/-- Product of all divisors of a natural number -/
def divisor_product (n : ℕ) : ℕ := sorry

/-- Theorem stating that 100 is the only natural number whose divisor product is 10^9 -/
theorem unique_divisor_product_100 :
  ∀ n : ℕ, divisor_product n = 10^9 ↔ n = 100 := by sorry

end NUMINAMATH_CALUDE_unique_divisor_product_100_l1196_119613


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1196_119622

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = -b / a := by sorry

theorem sum_of_roots_specific_equation :
  let x₁ := (-(-7) + Real.sqrt ((-7)^2 - 4*1*(-14))) / (2*1)
  let x₂ := (-(-7) - Real.sqrt ((-7)^2 - 4*1*(-14))) / (2*1)
  x₁ + x₂ = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1196_119622


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l1196_119652

theorem sphere_volume_ratio (S₁ S₂ S₃ V₁ V₂ V₃ : ℝ) :
  S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 →
  V₁ > 0 ∧ V₂ > 0 ∧ V₃ > 0 →
  S₂ / S₁ = 4 →
  S₃ / S₁ = 9 →
  (4 * π * (V₁ / (4/3 * π))^(2/3) = S₁) →
  (4 * π * (V₂ / (4/3 * π))^(2/3) = S₂) →
  (4 * π * (V₃ / (4/3 * π))^(2/3) = S₃) →
  V₁ + V₂ = (1/3) * V₃ := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l1196_119652


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1196_119654

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 + x*y) = f x * f y + y * f x + x * f (x + y)) :
  (∀ x : ℝ, f x = 1 - x) ∨ (∀ x : ℝ, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1196_119654


namespace NUMINAMATH_CALUDE_dividend_proof_l1196_119648

theorem dividend_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) : 
  dividend = 10918788 ∧ divisor = 12 ∧ quotient = 909899 → 
  dividend / divisor = quotient := by
  sorry

end NUMINAMATH_CALUDE_dividend_proof_l1196_119648


namespace NUMINAMATH_CALUDE_parabola_focus_l1196_119693

/-- The parabola equation: x = -1/4 * (y - 2)^2 -/
def parabola_equation (x y : ℝ) : Prop := x = -(1/4) * (y - 2)^2

/-- The focus of a parabola -/
structure Focus where
  x : ℝ
  y : ℝ

/-- Theorem: The focus of the parabola x = -1/4 * (y - 2)^2 is at (-1, 2) -/
theorem parabola_focus :
  ∃ (f : Focus), f.x = -1 ∧ f.y = 2 ∧
  ∀ (x y : ℝ), parabola_equation x y →
    (x - f.x)^2 + (y - f.y)^2 = (x + 1)^2 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l1196_119693


namespace NUMINAMATH_CALUDE_problem_solution_l1196_119664

theorem problem_solution (a b c d : ℝ) 
  (h1 : a + b - c - d = 3)
  (h2 : a * b - 3 * b * c + c * d - 3 * d * a = 4)
  (h3 : 3 * a * b - b * c + 3 * c * d - d * a = 5) :
  11 * (a - c)^2 + 17 * (b - d)^2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1196_119664


namespace NUMINAMATH_CALUDE_final_price_calculation_l1196_119674

/-- Calculate the final price of a shirt and pants after a series of price changes -/
theorem final_price_calculation (S P : ℝ) :
  let shirt_price_1 := S * 1.20
  let pants_price_1 := P * 0.90
  let combined_price_1 := shirt_price_1 + pants_price_1
  let combined_price_2 := combined_price_1 * 1.15
  let final_price := combined_price_2 * 0.95
  final_price = 1.311 * S + 0.98325 * P := by sorry

end NUMINAMATH_CALUDE_final_price_calculation_l1196_119674


namespace NUMINAMATH_CALUDE_function_domain_range_l1196_119671

/-- Given a function f(x) = √(-5 / (ax² + ax - 3)) with domain R, 
    prove that the range of values for the real number a is (-12, 0]. -/
theorem function_domain_range (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt (-5 / (a * x^2 + a * x - 3))) →
  a ∈ Set.Ioc (-12) 0 :=
by sorry

end NUMINAMATH_CALUDE_function_domain_range_l1196_119671


namespace NUMINAMATH_CALUDE_arrange_75510_eq_48_l1196_119619

/-- The number of ways to arrange the digits of 75,510 to form a 5-digit number not beginning with '0' -/
def arrange_75510 : ℕ :=
  let digits : List ℕ := [7, 5, 5, 1, 0]
  let total_digits := digits.length
  let non_zero_digits := digits.filter (· ≠ 0)
  let zero_count := total_digits - non_zero_digits.length
  let non_zero_permutations := Nat.factorial non_zero_digits.length / 
    (Nat.factorial 2 * Nat.factorial (non_zero_digits.length - 2))
  (total_digits - 1) * non_zero_permutations

theorem arrange_75510_eq_48 : arrange_75510 = 48 := by
  sorry

end NUMINAMATH_CALUDE_arrange_75510_eq_48_l1196_119619


namespace NUMINAMATH_CALUDE_pauls_remaining_books_l1196_119645

/-- Calculates the number of books remaining after a sale -/
def books_remaining (initial : ℕ) (sold : ℕ) : ℕ :=
  initial - sold

/-- Theorem: Paul's remaining books after the sale -/
theorem pauls_remaining_books :
  let initial_books : ℕ := 115
  let books_sold : ℕ := 78
  books_remaining initial_books books_sold = 37 := by
  sorry

end NUMINAMATH_CALUDE_pauls_remaining_books_l1196_119645


namespace NUMINAMATH_CALUDE_prime_4k_plus_1_properties_l1196_119696

theorem prime_4k_plus_1_properties (k : ℕ) (p : ℕ) (h_prime : Nat.Prime p) (h_form : p = 4 * k + 1) :
  (∃ x : ℤ, (x^2 + 1) % p = 0) ∧
  (∃ r₁ r₂ s₁ s₂ : ℕ,
    r₁ < Real.sqrt p ∧ r₂ < Real.sqrt p ∧ s₁ < Real.sqrt p ∧ s₂ < Real.sqrt p ∧
    (r₁ ≠ r₂ ∨ s₁ ≠ s₂) ∧
    ∃ x : ℤ, (r₁ * x + s₁) % p = (r₂ * x + s₂) % p) ∧
  (∃ r₁ r₂ s₁ s₂ : ℕ,
    r₁ < Real.sqrt p ∧ r₂ < Real.sqrt p ∧ s₁ < Real.sqrt p ∧ s₂ < Real.sqrt p ∧
    p = (r₁ - r₂)^2 + (s₁ - s₂)^2) :=
by sorry

end NUMINAMATH_CALUDE_prime_4k_plus_1_properties_l1196_119696


namespace NUMINAMATH_CALUDE_girls_in_college_l1196_119662

theorem girls_in_college (total_students : ℕ) (boys_ratio girls_ratio : ℕ) 
  (h1 : total_students = 600)
  (h2 : boys_ratio = 8)
  (h3 : girls_ratio = 4) :
  (girls_ratio * total_students) / (boys_ratio + girls_ratio) = 200 :=
sorry

end NUMINAMATH_CALUDE_girls_in_college_l1196_119662


namespace NUMINAMATH_CALUDE_like_terms_imply_equation_l1196_119698

/-- Two monomials are like terms if their variables and corresponding exponents are the same -/
def are_like_terms (m n : ℕ) : Prop :=
  m = 3 ∧ n = 1

theorem like_terms_imply_equation (m n : ℕ) :
  are_like_terms m n → m - 2*n = 1 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_equation_l1196_119698


namespace NUMINAMATH_CALUDE_sphere_cylinder_volume_difference_l1196_119623

/-- The volume of space inside a sphere and outside an inscribed right cylinder -/
theorem sphere_cylinder_volume_difference (r_sphere r_cylinder : ℝ) (h_sphere : r_sphere = 5) (h_cylinder : r_cylinder = 3) :
  ∃ (h_cylinder : ℝ),
    (4 / 3 * π * r_sphere ^ 3) - (π * r_cylinder ^ 2 * h_cylinder) = (284 / 3 : ℝ) * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_cylinder_volume_difference_l1196_119623


namespace NUMINAMATH_CALUDE_extremum_implies_slope_l1196_119692

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := (x - 2) * (x^2 + c)

-- State the theorem
theorem extremum_implies_slope (c : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f c x ≤ f c 2 ∨ f c x ≥ f c 2) →
  (deriv (f c)) 1 = -5 :=
sorry

end NUMINAMATH_CALUDE_extremum_implies_slope_l1196_119692


namespace NUMINAMATH_CALUDE_adjacent_i_probability_l1196_119631

theorem adjacent_i_probability : 
  let total_letters : ℕ := 10
  let unique_letters : ℕ := 9
  let repeated_letter : ℕ := 1
  let favorable_arrangements : ℕ := unique_letters.factorial
  let total_arrangements : ℕ := total_letters.factorial / (repeated_letter + 1).factorial
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_adjacent_i_probability_l1196_119631


namespace NUMINAMATH_CALUDE_damaged_glassware_count_l1196_119670

-- Define the constants from the problem
def total_glassware : ℕ := 1500
def undamaged_fee : ℚ := 5/2
def damaged_fee : ℕ := 3
def total_received : ℕ := 3618

-- Define the theorem
theorem damaged_glassware_count :
  ∃ x : ℕ, 
    x ≤ total_glassware ∧ 
    (undamaged_fee * (total_glassware - x) : ℚ) - (damaged_fee * x : ℚ) = total_received ∧
    x = 24 := by
  sorry

end NUMINAMATH_CALUDE_damaged_glassware_count_l1196_119670


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l1196_119636

theorem isosceles_right_triangle_area 
  (h : ℝ) -- hypotenuse length
  (is_isosceles_right : True) -- condition that the triangle is isosceles right
  (hyp_length : h = 6 * Real.sqrt 2) : -- condition for the hypotenuse length
  (1/2) * ((h / Real.sqrt 2) ^ 2) = 18 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l1196_119636


namespace NUMINAMATH_CALUDE_problem_pyramid_rows_l1196_119617

/-- Represents a pyramid display of cans -/
structure CanPyramid where
  topRowCans : ℕ
  rowIncrement : ℕ
  totalCans : ℕ

/-- Calculates the number of rows in a can pyramid -/
def numberOfRows (p : CanPyramid) : ℕ :=
  sorry

/-- The specific can pyramid from the problem -/
def problemPyramid : CanPyramid :=
  { topRowCans := 3
  , rowIncrement := 3
  , totalCans := 225 }

/-- Theorem stating that the number of rows in the problem pyramid is 12 -/
theorem problem_pyramid_rows :
  numberOfRows problemPyramid = 12 := by
  sorry

end NUMINAMATH_CALUDE_problem_pyramid_rows_l1196_119617


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l1196_119625

theorem sufficient_condition_range (a : ℝ) : 
  (∀ x : ℝ, |x + 1| ≤ 2 → x ≤ a) ∧ 
  (∃ x : ℝ, x ≤ a ∧ |x + 1| > 2) → 
  a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l1196_119625


namespace NUMINAMATH_CALUDE_rectangle_area_l1196_119672

/-- Given a rectangle with width 5 inches and length 3 times its width, prove its area is 75 square inches. -/
theorem rectangle_area (width : ℝ) (length : ℝ) (area : ℝ) : 
  width = 5 → length = 3 * width → area = length * width → area = 75 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1196_119672


namespace NUMINAMATH_CALUDE_ratio_to_eleven_l1196_119628

theorem ratio_to_eleven : ∃ x : ℚ, (5 : ℚ) / 1 = x / 11 ∧ x = 55 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_eleven_l1196_119628


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_when_m_3_A_union_B_equals_A_iff_m_in_range_l1196_119604

def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 5}
def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem complement_A_intersect_B_when_m_3 :
  (Set.univ \ A) ∩ B 3 = {5} := by sorry

theorem A_union_B_equals_A_iff_m_in_range (m : ℝ) :
  A ∪ B m = A ↔ m < 3 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_when_m_3_A_union_B_equals_A_iff_m_in_range_l1196_119604


namespace NUMINAMATH_CALUDE_interior_angles_sum_increase_l1196_119609

/-- The sum of interior angles of a convex polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

theorem interior_angles_sum_increase {n : ℕ} (h : sum_interior_angles n = 1800) :
  sum_interior_angles (n + 2) = 2160 := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_sum_increase_l1196_119609


namespace NUMINAMATH_CALUDE_min_sum_squares_l1196_119647

theorem min_sum_squares (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) :
  ∃ (min : ℝ), min = 1 ∧ ∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → a^2 + b^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1196_119647


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l1196_119620

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2033 = i := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l1196_119620


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l1196_119676

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l1196_119676


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l1196_119629

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a - Complex.I) / (2 + Complex.I) = Complex.I * b) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l1196_119629


namespace NUMINAMATH_CALUDE_land_properties_l1196_119641

/-- Represents a piece of land with specific measurements -/
structure Land where
  triangle_area : ℝ
  ac_length : ℝ
  cd_length : ℝ
  de_length : ℝ

/-- Calculates the total area of the land -/
def total_area (land : Land) : ℝ := sorry

/-- Calculates the length of CF to divide the land equally -/
def equal_division_length (land : Land) : ℝ := sorry

theorem land_properties (land : Land) 
  (h1 : land.triangle_area = 120)
  (h2 : land.ac_length = 20)
  (h3 : land.cd_length = 10)
  (h4 : land.de_length = 10) :
  total_area land = 270 ∧ equal_division_length land = 1.5 := by sorry

end NUMINAMATH_CALUDE_land_properties_l1196_119641


namespace NUMINAMATH_CALUDE_investor_share_price_l1196_119667

theorem investor_share_price 
  (dividend_rate : ℝ) 
  (face_value : ℝ) 
  (roi : ℝ) 
  (h1 : dividend_rate = 0.125)
  (h2 : face_value = 40)
  (h3 : roi = 0.25) :
  let dividend_per_share := dividend_rate * face_value
  let price := dividend_per_share / roi
  price = 20 := by sorry

end NUMINAMATH_CALUDE_investor_share_price_l1196_119667


namespace NUMINAMATH_CALUDE_simplest_fraction_of_0_63575_l1196_119626

theorem simplest_fraction_of_0_63575 :
  ∃ (a b : ℕ+), (a.val : ℚ) / b.val = 63575 / 100000 ∧
  ∀ (c d : ℕ+), (c.val : ℚ) / d.val = 63575 / 100000 → a.val ≤ c.val ∧ b.val ≤ d.val →
  a = 2543 ∧ b = 4000 := by
sorry

end NUMINAMATH_CALUDE_simplest_fraction_of_0_63575_l1196_119626


namespace NUMINAMATH_CALUDE_pipe_length_difference_l1196_119615

theorem pipe_length_difference (total_length shorter_length : ℕ) 
  (h1 : total_length = 68)
  (h2 : shorter_length = 28)
  (h3 : shorter_length < total_length - shorter_length) :
  total_length - shorter_length - shorter_length = 12 :=
by sorry

end NUMINAMATH_CALUDE_pipe_length_difference_l1196_119615


namespace NUMINAMATH_CALUDE_overall_loss_calculation_l1196_119658

def stock_worth : ℝ := 15000

def profit_percentage : ℝ := 0.1
def loss_percentage : ℝ := 0.05

def profit_stock_ratio : ℝ := 0.2
def loss_stock_ratio : ℝ := 0.8

def profit_amount : ℝ := stock_worth * profit_stock_ratio * profit_percentage
def loss_amount : ℝ := stock_worth * loss_stock_ratio * loss_percentage

def overall_selling_price : ℝ := 
  (stock_worth * profit_stock_ratio * (1 + profit_percentage)) +
  (stock_worth * loss_stock_ratio * (1 - loss_percentage))

def overall_loss : ℝ := stock_worth - overall_selling_price

theorem overall_loss_calculation :
  overall_loss = 300 :=
sorry

end NUMINAMATH_CALUDE_overall_loss_calculation_l1196_119658


namespace NUMINAMATH_CALUDE_range_of_a_l1196_119684

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ioo 0 1 → a ≤ x^2 - 4*x) → a ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1196_119684


namespace NUMINAMATH_CALUDE_x4_plus_y4_equals_47_l1196_119688

theorem x4_plus_y4_equals_47 (x y : ℝ) 
  (h1 : x^2 + 1/x^2 = 7) 
  (h2 : x*y = 1) : 
  x^4 + y^4 = 47 := by
sorry

end NUMINAMATH_CALUDE_x4_plus_y4_equals_47_l1196_119688


namespace NUMINAMATH_CALUDE_reporter_earnings_per_hour_l1196_119630

/-- Calculate reporter's earnings per hour given their pay rate and work conditions --/
theorem reporter_earnings_per_hour 
  (words_per_minute : ℕ)
  (pay_per_word : ℚ)
  (pay_per_article : ℕ)
  (num_articles : ℕ)
  (total_hours : ℕ)
  (h1 : words_per_minute = 10)
  (h2 : pay_per_word = 1/10)
  (h3 : pay_per_article = 60)
  (h4 : num_articles = 3)
  (h5 : total_hours = 4) :
  (words_per_minute * 60 * total_hours : ℚ) * pay_per_word + 
  (num_articles * pay_per_article : ℚ) / total_hours = 105 := by
  sorry

#eval (10 * 60 * 4 : ℚ) * (1/10) + (3 * 60 : ℚ) / 4

end NUMINAMATH_CALUDE_reporter_earnings_per_hour_l1196_119630


namespace NUMINAMATH_CALUDE_min_max_theorem_l1196_119638

theorem min_max_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (1/x + 1/y ≥ 2) ∧ (x * (y + 1) ≤ 9/4) := by
  sorry

end NUMINAMATH_CALUDE_min_max_theorem_l1196_119638


namespace NUMINAMATH_CALUDE_ratio_inequality_l1196_119627

theorem ratio_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / b + b / c + c / a ≤ a^2 / b^2 + b^2 / c^2 + c^2 / a^2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_inequality_l1196_119627


namespace NUMINAMATH_CALUDE_ad_greater_bc_l1196_119659

theorem ad_greater_bc (a b c d : ℝ) 
  (sum_eq : a + d = b + c) 
  (abs_ineq : |a - d| < |b - c|) : 
  a * d > b * c := by
  sorry

end NUMINAMATH_CALUDE_ad_greater_bc_l1196_119659


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1196_119697

/-- A hyperbola with foci on the x-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  hpos : 0 < a ∧ 0 < b
  hc : c = Real.sqrt 5
  hasymptote : b / a = 1 / 2

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) : Prop :=
  ∀ x y : ℝ, x^2 / 4 - y^2 = 1 ↔ x^2 / h.a^2 - y^2 / h.b^2 = 1

theorem hyperbola_equation (h : Hyperbola) : standard_equation h := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1196_119697


namespace NUMINAMATH_CALUDE_tommys_profit_l1196_119695

/-- Represents a type of crate --/
structure Crate where
  capacity : ℕ
  quantity : ℕ
  cost : ℕ
  rotten : ℕ
  price : ℕ

/-- Calculates the profit from selling tomatoes --/
def calculateProfit (crateA crateB crateC : Crate) : ℕ :=
  let totalCost := crateA.cost + crateB.cost + crateC.cost
  let revenueA := (crateA.capacity * crateA.quantity - crateA.rotten) * crateA.price
  let revenueB := (crateB.capacity * crateB.quantity - crateB.rotten) * crateB.price
  let revenueC := (crateC.capacity * crateC.quantity - crateC.rotten) * crateC.price
  let totalRevenue := revenueA + revenueB + revenueC
  totalRevenue - totalCost

/-- Tommy's profit from selling tomatoes is $14 --/
theorem tommys_profit :
  let crateA : Crate := ⟨20, 2, 220, 4, 5⟩
  let crateB : Crate := ⟨25, 3, 375, 5, 6⟩
  let crateC : Crate := ⟨30, 1, 180, 3, 7⟩
  calculateProfit crateA crateB crateC = 14 := by
  sorry


end NUMINAMATH_CALUDE_tommys_profit_l1196_119695


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l1196_119639

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (∃ (k : ℤ), (12*x + 3) * (12*x + 9) * (6*x + 6) = 324 * k) ∧
  (∀ (m : ℤ), m > 324 → ¬(∀ (x : ℤ), Odd x → ∃ (k : ℤ), (12*x + 3) * (12*x + 9) * (6*x + 6) = m * k)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l1196_119639


namespace NUMINAMATH_CALUDE_brown_eyed_brunettes_count_l1196_119689

/-- Represents the number of girls with specific characteristics -/
structure GirlCount where
  total : ℕ
  greenEyedBlondes : ℕ
  brunettes : ℕ
  brownEyed : ℕ

/-- Calculates the number of brown-eyed brunettes given the counts of girls with specific characteristics -/
def brownEyedBrunettes (gc : GirlCount) : ℕ :=
  gc.brownEyed - (gc.total - gc.brunettes - gc.greenEyedBlondes)

/-- Theorem stating that given the specific counts, there are 20 brown-eyed brunettes -/
theorem brown_eyed_brunettes_count (gc : GirlCount) 
  (h1 : gc.total = 60)
  (h2 : gc.greenEyedBlondes = 20)
  (h3 : gc.brunettes = 35)
  (h4 : gc.brownEyed = 25) :
  brownEyedBrunettes gc = 20 := by
  sorry

#eval brownEyedBrunettes { total := 60, greenEyedBlondes := 20, brunettes := 35, brownEyed := 25 }

end NUMINAMATH_CALUDE_brown_eyed_brunettes_count_l1196_119689


namespace NUMINAMATH_CALUDE_sqrt_1001_irreducible_l1196_119637

theorem sqrt_1001_irreducible : ∀ a b : ℕ, a * a = 1001 * (b * b) → a = 1001 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_1001_irreducible_l1196_119637


namespace NUMINAMATH_CALUDE_unpainted_cubes_in_6x6x6_l1196_119682

/-- Represents a cube with painted rows -/
structure PaintedCube where
  size : Nat
  totalUnitCubes : Nat
  paintedRowsPerFace : Nat

/-- Calculates the number of unpainted unit cubes in a painted cube -/
def unpaintedUnitCubes (cube : PaintedCube) : Nat :=
  cube.totalUnitCubes - (cube.size * cube.size * 6 - (cube.size - 2) * (cube.size - 2) * 6)

/-- Theorem stating that a 6x6x6 cube with two central rows painted on each face has 108 unpainted unit cubes -/
theorem unpainted_cubes_in_6x6x6 :
  let cube : PaintedCube := { size := 6, totalUnitCubes := 216, paintedRowsPerFace := 2 }
  unpaintedUnitCubes cube = 108 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_in_6x6x6_l1196_119682


namespace NUMINAMATH_CALUDE_seat_swapping_arrangements_l1196_119633

def number_of_students : ℕ := 7
def students_to_swap : ℕ := 3

theorem seat_swapping_arrangements :
  (number_of_students.choose students_to_swap) * (students_to_swap.factorial) = 70 := by
  sorry

end NUMINAMATH_CALUDE_seat_swapping_arrangements_l1196_119633


namespace NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l1196_119678

theorem smallest_integer_with_given_remainders : ∃ n : ℕ,
  n > 0 ∧
  n % 3 = 2 ∧
  n % 5 = 4 ∧
  n % 7 = 6 ∧
  n % 11 = 10 ∧
  ∀ m : ℕ, m > 0 ∧ m % 3 = 2 ∧ m % 5 = 4 ∧ m % 7 = 6 ∧ m % 11 = 10 → n ≤ m :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l1196_119678


namespace NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l1196_119660

theorem trigonometric_expression_evaluation :
  (Real.cos (40 * π / 180) + Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180))) /
  (Real.sin (70 * π / 180) * Real.sqrt (1 + Real.cos (40 * π / 180))) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l1196_119660


namespace NUMINAMATH_CALUDE_range_of_b_l1196_119661

theorem range_of_b (b : ℝ) : 
  (∀ a : ℝ, a ≤ -1 → a * 2 * b - b - 3 * a ≥ 0) → 
  b ∈ Set.Iic 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_b_l1196_119661


namespace NUMINAMATH_CALUDE_curve_symmetry_l1196_119618

-- Define the original curve E
def E (x y : ℝ) : Prop := 2 * x^2 + 4 * x * y + 5 * y^2 - 22 = 0

-- Define the line of symmetry l
def l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the symmetric curve E'
def E' (x y : ℝ) : Prop := 5 * x^2 + 4 * x * y + 2 * y^2 + 6 * x - 19 = 0

-- Theorem statement
theorem curve_symmetry :
  ∀ (x y : ℝ), E x y ↔ ∃ (x' y' : ℝ), l ((x + x') / 2) ((y + y') / 2) ∧ E' x' y' :=
sorry

end NUMINAMATH_CALUDE_curve_symmetry_l1196_119618


namespace NUMINAMATH_CALUDE_increase_by_fifty_percent_l1196_119691

theorem increase_by_fifty_percent : 
  let initial : ℝ := 100
  let percentage : ℝ := 50
  let increase : ℝ := initial * (percentage / 100)
  let final : ℝ := initial + increase
  final = 150
  := by sorry

end NUMINAMATH_CALUDE_increase_by_fifty_percent_l1196_119691


namespace NUMINAMATH_CALUDE_outfit_combinations_l1196_119677

theorem outfit_combinations (n : ℕ) (h : n = 7) : n^3 - n = 336 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l1196_119677


namespace NUMINAMATH_CALUDE_coin_value_theorem_l1196_119699

theorem coin_value_theorem (n d : ℕ) : 
  n + d = 25 →
  (10 * n + 5 * d) - (5 * n + 10 * d) = 100 →
  5 * n + 10 * d = 140 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_theorem_l1196_119699


namespace NUMINAMATH_CALUDE_square_sum_theorem_l1196_119675

theorem square_sum_theorem (a b : ℕ) 
  (h1 : ∃ k : ℕ, a * b = k^2)
  (h2 : ∃ m : ℕ, (2 * a + 1) * (2 * b + 1) = m^2) :
  ∃ n : ℕ, 
    n % 2 = 0 ∧ 
    n > 2 ∧ 
    ∃ p : ℕ, (a + n) * (b + n) = p^2 :=
by sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l1196_119675


namespace NUMINAMATH_CALUDE_mixture_weight_l1196_119607

/-- Given a mixture of zinc, copper, and silver in the ratio 9 : 11 : 7,
    where 27 kg of zinc is used, the total weight of the mixture is 81 kg. -/
theorem mixture_weight (zinc copper silver : ℕ) (zinc_weight : ℝ) :
  zinc = 9 →
  copper = 11 →
  silver = 7 →
  zinc_weight = 27 →
  (zinc_weight / zinc) * (zinc + copper + silver) = 81 :=
by sorry

end NUMINAMATH_CALUDE_mixture_weight_l1196_119607


namespace NUMINAMATH_CALUDE_mary_garden_potatoes_l1196_119611

/-- The number of potatoes left in Mary's garden after planting and rabbit eating -/
def potatoes_left (initial : ℕ) (added : ℕ) (eaten : ℕ) : ℕ :=
  let rows := initial
  let per_row := 1 + added
  max (rows * per_row - rows * eaten) 0

theorem mary_garden_potatoes :
  potatoes_left 8 2 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_mary_garden_potatoes_l1196_119611


namespace NUMINAMATH_CALUDE_special_polynomial_zeros_l1196_119690

/-- A polynomial of degree 5 with specific properties -/
def SpecialPolynomial (P : ℂ → ℂ) : Prop :=
  ∃ (r s : ℤ) (a b : ℤ),
    (∀ x, P x = x * (x - r) * (x - s) * (x^2 + a*x + b)) ∧
    (∀ x, ∃ (c : ℤ), P x = c)

theorem special_polynomial_zeros (P : ℂ → ℂ) (h : SpecialPolynomial P) :
  P ((1 + Complex.I * Real.sqrt 15) / 2) = 0 ∧
  P ((1 + Complex.I * Real.sqrt 17) / 2) = 0 :=
sorry

end NUMINAMATH_CALUDE_special_polynomial_zeros_l1196_119690


namespace NUMINAMATH_CALUDE_function_divisibility_property_l1196_119608

theorem function_divisibility_property (f : ℕ → ℕ) :
  (∀ x y : ℕ, (f x + f y) ∣ (x^2 - y^2)) →
  ∀ n : ℕ, f n = n :=
by sorry

end NUMINAMATH_CALUDE_function_divisibility_property_l1196_119608


namespace NUMINAMATH_CALUDE_bryson_new_shoes_l1196_119606

/-- Proves that buying 2 pairs of shoes results in 4 new shoes -/
theorem bryson_new_shoes : 
  ∀ (pairs_bought : ℕ) (shoes_per_pair : ℕ),
  pairs_bought = 2 → shoes_per_pair = 2 →
  pairs_bought * shoes_per_pair = 4 := by
  sorry

end NUMINAMATH_CALUDE_bryson_new_shoes_l1196_119606


namespace NUMINAMATH_CALUDE_x_value_when_one_in_set_l1196_119632

theorem x_value_when_one_in_set (x : ℝ) : 
  (1 ∈ ({x, x^2} : Set ℝ)) → x ≠ x^2 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_one_in_set_l1196_119632


namespace NUMINAMATH_CALUDE_log_20_over_27_not_calculable_l1196_119657

-- Define the given logarithms
def log5 : ℝ := 0.6990
def log3 : ℝ := 0.4771

-- Define a function to represent the ability to calculate a logarithm
def can_calculate (x : ℝ) : Prop := 
  ∃ (f : ℝ → ℝ → ℝ), x = f log5 log3

-- Theorem statement
theorem log_20_over_27_not_calculable :
  ¬(can_calculate (Real.log (20/27))) ∧
  (can_calculate (Real.log 225)) ∧
  (can_calculate (Real.log 750)) ∧
  (can_calculate (Real.log 0.03)) ∧
  (can_calculate (Real.log 9)) :=
sorry

end NUMINAMATH_CALUDE_log_20_over_27_not_calculable_l1196_119657


namespace NUMINAMATH_CALUDE_probability_of_selection_l1196_119653

/-- Given a group of students where each student has an equal chance of being selected as the group leader,
    prove that the probability of a specific student (Xiao Li) being chosen is 1/5. -/
theorem probability_of_selection (total_students : ℕ) (xiao_li : Fin total_students) :
  total_students = 5 →
  (∀ (student : Fin total_students), ℚ) →
  (∃! (prob : Fin total_students → ℚ), ∀ (student : Fin total_students), prob student = 1 / total_students) →
  (∃ (prob : Fin total_students → ℚ), prob xiao_li = 1 / 5) :=
by sorry

end NUMINAMATH_CALUDE_probability_of_selection_l1196_119653


namespace NUMINAMATH_CALUDE_parabola_tangent_and_intersecting_line_l1196_119668

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the tangent line passing through (-1, 0)
def tangent_line (x y : ℝ) : Prop := ∃ t : ℝ, x = t*y - 1

-- Define the point P in the first quadrant
def point_P (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ parabola x y ∧ tangent_line x y

-- Define the line l passing through (2, 0)
def line_l (x y : ℝ) : Prop := ∃ m : ℝ, x = m*y + 2

-- Define the circle M with AB as diameter passing through P
def circle_M (xa ya xb yb : ℝ) : Prop :=
  ∃ xc yc : ℝ, (xc - 1)^2 + (yc - 2)^2 = ((xa - xb)^2 + (ya - yb)^2) / 4

theorem parabola_tangent_and_intersecting_line :
  -- Part 1: Point of tangency P
  (∃! x y : ℝ, point_P x y ∧ x = 1 ∧ y = 2) ∧
  -- Part 2: Equation of line l
  (∀ xa ya xb yb : ℝ,
    parabola xa ya ∧ parabola xb yb ∧
    line_l xa ya ∧ line_l xb yb ∧
    circle_M xa ya xb yb →
    ∃ m : ℝ, m = -2/3 ∧ ∀ x y : ℝ, line_l x y ↔ y = m*x + 4/3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_and_intersecting_line_l1196_119668


namespace NUMINAMATH_CALUDE_ram_selection_probability_l1196_119663

theorem ram_selection_probability
  (p_ravi : ℝ)
  (p_both : ℝ)
  (h_ravi : p_ravi = 1 / 5)
  (h_both : p_both = 0.05714285714285714)
  (h_independent : ∀ p_ram : ℝ, p_both = p_ram * p_ravi) :
  ∃ p_ram : ℝ, p_ram = 2 / 7 :=
by sorry

end NUMINAMATH_CALUDE_ram_selection_probability_l1196_119663


namespace NUMINAMATH_CALUDE_mean_of_middle_numbers_l1196_119685

theorem mean_of_middle_numbers (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 90 →
  max a (max b (max c d)) = 105 →
  min a (min b (min c d)) = 75 →
  (a + b + c + d - 105 - 75) / 2 = 90 := by
sorry

end NUMINAMATH_CALUDE_mean_of_middle_numbers_l1196_119685


namespace NUMINAMATH_CALUDE_crazy_silly_school_movies_l1196_119649

theorem crazy_silly_school_movies :
  let number_of_books : ℕ := 8
  let movies_more_than_books : ℕ := 2
  let number_of_movies : ℕ := number_of_books + movies_more_than_books
  number_of_movies = 10 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_movies_l1196_119649


namespace NUMINAMATH_CALUDE_fraction_simplification_l1196_119640

theorem fraction_simplification (x : ℝ) : 
  (x^2 + 2*x + 3) / 4 + (3*x - 5) / 6 = (3*x^2 + 12*x - 1) / 12 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1196_119640


namespace NUMINAMATH_CALUDE_problem_solution_l1196_119687

theorem problem_solution : 45 / (7 - 3/4) = 36/5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1196_119687


namespace NUMINAMATH_CALUDE_smallest_side_of_triangle_l1196_119610

/-- Given a triangle with specific properties, prove its smallest side length -/
theorem smallest_side_of_triangle (S : ℝ) (p : ℝ) (d : ℝ) :
  S = 6 * Real.sqrt 6 →
  p = 18 →
  d = (2 * Real.sqrt 42) / 3 →
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = p ∧
    S = Real.sqrt (p/2 * (p/2 - a) * (p/2 - b) * (p/2 - c)) ∧
    d^2 = ((p/2 - b) * (p/2 - c) / (p/2))^2 + (S / p)^2 ∧
    min a (min b c) = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_side_of_triangle_l1196_119610


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1196_119656

theorem simplify_and_evaluate (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 1) :
  (1 - 1/a) / ((a^2 - 2*a + 1) / a) = 1 / (a - 1) ∧
  (1 - 1/2) / ((2^2 - 2*2 + 1) / 2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1196_119656


namespace NUMINAMATH_CALUDE_yoongi_age_l1196_119655

theorem yoongi_age (hoseok_age yoongi_age : ℕ) 
  (h1 : yoongi_age = hoseok_age - 2)
  (h2 : yoongi_age + hoseok_age = 18) :
  yoongi_age = 8 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_age_l1196_119655


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1196_119616

theorem cos_alpha_value (α : Real) 
  (h1 : Real.sin (π/6 + α) = 3/5)
  (h2 : π/3 < α ∧ α < 5*π/6) : 
  Real.cos α = (3 - 4 * Real.sqrt 3) / 10 := by sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1196_119616


namespace NUMINAMATH_CALUDE_toy_cost_price_l1196_119679

/-- The cost price of one toy -/
def cost_price : ℕ := sorry

/-- The selling price of 18 toys -/
def selling_price : ℕ := 18900

/-- The number of toys sold -/
def toys_sold : ℕ := 18

/-- The number of toys whose cost price equals the gain -/
def gain_toys : ℕ := 3

theorem toy_cost_price : 
  (toys_sold + gain_toys) * cost_price = selling_price → 
  cost_price = 900 := by sorry

end NUMINAMATH_CALUDE_toy_cost_price_l1196_119679


namespace NUMINAMATH_CALUDE_total_cost_of_supplies_l1196_119634

/-- Calculates the total cost of supplies for a class project -/
theorem total_cost_of_supplies (num_students : ℕ) 
  (bow_cost vinegar_cost baking_soda_cost : ℕ) : 
  num_students = 23 → 
  bow_cost = 5 → 
  vinegar_cost = 2 → 
  baking_soda_cost = 1 → 
  num_students * (bow_cost + vinegar_cost + baking_soda_cost) = 184 := by
  sorry

#check total_cost_of_supplies

end NUMINAMATH_CALUDE_total_cost_of_supplies_l1196_119634


namespace NUMINAMATH_CALUDE_sin_half_angle_second_quadrant_l1196_119669

theorem sin_half_angle_second_quadrant (θ : Real) 
  (h1 : π < θ ∧ θ < 3*π/2) 
  (h2 : 25 * Real.sin θ ^ 2 + Real.sin θ - 24 = 0) : 
  Real.sin (θ/2) = 4/5 ∨ Real.sin (θ/2) = -4/5 := by
sorry

end NUMINAMATH_CALUDE_sin_half_angle_second_quadrant_l1196_119669


namespace NUMINAMATH_CALUDE_triangle_side_length_l1196_119600

theorem triangle_side_length (a b c : ℝ) (A : ℝ) : 
  a = 7 → b = 8 → A = π/3 → (c = 3 ∨ c = 5) → 
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1196_119600


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1196_119621

/-- Given that M(5,5) is the midpoint of line segment CD and C has coordinates (10,10),
    prove that the sum of the coordinates of point D is 0. -/
theorem midpoint_coordinate_sum (C D M : ℝ × ℝ) : 
  M = (5, 5) → 
  C = (10, 10) → 
  M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 + D.2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1196_119621


namespace NUMINAMATH_CALUDE_be_length_is_fourth_root_three_l1196_119624

/-- A rhombus with specific properties and internal rectangles -/
structure SpecialRhombus where
  -- The side length of the rhombus
  side_length : ℝ
  -- The length of one diagonal of the rhombus
  diagonal_length : ℝ
  -- The length of the side BE of the internal rectangle EBCF
  be_length : ℝ
  -- The side length is 2
  side_length_eq : side_length = 2
  -- One diagonal measures 2√3
  diagonal_eq : diagonal_length = 2 * Real.sqrt 3
  -- EBCF is a square (implied by equal sides along BC and BE)
  ebcf_square : be_length * be_length = be_length * be_length
  -- Area of EBCF + Area of JKHG = Area of rhombus (since they are congruent and fit within the rhombus)
  area_eq : 2 * (be_length * be_length) = (1 / 2) * diagonal_length * side_length

/-- The length of BE in the special rhombus is ∜3 -/
theorem be_length_is_fourth_root_three (r : SpecialRhombus) : r.be_length = Real.sqrt (Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_be_length_is_fourth_root_three_l1196_119624


namespace NUMINAMATH_CALUDE_product_of_fractions_equals_81_l1196_119651

theorem product_of_fractions_equals_81 : 
  (1 / 3) * (9 / 1) * (1 / 27) * (81 / 1) * (1 / 243) * (729 / 1) * (1 / 2187) * (6561 / 1) = 81 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_equals_81_l1196_119651


namespace NUMINAMATH_CALUDE_no_parallelogram_polyhedron_1992_faces_l1196_119635

/-- A convex polyhedron with parallelogram faces -/
structure ParallelogramPolyhedron where
  faces : ℕ
  is_convex : Bool
  all_faces_parallelograms : Bool

/-- Theorem: There does not exist a convex polyhedron with all faces as parallelograms and exactly 1992 faces -/
theorem no_parallelogram_polyhedron_1992_faces :
  ¬ ∃ (P : ParallelogramPolyhedron), P.faces = 1992 ∧ P.is_convex ∧ P.all_faces_parallelograms :=
by sorry

end NUMINAMATH_CALUDE_no_parallelogram_polyhedron_1992_faces_l1196_119635


namespace NUMINAMATH_CALUDE_range_of_a_l1196_119666

theorem range_of_a (a : ℝ) : (∃ x : ℝ, Real.sqrt (3 * x + 6) + Real.sqrt (14 - x) > a) → a < 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1196_119666


namespace NUMINAMATH_CALUDE_number_line_problem_l1196_119643

theorem number_line_problem (a b c : ℚ) : 
  a = (-4)^2 - 8 →
  b = -c →
  |c - a| = 3 →
  ((b = -5 ∧ c = 5) ∨ (b = -11 ∧ c = 11)) ∧
  (-a^2 + b - c = -74 ∨ -a^2 + b - c = -86) :=
by sorry

end NUMINAMATH_CALUDE_number_line_problem_l1196_119643


namespace NUMINAMATH_CALUDE_mikey_jelly_beans_mikey_jelly_beans_holds_l1196_119602

/-- Proves that Mikey has 19 jelly beans given the conditions of the problem -/
theorem mikey_jelly_beans : ℕ → ℕ → ℕ → Prop :=
  fun napoleon sedrich mikey =>
    napoleon = 17 →
    sedrich = napoleon + 4 →
    2 * (napoleon + sedrich) = 4 * mikey →
    mikey = 19

/-- The theorem holds for the given values -/
theorem mikey_jelly_beans_holds : 
  ∃ (napoleon sedrich mikey : ℕ), mikey_jelly_beans napoleon sedrich mikey :=
by
  sorry

end NUMINAMATH_CALUDE_mikey_jelly_beans_mikey_jelly_beans_holds_l1196_119602


namespace NUMINAMATH_CALUDE_solve_equation_l1196_119601

theorem solve_equation : ∃ x : ℝ, 8 * x - (5 * 0.85 / 2.5) = 5.5 ∧ x = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1196_119601


namespace NUMINAMATH_CALUDE_tan70_cos10_sqrt3tan20_minus1_equals_negative1_l1196_119681

theorem tan70_cos10_sqrt3tan20_minus1_equals_negative1 :
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan70_cos10_sqrt3tan20_minus1_equals_negative1_l1196_119681


namespace NUMINAMATH_CALUDE_system_solution_l1196_119605

theorem system_solution (x y : ℝ) : 
  ((x = 2 ∧ y = 2) ∨ (x = 2 ∧ y = 4) ∨ (x = (Real.sqrt 17 - 1) / 2 ∧ y = (9 - Real.sqrt 17) / 2)) →
  (((x^2 * y^4)^(-Real.log x) = y^(Real.log (y / x^7))) ∧
   (y^2 - x*y - 2*x^2 + 8*x - 4*y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1196_119605


namespace NUMINAMATH_CALUDE_vacation_pictures_l1196_119665

theorem vacation_pictures (zoo museum amusement beach deleted : ℕ) 
  (h1 : zoo = 120)
  (h2 : museum = 34)
  (h3 : amusement = 25)
  (h4 : beach = 21)
  (h5 : deleted = 73) :
  zoo + museum + amusement + beach - deleted = 127 := by
  sorry

end NUMINAMATH_CALUDE_vacation_pictures_l1196_119665


namespace NUMINAMATH_CALUDE_P_intersect_Q_l1196_119650

def P : Set ℝ := {x | x^2 - 16 < 0}
def Q : Set ℝ := {x | ∃ n : ℤ, x = 2 * ↑n}

theorem P_intersect_Q : P ∩ Q = {-2, 0, 2} := by sorry

end NUMINAMATH_CALUDE_P_intersect_Q_l1196_119650


namespace NUMINAMATH_CALUDE_polynomial_equality_l1196_119644

theorem polynomial_equality (a b c d e : ℝ) :
  (∀ x : ℝ, (x - 3)^4 = a*x^4 + b*x^3 + c*x^2 + d*x + e) →
  b + c + d + e = 15 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1196_119644


namespace NUMINAMATH_CALUDE_unknown_number_exists_l1196_119680

theorem unknown_number_exists : ∃ x : ℝ, 
  (0.15 : ℝ)^3 - (0.06 : ℝ)^3 / (0.15 : ℝ)^2 + x + (0.06 : ℝ)^2 = 0.08999999999999998 ∧ 
  abs (x - 0.092625) < 0.000001 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_exists_l1196_119680


namespace NUMINAMATH_CALUDE_cloth_sale_worth_l1196_119683

/-- Represents the worth of cloth sold given a commission rate and amount -/
def worthOfClothSold (commissionRate : ℚ) (commissionAmount : ℚ) : ℚ :=
  commissionAmount / (commissionRate / 100)

/-- Theorem stating that given a 4% commission rate and Rs. 12.50 commission,
    the worth of cloth sold is Rs. 312.50 -/
theorem cloth_sale_worth :
  worthOfClothSold (4 : ℚ) (25 / 2) = 625 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_worth_l1196_119683


namespace NUMINAMATH_CALUDE_distance_walked_l1196_119686

-- Define the walking time in hours
def walking_time : ℝ := 1.25

-- Define the walking rate in miles per hour
def walking_rate : ℝ := 4.8

-- Theorem statement
theorem distance_walked : walking_time * walking_rate = 6 := by
  sorry

end NUMINAMATH_CALUDE_distance_walked_l1196_119686


namespace NUMINAMATH_CALUDE_range_of_k_for_two_distinct_roots_l1196_119673

/-- The quadratic equation (k-1)x^2 + 2x - 2 = 0 has two distinct real roots -/
def has_two_distinct_real_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (k - 1) * x₁^2 + 2 * x₁ - 2 = 0 ∧ (k - 1) * x₂^2 + 2 * x₂ - 2 = 0

/-- The range of k values for which the quadratic equation has two distinct real roots -/
theorem range_of_k_for_two_distinct_roots :
  ∀ k : ℝ, has_two_distinct_real_roots k ↔ k > 1/2 ∧ k ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_k_for_two_distinct_roots_l1196_119673
