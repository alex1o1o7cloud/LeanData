import Mathlib

namespace NUMINAMATH_CALUDE_initial_rope_length_l1948_194897

theorem initial_rope_length (additional_area : Real) : 
  additional_area = 942.8571428571429 → 
  ∃ (r : Real), π * (20^2 - r^2) = additional_area ∧ r = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_rope_length_l1948_194897


namespace NUMINAMATH_CALUDE_order_cost_proof_l1948_194874

def english_book_cost : ℝ := 7.50
def geography_book_cost : ℝ := 10.50
def num_books : ℕ := 35

def total_cost : ℝ := num_books * english_book_cost + num_books * geography_book_cost

theorem order_cost_proof : total_cost = 630 := by
  sorry

end NUMINAMATH_CALUDE_order_cost_proof_l1948_194874


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1948_194850

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem arithmetic_sequence_ratio (a₁ d : ℝ) :
  a₁ ≠ 0 →
  d ≠ 0 →
  (arithmetic_sequence a₁ d 2) * (arithmetic_sequence a₁ d 8) = (arithmetic_sequence a₁ d 4)^2 →
  (arithmetic_sequence a₁ d 3 + arithmetic_sequence a₁ d 6 + arithmetic_sequence a₁ d 9) /
  (arithmetic_sequence a₁ d 4 + arithmetic_sequence a₁ d 5) = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1948_194850


namespace NUMINAMATH_CALUDE_parabola_intersection_length_l1948_194838

/-- Given a parabola y = x^2 - mx - 3 that intersects the x-axis at points A and B,
    where m is an integer, the length of AB is 4. -/
theorem parabola_intersection_length (m : ℤ) (A B : ℝ) : 
  (∃ x : ℝ, x^2 - m*x - 3 = 0) → 
  (A^2 - m*A - 3 = 0) → 
  (B^2 - m*B - 3 = 0) → 
  |A - B| = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_intersection_length_l1948_194838


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l1948_194846

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l1948_194846


namespace NUMINAMATH_CALUDE_area_ratio_eab_abcd_l1948_194876

/-- Represents a trapezoid ABCD with an extended triangle EAB -/
structure ExtendedTrapezoid where
  /-- Length of base AB -/
  ab : ℝ
  /-- Length of base CD -/
  cd : ℝ
  /-- Height of the trapezoid -/
  h : ℝ
  /-- Assertion that AB = 7 -/
  ab_eq : ab = 7
  /-- Assertion that CD = 15 -/
  cd_eq : cd = 15
  /-- Assertion that the height of triangle EAB is thrice the height of the trapezoid -/
  eab_height : ℝ
  eab_height_eq : eab_height = 3 * h

/-- The ratio of the area of triangle EAB to the area of trapezoid ABCD is 21/22 -/
theorem area_ratio_eab_abcd (t : ExtendedTrapezoid) : 
  (t.ab * t.eab_height) / ((t.ab + t.cd) * t.h) = 21 / 22 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_eab_abcd_l1948_194876


namespace NUMINAMATH_CALUDE_soap_bubble_radius_l1948_194872

noncomputable def final_radius (α : ℝ) (r₀ : ℝ) (U : ℝ) (k : ℝ) : ℝ :=
  (U^2 * r₀^2 / (32 * k * Real.pi * α))^(1/3)

theorem soap_bubble_radius (α : ℝ) (r₀ : ℝ) (U : ℝ) (k : ℝ) 
  (h1 : α > 0) (h2 : r₀ > 0) (h3 : U ≠ 0) (h4 : k > 0) :
  ∃ (r : ℝ), r = final_radius α r₀ U k ∧ 
  r = (U^2 * r₀^2 / (32 * k * Real.pi * α))^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_soap_bubble_radius_l1948_194872


namespace NUMINAMATH_CALUDE_problem_1_l1948_194804

theorem problem_1 (x y : ℝ) (h : |x + 2| + |y - 3| = 0) : x - y + 1 = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1948_194804


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l1948_194895

theorem cube_sum_theorem (a b c : ℕ) : 
  a^3 = 1 + 7 ∧ 
  3^3 = 1 + 7 + b ∧ 
  4^3 = 1 + 7 + c → 
  a + b + c = 77 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l1948_194895


namespace NUMINAMATH_CALUDE_equal_reciprocal_sum_l1948_194884

theorem equal_reciprocal_sum (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x = 2 + 1 / y) (h2 : y = 2 + 1 / x) : y = x := by
  sorry

end NUMINAMATH_CALUDE_equal_reciprocal_sum_l1948_194884


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1948_194848

theorem binomial_coefficient_ratio (n k : ℕ) (h1 : n > 0) (h2 : k > 0) :
  (Nat.choose n k : ℚ) / (Nat.choose n (k + 1) : ℚ) = 1 / 3 ∧
  (Nat.choose n (k + 1) : ℚ) / (Nat.choose n (k + 2) : ℚ) = 1 / 2 →
  n + k = 6 :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1948_194848


namespace NUMINAMATH_CALUDE_largest_integral_x_l1948_194820

theorem largest_integral_x : ∃ (x : ℤ),
  (1 / 4 : ℚ) < (x : ℚ) / 6 ∧ 
  (x : ℚ) / 6 < (7 / 11 : ℚ) ∧
  (∀ (y : ℤ), (1 / 4 : ℚ) < (y : ℚ) / 6 ∧ (y : ℚ) / 6 < (7 / 11 : ℚ) → y ≤ x) ∧
  x = 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_integral_x_l1948_194820


namespace NUMINAMATH_CALUDE_solve_for_n_l1948_194885

def first_seven_multiples_of_six : List ℕ := [6, 12, 18, 24, 30, 36, 42]

def a : ℚ := (List.sum first_seven_multiples_of_six) / 7

def b (n : ℕ) : ℕ := 2 * n

theorem solve_for_n (n : ℕ) (h : n > 0) : a ^ 2 - (b n) ^ 2 = 0 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_n_l1948_194885


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1948_194861

theorem expand_and_simplify (x y : ℝ) : (x + 6) * (x + 8 + y) = x^2 + 14*x + x*y + 48 + 6*y := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1948_194861


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l1948_194881

/-- The area of a square with adjacent vertices at (0,5) and (5,0) is 50 -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (0, 5)
  let p2 : ℝ × ℝ := (5, 0)
  let side_length := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let area := side_length^2
  area = 50 := by sorry


end NUMINAMATH_CALUDE_square_area_from_vertices_l1948_194881


namespace NUMINAMATH_CALUDE_exp_two_ln_two_equals_four_l1948_194882

theorem exp_two_ln_two_equals_four : Real.exp (2 * Real.log 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_exp_two_ln_two_equals_four_l1948_194882


namespace NUMINAMATH_CALUDE_two_digit_number_representation_l1948_194851

/-- Represents a two-digit number -/
def two_digit_number (a b : ℕ) : ℕ := 10 * b + a

/-- Theorem stating that a two-digit number with digit a in the units place
    and digit b in the tens place is represented as 10b + a -/
theorem two_digit_number_representation (a b : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : b ≠ 0) :
  two_digit_number a b = 10 * b + a := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_representation_l1948_194851


namespace NUMINAMATH_CALUDE_expression_simplification_l1948_194835

variable (a b : ℝ)

theorem expression_simplification (h : a ≠ -1) :
  b * (a - 1 + 1 / (a + 1)) / ((a^2 + 2*a) / (a + 1)) = a * b / (a + 2) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1948_194835


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1948_194815

-- Define the quadratic function
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

-- Define the theorem
theorem quadratic_function_properties
  (b c : ℝ)
  (h1 : f b c 2 = f b c (-2))
  (h2 : f b c 1 = 0) :
  (∀ x, f b c x = x^2 - 1) ∧
  (∀ m : ℝ, (∀ x ≥ (1/2 : ℝ), ∃ y, 4*m*(f b c y) + f b c (y-1) = 4-4*m) →
    -1/4 < m ∧ m ≤ 19/4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1948_194815


namespace NUMINAMATH_CALUDE_range_of_expression_l1948_194889

theorem range_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 ≤ x^2 + y^2 + Real.sqrt (x * y) ∧ x^2 + y^2 + Real.sqrt (x * y) ≤ 9/8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l1948_194889


namespace NUMINAMATH_CALUDE_cube_equation_solution_l1948_194810

theorem cube_equation_solution :
  ∃ x : ℝ, (2*x - 8)^3 = 64 ∧ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l1948_194810


namespace NUMINAMATH_CALUDE_sum_of_roots_l1948_194836

theorem sum_of_roots (k c : ℝ) (x₁ x₂ : ℝ) : 
  x₁ ≠ x₂ → 
  (4 * x₁^2 - k * x₁ = c) → 
  (4 * x₂^2 - k * x₂ = c) → 
  x₁ + x₂ = k / 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1948_194836


namespace NUMINAMATH_CALUDE_molecular_weight_calculation_molecular_weight_proof_l1948_194801

/-- Given the molecular weight of 10 moles of a substance, 
    calculate the molecular weight of x moles of the same substance. -/
theorem molecular_weight_calculation 
  (mw_10_moles : ℝ)  -- molecular weight of 10 moles
  (x : ℝ)            -- number of moles we want to calculate
  (h : mw_10_moles > 0) -- assumption that molecular weight is positive
  : ℝ :=
  (mw_10_moles / 10) * x

/-- Prove that the molecular weight calculation is correct -/
theorem molecular_weight_proof 
  (mw_10_moles : ℝ)  -- molecular weight of 10 moles
  (x : ℝ)            -- number of moles we want to calculate
  (h : mw_10_moles > 0) -- assumption that molecular weight is positive
  : molecular_weight_calculation mw_10_moles x h = (mw_10_moles / 10) * x :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_calculation_molecular_weight_proof_l1948_194801


namespace NUMINAMATH_CALUDE_intersection_of_complex_equations_l1948_194863

open Complex

theorem intersection_of_complex_equations (k : ℝ) : 
  (∃! z : ℂ, (Complex.abs (z - 3) = 3 * Complex.abs (z + 3)) ∧ (Complex.abs z = k)) ↔ k = 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_complex_equations_l1948_194863


namespace NUMINAMATH_CALUDE_fiftieth_term_l1948_194890

/-- Sequence defined as a_n = (n + 4) * x^(n-1) for n ≥ 1 -/
def a (n : ℕ) (x : ℝ) : ℝ := (n + 4) * x^(n - 1)

/-- The 50th term of the sequence is 54x^49 -/
theorem fiftieth_term (x : ℝ) : a 50 x = 54 * x^49 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_l1948_194890


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_five_primes_l1948_194877

theorem smallest_number_divisible_by_five_primes : ∃ n : ℕ, 
  (∃ p₁ p₂ p₃ p₄ p₅ : ℕ, Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ 
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ 
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ 
    p₄ ≠ p₅ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0 ∧ n % p₅ = 0) ∧
  (∀ m : ℕ, m < n → 
    ¬(∃ q₁ q₂ q₃ q₄ q₅ : ℕ, Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ Prime q₅ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧ 
      q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧ 
      q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧ 
      q₄ ≠ q₅ ∧
      m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0 ∧ m % q₅ = 0)) ∧
  n = 2310 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_five_primes_l1948_194877


namespace NUMINAMATH_CALUDE_f_neg_two_eq_neg_three_l1948_194887

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_neg_two_eq_neg_three
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_pos : ∀ x > 0, f x = x^2 - 1) :
  f (-2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_neg_three_l1948_194887


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1948_194847

theorem rectangle_perimeter (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let area := l * b
  area = 363 → 2 * (l + b) = 88 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1948_194847


namespace NUMINAMATH_CALUDE_largest_integer_problem_l1948_194824

theorem largest_integer_problem (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧  -- Four different integers
  (a + b + c + d) / 4 = 72 ∧  -- Average is 72
  a = 21  -- Smallest integer is 21
  → d = 222 := by  -- Largest integer is 222
  sorry

end NUMINAMATH_CALUDE_largest_integer_problem_l1948_194824


namespace NUMINAMATH_CALUDE_solution_range_l1948_194841

theorem solution_range (a : ℝ) : 
  (∃ x ∈ Set.Icc (-1) 1, x^2 + 2*x - a = 0) ↔ a ∈ Set.Icc (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l1948_194841


namespace NUMINAMATH_CALUDE_fan_sales_theorem_l1948_194898

/-- Represents an electric fan model with its purchase price and selling price -/
structure FanModel where
  purchasePrice : ℕ
  sellingPrice : ℕ

/-- Represents the sales data for a week -/
structure WeekSales where
  modelA : ℕ
  modelB : ℕ
  revenue : ℕ

def modelA : FanModel := { purchasePrice := 200, sellingPrice := 250 }
def modelB : FanModel := { purchasePrice := 170, sellingPrice := 210 }

def week1 : WeekSales := { modelA := 3, modelB := 5, revenue := 1800 }
def week2 : WeekSales := { modelA := 4, modelB := 10, revenue := 3100 }

def totalBudget : ℕ := 5400
def totalUnits : ℕ := 30
def profitGoal : ℕ := 1400

theorem fan_sales_theorem :
  (week1.modelA * modelA.sellingPrice + week1.modelB * modelB.sellingPrice = week1.revenue) ∧
  (week2.modelA * modelA.sellingPrice + week2.modelB * modelB.sellingPrice = week2.revenue) ∧
  (∀ a : ℕ, a ≤ 10 ↔ a * modelA.purchasePrice + (totalUnits - a) * modelB.purchasePrice ≤ totalBudget) ∧
  (∀ a : ℕ, a ≤ 10 → a * (modelA.sellingPrice - modelA.purchasePrice) + 
    (totalUnits - a) * (modelB.sellingPrice - modelB.purchasePrice) < profitGoal) := by
  sorry

end NUMINAMATH_CALUDE_fan_sales_theorem_l1948_194898


namespace NUMINAMATH_CALUDE_added_balls_relationship_l1948_194800

/-- Proves the relationship between added white and black balls to maintain a specific probability -/
theorem added_balls_relationship (x y : ℕ) : 
  (x + 2 : ℚ) / (5 + x + y : ℚ) = 1/3 → y = 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_added_balls_relationship_l1948_194800


namespace NUMINAMATH_CALUDE_min_value_problem_l1948_194828

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 3/y = 1) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2/a + 3/b = 1 → 2*x + 3*y ≤ 2*a + 3*b ∧ 
  ∃ c d : ℝ, c > 0 ∧ d > 0 ∧ 2/c + 3/d = 1 ∧ 2*c + 3*d = 25 :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l1948_194828


namespace NUMINAMATH_CALUDE_product_of_numbers_l1948_194896

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 180) : x * y = 72 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1948_194896


namespace NUMINAMATH_CALUDE_binary_147_ones_zeros_difference_l1948_194857

def binary_representation (n : ℕ) : List Bool :=
  sorry

def count_ones (l : List Bool) : ℕ :=
  sorry

def count_zeros (l : List Bool) : ℕ :=
  sorry

theorem binary_147_ones_zeros_difference :
  let bin_147 := binary_representation 147
  let ones := count_ones bin_147
  let zeros := count_zeros bin_147
  ones - zeros = 0 := by sorry

end NUMINAMATH_CALUDE_binary_147_ones_zeros_difference_l1948_194857


namespace NUMINAMATH_CALUDE_problem_1_l1948_194829

theorem problem_1 : -9 + 5 - (-12) + (-3) = 8 := by sorry

end NUMINAMATH_CALUDE_problem_1_l1948_194829


namespace NUMINAMATH_CALUDE_car_rental_cost_per_mile_l1948_194856

theorem car_rental_cost_per_mile 
  (base_cost : ℝ) 
  (total_miles : ℝ) 
  (total_cost : ℝ) 
  (h1 : base_cost = 150)
  (h2 : total_miles = 1364)
  (h3 : total_cost = 832) :
  (total_cost - base_cost) / total_miles = 0.50 := by
sorry

end NUMINAMATH_CALUDE_car_rental_cost_per_mile_l1948_194856


namespace NUMINAMATH_CALUDE_products_inspected_fraction_l1948_194823

/-- The fraction of products inspected by John, Jane, and Roy is 1 -/
theorem products_inspected_fraction (j n r : ℝ) : 
  j ≥ 0 → n ≥ 0 → r ≥ 0 →
  0.007 * j + 0.008 * n + 0.01 * r = 0.0085 →
  j + n + r = 1 :=
by sorry

end NUMINAMATH_CALUDE_products_inspected_fraction_l1948_194823


namespace NUMINAMATH_CALUDE_exists_unsolvable_grid_l1948_194822

/-- Represents a 9x9 grid with values either 1 or -1 -/
def Grid := Fin 9 → Fin 9 → Int

/-- Defines a valid grid where all values are either 1 or -1 -/
def valid_grid (g : Grid) : Prop :=
  ∀ i j, g i j = 1 ∨ g i j = -1

/-- Defines the neighbors of a cell in the grid -/
def neighbors (i j : Fin 9) : List (Fin 9 × Fin 9) :=
  [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]

/-- Computes the new value of a cell after a move -/
def move (g : Grid) (i j : Fin 9) : Int :=
  (neighbors i j).foldl (λ acc (ni, nj) => acc * g ni nj) 1

/-- Applies a move to the entire grid -/
def apply_move (g : Grid) : Grid :=
  λ i j => move g i j

/-- Checks if all cells in the grid are 1 -/
def all_ones (g : Grid) : Prop :=
  ∀ i j, g i j = 1

/-- The main theorem: there exists a valid grid that cannot be transformed to all ones -/
theorem exists_unsolvable_grid :
  ∃ (g : Grid), valid_grid g ∧ 
    ∀ (n : ℕ), ¬(all_ones ((apply_move^[n]) g)) :=
  sorry

end NUMINAMATH_CALUDE_exists_unsolvable_grid_l1948_194822


namespace NUMINAMATH_CALUDE_cookie_calorie_consumption_l1948_194832

/-- Represents the number of calories in a single cookie of each type -/
structure CookieCalories where
  caramel : ℕ
  chocolate_chip : ℕ
  peanut_butter : ℕ

/-- Represents the number of cookies selected of each type -/
structure SelectedCookies where
  caramel : ℕ
  chocolate_chip : ℕ
  peanut_butter : ℕ

/-- Calculates the total calories consumed based on the number of cookies selected and their calorie content -/
def totalCalories (calories : CookieCalories) (selected : SelectedCookies) : ℕ :=
  calories.caramel * selected.caramel +
  calories.chocolate_chip * selected.chocolate_chip +
  calories.peanut_butter * selected.peanut_butter

/-- Proves that selecting 5 caramel, 3 chocolate chip, and 2 peanut butter cookies results in consuming 204 calories -/
theorem cookie_calorie_consumption :
  let calories := CookieCalories.mk 18 22 24
  let selected := SelectedCookies.mk 5 3 2
  totalCalories calories selected = 204 := by
  sorry

end NUMINAMATH_CALUDE_cookie_calorie_consumption_l1948_194832


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1948_194888

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 3) ↔ d ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1948_194888


namespace NUMINAMATH_CALUDE_first_company_visit_charge_is_55_l1948_194833

/-- The visit charge of the first plumbing company -/
def first_company_visit_charge : ℝ := sorry

/-- The hourly rate of the first plumbing company -/
def first_company_hourly_rate : ℝ := 35

/-- The visit charge of Reliable Plumbing -/
def reliable_plumbing_visit_charge : ℝ := 75

/-- The hourly rate of Reliable Plumbing -/
def reliable_plumbing_hourly_rate : ℝ := 30

/-- The number of labor hours -/
def labor_hours : ℝ := 4

theorem first_company_visit_charge_is_55 :
  first_company_visit_charge = 55 :=
by
  have h1 : first_company_visit_charge + labor_hours * first_company_hourly_rate =
            reliable_plumbing_visit_charge + labor_hours * reliable_plumbing_hourly_rate :=
    sorry
  sorry

end NUMINAMATH_CALUDE_first_company_visit_charge_is_55_l1948_194833


namespace NUMINAMATH_CALUDE_rectangle_with_equal_sides_is_square_inverse_proposition_inverse_proposition_is_true_l1948_194859

-- Define what a rectangle is
def is_rectangle (shape : Type) : Prop := sorry

-- Define what a square is
def is_square (shape : Type) : Prop := sorry

-- Define what it means for a shape to have equal adjacent sides
def has_equal_adjacent_sides (shape : Type) : Prop := sorry

-- The original proposition
theorem rectangle_with_equal_sides_is_square (shape : Type) :
  is_rectangle shape → has_equal_adjacent_sides shape → is_square shape := sorry

-- The inverse proposition
theorem inverse_proposition (shape : Type) :
  is_square shape → is_rectangle shape ∧ has_equal_adjacent_sides shape := sorry

-- The main theorem: proving that the inverse proposition is true
theorem inverse_proposition_is_true :
  (∀ shape, is_square shape → is_rectangle shape ∧ has_equal_adjacent_sides shape) := sorry

end NUMINAMATH_CALUDE_rectangle_with_equal_sides_is_square_inverse_proposition_inverse_proposition_is_true_l1948_194859


namespace NUMINAMATH_CALUDE_f_of_3_equals_155_l1948_194827

-- Define the function f
def f (x : ℝ) : ℝ := 8 * x^3 - 6 * x^2 - 4 * x + 5

-- Theorem statement
theorem f_of_3_equals_155 : f 3 = 155 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_equals_155_l1948_194827


namespace NUMINAMATH_CALUDE_triangle_perimeter_in_divided_square_l1948_194862

/-- Given a square of side length z divided into a central rectangle and four congruent right-angled triangles,
    where the shorter side of the rectangle is x, the perimeter of one of the triangles is 3z/2. -/
theorem triangle_perimeter_in_divided_square (z x : ℝ) (hz : z > 0) (hx : 0 < x ∧ x < z) :
  let triangle_perimeter := (z - x) / 2 + (z + x) / 2 + z / 2
  triangle_perimeter = 3 * z / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_in_divided_square_l1948_194862


namespace NUMINAMATH_CALUDE_number_problem_l1948_194809

theorem number_problem : ∃ x : ℝ, 1.3333 * x = 4.82 ∧ abs (x - 3.615) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1948_194809


namespace NUMINAMATH_CALUDE_sequence_properties_l1948_194870

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def arithmetic_sequence (b : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, b (n + 1) = b n + d

def c (a b : ℕ → ℝ) (n : ℕ) : ℝ := a n + b n

theorem sequence_properties
  (a : ℕ → ℝ) (b : ℕ → ℝ) (q d : ℝ)
  (h_geom : geometric_sequence a q)
  (h_arith : arithmetic_sequence b d)
  (h_q : q ≠ 1)
  (h_d : d ≠ 0) :
  (¬ arithmetic_sequence (c a b) ((c a b 2) - (c a b 1))) ∧
  (a 1 = 1 ∧ q = 2 → 
    ∃ f : ℝ → ℝ, f d = (c a b 2) ∧ 
    (∀ x : ℝ, x ≠ -1 ∧ x ≠ -2 ∧ x ≠ 0 → f x = x^2 + 3*x)) ∧
  (¬ geometric_sequence (c a b) ((c a b 2) / (c a b 1))) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l1948_194870


namespace NUMINAMATH_CALUDE_larger_integer_value_l1948_194865

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * b = 294) : 
  (max a b : ℝ) = 7 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_value_l1948_194865


namespace NUMINAMATH_CALUDE_alberts_current_funds_l1948_194814

/-- The problem of calculating Albert's current funds --/
theorem alberts_current_funds
  (total_cost : ℝ)
  (additional_needed : ℝ)
  (h1 : total_cost = 18.50)
  (h2 : additional_needed = 12) :
  total_cost - additional_needed = 6.50 := by
  sorry

end NUMINAMATH_CALUDE_alberts_current_funds_l1948_194814


namespace NUMINAMATH_CALUDE_vector_projection_and_perpendicular_l1948_194813

/-- Given two vectors a and b in ℝ², and a scalar k, we define vector c and prove properties about their relationships. -/
theorem vector_projection_and_perpendicular (a b : ℝ × ℝ) (k : ℝ) :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (3, 1)
  let c : ℝ × ℝ := b - k • a
  (a.1 * c.1 + a.2 * c.2 = 0) →
  (let proj := (a.1 * b.1 + a.2 * b.2) / Real.sqrt (a.1^2 + a.2^2)
   proj = Real.sqrt 5 ∧ k = 1 ∧ c = (2, -1)) := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_and_perpendicular_l1948_194813


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1948_194807

theorem trigonometric_identities (A B C : ℝ) (h : A + B + C = Real.pi) :
  (Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2 = 1 - 2 * Real.cos A * Real.cos B * Real.cos C ∧
  (Real.sin A)^2 + (Real.sin B)^2 + (Real.sin C)^2 = 2 * Real.cos A * Real.cos B * Real.cos C :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1948_194807


namespace NUMINAMATH_CALUDE_plane_equation_satisfies_conditions_l1948_194834

/-- A plane in 3D space --/
structure Plane where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- A point in 3D space --/
structure Point where
  x : ℚ
  y : ℚ
  z : ℚ

/-- Check if a point lies on a plane --/
def Point.liesOn (p : Point) (pl : Plane) : Prop :=
  pl.a * p.x + pl.b * p.y + pl.c * p.z + pl.d = 0

/-- Check if two planes are parallel --/
def Plane.isParallelTo (pl1 pl2 : Plane) : Prop :=
  ∃ (k : ℚ), k ≠ 0 ∧ pl1.a = k * pl2.a ∧ pl1.b = k * pl2.b ∧ pl1.c = k * pl2.c

theorem plane_equation_satisfies_conditions
  (given_plane : Plane)
  (given_point : Point)
  (parallel_plane : Plane)
  (h1 : given_plane.a = 3)
  (h2 : given_plane.b = 4)
  (h3 : given_plane.c = -2)
  (h4 : given_plane.d = 16)
  (h5 : given_point.x = 2)
  (h6 : given_point.y = -3)
  (h7 : given_point.z = 5)
  (h8 : parallel_plane.a = 3)
  (h9 : parallel_plane.b = 4)
  (h10 : parallel_plane.c = -2)
  (h11 : parallel_plane.d = 6)
  : given_point.liesOn given_plane ∧
    given_plane.isParallelTo parallel_plane ∧
    given_plane.a > 0 ∧
    Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs given_plane.a) (Int.natAbs given_plane.b)) (Int.natAbs given_plane.c)) (Int.natAbs given_plane.d) = 1 :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_satisfies_conditions_l1948_194834


namespace NUMINAMATH_CALUDE_live_bargaining_theorem_l1948_194825

/-- Represents the price reduction scenario in a live streaming bargaining event. -/
def live_bargaining_price_reduction (initial_price final_price : ℝ) (num_rounds : ℕ) (reduction_rate : ℝ) : Prop :=
  initial_price * (1 - reduction_rate) ^ num_rounds = final_price

/-- The live bargaining price reduction theorem. -/
theorem live_bargaining_theorem :
  ∃ (x : ℝ), live_bargaining_price_reduction 120 43.2 2 x :=
sorry

end NUMINAMATH_CALUDE_live_bargaining_theorem_l1948_194825


namespace NUMINAMATH_CALUDE_garden_minimum_cost_l1948_194821

/-- Represents the cost of a herb in dollars per square meter -/
structure HerbCost where
  cost : ℝ
  cost_positive : cost > 0

/-- Represents a region in the garden -/
structure Region where
  area : ℝ
  area_positive : area > 0

/-- Calculates the minimum cost for planting a garden given regions and herb costs -/
def minimum_garden_cost (regions : List Region) (herb_costs : List HerbCost) : ℝ :=
  sorry

/-- The main theorem stating the minimum cost for the given garden configuration -/
theorem garden_minimum_cost :
  let regions : List Region := [
    ⟨14, by norm_num⟩,
    ⟨35, by norm_num⟩,
    ⟨10, by norm_num⟩,
    ⟨21, by norm_num⟩,
    ⟨36, by norm_num⟩
  ]
  let herb_costs : List HerbCost := [
    ⟨1.00, by norm_num⟩,
    ⟨1.50, by norm_num⟩,
    ⟨2.00, by norm_num⟩,
    ⟨2.50, by norm_num⟩,
    ⟨3.00, by norm_num⟩
  ]
  minimum_garden_cost regions herb_costs = 195.50 := by
    sorry

end NUMINAMATH_CALUDE_garden_minimum_cost_l1948_194821


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_four_l1948_194839

theorem largest_four_digit_divisible_by_four :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 4 = 0 → n ≤ 9996 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_four_l1948_194839


namespace NUMINAMATH_CALUDE_fibonacci_inequality_l1948_194880

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_inequality (n : ℕ) :
  (fibonacci (n + 1) : ℝ) ^ (1 / n : ℝ) ≥ 1 + (fibonacci n : ℝ) ^ (-1 / n : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_inequality_l1948_194880


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l1948_194893

-- Define the circumference of the circle
def circumference : ℝ := 36

-- Theorem stating that the area of a circle with circumference 36 cm is 324/π cm²
theorem circle_area_from_circumference :
  (π * (circumference / (2 * π))^2) = 324 / π := by sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l1948_194893


namespace NUMINAMATH_CALUDE_combined_rectangle_perimeter_l1948_194806

/-- The perimeter of a rectangle formed by combining a square of side 8 cm
    with a rectangle of dimensions 8 cm x 4 cm is 48 cm. -/
theorem combined_rectangle_perimeter :
  let square_side : ℝ := 8
  let rect_length : ℝ := 8
  let rect_width : ℝ := 4
  let new_rect_length : ℝ := square_side + rect_length
  let new_rect_width : ℝ := square_side
  let perimeter : ℝ := 2 * (new_rect_length + new_rect_width)
  perimeter = 48 := by sorry

end NUMINAMATH_CALUDE_combined_rectangle_perimeter_l1948_194806


namespace NUMINAMATH_CALUDE_ellipse_min_area_l1948_194808

/-- An ellipse containing two specific circles has a minimum area of π -/
theorem ellipse_min_area (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → 
    ((x - 2)^2 + y^2 ≤ 4 ∧ (x + 2)^2 + y^2 ≤ 4)) → 
  π * a * b ≥ π := by sorry

end NUMINAMATH_CALUDE_ellipse_min_area_l1948_194808


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1948_194892

/-- A geometric sequence with third term 3 and fifth term 27 has first term 1/3 -/
theorem geometric_sequence_first_term (a : ℝ) (r : ℝ) : 
  a * r^2 = 3 → a * r^4 = 27 → a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1948_194892


namespace NUMINAMATH_CALUDE_sum_of_base_8_digits_of_888_l1948_194855

def base_8_representation (n : ℕ) : List ℕ :=
  sorry

def sum_of_digits (digits : List ℕ) : ℕ :=
  sorry

theorem sum_of_base_8_digits_of_888 :
  sum_of_digits (base_8_representation 888) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_base_8_digits_of_888_l1948_194855


namespace NUMINAMATH_CALUDE_pirate_treasure_l1948_194883

theorem pirate_treasure (m : ℕ) (n : ℕ) (u : ℕ) : 
  (2/3 * (2/3 * (2/3 * (m - 1) - 1) - 1) = 3 * n) →
  (110 ≤ 81 * u + 25) →
  (81 * u + 25 ≤ 200) →
  (m = 187 ∧ 
   1 + (187 - 1) / 3 + 18 = 81 ∧
   1 + (187 - (1 + (187 - 1) / 3) - 1) / 3 + 18 = 60 ∧
   1 + (187 - (1 + (187 - 1) / 3) - (1 + (187 - (1 + (187 - 1) / 3) - 1) / 3) - 1) / 3 + 18 = 46) :=
by sorry

end NUMINAMATH_CALUDE_pirate_treasure_l1948_194883


namespace NUMINAMATH_CALUDE_alex_coin_distribution_l1948_194866

/-- The minimum number of additional coins needed for unique distribution -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed for Alex -/
theorem alex_coin_distribution :
  min_additional_coins 15 90 = 30 := by
  sorry

end NUMINAMATH_CALUDE_alex_coin_distribution_l1948_194866


namespace NUMINAMATH_CALUDE_binary_11011_equals_27_l1948_194894

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11011_equals_27 :
  binary_to_decimal [true, true, false, true, true] = 27 := by
  sorry

end NUMINAMATH_CALUDE_binary_11011_equals_27_l1948_194894


namespace NUMINAMATH_CALUDE_vasyas_numbers_l1948_194899

theorem vasyas_numbers (x y : ℝ) : 
  x + y = x * y ∧ x + y = x / y ∧ x * y = x / y → x = (1 : ℝ) / 2 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_vasyas_numbers_l1948_194899


namespace NUMINAMATH_CALUDE_simplify_sqrt_a_squared_b_over_two_l1948_194852

theorem simplify_sqrt_a_squared_b_over_two
  (a b : ℝ) (ha : a < 0) :
  Real.sqrt ((a^2 * b) / 2) = -a / 2 * Real.sqrt (2 * b) :=
by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_a_squared_b_over_two_l1948_194852


namespace NUMINAMATH_CALUDE_replacement_count_l1948_194871

-- Define the replacement percentage
def replacement_percentage : ℝ := 0.2

-- Define the final milk percentage
def final_milk_percentage : ℝ := 0.5120000000000001

-- Define the function to calculate the remaining milk percentage after n replacements
def remaining_milk (n : ℕ) : ℝ := (1 - replacement_percentage) ^ n

-- Theorem statement
theorem replacement_count : ∃ n : ℕ, remaining_milk n = final_milk_percentage ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_replacement_count_l1948_194871


namespace NUMINAMATH_CALUDE_circle_area_probability_l1948_194864

theorem circle_area_probability (AB : ℝ) (h_AB : AB = 10) : 
  let prob := (Real.sqrt 64 - Real.sqrt 36) / AB
  prob = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_circle_area_probability_l1948_194864


namespace NUMINAMATH_CALUDE_final_price_after_discounts_l1948_194869

/-- Calculates the final price of an item after two successive discounts --/
theorem final_price_after_discounts (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  original_price = 200 ∧ discount1 = 0.4 ∧ discount2 = 0.25 →
  original_price * (1 - discount1) * (1 - discount2) = 90 := by
  sorry

#check final_price_after_discounts

end NUMINAMATH_CALUDE_final_price_after_discounts_l1948_194869


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1948_194879

theorem solution_set_of_inequality (x : ℝ) :
  (6 * x^2 + 5 * x < 4) ↔ (-4/3 < x ∧ x < 1/2) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1948_194879


namespace NUMINAMATH_CALUDE_coffee_shop_weekly_production_l1948_194826

/-- Represents the brewing rate and operating hours of a coffee shop for a specific day type -/
structure DayType where
  brewingRate : ℕ
  operatingHours : ℕ

/-- Calculates the total number of coffee cups brewed in a week -/
def totalCupsPerWeek (weekday : DayType) (weekend : DayType) : ℕ :=
  weekday.brewingRate * weekday.operatingHours * 5 +
  weekend.brewingRate * weekend.operatingHours * 2

/-- Theorem: The coffee shop brews 400 cups in a week -/
theorem coffee_shop_weekly_production :
  let weekday : DayType := { brewingRate := 10, operatingHours := 5 }
  let weekend : DayType := { brewingRate := 15, operatingHours := 5 }
  totalCupsPerWeek weekday { weekend with operatingHours := 6 } = 400 :=
by sorry

end NUMINAMATH_CALUDE_coffee_shop_weekly_production_l1948_194826


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l1948_194875

/-- The number of sides of a polygon given the sum of its interior angles -/
theorem polygon_sides_from_angle_sum (sum_of_angles : ℝ) : sum_of_angles = 720 → ∃ n : ℕ, n = 6 ∧ (n - 2) * 180 = sum_of_angles := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l1948_194875


namespace NUMINAMATH_CALUDE_trajectory_classification_l1948_194803

/-- The trajectory of a point P(x,y) satisfying |PF₁| + |PF₂| = 2a, where F₁(-5,0) and F₂(5,0) are fixed points -/
def trajectory (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | dist p (-5, 0) + dist p (5, 0) = 2 * a}

/-- The distance between F₁ and F₂ -/
def f₁f₂_distance : ℝ := 10

theorem trajectory_classification (a : ℝ) (h : a > 0) :
  (a = f₁f₂_distance → trajectory a = {p : ℝ × ℝ | p.1 ∈ Set.Icc (-5 : ℝ) 5 ∧ p.2 = 0}) ∧
  (a > f₁f₂_distance → ∃ c d : ℝ, c > 0 ∧ d > 0 ∧ trajectory a = {p : ℝ × ℝ | (p.1 / c)^2 + (p.2 / d)^2 = 1}) ∧
  (a < f₁f₂_distance → trajectory a = ∅) :=
sorry

end NUMINAMATH_CALUDE_trajectory_classification_l1948_194803


namespace NUMINAMATH_CALUDE_shrimp_trap_problem_l1948_194891

theorem shrimp_trap_problem (victor_shrimp : ℕ) (austin_shrimp : ℕ) :
  victor_shrimp = 26 →
  (victor_shrimp + austin_shrimp + (victor_shrimp + austin_shrimp) / 2) * 7 / 11 = 42 →
  austin_shrimp + 8 = victor_shrimp :=
by sorry

end NUMINAMATH_CALUDE_shrimp_trap_problem_l1948_194891


namespace NUMINAMATH_CALUDE_third_tea_price_is_175_5_l1948_194858

/-- The price of the third variety of tea -/
def third_tea_price (price1 price2 mixture_price : ℚ) : ℚ :=
  2 * mixture_price - (price1 + price2) / 2

/-- Theorem stating that the price of the third variety of tea is 175.5 given the conditions -/
theorem third_tea_price_is_175_5 :
  third_tea_price 126 135 153 = 175.5 := by
  sorry

end NUMINAMATH_CALUDE_third_tea_price_is_175_5_l1948_194858


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1948_194854

theorem arithmetic_geometric_mean_inequality 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  (a + b + c + d) / 4 ≥ (a * b * c * d) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1948_194854


namespace NUMINAMATH_CALUDE_square_field_side_length_l1948_194802

theorem square_field_side_length (area : ℝ) (side_length : ℝ) :
  area = 100 ∧ area = side_length ^ 2 → side_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_field_side_length_l1948_194802


namespace NUMINAMATH_CALUDE_sum_first_12_even_numbers_l1948_194812

def first_n_even_numbers (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => 2 * (i + 1))

theorem sum_first_12_even_numbers :
  (first_n_even_numbers 12).sum = 156 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_12_even_numbers_l1948_194812


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt11_l1948_194873

theorem consecutive_integers_around_sqrt11 (m n : ℤ) :
  (n = m + 1) →
  (m < Real.sqrt 11) →
  (Real.sqrt 11 < n) →
  m + n = 7 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt11_l1948_194873


namespace NUMINAMATH_CALUDE_union_A_B_intersection_A_complement_B_l1948_194849

-- Define the sets A and B
def A : Set ℝ := {x | -3 < x ∧ x < 2}
def B : Set ℝ := {x | (x - 1) / (x + 5) > 0}

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x | x < -5 ∨ x > -3} := by sorry

-- Theorem for the intersection of A and complement of B
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {x | -3 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_union_A_B_intersection_A_complement_B_l1948_194849


namespace NUMINAMATH_CALUDE_rod_length_relation_l1948_194868

/-- Two homogeneous rods with equal cross-sectional areas, different densities, and different
    coefficients of expansion are welded together. The system's center of gravity remains
    unchanged despite thermal expansion. -/
theorem rod_length_relation (l₁ l₂ d₁ d₂ α₁ α₂ : ℝ) 
    (h₁ : l₁ > 0) (h₂ : l₂ > 0) (h₃ : d₁ > 0) (h₄ : d₂ > 0) (h₅ : α₁ > 0) (h₆ : α₂ > 0) :
    (l₁ / l₂)^2 = (d₂ * α₂) / (d₁ * α₁) := by
  sorry

end NUMINAMATH_CALUDE_rod_length_relation_l1948_194868


namespace NUMINAMATH_CALUDE_cosine_identity_l1948_194819

theorem cosine_identity (a : Real) (h : 3 * Real.pi / 2 < a ∧ a < 2 * Real.pi) :
  Real.sqrt (1/2 + 1/2 * Real.sqrt (1/2 + 1/2 * Real.cos (2 * a))) = -Real.cos (a / 2) := by
  sorry

end NUMINAMATH_CALUDE_cosine_identity_l1948_194819


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1948_194853

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The x-axis is the set of all points with y-coordinate equal to 0 -/
def x_axis : Set Point := {p : Point | p.y = 0}

/-- Theorem: If a point A has y-coordinate equal to 0, then A lies on the x-axis -/
theorem point_on_x_axis (A : Point) (h : A.y = 0) : A ∈ x_axis := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1948_194853


namespace NUMINAMATH_CALUDE_cosine_sum_special_angle_l1948_194886

/-- 
If the terminal side of angle θ passes through the point (3, -4), 
then cos(θ + π/4) = 7√2/10.
-/
theorem cosine_sum_special_angle (θ : ℝ) : 
  (3 : ℝ) * Real.cos θ = 3 ∧ (3 : ℝ) * Real.sin θ = -4 → 
  Real.cos (θ + π/4) = 7 * Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_special_angle_l1948_194886


namespace NUMINAMATH_CALUDE_distribute_5_3_l1948_194842

/-- The number of ways to distribute n distinct objects into k distinct groups,
    where each group must contain at least one object -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: Distributing 5 distinct objects into 3 distinct groups,
    where each group must contain at least one object, can be done in 150 ways -/
theorem distribute_5_3 : distribute 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_3_l1948_194842


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l1948_194878

theorem quadratic_factorization_sum (a b c : ℤ) :
  (∀ x, x^2 + 19*x + 88 = (x + a) * (x + b)) →
  (∀ x, x^2 - 23*x + 132 = (x - b) * (x - c)) →
  a + b + c = 31 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l1948_194878


namespace NUMINAMATH_CALUDE_initial_velocity_is_three_l1948_194818

/-- The initial velocity of an object moving in a straight line is 3, given that its displacement s 
    is related to time t by the equation s = 3t - t^2, where t ∈ [0, +∞). -/
theorem initial_velocity_is_three (t : ℝ) (h : t ∈ Set.Ici 0) : 
  let s := fun (t : ℝ) => 3 * t - t^2
  let v := deriv s
  v 0 = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_velocity_is_three_l1948_194818


namespace NUMINAMATH_CALUDE_group_size_problem_l1948_194844

theorem group_size_problem (total_collection : ℕ) 
  (h1 : total_collection = 3249) 
  (h2 : ∃ n : ℕ, n * n = total_collection) : 
  ∃ n : ℕ, n = 57 ∧ n * n = total_collection :=
sorry

end NUMINAMATH_CALUDE_group_size_problem_l1948_194844


namespace NUMINAMATH_CALUDE_dice_probability_theorem_l1948_194816

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 6

/-- The total number of possible outcomes when rolling 6 dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of favorable outcomes (at least one pair but not a four-of-a-kind) -/
def favorableOutcomes : ℕ := 28800

/-- The probability of getting at least one pair but not a four-of-a-kind when rolling 6 dice -/
def probabilityPairNotFourOfAKind : ℚ := favorableOutcomes / totalOutcomes

theorem dice_probability_theorem : probabilityPairNotFourOfAKind = 25 / 81 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_theorem_l1948_194816


namespace NUMINAMATH_CALUDE_irrational_free_iff_zero_and_rational_l1948_194837

def M (a b c d : ℝ) : Set ℝ :=
  {y | ∃ x, y = a * x^3 + b * x^2 + c * x + d}

theorem irrational_free_iff_zero_and_rational (a b c d : ℝ) :
  (∀ y ∈ M a b c d, ∃ (q : ℚ), (y : ℝ) = q) ↔
  (a = 0 ∧ b = 0 ∧ c = 0 ∧ ∃ (q : ℚ), (d : ℝ) = q) :=
sorry

end NUMINAMATH_CALUDE_irrational_free_iff_zero_and_rational_l1948_194837


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1948_194860

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1948_194860


namespace NUMINAMATH_CALUDE_soap_brand_usage_ratio_l1948_194811

/-- Given a survey of households and their soap brand usage, prove the ratio of households
    using only brand B to those using both brands A and B. -/
theorem soap_brand_usage_ratio 
  (total : ℕ) 
  (neither : ℕ) 
  (only_A : ℕ) 
  (both : ℕ) 
  (h1 : total = 200)
  (h2 : neither = 80)
  (h3 : only_A = 60)
  (h4 : both = 5)
  (h5 : neither + only_A + both < total) :
  (total - (neither + only_A + both)) / both = 11 :=
by sorry

end NUMINAMATH_CALUDE_soap_brand_usage_ratio_l1948_194811


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l1948_194840

theorem no_positive_integer_solutions :
  ¬ ∃ (n : ℕ+) (p : ℕ), Prime p ∧ n.val^2 - 47*n.val + 660 = p := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l1948_194840


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l1948_194830

/-- The range of m for which the quadratic equation (m-1)x^2 + 2x + 1 = 0 has two distinct real roots -/
theorem quadratic_distinct_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (m - 1) * x^2 + 2 * x + 1 = 0 ∧ (m - 1) * y^2 + 2 * y + 1 = 0) ↔ 
  (m < 2 ∧ m ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l1948_194830


namespace NUMINAMATH_CALUDE_solution_correct_l1948_194805

def M : Matrix (Fin 2) (Fin 2) ℚ := !![5, 2; 4, 1]
def N : Matrix (Fin 2) (Fin 1) ℚ := !![5; 8]
def X : Matrix (Fin 2) (Fin 1) ℚ := !![11/3; -20/3]

theorem solution_correct : M * X = N := by sorry

end NUMINAMATH_CALUDE_solution_correct_l1948_194805


namespace NUMINAMATH_CALUDE_factor_x4_minus_81_l1948_194845

theorem factor_x4_minus_81 (x : ℝ) : x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_x4_minus_81_l1948_194845


namespace NUMINAMATH_CALUDE_number_of_teachers_l1948_194831

/-- Represents the number of students at King Middle School -/
def total_students : ℕ := 1200

/-- Represents the number of classes each student takes per day -/
def classes_per_student : ℕ := 5

/-- Represents the number of classes each teacher teaches -/
def classes_per_teacher : ℕ := 4

/-- Represents the number of students in each class -/
def students_per_class : ℕ := 30

/-- Represents the number of teachers in each class -/
def teachers_per_class : ℕ := 1

/-- Theorem stating that the number of teachers at King Middle School is 50 -/
theorem number_of_teachers : 
  (total_students * classes_per_student) / students_per_class / classes_per_teacher = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_of_teachers_l1948_194831


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1948_194867

theorem complex_number_quadrant : let z : ℂ := (Complex.I : ℂ) / (1 + Complex.I)
  ∃ (a b : ℝ), z = Complex.mk a b ∧ a > 0 ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1948_194867


namespace NUMINAMATH_CALUDE_prob_same_color_equal_one_l1948_194817

/-- Procedure A: Choose one card from k cards with equal probability 1/k and replace it with a different color card. -/
def procedureA (k : ℕ) : Unit := sorry

/-- The probability of reaching a state where all cards are of the same color after n repetitions of procedure A. -/
def probSameColor (k n : ℕ) : ℝ := sorry

theorem prob_same_color_equal_one (k n : ℕ) (h1 : k > 0) (h2 : n > 0) (h3 : k % 2 = 0) :
  probSameColor k n = 1 := by sorry

end NUMINAMATH_CALUDE_prob_same_color_equal_one_l1948_194817


namespace NUMINAMATH_CALUDE_marble_probability_correct_l1948_194843

def marble_probability (initial_red : ℕ) (initial_blue : ℕ) (initial_green : ℕ) (initial_white : ℕ)
                       (removed_red : ℕ) (removed_blue : ℕ) (added_green : ℕ) :
  (ℚ × ℚ × ℚ) :=
  let final_red : ℕ := initial_red - removed_red
  let final_blue : ℕ := initial_blue - removed_blue
  let final_green : ℕ := initial_green + added_green
  let total : ℕ := final_red + final_blue + final_green + initial_white
  ((final_red : ℚ) / total, (final_blue : ℚ) / total, (final_green : ℚ) / total)

theorem marble_probability_correct :
  marble_probability 12 10 8 5 5 4 3 = (7/29, 6/29, 11/29) := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_correct_l1948_194843
