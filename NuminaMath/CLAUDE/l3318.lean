import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_numbers_greater_than_04_l3318_331817

def numbers : List ℚ := [0.8, 1/2, 0.3, 1/3]

theorem sum_of_numbers_greater_than_04 : 
  (numbers.filter (λ x => x > 0.4)).sum = 1.3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_greater_than_04_l3318_331817


namespace NUMINAMATH_CALUDE_problem_statement_l3318_331846

theorem problem_statement (x y : ℚ) (hx : x = 3/4) (hy : y = 4/3) : 
  (1/2 : ℚ) * x^6 * y^7 = 2/3 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3318_331846


namespace NUMINAMATH_CALUDE_smallest_ellipse_area_l3318_331886

/-- The smallest area of an ellipse containing two specific circles -/
theorem smallest_ellipse_area (p q : ℝ) (h_ellipse : ∀ (x y : ℝ), x^2 / p^2 + y^2 / q^2 = 1 → 
  ((x - 2)^2 + y^2 = 4 ∨ (x + 2)^2 + y^2 = 4)) :
  ∃ (m : ℝ), m = 3 * Real.sqrt 3 / 2 ∧ 
    ∀ (p' q' : ℝ), (∀ (x y : ℝ), x^2 / p'^2 + y^2 / q'^2 = 1 → 
      ((x - 2)^2 + y^2 = 4 ∨ (x + 2)^2 + y^2 = 4)) → 
    p' * q' * Real.pi ≥ m * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_ellipse_area_l3318_331886


namespace NUMINAMATH_CALUDE_f_properties_l3318_331865

open Real

noncomputable def f (x : ℝ) : ℝ := (log (1 + x)) / x

theorem f_properties (x : ℝ) (h : x > 0) :
  (∀ y z, 0 < y ∧ y < z → f y > f z) ∧
  f x > 2 / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3318_331865


namespace NUMINAMATH_CALUDE_sector_properties_l3318_331834

/-- Given a sector OAB with central angle 120° and radius 6, 
    prove the length of arc AB and the area of segment AOB -/
theorem sector_properties :
  let angle : Real := 120 * π / 180
  let radius : Real := 6
  let arc_length : Real := radius * angle
  let sector_area : Real := (1 / 2) * radius * arc_length
  let triangle_area : Real := (1 / 2) * radius * radius * Real.sin angle
  let segment_area : Real := sector_area - triangle_area
  arc_length = 4 * π ∧ segment_area = 12 * π - 9 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_properties_l3318_331834


namespace NUMINAMATH_CALUDE_water_tank_solution_l3318_331840

/-- Represents the water tank problem --/
def WaterTankProblem (tankCapacity : ℝ) (initialFill : ℝ) (firstDayCollection : ℝ) (thirdDayOverflow : ℝ) : Prop :=
  let initialWater := tankCapacity * initialFill
  let afterFirstDay := initialWater + firstDayCollection
  let secondDayCollection := tankCapacity - afterFirstDay
  secondDayCollection - firstDayCollection = 30

/-- Theorem statement for the water tank problem --/
theorem water_tank_solution :
  WaterTankProblem 100 (2/5) 15 25 := by
  sorry


end NUMINAMATH_CALUDE_water_tank_solution_l3318_331840


namespace NUMINAMATH_CALUDE_factorization_theorem_l3318_331872

theorem factorization_theorem (a b : ℝ) : 6 * a * b - a^2 - 9 * b^2 = -(a - 3 * b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_l3318_331872


namespace NUMINAMATH_CALUDE_no_square_base_b_l3318_331800

theorem no_square_base_b : ¬ ∃ (b : ℤ), ∃ (n : ℤ), b^2 + 3*b + 1 = n^2 := by sorry

end NUMINAMATH_CALUDE_no_square_base_b_l3318_331800


namespace NUMINAMATH_CALUDE_platform_length_l3318_331866

/-- Given a train of length 600 m that crosses a platform in 39 seconds
    and a signal pole in 18 seconds, the length of the platform is 700 m. -/
theorem platform_length
  (train_length : ℝ)
  (time_platform : ℝ)
  (time_pole : ℝ)
  (h1 : train_length = 600)
  (h2 : time_platform = 39)
  (h3 : time_pole = 18) :
  (train_length * time_platform / time_pole) - train_length = 700 :=
by sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l3318_331866


namespace NUMINAMATH_CALUDE_valid_arrangements_l3318_331884

/-- The number of ways to arrange 4 boys and 4 girls in a row with specific conditions -/
def arrangement_count : ℕ := 504

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of girls -/
def num_girls : ℕ := 4

/-- The condition that adjacent individuals must be of opposite genders -/
def opposite_gender_adjacent : Prop := sorry

/-- The condition that a specific boy must stand next to a specific girl -/
def specific_pair_adjacent : Prop := sorry

/-- Theorem stating that the number of valid arrangements is 504 -/
theorem valid_arrangements :
  (num_boys = 4) →
  (num_girls = 4) →
  opposite_gender_adjacent →
  specific_pair_adjacent →
  arrangement_count = 504 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_l3318_331884


namespace NUMINAMATH_CALUDE_book_pages_from_digits_l3318_331885

/-- Given a book with pages numbered consecutively starting from 1,
    this function calculates the total number of digits used to number all pages. -/
def totalDigits (n : ℕ) : ℕ :=
  let oneDigit := min n 9
  let twoDigit := max 0 (min n 99 - 9)
  let threeDigit := max 0 (n - 99)
  oneDigit + 2 * twoDigit + 3 * threeDigit

/-- Theorem stating that a book with 672 digits used for page numbering has 260 pages. -/
theorem book_pages_from_digits :
  ∃ (n : ℕ), totalDigits n = 672 ∧ n = 260 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_from_digits_l3318_331885


namespace NUMINAMATH_CALUDE_corn_selling_price_l3318_331808

/-- Calculates the selling price per bag of corn to achieve a desired profit percentage --/
theorem corn_selling_price 
  (seed_cost fertilizer_cost labor_cost : ℕ) 
  (num_bags : ℕ) 
  (profit_percentage : ℚ) 
  (h1 : seed_cost = 50)
  (h2 : fertilizer_cost = 35)
  (h3 : labor_cost = 15)
  (h4 : num_bags = 10)
  (h5 : profit_percentage = 10 / 100) :
  (seed_cost + fertilizer_cost + labor_cost : ℚ) * (1 + profit_percentage) / num_bags = 11 := by
sorry

end NUMINAMATH_CALUDE_corn_selling_price_l3318_331808


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3318_331807

def A : Set ℝ := {x | x ≤ 1}
def B : Set ℝ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3318_331807


namespace NUMINAMATH_CALUDE_edge_sum_is_96_l3318_331836

/-- A rectangular solid with specific properties -/
structure RectangularSolid where
  -- Three dimensions in geometric progression
  a : ℝ
  r : ℝ
  -- Volume is 512 cm³
  volume_eq : a * (a * r) * (a * r * r) = 512
  -- Surface area is 384 cm²
  surface_area_eq : 2 * (a * (a * r) + a * (a * r * r) + (a * r) * (a * r * r)) = 384

/-- The sum of all edge lengths of the rectangular solid is 96 cm -/
theorem edge_sum_is_96 (solid : RectangularSolid) :
  4 * (solid.a + solid.a * solid.r + solid.a * solid.r * solid.r) = 96 := by
  sorry

end NUMINAMATH_CALUDE_edge_sum_is_96_l3318_331836


namespace NUMINAMATH_CALUDE_right_triangle_arctan_sum_l3318_331829

theorem right_triangle_arctan_sum (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) + Real.arctan (c / (a + b)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arctan_sum_l3318_331829


namespace NUMINAMATH_CALUDE_aaron_brothers_count_l3318_331878

/-- 
Given that Bennett has 6 brothers and the number of Bennett's brothers is two less than twice 
the number of Aaron's brothers, prove that Aaron has 4 brothers.
-/
theorem aaron_brothers_count :
  -- Define the number of Bennett's brothers
  let bennett_brothers : ℕ := 6
  -- Define the relationship between Aaron's and Bennett's brothers
  ∀ aaron_brothers : ℕ, bennett_brothers = 2 * aaron_brothers - 2 →
  -- Prove that Aaron has 4 brothers
  aaron_brothers = 4 := by
sorry

end NUMINAMATH_CALUDE_aaron_brothers_count_l3318_331878


namespace NUMINAMATH_CALUDE_fib_inequality_fib_upper_bound_l3318_331874

-- Define Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

-- Statement 1
theorem fib_inequality (n : ℕ) (h : n ≥ 2) : fib (n + 5) > 10 * fib n := by
  sorry

-- Statement 2
theorem fib_upper_bound (n k : ℕ) (h : fib (n + 1) < 10^k) : n ≤ 5 * k := by
  sorry

end NUMINAMATH_CALUDE_fib_inequality_fib_upper_bound_l3318_331874


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3318_331803

def repeating_decimal_4 : ℚ := 4/9
def repeating_decimal_7 : ℚ := 7/9
def repeating_decimal_3 : ℚ := 1/3

theorem sum_of_repeating_decimals :
  repeating_decimal_4 + repeating_decimal_7 - repeating_decimal_3 = 8/9 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3318_331803


namespace NUMINAMATH_CALUDE_pond_volume_1400_l3318_331869

/-- The volume of a rectangular prism-shaped pond -/
def pond_volume (length width depth : ℝ) : ℝ := length * width * depth

/-- Theorem: The volume of a pond with dimensions 28 m x 10 m x 5 m is 1400 cubic meters -/
theorem pond_volume_1400 :
  pond_volume 28 10 5 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_pond_volume_1400_l3318_331869


namespace NUMINAMATH_CALUDE_triple_solution_l3318_331856

theorem triple_solution (a b c : ℝ) : 
  a * b * c = 8 ∧ 
  a^2 * b + b^2 * c + c^2 * a = 73 ∧ 
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = 98 →
  ((a = 4 ∧ b = 4 ∧ c = 1/2) ∨
   (a = 4 ∧ b = 1/2 ∧ c = 4) ∨
   (a = 1/2 ∧ b = 4 ∧ c = 4) ∨
   (a = 1 ∧ b = 1 ∧ c = 8) ∨
   (a = 1 ∧ b = 8 ∧ c = 1) ∨
   (a = 8 ∧ b = 1 ∧ c = 1)) := by
  sorry

end NUMINAMATH_CALUDE_triple_solution_l3318_331856


namespace NUMINAMATH_CALUDE_base_conversion_and_arithmetic_l3318_331859

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

def decimal_2468_7 : Nat := base_to_decimal [8, 6, 4, 2] 7
def decimal_121_5 : Nat := base_to_decimal [1, 2, 1] 5
def decimal_3451_6 : Nat := base_to_decimal [1, 5, 4, 3] 6
def decimal_7891_7 : Nat := base_to_decimal [1, 9, 8, 7] 7

theorem base_conversion_and_arithmetic :
  (decimal_2468_7 / decimal_121_5 : Nat) - decimal_3451_6 + decimal_7891_7 = 2059 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_and_arithmetic_l3318_331859


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l3318_331850

theorem max_sum_of_factors (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 1995 →
  ∀ w x y z : ℕ+, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
    w * x * y * z = 1995 →
    a + b + c + d ≥ w + x + y + z :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l3318_331850


namespace NUMINAMATH_CALUDE_large_tent_fabric_is_8_l3318_331879

/-- The amount of fabric needed for a small tent -/
def small_tent_fabric : ℝ := 4

/-- The amount of fabric needed for a large tent -/
def large_tent_fabric : ℝ := 2 * small_tent_fabric

/-- Theorem: The fabric needed for a large tent is 8 square meters -/
theorem large_tent_fabric_is_8 : large_tent_fabric = 8 := by
  sorry

end NUMINAMATH_CALUDE_large_tent_fabric_is_8_l3318_331879


namespace NUMINAMATH_CALUDE_union_complement_equal_to_set_l3318_331828

def U : Finset Nat := {0, 1, 2, 4, 6, 8}
def M : Finset Nat := {0, 4, 6}
def N : Finset Nat := {0, 1, 6}

theorem union_complement_equal_to_set :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} := by sorry

end NUMINAMATH_CALUDE_union_complement_equal_to_set_l3318_331828


namespace NUMINAMATH_CALUDE_digit_removal_theorem_l3318_331801

theorem digit_removal_theorem :
  (∀ (n : ℕ), n ≥ 2 → 
    (∃! (x : ℕ), x = 625 * 10^(n-2) ∧ 
      (∃ (m : ℕ), x = 6 * 10^n + m ∧ m = x / 25))) ∧
  (¬ ∃ (x : ℕ), ∃ (n : ℕ), ∃ (m : ℕ), 
    x = 6 * 10^n + m ∧ m = x / 35) :=
by sorry

end NUMINAMATH_CALUDE_digit_removal_theorem_l3318_331801


namespace NUMINAMATH_CALUDE_residue_of_negative_998_mod_28_l3318_331845

theorem residue_of_negative_998_mod_28 :
  ∃ (q : ℤ), -998 = 28 * q + 10 ∧ (0 ≤ 10) ∧ (10 < 28) := by sorry

end NUMINAMATH_CALUDE_residue_of_negative_998_mod_28_l3318_331845


namespace NUMINAMATH_CALUDE_tangent_line_and_extrema_l3318_331877

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

theorem tangent_line_and_extrema :
  let a : ℝ := 0
  let b : ℝ := Real.pi / 2
  -- Tangent line at (0, f(0)) is y = 1
  (∀ y, HasDerivAt f 0 y → y = 0) ∧
  f 0 = 1 ∧
  -- Maximum value is 1 at x = 0
  (∀ x ∈ Set.Icc a b, f x ≤ f 0) ∧
  -- Minimum value is -π/2 at x = π/2
  (∀ x ∈ Set.Icc a b, f b ≤ f x) ∧
  f b = -Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_and_extrema_l3318_331877


namespace NUMINAMATH_CALUDE_intersecting_lines_sum_l3318_331892

/-- Two lines intersecting at a point implies a specific sum of their slopes and y-intercepts -/
theorem intersecting_lines_sum (m b : ℚ) : 
  (∀ x y : ℚ, y = m * x + 2 → y = 4 * x + b → x = 4 ∧ y = 8) → 
  b + m = -13/2 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_sum_l3318_331892


namespace NUMINAMATH_CALUDE_no_integer_satisfies_conditions_l3318_331832

theorem no_integer_satisfies_conditions : 
  ¬∃ (n : ℤ), (30 - 6 * n > 18) ∧ (2 * n + 5 = 11) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_satisfies_conditions_l3318_331832


namespace NUMINAMATH_CALUDE_percent_relation_l3318_331868

theorem percent_relation (a b : ℝ) (h : a = 1.25 * b) : 4 * b = 3.2 * a := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l3318_331868


namespace NUMINAMATH_CALUDE_range_of_expression_l3318_331830

theorem range_of_expression (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < π / 2) 
  (h3 : 0 < β) (h4 : β < π / 2) : 
  -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l3318_331830


namespace NUMINAMATH_CALUDE_tan_two_fifths_pi_plus_theta_l3318_331871

theorem tan_two_fifths_pi_plus_theta (θ : ℝ) 
  (h : Real.sin ((12 / 5) * π + θ) + 2 * Real.sin ((11 / 10) * π - θ) = 0) : 
  Real.tan ((2 / 5) * π + θ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_fifths_pi_plus_theta_l3318_331871


namespace NUMINAMATH_CALUDE_min_value_theorem_l3318_331889

theorem min_value_theorem (a b : ℝ) (h : a * b > 0) :
  a^2 + 4*b^2 + 1/(a*b) ≥ 4 ∧
  (a^2 + 4*b^2 + 1/(a*b) = 4 ↔ a = 1/Real.rpow 2 (1/4) ∧ b = 1/Real.rpow 2 (1/4)) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3318_331889


namespace NUMINAMATH_CALUDE_circle_passes_through_points_and_center_on_line_l3318_331841

-- Define the points M and N
def M : ℝ × ℝ := (5, 2)
def N : ℝ × ℝ := (3, 2)

-- Define the line equation y = 2x - 3
def line_equation (x y : ℝ) : Prop := y = 2 * x - 3

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 10

-- Theorem statement
theorem circle_passes_through_points_and_center_on_line :
  ∃ (center : ℝ × ℝ),
    line_equation center.1 center.2 ∧
    circle_equation M.1 M.2 ∧
    circle_equation N.1 N.2 :=
sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_and_center_on_line_l3318_331841


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l3318_331844

theorem sum_of_reciprocal_relations (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : 1/x - 1/y = -6) : 
  x + y = -4/5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l3318_331844


namespace NUMINAMATH_CALUDE_region_perimeter_l3318_331880

theorem region_perimeter (total_area : ℝ) (num_squares : ℕ) (row1_squares row2_squares : ℕ) :
  total_area = 400 →
  num_squares = 8 →
  row1_squares = 3 →
  row2_squares = 5 →
  row1_squares + row2_squares = num_squares →
  let square_area := total_area / num_squares
  let side_length := Real.sqrt square_area
  let perimeter := (2 * (row1_squares + row2_squares) + 2) * side_length
  perimeter = 90 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_region_perimeter_l3318_331880


namespace NUMINAMATH_CALUDE_solution_theorem_l3318_331887

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * a * x + a^2 - 3

theorem solution_theorem :
  -- Part I
  (∃ (l b : ℝ), ∀ x, f 3 x < 0 ↔ l < x ∧ x < b) ∧
  (∃ (l : ℝ), ∀ x, f 3 x < 0 ↔ l < x ∧ x < 2) ∧
  -- Part II
  (∀ a : ℝ, a < 0 → (∀ x, -3 ≤ x ∧ x ≤ 3 → f a x < 4) → -7/4 < a) ∧
  (∀ a : ℝ, a < 0 → (∀ x, -3 ≤ x ∧ x ≤ 3 → f a x < 4) → a < 0) :=
by sorry

end NUMINAMATH_CALUDE_solution_theorem_l3318_331887


namespace NUMINAMATH_CALUDE_min_value_a_l3318_331811

theorem min_value_a (a : ℝ) (h1 : a > 0) :
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → (x + y) * (1 / x + a / y) ≥ 16) →
  a ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l3318_331811


namespace NUMINAMATH_CALUDE_intersection_equals_target_l3318_331875

-- Define the sets M and N
def M : Set ℝ := {x | Real.log (x - 1) < 0}
def N : Set ℝ := {x | 2 * x^2 - 3 * x ≤ 0}

-- Define the intersection of M and N
def M_intersect_N : Set ℝ := M ∩ N

-- Define the open-closed interval (1, 3/2]
def target_set : Set ℝ := {x | 1 < x ∧ x ≤ 3/2}

-- Theorem statement
theorem intersection_equals_target : M_intersect_N = target_set := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_target_l3318_331875


namespace NUMINAMATH_CALUDE_product_inequality_l3318_331802

theorem product_inequality (a b c x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (hx₃ : 0 < x₃) (hx₄ : 0 < x₄) (hx₅ : 0 < x₅)
  (sum_abc : a + b + c = 1)
  (prod_x : x₁ * x₂ * x₃ * x₄ * x₅ = 1) :
  (a * x₁^2 + b * x₁ + c) * 
  (a * x₂^2 + b * x₂ + c) * 
  (a * x₃^2 + b * x₃ + c) * 
  (a * x₄^2 + b * x₄ + c) * 
  (a * x₅^2 + b * x₅ + c) ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_product_inequality_l3318_331802


namespace NUMINAMATH_CALUDE_odd_square_sum_of_consecutive_b_l3318_331897

def a : ℕ → ℕ
  | n => if n % 2 = 1 then 4 * ((n - 1) / 2) + 2 else 4 * (n / 2 - 1) + 3

def b : ℕ → ℕ
  | n => if n % 2 = 1 then 8 * ((n - 1) / 2) + 3 else 8 * (n / 2 - 1) + 6

theorem odd_square_sum_of_consecutive_b (k : ℕ) (hk : k > 0) :
  ∃ r : ℕ, (2 * k + 1)^2 = b r + b (r + 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_square_sum_of_consecutive_b_l3318_331897


namespace NUMINAMATH_CALUDE_water_mixture_percentage_l3318_331812

theorem water_mixture_percentage (initial_volume : ℝ) (added_water : ℝ) (final_water_percentage : ℝ) :
  initial_volume = 150 ∧
  added_water = 10 ∧
  final_water_percentage = 25 →
  (initial_volume * (20 / 100) + added_water) / (initial_volume + added_water) = final_water_percentage / 100 :=
by sorry

end NUMINAMATH_CALUDE_water_mixture_percentage_l3318_331812


namespace NUMINAMATH_CALUDE_ms_jones_class_size_l3318_331814

theorem ms_jones_class_size :
  ∀ (total_students : ℕ),
    (total_students : ℝ) * 0.3 * (1/3) * 10 = 50 →
    total_students = 50 := by
  sorry

end NUMINAMATH_CALUDE_ms_jones_class_size_l3318_331814


namespace NUMINAMATH_CALUDE_symmetric_cubic_at_1_l3318_331895

/-- A cubic function f(x) = x³ + ax² + bx + 2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 2

/-- The function f is symmetric about the point (2,0) -/
def is_symmetric_about_2_0 (a b : ℝ) : Prop :=
  ∀ x : ℝ, f a b (x + 2) = f a b (2 - x)

theorem symmetric_cubic_at_1 (a b : ℝ) : 
  is_symmetric_about_2_0 a b → f a b 1 = 4 := by
sorry

end NUMINAMATH_CALUDE_symmetric_cubic_at_1_l3318_331895


namespace NUMINAMATH_CALUDE_new_partner_associate_ratio_l3318_331806

/-- Given a firm with partners and associates, this theorem proves the new ratio
    after hiring additional associates. -/
theorem new_partner_associate_ratio
  (initial_partner_count : ℕ)
  (initial_associate_count : ℕ)
  (additional_associates : ℕ)
  (h1 : initial_partner_count = 18)
  (h2 : initial_associate_count = 567)
  (h3 : additional_associates = 45) :
  (initial_partner_count : ℚ) / (initial_associate_count + additional_associates : ℚ) = 1 / 34 := by
  sorry

#check new_partner_associate_ratio

end NUMINAMATH_CALUDE_new_partner_associate_ratio_l3318_331806


namespace NUMINAMATH_CALUDE_xy_length_l3318_331825

/-- Triangle similarity and side length properties -/
structure TriangleSimilarity where
  PQ : ℝ
  QR : ℝ
  YZ : ℝ
  perimeter_XYZ : ℝ
  similar : Bool  -- Represents that PQR is similar to XYZ

/-- Main theorem: XY length in similar triangles -/
theorem xy_length (t : TriangleSimilarity) 
  (h1 : t.PQ = 8)
  (h2 : t.QR = 16)
  (h3 : t.YZ = 24)
  (h4 : t.perimeter_XYZ = 60)
  (h5 : t.similar = true) : 
  ∃ XY : ℝ, XY = 12 ∧ XY + t.YZ + (t.perimeter_XYZ - XY - t.YZ) = t.perimeter_XYZ :=
sorry

end NUMINAMATH_CALUDE_xy_length_l3318_331825


namespace NUMINAMATH_CALUDE_half_sum_of_odd_squares_is_sum_of_squares_l3318_331898

theorem half_sum_of_odd_squares_is_sum_of_squares (a b : ℕ) (ha : Odd a) (hb : Odd b) (hab : a ≠ b) :
  ∃ x y : ℕ, (a^2 + b^2) / 2 = x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_half_sum_of_odd_squares_is_sum_of_squares_l3318_331898


namespace NUMINAMATH_CALUDE_set_expressions_correct_l3318_331860

def solution_set : Set ℝ := {x | x^2 - 4 = 0}
def prime_set : Set ℕ := {p | Nat.Prime p ∧ 0 < 2 * p ∧ 2 * p < 18}
def even_set : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}
def fourth_quadrant : Set (ℝ × ℝ) := {p | p.1 > 0 ∧ p.2 < 0}

theorem set_expressions_correct :
  solution_set = {-2, 2} ∧
  prime_set = {2, 3, 5, 7} ∧
  even_set = {x | ∃ n : ℤ, x = 2 * n} ∧
  fourth_quadrant = {p : ℝ × ℝ | p.1 > 0 ∧ p.2 < 0} :=
by sorry

end NUMINAMATH_CALUDE_set_expressions_correct_l3318_331860


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l3318_331813

theorem cube_volume_surface_area (x : ℝ) : x > 0 → 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 7*x ∧ 6*s^2 = 2*x) → x = 1323 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l3318_331813


namespace NUMINAMATH_CALUDE_symmetry_about_x_equals_one_l3318_331820

theorem symmetry_about_x_equals_one (f : ℝ → ℝ) (x : ℝ) : f (x - 1) = f (-(x - 1) + 1) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_about_x_equals_one_l3318_331820


namespace NUMINAMATH_CALUDE_binomial_600_600_eq_1_l3318_331815

theorem binomial_600_600_eq_1 : Nat.choose 600 600 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_600_600_eq_1_l3318_331815


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3318_331839

def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, -6]

theorem parallel_vectors_x_value :
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b x) → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3318_331839


namespace NUMINAMATH_CALUDE_log_difference_decreases_l3318_331835

theorem log_difference_decreases (m n : ℕ) (h : m > n) :
  Real.log (1 + 1 / (m : ℝ)) < Real.log (1 + 1 / (n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_log_difference_decreases_l3318_331835


namespace NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l3318_331876

theorem chinese_remainder_theorem_example (x : ℤ) : 
  x ≡ 2 [ZMOD 7] → x ≡ 3 [ZMOD 6] → x ≡ 9 [ZMOD 42] := by
  sorry

end NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l3318_331876


namespace NUMINAMATH_CALUDE_system_solution_l3318_331822

theorem system_solution : 
  ∃ (x₁ x₂ y₁ y₂ : ℝ), 
    (2 * x₁^2 - 5 * x₁ + 3 = 0) ∧ 
    (y₁ = 3 * x₁ + 1) ∧
    (2 * x₂^2 - 5 * x₂ + 3 = 0) ∧ 
    (y₂ = 3 * x₂ + 1) ∧
    (x₁ = 1.5 ∧ y₁ = 5.5) ∧ 
    (x₂ = 1 ∧ y₂ = 4) ∧
    (∀ (x y : ℝ), (2 * x^2 - 5 * x + 3 = 0) ∧ (y = 3 * x + 1) → 
      ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3318_331822


namespace NUMINAMATH_CALUDE_greatest_prime_divisor_plus_floor_sqrt_equality_l3318_331894

def greatest_prime_divisor (n : ℕ) : ℕ := sorry

def floor_sqrt (n : ℕ) : ℕ := sorry

theorem greatest_prime_divisor_plus_floor_sqrt_equality (n : ℕ) :
  n ≥ 2 →
  (greatest_prime_divisor n + floor_sqrt n = greatest_prime_divisor (n + 1) + floor_sqrt (n + 1)) ↔
  n = 3 := by sorry

end NUMINAMATH_CALUDE_greatest_prime_divisor_plus_floor_sqrt_equality_l3318_331894


namespace NUMINAMATH_CALUDE_mutuallyExclusive_but_not_complementary_l3318_331804

/-- Represents the color of a ball -/
inductive Color
| Red
| White

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first : Color)
  (second : Color)

/-- The set of all possible outcomes when drawing two balls -/
def sampleSpace : Set DrawOutcome := sorry

/-- The event of drawing exactly one white ball -/
def exactlyOneWhite : Set DrawOutcome := sorry

/-- The event of drawing exactly two white balls -/
def exactlyTwoWhite : Set DrawOutcome := sorry

/-- Two events are mutually exclusive if their intersection is empty -/
def mutuallyExclusive (A B : Set DrawOutcome) : Prop :=
  A ∩ B = ∅

/-- Two events are complementary if their union is the entire sample space -/
def complementary (A B : Set DrawOutcome) : Prop :=
  A ∪ B = sampleSpace

theorem mutuallyExclusive_but_not_complementary :
  mutuallyExclusive exactlyOneWhite exactlyTwoWhite ∧
  ¬complementary exactlyOneWhite exactlyTwoWhite :=
sorry

end NUMINAMATH_CALUDE_mutuallyExclusive_but_not_complementary_l3318_331804


namespace NUMINAMATH_CALUDE_tiffany_fastest_l3318_331858

structure Runner where
  name : String
  uphill_blocks : ℕ
  uphill_time : ℕ
  downhill_blocks : ℕ
  downhill_time : ℕ
  flat_blocks : ℕ
  flat_time : ℕ

def total_distance (r : Runner) : ℕ :=
  r.uphill_blocks + r.downhill_blocks + r.flat_blocks

def total_time (r : Runner) : ℕ :=
  r.uphill_time + r.downhill_time + r.flat_time

def average_speed (r : Runner) : ℚ :=
  (total_distance r : ℚ) / (total_time r : ℚ)

def tiffany : Runner :=
  { name := "Tiffany"
    uphill_blocks := 6
    uphill_time := 3
    downhill_blocks := 8
    downhill_time := 5
    flat_blocks := 6
    flat_time := 3 }

def moses : Runner :=
  { name := "Moses"
    uphill_blocks := 5
    uphill_time := 5
    downhill_blocks := 10
    downhill_time := 10
    flat_blocks := 5
    flat_time := 4 }

def morgan : Runner :=
  { name := "Morgan"
    uphill_blocks := 7
    uphill_time := 4
    downhill_blocks := 9
    downhill_time := 6
    flat_blocks := 4
    flat_time := 2 }

theorem tiffany_fastest : 
  average_speed tiffany > average_speed moses ∧ 
  average_speed tiffany > average_speed morgan ∧
  total_distance tiffany = 20 ∧
  total_distance moses = 20 ∧
  total_distance morgan = 20 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_fastest_l3318_331858


namespace NUMINAMATH_CALUDE_total_candies_l3318_331805

/-- The number of jellybeans in a dozen -/
def dozen : ℕ := 12

/-- Caleb's candies -/
def caleb_jellybeans : ℕ := 3 * dozen
def caleb_chocolate_bars : ℕ := 5
def caleb_gummy_bears : ℕ := 8

/-- Sophie's candies -/
def sophie_jellybeans : ℕ := caleb_jellybeans / 2
def sophie_chocolate_bars : ℕ := 3
def sophie_gummy_bears : ℕ := 12

/-- Max's candies -/
def max_jellybeans : ℕ := sophie_jellybeans + 2 * dozen
def max_chocolate_bars : ℕ := 6
def max_gummy_bears : ℕ := 10

/-- Total candies for each person -/
def caleb_total : ℕ := caleb_jellybeans + caleb_chocolate_bars + caleb_gummy_bears
def sophie_total : ℕ := sophie_jellybeans + sophie_chocolate_bars + sophie_gummy_bears
def max_total : ℕ := max_jellybeans + max_chocolate_bars + max_gummy_bears

/-- Theorem: The total number of candies is 140 -/
theorem total_candies : caleb_total + sophie_total + max_total = 140 := by
  sorry

end NUMINAMATH_CALUDE_total_candies_l3318_331805


namespace NUMINAMATH_CALUDE_periodic_sum_implies_periodic_increasing_sum_not_implies_increasing_l3318_331893

def PeriodicFunction (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem periodic_sum_implies_periodic
  (f g h : ℝ → ℝ) (T : ℝ) :
  (PeriodicFunction (fun x ↦ f x + g x) T) →
  (PeriodicFunction (fun x ↦ f x + h x) T) →
  (PeriodicFunction (fun x ↦ g x + h x) T) →
  (PeriodicFunction f T) ∧ (PeriodicFunction g T) ∧ (PeriodicFunction h T) := by
  sorry

theorem increasing_sum_not_implies_increasing :
  ∃ f g h : ℝ → ℝ,
    (IncreasingFunction (fun x ↦ f x + g x)) ∧
    (IncreasingFunction (fun x ↦ f x + h x)) ∧
    (IncreasingFunction (fun x ↦ g x + h x)) ∧
    (¬IncreasingFunction f ∨ ¬IncreasingFunction g ∨ ¬IncreasingFunction h) := by
  sorry

end NUMINAMATH_CALUDE_periodic_sum_implies_periodic_increasing_sum_not_implies_increasing_l3318_331893


namespace NUMINAMATH_CALUDE_remainder_theorem_l3318_331854

theorem remainder_theorem (x : ℂ) : 
  (x^2023 + 1) % (x^10 - x^8 + x^6 - x^4 + x^2 - 1) = -x^7 + 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3318_331854


namespace NUMINAMATH_CALUDE_smallest_divisible_by_999_l3318_331870

theorem smallest_divisible_by_999 :
  ∃ (a : ℕ), (∀ (n : ℕ), Odd n → (999 ∣ 2^(5*n) + a*5^n)) ∧ 
  (∀ (b : ℕ), b < a → ∃ (m : ℕ), Odd m ∧ ¬(999 ∣ 2^(5*m) + b*5^m)) ∧
  a = 539 := by
sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_999_l3318_331870


namespace NUMINAMATH_CALUDE_mountain_loop_trail_length_l3318_331848

/-- Represents the hiking trip on Mountain Loop Trail -/
structure HikingTrip where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ

/-- The conditions of the hiking trip -/
def validHikingTrip (trip : HikingTrip) : Prop :=
  trip.day1 + trip.day2 + trip.day3 = 45 ∧
  (trip.day2 + trip.day4) / 2 = 18 ∧
  trip.day3 + trip.day4 + trip.day5 = 60 ∧
  trip.day1 + trip.day4 = 32

/-- The theorem stating the total length of the trail -/
theorem mountain_loop_trail_length (trip : HikingTrip) 
  (h : validHikingTrip trip) : 
  trip.day1 + trip.day2 + trip.day3 + trip.day4 + trip.day5 = 69 := by
  sorry


end NUMINAMATH_CALUDE_mountain_loop_trail_length_l3318_331848


namespace NUMINAMATH_CALUDE_complex_real_condition_l3318_331826

theorem complex_real_condition (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (↑(1 : ℝ) - Complex.I) * (↑a + Complex.I) ∈ Set.range (Complex.ofReal) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_real_condition_l3318_331826


namespace NUMINAMATH_CALUDE_paper_pieces_sum_l3318_331855

/-- The number of pieces of paper picked up by Olivia and Edward -/
theorem paper_pieces_sum (olivia_pieces edward_pieces : ℕ) 
  (h_olivia : olivia_pieces = 16) 
  (h_edward : edward_pieces = 3) : 
  olivia_pieces + edward_pieces = 19 := by
  sorry

end NUMINAMATH_CALUDE_paper_pieces_sum_l3318_331855


namespace NUMINAMATH_CALUDE_expression_simplification_l3318_331888

theorem expression_simplification (x : ℤ) 
  (h1 : 2 * (x - 1) < x + 1) 
  (h2 : 5 * x + 3 ≥ 2 * x) : 
  (2 : ℚ) / (x^2 + x) / (1 - (x - 1) / (x^2 - 1)) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3318_331888


namespace NUMINAMATH_CALUDE_degree_of_specific_monomial_l3318_331890

/-- The degree of a monomial is the sum of the exponents of its variables -/
def degree_of_monomial (m : ℕ) (n : ℕ) : ℕ := m + n

/-- The degree of the monomial (1/7)mn^2 is 3 -/
theorem degree_of_specific_monomial : degree_of_monomial 1 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_specific_monomial_l3318_331890


namespace NUMINAMATH_CALUDE_monroe_collection_legs_l3318_331883

def spider_count : ℕ := 8
def ant_count : ℕ := 12
def spider_legs : ℕ := 8
def ant_legs : ℕ := 6

def total_legs : ℕ := spider_count * spider_legs + ant_count * ant_legs

theorem monroe_collection_legs : total_legs = 136 := by
  sorry

end NUMINAMATH_CALUDE_monroe_collection_legs_l3318_331883


namespace NUMINAMATH_CALUDE_m_value_proof_l3318_331882

theorem m_value_proof (a b c d e f : ℝ) (m n : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0) (h6 : f > 0)
  (h7 : a^m = b^3)
  (h8 : c^n = d^2)
  (h9 : ((a^(m+1))/(b^(m+1))) * ((c^n)/(d^n)) = 1/(2*(e^(35*f)))) :
  m = 3 := by sorry

end NUMINAMATH_CALUDE_m_value_proof_l3318_331882


namespace NUMINAMATH_CALUDE_exponential_function_point_l3318_331852

theorem exponential_function_point (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x + 1
  f 0 = 2 := by sorry

end NUMINAMATH_CALUDE_exponential_function_point_l3318_331852


namespace NUMINAMATH_CALUDE_no_solution_implies_m_equals_six_l3318_331819

/-- If the equation (3x - m) / (x - 2) = 1 has no solution, then m = 6 -/
theorem no_solution_implies_m_equals_six (m : ℝ) : 
  (∀ x : ℝ, (3 * x - m) / (x - 2) ≠ 1) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_equals_six_l3318_331819


namespace NUMINAMATH_CALUDE_min_value_cosine_function_l3318_331810

theorem min_value_cosine_function :
  ∀ x : ℝ, 2 * Real.cos x - 1 ≥ -3 ∧ ∃ x : ℝ, 2 * Real.cos x - 1 = -3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_cosine_function_l3318_331810


namespace NUMINAMATH_CALUDE_remainder_of_2357916_div_8_l3318_331837

theorem remainder_of_2357916_div_8 : 2357916 % 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_2357916_div_8_l3318_331837


namespace NUMINAMATH_CALUDE_no_digit_satisfies_property_l3318_331833

/-- Represents a single digit (0-9) -/
def Digit := Fin 10

/-- Checks if the decimal representation of a natural number ends with at least k repetitions of a digit -/
def EndsWithRepeatedDigit (num : ℕ) (d : Digit) (k : ℕ) : Prop :=
  ∃ m : ℕ, num % (10^k) = d.val * ((10^k - 1) / 9)

/-- The main theorem stating that no digit satisfies the given property -/
theorem no_digit_satisfies_property : 
  ¬ ∃ z : Digit, ∀ k : ℕ, k ≥ 1 → ∃ n : ℕ, n ≥ 1 ∧ EndsWithRepeatedDigit (n^9) z k :=
by sorry


end NUMINAMATH_CALUDE_no_digit_satisfies_property_l3318_331833


namespace NUMINAMATH_CALUDE_sin_q_in_special_right_triangle_l3318_331867

/-- Given a right triangle PQR with ∠P as the right angle, PR = 40, and QR = 41, prove that sin Q = 9/41 -/
theorem sin_q_in_special_right_triangle (P Q R : ℝ × ℝ) : 
  let pq := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  let pr := Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2)
  let qr := Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2)
  (pq^2 + pr^2 = qr^2) →  -- Right angle at P
  pr = 40 →
  qr = 41 →
  (Q.2 - P.2) / qr = 9 / 41 :=
by sorry

end NUMINAMATH_CALUDE_sin_q_in_special_right_triangle_l3318_331867


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_four_l3318_331857

theorem largest_digit_divisible_by_four :
  ∀ n : ℕ, 
    (n = 4969794) → 
    (∀ m : ℕ, m % 4 = 0 ↔ (m % 100) % 4 = 0) → 
    n % 4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_four_l3318_331857


namespace NUMINAMATH_CALUDE_quadratic_solution_inequality_solution_l3318_331818

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := 2 * x^2 + x - 2 = 0

-- Define the inequality system
def inequality_system (x : ℝ) : Prop := x + 3 > -2*x ∧ 2*x - 5 < 1

-- Theorem for the quadratic equation solution
theorem quadratic_solution :
  ∃ x1 x2 : ℝ, x1 = (-1 + Real.sqrt 17) / 4 ∧
              x2 = (-1 - Real.sqrt 17) / 4 ∧
              quadratic_equation x1 ∧
              quadratic_equation x2 :=
sorry

-- Theorem for the inequality system solution
theorem inequality_solution :
  ∀ x : ℝ, inequality_system x ↔ -1 < x ∧ x < 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_solution_inequality_solution_l3318_331818


namespace NUMINAMATH_CALUDE_power_equality_l3318_331864

theorem power_equality (m n : ℕ) (h1 : 2^m = 5) (h2 : 4^n = 3) : 4^(3*n - m) = 27/25 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3318_331864


namespace NUMINAMATH_CALUDE_gabby_savings_l3318_331891

/-- Represents the cost of the makeup set in dollars -/
def makeup_cost : ℕ := 65

/-- Represents the amount Gabby's mom gives her in dollars -/
def mom_gift : ℕ := 20

/-- Represents the additional amount Gabby needs after receiving the gift in dollars -/
def additional_needed : ℕ := 10

/-- Represents Gabby's initial savings in dollars -/
def initial_savings : ℕ := 35

theorem gabby_savings :
  initial_savings + mom_gift + additional_needed = makeup_cost :=
by sorry

end NUMINAMATH_CALUDE_gabby_savings_l3318_331891


namespace NUMINAMATH_CALUDE_work_hours_theorem_l3318_331821

def amber_hours : ℕ := 12

def armand_hours : ℕ := amber_hours / 3

def ella_hours : ℕ := 2 * amber_hours

def total_hours : ℕ := amber_hours + armand_hours + ella_hours

theorem work_hours_theorem : total_hours = 40 := by
  sorry

end NUMINAMATH_CALUDE_work_hours_theorem_l3318_331821


namespace NUMINAMATH_CALUDE_A_subset_of_neg_one_one_l3318_331842

def A : Set ℝ := {x | x^2 - 1 = 0}

theorem A_subset_of_neg_one_one : A ⊆ {-1, 1} := by
  sorry

end NUMINAMATH_CALUDE_A_subset_of_neg_one_one_l3318_331842


namespace NUMINAMATH_CALUDE_power_of_power_l3318_331863

theorem power_of_power : (3^3)^4 = 531441 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3318_331863


namespace NUMINAMATH_CALUDE_fraction_problem_l3318_331824

theorem fraction_problem (N : ℝ) (x : ℝ) : 
  (0.40 * N = 420) → 
  (x * (1/3) * (2/5) * N = 35) → 
  x = 1/4 := by sorry

end NUMINAMATH_CALUDE_fraction_problem_l3318_331824


namespace NUMINAMATH_CALUDE_number_operation_result_l3318_331831

theorem number_operation_result : 
  let x : ℕ := 265
  (x / 5 + 8 : ℚ) = 61 := by sorry

end NUMINAMATH_CALUDE_number_operation_result_l3318_331831


namespace NUMINAMATH_CALUDE_subset_condition_l3318_331896

def A : Set ℝ := {x | (x - 3) / (x + 1) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | a * x + 1 ≤ 0}

theorem subset_condition (a : ℝ) : 
  B a ⊆ A ↔ a ∈ Set.Icc (-1/3) 1 := by sorry

end NUMINAMATH_CALUDE_subset_condition_l3318_331896


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_34_l3318_331853

theorem consecutive_integers_sum_34 :
  ∃! (a : ℕ), a > 0 ∧ (a + (a + 1) + (a + 2) + (a + 3) = 34) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_34_l3318_331853


namespace NUMINAMATH_CALUDE_arctan_sum_special_l3318_331823

theorem arctan_sum_special : Real.arctan (3/4) + Real.arctan (4/3) = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_special_l3318_331823


namespace NUMINAMATH_CALUDE_polynomial_derivative_bound_l3318_331861

theorem polynomial_derivative_bound (p : ℝ → ℝ) :
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → |p x| ≤ 1) →
  (∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c) →
  ∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → |(deriv p) x| ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_derivative_bound_l3318_331861


namespace NUMINAMATH_CALUDE_eighth_fib_is_21_l3318_331838

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

-- Theorem: The 8th term of the Fibonacci sequence is 21
theorem eighth_fib_is_21 : fib 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_eighth_fib_is_21_l3318_331838


namespace NUMINAMATH_CALUDE_chess_team_selection_ways_l3318_331843

def total_members : ℕ := 18
def num_siblings : ℕ := 4
def team_size : ℕ := 8
def max_siblings_in_team : ℕ := 2

theorem chess_team_selection_ways :
  (Nat.choose total_members team_size) -
  (Nat.choose num_siblings (num_siblings) * Nat.choose (total_members - num_siblings) (team_size - num_siblings)) = 42757 := by
  sorry

end NUMINAMATH_CALUDE_chess_team_selection_ways_l3318_331843


namespace NUMINAMATH_CALUDE_lisa_window_width_l3318_331873

/-- Represents the dimensions of a window pane -/
structure Pane where
  width : ℝ
  height : ℝ

/-- Represents the dimensions and layout of a window -/
structure Window where
  pane : Pane
  rows : ℕ
  columns : ℕ
  border_width : ℝ

/-- Calculates the total width of the window -/
def window_width (w : Window) : ℝ :=
  w.columns * w.pane.width + (w.columns + 1) * w.border_width

/-- The main theorem stating the width of Lisa's window -/
theorem lisa_window_width :
  ∃ (w : Window),
    w.rows = 3 ∧
    w.columns = 4 ∧
    w.border_width = 3 ∧
    w.pane.width / w.pane.height = 3 / 4 ∧
    window_width w = 51 :=
sorry

end NUMINAMATH_CALUDE_lisa_window_width_l3318_331873


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l3318_331862

/-- A parabola with vertex at the origin and axis of symmetry along coordinate axes -/
structure Parabola where
  focus_on_line : ∃ (x y : ℝ), 2*x - y - 4 = 0

/-- The standard equation of a parabola -/
inductive StandardEquation where
  | vert : StandardEquation  -- y² = 8x
  | horz : StandardEquation  -- x² = -16y

/-- Theorem: Given a parabola with vertex at origin, axis of symmetry along coordinate axes,
    and focus on the line 2x - y - 4 = 0, its standard equation is either y² = 8x or x² = -16y -/
theorem parabola_standard_equation (p : Parabola) : 
  ∃ (eq : StandardEquation), 
    (eq = StandardEquation.vert ∨ eq = StandardEquation.horz) := by
  sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l3318_331862


namespace NUMINAMATH_CALUDE_degree_plus_one_divides_l3318_331827

/-- A polynomial with coefficients in {1, 2022} -/
def SpecialPoly (R : Type*) [CommRing R] := Polynomial R

/-- Predicate to check if a polynomial has coefficients only in {1, 2022} -/
def HasSpecialCoeffs (p : Polynomial ℤ) : Prop :=
  ∀ (i : ℕ), p.coeff i = 1 ∨ p.coeff i = 2022 ∨ p.coeff i = 0

theorem degree_plus_one_divides
  (f g : Polynomial ℤ)
  (hf : HasSpecialCoeffs f)
  (hg : HasSpecialCoeffs g)
  (h_div : f ∣ g) :
  (Polynomial.degree f + 1) ∣ (Polynomial.degree g + 1) :=
sorry

end NUMINAMATH_CALUDE_degree_plus_one_divides_l3318_331827


namespace NUMINAMATH_CALUDE_tangent_line_condition_f_positive_condition_three_zeros_condition_l3318_331816

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + a / x

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 / x - a * (1 + 1 / x^2)

theorem tangent_line_condition (a : ℝ) :
  (f_deriv a 1 = (4 - f a 1) / 2) → a = -1/2 := by sorry

theorem f_positive_condition (a : ℝ) :
  0 < a → a < 1 → f a (a^2 / 2) > 0 := by sorry

theorem three_zeros_condition (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔
  (0 < a ∧ a < 1/2) := by sorry

end

end NUMINAMATH_CALUDE_tangent_line_condition_f_positive_condition_three_zeros_condition_l3318_331816


namespace NUMINAMATH_CALUDE_circle_with_diameter_AB_l3318_331881

-- Define the line segment AB
def line_segment_AB (x y : ℝ) : Prop :=
  x + y - 2 = 0 ∧ 0 ≤ x ∧ x ≤ 2

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

-- Theorem statement
theorem circle_with_diameter_AB :
  ∀ x y : ℝ, line_segment_AB x y →
  ∃ center_x center_y radius : ℝ,
    (∀ p q : ℝ, (p - center_x)^2 + (q - center_y)^2 = radius^2 ↔ circle_equation p q) :=
by sorry

end NUMINAMATH_CALUDE_circle_with_diameter_AB_l3318_331881


namespace NUMINAMATH_CALUDE_points_ABD_collinear_l3318_331809

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define vectors a and b
variable (a b : V)

-- Define points A, B, C, D
variable (A B C D : V)

-- State the theorem
theorem points_ABD_collinear
  (h_not_collinear : ¬ ∃ (k : ℝ), a = k • b)
  (h_AB : B - A = a + 2 • b)
  (h_BC : C - B = -3 • a + 7 • b)
  (h_CD : D - C = 4 • a - 5 • b) :
  ∃ (k : ℝ), D - A = k • (B - A) :=
sorry

end NUMINAMATH_CALUDE_points_ABD_collinear_l3318_331809


namespace NUMINAMATH_CALUDE_final_tree_count_l3318_331847

/-- The number of dogwood trees in the park after planting -/
def total_trees (initial : ℕ) (planted_today : ℕ) (planted_tomorrow : ℕ) : ℕ :=
  initial + planted_today + planted_tomorrow

/-- Theorem stating the final number of dogwood trees in the park -/
theorem final_tree_count :
  total_trees 7 5 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_final_tree_count_l3318_331847


namespace NUMINAMATH_CALUDE_classroom_students_l3318_331849

theorem classroom_students (total_notebooks : ℕ) (notebooks_per_half1 : ℕ) (notebooks_per_half2 : ℕ) :
  total_notebooks = 112 →
  notebooks_per_half1 = 5 →
  notebooks_per_half2 = 3 →
  ∃ (num_students : ℕ),
    num_students % 2 = 0 ∧
    (num_students / 2) * notebooks_per_half1 + (num_students / 2) * notebooks_per_half2 = total_notebooks ∧
    num_students = 28 := by
  sorry

end NUMINAMATH_CALUDE_classroom_students_l3318_331849


namespace NUMINAMATH_CALUDE_monthly_salary_is_1000_l3318_331851

/-- Calculates the monthly salary given savings rate, expense increase, and new savings amount -/
def calculate_salary (savings_rate : ℚ) (expense_increase : ℚ) (new_savings : ℚ) : ℚ :=
  new_savings / (savings_rate - (1 - savings_rate) * expense_increase)

/-- Theorem stating that under the given conditions, the monthly salary is 1000 -/
theorem monthly_salary_is_1000 : 
  let savings_rate : ℚ := 25 / 100
  let expense_increase : ℚ := 10 / 100
  let new_savings : ℚ := 175
  calculate_salary savings_rate expense_increase new_savings = 1000 := by
  sorry

#eval calculate_salary (25/100) (10/100) 175

end NUMINAMATH_CALUDE_monthly_salary_is_1000_l3318_331851


namespace NUMINAMATH_CALUDE_fraction_transformation_l3318_331899

theorem fraction_transformation (x : ℝ) (h : x ≠ 3) : -1 / (3 - x) = 1 / (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l3318_331899
