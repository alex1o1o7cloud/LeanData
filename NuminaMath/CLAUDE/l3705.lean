import Mathlib

namespace NUMINAMATH_CALUDE_scientific_notation_of_104000000_l3705_370548

theorem scientific_notation_of_104000000 :
  (104000000 : ℝ) = 1.04 * (10 ^ 8) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_104000000_l3705_370548


namespace NUMINAMATH_CALUDE_identical_views_solids_l3705_370522

-- Define the set of all possible solids
inductive Solid
  | Sphere
  | TriangularPyramid
  | Cube
  | Cylinder

-- Define a predicate for solids with identical views
def has_identical_views (s : Solid) : Prop :=
  match s with
  | Solid.Sphere => true
  | Solid.TriangularPyramid => true
  | Solid.Cube => true
  | Solid.Cylinder => false

-- Theorem stating that the set of solids with identical views
-- is equal to the set containing Sphere, Triangular Pyramid, and Cube
theorem identical_views_solids :
  {s : Solid | has_identical_views s} =
  {Solid.Sphere, Solid.TriangularPyramid, Solid.Cube} :=
by sorry

end NUMINAMATH_CALUDE_identical_views_solids_l3705_370522


namespace NUMINAMATH_CALUDE_composite_property_l3705_370550

theorem composite_property (n : ℕ) 
  (h1 : ∃ a : ℕ, 3 * n + 1 = a ^ 2)
  (h2 : ∃ b : ℕ, 10 * n + 1 = b ^ 2) : 
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ 29 * n + 11 = x * y :=
sorry

end NUMINAMATH_CALUDE_composite_property_l3705_370550


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_algebraic_expression_value_l3705_370592

-- Part 1
theorem simplify_and_evaluate (x : ℤ) :
  x = -3 → x^2 + 4*x - (2*x^2 - x + x^2) - (3*x - 1) = -23 := by sorry

-- Part 2
theorem algebraic_expression_value (m n : ℤ) :
  m + n = 2 → m * n = -3 → 2*(m*n + (-3*m)) - 3*(2*n - m*n) = -27 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_algebraic_expression_value_l3705_370592


namespace NUMINAMATH_CALUDE_min_value_of_sum_l3705_370561

theorem min_value_of_sum (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_eq : a^2 + 2*a*b + 2*a*c + 4*b*c = 12) : 
  ∀ x y z, x > 0 → y > 0 → z > 0 → x^2 + 2*x*y + 2*x*z + 4*y*z = 12 → 
  a + b + c ≤ x + y + z ∧ a + b + c = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l3705_370561


namespace NUMINAMATH_CALUDE_curve_is_rhombus_not_square_l3705_370569

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the curve defined by the equation (|x+y|)/(2a) + (|x-y|)/(2b) = 1 -/
def Curve (a b : ℝ) : Set Point :=
  {p : Point | (|p.x + p.y|) / (2 * a) + (|p.x - p.y|) / (2 * b) = 1}

/-- Checks if a quadrilateral is a rhombus -/
def is_rhombus (A B C D : Point) : Prop :=
  let AB := ((B.x - A.x)^2 + (B.y - A.y)^2).sqrt
  let BC := ((C.x - B.x)^2 + (C.y - B.y)^2).sqrt
  let CD := ((D.x - C.x)^2 + (D.y - C.y)^2).sqrt
  let DA := ((A.x - D.x)^2 + (A.y - D.y)^2).sqrt
  AB = BC ∧ BC = CD ∧ CD = DA

/-- Checks if a quadrilateral is a square -/
def is_square (A B C D : Point) : Prop :=
  is_rhombus A B C D ∧
  let AC := ((C.x - A.x)^2 + (C.y - A.y)^2).sqrt
  let BD := ((D.x - B.x)^2 + (D.y - B.y)^2).sqrt
  AC = BD

theorem curve_is_rhombus_not_square (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  ∃ A B C D : Point,
    A ∈ Curve a b ∧ B ∈ Curve a b ∧ C ∈ Curve a b ∧ D ∈ Curve a b ∧
    is_rhombus A B C D ∧
    ¬is_square A B C D :=
  sorry

end NUMINAMATH_CALUDE_curve_is_rhombus_not_square_l3705_370569


namespace NUMINAMATH_CALUDE_equation_solutions_l3705_370559

theorem equation_solutions :
  (∃ x : ℝ, 3 * x + 6 = 31 - 2 * x ∧ x = 5) ∧
  (∃ x : ℝ, 1 - 8 * (1/4 + 0.5 * x) = 3 * (1 - 2 * x) ∧ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3705_370559


namespace NUMINAMATH_CALUDE_trig_sum_equality_l3705_370555

theorem trig_sum_equality : 
  (Real.cos (2 * π / 180)) / (Real.sin (47 * π / 180)) + 
  (Real.cos (88 * π / 180)) / (Real.sin (133 * π / 180)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_equality_l3705_370555


namespace NUMINAMATH_CALUDE_exists_palindrome_multiple_l3705_370547

/-- A number is a decimal palindrome if its decimal representation is mirror symmetric. -/
def IsDecimalPalindrome (n : ℕ) : Prop :=
  ∃ (digits : List ℕ), digits.reverse = digits ∧ n = digits.foldl (fun acc d => acc * 10 + d) 0

/-- Main theorem: For any positive integer not divisible by 10, 
    there exists a positive multiple that is a decimal palindrome. -/
theorem exists_palindrome_multiple {n : ℕ} (hn : n > 0) (hndiv : ¬ 10 ∣ n) :
  ∃ (m : ℕ), m > 0 ∧ n ∣ m ∧ IsDecimalPalindrome m := by
  sorry

end NUMINAMATH_CALUDE_exists_palindrome_multiple_l3705_370547


namespace NUMINAMATH_CALUDE_special_functions_bound_l3705_370506

open Real

/-- Two differentiable real functions satisfying the given conditions -/
structure SpecialFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  hf : Differentiable ℝ f
  hg : Differentiable ℝ g
  h_eq : ∀ x, deriv f x / deriv g x = exp (f x - g x)
  h_f0 : f 0 = 1
  h_g2003 : g 2003 = 1

/-- The theorem stating that f(2003) > 1 - ln(2) for any pair of functions satisfying the conditions,
    and that 1 - ln(2) is the largest such constant -/
theorem special_functions_bound (sf : SpecialFunctions) :
  sf.f 2003 > 1 - log 2 ∧ ∀ c, (∀ sf' : SpecialFunctions, sf'.f 2003 > c) → c ≤ 1 - log 2 := by
  sorry

end NUMINAMATH_CALUDE_special_functions_bound_l3705_370506


namespace NUMINAMATH_CALUDE_june_population_estimate_l3705_370524

/-- Represents the number of rabbits tagged on June 1 -/
def tagged_rabbits : ℕ := 50

/-- Represents the number of rabbits captured on October 1 -/
def captured_rabbits : ℕ := 80

/-- Represents the number of tagged rabbits found in the October capture -/
def tagged_captured : ℕ := 4

/-- Represents the percentage of original population no longer in the forest by October -/
def predation_rate : ℚ := 30 / 100

/-- Represents the percentage of October rabbits that were not in the forest in June -/
def new_birth_rate : ℚ := 50 / 100

/-- Estimates the number of rabbits in the forest on June 1 -/
def estimate_june_population : ℕ := 500

theorem june_population_estimate :
  tagged_rabbits * (captured_rabbits * (1 - new_birth_rate)) / tagged_captured = estimate_june_population :=
sorry

end NUMINAMATH_CALUDE_june_population_estimate_l3705_370524


namespace NUMINAMATH_CALUDE_tree_height_proof_l3705_370584

/-- Represents the height of a tree as a function of its breast diameter -/
def tree_height (x : ℝ) : ℝ := 25 * x + 15

theorem tree_height_proof :
  (tree_height 0.2 = 20) ∧
  (tree_height 0.28 = 22) ∧
  (tree_height 0.3 = 22.5) := by
  sorry

end NUMINAMATH_CALUDE_tree_height_proof_l3705_370584


namespace NUMINAMATH_CALUDE_no_valid_tetrahedron_labeling_l3705_370594

/-- Represents a labeling of a tetrahedron's vertices -/
def TetrahedronLabeling := Fin 4 → Fin 4

/-- Checks if a labeling is valid (uses each number exactly once) -/
def isValidLabeling (l : TetrahedronLabeling) : Prop :=
  ∀ i : Fin 4, ∃! j : Fin 4, l j = i

/-- Represents a face of the tetrahedron as a set of three vertex indices -/
def TetrahedronFace := Fin 3 → Fin 4

/-- The four faces of a tetrahedron -/
def tetrahedronFaces : Fin 4 → TetrahedronFace := sorry

/-- The sum of labels on a face -/
def faceSum (l : TetrahedronLabeling) (f : TetrahedronFace) : Nat :=
  (f 0).val + 1 + (f 1).val + 1 + (f 2).val + 1

/-- Theorem: No valid labeling exists such that all face sums are equal -/
theorem no_valid_tetrahedron_labeling :
  ¬∃ (l : TetrahedronLabeling),
    isValidLabeling l ∧
    ∃ (s : Nat), ∀ (f : Fin 4), faceSum l (tetrahedronFaces f) = s :=
  sorry

end NUMINAMATH_CALUDE_no_valid_tetrahedron_labeling_l3705_370594


namespace NUMINAMATH_CALUDE_value_of_2a_minus_b_l3705_370589

-- Define the functions
def f (a b x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 6
def h (a b x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem value_of_2a_minus_b (a b : ℝ) :
  (∀ x, h a b x = x - 9) →  -- h is the inverse of x + 9
  2 * a - b = 7 := by
  sorry

end NUMINAMATH_CALUDE_value_of_2a_minus_b_l3705_370589


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l3705_370515

theorem trigonometric_equation_solution (x : ℝ) :
  0.5 * (Real.cos (5 * x) + Real.cos (7 * x)) - (Real.cos (2 * x))^2 + (Real.sin (3 * x))^2 = 0 ↔
  (∃ k : ℤ, x = π / 2 * (2 * k + 1)) ∨ (∃ k : ℤ, x = 2 * k * π / 11) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l3705_370515


namespace NUMINAMATH_CALUDE_johns_annual_profit_l3705_370532

/-- John's apartment subletting profit calculation --/
theorem johns_annual_profit :
  ∀ (num_subletters : ℕ) 
    (subletter_payment : ℕ) 
    (rent_cost : ℕ) 
    (months_in_year : ℕ),
  num_subletters = 3 →
  subletter_payment = 400 →
  rent_cost = 900 →
  months_in_year = 12 →
  (num_subletters * subletter_payment - rent_cost) * months_in_year = 3600 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_annual_profit_l3705_370532


namespace NUMINAMATH_CALUDE_least_number_with_12_factors_l3705_370544

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Check if a number is the least positive integer with exactly 12 factors -/
def is_least_with_12_factors (n : ℕ+) : Prop :=
  (num_factors n = 12) ∧ ∀ m : ℕ+, m < n → num_factors m ≠ 12

theorem least_number_with_12_factors :
  is_least_with_12_factors 96 := by sorry

end NUMINAMATH_CALUDE_least_number_with_12_factors_l3705_370544


namespace NUMINAMATH_CALUDE_division_simplification_l3705_370567

theorem division_simplification (x y : ℝ) : -4 * x^5 * y^3 / (2 * x^3 * y) = -2 * x^2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l3705_370567


namespace NUMINAMATH_CALUDE_power_of_four_l3705_370541

theorem power_of_four (k : ℕ) (h : 4^k = 5) : 4^(2*k + 2) = 400 := by
  sorry

end NUMINAMATH_CALUDE_power_of_four_l3705_370541


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l3705_370542

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 3) % 18 = 0 ∧ (n + 3) % 70 = 0 ∧ (n + 3) % 100 = 0

theorem smallest_number_divisible : 
  (∀ m : ℕ, m < 6303 → ¬(is_divisible_by_all m)) ∧ 
  is_divisible_by_all 6303 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l3705_370542


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3705_370558

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (y^2 / a^2) - (x^2 / b^2) = 1 ∧ a > 0 ∧ b > 0

-- Define the focus
def focus (x y : ℝ) : Prop :=
  x = 0 ∧ y = -2

-- Define the asymptote slope
def asymptote_slope (slope : ℝ) : Prop :=
  slope = Real.sqrt 3

-- Theorem statement
theorem hyperbola_equation 
  (a b x y : ℝ) 
  (h1 : hyperbola a b x y) 
  (h2 : focus 0 (-2)) 
  (h3 : asymptote_slope (a/b)) : 
  (y^2 / 3) - x^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3705_370558


namespace NUMINAMATH_CALUDE_tough_week_sales_800_l3705_370560

/-- The amount Haji's mother sells on a good week -/
def good_week_sales : ℝ := sorry

/-- The amount Haji's mother sells on a tough week -/
def tough_week_sales : ℝ := sorry

/-- The total amount Haji's mother makes in 5 good weeks and 3 tough weeks -/
def total_sales : ℝ := 10400

/-- Tough week sales are half of good week sales -/
axiom tough_week_half_good : tough_week_sales = good_week_sales / 2

/-- Total sales equation -/
axiom total_sales_equation : 5 * good_week_sales + 3 * tough_week_sales = total_sales

theorem tough_week_sales_800 : tough_week_sales = 800 := by
  sorry

end NUMINAMATH_CALUDE_tough_week_sales_800_l3705_370560


namespace NUMINAMATH_CALUDE_unique_solution_l3705_370568

theorem unique_solution : ∃! x : ℝ, x^2 + 50 = (x - 10)^2 ∧ x = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3705_370568


namespace NUMINAMATH_CALUDE_jellybean_probability_l3705_370597

/-- The probability of drawing 3 blue jellybeans in succession without replacement from a bag containing 10 red and 10 blue jellybeans -/
theorem jellybean_probability : 
  let total_jellybeans : ℕ := 10 + 10
  let blue_jellybeans : ℕ := 10
  let draws : ℕ := 3
  (blue_jellybeans : ℚ) / total_jellybeans *
  ((blue_jellybeans - 1) : ℚ) / (total_jellybeans - 1) *
  ((blue_jellybeans - 2) : ℚ) / (total_jellybeans - 2) = 2 / 19 :=
by sorry

end NUMINAMATH_CALUDE_jellybean_probability_l3705_370597


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l3705_370588

theorem consecutive_integers_sum_of_squares : 
  ∀ x : ℕ, 
    x > 0 → 
    x * (x + 1) * (x + 2) = 12 * (x + (x + 1) + (x + 2)) → 
    x^2 + (x + 1)^2 + (x + 2)^2 = 77 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l3705_370588


namespace NUMINAMATH_CALUDE_range_of_b_l3705_370551

theorem range_of_b (a b c : ℝ) (sum_cond : a + b + c = 9) (prod_cond : a * b + b * c + c * a = 24) :
  1 ≤ b ∧ b ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_b_l3705_370551


namespace NUMINAMATH_CALUDE_kate_savings_ratio_l3705_370556

theorem kate_savings_ratio (pen_cost : ℕ) (kate_needs : ℕ) : 
  pen_cost = 30 → kate_needs = 20 → 
  (pen_cost - kate_needs) / pen_cost = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_kate_savings_ratio_l3705_370556


namespace NUMINAMATH_CALUDE_pool_perimeter_l3705_370579

/-- The perimeter of a rectangular pool in a garden with specific conditions -/
theorem pool_perimeter (garden_length : ℝ) (square_area : ℝ) (num_squares : ℕ) :
  garden_length = 10 →
  square_area = 20 →
  num_squares = 4 →
  ∃ (pool_length pool_width : ℝ),
    0 < pool_length ∧ 0 < pool_width ∧
    2 * (pool_length + pool_width) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_pool_perimeter_l3705_370579


namespace NUMINAMATH_CALUDE_circles_internally_tangent_l3705_370537

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 5 = 0

-- Define the center and radius of circle1
def center1 : ℝ × ℝ := (0, 0)
def radius1 : ℝ := 1

-- Define the center and radius of circle2
def center2 : ℝ × ℝ := (2, 0)
def radius2 : ℝ := 3

-- Define the distance between centers
def center_distance : ℝ := 2

-- Theorem stating that the circles are internally tangent
theorem circles_internally_tangent :
  center_distance = abs (radius2 - radius1) ∧
  center_distance < radius1 + radius2 := by sorry

end NUMINAMATH_CALUDE_circles_internally_tangent_l3705_370537


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3705_370574

theorem sufficient_not_necessary (a b : ℝ) : 
  (a^2 + b^2 ≤ 2 → -1 ≤ a*b ∧ a*b ≤ 1) ∧ 
  ∃ a b : ℝ, -1 ≤ a*b ∧ a*b ≤ 1 ∧ a^2 + b^2 > 2 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3705_370574


namespace NUMINAMATH_CALUDE_product_of_nonneg_quadratics_is_nonneg_l3705_370599

/-- Given two non-negative quadratic functions, their product is also non-negative. -/
theorem product_of_nonneg_quadratics_is_nonneg
  (a b c A B C : ℝ)
  (h1 : ∀ x : ℝ, a * x^2 + 2 * b * x + c ≥ 0)
  (h2 : ∀ x : ℝ, A * x^2 + 2 * B * x + C ≥ 0) :
  ∀ x : ℝ, a * A * x^2 + 2 * b * B * x + c * C ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_product_of_nonneg_quadratics_is_nonneg_l3705_370599


namespace NUMINAMATH_CALUDE_longest_pole_in_room_l3705_370508

theorem longest_pole_in_room (length width height : ℝ) 
  (h_length : length = 12)
  (h_width : width = 8)
  (h_height : height = 9) :
  Real.sqrt (length^2 + width^2 + height^2) = 17 := by
  sorry

end NUMINAMATH_CALUDE_longest_pole_in_room_l3705_370508


namespace NUMINAMATH_CALUDE_calculation_proof_l3705_370513

theorem calculation_proof :
  (1 : ℚ) * (5 / 7 : ℚ) * (-4 - 2/3 : ℚ) / (1 + 2/3 : ℚ) = -2 ∧
  (-2 - 1/7 : ℚ) / (-1.2 : ℚ) * (-1 - 2/5 : ℚ) = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3705_370513


namespace NUMINAMATH_CALUDE_quadratic_equation_root_and_q_l3705_370580

theorem quadratic_equation_root_and_q (p q : ℝ) : 
  (∃ x : ℂ, 5 * x^2 + p * x + q = 0 ∧ x = 3 + 2*I) →
  q = 65 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_and_q_l3705_370580


namespace NUMINAMATH_CALUDE_sweets_distribution_l3705_370590

theorem sweets_distribution (total_sweets : ℕ) (remaining_sweets : ℕ) (alt_children : ℕ) (alt_remaining : ℕ) :
  total_sweets = 358 →
  remaining_sweets = 8 →
  alt_children = 28 →
  alt_remaining = 22 →
  ∃ (children : ℕ), 
    children * ((total_sweets - remaining_sweets) / children) + remaining_sweets = total_sweets ∧
    alt_children * ((total_sweets - alt_remaining) / alt_children) + alt_remaining = total_sweets ∧
    children = 29 :=
by sorry

end NUMINAMATH_CALUDE_sweets_distribution_l3705_370590


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_l3705_370527

def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  let digits := Nat.digits base n
  digits = digits.reverse

theorem smallest_dual_palindrome : ∃ (n : ℕ),
  n > 10 ∧
  is_palindrome n 2 ∧
  is_palindrome n 8 ∧
  (∀ m : ℕ, m > 10 ∧ is_palindrome m 2 ∧ is_palindrome m 8 → n ≤ m) ∧
  n = 63 :=
sorry

end NUMINAMATH_CALUDE_smallest_dual_palindrome_l3705_370527


namespace NUMINAMATH_CALUDE_profit_percent_for_cost_selling_ratio_l3705_370539

theorem profit_percent_for_cost_selling_ratio (cost_price selling_price : ℝ) 
  (h : cost_price / selling_price = 4 / 5) : 
  (selling_price - cost_price) / cost_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_for_cost_selling_ratio_l3705_370539


namespace NUMINAMATH_CALUDE_find_other_number_l3705_370535

theorem find_other_number (A B : ℕ) (h1 : A = 24) (h2 : Nat.gcd A B = 15) (h3 : Nat.lcm A B = 312) : B = 195 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l3705_370535


namespace NUMINAMATH_CALUDE_parabolas_symmetric_about_y_axis_l3705_370540

-- Define the parabolas
def parabola1 (x : ℝ) : ℝ := 2 * x^2
def parabola2 (x : ℝ) : ℝ := -2 * x^2
def parabola3 (x : ℝ) : ℝ := x^2

-- Define symmetry about y-axis
def symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Theorem statement
theorem parabolas_symmetric_about_y_axis :
  symmetric_about_y_axis parabola1 ∧
  symmetric_about_y_axis parabola2 ∧
  symmetric_about_y_axis parabola3 :=
sorry

end NUMINAMATH_CALUDE_parabolas_symmetric_about_y_axis_l3705_370540


namespace NUMINAMATH_CALUDE_basketball_shots_l3705_370529

theorem basketball_shots (t h f : ℕ) : 
  (2 * t = 3 * h) →  -- Two-point shots scored double the points of three-point shots
  (f = h - 4) →      -- Number of free throws is four fewer than three-point shots
  (t + h + f = 40) → -- Total shots is 40
  (2 * t + 3 * h + f = 76) → -- Total score is 76
  h = 8 := by sorry

end NUMINAMATH_CALUDE_basketball_shots_l3705_370529


namespace NUMINAMATH_CALUDE_roots_log_sum_l3705_370503

-- Define the equation
def equation (x : ℝ) : Prop := (Real.log x)^2 - Real.log (x^2) = 2

-- Define α and β as the roots of the equation
axiom α : ℝ
axiom β : ℝ
axiom α_pos : α > 0
axiom β_pos : β > 0
axiom α_root : equation α
axiom β_root : equation β

-- State the theorem
theorem roots_log_sum : Real.log β / Real.log α + Real.log α / Real.log β = -4 := by
  sorry

end NUMINAMATH_CALUDE_roots_log_sum_l3705_370503


namespace NUMINAMATH_CALUDE_henrys_cd_collection_l3705_370502

theorem henrys_cd_collection :
  ∀ (country rock classical : ℕ),
    country = 23 →
    country = rock + 3 →
    rock = 2 * classical →
    classical = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_henrys_cd_collection_l3705_370502


namespace NUMINAMATH_CALUDE_coconut_grove_problem_l3705_370596

theorem coconut_grove_problem (x : ℕ) : 
  (3 * 60 + 2 * 120 + x * 180 = 100 * (3 + 2 + x)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_coconut_grove_problem_l3705_370596


namespace NUMINAMATH_CALUDE_households_using_both_is_15_l3705_370530

/-- Given information about soap brand usage in surveyed households -/
structure SoapSurvey where
  total : ℕ
  neither : ℕ
  only_E : ℕ
  both_to_only_B_ratio : ℕ
  h_total : total = 200
  h_neither : neither = 80
  h_only_E : only_E = 60
  h_ratio : both_to_only_B_ratio = 3

/-- The number of households using both brand E and brand B soap -/
def households_using_both (s : SoapSurvey) : ℕ := 15

/-- Theorem stating that the number of households using both brands is 15 -/
theorem households_using_both_is_15 (s : SoapSurvey) : 
  households_using_both s = 15 := by sorry

end NUMINAMATH_CALUDE_households_using_both_is_15_l3705_370530


namespace NUMINAMATH_CALUDE_shaded_area_in_square_configuration_l3705_370520

/-- The area of the shaded region in a square configuration -/
theorem shaded_area_in_square_configuration 
  (total_area : ℝ) 
  (overlap_area : ℝ) 
  (area_ratio : ℝ) 
  (h1 : total_area = 196) 
  (h2 : overlap_area = 1) 
  (h3 : area_ratio = 4) : 
  ∃ (shaded_area : ℝ), shaded_area = 72 ∧ 
  ∃ (small_square_area large_square_area : ℝ),
    large_square_area = area_ratio * small_square_area ∧
    shaded_area = large_square_area + small_square_area - overlap_area ∧
    large_square_area + small_square_area - overlap_area < total_area := by
  sorry


end NUMINAMATH_CALUDE_shaded_area_in_square_configuration_l3705_370520


namespace NUMINAMATH_CALUDE_parabola_intercepts_l3705_370553

-- Define the parabola equation
def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

-- Theorem statement
theorem parabola_intercepts :
  -- There is exactly one x-intercept at x = 3
  (∃! x : ℝ, x = 3 ∧ ∃ y : ℝ, parabola y = x) ∧
  -- There are exactly two y-intercepts
  (∃! y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ 
    parabola y₁ = 0 ∧ parabola y₂ = 0 ∧
    y₁ = (1 + Real.sqrt 10) / 3 ∧
    y₂ = (1 - Real.sqrt 10) / 3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intercepts_l3705_370553


namespace NUMINAMATH_CALUDE_sqrt_180_simplification_l3705_370585

theorem sqrt_180_simplification : Real.sqrt 180 = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_180_simplification_l3705_370585


namespace NUMINAMATH_CALUDE_unique_m_value_l3705_370518

def f (x a m : ℝ) : ℝ := |x - a| + m * |x + a|

theorem unique_m_value (a m : ℝ) 
  (h1 : 0 < m) (h2 : m < 1)
  (h3 : ∀ x : ℝ, f x a m ≥ 2)
  (h4 : a ≤ -5 ∨ a ≥ 5) :
  m = 1/5 := by
sorry

end NUMINAMATH_CALUDE_unique_m_value_l3705_370518


namespace NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l3705_370565

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 15th term of the sequence is 15 -/
def Term15Is15 (a : ℕ → ℝ) : Prop := a 15 = 15

/-- The 16th term of the sequence is 21 -/
def Term16Is21 (a : ℕ → ℝ) : Prop := a 16 = 21

/-- The 3rd term of the sequence is -57 -/
def Term3IsNeg57 (a : ℕ → ℝ) : Prop := a 3 = -57

theorem arithmetic_sequence_theorem (a : ℕ → ℝ) :
  ArithmeticSequence a → Term15Is15 a → Term16Is21 a → Term3IsNeg57 a := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l3705_370565


namespace NUMINAMATH_CALUDE_tens_digit_of_2013_pow_2018_minus_2019_l3705_370554

theorem tens_digit_of_2013_pow_2018_minus_2019 :
  ∃ n : ℕ, 2013^2018 - 2019 = 100 * n + 50 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_2013_pow_2018_minus_2019_l3705_370554


namespace NUMINAMATH_CALUDE_value_of_p_l3705_370523

variables (A B C p q r s : ℝ) 

/-- The roots of the first quadratic equation -/
def roots_eq1 : Prop := A * r^2 + B * r + C = 0 ∧ A * s^2 + B * s + C = 0

/-- The roots of the second quadratic equation -/
def roots_eq2 : Prop := r^2 + p * r + q = 0 ∧ s^2 + p * s + q = 0

/-- The theorem stating the value of p -/
theorem value_of_p (hA : A ≠ 0) (h1 : roots_eq1 A B C r s) (h2 : roots_eq2 p q r s) : 
  p = (2 * A * C - B^2) / A^2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_p_l3705_370523


namespace NUMINAMATH_CALUDE_inequality_range_l3705_370587

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 6| - |x - 4| ≤ a^2 - 3*a) ↔ 
  (a ≤ -2 ∨ a ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l3705_370587


namespace NUMINAMATH_CALUDE_coefficient_of_x_5_l3705_370586

-- Define the polynomials
def p (x : ℝ) : ℝ := x^6 - 4*x^5 + 6*x^4 - 5*x^3 + 3*x^2 - 2*x + 1
def q (x : ℝ) : ℝ := 3*x^4 - 2*x^3 + x^2 + 4*x + 5

-- Define the product of the polynomials
def product (x : ℝ) : ℝ := p x * q x

-- Theorem to prove
theorem coefficient_of_x_5 : 
  ∃ c, ∀ x, product x = c * x^5 + (fun x => x^6 * (-23) + (fun x => x^4 * 0 + x^3 * 0 + x^2 * 0 + x * 0 + 0) x) x :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_5_l3705_370586


namespace NUMINAMATH_CALUDE_simplify_expressions_l3705_370570

theorem simplify_expressions :
  let exp1 := ((0.064 ^ (1/5)) ^ (-2.5)) ^ (2/3) - (3 * (3/8)) ^ (1/3) - π ^ 0
  let exp2 := (2 * Real.log 2 + Real.log 3) / (1 + (1/2) * Real.log 0.36 + (1/4) * Real.log 16)
  (exp1 = 0) ∧ (exp2 = (2 * Real.log 2 + Real.log 3) / Real.log 24) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l3705_370570


namespace NUMINAMATH_CALUDE_find_x_value_l3705_370507

theorem find_x_value (A B : Set ℝ) (x : ℝ) : 
  A = {-1, 0} →
  B = {0, 1, x + 2} →
  A ⊆ B →
  x = -3 := by
sorry

end NUMINAMATH_CALUDE_find_x_value_l3705_370507


namespace NUMINAMATH_CALUDE_sequence_property_l3705_370593

theorem sequence_property (m : ℤ) (a : ℕ → ℤ) (r s : ℕ) :
  (∀ n : ℕ, a (n + 2) = a (n + 1) - m * a n) →
  (|m| ≥ 2) →
  (a 1 ≠ 0 ∨ a 2 ≠ 0) →
  (r > s) →
  (s ≥ 2) →
  (a r = a 1) →
  (a s = a 1) →
  (r - s : ℤ) ≥ |m| :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l3705_370593


namespace NUMINAMATH_CALUDE_items_not_washed_l3705_370576

theorem items_not_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (pants : ℕ) (jackets : ℕ) (washed : ℕ)
  (h1 : short_sleeve = 9)
  (h2 : long_sleeve = 21)
  (h3 : pants = 15)
  (h4 : jackets = 8)
  (h5 : washed = 43) :
  short_sleeve + long_sleeve + pants + jackets - washed = 10 := by
  sorry

end NUMINAMATH_CALUDE_items_not_washed_l3705_370576


namespace NUMINAMATH_CALUDE_james_balloons_count_l3705_370538

/-- The number of balloons Amy has -/
def amy_balloons : ℕ := 513

/-- The number of additional balloons James has compared to Amy -/
def james_extra_balloons : ℕ := 709

/-- The total number of balloons James has -/
def james_total_balloons : ℕ := amy_balloons + james_extra_balloons

theorem james_balloons_count : james_total_balloons = 1222 := by
  sorry

end NUMINAMATH_CALUDE_james_balloons_count_l3705_370538


namespace NUMINAMATH_CALUDE_gillians_total_spending_l3705_370549

def sandis_initial_amount : ℕ := 600
def gillians_additional_spending : ℕ := 150

def sandis_market_spending (initial_amount : ℕ) : ℕ :=
  initial_amount / 2

def gillians_market_spending (sandis_spending : ℕ) : ℕ :=
  3 * sandis_spending + gillians_additional_spending

theorem gillians_total_spending :
  gillians_market_spending (sandis_market_spending sandis_initial_amount) = 1050 := by
  sorry

end NUMINAMATH_CALUDE_gillians_total_spending_l3705_370549


namespace NUMINAMATH_CALUDE_discount_sum_is_22_percent_l3705_370521

/-- The discount rate for Pony jeans -/
def pony_discount : ℝ := 10.999999999999996

/-- The regular price of Fox jeans -/
def fox_price : ℝ := 15

/-- The regular price of Pony jeans -/
def pony_price : ℝ := 18

/-- The number of Fox jeans purchased -/
def fox_quantity : ℕ := 3

/-- The number of Pony jeans purchased -/
def pony_quantity : ℕ := 2

/-- The total savings from the discounts -/
def total_savings : ℝ := 8.91

/-- Theorem stating that the sum of discount rates for Fox and Pony jeans is 22% -/
theorem discount_sum_is_22_percent :
  ∃ (fox_discount : ℝ),
    fox_discount ≥ 0 ∧
    fox_discount ≤ 100 ∧
    fox_discount + pony_discount = 22 ∧
    (fox_price * fox_quantity * fox_discount / 100 + 
     pony_price * pony_quantity * pony_discount / 100 = total_savings) :=
by sorry

end NUMINAMATH_CALUDE_discount_sum_is_22_percent_l3705_370521


namespace NUMINAMATH_CALUDE_rope_length_proof_l3705_370578

/-- The length of the rope in meters -/
def rope_length : ℝ := 35

/-- The number of steps Xiaoming takes when walking in the same direction as the tractor -/
def steps_same_direction : ℕ := 140

/-- The number of steps Xiaoming takes when walking in the opposite direction of the tractor -/
def steps_opposite_direction : ℕ := 20

/-- The length of each of Xiaoming's steps in meters -/
def step_length : ℝ := 1

theorem rope_length_proof :
  ∃ (tractor_speed : ℝ),
    tractor_speed > 0 ∧
    rope_length + tractor_speed * steps_same_direction * step_length = steps_same_direction * step_length ∧
    rope_length - tractor_speed * steps_opposite_direction * step_length = steps_opposite_direction * step_length :=
by
  sorry

#check rope_length_proof

end NUMINAMATH_CALUDE_rope_length_proof_l3705_370578


namespace NUMINAMATH_CALUDE_floor_equation_solution_l3705_370517

theorem floor_equation_solution (n : ℤ) :
  (⌊n^2 / 3⌋ : ℤ) - (⌊n / 2⌋ : ℤ)^2 = 3 ↔ n = 6 :=
by sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l3705_370517


namespace NUMINAMATH_CALUDE_unique_number_property_l3705_370557

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l3705_370557


namespace NUMINAMATH_CALUDE_tims_change_theorem_l3705_370504

/-- Calculates the change received after a purchase --/
def calculate_change (initial_amount : ℕ) (purchase_amount : ℕ) : ℕ :=
  initial_amount - purchase_amount

/-- Proves that the change received is correct for Tim's candy bar purchase --/
theorem tims_change_theorem :
  let initial_amount : ℕ := 50
  let purchase_amount : ℕ := 45
  calculate_change initial_amount purchase_amount = 5 := by
  sorry

end NUMINAMATH_CALUDE_tims_change_theorem_l3705_370504


namespace NUMINAMATH_CALUDE_function_is_constant_l3705_370510

/-- Given a function f: ℝ → ℝ satisfying certain conditions, prove that f is constant. -/
theorem function_is_constant (f : ℝ → ℝ) (a : ℝ) (ha : a > 0)
  (h1 : ∀ x, 0 < f x ∧ f x ≤ a)
  (h2 : ∀ x y, Real.sqrt (f x * f y) ≥ f ((x + y) / 2)) :
  ∃ c, ∀ x, f x = c :=
sorry

end NUMINAMATH_CALUDE_function_is_constant_l3705_370510


namespace NUMINAMATH_CALUDE_cuboid_non_parallel_edges_l3705_370571

/-- Represents a cuboid with integer side lengths -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of edges not parallel to a given edge in a cuboid -/
def nonParallelEdges (c : Cuboid) : ℕ := sorry

/-- Theorem stating that a cuboid with side lengths 8, 6, and 4 has 8 edges not parallel to any given edge -/
theorem cuboid_non_parallel_edges :
  let c : Cuboid := { length := 8, width := 6, height := 4 }
  nonParallelEdges c = 8 := by sorry

end NUMINAMATH_CALUDE_cuboid_non_parallel_edges_l3705_370571


namespace NUMINAMATH_CALUDE_K_bounds_K_bounds_tight_l3705_370595

noncomputable def K (x y z : ℝ) : ℝ := 5 * x - 6 * y + 7 * z

theorem K_bounds :
  ∀ x y z : ℝ,
  x ≥ 0 → y ≥ 0 → z ≥ 0 →
  4 * x + y + 2 * z = 4 →
  3 * x + 6 * y - 2 * z = 6 →
  -5 ≤ K x y z ∧ K x y z ≤ 7 :=
by
  sorry

theorem K_bounds_tight :
  (∃ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
   4 * x + y + 2 * z = 4 ∧
   3 * x + 6 * y - 2 * z = 6 ∧
   K x y z = -5) ∧
  (∃ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
   4 * x + y + 2 * z = 4 ∧
   3 * x + 6 * y - 2 * z = 6 ∧
   K x y z = 7) :=
by
  sorry

end NUMINAMATH_CALUDE_K_bounds_K_bounds_tight_l3705_370595


namespace NUMINAMATH_CALUDE_percentage_of_sum_l3705_370505

theorem percentage_of_sum (x y : ℝ) (P : ℝ) : 
  (0.5 * (x - y) = (P / 100) * (x + y)) → 
  (y = 0.42857142857142854 * x) → 
  (P = 20) := by
sorry

end NUMINAMATH_CALUDE_percentage_of_sum_l3705_370505


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l3705_370591

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 9240 →
  Nat.gcd a b = 33 →
  a = 231 →
  b = 1320 := by
sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l3705_370591


namespace NUMINAMATH_CALUDE_rationalize_denominator_sqrt_5_l3705_370582

theorem rationalize_denominator_sqrt_5 :
  let x := 2 + Real.sqrt 5
  let y := 1 - Real.sqrt 5
  x / y = -7/4 - (3 * Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_sqrt_5_l3705_370582


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_9_l3705_370525

def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

def sum_arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1 : ℤ) * d) / 2

theorem arithmetic_sequence_sum_9 (a₁ d : ℤ) :
  a₁ = 2 →
  arithmetic_sequence a₁ d 5 = 3 * arithmetic_sequence a₁ d 3 →
  sum_arithmetic_sequence a₁ d 9 = -54 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_9_l3705_370525


namespace NUMINAMATH_CALUDE_cubic_root_formula_and_verification_l3705_370566

theorem cubic_root_formula_and_verification :
  let x₀ := Real.rpow (3 + (11/9) * Real.sqrt 6) (1/3) + Real.rpow (3 - (11/9) * Real.sqrt 6) (1/3)
  x₀ = 2 ∧ x₀^3 - x₀ - 6 = 0 := by sorry

end NUMINAMATH_CALUDE_cubic_root_formula_and_verification_l3705_370566


namespace NUMINAMATH_CALUDE_speed_in_still_water_l3705_370563

/-- The speed of a man rowing a boat in still water, given his downstream speed and the speed of the current. -/
theorem speed_in_still_water 
  (downstream_speed : ℝ) 
  (current_speed : ℝ) 
  (h1 : downstream_speed = 17.9997120913593) 
  (h2 : current_speed = 3) : 
  downstream_speed - current_speed = 14.9997120913593 := by
  sorry

#eval (17.9997120913593 : Float) - 3

end NUMINAMATH_CALUDE_speed_in_still_water_l3705_370563


namespace NUMINAMATH_CALUDE_P_is_ellipse_l3705_370509

-- Define the set of points P(x,y) satisfying the given equation
def P : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; Real.sqrt ((x + 4)^2 + y^2) + Real.sqrt ((x - 4)^2 + y^2) = 10}

-- Define an ellipse with foci at (-4, 0) and (4, 0), and sum of distances equal to 10
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; Real.sqrt ((x + 4)^2 + y^2) + Real.sqrt ((x - 4)^2 + y^2) = 10}

-- Theorem stating that the set P is equivalent to the Ellipse
theorem P_is_ellipse : P = Ellipse := by sorry

end NUMINAMATH_CALUDE_P_is_ellipse_l3705_370509


namespace NUMINAMATH_CALUDE_dental_removal_fraction_l3705_370581

theorem dental_removal_fraction :
  ∀ (x : ℚ),
  (∃ (t₁ t₂ t₃ t₄ : ℕ),
    t₁ + t₂ + t₃ + t₄ = 4 ∧  -- Four adults
    (∀ i, t₁ ≤ i ∧ i ≤ t₄ → 32 = 32) ∧  -- Each adult has 32 teeth
    x * 32 + 3/8 * 32 + 1/2 * 32 + 4 = 40)  -- Total teeth removed
  → x = 1/4 := by
sorry

end NUMINAMATH_CALUDE_dental_removal_fraction_l3705_370581


namespace NUMINAMATH_CALUDE_sum_of_angles_in_quadrilateral_figure_l3705_370531

/-- A geometric figure with six angles that form a quadrilateral -/
structure QuadrilateralFigure where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  F : ℝ
  G : ℝ

/-- The sum of angles in a quadrilateral is 360° -/
theorem sum_of_angles_in_quadrilateral_figure (q : QuadrilateralFigure) :
  q.A + q.B + q.C + q.D + q.F + q.G = 360 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_in_quadrilateral_figure_l3705_370531


namespace NUMINAMATH_CALUDE_sequence_inequality_l3705_370573

theorem sequence_inequality (a : ℕ → ℝ) 
  (h : ∀ (k m : ℕ), k > 0 → m > 0 → |a (k + m) - a k - a m| ≤ 1) :
  ∀ (p q : ℕ), p > 0 → q > 0 → |a p / p - a q / q| < 1 / p + 1 / q :=
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3705_370573


namespace NUMINAMATH_CALUDE_painting_class_selection_l3705_370575

theorem painting_class_selection (n k : ℕ) (hn : n = 10) (hk : k = 4) :
  Nat.choose n k = 210 := by
  sorry

end NUMINAMATH_CALUDE_painting_class_selection_l3705_370575


namespace NUMINAMATH_CALUDE_inequality_range_l3705_370500

theorem inequality_range (k : ℝ) : 
  (∀ x : ℝ, 3^(2*x) - (k+1)*3^x + 2 > 0) → k < 2*Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l3705_370500


namespace NUMINAMATH_CALUDE_pen_to_book_ratio_l3705_370577

theorem pen_to_book_ratio (pencils pens books : ℕ) 
  (h1 : pencils = 140)
  (h2 : books = 30)
  (h3 : pencils * 4 = pens * 14)
  (h4 : pencils * 3 = books * 14) : 
  4 * books = 3 * pens := by
  sorry

end NUMINAMATH_CALUDE_pen_to_book_ratio_l3705_370577


namespace NUMINAMATH_CALUDE_increase_by_percentage_l3705_370514

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 50 → percentage = 120 → result = initial * (1 + percentage / 100) → result = 110 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l3705_370514


namespace NUMINAMATH_CALUDE_rotated_line_equation_l3705_370528

/-- The y-coordinate of point P on the original line -/
def y : ℝ := 4

/-- The slope of the original line -/
def m₁ : ℝ := 1

/-- The slope of line l -/
def m₂ : ℝ := -1

/-- Point P on the original line -/
def P : ℝ × ℝ := (3, y)

/-- The equation of the original line -/
def original_line (x y : ℝ) : Prop := x - y + 1 = 0

/-- The equation of line l after rotation -/
def line_l (x y : ℝ) : Prop := x + y - 7 = 0

theorem rotated_line_equation : 
  ∀ x y : ℝ, original_line P.1 P.2 → (m₁ * m₂ = -1) → line_l x y :=
sorry

end NUMINAMATH_CALUDE_rotated_line_equation_l3705_370528


namespace NUMINAMATH_CALUDE_sum_squared_odd_l3705_370543

theorem sum_squared_odd (a b c : ℤ) (h : (a + b + c) % 2 = 1) : 
  (a^2 + b^2 - c^2 + 2*a*b) % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_squared_odd_l3705_370543


namespace NUMINAMATH_CALUDE_cake_and_muffin_buyers_l3705_370511

theorem cake_and_muffin_buyers (total : ℕ) (cake : ℕ) (muffin : ℕ) (neither_prob : ℚ) 
  (h_total : total = 100)
  (h_cake : cake = 50)
  (h_muffin : muffin = 40)
  (h_neither_prob : neither_prob = 1/4) :
  ∃ both : ℕ, 
    both = cake + muffin - (total * (1 - neither_prob)) ∧
    both = 15 := by
  sorry

end NUMINAMATH_CALUDE_cake_and_muffin_buyers_l3705_370511


namespace NUMINAMATH_CALUDE_train_speed_l3705_370545

/-- Proves that a train with given length and time to cross a pole has a specific speed -/
theorem train_speed (train_length : Real) (crossing_time : Real) (speed : Real) : 
  train_length = 200 → 
  crossing_time = 12 → 
  speed = (train_length / 1000) / (crossing_time / 3600) → 
  speed = 60 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l3705_370545


namespace NUMINAMATH_CALUDE_original_tree_count_l3705_370501

/-- The number of leaves each tree drops during fall. -/
def leaves_per_tree : ℕ := 100

/-- The total number of fallen leaves. -/
def total_fallen_leaves : ℕ := 1400

/-- The current number of trees is twice the original plan. -/
def current_trees_twice_original (original : ℕ) : Prop :=
  2 * original = total_fallen_leaves / leaves_per_tree

/-- Theorem stating the original number of trees the town council intended to plant. -/
theorem original_tree_count : ∃ (original : ℕ), 
  current_trees_twice_original original ∧ original = 7 :=
by
  sorry


end NUMINAMATH_CALUDE_original_tree_count_l3705_370501


namespace NUMINAMATH_CALUDE_absolute_value_equals_sqrt_l3705_370536

theorem absolute_value_equals_sqrt (x : ℝ) : 2 * |x| = Real.sqrt (4 * x^2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_sqrt_l3705_370536


namespace NUMINAMATH_CALUDE_hamburger_combinations_l3705_370572

/-- The number of condiments available -/
def num_condiments : ℕ := 9

/-- The number of bun choices available -/
def num_bun_choices : ℕ := 2

/-- The number of meat patty choices available -/
def num_patty_choices : ℕ := 3

/-- The total number of different hamburger combinations -/
def total_hamburgers : ℕ := 2^num_condiments * num_bun_choices * num_patty_choices

theorem hamburger_combinations :
  total_hamburgers = 3072 :=
sorry

end NUMINAMATH_CALUDE_hamburger_combinations_l3705_370572


namespace NUMINAMATH_CALUDE_distance_difference_l3705_370562

def sprint_distance : ℝ := 0.88
def jog_distance : ℝ := 0.75

theorem distance_difference : sprint_distance - jog_distance = 0.13 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l3705_370562


namespace NUMINAMATH_CALUDE_sqrt_two_value_l3705_370564

def f_property (f : ℝ → ℝ) : Prop :=
  ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → f (x₁ * x₂) = f x₁ + f x₂

theorem sqrt_two_value (f : ℝ → ℝ) (h1 : f_property f) (h2 : f 8 = 3) :
  f (Real.sqrt 2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_value_l3705_370564


namespace NUMINAMATH_CALUDE_intersection_problem_l3705_370546

theorem intersection_problem (a : ℝ) : 
  let A : Set ℝ := {-1, 1, 3}
  let B : Set ℝ := {a - 1, a^2 + 3}
  A ∩ B = {3} → a = 4 ∨ a = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_problem_l3705_370546


namespace NUMINAMATH_CALUDE_quadratic_relationship_l3705_370552

/-- A quadratic function f(x) = 3x^2 + ax + b where f(x - 1) is an even function -/
def f (a b : ℝ) (x : ℝ) : ℝ := 3 * x^2 + a * x + b

theorem quadratic_relationship (a b : ℝ) 
  (h : ∀ x, f a b (x - 1) = f a b (1 - x)) : 
  f a b (-1) < f a b (-3/2) ∧ f a b (-3/2) = f a b (3/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_relationship_l3705_370552


namespace NUMINAMATH_CALUDE_number_of_factors_of_b_power_n_l3705_370512

def b : ℕ := 6
def n : ℕ := 15

theorem number_of_factors_of_b_power_n : 
  b ≤ 15 → n ≤ 15 → (Nat.factors (b^n)).length + 1 = 256 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_of_b_power_n_l3705_370512


namespace NUMINAMATH_CALUDE_tuna_sales_difference_l3705_370519

/-- Calculates the difference in daily sales between peak and low seasons for tuna fish. -/
theorem tuna_sales_difference (peak_rate : ℕ) (low_rate : ℕ) (price : ℕ) (hours : ℕ) 
  (h1 : peak_rate = 6)
  (h2 : low_rate = 4)
  (h3 : price = 60)
  (h4 : hours = 15) :
  peak_rate * price * hours - low_rate * price * hours = 1800 := by
  sorry

#check tuna_sales_difference

end NUMINAMATH_CALUDE_tuna_sales_difference_l3705_370519


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l3705_370598

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 = v 1 * w 0

theorem parallel_vectors_sum (m : ℝ) :
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![m, 2]
  are_parallel a b →
  (3 • a + 2 • b) = ![14, 7] := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l3705_370598


namespace NUMINAMATH_CALUDE_relationship_between_exponents_l3705_370516

theorem relationship_between_exponents 
  (a c e f : ℝ) 
  (x y z w : ℝ) 
  (h1 : a^(2*x) = c^(3*y)) 
  (h2 : a^(2*x) = e) 
  (h3 : c^(3*y) = e) 
  (h4 : c^(4*z) = a^(3*w)) 
  (h5 : c^(4*z) = f) 
  (h6 : a^(3*w) = f) 
  (h7 : a ≠ 0) 
  (h8 : c ≠ 0) 
  (h9 : e > 0) 
  (h10 : f > 0) : 
  2*w*z = x*y := by
sorry

end NUMINAMATH_CALUDE_relationship_between_exponents_l3705_370516


namespace NUMINAMATH_CALUDE_fraction_quadrupled_l3705_370526

theorem fraction_quadrupled (a b : ℚ) (h : a ≠ 0) :
  (2 * b) / (a / 2) = 4 * (b / a) := by sorry

end NUMINAMATH_CALUDE_fraction_quadrupled_l3705_370526


namespace NUMINAMATH_CALUDE_birthday_cake_theorem_l3705_370583

/-- Represents a rectangular cake with dimensions length, width, and height -/
structure Cake where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of unit cubes with exactly two iced sides in a cake -/
def count_two_sided_iced_pieces (c : Cake) : ℕ :=
  sorry

/-- The main theorem stating that a 5 × 3 × 4 cake with five faces iced
    has 25 pieces with exactly two iced sides -/
theorem birthday_cake_theorem :
  let cake : Cake := { length := 5, width := 3, height := 4 }
  count_two_sided_iced_pieces cake = 25 := by
  sorry

end NUMINAMATH_CALUDE_birthday_cake_theorem_l3705_370583


namespace NUMINAMATH_CALUDE_infinite_occurrence_in_digit_sum_sequence_l3705_370534

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The sum of digits of an integer -/
def sumOfDigits (n : ℤ) : ℕ :=
  (n.natAbs.digits 10).sum

/-- The sequence of sums of digits of polynomial values -/
def digitSumSequence (w : IntPolynomial) : ℕ → ℕ := fun n ↦ sumOfDigits (w.eval n)

/-- There exists a value that occurs infinitely often in the digit sum sequence -/
theorem infinite_occurrence_in_digit_sum_sequence (w : IntPolynomial) :
  ∃ k : ℕ, Set.Infinite {n : ℕ | digitSumSequence w n = k} :=
sorry

end NUMINAMATH_CALUDE_infinite_occurrence_in_digit_sum_sequence_l3705_370534


namespace NUMINAMATH_CALUDE_trajectory_and_line_m_l3705_370533

/-- The distance ratio condition for point P -/
def distance_ratio (x y : ℝ) : Prop :=
  (((x - 3 * Real.sqrt 3)^2 + y^2).sqrt) / (|x - 4 * Real.sqrt 3|) = Real.sqrt 3 / 2

/-- The equation of the ellipse -/
def on_ellipse (x y : ℝ) : Prop :=
  x^2 / 36 + y^2 / 9 = 1

/-- The equation of line m -/
def on_line_m (x y : ℝ) : Prop :=
  x + 2 * y - 8 = 0

/-- The midpoint condition -/
def is_midpoint (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ + x₂) / 2 = 4 ∧ (y₁ + y₂) / 2 = 2

theorem trajectory_and_line_m :
  (∀ x y : ℝ, distance_ratio x y ↔ on_ellipse x y) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    on_ellipse x₁ y₁ ∧ on_ellipse x₂ y₂ ∧ is_midpoint x₁ y₁ x₂ y₂ →
    on_line_m x₁ y₁ ∧ on_line_m x₂ y₂) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_line_m_l3705_370533
