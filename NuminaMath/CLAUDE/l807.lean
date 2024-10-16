import Mathlib

namespace NUMINAMATH_CALUDE_hypotenuse_value_l807_80773

-- Define a right triangle with sides 3, 5, and x (hypotenuse)
def right_triangle (x : ℝ) : Prop :=
  x > 0 ∧ x^2 = 3^2 + 5^2

-- Theorem statement
theorem hypotenuse_value :
  ∃ x : ℝ, right_triangle x ∧ x = Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_hypotenuse_value_l807_80773


namespace NUMINAMATH_CALUDE_probability_is_31_145_l807_80731

-- Define the shoe collection
def total_pairs : ℕ := 15
def black_pairs : ℕ := 8
def brown_pairs : ℕ := 5
def red_pairs : ℕ := 2

-- Define the probability function
def probability_same_color_different_foot : ℚ :=
  -- Black shoes probability
  (2 * black_pairs : ℚ) / (2 * total_pairs) * (black_pairs : ℚ) / (2 * total_pairs - 1) +
  -- Brown shoes probability
  (2 * brown_pairs : ℚ) / (2 * total_pairs) * (brown_pairs : ℚ) / (2 * total_pairs - 1) +
  -- Red shoes probability
  (2 * red_pairs : ℚ) / (2 * total_pairs) * (red_pairs : ℚ) / (2 * total_pairs - 1)

-- Theorem statement
theorem probability_is_31_145 : probability_same_color_different_foot = 31 / 145 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_31_145_l807_80731


namespace NUMINAMATH_CALUDE_final_price_after_discounts_l807_80739

def original_price : ℝ := 200
def weekend_discount : ℝ := 0.4
def wednesday_discount : ℝ := 0.2

theorem final_price_after_discounts :
  (original_price * (1 - weekend_discount)) * (1 - wednesday_discount) = 96 := by
  sorry

end NUMINAMATH_CALUDE_final_price_after_discounts_l807_80739


namespace NUMINAMATH_CALUDE_cos_difference_formula_l807_80702

theorem cos_difference_formula (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3/2)
  (h2 : Real.cos A + Real.cos B = 1) :
  Real.cos (A - B) = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_formula_l807_80702


namespace NUMINAMATH_CALUDE_base3_to_base10_conversion_l807_80756

/-- Converts a list of digits in base 3 to a natural number in base 10 -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (3 ^ i)) 0

/-- The base-3 representation of the number -/
def base3Number : List Nat := [1, 2, 2, 0, 2]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Number = 187 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_conversion_l807_80756


namespace NUMINAMATH_CALUDE_evaluate_expression_l807_80751

theorem evaluate_expression : 3 + 3 * (3 ^ (3 ^ 3)) - 3 ^ 3 = 22876792454937 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l807_80751


namespace NUMINAMATH_CALUDE_sum_of_products_l807_80793

theorem sum_of_products (x : Fin 150 → ℝ) : 
  (∀ i, x i = Real.sqrt 2 + 1 ∨ x i = Real.sqrt 2 - 1) →
  (∃ x : Fin 150 → ℝ, (∀ i, x i = Real.sqrt 2 + 1 ∨ x i = Real.sqrt 2 - 1) ∧ 
    (Finset.sum (Finset.range 75) (λ i => x (2*i) * x (2*i+1)) = 111)) ∧
  (¬ ∃ x : Fin 150 → ℝ, (∀ i, x i = Real.sqrt 2 + 1 ∨ x i = Real.sqrt 2 - 1) ∧ 
    (Finset.sum (Finset.range 75) (λ i => x (2*i) * x (2*i+1)) = 121)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_products_l807_80793


namespace NUMINAMATH_CALUDE_bookshop_online_sales_l807_80781

theorem bookshop_online_sales (initial_books : ℕ) (saturday_instore : ℕ) (sunday_instore : ℕ)
  (sunday_online_increase : ℕ) (shipment : ℕ) (final_books : ℕ) :
  initial_books = 743 →
  saturday_instore = 37 →
  sunday_instore = 2 * saturday_instore →
  sunday_online_increase = 34 →
  shipment = 160 →
  final_books = 502 →
  ∃ (saturday_online : ℕ),
    final_books = initial_books - saturday_instore - saturday_online -
      sunday_instore - (saturday_online + sunday_online_increase) + shipment ∧
    saturday_online = 128 := by
  sorry

end NUMINAMATH_CALUDE_bookshop_online_sales_l807_80781


namespace NUMINAMATH_CALUDE_salt_solution_dilution_l807_80747

/-- Given a salt solution, prove that adding a specific amount of water yields the target concentration -/
theorem salt_solution_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (target_concentration : ℝ) (water_added : ℝ) : 
  initial_volume = 40 →
  initial_concentration = 0.25 →
  target_concentration = 0.15 →
  water_added = 400 / 15 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = target_concentration := by
  sorry

#eval (400 : ℚ) / 15

end NUMINAMATH_CALUDE_salt_solution_dilution_l807_80747


namespace NUMINAMATH_CALUDE_prove_smallest_positive_angle_l807_80744

def smallest_positive_angle_theorem : Prop :=
  ∃ θ : Real,
    θ > 0 ∧
    θ < 2 * Real.pi ∧
    Real.cos θ = Real.sin (60 * Real.pi / 180) + Real.cos (42 * Real.pi / 180) - 
                 Real.sin (12 * Real.pi / 180) - Real.cos (6 * Real.pi / 180) ∧
    θ = 66 * Real.pi / 180 ∧
    ∀ φ : Real, 
      φ > 0 → 
      φ < 2 * Real.pi → 
      Real.cos φ = Real.sin (60 * Real.pi / 180) + Real.cos (42 * Real.pi / 180) - 
                   Real.sin (12 * Real.pi / 180) - Real.cos (6 * Real.pi / 180) → 
      φ ≥ θ

theorem prove_smallest_positive_angle : smallest_positive_angle_theorem :=
sorry

end NUMINAMATH_CALUDE_prove_smallest_positive_angle_l807_80744


namespace NUMINAMATH_CALUDE_nabla_four_seven_l807_80770

-- Define the nabla operation
def nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

-- Theorem statement
theorem nabla_four_seven : nabla 4 7 = 11 / 29 := by
  sorry

end NUMINAMATH_CALUDE_nabla_four_seven_l807_80770


namespace NUMINAMATH_CALUDE_max_distinct_pairs_l807_80723

theorem max_distinct_pairs (n : ℕ) (h : n = 2023) :
  let S := Finset.range n
  ∃ (k : ℕ) (pairs : Finset (ℕ × ℕ)),
    k = 809 ∧
    pairs.card = k ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ n) ∧
    (∀ (m : ℕ) (larger_pairs : Finset (ℕ × ℕ)),
      m > k →
      (larger_pairs.card = m →
        ¬((∀ (p : ℕ × ℕ), p ∈ larger_pairs → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) ∧
          (∀ (p q : ℕ × ℕ), p ∈ larger_pairs → q ∈ larger_pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
          (∀ (p q : ℕ × ℕ), p ∈ larger_pairs → q ∈ larger_pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
          (∀ (p : ℕ × ℕ), p ∈ larger_pairs → p.1 + p.2 ≤ n)))) :=
by
  sorry

end NUMINAMATH_CALUDE_max_distinct_pairs_l807_80723


namespace NUMINAMATH_CALUDE_S_max_at_14_l807_80790

/-- The sequence term for index n -/
def a (n : ℕ) : ℤ := 43 - 3 * n

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℚ := n * (40 + 43 - 3 * n) / 2

/-- The theorem stating that S reaches its maximum when n = 14 -/
theorem S_max_at_14 : ∀ k : ℕ, k > 0 → S 14 ≥ S k := by sorry

end NUMINAMATH_CALUDE_S_max_at_14_l807_80790


namespace NUMINAMATH_CALUDE_dogwood_trees_tomorrow_l807_80792

/-- The number of dogwood trees to be planted tomorrow in the park --/
def trees_planted_tomorrow (initial_trees : ℕ) (planted_today : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial_trees + planted_today)

/-- Theorem: Given the initial number of trees, the number planted today, and the final total,
    prove that 20 trees will be planted tomorrow --/
theorem dogwood_trees_tomorrow :
  trees_planted_tomorrow 39 41 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_trees_tomorrow_l807_80792


namespace NUMINAMATH_CALUDE_least_three_digit_with_digit_product_8_l807_80771

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * tens * ones

theorem least_three_digit_with_digit_product_8 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 8 → 181 ≤ n :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_with_digit_product_8_l807_80771


namespace NUMINAMATH_CALUDE_min_links_remove_10x10_grid_l807_80765

/-- Represents a grid with horizontal and vertical lines -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)
  (horizontal_lines : ℕ)
  (vertical_lines : ℕ)

/-- Calculates the total number of links in the grid -/
def total_links (g : Grid) : ℕ :=
  (g.rows * g.vertical_lines) + (g.cols * g.horizontal_lines)

/-- Calculates the number of interior nodes in the grid -/
def interior_nodes (g : Grid) : ℕ :=
  (g.rows - 1) * (g.cols - 1)

/-- The minimum number of links to remove -/
def min_links_to_remove (g : Grid) : ℕ := 41

/-- Theorem stating the minimum number of links to remove for a 10x10 grid -/
theorem min_links_remove_10x10_grid :
  let g : Grid := { rows := 10, cols := 10, horizontal_lines := 11, vertical_lines := 11 }
  min_links_to_remove g = 41 :=
by sorry

end NUMINAMATH_CALUDE_min_links_remove_10x10_grid_l807_80765


namespace NUMINAMATH_CALUDE_cube_root_of_a_plus_b_plus_one_l807_80778

theorem cube_root_of_a_plus_b_plus_one (a b : ℝ) 
  (h1 : (2 * a - 1) = 9)
  (h2 : (3 * a + b - 1) = 16) : 
  (a + b + 1)^(1/3 : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_a_plus_b_plus_one_l807_80778


namespace NUMINAMATH_CALUDE_triangle_area_prove_triangle_area_l807_80716

/-- Parabola equation: x^2 = 16y -/
def parabola (x y : ℝ) : Prop := x^2 = 16 * y

/-- Hyperbola equation: x^2 - y^2 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

/-- Directrix of the parabola -/
def directrix : ℝ := -4

/-- Asymptotes of the hyperbola -/
def asymptote₁ (x : ℝ) : ℝ := x
def asymptote₂ (x : ℝ) : ℝ := -x

/-- Points where asymptotes intersect the directrix -/
def point₁ : ℝ × ℝ := (4, -4)
def point₂ : ℝ × ℝ := (-4, -4)

/-- The area of the triangle formed by the directrix and asymptotes -/
theorem triangle_area : ℝ := 16

/-- Proof that the area of the triangle is 16 -/
theorem prove_triangle_area : triangle_area = 16 := by sorry

end NUMINAMATH_CALUDE_triangle_area_prove_triangle_area_l807_80716


namespace NUMINAMATH_CALUDE_empty_bucket_weight_l807_80726

theorem empty_bucket_weight (full_weight : ℝ) (partial_weight : ℝ) : 
  full_weight = 3.4 →
  partial_weight = 2.98 →
  ∃ (empty_weight : ℝ),
    empty_weight = 1.3 ∧
    full_weight = empty_weight + (3.4 - empty_weight) ∧
    partial_weight = empty_weight + 4/5 * (3.4 - empty_weight) := by
  sorry

end NUMINAMATH_CALUDE_empty_bucket_weight_l807_80726


namespace NUMINAMATH_CALUDE_problem_solution_l807_80780

theorem problem_solution :
  let A : ℝ → ℝ → ℝ := λ x y => -4 * x^2 - 4 * x * y + 1
  let B : ℝ → ℝ → ℝ := λ x y => x^2 + x * y - 5
  let x : ℝ := 1
  let y : ℝ := -1
  2 * B x y - A x y = -11 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l807_80780


namespace NUMINAMATH_CALUDE_younger_major_A_probability_l807_80707

structure GraduatingClass where
  maleProportion : Real
  majorAProb : Real
  majorBProb : Real
  majorCProb : Real
  maleOlderProb : Real
  femaleOlderProb : Real
  majorAOlderProb : Real
  majorBOlderProb : Real
  majorCOlderProb : Real

def probabilityYoungerMajorA (gc : GraduatingClass) : Real :=
  gc.majorAProb * (1 - gc.majorAOlderProb)

theorem younger_major_A_probability (gc : GraduatingClass) 
  (h1 : gc.maleProportion = 0.4)
  (h2 : gc.majorAProb = 0.5)
  (h3 : gc.majorBProb = 0.3)
  (h4 : gc.majorCProb = 0.2)
  (h5 : gc.maleOlderProb = 0.5)
  (h6 : gc.femaleOlderProb = 0.3)
  (h7 : gc.majorAOlderProb = 0.6)
  (h8 : gc.majorBOlderProb = 0.4)
  (h9 : gc.majorCOlderProb = 0.2) :
  probabilityYoungerMajorA gc = 0.2 := by
  sorry

#check younger_major_A_probability

end NUMINAMATH_CALUDE_younger_major_A_probability_l807_80707


namespace NUMINAMATH_CALUDE_intersection_point_property_l807_80705

theorem intersection_point_property (α : ℝ) (h1 : α ≠ 0) (h2 : Real.tan α = -α) :
  (α^2 + 1) * (1 + Real.cos (2 * α)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_property_l807_80705


namespace NUMINAMATH_CALUDE_sum_of_smaller_radii_eq_original_radius_l807_80795

/-- A triangle with an inscribed circle and three smaller triangles formed by tangents -/
structure InscribedCircleTriangle where
  /-- The radius of the circle inscribed in the original triangle -/
  r : ℝ
  /-- The radius of the circle inscribed in the first smaller triangle -/
  r₁ : ℝ
  /-- The radius of the circle inscribed in the second smaller triangle -/
  r₂ : ℝ
  /-- The radius of the circle inscribed in the third smaller triangle -/
  r₃ : ℝ
  /-- Ensure all radii are positive -/
  r_pos : r > 0
  r₁_pos : r₁ > 0
  r₂_pos : r₂ > 0
  r₃_pos : r₃ > 0

/-- The sum of the radii of the inscribed circles in the smaller triangles
    equals the radius of the inscribed circle in the original triangle -/
theorem sum_of_smaller_radii_eq_original_radius (t : InscribedCircleTriangle) :
  t.r₁ + t.r₂ + t.r₃ = t.r := by
  sorry

end NUMINAMATH_CALUDE_sum_of_smaller_radii_eq_original_radius_l807_80795


namespace NUMINAMATH_CALUDE_xy_values_l807_80727

theorem xy_values (x y : ℝ) (h1 : x - y = 5) (h2 : x^3 - y^3 = 62) :
  x * y = -126/25 ∨ x * y = -6 := by
  sorry

end NUMINAMATH_CALUDE_xy_values_l807_80727


namespace NUMINAMATH_CALUDE_bug_walk_tiles_l807_80796

/-- The number of tiles a bug visits when walking in a straight line from one corner to the opposite corner of a rectangular floor. -/
def tilesVisited (width : ℕ) (length : ℕ) : ℕ :=
  width + length - Nat.gcd width length

/-- The theorem stating that for a 15x35 foot rectangular floor, a bug walking diagonally visits 45 tiles. -/
theorem bug_walk_tiles : tilesVisited 15 35 = 45 := by
  sorry

end NUMINAMATH_CALUDE_bug_walk_tiles_l807_80796


namespace NUMINAMATH_CALUDE_compute_expression_l807_80735

theorem compute_expression : 3 * ((25 + 15)^2 - (25 - 15)^2) = 4500 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l807_80735


namespace NUMINAMATH_CALUDE_range_of_m_l807_80732

/-- The function f(x) as defined in the problem -/
def f (a x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x

/-- The theorem statement -/
theorem range_of_m (m : ℝ) :
  (∀ a ∈ Set.Icc (-3 : ℝ) 0,
    ∀ x₁ ∈ Set.Icc 0 2,
    ∀ x₂ ∈ Set.Icc 0 2,
    m - a * m^2 ≥ |f a x₁ - f a x₂|) →
  m ∈ Set.Ici 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l807_80732


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l807_80788

/-- Given a complex number z satisfying zi = 2 + i, prove that the real part of z is positive
    and the imaginary part of z is negative. -/
theorem z_in_fourth_quadrant (z : ℂ) (h : z * Complex.I = 2 + Complex.I) :
  0 < z.re ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l807_80788


namespace NUMINAMATH_CALUDE_sin_alpha_cos_beta_value_l807_80772

/-- Given two points on the unit circle representing the terminal sides of angles α and β,
    prove that sin(α) * cos(β) equals a specific value. -/
theorem sin_alpha_cos_beta_value (α β : Real) :
  (∃ (x y : Real), x^2 + y^2 = 1 ∧ x = 12/13 ∧ y = 5/13 ∧ 
   Real.cos α = x ∧ Real.sin α = y) →
  (∃ (u v : Real), u^2 + v^2 = 1 ∧ u = -3/5 ∧ v = 4/5 ∧ 
   Real.cos β = u ∧ Real.sin β = v) →
  Real.sin α * Real.cos β = -15/65 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_cos_beta_value_l807_80772


namespace NUMINAMATH_CALUDE_max_value_on_circle_l807_80775

theorem max_value_on_circle (x y : ℝ) : 
  x^2 + y^2 = 20*x + 24*y + 26 → (5*x + 3*y ≤ 73) ∧ ∃ x y, x^2 + y^2 = 20*x + 24*y + 26 ∧ 5*x + 3*y = 73 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l807_80775


namespace NUMINAMATH_CALUDE_factorization_demonstrates_transformation_l807_80753

/-- Represents a quadratic equation of the form ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the method used to solve the equation -/
inductive SolvingMethod
  | Factorization

/-- Represents the mathematical idea demonstrated by the solving method -/
inductive MathematicalIdea
  | Transformation
  | Function
  | CombiningNumbersAndShapes
  | Axiomatic

/-- Solves a quadratic equation using the given method -/
def solveQuadratic (eq : QuadraticEquation) (method : SolvingMethod) : Set ℝ :=
  sorry

/-- Determines the mathematical idea demonstrated by the solving method -/
def demonstratedIdea (eq : QuadraticEquation) (method : SolvingMethod) : MathematicalIdea :=
  sorry

theorem factorization_demonstrates_transformation : 
  let eq : QuadraticEquation := { a := 3, b := -6, c := 0 }
  demonstratedIdea eq SolvingMethod.Factorization = MathematicalIdea.Transformation :=
by sorry

end NUMINAMATH_CALUDE_factorization_demonstrates_transformation_l807_80753


namespace NUMINAMATH_CALUDE_new_students_count_l807_80713

theorem new_students_count (n : ℕ) : 
  n < 600 ∧ 
  n % 28 = 27 ∧ 
  n % 26 = 20 → 
  n = 615 :=
by sorry

end NUMINAMATH_CALUDE_new_students_count_l807_80713


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l807_80783

theorem arithmetic_expression_equality : 70 + 5 * 12 / (180 / 3) = 71 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l807_80783


namespace NUMINAMATH_CALUDE_fraction_equality_l807_80752

theorem fraction_equality : ∃! (n m : ℕ) (d : ℚ), n > 0 ∧ m > 0 ∧ (n : ℚ) / m = d ∧ d = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l807_80752


namespace NUMINAMATH_CALUDE_root_difference_ratio_l807_80774

theorem root_difference_ratio (a b : ℝ) : 
  a > b ∧ b > 0 ∧ 
  a^2 - 6*a + 4 = 0 ∧ 
  b^2 - 6*b + 4 = 0 → 
  (Real.sqrt a - Real.sqrt b) / (Real.sqrt a + Real.sqrt b) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_root_difference_ratio_l807_80774


namespace NUMINAMATH_CALUDE_crayons_in_drawer_l807_80766

/-- The number of crayons remaining in a drawer after some are removed. -/
def crayons_remaining (initial : ℕ) (removed : ℕ) : ℕ :=
  initial - removed

/-- Theorem: Given 7 initial crayons and 3 removed, 4 crayons remain. -/
theorem crayons_in_drawer : crayons_remaining 7 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_crayons_in_drawer_l807_80766


namespace NUMINAMATH_CALUDE_sum_of_ages_age_difference_l807_80742

/-- Tyler's age -/
def tyler_age : ℕ := 7

/-- Tyler's brother's age -/
def brother_age : ℕ := 11 - tyler_age

/-- The sum of Tyler's and his brother's ages -/
theorem sum_of_ages : tyler_age + brother_age = 11 := by sorry

/-- The difference between Tyler's brother's age and Tyler's age -/
theorem age_difference : brother_age - tyler_age = 4 := by sorry

end NUMINAMATH_CALUDE_sum_of_ages_age_difference_l807_80742


namespace NUMINAMATH_CALUDE_quadratic_real_equal_roots_l807_80762

theorem quadratic_real_equal_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - (m - 1) * x + 2 * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - (m - 1) * y + 2 * y + 12 = 0 → y = x) ↔ 
  (m = -10 ∨ m = 14) := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_equal_roots_l807_80762


namespace NUMINAMATH_CALUDE_triangle_problem_l807_80797

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  -- Given conditions
  a = 1 ∧ b = 2 ∧ Real.cos C = 1/4 →
  -- Prove
  c = 2 ∧ Real.sin A = Real.sqrt 15 / 8 := by
    sorry

end NUMINAMATH_CALUDE_triangle_problem_l807_80797


namespace NUMINAMATH_CALUDE_ratio_fraction_equality_l807_80720

theorem ratio_fraction_equality (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 6) :
  (5 * A + 3 * B) / (5 * C - 2 * A) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_fraction_equality_l807_80720


namespace NUMINAMATH_CALUDE_tank_capacity_l807_80715

theorem tank_capacity (initial_fill : Rat) (added_amount : Rat) (final_fill : Rat) :
  initial_fill = 3 / 4 →
  added_amount = 8 →
  final_fill = 9 / 10 →
  ∃ (capacity : Rat), capacity = 160 / 3 ∧ 
    final_fill * capacity - initial_fill * capacity = added_amount :=
by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l807_80715


namespace NUMINAMATH_CALUDE_x_plus_y_value_l807_80741

theorem x_plus_y_value (x y : ℝ) (hx : |x| = 2) (hy : |y| = 3) (hxy : x > y) : x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l807_80741


namespace NUMINAMATH_CALUDE_special_numbers_count_l807_80743

def count_special_numbers (n : ℕ) : ℕ :=
  (n / 12) - (n / 60)

theorem special_numbers_count :
  count_special_numbers 2017 = 135 := by
  sorry

end NUMINAMATH_CALUDE_special_numbers_count_l807_80743


namespace NUMINAMATH_CALUDE_test_question_count_l807_80794

/-- Given a test with the following properties:
  * The test is worth 100 points
  * There are 2-point and 4-point questions
  * There are 30 two-point questions
  Prove that the total number of questions is 40 -/
theorem test_question_count (total_points : ℕ) (two_point_count : ℕ) :
  total_points = 100 →
  two_point_count = 30 →
  ∃ (four_point_count : ℕ),
    total_points = 2 * two_point_count + 4 * four_point_count ∧
    two_point_count + four_point_count = 40 :=
by sorry

end NUMINAMATH_CALUDE_test_question_count_l807_80794


namespace NUMINAMATH_CALUDE_two_card_picks_from_two_decks_l807_80733

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)

/-- Represents the total collection of cards from two shuffled decks -/
def ShuffledDecks (d : Deck) : Nat :=
  2 * d.cards

/-- The number of ways to pick two different cards from shuffled decks -/
def PickTwoCards (total : Nat) : Nat :=
  total * (total - 1)

theorem two_card_picks_from_two_decks :
  let standard_deck : Deck := { cards := 52, suits := 4, cards_per_suit := 13 }
  let shuffled_total := ShuffledDecks standard_deck
  PickTwoCards shuffled_total = 10692 := by
  sorry

end NUMINAMATH_CALUDE_two_card_picks_from_two_decks_l807_80733


namespace NUMINAMATH_CALUDE_frog_arrangement_count_l807_80754

def frog_arrangements (n : ℕ) (g r : ℕ) (b : ℕ) : Prop :=
  n = g + r + b ∧
  g = 3 ∧
  r = 3 ∧
  b = 1

theorem frog_arrangement_count :
  ∀ (n g r b : ℕ),
    frog_arrangements n g r b →
    (n - 1) * 2 * (g.factorial * r.factorial) = 504 :=
by sorry

end NUMINAMATH_CALUDE_frog_arrangement_count_l807_80754


namespace NUMINAMATH_CALUDE_complex_number_imaginary_part_l807_80768

theorem complex_number_imaginary_part (z : ℂ) (h : (1 + z) / (1 - z) = I) : z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_imaginary_part_l807_80768


namespace NUMINAMATH_CALUDE_f_2x_l807_80763

/-- Given a function f(x) = x^2 - 1, prove that f(2x) = 4x^2 - 1 --/
theorem f_2x (x : ℝ) : (fun x => x^2 - 1) (2*x) = 4*x^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_2x_l807_80763


namespace NUMINAMATH_CALUDE_triangle_side_validity_l807_80722

/-- Checks if three lengths can form a valid triangle -/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_side_validity :
  let side1 := 5
  let side2 := 7
  (is_valid_triangle side1 side2 6) ∧
  ¬(is_valid_triangle side1 side2 2) ∧
  ¬(is_valid_triangle side1 side2 17) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_validity_l807_80722


namespace NUMINAMATH_CALUDE_average_transformation_l807_80758

theorem average_transformation (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : (a₁ + a₂ + a₃ + a₄ + a₅) / 5 = 8) : 
  ((a₁ + 10) + (a₂ - 10) + (a₃ + 10) + (a₄ - 10) + (a₅ + 10)) / 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_transformation_l807_80758


namespace NUMINAMATH_CALUDE_retirement_total_is_70_l807_80748

/-- The retirement eligibility rule for a company -/
structure RetirementRule where
  hire_year : ℕ
  hire_age : ℕ
  eligible_year : ℕ

/-- Calculate the required total of age and years of employment for retirement -/
def retirement_total (rule : RetirementRule) : ℕ :=
  let years_of_employment := rule.eligible_year - rule.hire_year
  let age_at_eligibility := rule.hire_age + years_of_employment
  age_at_eligibility + years_of_employment

/-- Theorem stating the required total for retirement -/
theorem retirement_total_is_70 (rule : RetirementRule) 
  (h1 : rule.hire_year = 1990)
  (h2 : rule.hire_age = 32)
  (h3 : rule.eligible_year = 2009) :
  retirement_total rule = 70 := by
  sorry

#eval retirement_total ⟨1990, 32, 2009⟩

end NUMINAMATH_CALUDE_retirement_total_is_70_l807_80748


namespace NUMINAMATH_CALUDE_subset_intersection_complement_empty_l807_80701

theorem subset_intersection_complement_empty
  {U : Type} [Nonempty U]
  (M N : Set U)
  (h : M ⊆ N) :
  M ∩ (Set.univ \ N) = ∅ := by
sorry

end NUMINAMATH_CALUDE_subset_intersection_complement_empty_l807_80701


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l807_80759

theorem complex_number_quadrant : ∃ (x y : ℝ), (Complex.mk x y = (2 - Complex.I)^2) ∧ (x > 0) ∧ (y < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l807_80759


namespace NUMINAMATH_CALUDE_isabelle_concert_savings_l807_80700

/-- The number of weeks Isabelle needs to work to afford concert tickets for herself and her brothers -/
def weeks_to_work (isabelle_ticket_cost : ℕ) (brother_ticket_cost : ℕ) (isabelle_savings : ℕ) (brothers_savings : ℕ) (weekly_pay : ℕ) : ℕ :=
  let total_cost := isabelle_ticket_cost + 2 * brother_ticket_cost
  let total_savings := isabelle_savings + brothers_savings
  let remaining_cost := total_cost - total_savings
  remaining_cost / weekly_pay

theorem isabelle_concert_savings : weeks_to_work 20 10 5 5 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_isabelle_concert_savings_l807_80700


namespace NUMINAMATH_CALUDE_downstream_speed_theorem_l807_80791

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  stillWater : ℝ
  upstream : ℝ

/-- Calculates the downstream speed given the rowing speeds in still water and upstream -/
def downstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.upstream

/-- Theorem stating that given the specific conditions, the downstream speed is 31 kmph -/
theorem downstream_speed_theorem (s : RowingSpeed) 
  (h1 : s.stillWater = 28) 
  (h2 : s.upstream = 25) : 
  downstreamSpeed s = 31 := by
  sorry

#check downstream_speed_theorem

end NUMINAMATH_CALUDE_downstream_speed_theorem_l807_80791


namespace NUMINAMATH_CALUDE_total_amount_is_120_l807_80729

def amount_from_grandpa : ℕ := 30

def amount_from_grandma : ℕ := 3 * amount_from_grandpa

def total_amount : ℕ := amount_from_grandpa + amount_from_grandma

theorem total_amount_is_120 : total_amount = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_120_l807_80729


namespace NUMINAMATH_CALUDE_next_simultaneous_ring_l807_80789

def library_interval : ℕ := 18
def hospital_interval : ℕ := 24
def community_center_interval : ℕ := 30

def initial_time : ℕ := 8 * 60  -- 8:00 AM in minutes since midnight

theorem next_simultaneous_ring (t : ℕ) : 
  t > initial_time ∧ 
  t % library_interval = 0 ∧ 
  t % hospital_interval = 0 ∧ 
  t % community_center_interval = 0 →
  t - initial_time = 6 * 60 := by
sorry

end NUMINAMATH_CALUDE_next_simultaneous_ring_l807_80789


namespace NUMINAMATH_CALUDE_periodic_sequence_characterization_l807_80719

def is_periodic_sequence (x : ℕ → ℝ) : Prop :=
  ∃ T : ℕ, T > 0 ∧ ∀ n, x (n + T) = x n

theorem periodic_sequence_characterization
  (x : ℕ → ℝ)
  (h_pos : ∀ n, x n > 0)
  (h_periodic : is_periodic_sequence x)
  (h_recurrence : ∀ n, x (n + 2) = (1 / 2) * (1 / x (n + 1) + x n)) :
  ∃ a : ℝ, a > 0 ∧ ∀ n, x n = if n % 2 = 0 then a else 1 / a :=
sorry

end NUMINAMATH_CALUDE_periodic_sequence_characterization_l807_80719


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l807_80782

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (3 * i + 1) / (1 - i)
  Complex.im z = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l807_80782


namespace NUMINAMATH_CALUDE_focus_on_negative_y_axis_l807_80746

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 + y = 0

-- Define the focus of a parabola
def focus (p q : ℝ) : Prop := ∃ (a : ℝ), p = 0 ∧ q = -1/(4*a)

-- Theorem statement
theorem focus_on_negative_y_axis :
  ∃ (p q : ℝ), focus p q ∧ ∀ (x y : ℝ), parabola x y → q < 0 :=
sorry

end NUMINAMATH_CALUDE_focus_on_negative_y_axis_l807_80746


namespace NUMINAMATH_CALUDE_simplify_expression_l807_80709

theorem simplify_expression : 
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 98 / Real.sqrt 49) = (3 + 2 * Real.sqrt 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l807_80709


namespace NUMINAMATH_CALUDE_log_inequality_condition_l807_80786

theorem log_inequality_condition (a b : ℝ) : 
  (∀ a b, Real.log a > Real.log b → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(Real.log a > Real.log b)) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_condition_l807_80786


namespace NUMINAMATH_CALUDE_student_count_l807_80706

theorem student_count (total_erasers total_pencils leftover_erasers leftover_pencils : ℕ)
  (h1 : total_erasers = 49)
  (h2 : total_pencils = 66)
  (h3 : leftover_erasers = 4)
  (h4 : leftover_pencils = 6) :
  ∃ (students : ℕ),
    students > 0 ∧
    (total_erasers - leftover_erasers) % students = 0 ∧
    (total_pencils - leftover_pencils) % students = 0 ∧
    students = 15 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l807_80706


namespace NUMINAMATH_CALUDE_inverse_proportionality_l807_80737

theorem inverse_proportionality (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 5 * 10 = k) :
  40 * (5/4 : ℝ) = k := by sorry

end NUMINAMATH_CALUDE_inverse_proportionality_l807_80737


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l807_80724

theorem right_triangle_hypotenuse (a b c : ℝ) :
  a = 15 ∧ b = 36 ∧ c^2 = a^2 + b^2 → c = 39 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l807_80724


namespace NUMINAMATH_CALUDE_max_students_equal_distribution_l807_80740

theorem max_students_equal_distribution (pens pencils : ℕ) (h1 : pens = 1048) (h2 : pencils = 828) :
  (Nat.gcd pens pencils : ℕ) = 4 :=
sorry

end NUMINAMATH_CALUDE_max_students_equal_distribution_l807_80740


namespace NUMINAMATH_CALUDE_function_defined_for_all_reals_l807_80767

/-- The function f(t) is defined for all real numbers t. -/
theorem function_defined_for_all_reals :
  ∀ t : ℝ, ∃ y : ℝ, y = 1 / ((t - 1)^2 + (t + 1)^2) := by
  sorry

end NUMINAMATH_CALUDE_function_defined_for_all_reals_l807_80767


namespace NUMINAMATH_CALUDE_return_trip_amount_l807_80764

def initial_amount : ℝ := 50
def gasoline_cost : ℝ := 8
def lunch_cost : ℝ := 15.65
def gift_cost_per_person : ℝ := 5
def number_of_people : ℕ := 2
def grandma_gift_per_person : ℝ := 10

def total_expenses : ℝ := gasoline_cost + lunch_cost + (gift_cost_per_person * number_of_people)
def remaining_after_expenses : ℝ := initial_amount - total_expenses
def total_grandma_gift : ℝ := grandma_gift_per_person * number_of_people
def final_amount : ℝ := remaining_after_expenses + total_grandma_gift

theorem return_trip_amount :
  final_amount = 36.35 := by sorry

end NUMINAMATH_CALUDE_return_trip_amount_l807_80764


namespace NUMINAMATH_CALUDE_remainder_problem_l807_80728

theorem remainder_problem (N : ℤ) (h : N % 133 = 16) : N % 50 = 49 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l807_80728


namespace NUMINAMATH_CALUDE_arbitrarily_large_special_numbers_l807_80711

/-- A function that checks if all digits of a natural number are 2 or more -/
def all_digits_two_or_more (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d ≥ 2

/-- A function that checks if the product of any four digits of a number divides the number -/
def product_of_four_divides (n : ℕ) : Prop :=
  ∀ a b c d, a ∈ n.digits 10 → b ∈ n.digits 10 → c ∈ n.digits 10 → d ∈ n.digits 10 →
    (a * b * c * d) ∣ n

/-- The main theorem stating that for any k, there exists a number n > k satisfying the conditions -/
theorem arbitrarily_large_special_numbers :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ all_digits_two_or_more n ∧ product_of_four_divides n :=
sorry

end NUMINAMATH_CALUDE_arbitrarily_large_special_numbers_l807_80711


namespace NUMINAMATH_CALUDE_base_dimensions_of_divided_volume_l807_80760

/-- Given a volume of 120 cubic cubits divided into 10 parts, each with a height of 1 cubit,
    and a rectangular base with sides in the ratio 1:3/4, prove that the dimensions of the base
    are 4 cubits and 3 cubits. -/
theorem base_dimensions_of_divided_volume (total_volume : ℝ) (num_parts : ℕ) 
    (part_height : ℝ) (base_ratio : ℝ) :
  total_volume = 120 →
  num_parts = 10 →
  part_height = 1 →
  base_ratio = 3/4 →
  ∃ (a b : ℝ), a = 4 ∧ b = 3 ∧
    a * b * part_height * num_parts = total_volume ∧
    b / a = base_ratio :=
by sorry

end NUMINAMATH_CALUDE_base_dimensions_of_divided_volume_l807_80760


namespace NUMINAMATH_CALUDE_neg_cube_eq_cube_of_neg_l807_80704

theorem neg_cube_eq_cube_of_neg (x : ℚ) : -x^3 = (-x)^3 := by
  sorry

end NUMINAMATH_CALUDE_neg_cube_eq_cube_of_neg_l807_80704


namespace NUMINAMATH_CALUDE_solution_set_and_range_l807_80779

def f (x : ℝ) : ℝ := |x + 1|

theorem solution_set_and_range :
  (∀ x : ℝ, f x ≤ 5 - f (x - 3) ↔ -2 ≤ x ∧ x ≤ 3) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ 2 * f x + |x + a| ≤ x + 4 → -2 ≤ a ∧ a ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_solution_set_and_range_l807_80779


namespace NUMINAMATH_CALUDE_juniper_whiskers_l807_80777

/-- Represents the number of whiskers for each cat -/
structure CatWhiskers where
  puffy : ℕ
  scruffy : ℕ
  buffy : ℕ
  juniper : ℕ

/-- Defines the conditions for the cat whisker problem -/
def whisker_conditions (c : CatWhiskers) : Prop :=
  c.buffy = 40 ∧
  c.puffy = 3 * c.juniper ∧
  c.puffy = c.scruffy / 2 ∧
  c.buffy = (c.puffy + c.scruffy + c.juniper) / 3

/-- Theorem stating that under the given conditions, Juniper has 12 whiskers -/
theorem juniper_whiskers (c : CatWhiskers) : 
  whisker_conditions c → c.juniper = 12 := by
  sorry

#check juniper_whiskers

end NUMINAMATH_CALUDE_juniper_whiskers_l807_80777


namespace NUMINAMATH_CALUDE_division_remainder_problem_l807_80784

theorem division_remainder_problem (L S R : ℕ) : 
  L - S = 1365 → 
  L = 1631 → 
  L = 6 * S + R → 
  R = 35 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l807_80784


namespace NUMINAMATH_CALUDE_calculation_proof_l807_80757

theorem calculation_proof : 121 * (13 / 25) + 12 * (21 / 25) = 73 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l807_80757


namespace NUMINAMATH_CALUDE_polygon_with_170_diagonals_has_20_sides_l807_80717

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 170 diagonals has 20 sides -/
theorem polygon_with_170_diagonals_has_20_sides :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 170 → n = 20 :=
by
  sorry

#check polygon_with_170_diagonals_has_20_sides

end NUMINAMATH_CALUDE_polygon_with_170_diagonals_has_20_sides_l807_80717


namespace NUMINAMATH_CALUDE_square_area_with_circles_8_l807_80730

/-- The area of a square containing four circles of radius r, with two circles touching each side of the square. -/
def square_area_with_circles (r : ℝ) : ℝ :=
  (4 * r) ^ 2

/-- Theorem: The area of a square containing four circles of radius 8 inches, 
    with two circles touching each side of the square, is 1024 square inches. -/
theorem square_area_with_circles_8 : 
  square_area_with_circles 8 = 1024 :=
by sorry

end NUMINAMATH_CALUDE_square_area_with_circles_8_l807_80730


namespace NUMINAMATH_CALUDE_cube_side_length_is_one_l807_80734

/-- The surface area of a cuboid formed by joining two cubes with side length s is 10 -/
def cuboid_surface_area (s : ℝ) : ℝ := 10 * s^2

/-- Theorem: If two cubes with side length s are joined to form a cuboid with surface area 10, then s = 1 -/
theorem cube_side_length_is_one :
  ∃ (s : ℝ), s > 0 ∧ cuboid_surface_area s = 10 → s = 1 :=
by sorry

end NUMINAMATH_CALUDE_cube_side_length_is_one_l807_80734


namespace NUMINAMATH_CALUDE_two_distinct_prime_factors_iff_n_zero_l807_80761

def base_6_to_decimal (base_6_num : List Nat) : Nat :=
  base_6_num.enum.foldr (λ (i, digit) acc => acc + digit * (6 ^ i)) 0

def append_fives (n : Nat) : List Nat :=
  [1, 2, 0, 0] ++ List.replicate (10 * n + 2) 5

def result_number (n : Nat) : Nat :=
  base_6_to_decimal (append_fives n)

def has_exactly_two_distinct_prime_factors (x : Nat) : Prop :=
  ∃ p q : Nat, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧
  ∃ a b : Nat, x = p^a * q^b ∧ 
  ∀ r : Nat, Nat.Prime r → r ∣ x → (r = p ∨ r = q)

theorem two_distinct_prime_factors_iff_n_zero (n : Nat) :
  has_exactly_two_distinct_prime_factors (result_number n) ↔ n = 0 := by
  sorry

#check two_distinct_prime_factors_iff_n_zero

end NUMINAMATH_CALUDE_two_distinct_prime_factors_iff_n_zero_l807_80761


namespace NUMINAMATH_CALUDE_contrapositive_example_l807_80725

theorem contrapositive_example :
  (∀ x : ℝ, x = 2 → x^2 - 3*x + 2 = 0) ↔
  (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0 → x ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_example_l807_80725


namespace NUMINAMATH_CALUDE_subtract_fractions_l807_80799

theorem subtract_fractions (p q : ℚ) (h1 : 3 / p = 4) (h2 : 3 / q = 18) : p - q = 7/12 := by
  sorry

end NUMINAMATH_CALUDE_subtract_fractions_l807_80799


namespace NUMINAMATH_CALUDE_rhombus_diagonals_not_necessarily_equal_l807_80714

/-- Definition of a rhombus -/
structure Rhombus :=
  (sides : Fin 4 → ℝ)
  (equal_sides : ∀ i j : Fin 4, sides i = sides j)
  (perpendicular_diagonals : True)  -- We simplify this condition for the purpose of this problem

/-- Definition of diagonals of a rhombus -/
def diagonals (r : Rhombus) : Fin 2 → ℝ := sorry

/-- Theorem stating that the diagonals of a rhombus are not necessarily equal -/
theorem rhombus_diagonals_not_necessarily_equal :
  ¬ (∀ r : Rhombus, diagonals r 0 = diagonals r 1) :=
sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_not_necessarily_equal_l807_80714


namespace NUMINAMATH_CALUDE_investment_rate_problem_l807_80798

theorem investment_rate_problem (total_investment : ℝ) (first_investment : ℝ) (second_investment : ℝ)
  (first_rate : ℝ) (second_rate : ℝ) (desired_income : ℝ) :
  total_investment = 12000 →
  first_investment = 5000 →
  second_investment = 4000 →
  first_rate = 0.03 →
  second_rate = 0.035 →
  desired_income = 430 →
  let remainder := total_investment - first_investment - second_investment
  let income_from_first := first_investment * first_rate
  let income_from_second := second_investment * second_rate
  let required_additional_income := desired_income - income_from_first - income_from_second
  let required_rate := required_additional_income / remainder
  required_rate = 0.047 := by
  sorry

end NUMINAMATH_CALUDE_investment_rate_problem_l807_80798


namespace NUMINAMATH_CALUDE_cat_shortest_distance_to_origin_l807_80785

theorem cat_shortest_distance_to_origin :
  let center : ℝ × ℝ := (5, -2)
  let radius : ℝ := 8
  let origin : ℝ × ℝ := (0, 0)
  let distance_center_to_origin : ℝ := Real.sqrt ((center.1 - origin.1)^2 + (center.2 - origin.2)^2)
  ∀ p : ℝ × ℝ, (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 →
    Real.sqrt ((p.1 - origin.1)^2 + (p.2 - origin.2)^2) ≥ |distance_center_to_origin - radius| :=
by sorry

end NUMINAMATH_CALUDE_cat_shortest_distance_to_origin_l807_80785


namespace NUMINAMATH_CALUDE_nicky_trade_profit_l807_80736

/-- Calculates Nicky's profit or loss in a baseball card trade with Jill --/
theorem nicky_trade_profit :
  let nicky_card1_value : ℚ := 8
  let nicky_card1_count : ℕ := 2
  let nicky_card2_value : ℚ := 5
  let nicky_card2_count : ℕ := 3
  let jill_card1_value_cad : ℚ := 21
  let jill_card1_count : ℕ := 1
  let jill_card2_value_cad : ℚ := 6
  let jill_card2_count : ℕ := 2
  let exchange_rate_usd_per_cad : ℚ := 0.8
  let tax_rate : ℚ := 0.05

  let nicky_total_value := nicky_card1_value * nicky_card1_count + nicky_card2_value * nicky_card2_count
  let jill_total_value_cad := jill_card1_value_cad * jill_card1_count + jill_card2_value_cad * jill_card2_count
  let jill_total_value_usd := jill_total_value_cad * exchange_rate_usd_per_cad
  let total_trade_value_usd := nicky_total_value + jill_total_value_usd
  let tax_amount := total_trade_value_usd * tax_rate
  let nicky_profit := jill_total_value_usd - (nicky_total_value + tax_amount)

  nicky_profit = -7.47 := by sorry

end NUMINAMATH_CALUDE_nicky_trade_profit_l807_80736


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l807_80745

/-- Given an arithmetic sequence with first term 3 and last term 27,
    the sum of the two terms immediately preceding 27 is 42. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) : 
  (∃ d : ℝ, ∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  a 0 = 3 →  -- first term is 3
  (∃ k : ℕ, a k = 27 ∧ ∀ n > k, a n ≠ 27) →  -- 27 is the last term
  (∃ m : ℕ, a (m - 1) + a m = 42 ∧ a (m + 1) = 27) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l807_80745


namespace NUMINAMATH_CALUDE_average_words_in_crossword_puzzle_l807_80738

/-- The number of words needed to use up a pencil -/
def words_per_pencil : ℕ := 1050

/-- The number of days in two weeks -/
def days_in_two_weeks : ℕ := 14

/-- The number of puzzles completed in two weeks -/
def puzzles_in_two_weeks : ℕ := days_in_two_weeks

/-- The average number of words in each crossword puzzle -/
def average_words_per_puzzle : ℚ := words_per_pencil / puzzles_in_two_weeks

theorem average_words_in_crossword_puzzle :
  average_words_per_puzzle = 75 := by sorry

end NUMINAMATH_CALUDE_average_words_in_crossword_puzzle_l807_80738


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l807_80749

/-- 
For a quadratic equation x^2 - 2x + a = 0, if the square of the difference 
between its roots is 20, then a = -4.
-/
theorem quadratic_root_difference (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 - 2*x₁ + a = 0 ∧ x₂^2 - 2*x₂ + a = 0 ∧ (x₁ - x₂)^2 = 20) →
  a = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l807_80749


namespace NUMINAMATH_CALUDE_visible_cubes_count_l807_80787

/-- Represents a cube with gaps -/
structure CubeWithGaps where
  size : ℕ
  unit_cubes : ℕ
  has_gaps : Bool

/-- Calculates the number of visible or partially visible unit cubes from a corner -/
def visible_cubes (c : CubeWithGaps) : ℕ :=
  sorry

/-- The specific cube in the problem -/
def problem_cube : CubeWithGaps :=
  { size := 12
  , unit_cubes := 12^3
  , has_gaps := true }

theorem visible_cubes_count :
  visible_cubes problem_cube = 412 :=
sorry

end NUMINAMATH_CALUDE_visible_cubes_count_l807_80787


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_18_with_cube_root_between_8_and_8_1_l807_80769

theorem unique_integer_divisible_by_18_with_cube_root_between_8_and_8_1 :
  ∃! n : ℕ+, 18 ∣ n ∧ (8 : ℝ) < (n : ℝ)^(1/3) ∧ (n : ℝ)^(1/3) < (8.1 : ℝ) ∧ n = 522 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_18_with_cube_root_between_8_and_8_1_l807_80769


namespace NUMINAMATH_CALUDE_john_gum_purchase_l807_80750

/-- The number of packs of gum John bought -/
def num_gum_packs : ℕ := 2

/-- The number of candy bars John bought -/
def num_candy_bars : ℕ := 3

/-- The cost of one candy bar in dollars -/
def candy_bar_cost : ℚ := 3/2

/-- The total amount John paid in dollars -/
def total_paid : ℚ := 6

/-- The cost of one pack of gum in dollars -/
def gum_pack_cost : ℚ := candy_bar_cost / 2

theorem john_gum_purchase :
  num_gum_packs * gum_pack_cost + num_candy_bars * candy_bar_cost = total_paid :=
by sorry

end NUMINAMATH_CALUDE_john_gum_purchase_l807_80750


namespace NUMINAMATH_CALUDE_parade_average_l807_80721

theorem parade_average (boys girls rows : ℕ) (h1 : boys = 24) (h2 : girls = 24) (h3 : rows = 6) :
  (boys + girls) / rows = 8 :=
sorry

end NUMINAMATH_CALUDE_parade_average_l807_80721


namespace NUMINAMATH_CALUDE_inequality_proof_l807_80712

theorem inequality_proof (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l807_80712


namespace NUMINAMATH_CALUDE_f_five_l807_80708

/-- A function satisfying f(xy) = 3xf(y) for all real x and y, with f(1) = 10 -/
def f : ℝ → ℝ :=
  sorry

/-- The functional equation for f -/
axiom f_eq (x y : ℝ) : f (x * y) = 3 * x * f y

/-- The value of f at 1 -/
axiom f_one : f 1 = 10

/-- The main theorem: f(5) = 150 -/
theorem f_five : f 5 = 150 := by
  sorry

end NUMINAMATH_CALUDE_f_five_l807_80708


namespace NUMINAMATH_CALUDE_locus_equals_thales_circles_l807_80776

/-- A triangle in a plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A point in the plane -/
def Point : Type := ℝ × ℝ

/-- The angle subtended by a side of the triangle from a point -/
noncomputable def subtended_angle (t : Triangle) (p : Point) (side : Fin 3) : ℝ :=
  sorry

/-- The sum of angles subtended by the three sides of the triangle from a point -/
noncomputable def sum_of_subtended_angles (t : Triangle) (p : Point) : ℝ :=
  (subtended_angle t p 0) + (subtended_angle t p 1) + (subtended_angle t p 2)

/-- The Thales' circle for a side of the triangle -/
def thales_circle (t : Triangle) (side : Fin 3) : Set Point :=
  sorry

/-- The set of points on all Thales' circles, excluding the triangle's vertices -/
def thales_circles_points (t : Triangle) : Set Point :=
  (thales_circle t 0 ∪ thales_circle t 1 ∪ thales_circle t 2) \ {t.A, t.B, t.C}

/-- The theorem stating the equivalence of the locus and the Thales' circles points -/
theorem locus_equals_thales_circles (t : Triangle) :
  {p : Point | sum_of_subtended_angles t p = π} = thales_circles_points t :=
  sorry

end NUMINAMATH_CALUDE_locus_equals_thales_circles_l807_80776


namespace NUMINAMATH_CALUDE_bake_sale_brownie_cost_l807_80718

/-- Proves that the cost per brownie is $2 given the conditions of the bake sale --/
theorem bake_sale_brownie_cost (total_revenue : ℝ) (num_pans : ℕ) (pieces_per_pan : ℕ) :
  total_revenue = 32 →
  num_pans = 2 →
  pieces_per_pan = 8 →
  (total_revenue / (num_pans * pieces_per_pan : ℝ)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_bake_sale_brownie_cost_l807_80718


namespace NUMINAMATH_CALUDE_y_coordinate_range_l807_80710

-- Define the circle C
def CircleC (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- Define the condition MA^2 + MO^2 ≤ 10
def Condition (x y : ℝ) : Prop := (x - 2)^2 + y^2 + x^2 + y^2 ≤ 10

-- Theorem statement
theorem y_coordinate_range :
  ∀ x y : ℝ, CircleC x y → Condition x y →
  -Real.sqrt 7 / 2 ≤ y ∧ y ≤ Real.sqrt 7 / 2 :=
by sorry

end NUMINAMATH_CALUDE_y_coordinate_range_l807_80710


namespace NUMINAMATH_CALUDE_min_value_of_f_l807_80755

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := x^2 + 3*y^2 + 8*x - 6*y + x*y + 22

/-- Theorem stating that the minimum value of f is 3 -/
theorem min_value_of_f :
  ∃ (min : ℝ), min = 3 ∧ ∀ (x y : ℝ), f x y ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l807_80755


namespace NUMINAMATH_CALUDE_dice_game_probability_l807_80703

def score (roll1 roll2 : Nat) : Nat := max roll1 roll2

def is_favorable (roll1 roll2 : Nat) : Bool :=
  score roll1 roll2 ≤ 3

def total_outcomes : Nat := 36

def favorable_outcomes : Nat := 9

theorem dice_game_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_dice_game_probability_l807_80703
