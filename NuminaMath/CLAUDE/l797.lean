import Mathlib

namespace NUMINAMATH_CALUDE_mandy_jackson_age_difference_l797_79799

/-- Proves that Mandy is 10 years older than Jackson given the conditions of the problem -/
theorem mandy_jackson_age_difference :
  ∀ (mandy_age jackson_age adele_age : ℕ),
    jackson_age = 20 →
    adele_age = (3 * jackson_age) / 4 →
    mandy_age + jackson_age + adele_age + 30 = 95 →
    mandy_age > jackson_age →
    mandy_age - jackson_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_mandy_jackson_age_difference_l797_79799


namespace NUMINAMATH_CALUDE_square_table_capacity_square_table_capacity_proof_l797_79766

theorem square_table_capacity (rectangular_tables : ℕ) (rectangular_capacity : ℕ) 
  (square_tables : ℕ) (total_pupils : ℕ) : ℕ :=
  let remaining_pupils := total_pupils - rectangular_tables * rectangular_capacity
  remaining_pupils / square_tables

#check square_table_capacity 7 10 5 90 = 4

theorem square_table_capacity_proof 
  (h1 : rectangular_tables = 7)
  (h2 : rectangular_capacity = 10)
  (h3 : square_tables = 5)
  (h4 : total_pupils = 90) :
  square_table_capacity rectangular_tables rectangular_capacity square_tables total_pupils = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_table_capacity_square_table_capacity_proof_l797_79766


namespace NUMINAMATH_CALUDE_work_completion_time_l797_79701

theorem work_completion_time 
  (total_time : ℝ) 
  (joint_work_time : ℝ) 
  (remaining_work_time : ℝ) 
  (h1 : total_time = 24) 
  (h2 : joint_work_time = 16) 
  (h3 : remaining_work_time = 16) : 
  (total_time * remaining_work_time) / (total_time - joint_work_time) = 48 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l797_79701


namespace NUMINAMATH_CALUDE_quadratic_properties_l797_79784

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  pass_through_minus_one : a * (-1)^2 + b * (-1) + c = 0
  pass_through_zero : c = -1.5
  pass_through_one : a + b + c = -2
  pass_through_two : 4 * a + 2 * b + c = -1.5

theorem quadratic_properties (f : QuadraticFunction) :
  (∃ a' : ℝ, ∀ x, f.a * x^2 + f.b * x + f.c = a' * (x - 1)^2 - 2) ∧
  (f.a * 0^2 + f.b * 0 + f.c + 1.5 = 0 ∧ f.a * 2^2 + f.b * 2 + f.c + 1.5 = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l797_79784


namespace NUMINAMATH_CALUDE_lollipop_bouquets_l797_79793

theorem lollipop_bouquets (cherry orange raspberry lemon candycane chocolate : ℕ) 
  (h1 : cherry = 4)
  (h2 : orange = 6)
  (h3 : raspberry = 8)
  (h4 : lemon = 10)
  (h5 : candycane = 12)
  (h6 : chocolate = 14) :
  Nat.gcd cherry (Nat.gcd orange (Nat.gcd raspberry (Nat.gcd lemon (Nat.gcd candycane chocolate)))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_lollipop_bouquets_l797_79793


namespace NUMINAMATH_CALUDE_problem_solution_l797_79710

theorem problem_solution :
  (∃ a b c : ℝ, a * c = b * c ∧ a ≠ b) ∧
  (∀ a : ℝ, (¬ ∃ q : ℚ, a + 5 = q) ↔ (¬ ∃ q : ℚ, a = q)) ∧
  ((∀ a b : ℝ, a = b → a^2 = b^2) ∧ (∃ a b : ℝ, a^2 = b^2 ∧ a ≠ b)) ∧
  (∃ x : ℝ, x^2 < 1) :=
by sorry


end NUMINAMATH_CALUDE_problem_solution_l797_79710


namespace NUMINAMATH_CALUDE_quadrangular_pyramid_edge_sum_l797_79700

/-- Represents a hexagonal prism -/
structure HexagonalPrism where
  edge_length : ℝ
  total_edge_length : ℝ
  edges_equal : edge_length > 0
  total_length_constraint : total_edge_length = 18 * edge_length

/-- Represents a quadrangular pyramid -/
structure QuadrangularPyramid where
  edge_length : ℝ
  edges_equal : edge_length > 0

/-- Theorem stating the relationship between hexagonal prism and quadrangular pyramid edge lengths -/
theorem quadrangular_pyramid_edge_sum 
  (h : HexagonalPrism) 
  (q : QuadrangularPyramid) 
  (edge_equality : q.edge_length = h.edge_length) :
  8 * q.edge_length = 36 := by
  sorry

end NUMINAMATH_CALUDE_quadrangular_pyramid_edge_sum_l797_79700


namespace NUMINAMATH_CALUDE_joan_total_cents_l797_79772

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The number of quarters Joan has -/
def num_quarters : ℕ := 12

/-- The number of dimes Joan has -/
def num_dimes : ℕ := 8

/-- The number of nickels Joan has -/
def num_nickels : ℕ := 15

/-- The number of pennies Joan has -/
def num_pennies : ℕ := 25

/-- The total value of Joan's coins in cents -/
theorem joan_total_cents : 
  num_quarters * quarter_value + 
  num_dimes * dime_value + 
  num_nickels * nickel_value + 
  num_pennies * penny_value = 480 := by
  sorry

end NUMINAMATH_CALUDE_joan_total_cents_l797_79772


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l797_79769

theorem cubic_sum_theorem (x y z : ℝ) 
  (h1 : x + y + z = 2) 
  (h2 : x*y + x*z + y*z = -5) 
  (h3 : x*y*z = -6) : 
  x^3 + y^3 + z^3 = 18 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l797_79769


namespace NUMINAMATH_CALUDE_parabola_equation_l797_79707

/-- A parabola with vertex at the origin, axis of symmetry along the x-axis,
    and passing through the point (-2, 2√2) has the equation y^2 = -4x. -/
theorem parabola_equation (p : ℝ × ℝ) 
    (vertex_origin : p.1 = 0 ∧ p.2 = 0)
    (axis_x : ∀ (x y : ℝ), y^2 = -4*x → y^2 = -4*(-x))
    (point_on_parabola : (-2)^2 + (2*Real.sqrt 2)^2 = -4*(-2)) :
  ∀ (x y : ℝ), y^2 = -4*x ↔ (x, y) ∈ {(a, b) | b^2 = -4*a} :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l797_79707


namespace NUMINAMATH_CALUDE_allens_mother_age_l797_79742

-- Define Allen's age as a function of his mother's age
def allen_age (mother_age : ℕ) : ℕ := mother_age - 25

-- Define the condition that in 3 years, the sum of their ages will be 41
def future_age_sum (mother_age : ℕ) : Prop :=
  (mother_age + 3) + (allen_age mother_age + 3) = 41

-- Theorem stating that Allen's mother's present age is 30
theorem allens_mother_age :
  ∃ (mother_age : ℕ), 
    (allen_age mother_age = mother_age - 25) ∧ 
    (future_age_sum mother_age) ∧ 
    (mother_age = 30) := by
  sorry

end NUMINAMATH_CALUDE_allens_mother_age_l797_79742


namespace NUMINAMATH_CALUDE_quadratic_equation_sum_l797_79782

theorem quadratic_equation_sum (p q : ℝ) : 
  (∀ x, 9*x^2 - 36*x - 81 = 0 ↔ (x + p)^2 = q) → p + q = 11 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_sum_l797_79782


namespace NUMINAMATH_CALUDE_f_inequality_solution_l797_79705

noncomputable def f (a x : ℝ) : ℝ := |x - a| - |x + 3|

theorem f_inequality_solution (a : ℝ) :
  (a = -1 → {x : ℝ | f a x ≤ 1} = {x : ℝ | x ≥ -5/2}) ∧
  ({a : ℝ | ∀ x ∈ Set.Icc 0 3, f a x ≤ 4} = Set.Icc (-7) 7) :=
sorry

end NUMINAMATH_CALUDE_f_inequality_solution_l797_79705


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_fraction_l797_79764

/-- Given that z = (a - √2) + ai is a purely imaginary number where a ∈ ℝ,
    prove that (a + i⁷) / (1 + ai) = -i -/
theorem purely_imaginary_complex_fraction (a : ℝ) :
  (a - Real.sqrt 2 : ℂ) + a * I = (0 : ℂ) →
  (a + I^7) / (1 + a * I) = -I := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_fraction_l797_79764


namespace NUMINAMATH_CALUDE_inscribed_squares_product_l797_79752

theorem inscribed_squares_product (a b : ℝ) : 
  (∃ small_square large_square : ℝ → ℝ → Prop,
    (∀ x y, small_square x y → x^2 + y^2 ≤ 9) ∧
    (∀ x y, large_square x y → x^2 + y^2 ≤ 16) ∧
    (∀ x y, small_square x y → ∃ u v, large_square u v ∧ 
      ((x = u ∧ y ∈ [0, 4]) ∨ (x ∈ [0, 4] ∧ y = v) ∨ 
       (x = -u ∧ y ∈ [0, 4]) ∨ (x ∈ [0, 4] ∧ y = -v))) ∧
    (a + b = 4) ∧
    (a^2 + b^2 = 18)) →
  a * b = -1 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_squares_product_l797_79752


namespace NUMINAMATH_CALUDE_unique_solution_condition_l797_79718

theorem unique_solution_condition (s : ℝ) : 
  (∃! x : ℝ, (s * x - 3) / (x + 1) = x) ↔ (s = 1 + 2 * Real.sqrt 3 ∨ s = 1 - 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l797_79718


namespace NUMINAMATH_CALUDE_abs_neg_2023_l797_79743

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2023_l797_79743


namespace NUMINAMATH_CALUDE_last_two_digits_factorial_sum_l797_79717

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_factorial_sum :
  last_two_digits (sum_factorials 15) = last_two_digits (sum_factorials 9) :=
sorry

end NUMINAMATH_CALUDE_last_two_digits_factorial_sum_l797_79717


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_arithmetic_sequence_sum_l797_79719

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of terms with indices that add up to the same value is constant -/
theorem arithmetic_sequence_sum_property (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  ∀ i j k l : ℕ, i + l = j + k → a i + a l = a j + a k :=
sorry

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) 
  (h_sum : a 3 + a 7 = 37) : a 2 + a 4 + a 6 + a 8 = 74 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_arithmetic_sequence_sum_l797_79719


namespace NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l797_79714

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x + 1)^2 + (y - 2)^2 = 4

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-1, 2)

-- Theorem statement
theorem circle_tangent_to_x_axis :
  -- The circle equation represents a circle with the given center
  (∀ x y : ℝ, circle_equation x y ↔ ((x - circle_center.1)^2 + (y - circle_center.2)^2 = 4)) ∧
  -- The circle is tangent to the x-axis
  (∃ x : ℝ, circle_equation x 0 ∧ ∀ y : ℝ, y ≠ 0 → ¬ circle_equation x y) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l797_79714


namespace NUMINAMATH_CALUDE_georgia_black_buttons_l797_79740

theorem georgia_black_buttons
  (yellow_buttons : Nat)
  (green_buttons : Nat)
  (buttons_given : Nat)
  (buttons_left : Nat)
  (h1 : yellow_buttons = 4)
  (h2 : green_buttons = 3)
  (h3 : buttons_given = 4)
  (h4 : buttons_left = 5) :
  ∃ (black_buttons : Nat), black_buttons = 2 ∧
    yellow_buttons + black_buttons + green_buttons = buttons_left + buttons_given :=
by sorry

end NUMINAMATH_CALUDE_georgia_black_buttons_l797_79740


namespace NUMINAMATH_CALUDE_visitors_previous_day_l797_79750

/-- The number of visitors to Buckingham Palace over 25 days -/
def total_visitors : ℕ := 949

/-- The number of days over which visitors were counted -/
def total_days : ℕ := 25

/-- The number of visitors on the previous day -/
def previous_day_visitors : ℕ := 246

/-- Theorem stating that the number of visitors on the previous day was 246 -/
theorem visitors_previous_day : previous_day_visitors = 246 := by
  sorry

end NUMINAMATH_CALUDE_visitors_previous_day_l797_79750


namespace NUMINAMATH_CALUDE_recipe_pancakes_l797_79780

/-- The number of pancakes Bobby ate -/
def bobby_pancakes : ℕ := 5

/-- The number of pancakes Bobby's dog ate -/
def dog_pancakes : ℕ := 7

/-- The number of pancakes left -/
def leftover_pancakes : ℕ := 9

/-- The total number of pancakes made by the recipe -/
def total_pancakes : ℕ := bobby_pancakes + dog_pancakes + leftover_pancakes

theorem recipe_pancakes : total_pancakes = 21 := by
  sorry

end NUMINAMATH_CALUDE_recipe_pancakes_l797_79780


namespace NUMINAMATH_CALUDE_det_3_4_1_2_l797_79748

-- Define the determinant function for a 2x2 matrix
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Theorem statement
theorem det_3_4_1_2 : det2x2 3 4 1 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_det_3_4_1_2_l797_79748


namespace NUMINAMATH_CALUDE_ohara_triple_81_49_l797_79720

/-- O'Hara triple definition -/
def is_ohara_triple (a b x : ℕ) : Prop := Real.sqrt a + Real.sqrt b = x

/-- The main theorem -/
theorem ohara_triple_81_49 (x : ℕ) :
  is_ohara_triple 81 49 x → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_ohara_triple_81_49_l797_79720


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l797_79796

theorem arithmetic_expression_evaluation : 2 + 3 * 4^2 - 5 + 6 = 51 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l797_79796


namespace NUMINAMATH_CALUDE_circle_tangent_implies_m_equals_9_l797_79738

/-- Circle C with equation x^2 + y^2 - 6x - 8y + m = 0 -/
def circle_C (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 8*y + m = 0

/-- Unit circle with equation x^2 + y^2 = 1 -/
def unit_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (center1 center2 : ℝ × ℝ) (radius1 radius2 : ℝ) : Prop :=
  (center1.1 - center2.1)^2 + (center1.2 - center2.2)^2 = (radius1 + radius2)^2

/-- Main theorem: If circle C is externally tangent to the unit circle, then m = 9 -/
theorem circle_tangent_implies_m_equals_9 (m : ℝ) :
  (∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ x y, circle_C m x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    externally_tangent center (0, 0) radius 1) →
  m = 9 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_implies_m_equals_9_l797_79738


namespace NUMINAMATH_CALUDE_triangle_side_range_l797_79731

theorem triangle_side_range (A B C : ℝ × ℝ) (x : ℝ) :
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  AB = 16 ∧ AC = 7 ∧ BC = x →
  9 < x ∧ x < 23 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l797_79731


namespace NUMINAMATH_CALUDE_isosceles_triangle_rectangle_equal_area_l797_79790

/-- Given an isosceles triangle and a rectangle with equal areas,
    where the rectangle's length is twice its width and
    the triangle's base equals the rectangle's width,
    prove that the triangle's height is four times the rectangle's width. -/
theorem isosceles_triangle_rectangle_equal_area
  (w h : ℝ) -- w: width of rectangle, h: height of triangle
  (hw : w > 0) -- assume width is positive
  (triangle_area : ℝ → ℝ → ℝ) -- area function for triangle
  (rectangle_area : ℝ → ℝ → ℝ) -- area function for rectangle
  (h_triangle_area : triangle_area w h = 1/2 * w * h) -- definition of triangle area
  (h_rectangle_area : rectangle_area w (2*w) = 2 * w^2) -- definition of rectangle area
  (h_equal_area : triangle_area w h = rectangle_area w (2*w)) -- areas are equal
  : h = 4 * w :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_rectangle_equal_area_l797_79790


namespace NUMINAMATH_CALUDE_chess_program_ratio_l797_79733

theorem chess_program_ratio (total_students : ℕ) (chess_students : ℕ) (tournament_students : ℕ) 
  (h1 : total_students = 24)
  (h2 : tournament_students = 4)
  (h3 : chess_students = 2 * tournament_students)
  : (chess_students : ℚ) / total_students = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_chess_program_ratio_l797_79733


namespace NUMINAMATH_CALUDE_product_A_sample_size_l797_79704

/-- Represents the ratio of quantities for products A, B, and C -/
def productRatio : Fin 3 → ℕ
| 0 => 2  -- Product A
| 1 => 3  -- Product B
| 2 => 5  -- Product C
| _ => 0  -- Unreachable case

/-- The total sample size -/
def sampleSize : ℕ := 80

/-- Calculates the number of items for a given product in the sample -/
def itemsInSample (product : Fin 3) : ℕ :=
  (sampleSize * productRatio product) / (productRatio 0 + productRatio 1 + productRatio 2)

theorem product_A_sample_size :
  itemsInSample 0 = 16 := by sorry

end NUMINAMATH_CALUDE_product_A_sample_size_l797_79704


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l797_79794

theorem lcm_gcd_product (a b : ℕ) (ha : a = 30) (hb : b = 75) :
  Nat.lcm a b * Nat.gcd a b = 2250 ∧ Nat.lcm a b * Nat.gcd a b = a * b := by
  sorry

#check lcm_gcd_product

end NUMINAMATH_CALUDE_lcm_gcd_product_l797_79794


namespace NUMINAMATH_CALUDE_equation_proof_l797_79776

theorem equation_proof : (60 / 20) * (60 / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l797_79776


namespace NUMINAMATH_CALUDE_roots_product_l797_79746

theorem roots_product (a b c d : ℝ) : 
  (a^2 + 68*a + 1 = 0) →
  (b^2 + 68*b + 1 = 0) →
  (c^2 - 86*c + 1 = 0) →
  (d^2 - 86*d + 1 = 0) →
  (a+c)*(b+c)*(a-d)*(b-d) = 2772 := by
  sorry

end NUMINAMATH_CALUDE_roots_product_l797_79746


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l797_79791

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n, prove S_8 = 80 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2) →  -- Definition of S_n
  S 4 = 24 →                                                      -- Given condition
  a 8 = 17 →                                                      -- Given condition
  S 8 = 80 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l797_79791


namespace NUMINAMATH_CALUDE_tan_function_property_l797_79728

/-- 
Given positive constants a and b, if the function y = a * tan(b * x) 
has a period of π/2 and passes through the point (π/8, 1), then ab = 2.
-/
theorem tan_function_property (a b : ℝ) : 
  a > 0 → b > 0 → 
  (π / b = π / 2) → 
  (a * Real.tan (b * π / 8) = 1) → 
  a * b = 2 := by
sorry

end NUMINAMATH_CALUDE_tan_function_property_l797_79728


namespace NUMINAMATH_CALUDE_volume_is_1250_l797_79703

/-- The volume of the solid bounded by the given surfaces -/
def volume_of_solid : ℝ :=
  let surface1 := {(x, y, z) : ℝ × ℝ × ℝ | x^2 / 27 + y^2 / 25 = 1}
  let surface2 := {(x, y, z) : ℝ × ℝ × ℝ | z = y / Real.sqrt 3}
  let surface3 := {(x, y, z) : ℝ × ℝ × ℝ | z = 0}
  let constraint := {(x, y, z) : ℝ × ℝ × ℝ | y ≥ 0}
  1250 -- placeholder for the actual volume

/-- Theorem stating that the volume of the solid is 1250 -/
theorem volume_is_1250 : volume_of_solid = 1250 := by
  sorry

end NUMINAMATH_CALUDE_volume_is_1250_l797_79703


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l797_79747

theorem solution_set_of_inequality (x : ℝ) :
  (x - 2) / (1 - x) > 0 ↔ 1 < x ∧ x < 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l797_79747


namespace NUMINAMATH_CALUDE_chess_tournament_red_pairs_l797_79760

/-- Represents the number of pairs in a chess tournament where both players wear red hats. -/
def red_red_pairs (green_players : ℕ) (red_players : ℕ) (total_pairs : ℕ) (green_green_pairs : ℕ) : ℕ :=
  (red_players - (total_pairs * 2 - green_players - red_players)) / 2

/-- Theorem stating that in the given chess tournament scenario, there are 27 pairs where both players wear red hats. -/
theorem chess_tournament_red_pairs : 
  red_red_pairs 64 68 66 25 = 27 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_red_pairs_l797_79760


namespace NUMINAMATH_CALUDE_office_employees_l797_79736

theorem office_employees (total_employees : ℕ) 
  (h1 : (45 : ℚ) / 100 * total_employees = total_males)
  (h2 : (50 : ℚ) / 100 * total_males = males_50_and_above)
  (h3 : 1170 = total_males - males_50_and_above) :
  total_employees = 5200 :=
by sorry

end NUMINAMATH_CALUDE_office_employees_l797_79736


namespace NUMINAMATH_CALUDE_trig_identity_l797_79713

theorem trig_identity (x : ℝ) (h : Real.sin (x + π/6) = 1/4) :
  Real.sin (5*π/6 - x) + (Real.sin (π/3 - x))^2 = 19/16 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l797_79713


namespace NUMINAMATH_CALUDE_units_digit_of_17_pow_2041_l797_79777

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the main theorem
theorem units_digit_of_17_pow_2041 : unitsDigit (17^2041) = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_17_pow_2041_l797_79777


namespace NUMINAMATH_CALUDE_congruence_solutions_count_l797_79788

theorem congruence_solutions_count : 
  (Finset.filter (fun x : ℕ => 
    x > 0 ∧ x < 150 ∧ (x + 20) % 45 = 75 % 45) 
    (Finset.range 150)).card = 4 := by sorry

end NUMINAMATH_CALUDE_congruence_solutions_count_l797_79788


namespace NUMINAMATH_CALUDE_no_two_digit_factors_of_1729_l797_79749

theorem no_two_digit_factors_of_1729 : 
  ¬ ∃ (a b : ℕ), 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ a * b = 1729 := by
  sorry

end NUMINAMATH_CALUDE_no_two_digit_factors_of_1729_l797_79749


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l797_79771

theorem decimal_to_fraction :
  (2.35 : ℚ) = 47 / 20 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l797_79771


namespace NUMINAMATH_CALUDE_triangle_to_hexagon_area_ratio_l797_79729

/-- A regular hexagon with an inscribed equilateral triangle -/
structure RegularHexagonWithTriangle where
  -- The area of the regular hexagon
  hexagon_area : ℝ
  -- The area of the inscribed equilateral triangle
  triangle_area : ℝ

/-- The ratio of the inscribed triangle's area to the hexagon's area is 1/6 -/
theorem triangle_to_hexagon_area_ratio 
  (hex : RegularHexagonWithTriangle) : 
  hex.triangle_area / hex.hexagon_area = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_to_hexagon_area_ratio_l797_79729


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l797_79767

theorem smallest_n_for_inequality : ∀ n : ℤ, (5 + 3 * n > 300) ↔ (n ≥ 99) := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l797_79767


namespace NUMINAMATH_CALUDE_no_rectangle_with_given_cuts_l797_79783

theorem no_rectangle_with_given_cuts : ¬ ∃ (w h : ℕ), 
  (w * h = 37 + 135 * 3) ∧ 
  (w ≥ 2 ∧ h ≥ 2) ∧
  (w * h - 37 ≥ 135 * 3) :=
sorry

end NUMINAMATH_CALUDE_no_rectangle_with_given_cuts_l797_79783


namespace NUMINAMATH_CALUDE_foci_of_hyperbola_l797_79778

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 7 - y^2 / 3 = 1

/-- The foci coordinates -/
def foci_coordinates : Set (ℝ × ℝ) := {(Real.sqrt 10, 0), (-Real.sqrt 10, 0)}

/-- Theorem: The given coordinates are the foci of the hyperbola -/
theorem foci_of_hyperbola :
  ∀ (x y : ℝ), hyperbola_equation x y → (x, y) ∈ foci_coordinates ↔ 
    ∃ (x' y' : ℝ), hyperbola_equation x' y' ∧ 
      (x - x')^2 + (y - y')^2 = ((Real.sqrt 10 + x')^2 + y'^2).sqrt * 
                                ((Real.sqrt 10 - x')^2 + y'^2).sqrt :=
sorry

end NUMINAMATH_CALUDE_foci_of_hyperbola_l797_79778


namespace NUMINAMATH_CALUDE_complex_functional_equation_l797_79763

theorem complex_functional_equation 
  (f : ℂ → ℂ) 
  (h : ∀ z : ℂ, f z + z * f (1 - z) = 1 + z) : 
  ∀ w : ℂ, f w = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_functional_equation_l797_79763


namespace NUMINAMATH_CALUDE_total_expenses_calculation_l797_79787

-- Define the initial conditions
def initial_price : ℝ := 1.4
def daily_price_decrease : ℝ := 0.1
def first_purchase : ℝ := 10
def second_purchase : ℝ := 25
def total_trip_distance : ℝ := 320
def distance_before_friday : ℝ := 200
def fuel_efficiency : ℝ := 8

-- Define the theorem
theorem total_expenses_calculation :
  let friday_price := initial_price - 4 * daily_price_decrease
  let cost_monday := first_purchase * initial_price
  let cost_friday := second_purchase * friday_price
  let total_cost_35_liters := cost_monday + cost_friday
  let remaining_distance := total_trip_distance - distance_before_friday
  let additional_liters := remaining_distance / fuel_efficiency
  let cost_additional_liters := additional_liters * friday_price
  let total_expenses := total_cost_35_liters + cost_additional_liters
  total_expenses = 54 := by sorry

end NUMINAMATH_CALUDE_total_expenses_calculation_l797_79787


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l797_79774

/-- The distance between the foci of an ellipse defined by 25x^2 - 100x + 4y^2 + 8y + 36 = 0 -/
theorem ellipse_foci_distance : 
  let ellipse_eq := fun (x y : ℝ) => 25 * x^2 - 100 * x + 4 * y^2 + 8 * y + 36
  ∃ (h k a b : ℝ), 
    (∀ x y, ellipse_eq x y = 0 ↔ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) ∧
    2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 14.28 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l797_79774


namespace NUMINAMATH_CALUDE_dividend_calculation_l797_79779

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 10 * quotient)
  (h2 : divisor = 5 * remainder)
  (h3 : remainder = 46) :
  divisor * quotient + remainder = 5336 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l797_79779


namespace NUMINAMATH_CALUDE_dyck_path_correspondence_l797_79708

/-- A Dyck path is a lattice path of upsteps and downsteps that starts at the origin and never dips below the x-axis. -/
def DyckPath (n : ℕ) : Type := sorry

/-- A return in a Dyck path is a maximal sequence of contiguous downsteps that terminates on the x-axis. -/
def Return (path : DyckPath n) : Type := sorry

/-- Predicate to check if a return has even length -/
def hasEvenLengthReturn (path : DyckPath n) : Prop := sorry

/-- The number of Dyck n-paths -/
def numDyckPaths (n : ℕ) : ℕ := sorry

/-- The number of Dyck n-paths with no return of even length -/
def numDyckPathsNoEvenReturn (n : ℕ) : ℕ := sorry

/-- Theorem: The number of Dyck n-paths with no return of even length is equal to the number of Dyck (n-1) paths -/
theorem dyck_path_correspondence (n : ℕ) (h : n ≥ 1) :
  numDyckPathsNoEvenReturn n = numDyckPaths (n - 1) := by sorry

end NUMINAMATH_CALUDE_dyck_path_correspondence_l797_79708


namespace NUMINAMATH_CALUDE_divisible_by_48_l797_79797

theorem divisible_by_48 (n : ℕ) (h : Even n) : ∃ k : ℤ, (n^3 : ℤ) + 20*n = 48*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_48_l797_79797


namespace NUMINAMATH_CALUDE_airplane_fraction_is_one_third_l797_79755

/-- Represents the travel scenario with given conditions -/
structure TravelScenario where
  driving_time : ℕ
  airport_drive_time : ℕ
  airport_wait_time : ℕ
  post_flight_time : ℕ
  time_saved : ℕ

/-- Calculates the fraction of time spent on the airplane compared to driving -/
def airplane_time_fraction (scenario : TravelScenario) : ℚ :=
  let airplane_time := scenario.driving_time - scenario.airport_drive_time - 
                       scenario.airport_wait_time - scenario.post_flight_time - 
                       scenario.time_saved
  airplane_time / scenario.driving_time

/-- The main theorem stating that the fraction of time spent on the airplane is 1/3 -/
theorem airplane_fraction_is_one_third (scenario : TravelScenario) 
    (h1 : scenario.driving_time = 195)
    (h2 : scenario.airport_drive_time = 10)
    (h3 : scenario.airport_wait_time = 20)
    (h4 : scenario.post_flight_time = 10)
    (h5 : scenario.time_saved = 90) :
    airplane_time_fraction scenario = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_airplane_fraction_is_one_third_l797_79755


namespace NUMINAMATH_CALUDE_sum_of_digits_double_permutation_l797_79762

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Permutation relation between natural numbers -/
def isPermutationOf (a b : ℕ) : Prop := sorry

theorem sum_of_digits_double_permutation (A B : ℕ) 
  (h : isPermutationOf A B) : 
  sumOfDigits (2 * A) = sumOfDigits (2 * B) := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_double_permutation_l797_79762


namespace NUMINAMATH_CALUDE_largest_number_proof_l797_79739

theorem largest_number_proof (a b c d e : ℝ) 
  (ha : a = 0.997) (hb : b = 0.979) (hc : c = 0.99) (hd : d = 0.9709) (he : e = 0.999) :
  e = max a (max b (max c (max d e))) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_proof_l797_79739


namespace NUMINAMATH_CALUDE_solve_equation_l797_79798

theorem solve_equation : ∃ x : ℝ, 35 - (23 - (15 - x)) = 12 * 2 / (1 / 2) ∧ x = -21 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l797_79798


namespace NUMINAMATH_CALUDE_income_calculation_l797_79775

theorem income_calculation (a b c d e : ℝ) : 
  (a + b) / 2 = 4050 →
  (b + c) / 2 = 5250 →
  (a + c) / 2 = 4200 →
  (a + b + d) / 3 = 4800 →
  (c + d + e) / 3 = 6000 →
  (b + a + e) / 3 = 4500 →
  a = 3000 ∧ b = 5100 ∧ c = 5400 ∧ d = 6300 ∧ e = 5400 :=
by sorry

end NUMINAMATH_CALUDE_income_calculation_l797_79775


namespace NUMINAMATH_CALUDE_hyperbola_equation_l797_79765

def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def asymptote (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x

def parabola (x y : ℝ) : Prop :=
  y^2 = 24 * x

def directrix (x : ℝ) : Prop :=
  x = -6

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ x y : ℝ, hyperbola a b x y ∧ asymptote x y) →
  (∃ x : ℝ, directrix x ∧ ∃ y : ℝ, hyperbola a b x y) →
  (∀ x y : ℝ, hyperbola a b x y ↔ hyperbola 3 (Real.sqrt 27) x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l797_79765


namespace NUMINAMATH_CALUDE_certain_number_proof_l797_79741

theorem certain_number_proof (n m : ℕ+) 
  (h1 : Nat.lcm n m = 48)
  (h2 : Nat.gcd n m = 8)
  (h3 : n = 24) :
  m = 16 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l797_79741


namespace NUMINAMATH_CALUDE_expand_product_l797_79795

theorem expand_product (x : ℝ) : (5 * x + 7) * (3 * x^2 + 2 * x + 4) = 15 * x^3 + 31 * x^2 + 34 * x + 28 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l797_79795


namespace NUMINAMATH_CALUDE_triangle_centers_l797_79725

/-- Triangle XYZ with side lengths x, y, z -/
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Incenter coordinates (a, b, c) -/
structure Incenter where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_one : a + b + c = 1

/-- Centroid coordinates (p, q, r) -/
structure Centroid where
  p : ℝ
  q : ℝ
  r : ℝ
  sum_one : p + q + r = 1

/-- The theorem to be proved -/
theorem triangle_centers (t : Triangle) (i : Incenter) (c : Centroid) :
  t.x = 13 ∧ t.y = 15 ∧ t.z = 6 →
  i.a = 13/34 ∧ i.b = 15/34 ∧ i.c = 6/34 ∧
  c.p = 1/3 ∧ c.q = 1/3 ∧ c.r = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_centers_l797_79725


namespace NUMINAMATH_CALUDE_cube_surface_area_l797_79773

/-- The surface area of a cube with edge length 8 cm is 384 square centimeters. -/
theorem cube_surface_area : 
  let edge_length : ℝ := 8
  let face_area : ℝ := edge_length ^ 2
  let surface_area : ℝ := 6 * face_area
  surface_area = 384 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l797_79773


namespace NUMINAMATH_CALUDE_combined_surface_area_theorem_l797_79756

/-- Represents a cube with a given edge length -/
structure Cube where
  edgeLength : ℝ

/-- Represents the combined shape of two cubes -/
structure CombinedShape where
  largerCube : Cube
  smallerCube : Cube

/-- Calculates the surface area of a cube -/
def surfaceArea (c : Cube) : ℝ := 6 * c.edgeLength^2

/-- Calculates the surface area of the combined shape -/
def combinedSurfaceArea (cs : CombinedShape) : ℝ :=
  surfaceArea cs.largerCube + surfaceArea cs.smallerCube - 4 * cs.smallerCube.edgeLength^2

/-- The main theorem stating the surface area of the combined shape -/
theorem combined_surface_area_theorem (cs : CombinedShape) 
  (h1 : cs.largerCube.edgeLength = 2)
  (h2 : cs.smallerCube.edgeLength = cs.largerCube.edgeLength / 2) :
  combinedSurfaceArea cs = 32 := by
  sorry

#check combined_surface_area_theorem

end NUMINAMATH_CALUDE_combined_surface_area_theorem_l797_79756


namespace NUMINAMATH_CALUDE_number_equation_l797_79786

theorem number_equation (x : ℝ) : 38 + 2 * x = 124 ↔ x = 43 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l797_79786


namespace NUMINAMATH_CALUDE_five_g_growth_equation_l797_79745

theorem five_g_growth_equation (initial_users : ℕ) (target_users : ℕ) (x : ℝ) :
  initial_users = 30000 →
  target_users = 76800 →
  initial_users * (1 + x)^2 = target_users →
  3 * (1 + x)^2 = 7.68 :=
by sorry

end NUMINAMATH_CALUDE_five_g_growth_equation_l797_79745


namespace NUMINAMATH_CALUDE_fifth_term_value_l797_79727

/-- Given a sequence {aₙ} with sum of first n terms Sₙ = 2n(n+1), prove a₅ = 20 -/
theorem fifth_term_value (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h : ∀ n : ℕ, S n = 2 * n * (n + 1)) : 
  a 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_value_l797_79727


namespace NUMINAMATH_CALUDE_cubic_equation_root_b_value_l797_79712

theorem cubic_equation_root_b_value :
  ∀ (a b : ℚ),
  (∃ (x : ℝ), x = 2 + Real.sqrt 3 ∧ x^3 + a*x^2 + b*x + 10 = 0) →
  b = -39 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_root_b_value_l797_79712


namespace NUMINAMATH_CALUDE_tetrahedron_cut_vertices_l797_79758

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  vertices : Finset (Fin 4)

/-- The result of cutting off a vertex from a polyhedron -/
def cutVertex (p : RegularTetrahedron) (v : Fin 4) : ℕ := 3

/-- The number of vertices in the shape resulting from cutting off all vertices of a regular tetrahedron -/
def verticesAfterCutting (t : RegularTetrahedron) : ℕ :=
  t.vertices.sum (λ v => cutVertex t v)

/-- Theorem: Cutting off all vertices of a regular tetrahedron results in a shape with 12 vertices -/
theorem tetrahedron_cut_vertices (t : RegularTetrahedron) :
  verticesAfterCutting t = 12 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_cut_vertices_l797_79758


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l797_79768

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  (3 * a 1 - 2 * a 2 = (1/2) * a 3 - 2 * a 2) →
  (a 20 + a 19) / (a 18 + a 17) = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l797_79768


namespace NUMINAMATH_CALUDE_bicycle_store_promotion_correct_l797_79723

/-- Represents the promotion rules and sales data for a bicycle store. -/
structure BicycleStore where
  single_clamps : ℕ  -- Number of clamps given for a single bicycle purchase
  single_helmet : ℕ  -- Number of helmets given for a single bicycle purchase
  discount_rate : ℚ  -- Discount rate on the 3rd bicycle for a 3-bicycle purchase
  morning_single : ℕ  -- Number of single bicycle purchases in the morning
  morning_triple : ℕ  -- Number of 3-bicycle purchases in the morning
  afternoon_single : ℕ  -- Number of single bicycle purchases in the afternoon
  afternoon_triple : ℕ  -- Number of 3-bicycle purchases in the afternoon

/-- Calculates the total number of bike clamps given away. -/
def total_clamps (store : BicycleStore) : ℕ :=
  (store.morning_single + store.afternoon_single) * store.single_clamps +
  (store.morning_triple + store.afternoon_triple) * store.single_clamps

/-- Calculates the total number of helmets given away. -/
def total_helmets (store : BicycleStore) : ℕ :=
  (store.morning_single + store.afternoon_single) * store.single_helmet +
  (store.morning_triple + store.afternoon_triple) * store.single_helmet

/-- Calculates the overall discount value in terms of full-price bicycles. -/
def discount_value (store : BicycleStore) : ℚ :=
  (store.morning_triple + store.afternoon_triple) * store.discount_rate

/-- Theorem stating the correctness of the calculations based on the given data. -/
theorem bicycle_store_promotion_correct (store : BicycleStore) 
  (h1 : store.single_clamps = 2)
  (h2 : store.single_helmet = 1)
  (h3 : store.discount_rate = 1/5)
  (h4 : store.morning_single = 12)
  (h5 : store.morning_triple = 7)
  (h6 : store.afternoon_single = 24)
  (h7 : store.afternoon_triple = 3) :
  total_clamps store = 92 ∧ 
  total_helmets store = 46 ∧ 
  discount_value store = 2 := by
  sorry


end NUMINAMATH_CALUDE_bicycle_store_promotion_correct_l797_79723


namespace NUMINAMATH_CALUDE_point_on_line_l797_79721

theorem point_on_line (m n : ℝ) : 
  (m = n / 6 - 2 / 5) ∧ (m + p = (n + 18) / 6 - 2 / 5) → p = 3 :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_l797_79721


namespace NUMINAMATH_CALUDE_rain_in_first_hour_l797_79754

theorem rain_in_first_hour (first_hour : ℝ) (second_hour : ℝ) : 
  second_hour = 2 * first_hour + 7 →
  first_hour + second_hour = 22 →
  first_hour = 5 := by sorry

end NUMINAMATH_CALUDE_rain_in_first_hour_l797_79754


namespace NUMINAMATH_CALUDE_scientific_notation_43000000_l797_79734

theorem scientific_notation_43000000 :
  (43000000 : ℝ) = 4.3 * (10 : ℝ)^7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_43000000_l797_79734


namespace NUMINAMATH_CALUDE_unshaded_area_square_with_circles_l797_79770

/-- The area of the unshaded region in a square with three-quarter circles at corners -/
theorem unshaded_area_square_with_circles (side_length : ℝ) (h : side_length = 12) :
  let radius : ℝ := side_length / 4
  let square_area : ℝ := side_length ^ 2
  let circle_area : ℝ := π * radius ^ 2
  let total_circle_area : ℝ := 4 * (3 / 4) * circle_area
  square_area - total_circle_area = 144 - 27 * π :=
by sorry

end NUMINAMATH_CALUDE_unshaded_area_square_with_circles_l797_79770


namespace NUMINAMATH_CALUDE_complex_modulus_l797_79751

theorem complex_modulus (z : ℂ) : z = (1 + I) / (2 - I) → Complex.abs z = Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l797_79751


namespace NUMINAMATH_CALUDE_problem_solution_l797_79716

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then 2 * x^2 else a^x - 1

theorem problem_solution (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : Monotone (f a)) 
  (h4 : f a a = 5 * a - 2) : 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l797_79716


namespace NUMINAMATH_CALUDE_fraction_equality_l797_79709

theorem fraction_equality (P Q M N X : ℚ) 
  (hM : M = 0.4 * Q)
  (hQ : Q = 0.3 * P)
  (hN : N = 0.6 * P)
  (hX : X = 0.25 * M)
  (hP : P ≠ 0) : 
  X / N = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l797_79709


namespace NUMINAMATH_CALUDE_product_mod_23_is_zero_l797_79732

theorem product_mod_23_is_zero :
  (3001 * 3002 * 3003 * 3004 * 3005) % 23 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_23_is_zero_l797_79732


namespace NUMINAMATH_CALUDE_salary_problem_l797_79706

theorem salary_problem (A_salary B_salary : ℝ) 
  (h1 : A_salary = 4500)
  (h2 : A_salary * 0.05 = B_salary * 0.15)
  : A_salary + B_salary = 6000 := by
  sorry

end NUMINAMATH_CALUDE_salary_problem_l797_79706


namespace NUMINAMATH_CALUDE_greatest_3digit_base8_divisible_by_7_l797_79702

/-- Converts a base 8 number to decimal --/
def base8ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 8 --/
def decimalToBase8 (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 3-digit base 8 number --/
def isThreeDigitBase8 (n : ℕ) : Prop := 
  100 ≤ n ∧ n ≤ 777

theorem greatest_3digit_base8_divisible_by_7 :
  ∃ (n : ℕ), isThreeDigitBase8 n ∧ 
             n % 7 = 0 ∧
             base8ToDecimal n = 511 ∧
             decimalToBase8 511 = 777 ∧
             ∀ (m : ℕ), isThreeDigitBase8 m ∧ m % 7 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_3digit_base8_divisible_by_7_l797_79702


namespace NUMINAMATH_CALUDE_poultry_farm_hens_l797_79744

theorem poultry_farm_hens (total_chickens : ℕ) (hen_rooster_ratio : ℚ) (chicks_per_hen : ℕ) : 
  total_chickens = 76 → 
  hen_rooster_ratio = 3 → 
  chicks_per_hen = 5 → 
  ∃ (num_hens : ℕ), num_hens = 12 ∧ 
    num_hens + (num_hens : ℚ) / hen_rooster_ratio + (num_hens * chicks_per_hen) = total_chickens := by
  sorry

end NUMINAMATH_CALUDE_poultry_farm_hens_l797_79744


namespace NUMINAMATH_CALUDE_shoes_outside_library_l797_79730

/-- The total number of shoes outside the library -/
def total_shoes (regular_shoes sandals slippers : ℕ) : ℕ :=
  2 * regular_shoes + 2 * sandals + 2 * slippers

/-- Proof that the total number of shoes is 20 -/
theorem shoes_outside_library :
  let total_people : ℕ := 10
  let regular_shoe_wearers : ℕ := 4
  let sandal_wearers : ℕ := 3
  let slipper_wearers : ℕ := 3
  total_people = regular_shoe_wearers + sandal_wearers + slipper_wearers →
  total_shoes regular_shoe_wearers sandal_wearers slipper_wearers = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_shoes_outside_library_l797_79730


namespace NUMINAMATH_CALUDE_hannah_total_cost_l797_79711

/-- The total cost of Hannah's purchase of sweatshirts and T-shirts -/
def total_cost (num_sweatshirts num_tshirts sweatshirt_price tshirt_price : ℕ) : ℕ :=
  num_sweatshirts * sweatshirt_price + num_tshirts * tshirt_price

/-- Theorem stating that Hannah's total cost is $65 -/
theorem hannah_total_cost :
  total_cost 3 2 15 10 = 65 := by
  sorry

end NUMINAMATH_CALUDE_hannah_total_cost_l797_79711


namespace NUMINAMATH_CALUDE_three_lines_not_necessarily_coplanar_l797_79789

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a line in 3D space
structure Line3D where
  point : Point3D
  direction : Point3D

-- Define a plane in 3D space
structure Plane3D where
  point : Point3D
  normal : Point3D

-- Function to check if a line lies on a plane
def lineOnPlane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

-- Function to check if a point is on a line
def pointOnLine (pt : Point3D) (l : Line3D) : Prop :=
  sorry

-- Theorem statement
theorem three_lines_not_necessarily_coplanar :
  ∃ (p : Point3D) (l1 l2 l3 : Line3D),
    pointOnLine p l1 ∧ pointOnLine p l2 ∧ pointOnLine p l3 ∧
    ¬∃ (plane : Plane3D), lineOnPlane l1 plane ∧ lineOnPlane l2 plane ∧ lineOnPlane l3 plane :=
sorry

end NUMINAMATH_CALUDE_three_lines_not_necessarily_coplanar_l797_79789


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l797_79715

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x > 2 → x^2 - 3*x + 2 > 0) ∧
  (∃ x : ℝ, x^2 - 3*x + 2 > 0 ∧ ¬(x > 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l797_79715


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l797_79757

theorem parallelepiped_volume 
  (a b c : ℝ) 
  (h1 : Real.sqrt (a^2 + b^2 + c^2) = 13)
  (h2 : Real.sqrt (a^2 + b^2) = 3 * Real.sqrt 17)
  (h3 : Real.sqrt (b^2 + c^2) = 4 * Real.sqrt 10) :
  a * b * c = 144 := by
sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l797_79757


namespace NUMINAMATH_CALUDE_multiply_divide_sqrt_l797_79781

theorem multiply_divide_sqrt (x y : ℝ) : 
  x = 0.7142857142857143 → 
  x ≠ 0 → 
  Real.sqrt ((x * y) / 7) = x → 
  y = 5 := by
sorry

end NUMINAMATH_CALUDE_multiply_divide_sqrt_l797_79781


namespace NUMINAMATH_CALUDE_units_digit_of_8_power_2022_l797_79737

theorem units_digit_of_8_power_2022 : 8^2022 % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_8_power_2022_l797_79737


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_sum_21_l797_79722

theorem greatest_of_three_consecutive_integers_sum_21 :
  ∀ x y z : ℤ, 
    (y = x + 1) → 
    (z = y + 1) → 
    (x + y + z = 21) → 
    (max x (max y z) = 8) :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_sum_21_l797_79722


namespace NUMINAMATH_CALUDE_odd_even_function_sum_l797_79792

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_even_function_sum (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_even : is_even (fun x ↦ f (x + 2))) 
  (h_f1 : f 1 = 1) : 
  f 8 + f 17 = 1 := by sorry

end NUMINAMATH_CALUDE_odd_even_function_sum_l797_79792


namespace NUMINAMATH_CALUDE_farm_ratio_l797_79759

theorem farm_ratio (H C : ℕ) 
  (h1 : (H - 15 : ℚ) / (C + 15 : ℚ) = 17 / 7)
  (h2 : H - 15 = C + 15 + 50) :
  H / C = 5 / 1 := by
  sorry

end NUMINAMATH_CALUDE_farm_ratio_l797_79759


namespace NUMINAMATH_CALUDE_last_two_digits_sum_factorials_14_l797_79735

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_sum_factorials_14 :
  last_two_digits (sum_factorials 14) = last_two_digits 409113 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_factorials_14_l797_79735


namespace NUMINAMATH_CALUDE_sphere_volume_radius_3_l797_79726

/-- The volume of a sphere with radius 3 cm is 36π cm³. -/
theorem sphere_volume_radius_3 :
  let r : ℝ := 3
  let volume := (4 / 3) * Real.pi * r ^ 3
  volume = 36 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_radius_3_l797_79726


namespace NUMINAMATH_CALUDE_sweet_distribution_l797_79761

theorem sweet_distribution (total_sweets : ℕ) (initial_children : ℕ) : 
  (initial_children * 15 = total_sweets) → 
  ((initial_children - 32) * 21 = total_sweets) →
  initial_children = 112 := by
sorry

end NUMINAMATH_CALUDE_sweet_distribution_l797_79761


namespace NUMINAMATH_CALUDE_points_earned_in_level_l797_79724

/-- Calculates the points earned in a video game level -/
theorem points_earned_in_level 
  (points_per_enemy : ℕ) 
  (total_enemies : ℕ) 
  (enemies_not_destroyed : ℕ) : 
  points_per_enemy = 9 →
  total_enemies = 11 →
  enemies_not_destroyed = 3 →
  (total_enemies - enemies_not_destroyed) * points_per_enemy = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_points_earned_in_level_l797_79724


namespace NUMINAMATH_CALUDE_exists_steps_for_1001_free_ends_l797_79753

/-- Represents the number of free ends after k steps of construction -/
def free_ends (k : ℕ) : ℕ := 4 * k + 1

/-- Theorem stating that there exists a number of steps that results in 1001 free ends -/
theorem exists_steps_for_1001_free_ends : ∃ k : ℕ, free_ends k = 1001 := by
  sorry

end NUMINAMATH_CALUDE_exists_steps_for_1001_free_ends_l797_79753


namespace NUMINAMATH_CALUDE_sally_initial_cards_l797_79785

/-- The number of Pokemon cards Sally received from Dan -/
def cards_from_dan : ℕ := 41

/-- The number of Pokemon cards Sally bought -/
def cards_bought : ℕ := 20

/-- The total number of Pokemon cards Sally has now -/
def total_cards_now : ℕ := 88

/-- The number of Pokemon cards Sally had initially -/
def initial_cards : ℕ := total_cards_now - (cards_from_dan + cards_bought)

theorem sally_initial_cards : initial_cards = 27 := by
  sorry

end NUMINAMATH_CALUDE_sally_initial_cards_l797_79785
