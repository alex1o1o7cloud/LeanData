import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l4076_407691

-- Define the coefficients of a quadratic equation ax^2 + bx + c = 0
def QuadraticCoefficients (a b c : ℝ) : Prop :=
  ∀ x, a * x^2 + b * x + c = 0 ↔ x^2 - x + 3 = 0

-- Theorem statement
theorem quadratic_equation_coefficients :
  QuadraticCoefficients 1 (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l4076_407691


namespace NUMINAMATH_CALUDE_ellipse_equation_l4076_407609

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A parabola with equation y² = 4px where p is the focal distance -/
structure Parabola where
  p : ℝ
  h_pos : 0 < p

/-- A circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ
  h_pos : 0 < r

/-- The theorem stating the conditions and the result to be proved -/
theorem ellipse_equation (e : Ellipse) (p : Parabola) (c : Circle) 
  (h_focus : e.a^2 - e.b^2 = p.p^2) 
  (h_major_axis : 2 * e.a = c.r) 
  (h_parabola : p.p^2 = 3) :
  e.a^2 = 4 ∧ e.b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l4076_407609


namespace NUMINAMATH_CALUDE_intersection_sum_l4076_407680

theorem intersection_sum (a b : ℝ) : 
  (3 = (1/3) * 6 + a) ∧ (6 = (1/3) * 3 + b) → a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l4076_407680


namespace NUMINAMATH_CALUDE_no_integer_points_on_circle_l4076_407637

theorem no_integer_points_on_circle : 
  ∀ x : ℤ, (x - 3)^2 + (3*x + 1)^2 > 16 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_points_on_circle_l4076_407637


namespace NUMINAMATH_CALUDE_first_discount_percentage_l4076_407676

theorem first_discount_percentage (original_price : ℝ) (second_discount : ℝ) (final_price : ℝ) :
  original_price = 175 →
  second_discount = 5 →
  final_price = 133 →
  ∃ (first_discount : ℝ),
    first_discount = 20 ∧
    final_price = original_price * (100 - first_discount) / 100 * (100 - second_discount) / 100 :=
by sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l4076_407676


namespace NUMINAMATH_CALUDE_smallest_sum_arithmetic_geometric_sequence_l4076_407677

theorem smallest_sum_arithmetic_geometric_sequence (A B C D : ℤ) : 
  A > 0 → B > 0 → C > 0 →
  (∃ r : ℚ, C - B = B - A ∧ C = B * r ∧ D = C * r) →
  C = (5 : ℚ) / 3 * B →
  A + B + C + D ≥ 52 ∧ (∃ A' B' C' D' : ℤ, 
    A' > 0 ∧ B' > 0 ∧ C' > 0 ∧
    (∃ r' : ℚ, C' - B' = B' - A' ∧ C' = B' * r' ∧ D' = C' * r') ∧
    C' = (5 : ℚ) / 3 * B' ∧
    A' + B' + C' + D' = 52) := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_arithmetic_geometric_sequence_l4076_407677


namespace NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l4076_407669

theorem square_sum_given_diff_and_product (x y : ℝ) 
  (h1 : x - y = 20) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 418 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l4076_407669


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_equations_l4076_407611

-- Define the foci
def F₁ : ℝ × ℝ := (0, -5)
def F₂ : ℝ × ℝ := (0, 5)

-- Define the intersection point
def P : ℝ × ℝ := (3, 4)

-- Define the ellipse equation
def is_on_ellipse (x y : ℝ) : Prop :=
  y^2 / 40 + x^2 / 15 = 1

-- Define the hyperbola equation
def is_on_hyperbola (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / 9 = 1

-- Define the asymptote equation
def is_on_asymptote (x y : ℝ) : Prop :=
  y = (4/3) * x

-- Theorem statement
theorem hyperbola_ellipse_equations :
  (is_on_ellipse P.1 P.2) ∧
  (is_on_hyperbola P.1 P.2) ∧
  (is_on_asymptote P.1 P.2) ∧
  (F₁.2 = -F₂.2) ∧
  (F₁.1 = F₂.1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_equations_l4076_407611


namespace NUMINAMATH_CALUDE_candy_cost_calculation_l4076_407697

/-- The problem of calculating the total cost of candy -/
theorem candy_cost_calculation (cost_per_piece : ℕ) (num_gumdrops : ℕ) (total_cost : ℕ) : 
  cost_per_piece = 8 → num_gumdrops = 28 → total_cost = cost_per_piece * num_gumdrops → total_cost = 224 :=
by sorry

end NUMINAMATH_CALUDE_candy_cost_calculation_l4076_407697


namespace NUMINAMATH_CALUDE_range_of_m_l4076_407689

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x + 1) * (x - 3) < 0
def q (x m : ℝ) : Prop := 3 * x - 4 < m

-- Define the necessary but not sufficient condition
def necessary_but_not_sufficient (p q : ℝ → Prop) : Prop :=
  (∀ x, q x → p x) ∧ ∃ x, p x ∧ ¬q x

-- Theorem statement
theorem range_of_m (m : ℝ) :
  (∃ x, q x m) ∧ necessary_but_not_sufficient (p · ) (q · m) ↔ m ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l4076_407689


namespace NUMINAMATH_CALUDE_number_approximation_l4076_407635

-- Define the function f
def f (x : ℝ) : ℝ := x

-- Define the approximation relation
def approx (x y : ℝ) : Prop := abs (x - y) < 0.000000000000001

-- State the theorem
theorem number_approximation (x : ℝ) :
  approx (f (69.28 * 0.004) / x) 9.237333333333334 →
  approx x 0.03 :=
by
  sorry

end NUMINAMATH_CALUDE_number_approximation_l4076_407635


namespace NUMINAMATH_CALUDE_circle_on_grid_regions_l4076_407624

/-- Represents a grid with uniform spacing -/
structure Grid :=
  (spacing : ℝ)

/-- Represents a circle on the grid -/
structure CircleOnGrid :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- Represents a region formed by circle arcs and grid line segments -/
structure Region

/-- Calculates the number of regions formed by a circle on a grid -/
def count_regions (g : Grid) (c : CircleOnGrid) : ℕ :=
  sorry

/-- Calculates the areas of regions formed by a circle on a grid -/
def region_areas (g : Grid) (c : CircleOnGrid) : List ℝ :=
  sorry

/-- Main theorem: Number and areas of regions formed by a circle on a grid -/
theorem circle_on_grid_regions 
  (g : Grid) 
  (c : CircleOnGrid) 
  (h1 : g.spacing = 1) 
  (h2 : c.radius = 5) 
  (h3 : c.center = (0, 0)) :
  (count_regions g c = 56) ∧ 
  (region_areas g c ≈ [0.966, 0.761, 0.317, 0.547]) :=
by sorry

#check circle_on_grid_regions

end NUMINAMATH_CALUDE_circle_on_grid_regions_l4076_407624


namespace NUMINAMATH_CALUDE_susies_house_rooms_l4076_407607

/-- The number of rooms in Susie's house -/
def number_of_rooms : ℕ := 6

/-- The time it takes Susie to vacuum the whole house, in hours -/
def total_vacuum_time : ℝ := 2

/-- The time it takes Susie to vacuum one room, in minutes -/
def time_per_room : ℝ := 20

/-- Theorem stating that the number of rooms in Susie's house is 6 -/
theorem susies_house_rooms :
  number_of_rooms = (total_vacuum_time * 60) / time_per_room :=
by sorry

end NUMINAMATH_CALUDE_susies_house_rooms_l4076_407607


namespace NUMINAMATH_CALUDE_fourth_week_sales_l4076_407693

def chocolate_sales (week1 week2 week3 week4 week5 : ℕ) : Prop :=
  let total := week1 + week2 + week3 + week4 + week5
  (total : ℚ) / 5 = 71

theorem fourth_week_sales :
  ∀ week4 : ℕ,
  chocolate_sales 75 67 75 week4 68 →
  week4 = 70 := by
sorry

end NUMINAMATH_CALUDE_fourth_week_sales_l4076_407693


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4076_407667

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B : Set ℝ := {-3, -1, 1, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1, 1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4076_407667


namespace NUMINAMATH_CALUDE_kangaroo_jump_distance_l4076_407614

/-- Proves that a kangaroo jumping up and down a mountain with specific jump patterns covers a total distance of 3036 meters. -/
theorem kangaroo_jump_distance (total_jumps : ℕ) (uphill_distance downhill_distance : ℝ) 
  (h1 : total_jumps = 2024)
  (h2 : uphill_distance = 1)
  (h3 : downhill_distance = 3)
  (h4 : ∃ (uphill_jumps downhill_jumps : ℕ), 
    uphill_jumps + downhill_jumps = total_jumps ∧ 
    uphill_jumps = 3 * downhill_jumps) :
  ∃ (total_distance : ℝ), total_distance = 3036 := by
  sorry

end NUMINAMATH_CALUDE_kangaroo_jump_distance_l4076_407614


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_inequality_l4076_407645

theorem quadratic_no_real_roots_inequality (a b c : ℝ) :
  ((b + c) * x^2 + (a + c) * x + (a + b) = 0 → False) →
  4 * a * c - b^2 ≤ 3 * a * (a + b + c) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_inequality_l4076_407645


namespace NUMINAMATH_CALUDE_fifth_term_integer_probability_l4076_407631

def sequence_rule (prev : ℤ) (is_heads : Bool) : ℤ :=
  if is_heads then
    3 * prev + 1
  else if prev % 3 = 0 then
    prev / 3 - 1
  else
    prev - 2

def fourth_term_rule (third_term : ℤ) (third_term_was_heads : Bool) : ℤ :=
  sequence_rule third_term third_term_was_heads

def is_integer (x : ℚ) : Prop :=
  ∃ n : ℤ, x = n

theorem fifth_term_integer_probability :
  let first_term := 4
  let coin_probability := (1 : ℚ) / 2
  ∀ second_term_heads third_term_heads fifth_term_heads : Bool,
    is_integer (sequence_rule
      (fourth_term_rule
        (sequence_rule
          (sequence_rule first_term second_term_heads)
          third_term_heads)
        third_term_heads)
      fifth_term_heads) :=
by sorry

end NUMINAMATH_CALUDE_fifth_term_integer_probability_l4076_407631


namespace NUMINAMATH_CALUDE_line_intersects_circle_shortest_chord_l4076_407698

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 1 - 2 * k

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y - 7 = 0

-- Theorem 1: Line l always intersects circle C
theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, line_l k x y ∧ circle_C x y :=
sorry

-- Theorem 2: The line x + 2y - 4 = 0 produces the shortest chord
theorem shortest_chord :
  ∀ k : ℝ,
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    x₁ ≠ x₂ ∧
    line_l k x₁ y₁ ∧ circle_C x₁ y₁ ∧
    line_l k x₂ y₂ ∧ circle_C x₂ y₂) →
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    x₁ ≠ x₂ ∧
    x₁ + 2*y₁ - 4 = 0 ∧ circle_C x₁ y₁ ∧
    x₂ + 2*y₂ - 4 = 0 ∧ circle_C x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 ≤ (x₁ - x₂)^2 + (y₁ - y₂)^2) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_shortest_chord_l4076_407698


namespace NUMINAMATH_CALUDE_ab_value_l4076_407629

theorem ab_value (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 35) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l4076_407629


namespace NUMINAMATH_CALUDE_number_percentage_equality_l4076_407623

theorem number_percentage_equality (x : ℝ) :
  (40 / 100) * x = (30 / 100) * 50 → x = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_number_percentage_equality_l4076_407623


namespace NUMINAMATH_CALUDE_book_price_increase_l4076_407647

/-- Calculates the new price of a book after a percentage increase -/
theorem book_price_increase (original_price : ℝ) (increase_percentage : ℝ) :
  original_price = 300 ∧ increase_percentage = 30 →
  original_price * (1 + increase_percentage / 100) = 390 := by
sorry

end NUMINAMATH_CALUDE_book_price_increase_l4076_407647


namespace NUMINAMATH_CALUDE_perfect_square_solution_l4076_407670

theorem perfect_square_solution : 
  ∃! (n : ℤ), ∃ (m : ℤ), n^2 + 20*n + 11 = m^2 :=
by
  -- The unique solution is n = 35
  use 35
  sorry

end NUMINAMATH_CALUDE_perfect_square_solution_l4076_407670


namespace NUMINAMATH_CALUDE_equation_solutions_l4076_407657

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 81 = 0 ↔ x = 9 ∨ x = -9) ∧
  (∀ x : ℝ, x^3 - 3 = 3/8 ↔ x = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l4076_407657


namespace NUMINAMATH_CALUDE_matrix_cube_equals_negative_identity_l4076_407663

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, -1; 1, 0]

theorem matrix_cube_equals_negative_identity :
  A ^ 3 = !![(-1 : ℤ), 0; 0, -1] := by sorry

end NUMINAMATH_CALUDE_matrix_cube_equals_negative_identity_l4076_407663


namespace NUMINAMATH_CALUDE_absent_student_percentage_l4076_407627

theorem absent_student_percentage
  (total_students : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (absent_boys_fraction : ℚ)
  (absent_girls_fraction : ℚ)
  (h1 : total_students = 100)
  (h2 : boys = 50)
  (h3 : girls = 50)
  (h4 : boys + girls = total_students)
  (h5 : absent_boys_fraction = 1 / 5)
  (h6 : absent_girls_fraction = 1 / 4) :
  (↑boys * absent_boys_fraction + ↑girls * absent_girls_fraction) / ↑total_students = 225 / 1000 := by
  sorry

#check absent_student_percentage

end NUMINAMATH_CALUDE_absent_student_percentage_l4076_407627


namespace NUMINAMATH_CALUDE_multiple_with_specific_remainders_l4076_407650

theorem multiple_with_specific_remainders (n : ℕ) : 
  (∃ k : ℕ, n = 23 * k) ∧ 
  (n % 1821 = 710) ∧ 
  (n % 24 = 13) ∧ 
  (∀ m : ℕ, m < n → ¬((∃ k : ℕ, m = 23 * k) ∧ (m % 1821 = 710) ∧ (m % 24 = 13))) ∧ 
  (n = 3024) → 
  23 = 23 := by sorry

end NUMINAMATH_CALUDE_multiple_with_specific_remainders_l4076_407650


namespace NUMINAMATH_CALUDE_greatest_x_with_lcm_l4076_407652

theorem greatest_x_with_lcm (x : ℕ) : 
  (∃ (lcm : ℕ), lcm = Nat.lcm x (Nat.lcm 12 18) ∧ lcm = 108) →
  x ≤ 108 ∧ ∃ (y : ℕ), y = 108 ∧ Nat.lcm y (Nat.lcm 12 18) = 108 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_with_lcm_l4076_407652


namespace NUMINAMATH_CALUDE_height_difference_ruby_xavier_l4076_407679

-- Constants and conversion factors
def inch_to_cm : ℝ := 2.54
def m_to_cm : ℝ := 100

-- Given heights and relationships
def janet_height_inch : ℝ := 62.75
def charlene_height_factor : ℝ := 1.5
def pablo_charlene_diff_m : ℝ := 1.85
def ruby_pablo_diff_cm : ℝ := 0.5
def xavier_charlene_diff_m : ℝ := 2.13
def paul_xavier_diff_cm : ℝ := 97.75
def paul_ruby_diff_m : ℝ := 0.5

-- Theorem statement
theorem height_difference_ruby_xavier :
  let janet_height_cm := janet_height_inch * inch_to_cm
  let charlene_height_cm := charlene_height_factor * janet_height_cm
  let pablo_height_cm := charlene_height_cm + pablo_charlene_diff_m * m_to_cm
  let ruby_height_cm := pablo_height_cm - ruby_pablo_diff_cm
  let xavier_height_cm := charlene_height_cm + xavier_charlene_diff_m * m_to_cm
  let paul_height_cm := ruby_height_cm + paul_ruby_diff_m * m_to_cm
  let height_diff_cm := xavier_height_cm - ruby_height_cm
  let height_diff_inch := height_diff_cm / inch_to_cm
  ∃ ε > 0, |height_diff_inch - 18.78| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_height_difference_ruby_xavier_l4076_407679


namespace NUMINAMATH_CALUDE_valid_param_iff_l4076_407615

/-- A structure representing a vector parameterization of a line -/
structure VectorParam where
  x₀ : ℝ
  y₀ : ℝ
  a : ℝ
  b : ℝ

/-- The line equation y = 2x + 6 -/
def line_equation (x y : ℝ) : Prop := y = 2 * x + 6

/-- Predicate to check if a vector parameterization is valid for the line y = 2x + 6 -/
def is_valid_param (p : VectorParam) : Prop :=
  line_equation p.x₀ p.y₀ ∧ p.b = 2 * p.a

/-- Theorem stating the condition for a valid vector parameterization -/
theorem valid_param_iff (p : VectorParam) :
  is_valid_param p ↔
    (∀ t : ℝ, line_equation (p.x₀ + t * p.a) (p.y₀ + t * p.b)) :=
by sorry

end NUMINAMATH_CALUDE_valid_param_iff_l4076_407615


namespace NUMINAMATH_CALUDE_hypotenuse_length_is_5_sqrt_211_l4076_407654

/-- Right triangle ABC with specific properties -/
structure RightTriangle where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  -- AB and AC are legs of the right triangle
  ab_leg : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  -- X is on AB
  x_on_ab : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))
  -- Y is on AC
  y_on_ac : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ Y = (A.1 + s * (C.1 - A.1), A.2 + s * (C.2 - A.2))
  -- AX:XB = 2:3
  ax_xb_ratio : dist A X / dist X B = 2 / 3
  -- AY:YC = 2:3
  ay_yc_ratio : dist A Y / dist Y C = 2 / 3
  -- BY = 18 units
  by_length : dist B Y = 18
  -- CX = 15 units
  cx_length : dist C X = 15

/-- The length of hypotenuse BC in the right triangle -/
def hypotenuseLength (t : RightTriangle) : ℝ :=
  dist t.B t.C

/-- Theorem: The length of hypotenuse BC is 5√211 units -/
theorem hypotenuse_length_is_5_sqrt_211 (t : RightTriangle) :
  hypotenuseLength t = 5 * Real.sqrt 211 := by
  sorry


end NUMINAMATH_CALUDE_hypotenuse_length_is_5_sqrt_211_l4076_407654


namespace NUMINAMATH_CALUDE_factorization_equality_l4076_407632

theorem factorization_equality (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l4076_407632


namespace NUMINAMATH_CALUDE_cab_travel_time_l4076_407686

/-- Proves that if a cab travels at 5/6 of its usual speed and arrives 6 minutes late, its usual travel time is 30 minutes. -/
theorem cab_travel_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) 
  (h2 : usual_time > 0) 
  (h3 : usual_speed * usual_time = (5/6 * usual_speed) * (usual_time + 1/10)) : 
  usual_time = 1/2 := by
sorry

end NUMINAMATH_CALUDE_cab_travel_time_l4076_407686


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l4076_407665

theorem tangent_line_to_circle (m : ℝ) : 
  m > 0 → 
  (∀ x y : ℝ, x + y = 0 → (x - m)^2 + y^2 = 2) → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l4076_407665


namespace NUMINAMATH_CALUDE_probability_three_defective_before_two_good_l4076_407640

/-- Represents the number of good products in the box -/
def goodProducts : ℕ := 9

/-- Represents the number of defective products in the box -/
def defectiveProducts : ℕ := 3

/-- Represents the total number of products in the box -/
def totalProducts : ℕ := goodProducts + defectiveProducts

/-- Calculates the probability of selecting 3 defective products before 2 good products -/
def probabilityThreeDefectiveBeforeTwoGood : ℚ :=
  (4 : ℚ) / 55

/-- Theorem stating that the probability of selecting 3 defective products
    before 2 good products is 4/55 -/
theorem probability_three_defective_before_two_good :
  probabilityThreeDefectiveBeforeTwoGood = (4 : ℚ) / 55 := by
  sorry

#eval probabilityThreeDefectiveBeforeTwoGood

end NUMINAMATH_CALUDE_probability_three_defective_before_two_good_l4076_407640


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l4076_407685

theorem algebraic_expression_equality (x : ℝ) (h : x^2 - 4*x + 1 = 3) :
  3*x^2 - 12*x - 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l4076_407685


namespace NUMINAMATH_CALUDE_jason_pokemon_cards_l4076_407618

theorem jason_pokemon_cards (initial : ℕ) : 
  initial - 9 = 4 → initial = 13 := by
  sorry

end NUMINAMATH_CALUDE_jason_pokemon_cards_l4076_407618


namespace NUMINAMATH_CALUDE_machine_B_performs_better_l4076_407673

def machineA : List ℕ := [0, 1, 0, 2, 2, 0, 3, 1, 2, 4]
def machineB : List ℕ := [2, 3, 1, 1, 0, 2, 1, 1, 0, 1]

def average (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

def variance (l : List ℕ) : ℚ :=
  let avg := average l
  (l.map (fun x => ((x : ℚ) - avg) ^ 2)).sum / l.length

theorem machine_B_performs_better :
  average machineB < average machineA ∧
  variance machineB < variance machineA := by
  sorry

end NUMINAMATH_CALUDE_machine_B_performs_better_l4076_407673


namespace NUMINAMATH_CALUDE_imaginary_unit_equation_l4076_407606

theorem imaginary_unit_equation (a : ℝ) (h1 : a > 0) :
  Complex.abs ((a + Complex.I) / Complex.I) = 2 → a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_equation_l4076_407606


namespace NUMINAMATH_CALUDE_equation_solutions_l4076_407601

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 2*x = 0 ↔ x = 0 ∨ x = 2) ∧
  (∀ x : ℝ, x^2 - 4*x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l4076_407601


namespace NUMINAMATH_CALUDE_cyclic_inequality_l4076_407658

theorem cyclic_inequality (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (y + z) / (2 * x) + (z + x) / (2 * y) + (x + y) / (2 * z) ≥ 
  2 * x / (y + z) + 2 * y / (z + x) + 2 * z / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l4076_407658


namespace NUMINAMATH_CALUDE_wendy_shoes_theorem_l4076_407604

/-- The number of pairs of shoes Wendy gave away -/
def shoes_given_away (total : ℕ) (left : ℕ) : ℕ := total - left

/-- Theorem stating that Wendy gave away 14 pairs of shoes -/
theorem wendy_shoes_theorem (total : ℕ) (left : ℕ) 
  (h1 : total = 33) 
  (h2 : left = 19) : 
  shoes_given_away total left = 14 := by
  sorry

end NUMINAMATH_CALUDE_wendy_shoes_theorem_l4076_407604


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l4076_407620

-- Define the radius of the larger circle
def R : ℝ := 10

-- Define the radius of the smaller circles
def r : ℝ := 5

-- Theorem statement
theorem shaded_area_calculation :
  let larger_circle_area := π * R^2
  let smaller_circle_area := π * r^2
  let shaded_area := larger_circle_area - 2 * smaller_circle_area
  shaded_area = 50 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l4076_407620


namespace NUMINAMATH_CALUDE_cone_volume_l4076_407662

/-- A cone with an isosceles right triangle as its axis section and a lateral area of 16√2π has a volume of 64π/3 -/
theorem cone_volume (r l h : ℝ) : 
  r > 0 → l > 0 → h > 0 →
  2 * r = Real.sqrt 2 * l →  -- Isosceles right triangle condition
  π * r * l = 16 * Real.sqrt 2 * π →  -- Lateral area condition
  (1 / 3) * π * r^2 * h = (64 / 3) * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l4076_407662


namespace NUMINAMATH_CALUDE_min_value_on_line_l4076_407668

/-- Given real numbers x and y satisfying the equation x + 2y + 3 = 0,
    the minimum value of √(x² + y² - 2y + 1) is √5. -/
theorem min_value_on_line (x y : ℝ) (h : x + 2*y + 3 = 0) :
  ∃ (m : ℝ), m = Real.sqrt 5 ∧ ∀ (x' y' : ℝ), x' + 2*y' + 3 = 0 →
    m ≤ Real.sqrt (x'^2 + y'^2 - 2*y' + 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_line_l4076_407668


namespace NUMINAMATH_CALUDE_parabola_a_range_l4076_407687

/-- A parabola that opens downwards -/
structure DownwardParabola where
  a : ℝ
  eq : ℝ → ℝ := λ x => a * x^2 - 2 * a * x + 3
  opens_downward : a < 0

/-- The theorem stating the range of 'a' for a downward parabola with positive y-values in (0, 3) -/
theorem parabola_a_range (p : DownwardParabola) 
  (h : ∀ x, 0 < x → x < 3 → p.eq x > 0) : 
  -1 < p.a ∧ p.a < 0 := by
  sorry


end NUMINAMATH_CALUDE_parabola_a_range_l4076_407687


namespace NUMINAMATH_CALUDE_liya_number_preference_l4076_407692

theorem liya_number_preference (n : ℕ) : 
  (n % 3 = 0) ∧ (n % 10 = 0) → n % 10 = 0 := by
sorry

end NUMINAMATH_CALUDE_liya_number_preference_l4076_407692


namespace NUMINAMATH_CALUDE_large_circle_radius_large_circle_radius_value_l4076_407671

/-- The radius of a circle that internally touches two circles of radius 2 and both internally
    and externally touches a third circle of radius 2 (where all three smaller circles are
    externally tangent to each other) is equal to 4 + 2√3. -/
theorem large_circle_radius : ℝ → ℝ → Prop :=
  fun (small_radius large_radius : ℝ) =>
    small_radius = 2 ∧
    (∃ (centers : Fin 3 → ℝ × ℝ) (large_center : ℝ × ℝ),
      (∀ i j, i ≠ j → dist (centers i) (centers j) = 2 * small_radius) ∧
      (∀ i, dist (centers i) large_center ≤ large_radius + small_radius) ∧
      (∃ k, dist (centers k) large_center = large_radius - small_radius) ∧
      (∃ l, dist (centers l) large_center = large_radius + small_radius)) →
    large_radius = 4 + 2 * Real.sqrt 3

theorem large_circle_radius_value : large_circle_radius 2 (4 + 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_large_circle_radius_large_circle_radius_value_l4076_407671


namespace NUMINAMATH_CALUDE_fraction_equality_l4076_407672

theorem fraction_equality (x y : ℝ) (h : x ≠ y) : -x / (x - y) = x / (-x + y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4076_407672


namespace NUMINAMATH_CALUDE_fibonacci_eighth_term_l4076_407644

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem fibonacci_eighth_term : fibonacci 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_eighth_term_l4076_407644


namespace NUMINAMATH_CALUDE_projectile_max_height_l4076_407683

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 30

/-- The maximum height of the projectile -/
def max_height : ℝ := 155

/-- Theorem: The maximum height of the projectile is 155 feet -/
theorem projectile_max_height :
  ∀ t : ℝ, h t ≤ max_height :=
by
  sorry

end NUMINAMATH_CALUDE_projectile_max_height_l4076_407683


namespace NUMINAMATH_CALUDE_triangle_reconstruction_l4076_407603

-- Define the centers of the squares
structure SquareCenters where
  O₁ : ℝ × ℝ
  O₂ : ℝ × ℝ
  O₃ : ℝ × ℝ

-- Define a 90-degree rotation around a point
def rotate90 (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  sorry

-- Define the composition of rotations
def compositeRotation (centers : SquareCenters) (point : ℝ × ℝ) : ℝ × ℝ :=
  rotate90 centers.O₃ (rotate90 centers.O₂ (rotate90 centers.O₁ point))

-- Theorem stating the existence of an invariant point
theorem triangle_reconstruction (centers : SquareCenters) :
  ∃ (B : ℝ × ℝ), compositeRotation centers B = B :=
sorry

end NUMINAMATH_CALUDE_triangle_reconstruction_l4076_407603


namespace NUMINAMATH_CALUDE_ribbon_count_l4076_407625

theorem ribbon_count (morning_given afternoon_given remaining : ℕ) 
  (h1 : morning_given = 14)
  (h2 : afternoon_given = 16)
  (h3 : remaining = 8) :
  morning_given + afternoon_given + remaining = 38 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_count_l4076_407625


namespace NUMINAMATH_CALUDE_regular_hexagon_area_l4076_407664

/-- The area of a regular hexagon with side length 8 inches is 96√3 square inches. -/
theorem regular_hexagon_area :
  let side_length : ℝ := 8
  let area : ℝ := (3 * Real.sqrt 3 / 2) * side_length ^ 2
  area = 96 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_regular_hexagon_area_l4076_407664


namespace NUMINAMATH_CALUDE_beach_probability_l4076_407610

/-- Given a beach scenario where:
  * 75 people are wearing sunglasses
  * 60 people are wearing hats
  * The probability of wearing sunglasses given wearing a hat is 1/3
  This theorem proves that the probability of wearing a hat given wearing sunglasses is 4/15. -/
theorem beach_probability (total_sunglasses : ℕ) (total_hats : ℕ) 
  (prob_sunglasses_given_hat : ℚ) :
  total_sunglasses = 75 →
  total_hats = 60 →
  prob_sunglasses_given_hat = 1/3 →
  (total_hats * prob_sunglasses_given_hat : ℚ) / total_sunglasses = 4/15 :=
by sorry

end NUMINAMATH_CALUDE_beach_probability_l4076_407610


namespace NUMINAMATH_CALUDE_coin_coverage_theorem_l4076_407621

/-- Represents the arrangement of 7 identical coins on an infinite plane -/
structure CoinArrangement where
  radius : ℝ
  num_coins : Nat
  touches_six : Bool

/-- Calculates the percentage of the plane covered by the coins -/
def coverage_percentage (arrangement : CoinArrangement) : ℝ :=
  sorry

/-- Theorem stating that the coverage percentage is 50π/√3 % -/
theorem coin_coverage_theorem (arrangement : CoinArrangement) 
  (h1 : arrangement.num_coins = 7)
  (h2 : arrangement.touches_six = true) : 
  coverage_percentage arrangement = (50 * Real.pi) / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_coin_coverage_theorem_l4076_407621


namespace NUMINAMATH_CALUDE_fish_offspring_conversion_l4076_407648

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (n : ℕ) : ℕ :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7^1 + (n % 10) * 7^0

/-- The fish offspring count in base 7 --/
def fishOffspringBase7 : ℕ := 265

theorem fish_offspring_conversion :
  base7ToBase10 fishOffspringBase7 = 145 := by
  sorry

end NUMINAMATH_CALUDE_fish_offspring_conversion_l4076_407648


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l4076_407608

theorem necessary_but_not_sufficient :
  ∀ x : ℝ,
  (x + 2 = 0 → x^2 - 4 = 0) ∧
  ¬(x^2 - 4 = 0 → x + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l4076_407608


namespace NUMINAMATH_CALUDE_supplementary_angle_difference_l4076_407674

theorem supplementary_angle_difference : 
  let angle1 : ℝ := 99
  let angle2 : ℝ := 81
  -- Supplementary angles sum to 180°
  angle1 + angle2 = 180 →
  -- The difference between the larger and smaller angle is 18°
  max angle1 angle2 - min angle1 angle2 = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_supplementary_angle_difference_l4076_407674


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l4076_407659

theorem quadratic_roots_condition (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k * x₁^2 - 2 * x₁ - 1 = 0 ∧ k * x₂^2 - 2 * x₂ - 1 = 0) →
  (k > -1 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l4076_407659


namespace NUMINAMATH_CALUDE_john_has_14_burritos_left_l4076_407688

/-- The number of burritos John has left after buying, receiving a free box, giving away some, and eating for 10 days. -/
def burritos_left : ℕ :=
  let total_burritos : ℕ := 15 + 20 + 25 + 5
  let given_away : ℕ := (total_burritos / 3 : ℕ)
  let after_giving : ℕ := total_burritos - given_away
  let eaten : ℕ := 3 * 10
  after_giving - eaten

/-- Theorem stating that John has 14 burritos left -/
theorem john_has_14_burritos_left : burritos_left = 14 := by
  sorry

end NUMINAMATH_CALUDE_john_has_14_burritos_left_l4076_407688


namespace NUMINAMATH_CALUDE_min_additional_votes_to_win_l4076_407690

/-- Represents the number of candidates in the election -/
def num_candidates : ℕ := 5

/-- Represents the percentage of votes received by candidate A -/
def votes_a_percent : ℚ := 35 / 100

/-- Represents the percentage of votes received by candidate B -/
def votes_b_percent : ℚ := 20 / 100

/-- Represents the percentage of votes received by candidate C -/
def votes_c_percent : ℚ := 15 / 100

/-- Represents the percentage of votes received by candidate D -/
def votes_d_percent : ℚ := 10 / 100

/-- Represents the difference in votes between candidate A and B -/
def votes_difference : ℕ := 1200

/-- Represents the minimum percentage of votes needed to win -/
def win_percentage : ℚ := 36 / 100

/-- Theorem stating the minimum additional votes needed for candidate A to win -/
theorem min_additional_votes_to_win :
  ∃ (total_votes : ℕ) (votes_a : ℕ) (votes_needed : ℕ),
    (votes_a_percent : ℚ) * total_votes = votes_a ∧
    (votes_b_percent : ℚ) * total_votes = votes_a - votes_difference ∧
    (win_percentage : ℚ) * total_votes = votes_needed ∧
    votes_needed - votes_a = 80 :=
sorry

end NUMINAMATH_CALUDE_min_additional_votes_to_win_l4076_407690


namespace NUMINAMATH_CALUDE_negation_of_proposition_l4076_407602

theorem negation_of_proposition :
  (¬ (∀ n : ℕ, n^2 < 3*n + 4)) ↔ (∃ n : ℕ, n^2 ≥ 3*n + 4) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l4076_407602


namespace NUMINAMATH_CALUDE_max_value_m_l4076_407694

theorem max_value_m (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ m : ℝ, m / (3 * a + b) - 3 / a - 1 / b ≤ 0) →
  (∃ m : ℝ, m = 16 ∧ ∀ m' : ℝ, m' / (3 * a + b) - 3 / a - 1 / b ≤ 0 → m' ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_max_value_m_l4076_407694


namespace NUMINAMATH_CALUDE_total_chips_calculation_l4076_407619

/-- The total number of chips Viviana and Susana have together -/
def total_chips (viviana_chocolate viviana_vanilla susana_chocolate susana_vanilla : ℕ) : ℕ :=
  viviana_chocolate + viviana_vanilla + susana_chocolate + susana_vanilla

/-- Theorem stating the total number of chips Viviana and Susana have together -/
theorem total_chips_calculation :
  ∀ (viviana_chocolate viviana_vanilla susana_chocolate susana_vanilla : ℕ),
  viviana_chocolate = susana_chocolate + 5 →
  susana_vanilla = (3 * viviana_vanilla) / 4 →
  viviana_vanilla = 20 →
  susana_chocolate = 25 →
  total_chips viviana_chocolate viviana_vanilla susana_chocolate susana_vanilla = 90 :=
by sorry

end NUMINAMATH_CALUDE_total_chips_calculation_l4076_407619


namespace NUMINAMATH_CALUDE_pens_taken_after_second_month_pens_taken_after_second_month_is_41_l4076_407626

theorem pens_taken_after_second_month 
  (num_students : ℕ) 
  (red_pens_per_student : ℕ) 
  (black_pens_per_student : ℕ) 
  (pens_taken_first_month : ℕ) 
  (pens_per_student_after_split : ℕ) : ℕ :=
  let total_pens := num_students * (red_pens_per_student + black_pens_per_student)
  let pens_after_first_month := total_pens - pens_taken_first_month
  let pens_after_split := num_students * pens_per_student_after_split
  pens_after_first_month - pens_after_split

theorem pens_taken_after_second_month_is_41 :
  pens_taken_after_second_month 3 62 43 37 79 = 41 := by
  sorry

end NUMINAMATH_CALUDE_pens_taken_after_second_month_pens_taken_after_second_month_is_41_l4076_407626


namespace NUMINAMATH_CALUDE_min_sum_of_integers_l4076_407678

theorem min_sum_of_integers (m n : ℕ) : 
  m < n → 
  m > 0 → 
  n > 0 → 
  m * n = (m - 20) * (n + 23) → 
  ∀ k l : ℕ, k < l → k > 0 → l > 0 → k * l = (k - 20) * (l + 23) → m + n ≤ k + l →
  m + n = 321 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_integers_l4076_407678


namespace NUMINAMATH_CALUDE_soup_cans_bought_soup_cans_received_johns_soup_cans_l4076_407616

theorem soup_cans_bought (normal_price : ℝ) (total_paid : ℝ) : ℝ :=
  total_paid / normal_price

theorem soup_cans_received (cans_bought : ℝ) : ℝ :=
  2 * cans_bought

theorem johns_soup_cans (normal_price : ℝ) (total_paid : ℝ) : 
  soup_cans_received (soup_cans_bought normal_price total_paid) = 30 :=
by
  -- Assuming normal_price = 0.60 and total_paid = 9
  have h1 : normal_price = 0.60 := by sorry
  have h2 : total_paid = 9 := by sorry
  
  -- Calculate the number of cans bought
  have cans_bought : ℝ := soup_cans_bought normal_price total_paid
  
  -- Calculate the total number of cans received
  have total_cans : ℝ := soup_cans_received cans_bought
  
  -- Prove that the total number of cans is 30
  sorry

end NUMINAMATH_CALUDE_soup_cans_bought_soup_cans_received_johns_soup_cans_l4076_407616


namespace NUMINAMATH_CALUDE_valid_paths_count_l4076_407696

-- Define the grid dimensions
def rows : Nat := 5
def cols : Nat := 7

-- Define the blocked paths
def blocked_path1 : (Nat × Nat) × (Nat × Nat) := ((4, 2), (5, 2))
def blocked_path2 : (Nat × Nat) × (Nat × Nat) := ((2, 7), (3, 7))

-- Define a function to calculate valid paths
def valid_paths (r : Nat) (c : Nat) (blocked1 blocked2 : (Nat × Nat) × (Nat × Nat)) : Nat :=
  sorry

-- Theorem statement
theorem valid_paths_count : 
  valid_paths rows cols blocked_path1 blocked_path2 = 546 := by sorry

end NUMINAMATH_CALUDE_valid_paths_count_l4076_407696


namespace NUMINAMATH_CALUDE_condition_not_well_defined_l4076_407639

-- Define a type for students
structure Student :=
  (height : ℝ)
  (school : String)

-- Define a type for conditions
inductive Condition
  | TallStudents : Condition
  | PointsAwayFromOrigin : Condition
  | PrimesLessThan100 : Condition
  | QuadraticEquationSolutions : Condition

-- Define a predicate for well-defined sets
def IsWellDefinedSet (c : Condition) : Prop :=
  match c with
  | Condition.TallStudents => false
  | Condition.PointsAwayFromOrigin => true
  | Condition.PrimesLessThan100 => true
  | Condition.QuadraticEquationSolutions => true

-- Theorem statement
theorem condition_not_well_defined :
  ∃ c : Condition, ¬(IsWellDefinedSet c) ∧
  ∀ c' : Condition, c' ≠ c → IsWellDefinedSet c' :=
sorry

end NUMINAMATH_CALUDE_condition_not_well_defined_l4076_407639


namespace NUMINAMATH_CALUDE_factorial_fraction_simplification_l4076_407653

theorem factorial_fraction_simplification :
  (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 8 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_simplification_l4076_407653


namespace NUMINAMATH_CALUDE_multiply_and_add_equality_l4076_407643

theorem multiply_and_add_equality : 45 * 28 + 72 * 45 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_add_equality_l4076_407643


namespace NUMINAMATH_CALUDE_emma_coins_l4076_407605

theorem emma_coins (x : ℚ) (hx : x > 0) : 
  let lost := x / 3
  let found := (3 / 4) * lost
  x - (x - lost + found) = x / 12 := by sorry

end NUMINAMATH_CALUDE_emma_coins_l4076_407605


namespace NUMINAMATH_CALUDE_simplify_expression_l4076_407649

theorem simplify_expression (a b : ℝ) : (18*a + 45*b) + (15*a + 36*b) - (12*a + 40*b) = 21*a + 41*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4076_407649


namespace NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l4076_407661

theorem arithmetic_progression_of_primes (p : ℕ) (a : ℕ → ℕ) (d : ℕ) :
  Prime p →
  (∀ i, i ∈ Finset.range p → Prime (a i)) →
  (∀ i j, i < j → j < p → a j - a i = (j - i) * d) →
  a 0 > p →
  p ∣ d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l4076_407661


namespace NUMINAMATH_CALUDE_driver_net_pay_rate_l4076_407660

/-- Calculate the net rate of pay for a driver --/
theorem driver_net_pay_rate
  (travel_time : ℝ)
  (speed : ℝ)
  (fuel_efficiency : ℝ)
  (pay_rate : ℝ)
  (gasoline_cost : ℝ)
  (h1 : travel_time = 3)
  (h2 : speed = 50)
  (h3 : fuel_efficiency = 25)
  (h4 : pay_rate = 0.60)
  (h5 : gasoline_cost = 2.50)
  : (pay_rate * speed * travel_time - (speed * travel_time / fuel_efficiency) * gasoline_cost) / travel_time = 25 := by
  sorry

end NUMINAMATH_CALUDE_driver_net_pay_rate_l4076_407660


namespace NUMINAMATH_CALUDE_total_paths_count_l4076_407646

/-- Represents the number of paths between different types of points -/
structure PathCounts where
  redToBlue : Nat
  blueToGreen1 : Nat
  blueToGreen2 : Nat
  greenToOrange1 : Nat
  greenToOrange2 : Nat
  orange1ToB : Nat
  orange2ToB : Nat

/-- Calculates the total number of paths from A to B -/
def totalPaths (p : PathCounts) : Nat :=
  let blueToGreen := p.blueToGreen1 * 2 + p.blueToGreen2 * 2
  let greenToOrange := p.greenToOrange1 + p.greenToOrange2
  (p.redToBlue * blueToGreen * greenToOrange * p.orange1ToB) +
  (p.redToBlue * blueToGreen * greenToOrange * p.orange2ToB)

/-- The theorem stating the total number of paths from A to B -/
theorem total_paths_count (p : PathCounts) 
  (h1 : p.redToBlue = 14)
  (h2 : p.blueToGreen1 = 5)
  (h3 : p.blueToGreen2 = 7)
  (h4 : p.greenToOrange1 = 4)
  (h5 : p.greenToOrange2 = 3)
  (h6 : p.orange1ToB = 2)
  (h7 : p.orange2ToB = 8) :
  totalPaths p = 5376 := by
  sorry

end NUMINAMATH_CALUDE_total_paths_count_l4076_407646


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l4076_407699

theorem largest_constant_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (K : ℝ), K = Real.sqrt 3 ∧ 
  (∀ (K' : ℝ), (∀ (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0), 
    Real.sqrt (x * y / z) + Real.sqrt (y * z / x) + Real.sqrt (x * z / y) ≥ K' * Real.sqrt (x + y + z)) → 
  K' ≤ K) ∧
  Real.sqrt (a * b / c) + Real.sqrt (b * c / a) + Real.sqrt (a * c / b) ≥ K * Real.sqrt (a + b + c) :=
sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l4076_407699


namespace NUMINAMATH_CALUDE_consecutive_product_not_perfect_power_l4076_407630

theorem consecutive_product_not_perfect_power :
  ∀ x y : ℤ, ∀ n : ℕ, n > 1 → x * (x + 1) ≠ y^n := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_not_perfect_power_l4076_407630


namespace NUMINAMATH_CALUDE_candy_expenditure_l4076_407613

theorem candy_expenditure (initial : ℕ) (oranges apples left : ℕ) 
  (h1 : initial = 95)
  (h2 : oranges = 14)
  (h3 : apples = 25)
  (h4 : left = 50) :
  initial - (oranges + apples) - left = 6 := by
  sorry

end NUMINAMATH_CALUDE_candy_expenditure_l4076_407613


namespace NUMINAMATH_CALUDE_concyclic_intersection_points_l4076_407633

structure Circle where
  center : Point
  radius : ℝ

structure Chord (c : Circle) where
  endpoint1 : Point
  endpoint2 : Point

def midpoint_of_arc (c : Circle) (ch : Chord c) : Point := sorry

def intersect_chords (c : Circle) (ch1 ch2 : Chord c) : Point := sorry

def concyclic (p1 p2 p3 p4 : Point) : Prop := sorry

theorem concyclic_intersection_points 
  (c : Circle) 
  (bc : Chord c) 
  (a : Point) 
  (ad ae : Chord c) 
  (f g : Point) :
  a = midpoint_of_arc c bc →
  f = intersect_chords c bc ad →
  g = intersect_chords c bc ae →
  concyclic (ad.endpoint2) (ae.endpoint2) f g :=
sorry

end NUMINAMATH_CALUDE_concyclic_intersection_points_l4076_407633


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4076_407628

theorem complex_equation_solution (a : ℂ) :
  a / (1 - Complex.I) = (1 + Complex.I) / Complex.I → a = -2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4076_407628


namespace NUMINAMATH_CALUDE_number_difference_problem_l4076_407636

theorem number_difference_problem : ∃ (a b : ℕ), 
  a + b = 25650 ∧ 
  a % 100 = 0 ∧ 
  a / 100 = b ∧ 
  a - b = 25146 := by
sorry

end NUMINAMATH_CALUDE_number_difference_problem_l4076_407636


namespace NUMINAMATH_CALUDE_binary_to_decimal_11001001_l4076_407641

/-- Converts a list of binary digits to its decimal equivalent -/
def binaryToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 2^(digits.length - 1 - i)) 0

/-- The binary representation of the number -/
def binaryNumber : List Nat := [1, 1, 0, 0, 1, 0, 0, 1]

theorem binary_to_decimal_11001001 :
  binaryToDecimal binaryNumber = 201 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_decimal_11001001_l4076_407641


namespace NUMINAMATH_CALUDE_divides_condition_l4076_407656

theorem divides_condition (p k r : ℕ) : 
  Prime p → 
  k > 0 → 
  r > 0 → 
  p > r → 
  (pk + r) ∣ (p^p + 1) → 
  r ∣ k := by
sorry

end NUMINAMATH_CALUDE_divides_condition_l4076_407656


namespace NUMINAMATH_CALUDE_infinite_series_sum_l4076_407681

theorem infinite_series_sum : 
  ∑' n : ℕ, (1 / ((2*n+1)^2 - (2*n-1)^2)) * (1 / (2*n-1)^2 - 1 / (2*n+1)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l4076_407681


namespace NUMINAMATH_CALUDE_total_spending_l4076_407600

/-- Represents the amount spent by Ben -/
def ben_spent : ℝ := sorry

/-- Represents the amount spent by David -/
def david_spent : ℝ := sorry

/-- Ben spends $1 for every $0.75 David spends -/
axiom spending_ratio : david_spent = 0.75 * ben_spent

/-- Ben spends $12.50 more than David -/
axiom spending_difference : ben_spent = david_spent + 12.50

/-- The total amount spent by Ben and David -/
def total_spent : ℝ := ben_spent + david_spent

/-- Theorem: The total amount spent by Ben and David is $87.50 -/
theorem total_spending : total_spent = 87.50 := by sorry

end NUMINAMATH_CALUDE_total_spending_l4076_407600


namespace NUMINAMATH_CALUDE_shortest_side_length_l4076_407612

theorem shortest_side_length (A B C : Real) (a b c : Real) : 
  B = π/4 → C = π/3 → c = 1 → 
  A + B + C = π → 
  a / (Real.sin A) = b / (Real.sin B) → 
  b / (Real.sin B) = c / (Real.sin C) → 
  a / (Real.sin A) = c / (Real.sin C) → 
  b ≤ a ∧ b ≤ c → 
  b = Real.sqrt 6 / 3 := by sorry

end NUMINAMATH_CALUDE_shortest_side_length_l4076_407612


namespace NUMINAMATH_CALUDE_janes_earnings_is_75_l4076_407617

/-- The amount of money Jane earned for planting flower bulbs -/
def janes_earnings : ℚ :=
  let price_per_bulb : ℚ := 1/2
  let tulip_bulbs : ℕ := 20
  let iris_bulbs : ℕ := tulip_bulbs / 2
  let daffodil_bulbs : ℕ := 30
  let crocus_bulbs : ℕ := daffodil_bulbs * 3
  let total_bulbs : ℕ := tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_bulbs
  (total_bulbs : ℚ) * price_per_bulb

/-- Theorem stating that Jane earned $75 for planting flower bulbs -/
theorem janes_earnings_is_75 : janes_earnings = 75 := by
  sorry

end NUMINAMATH_CALUDE_janes_earnings_is_75_l4076_407617


namespace NUMINAMATH_CALUDE_ellipse_string_length_l4076_407655

/-- Represents an ellipse with given major and minor axes -/
structure Ellipse where
  major_axis : ℝ
  minor_axis : ℝ
  major_axis_positive : major_axis > 0
  minor_axis_positive : minor_axis > 0
  major_axis_ge_minor : major_axis ≥ minor_axis

/-- Represents the string used in the string method for drawing an ellipse -/
def string_length (e : Ellipse) : ℝ := 2 * e.major_axis

/-- Theorem: For an ellipse with major axis 12 cm and minor axis 8 cm, 
    the string length in the string method is 24 cm -/
theorem ellipse_string_length :
  let e : Ellipse := {
    major_axis := 12,
    minor_axis := 8,
    major_axis_positive := by norm_num,
    minor_axis_positive := by norm_num,
    major_axis_ge_minor := by norm_num
  }
  string_length e = 24 := by
  sorry


end NUMINAMATH_CALUDE_ellipse_string_length_l4076_407655


namespace NUMINAMATH_CALUDE_x_greater_than_ln_one_plus_x_l4076_407695

theorem x_greater_than_ln_one_plus_x {x : ℝ} (h : x > 0) : x > Real.log (1 + x) := by
  sorry

end NUMINAMATH_CALUDE_x_greater_than_ln_one_plus_x_l4076_407695


namespace NUMINAMATH_CALUDE_externally_tangent_circles_l4076_407642

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

/-- The equation of circle C₁ -/
def C1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 2*y + 1 = 0

/-- The equation of circle C₂ -/
def C2 (x y m : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y + m = 0

theorem externally_tangent_circles (m : ℝ) :
  (∃ c1 : ℝ × ℝ, ∃ r1 : ℝ, ∀ x y : ℝ, C1 x y ↔ (x - c1.1)^2 + (y - c1.2)^2 = r1^2) →
  (∃ c2 : ℝ × ℝ, ∃ r2 : ℝ, ∀ x y : ℝ, C2 x y m ↔ (x - c2.1)^2 + (y - c2.2)^2 = r2^2) →
  (∃ c1 c2 : ℝ × ℝ, ∃ r1 r2 : ℝ, externally_tangent c1 c2 r1 r2) →
  m = -3 := by
  sorry

end NUMINAMATH_CALUDE_externally_tangent_circles_l4076_407642


namespace NUMINAMATH_CALUDE_quadratic_order_l4076_407684

/-- Given m < -2 and points on a quadratic function, prove y3 < y2 < y1 -/
theorem quadratic_order (m : ℝ) (y1 y2 y3 : ℝ)
  (h_m : m < -2)
  (h_y1 : y1 = (m - 1)^2 + 2*(m - 1))
  (h_y2 : y2 = m^2 + 2*m)
  (h_y3 : y3 = (m + 1)^2 + 2*(m + 1)) :
  y3 < y2 ∧ y2 < y1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_order_l4076_407684


namespace NUMINAMATH_CALUDE_trigonometric_identity_l4076_407666

theorem trigonometric_identity (A B C : Real) 
  (h : A + B + C = Real.pi) : 
  (Real.sin A + Real.sin B - Real.sin C) / (Real.sin A + Real.sin B + Real.sin C) = 
  Real.tan (A/2) * Real.tan (B/2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l4076_407666


namespace NUMINAMATH_CALUDE_parallel_vectors_l4076_407634

-- Define the vectors a and b
def a : ℝ × ℝ := (-1, 1)
def b (t : ℝ) : ℝ × ℝ := (3, t)

-- Define the parallel condition
def is_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

-- Theorem statement
theorem parallel_vectors (t : ℝ) :
  is_parallel (b t) (a.1 + (b t).1, a.2 + (b t).2) → t = -3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_l4076_407634


namespace NUMINAMATH_CALUDE_inequality_proof_l4076_407675

theorem inequality_proof (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 1/2) :
  (1 - a) * (1 - b) ≤ 9/16 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4076_407675


namespace NUMINAMATH_CALUDE_polynomial_factor_coefficients_l4076_407682

theorem polynomial_factor_coefficients :
  ∀ (a b : ℚ),
  (∃ (c d : ℚ), ∀ (x : ℚ),
    a * x^4 + b * x^3 + 40 * x^2 - 20 * x + 9 =
    (4 * x^2 - 3 * x + 2) * (c * x^2 + d * x + 4.5)) →
  a = 11 ∧ b = -121/4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_coefficients_l4076_407682


namespace NUMINAMATH_CALUDE_triangle_side_b_value_l4076_407651

theorem triangle_side_b_value (A B C : ℝ) (a b c : ℝ) :
  c = Real.sqrt 6 →
  Real.cos C = -(1/4 : ℝ) →
  Real.sin A = 2 * Real.sin B →
  b = 1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_b_value_l4076_407651


namespace NUMINAMATH_CALUDE_triangle_inequality_l4076_407638

theorem triangle_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l4076_407638


namespace NUMINAMATH_CALUDE_definite_integral_x_squared_plus_sqrt_one_minus_x_squared_l4076_407622

theorem definite_integral_x_squared_plus_sqrt_one_minus_x_squared :
  ∫ x in (-1)..1, (x^2 + Real.sqrt (1 - x^2)) = 2/3 + π/2 := by sorry

end NUMINAMATH_CALUDE_definite_integral_x_squared_plus_sqrt_one_minus_x_squared_l4076_407622
