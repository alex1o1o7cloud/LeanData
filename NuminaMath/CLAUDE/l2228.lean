import Mathlib

namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l2228_222896

def p (x : ℝ) : ℝ := 4*x^8 - 2*x^6 + 5*x^4 - x^3 + 3*x - 15

theorem remainder_theorem (p : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - a) * q x + p a := sorry

theorem polynomial_remainder :
  ∃ q : ℝ → ℝ, ∀ x, p x = (2*x - 6) * q x + 25158 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l2228_222896


namespace NUMINAMATH_CALUDE_find_x_value_l2228_222870

def is_ascending (l : List ℝ) : Prop :=
  ∀ i j, i < j → i < l.length → j < l.length → l[i]! ≤ l[j]!

def median (l : List ℝ) : ℝ :=
  l[l.length / 2]!

theorem find_x_value (l : List ℝ) (h_length : l.length = 5) 
  (h_ascending : is_ascending l) (h_median : median l = 22) 
  (h_elements : l = [14, 19, x, 23, 27]) : x = 22 :=
sorry

end NUMINAMATH_CALUDE_find_x_value_l2228_222870


namespace NUMINAMATH_CALUDE_second_number_in_expression_l2228_222827

theorem second_number_in_expression : 
  ∃ x : ℝ, (26.3 * x * 20) / 3 + 125 = 2229 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_second_number_in_expression_l2228_222827


namespace NUMINAMATH_CALUDE_trig_expression_equals_half_l2228_222868

theorem trig_expression_equals_half : 
  2 * Real.cos (30 * π / 180) - Real.tan (60 * π / 180) + Real.sin (45 * π / 180) * Real.cos (45 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_half_l2228_222868


namespace NUMINAMATH_CALUDE_cloth_sale_profit_l2228_222885

/-- The number of meters of cloth sold by a trader -/
def meters_sold : ℕ := 40

/-- The profit per meter of cloth in Rupees -/
def profit_per_meter : ℕ := 25

/-- The total profit earned by the trader in Rupees -/
def total_profit : ℕ := 1000

/-- Theorem stating that the number of meters sold multiplied by the profit per meter equals the total profit -/
theorem cloth_sale_profit : meters_sold * profit_per_meter = total_profit := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_profit_l2228_222885


namespace NUMINAMATH_CALUDE_arithmetic_sequence_60th_term_l2228_222835

/-- Given an arithmetic sequence with first term 7 and 21st term 47, prove that the 60th term is 125 -/
theorem arithmetic_sequence_60th_term : 
  ∀ (a : ℕ → ℝ), 
    (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
    a 0 = 7 →                                -- first term
    a 20 = 47 →                              -- 21st term (index starts at 0)
    a 59 = 125 :=                            -- 60th term (index starts at 0)
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_60th_term_l2228_222835


namespace NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l2228_222852

theorem consecutive_integers_cube_sum (x : ℕ) (h : x > 0) 
  (h_prod : x * (x + 1) * (x + 2) = 12 * (3 * x + 3)) :
  x^3 + (x + 1)^3 + (x + 2)^3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l2228_222852


namespace NUMINAMATH_CALUDE_arithmetic_sequence_x_value_l2228_222824

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_x_value
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = 2 * x - 3)
  (h_a3 : a 3 = 5 * x + 4)
  : x = -11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_x_value_l2228_222824


namespace NUMINAMATH_CALUDE_max_area_special_quadrilateral_l2228_222893

/-- A quadrilateral with the property that the product of any two adjacent sides is 1 -/
structure SpecialQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  ab_eq_one : a * b = 1
  bc_eq_one : b * c = 1
  cd_eq_one : c * d = 1
  da_eq_one : d * a = 1

/-- The area of a quadrilateral -/
def area (q : SpecialQuadrilateral) : ℝ := sorry

/-- The maximum area of a SpecialQuadrilateral is 1 -/
theorem max_area_special_quadrilateral :
  ∀ q : SpecialQuadrilateral, area q ≤ 1 ∧ ∃ q' : SpecialQuadrilateral, area q' = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_area_special_quadrilateral_l2228_222893


namespace NUMINAMATH_CALUDE_root_of_series_fraction_l2228_222854

theorem root_of_series_fraction (f g : ℕ → ℝ) :
  (∀ k, f k = 8 * k^3) →
  (∀ k, g k = 27 * k^3) →
  (∑' k, f k) / (∑' k, g k) = 8 / 27 →
  ((∑' k, f k) / (∑' k, g k))^(1/3 : ℝ) = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_root_of_series_fraction_l2228_222854


namespace NUMINAMATH_CALUDE_min_value_theorem_l2228_222882

theorem min_value_theorem (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b) / c + (b + c) / a + (c + d) / b ≥ 6 ∧
  ((a + b) / c + (b + c) / a + (c + d) / b = 6 ↔ a = b ∧ b = c ∧ c = d) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2228_222882


namespace NUMINAMATH_CALUDE_allocation_schemes_eq_540_l2228_222877

/-- The number of ways to allocate teachers to schools -/
def allocation_schemes (math_teachers language_teachers schools : ℕ) : ℕ :=
  (math_teachers.factorial * language_teachers.factorial) / 
  ((math_teachers / schools).factorial ^ schools * 
   (language_teachers / schools).factorial ^ schools * schools.factorial)

/-- Theorem: The number of allocation schemes for the given problem is 540 -/
theorem allocation_schemes_eq_540 : 
  allocation_schemes 3 6 3 = 540 := by
  sorry

end NUMINAMATH_CALUDE_allocation_schemes_eq_540_l2228_222877


namespace NUMINAMATH_CALUDE_cube_volume_in_box_l2228_222825

/-- Given a box with dimensions 9 cm × 12 cm × 3 cm, filled with 108 identical cubes,
    the volume of each cube is 27 cm³. -/
theorem cube_volume_in_box (length width height : ℕ) (num_cubes : ℕ) :
  length = 9 ∧ width = 12 ∧ height = 3 ∧ num_cubes = 108 →
  ∃ (cube_volume : ℕ), cube_volume = 27 ∧ num_cubes * cube_volume = length * width * height :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_in_box_l2228_222825


namespace NUMINAMATH_CALUDE_digit_97_of_1_13_l2228_222847

/-- The decimal representation of 1/13 -/
def decimal_rep_1_13 : ℕ → Fin 10
  | n => match n % 6 with
    | 0 => 0
    | 1 => 7
    | 2 => 6
    | 3 => 9
    | 4 => 2
    | 5 => 3
    | _ => 0  -- This case should never occur due to % 6

/-- The 97th digit after the decimal point in the decimal representation of 1/13 is 0 -/
theorem digit_97_of_1_13 : decimal_rep_1_13 97 = 0 := by
  sorry

end NUMINAMATH_CALUDE_digit_97_of_1_13_l2228_222847


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2228_222819

/-- Given a cubic equation x√x - 9x + 9√x - 4 = 0 with real nonnegative roots,
    the sum of the squares of its roots is 63. -/
theorem sum_of_squares_of_roots : ∃ (r s t : ℝ),
  (∀ x : ℝ, x ≥ 0 → (x * Real.sqrt x - 9 * x + 9 * Real.sqrt x - 4 = 0 ↔ x = r * r ∨ x = s * s ∨ x = t * t)) →
  r * r + s * s + t * t = 63 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2228_222819


namespace NUMINAMATH_CALUDE_line_bisects_and_perpendicular_l2228_222860

/-- The circle C in the xy-plane -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 + x - 2*y + 1 = 0

/-- The line perpendicular to l -/
def PerpendicularLine (x y : ℝ) : Prop := x + 2*y + 3 = 0

/-- The line l -/
def Line_l (x y : ℝ) : Prop := 2*x - y + 2 = 0

/-- Theorem stating that line l bisects circle C and is perpendicular to the given line -/
theorem line_bisects_and_perpendicular :
  (∀ x y : ℝ, Line_l x y → (∃ x' y' : ℝ, Circle x' y' ∧ x = (x' + (-1/2))/2 ∧ y = (y' + 1)/2)) ∧ 
  (∀ x y : ℝ, Line_l x y → PerpendicularLine x y → x * 2 + y * 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_line_bisects_and_perpendicular_l2228_222860


namespace NUMINAMATH_CALUDE_binomial_expansion_constant_term_l2228_222829

theorem binomial_expansion_constant_term (a : ℝ) : 
  (∃ (f : ℝ → ℝ), (∀ x, x ≠ 0 → f x = (a * x^2 - 2 / Real.sqrt x)^5) ∧ 
   (∃ c, c = 160 ∧ (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → |f x - c| < ε))) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_constant_term_l2228_222829


namespace NUMINAMATH_CALUDE_fixed_point_on_circle_l2228_222845

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the directrix
def directrix : ℝ := -1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the circle
def circle_tangent_to_directrix (m : PointOnParabola) (p : ℝ × ℝ) : Prop :=
  let r := m.x - directrix
  (p.1 - m.x)^2 + (p.2 - m.y)^2 = r^2

-- Theorem statement
theorem fixed_point_on_circle (m : PointOnParabola) :
  circle_tangent_to_directrix m focus := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_circle_l2228_222845


namespace NUMINAMATH_CALUDE_bingley_final_bracelets_l2228_222863

/-- The number of bracelets Bingley has at the end of the exchange process. -/
def final_bracelets : ℕ :=
  let bingley_initial := 5
  let kelly_initial := 16
  let kelly_gives := kelly_initial / 4
  let kelly_sets := kelly_gives / 3
  let bingley_receives := kelly_sets
  let bingley_after_receiving := bingley_initial + bingley_receives
  let bingley_gives_away := bingley_receives / 2
  let bingley_before_sister := bingley_after_receiving - bingley_gives_away
  let sister_gets := bingley_before_sister / 3
  bingley_before_sister - sister_gets

/-- Theorem stating that Bingley ends up with 4 bracelets. -/
theorem bingley_final_bracelets : final_bracelets = 4 := by
  sorry

end NUMINAMATH_CALUDE_bingley_final_bracelets_l2228_222863


namespace NUMINAMATH_CALUDE_rachel_math_problems_l2228_222810

/-- The number of math problems Rachel solved in total -/
def total_problems (problems_per_minute : ℕ) (minutes_solved : ℕ) (problems_next_day : ℕ) : ℕ :=
  problems_per_minute * minutes_solved + problems_next_day

/-- Theorem stating that Rachel solved 151 math problems in total -/
theorem rachel_math_problems :
  total_problems 7 18 25 = 151 := by
  sorry

end NUMINAMATH_CALUDE_rachel_math_problems_l2228_222810


namespace NUMINAMATH_CALUDE_inequality_solution_comparison_l2228_222864

theorem inequality_solution_comparison (m n : ℝ) 
  (hm : 5 * m - 2 ≥ 3) 
  (hn : ¬(5 * n - 2 ≥ 3)) : 
  m > n :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_comparison_l2228_222864


namespace NUMINAMATH_CALUDE_sum_of_g_42_and_neg_42_l2228_222813

/-- Given a function g: ℝ → ℝ defined as g(x) = ax^8 + bx^6 - cx^4 + dx^2 + 5
    where a, b, c, d are real constants, if g(42) = 3,
    then g(42) + g(-42) = 6 -/
theorem sum_of_g_42_and_neg_42 (a b c d : ℝ) (g : ℝ → ℝ)
    (h1 : ∀ x, g x = a * x^8 + b * x^6 - c * x^4 + d * x^2 + 5)
    (h2 : g 42 = 3) :
  g 42 + g (-42) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_g_42_and_neg_42_l2228_222813


namespace NUMINAMATH_CALUDE_trapezoid_base_ratio_l2228_222867

/-- A trapezoid with bases a and b, where a > b, and its midsegment is divided into three equal parts by the diagonals. -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  h : a > b

/-- The ratio of the bases of a trapezoid with the given properties is 2:1 -/
theorem trapezoid_base_ratio (t : Trapezoid) : t.a / t.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_base_ratio_l2228_222867


namespace NUMINAMATH_CALUDE_unique_a_value_l2228_222826

def A (a : ℝ) : Set ℝ := {0, 2, a}
def B (a : ℝ) : Set ℝ := {1, a^2}

theorem unique_a_value : ∃! a : ℝ, A a ∪ B a = {0, 1, 2, 3, 9} ∧ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l2228_222826


namespace NUMINAMATH_CALUDE_frog_probability_l2228_222898

/-- Represents a position on the 4x4 grid -/
inductive Position
| Center : Position
| Interior : Position
| Edge : Position

/-- Represents the number of hops -/
def MaxHops : Nat := 5

/-- The probability of reaching an edge from a given position after n hops -/
noncomputable def probability (pos : Position) (n : Nat) : Real :=
  match pos, n with
  | Position.Edge, _ => 1
  | _, 0 => 0
  | Position.Center, n + 1 => 
      (1/4) * (probability Position.Interior n + probability Position.Interior n + 
               probability Position.Edge n + probability Position.Edge n)
  | Position.Interior, n + 1 => 
      (1/4) * (probability Position.Interior n + probability Position.Interior n + 
               probability Position.Edge n + probability Position.Edge n)

/-- The main theorem to be proved -/
theorem frog_probability : 
  probability Position.Center MaxHops = 121/128 := by
  sorry

end NUMINAMATH_CALUDE_frog_probability_l2228_222898


namespace NUMINAMATH_CALUDE_f_neg_l2228_222836

-- Define an odd function f on the real numbers
def f : ℝ → ℝ := sorry

-- Define the property of f being odd
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- Define f for positive x
axiom f_pos : ∀ x : ℝ, x > 0 → f x = x^2 + 1

-- Theorem to prove
theorem f_neg : ∀ x : ℝ, x < 0 → f x = -x^2 - 1 := by sorry

end NUMINAMATH_CALUDE_f_neg_l2228_222836


namespace NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l2228_222890

-- Part 1
theorem calculation_proof : 
  |(-Real.sqrt 3)| + (3 - Real.pi)^(0 : ℝ) + (1/3)^(-2 : ℝ) = Real.sqrt 3 + 10 := by sorry

-- Part 2
theorem inequality_system_solution :
  {x : ℝ | 3*x + 1 > 2*(x - 1) ∧ x - 1 ≤ 3*x + 3} = {x : ℝ | x ≥ -2} := by sorry

end NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l2228_222890


namespace NUMINAMATH_CALUDE_larger_number_proof_l2228_222828

theorem larger_number_proof (L S : ℕ) (hL : L > S) : 
  L - S = 1355 → L = 6 * S + 15 → L = 1623 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2228_222828


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l2228_222856

theorem arithmetic_expression_evaluation : 4 * 12 + 5 * 11 + 6^2 + 7 * 9 = 202 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l2228_222856


namespace NUMINAMATH_CALUDE_correct_statements_count_l2228_222844

-- Define the equation
def equation (x : ℝ) : Prop := (x + 1) / 4 = x - (5 * x - 1) / 12

-- Define the isosceles triangle
def isosceles_triangle (a b : ℝ) : Prop := a = 5 ∧ b = 9 ∧ a = b

-- Define the tiling shapes
inductive TilingShape
| EquilateralTriangle
| Square
| Hexagon
| Octagon

-- Define the property of being able to tile together
def can_tile_together (s1 s2 : TilingShape) : Prop :=
  match s1, s2 with
  | TilingShape.EquilateralTriangle, TilingShape.Octagon => False
  | _, _ => True

-- Define rotational symmetry
def has_rotational_symmetry (shape : String) : Prop :=
  shape = "equilateral triangle" ∨ shape = "line segment"

-- The main theorem
theorem correct_statements_count : ∃ (count : Nat),
  count = 2 ∧
  (∀ x : ℝ, equation x → 3 * (x + 1) = 12 * x - (5 * x - 1)) ∧
  (∀ a b : ℝ, isosceles_triangle a b → a + b + b = 23) ∧
  (¬ can_tile_together TilingShape.EquilateralTriangle TilingShape.Octagon) ∧
  (∀ shape : String, has_rotational_symmetry shape) :=
sorry

end NUMINAMATH_CALUDE_correct_statements_count_l2228_222844


namespace NUMINAMATH_CALUDE_purchase_price_calculation_l2228_222817

/-- Given a markup of $50, which includes 25% of cost for overhead and $12 of net profit,
    the purchase price of the article is $152. -/
theorem purchase_price_calculation (markup overhead_percentage net_profit : ℚ) 
    (h1 : markup = 50)
    (h2 : overhead_percentage = 25 / 100)
    (h3 : net_profit = 12)
    (h4 : markup = overhead_percentage * purchase_price + net_profit) :
  purchase_price = 152 :=
by sorry


end NUMINAMATH_CALUDE_purchase_price_calculation_l2228_222817


namespace NUMINAMATH_CALUDE_algebraic_simplification_l2228_222891

theorem algebraic_simplification (a : ℝ) : a * (a + 2) - 2 * a = a^2 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l2228_222891


namespace NUMINAMATH_CALUDE_parabola_equation_l2228_222894

/-- A parabola with vertex at the origin, axis of symmetry along the y-axis,
    and distance between vertex and focus equal to 6 -/
structure Parabola where
  vertex : ℝ × ℝ
  axis_of_symmetry : ℝ → ℝ
  focus_distance : ℝ
  h_vertex : vertex = (0, 0)
  h_axis : axis_of_symmetry = fun y => 0
  h_focus : focus_distance = 6

/-- The standard equation of the parabola -/
def standard_equation (p : Parabola) : Prop :=
  ∀ x y : ℝ, (x^2 = 24*y ∨ x^2 = -24*y) ↔ (x, y) ∈ {(x, y) | p.axis_of_symmetry x = y}

/-- Theorem stating that the standard equation holds for the given parabola -/
theorem parabola_equation (p : Parabola) : standard_equation p := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l2228_222894


namespace NUMINAMATH_CALUDE_range_of_m_for_root_in_interval_l2228_222839

/-- Given a function f(x) = 2x - m with a root in the interval (1, 2), 
    prove that the range of m is 2 < m < 4 -/
theorem range_of_m_for_root_in_interval 
  (f : ℝ → ℝ) 
  (m : ℝ) 
  (h1 : ∀ x, f x = 2 * x - m) 
  (h2 : ∃ x ∈ Set.Ioo 1 2, f x = 0) : 
  2 < m ∧ m < 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_for_root_in_interval_l2228_222839


namespace NUMINAMATH_CALUDE_min_value_cos_sin_l2228_222899

theorem min_value_cos_sin (θ : Real) (h : π/2 < θ ∧ θ < 3*π/2) :
  ∃ (min_val : Real), min_val = Real.sqrt 3 / 2 - 3 / 4 ∧
  ∀ (y : Real), y = Real.cos (θ/2) * (1 - Real.sin θ) → y ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_cos_sin_l2228_222899


namespace NUMINAMATH_CALUDE_family_boys_count_l2228_222832

/-- Represents a family with boys and girls -/
structure Family where
  boys : ℕ
  girls : ℕ

/-- A child in the family -/
structure Child where
  brothers : ℕ
  sisters : ℕ

/-- Defines a valid family based on the problem conditions -/
def isValidFamily (f : Family) : Prop :=
  ∃ (c1 c2 : Child),
    c1.brothers = 3 ∧ c1.sisters = 6 ∧
    c2.brothers = 4 ∧ c2.sisters = 5 ∧
    f.boys = c1.brothers + 1 ∧
    f.girls = c1.sisters + 1

theorem family_boys_count (f : Family) :
  isValidFamily f → f.boys = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_family_boys_count_l2228_222832


namespace NUMINAMATH_CALUDE_tangent_line_of_cubic_l2228_222831

/-- Given a cubic function f(x) with specific derivative conditions, 
    prove that its tangent line at x = 1 has a specific equation. -/
theorem tangent_line_of_cubic (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + b*x + 1
  let f' : ℝ → ℝ := λ x ↦ 3*x^2 + 2*a*x + b
  (f' 1 = 2*a) → (f' 2 = -b) → 
  ∃ m c : ℝ, m = -3 ∧ c = -5/2 ∧ 
    (∀ x y : ℝ, y - c = m * (x - 1) ↔ 6*x + 2*y - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_of_cubic_l2228_222831


namespace NUMINAMATH_CALUDE_abs_product_plus_four_gt_abs_sum_l2228_222840

def f (x : ℝ) := |x - 1| + |x + 1|

def M : Set ℝ := {x | f x < 4}

theorem abs_product_plus_four_gt_abs_sum {a b : ℝ} (ha : a ∈ M) (hb : b ∈ M) :
  |a * b + 4| > |a + b| := by
  sorry

end NUMINAMATH_CALUDE_abs_product_plus_four_gt_abs_sum_l2228_222840


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2228_222809

theorem polynomial_divisibility (a b c d : ℤ) : 
  (∀ x : ℤ, ∃ k : ℤ, a * x^3 + b * x^2 + c * x + d = 5 * k) →
  (∃ (ka kb kc kd : ℤ), a = 5 * ka ∧ b = 5 * kb ∧ c = 5 * kc ∧ d = 5 * kd) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2228_222809


namespace NUMINAMATH_CALUDE_plan_A_fixed_charge_l2228_222834

/-- The fixed charge for the first 4 minutes in Plan A -/
def fixed_charge : ℝ := sorry

/-- The per-minute rate after the first 4 minutes in Plan A -/
def rate_A : ℝ := 0.06

/-- The per-minute rate for Plan B -/
def rate_B : ℝ := 0.08

/-- The duration at which both plans charge the same amount -/
def equal_duration : ℝ := 18

theorem plan_A_fixed_charge :
  fixed_charge = 0.60 :=
by
  have h1 : fixed_charge + rate_A * (equal_duration - 4) = rate_B * equal_duration :=
    sorry
  sorry


end NUMINAMATH_CALUDE_plan_A_fixed_charge_l2228_222834


namespace NUMINAMATH_CALUDE_geometric_sequence_s4_l2228_222838

/-- A geometric sequence with partial sums S_n -/
structure GeometricSequence where
  S : ℕ → ℝ
  is_geometric : ∀ n : ℕ, S (n + 2) - S (n + 1) = (S (n + 1) - S n) * (S (n + 1) - S n) / (S n - S (n - 1))

/-- Theorem: In a geometric sequence where S_2 = 7 and S_6 = 91, S_4 = 28 -/
theorem geometric_sequence_s4 (seq : GeometricSequence) 
  (h2 : seq.S 2 = 7) (h6 : seq.S 6 = 91) : seq.S 4 = 28 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_s4_l2228_222838


namespace NUMINAMATH_CALUDE_equation_solutions_l2228_222821

theorem equation_solutions :
  (∀ x : ℝ, (3 * x - 1)^2 = 9 ↔ x = 4/3 ∨ x = -2/3) ∧
  (∀ x : ℝ, x * (2 * x - 4) = (2 - x)^2 ↔ x = 2 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2228_222821


namespace NUMINAMATH_CALUDE_smallest_coefficients_in_binomial_expansion_l2228_222857

theorem smallest_coefficients_in_binomial_expansion :
  let n : ℕ := 8
  let coefficients : Fin (n + 1) → ℕ := λ k => Nat.choose n k.val
  let fourth_term := coefficients ⟨3, by norm_num⟩
  let sixth_term := coefficients ⟨5, by norm_num⟩
  (∀ k : Fin (n + 1), fourth_term ≤ coefficients k) ∧
  (∀ k : Fin (n + 1), sixth_term ≤ coefficients k) ∧
  (∀ k : Fin (n + 1), coefficients k = fourth_term ∨ coefficients k = sixth_term ∨ coefficients k > fourth_term) :=
by sorry

end NUMINAMATH_CALUDE_smallest_coefficients_in_binomial_expansion_l2228_222857


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2228_222822

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), 2 * a * x - b * y + 2 = 0 ∧
                 x^2 + y^2 + 2*x - 4*y + 1 = 0 ∧
                 (∃ (x1 y1 x2 y2 : ℝ),
                    2 * a * x1 - b * y1 + 2 = 0 ∧
                    x1^2 + y1^2 + 2*x1 - 4*y1 + 1 = 0 ∧
                    2 * a * x2 - b * y2 + 2 = 0 ∧
                    x2^2 + y2^2 + 2*x2 - 4*y2 + 1 = 0 ∧
                    (x2 - x1)^2 + (y2 - y1)^2 = 16)) →
  (∀ c d : ℝ, c > 0 → d > 0 →
    (∃ (x y : ℝ), 2 * c * x - d * y + 2 = 0 ∧
                   x^2 + y^2 + 2*x - 4*y + 1 = 0) →
    1/a + 1/b ≤ 1/c + 1/d) ∧
  (1/a + 1/b = 4) :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2228_222822


namespace NUMINAMATH_CALUDE_marks_lost_is_one_l2228_222818

/-- Represents an examination with given parameters -/
structure Examination where
  total_questions : ℕ
  correct_score : ℕ
  total_score : ℤ
  correct_answers : ℕ

/-- Calculates the marks lost for each wrong answer in the examination -/
def marks_lost_per_wrong_answer (exam : Examination) : ℚ :=
  let wrong_answers := exam.total_questions - exam.correct_answers
  let total_correct_score := exam.correct_score * exam.correct_answers
  (total_correct_score - exam.total_score) / wrong_answers

/-- Theorem stating that the marks lost for each wrong answer is 1 -/
theorem marks_lost_is_one (exam : Examination) 
    (h1 : exam.total_questions = 60)
    (h2 : exam.correct_score = 4)
    (h3 : exam.total_score = 110)
    (h4 : exam.correct_answers = 34) :
  marks_lost_per_wrong_answer exam = 1 := by
  sorry

#eval marks_lost_per_wrong_answer { 
  total_questions := 60, 
  correct_score := 4, 
  total_score := 110, 
  correct_answers := 34 
}

end NUMINAMATH_CALUDE_marks_lost_is_one_l2228_222818


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2228_222872

-- Define the sets A and B
def A : Set ℝ := {x | x + 1 < 0}
def B : Set ℝ := {x | x - 3 < 0}

-- Define the complement of A in the universal set ℝ
def C_U_A : Set ℝ := {x | ¬ (x ∈ A)}

-- State the theorem
theorem complement_A_intersect_B :
  (C_U_A ∩ B) = {x : ℝ | -1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2228_222872


namespace NUMINAMATH_CALUDE_root_property_l2228_222859

theorem root_property (m : ℝ) : m^2 - m - 1 = 0 → m^2 - m = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_property_l2228_222859


namespace NUMINAMATH_CALUDE_find_c_l2228_222888

def f (a c x : ℝ) : ℝ := a * x^3 + c

theorem find_c (a c : ℝ) :
  (∃ x, x ∈ Set.Icc 1 2 ∧ ∀ y ∈ Set.Icc 1 2, f a c y ≤ f a c x) →
  (deriv (f a c) 1 = 6) →
  (∃ x, x ∈ Set.Icc 1 2 ∧ f a c x = 20) →
  c = 4 := by
sorry

end NUMINAMATH_CALUDE_find_c_l2228_222888


namespace NUMINAMATH_CALUDE_exist_three_numbers_with_equal_sum_l2228_222886

-- Define the sum of digits function
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

-- Statement of the theorem
theorem exist_three_numbers_with_equal_sum :
  ∃ (m n p : ℕ), m ≠ n ∧ n ≠ p ∧ m ≠ p ∧
    m + sumOfDigits m = n + sumOfDigits n ∧
    n + sumOfDigits n = p + sumOfDigits p :=
sorry

end NUMINAMATH_CALUDE_exist_three_numbers_with_equal_sum_l2228_222886


namespace NUMINAMATH_CALUDE_distance_to_x_axis_l2228_222815

def point : ℝ × ℝ := (2, -3)

theorem distance_to_x_axis : abs (point.2) = 3 := by sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_l2228_222815


namespace NUMINAMATH_CALUDE_triangle_area_in_square_l2228_222862

/-- The area of triangle ABC in a 12x12 square with specific point locations -/
theorem triangle_area_in_square : 
  let square_side : ℝ := 12
  let point_A : ℝ × ℝ := (square_side / 2, square_side)
  let point_B : ℝ × ℝ := (0, square_side / 4)
  let point_C : ℝ × ℝ := (square_side, square_side / 4)
  let triangle_area := (1 / 2) * 
    (|((point_C.1 - point_A.1) * (point_B.2 - point_A.2) - 
       (point_B.1 - point_A.1) * (point_C.2 - point_A.2))|)
  triangle_area = (27 * Real.sqrt 10) / 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_in_square_l2228_222862


namespace NUMINAMATH_CALUDE_circle_radius_l2228_222850

/-- The radius of the circle described by x^2 + y^2 - 4x + 6y = 0 is √13 -/
theorem circle_radius (x y : ℝ) : 
  (∀ x y, x^2 + y^2 - 4*x + 6*y = 0) → 
  ∃ r : ℝ, r = Real.sqrt 13 ∧ ∀ x y, (x - 2)^2 + (y + 3)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l2228_222850


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l2228_222871

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y = 1 → a + 2*b ≤ x + 2*y :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l2228_222871


namespace NUMINAMATH_CALUDE_dave_tickets_used_l2228_222895

/-- Given that Dave had 13 tickets initially and has 7 tickets left after buying toys,
    prove that he used 6 tickets to buy toys. -/
theorem dave_tickets_used (initial : ℕ) (left : ℕ) (used : ℕ) 
    (h1 : initial = 13) 
    (h2 : left = 7) 
    (h3 : used = initial - left) : 
  used = 6 := by
  sorry

end NUMINAMATH_CALUDE_dave_tickets_used_l2228_222895


namespace NUMINAMATH_CALUDE_overtime_pay_ratio_l2228_222849

/-- Calculates the ratio of overtime pay rate to regular pay rate -/
theorem overtime_pay_ratio (regular_rate : ℚ) (regular_hours : ℚ) (total_pay : ℚ) (overtime_hours : ℚ)
  (h1 : regular_rate = 3)
  (h2 : regular_hours = 40)
  (h3 : total_pay = 186)
  (h4 : overtime_hours = 11) :
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := total_pay - regular_pay
  let overtime_rate := overtime_pay / overtime_hours
  overtime_rate / regular_rate = 2 := by
  sorry


end NUMINAMATH_CALUDE_overtime_pay_ratio_l2228_222849


namespace NUMINAMATH_CALUDE_escalator_walking_rate_l2228_222800

/-- Given an escalator moving upwards at a certain rate with a specified length,
    prove that a person walking on it at a certain rate will take a specific time
    to cover the entire length. -/
theorem escalator_walking_rate
  (escalator_speed : ℝ)
  (escalator_length : ℝ)
  (time_taken : ℝ)
  (h1 : escalator_speed = 12)
  (h2 : escalator_length = 196)
  (h3 : time_taken = 14)
  : ∃ (walking_rate : ℝ),
    escalator_length = (walking_rate + escalator_speed) * time_taken ∧
    walking_rate = 2 :=
by sorry

end NUMINAMATH_CALUDE_escalator_walking_rate_l2228_222800


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2228_222816

theorem quadratic_roots_property (m n : ℝ) : 
  m^2 + 2*m - 5 = 0 → n^2 + 2*n - 5 = 0 → m^2 + m*n + 2*m = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2228_222816


namespace NUMINAMATH_CALUDE_find_x_l2228_222880

theorem find_x : ∃ x : ℝ, 5.76 = 0.12 * (0.40 * x) ∧ x = 120 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l2228_222880


namespace NUMINAMATH_CALUDE_equivalence_of_conditions_l2228_222855

theorem equivalence_of_conditions (x : ℝ) : (1 < x ∧ x < 3) ↔ |x - 2| < 1 := by
  sorry

end NUMINAMATH_CALUDE_equivalence_of_conditions_l2228_222855


namespace NUMINAMATH_CALUDE_find_f_2022_l2228_222812

/-- A function satisfying the given condition -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f ((a + 2 * b) / 3) = (f a + 2 * f b) / 3

/-- The theorem to prove -/
theorem find_f_2022 (f : ℝ → ℝ) (h : special_function f) (h1 : f 1 = 5) (h4 : f 4 = 2) :
  f 2022 = -2016 := by
  sorry


end NUMINAMATH_CALUDE_find_f_2022_l2228_222812


namespace NUMINAMATH_CALUDE_equation_solution_l2228_222803

theorem equation_solution : ∃! x : ℝ, (567.23 - x) * 45.7 + (64.89 / 11.5)^3 - 2.78 = 18756.120 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2228_222803


namespace NUMINAMATH_CALUDE_positive_integer_solutions_l2228_222843

theorem positive_integer_solutions :
  ∀ a b c : ℕ+,
  a < b →
  a < 4 * c →
  b * c^3 ≤ a * c^3 + b →
  ((a = 7 ∧ b = 8 ∧ c = 2) ∨
   (a = 1 ∧ b > 1 ∧ c = 1) ∨
   (a = 2 ∧ b > 2 ∧ c = 1) ∨
   (a = 3 ∧ b > 3 ∧ c = 1)) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_l2228_222843


namespace NUMINAMATH_CALUDE_last_digit_101_power_100_l2228_222851

theorem last_digit_101_power_100 : 101^100 ≡ 1 [ZMOD 10] := by sorry

end NUMINAMATH_CALUDE_last_digit_101_power_100_l2228_222851


namespace NUMINAMATH_CALUDE_article_price_proof_l2228_222861

-- Define the selling price and loss percentage
def selling_price : ℝ := 800
def loss_percentage : ℝ := 33.33333333333333

-- Define the original price
def original_price : ℝ := 1200

-- Theorem statement
theorem article_price_proof :
  (selling_price = (1 - loss_percentage / 100) * original_price) → 
  (original_price = 1200) := by
  sorry

end NUMINAMATH_CALUDE_article_price_proof_l2228_222861


namespace NUMINAMATH_CALUDE_events_related_confidence_l2228_222889

-- Define the confidence level
def confidence_level : ℝ := 0.95

-- Define the critical value for 95% confidence
def critical_value : ℝ := 3.841

-- Define the relationship between events A and B
def events_related (K : ℝ) : Prop := K^2 > critical_value

-- Theorem statement
theorem events_related_confidence (K : ℝ) :
  events_related K ↔ confidence_level = 0.95 :=
sorry

end NUMINAMATH_CALUDE_events_related_confidence_l2228_222889


namespace NUMINAMATH_CALUDE_juice_mixture_problem_l2228_222842

/-- Given two juices p and v mixed into two smoothies a and y, prove the amount of p in a. -/
theorem juice_mixture_problem (p_total v_total : ℚ) (p_a v_a p_y v_y : ℚ) :
  p_total = 24 ∧ v_total = 25 ∧  -- Total amounts of juices
  p_a + p_y = p_total ∧ v_a + v_y = v_total ∧  -- Conservation of juices
  p_a / v_a = 4 / 1 ∧  -- Ratio in smoothie a
  p_y / v_y = 1 / 5  -- Ratio in smoothie y
  → p_a = 20 := by sorry

end NUMINAMATH_CALUDE_juice_mixture_problem_l2228_222842


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_value_l2228_222823

/-- Given a curve y = ax - ln(x + 1), prove that if its tangent line at (0, 0) is y = 2x, then a = 3 -/
theorem tangent_line_implies_a_value (a : ℝ) : 
  (∀ x, ∃ y, y = a * x - Real.log (x + 1)) →  -- Curve equation
  (∃ m, ∀ x, 2 * x = m * x) →                 -- Tangent line at (0, 0) is y = 2x
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_implies_a_value_l2228_222823


namespace NUMINAMATH_CALUDE_train_length_calculation_l2228_222879

theorem train_length_calculation (train_speed : Real) (platform_length : Real) (crossing_time : Real) :
  train_speed = 55 * 1000 / 3600 →
  platform_length = 300 →
  crossing_time = 35.99712023038157 →
  let total_distance := train_speed * crossing_time
  let train_length := total_distance - platform_length
  train_length = 249.9999999999999 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2228_222879


namespace NUMINAMATH_CALUDE_josh_marbles_difference_l2228_222848

/-- Given Josh's marble collection scenario, prove the difference between lost and found marbles. -/
theorem josh_marbles_difference (initial : ℕ) (found : ℕ) (lost : ℕ) 
  (h1 : initial = 15) 
  (h2 : found = 9) 
  (h3 : lost = 23) : 
  lost - found = 14 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_difference_l2228_222848


namespace NUMINAMATH_CALUDE_min_value_xy_l2228_222846

theorem min_value_xy (x y : ℝ) (hx : x > 1) (hy : y > 1) 
  (h_geom : ∃ r : ℝ, (Real.log x) * r = (1/2) ∧ (1/2) * r = Real.log y) : 
  (∀ a b : ℝ, a > 1 → b > 1 → 
    (∃ r : ℝ, (Real.log a) * r = (1/2) ∧ (1/2) * r = Real.log b) → 
    x * y ≤ a * b) → 
  x * y = Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_xy_l2228_222846


namespace NUMINAMATH_CALUDE_adam_current_age_l2228_222806

/-- Adam's current age -/
def adam_age : ℕ := sorry

/-- Tom's current age -/
def tom_age : ℕ := 12

/-- Years into the future -/
def years_future : ℕ := 12

/-- Combined age in the future -/
def combined_future_age : ℕ := 44

theorem adam_current_age :
  adam_age = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_adam_current_age_l2228_222806


namespace NUMINAMATH_CALUDE_polly_mirror_rate_l2228_222865

/-- Polly's tweeting behavior -/
structure PollyTweets where
  happy_rate : ℕ      -- tweets per minute when happy
  hungry_rate : ℕ     -- tweets per minute when hungry
  mirror_rate : ℕ     -- tweets per minute when watching mirror
  happy_time : ℕ      -- time spent being happy (in minutes)
  hungry_time : ℕ     -- time spent being hungry (in minutes)
  mirror_time : ℕ     -- time spent watching mirror (in minutes)
  total_tweets : ℕ    -- total number of tweets

/-- Theorem about Polly's tweeting rate when watching the mirror -/
theorem polly_mirror_rate (p : PollyTweets)
  (h1 : p.happy_rate = 18)
  (h2 : p.hungry_rate = 4)
  (h3 : p.happy_time = 20)
  (h4 : p.hungry_time = 20)
  (h5 : p.mirror_time = 20)
  (h6 : p.total_tweets = 1340)
  (h7 : p.total_tweets = p.happy_rate * p.happy_time + p.hungry_rate * p.hungry_time + p.mirror_rate * p.mirror_time) :
  p.mirror_rate = 45 := by
  sorry

end NUMINAMATH_CALUDE_polly_mirror_rate_l2228_222865


namespace NUMINAMATH_CALUDE_competition_score_difference_l2228_222853

def score_60_percent : Real := 0.12
def score_85_percent : Real := 0.20
def score_95_percent : Real := 0.38
def score_105_percent : Real := 1 - (score_60_percent + score_85_percent + score_95_percent)

def mean_score : Real :=
  score_60_percent * 60 + score_85_percent * 85 + score_95_percent * 95 + score_105_percent * 105

def median_score : Real := 95

theorem competition_score_difference : median_score - mean_score = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_competition_score_difference_l2228_222853


namespace NUMINAMATH_CALUDE_algebra_test_correct_percentage_l2228_222897

theorem algebra_test_correct_percentage (x : ℕ) (h : x > 0) :
  let total_problems := 5 * x
  let missed_problems := x
  let correct_problems := total_problems - missed_problems
  (correct_problems : ℚ) / (total_problems : ℚ) * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_algebra_test_correct_percentage_l2228_222897


namespace NUMINAMATH_CALUDE_ladder_problem_l2228_222802

theorem ladder_problem (ladder_length height base : ℝ) :
  ladder_length = 15 ∧ height = 12 ∧ ladder_length ^ 2 = height ^ 2 + base ^ 2 →
  base = 9 := by
sorry

end NUMINAMATH_CALUDE_ladder_problem_l2228_222802


namespace NUMINAMATH_CALUDE_a_values_l2228_222805

def A : Set ℝ := {x | x^2 + 3*x + 2 = 0}

def B (a : ℝ) : Set ℝ := {x | a*x - 2 = 0}

theorem a_values (a : ℝ) : A ∪ B a = A → a = 0 ∨ a = -1 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_a_values_l2228_222805


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l2228_222866

theorem smallest_part_of_proportional_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 124)
  (h_prop : a = 2 ∧ b = 1/2 ∧ c = 1/4) :
  let x := total / (a + b + c)
  min (a * x) (min (b * x) (c * x)) = 124 / 11 := by
sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l2228_222866


namespace NUMINAMATH_CALUDE_multiples_of_five_average_l2228_222875

theorem multiples_of_five_average (n : ℕ) : 
  (((n : ℝ) / 2) * (5 + 5 * n)) / n = 55 → n = 21 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_five_average_l2228_222875


namespace NUMINAMATH_CALUDE_percentage_increase_l2228_222837

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 110 → final = 165 → (final - initial) / initial * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l2228_222837


namespace NUMINAMATH_CALUDE_A_sufficient_not_necessary_l2228_222833

-- Define propositions A and B
def A (x y : ℝ) : Prop := x + y ≠ 8
def B (x y : ℝ) : Prop := x ≠ 2 ∨ y ≠ 6

-- Theorem statement
theorem A_sufficient_not_necessary :
  (∀ x y : ℝ, A x y → B x y) ∧
  ¬(∀ x y : ℝ, B x y → A x y) :=
by sorry

end NUMINAMATH_CALUDE_A_sufficient_not_necessary_l2228_222833


namespace NUMINAMATH_CALUDE_parabola_line_intersection_perpendicular_l2228_222883

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define the line
def line (x y : ℝ) : Prop := y = x - 2

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem parabola_line_intersection_perpendicular :
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
  line A.1 A.2 ∧ line B.1 B.2 →
  (A.1 * B.1 + A.2 * B.2 = 0) := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_perpendicular_l2228_222883


namespace NUMINAMATH_CALUDE_magic_square_y_value_l2228_222884

/-- Represents a 3x3 magic square -/
def MagicSquare (a b c d e f g h i : ℚ) : Prop :=
  a + b + c = d + e + f ∧
  d + e + f = g + h + i ∧
  a + d + g = b + e + h ∧
  b + e + h = c + f + i ∧
  a + e + i = c + e + g

theorem magic_square_y_value :
  ∀ (y a b c d e : ℚ),
  MagicSquare y 7 24 8 a b c d e →
  y = 39.5 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_y_value_l2228_222884


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2228_222873

theorem midpoint_coordinate_sum : 
  let p₁ : ℝ × ℝ := (10, -3)
  let p₂ : ℝ × ℝ := (-4, 7)
  let midpoint := ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)
  (midpoint.1 + midpoint.2 : ℝ) = 5 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2228_222873


namespace NUMINAMATH_CALUDE_edward_lawn_problem_l2228_222814

/-- The number of dollars Edward earns per lawn -/
def dollars_per_lawn : ℕ := 4

/-- The number of lawns Edward forgot to mow -/
def forgotten_lawns : ℕ := 9

/-- The amount of money Edward actually earned -/
def actual_earnings : ℕ := 32

/-- The original number of lawns Edward had to mow -/
def original_lawns : ℕ := 17

theorem edward_lawn_problem :
  dollars_per_lawn * (original_lawns - forgotten_lawns) = actual_earnings :=
by sorry

end NUMINAMATH_CALUDE_edward_lawn_problem_l2228_222814


namespace NUMINAMATH_CALUDE_division_theorem_l2228_222887

/-- The dividend polynomial -/
def f (x : ℝ) : ℝ := 3*x^5 - 2*x^3 + 5*x - 9

/-- The divisor polynomial -/
def g (x : ℝ) : ℝ := x^2 - 3*x + 2

/-- The proposed remainder polynomial -/
def r (x : ℝ) : ℝ := 92*x - 95

/-- Statement: The remainder when f(x) is divided by g(x) is r(x) -/
theorem division_theorem : ∃ q : ℝ → ℝ, ∀ x, f x = g x * q x + r x := by
  sorry

end NUMINAMATH_CALUDE_division_theorem_l2228_222887


namespace NUMINAMATH_CALUDE_max_sum_xy_l2228_222801

-- Define the constraint function
def constraint (x y : ℝ) : Prop :=
  Real.log y / Real.log ((x^2 + y^2) / 2) ≥ 1 ∧ (x ≠ 0 ∨ y ≠ 0) ∧ x^2 + y^2 ≠ 2

-- State the theorem
theorem max_sum_xy :
  ∃ (max : ℝ), max = 1 + Real.sqrt 2 ∧
  (∀ x y : ℝ, constraint x y → x + y ≤ max) ∧
  (∃ x y : ℝ, constraint x y ∧ x + y = max) :=
sorry

end NUMINAMATH_CALUDE_max_sum_xy_l2228_222801


namespace NUMINAMATH_CALUDE_walking_sequence_intersection_l2228_222878

/-- A walking sequence is a sequence of integers where each term differs from the previous by ±1. -/
def IsWalkingSequence (a : Fin 2016 → ℤ) : Prop :=
  ∀ i : Fin 2015, a (i + 1) = a i + 1 ∨ a (i + 1) = a i - 1

/-- The sequence b as defined in the problem -/
def b : Fin 2016 → ℤ
  | ⟨i, h⟩ => if i < 1009 then i + 1 else 2018 - i

/-- The main theorem statement -/
theorem walking_sequence_intersection :
  ∃ (a : Fin 2016 → ℤ), IsWalkingSequence a ∧
  (∀ i, 1 ≤ a i ∧ a i ≤ 1010) →
  ∃ j, a j = b j :=
sorry

end NUMINAMATH_CALUDE_walking_sequence_intersection_l2228_222878


namespace NUMINAMATH_CALUDE_anns_age_l2228_222869

theorem anns_age (A B : ℕ) : 
  A + B = 52 → 
  B = (2 * B - A / 3) → 
  A = 39 := by sorry

end NUMINAMATH_CALUDE_anns_age_l2228_222869


namespace NUMINAMATH_CALUDE_exam_pupils_count_l2228_222811

theorem exam_pupils_count :
  ∀ (n : ℕ) (total_marks : ℕ),
    n > 4 →
    total_marks = 39 * n →
    (total_marks - 71) / (n - 4) = 44 →
    n = 21 := by
  sorry

end NUMINAMATH_CALUDE_exam_pupils_count_l2228_222811


namespace NUMINAMATH_CALUDE_job_completion_time_l2228_222881

/-- Represents the number of days needed to complete a job given initial and additional workers -/
def days_to_complete_job (initial_workers : ℕ) (initial_days : ℕ) (total_work_days : ℕ) 
  (days_before_joining : ℕ) (additional_workers : ℕ) : ℕ :=
  let total_workers := initial_workers + additional_workers
  let work_done := initial_workers * days_before_joining
  let remaining_work := total_work_days - work_done
  days_before_joining + (remaining_work + total_workers - 1) / total_workers

/-- Theorem stating that under the given conditions, the job will be completed in 6 days -/
theorem job_completion_time :
  days_to_complete_job 6 8 48 3 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2228_222881


namespace NUMINAMATH_CALUDE_M_equals_set_l2228_222841

def M (x y z : ℝ) : Set ℝ :=
  { w | ∃ (x y z : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
    w = (x / abs x) + (y / abs y) + (z / abs z) + (abs (x * y * z) / (x * y * z)) }

theorem M_equals_set (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  M x y z = {4, -4, 0} := by
  sorry

end NUMINAMATH_CALUDE_M_equals_set_l2228_222841


namespace NUMINAMATH_CALUDE_total_fruits_l2228_222807

/-- The number of pieces of fruit in three buckets -/
structure FruitBuckets where
  A : ℕ
  B : ℕ
  C : ℕ

/-- The conditions for the fruit bucket problem -/
def fruit_bucket_conditions (fb : FruitBuckets) : Prop :=
  fb.A = fb.B + 4 ∧ fb.B = fb.C + 3 ∧ fb.C = 9

/-- The theorem stating that the total number of fruits in all buckets is 37 -/
theorem total_fruits (fb : FruitBuckets) 
  (h : fruit_bucket_conditions fb) : fb.A + fb.B + fb.C = 37 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_l2228_222807


namespace NUMINAMATH_CALUDE_square_sum_inequality_l2228_222830

theorem square_sum_inequality (x y : ℝ) :
  x^2 + y^2 ≤ 2*(x + y - 1) → x = 1 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l2228_222830


namespace NUMINAMATH_CALUDE_problem_solution_l2228_222804

/-- Predicate to check if a number is divisible by another -/
def is_divisible (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

/-- Predicate to check if a number is prime -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬is_divisible n d

/-- The four statements in the problem -/
def statement1 (a b : ℕ) : Prop := is_divisible (a^2 + 6*a + 8) b
def statement2 (a b : ℕ) : Prop := a^2 + a*b - 6*b^2 - 15*b - 9 = 0
def statement3 (a b : ℕ) : Prop := is_divisible (a + 2*b + 2) 4
def statement4 (a b : ℕ) : Prop := is_prime (a + 6*b + 2)

/-- Predicate to check if exactly three out of four statements are true -/
def three_true (a b : ℕ) : Prop :=
  (statement1 a b ∧ statement2 a b ∧ statement3 a b ∧ ¬statement4 a b) ∨
  (statement1 a b ∧ statement2 a b ∧ ¬statement3 a b ∧ statement4 a b) ∨
  (statement1 a b ∧ ¬statement2 a b ∧ statement3 a b ∧ statement4 a b) ∨
  (¬statement1 a b ∧ statement2 a b ∧ statement3 a b ∧ statement4 a b)

theorem problem_solution :
  ∀ a b : ℕ, three_true a b ↔ ((a = 5 ∧ b = 1) ∨ (a = 17 ∧ b = 7)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2228_222804


namespace NUMINAMATH_CALUDE_tan_three_expression_zero_l2228_222820

theorem tan_three_expression_zero (θ : Real) (h : Real.tan θ = 3) :
  (1 + Real.cos θ) / Real.sin θ - Real.sin θ / (1 - Real.cos θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_three_expression_zero_l2228_222820


namespace NUMINAMATH_CALUDE_second_catch_size_l2228_222808

theorem second_catch_size (tagged_initial : ℕ) (tagged_second : ℕ) (total_fish : ℕ) :
  tagged_initial = 50 →
  tagged_second = 2 →
  total_fish = 1250 →
  (tagged_second : ℚ) / (tagged_initial : ℚ) = (tagged_second : ℚ) / x →
  x = 50 :=
by sorry

end NUMINAMATH_CALUDE_second_catch_size_l2228_222808


namespace NUMINAMATH_CALUDE_system_solution_l2228_222892

theorem system_solution : 
  ∃ (x y : ℚ), 2 * x + 3 * y = 1 ∧ 3 * x - 6 * y = 7 ∧ x = 9/7 ∧ y = -11/21 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2228_222892


namespace NUMINAMATH_CALUDE_book_arrangement_count_l2228_222874

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def arrangeBooksCount (mathBooks : ℕ) (englishBooks : ℕ) : ℕ :=
  2 * factorial mathBooks * factorial englishBooks

theorem book_arrangement_count :
  arrangeBooksCount 3 5 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l2228_222874


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2228_222858

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔ 
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2228_222858


namespace NUMINAMATH_CALUDE_odd_digits_base4_345_l2228_222876

/-- Converts a natural number to its base-4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of odd digits in the base-4 representation of 345 is 4 -/
theorem odd_digits_base4_345 : countOddDigits (toBase4 345) = 4 :=
  sorry

end NUMINAMATH_CALUDE_odd_digits_base4_345_l2228_222876
