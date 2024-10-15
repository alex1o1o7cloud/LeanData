import Mathlib

namespace NUMINAMATH_CALUDE_diamond_property_false_l1359_135983

/-- The diamond operation for real numbers -/
def diamond (x y : ℝ) : ℝ := |x + y - 1|

/-- The statement that is false -/
theorem diamond_property_false : ∃ x y : ℝ, 2 * (diamond x y) ≠ diamond (2 * x) (2 * y) := by
  sorry

end NUMINAMATH_CALUDE_diamond_property_false_l1359_135983


namespace NUMINAMATH_CALUDE_parabola_shift_l1359_135972

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift (p : Parabola) (h v : ℝ) : Parabola where
  f := fun x => p.f (x + h) + v

theorem parabola_shift :
  let p : Parabola := ⟨fun x => x^2⟩
  let shifted := shift p 2 (-5)
  ∀ x, shifted.f x = (x + 2)^2 - 5 := by sorry

end NUMINAMATH_CALUDE_parabola_shift_l1359_135972


namespace NUMINAMATH_CALUDE_student_arrangement_count_l1359_135951

/-- The number of male students -/
def num_male : ℕ := 3

/-- The number of female students -/
def num_female : ℕ := 2

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- The number of positions available for female students (not at the ends) -/
def female_positions : ℕ := total_students - 2

/-- The number of ways to arrange female students in available positions -/
def female_arrangements : ℕ := Nat.choose female_positions num_female

/-- The number of ways to arrange the remaining male students -/
def male_arrangements : ℕ := Nat.factorial num_male

/-- The total number of arrangements -/
def total_arrangements : ℕ := female_arrangements * male_arrangements

theorem student_arrangement_count :
  total_arrangements = 36 := by sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l1359_135951


namespace NUMINAMATH_CALUDE_lychee_ratio_proof_l1359_135981

theorem lychee_ratio_proof (total : ℕ) (remaining : ℕ) : 
  total = 500 →
  remaining = 100 →
  ∃ (sold : ℕ) (taken_home : ℕ),
    sold + taken_home = total ∧
    (2 * remaining : ℚ) = (2 / 5 : ℚ) * taken_home ∧
    2 * sold = total :=
by sorry

end NUMINAMATH_CALUDE_lychee_ratio_proof_l1359_135981


namespace NUMINAMATH_CALUDE_max_section_area_is_two_l1359_135900

/-- Represents a cone with its lateral surface unfolded into a sector -/
structure UnfoldedCone where
  radius : ℝ
  centralAngle : ℝ

/-- Calculates the maximum area of a section determined by two generatrices of the cone -/
def maxSectionArea (cone : UnfoldedCone) : ℝ :=
  sorry

/-- Theorem stating that for a cone with lateral surface unfolded into a sector
    with radius 2 and central angle 5π/3, the maximum section area is 2 -/
theorem max_section_area_is_two :
  let cone : UnfoldedCone := ⟨2, 5 * Real.pi / 3⟩
  maxSectionArea cone = 2 :=
sorry

end NUMINAMATH_CALUDE_max_section_area_is_two_l1359_135900


namespace NUMINAMATH_CALUDE_factorial_square_root_problem_l1359_135943

theorem factorial_square_root_problem : (Real.sqrt (Nat.factorial 5 * Nat.factorial 4))^2 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_factorial_square_root_problem_l1359_135943


namespace NUMINAMATH_CALUDE_sugar_mixture_percentage_l1359_135961

/-- Given two solutions, where one fourth of the first solution is replaced by the second solution,
    resulting in a mixture that is 17% sugar, and the second solution is 38% sugar,
    prove that the first solution was 10% sugar. -/
theorem sugar_mixture_percentage (first_solution second_solution final_mixture : ℝ) 
    (h1 : 3/4 * first_solution + 1/4 * second_solution = final_mixture)
    (h2 : final_mixture = 17)
    (h3 : second_solution = 38) :
    first_solution = 10 := by
  sorry

end NUMINAMATH_CALUDE_sugar_mixture_percentage_l1359_135961


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l1359_135907

theorem polygon_interior_angles (n : ℕ) : 
  180 * (n - 2) = 1440 → n = 10 := by sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l1359_135907


namespace NUMINAMATH_CALUDE_sugar_per_cookie_l1359_135995

theorem sugar_per_cookie (initial_cookies : ℕ) (initial_sugar_per_cookie : ℚ) 
  (new_cookies : ℕ) (total_sugar : ℚ) :
  initial_cookies = 50 →
  initial_sugar_per_cookie = 1 / 10 →
  new_cookies = 25 →
  total_sugar = initial_cookies * initial_sugar_per_cookie →
  total_sugar / new_cookies = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_sugar_per_cookie_l1359_135995


namespace NUMINAMATH_CALUDE_root_equation_a_value_l1359_135939

theorem root_equation_a_value (a b : ℚ) : 
  ((-2 : ℝ) - 5 * Real.sqrt 3)^3 + a * ((-2 : ℝ) - 5 * Real.sqrt 3)^2 + 
  b * ((-2 : ℝ) - 5 * Real.sqrt 3) - 48 = 0 → a = 4 := by
sorry

end NUMINAMATH_CALUDE_root_equation_a_value_l1359_135939


namespace NUMINAMATH_CALUDE_odd_function_sum_l1359_135967

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum (f : ℝ → ℝ) (h_odd : is_odd f) (h_neg : ∀ x, x < 0 → f x = x + 2) :
  f 0 + f 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_l1359_135967


namespace NUMINAMATH_CALUDE_no_intersection_at_vertex_l1359_135909

/-- The line equation y = x + b -/
def line (x b : ℝ) : ℝ := x + b

/-- The parabola equation y = x^2 + b^2 + 1 -/
def parabola (x b : ℝ) : ℝ := x^2 + b^2 + 1

/-- The x-coordinate of the vertex of the parabola -/
def vertex_x : ℝ := 0

/-- Theorem: There are no real values of b for which the line y = x + b passes through the vertex of the parabola y = x^2 + b^2 + 1 -/
theorem no_intersection_at_vertex :
  ¬∃ b : ℝ, line vertex_x b = parabola vertex_x b := by sorry

end NUMINAMATH_CALUDE_no_intersection_at_vertex_l1359_135909


namespace NUMINAMATH_CALUDE_inequality_condition_l1359_135958

theorem inequality_condition :
  (∀ a b c d : ℝ, a > b ∧ c > d → a + c > b + d) ∧
  (∃ a b c d : ℝ, a + c > b + d ∧ ¬(a > b ∧ c > d)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l1359_135958


namespace NUMINAMATH_CALUDE_man_pants_count_l1359_135985

theorem man_pants_count (t_shirts : ℕ) (total_outfits : ℕ) (pants : ℕ) : 
  t_shirts = 8 → total_outfits = 72 → total_outfits = t_shirts * pants → pants = 9 := by
sorry

end NUMINAMATH_CALUDE_man_pants_count_l1359_135985


namespace NUMINAMATH_CALUDE_system_of_inequalities_l1359_135931

theorem system_of_inequalities (x : ℝ) : 
  3 * (x + 1) < 4 * x + 5 → 2 * x > (x + 6) / 2 → x > 2 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_l1359_135931


namespace NUMINAMATH_CALUDE_choir_arrangement_l1359_135911

theorem choir_arrangement (n : ℕ) : 
  (∃ k : ℕ, n = k^2 + 11) ∧ 
  (∃ c : ℕ, n = c * (c + 5)) →
  n ≤ 300 :=
by sorry

end NUMINAMATH_CALUDE_choir_arrangement_l1359_135911


namespace NUMINAMATH_CALUDE_hotdog_sales_l1359_135945

theorem hotdog_sales (small_hotdogs : ℕ) (total_hotdogs : ℕ) (large_hotdogs : ℕ)
  (h1 : small_hotdogs = 58)
  (h2 : total_hotdogs = 79)
  (h3 : total_hotdogs = small_hotdogs + large_hotdogs) :
  large_hotdogs = 21 := by
  sorry

end NUMINAMATH_CALUDE_hotdog_sales_l1359_135945


namespace NUMINAMATH_CALUDE_swap_values_l1359_135947

theorem swap_values (a b : ℕ) (h1 : a = 3) (h2 : b = 2) :
  ∃ c : ℕ, (c = b) ∧ (b = a) ∧ (a = c) → a = 2 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_swap_values_l1359_135947


namespace NUMINAMATH_CALUDE_total_hamburger_combinations_l1359_135903

/-- The number of different hamburger combinations -/
def hamburger_combinations (num_buns num_condiments num_patty_choices : ℕ) : ℕ :=
  num_buns * (2 ^ num_condiments) * num_patty_choices

/-- Theorem stating the total number of different hamburger combinations -/
theorem total_hamburger_combinations :
  hamburger_combinations 3 9 3 = 4608 := by
  sorry

end NUMINAMATH_CALUDE_total_hamburger_combinations_l1359_135903


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1359_135973

def A : Set ℕ := {1, 2, 9}
def B : Set ℕ := {1, 7}

theorem intersection_of_A_and_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1359_135973


namespace NUMINAMATH_CALUDE_intersection_y_intercept_sum_l1359_135937

/-- Given two lines that intersect at (3,6), prove that the sum of their y-intercepts is 6 -/
theorem intersection_y_intercept_sum (a b : ℝ) : 
  (∀ x y : ℝ, x = (1/3)*y + a ∧ y = (1/3)*x + b → (x = 3 ∧ y = 6)) → 
  a + b = 6 := by
sorry

end NUMINAMATH_CALUDE_intersection_y_intercept_sum_l1359_135937


namespace NUMINAMATH_CALUDE_two_bedroom_units_l1359_135977

theorem two_bedroom_units (total_units : ℕ) (cost_one_bedroom : ℕ) (cost_two_bedroom : ℕ) (total_cost : ℕ) :
  total_units = 12 →
  cost_one_bedroom = 360 →
  cost_two_bedroom = 450 →
  total_cost = 4950 →
  ∃ (one_bedroom_units two_bedroom_units : ℕ),
    one_bedroom_units + two_bedroom_units = total_units ∧
    cost_one_bedroom * one_bedroom_units + cost_two_bedroom * two_bedroom_units = total_cost ∧
    two_bedroom_units = 7 :=
by sorry

end NUMINAMATH_CALUDE_two_bedroom_units_l1359_135977


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l1359_135925

/-- A circle with center (a, 2a) and radius √5 is tangent to the line 2x + y + 1 = 0 
    if and only if its equation is (x-1)² + (y-2)² = 5 -/
theorem circle_tangent_to_line (x y : ℝ) : 
  (∃ a : ℝ, (x - a)^2 + (y - 2*a)^2 = 5 ∧ 
   (|2*a + 2*a + 1| / Real.sqrt 5 = Real.sqrt 5)) ↔ 
  (x - 1)^2 + (y - 2)^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l1359_135925


namespace NUMINAMATH_CALUDE_three_pencils_two_pens_cost_l1359_135959

/-- The cost of a pencil -/
def pencil_cost : ℝ := sorry

/-- The cost of a pen -/
def pen_cost : ℝ := sorry

/-- The first condition: eight pencils and three pens cost $5.20 -/
axiom condition1 : 8 * pencil_cost + 3 * pen_cost = 5.20

/-- The second condition: two pencils and five pens cost $4.40 -/
axiom condition2 : 2 * pencil_cost + 5 * pen_cost = 4.40

/-- Theorem: The cost of three pencils and two pens is $2.5881 -/
theorem three_pencils_two_pens_cost : 
  3 * pencil_cost + 2 * pen_cost = 2.5881 := by sorry

end NUMINAMATH_CALUDE_three_pencils_two_pens_cost_l1359_135959


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1359_135976

theorem trigonometric_identity (α β γ : Real) 
  (h : Real.sin α + Real.sin γ = 2 * Real.sin β) : 
  Real.tan ((α + β) / 2) + Real.tan ((β + γ) / 2) = 2 * Real.tan ((γ + α) / 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1359_135976


namespace NUMINAMATH_CALUDE_triangle_conditions_equivalence_l1359_135914

theorem triangle_conditions_equivalence (x : ℝ) :
  (∀ (BC AC AB : ℝ),
    BC = x + 11 ∧ AC = x + 6 ∧ AB = 3*x + 2 →
    AB + AC > BC ∧ AB + BC > AC ∧ AC + BC > AB ∧
    BC > AB ∧ BC > AC) ↔
  (1 < x ∧ x < 4.5) :=
sorry

end NUMINAMATH_CALUDE_triangle_conditions_equivalence_l1359_135914


namespace NUMINAMATH_CALUDE_square_perimeter_l1359_135963

theorem square_perimeter (rectangle_length rectangle_width : ℝ) 
  (h1 : rectangle_length = 32)
  (h2 : rectangle_width = 64)
  (h3 : square_area = 2 * rectangle_length * rectangle_width) : 
  4 * Real.sqrt square_area = 256 :=
by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1359_135963


namespace NUMINAMATH_CALUDE_perpendicular_line_parallel_line_l1359_135906

-- Define the types for our points and lines
def Point := ℝ × ℝ
def Line := ℝ → ℝ → ℝ → Prop

-- Define the intersection point of two lines
def intersection (l1 l2 : Line) : Point :=
  let x := -1
  let y := 2
  (x, y)

-- Define perpendicularity of two lines
def perpendicular (l1 l2 : Line) : Prop :=
  ∃ (m1 m2 : ℝ), m1 * m2 = -1 ∧
    (∀ x y c, l1 x y c ↔ m1 * x + y = c) ∧
    (∀ x y c, l2 x y c ↔ m2 * x + y = c)

-- Define parallelism of two lines
def parallel (l1 l2 : Line) : Prop :=
  ∃ (m c1 c2 : ℝ), 
    (∀ x y c, l1 x y c ↔ m * x + y = c1) ∧
    (∀ x y c, l2 x y c ↔ m * x + y = c2)

-- Define the given lines
def line1 : Line := λ x y c => 3 * x + 4 * y - 5 = c
def line2 : Line := λ x y c => 2 * x + y = c
def line3 : Line := λ x y c => 3 * x - 2 * y - 1 = c

-- State the theorems
theorem perpendicular_line :
  let p := intersection line1 line2
  ∃ (l : Line), l p.1 p.2 (-4) ∧ perpendicular l line3 ∧ 
    ∀ x y c, l x y c ↔ 2 * x + 3 * y - 4 = c := by sorry

theorem parallel_line :
  let p := intersection line1 line2
  ∃ (l : Line), l p.1 p.2 7 ∧ parallel l line3 ∧ 
    ∀ x y c, l x y c ↔ 3 * x - 2 * y + 7 = c := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_parallel_line_l1359_135906


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1359_135950

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b (m : ℝ) : Fin 2 → ℝ := ![2, m]

-- Define the sum of vectors
def vector_sum (v w : Fin 2 → ℝ) : Fin 2 → ℝ := λ i => v i + w i

-- Define dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Theorem statement
theorem perpendicular_vectors (m : ℝ) : 
  dot_product (vector_sum a (b m)) a = 0 ↔ m = -7/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1359_135950


namespace NUMINAMATH_CALUDE_circle_equation_l1359_135970

/-- A circle with center (a, b) and radius r -/
structure Circle where
  a : ℝ
  b : ℝ
  r : ℝ

/-- The standard equation of a circle -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.a)^2 + (y - c.b)^2 = c.r^2

/-- A circle is tangent to the x-axis -/
def Circle.tangentToXAxis (c : Circle) : Prop :=
  c.b = c.r

/-- The center of the circle lies on the line y = 3x -/
def Circle.centerOnLine (c : Circle) : Prop :=
  c.b = 3 * c.a

/-- The chord intercepted by the circle on the line y = x has length 2√7 -/
def Circle.chordLength (c : Circle) : Prop :=
  2 * c.r^2 = (c.a - c.b)^2 + 14

/-- The main theorem -/
theorem circle_equation (c : Circle) 
  (h1 : c.tangentToXAxis)
  (h2 : c.centerOnLine)
  (h3 : c.chordLength) :
  (c.equation 1 3 ∧ c.r^2 = 9) ∨ (c.equation (-1) 3 ∧ c.r^2 = 9) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l1359_135970


namespace NUMINAMATH_CALUDE_non_intersecting_probability_is_two_thirds_l1359_135986

/-- Two persons start from opposite corners of a rectangular grid and can only move up or right one step at a time. -/
structure GridWalk where
  m : ℕ  -- number of rows
  n : ℕ  -- number of columns

/-- The probability that the routes of two persons do not intersect -/
def non_intersecting_probability (g : GridWalk) : ℚ :=
  2/3

/-- Theorem stating that the probability of non-intersecting routes is 2/3 -/
theorem non_intersecting_probability_is_two_thirds (g : GridWalk) :
  non_intersecting_probability g = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_non_intersecting_probability_is_two_thirds_l1359_135986


namespace NUMINAMATH_CALUDE_fencing_cost_per_meter_l1359_135998

/-- Proves that the cost of fencing per meter for a rectangular plot is 26.5 Rs. -/
theorem fencing_cost_per_meter 
  (length : ℝ) 
  (breadth : ℝ) 
  (total_cost : ℝ) 
  (h1 : length = 70) 
  (h2 : length = breadth + 40) 
  (h3 : total_cost = 5300) : 
  total_cost / (2 * length + 2 * breadth) = 26.5 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_per_meter_l1359_135998


namespace NUMINAMATH_CALUDE_special_action_figure_value_prove_special_figure_value_l1359_135912

theorem special_action_figure_value 
  (total_figures : Nat) 
  (regular_figure_value : Nat) 
  (regular_figure_count : Nat) 
  (discount : Nat) 
  (total_earnings : Nat) : Nat :=
  let special_figure_count := total_figures - regular_figure_count
  let regular_figures_earnings := regular_figure_count * (regular_figure_value - discount)
  let special_figure_earnings := total_earnings - regular_figures_earnings
  special_figure_earnings + discount

theorem prove_special_figure_value :
  special_action_figure_value 5 15 4 5 55 = 20 := by
  sorry

end NUMINAMATH_CALUDE_special_action_figure_value_prove_special_figure_value_l1359_135912


namespace NUMINAMATH_CALUDE_existence_of_d_l1359_135949

theorem existence_of_d : ∃ d : ℝ,
  (∃ n : ℤ, n = ⌊d⌋ ∧ 3 * (n : ℝ)^2 + 20 * (n : ℝ) - 67 = 0) ∧
  (4 * (d - ⌊d⌋)^2 - 15 * (d - ⌊d⌋) + 5 = 0) ∧
  (0 ≤ d - ⌊d⌋ ∧ d - ⌊d⌋ < 1) ∧
  d = -8.63 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_d_l1359_135949


namespace NUMINAMATH_CALUDE_product_equals_243_l1359_135957

theorem product_equals_243 : 
  (1/3 : ℚ) * 9 * (1/27 : ℚ) * 81 * (1/243 : ℚ) * 729 * (1/2187 : ℚ) * 6561 * (1/19683 : ℚ) * 59049 = 243 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_243_l1359_135957


namespace NUMINAMATH_CALUDE_arcsin_arccos_bound_l1359_135938

theorem arcsin_arccos_bound (x y : ℝ) (h : x^2 + y^2 = 1) :
  -5*π/2 ≤ 3 * Real.arcsin x - 2 * Real.arccos y ∧
  3 * Real.arcsin x - 2 * Real.arccos y ≤ π/2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_arccos_bound_l1359_135938


namespace NUMINAMATH_CALUDE_min_distance_sum_parabola_l1359_135941

/-- The minimum distance sum from a point on the parabola y^2 = 8x to two fixed points -/
theorem min_distance_sum_parabola :
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (6, 5)
  let parabola := {P : ℝ × ℝ | P.2^2 = 8 * P.1}
  ∃ (min_dist : ℝ), min_dist = 8 ∧ 
    ∀ P ∈ parabola, Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) + 
                     Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_parabola_l1359_135941


namespace NUMINAMATH_CALUDE_school_store_problem_l1359_135969

/-- Represents the cost of pencils and notebooks given certain pricing conditions -/
def school_store_cost (pencil_price notebook_price : ℚ) : Prop :=
  -- 10 pencils and 6 notebooks cost $3.50
  10 * pencil_price + 6 * notebook_price = (3.50 : ℚ) ∧
  -- 4 pencils and 9 notebooks cost $2.70
  4 * pencil_price + 9 * notebook_price = (2.70 : ℚ)

/-- Calculates the total cost including the fixed fee -/
def total_cost (pencil_price notebook_price : ℚ) (pencil_count notebook_count : ℕ) : ℚ :=
  let base_cost := pencil_count * pencil_price + notebook_count * notebook_price
  if pencil_count + notebook_count > 15 then base_cost + (0.50 : ℚ) else base_cost

/-- Theorem stating the cost of 24 pencils and 15 notebooks -/
theorem school_store_problem :
  ∃ (pencil_price notebook_price : ℚ),
    school_store_cost pencil_price notebook_price →
    total_cost pencil_price notebook_price 24 15 = (9.02 : ℚ) := by
  sorry


end NUMINAMATH_CALUDE_school_store_problem_l1359_135969


namespace NUMINAMATH_CALUDE_binomial_and_permutation_7_5_l1359_135944

theorem binomial_and_permutation_7_5 :
  (Nat.choose 7 5 = 21) ∧ (Nat.factorial 7 / Nat.factorial 2 = 2520) := by
  sorry

end NUMINAMATH_CALUDE_binomial_and_permutation_7_5_l1359_135944


namespace NUMINAMATH_CALUDE_prob_three_dice_sum_18_l1359_135990

/-- The probability of rolling a specific number on a standard die -/
def prob_single_die : ℚ := 1 / 6

/-- The number of faces on a standard die -/
def dice_faces : ℕ := 6

/-- The sum we're looking for -/
def target_sum : ℕ := 18

/-- The number of dice rolled -/
def num_dice : ℕ := 3

theorem prob_three_dice_sum_18 : 
  (prob_single_die ^ num_dice : ℚ) = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_dice_sum_18_l1359_135990


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l1359_135908

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 6 ways to distribute 5 indistinguishable balls into 4 indistinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l1359_135908


namespace NUMINAMATH_CALUDE_cos_alpha_sin_beta_range_l1359_135982

theorem cos_alpha_sin_beta_range (α β : Real) (h : Real.sin α * Real.cos β = -1/2) :
  ∃ (x : Real), Real.cos α * Real.sin β = x ∧ -1/2 ≤ x ∧ x ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_cos_alpha_sin_beta_range_l1359_135982


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1359_135984

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo (-2 : ℝ) 1 = {x | a * x^2 + b * x + c > 0}) :
  {x : ℝ | a * x^2 + (a + b) * x + c - a < 0} = 
    Set.Iic (-3 : ℝ) ∪ Set.Ioi (1 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1359_135984


namespace NUMINAMATH_CALUDE_line_symmetry_about_bisector_l1359_135920

/-- Given two lines l₁ and l₂ with angle bisector y = x, prove that if l₁ has equation ax + by + c = 0 (ab > 0), then l₂ has equation bx + ay + c = 0 -/
theorem line_symmetry_about_bisector (a b c : ℝ) (hab : a * b > 0) :
  let l₁ := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}
  let l₂ := {p : ℝ × ℝ | b * p.1 + a * p.2 + c = 0}
  let bisector := {p : ℝ × ℝ | p.1 = p.2}
  (∀ p : ℝ × ℝ, p ∈ bisector → (p ∈ l₁ ↔ p ∈ l₂)) →
  ∀ q : ℝ × ℝ, q ∈ l₂ := by
  sorry

end NUMINAMATH_CALUDE_line_symmetry_about_bisector_l1359_135920


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l1359_135952

theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * 3*x = 120 → x = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l1359_135952


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1359_135988

theorem cubic_equation_solution (p : ℝ) (a b c : ℝ) :
  (∀ x : ℝ, x^3 + p*x^2 + 3*x - 10 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  c - b = b - a →
  b - a > 0 →
  a = -1 ∧ b = -1 ∧ c = -1 ∧ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1359_135988


namespace NUMINAMATH_CALUDE_competition_participants_l1359_135916

theorem competition_participants : ℕ :=
  let initial_participants : ℕ := sorry
  let first_round_survival_rate : ℚ := 1 / 3
  let second_round_survival_rate : ℚ := 1 / 4
  let final_participants : ℕ := 18

  have h1 : (initial_participants : ℚ) * first_round_survival_rate * second_round_survival_rate = final_participants := by sorry

  initial_participants

end NUMINAMATH_CALUDE_competition_participants_l1359_135916


namespace NUMINAMATH_CALUDE_min_value_sum_equality_condition_l1359_135924

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (2 * c^2) + c^2 / (9 * a) ≥ 3 / Real.rpow 54 (1/3) :=
sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (2 * c^2) + c^2 / (9 * a) = 3 / Real.rpow 54 (1/3) ↔
  a = 6 * c^2 ∧ b = 2 * c^2 * Real.rpow 54 (1/3) ∧ c = Real.rpow 54 (1/4) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_equality_condition_l1359_135924


namespace NUMINAMATH_CALUDE_gift_amount_proof_l1359_135975

/-- The amount of money Josie received as a gift -/
def gift_amount : ℕ := 50

/-- The cost of one cassette tape -/
def cassette_cost : ℕ := 9

/-- The number of cassette tapes Josie plans to buy -/
def num_cassettes : ℕ := 2

/-- The cost of the headphone set -/
def headphone_cost : ℕ := 25

/-- The amount of money Josie will have left after her purchases -/
def money_left : ℕ := 7

/-- Theorem stating that the gift amount is equal to the sum of the purchases and remaining money -/
theorem gift_amount_proof : 
  gift_amount = num_cassettes * cassette_cost + headphone_cost + money_left :=
by sorry

end NUMINAMATH_CALUDE_gift_amount_proof_l1359_135975


namespace NUMINAMATH_CALUDE_range_of_m_m_value_when_sum_eq_neg_product_l1359_135926

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 - (2*m - 3)*x + m^2

-- Define the roots of the quadratic equation
def roots (m : ℝ) : Set ℝ := {x | quadratic m x = 0}

-- Theorem for the range of m
theorem range_of_m : ∀ m : ℝ, (∃ x₁ x₂ : ℝ, x₁ ∈ roots m ∧ x₂ ∈ roots m) → m ≤ 3/4 := by sorry

-- Theorem for the value of m when x₁ + x₂ = -x₁x₂
theorem m_value_when_sum_eq_neg_product : 
  ∀ m : ℝ, m ≤ 3/4 → 
  (∃ x₁ x₂ : ℝ, x₁ ∈ roots m ∧ x₂ ∈ roots m ∧ x₁ + x₂ = -(x₁ * x₂)) → 
  m = -3 := by sorry

end NUMINAMATH_CALUDE_range_of_m_m_value_when_sum_eq_neg_product_l1359_135926


namespace NUMINAMATH_CALUDE_lucy_fish_count_l1359_135964

theorem lucy_fish_count (initial_fish : ℝ) (bought_fish : ℝ) : 
  initial_fish = 212.0 → bought_fish = 280.0 → initial_fish + bought_fish = 492.0 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_count_l1359_135964


namespace NUMINAMATH_CALUDE_no_odd_3digit_div5_without5_l1359_135953

theorem no_odd_3digit_div5_without5 : 
  ¬∃ n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧  -- 3-digit number
    n % 2 = 1 ∧           -- odd
    n % 5 = 0 ∧           -- divisible by 5
    ∀ d : ℕ, d < 3 → (n / 10^d) % 10 ≠ 5  -- does not contain digit 5
    := by sorry

end NUMINAMATH_CALUDE_no_odd_3digit_div5_without5_l1359_135953


namespace NUMINAMATH_CALUDE_marias_stationery_cost_l1359_135929

/-- The total cost of Maria's stationery purchase after applying a coupon and including sales tax. -/
theorem marias_stationery_cost :
  let notebook_a_count : ℕ := 4
  let notebook_b_count : ℕ := 3
  let notebook_c_count : ℕ := 3
  let pen_count : ℕ := 5
  let highlighter_pack_count : ℕ := 1
  let notebook_a_price : ℚ := 3.5
  let notebook_b_price : ℚ := 2.25
  let notebook_c_price : ℚ := 1.75
  let pen_price : ℚ := 2
  let highlighter_pack_price : ℚ := 4.5
  let coupon_discount : ℚ := 0.1
  let sales_tax_rate : ℚ := 0.05

  let total_before_discount : ℚ := 
    notebook_a_count * notebook_a_price +
    notebook_b_count * notebook_b_price +
    notebook_c_count * notebook_c_price +
    pen_count * pen_price +
    highlighter_pack_count * highlighter_pack_price

  let discount_amount : ℚ := total_before_discount * coupon_discount
  let total_after_discount : ℚ := total_before_discount - discount_amount
  let sales_tax : ℚ := total_after_discount * sales_tax_rate
  let final_cost : ℚ := total_after_discount + sales_tax

  final_cost = 38.27 := by sorry

end NUMINAMATH_CALUDE_marias_stationery_cost_l1359_135929


namespace NUMINAMATH_CALUDE_nancy_savings_l1359_135980

-- Define the value of a dozen
def dozen : ℕ := 12

-- Define the value of a quarter in cents
def quarter_value : ℕ := 25

-- Define the number of cents in a dollar
def cents_per_dollar : ℕ := 100

-- Theorem statement
theorem nancy_savings (nancy_quarters : ℕ) (h1 : nancy_quarters = dozen) : 
  (nancy_quarters * quarter_value) / cents_per_dollar = 3 := by
  sorry

end NUMINAMATH_CALUDE_nancy_savings_l1359_135980


namespace NUMINAMATH_CALUDE_bus_journey_distance_l1359_135987

/-- Given a bus journey with two speeds, prove the distance covered at the slower speed. -/
theorem bus_journey_distance (total_distance : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ) 
  (h1 : total_distance = 250)
  (h2 : speed1 = 40)
  (h3 : speed2 = 60)
  (h4 : total_time = 5.4)
  (h5 : total_distance > 0)
  (h6 : speed1 > 0)
  (h7 : speed2 > 0)
  (h8 : total_time > 0)
  (h9 : speed1 < speed2) :
  ∃ (distance1 : ℝ), 
    distance1 / speed1 + (total_distance - distance1) / speed2 = total_time ∧ 
    distance1 = 148 := by
  sorry

end NUMINAMATH_CALUDE_bus_journey_distance_l1359_135987


namespace NUMINAMATH_CALUDE_function_composition_equality_l1359_135968

theorem function_composition_equality (C D : ℝ) (h : ℝ → ℝ) (k : ℝ → ℝ)
  (h_def : ∀ x, h x = C * x - 3 * D^2)
  (k_def : ∀ x, k x = D * x + 1)
  (D_neq_neg_one : D ≠ -1)
  (h_k_2_eq_zero : h (k 2) = 0) :
  C = 3 * D^2 / (2 * D + 1) := by
sorry

end NUMINAMATH_CALUDE_function_composition_equality_l1359_135968


namespace NUMINAMATH_CALUDE_food_to_budget_ratio_l1359_135962

def budget : ℚ := 3000
def supplies_fraction : ℚ := 1/4
def wages : ℚ := 1250

def food_expense : ℚ := budget - supplies_fraction * budget - wages

theorem food_to_budget_ratio :
  food_expense / budget = 1/3 := by sorry

end NUMINAMATH_CALUDE_food_to_budget_ratio_l1359_135962


namespace NUMINAMATH_CALUDE_max_yellow_apples_removal_max_total_apples_removal_l1359_135965

/-- Represents the number of apples of each color in the basket -/
structure AppleBasket where
  green : Nat
  yellow : Nat
  red : Nat

/-- Represents the number of apples removed from the basket -/
structure RemovedApples where
  green : Nat
  yellow : Nat
  red : Nat

/-- Checks if the removal condition is satisfied -/
def validRemoval (removed : RemovedApples) : Prop :=
  removed.green < removed.yellow ∧ removed.yellow < removed.red

/-- The initial state of the apple basket -/
def initialBasket : AppleBasket :=
  ⟨8, 11, 16⟩

theorem max_yellow_apples_removal (basket : AppleBasket) 
  (h : basket = initialBasket) :
  ∃ (removed : RemovedApples), 
    validRemoval removed ∧ 
    removed.yellow = 11 ∧
    ∀ (other : RemovedApples), 
      validRemoval other → other.yellow ≤ removed.yellow :=
sorry

theorem max_total_apples_removal (basket : AppleBasket) 
  (h : basket = initialBasket) :
  ∃ (removed : RemovedApples),
    validRemoval removed ∧
    removed.green + removed.yellow + removed.red = 33 ∧
    ∀ (other : RemovedApples),
      validRemoval other →
      other.green + other.yellow + other.red ≤ removed.green + removed.yellow + removed.red :=
sorry

end NUMINAMATH_CALUDE_max_yellow_apples_removal_max_total_apples_removal_l1359_135965


namespace NUMINAMATH_CALUDE_sarahs_stamp_collection_value_l1359_135918

/-- The value of a stamp collection given the total number of stamps,
    the number of stamps in a subset, and the value of that subset. -/
def stamp_collection_value (total_stamps : ℕ) (subset_stamps : ℕ) (subset_value : ℚ) : ℚ :=
  (total_stamps : ℚ) * subset_value / (subset_stamps : ℚ)

/-- Theorem stating that Sarah's stamp collection is worth 60 dollars -/
theorem sarahs_stamp_collection_value :
  stamp_collection_value 24 8 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_stamp_collection_value_l1359_135918


namespace NUMINAMATH_CALUDE_parentheses_number_l1359_135902

theorem parentheses_number (x : ℤ) : x - (-6) = 20 → x = 14 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_number_l1359_135902


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_pairs_l1359_135923

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem infinitely_many_divisible_pairs :
  ∀ k : ℕ, ∃ m n : ℕ+,
    (m : ℕ) ∣ (n : ℕ)^2 + 1 ∧
    (n : ℕ) ∣ (m : ℕ)^2 + 1 ∧
    (m : ℕ) = fib (2 * k + 1) ∧
    (n : ℕ) = fib (2 * k + 3) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_pairs_l1359_135923


namespace NUMINAMATH_CALUDE_exists_real_sqrt_x_minus_one_l1359_135930

theorem exists_real_sqrt_x_minus_one : ∃ x : ℝ, ∃ y : ℝ, y ^ 2 = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_real_sqrt_x_minus_one_l1359_135930


namespace NUMINAMATH_CALUDE_product_with_decimals_l1359_135932

theorem product_with_decimals (a b c : ℚ) (h : (125 : ℕ) * 384 = 48000) :
  a = 0.125 ∧ b = 3.84 ∧ c = 0.48 → a * b = c := by sorry

end NUMINAMATH_CALUDE_product_with_decimals_l1359_135932


namespace NUMINAMATH_CALUDE_pentagon_diagonals_l1359_135994

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A pentagon has 5 sides -/
def pentagon_sides : ℕ := 5

/-- Theorem: The number of diagonals in a pentagon is 5 -/
theorem pentagon_diagonals : num_diagonals pentagon_sides = 5 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_diagonals_l1359_135994


namespace NUMINAMATH_CALUDE_triangle_inequality_check_l1359_135960

def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality_check : 
  (¬ canFormTriangle 3 5 10) ∧ 
  (canFormTriangle 5 4 8) ∧ 
  (¬ canFormTriangle 2 4 6) ∧ 
  (¬ canFormTriangle 3 3 7) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_check_l1359_135960


namespace NUMINAMATH_CALUDE_overlap_area_is_half_unit_l1359_135991

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Calculate the area of overlap between two triangles on a 4x4 grid -/
def triangleOverlapArea (t1 t2 : Triangle) : ℝ :=
  sorry

/-- The main theorem stating that the overlap area is 0.5 square units -/
theorem overlap_area_is_half_unit : 
  let t1 := Triangle.mk (Point.mk 0 0) (Point.mk 3 2) (Point.mk 2 3)
  let t2 := Triangle.mk (Point.mk 0 3) (Point.mk 3 3) (Point.mk 3 0)
  triangleOverlapArea t1 t2 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_overlap_area_is_half_unit_l1359_135991


namespace NUMINAMATH_CALUDE_inequality_range_l1359_135996

theorem inequality_range (x : ℝ) :
  (∀ a : ℝ, a ≥ 1 → a * x^2 + (a - 3) * x + (a - 4) > 0) →
  x < -1 ∨ x > 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l1359_135996


namespace NUMINAMATH_CALUDE_sum_not_divisible_by_ten_l1359_135974

theorem sum_not_divisible_by_ten (n : ℕ) :
  ¬(10 ∣ (1981^n + 1982^n + 1983^n + 1984^n)) ↔ 4 ∣ n :=
sorry

end NUMINAMATH_CALUDE_sum_not_divisible_by_ten_l1359_135974


namespace NUMINAMATH_CALUDE_three_lines_theorem_l1359_135921

/-- Three lines in the plane -/
structure ThreeLines where
  l1 : Real → Real → Prop
  l2 : Real → Real → Prop
  l3 : Real → Real → Real → Prop

/-- The condition that the three lines divide the plane into six parts -/
def divides_into_six_parts (lines : ThreeLines) : Prop := sorry

/-- The main theorem -/
theorem three_lines_theorem (k : Real) :
  let lines : ThreeLines := {
    l1 := λ x y => x - 2*y + 1 = 0,
    l2 := λ x _ => x - 1 = 0,
    l3 := λ x y k => x + k*y = 0
  }
  divides_into_six_parts lines → k ∈ ({0, -1, -2} : Set Real) := by
  sorry

end NUMINAMATH_CALUDE_three_lines_theorem_l1359_135921


namespace NUMINAMATH_CALUDE_problem_solution_l1359_135927

theorem problem_solution (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : 
  x^2 + y^2 = 6 ∧ (x - y)^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1359_135927


namespace NUMINAMATH_CALUDE_plot_length_is_60_l1359_135946

/-- Represents a rectangular plot with its dimensions and fencing cost. -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencingCostPerMeter : ℝ
  totalFencingCost : ℝ

/-- Calculates the perimeter of a rectangular plot. -/
def perimeter (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth)

/-- Theorem stating the length of the plot given the conditions. -/
theorem plot_length_is_60 (plot : RectangularPlot)
  (h1 : plot.length = plot.breadth + 20)
  (h2 : plot.fencingCostPerMeter = 26.5)
  (h3 : plot.totalFencingCost = 5300)
  (h4 : plot.totalFencingCost = plot.fencingCostPerMeter * perimeter plot) :
  plot.length = 60 := by
  sorry

#check plot_length_is_60

end NUMINAMATH_CALUDE_plot_length_is_60_l1359_135946


namespace NUMINAMATH_CALUDE_binary_21_l1359_135993

/-- The binary representation of a natural number. -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Proposition: The binary representation of 21 is [true, false, true, false, true] -/
theorem binary_21 : toBinary 21 = [true, false, true, false, true] := by
  sorry

end NUMINAMATH_CALUDE_binary_21_l1359_135993


namespace NUMINAMATH_CALUDE_initial_ratio_is_11_to_9_l1359_135917

/-- Represents the contents of a can with milk and water -/
structure CanContents where
  milk : ℝ
  water : ℝ

/-- Proves that the initial ratio of milk to water is 11:9 given the conditions -/
theorem initial_ratio_is_11_to_9 (can : CanContents) : 
  can.milk + can.water = 20 → -- Initial contents
  can.milk + can.water + 10 = 30 → -- Adding 10L fills the can
  (can.milk + 10) / can.water = 5 / 2 → -- Resulting ratio is 5:2
  can.milk / can.water = 11 / 9 := by
  sorry

/-- Verify the solution satisfies the conditions -/
example : 
  let can : CanContents := { milk := 11, water := 9 }
  can.milk + can.water = 20 ∧
  can.milk + can.water + 10 = 30 ∧
  (can.milk + 10) / can.water = 5 / 2 ∧
  can.milk / can.water = 11 / 9 := by
  sorry

end NUMINAMATH_CALUDE_initial_ratio_is_11_to_9_l1359_135917


namespace NUMINAMATH_CALUDE_curve_C_properties_l1359_135936

-- Define the curve C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (x - 1)^2 + y^2 = 4 * ((x - 4)^2 + y^2)}

-- Define the line l
def l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               x - y + 3 = 0}

-- Theorem statement
theorem curve_C_properties :
  -- 1. The equation of C is x^2 + y^2 = 4
  (∀ p : ℝ × ℝ, p ∈ C ↔ let (x, y) := p; x^2 + y^2 = 4) ∧
  -- 2. The minimum distance from C to l is (3√2)/2 - 2
  (∃ d_min : ℝ, d_min = 3 * Real.sqrt 2 / 2 - 2 ∧
    (∀ p ∈ C, ∀ q ∈ l, dist p q ≥ d_min) ∧
    (∃ p ∈ C, ∃ q ∈ l, dist p q = d_min)) ∧
  -- 3. The maximum distance from C to l is 2 + (3√2)/2
  (∃ d_max : ℝ, d_max = 2 + 3 * Real.sqrt 2 / 2 ∧
    (∀ p ∈ C, ∀ q ∈ l, dist p q ≤ d_max) ∧
    (∃ p ∈ C, ∃ q ∈ l, dist p q = d_max)) :=
by sorry

end NUMINAMATH_CALUDE_curve_C_properties_l1359_135936


namespace NUMINAMATH_CALUDE_triangle_problem_l1359_135978

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) : 
  (Real.sqrt 3 * t.c = 2 * t.a * Real.sin t.C) →  -- Condition 2
  (t.A < π / 2) →  -- Condition 3: A is acute
  (t.a = 2 * Real.sqrt 3) →  -- Condition 4
  (1/2 * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3) →  -- Condition 5: Area
  (t.A = π / 3 ∧ 
   ((t.b = 4 ∧ t.c = 2) ∨ (t.b = 2 ∧ t.c = 4))) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1359_135978


namespace NUMINAMATH_CALUDE_no_square_divisible_by_six_between_39_and_120_l1359_135913

theorem no_square_divisible_by_six_between_39_and_120 :
  ¬∃ (x : ℕ), ∃ (y : ℕ), x = y^2 ∧ 6 ∣ x ∧ 39 < x ∧ x < 120 :=
by
  sorry

end NUMINAMATH_CALUDE_no_square_divisible_by_six_between_39_and_120_l1359_135913


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1359_135966

/-- A regular polygon with perimeter 150 cm and side length 15 cm has 10 sides. -/
theorem regular_polygon_sides (perimeter : ℝ) (side_length : ℝ) (n : ℕ) : 
  perimeter = 150 ∧ side_length = 15 ∧ perimeter = n * side_length → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1359_135966


namespace NUMINAMATH_CALUDE_geometric_sequence_from_arithmetic_l1359_135955

/-- An arithmetic sequence with non-zero common difference -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d ≠ 0, ∀ n, a (n + 1) - a n = d

/-- A geometric sequence -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ q ≠ 0, ∀ n, b (n + 1) / b n = q

theorem geometric_sequence_from_arithmetic (a b : ℕ → ℝ) :
  ArithmeticSequence a →
  GeometricSequence b →
  b 2 = 5 →
  a 5 = b 1 →
  a 8 = b 2 →
  a 13 = b 3 →
  ∀ n, b n = 3 * (5/3)^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_from_arithmetic_l1359_135955


namespace NUMINAMATH_CALUDE_ball_color_distribution_l1359_135999

theorem ball_color_distribution :
  ∀ (blue red green : ℕ),
  blue + red + green = 15 →
  (blue = red + 1 ∧ blue = green + 5) ∨
  (blue = red + 1 ∧ red = green) ∨
  (red = green ∧ blue = green + 5) →
  blue = 7 ∧ red = 6 ∧ green = 2 := by
sorry

end NUMINAMATH_CALUDE_ball_color_distribution_l1359_135999


namespace NUMINAMATH_CALUDE_gcd_powers_of_two_l1359_135934

theorem gcd_powers_of_two : Nat.gcd (2^2024 - 1) (2^2016 - 1) = 2^8 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_powers_of_two_l1359_135934


namespace NUMINAMATH_CALUDE_decimal_sum_difference_l1359_135956

theorem decimal_sum_difference : 0.5 - 0.03 + 0.007 + 0.0008 = 0.4778 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_difference_l1359_135956


namespace NUMINAMATH_CALUDE_dealership_sales_forecast_l1359_135942

theorem dealership_sales_forecast (sports_cars sedan_cars : ℕ) : 
  (5 : ℚ) / 8 = sports_cars / sedan_cars →
  sports_cars = 35 →
  sedan_cars = 56 := by
sorry

end NUMINAMATH_CALUDE_dealership_sales_forecast_l1359_135942


namespace NUMINAMATH_CALUDE_collinear_points_k_value_unique_k_value_l1359_135910

/-- Three points are collinear if the slope between any two pairs of points is the same. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) / (x₂ - x₁) = (y₃ - y₂) / (x₃ - x₂)

/-- Theorem: If the points (2,-3), (4,3), and (5, k/2) are collinear, then k = 12. -/
theorem collinear_points_k_value :
  ∀ k : ℝ, collinear 2 (-3) 4 3 5 (k/2) → k = 12 :=
by
  sorry

/-- Corollary: The only value of k that makes the points (2,-3), (4,3), and (5, k/2) collinear is 12. -/
theorem unique_k_value :
  ∃! k : ℝ, collinear 2 (-3) 4 3 5 (k/2) :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_unique_k_value_l1359_135910


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l1359_135922

theorem quadratic_solution_property (f g : ℝ) : 
  (3 * f^2 - 4 * f + 2 = 0) →
  (3 * g^2 - 4 * g + 2 = 0) →
  (f + 2) * (g + 2) = 22/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l1359_135922


namespace NUMINAMATH_CALUDE_handshakes_15_couples_l1359_135940

/-- The number of handshakes in a gathering of married couples -/
def num_handshakes (n : ℕ) : ℕ :=
  (n * 2 * (n * 2 - 2)) / 2 - n

/-- Theorem: In a gathering of 15 married couples, the total number of handshakes is 405 -/
theorem handshakes_15_couples :
  num_handshakes 15 = 405 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_15_couples_l1359_135940


namespace NUMINAMATH_CALUDE_equation_solution_l1359_135979

theorem equation_solution :
  ∃ (x : ℚ), x ≠ 1 ∧ (x^2 - 2*x + 3) / (x - 1) = x + 4 ↔ x = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1359_135979


namespace NUMINAMATH_CALUDE_fifth_occurrence_of_three_sevenths_l1359_135935

/-- Represents a fraction with numerator and denominator -/
structure Fraction where
  numerator : ℕ
  denominator : ℕ+

/-- The sequence of fractions as described in the problem -/
def fractionSequence : ℕ → Fraction := sorry

/-- Two fractions are equivalent if their cross products are equal -/
def areEquivalent (f1 f2 : Fraction) : Prop :=
  f1.numerator * f2.denominator = f2.numerator * f1.denominator

/-- The position of the nth occurrence of a fraction equivalent to the given fraction -/
def positionOfNthOccurrence (f : Fraction) (n : ℕ) : ℕ := sorry

/-- The main theorem to prove -/
theorem fifth_occurrence_of_three_sevenths :
  positionOfNthOccurrence ⟨3, 7⟩ 5 = 1211 := by sorry

end NUMINAMATH_CALUDE_fifth_occurrence_of_three_sevenths_l1359_135935


namespace NUMINAMATH_CALUDE_towel_rate_problem_l1359_135971

/-- Proves that the rate of two towels is 250 given the conditions of the problem -/
theorem towel_rate_problem (price1 price2 avg_price : ℕ) 
  (h1 : price1 = 100)
  (h2 : price2 = 150)
  (h3 : avg_price = 155)
  : ((10 * avg_price) - (3 * price1 + 5 * price2)) / 2 = 250 := by
  sorry

end NUMINAMATH_CALUDE_towel_rate_problem_l1359_135971


namespace NUMINAMATH_CALUDE_present_price_l1359_135954

theorem present_price (original_price : ℝ) (discount_rate : ℝ) (num_people : ℕ) 
  (individual_savings : ℝ) :
  original_price > 0 →
  discount_rate = 0.2 →
  num_people = 3 →
  individual_savings = 4 →
  original_price * (1 - discount_rate) = num_people * individual_savings →
  original_price * (1 - discount_rate) = 48 := by
sorry

end NUMINAMATH_CALUDE_present_price_l1359_135954


namespace NUMINAMATH_CALUDE_equation_roots_property_l1359_135992

theorem equation_roots_property :
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ > 0 ∧ 2 * x₁^2 - 5 = 20 ∧ 2 * x₂^2 - 5 = 20) ∧
  (∃ y₁ y₂ : ℝ, y₁ < 0 ∧ y₂ > 0 ∧ (3 * y₁ - 2)^2 = (2 * y₁ - 3)^2 ∧ (3 * y₂ - 2)^2 = (2 * y₂ - 3)^2) ∧
  (∃ z₁ z₂ : ℝ, z₁ < 0 ∧ z₂ > 0 ∧ (z₁^2 - 16 ≥ 0) ∧ (2 * z₁ - 2 ≥ 0) ∧ z₁^2 - 16 = 2 * z₁ - 2 ∧
                              (z₂^2 - 16 ≥ 0) ∧ (2 * z₂ - 2 ≥ 0) ∧ z₂^2 - 16 = 2 * z₂ - 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_property_l1359_135992


namespace NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_power_l1359_135904

theorem sqrt_abs_sum_zero_implies_power (a b : ℝ) :
  Real.sqrt (a + 2) + |b - 1| = 0 → (a + b) ^ 2017 = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_power_l1359_135904


namespace NUMINAMATH_CALUDE_rectangle_ratio_around_square_l1359_135933

/-- Given a square surrounded by four identical rectangles, this theorem proves
    that the ratio of the longer side to the shorter side of each rectangle is 2,
    when the area of the larger square formed is 9 times that of the inner square. -/
theorem rectangle_ratio_around_square : 
  ∀ (s x y : ℝ),
  s > 0 →  -- inner square side length is positive
  x > y → y > 0 →  -- rectangle dimensions are positive and x is longer
  (s + 2*y)^2 = 9*s^2 →  -- area relation
  (x + s)^2 = 9*s^2 →  -- outer square side length
  x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_around_square_l1359_135933


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1359_135915

theorem inequality_equivalence (x : ℝ) : x + 1 > 3 ↔ x > 2 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1359_135915


namespace NUMINAMATH_CALUDE_equality_of_arithmetic_progressions_l1359_135901

theorem equality_of_arithmetic_progressions (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : ∃ r : ℝ, b^2 - a^2 = r ∧ c^2 - b^2 = r ∧ d^2 - c^2 = r)
  (h2 : ∃ s : ℝ, 1/(a+b+d) - 1/(a+b+c) = s ∧ 
               1/(a+c+d) - 1/(a+b+d) = s ∧ 
               1/(b+c+d) - 1/(a+c+d) = s) :
  a = b ∧ b = c ∧ c = d := by
sorry

end NUMINAMATH_CALUDE_equality_of_arithmetic_progressions_l1359_135901


namespace NUMINAMATH_CALUDE_carlos_jogging_distance_l1359_135928

/-- Calculates the distance traveled given a constant speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Given Carlos' jogging speed and time, prove the distance he jogged -/
theorem carlos_jogging_distance :
  let jogging_speed : ℝ := 4
  let jogging_time : ℝ := 2
  distance jogging_speed jogging_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_carlos_jogging_distance_l1359_135928


namespace NUMINAMATH_CALUDE_min_value_constraint_l1359_135919

theorem min_value_constraint (a b : ℝ) (h : a + b^2 = 2) :
  ∃ (m : ℝ), m = 3 ∧ ∀ (x y : ℝ), x + y^2 = 2 → a^2 + 6*b^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_constraint_l1359_135919


namespace NUMINAMATH_CALUDE_school_students_count_l1359_135948

theorem school_students_count (boys girls : ℕ) 
  (h1 : 2 * boys / 3 + 3 * girls / 4 = 550)
  (h2 : 3 * girls / 4 = 150) : 
  boys + girls = 800 := by
  sorry

end NUMINAMATH_CALUDE_school_students_count_l1359_135948


namespace NUMINAMATH_CALUDE_hemisphere_exposed_area_l1359_135989

/-- Given a hemisphere of radius r, where half of it is submerged in liquid,
    the total exposed surface area (including the circular top) is 2πr². -/
theorem hemisphere_exposed_area (r : ℝ) (hr : r > 0) :
  let exposed_area := π * r^2 + (π * r^2)
  exposed_area = 2 * π * r^2 := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_exposed_area_l1359_135989


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l1359_135905

theorem z_in_first_quadrant :
  ∀ (z : ℂ), (z - Complex.I) * (2 - Complex.I) = 5 →
  ∃ (a b : ℝ), z = Complex.mk a b ∧ a > 0 ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l1359_135905


namespace NUMINAMATH_CALUDE_expression_equality_l1359_135997

theorem expression_equality : 
  Real.sqrt 4 + |1 - Real.sqrt 3| - (1/2)⁻¹ + 2023^0 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1359_135997
