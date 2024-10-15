import Mathlib

namespace NUMINAMATH_CALUDE_problem_statement_l3837_383719

/-- An arithmetic sequence with a non-zero common difference -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) - a n = d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) / b n = r

theorem problem_statement (a b : ℕ → ℝ) : 
  arithmetic_sequence a →
  geometric_sequence b →
  3 * a 2005 - (a 2007)^2 + 3 * a 2009 = 0 →
  b 2007 = a 2007 →
  b 2006 * b 2008 = 36 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3837_383719


namespace NUMINAMATH_CALUDE_salary_percentage_calculation_l3837_383700

/-- Given two employees X and Y with a total salary and Y's known salary,
    calculate the percentage of Y's salary that X is paid. -/
theorem salary_percentage_calculation
  (total_salary : ℝ) (y_salary : ℝ) (h1 : total_salary = 638)
  (h2 : y_salary = 290) :
  (total_salary - y_salary) / y_salary * 100 = 120 :=
by sorry

end NUMINAMATH_CALUDE_salary_percentage_calculation_l3837_383700


namespace NUMINAMATH_CALUDE_framed_photo_border_area_l3837_383732

/-- The area of the border of a framed rectangular photograph -/
theorem framed_photo_border_area 
  (photo_height : ℝ) 
  (photo_width : ℝ) 
  (border_width : ℝ) 
  (h1 : photo_height = 6) 
  (h2 : photo_width = 8) 
  (h3 : border_width = 3) : 
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - 
  photo_height * photo_width = 120 := by
  sorry

end NUMINAMATH_CALUDE_framed_photo_border_area_l3837_383732


namespace NUMINAMATH_CALUDE_min_value_on_interval_l3837_383784

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 4*x + 1

-- Define the interval
def interval : Set ℝ := Set.Icc 0 3

-- Theorem statement
theorem min_value_on_interval :
  ∃ (x : ℝ), x ∈ interval ∧ f x = -3 ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_min_value_on_interval_l3837_383784


namespace NUMINAMATH_CALUDE_triangle_area_comparison_l3837_383729

theorem triangle_area_comparison : 
  let a : Real := 3
  let b : Real := 5
  let c : Real := 6
  let p : Real := (a + b + c) / 2
  let area_A : Real := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  let area_B : Real := (3 * Real.sqrt 14) / 2
  area_A = 2 * Real.sqrt 14 ∧ area_A / area_B = 4 / 3 := by sorry

end NUMINAMATH_CALUDE_triangle_area_comparison_l3837_383729


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l3837_383789

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of the circle (x-1)^2 + y^2 = 3 -/
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 3

theorem circle_center_and_radius :
  ∃ (c : Circle), (∀ x y, circle_equation x y ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∧
                   c.center = (1, 0) ∧
                   c.radius = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l3837_383789


namespace NUMINAMATH_CALUDE_zoo_with_only_hippos_possible_l3837_383720

-- Define the universe of zoos
variable (Z : Type)

-- Define the subsets of zoos with hippos, rhinos, and giraffes
variable (H R G : Set Z)

-- Define the conditions
axiom condition1 : H ∩ R ⊆ Gᶜ
axiom condition2 : R ∩ Gᶜ ⊆ H
axiom condition3 : H ∩ G ⊆ R

-- Theorem to prove
theorem zoo_with_only_hippos_possible :
  ∃ (z : Z), z ∈ H ∧ z ∉ G ∧ z ∉ R :=
sorry

end NUMINAMATH_CALUDE_zoo_with_only_hippos_possible_l3837_383720


namespace NUMINAMATH_CALUDE_wednesday_distance_l3837_383788

theorem wednesday_distance (monday_distance tuesday_distance : ℕ) 
  (average_distance : ℚ) (total_days : ℕ) :
  monday_distance = 12 →
  tuesday_distance = 18 →
  average_distance = 17 →
  total_days = 3 →
  (monday_distance + tuesday_distance + (average_distance * total_days - monday_distance - tuesday_distance : ℚ)) / total_days = average_distance →
  average_distance * total_days - monday_distance - tuesday_distance = 21 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_distance_l3837_383788


namespace NUMINAMATH_CALUDE_triplet_sum_not_one_l3837_383783

theorem triplet_sum_not_one : ∃! (a b c : ℝ), 
  ((a = 1.1 ∧ b = -2.1 ∧ c = 1.0) ∨ 
   (a = 1/2 ∧ b = 1/3 ∧ c = 1/6) ∨ 
   (a = 2 ∧ b = -2 ∧ c = 1) ∨ 
   (a = 0.1 ∧ b = 0.3 ∧ c = 0.6) ∨ 
   (a = -3/2 ∧ b = -5/2 ∧ c = 5)) ∧ 
  a + b + c ≠ 1 := by
sorry

end NUMINAMATH_CALUDE_triplet_sum_not_one_l3837_383783


namespace NUMINAMATH_CALUDE_sin_cos_identity_tan_fraction_value_l3837_383777

-- Part 1
theorem sin_cos_identity (α : Real) :
  (Real.sin (3 * α) / Real.sin α) - (Real.cos (3 * α) / Real.cos α) = 2 := by
  sorry

-- Part 2
theorem tan_fraction_value (α : Real) (h : Real.tan (α / 2) = 2) :
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_tan_fraction_value_l3837_383777


namespace NUMINAMATH_CALUDE_shortest_distance_between_tangents_l3837_383705

-- Define the parabola C₁
def C₁ (x y : ℝ) : Prop := x^2 = 4*y

-- Define the point P on C₁
def P : ℝ × ℝ := (2, 1)

-- Define the point Q
def Q : ℝ × ℝ := (0, 2)

-- Define the line l (implicitly by Q and its intersection with C₁)
def l (x y : ℝ) : Prop := ∃ (k : ℝ), y - Q.2 = k * (x - Q.1)

-- Define the parabola C₂
def C₂ (x y : ℝ) : Prop := x^2 = 2*y - 4

-- Define the tangent lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := ∃ (x₃ y₃ : ℝ), C₁ x₃ y₃ ∧ 2*x*x₃ - 2*y - 2*x₃^2 = 0

def l₂ (x y : ℝ) : Prop := ∃ (x₄ y₄ : ℝ), C₂ x₄ y₄ ∧ 2*x*x₄ - 2*y - x₄^2 + 4 = 0

-- The theorem to prove
theorem shortest_distance_between_tangents :
  ∀ (x₃ : ℝ), l₁ x₃ (x₃^2/4) → l₂ (x₃/2) ((x₃/2)^2/2 + 2) →
  (x₃^2 + 4) / (2 * Real.sqrt (x₃^2 + 1)) ≥ Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_shortest_distance_between_tangents_l3837_383705


namespace NUMINAMATH_CALUDE_vector_dot_product_l3837_383754

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 4)
def c : ℝ × ℝ := (3, 2)

theorem vector_dot_product :
  (2 • a + b) • c = -3 := by sorry

end NUMINAMATH_CALUDE_vector_dot_product_l3837_383754


namespace NUMINAMATH_CALUDE_weight_of_smaller_cube_l3837_383704

/-- Given two cubes of the same material, where the second cube has sides twice
    as long as the first and weighs 40 pounds, prove that the weight of the first
    cube is 5 pounds. -/
theorem weight_of_smaller_cube (s : ℝ) (w : ℝ → ℝ → ℝ) :
  (∀ x y, w x y = (y / x^3) * w 1 1) →  -- weight is proportional to volume
  w (2*s) (8*s^3) = 40 →                -- weight of larger cube
  w s (s^3) = 5 := by
sorry


end NUMINAMATH_CALUDE_weight_of_smaller_cube_l3837_383704


namespace NUMINAMATH_CALUDE_gcd_problem_l3837_383718

theorem gcd_problem (a b : ℕ+) (h : Nat.gcd a.val b.val = 15) :
  Nat.gcd (12 * a.val) (18 * b.val) ≥ 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3837_383718


namespace NUMINAMATH_CALUDE_equation_classification_l3837_383717

-- Define what a linear equation in two variables is
def is_linear_equation_in_two_variables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y + c

-- Define the properties of the equation in question
def has_two_unknowns_and_degree_one (f : ℝ → ℝ → ℝ) : Prop :=
  (∃ (x y : ℝ), f x y ≠ f x 0 ∧ f x y ≠ f 0 y) ∧ 
  (∀ (x y : ℝ), ∃ (a b c : ℝ), f x y = a * x + b * y + c)

-- State the theorem
theorem equation_classification 
  (f : ℝ → ℝ → ℝ) 
  (h : has_two_unknowns_and_degree_one f) : 
  is_linear_equation_in_two_variables f :=
sorry

end NUMINAMATH_CALUDE_equation_classification_l3837_383717


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l3837_383727

theorem greatest_divisor_four_consecutive_integers :
  ∃ (d : ℕ), (∀ (n : ℕ), n > 0 → d ∣ (n * (n + 1) * (n + 2) * (n + 3))) ∧
  (∀ (k : ℕ), (∀ (n : ℕ), n > 0 → k ∣ (n * (n + 1) * (n + 2) * (n + 3))) → k ≤ d) ∧
  d = 12 :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l3837_383727


namespace NUMINAMATH_CALUDE_point_on_exponential_graph_tan_value_l3837_383772

theorem point_on_exponential_graph_tan_value :
  ∀ a : ℝ, (3 : ℝ)^a = 9 → Real.tan (a * π / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_exponential_graph_tan_value_l3837_383772


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3837_383763

/-- A hyperbola is defined by its standard equation and properties. -/
structure Hyperbola where
  /-- The coefficient of x² in the standard equation -/
  a : ℝ
  /-- The coefficient of y² in the standard equation -/
  b : ℝ
  /-- A point that the hyperbola passes through -/
  point : ℝ × ℝ
  /-- The slope of the asymptotes -/
  asymptote_slope : ℝ

/-- The standard equation of a hyperbola holds for its defining point. -/
def satisfies_equation (h : Hyperbola) : Prop :=
  h.a * h.point.1^2 - h.b * h.point.2^2 = 1

/-- The asymptote slope is related to the coefficients in the standard equation. -/
def asymptote_condition (h : Hyperbola) : Prop :=
  h.asymptote_slope^2 = h.a / h.b

/-- The theorem stating the standard equation of the hyperbola. -/
theorem hyperbola_equation (h : Hyperbola)
    (point_cond : h.point = (4, Real.sqrt 3))
    (slope_cond : h.asymptote_slope = 1/2)
    (eq_cond : satisfies_equation h)
    (asym_cond : asymptote_condition h) :
    h.a = 1/4 ∧ h.b = 1 :=
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3837_383763


namespace NUMINAMATH_CALUDE_total_books_calculation_l3837_383745

/-- The number of book shelves -/
def num_shelves : ℕ := 350

/-- The number of books per shelf -/
def books_per_shelf : ℕ := 25

/-- The total number of books on all shelves -/
def total_books : ℕ := num_shelves * books_per_shelf

theorem total_books_calculation : total_books = 8750 := by
  sorry

end NUMINAMATH_CALUDE_total_books_calculation_l3837_383745


namespace NUMINAMATH_CALUDE_solve_equation_l3837_383766

theorem solve_equation (y : ℤ) : 7 + y = 3 ↔ y = -4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3837_383766


namespace NUMINAMATH_CALUDE_fish_count_l3837_383780

/-- The number of fish caught by Jeffery -/
def jeffery_fish : ℕ := 60

/-- The number of fish caught by Ryan -/
def ryan_fish : ℕ := jeffery_fish / 2

/-- The number of fish caught by Jason -/
def jason_fish : ℕ := ryan_fish / 3

/-- The total number of fish caught by all three -/
def total_fish : ℕ := jason_fish + ryan_fish + jeffery_fish

theorem fish_count : total_fish = 100 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_l3837_383780


namespace NUMINAMATH_CALUDE_min_value_theorem_l3837_383726

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a 2 - a 1

theorem min_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  arithmetic_sequence a →
  (∀ n, a n > 0) →
  a 2018 = a 2017 + 2 * a 2016 →
  Real.sqrt (a m * a n) = 4 * a 1 →
  (1 : ℝ) / m + 5 / n ≥ 1 + Real.sqrt 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3837_383726


namespace NUMINAMATH_CALUDE_tip_percentage_is_thirty_percent_l3837_383735

/-- Calculates the tip percentage given meal costs and total price --/
def calculate_tip_percentage (appetizer_cost : ℚ) (entree_cost : ℚ) (num_entrees : ℕ) (dessert_cost : ℚ) (total_price : ℚ) : ℚ :=
  let meal_cost := appetizer_cost + entree_cost * num_entrees + dessert_cost
  let tip_amount := total_price - meal_cost
  (tip_amount / meal_cost) * 100

/-- Proves that the tip percentage is 30% given the specific meal costs --/
theorem tip_percentage_is_thirty_percent :
  calculate_tip_percentage 9 20 2 11 78 = 30 := by
  sorry

end NUMINAMATH_CALUDE_tip_percentage_is_thirty_percent_l3837_383735


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l3837_383702

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) :
  GeometricSequence a q →
  (a 3)^2 + 4 * (a 3) + 1 = 0 →
  (a 7)^2 + 4 * (a 7) + 1 = 0 →
  a 5 = -1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l3837_383702


namespace NUMINAMATH_CALUDE_soap_bars_per_pack_l3837_383794

/-- Given that Nancy bought 6 packs of soap and 30 bars of soap in total,
    prove that the number of bars in each pack is 5. -/
theorem soap_bars_per_pack :
  ∀ (total_packs : ℕ) (total_bars : ℕ),
    total_packs = 6 →
    total_bars = 30 →
    total_bars / total_packs = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_soap_bars_per_pack_l3837_383794


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_two_l3837_383706

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : 0 < a
  b_pos : 0 < b

/-- The left focus of a hyperbola -/
def left_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- A vertex on the imaginary axis of a hyperbola -/
def imaginary_vertex (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- A point on the right asymptote of a hyperbola -/
def right_asymptote_point (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- Vector from point p1 to point p2 -/
def vector (p1 p2 : ℝ × ℝ) : ℝ × ℝ := sorry

theorem hyperbola_eccentricity_sqrt_two (a b : ℝ) (h : Hyperbola a b) :
  let F := left_focus h
  let A := imaginary_vertex h
  let B := right_asymptote_point h
  vector F A = (Real.sqrt 2 - 1) • vector A B →
  eccentricity h = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_two_l3837_383706


namespace NUMINAMATH_CALUDE_childrens_tickets_sold_l3837_383721

theorem childrens_tickets_sold
  (adult_price senior_price children_price : ℚ)
  (total_tickets : ℕ)
  (total_revenue : ℚ)
  (h1 : adult_price = 6)
  (h2 : children_price = 9/2)
  (h3 : senior_price = 5)
  (h4 : total_tickets = 600)
  (h5 : total_revenue = 3250)
  : ∃ (A C S : ℕ),
    A + C + S = total_tickets ∧
    adult_price * A + children_price * C + senior_price * S = total_revenue ∧
    C = (350 - S) / (3/2) :=
sorry

end NUMINAMATH_CALUDE_childrens_tickets_sold_l3837_383721


namespace NUMINAMATH_CALUDE_apples_in_basket_l3837_383709

/-- Represents the number of oranges in the basket -/
def oranges : ℕ := sorry

/-- Represents the number of apples in the basket -/
def apples : ℕ := 4 * oranges

/-- The total number of fruits consumed if 2/3 of each fruit's quantity is eaten -/
def consumed_fruits : ℕ := 50

theorem apples_in_basket : apples = 60 := by
  sorry

end NUMINAMATH_CALUDE_apples_in_basket_l3837_383709


namespace NUMINAMATH_CALUDE_max_a_value_l3837_383769

def is_lattice_point (x y : ℤ) : Prop := True

def passes_through_lattice_point (m : ℚ) (b : ℤ) : Prop :=
  ∃ x y : ℤ, is_lattice_point x y ∧ 0 < x ∧ x ≤ 200 ∧ y = m * x + b

theorem max_a_value :
  let a : ℚ := 68 / 201
  ∀ m : ℚ, 1/3 < m → m < a →
    ¬(passes_through_lattice_point m 3 ∨ passes_through_lattice_point m 1) ∧
    ∀ a' : ℚ, a < a' →
      ∃ m : ℚ, 1/3 < m ∧ m < a' ∧
        (passes_through_lattice_point m 3 ∨ passes_through_lattice_point m 1) :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l3837_383769


namespace NUMINAMATH_CALUDE_curve_is_ellipse_with_foci_on_y_axis_l3837_383747

/-- The curve represented by x²sin(α) - y²cos(α) = 1 is an ellipse with foci on the y-axis when α is between π/2 and 3π/4 -/
theorem curve_is_ellipse_with_foci_on_y_axis (α : Real) 
  (h_α_range : α ∈ Set.Ioo (π / 2) (3 * π / 4)) :
  ∃ (a b : Real), a > 0 ∧ b > 0 ∧ a > b ∧
  ∀ (x y : Real), x^2 * Real.sin α - y^2 * Real.cos α = 1 ↔ 
    (x^2 / b^2) + (y^2 / a^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_curve_is_ellipse_with_foci_on_y_axis_l3837_383747


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3837_383799

theorem min_value_of_expression (x y z : ℝ) : (x^2*y - 1)^2 + (x + y + z)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3837_383799


namespace NUMINAMATH_CALUDE_average_and_difference_l3837_383767

theorem average_and_difference (y : ℝ) : 
  (46 + y) / 2 = 52 → |y - 46| = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_and_difference_l3837_383767


namespace NUMINAMATH_CALUDE_triple_reflection_opposite_l3837_383760

/-- Represents a 3D vector -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane mirror -/
inductive Mirror
  | XY
  | XZ
  | YZ

/-- Reflects a vector across a given mirror -/
def reflect (v : Vector3D) (m : Mirror) : Vector3D :=
  match m with
  | Mirror.XY => ⟨v.x, v.y, -v.z⟩
  | Mirror.XZ => ⟨v.x, -v.y, v.z⟩
  | Mirror.YZ => ⟨-v.x, v.y, v.z⟩

/-- Theorem: After three reflections on mutually perpendicular mirrors, 
    the resulting vector is opposite to the initial vector -/
theorem triple_reflection_opposite (f : Vector3D) :
  let f1 := reflect f Mirror.XY
  let f2 := reflect f1 Mirror.XZ
  let f3 := reflect f2 Mirror.YZ
  f3 = Vector3D.mk (-f.x) (-f.y) (-f.z) := by
  sorry


end NUMINAMATH_CALUDE_triple_reflection_opposite_l3837_383760


namespace NUMINAMATH_CALUDE_bev_is_third_oldest_l3837_383739

/-- Represents the age of a person -/
structure Age : Type where
  value : ℕ

/-- Represents a person with their name and age -/
structure Person : Type where
  name : String
  age : Age

/-- Defines the "older than" relation between two people -/
def olderThan (p1 p2 : Person) : Prop :=
  p1.age.value > p2.age.value

theorem bev_is_third_oldest 
  (andy bev cao dhruv elcim : Person)
  (h1 : olderThan dhruv bev)
  (h2 : olderThan bev elcim)
  (h3 : olderThan andy elcim)
  (h4 : olderThan bev andy)
  (h5 : olderThan cao bev) :
  ∃ (x y : Person), 
    (olderThan x bev ∧ olderThan y bev) ∧
    (∀ (z : Person), z ≠ x ∧ z ≠ y → olderThan bev z ∨ z = bev) :=
by sorry

end NUMINAMATH_CALUDE_bev_is_third_oldest_l3837_383739


namespace NUMINAMATH_CALUDE_log_sum_equals_ten_l3837_383793

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_sum_equals_ten :
  log 3 243 - log 3 (1/27) + log 3 9 = 10 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_ten_l3837_383793


namespace NUMINAMATH_CALUDE_abigail_initial_fences_l3837_383762

/-- The number of fences Abigail can build in 8 hours -/
def fences_in_8_hours : ℕ := 8 * 60 / 30

/-- The total number of fences after 8 hours of building -/
def total_fences : ℕ := 26

/-- The number of fences Abigail built initially -/
def initial_fences : ℕ := total_fences - fences_in_8_hours

theorem abigail_initial_fences : initial_fences = 10 := by
  sorry

end NUMINAMATH_CALUDE_abigail_initial_fences_l3837_383762


namespace NUMINAMATH_CALUDE_remainder_equality_l3837_383782

theorem remainder_equality (A B D S T u v : ℕ) 
  (h1 : A > B)
  (h2 : S = A % D)
  (h3 : T = B % D)
  (h4 : u = (A + B) % D)
  (h5 : v = (S + T) % D) :
  u = v := by
  sorry

end NUMINAMATH_CALUDE_remainder_equality_l3837_383782


namespace NUMINAMATH_CALUDE_sasha_leaves_picked_l3837_383781

/-- The number of apple trees along the road -/
def apple_trees : ℕ := 17

/-- The number of poplar trees along the road -/
def poplar_trees : ℕ := 20

/-- The index of the apple tree from which Sasha starts picking leaves -/
def start_index : ℕ := 8

/-- The total number of trees along the road -/
def total_trees : ℕ := apple_trees + poplar_trees

/-- The number of leaves Sasha picked -/
def leaves_picked : ℕ := total_trees - (start_index - 1)

theorem sasha_leaves_picked : leaves_picked = 24 := by
  sorry

end NUMINAMATH_CALUDE_sasha_leaves_picked_l3837_383781


namespace NUMINAMATH_CALUDE_min_value_log_quadratic_l3837_383744

theorem min_value_log_quadratic (x : ℝ) (h : x^2 - 2*x + 3 > 0) :
  Real.log (x^2 - 2*x + 3) ≥ Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_log_quadratic_l3837_383744


namespace NUMINAMATH_CALUDE_merchant_profit_percentage_l3837_383730

theorem merchant_profit_percentage (cost_price : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) : 
  markup_percentage = 75 →
  discount_percentage = 40 →
  cost_price > 0 →
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let selling_price := marked_price * (1 - discount_percentage / 100)
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 5 := by
sorry

end NUMINAMATH_CALUDE_merchant_profit_percentage_l3837_383730


namespace NUMINAMATH_CALUDE_elevator_initial_floor_l3837_383741

def elevator_problem (initial_floor final_floor top_floor down_move up_move1 up_move2 : ℕ) : Prop :=
  final_floor = top_floor ∧
  top_floor = 13 ∧
  final_floor = initial_floor - down_move + up_move1 + up_move2 ∧
  down_move = 7 ∧
  up_move1 = 3 ∧
  up_move2 = 8

theorem elevator_initial_floor :
  ∀ initial_floor final_floor top_floor down_move up_move1 up_move2 : ℕ,
    elevator_problem initial_floor final_floor top_floor down_move up_move1 up_move2 →
    initial_floor = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_elevator_initial_floor_l3837_383741


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l3837_383725

/-- A geometric sequence with a_2 = 2 and a_8 = 128 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  a 2 = 2 ∧ a 8 = 128

/-- The general formula for the sequence -/
def GeneralFormula (a : ℕ → ℝ) : Prop :=
  (∀ n, a n = 2^(n-1)) ∨ (∀ n, a n = -(-2)^(n-1))

/-- The sum of the first n terms -/
def SumFormula (S : ℕ → ℝ) : Prop :=
  (∀ n, S n = 2^n - 1) ∨ (∀ n, S n = (1/3) * ((-2)^n - 1))

theorem geometric_sequence_properties
  (a : ℕ → ℝ) (S : ℕ → ℝ) (h : GeometricSequence a) :
  GeneralFormula a ∧ SumFormula S :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l3837_383725


namespace NUMINAMATH_CALUDE_probability_at_least_one_switch_closed_l3837_383756

theorem probability_at_least_one_switch_closed 
  (p : ℝ) 
  (h1 : 0 < p) 
  (h2 : p < 1) :
  let prob_at_least_one_closed := 4*p - 6*p^2 + 4*p^3 - p^4
  prob_at_least_one_closed = 1 - (1 - p)^4 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_switch_closed_l3837_383756


namespace NUMINAMATH_CALUDE_absent_student_percentage_l3837_383750

theorem absent_student_percentage (total_students : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total_students = 180)
  (h2 : boys = 100)
  (h3 : girls = 80)
  (h4 : total_students = boys + girls)
  (absent_boys_ratio : ℚ)
  (absent_girls_ratio : ℚ)
  (h5 : absent_boys_ratio = 1 / 5)
  (h6 : absent_girls_ratio = 1 / 4) :
  (((boys * absent_boys_ratio + girls * absent_girls_ratio) / total_students) : ℚ) = 2222 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_absent_student_percentage_l3837_383750


namespace NUMINAMATH_CALUDE_sara_quarters_l3837_383790

theorem sara_quarters (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 21 → additional = 49 → total = initial + additional → total = 70 := by
  sorry

end NUMINAMATH_CALUDE_sara_quarters_l3837_383790


namespace NUMINAMATH_CALUDE_amelia_monday_sales_l3837_383761

/-- Represents the number of Jet Bars Amelia sold on Monday -/
def monday_sales : ℕ := sorry

/-- Represents the number of Jet Bars Amelia sold on Tuesday -/
def tuesday_sales : ℕ := sorry

/-- The weekly goal for Jet Bar sales -/
def weekly_goal : ℕ := 90

/-- The number of Jet Bars remaining to be sold -/
def remaining_sales : ℕ := 16

theorem amelia_monday_sales :
  monday_sales = 45 ∧
  tuesday_sales = monday_sales - 16 ∧
  monday_sales + tuesday_sales + remaining_sales = weekly_goal :=
by sorry

end NUMINAMATH_CALUDE_amelia_monday_sales_l3837_383761


namespace NUMINAMATH_CALUDE_marcus_gathered_25_bottles_l3837_383723

-- Define the total number of milk bottles
def total_bottles : ℕ := 45

-- Define the number of bottles John gathered
def john_bottles : ℕ := 20

-- Define Marcus' bottles as the difference between total and John's
def marcus_bottles : ℕ := total_bottles - john_bottles

-- Theorem to prove
theorem marcus_gathered_25_bottles : marcus_bottles = 25 := by
  sorry

end NUMINAMATH_CALUDE_marcus_gathered_25_bottles_l3837_383723


namespace NUMINAMATH_CALUDE_initial_green_balls_l3837_383724

theorem initial_green_balls (pink_balls : ℕ) (added_green_balls : ℕ) :
  pink_balls = 23 →
  added_green_balls = 14 →
  ∃ initial_green_balls : ℕ, 
    initial_green_balls + added_green_balls = pink_balls ∧
    initial_green_balls = 9 :=
by sorry

end NUMINAMATH_CALUDE_initial_green_balls_l3837_383724


namespace NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l3837_383785

-- Define the types for lines and planes
variable (L : Type) (P : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : L → P → Prop)
variable (parallel : L → P → Prop)
variable (planePerpendicular : P → P → Prop)

-- State the theorem
theorem line_perp_parallel_implies_planes_perp
  (l : L) (α β : P) 
  (h1 : perpendicular l α)
  (h2 : parallel l β) :
  planePerpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l3837_383785


namespace NUMINAMATH_CALUDE_seed_ratio_proof_l3837_383755

def total_seeds : ℕ := 120
def left_seeds : ℕ := 20
def additional_seeds : ℕ := 30
def remaining_seeds : ℕ := 30

theorem seed_ratio_proof :
  let used_seeds := total_seeds - remaining_seeds
  let right_seeds := used_seeds - left_seeds - additional_seeds
  (right_seeds : ℚ) / left_seeds = 2 / 1 := by
sorry

end NUMINAMATH_CALUDE_seed_ratio_proof_l3837_383755


namespace NUMINAMATH_CALUDE_remainder_thirteen_power_fiftyone_mod_five_l3837_383786

theorem remainder_thirteen_power_fiftyone_mod_five :
  13^51 % 5 = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_thirteen_power_fiftyone_mod_five_l3837_383786


namespace NUMINAMATH_CALUDE_triangle_with_perimeter_12_has_area_6_l3837_383773

-- Define a triangle with integral sides
def Triangle := (ℕ × ℕ × ℕ)

-- Function to calculate perimeter of a triangle
def perimeter (t : Triangle) : ℕ :=
  let (a, b, c) := t
  a + b + c

-- Function to check if three sides form a valid triangle
def is_valid_triangle (t : Triangle) : Prop :=
  let (a, b, c) := t
  a + b > c ∧ b + c > a ∧ c + a > b

-- Function to calculate the area of a triangle using Heron's formula
noncomputable def area (t : Triangle) : ℝ :=
  let (a, b, c) := t
  let s : ℝ := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem statement
theorem triangle_with_perimeter_12_has_area_6 :
  ∃ (t : Triangle), perimeter t = 12 ∧ is_valid_triangle t ∧ area t = 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_with_perimeter_12_has_area_6_l3837_383773


namespace NUMINAMATH_CALUDE_sum_remainder_mod_seven_l3837_383797

theorem sum_remainder_mod_seven :
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_seven_l3837_383797


namespace NUMINAMATH_CALUDE_fraction_multiplication_l3837_383787

theorem fraction_multiplication : (2 : ℚ) / 3 * 5 / 7 * 8 / 9 = 80 / 189 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l3837_383787


namespace NUMINAMATH_CALUDE_spring_math_camp_inconsistency_l3837_383771

theorem spring_math_camp_inconsistency : 
  ¬ ∃ (b g : ℕ), 11 * b + 7 * g = 4046 := by
  sorry

end NUMINAMATH_CALUDE_spring_math_camp_inconsistency_l3837_383771


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3837_383764

theorem complex_magnitude_problem : 
  let i : ℂ := Complex.I
  let z : ℂ := 1 + (1 - i)^2
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3837_383764


namespace NUMINAMATH_CALUDE_last_digit_of_product_l3837_383775

theorem last_digit_of_product : (3^101 * 5^89 * 6^127 * 7^139 * 11^79 * 13^67 * 17^53) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_product_l3837_383775


namespace NUMINAMATH_CALUDE_stating_meeting_handshakes_l3837_383748

/-- 
Given a group of people at a meeting, where each person shakes hands with at least
a certain number of others, this function calculates the minimum possible number of handshakes.
-/
def min_handshakes (n : ℕ) (min_shakes_per_person : ℕ) : ℕ :=
  (n * min_shakes_per_person) / 2

/-- 
Theorem stating that for a meeting of 30 people where each person shakes hands with
at least 3 others, the minimum possible number of handshakes is 45.
-/
theorem meeting_handshakes :
  min_handshakes 30 3 = 45 := by
  sorry

#eval min_handshakes 30 3

end NUMINAMATH_CALUDE_stating_meeting_handshakes_l3837_383748


namespace NUMINAMATH_CALUDE_smallest_sum_of_five_primes_with_unique_digits_l3837_383703

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the set of digits used in a number -/
def digitsUsed (n : ℕ) : Finset ℕ := sorry

/-- A function that returns the sum of a list of natural numbers -/
def sumList (l : List ℕ) : ℕ := sorry

/-- The theorem statement -/
theorem smallest_sum_of_five_primes_with_unique_digits :
  ∃ (primes : List ℕ),
    primes.length = 5 ∧
    (∀ p ∈ primes, isPrime p) ∧
    (digitsUsed (sumList primes) = Finset.range 9) ∧
    (∀ s : List ℕ,
      s.length = 5 →
      (∀ p ∈ s, isPrime p) →
      (digitsUsed (sumList s) = Finset.range 9) →
      sumList primes ≤ sumList s) ∧
    sumList primes = 106 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_five_primes_with_unique_digits_l3837_383703


namespace NUMINAMATH_CALUDE_initial_apples_count_l3837_383758

/-- The number of apples Sarah initially had in her bag -/
def initial_apples : ℕ := 25

/-- The number of apples Sarah gave to teachers -/
def apples_to_teachers : ℕ := 16

/-- The number of apples Sarah gave to friends -/
def apples_to_friends : ℕ := 5

/-- The number of apples Sarah ate -/
def apples_eaten : ℕ := 1

/-- The number of apples left in Sarah's bag when she got home -/
def apples_left : ℕ := 3

/-- Theorem stating that the initial number of apples equals the sum of apples given away, eaten, and left -/
theorem initial_apples_count : 
  initial_apples = apples_to_teachers + apples_to_friends + apples_eaten + apples_left :=
by sorry

end NUMINAMATH_CALUDE_initial_apples_count_l3837_383758


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l3837_383774

theorem smallest_number_divisibility : ∃! n : ℕ, 
  (∀ m : ℕ, m < n → ¬(∀ d ∈ [12, 16, 18, 21, 28, 35, 45], (m - 4) % d = 0)) ∧
  (∀ d ∈ [12, 16, 18, 21, 28, 35, 45], (n - 4) % d = 0) ∧
  n = 5044 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l3837_383774


namespace NUMINAMATH_CALUDE_female_average_score_l3837_383791

theorem female_average_score (total_average : ℝ) (male_average : ℝ) (male_count : ℕ) (female_count : ℕ) :
  total_average = 90 →
  male_average = 82 →
  male_count = 8 →
  female_count = 32 →
  (male_count * male_average + female_count * ((male_count + female_count) * total_average - male_count * male_average) / female_count) / (male_count + female_count) = 90 →
  ((male_count + female_count) * total_average - male_count * male_average) / female_count = 92 := by
sorry

end NUMINAMATH_CALUDE_female_average_score_l3837_383791


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l3837_383743

theorem gcd_factorial_eight_and_factorial_six_squared :
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 1440 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l3837_383743


namespace NUMINAMATH_CALUDE_circle_center_coordinate_sum_l3837_383742

/-- Given two points (5, 3) and (-7, 9) as endpoints of a circle's diameter,
    prove that the sum of the coordinates of the circle's center is 5. -/
theorem circle_center_coordinate_sum : 
  let p1 : ℝ × ℝ := (5, 3)
  let p2 : ℝ × ℝ := (-7, 9)
  let center : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  center.1 + center.2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_sum_l3837_383742


namespace NUMINAMATH_CALUDE_problem_solution_l3837_383707

theorem problem_solution (x : ℝ) (h : x + 1/x = 3) : 
  (x - 1)^2 + 16/((x - 1)^2) = 23/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3837_383707


namespace NUMINAMATH_CALUDE_rectangle_area_is_eight_l3837_383752

/-- A square inscribed in a circle, which is inscribed in a rectangle --/
structure SquareCircleRectangle where
  /-- Side length of the square --/
  s : ℝ
  /-- Radius of the circle --/
  r : ℝ
  /-- Width of the rectangle --/
  w : ℝ
  /-- Length of the rectangle --/
  l : ℝ
  /-- The square's diagonal is the circle's diameter --/
  h1 : s * Real.sqrt 2 = 2 * r
  /-- The circle's diameter is the rectangle's width --/
  h2 : 2 * r = w
  /-- The rectangle's length is twice its width --/
  h3 : l = 2 * w
  /-- The square's diagonal is 4 units --/
  h4 : s * Real.sqrt 2 = 4

/-- The area of the rectangle is 8 square units --/
theorem rectangle_area_is_eight (scr : SquareCircleRectangle) : scr.l * scr.w = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_eight_l3837_383752


namespace NUMINAMATH_CALUDE_log_less_than_square_l3837_383779

theorem log_less_than_square (x : ℝ) (h : x > 0) : Real.log (1 + x) < x^2 := by
  sorry

end NUMINAMATH_CALUDE_log_less_than_square_l3837_383779


namespace NUMINAMATH_CALUDE_money_distribution_problem_l3837_383734

/-- Represents the money distribution problem among three friends --/
structure MoneyDistribution where
  total : ℝ  -- Total amount to distribute
  neha : ℝ   -- Neha's share
  sabi : ℝ   -- Sabi's share
  mahi : ℝ   -- Mahi's share
  x : ℝ      -- Amount removed from Sabi's share

/-- The conditions of the problem --/
def problemConditions (d : MoneyDistribution) : Prop :=
  d.total = 1100 ∧
  d.mahi = 102 ∧
  d.neha + d.sabi + d.mahi = d.total ∧
  (d.neha - 5) / (d.sabi - d.x) = 1/4 ∧
  (d.neha - 5) / (d.mahi - 4) = 1/3

/-- The theorem to prove --/
theorem money_distribution_problem (d : MoneyDistribution) 
  (h : problemConditions d) : d.x = 829.67 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_problem_l3837_383734


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l3837_383708

theorem geometric_arithmetic_sequence_problem (b q a d : ℝ) 
  (h1 : b = a + d)
  (h2 : b * q = a + 3 * d)
  (h3 : b * q^2 = a + 6 * d)
  (h4 : b * (b * q) * (b * q^2) = 64) :
  b = 8 / 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l3837_383708


namespace NUMINAMATH_CALUDE_revenue_decrease_percentage_l3837_383711

def old_revenue : ℝ := 72.0
def new_revenue : ℝ := 48.0

theorem revenue_decrease_percentage :
  (old_revenue - new_revenue) / old_revenue * 100 = 33.33 := by
  sorry

end NUMINAMATH_CALUDE_revenue_decrease_percentage_l3837_383711


namespace NUMINAMATH_CALUDE_rods_in_mile_l3837_383736

/-- Represents the number of furlongs in a mile -/
def furlongs_per_mile : ℕ := 8

/-- Represents the number of rods in a furlong -/
def rods_per_furlong : ℕ := 40

/-- Theorem stating that one mile is equal to 320 rods -/
theorem rods_in_mile : furlongs_per_mile * rods_per_furlong = 320 := by
  sorry

end NUMINAMATH_CALUDE_rods_in_mile_l3837_383736


namespace NUMINAMATH_CALUDE_largest_coefficients_in_expansion_l3837_383712

def binomial_coefficient (n k : ℕ) : ℕ := sorry

def expansion_term (n r : ℕ) : ℕ := 2^r * binomial_coefficient n r

theorem largest_coefficients_in_expansion (n : ℕ) (h : n = 11) :
  (∀ k, k ≠ 5 ∧ k ≠ 6 → expansion_term n 5 ≥ expansion_term n k) ∧
  (∀ k, k ≠ 5 ∧ k ≠ 6 → expansion_term n 6 ≥ expansion_term n k) ∧
  expansion_term n 7 = expansion_term n 8 ∧
  expansion_term n 7 = 42240 ∧
  (∀ k, k ≠ 7 ∧ k ≠ 8 → expansion_term n 7 > expansion_term n k) :=
sorry

end NUMINAMATH_CALUDE_largest_coefficients_in_expansion_l3837_383712


namespace NUMINAMATH_CALUDE_unique_square_cube_factor_of_1800_l3837_383796

/-- A number is a perfect square if it can be expressed as the product of an integer with itself. -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

/-- A number is a perfect cube if it can be expressed as the product of an integer with itself three times. -/
def IsPerfectCube (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k * k

/-- A number is a factor of another number if it divides the latter without a remainder. -/
def IsFactor (a n : ℕ) : Prop :=
  n % a = 0

theorem unique_square_cube_factor_of_1800 :
  ∃! x : ℕ, x > 0 ∧ IsFactor x 1800 ∧ IsPerfectSquare x ∧ IsPerfectCube x :=
sorry

end NUMINAMATH_CALUDE_unique_square_cube_factor_of_1800_l3837_383796


namespace NUMINAMATH_CALUDE_multiplication_increase_l3837_383751

theorem multiplication_increase (x : ℝ) : x * 20 = 20 + 280 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_increase_l3837_383751


namespace NUMINAMATH_CALUDE_negation_of_or_statement_l3837_383749

theorem negation_of_or_statement (x y : ℝ) :
  ¬(x > 1 ∨ y > 1) ↔ x ≤ 1 ∧ y ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_negation_of_or_statement_l3837_383749


namespace NUMINAMATH_CALUDE_smallest_addition_and_quotient_l3837_383731

theorem smallest_addition_and_quotient : 
  let n := 897326
  let d := 456
  let x := d - (n % d)
  ∀ y, 0 ≤ y ∧ y < x → ¬(d ∣ (n + y)) ∧
  (d ∣ (n + x)) ∧
  ((n + x) / d = 1968) := by
  sorry

end NUMINAMATH_CALUDE_smallest_addition_and_quotient_l3837_383731


namespace NUMINAMATH_CALUDE_negation_existential_quadratic_l3837_383740

theorem negation_existential_quadratic :
  (¬ ∃ x : ℝ, x^2 + 2*x - 3 > 0) ↔ (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_existential_quadratic_l3837_383740


namespace NUMINAMATH_CALUDE_die_roll_count_l3837_383713

theorem die_roll_count (total_sides : ℕ) (red_sides : ℕ) (prob : ℚ) : 
  total_sides = 10 →
  red_sides = 3 →
  prob = 147/1000 →
  (red_sides / total_sides : ℚ) * (1 - red_sides / total_sides : ℚ)^2 = prob →
  3 = 3 :=
by sorry

end NUMINAMATH_CALUDE_die_roll_count_l3837_383713


namespace NUMINAMATH_CALUDE_sum_difference_absolute_values_l3837_383722

theorem sum_difference_absolute_values : 
  (3 + (-4) + (-5)) - (|3| + |-4| + |-5|) = -18 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_absolute_values_l3837_383722


namespace NUMINAMATH_CALUDE_divisibility_condition_l3837_383795

theorem divisibility_condition (n : ℕ) : 
  (∃ m : ℕ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → k ∣ m) ∧ 
            ¬((n + 1) ∣ m) ∧ ¬((n + 2) ∣ m) ∧ ¬((n + 3) ∣ m)) ↔ 
  n = 1 ∨ n = 2 ∨ n = 6 :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3837_383795


namespace NUMINAMATH_CALUDE_fraction_simplification_l3837_383757

theorem fraction_simplification (x y : ℝ) (h : y = x / (1 - 2*x)) :
  (2*x - 3*x*y - 2*y) / (y + x*y - x) = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3837_383757


namespace NUMINAMATH_CALUDE_limit_S_2_pow_n_to_infinity_l3837_383776

/-- S(n) represents the sum of digits of n in base 10 -/
def S (n : ℕ) : ℕ := sorry

/-- Main theorem: The limit of S(2^n) as n approaches infinity is infinity -/
theorem limit_S_2_pow_n_to_infinity :
  ∀ M : ℕ, ∃ N : ℕ, ∀ n : ℕ, n ≥ N → S (2^n) > M :=
sorry

end NUMINAMATH_CALUDE_limit_S_2_pow_n_to_infinity_l3837_383776


namespace NUMINAMATH_CALUDE_sequence_length_l3837_383733

theorem sequence_length (n : ℕ+) (b : ℕ → ℝ) : 
  b 0 = 41 →
  b 1 = 76 →
  b n = 0 →
  (∀ k : ℕ, 1 ≤ k ∧ k < n → b (k + 1) = b (k - 1) - 4 / b k) →
  n = 777 :=
by sorry

end NUMINAMATH_CALUDE_sequence_length_l3837_383733


namespace NUMINAMATH_CALUDE_pen_notebook_ratio_l3837_383738

/-- Given 50 pens and 40 notebooks, prove that the ratio of pens to notebooks is 5:4 -/
theorem pen_notebook_ratio :
  let num_pens : ℕ := 50
  let num_notebooks : ℕ := 40
  (num_pens : ℚ) / (num_notebooks : ℚ) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_pen_notebook_ratio_l3837_383738


namespace NUMINAMATH_CALUDE_yard_length_26_trees_l3837_383746

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (distance_between_trees : ℝ) : ℝ :=
  (num_trees - 1) * distance_between_trees

/-- Theorem: The length of a yard with 26 trees planted at equal distances,
    with one tree at each end and 10 meters between consecutive trees, is 250 meters. -/
theorem yard_length_26_trees :
  yard_length 26 10 = 250 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_26_trees_l3837_383746


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l3837_383716

theorem concentric_circles_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  π * b^2 - π * a^2 = 5 * (π * a^2) → a / b = 1 / Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l3837_383716


namespace NUMINAMATH_CALUDE_max_value_4x_3y_l3837_383765

theorem max_value_4x_3y (x y : ℝ) :
  x^2 + y^2 = 16*x + 8*y + 8 →
  4*x + 3*y ≤ Real.sqrt (5184 - 173.33) - 72 :=
by sorry

end NUMINAMATH_CALUDE_max_value_4x_3y_l3837_383765


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l3837_383710

theorem polynomial_multiplication (x z : ℝ) :
  (3 * x^5 - 7 * z^3) * (9 * x^10 + 21 * x^5 * z^3 + 49 * z^6) = 27 * x^15 - 343 * z^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l3837_383710


namespace NUMINAMATH_CALUDE_work_completion_l3837_383753

/-- The number of men in the first group -/
def men_first : ℕ := 15

/-- The number of days for the first group to complete the work -/
def days_first : ℚ := 25

/-- The number of days for the second group to complete the work -/
def days_second : ℚ := 37/2

/-- The total amount of work in man-days -/
def total_work : ℚ := men_first * days_first

/-- The number of men in the second group -/
def men_second : ℕ := 20

theorem work_completion :
  (men_second : ℚ) * days_second = total_work :=
sorry

end NUMINAMATH_CALUDE_work_completion_l3837_383753


namespace NUMINAMATH_CALUDE_max_subsequent_voters_l3837_383737

/-- Represents a movie rating system where:
  * Ratings are integers from 0 to 10
  * At moment T, the rating was an integer
  * After moment T, each subsequent voter decreased the rating by one unit
-/
structure MovieRating where
  initial_rating : ℕ
  initial_voters : ℕ
  subsequent_votes : List ℕ

/-- The rating at any given moment is the sum of all scores divided by their quantity -/
def current_rating (mr : MovieRating) : ℚ :=
  (mr.initial_rating * mr.initial_voters + mr.subsequent_votes.sum) / 
  (mr.initial_voters + mr.subsequent_votes.length)

/-- The condition that the rating decreases by 1 unit after each vote -/
def decreasing_by_one (mr : MovieRating) : Prop :=
  ∀ i, i < mr.subsequent_votes.length →
    current_rating { mr with 
      subsequent_votes := mr.subsequent_votes.take i
    } - current_rating { mr with 
      subsequent_votes := mr.subsequent_votes.take (i + 1)
    } = 1

/-- The main theorem: The maximum number of viewers who could have voted after moment T is 5 -/
theorem max_subsequent_voters (mr : MovieRating) 
    (h1 : mr.initial_rating ∈ Set.range (fun i => i : ℕ → ℕ) ∩ Set.Icc 0 10)
    (h2 : ∀ v ∈ mr.subsequent_votes, v ∈ Set.range (fun i => i : ℕ → ℕ) ∩ Set.Icc 0 10)
    (h3 : decreasing_by_one mr) :
    mr.subsequent_votes.length ≤ 5 :=
  sorry

end NUMINAMATH_CALUDE_max_subsequent_voters_l3837_383737


namespace NUMINAMATH_CALUDE_min_value_expression_l3837_383728

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ((a^2 + 4*a + 2)*(b^2 + 4*b + 2)*(c^2 + 4*c + 2)) / (a*b*c) ≥ 512 ∧
  (∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
    ((a₀^2 + 4*a₀ + 2)*(b₀^2 + 4*b₀ + 2)*(c₀^2 + 4*c₀ + 2)) / (a₀*b₀*c₀) = 512) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3837_383728


namespace NUMINAMATH_CALUDE_prob_more_heads_12_coins_l3837_383770

/-- The number of coins flipped -/
def n : ℕ := 12

/-- The probability of getting more heads than tails when flipping n coins -/
def prob_more_heads (n : ℕ) : ℚ :=
  1 / 2 - (n.choose (n / 2)) / (2 ^ n)

theorem prob_more_heads_12_coins : 
  prob_more_heads n = 793 / 2048 := by
  sorry

end NUMINAMATH_CALUDE_prob_more_heads_12_coins_l3837_383770


namespace NUMINAMATH_CALUDE_same_color_probability_three_colors_three_draws_l3837_383759

/-- The probability of drawing the same color ball three times in a row --/
def same_color_probability (total_colors : ℕ) (num_draws : ℕ) : ℚ :=
  (total_colors : ℚ) / (total_colors ^ num_draws : ℚ)

/-- Theorem: The probability of drawing the same color ball three times in a row,
    with replacement, from a bag containing one red, one yellow, and one green ball,
    is equal to 1/9. --/
theorem same_color_probability_three_colors_three_draws :
  same_color_probability 3 3 = 1 / 9 := by
  sorry

#eval same_color_probability 3 3

end NUMINAMATH_CALUDE_same_color_probability_three_colors_three_draws_l3837_383759


namespace NUMINAMATH_CALUDE_limit_at_neg_seven_l3837_383778

/-- The limit of (2x^2 + 15x + 7)/(x + 7) as x approaches -7 is -13 -/
theorem limit_at_neg_seven (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, x ≠ -7 →
    |x - (-7)| < δ → |(2*x^2 + 15*x + 7)/(x + 7) - (-13)| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_at_neg_seven_l3837_383778


namespace NUMINAMATH_CALUDE_minimum_parents_needed_minimum_parents_for_tour_l3837_383798

theorem minimum_parents_needed (num_children : ℕ) (car_capacity : ℕ) : ℕ :=
  let total_people := num_children
  let drivers_needed := (total_people + car_capacity - 1) / car_capacity
  drivers_needed

theorem minimum_parents_for_tour :
  minimum_parents_needed 50 6 = 10 :=
by sorry

end NUMINAMATH_CALUDE_minimum_parents_needed_minimum_parents_for_tour_l3837_383798


namespace NUMINAMATH_CALUDE_quaternary_201_equals_33_l3837_383792

/-- Converts a quaternary (base 4) number to its decimal equivalent -/
def quaternary_to_decimal (q : List Nat) : Nat :=
  q.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The quaternary representation of the number -/
def quaternary_201 : List Nat := [1, 0, 2]

theorem quaternary_201_equals_33 :
  quaternary_to_decimal quaternary_201 = 33 := by
  sorry

end NUMINAMATH_CALUDE_quaternary_201_equals_33_l3837_383792


namespace NUMINAMATH_CALUDE_tan_2_implies_sin_cos_2_5_l3837_383768

theorem tan_2_implies_sin_cos_2_5 (x : ℝ) (h : Real.tan x = 2) : 
  Real.sin x * Real.cos x = 2/5 := by
sorry

end NUMINAMATH_CALUDE_tan_2_implies_sin_cos_2_5_l3837_383768


namespace NUMINAMATH_CALUDE_problem_solution_l3837_383701

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := k - |x - 4|

-- Define the theorem
theorem problem_solution (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sol : Set.Icc (-1 : ℝ) 1 = {x : ℝ | f 1 (x + 4) ≥ 0})
  (h_eq : 1/a + 1/(2*b) + 1/(3*c) = 1) :
  1 = 1 ∧ (1/9)*a + (2/9)*b + (3/9)*c ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l3837_383701


namespace NUMINAMATH_CALUDE_solution_count_l3837_383714

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f y) = x + y + c

/-- The theorem stating the number of solutions based on the value of c -/
theorem solution_count (c : ℝ) :
  (c = 0 ∧ ∃! f : ℝ → ℝ, SatisfiesEquation f c ∧ f = id) ∨
  (c ≠ 0 ∧ ¬∃ f : ℝ → ℝ, SatisfiesEquation f c) :=
sorry

end NUMINAMATH_CALUDE_solution_count_l3837_383714


namespace NUMINAMATH_CALUDE_simplify_expression_l3837_383715

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 8) - (x + 6)*(3*x - 2) = 4*x - 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3837_383715
