import Mathlib

namespace NUMINAMATH_CALUDE_positive_root_equation_l465_46597

theorem positive_root_equation : ∃ x : ℝ, x > 0 ∧ x^3 - 3*x^2 - x - Real.sqrt 2 = 0 :=
by
  use 2 + Real.sqrt 2
  sorry

end NUMINAMATH_CALUDE_positive_root_equation_l465_46597


namespace NUMINAMATH_CALUDE_shipping_cost_per_unit_l465_46552

/-- A computer manufacturer produces electronic components with the following parameters:
  * Production cost per component: $80
  * Fixed monthly costs: $16,200
  * Monthly production and sales: 150 components
  * Lowest break-even selling price: $190 per component
  This theorem proves that the shipping cost per unit is $2. -/
theorem shipping_cost_per_unit (production_cost : ℝ) (fixed_costs : ℝ) (units : ℝ) (selling_price : ℝ)
  (h1 : production_cost = 80)
  (h2 : fixed_costs = 16200)
  (h3 : units = 150)
  (h4 : selling_price = 190) :
  ∃ (shipping_cost : ℝ), 
    units * (production_cost + shipping_cost) + fixed_costs = units * selling_price ∧ 
    shipping_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_shipping_cost_per_unit_l465_46552


namespace NUMINAMATH_CALUDE_sixth_root_of_24414062515625_l465_46576

theorem sixth_root_of_24414062515625 : (24414062515625 : ℝ) ^ (1/6 : ℝ) = 51 := by
  sorry

end NUMINAMATH_CALUDE_sixth_root_of_24414062515625_l465_46576


namespace NUMINAMATH_CALUDE_translated_parabola_vertex_l465_46562

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2 - 4*x + 2

-- Define the translation
def translation_left : ℝ := 3
def translation_down : ℝ := 2

-- Theorem stating the vertex of the translated parabola
theorem translated_parabola_vertex :
  let vertex_x : ℝ := 2 - translation_left
  let vertex_y : ℝ := original_parabola 2 - translation_down
  (vertex_x, vertex_y) = (-1, -4) := by sorry

end NUMINAMATH_CALUDE_translated_parabola_vertex_l465_46562


namespace NUMINAMATH_CALUDE_f_properties_l465_46538

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin (2 * x) + b * Real.cos (2 * x)

theorem f_properties (a b : ℝ) (h : a * b ≠ 0) :
  (∀ x, f 1 (-Real.sqrt 3) x = 2 * Real.sin (2 * (x - Real.pi / 6))) ∧
  (a = b → ∀ x, f a b (x + Real.pi / 4) = f a b (Real.pi / 4 - x)) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l465_46538


namespace NUMINAMATH_CALUDE_element_order_l465_46556

-- Define the elements as a custom type
inductive Element : Type
  | A | B | C | D | E

-- Define the properties
def in_same_period (e₁ e₂ : Element) : Prop := sorry

def forms_basic_oxide (e : Element) : Prop := sorry

def basicity (e : Element) : ℝ := sorry

def hydride_stability (e : Element) : ℝ := sorry

def ionic_radius (e : Element) : ℝ := sorry

def atomic_number (e : Element) : ℕ := sorry

-- State the theorem
theorem element_order :
  (∀ e₁ e₂ : Element, in_same_period e₁ e₂) →
  forms_basic_oxide Element.A →
  forms_basic_oxide Element.B →
  basicity Element.B > basicity Element.A →
  hydride_stability Element.C > hydride_stability Element.D →
  (∀ e : Element, ionic_radius Element.E ≤ ionic_radius e) →
  (atomic_number Element.B < atomic_number Element.A ∧
   atomic_number Element.A < atomic_number Element.E ∧
   atomic_number Element.E < atomic_number Element.D ∧
   atomic_number Element.D < atomic_number Element.C) :=
by sorry


end NUMINAMATH_CALUDE_element_order_l465_46556


namespace NUMINAMATH_CALUDE_area_perimeter_ratio_equal_l465_46507

/-- An isosceles trapezoid inscribed in a circle -/
structure InscribedIsoscelesTrapezoid where
  /-- Radius of the circle -/
  R : ℝ
  /-- Perimeter of the trapezoid -/
  P : ℝ
  /-- Radius is positive -/
  R_pos : R > 0
  /-- Perimeter is positive -/
  P_pos : P > 0

/-- Theorem: The ratio of the area of the trapezoid to the area of the circle
    is equal to the ratio of the perimeter of the trapezoid to the circumference of the circle -/
theorem area_perimeter_ratio_equal
  (trap : InscribedIsoscelesTrapezoid) :
  (trap.P * trap.R / 2) / (Real.pi * trap.R^2) = trap.P / (2 * Real.pi * trap.R) :=
sorry

end NUMINAMATH_CALUDE_area_perimeter_ratio_equal_l465_46507


namespace NUMINAMATH_CALUDE_vector_problem_l465_46508

/-- Define a 2D vector -/
def Vector2D := ℝ × ℝ

/-- Check if two vectors are collinear -/
def collinear (v w : Vector2D) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

/-- Dot product of two vectors -/
def dot_product (v w : Vector2D) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Check if two vectors are perpendicular -/
def perpendicular (v w : Vector2D) : Prop :=
  dot_product v w = 0

/-- The main theorem -/
theorem vector_problem :
  ∀ (m : ℝ),
  let a : Vector2D := (2, 1)
  let b : Vector2D := (3, -1)
  let c : Vector2D := (3, m)
  (collinear a c → m = 3/2) ∧
  (perpendicular (a.1 - 2*b.1, a.2 - 2*b.2) c → m = 4) :=
sorry

end NUMINAMATH_CALUDE_vector_problem_l465_46508


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l465_46549

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a - Complex.I) / (2 + Complex.I) = Complex.I * b) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l465_46549


namespace NUMINAMATH_CALUDE_min_value_problem_l465_46577

theorem min_value_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 * y * z) / 324 + (144 * y) / (x * z) + 9 / (4 * x * y^2) ≥ 3 ∧
  ((x^2 * y * z) / 324 + (144 * y) / (x * z) + 9 / (4 * x * y^2) = 3 →
    z / (16 * y) + x / 9 ≥ 2) ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧
    (x₀^2 * y₀ * z₀) / 324 + (144 * y₀) / (x₀ * z₀) + 9 / (4 * x₀ * y₀^2) = 3 ∧
    z₀ / (16 * y₀) + x₀ / 9 = 2 ∧
    x₀ = 9 ∧ y₀ = (1/2) ∧ z₀ = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l465_46577


namespace NUMINAMATH_CALUDE_red_snapper_cost_l465_46572

/-- The cost of a Red snapper given the fisherman's daily catch and earnings -/
theorem red_snapper_cost (red_snappers : ℕ) (tunas : ℕ) (tuna_cost : ℚ) (daily_earnings : ℚ) : 
  red_snappers = 8 → tunas = 14 → tuna_cost = 2 → daily_earnings = 52 → 
  (daily_earnings - (tunas * tuna_cost)) / red_snappers = 3 := by
sorry

end NUMINAMATH_CALUDE_red_snapper_cost_l465_46572


namespace NUMINAMATH_CALUDE_new_student_weight_l465_46520

/-- 
Given 5 students with an initial total weight W, 
if replacing two students weighing x and y with a new student 
causes the average weight to decrease by 8 kg, 
then the new student's weight is 40 kg less than x + y.
-/
theorem new_student_weight 
  (W : ℝ) -- Initial total weight of 5 students
  (x y : ℝ) -- Weights of the two replaced students
  (new_avg : ℝ) -- New average weight after replacement
  (h1 : new_avg = (W - x - y + (x + y - 40)) / 5) -- New average calculation
  (h2 : W / 5 - new_avg = 8) -- Average weight decrease
  : x + y - 40 = (x + y) - 40 := by sorry

end NUMINAMATH_CALUDE_new_student_weight_l465_46520


namespace NUMINAMATH_CALUDE_local_face_value_difference_l465_46519

def numeral : ℕ := 96348621

theorem local_face_value_difference :
  let digit : ℕ := 8
  let position : ℕ := 5  -- 1-indexed from right, so 8 is in the 5th position
  let local_value : ℕ := digit * (10 ^ (position - 1))
  let face_value : ℕ := digit
  local_value - face_value = 79992 :=
by sorry

end NUMINAMATH_CALUDE_local_face_value_difference_l465_46519


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l465_46583

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The theorem states that for an arithmetic sequence satisfying given conditions, the 9th term equals 7. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 2 + a 4 = 2)
  (h_fifth : a 5 = 3) :
  a 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l465_46583


namespace NUMINAMATH_CALUDE_unbroken_seashells_l465_46547

theorem unbroken_seashells (total_seashells broken_seashells : ℕ) 
  (h1 : total_seashells = 6)
  (h2 : broken_seashells = 4) :
  total_seashells - broken_seashells = 2 :=
by sorry

end NUMINAMATH_CALUDE_unbroken_seashells_l465_46547


namespace NUMINAMATH_CALUDE_fraction_equality_implies_zero_l465_46509

theorem fraction_equality_implies_zero (x : ℝ) : 
  (4 + x) / (6 + x) = (2 + x) / (3 + x) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_zero_l465_46509


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_when_p_necessary_not_sufficient_l465_46555

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Part 1
theorem range_of_x_when_a_is_one :
  ∀ x : ℝ, (p x 1 ∧ q x) ↔ x ∈ Set.Ioo 2 3 :=
sorry

-- Part 2
theorem range_of_a_when_p_necessary_not_sufficient :
  (∀ x : ℝ, q x → p x 1) ∧ 
  (∃ x : ℝ, p x 1 ∧ ¬q x) ↔
  1 ∈ Set.Ioo 1 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_when_p_necessary_not_sufficient_l465_46555


namespace NUMINAMATH_CALUDE_solve_for_n_l465_46532

/-- The number of balls labeled '2' -/
def n : ℕ := sorry

/-- The total number of balls in the bag -/
def total_balls : ℕ := n + 2

/-- The probability of drawing a ball labeled '2' -/
def prob_2 : ℚ := n / total_balls

theorem solve_for_n : 
  (prob_2 = 1/3) → n = 1 :=
by sorry

end NUMINAMATH_CALUDE_solve_for_n_l465_46532


namespace NUMINAMATH_CALUDE_classroom_size_is_81_l465_46514

/-- Represents the number of students in a classroom with specific shirt and shorts conditions. -/
def classroom_size : ℕ → Prop := fun n =>
  ∃ (striped checkered shorts : ℕ),
    -- Total number of students
    n = striped + checkered
    -- Two-thirds wear striped shirts, one-third wear checkered shirts
    ∧ 3 * striped = 2 * n
    ∧ 3 * checkered = n
    -- Shorts condition
    ∧ shorts = checkered + 19
    -- Striped shirts condition
    ∧ striped = shorts + 8

/-- The number of students in the classroom satisfying the given conditions is 81. -/
theorem classroom_size_is_81 : classroom_size 81 := by
  sorry

end NUMINAMATH_CALUDE_classroom_size_is_81_l465_46514


namespace NUMINAMATH_CALUDE_phi_bounded_by_one_l465_46592

/-- The functional equation satisfied by f and φ -/
def FunctionalEquation (f φ : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) + f (x - y) = 2 * φ y * f x

/-- f is not identically zero -/
def NotIdenticallyZero (f : ℝ → ℝ) : Prop :=
  ∃ x, f x ≠ 0

/-- The absolute value of f is bounded by 1 -/
def BoundedByOne (f : ℝ → ℝ) : Prop :=
  ∀ x, |f x| ≤ 1

/-- The main theorem -/
theorem phi_bounded_by_one
    (f φ : ℝ → ℝ)
    (h_eq : FunctionalEquation f φ)
    (h_nz : NotIdenticallyZero f)
    (h_bound : BoundedByOne f) :
    BoundedByOne φ := by
  sorry

end NUMINAMATH_CALUDE_phi_bounded_by_one_l465_46592


namespace NUMINAMATH_CALUDE_problem_solution_l465_46565

def p (a x : ℝ) : Prop := x^2 - (2*a - 3)*x - 6*a ≤ 0

def q (x : ℝ) : Prop := x - Real.sqrt x - 2 < 0

theorem problem_solution :
  (∀ x, (p 1 x ∧ q x) ↔ (0 ≤ x ∧ x ≤ 2)) ∧
  (∀ a, (∀ x, q x → p a x) ↔ a ≥ 2) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l465_46565


namespace NUMINAMATH_CALUDE_rational_square_plus_one_positive_l465_46528

theorem rational_square_plus_one_positive (a : ℚ) : a^2 + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_square_plus_one_positive_l465_46528


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l465_46541

theorem fraction_product_simplification (a b c : ℝ) 
  (ha : a ≠ 4) (hb : b ≠ 5) (hc : c ≠ 6) : 
  (a - 4) / (6 - c) * (b - 5) / (4 - a) * (c - 6) / (5 - b) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l465_46541


namespace NUMINAMATH_CALUDE_right_triangles_with_perimeter_equal_area_l465_46579

/-- A right triangle with integer side lengths. -/
structure RightTriangle where
  a : ℕ  -- First leg
  b : ℕ  -- Second leg
  c : ℕ  -- Hypotenuse
  right_angle : a^2 + b^2 = c^2

/-- The perimeter of a right triangle. -/
def perimeter (t : RightTriangle) : ℕ :=
  t.a + t.b + t.c

/-- The area of a right triangle. -/
def area (t : RightTriangle) : ℕ :=
  t.a * t.b / 2

/-- The property that the perimeter equals the area. -/
def perimeter_equals_area (t : RightTriangle) : Prop :=
  perimeter t = area t

theorem right_triangles_with_perimeter_equal_area :
  {t : RightTriangle | perimeter_equals_area t} =
  {⟨5, 12, 13, by sorry⟩, ⟨6, 8, 10, by sorry⟩} :=
by sorry

end NUMINAMATH_CALUDE_right_triangles_with_perimeter_equal_area_l465_46579


namespace NUMINAMATH_CALUDE_park_visitors_difference_l465_46516

theorem park_visitors_difference (saturday_visitors : ℕ) (total_visitors : ℕ) 
    (h1 : saturday_visitors = 200)
    (h2 : total_visitors = 440) : 
  total_visitors - 2 * saturday_visitors = 40 := by
  sorry

end NUMINAMATH_CALUDE_park_visitors_difference_l465_46516


namespace NUMINAMATH_CALUDE_geometric_sequence_tan_property_l465_46590

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_tan_property (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a)
  (h_condition : a 2 * a 6 + 2 * (a 4)^2 = Real.pi) :
  Real.tan (a 3 * a 5) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_tan_property_l465_46590


namespace NUMINAMATH_CALUDE_remaining_fabric_is_294_l465_46542

/-- Represents the flag-making scenario with given dimensions and quantities --/
structure FlagScenario where
  total_fabric : ℕ
  square_side : ℕ
  wide_length : ℕ
  wide_width : ℕ
  tall_length : ℕ
  tall_width : ℕ
  square_count : ℕ
  wide_count : ℕ
  tall_count : ℕ

/-- Calculates the remaining fabric after making flags --/
def remaining_fabric (scenario : FlagScenario) : ℕ :=
  scenario.total_fabric -
  (scenario.square_count * scenario.square_side * scenario.square_side +
   scenario.wide_count * scenario.wide_length * scenario.wide_width +
   scenario.tall_count * scenario.tall_length * scenario.tall_width)

/-- Theorem stating that the remaining fabric in the given scenario is 294 square feet --/
theorem remaining_fabric_is_294 (scenario : FlagScenario)
  (h1 : scenario.total_fabric = 1000)
  (h2 : scenario.square_side = 4)
  (h3 : scenario.wide_length = 5)
  (h4 : scenario.wide_width = 3)
  (h5 : scenario.tall_length = 3)
  (h6 : scenario.tall_width = 5)
  (h7 : scenario.square_count = 16)
  (h8 : scenario.wide_count = 20)
  (h9 : scenario.tall_count = 10) :
  remaining_fabric scenario = 294 := by
  sorry


end NUMINAMATH_CALUDE_remaining_fabric_is_294_l465_46542


namespace NUMINAMATH_CALUDE_base_equality_l465_46570

theorem base_equality : ∃ (n k : ℕ), n > 1 ∧ k > 1 ∧ n^2 + 1 = k^4 + k^3 + k + 1 := by
  sorry

end NUMINAMATH_CALUDE_base_equality_l465_46570


namespace NUMINAMATH_CALUDE_unique_function_solution_l465_46504

/-- A function f: ℕ → ℤ is an increasing function that satisfies the given conditions -/
def IsValidFunction (f : ℕ → ℤ) : Prop :=
  (∀ m n : ℕ, m < n → f m < f n) ∧ 
  (f 2 = 7) ∧
  (∀ m n : ℕ, f (m * n) = f m + f n + f m * f n)

/-- The theorem stating that the only function satisfying the conditions is f(n) = n³ - 1 -/
theorem unique_function_solution :
  ∀ f : ℕ → ℤ, IsValidFunction f → ∀ n : ℕ, f n = n^3 - 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_function_solution_l465_46504


namespace NUMINAMATH_CALUDE_gcd_51457_37958_is_1_l465_46536

theorem gcd_51457_37958_is_1 : Nat.gcd 51457 37958 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_51457_37958_is_1_l465_46536


namespace NUMINAMATH_CALUDE_find_a_value_l465_46599

def A (a : ℝ) : Set ℝ := {1, 3, a}
def B (a : ℝ) : Set ℝ := {1, a^2 - a + 1}

theorem find_a_value : ∃ a : ℝ, (B a ⊆ A a) ∧ (a = -1 ∨ a = 2) := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l465_46599


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l465_46560

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides -/
theorem regular_polygon_sides (n : ℕ) : n > 0 → (360 : ℝ) / n = 18 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l465_46560


namespace NUMINAMATH_CALUDE_average_value_of_z_squared_l465_46589

theorem average_value_of_z_squared (z : ℝ) : 
  (z^2 + 3*z^2 + 6*z^2 + 12*z^2 + 24*z^2) / 5 = (46 * z^2) / 5 := by
  sorry

end NUMINAMATH_CALUDE_average_value_of_z_squared_l465_46589


namespace NUMINAMATH_CALUDE_festival_worker_assignment_l465_46502

def number_of_workers : ℕ := 6

def number_of_desks : ℕ := 2

def min_workers_per_desk : ℕ := 2

def ways_to_assign_workers (n : ℕ) (k : ℕ) (min_per_group : ℕ) : ℕ :=
  sorry

theorem festival_worker_assignment :
  ways_to_assign_workers number_of_workers number_of_desks min_workers_per_desk = 28 :=
sorry

end NUMINAMATH_CALUDE_festival_worker_assignment_l465_46502


namespace NUMINAMATH_CALUDE_equal_division_possible_l465_46531

/-- Represents the state of the three vessels -/
structure VesselState :=
  (v1 v2 v3 : ℕ)

/-- Represents a pouring action between two vessels -/
inductive PourAction
  | from1to2 | from1to3 | from2to1 | from2to3 | from3to1 | from3to2

/-- Applies a pouring action to a vessel state -/
def applyPour (state : VesselState) (action : PourAction) : VesselState :=
  sorry

/-- Checks if a vessel state is valid (respects capacities) -/
def isValidState (state : VesselState) : Prop :=
  state.v1 ≤ 3 ∧ state.v2 ≤ 5 ∧ state.v3 ≤ 8

/-- Checks if a sequence of pours is valid -/
def isValidPourSequence (initialState : VesselState) (pours : List PourAction) : Prop :=
  sorry

/-- The theorem stating that it's possible to divide the liquid equally -/
theorem equal_division_possible : ∃ (pours : List PourAction),
  isValidPourSequence ⟨0, 0, 8⟩ pours ∧
  let finalState := pours.foldl applyPour ⟨0, 0, 8⟩
  finalState.v2 = 4 ∧ finalState.v3 = 4 :=
  sorry

end NUMINAMATH_CALUDE_equal_division_possible_l465_46531


namespace NUMINAMATH_CALUDE_divisibility_by_hundred_l465_46585

theorem divisibility_by_hundred (n : ℕ) : 
  ∃ (k : ℕ), 100 ∣ (5^n + 12*n^2 + 12*n + 3) ↔ n = 5*k + 2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_hundred_l465_46585


namespace NUMINAMATH_CALUDE_expression_with_eight_factors_l465_46539

theorem expression_with_eight_factors
  (x y : ℕ)
  (hx_prime : Nat.Prime x)
  (hy_prime : Nat.Prime y)
  (hx_odd : Odd x)
  (hy_odd : Odd y)
  (hxy_lt : x < y) :
  (Finset.filter (fun d => (x^3 * y) % d = 0) (Finset.range (x^3 * y + 1))).card = 8 :=
sorry

end NUMINAMATH_CALUDE_expression_with_eight_factors_l465_46539


namespace NUMINAMATH_CALUDE_algebraic_simplification_l465_46554

theorem algebraic_simplification (y : ℝ) (h : y ≠ 0) :
  (20 * y^3) * (8 * y^2) * (1 / (4*y)^3) = (5/2) * y^2 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l465_46554


namespace NUMINAMATH_CALUDE_circular_seating_arrangement_l465_46524

/-- Given a circular arrangement of students where the 5th position
    is opposite the 20th position, prove that there are 32 students in total. -/
theorem circular_seating_arrangement (n : ℕ) 
  (h : n > 0)  -- Ensure positive number of students
  (opposite : ∀ (a b : ℕ), a ≤ n → b ≤ n → (a + n / 2) % n = b % n → a = 5 ∧ b = 20) :
  n = 32 := by
  sorry

end NUMINAMATH_CALUDE_circular_seating_arrangement_l465_46524


namespace NUMINAMATH_CALUDE_total_cost_packages_A_and_B_l465_46561

/-- Represents a subscription package with monthly cost, duration, and discount rate -/
structure Package where
  monthlyCost : ℝ
  duration : ℕ
  discountRate : ℝ

/-- Calculates the discounted cost of a package -/
def discountedCost (p : Package) : ℝ :=
  p.monthlyCost * p.duration * (1 - p.discountRate)

/-- The newspaper subscription packages -/
def packageA : Package := { monthlyCost := 10, duration := 6, discountRate := 0.1 }
def packageB : Package := { monthlyCost := 12, duration := 9, discountRate := 0.15 }

/-- Theorem stating the total cost of subscribing to Package A followed by Package B -/
theorem total_cost_packages_A_and_B :
  discountedCost packageA + discountedCost packageB = 145.80 := by
  sorry

#eval discountedCost packageA + discountedCost packageB

end NUMINAMATH_CALUDE_total_cost_packages_A_and_B_l465_46561


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l465_46529

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = -b / a := by sorry

theorem sum_of_roots_specific_equation :
  let x₁ := (-(-7) + Real.sqrt ((-7)^2 - 4*1*(-14))) / (2*1)
  let x₂ := (-(-7) - Real.sqrt ((-7)^2 - 4*1*(-14))) / (2*1)
  x₁ + x₂ = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l465_46529


namespace NUMINAMATH_CALUDE_arrangement_count_l465_46526

-- Define the number of children
def n : ℕ := 6

-- Define the number of odd positions available for the specific child
def odd_positions : ℕ := 3

-- Define the function to calculate the number of arrangements
def arrangements (n : ℕ) (odd_positions : ℕ) : ℕ :=
  odd_positions * Nat.factorial (n - 1)

-- Theorem statement
theorem arrangement_count :
  arrangements n odd_positions = 360 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l465_46526


namespace NUMINAMATH_CALUDE_buckingham_palace_visitors_l465_46517

def visitors_previous_day : ℕ := 100
def additional_visitors : ℕ := 566

theorem buckingham_palace_visitors :
  visitors_previous_day + additional_visitors = 666 := by
  sorry

end NUMINAMATH_CALUDE_buckingham_palace_visitors_l465_46517


namespace NUMINAMATH_CALUDE_trig_identity_proof_l465_46581

/-- Proves that cos(70°)sin(80°) + cos(20°)sin(10°) = 1/2 -/
theorem trig_identity_proof : 
  Real.cos (70 * π / 180) * Real.sin (80 * π / 180) + 
  Real.cos (20 * π / 180) * Real.sin (10 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l465_46581


namespace NUMINAMATH_CALUDE_room_occupancy_l465_46512

theorem room_occupancy (total_chairs : ℕ) (seated_people : ℕ) (total_people : ℕ) : 
  (5 : ℚ) / 6 * total_people = seated_people →
  (5 : ℚ) / 6 * total_chairs = seated_people →
  total_chairs - seated_people = 10 →
  total_people = 60 := by
sorry

end NUMINAMATH_CALUDE_room_occupancy_l465_46512


namespace NUMINAMATH_CALUDE_money_distribution_l465_46501

theorem money_distribution (total : ℝ) (p q r : ℝ) : 
  p + q + r = total →
  p / q = 3 / 7 →
  q / r = 7 / 12 →
  q - p = 2400 →
  r - q = 3000 :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l465_46501


namespace NUMINAMATH_CALUDE_integer_solution_cyclic_equation_l465_46578

theorem integer_solution_cyclic_equation :
  ∀ x y z : ℤ, (x + y + z)^5 = 80*x*y*z*(x^2 + y^2 + z^2) →
  (∃ a : ℤ, (x = a ∧ y = -a ∧ z = 0) ∨
            (x = a ∧ y = 0 ∧ z = -a) ∨
            (x = 0 ∧ y = a ∧ z = -a) ∨
            (x = -a ∧ y = a ∧ z = 0) ∨
            (x = -a ∧ y = 0 ∧ z = a) ∨
            (x = 0 ∧ y = -a ∧ z = a)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solution_cyclic_equation_l465_46578


namespace NUMINAMATH_CALUDE_function_satisfying_conditions_l465_46544

theorem function_satisfying_conditions (f : ℝ → ℝ) : 
  (∀ x : ℝ, x ≠ 0 → f x ≠ 0) →
  f 1 = 1 →
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → x + y ≠ 0 → f (1 / (x + y)) = f (1 / x) + f (1 / y)) →
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → x + y ≠ 0 → (x + y) * f (x + y) = x * y * f x * f y) →
  (∀ x : ℝ, x ≠ 0 → f x = 1 / x) :=
by sorry

end NUMINAMATH_CALUDE_function_satisfying_conditions_l465_46544


namespace NUMINAMATH_CALUDE_line_slope_l465_46557

/-- The slope of the line given by the equation 4y + 5x = 20 is -5/4 -/
theorem line_slope (x y : ℝ) : 4 * y + 5 * x = 20 → (y - 5) / (-5 / 4) = x := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l465_46557


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l465_46563

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * x + a ≥ 0) ↔ a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l465_46563


namespace NUMINAMATH_CALUDE_magnitude_of_z_l465_46510

def z : ℂ := (1 + Complex.I) * (2 - Complex.I)

theorem magnitude_of_z : Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l465_46510


namespace NUMINAMATH_CALUDE_parabola_normals_intersection_l465_46593

/-- The condition for three distinct points on a parabola to have intersecting normals -/
theorem parabola_normals_intersection
  (a b c : ℝ)
  (h_distinct : (a - b) * (b - c) * (c - a) ≠ 0)
  (h_parabola : ∀ (x : ℝ), (x = a ∨ x = b ∨ x = c) → ∃ (y : ℝ), y = x^2) :
  (∃ (p : ℝ × ℝ),
    (∀ (x y : ℝ), (x = a ∨ x = b ∨ x = c) →
      (y - x^2) = -(1 / (2*x)) * (p.1 - x) ∧ p.2 = y)) ↔
  a + b + c = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_normals_intersection_l465_46593


namespace NUMINAMATH_CALUDE_no_integer_solution_l465_46550

theorem no_integer_solution : ¬ ∃ (m n : ℤ), m^2 + 1954 = n^2 := by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l465_46550


namespace NUMINAMATH_CALUDE_right_triangle_inscribed_circle_area_l465_46527

theorem right_triangle_inscribed_circle_area
  (r : ℝ) (c : ℝ) (h_r : r = 5) (h_c : c = 34) :
  let s := (c + 2 * r + (c - 2 * r)) / 2
  r * s = 195 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_inscribed_circle_area_l465_46527


namespace NUMINAMATH_CALUDE_cos_two_theta_collinear_vectors_l465_46584

/-- Given two vectors AB and BC in 2D space, and that points A, B, and C are collinear,
    prove that cos(2θ) = 7/9 where θ is the angle in the definition of BC. -/
theorem cos_two_theta_collinear_vectors 
  (AB : ℝ × ℝ) 
  (BC : ℝ → ℝ × ℝ) 
  (h_AB : AB = (-1, -3))
  (h_BC : ∀ θ, BC θ = (2 * Real.sin θ, 2))
  (h_collinear : ∀ θ, ∃ k : ℝ, AB = k • BC θ) :
  ∃ θ, Real.cos (2 * θ) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_theta_collinear_vectors_l465_46584


namespace NUMINAMATH_CALUDE_green_balls_removal_l465_46523

theorem green_balls_removal (total : ℕ) (initial_green_percentage : ℚ) 
  (final_green_percentage : ℚ) (removed : ℕ) : 
  total = 600 →
  initial_green_percentage = 7/10 →
  final_green_percentage = 3/5 →
  removed = 150 →
  (initial_green_percentage * total - removed) / (total - removed) = final_green_percentage := by
sorry

end NUMINAMATH_CALUDE_green_balls_removal_l465_46523


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l465_46588

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 / x + 4 / y ≥ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ 1 / x₀ + 4 / y₀ = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l465_46588


namespace NUMINAMATH_CALUDE_tank_plastering_cost_per_sqm_l465_46575

/-- Given a tank with specified dimensions and total plastering cost, 
    calculate the cost per square meter for plastering. -/
theorem tank_plastering_cost_per_sqm 
  (length width depth : ℝ) 
  (total_cost : ℝ) 
  (h1 : length = 25) 
  (h2 : width = 12) 
  (h3 : depth = 6) 
  (h4 : total_cost = 186) : 
  total_cost / (length * width + 2 * length * depth + 2 * width * depth) = 0.25 := by
  sorry

#check tank_plastering_cost_per_sqm

end NUMINAMATH_CALUDE_tank_plastering_cost_per_sqm_l465_46575


namespace NUMINAMATH_CALUDE_x_value_l465_46518

theorem x_value (x y : ℝ) (h1 : 2 * x - y = 14) (h2 : y = 2) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l465_46518


namespace NUMINAMATH_CALUDE_max_page_number_proof_l465_46596

/-- Counts the number of '5' digits used in numbering pages from 1 to n --/
def count_fives (n : ℕ) : ℕ := sorry

/-- The highest page number that can be labeled with 16 '5' digits --/
def max_page_number : ℕ := 75

theorem max_page_number_proof :
  count_fives max_page_number ≤ 16 ∧
  ∀ m : ℕ, m > max_page_number → count_fives m > 16 :=
sorry

end NUMINAMATH_CALUDE_max_page_number_proof_l465_46596


namespace NUMINAMATH_CALUDE_arithmetic_sequence_exponents_l465_46545

theorem arithmetic_sequence_exponents (a b : ℝ) (m : ℝ) : 
  a > 0 → b > 0 → 
  2^a = m → 3^b = m → 
  2 * a * b = a + b → 
  m = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_exponents_l465_46545


namespace NUMINAMATH_CALUDE_sum_of_roots_is_3pi_l465_46594

-- Define the equation
def tanEquation (x : ℝ) : Prop := Real.tan x ^ 2 - 12 * Real.tan x + 4 = 0

-- Define the interval
def inInterval (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2 * Real.pi

-- Theorem statement
theorem sum_of_roots_is_3pi :
  ∃ (roots : Finset ℝ), 
    (∀ x ∈ roots, tanEquation x ∧ inInterval x) ∧
    (∀ x, tanEquation x ∧ inInterval x → x ∈ roots) ∧
    (Finset.sum roots id = 3 * Real.pi) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_3pi_l465_46594


namespace NUMINAMATH_CALUDE_expression_equality_l465_46569

theorem expression_equality : (2004 - (2011 - 196)) + (2011 - (196 - 2004)) = 4008 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l465_46569


namespace NUMINAMATH_CALUDE_closer_to_origin_l465_46582

theorem closer_to_origin : abs (-2 : ℝ) < abs (3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_closer_to_origin_l465_46582


namespace NUMINAMATH_CALUDE_vector_decomposition_l465_46591

def x : Fin 3 → ℝ := ![(-9 : ℝ), 5, 5]
def p : Fin 3 → ℝ := ![(4 : ℝ), 1, 1]
def q : Fin 3 → ℝ := ![(2 : ℝ), 0, -3]
def r : Fin 3 → ℝ := ![(-1 : ℝ), 2, 1]

theorem vector_decomposition :
  x = (-1 : ℝ) • p + (-1 : ℝ) • q + (3 : ℝ) • r :=
by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l465_46591


namespace NUMINAMATH_CALUDE_profit_sharing_ratio_l465_46551

/-- Represents an investment in a business. -/
structure Investment where
  amount : ℕ
  duration : ℕ

/-- Calculates the total investment value considering the amount and duration. -/
def investmentValue (i : Investment) : ℕ := i.amount * i.duration

/-- Represents the ratio of two numbers as a pair of natural numbers. -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Theorem stating that the profit sharing ratio is 2:3 given the investments of A and B. -/
theorem profit_sharing_ratio 
  (a : Investment) 
  (b : Investment) 
  (h1 : a.amount = 3500) 
  (h2 : a.duration = 12) 
  (h3 : b.amount = 9000) 
  (h4 : b.duration = 7) : 
  ∃ (r : Ratio), r.numerator = 2 ∧ r.denominator = 3 ∧ 
  investmentValue a * r.denominator = investmentValue b * r.numerator := by
  sorry


end NUMINAMATH_CALUDE_profit_sharing_ratio_l465_46551


namespace NUMINAMATH_CALUDE_students_on_right_side_l465_46586

theorem students_on_right_side (total : ℕ) (left : ℕ) (right : ℕ) : 
  total = 63 → left = 36 → right = total - left → right = 27 := by
  sorry

end NUMINAMATH_CALUDE_students_on_right_side_l465_46586


namespace NUMINAMATH_CALUDE_max_d_value_l465_46515

def a (n : ℕ) : ℕ := 100 + n^2

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value : ∃ (k : ℕ), ∀ (n : ℕ), n > 0 → d n ≤ k ∧ ∃ (m : ℕ), m > 0 ∧ d m = k :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l465_46515


namespace NUMINAMATH_CALUDE_rectangular_plot_area_l465_46522

/-- A rectangular plot with length thrice its width and width of 12 meters has an area of 432 square meters. -/
theorem rectangular_plot_area : 
  ∀ (width length area : ℝ),
  width = 12 →
  length = 3 * width →
  area = length * width →
  area = 432 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_area_l465_46522


namespace NUMINAMATH_CALUDE_sin_period_omega_l465_46559

/-- 
Given a function y = sin(ωx - π/3) with ω > 0 and a minimum positive period of π,
prove that ω = 2.
-/
theorem sin_period_omega (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ x, ∃ y, y = Real.sin (ω * x - π / 3)) 
  (h3 : ∀ T > 0, (∀ x, Real.sin (ω * (x + T) - π / 3) = Real.sin (ω * x - π / 3)) → T ≥ π) 
  (h4 : ∀ x, Real.sin (ω * (x + π) - π / 3) = Real.sin (ω * x - π / 3)) : ω = 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_period_omega_l465_46559


namespace NUMINAMATH_CALUDE_equation_system_solution_l465_46574

theorem equation_system_solution :
  ∀ (x y z : ℝ),
    z ≠ 0 →
    3 * x - 5 * y - z = 0 →
    2 * x + 4 * y - 16 * z = 0 →
    (x^2 + 4*x*y) / (2*y^2 + z^2) = 4.35 := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l465_46574


namespace NUMINAMATH_CALUDE_bryans_collection_total_l465_46513

/-- Calculates the total number of reading materials in Bryan's collection --/
def total_reading_materials (
  num_shelves : ℕ
) (
  books_per_shelf : ℕ
) (
  magazines_per_shelf : ℕ
) (
  newspapers_per_shelf : ℕ
) (
  graphic_novels_per_shelf : ℕ
) : ℕ :=
  num_shelves * (books_per_shelf + magazines_per_shelf + newspapers_per_shelf + graphic_novels_per_shelf)

/-- Proves that Bryan's collection contains 4810 reading materials --/
theorem bryans_collection_total :
  total_reading_materials 37 23 61 17 29 = 4810 := by
  sorry

end NUMINAMATH_CALUDE_bryans_collection_total_l465_46513


namespace NUMINAMATH_CALUDE_min_digit_sum_of_sum_l465_46558

/-- Two-digit number type -/
def TwoDigitNumber := { n : ℕ // n ≥ 10 ∧ n ≤ 99 }

/-- Function to get the digits of a natural number -/
def digits (n : ℕ) : List ℕ := sorry

/-- Function to sum the digits of a natural number -/
def digitSum (n : ℕ) : ℕ := (digits n).sum

/-- Predicate to check if two two-digit numbers have exactly one common digit -/
def hasOneCommonDigit (a b : TwoDigitNumber) : Prop := sorry

/-- Theorem: The smallest possible digit sum of S, where S is the sum of two two-digit numbers
    with exactly one common digit, and S is a three-digit number, is 2. -/
theorem min_digit_sum_of_sum (a b : TwoDigitNumber) 
  (h1 : hasOneCommonDigit a b) 
  (h2 : a.val + b.val ≥ 100 ∧ a.val + b.val ≤ 999) : 
  ∃ (S : ℕ), S = a.val + b.val ∧ digitSum S = 2 ∧ 
  ∀ (T : ℕ), T = a.val + b.val → digitSum T ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_digit_sum_of_sum_l465_46558


namespace NUMINAMATH_CALUDE_parallelogram_area_l465_46553

/-- The area of a parallelogram with given side lengths and included angle -/
theorem parallelogram_area (a b : ℝ) (θ : Real) (ha : a = 32) (hb : b = 18) (hθ : θ = 75 * π / 180) :
  abs (a * b * Real.sin θ - 556.36) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l465_46553


namespace NUMINAMATH_CALUDE_count_three_digit_Q_equal_l465_46595

def Q (n : ℕ) : ℕ := 
  n % 3 + n % 5 + n % 7 + n % 11

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem count_three_digit_Q_equal : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, is_three_digit n ∧ Q n = Q (n + 1)) ∧ 
    S.card = 9 ∧
    (∀ n, is_three_digit n → Q n = Q (n + 1) → n ∈ S) :=
sorry

end NUMINAMATH_CALUDE_count_three_digit_Q_equal_l465_46595


namespace NUMINAMATH_CALUDE_shoe_size_increase_l465_46535

/-- Represents the increase in length (in inches) for each unit increase in shoe size -/
def length_increase : ℝ := 0.2

/-- The smallest shoe size -/
def min_size : ℕ := 8

/-- The largest shoe size -/
def max_size : ℕ := 17

/-- The length of the size 15 shoe (in inches) -/
def size_15_length : ℝ := 10.4

/-- The ratio of the largest size length to the smallest size length -/
def size_ratio : ℝ := 1.2

theorem shoe_size_increase :
  (min_size : ℝ) + (max_size - min_size) * length_increase = (min_size : ℝ) * size_ratio ∧
  (min_size : ℝ) + (15 - min_size) * length_increase = size_15_length ∧
  length_increase = 0.2 := by sorry

end NUMINAMATH_CALUDE_shoe_size_increase_l465_46535


namespace NUMINAMATH_CALUDE_scooter_safety_gear_cost_increase_l465_46505

/-- The percent increase in the combined cost of a scooter and safety gear set --/
theorem scooter_safety_gear_cost_increase (scooter_cost safety_gear_cost : ℝ)
  (scooter_increase safety_gear_increase : ℝ) :
  scooter_cost = 200 →
  safety_gear_cost = 50 →
  scooter_increase = 0.08 →
  safety_gear_increase = 0.15 →
  let new_scooter_cost := scooter_cost * (1 + scooter_increase)
  let new_safety_gear_cost := safety_gear_cost * (1 + safety_gear_increase)
  let total_original_cost := scooter_cost + safety_gear_cost
  let total_new_cost := new_scooter_cost + new_safety_gear_cost
  let percent_increase := (total_new_cost - total_original_cost) / total_original_cost * 100
  ∃ ε > 0, |percent_increase - 9| < ε :=
by sorry

end NUMINAMATH_CALUDE_scooter_safety_gear_cost_increase_l465_46505


namespace NUMINAMATH_CALUDE_ratio_inequality_l465_46598

theorem ratio_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / b + b / c + c / a ≤ a^2 / b^2 + b^2 / c^2 + c^2 / a^2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_inequality_l465_46598


namespace NUMINAMATH_CALUDE_largest_base5_5digit_in_base10_l465_46506

/-- The largest base-5 number with five digits -/
def largest_base5_5digit : ℕ := 44444

/-- Convert a base-5 number to base 10 -/
def base5_to_base10 (n : ℕ) : ℕ :=
  (n / 10000) * 5^4 + ((n / 1000) % 5) * 5^3 + ((n / 100) % 5) * 5^2 + ((n / 10) % 5) * 5^1 + (n % 5) * 5^0

theorem largest_base5_5digit_in_base10 :
  base5_to_base10 largest_base5_5digit = 3124 := by
  sorry

end NUMINAMATH_CALUDE_largest_base5_5digit_in_base10_l465_46506


namespace NUMINAMATH_CALUDE_triangle_area_l465_46566

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π) →
  (b * Real.sin C + c * Real.sin B = 4 * a * Real.sin B * Real.sin C) →
  (b^2 + c^2 - a^2 = 8) →
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧ S = (2 * Real.sqrt 3) / 3) :=
by sorry

#check triangle_area

end NUMINAMATH_CALUDE_triangle_area_l465_46566


namespace NUMINAMATH_CALUDE_share_change_l465_46521

theorem share_change (total money : ℝ) (ostap_share kisa_share : ℝ) 
  (h1 : ostap_share + kisa_share = total)
  (h2 : ostap_share = 1.5 * kisa_share) :
  let new_ostap_share := 1.5 * ostap_share
  let new_kisa_share := total - new_ostap_share
  new_kisa_share = 0.25 * kisa_share := by
sorry

end NUMINAMATH_CALUDE_share_change_l465_46521


namespace NUMINAMATH_CALUDE_complex_solution_l465_46540

def determinant (a b c d : ℂ) : ℂ := a * d - b * c

theorem complex_solution (z : ℂ) (h : determinant z 1 z (2 * Complex.I) = 3 + 2 * Complex.I) :
  z = (1 / 5 : ℂ) - (8 / 5 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_solution_l465_46540


namespace NUMINAMATH_CALUDE_rosie_pies_l465_46567

/-- Given that Rosie can make 3 pies out of 12 apples, 
    prove that she can make 9 pies out of 36 apples. -/
theorem rosie_pies (apples_per_batch : ℕ) (pies_per_batch : ℕ) 
  (h1 : apples_per_batch = 12) 
  (h2 : pies_per_batch = 3) 
  (h3 : 36 = 3 * apples_per_batch) : 
  (36 / (apples_per_batch / pies_per_batch) : ℕ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_rosie_pies_l465_46567


namespace NUMINAMATH_CALUDE_range_of_b_and_m_l465_46564

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := x - 1

-- Define the set of b values
def B : Set ℝ := {b | b < 0 ∨ b > 4}

-- Define the function F
def F (x m : ℝ) : ℝ := x^2 - m*(x - 1) + 1 - m - m^2

-- Define the set of m values
def M : Set ℝ := {m | -Real.sqrt (4/5) ≤ m ∧ m ≤ Real.sqrt (4/5) ∨ m ≥ 2}

theorem range_of_b_and_m :
  (∀ b : ℝ, (∃ x : ℝ, f x < b * g x) ↔ b ∈ B) ∧
  (∀ m : ℝ, (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → |F x m| < |F y m|) → m ∈ M) :=
sorry

end NUMINAMATH_CALUDE_range_of_b_and_m_l465_46564


namespace NUMINAMATH_CALUDE_pizza_slices_per_adult_l465_46543

theorem pizza_slices_per_adult (num_adults num_children num_pizzas slices_per_pizza slices_per_child : ℕ) :
  num_adults = 2 →
  num_children = 6 →
  num_pizzas = 3 →
  slices_per_pizza = 4 →
  slices_per_child = 1 →
  (num_pizzas * slices_per_pizza - num_children * slices_per_child) / num_adults = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_per_adult_l465_46543


namespace NUMINAMATH_CALUDE_equality_statements_l465_46548

theorem equality_statements :
  (∀ a b : ℝ, a - 3 = b - 3 → a = b) ∧
  (∀ a b m : ℝ, m ≠ 0 → a / m = b / m → a = b) := by sorry

end NUMINAMATH_CALUDE_equality_statements_l465_46548


namespace NUMINAMATH_CALUDE_function_form_exists_l465_46537

noncomputable def f (a b c x : ℝ) : ℝ := a * b^x + c

theorem function_form_exists :
  ∃ (a b c : ℝ),
    (∀ x : ℝ, x ≥ 0 → -2 ≤ f a b c x ∧ f a b c x < 3) ∧
    (0 < b ∧ b < 1) ∧
    (∀ x : ℝ, x ≥ 0 → f a b c x = -5 * b^x + 3) :=
by sorry

end NUMINAMATH_CALUDE_function_form_exists_l465_46537


namespace NUMINAMATH_CALUDE_people_speaking_neither_language_l465_46503

theorem people_speaking_neither_language (total : ℕ) (latin : ℕ) (french : ℕ) (both : ℕ) 
  (h_total : total = 25)
  (h_latin : latin = 13)
  (h_french : french = 15)
  (h_both : both = 9)
  : total - (latin + french - both) = 6 := by
  sorry

end NUMINAMATH_CALUDE_people_speaking_neither_language_l465_46503


namespace NUMINAMATH_CALUDE_scaled_job_workforce_l465_46587

/-- Calculates the number of men needed for a scaled job given the original workforce and timelines. -/
def men_needed_for_scaled_job (original_men : ℕ) (original_days : ℕ) (scale_factor : ℕ) (new_days : ℕ) : ℕ :=
  (original_men * original_days * scale_factor) / new_days

/-- Proves that 600 men are needed for a job 3 times the original size, given the original conditions. -/
theorem scaled_job_workforce :
  men_needed_for_scaled_job 250 16 3 20 = 600 := by
  sorry

#eval men_needed_for_scaled_job 250 16 3 20

end NUMINAMATH_CALUDE_scaled_job_workforce_l465_46587


namespace NUMINAMATH_CALUDE_specific_number_probability_l465_46546

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The total number of possible outcomes when tossing two dice -/
def total_outcomes : ℕ := num_sides * num_sides

/-- The number of favorable outcomes for a specific type of number -/
def favorable_outcomes : ℕ := 15

/-- The probability of getting a specific type of number when tossing two dice -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem specific_number_probability :
  probability = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_specific_number_probability_l465_46546


namespace NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_sum_l465_46511

/-- The sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) (a : ℕ → ℚ) : ℚ := (n : ℚ) * (a 1 + a n) / 2

/-- An arithmetic sequence with common difference -1 -/
def arithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = a n - 1

/-- S_1, S_2, S_4 form a geometric sequence -/
def geometricSequence (a : ℕ → ℚ) : Prop :=
  (S 2 a) ^ 2 = (S 1 a) * (S 4 a)

theorem arithmetic_sequence_with_geometric_sum 
  (a : ℕ → ℚ) 
  (h1 : arithmeticSequence a) 
  (h2 : geometricSequence a) : 
  ∀ n, a n = 1/2 - n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_sum_l465_46511


namespace NUMINAMATH_CALUDE_parallelogram_product_l465_46571

/-- Given a parallelogram EFGH with side lengths as specified, 
    prove that the product of x and y is 57√2 -/
theorem parallelogram_product (x y : ℝ) : 
  58 = 3 * x + 1 →   -- EF = GH
  2 * y^2 = 36 →     -- FG = HE
  x * y = 57 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_product_l465_46571


namespace NUMINAMATH_CALUDE_sphere_cylinder_volume_difference_l465_46530

/-- The volume of space inside a sphere and outside an inscribed right cylinder -/
theorem sphere_cylinder_volume_difference (r_sphere r_cylinder : ℝ) (h_sphere : r_sphere = 5) (h_cylinder : r_cylinder = 3) :
  ∃ (h_cylinder : ℝ),
    (4 / 3 * π * r_sphere ^ 3) - (π * r_cylinder ^ 2 * h_cylinder) = (284 / 3 : ℝ) * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_cylinder_volume_difference_l465_46530


namespace NUMINAMATH_CALUDE_smallest_k_for_digit_sum_945_l465_46525

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The number formed by k repetitions of the digit 7 -/
def repeated_sevens (k : ℕ) : ℕ := sorry

theorem smallest_k_for_digit_sum_945 :
  (∀ k < 312, sum_of_digits (7 * repeated_sevens k) < 945) ∧
  sum_of_digits (7 * repeated_sevens 312) = 945 := by sorry

end NUMINAMATH_CALUDE_smallest_k_for_digit_sum_945_l465_46525


namespace NUMINAMATH_CALUDE_intersection_point_l465_46568

-- Define the two lines
def line1 (x y : ℚ) : Prop := y = -3 * x + 1
def line2 (x y : ℚ) : Prop := y + 5 = 15 * x - 2

-- Theorem statement
theorem intersection_point :
  ∃ (x y : ℚ), line1 x y ∧ line2 x y ∧ x = 1/3 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l465_46568


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_plus_100_l465_46534

theorem arithmetic_series_sum_plus_100 : 
  let a₁ : ℕ := 10
  let aₙ : ℕ := 100
  let d : ℕ := 1
  let n : ℕ := (aₙ - a₁) / d + 1
  let series_sum : ℕ := n * (a₁ + aₙ) / 2
  series_sum + 100 = 5105 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_plus_100_l465_46534


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l465_46573

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (4 * x + 11) = 9 → x = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l465_46573


namespace NUMINAMATH_CALUDE_intersection_segment_length_l465_46533

/-- Line l in Cartesian coordinates -/
def line_l (x y : ℝ) : Prop := x + y = 0

/-- Curve C in Cartesian coordinates -/
def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

/-- The length of segment AB formed by the intersection of line l and curve C -/
theorem intersection_segment_length :
  ∃ (A B : ℝ × ℝ),
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_segment_length_l465_46533


namespace NUMINAMATH_CALUDE_factor_implies_b_equals_one_l465_46580

theorem factor_implies_b_equals_one (a b : ℤ) :
  (∃ c d : ℤ, ∀ x, (x^2 + x - 2) * (c*x + d) = a*x^3 - b*x^2 + x + 2) →
  b = 1 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_b_equals_one_l465_46580


namespace NUMINAMATH_CALUDE_percentage_of_older_female_students_l465_46500

theorem percentage_of_older_female_students
  (total_students : ℝ)
  (h1 : total_students > 0)
  (h2 : 0.4 * total_students = male_students)
  (h3 : 0.5 * male_students = older_male_students)
  (h4 : 0.56 * total_students = younger_students)
  : 0.4 * (total_students - male_students) = older_female_students :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_of_older_female_students_l465_46500
