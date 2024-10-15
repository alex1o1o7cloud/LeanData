import Mathlib

namespace NUMINAMATH_CALUDE_min_value_expression_l1238_123864

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  4 * a^3 + 8 * b^3 + 27 * c^3 + 1 / (3 * a * b * c) ≥ 6 ∧
  (4 * a^3 + 8 * b^3 + 27 * c^3 + 1 / (3 * a * b * c) = 6 ↔
    a = 1 / Real.rpow 6 (1/3) ∧ b = 1 / Real.rpow 12 (1/3) ∧ c = 1 / Real.rpow 54 (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1238_123864


namespace NUMINAMATH_CALUDE_extremum_at_one_l1238_123871

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

theorem extremum_at_one (a : ℝ) : 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1 ∨ f a x ≥ f a 1) → 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_extremum_at_one_l1238_123871


namespace NUMINAMATH_CALUDE_abs_sum_complex_roots_l1238_123851

theorem abs_sum_complex_roots (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1)
  (h4 : a^3 / (b^2 * c) + b^3 / (a^2 * c) + c^3 / (a^2 * b) = 1) :
  Complex.abs (a + b + c) = 1 ∨ Complex.abs (a + b + c) = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_complex_roots_l1238_123851


namespace NUMINAMATH_CALUDE_quadratic_coefficients_theorem_l1238_123865

/-- A quadratic function f(x) = ax^2 + bx + c satisfying specific conditions -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The set of possible coefficient triples (a, b, c) for the quadratic function -/
def PossibleCoefficients : Set (ℝ × ℝ × ℝ) :=
  {(4, -16, 14), (2, -6, 2), (2, -10, 10)}

theorem quadratic_coefficients_theorem (a b c : ℝ) :
  a > 0 ∧
  (∀ x ∈ ({1, 2, 3} : Set ℝ), |QuadraticFunction a b c x| = 2) →
  (a, b, c) ∈ PossibleCoefficients := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_theorem_l1238_123865


namespace NUMINAMATH_CALUDE_max_intersections_l1238_123881

/-- Represents a polynomial of degree 5 or less -/
def Polynomial5 := Fin 6 → ℝ

/-- The set of ten 5-degree polynomials -/
def TenPolynomials := Fin 10 → Polynomial5

/-- A linear function representing an arithmetic sequence -/
def ArithmeticSequence := ℝ → ℝ

/-- The number of intersections between a polynomial and a linear function -/
def intersections (p : Polynomial5) (f : ArithmeticSequence) : ℕ :=
  sorry

/-- The total number of intersections between ten polynomials and a linear function -/
def totalIntersections (polynomials : TenPolynomials) (f : ArithmeticSequence) : ℕ :=
  sorry

theorem max_intersections (polynomials : TenPolynomials) (f : ArithmeticSequence) :
  totalIntersections polynomials f ≤ 50 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_l1238_123881


namespace NUMINAMATH_CALUDE_det_specific_matrix_l1238_123852

theorem det_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![1, 2, 0; 4, 5, -3; 7, 8, 6]
  Matrix.det A = -36 := by
  sorry

end NUMINAMATH_CALUDE_det_specific_matrix_l1238_123852


namespace NUMINAMATH_CALUDE_B_power_five_eq_scalar_multiple_l1238_123822

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 6]

theorem B_power_five_eq_scalar_multiple :
  B^5 = (4096 : ℝ) • B := by sorry

end NUMINAMATH_CALUDE_B_power_five_eq_scalar_multiple_l1238_123822


namespace NUMINAMATH_CALUDE_vector_equation_y_axis_l1238_123884

/-- Given points O, A, and B in the plane, and a vector equation for OP,
    prove that if P is on the y-axis, then m = 2/3 -/
theorem vector_equation_y_axis (O A B P : ℝ × ℝ) (m : ℝ) :
  O = (0, 0) →
  A = (-1, 3) →
  B = (2, -4) →
  P.1 = 0 →
  P = (2 * A.1 + m * (B.1 - A.1), 2 * A.2 + m * (B.2 - A.2)) →
  m = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_y_axis_l1238_123884


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l1238_123824

theorem smallest_number_divisibility (n : ℕ) : n = 4722 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k₁ k₂ k₃ k₄ : ℕ, 
    m + 3 = 27 * k₁ ∧ 
    m + 3 = 35 * k₂ ∧ 
    m + 3 = 25 * k₃ ∧ 
    m + 3 = 21 * k₄)) ∧ 
  (∃ k₁ k₂ k₃ k₄ : ℕ, 
    n + 3 = 27 * k₁ ∧ 
    n + 3 = 35 * k₂ ∧ 
    n + 3 = 25 * k₃ ∧ 
    n + 3 = 21 * k₄) := by
  sorry

#check smallest_number_divisibility

end NUMINAMATH_CALUDE_smallest_number_divisibility_l1238_123824


namespace NUMINAMATH_CALUDE_count_non_adjacent_arrangements_l1238_123898

/-- The number of arrangements of 5 letters where two specific letters are not adjacent to a third specific letter -/
def non_adjacent_arrangements : ℕ :=
  let total_letters := 5
  let non_adjacent_three := 12  -- arrangements where a, b, c are not adjacent
  let adjacent_pair_not_third := 24  -- arrangements where a and b are adjacent, but not to c
  non_adjacent_three + adjacent_pair_not_third

/-- Theorem stating that the number of arrangements of a, b, c, d, e where both a and b are not adjacent to c is 36 -/
theorem count_non_adjacent_arrangements :
  non_adjacent_arrangements = 36 := by
  sorry

end NUMINAMATH_CALUDE_count_non_adjacent_arrangements_l1238_123898


namespace NUMINAMATH_CALUDE_inverse_A_cubed_l1238_123862

def A_inv : Matrix (Fin 2) (Fin 2) ℤ := !![3, 8; -2, -5]

theorem inverse_A_cubed :
  let A := A_inv⁻¹
  (A^3)⁻¹ = !![5, 0; -66, -137] := by
  sorry

end NUMINAMATH_CALUDE_inverse_A_cubed_l1238_123862


namespace NUMINAMATH_CALUDE_interest_group_members_l1238_123856

/-- Represents a math interest group -/
structure InterestGroup where
  members : ℕ
  average_age : ℝ

/-- The change in average age when members leave or join -/
def age_change (g : InterestGroup) : Prop :=
  (g.members * g.average_age - 5 * 9 = (g.average_age + 1) * (g.members - 5)) ∧
  (g.members * g.average_age + 17 * 5 = (g.average_age + 1) * (g.members + 5))

theorem interest_group_members :
  ∃ (g : InterestGroup), age_change g → g.members = 20 := by
  sorry

end NUMINAMATH_CALUDE_interest_group_members_l1238_123856


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1238_123843

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin (2 * α) = -Real.sin α) : 
  Real.tan α = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1238_123843


namespace NUMINAMATH_CALUDE_floor_equation_solution_l1238_123815

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊3*x⌋ - 1/2⌋ = ⌊x + 3⌋ ↔ 5/3 ≤ x ∧ x < 7/3 :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l1238_123815


namespace NUMINAMATH_CALUDE_middle_of_three_consecutive_sum_30_l1238_123813

theorem middle_of_three_consecutive_sum_30 (a b c : ℕ) :
  (a + 1 = b) ∧ (b + 1 = c) ∧ (a + b + c = 30) → b = 10 := by
  sorry

end NUMINAMATH_CALUDE_middle_of_three_consecutive_sum_30_l1238_123813


namespace NUMINAMATH_CALUDE_car_driving_east_when_sun_setting_in_mirror_l1238_123850

-- Define the direction type
inductive Direction
| East
| West
| North
| South

-- Define the position of the sun
inductive SunPosition
| Setting
| Rising
| Overhead

-- Define the view of the sun
structure SunView where
  position : SunPosition
  throughMirror : Bool

-- Define the state of the car
structure CarState where
  direction : Direction
  sunView : SunView

-- Theorem statement
theorem car_driving_east_when_sun_setting_in_mirror 
  (car : CarState) : 
  car.sunView.position = SunPosition.Setting ∧ 
  car.sunView.throughMirror = true → 
  car.direction = Direction.East :=
sorry

end NUMINAMATH_CALUDE_car_driving_east_when_sun_setting_in_mirror_l1238_123850


namespace NUMINAMATH_CALUDE_f_properties_l1238_123841

noncomputable section

def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2*x

theorem f_properties :
  (f 2 = 4) ∧
  (f (1/2) = 1/4) ∧
  (f (f (-1)) = 1) ∧
  (∃ a : ℝ, f a = 3 ∧ (a = 1 ∨ a = Real.sqrt 3)) :=
by sorry

end

end NUMINAMATH_CALUDE_f_properties_l1238_123841


namespace NUMINAMATH_CALUDE_log_expression_equality_l1238_123880

theorem log_expression_equality : 2 * Real.log 2 - Real.log (1 / 25) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l1238_123880


namespace NUMINAMATH_CALUDE_exponent_equation_l1238_123872

theorem exponent_equation (x s : ℕ) (h : (2^x) * (25^s) = 5 * (10^16)) : x = 16 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equation_l1238_123872


namespace NUMINAMATH_CALUDE_integral_sin_cos_l1238_123842

theorem integral_sin_cos : 
  ∫ x in (0)..(2*Real.pi/3), (1 + Real.sin x) / (1 + Real.cos x + Real.sin x) = Real.pi/3 + Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sin_cos_l1238_123842


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l1238_123816

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, x^2 - a*x - a ≤ -3) ↔ (a ≤ -6 ∨ a ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l1238_123816


namespace NUMINAMATH_CALUDE_u_general_term_l1238_123832

def u : ℕ → ℚ
  | 0 => 1
  | 1 => 2
  | 2 => 0
  | (n + 3) => 2 * u (n + 2) + u (n + 1) - 2 * u n

theorem u_general_term : ∀ n : ℕ, u n = 2 - (2/3) * (-1)^n - (1/3) * 2^n := by
  sorry

end NUMINAMATH_CALUDE_u_general_term_l1238_123832


namespace NUMINAMATH_CALUDE_second_shop_expense_l1238_123837

theorem second_shop_expense (first_shop_books : ℕ) (second_shop_books : ℕ) 
  (first_shop_cost : ℕ) (average_price : ℕ) (total_books : ℕ)
  (h1 : first_shop_books = 65)
  (h2 : second_shop_books = 35)
  (h3 : first_shop_cost = 6500)
  (h4 : average_price = 85)
  (h5 : total_books = first_shop_books + second_shop_books) :
  (average_price * total_books) - first_shop_cost = 2000 := by
  sorry

end NUMINAMATH_CALUDE_second_shop_expense_l1238_123837


namespace NUMINAMATH_CALUDE_distance_between_intersecting_circles_l1238_123869

/-- The distance between the centers of two intersecting circles -/
def distance_between_centers (a : ℝ) : Set ℝ :=
  {a / 6 * (3 + Real.sqrt 3), a / 6 * (3 - Real.sqrt 3)}

/-- Represents two intersecting circles with a common chord -/
structure IntersectingCircles (a : ℝ) where
  /-- The common chord length -/
  chord_length : ℝ
  /-- The chord is a side of a regular inscribed triangle in one circle -/
  is_triangle_side : Bool
  /-- The chord is a side of an inscribed square in the other circle -/
  is_square_side : Bool
  /-- The chord length is positive -/
  chord_positive : chord_length > 0
  /-- The chord length is equal to a -/
  chord_eq_a : chord_length = a
  /-- One circle has the chord as a triangle side, the other as a square side -/
  different_inscriptions : is_triangle_side ≠ is_square_side

/-- Theorem stating the distance between centers of intersecting circles -/
theorem distance_between_intersecting_circles (a : ℝ) (circles : IntersectingCircles a) :
  ∃ d ∈ distance_between_centers a,
    d = (circles.chord_length / 6) * (3 + Real.sqrt 3) ∨
    d = (circles.chord_length / 6) * (3 - Real.sqrt 3) :=
  sorry

end NUMINAMATH_CALUDE_distance_between_intersecting_circles_l1238_123869


namespace NUMINAMATH_CALUDE_root_relationship_l1238_123831

def f (x : ℝ) : ℝ := x^3 - 7*x^2 + 12*x - 10
def g (x : ℝ) : ℝ := x^3 - 10*x^2 - 2*x + 20

theorem root_relationship :
  ∃ (x₀ : ℝ), f x₀ = 0 ∧ g (2*x₀) = 0 →
  f 5 = 0 ∧ g 10 = 0 := by
sorry

end NUMINAMATH_CALUDE_root_relationship_l1238_123831


namespace NUMINAMATH_CALUDE_percentage_problem_l1238_123805

theorem percentage_problem (x : ℝ) : 
  (16 / 100) * ((40 / 100) * x) = 6 → x = 93.75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1238_123805


namespace NUMINAMATH_CALUDE_marc_total_spending_l1238_123818

/-- The total amount spent by Marc on his purchases -/
def total_spent (model_car_price : ℕ) (paint_bottle_price : ℕ) (paintbrush_price : ℕ) 
  (model_car_quantity : ℕ) (paint_bottle_quantity : ℕ) (paintbrush_quantity : ℕ) : ℕ :=
  model_car_price * model_car_quantity + 
  paint_bottle_price * paint_bottle_quantity + 
  paintbrush_price * paintbrush_quantity

/-- Theorem stating that Marc's total spending is $160 -/
theorem marc_total_spending :
  total_spent 20 10 2 5 5 5 = 160 := by
  sorry

end NUMINAMATH_CALUDE_marc_total_spending_l1238_123818


namespace NUMINAMATH_CALUDE_problem_statement_l1238_123821

theorem problem_statement (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) :
  (a + b + 2*c ≤ 3) ∧ (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1238_123821


namespace NUMINAMATH_CALUDE_dist_P_F₂_eq_two_l1238_123823

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 2 = 1

-- Define the foci
variable (F₁ F₂ : ℝ × ℝ)

-- Define a point on the ellipse
variable (P : ℝ × ℝ)

-- Axiom: P is on the ellipse
axiom P_on_ellipse : is_on_ellipse P.1 P.2

-- Axiom: Distance from P to F₁ is 4
axiom dist_P_F₁ : Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) = 4

-- Theorem to prove
theorem dist_P_F₂_eq_two : Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_dist_P_F₂_eq_two_l1238_123823


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l1238_123889

theorem bernoulli_inequality (x : ℝ) (n : ℕ+) (h1 : x ≠ 0) (h2 : x > -1) :
  (1 + x)^(n : ℝ) ≥ n * x := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l1238_123889


namespace NUMINAMATH_CALUDE_golden_ratio_range_l1238_123817

theorem golden_ratio_range : 
  let φ := (Real.sqrt 5 - 1) / 2
  0.6 < φ ∧ φ < 0.7 := by sorry

end NUMINAMATH_CALUDE_golden_ratio_range_l1238_123817


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1238_123849

theorem inequality_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1238_123849


namespace NUMINAMATH_CALUDE_no_infinite_harmonic_mean_sequence_l1238_123858

theorem no_infinite_harmonic_mean_sequence :
  ¬ ∃ (a : ℕ → ℕ), 
    (∃ i j, a i ≠ a j) ∧ 
    (∀ n : ℕ, n ≥ 2 → a n = (2 * a (n-1) * a (n+1)) / (a (n-1) + a (n+1))) :=
by sorry

end NUMINAMATH_CALUDE_no_infinite_harmonic_mean_sequence_l1238_123858


namespace NUMINAMATH_CALUDE_andrew_cookie_cost_l1238_123882

/-- The cost of cookies purchased by Andrew in May --/
def total_cost : ℕ := 1395

/-- The number of days in May --/
def days_in_may : ℕ := 31

/-- The number of cookies Andrew purchased each day --/
def cookies_per_day : ℕ := 3

/-- The total number of cookies Andrew purchased in May --/
def total_cookies : ℕ := days_in_may * cookies_per_day

/-- The cost of each cookie --/
def cookie_cost : ℚ := total_cost / total_cookies

theorem andrew_cookie_cost : cookie_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_andrew_cookie_cost_l1238_123882


namespace NUMINAMATH_CALUDE_blocks_left_l1238_123830

/-- Given that Randy has 97 blocks initially and uses 25 blocks to build a tower,
    prove that the number of blocks left is 72. -/
theorem blocks_left (initial_blocks : ℕ) (used_blocks : ℕ) (h1 : initial_blocks = 97) (h2 : used_blocks = 25) :
  initial_blocks - used_blocks = 72 := by
  sorry

end NUMINAMATH_CALUDE_blocks_left_l1238_123830


namespace NUMINAMATH_CALUDE_contradiction_assumptions_l1238_123896

theorem contradiction_assumptions :
  (∀ p q : ℝ, (p^3 + q^3 = 2) → (¬(p + q ≤ 2) ↔ p + q > 2)) ∧
  (∀ a b : ℝ, |a| + |b| < 1 →
    ∃ x₁ : ℝ, x₁^2 + a*x₁ + b = 0 ∧ |x₁| ≥ 1 →
      ∃ x₂ : ℝ, x₂^2 + a*x₂ + b = 0 ∧ |x₂| ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_contradiction_assumptions_l1238_123896


namespace NUMINAMATH_CALUDE_kekai_sales_ratio_l1238_123890

def shirts_sold : ℕ := 5
def pants_sold : ℕ := 5
def shirt_price : ℕ := 1
def pants_price : ℕ := 3
def money_left : ℕ := 10

def total_earnings : ℕ := shirts_sold * shirt_price + pants_sold * pants_price

def money_given_to_parents : ℕ := total_earnings - money_left

theorem kekai_sales_ratio :
  (money_given_to_parents : ℚ) / total_earnings = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_kekai_sales_ratio_l1238_123890


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1238_123804

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_properties
  (a b c : ℝ)
  (h1 : f a b c (-1) = 0)
  (h2 : f a b c 0 = -3)
  (h3 : f a b c 2 = -3) :
  (∃ x y : ℝ, 
    (∀ z : ℝ, f a b c z = z^2 - 2*z - 3) ∧
    (x = 1 ∧ y = -4 ∧ ∀ z : ℝ, f a b c z ≥ f a b c x) ∧
    (∀ z : ℝ, z > 1 → ∀ w : ℝ, w > z → f a b c w > f a b c z) ∧
    (∀ z : ℝ, -1 < z ∧ z < 2 → -4 < f a b c z ∧ f a b c z < 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1238_123804


namespace NUMINAMATH_CALUDE_rectangle_fold_theorem_l1238_123809

theorem rectangle_fold_theorem : ∃ (a b : ℕ+), 
  a ≤ b ∧ 
  (a.val : ℝ) / (b.val : ℝ) * Real.sqrt ((a.val : ℝ)^2 + (b.val : ℝ)^2) = 65 ∧
  2 * (a.val + b.val) = 408 := by
sorry

end NUMINAMATH_CALUDE_rectangle_fold_theorem_l1238_123809


namespace NUMINAMATH_CALUDE_class_average_score_l1238_123847

theorem class_average_score (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ)
  (group1_average : ℚ) (group2_average : ℚ) :
  total_students = group1_students + group2_students →
  group1_students = 10 →
  group2_students = 10 →
  group1_average = 80 →
  group2_average = 60 →
  (group1_students * group1_average + group2_students * group2_average) / total_students = 70 := by
  sorry

end NUMINAMATH_CALUDE_class_average_score_l1238_123847


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l1238_123807

/-- Calculates the total wet surface area of a rectangular cistern -/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth) + 2 * (width * depth)

/-- Theorem stating the total wet surface area of a specific cistern -/
theorem cistern_wet_surface_area :
  total_wet_surface_area 8 6 1.85 = 99.8 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l1238_123807


namespace NUMINAMATH_CALUDE_min_vertical_distance_l1238_123877

/-- The absolute value function -/
def abs_func (x : ℝ) : ℝ := |x - 1|

/-- The quadratic function -/
def quad_func (x : ℝ) : ℝ := -x^2 - 4*x - 3

/-- The vertical distance between the two functions -/
def vertical_distance (x : ℝ) : ℝ := abs_func x - quad_func x

theorem min_vertical_distance :
  ∃ (min_dist : ℝ), min_dist = 7/4 ∧
  ∀ (x : ℝ), vertical_distance x ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_min_vertical_distance_l1238_123877


namespace NUMINAMATH_CALUDE_paint_replacement_fractions_l1238_123866

/-- Represents the fraction of paint replaced -/
def fraction_replaced (initial_intensity final_intensity new_intensity : ℚ) : ℚ :=
  (initial_intensity - final_intensity) / (initial_intensity - new_intensity)

theorem paint_replacement_fractions :
  let red_initial := (50 : ℚ) / 100
  let blue_initial := (60 : ℚ) / 100
  let red_new := (35 : ℚ) / 100
  let blue_new := (45 : ℚ) / 100
  let red_final := (45 : ℚ) / 100
  let blue_final := (55 : ℚ) / 100
  (fraction_replaced red_initial red_final red_new = 1/3) ∧
  (fraction_replaced blue_initial blue_final blue_new = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_paint_replacement_fractions_l1238_123866


namespace NUMINAMATH_CALUDE_mass_of_X_in_BaX_l1238_123827

/-- The molar mass of barium in g/mol -/
def molar_mass_Ba : ℝ := 137.33

/-- The mass percentage of barium in the compound -/
def mass_percentage_Ba : ℝ := 66.18

/-- The mass of the compound in grams -/
def total_mass : ℝ := 100

theorem mass_of_X_in_BaX : 
  let mass_Ba := total_mass * (mass_percentage_Ba / 100)
  let mass_X := total_mass - mass_Ba
  mass_X = 33.82 := by sorry

end NUMINAMATH_CALUDE_mass_of_X_in_BaX_l1238_123827


namespace NUMINAMATH_CALUDE_horner_V3_value_l1238_123870

-- Define the polynomial coefficients
def a : List ℤ := [12, 35, -8, 79, 6, 5, 3]

-- Define Horner's method for a single step
def horner_step (v : ℤ) (x : ℤ) (a : ℤ) : ℤ := v * x + a

-- Define the function to compute V_3 using Horner's method
def compute_V3 (coeffs : List ℤ) (x : ℤ) : ℤ :=
  let v0 := coeffs.reverse.head!
  let v1 := horner_step v0 x (coeffs.reverse.tail!.head!)
  let v2 := horner_step v1 x (coeffs.reverse.tail!.tail!.head!)
  horner_step v2 x (coeffs.reverse.tail!.tail!.tail!.head!)

-- State the theorem
theorem horner_V3_value :
  compute_V3 a (-4) = -57 := by sorry

end NUMINAMATH_CALUDE_horner_V3_value_l1238_123870


namespace NUMINAMATH_CALUDE_bear_path_discrepancy_l1238_123836

/-- Represents the circular path of a polar bear on an ice floe -/
structure BearPath where
  diameter_instrument : ℝ  -- Diameter measured by instruments
  diameter_footprint : ℝ   -- Diameter measured from footprints
  is_in_still_water : Prop -- The ice floe is in still water

/-- The difference in measured diameters is due to relative motion -/
theorem bear_path_discrepancy (path : BearPath) 
  (h_instrument : path.diameter_instrument = 8.5)
  (h_footprint : path.diameter_footprint = 9)
  (h_water : path.is_in_still_water) :
  ∃ (relative_motion : ℝ), 
    relative_motion > 0 ∧ 
    path.diameter_footprint - path.diameter_instrument = relative_motion :=
by sorry

end NUMINAMATH_CALUDE_bear_path_discrepancy_l1238_123836


namespace NUMINAMATH_CALUDE_line_parallel_implies_plane_perpendicular_l1238_123893

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (contained_in : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_implies_plane_perpendicular
  (l : Line) (m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : contained_in m β)
  (h3 : parallel l m) :
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_implies_plane_perpendicular_l1238_123893


namespace NUMINAMATH_CALUDE_line_slope_l1238_123811

/-- Given a line described by the equation 3y = 4x - 9 + 2z where z = 3,
    prove that the slope of this line is 4/3 -/
theorem line_slope (x y : ℝ) :
  3 * y = 4 * x - 9 + 2 * 3 →
  (∃ m b : ℝ, y = m * x + b ∧ m = 4 / 3) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_l1238_123811


namespace NUMINAMATH_CALUDE_max_value_a_l1238_123899

theorem max_value_a (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : b = 1 - a)
  (h4 : ∀ x : ℝ, x ∈ Set.Icc 0 1 → Real.exp x ≤ (1 + a * x) / (1 - b * x)) :
  a ≤ (1 : ℝ) / 2 ∧ ∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ Real.exp x = (1 + (1/2) * x) / (1 - (1/2) * x) :=
by sorry

end NUMINAMATH_CALUDE_max_value_a_l1238_123899


namespace NUMINAMATH_CALUDE_complex_radical_equality_l1238_123802

theorem complex_radical_equality (a b : ℝ) (ha : a ≥ 0) (hb : b > 0) :
  2.355 * |a^(1/4) - b^(1/6)| = 
  Real.sqrt ((a - 8 * (a^3 * b^2)^(1/6) + 4 * b^(2/3)) / 
             (a^(1/2) - 2 * b^(1/3) + 2 * (a^3 * b^2)^(1/12)) + 3 * b^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_complex_radical_equality_l1238_123802


namespace NUMINAMATH_CALUDE_part_one_part_two_l1238_123825

-- Define the sets A and B
def A : Set ℝ := {x | (x + 2) * (x - 3) < 0}
def B (a : ℝ) : Set ℝ := {x | x - a > 0}

-- Theorem for part (1)
theorem part_one : 
  (Set.univ \ A) ∪ (B 1) = {x : ℝ | x ≤ -2 ∨ x > 1} := by sorry

-- Theorem for part (2)
theorem part_two : 
  ∀ a : ℝ, A ⊆ B a ↔ a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1238_123825


namespace NUMINAMATH_CALUDE_election_majority_l1238_123860

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 500 →
  winning_percentage = 70/100 →
  (winning_percentage * total_votes : ℚ).num - ((1 - winning_percentage) * total_votes : ℚ).num = 200 := by
sorry

end NUMINAMATH_CALUDE_election_majority_l1238_123860


namespace NUMINAMATH_CALUDE_rabbit_carrots_l1238_123844

theorem rabbit_carrots (rabbit_per_burrow deer_per_burrow : ℕ)
  (rabbit_burrows deer_burrows : ℕ) :
  rabbit_per_burrow = 4 →
  deer_per_burrow = 6 →
  rabbit_per_burrow * rabbit_burrows = deer_per_burrow * deer_burrows →
  rabbit_burrows = deer_burrows + 3 →
  rabbit_per_burrow * rabbit_burrows = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_rabbit_carrots_l1238_123844


namespace NUMINAMATH_CALUDE_tangent_slope_angle_at_one_l1238_123867

noncomputable def f (x : ℝ) : ℝ := -(Real.sqrt 3 / 3) * x^3 + 2

theorem tangent_slope_angle_at_one :
  let f' : ℝ → ℝ := λ x ↦ -(Real.sqrt 3) * x^2
  let slope : ℝ := f' 1
  let angle_with_neg_x : ℝ := Real.arctan (Real.sqrt 3)
  let angle_with_pos_x : ℝ := π - angle_with_neg_x
  angle_with_pos_x = 2 * π / 3 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_at_one_l1238_123867


namespace NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_in_open_unit_interval_l1238_123854

theorem quadratic_always_positive_implies_a_in_open_unit_interval (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → 0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_in_open_unit_interval_l1238_123854


namespace NUMINAMATH_CALUDE_x_seven_y_eight_l1238_123801

theorem x_seven_y_eight (x y : ℚ) (hx : x = 3/4) (hy : y = 4/3) : x^7 * y^8 = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_x_seven_y_eight_l1238_123801


namespace NUMINAMATH_CALUDE_two_year_inflation_rate_real_yield_bank_deposit_l1238_123894

-- Define the annual inflation rate
def annual_inflation_rate : ℝ := 0.015

-- Define the nominal annual interest rate
def nominal_interest_rate : ℝ := 0.07

-- Theorem for two-year inflation rate
theorem two_year_inflation_rate : 
  ((1 + annual_inflation_rate)^2 - 1) * 100 = 3.0225 := by sorry

-- Theorem for real yield of bank deposit
theorem real_yield_bank_deposit : 
  ((1 + nominal_interest_rate)^2 / (1 + ((1 + annual_inflation_rate)^2 - 1)) - 1) * 100 = 11.13 := by sorry

end NUMINAMATH_CALUDE_two_year_inflation_rate_real_yield_bank_deposit_l1238_123894


namespace NUMINAMATH_CALUDE_continued_fraction_equality_l1238_123878

theorem continued_fraction_equality : 
  1 + 1 / (2 + 1 / (2 + 1 / 3)) = 24 / 17 := by
sorry

end NUMINAMATH_CALUDE_continued_fraction_equality_l1238_123878


namespace NUMINAMATH_CALUDE_sixty_degrees_in_vlecs_l1238_123840

/-- Represents the number of vlecs in a full circle on Venus -/
def full_circle_vlecs : ℕ := 800

/-- Represents the number of degrees in a full circle on Earth -/
def full_circle_degrees : ℕ := 360

/-- Represents the angle in degrees we want to convert to vlecs -/
def angle_degrees : ℕ := 60

/-- Converts an angle from degrees to vlecs -/
def degrees_to_vlecs (degrees : ℕ) : ℕ :=
  (degrees * full_circle_vlecs + full_circle_degrees / 2) / full_circle_degrees

theorem sixty_degrees_in_vlecs :
  degrees_to_vlecs angle_degrees = 133 := by
  sorry

end NUMINAMATH_CALUDE_sixty_degrees_in_vlecs_l1238_123840


namespace NUMINAMATH_CALUDE_sum_equals_rounded_sum_l1238_123848

def round_to_nearest_five (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

def sum_to_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def sum_rounded_to_n (n : ℕ) : ℕ :=
  (List.range n).map round_to_nearest_five |>.sum

theorem sum_equals_rounded_sum (n : ℕ) (h : n = 200) : 
  sum_to_n n = sum_rounded_to_n n := by
  sorry

#eval sum_to_n 200
#eval sum_rounded_to_n 200

end NUMINAMATH_CALUDE_sum_equals_rounded_sum_l1238_123848


namespace NUMINAMATH_CALUDE_tangent_line_to_ln_curve_l1238_123826

/-- The line y = kx is tangent to the curve y = ln x if and only if k = 1/e -/
theorem tangent_line_to_ln_curve (k : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ k * x = Real.log x ∧ k = 1 / x) ↔ k = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_ln_curve_l1238_123826


namespace NUMINAMATH_CALUDE_f_4_has_eight_zeros_l1238_123853

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Define the recursive function f_n
def f_n : ℕ → (ℝ → ℝ)
  | 0 => id
  | 1 => f
  | (n + 1) => f ∘ f_n n

-- State the theorem
theorem f_4_has_eight_zeros :
  ∃! (zeros : Finset ℝ), zeros.card = 8 ∧ ∀ x ∈ zeros, f_n 4 x = 0 :=
sorry

end NUMINAMATH_CALUDE_f_4_has_eight_zeros_l1238_123853


namespace NUMINAMATH_CALUDE_parabola_equation_l1238_123895

/-- A parabola with vertex at the origin and focus on the line x - 2y - 2 = 0 --/
structure Parabola where
  /-- The focus of the parabola lies on this line --/
  focus_line : {(x, y) : ℝ × ℝ | x - 2*y - 2 = 0}
  /-- The axis of symmetry is either the x-axis or y-axis --/
  symmetry_axis : (Unit → Prop) ⊕ (Unit → Prop)

/-- The standard equation of the parabola is either y² = 8x or x² = -4y --/
theorem parabola_equation (p : Parabola) :
  (∃ (x y : ℝ), y^2 = 8*x) ∨ (∃ (x y : ℝ), x^2 = -4*y) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1238_123895


namespace NUMINAMATH_CALUDE_extended_fishing_rod_length_l1238_123883

theorem extended_fishing_rod_length 
  (original_length : ℝ) 
  (increase_factor : ℝ) 
  (extended_length : ℝ) : 
  original_length = 48 → 
  increase_factor = 1.33 → 
  extended_length = original_length * increase_factor → 
  extended_length = 63.84 :=
by sorry

end NUMINAMATH_CALUDE_extended_fishing_rod_length_l1238_123883


namespace NUMINAMATH_CALUDE_teddy_pillow_count_l1238_123835

/-- The amount of fluffy foam material used for each pillow in pounds -/
def material_per_pillow : ℝ := 5 - 3

/-- The amount of fluffy foam material Teddy has in tons -/
def total_material_tons : ℝ := 3

/-- The number of pounds in a ton -/
def pounds_per_ton : ℝ := 2000

/-- The theorem stating how many pillows Teddy can make -/
theorem teddy_pillow_count : 
  (total_material_tons * pounds_per_ton) / material_per_pillow = 3000 := by
  sorry

end NUMINAMATH_CALUDE_teddy_pillow_count_l1238_123835


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1238_123876

theorem perpendicular_vectors_x_value (a b : ℝ × ℝ) (x : ℝ) :
  a = (1, 2) →
  b = (2, x) →
  a.1 * b.1 + a.2 * b.2 = 0 →
  x = -1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1238_123876


namespace NUMINAMATH_CALUDE_group_bill_calculation_l1238_123810

/-- Calculates the total cost for a group at a restaurant where kids eat free. -/
def restaurant_bill (total_people : ℕ) (num_kids : ℕ) (adult_meal_cost : ℕ) : ℕ :=
  (total_people - num_kids) * adult_meal_cost

/-- Proves that the total cost for a group of 11 people, including 2 kids,
    at a restaurant where adult meals cost $8 and kids eat free, is $72. -/
theorem group_bill_calculation :
  restaurant_bill 11 2 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_group_bill_calculation_l1238_123810


namespace NUMINAMATH_CALUDE_middle_term_expansion_l1238_123857

theorem middle_term_expansion (n a : ℕ+) (h1 : n > a) (h2 : 1 + a ^ (n : ℕ) = 65) :
  let middle_term := Nat.choose n.val (n.val / 2) * a ^ (n.val / 2)
  middle_term = 160 := by
sorry

end NUMINAMATH_CALUDE_middle_term_expansion_l1238_123857


namespace NUMINAMATH_CALUDE_y_divisibility_l1238_123800

def y : ℕ := 32 + 48 + 64 + 96 + 200 + 224 + 1600

theorem y_divisibility :
  (∃ k : ℕ, y = 4 * k) ∧
  (∃ k : ℕ, y = 8 * k) ∧
  (∃ k : ℕ, y = 16 * k) ∧
  ¬(∃ k : ℕ, y = 32 * k) :=
by sorry

end NUMINAMATH_CALUDE_y_divisibility_l1238_123800


namespace NUMINAMATH_CALUDE_carnival_sales_proof_l1238_123859

/-- Represents the daily sales of popcorn in dollars -/
def daily_popcorn_sales : ℝ := 50

/-- Represents the daily sales of cotton candy in dollars -/
def daily_cotton_candy_sales : ℝ := 3 * daily_popcorn_sales

/-- Duration of the carnival in days -/
def carnival_duration : ℕ := 5

/-- Total expenses for rent and ingredients in dollars -/
def total_expenses : ℝ := 105

/-- Net earnings after expenses in dollars -/
def net_earnings : ℝ := 895

theorem carnival_sales_proof :
  daily_popcorn_sales * carnival_duration +
  daily_cotton_candy_sales * carnival_duration -
  total_expenses = net_earnings :=
by sorry

end NUMINAMATH_CALUDE_carnival_sales_proof_l1238_123859


namespace NUMINAMATH_CALUDE_common_chord_triangle_area_l1238_123820

/-- Circle type representing x^2 + y^2 + ax + by + c = 0 --/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Line type representing ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to find the common chord of two circles --/
def commonChord (c1 c2 : Circle) : Line := sorry

/-- Function to find the intersection points of a line with the coordinate axes --/
def axisIntersections (l : Line) : ℝ × ℝ := sorry

/-- Function to calculate the area of a triangle given two side lengths --/
def triangleArea (base height : ℝ) : ℝ := sorry

theorem common_chord_triangle_area :
  let c1 : Circle := { a := 0, b := 0, c := -1 }
  let c2 : Circle := { a := -2, b := 2, c := 0 }
  let commonChordLine := commonChord c1 c2
  let (xIntercept, yIntercept) := axisIntersections commonChordLine
  triangleArea xIntercept yIntercept = 1/8 := by sorry

end NUMINAMATH_CALUDE_common_chord_triangle_area_l1238_123820


namespace NUMINAMATH_CALUDE_m_range_l1238_123891

-- Define the propositions p and q
def p (x : ℝ) : Prop := x < -2 ∨ x > 10

def q (x m : ℝ) : Prop := x^2 - 2*x - (m^2 - 1) ≥ 0

-- Define the condition that ¬q is sufficient but not necessary for ¬p
def sufficient_not_necessary (m : ℝ) : Prop :=
  (∀ x, ¬(q x m) → ¬(p x)) ∧ ∃ x, ¬(p x) ∧ q x m

-- State the theorem
theorem m_range (m : ℝ) :
  (m > 0) ∧ (sufficient_not_necessary m) ↔ 0 < m ∧ m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_m_range_l1238_123891


namespace NUMINAMATH_CALUDE_highlighter_count_l1238_123879

/-- The number of highlighters in Kaya's teacher's desk -/
theorem highlighter_count : 
  let pink : ℕ := 12
  let yellow : ℕ := 15
  let blue : ℕ := 8
  let green : ℕ := 6
  let orange : ℕ := 4
  pink + yellow + blue + green + orange = 45 := by
  sorry

end NUMINAMATH_CALUDE_highlighter_count_l1238_123879


namespace NUMINAMATH_CALUDE_quadratic_rewrite_sum_l1238_123887

theorem quadratic_rewrite_sum (x : ℝ) :
  ∃ (a b c : ℝ),
    (-4 * x^2 + 16 * x + 128 = a * (x + b)^2 + c) ∧
    (a + b + c = 138) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_sum_l1238_123887


namespace NUMINAMATH_CALUDE_merchant_pricing_strategy_l1238_123828

theorem merchant_pricing_strategy 
  (list_price : ℝ) 
  (purchase_discount : ℝ) 
  (sale_discount : ℝ) 
  (profit_margin : ℝ) 
  (h1 : purchase_discount = 0.3) 
  (h2 : sale_discount = 0.2) 
  (h3 : profit_margin = 0.3) 
  (h4 : list_price > 0) :
  let purchase_price := list_price * (1 - purchase_discount)
  let marked_price := list_price * 1.25
  let selling_price := marked_price * (1 - sale_discount)
  selling_price = purchase_price * (1 + profit_margin) := by
sorry

end NUMINAMATH_CALUDE_merchant_pricing_strategy_l1238_123828


namespace NUMINAMATH_CALUDE_matchstick_20th_term_l1238_123897

/-- Arithmetic sequence with first term 4 and common difference 3 -/
def matchstick_sequence (n : ℕ) : ℕ := 4 + 3 * (n - 1)

/-- The 20th term of the matchstick sequence is 61 -/
theorem matchstick_20th_term : matchstick_sequence 20 = 61 := by
  sorry

end NUMINAMATH_CALUDE_matchstick_20th_term_l1238_123897


namespace NUMINAMATH_CALUDE_division_fraction_proof_l1238_123838

theorem division_fraction_proof : (5 : ℚ) / ((8 : ℚ) / 13) = 65 / 8 := by
  sorry

end NUMINAMATH_CALUDE_division_fraction_proof_l1238_123838


namespace NUMINAMATH_CALUDE_fedya_deposit_l1238_123812

theorem fedya_deposit (n : ℕ) (hn : 0 < n ∧ n < 30) : 
  (∃ (x : ℕ), x * (100 - n) = 847 * 100) → 
  (∃ (x : ℕ), x * (100 - n) = 847 * 100 ∧ x = 1100) :=
by sorry

end NUMINAMATH_CALUDE_fedya_deposit_l1238_123812


namespace NUMINAMATH_CALUDE_football_team_right_handed_players_l1238_123819

theorem football_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 70) 
  (h2 : throwers = 34) 
  (h3 : (total_players - throwers) % 3 = 0) -- Ensures non-throwers can be divided into thirds
  (h4 : throwers ≤ total_players) : -- Ensures there are not more throwers than total players
  throwers + ((total_players - throwers) - (total_players - throwers) / 3) = 58 := by
  sorry

end NUMINAMATH_CALUDE_football_team_right_handed_players_l1238_123819


namespace NUMINAMATH_CALUDE_fraction_sum_of_squares_is_integer_l1238_123814

theorem fraction_sum_of_squares_is_integer (a b : ℚ) 
  (h1 : ∃ k : ℤ, a + b = k) 
  (h2 : ∃ m : ℤ, a * b / (a + b) = m) : 
  ∃ n : ℤ, (a^2 + b^2) / (a + b) = n := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_of_squares_is_integer_l1238_123814


namespace NUMINAMATH_CALUDE_binomial_1500_1_l1238_123803

theorem binomial_1500_1 : Nat.choose 1500 1 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_binomial_1500_1_l1238_123803


namespace NUMINAMATH_CALUDE_function_monotonicity_l1238_123886

open Set
open Function

theorem function_monotonicity (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x > 0, f x > -x * deriv f x) :
  Monotone (fun x => x * f x) := by
sorry

end NUMINAMATH_CALUDE_function_monotonicity_l1238_123886


namespace NUMINAMATH_CALUDE_dick_jane_age_problem_l1238_123885

theorem dick_jane_age_problem :
  ∃ (d n : ℕ), 
    d > 27 ∧
    10 ≤ 27 + n ∧ 27 + n ≤ 99 ∧
    10 ≤ d + n ∧ d + n ≤ 99 ∧
    ∃ (a b : ℕ), 
      27 + n = 10 * a + b ∧
      d + n = 10 * b + a ∧
      Nat.Prime (a + b) ∧
      1 ≤ a ∧ a < b ∧ b ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_dick_jane_age_problem_l1238_123885


namespace NUMINAMATH_CALUDE_twenty_cent_items_count_l1238_123863

/-- Represents the number of items at each price point -/
structure ItemCounts where
  cents20 : ℕ
  dollars150 : ℕ
  dollars250 : ℕ

/-- Checks if the given item counts satisfy the problem conditions -/
def satisfiesConditions (counts : ItemCounts) : Prop :=
  counts.cents20 + counts.dollars150 + counts.dollars250 = 50 ∧
  20 * counts.cents20 + 150 * counts.dollars150 + 250 * counts.dollars250 = 5000

/-- Theorem stating that the number of 20-cent items is 31 -/
theorem twenty_cent_items_count :
  ∃ (counts : ItemCounts), satisfiesConditions counts ∧ counts.cents20 = 31 := by
  sorry

end NUMINAMATH_CALUDE_twenty_cent_items_count_l1238_123863


namespace NUMINAMATH_CALUDE_exists_m_n_for_k_l1238_123839

theorem exists_m_n_for_k (k : ℕ) : 
  (∃ m n : ℕ, m * (m + k) = n * (n + 1)) ↔ k ≠ 2 ∧ k ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_exists_m_n_for_k_l1238_123839


namespace NUMINAMATH_CALUDE_count_12_digit_numbers_with_consecutive_ones_l1238_123888

/-- The sequence of counts of n-digit numbers with digits 1, 2, or 3 without two consecutive 1's -/
def F : ℕ → ℕ
| 0 => 1
| 1 => 3
| (n+2) => 2 * F (n+1) + F n

/-- The count of n-digit numbers with digits 1, 2, or 3 -/
def total_count (n : ℕ) : ℕ := 3^n

/-- The count of n-digit numbers with digits 1, 2, or 3 and at least two consecutive 1's -/
def count_with_consecutive_ones (n : ℕ) : ℕ := total_count n - F n

theorem count_12_digit_numbers_with_consecutive_ones : 
  count_with_consecutive_ones 12 = 530456 := by
  sorry

end NUMINAMATH_CALUDE_count_12_digit_numbers_with_consecutive_ones_l1238_123888


namespace NUMINAMATH_CALUDE_extreme_value_derivative_condition_l1238_123868

open Real

theorem extreme_value_derivative_condition (f : ℝ → ℝ) (x₀ : ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → f x ≤ f x₀ + ε ∨ f x ≥ f x₀ - ε) →
  (deriv f) x₀ = 0 ∧
  ∃ g : ℝ → ℝ, (deriv g) 0 = 0 ∧ ¬(∀ ε > 0, ∃ δ > 0, ∀ x, |x - 0| < δ → g x ≤ g 0 + ε ∨ g x ≥ g 0 - ε) :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_derivative_condition_l1238_123868


namespace NUMINAMATH_CALUDE_akeno_extra_expenditure_l1238_123892

def akeno_expenditure : ℕ := 2985

def lev_expenditure (akeno : ℕ) : ℕ := akeno / 3

def ambrocio_expenditure (lev : ℕ) : ℕ := lev - 177

theorem akeno_extra_expenditure (akeno lev ambrocio : ℕ) 
  (h1 : akeno = akeno_expenditure)
  (h2 : lev = lev_expenditure akeno)
  (h3 : ambrocio = ambrocio_expenditure lev) :
  akeno - (lev + ambrocio) = 1172 := by
  sorry

end NUMINAMATH_CALUDE_akeno_extra_expenditure_l1238_123892


namespace NUMINAMATH_CALUDE_expression_bounds_l1238_123846

theorem expression_bounds : 1 < (3 * Real.sqrt 2 - Real.sqrt 12) * Real.sqrt 3 ∧ 
                            (3 * Real.sqrt 2 - Real.sqrt 12) * Real.sqrt 3 < 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l1238_123846


namespace NUMINAMATH_CALUDE_range_of_m_l1238_123829

theorem range_of_m (m : ℝ) : 
  (∃! (n : ℕ), n = 4 ∧ (∀ x : ℤ, (m < x ∧ x < 4) ↔ (0 ≤ x ∧ x < 4))) → 
  (-1 ≤ m ∧ m < 0) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1238_123829


namespace NUMINAMATH_CALUDE_triangle_properties_l1238_123806

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The vectors (cos A, cos B) and (a, 2c - b) are parallel -/
def vectors_parallel (t : Triangle) : Prop :=
  (2 * t.c - t.b) * Real.cos t.A = t.a * Real.cos t.B

/-- The theorem to be proved -/
theorem triangle_properties (t : Triangle) (h : vectors_parallel t) :
  t.A = π / 3 ∧ (t.a = 4 → ∃ (max_area : ℝ), max_area = 4 * Real.sqrt 3 ∧
    ∀ (area : ℝ), area = 1 / 2 * t.b * t.c * Real.sin t.A → area ≤ max_area) :=
sorry

end

end NUMINAMATH_CALUDE_triangle_properties_l1238_123806


namespace NUMINAMATH_CALUDE_f_extremum_l1238_123808

/-- The function f(x, y) -/
def f (x y : ℝ) : ℝ := x^3 + 3*x*y^2 - 18*x^2 - 18*x*y - 18*y^2 + 57*x + 138*y + 290

/-- Theorem stating the extremum of f(x, y) -/
theorem f_extremum :
  (∃ (x y : ℝ), f x y = 10 ∧ ∀ (a b : ℝ), f a b ≥ 10) ∧
  (∃ (x y : ℝ), f x y = 570 ∧ ∀ (a b : ℝ), f a b ≤ 570) :=
sorry

end NUMINAMATH_CALUDE_f_extremum_l1238_123808


namespace NUMINAMATH_CALUDE_fraction_difference_equals_sqrt_five_l1238_123855

theorem fraction_difference_equals_sqrt_five (a b : ℝ) (h1 : a ≠ b) (h2 : 1/a + 1/b = Real.sqrt 5) :
  a / (b * (a - b)) - b / (a * (a - b)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_equals_sqrt_five_l1238_123855


namespace NUMINAMATH_CALUDE_curve_transformation_l1238_123833

theorem curve_transformation (x : ℝ) : 
  Real.sin (4 * x + π / 3) = Real.cos (2 * (x - π / 24)) := by
  sorry

end NUMINAMATH_CALUDE_curve_transformation_l1238_123833


namespace NUMINAMATH_CALUDE_difference_of_squares_l1238_123861

theorem difference_of_squares (x : ℝ) : x^2 - 121 = (x + 11) * (x - 11) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1238_123861


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l1238_123845

/-- Conversion from polar coordinates to rectangular coordinates --/
theorem polar_to_rectangular (r θ : ℝ) :
  r = 6 ∧ θ = π / 3 →
  ∃ x y : ℝ, x = 3 ∧ y = 3 * Real.sqrt 3 ∧
  x = r * Real.cos θ ∧ y = r * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_l1238_123845


namespace NUMINAMATH_CALUDE_equation_solution_l1238_123874

theorem equation_solution (x y : ℝ) (hx : x ≠ 0) (hxy : x + y ≠ 0) :
  (x + y) / x = (y + 1) / (x + y) →
  (x = (-y + Real.sqrt (4 - 3 * y^2)) / 2 ∨ x = (-y - Real.sqrt (4 - 3 * y^2)) / 2) ∧
  -2 / Real.sqrt 3 ≤ y ∧ y ≤ 2 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1238_123874


namespace NUMINAMATH_CALUDE_sphere_radius_regular_tetrahedron_l1238_123873

/-- The radius of a sphere touching all edges of a regular tetrahedron with edge length √2 --/
theorem sphere_radius_regular_tetrahedron : 
  ∀ (tetrahedron_edge : ℝ) (sphere_radius : ℝ),
  tetrahedron_edge = Real.sqrt 2 →
  sphere_radius = 
    (1 / 2) * ((tetrahedron_edge * Real.sqrt 6) / 3) →
  sphere_radius = 1 := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_regular_tetrahedron_l1238_123873


namespace NUMINAMATH_CALUDE_solve_for_m_l1238_123875

theorem solve_for_m (x y m : ℝ) (h1 : x = 2) (h2 : y = m) (h3 : 3 * x + 2 * y = 10) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l1238_123875


namespace NUMINAMATH_CALUDE_fraction_equality_l1238_123834

theorem fraction_equality (a b c d : ℝ) (h1 : b ≠ c) 
  (h2 : (a * c - b^2) / (a - 2*b + c) = (b * d - c^2) / (b - 2*c + d)) : 
  (a * c - b^2) / (a - 2*b + c) = (a * d - b * c) / (a - b - c + d) ∧ 
  (b * d - c^2) / (b - 2*c + d) = (a * d - b * c) / (a - b - c + d) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1238_123834
