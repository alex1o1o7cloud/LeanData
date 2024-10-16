import Mathlib

namespace NUMINAMATH_CALUDE_inequality_equivalence_l1935_193599

theorem inequality_equivalence (x y z : ℝ) :
  x + 3 * y + 2 * z = 6 →
  (x^2 + 9 * y^2 - 2 * x - 6 * y + 4 * z ≤ 8 ↔
   z = 3 - 1/2 * x - 3/2 * y ∧ (x - 2)^2 + (3 * y - 2)^2 ≤ 4 ∧ 0 ≤ x ∧ x ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1935_193599


namespace NUMINAMATH_CALUDE_square_19_on_top_l1935_193505

/-- Represents a position on the 9x9 grid -/
structure Position :=
  (row : Fin 9)
  (col : Fin 9)

/-- Represents the state of the grid after folding -/
structure FoldedGrid :=
  (top_square : Nat)

/-- Defines the initial 9x9 grid -/
def initial_grid : List (List Nat) :=
  List.range 9 |> List.map (fun i => List.range 9 |> List.map (fun j => i * 9 + j + 1))

/-- Performs the sequence of folds on the grid -/
def fold_grid (grid : List (List Nat)) : FoldedGrid :=
  sorry

/-- The main theorem stating that square 19 is on top after folding -/
theorem square_19_on_top :
  (fold_grid initial_grid).top_square = 19 := by sorry

end NUMINAMATH_CALUDE_square_19_on_top_l1935_193505


namespace NUMINAMATH_CALUDE_MNP_collinear_tangent_equals_PA_l1935_193569

-- Define the circles and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

-- Define the problem setup
def S : Circle := sorry
def S₁ : Circle := sorry
def A : Point := sorry
def B : Point := sorry
def M : Point := sorry
def N : Point := sorry
def P : Point := sorry

-- Define the conditions
axiom chord_divides_circle : sorry
axiom S₁_touches_AB_at_M : sorry
axiom S₁_touches_arc_at_N : sorry
axiom P_is_midpoint_of_other_arc : sorry

-- Define helper functions
def collinear (p q r : Point) : Prop := sorry
def tangent_length (p : Point) (c : Circle) : ℝ := sorry
def distance (p q : Point) : ℝ := sorry

-- State the theorems to be proved
theorem MNP_collinear : collinear M N P := sorry

theorem tangent_equals_PA : tangent_length P S₁ = distance P A := sorry

end NUMINAMATH_CALUDE_MNP_collinear_tangent_equals_PA_l1935_193569


namespace NUMINAMATH_CALUDE_freelancer_earnings_l1935_193570

theorem freelancer_earnings (x : ℝ) : 
  x + (50 + 2*x) + 4*(x + (50 + 2*x)) = 5500 → x = 5300/15 := by
  sorry

end NUMINAMATH_CALUDE_freelancer_earnings_l1935_193570


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_2310_l1935_193571

theorem distinct_prime_factors_of_2310 : Nat.card (Nat.factors 2310).toFinset = 5 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_2310_l1935_193571


namespace NUMINAMATH_CALUDE_inequalities_hold_l1935_193532

theorem inequalities_hold (a b : ℝ) (h : a * b > 0) :
  (2 * (a^2 + b^2) ≥ (a + b)^2) ∧
  (b / a + a / b ≥ 2) ∧
  ((a + 1 / a) * (b + 1 / b) ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l1935_193532


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l1935_193568

-- Define the prices and quantities
def pasta_price : ℝ := 1.70
def pasta_quantity : ℝ := 3
def beef_price : ℝ := 8.20
def beef_quantity : ℝ := 0.5
def sauce_price : ℝ := 2.30
def sauce_quantity : ℝ := 3
def quesadillas_price : ℝ := 11.50
def discount_rate : ℝ := 0.10
def vat_rate : ℝ := 0.05

-- Define the total cost function
def total_cost : ℝ :=
  let pasta_cost := pasta_price * pasta_quantity
  let beef_cost := beef_price * beef_quantity
  let sauce_cost := sauce_price * sauce_quantity
  let discounted_sauce_cost := sauce_cost * (1 - discount_rate)
  let subtotal := pasta_cost + beef_cost + discounted_sauce_cost + quesadillas_price
  let vat := subtotal * vat_rate
  subtotal + vat

-- Theorem statement
theorem total_cost_is_correct : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ abs (total_cost - 28.26) < ε :=
sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l1935_193568


namespace NUMINAMATH_CALUDE_exist_nonzero_superintegers_with_zero_product_l1935_193547

-- Define a super-integer as a function from ℕ to ℕ
def SuperInteger := ℕ → ℕ

-- Define a zero super-integer
def isZeroSuperInteger (x : SuperInteger) : Prop :=
  ∀ n, x n = 0

-- Define non-zero super-integer
def isNonZeroSuperInteger (x : SuperInteger) : Prop :=
  ∃ n, x n ≠ 0

-- Define the product of two super-integers
def superIntegerProduct (x y : SuperInteger) : SuperInteger :=
  fun n => (x n * y n) % (10^n)

-- Theorem statement
theorem exist_nonzero_superintegers_with_zero_product :
  ∃ (x y : SuperInteger),
    isNonZeroSuperInteger x ∧
    isNonZeroSuperInteger y ∧
    isZeroSuperInteger (superIntegerProduct x y) := by
  sorry


end NUMINAMATH_CALUDE_exist_nonzero_superintegers_with_zero_product_l1935_193547


namespace NUMINAMATH_CALUDE_percentage_of_blue_shirts_l1935_193572

theorem percentage_of_blue_shirts (total_students : ℕ) 
  (red_percent green_percent : ℚ) (other_count : ℕ) : 
  total_students = 700 →
  red_percent = 23/100 →
  green_percent = 15/100 →
  other_count = 119 →
  (1 - (red_percent + green_percent + (other_count : ℚ) / total_students)) * 100 = 45 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_blue_shirts_l1935_193572


namespace NUMINAMATH_CALUDE_rectangle_not_always_similar_l1935_193543

-- Define the shapes
structure Square :=
  (side : ℝ)

structure IsoscelesRightTriangle :=
  (leg : ℝ)

structure Rectangle :=
  (length width : ℝ)

structure EquilateralTriangle :=
  (side : ℝ)

-- Define similarity for each shape
def similar_squares (s1 s2 : Square) : Prop :=
  true

def similar_isosceles_right_triangles (t1 t2 : IsoscelesRightTriangle) : Prop :=
  true

def similar_rectangles (r1 r2 : Rectangle) : Prop :=
  r1.length / r1.width = r2.length / r2.width

def similar_equilateral_triangles (e1 e2 : EquilateralTriangle) : Prop :=
  true

-- Theorem statement
theorem rectangle_not_always_similar :
  ∃ r1 r2 : Rectangle, ¬(similar_rectangles r1 r2) ∧
  (∀ s1 s2 : Square, similar_squares s1 s2) ∧
  (∀ t1 t2 : IsoscelesRightTriangle, similar_isosceles_right_triangles t1 t2) ∧
  (∀ e1 e2 : EquilateralTriangle, similar_equilateral_triangles e1 e2) :=
sorry

end NUMINAMATH_CALUDE_rectangle_not_always_similar_l1935_193543


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_equation_l1935_193548

theorem unique_solution_quadratic_equation (m n : ℤ) :
  m^2 - 2*m*n + 2*n^2 - 4*n + 4 = 0 → m = 2 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_equation_l1935_193548


namespace NUMINAMATH_CALUDE_point_satisfies_conditions_l1935_193555

def point (m : ℝ) : ℝ × ℝ := (2 - m, 2 * m - 1)

def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  |p.1|

theorem point_satisfies_conditions (m : ℝ) :
  in_fourth_quadrant (point m) ∧
  distance_to_y_axis (point m) = 3 →
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_point_satisfies_conditions_l1935_193555


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1935_193575

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, ∃ n : ℕ+, (n : ℝ) ≥ x^2)) ↔ (∃ x : ℝ, ∀ n : ℕ+, (n : ℝ) < x^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1935_193575


namespace NUMINAMATH_CALUDE_find_divisor_find_divisor_proof_l1935_193593

theorem find_divisor (original : ℕ) (divisible : ℕ) (divisor : ℕ) : Prop :=
  (original = 859622) →
  (divisible = 859560) →
  (divisor = 62) →
  (original - divisible = divisor) ∧
  (divisible % divisor = 0)

/-- The proof of the theorem --/
theorem find_divisor_proof : ∃ (d : ℕ), find_divisor 859622 859560 d :=
  sorry

end NUMINAMATH_CALUDE_find_divisor_find_divisor_proof_l1935_193593


namespace NUMINAMATH_CALUDE_moving_circle_theorem_l1935_193564

-- Define the moving circle
structure MovingCircle where
  center : ℝ × ℝ
  passes_through_A : center.1 ^ 2 + center.2 ^ 2 = (center.1 - 2) ^ 2 + center.2 ^ 2
  cuts_y_axis : ∃ (y : ℝ), center.1 ^ 2 + (y - center.2) ^ 2 = center.1 ^ 2 + center.2 ^ 2 ∧ y ^ 2 = 4

-- Define the trajectory
def trajectory (x y : ℝ) : Prop := y ^ 2 = 4 * x

-- Define the fixed point N
structure FixedPointN where
  x₀ : ℝ

-- Define the chord BD
structure ChordBD (n : FixedPointN) where
  m : ℝ
  passes_through_N : ∀ (y : ℝ), trajectory (n.x₀ + m * y) y

-- Define the angle BAD
def angle_BAD_obtuse (n : FixedPointN) (bd : ChordBD n) : Prop :=
  ∀ (y₁ y₂ : ℝ), 
    trajectory (n.x₀ + bd.m * y₁) y₁ → 
    trajectory (n.x₀ + bd.m * y₂) y₂ → 
    (n.x₀ + bd.m * y₁ - 2) * (n.x₀ + bd.m * y₂ - 2) + y₁ * y₂ < 0

-- The main theorem
theorem moving_circle_theorem :
  (∀ (mc : MovingCircle), trajectory mc.center.1 mc.center.2) ∧
  (∀ (n : FixedPointN), 
    (∀ (bd : ChordBD n), angle_BAD_obtuse n bd) → 
    (4 - 2 * Real.sqrt 3 < n.x₀ ∧ n.x₀ < 4 + 2 * Real.sqrt 3 ∧ n.x₀ ≠ 2)) :=
sorry

end NUMINAMATH_CALUDE_moving_circle_theorem_l1935_193564


namespace NUMINAMATH_CALUDE_odd_function_property_l1935_193529

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

-- Theorem statement
theorem odd_function_property (f : ℝ → ℝ) (h_odd : OddFunction f) 
  (h_g1 : g f 1 = 1) : g f (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l1935_193529


namespace NUMINAMATH_CALUDE_equation_two_solutions_l1935_193523

def equation (a x : ℝ) : Prop :=
  (Real.cos (2 * x) + 14 * Real.cos x - 14 * a)^7 - (6 * a * Real.cos x - 4 * a^2 - 1)^7 = 
  (6 * a - 14) * Real.cos x + 2 * Real.sin x^2 - 4 * a^2 + 14 * a - 2

theorem equation_two_solutions :
  ∃ (S₁ S₂ : Set ℝ),
    (S₁ = {a : ℝ | 3.25 ≤ a ∧ a < 4}) ∧
    (S₂ = {a : ℝ | -0.5 ≤ a ∧ a < 1}) ∧
    (∀ a ∈ S₁ ∪ S₂, ∃ (x₁ x₂ : ℝ),
      x₁ ≠ x₂ ∧
      -2 * Real.pi / 3 ≤ x₁ ∧ x₁ ≤ Real.pi ∧
      -2 * Real.pi / 3 ≤ x₂ ∧ x₂ ≤ Real.pi ∧
      equation a x₁ ∧
      equation a x₂ ∧
      (∀ x, -2 * Real.pi / 3 ≤ x ∧ x ≤ Real.pi ∧ equation a x → x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_two_solutions_l1935_193523


namespace NUMINAMATH_CALUDE_normal_dist_probability_l1935_193535

-- Define the normal distribution
def normal_dist (μ σ : ℝ) (X : ℝ → ℝ) : Prop :=
  ∃ (f : ℝ → ℝ), ∀ x, X x = f ((x - μ) / σ)

-- Define the probability function
noncomputable def P (a b : ℝ) (X : ℝ → ℝ) : ℝ := sorry

-- State the theorem
theorem normal_dist_probability 
  (X : ℝ → ℝ) (μ σ : ℝ) 
  (h1 : normal_dist μ σ X)
  (h2 : P (μ - 2*σ) (μ + 2*σ) X = 0.9544)
  (h3 : P (μ - σ) (μ + σ) X = 0.682)
  (h4 : μ = 4)
  (h5 : σ = 1) :
  P 5 6 X = 0.1359 := by sorry

end NUMINAMATH_CALUDE_normal_dist_probability_l1935_193535


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l1935_193520

theorem triangle_angle_sum (A B C : ℝ) (h1 : A = 75) (h2 : B = 40) : C = 65 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l1935_193520


namespace NUMINAMATH_CALUDE_sample_size_comparison_l1935_193594

/-- Given two samples with different means, prove that the number of elements in the first sample
    is less than or equal to the number of elements in the second sample, based on the combined mean. -/
theorem sample_size_comparison (m n : ℕ) (x_bar y_bar z_bar : ℝ) (a : ℝ) :
  x_bar ≠ y_bar →
  z_bar = a * x_bar + (1 - a) * y_bar →
  0 < a →
  a ≤ 1/2 →
  z_bar = (m * x_bar + n * y_bar) / (m + n : ℝ) →
  m ≤ n := by
  sorry


end NUMINAMATH_CALUDE_sample_size_comparison_l1935_193594


namespace NUMINAMATH_CALUDE_bus_driver_max_regular_hours_l1935_193521

/-- Represents the problem of finding the maximum regular hours for a bus driver -/
theorem bus_driver_max_regular_hours 
  (regular_rate : ℝ) 
  (overtime_rate_factor : ℝ) 
  (total_compensation : ℝ) 
  (total_hours : ℝ) 
  (h1 : regular_rate = 16)
  (h2 : overtime_rate_factor = 1.75)
  (h3 : total_compensation = 1116)
  (h4 : total_hours = 57) :
  ∃ (max_regular_hours : ℝ),
    max_regular_hours * regular_rate + 
    (total_hours - max_regular_hours) * (regular_rate * overtime_rate_factor) = 
    total_compensation ∧ 
    max_regular_hours = 40 :=
by sorry

end NUMINAMATH_CALUDE_bus_driver_max_regular_hours_l1935_193521


namespace NUMINAMATH_CALUDE_sheets_per_class_calculation_l1935_193516

/-- The number of sheets of paper used by the school per week -/
def sheets_per_week : ℕ := 9000

/-- The number of school days per week -/
def school_days_per_week : ℕ := 5

/-- The number of classes in the school -/
def num_classes : ℕ := 9

/-- The number of sheets of paper each class uses per day -/
def sheets_per_class_per_day : ℕ := sheets_per_week / school_days_per_week / num_classes

theorem sheets_per_class_calculation :
  sheets_per_class_per_day = 200 := by
  sorry

end NUMINAMATH_CALUDE_sheets_per_class_calculation_l1935_193516


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l1935_193556

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- Theorem statement
theorem opposite_of_negative_five :
  opposite (-5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l1935_193556


namespace NUMINAMATH_CALUDE_circumcenter_rational_coords_l1935_193504

/-- Given a triangle with rational coordinates, the center of its circumscribed circle has rational coordinates. -/
theorem circumcenter_rational_coords (a₁ a₂ a₃ b₁ b₂ b₃ : ℚ) :
  ∃ (x y : ℚ), 
    (x - a₁)^2 + (y - b₁)^2 = (x - a₂)^2 + (y - b₂)^2 ∧
    (x - a₁)^2 + (y - b₁)^2 = (x - a₃)^2 + (y - b₃)^2 :=
by sorry

end NUMINAMATH_CALUDE_circumcenter_rational_coords_l1935_193504


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1935_193503

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let A := (a^3*(b+c) + b^3*(c+a) + c^3*(a+b)) / ((a+b+c)^4 - 79*(a*b*c)^(4/3))
  A ≤ 3 ∧ (A = 3 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1935_193503


namespace NUMINAMATH_CALUDE_hundredth_ring_squares_l1935_193501

def ring_squares (n : ℕ) : ℕ :=
  4 + 8 * (n - 1)

theorem hundredth_ring_squares :
  ring_squares 100 = 796 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_ring_squares_l1935_193501


namespace NUMINAMATH_CALUDE_age_double_time_l1935_193590

/-- Proves that the number of years until a man's age is twice his son's age is 2,
    given that the man is currently 22 years older than his son and the son is currently 20 years old. -/
theorem age_double_time : ∃ (x : ℕ), 
  (20 + x) * 2 = (20 + 22 + x) ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_age_double_time_l1935_193590


namespace NUMINAMATH_CALUDE_joan_bought_72_eggs_l1935_193502

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The number of dozens Joan bought -/
def dozens_bought : ℕ := 6

/-- Theorem: Joan bought 72 eggs -/
theorem joan_bought_72_eggs : dozens_bought * eggs_per_dozen = 72 := by
  sorry

end NUMINAMATH_CALUDE_joan_bought_72_eggs_l1935_193502


namespace NUMINAMATH_CALUDE_pet_shop_total_l1935_193565

theorem pet_shop_total (dogs cats bunnies : ℕ) : 
  dogs = 154 → 
  dogs * 8 = bunnies * 7 → 
  dogs + bunnies = 330 :=
by
  sorry

end NUMINAMATH_CALUDE_pet_shop_total_l1935_193565


namespace NUMINAMATH_CALUDE_min_value_of_sum_min_value_is_nine_min_value_achieved_l1935_193507

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 1/y = 1 → 4*a + b ≤ 4*x + y :=
by sorry

theorem min_value_is_nine (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  4*a + b ≥ 9 :=
by sorry

theorem min_value_achieved (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 1/y = 1 ∧ 4*x + y = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_min_value_is_nine_min_value_achieved_l1935_193507


namespace NUMINAMATH_CALUDE_triangle_area_implies_p_value_l1935_193511

/-- Given a triangle ABC with vertices A(3, 15), B(15, 0), and C(0, p),
    prove that if the area of the triangle is 35, then p = 77.5/6 -/
theorem triangle_area_implies_p_value (p : ℝ) : 
  let A : ℝ × ℝ := (3, 15)
  let B : ℝ × ℝ := (15, 0)
  let C : ℝ × ℝ := (0, p)
  let triangle_area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  triangle_area = 35 → p = 77.5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_implies_p_value_l1935_193511


namespace NUMINAMATH_CALUDE_triangle_side_length_l1935_193557

/-- Given a triangle ABC where:
    - a, b, c are sides opposite to angles A, B, C respectively
    - A = 60°
    - b = 4
    - Area of triangle ABC = 4√3
    Prove that a = 4 -/
theorem triangle_side_length (a b c : ℝ) (A : Real) (S : ℝ) : 
  A = π / 3 → 
  b = 4 → 
  S = 4 * Real.sqrt 3 → 
  S = (1 / 2) * b * c * Real.sin A → 
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1935_193557


namespace NUMINAMATH_CALUDE_total_amount_calculation_total_amount_is_3693_2_l1935_193545

/-- Calculate the total amount received after selling three items with given prices, losses, and VAT -/
theorem total_amount_calculation (price_A price_B price_C : ℝ)
                                 (loss_A loss_B loss_C : ℝ)
                                 (vat : ℝ) : ℝ :=
  let selling_price_A := price_A * (1 - loss_A)
  let selling_price_B := price_B * (1 - loss_B)
  let selling_price_C := price_C * (1 - loss_C)
  let total_selling_price := selling_price_A + selling_price_B + selling_price_C
  let total_with_vat := total_selling_price * (1 + vat)
  total_with_vat

/-- The total amount received after selling all three items, including VAT, is Rs. 3693.2 -/
theorem total_amount_is_3693_2 :
  total_amount_calculation 1300 750 1800 0.20 0.15 0.10 0.12 = 3693.2 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_calculation_total_amount_is_3693_2_l1935_193545


namespace NUMINAMATH_CALUDE_project_scientists_count_l1935_193563

theorem project_scientists_count :
  ∀ S : ℕ,
  (S / 2 : ℚ) + (S / 5 : ℚ) + 21 = S →
  S = 70 := by
sorry

end NUMINAMATH_CALUDE_project_scientists_count_l1935_193563


namespace NUMINAMATH_CALUDE_complex_number_modulus_l1935_193512

theorem complex_number_modulus (z : ℂ) : (1 + z) / (1 - z) = Complex.I → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l1935_193512


namespace NUMINAMATH_CALUDE_cleanup_time_is_25_minutes_l1935_193558

/-- Represents the toy cleaning scenario -/
structure ToyCleaningScenario where
  totalToys : ℕ
  momPutRate : ℕ
  miaTakeRate : ℕ
  brotherTossRate : ℕ
  momCycleTime : ℕ
  brotherCycleTime : ℕ

/-- Calculates the time taken to clean up all toys -/
def cleanupTime (scenario : ToyCleaningScenario) : ℚ :=
  sorry

/-- Theorem stating that the cleanup time for the given scenario is 25 minutes -/
theorem cleanup_time_is_25_minutes :
  let scenario : ToyCleaningScenario := {
    totalToys := 40,
    momPutRate := 4,
    miaTakeRate := 3,
    brotherTossRate := 1,
    momCycleTime := 20,
    brotherCycleTime := 40
  }
  cleanupTime scenario = 25 := by
  sorry

end NUMINAMATH_CALUDE_cleanup_time_is_25_minutes_l1935_193558


namespace NUMINAMATH_CALUDE_product_of_decimals_l1935_193596

theorem product_of_decimals : (0.5 : ℝ) * 0.3 = 0.15 := by sorry

end NUMINAMATH_CALUDE_product_of_decimals_l1935_193596


namespace NUMINAMATH_CALUDE_new_average_after_grace_marks_l1935_193578

theorem new_average_after_grace_marks 
  (num_students : ℕ) 
  (original_average : ℚ) 
  (grace_marks : ℚ) 
  (h1 : num_students = 35) 
  (h2 : original_average = 37) 
  (h3 : grace_marks = 3) : 
  (num_students * original_average + num_students * grace_marks) / num_students = 40 := by
sorry

end NUMINAMATH_CALUDE_new_average_after_grace_marks_l1935_193578


namespace NUMINAMATH_CALUDE_bryans_mineral_samples_per_shelf_l1935_193508

/-- Given Bryan's mineral collection setup, prove the number of samples per shelf. -/
theorem bryans_mineral_samples_per_shelf :
  let total_samples : ℕ := 455
  let total_shelves : ℕ := 7
  let samples_per_shelf : ℕ := total_samples / total_shelves
  samples_per_shelf = 65 := by
  sorry

end NUMINAMATH_CALUDE_bryans_mineral_samples_per_shelf_l1935_193508


namespace NUMINAMATH_CALUDE_silver_division_representation_l1935_193581

/-- Represents the problem of dividing silver among guests -/
structure SilverDivision where
  guests : ℕ      -- number of guests
  silver : ℕ      -- total amount of silver in taels

/-- The conditions of the silver division problem are satisfied -/
def satisfiesConditions (sd : SilverDivision) : Prop :=
  (7 * sd.guests = sd.silver - 4) ∧ 
  (9 * sd.guests = sd.silver + 8)

/-- The system of equations correctly represents the silver division problem -/
theorem silver_division_representation (sd : SilverDivision) : 
  satisfiesConditions sd ↔ 
  (∃ x y : ℕ, 
    sd.guests = x ∧ 
    sd.silver = y ∧ 
    7 * x = y - 4 ∧ 
    9 * x = y + 8) :=
sorry

end NUMINAMATH_CALUDE_silver_division_representation_l1935_193581


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1935_193510

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∀ x : ℝ, 4 * x^2 + (a - 2) * x + 1 > 0) ↔ 
  (a ≤ -2 ∨ a ≥ 6) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1935_193510


namespace NUMINAMATH_CALUDE_translation_proof_l1935_193546

/-- Represents a line in the form y = mx + b -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Translates a line vertically by a given distance -/
def translateVertically (l : Line) (distance : ℝ) : Line :=
  { slope := l.slope, yIntercept := l.yIntercept + distance }

theorem translation_proof (l₁ l₂ : Line) :
  l₁.slope = 2 ∧ l₁.yIntercept = -2 ∧ l₂.slope = 2 ∧ l₂.yIntercept = 0 →
  translateVertically l₁ 2 = l₂ := by
  sorry

end NUMINAMATH_CALUDE_translation_proof_l1935_193546


namespace NUMINAMATH_CALUDE_candy_bar_payment_l1935_193539

/-- Calculates the number of dimes used to pay for a candy bar -/
def dimes_used (quarter_value : ℕ) (nickel_value : ℕ) (dime_value : ℕ) 
  (num_quarters : ℕ) (num_nickels : ℕ) (change : ℕ) (candy_cost : ℕ) : ℕ :=
  let total_paid := candy_cost + change
  let paid_without_dimes := num_quarters * quarter_value + num_nickels * nickel_value
  (total_paid - paid_without_dimes) / dime_value

theorem candy_bar_payment :
  dimes_used 25 5 10 4 1 4 131 = 3 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_payment_l1935_193539


namespace NUMINAMATH_CALUDE_sqrt_five_identity_l1935_193585

theorem sqrt_five_identity (m n a b c d : ℝ) :
  m + n * Real.sqrt 5 = (a + b * Real.sqrt 5) * (c + d * Real.sqrt 5) →
  m - n * Real.sqrt 5 = (a - b * Real.sqrt 5) * (c - d * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_five_identity_l1935_193585


namespace NUMINAMATH_CALUDE_shaded_area_of_carpet_l1935_193559

/-- Given a square carpet with side length 12 feet, one large shaded square,
    and twelve smaller congruent shaded squares, where the ratios of side lengths
    are as specified, the total shaded area is 15.75 square feet. -/
theorem shaded_area_of_carpet (S T : ℝ) : 
  (12 : ℝ) / S = 4 →
  S / T = 4 →
  S^2 + 12 * T^2 = 15.75 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_carpet_l1935_193559


namespace NUMINAMATH_CALUDE_new_cost_relation_l1935_193506

/-- Represents the manufacturing cost function -/
def cost (k t b : ℝ) : ℝ := k * (t * b) ^ 4

/-- Theorem: New cost after doubling batches and reducing time by 25% -/
theorem new_cost_relation (k t b : ℝ) (h_pos : t > 0 ∧ b > 0) :
  cost k (0.75 * t) (2 * b) = 25.62890625 * cost k t b := by
  sorry

#check new_cost_relation

end NUMINAMATH_CALUDE_new_cost_relation_l1935_193506


namespace NUMINAMATH_CALUDE_smallest_special_number_proof_l1935_193582

/-- A function that returns true if a natural number uses exactly four different digits -/
def uses_four_different_digits (n : ℕ) : Prop :=
  (Finset.card (Finset.image (λ d => d % 10) (Finset.range 4))) = 4

/-- The smallest natural number greater than 3429 that uses exactly four different digits -/
def smallest_special_number : ℕ := 3450

theorem smallest_special_number_proof :
  smallest_special_number > 3429 ∧
  uses_four_different_digits smallest_special_number ∧
  ∀ n : ℕ, n > 3429 ∧ n < smallest_special_number → ¬(uses_four_different_digits n) :=
sorry

end NUMINAMATH_CALUDE_smallest_special_number_proof_l1935_193582


namespace NUMINAMATH_CALUDE_triangle_sin_A_l1935_193587

theorem triangle_sin_A (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (B > 0) → (C > 0) →
  (A + B + C = Real.pi) →
  -- Given conditions
  (a = 2) →
  (b = 3) →
  (Real.tan B = 3) →
  -- Law of Sines (assumed as part of triangle definition)
  (a / Real.sin A = b / Real.sin B) →
  -- Conclusion
  Real.sin A = Real.sqrt 10 / 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_sin_A_l1935_193587


namespace NUMINAMATH_CALUDE_missy_tv_watching_l1935_193530

/-- The number of reality shows Missy watches -/
def num_reality_shows : ℕ := 5

/-- The duration of each reality show in minutes -/
def reality_show_duration : ℕ := 28

/-- The duration of the cartoon in minutes -/
def cartoon_duration : ℕ := 10

/-- The total time Missy spends watching TV in minutes -/
def total_watch_time : ℕ := 150

theorem missy_tv_watching :
  num_reality_shows * reality_show_duration + cartoon_duration = total_watch_time :=
by sorry

end NUMINAMATH_CALUDE_missy_tv_watching_l1935_193530


namespace NUMINAMATH_CALUDE_complex_sequence_counterexample_l1935_193517

-- Define the "sequence" relation on complex numbers
def complex_gt (z₁ z₂ : ℂ) : Prop :=
  z₁.re > z₂.re ∨ (z₁.re = z₂.re ∧ z₁.im > z₂.im)

-- Define positive complex numbers
def complex_pos (z : ℂ) : Prop :=
  complex_gt z 0

-- Theorem statement
theorem complex_sequence_counterexample :
  ∃ (z z₁ z₂ : ℂ), complex_pos z ∧ complex_gt z₁ z₂ ∧ ¬(complex_gt (z * z₁) (z * z₂)) := by
  sorry

end NUMINAMATH_CALUDE_complex_sequence_counterexample_l1935_193517


namespace NUMINAMATH_CALUDE_fraction_addition_l1935_193589

theorem fraction_addition : (1 : ℚ) / 4 + (3 : ℚ) / 8 = (5 : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1935_193589


namespace NUMINAMATH_CALUDE_count_equality_l1935_193509

/-- The count of natural numbers from 1 to 3998 that are divisible by 4 -/
def count_divisible_by_4 : ℕ := 999

/-- The count of natural numbers from 1 to 3998 whose digit sum is divisible by 4 -/
def count_digit_sum_divisible_by_4 : ℕ := 999

/-- The upper bound of the range of natural numbers being considered -/
def upper_bound : ℕ := 3998

theorem count_equality :
  count_divisible_by_4 = count_digit_sum_divisible_by_4 ∧
  count_divisible_by_4 = (upper_bound / 4 : ℕ) :=
sorry

end NUMINAMATH_CALUDE_count_equality_l1935_193509


namespace NUMINAMATH_CALUDE_supermarket_sales_problem_l1935_193560

/-- Represents the monthly sales volume as a function of selling price -/
def sales_volume (x : ℝ) : ℝ := -5 * x + 550

/-- Represents the monthly profit as a function of selling price -/
def monthly_profit (x : ℝ) : ℝ := sales_volume x * (x - 50)

/-- The cost per item -/
def cost : ℝ := 50

/-- The initial selling price -/
def initial_price : ℝ := 100

/-- The initial monthly sales -/
def initial_sales : ℝ := 50

/-- The change in sales for every 2 yuan decrease in price -/
def sales_change : ℝ := 10

theorem supermarket_sales_problem :
  (∀ x : ℝ, x ≥ cost → sales_volume x = -5 * x + 550) ∧
  (∃ x : ℝ, x ≥ cost ∧ monthly_profit x = 4000 ∧ x = 70) ∧
  (∃ x : ℝ, x ≥ cost ∧ ∀ y : ℝ, y ≥ cost → monthly_profit x ≥ monthly_profit y ∧ x = 80) :=
by sorry

end NUMINAMATH_CALUDE_supermarket_sales_problem_l1935_193560


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1935_193541

/-- Represents an isosceles triangle with base 4 and leg length x -/
structure IsoscelesTriangle where
  x : ℝ
  is_root : x^2 - 5*x + 6 = 0
  is_valid : x + x > 4

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := 2 * t.x + 4

theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, perimeter t = 10 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1935_193541


namespace NUMINAMATH_CALUDE_xy_negative_sufficient_not_necessary_l1935_193514

theorem xy_negative_sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x * y < 0 → |x - y| = |x| + |y|) ∧
  (∃ x y : ℝ, |x - y| = |x| + |y| ∧ x * y ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_xy_negative_sufficient_not_necessary_l1935_193514


namespace NUMINAMATH_CALUDE_equation_solution_l1935_193573

theorem equation_solution : 
  let t : ℚ := -8
  (1 : ℚ) / (t + 2) + (2 * t) / (t + 2) - (3 : ℚ) / (t + 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1935_193573


namespace NUMINAMATH_CALUDE_borrowing_period_is_one_year_l1935_193527

-- Define the problem parameters
def initial_amount : ℕ := 5000
def borrowing_rate : ℚ := 4 / 100
def lending_rate : ℚ := 6 / 100
def gain_per_year : ℕ := 100

-- Define the function to calculate interest
def calculate_interest (amount : ℕ) (rate : ℚ) : ℚ :=
  (amount : ℚ) * rate

-- Define the function to calculate the gain
def calculate_gain (amount : ℕ) (borrow_rate lending_rate : ℚ) : ℚ :=
  calculate_interest amount lending_rate - calculate_interest amount borrow_rate

-- Theorem statement
theorem borrowing_period_is_one_year :
  calculate_gain initial_amount borrowing_rate lending_rate = gain_per_year := by
  sorry

end NUMINAMATH_CALUDE_borrowing_period_is_one_year_l1935_193527


namespace NUMINAMATH_CALUDE_largest_R_under_condition_l1935_193515

theorem largest_R_under_condition : ∃ (R : ℕ), R > 0 ∧ R^2000 < 5^3000 ∧ ∀ (S : ℕ), S > R → S^2000 ≥ 5^3000 :=
by sorry

end NUMINAMATH_CALUDE_largest_R_under_condition_l1935_193515


namespace NUMINAMATH_CALUDE_total_tomatoes_l1935_193554

/-- The number of cucumber rows for each tomato row -/
def cucumber_rows_per_tomato_row : ℕ := 2

/-- The total number of rows in the garden -/
def total_rows : ℕ := 15

/-- The number of tomato plants in each row -/
def plants_per_row : ℕ := 8

/-- The number of tomatoes produced by each plant -/
def tomatoes_per_plant : ℕ := 3

/-- The theorem stating the total number of tomatoes Aubrey will have -/
theorem total_tomatoes : 
  (total_rows / (cucumber_rows_per_tomato_row + 1)) * plants_per_row * tomatoes_per_plant = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_tomatoes_l1935_193554


namespace NUMINAMATH_CALUDE_ecommerce_problem_l1935_193583

theorem ecommerce_problem (total_spent : ℝ) (price_difference : ℝ) (total_items : ℕ) 
  (subsidy_rate : ℝ) (max_subsidy : ℝ) 
  (h1 : total_spent = 3000)
  (h2 : price_difference = 600)
  (h3 : total_items = 300)
  (h4 : subsidy_rate = 0.1)
  (h5 : max_subsidy = 50000) :
  ∃ (leather_price sweater_price : ℝ) (min_sweaters : ℕ),
    leather_price = 2600 ∧ 
    sweater_price = 400 ∧ 
    min_sweaters = 128 ∧
    leather_price + sweater_price = total_spent ∧
    leather_price = 5 * sweater_price + price_difference ∧
    (↑min_sweaters : ℝ) ≥ (max_subsidy / subsidy_rate - total_items * leather_price) / (sweater_price - leather_price) := by
  sorry

end NUMINAMATH_CALUDE_ecommerce_problem_l1935_193583


namespace NUMINAMATH_CALUDE_probability_two_non_defective_pens_l1935_193552

/-- Represents the probability of selecting non-defective pens from a box -/
def probability_non_defective (total_pens : ℕ) (defective_pens : ℕ) (selected_pens : ℕ) : ℚ :=
  let non_defective := total_pens - defective_pens
  (non_defective.choose selected_pens : ℚ) / (total_pens.choose selected_pens)

/-- Theorem stating the probability of selecting 2 non-defective pens from a box of 16 pens with 3 defective pens -/
theorem probability_two_non_defective_pens :
  probability_non_defective 16 3 2 = 13/20 := by
  sorry

#eval probability_non_defective 16 3 2

end NUMINAMATH_CALUDE_probability_two_non_defective_pens_l1935_193552


namespace NUMINAMATH_CALUDE_min_students_with_both_devices_l1935_193549

theorem min_students_with_both_devices (n : ℕ) (laptop_users tablet_users : ℕ) : 
  laptop_users = (3 * n) / 7 →
  tablet_users = (5 * n) / 6 →
  ∃ (both : ℕ), both ≥ 11 ∧ n ≥ laptop_users + tablet_users - both :=
sorry

end NUMINAMATH_CALUDE_min_students_with_both_devices_l1935_193549


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l1935_193542

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem purely_imaginary_condition (a : ℝ) :
  is_purely_imaginary ((a^2 - 1 : ℝ) + (2 * (a + 1) : ℝ) * I) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l1935_193542


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l1935_193567

theorem vector_difference_magnitude (a b : ℝ × ℝ × ℝ) :
  a = (2, 3, -1) → b = (-2, 1, 3) → ‖a - b‖ = 6 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l1935_193567


namespace NUMINAMATH_CALUDE_percentage_students_taking_music_l1935_193537

/-- The percentage of students taking music in a school with various electives -/
theorem percentage_students_taking_music
  (total_students : ℕ)
  (dance_percent : ℝ)
  (art_percent : ℝ)
  (drama_percent : ℝ)
  (sports_percent : ℝ)
  (photography_percent : ℝ)
  (h_total : total_students = 3000)
  (h_dance : dance_percent = 12.5)
  (h_art : art_percent = 22)
  (h_drama : drama_percent = 13.5)
  (h_sports : sports_percent = 15)
  (h_photo : photography_percent = 8) :
  100 - (dance_percent + art_percent + drama_percent + sports_percent + photography_percent) = 29 := by
  sorry

end NUMINAMATH_CALUDE_percentage_students_taking_music_l1935_193537


namespace NUMINAMATH_CALUDE_binomial_divisibility_iff_prime_l1935_193598

theorem binomial_divisibility_iff_prime (m : ℕ) (h : m ≥ 2) :
  (∀ n : ℕ, m / 3 ≤ n ∧ n ≤ m / 2 → n ∣ Nat.choose n (m - 2*n)) ↔ Nat.Prime m :=
sorry

end NUMINAMATH_CALUDE_binomial_divisibility_iff_prime_l1935_193598


namespace NUMINAMATH_CALUDE_propositions_truth_l1935_193522

-- Definition of correlation coefficient
def correlation_strength (r : ℝ) : ℝ := 1 - |r|

-- Definition of perpendicular lines
def perpendicular (A B C A' B' C' : ℝ) : Prop := A * A' + B * B' = 0

theorem propositions_truth : 
  -- Proposition 1 (false)
  (∃ x : ℝ, x^2 < 0 ↔ ¬ ∀ x : ℝ, x^2 ≥ 0) ∧
  -- Proposition 2 (true)
  (∀ r : ℝ, |r| ≤ 1 → correlation_strength r ≤ correlation_strength 0) ∧
  -- Proposition 3 (false, not included)
  -- Proposition 4 (true)
  perpendicular 2 10 6 3 (-3/5) (13/5) :=
by sorry

end NUMINAMATH_CALUDE_propositions_truth_l1935_193522


namespace NUMINAMATH_CALUDE_circle_passes_through_points_circle_equation_equivalence_l1935_193550

-- Define the circle passing through points O(0,0), A(1,1), and B(4,2)
def circle_through_points (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 6*y = 0

-- Define the standard form of the circle
def circle_standard_form (x y : ℝ) : Prop :=
  (x - 4)^2 + (y + 3)^2 = 25

-- Theorem stating that the circle passes through the given points
theorem circle_passes_through_points :
  circle_through_points 0 0 ∧
  circle_through_points 1 1 ∧
  circle_through_points 4 2 := by sorry

-- Theorem stating the equivalence of the general and standard forms
theorem circle_equation_equivalence :
  ∀ x y : ℝ, circle_through_points x y ↔ circle_standard_form x y := by sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_circle_equation_equivalence_l1935_193550


namespace NUMINAMATH_CALUDE_train_length_is_286_l1935_193531

/-- The speed of the pedestrian in meters per second -/
def pedestrian_speed : ℝ := 1

/-- The speed of the cyclist in meters per second -/
def cyclist_speed : ℝ := 3

/-- The time it takes for the train to pass the pedestrian in seconds -/
def pedestrian_passing_time : ℝ := 22

/-- The time it takes for the train to pass the cyclist in seconds -/
def cyclist_passing_time : ℝ := 26

/-- The speed of the train in meters per second -/
def train_speed : ℝ := 14

/-- The length of the train in meters -/
def train_length : ℝ := (train_speed - pedestrian_speed) * pedestrian_passing_time

theorem train_length_is_286 : train_length = 286 := by
  sorry

end NUMINAMATH_CALUDE_train_length_is_286_l1935_193531


namespace NUMINAMATH_CALUDE_lcm_of_3_8_9_12_l1935_193525

theorem lcm_of_3_8_9_12 : Nat.lcm 3 (Nat.lcm 8 (Nat.lcm 9 12)) = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_3_8_9_12_l1935_193525


namespace NUMINAMATH_CALUDE_range_of_a_l1935_193586

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - a) * x + 3 else Real.log x - 2 * a

theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → -4 ≤ a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1935_193586


namespace NUMINAMATH_CALUDE_cubic_solutions_l1935_193528

-- Define the complex cubic equation
def cubic_equation (z : ℂ) : Prop := z^3 = -8

-- State the theorem
theorem cubic_solutions :
  ∃ (z₁ z₂ z₃ : ℂ),
    z₁ = -2 ∧
    z₂ = 1 + Complex.I * Real.sqrt 3 ∧
    z₃ = 1 - Complex.I * Real.sqrt 3 ∧
    cubic_equation z₁ ∧
    cubic_equation z₂ ∧
    cubic_equation z₃ ∧
    ∀ (z : ℂ), cubic_equation z → (z = z₁ ∨ z = z₂ ∨ z = z₃) :=
by sorry

end NUMINAMATH_CALUDE_cubic_solutions_l1935_193528


namespace NUMINAMATH_CALUDE_exponent_sum_l1935_193591

theorem exponent_sum (x a b c : ℝ) (h1 : x ≠ 1) (h2 : x * x^a * x^b * x^c = x^2024) : 
  a + b + c = 2023 := by
sorry

end NUMINAMATH_CALUDE_exponent_sum_l1935_193591


namespace NUMINAMATH_CALUDE_unique_solution_cubic_system_l1935_193595

theorem unique_solution_cubic_system (x y z : ℝ) 
  (h1 : x + y + z = 3)
  (h2 : x^2 + y^2 + z^2 = 3)
  (h3 : x^3 + y^3 + z^3 = 3) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_system_l1935_193595


namespace NUMINAMATH_CALUDE_projection_result_l1935_193584

def v1 : ℝ × ℝ := (3, 2)
def v2 : ℝ × ℝ := (2, 5)

theorem projection_result (u : ℝ × ℝ) (q : ℝ × ℝ) 
  (h1 : ∃ (k1 : ℝ), q = k1 • u ∧ (v1 - q) • u = 0)
  (h2 : ∃ (k2 : ℝ), q = k2 • u ∧ (v2 - q) • u = 0) :
  q = (33/10, 11/10) :=
sorry

end NUMINAMATH_CALUDE_projection_result_l1935_193584


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1935_193566

theorem square_area_from_diagonal (d : ℝ) (h : d = 40) : 
  (d^2 / 2) = 800 := by sorry

#check square_area_from_diagonal

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1935_193566


namespace NUMINAMATH_CALUDE_perp_planes_necessary_not_sufficient_l1935_193544

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset_line_plane : Line → Plane → Prop)

-- Main theorem
theorem perp_planes_necessary_not_sufficient
  (α β : Plane) (m : Line)
  (h_diff : α ≠ β)
  (h_subset : subset_line_plane m α) :
  (perp_planes α β → perp_line_plane m β) ∧
  ¬(perp_line_plane m β → perp_planes α β) :=
sorry

end NUMINAMATH_CALUDE_perp_planes_necessary_not_sufficient_l1935_193544


namespace NUMINAMATH_CALUDE_value_of_m_l1935_193561

theorem value_of_m : ∀ m : ℝ,
  let f : ℝ → ℝ := λ x ↦ 3 * x^3 - 1 / x + 5
  let g : ℝ → ℝ := λ x ↦ 3 * x^2 - m
  (f (-1) - g (-1) = 1) → m = 1 := by
sorry

end NUMINAMATH_CALUDE_value_of_m_l1935_193561


namespace NUMINAMATH_CALUDE_no_y_intercepts_l1935_193524

/-- The parabola equation -/
def parabola_equation (y : ℝ) : ℝ := 3 * y^2 - 5 * y + 9

/-- Theorem: The parabola x = 3y^2 - 5y + 9 has no y-intercepts -/
theorem no_y_intercepts : ¬ ∃ y : ℝ, parabola_equation y = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_y_intercepts_l1935_193524


namespace NUMINAMATH_CALUDE_tangent_line_intercept_l1935_193580

/-- Given a curve y = x³ + ax + 1 and a line y = kx + b tangent to the curve at (2, 3), prove b = -15 -/
theorem tangent_line_intercept (a k b : ℝ) : 
  (3 = 2^3 + a*2 + 1) →  -- The curve passes through (2, 3)
  (k = 3*2^2 + a) →      -- The slope of the tangent line equals the derivative at x = 2
  (3 = k*2 + b) →        -- The line passes through (2, 3)
  b = -15 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_intercept_l1935_193580


namespace NUMINAMATH_CALUDE_negative_fractions_comparison_l1935_193597

theorem negative_fractions_comparison : -1/3 < -1/4 := by
  sorry

end NUMINAMATH_CALUDE_negative_fractions_comparison_l1935_193597


namespace NUMINAMATH_CALUDE_optimal_planting_correct_l1935_193500

structure FlowerPlanting where
  cost_A_3_B_4 : ℝ
  cost_A_4_B_3 : ℝ
  total_pots : ℕ
  survival_rate_A : ℝ
  survival_rate_B : ℝ
  max_replacement : ℕ

def optimal_planting (fp : FlowerPlanting) : ℕ × ℕ × ℝ :=
  sorry

theorem optimal_planting_correct (fp : FlowerPlanting) 
  (h1 : fp.cost_A_3_B_4 = 330)
  (h2 : fp.cost_A_4_B_3 = 300)
  (h3 : fp.total_pots = 400)
  (h4 : fp.survival_rate_A = 0.7)
  (h5 : fp.survival_rate_B = 0.9)
  (h6 : fp.max_replacement = 80) :
  optimal_planting fp = (200, 200, 18000) :=
sorry

end NUMINAMATH_CALUDE_optimal_planting_correct_l1935_193500


namespace NUMINAMATH_CALUDE_sum_of_square_and_triangular_l1935_193538

theorem sum_of_square_and_triangular (k : ℕ) :
  let Sₖ := (6 * 10^k - 1) * 10^(k+2) + 5 * 10^(k+1) + 1
  let n := 2 * 10^(k+1) - 1
  Sₖ = n^2 + n * (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_and_triangular_l1935_193538


namespace NUMINAMATH_CALUDE_seans_soda_purchase_l1935_193577

/-- The number of cans of soda Sean bought -/
def num_sodas : ℕ := sorry

/-- The cost of one soup in dollars -/
def cost_soup : ℚ := sorry

/-- The cost of the sandwich in dollars -/
def cost_sandwich : ℚ := sorry

/-- The total cost of the purchase in dollars -/
def total_cost : ℚ := 18

theorem seans_soda_purchase :
  (num_sodas : ℚ) = cost_soup ∧
  cost_sandwich = 3 * cost_soup ∧
  (num_sodas : ℚ) * 1 + 2 * cost_soup + cost_sandwich = total_cost ∧
  num_sodas = 3 := by sorry

end NUMINAMATH_CALUDE_seans_soda_purchase_l1935_193577


namespace NUMINAMATH_CALUDE_triangle_side_equation_l1935_193592

/-- Given a triangle ABC with vertex A at (1,4), and angle bisectors of B and C
    represented by the equations x + y - 1 = 0 and x - 2y = 0 respectively,
    the side BC lies on the line with equation 4x + 17y + 12 = 0. -/
theorem triangle_side_equation (B C : ℝ × ℝ) : 
  let A : ℝ × ℝ := (1, 4)
  let angle_bisector_B : ℝ → ℝ → Prop := λ x y => x + y - 1 = 0
  let angle_bisector_C : ℝ → ℝ → Prop := λ x y => x - 2*y = 0
  let line_BC : ℝ → ℝ → Prop := λ x y => 4*x + 17*y + 12 = 0
  (∀ x y, x = B.1 ∧ y = B.2 → angle_bisector_B x y) →
  (∀ x y, x = C.1 ∧ y = C.2 → angle_bisector_C x y) →
  (∀ x y, (x = B.1 ∧ y = B.2) ∨ (x = C.1 ∧ y = C.2) → line_BC x y) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_equation_l1935_193592


namespace NUMINAMATH_CALUDE_square_polynomial_k_values_l1935_193513

theorem square_polynomial_k_values (k : ℝ) : 
  (∃ p : ℝ → ℝ, ∀ x, x^2 + 2*(k-1)*x + 64 = (p x)^2) → 
  (k = 9 ∨ k = -7) := by
  sorry

end NUMINAMATH_CALUDE_square_polynomial_k_values_l1935_193513


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_l1935_193518

theorem smallest_solution_quadratic : 
  let f : ℝ → ℝ := fun y ↦ 10 * y^2 - 47 * y + 49
  ∃ y : ℝ, f y = 0 ∧ (∀ z : ℝ, f z = 0 → y ≤ z) ∧ y = 1.4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_l1935_193518


namespace NUMINAMATH_CALUDE_no_solution_for_12x4x_divisible_by_99_l1935_193562

theorem no_solution_for_12x4x_divisible_by_99 : 
  ¬ ∃ x : ℕ, x ≤ 9 ∧ (12000 + 1000*x + 40 + x) % 99 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_12x4x_divisible_by_99_l1935_193562


namespace NUMINAMATH_CALUDE_berry_problem_l1935_193526

/-- Proves that given the conditions in the berry problem, Steve started with 8.5 berries and Amanda started with 3.5 berries. -/
theorem berry_problem (stacy_initial : ℝ) (steve_takes : ℝ) (amanda_takes : ℝ) (amanda_more : ℝ)
  (h1 : stacy_initial = 32)
  (h2 : steve_takes = 4)
  (h3 : amanda_takes = 3.25)
  (h4 : amanda_more = 5.75)
  (h5 : steve_takes + (stacy_initial / 2 - 7.5) = stacy_initial / 2 - 7.5 + steve_takes - amanda_takes + amanda_more) :
  (stacy_initial / 2 - 7.5 = 8.5) ∧ (stacy_initial / 2 - 7.5 + steve_takes - amanda_takes - amanda_more = 3.5) :=
by sorry

end NUMINAMATH_CALUDE_berry_problem_l1935_193526


namespace NUMINAMATH_CALUDE_solve_equations_l1935_193534

theorem solve_equations :
  (∃ x1 x2 : ℝ, 2 * x1 * (x1 - 1) = 1 ∧ 2 * x2 * (x2 - 1) = 1 ∧
    x1 = (1 + Real.sqrt 3) / 2 ∧ x2 = (1 - Real.sqrt 3) / 2) ∧
  (∃ y1 y2 : ℝ, y1^2 + 8*y1 + 7 = 0 ∧ y2^2 + 8*y2 + 7 = 0 ∧
    y1 = -7 ∧ y2 = -1) :=
by sorry

end NUMINAMATH_CALUDE_solve_equations_l1935_193534


namespace NUMINAMATH_CALUDE_factor_of_x4_plus_8_l1935_193536

theorem factor_of_x4_plus_8 (x : ℝ) : 
  (x^2 - 2*x + 4) * (x^2 + 2*x + 4) = x^4 + 8 := by
  sorry

end NUMINAMATH_CALUDE_factor_of_x4_plus_8_l1935_193536


namespace NUMINAMATH_CALUDE_exactly_two_solutions_l1935_193519

/-- The number of ordered pairs (a, b) of positive integers satisfying the equation -/
def solution_count : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    p.1 > 0 ∧ p.2 > 0 ∧
    2 * p.1 * p.2 + 108 = 15 * Nat.lcm p.1 p.2 + 18 * Nat.gcd p.1 p.2
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- The main theorem stating that there are exactly 2 solutions -/
theorem exactly_two_solutions : solution_count = 2 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_solutions_l1935_193519


namespace NUMINAMATH_CALUDE_kate_bouncy_balls_l1935_193574

/-- The number of packs of yellow bouncy balls Kate bought -/
def yellow_packs : ℕ := 6

/-- The number of packs of red bouncy balls Kate bought -/
def red_packs : ℕ := 7

/-- The number of bouncy balls in each pack -/
def balls_per_pack : ℕ := 18

theorem kate_bouncy_balls :
  yellow_packs * balls_per_pack + 18 = red_packs * balls_per_pack :=
by sorry

end NUMINAMATH_CALUDE_kate_bouncy_balls_l1935_193574


namespace NUMINAMATH_CALUDE_log_sqrt10_1000sqrt10_l1935_193576

theorem log_sqrt10_1000sqrt10 :
  Real.log (1000 * Real.sqrt 10) / Real.log (Real.sqrt 10) = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_sqrt10_1000sqrt10_l1935_193576


namespace NUMINAMATH_CALUDE_expression_simplification_l1935_193540

theorem expression_simplification (a x y : ℝ) : 
  ((-2*a)^6*(-3*a^3) + (2*a^2)^3 / (1 / ((-2)^2 * 3^2 * (x*y)^3))) = 192*a^9 + 288*a^6*(x*y)^3 ∧
  |-(1/8)| + π^3 + (-(1/2)^3 - (1/3)^2) = π^3 - 1/72 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1935_193540


namespace NUMINAMATH_CALUDE_factorization_m_squared_plus_3m_l1935_193553

theorem factorization_m_squared_plus_3m (m : ℝ) : m^2 + 3*m = m*(m+3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_m_squared_plus_3m_l1935_193553


namespace NUMINAMATH_CALUDE_circle_and_tangent_lines_l1935_193533

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define points A, B, and M
def A : ℝ × ℝ := (0, -6)
def B : ℝ × ℝ := (1, -5)
def M : ℝ × ℝ := (2, 8)

-- Define the line l: x - y + 1 = 0
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 + 1 = 0}

-- Theorem statement
theorem circle_and_tangent_lines 
  (C : ℝ × ℝ) -- Center of the circle
  (h1 : C ∈ l) -- Center lies on line l
  (h2 : A ∈ Circle C (|C.1 - A.1|)) -- A is on the circle
  (h3 : B ∈ Circle C (|C.1 - A.1|)) -- B is on the circle
  : 
  -- 1. Standard equation of the circle
  (∀ p : ℝ × ℝ, p ∈ Circle C (|C.1 - A.1|) ↔ (p.1 + 3)^2 + (p.2 + 2)^2 = 25) ∧
  -- 2. Equations of tangent lines
  (∀ p : ℝ × ℝ, (p.1 = 2 ∨ 3*p.1 - 4*p.2 + 26 = 0) ↔ 
    (p ∈ {q : ℝ × ℝ | (q.1 - M.1) * (C.1 - q.1) + (q.2 - M.2) * (C.2 - q.2) = 0} ∧
     p ∈ Circle C (|C.1 - A.1|))) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_tangent_lines_l1935_193533


namespace NUMINAMATH_CALUDE_line_equation_through_midpoint_l1935_193579

/-- Given an ellipse and a point M, prove the equation of a line passing through M and intersecting the ellipse at two points where M is their midpoint. -/
theorem line_equation_through_midpoint (x y : ℝ) : 
  let M : ℝ × ℝ := (2, 1)
  let ellipse (x y : ℝ) := x^2 / 16 + y^2 / 4 = 1
  ∃ A B : ℝ × ℝ, 
    ellipse A.1 A.2 ∧ 
    ellipse B.1 B.2 ∧ 
    M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
    ∃ l : ℝ → ℝ → Prop, 
      (∀ x y, l x y ↔ x + 2*y - 4 = 0) ∧
      l A.1 A.2 ∧ 
      l B.1 B.2 ∧ 
      l M.1 M.2 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_midpoint_l1935_193579


namespace NUMINAMATH_CALUDE_photocopy_cost_l1935_193588

/-- The cost of a single photocopy --/
def C : ℝ := sorry

/-- The discount rate for large orders --/
def discount_rate : ℝ := 0.25

/-- The number of copies in a large order --/
def large_order : ℕ := 160

/-- The total cost savings when placing a large order --/
def total_savings : ℝ := 0.80

theorem photocopy_cost :
  C = 0.02 :=
by sorry

end NUMINAMATH_CALUDE_photocopy_cost_l1935_193588


namespace NUMINAMATH_CALUDE_typhoon_tree_difference_l1935_193551

theorem typhoon_tree_difference (initial_trees : ℕ) (survival_rate : ℚ) : 
  initial_trees = 25 → 
  survival_rate = 2/5 → 
  (initial_trees - (survival_rate * initial_trees).floor) - (survival_rate * initial_trees).floor = 5 := by
  sorry

end NUMINAMATH_CALUDE_typhoon_tree_difference_l1935_193551
