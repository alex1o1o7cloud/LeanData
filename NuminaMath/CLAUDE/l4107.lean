import Mathlib

namespace NUMINAMATH_CALUDE_average_marks_l4107_410725

def english_marks : ℕ := 86
def mathematics_marks : ℕ := 85
def physics_marks : ℕ := 92
def chemistry_marks : ℕ := 87
def biology_marks : ℕ := 95

def total_marks : ℕ := english_marks + mathematics_marks + physics_marks + chemistry_marks + biology_marks
def num_subjects : ℕ := 5

theorem average_marks :
  (total_marks : ℚ) / num_subjects = 89 := by sorry

end NUMINAMATH_CALUDE_average_marks_l4107_410725


namespace NUMINAMATH_CALUDE_ladder_angle_approx_l4107_410745

/-- Given a right triangle with hypotenuse 19 meters and adjacent side 9.493063650744542 meters,
    the angle between the hypotenuse and the adjacent side is approximately 60 degrees. -/
theorem ladder_angle_approx (hypotenuse : ℝ) (adjacent : ℝ) (angle : ℝ) 
    (h1 : hypotenuse = 19)
    (h2 : adjacent = 9.493063650744542)
    (h3 : angle = Real.arccos (adjacent / hypotenuse)) :
    ∃ ε > 0, |angle - 60 * π / 180| < ε :=
  sorry

end NUMINAMATH_CALUDE_ladder_angle_approx_l4107_410745


namespace NUMINAMATH_CALUDE_intersection_M_N_l4107_410788

-- Define set M
def M : Set ℝ := {x | x * (x - 3) < 0}

-- Define set N
def N : Set ℝ := {x | |x| < 2}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l4107_410788


namespace NUMINAMATH_CALUDE_motorboat_speed_adjustment_l4107_410707

/-- 
Given two motorboats with the same initial speed traveling in opposite directions
relative to a river current, prove that if one boat increases its speed by x and
the other decreases by x, resulting in equal time changes, then x equals twice
the current speed.
-/
theorem motorboat_speed_adjustment (v a x : ℝ) (h1 : v > a) (h2 : v > 0) (h3 : a > 0) :
  (1 / (v - a) - 1 / (v + x - a) = 1 / (v + a - x) - 1 / (v + a)) →
  x = 2 * a := by
sorry

end NUMINAMATH_CALUDE_motorboat_speed_adjustment_l4107_410707


namespace NUMINAMATH_CALUDE_length_breadth_difference_l4107_410738

/-- Represents a rectangular plot with given properties -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area_is_23_times_breadth : area = 23 * breadth
  breadth_is_13 : breadth = 13

/-- The area of a rectangle -/
def area (r : RectangularPlot) : ℝ := r.length * r.breadth

/-- Theorem: The difference between length and breadth is 10 meters -/
theorem length_breadth_difference (r : RectangularPlot) :
  r.length - r.breadth = 10 := by
  sorry

#check length_breadth_difference

end NUMINAMATH_CALUDE_length_breadth_difference_l4107_410738


namespace NUMINAMATH_CALUDE_polynomial_simplification_l4107_410734

theorem polynomial_simplification (y : ℝ) :
  (3 * y - 2) * (5 * y^12 + 3 * y^11 + 6 * y^10 + 4 + 2 * y^9) =
  15 * y^13 - y^12 + 12 * y^11 - 6 * y^10 - 4 * y^9 + 12 * y - 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l4107_410734


namespace NUMINAMATH_CALUDE_second_order_size_l4107_410764

/-- Proves that given the specified production rates and average output, the second order contains 60 cogs. -/
theorem second_order_size
  (initial_rate : ℝ)
  (initial_order : ℝ)
  (second_rate : ℝ)
  (average_output : ℝ)
  (h1 : initial_rate = 36)
  (h2 : initial_order = 60)
  (h3 : second_rate = 60)
  (h4 : average_output = 45) :
  ∃ (second_order : ℝ),
    (initial_order + second_order) / ((initial_order / initial_rate) + (second_order / second_rate)) = average_output ∧
    second_order = 60 :=
by sorry

end NUMINAMATH_CALUDE_second_order_size_l4107_410764


namespace NUMINAMATH_CALUDE_black_marble_probability_l4107_410786

/-- The probability of drawing a black marble from a bag -/
theorem black_marble_probability 
  (yellow : ℕ) 
  (blue : ℕ) 
  (green : ℕ) 
  (black : ℕ) 
  (h_yellow : yellow = 12) 
  (h_blue : blue = 10) 
  (h_green : green = 5) 
  (h_black : black = 1) : 
  (black : ℚ) / (yellow + blue + green + black : ℚ) = 1 / 28 := by
  sorry

end NUMINAMATH_CALUDE_black_marble_probability_l4107_410786


namespace NUMINAMATH_CALUDE_product_ab_l4107_410751

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def complex_equation (a b : ℝ) : Prop :=
  (1 + 7 * i) / (2 - i) = (a : ℂ) + b * i

-- Theorem statement
theorem product_ab (a b : ℝ) (h : complex_equation a b) : a * b = -5 := by
  sorry

end NUMINAMATH_CALUDE_product_ab_l4107_410751


namespace NUMINAMATH_CALUDE_power_sum_problem_l4107_410772

theorem power_sum_problem (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 26)
  (h4 : a * x^4 + b * y^4 = 58) :
  a * x^6 + b * y^6 = -220 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_problem_l4107_410772


namespace NUMINAMATH_CALUDE_base_number_proof_l4107_410787

theorem base_number_proof (w : ℕ) (x : ℝ) (h1 : w = 12) (h2 : 2^(2*w) = x^(w-4)) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l4107_410787


namespace NUMINAMATH_CALUDE_hundredth_ring_squares_nth_ring_squares_l4107_410719

/-- The number of unit squares in the nth ring around a center square -/
def ring_squares (n : ℕ) : ℕ := 8 * n

/-- Theorem: The 100th ring contains 800 unit squares -/
theorem hundredth_ring_squares : ring_squares 100 = 800 := by
  sorry

/-- Theorem: For any positive integer n, the number of unit squares in the nth ring is 8n -/
theorem nth_ring_squares (n : ℕ) : ring_squares n = 8 * n := by
  sorry

end NUMINAMATH_CALUDE_hundredth_ring_squares_nth_ring_squares_l4107_410719


namespace NUMINAMATH_CALUDE_base8_digit_product_l4107_410762

/-- Convert a natural number to its base 8 representation -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculate the product of a list of natural numbers -/
def product (l : List ℕ) : ℕ :=
  sorry

/-- The product of the digits in the base 8 representation of 8927 is 126 -/
theorem base8_digit_product : product (toBase8 8927) = 126 :=
  sorry

end NUMINAMATH_CALUDE_base8_digit_product_l4107_410762


namespace NUMINAMATH_CALUDE_inequality_proof_l4107_410785

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*c*a) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4107_410785


namespace NUMINAMATH_CALUDE_evaluate_expression_l4107_410739

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y - 2 * y^x = 277 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l4107_410739


namespace NUMINAMATH_CALUDE_rectangles_with_one_gray_cell_l4107_410779

/-- The number of rectangles containing exactly one gray cell in a checkered rectangle -/
theorem rectangles_with_one_gray_cell 
  (total_gray_cells : ℕ) 
  (cells_with_four_rectangles : ℕ) 
  (cells_with_eight_rectangles : ℕ) 
  (h1 : total_gray_cells = 40)
  (h2 : cells_with_four_rectangles = 36)
  (h3 : cells_with_eight_rectangles = 4)
  (h4 : total_gray_cells = cells_with_four_rectangles + cells_with_eight_rectangles) :
  cells_with_four_rectangles * 4 + cells_with_eight_rectangles * 8 = 176 := by
sorry

end NUMINAMATH_CALUDE_rectangles_with_one_gray_cell_l4107_410779


namespace NUMINAMATH_CALUDE_inverse_of_A_squared_l4107_410737

theorem inverse_of_A_squared (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A⁻¹ = ![![-4, 1], ![0, 2]] →
  (A^2)⁻¹ = ![![16, -2], ![0, 4]] := by
sorry

end NUMINAMATH_CALUDE_inverse_of_A_squared_l4107_410737


namespace NUMINAMATH_CALUDE_coronavirus_radius_scientific_notation_l4107_410711

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem coronavirus_radius_scientific_notation :
  toScientificNotation 0.000000045 =
    ScientificNotation.mk 4.5 (-8) (by norm_num) :=
by sorry

end NUMINAMATH_CALUDE_coronavirus_radius_scientific_notation_l4107_410711


namespace NUMINAMATH_CALUDE_sum_difference_equals_3146_main_theorem_l4107_410716

theorem sum_difference_equals_3146 : ℕ → Prop :=
  fun n =>
    let even_sum := n * (n + 1)
    let multiples_of_3_sum := (n / 3) * ((n / 3) + 1) * 3 / 2
    let odd_sum := ((n - 1) / 2 + 1) ^ 2
    (even_sum - multiples_of_3_sum - odd_sum = 3146) ∧ (2 * n = 400)

theorem main_theorem : ∃ n : ℕ, sum_difference_equals_3146 n := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_equals_3146_main_theorem_l4107_410716


namespace NUMINAMATH_CALUDE_sum_of_cubes_equation_l4107_410778

theorem sum_of_cubes_equation (x y : ℝ) : 
  x^3 + 21*x*y + y^3 = 343 → (x + y = 7 ∨ x + y = -14) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equation_l4107_410778


namespace NUMINAMATH_CALUDE_four_letter_initials_count_l4107_410791

theorem four_letter_initials_count : 
  (Finset.range 10).card ^ 4 = 10000 := by sorry

end NUMINAMATH_CALUDE_four_letter_initials_count_l4107_410791


namespace NUMINAMATH_CALUDE_james_cattle_problem_l4107_410747

/-- Represents the problem of determining the number of cattle James bought --/
theorem james_cattle_problem (purchase_price feeding_cost_percentage cattle_weight selling_price_per_pound profit : ℝ) 
  (h1 : purchase_price = 40000)
  (h2 : feeding_cost_percentage = 0.2)
  (h3 : cattle_weight = 1000)
  (h4 : selling_price_per_pound = 2)
  (h5 : profit = 112000) :
  (purchase_price + purchase_price * feeding_cost_percentage) / 
  (cattle_weight * selling_price_per_pound) + 
  profit / (cattle_weight * selling_price_per_pound) = 100 := by
  sorry

end NUMINAMATH_CALUDE_james_cattle_problem_l4107_410747


namespace NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l4107_410755

theorem smallest_multiple_of_6_and_15 : 
  ∃ (b : ℕ), b > 0 ∧ 6 ∣ b ∧ 15 ∣ b ∧ ∀ (x : ℕ), x > 0 → 6 ∣ x → 15 ∣ x → b ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l4107_410755


namespace NUMINAMATH_CALUDE_percentage_problem_l4107_410718

theorem percentage_problem (x y P : ℝ) 
  (h1 : 0.3 * (x - y) = (P / 100) * (x + y))
  (h2 : y = 0.2 * x) : 
  P = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l4107_410718


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l4107_410706

/-- Given a tetrahedron with volume V, two faces with areas S₁ and S₂, 
    their common edge of length a, and the dihedral angle φ between these faces, 
    prove that V = (2/3) * (S₁ * S₂ * sin(φ)) / a. -/
theorem tetrahedron_volume (V S₁ S₂ a φ : ℝ) 
  (h₁ : V > 0) 
  (h₂ : S₁ > 0) 
  (h₃ : S₂ > 0) 
  (h₄ : a > 0) 
  (h₅ : 0 < φ ∧ φ < π) : 
  V = (2/3) * (S₁ * S₂ * Real.sin φ) / a := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l4107_410706


namespace NUMINAMATH_CALUDE_three_digit_number_proof_l4107_410750

def is_geometric_progression (a b c : ℕ) : Prop :=
  b * b = a * c

def swap_hundreds_units (abc : ℕ) : ℕ :=
  let a := abc / 100
  let b := (abc / 10) % 10
  let c := abc % 10
  c * 100 + b * 10 + a

def last_two_digits (abc : ℕ) : ℕ :=
  abc % 100

def swap_last_two_digits (abc : ℕ) : ℕ :=
  let b := (abc / 10) % 10
  let c := abc % 10
  c * 10 + b

theorem three_digit_number_proof (abc : ℕ) :
  abc ≥ 100 ∧ abc < 1000 ∧
  is_geometric_progression (abc / 100) ((abc / 10) % 10) (abc % 10) ∧
  swap_hundreds_units abc = abc - 594 ∧
  swap_last_two_digits (last_two_digits abc) = last_two_digits abc - 18 →
  abc = 842 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_proof_l4107_410750


namespace NUMINAMATH_CALUDE_trigonometric_equality_l4107_410710

theorem trigonometric_equality (α β : ℝ) :
  (Real.cos α)^6 / (Real.cos β)^3 + (Real.sin α)^6 / (Real.sin β)^3 = 2 →
  (Real.sin β)^6 / (Real.sin α)^3 + (Real.cos β)^6 / (Real.cos α)^3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l4107_410710


namespace NUMINAMATH_CALUDE_smallest_gcd_l4107_410784

theorem smallest_gcd (p q r : ℕ+) (h1 : Nat.gcd p q = 294) (h2 : Nat.gcd p r = 847) :
  ∃ (q' r' : ℕ+), Nat.gcd q' r' = 49 ∧ 
    ∀ (q'' r'' : ℕ+), Nat.gcd p q'' = 294 → Nat.gcd p r'' = 847 → 
      Nat.gcd q'' r'' ≥ 49 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_l4107_410784


namespace NUMINAMATH_CALUDE_perimeter_difference_l4107_410732

/-- Represents a figure made of unit squares -/
structure UnitSquareFigure where
  perimeter : ℕ

/-- The first figure in the problem -/
def figure1 : UnitSquareFigure :=
  { perimeter := 24 }

/-- The second figure in the problem -/
def figure2 : UnitSquareFigure :=
  { perimeter := 33 }

/-- The theorem stating the difference between the perimeters of the two figures -/
theorem perimeter_difference :
  (figure2.perimeter - figure1.perimeter : ℤ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_difference_l4107_410732


namespace NUMINAMATH_CALUDE_super_eighteen_total_games_l4107_410757

/-- Calculates the total number of games in the Super Eighteen Football League -/
def super_eighteen_games (num_divisions : ℕ) (teams_per_division : ℕ) : ℕ :=
  let intra_division_games := num_divisions * teams_per_division * (teams_per_division - 1)
  let inter_division_games := num_divisions * teams_per_division * teams_per_division
  intra_division_games + inter_division_games

/-- Theorem stating that the Super Eighteen Football League schedules 450 games -/
theorem super_eighteen_total_games :
  super_eighteen_games 2 9 = 450 := by
  sorry

end NUMINAMATH_CALUDE_super_eighteen_total_games_l4107_410757


namespace NUMINAMATH_CALUDE_point_arrangement_theorem_l4107_410746

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- A set of n points in a plane satisfying the given condition -/
structure PointSet where
  n : ℕ
  points : Fin n → Point
  angle_condition : ∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k →
    (angle (points i) (points j) (points k) > 120) ∨
    (angle (points j) (points k) (points i) > 120) ∨
    (angle (points k) (points i) (points j) > 120)

/-- The main theorem -/
theorem point_arrangement_theorem (ps : PointSet) :
  ∃ (σ : Fin ps.n ↪ Fin ps.n),
    ∀ (i j k : Fin ps.n), i < j → j < k →
      angle (ps.points (σ i)) (ps.points (σ j)) (ps.points (σ k)) > 120 := by sorry

end NUMINAMATH_CALUDE_point_arrangement_theorem_l4107_410746


namespace NUMINAMATH_CALUDE_ticket_sales_total_l4107_410749

/-- Calculates the total amount collected from ticket sales given the ticket prices, total tickets sold, and number of children attending. -/
def total_amount_collected (child_price adult_price total_tickets children_count : ℕ) : ℕ :=
  let adult_count := total_tickets - children_count
  child_price * children_count + adult_price * adult_count

/-- Theorem stating that the total amount collected from ticket sales is $1875 given the specified conditions. -/
theorem ticket_sales_total :
  total_amount_collected 6 9 225 50 = 1875 := by
  sorry

end NUMINAMATH_CALUDE_ticket_sales_total_l4107_410749


namespace NUMINAMATH_CALUDE_sector_area_proof_l4107_410742

-- Define the given conditions
def circle_arc_length : ℝ := 2
def central_angle : ℝ := 2

-- Define the theorem
theorem sector_area_proof :
  let radius := circle_arc_length / central_angle
  let sector_area := (1 / 2) * radius^2 * central_angle
  sector_area = 1 := by sorry

end NUMINAMATH_CALUDE_sector_area_proof_l4107_410742


namespace NUMINAMATH_CALUDE_vector_decomposition_l4107_410748

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![3, 1, 3]
def p : Fin 3 → ℝ := ![2, 1, 0]
def q : Fin 3 → ℝ := ![1, 0, 1]
def r : Fin 3 → ℝ := ![4, 2, 1]

/-- The decomposition of x in terms of p, q, and r -/
theorem vector_decomposition : x = (-3 : ℝ) • p + q + (2 : ℝ) • r := by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l4107_410748


namespace NUMINAMATH_CALUDE_curve_C_properties_l4107_410735

-- Define the curve C
def C (t : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / (4 - t) + p.2^2 / (t - 1) = 1}

-- Define what it means for C to be a hyperbola
def is_hyperbola (t : ℝ) : Prop :=
  t < 1 ∨ t > 4

-- Define what it means for C to be an ellipse with foci on the X-axis
def is_ellipse_x_axis (t : ℝ) : Prop :=
  1 < t ∧ t < 5/2

theorem curve_C_properties (t : ℝ) :
  (is_hyperbola t ↔ ∃ (a b : ℝ), C t = {p : ℝ × ℝ | p.1^2/a^2 - p.2^2/b^2 = 1}) ∧
  (is_ellipse_x_axis t ↔ ∃ (a b : ℝ), a > b ∧ C t = {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1}) :=
by sorry

end NUMINAMATH_CALUDE_curve_C_properties_l4107_410735


namespace NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_5_l4107_410752

/-- The nth odd multiple of 5 -/
def nthOddMultipleOf5 (n : ℕ) : ℕ := 10 * n - 5

/-- Theorem stating that the 15th positive integer that is both odd and a multiple of 5 is 145 -/
theorem fifteenth_odd_multiple_of_5 : nthOddMultipleOf5 15 = 145 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_5_l4107_410752


namespace NUMINAMATH_CALUDE_x_range_for_P_in_fourth_quadrant_l4107_410776

-- Define the point P
def P (x : ℝ) : ℝ × ℝ := (2*x - 6, x - 5)

-- Define the condition for a point to be in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem x_range_for_P_in_fourth_quadrant :
  ∀ x : ℝ, in_fourth_quadrant (P x) ↔ 3 < x ∧ x < 5 := by sorry

end NUMINAMATH_CALUDE_x_range_for_P_in_fourth_quadrant_l4107_410776


namespace NUMINAMATH_CALUDE_total_germs_count_l4107_410767

/-- The number of petri dishes in the biology lab -/
def num_dishes : ℕ := 10800

/-- The number of germs in a single petri dish -/
def germs_per_dish : ℕ := 500

/-- The total number of germs in the biology lab -/
def total_germs : ℕ := num_dishes * germs_per_dish

/-- Theorem stating that the total number of germs is 5,400,000 -/
theorem total_germs_count : total_germs = 5400000 := by
  sorry

end NUMINAMATH_CALUDE_total_germs_count_l4107_410767


namespace NUMINAMATH_CALUDE_indefinite_integral_proof_l4107_410774

theorem indefinite_integral_proof (x : ℝ) (h : x > 0) : 
  (deriv (λ x => -3 * (1 + x^(4/3))^(5/3) / (4 * x^(4/3))) x) = 
    (1 + x^(4/3))^(2/3) / (x^2 * x^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_indefinite_integral_proof_l4107_410774


namespace NUMINAMATH_CALUDE_intersection_M_N_l4107_410780

def M : Set ℝ := {x | |x + 1| ≤ 1}
def N : Set ℝ := {-1, 0, 1}

theorem intersection_M_N : M ∩ N = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l4107_410780


namespace NUMINAMATH_CALUDE_average_annual_cost_reduction_l4107_410769

theorem average_annual_cost_reduction (total_reduction : Real) 
  (h : total_reduction = 0.36) : 
  ∃ x : Real, x > 0 ∧ x < 1 ∧ (1 - x)^2 = 1 - total_reduction :=
sorry

end NUMINAMATH_CALUDE_average_annual_cost_reduction_l4107_410769


namespace NUMINAMATH_CALUDE_dividend_calculation_l4107_410777

theorem dividend_calculation (divisor quotient remainder : ℕ) : 
  divisor = 18 → quotient = 9 → remainder = 5 → 
  divisor * quotient + remainder = 167 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l4107_410777


namespace NUMINAMATH_CALUDE_inequality_implication_l4107_410730

theorem inequality_implication (m n : ℝ) : m > 0 → n > m → 1/m - 1/n > 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l4107_410730


namespace NUMINAMATH_CALUDE_product_of_decimals_l4107_410727

theorem product_of_decimals : (0.5 : ℝ) * 0.8 = 0.40 := by sorry

end NUMINAMATH_CALUDE_product_of_decimals_l4107_410727


namespace NUMINAMATH_CALUDE_tom_golf_performance_l4107_410722

/-- Represents a round of golf --/
structure GolfRound where
  holes : ℕ
  averageStrokes : ℚ
  parValue : ℕ

/-- Calculates the total strokes for a round --/
def totalStrokes (round : GolfRound) : ℚ :=
  round.averageStrokes * round.holes

/-- Calculates the par for a round --/
def parForRound (round : GolfRound) : ℕ :=
  round.parValue * round.holes

theorem tom_golf_performance :
  let rounds : List GolfRound := [
    { holes := 9, averageStrokes := 4, parValue := 3 },
    { holes := 9, averageStrokes := 3.5, parValue := 3 },
    { holes := 9, averageStrokes := 5, parValue := 3 },
    { holes := 9, averageStrokes := 3, parValue := 3 },
    { holes := 9, averageStrokes := 4.5, parValue := 3 }
  ]
  let totalStrokesTaken := (rounds.map totalStrokes).sum
  let totalPar := (rounds.map parForRound).sum
  totalStrokesTaken - totalPar = 45 := by sorry

end NUMINAMATH_CALUDE_tom_golf_performance_l4107_410722


namespace NUMINAMATH_CALUDE_tetrahedron_volume_is_sqrt3_over_3_l4107_410726

-- Define the square ABCD
def square_side_length : ℝ := 2

-- Define point E as the midpoint of AB
def E_is_midpoint (A B E : ℝ × ℝ) : Prop :=
  E = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the folding along EC and ED
def folded_square (A B C D E : ℝ × ℝ) : Prop :=
  E_is_midpoint A B E ∧
  (A.1 - E.1)^2 + (A.2 - E.2)^2 = (B.1 - E.1)^2 + (B.2 - E.2)^2

-- Define the tetrahedron CDEA
structure Tetrahedron :=
  (C D E A : ℝ × ℝ)

-- Define the volume of a tetrahedron
def tetrahedron_volume (t : Tetrahedron) : ℝ := sorry

-- Theorem statement
theorem tetrahedron_volume_is_sqrt3_over_3 
  (A B C D E : ℝ × ℝ) 
  (h1 : (C.1 - A.1)^2 + (C.2 - A.2)^2 = square_side_length^2)
  (h2 : (D.1 - B.1)^2 + (D.2 - B.2)^2 = square_side_length^2)
  (h3 : folded_square A B C D E) :
  tetrahedron_volume {C := C, D := D, E := E, A := A} = Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_is_sqrt3_over_3_l4107_410726


namespace NUMINAMATH_CALUDE_frustum_max_volume_l4107_410717

/-- The maximum volume of a frustum within a sphere -/
theorem frustum_max_volume (r : ℝ) (r_top : ℝ) (r_bottom : ℝ) (h_r : r = 5) (h_top : r_top = 3) (h_bottom : r_bottom = 4) :
  ∃ v : ℝ, v = (259 / 3) * Real.pi ∧ 
  (∀ v' : ℝ, v' ≤ v ∧ 
    (∃ h : ℝ, v' = (1 / 3) * h * (r_top^2 * Real.pi + r_top * r_bottom * Real.pi + r_bottom^2 * Real.pi) ∧
              0 < h ∧ h ≤ 2 * (r^2 - r_top^2).sqrt)) :=
sorry

end NUMINAMATH_CALUDE_frustum_max_volume_l4107_410717


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l4107_410701

theorem perfect_square_trinomial (a b c : ℤ) :
  (∀ x : ℤ, ∃ y : ℤ, a * x^2 + b * x + c = y^2) →
  ∃ d e : ℤ, ∀ x : ℤ, a * x^2 + b * x + c = (d * x + e)^2 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l4107_410701


namespace NUMINAMATH_CALUDE_connie_initial_marbles_l4107_410759

/-- The number of marbles Connie initially had -/
def initial_marbles : ℕ := 241

/-- The number of marbles Connie bought -/
def bought_marbles : ℕ := 45

/-- The number of marbles Connie gave to Juan -/
def given_to_juan : ℕ := 73

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 70

theorem connie_initial_marbles :
  (initial_marbles + bought_marbles) / 2 - given_to_juan = marbles_left :=
by sorry

end NUMINAMATH_CALUDE_connie_initial_marbles_l4107_410759


namespace NUMINAMATH_CALUDE_river_depth_ratio_l4107_410761

/-- Given the depths of a river at different times, prove the ratio of depths -/
theorem river_depth_ratio 
  (depth_may : ℝ) 
  (increase_june : ℝ) 
  (depth_july : ℝ) 
  (h1 : depth_may = 5)
  (h2 : depth_july = 45)
  (h3 : depth_may + increase_june = depth_may + 10) :
  depth_july / (depth_may + increase_june) = 3 := by
  sorry

end NUMINAMATH_CALUDE_river_depth_ratio_l4107_410761


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l4107_410709

theorem square_perimeter_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) (h_ratio : a / b = 36 / 49) :
  (4 * Real.sqrt a) / (4 * Real.sqrt b) = 6 / 7 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l4107_410709


namespace NUMINAMATH_CALUDE_union_of_A_and_B_complement_of_intersection_A_and_B_l4107_410775

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x < 4}
def B : Set ℝ := {x | x - 3 ≥ 0}

-- Theorem for A ∪ B
theorem union_of_A_and_B : A ∪ B = {x | x ≥ -2} := by sorry

-- Theorem for Ā ∩ B
theorem complement_of_intersection_A_and_B : (A ∩ B)ᶜ = {x | x < 3 ∨ x ≥ 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_complement_of_intersection_A_and_B_l4107_410775


namespace NUMINAMATH_CALUDE_smartphone_sample_correct_l4107_410796

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  group_size : ℕ
  first_item : ℕ

/-- Conditions for the smartphone sampling problem -/
def smartphone_sample : SystematicSample where
  population_size := 160
  sample_size := 20
  group_size := 8
  first_item := 2  -- This is what we want to prove

theorem smartphone_sample_correct :
  let s := smartphone_sample
  s.population_size = 160 ∧
  s.sample_size = 20 ∧
  s.group_size = 8 ∧
  (s.first_item + 8 * 8 + s.first_item + 9 * 8 = 140) →
  s.first_item = 2 := by sorry

end NUMINAMATH_CALUDE_smartphone_sample_correct_l4107_410796


namespace NUMINAMATH_CALUDE_circle_tangent_area_zero_l4107_410708

-- Define the circle struct
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line struct
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

def CircleTangentToLine (c : Circle) (l : Line) : Prop := sorry

def CircleInternallyTangent (c1 c2 : Circle) : Prop := sorry

def PointBetween (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

def TriangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem circle_tangent_area_zero 
  (P Q R : Circle)
  (l : Line)
  (P' Q' R' : ℝ × ℝ)
  (h1 : P.radius = 2)
  (h2 : Q.radius = 3)
  (h3 : R.radius = 4)
  (h4 : CircleTangentToLine P l)
  (h5 : CircleTangentToLine Q l)
  (h6 : CircleTangentToLine R l)
  (h7 : P'.1 = P.center.1 ∧ P'.2 = P.center.2 + P.radius)
  (h8 : Q'.1 = Q.center.1 ∧ Q'.2 = Q.center.2 + Q.radius)
  (h9 : R'.1 = R.center.1 ∧ R'.2 = R.center.2 + R.radius)
  (h10 : PointBetween P' Q' R')
  (h11 : CircleInternallyTangent Q P)
  (h12 : CircleInternallyTangent Q R) :
  TriangleArea P.center Q.center R.center = 0 := by sorry

end NUMINAMATH_CALUDE_circle_tangent_area_zero_l4107_410708


namespace NUMINAMATH_CALUDE_water_tower_capacity_l4107_410781

/-- The capacity of a water tower serving four neighborhoods --/
theorem water_tower_capacity :
  let first_neighborhood : ℕ := 150
  let second_neighborhood : ℕ := 2 * first_neighborhood
  let third_neighborhood : ℕ := second_neighborhood + 100
  let fourth_neighborhood : ℕ := 350
  first_neighborhood + second_neighborhood + third_neighborhood + fourth_neighborhood = 1200 :=
by sorry

end NUMINAMATH_CALUDE_water_tower_capacity_l4107_410781


namespace NUMINAMATH_CALUDE_six_people_arrangement_l4107_410798

/-- The number of arrangements of six people in a row,
    where A and B must be adjacent with B to the left of A -/
def arrangements_count : ℕ := 120

/-- Theorem stating that the number of arrangements is 120 -/
theorem six_people_arrangement :
  arrangements_count = 120 := by
  sorry

end NUMINAMATH_CALUDE_six_people_arrangement_l4107_410798


namespace NUMINAMATH_CALUDE_exactly_two_talents_l4107_410724

theorem exactly_two_talents (total_students : ℕ) 
  (cannot_sing cannot_dance cannot_act no_talent : ℕ) :
  total_students = 120 →
  cannot_sing = 50 →
  cannot_dance = 75 →
  cannot_act = 45 →
  no_talent = 15 →
  (∃ (two_talents : ℕ), two_talents = 70 ∧ 
    two_talents = total_students - 
      (cannot_sing + cannot_dance + cannot_act - 2 * no_talent)) :=
by sorry

end NUMINAMATH_CALUDE_exactly_two_talents_l4107_410724


namespace NUMINAMATH_CALUDE_function_properties_l4107_410731

noncomputable section

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * Real.exp x - x - 1

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the range of y
def y_range (x₁ x₂ : ℝ) (m : ℝ) : ℝ := 
  (Real.exp x₂ - Real.exp x₁) * ((Real.exp x₂ + Real.exp x₁)⁻¹ - m)

-- Theorem statement
theorem function_properties (m : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ f m x₁ = 0 ∧ f m x₂ = 0) →
  (∀ (x : ℝ), f m x > 0) →
  (f m 0 = 1) →
  (tangent_line 0 1) ∧
  (∀ (y : ℝ), y < 0 → ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ y = y_range x₁ x₂ m) ∧
  (1 < m ∧ m < Real.exp 1 → Real.exp (m - 1) < m ^ (Real.exp 1 - 1)) ∧
  (m = Real.exp 1 → Real.exp (m - 1) = m ^ (Real.exp 1 - 1)) ∧
  (m > Real.exp 1 → Real.exp (m - 1) > m ^ (Real.exp 1 - 1)) := by
  sorry

end

end NUMINAMATH_CALUDE_function_properties_l4107_410731


namespace NUMINAMATH_CALUDE_marble_doubling_l4107_410728

theorem marble_doubling (k : ℕ) : (∀ n : ℕ, n < k → 5 * 2^n ≤ 200) ∧ 5 * 2^k > 200 ↔ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_marble_doubling_l4107_410728


namespace NUMINAMATH_CALUDE_ordering_abc_l4107_410733

theorem ordering_abc : 
  let a : ℝ := 0.1 * Real.exp 0.1
  let b : ℝ := 1 / 9
  let c : ℝ := -Real.log 0.9
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_ordering_abc_l4107_410733


namespace NUMINAMATH_CALUDE_curve_self_intersection_l4107_410712

/-- A curve defined by parametric equations x = t^2 + 1 and y = t^4 - 9t^2 + 6 -/
def curve (t : ℝ) : ℝ × ℝ :=
  (t^2 + 1, t^4 - 9*t^2 + 6)

/-- The theorem stating that the curve crosses itself at (10, 6) -/
theorem curve_self_intersection :
  ∃ (t1 t2 : ℝ), t1 ≠ t2 ∧ curve t1 = curve t2 ∧ curve t1 = (10, 6) := by
  sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l4107_410712


namespace NUMINAMATH_CALUDE_unread_books_l4107_410700

theorem unread_books (total : ℕ) (read : ℕ) (h1 : total = 21) (h2 : read = 13) :
  total - read = 8 := by
  sorry

end NUMINAMATH_CALUDE_unread_books_l4107_410700


namespace NUMINAMATH_CALUDE_jasmine_purchase_cost_l4107_410773

/-- The cost of Jasmine's purchase -/
def total_cost (coffee_pounds : ℕ) (milk_gallons : ℕ) (coffee_price : ℚ) (milk_price : ℚ) : ℚ :=
  coffee_pounds * coffee_price + milk_gallons * milk_price

/-- Proof that Jasmine's purchase costs $17 -/
theorem jasmine_purchase_cost :
  total_cost 4 2 (5/2) (7/2) = 17 := by
  sorry

end NUMINAMATH_CALUDE_jasmine_purchase_cost_l4107_410773


namespace NUMINAMATH_CALUDE_min_abs_z_plus_2i_l4107_410723

theorem min_abs_z_plus_2i (z : ℂ) (h : Complex.abs (z^2 - 3) = Complex.abs (z * (z - 3*I))) :
  Complex.abs (z + 2*I) ≥ (7 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_abs_z_plus_2i_l4107_410723


namespace NUMINAMATH_CALUDE_fox_alice_numbers_l4107_410760

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def divisible_by_at_least_three (n : ℕ) : Prop :=
  (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 4 = 0) ∨
  (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0) ∨
  (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 6 = 0) ∨
  (n % 2 = 0 ∧ n % 4 = 0 ∧ n % 5 = 0) ∨
  (n % 2 = 0 ∧ n % 4 = 0 ∧ n % 6 = 0) ∨
  (n % 2 = 0 ∧ n % 5 = 0 ∧ n % 6 = 0) ∨
  (n % 3 = 0 ∧ n % 4 = 0 ∧ n % 5 = 0) ∨
  (n % 3 = 0 ∧ n % 4 = 0 ∧ n % 6 = 0) ∨
  (n % 3 = 0 ∧ n % 5 = 0 ∧ n % 6 = 0) ∨
  (n % 4 = 0 ∧ n % 5 = 0 ∧ n % 6 = 0)

def not_divisible_by_exactly_two (n : ℕ) : Prop :=
  ¬((n % 2 ≠ 0 ∧ n % 3 ≠ 0 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ n % 6 = 0) ∨
    (n % 2 ≠ 0 ∧ n % 3 = 0 ∧ n % 4 ≠ 0 ∧ n % 5 = 0 ∧ n % 6 = 0) ∨
    (n % 2 ≠ 0 ∧ n % 3 = 0 ∧ n % 4 = 0 ∧ n % 5 ≠ 0 ∧ n % 6 = 0) ∨
    (n % 2 ≠ 0 ∧ n % 3 = 0 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ n % 6 ≠ 0) ∨
    (n % 2 = 0 ∧ n % 3 ≠ 0 ∧ n % 4 ≠ 0 ∧ n % 5 = 0 ∧ n % 6 = 0) ∨
    (n % 2 = 0 ∧ n % 3 ≠ 0 ∧ n % 4 = 0 ∧ n % 5 ≠ 0 ∧ n % 6 = 0) ∨
    (n % 2 = 0 ∧ n % 3 ≠ 0 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ n % 6 ≠ 0) ∨
    (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 4 ≠ 0 ∧ n % 5 ≠ 0 ∧ n % 6 = 0) ∨
    (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 4 ≠ 0 ∧ n % 5 = 0 ∧ n % 6 ≠ 0) ∨
    (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 4 = 0 ∧ n % 5 ≠ 0 ∧ n % 6 ≠ 0))

theorem fox_alice_numbers :
  ∃! (s : Finset ℕ), ∀ n, n ∈ s ↔ 
    is_two_digit n ∧ 
    divisible_by_at_least_three n ∧ 
    not_divisible_by_exactly_two n ∧
    s.card = 8 := by sorry

end NUMINAMATH_CALUDE_fox_alice_numbers_l4107_410760


namespace NUMINAMATH_CALUDE_min_largest_median_l4107_410703

/-- Represents a 5 × 18 rectangle filled with numbers from 1 to 90 -/
def Rectangle := Fin 5 → Fin 18 → Fin 90

/-- The median of a column in the rectangle -/
def columnMedian (rect : Rectangle) (col : Fin 18) : Fin 90 :=
  sorry

/-- The largest median among all columns -/
def largestMedian (rect : Rectangle) : Fin 90 :=
  sorry

/-- Theorem stating the minimum possible value for the largest median -/
theorem min_largest_median :
  ∃ (rect : Rectangle), largestMedian rect = 54 ∧
  ∀ (rect' : Rectangle), largestMedian rect' ≥ 54 :=
sorry

end NUMINAMATH_CALUDE_min_largest_median_l4107_410703


namespace NUMINAMATH_CALUDE_both_tea_probability_l4107_410783

-- Define the setup
def total_people : ℕ := 6
def tables : ℕ := 3
def people_per_table : ℕ := 2
def coffee_drinkers : ℕ := 3
def tea_drinkers : ℕ := 3

-- Define the probability function
def probability_both_tea : ℚ := 0.6

-- Theorem statement
theorem both_tea_probability :
  probability_both_tea = 0.6 :=
sorry

end NUMINAMATH_CALUDE_both_tea_probability_l4107_410783


namespace NUMINAMATH_CALUDE_cubic_root_reciprocal_sum_l4107_410713

theorem cubic_root_reciprocal_sum (a b c d : ℝ) (p q r : ℝ) : 
  a ≠ 0 → d ≠ 0 →
  (a * p^3 + b * p^2 + c * p + d = 0) →
  (a * q^3 + b * q^2 + c * q + d = 0) →
  (a * r^3 + b * r^2 + c * r + d = 0) →
  (1 / p^2 + 1 / q^2 + 1 / r^2) = (c^2 - 2 * b * d) / d^2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_reciprocal_sum_l4107_410713


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l4107_410766

theorem unique_three_digit_number :
  ∃! n : ℕ,
    100 ≤ n ∧ n < 1000 ∧
    (n % 100 % 10 = 3 * (n / 100)) ∧
    (n % 5 = 4) ∧
    (n % 11 = 3) ∧
    n = 359 := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l4107_410766


namespace NUMINAMATH_CALUDE_percentage_problem_l4107_410792

theorem percentage_problem (x : ℝ) (h : 0.2 * x = 60) : 0.8 * x = 240 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l4107_410792


namespace NUMINAMATH_CALUDE_expected_points_100_games_prob_specific_envelope_l4107_410705

/- Define the game parameters -/
def num_envelopes : ℕ := 13
def win_points : ℕ := 6

/- Define the probability of winning a single question -/
def win_prob : ℚ := 1/2

/- Define the expected number of envelopes played in a single game -/
noncomputable def expected_envelopes_per_game : ℝ := 12

/- Theorem for the expected points over 100 games -/
theorem expected_points_100_games :
  ∃ (expected_points : ℕ), expected_points = 465 := by sorry

/- Theorem for the probability of choosing a specific envelope -/
theorem prob_specific_envelope :
  ∃ (prob : ℚ), prob = 12/13 := by sorry

end NUMINAMATH_CALUDE_expected_points_100_games_prob_specific_envelope_l4107_410705


namespace NUMINAMATH_CALUDE_subtraction_difference_l4107_410744

theorem subtraction_difference (original : ℝ) (percentage : ℝ) (flat_amount : ℝ) : 
  original = 200 → percentage = 25 → flat_amount = 25 →
  (original - flat_amount) - (original - percentage / 100 * original) = 25 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_difference_l4107_410744


namespace NUMINAMATH_CALUDE_largest_fraction_l4107_410763

theorem largest_fraction : 
  let f1 := 8 / 15
  let f2 := 5 / 11
  let f3 := 19 / 37
  let f4 := 101 / 199
  let f5 := 153 / 305
  (f1 > f2 ∧ f1 > f3 ∧ f1 > f4 ∧ f1 > f5) := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l4107_410763


namespace NUMINAMATH_CALUDE_solve_for_k_l4107_410793

/-- The function f(x) = 4x³ - 3x² + 2x + 5 -/
def f (x : ℝ) : ℝ := 4 * x^3 - 3 * x^2 + 2 * x + 5

/-- The function g(x) = x³ - (k+1)x² - 7x - 8 -/
def g (k x : ℝ) : ℝ := x^3 - (k + 1) * x^2 - 7 * x - 8

/-- If f(5) - g(5) = 24, then k = -16.36 -/
theorem solve_for_k : ∃ k : ℝ, f 5 - g k 5 = 24 ∧ k = -16.36 := by sorry

end NUMINAMATH_CALUDE_solve_for_k_l4107_410793


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l4107_410758

theorem cos_2alpha_value (α : Real) (h : Real.sin (α + 3 * Real.pi / 2) = Real.sqrt 3 / 3) :
  Real.cos (2 * α) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l4107_410758


namespace NUMINAMATH_CALUDE_helen_cookies_l4107_410799

/-- The number of chocolate chip cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 527

/-- The number of chocolate chip cookies Helen baked this morning -/
def cookies_today : ℕ := 554

/-- The total number of chocolate chip cookies Helen baked -/
def total_cookies : ℕ := cookies_yesterday + cookies_today

theorem helen_cookies : total_cookies = 1081 := by
  sorry

end NUMINAMATH_CALUDE_helen_cookies_l4107_410799


namespace NUMINAMATH_CALUDE_all_roots_nonzero_l4107_410790

theorem all_roots_nonzero :
  (∀ x : ℝ, 4 * x^2 - 6 = 34 → x ≠ 0) ∧
  (∀ x : ℝ, (3 * x - 1)^2 = (x + 2)^2 → x ≠ 0) ∧
  (∀ x : ℝ, (x^2 - 4 : ℝ) = (x + 3 : ℝ) → x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_all_roots_nonzero_l4107_410790


namespace NUMINAMATH_CALUDE_investment_triple_period_l4107_410754

/-- The annual interest rate as a real number -/
def r : ℝ := 0.341

/-- The condition for the investment to more than triple -/
def triple_condition (t : ℝ) : Prop := (1 + r) ^ t > 3

/-- The smallest investment period in years -/
def smallest_period : ℕ := 4

theorem investment_triple_period :
  (∀ t : ℝ, t < smallest_period → ¬(triple_condition t)) ∧
  (triple_condition (smallest_period : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_investment_triple_period_l4107_410754


namespace NUMINAMATH_CALUDE_range_of_w_l4107_410720

theorem range_of_w (x y : ℝ) (h : 2*x^2 + 4*x*y + 2*y^2 + x^2*y^2 = 9) :
  let w := 2*Real.sqrt 2*(x + y) + x*y
  ∃ (a b : ℝ), a = -3*Real.sqrt 5 ∧ b = Real.sqrt 5 ∧ 
    (∀ w', w' = w → a ≤ w' ∧ w' ≤ b) ∧
    (∃ w₁ w₂, w₁ = w ∧ w₂ = w ∧ w₁ = a ∧ w₂ = b) :=
by sorry


end NUMINAMATH_CALUDE_range_of_w_l4107_410720


namespace NUMINAMATH_CALUDE_triangle_area_triple_altitude_l4107_410741

theorem triangle_area_triple_altitude (b h : ℝ) (h_pos : 0 < h) :
  let A := (1/2) * b * h
  let A' := (1/2) * b * (3*h)
  A' = 3 * A := by sorry

end NUMINAMATH_CALUDE_triangle_area_triple_altitude_l4107_410741


namespace NUMINAMATH_CALUDE_set_c_forms_triangle_set_a_not_triangle_set_b_not_triangle_set_d_not_triangle_triangle_formation_result_l4107_410702

/-- A function that checks if three line segments can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that the set (4, 5, 7) can form a triangle --/
theorem set_c_forms_triangle : can_form_triangle 4 5 7 := by sorry

/-- Theorem stating that the set (1, 3, 4) cannot form a triangle --/
theorem set_a_not_triangle : ¬ can_form_triangle 1 3 4 := by sorry

/-- Theorem stating that the set (2, 2, 7) cannot form a triangle --/
theorem set_b_not_triangle : ¬ can_form_triangle 2 2 7 := by sorry

/-- Theorem stating that the set (3, 3, 6) cannot form a triangle --/
theorem set_d_not_triangle : ¬ can_form_triangle 3 3 6 := by sorry

/-- Main theorem combining all results --/
theorem triangle_formation_result :
  can_form_triangle 4 5 7 ∧
  ¬ can_form_triangle 1 3 4 ∧
  ¬ can_form_triangle 2 2 7 ∧
  ¬ can_form_triangle 3 3 6 := by sorry

end NUMINAMATH_CALUDE_set_c_forms_triangle_set_a_not_triangle_set_b_not_triangle_set_d_not_triangle_triangle_formation_result_l4107_410702


namespace NUMINAMATH_CALUDE_exactly_three_solutions_l4107_410782

/-- The function that we're interested in -/
def f (m : ℕ+) : ℚ :=
  1260 / ((m : ℚ)^2 - 6)

/-- Predicate for f(m) being a positive integer -/
def is_positive_integer (m : ℕ+) : Prop :=
  ∃ (k : ℕ+), f m = k

/-- The main theorem -/
theorem exactly_three_solutions :
  ∃! (s : Finset ℕ+), s.card = 3 ∧ ∀ m : ℕ+, m ∈ s ↔ is_positive_integer m :=
sorry

end NUMINAMATH_CALUDE_exactly_three_solutions_l4107_410782


namespace NUMINAMATH_CALUDE_inscribable_polygons_l4107_410704

/-- The number of evenly spaced holes on the circumference of the circle -/
def num_holes : ℕ := 24

/-- A function that determines if a regular polygon with 'n' sides can be inscribed in the circle -/
def can_inscribe (n : ℕ) : Prop :=
  n ≥ 3 ∧ num_holes % n = 0

/-- The set of numbers of sides for regular polygons that can be inscribed in the circle -/
def valid_polygons : Set ℕ := {n | can_inscribe n}

/-- Theorem stating that the only valid numbers of sides for inscribable regular polygons are 3, 4, 6, 8, 12, and 24 -/
theorem inscribable_polygons :
  valid_polygons = {3, 4, 6, 8, 12, 24} :=
sorry

end NUMINAMATH_CALUDE_inscribable_polygons_l4107_410704


namespace NUMINAMATH_CALUDE_friend_spent_more_l4107_410721

theorem friend_spent_more (total : ℕ) (friend_spent : ℕ) (you_spent : ℕ) : 
  total = 11 → friend_spent = 7 → total = friend_spent + you_spent → friend_spent > you_spent →
  friend_spent - you_spent = 3 := by
sorry

end NUMINAMATH_CALUDE_friend_spent_more_l4107_410721


namespace NUMINAMATH_CALUDE_doubled_average_l4107_410768

theorem doubled_average (n : ℕ) (original_average : ℝ) (h1 : n = 30) (h2 : original_average = 45) :
  let new_average := 2 * original_average
  new_average = 90 := by
sorry

end NUMINAMATH_CALUDE_doubled_average_l4107_410768


namespace NUMINAMATH_CALUDE_max_value_expression_l4107_410794

theorem max_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z)^2 / (x^2 + y^2 + z^2) ≤ 3 ∧
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ (a + b + c)^2 / (a^2 + b^2 + c^2) = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l4107_410794


namespace NUMINAMATH_CALUDE_sheridan_cats_goal_l4107_410740

/-- The number of cats Mrs. Sheridan currently has -/
def current_cats : ℕ := 11

/-- The number of additional cats Mrs. Sheridan needs -/
def additional_cats : ℕ := 32

/-- The total number of cats Mrs. Sheridan wants to have -/
def total_cats : ℕ := current_cats + additional_cats

theorem sheridan_cats_goal : total_cats = 43 := by
  sorry

end NUMINAMATH_CALUDE_sheridan_cats_goal_l4107_410740


namespace NUMINAMATH_CALUDE_smallest_integer_in_set_l4107_410770

theorem smallest_integer_in_set (n : ℤ) : 
  (n + 6 < 3 * ((n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6)) / 7)) →
  n ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_set_l4107_410770


namespace NUMINAMATH_CALUDE_cubic_root_sum_l4107_410789

theorem cubic_root_sum (r s t : ℝ) : 
  r^3 - 15*r^2 + 13*r - 6 = 0 ∧ 
  s^3 - 15*s^2 + 13*s - 6 = 0 ∧ 
  t^3 - 15*t^2 + 13*t - 6 = 0 →
  r / (1/r + s*t) + s / (1/s + t*r) + t / (1/t + r*s) = 199/7 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l4107_410789


namespace NUMINAMATH_CALUDE_complex_modulus_l4107_410736

theorem complex_modulus (z : ℂ) (h : z * (2 - Complex.I) = 1 + Complex.I) : Complex.abs z = Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l4107_410736


namespace NUMINAMATH_CALUDE_function_properties_l4107_410743

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem function_properties (a b : ℝ) 
  (h : (a - 1)^2 - 4*b < 0) : 
  (∀ x, f a b x > x) ∧ 
  (∀ x, f a b (f a b x) > x) ∧ 
  (a + b > 0) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l4107_410743


namespace NUMINAMATH_CALUDE_square_difference_of_sum_and_diff_l4107_410714

theorem square_difference_of_sum_and_diff (a b : ℕ+) 
  (h_sum : a + b = 60) 
  (h_diff : a - b = 14) : 
  a^2 - b^2 = 840 := by
sorry

end NUMINAMATH_CALUDE_square_difference_of_sum_and_diff_l4107_410714


namespace NUMINAMATH_CALUDE_cistern_fill_time_l4107_410771

/-- The time it takes to fill the cistern with both taps open -/
def both_taps_time : ℚ := 28 / 3

/-- The time it takes to empty the cistern with the second tap -/
def empty_time : ℚ := 7

/-- The time it takes to fill the cistern with the first tap -/
def fill_time : ℚ := 4

theorem cistern_fill_time :
  (1 / fill_time - 1 / empty_time) = 1 / both_taps_time :=
by sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l4107_410771


namespace NUMINAMATH_CALUDE_inequality_range_l4107_410797

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (Real.sin x)^2 + a * Real.cos x - a^2 ≤ 1 + Real.cos x) ↔ 
  (a ≤ -1 ∨ a ≥ 1/3) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l4107_410797


namespace NUMINAMATH_CALUDE_total_pies_baked_l4107_410729

/-- The number of pies Eddie can bake in a day -/
def eddie_pies_per_day : ℕ := 3

/-- The number of pies Eddie's sister can bake in a day -/
def sister_pies_per_day : ℕ := 6

/-- The number of pies Eddie's mother can bake in a day -/
def mother_pies_per_day : ℕ := 8

/-- The number of days they will bake pies -/
def days_baking : ℕ := 7

/-- Theorem stating the total number of pies baked in 7 days -/
theorem total_pies_baked : 
  (eddie_pies_per_day * days_baking) + 
  (sister_pies_per_day * days_baking) + 
  (mother_pies_per_day * days_baking) = 119 := by
sorry

end NUMINAMATH_CALUDE_total_pies_baked_l4107_410729


namespace NUMINAMATH_CALUDE_stratified_sampling_proportion_l4107_410795

theorem stratified_sampling_proportion (total_population : ℕ) (stratum_a : ℕ) (stratum_b : ℕ) (sample_size : ℕ) :
  total_population = stratum_a + stratum_b →
  total_population = 120 →
  stratum_a = 20 →
  stratum_b = 100 →
  sample_size = 12 →
  (sample_size * stratum_a) / total_population = 2 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_proportion_l4107_410795


namespace NUMINAMATH_CALUDE_inequality_proof_l4107_410756

theorem inequality_proof (x : ℝ) (n : ℕ) (h : x > 0) :
  1 + x^(n+1) ≥ (2*x)^n / (1+x)^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4107_410756


namespace NUMINAMATH_CALUDE_wendy_second_level_treasures_l4107_410765

def points_per_treasure : ℕ := 5
def treasures_first_level : ℕ := 4
def total_score : ℕ := 35

theorem wendy_second_level_treasures :
  (total_score - points_per_treasure * treasures_first_level) / points_per_treasure = 3 := by
  sorry

end NUMINAMATH_CALUDE_wendy_second_level_treasures_l4107_410765


namespace NUMINAMATH_CALUDE_dividend_percentage_calculation_l4107_410753

theorem dividend_percentage_calculation (face_value : ℝ) (purchase_price : ℝ) (return_on_investment : ℝ) :
  face_value = 40 →
  purchase_price = 20 →
  return_on_investment = 0.25 →
  (purchase_price * return_on_investment) / face_value = 0.125 :=
by sorry

end NUMINAMATH_CALUDE_dividend_percentage_calculation_l4107_410753


namespace NUMINAMATH_CALUDE_perfect_square_proof_l4107_410715

theorem perfect_square_proof (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  ((2 * l - n - k) * (2 * l - n + k)) / 2 = (l - n)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_proof_l4107_410715
