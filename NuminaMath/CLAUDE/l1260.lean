import Mathlib

namespace NUMINAMATH_CALUDE_bisection_diagram_type_l1260_126056

/-- The function we're finding the root for -/
def f (x : ℝ) : ℝ := x^2 - 2

/-- Represents the types of diagrams -/
inductive DiagramType
| ProcessFlowchart
| KnowledgeStructureDiagram
| ProgramFlowchart
| OrganizationalStructureDiagram

/-- Properties of the bisection method -/
structure BisectionMethod where
  continuous : ∀ a b, a < b → ContinuousOn f (Set.Icc a b)
  oppositeSign : ∃ a b, a < b ∧ f a * f b < 0
  iterative : ∀ a b, a < b → ∃ c, a < c ∧ c < b ∧ f c = (f a + f b) / 2

/-- The theorem stating that the bisection method for x^2 - 2 = 0 is represented by a Program Flowchart -/
theorem bisection_diagram_type (bm : BisectionMethod) : 
  ∃ d : DiagramType, d = DiagramType.ProgramFlowchart :=
sorry

end NUMINAMATH_CALUDE_bisection_diagram_type_l1260_126056


namespace NUMINAMATH_CALUDE_circle_area_difference_l1260_126036

theorem circle_area_difference : 
  let r1 : ℝ := 25
  let d2 : ℝ := 15
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1 ^ 2
  let area2 : ℝ := π * r2 ^ 2
  area1 - area2 = 568.75 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l1260_126036


namespace NUMINAMATH_CALUDE_complex_multiplication_l1260_126066

theorem complex_multiplication (i : ℂ) : i^2 = -1 → (1 - i)^2 * i = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1260_126066


namespace NUMINAMATH_CALUDE_darcy_folded_shirts_darcy_problem_l1260_126042

theorem darcy_folded_shirts (total_shirts : ℕ) (total_shorts : ℕ) (folded_shorts : ℕ) (remaining_to_fold : ℕ) : ℕ :=
  let total_clothing := total_shirts + total_shorts
  let folded_clothing := total_clothing - folded_shorts - remaining_to_fold
  let folded_shirts := folded_clothing - folded_shorts
  folded_shirts

theorem darcy_problem :
  darcy_folded_shirts 20 8 5 11 = 7 := by
  sorry

end NUMINAMATH_CALUDE_darcy_folded_shirts_darcy_problem_l1260_126042


namespace NUMINAMATH_CALUDE_jeans_pricing_l1260_126054

theorem jeans_pricing (C : ℝ) (h1 : C > 0) : 
  let retailer_price := 1.40 * C
  let customer_price := 1.96 * C
  (customer_price - retailer_price) / retailer_price = 0.40 := by sorry

end NUMINAMATH_CALUDE_jeans_pricing_l1260_126054


namespace NUMINAMATH_CALUDE_sum_of_fraction_and_decimal_l1260_126084

theorem sum_of_fraction_and_decimal : (1 : ℚ) / 25 + (25 : ℚ) / 100 = (29 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fraction_and_decimal_l1260_126084


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l1260_126024

theorem ferris_wheel_capacity (total_people : ℕ) (num_seats : ℕ) (people_per_seat : ℕ) : 
  total_people = 18 → num_seats = 2 → people_per_seat = total_people / num_seats → people_per_seat = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l1260_126024


namespace NUMINAMATH_CALUDE_polynomial_equality_constants_l1260_126081

theorem polynomial_equality_constants (k1 k2 k3 : ℤ) : 
  (∀ x : ℝ, -x^4 - (k1 + 11)*x^3 - k2*x^2 - 8*x - k3 = -(x - 2)*(x^3 - 6*x^2 + 8*x - 4)) ↔ 
  (k1 = -19 ∧ k2 = 20 ∧ k3 = 8) := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_constants_l1260_126081


namespace NUMINAMATH_CALUDE_specific_figure_perimeter_l1260_126048

/-- Represents a figure composed of unit squares -/
structure UnitSquareFigure where
  rows : Nat
  columns : Nat
  extra_column : Nat

/-- Calculates the perimeter of a UnitSquareFigure -/
def perimeter (figure : UnitSquareFigure) : Nat :=
  sorry

/-- The specific figure described in the problem -/
def specific_figure : UnitSquareFigure :=
  { rows := 3, columns := 4, extra_column := 2 }

theorem specific_figure_perimeter :
  perimeter specific_figure = 13 := by sorry

end NUMINAMATH_CALUDE_specific_figure_perimeter_l1260_126048


namespace NUMINAMATH_CALUDE_min_force_to_submerge_cube_l1260_126085

/-- Minimum force required to submerge a cube -/
theorem min_force_to_submerge_cube (V : Real) (ρ_cube ρ_water g : Real) :
  V = 1e-5 →
  ρ_cube = 700 →
  ρ_water = 1000 →
  g = 10 →
  (ρ_water * V * g) - (ρ_cube * V * g) = 0.03 := by
  sorry

end NUMINAMATH_CALUDE_min_force_to_submerge_cube_l1260_126085


namespace NUMINAMATH_CALUDE_max_amount_C_is_correct_l1260_126020

/-- Represents the maximum amount of 11% saline solution (C) that can be used
    to prepare 100 kg of 7% saline solution, given 3% (A) and 8% (B) solutions
    are also available. -/
def maxAmountC : ℝ := 50

/-- The concentration of saline solution A -/
def concentrationA : ℝ := 0.03

/-- The concentration of saline solution B -/
def concentrationB : ℝ := 0.08

/-- The concentration of saline solution C -/
def concentrationC : ℝ := 0.11

/-- The target concentration of the final solution -/
def targetConcentration : ℝ := 0.07

/-- The total amount of the final solution -/
def totalAmount : ℝ := 100

theorem max_amount_C_is_correct :
  ∃ (y : ℝ),
    0 ≤ y ∧
    0 ≤ (totalAmount - maxAmountC - y) ∧
    concentrationC * maxAmountC + concentrationB * y +
      concentrationA * (totalAmount - maxAmountC - y) =
    targetConcentration * totalAmount ∧
    ∀ (x : ℝ),
      x > maxAmountC →
      ¬∃ (z : ℝ),
        0 ≤ z ∧
        0 ≤ (totalAmount - x - z) ∧
        concentrationC * x + concentrationB * z +
          concentrationA * (totalAmount - x - z) =
        targetConcentration * totalAmount :=
by sorry

end NUMINAMATH_CALUDE_max_amount_C_is_correct_l1260_126020


namespace NUMINAMATH_CALUDE_PQ_length_range_l1260_126086

/-- The circle C in the Cartesian coordinate system -/
def C : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - 3)^2 = 2}

/-- A point on the x-axis -/
def A : ℝ → ℝ × ℝ := λ x => (x, 0)

/-- The tangent points P and Q on the circle C -/
noncomputable def P (x : ℝ) : ℝ × ℝ := sorry
noncomputable def Q (x : ℝ) : ℝ × ℝ := sorry

/-- The length of segment PQ -/
noncomputable def PQ_length (x : ℝ) : ℝ :=
  Real.sqrt ((P x).1 - (Q x).1)^2 + ((P x).2 - (Q x).2)^2

/-- The theorem stating the range of PQ length -/
theorem PQ_length_range :
  ∀ x : ℝ, 2 * Real.sqrt 14 / 3 < PQ_length x ∧ PQ_length x < 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_PQ_length_range_l1260_126086


namespace NUMINAMATH_CALUDE_certain_number_value_l1260_126089

theorem certain_number_value (t b c : ℝ) (x : ℝ) :
  (t + b + c + 14 + x) / 5 = 12 →
  (t + b + c + 29) / 4 = 15 →
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_certain_number_value_l1260_126089


namespace NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l1260_126041

/-- The area of a circle with diameter endpoints C(5,9) and D(13,17) is 32π square units. -/
theorem circle_area_from_diameter_endpoints :
  let c : ℝ × ℝ := (5, 9)
  let d : ℝ × ℝ := (13, 17)
  let diameter_squared := (d.1 - c.1)^2 + (d.2 - c.2)^2
  let radius_squared := diameter_squared / 4
  π * radius_squared = 32 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l1260_126041


namespace NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l1260_126040

/-- Given a right triangle XYZ with coordinates X(0,0), Y(a,0), Z(a,a),
    where 'a' is a positive real number, and a circle with radius 'a'
    inscribed in the rectangle formed by extending sides XY and YZ,
    prove that the area of the rectangle is 4a² when the hypotenuse XZ = 2a. -/
theorem rectangle_area_with_inscribed_circle (a : ℝ) (ha : a > 0) :
  let X : ℝ × ℝ := (0, 0)
  let Y : ℝ × ℝ := (a, 0)
  let Z : ℝ × ℝ := (a, a)
  let hypotenuse := Real.sqrt ((Z.1 - X.1)^2 + (Z.2 - X.2)^2)
  hypotenuse = 2 * a →
  (2 * a) * (2 * a) = 4 * a^2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l1260_126040


namespace NUMINAMATH_CALUDE_abs_eq_neg_self_implies_nonpositive_l1260_126051

theorem abs_eq_neg_self_implies_nonpositive (a : ℝ) : |a| = -a → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_neg_self_implies_nonpositive_l1260_126051


namespace NUMINAMATH_CALUDE_nala_seashell_count_l1260_126096

/-- The number of seashells Nala found on Monday -/
def monday_shells : ℕ := 5

/-- The number of seashells Nala found on Tuesday -/
def tuesday_shells : ℕ := 7

/-- The number of seashells Nala discarded on Tuesday -/
def tuesday_discarded : ℕ := 3

/-- The number of seashells Nala found on Wednesday relative to Monday -/
def wednesday_multiplier : ℕ := 2

/-- The fraction of seashells Nala discarded on Wednesday -/
def wednesday_discard_fraction : ℚ := 1/2

/-- The number of seashells Nala found on Thursday relative to Tuesday -/
def thursday_multiplier : ℕ := 3

/-- The total number of unbroken seashells Nala has by the end of Thursday -/
def total_shells : ℕ := 35

theorem nala_seashell_count : 
  monday_shells + 
  (tuesday_shells - tuesday_discarded) + 
  (wednesday_multiplier * monday_shells - Nat.floor (↑(wednesday_multiplier * monday_shells) * wednesday_discard_fraction)) + 
  (thursday_multiplier * tuesday_shells) = total_shells := by
  sorry

end NUMINAMATH_CALUDE_nala_seashell_count_l1260_126096


namespace NUMINAMATH_CALUDE_patio_length_l1260_126073

theorem patio_length (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  width > 0 →
  length = 4 * width →
  perimeter = 2 * (length + width) →
  perimeter = 100 →
  length = 40 := by
sorry

end NUMINAMATH_CALUDE_patio_length_l1260_126073


namespace NUMINAMATH_CALUDE_parabola_max_value_l1260_126045

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

/-- Whether a parabola opens downwards -/
def opens_downwards (p : Parabola) : Prop := p.a < 0

/-- The maximum value of a parabola -/
def max_value (p : Parabola) : ℝ := sorry

theorem parabola_max_value (p : Parabola) 
  (h1 : vertex p = (-3, 2)) 
  (h2 : opens_downwards p) : 
  max_value p = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_max_value_l1260_126045


namespace NUMINAMATH_CALUDE_range_of_a_l1260_126074

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define set A
def A : Set ℝ := {x | f (x - 3) < 1}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | f (x - 2*a) < a^2}

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (A ∪ B a = B a) → (1 ≤ a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1260_126074


namespace NUMINAMATH_CALUDE_min_value_fraction_min_value_fraction_equality_l1260_126014

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x - 2*y + 3*z = 0) : 
  (y^2 / (x*z)) ≥ 3 := by
sorry

theorem min_value_fraction_equality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x - 2*y + 3*z = 0) : 
  (y^2 / (x*z) = 3) ↔ (x = 3*z) := by
sorry

end NUMINAMATH_CALUDE_min_value_fraction_min_value_fraction_equality_l1260_126014


namespace NUMINAMATH_CALUDE_remainder_problem_l1260_126011

theorem remainder_problem (k : ℕ) 
  (h1 : k > 0) 
  (h2 : k % 5 = 2) 
  (h3 : k < 41) 
  (h4 : k % 7 = 3) : 
  k % 6 = 5 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1260_126011


namespace NUMINAMATH_CALUDE_volunteer_distribution_theorem_l1260_126034

/-- The number of ways to distribute n people into two activities with capacity constraints -/
def distributeVolunteers (n : ℕ) (maxPerActivity : ℕ) : ℕ :=
  -- We don't implement the function here, just declare it
  sorry

/-- Theorem: The number of ways to distribute 6 people into two activities,
    where each activity can accommodate no more than 4 people, is equal to 50 -/
theorem volunteer_distribution_theorem :
  distributeVolunteers 6 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_distribution_theorem_l1260_126034


namespace NUMINAMATH_CALUDE_right_triangle_median_hypotenuse_l1260_126063

/-- 
A right triangle with hypotenuse length 6 has a median to the hypotenuse of length 3.
-/
theorem right_triangle_median_hypotenuse : 
  ∀ (a b c : ℝ), 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for right triangle
  c = 6 →           -- Hypotenuse length is 6
  ∃ (m : ℝ),        -- There exists a median m
    m^2 = (a^2 + b^2) / 4 ∧  -- Median formula
    m = 3 :=        -- Median length is 3
by sorry

end NUMINAMATH_CALUDE_right_triangle_median_hypotenuse_l1260_126063


namespace NUMINAMATH_CALUDE_paper_size_problem_l1260_126006

theorem paper_size_problem (L : ℝ) :
  (L > 0) →
  (2 * (L * 11) = 2 * (5.5 * 11) + 100) →
  L = 10 := by
sorry

end NUMINAMATH_CALUDE_paper_size_problem_l1260_126006


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1260_126021

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 19 * n ≡ 678 [ZMOD 11] ∧ ∀ m : ℕ, m > 0 ∧ m < n → ¬(19 * m ≡ 678 [ZMOD 11])) ↔ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1260_126021


namespace NUMINAMATH_CALUDE_total_hotdogs_by_wednesday_l1260_126065

def hotdog_sequence (n : ℕ) : ℕ := 10 + 2 * n

theorem total_hotdogs_by_wednesday :
  (hotdog_sequence 0) + (hotdog_sequence 1) + (hotdog_sequence 2) = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_hotdogs_by_wednesday_l1260_126065


namespace NUMINAMATH_CALUDE_girls_boys_acquaintance_l1260_126059

theorem girls_boys_acquaintance (n : ℕ) :
  n > 1 →
  (∃ (girls_know : Fin (n + 1) → Fin (n + 1)) (boys_know : Fin n → ℕ),
    Function.Injective girls_know ∧
    (∀ i : Fin n, boys_know i = (n + 1) / 2) ∧
    (∀ i : Fin (n + 1), girls_know i ≤ n)) →
  Odd n :=
by sorry

end NUMINAMATH_CALUDE_girls_boys_acquaintance_l1260_126059


namespace NUMINAMATH_CALUDE_equation_solution_l1260_126077

theorem equation_solution : ∃! x : ℝ, (1 / (x - 1) = 3 / (x - 3)) ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1260_126077


namespace NUMINAMATH_CALUDE_division_problem_l1260_126098

theorem division_problem (n : ℕ) : 
  n % 7 = 5 ∧ n / 7 = 12 → n / 8 = 11 := by sorry

end NUMINAMATH_CALUDE_division_problem_l1260_126098


namespace NUMINAMATH_CALUDE_not_divisible_by_qplus1_l1260_126008

theorem not_divisible_by_qplus1 (q : ℕ) (hodd : Odd q) (hq : q > 2) :
  ¬ (q + 1 ∣ (q + 1)^((q - 1)/2) + 2) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_qplus1_l1260_126008


namespace NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l1260_126078

theorem smallest_prime_after_six_nonprimes : 
  ∃ (n : ℕ), 
    (∀ k ∈ Finset.range 6, ¬ Nat.Prime (n + k + 1)) ∧ 
    Nat.Prime (n + 7) ∧
    (∀ m < n, ¬(∀ k ∈ Finset.range 6, ¬ Nat.Prime (m + k + 1)) ∨ ¬Nat.Prime (m + 7)) ∧
    n + 7 = 37 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l1260_126078


namespace NUMINAMATH_CALUDE_cubic_inequality_l1260_126080

theorem cubic_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^3 + b^3 + a + b ≥ 4*a*b := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l1260_126080


namespace NUMINAMATH_CALUDE_same_root_implies_a_value_l1260_126003

theorem same_root_implies_a_value (a : ℝ) : 
  (∃ x : ℝ, x - a = 0 ∧ x^2 + a*x - 2 = 0) → (a = 1 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_same_root_implies_a_value_l1260_126003


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l1260_126009

/-- The cost of pens and pencils -/
def CostProblem (pen_cost : ℚ) (pencil_cost : ℚ) : Prop :=
  -- Condition 1: The cost of 3 pens and 5 pencils is Rs. 260
  3 * pen_cost + 5 * pencil_cost = 260 ∧
  -- Condition 2: The cost ratio of one pen to one pencil is 5:1
  pen_cost = 5 * pencil_cost

/-- The cost of one dozen pens is Rs. 780 -/
theorem cost_of_dozen_pens 
  (pen_cost : ℚ) (pencil_cost : ℚ) 
  (h : CostProblem pen_cost pencil_cost) : 
  12 * pen_cost = 780 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l1260_126009


namespace NUMINAMATH_CALUDE_sine_bounds_l1260_126060

theorem sine_bounds (x : ℝ) (h : x ∈ Set.Icc 0 1) :
  (Real.sqrt 2 / 2) * x ≤ Real.sin x ∧ Real.sin x ≤ x := by
  sorry

end NUMINAMATH_CALUDE_sine_bounds_l1260_126060


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1260_126012

theorem inscribed_square_area (XY ZC : ℝ) (h1 : XY = 40) (h2 : ZC = 70) :
  let s := Real.sqrt (XY * ZC)
  s * s = 2800 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1260_126012


namespace NUMINAMATH_CALUDE_reflected_light_equation_l1260_126097

/-- The incident light line -/
def incident_line (x y : ℝ) : Prop := 2 * x - y + 6 = 0

/-- The reflection line -/
def reflection_line (x y : ℝ) : Prop := y = x

/-- The reflected light line -/
def reflected_line (x y : ℝ) : Prop := x + 2 * y + 18 = 0

/-- 
Given an incident light line 2x - y + 6 = 0 striking the line y = x, 
prove that the reflected light line has the equation x + 2y + 18 = 0.
-/
theorem reflected_light_equation :
  ∀ x y : ℝ, incident_line x y ∧ reflection_line x y → reflected_line x y :=
by sorry

end NUMINAMATH_CALUDE_reflected_light_equation_l1260_126097


namespace NUMINAMATH_CALUDE_probability_is_correct_l1260_126079

/-- Represents the total number of cards -/
def t : ℕ := 93

/-- Represents the number of cards with blue dinosaurs -/
def blue_dinosaurs : ℕ := 16

/-- Represents the number of cards with green robots -/
def green_robots : ℕ := 14

/-- Represents the number of cards with blue robots -/
def blue_robots : ℕ := 36

/-- Represents the number of cards with green dinosaurs -/
def green_dinosaurs : ℕ := t - (blue_dinosaurs + green_robots + blue_robots)

/-- The probability of choosing a card with either a green dinosaur or a blue robot -/
def probability : ℚ := (green_dinosaurs + blue_robots : ℚ) / t

theorem probability_is_correct : probability = 21 / 31 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_correct_l1260_126079


namespace NUMINAMATH_CALUDE_matrix_equation_satisfied_l1260_126025

/-- The matrix M that satisfies the given equation -/
def M : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 2]

/-- The right-hand side matrix of the equation -/
def RHS : Matrix (Fin 2) (Fin 2) ℝ := !![10, 20; 5, 10]

/-- Theorem stating that M satisfies the given matrix equation -/
theorem matrix_equation_satisfied :
  M^3 - 4 • M^2 + 5 • M = RHS := by sorry

end NUMINAMATH_CALUDE_matrix_equation_satisfied_l1260_126025


namespace NUMINAMATH_CALUDE_circle_center_correct_l1260_126069

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation, return its center -/
def findCircleCenter (eq : CircleEquation) : CircleCenter :=
  sorry

theorem circle_center_correct :
  let eq := CircleEquation.mk 1 (-4) 1 (-6) (-12)
  findCircleCenter eq = CircleCenter.mk 2 3 := by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l1260_126069


namespace NUMINAMATH_CALUDE_sequence_sum_equals_642_l1260_126068

def a (n : ℕ) : ℤ := (-2) ^ n
def b (n : ℕ) : ℤ := (-2) ^ n + 2
def c (n : ℕ) : ℚ := ((-2) ^ n : ℚ) / 2

theorem sequence_sum_equals_642 :
  ∃! n : ℕ, (a n : ℚ) + (b n : ℚ) + c n = 642 ∧ n = 8 :=
sorry

end NUMINAMATH_CALUDE_sequence_sum_equals_642_l1260_126068


namespace NUMINAMATH_CALUDE_expression_a_result_l1260_126010

theorem expression_a_result : 
  (7 * (2 / 3) + 16 * (5 / 12)) = 34 / 3 := by sorry

end NUMINAMATH_CALUDE_expression_a_result_l1260_126010


namespace NUMINAMATH_CALUDE_first_runner_pace_correct_l1260_126050

/-- The average pace of the first runner in a race with the following conditions:
  * The race is 10 miles long.
  * The second runner's pace is 7 minutes per mile.
  * The second runner stops after 56 minutes.
  * The second runner could remain stopped for 8 minutes before the first runner catches up.
-/
def firstRunnerPace : ℝ :=
  let raceLength : ℝ := 10
  let secondRunnerPace : ℝ := 7
  let secondRunnerStopTime : ℝ := 56
  let catchUpTime : ℝ := 8
  
  4  -- The actual pace, to be proved

theorem first_runner_pace_correct :
  let raceLength : ℝ := 10
  let secondRunnerPace : ℝ := 7
  let secondRunnerStopTime : ℝ := 56
  let catchUpTime : ℝ := 8
  
  firstRunnerPace = 4 := by sorry

end NUMINAMATH_CALUDE_first_runner_pace_correct_l1260_126050


namespace NUMINAMATH_CALUDE_conference_handshakes_l1260_126093

/-- The number of handshakes in a conference where each person shakes hands with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a conference of 35 people where each person shakes hands with every other person exactly once, the total number of handshakes is 595. -/
theorem conference_handshakes :
  handshakes 35 = 595 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l1260_126093


namespace NUMINAMATH_CALUDE_negation_of_universal_nonnegative_square_l1260_126088

theorem negation_of_universal_nonnegative_square (P : ℝ → Prop) : 
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) :=
sorry

end NUMINAMATH_CALUDE_negation_of_universal_nonnegative_square_l1260_126088


namespace NUMINAMATH_CALUDE_subset_condition_l1260_126007

def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}
def B (a : ℝ) : Set ℝ := {x | 0 < x ∧ x < a}

theorem subset_condition (a : ℝ) : A ⊆ B a ↔ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l1260_126007


namespace NUMINAMATH_CALUDE_marnie_chips_consumption_l1260_126019

/-- Given a bag of chips and Marnie's eating pattern, calculate the number of days to finish the bag -/
def days_to_finish_chips (total_chips : ℕ) (first_day_consumption : ℕ) (daily_consumption : ℕ) : ℕ :=
  1 + ((total_chips - first_day_consumption) + daily_consumption - 1) / daily_consumption

/-- Theorem: It takes Marnie 10 days to eat the whole bag of chips -/
theorem marnie_chips_consumption :
  days_to_finish_chips 100 10 10 = 10 := by
  sorry

#eval days_to_finish_chips 100 10 10

end NUMINAMATH_CALUDE_marnie_chips_consumption_l1260_126019


namespace NUMINAMATH_CALUDE_point_position_l1260_126028

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space with equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Determines if a point is on the upper right side of a line -/
def isUpperRight (l : Line) (p : Point) : Prop :=
  l.A * p.x + l.B * p.y + l.C < 0 ∧ l.A > 0 ∧ l.B < 0

theorem point_position (l : Line) (p : Point) :
  isUpperRight l p → p.y > (-l.A * p.x - l.C) / l.B :=
by sorry

end NUMINAMATH_CALUDE_point_position_l1260_126028


namespace NUMINAMATH_CALUDE_factor_tree_problem_l1260_126075

theorem factor_tree_problem (X Y Z F G : ℕ) : 
  X = Y * Z ∧ 
  Y = 7 * F ∧ 
  Z = 11 * G ∧ 
  F = 2 * 5 ∧ 
  G = 7 * 3 → 
  X = 16170 := by
sorry


end NUMINAMATH_CALUDE_factor_tree_problem_l1260_126075


namespace NUMINAMATH_CALUDE_percent_to_decimal_four_percent_to_decimal_l1260_126030

theorem percent_to_decimal (x : ℚ) :
  x / 100 = x * (1 / 100) := by sorry

theorem four_percent_to_decimal :
  (4 : ℚ) / 100 = (4 : ℚ) * (1 / 100) ∧ (4 : ℚ) * (1 / 100) = 0.04 := by sorry

end NUMINAMATH_CALUDE_percent_to_decimal_four_percent_to_decimal_l1260_126030


namespace NUMINAMATH_CALUDE_triangle_longest_side_l1260_126026

theorem triangle_longest_side :
  ∀ x : ℝ,
  let side1 := 7
  let side2 := x + 4
  let side3 := 2*x + 1
  (side1 + side2 + side3 = 36) →
  (∃ longest : ℝ, longest = max side1 (max side2 side3) ∧ longest = 17) :=
by sorry

end NUMINAMATH_CALUDE_triangle_longest_side_l1260_126026


namespace NUMINAMATH_CALUDE_tree_spacing_l1260_126049

/-- Given a road of length 151 feet where 11 trees can be planted, with each tree occupying 1 foot of space, 
    the distance between each tree is 14 feet. -/
theorem tree_spacing (road_length : ℕ) (num_trees : ℕ) (tree_space : ℕ) 
    (h1 : road_length = 151)
    (h2 : num_trees = 11)
    (h3 : tree_space = 1) : 
  (road_length - num_trees * tree_space) / (num_trees - 1) = 14 := by
  sorry


end NUMINAMATH_CALUDE_tree_spacing_l1260_126049


namespace NUMINAMATH_CALUDE_shot_put_distance_l1260_126055

/-- The horizontal distance at which a shot put hits the ground, given its trajectory. -/
theorem shot_put_distance : ∃ x : ℝ, x > 0 ∧ 
  (-1/12 * x^2 + 2/3 * x + 5/3 = 0) ∧ x = 10 := by sorry

end NUMINAMATH_CALUDE_shot_put_distance_l1260_126055


namespace NUMINAMATH_CALUDE_female_students_count_l1260_126047

theorem female_students_count (total : ℕ) (ways : ℕ) (f : ℕ) : 
  total = 8 → 
  ways = 30 → 
  (total - f) * (total - f - 1) * f = 2 * ways → 
  f = 3 := by
sorry

end NUMINAMATH_CALUDE_female_students_count_l1260_126047


namespace NUMINAMATH_CALUDE_douglas_vote_percentage_l1260_126082

theorem douglas_vote_percentage (total_percentage : ℝ) (ratio_x_to_y : ℝ) (y_percentage : ℝ) :
  total_percentage = 54 →
  ratio_x_to_y = 2 →
  y_percentage = 38.000000000000014 →
  ∃ x_percentage : ℝ,
    x_percentage = 62 ∧
    (x_percentage * (ratio_x_to_y / (ratio_x_to_y + 1)) + y_percentage * (1 / (ratio_x_to_y + 1))) = total_percentage :=
by sorry

end NUMINAMATH_CALUDE_douglas_vote_percentage_l1260_126082


namespace NUMINAMATH_CALUDE_nine_knights_among_travelers_total_travelers_is_sixteen_l1260_126046

/-- A traveler can be either a knight or a liar -/
inductive TravelerType
  | Knight
  | Liar

/-- Represents a room in the hotel -/
structure Room where
  knights : Nat
  liars : Nat

/-- Represents the hotel with three rooms -/
structure Hotel where
  room1 : Room
  room2 : Room
  room3 : Room

def total_travelers : Nat := 16

/-- Vasily, who makes contradictory statements -/
def vasily : TravelerType := TravelerType.Liar

/-- The theorem stating that there must be 9 knights among the 16 travelers -/
theorem nine_knights_among_travelers (h : Hotel) : 
  h.room1.knights + h.room2.knights + h.room3.knights = 9 :=
by
  sorry

/-- The theorem stating that the total number of travelers is 16 -/
theorem total_travelers_is_sixteen (h : Hotel) :
  h.room1.knights + h.room1.liars + 
  h.room2.knights + h.room2.liars + 
  h.room3.knights + h.room3.liars = total_travelers :=
by
  sorry

end NUMINAMATH_CALUDE_nine_knights_among_travelers_total_travelers_is_sixteen_l1260_126046


namespace NUMINAMATH_CALUDE_oxford_high_total_people_l1260_126023

-- Define the school structure
structure School where
  teachers : Nat
  principal : Nat
  vice_principals : Nat
  other_staff : Nat
  classes : Nat
  avg_students_per_class : Nat

-- Define Oxford High School
def oxford_high : School :=
  { teachers := 75,
    principal := 1,
    vice_principals := 3,
    other_staff := 20,
    classes := 35,
    avg_students_per_class := 23 }

-- Define the function to calculate total people
def total_people (s : School) : Nat :=
  s.teachers + s.principal + s.vice_principals + s.other_staff +
  (s.classes * s.avg_students_per_class)

-- Theorem statement
theorem oxford_high_total_people :
  total_people oxford_high = 904 := by
  sorry

end NUMINAMATH_CALUDE_oxford_high_total_people_l1260_126023


namespace NUMINAMATH_CALUDE_find_k_l1260_126000

theorem find_k (k : ℚ) (h : 56 / k = 4) : k = 14 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l1260_126000


namespace NUMINAMATH_CALUDE_museum_ticket_fraction_l1260_126027

theorem museum_ticket_fraction (total : ℚ) (sandwich_fraction : ℚ) (book_fraction : ℚ) (leftover : ℚ) :
  total = 150 →
  sandwich_fraction = 1/5 →
  book_fraction = 1/2 →
  leftover = 20 →
  (total - (sandwich_fraction * total + book_fraction * total + leftover)) / total = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_museum_ticket_fraction_l1260_126027


namespace NUMINAMATH_CALUDE_correct_distribution_probability_l1260_126039

def num_guests : ℕ := 3
def num_roll_types : ℕ := 4
def total_rolls : ℕ := 12
def rolls_per_guest : ℕ := 4

def probability_correct_distribution : ℚ := 2 / 103950

theorem correct_distribution_probability :
  let total_ways := (total_rolls.choose rolls_per_guest) * 
                    ((total_rolls - rolls_per_guest).choose rolls_per_guest) *
                    ((total_rolls - 2*rolls_per_guest).choose rolls_per_guest)
  let correct_ways := (num_roll_types.factorial) * 
                      (2^num_roll_types) * 
                      (1^num_roll_types)
  (correct_ways : ℚ) / total_ways = probability_correct_distribution :=
sorry

end NUMINAMATH_CALUDE_correct_distribution_probability_l1260_126039


namespace NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l1260_126064

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: Solution set when a = 2
theorem solution_set_a_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} := by sorry

-- Part 2: Range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f x a ≥ 4} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l1260_126064


namespace NUMINAMATH_CALUDE_lighthouse_model_height_l1260_126038

def original_height : ℝ := 60
def original_base_height : ℝ := 12
def original_base_volume : ℝ := 150000
def model_base_volume : ℝ := 0.15

theorem lighthouse_model_height :
  let scale_factor := (model_base_volume / original_base_volume) ^ (1/3)
  let model_height := original_height * scale_factor
  model_height * 100 = 60 := by sorry

end NUMINAMATH_CALUDE_lighthouse_model_height_l1260_126038


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1260_126035

theorem simplify_and_evaluate (x : ℝ) :
  x = -2 →
  (1 - 2 / (2 - x)) / (x / (x^2 - 4*x + 4)) = x - 2 ∧
  x - 2 = -4 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1260_126035


namespace NUMINAMATH_CALUDE_square_field_area_l1260_126072

theorem square_field_area (side_length : ℝ) (h : side_length = 13) :
  side_length * side_length = 169 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l1260_126072


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1260_126031

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1260_126031


namespace NUMINAMATH_CALUDE_greatest_prime_factor_factorial_sum_l1260_126017

theorem greatest_prime_factor_factorial_sum : 
  ∃ p : ℕ, p.Prime ∧ p ∣ (Nat.factorial 15 + Nat.factorial 18) ∧ 
  ∀ q : ℕ, q.Prime → q ∣ (Nat.factorial 15 + Nat.factorial 18) → q ≤ p :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_factorial_sum_l1260_126017


namespace NUMINAMATH_CALUDE_fish_distribution_l1260_126005

theorem fish_distribution (total_fish : ℕ) (num_bowls : ℕ) (fish_per_bowl : ℕ) :
  total_fish = 6003 →
  num_bowls = 261 →
  total_fish = num_bowls * fish_per_bowl →
  fish_per_bowl = 23 := by
  sorry

end NUMINAMATH_CALUDE_fish_distribution_l1260_126005


namespace NUMINAMATH_CALUDE_opposite_direction_speed_l1260_126087

/-- Given two people moving in opposite directions, this theorem proves
    the speed of one person given the conditions of the problem. -/
theorem opposite_direction_speed
  (time : ℝ)
  (total_distance : ℝ)
  (speed_person1 : ℝ)
  (h1 : time = 4)
  (h2 : total_distance = 28)
  (h3 : speed_person1 = 3)
  : ∃ speed_person2 : ℝ,
    speed_person2 = 4 ∧ 
    total_distance = time * (speed_person1 + speed_person2) :=
by
  sorry

#check opposite_direction_speed

end NUMINAMATH_CALUDE_opposite_direction_speed_l1260_126087


namespace NUMINAMATH_CALUDE_mobile_phone_price_l1260_126071

theorem mobile_phone_price (x : ℝ) : 
  (0.8 * (1.4 * x)) - x = 270 → x = 2250 := by
  sorry

end NUMINAMATH_CALUDE_mobile_phone_price_l1260_126071


namespace NUMINAMATH_CALUDE_square_area_ratio_l1260_126090

theorem square_area_ratio (a b : ℝ) (h : 4 * a = 16 * b) : a^2 = 16 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1260_126090


namespace NUMINAMATH_CALUDE_airplane_passengers_l1260_126015

theorem airplane_passengers (P : ℕ) 
  (h1 : P - 58 + 24 - 47 + 14 + 10 = 67) : P = 124 := by
  sorry

end NUMINAMATH_CALUDE_airplane_passengers_l1260_126015


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l1260_126044

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem tenth_term_of_sequence (a : ℤ) (d : ℤ) :
  arithmetic_sequence a d 4 = 23 →
  arithmetic_sequence a d 8 = 55 →
  arithmetic_sequence a d 10 = 71 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l1260_126044


namespace NUMINAMATH_CALUDE_root_in_interval_l1260_126070

-- Define the function f(x) = x^3 - x - 5
def f (x : ℝ) : ℝ := x^3 - x - 5

-- State the theorem
theorem root_in_interval :
  (f 1 < 0) → (f 2 > 0) → (f 1.5 < 0) →
  ∃ x : ℝ, x ∈ Set.Ioo 1.5 2 ∧ f x = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_root_in_interval_l1260_126070


namespace NUMINAMATH_CALUDE_curve_arc_length_l1260_126001

noncomputable def arcLength (ρ : Real → Real) (φ₁ φ₂ : Real) : Real :=
  ∫ x in φ₁..φ₂, Real.sqrt (ρ x ^ 2 + (deriv ρ x) ^ 2)

theorem curve_arc_length :
  let ρ := fun φ => 3 * Real.exp (3 * φ / 4)
  let φ₁ := -π / 2
  let φ₂ := π / 2
  arcLength ρ φ₁ φ₂ = 10 * Real.sinh (3 * π / 8) := by
  sorry

end NUMINAMATH_CALUDE_curve_arc_length_l1260_126001


namespace NUMINAMATH_CALUDE_quinary_1234_eq_194_l1260_126043

/-- Converts a quinary (base-5) number to decimal. -/
def quinary_to_decimal (q : List Nat) : Nat :=
  q.enum.foldr (fun (i, d) acc => acc + d * (5 ^ i)) 0

/-- The quinary representation of 1234₍₅₎ -/
def quinary_1234 : List Nat := [4, 3, 2, 1]

theorem quinary_1234_eq_194 : quinary_to_decimal quinary_1234 = 194 := by
  sorry

end NUMINAMATH_CALUDE_quinary_1234_eq_194_l1260_126043


namespace NUMINAMATH_CALUDE_fraction_addition_l1260_126013

theorem fraction_addition : (1 : ℚ) / 4 + (3 : ℚ) / 5 = (17 : ℚ) / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1260_126013


namespace NUMINAMATH_CALUDE_roots_sum_product_l1260_126083

theorem roots_sum_product (a b : ℝ) : 
  (a^4 - 6*a - 1 = 0) → 
  (b^4 - 6*b - 1 = 0) → 
  (a ≠ b) →
  (a*b + a + b = 1) := by
sorry

end NUMINAMATH_CALUDE_roots_sum_product_l1260_126083


namespace NUMINAMATH_CALUDE_average_marks_l1260_126018

theorem average_marks (total_subjects : ℕ) (avg_five_subjects : ℝ) (sixth_subject_marks : ℝ) :
  total_subjects = 6 →
  avg_five_subjects = 74 →
  sixth_subject_marks = 104 →
  (avg_five_subjects * 5 + sixth_subject_marks) / total_subjects = 79 :=
by
  sorry

end NUMINAMATH_CALUDE_average_marks_l1260_126018


namespace NUMINAMATH_CALUDE_parallel_transitivity_false_l1260_126052

-- Define the necessary types
variable (Point Line Plane : Type)

-- Define the relations
variable (belongs_to : Point → Line → Prop)
variable (intersects : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem parallel_transitivity_false :
  ¬(∀ (l m : Line) (α β : Plane),
    parallel_line_plane l α →
    parallel_line_plane m β →
    parallel_planes α β →
    parallel_lines l m) :=
sorry

end NUMINAMATH_CALUDE_parallel_transitivity_false_l1260_126052


namespace NUMINAMATH_CALUDE_negation_of_forall_cube_greater_square_l1260_126022

theorem negation_of_forall_cube_greater_square :
  (¬ ∀ x : ℕ, x^3 > x^2) ↔ (∃ x : ℕ, x^3 ≤ x^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_cube_greater_square_l1260_126022


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1260_126057

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The statement to be proved -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 3)^2 + 7*(a 3) + 9 = 0 →
  (a 7)^2 + 7*(a 7) + 9 = 0 →
  a 5 = -3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1260_126057


namespace NUMINAMATH_CALUDE_smallest_integer_l1260_126076

theorem smallest_integer (a b : ℕ+) (h1 : a = 60) (h2 : (Nat.lcm a b) / (Nat.gcd a b) = 44) : 
  b ≥ 165 ∧ ∃ (b' : ℕ+), b' = 165 ∧ (Nat.lcm a b') / (Nat.gcd a b') = 44 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_l1260_126076


namespace NUMINAMATH_CALUDE_function_identity_l1260_126058

theorem function_identity (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → f ((x - 2) / (x + 1)) + f ((3 + x) / (1 - x)) = x) →
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → f x = (x^3 + 7*x) / (2 - 2*x^2)) :=
by sorry

end NUMINAMATH_CALUDE_function_identity_l1260_126058


namespace NUMINAMATH_CALUDE_ana_win_probability_l1260_126033

/-- Represents the probability of winning for a player in the coin flipping game -/
def winProbability (playerPosition : ℕ) : ℚ :=
  (1 / 2) ^ playerPosition / (1 - (1 / 2) ^ 4)

/-- The coin flipping game with four players -/
theorem ana_win_probability :
  winProbability 4 = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_ana_win_probability_l1260_126033


namespace NUMINAMATH_CALUDE_largest_factor_and_smallest_multiple_of_18_l1260_126067

theorem largest_factor_and_smallest_multiple_of_18 :
  (∃ n : ℕ, n ≤ 18 ∧ 18 % n = 0 ∧ ∀ m : ℕ, m ≤ 18 ∧ 18 % m = 0 → m ≤ n) ∧
  (∃ k : ℕ, 18 ∣ k ∧ ∀ j : ℕ, 18 ∣ j → k ≤ j) :=
by sorry

end NUMINAMATH_CALUDE_largest_factor_and_smallest_multiple_of_18_l1260_126067


namespace NUMINAMATH_CALUDE_correct_initial_driving_time_l1260_126092

/-- Represents the driving scenario with given conditions -/
structure DrivingScenario where
  totalDistance : ℝ
  initialSpeed : ℝ
  finalSpeed : ℝ
  lateTime : ℝ
  earlyTime : ℝ

/-- Calculates the time driven at the initial speed -/
def initialDrivingTime (scenario : DrivingScenario) : ℝ :=
  sorry

/-- Theorem stating the correct initial driving time for the given scenario -/
theorem correct_initial_driving_time (scenario : DrivingScenario) 
  (h1 : scenario.totalDistance = 45)
  (h2 : scenario.initialSpeed = 15)
  (h3 : scenario.finalSpeed = 60)
  (h4 : scenario.lateTime = 1)
  (h5 : scenario.earlyTime = 0.5) :
  initialDrivingTime scenario = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_correct_initial_driving_time_l1260_126092


namespace NUMINAMATH_CALUDE_long_jump_solution_l1260_126004

/-- Represents the long jump problem with given conditions -/
def LongJumpProblem (initial_avg : ℝ) (second_jump : ℝ) (second_avg : ℝ) (final_avg : ℝ) : Prop :=
  ∃ (n : ℕ) (third_jump : ℝ),
    -- Initial condition
    initial_avg = 3.80
    -- Second jump condition
    ∧ second_jump = 3.99
    -- New average after second jump
    ∧ second_avg = 3.81
    -- Final average after third jump
    ∧ final_avg = 3.82
    -- Relationship between jumps and averages
    ∧ (initial_avg * n + second_jump) / (n + 1) = second_avg
    ∧ (initial_avg * n + second_jump + third_jump) / (n + 2) = final_avg
    -- The third jump is the solution
    ∧ third_jump = 4.01

/-- Theorem stating the solution to the long jump problem -/
theorem long_jump_solution :
  LongJumpProblem 3.80 3.99 3.81 3.82 :=
by
  sorry

#check long_jump_solution

end NUMINAMATH_CALUDE_long_jump_solution_l1260_126004


namespace NUMINAMATH_CALUDE_collinear_points_m_value_l1260_126029

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem collinear_points_m_value :
  ∀ m : ℝ,
  let p1 : Point := ⟨3, -4⟩
  let p2 : Point := ⟨6, 5⟩
  let p3 : Point := ⟨8, m⟩
  collinear p1 p2 p3 → m = 11 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_m_value_l1260_126029


namespace NUMINAMATH_CALUDE_sum_even_factors_720_l1260_126095

def even_factor_sum (n : ℕ) : ℕ := sorry

theorem sum_even_factors_720 : even_factor_sum 720 = 2340 := by sorry

end NUMINAMATH_CALUDE_sum_even_factors_720_l1260_126095


namespace NUMINAMATH_CALUDE_pipe_speed_ratio_l1260_126099

-- Define the rates of pipes A, B, and C
def rate_A : ℚ := 1 / 21
def rate_B : ℚ := 2 / 21
def rate_C : ℚ := 4 / 21

-- State the theorem
theorem pipe_speed_ratio :
  -- Conditions
  (rate_A + rate_B + rate_C = 1 / 3) →  -- All pipes fill the tank in 3 hours
  (rate_C = 2 * rate_B) →               -- Pipe C is twice as fast as B
  (rate_A = 1 / 21) →                   -- Pipe A alone takes 21 hours
  -- Conclusion
  (rate_B / rate_A = 2) :=
by sorry

end NUMINAMATH_CALUDE_pipe_speed_ratio_l1260_126099


namespace NUMINAMATH_CALUDE_specific_frustum_small_cone_altitude_l1260_126037

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  altitude : ℝ
  lower_base_area : ℝ
  upper_base_area : ℝ

/-- Calculates the altitude of the small cone cut off from a frustum -/
def small_cone_altitude (f : Frustum) : ℝ :=
  f.altitude

/-- Theorem: The altitude of the small cone cut off from a specific frustum is 18 -/
theorem specific_frustum_small_cone_altitude :
  let f : Frustum := { altitude := 18, lower_base_area := 400 * Real.pi, upper_base_area := 100 * Real.pi }
  small_cone_altitude f = 18 := by sorry

end NUMINAMATH_CALUDE_specific_frustum_small_cone_altitude_l1260_126037


namespace NUMINAMATH_CALUDE_parallel_postulate_l1260_126053

-- Define a Point type
def Point : Type := ℝ × ℝ

-- Define a Line type
def Line : Type := Point → Point → Prop

-- Define a parallel relation between lines
def Parallel (l1 l2 : Line) : Prop := sorry

-- Define a point being on a line
def OnLine (p : Point) (l : Line) : Prop := sorry

-- State the theorem
theorem parallel_postulate (l : Line) (p : Point) : 
  ¬(OnLine p l) → ∃! (m : Line), Parallel m l ∧ OnLine p m := by sorry

end NUMINAMATH_CALUDE_parallel_postulate_l1260_126053


namespace NUMINAMATH_CALUDE_cos_sum_17th_roots_l1260_126094

theorem cos_sum_17th_roots : 
  Real.cos (2 * Real.pi / 17) + Real.cos (6 * Real.pi / 17) + Real.cos (8 * Real.pi / 17) = (Real.sqrt 13 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_17th_roots_l1260_126094


namespace NUMINAMATH_CALUDE_root_sum_quotient_l1260_126032

/-- Given a quadratic equation m(x^2 - 2x) + 3x + 4 = 0 with roots p and q,
    and m₁ and m₂ are values of m for which p/q + q/p = 2,
    prove that m₁/m₂ + m₂/m₁ = 178/9 -/
theorem root_sum_quotient (m₁ m₂ : ℝ) (p q : ℝ) :
  (m₁ * (p^2 - 2*p) + 3*p + 4 = 0) →
  (m₁ * (q^2 - 2*q) + 3*q + 4 = 0) →
  (m₂ * (p^2 - 2*p) + 3*p + 4 = 0) →
  (m₂ * (q^2 - 2*q) + 3*q + 4 = 0) →
  (p / q + q / p = 2) →
  m₁ / m₂ + m₂ / m₁ = 178 / 9 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_quotient_l1260_126032


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l1260_126061

theorem chess_tournament_participants (n : ℕ) (m : ℕ) : 
  (2 : ℕ) + n = number_of_participants →
  8 = points_scored_by_7th_graders →
  m * n = points_scored_by_8th_graders →
  m * n + 8 = total_points_scored →
  (n + 2) * (n + 1) / 2 = total_games_played →
  total_points_scored = total_games_played →
  n = 7 :=
sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l1260_126061


namespace NUMINAMATH_CALUDE_range_of_a_l1260_126091

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + a| < 3 ↔ 2 < x ∧ x < 3) → 
  -5 ≤ a ∧ a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1260_126091


namespace NUMINAMATH_CALUDE_log_sum_squares_primes_l1260_126062

theorem log_sum_squares_primes (a b : ℕ) (ha : Prime a) (hb : Prime b) 
  (hab : a ≠ b) (ha_gt_2 : a > 2) (hb_gt_2 : b > 2) :
  Real.log (a^2) / Real.log (a * b) + Real.log (b^2) / Real.log (a * b) = 2 := by
sorry

end NUMINAMATH_CALUDE_log_sum_squares_primes_l1260_126062


namespace NUMINAMATH_CALUDE_sons_age_l1260_126002

theorem sons_age (son_age father_age : ℕ) : 
  father_age = 7 * (son_age - 8) →
  father_age / 4 = 14 →
  son_age = 16 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l1260_126002


namespace NUMINAMATH_CALUDE_stratified_sampling_primary_schools_l1260_126016

theorem stratified_sampling_primary_schools 
  (total_schools : ℕ) 
  (primary_schools : ℕ) 
  (selected_schools : ℕ) 
  (h1 : total_schools = 250) 
  (h2 : primary_schools = 150) 
  (h3 : selected_schools = 30) :
  (primary_schools : ℚ) / (total_schools : ℚ) * (selected_schools : ℚ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_primary_schools_l1260_126016
