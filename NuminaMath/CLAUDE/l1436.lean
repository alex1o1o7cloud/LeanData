import Mathlib

namespace NUMINAMATH_CALUDE_tank_filling_l1436_143620

theorem tank_filling (tank_capacity : ℕ) (buckets_case1 buckets_case2 capacity_case1 : ℕ) :
  tank_capacity = buckets_case1 * capacity_case1 →
  tank_capacity = buckets_case2 * (tank_capacity / buckets_case2) →
  buckets_case1 = 12 →
  capacity_case1 = 81 →
  buckets_case2 = 108 →
  tank_capacity / buckets_case2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_tank_filling_l1436_143620


namespace NUMINAMATH_CALUDE_cube_expansion_2013_l1436_143650

theorem cube_expansion_2013 : ∃! n : ℕ, 
  n > 0 ∧ 
  (n - 1)^2 + (n - 1) ≤ 2013 ∧ 
  2013 < n^2 + n ∧
  n = 45 := by sorry

end NUMINAMATH_CALUDE_cube_expansion_2013_l1436_143650


namespace NUMINAMATH_CALUDE_group_size_l1436_143624

/-- The number of members in the group -/
def n : ℕ := sorry

/-- The total collection in paise -/
def total_paise : ℕ := 5776

/-- Each member contributes as many paise as there are members -/
axiom member_contribution : n = total_paise / n

theorem group_size : n = 76 := by sorry

end NUMINAMATH_CALUDE_group_size_l1436_143624


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1436_143686

theorem least_subtraction_for_divisibility : 
  ∃! x : ℕ, x ≤ 14 ∧ (42398 - x) % 15 = 0 ∧ ∀ y : ℕ, y < x → (42398 - y) % 15 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1436_143686


namespace NUMINAMATH_CALUDE_no_natural_numbers_satisfying_condition_l1436_143694

theorem no_natural_numbers_satisfying_condition :
  ∀ (x y : ℕ), x + y - 2021 ≥ Nat.gcd x y + Nat.lcm x y :=
by sorry

end NUMINAMATH_CALUDE_no_natural_numbers_satisfying_condition_l1436_143694


namespace NUMINAMATH_CALUDE_range_of_product_l1436_143684

theorem range_of_product (x y z w : ℝ) 
  (sum_zero : x + y + z + w = 0)
  (sum_seventh_power_zero : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_product_l1436_143684


namespace NUMINAMATH_CALUDE_pi_approximation_l1436_143631

theorem pi_approximation (π : Real) (h : π = 4 * Real.sin (52 * π / 180)) :
  (1 - 2 * (Real.cos (7 * π / 180))^2) / (π * Real.sqrt (16 - π^2)) = -1/8 := by
  sorry

end NUMINAMATH_CALUDE_pi_approximation_l1436_143631


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l1436_143626

theorem prime_sum_theorem (p q r s : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧ Prime s ∧  -- p, q, r, s are primes
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧  -- p, q, r, s are distinct
  Prime (p + q + r + s) ∧  -- their sum is prime
  ∃ a, p^2 + q*r = a^2 ∧  -- p² + qr is a perfect square
  ∃ b, p^2 + q*s = b^2  -- p² + qs is a perfect square
  → p + q + r + s = 23 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l1436_143626


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l1436_143604

theorem quadratic_roots_ratio (k : ℝ) : 
  (∃ r s : ℝ, r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 / 2 ∧ 
   r^2 + 10*r + k = 0 ∧ s^2 + 10*s + k = 0) → k = 24 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l1436_143604


namespace NUMINAMATH_CALUDE_diameter_of_figure_F_l1436_143678

/-- A triangle with semicircles constructed outwardly on each side -/
structure TriangleWithSemicircles where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

/-- The figure F composed of the triangle and the three semicircles -/
def FigureF (t : TriangleWithSemicircles) : Set (ℝ × ℝ) :=
  sorry

/-- The diameter of a set in the plane -/
def diameter (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem: The diameter of figure F is equal to the semi-perimeter of the triangle -/
theorem diameter_of_figure_F (t : TriangleWithSemicircles) :
    diameter (FigureF t) = (t.a + t.b + t.c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_diameter_of_figure_F_l1436_143678


namespace NUMINAMATH_CALUDE_original_proposition_contrapositive_proposition_both_true_l1436_143635

-- Define the quadratic equation
def has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + x - m = 0

-- Original proposition
theorem original_proposition : 
  ∀ m : ℝ, m > 0 → has_real_roots m :=
sorry

-- Contrapositive of the original proposition
theorem contrapositive_proposition :
  ∀ m : ℝ, ¬(has_real_roots m) → ¬(m > 0) :=
sorry

-- Both the original proposition and its contrapositive are true
theorem both_true : 
  (∀ m : ℝ, m > 0 → has_real_roots m) ∧ 
  (∀ m : ℝ, ¬(has_real_roots m) → ¬(m > 0)) :=
sorry

end NUMINAMATH_CALUDE_original_proposition_contrapositive_proposition_both_true_l1436_143635


namespace NUMINAMATH_CALUDE_cakes_ratio_l1436_143696

/-- Carter's usual weekly baking schedule -/
def usual_cheesecakes : ℕ := 6
def usual_muffins : ℕ := 5
def usual_red_velvet : ℕ := 8

/-- Total number of cakes Carter usually bakes in a week -/
def usual_total : ℕ := usual_cheesecakes + usual_muffins + usual_red_velvet

/-- Additional cakes baked this week -/
def additional_cakes : ℕ := 38

/-- Theorem stating the ratio of cakes baked this week to usual weeks -/
theorem cakes_ratio :
  ∃ (x : ℕ), x * usual_total = usual_total + additional_cakes ∧
  (x * usual_total : ℚ) / usual_total = 3 := by
  sorry

end NUMINAMATH_CALUDE_cakes_ratio_l1436_143696


namespace NUMINAMATH_CALUDE_derivative_at_one_l1436_143659

/-- Given a function f: ℝ → ℝ satisfying f(x) = 2x * f'(1) + 1/x for all x ≠ 0,
    prove that f'(1) = 1 -/
theorem derivative_at_one (f : ℝ → ℝ) (hf : ∀ x ≠ 0, f x = 2 * x * (deriv f 1) + 1 / x) :
  deriv f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_one_l1436_143659


namespace NUMINAMATH_CALUDE_other_intersection_point_l1436_143623

/-- Two circles with centers on a line intersecting at two points -/
structure TwoCirclesIntersection where
  -- The line equation: x - y + 1 = 0
  line : ℝ → ℝ → Prop
  line_eq : ∀ x y, line x y ↔ x - y + 1 = 0
  
  -- The circles intersect at two different points
  intersect_points : Fin 2 → ℝ × ℝ
  different_points : intersect_points 0 ≠ intersect_points 1
  
  -- One intersection point is (-2, 2)
  known_point : intersect_points 0 = (-2, 2)

/-- The other intersection point has coordinates (1, -1) -/
theorem other_intersection_point (c : TwoCirclesIntersection) : 
  c.intersect_points 1 = (1, -1) := by
  sorry

end NUMINAMATH_CALUDE_other_intersection_point_l1436_143623


namespace NUMINAMATH_CALUDE_x_value_proof_l1436_143616

theorem x_value_proof (x y z : ℤ) 
  (eq1 : x + y = 20) 
  (eq2 : x - y = 10) 
  (eq3 : x + y + z = 30) : x = 15 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1436_143616


namespace NUMINAMATH_CALUDE_garden_area_ratio_l1436_143688

theorem garden_area_ratio :
  ∀ (L W : ℝ),
  L / W = 5 / 4 →
  L + W = 50 →
  (L * W) / (π * (W / 2)^2) = 5 / π :=
λ L W h1 h2 => by
  sorry

end NUMINAMATH_CALUDE_garden_area_ratio_l1436_143688


namespace NUMINAMATH_CALUDE_road_repaving_l1436_143625

theorem road_repaving (total_repaved : ℕ) (repaved_today : ℕ) 
  (h1 : total_repaved = 4938)
  (h2 : repaved_today = 805) :
  total_repaved - repaved_today = 4133 := by
  sorry

end NUMINAMATH_CALUDE_road_repaving_l1436_143625


namespace NUMINAMATH_CALUDE_bridget_apples_bridget_bought_14_apples_l1436_143637

theorem bridget_apples : ℕ → Prop :=
  fun total : ℕ =>
    let remaining_after_ann : ℕ := total / 2
    let remaining_after_cassie : ℕ := remaining_after_ann - 3
    remaining_after_cassie = 4 → total = 14

-- The proof
theorem bridget_bought_14_apples : bridget_apples 14 := by
  sorry

end NUMINAMATH_CALUDE_bridget_apples_bridget_bought_14_apples_l1436_143637


namespace NUMINAMATH_CALUDE_electricity_price_correct_l1436_143609

/-- The electricity price per kWh in Coco's town -/
def electricity_price : ℝ := 0.1

/-- Coco's oven consumption rate in kWh -/
def oven_consumption_rate : ℝ := 2.4

/-- The number of hours Coco used his oven -/
def hours_used : ℝ := 25

/-- The amount Coco paid for using his oven -/
def amount_paid : ℝ := 6

/-- Theorem stating that the electricity price is correct -/
theorem electricity_price_correct : 
  electricity_price = amount_paid / (oven_consumption_rate * hours_used) :=
by sorry

end NUMINAMATH_CALUDE_electricity_price_correct_l1436_143609


namespace NUMINAMATH_CALUDE_range_of_PQ_length_l1436_143685

/-- Circle C in the Cartesian coordinate system -/
def CircleC (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 2

/-- Point A is on the x-axis -/
def PointA (x : ℝ) : Prop := true

/-- AP is tangent to circle C at point P -/
def TangentAP (A P : ℝ × ℝ) : Prop := sorry

/-- AQ is tangent to circle C at point Q -/
def TangentAQ (A Q : ℝ × ℝ) : Prop := sorry

/-- The length of segment PQ -/
def LengthPQ (P Q : ℝ × ℝ) : ℝ := sorry

theorem range_of_PQ_length :
  ∀ A P Q : ℝ × ℝ,
    PointA A.1 →
    CircleC P.1 P.2 →
    CircleC Q.1 Q.2 →
    TangentAP A P →
    TangentAQ A Q →
    (2 * Real.sqrt 14 / 3 ≤ LengthPQ P Q) ∧ (LengthPQ P Q < 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_PQ_length_l1436_143685


namespace NUMINAMATH_CALUDE_power_of_power_l1436_143627

theorem power_of_power : (3^4)^2 = 6561 := by sorry

end NUMINAMATH_CALUDE_power_of_power_l1436_143627


namespace NUMINAMATH_CALUDE_power_equation_solution_l1436_143676

theorem power_equation_solution :
  ∃ y : ℝ, ((1/8 : ℝ) * 2^36 = 4^y) → y = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1436_143676


namespace NUMINAMATH_CALUDE_solution_range_l1436_143608

def M (a : ℝ) := {x : ℝ | (a - 2) * x^2 + (2*a - 1) * x + 6 > 0}

theorem solution_range (a : ℝ) (h1 : 3 ∈ M a) (h2 : 5 ∉ M a) : 1 < a ∧ a ≤ 7/5 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l1436_143608


namespace NUMINAMATH_CALUDE_prob_at_least_one_white_correct_l1436_143614

def total_balls : ℕ := 9
def red_balls : ℕ := 5
def white_balls : ℕ := 4

def prob_at_least_one_white : ℚ := 13 / 18

theorem prob_at_least_one_white_correct :
  let prob_two_red : ℚ := (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1))
  1 - prob_two_red = prob_at_least_one_white := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_white_correct_l1436_143614


namespace NUMINAMATH_CALUDE_reciprocal_minus_one_l1436_143633

theorem reciprocal_minus_one (x : ℝ) : (1 / x = -1) → |-x - 1| = 0 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_minus_one_l1436_143633


namespace NUMINAMATH_CALUDE_mrs_hilt_total_miles_l1436_143679

/-- The total miles run by Mrs. Hilt in a week -/
def total_miles (monday wednesday friday : ℕ) : ℕ := monday + wednesday + friday

/-- Theorem: Mrs. Hilt's total miles run in the week is 12 -/
theorem mrs_hilt_total_miles : total_miles 3 2 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_total_miles_l1436_143679


namespace NUMINAMATH_CALUDE_special_triangle_secant_sum_range_l1436_143665

-- Define a structure for a triangle with the given condition
structure SpecialTriangle where
  A : Real
  B : Real
  C : Real
  angle_sum : A + B + C = π
  special_condition : A + C = 2 * B

-- Define the secant function
noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

-- State the theorem
theorem special_triangle_secant_sum_range (t : SpecialTriangle) :
  ∃ (f : Real → Real), 
    (∀ x, f x = sec t.A + sec t.C) ∧ 
    (Set.range f = {y | y < -1 ∨ y ≥ 4}) := by
  sorry


end NUMINAMATH_CALUDE_special_triangle_secant_sum_range_l1436_143665


namespace NUMINAMATH_CALUDE_curve_is_ellipse_with_foci_on_y_axis_l1436_143621

-- Define the angle α in radians
variable (α : Real)

-- Define the condition 0° < α < 90°
axiom alpha_range : 0 < α ∧ α < Real.pi / 2

-- Define the equation of the curve
def curve_equation (x y : Real) : Prop :=
  x^2 + y^2 * Real.cos α = 1

-- State the theorem
theorem curve_is_ellipse_with_foci_on_y_axis :
  ∃ (a b : Real), a > b ∧ b > 0 ∧
  ∀ (x y : Real), curve_equation α x y ↔ (x^2 / b^2) + (y^2 / a^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_curve_is_ellipse_with_foci_on_y_axis_l1436_143621


namespace NUMINAMATH_CALUDE_second_scenario_cost_l1436_143610

/-- The cost of a single shirt -/
def shirt_cost : ℝ := sorry

/-- The cost of a single trouser -/
def trouser_cost : ℝ := sorry

/-- The cost of a single tie -/
def tie_cost : ℝ := sorry

/-- The first scenario: 6 shirts, 4 trousers, and 2 ties cost $80 -/
def scenario1 : Prop := 6 * shirt_cost + 4 * trouser_cost + 2 * tie_cost = 80

/-- The third scenario: 5 shirts, 3 trousers, and 2 ties cost $110 -/
def scenario3 : Prop := 5 * shirt_cost + 3 * trouser_cost + 2 * tie_cost = 110

/-- Theorem: Given scenario1 and scenario3, the cost of 4 shirts, 2 trousers, and 2 ties is $50 -/
theorem second_scenario_cost (h1 : scenario1) (h3 : scenario3) : 
  4 * shirt_cost + 2 * trouser_cost + 2 * tie_cost = 50 := by sorry

end NUMINAMATH_CALUDE_second_scenario_cost_l1436_143610


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l1436_143615

theorem complex_power_magnitude (z : ℂ) (h : z = 4/5 + 3/5 * I) :
  Complex.abs (z^8) = 1 := by sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l1436_143615


namespace NUMINAMATH_CALUDE_square_root_equation_solution_l1436_143656

theorem square_root_equation_solution (x : ℝ) (h1 : x ≠ 0) (h2 : Real.sqrt ((7 * x) / 5) = x) : x = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_solution_l1436_143656


namespace NUMINAMATH_CALUDE_projection_of_a_onto_b_l1436_143657

def vector_a : ℝ × ℝ := (-1, 1)
def vector_b : ℝ × ℝ := (3, 4)

theorem projection_of_a_onto_b :
  (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2) / Real.sqrt (vector_b.1^2 + vector_b.2^2) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_projection_of_a_onto_b_l1436_143657


namespace NUMINAMATH_CALUDE_fedora_cleaning_time_l1436_143630

/-- Represents the cleaning problem of Fedora Egorovna's stove wall. -/
def CleaningProblem (total_sections : ℕ) (cleaned_sections : ℕ) (time_spent : ℕ) : Prop :=
  let cleaning_rate := time_spent / cleaned_sections
  let total_time := total_sections * cleaning_rate
  let additional_time := total_time - time_spent
  additional_time = 192

/-- Theorem stating that given the conditions of Fedora's cleaning,
    the additional time required is 192 minutes. -/
theorem fedora_cleaning_time :
  CleaningProblem 27 3 24 :=
by
  sorry

#check fedora_cleaning_time

end NUMINAMATH_CALUDE_fedora_cleaning_time_l1436_143630


namespace NUMINAMATH_CALUDE_shaded_area_concentric_circles_l1436_143689

theorem shaded_area_concentric_circles 
  (r₁ r₂ : ℝ) 
  (h₁ : r₁ > 0) 
  (h₂ : r₂ > r₁) 
  (h₃ : r₁ / (r₂ - r₁) = 1 / 2) 
  (h₄ : r₂ = 9) : 
  π * r₂^2 - π * r₁^2 = 72 * π := by
sorry

end NUMINAMATH_CALUDE_shaded_area_concentric_circles_l1436_143689


namespace NUMINAMATH_CALUDE_contractor_engagement_l1436_143651

/-- Contractor engagement problem -/
theorem contractor_engagement
  (daily_wage : ℝ)
  (daily_fine : ℝ)
  (total_payment : ℝ)
  (absent_days : ℕ)
  (h1 : daily_wage = 25)
  (h2 : daily_fine = 7.5)
  (h3 : total_payment = 425)
  (h4 : absent_days = 10) :
  ∃ (worked_days : ℕ) (total_days : ℕ),
    worked_days * daily_wage - absent_days * daily_fine = total_payment ∧
    total_days = worked_days + absent_days ∧
    total_days = 30 := by
  sorry

end NUMINAMATH_CALUDE_contractor_engagement_l1436_143651


namespace NUMINAMATH_CALUDE_root_sum_fraction_l1436_143655

theorem root_sum_fraction (m₁ m₂ : ℝ) : 
  (∃ a b : ℝ, m₁ * (a^2 - 3*a) + 2*a + 7 = 0 ∧ 
              m₂ * (b^2 - 3*b) + 2*b + 7 = 0 ∧ 
              a/b + b/a = 7/3) →
  m₁/m₂ + m₂/m₁ = 15481/324 := by
sorry

end NUMINAMATH_CALUDE_root_sum_fraction_l1436_143655


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1436_143607

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence {aₙ}, if a₄ + a₅ + a₆ = 90, then a₅ = 30 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h1 : ArithmeticSequence a) (h2 : a 4 + a 5 + a 6 = 90) :
  a 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1436_143607


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1436_143674

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, x > a → x > 2) ∧ (∃ x, x > 2 ∧ x ≤ a) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1436_143674


namespace NUMINAMATH_CALUDE_five_sixths_of_twelve_fifths_minus_half_l1436_143693

theorem five_sixths_of_twelve_fifths_minus_half :
  (5 / 6 : ℚ) * (12 / 5 : ℚ) - (1 / 2 : ℚ) = (3 / 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_five_sixths_of_twelve_fifths_minus_half_l1436_143693


namespace NUMINAMATH_CALUDE_frieda_prob_reach_edge_l1436_143648

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (row : Fin 4)
  (col : Fin 4)

/-- Defines the center position -/
def center : Position := ⟨1, 1⟩

/-- Checks if a position is on the edge of the grid -/
def isEdge (p : Position) : Bool :=
  p.row = 0 || p.row = 3 || p.col = 0 || p.col = 3

/-- Defines the possible moves -/
inductive Move
  | up
  | down
  | left
  | right

/-- Applies a move to a position -/
def applyMove (p : Position) (m : Move) : Position :=
  match m with
  | Move.up    => ⟨(p.row + 1) % 4, p.col⟩
  | Move.down  => ⟨(p.row - 1 + 4) % 4, p.col⟩
  | Move.left  => ⟨p.row, (p.col - 1 + 4) % 4⟩
  | Move.right => ⟨p.row, (p.col + 1) % 4⟩

/-- Calculates the probability of reaching an edge within n hops -/
def probReachEdge (n : Nat) : ℚ :=
  sorry

theorem frieda_prob_reach_edge :
  probReachEdge 3 = 5/8 :=
sorry

end NUMINAMATH_CALUDE_frieda_prob_reach_edge_l1436_143648


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1436_143611

theorem polynomial_division_remainder (x : ℝ) :
  ∃ (Q : ℝ → ℝ) (S : ℝ → ℝ),
    (∀ x, x^50 = (x^2 - 5*x + 6) * Q x + S x) ∧
    (∃ a b : ℝ, ∀ x, S x = a * x + b) ∧
    S x = (3^50 - 2^50) * x + (4^50 - 6^50) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1436_143611


namespace NUMINAMATH_CALUDE_quadratic_sum_l1436_143622

theorem quadratic_sum (a b : ℝ) : 
  ({b} : Set ℝ) = {x : ℝ | a * x^2 - 4 * x + 1 = 0} → 
  a + b = 1/4 ∨ a + b = 9/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1436_143622


namespace NUMINAMATH_CALUDE_quadratic_decreasing_interval_l1436_143660

/-- A quadratic function f(x) = x^2 + bx + c -/
def quadratic (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- The derivative of a quadratic function -/
def quadratic_derivative (b : ℝ) (x : ℝ) : ℝ := 2*x + b

theorem quadratic_decreasing_interval (b c : ℝ) :
  (∀ x ≤ 1, quadratic_derivative b x ≤ 0) →
  (∃ x > 1, quadratic_derivative b x > 0) →
  b = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_interval_l1436_143660


namespace NUMINAMATH_CALUDE_tangent_line_at_2_g_unique_minimum_l1436_143636

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := x^2 / (1 + x) + 1 / x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f x - 1 / x - a * Real.log x

-- Statement 1: Tangent line equation
theorem tangent_line_at_2 :
  ∃ (m b : ℝ), ∀ x y, y = m * x + b ↔ 23 * x - 36 * y + 20 = 0 :=
sorry

-- Statement 2: Unique minimum point of g
theorem g_unique_minimum (a : ℝ) (h : a > 0) :
  ∃! x, x > 0 ∧ ∀ y, y > 0 → g a y ≥ g a x :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_g_unique_minimum_l1436_143636


namespace NUMINAMATH_CALUDE_sqrt_equality_condition_l1436_143673

theorem sqrt_equality_condition (a b c : ℕ+) :
  (Real.sqrt (a + b / (c ^ 2 : ℝ)) = a * Real.sqrt (b / (c ^ 2 : ℝ))) ↔ 
  (c ^ 2 : ℝ) = b * (a ^ 2 - 1) / a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_condition_l1436_143673


namespace NUMINAMATH_CALUDE_ed_doug_marble_difference_l1436_143654

-- Define the initial number of marbles for Ed and Doug
def ed_marbles : ℕ := 45
def doug_initial_marbles : ℕ := ed_marbles - 10

-- Define the number of marbles Doug lost
def doug_lost_marbles : ℕ := 11

-- Define Doug's final number of marbles
def doug_final_marbles : ℕ := doug_initial_marbles - doug_lost_marbles

-- Theorem statement
theorem ed_doug_marble_difference :
  ed_marbles - doug_final_marbles = 21 :=
by sorry

end NUMINAMATH_CALUDE_ed_doug_marble_difference_l1436_143654


namespace NUMINAMATH_CALUDE_specific_hexagon_area_l1436_143661

/-- An irregular hexagon in 2D space -/
structure IrregularHexagon where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ
  v5 : ℝ × ℝ
  v6 : ℝ × ℝ

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Calculate the area of an irregular hexagon -/
def hexagonArea (h : IrregularHexagon) : ℝ := sorry

/-- The specific irregular hexagon from the problem -/
def specificHexagon : IrregularHexagon :=
  { v1 := (0, 0)
  , v2 := (2, 4)
  , v3 := (5, 4)
  , v4 := (7, 0)
  , v5 := (5, -4)
  , v6 := (2, -4) }

/-- Theorem: The area of the specific irregular hexagon is 32 square units -/
theorem specific_hexagon_area :
  hexagonArea specificHexagon = 32 := by sorry

end NUMINAMATH_CALUDE_specific_hexagon_area_l1436_143661


namespace NUMINAMATH_CALUDE_ribbon_ratio_l1436_143697

theorem ribbon_ratio : 
  ∀ (original reduced : ℕ), 
  original = 55 → reduced = 35 → 
  (original : ℚ) / (reduced : ℚ) = 11 / 7 := by
sorry

end NUMINAMATH_CALUDE_ribbon_ratio_l1436_143697


namespace NUMINAMATH_CALUDE_kristoff_sticker_count_l1436_143642

/-- The number of stickers Riku has -/
def riku_stickers : ℕ := 2210

/-- The ratio of Riku's stickers to Kristoff's stickers -/
def sticker_ratio : ℕ := 25

/-- The number of stickers Kristoff has -/
def kristoff_stickers : ℕ := riku_stickers / sticker_ratio

theorem kristoff_sticker_count : kristoff_stickers = 88 := by
  sorry

end NUMINAMATH_CALUDE_kristoff_sticker_count_l1436_143642


namespace NUMINAMATH_CALUDE_students_with_two_skills_l1436_143664

theorem students_with_two_skills (total : ℕ) (cant_paint cant_write cant_music : ℕ) : 
  total = 150 →
  cant_paint = 75 →
  cant_write = 90 →
  cant_music = 45 →
  ∃ (two_skills : ℕ), two_skills = 90 ∧ 
    two_skills = (total - cant_paint) + (total - cant_write) + (total - cant_music) - total :=
by sorry

end NUMINAMATH_CALUDE_students_with_two_skills_l1436_143664


namespace NUMINAMATH_CALUDE_identity_function_theorem_l1436_143601

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

theorem identity_function_theorem (f : ℕ+ → ℕ+) : 
  (∀ x y : ℕ+, is_perfect_square (x * f x + 2 * x * f y + (f y) ^ 2)) → 
  (∀ x : ℕ+, f x = x) :=
sorry

end NUMINAMATH_CALUDE_identity_function_theorem_l1436_143601


namespace NUMINAMATH_CALUDE_perpendicular_equal_diagonals_not_sufficient_for_square_l1436_143617

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define properties of quadrilaterals
def has_perpendicular_diagonals (q : Quadrilateral) : Prop := sorry
def has_equal_diagonals (q : Quadrilateral) : Prop := sorry
def is_square (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem perpendicular_equal_diagonals_not_sufficient_for_square :
  ∃ (q : Quadrilateral), has_perpendicular_diagonals q ∧ has_equal_diagonals q ∧ ¬is_square q :=
sorry

end NUMINAMATH_CALUDE_perpendicular_equal_diagonals_not_sufficient_for_square_l1436_143617


namespace NUMINAMATH_CALUDE_radio_operator_distribution_probability_l1436_143603

theorem radio_operator_distribution_probability :
  let total_soldiers : ℕ := 12
  let radio_operators : ℕ := 3
  let group_sizes : List ℕ := [3, 4, 5]
  
  let total_distributions : ℕ := (total_soldiers.choose group_sizes[0]!) * ((total_soldiers - group_sizes[0]!).choose group_sizes[1]!) * 1
  
  let favorable_distributions : ℕ := ((total_soldiers - radio_operators).choose (group_sizes[0]! - 1)) *
    ((total_soldiers - radio_operators - (group_sizes[0]! - 1)).choose (group_sizes[1]! - 1)) * 
    ((radio_operators).factorial)
  
  (favorable_distributions : ℚ) / total_distributions = 3 / 11 := by
  sorry

end NUMINAMATH_CALUDE_radio_operator_distribution_probability_l1436_143603


namespace NUMINAMATH_CALUDE_chord_length_polar_curve_l1436_143663

/-- The length of the chord AB, where A is the point (3, 0) and B is the other intersection point
    of the line x = 3 with the curve ρ = 4cosθ in polar coordinates. -/
theorem chord_length_polar_curve : ∃ (A B : ℝ × ℝ),
  A = (3, 0) ∧
  B.1 = 3 ∧
  (B.1 - 2)^2 + B.2^2 = 4 ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_polar_curve_l1436_143663


namespace NUMINAMATH_CALUDE_choose_four_from_ten_l1436_143698

theorem choose_four_from_ten : Nat.choose 10 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_ten_l1436_143698


namespace NUMINAMATH_CALUDE_family_d_members_l1436_143652

/-- Represents the number of members in each family -/
structure FamilyMembers where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ

/-- The initial number of members in each family -/
def initial : FamilyMembers :=
  { a := 7
    b := 8
    c := 10
    d := 13  -- This is what we want to prove
    e := 6
    f := 10 }

/-- The number of families -/
def numFamilies : ℕ := 6

/-- The number of members who left each family -/
def membersLeft : ℕ := 1

/-- The average number of members after some left -/
def newAverage : ℕ := 8

/-- Theorem: The initial number of members in family d is 13 -/
theorem family_d_members : initial.d = 13 := by sorry

end NUMINAMATH_CALUDE_family_d_members_l1436_143652


namespace NUMINAMATH_CALUDE_younger_person_age_l1436_143662

/-- Given two people with an age difference of 20 years, where 15 years ago the elder was twice as old as the younger, prove that the younger person's present age is 35 years. -/
theorem younger_person_age (younger elder : ℕ) : 
  elder - younger = 20 →
  elder - 15 = 2 * (younger - 15) →
  younger = 35 := by
  sorry

end NUMINAMATH_CALUDE_younger_person_age_l1436_143662


namespace NUMINAMATH_CALUDE_restaurant_meals_count_l1436_143649

theorem restaurant_meals_count (kids_meals : ℕ) (adult_meals : ℕ) : 
  kids_meals = 8 → 
  2 * adult_meals = kids_meals → 
  kids_meals + adult_meals = 12 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_meals_count_l1436_143649


namespace NUMINAMATH_CALUDE_olivias_initial_amount_l1436_143677

/-- The amount of money Olivia had in her wallet initially -/
def initial_amount : ℕ := sorry

/-- The amount Olivia spent at the supermarket -/
def supermarket_expense : ℕ := 31

/-- The amount Olivia spent at the showroom -/
def showroom_expense : ℕ := 49

/-- The amount Olivia had left after spending -/
def remaining_amount : ℕ := 26

/-- Theorem stating that Olivia's initial amount was $106 -/
theorem olivias_initial_amount : 
  initial_amount = supermarket_expense + showroom_expense + remaining_amount := by sorry

end NUMINAMATH_CALUDE_olivias_initial_amount_l1436_143677


namespace NUMINAMATH_CALUDE_beaver_carrots_l1436_143692

theorem beaver_carrots :
  ∀ (beaver_burrows rabbit_burrows : ℕ),
    beaver_burrows = rabbit_burrows + 5 →
    5 * beaver_burrows = 7 * rabbit_burrows →
    5 * beaver_burrows = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_beaver_carrots_l1436_143692


namespace NUMINAMATH_CALUDE_circle_origin_inside_l1436_143653

theorem circle_origin_inside (m : ℝ) : 
  (∀ x y : ℝ, (x - m)^2 + (y + m)^2 = 8 → (0 : ℝ)^2 + (0 : ℝ)^2 < 8) → 
  -2 < m ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_origin_inside_l1436_143653


namespace NUMINAMATH_CALUDE_philip_banana_count_l1436_143638

/-- The number of banana groups in Philip's collection -/
def banana_groups : ℕ := 7

/-- The number of bananas in each group -/
def bananas_per_group : ℕ := 29

/-- The total number of bananas in Philip's collection -/
def total_bananas : ℕ := banana_groups * bananas_per_group

/-- Theorem stating that the total number of bananas is 203 -/
theorem philip_banana_count : total_bananas = 203 := by
  sorry

end NUMINAMATH_CALUDE_philip_banana_count_l1436_143638


namespace NUMINAMATH_CALUDE_taylor_family_reunion_l1436_143606

theorem taylor_family_reunion (kids : ℕ) (adults : ℕ) (tables : ℕ) 
  (h1 : kids = 45) 
  (h2 : adults = 123) 
  (h3 : tables = 14) : 
  (kids + adults) / tables = 12 := by
sorry

end NUMINAMATH_CALUDE_taylor_family_reunion_l1436_143606


namespace NUMINAMATH_CALUDE_factors_of_M_l1436_143645

/-- The number of natural-number factors of M, where M = 2^4 · 3^3 · 5^2 · 7^1 -/
def number_of_factors (M : ℕ) : ℕ :=
  (4 + 1) * (3 + 1) * (2 + 1) * (1 + 1)

/-- Theorem stating that the number of natural-number factors of M is 120 -/
theorem factors_of_M :
  let M : ℕ := 2^4 * 3^3 * 5^2 * 7^1
  number_of_factors M = 120 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_M_l1436_143645


namespace NUMINAMATH_CALUDE_g_of_8_equals_69_l1436_143666

-- Define the function g
def g (n : ℤ) : ℤ := n^2 - 3*n + 29

-- State the theorem
theorem g_of_8_equals_69 : g 8 = 69 := by
  sorry

end NUMINAMATH_CALUDE_g_of_8_equals_69_l1436_143666


namespace NUMINAMATH_CALUDE_cookies_per_box_l1436_143628

/-- The number of cookies Basil consumes per day -/
def cookies_per_day : ℚ := 1/2 + 1/2 + 2

/-- The number of days Basil's cookies should last -/
def days : ℕ := 30

/-- The number of boxes needed for the given number of days -/
def boxes : ℕ := 2

/-- Theorem stating the number of cookies in each box -/
theorem cookies_per_box : 
  (cookies_per_day * days) / boxes = 45 := by sorry

end NUMINAMATH_CALUDE_cookies_per_box_l1436_143628


namespace NUMINAMATH_CALUDE_min_nSn_l1436_143634

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  h1 : a 5 = 3  -- a_5 = 3
  h2 : S 10 = 40  -- S_10 = 40

/-- The property that the sequence is arithmetic -/
def isArithmetic (seq : ArithmeticSequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, seq.a (n + 1) = seq.a n + d

/-- The sum function definition -/
def sumProperty (seq : ArithmeticSequence) : Prop :=
  ∀ n : ℕ, seq.S n = (n : ℝ) * (seq.a 1 + seq.a n) / 2

/-- The main theorem -/
theorem min_nSn (seq : ArithmeticSequence) 
  (hArith : isArithmetic seq) (hSum : sumProperty seq) : 
  ∃ m : ℝ, m = -32 ∧ ∀ n : ℕ, (n : ℝ) * seq.S n ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_nSn_l1436_143634


namespace NUMINAMATH_CALUDE_sqrt_7_simplest_l1436_143671

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℝ, y ≥ 0 → (∃ n : ℕ, y = n ^ 2 * x) → y = x

theorem sqrt_7_simplest :
  is_simplest_quadratic_radical 7 ∧
  ¬ is_simplest_quadratic_radical 9 ∧
  ¬ is_simplest_quadratic_radical 12 ∧
  ¬ is_simplest_quadratic_radical (2/3) :=
sorry

end NUMINAMATH_CALUDE_sqrt_7_simplest_l1436_143671


namespace NUMINAMATH_CALUDE_allison_wins_prob_l1436_143600

/-- Represents a 6-sided cube with specific face configurations -/
structure Cube where
  faces : Fin 6 → ℕ

/-- Allison's cube configuration -/
def allison_cube : Cube :=
  { faces := λ i => if i.val < 3 then 3 else 4 }

/-- Brian's cube configuration -/
def brian_cube : Cube :=
  { faces := λ i => i.val }

/-- Noah's cube configuration -/
def noah_cube : Cube :=
  { faces := λ i => if i.val < 3 then 2 else 6 }

/-- Probability of rolling a specific value on a cube -/
def prob_roll (c : Cube) (v : ℕ) : ℚ :=
  (Finset.filter (λ i => c.faces i = v) (Finset.univ : Finset (Fin 6))).card / 6

/-- Probability of rolling less than a value on a cube -/
def prob_roll_less (c : Cube) (v : ℕ) : ℚ :=
  (Finset.filter (λ i => c.faces i < v) (Finset.univ : Finset (Fin 6))).card / 6

/-- The main theorem stating the probability of Allison winning -/
theorem allison_wins_prob :
  (1 / 2) * (prob_roll_less brian_cube 3 * prob_roll_less noah_cube 3 +
             prob_roll_less brian_cube 4 * prob_roll_less noah_cube 4) = 7 / 24 := by
  sorry

#check allison_wins_prob

end NUMINAMATH_CALUDE_allison_wins_prob_l1436_143600


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1436_143613

theorem complex_number_in_fourth_quadrant :
  let i : ℂ := Complex.I
  let z : ℂ := (2 * i^3) / (1 - i)
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1436_143613


namespace NUMINAMATH_CALUDE_power_function_sum_l1436_143669

/-- A power function passing through (4, 2) has k + a = 3/2 --/
theorem power_function_sum (k a : ℝ) : 
  (∀ x : ℝ, x > 0 → ∃ f : ℝ → ℝ, f x = k * x^a) → 
  k * 4^a = 2 → 
  k + a = 3/2 := by sorry

end NUMINAMATH_CALUDE_power_function_sum_l1436_143669


namespace NUMINAMATH_CALUDE_banana_pancakes_count_l1436_143658

/-- The number of banana pancakes given the total, blueberry, and plain pancake counts. -/
def banana_pancakes (total blueberry plain : ℕ) : ℕ :=
  total - blueberry - plain

/-- Theorem stating that the number of banana pancakes is 24 given the specific counts. -/
theorem banana_pancakes_count :
  banana_pancakes 67 20 23 = 24 := by
  sorry

end NUMINAMATH_CALUDE_banana_pancakes_count_l1436_143658


namespace NUMINAMATH_CALUDE_octagon_area_l1436_143619

/-- The area of an octagon inscribed in a rectangle --/
theorem octagon_area (rectangle_width rectangle_height triangle_base triangle_height : ℝ) 
  (hw : rectangle_width = 5)
  (hh : rectangle_height = 8)
  (htb : triangle_base = 1)
  (hth : triangle_height = 4) :
  rectangle_width * rectangle_height - 4 * (1/2 * triangle_base * triangle_height) = 32 :=
by sorry

end NUMINAMATH_CALUDE_octagon_area_l1436_143619


namespace NUMINAMATH_CALUDE_chastity_money_left_l1436_143687

/-- The amount of money Chastity was left with after buying lollipops and gummies -/
def money_left (initial_amount : ℝ) (lollipop_price : ℝ) (lollipop_count : ℕ) 
                (gummy_price : ℝ) (gummy_count : ℕ) : ℝ :=
  initial_amount - (lollipop_price * lollipop_count + gummy_price * gummy_count)

/-- Theorem stating that Chastity was left with $5 after her candy purchase -/
theorem chastity_money_left : 
  money_left 15 1.5 4 2 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_chastity_money_left_l1436_143687


namespace NUMINAMATH_CALUDE_grocery_spending_l1436_143699

theorem grocery_spending (X : ℚ) : 
  X > 0 → X - 3 - 2 - (1/3)*(X - 5) = 18 → X = 32 := by
  sorry

end NUMINAMATH_CALUDE_grocery_spending_l1436_143699


namespace NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_six_l1436_143670

theorem no_linear_term_implies_m_equals_six (m : ℝ) : 
  (∀ x : ℝ, (2*x + m) * (x - 3) = 2*x^2 - 3*m) → m = 6 := by
sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_six_l1436_143670


namespace NUMINAMATH_CALUDE_factorization_4m_squared_minus_16_l1436_143667

theorem factorization_4m_squared_minus_16 (m : ℝ) :
  4 * m^2 - 16 = 4 * (m + 2) * (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_4m_squared_minus_16_l1436_143667


namespace NUMINAMATH_CALUDE_homework_question_count_l1436_143640

/-- Calculates the number of true/false questions in a homework assignment -/
theorem homework_question_count (total : ℕ) (mc_ratio : ℕ) (fr_diff : ℕ) (h1 : total = 45) (h2 : mc_ratio = 2) (h3 : fr_diff = 7) : 
  ∃ (tf : ℕ) (fr : ℕ) (mc : ℕ), 
    tf + fr + mc = total ∧ 
    mc = mc_ratio * fr ∧ 
    fr = tf + fr_diff ∧ 
    tf = 6 := by
  sorry

end NUMINAMATH_CALUDE_homework_question_count_l1436_143640


namespace NUMINAMATH_CALUDE_melanie_football_games_l1436_143672

theorem melanie_football_games (total_games missed_games : ℕ) 
  (h1 : total_games = 7)
  (h2 : missed_games = 4) :
  total_games - missed_games = 3 := by
  sorry

end NUMINAMATH_CALUDE_melanie_football_games_l1436_143672


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l1436_143602

theorem sum_of_roots_equation (x : ℝ) : 
  let eq := (3*x + 4)*(x - 3) + (3*x + 4)*(x - 5) = 0
  ∃ (r₁ r₂ : ℝ), (3*r₁ + 4)*(r₁ - 3) + (3*r₁ + 4)*(r₁ - 5) = 0 ∧
                 (3*r₂ + 4)*(r₂ - 3) + (3*r₂ + 4)*(r₂ - 5) = 0 ∧
                 r₁ + r₂ = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l1436_143602


namespace NUMINAMATH_CALUDE_inequality_proof_l1436_143647

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  Real.sqrt (a^2 + 1/a) + Real.sqrt (b^2 + 1/b) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1436_143647


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l1436_143668

theorem max_value_cos_sin (θ : Real) (h : 0 ≤ θ ∧ θ ≤ Real.pi / 2) :
  (Real.cos (θ / 2))^2 * (1 - Real.sin θ) ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l1436_143668


namespace NUMINAMATH_CALUDE_friend_lunch_cost_l1436_143690

theorem friend_lunch_cost (total : ℝ) (difference : ℝ) (friend_cost : ℝ) : 
  total = 15 → difference = 5 → friend_cost = total / 2 + difference / 2 → friend_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_friend_lunch_cost_l1436_143690


namespace NUMINAMATH_CALUDE_remainder_3n_div_7_l1436_143681

theorem remainder_3n_div_7 (n : Int) (h : n % 7 = 1) : (3 * n) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3n_div_7_l1436_143681


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l1436_143605

/-- The number of possible starting lineups for a basketball team --/
theorem basketball_lineup_combinations (total_players : ℕ) 
  (guaranteed_players : ℕ) (excluded_players : ℕ) (lineup_size : ℕ) : 
  total_players = 15 → 
  guaranteed_players = 2 → 
  excluded_players = 1 → 
  lineup_size = 6 → 
  Nat.choose (total_players - guaranteed_players - excluded_players) 
             (lineup_size - guaranteed_players) = 495 := by
  sorry

#check basketball_lineup_combinations

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l1436_143605


namespace NUMINAMATH_CALUDE_safe_lock_configuration_l1436_143632

/-- The number of commission members -/
def n : ℕ := 9

/-- The minimum number of members required to access the safe -/
def k : ℕ := 6

/-- The number of keys for each lock -/
def keys_per_lock : ℕ := n - k + 1

/-- The number of locks needed for the safe -/
def num_locks : ℕ := Nat.choose n (n - k + 1)

theorem safe_lock_configuration :
  num_locks = 126 ∧ keys_per_lock = 4 :=
sorry

end NUMINAMATH_CALUDE_safe_lock_configuration_l1436_143632


namespace NUMINAMATH_CALUDE_f_equals_n_plus_one_f_1993_l1436_143682

def N₀ : Set ℕ := {n : ℕ | True}

def is_valid_f (f : ℕ → ℕ) : Prop :=
  ∀ n, f (f n) + f n = 2 * n + 3

theorem f_equals_n_plus_one (f : ℕ → ℕ) (h : is_valid_f f) :
  ∀ n, f n = n + 1 :=
by
  sorry

-- The original question can be answered as a corollary
theorem f_1993 (f : ℕ → ℕ) (h : is_valid_f f) :
  f 1993 = 1994 :=
by
  sorry

end NUMINAMATH_CALUDE_f_equals_n_plus_one_f_1993_l1436_143682


namespace NUMINAMATH_CALUDE_p_is_third_degree_trinomial_l1436_143691

-- Define the polynomial
def p (x y : ℝ) : ℝ := 2 * x^2 - 3 * x * y + 5 * x * y^2

-- Theorem statement
theorem p_is_third_degree_trinomial :
  (∃ (a b c : ℝ) (f g h : ℕ → ℕ → ℕ), 
    (∀ x y, p x y = a * x^(f 0 0) * y^(f 0 1) + b * x^(g 0 0) * y^(g 0 1) + c * x^(h 0 0) * y^(h 0 1)) ∧
    (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧
    (max (f 0 0 + f 0 1) (max (g 0 0 + g 0 1) (h 0 0 + h 0 1)) = 3)) :=
by sorry


end NUMINAMATH_CALUDE_p_is_third_degree_trinomial_l1436_143691


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1436_143675

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) :
  z.im = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1436_143675


namespace NUMINAMATH_CALUDE_tangent_line_power_function_l1436_143612

theorem tangent_line_power_function (n : ℝ) :
  (2 : ℝ) ^ n = 8 →
  let f := λ x : ℝ => x ^ n
  let f' := λ x : ℝ => n * x ^ (n - 1)
  let tangent_slope := f' 2
  let tangent_eq := λ x y : ℝ => tangent_slope * (x - 2) = y - 8
  tangent_eq = λ x y : ℝ => 12 * x - y - 16 = 0 := by sorry

end NUMINAMATH_CALUDE_tangent_line_power_function_l1436_143612


namespace NUMINAMATH_CALUDE_find_x_l1436_143639

-- Define the binary operation ★
def star (a b c d : ℤ) : ℤ × ℤ := (a + c, b - 2*d)

-- Theorem statement
theorem find_x : ∀ x y : ℤ, star (x + 1) (y - 1) 1 3 = (2, -4) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l1436_143639


namespace NUMINAMATH_CALUDE_total_baseball_cards_l1436_143618

theorem total_baseball_cards : 
  let number_of_people : ℕ := 6
  let cards_per_person : ℕ := 8
  number_of_people * cards_per_person = 48 :=
by sorry

end NUMINAMATH_CALUDE_total_baseball_cards_l1436_143618


namespace NUMINAMATH_CALUDE_ratio_problem_l1436_143641

/-- Custom operation @ for positive integers -/
def custom_op (k j : ℕ+) : ℕ+ :=
  sorry

theorem ratio_problem (a b : ℕ+) (t : ℚ) : 
  a = 2020 → t = (a : ℚ) / (b : ℚ) → t = 1/2 → b = 4040 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1436_143641


namespace NUMINAMATH_CALUDE_min_omega_value_l1436_143680

/-- Given that ω > 0 and the graph of y = 2cos(ωx + π/5) - 1 overlaps with itself
    after shifting right by 5π/4 units, prove that the minimum value of ω is 8/5. -/
theorem min_omega_value (ω : ℝ) (h1 : ω > 0)
  (h2 : ∀ x : ℝ, 2 * Real.cos (ω * x + π / 5) - 1 = 2 * Real.cos (ω * (x + 5 * π / 4) + π / 5) - 1) :
  ω ≥ 8 / 5 ∧ ∀ ω' > 0, (∀ x : ℝ, 2 * Real.cos (ω' * x + π / 5) - 1 = 2 * Real.cos (ω' * (x + 5 * π / 4) + π / 5) - 1) → ω' ≥ ω :=
by sorry

end NUMINAMATH_CALUDE_min_omega_value_l1436_143680


namespace NUMINAMATH_CALUDE_vector_operations_l1436_143695

/-- Given vectors in ℝ², prove that they are not collinear, find the cosine of the angle between them, and calculate the projection of one vector onto another. -/
theorem vector_operations (a b c : ℝ × ℝ) (h1 : a = (-1, 1)) (h2 : b = (4, 3)) (h3 : c = (5, -2)) :
  ¬ (∃ k : ℝ, a = k • b) ∧
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = -Real.sqrt 2 / 10 ∧
  ((a.1 * c.1 + a.2 * c.2) / (a.1^2 + a.2^2)) • a = (7/2 * Real.sqrt 2) • (-1, 1) :=
by sorry

end NUMINAMATH_CALUDE_vector_operations_l1436_143695


namespace NUMINAMATH_CALUDE_wheel_turns_time_l1436_143629

theorem wheel_turns_time (turns_per_two_hours : ℕ) (h : turns_per_two_hours = 1440) :
  (6 : ℝ) * (3600 : ℝ) / (turns_per_two_hours : ℝ) * 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_wheel_turns_time_l1436_143629


namespace NUMINAMATH_CALUDE_seating_arrangements_l1436_143643

/-- Represents the number of seats in a row -/
def total_seats : ℕ := 12

/-- Represents the number of people to be seated -/
def num_people : ℕ := 3

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Represents the number of possible arrangements of A between the other two people -/
def a_between_arrangements : ℕ := 2

/-- Represents the number of empty seats after arranging people and mandatory empty seats -/
def remaining_empty_seats : ℕ := 8

/-- Represents the number of empty seats to be chosen from remaining empty seats -/
def seats_to_choose : ℕ := 5

/-- The main theorem stating the total number of seating arrangements -/
theorem seating_arrangements :
  a_between_arrangements * choose remaining_empty_seats seats_to_choose = 112 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l1436_143643


namespace NUMINAMATH_CALUDE_fraction_simplification_l1436_143646

theorem fraction_simplification :
  (1 : ℚ) / 330 + 19 / 30 = 7 / 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1436_143646


namespace NUMINAMATH_CALUDE_no_function_exists_l1436_143644

theorem no_function_exists : ¬∃ (f : ℤ → ℤ), ∀ (x y z : ℤ), f (x * y) + f (x * z) - f x * f (y * z) ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_no_function_exists_l1436_143644


namespace NUMINAMATH_CALUDE_tetrahedron_edge_length_is_sqrt_2_l1436_143683

/-- Represents a cube with unit side length -/
structure UnitCube where
  center : ℝ × ℝ × ℝ

/-- Represents a tetrahedron circumscribed around four unit cubes -/
structure Tetrahedron where
  cubes : Fin 4 → UnitCube

/-- The edge length of the tetrahedron -/
def tetrahedron_edge_length (t : Tetrahedron) : ℝ := sorry

/-- The configuration of four unit cubes as described in the problem -/
def cube_configuration : Tetrahedron := sorry

theorem tetrahedron_edge_length_is_sqrt_2 :
  tetrahedron_edge_length cube_configuration = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_edge_length_is_sqrt_2_l1436_143683
