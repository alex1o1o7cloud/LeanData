import Mathlib

namespace NUMINAMATH_CALUDE_digit_sum_problem_l2818_281840

theorem digit_sum_problem (a p v e s r : ℕ) 
  (h1 : a + p = v)
  (h2 : v + e = s)
  (h3 : s + a = r)
  (h4 : p + e + r = 14)
  (h5 : a ≠ 0 ∧ p ≠ 0 ∧ v ≠ 0 ∧ e ≠ 0 ∧ s ≠ 0 ∧ r ≠ 0) :
  s = 7 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l2818_281840


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_391_l2818_281864

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_next_divisor_after_391 (m : ℕ) 
  (h1 : is_even m) 
  (h2 : is_four_digit m) 
  (h3 : m % 391 = 0) : 
  ∃ d : ℕ, d > 391 ∧ m % d = 0 ∧ (∀ k : ℕ, 391 < k ∧ k < d → m % k ≠ 0) → d = 782 :=
sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_391_l2818_281864


namespace NUMINAMATH_CALUDE_paint_per_statue_l2818_281841

theorem paint_per_statue (total_paint : ℚ) (num_statues : ℕ) 
  (h1 : total_paint = 7/8)
  (h2 : num_statues = 7) : 
  total_paint / num_statues = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_paint_per_statue_l2818_281841


namespace NUMINAMATH_CALUDE_penny_revenue_l2818_281839

/-- Calculates the total money earned from selling cheesecake pies -/
def cheesecake_revenue (price_per_slice : ℕ) (slices_per_pie : ℕ) (pies_sold : ℕ) : ℕ :=
  price_per_slice * slices_per_pie * pies_sold

/-- Proves that Penny makes $294 from selling 7 cheesecake pies -/
theorem penny_revenue : cheesecake_revenue 7 6 7 = 294 := by
  sorry

end NUMINAMATH_CALUDE_penny_revenue_l2818_281839


namespace NUMINAMATH_CALUDE_planes_perpendicular_l2818_281876

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel, perpendicular, and subset relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular 
  (m n : Line) (α β : Plane)
  (h1 : parallel m n)
  (h2 : perpendicular n β)
  (h3 : subset m α) :
  perp_planes α β := by sorry

end NUMINAMATH_CALUDE_planes_perpendicular_l2818_281876


namespace NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l2818_281842

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of Quadrant I -/
def inQuadrantI (p : Point2D) : Prop := p.x > 0 ∧ p.y > 0

/-- Definition of Quadrant II -/
def inQuadrantII (p : Point2D) : Prop := p.x < 0 ∧ p.y > 0

/-- The set of points satisfying the given inequalities -/
def satisfiesInequalities (p : Point2D) : Prop :=
  p.y > 3 * p.x ∧ p.y > 6 - 2 * p.x

theorem points_in_quadrants_I_and_II :
  ∀ p : Point2D, satisfiesInequalities p → inQuadrantI p ∨ inQuadrantII p :=
by sorry

end NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l2818_281842


namespace NUMINAMATH_CALUDE_patio_rows_l2818_281865

theorem patio_rows (r c : ℕ) : 
  r * c = 30 →
  (r + 4) * (c - 2) = 30 →
  r = 3 :=
by sorry

end NUMINAMATH_CALUDE_patio_rows_l2818_281865


namespace NUMINAMATH_CALUDE_leet_puzzle_solution_l2818_281847

theorem leet_puzzle_solution :
  ∀ (L E T M : ℕ),
    L ≠ 0 →
    L < 10 ∧ E < 10 ∧ T < 10 ∧ M < 10 →
    1000 * L + 110 * E + T + 100 * L + 10 * M + T = 1000 * T + L →
    T = L + 1 →
    1000 * E + 100 * L + 10 * M + 0 = 1880 :=
by
  sorry

end NUMINAMATH_CALUDE_leet_puzzle_solution_l2818_281847


namespace NUMINAMATH_CALUDE_ratio_problem_l2818_281813

theorem ratio_problem (a b c d : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0)
  (h5 : a / b = 1 / 4)
  (h6 : c / d = 5 / 13)
  (h7 : a / d = 0.1388888888888889) :
  b / c = 13 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2818_281813


namespace NUMINAMATH_CALUDE_domain_of_composite_function_l2818_281844

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Ioo (-1) 1

-- State the theorem
theorem domain_of_composite_function :
  {x : ℝ | ∃ y ∈ domain_f, y = 2*x + 1} = Set.Ioo (-1) 0 := by sorry

end NUMINAMATH_CALUDE_domain_of_composite_function_l2818_281844


namespace NUMINAMATH_CALUDE_cubic_difference_l2818_281819

theorem cubic_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) :
  x^3 - y^3 = 176 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_l2818_281819


namespace NUMINAMATH_CALUDE_smallest_number_theorem_l2818_281826

def is_multiple_of_36 (n : ℕ) : Prop := ∃ k : ℕ, n = 36 * k

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_multiple_of_36 n ∧ (digit_product n % 9 = 0)

theorem smallest_number_theorem :
  satisfies_conditions 936 ∧ ∀ m : ℕ, m < 936 → ¬(satisfies_conditions m) :=
sorry

end NUMINAMATH_CALUDE_smallest_number_theorem_l2818_281826


namespace NUMINAMATH_CALUDE_robot_rascals_shipment_l2818_281897

theorem robot_rascals_shipment (total : ℝ) : 
  (0.7 * total = 168) → total = 240 := by
  sorry

end NUMINAMATH_CALUDE_robot_rascals_shipment_l2818_281897


namespace NUMINAMATH_CALUDE_simplify_expression_l2818_281895

theorem simplify_expression (p : ℝ) : 
  ((7 * p + 3) - 3 * p * 2) * 4 + (5 - 2 / 4) * (8 * p - 12) = 40 * p - 42 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2818_281895


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2818_281875

def U : Set ℕ := {1, 2, 3, 4}

def A : Set ℕ := {x : ℕ | x ^ 2 - 5 * x + 4 < 0}

theorem complement_of_A_in_U :
  U \ A = {1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2818_281875


namespace NUMINAMATH_CALUDE_quartic_polynomial_extrema_bounds_l2818_281878

/-- A polynomial of degree 4 with real coefficients -/
def QuarticPolynomial (a₀ a₁ a₂ : ℝ) : ℝ → ℝ := λ x ↦ x^4 + a₁*x^3 + a₂*x^2 + a₁*x + a₀

/-- The local maximum of a function -/
noncomputable def LocalMax (f : ℝ → ℝ) : ℝ := sorry

/-- The local minimum of a function -/
noncomputable def LocalMin (f : ℝ → ℝ) : ℝ := sorry

/-- Theorem: Bounds for the difference between local maximum and minimum of a quartic polynomial -/
theorem quartic_polynomial_extrema_bounds (a₀ a₁ a₂ : ℝ) :
  let f := QuarticPolynomial a₀ a₁ a₂
  let M := LocalMax f
  let m := LocalMin f
  3/10 * (a₁^2/4 - 2*a₂/9)^2 < M - m ∧ M - m < 3 * (a₁^2/4 - 2*a₂/9)^2 := by
  sorry

end NUMINAMATH_CALUDE_quartic_polynomial_extrema_bounds_l2818_281878


namespace NUMINAMATH_CALUDE_longer_strap_length_l2818_281884

theorem longer_strap_length (short long : ℕ) : 
  long = short + 72 →
  short + long = 348 →
  long = 210 :=
by sorry

end NUMINAMATH_CALUDE_longer_strap_length_l2818_281884


namespace NUMINAMATH_CALUDE_peaches_at_stand_l2818_281837

/-- The total number of peaches at Sally's stand after picking more -/
def total_peaches (initial : ℕ) (picked : ℕ) : ℕ :=
  initial + picked

/-- Theorem stating that the total number of peaches is 55 given the initial and picked amounts -/
theorem peaches_at_stand (initial : ℕ) (picked : ℕ) 
  (h1 : initial = 13) (h2 : picked = 42) : 
  total_peaches initial picked = 55 := by
  sorry

end NUMINAMATH_CALUDE_peaches_at_stand_l2818_281837


namespace NUMINAMATH_CALUDE_product_equals_two_l2818_281855

theorem product_equals_two : 
  (∀ (a b c : ℝ), a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) →
  6 * 15 * 5 = 2 :=
by sorry

end NUMINAMATH_CALUDE_product_equals_two_l2818_281855


namespace NUMINAMATH_CALUDE_difference_of_squares_fraction_l2818_281804

theorem difference_of_squares_fraction : (235^2 - 221^2) / 14 = 456 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_fraction_l2818_281804


namespace NUMINAMATH_CALUDE_division_problem_l2818_281807

theorem division_problem (total : ℕ) (a b c : ℕ) : 
  total = 770 →
  a = b + 40 →
  c = a + 30 →
  total = a + b + c →
  b = 220 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2818_281807


namespace NUMINAMATH_CALUDE_faye_flowers_proof_l2818_281808

theorem faye_flowers_proof (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) (remaining_bouquets : ℕ) 
  (h1 : flowers_per_bouquet = 5)
  (h2 : wilted_flowers = 48)
  (h3 : remaining_bouquets = 8) :
  flowers_per_bouquet * remaining_bouquets + wilted_flowers = 88 :=
by sorry

end NUMINAMATH_CALUDE_faye_flowers_proof_l2818_281808


namespace NUMINAMATH_CALUDE_fred_balloon_count_l2818_281869

theorem fred_balloon_count (total sam dan : ℕ) (h1 : total = 72) (h2 : sam = 46) (h3 : dan = 16) :
  total - (sam + dan) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fred_balloon_count_l2818_281869


namespace NUMINAMATH_CALUDE_reciprocal_comparison_l2818_281893

theorem reciprocal_comparison :
  let numbers : List ℚ := [1/3, 1/2, 1, 2, 3]
  ∀ x ∈ numbers, x < (1 / x) ↔ (x = 1/3 ∨ x = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_comparison_l2818_281893


namespace NUMINAMATH_CALUDE_interior_nodes_line_property_l2818_281829

/-- A point with integer coordinates -/
structure Node where
  x : ℤ
  y : ℤ

/-- A triangle with vertices at nodes -/
structure Triangle where
  a : Node
  b : Node
  c : Node

/-- Checks if a node is inside a triangle -/
def Node.isInside (n : Node) (t : Triangle) : Prop :=
  sorry

/-- Checks if a line through two nodes passes through a vertex of a triangle -/
def Line.passesThroughVertex (p q : Node) (t : Triangle) : Prop :=
  sorry

/-- Checks if a line through two nodes is parallel to a side of a triangle -/
def Line.isParallelToSide (p q : Node) (t : Triangle) : Prop :=
  sorry

/-- Main theorem -/
theorem interior_nodes_line_property (t : Triangle) (p q : Node) :
  p.isInside t ∧ q.isInside t →
  (∀ r : Node, r.isInside t → r = p ∨ r = q) →
  Line.passesThroughVertex p q t ∨ Line.isParallelToSide p q t :=
sorry

end NUMINAMATH_CALUDE_interior_nodes_line_property_l2818_281829


namespace NUMINAMATH_CALUDE_digital_display_overlap_l2818_281858

/-- Represents a digital number display in a rectangle -/
structure DigitalDisplay where
  width : Nat
  height : Nat
  numbers : List Nat

/-- Represents the overlap of two digital displays -/
def overlap (d1 d2 : DigitalDisplay) : Nat :=
  sorry

/-- The main theorem about overlapping digital displays -/
theorem digital_display_overlap :
  ∀ (d : DigitalDisplay),
    d.width = 8 ∧ 
    d.height = 5 ∧ 
    d.numbers = [1, 2, 1, 9] ∧
    (overlap d (DigitalDisplay.mk 8 5 [6, 1, 2, 1])) = 30 := by
  sorry

end NUMINAMATH_CALUDE_digital_display_overlap_l2818_281858


namespace NUMINAMATH_CALUDE_expanded_ohara_triple_64_49_l2818_281834

/-- Definition of an Expanded O'Hara triple -/
def is_expanded_ohara_triple (a b x : ℕ) : Prop :=
  2 * (Real.sqrt a + Real.sqrt b) = x

/-- Theorem: If (64, 49, x) is an Expanded O'Hara triple, then x = 30 -/
theorem expanded_ohara_triple_64_49 (x : ℕ) :
  is_expanded_ohara_triple 64 49 x → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_expanded_ohara_triple_64_49_l2818_281834


namespace NUMINAMATH_CALUDE_line_passes_through_circle_center_l2818_281857

/-- The center of a circle given by the equation x^2 + y^2 + 2x - 4y = 0 -/
def circle_center : ℝ × ℝ := (-1, 2)

/-- The line equation 3x + y + a = 0 -/
def line_equation (a : ℝ) (x y : ℝ) : Prop :=
  3 * x + y + a = 0

/-- The circle equation x^2 + y^2 + 2x - 4y = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 0

/-- Theorem: If the line 3x + y + a = 0 passes through the center of the circle
    x^2 + y^2 + 2x - 4y = 0, then a = 1 -/
theorem line_passes_through_circle_center (a : ℝ) :
  line_equation a (circle_center.1) (circle_center.2) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_circle_center_l2818_281857


namespace NUMINAMATH_CALUDE_urn_probability_l2818_281866

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Blue

/-- Represents the state of the urn -/
structure UrnState :=
  (red : ℕ)
  (blue : ℕ)

/-- Represents a single operation -/
def perform_operation (state : UrnState) (drawn : BallColor) : UrnState :=
  match drawn with
  | BallColor.Red => UrnState.mk (state.red + 3) state.blue
  | BallColor.Blue => UrnState.mk state.red (state.blue + 3)

/-- Represents the sequence of operations -/
def operation_sequence := List BallColor

/-- Calculates the probability of a specific operation sequence -/
def sequence_probability (seq : operation_sequence) : ℚ :=
  sorry

/-- Counts the number of valid operation sequences -/
def count_valid_sequences : ℕ :=
  sorry

theorem urn_probability :
  let initial_state : UrnState := UrnState.mk 2 1
  let final_state : UrnState := UrnState.mk 10 6
  let num_operations : ℕ := 5
  (count_valid_sequences * sequence_probability (List.replicate num_operations BallColor.Red)) = 16/115 :=
sorry

end NUMINAMATH_CALUDE_urn_probability_l2818_281866


namespace NUMINAMATH_CALUDE_running_increase_calculation_l2818_281852

theorem running_increase_calculation 
  (initial_miles : ℕ) 
  (increase_percentage : ℚ) 
  (total_days : ℕ) 
  (days_per_week : ℕ) : 
  initial_miles = 100 →
  increase_percentage = 1/5 →
  total_days = 280 →
  days_per_week = 7 →
  (initial_miles * (1 + increase_percentage) - initial_miles) / (total_days / days_per_week) = 3 :=
by sorry

end NUMINAMATH_CALUDE_running_increase_calculation_l2818_281852


namespace NUMINAMATH_CALUDE_value_of_x_l2818_281805

theorem value_of_x : ∃ x : ℝ, 3 * x + 15 = (1 / 3) * (7 * x + 45) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l2818_281805


namespace NUMINAMATH_CALUDE_no_prime_of_form_3811_l2818_281803

def a (n : ℕ) : ℕ := 38 * 10^n + (10^n - 1)

theorem no_prime_of_form_3811 : ∀ n : ℕ, ¬ Nat.Prime (a n) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_of_form_3811_l2818_281803


namespace NUMINAMATH_CALUDE_condo_units_calculation_l2818_281871

/-- Calculates the total number of units in a condo development -/
theorem condo_units_calculation (total_floors : ℕ) (regular_units_per_floor : ℕ) 
  (penthouse_units_per_floor : ℕ) (penthouse_floors : ℕ) : 
  total_floors = 23 → 
  regular_units_per_floor = 12 → 
  penthouse_units_per_floor = 2 → 
  penthouse_floors = 2 → 
  (total_floors - penthouse_floors) * regular_units_per_floor + 
    penthouse_floors * penthouse_units_per_floor = 256 := by
  sorry

#check condo_units_calculation

end NUMINAMATH_CALUDE_condo_units_calculation_l2818_281871


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2818_281848

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  roots_property : a 1 + a 2011 = 10 ∧ a 1 * a 2011 = 16

/-- The sum of specific terms in the arithmetic sequence is 15 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) : 
  seq.a 2 + seq.a 1006 + seq.a 2010 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2818_281848


namespace NUMINAMATH_CALUDE_concurrent_lines_theorem_l2818_281814

/-- A line that intersects opposite sides of a square -/
structure DividingLine where
  divides_square : Bool
  area_ratio : Rat
  intersects_opposite_sides : Bool

/-- A configuration of lines dividing a square -/
structure SquareDivision where
  lines : Finset DividingLine
  square : Set (ℝ × ℝ)

/-- The number of concurrent lines in a square division -/
def num_concurrent (sd : SquareDivision) : ℕ := sorry

theorem concurrent_lines_theorem (sd : SquareDivision) 
  (h1 : sd.lines.card = 2005)
  (h2 : ∀ l ∈ sd.lines, l.divides_square)
  (h3 : ∀ l ∈ sd.lines, l.area_ratio = 2 / 3)
  (h4 : ∀ l ∈ sd.lines, l.intersects_opposite_sides) :
  num_concurrent sd ≥ 502 := by sorry

end NUMINAMATH_CALUDE_concurrent_lines_theorem_l2818_281814


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l2818_281810

/-- Given a point M with coordinates (3, -5), prove that its symmetric point
    with respect to the origin has coordinates (-3, 5). -/
theorem symmetric_point_wrt_origin :
  let M : ℝ × ℝ := (3, -5)
  let symmetric_point : ℝ × ℝ → ℝ × ℝ := λ (x, y) => (-x, -y)
  symmetric_point M = (-3, 5) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l2818_281810


namespace NUMINAMATH_CALUDE_M_equals_N_l2818_281863

def M : Set ℤ := {-1, 0, 1}

def N : Set ℤ := {x | ∃ a b, a ∈ M ∧ b ∈ M ∧ x = a * b}

theorem M_equals_N : M = N := by
  sorry

end NUMINAMATH_CALUDE_M_equals_N_l2818_281863


namespace NUMINAMATH_CALUDE_colten_chicken_count_l2818_281879

/-- Represents the number of chickens each person has. -/
structure ChickenCount where
  colten : ℕ
  skylar : ℕ
  quentin : ℕ

/-- The conditions of the chicken problem. -/
def ChickenProblem (c : ChickenCount) : Prop :=
  c.colten + c.skylar + c.quentin = 383 ∧
  c.quentin = 25 + 2 * c.skylar ∧
  c.skylar = 3 * c.colten - 4

theorem colten_chicken_count :
  ∀ c : ChickenCount, ChickenProblem c → c.colten = 37 := by
  sorry

end NUMINAMATH_CALUDE_colten_chicken_count_l2818_281879


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l2818_281892

/-- The equation 9x^2 - 36y^2 = 36 represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), 9 * x^2 - 36 * y^2 = 36 ↔ x^2 / a^2 - y^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l2818_281892


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2818_281887

/-- A quadratic function with vertex form (x + h)^2 + k -/
def QuadraticFunction (a h k : ℝ) (x : ℝ) : ℝ := a * (x + h)^2 + k

theorem quadratic_coefficient (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = QuadraticFunction a 3 0 x) →  -- vertex at (-3, 0)
  f 2 = -50 →                              -- passes through (2, -50)
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2818_281887


namespace NUMINAMATH_CALUDE_stating_three_card_draw_probability_value_l2818_281877

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck := Fin 52

/-- The probability of drawing a specific sequence of three cards from a standard deck -/
def three_card_draw_probability : ℚ :=
  -- Probability of first card being a non-heart King
  (3 : ℚ) / 52 *
  -- Probability of second card being a heart (not King of hearts)
  12 / 51 *
  -- Probability of third card being a spade or diamond
  26 / 50

/-- 
Theorem stating that the probability of drawing a non-heart King, 
then a heart (not King of hearts), then a spade or diamond 
from a standard 52-card deck is 26/3675
-/
theorem three_card_draw_probability_value : 
  three_card_draw_probability = 26 / 3675 := by
  sorry

end NUMINAMATH_CALUDE_stating_three_card_draw_probability_value_l2818_281877


namespace NUMINAMATH_CALUDE_max_planes_from_four_lines_l2818_281812

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of lines -/
def num_lines : ℕ := 4

/-- The number of lines needed to define a plane -/
def lines_per_plane : ℕ := 2

/-- The maximum number of planes that can be defined by four lines starting from the same point -/
def max_planes : ℕ := choose num_lines lines_per_plane

theorem max_planes_from_four_lines : 
  max_planes = 6 := by sorry

end NUMINAMATH_CALUDE_max_planes_from_four_lines_l2818_281812


namespace NUMINAMATH_CALUDE_cow_milk_production_l2818_281862

/-- Given a number of cows and total weekly milk production, 
    calculate the daily milk production per cow. -/
def daily_milk_per_cow (num_cows : ℕ) (weekly_milk : ℕ) : ℚ :=
  (weekly_milk : ℚ) / 7 / num_cows

theorem cow_milk_production : daily_milk_per_cow 52 1820 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cow_milk_production_l2818_281862


namespace NUMINAMATH_CALUDE_equal_fractions_imply_x_equals_four_l2818_281874

theorem equal_fractions_imply_x_equals_four (x : ℝ) :
  (x ≠ 0) → (x ≠ -2) → (6 / (x + 2) = 4 / x) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equal_fractions_imply_x_equals_four_l2818_281874


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_13_l2818_281851

theorem smallest_n_divisible_by_13 : 
  ∃ (n : ℕ), (13 ∣ (5^n + n^5)) ∧ (∀ m : ℕ, m < n → ¬(13 ∣ (5^m + m^5))) ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_13_l2818_281851


namespace NUMINAMATH_CALUDE_line_properties_l2818_281891

/-- A line in the xy-plane represented by the equation x = ky + b -/
structure Line where
  k : ℝ
  b : ℝ

/-- Predicate for a line being perpendicular to the y-axis -/
def perpendicular_to_y_axis (l : Line) : Prop :=
  ∃ (x : ℝ), ∀ (y : ℝ), x = l.k * y + l.b

/-- Predicate for a line being perpendicular to the x-axis -/
def perpendicular_to_x_axis (l : Line) : Prop :=
  ∀ (y : ℝ), l.k * y + l.b = l.b

theorem line_properties :
  (¬ ∃ (l : Line), perpendicular_to_y_axis l) ∧
  (∃ (l : Line), perpendicular_to_x_axis l) :=
sorry

end NUMINAMATH_CALUDE_line_properties_l2818_281891


namespace NUMINAMATH_CALUDE_hyperbola_focus_to_asymptote_distance_l2818_281867

-- Define the hyperbola and its properties
theorem hyperbola_focus_to_asymptote_distance :
  ∀ (M : ℝ × ℝ),
  let F₁ : ℝ × ℝ := (-Real.sqrt 10, 0)
  let F₂ : ℝ × ℝ := (Real.sqrt 10, 0)
  let MF₁ : ℝ × ℝ := (M.1 - F₁.1, M.2 - F₁.2)
  let MF₂ : ℝ × ℝ := (M.2 - F₂.1, M.2 - F₂.2)
  -- M is on the hyperbola
  -- MF₁ · MF₂ = 0
  (MF₁.1 * MF₂.1 + MF₁.2 * MF₂.2 = 0) →
  -- |MF₁| · |MF₂| = 2
  (Real.sqrt (MF₁.1^2 + MF₁.2^2) * Real.sqrt (MF₂.1^2 + MF₂.2^2) = 2) →
  -- The distance from a focus to one of its asymptotes is 1
  (1 : ℝ) = 
    (Real.sqrt 10) / Real.sqrt (1 + (1/3)^2) :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_focus_to_asymptote_distance_l2818_281867


namespace NUMINAMATH_CALUDE_janet_piano_hours_l2818_281818

/-- Represents the number of hours per week Janet takes piano lessons -/
def piano_hours : ℕ := sorry

/-- The cost per hour of clarinet lessons -/
def clarinet_cost_per_hour : ℕ := 40

/-- The number of hours per week of clarinet lessons -/
def clarinet_hours_per_week : ℕ := 3

/-- The cost per hour of piano lessons -/
def piano_cost_per_hour : ℕ := 28

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- The additional amount spent on piano lessons compared to clarinet lessons in a year -/
def additional_piano_cost : ℕ := 1040

theorem janet_piano_hours :
  piano_hours = 5 ∧
  clarinet_cost_per_hour * clarinet_hours_per_week * weeks_per_year + additional_piano_cost =
  piano_cost_per_hour * piano_hours * weeks_per_year :=
by sorry

end NUMINAMATH_CALUDE_janet_piano_hours_l2818_281818


namespace NUMINAMATH_CALUDE_perpendicular_lines_l2818_281830

-- Define the types for lines and planes in 3D space
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Line → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_lines 
  (a b c d : Line) (α β : Plane)
  (h1 : perp a b)
  (h2 : perp_line_plane a α)
  (h3 : perp_line_plane c α) :
  perp c b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l2818_281830


namespace NUMINAMATH_CALUDE_roots_of_equation_l2818_281859

theorem roots_of_equation (x : ℝ) : 
  (x - 3)^2 = 4 ↔ x = 5 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2818_281859


namespace NUMINAMATH_CALUDE_product_difference_theorem_l2818_281825

theorem product_difference_theorem (N : ℕ) : ∃ (a b c d : ℕ),
  a + b = c + d ∧ c * d = N * (a * b) + a * b := by
  use 4*N - 2, 1, 2*N, 2*N - 1
  sorry

end NUMINAMATH_CALUDE_product_difference_theorem_l2818_281825


namespace NUMINAMATH_CALUDE_alison_small_tub_cost_l2818_281801

/-- The cost of small tubs given the number of large and small tubs, their total cost, and the cost of large tubs. -/
def small_tub_cost (num_large : ℕ) (num_small : ℕ) (total_cost : ℕ) (large_cost : ℕ) : ℕ :=
  (total_cost - num_large * large_cost) / num_small

/-- Theorem stating that the cost of each small tub is 5 dollars. -/
theorem alison_small_tub_cost :
  small_tub_cost 3 6 48 6 = 5 := by
sorry

#eval small_tub_cost 3 6 48 6

end NUMINAMATH_CALUDE_alison_small_tub_cost_l2818_281801


namespace NUMINAMATH_CALUDE_exam_score_unique_solution_l2818_281860

theorem exam_score_unique_solution (n : ℕ) : 
  (∃ t : ℚ, t > 0 ∧ 
    15 * t + (1/3 : ℚ) * ((n : ℚ) - 20) * t = (1/2 : ℚ) * (n : ℚ) * t) → 
  n = 50 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_unique_solution_l2818_281860


namespace NUMINAMATH_CALUDE_ball_probabilities_l2818_281881

-- Define the number of balls in each can
def can_A : Fin 3 → ℕ
| 0 => 5  -- red balls
| 1 => 2  -- white balls
| 2 => 3  -- black balls

def can_B : Fin 3 → ℕ
| 0 => 4  -- red balls
| 1 => 3  -- white balls
| 2 => 3  -- black balls

-- Define the probability of drawing a ball of each color from can A
def prob_A (i : Fin 3) : ℚ :=
  (can_A i : ℚ) / (can_A 0 + can_A 1 + can_A 2 : ℚ)

-- Define the probability of drawing a red ball from can B after moving a ball from A
def prob_B_red (i : Fin 3) : ℚ :=
  (can_B 0 + (if i = 0 then 1 else 0) : ℚ) / 
  ((can_B 0 + can_B 1 + can_B 2 + 1) : ℚ)

theorem ball_probabilities :
  (prob_B_red 0 = 5/11) ∧ 
  (prob_A 2 * prob_B_red 2 = 6/55) ∧
  (prob_A 0 * prob_B_red 0 / (prob_A 0 * prob_B_red 0 + prob_A 1 * prob_B_red 1 + prob_A 2 * prob_B_red 2) = 5/9) := by
  sorry


end NUMINAMATH_CALUDE_ball_probabilities_l2818_281881


namespace NUMINAMATH_CALUDE_abcd_16_bits_l2818_281846

def base_16_to_decimal (a b c d : ℕ) : ℕ :=
  a * 16^3 + b * 16^2 + c * 16 + d

def bits_required (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

theorem abcd_16_bits :
  bits_required (base_16_to_decimal 10 11 12 13) = 16 := by
  sorry

end NUMINAMATH_CALUDE_abcd_16_bits_l2818_281846


namespace NUMINAMATH_CALUDE_josh_marbles_l2818_281889

theorem josh_marbles (initial : ℕ) (lost : ℕ) (remaining : ℕ) : 
  initial = 16 → lost = 7 → remaining = initial - lost → remaining = 9 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_l2818_281889


namespace NUMINAMATH_CALUDE_remainder_of_7n_mod_4_l2818_281832

theorem remainder_of_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_7n_mod_4_l2818_281832


namespace NUMINAMATH_CALUDE_ohara_quadruple_example_l2818_281831

theorem ohara_quadruple_example :
  ∀ (x : ℤ), (Real.sqrt 9 + Real.sqrt 16 + 3^2 : ℝ) = x → x = 16 := by
sorry

end NUMINAMATH_CALUDE_ohara_quadruple_example_l2818_281831


namespace NUMINAMATH_CALUDE_symmetric_line_y_axis_l2818_281870

/-- Given a line ax + by + c = 0, returns the line symmetric to it with respect to the y-axis -/
def symmetricLineY (a b c : ℝ) : ℝ × ℝ × ℝ := (-a, b, c)

/-- The equation of a line passing through two points (x₁, y₁) and (x₂, y₂) -/
def lineThroughPoints (x₁ y₁ x₂ y₂ : ℝ) : ℝ × ℝ × ℝ :=
  let a := y₂ - y₁
  let b := x₁ - x₂
  let c := x₂ * y₁ - x₁ * y₂
  (a, b, c)

theorem symmetric_line_y_axis :
  let original_line := (3, -4, 5)
  let symmetric_line := symmetricLineY 3 (-4) 5
  let y_intercept := (0, 5/4)
  let x_intercept_symmetric := (5/3, 0)
  let line_through_points := lineThroughPoints 0 (5/4) (5/3) 0
  symmetric_line = line_through_points := by sorry

end NUMINAMATH_CALUDE_symmetric_line_y_axis_l2818_281870


namespace NUMINAMATH_CALUDE_c_increases_as_n_increases_l2818_281888

/-- Given a formula for C, prove that C increases as n increases. -/
theorem c_increases_as_n_increases
  (e R r : ℝ)
  (he : e > 0)
  (hR : R > 0)
  (hr : r > 0)
  (C : ℝ → ℝ)
  (hC : ∀ n, n > 0 → C n = (e^2 * n) / (R + n*r)) :
  ∀ n₁ n₂, 0 < n₁ → n₁ < n₂ → C n₁ < C n₂ :=
by sorry

end NUMINAMATH_CALUDE_c_increases_as_n_increases_l2818_281888


namespace NUMINAMATH_CALUDE_r_squared_ssr_inverse_relation_l2818_281873

/-- Represents a regression model -/
structure RegressionModel where
  R_squared : ℝ  -- Coefficient of determination
  SSR : ℝ        -- Sum of squares of residuals

/-- States that as R² increases, SSR decreases in a regression model -/
theorem r_squared_ssr_inverse_relation (model1 model2 : RegressionModel) :
  model1.R_squared > model2.R_squared → model1.SSR < model2.SSR := by
  sorry

end NUMINAMATH_CALUDE_r_squared_ssr_inverse_relation_l2818_281873


namespace NUMINAMATH_CALUDE_smallest_positive_b_existence_l2818_281861

theorem smallest_positive_b_existence :
  ∃ (b y : ℝ), b > 0 ∧ y > 0 ∧
  ((9 * Real.sqrt ((3*b)^2 + 2^2) + 5*b^2 - 2) / (Real.sqrt (2 + 5*b^2) - 5) = -4) ∧
  (y^4 + 105*y^2 + 562 = 0) ∧
  (y^2 > 2) ∧
  (b = Real.sqrt (y^2 - 2) / Real.sqrt 5) ∧
  (∀ (b' : ℝ), b' > 0 → 
    ((9 * Real.sqrt ((3*b')^2 + 2^2) + 5*b'^2 - 2) / (Real.sqrt (2 + 5*b'^2) - 5) = -4) →
    b ≤ b') :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_b_existence_l2818_281861


namespace NUMINAMATH_CALUDE_complex_equation_real_solution_l2818_281836

theorem complex_equation_real_solution (a : ℝ) : 
  (((a : ℂ) / (1 + Complex.I) + (1 + Complex.I) / 2).im = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_real_solution_l2818_281836


namespace NUMINAMATH_CALUDE_c_oxen_count_l2818_281811

/-- Represents the number of oxen-months for a person's grazing arrangement -/
def oxen_months (oxen : ℕ) (months : ℕ) : ℕ := oxen * months

/-- Calculates the share of rent based on oxen-months and total rent -/
def rent_share (own_oxen_months : ℕ) (total_oxen_months : ℕ) (total_rent : ℕ) : ℕ :=
  (own_oxen_months * total_rent) / total_oxen_months

theorem c_oxen_count (x : ℕ) : 
  oxen_months 10 7 + oxen_months 12 5 + oxen_months x 3 = 130 + 3 * x →
  rent_share (oxen_months x 3) (130 + 3 * x) 280 = 72 →
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_c_oxen_count_l2818_281811


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l2818_281845

/-- Theorem: Weight of replaced person in a group
Given a group of 9 persons, if replacing one person with a new person weighing 87.5 kg
increases the average weight by 2.5 kg, then the weight of the replaced person was 65 kg. -/
theorem weight_of_replaced_person
  (n : ℕ) -- number of persons in the group
  (w : ℝ) -- total weight of the original group
  (new_weight : ℝ) -- weight of the new person
  (avg_increase : ℝ) -- increase in average weight
  (h1 : n = 9)
  (h2 : new_weight = 87.5)
  (h3 : avg_increase = 2.5)
  (h4 : (w - (w / n) + new_weight) / n = (w / n) + avg_increase) :
  w / n = 65 :=
sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l2818_281845


namespace NUMINAMATH_CALUDE_parallel_line_plane_not_implies_parallel_all_lines_l2818_281896

-- Define the basic geometric objects
variable (Point Line Plane : Type)

-- Define the geometric relations
variable (contains : Plane → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem parallel_line_plane_not_implies_parallel_all_lines 
  (α : Plane) (a b : Line) : 
  ¬(∀ (p : Plane) (l m : Line), 
    parallel_line_plane l p → 
    contains p m → 
    parallel_lines l m) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_plane_not_implies_parallel_all_lines_l2818_281896


namespace NUMINAMATH_CALUDE_article_price_l2818_281815

theorem article_price (decreased_price : ℚ) (decrease_percentage : ℚ) (original_price : ℚ) : 
  decreased_price = 1050 ∧ 
  decrease_percentage = 40 ∧ 
  decreased_price = original_price * (1 - decrease_percentage / 100) → 
  original_price = 1750 := by
sorry

end NUMINAMATH_CALUDE_article_price_l2818_281815


namespace NUMINAMATH_CALUDE_badminton_players_l2818_281802

theorem badminton_players (total : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 30)
  (h2 : tennis = 19)
  (h3 : both = 7)
  (h4 : neither = 2) :
  total - tennis - neither + both = 16 :=
by sorry

end NUMINAMATH_CALUDE_badminton_players_l2818_281802


namespace NUMINAMATH_CALUDE_inscribed_circle_diameter_l2818_281817

/-- Given a right triangle with legs of length 8 and 15, the diameter of its inscribed circle is 6 -/
theorem inscribed_circle_diameter (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_a : a = 8) (h_b : b = 15) : 
  2 * (a * b) / (a + b + c) = 6 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_diameter_l2818_281817


namespace NUMINAMATH_CALUDE_room_length_calculation_l2818_281800

/-- Given a rectangular room with width 12 m, surrounded by a 2 m wide veranda on all sides,
    if the area of the veranda is 132 m², then the length of the room is 17 m. -/
theorem room_length_calculation (room_width : ℝ) (veranda_width : ℝ) (veranda_area : ℝ) :
  room_width = 12 →
  veranda_width = 2 →
  veranda_area = 132 →
  ∃ (room_length : ℝ),
    (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) -
    room_length * room_width = veranda_area ∧
    room_length = 17 :=
by sorry

end NUMINAMATH_CALUDE_room_length_calculation_l2818_281800


namespace NUMINAMATH_CALUDE_square_root_of_one_incorrect_l2818_281827

theorem square_root_of_one_incorrect : ¬(∀ x : ℝ, x^2 = 1 → x = 1) := by
  sorry

#check square_root_of_one_incorrect

end NUMINAMATH_CALUDE_square_root_of_one_incorrect_l2818_281827


namespace NUMINAMATH_CALUDE_f_properties_l2818_281821

-- Define the function f
def f (x : ℝ) : ℝ := |x + 7| + |x - 1|

-- Theorem statement
theorem f_properties :
  (∀ x : ℝ, f x ≥ 8) ∧
  (∀ x : ℝ, |x - 3| - 2*x ≤ 4 ↔ x ≥ -1/3) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2818_281821


namespace NUMINAMATH_CALUDE_intersection_right_angle_coordinates_l2818_281806

-- Define the line and parabola
def line (x y : ℝ) : Prop := x - 2*y - 1 = 0
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define points A and B as intersections
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line A.1 A.2 ∧ parabola A.1 A.2 ∧
  line B.1 B.2 ∧ parabola B.1 B.2 ∧
  A ≠ B

-- Define point C on the parabola
def point_on_parabola (C : ℝ × ℝ) : Prop := parabola C.1 C.2

-- Define right angle ACB
def right_angle (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0

-- Theorem statement
theorem intersection_right_angle_coordinates :
  ∀ A B C : ℝ × ℝ,
  intersection_points A B →
  point_on_parabola C →
  right_angle A B C →
  (C = (1, -2) ∨ C = (9, -6)) :=
sorry

end NUMINAMATH_CALUDE_intersection_right_angle_coordinates_l2818_281806


namespace NUMINAMATH_CALUDE_g_properties_and_range_l2818_281880

def f (x : ℝ) : ℝ := x^2 - 3*x + 2

def g (x : ℝ) : ℝ := |x|^2 - 3*|x| + 2

theorem g_properties_and_range :
  (∀ x : ℝ, g (-x) = g x) ∧
  (∀ x : ℝ, x ≥ 0 → g x = f x) ∧
  ({m : ℝ | g m > 2} = {m : ℝ | m < -3 ∨ m > 3}) := by
  sorry

end NUMINAMATH_CALUDE_g_properties_and_range_l2818_281880


namespace NUMINAMATH_CALUDE_brothers_combined_age_l2818_281824

/-- Given the ages of Michael and his three brothers, prove their combined age is 53 years. -/
theorem brothers_combined_age :
  ∀ (michael oldest older younger : ℕ),
  -- The oldest brother is 1 year older than twice Michael's age when Michael was a year younger
  oldest = 2 * (michael - 1) + 1 →
  -- The younger brother is 5 years old
  younger = 5 →
  -- The younger brother's age is a third of the older brother's age
  older = 3 * younger →
  -- The other brother is half the age of the oldest brother
  older = oldest / 2 →
  -- The other brother is three years younger than Michael
  older = michael - 3 →
  -- The other brother is twice as old as their youngest brother
  older = 2 * younger →
  -- The combined age of all four brothers is 53
  michael + oldest + older + younger = 53 := by
sorry


end NUMINAMATH_CALUDE_brothers_combined_age_l2818_281824


namespace NUMINAMATH_CALUDE_tan_triple_angle_l2818_281882

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 := by
  sorry

end NUMINAMATH_CALUDE_tan_triple_angle_l2818_281882


namespace NUMINAMATH_CALUDE_odd_prime_square_root_l2818_281853

theorem odd_prime_square_root (p : ℕ) (hp : Prime p) (hp_odd : Odd p) :
  ∀ k : ℕ, (∃ m : ℕ, m > 0 ∧ m * m = k * k - p * k) ↔ k = ((p + 1) / 2) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_square_root_l2818_281853


namespace NUMINAMATH_CALUDE_no_equal_group_division_l2818_281868

theorem no_equal_group_division (k : ℕ) : 
  ¬ ∃ (g1 g2 : List ℕ), 
    (∀ n, n ∈ g1 ∪ g2 ↔ 1 ≤ n ∧ n ≤ k) ∧ 
    (∀ n, n ∈ g1 → n ∉ g2) ∧
    (∀ n, n ∈ g2 → n ∉ g1) ∧
    (g1.foldl (λ acc x => acc * 10 + x) 0 = g2.foldl (λ acc x => acc * 10 + x) 0) :=
by sorry

end NUMINAMATH_CALUDE_no_equal_group_division_l2818_281868


namespace NUMINAMATH_CALUDE_cosine_sum_difference_l2818_281822

theorem cosine_sum_difference : 
  Real.cos (π / 15) - Real.cos (2 * π / 15) - Real.cos (4 * π / 15) + Real.cos (7 * π / 15) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_difference_l2818_281822


namespace NUMINAMATH_CALUDE_percentage_increase_l2818_281843

theorem percentage_increase (x : ℝ) (h1 : x = 62.4) (h2 : x > 52) :
  (x - 52) / 52 * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l2818_281843


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l2818_281890

theorem quadratic_solution_property (a : ℝ) : 
  a^2 - 2*a - 1 = 0 → 2*a^2 - 4*a + 2022 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l2818_281890


namespace NUMINAMATH_CALUDE_train_length_proof_l2818_281885

/-- Proves that a train with given speed and crossing time has a specific length -/
theorem train_length_proof (speed : ℝ) (crossing_time : ℝ) (train_length : ℝ) : 
  speed = 90 → -- speed in km/hr
  crossing_time = 1 / 60 → -- crossing time in hours (1 minute = 1/60 hour)
  train_length = speed * crossing_time / 2 → -- length calculation
  train_length = 750 / 1000 -- length in km (750 m = 0.75 km)
  := by sorry

end NUMINAMATH_CALUDE_train_length_proof_l2818_281885


namespace NUMINAMATH_CALUDE_modulo_residue_sum_l2818_281833

theorem modulo_residue_sum : (255 + 7 * 51 + 9 * 187 + 5 * 34) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_modulo_residue_sum_l2818_281833


namespace NUMINAMATH_CALUDE_base_b_square_theorem_l2818_281883

/-- Converts a number from base b representation to base 10 -/
def base_b_to_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun digit acc => b * acc + digit) 0

/-- Theorem: If 1325 in base b is the square of 35 in base b, then b = 7 in base 10 -/
theorem base_b_square_theorem :
  ∀ b : Nat,
  (base_b_to_10 [1, 3, 2, 5] b = (base_b_to_10 [3, 5] b) ^ 2) →
  b = 7 :=
by sorry

end NUMINAMATH_CALUDE_base_b_square_theorem_l2818_281883


namespace NUMINAMATH_CALUDE_world_cup_stats_l2818_281849

def world_cup_data : List ℕ := [32, 31, 16, 16, 14, 12]

def median (l : List ℕ) : ℚ := sorry

def mode (l : List ℕ) : ℕ := sorry

theorem world_cup_stats :
  median world_cup_data = 16 ∧ mode world_cup_data = 16 := by sorry

end NUMINAMATH_CALUDE_world_cup_stats_l2818_281849


namespace NUMINAMATH_CALUDE_brady_earnings_brady_earnings_200_l2818_281828

/-- Brady's earnings for transcribing recipe cards -/
theorem brady_earnings : ℕ → ℚ
  | cards => 
    let base_pay := (70 : ℚ) / 100 * cards
    let bonus := 10 * (cards / 100 : ℕ)
    base_pay + bonus

/-- Proof of Brady's earnings for 200 cards -/
theorem brady_earnings_200 : brady_earnings 200 = 160 := by
  sorry

end NUMINAMATH_CALUDE_brady_earnings_brady_earnings_200_l2818_281828


namespace NUMINAMATH_CALUDE_red_highest_probability_l2818_281899

/-- Represents the colors of the balls in the box -/
inductive Color
  | Red
  | Yellow
  | Black

/-- Represents the box of balls -/
structure Box where
  total : Nat
  red : Nat
  yellow : Nat
  black : Nat

/-- Calculates the probability of drawing a ball of a given color -/
def probability (box : Box) (color : Color) : Rat :=
  match color with
  | Color.Red => box.red / box.total
  | Color.Yellow => box.yellow / box.total
  | Color.Black => box.black / box.total

/-- The box with the given conditions -/
def givenBox : Box :=
  { total := 10
    red := 7
    yellow := 2
    black := 1 }

theorem red_highest_probability :
  probability givenBox Color.Red > probability givenBox Color.Yellow ∧
  probability givenBox Color.Red > probability givenBox Color.Black :=
by sorry

end NUMINAMATH_CALUDE_red_highest_probability_l2818_281899


namespace NUMINAMATH_CALUDE_circles_and_tangent_line_l2818_281809

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 4
def circle_O2_center : ℝ × ℝ := (3, 3)

-- Define the external tangency condition
def externally_tangent (O1 O2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (O1.1 - O2.1)^2 + (O1.2 - O2.2)^2 = (r1 + r2)^2

-- Theorem statement
theorem circles_and_tangent_line :
  ∃ (r2 : ℝ),
    -- Circle O₂ equation
    (∀ x y : ℝ, (x - 3)^2 + (y - 3)^2 = r2^2) ∧
    -- External tangency condition
    externally_tangent (0, -1) circle_O2_center 2 r2 ∧
    -- Common internal tangent line equation
    (∀ x y : ℝ, circle_O1 x y ∧ (x - 3)^2 + (y - 3)^2 = r2^2 →
      3*x + 4*y = 6) :=
by
  sorry

end NUMINAMATH_CALUDE_circles_and_tangent_line_l2818_281809


namespace NUMINAMATH_CALUDE_volleyball_match_probability_l2818_281854

-- Define the probability of team A winning a set in the first four sets
def p_win_first_four : ℚ := 2 / 3

-- Define the probability of team A winning the fifth set
def p_win_fifth : ℚ := 1 / 2

-- Define the number of ways to choose 2 wins out of 4 sets
def ways_to_win_two_of_four : ℕ := 6

-- State the theorem
theorem volleyball_match_probability :
  let p_three_two := ways_to_win_two_of_four * p_win_first_four^2 * (1 - p_win_first_four)^2 * p_win_fifth
  p_three_two = 4 / 27 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_match_probability_l2818_281854


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2818_281850

theorem complex_equation_solution (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) : 
  z = Complex.I + 1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2818_281850


namespace NUMINAMATH_CALUDE_math_test_problem_l2818_281872

theorem math_test_problem (total : ℕ) (word_problems : ℕ) (answered : ℕ) (blank : ℕ) :
  total = 45 →
  word_problems = 17 →
  answered = 38 →
  blank = 7 →
  total = answered + blank →
  total - word_problems - blank = 21 := by
  sorry

end NUMINAMATH_CALUDE_math_test_problem_l2818_281872


namespace NUMINAMATH_CALUDE_propositions_proof_l2818_281835

theorem propositions_proof :
  (∃ a b : ℝ, a > b ∧ b > 0 ∧ a + 1/a ≤ b + 1/b) ∧
  (∀ m n : ℝ, m > n ∧ n > 0 → (m + 1) / (n + 1) < m / n) ∧
  (∀ c a b : ℝ, c > a ∧ a > b ∧ b > 0 → a / (c - a) > b / (c - b)) ∧
  (∀ a b : ℝ, a ≥ b ∧ b > -1 → a / (a + 1) ≥ b / (b + 1)) :=
by sorry

end NUMINAMATH_CALUDE_propositions_proof_l2818_281835


namespace NUMINAMATH_CALUDE_election_votes_l2818_281838

theorem election_votes (candidate_percentage : ℝ) (vote_difference : ℕ) (total_votes : ℕ) : 
  candidate_percentage = 35 / 100 → 
  vote_difference = 2100 →
  total_votes = (vote_difference : ℝ) / (1 - 2 * candidate_percentage) →
  total_votes = 7000 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l2818_281838


namespace NUMINAMATH_CALUDE_max_distance_line_circle_l2818_281898

/-- Given a line ax + 2by = 1 intersecting a circle x^2 + y^2 = 1 at points A and B,
    where triangle AOB is right-angled (O is the origin), prove that the maximum
    distance between P(a,b) and Q(0,0) is √2. -/
theorem max_distance_line_circle (a b : ℝ) : 
  (∃ A B : ℝ × ℝ, (a * A.1 + 2 * b * A.2 = 1 ∧ A.1^2 + A.2^2 = 1) ∧
                   (a * B.1 + 2 * b * B.2 = 1 ∧ B.1^2 + B.2^2 = 1) ∧
                   ((A.1 - B.1) * (A.1 + B.1) + (A.2 - B.2) * (A.2 + B.2) = 0)) →
  (∃ P : ℝ × ℝ, P.1 = a ∧ P.2 = b) →
  (∃ d : ℝ, d = Real.sqrt (a^2 + b^2) ∧ d ≤ Real.sqrt 2 ∧
            (∀ a' b' : ℝ, Real.sqrt (a'^2 + b'^2) ≤ d)) :=
by sorry


end NUMINAMATH_CALUDE_max_distance_line_circle_l2818_281898


namespace NUMINAMATH_CALUDE_product_of_numbers_l2818_281894

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x^2 + y^2 = 404) : x * y = 86 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2818_281894


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l2818_281823

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : ℕ := sorry

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (d : ℕ) : List (Fin 4) := sorry

theorem binary_to_quaternary_conversion :
  let binary : List Bool := [true, true, false, true, true, false, true, true, false, true]
  let quaternary : List (Fin 4) := [3, 1, 1, 3, 1]
  binary_to_decimal binary = (quaternary.map (λ x => x.val)).foldl (λ acc x => acc * 4 + x) 0 :=
by sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l2818_281823


namespace NUMINAMATH_CALUDE_dollar_equality_l2818_281886

/-- Custom operation definition -/
def dollar (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem statement -/
theorem dollar_equality (x y z : ℝ) : dollar ((x - y + z)^2) ((y - x - z)^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_dollar_equality_l2818_281886


namespace NUMINAMATH_CALUDE_paper_towel_case_rolls_l2818_281856

theorem paper_towel_case_rolls : ∀ (case_price individual_price : ℚ) (savings_percentage : ℚ),
  case_price = 9 →
  individual_price = 1 →
  savings_percentage = 25 →
  ∃ (n : ℕ), n = 12 ∧ case_price = (1 - savings_percentage / 100) * (n * individual_price) :=
by
  sorry

end NUMINAMATH_CALUDE_paper_towel_case_rolls_l2818_281856


namespace NUMINAMATH_CALUDE_ages_sum_five_years_ago_l2818_281820

/-- Proves that the sum of Angela's and Beth's ages 5 years ago was 39 years -/
theorem ages_sum_five_years_ago : 
  ∀ (angela_age beth_age : ℕ),
  angela_age = 4 * beth_age →
  angela_age + 5 = 44 →
  (angela_age - 5) + (beth_age - 5) = 39 :=
by
  sorry

end NUMINAMATH_CALUDE_ages_sum_five_years_ago_l2818_281820


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2818_281816

theorem no_integer_solutions : 
  ¬∃ (x y z : ℤ), 
    (x^2 - 4*x*y + 3*y^2 - z^2 = 41) ∧ 
    (-x^2 + 4*y*z + 2*z^2 = 52) ∧ 
    (x^2 + x*y + 5*z^2 = 110) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2818_281816
