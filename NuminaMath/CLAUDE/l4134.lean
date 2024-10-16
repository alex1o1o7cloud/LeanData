import Mathlib

namespace NUMINAMATH_CALUDE_min_distance_theorem_l4134_413444

/-- Represents a rectangular cave with four points A, B, C, and D -/
structure RectangularCave where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  AD : ℝ

/-- The minimum distance to cover all paths from A to C in a rectangular cave -/
def min_distance_all_paths (cave : RectangularCave) : ℝ :=
  cave.AB + cave.BC + cave.CD + cave.AD

/-- Theorem stating the minimum distance to cover all paths from A to C -/
theorem min_distance_theorem (cave : RectangularCave) 
  (h1 : cave.AB + cave.BC + cave.CD = 22)
  (h2 : cave.AD + cave.CD + cave.BC = 29)
  (h3 : cave.AB + cave.BC + (cave.AB + cave.AD) = 30) :
  min_distance_all_paths cave = 47 := by
  sorry

#eval min_distance_all_paths ⟨10, 5, 7, 12⟩

end NUMINAMATH_CALUDE_min_distance_theorem_l4134_413444


namespace NUMINAMATH_CALUDE_initial_charge_value_l4134_413479

/-- The charge for the first 1/5 of a minute in cents -/
def initial_charge : ℝ := sorry

/-- The charge for each additional 1/5 of a minute in cents -/
def additional_charge : ℝ := 0.40

/-- The total charge for an 8-minute call in cents -/
def total_charge : ℝ := 18.70

/-- The number of 1/5 minute intervals in 8 minutes -/
def total_intervals : ℕ := 8 * 5

/-- The number of additional 1/5 minute intervals after the first one -/
def additional_intervals : ℕ := total_intervals - 1

theorem initial_charge_value :
  initial_charge = 3.10 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_charge_value_l4134_413479


namespace NUMINAMATH_CALUDE_oliver_tickets_used_l4134_413419

theorem oliver_tickets_used (ferris_rides bumper_rides ticket_cost : ℕ) 
  (h1 : ferris_rides = 5)
  (h2 : bumper_rides = 4)
  (h3 : ticket_cost = 7) :
  (ferris_rides + bumper_rides) * ticket_cost = 63 := by
  sorry

end NUMINAMATH_CALUDE_oliver_tickets_used_l4134_413419


namespace NUMINAMATH_CALUDE_three_number_sum_l4134_413421

theorem three_number_sum : ∀ a b c : ℝ,
  a ≤ b → b ≤ c →
  b = 7 →
  (a + b + c) / 3 = a + 8 →
  (a + b + c) / 3 = c - 20 →
  a + b + c = 57 := by
sorry

end NUMINAMATH_CALUDE_three_number_sum_l4134_413421


namespace NUMINAMATH_CALUDE_binomial_10_choose_3_l4134_413402

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_3_l4134_413402


namespace NUMINAMATH_CALUDE_tacos_wanted_l4134_413464

/-- Proves the number of tacos given the cheese requirements and constraints -/
theorem tacos_wanted (cheese_per_burrito : ℕ) (cheese_per_taco : ℕ) 
  (burritos_wanted : ℕ) (total_cheese : ℕ) : ℕ :=
by
  sorry

#check tacos_wanted 4 9 7 37 = 1

end NUMINAMATH_CALUDE_tacos_wanted_l4134_413464


namespace NUMINAMATH_CALUDE_sunday_no_arguments_l4134_413487

/-- Probability of a spouse arguing with their mother-in-law -/
def p_argue_with_mil : ℚ := 2/3

/-- Probability of siding with own mother in case of conflict -/
def p_side_with_mother : ℚ := 1/2

/-- Probability of no arguments between spouses on a Sunday -/
def p_no_arguments : ℚ := 4/9

theorem sunday_no_arguments : 
  p_no_arguments = 1 - (2 * p_argue_with_mil * p_side_with_mother - (p_argue_with_mil * p_side_with_mother)^2) := by
  sorry

end NUMINAMATH_CALUDE_sunday_no_arguments_l4134_413487


namespace NUMINAMATH_CALUDE_min_tiles_cover_rect_l4134_413434

/-- The side length of a square tile in inches -/
def tile_side : ℕ := 6

/-- The length of the rectangular region in feet -/
def rect_length : ℕ := 6

/-- The width of the rectangular region in feet -/
def rect_width : ℕ := 3

/-- The number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- The minimum number of tiles needed to cover the rectangular region -/
def min_tiles : ℕ := 72

theorem min_tiles_cover_rect : 
  (rect_length * inches_per_foot) * (rect_width * inches_per_foot) = 
  min_tiles * (tile_side * tile_side) :=
by sorry

end NUMINAMATH_CALUDE_min_tiles_cover_rect_l4134_413434


namespace NUMINAMATH_CALUDE_power_of_two_equality_l4134_413456

theorem power_of_two_equality (n : ℕ) : 2^n = 2 * 16^2 * 64^3 → n = 27 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l4134_413456


namespace NUMINAMATH_CALUDE_nicoles_clothes_theorem_l4134_413470

/-- Calculates the total number of clothing pieces Nicole ends up with --/
def nicoles_total_clothes (nicole_start : ℕ) : ℕ :=
  let first_sister := nicole_start / 3
  let second_sister := nicole_start + 5
  let third_sister := 2 * first_sister
  let youngest_four_total := nicole_start + first_sister + second_sister + third_sister
  let oldest_sister := (youngest_four_total / 4 * 3 + (youngest_four_total % 4) / 2 + 1) / 2
  nicole_start + first_sister + second_sister + third_sister + oldest_sister

theorem nicoles_clothes_theorem :
  nicoles_total_clothes 15 = 69 := by
  sorry

end NUMINAMATH_CALUDE_nicoles_clothes_theorem_l4134_413470


namespace NUMINAMATH_CALUDE_intersection_A_B_range_of_p_l4134_413425

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B : Set ℝ := {x | 3 - |x| ≥ 0}

-- Define the set C
def C (p : ℝ) : Set ℝ := {x | x^2 + 4*x + 4 - p^2 < 0}

-- Statement 1: A ∩ B = {x | -3 ≤ x < -1 or 2 < x ≤ 3}
theorem intersection_A_B : A ∩ B = {x | -3 ≤ x ∧ x < -1 ∨ 2 < x ∧ x ≤ 3} := by sorry

-- Statement 2: The range of p satisfying the given conditions is 0 < p ≤ 1
theorem range_of_p (p : ℝ) (h_p : p > 0) : 
  (C p ⊆ (A ∩ B)) ↔ (p > 0 ∧ p ≤ 1) := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_range_of_p_l4134_413425


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l4134_413499

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := (m + 3) * x^2 - 4 * m * x + 2 * m - 1

-- Define the condition for roots having opposite signs
def opposite_signs (x₁ x₂ : ℝ) : Prop := x₁ * x₂ < 0

-- Define the condition for the absolute value of the negative root being greater than the positive root
def negative_root_greater (x₁ x₂ : ℝ) : Prop := 
  (x₁ < 0 ∧ x₂ > 0 ∧ abs x₁ > x₂) ∨ (x₂ < 0 ∧ x₁ > 0 ∧ abs x₂ > x₁)

-- The main theorem
theorem quadratic_roots_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    quadratic m x₁ = 0 ∧ 
    quadratic m x₂ = 0 ∧ 
    opposite_signs x₁ x₂ ∧ 
    negative_root_greater x₁ x₂) →
  -3 < m ∧ m < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l4134_413499


namespace NUMINAMATH_CALUDE_consecutive_product_not_perfect_power_l4134_413408

theorem consecutive_product_not_perfect_power (n : ℕ) :
  ∀ m : ℕ, m ≥ 2 → ¬∃ k : ℕ, (n - 1) * n * (n + 1) = k^m :=
by sorry

end NUMINAMATH_CALUDE_consecutive_product_not_perfect_power_l4134_413408


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l4134_413442

-- Problem 1
theorem problem_one : 4 * Real.sqrt 5 + Real.sqrt 45 - Real.sqrt 8 + 4 * Real.sqrt 2 = 7 * Real.sqrt 5 + 2 * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_two : (4 * Real.sqrt 3 - 6 * Real.sqrt (1/3) + 3 * Real.sqrt 12) / (2 * Real.sqrt 3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l4134_413442


namespace NUMINAMATH_CALUDE_car_distance_theorem_l4134_413420

theorem car_distance_theorem (initial_time : ℝ) (new_speed : ℝ) :
  initial_time = 6 →
  new_speed = 56 →
  ∃ (distance : ℝ),
    distance = new_speed * (3/2 * initial_time) ∧
    distance = 504 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l4134_413420


namespace NUMINAMATH_CALUDE_boat_return_time_boat_return_time_example_l4134_413407

/-- The time taken for a boat to return upstream along a riverbank, given its downstream travel details and river flow speeds. -/
theorem boat_return_time (downstream_speed : ℝ) (downstream_time : ℝ) (downstream_distance : ℝ)
  (main_flow_speed : ℝ) (bank_flow_speed : ℝ) : ℝ :=
  let boat_speed := downstream_speed - main_flow_speed
  let upstream_speed := boat_speed - bank_flow_speed
  downstream_distance / upstream_speed

/-- The boat's return time is 20 hours given the specified conditions. -/
theorem boat_return_time_example : 
  boat_return_time 36 10 360 10 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_boat_return_time_boat_return_time_example_l4134_413407


namespace NUMINAMATH_CALUDE_smallest_distance_to_2i_l4134_413426

theorem smallest_distance_to_2i (z : ℂ) (h : Complex.abs (z^2 + 3 + Complex.I) = Complex.abs (z * (z + 1 + 3 * Complex.I))) :
  Complex.abs (z - 2 * Complex.I) ≥ (1 : ℝ) / 2 ∧
  ∃ w : ℂ, Complex.abs (w^2 + 3 + Complex.I) = Complex.abs (w * (w + 1 + 3 * Complex.I)) ∧
           Complex.abs (w - 2 * Complex.I) = (1 : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_to_2i_l4134_413426


namespace NUMINAMATH_CALUDE_equation_solution_l4134_413431

theorem equation_solution (x : ℝ) : 
  1 / (x + 5) + 1 / (x - 5) = 1 / (x - 5) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4134_413431


namespace NUMINAMATH_CALUDE_subset_implies_a_values_l4134_413448

/-- The set M of solutions to the quadratic equation 2x^2 - 3x - 2 = 0 -/
def M : Set ℝ := {x | 2 * x^2 - 3 * x - 2 = 0}

/-- The set N of solutions to the linear equation ax = 1 -/
def N (a : ℝ) : Set ℝ := {x | a * x = 1}

/-- Theorem stating that if N is a subset of M, then a must be 0, -2, or 1/2 -/
theorem subset_implies_a_values (a : ℝ) (h : N a ⊆ M) : 
  a = 0 ∨ a = -2 ∨ a = 1/2 := by sorry

end NUMINAMATH_CALUDE_subset_implies_a_values_l4134_413448


namespace NUMINAMATH_CALUDE_arrangementsWithRestrictionsCorrect_l4134_413414

/-- The number of ways to arrange 5 distinct items in a row with restrictions -/
def arrangementsWithRestrictions : ℕ :=
  let n := 5  -- total number of items
  let mustBeAdjacent := 2  -- number of items that must be adjacent
  let cannotBeAdjacent := 2  -- number of items that cannot be adjacent
  24  -- the result we want to prove

theorem arrangementsWithRestrictionsCorrect :
  arrangementsWithRestrictions = 24 :=
by sorry

end NUMINAMATH_CALUDE_arrangementsWithRestrictionsCorrect_l4134_413414


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l4134_413462

/-- 
Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
prove the values of a, b, and the area of the triangle.
-/
theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  c = 3 →
  C = π / 3 →
  Real.sin B = 2 * Real.sin A →
  (a = Real.sqrt 3 ∧ b = 2 * Real.sqrt 3) ∧
  (1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l4134_413462


namespace NUMINAMATH_CALUDE_quadratic_to_cubic_approximation_l4134_413489

/-- Given that x^2 - 6x + 1 can be approximated by a(x-h)^3 + k for some constants a and k,
    prove that h = 2. -/
theorem quadratic_to_cubic_approximation (x : ℝ) :
  ∃ (a k : ℝ), (∀ ε > 0, ∃ δ > 0, ∀ x, |x - 3| < δ → 
    |x^2 - 6*x + 1 - (a * (x - 2)^3 + k)| < ε) →
  2 = 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_to_cubic_approximation_l4134_413489


namespace NUMINAMATH_CALUDE_bakery_pie_distribution_l4134_413483

theorem bakery_pie_distribution (initial_pie : ℚ) (additional_percentage : ℚ) (num_employees : ℕ) :
  initial_pie = 8/9 →
  additional_percentage = 1/10 →
  num_employees = 4 →
  (initial_pie + initial_pie * additional_percentage) / num_employees = 11/45 := by
  sorry

end NUMINAMATH_CALUDE_bakery_pie_distribution_l4134_413483


namespace NUMINAMATH_CALUDE_exists_close_to_integer_l4134_413454

theorem exists_close_to_integer (a : ℝ) (n : ℕ) (ha : a > 0) (hn : n > 1) :
  ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ ∃ m : ℤ, |k * a - m| ≤ 1 / n := by
  sorry

end NUMINAMATH_CALUDE_exists_close_to_integer_l4134_413454


namespace NUMINAMATH_CALUDE_binary_multiplication_l4134_413459

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Represents a binary number as a list of bits -/
def binary_1101 : List Bool := [true, false, true, true]
def binary_111 : List Bool := [true, true, true]
def binary_10010111 : List Bool := [true, true, true, false, true, false, false, true]

/-- Theorem stating that the product of 1101₂ and 111₂ is equal to 10010111₂ -/
theorem binary_multiplication :
  (binary_to_decimal binary_1101) * (binary_to_decimal binary_111) =
  binary_to_decimal binary_10010111 := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_l4134_413459


namespace NUMINAMATH_CALUDE_square_dissection_existence_l4134_413473

theorem square_dissection_existence :
  ∃ (S a b c : ℝ), 
    S > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    S^2 = a^2 + 3*b^2 + 5*c^2 :=
by sorry

end NUMINAMATH_CALUDE_square_dissection_existence_l4134_413473


namespace NUMINAMATH_CALUDE_first_applicant_earnings_l4134_413403

def first_applicant_salary : ℕ := 42000
def first_applicant_training_months : ℕ := 3
def first_applicant_training_cost_per_month : ℕ := 1200

def second_applicant_salary : ℕ := 45000
def second_applicant_revenue : ℕ := 92000
def second_applicant_bonus_percentage : ℚ := 1 / 100

def difference_in_earnings : ℕ := 850

theorem first_applicant_earnings :
  let first_total_cost := first_applicant_salary + first_applicant_training_months * first_applicant_training_cost_per_month
  let second_total_cost := second_applicant_salary + (second_applicant_salary : ℚ) * second_applicant_bonus_percentage
  let second_net_earnings := second_applicant_revenue - second_total_cost
  first_total_cost + (second_net_earnings - difference_in_earnings) = 45700 :=
by sorry

end NUMINAMATH_CALUDE_first_applicant_earnings_l4134_413403


namespace NUMINAMATH_CALUDE_simplify_polynomial_l4134_413433

theorem simplify_polynomial (x : ℝ) : 
  2 * x^2 * (4 * x^3 - 3 * x + 5) - 4 * (x^3 - x^2 + 3 * x - 8) = 
  8 * x^5 - 10 * x^3 + 14 * x^2 - 12 * x + 32 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l4134_413433


namespace NUMINAMATH_CALUDE_min_value_x_plus_3y_l4134_413490

theorem min_value_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 1) + 1 / (y + 1) = 1 / 4) :
  x + 3 * y ≥ 5 + 4 * Real.sqrt 3 ∧
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧
    1 / (x₀ + 1) + 1 / (y₀ + 1) = 1 / 4 ∧
    x₀ + 3 * y₀ = 5 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_3y_l4134_413490


namespace NUMINAMATH_CALUDE_min_value_of_expression_l4134_413406

theorem min_value_of_expression (a : ℝ) (h : a > 1) : 
  a + 1 / (a - 1) ≥ 3 ∧ ∃ a₀ > 1, a₀ + 1 / (a₀ - 1) = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l4134_413406


namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l4134_413497

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a7 (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 3 * a 11 = 16 →
  a 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a7_l4134_413497


namespace NUMINAMATH_CALUDE_assistant_end_time_l4134_413478

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a bracelet producer -/
structure Producer where
  startTime : Time
  endTime : Time
  rate : Nat
  interval : Nat
  deriving Repr

def craftsman : Producer := {
  startTime := { hours := 8, minutes := 0 }
  endTime := { hours := 12, minutes := 0 }
  rate := 6
  interval := 20
}

def assistant : Producer := {
  startTime := { hours := 9, minutes := 0 }
  endTime := { hours := 0, minutes := 0 }  -- To be determined
  rate := 8
  interval := 30
}

def calculateProduction (p : Producer) : Nat :=
  sorry

def calculateEndTime (p : Producer) (targetProduction : Nat) : Time :=
  sorry

theorem assistant_end_time :
  calculateEndTime assistant (calculateProduction craftsman) = { hours := 13, minutes := 30 } :=
sorry

end NUMINAMATH_CALUDE_assistant_end_time_l4134_413478


namespace NUMINAMATH_CALUDE_cube_diagonal_length_l4134_413496

theorem cube_diagonal_length (surface_area : ℝ) (h : surface_area = 864) :
  let side_length := Real.sqrt (surface_area / 6)
  let diagonal_length := side_length * Real.sqrt 3
  diagonal_length = 12 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_diagonal_length_l4134_413496


namespace NUMINAMATH_CALUDE_quadratic_form_constant_l4134_413481

theorem quadratic_form_constant (x : ℝ) :
  ∃ (a h k : ℝ), x^2 - 7*x = a*(x - h)^2 + k ∧ k = -49/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_constant_l4134_413481


namespace NUMINAMATH_CALUDE_original_equals_scientific_l4134_413443

/-- Represents 1 million -/
def million : ℝ := 10^6

/-- The number to be converted -/
def original_number : ℝ := 456.87 * million

/-- The scientific notation representation -/
def scientific_notation : ℝ := 4.5687 * 10^8

theorem original_equals_scientific : original_number = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l4134_413443


namespace NUMINAMATH_CALUDE_willam_tax_is_960_l4134_413455

/-- Represents the farm tax scenario in Mr. Willam's village -/
structure FarmTax where
  -- Total taxable land in the village
  total_taxable_land : ℝ
  -- Tax rate per unit of taxable land
  tax_rate : ℝ
  -- Percentage of Mr. Willam's taxable land
  willam_land_percentage : ℝ

/-- Calculates Mr. Willam's tax payment -/
def willam_tax_payment (ft : FarmTax) : ℝ :=
  ft.total_taxable_land * ft.tax_rate * ft.willam_land_percentage

/-- Theorem stating that Mr. Willam's tax payment is $960 -/
theorem willam_tax_is_960 (ft : FarmTax) 
    (h1 : ft.tax_rate * ft.total_taxable_land = 3840)
    (h2 : ft.willam_land_percentage = 0.25) : 
  willam_tax_payment ft = 960 := by
  sorry


end NUMINAMATH_CALUDE_willam_tax_is_960_l4134_413455


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l4134_413468

theorem smallest_solution_of_equation (x : ℝ) :
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  (∀ y : ℝ, 1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4) → y ≥ x) →
  x = 4 - Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l4134_413468


namespace NUMINAMATH_CALUDE_line_opposite_sides_m_range_l4134_413416

/-- A line in 2D space defined by the equation 3x - 2y + m = 0 -/
structure Line (m : ℝ) where
  equation : ℝ → ℝ → ℝ
  eq_def : equation = fun x y => 3 * x - 2 * y + m

/-- Determines if two points are on opposite sides of a line -/
def opposite_sides (l : Line m) (p1 p2 : ℝ × ℝ) : Prop :=
  l.equation p1.1 p1.2 * l.equation p2.1 p2.2 < 0

theorem line_opposite_sides_m_range (m : ℝ) (l : Line m) :
  opposite_sides l (3, 1) (-4, 6) → -7 < m ∧ m < 24 := by
  sorry


end NUMINAMATH_CALUDE_line_opposite_sides_m_range_l4134_413416


namespace NUMINAMATH_CALUDE_twelfth_term_of_specific_arithmetic_sequence_l4134_413486

/-- An arithmetic sequence with a given first term and second term -/
def arithmeticSequence (a₁ : ℚ) (a₂ : ℚ) : ℕ → ℚ :=
  λ n => a₁ + (n - 1) * (a₂ - a₁)

/-- Theorem: The 12th term of the arithmetic sequence with first term 1/2 and second term 5/6 is 25/6 -/
theorem twelfth_term_of_specific_arithmetic_sequence :
  arithmeticSequence (1/2) (5/6) 12 = 25/6 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_of_specific_arithmetic_sequence_l4134_413486


namespace NUMINAMATH_CALUDE_total_words_eq_443_l4134_413424

def count_words (n : ℕ) : ℕ :=
  if n ≤ 20 ∨ n = 30 ∨ n = 40 ∨ n = 50 ∨ n = 60 ∨ n = 70 ∨ n = 80 ∨ n = 90 ∨ n = 100 ∨ n = 200 then 1
  else if n ≤ 99 then 2
  else if n ≤ 199 then 3
  else 0

def total_words : ℕ := (List.range 200).map (λ i => count_words (i + 1)) |>.sum

theorem total_words_eq_443 : total_words = 443 := by
  sorry

end NUMINAMATH_CALUDE_total_words_eq_443_l4134_413424


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_3_and_8_l4134_413428

theorem smallest_four_digit_divisible_by_3_and_8 :
  ∃ n : ℕ, 
    (1000 ≤ n ∧ n < 10000) ∧ 
    n % 3 = 0 ∧ 
    n % 8 = 0 ∧
    (∀ m : ℕ, (1000 ≤ m ∧ m < 10000) → m % 3 = 0 → m % 8 = 0 → n ≤ m) ∧
    n = 1008 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_3_and_8_l4134_413428


namespace NUMINAMATH_CALUDE_weight_2019_is_9_5_l4134_413409

/-- The weight of a single stick in kilograms -/
def stick_weight : ℝ := 0.5

/-- The number of sticks used to form each digit -/
def sticks_per_digit : Fin 10 → ℕ
  | 0 => 6
  | 1 => 2
  | 2 => 5
  | 9 => 6
  | _ => 0  -- We only care about digits 0, 1, 2, and 9 for this problem

/-- The weight of the number 2019 in kilograms -/
def weight_2019 : ℝ :=
  (sticks_per_digit 2 + sticks_per_digit 0 + sticks_per_digit 1 + sticks_per_digit 9) * stick_weight

/-- The theorem stating that the weight of 2019 is 9.5 kg -/
theorem weight_2019_is_9_5 : weight_2019 = 9.5 := by
  sorry

#eval weight_2019

end NUMINAMATH_CALUDE_weight_2019_is_9_5_l4134_413409


namespace NUMINAMATH_CALUDE_same_month_same_gender_exists_l4134_413498

/-- Represents a student with their gender and birth month. -/
structure Student where
  gender : Bool  -- True for girl, False for boy
  birthMonth : Fin 12

/-- Theorem: In a class of 25 students, there must be at least two girls
    or two boys born in the same month. -/
theorem same_month_same_gender_exists (students : Finset Student)
    (h_count : students.card = 25) :
    (∃ (m : Fin 12), 2 ≤ (students.filter (fun s => s.gender ∧ s.birthMonth = m)).card) ∨
    (∃ (m : Fin 12), 2 ≤ (students.filter (fun s => ¬s.gender ∧ s.birthMonth = m)).card) :=
  sorry


end NUMINAMATH_CALUDE_same_month_same_gender_exists_l4134_413498


namespace NUMINAMATH_CALUDE_opposite_of_negative_six_l4134_413418

theorem opposite_of_negative_six : -(- 6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_six_l4134_413418


namespace NUMINAMATH_CALUDE_average_weight_increase_l4134_413476

theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 8 * initial_average
  let new_total := initial_total - 65 + 85
  let new_average := new_total / 8
  new_average - initial_average = 2.5 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l4134_413476


namespace NUMINAMATH_CALUDE_ac_value_l4134_413430

theorem ac_value (x : ℕ+) 
  (h1 : ∃ y : ℕ, (2 * x + 1 : ℕ) = y^2)
  (h2 : ∃ z : ℕ, (3 * x + 1 : ℕ) = z^2) : 
  x = 40 := by
sorry

end NUMINAMATH_CALUDE_ac_value_l4134_413430


namespace NUMINAMATH_CALUDE_circumscribed_sphere_area_l4134_413404

theorem circumscribed_sphere_area (x y z : ℝ) (h1 : x * y = 6) (h2 : y * z = 10) (h3 : z * x = 15) :
  4 * Real.pi * ((x^2 + y^2 + z^2) / 4) = 38 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_area_l4134_413404


namespace NUMINAMATH_CALUDE_middle_person_height_l4134_413475

/-- Represents the heights of 5 people in a line -/
def Heights := Fin 5 → ℝ

/-- The condition that the heights form a geometric sequence -/
def is_geometric_sequence (h : Heights) : Prop :=
  ∃ r : ℝ, ∀ i : Fin 4, h (i + 1) = h i * r

theorem middle_person_height (h : Heights) :
  (∀ i j : Fin 5, i < j → h i ≤ h j) →  -- Heights are in ascending order
  h 0 = 3 →  -- Shortest person is 3 feet tall
  h 4 = 7 →  -- Tallest person is 7 feet tall
  is_geometric_sequence h →  -- Heights form a geometric sequence
  h 2 = Real.sqrt 21 :=  -- Middle person's height is √21 feet
by sorry

end NUMINAMATH_CALUDE_middle_person_height_l4134_413475


namespace NUMINAMATH_CALUDE_trig_expression_equals_sqrt2_over_2_l4134_413492

theorem trig_expression_equals_sqrt2_over_2 :
  (Real.sin (20 * π / 180)) * Real.sqrt (1 + Real.cos (40 * π / 180)) / (Real.cos (50 * π / 180)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_sqrt2_over_2_l4134_413492


namespace NUMINAMATH_CALUDE_pentagon_area_sum_l4134_413485

/-- A pentagon formed by 11 segments of length 2 -/
structure Pentagon where
  /-- The number of segments forming the pentagon -/
  num_segments : ℕ
  /-- The length of each segment -/
  segment_length : ℝ
  /-- The area of the pentagon -/
  area : ℝ
  /-- The first positive integer in the area expression -/
  m : ℕ
  /-- The second positive integer in the area expression -/
  n : ℕ
  /-- Condition: The number of segments is 11 -/
  h_num_segments : num_segments = 11
  /-- Condition: The length of each segment is 2 -/
  h_segment_length : segment_length = 2
  /-- Condition: The area is expressed as √m + √n -/
  h_area : area = Real.sqrt m + Real.sqrt n
  /-- Condition: m is positive -/
  h_m_pos : m > 0
  /-- Condition: n is positive -/
  h_n_pos : n > 0

/-- Theorem: For the given pentagon, m + n = 23 -/
theorem pentagon_area_sum (p : Pentagon) : p.m + p.n = 23 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_sum_l4134_413485


namespace NUMINAMATH_CALUDE_triangle_area_determines_p_l4134_413474

/-- Given a triangle ABC with vertices A(3, 15), B(15, 0), and C(0, p),
    prove that if the area of the triangle is 36, then p = 12.75 -/
theorem triangle_area_determines_p :
  let A : ℝ × ℝ := (3, 15)
  let B : ℝ × ℝ := (15, 0)
  let C : ℝ × ℝ := (0, p)
  let triangle_area (A B C : ℝ × ℝ) : ℝ :=
    (1/2) * abs ((A.1 - C.1) * (B.2 - C.2) - (B.1 - C.1) * (A.2 - C.2))
  ∀ p : ℝ, triangle_area A B C = 36 → p = 12.75 := by
  sorry

#check triangle_area_determines_p

end NUMINAMATH_CALUDE_triangle_area_determines_p_l4134_413474


namespace NUMINAMATH_CALUDE_intersection_line_l4134_413458

-- Define the first circle
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0

-- Define the second circle
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the line
def line (x y : ℝ) : Prop := x - y - 3 = 0

-- Theorem statement
theorem intersection_line : 
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_l4134_413458


namespace NUMINAMATH_CALUDE_least_n_square_and_cube_n_144_satisfies_least_n_is_144_l4134_413452

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem least_n_square_and_cube :
  ∀ n : ℕ, n > 0 →
    (is_perfect_square (9*n) ∧ is_perfect_cube (12*n)) →
    n ≥ 144 :=
by sorry

theorem n_144_satisfies :
  is_perfect_square (9*144) ∧ is_perfect_cube (12*144) :=
by sorry

theorem least_n_is_144 :
  ∀ n : ℕ, n > 0 →
    (is_perfect_square (9*n) ∧ is_perfect_cube (12*n)) →
    n = 144 :=
by sorry

end NUMINAMATH_CALUDE_least_n_square_and_cube_n_144_satisfies_least_n_is_144_l4134_413452


namespace NUMINAMATH_CALUDE_cos_75_deg_l4134_413401

/-- Proves that cos 75° = (√6 - √2) / 4 using cos 60° and cos 15° -/
theorem cos_75_deg (cos_60_deg : Real) (cos_15_deg : Real) :
  cos_60_deg = 1 / 2 →
  cos_15_deg = (Real.sqrt 6 + Real.sqrt 2) / 4 →
  Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_deg_l4134_413401


namespace NUMINAMATH_CALUDE_prob_twice_daughters_is_37_256_l4134_413436

-- Define the number of children
def num_children : ℕ := 8

-- Define the probability of having a daughter (equal to the probability of having a son)
def p_daughter : ℚ := 1/2

-- Define the function to calculate the probability of having exactly k daughters out of n children
def prob_k_daughters (n k : ℕ) : ℚ :=
  (n.choose k) * (p_daughter ^ k) * ((1 - p_daughter) ^ (n - k))

-- Define the probability of having at least twice as many daughters as sons
def prob_twice_daughters : ℚ :=
  prob_k_daughters num_children num_children +
  prob_k_daughters num_children (num_children - 1) +
  prob_k_daughters num_children (num_children - 2)

-- Theorem statement
theorem prob_twice_daughters_is_37_256 : prob_twice_daughters = 37/256 := by
  sorry

end NUMINAMATH_CALUDE_prob_twice_daughters_is_37_256_l4134_413436


namespace NUMINAMATH_CALUDE_fuel_cost_savings_l4134_413477

theorem fuel_cost_savings (old_efficiency : ℝ) (old_fuel_cost : ℝ) 
  (h1 : old_efficiency > 0) (h2 : old_fuel_cost > 0) : 
  let new_efficiency := 1.5 * old_efficiency
  let new_fuel_cost := 1.2 * old_fuel_cost
  let old_trip_cost := (1 / old_efficiency) * old_fuel_cost
  let new_trip_cost := (1 / new_efficiency) * new_fuel_cost
  (old_trip_cost - new_trip_cost) / old_trip_cost = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_fuel_cost_savings_l4134_413477


namespace NUMINAMATH_CALUDE_first_cousin_ate_two_l4134_413491

/-- The number of sandwiches Ruth prepared -/
def total_sandwiches : ℕ := 10

/-- The number of sandwiches Ruth ate -/
def ruth_ate : ℕ := 1

/-- The number of sandwiches Ruth gave to her brother -/
def brother_ate : ℕ := 2

/-- The number of sandwiches eaten by the two other cousins -/
def other_cousins_ate : ℕ := 2

/-- The number of sandwiches left -/
def sandwiches_left : ℕ := 3

/-- The number of sandwiches eaten by the first cousin -/
def first_cousin_ate : ℕ := total_sandwiches - (ruth_ate + brother_ate + other_cousins_ate + sandwiches_left)

theorem first_cousin_ate_two : first_cousin_ate = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_cousin_ate_two_l4134_413491


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l4134_413451

/-- A geometric sequence -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

theorem geometric_sequence_property 
  (b : ℕ → ℝ) (m n p : ℕ) 
  (h_geometric : GeometricSequence b)
  (h_distinct : m ≠ n ∧ n ≠ p ∧ m ≠ p)
  (h_positive : 0 < m ∧ 0 < n ∧ 0 < p) :
  (b p) ^ (m - n) * (b m) ^ (n - p) * (b n) ^ (p - m) = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l4134_413451


namespace NUMINAMATH_CALUDE_fraction_equality_implies_conditions_l4134_413417

theorem fraction_equality_implies_conditions (a b c d : ℝ) :
  (2*a + b) / (b + 2*c) = (c + 3*d) / (4*d + a) →
  (a = c ∨ 3*a + 4*b + 5*c + 6*d = 0) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_conditions_l4134_413417


namespace NUMINAMATH_CALUDE_parity_of_S_l4134_413484

theorem parity_of_S (a b c n : ℤ) 
  (h1 : (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 0) ∨ 
        (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1) ∨ 
        (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 1)) :
  let S := (a + 2*n + 1) * (b + 2*n + 2) * (c + 2*n + 3)
  S % 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_parity_of_S_l4134_413484


namespace NUMINAMATH_CALUDE_minimum_books_in_library_l4134_413457

theorem minimum_books_in_library (physics chemistry biology : ℕ) : 
  physics + chemistry + biology > 0 →
  3 * chemistry = 2 * physics →
  4 * biology = 3 * chemistry →
  ∃ (k : ℕ), k * (physics + chemistry + biology) = 3003 →
  3003 ≤ physics + chemistry + biology :=
by sorry

end NUMINAMATH_CALUDE_minimum_books_in_library_l4134_413457


namespace NUMINAMATH_CALUDE_train_crossing_tree_time_l4134_413432

/-- Given a train and a platform with specific properties, calculate the time it takes for the train to cross a tree. -/
theorem train_crossing_tree_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (time_pass_platform : ℝ) 
  (h1 : train_length = 1200)
  (h2 : platform_length = 400)
  (h3 : time_pass_platform = 160) :
  (train_length / ((train_length + platform_length) / time_pass_platform)) = 120 := by
  sorry

#check train_crossing_tree_time

end NUMINAMATH_CALUDE_train_crossing_tree_time_l4134_413432


namespace NUMINAMATH_CALUDE_min_toothpicks_theorem_l4134_413435

/-- A geometric figure made of toothpicks -/
structure ToothpickFigure where
  upward_triangles : ℕ
  downward_triangles : ℕ
  horizontal_toothpicks : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (figure : ToothpickFigure) : ℕ :=
  figure.horizontal_toothpicks

/-- Theorem stating the minimum number of toothpicks to remove -/
theorem min_toothpicks_theorem (figure : ToothpickFigure) 
  (h1 : figure.upward_triangles = 15)
  (h2 : figure.downward_triangles = 10)
  (h3 : figure.horizontal_toothpicks = 15) :
  min_toothpicks_to_remove figure = 15 := by
  sorry

#check min_toothpicks_theorem

end NUMINAMATH_CALUDE_min_toothpicks_theorem_l4134_413435


namespace NUMINAMATH_CALUDE_negative_fractions_comparison_l4134_413446

theorem negative_fractions_comparison : -4/5 < -2/3 := by
  sorry

end NUMINAMATH_CALUDE_negative_fractions_comparison_l4134_413446


namespace NUMINAMATH_CALUDE_trash_time_fraction_l4134_413488

def movie_time : ℕ := 120 -- 2 hours in minutes
def homework_time : ℕ := 30
def cleaning_time : ℕ := homework_time / 2
def dog_walking_time : ℕ := homework_time + 5
def time_left : ℕ := 35

def total_known_tasks : ℕ := homework_time + cleaning_time + dog_walking_time

theorem trash_time_fraction (trash_time : ℕ) : 
  trash_time = movie_time - time_left - total_known_tasks →
  trash_time * 6 = homework_time :=
by sorry

end NUMINAMATH_CALUDE_trash_time_fraction_l4134_413488


namespace NUMINAMATH_CALUDE_unequal_grandchildren_probability_l4134_413461

/-- The number of grandchildren -/
def n : ℕ := 12

/-- The probability of a child being male or female -/
def p : ℚ := 1/2

/-- The probability of having an unequal number of grandsons and granddaughters -/
def unequal_probability : ℚ := 793/1024

theorem unequal_grandchildren_probability :
  (1 : ℚ) - (n.choose (n/2) : ℚ) / (2^n : ℚ) = unequal_probability :=
sorry

end NUMINAMATH_CALUDE_unequal_grandchildren_probability_l4134_413461


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_3913_l4134_413472

theorem largest_prime_factor_of_3913 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 3913 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 3913 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_3913_l4134_413472


namespace NUMINAMATH_CALUDE_problem_statement_l4134_413450

theorem problem_statement (a b c : ℝ) 
  (h1 : a^2 - b^2 = 10) 
  (h2 : a * b = 5) 
  (h3 : a^2 + b^2 + c^2 = 20) : 
  a^4 + b^4 + c^4 = 150 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4134_413450


namespace NUMINAMATH_CALUDE_power_calculation_l4134_413495

theorem power_calculation : (8^5 / 8^2) * 4^6 = 2^21 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l4134_413495


namespace NUMINAMATH_CALUDE_point_P_in_fourth_quadrant_iff_a_in_range_l4134_413440

/-- A point in the fourth quadrant has a positive x-coordinate and a negative y-coordinate -/
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The coordinates of point P are defined in terms of parameter a -/
def point_P (a : ℝ) : ℝ × ℝ := (2*a + 4, 3*a - 6)

/-- Theorem stating the range of a for point P to be in the fourth quadrant -/
theorem point_P_in_fourth_quadrant_iff_a_in_range :
  ∀ a : ℝ, fourth_quadrant (point_P a).1 (point_P a).2 ↔ -2 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_point_P_in_fourth_quadrant_iff_a_in_range_l4134_413440


namespace NUMINAMATH_CALUDE_same_color_probability_l4134_413460

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  blue : ℕ
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total number of jelly beans a person has -/
def JellyBeans.total (jb : JellyBeans) : ℕ :=
  jb.blue + jb.green + jb.yellow + jb.red

/-- Represents the jelly bean distribution for each person -/
def abe : JellyBeans := { blue := 2, green := 1, yellow := 0, red := 0 }
def bob : JellyBeans := { blue := 1, green := 2, yellow := 1, red := 0 }
def cara : JellyBeans := { blue := 3, green := 2, yellow := 0, red := 1 }

/-- Calculates the probability of picking a specific color for a person -/
def prob_pick_color (jb : JellyBeans) (color : ℕ) : ℚ :=
  color / jb.total

/-- Theorem: The probability of all three people picking jelly beans of the same color is 5/36 -/
theorem same_color_probability :
  (prob_pick_color abe abe.blue * prob_pick_color bob bob.blue * prob_pick_color cara cara.blue) +
  (prob_pick_color abe abe.green * prob_pick_color bob bob.green * prob_pick_color cara cara.green) =
  5 / 36 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l4134_413460


namespace NUMINAMATH_CALUDE_average_of_remaining_checks_l4134_413471

def travelers_checks_problem (x y z : ℕ) : Prop :=
  x + y = 30 ∧ 
  50 * x + z * y = 1800 ∧ 
  x ≥ 24 ∧
  z > 0

theorem average_of_remaining_checks (x y z : ℕ) 
  (h : travelers_checks_problem x y z) : 
  (1800 - 50 * 24) / (30 - 24) = 100 :=
sorry

end NUMINAMATH_CALUDE_average_of_remaining_checks_l4134_413471


namespace NUMINAMATH_CALUDE_fraction_sum_to_ratio_proof_l4134_413400

theorem fraction_sum_to_ratio_proof (x y : ℝ) 
  (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) : 
  (x + y) / (x - y) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_to_ratio_proof_l4134_413400


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4134_413439

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 2 → a^2 > 2*a) ∧ 
  (∃ a, a ≤ 2 ∧ a^2 > 2*a) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4134_413439


namespace NUMINAMATH_CALUDE_lightning_rod_height_l4134_413405

/-- Given a lightning rod that breaks twice under strong wind conditions, 
    this theorem proves the height of the rod. -/
theorem lightning_rod_height (h : ℝ) (x₁ : ℝ) (x₂ : ℝ) : 
  h > 0 → 
  x₁ > 0 → 
  x₂ > 0 → 
  h^2 - x₁^2 = 400 → 
  h^2 - x₂^2 = 900 → 
  x₂ = x₁ - 5 → 
  h = Real.sqrt 3156.25 := by
sorry

end NUMINAMATH_CALUDE_lightning_rod_height_l4134_413405


namespace NUMINAMATH_CALUDE_students_taking_no_subjects_l4134_413466

theorem students_taking_no_subjects (total : ℕ) (music art sports : ℕ) 
  (music_and_art music_and_sports art_and_sports : ℕ) (all_three : ℕ) : 
  total = 1200 →
  music = 60 →
  art = 80 →
  sports = 30 →
  music_and_art = 25 →
  music_and_sports = 15 →
  art_and_sports = 20 →
  all_three = 10 →
  total - (music + art + sports - music_and_art - music_and_sports - art_and_sports + all_three) = 1080 := by
  sorry

#check students_taking_no_subjects

end NUMINAMATH_CALUDE_students_taking_no_subjects_l4134_413466


namespace NUMINAMATH_CALUDE_orange_bin_problem_l4134_413412

theorem orange_bin_problem (initial_oranges : ℕ) : 
  initial_oranges - 2 + 28 = 31 → initial_oranges = 5 := by
  sorry

end NUMINAMATH_CALUDE_orange_bin_problem_l4134_413412


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4134_413410

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x + 3 > 2 → -x < 6) ∧ 
  (∃ x : ℝ, -x < 6 ∧ ¬(x + 3 > 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4134_413410


namespace NUMINAMATH_CALUDE_nell_remaining_cards_l4134_413437

/-- Proves that Nell has 276 cards after giving away 28 cards from her initial 304 cards. -/
theorem nell_remaining_cards (initial_cards : ℕ) (cards_given : ℕ) (remaining_cards : ℕ) : 
  initial_cards = 304 → cards_given = 28 → remaining_cards = initial_cards - cards_given → remaining_cards = 276 := by
  sorry

end NUMINAMATH_CALUDE_nell_remaining_cards_l4134_413437


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4134_413467

theorem right_triangle_hypotenuse (a b h : ℝ) : 
  a = 24 → b = 10 → h^2 = a^2 + b^2 → h = 26 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4134_413467


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l4134_413465

theorem unique_six_digit_number : ∃! n : ℕ, 
  (100000 ≤ n ∧ n < 1000000) ∧ 
  (n / 100000 = 2) ∧ 
  ((n % 100000) * 10 + 2 = 3 * n) := by
  sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l4134_413465


namespace NUMINAMATH_CALUDE_roots_product_l4134_413463

-- Define the logarithm function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop := 
  (lg x)^2 + (lg 5 + lg 7) * (lg x) + (lg 5) * (lg 7) = 0

-- Theorem statement
theorem roots_product (m n : ℝ) : 
  equation m ∧ equation n ∧ m ≠ n → m * n = 1/35 := by
  sorry

end NUMINAMATH_CALUDE_roots_product_l4134_413463


namespace NUMINAMATH_CALUDE_lowella_score_l4134_413469

/-- Given a 100-item exam, prove that Lowella's score is 22% when:
    - Pamela's score is 20 percentage points higher than Lowella's
    - Mandy's score is twice Pamela's score
    - Mandy's score is 84% -/
theorem lowella_score (pamela_score mandy_score lowella_score : ℚ) : 
  pamela_score = lowella_score + 20 →
  mandy_score = 2 * pamela_score →
  mandy_score = 84 →
  lowella_score = 22 := by
  sorry

#check lowella_score

end NUMINAMATH_CALUDE_lowella_score_l4134_413469


namespace NUMINAMATH_CALUDE_fishing_result_l4134_413438

/-- The total number of fishes Will and Henry have after fishing -/
def total_fishes (will_catfish : ℕ) (will_eels : ℕ) (henry_trout_ratio : ℕ) : ℕ :=
  let will_total := will_catfish + will_eels
  let henry_total := will_catfish * henry_trout_ratio
  let henry_kept := henry_total / 2
  will_total + henry_kept

/-- Theorem stating the total number of fishes Will and Henry have -/
theorem fishing_result : total_fishes 16 10 3 = 50 := by
  sorry

#eval total_fishes 16 10 3

end NUMINAMATH_CALUDE_fishing_result_l4134_413438


namespace NUMINAMATH_CALUDE_probability_two_black_balls_is_three_tenths_l4134_413494

def total_balls : ℕ := 16
def black_balls : ℕ := 9

def probability_two_black_balls : ℚ :=
  (black_balls.choose 2) / (total_balls.choose 2)

theorem probability_two_black_balls_is_three_tenths :
  probability_two_black_balls = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_black_balls_is_three_tenths_l4134_413494


namespace NUMINAMATH_CALUDE_john_initial_money_l4134_413493

theorem john_initial_money (spent : ℕ) (left : ℕ) : 
  left = 500 → 
  spent = left + 600 → 
  spent + left = 1600 :=
by
  sorry

end NUMINAMATH_CALUDE_john_initial_money_l4134_413493


namespace NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l4134_413415

/-- Represents a square grid with shaded squares -/
structure SquareGrid :=
  (size : ℕ)
  (shaded : Set (ℕ × ℕ))

/-- Checks if a SquareGrid has at least one line of symmetry -/
def has_line_symmetry (grid : SquareGrid) : Prop :=
  sorry

/-- Checks if a SquareGrid has rotational symmetry of order 2 -/
def has_rotational_symmetry_order_2 (grid : SquareGrid) : Prop :=
  sorry

/-- Counts the number of additional squares shaded -/
def count_additional_shaded (initial grid : SquareGrid) : ℕ :=
  sorry

/-- The main theorem stating the minimum number of additional squares to be shaded -/
theorem min_additional_squares_for_symmetry (initial : SquareGrid) :
  ∃ (final : SquareGrid),
    (has_line_symmetry final ∧ has_rotational_symmetry_order_2 final) ∧
    (count_additional_shaded initial final = 3) ∧
    (∀ (other : SquareGrid),
      (has_line_symmetry other ∧ has_rotational_symmetry_order_2 other) →
      count_additional_shaded initial other ≥ 3) :=
  sorry

end NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l4134_413415


namespace NUMINAMATH_CALUDE_sum_interior_angles_convex_polygon_l4134_413429

theorem sum_interior_angles_convex_polygon (n : ℕ) (h : n = 10) :
  (∃ (v : ℕ), v + 3 = n ∧ v = 7) →
  (n - 2) * 180 = 1440 :=
sorry

end NUMINAMATH_CALUDE_sum_interior_angles_convex_polygon_l4134_413429


namespace NUMINAMATH_CALUDE_student_selection_probability_l4134_413413

/-- Given a set of 3 students where 2 are to be selected, 
    the probability of a specific student being selected is 2/3 -/
theorem student_selection_probability 
  (S : Finset Nat) 
  (h_card : S.card = 3) 
  (A : Nat) 
  (h_A_in_S : A ∈ S) : 
  Nat.card {pair : Finset Nat | pair ⊆ S ∧ pair.card = 2 ∧ A ∈ pair} / 
  Nat.card {pair : Finset Nat | pair ⊆ S ∧ pair.card = 2} = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_probability_l4134_413413


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l4134_413422

theorem expression_simplification_and_evaluation :
  let x : ℚ := 1/3
  let y : ℚ := -6
  let original_expression := 3 * x^2 * y - (6 * x * y^2 - 2 * (x * y + 3/2 * x^2 * y)) + 2 * (3 * x * y^2 - x * y)
  let simplified_expression := 6 * x^2 * y
  original_expression = simplified_expression ∧ simplified_expression = -4 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l4134_413422


namespace NUMINAMATH_CALUDE_room_occupancy_l4134_413482

theorem room_occupancy (total_chairs : ℕ) (seated_people : ℕ) (total_people : ℕ) :
  total_chairs = 25 →
  seated_people = (4 : ℕ) * total_chairs / 5 →
  seated_people = (3 : ℕ) * total_people / 5 →
  total_people = 33 :=
by
  sorry

#check room_occupancy

end NUMINAMATH_CALUDE_room_occupancy_l4134_413482


namespace NUMINAMATH_CALUDE_time_before_first_rewind_is_35_l4134_413445

/-- Represents the viewing time of a movie with interruptions -/
structure MovieViewing where
  totalTime : ℕ
  firstRewindTime : ℕ
  timeBetweenRewinds : ℕ
  secondRewindTime : ℕ
  timeAfterSecondRewind : ℕ

/-- Calculates the time watched before the first rewind -/
def timeBeforeFirstRewind (mv : MovieViewing) : ℕ :=
  mv.totalTime - (mv.firstRewindTime + mv.timeBetweenRewinds + mv.secondRewindTime + mv.timeAfterSecondRewind)

/-- Theorem stating that for the given movie viewing scenario, 
    the time watched before the first rewind is 35 minutes -/
theorem time_before_first_rewind_is_35 : 
  let mv : MovieViewing := {
    totalTime := 120,
    firstRewindTime := 5,
    timeBetweenRewinds := 45,
    secondRewindTime := 15,
    timeAfterSecondRewind := 20
  }
  timeBeforeFirstRewind mv = 35 := by
  sorry

end NUMINAMATH_CALUDE_time_before_first_rewind_is_35_l4134_413445


namespace NUMINAMATH_CALUDE_two_segment_train_journey_time_l4134_413411

/-- Calculates the total time for a two-segment train journey -/
theorem two_segment_train_journey_time
  (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ)
  (h1 : distance1 = 80)
  (h2 : speed1 = 50)
  (h3 : distance2 = 150)
  (h4 : speed2 = 75)
  (h5 : speed1 > 0)
  (h6 : speed2 > 0) :
  distance1 / speed1 + distance2 / speed2 = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_two_segment_train_journey_time_l4134_413411


namespace NUMINAMATH_CALUDE_friday_13th_more_frequent_l4134_413441

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a year in the Gregorian calendar -/
structure GregorianYear where
  year : Nat

/-- Determines if a given year is a leap year -/
def isLeapYear (y : GregorianYear) : Bool :=
  (y.year % 4 == 0 && y.year % 100 != 0) || y.year % 400 == 0

/-- Calculates the day of the week for the 13th of a given month in a given year -/
def dayOf13th (y : GregorianYear) (month : Nat) : DayOfWeek :=
  sorry

/-- Counts the frequency of each day of the week being the 13th over a 400-year cycle -/
def countDayOf13thIn400Years : DayOfWeek → Nat :=
  sorry

/-- Theorem: The 13th is more likely to be a Friday than any other day of the week -/
theorem friday_13th_more_frequent :
  ∀ d : DayOfWeek, d ≠ DayOfWeek.Friday →
    countDayOf13thIn400Years DayOfWeek.Friday > countDayOf13thIn400Years d :=
  sorry

end NUMINAMATH_CALUDE_friday_13th_more_frequent_l4134_413441


namespace NUMINAMATH_CALUDE_rational_expression_evaluation_l4134_413423

theorem rational_expression_evaluation :
  let x : ℝ := 7
  (x^8 + 18*x^4 + 81) / (x^4 + 9) = 2410 := by
  sorry

end NUMINAMATH_CALUDE_rational_expression_evaluation_l4134_413423


namespace NUMINAMATH_CALUDE_factorial_sum_equation_l4134_413447

theorem factorial_sum_equation : ∃ (S : Finset ℕ), (∀ n ∈ S, ∃ k l : ℕ, n.factorial / 2 = k.factorial + l.factorial) ∧ S.sum id = 10 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equation_l4134_413447


namespace NUMINAMATH_CALUDE_f_properties_l4134_413427

/-- The function f(x) that attains an extremum of 0 at x = -1 -/
def f (a b x : ℝ) : ℝ := x^3 + 2*a*x^2 + b*x + a - 1

/-- Theorem stating the properties of f(x) -/
theorem f_properties (a b : ℝ) :
  (f a b (-1) = 0 ∧ (deriv (f a b)) (-1) = 0) →
  (a = 1 ∧ b = 1 ∧ ∀ x ∈ Set.Icc (-1 : ℝ) 1, f 1 1 x ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l4134_413427


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l4134_413480

theorem lcm_factor_problem (A B : ℕ+) (X : ℕ) : 
  Nat.gcd A B = 23 →
  A = 230 →
  Nat.lcm A B = 23 * X * 10 →
  X = 1 := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l4134_413480


namespace NUMINAMATH_CALUDE_sum_of_cubes_l4134_413449

theorem sum_of_cubes (a b c : ℝ) 
  (h1 : a + b + c = 2) 
  (h2 : a * b + a * c + b * c = 5) 
  (h3 : a * b * c = -6) : 
  a^3 + b^3 + c^3 = -40 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l4134_413449


namespace NUMINAMATH_CALUDE_two_digit_multiple_plus_two_l4134_413453

theorem two_digit_multiple_plus_two : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ ∃ k : ℕ, n = 3 * 4 * 5 * k + 2 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_two_digit_multiple_plus_two_l4134_413453
