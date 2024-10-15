import Mathlib

namespace NUMINAMATH_CALUDE_smallest_difference_in_triangle_l2734_273430

theorem smallest_difference_in_triangle (XZ XY YZ : ℕ) : 
  XZ + XY + YZ = 3030 →
  XZ < XY →
  XY ≤ YZ →
  ∃ k : ℕ, XY = 5 * k →
  ∀ XZ' XY' YZ' : ℕ, 
    XZ' + XY' + YZ' = 3030 →
    XZ' < XY' →
    XY' ≤ YZ' →
    (∃ k' : ℕ, XY' = 5 * k') →
    XY - XZ ≤ XY' - XZ' :=
by sorry

end NUMINAMATH_CALUDE_smallest_difference_in_triangle_l2734_273430


namespace NUMINAMATH_CALUDE_school_girls_count_l2734_273479

theorem school_girls_count (boys : ℕ) (girls : ℝ) : 
  boys = 387 →
  girls = boys + 0.54 * boys →
  ⌊girls + 0.5⌋ = 596 := by
  sorry

end NUMINAMATH_CALUDE_school_girls_count_l2734_273479


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l2734_273487

theorem other_root_of_quadratic (m : ℝ) :
  (3 * (1 : ℝ)^2 + m * 1 = 5) →
  (3 * (-5/3 : ℝ)^2 + m * (-5/3) = 5) ∧
  (∀ x : ℝ, 3 * x^2 + m * x = 5 → x = 1 ∨ x = -5/3) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l2734_273487


namespace NUMINAMATH_CALUDE_equation_solution_l2734_273417

def solution_set : Set ℝ := {0, -6}

theorem equation_solution :
  ∀ x : ℝ, (2 * |x + 3| - 4 = 2) ↔ x ∈ solution_set := by
  sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l2734_273417


namespace NUMINAMATH_CALUDE_base8_to_base6_conversion_l2734_273424

-- Define a function to convert from base 8 to base 10
def base8ToBase10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

-- Define a function to convert from base 10 to base 6
def base10ToBase6 (n : ℕ) : ℕ :=
  (n / 216) * 1000 + ((n / 36) % 6) * 100 + ((n / 6) % 6) * 10 + (n % 6)

-- Theorem statement
theorem base8_to_base6_conversion :
  base10ToBase6 (base8ToBase10 753) = 2135 := by
  sorry

end NUMINAMATH_CALUDE_base8_to_base6_conversion_l2734_273424


namespace NUMINAMATH_CALUDE_valid_assignment_probability_l2734_273492

/-- A regular dodecahedron with 12 numbered faces -/
structure NumberedDodecahedron :=
  (assignment : Fin 12 → Fin 12)
  (injective : Function.Injective assignment)

/-- Two numbers are consecutive if they differ by 1 or are 1 and 12 -/
def consecutive (a b : Fin 12) : Prop :=
  (a + 1 = b) ∨ (b + 1 = a) ∨ (a = 0 ∧ b = 11) ∨ (a = 11 ∧ b = 0)

/-- The set of all possible numbered dodecahedrons -/
def allAssignments : Finset NumberedDodecahedron := sorry

/-- The set of valid assignments where no consecutive numbers are on adjacent faces -/
def validAssignments : Finset NumberedDodecahedron := sorry

/-- The probability of a valid assignment -/
def validProbability : ℚ := (validAssignments.card : ℚ) / (allAssignments.card : ℚ)

/-- The main theorem stating that the probability is 1/100 -/
theorem valid_assignment_probability :
  validProbability = 1 / 100 := by sorry

end NUMINAMATH_CALUDE_valid_assignment_probability_l2734_273492


namespace NUMINAMATH_CALUDE_solution_value_l2734_273454

theorem solution_value (x m : ℝ) : x = 3 ∧ (11 - 2*x = m*x - 1) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l2734_273454


namespace NUMINAMATH_CALUDE_seventh_root_unity_product_l2734_273433

theorem seventh_root_unity_product (s : ℂ) (h1 : s^7 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) * (s^6 - 1) = 7 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_unity_product_l2734_273433


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2734_273480

/-- A sequence is geometric if the ratio between any two consecutive terms is constant. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geometric : IsGeometric a) 
    (h_sum : a 4 + a 6 = 10) : 
  a 1 * a 7 + 2 * a 3 * a 7 + a 3 * a 9 = 100 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2734_273480


namespace NUMINAMATH_CALUDE_min_days_to_triple_debt_l2734_273435

/-- The borrowed amount in dollars -/
def borrowed_amount : ℝ := 15

/-- The daily interest rate as a decimal -/
def daily_interest_rate : ℝ := 0.1

/-- Calculate the amount owed after a given number of days -/
def amount_owed (days : ℝ) : ℝ :=
  borrowed_amount * (1 + daily_interest_rate * days)

/-- The minimum number of days needed to owe at least triple the borrowed amount -/
def min_days : ℕ := 20

theorem min_days_to_triple_debt :
  (∀ d : ℕ, d < min_days → amount_owed d < 3 * borrowed_amount) ∧
  amount_owed min_days ≥ 3 * borrowed_amount :=
sorry

end NUMINAMATH_CALUDE_min_days_to_triple_debt_l2734_273435


namespace NUMINAMATH_CALUDE_range_of_a_l2734_273489

theorem range_of_a (a : ℝ) : 
  (∃ x₀ : ℝ, 2 * x₀^2 + (a - 1) * x₀ + 1/2 ≤ 0) ↔ 
  a ≤ -1 ∨ a ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2734_273489


namespace NUMINAMATH_CALUDE_first_operation_result_l2734_273427

theorem first_operation_result (x : ℝ) : (x - 24) / 10 = 3 → (x - 5) / 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_first_operation_result_l2734_273427


namespace NUMINAMATH_CALUDE_f_comp_three_roots_l2734_273490

/-- A quadratic function f(x) = x^2 + 6x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 6*x + c

/-- The composition of f with itself -/
def f_comp (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

/-- The number of distinct real roots of f_comp -/
noncomputable def num_distinct_roots (c : ℝ) : ℕ := sorry

theorem f_comp_three_roots :
  ∀ c : ℝ, num_distinct_roots c = 3 ↔ c = (11 - Real.sqrt 13) / 2 :=
by sorry

end NUMINAMATH_CALUDE_f_comp_three_roots_l2734_273490


namespace NUMINAMATH_CALUDE_valid_placement_iff_even_l2734_273447

/-- Represents a chessboard with one corner cut off -/
structure Chessboard (n : ℕ) :=
  (size : ℕ := 2*n + 1)
  (corner_cut : Bool := true)

/-- Represents a domino placement on the chessboard -/
structure DominoPlacement (n : ℕ) :=
  (board : Chessboard n)
  (total_dominos : ℕ)
  (horizontal_dominos : ℕ)

/-- Checks if a domino placement is valid -/
def is_valid_placement (n : ℕ) (placement : DominoPlacement n) : Prop :=
  placement.total_dominos * 2 = placement.board.size^2 - 1 ∧
  placement.horizontal_dominos * 2 = placement.total_dominos

/-- The main theorem stating the condition for valid placement -/
theorem valid_placement_iff_even (n : ℕ) :
  (∃ (placement : DominoPlacement n), is_valid_placement n placement) ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_valid_placement_iff_even_l2734_273447


namespace NUMINAMATH_CALUDE_fibonacci_like_sequence_a8_l2734_273428

def fibonacci_like_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 2) = a (n + 1) + a n

theorem fibonacci_like_sequence_a8 (a : ℕ → ℕ) :
  fibonacci_like_sequence a →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) > a n) →
  (∀ n : ℕ, a n > 0) →
  a 7 = 240 →
  a 8 = 386 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_like_sequence_a8_l2734_273428


namespace NUMINAMATH_CALUDE_oil_for_rest_of_bike_l2734_273402

/-- Proves the amount of oil needed for the rest of the bike --/
theorem oil_for_rest_of_bike 
  (oil_per_wheel : ℝ) 
  (num_wheels : ℕ) 
  (total_oil : ℝ) 
  (h1 : oil_per_wheel = 10)
  (h2 : num_wheels = 2)
  (h3 : total_oil = 25) :
  total_oil - (oil_per_wheel * num_wheels) = 5 := by
sorry

end NUMINAMATH_CALUDE_oil_for_rest_of_bike_l2734_273402


namespace NUMINAMATH_CALUDE_smallest_angle_in_special_triangle_l2734_273438

theorem smallest_angle_in_special_triangle : 
  ∀ (a b c : ℝ),
  a + b + c = 180 →
  c = 5 * a →
  b = 3 * a →
  a = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_in_special_triangle_l2734_273438


namespace NUMINAMATH_CALUDE_lcm_18_30_l2734_273421

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_30_l2734_273421


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l2734_273412

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) :
  ∃ (k : ℤ), (n - 1) + n + (n + 1) + (n + 2) = 2 * k := by
  sorry

#check sum_of_four_consecutive_integers_divisible_by_two

end NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l2734_273412


namespace NUMINAMATH_CALUDE_triangle_properties_l2734_273459

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  (sin A) / a = (sin B) / b ∧ (sin B) / b = (sin C) / c ∧
  cos B * sin (B + π/6) = 1/2 ∧
  c / a + a / c = 4 →
  B = π/3 ∧ 1 / tan A + 1 / tan C = 2 * sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2734_273459


namespace NUMINAMATH_CALUDE_room_area_in_sqm_l2734_273408

-- Define the room dimensions
def room_length : Real := 18
def room_width : Real := 9

-- Define the conversion factor
def sqft_to_sqm : Real := 10.7639

-- Theorem statement
theorem room_area_in_sqm :
  let area_sqft := room_length * room_width
  let area_sqm := area_sqft / sqft_to_sqm
  ⌊area_sqm⌋ = 15 := by sorry

end NUMINAMATH_CALUDE_room_area_in_sqm_l2734_273408


namespace NUMINAMATH_CALUDE_clock_rotation_proof_l2734_273401

/-- The number of large divisions on a clock face -/
def clock_divisions : ℕ := 12

/-- The number of degrees in one large division -/
def degrees_per_division : ℝ := 30

/-- The number of hours between 3 o'clock and 6 o'clock -/
def hours_elapsed : ℕ := 3

/-- The degree of rotation of the hour hand from 3 o'clock to 6 o'clock -/
def hour_hand_rotation : ℝ := hours_elapsed * degrees_per_division

theorem clock_rotation_proof :
  hour_hand_rotation = 90 :=
by sorry

end NUMINAMATH_CALUDE_clock_rotation_proof_l2734_273401


namespace NUMINAMATH_CALUDE_circle_ratio_l2734_273400

theorem circle_ratio (a b : ℝ) (h : a > 0) (k : b > 0) 
  (h1 : π * b^2 - π * a^2 = 4 * (π * a^2)) : a / b = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l2734_273400


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2734_273493

theorem fraction_subtraction : 
  (1 + 4 + 7) / (2 + 5 + 8) - (2 + 5 + 8) / (1 + 4 + 7) = -9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2734_273493


namespace NUMINAMATH_CALUDE_work_isothermal_expansion_l2734_273442

/-- Work done during isothermal expansion of an ideal gas -/
theorem work_isothermal_expansion 
  (m μ R T V₁ V₂ : ℝ) 
  (hm : m > 0) 
  (hμ : μ > 0) 
  (hR : R > 0) 
  (hT : T > 0) 
  (hV₁ : V₁ > 0) 
  (hV₂ : V₂ > 0) 
  (hexpand : V₂ > V₁) :
  ∃ A : ℝ, A = (m / μ) * R * T * Real.log (V₂ / V₁) ∧
  (∀ V : ℝ, V > 0 → (m / μ) * R * T = V * (m / μ) * R * T / V) :=
sorry

end NUMINAMATH_CALUDE_work_isothermal_expansion_l2734_273442


namespace NUMINAMATH_CALUDE_right_triangle_set_l2734_273439

theorem right_triangle_set (a b c : ℝ) : 
  (a = 1.5 ∧ b = 2 ∧ c = 2.5) → 
  a^2 + b^2 = c^2 ∧
  ¬(4^2 + 5^2 = 6^2) ∧
  ¬(1^2 + (Real.sqrt 2)^2 = 2.5^2) ∧
  ¬(2^2 + 3^2 = 4^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_set_l2734_273439


namespace NUMINAMATH_CALUDE_max_volume_container_l2734_273477

/-- Represents the dimensions of a rectangular container --/
structure ContainerDimensions where
  length : Real
  width : Real
  height : Real

/-- Calculates the volume of a rectangular container --/
def volume (d : ContainerDimensions) : Real :=
  d.length * d.width * d.height

/-- Represents the constraints of the problem --/
def containerConstraints (d : ContainerDimensions) : Prop :=
  d.length + d.width + d.height = 7.4 ∧  -- Half of the total bar length
  d.length = d.width + 0.5

/-- The main theorem to prove --/
theorem max_volume_container :
  ∃ (d : ContainerDimensions),
    containerConstraints d ∧
    d.height = 1.2 ∧
    volume d = 1.8 ∧
    (∀ (d' : ContainerDimensions), containerConstraints d' → volume d' ≤ volume d) :=
sorry

end NUMINAMATH_CALUDE_max_volume_container_l2734_273477


namespace NUMINAMATH_CALUDE_point_outside_circle_l2734_273473

/-- Given a circle with center O and radius 3, and a point P such that OP = 5,
    prove that P is outside the circle. -/
theorem point_outside_circle (O P : ℝ × ℝ) (r : ℝ) (h1 : r = 3) (h2 : dist O P = 5) :
  dist O P > r := by
  sorry

end NUMINAMATH_CALUDE_point_outside_circle_l2734_273473


namespace NUMINAMATH_CALUDE_sin_540_plus_alpha_implies_cos_alpha_minus_270_l2734_273470

theorem sin_540_plus_alpha_implies_cos_alpha_minus_270
  (α : Real)
  (h : Real.sin (540 * Real.pi / 180 + α) = -4/5) :
  Real.cos (α - 270 * Real.pi / 180) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_540_plus_alpha_implies_cos_alpha_minus_270_l2734_273470


namespace NUMINAMATH_CALUDE_hannah_practice_hours_l2734_273451

/-- Hannah's weekend practice hours -/
def weekend_hours : ℕ := sorry

theorem hannah_practice_hours : 
  (weekend_hours + (weekend_hours + 17) = 33) → 
  weekend_hours = 8 := by sorry

end NUMINAMATH_CALUDE_hannah_practice_hours_l2734_273451


namespace NUMINAMATH_CALUDE_tan_alpha_max_value_l2734_273405

open Real

theorem tan_alpha_max_value (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) 
  (h3 : tan (α + β) = 9 * tan β) : 
  ∃ (max_tan_α : Real), max_tan_α = 4/3 ∧ ∀ (γ : Real), 
    (0 < γ ∧ γ < π/2 ∧ ∃ (δ : Real), (0 < δ ∧ δ < π/2 ∧ tan (γ + δ) = 9 * tan δ)) → 
    tan γ ≤ max_tan_α := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_max_value_l2734_273405


namespace NUMINAMATH_CALUDE_max_value_of_f_l2734_273453

noncomputable def f (x : ℝ) : ℝ := (4 * x - 4 * x^3) / (1 + 2 * x^2 + x^4)

theorem max_value_of_f :
  ∃ (M : ℝ), M = 1 ∧ ∀ (x : ℝ), f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2734_273453


namespace NUMINAMATH_CALUDE_better_to_answer_B_first_l2734_273463

-- Define the probabilities and point values
def prob_correct_A : Real := 0.8
def prob_correct_B : Real := 0.6
def points_A : ℕ := 20
def points_B : ℕ := 80

-- Define the expected score functions
def expected_score_A_first : Real :=
  0 * (1 - prob_correct_A) +
  points_A * (prob_correct_A * (1 - prob_correct_B)) +
  (points_A + points_B) * (prob_correct_A * prob_correct_B)

def expected_score_B_first : Real :=
  0 * (1 - prob_correct_B) +
  points_B * (prob_correct_B * (1 - prob_correct_A)) +
  (points_A + points_B) * (prob_correct_B * prob_correct_A)

-- Theorem statement
theorem better_to_answer_B_first :
  expected_score_B_first > expected_score_A_first := by
  sorry


end NUMINAMATH_CALUDE_better_to_answer_B_first_l2734_273463


namespace NUMINAMATH_CALUDE_three_rulers_left_l2734_273414

/-- The number of rulers left in a drawer after some are removed -/
def rulers_left (initial : ℕ) (removed : ℕ) : ℕ :=
  initial - removed

/-- Theorem stating that 3 rulers are left in the drawer -/
theorem three_rulers_left : rulers_left 14 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_rulers_left_l2734_273414


namespace NUMINAMATH_CALUDE_regular_tetrahedron_has_four_faces_l2734_273410

/-- A regular tetrahedron is a three-dimensional shape with four congruent equilateral triangular faces. -/
structure RegularTetrahedron where
  -- We don't need to define the internal structure for this problem

/-- The number of faces in a regular tetrahedron -/
def num_faces (t : RegularTetrahedron) : ℕ := 4

/-- Theorem: A regular tetrahedron has 4 faces -/
theorem regular_tetrahedron_has_four_faces (t : RegularTetrahedron) : num_faces t = 4 := by
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_has_four_faces_l2734_273410


namespace NUMINAMATH_CALUDE_tank_water_calculation_l2734_273445

theorem tank_water_calculation : 
  let tank1_capacity : ℚ := 7000
  let tank2_capacity : ℚ := 5000
  let tank3_capacity : ℚ := 3000
  let tank1_fill_ratio : ℚ := 3/4
  let tank2_fill_ratio : ℚ := 4/5
  let tank3_fill_ratio : ℚ := 1/2
  let total_water : ℚ := tank1_capacity * tank1_fill_ratio + 
                         tank2_capacity * tank2_fill_ratio + 
                         tank3_capacity * tank3_fill_ratio
  total_water = 10750 := by
sorry

end NUMINAMATH_CALUDE_tank_water_calculation_l2734_273445


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2734_273462

/-- The constant term in the expansion of (x^2 + 1/x^3)^5 -/
def constant_term : ℕ := 10

/-- The binomial coefficient C(5,2) -/
def C_5_2 : ℕ := Nat.choose 5 2

theorem constant_term_expansion :
  constant_term = C_5_2 :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2734_273462


namespace NUMINAMATH_CALUDE_positive_X_value_l2734_273407

-- Define the * operation
def star (X Y : ℝ) : ℝ := X^3 + Y^2

-- Theorem statement
theorem positive_X_value :
  ∃ X : ℝ, X > 0 ∧ star X 4 = 280 ∧ X = 6 :=
sorry

end NUMINAMATH_CALUDE_positive_X_value_l2734_273407


namespace NUMINAMATH_CALUDE_paul_reading_theorem_l2734_273471

/-- Calculates the total number of books read given a weekly reading rate and number of weeks -/
def total_books_read (books_per_week : ℕ) (num_weeks : ℕ) : ℕ :=
  books_per_week * num_weeks

/-- Proves that reading 4 books per week for 5 weeks results in 20 books read -/
theorem paul_reading_theorem : 
  total_books_read 4 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_paul_reading_theorem_l2734_273471


namespace NUMINAMATH_CALUDE_frank_can_collection_l2734_273444

/-- Represents the number of cans in each bag for a given day -/
def BagContents := List Nat

/-- Calculates the total number of cans from a list of bag contents -/
def totalCans (bags : BagContents) : Nat :=
  bags.sum

theorem frank_can_collection :
  let saturday : BagContents := [4, 6, 5, 7, 8]
  let sunday : BagContents := [6, 5, 9]
  let monday : BagContents := [8, 8]
  totalCans saturday + totalCans sunday + totalCans monday = 66 := by
  sorry

end NUMINAMATH_CALUDE_frank_can_collection_l2734_273444


namespace NUMINAMATH_CALUDE_dodecahedron_edge_probability_l2734_273418

/-- A regular dodecahedron -/
structure Dodecahedron where
  vertices : Finset (Fin 20)
  edges : Finset (Fin 20 × Fin 20)
  valid_edge : ∀ e ∈ edges, e.1 ≠ e.2
  vertex_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- The probability of randomly selecting two vertices that are endpoints of an edge in a regular dodecahedron -/
def edge_probability (d : Dodecahedron) : ℚ :=
  d.edges.card / Nat.choose 20 2

/-- Theorem: The probability of randomly selecting two vertices that are endpoints of an edge
    in a regular dodecahedron with 20 vertices is 3/19 -/
theorem dodecahedron_edge_probability (d : Dodecahedron) :
  edge_probability d = 3 / 19 := by
  sorry


end NUMINAMATH_CALUDE_dodecahedron_edge_probability_l2734_273418


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2734_273498

theorem quadratic_inequality (a b c A B C : ℝ) 
  (ha : a ≠ 0) (hA : A ≠ 0)
  (h : ∀ x : ℝ, |a * x^2 + b * x + c| ≤ |A * x^2 + B * x + C|) :
  |b^2 - 4*a*c| ≤ |B^2 - 4*A*C| := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2734_273498


namespace NUMINAMATH_CALUDE_swimming_speed_in_still_water_l2734_273429

/-- Given a person swimming against a current, calculates their swimming speed in still water. -/
theorem swimming_speed_in_still_water 
  (current_speed : ℝ) 
  (distance_against_current : ℝ) 
  (time_against_current : ℝ) 
  (h1 : current_speed = 10)
  (h2 : distance_against_current = 8)
  (h3 : time_against_current = 4) :
  distance_against_current = (swimming_speed - current_speed) * time_against_current →
  swimming_speed = 12 :=
by
  sorry

#check swimming_speed_in_still_water

end NUMINAMATH_CALUDE_swimming_speed_in_still_water_l2734_273429


namespace NUMINAMATH_CALUDE_trapezoid_bases_l2734_273426

/-- Given a trapezoid with midline 6 and difference between bases 4, prove the bases are 4 and 8 -/
theorem trapezoid_bases (a b : ℝ) : 
  (a + b) / 2 = 6 → -- midline is 6
  a - b = 4 →       -- difference between bases is 4
  (a = 8 ∧ b = 4) := by
sorry

end NUMINAMATH_CALUDE_trapezoid_bases_l2734_273426


namespace NUMINAMATH_CALUDE_function_inequality_l2734_273486

theorem function_inequality (f : ℝ → ℝ) (h1 : Differentiable ℝ f) 
  (h2 : ∀ x > 0, deriv f x + f x / x > 0) :
  ∀ a b, a > 0 → b > 0 → a > b → a * f a > b * f b := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2734_273486


namespace NUMINAMATH_CALUDE_books_per_box_l2734_273460

theorem books_per_box (total_books : ℕ) (num_boxes : ℕ) (h1 : total_books = 24) (h2 : num_boxes = 8) :
  total_books / num_boxes = 3 := by
  sorry

end NUMINAMATH_CALUDE_books_per_box_l2734_273460


namespace NUMINAMATH_CALUDE_ap_terms_count_l2734_273484

theorem ap_terms_count (n : ℕ) (a d : ℝ) : 
  n % 2 = 0 ∧ 
  n > 0 ∧
  (n / 2 : ℝ) * (2 * a + (n - 2) * d) = 30 ∧ 
  (n / 2 : ℝ) * (2 * a + n * d) = 36 ∧ 
  a + (n - 1) * d - a = 15 → 
  n = 6 := by sorry

end NUMINAMATH_CALUDE_ap_terms_count_l2734_273484


namespace NUMINAMATH_CALUDE_range_of_a_l2734_273481

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 2| > a^2 + a + 1) ↔ -1 < a ∧ a < 0 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2734_273481


namespace NUMINAMATH_CALUDE_sarahs_laundry_l2734_273441

theorem sarahs_laundry (machine_capacity : ℕ) (sweaters : ℕ) (loads : ℕ) (shirts : ℕ) : 
  machine_capacity = 5 →
  sweaters = 2 →
  loads = 9 →
  shirts = loads * machine_capacity - sweaters →
  shirts = 43 := by
sorry

end NUMINAMATH_CALUDE_sarahs_laundry_l2734_273441


namespace NUMINAMATH_CALUDE_smallest_consecutive_integer_l2734_273465

theorem smallest_consecutive_integer (a b c d : ℕ) : 
  a > 0 ∧ b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ a * b * c * d = 1680 → a = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_consecutive_integer_l2734_273465


namespace NUMINAMATH_CALUDE_smallest_block_volume_l2734_273472

theorem smallest_block_volume (a b c : ℕ) : 
  (a - 1) * (b - 1) * (c - 1) = 240 → 
  a * b * c ≥ 385 ∧ 
  ∃ (a₀ b₀ c₀ : ℕ), (a₀ - 1) * (b₀ - 1) * (c₀ - 1) = 240 ∧ a₀ * b₀ * c₀ = 385 :=
by sorry

end NUMINAMATH_CALUDE_smallest_block_volume_l2734_273472


namespace NUMINAMATH_CALUDE_husband_age_is_54_l2734_273431

/-- Represents a person's age as a two-digit number -/
structure Age :=
  (tens : Nat)
  (ones : Nat)
  (h1 : tens ≤ 9)
  (h2 : ones ≤ 9)

/-- Converts an Age to its numerical value -/
def Age.toNat (a : Age) : Nat :=
  10 * a.tens + a.ones

/-- Reverses the digits of an Age -/
def Age.reverse (a : Age) : Age :=
  ⟨a.ones, a.tens, a.h2, a.h1⟩

theorem husband_age_is_54 (wife : Age) (husband : Age) :
  husband = wife.reverse →
  husband.toNat > wife.toNat →
  husband.toNat - wife.toNat = (husband.toNat + wife.toNat) / 11 →
  husband.toNat = 54 := by
  sorry

end NUMINAMATH_CALUDE_husband_age_is_54_l2734_273431


namespace NUMINAMATH_CALUDE_m_range_theorem_l2734_273440

def f (x : ℝ) : ℝ := x^2 - 2*x

def g (m : ℝ) (x : ℝ) : ℝ := m*x + 2

theorem m_range_theorem (m : ℝ) :
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 2, ∃ x₀ ∈ Set.Icc (-1 : ℝ) 2, g m x₁ = f x₀) →
  m ∈ Set.Icc (-1 : ℝ) (1/2) :=
by sorry

end NUMINAMATH_CALUDE_m_range_theorem_l2734_273440


namespace NUMINAMATH_CALUDE_y_derivative_l2734_273448

noncomputable def y (x : ℝ) : ℝ := 
  Real.sqrt (1 + 2*x - x^2) * Real.arcsin (x * Real.sqrt 2 / (1 + x)) - Real.sqrt 2 * Real.log (1 + x)

theorem y_derivative (x : ℝ) (h : x ≠ -1) : 
  deriv y x = (1 - x) / Real.sqrt (1 + 2*x - x^2) * Real.arcsin (x * Real.sqrt 2 / (1 + x)) := by
  sorry

end NUMINAMATH_CALUDE_y_derivative_l2734_273448


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2021_l2734_273422

theorem reciprocal_of_negative_2021 :
  let reciprocal (x : ℚ) := 1 / x
  reciprocal (-2021) = -1 / 2021 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2021_l2734_273422


namespace NUMINAMATH_CALUDE_min_presses_to_exceed_200_l2734_273413

def repeated_square (x : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => (repeated_square x n) ^ 2

def exceed_200 (x : ℕ) : ℕ :=
  match x with
  | 0 => 0
  | n + 1 => if repeated_square 3 n > 200 then n else exceed_200 n

theorem min_presses_to_exceed_200 : exceed_200 0 = 3 := by sorry

end NUMINAMATH_CALUDE_min_presses_to_exceed_200_l2734_273413


namespace NUMINAMATH_CALUDE_flower_bed_area_ratio_l2734_273415

theorem flower_bed_area_ratio :
  ∀ (l w : ℝ), l > 0 → w > 0 →
  (l * w) / ((2 * l) * (3 * w)) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_area_ratio_l2734_273415


namespace NUMINAMATH_CALUDE_sum_of_even_indexed_coefficients_l2734_273499

theorem sum_of_even_indexed_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, x^10 = a + a₁*(1-x) + a₂*(1-x)^2 + a₃*(1-x)^3 + a₄*(1-x)^4 + 
            a₅*(1-x)^5 + a₆*(1-x)^6 + a₇*(1-x)^7 + a₈*(1-x)^8 + a₉*(1-x)^9 + a₁₀*(1-x)^10) →
  a + a₂ + a₄ + a₆ + a₈ + a₁₀ = 2^9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_even_indexed_coefficients_l2734_273499


namespace NUMINAMATH_CALUDE_costume_material_cost_l2734_273482

/-- Calculates the total cost of material for Jenna's costume --/
theorem costume_material_cost : 
  let skirt_length : ℕ := 12
  let skirt_width : ℕ := 4
  let num_skirts : ℕ := 3
  let bodice_area : ℕ := 2
  let sleeve_area : ℕ := 5
  let num_sleeves : ℕ := 2
  let cost_per_sqft : ℕ := 3
  
  skirt_length * skirt_width * num_skirts + 
  bodice_area + 
  sleeve_area * num_sleeves * cost_per_sqft = 468 := by
  sorry

end NUMINAMATH_CALUDE_costume_material_cost_l2734_273482


namespace NUMINAMATH_CALUDE_sin_2theta_from_exp_l2734_273423

theorem sin_2theta_from_exp (θ : ℝ) (h : Complex.exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 8) / 5) : 
  Real.sin (2 * θ) = 12 * Real.sqrt 2 / 25 := by
sorry

end NUMINAMATH_CALUDE_sin_2theta_from_exp_l2734_273423


namespace NUMINAMATH_CALUDE_bread_rising_times_l2734_273475

/-- Represents the bread-making process with given time constraints --/
def BreadMaking (total_time rising_time kneading_time baking_time : ℕ) :=
  {n : ℕ // n * rising_time + kneading_time + baking_time = total_time}

/-- Theorem stating that Mark lets the bread rise twice --/
theorem bread_rising_times :
  BreadMaking 280 120 10 30 = {n : ℕ // n = 2} :=
by sorry

end NUMINAMATH_CALUDE_bread_rising_times_l2734_273475


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l2734_273497

/-- The rectangular coordinate equation of a curve given its polar equation -/
theorem polar_to_rectangular (ρ θ : ℝ) (h : ρ * Real.cos θ = 2) : 
  ∃ x : ℝ, x = 2 := by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_l2734_273497


namespace NUMINAMATH_CALUDE_cosine_inequality_solution_l2734_273474

theorem cosine_inequality_solution (y : Real) : 
  (y ∈ Set.Icc 0 Real.pi) ∧ 
  (∀ x ∈ Set.Icc 0 Real.pi, Real.cos (x + y) ≥ Real.cos x + Real.cos y) → 
  y = 0 := by
sorry

end NUMINAMATH_CALUDE_cosine_inequality_solution_l2734_273474


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_perimeter_bound_l2734_273432

/-- A rectangle in 2D space -/
structure Rectangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ
  d : ℝ × ℝ
  is_rectangle : sorry

/-- A quadrilateral inscribed in a rectangle -/
structure InscribedQuadrilateral (rect : Rectangle) where
  k : ℝ × ℝ
  l : ℝ × ℝ
  m : ℝ × ℝ
  n : ℝ × ℝ
  on_sides : sorry

/-- Calculate the perimeter of a quadrilateral -/
def perimeter (q : InscribedQuadrilateral rect) : ℝ := sorry

/-- Calculate the length of the diagonal of a rectangle -/
def diagonal_length (rect : Rectangle) : ℝ := sorry

/-- Theorem: The perimeter of an inscribed quadrilateral is at least twice the diagonal of the rectangle -/
theorem inscribed_quadrilateral_perimeter_bound (rect : Rectangle) (q : InscribedQuadrilateral rect) :
  perimeter q ≥ 2 * diagonal_length rect := by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_perimeter_bound_l2734_273432


namespace NUMINAMATH_CALUDE_employee_salaries_calculation_l2734_273485

/-- Given a total revenue and a ratio for division between employee salaries and stock purchases,
    calculate the amount spent on employee salaries. -/
def calculate_employee_salaries (total_revenue : ℚ) (salary_ratio stock_ratio : ℕ) : ℚ :=
  (salary_ratio : ℚ) / ((salary_ratio : ℚ) + (stock_ratio : ℚ)) * total_revenue

/-- Theorem stating that given a total revenue of 3000 and a division ratio of 4:11
    for employee salaries to stock purchases, the amount spent on employee salaries is 800. -/
theorem employee_salaries_calculation :
  calculate_employee_salaries 3000 4 11 = 800 := by
  sorry

#eval calculate_employee_salaries 3000 4 11

end NUMINAMATH_CALUDE_employee_salaries_calculation_l2734_273485


namespace NUMINAMATH_CALUDE_triangle_inequality_satisfied_l2734_273425

theorem triangle_inequality_satisfied (a b c : ℝ) (ha : a = 25) (hb : b = 24) (hc : c = 7) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_satisfied_l2734_273425


namespace NUMINAMATH_CALUDE_smallest_n_dividing_m_pow_n_minus_one_l2734_273469

theorem smallest_n_dividing_m_pow_n_minus_one (m : ℕ) (h_m_odd : Odd m) (h_m_gt_1 : m > 1) :
  (∀ n : ℕ, n > 0 → (2^1989 ∣ m^n - 1)) ↔ n ≥ 2^1987 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_dividing_m_pow_n_minus_one_l2734_273469


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_zero_range_of_a_when_p_implies_q_l2734_273411

-- Define the conditions
def p (x : ℝ) : Prop := 4 * x^2 + 12 * x - 7 ≤ 0
def q (a x : ℝ) : Prop := a - 3 ≤ x ∧ x ≤ a + 3

-- Theorem for the first question
theorem range_of_x_when_a_is_zero :
  ∀ x : ℝ, (p x ∧ ¬(q 0 x)) → (-7/2 ≤ x ∧ x < -3) :=
sorry

-- Theorem for the second question
theorem range_of_a_when_p_implies_q :
  (∀ x : ℝ, p x → q a x) → (-5/2 ≤ a ∧ a ≤ -1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_zero_range_of_a_when_p_implies_q_l2734_273411


namespace NUMINAMATH_CALUDE_no_winning_strategy_strategy_independent_no_strategy_better_than_half_l2734_273404

/-- Represents a deck of cards with red and black suits -/
structure Deck :=
  (red : ℕ)
  (black : ℕ)

/-- A strategy is a function that decides whether to stop based on the current deck state -/
def Strategy := Deck → Bool

/-- The probability of winning given a deck state -/
def winProbability (d : Deck) : ℚ :=
  d.red / (d.red + d.black)

/-- Theorem stating that no strategy can achieve a winning probability greater than 0.5 -/
theorem no_winning_strategy (d : Deck) (s : Strategy) :
  d.red = d.black → winProbability d ≤ 1/2 := by
  sorry

/-- Theorem stating that the winning probability is independent of the strategy -/
theorem strategy_independent (d : Deck) (s₁ s₂ : Strategy) :
  winProbability d = winProbability d := by
  sorry

/-- Main theorem: No strategy exists that guarantees a winning probability greater than 0.5 -/
theorem no_strategy_better_than_half (d : Deck) :
  d.red = d.black → ∀ s : Strategy, winProbability d ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_no_winning_strategy_strategy_independent_no_strategy_better_than_half_l2734_273404


namespace NUMINAMATH_CALUDE_doctor_selection_ways_l2734_273491

/-- The number of ways to choose a team of doctors from internists and surgeons --/
def choose_doctors (internists surgeons team_size : ℕ) : ℕ :=
  Nat.choose (internists + surgeons) team_size -
  (Nat.choose internists team_size + Nat.choose surgeons team_size)

/-- Theorem stating the number of ways to choose 4 doctors from 5 internists and 6 surgeons --/
theorem doctor_selection_ways :
  choose_doctors 5 6 4 = 310 := by
  sorry

end NUMINAMATH_CALUDE_doctor_selection_ways_l2734_273491


namespace NUMINAMATH_CALUDE_comm_add_comm_mul_distrib_l2734_273476

-- Commutative law of addition
theorem comm_add (a b : ℝ) : a + b = b + a := by sorry

-- Commutative law of multiplication
theorem comm_mul (a b : ℝ) : a * b = b * a := by sorry

-- Distributive law of multiplication over addition
theorem distrib (a b c : ℝ) : (a + b) * c = a * c + b * c := by sorry

end NUMINAMATH_CALUDE_comm_add_comm_mul_distrib_l2734_273476


namespace NUMINAMATH_CALUDE_triangle_pqr_properties_l2734_273434

/-- Triangle PQR with vertices P(-2,3), Q(4,5), and R(1,-4), and point S(p,q) inside the triangle such that triangles PQS, QRS, and RPS have equal areas -/
structure TrianglePQR where
  P : ℝ × ℝ := (-2, 3)
  Q : ℝ × ℝ := (4, 5)
  R : ℝ × ℝ := (1, -4)
  S : ℝ × ℝ
  equal_areas : True  -- Placeholder for the equal areas condition

/-- The coordinates of point S -/
def point_S (t : TrianglePQR) : ℝ × ℝ := t.S

/-- The perimeter of triangle PQR -/
noncomputable def perimeter (t : TrianglePQR) : ℝ :=
  Real.sqrt 40 + Real.sqrt 90 + Real.sqrt 58

/-- Main theorem about the triangle PQR and point S -/
theorem triangle_pqr_properties (t : TrianglePQR) :
  point_S t = (1, 4/3) ∧
  10 * (point_S t).1 + (point_S t).2 = 34/3 ∧
  perimeter t = Real.sqrt 40 + Real.sqrt 90 + Real.sqrt 58 ∧
  34/3 < perimeter t :=
by sorry

end NUMINAMATH_CALUDE_triangle_pqr_properties_l2734_273434


namespace NUMINAMATH_CALUDE_clock_correction_time_l2734_273468

/-- The number of minutes in 12 hours -/
def minutes_in_12_hours : ℕ := 12 * 60

/-- The number of minutes the clock gains per day -/
def minutes_gained_per_day : ℕ := 3

/-- The minimum number of days for the clock to show the correct time again -/
def min_days_to_correct_time : ℕ := minutes_in_12_hours / minutes_gained_per_day

theorem clock_correction_time :
  min_days_to_correct_time = 240 :=
sorry

end NUMINAMATH_CALUDE_clock_correction_time_l2734_273468


namespace NUMINAMATH_CALUDE_problem_solution_l2734_273483

def f (a x : ℝ) : ℝ := |x - 1| + |x + a^2|

theorem problem_solution :
  (∀ x : ℝ, f (Real.sqrt 2) x ≥ 6 ↔ x ≤ -7/2 ∨ x ≥ 5/2) ∧
  (∃ x₀ : ℝ, f a x₀ < 4*a ↔ 2 - Real.sqrt 3 < a ∧ a < 2 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2734_273483


namespace NUMINAMATH_CALUDE_g_18_value_l2734_273466

-- Define the properties of g
def is_valid_g (g : ℕ+ → ℕ+) : Prop :=
  (∀ n : ℕ+, g (n + 1) > g n) ∧ 
  (∀ m n : ℕ+, g (m * n) = g m * g n) ∧
  (∀ m n : ℕ+, m ≠ n → m ^ (n : ℕ) = n ^ (m : ℕ) → g m = n ^ 2 ∨ g n = m ^ 2)

-- State the theorem
theorem g_18_value (g : ℕ+ → ℕ+) (h : is_valid_g g) : g 18 = 104976 := by
  sorry

end NUMINAMATH_CALUDE_g_18_value_l2734_273466


namespace NUMINAMATH_CALUDE_number_of_non_officers_l2734_273450

/-- Proves that the number of non-officers is 525 given the salary conditions --/
theorem number_of_non_officers (avg_salary : ℝ) (officer_salary : ℝ) (non_officer_salary : ℝ) 
  (num_officers : ℕ) (h1 : avg_salary = 120) (h2 : officer_salary = 470) 
  (h3 : non_officer_salary = 110) (h4 : num_officers = 15) : 
  ∃ (num_non_officers : ℕ), 
    (↑num_officers * officer_salary + ↑num_non_officers * non_officer_salary) / 
    (↑num_officers + ↑num_non_officers) = avg_salary ∧ num_non_officers = 525 := by
  sorry

#check number_of_non_officers

end NUMINAMATH_CALUDE_number_of_non_officers_l2734_273450


namespace NUMINAMATH_CALUDE_actual_average_height_l2734_273461

/-- Represents the average height calculation problem in a class --/
structure HeightProblem where
  totalStudents : ℕ
  initialAverage : ℚ
  incorrectHeights : List ℚ
  actualHeights : List ℚ

/-- Calculates the actual average height given the problem data --/
def calculateActualAverage (problem : HeightProblem) : ℚ :=
  let initialTotal := problem.initialAverage * problem.totalStudents
  let heightDifference := (problem.incorrectHeights.sum - problem.actualHeights.sum)
  let correctedTotal := initialTotal - heightDifference
  correctedTotal / problem.totalStudents

/-- The theorem stating that the actual average height is 164.5 cm --/
theorem actual_average_height
  (problem : HeightProblem)
  (h1 : problem.totalStudents = 50)
  (h2 : problem.initialAverage = 165)
  (h3 : problem.incorrectHeights = [150, 175, 190])
  (h4 : problem.actualHeights = [135, 170, 185]) :
  calculateActualAverage problem = 164.5 := by
  sorry


end NUMINAMATH_CALUDE_actual_average_height_l2734_273461


namespace NUMINAMATH_CALUDE_repel_creatures_l2734_273496

/-- Represents the number of cloves needed to repel creatures -/
def cloves_needed (vampires wights vampire_bats : ℕ) : ℕ :=
  let vampires_cloves := (3 * vampires + 1) / 2
  let wights_cloves := wights
  let bats_cloves := (3 * vampire_bats + 7) / 8
  vampires_cloves + wights_cloves + bats_cloves

/-- Theorem stating the number of cloves needed to repel specific numbers of creatures -/
theorem repel_creatures : cloves_needed 30 12 40 = 72 := by
  sorry

end NUMINAMATH_CALUDE_repel_creatures_l2734_273496


namespace NUMINAMATH_CALUDE_optimal_launch_angle_l2734_273403

/-- 
Given a target at horizontal distance A and height B, 
the angle α that minimizes the initial speed of a projectile to hit the target 
is given by α = arctan((B + √(A² + B²))/A).
-/
theorem optimal_launch_angle (A B : ℝ) (hA : A > 0) (hB : B ≥ 0) :
  let C := Real.sqrt (A^2 + B^2)
  let α := Real.arctan ((B + C) / A)
  ∀ θ : ℝ, 
    0 < θ ∧ θ < π / 2 → 
    (Real.sin θ)^2 * (A^2 + B^2) ≤ (Real.sin (2*α)) * (A^2 + B^2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_optimal_launch_angle_l2734_273403


namespace NUMINAMATH_CALUDE_intersection_point_of_g_and_inverse_l2734_273449

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 2*x^2 + 18*x + 36

-- State the theorem
theorem intersection_point_of_g_and_inverse :
  ∃! p : ℝ × ℝ, 
    (∀ x : ℝ, (x, g x) ≠ p → (g x, x) ≠ p) ∧ 
    p.1 = g p.2 ∧ 
    p.2 = g p.1 ∧
    p = (-3, -3) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_of_g_and_inverse_l2734_273449


namespace NUMINAMATH_CALUDE_team_selection_with_girls_l2734_273406

theorem team_selection_with_girls (boys girls team_size min_girls : ℕ) 
  (h_boys : boys = 10)
  (h_girls : girls = 12)
  (h_team_size : team_size = 6)
  (h_min_girls : min_girls = 2) : 
  (Finset.range (team_size - min_girls + 1)).sum (λ i => 
    Nat.choose girls (i + min_girls) * Nat.choose boys (team_size - (i + min_girls))) = 71379 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_with_girls_l2734_273406


namespace NUMINAMATH_CALUDE_triangle_area_problem_l2734_273452

theorem triangle_area_problem (x : ℝ) (h1 : x > 0) 
  (h2 : (1/2) * x * (3*x) = 96) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_problem_l2734_273452


namespace NUMINAMATH_CALUDE_cleaning_times_l2734_273419

/-- Proves the cleaning times for Bob and Carol given Alice's cleaning time -/
theorem cleaning_times (alice_time : ℕ) (bob_time carol_time : ℕ) : 
  alice_time = 40 →
  bob_time = alice_time / 4 →
  carol_time = 2 * bob_time →
  (bob_time = 10 ∧ carol_time = 20) := by
  sorry

end NUMINAMATH_CALUDE_cleaning_times_l2734_273419


namespace NUMINAMATH_CALUDE_letters_in_small_envelopes_l2734_273416

/-- Given the total number of letters, the number of large envelopes, and the number of letters
    per large envelope, calculate the number of letters in small envelopes. -/
theorem letters_in_small_envelopes 
  (total_letters : ℕ) 
  (large_envelopes : ℕ) 
  (letters_per_large_envelope : ℕ) 
  (h1 : total_letters = 80)
  (h2 : large_envelopes = 30)
  (h3 : letters_per_large_envelope = 2) : 
  total_letters - large_envelopes * letters_per_large_envelope = 20 :=
by sorry

end NUMINAMATH_CALUDE_letters_in_small_envelopes_l2734_273416


namespace NUMINAMATH_CALUDE_expression_evaluation_l2734_273436

theorem expression_evaluation :
  let a : ℤ := -4
  (4 * a^2 - 3*a) - (2 * a^2 + a - 1) + (2 - a^2 + 4*a) = 19 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2734_273436


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l2734_273467

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (|x₁ + 3| = 15) ∧ 
  (|x₂ + 3| = 15) ∧ 
  (x₁ ≠ x₂) ∧
  (x₁ - x₂ = 30 ∨ x₂ - x₁ = 30) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l2734_273467


namespace NUMINAMATH_CALUDE_no_real_roots_l2734_273446

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (3 * x + 9) + 8 / Real.sqrt (3 * x + 9) = 4 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l2734_273446


namespace NUMINAMATH_CALUDE_sum_of_bases_equals_999_l2734_273437

/-- Converts a number from base 11 to base 10 -/
def base11To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 12 to base 10 -/
def base12To10 (n : ℕ) : ℕ := sorry

/-- Represents the digit A in base 12 -/
def A : ℕ := 10

theorem sum_of_bases_equals_999 :
  base11To10 379 + base12To10 (3 * 12^2 + A * 12 + 9) = 999 := by sorry

end NUMINAMATH_CALUDE_sum_of_bases_equals_999_l2734_273437


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2734_273478

/-- An isosceles triangle with side lengths 5 and 6 has a perimeter of either 16 or 17 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 5 ∧ b = 6 ∧
  ((a = b ∧ c ≤ a + b) ∨ (a = c ∧ b ≤ a + c) ∨ (b = c ∧ a ≤ b + c)) →
  a + b + c = 16 ∨ a + b + c = 17 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2734_273478


namespace NUMINAMATH_CALUDE_printer_task_time_l2734_273494

/-- Given two printers A and B, this theorem proves the time taken to complete a task together -/
theorem printer_task_time (pages : ℕ) (time_A : ℕ) (rate_diff : ℕ) : 
  pages = 480 → 
  time_A = 60 → 
  rate_diff = 4 → 
  (pages : ℚ) / ((pages : ℚ) / time_A + ((pages : ℚ) / time_A + rate_diff)) = 24 := by
  sorry

#check printer_task_time

end NUMINAMATH_CALUDE_printer_task_time_l2734_273494


namespace NUMINAMATH_CALUDE_fraction_identity_l2734_273409

theorem fraction_identity (a b c : ℝ) 
  (h1 : a + b + c ≠ 0) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0)
  (h5 : (a + b + c)⁻¹ = a⁻¹ + b⁻¹ + c⁻¹) :
  (a^5 + b^5 + c^5)⁻¹ = a⁻¹^5 + b⁻¹^5 + c⁻¹^5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_identity_l2734_273409


namespace NUMINAMATH_CALUDE_ellipse_slope_product_l2734_273420

/-- The ellipse C with semi-major axis a and semi-minor axis b -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The line tangent to the circle -/
def tangent_line (x y : ℝ) : Prop :=
  Real.sqrt 7 * x - Real.sqrt 5 * y + 12 = 0

/-- The point A -/
def A : ℝ × ℝ := (-4, 0)

/-- The point R -/
def R : ℝ × ℝ := (3, 0)

/-- The vertical line that M and N lie on -/
def vertical_line (x : ℝ) : Prop :=
  x = 16/3

theorem ellipse_slope_product (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : (a^2 - b^2) / a^2 = 1/4) 
  (h4 : ∃ (x y : ℝ), ellipse b b x y ∧ tangent_line x y) :
  ∃ (P Q M N : ℝ × ℝ) (k1 k2 : ℝ),
    ellipse a b P.1 P.2 ∧
    ellipse a b Q.1 Q.2 ∧
    vertical_line M.1 ∧
    vertical_line N.1 ∧
    k1 * k2 = -12/7 :=
  sorry

end NUMINAMATH_CALUDE_ellipse_slope_product_l2734_273420


namespace NUMINAMATH_CALUDE_afternoon_campers_calculation_l2734_273464

-- Define the number of campers who went rowing in the morning
def morning_campers : ℝ := 15.5

-- Define the total number of campers who went rowing that day
def total_campers : ℝ := 32.75

-- Define the number of campers who went rowing in the afternoon
def afternoon_campers : ℝ := total_campers - morning_campers

-- Theorem to prove
theorem afternoon_campers_calculation :
  afternoon_campers = 17.25 := by sorry

end NUMINAMATH_CALUDE_afternoon_campers_calculation_l2734_273464


namespace NUMINAMATH_CALUDE_solve_for_y_l2734_273488

theorem solve_for_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 16) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2734_273488


namespace NUMINAMATH_CALUDE_cube_root_ratio_l2734_273443

theorem cube_root_ratio (r_old r_new : ℝ) (a_old a_new : ℝ) : 
  a_old = (2 * r_old)^3 → 
  a_new = (2 * r_new)^3 → 
  a_new = 0.125 * a_old → 
  r_new / r_old = 1/2 := by
sorry

end NUMINAMATH_CALUDE_cube_root_ratio_l2734_273443


namespace NUMINAMATH_CALUDE_train_crossing_time_l2734_273495

/-- Proves that a train of given length, passing a platform of given length in a given time,
    will take a specific time to cross a tree. -/
theorem train_crossing_time
  (train_length : ℝ)
  (platform_length : ℝ)
  (platform_crossing_time : ℝ)
  (h1 : train_length = 1200)
  (h2 : platform_length = 700)
  (h3 : platform_crossing_time = 190)
  : (train_length / ((train_length + platform_length) / platform_crossing_time)) = 120 :=
by sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l2734_273495


namespace NUMINAMATH_CALUDE_simplify_expression_l2734_273455

theorem simplify_expression : 2^2 + 2^2 + 2^2 + 2^2 = 2^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2734_273455


namespace NUMINAMATH_CALUDE_suitcase_profit_l2734_273457

/-- Calculates the total profit and profit per suitcase for a store selling suitcases. -/
theorem suitcase_profit (num_suitcases : ℕ) (purchase_price : ℕ) (total_revenue : ℕ) :
  num_suitcases = 60 →
  purchase_price = 100 →
  total_revenue = 8100 →
  (total_revenue - num_suitcases * purchase_price = 2100) ∧
  ((total_revenue - num_suitcases * purchase_price) / num_suitcases = 35) := by
  sorry

#check suitcase_profit

end NUMINAMATH_CALUDE_suitcase_profit_l2734_273457


namespace NUMINAMATH_CALUDE_fourth_term_max_implies_n_six_l2734_273456

theorem fourth_term_max_implies_n_six (n : ℕ) : 
  (∀ k : ℕ, k ≠ 3 → (n.choose k) ≤ (n.choose 3)) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_max_implies_n_six_l2734_273456


namespace NUMINAMATH_CALUDE_factor_3x_squared_minus_75_l2734_273458

theorem factor_3x_squared_minus_75 (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_3x_squared_minus_75_l2734_273458
