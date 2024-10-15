import Mathlib

namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l324_32427

def f (x : ℝ) : ℝ := 3 * x^4 + 4 * x^3 + 34

theorem max_min_f_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-2 : ℝ) 1, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = max) ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 1, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = min) ∧
    max = 50 ∧ min = 33 :=
by sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l324_32427


namespace NUMINAMATH_CALUDE_negative_integer_sum_and_square_is_fifteen_l324_32495

theorem negative_integer_sum_and_square_is_fifteen (N : ℤ) : 
  N < 0 → N^2 + N = 15 → N = -5 := by sorry

end NUMINAMATH_CALUDE_negative_integer_sum_and_square_is_fifteen_l324_32495


namespace NUMINAMATH_CALUDE_machine_sale_price_l324_32449

def selling_price (purchase_price repair_cost transport_cost profit_percentage : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost + transport_cost
  let profit := total_cost * profit_percentage / 100
  total_cost + profit

theorem machine_sale_price :
  selling_price 11000 5000 1000 50 = 25500 := by
  sorry

end NUMINAMATH_CALUDE_machine_sale_price_l324_32449


namespace NUMINAMATH_CALUDE_factorization_equality_l324_32475

theorem factorization_equality (a b : ℝ) : a^2 * b - 9 * b = b * (a + 3) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l324_32475


namespace NUMINAMATH_CALUDE_existence_of_a_for_minimum_value_l324_32488

theorem existence_of_a_for_minimum_value (e : Real) (h_e : e > 0) : ∃ a : Real,
  (∀ x : Real, 0 < x ∧ x ≤ e → ax - Real.log x ≥ 3) ∧
  (∃ x : Real, 0 < x ∧ x ≤ e ∧ ax - Real.log x = 3) ∧
  a = Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_a_for_minimum_value_l324_32488


namespace NUMINAMATH_CALUDE_equation_solution_l324_32485

theorem equation_solution :
  ∃! x : ℚ, x ≠ -5 ∧ (x^2 + 3*x + 4) / (x + 5) = x + 7 :=
by
  use (-31 / 9)
  sorry

end NUMINAMATH_CALUDE_equation_solution_l324_32485


namespace NUMINAMATH_CALUDE_original_number_exists_l324_32460

theorem original_number_exists : ∃ x : ℝ, 3 * (2 * x + 5) = 123 := by
  sorry

end NUMINAMATH_CALUDE_original_number_exists_l324_32460


namespace NUMINAMATH_CALUDE_nickys_card_value_l324_32408

/-- Proves that if Nicky trades two cards of equal value for one card worth $21 
    and makes a profit of $5, then each of Nicky's cards is worth $8. -/
theorem nickys_card_value (card_value : ℝ) : 
  (2 * card_value + 5 = 21) → card_value = 8 := by
  sorry

end NUMINAMATH_CALUDE_nickys_card_value_l324_32408


namespace NUMINAMATH_CALUDE_rectangle_ratio_l324_32452

theorem rectangle_ratio (width : ℕ) (area : ℕ) (length : ℕ) : 
  width = 7 → 
  area = 196 → 
  length * width = area → 
  ∃ k : ℕ, length = k * width → 
  (length : ℚ) / width = 4 := by
sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l324_32452


namespace NUMINAMATH_CALUDE_tom_age_l324_32432

theorem tom_age (carla_age : ℕ) (tom_age : ℕ) (dave_age : ℕ) : 
  (tom_age = 2 * carla_age - 1) →
  (dave_age = carla_age + 3) →
  (carla_age + tom_age + dave_age = 30) →
  tom_age = 13 := by
sorry

end NUMINAMATH_CALUDE_tom_age_l324_32432


namespace NUMINAMATH_CALUDE_exponential_function_property_l324_32445

theorem exponential_function_property (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (∀ x ∈ Set.Icc 0 2, a^x ≤ 1) ∧
  (∀ x ∈ Set.Icc 0 2, a^x ≥ a^2) ∧
  (1 - a^2 = 3/4) →
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_property_l324_32445


namespace NUMINAMATH_CALUDE_triangle_inradius_l324_32425

/-- Given a triangle with perimeter 24 cm and area 30 cm², prove that its inradius is 2.5 cm. -/
theorem triangle_inradius (perimeter : ℝ) (area : ℝ) (inradius : ℝ) : 
  perimeter = 24 → area = 30 → inradius * (perimeter / 2) = area → inradius = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l324_32425


namespace NUMINAMATH_CALUDE_function_equality_implies_sum_l324_32413

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- Define the constants a, b, and c
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry

-- State the theorem
theorem function_equality_implies_sum (x : ℝ) :
  (∀ x, f (x + 4) = 2 * x^2 + 8 * x + 10) ∧
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 4 := by sorry

end NUMINAMATH_CALUDE_function_equality_implies_sum_l324_32413


namespace NUMINAMATH_CALUDE_certain_number_subtraction_l324_32400

theorem certain_number_subtraction (x : ℤ) : x + 468 = 954 → x - 3 = 483 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_subtraction_l324_32400


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_is_one_fifth_l324_32440

/-- Represents a shape created by joining nine unit cubes -/
structure CubeShape where
  /-- The total number of unit cubes in the shape -/
  total_cubes : ℕ
  /-- The number of exposed faces of the shape -/
  exposed_faces : ℕ
  /-- Assertion that the total number of cubes is 9 -/
  cube_count : total_cubes = 9
  /-- Assertion that the number of exposed faces is 45 -/
  face_count : exposed_faces = 45

/-- Calculates the ratio of volume to surface area for the cube shape -/
def volumeToSurfaceAreaRatio (shape : CubeShape) : ℚ :=
  shape.total_cubes / shape.exposed_faces

/-- Theorem stating that the ratio of volume to surface area is 1/5 -/
theorem volume_to_surface_area_ratio_is_one_fifth (shape : CubeShape) :
  volumeToSurfaceAreaRatio shape = 1 / 5 := by
  sorry

#check volume_to_surface_area_ratio_is_one_fifth

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_is_one_fifth_l324_32440


namespace NUMINAMATH_CALUDE_range_of_a_l324_32477

theorem range_of_a (x a : ℝ) : 
  (∀ x, -x^2 + 5*x - 6 > 0 → |x - a| < 4) ∧ 
  (∃ x, |x - a| < 4 ∧ -x^2 + 5*x - 6 ≤ 0) →
  a ∈ Set.Icc (-1 : ℝ) 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l324_32477


namespace NUMINAMATH_CALUDE_cubic_coefficient_in_product_l324_32430

/-- The coefficient of x^3 in the expansion of (3x^3 + 2x^2 + 4x + 5)(4x^3 + 3x^2 + 5x + 6) -/
def cubic_coefficient : ℤ := 40

/-- The first polynomial in the product -/
def polynomial1 (x : ℚ) : ℚ := 3 * x^3 + 2 * x^2 + 4 * x + 5

/-- The second polynomial in the product -/
def polynomial2 (x : ℚ) : ℚ := 4 * x^3 + 3 * x^2 + 5 * x + 6

/-- The theorem stating that the coefficient of x^3 in the expansion of the product of polynomial1 and polynomial2 is equal to cubic_coefficient -/
theorem cubic_coefficient_in_product : 
  ∃ (a b c d e f g : ℚ), 
    polynomial1 x * polynomial2 x = a * x^6 + b * x^5 + c * x^4 + cubic_coefficient * x^3 + d * x^2 + e * x + f :=
by sorry

end NUMINAMATH_CALUDE_cubic_coefficient_in_product_l324_32430


namespace NUMINAMATH_CALUDE_estimate_smaller_than_exact_l324_32473

theorem estimate_smaller_than_exact (a b c d a' b' c' d' : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (ha' : 0 < a' ∧ a' ≤ a) (hb' : 0 < b' ∧ b ≤ b')
  (hc' : 0 < c' ∧ c' ≤ c) (hd' : 0 < d' ∧ d ≤ d') :
  d' * (a' / b') + c' < d * (a / b) + c := by
  sorry

end NUMINAMATH_CALUDE_estimate_smaller_than_exact_l324_32473


namespace NUMINAMATH_CALUDE_triangle_side_length_l324_32461

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  a = 2 → b = 1 → C = Real.pi / 3 →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  c = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l324_32461


namespace NUMINAMATH_CALUDE_marble_selection_theorem_l324_32454

/-- The number of marbles John has in total -/
def total_marbles : ℕ := 15

/-- The number of colors with exactly two marbles each -/
def special_colors : ℕ := 3

/-- The number of marbles for each special color -/
def marbles_per_special_color : ℕ := 2

/-- The number of marbles to be chosen -/
def marbles_to_choose : ℕ := 5

/-- The number of special colored marbles to be chosen -/
def special_marbles_to_choose : ℕ := 2

/-- The number of ways to choose the marbles under the given conditions -/
def ways_to_choose : ℕ := 1008

theorem marble_selection_theorem :
  (Nat.choose special_colors special_marbles_to_choose) *
  (Nat.choose marbles_per_special_color 1) ^ special_marbles_to_choose *
  (Nat.choose (total_marbles - special_colors * marbles_per_special_color) (marbles_to_choose - special_marbles_to_choose)) =
  ways_to_choose := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_theorem_l324_32454


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l324_32436

/-- The value of m^2 for which the line y = mx + 2 is tangent to the ellipse x^2 + 9y^2 = 9 -/
theorem line_tangent_to_ellipse (m : ℝ) : 
  (∀ x y : ℝ, y = m * x + 2 ∧ x^2 + 9 * y^2 = 9 → 
    ∃! p : ℝ × ℝ, p.1^2 + 9 * p.2^2 = 9 ∧ p.2 = m * p.1 + 2) ↔ 
  m^2 = 1/3 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l324_32436


namespace NUMINAMATH_CALUDE_charlies_share_l324_32422

/-- Represents the share of money each person receives -/
structure Share where
  alice : ℚ
  bond : ℚ
  charlie : ℚ

/-- The conditions of the problem -/
def satisfiesConditions (s : Share) : Prop :=
  s.alice + s.bond + s.charlie = 1105 ∧
  (s.alice - 10) / (s.bond - 20) = 11 / 18 ∧
  (s.alice - 10) / (s.charlie - 15) = 11 / 24

/-- The theorem stating Charlie's share -/
theorem charlies_share :
  ∃ (s : Share), satisfiesConditions s ∧ s.charlie = 495 := by
  sorry


end NUMINAMATH_CALUDE_charlies_share_l324_32422


namespace NUMINAMATH_CALUDE_sum_57_68_rounded_l324_32463

/-- Rounds a number to the nearest ten -/
def roundToNearestTen (x : ℤ) : ℤ :=
  10 * ((x + 5) / 10)

/-- The sum of 57 and 68, when rounded to the nearest ten, equals 130 -/
theorem sum_57_68_rounded : roundToNearestTen (57 + 68) = 130 := by
  sorry

end NUMINAMATH_CALUDE_sum_57_68_rounded_l324_32463


namespace NUMINAMATH_CALUDE_losing_teams_total_score_l324_32426

/-- Represents a basketball game between two teams -/
structure Game where
  team1_score : ℕ
  team2_score : ℕ

/-- The total score of a game -/
def Game.total_score (g : Game) : ℕ := g.team1_score + g.team2_score

/-- The margin of victory in a game -/
def Game.margin (g : Game) : ℤ := g.team1_score - g.team2_score

theorem losing_teams_total_score (game1 game2 : Game) 
  (h1 : game1.total_score = 150)
  (h2 : game1.margin = 10)
  (h3 : game2.total_score = 140)
  (h4 : game2.margin = -20) :
  game1.team2_score + game2.team1_score = 130 := by
sorry

end NUMINAMATH_CALUDE_losing_teams_total_score_l324_32426


namespace NUMINAMATH_CALUDE_chord_length_specific_case_l324_32492

/-- The length of the chord formed by the intersection of a line and a circle -/
def chord_length (line_point : ℝ × ℝ) (line_angle : ℝ) (circle_center : ℝ × ℝ) (circle_radius : ℝ) : ℝ :=
  sorry

theorem chord_length_specific_case :
  let line_point : ℝ × ℝ := (1, 0)
  let line_angle : ℝ := 30 * π / 180  -- 30 degrees in radians
  let circle_center : ℝ × ℝ := (2, 0)
  let circle_radius : ℝ := 1
  chord_length line_point line_angle circle_center circle_radius = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_specific_case_l324_32492


namespace NUMINAMATH_CALUDE_mans_speed_in_still_water_l324_32446

/-- 
Given a man's upstream and downstream rowing speeds, 
calculate his speed in still water.
-/
theorem mans_speed_in_still_water 
  (upstream_speed downstream_speed : ℝ) 
  (h1 : upstream_speed = 20) 
  (h2 : downstream_speed = 60) : 
  (upstream_speed + downstream_speed) / 2 = 40 := by
  sorry

#check mans_speed_in_still_water

end NUMINAMATH_CALUDE_mans_speed_in_still_water_l324_32446


namespace NUMINAMATH_CALUDE_expected_sum_of_rook_positions_l324_32406

/-- Represents a chessboard with 64 fields -/
def ChessboardSize : ℕ := 64

/-- Number of rooks placed on the board -/
def NumRooks : ℕ := 6

/-- Expected value of a single randomly chosen position -/
def ExpectedSinglePosition : ℚ := (ChessboardSize + 1) / 2

/-- Theorem: The expected value of the sum of positions of NumRooks rooks 
    on a chessboard of size ChessboardSize is NumRooks * ExpectedSinglePosition -/
theorem expected_sum_of_rook_positions :
  NumRooks * ExpectedSinglePosition = 195 := by sorry

end NUMINAMATH_CALUDE_expected_sum_of_rook_positions_l324_32406


namespace NUMINAMATH_CALUDE_negation_equivalence_l324_32417

theorem negation_equivalence :
  (¬ ∃ x : ℝ, Real.exp x > x) ↔ (∀ x : ℝ, Real.exp x ≤ x) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l324_32417


namespace NUMINAMATH_CALUDE_x_squared_is_quadratic_l324_32476

/-- A quadratic equation in one variable is of the form ax² + bx + c = 0, where a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = x² is a quadratic equation in one variable -/
theorem x_squared_is_quadratic : is_quadratic_equation (λ x : ℝ => x^2) := by
  sorry

#check x_squared_is_quadratic

end NUMINAMATH_CALUDE_x_squared_is_quadratic_l324_32476


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l324_32405

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (d : RepeatingDecimal) : ℚ :=
  d.integerPart + d.repeatingPart / (99 : ℚ)

/-- The repeating decimal 0.72̅ -/
def zero_point_72_repeating : RepeatingDecimal :=
  ⟨0, 72⟩

/-- The repeating decimal 2.09̅ -/
def two_point_09_repeating : RepeatingDecimal :=
  ⟨2, 9⟩

/-- Theorem stating that the division of the two given repeating decimals equals 8/23 -/
theorem repeating_decimal_division :
    (toRational zero_point_72_repeating) / (toRational two_point_09_repeating) = 8 / 23 := by
  sorry


end NUMINAMATH_CALUDE_repeating_decimal_division_l324_32405


namespace NUMINAMATH_CALUDE_min_c_value_l324_32404

theorem min_c_value (a b c : ℕ) (h1 : a < b) (h2 : b < c) :
  (∃! x y : ℝ, 2 * x + y = 2031 ∧ y = |x - a| + |x - b| + |x - c|) →
  c ≥ 1016 ∧ ∃ a' b' : ℕ, a' < b' ∧ b' < 1016 ∧
    (∃! x y : ℝ, 2 * x + y = 2031 ∧ y = |x - a'| + |x - b'| + |x - 1016|) :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l324_32404


namespace NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l324_32458

/-- An arithmetic sequence with common difference d ≠ 0 and first term a₁ = 2d -/
def arithmetic_sequence (d : ℝ) (n : ℕ) : ℝ :=
  2 * d + (n - 1) * d

theorem arithmetic_sequence_geometric_mean (d : ℝ) (k : ℕ) (h : d ≠ 0) :
  (arithmetic_sequence d k) ^ 2 = (arithmetic_sequence d 1) * (arithmetic_sequence d (2 * k + 7)) →
  k = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l324_32458


namespace NUMINAMATH_CALUDE_dog_tricks_conversion_l324_32415

def base5_to_base10 (a b c d : ℕ) : ℕ :=
  d * 5^0 + c * 5^1 + b * 5^2 + a * 5^3

theorem dog_tricks_conversion :
  base5_to_base10 1 2 3 4 = 194 := by
  sorry

end NUMINAMATH_CALUDE_dog_tricks_conversion_l324_32415


namespace NUMINAMATH_CALUDE_find_M_l324_32483

theorem find_M : ∃ M : ℝ, (0.25 * M = 0.35 * 1800) ∧ (M = 2520) := by
  sorry

end NUMINAMATH_CALUDE_find_M_l324_32483


namespace NUMINAMATH_CALUDE_triangle_area_sqrt_3_l324_32428

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, prove that its area is √3 -/
theorem triangle_area_sqrt_3 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : b * Real.cos C + c * Real.cos B = a * Real.cos C + c * Real.cos A)
  (h2 : b * Real.cos C + c * Real.cos B = 2)
  (h3 : a * Real.cos C + Real.sqrt 3 * a * Real.sin C = b + c) :
  (1/2) * a * b * Real.sin C = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_sqrt_3_l324_32428


namespace NUMINAMATH_CALUDE_one_bee_has_six_legs_l324_32486

/-- The number of legs a bee has -/
def bee_legs : ℕ := sorry

/-- Two bees have 12 legs -/
axiom two_bees_legs : 2 * bee_legs = 12

/-- Prove that one bee has 6 legs -/
theorem one_bee_has_six_legs : bee_legs = 6 := by sorry

end NUMINAMATH_CALUDE_one_bee_has_six_legs_l324_32486


namespace NUMINAMATH_CALUDE_exam_pass_probability_l324_32498

/-- The probability of passing an exam given the following conditions:
  - There are 5 total questions
  - The candidate is familiar with 3 questions
  - The candidate randomly selects 3 questions to answer
  - The candidate needs to answer 2 questions correctly to pass
-/
theorem exam_pass_probability :
  let total_questions : ℕ := 5
  let familiar_questions : ℕ := 3
  let selected_questions : ℕ := 3
  let required_correct : ℕ := 2
  let pass_probability : ℚ := 7 / 10
  (Nat.choose familiar_questions selected_questions +
   Nat.choose familiar_questions (selected_questions - 1) * Nat.choose (total_questions - familiar_questions) 1) /
  Nat.choose total_questions selected_questions = pass_probability :=
by sorry

end NUMINAMATH_CALUDE_exam_pass_probability_l324_32498


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l324_32469

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (75 - c) = 9) :
  6 / (30 - a) + 14 / (70 - b) + 15 / (75 - c) = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l324_32469


namespace NUMINAMATH_CALUDE_digits_for_369_pages_l324_32459

/-- The total number of digits used to number pages in a book -/
def totalDigits (n : ℕ) : ℕ :=
  (min n 9) + 
  (max (min n 99 - 9) 0) * 2 + 
  (max (n - 99) 0) * 3

/-- Theorem: The total number of digits used in numbering the pages of a book with 369 pages is 999 -/
theorem digits_for_369_pages : totalDigits 369 = 999 := by
  sorry

end NUMINAMATH_CALUDE_digits_for_369_pages_l324_32459


namespace NUMINAMATH_CALUDE_customers_added_during_lunch_rush_l324_32418

theorem customers_added_during_lunch_rush 
  (initial_customers : ℕ) 
  (no_tip_customers : ℕ) 
  (tip_customers : ℕ) 
  (h1 : initial_customers = 29)
  (h2 : no_tip_customers = 34)
  (h3 : tip_customers = 15)
  (h4 : no_tip_customers + tip_customers = initial_customers + (customers_added : ℕ)) :
  customers_added = 20 :=
by sorry

end NUMINAMATH_CALUDE_customers_added_during_lunch_rush_l324_32418


namespace NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_implies_m_range_l324_32470

theorem intersection_in_fourth_quadrant_implies_m_range 
  (m : ℝ) 
  (line1 : ℝ → ℝ → Prop) 
  (line2 : ℝ → ℝ → Prop)
  (h1 : ∀ x y, line1 x y ↔ x + y - 3*m = 0)
  (h2 : ∀ x y, line2 x y ↔ 2*x - y + 2*m - 1 = 0)
  (h_intersect : ∃ x y, line1 x y ∧ line2 x y ∧ x > 0 ∧ y < 0) :
  -1 < m ∧ m < 1/8 := by
sorry

end NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_implies_m_range_l324_32470


namespace NUMINAMATH_CALUDE_f_properties_l324_32410

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2 + x + 1) * Real.exp x

theorem f_properties :
  ∀ a : ℝ,
  (∃ x_min : ℝ, ∀ x : ℝ, f 0 x_min ≤ f 0 x ∧ f 0 x_min = -Real.exp (-2)) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ →
    (a < 0 → (x₂ < -2 ∨ x₂ > -1/a) → f a x₁ > f a x₂) ∧
    (a < 0 → -2 < x₁ ∧ x₂ < -1/a → f a x₁ < f a x₂) ∧
    (a = 0 → x₂ < -2 → f a x₁ > f a x₂) ∧
    (a = 0 → -2 < x₁ → f a x₁ < f a x₂) ∧
    (0 < a ∧ a < 1/2 → -1/a < x₁ ∧ x₂ < -2 → f a x₁ > f a x₂) ∧
    (0 < a ∧ a < 1/2 → (x₂ < -1/a ∨ -2 < x₁) → f a x₁ < f a x₂) ∧
    (a = 1/2 → f a x₁ < f a x₂) ∧
    (a > 1/2 → -2 < x₁ ∧ x₂ < -1/a → f a x₁ > f a x₂) ∧
    (a > 1/2 → (x₂ < -2 ∨ -1/a < x₁) → f a x₁ < f a x₂)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l324_32410


namespace NUMINAMATH_CALUDE_town_growth_is_21_percent_l324_32438

/-- Represents the population of a town over a 20-year period -/
structure TownPopulation where
  pop1991 : Nat
  pop2001 : Nat
  pop2011 : Nat

/-- Conditions for the town population -/
def ValidPopulation (t : TownPopulation) : Prop :=
  ∃ p q : Nat,
    t.pop1991 = p^2 ∧
    t.pop2001 = t.pop1991 + 180 ∧
    t.pop2001 = q^2 + 16 ∧
    t.pop2011 = t.pop2001 + 180

/-- The percent growth of the population over 20 years -/
def PercentGrowth (t : TownPopulation) : ℚ :=
  (t.pop2011 - t.pop1991 : ℚ) / t.pop1991 * 100

/-- Theorem stating that the percent growth is 21% -/
theorem town_growth_is_21_percent (t : TownPopulation) 
  (h : ValidPopulation t) : PercentGrowth t = 21 := by
  sorry

#check town_growth_is_21_percent

end NUMINAMATH_CALUDE_town_growth_is_21_percent_l324_32438


namespace NUMINAMATH_CALUDE_solve_for_a_l324_32451

theorem solve_for_a (x y a : ℝ) (h1 : x = 2) (h2 : y = 3) (h3 : a * x + 3 * y = 13) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l324_32451


namespace NUMINAMATH_CALUDE_problem_statement_l324_32409

theorem problem_statement (x y : ℝ) 
  (h1 : 4 + x = 5 - y) 
  (h2 : 3 + y = 6 + x) : 
  4 - x = 5 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l324_32409


namespace NUMINAMATH_CALUDE_max_k_for_quadratic_roots_difference_l324_32443

theorem max_k_for_quadratic_roots_difference (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   x^2 + k*x - 3 = 0 ∧ 
   y^2 + k*y - 3 = 0 ∧ 
   |x - y| = 10) →
  k ≤ Real.sqrt 88 :=
sorry

end NUMINAMATH_CALUDE_max_k_for_quadratic_roots_difference_l324_32443


namespace NUMINAMATH_CALUDE_kitchen_length_l324_32489

theorem kitchen_length (tile_area : ℝ) (kitchen_width : ℝ) (num_tiles : ℕ) :
  tile_area = 6 →
  kitchen_width = 48 →
  num_tiles = 96 →
  (kitchen_width * (num_tiles * tile_area / kitchen_width) : ℝ) = 12 * kitchen_width :=
by sorry

end NUMINAMATH_CALUDE_kitchen_length_l324_32489


namespace NUMINAMATH_CALUDE_linear_regression_change_specific_regression_change_l324_32433

/-- Given a linear regression equation y = a + bx, this theorem proves
    that when x increases by 1 unit, y changes by b units. -/
theorem linear_regression_change (a b : ℝ) :
  let y : ℝ → ℝ := λ x ↦ a + b * x
  ∀ x : ℝ, y (x + 1) - y x = b := by
  sorry

/-- For the specific linear regression equation y = 2 - 3.5x,
    this theorem proves that when x increases by 1 unit, y decreases by 3.5 units. -/
theorem specific_regression_change :
  let y : ℝ → ℝ := λ x ↦ 2 - 3.5 * x
  ∀ x : ℝ, y (x + 1) - y x = -3.5 := by
  sorry

end NUMINAMATH_CALUDE_linear_regression_change_specific_regression_change_l324_32433


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l324_32455

theorem smallest_n_congruence :
  ∃ (n : ℕ), n > 0 ∧ 5 * n ≡ 1723 [MOD 26] ∧
  ∀ (m : ℕ), m > 0 ∧ 5 * m ≡ 1723 [MOD 26] → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l324_32455


namespace NUMINAMATH_CALUDE_intersection_of_lines_l324_32419

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (7/18, -1/6)

/-- First line equation: y = -3x + 1 -/
def line1 (x y : ℚ) : Prop := y = -3 * x + 1

/-- Second line equation: y + 4 = 15x - 2 -/
def line2 (x y : ℚ) : Prop := y + 4 = 15 * x - 2

theorem intersection_of_lines :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ x' y', (line1 x' y' ∧ line2 x' y') → (x' = x ∧ y' = y) :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l324_32419


namespace NUMINAMATH_CALUDE_tree_spacing_l324_32456

/-- Given a yard of length 400 meters with 26 equally spaced trees, including one at each end,
    the distance between consecutive trees is 16 meters. -/
theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) (h1 : yard_length = 400) (h2 : num_trees = 26) :
  yard_length / (num_trees - 1) = 16 :=
sorry

end NUMINAMATH_CALUDE_tree_spacing_l324_32456


namespace NUMINAMATH_CALUDE_simplify_fraction_l324_32472

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : (x^2 / (x - 1)) - (1 / (x - 1)) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l324_32472


namespace NUMINAMATH_CALUDE_second_number_proof_l324_32447

theorem second_number_proof (N : ℕ) : 
  (N % 144 = 29) → (6215 % 144 = 23) → N = 6365 :=
by
  sorry

end NUMINAMATH_CALUDE_second_number_proof_l324_32447


namespace NUMINAMATH_CALUDE_sector_area_l324_32464

theorem sector_area (θ : Real) (r : Real) (h1 : θ = 135) (h2 : r = 20) :
  (θ * π * r^2) / 360 = 150 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l324_32464


namespace NUMINAMATH_CALUDE_hoopit_toes_count_l324_32402

/-- Represents the number of toes a Hoopit has on each hand -/
def hoopit_toes_per_hand : ℕ := 3

/-- Represents the number of hands a Hoopit has -/
def hoopit_hands : ℕ := 4

/-- Represents the number of toes a Neglart has on each hand -/
def neglart_toes_per_hand : ℕ := 2

/-- Represents the number of hands a Neglart has -/
def neglart_hands : ℕ := 5

/-- Represents the number of Hoopit students on the bus -/
def hoopit_students : ℕ := 7

/-- Represents the number of Neglart students on the bus -/
def neglart_students : ℕ := 8

/-- Represents the total number of toes on the bus -/
def total_toes : ℕ := 164

theorem hoopit_toes_count : 
  hoopit_toes_per_hand * hoopit_hands * hoopit_students + 
  neglart_toes_per_hand * neglart_hands * neglart_students = total_toes :=
by sorry

end NUMINAMATH_CALUDE_hoopit_toes_count_l324_32402


namespace NUMINAMATH_CALUDE_book_sale_problem_l324_32491

theorem book_sale_problem (cost_loss : ℝ) (sale_price : ℝ) :
  cost_loss = 315 →
  sale_price = cost_loss * 0.85 →
  sale_price = (cost_loss + (2565 - 315)) * 1.19 →
  cost_loss + (2565 - 315) = 2565 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_problem_l324_32491


namespace NUMINAMATH_CALUDE_fifty_three_days_from_friday_l324_32457

/-- Days of the week represented as integers mod 7 -/
inductive DayOfWeek
| Sunday : DayOfWeek
| Monday : DayOfWeek
| Tuesday : DayOfWeek
| Wednesday : DayOfWeek
| Thursday : DayOfWeek
| Friday : DayOfWeek
| Saturday : DayOfWeek

def days_in_week : Nat := 7

def friday_to_int : Nat := 5

def add_days (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match (friday_to_int + n) % days_in_week with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

theorem fifty_three_days_from_friday :
  add_days DayOfWeek.Friday 53 = DayOfWeek.Tuesday := by
  sorry

end NUMINAMATH_CALUDE_fifty_three_days_from_friday_l324_32457


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_l324_32450

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  4 * (x + 1)^2 - 25 = 0 ↔ x = 3/2 ∨ x = -7/2 := by sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) :
  (x + 10)^3 = -125 ↔ x = -15 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_l324_32450


namespace NUMINAMATH_CALUDE_find_b_l324_32482

theorem find_b (a b c : ℚ) 
  (sum_eq : a + b + c = 150)
  (eq_after_changes : a + 10 = b - 5 ∧ b - 5 = 7 * c) : 
  b = 232 / 3 := by
sorry

end NUMINAMATH_CALUDE_find_b_l324_32482


namespace NUMINAMATH_CALUDE_inequality_proof_l324_32429

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h : 1/x + 1/y + 1/z = 2) :
  Real.sqrt (x + y + z) ≥ Real.sqrt (x - 1) + Real.sqrt (y - 1) + Real.sqrt (z - 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l324_32429


namespace NUMINAMATH_CALUDE_sum_of_squares_l324_32467

theorem sum_of_squares (a b : ℝ) (h1 : a - b = 10) (h2 : a * b = 25) : a^2 + b^2 = 150 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l324_32467


namespace NUMINAMATH_CALUDE_papi_calot_plants_to_buy_l324_32466

/-- Calculates the total number of plants needed for a given crop -/
def totalPlants (rows : ℕ) (plantsPerRow : ℕ) (additional : ℕ) : ℕ :=
  rows * plantsPerRow + additional

/-- Represents Papi Calot's garden planning -/
structure GardenPlan where
  potatoRows : ℕ
  potatoPlantsPerRow : ℕ
  additionalPotatoes : ℕ
  carrotRows : ℕ
  carrotPlantsPerRow : ℕ
  additionalCarrots : ℕ
  onionRows : ℕ
  onionPlantsPerRow : ℕ
  additionalOnions : ℕ

/-- Theorem stating the correct number of plants Papi Calot needs to buy -/
theorem papi_calot_plants_to_buy (plan : GardenPlan)
  (h_potato : plan.potatoRows = 10 ∧ plan.potatoPlantsPerRow = 25 ∧ plan.additionalPotatoes = 20)
  (h_carrot : plan.carrotRows = 15 ∧ plan.carrotPlantsPerRow = 30 ∧ plan.additionalCarrots = 30)
  (h_onion : plan.onionRows = 12 ∧ plan.onionPlantsPerRow = 20 ∧ plan.additionalOnions = 10) :
  totalPlants plan.potatoRows plan.potatoPlantsPerRow plan.additionalPotatoes = 270 ∧
  totalPlants plan.carrotRows plan.carrotPlantsPerRow plan.additionalCarrots = 480 ∧
  totalPlants plan.onionRows plan.onionPlantsPerRow plan.additionalOnions = 250 := by
  sorry

end NUMINAMATH_CALUDE_papi_calot_plants_to_buy_l324_32466


namespace NUMINAMATH_CALUDE_common_difference_is_one_l324_32423

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

theorem common_difference_is_one
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : d ≠ 0)
  (h2 : a 1 = 1)
  (h3 : arithmetic_sequence a d)
  (h4 : geometric_sequence (a 1) (a 3) (a 9)) :
  d = 1 :=
sorry

end NUMINAMATH_CALUDE_common_difference_is_one_l324_32423


namespace NUMINAMATH_CALUDE_total_basketballs_donated_prove_total_basketballs_l324_32474

/-- Calculates the total number of basketballs donated to a school --/
theorem total_basketballs_donated (total_donations : ℕ) (basketball_hoops : ℕ) (pool_floats : ℕ) 
  (footballs : ℕ) (tennis_balls : ℕ) : ℕ :=
  let basketballs_with_hoops := basketball_hoops / 2
  let undamaged_pool_floats := pool_floats * 3 / 4
  let accounted_donations := basketball_hoops + undamaged_pool_floats + footballs + tennis_balls
  let separate_basketballs := total_donations - accounted_donations
  basketballs_with_hoops + separate_basketballs

/-- Proves that the total number of basketballs donated is 90 --/
theorem prove_total_basketballs :
  total_basketballs_donated 300 60 120 50 40 = 90 := by
  sorry

end NUMINAMATH_CALUDE_total_basketballs_donated_prove_total_basketballs_l324_32474


namespace NUMINAMATH_CALUDE_angle_complement_relation_l324_32496

theorem angle_complement_relation (x : ℝ) : x = 70 → x = 2 * (90 - x) + 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_relation_l324_32496


namespace NUMINAMATH_CALUDE_unique_perfect_square_polynomial_l324_32439

theorem unique_perfect_square_polynomial : 
  ∃! y : ℤ, ∃ n : ℤ, y^4 + 4*y^3 + 9*y^2 + 2*y + 17 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_perfect_square_polynomial_l324_32439


namespace NUMINAMATH_CALUDE_unique_triples_l324_32420

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem unique_triples : 
  ∀ a b c : ℕ,
    (is_prime (a^2 - 23)) →
    (is_prime (b^2 - 23)) →
    ((a^2 - 23) * (b^2 - 23) = c^2 - 23) →
    ((a = 5 ∧ b = 6 ∧ c = 7) ∨ (a = 6 ∧ b = 5 ∧ c = 7)) :=
by sorry

end NUMINAMATH_CALUDE_unique_triples_l324_32420


namespace NUMINAMATH_CALUDE_smallest_cube_for_cone_l324_32494

/-- Represents a cone with given height and base diameter -/
structure Cone where
  height : ℝ
  baseDiameter : ℝ

/-- Represents a cube with given side length -/
structure Cube where
  sideLength : ℝ

/-- The volume of a cube -/
def cubeVolume (c : Cube) : ℝ := c.sideLength ^ 3

/-- A cube contains a cone if its side length is at least as large as both
    the cone's height and base diameter -/
def cubeContainsCone (cube : Cube) (cone : Cone) : Prop :=
  cube.sideLength ≥ cone.height ∧ cube.sideLength ≥ cone.baseDiameter

theorem smallest_cube_for_cone (c : Cone)
    (h1 : c.height = 15)
    (h2 : c.baseDiameter = 8) :
    ∃ (cube : Cube),
      cubeContainsCone cube c ∧
      cubeVolume cube = 3375 ∧
      ∀ (other : Cube), cubeContainsCone other c → cubeVolume other ≥ cubeVolume cube :=
  sorry

end NUMINAMATH_CALUDE_smallest_cube_for_cone_l324_32494


namespace NUMINAMATH_CALUDE_cone_spheres_radius_theorem_l324_32468

/-- A right circular cone with four congruent spheres inside --/
structure ConeWithSpheres where
  base_radius : ℝ
  height : ℝ
  sphere_radius : ℝ
  is_right_circular : Bool
  spheres_count : Nat
  spheres_congruent : Bool
  spheres_tangent_to_each_other : Bool
  spheres_tangent_to_base : Bool
  spheres_tangent_to_side : Bool

/-- The theorem stating the relationship between cone dimensions and sphere radius --/
theorem cone_spheres_radius_theorem (c : ConeWithSpheres) :
  c.base_radius = 6 ∧
  c.height = 15 ∧
  c.is_right_circular = true ∧
  c.spheres_count = 4 ∧
  c.spheres_congruent = true ∧
  c.spheres_tangent_to_each_other = true ∧
  c.spheres_tangent_to_base = true ∧
  c.spheres_tangent_to_side = true →
  c.sphere_radius = 45 / 7 := by
sorry

end NUMINAMATH_CALUDE_cone_spheres_radius_theorem_l324_32468


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l324_32437

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  ∃ (k : ℕ), k = 3 ∧ (5474827 - k) % 12 = 0 ∧ ∀ (m : ℕ), m < k → (5474827 - m) % 12 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l324_32437


namespace NUMINAMATH_CALUDE_seating_probability_l324_32442

/-- Represents a seating arrangement of 6 students in a 2x3 grid -/
def SeatingArrangement := Fin 6 → Fin 6

/-- The total number of possible seating arrangements -/
def totalArrangements : ℕ := 720

/-- Checks if three students are seated next to each other and adjacent in the same row or column -/
def isAdjacentArrangement (arr : SeatingArrangement) (a b c : Fin 6) : Prop :=
  sorry

/-- The number of arrangements where Abby, Bridget, and Chris are seated next to each other and adjacent in the same row or column -/
def favorableArrangements : ℕ := 114

/-- The probability of Abby, Bridget, and Chris being seated in a specific arrangement -/
def probability : ℚ := 19 / 120

theorem seating_probability :
  (favorableArrangements : ℚ) / totalArrangements = probability :=
sorry

end NUMINAMATH_CALUDE_seating_probability_l324_32442


namespace NUMINAMATH_CALUDE_min_abs_sum_l324_32499

theorem min_abs_sum (a b c : ℝ) (h1 : a + b + c = -2) (h2 : a * b * c = -4) :
  ∀ x y z : ℝ, x + y + z = -2 → x * y * z = -4 → |a| + |b| + |c| ≤ |x| + |y| + |z| ∧ 
  ∃ a₀ b₀ c₀ : ℝ, a₀ + b₀ + c₀ = -2 ∧ a₀ * b₀ * c₀ = -4 ∧ |a₀| + |b₀| + |c₀| = 6 := by
sorry

end NUMINAMATH_CALUDE_min_abs_sum_l324_32499


namespace NUMINAMATH_CALUDE_angle_C_in_triangle_ABC_l324_32471

theorem angle_C_in_triangle_ABC (a b c : ℝ) (A B C : ℝ) : 
  c = Real.sqrt 2 →
  b = Real.sqrt 6 →
  B = 2 * π / 3 →  -- 120° in radians
  C = π / 6  -- 30° in radians
:= by sorry

end NUMINAMATH_CALUDE_angle_C_in_triangle_ABC_l324_32471


namespace NUMINAMATH_CALUDE_evaluate_expression_l324_32487

theorem evaluate_expression : ((5^2 + 3)^2 - (5^2 - 3)^2)^3 = 27000000 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l324_32487


namespace NUMINAMATH_CALUDE_minimum_canvas_dimensions_l324_32481

/-- Represents the dimensions of a canvas --/
structure CanvasDimensions where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle given its width and height --/
def rectangleArea (w h : ℝ) : ℝ := w * h

/-- Represents the constraints for the canvas problem --/
structure CanvasConstraints where
  miniatureArea : ℝ
  topBottomMargin : ℝ
  sideMargin : ℝ

/-- Calculates the total canvas dimensions given the miniature dimensions and margins --/
def totalCanvasDimensions (miniWidth miniHeight topBottomMargin sideMargin : ℝ) : CanvasDimensions :=
  { width := miniWidth + 2 * sideMargin,
    height := miniHeight + 2 * topBottomMargin }

/-- Theorem stating the minimum dimensions of the required canvas --/
theorem minimum_canvas_dimensions (constraints : CanvasConstraints) 
  (h1 : constraints.miniatureArea = 72)
  (h2 : constraints.topBottomMargin = 4)
  (h3 : constraints.sideMargin = 2) :
  ∃ (minCanvas : CanvasDimensions),
    minCanvas.width = 10 ∧ 
    minCanvas.height = 20 ∧ 
    ∀ (canvas : CanvasDimensions),
      (∃ (miniWidth miniHeight : ℝ),
        rectangleArea miniWidth miniHeight = constraints.miniatureArea ∧
        canvas = totalCanvasDimensions miniWidth miniHeight constraints.topBottomMargin constraints.sideMargin) →
      canvas.width * canvas.height ≥ minCanvas.width * minCanvas.height :=
sorry

end NUMINAMATH_CALUDE_minimum_canvas_dimensions_l324_32481


namespace NUMINAMATH_CALUDE_factorial_sum_perfect_square_l324_32480

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sumFactorials (m : ℕ) : ℕ := (List.range m).map factorial |>.sum

def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem factorial_sum_perfect_square :
  ∀ m : ℕ, m > 0 → (isPerfectSquare (sumFactorials m) ↔ m = 1 ∨ m = 3) :=
by sorry

end NUMINAMATH_CALUDE_factorial_sum_perfect_square_l324_32480


namespace NUMINAMATH_CALUDE_shaded_area_problem_l324_32441

/-- The area of the shaded region in a figure where a 4-inch by 4-inch square 
    adjoins a 12-inch by 12-inch square. -/
theorem shaded_area_problem : 
  let small_square_side : ℝ := 4
  let large_square_side : ℝ := 12
  let small_square_area := small_square_side ^ 2
  let triangle_base := small_square_side
  let triangle_height := small_square_side * large_square_side / (large_square_side + small_square_side)
  let triangle_area := (1 / 2) * triangle_base * triangle_height
  let shaded_area := small_square_area - triangle_area
  shaded_area = 10
  := by sorry

end NUMINAMATH_CALUDE_shaded_area_problem_l324_32441


namespace NUMINAMATH_CALUDE_jerome_contacts_l324_32431

/-- Calculates the total number of contacts on Jerome's list --/
def total_contacts (classmates : ℕ) (family_members : ℕ) : ℕ :=
  classmates + (classmates / 2) + family_members

/-- Theorem stating that Jerome's contact list has 33 people --/
theorem jerome_contacts : total_contacts 20 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_jerome_contacts_l324_32431


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l324_32465

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The line equation passing through the intersection points of two circles --/
def intersectionLine (c1 c2 : Circle) : ℝ → ℝ → Prop :=
  fun x y => x + y = 23/8

theorem intersection_line_of_circles :
  let c1 : Circle := { center := (0, 0), radius := 5 }
  let c2 : Circle := { center := (4, 4), radius := 3 }
  ∀ x y : ℝ, (x^2 + y^2 = c1.radius^2) ∧ ((x - c2.center.1)^2 + (y - c2.center.2)^2 = c2.radius^2) →
    intersectionLine c1 c2 x y := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l324_32465


namespace NUMINAMATH_CALUDE_set_difference_M_N_l324_32416

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := {2, 3, 5}

def setDifference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem set_difference_M_N : setDifference M N = {1, 7, 9} := by
  sorry

end NUMINAMATH_CALUDE_set_difference_M_N_l324_32416


namespace NUMINAMATH_CALUDE_rotation_composition_l324_32407

/-- Represents a rotation in a plane -/
structure Rotation where
  center : ℝ × ℝ
  angle : ℝ

/-- Represents a translation in a plane -/
structure Translation where
  direction : ℝ × ℝ

/-- Represents the result of composing two rotations -/
inductive RotationComposition
  | IsRotation : Rotation → RotationComposition
  | IsTranslation : Translation → RotationComposition

/-- 
  Theorem: The composition of two rotations is either a rotation or a translation
  depending on the sum of their angles.
-/
theorem rotation_composition (r1 r2 : Rotation) :
  ∃ (result : RotationComposition),
    (¬ ∃ (k : ℤ), r1.angle + r2.angle = 2 * π * k → 
      ∃ (c : ℝ × ℝ), result = RotationComposition.IsRotation ⟨c, r1.angle + r2.angle⟩) ∧
    (∃ (k : ℤ), r1.angle + r2.angle = 2 * π * k → 
      ∃ (d : ℝ × ℝ), result = RotationComposition.IsTranslation ⟨d⟩) :=
by sorry


end NUMINAMATH_CALUDE_rotation_composition_l324_32407


namespace NUMINAMATH_CALUDE_net_gain_calculation_l324_32462

def initial_value : ℝ := 15000
def profit_percentage : ℝ := 0.20
def loss_percentage : ℝ := 0.15
def transaction_fee : ℝ := 300

def first_sale_price : ℝ := initial_value * (1 + profit_percentage)
def second_sale_price : ℝ := first_sale_price * (1 - loss_percentage)
def total_cost : ℝ := second_sale_price + transaction_fee

theorem net_gain_calculation :
  first_sale_price - total_cost = 2400 := by sorry

end NUMINAMATH_CALUDE_net_gain_calculation_l324_32462


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l324_32497

/-- Given a geometric sequence with four terms and common ratio 2,
    prove that (2a₁ + a₂) / (2a₃ + a₄) = 1/4 -/
theorem geometric_sequence_ratio (a₁ a₂ a₃ a₄ : ℝ) :
  a₂ = 2 * a₁ → a₃ = 2 * a₂ → a₄ = 2 * a₃ →
  (2 * a₁ + a₂) / (2 * a₃ + a₄) = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l324_32497


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l324_32434

theorem line_tangent_to_parabola :
  ∃ (m : ℝ), m = 49 ∧
  ∀ (x y : ℝ),
    (4 * x + 7 * y + m = 0) →
    (y^2 = 16 * x) →
    ∃! (x₀ y₀ : ℝ), 4 * x₀ + 7 * y₀ + m = 0 ∧ y₀^2 = 16 * x₀ :=
by sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l324_32434


namespace NUMINAMATH_CALUDE_largest_root_of_g_l324_32424

def g (x : ℝ) : ℝ := 24 * x^4 - 34 * x^2 + 6

theorem largest_root_of_g :
  ∃ (r : ℝ), r = 1/2 ∧ g r = 0 ∧ ∀ (x : ℝ), g x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_largest_root_of_g_l324_32424


namespace NUMINAMATH_CALUDE_complex_number_problem_l324_32479

theorem complex_number_problem (z : ℂ) :
  Complex.abs z = 1 ∧ (Complex.I * Complex.im ((3 + 4*Complex.I) * z) = (3 + 4*Complex.I) * z) →
  z = 4/5 + 3/5*Complex.I ∨ z = -4/5 - 3/5*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l324_32479


namespace NUMINAMATH_CALUDE_divisor_problem_l324_32453

theorem divisor_problem (f y d : ℕ) : 
  (∃ k : ℕ, f = k * d + 3) →
  (∃ l : ℕ, y = l * d + 4) →
  (∃ m : ℕ, f + y = m * d + 2) →
  d = 5 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l324_32453


namespace NUMINAMATH_CALUDE_inscribed_cylinder_properties_l324_32411

/-- An equilateral cylinder inscribed in a regular tetrahedron --/
structure InscribedCylinder where
  a : ℝ  -- Edge length of the tetrahedron
  r : ℝ  -- Radius of the cylinder
  h : ℝ  -- Height of the cylinder
  cylinder_equilateral : h = 2 * r
  cylinder_inscribed : r = (a * (2 * Real.sqrt 3 - Real.sqrt 6)) / 6

/-- Theorem about the properties of the inscribed cylinder --/
theorem inscribed_cylinder_properties (c : InscribedCylinder) :
  c.r = (c.a * (2 * Real.sqrt 3 - Real.sqrt 6)) / 6 ∧
  (4 * Real.pi * c.r^2 : ℝ) = 4 * Real.pi * ((c.a * (2 * Real.sqrt 3 - Real.sqrt 6)) / 6)^2 ∧
  (2 * Real.pi * c.r^3 : ℝ) = 2 * Real.pi * ((c.a * (2 * Real.sqrt 3 - Real.sqrt 6)) / 6)^3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_properties_l324_32411


namespace NUMINAMATH_CALUDE_no_quadratic_factorization_l324_32401

theorem no_quadratic_factorization :
  ¬ ∃ (a b c d : ℤ), ∀ (x : ℝ),
    x^4 + 2*x^2 + 2*x + 2 = (x^2 + a*x + b) * (x^2 + c*x + d) :=
by sorry

end NUMINAMATH_CALUDE_no_quadratic_factorization_l324_32401


namespace NUMINAMATH_CALUDE_triangle_is_right_angled_l324_32478

theorem triangle_is_right_angled : 
  let A : ℂ := 1
  let B : ℂ := Complex.I * 2
  let C : ℂ := 5 + Complex.I * 2
  let AB : ℂ := B - A
  let BC : ℂ := C - B
  let CA : ℂ := A - C
  Complex.abs AB ^ 2 + Complex.abs CA ^ 2 = Complex.abs BC ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_right_angled_l324_32478


namespace NUMINAMATH_CALUDE_a2_value_l324_32435

theorem a2_value (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, 1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 = 
    a₀ + a₁ * (x - 1) + a₂ * (x - 1)^2 + a₃ * (x - 1)^3 + 
    a₄ * (x - 1)^4 + a₅ * (x - 1)^5 + a₆ * (x - 1)^6 + a₇ * (x - 1)^7) →
  a₂ = 56 := by
sorry

end NUMINAMATH_CALUDE_a2_value_l324_32435


namespace NUMINAMATH_CALUDE_regular_octagon_extended_sides_angle_l324_32493

/-- A regular octagon with vertices A, B, C, D, E, F, G, H -/
structure RegularOctagon where
  vertices : Fin 8 → Point

/-- The angle formed by extending sides AB and GH of a regular octagon to meet at point Q -/
def angle_Q (octagon : RegularOctagon) : ℝ :=
  sorry

theorem regular_octagon_extended_sides_angle (octagon : RegularOctagon) :
  angle_Q octagon = 90 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_extended_sides_angle_l324_32493


namespace NUMINAMATH_CALUDE_polynomial_division_l324_32403

theorem polynomial_division (a b : ℝ) (h : b ≠ 2 * a) :
  (4 * a^2 - b^2) / (b - 2 * a) = -2 * a - b := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l324_32403


namespace NUMINAMATH_CALUDE_mart_income_percentage_l324_32490

def income_comparison (juan tim mart : ℝ) : Prop :=
  tim = juan * 0.6 ∧ mart = tim * 1.6

theorem mart_income_percentage (juan tim mart : ℝ) 
  (h : income_comparison juan tim mart) : mart = juan * 0.96 := by
  sorry

end NUMINAMATH_CALUDE_mart_income_percentage_l324_32490


namespace NUMINAMATH_CALUDE_simplify_fraction_l324_32448

theorem simplify_fraction : (150 : ℚ) / 6000 * 75 = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l324_32448


namespace NUMINAMATH_CALUDE_pet_shop_grooming_time_l324_32421

/-- The time it takes to groom all dogs in a pet shop -/
theorem pet_shop_grooming_time 
  (poodle_time : ℝ) 
  (terrier_time : ℝ) 
  (num_poodles : ℕ) 
  (num_terriers : ℕ) 
  (num_employees : ℕ) 
  (h1 : poodle_time = 30) 
  (h2 : terrier_time = poodle_time / 2) 
  (h3 : num_poodles = 3) 
  (h4 : num_terriers = 8) 
  (h5 : num_employees = 4) 
  (h6 : num_employees > 0) :
  (num_poodles * poodle_time + num_terriers * terrier_time) / num_employees = 52.5 := by
  sorry


end NUMINAMATH_CALUDE_pet_shop_grooming_time_l324_32421


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l324_32484

-- Define the quadratic function
def f (x : ℝ) : ℝ := (x - 2)^2 - 2

-- Theorem stating that f satisfies the given conditions
theorem quadratic_function_properties :
  (∃ (a : ℝ), f a = -2 ∧ ∀ x, f x ≥ f a) ∧  -- Vertex condition
  f 0 = 2                                   -- Y-intercept condition
  := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l324_32484


namespace NUMINAMATH_CALUDE_line_translation_l324_32414

/-- Given a line with equation y = -2x, prove that translating it upward by 1 unit results in the equation y = -2x + 1 -/
theorem line_translation (x y : ℝ) :
  (y = -2 * x) →  -- Original line equation
  (∃ (y' : ℝ), y' = y + 1 ∧ y' = -2 * x + 1) -- Translated line equation
  := by sorry

end NUMINAMATH_CALUDE_line_translation_l324_32414


namespace NUMINAMATH_CALUDE_line_system_properties_l324_32412

-- Define the line system M
def line_system (θ : ℝ) (x y : ℝ) : Prop :=
  x * Real.cos θ + y * Real.sin θ = 1

-- Define the region enclosed by the lines
def enclosed_region (p : ℝ × ℝ) : Prop :=
  ∃ θ, line_system θ p.1 p.2

-- Theorem statement
theorem line_system_properties :
  -- 1. The area of the region enclosed by the lines is π
  (∃ A : Set (ℝ × ℝ), (∀ p, p ∈ A ↔ enclosed_region p) ∧ MeasureTheory.volume A = π) ∧
  -- 2. Not all lines in the system are parallel
  (∃ θ₁ θ₂, θ₁ ≠ θ₂ ∧ ¬ (∀ x y, line_system θ₁ x y ↔ line_system θ₂ x y)) ∧
  -- 3. Not all lines in the system pass through a fixed point
  (¬ ∃ p : ℝ × ℝ, ∀ θ, line_system θ p.1 p.2) ∧
  -- 4. For any integer n ≥ 3, there exists a regular n-gon with edges on the lines of the system
  (∀ n : ℕ, n ≥ 3 → ∃ vertices : Fin n → ℝ × ℝ,
    (∀ i : Fin n, ∃ θ, line_system θ (vertices i).1 (vertices i).2) ∧
    (∀ i j : Fin n, (vertices i).1^2 + (vertices i).2^2 = (vertices j).1^2 + (vertices j).2^2) ∧
    (∀ i j : Fin n, i ≠ j → (vertices i).1 ≠ (vertices j).1 ∨ (vertices i).2 ≠ (vertices j).2)) :=
by sorry


end NUMINAMATH_CALUDE_line_system_properties_l324_32412


namespace NUMINAMATH_CALUDE_zhang_or_beibei_probability_l324_32444

/-- The number of singers in total -/
def total_singers : ℕ := 5

/-- The number of singers to be signed -/
def singers_to_sign : ℕ := 3

/-- The probability of signing a specific combination of singers -/
def prob_combination : ℚ := 1 / (total_singers.choose singers_to_sign)

/-- The probability that either Zhang Lei or Beibei will be signed -/
def prob_zhang_or_beibei : ℚ := 1 - ((total_singers - 2).choose singers_to_sign) * prob_combination

theorem zhang_or_beibei_probability :
  prob_zhang_or_beibei = 9 / 10 := by sorry

end NUMINAMATH_CALUDE_zhang_or_beibei_probability_l324_32444
