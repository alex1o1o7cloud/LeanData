import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l2690_269090

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 2 - t) 
  (h2 : y = 4 * t + 7) 
  (h3 : x = -3) : 
  y = 27 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2690_269090


namespace NUMINAMATH_CALUDE_triangle_larger_segment_l2690_269046

theorem triangle_larger_segment (a b c h x : ℝ) : 
  a = 35 → b = 65 → c = 85 → 
  a^2 = x^2 + h^2 → 
  b^2 = (c - x)^2 + h^2 → 
  c - x = 60 :=
by sorry

end NUMINAMATH_CALUDE_triangle_larger_segment_l2690_269046


namespace NUMINAMATH_CALUDE_fraction_equality_problem_l2690_269080

theorem fraction_equality_problem (x y : ℚ) :
  x / y = 12 / 5 → y = 25 → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_problem_l2690_269080


namespace NUMINAMATH_CALUDE_chromatic_number_iff_k_constructible_l2690_269055

/-- A graph is k-constructible if it can be built up from K_k by repeatedly adding a new vertex
    and joining it to a k-clique in the existing graph. -/
def is_k_constructible (G : SimpleGraph V) (k : ℕ) : Prop :=
  sorry

theorem chromatic_number_iff_k_constructible (G : SimpleGraph V) (k : ℕ) :
  G.chromaticNumber ≥ k ↔ ∃ H : SimpleGraph V, H ≤ G ∧ is_k_constructible H k :=
sorry

end NUMINAMATH_CALUDE_chromatic_number_iff_k_constructible_l2690_269055


namespace NUMINAMATH_CALUDE_triangle_circumcircle_identity_l2690_269023

/-- Given a triangle inscribed in a circle, this theorem states the relationship
    between the sides, angles, and the radius of the circumscribed circle. -/
theorem triangle_circumcircle_identity 
  (R : ℝ) (A B C : ℝ) (a b c : ℝ)
  (h_triangle : A + B + C = π)
  (h_a : a = 2 * R * Real.sin A)
  (h_b : b = 2 * R * Real.sin B)
  (h_c : c = 2 * R * Real.sin C) :
  a * Real.cos A + b * Real.cos B + c * Real.cos C = 4 * R * Real.sin A * Real.sin B * Real.sin C :=
by sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_identity_l2690_269023


namespace NUMINAMATH_CALUDE_binary_1101_to_base5_l2690_269009

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a natural number to its base-5 representation -/
def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: decimal_to_base5 (n / 5)

/-- The binary representation of the number we want to convert -/
def binary_1101 : List Bool := [true, false, true, true]

theorem binary_1101_to_base5 :
  decimal_to_base5 (binary_to_decimal binary_1101) = [3, 2] :=
by sorry

end NUMINAMATH_CALUDE_binary_1101_to_base5_l2690_269009


namespace NUMINAMATH_CALUDE_problem_solution_l2690_269068

theorem problem_solution (x y z : ℝ) : 
  (3 * x = 0.75 * y) →
  (x = 24) →
  (z = 0.5 * y) →
  (z = 48) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2690_269068


namespace NUMINAMATH_CALUDE_jeremy_overall_accuracy_l2690_269076

theorem jeremy_overall_accuracy 
  (individual_portion : Real) 
  (collaborative_portion : Real)
  (terry_individual_accuracy : Real)
  (terry_overall_accuracy : Real)
  (jeremy_individual_accuracy : Real)
  (h1 : individual_portion = 0.6)
  (h2 : collaborative_portion = 0.4)
  (h3 : individual_portion + collaborative_portion = 1)
  (h4 : terry_individual_accuracy = 0.75)
  (h5 : terry_overall_accuracy = 0.85)
  (h6 : jeremy_individual_accuracy = 0.8) :
  jeremy_individual_accuracy * individual_portion + 
  (terry_overall_accuracy - terry_individual_accuracy * individual_portion) = 0.88 :=
sorry

end NUMINAMATH_CALUDE_jeremy_overall_accuracy_l2690_269076


namespace NUMINAMATH_CALUDE_cube_of_hundred_l2690_269042

theorem cube_of_hundred : 99^3 + 3*(99^2) + 3*99 + 1 = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_hundred_l2690_269042


namespace NUMINAMATH_CALUDE_semicircles_to_circle_area_ratio_l2690_269038

theorem semicircles_to_circle_area_ratio :
  ∀ r : ℝ,
  r > 0 →
  let circle_area := π * (2*r)^2
  let semicircle_area := π * r^2
  (semicircle_area / circle_area) = 1/4 :=
by
  sorry

end NUMINAMATH_CALUDE_semicircles_to_circle_area_ratio_l2690_269038


namespace NUMINAMATH_CALUDE_production_rates_satisfy_conditions_unique_solution_l2690_269043

/-- The number of parts person A can make per day -/
def parts_per_day_A : ℕ := 60

/-- The number of parts person B can make per day -/
def parts_per_day_B : ℕ := 80

/-- The total number of machine parts -/
def total_parts : ℕ := 400

/-- Theorem stating that the given production rates satisfy the problem conditions -/
theorem production_rates_satisfy_conditions :
  (parts_per_day_A + 2 * parts_per_day_A + 2 * parts_per_day_B = total_parts - 60) ∧
  (3 * parts_per_day_A + 3 * parts_per_day_B = total_parts + 20) := by
  sorry

/-- Theorem proving the uniqueness of the solution -/
theorem unique_solution (a b : ℕ) 
  (h1 : a + 2 * a + 2 * b = total_parts - 60)
  (h2 : 3 * a + 3 * b = total_parts + 20) :
  a = parts_per_day_A ∧ b = parts_per_day_B := by
  sorry

end NUMINAMATH_CALUDE_production_rates_satisfy_conditions_unique_solution_l2690_269043


namespace NUMINAMATH_CALUDE_village_population_l2690_269095

theorem village_population (P : ℕ) (h : (90 : ℕ) * P = 8100 * 100) : P = 9000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l2690_269095


namespace NUMINAMATH_CALUDE_sum_reciprocal_squares_geq_sum_squares_l2690_269012

theorem sum_reciprocal_squares_geq_sum_squares 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (sum_eq_3 : a + b + c = 3) :
  1/a^2 + 1/b^2 + 1/c^2 ≥ a^2 + b^2 + c^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_squares_geq_sum_squares_l2690_269012


namespace NUMINAMATH_CALUDE_sqrt_six_div_sqrt_two_eq_sqrt_three_l2690_269011

theorem sqrt_six_div_sqrt_two_eq_sqrt_three : 
  Real.sqrt 6 / Real.sqrt 2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_div_sqrt_two_eq_sqrt_three_l2690_269011


namespace NUMINAMATH_CALUDE_sum_of_distances_l2690_269034

/-- A circle touches the sides of an angle at points A and B. 
    C is a point on the circle. -/
structure CircleConfig where
  A : Point
  B : Point
  C : Point

/-- The distance from C to line AB is 6 -/
def distance_to_AB (config : CircleConfig) : ℝ := 6

/-- The distances from C to the sides of the angle -/
def distance_to_sides (config : CircleConfig) : ℝ × ℝ := sorry

/-- One distance is 9 times less than the other -/
axiom distance_ratio (config : CircleConfig) : 
  let (d₁, d₂) := distance_to_sides config
  d₁ = (1/9) * d₂ ∨ d₂ = (1/9) * d₁

theorem sum_of_distances (config : CircleConfig) : 
  let (d₁, d₂) := distance_to_sides config
  distance_to_AB config + d₁ + d₂ = 12 :=
sorry

end NUMINAMATH_CALUDE_sum_of_distances_l2690_269034


namespace NUMINAMATH_CALUDE_coefficient_x_10_in_expansion_l2690_269032

theorem coefficient_x_10_in_expansion : ∃ (c : ℤ), c = -11 ∧ 
  (∀ (x : ℝ), (x - 1)^11 = c * x^10 + (λ (y : ℝ) => (y - 1)^11 - c * y^10) x) := by
sorry

end NUMINAMATH_CALUDE_coefficient_x_10_in_expansion_l2690_269032


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2690_269051

theorem complex_fraction_simplification :
  (5 + 6 * Complex.I) / (3 + Complex.I) = 21/10 + 13/10 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2690_269051


namespace NUMINAMATH_CALUDE_common_root_and_parameter_l2690_269002

theorem common_root_and_parameter :
  ∃ (x p : ℚ), 
    x = -5 ∧ 
    p = 14/3 ∧ 
    p = -(x^2 - x - 2) / (x - 1) ∧ 
    p = -(x^2 + 2*x - 1) / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_common_root_and_parameter_l2690_269002


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2690_269003

theorem quadratic_inequality_solution_sets 
  (a b c : ℝ) 
  (h : ∀ x, ax^2 + b*x + c > 0 ↔ 3 < x ∧ x < 6) :
  ∀ x, c*x^2 + b*x + a < 0 ↔ x < 1/6 ∨ x > 1/3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2690_269003


namespace NUMINAMATH_CALUDE_max_candies_eaten_is_27_l2690_269040

/-- Represents a box of candies with a label -/
structure CandyBox where
  label : Nat
  candies : Nat

/-- Represents the state of all candy boxes -/
def GameState := List CandyBox

/-- Initializes the game state with three boxes -/
def initialState : GameState :=
  [{ label := 4, candies := 10 }, { label := 7, candies := 10 }, { label := 10, candies := 10 }]

/-- Performs one operation on the game state -/
def performOperation (state : GameState) (boxIndex : Nat) : Option GameState :=
  sorry

/-- Calculates the total number of candies eaten after a sequence of operations -/
def candiesEaten (operations : List Nat) : Nat :=
  sorry

/-- The maximum number of candies that can be eaten -/
def maxCandiesEaten : Nat :=
  sorry

/-- Theorem stating the maximum number of candies that can be eaten is 27 -/
theorem max_candies_eaten_is_27 : maxCandiesEaten = 27 := by
  sorry

end NUMINAMATH_CALUDE_max_candies_eaten_is_27_l2690_269040


namespace NUMINAMATH_CALUDE_watson_class_first_graders_l2690_269039

/-- The number of first graders in Ms. Watson's class -/
def first_graders (total : ℕ) (kindergartners : ℕ) (second_graders : ℕ) : ℕ :=
  total - (kindergartners + second_graders)

/-- Theorem stating the number of first graders in Ms. Watson's class -/
theorem watson_class_first_graders :
  first_graders 42 14 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_watson_class_first_graders_l2690_269039


namespace NUMINAMATH_CALUDE_range_of_x_l2690_269087

theorem range_of_x (x : ℝ) : 
  (¬ (x ∈ Set.Icc 2 5 ∨ x < 1 ∨ x > 4)) → 
  x ∈ Set.Ico 1 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_x_l2690_269087


namespace NUMINAMATH_CALUDE_downstream_distance_l2690_269016

theorem downstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (travel_time : ℝ) 
  (h1 : boat_speed = 24) 
  (h2 : stream_speed = 4) 
  (h3 : travel_time = 3) : 
  boat_speed + stream_speed * travel_time = 84 := by
sorry

end NUMINAMATH_CALUDE_downstream_distance_l2690_269016


namespace NUMINAMATH_CALUDE_center_sum_is_six_l2690_269049

/-- A circle in a shifted coordinate system -/
structure ShiftedCircle where
  -- The equation of the circle in the shifted system
  equation : (x y : ℝ) → Prop := fun x y => (x - 1)^2 + (y + 2)^2 = 4*x + 12*y + 6
  -- The shift of the coordinate system
  shift : ℝ × ℝ := (1, -2)

/-- The center of a circle in the standard coordinate system -/
def standardCenter (c : ShiftedCircle) : ℝ × ℝ := sorry

theorem center_sum_is_six (c : ShiftedCircle) : 
  let (h, k) := standardCenter c
  h + k = 6 := by sorry

end NUMINAMATH_CALUDE_center_sum_is_six_l2690_269049


namespace NUMINAMATH_CALUDE_score_96_not_possible_l2690_269079

/-- Represents the score on a test with 25 questions -/
structure TestScore where
  correct : Nat
  unanswered : Nat
  incorrect : Nat
  h_total : correct + unanswered + incorrect = 25

/-- Calculates the total score for a given TestScore -/
def totalScore (ts : TestScore) : Nat :=
  4 * ts.correct + 2 * ts.unanswered

/-- Theorem stating that a score of 96 is not achievable -/
theorem score_96_not_possible :
  ¬ ∃ (ts : TestScore), totalScore ts = 96 := by
  sorry

end NUMINAMATH_CALUDE_score_96_not_possible_l2690_269079


namespace NUMINAMATH_CALUDE_tan_105_degrees_l2690_269066

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l2690_269066


namespace NUMINAMATH_CALUDE_water_flow_problem_l2690_269091

/-- The water flow problem -/
theorem water_flow_problem (x : ℝ) 
  (h1 : x > 0) -- Ensure x is positive
  (h2 : (2 * (30 / x) + 2 * (30 / x) + 4 * (60 / x)) / 2 = 18) -- Total water collected and dumped
  : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_water_flow_problem_l2690_269091


namespace NUMINAMATH_CALUDE_three_solutions_iff_specific_a_l2690_269050

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  ((abs (y + 2) + abs (x - 11) - 3) * (x^2 + y^2 - 13) = 0) ∧
  ((x - 5)^2 + (y + 2)^2 = a)

-- Define the condition for exactly three solutions
def has_exactly_three_solutions (a : ℝ) : Prop :=
  ∃! (s₁ s₂ s₃ : ℝ × ℝ), 
    system s₁.1 s₁.2 a ∧ 
    system s₂.1 s₂.2 a ∧ 
    system s₃.1 s₃.2 a ∧
    s₁ ≠ s₂ ∧ s₁ ≠ s₃ ∧ s₂ ≠ s₃

-- Theorem statement
theorem three_solutions_iff_specific_a :
  ∀ a : ℝ, has_exactly_three_solutions a ↔ (a = 9 ∨ a = 42 + 2 * Real.sqrt 377) :=
sorry

end NUMINAMATH_CALUDE_three_solutions_iff_specific_a_l2690_269050


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l2690_269075

/-- Given a rhombus with one diagonal of 65 meters and an area of 1950 square meters,
    prove that the length of the other diagonal is 60 meters. -/
theorem rhombus_diagonal (d₁ : ℝ) (area : ℝ) (d₂ : ℝ) : 
  d₁ = 65 → area = 1950 → area = (d₁ * d₂) / 2 → d₂ = 60 := by
sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l2690_269075


namespace NUMINAMATH_CALUDE_min_value_sum_product_l2690_269099

theorem min_value_sum_product (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 1 / (b + c) + 1 / (b + d) + 1 / (c + d)) ≥ 36 / 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l2690_269099


namespace NUMINAMATH_CALUDE_highest_divisible_digit_l2690_269065

theorem highest_divisible_digit : 
  ∃ (a : ℕ), a ≤ 9 ∧ 
  (365 * 1000 + a * 100 + 16) % 8 = 0 ∧
  ∀ (b : ℕ), b ≤ 9 → b > a → (365 * 1000 + b * 100 + 16) % 8 ≠ 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_highest_divisible_digit_l2690_269065


namespace NUMINAMATH_CALUDE_robie_chocolate_bags_l2690_269014

/-- Calculates the final number of chocolate bags after transactions -/
def final_chocolate_bags (initial : ℕ) (given_away : ℕ) (additional : ℕ) : ℕ :=
  initial - given_away + additional

/-- Proves that Robie's final number of chocolate bags is 4 -/
theorem robie_chocolate_bags : 
  final_chocolate_bags 3 2 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_robie_chocolate_bags_l2690_269014


namespace NUMINAMATH_CALUDE_lizzy_money_theorem_l2690_269060

def lizzy_money_problem (mother_gave uncle_gave father_gave spent_on_candy : ℕ) : Prop :=
  let initial_amount := mother_gave + father_gave
  let amount_after_spending := initial_amount - spent_on_candy
  let final_amount := amount_after_spending + uncle_gave
  final_amount = 140

theorem lizzy_money_theorem :
  lizzy_money_problem 80 70 40 50 := by
  sorry

end NUMINAMATH_CALUDE_lizzy_money_theorem_l2690_269060


namespace NUMINAMATH_CALUDE_percentage_10_years_or_more_is_correct_l2690_269026

/-- Represents the employment distribution at Apex Innovations -/
structure EmploymentDistribution (X : ℕ) :=
  (less_than_2_years : ℕ := 7 * X)
  (two_to_4_years : ℕ := 4 * X)
  (four_to_6_years : ℕ := 3 * X)
  (six_to_8_years : ℕ := 3 * X)
  (eight_to_10_years : ℕ := 2 * X)
  (ten_to_12_years : ℕ := 2 * X)
  (twelve_to_14_years : ℕ := X)
  (fourteen_to_16_years : ℕ := X)
  (sixteen_to_18_years : ℕ := X)

/-- Calculates the percentage of employees who have worked for 10 years or more -/
def percentage_10_years_or_more (dist : EmploymentDistribution X) : ℚ :=
  let total_employees := 23 * X
  let employees_10_years_or_more := 5 * X
  (employees_10_years_or_more : ℚ) / total_employees * 100

/-- Theorem stating that the percentage of employees who have worked for 10 years or more is (5/23) * 100 -/
theorem percentage_10_years_or_more_is_correct (X : ℕ) (dist : EmploymentDistribution X) :
  percentage_10_years_or_more dist = 5 / 23 * 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_10_years_or_more_is_correct_l2690_269026


namespace NUMINAMATH_CALUDE_combined_girls_avg_is_88_l2690_269092

/-- Represents a high school with average scores for boys, girls, and combined -/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  combined_avg : ℝ

/-- Represents the combined data for two schools -/
structure CombinedSchools where
  school1 : School
  school2 : School
  combined_boys_avg : ℝ

/-- Calculates the combined average score for girls across two schools -/
def combined_girls_avg (schools : CombinedSchools) : ℝ :=
  sorry

/-- The theorem stating that the combined average score for girls is 88 -/
theorem combined_girls_avg_is_88 (schools : CombinedSchools) 
  (h1 : schools.school1 = { boys_avg := 74, girls_avg := 77, combined_avg := 75 })
  (h2 : schools.school2 = { boys_avg := 83, girls_avg := 94, combined_avg := 90 })
  (h3 : schools.combined_boys_avg = 80) :
  combined_girls_avg schools = 88 := by
  sorry

end NUMINAMATH_CALUDE_combined_girls_avg_is_88_l2690_269092


namespace NUMINAMATH_CALUDE_marble_ratio_l2690_269029

/-- Proves that the ratio of marbles Lori gave to marbles Hilton lost is 2:1 --/
theorem marble_ratio (initial : ℕ) (found : ℕ) (lost : ℕ) (final : ℕ) 
  (h_initial : initial = 26)
  (h_found : found = 6)
  (h_lost : lost = 10)
  (h_final : final = 42) :
  (final - (initial + found - lost)) / lost = 2 := by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_l2690_269029


namespace NUMINAMATH_CALUDE_tan_negative_thirteen_fourths_pi_l2690_269044

theorem tan_negative_thirteen_fourths_pi : Real.tan (-13/4 * π) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_thirteen_fourths_pi_l2690_269044


namespace NUMINAMATH_CALUDE_stamp_collection_total_l2690_269030

theorem stamp_collection_total (foreign : ℕ) (old : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : foreign = 90)
  (h2 : old = 60)
  (h3 : both = 20)
  (h4 : neither = 70) :
  foreign + old - both + neither = 200 := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_total_l2690_269030


namespace NUMINAMATH_CALUDE_min_value_theorem_l2690_269067

theorem min_value_theorem (a b : ℝ) (h : a * b > 0) :
  (4 * b / a + (a - 2 * b) / b) ≥ 2 ∧
  (4 * b / a + (a - 2 * b) / b = 2 ↔ a = 2 * b) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2690_269067


namespace NUMINAMATH_CALUDE_boat_travel_time_difference_l2690_269007

def distance : ℝ := 90
def downstream_time : ℝ := 2.5191640969412834

theorem boat_travel_time_difference (v : ℝ) : 
  v > 3 →
  distance / (v - 3) - distance / (v + 3) = downstream_time →
  ∃ (diff : ℝ), abs (diff - 0.5088359030587166) < 1e-10 ∧ 
                 distance / (v - 3) - downstream_time = diff :=
by sorry

end NUMINAMATH_CALUDE_boat_travel_time_difference_l2690_269007


namespace NUMINAMATH_CALUDE_box_filling_cubes_l2690_269089

/-- Given a box with dimensions 49 inches long, 42 inches wide, and 14 inches deep,
    the smallest number of identical cubes that can completely fill the box without
    leaving any space is 84. -/
theorem box_filling_cubes : ∀ (length width depth : ℕ),
  length = 49 → width = 42 → depth = 14 →
  ∃ (cube_side : ℕ), cube_side > 0 ∧
    length % cube_side = 0 ∧
    width % cube_side = 0 ∧
    depth % cube_side = 0 ∧
    (length / cube_side) * (width / cube_side) * (depth / cube_side) = 84 :=
by sorry

end NUMINAMATH_CALUDE_box_filling_cubes_l2690_269089


namespace NUMINAMATH_CALUDE_nine_fourth_cubed_eq_three_to_nine_l2690_269083

theorem nine_fourth_cubed_eq_three_to_nine :
  9^4 + 9^4 + 9^4 = 3^9 := by sorry

end NUMINAMATH_CALUDE_nine_fourth_cubed_eq_three_to_nine_l2690_269083


namespace NUMINAMATH_CALUDE_improper_fraction_subtraction_l2690_269053

theorem improper_fraction_subtraction (a b n : ℕ) 
  (h1 : a > b) 
  (h2 : n < b) : 
  (a - n : ℚ) / (b - n) > (a : ℚ) / b := by
sorry

end NUMINAMATH_CALUDE_improper_fraction_subtraction_l2690_269053


namespace NUMINAMATH_CALUDE_parabola_circle_intersection_l2690_269086

/-- Parabola C₁: y² = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Circle C₂: (x-1)² + y² = 1 -/
def Circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- Point on the parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  hy : y^2 = 2 * C.p * x

/-- Only the vertex of C₁ is on C₂, all other points are outside -/
axiom vertex_on_circle_others_outside (C : Parabola) :
  Circle 0 0 ∧ ∀ (P : PointOnParabola C), P.x ≠ 0 → ¬Circle P.x P.y

/-- Fixed point M on C₁ with y₀ > 0 -/
structure FixedPoint (C : Parabola) extends PointOnParabola C where
  hy_pos : y > 0

/-- Two points A and B on C₁ -/
structure IntersectionPoints (C : Parabola) where
  A : PointOnParabola C
  B : PointOnParabola C

/-- Slopes of MA and MB exist and their angles are complementary -/
axiom complementary_slopes (C : Parabola) (M : FixedPoint C) (I : IntersectionPoints C) :
  ∃ (k : ℝ), k ≠ 0 ∧
    (I.A.y - M.y) / (I.A.x - M.x) = k ∧
    (I.B.y - M.y) / (I.B.x - M.x) = -k

/-- Main theorem -/
theorem parabola_circle_intersection (C : Parabola) (M : FixedPoint C) (I : IntersectionPoints C) :
  C.p ≥ 1 ∧
  ∃ (slope : ℝ), slope = -C.p / M.y ∧ slope ≠ 0 ∧
    (I.B.y - I.A.y) / (I.B.x - I.A.x) = slope := by sorry

end NUMINAMATH_CALUDE_parabola_circle_intersection_l2690_269086


namespace NUMINAMATH_CALUDE_peter_five_theorem_l2690_269015

theorem peter_five_theorem (N : ℕ+) :
  ∃ K : ℕ, ∀ k : ℕ, k ≥ K → (∃ d m n : ℕ, N * 5^k = 10^n * (10 * m + 5) + d ∧ d < 10^n) :=
sorry

end NUMINAMATH_CALUDE_peter_five_theorem_l2690_269015


namespace NUMINAMATH_CALUDE_min_distance_theorem_l2690_269071

/-- Line l in polar form -/
def line_l (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 4) = 2 * Real.sqrt 2

/-- Circle C in Cartesian form -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Scaling transformation -/
def scaling (x y x' y' : ℝ) : Prop := x' = 2 * x ∧ y' = 3 * y

/-- Curve C' after scaling transformation -/
def curve_C' (x' y' : ℝ) : Prop := x'^2 / 4 + y'^2 / 9 = 1

theorem min_distance_theorem :
  (∀ ρ θ, line_l ρ θ) →
  (∀ x y, circle_C x y) →
  (∃ d : ℝ, d = 2 * Real.sqrt 2 - 1 ∧
    (∀ x y, circle_C x y → (x + y - 4)^2 / 2 ≥ d^2)) ∧
  (∃ d' : ℝ, d' = 2 * Real.sqrt 2 - 2 ∧
    (∀ x' y', curve_C' x' y' → (x' + y' - 4)^2 / 2 ≥ d'^2)) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_theorem_l2690_269071


namespace NUMINAMATH_CALUDE_orange_distribution_l2690_269093

theorem orange_distribution (oranges_per_child : ℕ) (total_oranges : ℕ) (num_children : ℕ) : 
  oranges_per_child = 3 → 
  total_oranges = 12 → 
  num_children * oranges_per_child = total_oranges →
  num_children = 4 := by
sorry

end NUMINAMATH_CALUDE_orange_distribution_l2690_269093


namespace NUMINAMATH_CALUDE_equal_selection_probability_l2690_269085

/-- Represents the selection process for voluntary labor --/
structure SelectionProcess where
  total_students : ℕ
  excluded : ℕ
  selected : ℕ
  h_total : total_students = 1008
  h_excluded : excluded = 8
  h_selected : selected = 20
  h_remaining : total_students - excluded = 1000

/-- The probability of being selected for an individual student --/
def selection_probability (process : SelectionProcess) : ℚ :=
  process.selected / process.total_students

/-- States that the selection probability is equal for all students --/
theorem equal_selection_probability (process : SelectionProcess) :
  ∀ (student1 student2 : Fin process.total_students),
    selection_probability process = selection_probability process :=
by sorry

end NUMINAMATH_CALUDE_equal_selection_probability_l2690_269085


namespace NUMINAMATH_CALUDE_carnival_game_days_l2690_269000

def carnival_game (first_period_earnings : ℕ) (remaining_earnings : ℕ) (daily_earnings : ℕ) : Prop :=
  let first_period_days : ℕ := 20
  let remaining_days : ℕ := remaining_earnings / daily_earnings
  let total_days : ℕ := first_period_days + remaining_days
  (first_period_earnings = first_period_days * daily_earnings) ∧
  (remaining_earnings = remaining_days * daily_earnings) ∧
  (total_days = 31)

theorem carnival_game_days :
  carnival_game 120 66 6 := by
  sorry

end NUMINAMATH_CALUDE_carnival_game_days_l2690_269000


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2690_269084

/-- The value of a for which a line with equation ρsin(θ+ π/3)=a is tangent to a circle with equation ρ = 2sinθ in the polar coordinate system -/
theorem tangent_line_to_circle (a : ℝ) : 
  (∃ θ ρ, ρ = 2 * Real.sin θ ∧ ρ * Real.sin (θ + π/3) = a ∧ 
   ∀ θ' ρ', ρ' = 2 * Real.sin θ' → ρ' * Real.sin (θ' + π/3) ≠ a ∨ (θ' = θ ∧ ρ' = ρ)) →
  a = 3/2 ∨ a = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2690_269084


namespace NUMINAMATH_CALUDE_bales_stored_l2690_269019

theorem bales_stored (initial_bales final_bales : ℕ) 
  (h1 : initial_bales = 73)
  (h2 : final_bales = 96) :
  final_bales - initial_bales = 23 := by
sorry

end NUMINAMATH_CALUDE_bales_stored_l2690_269019


namespace NUMINAMATH_CALUDE_gold_cost_calculation_l2690_269072

/-- The cost of Gary and Anna's combined gold -/
def combined_gold_cost (gary_grams : ℝ) (gary_price : ℝ) (anna_grams : ℝ) (anna_price : ℝ) : ℝ :=
  gary_grams * gary_price + anna_grams * anna_price

/-- Theorem stating the combined cost of Gary and Anna's gold -/
theorem gold_cost_calculation :
  combined_gold_cost 30 15 50 20 = 1450 := by
  sorry

end NUMINAMATH_CALUDE_gold_cost_calculation_l2690_269072


namespace NUMINAMATH_CALUDE_cos_pi_minus_2alpha_l2690_269025

theorem cos_pi_minus_2alpha (α : Real) (h : Real.sin α = 2/3) :
  Real.cos (π - 2*α) = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_minus_2alpha_l2690_269025


namespace NUMINAMATH_CALUDE_sum_due_example_l2690_269078

/-- Given a Banker's Discount and a True Discount, calculate the sum due -/
def sum_due (BD TD : ℕ) : ℕ := TD + (BD - TD)

/-- Theorem: For a Banker's Discount of 288 and a True Discount of 240, the sum due is 288 -/
theorem sum_due_example : sum_due 288 240 = 288 := by
  sorry

end NUMINAMATH_CALUDE_sum_due_example_l2690_269078


namespace NUMINAMATH_CALUDE_parallelogram_product_l2690_269077

-- Define the parallelogram EFGH
structure Parallelogram :=
  (EF : ℝ)
  (FG : ℝ)
  (GH : ℝ)
  (HE : ℝ)

-- Define the theorem
theorem parallelogram_product (EFGH : Parallelogram) (w z : ℝ) :
  EFGH.EF = 42 ∧
  EFGH.FG = 4 * z^3 ∧
  EFGH.GH = 2 * w + 6 ∧
  EFGH.HE = 32 →
  w * z = 36 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_product_l2690_269077


namespace NUMINAMATH_CALUDE_factorization_proof_l2690_269058

theorem factorization_proof (a : ℝ) :
  74 * a^2 + 222 * a + 148 * a^3 = 74 * a * (2 * a^2 + a + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2690_269058


namespace NUMINAMATH_CALUDE_complex_simplification_l2690_269070

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_simplification :
  7 * (4 - 2*i) + 4*i * (3 - 2*i) = 36 - 2*i :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l2690_269070


namespace NUMINAMATH_CALUDE_initial_daily_consumption_l2690_269004

/-- Proves that the initial daily consumption per soldier is 3 kg -/
theorem initial_daily_consumption (initial_soldiers : ℕ) (initial_days : ℕ) 
  (new_soldiers : ℕ) (new_days : ℕ) (new_consumption : ℚ) : 
  initial_soldiers = 1200 →
  initial_days = 30 →
  new_soldiers = 528 →
  new_days = 25 →
  new_consumption = 5/2 →
  (initial_soldiers * initial_days * (3 : ℚ) = 
   (initial_soldiers + new_soldiers) * new_days * new_consumption) := by
  sorry

end NUMINAMATH_CALUDE_initial_daily_consumption_l2690_269004


namespace NUMINAMATH_CALUDE_unique_angle_solution_l2690_269021

theorem unique_angle_solution :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
  Real.tan ((150 - x) * π / 180) =
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) /
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
  x = 130 := by
sorry

end NUMINAMATH_CALUDE_unique_angle_solution_l2690_269021


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binom_300_150_l2690_269036

theorem largest_two_digit_prime_factor_of_binom_300_150 :
  ∃ (p : ℕ), p = 97 ∧ 
  Prime p ∧ 
  10 ≤ p ∧ p < 100 ∧
  p ∣ Nat.choose 300 150 ∧
  ∀ (q : ℕ), Prime q → 10 ≤ q → q < 100 → q ∣ Nat.choose 300 150 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binom_300_150_l2690_269036


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2690_269056

theorem fraction_subtraction : (4 : ℚ) / 5 - (1 : ℚ) / 5 = (3 : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2690_269056


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2690_269041

theorem greatest_divisor_with_remainders (d : ℕ) : d > 0 ∧ 
  d ∣ (4351 - 8) ∧ 
  d ∣ (5161 - 10) ∧ 
  (∀ k : ℕ, k > d → k ∣ (4351 - 8) → k ∣ (5161 - 10) → 
    (4351 % k ≠ 8 ∨ 5161 % k ≠ 10)) → 
  d = 1 :=
sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2690_269041


namespace NUMINAMATH_CALUDE_zephyria_license_plates_l2690_269048

/-- The number of letters in the alphabet. -/
def num_letters : ℕ := 26

/-- The number of digits (0-9). -/
def num_digits : ℕ := 10

/-- The number of letters in a Zephyrian license plate. -/
def letters_in_plate : ℕ := 3

/-- The number of digits in a Zephyrian license plate. -/
def digits_in_plate : ℕ := 4

/-- The total number of valid license plates in Zephyria. -/
def total_license_plates : ℕ := num_letters ^ letters_in_plate * num_digits ^ digits_in_plate

theorem zephyria_license_plates :
  total_license_plates = 175760000 := by
  sorry

end NUMINAMATH_CALUDE_zephyria_license_plates_l2690_269048


namespace NUMINAMATH_CALUDE_factorization_equality_l2690_269017

theorem factorization_equality (a b : ℝ) :
  a * b^2 - 2 * a^2 * b + a^2 = a * (b - a)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2690_269017


namespace NUMINAMATH_CALUDE_direct_inverse_variation_l2690_269064

theorem direct_inverse_variation (k : ℝ) : 
  (∃ (R S T : ℝ), R = k * S / T ∧ R = 2 ∧ S = 6 ∧ T = 3) →
  (∀ (R S T : ℝ), R = k * S / T → R = 8 ∧ T = 2 → S = 16) :=
by sorry

end NUMINAMATH_CALUDE_direct_inverse_variation_l2690_269064


namespace NUMINAMATH_CALUDE_tax_rate_calculation_l2690_269006

/-- Given a purchase in country B with a tax-free threshold, calculate the tax rate -/
theorem tax_rate_calculation (total_value tax_free_threshold tax_paid : ℝ) : 
  total_value = 1720 →
  tax_free_threshold = 600 →
  tax_paid = 123.2 →
  (tax_paid / (total_value - tax_free_threshold)) * 100 = 11 := by
sorry

end NUMINAMATH_CALUDE_tax_rate_calculation_l2690_269006


namespace NUMINAMATH_CALUDE_ashley_champagne_toast_l2690_269069

/-- The number of bottles of champagne needed for a wedding toast --/
def bottlesNeeded (guests : ℕ) (glassesPerGuest : ℕ) (servingsPerBottle : ℕ) : ℕ :=
  (guests * glassesPerGuest + servingsPerBottle - 1) / servingsPerBottle

/-- Theorem: Ashley needs 40 bottles of champagne for her wedding toast --/
theorem ashley_champagne_toast :
  bottlesNeeded 120 2 6 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ashley_champagne_toast_l2690_269069


namespace NUMINAMATH_CALUDE_leaf_decrease_l2690_269031

theorem leaf_decrease (green_yesterday red_yesterday yellow_yesterday 
                       green_today yellow_today red_today : ℕ) :
  green_yesterday = red_yesterday →
  yellow_yesterday = 7 * red_yesterday →
  green_today = yellow_today →
  red_today = 7 * yellow_today →
  green_today + yellow_today + red_today ≤ (green_yesterday + red_yesterday + yellow_yesterday) / 4 :=
by sorry

end NUMINAMATH_CALUDE_leaf_decrease_l2690_269031


namespace NUMINAMATH_CALUDE_set_operation_result_l2690_269010

def U : Set ℤ := {x | -3 < x ∧ x < 3}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 2}

theorem set_operation_result :
  A ∪ (U \ B) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_set_operation_result_l2690_269010


namespace NUMINAMATH_CALUDE_paint_intensity_problem_l2690_269018

theorem paint_intensity_problem (original_intensity new_intensity replacement_fraction : ℝ) 
  (h1 : original_intensity = 0.1)
  (h2 : new_intensity = 0.15)
  (h3 : replacement_fraction = 0.5) :
  let added_intensity := (new_intensity - (1 - replacement_fraction) * original_intensity) / replacement_fraction
  added_intensity = 0.2 := by
sorry

end NUMINAMATH_CALUDE_paint_intensity_problem_l2690_269018


namespace NUMINAMATH_CALUDE_jewel_price_after_one_cycle_l2690_269098

/-- Given an initial price P and a percentage x, if the price after two cycles
    of raising and lowering by x% is 2304, then the price after one such cycle
    is 2304 / (1 - (x/100)^2) -/
theorem jewel_price_after_one_cycle (P x : ℝ) :
  P * (1 - (x/100)^2)^2 = 2304 →
  P * (1 - (x/100)^2) = 2304 / (1 - (x/100)^2) := by
  sorry

end NUMINAMATH_CALUDE_jewel_price_after_one_cycle_l2690_269098


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l2690_269057

theorem smallest_number_with_given_remainders : ∃ (n : ℕ), n = 838 ∧ 
  (∃ (a : ℕ), 0 ≤ a ∧ a ≤ 19 ∧ 
    n % 20 = a ∧ 
    n % 21 = a + 1 ∧ 
    n % 22 = 2) ∧ 
  (∀ (m : ℕ), m < n → 
    ¬(∃ (b : ℕ), 0 ≤ b ∧ b ≤ 19 ∧ 
      m % 20 = b ∧ 
      m % 21 = b + 1 ∧ 
      m % 22 = 2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l2690_269057


namespace NUMINAMATH_CALUDE_fourth_circle_radius_l2690_269045

def circle_configuration (radii : Fin 7 → ℝ) : Prop :=
  ∀ i : Fin 6, radii i < radii (i + 1) ∧ 
  ∃ (r : ℝ), ∀ i : Fin 6, radii (i + 1) = radii i * r

theorem fourth_circle_radius 
  (radii : Fin 7 → ℝ) 
  (h_config : circle_configuration radii) 
  (h_smallest : radii 0 = 6) 
  (h_largest : radii 6 = 24) : 
  radii 3 = 12 :=
sorry

end NUMINAMATH_CALUDE_fourth_circle_radius_l2690_269045


namespace NUMINAMATH_CALUDE_sweater_a_markup_sweater_b_markup_l2690_269047

/-- Calculates the final price after applying a markup and two discounts -/
def final_price (wholesale : ℝ) (markup discount1 discount2 : ℝ) : ℝ :=
  wholesale * (1 + markup) * (1 - discount1) * (1 - discount2)

/-- Theorem for Sweater A -/
theorem sweater_a_markup (wholesale : ℝ) :
  final_price wholesale 3 0.2 0.5 = wholesale * 1.6 := by sorry

/-- Theorem for Sweater B -/
theorem sweater_b_markup (wholesale : ℝ) :
  ∃ ε > 0, ε < 0.0001 ∧ 
  |final_price wholesale 3.60606 0.25 0.45 - wholesale * 1.9| < ε := by sorry

end NUMINAMATH_CALUDE_sweater_a_markup_sweater_b_markup_l2690_269047


namespace NUMINAMATH_CALUDE_distance_is_95_over_17_l2690_269063

def point : ℝ × ℝ × ℝ := (2, 4, 5)
def line_point : ℝ × ℝ × ℝ := (5, 8, 9)
def line_direction : ℝ × ℝ × ℝ := (4, 3, -3)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_is_95_over_17 : 
  distance_to_line point line_point line_direction = 95 / 17 := by
  sorry

end NUMINAMATH_CALUDE_distance_is_95_over_17_l2690_269063


namespace NUMINAMATH_CALUDE_final_roll_probability_l2690_269037

/-- Probability of rolling a specific number on a standard die -/
def standardProbability : ℚ := 1 / 6

/-- Probability of not rolling the same number as the previous roll -/
def differentRollProbability : ℚ := 5 / 6

/-- Probability of rolling a 6 on the 15th roll if the 14th roll was 6 -/
def specialSixProbability : ℚ := 1 / 2

/-- Number of rolls before the final roll -/
def numPreviousRolls : ℕ := 13

/-- Probability that the 14th roll is a 6 given it's different from the 13th -/
def fourteenthRollSixProbability : ℚ := 1 / 5

/-- Combined probability for the 15th roll being the last -/
def fifteenthRollProbability : ℚ := 7 / 30

theorem final_roll_probability :
  (differentRollProbability ^ numPreviousRolls) * fifteenthRollProbability =
  (5 / 6 : ℚ) ^ 13 * (7 / 30 : ℚ) := by sorry

end NUMINAMATH_CALUDE_final_roll_probability_l2690_269037


namespace NUMINAMATH_CALUDE_wand_original_price_l2690_269013

/-- If a price is one-eighth of the original price and equals $12, then the original price is $96. -/
theorem wand_original_price (price : ℝ) (original : ℝ) : 
  price = original * (1/8) → price = 12 → original = 96 := by sorry

end NUMINAMATH_CALUDE_wand_original_price_l2690_269013


namespace NUMINAMATH_CALUDE_function_equality_l2690_269020

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 5

-- Theorem statement
theorem function_equality (x : ℝ) : 
  (2 * f x - 10 = f (x - 2)) ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l2690_269020


namespace NUMINAMATH_CALUDE_zeros_after_decimal_for_40_pow_40_l2690_269054

/-- The number of zeros immediately following the decimal point in 1/(40^40) -/
def zeros_after_decimal (n : ℕ) : ℕ :=
  let base := 40
  let exponent := 40
  let denominator := base ^ exponent
  -- The actual computation of zeros is not implemented here
  sorry

/-- Theorem stating that the number of zeros after the decimal point in 1/(40^40) is 76 -/
theorem zeros_after_decimal_for_40_pow_40 : zeros_after_decimal 40 = 76 := by
  sorry

end NUMINAMATH_CALUDE_zeros_after_decimal_for_40_pow_40_l2690_269054


namespace NUMINAMATH_CALUDE_girls_average_height_l2690_269027

theorem girls_average_height
  (num_boys : ℕ)
  (num_girls : ℕ)
  (total_students : ℕ)
  (avg_height_all : ℝ)
  (avg_height_boys : ℝ)
  (h1 : num_boys = 12)
  (h2 : num_girls = 10)
  (h3 : total_students = num_boys + num_girls)
  (h4 : avg_height_all = 103)
  (h5 : avg_height_boys = 108) :
  (total_students : ℝ) * avg_height_all - (num_boys : ℝ) * avg_height_boys = (num_girls : ℝ) * 97 :=
sorry

end NUMINAMATH_CALUDE_girls_average_height_l2690_269027


namespace NUMINAMATH_CALUDE_butterfly_stickers_l2690_269073

/-- Given a collection of butterflies with the following properties:
  * There are 330 butterflies in total
  * They are numbered consecutively starting from 1
  * 21 butterflies have double-digit numbers
  * 4 butterflies have triple-digit numbers
  Prove that the total number of single-digit stickers needed is 63 -/
theorem butterfly_stickers (total : ℕ) (double_digit : ℕ) (triple_digit : ℕ)
  (h_total : total = 330)
  (h_double : double_digit = 21)
  (h_triple : triple_digit = 4)
  (h_consecutive : ∀ n : ℕ, n ≤ total → n ≥ 1)
  (h_double_range : ∀ n : ℕ, n ≥ 10 ∧ n < 100 → n ≤ 30)
  (h_triple_range : ∀ n : ℕ, n ≥ 100 ∧ n < 1000 → n ≤ 103) :
  (total - double_digit - triple_digit) +
  (double_digit * 2) +
  (triple_digit * 3) = 63 := by
sorry

end NUMINAMATH_CALUDE_butterfly_stickers_l2690_269073


namespace NUMINAMATH_CALUDE_paving_cost_l2690_269001

/-- The cost of paving a rectangular floor -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 4) (h3 : rate = 750) :
  length * width * rate = 16500 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_l2690_269001


namespace NUMINAMATH_CALUDE_kopeck_payment_l2690_269008

theorem kopeck_payment (n : ℕ) (h : n > 7) : ∃ a b : ℕ, n = 3 * a + 5 * b := by
  sorry

end NUMINAMATH_CALUDE_kopeck_payment_l2690_269008


namespace NUMINAMATH_CALUDE_fourth_smallest_is_six_probability_l2690_269096

def S : Finset ℕ := Finset.range 15

def probability_fourth_smallest_is_six (n : ℕ) : ℚ :=
  let total_combinations := Nat.choose 15 8
  let favorable_outcomes := Nat.choose 5 3 * Nat.choose 9 5
  (favorable_outcomes : ℚ) / total_combinations

theorem fourth_smallest_is_six_probability :
  probability_fourth_smallest_is_six 6 = 4 / 21 := by
  sorry

#eval probability_fourth_smallest_is_six 6

end NUMINAMATH_CALUDE_fourth_smallest_is_six_probability_l2690_269096


namespace NUMINAMATH_CALUDE_min_value_f1_div_f2prime0_l2690_269024

/-- A quadratic function f(x) = ax² + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  f_prime_0_pos : 2 * a * 0 + b > 0
  range_nonneg : ∀ x, a * x^2 + b * x + c ≥ 0

/-- The theorem stating the minimum value of f(1) / f''(0) for quadratic functions with specific properties -/
theorem min_value_f1_div_f2prime0 (f : QuadraticFunction) :
  (∀ g : QuadraticFunction, (g.a + g.b + g.c) / (2 * g.a) ≥ (f.a + f.b + f.c) / (2 * f.a)) →
  (f.a + f.b + f.c) / (2 * f.a) = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_f1_div_f2prime0_l2690_269024


namespace NUMINAMATH_CALUDE_fraction_equality_l2690_269005

theorem fraction_equality (q r s t : ℚ) 
  (h1 : q / r = 9)
  (h2 : s / r = 6)
  (h3 : s / t = 1 / 2) :
  t / q = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l2690_269005


namespace NUMINAMATH_CALUDE_ryan_chinese_hours_l2690_269059

def hours_english : ℕ := 2
def hours_spanish : ℕ := 4

theorem ryan_chinese_hours :
  ∀ hours_chinese : ℕ,
  hours_chinese = hours_spanish + 1 →
  hours_chinese = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ryan_chinese_hours_l2690_269059


namespace NUMINAMATH_CALUDE_equal_selection_probability_l2690_269094

/-- Represents the probability of a student being selected in the survey -/
def probability_of_selection (total_students : ℕ) (students_to_select : ℕ) (students_to_eliminate : ℕ) : ℚ :=
  (students_to_select : ℚ) / (total_students : ℚ)

/-- Theorem stating that the probability of selection is equal for all students and is 50/2007 -/
theorem equal_selection_probability :
  let total_students : ℕ := 2007
  let students_to_select : ℕ := 50
  let students_to_eliminate : ℕ := 7
  probability_of_selection total_students students_to_select students_to_eliminate = 50 / 2007 := by
  sorry

#check equal_selection_probability

end NUMINAMATH_CALUDE_equal_selection_probability_l2690_269094


namespace NUMINAMATH_CALUDE_point_four_units_from_origin_l2690_269022

theorem point_four_units_from_origin (x : ℝ) : 
  |x| = 4 → x = 4 ∨ x = -4 := by
sorry

end NUMINAMATH_CALUDE_point_four_units_from_origin_l2690_269022


namespace NUMINAMATH_CALUDE_polynomial_properties_l2690_269061

-- Define the polynomial coefficients
variable (a : Fin 12 → ℚ)

-- Define the main equation
def main_equation (x : ℚ) : Prop :=
  (x - 2)^11 = a 0 + a 1 * (x - 1) + a 2 * (x - 1)^2 + 
               a 3 * (x - 1)^3 + a 4 * (x - 1)^4 + a 5 * (x - 1)^5 + 
               a 6 * (x - 1)^6 + a 7 * (x - 1)^7 + a 8 * (x - 1)^8 + 
               a 9 * (x - 1)^9 + a 10 * (x - 1)^10 + a 11 * (x - 1)^11

-- Theorem to prove
theorem polynomial_properties (a : Fin 12 → ℚ) 
  (h : ∀ x, main_equation a x) : 
  a 10 = -11 ∧ a 2 + a 4 + a 6 + a 8 + a 10 = -1023 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_properties_l2690_269061


namespace NUMINAMATH_CALUDE_quadratic_integer_root_l2690_269088

theorem quadratic_integer_root (a : ℤ) : 
  (∃ x : ℤ, x^2 + a*x + a = 0) ↔ (a = 0 ∨ a = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integer_root_l2690_269088


namespace NUMINAMATH_CALUDE_cubic_roots_determinant_l2690_269052

theorem cubic_roots_determinant (r s t : ℝ) (a b c : ℝ) : 
  a^3 - r*a^2 + s*a + t = 0 →
  b^3 - r*b^2 + s*b + t = 0 →
  c^3 - r*c^2 + s*c + t = 0 →
  Matrix.det !![1 + a^2, 1, 1; 1, 1 + b^2, 1; 1, 1, 1 + c^2] = r^2 + s^2 - 2*t :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_determinant_l2690_269052


namespace NUMINAMATH_CALUDE_max_intersection_points_l2690_269028

-- Define a circle on a plane
def Circle : Type := Unit

-- Define a line on a plane
def Line : Type := Unit

-- Function to count intersection points between a circle and a line
def circleLineIntersections (c : Circle) (l : Line) : ℕ := 2

-- Function to count intersection points between two lines
def lineLineIntersections (l1 l2 : Line) : ℕ := 1

-- Theorem stating the maximum number of intersection points
theorem max_intersection_points (c : Circle) (l1 l2 l3 : Line) :
  ∃ (n : ℕ), n ≤ 9 ∧ 
  (∀ (m : ℕ), m ≤ circleLineIntersections c l1 + 
               circleLineIntersections c l2 + 
               circleLineIntersections c l3 + 
               lineLineIntersections l1 l2 + 
               lineLineIntersections l1 l3 + 
               lineLineIntersections l2 l3 → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_max_intersection_points_l2690_269028


namespace NUMINAMATH_CALUDE_square_area_difference_l2690_269062

def original_side_length : ℝ := 6
def increase_in_length : ℝ := 1

theorem square_area_difference :
  let new_side_length := original_side_length + increase_in_length
  let original_area := original_side_length ^ 2
  let new_area := new_side_length ^ 2
  new_area - original_area = 13 := by sorry

end NUMINAMATH_CALUDE_square_area_difference_l2690_269062


namespace NUMINAMATH_CALUDE_ratio_of_a_to_c_l2690_269074

theorem ratio_of_a_to_c (a b c : ℚ) 
  (h1 : a / b = 5 / 3) 
  (h2 : b / c = 1 / 5) : 
  a / c = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_a_to_c_l2690_269074


namespace NUMINAMATH_CALUDE_katherines_bananas_l2690_269035

theorem katherines_bananas (apples pears bananas total : ℕ) : 
  apples = 4 →
  pears = 3 * apples →
  total = 21 →
  total = apples + pears + bananas →
  bananas = 5 := by
sorry

end NUMINAMATH_CALUDE_katherines_bananas_l2690_269035


namespace NUMINAMATH_CALUDE_first_class_occupancy_is_three_l2690_269082

/-- Represents the seating configuration and occupancy of an airplane -/
structure Airplane where
  first_class_capacity : ℕ
  business_class_capacity : ℕ
  economy_class_capacity : ℕ
  economy_occupancy : ℕ
  business_occupancy : ℕ
  first_class_occupancy : ℕ

/-- Theorem stating the number of people in first class -/
theorem first_class_occupancy_is_three (plane : Airplane) : plane.first_class_occupancy = 3 :=
  by
  have h1 : plane.first_class_capacity = 10 := by sorry
  have h2 : plane.business_class_capacity = 30 := by sorry
  have h3 : plane.economy_class_capacity = 50 := by sorry
  have h4 : plane.economy_occupancy = plane.economy_class_capacity / 2 := by sorry
  have h5 : plane.first_class_occupancy + plane.business_occupancy = plane.economy_occupancy := by sorry
  have h6 : plane.business_class_capacity - plane.business_occupancy = 8 := by sorry
  sorry

#check first_class_occupancy_is_three

end NUMINAMATH_CALUDE_first_class_occupancy_is_three_l2690_269082


namespace NUMINAMATH_CALUDE_winning_strategy_exists_l2690_269081

/-- Represents the three jars --/
inductive Jar
  | one
  | two
  | three

/-- Represents the three players --/
inductive Player
  | W
  | R
  | P

/-- The state of the game, tracking the number of nuts in each jar --/
structure GameState where
  jar1 : Nat
  jar2 : Nat
  jar3 : Nat

/-- Defines valid moves for each player --/
def validMove (p : Player) (j : Jar) : Prop :=
  match p, j with
  | Player.W, Jar.one => True
  | Player.W, Jar.two => True
  | Player.R, Jar.two => True
  | Player.R, Jar.three => True
  | Player.P, Jar.one => True
  | Player.P, Jar.three => True
  | _, _ => False

/-- Defines a winning state (any jar contains exactly 1999 nuts) --/
def isWinningState (s : GameState) : Prop :=
  s.jar1 = 1999 ∨ s.jar2 = 1999 ∨ s.jar3 = 1999

/-- Defines a strategy for W and P --/
def Strategy := GameState → Player → Jar

/-- Theorem: There exists a strategy for W and P that forces R to lose --/
theorem winning_strategy_exists :
  ∃ (strat : Strategy),
    ∀ (initial_state : GameState),
      ∀ (moves : Nat → Player → Jar),
        (∀ (n : Nat) (p : Player), validMove p (moves n p)) →
        ∃ (n : Nat),
          let final_state := -- state after n moves
            sorry -- Implementation of game progression
          isWinningState final_state ∧ moves n Player.R = (moves n Player.R) :=
sorry

end NUMINAMATH_CALUDE_winning_strategy_exists_l2690_269081


namespace NUMINAMATH_CALUDE_symmetry_of_regular_polygons_l2690_269097

-- Define the types of regular polygons
inductive RegularPolygon
  | EquilateralTriangle
  | Square
  | RegularPentagon
  | RegularHexagon

-- Define the properties of symmetry
def isAxiSymmetric (p : RegularPolygon) : Prop :=
  match p with
  | RegularPolygon.EquilateralTriangle => true
  | RegularPolygon.Square => true
  | RegularPolygon.RegularPentagon => true
  | RegularPolygon.RegularHexagon => true

def isCentrallySymmetric (p : RegularPolygon) : Prop :=
  match p with
  | RegularPolygon.EquilateralTriangle => false
  | RegularPolygon.Square => true
  | RegularPolygon.RegularPentagon => false
  | RegularPolygon.RegularHexagon => true

-- Theorem statement
theorem symmetry_of_regular_polygons :
  ∀ p : RegularPolygon,
    (isAxiSymmetric p ∧ isCentrallySymmetric p) ↔
    (p = RegularPolygon.Square ∨ p = RegularPolygon.RegularHexagon) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_of_regular_polygons_l2690_269097


namespace NUMINAMATH_CALUDE_true_discount_calculation_l2690_269033

/-- Given a bill with face value 540 and banker's discount 108, prove the true discount is 90 -/
theorem true_discount_calculation (face_value banker_discount : ℚ) 
  (h1 : face_value = 540)
  (h2 : banker_discount = 108)
  (h3 : ∀ (td : ℚ), banker_discount = td + (td * banker_discount / face_value)) :
  ∃ (true_discount : ℚ), true_discount = 90 ∧ 
    banker_discount = true_discount + (true_discount * banker_discount / face_value) := by
  sorry

#check true_discount_calculation

end NUMINAMATH_CALUDE_true_discount_calculation_l2690_269033
