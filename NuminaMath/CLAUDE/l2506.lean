import Mathlib

namespace NUMINAMATH_CALUDE_average_weight_increase_l2506_250644

theorem average_weight_increase (initial_count : ℕ) (initial_weight : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  initial_count = 8 →
  old_weight = 65 →
  new_weight = 89 →
  (initial_count * initial_weight - old_weight + new_weight) / initial_count - initial_weight = 3 :=
by sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2506_250644


namespace NUMINAMATH_CALUDE_gold_bars_lost_l2506_250670

theorem gold_bars_lost (initial_bars : ℕ) (num_friends : ℕ) (bars_per_friend : ℕ) : 
  initial_bars = 100 →
  num_friends = 4 →
  bars_per_friend = 20 →
  initial_bars - (num_friends * bars_per_friend) = 20 := by
  sorry

end NUMINAMATH_CALUDE_gold_bars_lost_l2506_250670


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l2506_250607

/-- Given a circle intersected by three equally spaced parallel lines creating chords of lengths 40, 36, and 30, the distance between two adjacent parallel lines is 2√10. -/
theorem parallel_lines_distance (r : ℝ) (d : ℝ) : 
  (∃ (chord1 chord2 chord3 : ℝ), 
    chord1 = 40 ∧ 
    chord2 = 36 ∧ 
    chord3 = 30 ∧ 
    chord1^2 = 4 * (r^2 - (d/2)^2) ∧
    chord2^2 = 4 * (r^2 - d^2) ∧
    chord3^2 = 4 * (r^2 - (3*d/2)^2)) →
  d = 2 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l2506_250607


namespace NUMINAMATH_CALUDE_students_in_two_classes_l2506_250686

theorem students_in_two_classes
  (total_students : ℕ)
  (history : ℕ)
  (math : ℕ)
  (english : ℕ)
  (science : ℕ)
  (geography : ℕ)
  (all_five : ℕ)
  (history_and_math : ℕ)
  (english_and_science : ℕ)
  (math_and_geography : ℕ)
  (h_total : total_students = 500)
  (h_history : history = 120)
  (h_math : math = 105)
  (h_english : english = 145)
  (h_science : science = 133)
  (h_geography : geography = 107)
  (h_all_five : all_five = 15)
  (h_history_and_math : history_and_math = 40)
  (h_english_and_science : english_and_science = 35)
  (h_math_and_geography : math_and_geography = 25)
  (h_at_least_one : total_students ≤ history + math + english + science + geography) :
  (history_and_math - all_five) + (english_and_science - all_five) + (math_and_geography - all_five) = 55 := by
  sorry

end NUMINAMATH_CALUDE_students_in_two_classes_l2506_250686


namespace NUMINAMATH_CALUDE_cos_75_cos_15_minus_sin_435_sin_15_eq_zero_l2506_250646

theorem cos_75_cos_15_minus_sin_435_sin_15_eq_zero :
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) - 
  Real.sin (435 * π / 180) * Real.sin (15 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_cos_15_minus_sin_435_sin_15_eq_zero_l2506_250646


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2506_250667

/-- An arithmetic sequence with the given property -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  property : a 2 + 4 * a 7 + a 12 = 96
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  2 * seq.a 3 + seq.a 15 = 48 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2506_250667


namespace NUMINAMATH_CALUDE_balloon_count_l2506_250683

/-- Represents the balloon shooting game with two levels. -/
structure BalloonGame where
  /-- The number of balloons missed in the first level -/
  missed_first_level : ℕ
  /-- The total number of balloons in each level -/
  total_balloons : ℕ

/-- The conditions of the balloon shooting game -/
def game_conditions (game : BalloonGame) : Prop :=
  let hit_first_level := 4 * game.missed_first_level + 2
  let hit_second_level := hit_first_level + 8
  hit_second_level = 6 * game.missed_first_level ∧
  game.total_balloons = hit_first_level + game.missed_first_level

/-- The theorem stating that the number of balloons in each level is 147 -/
theorem balloon_count (game : BalloonGame) 
  (h : game_conditions game) : game.total_balloons = 147 := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_l2506_250683


namespace NUMINAMATH_CALUDE_floor_equation_solutions_l2506_250651

theorem floor_equation_solutions (n : ℤ) : 
  (⌊n^2 / 9⌋ : ℤ) - (⌊n / 3⌋ : ℤ)^2 = 3 ↔ n = 8 ∨ n = 10 :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solutions_l2506_250651


namespace NUMINAMATH_CALUDE_prime_product_sum_l2506_250653

theorem prime_product_sum (m n p : ℕ) : 
  Prime m ∧ Prime n ∧ Prime p ∧ m * n * p = 5 * (m + n + p) → m^2 + n^2 + p^2 = 78 :=
by sorry

end NUMINAMATH_CALUDE_prime_product_sum_l2506_250653


namespace NUMINAMATH_CALUDE_smallest_overlap_percentage_l2506_250698

theorem smallest_overlap_percentage (coffee_drinkers tea_drinkers : ℝ) 
  (h1 : coffee_drinkers = 75)
  (h2 : tea_drinkers = 80) :
  coffee_drinkers + tea_drinkers - 100 = 55 :=
by sorry

end NUMINAMATH_CALUDE_smallest_overlap_percentage_l2506_250698


namespace NUMINAMATH_CALUDE_f_max_and_g_dominance_l2506_250614

open Real

noncomputable def f (x : ℝ) : ℝ := log x - 2 * x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (1/2) * m * x^2 + (m - 3) * x - 1

theorem f_max_and_g_dominance :
  (∃ (c : ℝ), c = -log 2 - 1 ∧ ∀ x > 0, f x ≤ c) ∧
  (∀ m : ℤ, (∀ x > 0, f x ≤ g m x) → m ≥ 2) ∧
  (∀ x > 0, f x ≤ g 2 x) :=
sorry

end NUMINAMATH_CALUDE_f_max_and_g_dominance_l2506_250614


namespace NUMINAMATH_CALUDE_cost_23_days_l2506_250615

/-- Calculate the cost of staying in a student youth hostel for a given number of days. -/
def hostelCost (days : ℕ) : ℚ :=
  let firstWeekRate : ℚ := 18
  let additionalWeekRate : ℚ := 12
  let firstWeekDays : ℕ := min 7 days
  let additionalDays : ℕ := days - firstWeekDays
  let firstWeekCost : ℚ := firstWeekRate * firstWeekDays
  let additionalCost : ℚ := additionalWeekRate * additionalDays
  firstWeekCost + additionalCost

/-- Theorem stating that the cost for a 23-day stay is $318.00 -/
theorem cost_23_days :
  hostelCost 23 = 318 := by
  sorry

#eval hostelCost 23

end NUMINAMATH_CALUDE_cost_23_days_l2506_250615


namespace NUMINAMATH_CALUDE_min_value_exponential_sum_l2506_250685

theorem min_value_exponential_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : 2 * x + y = 6) :
  ∃ (m : ℝ), m = 54 ∧ ∀ (z : ℝ), 9^x + 3^y ≥ z → z ≤ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_exponential_sum_l2506_250685


namespace NUMINAMATH_CALUDE_all_signs_used_l2506_250604

/-- Proves that all signs are used in the area code system --/
theorem all_signs_used (total_signs : Nat) (used_signs : Nat) (additional_codes : Nat) 
  (h1 : total_signs = 224)
  (h2 : used_signs = 222)
  (h3 : additional_codes = 888)
  (h4 : ∀ (sign : Nat), sign ≤ total_signs → (additional_codes / used_signs) * sign ≤ additional_codes) :
  total_signs - used_signs = 0 := by
  sorry

end NUMINAMATH_CALUDE_all_signs_used_l2506_250604


namespace NUMINAMATH_CALUDE_point_A_movement_l2506_250606

def point_movement (initial_x initial_y right_movement down_movement : ℝ) : ℝ × ℝ :=
  (initial_x + right_movement, initial_y - down_movement)

theorem point_A_movement :
  point_movement 1 0 2 3 = (3, -3) := by sorry

end NUMINAMATH_CALUDE_point_A_movement_l2506_250606


namespace NUMINAMATH_CALUDE_expression_factorization_l2506_250671

theorem expression_factorization (x y z : ℤ) :
  x^2 - (y + z)^2 + 2*x + y - z = (x - y - z) * (x + 2*y + 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2506_250671


namespace NUMINAMATH_CALUDE_linear_system_solution_l2506_250654

theorem linear_system_solution : 
  ∀ (x y : ℝ), 
    (2 * x + 3 * y = 4) → 
    (x = -y) → 
    (x = -4 ∧ y = 4) := by
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l2506_250654


namespace NUMINAMATH_CALUDE_moles_of_products_l2506_250669

-- Define the molar mass of Ammonium chloride
def molar_mass_NH4Cl : ℝ := 53.50

-- Define the mass of Ammonium chloride used
def mass_NH4Cl : ℝ := 53

-- Define the number of moles of Potassium hydroxide
def moles_KOH : ℝ := 1

-- Define the reaction ratio (1:1:1:1)
def reaction_ratio : ℝ := 1

-- Theorem stating the number of moles of products formed
theorem moles_of_products (ε : ℝ) (h_ε : ε > 0) :
  ∃ (moles_product : ℝ),
    moles_product > 0 ∧
    abs (moles_product - (mass_NH4Cl / molar_mass_NH4Cl)) < ε ∧
    moles_product * reaction_ratio = (mass_NH4Cl / molar_mass_NH4Cl) * reaction_ratio :=
by sorry

end NUMINAMATH_CALUDE_moles_of_products_l2506_250669


namespace NUMINAMATH_CALUDE_book_arrangement_count_l2506_250638

/-- The number of ways to arrange math and history books -/
def arrange_books (math_books : ℕ) (history_books : ℕ) : ℕ :=
  let math_groupings := (math_books + 2 - 1).choose 2
  let math_permutations := Nat.factorial math_books
  let history_placements := history_books.choose 3 * history_books.choose 3
  math_groupings * math_permutations * history_placements

/-- Theorem stating the number of arrangements for 4 math books and 6 history books -/
theorem book_arrangement_count :
  arrange_books 4 6 = 96000 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l2506_250638


namespace NUMINAMATH_CALUDE_max_area_inscribed_quadrilateral_l2506_250658

/-- A quadrilateral inscribed in a semi-circle -/
structure InscribedQuadrilateral (r : ℝ) where
  vertices : Fin 4 → ℝ × ℝ
  inside_semicircle : ∀ i, (vertices i).1^2 + (vertices i).2^2 ≤ r^2 ∧ (vertices i).2 ≥ 0

/-- The area of a quadrilateral -/
def area (q : InscribedQuadrilateral r) : ℝ :=
  sorry

/-- The shape of a half regular hexagon -/
def half_regular_hexagon (r : ℝ) : InscribedQuadrilateral r :=
  sorry

theorem max_area_inscribed_quadrilateral (r : ℝ) (hr : r > 0) :
  (∀ q : InscribedQuadrilateral r, area q ≤ (3 * Real.sqrt 3 / 4) * r^2) ∧
  area (half_regular_hexagon r) = (3 * Real.sqrt 3 / 4) * r^2 :=
sorry

end NUMINAMATH_CALUDE_max_area_inscribed_quadrilateral_l2506_250658


namespace NUMINAMATH_CALUDE_equal_chords_subtend_equal_arcs_l2506_250610

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A chord in a circle -/
structure Chord (c : Circle) where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- An arc in a circle -/
structure Arc (c : Circle) where
  startPoint : ℝ × ℝ
  endPoint : ℝ × ℝ

/-- The length of a chord -/
def chordLength (c : Circle) (ch : Chord c) : ℝ :=
  sorry

/-- The measure of an arc -/
def arcMeasure (c : Circle) (a : Arc c) : ℝ :=
  sorry

/-- A chord subtends an arc -/
def subtends (c : Circle) (ch : Chord c) (a : Arc c) : Prop :=
  sorry

theorem equal_chords_subtend_equal_arcs (c : Circle) (ch1 ch2 : Chord c) (a1 a2 : Arc c) :
  chordLength c ch1 = chordLength c ch2 →
  subtends c ch1 a1 →
  subtends c ch2 a2 →
  arcMeasure c a1 = arcMeasure c a2 :=
sorry

end NUMINAMATH_CALUDE_equal_chords_subtend_equal_arcs_l2506_250610


namespace NUMINAMATH_CALUDE_max_third_side_triangle_l2506_250660

theorem max_third_side_triangle (D E F : Real) (a b : Real) :
  -- Triangle DEF exists
  0 < D ∧ 0 < E ∧ 0 < F ∧ D + E + F = π →
  -- Two sides are 12 and 15
  a = 12 ∧ b = 15 →
  -- Angle condition
  Real.cos (2 * D) + Real.cos (2 * E) + Real.cos (2 * F) = 1 / 2 →
  -- Maximum length of third side
  ∃ c : Real, c ≤ Real.sqrt 549 ∧
    ∀ c' : Real, (c' = Real.sqrt (a^2 + b^2 - 2*a*b*(Real.cos E)) → c' ≤ c) :=
by sorry

end NUMINAMATH_CALUDE_max_third_side_triangle_l2506_250660


namespace NUMINAMATH_CALUDE_robin_additional_cupcakes_l2506_250628

/-- Calculates the number of additional cupcakes made given the initial number,
    the number sold, and the final number of cupcakes. -/
def additional_cupcakes (initial : ℕ) (sold : ℕ) (final : ℕ) : ℕ :=
  final - (initial - sold)

/-- Proves that Robin made 39 additional cupcakes given the problem conditions. -/
theorem robin_additional_cupcakes :
  additional_cupcakes 42 22 59 = 39 := by
  sorry

end NUMINAMATH_CALUDE_robin_additional_cupcakes_l2506_250628


namespace NUMINAMATH_CALUDE_student_line_arrangements_l2506_250627

def number_of_arrangements (n : ℕ) : ℕ := n.factorial

def arrangements_with_two_together (n : ℕ) : ℕ := (n - 1).factorial * 2

theorem student_line_arrangements :
  let total_students : ℕ := 5
  let total_arrangements := number_of_arrangements total_students
  let arrangements_with_specific_two_together := arrangements_with_two_together total_students
  total_arrangements - arrangements_with_specific_two_together = 72 := by
sorry

end NUMINAMATH_CALUDE_student_line_arrangements_l2506_250627


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2506_250662

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x^2 - 1)}
def B : Set ℝ := {x : ℝ | ∃ y : ℝ, y = x^2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | x ≥ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2506_250662


namespace NUMINAMATH_CALUDE_inheritance_problem_l2506_250602

theorem inheritance_problem (total_inheritance : ℝ) (additional_share : ℝ) 
  (h1 : total_inheritance = 84000)
  (h2 : additional_share = 3500)
  (h3 : ∃ x : ℕ, x > 2 ∧ 
    total_inheritance / x + additional_share = total_inheritance / (x - 2)) :
  ∃ x : ℕ, x = 8 ∧ x > 2 ∧ 
    total_inheritance / x + additional_share = total_inheritance / (x - 2) :=
sorry

end NUMINAMATH_CALUDE_inheritance_problem_l2506_250602


namespace NUMINAMATH_CALUDE_KBrO3_molecular_weight_l2506_250601

/-- Atomic weight of potassium in g/mol -/
def atomic_weight_K : ℝ := 39.10

/-- Atomic weight of bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- Atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- Molecular weight of KBrO3 in g/mol -/
def molecular_weight_KBrO3 : ℝ :=
  atomic_weight_K + atomic_weight_Br + 3 * atomic_weight_O

/-- Theorem stating that the molecular weight of KBrO3 is 167.00 g/mol -/
theorem KBrO3_molecular_weight :
  molecular_weight_KBrO3 = 167.00 := by
  sorry

end NUMINAMATH_CALUDE_KBrO3_molecular_weight_l2506_250601


namespace NUMINAMATH_CALUDE_fraction_sum_l2506_250611

theorem fraction_sum (a b : ℚ) (h : a / b = 3 / 5) : (a + b) / b = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l2506_250611


namespace NUMINAMATH_CALUDE_angle_A_value_min_side_a_value_l2506_250650

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the conditions
axiom angle_side_relation : (2 * c - b) * Real.cos A = a * Real.cos B
axiom triangle_area : (1 / 2) * b * c * Real.sin A = 2 * Real.sqrt 3

-- Theorem statements
theorem angle_A_value : A = π / 3 := by sorry

theorem min_side_a_value : ∃ (a_min : ℝ), a_min = 2 * Real.sqrt 2 ∧ ∀ (a : ℝ), a ≥ a_min := by sorry

end NUMINAMATH_CALUDE_angle_A_value_min_side_a_value_l2506_250650


namespace NUMINAMATH_CALUDE_intersection_distance_l2506_250655

/-- The distance between the intersection points of the line y = x and the circle (x-2)^2 + (y-1)^2 = 1 is √2. -/
theorem intersection_distance :
  ∃ (P Q : ℝ × ℝ),
    (P.1 = P.2 ∧ (P.1 - 2)^2 + (P.2 - 1)^2 = 1) ∧
    (Q.1 = Q.2 ∧ (Q.1 - 2)^2 + (Q.2 - 1)^2 = 1) ∧
    P ≠ Q ∧
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l2506_250655


namespace NUMINAMATH_CALUDE_quadratic_pair_sum_zero_l2506_250619

/-- A quadratic function and its inverse -/
def QuadraticPair (a b c : ℝ) : Prop :=
  ∃ (g : ℝ → ℝ) (g_inv : ℝ → ℝ),
    (∀ x, g x = a * x^2 + b * x + c) ∧
    (∀ x, g_inv x = c * x^2 + b * x + a) ∧
    (∀ x, g (g_inv x) = x) ∧
    (∀ x, g_inv (g x) = x)

theorem quadratic_pair_sum_zero (a b c : ℝ) (h : QuadraticPair a b c) :
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_pair_sum_zero_l2506_250619


namespace NUMINAMATH_CALUDE_axis_of_symmetry_sine_curve_l2506_250624

/-- The axis of symmetry for the sine curve y = sin(2πx - π/3) is x = 5/12 -/
theorem axis_of_symmetry_sine_curve (x : ℝ) : 
  (∃ (k : ℤ), x = k / 2 + 5 / 12) ↔ 
  (∃ (n : ℤ), 2 * π * x - π / 3 = n * π + π / 2) :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_sine_curve_l2506_250624


namespace NUMINAMATH_CALUDE_smallest_k_sum_squares_div_180_l2506_250639

/-- Sum of squares from 1 to n -/
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- Predicate for a number being divisible by 180 -/
def divisible_by_180 (n : ℕ) : Prop := ∃ m : ℕ, n = 180 * m

theorem smallest_k_sum_squares_div_180 :
  (∀ k < 216, ¬(divisible_by_180 (sum_of_squares k))) ∧
  (divisible_by_180 (sum_of_squares 216)) := by sorry

end NUMINAMATH_CALUDE_smallest_k_sum_squares_div_180_l2506_250639


namespace NUMINAMATH_CALUDE_washington_high_ratio_l2506_250663

/-- The student-teacher ratio at Washington High School -/
def student_teacher_ratio (num_students : ℕ) (num_teachers : ℕ) : ℚ :=
  num_students / num_teachers

/-- Theorem: The student-teacher ratio at Washington High School is 27.5 to 1 -/
theorem washington_high_ratio :
  student_teacher_ratio 1155 42 = 27.5 := by
  sorry

end NUMINAMATH_CALUDE_washington_high_ratio_l2506_250663


namespace NUMINAMATH_CALUDE_number_of_clients_l2506_250625

/-- Proves that the number of clients who visited the garage is 15, given the specified conditions. -/
theorem number_of_clients (num_cars : ℕ) (selections_per_client : ℕ) (selections_per_car : ℕ) : 
  num_cars = 10 → selections_per_client = 2 → selections_per_car = 3 → 
  (num_cars * selections_per_car) / selections_per_client = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_of_clients_l2506_250625


namespace NUMINAMATH_CALUDE_hyperbola_from_circle_intersection_l2506_250682

/-- Circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 9 = 0

/-- Points A and B on y-axis -/
def points_on_y_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = 0 ∧ B.1 = 0 ∧ circle_eq A.1 A.2 ∧ circle_eq B.1 B.2

/-- Points A and B trisect focal distance -/
def trisect_focal_distance (A B : ℝ × ℝ) (c : ℝ) : Prop :=
  abs (A.2 - B.2) = 2 * c / 3

/-- Standard hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop := y^2/9 - x^2/72 = 1

/-- Main theorem -/
theorem hyperbola_from_circle_intersection :
  ∀ (A B : ℝ × ℝ) (c : ℝ),
  points_on_y_axis A B →
  trisect_focal_distance A B c →
  ∀ (x y : ℝ), hyperbola_eq x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_from_circle_intersection_l2506_250682


namespace NUMINAMATH_CALUDE_inequality_chain_l2506_250640

theorem inequality_chain (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (2 : ℝ) / ((1 / a) + (1 / b)) < Real.sqrt (a * b) ∧ Real.sqrt (a * b) < (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l2506_250640


namespace NUMINAMATH_CALUDE_sin_difference_l2506_250668

theorem sin_difference (A B : Real) 
  (h1 : Real.tan A = 2 * Real.tan B) 
  (h2 : Real.sin (A + B) = 1/4) : 
  Real.sin (A - B) = 1/12 := by
sorry

end NUMINAMATH_CALUDE_sin_difference_l2506_250668


namespace NUMINAMATH_CALUDE_fib_identity_fib_1094_1096_minus_1095_squared_l2506_250696

/-- The Fibonacci sequence -/
def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The identity for Fibonacci numbers -/
theorem fib_identity (n : ℕ) :
  fib (n + 2) * fib n - fib (n + 1)^2 = (-1)^(n + 1) := by sorry

/-- The main theorem to prove -/
theorem fib_1094_1096_minus_1095_squared :
  fib 1094 * fib 1096 - fib 1095^2 = -1 := by sorry

end NUMINAMATH_CALUDE_fib_identity_fib_1094_1096_minus_1095_squared_l2506_250696


namespace NUMINAMATH_CALUDE_fraction_simplification_l2506_250676

theorem fraction_simplification :
  (270 : ℚ) / 18 * 7 / 140 * 9 / 4 = 27 / 16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2506_250676


namespace NUMINAMATH_CALUDE_lucia_dance_cost_l2506_250634

/-- Represents the cost of dance classes for a week -/
structure DanceClassesCost where
  hip_hop_classes : Nat
  ballet_classes : Nat
  jazz_classes : Nat
  hip_hop_cost : Nat
  ballet_cost : Nat
  jazz_cost : Nat

/-- Calculates the total cost of dance classes for a week -/
def total_cost (c : DanceClassesCost) : Nat :=
  c.hip_hop_classes * c.hip_hop_cost +
  c.ballet_classes * c.ballet_cost +
  c.jazz_classes * c.jazz_cost

/-- Theorem stating that Lucia's total dance class cost for a week is $52 -/
theorem lucia_dance_cost :
  let c : DanceClassesCost := {
    hip_hop_classes := 2,
    ballet_classes := 2,
    jazz_classes := 1,
    hip_hop_cost := 10,
    ballet_cost := 12,
    jazz_cost := 8
  }
  total_cost c = 52 := by
  sorry


end NUMINAMATH_CALUDE_lucia_dance_cost_l2506_250634


namespace NUMINAMATH_CALUDE_baseball_average_runs_l2506_250629

/-- Represents the scoring pattern of a baseball team over a series of games -/
structure ScoringPattern where
  games : ℕ
  oneRun : ℕ
  fourRuns : ℕ
  fiveRuns : ℕ

/-- Calculates the average runs per game given a scoring pattern -/
def averageRuns (pattern : ScoringPattern) : ℚ :=
  (pattern.oneRun * 1 + pattern.fourRuns * 4 + pattern.fiveRuns * 5) / pattern.games

/-- Theorem stating that for the given scoring pattern, the average runs per game is 4 -/
theorem baseball_average_runs :
  let pattern : ScoringPattern := {
    games := 6,
    oneRun := 1,
    fourRuns := 2,
    fiveRuns := 3
  }
  averageRuns pattern = 4 := by sorry

end NUMINAMATH_CALUDE_baseball_average_runs_l2506_250629


namespace NUMINAMATH_CALUDE_emily_calculation_l2506_250684

theorem emily_calculation (x y z : ℝ) 
  (h1 : 2*x - 3*y + z = 14) 
  (h2 : 2*x - 3*y - z = 6) : 
  2*x - 3*y = 10 := by
sorry

end NUMINAMATH_CALUDE_emily_calculation_l2506_250684


namespace NUMINAMATH_CALUDE_apples_per_box_l2506_250623

/-- Given the following conditions:
    - There are 180 apples in each crate
    - 12 crates of apples were delivered
    - 160 apples were rotten and thrown away
    - The remaining apples were packed into 100 boxes
    Prove that there are 20 apples in each box -/
theorem apples_per_box :
  ∀ (apples_per_crate crates_delivered rotten_apples total_boxes : ℕ),
    apples_per_crate = 180 →
    crates_delivered = 12 →
    rotten_apples = 160 →
    total_boxes = 100 →
    (apples_per_crate * crates_delivered - rotten_apples) / total_boxes = 20 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_box_l2506_250623


namespace NUMINAMATH_CALUDE_power_of_power_l2506_250622

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2506_250622


namespace NUMINAMATH_CALUDE_probability_neither_red_nor_purple_l2506_250689

theorem probability_neither_red_nor_purple (total : ℕ) (red : ℕ) (purple : ℕ) 
  (h1 : total = 60) (h2 : red = 5) (h3 : purple = 7) :
  (total - (red + purple)) / total = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_neither_red_nor_purple_l2506_250689


namespace NUMINAMATH_CALUDE_kindWizardCanAchieveGoal_l2506_250675

/-- Represents a gnome -/
structure Gnome where
  id : Nat

/-- Represents a friendship between two gnomes -/
structure Friendship where
  gnome1 : Gnome
  gnome2 : Gnome

/-- Represents a round table with gnomes -/
structure RoundTable where
  gnomes : List Gnome

/-- The kind wizard's action of making gnomes friends -/
def makeGnomesFriends (pairs : List (Gnome × Gnome)) : List Friendship :=
  sorry

/-- The evil wizard's action of making gnomes unfriends -/
def makeGnomesUnfriends (friendships : List Friendship) (n : Nat) : List Friendship :=
  sorry

/-- Check if a seating arrangement is valid (all adjacent gnomes are friends) -/
def isValidSeating (seating : List Gnome) (friendships : List Friendship) : Prop :=
  sorry

theorem kindWizardCanAchieveGoal (n : Nat) (hn : n > 1 ∧ Odd n) :
  ∃ (table1 table2 : RoundTable),
    table1.gnomes.length = n ∧
    table2.gnomes.length = n ∧
    (∀ (pairs : List (Gnome × Gnome)),
      pairs.length = 2 * n →
      ∀ (evilAction : List Friendship → List Friendship),
        ∃ (finalSeating : List Gnome),
          finalSeating.length = 2 * n ∧
          isValidSeating finalSeating (evilAction (makeGnomesFriends pairs))) :=
by sorry

end NUMINAMATH_CALUDE_kindWizardCanAchieveGoal_l2506_250675


namespace NUMINAMATH_CALUDE_ferris_wheel_line_l2506_250641

theorem ferris_wheel_line (capacity : ℕ) (not_riding : ℕ) (total : ℕ) : 
  capacity = 56 → not_riding = 36 → total = capacity + not_riding → total = 92 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_line_l2506_250641


namespace NUMINAMATH_CALUDE_vector_at_negative_seven_l2506_250666

/-- A parametric line in 2D space -/
structure ParametricLine where
  /-- The position vector at t = 0 -/
  a : ℝ × ℝ
  /-- The direction vector of the line -/
  d : ℝ × ℝ

/-- Get the vector on the line at a given t -/
def ParametricLine.vectorAt (line : ParametricLine) (t : ℝ) : ℝ × ℝ :=
  (line.a.1 + t * line.d.1, line.a.2 + t * line.d.2)

/-- The main theorem -/
theorem vector_at_negative_seven
  (line : ParametricLine)
  (h1 : line.vectorAt 2 = (1, 4))
  (h2 : line.vectorAt 3 = (3, -4)) :
  line.vectorAt (-7) = (-17, 76) := by
  sorry


end NUMINAMATH_CALUDE_vector_at_negative_seven_l2506_250666


namespace NUMINAMATH_CALUDE_book_selection_theorem_l2506_250637

theorem book_selection_theorem (n m k : ℕ) (h1 : n = 8) (h2 : m = 5) (h3 : k = 2) :
  (Nat.choose (n - k) (m - k)) = (Nat.choose 6 3) :=
sorry

end NUMINAMATH_CALUDE_book_selection_theorem_l2506_250637


namespace NUMINAMATH_CALUDE_vector_arrangement_exists_l2506_250632

theorem vector_arrangement_exists : ∃ (a b c : ℝ × ℝ),
  (‖a + b‖ = 1) ∧
  (‖b + c‖ = 1) ∧
  (‖c + a‖ = 1) ∧
  (a + b + c = (0, 0)) := by
  sorry

end NUMINAMATH_CALUDE_vector_arrangement_exists_l2506_250632


namespace NUMINAMATH_CALUDE_yard_to_stride_l2506_250665

-- Define the units of measurement
variable (step stride leap yard : ℚ)

-- Define the relationships between units
axiom step_stride_relation : 3 * step = 4 * stride
axiom leap_step_relation : 5 * leap = 2 * step
axiom leap_yard_relation : 7 * leap = 6 * yard

-- Theorem to prove
theorem yard_to_stride : yard = 28/45 * stride := by
  sorry

end NUMINAMATH_CALUDE_yard_to_stride_l2506_250665


namespace NUMINAMATH_CALUDE_inequality_of_reciprocal_logs_l2506_250609

theorem inequality_of_reciprocal_logs (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  1 / Real.log a > 1 / Real.log b :=
by sorry

end NUMINAMATH_CALUDE_inequality_of_reciprocal_logs_l2506_250609


namespace NUMINAMATH_CALUDE_leo_current_weight_l2506_250603

-- Define Leo's current weight
def leo_weight : ℝ := sorry

-- Define Kendra's current weight
def kendra_weight : ℝ := sorry

-- Condition 1: If Leo gains 10 pounds, he will weigh 50% more than Kendra
axiom condition_1 : leo_weight + 10 = 1.5 * kendra_weight

-- Condition 2: Their combined current weight is 150 pounds
axiom condition_2 : leo_weight + kendra_weight = 150

-- Theorem to prove
theorem leo_current_weight : leo_weight = 86 := by sorry

end NUMINAMATH_CALUDE_leo_current_weight_l2506_250603


namespace NUMINAMATH_CALUDE_count_valid_assignments_five_l2506_250690

/-- Represents a valid assignment of students to tests -/
def ValidAssignment (n : ℕ) := Fin n → Fin n → Prop

/-- The number of valid assignments for n students and n tests -/
def CountValidAssignments (n : ℕ) : ℕ := sorry

/-- The condition that each student takes exactly 2 distinct tests -/
def StudentTakesTwoTests (assignment : ValidAssignment 5) : Prop :=
  ∀ s : Fin 5, ∃! t1 t2 : Fin 5, t1 ≠ t2 ∧ assignment s t1 ∧ assignment s t2

/-- The condition that each test is taken by exactly 2 students -/
def TestTakenByTwoStudents (assignment : ValidAssignment 5) : Prop :=
  ∀ t : Fin 5, ∃! s1 s2 : Fin 5, s1 ≠ s2 ∧ assignment s1 t ∧ assignment s2 t

theorem count_valid_assignments_five :
  (∀ assignment : ValidAssignment 5,
    StudentTakesTwoTests assignment ∧ TestTakenByTwoStudents assignment) →
  CountValidAssignments 5 = 2040 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_assignments_five_l2506_250690


namespace NUMINAMATH_CALUDE_cubic_inequality_l2506_250600

theorem cubic_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 + b^3 + c^3 + 3*a*b*c ≥ a*b*(a + b) + b*c*(b + c) + c*a*(c + a) := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2506_250600


namespace NUMINAMATH_CALUDE_streaming_bill_fixed_fee_l2506_250694

/-- Represents the billing structure for a streaming service -/
structure StreamingBill where
  fixedFee : ℝ
  movieCharge : ℝ

/-- Calculates the total bill given number of movies watched -/
def StreamingBill.totalBill (bill : StreamingBill) (movies : ℝ) : ℝ :=
  bill.fixedFee + bill.movieCharge * movies

theorem streaming_bill_fixed_fee (bill : StreamingBill) :
  bill.totalBill 1 = 15.30 →
  bill.totalBill 1.5 = 20.55 →
  bill.fixedFee = 4.80 := by
  sorry

end NUMINAMATH_CALUDE_streaming_bill_fixed_fee_l2506_250694


namespace NUMINAMATH_CALUDE_min_value_squared_distance_l2506_250647

theorem min_value_squared_distance (a b c d : ℝ) 
  (h : |b - (Real.log a) / a| + |c - d + 2| = 0) : 
  ∃ (min : ℝ), min = (9 : ℝ) / 2 ∧ 
  ∀ (x y : ℝ), (x - y)^2 + (b - d)^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_squared_distance_l2506_250647


namespace NUMINAMATH_CALUDE_number_of_benches_l2506_250656

/-- Converts a base 6 number to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

/-- The number of people that can be seated in the shop -/
def totalSeats : ℕ := base6ToBase10 204

/-- The number of people that sit on one bench -/
def peoplePerBench : ℕ := 2

/-- Theorem: The number of benches in the shop is 38 -/
theorem number_of_benches :
  totalSeats / peoplePerBench = 38 := by
  sorry

end NUMINAMATH_CALUDE_number_of_benches_l2506_250656


namespace NUMINAMATH_CALUDE_goods_train_length_goods_train_length_approx_280m_l2506_250678

/-- The length of a goods train passing a passenger train in opposite directions --/
theorem goods_train_length (v_passenger : ℝ) (v_goods : ℝ) (t_pass : ℝ) : ℝ :=
  let v_relative : ℝ := v_passenger + v_goods
  let v_relative_ms : ℝ := v_relative * 1000 / 3600
  let length : ℝ := v_relative_ms * t_pass
  by
    -- Proof goes here
    sorry

/-- The length of the goods train is approximately 280 meters --/
theorem goods_train_length_approx_280m :
  ∃ ε > 0, |goods_train_length 70 42 9 - 280| < ε :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_goods_train_length_goods_train_length_approx_280m_l2506_250678


namespace NUMINAMATH_CALUDE_henrys_initial_income_l2506_250681

theorem henrys_initial_income (initial_income : ℝ) : 
  initial_income * 1.5 = 180 → initial_income = 120 := by
  sorry

end NUMINAMATH_CALUDE_henrys_initial_income_l2506_250681


namespace NUMINAMATH_CALUDE_bottles_remaining_l2506_250618

theorem bottles_remaining (cases : ℕ) (bottles_per_case : ℕ) (used_first_game : ℕ) (used_second_game : ℕ) :
  cases = 10 →
  bottles_per_case = 20 →
  used_first_game = 70 →
  used_second_game = 110 →
  cases * bottles_per_case - used_first_game - used_second_game = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_bottles_remaining_l2506_250618


namespace NUMINAMATH_CALUDE_min_scabs_per_day_l2506_250677

def total_scabs : ℕ := 220
def days_in_week : ℕ := 7

theorem min_scabs_per_day :
  ∃ (n : ℕ), n * days_in_week ≥ total_scabs ∧
  ∀ (m : ℕ), m * days_in_week ≥ total_scabs → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_min_scabs_per_day_l2506_250677


namespace NUMINAMATH_CALUDE_divisibility_rule_37_l2506_250645

/-- Given a positive integer n, returns a list of its three-digit segments from right to left -/
def threeDigitSegments (n : ℕ+) : List ℕ :=
  sorry

/-- The divisibility rule for 37 -/
theorem divisibility_rule_37 (n : ℕ+) : 
  37 ∣ n ↔ 37 ∣ (threeDigitSegments n).sum :=
sorry

end NUMINAMATH_CALUDE_divisibility_rule_37_l2506_250645


namespace NUMINAMATH_CALUDE_average_waiting_time_is_nineteen_sixths_l2506_250648

/-- Represents a bus schedule with a given departure interval -/
structure BusSchedule where
  interval : ℕ

/-- Calculates the average waiting time for a set of bus schedules -/
def averageWaitingTime (schedules : List BusSchedule) : ℚ :=
  sorry

/-- Theorem stating that the average waiting time for the given bus schedules is 19/6 minutes -/
theorem average_waiting_time_is_nineteen_sixths :
  let schedules := [
    BusSchedule.mk 10,  -- Bus A
    BusSchedule.mk 12,  -- Bus B
    BusSchedule.mk 15   -- Bus C
  ]
  averageWaitingTime schedules = 19 / 6 := by
  sorry

end NUMINAMATH_CALUDE_average_waiting_time_is_nineteen_sixths_l2506_250648


namespace NUMINAMATH_CALUDE_percentage_of_360_equals_108_l2506_250673

theorem percentage_of_360_equals_108 : 
  ∃ (p : ℝ), p * 360 / 100 = 108.0 ∧ p = 30 := by sorry

end NUMINAMATH_CALUDE_percentage_of_360_equals_108_l2506_250673


namespace NUMINAMATH_CALUDE_goldfish_disappeared_l2506_250659

theorem goldfish_disappeared (original : ℕ) (left : ℕ) (disappeared : ℕ) : 
  original = 15 → left = 4 → disappeared = original - left → disappeared = 11 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_disappeared_l2506_250659


namespace NUMINAMATH_CALUDE_fish_cost_l2506_250697

/-- Given that 530 pesos can buy 4 kg of fish and 2 kg of pork,
    and 875 pesos can buy 7 kg of fish and 3 kg of pork,
    prove that the cost of 1 kg of fish is 80 pesos. -/
theorem fish_cost (fish_price pork_price : ℝ) 
  (h1 : 4 * fish_price + 2 * pork_price = 530)
  (h2 : 7 * fish_price + 3 * pork_price = 875) : 
  fish_price = 80 := by
  sorry

end NUMINAMATH_CALUDE_fish_cost_l2506_250697


namespace NUMINAMATH_CALUDE_planes_parallel_l2506_250630

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (non_coincident : Line → Line → Prop)
variable (plane_non_coincident : Plane → Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel
  (l m : Line) (α β γ : Plane)
  (h1 : non_coincident l m)
  (h2 : plane_non_coincident α β γ)
  (h3 : perpendicular l α)
  (h4 : perpendicular m β)
  (h5 : parallel l m) :
  plane_parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_l2506_250630


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l2506_250672

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 5)) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l2506_250672


namespace NUMINAMATH_CALUDE_total_cats_l2506_250635

/-- The number of cats that can jump -/
def jump : ℕ := 60

/-- The number of cats that can fetch -/
def fetch : ℕ := 35

/-- The number of cats that can meow on command -/
def meow : ℕ := 40

/-- The number of cats that can jump and fetch -/
def jump_fetch : ℕ := 20

/-- The number of cats that can fetch and meow -/
def fetch_meow : ℕ := 15

/-- The number of cats that can jump and meow -/
def jump_meow : ℕ := 25

/-- The number of cats that can do all three tricks -/
def all_three : ℕ := 11

/-- The number of cats that can do none of the tricks -/
def no_tricks : ℕ := 10

/-- Theorem stating the total number of cats in the training center -/
theorem total_cats : 
  jump + fetch + meow - jump_fetch - fetch_meow - jump_meow + all_three + no_tricks = 96 := by
  sorry

end NUMINAMATH_CALUDE_total_cats_l2506_250635


namespace NUMINAMATH_CALUDE_sales_tax_difference_specific_sales_tax_difference_l2506_250691

/-- The difference between state and local sales taxes on a discounted sweater --/
theorem sales_tax_difference (original_price : ℝ) (discount_rate : ℝ) 
  (state_tax_rate : ℝ) (local_tax_rate : ℝ) : ℝ :=
by
  -- Define the discounted price
  let discounted_price := original_price * (1 - discount_rate)
  
  -- Calculate state and local taxes
  let state_tax := discounted_price * state_tax_rate
  let local_tax := discounted_price * local_tax_rate
  
  -- Calculate the difference
  exact state_tax - local_tax

/-- The specific case for the given problem --/
theorem specific_sales_tax_difference : 
  sales_tax_difference 50 0.1 0.075 0.07 = 0.225 :=
by
  sorry

end NUMINAMATH_CALUDE_sales_tax_difference_specific_sales_tax_difference_l2506_250691


namespace NUMINAMATH_CALUDE_parabola_symmetry_l2506_250631

/-- Represents a quadratic function of the form ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a quadratic function vertically by a given amount -/
def shift_vertical (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a, b := f.b, c := f.c + shift }

/-- Checks if two quadratic functions are symmetric about the y-axis -/
def symmetric_about_y_axis (f g : QuadraticFunction) : Prop :=
  f.a = g.a ∧ f.b = -g.b ∧ f.c = g.c

theorem parabola_symmetry (a b : ℝ) (h_a : a ≠ 0) :
  let f : QuadraticFunction := { a := a, b := b, c := -2 }
  let g : QuadraticFunction := { a := 1/2, b := 1, c := -4 }
  symmetric_about_y_axis (shift_vertical f (-2)) g →
  a = 1/2 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l2506_250631


namespace NUMINAMATH_CALUDE_power_multiplication_l2506_250679

theorem power_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2506_250679


namespace NUMINAMATH_CALUDE_number_difference_l2506_250674

theorem number_difference (L S : ℕ) (h1 : L = 1614) (h2 : L = 6 * S + 15) : L - S = 1348 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l2506_250674


namespace NUMINAMATH_CALUDE_book_arrangement_l2506_250657

theorem book_arrangement (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 3) :
  (Nat.factorial n) / ((Nat.factorial (n / k))^k * Nat.factorial k) =
  (Nat.factorial 30) / ((Nat.factorial 10)^3 * Nat.factorial 3) :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_l2506_250657


namespace NUMINAMATH_CALUDE_oscar_coco_difference_l2506_250642

/-- The number of strides Coco takes between consecutive poles -/
def coco_strides : ℕ := 22

/-- The number of leaps Oscar takes between consecutive poles -/
def oscar_leaps : ℕ := 6

/-- The number of strides Elmer takes between consecutive poles -/
def elmer_strides : ℕ := 11

/-- The number of poles -/
def num_poles : ℕ := 31

/-- The total distance in feet between the first and last pole -/
def total_distance : ℕ := 7920

/-- The length of Coco's stride in feet -/
def coco_stride_length : ℚ := total_distance / (coco_strides * (num_poles - 1))

/-- The length of Oscar's leap in feet -/
def oscar_leap_length : ℚ := total_distance / (oscar_leaps * (num_poles - 1))

theorem oscar_coco_difference :
  oscar_leap_length - coco_stride_length = 32 := by sorry

end NUMINAMATH_CALUDE_oscar_coco_difference_l2506_250642


namespace NUMINAMATH_CALUDE_max_value_f_monotonic_condition_inequality_condition_l2506_250661

noncomputable section

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := (-x^2 + 2*a*x) * Real.exp x
def g (x : ℝ) : ℝ := (x - 1) * Real.exp (2*x)

-- Theorem for part (I)
theorem max_value_f (a : ℝ) (h : a ≥ 0) :
  ∃ x : ℝ, x = a - 1 + Real.sqrt (a^2 + 1) ∨ x = a - 1 - Real.sqrt (a^2 + 1) ∧
  ∀ y : ℝ, f a y ≤ f a x :=
sorry

-- Theorem for part (II)
theorem monotonic_condition (a : ℝ) :
  (∀ x y : ℝ, -1 ≤ x ∧ x < y ∧ y ≤ 1 → f a x < f a y) ↔ a ≥ 3/4 :=
sorry

-- Theorem for part (III)
theorem inequality_condition (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x ≤ g x) ↔ 0 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_max_value_f_monotonic_condition_inequality_condition_l2506_250661


namespace NUMINAMATH_CALUDE_fidos_yard_l2506_250692

theorem fidos_yard (r : ℝ) (h : r > 0) : 
  let circle_area := π * r^2
  let hexagon_area := 3 * r^2 * Real.sqrt 3 / 2
  let ratio := circle_area / hexagon_area
  ratio = Real.sqrt 3 * π / 6 ∧ 3 * 6 = 18 := by sorry

end NUMINAMATH_CALUDE_fidos_yard_l2506_250692


namespace NUMINAMATH_CALUDE_sqrt_plus_reciprocal_inequality_l2506_250688

theorem sqrt_plus_reciprocal_inequality (x : ℝ) (h : x > 0) : Real.sqrt x + 1 / Real.sqrt x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_plus_reciprocal_inequality_l2506_250688


namespace NUMINAMATH_CALUDE_min_omega_value_l2506_250617

theorem min_omega_value (ω : ℕ+) : 
  (∀ k : ℕ+, 2 * Real.sin (2 * Real.pi * ↑k + Real.pi / 3) = Real.sqrt 3 → ω ≤ k) →
  2 * Real.sin (2 * Real.pi * ↑ω + Real.pi / 3) = Real.sqrt 3 →
  ω = 1 := by sorry

end NUMINAMATH_CALUDE_min_omega_value_l2506_250617


namespace NUMINAMATH_CALUDE_y₁_less_than_y₂_l2506_250687

/-- A linear function f(x) = -4x + 3 -/
def f (x : ℝ) : ℝ := -4 * x + 3

/-- Point P₁ is on the graph of f -/
def P₁_on_graph (y₁ : ℝ) : Prop := f 1 = y₁

/-- Point P₂ is on the graph of f -/
def P₂_on_graph (y₂ : ℝ) : Prop := f (-3) = y₂

/-- Theorem stating the relationship between y₁ and y₂ -/
theorem y₁_less_than_y₂ (y₁ y₂ : ℝ) (h₁ : P₁_on_graph y₁) (h₂ : P₂_on_graph y₂) : y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_less_than_y₂_l2506_250687


namespace NUMINAMATH_CALUDE_hyperbola_center_l2506_250626

/-- The equation of a hyperbola in general form -/
def HyperbolaEquation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 16 * y^2 + 128 * y - 400 = 0

/-- The center of a hyperbola -/
def HyperbolaCenter : ℝ × ℝ := (3, 4)

/-- Theorem: The center of the hyperbola defined by the given equation is (3, 4) -/
theorem hyperbola_center :
  ∀ (x y : ℝ), HyperbolaEquation x y →
  ∃ (a b : ℝ), (x - HyperbolaCenter.1)^2 / a^2 - (y - HyperbolaCenter.2)^2 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_center_l2506_250626


namespace NUMINAMATH_CALUDE_parallelogram_d_not_two_neg_two_l2506_250616

/-- Definition of a point in 2D space -/
def Point := ℝ × ℝ

/-- Definition of a parallelogram -/
def is_parallelogram (A B C D : Point) : Prop :=
  (A.1 + C.1 = B.1 + D.1) ∧ (A.2 + C.2 = B.2 + D.2)

/-- Theorem: If ABCD is a parallelogram with A(0,0), B(2,2), C(3,0), then D cannot be (2,-2) -/
theorem parallelogram_d_not_two_neg_two :
  let A : Point := (0, 0)
  let B : Point := (2, 2)
  let C : Point := (3, 0)
  let D : Point := (2, -2)
  ¬(is_parallelogram A B C D) := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_d_not_two_neg_two_l2506_250616


namespace NUMINAMATH_CALUDE_wage_increase_proof_l2506_250608

/-- The original daily wage of a worker -/
def original_wage : ℝ := 20

/-- The percentage increase in the worker's wage -/
def wage_increase_percent : ℝ := 40

/-- The new daily wage after the increase -/
def new_wage : ℝ := 28

/-- Theorem stating that the original wage increased by 40% equals the new wage -/
theorem wage_increase_proof : 
  original_wage * (1 + wage_increase_percent / 100) = new_wage := by
  sorry

end NUMINAMATH_CALUDE_wage_increase_proof_l2506_250608


namespace NUMINAMATH_CALUDE_trees_on_road_l2506_250613

/-- Calculates the number of trees that can be planted along a road -/
def numTrees (roadLength : ℕ) (interval : ℕ) : ℕ :=
  roadLength / interval + 1

/-- Theorem stating the number of trees that can be planted on a 100-meter road with 5-meter intervals -/
theorem trees_on_road :
  numTrees 100 5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_trees_on_road_l2506_250613


namespace NUMINAMATH_CALUDE_smallest_number_l2506_250633

-- Define the numbers in their respective bases
def binary_num : ℕ := 63  -- 111111₍₂₎
def base_6_num : ℕ := 66  -- 150₍₆₎
def base_4_num : ℕ := 64  -- 1000₍₄₎
def octal_num : ℕ := 65   -- 101₍₈₎

-- Theorem statement
theorem smallest_number :
  binary_num < base_6_num ∧ 
  binary_num < base_4_num ∧ 
  binary_num < octal_num :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l2506_250633


namespace NUMINAMATH_CALUDE_inconsistent_game_statistics_l2506_250680

theorem inconsistent_game_statistics :
  ∀ (total_games : ℕ) (first_part_games : ℕ) (win_percentage : ℚ),
  total_games = 75 →
  first_part_games = 100 →
  (0 : ℚ) ≤ win_percentage ∧ win_percentage ≤ 1 →
  ¬(∃ (first_part_win_percentage : ℚ) (remaining_win_percentage : ℚ),
    first_part_win_percentage * (first_part_games : ℚ) / (total_games : ℚ) +
    remaining_win_percentage * ((total_games - first_part_games) : ℚ) / (total_games : ℚ) = win_percentage ∧
    remaining_win_percentage = 1/2 ∧
    (0 : ℚ) ≤ first_part_win_percentage ∧ first_part_win_percentage ≤ 1) :=
by
  sorry


end NUMINAMATH_CALUDE_inconsistent_game_statistics_l2506_250680


namespace NUMINAMATH_CALUDE_linear_systems_solutions_l2506_250636

theorem linear_systems_solutions :
  -- System 1
  let system1 (x y : ℚ) := (y = x - 5) ∧ (3 * x - y = 8)
  let solution1 := (3/2, -7/2)
  -- System 2
  let system2 (x y : ℚ) := (3 * x - 2 * y = 1) ∧ (7 * x + 4 * y = 11)
  let solution2 := (1, 1)
  -- Proof statements
  (∃! p : ℚ × ℚ, system1 p.1 p.2 ∧ p = solution1) ∧
  (∃! q : ℚ × ℚ, system2 q.1 q.2 ∧ q = solution2) :=
by
  sorry

end NUMINAMATH_CALUDE_linear_systems_solutions_l2506_250636


namespace NUMINAMATH_CALUDE_remainder_polynomial_division_l2506_250649

theorem remainder_polynomial_division (x : ℝ) : 
  let g (x : ℝ) := x^5 + x^4 + x^3 + x^2 + x + 1
  (g (x^12)) % (g x) = 6 := by sorry

end NUMINAMATH_CALUDE_remainder_polynomial_division_l2506_250649


namespace NUMINAMATH_CALUDE_martha_savings_l2506_250664

/-- Martha's daily allowance in dollars -/
def daily_allowance : ℚ := 12

/-- Number of days in a week -/
def days_in_week : ℕ := 7

/-- Number of days Martha saved half her allowance -/
def days_half_saved : ℕ := days_in_week - 1

/-- Amount saved when Martha saves half her allowance -/
def half_savings : ℚ := daily_allowance / 2

/-- Amount saved when Martha saves a quarter of her allowance -/
def quarter_savings : ℚ := daily_allowance / 4

/-- Martha's total savings for the week -/
def total_savings : ℚ := days_half_saved * half_savings + quarter_savings

theorem martha_savings : total_savings = 39 := by
  sorry

end NUMINAMATH_CALUDE_martha_savings_l2506_250664


namespace NUMINAMATH_CALUDE_tank_capacity_l2506_250643

theorem tank_capacity : 
  ∀ (C : ℝ),
    (C / 6 + C / 12 = (2.5 * 60 + 1.5 * 60) * 8) →
    C = 640 :=
by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l2506_250643


namespace NUMINAMATH_CALUDE_equivalent_discount_equivalent_discount_proof_l2506_250621

theorem equivalent_discount : ℝ → Prop :=
  fun x => 
    let first_discount := 0.15
    let second_discount := 0.10
    let third_discount := 0.05
    let price_after_discounts := (1 - first_discount) * (1 - second_discount) * (1 - third_discount) * x
    let equivalent_single_discount := 0.273
    price_after_discounts = (1 - equivalent_single_discount) * x

-- The proof is omitted
theorem equivalent_discount_proof : ∀ x : ℝ, equivalent_discount x :=
  sorry

end NUMINAMATH_CALUDE_equivalent_discount_equivalent_discount_proof_l2506_250621


namespace NUMINAMATH_CALUDE_average_math_chem_score_l2506_250605

theorem average_math_chem_score (math physics chem : ℕ) : 
  math + physics = 60 → 
  chem = physics + 20 → 
  (math + chem) / 2 = 40 := by
sorry

end NUMINAMATH_CALUDE_average_math_chem_score_l2506_250605


namespace NUMINAMATH_CALUDE_min_distance_to_line_l2506_250612

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x + 2*y + 1 = 0

/-- The line equation -/
def line_equation (a b x y : ℝ) : Prop :=
  a*x + b*y + 1 = 0

/-- The line bisects the circle's circumference -/
def line_bisects_circle (a b : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y → line_equation a b x y

/-- The theorem to be proved -/
theorem min_distance_to_line (a b : ℝ) :
  line_bisects_circle a b →
  (∃ min : ℝ, min = 5 ∧ ∀ a' b' : ℝ, line_bisects_circle a' b' →
    (a' - 2)^2 + (b' - 2)^2 ≥ min) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l2506_250612


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2506_250699

theorem sum_of_fractions : 
  (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + 
  (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2506_250699


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l2506_250620

theorem greatest_of_three_consecutive_integers (x y z : ℤ) : 
  (y = x + 1) → (z = y + 1) → (x + y + z = 33) → (max x (max y z) = 12) := by
  sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l2506_250620


namespace NUMINAMATH_CALUDE_center_of_hyperbola_l2506_250652

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 36 * y^2 + 360 * y - 891 = 0

-- Define the center of a hyperbola
def is_center (c : ℝ × ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), ∀ (x y : ℝ),
    eq x y ↔ (x - c.1)^2 / a^2 - (y - c.2)^2 / b^2 = 1

-- Theorem stating that (3, 5) is the center of the given hyperbola
theorem center_of_hyperbola :
  is_center (3, 5) hyperbola_eq :=
sorry

end NUMINAMATH_CALUDE_center_of_hyperbola_l2506_250652


namespace NUMINAMATH_CALUDE_rectangle_area_l2506_250695

/-- The area of a rectangle with perimeter equal to a triangle with sides 7, 9, and 10,
    and length twice its width, is 338/9 square centimeters. -/
theorem rectangle_area (w : ℝ) (h : 2 * (2 * w + w) = 7 + 9 + 10) :
  2 * w * w = 338 / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2506_250695


namespace NUMINAMATH_CALUDE_jasmine_books_pages_l2506_250693

theorem jasmine_books_pages (books : Set ℕ) 
  (shortest longest middle : ℕ) 
  (h1 : shortest ∈ books) 
  (h2 : longest ∈ books) 
  (h3 : middle ∈ books)
  (h4 : shortest = longest / 4)
  (h5 : middle = 297)
  (h6 : middle = 3 * shortest)
  (h7 : ∀ b ∈ books, b ≤ longest) :
  longest = 396 := by
  sorry

end NUMINAMATH_CALUDE_jasmine_books_pages_l2506_250693
