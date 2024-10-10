import Mathlib

namespace triangle_triple_sine_sum_l2843_284330

theorem triangle_triple_sine_sum (A B C : ℝ) : 
  A + B + C = π ∧ (A = π/3 ∨ B = π/3 ∨ C = π/3) → 
  Real.sin (3*A) + Real.sin (3*B) + Real.sin (3*C) = 0 := by
  sorry

end triangle_triple_sine_sum_l2843_284330


namespace complex_purely_imaginary_l2843_284386

/-- If m^2(1+i) + (m+i)i^2 is purely imaginary and m is a real number, then m = 0 -/
theorem complex_purely_imaginary (m : ℝ) : 
  (Complex.I * (m^2 - 1) = m^2*(1 + Complex.I) + (m + Complex.I)*Complex.I^2) → m = 0 := by
  sorry

end complex_purely_imaginary_l2843_284386


namespace area_between_concentric_circles_l2843_284368

/-- Given two concentric circles where a chord of length 120 units is tangent to the smaller circle
    with radius 40 units, the area between the circles is 3600π square units. -/
theorem area_between_concentric_circles :
  ∀ (r R : ℝ) (chord_length : ℝ),
  r = 40 →
  chord_length = 120 →
  chord_length^2 = 4 * R * r →
  (R^2 - r^2) * π = 3600 * π :=
by sorry

end area_between_concentric_circles_l2843_284368


namespace complex_modulus_l2843_284320

theorem complex_modulus (z : ℂ) (h : (1 + Complex.I) * z = Complex.I - 1) : Complex.abs z = 1 := by
  sorry

end complex_modulus_l2843_284320


namespace A_div_B_between_zero_and_one_l2843_284343

def A : ℕ := 1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28
def B : ℕ := 1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20

theorem A_div_B_between_zero_and_one : 0 < (A : ℚ) / B ∧ (A : ℚ) / B < 1 := by
  sorry

end A_div_B_between_zero_and_one_l2843_284343


namespace smallest_total_students_l2843_284319

/-- The number of successful configuration days --/
def successful_days : ℕ := 14

/-- The maximum number of students per leader --/
def max_students_per_leader : ℕ := 12

/-- The number of students per leader on the first day --/
def first_day_students_per_leader : ℕ := 12

/-- The number of students per leader on the last successful day --/
def last_day_students_per_leader : ℕ := 5

/-- A function to check if a number satisfies all conditions --/
def satisfies_conditions (n : ℕ) : Prop :=
  (n % first_day_students_per_leader = 0) ∧
  (n % last_day_students_per_leader = 0) ∧
  (∃ (configs : Finset (Finset ℕ)), configs.card = successful_days ∧
    ∀ c ∈ configs, c.card > 0 ∧ c.card ≤ n ∧
    (∀ g ∈ c, g ≤ max_students_per_leader) ∧
    (n % c.card = 0))

theorem smallest_total_students :
  satisfies_conditions 360 ∧
  ∀ m < 360, ¬ satisfies_conditions m :=
sorry

end smallest_total_students_l2843_284319


namespace altitude_inradius_inequality_l2843_284310

-- Define a triangle with altitudes and inradius
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ
  r : ℝ
  h_a_positive : h_a > 0
  h_b_positive : h_b > 0
  h_c_positive : h_c > 0
  r_positive : r > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- State the theorem
theorem altitude_inradius_inequality (t : Triangle) : t.h_a + 4 * t.h_b + 9 * t.h_c > 36 * t.r := by
  sorry

end altitude_inradius_inequality_l2843_284310


namespace square_sum_zero_implies_both_zero_l2843_284339

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end square_sum_zero_implies_both_zero_l2843_284339


namespace irregular_polygon_rotation_implies_composite_l2843_284365

/-- An n-gon inscribed in a circle -/
structure InscribedPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  center : ℝ × ℝ
  radius : ℝ

/-- Rotation of a point about a center by an angle -/
def rotate (center : ℝ × ℝ) (angle : ℝ) (point : ℝ × ℝ) : ℝ × ℝ := sorry

/-- A polygon is irregular if not all its sides are equal -/
def isIrregular (p : InscribedPolygon n) : Prop := sorry

/-- A polygon coincides with itself under rotation -/
def coincidesSelfUnderRotation (p : InscribedPolygon n) (angle : ℝ) : Prop := sorry

/-- A number is composite if it's not prime and greater than 1 -/
def isComposite (n : ℕ) : Prop := ¬(Nat.Prime n) ∧ n > 1

theorem irregular_polygon_rotation_implies_composite 
  (n : ℕ) (p : InscribedPolygon n) (α : ℝ) :
  isIrregular p →
  α ≠ 2 * Real.pi →
  coincidesSelfUnderRotation p α →
  isComposite n := by
  sorry

end irregular_polygon_rotation_implies_composite_l2843_284365


namespace sets_intersection_and_complement_l2843_284394

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 * x^2 + a * x + 2 = 0}
def B (b : ℝ) : Set ℝ := {x | x^2 + 3 * x - b = 0}

-- State the theorem
theorem sets_intersection_and_complement (a b : ℝ) :
  (A a ∩ B b = {2}) →
  ∃ (U : Set ℝ),
    a = -5 ∧
    b = 10 ∧
    U = A a ∪ B b ∧
    (Uᶜ ∩ A a) ∪ (Uᶜ ∩ B b) = {-5, 1/2} := by
  sorry

end sets_intersection_and_complement_l2843_284394


namespace mutually_exclusive_pairs_count_l2843_284382

-- Define the type for balls
inductive Ball : Type
| Red : Ball
| White : Ball

-- Define the type for events
inductive Event : Type
| AtLeastOneWhite : Event
| BothWhite : Event
| AtLeastOneRed : Event
| ExactlyOneWhite : Event
| ExactlyTwoWhite : Event
| BothRed : Event

-- Define a function to check if two events are mutually exclusive
def mutually_exclusive (e1 e2 : Event) : Prop := sorry

-- Define the bag of balls
def bag : Multiset Ball := sorry

-- Define the function to count mutually exclusive pairs
def count_mutually_exclusive_pairs (events : List (Event × Event)) : Nat := sorry

-- Main theorem
theorem mutually_exclusive_pairs_count :
  let events : List (Event × Event) := [
    (Event.AtLeastOneWhite, Event.BothWhite),
    (Event.AtLeastOneWhite, Event.AtLeastOneRed),
    (Event.ExactlyOneWhite, Event.ExactlyTwoWhite),
    (Event.AtLeastOneWhite, Event.BothRed)
  ]
  count_mutually_exclusive_pairs events = 2 := by sorry

end mutually_exclusive_pairs_count_l2843_284382


namespace intersection_complement_l2843_284303

def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B : Set ℝ := {x | 2 < x}

theorem intersection_complement : A ∩ (Bᶜ) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end intersection_complement_l2843_284303


namespace smaller_number_l2843_284356

theorem smaller_number (u v : ℝ) (hu : u > 0) (hv : v > 0) 
  (h_ratio : u / v = 3 / 5) (h_sum : u + v = 16) : 
  min u v = 6 := by
sorry

end smaller_number_l2843_284356


namespace diane_has_27_cents_l2843_284367

/-- The amount of money Diane has, given the cost of cookies and the additional amount needed. -/
def dianes_money (cookie_cost : ℕ) (additional_needed : ℕ) : ℕ :=
  cookie_cost - additional_needed

/-- Theorem stating that Diane has 27 cents given the problem conditions. -/
theorem diane_has_27_cents :
  dianes_money 65 38 = 27 := by
  sorry

end diane_has_27_cents_l2843_284367


namespace standard_deviation_best_dispersion_measure_l2843_284381

-- Define the possible measures of central tendency and dispersion
inductive DataMeasure
  | Mode
  | Mean
  | StandardDeviation
  | Range

-- Define a function to determine if a measure reflects dispersion
def reflectsDispersion (measure : DataMeasure) : Prop :=
  match measure with
  | DataMeasure.StandardDeviation => true
  | _ => false

-- Theorem stating that standard deviation is the best measure of dispersion
theorem standard_deviation_best_dispersion_measure :
  ∀ (measure : DataMeasure),
    reflectsDispersion measure ↔ measure = DataMeasure.StandardDeviation :=
by sorry

end standard_deviation_best_dispersion_measure_l2843_284381


namespace square_difference_of_sums_l2843_284323

theorem square_difference_of_sums (a b : ℝ) :
  a = Real.sqrt 3 + Real.sqrt 2 →
  b = Real.sqrt 3 - Real.sqrt 2 →
  a^2 - b^2 = 4 * Real.sqrt 6 := by
  sorry

end square_difference_of_sums_l2843_284323


namespace power_division_rule_l2843_284325

theorem power_division_rule (x : ℝ) (h : x ≠ 0) : x^10 / x^5 = x^5 := by
  sorry

end power_division_rule_l2843_284325


namespace arithmetic_sequence_common_difference_l2843_284379

/-- Arithmetic sequence properties -/
structure ArithmeticSequence where
  first_term : ℝ
  last_term : ℝ
  sum : ℝ
  num_terms : ℕ

/-- Theorem: Common difference of a specific arithmetic sequence -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h1 : seq.first_term = 5)
  (h2 : seq.last_term = 45)
  (h3 : seq.sum = 250) :
  let d := (seq.last_term - seq.first_term) / (seq.num_terms - 1)
  d = 40 / 9 := by
  sorry

end arithmetic_sequence_common_difference_l2843_284379


namespace quadratic_function_unique_solution_l2843_284313

/-- Given a quadratic function f(x) = ax² + bx + c, prove that if f(-1) = 3, f(0) = 1, and f(1) = 1, then a = 1, b = -1, and c = 1. -/
theorem quadratic_function_unique_solution (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = (fun x => if x = -1 then 3 else if x = 0 then 1 else if x = 1 then 1 else 0) x) →
  a = 1 ∧ b = -1 ∧ c = 1 := by
sorry

end quadratic_function_unique_solution_l2843_284313


namespace same_side_theorem_l2843_284395

/-- The set of values for parameter a where points A and B lie on the same side of the line 2x - y = 5 -/
def same_side_values : Set ℝ :=
  {a : ℝ | a ∈ Set.Ioo (-5/2) (-1/2) ∪ Set.Ioo 0 3}

/-- The equation of point A in the plane -/
def point_A_equation (a x y : ℝ) : Prop :=
  5 * a^2 - 4 * a * y + 8 * x^2 - 4 * x * y + y^2 + 12 * a * x = 0

/-- The equation of the parabola with vertex B -/
def parabola_B_equation (a x y : ℝ) : Prop :=
  a * x^2 - 2 * a^2 * x - a * y + a^3 + 3 = 0

/-- The line equation 2x - y = 5 -/
def line_equation (x y : ℝ) : Prop :=
  2 * x - y = 5

theorem same_side_theorem (a : ℝ) :
  (∃ x y : ℝ, point_A_equation a x y) ∧
  (∃ x y : ℝ, parabola_B_equation a x y) ∧
  (∀ x y : ℝ, point_A_equation a x y → ¬line_equation x y) ∧
  (∀ x y : ℝ, parabola_B_equation a x y → ¬line_equation x y) →
  (a ∈ same_side_values ↔
    (∃ x₁ y₁ x₂ y₂ : ℝ,
      point_A_equation a x₁ y₁ ∧
      parabola_B_equation a x₂ y₂ ∧
      (2 * x₁ - y₁ - 5) * (2 * x₂ - y₂ - 5) > 0)) :=
sorry

end same_side_theorem_l2843_284395


namespace inequality_solution_set_l2843_284312

theorem inequality_solution_set (c : ℝ) : 
  (c / 3 ≤ 2 + c ∧ 2 + c < -2 * (1 + c)) ↔ c ∈ Set.Icc (-3) (-4/3) := by
  sorry

end inequality_solution_set_l2843_284312


namespace chicken_price_chicken_price_is_8_l2843_284371

/-- Calculates the price of a chicken given the conditions of the farmer's sales. -/
theorem chicken_price : ℝ → Prop :=
  fun price =>
    let duck_price := 10
    let num_ducks := 2
    let num_chickens := 5
    let total_earnings := duck_price * num_ducks + price * num_chickens
    let wheelbarrow_cost := total_earnings / 2
    let wheelbarrow_sale := wheelbarrow_cost * 2
    let additional_earnings := 60
    wheelbarrow_sale - wheelbarrow_cost = additional_earnings →
    price = 8

/-- The price of a chicken is $8. -/
theorem chicken_price_is_8 : chicken_price 8 := by
  sorry

end chicken_price_chicken_price_is_8_l2843_284371


namespace road_length_difference_l2843_284337

/-- The length of Telegraph Road in kilometers -/
def telegraph_road_length : ℝ := 162

/-- The length of Pardee Road in meters -/
def pardee_road_length : ℝ := 12000

/-- Conversion factor from kilometers to meters -/
def km_to_m : ℝ := 1000

theorem road_length_difference :
  (telegraph_road_length * km_to_m - pardee_road_length) / km_to_m = 150 := by
  sorry

end road_length_difference_l2843_284337


namespace num_routes_eq_binomial_num_routes_is_six_l2843_284331

/-- The number of different routes from the bottom-left corner to the top-right corner of a 2x2 grid,
    moving only upwards or to the right one square at a time. -/
def num_routes : ℕ := 6

/-- The size of the grid (2x2 in this case) -/
def grid_size : ℕ := 2

/-- The total number of moves required to reach the top-right corner from the bottom-left corner -/
def total_moves : ℕ := grid_size * 2

/-- Theorem stating that the number of routes is equal to the binomial coefficient (total_moves choose grid_size) -/
theorem num_routes_eq_binomial :
  num_routes = Nat.choose total_moves grid_size :=
by sorry

/-- Theorem proving that the number of routes is 6 -/
theorem num_routes_is_six :
  num_routes = 6 :=
by sorry

end num_routes_eq_binomial_num_routes_is_six_l2843_284331


namespace angle_measure_proof_l2843_284328

/-- Given two supplementary angles C and D, where the measure of angle C is 5 times
    the measure of angle D, prove that the measure of angle C is 150°. -/
theorem angle_measure_proof (C D : ℝ) : 
  C + D = 180 →  -- Angles C and D are supplementary
  C = 5 * D →    -- Measure of angle C is 5 times angle D
  C = 150 := by  -- Measure of angle C is 150°
  sorry

end angle_measure_proof_l2843_284328


namespace rod_length_l2843_284317

/-- Given a rod from which 40 pieces of 85 cm each can be cut, prove that its length is 3400 cm. -/
theorem rod_length (num_pieces : ℕ) (piece_length : ℕ) (h1 : num_pieces = 40) (h2 : piece_length = 85) :
  num_pieces * piece_length = 3400 := by
  sorry

end rod_length_l2843_284317


namespace sarah_test_performance_l2843_284349

theorem sarah_test_performance :
  let test1_questions : ℕ := 30
  let test2_questions : ℕ := 20
  let test3_questions : ℕ := 50
  let test1_correct_rate : ℚ := 85 / 100
  let test2_correct_rate : ℚ := 75 / 100
  let test3_correct_rate : ℚ := 90 / 100
  let calculation_mistakes : ℕ := 3
  let total_questions := test1_questions + test2_questions + test3_questions
  let correct_before_mistakes := 
    (test1_correct_rate * test1_questions).ceil +
    (test2_correct_rate * test2_questions).floor +
    (test3_correct_rate * test3_questions).floor
  let correct_after_mistakes := correct_before_mistakes - calculation_mistakes
  (correct_after_mistakes : ℚ) / total_questions = 83 / 100 :=
by sorry

end sarah_test_performance_l2843_284349


namespace smallest_special_number_l2843_284354

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem smallest_special_number :
  ∀ n : ℕ,
    is_two_digit n →
    n % 6 = 0 →
    n % 3 = 0 →
    is_perfect_square (digit_product n) →
    30 ≤ n :=
by sorry

end smallest_special_number_l2843_284354


namespace sum_ratio_simplification_main_result_l2843_284301

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => n * double_factorial n

def sum_ratio (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => (double_factorial (2*i+1)) / (double_factorial (2*i+2)))

theorem sum_ratio_simplification (n : ℕ) :
  ∃ (c : ℕ), Odd c ∧ sum_ratio n = c / 2^(2*n - 7) := by sorry

theorem main_result :
  ∃ (c : ℕ), Odd c ∧ sum_ratio 2010 = c / 2^4013 ∧ 4013 / 10 = 401.3 := by sorry

end sum_ratio_simplification_main_result_l2843_284301


namespace crosswalk_lines_total_l2843_284374

theorem crosswalk_lines_total (num_intersections : ℕ) (crosswalks_per_intersection : ℕ) (lines_per_crosswalk : ℕ) : 
  num_intersections = 5 → 
  crosswalks_per_intersection = 4 → 
  lines_per_crosswalk = 20 → 
  num_intersections * crosswalks_per_intersection * lines_per_crosswalk = 400 :=
by sorry

end crosswalk_lines_total_l2843_284374


namespace problem_1_problem_2_l2843_284333

-- Problem 1
theorem problem_1 (x y z : ℝ) :
  2 * x^3 * y^2 * (-2 * x * y^2 * z)^2 = 8 * x^5 * y^6 * z^2 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) :
  (-2 * x^2)^3 + x^2 * x^4 - (-3 * x^3)^2 = -16 * x^6 := by
  sorry

end problem_1_problem_2_l2843_284333


namespace mysterious_number_properties_l2843_284388

/-- A positive integer that can be expressed as the difference of the squares of two consecutive even numbers. -/
def MysteriousNumber (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2*k + 2)^2 - (2*k)^2 ∧ k ≥ 0

theorem mysterious_number_properties :
  (MysteriousNumber 28 ∧ MysteriousNumber 2020) ∧
  (∀ k : ℕ, (2*k + 2)^2 - (2*k)^2 % 4 = 0) ∧
  (∀ k : ℕ, ¬MysteriousNumber ((2*k + 1)^2 - (2*k - 1)^2)) :=
by sorry

end mysterious_number_properties_l2843_284388


namespace trees_along_path_l2843_284316

/-- Calculates the total number of trees that can be planted along a path -/
def totalTrees (pathLength : ℕ) (treeSpacing : ℕ) : ℕ :=
  2 * (pathLength / treeSpacing + 1)

/-- Theorem: Given a path of 80 meters with trees planted on both sides every 4 meters,
    including at both ends, the total number of trees that can be planted is 42. -/
theorem trees_along_path :
  totalTrees 80 4 = 42 := by
  sorry

end trees_along_path_l2843_284316


namespace point_in_second_quadrant_l2843_284315

-- Define a point in 2D space
def point : ℝ × ℝ := (-8, 2)

-- Define the second quadrant
def second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- Theorem statement
theorem point_in_second_quadrant :
  second_quadrant point := by
  sorry

end point_in_second_quadrant_l2843_284315


namespace water_polo_team_selection_l2843_284347

/-- The number of members in the water polo club -/
def total_members : ℕ := 18

/-- The number of players in the starting team -/
def team_size : ℕ := 8

/-- The number of field players (excluding captain and goalie) -/
def field_players : ℕ := 6

/-- Calculates the number of ways to choose the starting team -/
def choose_team : ℕ := total_members * (total_members - 1) * (Nat.choose (total_members - 2) field_players)

theorem water_polo_team_selection :
  choose_team = 2459528 :=
sorry

end water_polo_team_selection_l2843_284347


namespace parallel_vectors_x_value_l2843_284369

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (4, x) (-4, 4) → x = -4 := by
  sorry

end parallel_vectors_x_value_l2843_284369


namespace inequality_proof_l2843_284390

theorem inequality_proof : (1/2: ℝ)^(2/3) < (1/2: ℝ)^(1/3) ∧ (1/2: ℝ)^(1/3) < 1 := by
  sorry

end inequality_proof_l2843_284390


namespace power_division_equality_l2843_284358

theorem power_division_equality : (4 ^ (3^2)) / ((4^3)^2) = 64 := by sorry

end power_division_equality_l2843_284358


namespace cylinder_water_transfer_l2843_284389

theorem cylinder_water_transfer (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let original_volume := π * r^2 * h
  let new_volume := π * (1.25 * r)^2 * (0.72 * h)
  (3/5) * new_volume = 0.675 * original_volume :=
by sorry

end cylinder_water_transfer_l2843_284389


namespace tensor_equation_solution_l2843_284363

/-- Custom binary operation ⊗ for positive real numbers -/
def tensor (a b : ℝ) : ℝ := a * b + a + b^2

/-- Theorem stating that if 1⊗m = 3, then m = 1 -/
theorem tensor_equation_solution (m : ℝ) (h1 : m > 0) (h2 : tensor 1 m = 3) : m = 1 := by
  sorry

end tensor_equation_solution_l2843_284363


namespace backpack_profit_theorem_l2843_284338

/-- Represents the profit equation for a backpack sale -/
def profit_equation (x : ℝ) : Prop :=
  (1 + 0.5) * x * 0.8 - x = 8

/-- Theorem stating the profit equation holds for a backpack sale with given conditions -/
theorem backpack_profit_theorem (x : ℝ) 
  (h_markup : ℝ → ℝ := λ price => (1 + 0.5) * price)
  (h_discount : ℝ → ℝ := λ price => 0.8 * price)
  (h_profit : ℝ := 8) :
  profit_equation x := by
  sorry

end backpack_profit_theorem_l2843_284338


namespace unique_consecutive_set_sum_20_l2843_284384

/-- A set of consecutive positive integers -/
structure ConsecutiveSet where
  start : ℕ
  length : ℕ
  sum : ℕ
  h1 : start ≥ 2
  h2 : length ≥ 2
  h3 : sum = (length * (2 * start + length - 1)) / 2

/-- The theorem stating that there is exactly one set of consecutive positive integers
    starting from 2 or higher, with at least two numbers, whose sum is 20 -/
theorem unique_consecutive_set_sum_20 :
  ∃! (s : ConsecutiveSet), s.sum = 20 :=
sorry

end unique_consecutive_set_sum_20_l2843_284384


namespace equivalence_theorem_l2843_284366

theorem equivalence_theorem (x y z : ℝ) : 
  (x > 0 ∧ y > 0 ∧ z > 0 ∧ 1/x + 1/y + 1/z ≤ 1) ↔ 
  (∀ (a b c d : ℝ), (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a + b + c > d) → a^2*x + b^2*y + c^2*z > d^2) :=
by sorry

end equivalence_theorem_l2843_284366


namespace sum_of_reciprocals_positive_l2843_284342

theorem sum_of_reciprocals_positive (a b c d : ℝ) 
  (ha : |a| > 1) (hb : |b| > 1) (hc : |c| > 1) (hd : |d| > 1)
  (h_eq : a * b * c + a * b * d + a * c * d + b * c * d + a + b + c + d = 0) :
  1 / (a - 1) + 1 / (b - 1) + 1 / (c - 1) + 1 / (d - 1) > 0 := by
  sorry

end sum_of_reciprocals_positive_l2843_284342


namespace principal_calculation_l2843_284397

/-- Calculates the principal given simple interest, rate, and time -/
def calculate_principal (simple_interest : ℚ) (rate : ℚ) (time : ℕ) : ℚ :=
  simple_interest * 100 / (rate * time)

/-- Proves that the given conditions result in the correct principal -/
theorem principal_calculation :
  let simple_interest : ℚ := 4016.25
  let rate : ℚ := 9
  let time : ℕ := 5
  calculate_principal simple_interest rate time = 8925 := by
  sorry

end principal_calculation_l2843_284397


namespace phone_answer_probability_l2843_284321

theorem phone_answer_probability (p1 p2 p3 p4 : ℚ) 
  (h1 : p1 = 1/10)
  (h2 : p2 = 1/5)
  (h3 : p3 = 3/10)
  (h4 : p4 = 1/10) :
  p1 + p2 + p3 + p4 = 7/10 := by
  sorry

#check phone_answer_probability

end phone_answer_probability_l2843_284321


namespace min_perimeter_52_l2843_284326

/-- Represents the side lengths of the squares in the rectangle --/
structure SquareSides where
  a : ℕ
  b : ℕ

/-- Calculates the perimeter of a rectangle given its length and width --/
def rectangle_perimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- Represents the configuration of squares in the rectangle --/
def square_configuration (sides : SquareSides) : Prop :=
  ∃ (left_column middle_column right_column bottom_row : ℕ),
    left_column = 2 * sides.a + sides.b ∧
    middle_column = 3 * sides.a + sides.b ∧
    right_column = 12 * sides.a - 2 * sides.b ∧
    bottom_row = 8 * sides.a - sides.b ∧
    left_column > 0 ∧ middle_column > 0 ∧ right_column > 0 ∧ bottom_row > 0

theorem min_perimeter_52 :
  ∀ (sides : SquareSides),
    square_configuration sides →
    ∀ (length width : ℕ),
      length = 2 * sides.a + sides.b + 3 * sides.a + sides.b + 12 * sides.a - 2 * sides.b →
      width = 2 * sides.a + sides.b + 8 * sides.a - sides.b →
      rectangle_perimeter length width ≥ 52 :=
sorry

end min_perimeter_52_l2843_284326


namespace division_multiplication_equality_l2843_284357

theorem division_multiplication_equality : (0.45 / 0.005) * 0.1 = 9 := by
  sorry

end division_multiplication_equality_l2843_284357


namespace ellipse_hyperbola_eccentricity_l2843_284334

theorem ellipse_hyperbola_eccentricity (m n : ℝ) (e₁ e₂ : ℝ) : 
  m > 1 → 
  n > 0 → 
  (∀ x y : ℝ, x^2 / m^2 + y^2 = 1 ↔ x^2 / n^2 - y^2 = 1) → 
  e₁ = Real.sqrt (1 - 1 / m^2) → 
  e₂ = Real.sqrt (1 + 1 / n^2) → 
  m > n ∧ e₁ * e₂ > 1 := by
  sorry

end ellipse_hyperbola_eccentricity_l2843_284334


namespace june_earnings_l2843_284396

/-- Represents the number of clovers June picks -/
def total_clovers : ℕ := 200

/-- Represents the percentage of clovers with 3 petals -/
def three_petal_percentage : ℚ := 75 / 100

/-- Represents the percentage of clovers with 2 petals -/
def two_petal_percentage : ℚ := 24 / 100

/-- Represents the percentage of clovers with 4 petals -/
def four_petal_percentage : ℚ := 1 / 100

/-- Represents the payment in cents for each clover -/
def payment_per_clover : ℕ := 1

/-- Theorem stating that June earns 200 cents -/
theorem june_earnings : 
  (total_clovers * payment_per_clover : ℕ) = 200 := by
  sorry

end june_earnings_l2843_284396


namespace A_investment_l2843_284360

-- Define the investments and profit shares
def investment_B : ℝ := 10000
def investment_C : ℝ := 12000
def profit_share_B : ℝ := 2500
def profit_difference_AC : ℝ := 999.9999999999998

-- Define the theorem
theorem A_investment (investment_A : ℝ) : 
  (investment_A / investment_B * profit_share_B) - 
  (investment_C / investment_B * profit_share_B) = profit_difference_AC → 
  investment_A = 16000 := by
sorry

end A_investment_l2843_284360


namespace cubic_function_properties_l2843_284302

/-- The cubic function f(x) = x³ - kx + k² -/
def f (k : ℝ) (x : ℝ) : ℝ := x^3 - k*x + k^2

theorem cubic_function_properties (k : ℝ) :
  (∀ x y, x < y → f k x < f k y) ∨ 
  ((∃ x y z, x < y ∧ y < z ∧ f k x = 0 ∧ f k y = 0 ∧ f k z = 0) ↔ 0 < k ∧ k < 4/27) :=
sorry

end cubic_function_properties_l2843_284302


namespace travel_agency_problem_l2843_284376

/-- Represents the travel agency problem --/
theorem travel_agency_problem 
  (seats_per_bus : ℕ) 
  (incomplete_bus_2005 : ℕ) 
  (increase_2006 : ℕ) 
  (h1 : seats_per_bus = 27)
  (h2 : incomplete_bus_2005 = 19)
  (h3 : increase_2006 = 53) :
  ∃ (k : ℕ),
    (seats_per_bus * k + incomplete_bus_2005 + increase_2006) / seats_per_bus - 
    (seats_per_bus * k + incomplete_bus_2005) / seats_per_bus = 2 ∧
    (seats_per_bus * k + incomplete_bus_2005 + increase_2006) % seats_per_bus = 9 :=
by sorry

end travel_agency_problem_l2843_284376


namespace ball_return_theorem_l2843_284336

/-- The number of ways a ball returns to the initial person after n passes among m people. -/
def ball_return_ways (m n : ℕ) : ℚ :=
  ((m - 1)^n : ℚ) / m + ((-1)^n : ℚ) * ((m - 1) : ℚ) / m

/-- Theorem: The number of ways a ball returns to the initial person after n passes among m people,
    where m ≥ 2, is given by ((m-1)^n / m) + ((-1)^n * (m-1) / m) -/
theorem ball_return_theorem (m n : ℕ) (h : m ≥ 2) :
  ∃ (a_n : ℕ → ℚ),
    (∀ k, a_n k = ball_return_ways m k) ∧
    (∀ k, a_n k ≥ 0) ∧
    (a_n 0 = 0) ∧
    (a_n 1 = 1) :=
  sorry


end ball_return_theorem_l2843_284336


namespace binary_1010_equals_decimal_10_l2843_284341

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.zip b (List.reverse (List.range b.length))).foldl
    (fun acc (digit, power) => acc + if digit then 2^power else 0) 0

theorem binary_1010_equals_decimal_10 :
  binary_to_decimal [true, false, true, false] = 10 := by
  sorry

end binary_1010_equals_decimal_10_l2843_284341


namespace grid_puzzle_solution_l2843_284311

-- Define the type for our grid cells
def Cell := Fin 16

-- Define our grid
structure Grid :=
  (A B C D E F G H J K L M N P Q R : Cell)

-- Define the conditions
def conditions (g : Grid) : Prop :=
  (g.A.val + g.C.val + g.F.val = 10) ∧
  (g.B.val + g.H.val = g.R.val) ∧
  (g.D.val - g.C.val = 13) ∧
  (g.E.val * g.M.val = 126) ∧
  (g.F.val + g.G.val = 21) ∧
  (g.G.val / g.J.val = 2) ∧
  (g.H.val * g.M.val = 36) ∧
  (g.J.val * g.P.val = 80) ∧
  (g.K.val - g.N.val = g.Q.val) ∧
  (∀ i j : Fin 16, i ≠ j → 
    g.A.val ≠ g.B.val ∧ g.A.val ≠ g.C.val ∧ g.A.val ≠ g.D.val ∧
    g.A.val ≠ g.E.val ∧ g.A.val ≠ g.F.val ∧ g.A.val ≠ g.G.val ∧
    g.A.val ≠ g.H.val ∧ g.A.val ≠ g.J.val ∧ g.A.val ≠ g.K.val ∧
    g.A.val ≠ g.L.val ∧ g.A.val ≠ g.M.val ∧ g.A.val ≠ g.N.val ∧
    g.A.val ≠ g.P.val ∧ g.A.val ≠ g.Q.val ∧ g.A.val ≠ g.R.val ∧
    g.B.val ≠ g.C.val ∧ g.B.val ≠ g.D.val ∧ g.B.val ≠ g.E.val ∧
    g.B.val ≠ g.F.val ∧ g.B.val ≠ g.G.val ∧ g.B.val ≠ g.H.val ∧
    g.B.val ≠ g.J.val ∧ g.B.val ≠ g.K.val ∧ g.B.val ≠ g.L.val ∧
    g.B.val ≠ g.M.val ∧ g.B.val ≠ g.N.val ∧ g.B.val ≠ g.P.val ∧
    g.B.val ≠ g.Q.val ∧ g.B.val ≠ g.R.val ∧ g.C.val ≠ g.D.val ∧
    g.C.val ≠ g.E.val ∧ g.C.val ≠ g.F.val ∧ g.C.val ≠ g.G.val ∧
    g.C.val ≠ g.H.val ∧ g.C.val ≠ g.J.val ∧ g.C.val ≠ g.K.val ∧
    g.C.val ≠ g.L.val ∧ g.C.val ≠ g.M.val ∧ g.C.val ≠ g.N.val ∧
    g.C.val ≠ g.P.val ∧ g.C.val ≠ g.Q.val ∧ g.C.val ≠ g.R.val ∧
    g.D.val ≠ g.E.val ∧ g.D.val ≠ g.F.val ∧ g.D.val ≠ g.G.val ∧
    g.D.val ≠ g.H.val ∧ g.D.val ≠ g.J.val ∧ g.D.val ≠ g.K.val ∧
    g.D.val ≠ g.L.val ∧ g.D.val ≠ g.M.val ∧ g.D.val ≠ g.N.val ∧
    g.D.val ≠ g.P.val ∧ g.D.val ≠ g.Q.val ∧ g.D.val ≠ g.R.val ∧
    g.E.val ≠ g.F.val ∧ g.E.val ≠ g.G.val ∧ g.E.val ≠ g.H.val ∧
    g.E.val ≠ g.J.val ∧ g.E.val ≠ g.K.val ∧ g.E.val ≠ g.L.val ∧
    g.E.val ≠ g.M.val ∧ g.E.val ≠ g.N.val ∧ g.E.val ≠ g.P.val ∧
    g.E.val ≠ g.Q.val ∧ g.E.val ≠ g.R.val ∧ g.F.val ≠ g.G.val ∧
    g.F.val ≠ g.H.val ∧ g.F.val ≠ g.J.val ∧ g.F.val ≠ g.K.val ∧
    g.F.val ≠ g.L.val ∧ g.F.val ≠ g.M.val ∧ g.F.val ≠ g.N.val ∧
    g.F.val ≠ g.P.val ∧ g.F.val ≠ g.Q.val ∧ g.F.val ≠ g.R.val ∧
    g.G.val ≠ g.H.val ∧ g.G.val ≠ g.J.val ∧ g.G.val ≠ g.K.val ∧
    g.G.val ≠ g.L.val ∧ g.G.val ≠ g.M.val ∧ g.G.val ≠ g.N.val ∧
    g.G.val ≠ g.P.val ∧ g.G.val ≠ g.Q.val ∧ g.G.val ≠ g.R.val ∧
    g.H.val ≠ g.J.val ∧ g.H.val ≠ g.K.val ∧ g.H.val ≠ g.L.val ∧
    g.H.val ≠ g.M.val ∧ g.H.val ≠ g.N.val ∧ g.H.val ≠ g.P.val ∧
    g.H.val ≠ g.Q.val ∧ g.H.val ≠ g.R.val ∧ g.J.val ≠ g.K.val ∧
    g.J.val ≠ g.L.val ∧ g.J.val ≠ g.M.val ∧ g.J.val ≠ g.N.val ∧
    g.J.val ≠ g.P.val ∧ g.J.val ≠ g.Q.val ∧ g.J.val ≠ g.R.val ∧
    g.K.val ≠ g.L.val ∧ g.K.val ≠ g.M.val ∧ g.K.val ≠ g.N.val ∧
    g.K.val ≠ g.P.val ∧ g.K.val ≠ g.Q.val ∧ g.K.val ≠ g.R.val ∧
    g.L.val ≠ g.M.val ∧ g.L.val ≠ g.N.val ∧ g.L.val ≠ g.P.val ∧
    g.L.val ≠ g.Q.val ∧ g.L.val ≠ g.R.val ∧ g.M.val ≠ g.N.val ∧
    g.M.val ≠ g.P.val ∧ g.M.val ≠ g.Q.val ∧ g.M.val ≠ g.R.val ∧
    g.N.val ≠ g.P.val ∧ g.N.val ≠ g.Q.val ∧ g.N.val ≠ g.R.val ∧
    g.P.val ≠ g.Q.val ∧ g.P.val ≠ g.R.val ∧ g.Q.val ≠ g.R.val)

-- State the theorem
theorem grid_puzzle_solution (g : Grid) (h : conditions g) : g.L.val = 6 := by
  sorry

end grid_puzzle_solution_l2843_284311


namespace line_equations_l2843_284385

-- Define the lines
def l₁ (x y : ℝ) : Prop := 2 * x - 3 * y + 4 = 0
def l₂ (x y : ℝ) : Prop := x + y - 3 = 0
def l₃ (x y : ℝ) : Prop := x - 2 * y + 5 = 0

-- Define the intersection point M
def M : ℝ × ℝ := (1, 2)

-- Define the perpendicular line l
def l (x y : ℝ) : Prop := 3 * x + 2 * y - 7 = 0

-- Define the parallel line l'
def l' (x y : ℝ) : Prop := x - 2 * y + 3 = 0

theorem line_equations :
  (∀ x y : ℝ, l₁ x y ∧ l₂ x y → (x, y) = M) →
  (∀ x y : ℝ, l x y ↔ (3 * x + 2 * y - 7 = 0 ∧ (x, y) = M ∨ l₁ x y)) →
  (∀ x y : ℝ, l' x y ↔ (x - 2 * y + 3 = 0 ∧ (x, y) = M ∨ l₃ x y)) :=
by sorry

end line_equations_l2843_284385


namespace tan_sum_of_roots_l2843_284383

theorem tan_sum_of_roots (α β : Real) : 
  (∃ (x : Real), x^2 - 3 * Real.sqrt 3 * x + 4 = 0 ∧ x = Real.tan α) ∧
  (∃ (y : Real), y^2 - 3 * Real.sqrt 3 * y + 4 = 0 ∧ y = Real.tan β) ∧
  α ∈ Set.Ioo (-π/2) (π/2) ∧
  β ∈ Set.Ioo (-π/2) (π/2) →
  Real.tan (α + β) = -Real.sqrt 3 := by
sorry

end tan_sum_of_roots_l2843_284383


namespace sum_of_four_numbers_is_zero_l2843_284322

theorem sum_of_four_numbers_is_zero 
  (x y s t : ℝ) 
  (h_distinct : x ≠ y ∧ x ≠ s ∧ x ≠ t ∧ y ≠ s ∧ y ≠ t ∧ s ≠ t) 
  (h_equality : (x + s) / (x + t) = (y + t) / (y + s)) : 
  x + y + s + t = 0 := by
sorry

end sum_of_four_numbers_is_zero_l2843_284322


namespace no_three_digit_even_with_digit_sum_27_l2843_284329

/-- A function that returns the digit sum of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is a 3-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A theorem stating that there are no 3-digit even numbers with a digit sum of 27 -/
theorem no_three_digit_even_with_digit_sum_27 :
  ¬ ∃ n : ℕ, is_three_digit n ∧ Even n ∧ digit_sum n = 27 := by sorry

end no_three_digit_even_with_digit_sum_27_l2843_284329


namespace inscribed_circle_segment_lengths_l2843_284387

/-- Given a triangle with sides a, b, c and an inscribed circle, 
    the lengths of the segments into which the points of tangency divide the sides 
    are (a + b - c)/2, (a + c - b)/2, and (b + c - a)/2. -/
theorem inscribed_circle_segment_lengths 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  ∃ (x y z : ℝ),
    x = (a + b - c) / 2 ∧
    y = (a + c - b) / 2 ∧
    z = (b + c - a) / 2 ∧
    x + y = a ∧
    x + z = b ∧
    y + z = c :=
by sorry


end inscribed_circle_segment_lengths_l2843_284387


namespace hari_contribution_is_9000_l2843_284351

/-- Represents the business partnership between Praveen and Hari -/
structure Partnership where
  praveen_investment : ℕ
  praveen_months : ℕ
  hari_months : ℕ
  profit_ratio_praveen : ℕ
  profit_ratio_hari : ℕ

/-- Calculates Hari's contribution to the capital -/
def hari_contribution (p : Partnership) : ℕ :=
  (p.praveen_investment * p.praveen_months * p.profit_ratio_hari) / (p.hari_months * p.profit_ratio_praveen)

/-- Theorem stating that Hari's contribution is 9000 given the specified conditions -/
theorem hari_contribution_is_9000 :
  let p : Partnership := {
    praveen_investment := 3500,
    praveen_months := 12,
    hari_months := 7,
    profit_ratio_praveen := 2,
    profit_ratio_hari := 3
  }
  hari_contribution p = 9000 := by sorry

end hari_contribution_is_9000_l2843_284351


namespace sum_product_equality_l2843_284359

theorem sum_product_equality : 1235 + 2346 + 3412 * 2 + 4124 = 15529 := by
  sorry

end sum_product_equality_l2843_284359


namespace impossibility_of_simultaneous_inequalities_l2843_284355

theorem impossibility_of_simultaneous_inequalities (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) :
  ¬(a * (1 - b) > 1/4 ∧ b * (1 - c) > 1/4 ∧ c * (1 - a) > 1/4) := by
sorry

end impossibility_of_simultaneous_inequalities_l2843_284355


namespace sum_of_solutions_is_zero_l2843_284362

theorem sum_of_solutions_is_zero : 
  ∃ (x₁ x₂ : ℝ), (6 * x₁) / 30 = 8 / x₁ ∧ 
                 (6 * x₂) / 30 = 8 / x₂ ∧ 
                 x₁ + x₂ = 0 ∧
                 ∀ (y : ℝ), (6 * y) / 30 = 8 / y → y = x₁ ∨ y = x₂ := by
  sorry

end sum_of_solutions_is_zero_l2843_284362


namespace pencil_pen_cost_l2843_284314

theorem pencil_pen_cost (pencil_cost pen_cost : ℝ) 
  (h1 : 3 * pencil_cost + 4 * pen_cost = 5.20)
  (h2 : 4 * pencil_cost + 3 * pen_cost = 4.90) : 
  pencil_cost + 3 * pen_cost = 3.1857 := by
  sorry

end pencil_pen_cost_l2843_284314


namespace angle_between_vectors_l2843_284307

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- Given nonzero vectors a and b such that ||a|| = ||b|| = 2||a + b||,
    the cosine of the angle between them is -7/8 -/
theorem angle_between_vectors (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : ‖a‖ = ‖b‖ ∧ ‖a‖ = 2 * ‖a + b‖) : 
  inner a b / (‖a‖ * ‖b‖) = -7/8 := by
  sorry

end angle_between_vectors_l2843_284307


namespace rebecca_earnings_l2843_284308

/-- Rebecca's hair salon earnings calculation --/
theorem rebecca_earnings : 
  let haircut_price : ℕ := 30
  let perm_price : ℕ := 40
  let dye_job_price : ℕ := 60
  let dye_cost : ℕ := 10
  let haircut_count : ℕ := 4
  let perm_count : ℕ := 1
  let dye_job_count : ℕ := 2
  let tips : ℕ := 50
  
  haircut_price * haircut_count + 
  perm_price * perm_count + 
  (dye_job_price - dye_cost) * dye_job_count + 
  tips = 310 :=
by
  sorry


end rebecca_earnings_l2843_284308


namespace range_of_a_l2843_284350

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x, x^2 - (a-1)*x + 1 > 0

def q (a : ℝ) : Prop := ∀ x y, x < y → (a+1)^x < (a+1)^y

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (¬(p a ∧ q a) ∧ (p a ∨ q a)) →
  ((-1 < a ∧ a ≤ 0) ∨ (a ≥ 3)) :=
by sorry

end range_of_a_l2843_284350


namespace even_function_graph_l2843_284344

/-- An even function is a function that satisfies f(-x) = f(x) for all x in its domain -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The statement that (-a, f(a)) lies on the graph of f for any even function f and any real a -/
theorem even_function_graph (f : ℝ → ℝ) (h : EvenFunction f) (a : ℝ) :
  f (-a) = f a := by sorry

end even_function_graph_l2843_284344


namespace no_valid_coloring_l2843_284304

theorem no_valid_coloring :
  ¬ ∃ (f : ℕ+ → Fin 3),
    (∀ c : Fin 3, ∃ n : ℕ+, f n = c) ∧
    (∀ a b : ℕ+, f a ≠ f b → f (a * b) ≠ f a ∧ f (a * b) ≠ f b) :=
by sorry

end no_valid_coloring_l2843_284304


namespace triangle_side_length_l2843_284392

theorem triangle_side_length (A B C M : ℝ × ℝ) : 
  -- Triangle ABC is right-angled at C
  (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2) →
  -- AC = BC
  ((C.1 - A.1)^2 + (C.2 - A.2)^2) = ((C.1 - B.1)^2 + (C.2 - B.2)^2) →
  -- M is an interior point (implied by the distances)
  -- MC = 1
  ((M.1 - C.1)^2 + (M.2 - C.2)^2) = 1 →
  -- MA = 2
  ((M.1 - A.1)^2 + (M.2 - A.2)^2) = 4 →
  -- MB = √2
  ((M.1 - B.1)^2 + (M.2 - B.2)^2) = 2 →
  -- AB = √10
  ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 10 :=
by
  sorry

end triangle_side_length_l2843_284392


namespace expression_simplification_l2843_284324

theorem expression_simplification (x : ℝ) (h : x = 1) : 
  (4 + (4 + x^2) / x) / ((x + 2) / x) = 3 := by
  sorry

end expression_simplification_l2843_284324


namespace perpendicular_vector_k_value_l2843_284378

theorem perpendicular_vector_k_value :
  let a : Fin 2 → ℝ := ![1, 1]
  let b : Fin 2 → ℝ := ![2, -3]
  ∀ k : ℝ, (k • a - 2 • b) • a = 0 → k = -1 :=
by
  sorry

end perpendicular_vector_k_value_l2843_284378


namespace point_inside_circle_a_range_l2843_284335

theorem point_inside_circle_a_range (a : ℝ) : 
  (((1 - a)^2 + (1 + a)^2) < 4) → (-1 < a ∧ a < 1) :=
by sorry

end point_inside_circle_a_range_l2843_284335


namespace min_value_of_f_on_interval_l2843_284361

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_minus_x_squared_odd : ∀ x : ℝ, f (-x) - (-x)^2 = -(f x - x^2)
axiom f_plus_2_pow_x_even : ∀ x : ℝ, f (-x) + 2^(-x) = f x + 2^x

-- Define the interval
def interval : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ -1}

-- State the theorem
theorem min_value_of_f_on_interval :
  ∃ x₀ ∈ interval, ∀ x ∈ interval, f x₀ ≤ f x ∧ f x₀ = 7/4 :=
sorry

end min_value_of_f_on_interval_l2843_284361


namespace fraction_comparison_l2843_284352

theorem fraction_comparison (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) : 
  b / (a - c) < a / (b - d) := by
  sorry

end fraction_comparison_l2843_284352


namespace power_of_power_at_three_l2843_284340

theorem power_of_power_at_three : (3^3)^(3^3) = 27^27 := by
  sorry

end power_of_power_at_three_l2843_284340


namespace treat_cost_theorem_l2843_284346

/-- Represents the cost of treats -/
structure TreatCost where
  chocolate : ℚ
  popsicle : ℚ
  lollipop : ℚ

/-- The cost relationships between treats -/
def cost_relationship (c : TreatCost) : Prop :=
  3 * c.chocolate = 2 * c.popsicle ∧ 2 * c.lollipop = 5 * c.chocolate

/-- The number of popsicles that can be bought with the money for 3 lollipops -/
def popsicles_for_lollipops (c : TreatCost) : ℚ :=
  (3 * c.lollipop) / c.popsicle

/-- The number of chocolates that can be bought with the money for 3 chocolates, 2 popsicles, and 2 lollipops -/
def chocolates_for_combination (c : TreatCost) : ℚ :=
  (3 * c.chocolate + 2 * c.popsicle + 2 * c.lollipop) / c.chocolate

theorem treat_cost_theorem (c : TreatCost) :
  cost_relationship c →
  popsicles_for_lollipops c = 5 ∧
  chocolates_for_combination c = 11 := by
  sorry

end treat_cost_theorem_l2843_284346


namespace opera_house_seats_l2843_284364

theorem opera_house_seats (rows : ℕ) (revenue : ℕ) (ticket_price : ℕ) (occupancy_rate : ℚ) :
  rows = 150 →
  revenue = 12000 →
  ticket_price = 10 →
  occupancy_rate = 4/5 →
  ∃ (seats_per_row : ℕ), seats_per_row = 10 ∧ 
    (revenue / ticket_price : ℚ) = (occupancy_rate * (rows * seats_per_row : ℚ)) :=
by sorry

end opera_house_seats_l2843_284364


namespace negation_equivalence_l2843_284345

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for "man" and "tall"
variable (man : U → Prop)
variable (tall : U → Prop)

-- Define the original statement "all men are tall"
def all_men_are_tall : Prop := ∀ x : U, man x → tall x

-- Define the negation of the original statement
def negation_of_all_men_are_tall : Prop := ¬(∀ x : U, man x → tall x)

-- Define "some men are short"
def some_men_are_short : Prop := ∃ x : U, man x ∧ ¬(tall x)

-- Theorem stating that the negation is equivalent to "some men are short"
theorem negation_equivalence : 
  negation_of_all_men_are_tall U man tall ↔ some_men_are_short U man tall :=
sorry

end negation_equivalence_l2843_284345


namespace investment_problem_l2843_284380

theorem investment_problem (P : ℝ) : 
  (P * 0.15 * 2 - P * 0.12 * 2 = 840) → P = 14000 := by
  sorry

end investment_problem_l2843_284380


namespace ram_work_time_l2843_284353

/-- Ram's efficiency compared to Krish's -/
def ram_efficiency : ℚ := 1/2

/-- Time taken by Ram and Krish working together (in days) -/
def combined_time : ℕ := 7

/-- Time taken by Ram working alone (in days) -/
def ram_alone_time : ℕ := 21

theorem ram_work_time :
  ram_efficiency * combined_time * 2 = ram_alone_time := by
  sorry

end ram_work_time_l2843_284353


namespace inverse_j_minus_j_inv_l2843_284306

-- Define the complex number i
def i : ℂ := Complex.I

-- Define j in terms of i
def j : ℂ := i + 1

-- Theorem statement
theorem inverse_j_minus_j_inv :
  (j - j⁻¹)⁻¹ = (-3 * i + 1) / 5 :=
by
  sorry

end inverse_j_minus_j_inv_l2843_284306


namespace max_value_inequality_equality_at_six_strict_inequality_for_greater_than_six_l2843_284318

theorem max_value_inequality (a : ℝ) : 
  (∀ x > 1, (x^2 + 3) / (x - 1) ≥ a) → a ≤ 6 := by
  sorry

theorem equality_at_six : 
  ∃ x > 1, (x^2 + 3) / (x - 1) = 6 := by
  sorry

theorem strict_inequality_for_greater_than_six : 
  ∀ b > 6, ∃ x > 1, (x^2 + 3) / (x - 1) < b := by
  sorry

end max_value_inequality_equality_at_six_strict_inequality_for_greater_than_six_l2843_284318


namespace solution_set_inequality_l2843_284377

theorem solution_set_inequality (x : ℝ) : (x - 1) * (3 - x) ≥ 0 ↔ 1 ≤ x ∧ x ≤ 3 := by
  sorry

end solution_set_inequality_l2843_284377


namespace problem_solution_l2843_284372

def digits : Finset Nat := {0, 1, 2, 3, 4, 5}

def naturalNumbersWithoutRepeats (d : Finset Nat) : Nat :=
  sorry

def fourDigitEvenWithoutRepeats (d : Finset Nat) : Nat :=
  sorry

def fourDigitGreaterThan4023WithoutRepeats (d : Finset Nat) : Nat :=
  sorry

theorem problem_solution (d : Finset Nat) (h : d = digits) :
  naturalNumbersWithoutRepeats d = 1631 ∧
  fourDigitEvenWithoutRepeats d = 156 ∧
  fourDigitGreaterThan4023WithoutRepeats d = 115 :=
by sorry

end problem_solution_l2843_284372


namespace triangle_most_stable_l2843_284393

-- Define the shapes
inductive Shape
  | Rectangle
  | Trapezoid
  | Parallelogram
  | Triangle

-- Define stability as a property of shapes
def is_stable (s : Shape) : Prop :=
  match s with
  | Shape.Triangle => true
  | _ => false

-- Define the stability comparison
def more_stable (s1 s2 : Shape) : Prop :=
  is_stable s1 ∧ ¬is_stable s2

-- Theorem statement
theorem triangle_most_stable :
  ∀ s : Shape, s ≠ Shape.Triangle → more_stable Shape.Triangle s :=
sorry

end triangle_most_stable_l2843_284393


namespace simplify_K_simplify_L_l2843_284309

-- Part (a)
theorem simplify_K (x y : ℝ) (h : x ≥ y^2) :
  Real.sqrt (x + 2*y*Real.sqrt (x - y^2)) + Real.sqrt (x - 2*y*Real.sqrt (x - y^2)) = 
  max (2*abs y) (2*Real.sqrt (x - y^2)) := by sorry

-- Part (b)
theorem simplify_L (x y z : ℝ) (h : x*y + y*z + z*x = 1) :
  (2*x*y*z) / Real.sqrt ((1 + x^2)*(1 + y^2)*(1 + z^2)) = 
  (2*x*y*z) / abs (x + y + z - x*y*z) := by sorry

end simplify_K_simplify_L_l2843_284309


namespace solution_set_implies_a_value_l2843_284398

theorem solution_set_implies_a_value (a b : ℝ) : 
  (∀ x, |x - a| < b ↔ 2 < x ∧ x < 4) → a = 3 := by
sorry

end solution_set_implies_a_value_l2843_284398


namespace power_of_two_problem_l2843_284399

theorem power_of_two_problem (a b : ℕ+) 
  (h1 : (2 ^ a.val) ^ b.val = 2 ^ 2) 
  (h2 : 2 ^ a.val * 2 ^ b.val = 8) : 
  2 ^ a.val = 2 := by
  sorry

end power_of_two_problem_l2843_284399


namespace chime_2500_date_l2843_284348

/-- Represents a date --/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Represents a time --/
structure Time where
  hour : ℕ
  minute : ℕ

/-- Represents a clock with a specific chiming pattern --/
structure ChimingClock where
  /-- Chimes once at 30 minutes past each hour --/
  chimeAtHalfHour : Bool
  /-- Chimes on the hour according to the hour number --/
  chimeOnHour : ℕ → ℕ

/-- Calculates the number of chimes between two dates and times --/
def countChimes (clock : ChimingClock) (startDate : Date) (startTime : Time) (endDate : Date) (endTime : Time) : ℕ := sorry

/-- The theorem to be proved --/
theorem chime_2500_date (clock : ChimingClock) : 
  let startDate := Date.mk 2003 2 26
  let startTime := Time.mk 10 45
  let endDate := Date.mk 2003 3 21
  countChimes clock startDate startTime endDate (Time.mk 23 59) = 2500 := by sorry

end chime_2500_date_l2843_284348


namespace cow_field_theorem_l2843_284327

theorem cow_field_theorem (total_cows : ℕ) (female_cows : ℕ) (male_cows : ℕ) 
  (spotted_females : ℕ) (horned_males : ℕ) : 
  total_cows = 300 →
  female_cows = 2 * male_cows →
  female_cows + male_cows = total_cows →
  spotted_females = female_cows / 2 →
  horned_males = male_cows / 2 →
  spotted_females - horned_males = 50 := by
sorry

end cow_field_theorem_l2843_284327


namespace additional_wax_is_22_l2843_284332

/-- The amount of additional wax needed for painting feathers -/
def additional_wax_needed (total_wax : ℕ) (available_wax : ℕ) : ℕ :=
  total_wax - available_wax

/-- Theorem stating that the additional wax needed is 22 grams -/
theorem additional_wax_is_22 :
  additional_wax_needed 353 331 = 22 := by
  sorry

end additional_wax_is_22_l2843_284332


namespace pie_count_correct_l2843_284375

/-- Represents the number of pie slices served in a meal -/
structure MealServing :=
  (apple : ℕ)
  (blueberry : ℕ)
  (cherry : ℕ)
  (pumpkin : ℕ)

/-- Represents the total number of pie slices served over two days -/
structure TotalServing :=
  (apple : ℕ)
  (blueberry : ℕ)
  (cherry : ℕ)
  (pumpkin : ℕ)

def lunch_today : MealServing := ⟨3, 2, 2, 0⟩
def dinner_today : MealServing := ⟨1, 2, 1, 1⟩
def yesterday : MealServing := ⟨8, 8, 0, 0⟩

def total_served : TotalServing := ⟨12, 12, 3, 1⟩

theorem pie_count_correct : 
  lunch_today.apple + dinner_today.apple + yesterday.apple = total_served.apple ∧
  lunch_today.blueberry + dinner_today.blueberry + yesterday.blueberry = total_served.blueberry ∧
  lunch_today.cherry + dinner_today.cherry + yesterday.cherry = total_served.cherry ∧
  lunch_today.pumpkin + dinner_today.pumpkin + yesterday.pumpkin = total_served.pumpkin :=
by sorry

end pie_count_correct_l2843_284375


namespace power_of_product_l2843_284370

theorem power_of_product (a : ℝ) : (2 * a) ^ 3 = 8 * a ^ 3 := by
  sorry

end power_of_product_l2843_284370


namespace arithmetic_sequence_fifth_term_l2843_284391

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_first : a 1 = 3/5)
  (h_ninth : a 9 = 2/3) :
  a 5 = 19/30 := by
  sorry

end arithmetic_sequence_fifth_term_l2843_284391


namespace find_divisor_l2843_284300

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 507 → quotient = 61 → remainder = 19 →
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 8 := by
  sorry

end find_divisor_l2843_284300


namespace parallel_iff_a_eq_two_l2843_284305

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (m1 n1 c1 m2 n2 c2 : ℝ) : Prop :=
  m1 * n2 = m2 * n1 ∧ m1 ≠ 0 ∧ m2 ≠ 0

/-- The theorem states that a = 2 is a necessary and sufficient condition
    for the lines 2x - ay + 1 = 0 and (a-1)x - y + a = 0 to be parallel -/
theorem parallel_iff_a_eq_two (a : ℝ) :
  parallel 2 (-a) 1 (a-1) (-1) a ↔ a = 2 := by
  sorry

end parallel_iff_a_eq_two_l2843_284305


namespace geometric_sequence_product_l2843_284373

/-- A positive geometric sequence -/
def IsPositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  IsPositiveGeometricSequence a →
  a 1 * a 19 = 16 →
  a 8 * a 10 * a 12 = 64 := by
    sorry

end geometric_sequence_product_l2843_284373
