import Mathlib

namespace sams_age_two_years_ago_l3794_379482

/-- Given the ages of John and Sam, prove Sam's age two years ago -/
theorem sams_age_two_years_ago (john_age sam_age : ℕ) : 
  john_age = 3 * sam_age →
  john_age + 9 = 2 * (sam_age + 9) →
  sam_age - 2 = 7 := by
  sorry

#check sams_age_two_years_ago

end sams_age_two_years_ago_l3794_379482


namespace greatest_of_five_consecutive_integers_sum_cube_l3794_379426

theorem greatest_of_five_consecutive_integers_sum_cube (n : ℤ) (m : ℤ) : 
  (5 * n + 10 = m^3) → 
  (∀ k : ℤ, k > n → 5 * k + 10 ≠ m^3) → 
  202 = n + 4 :=
by sorry

end greatest_of_five_consecutive_integers_sum_cube_l3794_379426


namespace problem_statement_l3794_379401

variable (a : ℝ)

def p : Prop := ∀ x : ℝ, x^2 + (a-1)*x + a^2 > 0

def q : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → (2*a^2 - a)^x₁ < (2*a^2 - a)^x₂

theorem problem_statement : (p a ∨ q a) → (a < -1/2 ∨ a > 1/3) := by
  sorry

end problem_statement_l3794_379401


namespace george_number_l3794_379457

/-- Checks if a number is skipped by a student given their position in the sequence -/
def isSkipped (num : ℕ) (studentPosition : ℕ) : Prop :=
  ∃ k : ℕ, num = 5^studentPosition * (5 * k - 1) - 1

/-- Checks if a number is the sum of squares of two consecutive integers -/
def isSumOfConsecutiveSquares (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 + (k+1)^2

theorem george_number :
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 1005 ∧
  (∀ i : ℕ, i ≥ 1 ∧ i ≤ 6 → ¬isSkipped n i) ∧
  isSumOfConsecutiveSquares n ∧
  n = 25 := by sorry

end george_number_l3794_379457


namespace range_of_x_minus_cos_y_l3794_379461

theorem range_of_x_minus_cos_y :
  ∀ x y : ℝ, x^2 + 2 * Real.cos y = 1 →
  ∃ z : ℝ, z = x - Real.cos y ∧ -1 ≤ z ∧ z ≤ Real.sqrt 3 + 1 ∧
  (∃ x₁ y₁ : ℝ, x₁^2 + 2 * Real.cos y₁ = 1 ∧ x₁ - Real.cos y₁ = -1) ∧
  (∃ x₂ y₂ : ℝ, x₂^2 + 2 * Real.cos y₂ = 1 ∧ x₂ - Real.cos y₂ = Real.sqrt 3 + 1) :=
by sorry

end range_of_x_minus_cos_y_l3794_379461


namespace triangle_sides_proportion_l3794_379470

/-- Represents a triangle with sides a, b, c and incircle diameter 2r --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  r : ℝ

/-- 
  Theorem: If the lengths of the sides of a triangle and the diameter of its incircle 
  form four consecutive terms of an arithmetic progression, then the sides of the 
  triangle are proportional to 3, 4, and 5.
--/
theorem triangle_sides_proportion (t : Triangle) : 
  (∃ (d : ℝ), d > 0 ∧ t.a = t.r * 2 + d ∧ t.b = t.r * 2 + 2 * d ∧ t.c = t.r * 2 + 3 * d) →
  ∃ (k : ℝ), k > 0 ∧ t.a = 3 * k ∧ t.b = 4 * k ∧ t.c = 5 * k :=
by sorry

end triangle_sides_proportion_l3794_379470


namespace bill_share_proof_l3794_379445

def total_bill : ℝ := 139.00
def num_people : ℕ := 9
def tip_percentage : ℝ := 0.10

theorem bill_share_proof :
  let tip := total_bill * tip_percentage
  let total_with_tip := total_bill + tip
  let share_per_person := total_with_tip / num_people
  ∃ ε > 0, |share_per_person - 16.99| < ε :=
by sorry

end bill_share_proof_l3794_379445


namespace line_reflection_x_axis_l3794_379476

/-- Given a line with equation x - y + 1 = 0, its reflection with respect to the x-axis has the equation x + y + 1 = 0 -/
theorem line_reflection_x_axis :
  let original_line := {(x, y) : ℝ × ℝ | x - y + 1 = 0}
  let reflected_line := {(x, y) : ℝ × ℝ | x + y + 1 = 0}
  (∀ (x y : ℝ), (x, y) ∈ original_line ↔ (x, -y) ∈ reflected_line) :=
by sorry

end line_reflection_x_axis_l3794_379476


namespace divisible_by_ten_l3794_379464

theorem divisible_by_ten : ∃ k : ℕ, 11^11 + 12^12 + 13^13 = 10 * k := by
  sorry

end divisible_by_ten_l3794_379464


namespace interest_rate_equation_l3794_379441

/-- Given a principal that doubles in 10 years with quarterly compound interest,
    prove that the annual interest rate satisfies the equation 2 = (1 + r/4)^40 -/
theorem interest_rate_equation (r : ℝ) : 2 = (1 + r/4)^40 ↔ 
  ∀ (P : ℝ), P > 0 → 2*P = P * (1 + r/4)^40 := by
  sorry

end interest_rate_equation_l3794_379441


namespace function_inequality_l3794_379406

-- Define the function f and its derivative
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- State the theorem
theorem function_inequality (hf : ∀ x > 0, x * f' x + x^2 < f x) :
  (2 * f 1 > f 2 + 2) ∧ (3 * f 1 > f 3 + 3) := by
  sorry

end function_inequality_l3794_379406


namespace x_equals_plus_minus_fifteen_l3794_379490

theorem x_equals_plus_minus_fifteen (x : ℝ) : 
  (x / 5) / 3 = 5 / (x / 3) → x = 15 ∨ x = -15 := by
  sorry

end x_equals_plus_minus_fifteen_l3794_379490


namespace total_cost_is_180_l3794_379474

/-- The cost to fill all planter pots at the corners of a rectangle-shaped pool -/
def total_cost : ℝ :=
  let corners_of_rectangle := 4
  let palm_fern_cost := 15.00
  let creeping_jenny_cost := 4.00
  let geranium_cost := 3.50
  let palm_ferns_per_pot := 1
  let creeping_jennies_per_pot := 4
  let geraniums_per_pot := 4
  let cost_per_pot := palm_fern_cost * palm_ferns_per_pot + 
                      creeping_jenny_cost * creeping_jennies_per_pot + 
                      geranium_cost * geraniums_per_pot
  corners_of_rectangle * cost_per_pot

/-- Theorem stating that the total cost to fill all planter pots is $180.00 -/
theorem total_cost_is_180 : total_cost = 180.00 := by
  sorry

end total_cost_is_180_l3794_379474


namespace find_b_plus_c_l3794_379428

theorem find_b_plus_c (a b c d : ℚ) 
  (h1 : a * b + a * c + b * d + c * d = 40) 
  (h2 : a + d = 6) : 
  b + c = 20 / 3 := by
sorry

end find_b_plus_c_l3794_379428


namespace hyperbola_real_semiaxis_range_l3794_379472

/-- The range of values for the length of the real semi-axis of a hyperbola -/
theorem hyperbola_real_semiaxis_range (a b : ℝ) (c : ℝ) :
  a > 0 →
  b > 0 →
  c = 4 →
  c^2 = a^2 + b^2 →
  b / a < Real.sqrt 3 →
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ y / x = Real.tan (60 * π / 180)) →
  2 < a ∧ a < 4 := by
sorry

end hyperbola_real_semiaxis_range_l3794_379472


namespace prob_at_least_one_2_l3794_379484

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The probability of at least one die showing 2 when two fair 8-sided dice are rolled -/
def probAtLeastOne2 : ℚ := 15 / 64

/-- Theorem stating that the probability of at least one die showing 2 
    when two fair 8-sided dice are rolled is 15/64 -/
theorem prob_at_least_one_2 : 
  probAtLeastOne2 = (numSides^2 - (numSides - 1)^2) / numSides^2 := by
  sorry

end prob_at_least_one_2_l3794_379484


namespace simultaneous_equations_solution_l3794_379436

theorem simultaneous_equations_solution (n : ℕ+) (u v : ℝ) :
  (∃ (a b c : ℕ+),
    (a^2 + b^2 + c^2 : ℝ) = 169 * (n : ℝ)^2 ∧
    (a^2 * (u * a^2 + v * b^2) + b^2 * (u * b^2 + v * c^2) + c^2 * (u * c^2 + v * a^2) : ℝ) = 
      ((2 * u + v) * (13 * (n : ℝ))^4) / 4) ↔
  v = 2 * u :=
by sorry

end simultaneous_equations_solution_l3794_379436


namespace triangle_equal_sides_l3794_379486

/-- 
Given a triangle with one side of length 6 cm and two equal sides,
where the sum of all sides is 20 cm, prove that each of the equal sides is 7 cm long.
-/
theorem triangle_equal_sides (a b c : ℝ) : 
  a = 6 → -- One side is 6 cm
  b = c → -- Two sides are equal
  a + b + c = 20 → -- Sum of all sides is 20 cm
  b = 7 := by
sorry

end triangle_equal_sides_l3794_379486


namespace giraffe_height_difference_l3794_379463

/-- The height of the tallest giraffe in inches -/
def tallest_giraffe : ℕ := 96

/-- The height of the shortest giraffe in inches -/
def shortest_giraffe : ℕ := 68

/-- The number of adult giraffes at the zoo -/
def num_giraffes : ℕ := 14

/-- The difference in height between the tallest and shortest giraffe -/
def height_difference : ℕ := tallest_giraffe - shortest_giraffe

theorem giraffe_height_difference :
  height_difference = 28 :=
sorry

end giraffe_height_difference_l3794_379463


namespace factor_calculation_l3794_379424

theorem factor_calculation (n : ℝ) (f : ℝ) (h1 : n = 155) (h2 : n * f - 200 = 110) : f = 2 := by
  sorry

end factor_calculation_l3794_379424


namespace total_clothing_is_934_l3794_379489

/-- The number of shirts Mr. Anderson gave out -/
def shirts : ℕ := 589

/-- The number of trousers Mr. Anderson gave out -/
def trousers : ℕ := 345

/-- The total number of clothing pieces Mr. Anderson gave out -/
def total_clothing : ℕ := shirts + trousers

/-- Theorem stating that the total number of clothing pieces is 934 -/
theorem total_clothing_is_934 : total_clothing = 934 := by
  sorry

end total_clothing_is_934_l3794_379489


namespace child_height_calculation_l3794_379496

/-- Calculates a child's current height given their previous height and growth. -/
def current_height (previous_height growth : Float) : Float :=
  previous_height + growth

theorem child_height_calculation :
  let previous_height : Float := 38.5
  let growth : Float := 3.0
  current_height previous_height growth = 41.5 := by
  sorry

end child_height_calculation_l3794_379496


namespace sum_of_squares_l3794_379404

/-- Given a system of equations, prove that x² + y² + z² = 29 -/
theorem sum_of_squares (x y z : ℝ) 
  (eq1 : 2*x + y + 4*x*y + 6*x*z = -6)
  (eq2 : y + 2*z + 2*x*y + 6*y*z = 4)
  (eq3 : x - z + 2*x*z - 4*y*z = -3) :
  x^2 + y^2 + z^2 = 29 := by
sorry

end sum_of_squares_l3794_379404


namespace tanning_salon_revenue_l3794_379498

/-- Revenue calculation for a tanning salon --/
theorem tanning_salon_revenue :
  let first_visit_cost : ℕ := 10
  let subsequent_visit_cost : ℕ := 8
  let total_customers : ℕ := 100
  let second_visit_customers : ℕ := 30
  let third_visit_customers : ℕ := 10
  
  let first_visit_revenue := first_visit_cost * total_customers
  let second_visit_revenue := subsequent_visit_cost * second_visit_customers
  let third_visit_revenue := subsequent_visit_cost * third_visit_customers
  
  first_visit_revenue + second_visit_revenue + third_visit_revenue = 1320 :=
by
  sorry


end tanning_salon_revenue_l3794_379498


namespace hit_probability_random_gun_selection_l3794_379494

/-- The probability of hitting a target when randomly selecting a gun from a set of calibrated and uncalibrated guns. -/
theorem hit_probability_random_gun_selection 
  (total_guns : ℕ) 
  (calibrated_guns : ℕ) 
  (uncalibrated_guns : ℕ) 
  (calibrated_accuracy : ℝ) 
  (uncalibrated_accuracy : ℝ) 
  (h1 : total_guns = 5)
  (h2 : calibrated_guns = 3)
  (h3 : uncalibrated_guns = 2)
  (h4 : calibrated_guns + uncalibrated_guns = total_guns)
  (h5 : calibrated_accuracy = 0.9)
  (h6 : uncalibrated_accuracy = 0.4) :
  (calibrated_guns : ℝ) / total_guns * calibrated_accuracy + 
  (uncalibrated_guns : ℝ) / total_guns * uncalibrated_accuracy = 0.7 := by
  sorry

end hit_probability_random_gun_selection_l3794_379494


namespace parallel_condition_l3794_379451

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m₁ n₁ : ℝ) (m₂ n₂ : ℝ) : Prop := m₁ * n₂ = m₂ * n₁

/-- Definition of line l₁ -/
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y - 1 = 0

/-- Definition of line l₂ -/
def l₂ (a : ℝ) (x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

theorem parallel_condition (a : ℝ) :
  (a = 1 → are_parallel a 2 1 (a + 1)) ∧
  (∃ b : ℝ, b ≠ 1 ∧ are_parallel b 2 1 (b + 1)) :=
by sorry

end parallel_condition_l3794_379451


namespace impossible_coloring_exists_l3794_379485

-- Define the grid
def Grid := ℤ × ℤ

-- Define a chessboard polygon
def ChessboardPolygon := Set Grid

-- Define a coloring of the grid
def Coloring := Grid → Bool

-- Define congruence for chessboard polygons
def Congruent (F G : ChessboardPolygon) : Prop := sorry

-- Define the number of green cells in a polygon given a coloring
def GreenCells (F : ChessboardPolygon) (c : Coloring) : ℕ := sorry

-- The main theorem
theorem impossible_coloring_exists :
  ∃ F : ChessboardPolygon,
    ∀ c : Coloring,
      (∃ G : ChessboardPolygon, Congruent F G ∧ GreenCells G c = 0) ∨
      (∃ G : ChessboardPolygon, Congruent F G ∧ GreenCells G c > 2020) := by
  sorry

end impossible_coloring_exists_l3794_379485


namespace chess_board_pawn_placement_l3794_379437

theorem chess_board_pawn_placement :
  let board_size : ℕ := 5
  let num_pawns : ℕ := 5
  let ways_to_place_in_rows : ℕ := Nat.factorial board_size
  let ways_to_arrange_pawns : ℕ := Nat.factorial num_pawns
  ways_to_place_in_rows * ways_to_arrange_pawns = 14400 :=
by
  sorry

end chess_board_pawn_placement_l3794_379437


namespace quadratic_root_value_l3794_379473

theorem quadratic_root_value (k : ℝ) : 
  (∀ x : ℂ, 5 * x^2 + 7 * x + k = 0 ↔ x = Complex.mk (-7/10) ((Real.sqrt 399)/10) ∨ x = Complex.mk (-7/10) (-(Real.sqrt 399)/10)) →
  k = 22.4 := by
sorry

end quadratic_root_value_l3794_379473


namespace potato_price_proof_l3794_379462

/-- The original price of one bag of potatoes in rubles -/
def original_price : ℝ := 250

/-- The number of bags each trader bought -/
def bags_bought : ℕ := 60

/-- Andrey's price increase percentage -/
def andrey_increase : ℝ := 100

/-- Boris's first price increase percentage -/
def boris_first_increase : ℝ := 60

/-- Boris's second price increase percentage -/
def boris_second_increase : ℝ := 40

/-- Number of bags Boris sold at first price -/
def boris_first_sale : ℕ := 15

/-- Number of bags Boris sold at second price -/
def boris_second_sale : ℕ := 45

/-- The difference in earnings between Boris and Andrey in rubles -/
def earnings_difference : ℝ := 1200

theorem potato_price_proof :
  let andrey_earnings := bags_bought * original_price * (1 + andrey_increase / 100)
  let boris_first_earnings := boris_first_sale * original_price * (1 + boris_first_increase / 100)
  let boris_second_earnings := boris_second_sale * original_price * (1 + boris_first_increase / 100) * (1 + boris_second_increase / 100)
  boris_first_earnings + boris_second_earnings - andrey_earnings = earnings_difference :=
by sorry

end potato_price_proof_l3794_379462


namespace family_game_score_l3794_379471

theorem family_game_score : 
  let dad_score : ℕ := 7
  let olaf_score : ℕ := 3 * dad_score
  let sister_score : ℕ := dad_score + 4
  let mom_score : ℕ := 2 * sister_score
  dad_score + olaf_score + sister_score + mom_score = 61 :=
by sorry

end family_game_score_l3794_379471


namespace isosceles_triangle_perimeter_l3794_379450

/-- The roots of the quadratic equation x^2 - 7x + 10 = 0 -/
def roots : Set ℝ := {x : ℝ | x^2 - 7*x + 10 = 0}

/-- An isosceles triangle with two sides equal to the roots of x^2 - 7x + 10 = 0 -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : a = b ∨ b = c ∨ a = c
  rootSides : a ∈ roots ∧ b ∈ roots

/-- The perimeter of a triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

/-- Theorem: The perimeter of the isosceles triangle is 12 -/
theorem isosceles_triangle_perimeter : 
  ∀ t : IsoscelesTriangle, perimeter t = 12 := by
  sorry


end isosceles_triangle_perimeter_l3794_379450


namespace lines_perpendicular_to_plane_are_parallel_l3794_379421

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

end lines_perpendicular_to_plane_are_parallel_l3794_379421


namespace min_values_theorem_l3794_379480

theorem min_values_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a * b = a + 3 * b) :
  (∀ x y, x > 0 → y > 0 → 3 * x * y = x + 3 * y → 3 * a + b ≤ 3 * x + y) ∧
  (∀ x y, x > 0 → y > 0 → 3 * x * y = x + 3 * y → a * b ≤ x * y) ∧
  (∀ x y, x > 0 → y > 0 → 3 * x * y = x + 3 * y → a^2 + 9 * b^2 ≤ x^2 + 9 * y^2) ∧
  (3 * a + b = 16 / 3 ∨ a * b = 4 / 3 ∨ a^2 + 9 * b^2 = 8) :=
by sorry

end min_values_theorem_l3794_379480


namespace range_of_m_solution_set_l3794_379447

-- Define the function f
def f (x : ℝ) : ℝ := |x - 5| - |x - 2|

-- Theorem for the range of m
theorem range_of_m : 
  {m : ℝ | ∃ x, f x ≤ m} = {m : ℝ | m ≥ -3} := by sorry

-- Theorem for the solution set of the inequality
theorem solution_set : 
  {x : ℝ | x^2 - 8*x + 15 + f x ≤ 0} = {x : ℝ | 5 - Real.sqrt 3 ≤ x ∧ x ≤ 6} := by sorry

end range_of_m_solution_set_l3794_379447


namespace only_two_reduces_to_zero_l3794_379495

/-- A move on a table is either subtracting n from a column or multiplying a row by n -/
inductive Move (n : ℕ+)
  | subtract_column : Move n
  | multiply_row : Move n

/-- A table is a rectangular array of positive integers -/
def Table := Array (Array ℕ+)

/-- Apply a move to a table -/
def apply_move (t : Table) (m : Move n) : Table :=
  sorry

/-- A table is reducible to zero if there exists a sequence of moves that makes all entries zero -/
def reducible_to_zero (t : Table) (n : ℕ+) : Prop :=
  sorry

/-- The main theorem: n = 2 is the only value that allows any table to be reduced to zero -/
theorem only_two_reduces_to_zero :
  ∀ n : ℕ+, (∀ t : Table, reducible_to_zero t n) ↔ n = 2 :=
  sorry

end only_two_reduces_to_zero_l3794_379495


namespace boys_without_calculators_l3794_379449

theorem boys_without_calculators (total_boys : ℕ) (students_with_calc : ℕ) (girls_with_calc : ℕ) (forgot_calc : ℕ) :
  total_boys = 20 →
  students_with_calc = 26 →
  girls_with_calc = 15 →
  forgot_calc = 3 →
  total_boys - (students_with_calc - girls_with_calc + (forgot_calc * total_boys) / (students_with_calc + forgot_calc)) = 8 :=
by
  sorry


end boys_without_calculators_l3794_379449


namespace driving_distance_differences_l3794_379440

/-- Represents the driving scenario with Ian, Han, and Jan -/
structure DrivingScenario where
  ian_time : ℝ
  ian_speed : ℝ
  han_extra_time : ℝ := 2
  han_extra_speed : ℝ := 10
  jan_extra_time : ℝ := 3
  jan_extra_speed : ℝ := 15
  han_extra_distance : ℝ := 100

/-- Calculate the distance driven by Ian -/
def ian_distance (scenario : DrivingScenario) : ℝ :=
  scenario.ian_time * scenario.ian_speed

/-- Calculate the distance driven by Han -/
def han_distance (scenario : DrivingScenario) : ℝ :=
  (scenario.ian_time + scenario.han_extra_time) * (scenario.ian_speed + scenario.han_extra_speed)

/-- Calculate the distance driven by Jan -/
def jan_distance (scenario : DrivingScenario) : ℝ :=
  (scenario.ian_time + scenario.jan_extra_time) * (scenario.ian_speed + scenario.jan_extra_speed)

/-- The main theorem stating the differences in distances driven -/
theorem driving_distance_differences (scenario : DrivingScenario) :
  jan_distance scenario - ian_distance scenario = 150 ∧
  jan_distance scenario - han_distance scenario = 150 := by
  sorry


end driving_distance_differences_l3794_379440


namespace square_of_sum_85_7_l3794_379488

theorem square_of_sum_85_7 : (85 + 7)^2 = 8464 := by
  sorry

end square_of_sum_85_7_l3794_379488


namespace intersection_with_complement_l3794_379403

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 3, 4, 5}
def B : Set ℕ := {2, 4, 6, 8}

theorem intersection_with_complement :
  A ∩ (U \ B) = {3, 5} := by sorry

end intersection_with_complement_l3794_379403


namespace cost_price_calculation_l3794_379448

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) :
  selling_price = 150 ∧ profit_percentage = 25 →
  ∃ (cost_price : ℝ), cost_price = 120 ∧
    selling_price = cost_price * (1 + profit_percentage / 100) :=
by sorry

end cost_price_calculation_l3794_379448


namespace line_passes_through_point_l3794_379420

theorem line_passes_through_point :
  ∀ (k : ℝ), (1 + 4 * k^2) * 2 - (2 - 3 * k) * 2 + (2 - 14 * k) = 0 := by
  sorry

end line_passes_through_point_l3794_379420


namespace arithmetic_sequence_problem_l3794_379423

/-- Given that -1, a, b, c, -9 form an arithmetic sequence, prove that b = -5 and ac = 21 -/
theorem arithmetic_sequence_problem (a b c : ℝ) 
  (h1 : ∃ (d : ℝ), a - (-1) = d ∧ b - a = d ∧ c - b = d ∧ (-9) - c = d) : 
  b = -5 ∧ a * c = 21 := by
sorry

end arithmetic_sequence_problem_l3794_379423


namespace smallest_multiplier_for_120_perfect_square_l3794_379422

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y * y

theorem smallest_multiplier_for_120_perfect_square :
  ∃! n : ℕ, n > 0 ∧ is_perfect_square (120 * n) ∧ 
  ∀ m : ℕ, m > 0 → is_perfect_square (120 * m) → n ≤ m :=
by sorry

end smallest_multiplier_for_120_perfect_square_l3794_379422


namespace two_heads_in_three_tosses_l3794_379427

/-- The probability of getting exactly k successes in n trials with probability p of success on each trial. -/
def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

/-- Theorem: The probability of getting exactly 2 heads when a fair coin is tossed 3 times is 0.375 -/
theorem two_heads_in_three_tosses :
  binomialProbability 3 2 (1/2) = 0.375 := by
  sorry

end two_heads_in_three_tosses_l3794_379427


namespace fries_popcorn_ratio_is_two_to_one_l3794_379407

/-- Represents the movie night scenario with Joseph and his friends -/
structure MovieNight where
  first_movie_length : ℕ
  second_movie_length : ℕ
  popcorn_time : ℕ
  total_time : ℕ

/-- Calculates the ratio of fries-making time to popcorn-making time -/
def fries_to_popcorn_ratio (mn : MovieNight) : ℚ :=
  let total_movie_time := mn.first_movie_length + mn.second_movie_length
  let fries_time := mn.total_time - total_movie_time - mn.popcorn_time
  fries_time / mn.popcorn_time

/-- Theorem stating the ratio of fries-making time to popcorn-making time is 2:1 -/
theorem fries_popcorn_ratio_is_two_to_one (mn : MovieNight)
    (h1 : mn.first_movie_length = 90)
    (h2 : mn.second_movie_length = mn.first_movie_length + 30)
    (h3 : mn.popcorn_time = 10)
    (h4 : mn.total_time = 240) :
    fries_to_popcorn_ratio mn = 2 := by
  sorry

end fries_popcorn_ratio_is_two_to_one_l3794_379407


namespace cookies_left_after_six_days_l3794_379468

/-- Represents the number of cookies baked and eaten over six days -/
structure CookieCount where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  parentEaten : ℕ
  neighborEaten : ℕ

/-- Calculates the total number of cookies left after six days -/
def totalCookiesLeft (c : CookieCount) : ℕ :=
  c.monday + c.tuesday + c.wednesday + c.thursday + c.friday + c.saturday - (c.parentEaten + c.neighborEaten)

/-- Theorem stating the number of cookies left after six days -/
theorem cookies_left_after_six_days :
  ∃ (c : CookieCount),
    c.monday = 32 ∧
    c.tuesday = c.monday / 2 ∧
    c.wednesday = (c.tuesday * 3) - 4 ∧
    c.thursday = (c.monday * 2) - 10 ∧
    c.friday = (c.tuesday * 3) - 6 ∧
    c.saturday = c.monday + c.friday ∧
    c.parentEaten = 2 * 6 ∧
    c.neighborEaten = 8 ∧
    totalCookiesLeft c = 242 := by
  sorry


end cookies_left_after_six_days_l3794_379468


namespace least_number_of_cans_l3794_379466

def maaza_liters : ℕ := 157
def pepsi_liters : ℕ := 173
def sprite_liters : ℕ := 389

def total_cans : ℕ := maaza_liters + pepsi_liters + sprite_liters

theorem least_number_of_cans :
  ∀ (can_size : ℕ),
    can_size > 0 →
    can_size ∣ maaza_liters →
    can_size ∣ pepsi_liters →
    can_size ∣ sprite_liters →
    total_cans ≤ maaza_liters / can_size + pepsi_liters / can_size + sprite_liters / can_size :=
by sorry

end least_number_of_cans_l3794_379466


namespace billy_watched_79_videos_l3794_379405

/-- The number of videos Billy watches before finding one he likes -/
def total_videos_watched (suggestions_per_attempt : ℕ) (unsuccessful_attempts : ℕ) (position_of_liked_video : ℕ) : ℕ :=
  suggestions_per_attempt * unsuccessful_attempts + (position_of_liked_video - 1)

/-- Theorem stating that Billy watches 79 videos before finding one he likes -/
theorem billy_watched_79_videos :
  total_videos_watched 15 5 5 = 79 := by
sorry

end billy_watched_79_videos_l3794_379405


namespace rajan_profit_share_l3794_379434

/-- Calculates the share of profit for a partner in a business --/
def calculate_profit_share (
  rajan_investment : ℕ) (rajan_duration : ℕ)
  (rakesh_investment : ℕ) (rakesh_duration : ℕ)
  (mukesh_investment : ℕ) (mukesh_duration : ℕ)
  (total_profit : ℕ) : ℕ :=
  let rajan_ratio := rajan_investment * rajan_duration
  let rakesh_ratio := rakesh_investment * rakesh_duration
  let mukesh_ratio := mukesh_investment * mukesh_duration
  let total_ratio := rajan_ratio + rakesh_ratio + mukesh_ratio
  (rajan_ratio * total_profit) / total_ratio

/-- Theorem stating that Rajan's share of the profit is 2400 --/
theorem rajan_profit_share :
  calculate_profit_share 20000 12 25000 4 15000 8 4600 = 2400 :=
by sorry

end rajan_profit_share_l3794_379434


namespace fifteenth_student_age_l3794_379439

theorem fifteenth_student_age 
  (total_students : Nat) 
  (avg_age_all : ℝ) 
  (group1_students : Nat) 
  (avg_age_group1 : ℝ) 
  (group2_students : Nat) 
  (avg_age_group2 : ℝ)
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : group1_students = 5)
  (h4 : avg_age_group1 = 14)
  (h5 : group2_students = 9)
  (h6 : avg_age_group2 = 16) :
  (total_students : ℝ) * avg_age_all - 
  ((group1_students : ℝ) * avg_age_group1 + (group2_students : ℝ) * avg_age_group2) = 11 := by
  sorry

end fifteenth_student_age_l3794_379439


namespace isosceles_triangle_largest_angle_l3794_379433

theorem isosceles_triangle_largest_angle (α β γ : ℝ) : 
  -- The triangle is isosceles with two 60° angles
  α = 60 ∧ β = 60 ∧ 
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- The largest angle is 60°
  max α (max β γ) = 60 := by
sorry

end isosceles_triangle_largest_angle_l3794_379433


namespace correct_propositions_l3794_379481

theorem correct_propositions :
  let prop1 := (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) ↔ (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0)
  let prop2 := ∀ p q : Prop, (p ∨ q) → (p ∧ q)
  let prop3 := ∀ p q : Prop, ¬(p ∧ q) → (¬p ∧ ¬q)
  let prop4 := (∃ x : ℝ, x^2 + x + 1 < 0) ↔ ¬(∀ x : ℝ, x^2 + x + 1 ≥ 0)
  prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4 := by sorry

end correct_propositions_l3794_379481


namespace min_teachers_for_our_school_l3794_379425

/-- Represents a school with math, physics, and chemistry teachers -/
structure School where
  mathTeachers : ℕ
  physicsTeachers : ℕ
  chemistryTeachers : ℕ
  maxSubjectsPerTeacher : ℕ

/-- The minimum number of teachers required for a given school -/
def minTeachersRequired (s : School) : ℕ := sorry

/-- Our specific school configuration -/
def ourSchool : School :=
  { mathTeachers := 4
    physicsTeachers := 3
    chemistryTeachers := 3
    maxSubjectsPerTeacher := 2 }

/-- Theorem stating that the minimum number of teachers required for our school is 6 -/
theorem min_teachers_for_our_school :
  minTeachersRequired ourSchool = 6 := by sorry

end min_teachers_for_our_school_l3794_379425


namespace inequality_proof_l3794_379479

theorem inequality_proof (a b c d : ℝ) (n : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hn : n ≥ 9) :
  a^n + b^n + c^n + d^n ≥ 
  a^(n-9) * b^4 * c^3 * d^2 + 
  b^(n-9) * c^4 * d^3 * a^2 + 
  c^(n-9) * d^4 * a^3 * b^2 + 
  d^(n-9) * a^4 * b^3 * c^2 := by
sorry

end inequality_proof_l3794_379479


namespace point_P_coordinates_l3794_379410

def P (m : ℝ) : ℝ × ℝ := (m + 3, 2*m - 1)

theorem point_P_coordinates :
  (∀ m : ℝ, P m = (0, -7) ↔ (P m).1 = 0) ∧
  (∀ m : ℝ, P m = (10, 13) ↔ (P m).2 = (P m).1 + 3) ∧
  (∀ m : ℝ, P m = (5/2, -2) ↔ |(P m).2| = 2 ∧ (P m).1 > 0 ∧ (P m).2 < 0) :=
by sorry

end point_P_coordinates_l3794_379410


namespace fraction_equality_l3794_379492

theorem fraction_equality (a b c d : ℝ) 
  (h1 : a / b = 4)
  (h2 : b / c = 1 / 3)
  (h3 : c / d = 6) :
  d / a = 1 / 8 := by sorry

end fraction_equality_l3794_379492


namespace outfits_count_l3794_379483

/-- The number of possible outfits given a set of clothing items -/
def number_of_outfits (shirts : ℕ) (ties : ℕ) (pants : ℕ) : ℕ :=
  shirts * pants * (ties + 1)

/-- Theorem stating that with 6 shirts, 4 ties, and 3 pairs of pants,
    the number of possible outfits is 90 -/
theorem outfits_count : number_of_outfits 6 4 3 = 90 := by
  sorry

end outfits_count_l3794_379483


namespace simple_interest_months_l3794_379416

/-- Simple interest calculation -/
theorem simple_interest_months (P R SI : ℚ) (h1 : P = 10000) (h2 : R = 4/100) (h3 : SI = 400) :
  SI = P * R * (12/12) :=
by sorry

end simple_interest_months_l3794_379416


namespace prime_square_sum_equation_l3794_379477

theorem prime_square_sum_equation (p q : ℕ) : 
  (Prime p ∧ Prime q) → 
  (∃ (x y z : ℕ), p^(2*x) + q^(2*y) = z^2) ↔ 
  ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) := by
  sorry

#check prime_square_sum_equation

end prime_square_sum_equation_l3794_379477


namespace largest_power_l3794_379442

theorem largest_power : 
  3^4000 > 2^5000 ∧ 
  3^4000 > 4^3000 ∧ 
  3^4000 > 5^2000 ∧ 
  3^4000 > 6^1000 := by sorry

end largest_power_l3794_379442


namespace midpoint_chain_l3794_379444

/-- Given points A, B, C, D, E, F on a line segment, where:
    C is the midpoint of AB,
    D is the midpoint of AC,
    E is the midpoint of AD,
    F is the midpoint of AE,
    and AB = 64,
    prove that AF = 4. -/
theorem midpoint_chain (A B C D E F : ℝ) : 
  C = (A + B) / 2 →
  D = (A + C) / 2 →
  E = (A + D) / 2 →
  F = (A + E) / 2 →
  B - A = 64 →
  F - A = 4 := by
  sorry

end midpoint_chain_l3794_379444


namespace parallel_lines_l3794_379429

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (m₁ n₁ c₁ m₂ n₂ c₂ : ℝ) : Prop :=
  m₁ * n₂ = m₂ * n₁ ∧ m₁ * c₂ ≠ n₁ * c₂

theorem parallel_lines (a : ℝ) :
  parallel 1 (2*a) (-1) (a-1) (-a) 1 ↔ a = 1/2 :=
sorry

#check parallel_lines

end parallel_lines_l3794_379429


namespace selling_price_with_loss_l3794_379412

theorem selling_price_with_loss (cost_price : ℝ) (loss_percent : ℝ) (selling_price : ℝ) :
  cost_price = 600 →
  loss_percent = 8.333333333333329 →
  selling_price = cost_price * (1 - loss_percent / 100) →
  selling_price = 550 := by
  sorry

end selling_price_with_loss_l3794_379412


namespace min_value_expression_l3794_379402

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (ha' : a < 2) (hb' : b < 2) (hc' : c < 2) :
  (1 / ((2 - a) * (2 - b) * (2 - c))) + (1 / ((2 + a) * (2 + b) * (2 + c))) ≥ 1/4 := by
  sorry

end min_value_expression_l3794_379402


namespace infinitely_many_divisible_by_prime_l3794_379487

theorem infinitely_many_divisible_by_prime (p : Nat) (hp : Prime p) :
  ∃ f : ℕ → ℕ, ∀ k : ℕ, p ∣ (2^(f k) - f k) := by
  sorry

end infinitely_many_divisible_by_prime_l3794_379487


namespace circle_equation_alternatives_l3794_379409

/-- A circle with center on the y-axis, radius 5, passing through (3, -4) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_y_axis : center.1 = 0
  radius_is_5 : radius = 5
  passes_through_point : (center.1 - 3)^2 + (center.2 - (-4))^2 = radius^2

/-- The equation of the circle -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_equation_alternatives (c : Circle) :
  (∀ x y, circle_equation c x y ↔ x^2 + y^2 = 25) ∨
  (∀ x y, circle_equation c x y ↔ x^2 + (y + 8)^2 = 25) := by
  sorry

end circle_equation_alternatives_l3794_379409


namespace rectangle_area_l3794_379478

theorem rectangle_area (x : ℝ) (h : x > 0) : 
  ∃ (w l : ℝ), w > 0 ∧ l > 0 ∧ l = 3 * w ∧ x^2 = w^2 + l^2 ∧ w * l = (3 * x^2) / 10 :=
by
  sorry

end rectangle_area_l3794_379478


namespace tennis_players_count_l3794_379418

theorem tennis_players_count (total_members : ℕ) (badminton_players : ℕ) (both_players : ℕ) (neither_players : ℕ) :
  total_members = 30 →
  badminton_players = 16 →
  both_players = 7 →
  neither_players = 2 →
  ∃ (tennis_players : ℕ), tennis_players = 19 ∧
    tennis_players = total_members - neither_players - (badminton_players - both_players) := by
  sorry


end tennis_players_count_l3794_379418


namespace equation1_solution_equation2_no_solution_l3794_379408

-- Define the equations
def equation1 (x : ℝ) : Prop := (2*x + 9) / (3 - x) = (4*x - 7) / (x - 3)
def equation2 (x : ℝ) : Prop := (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1

-- Theorem for equation1
theorem equation1_solution :
  ∃! x : ℝ, equation1 x ∧ x ≠ 3 := by sorry

-- Theorem for equation2
theorem equation2_no_solution :
  ¬∃ x : ℝ, equation2 x ∧ x ≠ 1 ∧ x ≠ -1 := by sorry

end equation1_solution_equation2_no_solution_l3794_379408


namespace quadratic_discriminant_l3794_379469

/-- The discriminant of a quadratic equation ax^2 + bx + c is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation 6x^2 - 14x + 10 has discriminant -44 -/
theorem quadratic_discriminant :
  discriminant 6 (-14) 10 = -44 := by
  sorry

end quadratic_discriminant_l3794_379469


namespace square_corner_distance_l3794_379458

theorem square_corner_distance (small_perimeter large_area : ℝ) 
  (h_small : small_perimeter = 8)
  (h_large : large_area = 36) : ∃ (distance : ℝ), distance = Real.sqrt 32 :=
by
  sorry

end square_corner_distance_l3794_379458


namespace billy_ticket_usage_l3794_379454

/-- The number of times Billy rode the ferris wheel -/
def ferris_rides : ℕ := 7

/-- The number of times Billy rode the bumper cars -/
def bumper_rides : ℕ := 3

/-- The cost in tickets for each ride -/
def tickets_per_ride : ℕ := 5

/-- The total number of tickets Billy used -/
def total_tickets : ℕ := (ferris_rides + bumper_rides) * tickets_per_ride

theorem billy_ticket_usage : total_tickets = 50 := by
  sorry

end billy_ticket_usage_l3794_379454


namespace square_difference_to_fourth_power_l3794_379415

theorem square_difference_to_fourth_power : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end square_difference_to_fourth_power_l3794_379415


namespace olaf_sailing_speed_l3794_379460

/-- Given the conditions of Olaf's sailing trip, prove the boat's daily travel distance. -/
theorem olaf_sailing_speed :
  -- Total distance to travel
  ∀ (total_distance : ℝ)
  -- Total number of men
  (total_men : ℕ)
  -- Water consumption per man per day (in gallons)
  (water_per_man_per_day : ℝ)
  -- Total water available (in gallons)
  (total_water : ℝ),
  total_distance = 4000 →
  total_men = 25 →
  water_per_man_per_day = 1/2 →
  total_water = 250 →
  -- The boat can travel this many miles per day
  (total_distance / (total_water / (total_men * water_per_man_per_day))) = 200 :=
by
  sorry


end olaf_sailing_speed_l3794_379460


namespace pure_imaginary_ratio_l3794_379411

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ y : ℝ, (3 - 9 * I) * (a + b * I) = y * I) : a / b = -3 :=
by sorry

end pure_imaginary_ratio_l3794_379411


namespace odd_function_inequality_l3794_379465

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x / 3 - 2^x
  else if x < 0 then x / 3 + 2^(-x)
  else 0

-- State the theorem
theorem odd_function_inequality (k : ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ t, f (t^2 - 2*t) + f (2*t^2 - k) < 0) →
  k < -1/3 := by
  sorry

end odd_function_inequality_l3794_379465


namespace downstream_distance_l3794_379497

/-- Calculates the distance traveled downstream by a boat -/
theorem downstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (time : ℝ) 
  (h1 : boat_speed = 24) 
  (h2 : stream_speed = 4) 
  (h3 : time = 2) : 
  boat_speed * time + stream_speed * time = 56 := by
  sorry

#check downstream_distance

end downstream_distance_l3794_379497


namespace relay_race_probability_l3794_379443

/-- The number of short-distance runners --/
def total_runners : ℕ := 6

/-- The number of runners needed for the relay race --/
def team_size : ℕ := 4

/-- The probability that athlete A is not running the first leg
    and athlete B is not running the last leg in a 4x100 meter relay race --/
theorem relay_race_probability : 
  (total_runners.factorial / (total_runners - team_size).factorial - 
   (total_runners - 1).factorial / (total_runners - team_size).factorial - 
   (total_runners - 1).factorial / (total_runners - team_size).factorial + 
   (total_runners - 2).factorial / (total_runners - team_size + 1).factorial) /
  (total_runners.factorial / (total_runners - team_size).factorial) = 7 / 10 := by
sorry

end relay_race_probability_l3794_379443


namespace product_equals_quadratic_l3794_379491

theorem product_equals_quadratic : ∃ m : ℤ, 72516 * 9999 = m^2 - 5*m + 7 ∧ m = 26926 := by
  sorry

end product_equals_quadratic_l3794_379491


namespace hospital_bed_charge_l3794_379459

theorem hospital_bed_charge 
  (days_in_hospital : ℕ) 
  (specialist_hourly_rate : ℚ) 
  (specialist_time : ℚ) 
  (num_specialists : ℕ) 
  (ambulance_cost : ℚ) 
  (total_bill : ℚ) :
  days_in_hospital = 3 →
  specialist_hourly_rate = 250 →
  specialist_time = 1/4 →
  num_specialists = 2 →
  ambulance_cost = 1800 →
  total_bill = 4625 →
  let daily_bed_charge := (total_bill - num_specialists * specialist_hourly_rate * specialist_time - ambulance_cost) / days_in_hospital
  daily_bed_charge = 900 := by
sorry

end hospital_bed_charge_l3794_379459


namespace watermelon_count_l3794_379446

theorem watermelon_count (seeds_per_watermelon : ℕ) (total_seeds : ℕ) (h1 : seeds_per_watermelon = 100) (h2 : total_seeds = 400) :
  total_seeds / seeds_per_watermelon = 4 := by
  sorry

end watermelon_count_l3794_379446


namespace total_camp_attendance_l3794_379419

/-- The number of kids from Lawrence county who went to camp -/
def lawrence_camp : ℕ := 34044

/-- The number of kids from outside the county who attended the camp -/
def outside_camp : ℕ := 424944

/-- The total number of kids who attended the camp -/
def total_camp : ℕ := lawrence_camp + outside_camp

/-- Theorem stating that the total number of kids who attended the camp is 458988 -/
theorem total_camp_attendance : total_camp = 458988 := by
  sorry

end total_camp_attendance_l3794_379419


namespace cubic_equation_geometric_progression_l3794_379438

theorem cubic_equation_geometric_progression (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
   x^3 + 16*x^2 + a*x + 64 = 0 ∧
   y^3 + 16*y^2 + a*y + 64 = 0 ∧
   z^3 + 16*z^2 + a*z + 64 = 0 ∧
   ∃ q : ℝ, q ≠ 0 ∧ q ≠ 1 ∧ y = x*q ∧ z = y*q) →
  a = 64 :=
by sorry

end cubic_equation_geometric_progression_l3794_379438


namespace smallest_sum_of_pairwise_sums_l3794_379435

theorem smallest_sum_of_pairwise_sums (a b c d : ℝ) (y : ℝ) : 
  let sums := {a + b, a + c, a + d, b + c, b + d, c + d}
  ({170, 305, 270, 255, 320, y} : Set ℝ) = sums →
  (320 ∈ sums) →
  (∀ z ∈ sums, 320 + y ≤ z + y) →
  320 + y = 255 := by
sorry

end smallest_sum_of_pairwise_sums_l3794_379435


namespace seventh_observation_value_l3794_379417

theorem seventh_observation_value
  (n : ℕ) -- number of initial observations
  (initial_avg : ℚ) -- initial average
  (new_avg : ℚ) -- new average after adding one observation
  (h1 : n = 6) -- there are 6 initial observations
  (h2 : initial_avg = 16) -- the initial average is 16
  (h3 : new_avg = initial_avg - 1) -- the new average is decreased by 1
  : (n + 1) * new_avg - n * initial_avg = 9 := by
  sorry

end seventh_observation_value_l3794_379417


namespace irregular_decagon_angle_l3794_379400

/-- Theorem: In a 10-sided polygon where the sum of all interior angles is 1470°,
    and 9 of the angles are equal, the measure of the non-equal angle is 174°. -/
theorem irregular_decagon_angle (n : ℕ) (sum : ℝ) (regular_angle : ℝ) (irregular_angle : ℝ) :
  n = 10 ∧ 
  sum = 1470 ∧
  (n - 1) * regular_angle + irregular_angle = sum ∧
  (n - 1) * regular_angle = (n - 2) * 180 →
  irregular_angle = 174 := by
  sorry

end irregular_decagon_angle_l3794_379400


namespace bruce_payment_l3794_379431

/-- The amount Bruce paid to the shopkeeper for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Bruce paid 1125 to the shopkeeper -/
theorem bruce_payment :
  total_amount 9 70 9 55 = 1125 := by
  sorry

end bruce_payment_l3794_379431


namespace general_drinking_horse_shortest_distance_l3794_379452

/-- The shortest distance for the "General Drinking Horse" problem -/
theorem general_drinking_horse_shortest_distance :
  let camp := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 1}
  let A : ℝ × ℝ := (2, 0)
  let riverbank := {p : ℝ × ℝ | p.1 + p.2 = 3}
  ∃ (B : ℝ × ℝ) (C : ℝ × ℝ),
    B ∈ riverbank ∧ C ∈ camp ∧
    ∀ (B' : ℝ × ℝ) (C' : ℝ × ℝ),
      B' ∈ riverbank → C' ∈ camp →
      Real.sqrt 10 - 1 ≤ dist A B' + dist B' C' :=
sorry

end general_drinking_horse_shortest_distance_l3794_379452


namespace expression_simplification_l3794_379493

theorem expression_simplification (a b c : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a * b^2 = c / a - b) :
  let expr := (a^2 * b^2 / c^2 - 2 / c + 1 / (a^2 * b^2) + 2 * a * b / c^2 - 2 / (a * b * c)) /
               (2 / (a * b) - 2 * a * b / c) /
               (101 / c)
  expr = -1 / 202 := by
  sorry

end expression_simplification_l3794_379493


namespace solid_is_cone_l3794_379432

/-- Represents a three-dimensional solid -/
structure Solid where
  -- Add necessary fields

/-- Represents a view of a solid -/
inductive View
  | Front
  | Side
  | Top

/-- Represents a shape -/
inductive Shape
  | Cone
  | Pyramid
  | Prism
  | Cylinder

/-- Returns true if the given view of the solid is an equilateral triangle -/
def isEquilateralTriangle (s : Solid) (v : View) : Prop :=
  sorry

/-- Returns true if the given view of the solid is a circle with its center -/
def isCircleWithCenter (s : Solid) (v : View) : Prop :=
  sorry

/-- Returns true if the front and side view triangles have equal sides -/
def hasFrontSideEqualSides (s : Solid) : Prop :=
  sorry

/-- Determines the shape of the solid based on its properties -/
def determineShape (s : Solid) : Shape :=
  sorry

theorem solid_is_cone (s : Solid) 
  (h1 : isEquilateralTriangle s View.Front)
  (h2 : isEquilateralTriangle s View.Side)
  (h3 : hasFrontSideEqualSides s)
  (h4 : isCircleWithCenter s View.Top) :
  determineShape s = Shape.Cone :=
sorry

end solid_is_cone_l3794_379432


namespace rate_squares_sum_l3794_379430

theorem rate_squares_sum : ∀ (b j s : ℕ),
  (3 * b + 4 * j + 2 * s = 86) →
  (5 * b + 2 * j + 4 * s = 110) →
  (b * b + j * j + s * s = 3349) :=
by sorry

end rate_squares_sum_l3794_379430


namespace sqrt_equation_solution_l3794_379467

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 12) = 10 → x = 88 := by
  sorry

end sqrt_equation_solution_l3794_379467


namespace equation_solution_l3794_379455

theorem equation_solution : ∃! x : ℚ, (1 / (x + 11) + 1 / (x + 8) = 1 / (x + 12) + 1 / (x + 7)) ∧ x = -19/2 := by
  sorry

end equation_solution_l3794_379455


namespace quadratic_function_comparison_l3794_379414

theorem quadratic_function_comparison : ∀ (y₁ y₂ : ℝ),
  y₁ = -(1:ℝ)^2 + 2 →
  y₂ = -(3:ℝ)^2 + 2 →
  y₁ > y₂ := by
sorry

end quadratic_function_comparison_l3794_379414


namespace pears_juice_calculation_l3794_379456

/-- The amount of pears processed into juice given a total harvest and export percentage -/
def pears_processed_into_juice (total_harvest : ℝ) (export_percentage : ℝ) (juice_percentage : ℝ) : ℝ :=
  total_harvest * (1 - export_percentage) * juice_percentage

theorem pears_juice_calculation (total_harvest : ℝ) (export_percentage : ℝ) (juice_percentage : ℝ) 
  (h1 : total_harvest = 8.5)
  (h2 : export_percentage = 0.3)
  (h3 : juice_percentage = 0.6) :
  pears_processed_into_juice total_harvest export_percentage juice_percentage = 3.57 := by
  sorry

#eval pears_processed_into_juice 8.5 0.3 0.6

end pears_juice_calculation_l3794_379456


namespace additional_blue_tickets_for_bible_l3794_379453

/-- Represents the number of tickets Tom has of each color -/
structure TicketCounts where
  yellow : ℕ
  red : ℕ
  green : ℕ
  blue : ℕ

/-- Represents the conversion rates between ticket colors -/
structure TicketRates where
  yellow_to_red : ℕ
  red_to_green : ℕ
  green_to_blue : ℕ

def calculate_additional_blue_tickets (
  bible_yellow_requirement : ℕ
  ) (rates : TicketRates) (current_tickets : TicketCounts) : ℕ :=
  sorry

theorem additional_blue_tickets_for_bible (
  bible_yellow_requirement : ℕ
  ) (rates : TicketRates) (current_tickets : TicketCounts) :
  bible_yellow_requirement = 20 →
  rates.yellow_to_red = 15 →
  rates.red_to_green = 12 →
  rates.green_to_blue = 10 →
  current_tickets.yellow = 12 →
  current_tickets.red = 8 →
  current_tickets.green = 14 →
  current_tickets.blue = 27 →
  calculate_additional_blue_tickets bible_yellow_requirement rates current_tickets = 13273 :=
by sorry

end additional_blue_tickets_for_bible_l3794_379453


namespace lottery_not_guaranteed_win_l3794_379413

/-- Represents a lottery with a total number of tickets and a winning rate. -/
structure Lottery where
  totalTickets : ℕ
  winningRate : ℝ
  winningRate_pos : winningRate > 0
  winningRate_le_one : winningRate ≤ 1

/-- The probability of not winning with a single ticket. -/
def Lottery.loseProb (l : Lottery) : ℝ := 1 - l.winningRate

/-- The probability of not winning with n tickets. -/
def Lottery.loseProbN (l : Lottery) (n : ℕ) : ℝ := (l.loseProb) ^ n

theorem lottery_not_guaranteed_win (l : Lottery) (h1 : l.totalTickets = 1000000) (h2 : l.winningRate = 0.001) :
  l.loseProbN 1000 > 0 := by sorry

end lottery_not_guaranteed_win_l3794_379413


namespace final_inventory_calculation_l3794_379475

def initial_inventory : ℕ := 4500
def monday_sales : ℕ := 2445
def tuesday_sales : ℕ := 900
def daily_sales_wed_to_sun : ℕ := 50
def days_wed_to_sun : ℕ := 5
def saturday_delivery : ℕ := 650

theorem final_inventory_calculation :
  initial_inventory - 
  (monday_sales + tuesday_sales + daily_sales_wed_to_sun * days_wed_to_sun) + 
  saturday_delivery = 1555 := by
  sorry

end final_inventory_calculation_l3794_379475


namespace area_of_bounded_region_l3794_379499

/-- The equation of the curve -/
def curve_equation (x y : ℝ) : Prop :=
  y^2 + 2*x*y + 30*|x| = 500

/-- The bounded region created by the curve -/
def bounded_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | curve_equation p.1 p.2}

/-- The area of the bounded region -/
noncomputable def area : ℝ := sorry

/-- Theorem stating that the area of the bounded region is 5000/3 -/
theorem area_of_bounded_region : area = 5000/3 := by sorry

end area_of_bounded_region_l3794_379499
