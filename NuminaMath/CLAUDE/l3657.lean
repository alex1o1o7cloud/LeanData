import Mathlib

namespace NUMINAMATH_CALUDE_bag_slips_problem_l3657_365749

theorem bag_slips_problem (total_slips : ℕ) (num1 num2 : ℕ) (expected_value : ℚ) :
  total_slips = 15 →
  num1 = 3 →
  num2 = 8 →
  expected_value = 5 →
  ∃ (slips_with_num1 : ℕ),
    slips_with_num1 ≤ total_slips ∧
    (slips_with_num1 : ℚ) / total_slips * num1 + 
    ((total_slips - slips_with_num1) : ℚ) / total_slips * num2 = expected_value →
    slips_with_num1 = 9 :=
by sorry

end NUMINAMATH_CALUDE_bag_slips_problem_l3657_365749


namespace NUMINAMATH_CALUDE_parabola_directrix_l3657_365789

/-- Given a parabola x² = ay with directrix y = -1/4, prove that a = 1 -/
theorem parabola_directrix (x y a : ℝ) : 
  (x^2 = a * y) →  -- Parabola equation
  (y = -1/4 → a = 1) :=  -- Directrix equation implies a = 1
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3657_365789


namespace NUMINAMATH_CALUDE_gcd_g_x_equals_120_l3657_365736

def g (x : ℤ) : ℤ := (5*x + 7)*(11*x + 3)*(17*x + 8)*(4*x + 5)

theorem gcd_g_x_equals_120 (x : ℤ) (h : ∃ k : ℤ, x = 17280 * k) :
  Nat.gcd (Int.natAbs (g x)) (Int.natAbs x) = 120 := by
  sorry

end NUMINAMATH_CALUDE_gcd_g_x_equals_120_l3657_365736


namespace NUMINAMATH_CALUDE_polynomial_root_product_l3657_365716

theorem polynomial_root_product (k : ℝ) : 
  (∃ (x₁ x₂ x₃ x₄ : ℝ), 
    (x₁^4 - 18*x₁^3 + k*x₁^2 + 200*x₁ - 1984 = 0) ∧
    (x₂^4 - 18*x₂^3 + k*x₂^2 + 200*x₂ - 1984 = 0) ∧
    (x₃^4 - 18*x₃^3 + k*x₃^2 + 200*x₃ - 1984 = 0) ∧
    (x₄^4 - 18*x₄^3 + k*x₄^2 + 200*x₄ - 1984 = 0) ∧
    (x₁ * x₂ = -32 ∨ x₁ * x₃ = -32 ∨ x₁ * x₄ = -32 ∨ 
     x₂ * x₃ = -32 ∨ x₂ * x₄ = -32 ∨ x₃ * x₄ = -32)) →
  k = 86 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_root_product_l3657_365716


namespace NUMINAMATH_CALUDE_roulette_probability_l3657_365781

theorem roulette_probability (p_X p_Y p_Z p_W : ℚ) : 
  p_X = 1/4 → p_Y = 1/3 → p_W = 1/6 → p_X + p_Y + p_Z + p_W = 1 → p_Z = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_roulette_probability_l3657_365781


namespace NUMINAMATH_CALUDE_average_speed_calculation_toms_trip_average_speed_l3657_365719

theorem average_speed_calculation (total_distance : Real) (first_part_distance : Real) 
  (first_part_speed : Real) (second_part_speed : Real) : Real :=
  let second_part_distance := total_distance - first_part_distance
  let first_part_time := first_part_distance / first_part_speed
  let second_part_time := second_part_distance / second_part_speed
  let total_time := first_part_time + second_part_time
  total_distance / total_time

theorem toms_trip_average_speed : 
  average_speed_calculation 60 12 24 48 = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_toms_trip_average_speed_l3657_365719


namespace NUMINAMATH_CALUDE_circle_intersection_m_range_l3657_365729

theorem circle_intersection_m_range (m : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 - 2*x + 10*y + 1 = 0 ∧ x^2 + y^2 - 2*x + 2*y - m = 0) →
  -1 < m ∧ m < 79 := by
sorry

end NUMINAMATH_CALUDE_circle_intersection_m_range_l3657_365729


namespace NUMINAMATH_CALUDE_negation_equivalence_l3657_365796

-- Define the set S
variable (S : Set ℝ)

-- Define the original statement
def original_statement : Prop :=
  ∀ x ∈ S, |x| ≥ 2

-- Define the negation of the original statement
def negation_statement : Prop :=
  ∃ x ∈ S, |x| < 2

-- Theorem stating the equivalence
theorem negation_equivalence :
  ¬(original_statement S) ↔ negation_statement S :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3657_365796


namespace NUMINAMATH_CALUDE_smallest_class_size_l3657_365752

theorem smallest_class_size (n : ℕ) (scores : Fin n → ℕ) : 
  (∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    scores i = 100 ∧ scores j = 100 ∧ scores k = 100) →
  (∀ i : Fin n, scores i ≥ 70) →
  (∀ i : Fin n, scores i ≤ 100) →
  (Finset.sum Finset.univ scores / n = 85) →
  n ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_class_size_l3657_365752


namespace NUMINAMATH_CALUDE_train_crossing_time_l3657_365705

/-- Calculates the time for a train to cross a signal pole given its length and the time it takes to cross a platform of equal length. -/
theorem train_crossing_time (train_length platform_length : ℝ) (platform_crossing_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 300)
  (h3 : platform_crossing_time = 36) :
  let total_distance := train_length + platform_length
  let train_speed := total_distance / platform_crossing_time
  train_length / train_speed = 18 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3657_365705


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3657_365707

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3657_365707


namespace NUMINAMATH_CALUDE_company_reduction_l3657_365757

/-- The original number of employees before reductions -/
def original_employees : ℕ := 344

/-- The number of employees after both reductions -/
def final_employees : ℕ := 263

/-- The reduction factor after the first quarter -/
def first_reduction : ℚ := 9/10

/-- The reduction factor after the second quarter -/
def second_reduction : ℚ := 85/100

theorem company_reduction :
  ⌊(second_reduction * first_reduction * original_employees : ℚ)⌋ = final_employees := by
  sorry

end NUMINAMATH_CALUDE_company_reduction_l3657_365757


namespace NUMINAMATH_CALUDE_sum_34_47_in_base4_l3657_365733

/-- Converts a natural number from base 10 to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Adds two numbers in base 10 and returns the result in base 4 -/
def addAndConvertToBase4 (a b : ℕ) : List ℕ :=
  toBase4 (a + b)

theorem sum_34_47_in_base4 :
  addAndConvertToBase4 34 47 = [1, 1, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_sum_34_47_in_base4_l3657_365733


namespace NUMINAMATH_CALUDE_financial_equation_proof_l3657_365788

-- Define variables
variable (q v j p : ℝ)

-- Define the theorem
theorem financial_equation_proof :
  (3 * q - v = 8000) →
  (q = 4) →
  (v = 4 + 50 * j) →
  (p = 2669 + (50/3) * j) := by
sorry

end NUMINAMATH_CALUDE_financial_equation_proof_l3657_365788


namespace NUMINAMATH_CALUDE_jack_classic_collection_l3657_365755

/-- The number of books each author has in Jack's classic collection -/
def books_per_author (total_books : ℕ) (num_authors : ℕ) : ℕ :=
  total_books / num_authors

/-- Theorem stating that each author has 33 books in Jack's classic collection -/
theorem jack_classic_collection :
  let total_books : ℕ := 198
  let num_authors : ℕ := 6
  books_per_author total_books num_authors = 33 := by
sorry

end NUMINAMATH_CALUDE_jack_classic_collection_l3657_365755


namespace NUMINAMATH_CALUDE_age_calculation_l3657_365731

/-- Given the ages and relationships of several people, calculate their current ages. -/
theorem age_calculation (tim john james lisa kate michael anna : ℚ) : 
  tim = 79 ∧ 
  james + 23 = john + 35 ∧
  tim = 2 * john - 5 ∧
  lisa = (james + tim) / 2 ∧
  kate = james + 4 ∧
  michael = 3 * (lisa - kate) ∧
  anna = michael - 7 →
  james = 30 ∧ 
  lisa = 54.5 ∧ 
  kate = 34 ∧ 
  michael = 61.5 ∧ 
  anna = 54.5 := by
  sorry


end NUMINAMATH_CALUDE_age_calculation_l3657_365731


namespace NUMINAMATH_CALUDE_lottery_expected_profit_l3657_365764

-- Define the lottery ticket parameters
def ticket_cost : ℝ := 2
def winning_probability : ℝ := 0.01
def prize : ℝ := 50

-- Define the expected profit function
def expected_profit (cost winning_prob prize : ℝ) : ℝ :=
  winning_prob * prize - cost

-- Theorem statement
theorem lottery_expected_profit :
  expected_profit ticket_cost winning_probability prize = -1.5 := by
  sorry

end NUMINAMATH_CALUDE_lottery_expected_profit_l3657_365764


namespace NUMINAMATH_CALUDE_overtime_pay_rate_l3657_365772

def regular_pay_rate : ℝ := 10
def regular_hours : ℝ := 40
def total_hours : ℝ := 60
def total_earnings : ℝ := 700

theorem overtime_pay_rate :
  ∃ (overtime_rate : ℝ),
    regular_pay_rate * regular_hours +
    overtime_rate * (total_hours - regular_hours) =
    total_earnings ∧
    overtime_rate = 15 :=
by sorry

end NUMINAMATH_CALUDE_overtime_pay_rate_l3657_365772


namespace NUMINAMATH_CALUDE_arccos_cos_nine_l3657_365786

theorem arccos_cos_nine : Real.arccos (Real.cos 9) = 9 - 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_nine_l3657_365786


namespace NUMINAMATH_CALUDE_earliest_time_84_degrees_l3657_365776

/-- Temperature function representing the temperature in Austin, TX on a summer day -/
def T (t : ℝ) : ℝ := -t^2 + 14*t + 40

/-- The earliest positive real solution to the temperature equation when it equals 84 degrees -/
theorem earliest_time_84_degrees :
  ∀ t : ℝ, t > 0 → T t = 84 → t ≥ 22 :=
by sorry

end NUMINAMATH_CALUDE_earliest_time_84_degrees_l3657_365776


namespace NUMINAMATH_CALUDE_parametric_line_unique_constants_l3657_365783

/-- A line passing through two points with given parametric equations -/
structure ParametricLine where
  a : ℝ
  b : ℝ
  passes_through_P : 0 = 0 + a ∧ 2 = (b/2) * 0 + 1
  passes_through_Q : 1 = 1 + a ∧ 3 = (b/2) * 1 + 1

/-- Theorem stating the unique values of a and b for the given line -/
theorem parametric_line_unique_constants (l : ParametricLine) : l.a = -1 ∧ l.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_parametric_line_unique_constants_l3657_365783


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l3657_365792

theorem no_positive_integer_solutions : 
  ¬∃ (x y : ℕ+), x^2 + y^2 + x = 2 * x^3 := by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l3657_365792


namespace NUMINAMATH_CALUDE_trig_product_equality_l3657_365768

theorem trig_product_equality : 
  Real.sin (-15 * Real.pi / 6) * Real.cos (20 * Real.pi / 3) * Real.tan (-7 * Real.pi / 6) = Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_trig_product_equality_l3657_365768


namespace NUMINAMATH_CALUDE_triangle_side_length_l3657_365799

/-- Given a triangle ABC with area 3√3/4, side a = 3, and angle B = π/3, prove that side b = √7 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  (1/2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 4) →  -- Area formula
  (a = 3) →  -- Given side length
  (B = π/3) →  -- Given angle
  (b^2 = a^2 + c^2 - 2*a*c*Real.cos B) →  -- Law of cosines
  (b = Real.sqrt 7) := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3657_365799


namespace NUMINAMATH_CALUDE_total_items_for_58_slices_l3657_365775

/-- Given the number of slices of bread, calculate the total number of items -/
def totalItems (slices : ℕ) : ℕ :=
  let milk := slices - 18
  let cookies := slices + 27
  slices + milk + cookies

theorem total_items_for_58_slices :
  totalItems 58 = 183 := by
  sorry

end NUMINAMATH_CALUDE_total_items_for_58_slices_l3657_365775


namespace NUMINAMATH_CALUDE_polygon_sides_greater_than_diagonals_l3657_365741

theorem polygon_sides_greater_than_diagonals (n : ℕ) (d : ℕ) : 
  (n ≥ 3 ∧ d = n * (n - 3) / 2) → (n > d ↔ n = 3 ∨ n = 4) := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_greater_than_diagonals_l3657_365741


namespace NUMINAMATH_CALUDE_slope_determines_y_coordinate_l3657_365706

/-- Given two points R and S in a coordinate plane, if the slope of the line through R and S
    is equal to -4/3, then the y-coordinate of S is -8/3. -/
theorem slope_determines_y_coordinate (x_R y_R x_S : ℚ) : 
  let R : ℚ × ℚ := (x_R, y_R)
  let S : ℚ × ℚ := (x_S, y_S)
  x_R = -3 →
  y_R = 8 →
  x_S = 5 →
  (y_S - y_R) / (x_S - x_R) = -4/3 →
  y_S = -8/3 := by
sorry

end NUMINAMATH_CALUDE_slope_determines_y_coordinate_l3657_365706


namespace NUMINAMATH_CALUDE_second_game_points_l3657_365747

/-- The number of points scored in each of the four games -/
structure GamePoints where
  game1 : ℕ
  game2 : ℕ
  game3 : ℕ
  game4 : ℕ

/-- The conditions of the basketball game scenario -/
def basketball_scenario (p : GamePoints) : Prop :=
  p.game1 = 10 ∧
  p.game3 = 6 ∧
  p.game4 = (p.game1 + p.game2 + p.game3) / 3 ∧
  p.game1 + p.game2 + p.game3 + p.game4 = 40

theorem second_game_points :
  ∃ p : GamePoints, basketball_scenario p ∧ p.game2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_second_game_points_l3657_365747


namespace NUMINAMATH_CALUDE_prime_product_sum_difference_l3657_365739

theorem prime_product_sum_difference : ∃ x y : ℕ, 
  x.Prime ∧ y.Prime ∧ 
  x ≠ y ∧ 
  20 < x ∧ x < 40 ∧ 
  20 < y ∧ y < 40 ∧ 
  x * y - (x + y) = 899 := by
  sorry

end NUMINAMATH_CALUDE_prime_product_sum_difference_l3657_365739


namespace NUMINAMATH_CALUDE_p_or_q_is_true_l3657_365797

-- Define proposition p
def p (a b : ℝ) : Prop := a^2 + b^2 < 0

-- Define proposition q
def q (a b : ℝ) : Prop := (a - 2)^2 + |b - 3| ≥ 0

-- Theorem statement
theorem p_or_q_is_true :
  (∀ a b : ℝ, ¬(p a b)) ∧ (∀ a b : ℝ, q a b) → ∀ a b : ℝ, p a b ∨ q a b :=
by sorry

end NUMINAMATH_CALUDE_p_or_q_is_true_l3657_365797


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3657_365724

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1

-- Theorem 1
theorem problem_1 (a : ℝ) :
  (∀ x ∈ Set.Icc 1 3, g a x ∈ Set.Icc 0 4) ∧
  (∃ x ∈ Set.Icc 1 3, g a x = 0) ∧
  (∃ x ∈ Set.Icc 1 3, g a x = 4) →
  a = 1 := by sorry

-- Theorem 2
theorem problem_2 (k : ℝ) :
  (∀ x ≥ 1, g 1 (2^x) - k * 4^x ≥ 0) →
  k ≤ 1/4 := by sorry

-- Theorem 3
theorem problem_3 (k : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (g 1 (|2^x₁ - 1|) / |2^x₁ - 1| + k * (2 / |2^x₁ - 1|) - 3*k = 0) ∧
    (g 1 (|2^x₂ - 1|) / |2^x₂ - 1| + k * (2 / |2^x₂ - 1|) - 3*k = 0) ∧
    (g 1 (|2^x₃ - 1|) / |2^x₃ - 1| + k * (2 / |2^x₃ - 1|) - 3*k = 0)) →
  k > 0 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3657_365724


namespace NUMINAMATH_CALUDE_a_worked_days_proof_l3657_365795

/-- The number of days A needs to complete the entire work alone -/
def a_complete_days : ℝ := 40

/-- The number of days B needs to complete the entire work alone -/
def b_complete_days : ℝ := 60

/-- The number of days B needs to complete the remaining work after A leaves -/
def b_remaining_days : ℝ := 45

/-- The number of days A worked before leaving -/
def a_worked_days : ℝ := 10

theorem a_worked_days_proof :
  (1 / a_complete_days * a_worked_days) + (b_remaining_days / b_complete_days) = 1 :=
sorry

end NUMINAMATH_CALUDE_a_worked_days_proof_l3657_365795


namespace NUMINAMATH_CALUDE_program_attendance_l3657_365727

/-- The total number of people present at the program -/
def total_people (parents pupils teachers staff family_members : ℕ) : ℕ :=
  parents + pupils + teachers + staff + family_members

/-- The number of family members accompanying the pupils -/
def accompanying_family_members (pupils : ℕ) : ℕ :=
  (pupils / 6) * 2

theorem program_attendance : 
  let parents : ℕ := 83
  let pupils : ℕ := 956
  let teachers : ℕ := 154
  let staff : ℕ := 27
  let family_members : ℕ := accompanying_family_members pupils
  total_people parents pupils teachers staff family_members = 1379 := by
sorry

end NUMINAMATH_CALUDE_program_attendance_l3657_365727


namespace NUMINAMATH_CALUDE_book_loss_percentage_l3657_365701

/-- If the cost price of 8 books equals the selling price of 16 books, then the loss percentage is 50%. -/
theorem book_loss_percentage (C S : ℝ) (h : 8 * C = 16 * S) : 
  (C - S) / C * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_book_loss_percentage_l3657_365701


namespace NUMINAMATH_CALUDE_two_valid_configurations_l3657_365714

/-- Represents a quadrant in the yard --/
inductive Quadrant
| I
| II
| III
| IV

/-- Represents a configuration of apple trees in the yard --/
def Configuration := Quadrant → Nat

/-- Checks if a configuration is valid (total of 4 trees) --/
def is_valid_configuration (c : Configuration) : Prop :=
  c Quadrant.I + c Quadrant.II + c Quadrant.III + c Quadrant.IV = 4

/-- Checks if a configuration has equal trees on both sides of each path --/
def is_balanced_configuration (c : Configuration) : Prop :=
  c Quadrant.I + c Quadrant.II = c Quadrant.III + c Quadrant.IV ∧
  c Quadrant.I + c Quadrant.IV = c Quadrant.II + c Quadrant.III ∧
  c Quadrant.I + c Quadrant.III = c Quadrant.II + c Quadrant.IV

/-- Theorem: There exist at least two different valid and balanced configurations --/
theorem two_valid_configurations : ∃ (c1 c2 : Configuration),
  c1 ≠ c2 ∧
  is_valid_configuration c1 ∧
  is_valid_configuration c2 ∧
  is_balanced_configuration c1 ∧
  is_balanced_configuration c2 :=
sorry

end NUMINAMATH_CALUDE_two_valid_configurations_l3657_365714


namespace NUMINAMATH_CALUDE_sqrt_3_minus_pi_squared_l3657_365780

theorem sqrt_3_minus_pi_squared (π : ℝ) (h : π > 3) : 
  Real.sqrt ((3 - π)^2) = π - 3 := by
sorry

end NUMINAMATH_CALUDE_sqrt_3_minus_pi_squared_l3657_365780


namespace NUMINAMATH_CALUDE_orange_picking_theorem_l3657_365712

/-- The total number of oranges picked over three days -/
def total_oranges (day1 day2 day3 : ℕ) : ℕ := day1 + day2 + day3

/-- Theorem stating the total number of oranges picked over three days -/
theorem orange_picking_theorem (day1 day2 day3 : ℕ) 
  (h1 : day1 = 100)
  (h2 : day2 = 3 * day1)
  (h3 : day3 = 70) :
  total_oranges day1 day2 day3 = 470 := by
  sorry


end NUMINAMATH_CALUDE_orange_picking_theorem_l3657_365712


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_l3657_365718

theorem line_ellipse_intersection (m : ℝ) : 
  (∃ x y : ℝ, 4 * x^2 + 9 * y^2 = 36 ∧ y = m * x + 3) → 
  m ≤ -Real.sqrt 5 / 3 ∨ m ≥ Real.sqrt 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_l3657_365718


namespace NUMINAMATH_CALUDE_emily_egg_collection_l3657_365742

/-- The number of baskets Emily used -/
def num_baskets : ℕ := 303

/-- The number of eggs in each basket -/
def eggs_per_basket : ℕ := 28

/-- The total number of eggs Emily collected -/
def total_eggs : ℕ := num_baskets * eggs_per_basket

theorem emily_egg_collection : total_eggs = 8484 := by
  sorry

end NUMINAMATH_CALUDE_emily_egg_collection_l3657_365742


namespace NUMINAMATH_CALUDE_kerman_triple_49_64_15_l3657_365726

/-- Definition of a Kerman triple -/
def is_kerman_triple (a b x : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ x > 0 ∧ Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) = x

/-- Theorem: (49, 64, 15) is a Kerman triple -/
theorem kerman_triple_49_64_15 :
  is_kerman_triple 49 64 15 := by
  sorry

end NUMINAMATH_CALUDE_kerman_triple_49_64_15_l3657_365726


namespace NUMINAMATH_CALUDE_angle_1303_equiv_neg137_l3657_365769

-- Define a function to represent angles with the same terminal side
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ n : ℤ, β = α + n * 360

-- State the theorem
theorem angle_1303_equiv_neg137 :
  same_terminal_side 1303 (-137) :=
sorry

end NUMINAMATH_CALUDE_angle_1303_equiv_neg137_l3657_365769


namespace NUMINAMATH_CALUDE_books_grabbed_l3657_365759

/-- Calculates the number of books Henry grabbed from the "free to a good home" box -/
theorem books_grabbed (initial_books : ℕ) (donated_boxes : ℕ) (books_per_box : ℕ) 
  (room_books : ℕ) (coffee_table_books : ℕ) (kitchen_books : ℕ) (final_books : ℕ) : 
  initial_books = 99 →
  donated_boxes = 3 →
  books_per_box = 15 →
  room_books = 21 →
  coffee_table_books = 4 →
  kitchen_books = 18 →
  final_books = 23 →
  final_books - (initial_books - (donated_boxes * books_per_box + room_books + coffee_table_books + kitchen_books)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_books_grabbed_l3657_365759


namespace NUMINAMATH_CALUDE_marble_theorem_l3657_365730

def marble_problem (adam mary greg john sarah peter emily : ℚ) : Prop :=
  adam = 29 ∧
  mary = adam - 11 ∧
  greg = adam + 14 ∧
  john = 2 * mary ∧
  sarah = greg - 7 ∧
  peter = 3 * adam ∧
  emily = (mary + greg) / 2 ∧
  peter + john + sarah - (adam + mary + greg + emily) = 38.5

theorem marble_theorem :
  ∀ adam mary greg john sarah peter emily : ℚ,
  marble_problem adam mary greg john sarah peter emily :=
by
  sorry

end NUMINAMATH_CALUDE_marble_theorem_l3657_365730


namespace NUMINAMATH_CALUDE_vector_dot_product_and_triangle_sides_l3657_365748

/-- Given vectors p and q, prove the range of t and the sum of b and c. -/
theorem vector_dot_product_and_triangle_sides (A : ℝ) (t : ℝ) (b c : ℝ) : 
  let p : ℝ × ℝ := (Real.sin A, Real.cos A)
  let q : ℝ × ℝ := (Real.sqrt 3 * Real.cos A, -Real.cos A)
  0 < A → A < Real.pi / 2 → q ≠ (0, 0) →
  p.1 * q.1 + p.2 * q.2 = t - 1/2 →
  ((-1/2 < t ∧ t ≤ 1/2) ∨ t = 1) ∧
  (∃ (B C : ℝ), 
    Real.sqrt 3 / 2 = Real.sin B → 
    p.1 / q.1 = p.2 / q.2 → 
    Real.sqrt 3 / 2 < b + c ∧ b + c ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_vector_dot_product_and_triangle_sides_l3657_365748


namespace NUMINAMATH_CALUDE_vectors_opposite_direction_l3657_365785

/-- Given non-zero vectors a and b satisfying a + 4b = 0, prove that the directions of a and b are opposite. -/
theorem vectors_opposite_direction {n : Type*} [NormedAddCommGroup n] [NormedSpace ℝ n] 
  (a b : n) (ha : a ≠ 0) (hb : b ≠ 0) (h : a + 4 • b = 0) : 
  ∃ (k : ℝ), k < 0 ∧ a = k • b :=
sorry

end NUMINAMATH_CALUDE_vectors_opposite_direction_l3657_365785


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l3657_365740

/-- The radius of a circle tangent to two semicircles and a quarter circle arc -/
theorem tangent_circle_radius (r : ℝ) : r = (9 - 3 * Real.sqrt 2) / 7 :=
  by
  -- Given:
  -- Quarter circle AOB with center O and radius 1
  -- A1 and B1 are first trisection points of radii OA and OB
  -- Semicircles with diameters AA1 and BB1 (2/3 of the unit radius)
  -- Circle k with radius r is tangent to these semicircles and the arc AB
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l3657_365740


namespace NUMINAMATH_CALUDE_binomial_product_l3657_365737

theorem binomial_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_l3657_365737


namespace NUMINAMATH_CALUDE_seven_valid_configurations_l3657_365778

/-- Represents a square piece --/
structure Square :=
  (label : Char)

/-- Represents the T-shaped figure --/
structure TShape

/-- Represents a configuration of squares added to the T-shape --/
structure Configuration :=
  (square1 : Square)
  (square2 : Square)

/-- Checks if a configuration can be folded into a closed cubical box --/
def can_fold_into_cube (config : Configuration) : Prop :=
  sorry

/-- The set of all possible configurations --/
def all_configurations (squares : Finset Square) : Finset Configuration :=
  sorry

/-- The set of valid configurations that can be folded into a cube --/
def valid_configurations (squares : Finset Square) : Finset Configuration :=
  sorry

theorem seven_valid_configurations :
  ∀ (t : TShape) (squares : Finset Square),
    (Finset.card squares = 8) →
    (Finset.card (valid_configurations squares) = 7) :=
  sorry

end NUMINAMATH_CALUDE_seven_valid_configurations_l3657_365778


namespace NUMINAMATH_CALUDE_min_cost_at_optimal_distance_l3657_365745

noncomputable def f (x : ℝ) : ℝ := (1/2) * (x + 5)^2 + 1000 / (x + 5)

theorem min_cost_at_optimal_distance :
  ∃ (x : ℝ), 2 ≤ x ∧ x ≤ 8 ∧
  (∀ y : ℝ, 2 ≤ y ∧ y ≤ 8 → f y ≥ f x) ∧
  x = 5 ∧ f x = 150 := by
sorry

end NUMINAMATH_CALUDE_min_cost_at_optimal_distance_l3657_365745


namespace NUMINAMATH_CALUDE_base_representation_of_500_l3657_365728

theorem base_representation_of_500 :
  ∃! b : ℕ, 
    b > 1 ∧ 
    (∃ (a₁ a₂ a₃ a₄ a₅ : ℕ), 
      a₁ < b ∧ a₂ < b ∧ a₃ < b ∧ a₄ < b ∧ a₅ < b ∧
      500 = a₁ * b^4 + a₂ * b^3 + a₃ * b^2 + a₄ * b + a₅) ∧
    b^4 ≤ 500 ∧ 
    500 < b^5 :=
by sorry

end NUMINAMATH_CALUDE_base_representation_of_500_l3657_365728


namespace NUMINAMATH_CALUDE_veggie_servings_per_week_l3657_365784

/-- The number of veggie servings eaten in one week -/
def veggieServingsPerWeek (dailyServings : ℕ) (daysInWeek : ℕ) : ℕ :=
  dailyServings * daysInWeek

/-- Theorem: Given 3 servings daily and 7 days in a week, the total veggie servings per week is 21 -/
theorem veggie_servings_per_week :
  veggieServingsPerWeek 3 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_veggie_servings_per_week_l3657_365784


namespace NUMINAMATH_CALUDE_largest_prime_divisor_101010101_base5_l3657_365793

theorem largest_prime_divisor_101010101_base5 :
  let n : ℕ := 5^8 + 5^6 + 5^4 + 5^2 + 1
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ (∀ q : ℕ, Nat.Prime q → q ∣ n → q ≤ p) ∧ p = 601 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_101010101_base5_l3657_365793


namespace NUMINAMATH_CALUDE_intersection_in_interval_l3657_365758

theorem intersection_in_interval :
  ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < 1 ∧ x₀^3 = (1/2)^x₀ := by sorry

end NUMINAMATH_CALUDE_intersection_in_interval_l3657_365758


namespace NUMINAMATH_CALUDE_monotonic_function_upper_bound_l3657_365790

open Real

/-- A monotonic function on (0, +∞) satisfying certain conditions -/
def MonotonicFunction (f : ℝ → ℝ) : Prop :=
  Monotone f ∧ 
  (∀ x > 0, f (f x - exp x + x) = exp 1) ∧
  (∀ x > 0, DifferentiableAt ℝ f x)

/-- The theorem stating the upper bound of a -/
theorem monotonic_function_upper_bound 
  (f : ℝ → ℝ) 
  (hf : MonotonicFunction f) 
  (h : ∀ x > 0, f x + deriv f x ≥ (a : ℝ) * x) :
  a ≤ 2 * exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_function_upper_bound_l3657_365790


namespace NUMINAMATH_CALUDE_binomial_distribution_parameters_l3657_365725

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The mean of a binomial distribution -/
def mean (X : BinomialDistribution) : ℝ := X.n * X.p

/-- The variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_distribution_parameters :
  ∃ X : BinomialDistribution, mean X = 15 ∧ variance X = 12 ∧ X.n = 60 ∧ X.p = 0.25 := by sorry

end NUMINAMATH_CALUDE_binomial_distribution_parameters_l3657_365725


namespace NUMINAMATH_CALUDE_estimate_population_size_l3657_365777

theorem estimate_population_size (sample1 : ℕ) (sample2 : ℕ) (overlap : ℕ) (total : ℕ) : 
  sample1 = 80 → sample2 = 100 → overlap = 20 → 
  (sample1 : ℝ) / total * ((sample2 : ℝ) / total) = (overlap : ℝ) / total → 
  total = 400 := by
sorry

end NUMINAMATH_CALUDE_estimate_population_size_l3657_365777


namespace NUMINAMATH_CALUDE_factor_implies_coefficients_l3657_365721

/-- If (x + 5) is a factor of x^4 - mx^3 + nx^2 - px + q, then m = 0, n = 0, p = 0, and q = -625 -/
theorem factor_implies_coefficients (m n p q : ℝ) : 
  (∀ x : ℝ, (x + 5) ∣ (x^4 - m*x^3 + n*x^2 - p*x + q)) →
  (m = 0 ∧ n = 0 ∧ p = 0 ∧ q = -625) := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_coefficients_l3657_365721


namespace NUMINAMATH_CALUDE_product_of_roots_l3657_365798

theorem product_of_roots (x : ℝ) (hx : x + 16 / x = 12) : 
  ∃ y : ℝ, y + 16 / y = 12 ∧ x * y = 32 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l3657_365798


namespace NUMINAMATH_CALUDE_johns_money_l3657_365709

/-- Given that John needs a total amount of money and still needs some more,
    prove that the amount he already has is the difference between the total needed and the amount still needed. -/
theorem johns_money (total_needed : ℚ) (still_needed : ℚ) (already_has : ℚ) :
  total_needed = 2.5 →
  still_needed = 1.75 →
  already_has = total_needed - still_needed →
  already_has = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_johns_money_l3657_365709


namespace NUMINAMATH_CALUDE_polynomial_relationship_l3657_365723

def x : Fin 5 → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | 4 => 5

def y : Fin 5 → ℕ
  | 0 => 1
  | 1 => 4
  | 2 => 9
  | 3 => 16
  | 4 => 25

theorem polynomial_relationship : ∀ i : Fin 5, y i = (x i) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_relationship_l3657_365723


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l3657_365715

-- Define the polynomial
def f (x : ℂ) : ℂ := x^4 + 10*x^3 + 20*x^2 + 15*x + 6

-- Define the roots
axiom p : ℂ
axiom q : ℂ
axiom r : ℂ
axiom s : ℂ

-- Axiom that p, q, r, s are roots of f
axiom root_p : f p = 0
axiom root_q : f q = 0
axiom root_r : f r = 0
axiom root_s : f s = 0

-- The theorem to prove
theorem root_sum_reciprocals :
  1 / (p * q) + 1 / (p * r) + 1 / (p * s) + 1 / (q * r) + 1 / (q * s) + 1 / (r * s) = -10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l3657_365715


namespace NUMINAMATH_CALUDE_three_digit_powers_of_two_l3657_365735

theorem three_digit_powers_of_two (n : ℕ) : 
  (∃ k, 100 ≤ 2^k ∧ 2^k ≤ 999) ↔ (n = 7 ∨ n = 8 ∨ n = 9) :=
sorry

end NUMINAMATH_CALUDE_three_digit_powers_of_two_l3657_365735


namespace NUMINAMATH_CALUDE_flower_pots_on_path_l3657_365762

/-- Calculates the number of flower pots on a path -/
def flowerPots (pathLength : ℕ) (interval : ℕ) : ℕ :=
  pathLength / interval + 1

/-- Theorem: On a 15-meter path with flower pots every 3 meters, there are 6 flower pots -/
theorem flower_pots_on_path : flowerPots 15 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_flower_pots_on_path_l3657_365762


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l3657_365720

theorem lcm_gcf_problem (n : ℕ+) : 
  Nat.lcm n 14 = 56 → Nat.gcd n 14 = 12 → n = 48 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l3657_365720


namespace NUMINAMATH_CALUDE_trig_inequalities_l3657_365751

theorem trig_inequalities :
  (Real.cos (3 * Real.pi / 5) > Real.cos (-4 * Real.pi / 5)) ∧
  (Real.sin (Real.pi / 10) < Real.cos (Real.pi / 10)) := by
  sorry

end NUMINAMATH_CALUDE_trig_inequalities_l3657_365751


namespace NUMINAMATH_CALUDE_max_value_T_l3657_365713

theorem max_value_T (a b c : ℝ) (ha : 1 ≤ a) (ha' : a ≤ 2) 
                     (hb : 1 ≤ b) (hb' : b ≤ 2)
                     (hc : 1 ≤ c) (hc' : c ≤ 2) : 
  (∃ (x y z : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 1 ≤ y ∧ y ≤ 2) (hz : 1 ≤ z ∧ z ≤ 2), 
    (x - y)^2018 + (y - z)^2018 + (z - x)^2018 = 2) ∧ 
  (∀ (x y z : ℝ), 1 ≤ x ∧ x ≤ 2 → 1 ≤ y ∧ y ≤ 2 → 1 ≤ z ∧ z ≤ 2 → 
    (x - y)^2018 + (y - z)^2018 + (z - x)^2018 ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_T_l3657_365713


namespace NUMINAMATH_CALUDE_barry_vitamin_d3_serving_size_l3657_365794

/-- Calculates the daily serving size of capsules given the total number of days,
    capsules per bottle, and number of bottles. -/
def daily_serving_size (days : ℕ) (capsules_per_bottle : ℕ) (bottles : ℕ) : ℕ :=
  (capsules_per_bottle * bottles) / days

theorem barry_vitamin_d3_serving_size :
  let days : ℕ := 180
  let capsules_per_bottle : ℕ := 60
  let bottles : ℕ := 6
  daily_serving_size days capsules_per_bottle bottles = 2 := by
  sorry

end NUMINAMATH_CALUDE_barry_vitamin_d3_serving_size_l3657_365794


namespace NUMINAMATH_CALUDE_min_overlap_percentage_l3657_365750

theorem min_overlap_percentage (laptop_users smartphone_users : ℚ) 
  (h1 : laptop_users = 90/100) 
  (h2 : smartphone_users = 80/100) : 
  (laptop_users + smartphone_users - 1 : ℚ) ≥ 70/100 := by
  sorry

end NUMINAMATH_CALUDE_min_overlap_percentage_l3657_365750


namespace NUMINAMATH_CALUDE_total_yells_is_sixty_l3657_365722

/-- Represents the number of times Missy yells at her dogs -/
structure DogYells where
  obedient : ℕ
  stubborn : ℕ

/-- The relationship between yells at obedient and stubborn dogs -/
def stubborn_to_obedient_ratio : ℕ := 4

/-- The number of times Missy yells at her obedient dog -/
def obedient_yells : ℕ := 12

/-- Calculates the total number of yells based on the given conditions -/
def total_yells (yells : DogYells) : ℕ :=
  yells.obedient + yells.stubborn

/-- Theorem stating that the total number of yells is 60 -/
theorem total_yells_is_sixty :
  ∃ (yells : DogYells),
    yells.obedient = obedient_yells ∧
    yells.stubborn = stubborn_to_obedient_ratio * obedient_yells ∧
    total_yells yells = 60 := by
  sorry


end NUMINAMATH_CALUDE_total_yells_is_sixty_l3657_365722


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l3657_365753

theorem rectangular_box_volume (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : ∃ (k : ℕ), k > 0 ∧ a = 2 * k ∧ b = 4 * k ∧ c = 5 * k) :
  a * b * c = 320 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l3657_365753


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_35_degree_angle_l3657_365711

/-- The degree measure of the supplement of the complement of a 35-degree angle is 125°. -/
theorem supplement_of_complement_of_35_degree_angle : 
  let angle : ℝ := 35
  let complement := 90 - angle
  let supplement := 180 - complement
  supplement = 125 := by sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_35_degree_angle_l3657_365711


namespace NUMINAMATH_CALUDE_a_value_when_A_equals_B_l3657_365734

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + 1 = 0}

-- Define the set B
def B : Set ℝ := {1, 2}

-- Theorem statement
theorem a_value_when_A_equals_B (a : ℝ) : A a = B → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_a_value_when_A_equals_B_l3657_365734


namespace NUMINAMATH_CALUDE_collinear_points_ratio_l3657_365770

/-- Given four collinear points E, F, G, H in that order, with EF = 3, FG = 6, and EH = 20,
    prove that the ratio of EG to FH is 9/17. -/
theorem collinear_points_ratio (E F G H : ℝ) : 
  (F - E = 3) → (G - F = 6) → (H - E = 20) → 
  (E < F) → (F < G) → (G < H) →
  (G - E) / (H - F) = 9 / 17 := by
sorry

end NUMINAMATH_CALUDE_collinear_points_ratio_l3657_365770


namespace NUMINAMATH_CALUDE_intersection_length_l3657_365782

-- Define the line l passing through A(0,1) with slope k
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

-- Define the circle C: (x-2)^2 + (y-3)^2 = 1
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

-- Define the condition that line l intersects circle C at points M and N
def intersects (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧ 
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Define the dot product condition
def dot_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 12

-- Main theorem
theorem intersection_length (k : ℝ) :
  intersects k →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧ 
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    dot_product_condition x₁ y₁ x₂ y₂) →
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧ 
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_intersection_length_l3657_365782


namespace NUMINAMATH_CALUDE_smallest_n_with_three_triples_l3657_365738

/-- Function that counts the number of distinct ordered triples (a, b, c) of positive integers
    such that a^2 + b^2 + c^2 = n -/
def g (n : ℕ) : ℕ := 
  (Finset.filter (fun t : ℕ × ℕ × ℕ => 
    t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0 ∧ 
    t.1^2 + t.2.1^2 + t.2.2^2 = n) (Finset.product (Finset.range n) (Finset.product (Finset.range n) (Finset.range n)))).card

/-- 11 is the smallest positive integer n for which g(n) = 3 -/
theorem smallest_n_with_three_triples : 
  (∀ m : ℕ, m > 0 ∧ m < 11 → g m ≠ 3) ∧ g 11 = 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_three_triples_l3657_365738


namespace NUMINAMATH_CALUDE_girls_in_school_l3657_365700

theorem girls_in_school (total_students sample_size girls_sampled : ℕ) 
  (h1 : total_students = 1600)
  (h2 : sample_size = 200)
  (h3 : girls_sampled = 95)
  (h4 : sample_size ≤ total_students)
  (h5 : girls_sampled ≤ sample_size) : 
  ∃ (total_girls : ℕ), 
    total_girls * sample_size = girls_sampled * total_students ∧ 
    total_girls = 760 := by
sorry

end NUMINAMATH_CALUDE_girls_in_school_l3657_365700


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l3657_365791

theorem smallest_positive_solution (x : ℕ) : x = 21 ↔ 
  (x > 0 ∧ 
   (45 * x + 7) % 25 = 3 ∧ 
   ∀ y : ℕ, y > 0 → y < x → (45 * y + 7) % 25 ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l3657_365791


namespace NUMINAMATH_CALUDE_promotion_savings_l3657_365744

/-- Represents a promotion offered by the department store -/
structure Promotion where
  name : String
  first_pair_price : ℝ
  second_pair_price : ℝ
  additional_discount : ℝ

/-- Calculates the total cost for a given promotion -/
def total_cost (p : Promotion) (handbag_price : ℝ) : ℝ :=
  p.first_pair_price + p.second_pair_price + handbag_price - p.additional_discount

/-- The main theorem stating that Promotion A saves $19.5 more than Promotion B -/
theorem promotion_savings :
  let shoe_price : ℝ := 50
  let handbag_price : ℝ := 20
  let promotion_a : Promotion := {
    name := "A",
    first_pair_price := shoe_price,
    second_pair_price := shoe_price / 2,
    additional_discount := (shoe_price + shoe_price / 2 + handbag_price) * 0.1
  }
  let promotion_b : Promotion := {
    name := "B",
    first_pair_price := shoe_price,
    second_pair_price := shoe_price - 15,
    additional_discount := 0
  }
  total_cost promotion_b handbag_price - total_cost promotion_a handbag_price = 19.5 := by
  sorry


end NUMINAMATH_CALUDE_promotion_savings_l3657_365744


namespace NUMINAMATH_CALUDE_triangle_perimeter_is_19_l3657_365787

/-- Triangle PQR with given properties -/
structure Triangle where
  /-- Length of side PQ -/
  PQ : ℝ
  /-- Length of side QR -/
  QR : ℝ
  /-- Length of side PR -/
  PR : ℝ
  /-- Angle PQR equals angle PRQ -/
  angle_equality : PQ = PR

/-- The perimeter of a triangle is the sum of its side lengths -/
def perimeter (t : Triangle) : ℝ := t.PQ + t.QR + t.PR

/-- Theorem: The perimeter of the given triangle is 19 -/
theorem triangle_perimeter_is_19 (t : Triangle) 
  (h1 : t.QR = 5) 
  (h2 : t.PR = 7) : 
  perimeter t = 19 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_is_19_l3657_365787


namespace NUMINAMATH_CALUDE_second_number_proof_l3657_365763

theorem second_number_proof (a b c : ℝ) 
  (sum_eq : a + b + c = 98)
  (ratio_ab : a / b = 2 / 3)
  (ratio_bc : b / c = 5 / 8)
  (pos_a : 0 < a)
  (pos_b : 0 < b)
  (pos_c : 0 < c) :
  b = 30 := by
sorry

end NUMINAMATH_CALUDE_second_number_proof_l3657_365763


namespace NUMINAMATH_CALUDE_largest_three_digit_congruence_l3657_365773

theorem largest_three_digit_congruence :
  ∃ (n : ℕ), n = 993 ∧ 
  (∀ m : ℕ, m ≤ 999 → 45 * m ≡ 270 [MOD 315] → m ≤ n) ∧
  45 * n ≡ 270 [MOD 315] :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_congruence_l3657_365773


namespace NUMINAMATH_CALUDE_spade_evaluation_l3657_365704

def spade (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

theorem spade_evaluation : spade 2 (spade 3 4) = 384 := by
  sorry

end NUMINAMATH_CALUDE_spade_evaluation_l3657_365704


namespace NUMINAMATH_CALUDE_number_of_correct_statements_l3657_365766

-- Define the properties
def is_rational (m : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ m = a / b
def is_real (m : ℝ) : Prop := True

def tan_equal (A B : ℝ) : Prop := Real.tan A = Real.tan B
def angle_equal (A B : ℝ) : Prop := A = B

def quadratic_equation (x : ℝ) : Prop := x^2 - 2*x - 3 = 0
def x_equals_3 (x : ℝ) : Prop := x = 3

-- Define the statements
def statement1 : Prop := 
  (∀ m : ℝ, is_rational m → is_real m) ∧ 
  ¬(∀ m : ℝ, is_real m → is_rational m)

def statement2 : Prop := 
  (∀ A B : ℝ, tan_equal A B → angle_equal A B) ∧ 
  ¬(∀ A B : ℝ, angle_equal A B → tan_equal A B)

def statement3 : Prop := 
  (∀ x : ℝ, x_equals_3 x → quadratic_equation x) ∧ 
  ¬(∀ x : ℝ, quadratic_equation x → x_equals_3 x)

-- Theorem to prove
theorem number_of_correct_statements : 
  (statement1 ∧ ¬statement2 ∧ statement3) → 
  (Nat.card {s | s = statement1 ∨ s = statement2 ∨ s = statement3 ∧ s} = 2) :=
sorry

end NUMINAMATH_CALUDE_number_of_correct_statements_l3657_365766


namespace NUMINAMATH_CALUDE_special_functions_at_zero_l3657_365702

/-- Two non-constant functions satisfying specific addition formulas -/
class SpecialFunctions (f g : ℝ → ℝ) : Prop where
  add_f : ∀ x y, f (x + y) = f x * g y + g x * f y
  add_g : ∀ x y, g (x + y) = g x * g y - f x * f y
  non_constant_f : ∃ x y, f x ≠ f y
  non_constant_g : ∃ x y, g x ≠ g y

/-- The values of f(0) and g(0) for special functions f and g -/
theorem special_functions_at_zero {f g : ℝ → ℝ} [SpecialFunctions f g] :
  f 0 = 0 ∧ g 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_special_functions_at_zero_l3657_365702


namespace NUMINAMATH_CALUDE_equation_solution_l3657_365710

theorem equation_solution :
  let f : ℝ → ℝ := λ x => x + 36 / (x - 3)
  {x : ℝ | f x = -12} = {0, -9} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3657_365710


namespace NUMINAMATH_CALUDE_concert_revenue_l3657_365765

def adult_price : ℕ := 26
def child_price : ℕ := adult_price / 2
def adult_attendees : ℕ := 183
def child_attendees : ℕ := 28

theorem concert_revenue :
  adult_price * adult_attendees + child_price * child_attendees = 5122 :=
by sorry

end NUMINAMATH_CALUDE_concert_revenue_l3657_365765


namespace NUMINAMATH_CALUDE_number_machine_input_l3657_365760

/-- A number machine that adds 15 and then subtracts 6 -/
def number_machine (x : ℤ) : ℤ := x + 15 - 6

/-- Theorem stating that if the number machine outputs 77, the input must have been 68 -/
theorem number_machine_input (x : ℤ) : number_machine x = 77 → x = 68 := by
  sorry

end NUMINAMATH_CALUDE_number_machine_input_l3657_365760


namespace NUMINAMATH_CALUDE_cat_weight_l3657_365717

/-- Given a cat and a dog with specific weight relationships, prove the cat's weight -/
theorem cat_weight (cat_weight dog_weight : ℝ) : 
  dog_weight = cat_weight + 6 →
  cat_weight = dog_weight / 3 →
  cat_weight = 3 := by
sorry

end NUMINAMATH_CALUDE_cat_weight_l3657_365717


namespace NUMINAMATH_CALUDE_wilmas_garden_rows_l3657_365756

/-- The number of rows in Wilma's garden --/
def garden_rows : ℕ :=
  let yellow_flowers : ℕ := 12
  let green_flowers : ℕ := 2 * yellow_flowers
  let red_flowers : ℕ := 42
  let total_flowers : ℕ := yellow_flowers + green_flowers + red_flowers
  let flowers_per_row : ℕ := 13
  total_flowers / flowers_per_row

/-- Theorem stating that the number of rows in Wilma's garden is 6 --/
theorem wilmas_garden_rows :
  garden_rows = 6 := by
  sorry

end NUMINAMATH_CALUDE_wilmas_garden_rows_l3657_365756


namespace NUMINAMATH_CALUDE_ferry_tourists_sum_l3657_365767

/-- The number of trips made by the ferry -/
def num_trips : ℕ := 7

/-- The initial number of tourists -/
def initial_tourists : ℕ := 100

/-- The decrease in tourists per trip -/
def tourist_decrease : ℕ := 2

/-- The sum of tourists over all trips -/
def total_tourists : ℕ := 658

/-- Theorem stating that the sum of the arithmetic sequence
    representing the number of tourists per trip equals the total number of tourists -/
theorem ferry_tourists_sum :
  (num_trips / 2 : ℚ) * (2 * initial_tourists - (num_trips - 1) * tourist_decrease) = total_tourists := by
  sorry

end NUMINAMATH_CALUDE_ferry_tourists_sum_l3657_365767


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3657_365761

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x > 1 → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3657_365761


namespace NUMINAMATH_CALUDE_delta_equation_solution_l3657_365774

-- Define the Δ operation
def delta (a b : ℝ) : ℝ := a * b + a + b

-- Theorem statement
theorem delta_equation_solution :
  ∀ p : ℝ, delta p 3 = 39 → p = 9 := by
  sorry

end NUMINAMATH_CALUDE_delta_equation_solution_l3657_365774


namespace NUMINAMATH_CALUDE_symmetry_wrt_x_axis_l3657_365732

/-- Given a point A(-2, 3) in a Cartesian coordinate system, 
    its symmetrical point with respect to the x-axis has coordinates (-2, -3). -/
theorem symmetry_wrt_x_axis : 
  let A : ℝ × ℝ := (-2, 3)
  let symmetrical_point : ℝ × ℝ := (-2, -3)
  (∀ (x y : ℝ), (x, y) = A → (x, -y) = symmetrical_point) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_wrt_x_axis_l3657_365732


namespace NUMINAMATH_CALUDE_white_balls_count_l3657_365771

theorem white_balls_count (red : ℕ) (yellow : ℕ) (white : ℕ) 
  (h_red : red = 3)
  (h_yellow : yellow = 2)
  (h_prob : (yellow : ℚ) / (red + yellow + white) = 1/4) :
  white = 3 := by
sorry

end NUMINAMATH_CALUDE_white_balls_count_l3657_365771


namespace NUMINAMATH_CALUDE_min_value_x_plus_81_over_x_l3657_365743

theorem min_value_x_plus_81_over_x (x : ℝ) (h : x > 0) :
  x + 81 / x ≥ 18 ∧ (x + 81 / x = 18 ↔ x = 9) := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_81_over_x_l3657_365743


namespace NUMINAMATH_CALUDE_remainder_of_difference_l3657_365754

theorem remainder_of_difference (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (ha_mod : a % 6 = 2) (hb_mod : b % 6 = 3) (hab : a > b) : 
  (a - b) % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_difference_l3657_365754


namespace NUMINAMATH_CALUDE_power_boat_travel_time_l3657_365746

/-- Represents the scenario of a power boat and raft on a river --/
structure RiverScenario where
  r : ℝ  -- Speed of the river current (and raft)
  p : ℝ  -- Speed of the power boat relative to the river
  t : ℝ  -- Time taken by power boat from A to B

/-- The conditions of the problem --/
def scenario_conditions (s : RiverScenario) : Prop :=
  s.r > 0 ∧ s.p > 0 ∧ s.t > 0 ∧
  (s.p + s.r) * s.t + (s.p - s.r) * (9 - s.t) = 9 * s.r

/-- The theorem to be proved --/
theorem power_boat_travel_time (s : RiverScenario) :
  scenario_conditions s → s.t = 4.5 := by
  sorry


end NUMINAMATH_CALUDE_power_boat_travel_time_l3657_365746


namespace NUMINAMATH_CALUDE_wider_can_radius_l3657_365779

/-- Given two cylindrical cans with the same volume, where the height of one can is five times
    the height of the other, and the radius of the narrower can is 10 units,
    prove that the radius of the wider can is 10√5 units. -/
theorem wider_can_radius (h : ℝ) (volume : ℝ) : 
  volume = π * 10^2 * (5 * h) → 
  volume = π * ((10 * Real.sqrt 5)^2) * h := by
  sorry

end NUMINAMATH_CALUDE_wider_can_radius_l3657_365779


namespace NUMINAMATH_CALUDE_min_value_ab_l3657_365703

/-- Given that ab > 0 and points A(a, 0), B(0, b), and C(-2, -2) are collinear,
    the minimum value of ab is 16. -/
theorem min_value_ab (a b : ℝ) (hab : a * b > 0)
  (h_collinear : (a - 0) * (-2 - b) = (-2 - a) * (b - 0)) :
  ∀ x y : ℝ, x * y > 0 → (x - 0) * (-2 - y) = (-2 - x) * (y - 0) → a * b ≤ x * y → a * b = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_ab_l3657_365703


namespace NUMINAMATH_CALUDE_min_pieces_for_rearrangement_l3657_365708

/-- Represents a shape made of small squares -/
structure Shape :=
  (squares : Nat)

/-- Represents the goal configuration -/
structure GoalSquare :=
  (side : Nat)

/-- Represents a cutting of the shape into pieces -/
structure Cutting :=
  (num_pieces : Nat)

/-- Predicate to check if a cutting is valid for rearrangement -/
def is_valid_cutting (s : Shape) (g : GoalSquare) (c : Cutting) : Prop :=
  c.num_pieces ≥ 1 ∧ c.num_pieces ≤ s.squares

/-- Predicate to check if a cutting allows rearrangement into the goal square -/
def allows_rearrangement (s : Shape) (g : GoalSquare) (c : Cutting) : Prop :=
  is_valid_cutting s g c ∧ s.squares = g.side * g.side

/-- The main theorem stating the minimum number of pieces required -/
theorem min_pieces_for_rearrangement (s : Shape) (g : GoalSquare) :
  s.squares = 9 → g.side = 3 →
  ∃ (c : Cutting), 
    c.num_pieces = 3 ∧ 
    allows_rearrangement s g c ∧
    ∀ (c' : Cutting), allows_rearrangement s g c' → c'.num_pieces ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_pieces_for_rearrangement_l3657_365708
