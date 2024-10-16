import Mathlib

namespace NUMINAMATH_CALUDE_product_of_cosines_l3276_327632

theorem product_of_cosines : 
  (1 + Real.cos (π / 9)) * (1 + Real.cos (2 * π / 9)) * 
  (1 + Real.cos (8 * π / 9)) * (1 + Real.cos (7 * π / 9)) = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cosines_l3276_327632


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_t_value_l3276_327656

theorem sqrt_equality_implies_t_value :
  ∀ t : ℝ, (Real.sqrt (3 * Real.sqrt (t - 3)) = (10 - t) ^ (1/4)) → t = 37/10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_t_value_l3276_327656


namespace NUMINAMATH_CALUDE_solution_set_implies_ab_value_l3276_327663

theorem solution_set_implies_ab_value (a b : ℝ) : 
  (∀ x, x^2 + 2*a*x - 4*b ≤ 0 ↔ -2 ≤ x ∧ x ≤ 6) → 
  a^b = -8 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_ab_value_l3276_327663


namespace NUMINAMATH_CALUDE_walking_distance_time_relation_l3276_327679

/-- 
Given two people walking in opposite directions at different speeds, 
this theorem proves the relationship between the distance they are apart 
and the time it takes to reach that distance.
-/
theorem walking_distance_time_relation 
  (mary_speed sharon_speed : ℝ) 
  (initial_time initial_distance : ℝ) 
  (d t : ℝ) : 
  mary_speed = 4 →
  sharon_speed = 6 →
  initial_time = 0.3 →
  initial_distance = 3 →
  (mary_speed + sharon_speed) * initial_time = initial_distance →
  (mary_speed + sharon_speed) * t = d →
  t = d / 10 := by
  sorry

end NUMINAMATH_CALUDE_walking_distance_time_relation_l3276_327679


namespace NUMINAMATH_CALUDE_negation_cube_even_number_l3276_327670

theorem negation_cube_even_number (n : ℤ) :
  ¬(∀ n : ℤ, 2 ∣ n → 2 ∣ n^3) ↔ ∃ n : ℤ, 2 ∣ n ∧ ¬(2 ∣ n^3) :=
sorry

end NUMINAMATH_CALUDE_negation_cube_even_number_l3276_327670


namespace NUMINAMATH_CALUDE_basketball_team_selection_l3276_327675

theorem basketball_team_selection (n : ℕ) (k : ℕ) (twins : ℕ) : 
  n = 15 → k = 5 → twins = 2 →
  (Nat.choose n k) - (Nat.choose (n - twins) k) = 1716 := by
sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l3276_327675


namespace NUMINAMATH_CALUDE_map_scale_conversion_l3276_327612

/-- Given a map scale where 10 cm represents 50 km, prove that 15 cm represents 75 km -/
theorem map_scale_conversion (scale : ℝ) (h1 : scale = 50 / 10) : 15 * scale = 75 := by
  sorry

end NUMINAMATH_CALUDE_map_scale_conversion_l3276_327612


namespace NUMINAMATH_CALUDE_dice_probability_l3276_327687

/-- The number of sides on each die -/
def num_sides : ℕ := 12

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability of rolling a 'low' number (1-8) -/
def prob_low : ℚ := 2/3

/-- The probability of rolling a 'mid' or 'high' number (9-12) -/
def prob_mid_high : ℚ := 1/3

/-- The number of ways to choose 2 dice out of 5 -/
def choose_two_from_five : ℕ := 10

/-- The probability of the desired outcome -/
theorem dice_probability : 
  (choose_two_from_five : ℚ) * prob_low^2 * prob_mid_high^3 = 40/243 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l3276_327687


namespace NUMINAMATH_CALUDE_furniture_payment_l3276_327698

theorem furniture_payment (a b c d e : ℝ) : 
  a + b + c + d + e = 120 ∧
  a = (1/3) * (b + c + d + e) ∧
  b = (1/4) * (a + c + d + e) ∧
  c = (1/5) * (a + b + d + e) ∧
  d = (1/6) * (a + b + c + e) →
  e = 41.33 := by sorry

end NUMINAMATH_CALUDE_furniture_payment_l3276_327698


namespace NUMINAMATH_CALUDE_new_average_salary_l3276_327630

/-- Calculates the new average monthly salary after a change in supervisor --/
theorem new_average_salary
  (num_people : ℕ)
  (num_workers : ℕ)
  (old_average : ℚ)
  (old_supervisor_salary : ℚ)
  (new_supervisor_salary : ℚ)
  (h_num_people : num_people = 9)
  (h_num_workers : num_workers = 8)
  (h_old_average : old_average = 430)
  (h_old_supervisor : old_supervisor_salary = 870)
  (h_new_supervisor : new_supervisor_salary = 960) :
  (num_people * old_average - old_supervisor_salary + new_supervisor_salary) / num_people = 440 :=
sorry

end NUMINAMATH_CALUDE_new_average_salary_l3276_327630


namespace NUMINAMATH_CALUDE_ceiling_sum_of_square_roots_l3276_327609

theorem ceiling_sum_of_square_roots : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_of_square_roots_l3276_327609


namespace NUMINAMATH_CALUDE_hundredth_ring_squares_nth_ring_squares_l3276_327642

/-- The number of unit squares in the nth ring around a center square -/
def ring_squares (n : ℕ) : ℕ := 8 * n

/-- Theorem: The 100th ring contains 800 unit squares -/
theorem hundredth_ring_squares : ring_squares 100 = 800 := by
  sorry

/-- Theorem: For any positive integer n, the number of unit squares in the nth ring is 8n -/
theorem nth_ring_squares (n : ℕ) : ring_squares n = 8 * n := by
  sorry

end NUMINAMATH_CALUDE_hundredth_ring_squares_nth_ring_squares_l3276_327642


namespace NUMINAMATH_CALUDE_elaine_rent_percentage_l3276_327643

/-- Represents Elaine's earnings and rent expenses over two years -/
structure ElaineFinances where
  lastYearEarnings : ℝ
  lastYearRentPercentage : ℝ
  earningsIncrease : ℝ
  rentIncrease : ℝ
  thisYearRentPercentage : ℝ

/-- The conditions of Elaine's finances -/
def elaineFinancesConditions (e : ElaineFinances) : Prop :=
  e.lastYearRentPercentage = 20 ∧
  e.earningsIncrease = 35 ∧
  e.rentIncrease = 202.5

/-- Theorem stating that given the conditions, Elaine's rent percentage this year is 30% -/
theorem elaine_rent_percentage (e : ElaineFinances) 
  (h : elaineFinancesConditions e) : e.thisYearRentPercentage = 30 := by
  sorry


end NUMINAMATH_CALUDE_elaine_rent_percentage_l3276_327643


namespace NUMINAMATH_CALUDE_function_passes_through_point_l3276_327619

theorem function_passes_through_point (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 3) + 2
  f 3 = 3 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l3276_327619


namespace NUMINAMATH_CALUDE_nested_average_calculation_l3276_327668

-- Define the average of two numbers
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

-- Define the average of three numbers
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

-- Theorem statement
theorem nested_average_calculation : 
  avg3 (avg3 2 4 1) (avg2 3 2) 5 = 59 / 18 := by
  sorry

end NUMINAMATH_CALUDE_nested_average_calculation_l3276_327668


namespace NUMINAMATH_CALUDE_set_operations_l3276_327635

open Set

def U : Set ℝ := univ

def A : Set ℝ := {x | 1 ≤ x - 1 ∧ x - 1 < 3}

def B : Set ℝ := {x | 2*x - 9 ≥ 6 - 3*x}

theorem set_operations :
  (A ∪ B = {x | x ≥ 2}) ∧
  (Aᶜ ∩ Bᶜ = {x | x < 3 ∨ x ≥ 4}) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l3276_327635


namespace NUMINAMATH_CALUDE_PL_length_l3276_327608

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a rectangle -/
structure Rectangle :=
  (topLeft : Point) (bottomRight : Point)

/-- The square WXYZ -/
def square : Rectangle :=
  { topLeft := { x := 0, y := 2 },
    bottomRight := { x := 2, y := 0 } }

/-- The length of PL -/
def PL : ℝ := 1

/-- States that two rectangles are congruent -/
def congruentRectangles (r1 r2 : Rectangle) : Prop :=
  (r1.bottomRight.x - r1.topLeft.x) * (r1.topLeft.y - r1.bottomRight.y) =
  (r2.bottomRight.x - r2.topLeft.x) * (r2.topLeft.y - r2.bottomRight.y)

/-- The theorem to be proved -/
theorem PL_length :
  ∀ (LMNO PQRS : Rectangle),
    congruentRectangles LMNO PQRS →
    PL = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_PL_length_l3276_327608


namespace NUMINAMATH_CALUDE_symmetric_point_correct_l3276_327623

/-- Given a point A and a line l, find the point B symmetric to A about l -/
def symmetricPoint (A : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ × ℝ := sorry

/-- The line x - y - 1 = 0 -/
def line (p : ℝ × ℝ) : Prop :=
  p.1 - p.2 - 1 = 0

theorem symmetric_point_correct :
  let A : ℝ × ℝ := (-1, 1)
  let B : ℝ × ℝ := (2, -2)
  symmetricPoint A line = B := by sorry

end NUMINAMATH_CALUDE_symmetric_point_correct_l3276_327623


namespace NUMINAMATH_CALUDE_percentage_problem_l3276_327641

theorem percentage_problem (x y P : ℝ) 
  (h1 : 0.3 * (x - y) = (P / 100) * (x + y))
  (h2 : y = 0.2 * x) : 
  P = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l3276_327641


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l3276_327621

theorem gcd_lcm_product (a b : ℤ) : Nat.gcd a.natAbs b.natAbs * Nat.lcm a.natAbs b.natAbs = a.natAbs * b.natAbs := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l3276_327621


namespace NUMINAMATH_CALUDE_school_pupils_count_l3276_327606

theorem school_pupils_count (girls boys : ℕ) (h1 : girls = 542) (h2 : boys = 387) :
  girls + boys = 929 := by
  sorry

end NUMINAMATH_CALUDE_school_pupils_count_l3276_327606


namespace NUMINAMATH_CALUDE_gcd_105_88_l3276_327688

theorem gcd_105_88 : Nat.gcd 105 88 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_105_88_l3276_327688


namespace NUMINAMATH_CALUDE_line_point_k_value_l3276_327669

/-- Given a line containing the points (0, 7), (15, k), and (20, 3), prove that k = 4 -/
theorem line_point_k_value (k : ℝ) : 
  (∀ (x y : ℝ), (x = 0 ∧ y = 7) ∨ (x = 15 ∧ y = k) ∨ (x = 20 ∧ y = 3) → 
    ∃ (m b : ℝ), y = m * x + b) → 
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_line_point_k_value_l3276_327669


namespace NUMINAMATH_CALUDE_log_cutting_ratio_l3276_327650

/-- Given a log of length 20 feet where each linear foot weighs 150 pounds,
    if the log is cut into two equal pieces each weighing 1500 pounds,
    then the ratio of the length of each cut piece to the length of the original log is 1/2. -/
theorem log_cutting_ratio :
  ∀ (original_length cut_length : ℝ) (weight_per_foot cut_weight : ℝ),
    original_length = 20 →
    weight_per_foot = 150 →
    cut_weight = 1500 →
    cut_length * weight_per_foot = cut_weight →
    cut_length / original_length = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_cutting_ratio_l3276_327650


namespace NUMINAMATH_CALUDE_bear_age_addition_l3276_327614

theorem bear_age_addition : 24 + 36 = 60 := by
  sorry

end NUMINAMATH_CALUDE_bear_age_addition_l3276_327614


namespace NUMINAMATH_CALUDE_angle_ABC_is_30_l3276_327689

-- Define the angles
def angle_CBD : ℝ := 90
def angle_ABD : ℝ := 60

-- Theorem statement
theorem angle_ABC_is_30 :
  ∀ (angle_ABC : ℝ),
  angle_ABD + angle_ABC + angle_CBD = 180 →
  angle_ABC = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_ABC_is_30_l3276_327689


namespace NUMINAMATH_CALUDE_line_passes_through_point_min_triangle_area_min_area_line_equation_l3276_327691

/-- Definition of the line l with parameter k -/
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 4 * k + 2 = 0

/-- Theorem stating that the line l always passes through the point (-4, 2) -/
theorem line_passes_through_point (k : ℝ) : line_l k (-4) 2 := by sorry

/-- Definition of the area of the triangle formed by the line and coordinate axes -/
noncomputable def triangle_area (k : ℝ) : ℝ := sorry

/-- Theorem stating the minimum area of the triangle -/
theorem min_triangle_area : 
  ∃ (k : ℝ), triangle_area k = 16 ∧ ∀ (k' : ℝ), triangle_area k' ≥ 16 := by sorry

/-- Theorem stating the equation of the line when the area is minimum -/
theorem min_area_line_equation (k : ℝ) : 
  triangle_area k = 16 → line_l k x y ↔ x - 2 * y + 8 = 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_min_triangle_area_min_area_line_equation_l3276_327691


namespace NUMINAMATH_CALUDE_difference_of_squares_division_l3276_327633

theorem difference_of_squares_division : (245^2 - 225^2) / 20 = 470 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_division_l3276_327633


namespace NUMINAMATH_CALUDE_largest_integral_x_l3276_327651

theorem largest_integral_x : ∃ (x : ℤ), 
  (∀ (y : ℤ), (1 : ℚ) / 3 < (y : ℚ) / 5 ∧ (y : ℚ) / 5 < 5 / 8 → y ≤ x) ∧
  (1 : ℚ) / 3 < (x : ℚ) / 5 ∧ (x : ℚ) / 5 < 5 / 8 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integral_x_l3276_327651


namespace NUMINAMATH_CALUDE_sequence_non_positive_l3276_327644

theorem sequence_non_positive (N : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 0) 
  (hN : a N = 0) 
  (h_rec : ∀ i ∈ Finset.range (N - 1), 
    a (i + 2) - 2 * a (i + 1) + a i = (a (i + 1))^2) :
  ∀ i ∈ Finset.range (N - 1), a (i + 1) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sequence_non_positive_l3276_327644


namespace NUMINAMATH_CALUDE_largest_common_term_l3276_327657

def is_in_sequence (start : ℤ) (diff : ℤ) (n : ℤ) : Prop :=
  ∃ k : ℤ, n = start + k * diff

theorem largest_common_term : ∃ n : ℤ,
  (1 ≤ n ∧ n ≤ 100) ∧
  (is_in_sequence 2 5 n) ∧
  (is_in_sequence 3 8 n) ∧
  (∀ m : ℤ, (1 ≤ m ∧ m ≤ 100) → 
    (is_in_sequence 2 5 m) → 
    (is_in_sequence 3 8 m) → 
    m ≤ n) ∧
  n = 67 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_term_l3276_327657


namespace NUMINAMATH_CALUDE_work_completion_time_l3276_327677

/-- Proves that A can complete the work in 15 days given the conditions -/
theorem work_completion_time (x : ℝ) : 
  (x > 0) →  -- A's completion time is positive
  (4 * (1 / x + 1 / 20) = 1 - 0.5333333333333333) →  -- Condition after 4 days of joint work
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3276_327677


namespace NUMINAMATH_CALUDE_jasons_cousins_l3276_327666

theorem jasons_cousins (cupcakes_bought : ℕ) (cupcakes_per_cousin : ℕ) : 
  cupcakes_bought = 4 * 12 → cupcakes_per_cousin = 3 → 
  cupcakes_bought / cupcakes_per_cousin = 16 := by
  sorry

end NUMINAMATH_CALUDE_jasons_cousins_l3276_327666


namespace NUMINAMATH_CALUDE_coronavirus_size_scientific_notation_l3276_327645

/-- The size of a novel coronavirus in meters -/
def coronavirus_size : ℝ := 0.000000125

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Convert a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation := sorry

theorem coronavirus_size_scientific_notation :
  to_scientific_notation coronavirus_size = ScientificNotation.mk 1.25 (-7) := by sorry

end NUMINAMATH_CALUDE_coronavirus_size_scientific_notation_l3276_327645


namespace NUMINAMATH_CALUDE_gcd_2814_1806_l3276_327692

theorem gcd_2814_1806 : Nat.gcd 2814 1806 = 42 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2814_1806_l3276_327692


namespace NUMINAMATH_CALUDE_donation_ratio_l3276_327683

theorem donation_ratio : 
  ∀ (total parents teachers students : ℝ),
  parents = 0.25 * total →
  teachers + students = 0.75 * total →
  teachers = (2/5) * (teachers + students) →
  students = (3/5) * (teachers + students) →
  parents / students = 5 / 9 := by
sorry

end NUMINAMATH_CALUDE_donation_ratio_l3276_327683


namespace NUMINAMATH_CALUDE_pool_filling_time_l3276_327676

theorem pool_filling_time (pipe1 pipe2 pipe3 pipe4 : ℚ) 
  (h1 : pipe1 = 1)
  (h2 : pipe2 = 1/2)
  (h3 : pipe3 = 1/3)
  (h4 : pipe4 = 1/4) :
  1 / (pipe1 + pipe2 + pipe3 + pipe4) = 12/25 := by sorry

end NUMINAMATH_CALUDE_pool_filling_time_l3276_327676


namespace NUMINAMATH_CALUDE_smallest_value_for_x_between_1_and_2_l3276_327655

theorem smallest_value_for_x_between_1_and_2 (x : ℝ) (h : 1 < x ∧ x < 2) :
  (1 / x < x) ∧ (1 / x < x^2) ∧ (1 / x < 2*x) ∧ (1 / x < Real.sqrt x) := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_for_x_between_1_and_2_l3276_327655


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3276_327617

theorem sum_of_coefficients (g h i j k : ℤ) : 
  (∀ y : ℝ, 1000 * y^3 + 27 = (g * y + h) * (i * y^2 + j * y + k)) →
  g + h + i + j + k = 92 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3276_327617


namespace NUMINAMATH_CALUDE_max_value_constraint_l3276_327664

theorem max_value_constraint (x y z : ℝ) (h : x^2 + y^2 + z^2 = 25) :
  x + 2*y + 2*z ≤ 15 := by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l3276_327664


namespace NUMINAMATH_CALUDE_table_relationship_l3276_327673

def f (x : ℝ) : ℝ := -5 * x^2 - 10 * x

theorem table_relationship : 
  (f 0 = 0) ∧ 
  (f 1 = -15) ∧ 
  (f 2 = -40) ∧ 
  (f 3 = -75) ∧ 
  (f 4 = -120) := by
  sorry

end NUMINAMATH_CALUDE_table_relationship_l3276_327673


namespace NUMINAMATH_CALUDE_balls_distribution_l3276_327615

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem balls_distribution : distribute_balls 7 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_balls_distribution_l3276_327615


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l3276_327653

theorem rectangle_area_increase (l w : ℝ) (h1 : l > 0) (h2 : w > 0) : 
  let original_area := l * w
  let new_area := (2 * l) * (2 * w)
  (new_area - original_area) / original_area = 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l3276_327653


namespace NUMINAMATH_CALUDE_max_value_of_f_l3276_327629

noncomputable def f (x : ℝ) : ℝ := (x + 3) / (x^2 + 6*x + 13)

theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc (-2 : ℝ) 2 ∧ 
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ≤ f c) ∧
  f c = 1/4 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3276_327629


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l3276_327680

/-- The cost of fruits at Zoe's Zesty Market -/
structure FruitCost where
  banana : ℕ
  apple : ℕ
  orange : ℕ

/-- The cost relationship between fruits -/
def cost_relationship (fc : FruitCost) : Prop :=
  5 * fc.banana = 4 * fc.apple ∧ 8 * fc.apple = 6 * fc.orange

/-- The theorem stating the equivalence of 40 bananas and 24 oranges in cost -/
theorem banana_orange_equivalence (fc : FruitCost) 
  (h : cost_relationship fc) : 40 * fc.banana = 24 * fc.orange := by
  sorry

#check banana_orange_equivalence

end NUMINAMATH_CALUDE_banana_orange_equivalence_l3276_327680


namespace NUMINAMATH_CALUDE_tower_configurations_count_l3276_327660

/-- The number of ways to build a tower of 10 cubes high using 3 red cubes, 4 blue cubes, and 5 yellow cubes, where two cubes are not used. -/
def towerConfigurations (red : Nat) (blue : Nat) (yellow : Nat) (towerHeight : Nat) : Nat :=
  sorry

/-- Theorem stating that the number of different tower configurations is 277,200 -/
theorem tower_configurations_count :
  towerConfigurations 3 4 5 10 = 277200 := by
  sorry

end NUMINAMATH_CALUDE_tower_configurations_count_l3276_327660


namespace NUMINAMATH_CALUDE_difference_set_not_always_equal_l3276_327628

theorem difference_set_not_always_equal :
  ∃ (A B : Set α) (hA : A.Nonempty) (hB : B.Nonempty),
    (A \ B) ≠ (B \ A) :=
by sorry

end NUMINAMATH_CALUDE_difference_set_not_always_equal_l3276_327628


namespace NUMINAMATH_CALUDE_arithmetic_operations_l3276_327652

theorem arithmetic_operations :
  ((-20) - (-14) + (-18) - 13 = -37) ∧
  (((-3/4) + (1/6) - (5/8)) / (-1/24) = 29) ∧
  ((-3^2) + (-3)^2 + 3*2 + |(-4)| = 10) ∧
  (16 / (-2)^3 - (-1/6) * (-4) + (-1)^2024 = -5/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l3276_327652


namespace NUMINAMATH_CALUDE_inequality_proof_l3276_327694

theorem inequality_proof (x : ℝ) (n : ℕ) (h : x > 0) :
  1 + x^(n+1) ≥ (2*x)^n / (1+x)^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3276_327694


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l3276_327601

theorem unique_solution_for_equation (N : ℕ+) :
  ∃! (m n : ℕ+), m + (1/2 : ℚ) * (m + n - 1) * (m + n - 2) = N := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l3276_327601


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l3276_327605

theorem opposite_of_negative_2023 :
  ∃ x : ℤ, x + (-2023) = 0 ∧ x = 2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l3276_327605


namespace NUMINAMATH_CALUDE_object3_length_is_15_l3276_327659

def longest_tape : ℕ := 5

def object1_length : ℕ := 225
def object2_length : ℕ := 780

def object3_length : ℕ := Nat.gcd object1_length object2_length

theorem object3_length_is_15 :
  longest_tape = 5 ∧
  object1_length = 225 ∧
  object2_length = 780 ∧
  object3_length = Nat.gcd object1_length object2_length →
  object3_length = 15 :=
by sorry

end NUMINAMATH_CALUDE_object3_length_is_15_l3276_327659


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3276_327662

theorem min_value_sum_reciprocals (n : ℕ+) (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  (1 / (1 + a ^ n.val)) + (1 / (1 + b ^ n.val)) ≥ 1 ∧
  ((1 / (1 + a ^ n.val)) + (1 / (1 + b ^ n.val)) = 1 ↔ a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3276_327662


namespace NUMINAMATH_CALUDE_principal_arg_range_l3276_327646

open Complex

theorem principal_arg_range (z ω : ℂ) 
  (h1 : abs (z - I) = 1)
  (h2 : z ≠ 0)
  (h3 : z ≠ 2 * I)
  (h4 : ∃ (r : ℝ), (ω - 2 * I) / ω * z / (z - 2 * I) = r) :
  ∃ (θ : ℝ), θ ∈ (Set.Ioo 0 π ∪ Set.Ioo π (2 * π)) ∧ arg (ω - 2) = θ :=
sorry

end NUMINAMATH_CALUDE_principal_arg_range_l3276_327646


namespace NUMINAMATH_CALUDE_parabola_equation_for_given_focus_and_directrix_l3276_327622

/-- A parabola is defined by a focus point and a directrix line parallel to the x-axis. -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ

/-- The equation of a parabola given its focus and directrix. -/
def parabola_equation (p : Parabola) : ℝ → ℝ → Prop :=
  λ x y => x^2 = 4 * (p.focus.2 - p.directrix) * (y - (p.focus.2 + p.directrix) / 2)

theorem parabola_equation_for_given_focus_and_directrix :
  let p : Parabola := { focus := (0, 4), directrix := -4 }
  ∀ x y : ℝ, parabola_equation p x y ↔ x^2 = 16 * y := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_for_given_focus_and_directrix_l3276_327622


namespace NUMINAMATH_CALUDE_QY_eq_10_l3276_327626

/-- Circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ

/-- Point outside the circle -/
def Q : ℝ × ℝ := sorry

/-- Circle C -/
def C : Circle := sorry

/-- Points on the circle -/
def X : ℝ × ℝ := sorry
def Y : ℝ × ℝ := sorry
def Z : ℝ × ℝ := sorry

/-- Distances -/
def QX : ℝ := sorry
def QY : ℝ := sorry
def QZ : ℝ := sorry

/-- Q is outside C -/
axiom h_Q_outside : Q ∉ {p | (p.1 - C.O.1)^2 + (p.2 - C.O.2)^2 ≤ C.r^2}

/-- QZ is tangent to C at Z -/
axiom h_QZ_tangent : (Z.1 - C.O.1)^2 + (Z.2 - C.O.2)^2 = C.r^2 ∧
  ((Z.1 - Q.1) * (Z.1 - C.O.1) + (Z.2 - Q.2) * (Z.2 - C.O.2) = 0)

/-- X and Y are on C -/
axiom h_X_on_C : (X.1 - C.O.1)^2 + (X.2 - C.O.2)^2 = C.r^2
axiom h_Y_on_C : (Y.1 - C.O.1)^2 + (Y.2 - C.O.2)^2 = C.r^2

/-- QX < QY -/
axiom h_QX_lt_QY : QX < QY

/-- QX = 5 -/
axiom h_QX_eq_5 : QX = 5

/-- QZ = 2(QY - QX) -/
axiom h_QZ_eq : QZ = 2 * (QY - QX)

/-- Power of a Point theorem -/
axiom power_of_point : QX * QY = QZ^2

theorem QY_eq_10 : QY = 10 := by sorry

end NUMINAMATH_CALUDE_QY_eq_10_l3276_327626


namespace NUMINAMATH_CALUDE_overall_average_marks_l3276_327638

/-- Given three batches of students with their respective sizes and average marks,
    calculate the overall average marks for all students combined. -/
theorem overall_average_marks
  (batch1_size batch2_size batch3_size : ℕ)
  (batch1_avg batch2_avg batch3_avg : ℚ)
  (h1 : batch1_size = 40)
  (h2 : batch2_size = 50)
  (h3 : batch3_size = 60)
  (h4 : batch1_avg = 45)
  (h5 : batch2_avg = 55)
  (h6 : batch3_avg = 65) :
  (batch1_size * batch1_avg + batch2_size * batch2_avg + batch3_size * batch3_avg) /
  (batch1_size + batch2_size + batch3_size) = 8450 / 150 := by
  sorry

#eval (8450 : ℚ) / 150  -- To verify the result

end NUMINAMATH_CALUDE_overall_average_marks_l3276_327638


namespace NUMINAMATH_CALUDE_f_has_three_distinct_roots_l3276_327667

/-- The polynomial function whose roots we want to count -/
def f (x : ℝ) : ℝ := (x + 5) * (x^2 + 5*x - 6)

/-- The statement that f has exactly 3 distinct real roots -/
theorem f_has_three_distinct_roots : ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c) :=
sorry

end NUMINAMATH_CALUDE_f_has_three_distinct_roots_l3276_327667


namespace NUMINAMATH_CALUDE_two_digit_decimal_bounds_l3276_327602

-- Define a two-digit decimal number accurate to the tenth place
def TwoDigitDecimal (x : ℝ) : Prop :=
  10 ≤ x ∧ x < 100 ∧ ∃ (n : ℤ), x = n / 10

-- Define the approximation to the tenth place
def ApproximateToTenth (x y : ℝ) : Prop :=
  ∃ (n : ℤ), y = n / 10 ∧ |x - y| < 0.05

-- Theorem statement
theorem two_digit_decimal_bounds :
  ∀ x : ℝ,
  TwoDigitDecimal x →
  ApproximateToTenth x 15.6 →
  x ≤ 15.64 ∧ x ≥ 15.55 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_decimal_bounds_l3276_327602


namespace NUMINAMATH_CALUDE_maria_earnings_l3276_327648

/-- The cost of brushes in dollars -/
def brush_cost : ℕ := 20

/-- The cost of canvas in dollars -/
def canvas_cost : ℕ := 3 * brush_cost

/-- The cost of paint per liter in dollars -/
def paint_cost_per_liter : ℕ := 8

/-- The minimum number of liters of paint needed -/
def paint_liters : ℕ := 5

/-- The selling price of the painting in dollars -/
def selling_price : ℕ := 200

/-- Maria's earnings from selling the painting -/
def earnings : ℕ := selling_price - (brush_cost + canvas_cost + paint_cost_per_liter * paint_liters)

theorem maria_earnings : earnings = 80 := by
  sorry

end NUMINAMATH_CALUDE_maria_earnings_l3276_327648


namespace NUMINAMATH_CALUDE_janet_sculpture_weight_l3276_327613

/-- Given Janet's work details, prove the weight of the first sculpture -/
theorem janet_sculpture_weight
  (exterminator_rate : ℝ)
  (sculpture_rate : ℝ)
  (exterminator_hours : ℝ)
  (second_sculpture_weight : ℝ)
  (total_income : ℝ)
  (h1 : exterminator_rate = 70)
  (h2 : sculpture_rate = 20)
  (h3 : exterminator_hours = 20)
  (h4 : second_sculpture_weight = 7)
  (h5 : total_income = 1640)
  : ∃ (first_sculpture_weight : ℝ),
    first_sculpture_weight = 5 ∧
    total_income = exterminator_rate * exterminator_hours +
                   sculpture_rate * (first_sculpture_weight + second_sculpture_weight) :=
by sorry

end NUMINAMATH_CALUDE_janet_sculpture_weight_l3276_327613


namespace NUMINAMATH_CALUDE_max_market_women_eight_market_women_l3276_327665

def farthings_in_2s_2_1_4d : ℕ := 105

theorem max_market_women (n : ℕ) : n ∣ farthings_in_2s_2_1_4d → n ≤ 8 :=
sorry

theorem eight_market_women : ∃ (s : Finset ℕ), s.card = 8 ∧ ∀ n ∈ s, n ∣ farthings_in_2s_2_1_4d :=
sorry

end NUMINAMATH_CALUDE_max_market_women_eight_market_women_l3276_327665


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l3276_327624

def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x - a)

theorem even_function_implies_a_equals_one (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l3276_327624


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l3276_327611

/-- Calculates the total shaded area of a right triangle and half of an adjacent rectangle -/
theorem shaded_area_calculation (triangle_base : ℝ) (triangle_height : ℝ) (rectangle_width : ℝ) :
  triangle_base = 6 →
  triangle_height = 8 →
  rectangle_width = 5 →
  (1 / 2 * triangle_base * triangle_height) + (1 / 2 * rectangle_width * triangle_height) = 44 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l3276_327611


namespace NUMINAMATH_CALUDE_largest_digit_sum_for_special_fraction_l3276_327684

/-- A digit is a natural number between 0 and 9 inclusive -/
def Digit := {n : ℕ // n ≤ 9}

/-- abc represents a three-digit number -/
def ThreeDigitNumber (a b c : Digit) : ℕ := 100 * a.val + 10 * b.val + c.val

theorem largest_digit_sum_for_special_fraction :
  ∃ (a b c : Digit) (y : ℕ),
    (10 ≤ y ∧ y ≤ 99) ∧
    (ThreeDigitNumber a b c : ℚ) / 1000 = 1 / y ∧
    ∀ (a' b' c' : Digit) (y' : ℕ),
      (10 ≤ y' ∧ y' ≤ 99) →
      (ThreeDigitNumber a' b' c' : ℚ) / 1000 = 1 / y' →
      a.val + b.val + c.val ≥ a'.val + b'.val + c'.val ∧
      a.val + b.val + c.val = 7 :=
sorry

end NUMINAMATH_CALUDE_largest_digit_sum_for_special_fraction_l3276_327684


namespace NUMINAMATH_CALUDE_initial_distance_is_40_l3276_327671

/-- The initial distance between two people walking towards each other -/
def initial_distance (speed : ℝ) (distance_walked : ℝ) : ℝ :=
  2 * distance_walked

/-- Theorem: The initial distance between Fred and Sam is 40 miles -/
theorem initial_distance_is_40 :
  let fred_speed : ℝ := 4
  let sam_speed : ℝ := 4
  let sam_distance : ℝ := 20
  initial_distance fred_speed sam_distance = 40 := by
  sorry


end NUMINAMATH_CALUDE_initial_distance_is_40_l3276_327671


namespace NUMINAMATH_CALUDE_system_solution_l3276_327678

theorem system_solution (x y a : ℝ) : 
  3 * x + y = 1 + 3 * a →
  x + 3 * y = 1 - a →
  x + y = 0 →
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3276_327678


namespace NUMINAMATH_CALUDE_not_both_perfect_cubes_l3276_327686

theorem not_both_perfect_cubes (n : ℕ) : 
  ¬(∃ a b : ℕ, (n + 2 = a^3) ∧ (n^2 + n + 1 = b^3)) := by
  sorry

end NUMINAMATH_CALUDE_not_both_perfect_cubes_l3276_327686


namespace NUMINAMATH_CALUDE_max_perfect_squares_pairwise_products_l3276_327616

theorem max_perfect_squares_pairwise_products (a b : ℕ) (h : a ≠ b) :
  let products := {a * (a + 2), b * (b + 2), a * b, a * (b + 2), (a + 2) * b, (a + 2) * (b + 2)}
  (∃ (s : Finset ℕ), s ⊆ products ∧ (∀ x ∈ s, ∃ y : ℕ, x = y ^ 2) ∧ s.card = 2) ∧
  (∀ (s : Finset ℕ), s ⊆ products → (∀ x ∈ s, ∃ y : ℕ, x = y ^ 2) → s.card ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_max_perfect_squares_pairwise_products_l3276_327616


namespace NUMINAMATH_CALUDE_area_between_circles_l3276_327620

/-- The area between two concentric circles -/
theorem area_between_circles (R r : ℝ) (h1 : R = 10) (h2 : r = 4) :
  (π * R^2) - (π * r^2) = 84 * π := by
  sorry

end NUMINAMATH_CALUDE_area_between_circles_l3276_327620


namespace NUMINAMATH_CALUDE_distribution_schemes_count_l3276_327647

/-- The number of ways to distribute students from classes to districts -/
def distribute_students (num_classes : ℕ) (students_per_class : ℕ) (num_districts : ℕ) (students_per_district : ℕ) : ℕ :=
  -- Number of ways to choose 2 classes out of 4
  (num_classes.choose 2) *
  -- Number of ways to choose 2 districts out of 4
  (num_districts.choose 2) *
  -- Number of ways to choose 1 student from each of the remaining 2 classes
  (students_per_class.choose 1) * (students_per_class.choose 1) *
  -- Number of ways to assign these 2 students to the remaining 2 districts
  2

/-- Theorem stating that the number of distribution schemes is 288 -/
theorem distribution_schemes_count :
  distribute_students 4 2 4 2 = 288 := by
  sorry

end NUMINAMATH_CALUDE_distribution_schemes_count_l3276_327647


namespace NUMINAMATH_CALUDE_solve_equation_and_evaluate_l3276_327682

theorem solve_equation_and_evaluate (x : ℚ) : 
  (4 * x - 8 = 12 * x + 4) → (5 * (x - 3) = -45 / 2) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_and_evaluate_l3276_327682


namespace NUMINAMATH_CALUDE_orange_sale_savings_l3276_327685

/-- Calculates the total savings for a mother's birthday gift based on orange sales. -/
theorem orange_sale_savings 
  (liam_oranges : ℕ) 
  (liam_price : ℚ) 
  (claire_oranges : ℕ) 
  (claire_price : ℚ) 
  (h1 : liam_oranges = 40)
  (h2 : liam_price = 5/2)
  (h3 : claire_oranges = 30)
  (h4 : claire_price = 6/5)
  : ℚ :=
by
  sorry

#check orange_sale_savings

end NUMINAMATH_CALUDE_orange_sale_savings_l3276_327685


namespace NUMINAMATH_CALUDE_triangle_arithmetic_sequence_angles_l3276_327693

theorem triangle_arithmetic_sequence_angles (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C →  -- angles are positive
  A + B + C = 180 →  -- sum of angles in a triangle
  ∃ (d : ℝ), C - B = B - A →  -- arithmetic sequence condition
  B = 60 := by sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_sequence_angles_l3276_327693


namespace NUMINAMATH_CALUDE_abs_ab_value_l3276_327634

/-- Given an ellipse and a hyperbola with specific foci, prove that |ab| = 2√65 -/
theorem abs_ab_value (a b : ℝ) : 
  (∀ x y, x^2/a^2 + y^2/b^2 = 1 → (x = 0 ∧ y = 4) ∨ (x = 0 ∧ y = -4)) →
  (∀ x y, x^2/a^2 - y^2/b^2 = 1 → (x = 6 ∧ y = 0) ∨ (x = -6 ∧ y = 0)) →
  |a * b| = 2 * Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_abs_ab_value_l3276_327634


namespace NUMINAMATH_CALUDE_solve_for_q_l3276_327658

theorem solve_for_q (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 10) 
  (eq2 : 6 * p + 5 * q = 17) : 
  q = -25 / 11 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l3276_327658


namespace NUMINAMATH_CALUDE_smallest_a_for_two_roots_less_than_one_l3276_327637

theorem smallest_a_for_two_roots_less_than_one : 
  ∃ (a b c : ℤ), 
    (a > 0) ∧ 
    (∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ 0 < r₁ ∧ r₁ < 1 ∧ 0 < r₂ ∧ r₂ < 1 ∧ 
      (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = r₁ ∨ x = r₂)) ∧
    (∀ a' : ℤ, 0 < a' ∧ a' < a → 
      ¬∃ (b' c' : ℤ), ∃ (s₁ s₂ : ℝ), s₁ ≠ s₂ ∧ 0 < s₁ ∧ s₁ < 1 ∧ 0 < s₂ ∧ s₂ < 1 ∧ 
        (∀ x : ℝ, a' * x^2 + b' * x + c' = 0 ↔ x = s₁ ∨ x = s₂)) ∧
    a = 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_for_two_roots_less_than_one_l3276_327637


namespace NUMINAMATH_CALUDE_total_books_on_shelves_l3276_327607

theorem total_books_on_shelves (num_shelves : ℕ) (books_per_shelf : ℚ) : 
  num_shelves = 520 → books_per_shelf = 37.5 → num_shelves * books_per_shelf = 19500 := by
  sorry

end NUMINAMATH_CALUDE_total_books_on_shelves_l3276_327607


namespace NUMINAMATH_CALUDE_segment_length_from_perpendicular_lines_and_midpoint_l3276_327654

/-- Given two perpendicular lines and a midpoint, prove the length of the segment. -/
theorem segment_length_from_perpendicular_lines_and_midpoint
  (A B : ℝ × ℝ) -- Points A and B
  (a : ℝ) -- Parameter in the equation of the second line
  (h1 : (2 * A.1 - A.2 = 0)) -- A is on the line 2x - y = 0
  (h2 : (B.1 + a * B.2 = 0)) -- B is on the line x + ay = 0
  (h3 : (2 : ℝ) * A.1 + (-1 : ℝ) * a = 0) -- Perpendicularity condition
  (h4 : (A.1 + B.1) / 2 = 0 ∧ (A.2 + B.2) / 2 = 10 / a) -- Midpoint condition
  : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_segment_length_from_perpendicular_lines_and_midpoint_l3276_327654


namespace NUMINAMATH_CALUDE_constant_distance_l3276_327672

/-- Represents an ellipse centered at the origin with foci on the x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h : 0 < b ∧ b < a
  h_e : e = Real.sqrt 2 / 2
  h_eq : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1}

/-- A line intersecting the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  m : ℝ
  h_intersect : ∃ (x y : ℝ), x^2 / E.a^2 + y^2 / E.b^2 = 1 ∧ y = k * x + m

/-- The theorem to be proved -/
theorem constant_distance (E : Ellipse) (l : IntersectingLine E) :
  ∃ (P Q : ℝ × ℝ) (d : ℝ),
    P ∈ {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1} ∧
    Q ∈ {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1} ∧
    P.1 * Q.1 + P.2 * Q.2 = 0 ∧
    d = Real.sqrt 6 / 3 ∧
    d = abs m / Real.sqrt (l.k^2 + 1) :=
  sorry

end NUMINAMATH_CALUDE_constant_distance_l3276_327672


namespace NUMINAMATH_CALUDE_expression_value_l3276_327610

theorem expression_value : 
  2 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 4000 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3276_327610


namespace NUMINAMATH_CALUDE_fraction_of_boys_reading_l3276_327618

theorem fraction_of_boys_reading (total_girls : ℕ) (total_boys : ℕ) 
  (fraction_girls_reading : ℚ) (not_reading : ℕ) :
  total_girls = 12 →
  total_boys = 10 →
  fraction_girls_reading = 5/6 →
  not_reading = 4 →
  (total_boys - (not_reading - (total_girls - (fraction_girls_reading * total_girls).num))) / total_boys = 4/5 := by
  sorry


end NUMINAMATH_CALUDE_fraction_of_boys_reading_l3276_327618


namespace NUMINAMATH_CALUDE_point_equidistant_from_origin_and_A_l3276_327639

/-- Given a point P(x, y) that is 17 units away from both the origin O(0,0) and point A(16,0),
    prove that the coordinates of P must be either (8, 15) or (8, -15). -/
theorem point_equidistant_from_origin_and_A : ∀ x y : ℝ,
  (x^2 + y^2 = 17^2) →
  ((x - 16)^2 + y^2 = 17^2) →
  ((x = 8 ∧ y = 15) ∨ (x = 8 ∧ y = -15)) :=
by sorry

end NUMINAMATH_CALUDE_point_equidistant_from_origin_and_A_l3276_327639


namespace NUMINAMATH_CALUDE_function_with_period_two_is_even_l3276_327627

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ is_periodic f p ∧ ∀ q, 0 < q ∧ q < p → ¬ is_periodic f q

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem function_with_period_two_is_even
  (f : ℝ → ℝ)
  (h_period : smallest_positive_period f 2)
  (h_symmetry : ∀ x, f (x + 2) = f (2 - x)) :
  is_even f :=
sorry

end NUMINAMATH_CALUDE_function_with_period_two_is_even_l3276_327627


namespace NUMINAMATH_CALUDE_total_points_target_l3276_327631

def average_points_after_two_games : ℝ := 61.5
def points_in_game_three : ℕ := 47
def additional_points_needed : ℕ := 330

theorem total_points_target :
  (2 * average_points_after_two_games + points_in_game_three + additional_points_needed : ℝ) = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_points_target_l3276_327631


namespace NUMINAMATH_CALUDE_quadratic_no_solution_l3276_327636

theorem quadratic_no_solution : 
  {x : ℝ | x^2 - 2*x + 3 = 0} = ∅ := by sorry

end NUMINAMATH_CALUDE_quadratic_no_solution_l3276_327636


namespace NUMINAMATH_CALUDE_ellipse_x_axis_iff_l3276_327604

/-- Defines an ellipse with foci on the x-axis -/
def is_ellipse_x_axis (k : ℝ) : Prop :=
  0 < k ∧ k < 2 ∧ ∀ x y : ℝ, x^2 / 2 + y^2 / k = 1 → 
    ∃ c : ℝ, c > 0 ∧ c < 1 ∧
      ∀ p : ℝ × ℝ, (p.1 - c)^2 + p.2^2 + (p.1 + c)^2 + p.2^2 = 2

/-- The condition 0 < k < 2 is necessary and sufficient for the equation
    x^2/2 + y^2/k = 1 to represent an ellipse with foci on the x-axis -/
theorem ellipse_x_axis_iff (k : ℝ) : is_ellipse_x_axis k ↔ (0 < k ∧ k < 2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_x_axis_iff_l3276_327604


namespace NUMINAMATH_CALUDE_complex_modulus_l3276_327699

theorem complex_modulus (z : ℂ) (h : z * (2 - Complex.I) = 1 + Complex.I) : Complex.abs z = Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3276_327699


namespace NUMINAMATH_CALUDE_grass_field_width_l3276_327696

/-- Proves that the width of a rectangular grass field is 192 meters, given specific conditions --/
theorem grass_field_width : 
  ∀ (w : ℝ),
  (82 * (w + 7) - 75 * w = 1918) →
  w = 192 := by
  sorry

end NUMINAMATH_CALUDE_grass_field_width_l3276_327696


namespace NUMINAMATH_CALUDE_parabola_transformation_l3276_327640

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  equation : ℝ → ℝ

/-- Shifts a parabola vertically -/
def vertical_shift (p : Parabola) (shift : ℝ) : Parabola :=
  { equation := λ x => p.equation x + shift }

/-- Shifts a parabola horizontally -/
def horizontal_shift (p : Parabola) (shift : ℝ) : Parabola :=
  { equation := λ x => p.equation (x - shift) }

theorem parabola_transformation (p : Parabola) (h : p.equation = λ x => 2 * x^2) :
  (horizontal_shift (vertical_shift p 3) 1).equation = λ x => 2 * (x - 1)^2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_transformation_l3276_327640


namespace NUMINAMATH_CALUDE_sin_1200_degrees_l3276_327600

theorem sin_1200_degrees : Real.sin (1200 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_1200_degrees_l3276_327600


namespace NUMINAMATH_CALUDE_super_eighteen_total_games_l3276_327695

/-- Calculates the total number of games in the Super Eighteen Football League -/
def super_eighteen_games (num_divisions : ℕ) (teams_per_division : ℕ) : ℕ :=
  let intra_division_games := num_divisions * teams_per_division * (teams_per_division - 1)
  let inter_division_games := num_divisions * teams_per_division * teams_per_division
  intra_division_games + inter_division_games

/-- Theorem stating that the Super Eighteen Football League schedules 450 games -/
theorem super_eighteen_total_games :
  super_eighteen_games 2 9 = 450 := by
  sorry

end NUMINAMATH_CALUDE_super_eighteen_total_games_l3276_327695


namespace NUMINAMATH_CALUDE_smallest_negative_integer_congruence_l3276_327697

theorem smallest_negative_integer_congruence :
  ∃ (x : ℤ), x < 0 ∧ (45 * x + 8) % 24 = 5 ∧
  ∀ (y : ℤ), y < 0 ∧ (45 * y + 8) % 24 = 5 → x ≥ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_negative_integer_congruence_l3276_327697


namespace NUMINAMATH_CALUDE_sum_remainder_mod_11_l3276_327625

theorem sum_remainder_mod_11 : (8735 + 8736 + 8737 + 8738) % 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_11_l3276_327625


namespace NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l3276_327661

theorem factorization_of_difference_of_squares (x : ℝ) :
  x^2 - 9 = (x + 3) * (x - 3) := by sorry

end NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l3276_327661


namespace NUMINAMATH_CALUDE_correct_average_l3276_327690

theorem correct_average (n : ℕ) (initial_avg : ℚ) (wrong_num correct_num : ℚ) :
  n = 10 →
  initial_avg = 15 →
  wrong_num = 26 →
  correct_num = 36 →
  (n : ℚ) * initial_avg + (correct_num - wrong_num) = n * 16 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l3276_327690


namespace NUMINAMATH_CALUDE_fuel_cost_savings_l3276_327674

theorem fuel_cost_savings
  (old_efficiency : ℝ)
  (old_fuel_cost : ℝ)
  (efficiency_increase : ℝ)
  (fuel_cost_increase : ℝ)
  (h1 : efficiency_increase = 0.6)
  (h2 : fuel_cost_increase = 0.3)
  : (1 - (1 + fuel_cost_increase) / (1 + efficiency_increase)) * 100 = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_fuel_cost_savings_l3276_327674


namespace NUMINAMATH_CALUDE_elberta_money_l3276_327681

theorem elberta_money (granny_smith : ℕ) (anjou elberta : ℝ) : 
  granny_smith = 72 →
  anjou = (1 / 4 : ℝ) * granny_smith →
  elberta = anjou + 3 →
  elberta = 21 := by sorry

end NUMINAMATH_CALUDE_elberta_money_l3276_327681


namespace NUMINAMATH_CALUDE_fib_mod_eq_closed_form_twelve_squared_eq_five_solutions_of_quadratic_inverse_of_twelve_l3276_327649

/-- The Fibonacci sequence modulo 139 -/
def fib_mod (n : ℕ) : Fin 139 :=
  if n = 0 then 0
  else if n = 1 then 1
  else (fib_mod (n - 1) + fib_mod (n - 2))

/-- The closed form expression for the Fibonacci sequence modulo 139 -/
def fib_closed_form (n : ℕ) : Fin 139 :=
  58 * (76^n - 64^n)

/-- Theorem stating that the Fibonacci sequence modulo 139 is equivalent to the closed form expression -/
theorem fib_mod_eq_closed_form (n : ℕ) : fib_mod n = fib_closed_form n := by
  sorry

/-- 12 is a solution of y² ≡ 5 (mod 139) -/
theorem twelve_squared_eq_five : (12 : Fin 139)^2 = 5 := by
  sorry

/-- 64 and 76 are solutions of x² - x - 1 ≡ 0 (mod 139) -/
theorem solutions_of_quadratic : 
  ((64 : Fin 139)^2 - 64 - 1 = 0) ∧ ((76 : Fin 139)^2 - 76 - 1 = 0) := by
  sorry

/-- 58 is the modular multiplicative inverse of 12 modulo 139 -/
theorem inverse_of_twelve : (12 : Fin 139) * 58 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fib_mod_eq_closed_form_twelve_squared_eq_five_solutions_of_quadratic_inverse_of_twelve_l3276_327649


namespace NUMINAMATH_CALUDE_selling_price_calculation_l3276_327603

def cost_price : ℕ := 50
def profit_rate : ℕ := 100

theorem selling_price_calculation (cost_price : ℕ) (profit_rate : ℕ) :
  cost_price = 50 → profit_rate = 100 → cost_price + (profit_rate * cost_price) / 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l3276_327603
