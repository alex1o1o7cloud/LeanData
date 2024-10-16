import Mathlib

namespace NUMINAMATH_CALUDE_correct_sampling_methods_l1104_110487

-- Define the types of sampling methods
inductive SamplingMethod
  | Systematic
  | Stratified
  | SimpleRandom

-- Define a situation
structure Situation where
  description : String
  populationSize : Nat
  sampleSize : Nat

-- Define a function to determine the appropriate sampling method
def appropriateSamplingMethod (s : Situation) : SamplingMethod :=
  sorry

-- Define the three situations
def situation1 : Situation :=
  { description := "Selecting 2 students from each class"
  , populationSize := 0  -- We don't know the exact population size
  , sampleSize := 2 }

def situation2 : Situation :=
  { description := "Selecting 12 students from a class with different score ranges"
  , populationSize := 62  -- 10 + 40 + 12
  , sampleSize := 12 }

def situation3 : Situation :=
  { description := "Arranging tracks for 6 students in a 400m final"
  , populationSize := 6
  , sampleSize := 6 }

-- Theorem stating the correct sampling methods for each situation
theorem correct_sampling_methods :
  (appropriateSamplingMethod situation1 = SamplingMethod.Systematic) ∧
  (appropriateSamplingMethod situation2 = SamplingMethod.Stratified) ∧
  (appropriateSamplingMethod situation3 = SamplingMethod.SimpleRandom) :=
  sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l1104_110487


namespace NUMINAMATH_CALUDE_point_movement_on_number_line_l1104_110492

theorem point_movement_on_number_line : 
  let start : ℤ := -2
  let move_right : ℤ := 7
  let move_left : ℤ := 4
  start + move_right - move_left = 1 :=
by sorry

end NUMINAMATH_CALUDE_point_movement_on_number_line_l1104_110492


namespace NUMINAMATH_CALUDE_exists_non_square_product_l1104_110420

theorem exists_non_square_product (a b : ℤ) (ha : a > 1) (hb : b > 1) (hab : a ≠ b) :
  ∃ n : ℕ+, ¬∃ m : ℤ, (a^n.val - 1) * (b^n.val - 1) = m^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_non_square_product_l1104_110420


namespace NUMINAMATH_CALUDE_no_one_blue_point_coloring_l1104_110413

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a color type
inductive Color
  | Red
  | Blue

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define a circle of radius 1
def unitCircle (center : Point) : Set Point :=
  {p : Point | (p.x - center.x)^2 + (p.y - center.y)^2 = 1}

-- State the theorem
theorem no_one_blue_point_coloring :
  ¬ (∀ (center : Point),
      ∃! (p : Point), p ∈ unitCircle center ∧ coloring p = Color.Blue) ∧
    (∃ (p q : Point), coloring p ≠ coloring q) :=
  sorry

end NUMINAMATH_CALUDE_no_one_blue_point_coloring_l1104_110413


namespace NUMINAMATH_CALUDE_thirty_five_power_pq_l1104_110406

theorem thirty_five_power_pq (p q : ℤ) (A B : ℝ) (hA : A = 5^p) (hB : B = 7^q) :
  A^q * B^p = 35^(p*q) := by
  sorry

end NUMINAMATH_CALUDE_thirty_five_power_pq_l1104_110406


namespace NUMINAMATH_CALUDE_inverse_exponential_is_logarithm_l1104_110495

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem inverse_exponential_is_logarithm (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : f a 2 = 1) : 
  ∀ x, f a x = Real.log x / Real.log 2 := by
sorry

end NUMINAMATH_CALUDE_inverse_exponential_is_logarithm_l1104_110495


namespace NUMINAMATH_CALUDE_quadratic_roots_real_and_equal_l1104_110491

theorem quadratic_roots_real_and_equal :
  let a : ℝ := 1
  let b : ℝ := -4 * Real.sqrt 2
  let c : ℝ := 8
  let discriminant := b^2 - 4*a*c
  discriminant = 0 ∧ ∃ x : ℝ, x^2 - 4*x*(Real.sqrt 2) + 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_real_and_equal_l1104_110491


namespace NUMINAMATH_CALUDE_lizas_rent_calculation_l1104_110422

def initial_balance : ℚ := 800
def paycheck : ℚ := 1500
def electricity_bill : ℚ := 117
def internet_bill : ℚ := 100
def phone_bill : ℚ := 70
def final_balance : ℚ := 1563

theorem lizas_rent_calculation :
  ∃ (rent : ℚ), 
    initial_balance - rent + paycheck - electricity_bill - internet_bill - phone_bill = final_balance ∧
    rent = 450 :=
by sorry

end NUMINAMATH_CALUDE_lizas_rent_calculation_l1104_110422


namespace NUMINAMATH_CALUDE_three_planes_intersection_count_l1104_110486

/-- A plane in 3D space -/
structure Plane3D where
  -- We don't need to define the internal structure of a plane for this problem

/-- Represents the intersection of two planes -/
inductive PlanesIntersection
  | Line
  | Empty

/-- Represents the number of intersection lines between three planes -/
inductive IntersectionCount
  | One
  | Three

/-- Function to determine how two planes intersect -/
def planesIntersect (p1 p2 : Plane3D) : PlanesIntersection :=
  sorry

/-- 
Given three planes in 3D space that intersect each other pairwise,
prove that the number of their intersection lines is either 1 or 3.
-/
theorem three_planes_intersection_count 
  (p1 p2 p3 : Plane3D)
  (h12 : planesIntersect p1 p2 = PlanesIntersection.Line)
  (h23 : planesIntersect p2 p3 = PlanesIntersection.Line)
  (h31 : planesIntersect p3 p1 = PlanesIntersection.Line) :
  ∃ (count : IntersectionCount), 
    (count = IntersectionCount.One ∨ count = IntersectionCount.Three) :=
by
  sorry

end NUMINAMATH_CALUDE_three_planes_intersection_count_l1104_110486


namespace NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_for_q_l1104_110459

theorem condition_p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, |x + 1| ≤ 2 → -3 ≤ x ∧ x ≤ 2) ∧
  (∃ x : ℝ, -3 ≤ x ∧ x ≤ 2 ∧ ¬(|x + 1| ≤ 2)) := by
  sorry

end NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_for_q_l1104_110459


namespace NUMINAMATH_CALUDE_caterpillars_left_tree_l1104_110416

/-- Proves that the number of caterpillars that left the tree is 8 --/
theorem caterpillars_left_tree (initial : ℕ) (hatched : ℕ) (final : ℕ) : 
  initial = 14 → hatched = 4 → final = 10 → initial + hatched - final = 8 := by
  sorry

end NUMINAMATH_CALUDE_caterpillars_left_tree_l1104_110416


namespace NUMINAMATH_CALUDE_unique_nine_digit_number_l1104_110471

def is_nine_digit (n : ℕ) : Prop := 100000000 ≤ n ∧ n ≤ 999999999

def sum_of_digits (n : ℕ) : ℕ := sorry

def product_of_digits (n : ℕ) : ℕ := sorry

def round_to_millions (n : ℕ) : ℕ := sorry

theorem unique_nine_digit_number :
  ∃! n : ℕ,
    is_nine_digit n ∧
    n % 2 = 1 ∧
    sum_of_digits n = 10 ∧
    product_of_digits n ≠ 0 ∧
    n % 7 = 0 ∧
    round_to_millions n = 112 :=
by sorry

end NUMINAMATH_CALUDE_unique_nine_digit_number_l1104_110471


namespace NUMINAMATH_CALUDE_correct_calculation_l1104_110421

theorem correct_calculation : 
  (Real.sqrt 27 / Real.sqrt 3 = 3) ∧ 
  (Real.sqrt 2 + Real.sqrt 3 ≠ Real.sqrt 5) ∧ 
  (5 * Real.sqrt 2 - 4 * Real.sqrt 2 ≠ 1) ∧ 
  (2 * Real.sqrt 3 * 3 * Real.sqrt 3 ≠ 6 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_correct_calculation_l1104_110421


namespace NUMINAMATH_CALUDE_power_equality_l1104_110447

theorem power_equality (x y : ℕ) :
  2 * (3^8)^2 * (2^3)^2 * 3 = 2^x * 3^y → x = 7 ∧ y = 17 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l1104_110447


namespace NUMINAMATH_CALUDE_linear_equation_rewrite_l1104_110496

theorem linear_equation_rewrite (k m : ℚ) : 
  (∀ x y : ℚ, 2 * x + 3 * y - 4 = 0 ↔ y = k * x + m) → 
  k + m = 2/3 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_rewrite_l1104_110496


namespace NUMINAMATH_CALUDE_siblings_age_sum_l1104_110494

theorem siblings_age_sum (R D S J : ℕ) : 
  R = D + 6 →
  D = S + 8 →
  J = R - 5 →
  R + 8 = 2 * (S + 8) →
  J + 10 = (D + 10) / 2 + 4 →
  (R - 3) + (D - 3) + (S - 3) + (J - 3) = 43 :=
by sorry

end NUMINAMATH_CALUDE_siblings_age_sum_l1104_110494


namespace NUMINAMATH_CALUDE_curve_is_two_lines_l1104_110484

/-- The equation of the curve -/
def curve_equation (x y : ℝ) : Prop := x^2 - x*y - 2*y^2 = 0

/-- The curve represents two straight lines -/
theorem curve_is_two_lines : 
  ∃ (a b c d : ℝ), ∀ (x y : ℝ), 
    curve_equation x y ↔ (a*x + b*y = 0 ∧ c*x + d*y = 0) :=
sorry

end NUMINAMATH_CALUDE_curve_is_two_lines_l1104_110484


namespace NUMINAMATH_CALUDE_eight_items_four_categories_l1104_110457

/-- The number of ways to assign n distinguishable items to k distinct categories -/
def assignments (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 65536 ways to assign 8 distinguishable items to 4 distinct categories -/
theorem eight_items_four_categories : assignments 8 4 = 65536 := by
  sorry

end NUMINAMATH_CALUDE_eight_items_four_categories_l1104_110457


namespace NUMINAMATH_CALUDE_parabola_translation_l1104_110415

/-- Represents a parabola in the form y = ax² + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically --/
def translate (p : Parabola) (h : ℝ) (k : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + k }

theorem parabola_translation :
  let original := Parabola.mk 1 0 0  -- y = x²
  let translated := translate original (-1) (-2)  -- 1 unit left, 2 units down
  translated = Parabola.mk 1 2 (-2)  -- y = (x+1)² - 2
  := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l1104_110415


namespace NUMINAMATH_CALUDE_student_rabbit_difference_l1104_110469

/-- Given 4 classrooms, each with 18 students and 2 rabbits, prove that the difference
    between the total number of students and rabbits is 64. -/
theorem student_rabbit_difference (num_classrooms : ℕ) (students_per_class : ℕ) (rabbits_per_class : ℕ)
    (h1 : num_classrooms = 4)
    (h2 : students_per_class = 18)
    (h3 : rabbits_per_class = 2) :
    num_classrooms * students_per_class - num_classrooms * rabbits_per_class = 64 := by
  sorry

end NUMINAMATH_CALUDE_student_rabbit_difference_l1104_110469


namespace NUMINAMATH_CALUDE_michael_remaining_yards_l1104_110463

/-- Represents the length of an ultra-marathon in miles and yards -/
structure UltraMarathon where
  miles : ℕ
  yards : ℕ

/-- Represents a distance in miles and yards -/
structure Distance where
  miles : ℕ
  yards : ℕ

def yards_per_mile : ℕ := 1760

def ultra_marathon : UltraMarathon := ⟨50, 800⟩

def michael_marathons : ℕ := 5

theorem michael_remaining_yards :
  ∃ (m : ℕ) (y : ℕ), 
    y < yards_per_mile ∧
    (michael_marathons * ultra_marathon.miles * yards_per_mile + 
     michael_marathons * ultra_marathon.yards) = 
    (m * yards_per_mile + y) ∧
    y = 480 := by
  sorry

end NUMINAMATH_CALUDE_michael_remaining_yards_l1104_110463


namespace NUMINAMATH_CALUDE_problem_statement_l1104_110400

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a * (6 - a) ≤ 9) ∧ 
  (a * b = a + b + 3 → a * b ≥ 9) ∧
  (a + b = 2 → 1 / a + 2 / b ≥ 3 / 2 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1104_110400


namespace NUMINAMATH_CALUDE_intersection_range_l1104_110417

/-- Given two curves C₁ and C₂, prove the range of m for which they intersect at exactly one point above the x-axis. -/
theorem intersection_range (a : ℝ) (h_a : a > 0) :
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (x^2 / a^2 + y^2 = 1 ∧ y^2 = 2*(x + m) ∧ y > 0) →
    (0 < a ∧ a < 1 → (m = (a^2 + 1) / 2 ∨ (-a < m ∧ m ≤ a))) ∧
    (a ≥ 1 → (-a < m ∧ m < a)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_l1104_110417


namespace NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l1104_110419

theorem ratio_of_sum_to_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) 
  (h : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l1104_110419


namespace NUMINAMATH_CALUDE_arithmetic_operations_l1104_110445

theorem arithmetic_operations : 
  (6 + (-8) - (-5) = 3) ∧ (18 / (-3) + (-2) * (-4) = 2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l1104_110445


namespace NUMINAMATH_CALUDE_dan_picked_more_apples_l1104_110409

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The number of apples Dan picked -/
def dan_apples : ℕ := 9

/-- Theorem: Dan picked 7 more apples than Benny -/
theorem dan_picked_more_apples : dan_apples - benny_apples = 7 := by
  sorry

end NUMINAMATH_CALUDE_dan_picked_more_apples_l1104_110409


namespace NUMINAMATH_CALUDE_component_unqualified_l1104_110477

/-- A component is qualified if its diameter is within the specified range. -/
def IsQualified (measured : ℝ) (nominal : ℝ) (tolerance : ℝ) : Prop :=
  nominal - tolerance ≤ measured ∧ measured ≤ nominal + tolerance

/-- The component is unqualified if it's not qualified. -/
def IsUnqualified (measured : ℝ) (nominal : ℝ) (tolerance : ℝ) : Prop :=
  ¬(IsQualified measured nominal tolerance)

theorem component_unqualified (measured : ℝ) (h : measured = 19.9) :
  IsUnqualified measured 20 0.02 := by
  sorry

#check component_unqualified

end NUMINAMATH_CALUDE_component_unqualified_l1104_110477


namespace NUMINAMATH_CALUDE_parallelogram_area_from_boards_l1104_110412

/-- The area of a parallelogram formed by two boards crossing at a 45-degree angle -/
theorem parallelogram_area_from_boards (board1_width board2_width : ℝ) 
  (h1 : board1_width = 5)
  (h2 : board2_width = 8)
  (h3 : Real.pi / 4 = 45 * Real.pi / 180) :
  board2_width * (board1_width * Real.sin (Real.pi / 4)) = 20 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_from_boards_l1104_110412


namespace NUMINAMATH_CALUDE_quadratic_one_root_l1104_110482

/-- A quadratic function with coefficients a = 1, b = 4, and c = n -/
def f (n : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + n

/-- The discriminant of the quadratic function f -/
def discriminant (n : ℝ) : ℝ := 4^2 - 4*1*n

theorem quadratic_one_root (n : ℝ) :
  (∃! x, f n x = 0) ↔ n = 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l1104_110482


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1104_110411

/-- An isosceles triangle with congruent sides of 6 cm and perimeter of 20 cm has a base of 8 cm. -/
theorem isosceles_triangle_base_length : 
  ∀ (base : ℝ), 
    base > 0 → 
    6 + 6 + base = 20 → 
    base = 8 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1104_110411


namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l1104_110476

theorem quadratic_real_roots_range (k : ℝ) :
  (∃ x : ℝ, x^2 - 3*x + k = 0) ↔ k ≤ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l1104_110476


namespace NUMINAMATH_CALUDE_imaginary_sum_equals_two_l1104_110407

def i : ℂ := Complex.I

theorem imaginary_sum_equals_two :
  i^15 + i^20 + i^25 + i^30 + i^35 + i^40 = (2 : ℂ) :=
by sorry

end NUMINAMATH_CALUDE_imaginary_sum_equals_two_l1104_110407


namespace NUMINAMATH_CALUDE_no_three_digit_even_sum_27_l1104_110451

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is even -/
def is_even (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number has three digits -/
def is_three_digit (n : ℕ) : Prop := sorry

theorem no_three_digit_even_sum_27 :
  ¬ ∃ n : ℕ, is_three_digit n ∧ is_even n ∧ digit_sum n = 27 := by sorry

end NUMINAMATH_CALUDE_no_three_digit_even_sum_27_l1104_110451


namespace NUMINAMATH_CALUDE_admission_probability_l1104_110461

theorem admission_probability (P_A P_B : ℝ) (h_A : P_A = 0.6) (h_B : P_B = 0.7) 
  (h_indep : P_A + P_B - P_A * P_B = P_A + P_B - (P_A * P_B)) : 
  P_A + P_B - P_A * P_B = 0.88 := by
sorry

end NUMINAMATH_CALUDE_admission_probability_l1104_110461


namespace NUMINAMATH_CALUDE_car_collision_frequency_l1104_110470

theorem car_collision_frequency :
  ∀ (x : ℝ),
    (x > 0) →
    (240 / x + 240 / 20 = 36) →
    x = 10 :=
by
  sorry

#check car_collision_frequency

end NUMINAMATH_CALUDE_car_collision_frequency_l1104_110470


namespace NUMINAMATH_CALUDE_total_goals_in_five_matches_l1104_110448

/-- A football player's goal scoring record over 5 matches -/
structure FootballPlayer where
  /-- The average number of goals per match before the fifth match -/
  initial_average : ℝ
  /-- The number of goals scored in the fifth match -/
  fifth_match_goals : ℕ
  /-- The increase in average goals after the fifth match -/
  average_increase : ℝ

/-- Theorem stating the total number of goals scored over 5 matches -/
theorem total_goals_in_five_matches (player : FootballPlayer)
    (h1 : player.fifth_match_goals = 2)
    (h2 : player.average_increase = 0.3) :
    (player.initial_average * 4 + player.fifth_match_goals : ℝ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_total_goals_in_five_matches_l1104_110448


namespace NUMINAMATH_CALUDE_milk_remainder_l1104_110489

theorem milk_remainder (initial_milk : ℚ) (given_away : ℚ) (remainder : ℚ) : 
  initial_milk = 4 → given_away = 7/3 → remainder = initial_milk - given_away → remainder = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_milk_remainder_l1104_110489


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l1104_110479

/-- 
Given an arithmetic series of consecutive integers with first term (k^2 - k + 1),
prove that the sum of the first (k + 2) terms is equal to k^3 + (3k^2)/2 + k/2 + 2.
-/
theorem arithmetic_series_sum (k : ℕ) : 
  let a₁ : ℚ := k^2 - k + 1
  let n : ℕ := k + 2
  let S := (n : ℚ) / 2 * (a₁ + (a₁ + (n - 1)))
  S = k^3 + (3 * k^2) / 2 + k / 2 + 2 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_series_sum_l1104_110479


namespace NUMINAMATH_CALUDE_range_of_x_l1104_110444

/-- Given a set M containing two elements, x^2 - 5x + 7 and 1, 
    prove that the range of real numbers x is all real numbers except 2 and 3. -/
theorem range_of_x (M : Set ℝ) (h : M = {x^2 - 5*x + 7 | x : ℝ} ∪ {1}) :
  {x : ℝ | x^2 - 5*x + 7 ≠ 1} = {x : ℝ | x ≠ 2 ∧ x ≠ 3} :=
sorry

end NUMINAMATH_CALUDE_range_of_x_l1104_110444


namespace NUMINAMATH_CALUDE_power_two_2005_mod_7_l1104_110462

theorem power_two_2005_mod_7 : 2^2005 % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_two_2005_mod_7_l1104_110462


namespace NUMINAMATH_CALUDE_suitcase_electronics_weight_l1104_110467

/-- Given a suitcase with books, clothes, and electronics, prove the weight of electronics. -/
theorem suitcase_electronics_weight 
  (B C E : ℝ) -- Weights of books, clothes, and electronics
  (h1 : B / C = 7 / 4) -- Initial ratio of books to clothes
  (h2 : C / E = 4 / 3) -- Initial ratio of clothes to electronics
  (h3 : B / (C - 6) = 2 * (B / C)) -- Ratio doubles after removing 6 pounds of clothes
  : E = 9 := by
  sorry

end NUMINAMATH_CALUDE_suitcase_electronics_weight_l1104_110467


namespace NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l1104_110441

theorem largest_n_satisfying_conditions : 
  ∃ (n : ℤ), n = 313 ∧ 
  (∀ (x : ℤ), x > n → 
    (¬∃ (m : ℤ), x^2 = (m+1)^3 - m^3) ∨ 
    (¬∃ (k : ℤ), 2*x + 103 = k^2)) ∧
  (∃ (m : ℤ), n^2 = (m+1)^3 - m^3) ∧
  (∃ (k : ℤ), 2*n + 103 = k^2) := by
sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l1104_110441


namespace NUMINAMATH_CALUDE_parabola_intersection_length_l1104_110418

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (A B : ℝ × ℝ) : Prop :=
  (A.1 - focus.1) * (B.2 - focus.2) = (B.1 - focus.1) * (A.2 - focus.2)

-- Define the condition that A and B are on the parabola
def points_on_parabola (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2

-- Define the sum of x-coordinates condition
def sum_of_x_coordinates (A B : ℝ × ℝ) : Prop :=
  A.1 + B.1 = 3

-- The main theorem
theorem parabola_intersection_length (A B : ℝ × ℝ) :
  line_through_focus A B →
  points_on_parabola A B →
  sum_of_x_coordinates A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_length_l1104_110418


namespace NUMINAMATH_CALUDE_solution_when_m_3_no_solution_conditions_l1104_110488

-- Define the fractional equation
def fractional_equation (x m : ℝ) : Prop :=
  (3 - 2*x) / (x - 2) - (m*x - 2) / (2 - x) = -1

-- Theorem 1: When m = 3, the solution is x = 1/2
theorem solution_when_m_3 :
  ∃ x : ℝ, fractional_equation x 3 ∧ x = 1/2 :=
sorry

-- Theorem 2: The equation has no solution when m = 1 or m = 3/2
theorem no_solution_conditions :
  (∀ x : ℝ, ¬ fractional_equation x 1) ∧
  (∀ x : ℝ, ¬ fractional_equation x (3/2)) :=
sorry

end NUMINAMATH_CALUDE_solution_when_m_3_no_solution_conditions_l1104_110488


namespace NUMINAMATH_CALUDE_angle_and_function_properties_l1104_110405

-- Define the angle equivalence relation
def angle_equiv (a b : ℝ) : Prop := ∃ k : ℤ, a = b + k * 360

-- Define evenness for functions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Theorem statement
theorem angle_and_function_properties :
  (angle_equiv (-497) 2023) ∧
  (is_even_function (λ x => Real.sin ((2/3)*x - 7*Real.pi/2))) :=
by sorry

end NUMINAMATH_CALUDE_angle_and_function_properties_l1104_110405


namespace NUMINAMATH_CALUDE_longer_train_length_l1104_110497

/-- Calculates the length of the longer train given the speeds of two trains,
    the time they take to cross each other, and the length of the shorter train. -/
theorem longer_train_length
  (speed1 speed2 : ℝ)
  (crossing_time : ℝ)
  (shorter_train_length : ℝ)
  (h1 : speed1 = 68)
  (h2 : speed2 = 40)
  (h3 : crossing_time = 11.999040076793857)
  (h4 : shorter_train_length = 160)
  (h5 : speed1 > 0)
  (h6 : speed2 > 0)
  (h7 : crossing_time > 0)
  (h8 : shorter_train_length > 0) :
  let relative_speed := (speed1 + speed2) * (1000 / 3600)
  let total_distance := relative_speed * crossing_time
  total_distance - shorter_train_length = 200 := by
  sorry

end NUMINAMATH_CALUDE_longer_train_length_l1104_110497


namespace NUMINAMATH_CALUDE_simplify_expression_l1104_110423

theorem simplify_expression (x : ℝ) : 4*x - 3*x^2 + 6 + (8 - 5*x + 2*x^2) = -x^2 - x + 14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1104_110423


namespace NUMINAMATH_CALUDE_lives_lost_l1104_110483

theorem lives_lost (starting_lives ending_lives : ℕ) 
  (h1 : starting_lives = 98)
  (h2 : ending_lives = 73) :
  starting_lives - ending_lives = 25 := by
  sorry

end NUMINAMATH_CALUDE_lives_lost_l1104_110483


namespace NUMINAMATH_CALUDE_exactly_one_machine_maintenance_probability_l1104_110446

/-- The probability that exactly one of three independent machines needs maintenance,
    given their individual maintenance probabilities. -/
theorem exactly_one_machine_maintenance_probability
  (p_A p_B p_C : ℝ)
  (h_A : 0 ≤ p_A ∧ p_A ≤ 1)
  (h_B : 0 ≤ p_B ∧ p_B ≤ 1)
  (h_C : 0 ≤ p_C ∧ p_C ≤ 1)
  (h_p_A : p_A = 0.1)
  (h_p_B : p_B = 0.2)
  (h_p_C : p_C = 0.4) :
  p_A * (1 - p_B) * (1 - p_C) +
  (1 - p_A) * p_B * (1 - p_C) +
  (1 - p_A) * (1 - p_B) * p_C = 0.444 :=
sorry

end NUMINAMATH_CALUDE_exactly_one_machine_maintenance_probability_l1104_110446


namespace NUMINAMATH_CALUDE_ryan_overall_score_l1104_110480

def first_test_questions : ℕ := 30
def first_test_score : ℚ := 85 / 100

def second_test_math_questions : ℕ := 20
def second_test_math_score : ℚ := 95 / 100
def second_test_science_questions : ℕ := 15
def second_test_science_score : ℚ := 80 / 100

def third_test_questions : ℕ := 15
def third_test_score : ℚ := 65 / 100

theorem ryan_overall_score :
  let total_questions := first_test_questions + second_test_math_questions + second_test_science_questions + third_test_questions
  let correct_answers := (first_test_questions : ℚ) * first_test_score +
                         (second_test_math_questions : ℚ) * second_test_math_score +
                         (second_test_science_questions : ℚ) * second_test_science_score +
                         (third_test_questions : ℚ) * third_test_score
  correct_answers / (total_questions : ℚ) = 8281 / 10000 :=
by sorry

end NUMINAMATH_CALUDE_ryan_overall_score_l1104_110480


namespace NUMINAMATH_CALUDE_D_72_l1104_110490

/-- The number of ways to write a positive integer as a product of integers greater than 1, considering order. -/
def D (n : ℕ+) : ℕ :=
  sorry

/-- The prime factorization of 72 is 2^3 * 3^2 -/
axiom prime_factorization_72 : ∃ (a b : ℕ), 72 = 2^3 * 3^2

/-- Theorem: The number of ways to write 72 as a product of integers greater than 1, considering order, is 26 -/
theorem D_72 : D 72 = 26 := by
  sorry

end NUMINAMATH_CALUDE_D_72_l1104_110490


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_l1104_110442

-- Define the total number of chocolate bars
def total_bars : ℕ := 9

-- Define the number of unsold bars
def unsold_bars : ℕ := 3

-- Define the total amount made from the sale
def total_amount : ℕ := 18

-- Theorem to prove
theorem chocolate_bar_cost :
  ∃ (cost : ℚ), cost * (total_bars - unsold_bars) = total_amount ∧ cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_l1104_110442


namespace NUMINAMATH_CALUDE_square_side_length_l1104_110458

theorem square_side_length (circle_area : ℝ) (h1 : circle_area = 100) :
  let square_perimeter := circle_area
  let square_side := square_perimeter / 4
  square_side = 25 := by sorry

end NUMINAMATH_CALUDE_square_side_length_l1104_110458


namespace NUMINAMATH_CALUDE_system_solution_l1104_110431

theorem system_solution (x y z u : ℝ) : 
  (x^3 * y^2 * z = 2 ∧ 
   z^3 * u^2 * x = 32 ∧ 
   y^3 * z^2 * u = 8 ∧ 
   u^3 * x^2 * y = 8) ↔ 
  ((x = 1 ∧ y = 1 ∧ z = 2 ∧ u = 2) ∨
   (x = 1 ∧ y = -1 ∧ z = 2 ∧ u = -2) ∨
   (x = -1 ∧ y = 1 ∧ z = -2 ∧ u = 2) ∨
   (x = -1 ∧ y = -1 ∧ z = -2 ∧ u = -2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1104_110431


namespace NUMINAMATH_CALUDE_other_communities_count_l1104_110439

/-- The number of boys belonging to other communities in a school with given total and percentages of specific communities -/
theorem other_communities_count (total : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total = 850 →
  muslim_percent = 44 / 100 →
  hindu_percent = 14 / 100 →
  sikh_percent = 10 / 100 →
  ↑total * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 272 := by
  sorry

#check other_communities_count

end NUMINAMATH_CALUDE_other_communities_count_l1104_110439


namespace NUMINAMATH_CALUDE_pressure_change_l1104_110443

/-- Given a relationship between pressure (P), area (A), and velocity (V),
    prove that doubling the area and increasing velocity from 20 to 30
    results in a specific pressure change. -/
theorem pressure_change (k : ℝ) :
  (∃ (P₀ A₀ V₀ : ℝ), P₀ = k * A₀ * V₀^2 ∧ P₀ = 0.5 ∧ A₀ = 1 ∧ V₀ = 20) →
  (∃ (P₁ A₁ V₁ : ℝ), P₁ = k * A₁ * V₁^2 ∧ A₁ = 2 ∧ V₁ = 30 ∧ P₁ = 2.25) :=
by sorry

end NUMINAMATH_CALUDE_pressure_change_l1104_110443


namespace NUMINAMATH_CALUDE_max_m_value_l1104_110468

theorem max_m_value (A B C D : ℝ × ℝ) (m : ℝ) : 
  A = (1, 0) →
  B = (0, 1) →
  C = (a, b) →
  D = (c, d) →
  (∀ a b c d : ℝ, 
    (c - a)^2 + (d - b)^2 ≥ 
    (m - 2) * (a * c + b * d) + 
    m * (a * 0 + b * 1) * (c * 1 + d * 0)) →
  ∃ m_max : ℝ, m_max = Real.sqrt 5 - 1 ∧ 
    (∀ m' : ℝ, (∀ a b c d : ℝ, 
      (c - a)^2 + (d - b)^2 ≥ 
      (m' - 2) * (a * c + b * d) + 
      m' * (a * 0 + b * 1) * (c * 1 + d * 0)) → 
    m' ≤ m_max) :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l1104_110468


namespace NUMINAMATH_CALUDE_infinite_centers_of_symmetry_l1104_110472

/-- A type representing a geometric figure. -/
structure Figure where
  -- Add necessary fields here
  mk :: -- Constructor

/-- A type representing a point in the figure. -/
structure Point where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a symmetry operation on a figure. -/
def SymmetryOperation : Type := Figure → Figure

/-- Represents a center of symmetry for a figure. -/
def CenterOfSymmetry (f : Figure) : Type := Point

/-- Composition of symmetry operations. -/
def composeSymmetry (s1 s2 : SymmetryOperation) : SymmetryOperation :=
  fun f => s1 (s2 f)

/-- 
  If a figure has more than one center of symmetry, 
  it must have infinitely many centers of symmetry.
-/
theorem infinite_centers_of_symmetry (f : Figure) :
  (∃ (c1 c2 : CenterOfSymmetry f), c1 ≠ c2) →
  ∀ n : ℕ, ∃ (centers : Finset (CenterOfSymmetry f)), centers.card > n :=
sorry

end NUMINAMATH_CALUDE_infinite_centers_of_symmetry_l1104_110472


namespace NUMINAMATH_CALUDE_fermat_number_prime_factor_l1104_110499

theorem fermat_number_prime_factor (n : ℕ) (hn : n ≥ 3) :
  ∃ p : ℕ, Prime p ∧ p ∣ (2^(2^n) + 1) ∧ p > 2^(n+2) * (n+1) := by
  sorry

end NUMINAMATH_CALUDE_fermat_number_prime_factor_l1104_110499


namespace NUMINAMATH_CALUDE_research_team_probabilities_l1104_110424

/-- Represents a research team member -/
structure Member where
  gender : Bool  -- true for male, false for female
  speaksEnglish : Bool

/-- Represents a research team -/
def ResearchTeam : Type := List Member

/-- Creates a research team with the given specifications -/
def createTeam : ResearchTeam :=
  [
    { gender := true, speaksEnglish := false },   -- non-English speaking male
    { gender := true, speaksEnglish := true },    -- English speaking male
    { gender := true, speaksEnglish := true },    -- English speaking male
    { gender := true, speaksEnglish := true },    -- English speaking male
    { gender := false, speaksEnglish := false },  -- non-English speaking female
    { gender := false, speaksEnglish := true }    -- English speaking female
  ]

/-- Calculates the probability of selecting two members with a given property -/
def probabilityOfSelection (team : ResearchTeam) (property : Member → Member → Bool) : Rat :=
  sorry

theorem research_team_probabilities (team : ResearchTeam) 
  (h1 : team = createTeam) : 
  (probabilityOfSelection team (fun m1 m2 => m1.gender = m2.gender) = 7/15) ∧ 
  (probabilityOfSelection team (fun m1 m2 => m1.speaksEnglish ∨ m2.speaksEnglish) = 14/15) ∧
  (probabilityOfSelection team (fun m1 m2 => m1.gender ≠ m2.gender ∧ (m1.speaksEnglish ∨ m2.speaksEnglish)) = 7/15) :=
by sorry

end NUMINAMATH_CALUDE_research_team_probabilities_l1104_110424


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l1104_110465

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  max (a + 1/b) (max (b + 1/c) (c + 1/a)) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l1104_110465


namespace NUMINAMATH_CALUDE_x_gt_1_sufficient_not_necessary_for_x_sq_gt_x_l1104_110454

theorem x_gt_1_sufficient_not_necessary_for_x_sq_gt_x :
  (∀ x : ℝ, x > 1 → x^2 > x) ∧ 
  (∃ x : ℝ, x^2 > x ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_1_sufficient_not_necessary_for_x_sq_gt_x_l1104_110454


namespace NUMINAMATH_CALUDE_journey_takes_four_days_l1104_110434

/-- Represents the journey of a young man returning home from vacation. -/
structure Journey where
  totalDistance : ℕ
  firstLegDistance : ℕ
  secondLegDistance : ℕ
  totalDays : ℕ
  remainingDays : ℕ

/-- Checks if the journey satisfies the given conditions. -/
def isValidJourney (j : Journey) : Prop :=
  j.totalDistance = j.firstLegDistance + j.secondLegDistance ∧
  j.firstLegDistance = 246 ∧
  j.secondLegDistance = 276 ∧
  j.totalDays - j.remainingDays = j.remainingDays / 2 + 1 ∧
  j.totalDays > 0 ∧
  j.remainingDays > 0

/-- Theorem stating that the journey takes 4 days in total. -/
theorem journey_takes_four_days :
  ∃ (j : Journey), isValidJourney j ∧ j.totalDays = 4 :=
sorry


end NUMINAMATH_CALUDE_journey_takes_four_days_l1104_110434


namespace NUMINAMATH_CALUDE_perfect_square_between_prime_sums_l1104_110437

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The sum of the first n prime numbers -/
def sumOfFirstNPrimes (n : ℕ) : ℕ := (List.range n).map (nthPrime ∘ (· + 1)) |>.sum

/-- There exists a perfect square between the sum of the first n primes and the sum of the first n+1 primes -/
theorem perfect_square_between_prime_sums (n : ℕ) :
  ∃ k : ℕ, sumOfFirstNPrimes n < k^2 ∧ k^2 < sumOfFirstNPrimes (n + 1) := by sorry

end NUMINAMATH_CALUDE_perfect_square_between_prime_sums_l1104_110437


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l1104_110404

theorem inequality_system_solution_set :
  {x : ℝ | 6 > 2 * (x + 1) ∧ 1 - x < 2} = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l1104_110404


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_smallest_odd_primes_l1104_110428

theorem smallest_four_digit_divisible_by_smallest_odd_primes : 
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 → (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ (11 ∣ n) → n ≥ 1155 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_smallest_odd_primes_l1104_110428


namespace NUMINAMATH_CALUDE_problem_statement_l1104_110402

theorem problem_statement :
  (¬ (∃ x : ℝ, Real.tan x = 1 ∧ ∃ x : ℝ, x^2 - x + 1 ≤ 0)) ∧
  (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0) ↔ (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1104_110402


namespace NUMINAMATH_CALUDE_angle_ratio_not_right_triangle_l1104_110433

/-- Triangle ABC with angles A, B, and C in the ratio 3:4:5 is not necessarily a right triangle -/
theorem angle_ratio_not_right_triangle (A B C : ℝ) : 
  A / B = 3 / 4 ∧ B / C = 4 / 5 ∧ A + B + C = π → 
  ¬ (A = π / 2 ∨ B = π / 2 ∨ C = π / 2) :=
by sorry

end NUMINAMATH_CALUDE_angle_ratio_not_right_triangle_l1104_110433


namespace NUMINAMATH_CALUDE_triangle_angle_obtuse_l1104_110466

theorem triangle_angle_obtuse (α : Real) (h1 : 0 < α ∧ α < π) 
  (h2 : Real.sin α + Real.cos α = 2/3) : α > π/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_obtuse_l1104_110466


namespace NUMINAMATH_CALUDE_church_distance_l1104_110425

theorem church_distance (horse_speed : ℝ) (hourly_rate : ℝ) (flat_fee : ℝ) (total_paid : ℝ) 
  (h1 : horse_speed = 10)
  (h2 : hourly_rate = 30)
  (h3 : flat_fee = 20)
  (h4 : total_paid = 80) : 
  (total_paid - flat_fee) / hourly_rate * horse_speed = 20 := by
  sorry

#check church_distance

end NUMINAMATH_CALUDE_church_distance_l1104_110425


namespace NUMINAMATH_CALUDE_adams_age_problem_l1104_110449

theorem adams_age_problem :
  ∃! x : ℕ,
    x > 0 ∧
    ∃ m : ℕ, x - 2 = m ^ 2 ∧
    ∃ n : ℕ, x + 2 = n ^ 3 ∧
    x = 6 := by
  sorry

end NUMINAMATH_CALUDE_adams_age_problem_l1104_110449


namespace NUMINAMATH_CALUDE_problem_statement_l1104_110426

/-- The equation x^2 - x + a^2 - 6a = 0 has one positive root and one negative root. -/
def p (a : ℝ) : Prop :=
  ∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 - x + a^2 - 6*a = 0 ∧ y^2 - y + a^2 - 6*a = 0

/-- The graph of y = x^2 + (a-3)x + 1 has no common points with the x-axis. -/
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + (a-3)*x + 1 ≠ 0

/-- The range of values for a is 0 < a ≤ 1 or 5 ≤ a < 6. -/
def range_of_a (a : ℝ) : Prop :=
  (0 < a ∧ a ≤ 1) ∨ (5 ≤ a ∧ a < 6)

theorem problem_statement (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_of_a a := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1104_110426


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1104_110493

theorem fraction_sum_equality (m n p : ℝ) 
  (h : m / (140 - m) + n / (210 - n) + p / (180 - p) = 9) :
  10 / (140 - m) + 14 / (210 - n) + 12 / (180 - p) = 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1104_110493


namespace NUMINAMATH_CALUDE_second_group_size_l1104_110440

/-- Represents a tour group with a number of people -/
structure TourGroup where
  people : ℕ

/-- Represents a day's tour schedule -/
structure TourSchedule where
  group1 : TourGroup
  group2 : TourGroup
  group3 : TourGroup
  group4 : TourGroup

def questions_per_tourist : ℕ := 2

def total_questions : ℕ := 68

theorem second_group_size (schedule : TourSchedule) : 
  schedule.group1.people = 6 ∧ 
  schedule.group3.people = 8 ∧ 
  schedule.group4.people = 7 ∧
  questions_per_tourist * (schedule.group1.people + schedule.group2.people + schedule.group3.people + schedule.group4.people) = total_questions →
  schedule.group2.people = 13 := by
  sorry

end NUMINAMATH_CALUDE_second_group_size_l1104_110440


namespace NUMINAMATH_CALUDE_actual_weight_of_three_bags_l1104_110481

/-- The actual weight of three bags of food given their labeled weight and deviations -/
theorem actual_weight_of_three_bags 
  (labeled_weight : ℕ) 
  (num_bags : ℕ) 
  (deviation1 deviation2 deviation3 : ℤ) : 
  labeled_weight = 200 → 
  num_bags = 3 → 
  deviation1 = 10 → 
  deviation2 = -16 → 
  deviation3 = -11 → 
  (labeled_weight * num_bags : ℤ) + deviation1 + deviation2 + deviation3 = 583 := by
  sorry

end NUMINAMATH_CALUDE_actual_weight_of_three_bags_l1104_110481


namespace NUMINAMATH_CALUDE_place_mat_length_l1104_110435

theorem place_mat_length (r : ℝ) (n : ℕ) (h_r : r = 5) (h_n : n = 8) : 
  let x := 2 * r * Real.sin (π / (2 * n))
  x = r * Real.sqrt (2 - Real.sqrt 2) := by
  sorry

#check place_mat_length

end NUMINAMATH_CALUDE_place_mat_length_l1104_110435


namespace NUMINAMATH_CALUDE_martha_blocks_theorem_l1104_110430

/-- The number of blocks Martha starts with -/
def starting_blocks : ℕ := 11

/-- The number of blocks Martha finds -/
def found_blocks : ℕ := 129

/-- The total number of blocks Martha ends up with -/
def total_blocks : ℕ := starting_blocks + found_blocks

theorem martha_blocks_theorem : total_blocks = 140 := by
  sorry

end NUMINAMATH_CALUDE_martha_blocks_theorem_l1104_110430


namespace NUMINAMATH_CALUDE_light_pulse_reflections_l1104_110473

theorem light_pulse_reflections :
  ∃ (n : ℕ), n > 0 ∧
  (∃ (a b : ℕ), (a + 2) * (b + 2) = 4042 ∧ Nat.gcd (a + 1) (b + 1) = 1 ∧ n = a + b) ∧
  (∀ (m : ℕ), m > 0 →
    (∃ (a b : ℕ), (a + 2) * (b + 2) = 4042 ∧ Nat.gcd (a + 1) (b + 1) = 1 ∧ m = a + b) →
    m ≥ n) ∧
  n = 129 :=
by sorry

end NUMINAMATH_CALUDE_light_pulse_reflections_l1104_110473


namespace NUMINAMATH_CALUDE_parking_theorem_l1104_110452

/-- The number of parking spaces -/
def total_spaces : ℕ := 7

/-- The number of cars to be parked -/
def num_cars : ℕ := 3

/-- The number of spaces that must remain empty and connected -/
def empty_spaces : ℕ := 4

/-- The number of possible positions for the block of empty spaces -/
def empty_block_positions : ℕ := total_spaces - empty_spaces + 1

/-- The number of distinct parking arrangements -/
def parking_arrangements : ℕ := empty_block_positions * (Nat.factorial num_cars)

theorem parking_theorem : parking_arrangements = 24 := by
  sorry

end NUMINAMATH_CALUDE_parking_theorem_l1104_110452


namespace NUMINAMATH_CALUDE_katies_games_l1104_110455

theorem katies_games (friends_games : ℕ) (katies_extra_games : ℕ) 
  (h1 : friends_games = 59)
  (h2 : katies_extra_games = 22) : 
  friends_games + katies_extra_games = 81 :=
by sorry

end NUMINAMATH_CALUDE_katies_games_l1104_110455


namespace NUMINAMATH_CALUDE_cylinder_height_l1104_110410

/-- A cylinder with given lateral area and volume has height 3 -/
theorem cylinder_height (r h : ℝ) (h_positive : h > 0) (r_positive : r > 0) 
  (lateral_area : 2 * π * r * h = 12 * π) 
  (volume : π * r^2 * h = 12 * π) : h = 3 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_l1104_110410


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l1104_110403

def set_A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

def set_B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + (a^2-5) = 0}

theorem intersection_implies_a_value :
  ∀ a : ℝ, set_A ∩ set_B a = {2} → a = -1 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l1104_110403


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l1104_110475

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (l : Line) (α β : Plane) 
  (h1 : parallel l α) 
  (h2 : perpendicular l β) : 
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l1104_110475


namespace NUMINAMATH_CALUDE_trouser_original_price_l1104_110498

theorem trouser_original_price (sale_price : ℝ) (discount_percentage : ℝ) (original_price : ℝ) :
  sale_price = 10 →
  discount_percentage = 90 →
  sale_price = original_price * (1 - discount_percentage / 100) →
  original_price = 100 :=
by sorry

end NUMINAMATH_CALUDE_trouser_original_price_l1104_110498


namespace NUMINAMATH_CALUDE_common_factor_of_polynomial_l1104_110414

theorem common_factor_of_polynomial (a b : ℤ) :
  ∃ (k : ℤ), (6 * a^2 * b - 3 * a * b^2) = k * (3 * a * b) :=
by sorry

end NUMINAMATH_CALUDE_common_factor_of_polynomial_l1104_110414


namespace NUMINAMATH_CALUDE_line_parallel_plane_iff_no_common_points_l1104_110474

-- Define a type for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for lines in 3D space
structure Line3D where
  point : Point3D
  direction : Point3D

-- Define a type for planes in 3D space
structure Plane3D where
  point : Point3D
  normal : Point3D

-- Define what it means for a line to be parallel to a plane
def is_parallel (l : Line3D) (p : Plane3D) : Prop :=
  l.direction.x * p.normal.x + l.direction.y * p.normal.y + l.direction.z * p.normal.z = 0

-- Define what it means for a line and a plane to have no common points
def no_common_points (l : Line3D) (p : Plane3D) : Prop :=
  ∀ t : ℝ, 
    (l.point.x + t * l.direction.x - p.point.x) * p.normal.x +
    (l.point.y + t * l.direction.y - p.point.y) * p.normal.y +
    (l.point.z + t * l.direction.z - p.point.z) * p.normal.z ≠ 0

-- State the theorem
theorem line_parallel_plane_iff_no_common_points (l : Line3D) (p : Plane3D) :
  is_parallel l p ↔ no_common_points l p :=
sorry

end NUMINAMATH_CALUDE_line_parallel_plane_iff_no_common_points_l1104_110474


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_y_eq_x_l1104_110401

def is_symmetric_point (p1 p2 : ℝ × ℝ) : Prop :=
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint.1 = midpoint.2 ∧ (p2.2 - p1.2) / (p2.1 - p1.1) = -1

theorem symmetric_point_wrt_y_eq_x : 
  is_symmetric_point (3, 1) (1, 3) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_y_eq_x_l1104_110401


namespace NUMINAMATH_CALUDE_combined_selling_price_l1104_110485

/-- Calculate the combined selling price of three articles given their cost prices and profit/loss percentages. -/
theorem combined_selling_price (cost1 cost2 cost3 : ℝ) : 
  cost1 = 70 →
  cost2 = 120 →
  cost3 = 150 →
  ∃ (sell1 sell2 sell3 : ℝ),
    (2/3 * sell1 = 0.85 * cost1) ∧
    (sell2 = cost2 * 1.3) ∧
    (sell3 = cost3 * 0.8) ∧
    (sell1 + sell2 + sell3 = 365.25) := by
  sorry

#check combined_selling_price

end NUMINAMATH_CALUDE_combined_selling_price_l1104_110485


namespace NUMINAMATH_CALUDE_apple_weight_l1104_110408

theorem apple_weight (total_weight orange_weight grape_weight strawberry_weight : ℝ) 
  (h1 : total_weight = 10)
  (h2 : orange_weight = 1)
  (h3 : grape_weight = 3)
  (h4 : strawberry_weight = 3) :
  total_weight - (orange_weight + grape_weight + strawberry_weight) = 3 :=
by sorry

end NUMINAMATH_CALUDE_apple_weight_l1104_110408


namespace NUMINAMATH_CALUDE_ellipse_intersection_parallel_line_l1104_110450

-- Define the ellipse C
def ellipse_C (b : ℝ) (x y : ℝ) : Prop :=
  b > 0 ∧ x^2 / (5 * b^2) + y^2 / b^2 = 1

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a line on the plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the focus of the ellipse
def focus : Point := ⟨2, 0⟩

-- Define point E
def point_E : Point := ⟨3, 0⟩

-- Define the line x = 5
def line_x_5 : Line := ⟨1, 0, -5⟩

-- Define the property of line l passing through (1,0) and not coinciding with x-axis
def line_l_property (l : Line) : Prop :=
  l.a * 1 + l.b * 0 + l.c = 0 ∧ l.b ≠ 0

-- Define the intersection of a line and the ellipse
def intersect_line_ellipse (l : Line) (b : ℝ) : Prop :=
  ∃ (M N : Point), 
    line_l_property l ∧
    ellipse_C b M.x M.y ∧ 
    ellipse_C b N.x N.y ∧
    l.a * M.x + l.b * M.y + l.c = 0 ∧
    l.a * N.x + l.b * N.y + l.c = 0

-- Define the intersection of two lines
def intersect_lines (l1 l2 : Line) (P : Point) : Prop :=
  l1.a * P.x + l1.b * P.y + l1.c = 0 ∧
  l2.a * P.x + l2.b * P.y + l2.c = 0

-- Define parallel lines
def parallel_lines (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

-- The main theorem
theorem ellipse_intersection_parallel_line (b : ℝ) (l : Line) (M N F : Point) :
  ellipse_C b focus.x focus.y →
  intersect_line_ellipse l b →
  intersect_lines (Line.mk (M.y - point_E.y) (point_E.x - M.x) (M.x * point_E.y - M.y * point_E.x)) line_x_5 F →
  parallel_lines (Line.mk (F.y - N.y) (N.x - F.x) (F.x * N.y - F.y * N.x)) (Line.mk 0 1 0) :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_parallel_line_l1104_110450


namespace NUMINAMATH_CALUDE_kates_average_speed_l1104_110464

theorem kates_average_speed (bike_speed : ℝ) (bike_time : ℝ) (walk_speed : ℝ) (walk_time : ℝ) 
  (h1 : bike_speed = 20) 
  (h2 : bike_time = 45 / 60) 
  (h3 : walk_speed = 3) 
  (h4 : walk_time = 60 / 60) : 
  (bike_speed * bike_time + walk_speed * walk_time) / (bike_time + walk_time) = 10 := by
  sorry

#check kates_average_speed

end NUMINAMATH_CALUDE_kates_average_speed_l1104_110464


namespace NUMINAMATH_CALUDE_not_all_five_digit_extendable_all_one_digit_extendable_minimal_extension_l1104_110453

/-- Represents a natural number with a specified number of digits -/
def NDigitNumber (n : ℕ) := { x : ℕ // x ≥ 10^(n-1) ∧ x < 10^n }

/-- Checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- Appends k digits to an n-digit number -/
def append_digits (x : NDigitNumber n) (y : ℕ) (k : ℕ) : ℕ :=
  x.val * 10^k + y

theorem not_all_five_digit_extendable : ∃ x : NDigitNumber 6, 
  x.val ≥ 5 * 10^5 ∧ x.val < 6 * 10^5 ∧ 
  ¬∃ y : ℕ, y < 10^6 ∧ is_perfect_square (append_digits x y 6) :=
sorry

theorem all_one_digit_extendable : ∀ x : NDigitNumber 6, 
  x.val ≥ 10^5 ∧ x.val < 2 * 10^5 → 
  ∃ y : ℕ, y < 10^6 ∧ is_perfect_square (append_digits x y 6) :=
sorry

theorem minimal_extension (n : ℕ) : 
  (∀ x : NDigitNumber n, ∃ y : ℕ, y < 10^(n+1) ∧ is_perfect_square (append_digits x y (n+1))) ∧
  (∃ x : NDigitNumber n, ∀ y : ℕ, y < 10^n → ¬is_perfect_square (append_digits x y n)) :=
sorry

end NUMINAMATH_CALUDE_not_all_five_digit_extendable_all_one_digit_extendable_minimal_extension_l1104_110453


namespace NUMINAMATH_CALUDE_problem_statement_l1104_110429

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) : 
  (-1 < x - y ∧ x - y < 1) ∧ 
  ((1 / x + x / y) ≥ 3 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 1 ∧ 1 / a + a / b = 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1104_110429


namespace NUMINAMATH_CALUDE_column_sorting_preserves_row_order_l1104_110438

/-- Represents a 10x10 table of natural numbers -/
def Table := Fin 10 → Fin 10 → ℕ

/-- Check if a row is sorted in ascending order -/
def is_row_sorted (t : Table) (row : Fin 10) : Prop :=
  ∀ i j : Fin 10, i < j → t row i ≤ t row j

/-- Check if a column is sorted in ascending order -/
def is_column_sorted (t : Table) (col : Fin 10) : Prop :=
  ∀ i j : Fin 10, i < j → t i col ≤ t j col

/-- Check if all rows are sorted in ascending order -/
def are_all_rows_sorted (t : Table) : Prop :=
  ∀ row : Fin 10, is_row_sorted t row

/-- Check if all columns are sorted in ascending order -/
def are_all_columns_sorted (t : Table) : Prop :=
  ∀ col : Fin 10, is_column_sorted t col

/-- The table contains the first 100 natural numbers -/
def contains_first_100_numbers (t : Table) : Prop :=
  ∀ n : ℕ, n ≤ 100 → ∃ i j : Fin 10, t i j = n

theorem column_sorting_preserves_row_order :
  ∀ t : Table,
  contains_first_100_numbers t →
  are_all_rows_sorted t →
  ∃ t' : Table,
    (∀ i j : Fin 10, t i j ≤ t' i j) ∧
    are_all_columns_sorted t' ∧
    are_all_rows_sorted t' :=
sorry

end NUMINAMATH_CALUDE_column_sorting_preserves_row_order_l1104_110438


namespace NUMINAMATH_CALUDE_monday_tuesday_widget_difference_l1104_110427

/-- The number of widgets David produces on Monday minus the number of widgets he produces on Tuesday -/
def widget_difference (w t : ℕ) : ℕ :=
  w * t - (w + 5) * (t - 3)

theorem monday_tuesday_widget_difference (t : ℕ) (h : t ≥ 3) :
  widget_difference (2 * t) t = t + 15 := by
  sorry

end NUMINAMATH_CALUDE_monday_tuesday_widget_difference_l1104_110427


namespace NUMINAMATH_CALUDE_calculation_proof_l1104_110478

theorem calculation_proof : (3.25 - 1.57) * 2 = 3.36 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1104_110478


namespace NUMINAMATH_CALUDE_baker_remaining_cakes_l1104_110456

/-- The number of cakes remaining after a sale -/
def cakes_remaining (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

/-- Proof that the baker has 32 cakes remaining -/
theorem baker_remaining_cakes :
  let initial_cakes : ℕ := 169
  let sold_cakes : ℕ := 137
  cakes_remaining initial_cakes sold_cakes = 32 := by
sorry

end NUMINAMATH_CALUDE_baker_remaining_cakes_l1104_110456


namespace NUMINAMATH_CALUDE_julia_tuesday_kids_l1104_110460

/-- The number of kids Julia played with on Tuesday -/
def kids_on_tuesday (total kids_monday kids_wednesday : ℕ) : ℕ :=
  total - kids_monday - kids_wednesday

theorem julia_tuesday_kids :
  kids_on_tuesday 34 17 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_julia_tuesday_kids_l1104_110460


namespace NUMINAMATH_CALUDE_product_of_two_numbers_with_sum_100_l1104_110432

theorem product_of_two_numbers_with_sum_100 (a : ℝ) : 
  let b := 100 - a
  (a + b = 100) → (a * b = a * (100 - a)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_two_numbers_with_sum_100_l1104_110432


namespace NUMINAMATH_CALUDE_parabola_c_value_l1104_110436

-- Define the parabola equation
def parabola (x b c : ℝ) : ℝ := x^2 + b*x + c

-- Theorem statement
theorem parabola_c_value :
  ∀ b c : ℝ,
  (parabola 2 b c = 12) ∧ (parabola (-2) b c = 8) →
  c = 6 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l1104_110436
