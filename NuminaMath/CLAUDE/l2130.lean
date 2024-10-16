import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2130_213049

theorem arithmetic_sequence_length (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 2 →
  aₙ = 3006 →
  d = 4 →
  aₙ = a₁ + (n - 1) * d →
  n = 752 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2130_213049


namespace NUMINAMATH_CALUDE_liquid_rise_ratio_l2130_213018

-- Define the cones and marble
def small_cone_radius : ℝ := 5
def large_cone_radius : ℝ := 10
def marble_radius : ℝ := 2

-- Define the theorem
theorem liquid_rise_ratio :
  ∀ (h₁ h₂ : ℝ),
  h₁ > 0 → h₂ > 0 →
  (1/3 * π * small_cone_radius^2 * h₁ = 1/3 * π * large_cone_radius^2 * h₂) →
  ∃ (x : ℝ),
    x > 1 ∧
    (1/3 * π * (small_cone_radius * x)^2 * (h₁ * x) = 1/3 * π * small_cone_radius^2 * h₁ + 4/3 * π * marble_radius^3) ∧
    (1/3 * π * (large_cone_radius * x)^2 * (h₂ * x) = 1/3 * π * large_cone_radius^2 * h₂ + 4/3 * π * marble_radius^3) ∧
    (h₁ * (x - 1)) / (h₂ * (x - 1)) = 4 :=
sorry

end NUMINAMATH_CALUDE_liquid_rise_ratio_l2130_213018


namespace NUMINAMATH_CALUDE_age_ratio_problem_l2130_213019

theorem age_ratio_problem (a b c : ℕ) : 
  a = b + 2 →
  b + c = 25 →
  b = 10 →
  (b : ℚ) / c = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l2130_213019


namespace NUMINAMATH_CALUDE_part1_part2_l2130_213001

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x - 12 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 3*a + 2}

-- Part 1
theorem part1 : A ∩ (Set.univ \ (B 1)) = {x | -2 ≤ x ∧ x ≤ 0 ∨ 5 ≤ x ∧ x ≤ 6} := by sorry

-- Part 2
theorem part2 : ∀ a : ℝ, A ∩ (B a) = B a ↔ a ∈ Set.Iic (-3/2) ∪ Set.Icc (-1) (4/3) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l2130_213001


namespace NUMINAMATH_CALUDE_g_of_4_l2130_213065

def g (x : ℝ) : ℝ := 5 * x + 2

theorem g_of_4 : g 4 = 22 := by sorry

end NUMINAMATH_CALUDE_g_of_4_l2130_213065


namespace NUMINAMATH_CALUDE_system_solution_l2130_213063

theorem system_solution : ∃ (x y : ℝ), (3 * x = -9 - 3 * y) ∧ (2 * x = 3 * y - 22) := by
  use -5, 2
  sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l2130_213063


namespace NUMINAMATH_CALUDE_concentric_circles_chords_l2130_213098

/-- Given two concentric circles with chords of the larger circle tangent to the smaller circle,
    if the angle between two consecutive chords is 60°, then the number of chords needed to
    complete a full rotation is 3. -/
theorem concentric_circles_chords (angle : ℝ) (n : ℕ) : 
  angle = 60 → n * angle = 360 → n = 3 := by sorry

end NUMINAMATH_CALUDE_concentric_circles_chords_l2130_213098


namespace NUMINAMATH_CALUDE_fruit_basket_ratio_l2130_213060

/-- The number of bananas in the blue basket -/
def blue_bananas : ℕ := 12

/-- The number of apples in the blue basket -/
def blue_apples : ℕ := 4

/-- The number of fruits in the red basket -/
def red_fruits : ℕ := 8

/-- The total number of fruits in the blue basket -/
def blue_total : ℕ := blue_bananas + blue_apples

/-- The ratio of fruits in the red basket to the blue basket -/
def fruit_ratio : ℚ := red_fruits / blue_total

theorem fruit_basket_ratio : fruit_ratio = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_ratio_l2130_213060


namespace NUMINAMATH_CALUDE_total_cost_of_pen_and_pencil_l2130_213077

def pencil_cost : ℝ := 2
def pen_cost : ℝ := pencil_cost + 9

theorem total_cost_of_pen_and_pencil :
  pencil_cost + pen_cost = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_pen_and_pencil_l2130_213077


namespace NUMINAMATH_CALUDE_arithmetic_increasing_iff_positive_difference_l2130_213002

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- Theorem: An arithmetic sequence is increasing if and only if its common difference is positive -/
theorem arithmetic_increasing_iff_positive_difference (a : ℕ → ℝ) :
  ArithmeticSequence a → (IncreasingSequence a ↔ ∃ d : ℝ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_increasing_iff_positive_difference_l2130_213002


namespace NUMINAMATH_CALUDE_complex_square_l2130_213009

theorem complex_square (z : ℂ) : z = 2 + 5*I → z^2 = -21 + 20*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_l2130_213009


namespace NUMINAMATH_CALUDE_sheila_cinnamon_balls_l2130_213090

/-- The number of days Sheila can place cinnamon balls -/
def days : ℕ := 10

/-- The total number of cinnamon balls Sheila bought -/
def total_balls : ℕ := 50

/-- The number of family members Sheila placed a cinnamon ball for every day -/
def family_members : ℕ := total_balls / days

theorem sheila_cinnamon_balls : family_members = 5 := by
  sorry

end NUMINAMATH_CALUDE_sheila_cinnamon_balls_l2130_213090


namespace NUMINAMATH_CALUDE_tourist_speeds_l2130_213041

theorem tourist_speeds (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  (20 / x + 2.5 = 20 / y) ∧ (20 / (x - 2) = 20 / (1.5 * y)) → x = 8 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_tourist_speeds_l2130_213041


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l2130_213020

theorem triangle_side_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b ≤ 2 * c) (h5 : b + c ≤ 3 * a) :
  2/3 < c/a ∧ c/a < 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l2130_213020


namespace NUMINAMATH_CALUDE_largest_area_error_l2130_213050

theorem largest_area_error (actual_circumference : ℝ) (error_percent : ℝ) : 
  actual_circumference = 30 →
  error_percent = 10 →
  ∃ (computed_area actual_area : ℝ),
    computed_area = π * ((actual_circumference * (1 + error_percent / 100)) / (2 * π))^2 ∧
    actual_area = π * (actual_circumference / (2 * π))^2 ∧
    (computed_area - actual_area) / actual_area * 100 ≤ 21 ∧
    ∃ (other_computed_area : ℝ),
      other_computed_area = π * ((actual_circumference * (1 - error_percent / 100)) / (2 * π))^2 ∧
      (other_computed_area - actual_area) / actual_area * 100 < 21 :=
by sorry

end NUMINAMATH_CALUDE_largest_area_error_l2130_213050


namespace NUMINAMATH_CALUDE_winter_olympics_theorem_l2130_213016

/-- Represents the scoring system for the Winter Olympics knowledge competition. -/
structure ScoringSystem where
  num_questions : ℕ
  correct_points : ℕ
  incorrect_points : ℤ

/-- Calculates the total score given the number of correct and incorrect answers. -/
def calculate_score (system : ScoringSystem) (correct : ℕ) (incorrect : ℕ) : ℤ :=
  (correct : ℤ) * system.correct_points - incorrect * system.incorrect_points

/-- Calculates the minimum number of students required for at least 3 to have the same score. -/
def min_students_for_same_score (system : ScoringSystem) : ℕ :=
  (system.num_questions * system.correct_points + 1) * 2 + 1

/-- The Winter Olympics knowledge competition theorem. -/
theorem winter_olympics_theorem (system : ScoringSystem)
  (h_num_questions : system.num_questions = 10)
  (h_correct_points : system.correct_points = 5)
  (h_incorrect_points : system.incorrect_points = 1)
  (h_xiao_ming_correct : ℕ)
  (h_xiao_ming_incorrect : ℕ)
  (h_xiao_ming_total : h_xiao_ming_correct + h_xiao_ming_incorrect = system.num_questions)
  (h_xiao_ming_correct_8 : h_xiao_ming_correct = 8)
  (h_xiao_ming_incorrect_2 : h_xiao_ming_incorrect = 2) :
  (calculate_score system h_xiao_ming_correct h_xiao_ming_incorrect = 38) ∧
  (min_students_for_same_score system = 23) := by
  sorry

end NUMINAMATH_CALUDE_winter_olympics_theorem_l2130_213016


namespace NUMINAMATH_CALUDE_penguin_colony_size_l2130_213028

theorem penguin_colony_size (P : ℕ) : 
  (P * (1.5 : ℝ) = 237) →  -- Initial fish consumption
  (6 * P + 129 = 1077)     -- Current colony size
  := by sorry

end NUMINAMATH_CALUDE_penguin_colony_size_l2130_213028


namespace NUMINAMATH_CALUDE_min_young_rank_is_11_l2130_213039

/-- Yuna's rank in the running event -/
def yuna_rank : ℕ := 6

/-- The number of people who finished between Yuna and Min-Young -/
def people_between : ℕ := 5

/-- Min-Young's rank in the running event -/
def min_young_rank : ℕ := yuna_rank + people_between

/-- Theorem stating Min-Young's rank -/
theorem min_young_rank_is_11 : min_young_rank = 11 := by
  sorry

end NUMINAMATH_CALUDE_min_young_rank_is_11_l2130_213039


namespace NUMINAMATH_CALUDE_coefficient_x3y7_expansion_l2130_213047

theorem coefficient_x3y7_expansion :
  let n : ℕ := 10
  let k : ℕ := 3
  let coeff : ℚ := (n.choose k) * (2/3)^k * (-3/5)^(n-k)
  coeff = -256/257 := by sorry

end NUMINAMATH_CALUDE_coefficient_x3y7_expansion_l2130_213047


namespace NUMINAMATH_CALUDE_kamal_math_marks_l2130_213005

/-- Calculates Kamal's marks in Mathematics given his marks in other subjects and his average -/
theorem kamal_math_marks (english : ℕ) (physics : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℕ) 
  (h1 : english = 76)
  (h2 : physics = 82)
  (h3 : chemistry = 67)
  (h4 : biology = 85)
  (h5 : average = 74) :
  let total := average * 5
  let math := total - (english + physics + chemistry + biology)
  math = 60 := by sorry

end NUMINAMATH_CALUDE_kamal_math_marks_l2130_213005


namespace NUMINAMATH_CALUDE_emily_commute_time_l2130_213062

/-- Calculates the total commute time for Emily given her travel distances and local road time --/
theorem emily_commute_time 
  (freeway_distance : ℝ) 
  (local_distance : ℝ) 
  (local_time : ℝ) 
  (h1 : freeway_distance = 100) 
  (h2 : local_distance = 25) 
  (h3 : local_time = 50) 
  (h4 : freeway_distance / local_distance = 4) : 
  local_time + freeway_distance / (2 * local_distance / local_time) = 150 := by
  sorry

#check emily_commute_time

end NUMINAMATH_CALUDE_emily_commute_time_l2130_213062


namespace NUMINAMATH_CALUDE_block_placement_probability_l2130_213094

/-- Represents a person in the block placement problem -/
inductive Person
  | Louis
  | Maria
  | Neil

/-- Represents the colors of the blocks -/
inductive Color
  | Red
  | Blue
  | Yellow
  | White
  | Green
  | Purple

/-- The number of boxes -/
def num_boxes : ℕ := 6

/-- The number of blocks each person has -/
def num_blocks_per_person : ℕ := 6

/-- A function representing a random block placement for a person -/
def block_placement := Person → Fin num_boxes → Color

/-- The probability of a specific color being chosen for a specific box by all three people -/
def prob_color_match : ℚ := 1 / 216

/-- The probability that at least one box receives exactly 3 blocks of the same color,
    placed in alphabetical order by the people's names -/
def prob_at_least_one_box_match : ℚ := 235 / 1296

theorem block_placement_probability :
  prob_at_least_one_box_match = 1 - (1 - prob_color_match) ^ num_boxes :=
sorry

end NUMINAMATH_CALUDE_block_placement_probability_l2130_213094


namespace NUMINAMATH_CALUDE_a_zero_sufficient_not_necessary_l2130_213061

def M (a : ℝ) : Set ℝ := {1, a}
def N : Set ℝ := {-1, 0, 1}

theorem a_zero_sufficient_not_necessary (a : ℝ) :
  (a = 0 → M a ⊆ N) ∧ ¬(M a ⊆ N → a = 0) :=
sorry

end NUMINAMATH_CALUDE_a_zero_sufficient_not_necessary_l2130_213061


namespace NUMINAMATH_CALUDE_num_pupils_correct_l2130_213024

/-- The number of pupils sent up for examination -/
def num_pupils : ℕ := 21

/-- The average marks of all pupils -/
def average_marks : ℚ := 39

/-- The marks of the 4 specific pupils -/
def specific_pupils_marks : List ℕ := [25, 12, 15, 19]

/-- The average marks if the 4 specific pupils were removed -/
def average_without_specific : ℚ := 44

/-- Theorem stating that the number of pupils is correct given the conditions -/
theorem num_pupils_correct :
  (average_marks * num_pupils : ℚ) =
  (average_without_specific * (num_pupils - 4) : ℚ) + (specific_pupils_marks.sum : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_num_pupils_correct_l2130_213024


namespace NUMINAMATH_CALUDE_set_relations_l2130_213056

variable {α : Type*}
variable (I A B : Set α)

theorem set_relations (h : A ∪ B = I) :
  (Aᶜ ∩ Bᶜ = ∅) ∧ (B ⊇ Aᶜ) := by
  sorry

end NUMINAMATH_CALUDE_set_relations_l2130_213056


namespace NUMINAMATH_CALUDE_xy_value_l2130_213097

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 15) : x * y = 15 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2130_213097


namespace NUMINAMATH_CALUDE_fraction_simplification_l2130_213013

theorem fraction_simplification : (144 : ℚ) / 1296 = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2130_213013


namespace NUMINAMATH_CALUDE_inequality_proof_l2130_213033

theorem inequality_proof (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : 
  x + y + z ≤ x*y*z + 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2130_213033


namespace NUMINAMATH_CALUDE_cubic_factorization_l2130_213076

theorem cubic_factorization (x : ℝ) : 2*x^3 - 4*x^2 + 2*x = 2*x*(x-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2130_213076


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l2130_213073

theorem fraction_equation_solution (x : ℚ) :
  (1 / (x + 2) + 2 / (x + 2) + x / (x + 2) + 3 / (x + 2) = 4) → x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l2130_213073


namespace NUMINAMATH_CALUDE_license_plate_difference_l2130_213078

/-- The number of letters in the alphabet -/
def numLetters : Nat := 26

/-- The number of digits available -/
def numDigits : Nat := 10

/-- The number of license plates Sunland can issue -/
def sunlandPlates : Nat := numLetters^5 * numDigits^2

/-- The number of license plates Moonland can issue -/
def moonlandPlates : Nat := numLetters^3 * numDigits^3

/-- The difference in the number of license plates between Sunland and Moonland -/
def plateDifference : Nat := sunlandPlates - moonlandPlates

theorem license_plate_difference : plateDifference = 1170561600 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l2130_213078


namespace NUMINAMATH_CALUDE_problem_solution_l2130_213007

theorem problem_solution (x y m a b : ℝ) : 
  (∃ k : ℤ, (x - 1 = k^2 * 4)) →
  ((4 * x + y)^(1/3) = 3) →
  (m^2 = y - x) →
  (5 + m = a + b) →
  (∃ n : ℤ, a = n) →
  (0 < b) →
  (b < 1) →
  (m = Real.sqrt 2 ∧ a - (Real.sqrt 2 - b)^2 = 5) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2130_213007


namespace NUMINAMATH_CALUDE_area_circle_outside_square_l2130_213027

/-- The area inside a circle but outside a square, when they share the same center -/
theorem area_circle_outside_square (r : ℝ) (s : ℝ) : 
  r = (1 : ℝ) / 2 → s = 1 → 
  (π * r^2) - s^2 + 4 * (s / 2 * s / 2 / 2 - π * r^2 / 4) = (π - 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_circle_outside_square_l2130_213027


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l2130_213075

theorem gain_percent_calculation (cost_price selling_price : ℝ) :
  selling_price = 2.5 * cost_price →
  (selling_price - cost_price) / cost_price * 100 = 150 :=
by sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l2130_213075


namespace NUMINAMATH_CALUDE_problem_statement_l2130_213029

theorem problem_statement (M N : ℝ) 
  (h1 : (4 : ℝ) / 7 = M / 77)
  (h2 : (4 : ℝ) / 7 = 98 / (N^2)) : 
  M + N = 57.1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2130_213029


namespace NUMINAMATH_CALUDE_factory_equation_correctness_l2130_213043

/-- Represents the factory worker assignment problem -/
def factory_problem (x y : ℕ) : Prop :=
  -- Total number of workers is 95
  x + y = 95 ∧
  -- Production ratio for sets (2 nuts : 1 screw)
  16 * x = 22 * y

/-- The system of linear equations correctly represents the factory problem -/
theorem factory_equation_correctness :
  ∀ x y : ℕ,
  factory_problem x y ↔ 
  (x + y = 95 ∧ 16 * x - 22 * y = 0) :=
by sorry

end NUMINAMATH_CALUDE_factory_equation_correctness_l2130_213043


namespace NUMINAMATH_CALUDE_slope_product_constant_l2130_213037

/-- The trajectory C -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1 ∧ p.2 ≠ 0}

/-- The line y = kx -/
def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1}

theorem slope_product_constant
  (M : ℝ × ℝ) (h_M : M ∈ C)
  (k : ℝ)
  (A B : ℝ × ℝ) (h_A : A ∈ C ∩ Line k) (h_B : B ∈ C ∩ Line k)
  (h_AB : A.1 = -B.1 ∧ A.2 = -B.2)
  (h_MA : M.1 ≠ A.1) (h_MB : M.1 ≠ B.1) :
  let K_MA := (M.2 - A.2) / (M.1 - A.1)
  let K_MB := (M.2 - B.2) / (M.1 - B.1)
  K_MA * K_MB = -1/4 :=
sorry

end NUMINAMATH_CALUDE_slope_product_constant_l2130_213037


namespace NUMINAMATH_CALUDE_log_sum_property_l2130_213006

theorem log_sum_property (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h : Real.log (a + b) = Real.log a + Real.log b) : 
  Real.log (a - 1) + Real.log (b - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_property_l2130_213006


namespace NUMINAMATH_CALUDE_tangent_line_at_one_monotonicity_intervals_minimum_value_on_interval_l2130_213003

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 1

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 6 * x^2 + 6 * a * x

-- Theorem for the tangent line equation when a = 0
theorem tangent_line_at_one (x y : ℝ) :
  f 0 1 = 3 ∧ f' 0 1 = 6 → (6 * x - y - 3 = 0 ↔ y - 3 = 6 * (x - 1)) :=
sorry

-- Theorem for monotonicity intervals
theorem monotonicity_intervals (a : ℝ) :
  (a = 0 → ∀ x, f' a x ≥ 0) ∧
  (a > 0 → ∀ x, (x < -a ∨ x > 0) ↔ f' a x > 0) ∧
  (a < 0 → ∀ x, (x < 0 ∨ x > -a) ↔ f' a x > 0) :=
sorry

-- Theorem for minimum value on [0, 2]
theorem minimum_value_on_interval (a : ℝ) :
  (a ≥ 0 → ∀ x ∈ Set.Icc 0 2, f a x ≥ f a 0) ∧
  (-2 < a ∧ a < 0 → ∀ x ∈ Set.Icc 0 2, f a x ≥ f a (-a)) ∧
  (a ≤ -2 → ∀ x ∈ Set.Icc 0 2, f a x ≥ f a 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_monotonicity_intervals_minimum_value_on_interval_l2130_213003


namespace NUMINAMATH_CALUDE_waiter_problem_l2130_213079

theorem waiter_problem (initial_customers : ℕ) (left_customers : ℕ) (num_tables : ℕ) 
  (h1 : initial_customers = 62)
  (h2 : left_customers = 17)
  (h3 : num_tables = 5) :
  (initial_customers - left_customers) / num_tables = 9 := by
  sorry

end NUMINAMATH_CALUDE_waiter_problem_l2130_213079


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2130_213083

theorem diophantine_equation_solution (x y z t : ℤ) : 
  x^4 - 2*y^4 - 4*z^4 - 8*t^4 = 0 → x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2130_213083


namespace NUMINAMATH_CALUDE_length_PQ_is_sqrt_82_l2130_213081

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 8*y - 5 = 0

-- Define the line y = 2x
def line_center (x y : ℝ) : Prop :=
  y = 2*x

-- Define the line m
def line_m (x y : ℝ) : Prop :=
  y = x - 1

-- Define the point M
def point_M : ℝ × ℝ := (3, 2)

-- Theorem statement
theorem length_PQ_is_sqrt_82 :
  ∀ (P Q : ℝ × ℝ),
  circle_C (-2) 1 →  -- Point A on circle C
  circle_C 5 0 →     -- Point B on circle C
  (∃ (cx cy : ℝ), circle_C cx cy ∧ line_center cx cy) →  -- Center of C on y = 2x
  line_m P.1 P.2 →   -- P is on line m
  line_m Q.1 Q.2 →   -- Q is on line m
  circle_C P.1 P.2 → -- P is on circle C
  circle_C Q.1 Q.2 → -- Q is on circle C
  line_m point_M.1 point_M.2 →  -- M is on line m
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 82 :=
by
  sorry

end NUMINAMATH_CALUDE_length_PQ_is_sqrt_82_l2130_213081


namespace NUMINAMATH_CALUDE_not_p_sufficient_for_not_q_l2130_213025

/-- Definition of proposition p -/
def p (x : ℝ) : Prop := |3*x - 4| > 2

/-- Definition of proposition q -/
def q (x : ℝ) : Prop := 1 / (x^2 - x - 2) > 0

/-- Theorem stating that not p implies not q, but not q does not necessarily imply not p -/
theorem not_p_sufficient_for_not_q :
  (∃ x : ℝ, ¬(p x) ∧ ¬(q x)) ∧ (∃ x : ℝ, ¬(q x) ∧ p x) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_for_not_q_l2130_213025


namespace NUMINAMATH_CALUDE_nine_bounces_on_12x10_table_l2130_213032

/-- Represents a rectangular pool table -/
structure PoolTable where
  width : ℕ
  height : ℕ

/-- Represents a ball's path on the pool table -/
structure BallPath where
  start_x : ℕ
  start_y : ℕ
  slope : ℚ

/-- Calculates the number of wall bounces for a ball's path on a pool table -/
def count_wall_bounces (table : PoolTable) (path : BallPath) : ℕ :=
  sorry

/-- Theorem stating that a ball hit from (0,0) along y=x on a 12x10 table bounces 9 times -/
theorem nine_bounces_on_12x10_table :
  let table : PoolTable := { width := 12, height := 10 }
  let path : BallPath := { start_x := 0, start_y := 0, slope := 1 }
  count_wall_bounces table path = 9 :=
sorry

end NUMINAMATH_CALUDE_nine_bounces_on_12x10_table_l2130_213032


namespace NUMINAMATH_CALUDE_prudence_weekend_sleep_l2130_213086

/-- Represents Prudence's sleep schedule over 4 weeks -/
structure SleepSchedule where
  weekdayNightSleep : ℕ  -- Hours of sleep on weeknights (Sun-Thu)
  weekendNapHours : ℕ    -- Hours of nap on weekend days
  totalSleepHours : ℕ    -- Total hours of sleep in 4 weeks
  weekdayNights : ℕ      -- Number of weekday nights in 4 weeks
  weekendNights : ℕ      -- Number of weekend nights in 4 weeks
  weekendDays : ℕ        -- Number of weekend days in 4 weeks

/-- Calculates the hours of sleep per night on weekends given Prudence's sleep schedule -/
def weekendNightSleep (s : SleepSchedule) : ℚ :=
  let weekdaySleep := s.weekdayNightSleep * s.weekdayNights
  let weekendNapSleep := s.weekendNapHours * s.weekendDays
  let remainingSleep := s.totalSleepHours - weekdaySleep - weekendNapSleep
  remainingSleep / s.weekendNights

/-- Theorem stating that Prudence sleeps 9 hours per night on weekends -/
theorem prudence_weekend_sleep (s : SleepSchedule)
  (h1 : s.weekdayNightSleep = 6)
  (h2 : s.weekendNapHours = 1)
  (h3 : s.totalSleepHours = 200)
  (h4 : s.weekdayNights = 20)
  (h5 : s.weekendNights = 8)
  (h6 : s.weekendDays = 8) :
  weekendNightSleep s = 9 := by
  sorry

#eval weekendNightSleep {
  weekdayNightSleep := 6,
  weekendNapHours := 1,
  totalSleepHours := 200,
  weekdayNights := 20,
  weekendNights := 8,
  weekendDays := 8
}

end NUMINAMATH_CALUDE_prudence_weekend_sleep_l2130_213086


namespace NUMINAMATH_CALUDE_four_dice_probability_l2130_213038

/-- The probability of a single standard six-sided die showing a specific number -/
def single_die_prob : ℚ := 1 / 6

/-- The number of dice being tossed simultaneously -/
def num_dice : ℕ := 4

/-- The probability of all dice showing the same specific number -/
def all_dice_prob : ℚ := (single_die_prob ^ num_dice)

/-- Theorem stating that the probability of all four standard six-sided dice 
    showing the number 3 when tossed simultaneously is 1/1296 -/
theorem four_dice_probability : all_dice_prob = 1 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_four_dice_probability_l2130_213038


namespace NUMINAMATH_CALUDE_real_part_of_complex_fraction_l2130_213095

theorem real_part_of_complex_fraction (θ : ℝ) :
  let z : ℂ := Complex.exp (θ * Complex.I)
  Complex.abs z = 1 →
  (1 / (2 - z)).re = (2 - Real.cos θ) / (5 - 4 * Real.cos θ) := by
sorry

end NUMINAMATH_CALUDE_real_part_of_complex_fraction_l2130_213095


namespace NUMINAMATH_CALUDE_dog_count_l2130_213042

theorem dog_count (dogs people : ℕ) : 
  (4 * dogs + 2 * people = 2 * (dogs + people) + 28) → dogs = 14 := by
  sorry

end NUMINAMATH_CALUDE_dog_count_l2130_213042


namespace NUMINAMATH_CALUDE_classroom_writing_instruments_l2130_213092

theorem classroom_writing_instruments :
  let total_bags : ℕ := 16
  let compartments_per_bag : ℕ := 6
  let max_instruments_per_compartment : ℕ := 8
  let empty_compartments : ℕ := 5
  let partially_filled_compartment : ℕ := 1
  let instruments_in_partially_filled : ℕ := 6
  
  let total_compartments : ℕ := total_bags * compartments_per_bag
  let filled_compartments : ℕ := total_compartments - empty_compartments - partially_filled_compartment
  
  let total_instruments : ℕ := 
    filled_compartments * max_instruments_per_compartment + 
    partially_filled_compartment * instruments_in_partially_filled
  
  total_instruments = 726 := by
  sorry

end NUMINAMATH_CALUDE_classroom_writing_instruments_l2130_213092


namespace NUMINAMATH_CALUDE_garage_motorcycles_l2130_213022

theorem garage_motorcycles (total_wheels : ℕ) (bicycles : ℕ) (cars : ℕ) 
  (bicycle_wheels : ℕ) (car_wheels : ℕ) (motorcycle_wheels : ℕ) :
  total_wheels = 90 ∧ 
  bicycles = 20 ∧ 
  cars = 10 ∧ 
  bicycle_wheels = 2 ∧ 
  car_wheels = 4 ∧ 
  motorcycle_wheels = 2 → 
  (total_wheels - (bicycles * bicycle_wheels + cars * car_wheels)) / motorcycle_wheels = 5 :=
by sorry

end NUMINAMATH_CALUDE_garage_motorcycles_l2130_213022


namespace NUMINAMATH_CALUDE_solution_properties_l2130_213054

def system_of_equations (t x y : ℝ) : Prop :=
  (4*t^2 + t + 4)*x + (5*t + 1)*y = 4*t^2 - t - 3 ∧
  (t + 2)*x + 2*y = t

theorem solution_properties :
  ∀ t : ℝ,
  (∀ x y : ℝ, system_of_equations t x y →
    (t < -1 → x < 0 ∧ y < 0) ∧
    (-1 < t ∧ t < 1 ∧ t ≠ 0 → x = (t+1)/(t-1) ∧ y = (2*t+1)/(t-1)) ∧
    (t = 1 → ∀ k : ℝ, ∃ x y : ℝ, system_of_equations t x y) ∧
    (t = 2 → ¬∃ x y : ℝ, system_of_equations t x y) ∧
    (t > 2 → x > 0 ∧ y > 0)) :=
by sorry

end NUMINAMATH_CALUDE_solution_properties_l2130_213054


namespace NUMINAMATH_CALUDE_balloon_count_l2130_213011

theorem balloon_count (green blue yellow red : ℚ) (total : ℕ) : 
  green = 2/9 →
  blue = 1/3 →
  yellow = 1/4 →
  red = 7/36 →
  green + blue + yellow + red = 1 →
  (yellow * total / 2 : ℚ) = 50 →
  total = 400 := by
sorry

end NUMINAMATH_CALUDE_balloon_count_l2130_213011


namespace NUMINAMATH_CALUDE_percentage_error_multiplication_l2130_213008

theorem percentage_error_multiplication : 
  let correct_factor : ℚ := 5 / 3
  let incorrect_factor : ℚ := 3 / 5
  let percentage_error := (correct_factor - incorrect_factor) / correct_factor * 100
  percentage_error = 64 := by
sorry

end NUMINAMATH_CALUDE_percentage_error_multiplication_l2130_213008


namespace NUMINAMATH_CALUDE_value_of_expression_l2130_213035

theorem value_of_expression (a b : ℝ) (h : 2 * a - b = -1) : 
  2021 + 4 * a - 2 * b = 2019 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2130_213035


namespace NUMINAMATH_CALUDE_geometric_arithmetic_progression_l2130_213068

theorem geometric_arithmetic_progression (a b c : ℤ) : 
  (∃ (q : ℚ), b = a * q ∧ c = b * q) →  -- Geometric progression condition
  (2 * (b + 8) = a + c) →               -- Arithmetic progression condition
  ((b + 8)^2 = a * (c + 64)) →          -- Second geometric progression condition
  (a = 4 ∧ b = 12 ∧ c = 36) :=           -- Conclusion
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_progression_l2130_213068


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_fourth_minus_n_l2130_213080

theorem largest_divisor_of_n_fourth_minus_n (n : ℤ) (h : 4 ∣ n) :
  (∃ k : ℤ, n^4 - n = 4 * k) ∧ 
  (∀ m : ℤ, m > 4 → ¬(∀ n : ℤ, 4 ∣ n → m ∣ (n^4 - n))) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_fourth_minus_n_l2130_213080


namespace NUMINAMATH_CALUDE_toucan_count_l2130_213014

/-- The number of toucans on the first limb initially -/
def initial_first_limb : ℕ := 3

/-- The number of toucans on the second limb initially -/
def initial_second_limb : ℕ := 4

/-- The number of toucans that join the first group -/
def join_first_limb : ℕ := 2

/-- The number of toucans that join the second group -/
def join_second_limb : ℕ := 3

/-- The total number of toucans after all changes -/
def total_toucans : ℕ := initial_first_limb + initial_second_limb + join_first_limb + join_second_limb

theorem toucan_count : total_toucans = 12 := by
  sorry

end NUMINAMATH_CALUDE_toucan_count_l2130_213014


namespace NUMINAMATH_CALUDE_covered_number_value_l2130_213010

theorem covered_number_value : ∃ a : ℝ, 
  (∀ x : ℝ, (x - a) / 2 = x + 3 ↔ x = -7) ∧ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_covered_number_value_l2130_213010


namespace NUMINAMATH_CALUDE_nested_square_root_equality_l2130_213059

theorem nested_square_root_equality : 
  Real.sqrt (1 + 2014 * Real.sqrt (1 + 2015 * Real.sqrt (1 + 2016 * 2018))) = 2015 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_equality_l2130_213059


namespace NUMINAMATH_CALUDE_seven_eighths_of_64_l2130_213057

theorem seven_eighths_of_64 : (7 / 8 : ℚ) * 64 = 56 := by sorry

end NUMINAMATH_CALUDE_seven_eighths_of_64_l2130_213057


namespace NUMINAMATH_CALUDE_smallest_b_in_arithmetic_geometric_progression_l2130_213048

theorem smallest_b_in_arithmetic_geometric_progression (a b c : ℤ) : 
  a < c → c < b → 
  (2 * c = a + b) →  -- arithmetic progression condition
  (b * b = a * c) →  -- geometric progression condition
  (∀ b' : ℤ, (∃ a' c' : ℤ, a' < c' ∧ c' < b' ∧ 
    (2 * c' = a' + b') ∧ 
    (b' * b' = a' * c')) → b' ≥ 2) →
  b = 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_in_arithmetic_geometric_progression_l2130_213048


namespace NUMINAMATH_CALUDE_parsley_sprig_count_l2130_213067

/-- The number of parsley sprigs Carmen started with -/
def initial_sprigs : ℕ := 25

/-- The number of plates decorated with whole sprigs -/
def whole_sprig_plates : ℕ := 8

/-- The number of plates decorated with half sprigs -/
def half_sprig_plates : ℕ := 12

/-- The number of sprigs left after decorating -/
def remaining_sprigs : ℕ := 11

theorem parsley_sprig_count : 
  initial_sprigs = whole_sprig_plates + (half_sprig_plates / 2) + remaining_sprigs :=
by sorry

end NUMINAMATH_CALUDE_parsley_sprig_count_l2130_213067


namespace NUMINAMATH_CALUDE_region_area_l2130_213089

/-- The region in the plane defined by |x + 2y| + |x - 2y| ≤ 6 -/
def Region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1 + 2*p.2| + |p.1 - 2*p.2| ≤ 6}

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

theorem region_area : area Region = 9 := by
  sorry

end NUMINAMATH_CALUDE_region_area_l2130_213089


namespace NUMINAMATH_CALUDE_soda_packing_l2130_213046

theorem soda_packing (total : ℕ) (regular : ℕ) (diet : ℕ) (pack_size : ℕ) :
  total = 200 →
  regular = 55 →
  diet = 40 →
  pack_size = 3 →
  let energy := total - regular - diet
  let complete_packs := energy / pack_size
  let leftover := energy % pack_size
  complete_packs = 35 ∧ leftover = 0 := by
  sorry

end NUMINAMATH_CALUDE_soda_packing_l2130_213046


namespace NUMINAMATH_CALUDE_monicas_savings_l2130_213012

theorem monicas_savings (weekly_savings : ℕ) (weeks_to_fill : ℕ) (repetitions : ℕ) : 
  weekly_savings = 15 → weeks_to_fill = 60 → repetitions = 5 →
  weekly_savings * weeks_to_fill * repetitions = 4500 := by
  sorry

end NUMINAMATH_CALUDE_monicas_savings_l2130_213012


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2130_213044

theorem inequality_solution_set : 
  {x : ℕ | 1 + x ≥ 2 * x - 1} = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2130_213044


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l2130_213015

/-- Given a geometric sequence with positive terms and common ratio q > 0, q ≠ 1,
    the sum of the first and fourth terms is greater than the sum of the second and third terms. -/
theorem geometric_sequence_sum_inequality {a : ℕ → ℝ} {q : ℝ} 
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h_pos : ∀ n, a n > 0)
  (h_q_pos : q > 0)
  (h_q_neq_1 : q ≠ 1) :
  a 1 + a 4 > a 2 + a 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l2130_213015


namespace NUMINAMATH_CALUDE_correct_calculation_l2130_213064

theorem correct_calculation (x : ℝ) : 
  x / 3.6 = 2.5 → (x * 3.6) / 2 = 16.2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2130_213064


namespace NUMINAMATH_CALUDE_max_pies_without_ingredients_l2130_213036

theorem max_pies_without_ingredients (total_pies : ℕ) 
  (h_total : total_pies = 36)
  (h_chocolate : total_pies / 3 ≤ total_pies)
  (h_marshmallow : total_pies / 4 ≤ total_pies)
  (h_cayenne : total_pies / 2 ≤ total_pies)
  (h_soy_nuts : total_pies / 8 ≤ total_pies) :
  total_pies - max (total_pies / 3) (max (total_pies / 4) (max (total_pies / 2) (total_pies / 8))) = 18 := by
  sorry

end NUMINAMATH_CALUDE_max_pies_without_ingredients_l2130_213036


namespace NUMINAMATH_CALUDE_equation_solution_l2130_213091

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁ = (3 + Real.sqrt 105) / 24 ∧ 
     x₂ = (3 - Real.sqrt 105) / 24) ∧ 
    (∀ x : ℝ, 4 * (3 * x)^2 + 2 * (3 * x) + 7 = 3 * (8 * x^2 + 3 * x + 3) ↔ 
      x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2130_213091


namespace NUMINAMATH_CALUDE_mirror_side_length_l2130_213021

/-- Proves that the length of each side of a square mirror is 18 inches, given the specified conditions --/
theorem mirror_side_length :
  ∀ (wall_width wall_length mirror_area : ℝ),
    wall_width = 32 →
    wall_length = 20.25 →
    mirror_area = (wall_width * wall_length) / 2 →
    ∃ (mirror_side : ℝ),
      mirror_side * mirror_side = mirror_area ∧
      mirror_side = 18 :=
by sorry

end NUMINAMATH_CALUDE_mirror_side_length_l2130_213021


namespace NUMINAMATH_CALUDE_tanners_savings_l2130_213071

/-- Tanner's savings problem -/
theorem tanners_savings (september : ℕ) (october : ℕ) (november : ℕ) (spent : ℕ) (left : ℕ) : 
  september = 17 → 
  october = 48 → 
  spent = 49 → 
  left = 41 → 
  september + october + november - spent = left → 
  november = 25 := by
sorry

end NUMINAMATH_CALUDE_tanners_savings_l2130_213071


namespace NUMINAMATH_CALUDE_square_sum_from_means_l2130_213070

theorem square_sum_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 20) 
  (h_geometric : Real.sqrt (a * b) = Real.sqrt 104) : 
  a^2 + b^2 = 1392 := by sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l2130_213070


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_l2130_213072

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + |2*x - 1|

-- Theorem for Part I
theorem solution_set_when_a_is_3 :
  {x : ℝ | f x 3 ≤ 6} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 5/2} := by sorry

-- Theorem for Part II
theorem range_of_a :
  {a : ℝ | ∀ x, f x a ≥ a^2 - a - 13} = {a : ℝ | -Real.sqrt 14 ≤ a ∧ a ≤ 1 + Real.sqrt 13} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_l2130_213072


namespace NUMINAMATH_CALUDE_work_completed_in_five_days_l2130_213074

/-- Represents the fraction of work completed by a person in one day -/
def work_rate (days : ℚ) : ℚ := 1 / days

/-- Represents the total work completed by all workers in one day -/
def total_work_rate (a b c d : ℚ) : ℚ := work_rate a + work_rate b + work_rate c + work_rate d

/-- Represents the work completed in a given number of days -/
def work_completed (rate : ℚ) (days : ℚ) : ℚ := min 1 (rate * days)

/-- Theorem: Given the work rates of A, B, C, and D, prove that after 5 days of working together, no work is left -/
theorem work_completed_in_five_days :
  let a := 10
  let b := 15
  let c := 20
  let d := 30
  let rate := total_work_rate a b c d
  work_completed rate 5 = 1 :=
by sorry

end NUMINAMATH_CALUDE_work_completed_in_five_days_l2130_213074


namespace NUMINAMATH_CALUDE_george_sock_order_l2130_213066

/-- The ratio of black to blue socks in George's original order -/
def sock_ratio : ℚ := 2 / 11

theorem george_sock_order :
  ∀ (black_price blue_price : ℝ) (blue_count : ℝ),
    black_price = 2 * blue_price →
    3 * black_price + blue_count * blue_price = 
      (blue_count * black_price + 3 * blue_price) * (1 - 0.6) →
    sock_ratio = 3 / blue_count :=
by
  sorry

end NUMINAMATH_CALUDE_george_sock_order_l2130_213066


namespace NUMINAMATH_CALUDE_positive_real_inequality_l2130_213040

theorem positive_real_inequality (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l2130_213040


namespace NUMINAMATH_CALUDE_equation_solution_l2130_213053

theorem equation_solution (x : ℝ) : 
  (1 - |Real.cos x|) / (1 + |Real.cos x|) = Real.sin x → 
  (∃ k : ℤ, x = k * Real.pi ∨ x = 2 * k * Real.pi + Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2130_213053


namespace NUMINAMATH_CALUDE_expression_evaluation_l2130_213082

theorem expression_evaluation : 
  -2^3 * (-3)^2 / (9/8) - |1/2 - 3/2| = -65 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2130_213082


namespace NUMINAMATH_CALUDE_star_op_two_neg_four_l2130_213004

-- Define the * operation for rational numbers
def star_op (x y : ℚ) : ℚ := (x * y) / (x + y)

-- Theorem statement
theorem star_op_two_neg_four : star_op 2 (-4) = 4 := by sorry

end NUMINAMATH_CALUDE_star_op_two_neg_four_l2130_213004


namespace NUMINAMATH_CALUDE_min_ping_pong_balls_l2130_213045

def is_valid_box_count (n : ℕ) : Prop :=
  n ≥ 11 ∧ n ≠ 17 ∧ n % 6 ≠ 0

def distinct_counts (counts : List ℕ) : Prop :=
  counts.Nodup

theorem min_ping_pong_balls :
  ∃ (counts : List ℕ),
    counts.length = 10 ∧
    (∀ n ∈ counts, is_valid_box_count n) ∧
    distinct_counts counts ∧
    counts.sum = 174 ∧
    (∀ (other_counts : List ℕ),
      other_counts.length = 10 →
      (∀ n ∈ other_counts, is_valid_box_count n) →
      distinct_counts other_counts →
      other_counts.sum ≥ 174) :=
by sorry

end NUMINAMATH_CALUDE_min_ping_pong_balls_l2130_213045


namespace NUMINAMATH_CALUDE_relationship_abc_l2130_213069

theorem relationship_abc : 
  let a : ℝ := Real.rpow 0.8 0.7
  let b : ℝ := Real.rpow 0.8 0.9
  let c : ℝ := Real.rpow 1.1 0.6
  c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l2130_213069


namespace NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_line_not_perp_to_intersection_not_perp_to_other_plane_l2130_213088

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_to_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (in_plane : Line → Plane → Prop)
variable (planes_perpendicular : Plane → Plane → Prop)
variable (intersection_line : Plane → Plane → Line)

-- Theorem 1: Two lines perpendicular to the same plane are parallel
theorem lines_perp_to_plane_are_parallel
  (p : Plane) (l1 l2 : Line)
  (h1 : perpendicular_to_plane l1 p)
  (h2 : perpendicular_to_plane l2 p) :
  parallel l1 l2 :=
sorry

-- Theorem 2: In perpendicular planes, a line not perpendicular to the intersection
-- is not perpendicular to the other plane
theorem line_not_perp_to_intersection_not_perp_to_other_plane
  (p1 p2 : Plane) (l : Line)
  (h1 : planes_perpendicular p1 p2)
  (h2 : in_plane l p1)
  (h3 : ¬ perpendicular l (intersection_line p1 p2)) :
  ¬ perpendicular_to_plane l p2 :=
sorry

end NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_line_not_perp_to_intersection_not_perp_to_other_plane_l2130_213088


namespace NUMINAMATH_CALUDE_last_digit_square_periodicity_and_symmetry_l2130_213051

theorem last_digit_square_periodicity_and_symmetry :
  ∀ (n : ℕ), 
    (n^2 % 10 = ((n + 10)^2) % 10) ∧
    (∀ (k : ℕ), k ≤ 4 → (k^2 % 10 = ((10 - k)^2) % 10)) :=
by sorry

end NUMINAMATH_CALUDE_last_digit_square_periodicity_and_symmetry_l2130_213051


namespace NUMINAMATH_CALUDE_male_attendees_fraction_l2130_213000

theorem male_attendees_fraction (M F : ℚ) : 
  M + F = 1 →
  (3/4 : ℚ) * M + (5/6 : ℚ) * F = 7/9 →
  M = 2/3 := by
sorry

end NUMINAMATH_CALUDE_male_attendees_fraction_l2130_213000


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2130_213084

def M : Set ℕ := {1, 2}

def N : Set ℕ := {b | ∃ a ∈ M, b = 2 * a - 1}

theorem union_of_M_and_N : M ∪ N = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2130_213084


namespace NUMINAMATH_CALUDE_polynomial_product_l2130_213096

variables (a b : ℚ)

theorem polynomial_product (a b : ℚ) :
  (-3 * a^2 * b) * (-2 * a * b + b - 3) = 6 * a^3 * b^2 - 3 * a^2 * b^2 + 9 * a^2 * b :=
by sorry

end NUMINAMATH_CALUDE_polynomial_product_l2130_213096


namespace NUMINAMATH_CALUDE_quadratic_trinomial_condition_l2130_213099

theorem quadratic_trinomial_condition (m : ℤ) : 
  (|m| = 2 ∧ m ≠ 2) ↔ m = -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_condition_l2130_213099


namespace NUMINAMATH_CALUDE_fifth_day_distance_l2130_213026

/-- Represents the daily walking distance of a man -/
def walkingSequence (firstDay : ℕ) (dailyIncrease : ℕ) : ℕ → ℕ :=
  fun n => firstDay + (n - 1) * dailyIncrease

theorem fifth_day_distance
  (firstDay : ℕ)
  (dailyIncrease : ℕ)
  (h1 : firstDay = 100)
  (h2 : (Finset.range 9).sum (walkingSequence firstDay dailyIncrease) = 1260) :
  walkingSequence firstDay dailyIncrease 5 = 140 := by
  sorry

#check fifth_day_distance

end NUMINAMATH_CALUDE_fifth_day_distance_l2130_213026


namespace NUMINAMATH_CALUDE_parallel_line_with_chord_sum_exists_l2130_213058

-- Define a circle in a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line in a plane
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

-- Theorem statement
theorem parallel_line_with_chord_sum_exists 
  (S₁ S₂ : Circle) (l : Line) (a : ℝ) (h : a > 0) :
  ∃ (l' : Line),
    (∀ (p : ℝ × ℝ), l.point.1 + p.1 * l.direction.1 = l'.point.1 + p.1 * l'.direction.1 ∧
                     l.point.2 + p.2 * l.direction.2 = l'.point.2 + p.2 * l'.direction.2) ∧
    ∃ (A B C D : ℝ × ℝ),
      (A.1 - B.1)^2 + (A.2 - B.2)^2 = ((A.1 - S₁.center.1)^2 + (A.2 - S₁.center.2)^2 - S₁.radius^2) ∧
      (C.1 - D.1)^2 + (C.2 - D.2)^2 = ((C.1 - S₂.center.1)^2 + (C.2 - S₂.center.2)^2 - S₂.radius^2) ∧
      (A.1 - B.1)^2 + (A.2 - B.2)^2 + (C.1 - D.1)^2 + (C.2 - D.2)^2 = a^2 :=
by
  sorry


end NUMINAMATH_CALUDE_parallel_line_with_chord_sum_exists_l2130_213058


namespace NUMINAMATH_CALUDE_farm_animal_leg_difference_l2130_213052

/-- Represents the number of legs for a cow -/
def cow_legs : ℕ := 4

/-- Represents the number of legs for a chicken -/
def chicken_legs : ℕ := 2

/-- Represents the number of cows in the group -/
def num_cows : ℕ := 6

theorem farm_animal_leg_difference 
  (num_chickens : ℕ) 
  (total_legs : ℕ) 
  (h1 : total_legs = cow_legs * num_cows + chicken_legs * num_chickens)
  (h2 : total_legs > 2 * (num_cows + num_chickens)) :
  total_legs - 2 * (num_cows + num_chickens) = 12 := by
  sorry

end NUMINAMATH_CALUDE_farm_animal_leg_difference_l2130_213052


namespace NUMINAMATH_CALUDE_necessary_condition_range_l2130_213055

theorem necessary_condition_range (a : ℝ) : 
  (∀ x : ℝ, x < a + 2 → x ≤ 2) → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_range_l2130_213055


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_l2130_213017

/-- If x^2 - 10x + m is a perfect square trinomial, then m = 25 -/
theorem perfect_square_trinomial_m (m : ℝ) : 
  (∃ a b : ℝ, ∀ x, x^2 - 10*x + m = (a*x + b)^2) → m = 25 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_l2130_213017


namespace NUMINAMATH_CALUDE_max_intersection_lines_l2130_213034

/-- A plane in 3D space -/
structure Plane

/-- Represents the intersection of two planes -/
def intersect (p1 p2 : Plane) : Prop := sorry

/-- The number of intersection lines between two intersecting planes -/
def numIntersectionLines (p1 p2 : Plane) (h : intersect p1 p2) : ℕ := 1

/-- The theorem stating the maximum number of intersection lines for three intersecting planes -/
theorem max_intersection_lines (p1 p2 p3 : Plane) 
  (h12 : intersect p1 p2) (h23 : intersect p2 p3) (h13 : intersect p1 p3) :
  (numIntersectionLines p1 p2 h12 + 
   numIntersectionLines p2 p3 h23 + 
   numIntersectionLines p1 p3 h13) ≤ 3 := by sorry

end NUMINAMATH_CALUDE_max_intersection_lines_l2130_213034


namespace NUMINAMATH_CALUDE_fathers_age_fathers_current_age_l2130_213085

theorem fathers_age (sons_age_next_year : ℕ) (father_age_ratio : ℕ) : ℕ :=
  let sons_current_age := sons_age_next_year - 1
  father_age_ratio * sons_current_age

theorem fathers_current_age :
  fathers_age 8 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_fathers_current_age_l2130_213085


namespace NUMINAMATH_CALUDE_student_attendance_probability_l2130_213023

theorem student_attendance_probability :
  let p_absent : ℝ := 1 / 20
  let p_present : ℝ := 1 - p_absent
  let p_one_absent_one_present : ℝ := p_absent * p_present + p_present * p_absent
  p_one_absent_one_present = 0.095 := by
  sorry

end NUMINAMATH_CALUDE_student_attendance_probability_l2130_213023


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l2130_213030

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_first_term
  (a : ℕ → ℚ)
  (h_geometric : GeometricSequence a)
  (h_third_term : a 3 = 36)
  (h_fourth_term : a 4 = 54) :
  a 1 = 16 := by
  sorry

#check geometric_sequence_first_term

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l2130_213030


namespace NUMINAMATH_CALUDE_quadratic_solution_l2130_213031

/-- A quadratic function passing through specific points -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating the solutions of the quadratic equation -/
theorem quadratic_solution (a b c : ℝ) :
  (f a b c (-1) = 8) →
  (f a b c 0 = 3) →
  (f a b c 1 = 0) →
  (f a b c 2 = -1) →
  (f a b c 3 = 0) →
  (∀ x : ℝ, f a b c x = 0 ↔ x = 1 ∨ x = 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_solution_l2130_213031


namespace NUMINAMATH_CALUDE_symmetry_xoy_plane_l2130_213087

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the xOy plane -/
def symmetricXOY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

theorem symmetry_xoy_plane :
  let A : Point3D := { x := 1, y := 2, z := 3 }
  let B : Point3D := symmetricXOY A
  B = { x := 1, y := 2, z := -3 } := by
  sorry

end NUMINAMATH_CALUDE_symmetry_xoy_plane_l2130_213087


namespace NUMINAMATH_CALUDE_stating_same_white_wins_exist_l2130_213093

/-- Represents a chess tournament with participants and their scores. -/
structure ChessTournament where
  /-- The number of participants in the tournament. -/
  participants : Nat
  /-- The number of games won with white pieces by each participant. -/
  white_wins : Fin participants → Nat
  /-- Assumption that all participants have the same total score. -/
  same_total_score : ∀ i j : Fin participants, 
    white_wins i + (participants - 1 - white_wins j) = participants - 1

/-- 
Theorem stating that in a chess tournament where all participants have the same total score,
there must be at least two participants who won the same number of games with white pieces.
-/
theorem same_white_wins_exist (t : ChessTournament) : 
  ∃ i j : Fin t.participants, i ≠ j ∧ t.white_wins i = t.white_wins j := by
  sorry


end NUMINAMATH_CALUDE_stating_same_white_wins_exist_l2130_213093
