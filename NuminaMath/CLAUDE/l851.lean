import Mathlib

namespace NUMINAMATH_CALUDE_davids_biology_marks_l851_85165

def english_marks : ℕ := 86
def math_marks : ℕ := 89
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 87
def average_marks : ℕ := 85
def num_subjects : ℕ := 5

theorem davids_biology_marks :
  let known_subjects_total := english_marks + math_marks + physics_marks + chemistry_marks
  let all_subjects_total := average_marks * num_subjects
  all_subjects_total - known_subjects_total = 81 := by sorry

end NUMINAMATH_CALUDE_davids_biology_marks_l851_85165


namespace NUMINAMATH_CALUDE_max_power_under_500_l851_85180

theorem max_power_under_500 (a b : ℕ) (ha : a > 0) (hb : b > 1) (hab : a^b < 500) :
  (∀ (c d : ℕ), c > 0 → d > 1 → c^d < 500 → a^b ≥ c^d) →
  a = 22 ∧ b = 2 ∧ a + b = 24 :=
sorry

end NUMINAMATH_CALUDE_max_power_under_500_l851_85180


namespace NUMINAMATH_CALUDE_train_crossing_time_l851_85134

/-- Given a train and a platform with specific dimensions and time to pass,
    calculate the time it takes for the train to cross a stationary point. -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (time_to_pass_platform : ℝ) 
  (h1 : train_length = 600)
  (h2 : platform_length = 450)
  (h3 : time_to_pass_platform = 105) :
  (train_length / ((train_length + platform_length) / time_to_pass_platform)) = 60 :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l851_85134


namespace NUMINAMATH_CALUDE_matrix_inverse_l851_85104

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 5; -2, 9]

theorem matrix_inverse :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![9/46, -5/46; 2/46, 4/46]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_matrix_inverse_l851_85104


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l851_85167

theorem isosceles_triangle_perimeter : ∀ x y : ℝ,
  x^2 - 9*x + 18 = 0 →
  y^2 - 9*y + 18 = 0 →
  x ≠ y →
  (x + 2*y = 15 ∨ y + 2*x = 15) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l851_85167


namespace NUMINAMATH_CALUDE_sum_of_min_max_z_l851_85164

-- Define the feasible region
def FeasibleRegion (x y : ℝ) : Prop :=
  2 * x - y + 2 ≥ 0 ∧ 2 * x + y - 2 ≥ 0 ∧ y ≥ 0

-- Define the function z
def z (x y : ℝ) : ℝ := x - y

-- Theorem statement
theorem sum_of_min_max_z :
  ∃ (min_z max_z : ℝ),
    (∀ (x y : ℝ), FeasibleRegion x y → z x y ≥ min_z) ∧
    (∃ (x y : ℝ), FeasibleRegion x y ∧ z x y = min_z) ∧
    (∀ (x y : ℝ), FeasibleRegion x y → z x y ≤ max_z) ∧
    (∃ (x y : ℝ), FeasibleRegion x y ∧ z x y = max_z) ∧
    min_z + max_z = -1 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_min_max_z_l851_85164


namespace NUMINAMATH_CALUDE_kekai_garage_sale_earnings_l851_85128

/-- Calculates the amount of money Kekai has left after a garage sale --/
def kekais_money (num_shirts : ℕ) (num_pants : ℕ) (shirt_price : ℕ) (pants_price : ℕ) (share_fraction : ℚ) : ℚ :=
  let total_earned := num_shirts * shirt_price + num_pants * pants_price
  (total_earned : ℚ) * (1 - share_fraction)

/-- Proves that Kekai has $10 left after the garage sale --/
theorem kekai_garage_sale_earnings : kekais_money 5 5 1 3 (1/2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_kekai_garage_sale_earnings_l851_85128


namespace NUMINAMATH_CALUDE_prob_at_least_half_girls_l851_85120

/-- The number of children in the family -/
def num_children : ℕ := 5

/-- The probability of having a girl for each child -/
def prob_girl : ℚ := 1/2

/-- The number of possible combinations of boys and girls -/
def total_combinations : ℕ := 2^num_children

/-- The number of combinations with at least half girls -/
def favorable_combinations : ℕ := (num_children.choose 3) + (num_children.choose 4) + (num_children.choose 5)

/-- The probability of having at least half girls in a family of five children -/
theorem prob_at_least_half_girls : 
  (favorable_combinations : ℚ) / total_combinations = 1/2 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_half_girls_l851_85120


namespace NUMINAMATH_CALUDE_no_solution_for_sqrt_equation_l851_85193

theorem no_solution_for_sqrt_equation :
  ¬∃ x : ℝ, Real.sqrt (3*x - 2) + Real.sqrt (2*x - 2) + Real.sqrt (x - 1) = 3 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_sqrt_equation_l851_85193


namespace NUMINAMATH_CALUDE_unique_sequence_l851_85139

theorem unique_sequence (n : ℕ) (h : n > 1) :
  ∃! (x : ℕ → ℕ), 
    (∀ k, k ∈ Finset.range (n - 1) → x k > 0) ∧ 
    (∀ i j, i < j ∧ j < n - 1 → x i < x j) ∧
    (∀ i, i ∈ Finset.range (n - 1) → x i + x (n - 1 - i) = 2 * n) ∧
    (∀ i j, i ∈ Finset.range (n - 1) ∧ j ∈ Finset.range (n - 1) ∧ x i + x j < 2 * n → 
      ∃ k, k ∈ Finset.range (n - 1) ∧ x i + x j = x k) ∧
    (∀ k, k ∈ Finset.range (n - 1) → x k = 2 * (k + 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_sequence_l851_85139


namespace NUMINAMATH_CALUDE_point_movement_on_number_line_l851_85160

theorem point_movement_on_number_line (A : ℝ) : 
  A + 7 - 4 = 0 → A = -3 := by
  sorry

end NUMINAMATH_CALUDE_point_movement_on_number_line_l851_85160


namespace NUMINAMATH_CALUDE_root_implies_inequality_l851_85140

theorem root_implies_inequality (a b : ℝ) 
  (h : ∃ x, (x + a) * (x + b) = 9 ∧ x = a + b) : a * b ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_inequality_l851_85140


namespace NUMINAMATH_CALUDE_cos_2x_value_l851_85156

theorem cos_2x_value (x : ℝ) (h : Real.sin (π / 4 + x / 2) = 3 / 5) : 
  Real.cos (2 * x) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cos_2x_value_l851_85156


namespace NUMINAMATH_CALUDE_number_wall_solve_l851_85177

/-- Represents a row in the Number Wall -/
structure NumberWallRow :=
  (left : ℤ) (middle_left : ℤ) (middle_right : ℤ) (right : ℤ)

/-- Defines the Number Wall structure and rules -/
def NumberWall (bottom : NumberWallRow) : Prop :=
  ∃ (second : NumberWallRow) (third : NumberWallRow) (top : ℤ),
    second.left = bottom.left + bottom.middle_left
    ∧ second.middle_left = bottom.middle_left + bottom.middle_right
    ∧ second.middle_right = bottom.middle_right + bottom.right
    ∧ third.left = second.left + second.middle_left
    ∧ third.right = second.middle_right + second.right
    ∧ top = third.left + third.right
    ∧ top = 36

/-- The main theorem to prove -/
theorem number_wall_solve :
  ∀ m : ℤ, NumberWall ⟨m, 6, 12, 10⟩ → m = -28 :=
by sorry

end NUMINAMATH_CALUDE_number_wall_solve_l851_85177


namespace NUMINAMATH_CALUDE_root_minus_one_implies_k_eq_neg_two_l851_85154

theorem root_minus_one_implies_k_eq_neg_two (k : ℝ) :
  ((-1 : ℝ)^2 - k*(-1) + 1 = 0) → k = -2 :=
by sorry

end NUMINAMATH_CALUDE_root_minus_one_implies_k_eq_neg_two_l851_85154


namespace NUMINAMATH_CALUDE_right_triangle_angle_measure_l851_85105

/-- In a right triangle ABC where angle C is 90° and tan A is √3, angle A measures 60°. -/
theorem right_triangle_angle_measure (A B C : Real) (h1 : A + B + C = 180) 
  (h2 : C = 90) (h3 : Real.tan A = Real.sqrt 3) : A = 60 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_measure_l851_85105


namespace NUMINAMATH_CALUDE_ratio_problem_l851_85166

theorem ratio_problem (a b c d : ℚ) 
  (h1 : a / b = 5)
  (h2 : c / b = 3)
  (h3 : c / d = 2) :
  d / a = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l851_85166


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l851_85143

/-- Represents a pyramid with an equilateral triangular base and isosceles lateral faces -/
structure Pyramid where
  base_side_length : ℝ
  lateral_side_length : ℝ
  (base_is_equilateral : base_side_length = 2)
  (lateral_is_isosceles : lateral_side_length = 3)

/-- Represents a cube inscribed in the pyramid -/
structure InscribedCube (p : Pyramid) where
  side_length : ℝ
  (base_on_pyramid_base : True)
  (top_vertices_touch_midpoints : True)

/-- The volume of the inscribed cube -/
def cube_volume (p : Pyramid) (c : InscribedCube p) : ℝ :=
  c.side_length ^ 3

theorem inscribed_cube_volume (p : Pyramid) (c : InscribedCube p) :
  cube_volume p c = (4 * Real.sqrt 2 - 3) ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l851_85143


namespace NUMINAMATH_CALUDE_abacus_problem_l851_85184

def is_valid_abacus_division (upper lower : ℕ) : Prop :=
  upper ≥ 100 ∧ upper < 1000 ∧ lower ≥ 100 ∧ lower < 1000 ∧
  upper + lower = 1110 ∧
  (∃ k : ℕ, upper = k * lower) ∧
  (∃ a b c : ℕ, upper = 100 * a + 10 * b + c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c)

theorem abacus_problem : ∃ upper lower : ℕ, is_valid_abacus_division upper lower ∧ upper = 925 := by
  sorry

end NUMINAMATH_CALUDE_abacus_problem_l851_85184


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l851_85112

/-- Represents a trapezoid ABCD with sides AB and CD -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ

/-- The theorem statement -/
theorem trapezoid_segment_length (t : Trapezoid) :
  (t.AB / t.CD = 3) →  -- Area ratio implies base ratio
  (t.AB + t.CD = 320) →
  t.AB = 240 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l851_85112


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l851_85107

/-- A point in a 2D Cartesian coordinate system. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines symmetry with respect to the x-axis. -/
def symmetricXAxis (p q : Point) : Prop :=
  q.x = p.x ∧ q.y = -p.y

/-- The theorem stating that if Q is symmetric to P(-3, 2) with respect to the x-axis,
    then Q has coordinates (-3, -2). -/
theorem symmetric_point_coordinates :
  let p : Point := ⟨-3, 2⟩
  let q : Point := ⟨-3, -2⟩
  symmetricXAxis p q → q = ⟨-3, -2⟩ := by
  sorry


end NUMINAMATH_CALUDE_symmetric_point_coordinates_l851_85107


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_sqrt_12n_l851_85175

theorem smallest_n_for_integer_sqrt_12n :
  ∀ n : ℕ+, (∃ k : ℕ, k^2 = 12*n) → (∀ m : ℕ+, m < n → ¬∃ j : ℕ, j^2 = 12*m) → n = 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_sqrt_12n_l851_85175


namespace NUMINAMATH_CALUDE_arithmetic_sequence_probability_l851_85108

-- Define the set of numbers
def S : Set Nat := Finset.range 20

-- Define a function to check if three numbers form an arithmetic sequence
def isArithmeticSequence (a b c : Nat) : Prop := a + c = 2 * b

-- Define the total number of ways to choose 3 numbers from 20
def totalCombinations : Nat := Nat.choose 20 3

-- Define the number of valid arithmetic sequences
def validSequences : Nat := 90

-- State the theorem
theorem arithmetic_sequence_probability :
  (validSequences : ℚ) / totalCombinations = 1 / 38 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_probability_l851_85108


namespace NUMINAMATH_CALUDE_absolute_value_sum_zero_implies_product_l851_85135

theorem absolute_value_sum_zero_implies_product (x y : ℝ) :
  |x - 1| + |y + 3| = 0 → (x + 1) * (y - 3) = -12 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_zero_implies_product_l851_85135


namespace NUMINAMATH_CALUDE_six_years_passed_l851_85163

/-- Represents a stem-and-leaf plot --/
structure StemAndLeafPlot where
  stem : List Nat
  leaves : List (List Nat)

/-- The initial stem-and-leaf plot --/
def initial_plot : StemAndLeafPlot := {
  stem := [0, 1, 2, 3, 4, 5],
  leaves := [[3], [0, 1, 2, 3, 4, 5], [2, 3, 5, 6, 8, 9], [4, 6], [0, 2], []]
}

/-- The final stem-and-leaf plot with obscured numbers --/
def final_plot : StemAndLeafPlot := {
  stem := [0, 1, 2, 3, 4, 5],
  leaves := [[], [6, 9], [4, 7], [0], [2, 8], []]
}

/-- Function to calculate the years passed --/
def years_passed (initial : StemAndLeafPlot) (final : StemAndLeafPlot) : Nat :=
  sorry

/-- Theorem stating that 6 years have passed --/
theorem six_years_passed :
  years_passed initial_plot final_plot = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_years_passed_l851_85163


namespace NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l851_85151

theorem sum_of_squares_and_square_of_sum : (3 + 5)^2 + (3^2 + 5^2) = 98 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l851_85151


namespace NUMINAMATH_CALUDE_parallel_iff_m_eq_neg_two_l851_85192

-- Define the lines as functions of x and y
def line1 (m : ℝ) (x y : ℝ) : Prop := 2*x + m*y - 2*m + 4 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := m*x + 2*y - m + 2 = 0

-- Define what it means for two lines to be parallel
def parallel (m : ℝ) : Prop := ∀ (x y : ℝ), line1 m x y ↔ ∃ (k : ℝ), line2 m (x + k) (y + k)

-- State the theorem
theorem parallel_iff_m_eq_neg_two :
  ∀ m : ℝ, parallel m ↔ m = -2 := by sorry

end NUMINAMATH_CALUDE_parallel_iff_m_eq_neg_two_l851_85192


namespace NUMINAMATH_CALUDE_fish_sharing_l851_85182

/-- Represents the number of fish caught by each cat -/
structure CatCatch where
  white : ℕ
  black : ℕ
  calico : ℕ

/-- Represents the money transactions for each cat -/
structure CatMoney where
  white : ℚ
  black : ℚ
  calico : ℚ

def totalFish (c : CatCatch) : ℕ := c.white + c.black + c.calico

def averageShare (c : CatCatch) : ℚ := (totalFish c : ℚ) / 3

theorem fish_sharing (c : CatCatch) (m : CatMoney) : 
  c.white = 5 → c.black = 3 → c.calico = 0 → m.calico = -4/5 →
  (averageShare c = 8/3) ∧ 
  (m.white = 7) ∧ 
  (m.black = 1) ∧
  (m.white + m.black + m.calico = 0) ∧
  ((totalFish c : ℚ) * (3 : ℚ) / (totalFish c : ℚ) = 3) := by
  sorry

#check fish_sharing

end NUMINAMATH_CALUDE_fish_sharing_l851_85182


namespace NUMINAMATH_CALUDE_bus_driver_hours_l851_85114

theorem bus_driver_hours (regular_rate overtime_rate_factor total_compensation : ℚ) : 
  regular_rate = 14 →
  overtime_rate_factor = 1.75 →
  total_compensation = 982 →
  ∃ (regular_hours overtime_hours : ℕ),
    regular_hours = 40 ∧
    overtime_hours = 17 ∧
    regular_hours + overtime_hours = 57 ∧
    regular_rate * regular_hours + (regular_rate * overtime_rate_factor) * overtime_hours = total_compensation :=
by sorry

end NUMINAMATH_CALUDE_bus_driver_hours_l851_85114


namespace NUMINAMATH_CALUDE_concert_ticket_cost_l851_85141

/-- The price of a child ticket -/
def child_ticket_price : ℝ := sorry

/-- The price of an adult ticket -/
def adult_ticket_price : ℝ := 2 * child_ticket_price

/-- The condition that 6 adult tickets and 5 child tickets cost $37.50 -/
axiom ticket_condition : 6 * adult_ticket_price + 5 * child_ticket_price = 37.50

/-- The theorem to prove -/
theorem concert_ticket_cost : 
  10 * adult_ticket_price + 8 * child_ticket_price = 61.78 := by sorry

end NUMINAMATH_CALUDE_concert_ticket_cost_l851_85141


namespace NUMINAMATH_CALUDE_city_households_l851_85136

/-- The number of deer that entered the city -/
def num_deer : ℕ := 100

/-- The number of households in the city -/
def num_households : ℕ := 75

theorem city_households : 
  (num_households < num_deer) ∧ 
  (4 * num_households = 3 * num_deer) := by
  sorry

end NUMINAMATH_CALUDE_city_households_l851_85136


namespace NUMINAMATH_CALUDE_store_uniforms_l851_85129

theorem store_uniforms (total_uniforms : ℕ) (additional_uniform : ℕ) : 
  total_uniforms = 927 → 
  additional_uniform = 1 → 
  ∃ (employees : ℕ), 
    employees > 1 ∧ 
    (total_uniforms + additional_uniform) % employees = 0 ∧ 
    total_uniforms % employees ≠ 0 ∧
    ∀ (n : ℕ), n > employees → (total_uniforms + additional_uniform) % n ≠ 0 ∨ total_uniforms % n = 0 →
    employees = 29 := by
sorry

end NUMINAMATH_CALUDE_store_uniforms_l851_85129


namespace NUMINAMATH_CALUDE_max_distance_for_given_tires_l851_85171

/-- Represents the maximum distance a car can travel by switching tires -/
def max_distance (front_tire_life rear_tire_life : ℕ) : ℕ :=
  let swap_point := front_tire_life / 2
  swap_point + min (rear_tire_life - swap_point) (front_tire_life - swap_point)

/-- Theorem stating the maximum distance a car can travel with given tire lifespans -/
theorem max_distance_for_given_tires :
  max_distance 21000 28000 = 24000 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_for_given_tires_l851_85171


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l851_85172

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + k^2 - 3 > 0) ↔ (k > 2 ∨ k < -2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l851_85172


namespace NUMINAMATH_CALUDE_range_of_m_l851_85102

-- Define a decreasing function on (-∞, 0)
def DecreasingOnNegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y < 0 → f x > f y

-- Define the theorem
theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h1 : DecreasingOnNegative f) 
  (h2 : f (1 - m) < f (m - 3)) : 
  1 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l851_85102


namespace NUMINAMATH_CALUDE_handball_final_score_l851_85130

/-- Represents the score of a handball match -/
structure Score where
  home : ℕ
  visitors : ℕ

/-- Calculates the final score given the initial score and goals scored in the second half -/
def finalScore (initial : Score) (visitorGoals : ℕ) : Score :=
  { home := initial.home + 2 * visitorGoals,
    visitors := initial.visitors + visitorGoals }

/-- Theorem stating the final score of the handball match -/
theorem handball_final_score :
  ∀ (initial : Score) (visitorGoals : ℕ),
    initial.home = 9 →
    initial.visitors = 14 →
    let final := finalScore initial visitorGoals
    (final.home = final.visitors + 1) →
    final.home = 21 ∧ final.visitors = 20 := by
  sorry

#check handball_final_score

end NUMINAMATH_CALUDE_handball_final_score_l851_85130


namespace NUMINAMATH_CALUDE_proposition_q_undetermined_l851_85194

theorem proposition_q_undetermined (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬p) : 
  (q ∨ ¬q) ∧ ¬(q ∧ ¬q) := by
sorry

end NUMINAMATH_CALUDE_proposition_q_undetermined_l851_85194


namespace NUMINAMATH_CALUDE_robot_return_distance_l851_85181

/-- A robot's walk pattern -/
structure RobotWalk where
  step_distance : ℝ
  turn_angle : ℝ

/-- The total angle turned by the robot -/
def total_angle (w : RobotWalk) (n : ℕ) : ℝ := n * w.turn_angle

/-- The distance walked by the robot -/
def total_distance (w : RobotWalk) (n : ℕ) : ℝ := n * w.step_distance

/-- Theorem: A robot walking 1m and turning left 45° each time will return to its starting point after 8 steps -/
theorem robot_return_distance (w : RobotWalk) (h1 : w.step_distance = 1) (h2 : w.turn_angle = 45) :
  ∃ n : ℕ, total_angle w n = 360 ∧ total_distance w n = 8 := by
  sorry

end NUMINAMATH_CALUDE_robot_return_distance_l851_85181


namespace NUMINAMATH_CALUDE_vector_b_coordinates_l851_85145

def vector_a : ℝ × ℝ := (3, -4)

def opposite_direction (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k < 0 ∧ w = (k * v.1, k * v.2)

theorem vector_b_coordinates :
  ∀ (b : ℝ × ℝ),
    opposite_direction vector_a b →
    Real.sqrt (b.1^2 + b.2^2) = 10 →
    b = (-6, 8) := by sorry

end NUMINAMATH_CALUDE_vector_b_coordinates_l851_85145


namespace NUMINAMATH_CALUDE_log3_derivative_l851_85149

theorem log3_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 3) x = 1 / (x * Real.log 3) := by
sorry

end NUMINAMATH_CALUDE_log3_derivative_l851_85149


namespace NUMINAMATH_CALUDE_storks_on_fence_l851_85185

/-- The number of storks on a fence, given the initial number of birds,
    the number of birds that join, and the final difference between birds and storks. -/
def number_of_storks (initial_birds : ℕ) (joining_birds : ℕ) (final_difference : ℕ) : ℕ :=
  initial_birds + joining_birds - final_difference

/-- Theorem stating that the number of storks is 4 under the given conditions. -/
theorem storks_on_fence : number_of_storks 3 2 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_storks_on_fence_l851_85185


namespace NUMINAMATH_CALUDE_square_vector_problem_l851_85133

theorem square_vector_problem (a b c : ℝ × ℝ) : 
  (∀ x : ℝ × ℝ, ‖x‖ = 1 → ‖x + x‖ = ‖a‖) →  -- side length is 1
  ‖a‖ = 1 →                                -- |a| = 1 (side length)
  ‖c‖ = Real.sqrt 2 →                      -- |c| = √2 (diagonal)
  a + b = c →                              -- vector addition
  ‖b - a - c‖ = 2 := by sorry

end NUMINAMATH_CALUDE_square_vector_problem_l851_85133


namespace NUMINAMATH_CALUDE_greatest_root_of_g_l851_85186

def g (x : ℝ) : ℝ := 21 * x^4 - 20 * x^2 + 3

theorem greatest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt (3/7) ∧
  g r = 0 ∧
  ∀ (x : ℝ), g x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_greatest_root_of_g_l851_85186


namespace NUMINAMATH_CALUDE_even_digits_base7_512_l851_85103

/-- Converts a natural number to its base-7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers -/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Theorem: The number of even digits in the base-7 representation of 512₁₀ is 0 -/
theorem even_digits_base7_512 : countEvenDigits (toBase7 512) = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_digits_base7_512_l851_85103


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l851_85195

theorem purely_imaginary_complex_number (m : ℝ) : 
  (m^2 + 3*m - 4 = 0) ∧ (m + 4 ≠ 0) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l851_85195


namespace NUMINAMATH_CALUDE_employee_pay_l851_85115

theorem employee_pay (total_pay : ℚ) (a_pay : ℚ) (b_pay : ℚ) :
  total_pay = 570 →
  a_pay = 1.5 * b_pay →
  total_pay = a_pay + b_pay →
  b_pay = 228 := by
sorry

end NUMINAMATH_CALUDE_employee_pay_l851_85115


namespace NUMINAMATH_CALUDE_area_diagonal_constant_for_specific_rectangle_l851_85123

/-- Represents a rectangle with given ratio and perimeter -/
structure Rectangle where
  ratio : Rat
  perimeter : ℝ

/-- The constant k for which the area of the rectangle equals k * d^2, where d is the diagonal length -/
def area_diagonal_constant (rect : Rectangle) : ℝ :=
  sorry

theorem area_diagonal_constant_for_specific_rectangle :
  let rect : Rectangle := { ratio := 5/2, perimeter := 28 }
  area_diagonal_constant rect = 10/29 := by
  sorry

end NUMINAMATH_CALUDE_area_diagonal_constant_for_specific_rectangle_l851_85123


namespace NUMINAMATH_CALUDE_calligraphy_students_l851_85150

theorem calligraphy_students (x : ℕ) : 
  (50 : ℕ) = (2 * x - 1) + x + (51 - 3 * x) :=
by sorry

end NUMINAMATH_CALUDE_calligraphy_students_l851_85150


namespace NUMINAMATH_CALUDE_deposit_percentage_l851_85116

theorem deposit_percentage (deposit : ℝ) (remaining : ℝ) : 
  deposit = 130 → remaining = 1170 → (deposit / (deposit + remaining)) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_deposit_percentage_l851_85116


namespace NUMINAMATH_CALUDE_cube_division_impossibility_l851_85137

/-- Represents a rectangular parallelepiped with dimensions (n, n+1, n+2) --/
structure Parallelepiped where
  n : ℕ

/-- The volume of a parallelepiped --/
def volume (p : Parallelepiped) : ℕ := p.n * (p.n + 1) * (p.n + 2)

/-- Theorem: It's impossible to divide a cube of volume 8000 into parallelepipeds
    with consecutive natural number dimensions --/
theorem cube_division_impossibility :
  ¬ ∃ (parallelepipeds : List Parallelepiped),
    (parallelepipeds.map volume).sum = 8000 :=
sorry

end NUMINAMATH_CALUDE_cube_division_impossibility_l851_85137


namespace NUMINAMATH_CALUDE_other_integer_is_30_l851_85117

theorem other_integer_is_30 (a b : ℤ) (h1 : 3 * a + 2 * b = 135) (h2 : a = 25 ∨ b = 25) : 
  (a ≠ 25 → b = 30) ∧ (b ≠ 25 → a = 30) := by
  sorry

end NUMINAMATH_CALUDE_other_integer_is_30_l851_85117


namespace NUMINAMATH_CALUDE_book_page_digits_l851_85169

/-- The total number of digits used to number pages in a book -/
def totalDigits (n : ℕ) : ℕ :=
  (min n 9) + 
  (2 * (min n 99 - 9)) + 
  (3 * (n - min n 99))

/-- Theorem: The total number of digits used in numbering the pages of a book with 346 pages is 930 -/
theorem book_page_digits : totalDigits 346 = 930 := by
  sorry

end NUMINAMATH_CALUDE_book_page_digits_l851_85169


namespace NUMINAMATH_CALUDE_product_of_solutions_l851_85196

theorem product_of_solutions (x : ℝ) : 
  (∃ α β : ℝ, (α * β = -10) ∧ (10 = -α^2 - 4*α) ∧ (10 = -β^2 - 4*β)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_solutions_l851_85196


namespace NUMINAMATH_CALUDE_question_always_truthful_l851_85144

-- Define the types of residents
inductive ResidentType
| Knight
| Liar

-- Define the possible answers
inductive Answer
| Yes
| No

-- Define a function to represent the truth about having a crocodile
def hasCrocodile : ResidentType → Bool → Answer
| ResidentType.Knight, true => Answer.Yes
| ResidentType.Knight, false => Answer.No
| ResidentType.Liar, true => Answer.No
| ResidentType.Liar, false => Answer.Yes

-- Define the function that represents the response to the question
def responseToQuestion (resident : ResidentType) (hasCroc : Bool) : Answer :=
  hasCrocodile resident hasCroc

-- Theorem: The response to the question always gives the truthful answer
theorem question_always_truthful (resident : ResidentType) (hasCroc : Bool) :
  responseToQuestion resident hasCroc = hasCrocodile ResidentType.Knight hasCroc :=
by sorry

end NUMINAMATH_CALUDE_question_always_truthful_l851_85144


namespace NUMINAMATH_CALUDE_inequality_proof_l851_85106

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 1) :
  (1 / x^2 + x) * (1 / y^2 + y) * (1 / z^2 + z) ≥ (28/3)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l851_85106


namespace NUMINAMATH_CALUDE_sum_of_numbers_l851_85110

def total_numbers (joyce xavier coraline jayden mickey yvonne : ℕ) : Prop :=
  xavier = 4 * joyce ∧
  coraline = xavier + 50 ∧
  jayden = coraline - 40 ∧
  mickey = jayden + 20 ∧
  yvonne = xavier + joyce ∧
  joyce = 30 ∧
  joyce + xavier + coraline + jayden + mickey + yvonne = 750

theorem sum_of_numbers :
  ∃ (joyce xavier coraline jayden mickey yvonne : ℕ),
    total_numbers joyce xavier coraline jayden mickey yvonne :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l851_85110


namespace NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l851_85176

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + x

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ x + 2} = {x : ℝ | x ≥ 3 ∨ x ≤ -1} :=
sorry

-- Part 2
theorem solution_set_part2 (a : ℝ) (h : a > 0) :
  ({x : ℝ | f a x ≤ 3*x} = {x : ℝ | x ≥ 2}) → a = 6 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l851_85176


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l851_85191

/-- Given a geometric sequence {aₙ} where a₁a₃ = a₄ = 4, prove that a₆ = 8 -/
theorem geometric_sequence_sixth_term (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- a is a geometric sequence
  a 1 * a 3 = 4 →  -- given condition
  a 4 = 4 →        -- given condition
  a 6 = 8 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l851_85191


namespace NUMINAMATH_CALUDE_positive_real_inequality_l851_85121

theorem positive_real_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * (a^2 + b*c)) / (b + c) + (b * (b^2 + c*a)) / (c + a) + (c * (c^2 + a*b)) / (a + b) ≥ a*b + b*c + c*a := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l851_85121


namespace NUMINAMATH_CALUDE_cubic_expression_value_l851_85158

theorem cubic_expression_value (p q : ℝ) : 
  3 * p^2 - 5 * p - 12 = 0 →
  3 * q^2 - 5 * q - 12 = 0 →
  p ≠ q →
  (9 * p^3 - 9 * q^3) / (p - q) = 61 := by
sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l851_85158


namespace NUMINAMATH_CALUDE_solutions_to_quartic_equation_l851_85122

theorem solutions_to_quartic_equation :
  let S : Set ℂ := {x : ℂ | x^4 - 81 = 0}
  S = {3, -3, 3*I, -3*I} := by
  sorry

end NUMINAMATH_CALUDE_solutions_to_quartic_equation_l851_85122


namespace NUMINAMATH_CALUDE_dehydrated_men_fraction_l851_85170

theorem dehydrated_men_fraction (total_men : ℕ) (finished_men : ℕ) 
  (h1 : total_men = 80)
  (h2 : finished_men = 52)
  (h3 : (1 : ℚ) / 4 * total_men = total_men - (3 : ℚ) / 4 * total_men)
  (h4 : (2 : ℚ) / 3 * ((3 : ℚ) / 4 * total_men) = total_men - finished_men - ((1 : ℚ) / 4 * total_men)) :
  (total_men - finished_men - (1 : ℚ) / 4 * total_men) / ((2 : ℚ) / 3 * ((3 : ℚ) / 4 * total_men)) = (1 : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_dehydrated_men_fraction_l851_85170


namespace NUMINAMATH_CALUDE_digit_sum_properties_l851_85146

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate to check if two natural numbers have the same digits in a different order -/
def same_digits (m k : ℕ) : Prop := sorry

theorem digit_sum_properties (M K : ℕ) (h : same_digits M K) :
  (sum_of_digits (2 * M) = sum_of_digits (2 * K)) ∧
  (M % 2 = 0 → K % 2 = 0 → sum_of_digits (M / 2) = sum_of_digits (K / 2)) ∧
  (sum_of_digits (5 * M) = sum_of_digits (5 * K)) := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_properties_l851_85146


namespace NUMINAMATH_CALUDE_min_value_fraction_l851_85127

theorem min_value_fraction (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  (∀ y, 0 < y ∧ y < 1 → (1 / (4 * x) + 4 / (1 - x)) ≤ (1 / (4 * y) + 4 / (1 - y))) →
  1 / (4 * x) + 4 / (1 - x) = 25 / 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l851_85127


namespace NUMINAMATH_CALUDE_monkey_reaches_top_l851_85148

/-- A monkey climbing a tree -/
def monkey_climb (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) : ℕ → ℕ
| 0 => 0
| (n + 1) => min tree_height (monkey_climb tree_height hop_distance slip_distance n + hop_distance - slip_distance)

theorem monkey_reaches_top (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) 
  (h1 : tree_height = 50)
  (h2 : hop_distance = 4)
  (h3 : slip_distance = 3)
  (h4 : hop_distance > slip_distance) :
  ∃ t : ℕ, monkey_climb tree_height hop_distance slip_distance t = tree_height ∧ t = 50 := by
  sorry

end NUMINAMATH_CALUDE_monkey_reaches_top_l851_85148


namespace NUMINAMATH_CALUDE_problem_solution_l851_85178

theorem problem_solution (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - 3*k) * (x + 3*k) = x^3 + 3*k*(x^2 - x - 7)) →
  k = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l851_85178


namespace NUMINAMATH_CALUDE_triangle_angle_solution_l851_85198

/-- Given a triangle with angles measuring 60°, (5x)°, and (3x)°, prove that x = 15 -/
theorem triangle_angle_solution (x : ℝ) : 
  (60 : ℝ) + 5*x + 3*x = 180 → x = 15 := by
  sorry

#check triangle_angle_solution

end NUMINAMATH_CALUDE_triangle_angle_solution_l851_85198


namespace NUMINAMATH_CALUDE_brownie_triangles_l851_85131

theorem brownie_triangles (pan_length : ℝ) (pan_width : ℝ) 
                          (triangle_base : ℝ) (triangle_height : ℝ) :
  pan_length = 15 →
  pan_width = 24 →
  triangle_base = 3 →
  triangle_height = 4 →
  (pan_length * pan_width) / ((1/2) * triangle_base * triangle_height) = 60 := by
  sorry

end NUMINAMATH_CALUDE_brownie_triangles_l851_85131


namespace NUMINAMATH_CALUDE_smallest_n_divisors_not_multiple_of_ten_l851_85138

def is_perfect_cube (m : ℕ) : Prop := ∃ k : ℕ, m = k^3

def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = k^2

def is_perfect_seventh (m : ℕ) : Prop := ∃ k : ℕ, m = k^7

def count_non_ten_divisors (n : ℕ) : ℕ := 
  (Finset.filter (fun d => ¬(10 ∣ d)) (Nat.divisors n)).card

theorem smallest_n_divisors_not_multiple_of_ten :
  ∃ n : ℕ, 
    (∀ m < n, ¬(is_perfect_cube (m / 2) ∧ is_perfect_square (m / 3) ∧ is_perfect_seventh (m / 5))) ∧
    is_perfect_cube (n / 2) ∧
    is_perfect_square (n / 3) ∧
    is_perfect_seventh (n / 5) ∧
    count_non_ten_divisors n = 52 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisors_not_multiple_of_ten_l851_85138


namespace NUMINAMATH_CALUDE_simplify_expression_l851_85197

/-- Proves that the simplified expression is equal to the original expression for all real x. -/
theorem simplify_expression (x : ℝ) : 3*x + 9*x^2 + 16 - (5 - 3*x - 9*x^2 + x^3) = -x^3 + 18*x^2 + 6*x + 11 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l851_85197


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l851_85179

theorem largest_digit_divisible_by_six :
  ∃ (N : ℕ), N ≤ 9 ∧ (5217 * 10 + N) % 6 = 0 ∧
  ∀ (M : ℕ), M ≤ 9 → (5217 * 10 + M) % 6 = 0 → M ≤ N :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l851_85179


namespace NUMINAMATH_CALUDE_board_cut_ratio_l851_85168

/-- Given a board of length 69 inches cut into two pieces, where the shorter piece is 23 inches long,
    the ratio of the longer piece to the shorter piece is 2:1. -/
theorem board_cut_ratio : 
  ∀ (short_piece long_piece : ℝ),
  short_piece = 23 →
  short_piece + long_piece = 69 →
  long_piece / short_piece = 2 := by
sorry

end NUMINAMATH_CALUDE_board_cut_ratio_l851_85168


namespace NUMINAMATH_CALUDE_currency_denominations_l851_85125

/-- The number of different denominations that can be formed with a given number of coins/bills of three types -/
def total_denominations (fifty_cent : ℕ) (five_yuan : ℕ) (hundred_yuan : ℕ) : ℕ :=
  let single_denom := fifty_cent + five_yuan + hundred_yuan
  let double_denom := fifty_cent * five_yuan + five_yuan * hundred_yuan + hundred_yuan * fifty_cent
  let triple_denom := fifty_cent * five_yuan * hundred_yuan
  single_denom + double_denom + triple_denom

/-- Theorem stating that the total number of denominations with 3 fifty-cent coins, 
    6 five-yuan bills, and 4 one-hundred-yuan bills is 139 -/
theorem currency_denominations : 
  total_denominations 3 6 4 = 139 := by
  sorry

end NUMINAMATH_CALUDE_currency_denominations_l851_85125


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l851_85100

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 + a 4 + a 6 = 3 →
  a 1 + a 3 + a 5 + a 7 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l851_85100


namespace NUMINAMATH_CALUDE_dog_ate_cost_l851_85113

-- Define the given conditions
def total_slices : ℕ := 6
def total_cost : ℚ := 9
def mother_slices : ℕ := 2

-- Define the theorem
theorem dog_ate_cost : 
  (total_cost / total_slices) * (total_slices - mother_slices) = 6 := by
  sorry

end NUMINAMATH_CALUDE_dog_ate_cost_l851_85113


namespace NUMINAMATH_CALUDE_stockholm_to_malmo_via_gothenburg_l851_85132

/-- Represents a distance on a map --/
structure MapDistance :=
  (cm : ℝ)

/-- Represents a real-world distance --/
structure RealDistance :=
  (km : ℝ)

/-- Represents a map scale --/
structure MapScale :=
  (km_per_cm : ℝ)

/-- Converts a map distance to a real distance given a scale --/
def convert_distance (md : MapDistance) (scale : MapScale) : RealDistance :=
  ⟨md.cm * scale.km_per_cm⟩

/-- Adds two real distances --/
def add_distances (d1 d2 : RealDistance) : RealDistance :=
  ⟨d1.km + d2.km⟩

theorem stockholm_to_malmo_via_gothenburg 
  (stockholm_gothenburg : MapDistance)
  (gothenburg_malmo : MapDistance)
  (scale : MapScale)
  (h1 : stockholm_gothenburg.cm = 120)
  (h2 : gothenburg_malmo.cm = 150)
  (h3 : scale.km_per_cm = 20) :
  (add_distances 
    (convert_distance stockholm_gothenburg scale)
    (convert_distance gothenburg_malmo scale)).km = 5400 :=
by
  sorry

end NUMINAMATH_CALUDE_stockholm_to_malmo_via_gothenburg_l851_85132


namespace NUMINAMATH_CALUDE_range_of_a_l851_85101

open Set Real

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Ioo 2 3, x^2 + 5 > a*x) = false → 
  a ∈ Ici (2 * sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l851_85101


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l851_85118

theorem fractional_equation_solution :
  ∃ (x : ℝ), (2 / (x - 2) - (2 * x) / (2 - x) = 1) ∧ (x - 2 ≠ 0) ∧ (2 - x ≠ 0) ∧ (x = -4) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l851_85118


namespace NUMINAMATH_CALUDE_three_numbers_average_l851_85142

theorem three_numbers_average (a b c : ℝ) 
  (h1 : a + (b + c) / 2 = 65)
  (h2 : b + (a + c) / 2 = 69)
  (h3 : c + (a + b) / 2 = 76)
  : (a + b + c) / 3 = 35 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_average_l851_85142


namespace NUMINAMATH_CALUDE_debby_candy_eaten_l851_85126

/-- Given that Debby initially had 12 pieces of candy and ended up with 3 pieces,
    prove that she ate 9 pieces. -/
theorem debby_candy_eaten (initial : ℕ) (final : ℕ) (eaten : ℕ) 
    (h1 : initial = 12) 
    (h2 : final = 3) 
    (h3 : initial = final + eaten) : eaten = 9 := by
  sorry

end NUMINAMATH_CALUDE_debby_candy_eaten_l851_85126


namespace NUMINAMATH_CALUDE_max_gcd_sum_1998_l851_85124

theorem max_gcd_sum_1998 : ∃ (a b c : ℕ+), 
  (a + b + c : ℕ) = 1998 ∧ 
  Nat.gcd (Nat.gcd a.val b.val) c.val = 74 ∧ 
  0 < a.val ∧ a.val < b.val ∧ b.val ≤ c.val ∧ c.val < 2 * a.val := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_sum_1998_l851_85124


namespace NUMINAMATH_CALUDE_log_equation_solution_l851_85183

theorem log_equation_solution :
  ∃! x : ℝ, Real.log (3 * x + 4) = 1 :=
by
  use 2
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l851_85183


namespace NUMINAMATH_CALUDE_division_problem_l851_85155

theorem division_problem (a b q : ℕ) 
  (h1 : a - b = 1200)
  (h2 : a = 1495)
  (h3 : a = b * q + 4) :
  q = 5 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l851_85155


namespace NUMINAMATH_CALUDE_total_time_calculation_l851_85119

-- Define the time spent sharpening the knife
def sharpening_time : ℕ := 10

-- Define the multiplier for peeling time
def peeling_multiplier : ℕ := 3

-- Theorem to prove
theorem total_time_calculation :
  sharpening_time + peeling_multiplier * sharpening_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_time_calculation_l851_85119


namespace NUMINAMATH_CALUDE_negation_of_existence_l851_85161

theorem negation_of_existence (n : ℝ) :
  (¬ ∃ a : ℝ, a ≥ -1 ∧ Real.log (Real.exp n + 1) > 1/2) ↔
  (∀ a : ℝ, a ≥ -1 → Real.log (Real.exp n + 1) ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l851_85161


namespace NUMINAMATH_CALUDE_infinitely_many_non_representable_l851_85111

theorem infinitely_many_non_representable : 
  ∃ f : ℕ → ℤ, Function.Injective f ∧ 
    ∀ (k : ℕ) (a b c : ℕ), f k ≠ 2^a + 3^b - 5^c := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_non_representable_l851_85111


namespace NUMINAMATH_CALUDE_smallest_positive_integer_l851_85173

theorem smallest_positive_integer : ∀ n : ℕ, n > 0 → n ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_l851_85173


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l851_85174

theorem consecutive_page_numbers_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20412 → n + (n + 1) = 287 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l851_85174


namespace NUMINAMATH_CALUDE_cos_theta_plus_pi_fourth_l851_85152

theorem cos_theta_plus_pi_fourth (θ : ℝ) (h : Real.sin (θ - π/4) = 1/5) : 
  Real.cos (θ + π/4) = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_theta_plus_pi_fourth_l851_85152


namespace NUMINAMATH_CALUDE_zero_is_global_minimum_l851_85162

-- Define the function f(x) = (x - 1)e^(x - 1)
noncomputable def f (x : ℝ) : ℝ := (x - 1) * Real.exp (x - 1)

-- Theorem statement
theorem zero_is_global_minimum :
  ∀ x : ℝ, f 0 ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_zero_is_global_minimum_l851_85162


namespace NUMINAMATH_CALUDE_line_not_parallel_in_plane_l851_85157

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contained_in : Line → Plane → Prop)
variable (not_parallel : Line → Plane → Prop)
variable (coplanar : Line → Line → Plane → Prop)
variable (not_parallel_lines : Line → Line → Prop)

-- State the theorem
theorem line_not_parallel_in_plane 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) 
  (h2 : α ≠ β) 
  (h3 : contained_in m α) 
  (h4 : not_parallel n α) 
  (h5 : coplanar m n β) : 
  not_parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_line_not_parallel_in_plane_l851_85157


namespace NUMINAMATH_CALUDE_shark_sightings_multiple_l851_85109

/-- The number of shark sightings in Daytona Beach -/
def daytona_sightings : ℕ := 26

/-- The number of shark sightings in Cape May -/
def cape_may_sightings : ℕ := 7

/-- The additional number of sightings in Daytona Beach beyond the multiple -/
def additional_sightings : ℕ := 5

/-- The theorem stating the multiple of shark sightings in Cape May compared to Daytona Beach -/
theorem shark_sightings_multiple :
  ∃ (x : ℚ), x * cape_may_sightings + additional_sightings = daytona_sightings ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_shark_sightings_multiple_l851_85109


namespace NUMINAMATH_CALUDE_second_strongest_in_final_probability_l851_85190

/-- Represents a player in the tournament -/
structure Player where
  strength : ℕ

/-- Represents a tournament with 8 players -/
structure Tournament where
  players : Fin 8 → Player
  strength_ordered : ∀ i j, i < j → (players i).strength > (players j).strength

/-- The probability that the second strongest player reaches the final -/
def probability_second_strongest_in_final (t : Tournament) : ℚ :=
  4 / 7

/-- Theorem stating that the probability of the second strongest player
    reaching the final is 4/7 -/
theorem second_strongest_in_final_probability (t : Tournament) :
  probability_second_strongest_in_final t = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_second_strongest_in_final_probability_l851_85190


namespace NUMINAMATH_CALUDE_f_f_two_equals_two_l851_85147

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else 2/x

theorem f_f_two_equals_two : f (f 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_f_two_equals_two_l851_85147


namespace NUMINAMATH_CALUDE_hotel_light_bulbs_l851_85189

theorem hotel_light_bulbs 
  (I F : ℕ) -- I: number of incandescent bulbs, F: number of fluorescent bulbs
  (h_positive : I > 0 ∧ F > 0) -- ensure positive numbers of bulbs
  (h_incandescent_on : (3 : ℝ) / 10 * I = (1 : ℝ) / 7 * (7 : ℝ) / 10 * (I + F)) -- 30% of incandescent on, which is 1/7 of all on bulbs
  (h_total_on : (7 : ℝ) / 10 * (I + F) = (3 : ℝ) / 10 * I + x * F) -- 70% of all bulbs are on
  (x : ℝ) -- x is the fraction of fluorescent bulbs that are on
  : x = (9 : ℝ) / 10 := by
sorry

end NUMINAMATH_CALUDE_hotel_light_bulbs_l851_85189


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l851_85187

/-- The number of ways to distribute n indistinguishable items into k distinguishable categories -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of flavors that can be created by combining 5 scoops from 3 basic flavors -/
theorem ice_cream_flavors : distribute 5 3 = 21 := by sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l851_85187


namespace NUMINAMATH_CALUDE_not_prime_for_all_positive_n_l851_85188

def f (n : ℕ+) : ℤ := (n : ℤ)^3 - 9*(n : ℤ)^2 + 23*(n : ℤ) - 17

theorem not_prime_for_all_positive_n : ∀ n : ℕ+, ¬(Nat.Prime (Int.natAbs (f n))) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_for_all_positive_n_l851_85188


namespace NUMINAMATH_CALUDE_square_sum_equality_l851_85153

theorem square_sum_equality (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l851_85153


namespace NUMINAMATH_CALUDE_runners_speed_ratio_l851_85159

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- Represents the circular track -/
structure Track where
  circumference : ℝ

/-- Represents the state of the runners on the track -/
structure RunnerState where
  track : Track
  runner1 : Runner
  runner2 : Runner
  meetingPoints : Finset ℝ  -- Set of points where runners meet

/-- The theorem statement -/
theorem runners_speed_ratio 
  (state : RunnerState) 
  (h1 : state.runner1.direction ≠ state.runner2.direction)  -- Runners move in opposite directions
  (h2 : state.runner1.speed ≠ 0 ∧ state.runner2.speed ≠ 0)  -- Both runners have non-zero speed
  (h3 : state.meetingPoints.card = 3)  -- There are exactly three meeting points
  (h4 : ∀ p ∈ state.meetingPoints, p < state.track.circumference)  -- Meeting points are on the track
  : state.runner2.speed / state.runner1.speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_runners_speed_ratio_l851_85159


namespace NUMINAMATH_CALUDE_smallest_k_for_monochromatic_rectangle_l851_85199

/-- A chessboard coloring is a function that assigns a color to each square of the board. -/
def Coloring (n k : ℕ) := Fin (2 * n) → Fin k → Fin n

/-- Predicate that checks if there exist 2 columns and 2 rows with 4 squares of the same color at their intersections. -/
def HasMonochromaticRectangle (n k : ℕ) (c : Coloring n k) : Prop :=
  ∃ (i j : Fin (2 * n)) (x y : Fin k),
    i ≠ j ∧ x ≠ y ∧ 
    c i x = c i y ∧ c j x = c j y ∧ c i x = c j x

/-- The main theorem stating the smallest k that guarantees a monochromatic rectangle for any n-coloring. -/
theorem smallest_k_for_monochromatic_rectangle (n : ℕ+) :
  ∃ (k : ℕ), k = 2 * n^2 - n + 1 ∧
  (∀ (m : ℕ), m ≥ k → ∀ (c : Coloring n m), HasMonochromaticRectangle n m c) ∧
  (∀ (m : ℕ), m < k → ∃ (c : Coloring n m), ¬HasMonochromaticRectangle n m c) :=
sorry

#check smallest_k_for_monochromatic_rectangle

end NUMINAMATH_CALUDE_smallest_k_for_monochromatic_rectangle_l851_85199
