import Mathlib

namespace NUMINAMATH_CALUDE_a_5_value_l1486_148687

/-- A geometric sequence with positive terms satisfying a_n * a_(n+1) = 2^(2n+1) -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n) ∧
  (∀ n, a n * a (n + 1) = 2^(2*n + 1))

theorem a_5_value (a : ℕ → ℝ) (h : geometric_sequence a) : a 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_a_5_value_l1486_148687


namespace NUMINAMATH_CALUDE_smallest_n_below_threshold_l1486_148675

/-- The number of boxes in the warehouse -/
def num_boxes : ℕ := 2023

/-- The probability of drawing a green marble on the nth draw -/
def Q (n : ℕ) : ℚ := 1 / (n * (2 * n + 1))

/-- The threshold probability -/
def threshold : ℚ := 1 / num_boxes

/-- 32 is the smallest positive integer n such that Q(n) < 1/2023 -/
theorem smallest_n_below_threshold : 
  (∀ k < 32, Q k ≥ threshold) ∧ Q 32 < threshold :=
sorry

end NUMINAMATH_CALUDE_smallest_n_below_threshold_l1486_148675


namespace NUMINAMATH_CALUDE_two_digit_number_puzzle_l1486_148693

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def reverse_digits (n : ℕ) : ℕ := (n % 10) * 10 + (n / 10)

theorem two_digit_number_puzzle (n : ℕ) :
  is_two_digit n ∧ 
  (digit_sum n) % 3 = 0 ∧ 
  n - 27 = reverse_digits n → 
  n = 63 ∨ n = 96 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_puzzle_l1486_148693


namespace NUMINAMATH_CALUDE_boy_walking_time_l1486_148642

/-- Given a boy who walks at 6/7 of his usual rate and reaches school 4 minutes early, 
    his usual time to reach the school is 24 minutes. -/
theorem boy_walking_time (usual_rate : ℝ) (usual_time : ℝ) 
  (h1 : usual_rate > 0) (h2 : usual_time > 0) : 
  (6 / 7 * usual_rate) * (usual_time - 4) = usual_rate * usual_time → 
  usual_time = 24 := by
  sorry

end NUMINAMATH_CALUDE_boy_walking_time_l1486_148642


namespace NUMINAMATH_CALUDE_descent_time_l1486_148673

/-- Prove that the time to descend a hill is 2 hours given the specified conditions -/
theorem descent_time (climb_time : ℝ) (climb_speed : ℝ) (total_avg_speed : ℝ) :
  climb_time = 4 →
  climb_speed = 2.625 →
  total_avg_speed = 3.5 →
  ∃ (descent_time : ℝ),
    descent_time = 2 ∧
    (2 * climb_time * climb_speed) = (total_avg_speed * (climb_time + descent_time)) :=
by sorry

end NUMINAMATH_CALUDE_descent_time_l1486_148673


namespace NUMINAMATH_CALUDE_odometer_problem_l1486_148645

theorem odometer_problem (a b c : ℕ) (ha : a ≥ 1) (hsum : a + b + c = 9)
  (hx : ∃ x : ℕ, x > 0 ∧ 60 * x = 100 * c + 10 * a + b - (100 * a + 10 * b + c)) :
  a^2 + b^2 + c^2 = 51 := by
sorry

end NUMINAMATH_CALUDE_odometer_problem_l1486_148645


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1486_148600

theorem algebraic_expression_value (a : ℝ) (h : a^2 + a - 1 = 0) : 
  2*a^2 + 2*a + 2021 = 2023 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1486_148600


namespace NUMINAMATH_CALUDE_daniels_age_l1486_148696

theorem daniels_age (uncle_bob_age : ℕ) (elizabeth_age : ℕ) (daniel_age : ℕ) :
  uncle_bob_age = 60 →
  elizabeth_age = (2 * uncle_bob_age) / 3 →
  daniel_age = elizabeth_age - 10 →
  daniel_age = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_daniels_age_l1486_148696


namespace NUMINAMATH_CALUDE_tens_digit_of_2031_pow_2024_minus_2033_l1486_148619

theorem tens_digit_of_2031_pow_2024_minus_2033 :
  ∃ n : ℕ, n < 10 ∧ (2031^2024 - 2033) % 100 = 80 + n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_2031_pow_2024_minus_2033_l1486_148619


namespace NUMINAMATH_CALUDE_penny_difference_l1486_148655

theorem penny_difference (kate_pennies john_pennies : ℕ) 
  (h1 : kate_pennies = 223) 
  (h2 : john_pennies = 388) : 
  john_pennies - kate_pennies = 165 := by
sorry

end NUMINAMATH_CALUDE_penny_difference_l1486_148655


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l1486_148666

theorem sqrt_expression_equality (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - x^3) / (3 * x^3))^2) = (x^3 - 1 + Real.sqrt (x^6 - 2*x^3 + 10)) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l1486_148666


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l1486_148643

/-- Given that the solution set of ax^2 - bx + c > 0 is (-1, 2), prove the following statements -/
theorem quadratic_inequality_properties (a b c : ℝ) 
  (h : Set.Ioo (-1 : ℝ) 2 = {x : ℝ | a * x^2 - b * x + c > 0}) :
  (b < 0 ∧ c > 0) ∧ 
  (a - b + c > 0) ∧ 
  ({x : ℝ | a * x^2 + b * x + c > 0} = Set.Ioo (-2 : ℝ) 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l1486_148643


namespace NUMINAMATH_CALUDE_square_area_12m_l1486_148613

theorem square_area_12m (s : ℝ) (h : s = 12) : s^2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_square_area_12m_l1486_148613


namespace NUMINAMATH_CALUDE_partition_positive_integers_l1486_148660

def is_arithmetic_sequence (x y z : ℕ) : Prop :=
  y - x = z - y ∧ x < y ∧ y < z

def has_infinite_arithmetic_subsequence (S : Set ℕ) : Prop :=
  ∃ (a d : ℕ), d ≠ 0 ∧ ∀ n : ℕ, (a + n * d) ∈ S

theorem partition_positive_integers :
  ∃ (A B : Set ℕ),
    (A ∪ B = {n : ℕ | n > 0}) ∧
    (A ∩ B = ∅) ∧
    (∀ x y z : ℕ, x ∈ A → y ∈ A → z ∈ A → x ≠ y → y ≠ z → x ≠ z →
      ¬is_arithmetic_sequence x y z) ∧
    ¬has_infinite_arithmetic_subsequence B :=
by sorry

end NUMINAMATH_CALUDE_partition_positive_integers_l1486_148660


namespace NUMINAMATH_CALUDE_parking_spaces_on_first_level_l1486_148641

/-- Represents a 4-level parking garage -/
structure ParkingGarage where
  level1 : ℕ
  level2 : ℕ
  level3 : ℕ
  level4 : ℕ

/-- The conditions of the parking garage problem -/
def validParkingGarage (g : ParkingGarage) : Prop :=
  g.level2 = g.level1 + 8 ∧
  g.level3 = g.level2 + 12 ∧
  g.level4 = g.level3 - 9 ∧
  g.level1 + g.level2 + g.level3 + g.level4 = 299 - 100

theorem parking_spaces_on_first_level (g : ParkingGarage) 
  (h : validParkingGarage g) : g.level1 = 40 := by
  sorry

end NUMINAMATH_CALUDE_parking_spaces_on_first_level_l1486_148641


namespace NUMINAMATH_CALUDE_circle_area_equality_l1486_148638

theorem circle_area_equality (r₁ r₂ r₃ : ℝ) : 
  r₁ = 17 → r₂ = 27 → r₃ = 10 * Real.sqrt 11 → 
  π * r₃^2 = π * (r₂^2 - r₁^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equality_l1486_148638


namespace NUMINAMATH_CALUDE_slope_of_line_l1486_148674

theorem slope_of_line (x y : ℝ) : y = x - 1 → (y - (x - 1)) / (x - x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l1486_148674


namespace NUMINAMATH_CALUDE_coin_array_problem_l1486_148635

/-- The number of coins in a triangular array with n rows -/
def triangle_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem coin_array_problem :
  ∃ N : ℕ, triangle_sum N = 2080 ∧ sum_of_digits N = 10 := by
  sorry

end NUMINAMATH_CALUDE_coin_array_problem_l1486_148635


namespace NUMINAMATH_CALUDE_quadratic_one_root_l1486_148627

theorem quadratic_one_root (a : ℝ) : 
  (∃! x : ℝ, a * x^2 - 2 * x + 1 = 0) ↔ (a = 0 ∨ a = 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l1486_148627


namespace NUMINAMATH_CALUDE_intersection_equals_one_l1486_148617

def M : Set ℕ := {0, 1}

def N : Set ℕ := {y | ∃ x ∈ M, y = 2*x + 1}

theorem intersection_equals_one : M ∩ N = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_one_l1486_148617


namespace NUMINAMATH_CALUDE_occupancy_theorem_hundred_mathematicians_l1486_148688

/-- The number of ways k mathematicians can occupy k rooms under the given conditions -/
def occupancy_ways (k : ℕ) : ℕ :=
  2^(k - 1)

/-- Theorem stating that the number of ways k mathematicians can occupy k rooms is 2^(k-1) -/
theorem occupancy_theorem (k : ℕ) (h : k > 0) :
  occupancy_ways k = 2^(k - 1) :=
by sorry

/-- Corollary for the specific case of 100 mathematicians -/
theorem hundred_mathematicians :
  occupancy_ways 100 = 2^99 :=
by sorry

end NUMINAMATH_CALUDE_occupancy_theorem_hundred_mathematicians_l1486_148688


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l1486_148676

def euler_family_ages : List ℕ := [6, 6, 6, 6, 12, 14, 14, 16]

theorem euler_family_mean_age : 
  (euler_family_ages.sum / euler_family_ages.length : ℚ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l1486_148676


namespace NUMINAMATH_CALUDE_degree_of_5x_cubed_plus_9_to_10_l1486_148699

/-- The degree of a polynomial of the form (ax³ + b)ⁿ where a and b are constants and n is a positive integer -/
def degree_of_cubic_plus_constant_to_power (a b : ℝ) (n : ℕ+) : ℕ :=
  3 * n

/-- Theorem stating that the degree of (5x³ + 9)¹⁰ is 30 -/
theorem degree_of_5x_cubed_plus_9_to_10 :
  degree_of_cubic_plus_constant_to_power 5 9 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_5x_cubed_plus_9_to_10_l1486_148699


namespace NUMINAMATH_CALUDE_heights_sum_l1486_148622

/-- Given the heights of John, Lena, and Rebeca, prove that the sum of Lena's and Rebeca's heights is 295 cm. -/
theorem heights_sum (john lena rebeca : ℕ) 
  (h1 : john = 152)
  (h2 : john = lena + 15)
  (h3 : rebeca = john + 6) :
  lena + rebeca = 295 := by
  sorry

end NUMINAMATH_CALUDE_heights_sum_l1486_148622


namespace NUMINAMATH_CALUDE_zark_game_threshold_l1486_148607

/-- The score for dropping n zarks -/
def drop_score (n : ℕ) : ℕ := n^2

/-- The score for eating n zarks -/
def eat_score (n : ℕ) : ℕ := 15 * n

/-- 16 is the smallest positive integer n for which dropping n zarks scores more than eating them -/
theorem zark_game_threshold : ∀ n : ℕ, n > 0 → (drop_score n > eat_score n ↔ n ≥ 16) :=
by sorry

end NUMINAMATH_CALUDE_zark_game_threshold_l1486_148607


namespace NUMINAMATH_CALUDE_sqrt_2_simplest_l1486_148624

-- Define a function to represent the simplicity of a square root
def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → y ≠ 1 → x ≠ y * y * (x / (y * y))

-- State the theorem
theorem sqrt_2_simplest : 
  is_simplest_sqrt (Real.sqrt 2) ∧ 
  ¬ is_simplest_sqrt (Real.sqrt 20) ∧ 
  ¬ is_simplest_sqrt (Real.sqrt (1/2)) ∧ 
  ¬ is_simplest_sqrt (Real.sqrt 0.2) :=
sorry

end NUMINAMATH_CALUDE_sqrt_2_simplest_l1486_148624


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l1486_148686

/-- The distance between the foci of a hyperbola with equation xy = 4 is 8 -/
theorem hyperbola_foci_distance : 
  ∀ (x y : ℝ), x * y = 4 → ∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁.1 * f₁.2 = 4 ∧ f₂.1 * f₂.2 = 4) ∧ 
    (Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = 8) := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_foci_distance_l1486_148686


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l1486_148670

theorem no_positive_integer_solutions : 
  ¬ ∃ (a b c : ℕ+), (a * b + b * c = 66) ∧ (a * c + b * c = 35) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l1486_148670


namespace NUMINAMATH_CALUDE_dance_team_problem_l1486_148680

def student_heights : List ℝ := [161, 162, 162, 164, 165, 165, 165, 166, 166, 167, 168, 168, 170, 172, 172, 175]

def average_height : ℝ := 166.75

def group_A : List ℝ := [162, 165, 165, 166, 166]
def group_B : List ℝ := [161, 162, 164, 165, 175]

def preselected_heights : List ℝ := [168, 168, 172]

def median (l : List ℝ) : ℝ := sorry
def mode (l : List ℝ) : ℝ := sorry
def variance (l : List ℝ) : ℝ := sorry

theorem dance_team_problem :
  (median student_heights = 166) ∧
  (mode student_heights = 165) ∧
  (variance group_A < variance group_B) ∧
  (∃ (h1 h2 : ℝ), h1 ∈ student_heights ∧ h2 ∈ student_heights ∧
    h1 = 170 ∧ h2 = 172 ∧
    variance (h1 :: h2 :: preselected_heights) < 32/9 ∧
    ∀ (x y : ℝ), x ∈ student_heights → y ∈ student_heights →
      variance (x :: y :: preselected_heights) < 32/9 →
      (x + y) / 2 ≤ (h1 + h2) / 2) :=
by sorry

#check dance_team_problem

end NUMINAMATH_CALUDE_dance_team_problem_l1486_148680


namespace NUMINAMATH_CALUDE_extreme_value_implies_ab_eq_neg_three_l1486_148658

/-- A function f(x) = ax³ + bx has an extreme value at x = 1/a -/
def has_extreme_value (a b : ℝ) : Prop :=
  let f := fun x : ℝ => a * x^3 + b * x
  ∃ (h : ℝ), h = (1 : ℝ) / a ∧ (deriv f) h = 0

/-- If f(x) = ax³ + bx has an extreme value at x = 1/a, then ab = -3 -/
theorem extreme_value_implies_ab_eq_neg_three (a b : ℝ) (h : a ≠ 0) :
  has_extreme_value a b → a * b = -3 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_implies_ab_eq_neg_three_l1486_148658


namespace NUMINAMATH_CALUDE_system_two_solutions_l1486_148605

/-- The system of equations has exactly two solutions if and only if a = 1 or a = 25 -/
theorem system_two_solutions (a : ℝ) :
  (∃! (x₁ y₁ x₂ y₂ : ℝ), 
    (abs (y₁ - 3 - x₁) + abs (y₁ - 3 + x₁) = 6 ∧
     (abs x₁ - 4)^2 + (abs y₁ - 3)^2 = a) ∧
    (abs (y₂ - 3 - x₂) + abs (y₂ - 3 + x₂) = 6 ∧
     (abs x₂ - 4)^2 + (abs y₂ - 3)^2 = a) ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)) ↔ 
  (a = 1 ∨ a = 25) :=
by sorry

end NUMINAMATH_CALUDE_system_two_solutions_l1486_148605


namespace NUMINAMATH_CALUDE_expand_product_l1486_148652

theorem expand_product (x : ℝ) : (x + 5) * (x + 9) = x^2 + 14*x + 45 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1486_148652


namespace NUMINAMATH_CALUDE_shaded_area_proof_l1486_148667

/-- Given a grid and two right triangles, prove the area of the smaller triangle -/
theorem shaded_area_proof (grid_width grid_height : ℕ) 
  (large_triangle_base large_triangle_height : ℕ)
  (small_triangle_base small_triangle_height : ℕ) :
  grid_width = 15 →
  grid_height = 5 →
  large_triangle_base = grid_width →
  large_triangle_height = grid_height - 1 →
  small_triangle_base = 12 →
  small_triangle_height = 3 →
  (small_triangle_base * small_triangle_height) / 2 = 18 := by
  sorry


end NUMINAMATH_CALUDE_shaded_area_proof_l1486_148667


namespace NUMINAMATH_CALUDE_books_selling_price_l1486_148690

/-- Calculates the total selling price of two books given their costs and profit/loss percentages -/
def total_selling_price (total_cost book1_cost loss_percent gain_percent : ℚ) : ℚ :=
  let book2_cost := total_cost - book1_cost
  let book1_sell := book1_cost * (1 - loss_percent / 100)
  let book2_sell := book2_cost * (1 + gain_percent / 100)
  book1_sell + book2_sell

/-- Theorem stating that the total selling price of two books is 297.50 Rs given the specified conditions -/
theorem books_selling_price :
  total_selling_price 300 175 15 19 = 297.50 := by
  sorry

end NUMINAMATH_CALUDE_books_selling_price_l1486_148690


namespace NUMINAMATH_CALUDE_power_product_equality_l1486_148637

theorem power_product_equality : (-0.25)^2014 * (-4)^2015 = -4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l1486_148637


namespace NUMINAMATH_CALUDE_f_properties_l1486_148611

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2

theorem f_properties :
  (∀ x, -2 ≤ f x ∧ f x ≤ 2) ∧
  (∀ p > 0, (∀ x, f (x + p) = f x) → p ≥ π) ∧
  (∀ x, f (x + π) = f x) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1486_148611


namespace NUMINAMATH_CALUDE_square_garden_multiple_l1486_148682

/-- Given a square garden with perimeter 40 feet and area equal to a multiple of the perimeter plus 20, prove that the multiple is 2. -/
theorem square_garden_multiple (side : ℝ) (multiple : ℝ) : 
  side > 0 →
  4 * side = 40 →
  side^2 = multiple * 40 + 20 →
  multiple = 2 := by sorry

end NUMINAMATH_CALUDE_square_garden_multiple_l1486_148682


namespace NUMINAMATH_CALUDE_arrow_sequence_equivalence_l1486_148657

/-- Represents a point in the cycle -/
def CyclePoint := ℕ

/-- The length of the cycle -/
def cycleLength : ℕ := 5

/-- Returns the equivalent point within the cycle -/
def cycleEquivalent (n : ℕ) : CyclePoint :=
  n % cycleLength

/-- Theorem: The sequence of arrows from point 630 to point 633 is equivalent
    to the sequence from point 0 to point 3 in a cycle of length 5 -/
theorem arrow_sequence_equivalence :
  (cycleEquivalent 630 = cycleEquivalent 0) ∧
  (cycleEquivalent 631 = cycleEquivalent 1) ∧
  (cycleEquivalent 632 = cycleEquivalent 2) ∧
  (cycleEquivalent 633 = cycleEquivalent 3) := by
  sorry


end NUMINAMATH_CALUDE_arrow_sequence_equivalence_l1486_148657


namespace NUMINAMATH_CALUDE_solution_t_l1486_148694

theorem solution_t (t : ℝ) : 
  Real.sqrt (3 * Real.sqrt (t - 3)) = (10 - t) ^ (1/4) → t = 37/10 := by
  sorry

end NUMINAMATH_CALUDE_solution_t_l1486_148694


namespace NUMINAMATH_CALUDE_square_equation_solution_l1486_148644

theorem square_equation_solution : ∃ (M : ℕ), M > 0 ∧ 33^2 * 66^2 = 15^2 * M^2 ∧ M = 726 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l1486_148644


namespace NUMINAMATH_CALUDE_largest_number_in_sequence_l1486_148615

/-- A sequence of 8 increasing real numbers -/
def IncreasingSequence : Type := { s : Fin 8 → ℝ // ∀ i j, i < j → s i < s j }

/-- Checks if a subsequence of 4 consecutive numbers is an arithmetic progression -/
def IsArithmeticProgression (s : IncreasingSequence) (start : Fin 5) (d : ℝ) : Prop :=
  ∀ i : Fin 3, s.val (start + i + 1) - s.val (start + i) = d

/-- Checks if a subsequence of 4 consecutive numbers is a geometric progression -/
def IsGeometricProgression (s : IncreasingSequence) (start : Fin 5) : Prop :=
  ∃ r : ℝ, ∀ i : Fin 3, s.val (start + i + 1) / s.val (start + i) = r

/-- The main theorem -/
theorem largest_number_in_sequence (s : IncreasingSequence) 
  (h1 : ∃ start1 : Fin 5, IsArithmeticProgression s start1 4)
  (h2 : ∃ start2 : Fin 5, IsArithmeticProgression s start2 36)
  (h3 : ∃ start3 : Fin 5, IsGeometricProgression s start3) :
  s.val 7 = 126 ∨ s.val 7 = 6 := by
  sorry


end NUMINAMATH_CALUDE_largest_number_in_sequence_l1486_148615


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l1486_148616

theorem smaller_number_in_ratio (a b d u v : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : u > 0) (h4 : v > 0)
  (h5 : u / v = b / a) (h6 : u + v = d) : 
  min u v = a * d / (a + b) := by
sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l1486_148616


namespace NUMINAMATH_CALUDE_correct_factorization_l1486_148603

theorem correct_factorization (a : ℝ) : a^2 - 3*a - 4 = (a - 4) * (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l1486_148603


namespace NUMINAMATH_CALUDE_carissa_street_crossing_l1486_148623

/-- Carissa's street crossing problem -/
theorem carissa_street_crossing 
  (walking_speed : ℝ) 
  (street_width : ℝ) 
  (total_time : ℝ) 
  (n : ℝ) 
  (h1 : walking_speed = 2) 
  (h2 : street_width = 260) 
  (h3 : total_time = 30) 
  (h4 : n > 0) :
  let running_speed := n * walking_speed
  let walking_time := total_time / (1 + n)
  let running_time := n * walking_time
  walking_speed * walking_time + running_speed * running_time = street_width →
  running_speed = 10 := by sorry

end NUMINAMATH_CALUDE_carissa_street_crossing_l1486_148623


namespace NUMINAMATH_CALUDE_problems_solved_l1486_148620

theorem problems_solved (first last : ℕ) (h : first = 78 ∧ last = 125) : 
  (last - first + 1 : ℕ) = 49 := by
  sorry

end NUMINAMATH_CALUDE_problems_solved_l1486_148620


namespace NUMINAMATH_CALUDE_roots_equation_value_l1486_148672

theorem roots_equation_value (a b : ℝ) : 
  a^2 - a - 3 = 0 ∧ b^2 - b - 3 = 0 →
  2*a^3 + b^2 + 3*a^2 - 11*a - b + 5 = 23 :=
by sorry

end NUMINAMATH_CALUDE_roots_equation_value_l1486_148672


namespace NUMINAMATH_CALUDE_area_ratio_dodecagon_quadrilateral_l1486_148648

/-- A regular dodecagon -/
structure RegularDodecagon where
  -- We don't need to define the vertices explicitly
  area : ℝ

/-- A quadrilateral formed by connecting every third vertex of a regular dodecagon -/
structure Quadrilateral where
  area : ℝ

/-- The theorem stating the ratio of areas -/
theorem area_ratio_dodecagon_quadrilateral 
  (d : RegularDodecagon) 
  (q : Quadrilateral) : 
  q.area / d.area = Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_dodecagon_quadrilateral_l1486_148648


namespace NUMINAMATH_CALUDE_line_point_sum_l1486_148662

/-- The line equation y = -1/2x + 8 -/
def line_equation (x y : ℝ) : Prop := y = -1/2 * x + 8

/-- Point P is where the line crosses the x-axis -/
def point_P : ℝ × ℝ := (16, 0)

/-- Point Q is where the line crosses the y-axis -/
def point_Q : ℝ × ℝ := (0, 8)

/-- Point T is on the line segment PQ -/
def point_T_on_PQ (r s : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
  r = t * point_P.1 + (1 - t) * point_Q.1 ∧
  s = t * point_P.2 + (1 - t) * point_Q.2

/-- The area of triangle POQ is four times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop :=
  abs ((1/2) * point_P.1 * point_Q.2) = 4 * abs ((1/2) * r * s)

theorem line_point_sum (r s : ℝ) :
  line_equation r s →
  point_T_on_PQ r s →
  area_condition r s →
  r + s = 14 :=
by sorry

end NUMINAMATH_CALUDE_line_point_sum_l1486_148662


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1486_148663

theorem pure_imaginary_complex_number (x : ℝ) :
  let z : ℂ := (x^2 - 1) + (x - 1) * Complex.I
  (∀ r : ℝ, z ≠ r) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1486_148663


namespace NUMINAMATH_CALUDE_watermelon_seeds_l1486_148640

/-- Given 4 watermelons with a total of 400 seeds, prove that each watermelon has 100 seeds. -/
theorem watermelon_seeds (num_watermelons : ℕ) (total_seeds : ℕ) 
  (h1 : num_watermelons = 4) 
  (h2 : total_seeds = 400) : 
  total_seeds / num_watermelons = 100 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_seeds_l1486_148640


namespace NUMINAMATH_CALUDE_both_selected_probability_l1486_148610

theorem both_selected_probability (ram_prob ravi_prob : ℚ) 
  (h1 : ram_prob = 2/7)
  (h2 : ravi_prob = 1/5) :
  ram_prob * ravi_prob = 2/35 := by
  sorry

end NUMINAMATH_CALUDE_both_selected_probability_l1486_148610


namespace NUMINAMATH_CALUDE_log_y_equals_value_l1486_148612

theorem log_y_equals_value (y : ℝ) (h : Real.log y / Real.log 8 = 2.75) : 
  y = 256 * Real.rpow 2 (1/4) := by
  sorry

end NUMINAMATH_CALUDE_log_y_equals_value_l1486_148612


namespace NUMINAMATH_CALUDE_principal_amount_l1486_148692

/-- Proves that given the conditions of the problem, the principal amount must be 600 --/
theorem principal_amount (P R : ℝ) : 
  (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 300 →
  P = 600 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_l1486_148692


namespace NUMINAMATH_CALUDE_circle_radius_l1486_148684

theorem circle_radius (C : ℝ) (r : ℝ) (h : C = 72 * Real.pi) : C = 2 * Real.pi * r → r = 36 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l1486_148684


namespace NUMINAMATH_CALUDE_equation_solution_l1486_148636

theorem equation_solution (x y : ℝ) (h : x - y / 2 = 1) : y = 2 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1486_148636


namespace NUMINAMATH_CALUDE_special_sequence_property_l1486_148634

/-- A sequence of positive real numbers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ 
  (∀ m n : ℕ, m > 0 → n > 0 → (m + n : ℝ) * a (m + n) ≤ a m + a n) ∧
  (∀ i : ℕ, i > 0 → a i > 0)

/-- The main theorem to be proved -/
theorem special_sequence_property (a : ℕ → ℝ) (h : SpecialSequence a) : 
  1 / a 200 > 4 * 10^7 := by sorry

end NUMINAMATH_CALUDE_special_sequence_property_l1486_148634


namespace NUMINAMATH_CALUDE_fundraising_ratio_l1486_148649

-- Define the fundraising goal
def goal : ℕ := 4000

-- Define Ken's collection
def ken_collection : ℕ := 600

-- Define the amount they exceeded the goal by
def excess : ℕ := 600

-- Define the total amount collected
def total_collected : ℕ := goal + excess

-- Define Mary's collection as a function of Ken's
def mary_collection (x : ℚ) : ℚ := x * ken_collection

-- Define Scott's collection as a function of Mary's
def scott_collection (x : ℚ) : ℚ := (1 / 3) * mary_collection x

-- State the theorem
theorem fundraising_ratio : 
  ∃ x : ℚ, 
    scott_collection x + mary_collection x + ken_collection = total_collected ∧ 
    mary_collection x / ken_collection = 5 := by
  sorry

end NUMINAMATH_CALUDE_fundraising_ratio_l1486_148649


namespace NUMINAMATH_CALUDE_negation_equivalence_l1486_148628

theorem negation_equivalence : 
  (¬ ∃ (x : ℝ), x > 0 ∧ Real.sqrt x ≤ x + 1) ↔ 
  (∀ (x : ℝ), x > 0 → Real.sqrt x > x + 1) := by
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1486_148628


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_exp_greater_than_x_l1486_148609

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem negation_of_exp_greater_than_x :
  (¬ ∃ x : ℝ, Real.exp x > x) ↔ (∀ x : ℝ, Real.exp x ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_exp_greater_than_x_l1486_148609


namespace NUMINAMATH_CALUDE_abs_neg_three_l1486_148626

theorem abs_neg_three : abs (-3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_l1486_148626


namespace NUMINAMATH_CALUDE_multiples_of_3_or_5_not_6_l1486_148625

def count_multiples (n : ℕ) (max : ℕ) : ℕ :=
  (max / n)

theorem multiples_of_3_or_5_not_6 (max : ℕ) (h : max = 150) : 
  count_multiples 3 max + count_multiples 5 max - count_multiples 15 max - count_multiples 6 max = 45 := by
  sorry

#check multiples_of_3_or_5_not_6

end NUMINAMATH_CALUDE_multiples_of_3_or_5_not_6_l1486_148625


namespace NUMINAMATH_CALUDE_abc_product_l1486_148661

theorem abc_product (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (h1 : a + 1 = b + 2) (h2 : b + 2 = c + 3) :
  a * b * c = c * (c + 1) * (c + 2) := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l1486_148661


namespace NUMINAMATH_CALUDE_triangle_cosine_difference_l1486_148646

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if 4b * sin A = √7 * a and a, b, c are in arithmetic progression
    with positive common difference, then cos A - cos C = √7/2 -/
theorem triangle_cosine_difference (a b c : ℝ) (A B C : ℝ) (h1 : 4 * b * Real.sin A = Real.sqrt 7 * a)
  (h2 : ∃ (d : ℝ), d > 0 ∧ b = a + d ∧ c = b + d) :
  Real.cos A - Real.cos C = Real.sqrt 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_difference_l1486_148646


namespace NUMINAMATH_CALUDE_min_value_exponential_sum_l1486_148698

theorem min_value_exponential_sum (x y : ℝ) (h : 2 * x + 3 * y = 6) :
  ∃ (m : ℝ), m = 16 ∧ ∀ a b, 2 * a + 3 * b = 6 → 4^a + 8^b ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_exponential_sum_l1486_148698


namespace NUMINAMATH_CALUDE_x_value_proof_l1486_148606

theorem x_value_proof : ∀ x : ℝ, x + Real.sqrt 25 = Real.sqrt 36 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1486_148606


namespace NUMINAMATH_CALUDE_geometric_sequence_sufficient_not_necessary_l1486_148653

/-- Defines a geometric sequence of three real numbers -/
def is_geometric_sequence (a b c : ℝ) : Prop :=
  b ≠ 0 ∧ c / b = b / a

/-- Proves that "a, b, c form a geometric sequence" is a sufficient but not necessary condition for "b^2 = ac" -/
theorem geometric_sequence_sufficient_not_necessary :
  (∀ a b c : ℝ, is_geometric_sequence a b c → b^2 = a*c) ∧
  (∃ a b c : ℝ, b^2 = a*c ∧ ¬is_geometric_sequence a b c) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sufficient_not_necessary_l1486_148653


namespace NUMINAMATH_CALUDE_second_square_size_l1486_148632

/-- Represents a square on the board -/
structure Square :=
  (size : Nat)
  (position : Nat × Nat)

/-- Represents the board configuration -/
def BoardConfiguration := List Square

/-- Checks if a given configuration covers the entire 10x10 board -/
def covers_board (config : BoardConfiguration) : Prop := sorry

/-- Checks if all squares in the configuration have different sizes -/
def all_different_sizes (config : BoardConfiguration) : Prop := sorry

/-- Checks if the last two squares in the configuration are 5x5 and 4x4 -/
def last_two_squares_correct (config : BoardConfiguration) : Prop := sorry

/-- Checks if the second square in the configuration is 8x8 -/
def second_square_is_8x8 (config : BoardConfiguration) : Prop := sorry

theorem second_square_size (config : BoardConfiguration) :
  config.length = 6 →
  covers_board config →
  all_different_sizes config →
  last_two_squares_correct config →
  second_square_is_8x8 config :=
sorry

end NUMINAMATH_CALUDE_second_square_size_l1486_148632


namespace NUMINAMATH_CALUDE_min_value_theorem_l1486_148618

theorem min_value_theorem (a b c d : ℝ) (sum_constraint : a + b + c + d = 8) :
  20 * (a^2 + b^2 + c^2 + d^2) - (a^3 * b + a^3 * c + a^3 * d + b^3 * a + b^3 * c + b^3 * d + c^3 * a + c^3 * b + c^3 * d + d^3 * a + d^3 * b + d^3 * c) ≥ 112 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1486_148618


namespace NUMINAMATH_CALUDE_dot_product_parallel_l1486_148631

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define parallel vectors
def parallel (a b : V) : Prop := ∃ k : ℝ, a = k • b ∨ b = k • a

theorem dot_product_parallel (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  (inner a b = ‖a‖ * ‖b‖ → parallel a b) ∧
  ¬(parallel a b → inner a b = ‖a‖ * ‖b‖) :=
sorry

end NUMINAMATH_CALUDE_dot_product_parallel_l1486_148631


namespace NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l1486_148691

theorem tic_tac_toe_tie_probability (max_win_prob zoe_win_prob : ℚ) :
  max_win_prob = 4/9 →
  zoe_win_prob = 5/12 →
  1 - (max_win_prob + zoe_win_prob) = 5/36 := by
sorry

end NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l1486_148691


namespace NUMINAMATH_CALUDE_tangent_line_min_slope_l1486_148695

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 6 * x^2 - 3

-- Theorem statement
theorem tangent_line_min_slope :
  ∃ (a b : ℝ), 
    (∀ x : ℝ, f' x ≥ f' a) ∧ 
    (∀ x : ℝ, f x = f a + f' a * (x - a)) ∧ 
    (b = -3 * a) ∧
    (∀ x : ℝ, f x = f a + b * (x - a)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_min_slope_l1486_148695


namespace NUMINAMATH_CALUDE_consecutive_numbers_product_invariant_l1486_148669

theorem consecutive_numbers_product_invariant :
  ∃ (a : ℕ), 
    let original := [a, a+1, a+2, a+3, a+4, a+5, a+6]
    ∃ (modified : List ℕ),
      (∀ i, i ∈ original → ∃ j, j ∈ modified ∧ (j = i - 1 ∨ j = i ∨ j = i + 1)) ∧
      (modified.length = 7) ∧
      (original.prod = modified.prod) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_product_invariant_l1486_148669


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l1486_148651

theorem sum_of_squares_and_products (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : x^2 + y^2 + z^2 = 75)
  (h5 : x*y + y*z + z*x = 28) :
  x + y + z = Real.sqrt 131 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l1486_148651


namespace NUMINAMATH_CALUDE_fraction_product_l1486_148689

theorem fraction_product : (2 : ℚ) / 9 * (-4 : ℚ) / 5 = (-8 : ℚ) / 45 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l1486_148689


namespace NUMINAMATH_CALUDE_not_all_right_triangles_are_isosceles_l1486_148621

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  sum_angles : angleA + angleB + angleC = 180
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- Define an isosceles triangle
def IsIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

-- Define a right triangle
def IsRight (t : Triangle) : Prop :=
  t.angleA = 90 ∨ t.angleB = 90 ∨ t.angleC = 90

-- The theorem to prove
theorem not_all_right_triangles_are_isosceles :
  ¬ (∀ t : Triangle, IsRight t → IsIsosceles t) :=
sorry

end NUMINAMATH_CALUDE_not_all_right_triangles_are_isosceles_l1486_148621


namespace NUMINAMATH_CALUDE_P_in_second_quadrant_l1486_148639

-- Define the point P
def P (x : ℝ) : ℝ × ℝ := (-2, x^2 + 1)

-- Define what it means for a point to be in the second quadrant
def is_in_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- Theorem stating that P is in the second quadrant for all real x
theorem P_in_second_quadrant (x : ℝ) : is_in_second_quadrant (P x) := by
  sorry


end NUMINAMATH_CALUDE_P_in_second_quadrant_l1486_148639


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1486_148656

/-- Represents the side lengths of an isosceles triangle -/
structure IsoscelesTriangle where
  equalSide : ℝ
  baseSide : ℝ

/-- Checks if the triangle satisfies the given conditions -/
def satisfiesConditions (t : IsoscelesTriangle) : Prop :=
  t.equalSide = 20 ∧ t.baseSide = (2/5) * t.equalSide

/-- Calculates the perimeter of the triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  2 * t.equalSide + t.baseSide

/-- Theorem stating that the perimeter of the triangle is 48 cm -/
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, satisfiesConditions t → perimeter t = 48 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1486_148656


namespace NUMINAMATH_CALUDE_binomial_coefficient_x_squared_l1486_148665

theorem binomial_coefficient_x_squared (x : ℝ) : 
  (Finset.range 11).sum (fun k => Nat.choose 10 k * x^(10 - k) * (1/x)^k) = 
  210 * x^2 + (Finset.range 11).sum (fun k => if k ≠ 4 then Nat.choose 10 k * x^(10 - k) * (1/x)^k else 0) :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_x_squared_l1486_148665


namespace NUMINAMATH_CALUDE_roots_of_equation_l1486_148659

/-- The polynomial equation whose roots we want to find -/
def f (x : ℝ) : ℝ := (x^3 - 4*x^2 - x + 4)*(x-3)*(x+2)

/-- The set of roots we claim to be correct -/
def root_set : Set ℝ := {-2, -1, 1, 3, 4}

/-- Theorem stating that the roots of the equation are exactly the elements of root_set -/
theorem roots_of_equation : 
  ∀ x : ℝ, f x = 0 ↔ x ∈ root_set :=
by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l1486_148659


namespace NUMINAMATH_CALUDE_digit_150_of_1_13_l1486_148602

/-- The decimal representation of 1/13 as a sequence of digits -/
def decimal_rep_1_13 : ℕ → Fin 10 := fun n => 
  match n % 6 with
  | 0 => 0
  | 1 => 7
  | 2 => 6
  | 3 => 9
  | 4 => 2
  | 5 => 3
  | _ => 0 -- This case is unreachable, but needed for exhaustiveness

/-- The 150th digit after the decimal point in the decimal representation of 1/13 is 3 -/
theorem digit_150_of_1_13 : decimal_rep_1_13 150 = 3 := by
  sorry

end NUMINAMATH_CALUDE_digit_150_of_1_13_l1486_148602


namespace NUMINAMATH_CALUDE_mean_temperature_is_80_point_2_l1486_148681

def temperatures : List ℝ := [75, 77, 76, 78, 80, 81, 83, 82, 84, 86]

theorem mean_temperature_is_80_point_2 :
  (temperatures.sum / temperatures.length : ℝ) = 80.2 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_is_80_point_2_l1486_148681


namespace NUMINAMATH_CALUDE_farm_animals_l1486_148650

theorem farm_animals (total_legs : ℕ) (total_animals : ℕ) (chicken_legs : ℕ) (sheep_legs : ℕ) :
  total_legs = 60 →
  total_animals = 20 →
  chicken_legs = 2 →
  sheep_legs = 4 →
  ∃ (num_chickens num_sheep : ℕ),
    num_chickens + num_sheep = total_animals ∧
    num_chickens * chicken_legs + num_sheep * sheep_legs = total_legs ∧
    num_sheep = 10 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l1486_148650


namespace NUMINAMATH_CALUDE_praveen_initial_investment_l1486_148608

-- Define the initial parameters
def haris_investment : ℕ := 8280
def praveens_time : ℕ := 12
def haris_time : ℕ := 7
def profit_ratio_praveen : ℕ := 2
def profit_ratio_hari : ℕ := 3

-- Define Praveen's investment as a function
def praveens_investment : ℕ := 
  (haris_investment * haris_time * profit_ratio_praveen) / (praveens_time * profit_ratio_hari)

-- Theorem statement
theorem praveen_initial_investment :
  praveens_investment = 3220 :=
sorry

end NUMINAMATH_CALUDE_praveen_initial_investment_l1486_148608


namespace NUMINAMATH_CALUDE_money_division_l1486_148633

theorem money_division (total : ℚ) (a b c : ℚ) 
  (h1 : total = 527)
  (h2 : a = (2/3) * b)
  (h3 : b = (1/4) * c)
  (h4 : a + b + c = total) :
  b = 93 := by
  sorry

end NUMINAMATH_CALUDE_money_division_l1486_148633


namespace NUMINAMATH_CALUDE_hair_cut_calculation_l1486_148677

/-- Given the total amount of hair cut and the amount cut on the first day,
    calculate the amount cut on the second day. -/
theorem hair_cut_calculation (total : ℝ) (first_day : ℝ) (h1 : total = 0.88) (h2 : first_day = 0.38) :
  total - first_day = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_hair_cut_calculation_l1486_148677


namespace NUMINAMATH_CALUDE_absolute_value_sum_zero_l1486_148697

theorem absolute_value_sum_zero (x y : ℝ) :
  |x - 3| + |y + 2| = 0 → (y - x = -5 ∧ x * y = -6) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_zero_l1486_148697


namespace NUMINAMATH_CALUDE_area_relation_implies_parallel_diagonals_l1486_148683

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (P Q R S : Point)

/-- Area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Check if two line segments are parallel -/
def parallel (A B C D : Point) : Prop := sorry

/-- Points A, B, C, D lie on the sides of quadrilateral PQRS -/
def pointsOnSides (PQRS : Quadrilateral) (A B C D : Point) : Prop := sorry

theorem area_relation_implies_parallel_diagonals 
  (PQRS : Quadrilateral) (A B C D : Point) :
  pointsOnSides PQRS A B C D →
  area PQRS = 2 * area ⟨A, B, C, D⟩ →
  parallel A C Q R ∨ parallel B D P Q := by
  sorry

end NUMINAMATH_CALUDE_area_relation_implies_parallel_diagonals_l1486_148683


namespace NUMINAMATH_CALUDE_loan_amount_proof_l1486_148664

/-- Represents the interest rate as a decimal -/
def interest_rate : ℝ := 0.04

/-- Represents the loan duration in years -/
def years : ℕ := 2

/-- Calculates the compound interest amount after n years -/
def compound_interest (P : ℝ) : ℝ := P * (1 + interest_rate) ^ years

/-- Calculates the simple interest amount after n years -/
def simple_interest (P : ℝ) : ℝ := P * (1 + interest_rate * years)

/-- The difference between compound and simple interest -/
def interest_difference : ℝ := 10.40

theorem loan_amount_proof (P : ℝ) : 
  compound_interest P - simple_interest P = interest_difference → P = 6500 := by
  sorry

end NUMINAMATH_CALUDE_loan_amount_proof_l1486_148664


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1486_148685

theorem closest_integer_to_cube_root : ∃ n : ℤ, 
  n = 10 ∧ ∀ m : ℤ, |n - (7^3 + 9^3)^(1/3)| ≤ |m - (7^3 + 9^3)^(1/3)| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1486_148685


namespace NUMINAMATH_CALUDE_subtraction_multiplication_fractions_l1486_148678

theorem subtraction_multiplication_fractions :
  (5 / 12 - 1 / 6) * (3 / 4) = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_multiplication_fractions_l1486_148678


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l1486_148679

theorem quadratic_roots_sum_product (m n : ℝ) : 
  (m^2 + 4*m - 1 = 0) → 
  (n^2 + 4*n - 1 = 0) → 
  m + n + m*n = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l1486_148679


namespace NUMINAMATH_CALUDE_possible_AC_values_l1486_148647

/-- Three points on a line with given distances between them -/
structure ThreePointsOnLine where
  A : ℝ
  B : ℝ
  C : ℝ
  AB_eq : |A - B| = 3
  BC_eq : |B - C| = 5

/-- The possible values for AC given AB = 3 and BC = 5 -/
theorem possible_AC_values (p : ThreePointsOnLine) : 
  |p.A - p.C| = 2 ∨ |p.A - p.C| = 8 :=
by sorry

end NUMINAMATH_CALUDE_possible_AC_values_l1486_148647


namespace NUMINAMATH_CALUDE_shirt_price_change_l1486_148668

theorem shirt_price_change (P : ℝ) (x : ℝ) (h : P > 0) :
  P * (1 + x / 100) * (1 - x / 100) = 0.84 * P → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_shirt_price_change_l1486_148668


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l1486_148604

theorem geometric_series_ratio (a r : ℝ) (hr : |r| < 1) :
  (a / (1 - r) = 16 * (a * r^2 / (1 - r))) → |r| = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l1486_148604


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1486_148629

open Real

noncomputable def seriesTerms (k : ℕ) : ℝ :=
  (7^k) / ((4^k - 3^k) * (4^(k+1) - 3^(k+1)))

theorem infinite_series_sum : 
  ∑' k, seriesTerms k = 7 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1486_148629


namespace NUMINAMATH_CALUDE_sophia_pie_consumption_l1486_148654

theorem sophia_pie_consumption (pie_weight : ℝ) (fridge_weight : ℝ) : 
  fridge_weight = (5/6) * pie_weight ∧ fridge_weight = 1200 → 
  pie_weight - fridge_weight = 240 := by
sorry

end NUMINAMATH_CALUDE_sophia_pie_consumption_l1486_148654


namespace NUMINAMATH_CALUDE_complex_expression_value_l1486_148671

theorem complex_expression_value : 
  (1 : ℝ) * (2 * 7 / 9) ^ (1 / 2 : ℝ) - (2 * Real.sqrt 3 - Real.pi) ^ (0 : ℝ) - 
  (2 * 10 / 27) ^ (-(2 / 3 : ℝ)) + (1 / 4 : ℝ) ^ (-(3 / 2 : ℝ)) = 389 / 48 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_value_l1486_148671


namespace NUMINAMATH_CALUDE_coconut_juice_unit_electric_water_heater_unit_l1486_148630

-- Define the types of containers
inductive Container
| CoconutJuiceBottle
| ElectricWaterHeater

-- Define the volume units
inductive VolumeUnit
| Milliliter
| Liter

-- Define a function to get the appropriate volume unit for a container
def appropriateUnit (container : Container) (volume : ℕ) : VolumeUnit :=
  match container with
  | Container.CoconutJuiceBottle => VolumeUnit.Milliliter
  | Container.ElectricWaterHeater => VolumeUnit.Liter

-- Theorem for coconut juice bottle
theorem coconut_juice_unit : 
  appropriateUnit Container.CoconutJuiceBottle 200 = VolumeUnit.Milliliter :=
by sorry

-- Theorem for electric water heater
theorem electric_water_heater_unit : 
  appropriateUnit Container.ElectricWaterHeater 50 = VolumeUnit.Liter :=
by sorry

end NUMINAMATH_CALUDE_coconut_juice_unit_electric_water_heater_unit_l1486_148630


namespace NUMINAMATH_CALUDE_total_turnips_after_selling_l1486_148614

/-- The total number of turnips after selling some -/
def totalTurnipsAfterSelling (melanieTurnips bennyTurnips sarahTurnips davidTurnips melanieSold davidSold : ℕ) : ℕ :=
  (melanieTurnips - melanieSold) + bennyTurnips + sarahTurnips + (davidTurnips - davidSold)

/-- Theorem stating the total number of turnips after selling -/
theorem total_turnips_after_selling :
  totalTurnipsAfterSelling 139 113 195 87 32 15 = 487 := by
  sorry

#eval totalTurnipsAfterSelling 139 113 195 87 32 15

end NUMINAMATH_CALUDE_total_turnips_after_selling_l1486_148614


namespace NUMINAMATH_CALUDE_min_correct_answers_to_advance_l1486_148601

/-- Represents a math competition with specified rules -/
structure MathCompetition where
  total_questions : ℕ
  points_correct : ℕ
  points_incorrect : ℕ
  min_score : ℕ

/-- Calculates the score for a given number of correct answers in the competition -/
def calculate_score (comp : MathCompetition) (correct_answers : ℕ) : ℤ :=
  (correct_answers * comp.points_correct : ℤ) - 
  ((comp.total_questions - correct_answers) * comp.points_incorrect : ℤ)

/-- Theorem stating the minimum number of correct answers needed to advance -/
theorem min_correct_answers_to_advance (comp : MathCompetition) 
  (h1 : comp.total_questions = 25)
  (h2 : comp.points_correct = 4)
  (h3 : comp.points_incorrect = 1)
  (h4 : comp.min_score = 60) :
  ∃ (n : ℕ), n = 17 ∧ 
    (∀ (m : ℕ), m ≥ n → calculate_score comp m ≥ comp.min_score) ∧
    (∀ (m : ℕ), m < n → calculate_score comp m < comp.min_score) :=
  sorry

end NUMINAMATH_CALUDE_min_correct_answers_to_advance_l1486_148601
