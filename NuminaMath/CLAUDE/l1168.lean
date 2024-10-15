import Mathlib

namespace NUMINAMATH_CALUDE_bridget_apples_l1168_116886

theorem bridget_apples (x : ℕ) : 
  (x : ℚ) / 3 + 5 + 6 = x → x = 17 := by
  sorry

end NUMINAMATH_CALUDE_bridget_apples_l1168_116886


namespace NUMINAMATH_CALUDE_combination_sum_l1168_116889

theorem combination_sum : Nat.choose 5 2 + Nat.choose 5 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_l1168_116889


namespace NUMINAMATH_CALUDE_correct_calculation_l1168_116851

theorem correct_calculation (x : ℝ) : 5 * x + 4 = 104 → (x + 5) / 4 = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1168_116851


namespace NUMINAMATH_CALUDE_power_equation_solution_l1168_116820

theorem power_equation_solution (n : ℕ) : 2^n = 8^20 → n = 60 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1168_116820


namespace NUMINAMATH_CALUDE_triangle_side_length_l1168_116844

theorem triangle_side_length (a c : ℝ) (B : ℝ) (h1 : a = 2) (h2 : c = 4) (h3 : B = π / 3) :
  let b := Real.sqrt (a^2 + c^2 - 2 * a * c * Real.cos B)
  b = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1168_116844


namespace NUMINAMATH_CALUDE_parallel_lines_m_opposite_sides_m_range_l1168_116896

-- Define the lines and points
def l1 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def l2 (m : ℝ) (x y : ℝ) : Prop := (x + 2) * (y - 4) = (x - m) * (y - m)
def point_A (m : ℝ) := (-2, m)
def point_B (m : ℝ) := (m, 4)

-- Define parallel lines
def parallel (m : ℝ) : Prop := ∀ x y, l1 x y → l2 m x y

-- Define points on opposite sides of a line
def opposite_sides (m : ℝ) : Prop :=
  (2 * (-2) + m - 1) * (2 * m + 4 - 1) < 0

-- Theorem statements
theorem parallel_lines_m (m : ℝ) : parallel m → m = -8 := by sorry

theorem opposite_sides_m_range (m : ℝ) : opposite_sides m → -3/2 < m ∧ m < 5 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_opposite_sides_m_range_l1168_116896


namespace NUMINAMATH_CALUDE_squats_on_third_day_l1168_116869

/-- Calculates the number of squats on a given day, given the initial number and daily increase. -/
def squatsOnDay (initialSquats : ℕ) (dailyIncrease : ℕ) (day : ℕ) : ℕ :=
  initialSquats + (day * dailyIncrease)

/-- Theorem: Given an initial number of 30 squats and a daily increase of 5 squats,
    the number of squats on the third day will be 45. -/
theorem squats_on_third_day :
  squatsOnDay 30 5 2 = 45 := by
  sorry


end NUMINAMATH_CALUDE_squats_on_third_day_l1168_116869


namespace NUMINAMATH_CALUDE_expression_value_l1168_116898

theorem expression_value : 
  let a : ℝ := Real.sqrt 3 - Real.sqrt 2
  let b : ℝ := Real.sqrt 3 + Real.sqrt 2
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) + 2*a*(b - a) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1168_116898


namespace NUMINAMATH_CALUDE_guinea_pig_food_theorem_l1168_116899

/-- The amount of food eaten by the first guinea pig -/
def first_guinea_pig_food : ℝ := 2

/-- The amount of food eaten by the second guinea pig -/
def second_guinea_pig_food : ℝ := 2 * first_guinea_pig_food

/-- The amount of food eaten by the third guinea pig -/
def third_guinea_pig_food : ℝ := second_guinea_pig_food + 3

/-- The total amount of food eaten by all three guinea pigs -/
def total_food : ℝ := first_guinea_pig_food + second_guinea_pig_food + third_guinea_pig_food

theorem guinea_pig_food_theorem :
  first_guinea_pig_food = 2 ∧ total_food = 13 :=
sorry

end NUMINAMATH_CALUDE_guinea_pig_food_theorem_l1168_116899


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l1168_116806

theorem banana_orange_equivalence : 
  ∀ (banana_value orange_value : ℚ),
  (3/4 : ℚ) * 12 * banana_value = 9 * orange_value →
  (2/3 : ℚ) * 6 * banana_value = 4 * orange_value :=
by
  sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l1168_116806


namespace NUMINAMATH_CALUDE_problem_solution_l1168_116843

def p₁ : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

def p₂ : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - 1 ≥ 0

theorem problem_solution : ¬p₁ ∧ p₂ := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1168_116843


namespace NUMINAMATH_CALUDE_total_gumballs_l1168_116858

/-- Represents the number of gumballs in a small package -/
def small_package : ℕ := 5

/-- Represents the number of gumballs in a medium package -/
def medium_package : ℕ := 12

/-- Represents the number of gumballs in a large package -/
def large_package : ℕ := 20

/-- Represents the number of small packages Nathan bought -/
def small_quantity : ℕ := 4

/-- Represents the number of medium packages Nathan bought -/
def medium_quantity : ℕ := 3

/-- Represents the number of large packages Nathan bought -/
def large_quantity : ℕ := 2

/-- Theorem stating the total number of gumballs Nathan ate -/
theorem total_gumballs : 
  small_quantity * small_package + 
  medium_quantity * medium_package + 
  large_quantity * large_package = 96 := by
  sorry

end NUMINAMATH_CALUDE_total_gumballs_l1168_116858


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1168_116868

theorem circle_center_and_radius :
  let eq := fun (x y : ℝ) => x^2 - 6*x + y^2 + 2*y - 9 = 0
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (3, -1) ∧ 
    radius = Real.sqrt 19 ∧
    ∀ (x y : ℝ), eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l1168_116868


namespace NUMINAMATH_CALUDE_min_value_of_function_l1168_116883

theorem min_value_of_function (x : ℝ) (h : 0 < x ∧ x < π) :
  ∃ (y : ℝ), y = (2 - Real.cos x) / Real.sin x ∧
  (∀ (z : ℝ), z = (2 - Real.cos x) / Real.sin x → y ≤ z) ∧
  y = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1168_116883


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l1168_116808

theorem sqrt_difference_equality (x a : ℝ) (m n : ℤ) (h : 0 < a) (h1 : 0 < m) (h2 : 0 < n) 
  (h3 : x + Real.sqrt (x^2 - 1) = a^((m - n : ℝ) / (2 * m * n : ℝ))) :
  x - Real.sqrt (x^2 - 1) = a^((n - m : ℝ) / (2 * m * n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l1168_116808


namespace NUMINAMATH_CALUDE_lara_age_proof_l1168_116854

/-- Lara's age 7 years ago -/
def lara_age_7_years_ago : ℕ := 9

/-- Years since Lara was 9 -/
def years_since_9 : ℕ := 7

/-- Years until future age -/
def years_to_future : ℕ := 10

/-- Lara's future age -/
def lara_future_age : ℕ := lara_age_7_years_ago + years_since_9 + years_to_future

theorem lara_age_proof : lara_future_age = 26 := by
  sorry

end NUMINAMATH_CALUDE_lara_age_proof_l1168_116854


namespace NUMINAMATH_CALUDE_square_difference_l1168_116817

theorem square_difference (m n : ℝ) (h1 : m + n = 3) (h2 : m - n = 4) : m^2 - n^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1168_116817


namespace NUMINAMATH_CALUDE_binary_representation_theorem_l1168_116857

def is_multiple_of_17 (n : ℕ) : Prop := ∃ k : ℕ, n = 17 * k

def binary_ones_count (n : ℕ) : ℕ := (n.digits 2).count 1

def binary_zeros_count (n : ℕ) : ℕ := (n.digits 2).length - binary_ones_count n

theorem binary_representation_theorem (n : ℕ) 
  (h1 : is_multiple_of_17 n) 
  (h2 : binary_ones_count n = 3) : 
  (binary_zeros_count n ≥ 6) ∧ 
  (binary_zeros_count n = 7 → Even n) := by
sorry

end NUMINAMATH_CALUDE_binary_representation_theorem_l1168_116857


namespace NUMINAMATH_CALUDE_snail_return_time_is_integer_l1168_116833

/-- Represents the snail's position on the plane -/
structure SnailPosition :=
  (x : ℝ) (y : ℝ)

/-- Represents the snail's movement parameters -/
structure SnailMovement :=
  (speed : ℝ)
  (turnAngle : ℝ)
  (turnInterval : ℝ)

/-- Calculates the snail's position after a given time -/
def snailPositionAfterTime (initialPos : SnailPosition) (movement : SnailMovement) (time : ℝ) : SnailPosition :=
  sorry

/-- Checks if the snail has returned to the origin -/
def hasReturnedToOrigin (pos : SnailPosition) : Prop :=
  pos.x = 0 ∧ pos.y = 0

/-- Theorem: The snail can only return to the origin after an integer number of hours -/
theorem snail_return_time_is_integer 
  (movement : SnailMovement) 
  (h1 : movement.speed > 0)
  (h2 : movement.turnAngle = π / 3)
  (h3 : movement.turnInterval = 1 / 2) :
  ∀ t : ℝ, hasReturnedToOrigin (snailPositionAfterTime ⟨0, 0⟩ movement t) → ∃ n : ℕ, t = n :=
sorry

end NUMINAMATH_CALUDE_snail_return_time_is_integer_l1168_116833


namespace NUMINAMATH_CALUDE_semipro_max_salary_l1168_116835

/-- Represents the structure of a baseball team with salary constraints -/
structure BaseballTeam where
  players : ℕ
  minSalary : ℕ
  salaryCap : ℕ

/-- Calculates the maximum possible salary for a single player in a baseball team -/
def maxPlayerSalary (team : BaseballTeam) : ℕ :=
  team.salaryCap - (team.players - 1) * team.minSalary

/-- Theorem stating the maximum possible salary for a single player
    given the specific constraints of the semipro baseball league -/
theorem semipro_max_salary :
  let team : BaseballTeam := ⟨25, 15000, 875000⟩
  maxPlayerSalary team = 515000 := by
  sorry


end NUMINAMATH_CALUDE_semipro_max_salary_l1168_116835


namespace NUMINAMATH_CALUDE_prob_different_numbers_l1168_116876

/-- The number of balls in the bag -/
def num_balls : ℕ := 6

/-- The probability of drawing different numbers -/
def prob_different : ℚ := 5/6

/-- Theorem stating the probability of drawing different numbers -/
theorem prob_different_numbers :
  (num_balls - 1 : ℚ) / num_balls = prob_different :=
sorry

end NUMINAMATH_CALUDE_prob_different_numbers_l1168_116876


namespace NUMINAMATH_CALUDE_weight_change_result_l1168_116861

/-- Calculate the final weight after weight loss and gain -/
def final_weight (initial_weight : ℝ) (loss_percentage : ℝ) (weight_gain : ℝ) : ℝ :=
  initial_weight - (initial_weight * loss_percentage) + weight_gain

/-- Theorem stating that the given weight changes result in a final weight of 200 pounds -/
theorem weight_change_result : 
  final_weight 220 0.1 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_weight_change_result_l1168_116861


namespace NUMINAMATH_CALUDE_yan_distance_ratio_l1168_116875

theorem yan_distance_ratio :
  ∀ (a b v : ℝ),
  a > 0 → b > 0 → v > 0 →
  (b / v = a / v + (a + b) / (7 * v)) →
  (a / b = 3 / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_yan_distance_ratio_l1168_116875


namespace NUMINAMATH_CALUDE_page_lines_increase_l1168_116864

theorem page_lines_increase (original : ℕ) (increased : ℕ) (percentage : ℚ) : 
  percentage = 100/3 →
  increased = 240 →
  increased = original + (percentage / 100 * original).floor →
  increased - original = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_page_lines_increase_l1168_116864


namespace NUMINAMATH_CALUDE_line_m_equation_l1168_116871

-- Define the xy-plane
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a line in the xy-plane
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

-- Define the reflection of a point about a line
def reflect (p : Point) (l : Line) : Point :=
  sorry

-- Define the given conditions
def problem_setup :=
  ∃ (ℓ m : Line) (P P' P'' : Point),
    ℓ ≠ m ∧
    ℓ.a * 0 + ℓ.b * 0 + ℓ.c = 0 ∧
    m.a * 0 + m.b * 0 + m.c = 0 ∧
    ℓ = Line.mk 5 (-1) 0 ∧
    P = Point.mk (-1) 4 ∧
    P'' = Point.mk 4 1 ∧
    P' = reflect P ℓ ∧
    P'' = reflect P' m

-- State the theorem
theorem line_m_equation (h : problem_setup) :
  ∃ (m : Line), m = Line.mk 2 (-3) 0 :=
sorry

end NUMINAMATH_CALUDE_line_m_equation_l1168_116871


namespace NUMINAMATH_CALUDE_smallest_sum_of_three_non_coprime_integers_with_prime_sum_l1168_116837

/-- Two natural numbers are not coprime if their greatest common divisor is greater than 1 -/
def not_coprime (a b : ℕ) : Prop := Nat.gcd a b > 1

/-- A natural number is prime if it's greater than 1 and its only divisors are 1 and itself -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem smallest_sum_of_three_non_coprime_integers_with_prime_sum :
  ∀ a b c : ℕ,
    a > 0 → b > 0 → c > 0 →
    (not_coprime a b ∨ not_coprime b c ∨ not_coprime a c) →
    is_prime (a + b + c) →
    ∀ x y z : ℕ,
      x > 0 → y > 0 → z > 0 →
      (not_coprime x y ∨ not_coprime y z ∨ not_coprime x z) →
      is_prime (x + y + z) →
      a + b + c ≤ x + y + z →
      a + b + c = 31 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_three_non_coprime_integers_with_prime_sum_l1168_116837


namespace NUMINAMATH_CALUDE_largest_number_l1168_116813

theorem largest_number (a b c d e : ℝ) : 
  a = 0.9891 → b = 0.9799 → c = 0.989 → d = 0.978 → e = 0.979 →
  (a ≥ b ∧ a ≥ c ∧ a ≥ d ∧ a ≥ e) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l1168_116813


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1168_116894

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a * b * c + a * b * d + a * c * d + b * c * d) / 4 ≤ 
    ((a * b + a * c + a * d + b * c + b * d + c * d) / 6) ^ (3/2) ∧
  ((a * b * c + a * b * d + a * c * d + b * c * d) / 4 = 
    ((a * b + a * c + a * d + b * c + b * d + c * d) / 6) ^ (3/2) ↔ 
      a = b ∧ b = c ∧ c = d) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1168_116894


namespace NUMINAMATH_CALUDE_log_base_2_derivative_l1168_116841

theorem log_base_2_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) :=
sorry

end NUMINAMATH_CALUDE_log_base_2_derivative_l1168_116841


namespace NUMINAMATH_CALUDE_exp_inequality_equivalence_l1168_116873

theorem exp_inequality_equivalence (x : ℝ) : 1 < Real.exp x ∧ Real.exp x < 2 ↔ 0 < x ∧ x < Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_exp_inequality_equivalence_l1168_116873


namespace NUMINAMATH_CALUDE_specific_hexagon_area_l1168_116800

/-- A hexagon formed by attaching six isosceles triangles to a central rectangle -/
structure Hexagon where
  /-- The base length of each isosceles triangle -/
  triangle_base : ℝ
  /-- The height of each isosceles triangle -/
  triangle_height : ℝ
  /-- The length of the central rectangle -/
  rectangle_length : ℝ
  /-- The width of the central rectangle -/
  rectangle_width : ℝ

/-- Calculate the area of the hexagon -/
def hexagon_area (h : Hexagon) : ℝ :=
  6 * (0.5 * h.triangle_base * h.triangle_height) + h.rectangle_length * h.rectangle_width

/-- Theorem stating that the area of the specific hexagon is 20 square units -/
theorem specific_hexagon_area :
  let h : Hexagon := {
    triangle_base := 2,
    triangle_height := 2,
    rectangle_length := 4,
    rectangle_width := 2
  }
  hexagon_area h = 20 := by sorry

end NUMINAMATH_CALUDE_specific_hexagon_area_l1168_116800


namespace NUMINAMATH_CALUDE_gcd_of_256_162_450_l1168_116826

theorem gcd_of_256_162_450 : Nat.gcd 256 (Nat.gcd 162 450) = 2 := by sorry

end NUMINAMATH_CALUDE_gcd_of_256_162_450_l1168_116826


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1168_116816

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 2) :
  5 * x / ((x - 4) * (x - 2)^2) = 5 / (x - 4) + (-5) / (x - 2) + (-5) / (x - 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1168_116816


namespace NUMINAMATH_CALUDE_parabola_properties_l1168_116866

/-- A parabola with the given properties -/
structure Parabola where
  a : ℝ
  b : ℝ
  intersectsXAxis : (a * (-1)^2 + b * (-1) + 2 = 0) ∧ (a ≠ 0)
  distanceAB : ∃ x, x ≠ -1 ∧ a * x^2 + b * x + 2 = 0 ∧ |x - (-1)| = 3
  increasingAfterA : ∀ x > -1, ∀ y > -1, 
    a * x^2 + b * x + 2 > a * y^2 + b * y + 2 → x > y

/-- The axis of symmetry and point P for the parabola -/
theorem parabola_properties (p : Parabola) :
  (∃ x, x = -(p.a + 2) / (2 * p.a) ∧ 
    ∀ y, p.a * (x + y)^2 + p.b * (x + y) + 2 = p.a * (x - y)^2 + p.b * (x - y) + 2) ∧
  (∃ x y, (x = -3 ∨ x = -2) ∧ y = -1 ∧ 
    p.a * x^2 + p.b * x + 2 = y ∧ 
    y < 0 ∧
    ∃ xB yC, p.a * xB^2 + p.b * xB + 2 = 0 ∧ yC = 2 ∧
    2 * (yC - y) = xB - x) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l1168_116866


namespace NUMINAMATH_CALUDE_f_inequality_l1168_116877

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem f_inequality (x : ℝ) (h : 0 < x ∧ x < 1) : f x < f (x^2) ∧ f (x^2) < (f x)^2 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l1168_116877


namespace NUMINAMATH_CALUDE_fraction_equality_l1168_116802

theorem fraction_equality (a b : ℝ) (h : a ≠ b) :
  (a^2 - b^2) / (a - b)^2 = (a + b) / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1168_116802


namespace NUMINAMATH_CALUDE_triangle_area_difference_l1168_116878

/-- Given a square with side length 10 meters, divided by three straight line segments,
    where P and Q are the areas of two triangles formed by these segments, 
    prove that P - Q = 0 -/
theorem triangle_area_difference (P Q : ℝ) : 
  (∃ R : ℝ, P + R = 50 ∧ Q + R = 50) → P - Q = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_difference_l1168_116878


namespace NUMINAMATH_CALUDE_det_cyclic_matrix_cubic_roots_l1168_116828

/-- Given a cubic equation x³ - 2x² + px + q = 0 with roots a, b, and c,
    the determinant of the matrix [[a,b,c],[b,c,a],[c,a,b]] is -p - 8 -/
theorem det_cyclic_matrix_cubic_roots (p q : ℝ) (a b c : ℝ) 
    (h₁ : a^3 - 2*a^2 + p*a + q = 0)
    (h₂ : b^3 - 2*b^2 + p*b + q = 0)
    (h₃ : c^3 - 2*c^2 + p*c + q = 0) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![a,b,c; b,c,a; c,a,b]
  Matrix.det M = -p - 8 := by
sorry

end NUMINAMATH_CALUDE_det_cyclic_matrix_cubic_roots_l1168_116828


namespace NUMINAMATH_CALUDE_parabola_focus_l1168_116880

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = -4*y

-- Define the focus of a parabola
def focus (p : ℝ × ℝ) (parabola : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), p = (a, b) ∧
  ∀ (x y : ℝ), parabola x y → (x - a)^2 + (y - b)^2 = (y - b + 1/4)^2

-- Theorem statement
theorem parabola_focus :
  focus (0, -1) parabola :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l1168_116880


namespace NUMINAMATH_CALUDE_total_interest_earned_l1168_116891

/-- Calculates the total interest earned from two investments --/
theorem total_interest_earned
  (amount1 : ℝ)  -- Amount invested in the first account
  (amount2 : ℝ)  -- Amount invested in the second account
  (rate1 : ℝ)    -- Interest rate for the first account
  (rate2 : ℝ)    -- Interest rate for the second account
  (h1 : amount2 = amount1 + 800)  -- Second account has $800 more
  (h2 : amount1 + amount2 = 2000) -- Total investment is $2000
  (h3 : rate1 = 0.02)  -- 2% interest rate for first account
  (h4 : rate2 = 0.04)  -- 4% interest rate for second account
  : amount1 * rate1 + amount2 * rate2 = 68 := by
  sorry


end NUMINAMATH_CALUDE_total_interest_earned_l1168_116891


namespace NUMINAMATH_CALUDE_arithmetic_operation_proof_l1168_116848

theorem arithmetic_operation_proof : 65 + 5 * 12 / (180 / 3) = 66 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operation_proof_l1168_116848


namespace NUMINAMATH_CALUDE_first_play_duration_is_20_l1168_116814

/-- Represents the duration of a soccer game in minutes -/
def game_duration : ℕ := 90

/-- Represents the duration of the second part of play in minutes -/
def second_play_duration : ℕ := 35

/-- Represents the duration of sideline time in minutes -/
def sideline_duration : ℕ := 35

/-- Calculates the duration of the first part of play given the total game duration,
    second part play duration, and sideline duration -/
def first_play_duration (total : ℕ) (second : ℕ) (sideline : ℕ) : ℕ :=
  total - second - sideline

theorem first_play_duration_is_20 :
  first_play_duration game_duration second_play_duration sideline_duration = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_play_duration_is_20_l1168_116814


namespace NUMINAMATH_CALUDE_ravi_overall_profit_l1168_116819

/-- Calculates the overall profit or loss for Ravi's sales -/
theorem ravi_overall_profit (refrigerator_cost mobile_cost : ℕ)
  (refrigerator_loss_percent mobile_profit_percent : ℚ) :
  refrigerator_cost = 15000 →
  mobile_cost = 8000 →
  refrigerator_loss_percent = 4 / 100 →
  mobile_profit_percent = 10 / 100 →
  (refrigerator_cost * (1 - refrigerator_loss_percent) +
   mobile_cost * (1 + mobile_profit_percent) -
   (refrigerator_cost + mobile_cost) : ℚ) = 200 := by
  sorry

end NUMINAMATH_CALUDE_ravi_overall_profit_l1168_116819


namespace NUMINAMATH_CALUDE_basketball_free_throws_l1168_116822

theorem basketball_free_throws (two_point_shots three_point_shots free_throws : ℕ) : 
  (2 * two_point_shots = 3 * three_point_shots) →
  (free_throws = two_point_shots + 1) →
  (2 * two_point_shots + 3 * three_point_shots + free_throws = 61) →
  free_throws = 13 := by
  sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l1168_116822


namespace NUMINAMATH_CALUDE_amy_small_gardens_l1168_116803

def small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  (total_seeds - big_garden_seeds) / seeds_per_small_garden

theorem amy_small_gardens :
  small_gardens 101 47 6 = 9 :=
by sorry

end NUMINAMATH_CALUDE_amy_small_gardens_l1168_116803


namespace NUMINAMATH_CALUDE_train_speed_l1168_116860

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 350) (h2 : time = 7) :
  length / time = 50 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1168_116860


namespace NUMINAMATH_CALUDE_vector_decomposition_l1168_116832

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![0, -8, 9]
def p : Fin 3 → ℝ := ![0, -2, 1]
def q : Fin 3 → ℝ := ![3, 1, -1]
def r : Fin 3 → ℝ := ![4, 0, 1]

/-- Theorem stating that x can be expressed as a linear combination of p, q, and r -/
theorem vector_decomposition :
  x = (2 : ℝ) • p + (-4 : ℝ) • q + (3 : ℝ) • r :=
by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l1168_116832


namespace NUMINAMATH_CALUDE_convention_handshakes_specific_l1168_116812

/-- The number of handshakes in a convention with multiple companies --/
def convention_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_people := num_companies * reps_per_company
  let handshakes_per_person := total_people - reps_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem stating the number of handshakes for the specific convention described --/
theorem convention_handshakes_specific : convention_handshakes 5 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_convention_handshakes_specific_l1168_116812


namespace NUMINAMATH_CALUDE_pollution_filtration_time_l1168_116888

/-- Given a pollution filtration process where:
    1. The relationship between pollutants (P mg/L) and time (t h) is given by P = P₀e^(-kt)
    2. 10% of pollutants were removed in the first 5 hours
    
    This theorem proves that the time required to remove 27.1% of pollutants is 15 hours. -/
theorem pollution_filtration_time (P₀ k : ℝ) (h1 : P₀ > 0) (h2 : k > 0) : 
  (∃ t : ℝ, t > 0 ∧ P₀ * Real.exp (-k * 5) = 0.9 * P₀) → 
  (∃ t : ℝ, t > 0 ∧ P₀ * Real.exp (-k * t) = 0.271 * P₀ ∧ t = 15) :=
by sorry


end NUMINAMATH_CALUDE_pollution_filtration_time_l1168_116888


namespace NUMINAMATH_CALUDE_daily_wage_c_value_l1168_116830

/-- The daily wage of worker c given the conditions of the problem -/
def daily_wage_c (days_a days_b days_c : ℕ) 
                 (wage_ratio_a wage_ratio_b wage_ratio_c : ℕ) 
                 (total_earning : ℚ) : ℚ :=
  let wage_a := total_earning * wage_ratio_a / 
    (days_a * wage_ratio_a + days_b * wage_ratio_b + days_c * wage_ratio_c)
  wage_a * wage_ratio_c / wage_ratio_a

theorem daily_wage_c_value : 
  daily_wage_c 6 9 4 3 4 5 1480 = 100 / 3 := by
  sorry

#eval daily_wage_c 6 9 4 3 4 5 1480

end NUMINAMATH_CALUDE_daily_wage_c_value_l1168_116830


namespace NUMINAMATH_CALUDE_fraction_sum_integer_implies_not_divisible_by_three_l1168_116897

theorem fraction_sum_integer_implies_not_divisible_by_three (n : ℕ+) 
  (h : ∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / n.val = k) : 
  ¬(3 ∣ n.val) := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_integer_implies_not_divisible_by_three_l1168_116897


namespace NUMINAMATH_CALUDE_circle_area_approximation_l1168_116823

/-- The area of a circle with radius 0.6 meters is 1.08 square meters when pi is approximated as 3 -/
theorem circle_area_approximation (r : ℝ) (π : ℝ) (A : ℝ) : 
  r = 0.6 → π = 3 → A = π * r^2 → A = 1.08 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_approximation_l1168_116823


namespace NUMINAMATH_CALUDE_diameter_endpoint_theorem_l1168_116839

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ

/-- A diameter of a circle --/
structure Diameter where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- Theorem: Given a circle with center at (0,0) and one endpoint of a diameter at (3,4),
    the other endpoint of the diameter is at (-3, -4) --/
theorem diameter_endpoint_theorem (c : Circle) (d : Diameter) :
  c.center = (0, 0) ∧ d.endpoint1 = (3, 4) →
  d.endpoint2 = (-3, -4) := by
  sorry

end NUMINAMATH_CALUDE_diameter_endpoint_theorem_l1168_116839


namespace NUMINAMATH_CALUDE_sin_75_degrees_l1168_116810

theorem sin_75_degrees : 
  let sin75 := Real.sin (75 * Real.pi / 180)
  let sin45 := Real.sin (45 * Real.pi / 180)
  let cos45 := Real.cos (45 * Real.pi / 180)
  let sin30 := Real.sin (30 * Real.pi / 180)
  let cos30 := Real.cos (30 * Real.pi / 180)
  sin75 = (Real.sqrt 6 + Real.sqrt 2) / 4 ∧
  sin45 = Real.sqrt 2 / 2 ∧
  cos45 = Real.sqrt 2 / 2 ∧
  sin30 = 1 / 2 ∧
  cos30 = Real.sqrt 3 / 2 ∧
  sin75 = sin45 * cos30 + cos45 * sin30 :=
by sorry


end NUMINAMATH_CALUDE_sin_75_degrees_l1168_116810


namespace NUMINAMATH_CALUDE_jellybean_count_l1168_116885

/-- The number of blue jellybeans in the jar -/
def blue_jellybeans : ℕ := 14

/-- The number of purple jellybeans in the jar -/
def purple_jellybeans : ℕ := 26

/-- The number of orange jellybeans in the jar -/
def orange_jellybeans : ℕ := 40

/-- The number of red jellybeans in the jar -/
def red_jellybeans : ℕ := 120

/-- The total number of jellybeans in the jar -/
def total_jellybeans : ℕ := blue_jellybeans + purple_jellybeans + orange_jellybeans + red_jellybeans

theorem jellybean_count : total_jellybeans = 200 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_l1168_116885


namespace NUMINAMATH_CALUDE_price_is_four_l1168_116809

/-- The price per bag of leaves for Bob and Johnny's leaf raking business -/
def price_per_bag (monday_bags : ℕ) (tuesday_bags : ℕ) (wednesday_bags : ℕ) (total_earnings : ℕ) : ℚ :=
  total_earnings / (monday_bags + tuesday_bags + wednesday_bags)

/-- Theorem stating that the price per bag is $4 given the conditions -/
theorem price_is_four :
  price_per_bag 5 3 9 68 = 4 := by
  sorry

end NUMINAMATH_CALUDE_price_is_four_l1168_116809


namespace NUMINAMATH_CALUDE_positive_x_squared_1024_l1168_116855

theorem positive_x_squared_1024 (x : ℝ) (h1 : x > 0) (h2 : 4 * x^2 = 1024) : x = 16 := by
  sorry

end NUMINAMATH_CALUDE_positive_x_squared_1024_l1168_116855


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l1168_116892

def a : Fin 2 → ℝ := ![4, 3]
def b : Fin 2 → ℝ := ![-1, 2]

theorem perpendicular_vectors_k_value :
  ∃ k : ℝ, (∀ i : Fin 2, (a + k • b) i * (a - b) i = 0) → k = 23/3 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l1168_116892


namespace NUMINAMATH_CALUDE_boy_bike_rest_time_l1168_116849

theorem boy_bike_rest_time 
  (total_distance : ℝ) 
  (outbound_speed inbound_speed : ℝ) 
  (total_time : ℝ) :
  total_distance = 15 →
  outbound_speed = 5 →
  inbound_speed = 3 →
  total_time = 6 →
  (total_distance / 2) / outbound_speed + 
  (total_distance / 2) / inbound_speed + 
  (total_time - (total_distance / 2) / outbound_speed - (total_distance / 2) / inbound_speed) = 2 :=
by sorry

end NUMINAMATH_CALUDE_boy_bike_rest_time_l1168_116849


namespace NUMINAMATH_CALUDE_vehicle_overtake_problem_l1168_116884

/-- The initial distance between two vehicles, where one overtakes the other --/
def initial_distance (speed_x speed_y : ℝ) (time : ℝ) (final_distance : ℝ) : ℝ :=
  (speed_y - speed_x) * time - final_distance

theorem vehicle_overtake_problem :
  let speed_x : ℝ := 36
  let speed_y : ℝ := 45
  let time : ℝ := 5
  let final_distance : ℝ := 23
  initial_distance speed_x speed_y time final_distance = 22 := by
  sorry

end NUMINAMATH_CALUDE_vehicle_overtake_problem_l1168_116884


namespace NUMINAMATH_CALUDE_minimum_shoeing_time_l1168_116867

theorem minimum_shoeing_time 
  (blacksmiths : ℕ) 
  (horses : ℕ) 
  (time_per_shoe : ℕ) 
  (h1 : blacksmiths = 48) 
  (h2 : horses = 60) 
  (h3 : time_per_shoe = 5) : 
  (horses * 4 * time_per_shoe) / blacksmiths = 25 := by
  sorry

end NUMINAMATH_CALUDE_minimum_shoeing_time_l1168_116867


namespace NUMINAMATH_CALUDE_worker_assignment_proof_l1168_116811

/-- The number of shifts -/
def num_shifts : ℕ := 5

/-- The number of workers per shift -/
def workers_per_shift : ℕ := 2

/-- The total number of ways to assign workers -/
def total_assignments : ℕ := 45

/-- The total number of new workers -/
def total_workers : ℕ := 15

/-- Theorem: The number of ways to choose 2 workers from 15 workers is equal to 45 -/
theorem worker_assignment_proof :
  Nat.choose total_workers workers_per_shift = total_assignments :=
by sorry

end NUMINAMATH_CALUDE_worker_assignment_proof_l1168_116811


namespace NUMINAMATH_CALUDE_f_properties_l1168_116815

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2 * |x|

-- Theorem for the properties of f
theorem f_properties :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 1 < x → x < y → f x < f y) ∧
  ({a : ℝ | f (|a| + 3/2) > 0} = {a : ℝ | a > 1/2 ∨ a < -1/2}) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1168_116815


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l1168_116853

/-- Given a quadratic equation x^2 - 6x + 5 = 0, when rewritten in the form (x + b)^2 = c
    where b and c are integers, prove that b + c = 11 -/
theorem quadratic_complete_square (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 6*x + 5 = 0 ↔ (x + b)^2 = c) → b + c = 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l1168_116853


namespace NUMINAMATH_CALUDE_shelter_ratio_l1168_116895

/-- 
Given a shelter with dogs and cats, prove that if there are 75 dogs, 
and adding 20 cats would make the ratio of dogs to cats 15:11, 
then the initial ratio of dogs to cats is 15:7.
-/
theorem shelter_ratio (initial_cats : ℕ) : 
  (75 : ℚ) / (initial_cats + 20) = 15 / 11 → 
  75 / initial_cats = 15 / 7 := by
sorry

end NUMINAMATH_CALUDE_shelter_ratio_l1168_116895


namespace NUMINAMATH_CALUDE_towel_area_decrease_l1168_116846

theorem towel_area_decrease (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let new_length := 0.7 * L
  let new_breadth := 0.75 * B
  let original_area := L * B
  let new_area := new_length * new_breadth
  (original_area - new_area) / original_area = 0.475
  := by sorry

end NUMINAMATH_CALUDE_towel_area_decrease_l1168_116846


namespace NUMINAMATH_CALUDE_ln_inequality_implies_p_range_l1168_116821

theorem ln_inequality_implies_p_range (p : ℝ) : 
  (∀ x : ℝ, x > 0 → Real.log x ≤ p * x - 1) → p ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ln_inequality_implies_p_range_l1168_116821


namespace NUMINAMATH_CALUDE_train_distance_theorem_l1168_116836

/-- The distance a train can travel given its fuel efficiency and remaining coal -/
def train_distance (miles_per_coal : ℚ) (remaining_coal : ℚ) : ℚ :=
  miles_per_coal * remaining_coal

/-- Theorem: A train traveling 5 miles for every 2 pounds of coal with 160 pounds remaining can travel 400 miles -/
theorem train_distance_theorem :
  let miles_per_coal : ℚ := 5 / 2
  let remaining_coal : ℚ := 160
  train_distance miles_per_coal remaining_coal = 400 := by
sorry

end NUMINAMATH_CALUDE_train_distance_theorem_l1168_116836


namespace NUMINAMATH_CALUDE_quadratic_polynomial_inequality_l1168_116845

/-- A quadratic polynomial with non-negative coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonneg : 0 ≤ a
  b_nonneg : 0 ≤ b
  c_nonneg : 0 ≤ c

/-- The value of a quadratic polynomial at a given point -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Theorem: For any quadratic polynomial with non-negative coefficients and any real numbers x and y,
    the square of the polynomial evaluated at xy is less than or equal to 
    the product of the polynomial evaluated at x^2 and y^2 -/
theorem quadratic_polynomial_inequality (p : QuadraticPolynomial) (x y : ℝ) :
  (p.eval (x * y))^2 ≤ (p.eval (x^2)) * (p.eval (y^2)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_inequality_l1168_116845


namespace NUMINAMATH_CALUDE_total_heads_eq_97_l1168_116882

/-- Represents the number of Lumix aliens -/
def l : ℕ := 23

/-- Represents the number of Obscra aliens -/
def o : ℕ := 37

/-- The total number of aliens -/
def total_aliens : ℕ := 60

/-- The total number of legs -/
def total_legs : ℕ := 129

/-- Lumix aliens have 1 head and 4 legs -/
axiom lumix_anatomy : l * 1 + l * 4 = l + 4 * l

/-- Obscra aliens have 2 heads and 1 leg -/
axiom obscra_anatomy : o * 2 + o * 1 = 2 * o + o

/-- The total number of aliens is 60 -/
axiom total_aliens_eq : l + o = total_aliens

/-- The total number of legs is 129 -/
axiom total_legs_eq : 4 * l + o = total_legs

/-- The theorem to be proved -/
theorem total_heads_eq_97 : l + 2 * o = 97 := by
  sorry

end NUMINAMATH_CALUDE_total_heads_eq_97_l1168_116882


namespace NUMINAMATH_CALUDE_shaded_area_rectangle_with_circles_l1168_116834

/-- Given a rectangle with width 30 inches and length 60 inches, and four identical circles
    each tangent to two adjacent sides of the rectangle and its neighboring circles,
    the total shaded area when the circles are excluded is 1800 - 225π square inches. -/
theorem shaded_area_rectangle_with_circles :
  let rectangle_width : ℝ := 30
  let rectangle_length : ℝ := 60
  let circle_radius : ℝ := rectangle_width / 4
  let rectangle_area : ℝ := rectangle_width * rectangle_length
  let circle_area : ℝ := π * circle_radius^2
  let total_circle_area : ℝ := 4 * circle_area
  let shaded_area : ℝ := rectangle_area - total_circle_area
  shaded_area = 1800 - 225 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_rectangle_with_circles_l1168_116834


namespace NUMINAMATH_CALUDE_roof_length_width_difference_roof_area_is_720_length_is_5_times_width_l1168_116801

/-- Represents the dimensions of a rectangular roof -/
structure RoofDimensions where
  width : ℝ
  length : ℝ

/-- The roof of an apartment building -/
def apartmentRoof : RoofDimensions where
  width := (720 / 5).sqrt
  length := 5 * (720 / 5).sqrt

theorem roof_length_width_difference : 
  apartmentRoof.length - apartmentRoof.width = 48 := by
  sorry

/-- The area of the roof -/
def roofArea (roof : RoofDimensions) : ℝ :=
  roof.length * roof.width

theorem roof_area_is_720 : roofArea apartmentRoof = 720 := by
  sorry

theorem length_is_5_times_width : 
  apartmentRoof.length = 5 * apartmentRoof.width := by
  sorry

end NUMINAMATH_CALUDE_roof_length_width_difference_roof_area_is_720_length_is_5_times_width_l1168_116801


namespace NUMINAMATH_CALUDE_same_number_probability_l1168_116862

def max_number : ℕ := 250
def billy_multiple : ℕ := 20
def bobbi_multiple : ℕ := 30

theorem same_number_probability :
  let billy_choices := (max_number - 1) / billy_multiple
  let bobbi_choices := (max_number - 1) / bobbi_multiple
  let common_choices := (max_number - 1) / (lcm billy_multiple bobbi_multiple)
  (common_choices : ℚ) / (billy_choices * bobbi_choices) = 1 / 24 := by
sorry

end NUMINAMATH_CALUDE_same_number_probability_l1168_116862


namespace NUMINAMATH_CALUDE_two_digit_integer_problem_l1168_116824

theorem two_digit_integer_problem :
  ∃ (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧  -- a is a 2-digit positive integer
    10 ≤ b ∧ b < 100 ∧  -- b is a 2-digit positive integer
    a ≠ b ∧             -- a and b are different
    (a + b) / 2 = a + b / 100 ∧  -- average equals the special number
    a < b ∧             -- a is smaller than b
    a = 49 :=           -- a is 49
by sorry

end NUMINAMATH_CALUDE_two_digit_integer_problem_l1168_116824


namespace NUMINAMATH_CALUDE_triangle_similarity_condition_l1168_116872

/-- Two triangles with side lengths a, b, c and a₁, b₁, c₁ are similar if and only if
    √(a·a₁) + √(b·b₁) + √(c·c₁) = √((a+b+c)·(a₁+b₁+c₁)) -/
theorem triangle_similarity_condition 
  (a b c a₁ b₁ c₁ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (ha₁ : a₁ > 0) (hb₁ : b₁ > 0) (hc₁ : c₁ > 0) :
  (∃ (k : ℝ), k > 0 ∧ a₁ = k * a ∧ b₁ = k * b ∧ c₁ = k * c) ↔ 
  Real.sqrt (a * a₁) + Real.sqrt (b * b₁) + Real.sqrt (c * c₁) = 
  Real.sqrt ((a + b + c) * (a₁ + b₁ + c₁)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_similarity_condition_l1168_116872


namespace NUMINAMATH_CALUDE_smallest_z_value_l1168_116870

/-- Given consecutive positive integers w, x, y, z where w = n and z = w + 4,
    the smallest z satisfying w^3 + x^3 + y^3 = z^3 is 9. -/
theorem smallest_z_value (n : ℕ) (w x y z : ℕ) : 
  w = n → 
  x = n + 1 → 
  y = n + 2 → 
  z = n + 4 → 
  w^3 + x^3 + y^3 = z^3 → 
  z ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_z_value_l1168_116870


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1168_116852

theorem diophantine_equation_solutions (n : ℕ) :
  let solutions := {(a, b, c, d) : ℕ × ℕ × ℕ × ℕ | a^2 + b^2 + c^2 + d^2 = 7 * 4^n}
  solutions = {(5 * 2^(n-1), 2^(n-1), 2^(n-1), 2^(n-1)),
               (2^(n+1), 2^n, 2^n, 2^n),
               (3 * 2^(n-1), 3 * 2^(n-1), 3 * 2^(n-1), 2^(n-1))} :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1168_116852


namespace NUMINAMATH_CALUDE_decagon_triangles_l1168_116838

theorem decagon_triangles : ∀ (n : ℕ), n = 10 → (n.choose 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangles_l1168_116838


namespace NUMINAMATH_CALUDE_map_scale_conversion_l1168_116890

theorem map_scale_conversion (map_cm : ℝ) (real_km : ℝ) : 
  (20 : ℝ) * real_km = 100 * map_cm → 25 * real_km = 125 * map_cm := by
  sorry

end NUMINAMATH_CALUDE_map_scale_conversion_l1168_116890


namespace NUMINAMATH_CALUDE_fibonacci_geometric_sequence_sum_l1168_116818

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

def isGeometricSequence (a b c : ℕ) : Prop :=
  (fib b) ^ 2 = (fib a) * (fib c)

theorem fibonacci_geometric_sequence_sum (a b c : ℕ) :
  isGeometricSequence a b c ∧ 
  fib a ≤ fib b ∧ 
  fib b ≤ fib c ∧ 
  a + b + c = 1500 → 
  a = 499 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_geometric_sequence_sum_l1168_116818


namespace NUMINAMATH_CALUDE_mortezas_wish_impossible_l1168_116829

theorem mortezas_wish_impossible :
  ¬ ∃ (x₁ x₂ x₃ x₄ x₅ x₆ S P : ℝ),
    (x₁ ≠ x₂) ∧ (x₁ ≠ x₃) ∧ (x₁ ≠ x₄) ∧ (x₁ ≠ x₅) ∧ (x₁ ≠ x₆) ∧
    (x₂ ≠ x₃) ∧ (x₂ ≠ x₄) ∧ (x₂ ≠ x₅) ∧ (x₂ ≠ x₆) ∧
    (x₃ ≠ x₄) ∧ (x₃ ≠ x₅) ∧ (x₃ ≠ x₆) ∧
    (x₄ ≠ x₅) ∧ (x₄ ≠ x₆) ∧
    (x₅ ≠ x₆) ∧
    ((x₁ + x₂ + x₃ = S) ∨ (x₁ * x₂ * x₃ = P)) ∧
    ((x₂ + x₃ + x₄ = S) ∨ (x₂ * x₃ * x₄ = P)) ∧
    ((x₃ + x₄ + x₅ = S) ∨ (x₃ * x₄ * x₅ = P)) ∧
    ((x₄ + x₅ + x₆ = S) ∨ (x₄ * x₅ * x₆ = P)) ∧
    ((x₅ + x₆ + x₁ = S) ∨ (x₅ * x₆ * x₁ = P)) ∧
    ((x₆ + x₁ + x₂ = S) ∨ (x₆ * x₁ * x₂ = P)) :=
by sorry

end NUMINAMATH_CALUDE_mortezas_wish_impossible_l1168_116829


namespace NUMINAMATH_CALUDE_scalene_triangle_with_double_angle_and_36_degree_l1168_116887

def is_valid_triangle (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 180

def is_scalene (a b c : ℝ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem scalene_triangle_with_double_angle_and_36_degree :
  ∀ a b c : ℝ,
  is_valid_triangle a b c →
  is_scalene a b c →
  ((a = 2 * b ∨ b = 2 * a ∨ a = 2 * c ∨ c = 2 * a ∨ b = 2 * c ∨ c = 2 * b) ∧
   (a = 36 ∨ b = 36 ∨ c = 36)) →
  ((a = 36 ∧ b = 48 ∧ c = 96) ∨ (a = 18 ∧ b = 36 ∧ c = 126) ∨
   (a = 48 ∧ b = 96 ∧ c = 36) ∨ (a = 36 ∧ b = 126 ∧ c = 18) ∨
   (a = 96 ∧ b = 36 ∧ c = 48) ∨ (a = 126 ∧ b = 18 ∧ c = 36)) :=
by sorry

end NUMINAMATH_CALUDE_scalene_triangle_with_double_angle_and_36_degree_l1168_116887


namespace NUMINAMATH_CALUDE_expression_evaluation_l1168_116831

theorem expression_evaluation (a b : ℝ) 
  (h : |a + 1| + (b - 2)^2 = 0) : 
  2 * (3 * a^2 - a * b + 1) - (-a^2 + 2 * a * b + 1) = 16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1168_116831


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1168_116893

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2015) + 2015
  f 2015 = 2016 ∧ ∃ (x : ℝ), f x = x := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1168_116893


namespace NUMINAMATH_CALUDE_special_polygon_exists_l1168_116881

/-- A polygon with the specified properties --/
structure SpecialPolygon where
  vertices : Finset (ℝ × ℝ)
  inside_square : ∀ (v : ℝ × ℝ), v ∈ vertices → v.1 ∈ [-1, 1] ∧ v.2 ∈ [-1, 1]
  side_count : vertices.card = 12
  side_length : ∀ (v w : ℝ × ℝ), v ∈ vertices → w ∈ vertices → v ≠ w →
    Real.sqrt ((v.1 - w.1)^2 + (v.2 - w.2)^2) = 1
  angle_multiples : ∀ (u v w : ℝ × ℝ), u ∈ vertices → v ∈ vertices → w ∈ vertices →
    u ≠ v → v ≠ w → u ≠ w →
    ∃ (n : ℕ), Real.cos (n * (Real.pi / 4)) = 
      ((u.1 - v.1) * (w.1 - v.1) + (u.2 - v.2) * (w.2 - v.2)) /
      (Real.sqrt ((u.1 - v.1)^2 + (u.2 - v.2)^2) * Real.sqrt ((w.1 - v.1)^2 + (w.2 - v.2)^2))

/-- The main theorem stating the existence of the special polygon --/
theorem special_polygon_exists : ∃ (p : SpecialPolygon), True := by
  sorry


end NUMINAMATH_CALUDE_special_polygon_exists_l1168_116881


namespace NUMINAMATH_CALUDE_max_ab_line_circle_intersection_l1168_116805

/-- Given a line ax + by - 8 = 0 (where a > 0 and b > 0) intersecting the circle x² + y² - 2x - 4y = 0
    with a chord length of 2√5, the maximum value of ab is 8. -/
theorem max_ab_line_circle_intersection (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, a * x + b * y - 8 = 0 → x^2 + y^2 - 2*x - 4*y = 0) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    a * x₁ + b * y₁ - 8 = 0 ∧ 
    a * x₂ + b * y₂ - 8 = 0 ∧
    x₁^2 + y₁^2 - 2*x₁ - 4*y₁ = 0 ∧
    x₂^2 + y₂^2 - 2*x₂ - 4*y₂ = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 20) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' * b' ≤ a * b) →
  a * b = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_ab_line_circle_intersection_l1168_116805


namespace NUMINAMATH_CALUDE_total_buttons_eq_1600_l1168_116807

/-- The number of 3-button shirts ordered -/
def shirts_3_button : ℕ := 200

/-- The number of 5-button shirts ordered -/
def shirts_5_button : ℕ := 200

/-- The number of buttons on a 3-button shirt -/
def buttons_per_3_button_shirt : ℕ := 3

/-- The number of buttons on a 5-button shirt -/
def buttons_per_5_button_shirt : ℕ := 5

/-- The total number of buttons used for the order -/
def total_buttons : ℕ := shirts_3_button * buttons_per_3_button_shirt + shirts_5_button * buttons_per_5_button_shirt

theorem total_buttons_eq_1600 : total_buttons = 1600 := by
  sorry

end NUMINAMATH_CALUDE_total_buttons_eq_1600_l1168_116807


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l1168_116850

theorem triangle_side_calculation (A B C : Real) (a b c : Real) :
  A = 45 * π / 180 →
  B = 60 * π / 180 →
  a = Real.sqrt 2 →
  b = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l1168_116850


namespace NUMINAMATH_CALUDE_heavy_wash_water_usage_l1168_116825

/-- Represents the amount of water used for different types of washes -/
structure WashingMachine where
  heavy_wash : ℚ
  regular_wash : ℚ
  light_wash : ℚ

/-- Calculates the total water usage for a given washing machine and set of loads -/
def total_water_usage (wm : WashingMachine) (heavy_loads bleach_loads : ℕ) : ℚ :=
  wm.heavy_wash * heavy_loads +
  wm.regular_wash * 3 +
  wm.light_wash * (1 + bleach_loads)

/-- Theorem stating that the heavy wash uses 20 gallons of water -/
theorem heavy_wash_water_usage :
  ∃ (wm : WashingMachine),
    wm.regular_wash = 10 ∧
    wm.light_wash = 2 ∧
    total_water_usage wm 2 2 = 76 ∧
    wm.heavy_wash = 20 := by
  sorry

end NUMINAMATH_CALUDE_heavy_wash_water_usage_l1168_116825


namespace NUMINAMATH_CALUDE_min_xy_given_otimes_l1168_116847

/-- The custom operation ⊗ defined for positive real numbers -/
def otimes (a b : ℝ) : ℝ := a * b - a - b

/-- Theorem stating the minimum value of xy given the conditions -/
theorem min_xy_given_otimes (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : otimes x y = 3) :
  ∀ z w : ℝ, z > 0 → w > 0 → otimes z w = 3 → x * y ≤ z * w :=
sorry

end NUMINAMATH_CALUDE_min_xy_given_otimes_l1168_116847


namespace NUMINAMATH_CALUDE_treys_chores_l1168_116874

theorem treys_chores (task_duration : ℕ) (total_time : ℕ) (shower_tasks : ℕ) (dinner_tasks : ℕ) :
  task_duration = 10 →
  total_time = 120 →
  shower_tasks = 1 →
  dinner_tasks = 4 →
  (total_time / task_duration) - shower_tasks - dinner_tasks = 7 :=
by sorry

end NUMINAMATH_CALUDE_treys_chores_l1168_116874


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l1168_116879

theorem quadratic_rewrite (b : ℝ) (m : ℝ) : 
  b < 0 → 
  (∀ x, x^2 + b*x + 1/6 = (x+m)^2 + 1/18) → 
  b = -2/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l1168_116879


namespace NUMINAMATH_CALUDE_square_plus_difference_of_squares_l1168_116863

theorem square_plus_difference_of_squares (x y : ℝ) : 
  x^2 + (y - x) * (y + x) = y^2 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_difference_of_squares_l1168_116863


namespace NUMINAMATH_CALUDE_at_most_one_equal_area_point_l1168_116842

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A convex quadrilateral in a 2D plane -/
structure ConvexQuadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D
  convex : Bool  -- Assumption that the quadrilateral is convex

/-- Calculate the area of a triangle given its three vertices -/
def triangleArea (p1 p2 p3 : Point2D) : ℝ :=
  sorry

/-- Check if four triangles have equal areas -/
def equalAreaTriangles (p : Point2D) (quad : ConvexQuadrilateral) : Prop :=
  let areaABP := triangleArea quad.A quad.B p
  let areaBCP := triangleArea quad.B quad.C p
  let areaCDP := triangleArea quad.C quad.D p
  let areaDPA := triangleArea quad.D quad.A p
  areaABP = areaBCP ∧ areaBCP = areaCDP ∧ areaCDP = areaDPA

/-- Main theorem: There exists at most one point P that satisfies the equal area condition -/
theorem at_most_one_equal_area_point (quad : ConvexQuadrilateral) :
  ∃! p : Point2D, equalAreaTriangles p quad :=
sorry

end NUMINAMATH_CALUDE_at_most_one_equal_area_point_l1168_116842


namespace NUMINAMATH_CALUDE_sixth_side_formula_l1168_116827

/-- A hexagon described around a circle with six sides -/
structure CircumscribedHexagon where
  sides : Fin 6 → ℝ
  is_positive : ∀ i, sides i > 0

/-- The property that the sum of alternating sides in a circumscribed hexagon is constant -/
def alternating_sum_constant (h : CircumscribedHexagon) : Prop :=
  h.sides 0 + h.sides 2 + h.sides 4 = h.sides 1 + h.sides 3 + h.sides 5

theorem sixth_side_formula (h : CircumscribedHexagon) 
  (sum_constant : alternating_sum_constant h) :
  h.sides 5 = h.sides 0 - h.sides 1 + h.sides 2 - h.sides 3 + h.sides 4 := by
  sorry

end NUMINAMATH_CALUDE_sixth_side_formula_l1168_116827


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l1168_116859

-- Define the structure of a three-digit number
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  h_hundreds : hundreds < 10
  h_tens : tens < 10
  h_units : units < 10

def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

def ThreeDigitNumber.reverse (n : ThreeDigitNumber) : Nat :=
  100 * n.units + 10 * n.tens + n.hundreds

def ThreeDigitNumber.sumOfDigits (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.units

theorem three_digit_number_problem (n : ThreeDigitNumber) : 
  n.toNat = 253 → 
  n.sumOfDigits = 10 ∧ 
  n.tens = n.hundreds + n.units ∧ 
  n.reverse = n.toNat + 99 := by
  sorry

#eval ThreeDigitNumber.toNat ⟨2, 5, 3, by norm_num, by norm_num, by norm_num⟩

end NUMINAMATH_CALUDE_three_digit_number_problem_l1168_116859


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1168_116840

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1168_116840


namespace NUMINAMATH_CALUDE_cosine_midline_l1168_116865

/-- Given a cosine function y = a cos(bx + c) + d where a, b, c, and d are positive constants,
    if the graph oscillates between 5 and 1, then d = 3 -/
theorem cosine_midline (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x, 1 ≤ a * Real.cos (b * x + c) + d ∧ a * Real.cos (b * x + c) + d ≤ 5) →
  d = 3 := by
  sorry

end NUMINAMATH_CALUDE_cosine_midline_l1168_116865


namespace NUMINAMATH_CALUDE_secure_app_theorem_l1168_116804

/-- Represents an online store application -/
structure OnlineStoreApp where
  paymentGateway : Bool
  dataEncryption : Bool
  transitEncryption : Bool
  codeObfuscation : Bool
  rootedDeviceRestriction : Bool
  antivirusAgent : Bool

/-- Defines the security level of an application -/
def securityLevel (app : OnlineStoreApp) : ℕ :=
  (if app.paymentGateway then 1 else 0) +
  (if app.dataEncryption then 1 else 0) +
  (if app.transitEncryption then 1 else 0) +
  (if app.codeObfuscation then 1 else 0) +
  (if app.rootedDeviceRestriction then 1 else 0) +
  (if app.antivirusAgent then 1 else 0)

/-- Defines a secure application -/
def isSecure (app : OnlineStoreApp) : Prop :=
  securityLevel app = 6

/-- Theorem: An online store app with all security measures implemented is secure -/
theorem secure_app_theorem (app : OnlineStoreApp) 
  (h1 : app.paymentGateway = true)
  (h2 : app.dataEncryption = true)
  (h3 : app.transitEncryption = true)
  (h4 : app.codeObfuscation = true)
  (h5 : app.rootedDeviceRestriction = true)
  (h6 : app.antivirusAgent = true) : 
  isSecure app :=
by
  sorry


end NUMINAMATH_CALUDE_secure_app_theorem_l1168_116804


namespace NUMINAMATH_CALUDE_parabola_vertex_in_fourth_quadrant_l1168_116856

/-- Given a parabola y = 2x^2 + ax - 5 where a < 0, its vertex is in the fourth quadrant -/
theorem parabola_vertex_in_fourth_quadrant (a : ℝ) (ha : a < 0) :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 + a * x - 5
  let vertex_x : ℝ := -a / 4
  let vertex_y : ℝ := f vertex_x
  vertex_x > 0 ∧ vertex_y < 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_in_fourth_quadrant_l1168_116856
