import Mathlib

namespace NUMINAMATH_CALUDE_least_k_squared_divisible_by_240_l3905_390512

theorem least_k_squared_divisible_by_240 : 
  ∃ k : ℕ+, k.val = 60 ∧ 
  (∀ m : ℕ+, m.val < k.val → ¬(240 ∣ m.val^2)) ∧
  (240 ∣ k.val^2) := by
  sorry

end NUMINAMATH_CALUDE_least_k_squared_divisible_by_240_l3905_390512


namespace NUMINAMATH_CALUDE_mean_score_of_all_students_l3905_390501

/-- Calculates the mean score of all students given the mean scores of two classes and the ratio of students in those classes. -/
theorem mean_score_of_all_students
  (morning_mean : ℝ)
  (afternoon_mean : ℝ)
  (morning_students : ℕ)
  (afternoon_students : ℕ)
  (h1 : morning_mean = 90)
  (h2 : afternoon_mean = 75)
  (h3 : morning_students = 2 * afternoon_students / 5) :
  let total_students := morning_students + afternoon_students
  let total_score := morning_mean * morning_students + afternoon_mean * afternoon_students
  total_score / total_students = 79 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_score_of_all_students_l3905_390501


namespace NUMINAMATH_CALUDE_area_triangle_ABC_area_DEFGH_area_triangle_JKL_l3905_390564

-- Define the grid unit
def grid_unit : ℝ := 1

-- Define the dimensions of triangle ABC
def triangle_ABC_base : ℝ := 2 * grid_unit
def triangle_ABC_height : ℝ := 3 * grid_unit

-- Define the dimensions of the square for DEFGH and JKL
def square_side : ℝ := 5 * grid_unit

-- Theorem for the area of triangle ABC
theorem area_triangle_ABC : 
  (1/2) * triangle_ABC_base * triangle_ABC_height = 3 := by sorry

-- Theorem for the area of figure DEFGH
theorem area_DEFGH : 
  square_side^2 - (1/2) * triangle_ABC_base * triangle_ABC_height = 22 := by sorry

-- Theorem for the area of triangle JKL
theorem area_triangle_JKL : 
  square_side^2 - ((1/2) * triangle_ABC_base * triangle_ABC_height + 
  (1/2) * square_side * (square_side - triangle_ABC_height) + 
  (1/2) * square_side * triangle_ABC_base) = 19/2 := by sorry

end NUMINAMATH_CALUDE_area_triangle_ABC_area_DEFGH_area_triangle_JKL_l3905_390564


namespace NUMINAMATH_CALUDE_sufficient_condition_absolute_value_l3905_390574

theorem sufficient_condition_absolute_value (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) → a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_absolute_value_l3905_390574


namespace NUMINAMATH_CALUDE_saxbridge_parade_max_members_l3905_390577

theorem saxbridge_parade_max_members :
  ∀ n : ℕ,
  (15 * n < 1200) →
  (15 * n) % 24 = 3 →
  (∀ m : ℕ, (15 * m < 1200) ∧ (15 * m) % 24 = 3 → 15 * m ≤ 15 * n) →
  15 * n = 1155 :=
sorry

end NUMINAMATH_CALUDE_saxbridge_parade_max_members_l3905_390577


namespace NUMINAMATH_CALUDE_employee_pay_l3905_390557

theorem employee_pay (x y z : ℝ) : 
  x + y + z = 900 →
  x = 1.2 * y →
  z = 0.8 * y →
  y = 300 := by
sorry

end NUMINAMATH_CALUDE_employee_pay_l3905_390557


namespace NUMINAMATH_CALUDE_five_digit_palindromes_count_l3905_390583

/-- A function that counts the number of 5-digit palindromes -/
def count_5digit_palindromes : ℕ :=
  let A := 9  -- digits 1 to 9
  let B := 10 -- digits 0 to 9
  let C := 10 -- digits 0 to 9
  A * B * C

/-- Theorem stating that the number of 5-digit palindromes is 900 -/
theorem five_digit_palindromes_count : count_5digit_palindromes = 900 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_palindromes_count_l3905_390583


namespace NUMINAMATH_CALUDE_last_number_is_25_l3905_390587

theorem last_number_is_25 (numbers : Fin 7 → ℝ) :
  (numbers 0 + numbers 1 + numbers 2 + numbers 3) / 4 = 13 →
  (numbers 3 + numbers 4 + numbers 5 + numbers 6) / 4 = 15 →
  numbers 4 + numbers 5 + numbers 6 = 55 →
  (numbers 3) ^ 2 = numbers 6 →
  numbers 6 = 25 := by
sorry

end NUMINAMATH_CALUDE_last_number_is_25_l3905_390587


namespace NUMINAMATH_CALUDE_line_direction_vector_l3905_390510

/-- Given a line passing through points (-3, 4) and (4, -1) with direction vector (a, a/2), prove a = -10 -/
theorem line_direction_vector (a : ℝ) : 
  (∃ (k : ℝ), k ≠ 0 ∧ k * (4 - (-3)) = a ∧ k * (-1 - 4) = a/2) → 
  a = -10 := by
sorry

end NUMINAMATH_CALUDE_line_direction_vector_l3905_390510


namespace NUMINAMATH_CALUDE_sum_congruence_mod_seven_l3905_390535

theorem sum_congruence_mod_seven :
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_mod_seven_l3905_390535


namespace NUMINAMATH_CALUDE_homework_points_l3905_390547

theorem homework_points (total_points : ℕ) (test_quiz_ratio : ℕ) (quiz_homework_diff : ℕ)
  (h1 : total_points = 265)
  (h2 : test_quiz_ratio = 4)
  (h3 : quiz_homework_diff = 5) :
  ∃ (homework : ℕ), 
    homework + (homework + quiz_homework_diff) + test_quiz_ratio * (homework + quiz_homework_diff) = total_points ∧ 
    homework = 40 := by
  sorry

end NUMINAMATH_CALUDE_homework_points_l3905_390547


namespace NUMINAMATH_CALUDE_fraction_value_at_three_l3905_390540

theorem fraction_value_at_three :
  let x : ℝ := 3
  (x^8 + 8*x^4 + 16) / (x^4 - 4) = 93 := by sorry

end NUMINAMATH_CALUDE_fraction_value_at_three_l3905_390540


namespace NUMINAMATH_CALUDE_rectangle_tiling_l3905_390539

/-- A rectangle can be tiled with 4x4 squares -/
def is_tileable (m n : ℕ) : Prop :=
  ∃ (a b : ℕ), m = 4 * a ∧ n = 4 * b

/-- If a rectangle with dimensions m × n can be tiled with 4 × 4 squares, 
    then m and n are divisible by 4 -/
theorem rectangle_tiling (m n : ℕ) :
  is_tileable m n → (4 ∣ m) ∧ (4 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_tiling_l3905_390539


namespace NUMINAMATH_CALUDE_factor_tree_problem_l3905_390568

theorem factor_tree_problem (X Y Z F G : ℕ) :
  X = Y * Z ∧
  Y = 5 * F ∧
  Z = 7 * G ∧
  F = 5 * 3 ∧
  G = 7 * 3 →
  X = 11025 :=
by sorry

end NUMINAMATH_CALUDE_factor_tree_problem_l3905_390568


namespace NUMINAMATH_CALUDE_no_equal_xyz_l3905_390527

theorem no_equal_xyz : ¬∃ t : ℝ, (1 - 3*t = 2*t - 3) ∧ (1 - 3*t = 4*t^2 - 5*t + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_equal_xyz_l3905_390527


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l3905_390524

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

def Rectangle.area (r : Rectangle) : ℝ := sorry

def angle (p1 p2 p3 : Point) : ℝ := sorry

def distance (p1 p2 : Point) : ℝ := sorry

def foldPoint (p : Point) (line : Point × Point) : Point := sorry

theorem rectangle_area_theorem (ABCD : Rectangle) (E F : Point) (B' C' : Point) :
  E.x = ABCD.A.x ∧ F.x = ABCD.D.x →
  distance ABCD.B E < distance ABCD.C F →
  B' = foldPoint ABCD.B (E, F) →
  C' = foldPoint ABCD.C (E, F) →
  C'.x = ABCD.A.x →
  angle ABCD.A B' C' = 2 * angle B' E ABCD.A →
  distance ABCD.A B' = 8 →
  distance ABCD.B E = 15 →
  ∃ (a b c : ℕ), 
    Rectangle.area ABCD = a + b * Real.sqrt c ∧
    a = 100 ∧ b = 4 ∧ c = 23 ∧
    a + b + c = 127 ∧
    ∀ (p : ℕ), Prime p → c % (p * p) ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l3905_390524


namespace NUMINAMATH_CALUDE_sunset_ridge_farm_arrangements_l3905_390550

/-- The number of ways to arrange animals in a row with specific conditions -/
def animalArrangements (nRabbits nDogs nGoats nParrots : ℕ) : ℕ :=
  Nat.factorial 4 * Nat.factorial nRabbits * Nat.factorial nDogs * Nat.factorial nGoats * Nat.factorial nParrots

/-- Theorem stating the number of arrangements for the given problem -/
theorem sunset_ridge_farm_arrangements :
  animalArrangements 5 3 4 2 = 414720 := by
  sorry

end NUMINAMATH_CALUDE_sunset_ridge_farm_arrangements_l3905_390550


namespace NUMINAMATH_CALUDE_tire_price_proof_l3905_390520

theorem tire_price_proof :
  let regular_price : ℝ := 90
  let third_tire_price : ℝ := 5
  let total_cost : ℝ := 185
  (2 * regular_price + third_tire_price = total_cost) →
  regular_price = 90 := by
sorry

end NUMINAMATH_CALUDE_tire_price_proof_l3905_390520


namespace NUMINAMATH_CALUDE_inscribed_triangle_theorem_l3905_390542

/-- A triangle with an inscribed circle -/
structure InscribedTriangle where
  -- The sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The radius of the inscribed circle
  r : ℝ
  -- The segments of one side divided by the point of tangency
  s₁ : ℝ
  s₂ : ℝ
  -- Conditions
  side_division : a = s₁ + s₂
  radius_positive : r > 0
  sides_positive : a > 0 ∧ b > 0 ∧ c > 0

/-- The theorem stating the relationship between the sides and radius -/
theorem inscribed_triangle_theorem (t : InscribedTriangle) 
  (h₁ : t.s₁ = 10 ∧ t.s₂ = 14)
  (h₂ : t.r = 5)
  (h₃ : t.b = 30) :
  t.c = 36 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_theorem_l3905_390542


namespace NUMINAMATH_CALUDE_candle_height_relation_l3905_390597

/-- Represents the remaining height of a burning candle -/
def remaining_height (initial_height burning_rate t : ℝ) : ℝ :=
  initial_height - burning_rate * t

/-- Theorem stating the relationship between remaining height and burning time for a specific candle -/
theorem candle_height_relation (h t : ℝ) :
  remaining_height 20 4 t = h ↔ h = 20 - 4 * t := by sorry

end NUMINAMATH_CALUDE_candle_height_relation_l3905_390597


namespace NUMINAMATH_CALUDE_days_until_lifting_heavy_l3905_390534

/-- The number of days it takes for James' pain to subside -/
def pain_subsiding_days : ℕ := 3

/-- The factor by which the full healing time exceeds the pain subsiding time -/
def healing_factor : ℕ := 5

/-- The number of days James waits after full healing before working out -/
def waiting_days : ℕ := 3

/-- The number of weeks James waits before lifting heavy after starting to work out -/
def weeks_before_lifting : ℕ := 3

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem stating the total number of days until James can lift heavy again -/
theorem days_until_lifting_heavy : 
  pain_subsiding_days * healing_factor + waiting_days + weeks_before_lifting * days_per_week = 39 := by
  sorry

end NUMINAMATH_CALUDE_days_until_lifting_heavy_l3905_390534


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3905_390519

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3905_390519


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_999_l3905_390515

theorem largest_prime_factor_of_999 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 999 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 999 → q ≤ p := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_999_l3905_390515


namespace NUMINAMATH_CALUDE_pigeon_hole_problem_l3905_390558

theorem pigeon_hole_problem (pigeonholes : ℕ) (pigeons : ℕ) : 
  (pigeons = 6 * pigeonholes + 3) →
  (pigeons + 5 = 8 * pigeonholes) →
  (pigeons = 27 ∧ pigeonholes = 4) := by
  sorry

end NUMINAMATH_CALUDE_pigeon_hole_problem_l3905_390558


namespace NUMINAMATH_CALUDE_equation_solutions_l3905_390551

theorem equation_solutions :
  (let x1 : ℝ := Real.sqrt 2
   let x2 : ℝ := -Real.sqrt 2
   x1^2 = 2 ∧ x2^2 = 2) ∧
  (let x1 : ℝ := 1/2
   let x2 : ℝ := -1/2
   4*x1^2 - 1 = 0 ∧ 4*x2^2 - 1 = 0) ∧
  (let x1 : ℝ := 3
   let x2 : ℝ := -1
   (x1-1)^2 - 4 = 0 ∧ (x2-1)^2 - 4 = 0) ∧
  (let x1 : ℝ := 1
   let x2 : ℝ := 5
   12*(3-x1)^2 - 48 = 0 ∧ 12*(3-x2)^2 - 48 = 0) := by
  sorry


end NUMINAMATH_CALUDE_equation_solutions_l3905_390551


namespace NUMINAMATH_CALUDE_prob_at_least_three_even_is_five_sixteenths_l3905_390582

/-- Probability of rolling an even number on a fair die -/
def prob_even : ℚ := 1/2

/-- Number of rolls -/
def num_rolls : ℕ := 4

/-- Probability of rolling an even number at least three times in four rolls -/
def prob_at_least_three_even : ℚ :=
  Nat.choose num_rolls 3 * prob_even^3 * (1 - prob_even) +
  Nat.choose num_rolls 4 * prob_even^4

theorem prob_at_least_three_even_is_five_sixteenths :
  prob_at_least_three_even = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_three_even_is_five_sixteenths_l3905_390582


namespace NUMINAMATH_CALUDE_production_difference_formula_l3905_390559

/-- The number of widgets David produces per hour on Monday -/
def w (t : ℝ) : ℝ := 2 * t

/-- The number of hours David works on Monday -/
def monday_hours (t : ℝ) : ℝ := t

/-- The number of hours David works on Tuesday -/
def tuesday_hours (t : ℝ) : ℝ := t - 1

/-- The number of widgets David produces per hour on Tuesday -/
def tuesday_rate (t : ℝ) : ℝ := w t + 5

/-- The difference in widget production between Monday and Tuesday -/
def production_difference (t : ℝ) : ℝ :=
  w t * monday_hours t - tuesday_rate t * tuesday_hours t

theorem production_difference_formula (t : ℝ) :
  production_difference t = -3 * t + 5 := by
  sorry

end NUMINAMATH_CALUDE_production_difference_formula_l3905_390559


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3905_390576

theorem polynomial_factorization (x : ℝ) : 
  2 * x^3 - 4 * x^2 + 2 * x = 2 * x * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3905_390576


namespace NUMINAMATH_CALUDE_west_notation_l3905_390563

-- Define a type for distance with direction
inductive DirectedDistance
  | east (km : ℝ)
  | west (km : ℝ)

-- Define a function to convert DirectedDistance to a signed real number
def directedDistanceToSigned : DirectedDistance → ℝ
  | DirectedDistance.east km => km
  | DirectedDistance.west km => -km

-- State the theorem
theorem west_notation (d : ℝ) :
  directedDistanceToSigned (DirectedDistance.east 3) = 3 →
  directedDistanceToSigned (DirectedDistance.west 2) = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_west_notation_l3905_390563


namespace NUMINAMATH_CALUDE_lcm_of_9_12_15_l3905_390599

theorem lcm_of_9_12_15 : Nat.lcm 9 (Nat.lcm 12 15) = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_9_12_15_l3905_390599


namespace NUMINAMATH_CALUDE_equation_solution_l3905_390588

theorem equation_solution (x : ℝ) : (4 + 2*x) / (7 + x) = (2 + x) / (3 + x) ↔ x = -2 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3905_390588


namespace NUMINAMATH_CALUDE_solve_for_b_l3905_390555

theorem solve_for_b (a b : ℚ) (h1 : a = 5) (h2 : b - a + (2 * b / 3) = 7) : b = 36 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l3905_390555


namespace NUMINAMATH_CALUDE_game_value_proof_l3905_390585

def super_nintendo_value : ℝ := 150
def store_credit_percentage : ℝ := 0.8
def tom_payment : ℝ := 80
def tom_change : ℝ := 10
def nes_sale_price : ℝ := 160

theorem game_value_proof :
  let credit := super_nintendo_value * store_credit_percentage
  let tom_actual_payment := tom_payment - tom_change
  let credit_used := nes_sale_price - tom_actual_payment
  credit - credit_used = 30 := by sorry

end NUMINAMATH_CALUDE_game_value_proof_l3905_390585


namespace NUMINAMATH_CALUDE_average_marks_two_classes_l3905_390571

theorem average_marks_two_classes 
  (n1 : ℕ) (n2 : ℕ) (avg1 : ℝ) (avg2 : ℝ) :
  n1 = 30 →
  n2 = 50 →
  avg1 = 50 →
  avg2 = 60 →
  ((n1 : ℝ) * avg1 + (n2 : ℝ) * avg2) / ((n1 : ℝ) + (n2 : ℝ)) = 56.25 :=
by sorry

end NUMINAMATH_CALUDE_average_marks_two_classes_l3905_390571


namespace NUMINAMATH_CALUDE_lcm_factor_proof_l3905_390509

theorem lcm_factor_proof (A B : ℕ+) (X : ℕ+) : 
  Nat.gcd A B = 23 →
  Nat.lcm A B = 23 * 13 * X →
  A = 322 →
  X = 14 := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_proof_l3905_390509


namespace NUMINAMATH_CALUDE_greg_sisters_count_l3905_390566

def number_of_sisters (total_bars : ℕ) (days_in_week : ℕ) (traded_bars : ℕ) (bars_per_sister : ℕ) : ℕ :=
  (total_bars - days_in_week - traded_bars) / bars_per_sister

theorem greg_sisters_count :
  let total_bars : ℕ := 20
  let days_in_week : ℕ := 7
  let traded_bars : ℕ := 3
  let bars_per_sister : ℕ := 5
  number_of_sisters total_bars days_in_week traded_bars bars_per_sister = 2 := by
  sorry

end NUMINAMATH_CALUDE_greg_sisters_count_l3905_390566


namespace NUMINAMATH_CALUDE_circle_radius_zero_l3905_390505

theorem circle_radius_zero (x y : ℝ) : 
  (∀ x y, x^2 + 8*x + y^2 - 10*y + 41 = 0) → 
  ∃ h k r, (∀ x y, (x - h)^2 + (y - k)^2 = r^2) ∧ r = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_zero_l3905_390505


namespace NUMINAMATH_CALUDE_div_value_problem_l3905_390598

theorem div_value_problem (a b d : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / d = 2 / 5) : 
  d / a = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_div_value_problem_l3905_390598


namespace NUMINAMATH_CALUDE_triangle_areas_product_l3905_390596

theorem triangle_areas_product (h₁ h₂ h₃ : ℝ) 
  (h1 : h₁ = 1)
  (h2 : h₂ = 1 + Real.sqrt 3 / 2)
  (h3 : h₃ = 1 - Real.sqrt 3 / 2) :
  (1/2 * 1 * h₁) * (1/2 * 1 * h₂) * (1/2 * 1 * h₃) = 1/32 := by
  sorry

#check triangle_areas_product

end NUMINAMATH_CALUDE_triangle_areas_product_l3905_390596


namespace NUMINAMATH_CALUDE_smallest_consecutive_number_l3905_390504

theorem smallest_consecutive_number (x : ℕ) : 
  (∃ (a b c d : ℕ), x + a + b + c + d = 225 ∧ 
   a = x + 1 ∧ b = x + 2 ∧ c = x + 3 ∧ d = x + 4 ∧
   ∃ (k : ℕ), x = 7 * k) → 
  x = 42 := by
sorry

end NUMINAMATH_CALUDE_smallest_consecutive_number_l3905_390504


namespace NUMINAMATH_CALUDE_interior_angles_integral_count_l3905_390561

theorem interior_angles_integral_count : 
  (Finset.filter (fun n : ℕ => n > 2 ∧ (n - 2) * 180 % n = 0) (Finset.range 361)).card = 22 := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_integral_count_l3905_390561


namespace NUMINAMATH_CALUDE_jacket_cost_calculation_l3905_390591

def total_spent : ℚ := 33.56
def shorts_cost : ℚ := 13.99
def shirt_cost : ℚ := 12.14

theorem jacket_cost_calculation : 
  total_spent - shorts_cost - shirt_cost = 7.43 := by sorry

end NUMINAMATH_CALUDE_jacket_cost_calculation_l3905_390591


namespace NUMINAMATH_CALUDE_square_division_negative_numbers_l3905_390554

theorem square_division_negative_numbers : (-128)^2 / (-64)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_division_negative_numbers_l3905_390554


namespace NUMINAMATH_CALUDE_b_share_calculation_l3905_390518

theorem b_share_calculation (total : ℝ) : 
  let a := (2 : ℝ) / 15 * total
  let b := (3 : ℝ) / 15 * total
  let c := (4 : ℝ) / 15 * total
  let d := (6 : ℝ) / 15 * total
  d - c = 700 → b = 1050 := by
  sorry

end NUMINAMATH_CALUDE_b_share_calculation_l3905_390518


namespace NUMINAMATH_CALUDE_road_planting_equation_l3905_390569

/-- Represents the road planting scenario --/
structure RoadPlanting where
  x : ℕ  -- Original number of saplings
  shortage : ℕ  -- Number of saplings short when planting every 6 meters
  interval1 : ℕ  -- First planting interval (in meters)
  interval2 : ℕ  -- Second planting interval (in meters)

/-- Theorem representing the road planting problem --/
theorem road_planting_equation (rp : RoadPlanting) 
  (h1 : rp.interval1 = 6) 
  (h2 : rp.interval2 = 7) 
  (h3 : rp.shortage = 22) : 
  rp.interval1 * (rp.x + rp.shortage - 1) = rp.interval2 * (rp.x - 1) := by
  sorry

#check road_planting_equation

end NUMINAMATH_CALUDE_road_planting_equation_l3905_390569


namespace NUMINAMATH_CALUDE_number_puzzle_l3905_390522

theorem number_puzzle : ∃ x : ℝ, (2 * x) / 16 = 25 ∧ x = 200 := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l3905_390522


namespace NUMINAMATH_CALUDE_final_alloy_mass_l3905_390589

/-- Given two alloys with different copper percentages and their masses,
    prove that the total mass of the final alloy is the sum of the masses of the component alloys. -/
theorem final_alloy_mass
  (alloy1_copper_percent : ℚ)
  (alloy2_copper_percent : ℚ)
  (final_alloy_copper_percent : ℚ)
  (alloy1_mass : ℚ)
  (alloy2_mass : ℚ)
  (h1 : alloy1_copper_percent = 25 / 100)
  (h2 : alloy2_copper_percent = 50 / 100)
  (h3 : final_alloy_copper_percent = 45 / 100)
  (h4 : alloy1_mass = 200)
  (h5 : alloy2_mass = 800) :
  alloy1_mass + alloy2_mass = 1000 := by
  sorry

end NUMINAMATH_CALUDE_final_alloy_mass_l3905_390589


namespace NUMINAMATH_CALUDE_inequality_holds_l3905_390593

theorem inequality_holds (p q : ℝ) (h_p : 0 < p) (h_p_upper : p < 2) (h_q : 0 < q) :
  (4 * (p * q^2 + 2 * p^2 * q + 2 * q^2 + 2 * p * q)) / (p + q) > 3 * p^2 * q :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_l3905_390593


namespace NUMINAMATH_CALUDE_exists_k_not_equal_f_diff_l3905_390548

/-- f(n) is the largest integer k such that 2^k divides n -/
def f (n : ℕ) : ℕ := Nat.log2 (n.gcd (2^n))

/-- Theorem statement -/
theorem exists_k_not_equal_f_diff (n : ℕ) (h : n ≥ 2) (a : Fin n → ℕ)
  (h_sorted : ∀ i j, i < j → a i < a j) :
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧
    ∀ i j : Fin n, j ≤ i → f (a i - a j) ≠ k :=
  sorry

end NUMINAMATH_CALUDE_exists_k_not_equal_f_diff_l3905_390548


namespace NUMINAMATH_CALUDE_ellipse_k_values_l3905_390549

def ellipse_equation (x y k : ℝ) : Prop := x^2/5 + y^2/k = 1

def eccentricity (e : ℝ) : Prop := e = Real.sqrt 10 / 5

theorem ellipse_k_values (k : ℝ) :
  (∃ x y, ellipse_equation x y k) ∧ eccentricity (Real.sqrt 10 / 5) →
  k = 3 ∨ k = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_k_values_l3905_390549


namespace NUMINAMATH_CALUDE_range_of_a_l3905_390565

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x + f (-x) = x^2) ∧
  (∀ x ≥ 0, deriv f x - x - 1 < 0)

/-- The main theorem -/
theorem range_of_a (f : ℝ → ℝ) (h : special_function f) :
  ∀ a, (f (2 - a) ≥ f a + 4 - 4*a) → a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3905_390565


namespace NUMINAMATH_CALUDE_solution_set_of_increasing_function_l3905_390572

theorem solution_set_of_increasing_function 
  (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_f_0 : f 0 = -1) 
  (h_f_3 : f 3 = 1) : 
  {x : ℝ | |f (x + 1)| < 1} = Set.Ioo (-1 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_increasing_function_l3905_390572


namespace NUMINAMATH_CALUDE_equation_solutions_l3905_390525

theorem equation_solutions :
  (∀ x : ℝ, (5 - 2*x)^2 - 16 = 0 ↔ (x = 1/2 ∨ x = 9/2)) ∧
  (∀ x : ℝ, 2*(x - 3) = x^2 - 9 ↔ (x = 3 ∨ x = -1)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3905_390525


namespace NUMINAMATH_CALUDE_existence_of_polynomial_and_c1_value_l3905_390546

/-- D(m) counts the number of quadruples (a₁, a₂, a₃, a₄) of distinct integers 
    with 1 ≤ aᵢ ≤ m for all i such that m divides a₁+a₂+a₃+a₄ -/
def D (m : ℕ) : ℕ := sorry

/-- The polynomial q(x) = c₃x³ + c₂x² + c₁x + c₀ -/
def q (x : ℕ) : ℕ := sorry

theorem existence_of_polynomial_and_c1_value :
  ∃ (c₃ c₂ c₁ c₀ : ℤ), 
    (∀ m : ℕ, m ≥ 5 → Odd m → D m = c₃ * m^3 + c₂ * m^2 + c₁ * m + c₀) ∧ 
    c₁ = 11 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_polynomial_and_c1_value_l3905_390546


namespace NUMINAMATH_CALUDE_total_chickens_and_ducks_prove_total_chickens_and_ducks_l3905_390521

theorem total_chickens_and_ducks : ℕ → ℕ → ℕ → Prop :=
  fun (chickens ducks total : ℕ) =>
    chickens = 45 ∧ 
    chickens = ducks + 8 ∧ 
    total = chickens + ducks → 
    total = 82

-- Proof
theorem prove_total_chickens_and_ducks : 
  ∃ (chickens ducks total : ℕ), total_chickens_and_ducks chickens ducks total :=
by
  sorry

end NUMINAMATH_CALUDE_total_chickens_and_ducks_prove_total_chickens_and_ducks_l3905_390521


namespace NUMINAMATH_CALUDE_egg_production_increase_l3905_390502

theorem egg_production_increase (last_year_production this_year_production : ℕ) 
  (h1 : last_year_production = 1416)
  (h2 : this_year_production = 4636) :
  this_year_production - last_year_production = 3220 :=
by sorry

end NUMINAMATH_CALUDE_egg_production_increase_l3905_390502


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3905_390526

theorem solution_set_quadratic_inequality :
  Set.Icc (-(1/2) : ℝ) 1 = {x : ℝ | 2 * x^2 - x ≤ 1} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3905_390526


namespace NUMINAMATH_CALUDE_circular_table_arrangements_l3905_390578

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem circular_table_arrangements (num_mathletes : ℕ) (num_coaches : ℕ) : 
  num_mathletes = 4 → num_coaches = 2 → 
  (factorial num_mathletes * 2) / 2 = 24 := by
  sorry

#check circular_table_arrangements

end NUMINAMATH_CALUDE_circular_table_arrangements_l3905_390578


namespace NUMINAMATH_CALUDE_series_sum_l3905_390507

def series_term (n : ℕ) : ℚ :=
  (6 * n + 1) / ((6 * n - 1)^2 * (6 * n + 5)^2)

theorem series_sum : ∑' (n : ℕ), series_term (n + 1) = 1 / 300 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l3905_390507


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3905_390580

theorem trigonometric_identities :
  (Real.sin (30 * π / 180) + Real.cos (45 * π / 180) = (1 + Real.sqrt 2) / 2) ∧
  (Real.sin (60 * π / 180) ^ 2 + Real.cos (60 * π / 180) ^ 2 - Real.tan (45 * π / 180) = 0) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3905_390580


namespace NUMINAMATH_CALUDE_geometry_test_passing_l3905_390523

theorem geometry_test_passing (total_problems : ℕ) (passing_percentage : ℚ) 
  (hp : passing_percentage = 85 / 100) (ht : total_problems = 50) :
  ∃ (max_missable : ℕ), 
    (max_missable : ℚ) / total_problems ≤ 1 - passing_percentage ∧
    ∀ (n : ℕ), (n : ℚ) / total_problems ≤ 1 - passing_percentage → n ≤ max_missable :=
by sorry

end NUMINAMATH_CALUDE_geometry_test_passing_l3905_390523


namespace NUMINAMATH_CALUDE_circle_bisection_minimum_l3905_390543

theorem circle_bisection_minimum (a b : ℝ) :
  a > 0 →
  b > 0 →
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 1 = 0 ∧ 2*a*x - b*y + 2 = 0) →
  (∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 1 = 0 → (2*a*x - b*y + 2 = 0 ∨ 2*a*x - b*y + 2 ≠ 0)) →
  (1/a + 4/b) ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_circle_bisection_minimum_l3905_390543


namespace NUMINAMATH_CALUDE_sixth_quiz_score_l3905_390579

def existing_scores : List ℕ := [86, 91, 83, 88, 97]
def target_mean : ℕ := 90
def num_quizzes : ℕ := 6

theorem sixth_quiz_score (x : ℕ) :
  (existing_scores.sum + x) / num_quizzes = target_mean ↔ x = 95 := by
  sorry

end NUMINAMATH_CALUDE_sixth_quiz_score_l3905_390579


namespace NUMINAMATH_CALUDE_not_divisible_by_three_l3905_390513

theorem not_divisible_by_three (n : ℤ) : ¬(3 ∣ (n^2 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_three_l3905_390513


namespace NUMINAMATH_CALUDE_correct_final_bill_amount_l3905_390537

/-- Calculates the final bill amount after applying surcharges -/
def final_bill_amount (initial_bill : ℝ) (first_surcharge_rate : ℝ) (second_surcharge_rate : ℝ) : ℝ :=
  initial_bill * (1 + first_surcharge_rate) * (1 + second_surcharge_rate)

/-- Theorem stating that the final bill amount is correct -/
theorem correct_final_bill_amount :
  final_bill_amount 800 0.05 0.08 = 907.2 := by sorry

end NUMINAMATH_CALUDE_correct_final_bill_amount_l3905_390537


namespace NUMINAMATH_CALUDE_group_size_correct_l3905_390560

/-- The number of members in the group -/
def n : ℕ := 93

/-- The total collection in paise -/
def total_paise : ℕ := 8649

/-- Theorem stating that n is the correct number of members -/
theorem group_size_correct : n * n = total_paise := by sorry

end NUMINAMATH_CALUDE_group_size_correct_l3905_390560


namespace NUMINAMATH_CALUDE_unique_solution_to_x_equals_negative_x_l3905_390545

theorem unique_solution_to_x_equals_negative_x : 
  ∀ x : ℝ, x = -x ↔ x = 0 := by sorry

end NUMINAMATH_CALUDE_unique_solution_to_x_equals_negative_x_l3905_390545


namespace NUMINAMATH_CALUDE_linda_tees_sold_l3905_390595

/-- Calculates the number of tees sold given the prices, number of jeans sold, and total money -/
def tees_sold (jeans_price tee_price : ℕ) (jeans_sold : ℕ) (total_money : ℕ) : ℕ :=
  (total_money - jeans_price * jeans_sold) / tee_price

theorem linda_tees_sold :
  tees_sold 11 8 4 100 = 7 := by
  sorry

end NUMINAMATH_CALUDE_linda_tees_sold_l3905_390595


namespace NUMINAMATH_CALUDE_m_range_theorem_l3905_390500

-- Define proposition p
def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 - m * x + 1 < 0

-- Define proposition q
def q (m : ℝ) : Prop := (m - 1) * (3 - m) < 0

-- Define the range of m
def m_range (m : ℝ) : Prop := (0 < m ∧ m ≤ 1) ∨ (3 ≤ m ∧ m < 4)

-- Theorem statement
theorem m_range_theorem (m : ℝ) : 
  (¬(p m ∧ q m) ∧ (p m ∨ q m)) → m_range m :=
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l3905_390500


namespace NUMINAMATH_CALUDE_fraction_inequality_l3905_390538

theorem fraction_inequality : 
  (1 + 1/3 : ℚ) = 4/3 ∧ 
  12/9 = 4/3 ∧ 
  8/6 = 4/3 ∧ 
  (1 + 2/7 : ℚ) ≠ 4/3 ∧ 
  16/12 = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3905_390538


namespace NUMINAMATH_CALUDE_smurfs_gold_coins_l3905_390530

theorem smurfs_gold_coins (total : ℕ) (smurfs : ℕ) (gargamel : ℕ) 
  (h1 : total = 200)
  (h2 : smurfs + gargamel = total)
  (h3 : (2 : ℚ) / 3 * smurfs = (4 : ℚ) / 5 * gargamel + 38) :
  smurfs = 135 := by
  sorry

end NUMINAMATH_CALUDE_smurfs_gold_coins_l3905_390530


namespace NUMINAMATH_CALUDE_imaginary_unit_power_2018_l3905_390556

theorem imaginary_unit_power_2018 (i : ℂ) (hi : i^2 = -1) : i^2018 = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_2018_l3905_390556


namespace NUMINAMATH_CALUDE_arman_second_week_hours_l3905_390503

/-- Calculates the number of hours worked in the second week given the conditions of Arman's work schedule and earnings. -/
def hours_worked_second_week (
  first_week_hours : ℕ)
  (first_week_rate : ℚ)
  (rate_increase : ℚ)
  (total_earnings : ℚ) : ℚ :=
  let first_week_earnings := first_week_hours * first_week_rate
  let second_week_earnings := total_earnings - first_week_earnings
  let new_rate := first_week_rate + rate_increase
  second_week_earnings / new_rate

/-- Theorem stating that given the conditions of Arman's work schedule and earnings, 
    the number of hours worked in the second week is 40. -/
theorem arman_second_week_hours :
  hours_worked_second_week 35 10 0.5 770 = 40 := by
  sorry

end NUMINAMATH_CALUDE_arman_second_week_hours_l3905_390503


namespace NUMINAMATH_CALUDE_biology_books_count_l3905_390528

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of different chemistry books -/
def chem_books : ℕ := 8

/-- The total number of ways to choose 2 books of each type -/
def total_ways : ℕ := 1260

/-- The number of different biology books -/
def bio_books : ℕ := 10

theorem biology_books_count :
  choose_two bio_books * choose_two chem_books = total_ways :=
sorry

#check biology_books_count

end NUMINAMATH_CALUDE_biology_books_count_l3905_390528


namespace NUMINAMATH_CALUDE_room_length_proof_l3905_390544

theorem room_length_proof (width : Real) (cost_per_sqm : Real) (total_cost : Real) :
  width = 3.75 →
  cost_per_sqm = 700 →
  total_cost = 14437.5 →
  (total_cost / cost_per_sqm) / width = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_room_length_proof_l3905_390544


namespace NUMINAMATH_CALUDE_system_solution_l3905_390594

theorem system_solution (a b x y : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  x^5 * y^17 = a ∧ x^2 * y^7 = b → x = a^7 / b^17 ∧ y = b^5 / a^2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3905_390594


namespace NUMINAMATH_CALUDE_intersection_condition_l3905_390570

/-- 
Given two equations:
1) y = √(2x^2 + 2x - m)
2) y = x - 2
This theorem states that for these equations to have a real intersection, 
m must be greater than or equal to 12.
-/
theorem intersection_condition (x y m : ℝ) : 
  (y = Real.sqrt (2 * x^2 + 2 * x - m) ∧ y = x - 2) → m ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l3905_390570


namespace NUMINAMATH_CALUDE_base_conversion_correct_l3905_390581

-- Define the base 10 number
def base_10_num : ℕ := 3527

-- Define the base 7 representation
def base_7_representation : List ℕ := [1, 3, 1, 6, 6]

-- Theorem statement
theorem base_conversion_correct :
  base_10_num = (List.foldr (λ (digit : ℕ) (acc : ℕ) => digit + 7 * acc) 0 base_7_representation) :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_correct_l3905_390581


namespace NUMINAMATH_CALUDE_x_cubed_coefficient_l3905_390541

/-- The coefficient of x^3 in the expansion of (3x^3 + 2x^2 + 3x + 4)(5x^2 + 7x + 6) is 47 -/
theorem x_cubed_coefficient : 
  let p₁ : Polynomial ℤ := 3 * X^3 + 2 * X^2 + 3 * X + 4
  let p₂ : Polynomial ℤ := 5 * X^2 + 7 * X + 6
  (p₁ * p₂).coeff 3 = 47 := by
sorry

end NUMINAMATH_CALUDE_x_cubed_coefficient_l3905_390541


namespace NUMINAMATH_CALUDE_circle_area_ratio_l3905_390506

theorem circle_area_ratio (r : ℝ) (h : r > 0) : (π * (3 * r)^2) / (π * r^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l3905_390506


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3905_390553

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

-- Define monotonicity on an interval
def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y ∨ (∀ z, a ≤ z ∧ z ≤ b → f z = f x)

-- State the theorem
theorem sufficient_not_necessary (a : ℝ) :
  (a ≥ 2 → monotonic_on (f a) 1 2) ∧
  (∃ b : ℝ, b < 2 ∧ monotonic_on (f b) 1 2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3905_390553


namespace NUMINAMATH_CALUDE_cades_marbles_l3905_390575

theorem cades_marbles (initial_marbles : ℕ) (marbles_given : ℕ) : 
  initial_marbles = 87 → marbles_given = 8 → initial_marbles - marbles_given = 79 := by
  sorry

end NUMINAMATH_CALUDE_cades_marbles_l3905_390575


namespace NUMINAMATH_CALUDE_point_on_xOz_plane_l3905_390536

/-- A point in 3D Cartesian space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOz plane in 3D Cartesian space -/
def xOzPlane : Set Point3D :=
  {p : Point3D | p.y = 0}

/-- The given point (1, 0, 4) -/
def givenPoint : Point3D :=
  ⟨1, 0, 4⟩

/-- Theorem: The given point (1, 0, 4) lies on the xOz plane -/
theorem point_on_xOz_plane : givenPoint ∈ xOzPlane := by
  sorry

end NUMINAMATH_CALUDE_point_on_xOz_plane_l3905_390536


namespace NUMINAMATH_CALUDE_dogs_food_average_l3905_390567

theorem dogs_food_average (num_dogs : ℕ) (dog1_food : ℝ) (dog2_food : ℝ) (dog3_food : ℝ) :
  num_dogs = 3 →
  dog1_food = 13 →
  dog2_food = 2 * dog1_food →
  dog3_food = 6 →
  (dog1_food + dog2_food + dog3_food) / num_dogs = 15 := by
sorry

end NUMINAMATH_CALUDE_dogs_food_average_l3905_390567


namespace NUMINAMATH_CALUDE_people_needed_to_recruit_l3905_390584

def total_funding : ℝ := 1000
def current_funds : ℝ := 200
def average_funding_per_person : ℝ := 10

theorem people_needed_to_recruit : 
  (total_funding - current_funds) / average_funding_per_person = 80 := by
  sorry

end NUMINAMATH_CALUDE_people_needed_to_recruit_l3905_390584


namespace NUMINAMATH_CALUDE_line_through_points_l3905_390573

/-- Proves that for a line passing through (-3, 1) and (1, 7), m + b = 7 -/
theorem line_through_points (m b : ℝ) : 
  (1 = m * (-3) + b) → (7 = m * 1 + b) → m + b = 7 := by sorry

end NUMINAMATH_CALUDE_line_through_points_l3905_390573


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3905_390532

theorem max_value_of_expression (x y : ℤ) 
  (h1 : x^2 + y^2 < 16) 
  (h2 : x * y > 4) : 
  ∃ (max : ℤ), max = 3 ∧ x^2 - 2*x*y - 3*y ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3905_390532


namespace NUMINAMATH_CALUDE_product_of_numbers_l3905_390529

theorem product_of_numbers (x y : ℝ) 
  (sum_condition : x + y = 20) 
  (sum_squares_condition : x^2 + y^2 = 200) : 
  x * y = 100 := by
sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3905_390529


namespace NUMINAMATH_CALUDE_brothers_book_pages_l3905_390562

/-- Represents the number of pages read by a person in a week --/
structure WeeklyReading where
  total_pages : ℕ
  books_per_week : ℕ
  days_to_finish : ℕ

/-- Calculates the average pages read per day --/
def average_pages_per_day (r : WeeklyReading) : ℕ :=
  r.total_pages / r.days_to_finish

theorem brothers_book_pages 
  (ryan : WeeklyReading)
  (ryan_brother : WeeklyReading)
  (h1 : ryan.total_pages = 2100)
  (h2 : ryan.books_per_week = 5)
  (h3 : ryan.days_to_finish = 7)
  (h4 : ryan_brother.books_per_week = 7)
  (h5 : ryan_brother.days_to_finish = 7)
  (h6 : average_pages_per_day ryan = average_pages_per_day ryan_brother + 100) :
  ryan_brother.total_pages / ryan_brother.books_per_week = 200 :=
by sorry

end NUMINAMATH_CALUDE_brothers_book_pages_l3905_390562


namespace NUMINAMATH_CALUDE_john_cookies_left_l3905_390516

/-- The number of cookies John has left after sharing with his friend -/
def cookies_left : ℕ :=
  let initial_cookies : ℕ := 2 * 12
  let after_first_day : ℕ := initial_cookies - (initial_cookies / 4)
  let after_second_day : ℕ := after_first_day - 5
  let shared_cookies : ℕ := after_second_day / 3
  after_second_day - shared_cookies

theorem john_cookies_left : cookies_left = 9 := by
  sorry

end NUMINAMATH_CALUDE_john_cookies_left_l3905_390516


namespace NUMINAMATH_CALUDE_missing_number_solution_l3905_390590

theorem missing_number_solution : 
  ∃ x : ℝ, 0.72 * 0.43 + 0.12 * x = 0.3504 ∧ x = 0.34 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_solution_l3905_390590


namespace NUMINAMATH_CALUDE_frequency_not_exceeding_15_minutes_l3905_390592

def duration_intervals : List (Real × Real) := [(0, 5), (5, 10), (10, 15), (15, 20)]
def frequencies : List Nat := [20, 16, 9, 5]

def total_calls : Nat := frequencies.sum

def calls_not_exceeding_15 : Nat := (frequencies.take 3).sum

theorem frequency_not_exceeding_15_minutes : 
  (calls_not_exceeding_15 : Real) / total_calls = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_frequency_not_exceeding_15_minutes_l3905_390592


namespace NUMINAMATH_CALUDE_money_redistribution_theorem_l3905_390508

/-- Represents the money redistribution problem among three friends. -/
def MoneyRedistribution (a j t : ℚ) : Prop :=
  -- Initial conditions
  (t = 24) ∧
  -- First redistribution (Amy's turn)
  let a₁ := a - 2*j - t
  let j₁ := 3*j
  let t₁ := 2*t
  -- Second redistribution (Jan's turn)
  let a₂ := 2*a₁
  let j₂ := j₁ - (a₁ + t₁)
  let t₂ := 3*t₁
  -- Final redistribution (Toy's turn)
  let a₃ := 3*a₂
  let j₃ := 3*j₂
  let t₃ := t₂ - (a₃ - a₂ + j₃ - j₂)
  -- Final condition
  (t₃ = 24) →
  -- Conclusion
  (a + j + t = 72)

/-- The total amount of money among the three friends is 72 dollars. -/
theorem money_redistribution_theorem (a j t : ℚ) :
  MoneyRedistribution a j t → (a + j + t = 72) :=
by
  sorry


end NUMINAMATH_CALUDE_money_redistribution_theorem_l3905_390508


namespace NUMINAMATH_CALUDE_max_pieces_is_100_l3905_390552

/-- The size of the large cake in inches -/
def large_cake_size : ℕ := 20

/-- The size of the small cake pieces in inches -/
def small_piece_size : ℕ := 2

/-- The area of the large cake in square inches -/
def large_cake_area : ℕ := large_cake_size * large_cake_size

/-- The area of a small cake piece in square inches -/
def small_piece_area : ℕ := small_piece_size * small_piece_size

/-- The maximum number of small pieces that can be cut from the large cake -/
def max_pieces : ℕ := large_cake_area / small_piece_area

theorem max_pieces_is_100 : max_pieces = 100 := by
  sorry

end NUMINAMATH_CALUDE_max_pieces_is_100_l3905_390552


namespace NUMINAMATH_CALUDE_min_glass_pieces_for_intensity_reduction_l3905_390586

def light_intensity (a : ℝ) (x : ℕ) : ℝ := a * (0.9 ^ x)

theorem min_glass_pieces_for_intensity_reduction (a : ℝ) (h : a > 0) :
  ∃ n : ℕ, (∀ x : ℕ, x < n → light_intensity a x > a / 3) ∧
           light_intensity a n ≤ a / 3 ∧
           n = 11 :=
sorry

end NUMINAMATH_CALUDE_min_glass_pieces_for_intensity_reduction_l3905_390586


namespace NUMINAMATH_CALUDE_complex_magnitudes_sum_l3905_390533

theorem complex_magnitudes_sum : Complex.abs (3 - 5*I) + Complex.abs (3 + 7*I) = Real.sqrt 34 + Real.sqrt 58 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitudes_sum_l3905_390533


namespace NUMINAMATH_CALUDE_f_of_one_eq_two_l3905_390511

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + |x - 2|

-- State the theorem
theorem f_of_one_eq_two : f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_one_eq_two_l3905_390511


namespace NUMINAMATH_CALUDE_new_york_squares_count_l3905_390514

/-- The number of squares in New York City -/
def num_squares : ℕ := 15

/-- The total number of streetlights bought by the city council -/
def total_streetlights : ℕ := 200

/-- The number of streetlights required for each square -/
def streetlights_per_square : ℕ := 12

/-- The number of unused streetlights -/
def unused_streetlights : ℕ := 20

/-- Theorem stating that the number of squares in New York City is correct -/
theorem new_york_squares_count :
  num_squares * streetlights_per_square + unused_streetlights = total_streetlights :=
by sorry

end NUMINAMATH_CALUDE_new_york_squares_count_l3905_390514


namespace NUMINAMATH_CALUDE_unique_number_with_conditions_l3905_390531

theorem unique_number_with_conditions : ∃! N : ℤ,
  35 < N ∧ N < 70 ∧ N % 6 = 3 ∧ N % 8 = 1 ∧ N = 57 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_conditions_l3905_390531


namespace NUMINAMATH_CALUDE_sum_of_abc_l3905_390517

theorem sum_of_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : a^2 + b^2 + c^2 = 48) (h5 : a * b + b * c + c * a = 26) (h6 : a = 2 * b) :
  a + b + c = 6 + 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_abc_l3905_390517
