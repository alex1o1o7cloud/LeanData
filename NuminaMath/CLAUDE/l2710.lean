import Mathlib

namespace NUMINAMATH_CALUDE_batsman_average_increase_l2710_271007

/-- Represents a batsman's scoring record -/
structure BatsmanRecord where
  inningsPlayed : ℕ
  totalRuns : ℕ
  averageRuns : ℚ

/-- Calculates the increase in average runs after a new inning -/
def averageIncrease (oldRecord : BatsmanRecord) (newInningRuns : ℕ) : ℚ :=
  let newRecord : BatsmanRecord := {
    inningsPlayed := oldRecord.inningsPlayed + 1,
    totalRuns := oldRecord.totalRuns + newInningRuns,
    averageRuns := (oldRecord.totalRuns + newInningRuns : ℚ) / (oldRecord.inningsPlayed + 1)
  }
  newRecord.averageRuns - oldRecord.averageRuns

theorem batsman_average_increase :
  ∀ (oldRecord : BatsmanRecord),
    oldRecord.inningsPlayed = 16 →
    averageIncrease oldRecord 88 = 40 - oldRecord.averageRuns →
    averageIncrease oldRecord 88 = 3 := by
  sorry

#eval averageIncrease
  { inningsPlayed := 16, totalRuns := 592, averageRuns := 37 }
  88

end NUMINAMATH_CALUDE_batsman_average_increase_l2710_271007


namespace NUMINAMATH_CALUDE_min_value_on_circle_l2710_271048

theorem min_value_on_circle (x y : ℝ) (h : x^2 + y^2 - 2*x - 2*y + 1 = 0) :
  ∃ (m : ℝ), m = 4/3 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 - 2*x' - 2*y' + 1 = 0 →
    (y' - 4) / (x' - 2) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l2710_271048


namespace NUMINAMATH_CALUDE_constant_function_theorem_l2710_271039

theorem constant_function_theorem (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = f (x - y)) →
  ∃ C : ℝ, ∀ x : ℝ, f x = C :=
by sorry

end NUMINAMATH_CALUDE_constant_function_theorem_l2710_271039


namespace NUMINAMATH_CALUDE_rectangle_segment_comparison_l2710_271080

/-- Given a rectangle ABCD with specific properties, prove AM > BK -/
theorem rectangle_segment_comparison (A B C D M K : ℝ × ℝ) : 
  let AB : ℝ := 2
  let BD : ℝ := Real.sqrt 7
  let AC : ℝ := Real.sqrt (AB^2 + BD^2 - AB^2)
  -- Rectangle properties
  (B.1 - A.1 = AB ∧ B.2 = A.2) →
  (C.1 = B.1 ∧ C.2 - A.2 = AC) →
  (D.1 = A.1 ∧ D.2 = C.2) →
  -- M divides CD in 1:2 ratio
  (M.1 - C.1 = (1/3) * (D.1 - C.1) ∧ M.2 - C.2 = (1/3) * (D.2 - C.2)) →
  -- K is midpoint of AD
  (K.1 = (A.1 + D.1) / 2 ∧ K.2 = (A.2 + D.2) / 2) →
  -- Prove AM > BK
  Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) > Real.sqrt ((K.1 - B.1)^2 + (K.2 - B.2)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_rectangle_segment_comparison_l2710_271080


namespace NUMINAMATH_CALUDE_intersection_M_N_l2710_271054

-- Define set M
def M : Set ℝ := {x : ℝ | -4 ≤ x ∧ x ≤ 7}

-- Define set N
def N : Set ℝ := {x : ℝ | x^2 - x - 6 > 0}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {x : ℝ | (3 < x ∧ x ≤ 7) ∨ (-4 ≤ x ∧ x < -2)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2710_271054


namespace NUMINAMATH_CALUDE_tank_emptied_in_three_minutes_l2710_271064

/-- Represents the time to empty a water tank given specific conditions. -/
def time_to_empty_tank (initial_fill : ℚ) (fill_rate : ℚ) (empty_rate : ℚ) : ℚ :=
  initial_fill / (empty_rate - fill_rate)

/-- Theorem stating that under given conditions, the tank will be emptied in 3 minutes. -/
theorem tank_emptied_in_three_minutes :
  let initial_fill : ℚ := 1/5
  let fill_rate : ℚ := 1/10
  let empty_rate : ℚ := 1/6
  time_to_empty_tank initial_fill fill_rate empty_rate = 3 := by
  sorry

#eval time_to_empty_tank (1/5) (1/10) (1/6)

end NUMINAMATH_CALUDE_tank_emptied_in_three_minutes_l2710_271064


namespace NUMINAMATH_CALUDE_three_numbers_sum_and_ratio_l2710_271053

theorem three_numbers_sum_and_ratio (A B C : ℝ) : 
  A + B + C = 36 →
  (A + B) / (B + C) = 2 / 3 →
  (B + C) / (A + C) = 3 / 4 →
  A = 12 ∧ B = 4 ∧ C = 20 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_and_ratio_l2710_271053


namespace NUMINAMATH_CALUDE_smaller_number_problem_l2710_271041

theorem smaller_number_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 45) (h4 : y = 4 * x) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l2710_271041


namespace NUMINAMATH_CALUDE_path_area_and_cost_l2710_271014

/-- Represents the dimensions of a rectangular field with a path around it -/
structure FieldWithPath where
  fieldLength : ℝ
  fieldWidth : ℝ
  pathWidth : ℝ

/-- Calculates the area of the path around a rectangular field -/
def areaOfPath (f : FieldWithPath) : ℝ :=
  (f.fieldLength + 2 * f.pathWidth) * (f.fieldWidth + 2 * f.pathWidth) - f.fieldLength * f.fieldWidth

/-- Calculates the cost of constructing the path given the cost per square meter -/
def costOfPath (f : FieldWithPath) (costPerSqm : ℝ) : ℝ :=
  areaOfPath f * costPerSqm

/-- Theorem stating the area of the path and its construction cost for the given field dimensions -/
theorem path_area_and_cost (f : FieldWithPath) (h1 : f.fieldLength = 65) (h2 : f.fieldWidth = 55) 
    (h3 : f.pathWidth = 2.5) (h4 : costPerSqm = 2) : 
    areaOfPath f = 625 ∧ costOfPath f costPerSqm = 1250 := by
  sorry

end NUMINAMATH_CALUDE_path_area_and_cost_l2710_271014


namespace NUMINAMATH_CALUDE_smallest_n_purple_candy_l2710_271096

def orange_candy : ℕ := 10
def yellow_candy : ℕ := 16
def gray_candy : ℕ := 18
def purple_candy_cost : ℕ := 18

theorem smallest_n_purple_candy : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (total_cost : ℕ), 
    total_cost = orange_candy * n ∧
    total_cost = yellow_candy * n ∧
    total_cost = gray_candy * n ∧
    total_cost = purple_candy_cost * n) ∧
  (∀ (m : ℕ), m < n → 
    ¬(∃ (total_cost : ℕ), 
      total_cost = orange_candy * m ∧
      total_cost = yellow_candy * m ∧
      total_cost = gray_candy * m ∧
      total_cost = purple_candy_cost * m)) ∧
  n = 40 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_purple_candy_l2710_271096


namespace NUMINAMATH_CALUDE_population_growth_duration_l2710_271001

/-- Proves that given specific birth and death rates and a total net increase,
    the duration of the period is 12 hours. -/
theorem population_growth_duration
  (birth_rate : ℝ)
  (death_rate : ℝ)
  (total_net_increase : ℝ)
  (h1 : birth_rate = 2)
  (h2 : death_rate = 1)
  (h3 : total_net_increase = 86400)
  : (total_net_increase / (birth_rate - death_rate)) / 3600 = 12 := by
  sorry


end NUMINAMATH_CALUDE_population_growth_duration_l2710_271001


namespace NUMINAMATH_CALUDE_right_triangle_in_sets_l2710_271051

/-- Checks if three numbers can form a right-angled triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- The given sets of numbers --/
def number_sets : List (ℕ × ℕ × ℕ) :=
  [(1, 2, 3), (2, 3, 4), (3, 4, 5), (9, 13, 17)]

theorem right_triangle_in_sets :
  ∃! (a b c : ℕ), (a, b, c) ∈ number_sets ∧ is_right_triangle a b c :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_in_sets_l2710_271051


namespace NUMINAMATH_CALUDE_multiply_993_879_l2710_271066

theorem multiply_993_879 : 993 * 879 = 872847 := by
  -- Define the method
  let a := 993
  let b := 879
  let n := 7
  
  -- Step 1: Subtract n from b
  let b_minus_n := b - n
  
  -- Step 2: Add n to a
  let a_plus_n := a + n
  
  -- Step 3: Multiply results of steps 1 and 2
  let product_step3 := b_minus_n * a_plus_n
  
  -- Step 4: Calculate the difference
  let diff := a - b_minus_n
  
  -- Step 5: Multiply the difference by n
  let product_step5 := diff * n
  
  -- Step 6: Add results of steps 3 and 5
  let result := product_step3 + product_step5
  
  -- Prove that the result equals 872847
  sorry

end NUMINAMATH_CALUDE_multiply_993_879_l2710_271066


namespace NUMINAMATH_CALUDE_quadratic_unique_root_l2710_271069

/-- A function that represents the quadratic equation (m-4)x^2 - 2mx - m - 6 -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 4) * x^2 - 2 * m * x - m - 6

/-- Condition for the quadratic function to have exactly one root -/
def has_unique_root (m : ℝ) : Prop :=
  (∃ x : ℝ, f m x = 0) ∧ (∀ x y : ℝ, f m x = 0 → f m y = 0 → x = y)

theorem quadratic_unique_root (m : ℝ) :
  has_unique_root m → m = -4 ∨ m = 3 ∨ m = 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_unique_root_l2710_271069


namespace NUMINAMATH_CALUDE_driver_speed_problem_l2710_271073

theorem driver_speed_problem (v : ℝ) : 
  (v * 1 = (v + 18) * (2/3)) → v = 36 :=
by sorry

end NUMINAMATH_CALUDE_driver_speed_problem_l2710_271073


namespace NUMINAMATH_CALUDE_pages_per_sheet_calculation_l2710_271061

/-- The number of stories John writes per week -/
def stories_per_week : ℕ := 3

/-- The number of pages in each story -/
def pages_per_story : ℕ := 50

/-- The number of weeks John writes -/
def weeks : ℕ := 12

/-- The number of reams of paper John uses over 12 weeks -/
def reams_used : ℕ := 3

/-- The number of sheets in each ream of paper -/
def sheets_per_ream : ℕ := 500

/-- Calculate the number of pages each sheet of paper can hold -/
def pages_per_sheet : ℕ := 1

theorem pages_per_sheet_calculation :
  pages_per_sheet = 1 :=
by sorry

end NUMINAMATH_CALUDE_pages_per_sheet_calculation_l2710_271061


namespace NUMINAMATH_CALUDE_rubber_duck_race_l2710_271046

theorem rubber_duck_race (regular_price : ℚ) (large_price : ℚ) (regular_sold : ℕ) (total_raised : ℚ) :
  regular_price = 3 →
  large_price = 5 →
  regular_sold = 221 →
  total_raised = 1588 →
  ∃ (large_sold : ℕ), large_sold = 185 ∧ 
    regular_price * regular_sold + large_price * large_sold = total_raised :=
by sorry

end NUMINAMATH_CALUDE_rubber_duck_race_l2710_271046


namespace NUMINAMATH_CALUDE_hexagon_diagonals_intersect_l2710_271078

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A hexagon in a 2D plane -/
structure Hexagon where
  vertices : Fin 6 → ℝ × ℝ

/-- A line dividing a side of a triangle into three equal parts -/
def dividingLine (T : Triangle) (vertex : Fin 3) : ℝ × ℝ → ℝ × ℝ → Prop :=
  sorry

/-- The hexagon formed by the dividing lines -/
def formHexagon (T : Triangle) : Hexagon :=
  sorry

/-- The diagonals of a hexagon -/
def diagonals (H : Hexagon) : List (ℝ × ℝ → ℝ × ℝ → Prop) :=
  sorry

/-- The intersection point of lines -/
def intersectionPoint (lines : List (ℝ × ℝ → ℝ × ℝ → Prop)) : Option (ℝ × ℝ) :=
  sorry

/-- Main theorem -/
theorem hexagon_diagonals_intersect (T : Triangle) :
  let H := formHexagon T
  let diag := diagonals H
  ∃ p : ℝ × ℝ, intersectionPoint diag = some p :=
by sorry

end NUMINAMATH_CALUDE_hexagon_diagonals_intersect_l2710_271078


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2710_271005

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x : ℝ | |x - 1| ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2710_271005


namespace NUMINAMATH_CALUDE_min_draw_count_correct_l2710_271027

/-- Represents a box of colored balls -/
structure ColoredBallBox where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat

/-- The setup of the two boxes -/
def box1 : ColoredBallBox := ⟨40, 30, 25, 15⟩
def box2 : ColoredBallBox := ⟨35, 25, 20, 0⟩

/-- The target number of balls of a single color -/
def targetCount : Nat := 20

/-- The minimum number of balls to draw -/
def minDrawCount : Nat := 73

/-- Theorem stating the minimum number of balls to draw -/
theorem min_draw_count_correct : 
  ∀ (draw : Nat), draw < minDrawCount → 
  ∃ (redCount greenCount yellowCount blueCount : Nat),
    redCount < targetCount ∧
    greenCount < targetCount ∧
    yellowCount < targetCount ∧
    blueCount < targetCount ∧
    redCount + greenCount + yellowCount + blueCount = draw ∧
    redCount ≤ box1.red + box2.red ∧
    greenCount ≤ box1.green + box2.green ∧
    yellowCount ≤ box1.yellow + box2.yellow ∧
    blueCount ≤ box1.blue + box2.blue :=
by sorry

#check min_draw_count_correct

end NUMINAMATH_CALUDE_min_draw_count_correct_l2710_271027


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l2710_271020

/-- A geometric sequence with given conditions -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_formula 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_a2 : a 2 = 6) 
  (h_a5 : a 5 = 162) : 
  ∀ n : ℕ, a n = 2 * 3^(n - 1) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l2710_271020


namespace NUMINAMATH_CALUDE_track_length_l2710_271070

/-- The length of a track AB given specific meeting points of two athletes --/
theorem track_length (v₁ v₂ : ℝ) (h₁ : v₁ > 0) (h₂ : v₂ > 0) : 
  let x := (v₁ + v₂) * 300 / v₂
  (300 / v₁ = (x - 300) / v₂) ∧ ((x + 100) / v₁ = (x - 100) / v₂) → x = 500 := by
  sorry

#check track_length

end NUMINAMATH_CALUDE_track_length_l2710_271070


namespace NUMINAMATH_CALUDE_half_x_sixth_y_seventh_l2710_271093

theorem half_x_sixth_y_seventh (x y : ℚ) (hx : x = 3/4) (hy : y = 4/3) :
  (1/2) * x^6 * y^7 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_half_x_sixth_y_seventh_l2710_271093


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l2710_271062

theorem arithmetic_series_sum (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) (n : ℕ) :
  a₁ = -300 →
  aₙ = 309 →
  d = 3 →
  n = (aₙ - a₁) / d + 1 →
  (n : ℤ) * (a₁ + aₙ) / 2 = 918 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_l2710_271062


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l2710_271091

/-- Given a parallelogram with area 98 sq m and altitude twice the base, prove the base is 7 m -/
theorem parallelogram_base_length : 
  ∀ (base altitude : ℝ), 
  (base * altitude = 98) →  -- Area of parallelogram
  (altitude = 2 * base) →   -- Altitude is twice the base
  base = 7 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l2710_271091


namespace NUMINAMATH_CALUDE_square_difference_401_399_l2710_271021

theorem square_difference_401_399 : 401^2 - 399^2 = 1600 := by sorry

end NUMINAMATH_CALUDE_square_difference_401_399_l2710_271021


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l2710_271003

theorem difference_of_squares_example : (23 + 12)^2 - (23 - 12)^2 = 1104 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l2710_271003


namespace NUMINAMATH_CALUDE_sum_of_mean_and_median_l2710_271057

def number_set : List ℕ := [1, 2, 3, 0, 1]

def median (l : List ℕ) : ℚ := sorry

def mean (l : List ℕ) : ℚ := sorry

theorem sum_of_mean_and_median :
  median number_set + mean number_set = 12/5 := by sorry

end NUMINAMATH_CALUDE_sum_of_mean_and_median_l2710_271057


namespace NUMINAMATH_CALUDE_saree_price_calculation_l2710_271065

theorem saree_price_calculation (P : ℝ) : 
  (P * (1 - 0.20) * (1 - 0.05) = 133) → P = 175 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_calculation_l2710_271065


namespace NUMINAMATH_CALUDE_largest_number_l2710_271008

theorem largest_number : 
  let a := 0.989
  let b := 0.9098
  let c := 0.9899
  let d := 0.9009
  let e := 0.9809
  c > a ∧ c > b ∧ c > d ∧ c > e := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l2710_271008


namespace NUMINAMATH_CALUDE_rem_prime_specific_value_l2710_271085

/-- Modified remainder function -/
def rem' (x y : ℚ) : ℚ := x - y * ⌊x / (2 * y)⌋

/-- Theorem stating the value of rem'(5/9, -3/7) -/
theorem rem_prime_specific_value : rem' (5/9) (-3/7) = 62/63 := by
  sorry

end NUMINAMATH_CALUDE_rem_prime_specific_value_l2710_271085


namespace NUMINAMATH_CALUDE_sum_of_integers_with_lcm_gcd_l2710_271072

theorem sum_of_integers_with_lcm_gcd (m n : ℕ) : 
  m > 50 → 
  n > 50 → 
  Nat.lcm m n = 480 → 
  Nat.gcd m n = 12 → 
  m + n = 156 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_with_lcm_gcd_l2710_271072


namespace NUMINAMATH_CALUDE_three_intersecting_lines_l2710_271033

/-- The parabola defined by y² = 3x -/
def parabola (x y : ℝ) : Prop := y^2 = 3*x

/-- A point lies on a line through (0, 2) -/
def line_through_A (m : ℝ) (x y : ℝ) : Prop := y = m*x + 2

/-- A line intersects the parabola at exactly one point -/
def single_intersection (m : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, parabola p.1 p.2 ∧ line_through_A m p.1 p.2

/-- There are exactly 3 lines through (0, 2) that intersect the parabola at one point -/
theorem three_intersecting_lines : ∃! l : Finset ℝ, 
  l.card = 3 ∧ (∀ m ∈ l, single_intersection m) ∧
  (∀ m : ℝ, single_intersection m → m ∈ l) :=
sorry

end NUMINAMATH_CALUDE_three_intersecting_lines_l2710_271033


namespace NUMINAMATH_CALUDE_coach_a_basketballs_l2710_271016

/-- The number of basketballs Coach A bought -/
def num_basketballs : ℕ := 10

/-- The cost of each basketball in dollars -/
def basketball_cost : ℚ := 29

/-- The total cost of Coach B's purchases in dollars -/
def coach_b_cost : ℚ := 14 * 2.5 + 18

/-- The difference in cost between Coach A and Coach B's purchases in dollars -/
def cost_difference : ℚ := 237

theorem coach_a_basketballs :
  basketball_cost * num_basketballs = coach_b_cost + cost_difference := by
  sorry


end NUMINAMATH_CALUDE_coach_a_basketballs_l2710_271016


namespace NUMINAMATH_CALUDE_bits_of_base16_ABCD_l2710_271099

/-- The number of bits in the binary representation of a base-16 number ABCD₁₆ --/
theorem bits_of_base16_ABCD : ∃ (A B C D : ℕ), 
  A < 16 ∧ B < 16 ∧ C < 16 ∧ D < 16 →
  let base16_value := A * 16^3 + B * 16^2 + C * 16^1 + D * 16^0
  let binary_repr := Nat.bits base16_value
  binary_repr.length = 16 := by
  sorry

end NUMINAMATH_CALUDE_bits_of_base16_ABCD_l2710_271099


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2710_271032

theorem cyclic_sum_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h_sum : x^2 + y^2 + z^2 = 3) :
  (x^2 + y*z) / (x^2 + y*z + 1) + (y^2 + z*x) / (y^2 + z*x + 1) + (z^2 + x*y) / (z^2 + x*y + 1) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2710_271032


namespace NUMINAMATH_CALUDE_axis_of_symmetry_translated_sine_l2710_271044

theorem axis_of_symmetry_translated_sine (k : ℤ) :
  let f : ℝ → ℝ := λ x => Real.sin (2 * x + π / 3)
  ∃ (x : ℝ), x = k * π / 2 + π / 12 ∧
    ∀ (y : ℝ), f (x - y) = f (x + y) := by
  sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_translated_sine_l2710_271044


namespace NUMINAMATH_CALUDE_intersection_range_l2710_271050

/-- The set M representing an ellipse -/
def M : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.2^2 = 3}

/-- The set N representing a line with slope m and y-intercept b -/
def N (m b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m*p.1 + b}

/-- Theorem stating the range of b for which M and N always intersect -/
theorem intersection_range :
  (∀ m : ℝ, (M ∩ N m b).Nonempty) ↔ b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) := by
  sorry

#check intersection_range

end NUMINAMATH_CALUDE_intersection_range_l2710_271050


namespace NUMINAMATH_CALUDE_max_songs_is_56_l2710_271018

/-- Calculates the maximum number of songs that can be played given the specified conditions -/
def max_songs_played (short_songs : ℕ) (long_songs : ℕ) (short_duration : ℕ) (long_duration : ℕ) (total_time : ℕ) : ℕ :=
  let time_for_short := min (short_songs * short_duration) total_time
  let remaining_time := total_time - time_for_short
  let short_count := time_for_short / short_duration
  let long_count := remaining_time / long_duration
  short_count + long_count

/-- Theorem stating that the maximum number of songs that can be played is 56 -/
theorem max_songs_is_56 : 
  max_songs_played 50 50 3 5 (3 * 60) = 56 := by
  sorry

end NUMINAMATH_CALUDE_max_songs_is_56_l2710_271018


namespace NUMINAMATH_CALUDE_car_travel_distance_l2710_271043

theorem car_travel_distance (rate : ℚ) (time : ℚ) : 
  rate = 3 / 4 → time = 2 → rate * time * 60 = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_distance_l2710_271043


namespace NUMINAMATH_CALUDE_evaluate_F_of_f_l2710_271026

-- Define the functions f and F
def f (a : ℝ) : ℝ := a^2 + 1
def F (a b : ℝ) : ℝ := a * b + b^2

-- State the theorem
theorem evaluate_F_of_f : F 4 (f 3) = 140 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_F_of_f_l2710_271026


namespace NUMINAMATH_CALUDE_always_two_real_roots_unique_m_value_l2710_271082

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 - 4*m*x + 3*m^2 = 0

-- Theorem 1: The equation always has two real roots
theorem always_two_real_roots (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂ :=
sorry

-- Theorem 2: When m > 0 and the difference between roots is 2, m = 1
theorem unique_m_value (m : ℝ) (h₁ : m > 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂ ∧ x₁ - x₂ = 2) →
  m = 1 :=
sorry

end NUMINAMATH_CALUDE_always_two_real_roots_unique_m_value_l2710_271082


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2710_271023

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, x > a → x > 2) ∧ (∃ x, x > 2 ∧ x ≤ a) ↔ a > 2 :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2710_271023


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2710_271060

theorem necessary_but_not_sufficient :
  (∀ x : ℝ, (-3 ≤ x ∧ x ≤ 1) → (x ≤ 2 ∨ x ≥ 3)) ∧
  (∃ x : ℝ, (x ≤ 2 ∨ x ≥ 3) ∧ ¬(-3 ≤ x ∧ x ≤ 1)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2710_271060


namespace NUMINAMATH_CALUDE_neg_three_point_fourteen_gt_neg_pi_l2710_271076

theorem neg_three_point_fourteen_gt_neg_pi : -3.14 > -Real.pi := by sorry

end NUMINAMATH_CALUDE_neg_three_point_fourteen_gt_neg_pi_l2710_271076


namespace NUMINAMATH_CALUDE_jim_scuba_diving_bags_l2710_271035

/-- The number of smaller bags Jim found while scuba diving -/
def number_of_smaller_bags : ℕ := by sorry

theorem jim_scuba_diving_bags :
  let hours_diving : ℕ := 8
  let coins_per_hour : ℕ := 25
  let treasure_chest_coins : ℕ := 100
  let total_coins := hours_diving * coins_per_hour
  let remaining_coins := total_coins - treasure_chest_coins
  let coins_per_smaller_bag := treasure_chest_coins / 2
  number_of_smaller_bags = remaining_coins / coins_per_smaller_bag :=
by sorry

end NUMINAMATH_CALUDE_jim_scuba_diving_bags_l2710_271035


namespace NUMINAMATH_CALUDE_inequalities_for_positive_reals_l2710_271022

theorem inequalities_for_positive_reals (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  a * b ≤ 1 ∧ a^2 + b^2 ≥ 2 ∧ Real.sqrt a + Real.sqrt b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequalities_for_positive_reals_l2710_271022


namespace NUMINAMATH_CALUDE_f_derivative_at_pi_third_l2710_271056

noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.sqrt 3 * Real.sin x

theorem f_derivative_at_pi_third : 
  (deriv f) (π / 3) = 0 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_pi_third_l2710_271056


namespace NUMINAMATH_CALUDE_equation_solutions_l2710_271028

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 - Real.sqrt 2 ∧ x₂ = 2 + Real.sqrt 2 ∧
    x₁^2 - 4*x₁ = 4 ∧ x₂^2 - 4*x₂ = 4) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -5 ∧ x₂ = 2 ∧
    (x₁ + 2)*(x₁ + 1) = 12 ∧ (x₂ + 2)*(x₂ + 1) = 12) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 5/2 ∧ x₂ = 5 ∧
    0.2*x₁^2 + 5/2 = 3/2*x₁ ∧ 0.2*x₂^2 + 5/2 = 3/2*x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2710_271028


namespace NUMINAMATH_CALUDE_sum_equals_1300_l2710_271097

/-- Converts a number from base 15 to base 10 -/
def base15ToBase10 (n : Nat) : Nat :=
  (n / 100) * 225 + ((n / 10) % 10) * 15 + (n % 10)

/-- Converts a number from base 7 to base 10, where 'A' represents 10 -/
def base7ToBase10 (n : Nat) : Nat :=
  (n / 100) * 49 + ((n / 10) % 10) * 7 + (n % 10)

/-- Theorem stating that the sum of 537 (base 15) and 1A4 (base 7) equals 1300 in base 10 -/
theorem sum_equals_1300 : 
  base15ToBase10 537 + base7ToBase10 194 = 1300 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_1300_l2710_271097


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2710_271052

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_product : a 1 * a 7 = 36) :
  a 4 = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2710_271052


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_sixth_l2710_271071

theorem tan_alpha_plus_pi_sixth (α : Real) (h : α > 0) (h' : α < π / 2) 
  (h_eq : Real.sqrt 3 * Real.sin α + Real.cos α = 8 / 5) : 
  Real.tan (α + π / 6) = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_sixth_l2710_271071


namespace NUMINAMATH_CALUDE_walnut_trees_remaining_l2710_271002

/-- The number of walnut trees remaining after some are cut down -/
def remaining_walnut_trees (initial : ℕ) (cut_down : ℕ) : ℕ :=
  initial - cut_down

/-- Theorem stating that the number of remaining walnut trees is 29 -/
theorem walnut_trees_remaining :
  remaining_walnut_trees 42 13 = 29 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_remaining_l2710_271002


namespace NUMINAMATH_CALUDE_identities_proof_l2710_271025

theorem identities_proof (a : ℝ) (n k : ℤ) : 
  ((-a^3 * (-a)^3)^2 + (-a^2 * (-a)^2)^3 = 0) ∧ 
  ((-1:ℝ)^n * a^(n+k) = (-a)^n * a^k) := by
  sorry

end NUMINAMATH_CALUDE_identities_proof_l2710_271025


namespace NUMINAMATH_CALUDE_product_digit_sum_l2710_271077

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The problem statement -/
theorem product_digit_sum :
  let c : ℕ := 777
  let d : ℕ := 444
  sum_of_digits (7 * c * d) = 27 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_l2710_271077


namespace NUMINAMATH_CALUDE_inverse_difference_equals_negative_one_l2710_271006

theorem inverse_difference_equals_negative_one 
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : 
  1 / x - 1 / y = -1 :=
sorry

end NUMINAMATH_CALUDE_inverse_difference_equals_negative_one_l2710_271006


namespace NUMINAMATH_CALUDE_M_intersect_N_l2710_271040

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x^2 = x}

theorem M_intersect_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_l2710_271040


namespace NUMINAMATH_CALUDE_pipe_cut_theorem_l2710_271081

theorem pipe_cut_theorem (total_length : ℝ) (difference : ℝ) (shorter_piece : ℝ) :
  total_length = 68 →
  difference = 12 →
  total_length = shorter_piece + (shorter_piece + difference) →
  shorter_piece = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_pipe_cut_theorem_l2710_271081


namespace NUMINAMATH_CALUDE_willy_distance_theorem_l2710_271030

/-- Represents the distances from Willy to the corners of the square lot -/
structure Distances where
  d₁ : ℝ
  d₂ : ℝ
  d₃ : ℝ
  d₄ : ℝ

/-- The conditions given in the problem -/
def satisfies_conditions (d : Distances) : Prop :=
  d.d₁ < d.d₂ ∧ d.d₂ < d.d₄ ∧ d.d₄ < d.d₃ ∧
  d.d₂ = (d.d₁ + d.d₃) / 2 ∧
  d.d₄ ^ 2 = d.d₂ * d.d₃

/-- The theorem to be proved -/
theorem willy_distance_theorem (d : Distances) (h : satisfies_conditions d) :
  d.d₁ ^ 2 = (4 * d.d₁ * d.d₃ - d.d₃ ^ 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_willy_distance_theorem_l2710_271030


namespace NUMINAMATH_CALUDE_evaluate_expression_l2710_271098

theorem evaluate_expression (b x : ℝ) (h : x = b + 10) : 2*x - b + 5 = b + 25 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2710_271098


namespace NUMINAMATH_CALUDE_mans_usual_time_to_office_l2710_271017

/-- Proves that if a man walks at 3/4 of his usual pace and arrives at his office 20 minutes late, 
    his usual time to reach the office is 80 minutes. -/
theorem mans_usual_time_to_office (usual_pace : ℝ) (usual_time : ℝ) 
    (h1 : usual_pace > 0) (h2 : usual_time > 0) : 
    (3 / 4 * usual_pace) * (usual_time + 20) = usual_pace * usual_time → 
    usual_time = 80 := by
  sorry


end NUMINAMATH_CALUDE_mans_usual_time_to_office_l2710_271017


namespace NUMINAMATH_CALUDE_olivers_mom_money_l2710_271038

/-- Calculates the amount of money Oliver's mom gave him -/
theorem olivers_mom_money (initial : ℕ) (spent : ℕ) (final : ℕ) : 
  initial - spent + (final - (initial - spent)) = final ∧ 
  final - (initial - spent) = 32 :=
by
  sorry

#check olivers_mom_money 33 4 61

end NUMINAMATH_CALUDE_olivers_mom_money_l2710_271038


namespace NUMINAMATH_CALUDE_exponential_distribution_expected_value_l2710_271013

/-- The expected value of an exponentially distributed random variable -/
theorem exponential_distribution_expected_value (α : ℝ) (hα : α > 0) :
  let X : ℝ → ℝ := λ x => if x ≥ 0 then α * Real.exp (-α * x) else 0
  ∫ x in Set.Ici 0, x * X x = 1 / α :=
sorry

end NUMINAMATH_CALUDE_exponential_distribution_expected_value_l2710_271013


namespace NUMINAMATH_CALUDE_area_ratio_of_inner_triangle_l2710_271083

/-- Given a triangle with area T, if we divide each side of the triangle in the ratio of 1:2
    (starting from each vertex) and form a new triangle by connecting these points,
    the area of the new triangle S is related to the area of the original triangle T
    by the equation: S / T = 1 / 9 -/
theorem area_ratio_of_inner_triangle (T : ℝ) (S : ℝ) (h : T > 0) :
  (∀ (side : ℝ), ∃ (new_side : ℝ), new_side = side / 3) →
  S / T = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_inner_triangle_l2710_271083


namespace NUMINAMATH_CALUDE_octagon_side_length_l2710_271088

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a rectangle -/
structure Rectangle :=
  (P Q R S : Point)

/-- Represents an octagon -/
structure Octagon :=
  (A B C D E F G H : Point)

/-- Checks if an octagon is equilateral -/
def is_equilateral (oct : Octagon) : Prop := sorry

/-- Checks if an octagon is convex -/
def is_convex (oct : Octagon) : Prop := sorry

/-- Checks if an octagon is inscribed in a rectangle -/
def is_inscribed (oct : Octagon) (rect : Rectangle) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if a number is not divisible by the square of any prime -/
def not_divisible_by_square_of_prime (n : ℕ) : Prop := sorry

theorem octagon_side_length (rect : Rectangle) (oct : Octagon) :
  distance rect.P rect.Q = 8 →
  distance rect.Q rect.R = 6 →
  is_inscribed oct rect →
  is_equilateral oct →
  is_convex oct →
  distance oct.A rect.P = distance oct.B rect.Q →
  distance oct.A rect.P < 4 →
  ∃ (k m n : ℕ), 
    distance oct.A oct.B = k + m * Real.sqrt n ∧ 
    not_divisible_by_square_of_prime n ∧
    k + m + n = 7 :=
by sorry

end NUMINAMATH_CALUDE_octagon_side_length_l2710_271088


namespace NUMINAMATH_CALUDE_color_film_fraction_l2710_271090

/-- Given a film festival selection process, this theorem proves the fraction of selected films that are in color. -/
theorem color_film_fraction (x y : ℝ) (x_pos : 0 < x) (y_pos : 0 < y) : 
  let total_bw : ℝ := 20 * x
  let total_color : ℝ := 4 * y
  let selected_bw : ℝ := (y / x) * total_bw / 100
  let selected_color : ℝ := total_color
  (selected_color) / (selected_bw + selected_color) = 20 / (x + 20) := by
  sorry

end NUMINAMATH_CALUDE_color_film_fraction_l2710_271090


namespace NUMINAMATH_CALUDE_max_hollow_cube_volume_l2710_271058

/-- The number of available unit cubes --/
def available_cubes : ℕ := 1000

/-- Function to calculate the number of cubes used for a given side length --/
def cubes_used (x : ℕ) : ℕ :=
  2 * x^2 + 2 * x * (x - 2) + 2 * (x - 2)^2

/-- The maximum side length that can be achieved --/
def max_side_length : ℕ := 13

/-- Theorem stating the maximum volume that can be achieved --/
theorem max_hollow_cube_volume :
  (∀ x : ℕ, cubes_used x ≤ available_cubes → x ≤ max_side_length) ∧
  cubes_used max_side_length ≤ available_cubes ∧
  max_side_length^3 = 2197 :=
sorry

end NUMINAMATH_CALUDE_max_hollow_cube_volume_l2710_271058


namespace NUMINAMATH_CALUDE_infinitely_many_primes_2_mod_3_l2710_271086

theorem infinitely_many_primes_2_mod_3 : Set.Infinite {p : ℕ | Nat.Prime p ∧ p % 3 = 2} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_2_mod_3_l2710_271086


namespace NUMINAMATH_CALUDE_circle_tangent_y_intercept_l2710_271047

/-- Two circles with given centers and radii have a common external tangent with y-intercept 135/28 -/
theorem circle_tangent_y_intercept :
  ∃ (m b : ℝ),
    m > 0 ∧
    b = 135 / 28 ∧
    ∀ (x y : ℝ),
      (y = m * x + b) →
      ((x - 1)^2 + (y - 3)^2 = 3^2 ∨ (x - 10)^2 + (y - 8)^2 = 6^2) →
      ∀ (x' y' : ℝ),
        ((x' - 1)^2 + (y' - 3)^2 < 3^2 ∧ (x' - 10)^2 + (y' - 8)^2 < 6^2) →
        (y' ≠ m * x' + b) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_y_intercept_l2710_271047


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_tangent_l2710_271031

/-- The value of m for which the ellipse 3x^2 + 9y^2 = 9 and 
    the hyperbola (x-2)^2 - m(y+1)^2 = 1 are tangent -/
theorem ellipse_hyperbola_tangent : 
  ∃! m : ℝ, ∀ x y : ℝ, 
    (3 * x^2 + 9 * y^2 = 9 ∧ (x - 2)^2 - m * (y + 1)^2 = 1) →
    (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y ∧ 
      3 * p.1^2 + 9 * p.2^2 = 9 ∧ 
      (p.1 - 2)^2 - m * (p.2 + 1)^2 = 1) →
    m = 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_tangent_l2710_271031


namespace NUMINAMATH_CALUDE_initially_calculated_average_height_l2710_271059

theorem initially_calculated_average_height
  (n : ℕ)
  (wrong_height actual_height : ℝ)
  (actual_average : ℝ)
  (h1 : n = 35)
  (h2 : wrong_height = 166)
  (h3 : actual_height = 106)
  (h4 : actual_average = 183) :
  let initially_calculated_average := 
    (n * actual_average - (wrong_height - actual_height)) / n
  initially_calculated_average = 181 := by
sorry

end NUMINAMATH_CALUDE_initially_calculated_average_height_l2710_271059


namespace NUMINAMATH_CALUDE_f_equals_g_l2710_271024

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := (x^6)^(1/3)

-- Theorem statement
theorem f_equals_g : f = g := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l2710_271024


namespace NUMINAMATH_CALUDE_min_value_h_l2710_271063

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

noncomputable def g (x : ℝ) : ℝ := x * Real.exp x

noncomputable def h (x : ℝ) : ℝ := f x / g x

theorem min_value_h :
  ∃ (min : ℝ), min = 2 / Real.pi ∧
  ∀ x ∈ Set.Ioo 0 (Real.pi / 2), h x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_h_l2710_271063


namespace NUMINAMATH_CALUDE_carl_pink_hats_solution_l2710_271019

/-- The number of pink hard hats Carl took away from the truck -/
def carl_pink_hats : ℕ := sorry

theorem carl_pink_hats_solution : carl_pink_hats = 4 := by
  have initial_pink : ℕ := 26
  have initial_green : ℕ := 15
  have initial_yellow : ℕ := 24
  have john_pink : ℕ := 6
  have john_green : ℕ := 2 * john_pink
  have remaining_hats : ℕ := 43

  sorry

end NUMINAMATH_CALUDE_carl_pink_hats_solution_l2710_271019


namespace NUMINAMATH_CALUDE_set_M_membership_l2710_271068

def M : Set ℕ := {x : ℕ | (1 : ℚ) / (x - 2 : ℚ) ≤ 0}

theorem set_M_membership :
  1 ∈ M ∧ 2 ∉ M ∧ 3 ∉ M ∧ 4 ∉ M :=
by sorry

end NUMINAMATH_CALUDE_set_M_membership_l2710_271068


namespace NUMINAMATH_CALUDE_goose_eggs_count_l2710_271045

theorem goose_eggs_count (total_eggs : ℕ) : 
  (1 : ℚ) / 3 * (3 : ℚ) / 4 * (2 : ℚ) / 5 * total_eggs = 120 →
  total_eggs = 1200 := by
  sorry

end NUMINAMATH_CALUDE_goose_eggs_count_l2710_271045


namespace NUMINAMATH_CALUDE_triangle_special_case_l2710_271087

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the circumcenter and orthocenter
def circumcenter (t : Triangle) : ℝ × ℝ := sorry
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle at vertex B
def angle_B (t : Triangle) : ℝ := sorry

-- Main theorem
theorem triangle_special_case (t : Triangle) :
  let O := circumcenter t
  let H := orthocenter t
  distance t.B O = distance t.B H →
  (angle_B t = 60 ∨ angle_B t = 120) :=
by sorry

end NUMINAMATH_CALUDE_triangle_special_case_l2710_271087


namespace NUMINAMATH_CALUDE_smallest_sum_with_gcd_lcm_condition_l2710_271095

theorem smallest_sum_with_gcd_lcm_condition (a b : ℕ+) : 
  (Nat.gcd a b + Nat.lcm a b = 3 * (a + b)) → 
  (∀ c d : ℕ+, (Nat.gcd c d + Nat.lcm c d = 3 * (c + d)) → (a + b ≤ c + d)) → 
  a + b = 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_with_gcd_lcm_condition_l2710_271095


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_15_6_minus_9_6_l2710_271012

-- Define the valuation function v₂
def v₂ (n : ℤ) : ℕ := (n.natAbs.factors.count 2)

-- State the theorem
theorem largest_power_of_two_dividing_15_6_minus_9_6 :
  2^(v₂ (15^6 - 9^6)) = 32 := by sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_15_6_minus_9_6_l2710_271012


namespace NUMINAMATH_CALUDE_hypotenuse_length_l2710_271015

theorem hypotenuse_length (x y : ℝ) : 
  2 * x^2 - 8 * x + 7 = 0 →
  2 * y^2 - 8 * y + 7 = 0 →
  x ≠ y →
  x > 0 →
  y > 0 →
  x^2 + y^2 = 3^2 :=
by sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l2710_271015


namespace NUMINAMATH_CALUDE_anna_cupcake_earnings_l2710_271009

/-- Calculates the earnings from selling cupcakes given the number of trays, cupcakes per tray,
    price per cupcake, and fraction of cupcakes sold. -/
def cupcake_earnings (num_trays : ℕ) (cupcakes_per_tray : ℕ) (price_per_cupcake : ℚ) (fraction_sold : ℚ) : ℚ :=
  (num_trays * cupcakes_per_tray : ℚ) * fraction_sold * price_per_cupcake

/-- Proves that Anna's earnings from selling cupcakes equal $96 given the specified conditions. -/
theorem anna_cupcake_earnings :
  cupcake_earnings 4 20 2 (3/5) = 96 := by
  sorry

end NUMINAMATH_CALUDE_anna_cupcake_earnings_l2710_271009


namespace NUMINAMATH_CALUDE_janet_pill_intake_l2710_271037

/-- Represents Janet's pill intake schedule for a month --/
structure PillSchedule where
  multivitamins_per_day : ℕ
  calcium_first_two_weeks : ℕ
  calcium_last_two_weeks : ℕ
  weeks_in_month : ℕ

/-- Calculates the total number of pills Janet takes in a month --/
def total_pills (schedule : PillSchedule) : ℕ :=
  let days_per_period := schedule.weeks_in_month / 2 * 7
  let pills_first_two_weeks := (schedule.multivitamins_per_day + schedule.calcium_first_two_weeks) * days_per_period
  let pills_last_two_weeks := (schedule.multivitamins_per_day + schedule.calcium_last_two_weeks) * days_per_period
  pills_first_two_weeks + pills_last_two_weeks

/-- Theorem stating that Janet's total pill intake for the month is 112 --/
theorem janet_pill_intake :
  ∃ (schedule : PillSchedule),
    schedule.multivitamins_per_day = 2 ∧
    schedule.calcium_first_two_weeks = 3 ∧
    schedule.calcium_last_two_weeks = 1 ∧
    schedule.weeks_in_month = 4 ∧
    total_pills schedule = 112 := by
  sorry

end NUMINAMATH_CALUDE_janet_pill_intake_l2710_271037


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2710_271084

theorem fraction_evaluation : (2 + 1/2) / (1 - 3/4) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2710_271084


namespace NUMINAMATH_CALUDE_distance_A_B_l2710_271089

/-- The distance between points A(0, 0, 1) and B(0, 1, 0) in a spatial Cartesian coordinate system is √2. -/
theorem distance_A_B : Real.sqrt 2 = (Real.sqrt ((0 - 0)^2 + (1 - 0)^2 + (0 - 1)^2)) := by
  sorry

end NUMINAMATH_CALUDE_distance_A_B_l2710_271089


namespace NUMINAMATH_CALUDE_triharmonic_properties_l2710_271004

-- Define a triharmonic quadruple
def is_triharmonic (A B C D : ℝ × ℝ) : Prop :=
  (dist A B) * (dist C D) = (dist A C) * (dist B D) ∧
  (dist A B) * (dist C D) = (dist A D) * (dist B C)

-- Define concyclicity
def are_concyclic (A B C D : ℝ × ℝ) : Prop :=
  ∃ (O : ℝ × ℝ) (r : ℝ), r > 0 ∧
    dist O A = r ∧ dist O B = r ∧ dist O C = r ∧ dist O D = r

theorem triharmonic_properties
  (A B C D A1 B1 C1 D1 : ℝ × ℝ)
  (h1 : is_triharmonic A B C D)
  (h2 : is_triharmonic A1 B C D)
  (h3 : is_triharmonic A B1 C D)
  (h4 : is_triharmonic A B C1 D)
  (h5 : is_triharmonic A B C D1)
  (hA : A1 ≠ A) (hB : B1 ≠ B) (hC : C1 ≠ C) (hD : D1 ≠ D) :
  are_concyclic A B C1 D1 ∧ is_triharmonic A1 B1 C1 D1 := by
  sorry

end NUMINAMATH_CALUDE_triharmonic_properties_l2710_271004


namespace NUMINAMATH_CALUDE_swap_values_l2710_271034

/-- Swaps the values of two variables using an intermediate variable -/
theorem swap_values (a b : ℕ) : 
  let a_init := a
  let b_init := b
  let c := a_init
  let a_new := b_init
  let b_new := c
  (a_new = b_init ∧ b_new = a_init) := by sorry

end NUMINAMATH_CALUDE_swap_values_l2710_271034


namespace NUMINAMATH_CALUDE_soccer_club_girls_l2710_271042

theorem soccer_club_girls (total_members : ℕ) (attended_members : ℕ) 
  (h1 : total_members = 30)
  (h2 : attended_members = 18)
  (h3 : ∃ (boys girls : ℕ), boys + girls = total_members ∧ boys + girls / 3 = attended_members) :
  ∃ (girls : ℕ), girls = 18 ∧ ∃ (boys : ℕ), boys + girls = total_members := by
sorry

end NUMINAMATH_CALUDE_soccer_club_girls_l2710_271042


namespace NUMINAMATH_CALUDE_birthday_number_l2710_271029

theorem birthday_number (T : ℕ) (x y : ℕ+) : 
  200 < T → T < 225 → T^2 = 4 * 10000 + x * 1000 + y * 100 + 29 → T = 223 := by
sorry

end NUMINAMATH_CALUDE_birthday_number_l2710_271029


namespace NUMINAMATH_CALUDE_resulting_polygon_has_18_sides_l2710_271049

/-- Represents a regular polygon with a given number of sides. -/
structure RegularPolygon where
  sides : ℕ

/-- Represents the arrangement of polygons. -/
structure PolygonArrangement where
  pentagon : RegularPolygon
  triangle : RegularPolygon
  octagon : RegularPolygon
  hexagon : RegularPolygon
  square : RegularPolygon

/-- The number of sides exposed to the outside for polygons adjacent to one other shape. -/
def exposedSidesOneAdjacent (p1 p2 : RegularPolygon) : ℕ :=
  p1.sides + p2.sides - 2

/-- The number of sides exposed to the outside for polygons adjacent to two other shapes. -/
def exposedSidesTwoAdjacent (p1 p2 p3 : RegularPolygon) : ℕ :=
  p1.sides + p2.sides + p3.sides - 6

/-- The total number of sides in the resulting polygon. -/
def totalSides (arrangement : PolygonArrangement) : ℕ :=
  exposedSidesOneAdjacent arrangement.pentagon arrangement.square +
  exposedSidesTwoAdjacent arrangement.triangle arrangement.octagon arrangement.hexagon

/-- Theorem stating that the resulting polygon has 18 sides. -/
theorem resulting_polygon_has_18_sides (arrangement : PolygonArrangement)
  (h1 : arrangement.pentagon.sides = 5)
  (h2 : arrangement.triangle.sides = 3)
  (h3 : arrangement.octagon.sides = 8)
  (h4 : arrangement.hexagon.sides = 6)
  (h5 : arrangement.square.sides = 4) :
  totalSides arrangement = 18 := by
  sorry

end NUMINAMATH_CALUDE_resulting_polygon_has_18_sides_l2710_271049


namespace NUMINAMATH_CALUDE_absolute_value_inequality_range_l2710_271079

theorem absolute_value_inequality_range (k : ℝ) :
  (∃ x : ℝ, |x + 1| - |x - 2| < k) ↔ k > -3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_range_l2710_271079


namespace NUMINAMATH_CALUDE_binomial_30_3_l2710_271092

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_3_l2710_271092


namespace NUMINAMATH_CALUDE_export_probabilities_l2710_271010

/-- The number of inspections required for each batch -/
def num_inspections : ℕ := 5

/-- The probability of failing any given inspection -/
def fail_prob : ℝ := 0.2

/-- The probability of passing any given inspection -/
def pass_prob : ℝ := 1 - fail_prob

/-- The probability that a batch cannot be exported -/
def cannot_export_prob : ℝ := 1 - (pass_prob ^ num_inspections + num_inspections * fail_prob * pass_prob ^ (num_inspections - 1))

/-- The probability that all five inspections must be completed -/
def all_inspections_prob : ℝ := (num_inspections - 1) * fail_prob * pass_prob ^ (num_inspections - 2)

theorem export_probabilities :
  (cannot_export_prob = 0.26) ∧ (all_inspections_prob = 0.41) := by
  sorry

end NUMINAMATH_CALUDE_export_probabilities_l2710_271010


namespace NUMINAMATH_CALUDE_system_solution_l2710_271075

theorem system_solution (x y : ℚ) :
  (3 * x - 7 * y = 31) ∧ (5 * x + 2 * y = -10) → x = -336/205 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2710_271075


namespace NUMINAMATH_CALUDE_ellipse_sum_l2710_271036

/-- Represents an ellipse with center (h, k), semi-major axis a, and semi-minor axis c -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  c : ℝ

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  (x - e.h)^2 / e.a^2 + (y - e.k)^2 / e.c^2 = 1

theorem ellipse_sum (e : Ellipse) 
    (center_h : e.h = 3)
    (center_k : e.k = -5)
    (major_axis : e.a = 7)
    (minor_axis : e.c = 4) :
  e.h + e.k + e.a + e.c = 9 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_l2710_271036


namespace NUMINAMATH_CALUDE_correlation_index_approaching_one_improves_fitting_l2710_271094

/-- The correlation index in regression analysis -/
def correlation_index : ℝ → ℝ := sorry

/-- The fitting effect of a regression model -/
def fitting_effect : ℝ → ℝ := sorry

/-- As the correlation index approaches 1, the fitting effect improves -/
theorem correlation_index_approaching_one_improves_fitting :
  ∀ ε > 0, ∃ δ > 0, ∀ r : ℝ,
    1 - δ < correlation_index r → 
    fitting_effect r > fitting_effect 0 + ε :=
sorry

end NUMINAMATH_CALUDE_correlation_index_approaching_one_improves_fitting_l2710_271094


namespace NUMINAMATH_CALUDE_complement_of_N_in_M_l2710_271011

def M : Set ℕ := {0, 1, 2, 3, 4, 5}
def N : Set ℕ := {0, 2, 3}

theorem complement_of_N_in_M :
  M \ N = {1, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_N_in_M_l2710_271011


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2710_271055

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 ∧ 
  (∃ r : ℝ, 210 * r = b ∧ b * r = 135 / 56) →
  b = 22.5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2710_271055


namespace NUMINAMATH_CALUDE_rectangle_length_problem_l2710_271067

theorem rectangle_length_problem (b : ℝ) (h1 : b > 0) : 
  (2 * b - 5) * (b + 5) - 2 * b * b = 75 → 2 * b = 40 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_problem_l2710_271067


namespace NUMINAMATH_CALUDE_ellipse_C_properties_l2710_271000

/-- Given an ellipse C with equation x²/a² + y²/b² = 1 (a > b > 0), 
    eccentricity √3/3, and major axis length 2√3 --/
def ellipse_C (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ 
  (a^2 - b^2) / a^2 = 3 / 9 ∧
  2 * a = 2 * Real.sqrt 3

/-- The equation of ellipse C --/
def ellipse_C_equation (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 / 2 = 1

/-- Circle O with major axis of ellipse C as its diameter --/
def circle_O (x y : ℝ) : Prop :=
  x^2 + y^2 = 3

/-- Point on circle O --/
def point_on_circle_O (M : ℝ × ℝ) : Prop :=
  circle_O M.1 M.2

/-- Line perpendicular to OM passing through M --/
def perpendicular_line (M : ℝ × ℝ) (x y : ℝ) : Prop :=
  M.1 * (x - M.1) + M.2 * (y - M.2) = 0

theorem ellipse_C_properties (a b : ℝ) (h : ellipse_C a b) :
  (∀ x y, ellipse_C_equation x y ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  (∀ M : ℝ × ℝ, point_on_circle_O M →
    ∃ x y, perpendicular_line M x y ∧ x = 1 ∧ y = 0) :=
sorry

end NUMINAMATH_CALUDE_ellipse_C_properties_l2710_271000


namespace NUMINAMATH_CALUDE_first_term_of_constant_ratio_l2710_271074

def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a + (n - 1) * d) / 2

theorem first_term_of_constant_ratio (d : ℚ) (h : d = 5) :
  (∃ (c : ℚ), ∀ (n : ℕ), n > 0 → 
    arithmetic_sum a d (5 * n) / arithmetic_sum a d n = c) →
  a = 5 / 2 :=
sorry

end NUMINAMATH_CALUDE_first_term_of_constant_ratio_l2710_271074
