import Mathlib

namespace NUMINAMATH_CALUDE_sugar_measurement_l888_88896

theorem sugar_measurement (sugar_needed : ℚ) (cup_capacity : ℚ) : 
  sugar_needed = 3 + 3 / 4 ∧ cup_capacity = 1 / 3 → 
  ↑(Int.ceil ((sugar_needed / cup_capacity) : ℚ)) = 12 :=
by sorry

end NUMINAMATH_CALUDE_sugar_measurement_l888_88896


namespace NUMINAMATH_CALUDE_field_length_proof_l888_88832

theorem field_length_proof (l w : ℝ) (h1 : l = 2 * w) (h2 : (8 * 8) = (1 / 98) * (l * w)) : l = 112 := by
  sorry

end NUMINAMATH_CALUDE_field_length_proof_l888_88832


namespace NUMINAMATH_CALUDE_solution_to_equation_l888_88807

theorem solution_to_equation : 
  ∃! x : ℝ, (Real.sqrt x + 3 * Real.sqrt (x^3 + 7*x) + Real.sqrt (x + 7) = 50 - x^2) ∧ 
            (x = (29/12)^2) := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l888_88807


namespace NUMINAMATH_CALUDE_collinearity_condition_l888_88841

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Three points are collinear if the area of the triangle formed by them is zero -/
def collinear (A B C : Point) : Prop :=
  (B.x - A.x) * (C.y - A.y) = (C.x - A.x) * (B.y - A.y)

theorem collinearity_condition (n : ℝ) : 
  let A : Point := ⟨1, 1⟩
  let B : Point := ⟨4, 0⟩
  let C : Point := ⟨0, n⟩
  collinear A B C ↔ n = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_collinearity_condition_l888_88841


namespace NUMINAMATH_CALUDE_matthew_score_proof_l888_88865

def basket_value : ℕ := 3
def total_baskets : ℕ := 5
def shawn_points : ℕ := 6

def matthew_points : ℕ := 9

theorem matthew_score_proof :
  matthew_points = basket_value * total_baskets - shawn_points :=
by sorry

end NUMINAMATH_CALUDE_matthew_score_proof_l888_88865


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l888_88889

/-- Two circles are externally tangent if and only if the distance between their centers
    is equal to the sum of their radii. -/
def externally_tangent (r₁ r₂ d : ℝ) : Prop := d = r₁ + r₂

/-- The theorem stating that two circles with radii 2 and 3, whose centers are 5 units apart,
    are externally tangent. -/
theorem circles_externally_tangent :
  let r₁ : ℝ := 2
  let r₂ : ℝ := 3
  let d : ℝ := 5
  externally_tangent r₁ r₂ d :=
by
  sorry


end NUMINAMATH_CALUDE_circles_externally_tangent_l888_88889


namespace NUMINAMATH_CALUDE_shaded_squares_in_six_by_six_grid_l888_88873

/-- Represents a grid with a given size and number of unshaded squares per row -/
structure Grid where
  size : Nat
  unshadedPerRow : Nat

/-- Calculates the number of shaded squares in the grid -/
def shadedSquares (g : Grid) : Nat :=
  g.size * (g.size - g.unshadedPerRow)

theorem shaded_squares_in_six_by_six_grid :
  ∀ (g : Grid), g.size = 6 → g.unshadedPerRow = 1 → shadedSquares g = 30 := by
  sorry

end NUMINAMATH_CALUDE_shaded_squares_in_six_by_six_grid_l888_88873


namespace NUMINAMATH_CALUDE_expression_equals_two_l888_88864

/-- Given real numbers a, b, and c satisfying two conditions, 
    prove that a certain expression equals 2 -/
theorem expression_equals_two (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = 8)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 7) :
  b / (a + b) + c / (b + c) + a / (c + a) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_two_l888_88864


namespace NUMINAMATH_CALUDE_product_of_largest_primes_l888_88881

/-- The largest one-digit prime -/
def largest_one_digit_prime : ℕ := 7

/-- The second largest one-digit prime -/
def second_largest_one_digit_prime : ℕ := 5

/-- The largest three-digit prime -/
def largest_three_digit_prime : ℕ := 997

/-- Theorem stating that the product of the two largest one-digit primes
    and the largest three-digit prime is 34895 -/
theorem product_of_largest_primes :
  largest_one_digit_prime * second_largest_one_digit_prime * largest_three_digit_prime = 34895 := by
  sorry

end NUMINAMATH_CALUDE_product_of_largest_primes_l888_88881


namespace NUMINAMATH_CALUDE_truncated_cube_edge_count_l888_88890

/-- Represents a cube with truncated corners -/
structure TruncatedCube where
  -- The number of original cube edges
  original_edges : ℕ := 12
  -- The number of corners (vertices) in the original cube
  corners : ℕ := 8
  -- The number of edges in each pentagonal face created by truncation
  pentagonal_edges : ℕ := 5
  -- Condition that cutting planes do not intersect within the cube
  non_intersecting_cuts : Prop

/-- The number of edges in a truncated cube -/
def edge_count (tc : TruncatedCube) : ℕ :=
  tc.original_edges + (tc.corners * tc.pentagonal_edges) / 2

/-- Theorem stating that a truncated cube has 32 edges -/
theorem truncated_cube_edge_count (tc : TruncatedCube) :
  edge_count tc = 32 := by
  sorry

#check truncated_cube_edge_count

end NUMINAMATH_CALUDE_truncated_cube_edge_count_l888_88890


namespace NUMINAMATH_CALUDE_square_root_equation_solution_l888_88803

theorem square_root_equation_solution : 
  ∃ x : ℝ, (56^2 + 56^2) / x^2 = 8 ∧ x = 28 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_solution_l888_88803


namespace NUMINAMATH_CALUDE_xiaoming_mother_expenses_l888_88813

/-- Represents a financial transaction with an amount in Yuan -/
structure Transaction where
  amount : Int

/-- Calculates the net result of a list of transactions -/
def netResult (transactions : List Transaction) : Int :=
  transactions.foldl (fun acc t => acc + t.amount) 0

theorem xiaoming_mother_expenses : 
  let transactions : List Transaction := [
    { amount := 42 },   -- Transfer from Hong
    { amount := -30 },  -- Paying phone bill
    { amount := -51 }   -- Scan QR code for payment
  ]
  netResult transactions = -39 := by
  sorry

end NUMINAMATH_CALUDE_xiaoming_mother_expenses_l888_88813


namespace NUMINAMATH_CALUDE_prime_power_sum_perfect_square_l888_88821

theorem prime_power_sum_perfect_square (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r →
  (∃ n : ℕ, p^q + p^r = n^2) ↔ 
  ((p = 2 ∧ q = r ∧ ∃ k : ℕ, Prime k ∧ k > 2 ∧ q = k) ∨
   (p = 3 ∧ ((q = 3 ∧ r = 2) ∨ (q = 2 ∧ r = 3)))) :=
by sorry

end NUMINAMATH_CALUDE_prime_power_sum_perfect_square_l888_88821


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l888_88844

theorem complete_square_quadratic (x : ℝ) : 
  ∃ (c d : ℝ), x^2 + 14*x + 24 = 0 ↔ (x + c)^2 = d ∧ d = 25 := by
sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l888_88844


namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l888_88897

/-- Given a set of 60 numbers with an arithmetic mean of 42, 
    prove that removing 50 and 60 results in a new arithmetic mean of 41.5 -/
theorem arithmetic_mean_after_removal (S : Finset ℝ) (x y : ℝ) : 
  S.card = 60 → 
  x ∈ S → 
  y ∈ S → 
  x = 50 → 
  y = 60 → 
  (S.sum id) / S.card = 42 → 
  ((S.sum id) - x - y) / (S.card - 2) = 41.5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l888_88897


namespace NUMINAMATH_CALUDE_regions_in_circle_l888_88891

/-- The number of regions created by radii and concentric circles inside a circle -/
def num_regions (num_radii : ℕ) (num_concentric_circles : ℕ) : ℕ :=
  (num_concentric_circles + 1) * num_radii

/-- Theorem: 16 radii and 10 concentric circles create 176 regions -/
theorem regions_in_circle (num_radii : ℕ) (num_concentric_circles : ℕ) 
  (h1 : num_radii = 16) (h2 : num_concentric_circles = 10) : 
  num_regions num_radii num_concentric_circles = 176 := by
  sorry

#eval num_regions 16 10  -- Should output 176

end NUMINAMATH_CALUDE_regions_in_circle_l888_88891


namespace NUMINAMATH_CALUDE_age_ratio_theorem_l888_88858

/-- Represents the ages of John and Emily -/
structure Ages where
  john : ℕ
  emily : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  (ages.john - 3 = 5 * (ages.emily - 3)) ∧
  (ages.john - 7 = 6 * (ages.emily - 7))

/-- The theorem to be proved -/
theorem age_ratio_theorem (ages : Ages) :
  problem_conditions ages →
  ∃ x : ℕ, x = 17 ∧ (ages.john + x) / (ages.emily + x) = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_age_ratio_theorem_l888_88858


namespace NUMINAMATH_CALUDE_tangent_line_at_point_A_l888_88800

/-- The function f(x) = x³ - 3x -/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

theorem tangent_line_at_point_A :
  ∃ (m b : ℝ), 
    (f 0 = 16) ∧ 
    (∀ x : ℝ, m * x + b = f' 0 * x + f 0) ∧
    (m = 9 ∧ b = 22) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_A_l888_88800


namespace NUMINAMATH_CALUDE_sum_of_remainders_l888_88819

theorem sum_of_remainders (d e f : ℕ+) 
  (hd : d ≡ 19 [ZMOD 53])
  (he : e ≡ 33 [ZMOD 53])
  (hf : f ≡ 14 [ZMOD 53]) :
  (d + e + f : ℤ) ≡ 13 [ZMOD 53] := by
  sorry

end NUMINAMATH_CALUDE_sum_of_remainders_l888_88819


namespace NUMINAMATH_CALUDE_student_count_l888_88822

/-- Proves the number of students in a class given certain height data -/
theorem student_count (initial_avg : ℝ) (incorrect_height : ℝ) (correct_height : ℝ) (actual_avg : ℝ) :
  initial_avg = 175 →
  incorrect_height = 151 →
  correct_height = 111 →
  actual_avg = 173 →
  ∃ n : ℕ, n * actual_avg = n * initial_avg - (incorrect_height - correct_height) ∧ n = 20 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l888_88822


namespace NUMINAMATH_CALUDE_angle_C_is_30_degrees_ab_range_when_c_is_1_l888_88839

/-- Represents an acute triangle with sides a, b, c opposite to angles A, B, C respectively. -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : Real
  B : Real
  C : Real
  acute_A : 0 < A ∧ A < Real.pi / 2
  acute_B : 0 < B ∧ B < Real.pi / 2
  acute_C : 0 < C ∧ C < Real.pi / 2
  tan_C_eq : Real.tan C = (a * b) / (a^2 + b^2 - c^2)

/-- Theorem stating that if tan C = (ab) / (a² + b² - c²) in an acute triangle, then C = 30° -/
theorem angle_C_is_30_degrees (t : AcuteTriangle) : t.C = Real.pi / 6 := by
  sorry

/-- Theorem stating that if c = 1 and tan C = (ab) / (a² + b² - 1) in an acute triangle, 
    then 2√3 < ab ≤ 2 + √3 -/
theorem ab_range_when_c_is_1 (t : AcuteTriangle) (h : t.c = 1) : 
  2 * Real.sqrt 3 < t.a * t.b ∧ t.a * t.b ≤ 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_is_30_degrees_ab_range_when_c_is_1_l888_88839


namespace NUMINAMATH_CALUDE_three_digit_rounding_l888_88856

theorem three_digit_rounding (A : ℕ) : 
  (100 ≤ A * 100 + 76) ∧ (A * 100 + 76 < 1000) ∧ 
  ((A * 100 + 76) / 100 * 100 = 700) → A = 7 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_rounding_l888_88856


namespace NUMINAMATH_CALUDE_C_ℝP_subset_Q_l888_88811

-- Define set P
def P : Set ℝ := {y | ∃ x : ℝ, y = -x^2 + 1}

-- Define set Q
def Q : Set ℝ := {y | ∃ x : ℝ, y = 2^x}

-- Define the complement of P in ℝ
def C_ℝP : Set ℝ := {y | y ∉ P}

-- Theorem statement
theorem C_ℝP_subset_Q : C_ℝP ⊆ Q := by
  sorry

end NUMINAMATH_CALUDE_C_ℝP_subset_Q_l888_88811


namespace NUMINAMATH_CALUDE_least_zogs_for_dropping_beats_eating_l888_88869

theorem least_zogs_for_dropping_beats_eating :
  ∀ n : ℕ, n > 0 → (∀ k : ℕ, k > 0 → k < n → k * (k + 1) ≤ 15 * k) → 15 * 15 < 15 * (15 + 1) :=
sorry

end NUMINAMATH_CALUDE_least_zogs_for_dropping_beats_eating_l888_88869


namespace NUMINAMATH_CALUDE_preimage_of_4_3_l888_88895

def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2 * p.2, 2 * p.1 - p.2)

theorem preimage_of_4_3 :
  ∃ (p : ℝ × ℝ), f p = (4, 3) ∧ p = (2, 1) := by
sorry

end NUMINAMATH_CALUDE_preimage_of_4_3_l888_88895


namespace NUMINAMATH_CALUDE_bacteria_growth_30_minutes_l888_88804

/-- The number of bacteria after a given number of 2-minute intervals, 
    given an initial population and a tripling growth rate every 2 minutes. -/
def bacteria_population (initial_population : ℕ) (intervals : ℕ) : ℕ :=
  initial_population * (3 ^ intervals)

/-- Theorem stating that after 15 intervals (30 minutes), 
    an initial population of 30 bacteria will grow to 430467210. -/
theorem bacteria_growth_30_minutes :
  bacteria_population 30 15 = 430467210 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_30_minutes_l888_88804


namespace NUMINAMATH_CALUDE_skateboard_cost_l888_88899

theorem skateboard_cost (total_toys : ℝ) (toy_cars : ℝ) (toy_trucks : ℝ) 
  (h1 : total_toys = 25.62)
  (h2 : toy_cars = 14.88)
  (h3 : toy_trucks = 5.86) :
  total_toys - toy_cars - toy_trucks = 4.88 := by
  sorry

end NUMINAMATH_CALUDE_skateboard_cost_l888_88899


namespace NUMINAMATH_CALUDE_hardey_fitness_center_ratio_l888_88823

theorem hardey_fitness_center_ratio 
  (avg_female : ℝ) 
  (avg_male : ℝ) 
  (avg_child : ℝ) 
  (avg_overall : ℝ) 
  (h1 : avg_female = 35)
  (h2 : avg_male = 30)
  (h3 : avg_child = 10)
  (h4 : avg_overall = 25) :
  ∃ (f m c : ℝ), 
    f > 0 ∧ m > 0 ∧ c > 0 ∧
    (avg_female * f + avg_male * m + avg_child * c) / (f + m + c) = avg_overall ∧
    c / (f + m) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hardey_fitness_center_ratio_l888_88823


namespace NUMINAMATH_CALUDE_cos_A_value_cos_2A_plus_pi_over_4_l888_88814

-- Define the triangle ABC
def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  B = C ∧ 2 * b = Real.sqrt 3 * a

-- Theorem 1: cos A = 1/3
theorem cos_A_value (A B C : ℝ) (a b c : ℝ) 
  (h : triangle A B C a b c) : Real.cos A = 1 / 3 := by
  sorry

-- Theorem 2: cos(2A + π/4) = -(8 + 7√2)/18
theorem cos_2A_plus_pi_over_4 (A B C : ℝ) (a b c : ℝ) 
  (h : triangle A B C a b c) : 
  Real.cos (2 * A + Real.pi / 4) = -(8 + 7 * Real.sqrt 2) / 18 := by
  sorry

end NUMINAMATH_CALUDE_cos_A_value_cos_2A_plus_pi_over_4_l888_88814


namespace NUMINAMATH_CALUDE_cell_diameter_scientific_notation_l888_88859

/-- Expresses a given number in scientific notation -/
def scientificNotation (n : ℝ) : ℝ × ℤ :=
  sorry

theorem cell_diameter_scientific_notation :
  scientificNotation 0.00065 = (6.5, -4) := by sorry

end NUMINAMATH_CALUDE_cell_diameter_scientific_notation_l888_88859


namespace NUMINAMATH_CALUDE_customers_who_tipped_l888_88802

/-- The number of customers who left a tip at 'The Greasy Spoon' restaurant -/
theorem customers_who_tipped (initial_customers : ℕ) (additional_customers : ℕ) (non_tipping_customers : ℕ) :
  initial_customers = 29 →
  additional_customers = 20 →
  non_tipping_customers = 34 →
  initial_customers + additional_customers - non_tipping_customers = 15 :=
by sorry

end NUMINAMATH_CALUDE_customers_who_tipped_l888_88802


namespace NUMINAMATH_CALUDE_max_enclosed_area_l888_88848

/-- Represents an infinite chessboard -/
structure InfiniteChessboard where

/-- Represents a closed non-self-intersecting polygonal line on the chessboard -/
structure PolygonalLine where
  chessboard : InfiniteChessboard
  is_closed : Bool
  is_non_self_intersecting : Bool
  along_cell_sides : Bool

/-- Represents the area enclosed by a polygonal line -/
def EnclosedArea (line : PolygonalLine) : ℕ := sorry

/-- Counts the number of black cells inside a polygonal line -/
def BlackCellsCount (line : PolygonalLine) : ℕ := sorry

/-- Theorem stating the maximum area enclosed by a polygonal line -/
theorem max_enclosed_area (line : PolygonalLine) (k : ℕ) 
  (h1 : line.is_closed = true)
  (h2 : line.is_non_self_intersecting = true)
  (h3 : line.along_cell_sides = true)
  (h4 : BlackCellsCount line = k) :
  EnclosedArea line ≤ 4 * k + 1 := by sorry

end NUMINAMATH_CALUDE_max_enclosed_area_l888_88848


namespace NUMINAMATH_CALUDE_find_A_l888_88830

theorem find_A : ∃ A : ℕ, A ≥ 1 ∧ A ≤ 9 ∧ (10 * A + 72) - 23 = 549 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l888_88830


namespace NUMINAMATH_CALUDE_annas_walking_challenge_l888_88806

/-- Anna's walking challenge in March -/
theorem annas_walking_challenge 
  (total_days : ℕ) 
  (daily_target : ℝ) 
  (days_passed : ℕ) 
  (distance_walked : ℝ) 
  (h1 : total_days = 31) 
  (h2 : daily_target = 5) 
  (h3 : days_passed = 16) 
  (h4 : distance_walked = 95) : 
  (total_days * daily_target - distance_walked) / (total_days - days_passed) = 4 := by
  sorry

end NUMINAMATH_CALUDE_annas_walking_challenge_l888_88806


namespace NUMINAMATH_CALUDE_find_genuine_stacks_l888_88871

/-- Represents a stack of coins -/
structure CoinStack :=
  (count : Nat)
  (hasOddCoin : Bool)

/-- Represents the result of weighing two stacks -/
inductive WeighResult
  | Equal
  | Unequal

/-- Represents the state of the coin stacks -/
structure CoinStacks :=
  (stack1 : CoinStack)
  (stack2 : CoinStack)
  (stack3 : CoinStack)
  (stack4 : CoinStack)

/-- Represents a weighing action -/
def weigh (s1 s2 : CoinStack) : WeighResult :=
  if s1.hasOddCoin = s2.hasOddCoin then WeighResult.Equal else WeighResult.Unequal

/-- The main theorem -/
theorem find_genuine_stacks 
  (stacks : CoinStacks)
  (h1 : stacks.stack1.count = 5)
  (h2 : stacks.stack2.count = 6)
  (h3 : stacks.stack3.count = 7)
  (h4 : stacks.stack4.count = 19)
  (h5 : (stacks.stack1.hasOddCoin || stacks.stack2.hasOddCoin || stacks.stack3.hasOddCoin || stacks.stack4.hasOddCoin) ∧ 
        (¬stacks.stack1.hasOddCoin ∨ ¬stacks.stack2.hasOddCoin ∨ ¬stacks.stack3.hasOddCoin ∨ ¬stacks.stack4.hasOddCoin)) :
  ∃ (s1 s2 : CoinStack), s1 ∈ [stacks.stack1, stacks.stack2, stacks.stack3, stacks.stack4] ∧ 
                         s2 ∈ [stacks.stack1, stacks.stack2, stacks.stack3, stacks.stack4] ∧ 
                         s1 ≠ s2 ∧ 
                         ¬s1.hasOddCoin ∧ 
                         ¬s2.hasOddCoin := by
  sorry


end NUMINAMATH_CALUDE_find_genuine_stacks_l888_88871


namespace NUMINAMATH_CALUDE_pentagon_area_fraction_l888_88834

/-- Represents a rectangle with length 3 times its width -/
structure Rectangle where
  width : ℝ
  length : ℝ
  length_eq_3width : length = 3 * width

/-- Represents a pentagon formed by folding the rectangle -/
structure Pentagon where
  original : Rectangle
  area : ℝ

/-- The theorem to be proved -/
theorem pentagon_area_fraction (r : Rectangle) (p : Pentagon) 
  (h : p.original = r) : 
  p.area = (13 / 18) * (r.width * r.length) := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_fraction_l888_88834


namespace NUMINAMATH_CALUDE_equation_solution_l888_88875

theorem equation_solution :
  ∃ x : ℚ, (x - 55) / 3 = (2 - 3*x + x^2) / 4 ∧ (x = 20/3 ∨ x = -11) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l888_88875


namespace NUMINAMATH_CALUDE_solve_equation_l888_88898

theorem solve_equation (x : ℝ) (h1 : x > 5) 
  (h2 : Real.sqrt (x - 3 * Real.sqrt (x - 5)) + 3 = Real.sqrt (x + 3 * Real.sqrt (x - 5)) - 3) : 
  x = 41 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l888_88898


namespace NUMINAMATH_CALUDE_triangle_count_theorem_l888_88863

/-- The number of triangles formed by selecting three non-collinear points from a set of points on a triangle -/
def num_triangles (a b c : ℕ) : ℕ :=
  let total_points := 3 + a + b + c
  let total_combinations := (total_points.choose 3)
  let collinear_combinations := (a + 2).choose 3 + (b + 2).choose 3 + (c + 2).choose 3
  total_combinations - collinear_combinations

/-- Theorem stating that the number of triangles formed in the given configuration is 357 -/
theorem triangle_count_theorem : num_triangles 2 3 7 = 357 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_theorem_l888_88863


namespace NUMINAMATH_CALUDE_modified_triangle_property_unbounded_l888_88843

/-- A function that checks if three numbers can form a right triangle -/
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- A function that checks if a set of 10 consecutive integers contains a right triangle -/
def has_right_triangle (start : ℕ) : Prop :=
  ∃ (a b c : ℕ), start ≤ a ∧ a < b ∧ b < c ∧ c < start + 10 ∧ is_right_triangle a b c

/-- The main theorem stating that for any k ≥ 10, the set {5, 6, ..., k} 
    satisfies the modified triangle property for all 10-element subsets -/
theorem modified_triangle_property_unbounded (k : ℕ) (h : k ≥ 10) :
  ∀ (n : ℕ), 5 ≤ n ∧ n ≤ k - 9 → has_right_triangle n :=
sorry

end NUMINAMATH_CALUDE_modified_triangle_property_unbounded_l888_88843


namespace NUMINAMATH_CALUDE_quadratic_roots_preservation_l888_88868

theorem quadratic_roots_preservation (p q α : ℝ) 
  (h1 : ∃ x : ℝ, x^2 + p*x + q = 0) 
  (h2 : 0 < α) (h3 : α ≤ 1) : 
  ∃ y : ℝ, α*y^2 + p*y + q = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_preservation_l888_88868


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_train_passes_jogger_in_25_seconds_l888_88877

/-- Time taken for a train to pass a jogger -/
theorem train_passing_jogger_time (jogger_speed train_speed : ℝ) 
  (train_length initial_distance : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed * (1000 / 3600)
  let train_speed_ms := train_speed * (1000 / 3600)
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- The time taken for the train to pass the jogger is 25 seconds -/
theorem train_passes_jogger_in_25_seconds : 
  train_passing_jogger_time 9 45 100 150 = 25 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_train_passes_jogger_in_25_seconds_l888_88877


namespace NUMINAMATH_CALUDE_soccer_team_average_goals_l888_88829

theorem soccer_team_average_goals (pizzas : ℕ) (slices_per_pizza : ℕ) (games : ℕ)
  (h1 : pizzas = 6)
  (h2 : slices_per_pizza = 12)
  (h3 : games = 8) :
  (pizzas * slices_per_pizza) / games = 9 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_average_goals_l888_88829


namespace NUMINAMATH_CALUDE_equality_of_pairs_l888_88826

theorem equality_of_pairs (a b x y : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_x : 0 < x) (pos_y : 0 < y)
  (sum_lt_two : a + b + x + y < 2)
  (eq_one : a + b^2 = x + y^2)
  (eq_two : a^2 + b = x^2 + y) :
  a = x ∧ b = y := by
sorry

end NUMINAMATH_CALUDE_equality_of_pairs_l888_88826


namespace NUMINAMATH_CALUDE_unique_element_quadratic_set_l888_88833

theorem unique_element_quadratic_set (a : ℝ) :
  (∃! x : ℝ, a * x^2 - 2 * x + 1 = 0) ↔ (a = 0 ∨ a = 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_element_quadratic_set_l888_88833


namespace NUMINAMATH_CALUDE_percentage_of_b_l888_88860

/-- Given that 12 is 6% of a, a certain percentage of b is 6, and c equals b / a,
    prove that the percentage of b is 6 / (200 * c) * 100 -/
theorem percentage_of_b (a b c : ℝ) (h1 : 0.06 * a = 12) (h2 : ∃ p, p * b = 6) (h3 : c = b / a) :
  ∃ p, p * b = 6 ∧ p * 100 = 6 / (200 * c) * 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_b_l888_88860


namespace NUMINAMATH_CALUDE_geometry_books_shelf_filling_l888_88850

/-- Represents the number of books that fill a shelf. -/
structure ShelfFilling where
  algebra : ℕ
  geometry : ℕ

/-- Represents the properties of the book arrangement problem. -/
structure BookArrangement where
  P : ℕ  -- Total number of algebra books
  Q : ℕ  -- Total number of geometry books
  X : ℕ  -- Number of algebra books that fill the shelf
  Y : ℕ  -- Number of geometry books that fill the shelf

/-- The main theorem about the number of geometry books (Z) that fill the shelf. -/
theorem geometry_books_shelf_filling 
  (arr : BookArrangement) 
  (fill1 : ShelfFilling)
  (fill2 : ShelfFilling)
  (h1 : fill1.algebra = arr.X ∧ fill1.geometry = arr.Y)
  (h2 : fill2.algebra = 2 * fill2.geometry)
  (h3 : arr.P + 2 * arr.Q = arr.X + 2 * arr.Y) :
  ∃ Z : ℕ, Z = (arr.P + 2 * arr.Q) / 2 ∧ 
             Z * 2 = arr.P + 2 * arr.Q ∧
             fill2.geometry = Z :=
by sorry

end NUMINAMATH_CALUDE_geometry_books_shelf_filling_l888_88850


namespace NUMINAMATH_CALUDE_apple_difference_is_twenty_l888_88851

/-- The number of apples Cecile bought -/
def cecile_apples : ℕ := 15

/-- The total number of apples bought by Diane and Cecile -/
def total_apples : ℕ := 50

/-- The number of apples Diane bought -/
def diane_apples : ℕ := total_apples - cecile_apples

/-- Diane bought more apples than Cecile -/
axiom diane_bought_more : diane_apples > cecile_apples

/-- The difference between the number of apples Diane and Cecile bought -/
def apple_difference : ℕ := diane_apples - cecile_apples

theorem apple_difference_is_twenty : apple_difference = 20 :=
sorry

end NUMINAMATH_CALUDE_apple_difference_is_twenty_l888_88851


namespace NUMINAMATH_CALUDE_complex_equation_solution_l888_88883

theorem complex_equation_solution (a : ℝ) : (a + Complex.I) * (1 - a * Complex.I) = 2 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l888_88883


namespace NUMINAMATH_CALUDE_certain_number_proof_l888_88862

theorem certain_number_proof (x : ℕ) (certain_number : ℕ) : 
  (certain_number = 3 * x + 36) → (x = 4) → (certain_number = 48) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l888_88862


namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l888_88874

def parabola (y : ℝ) : ℝ := 2 * y^2 - 6 * y + 3

theorem parabola_intercepts_sum :
  ∃ (a b c : ℝ),
    (parabola 0 = a) ∧
    (parabola b = 0) ∧
    (parabola c = 0) ∧
    (b ≠ c) ∧
    (a + b + c = 6) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l888_88874


namespace NUMINAMATH_CALUDE_tangent_line_range_l888_88872

/-- A line is tangent to a circle if and only if the distance from the center of the circle to the line is equal to the radius of the circle. -/
def is_tangent_line (m n : ℝ) : Prop :=
  1 = |(m + 1) + (n + 1) - 2| / Real.sqrt ((m + 1)^2 + (n + 1)^2)

/-- The range of m + n when the line (m+1)x + (n+1)y - 2 = 0 is tangent to the circle (x-1)^2 + (y-1)^2 = 1 -/
theorem tangent_line_range (m n : ℝ) :
  is_tangent_line m n →
  (m + n ≤ 2 - 2 * Real.sqrt 2 ∨ m + n ≥ 2 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_range_l888_88872


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l888_88840

-- Define the original quadratic equation
def original_equation (x : ℝ) : Prop := x^2 - 4*x - 1 = 0

-- Define the completed square form
def completed_square (x : ℝ) : Prop := (x - 2)^2 = 5

-- Theorem stating that the completed square form is equivalent to the original equation
theorem completing_square_equivalence :
  ∀ x : ℝ, original_equation x ↔ completed_square x :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l888_88840


namespace NUMINAMATH_CALUDE_prob_B_not_occur_expected_value_B_l888_88818

-- Define the sample space for a single die roll
def Ω : Finset ℕ := Finset.range 6

-- Define events A and B
def A : Finset ℕ := {0, 1, 2}
def B : Finset ℕ := {0, 1, 3}

-- Number of rolls
def n : ℕ := 10

-- Number of times event A occurred
def k : ℕ := 6

-- Probability of event A
def p_A : ℚ := (A.card : ℚ) / Ω.card

-- Probability of event B given A
def p_B_given_A : ℚ := ((A ∩ B).card : ℚ) / A.card

-- Probability of event B given not A
def p_B_given_not_A : ℚ := ((B \ A).card : ℚ) / (Ω \ A).card

-- Theorem for part (a)
theorem prob_B_not_occur (h : k = 6) :
  (Finset.card Ω)^n * (A.card)^k * ((Ω \ A).card)^(n - k) * ((A \ B).card)^k * ((Ω \ (A ∪ B)).card)^(n - k) / 
  (Finset.card Ω)^n / (Finset.card Ω)^n * Nat.choose n k = 64 / 236486 := by sorry

-- Theorem for part (b)
theorem expected_value_B (h : k = 6) :
  k * p_B_given_A + (n - k) * p_B_given_not_A = 16 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_B_not_occur_expected_value_B_l888_88818


namespace NUMINAMATH_CALUDE_alcohol_percentage_after_dilution_l888_88824

/-- Calculates the percentage of alcohol in a mixture after adding water -/
theorem alcohol_percentage_after_dilution
  (initial_volume : ℝ)
  (initial_alcohol_percentage : ℝ)
  (water_added : ℝ)
  (h1 : initial_volume = 11)
  (h2 : initial_alcohol_percentage = 42)
  (h3 : water_added = 3) :
  let initial_alcohol_volume := initial_volume * (initial_alcohol_percentage / 100)
  let final_volume := initial_volume + water_added
  let final_alcohol_percentage := (initial_alcohol_volume / final_volume) * 100
  final_alcohol_percentage = 33 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_percentage_after_dilution_l888_88824


namespace NUMINAMATH_CALUDE_birds_left_in_cage_l888_88878

/-- The number of birds initially in the cage -/
def initial_birds : ℕ := 19

/-- The number of birds taken out of the cage -/
def birds_taken_out : ℕ := 10

/-- Theorem stating that the number of birds left in the cage is 9 -/
theorem birds_left_in_cage : initial_birds - birds_taken_out = 9 := by
  sorry

end NUMINAMATH_CALUDE_birds_left_in_cage_l888_88878


namespace NUMINAMATH_CALUDE_shaggy_seed_count_l888_88861

/-- Represents the number of seeds Shaggy ate -/
def shaggy_seeds : ℕ := 54

/-- Represents the total number of seeds -/
def total_seeds : ℕ := 60

/-- Represents the ratio of Shaggy's berry eating speed to Fluffball's -/
def berry_speed_ratio : ℕ := 6

/-- Represents the ratio of Shaggy's seed eating speed to Fluffball's -/
def seed_speed_ratio : ℕ := 3

/-- Represents the ratio of berries Shaggy ate to Fluffball -/
def berry_ratio : ℕ := 2

theorem shaggy_seed_count : 
  50 < total_seeds ∧ 
  total_seeds < 65 ∧ 
  berry_speed_ratio = 6 ∧ 
  seed_speed_ratio = 3 ∧ 
  berry_ratio = 2 → 
  shaggy_seeds = 54 := by sorry

end NUMINAMATH_CALUDE_shaggy_seed_count_l888_88861


namespace NUMINAMATH_CALUDE_isosceles_triangle_smallest_angle_isosceles_triangle_smallest_angle_proof_l888_88847

/-- An isosceles triangle with one angle 40% larger than a right angle has two smallest angles measuring 27°. -/
theorem isosceles_triangle_smallest_angle : ℝ → Prop :=
  fun x =>
    let right_angle := 90
    let large_angle := 1.4 * right_angle
    let sum_of_angles := 180
    x > 0 ∧ 
    x < large_angle ∧ 
    2 * x + large_angle = sum_of_angles →
    x = 27

/-- Proof of the theorem -/
theorem isosceles_triangle_smallest_angle_proof : isosceles_triangle_smallest_angle 27 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_smallest_angle_isosceles_triangle_smallest_angle_proof_l888_88847


namespace NUMINAMATH_CALUDE_percentage_in_at_least_two_trips_l888_88885

/-- Represents the percentage of students who went on a specific trip -/
structure TripParticipation where
  threeDay : Rat
  twoDay : Rat
  oneDay : Rat

/-- Represents the percentage of students who participated in multiple trips -/
structure MultipleTrips where
  threeDayAndOneDay : Rat
  twoDayAndOther : Rat

/-- Calculates the percentage of students who participated in at least two trips -/
def percentageInAtLeastTwoTrips (tp : TripParticipation) (mt : MultipleTrips) : Rat :=
  mt.threeDayAndOneDay + mt.twoDayAndOther

/-- Main theorem: The percentage of students who participated in at least two trips is 22% -/
theorem percentage_in_at_least_two_trips :
  ∀ (tp : TripParticipation) (mt : MultipleTrips),
  tp.threeDay = 25/100 ∧
  tp.twoDay = 10/100 ∧
  mt.threeDayAndOneDay = 65/100 * tp.threeDay ∧
  mt.twoDayAndOther = 60/100 * tp.twoDay →
  percentageInAtLeastTwoTrips tp mt = 22/100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_in_at_least_two_trips_l888_88885


namespace NUMINAMATH_CALUDE_sum_of_divisors_540_has_4_prime_factors_l888_88849

-- Define the number we're working with
def n : ℕ := 540

-- Define the sum of positive divisors function
noncomputable def sum_of_divisors (m : ℕ) : ℕ := sorry

-- Define a function to count distinct prime factors
noncomputable def count_distinct_prime_factors (m : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_divisors_540_has_4_prime_factors :
  count_distinct_prime_factors (sum_of_divisors n) = 4 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_540_has_4_prime_factors_l888_88849


namespace NUMINAMATH_CALUDE_tan_beta_value_l888_88876

theorem tan_beta_value (α β : ℝ) 
  (h1 : Real.tan (π - α) = -(1 / 5))
  (h2 : Real.tan (α - β) = 1 / 3) : 
  Real.tan β = -(1 / 8) := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_value_l888_88876


namespace NUMINAMATH_CALUDE_ball_probability_l888_88812

theorem ball_probability (x : ℕ) : 
  (4 : ℝ) / (4 + x) = (2 : ℝ) / 5 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l888_88812


namespace NUMINAMATH_CALUDE_minuend_calculation_l888_88882

theorem minuend_calculation (subtrahend difference : ℝ) 
  (h1 : subtrahend = 1.34)
  (h2 : difference = 3.66) : 
  subtrahend + difference = 5 := by
  sorry

end NUMINAMATH_CALUDE_minuend_calculation_l888_88882


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l888_88809

theorem isosceles_triangle_area (h : ℝ) (p : ℝ) :
  h = 8 →
  p = 32 →
  ∃ (base : ℝ) (leg : ℝ),
    leg + leg + base = p ∧
    h^2 + (base/2)^2 = leg^2 ∧
    (1/2) * base * h = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l888_88809


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l888_88836

/-- Given a triangle with angle α, internal angle bisector length f, and external angle bisector length g,
    calculate the side lengths a, b, and c. -/
theorem triangle_side_lengths
  (α : Real) (f g : ℝ) (h_α : 0 < α ∧ α < π) (h_f : f > 0) (h_g : g > 0) :
  ∃ (a b c : ℝ),
    a = (f * g * Real.sqrt (f^2 + g^2) * Real.sin α) / (g^2 * (Real.cos (α/2))^2 - f^2 * (Real.sin (α/2))^2) ∧
    b = (f * g) / (g * Real.cos (α/2) + f * Real.sin (α/2)) ∧
    c = (f * g) / (g * Real.cos (α/2) - f * Real.sin (α/2)) ∧
    0 < a ∧ 0 < b ∧ 0 < c ∧
    a + b > c ∧ b + c > a ∧ c + a > b :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l888_88836


namespace NUMINAMATH_CALUDE_max_two_match_winners_100_l888_88892

/-- Represents a single-elimination tournament -/
structure Tournament :=
  (participants : ℕ)

/-- Represents the number of matches a participant has won -/
def wins (t : Tournament) (p : ℕ) : ℕ := sorry

/-- The number of participants who have won exactly two matches -/
def two_match_winners (t : Tournament) : ℕ := sorry

/-- The maximum possible number of two-match winners -/
def max_two_match_winners (t : Tournament) : ℕ := sorry

theorem max_two_match_winners_100 :
  ∀ t : Tournament, t.participants = 100 → max_two_match_winners t = 49 := by
  sorry

end NUMINAMATH_CALUDE_max_two_match_winners_100_l888_88892


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l888_88894

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l888_88894


namespace NUMINAMATH_CALUDE_percentage_subtraction_l888_88870

theorem percentage_subtraction (original : ℝ) (incorrect_subtraction : ℝ) (difference : ℝ) : 
  original = 200 →
  incorrect_subtraction = 25 →
  difference = 25 →
  let incorrect_result := original - incorrect_subtraction
  let correct_result := incorrect_result - difference
  let percentage := (original - correct_result) / original * 100
  percentage = 25 := by sorry

end NUMINAMATH_CALUDE_percentage_subtraction_l888_88870


namespace NUMINAMATH_CALUDE_bob_has_ten_candies_l888_88887

/-- The number of candies Bob has after trick-or-treating. -/
def bob_candies (mary_candies sue_candies john_candies sam_candies total_candies : ℕ) : ℕ :=
  total_candies - (mary_candies + sue_candies + john_candies + sam_candies)

/-- Theorem stating that Bob has 10 candies given the conditions from the problem. -/
theorem bob_has_ten_candies :
  bob_candies 5 20 5 10 50 = 10 := by sorry

end NUMINAMATH_CALUDE_bob_has_ten_candies_l888_88887


namespace NUMINAMATH_CALUDE_art_club_theorem_l888_88820

/-- Represents the number of artworks created by the art club over three school years. -/
def artworks_three_years (initial_students : ℕ) (artworks_per_student : ℕ) 
  (joining_students : ℕ) (leaving_students : ℕ) (quarters_per_year : ℕ) (years : ℕ) : ℕ :=
  let students_q1 := initial_students
  let students_q2_q3 := initial_students + joining_students
  let students_q4_q5 := students_q2_q3 - leaving_students
  let artworks_q1 := students_q1 * artworks_per_student
  let artworks_q2_q3 := students_q2_q3 * artworks_per_student
  let artworks_q4_q5 := students_q4_q5 * artworks_per_student
  let artworks_per_year := artworks_q1 + 2 * artworks_q2_q3 + 2 * artworks_q4_q5
  artworks_per_year * years

/-- Represents the number of artworks created in each quarter for the entire club. -/
def artworks_per_quarter (initial_students : ℕ) (artworks_per_student : ℕ) 
  (joining_students : ℕ) (leaving_students : ℕ) : List ℕ :=
  let students_q1 := initial_students
  let students_q2_q3 := initial_students + joining_students
  let students_q4_q5 := students_q2_q3 - leaving_students
  [students_q1 * artworks_per_student,
   students_q2_q3 * artworks_per_student,
   students_q2_q3 * artworks_per_student,
   students_q4_q5 * artworks_per_student,
   students_q4_q5 * artworks_per_student]

theorem art_club_theorem :
  artworks_three_years 30 3 4 6 5 3 = 1386 ∧
  artworks_per_quarter 30 3 4 6 = [90, 102, 102, 84, 84] := by
  sorry

end NUMINAMATH_CALUDE_art_club_theorem_l888_88820


namespace NUMINAMATH_CALUDE_room_perimeter_l888_88888

theorem room_perimeter (breadth : ℝ) (length : ℝ) (area : ℝ) (perimeter : ℝ) : 
  length = 3 * breadth →
  area = 12 →
  area = length * breadth →
  perimeter = 2 * (length + breadth) →
  perimeter = 16 := by
  sorry

end NUMINAMATH_CALUDE_room_perimeter_l888_88888


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l888_88842

/-- The length of a train given specific crossing times -/
theorem train_length (t_man : ℝ) (t_platform : ℝ) (l_platform : ℝ) : ℝ :=
  let train_length := (t_platform * l_platform) / (t_platform - t_man)
  186

/-- The train passes a stationary point in 8 seconds -/
def time_passing_man : ℝ := 8

/-- The train crosses a platform in 20 seconds -/
def time_crossing_platform : ℝ := 20

/-- The length of the platform is 279 meters -/
def platform_length : ℝ := 279

theorem train_length_proof :
  train_length time_passing_man time_crossing_platform platform_length = 186 := by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l888_88842


namespace NUMINAMATH_CALUDE_rhombus_area_in_hexagon_l888_88854

/-- A regular hexagon -/
structure RegularHexagon where
  area : ℝ

/-- The total area of rhombuses that can be formed within a regular hexagon -/
def total_rhombus_area (h : RegularHexagon) : ℝ :=
  sorry

/-- Theorem: In a regular hexagon with area 80, the total area of rhombuses is 45 -/
theorem rhombus_area_in_hexagon (h : RegularHexagon) 
  (h_area : h.area = 80) : total_rhombus_area h = 45 :=
  sorry

end NUMINAMATH_CALUDE_rhombus_area_in_hexagon_l888_88854


namespace NUMINAMATH_CALUDE_line_circle_intersection_l888_88893

/-- Given a line and a circle that intersect, prove the value of the line's slope --/
theorem line_circle_intersection (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (a * A.1 - A.2 + 3 = 0) ∧ 
    (a * B.1 - B.2 + 3 = 0) ∧ 
    ((A.1 - 1)^2 + (A.2 - 2)^2 = 4) ∧ 
    ((B.1 - 1)^2 + (B.2 - 2)^2 = 4) ∧ 
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 12)) →
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l888_88893


namespace NUMINAMATH_CALUDE_complex_arithmetic_calculation_l888_88805

theorem complex_arithmetic_calculation :
  let B : ℂ := 5 - 2*I
  let N : ℂ := -3 + 2*I
  let T : ℂ := 2*I
  let Q : ℝ := 3
  B - N + T - 2 * (Q : ℂ) = 2 - 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_calculation_l888_88805


namespace NUMINAMATH_CALUDE_subset_A_l888_88884

def A : Set ℝ := {x | x > -1}

theorem subset_A : {0} ⊆ A := by sorry

end NUMINAMATH_CALUDE_subset_A_l888_88884


namespace NUMINAMATH_CALUDE_midpoint_chain_l888_88828

theorem midpoint_chain (A B C D E F G : ℝ) : 
  C = (A + B) / 2 →  -- C is midpoint of AB
  D = (A + C) / 2 →  -- D is midpoint of AC
  E = (A + D) / 2 →  -- E is midpoint of AD
  F = (A + E) / 2 →  -- F is midpoint of AE
  G = (A + F) / 2 →  -- G is midpoint of AF
  G - A = 2 →        -- AG = 2
  B - A = 64 :=      -- AB = 64
by sorry

end NUMINAMATH_CALUDE_midpoint_chain_l888_88828


namespace NUMINAMATH_CALUDE_three_zeros_condition_l888_88857

/-- The cubic function f(x) = x^3 + ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 2

/-- The derivative of f(x) with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

/-- Theorem: For f(x) to have exactly 3 zeros, 'a' must be in the range (-∞, -3) -/
theorem three_zeros_condition (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔ a < -3 :=
sorry

end NUMINAMATH_CALUDE_three_zeros_condition_l888_88857


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l888_88835

theorem sum_with_radical_conjugate :
  (12 - Real.sqrt 2023) + (12 + Real.sqrt 2023) = 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l888_88835


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l888_88825

theorem complex_subtraction_simplification :
  (-5 - 3*I : ℂ) - (2 + 6*I) = -7 - 9*I := by sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l888_88825


namespace NUMINAMATH_CALUDE_problem_statement_l888_88816

theorem problem_statement (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) : 
  (a + b + 2*c ≤ 3) ∧ 
  (b = 2*c → 1/a + 1/c ≥ 3) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l888_88816


namespace NUMINAMATH_CALUDE_dividend_calculation_l888_88852

theorem dividend_calculation (divisor quotient remainder : Int) 
  (h1 : divisor = 800)
  (h2 : quotient = 594)
  (h3 : remainder = -968) :
  divisor * quotient + remainder = 474232 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l888_88852


namespace NUMINAMATH_CALUDE_inverse_of_proposition_l888_88879

-- Define the original proposition
def original_proposition (x : ℝ) : Prop := x > 2 → x > 1

-- Define the inverse proposition
def inverse_proposition (x : ℝ) : Prop := x > 1 → x > 2

-- Theorem stating that the inverse_proposition is indeed the inverse of the original_proposition
theorem inverse_of_proposition :
  (∀ x, original_proposition x) ↔ (∀ x, inverse_proposition x) :=
sorry

end NUMINAMATH_CALUDE_inverse_of_proposition_l888_88879


namespace NUMINAMATH_CALUDE_thirty_sided_polygon_diagonals_l888_88853

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem thirty_sided_polygon_diagonals :
  num_diagonals 30 = 405 := by
  sorry

#eval num_diagonals 30  -- This will evaluate to 405

end NUMINAMATH_CALUDE_thirty_sided_polygon_diagonals_l888_88853


namespace NUMINAMATH_CALUDE_chord_length_square_of_quarter_circle_l888_88866

/-- Given a circular sector with central angle 90° and radius 10 cm,
    the square of the chord length connecting the arc endpoints is 200 cm². -/
theorem chord_length_square_of_quarter_circle (r : ℝ) (h : r = 10) :
  let chord_length_square := 2 * r^2
  chord_length_square = 200 := by sorry

end NUMINAMATH_CALUDE_chord_length_square_of_quarter_circle_l888_88866


namespace NUMINAMATH_CALUDE_lincoln_county_population_l888_88846

/-- The number of cities in the County of Lincoln -/
def num_cities : ℕ := 25

/-- The lower bound of the average population -/
def lower_bound : ℕ := 5200

/-- The upper bound of the average population -/
def upper_bound : ℕ := 5700

/-- The average population of the cities -/
def avg_population : ℚ := (lower_bound + upper_bound) / 2

/-- The total population of all cities -/
def total_population : ℕ := 136250

theorem lincoln_county_population :
  (num_cities : ℚ) * avg_population = total_population := by
  sorry

end NUMINAMATH_CALUDE_lincoln_county_population_l888_88846


namespace NUMINAMATH_CALUDE_cosine_fourth_minus_sine_fourth_l888_88837

theorem cosine_fourth_minus_sine_fourth (θ : ℝ) : 
  Real.cos θ ^ 4 - Real.sin θ ^ 4 = Real.cos (2 * θ) := by
  sorry

end NUMINAMATH_CALUDE_cosine_fourth_minus_sine_fourth_l888_88837


namespace NUMINAMATH_CALUDE_doubled_cost_percentage_doubled_cost_percentage_1600_l888_88815

-- Define the cost function
def cost (t : ℝ) (b : ℝ) : ℝ := t * b^4

-- Theorem statement
theorem doubled_cost_percentage (t : ℝ) (b : ℝ) (h : t > 0) (h2 : b > 0) :
  cost t (2 * b) = 16 * cost t b := by
  sorry

-- Corollary to express the result as a percentage
theorem doubled_cost_percentage_1600 (t : ℝ) (b : ℝ) (h : t > 0) (h2 : b > 0) :
  cost t (2 * b) / cost t b = 16 := by
  sorry

end NUMINAMATH_CALUDE_doubled_cost_percentage_doubled_cost_percentage_1600_l888_88815


namespace NUMINAMATH_CALUDE_a_bounds_circle_D_equation_l888_88855

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define the line L
def line_L (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the function a
def a (x y : ℝ) : ℝ := y - x

-- Theorem for the maximum and minimum values of a on circle C
theorem a_bounds :
  (∃ x y : ℝ, circle_C x y ∧ a x y = 2 * Real.sqrt 2 + 1) ∧
  (∃ x y : ℝ, circle_C x y ∧ a x y = 1 - 2 * Real.sqrt 2) ∧
  (∀ x y : ℝ, circle_C x y → 1 - 2 * Real.sqrt 2 ≤ a x y ∧ a x y ≤ 2 * Real.sqrt 2 + 1) :=
sorry

-- Define circle D
def circle_D (center_x center_y x y : ℝ) : Prop :=
  (x - center_x)^2 + (y - center_y)^2 = 9

-- Theorem for the equation of circle D
theorem circle_D_equation :
  ∃ center_x center_y : ℝ,
    line_L center_x center_y ∧
    ((∀ x y : ℝ, circle_D center_x center_y x y ↔ (x - 3)^2 + (y + 1)^2 = 9) ∨
     (∀ x y : ℝ, circle_D center_x center_y x y ↔ (x + 2)^2 + (y - 4)^2 = 9)) ∧
    (∃ x y : ℝ, circle_C x y ∧ (x - center_x)^2 + (y - center_y)^2 = 25) :=
sorry

end NUMINAMATH_CALUDE_a_bounds_circle_D_equation_l888_88855


namespace NUMINAMATH_CALUDE_complex_equation_solution_l888_88886

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 1 + Complex.I) → z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l888_88886


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l888_88808

/-- The repeating decimal 0.overline{43} -/
def repeating_decimal : ℚ := 43 / 99

theorem repeating_decimal_equals_fraction : 
  repeating_decimal = 43 / 99 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l888_88808


namespace NUMINAMATH_CALUDE_shortest_side_is_10_area_is_integer_l888_88867

/-- Represents a triangle with integer side lengths and area --/
structure IntegerTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  area : ℕ
  sum_eq_perimeter : a + b + c = 48
  one_side_eq_21 : a = 21 ∨ b = 21 ∨ c = 21

/-- The shortest side of the triangle is 10 --/
theorem shortest_side_is_10 (t : IntegerTriangle) : 
  min t.a (min t.b t.c) = 10 := by
  sorry

/-- The area of the triangle is an integer --/
theorem area_is_integer (t : IntegerTriangle) : 
  ∃ (s : ℕ), 4 * t.area = s * s * (t.a + t.b - t.c) * (t.a + t.c - t.b) * (t.b + t.c - t.a) := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_is_10_area_is_integer_l888_88867


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l888_88831

theorem ratio_of_numbers (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : a + b = 7 * (a - b)) : a / b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l888_88831


namespace NUMINAMATH_CALUDE_candles_remaining_l888_88817

def total_candles : ℕ := 40
def alyssa_fraction : ℚ := 1/2
def chelsea_fraction : ℚ := 70/100

theorem candles_remaining (total : ℕ) (alyssa_frac chelsea_frac : ℚ) : 
  total - (alyssa_frac * total).floor - (chelsea_frac * (total - (alyssa_frac * total).floor)).floor = 6 :=
by sorry

#check candles_remaining total_candles alyssa_fraction chelsea_fraction

end NUMINAMATH_CALUDE_candles_remaining_l888_88817


namespace NUMINAMATH_CALUDE_teacher_arrangements_eq_144_l888_88827

/-- The number of ways to arrange 6 teachers (3 math, 2 English, 1 Chinese) such that the 3 math teachers are not adjacent -/
def teacher_arrangements : ℕ :=
  Nat.factorial 3 * (Nat.factorial 3 * Nat.choose 4 3)

theorem teacher_arrangements_eq_144 : teacher_arrangements = 144 := by
  sorry

end NUMINAMATH_CALUDE_teacher_arrangements_eq_144_l888_88827


namespace NUMINAMATH_CALUDE_quadratic_inequality_l888_88880

theorem quadratic_inequality (m : ℝ) :
  (∀ x : ℝ, m * x^2 + 2 * m * x - 1 < 0) ↔ (-1 < m ∧ m ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l888_88880


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l888_88801

/-- 
Given a quantity y divided into three parts proportional to 1, 3, and 5,
the smallest part is equal to y/9.
-/
theorem smallest_part_of_proportional_division (y : ℝ) : 
  ∃ (x₁ x₂ x₃ : ℝ), 
    x₁ + x₂ + x₃ = y ∧ 
    x₂ = 3 * x₁ ∧ 
    x₃ = 5 * x₁ ∧ 
    x₁ = y / 9 ∧
    x₁ ≤ x₂ ∧ 
    x₁ ≤ x₃ := by
  sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l888_88801


namespace NUMINAMATH_CALUDE_constant_term_expansion_l888_88810

/-- The constant term in the expansion of (x-1)(x^2- 1/x)^6 is -15 -/
theorem constant_term_expansion : ∃ (f : ℝ → ℝ), 
  (∀ x ≠ 0, f x = (x - 1) * (x^2 - 1/x)^6) ∧ 
  (∃ c : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x - c| < ε) ∧
  (∃ c : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x - c| < ε) → c = -15 :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l888_88810


namespace NUMINAMATH_CALUDE_tan_product_pi_ninths_l888_88845

theorem tan_product_pi_ninths : 
  Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_pi_ninths_l888_88845


namespace NUMINAMATH_CALUDE_marathon_theorem_l888_88838

def marathon_problem (total_miles : ℝ) (day1_percent : ℝ) (day3_miles : ℝ) : Prop :=
  let day1_miles := total_miles * day1_percent / 100
  let remaining_miles := total_miles - day1_miles
  let day2_miles := total_miles - day1_miles - day3_miles
  (day2_miles / remaining_miles) * 100 = 50

theorem marathon_theorem : 
  marathon_problem 70 20 28 := by
  sorry

end NUMINAMATH_CALUDE_marathon_theorem_l888_88838
