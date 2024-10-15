import Mathlib

namespace NUMINAMATH_CALUDE_product_of_distinct_roots_l2228_222882

theorem product_of_distinct_roots (x y : ℝ) : 
  x ≠ 0 → y ≠ 0 → x ≠ y → (x + 6 / x = y + 6 / y) → x * y = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_of_distinct_roots_l2228_222882


namespace NUMINAMATH_CALUDE_percentage_equality_l2228_222819

theorem percentage_equality : ∃ P : ℝ, (P / 100) * 400 = (20 / 100) * 700 ∧ P = 35 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l2228_222819


namespace NUMINAMATH_CALUDE_average_speed_problem_l2228_222899

/-- The average speed for an hour drive, given that driving twice as fast for 4 hours covers 528 miles. -/
theorem average_speed_problem (v : ℝ) : v > 0 → 2 * v * 4 = 528 → v = 66 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_problem_l2228_222899


namespace NUMINAMATH_CALUDE_height_difference_pablo_charlene_l2228_222872

/-- Given the heights of various people, prove the height difference between Pablo and Charlene. -/
theorem height_difference_pablo_charlene :
  ∀ (height_janet height_ruby height_pablo height_charlene : ℕ),
  height_janet = 62 →
  height_charlene = 2 * height_janet →
  height_ruby = 192 →
  height_pablo = height_ruby + 2 →
  height_pablo - height_charlene = 70 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_pablo_charlene_l2228_222872


namespace NUMINAMATH_CALUDE_women_average_age_l2228_222854

theorem women_average_age (n : ℕ) (A : ℝ) :
  n = 12 ∧
  (n * (A + 3) = n * A + 3 * 42 - (25 + 30 + 35)) →
  42 = (3 * 42) / 3 :=
by sorry

end NUMINAMATH_CALUDE_women_average_age_l2228_222854


namespace NUMINAMATH_CALUDE_thirty_blocks_differ_in_two_ways_l2228_222857

/-- Represents the number of options for each property of a block -/
structure BlockOptions :=
  (material : Nat)
  (size : Nat)
  (color : Nat)
  (shape : Nat)

/-- Calculates the number of blocks that differ in exactly k ways from a specific block -/
def blocksWithKDifferences (options : BlockOptions) (k : Nat) : Nat :=
  sorry

/-- The specific block options for our problem -/
def ourOptions : BlockOptions :=
  { material := 2, size := 4, color := 4, shape := 4 }

/-- The main theorem stating that 30 blocks differ in exactly 2 ways -/
theorem thirty_blocks_differ_in_two_ways :
  blocksWithKDifferences ourOptions 2 = 30 := by sorry

end NUMINAMATH_CALUDE_thirty_blocks_differ_in_two_ways_l2228_222857


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l2228_222833

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![7, -2; -3, 6]
  Matrix.det A = 36 := by
sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l2228_222833


namespace NUMINAMATH_CALUDE_jesse_blocks_count_l2228_222802

theorem jesse_blocks_count (building_blocks farm_blocks fence_blocks remaining_blocks : ℕ) 
  (h1 : building_blocks = 80)
  (h2 : farm_blocks = 123)
  (h3 : fence_blocks = 57)
  (h4 : remaining_blocks = 84) :
  building_blocks + farm_blocks + fence_blocks + remaining_blocks = 344 :=
by sorry

end NUMINAMATH_CALUDE_jesse_blocks_count_l2228_222802


namespace NUMINAMATH_CALUDE_parabola_point_y_coordinate_l2228_222871

/-- The y-coordinate of a point on a parabola at a given distance from the focus -/
theorem parabola_point_y_coordinate (x y : ℝ) :
  y = -4 * x^2 →  -- Point M is on the parabola y = -4x²
  (x^2 + (y - 1/4)^2) = 1 →  -- Distance from M to focus (0, 1/4) is 1
  y = -15/16 := by
sorry

end NUMINAMATH_CALUDE_parabola_point_y_coordinate_l2228_222871


namespace NUMINAMATH_CALUDE_andrews_yearly_donation_l2228_222879

/-- Calculates the yearly donation amount given the starting age, current age, and total donation --/
def yearly_donation (start_age : ℕ) (current_age : ℕ) (total_donation : ℕ) : ℚ :=
  (total_donation : ℚ) / ((current_age - start_age) : ℚ)

/-- Theorem stating that Andrew's yearly donation is approximately 7388.89 --/
theorem andrews_yearly_donation :
  let start_age : ℕ := 11
  let current_age : ℕ := 29
  let total_donation : ℕ := 133000
  abs (yearly_donation start_age current_age total_donation - 7388.89) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_andrews_yearly_donation_l2228_222879


namespace NUMINAMATH_CALUDE_shift_graph_l2228_222859

-- Define a function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem shift_graph (h : f 0 = 1) : f ((-1) + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_shift_graph_l2228_222859


namespace NUMINAMATH_CALUDE_baker_cakes_sold_l2228_222838

/-- The number of cakes a baker intends to sell given certain pricing conditions -/
theorem baker_cakes_sold (n : ℝ) (h1 : n > 0) : ∃ x : ℕ,
  (n * x = 320) ∧
  (0.8 * n * (x + 2) = 320) ∧
  (x = 8) := by
sorry

end NUMINAMATH_CALUDE_baker_cakes_sold_l2228_222838


namespace NUMINAMATH_CALUDE_parallel_line_slope_l2228_222860

/-- Given a line with equation 2x - 4y = 9, prove that any parallel line has slope 1/2 -/
theorem parallel_line_slope (x y : ℝ) :
  (2 * x - 4 * y = 9) → (slope_of_parallel_line : ℝ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l2228_222860


namespace NUMINAMATH_CALUDE_algorithm_output_is_36_l2228_222820

def algorithm_result : ℕ := 
  let s := (List.range 3).foldl (fun acc i => acc + (i + 1)) 0
  let t := (List.range 3).foldl (fun acc i => acc * (i + 1)) 1
  s * t

theorem algorithm_output_is_36 : algorithm_result = 36 := by
  sorry

end NUMINAMATH_CALUDE_algorithm_output_is_36_l2228_222820


namespace NUMINAMATH_CALUDE_paths_count_is_36_l2228_222842

/-- Represents the circular arrangement of numbers -/
structure CircularArrangement where
  center : Nat
  surrounding : Nat
  zeroAdjacent : Nat
  fiveAdjacent : Nat

/-- Calculates the number of distinct paths to form 2005 -/
def countPaths (arrangement : CircularArrangement) : Nat :=
  arrangement.surrounding * arrangement.zeroAdjacent * arrangement.fiveAdjacent

/-- The specific arrangement for the problem -/
def problemArrangement : CircularArrangement :=
  { center := 2
  , surrounding := 6
  , zeroAdjacent := 2
  , fiveAdjacent := 3 }

theorem paths_count_is_36 :
  countPaths problemArrangement = 36 := by
  sorry

end NUMINAMATH_CALUDE_paths_count_is_36_l2228_222842


namespace NUMINAMATH_CALUDE_point_A_in_second_quadrant_l2228_222824

/-- A point in the 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def isInSecondQuadrant (p : Point2D) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem: The point A(-2, 3) is in the second quadrant -/
theorem point_A_in_second_quadrant :
  let A : Point2D := ⟨-2, 3⟩
  isInSecondQuadrant A := by
  sorry


end NUMINAMATH_CALUDE_point_A_in_second_quadrant_l2228_222824


namespace NUMINAMATH_CALUDE_subset_implies_a_values_l2228_222851

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem subset_implies_a_values :
  ∀ a : ℝ, (B a ⊆ A) ↔ (a = 0 ∨ a = 1/3 ∨ a = 1/5) := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_values_l2228_222851


namespace NUMINAMATH_CALUDE_plane_P_satisfies_conditions_l2228_222895

def plane1 (x y z : ℝ) : ℝ := 2*x - y + 2*z - 4
def plane2 (x y z : ℝ) : ℝ := 3*x + 4*y - z - 6
def planeP (x y z : ℝ) : ℝ := x + 63*y - 35*z - 34

def point : ℝ × ℝ × ℝ := (4, -2, 2)

theorem plane_P_satisfies_conditions :
  (∀ x y z, plane1 x y z = 0 ∧ plane2 x y z = 0 → planeP x y z = 0) ∧
  (planeP ≠ plane1 ∧ planeP ≠ plane2) ∧
  (abs (planeP point.1 point.2.1 point.2.2) / 
   Real.sqrt ((1:ℝ)^2 + 63^2 + (-35)^2) = 3 / Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_plane_P_satisfies_conditions_l2228_222895


namespace NUMINAMATH_CALUDE_f_properties_l2228_222801

noncomputable def f (x a : ℝ) := 2 * (Real.cos x)^2 + Real.sin (2 * x) + a

theorem f_properties (a : ℝ) :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f x a = f (x + T) a ∧
    ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f x a = f (x + T') a) → T ≤ T') ∧
  (∀ (k : ℤ), ∀ (x : ℝ), x ∈ Set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8) →
    ∀ (y : ℝ), y ∈ Set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8) →
      x ≤ y → f x a ≤ f y a) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 6) → f x a ≤ 2) →
  a = 1 - Real.sqrt 2 ∧
  ∀ (k : ℤ), ∀ (x : ℝ), f x a = f (k * Real.pi + Real.pi / 4 - x) a :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2228_222801


namespace NUMINAMATH_CALUDE_original_network_engineers_l2228_222876

/-- The number of new network engineers hired from University A -/
def new_hires : ℕ := 8

/-- The fraction of network engineers from University A after hiring -/
def fraction_after : ℚ := 3/4

/-- The fraction of original network engineers from University A -/
def fraction_original : ℚ := 13/20

/-- The original number of network engineers -/
def original_count : ℕ := 20

theorem original_network_engineers :
  ∃ (o : ℕ), 
    (o : ℚ) * fraction_original + new_hires = 
    ((o : ℚ) + new_hires) * fraction_after ∧
    o = original_count :=
by sorry

end NUMINAMATH_CALUDE_original_network_engineers_l2228_222876


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l2228_222889

theorem largest_angle_in_triangle : ∀ x : ℝ,
  x + 35 + 70 = 180 →
  max x (max 35 70) = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l2228_222889


namespace NUMINAMATH_CALUDE_f_properties_l2228_222806

noncomputable def f (x : ℝ) := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  (∀ (x : ℝ), 0 < x → x ≤ π / 3 → 2 ≤ f x ∧ f x ≤ 3) ∧
  (∃ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ ≤ π / 3 ∧ 0 < x₂ ∧ x₂ ≤ π / 3 ∧ f x₁ = 2 ∧ f x₂ = 3) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2228_222806


namespace NUMINAMATH_CALUDE_circle_cover_theorem_l2228_222858

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in a plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Check if a point is inside or on the boundary of a circle -/
def Point.insideCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2

/-- Check if a set of points can be covered by a circle -/
def coveredBy (points : Set Point) (c : Circle) : Prop :=
  ∀ p ∈ points, p.insideCircle c

/-- Main theorem -/
theorem circle_cover_theorem (n : ℕ) (points : Set Point) 
  (h : ∀ (p1 p2 p3 : Point), p1 ∈ points → p2 ∈ points → p3 ∈ points → 
    ∃ (c : Circle), c.radius = 1 ∧ coveredBy {p1, p2, p3} c) :
  ∃ (c : Circle), c.radius = 1 ∧ coveredBy points c := by
  sorry

end NUMINAMATH_CALUDE_circle_cover_theorem_l2228_222858


namespace NUMINAMATH_CALUDE_go_match_probability_l2228_222829

/-- The probability that two more games will conclude a Go match given the specified conditions -/
theorem go_match_probability (p_a : ℝ) (p_b : ℝ) : 
  p_a = 0.6 →
  p_b = 0.4 →
  p_a + p_b = 1 →
  (p_a ^ 2 + p_b ^ 2 : ℝ) = 0.52 := by
  sorry

end NUMINAMATH_CALUDE_go_match_probability_l2228_222829


namespace NUMINAMATH_CALUDE_computer_price_l2228_222894

theorem computer_price (new_price : ℝ) (price_increase : ℝ) (double_original : ℝ) 
  (h1 : price_increase = 0.3)
  (h2 : new_price = 377)
  (h3 : double_original = 580) : 
  ∃ (original_price : ℝ), 
    original_price * (1 + price_increase) = new_price ∧ 
    2 * original_price = double_original ∧
    original_price = 290 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_l2228_222894


namespace NUMINAMATH_CALUDE_registration_methods_count_l2228_222809

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of extracurricular activity groups -/
def num_groups : ℕ := 3

/-- Each student must sign up for exactly one group -/
axiom one_group_per_student : True

/-- The total number of different registration methods -/
def total_registration_methods : ℕ := num_groups ^ num_students

theorem registration_methods_count :
  total_registration_methods = 3^4 :=
by sorry

end NUMINAMATH_CALUDE_registration_methods_count_l2228_222809


namespace NUMINAMATH_CALUDE_sqrt_54_minus_sqrt_6_l2228_222852

theorem sqrt_54_minus_sqrt_6 : Real.sqrt 54 - Real.sqrt 6 = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_54_minus_sqrt_6_l2228_222852


namespace NUMINAMATH_CALUDE_union_of_sets_l2228_222868

theorem union_of_sets : 
  let A : Set ℤ := {1, 3, 5, 6}
  let B : Set ℤ := {-1, 5, 7}
  A ∪ B = {-1, 1, 3, 5, 6, 7} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l2228_222868


namespace NUMINAMATH_CALUDE_equation_solution_l2228_222848

theorem equation_solution : 
  ∀ x : ℝ, x^4 + (3 - x)^4 = 82 ↔ x = 2.5 ∨ x = 0.5 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2228_222848


namespace NUMINAMATH_CALUDE_certain_number_proof_l2228_222811

theorem certain_number_proof (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * n * 49) : n = 5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2228_222811


namespace NUMINAMATH_CALUDE_shoes_polished_percentage_l2228_222830

def shoes_polished (pairs : ℕ) (left_to_polish : ℕ) : ℚ :=
  let total_shoes := 2 * pairs
  let polished := total_shoes - left_to_polish
  (polished : ℚ) / (total_shoes : ℚ) * 100

theorem shoes_polished_percentage :
  shoes_polished 10 11 = 45 := by
  sorry

end NUMINAMATH_CALUDE_shoes_polished_percentage_l2228_222830


namespace NUMINAMATH_CALUDE_product_of_numbers_l2228_222834

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 460) : x * y = 40 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2228_222834


namespace NUMINAMATH_CALUDE_fraction_problem_l2228_222843

theorem fraction_problem (N : ℝ) (x : ℝ) 
  (h1 : N = 24.000000000000004) 
  (h2 : (1/4) * N = x * (N + 1) + 1) : 
  x = 0.20000000000000004 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2228_222843


namespace NUMINAMATH_CALUDE_leak_drain_time_l2228_222817

-- Define the pump filling rate
def pump_rate : ℚ := 1 / 2

-- Define the time to fill with leak
def fill_time_with_leak : ℚ := 7 / 3

-- Define the combined rate (pump - leak)
def combined_rate : ℚ := 1 / fill_time_with_leak

-- Define the leak rate
def leak_rate : ℚ := pump_rate - combined_rate

-- Theorem statement
theorem leak_drain_time (pump_rate : ℚ) (fill_time_with_leak : ℚ) 
  (combined_rate : ℚ) (leak_rate : ℚ) :
  pump_rate = 1 / 2 →
  fill_time_with_leak = 7 / 3 →
  combined_rate = 1 / fill_time_with_leak →
  leak_rate = pump_rate - combined_rate →
  1 / leak_rate = 14 := by
  sorry

end NUMINAMATH_CALUDE_leak_drain_time_l2228_222817


namespace NUMINAMATH_CALUDE_sun_division_l2228_222810

theorem sun_division (x y z total : ℝ) : 
  (∀ (r : ℝ), r > 0 → y = 0.45 * r ∧ z = 0.3 * r) →  -- For each rupee x gets, y gets 45 paisa and z gets 30 paisa
  y = 45 →  -- y's share is Rs. 45
  total = x + y + z →  -- Total is the sum of all shares
  total = 175 :=  -- The total amount is Rs. 175
by sorry

end NUMINAMATH_CALUDE_sun_division_l2228_222810


namespace NUMINAMATH_CALUDE_fish_caught_difference_l2228_222844

/-- Represents the number of fish caught by each fisherman -/
def fish_caught (season_length first_rate second_rate_1 second_rate_2 second_rate_3 : ℕ) 
  (second_period_1 second_period_2 : ℕ) : ℕ := 
  let first_total := first_rate * season_length
  let second_total := second_rate_1 * second_period_1 + 
                      second_rate_2 * second_period_2 + 
                      second_rate_3 * (season_length - second_period_1 - second_period_2)
  (max first_total second_total) - (min first_total second_total)

/-- The difference in fish caught between the two fishermen is 3 -/
theorem fish_caught_difference : 
  fish_caught 213 3 1 2 4 30 60 = 3 := by sorry

end NUMINAMATH_CALUDE_fish_caught_difference_l2228_222844


namespace NUMINAMATH_CALUDE_smithtown_left_handed_women_percentage_l2228_222892

/-- Represents the population of Smithtown -/
structure Population where
  right_handed : ℕ  -- Number of right-handed people
  left_handed : ℕ   -- Number of left-handed people
  men : ℕ           -- Number of men
  women : ℕ         -- Number of women

/-- The conditions of the Smithtown population problem -/
def smithtown_conditions (p : Population) : Prop :=
  -- Ratio of right-handed to left-handed is 3:1
  p.right_handed = 3 * p.left_handed ∧
  -- Ratio of men to women is 3:2
  3 * p.women = 2 * p.men ∧
  -- Number of right-handed men is maximized (all men are right-handed)
  p.men = p.right_handed

/-- The theorem stating that 25% of the population are left-handed women -/
theorem smithtown_left_handed_women_percentage (p : Population)
  (h : smithtown_conditions p) :
  (p.left_handed : ℚ) / (p.right_handed + p.left_handed : ℚ) = 1/4 := by
  sorry

#check smithtown_left_handed_women_percentage

end NUMINAMATH_CALUDE_smithtown_left_handed_women_percentage_l2228_222892


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l2228_222865

theorem sqrt_expression_equality : 
  (Real.sqrt 3 + Real.sqrt 2 - 1) * (Real.sqrt 3 - Real.sqrt 2 + 1) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l2228_222865


namespace NUMINAMATH_CALUDE_shortest_path_on_cube_is_four_l2228_222888

/-- The shortest path on the surface of a regular cube with edge length 2,
    from one corner to the opposite corner. -/
def shortest_path_on_cube : ℝ := 4

/-- Proof that the shortest path on the surface of a regular cube with edge length 2,
    from one corner to the opposite corner, is equal to 4. -/
theorem shortest_path_on_cube_is_four :
  shortest_path_on_cube = 4 := by sorry

end NUMINAMATH_CALUDE_shortest_path_on_cube_is_four_l2228_222888


namespace NUMINAMATH_CALUDE_lowest_possible_score_l2228_222886

theorem lowest_possible_score (num_tests : Nat) (max_score : Nat) (avg_score : Nat) :
  num_tests = 4 →
  max_score = 100 →
  avg_score = 88 →
  ∃ (scores : Fin num_tests → Nat),
    (∀ i, scores i ≤ max_score) ∧
    (Finset.sum Finset.univ (λ i => scores i) = num_tests * avg_score) ∧
    (∃ i, scores i = 52) ∧
    (∀ i, scores i ≥ 52) :=
by
  sorry

#check lowest_possible_score

end NUMINAMATH_CALUDE_lowest_possible_score_l2228_222886


namespace NUMINAMATH_CALUDE_stating_ladder_of_twos_theorem_l2228_222891

/-- 
A function that represents the number of distinct integers obtainable 
from a ladder of n twos by placing nested parentheses.
-/
def ladder_of_twos (n : ℕ) : ℕ :=
  if n ≥ 3 then 2^(n-3) else 0

/-- 
Theorem stating that for n ≥ 3, the number of distinct integers obtainable 
from a ladder of n twos by placing nested parentheses is 2^(n-3).
-/
theorem ladder_of_twos_theorem (n : ℕ) (h : n ≥ 3) : 
  ladder_of_twos n = 2^(n-3) := by
  sorry

end NUMINAMATH_CALUDE_stating_ladder_of_twos_theorem_l2228_222891


namespace NUMINAMATH_CALUDE_units_digit_of_7_pow_6_pow_5_l2228_222898

-- Define the cycle of units digits for powers of 7
def units_cycle : List Nat := [7, 9, 3, 1]

-- Theorem statement
theorem units_digit_of_7_pow_6_pow_5 : 
  ∃ (n : Nat), 7^(6^5) ≡ 7 [ZMOD 10] ∧ n = 7^(6^5) % 10 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_pow_6_pow_5_l2228_222898


namespace NUMINAMATH_CALUDE_matrix_determinant_l2228_222875

def matrix : Matrix (Fin 2) (Fin 2) ℤ := !![7, -2; -3, 6]

theorem matrix_determinant :
  Matrix.det matrix = 36 := by sorry

end NUMINAMATH_CALUDE_matrix_determinant_l2228_222875


namespace NUMINAMATH_CALUDE_solve_a_given_set_membership_l2228_222866

theorem solve_a_given_set_membership (a : ℝ) : 
  -3 ∈ ({a - 3, 2 * a - 1, a^2 + 1} : Set ℝ) → a = 0 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_a_given_set_membership_l2228_222866


namespace NUMINAMATH_CALUDE_avery_egg_cartons_l2228_222864

/-- Given the number of chickens, eggs per chicken, and number of filled cartons,
    calculate the number of eggs per carton. -/
def eggs_per_carton (num_chickens : ℕ) (eggs_per_chicken : ℕ) (num_cartons : ℕ) : ℕ :=
  (num_chickens * eggs_per_chicken) / num_cartons

/-- Prove that with 20 chickens laying 6 eggs each, filling 10 cartons results in 12 eggs per carton. -/
theorem avery_egg_cartons : eggs_per_carton 20 6 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_avery_egg_cartons_l2228_222864


namespace NUMINAMATH_CALUDE_sum_remainder_three_l2228_222815

theorem sum_remainder_three (m : ℤ) : (9 - m + (m + 4)) % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_three_l2228_222815


namespace NUMINAMATH_CALUDE_frank_unfilled_boxes_l2228_222800

/-- Given a total number of boxes and a number of filled boxes,
    calculate the number of unfilled boxes. -/
def unfilled_boxes (total : ℕ) (filled : ℕ) : ℕ :=
  total - filled

/-- Theorem: Frank has 5 unfilled boxes -/
theorem frank_unfilled_boxes :
  unfilled_boxes 13 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_frank_unfilled_boxes_l2228_222800


namespace NUMINAMATH_CALUDE_solution_to_equation_l2228_222837

theorem solution_to_equation (x : ℝ) (h : 1/4 - 1/5 = 1/x) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l2228_222837


namespace NUMINAMATH_CALUDE_divisibility_by_three_and_nine_l2228_222812

theorem divisibility_by_three_and_nine (n : ℕ) :
  (∃ (d₁ d₂ : ℕ), d₁ ≠ d₂ ∧ d₁ < 10 ∧ d₂ < 10 ∧ 
   (n * 10 + d₁) % 9 = 0 ∧ (n * 10 + d₂) % 9 = 0) →
  (∃! (digits : Finset ℕ), digits.card = 4 ∧ 
   ∀ d ∈ digits, d < 10 ∧ (n * 10 + d) % 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_three_and_nine_l2228_222812


namespace NUMINAMATH_CALUDE_martin_bell_rings_l2228_222832

theorem martin_bell_rings (s b : ℕ) : s = 4 + b / 3 → s + b = 52 → b = 36 := by sorry

end NUMINAMATH_CALUDE_martin_bell_rings_l2228_222832


namespace NUMINAMATH_CALUDE_sum_x_coordinates_preserved_l2228_222862

/-- A polygon in the Cartesian plane -/
structure Polygon :=
  (vertices : List (ℝ × ℝ))

/-- Create a new polygon from the midpoints of the sides of a given polygon -/
def midpointPolygon (p : Polygon) : Polygon :=
  sorry

/-- Sum of x-coordinates of a polygon's vertices -/
def sumXCoordinates (p : Polygon) : ℝ :=
  sorry

theorem sum_x_coordinates_preserved (n : ℕ) (Q1 : Polygon) 
  (h1 : Q1.vertices.length = n)
  (Q2 := midpointPolygon Q1)
  (Q3 := midpointPolygon Q2) :
  sumXCoordinates Q3 = sumXCoordinates Q1 :=
sorry

end NUMINAMATH_CALUDE_sum_x_coordinates_preserved_l2228_222862


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l2228_222883

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 10)
  (h2 : selling_price = 15) : 
  (selling_price - cost_price) / cost_price * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l2228_222883


namespace NUMINAMATH_CALUDE_ratio_problem_l2228_222855

theorem ratio_problem (a b c d e f : ℚ) 
  (h1 : a * b * c / (d * e * f) = 1.875)
  (h2 : a / b = 5 / 2)
  (h3 : b / c = 1 / 2)
  (h4 : c / d = 1)
  (h5 : d / e = 3 / 2) :
  e / f = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2228_222855


namespace NUMINAMATH_CALUDE_inequality_proof_l2228_222867

theorem inequality_proof (m n : ℝ) (h1 : m > n) (h2 : n > 0) : 
  m * Real.exp n + n < n * Real.exp m + m := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2228_222867


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2228_222822

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ((2*x^2 + 5*x + 2)*(2*y^2 + 5*y + 2)*(2*z^2 + 5*z + 2)) / (x*y*z*(1+x)*(1+y)*(1+z)) ≥ 729/8 :=
by sorry

theorem min_value_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  ((2*x^2 + 5*x + 2)*(2*y^2 + 5*y + 2)*(2*z^2 + 5*z + 2)) / (x*y*z*(1+x)*(1+y)*(1+z)) = 729/8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2228_222822


namespace NUMINAMATH_CALUDE_max_value_of_a_l2228_222828

theorem max_value_of_a (a : ℝ) : 
  (∀ x : ℝ, x < a → x^2 > 1) ∧ 
  (∃ x : ℝ, x^2 > 1 ∧ x ≥ a) → 
  a ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l2228_222828


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l2228_222881

theorem arithmetic_sequence_tenth_term :
  let a : ℚ := 1/4  -- First term
  let d : ℚ := 1/2  -- Common difference
  let n : ℕ := 10   -- Term number we're looking for
  let a_n : ℚ := a + (n - 1) * d  -- Formula for nth term of arithmetic sequence
  a_n = 19/4 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l2228_222881


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2228_222821

theorem unique_solution_condition (a b : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + a = (b - 1) * x + 2) ↔ b ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2228_222821


namespace NUMINAMATH_CALUDE_z_ratio_equals_neg_i_l2228_222840

-- Define the complex numbers z₁ and z₂
variable (z₁ z₂ : ℂ)

-- Define the condition that z₁ and z₂ are symmetric with respect to the imaginary axis
def symmetric_wrt_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

-- Theorem statement
theorem z_ratio_equals_neg_i
  (h_sym : symmetric_wrt_imaginary_axis z₁ z₂)
  (h_z₁ : z₁ = 1 + I) :
  z₁ / z₂ = -I :=
sorry

end NUMINAMATH_CALUDE_z_ratio_equals_neg_i_l2228_222840


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2228_222816

theorem complex_equation_solution (i : ℂ) (m : ℝ) 
  (h1 : i ^ 2 = -1)
  (h2 : (2 : ℂ) / (1 + i) = 1 + m * i) : 
  m = -1 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2228_222816


namespace NUMINAMATH_CALUDE_shooting_probabilities_shooting_probabilities_alt_l2228_222890

-- Define the probabilities given in the problem
def p_9_or_more : ℝ := 0.56
def p_8 : ℝ := 0.22
def p_7 : ℝ := 0.12

-- Theorem statement
theorem shooting_probabilities :
  let p_less_than_8 := 1 - p_9_or_more - p_8
  let p_at_least_7 := p_7 + p_8 + p_9_or_more
  (p_less_than_8 = 0.22) ∧ (p_at_least_7 = 0.9) := by
  sorry

-- Alternative formulation using complement for p_at_least_7
theorem shooting_probabilities_alt :
  let p_less_than_7 := 1 - p_9_or_more - p_8 - p_7
  let p_less_than_8 := p_less_than_7 + p_7
  let p_at_least_7 := 1 - p_less_than_7
  (p_less_than_8 = 0.22) ∧ (p_at_least_7 = 0.9) := by
  sorry

end NUMINAMATH_CALUDE_shooting_probabilities_shooting_probabilities_alt_l2228_222890


namespace NUMINAMATH_CALUDE_rectangles_on_4x3_grid_l2228_222826

/-- The number of ways to choose k items from n items without replacement and where order doesn't matter. -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of rectangles that can be formed on a grid. -/
def rectangles_on_grid (columns rows : ℕ) : ℕ :=
  binomial columns 2 * binomial rows 2

/-- Theorem: The number of rectangles on a 4x3 grid is 18. -/
theorem rectangles_on_4x3_grid : rectangles_on_grid 4 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_on_4x3_grid_l2228_222826


namespace NUMINAMATH_CALUDE_range_of_x_l2228_222805

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_decreasing : ∀ x y, x ≤ y → y ≤ 0 → f y ≤ f x
axiom f_at_neg_one : f (-1) = 1/2

-- State the theorem
theorem range_of_x (x : ℝ) : 2 * f (2*x - 1) - 1 < 0 ↔ 0 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_range_of_x_l2228_222805


namespace NUMINAMATH_CALUDE_range_of_a_l2228_222831

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x - 10 < 0}
def B : Set ℝ := {x | x^2 + 2*x - 8 > 0}
def C (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < a + 3}

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, (((A ∩ B) ∩ C a) = C a) ↔ 1 ≤ a := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2228_222831


namespace NUMINAMATH_CALUDE_annulus_area_single_element_l2228_222896

/-- The area of an annulus can be expressed using only one linear element -/
theorem annulus_area_single_element (R r : ℝ) (h : R > r) :
  ∃ (d : ℝ), (d = R - r ∨ d = R + r) ∧
  (π * (R^2 - r^2) = π * d * (2*R - d) ∨ π * (R^2 - r^2) = π * d * (2*r + d)) := by
  sorry

end NUMINAMATH_CALUDE_annulus_area_single_element_l2228_222896


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l2228_222853

theorem quadratic_roots_properties (p : ℝ) (x₁ x₂ : ℂ) :
  x₁^2 + p * x₁ + 2 = 0 →
  x₂^2 + p * x₂ + 2 = 0 →
  x₁ = 1 + I →
  (x₂ = 1 - I ∧ x₁ / x₂ = I) := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l2228_222853


namespace NUMINAMATH_CALUDE_shaded_area_circle_configuration_l2228_222807

/-- The area of the shaded region in a circle configuration --/
theorem shaded_area_circle_configuration (R : ℝ) (h : R = 8) : 
  R^2 * Real.pi - 3 * (R/2)^2 * Real.pi = 16 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_circle_configuration_l2228_222807


namespace NUMINAMATH_CALUDE_f_property_f_upper_bound_minimum_M_l2228_222841

/-- The function f(x) = x^2 + bx + c -/
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

/-- The derivative of f(x) -/
def f_derivative (b : ℝ) (x : ℝ) : ℝ := 2*x + b

theorem f_property (b c : ℝ) :
  ∀ x, f_derivative b x ≤ f b c x := sorry

theorem f_upper_bound (b c : ℝ) (h : ∀ x, f_derivative b x ≤ f b c x) :
  ∀ x ≥ 0, f b c x ≤ (x + c)^2 := sorry

theorem minimum_M (b c : ℝ) (h : ∀ x, f_derivative b x ≤ f b c x) :
  ∃ M, (∀ b c, f b c c - f b c b ≤ M * (c^2 - b^2)) ∧
       (∀ M', (∀ b c, f b c c - f b c b ≤ M' * (c^2 - b^2)) → M ≤ M') ∧
       M = 3/2 := sorry

end NUMINAMATH_CALUDE_f_property_f_upper_bound_minimum_M_l2228_222841


namespace NUMINAMATH_CALUDE_crayon_selection_ways_l2228_222863

/-- The number of ways to choose k items from n items without replacement and where order doesn't matter. -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of crayons in the box. -/
def total_crayons : ℕ := 15

/-- The number of crayons Karl selects. -/
def selected_crayons : ℕ := 5

/-- Theorem stating that selecting 5 crayons from 15 crayons can be done in 3003 ways. -/
theorem crayon_selection_ways : binomial total_crayons selected_crayons = 3003 := by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_ways_l2228_222863


namespace NUMINAMATH_CALUDE_rob_baseball_cards_l2228_222847

theorem rob_baseball_cards (rob_total : ℕ) (rob_doubles : ℕ) (jess_doubles : ℕ) :
  rob_doubles = rob_total / 3 →
  jess_doubles = 5 * rob_doubles →
  jess_doubles = 40 →
  rob_total = 24 := by
sorry

end NUMINAMATH_CALUDE_rob_baseball_cards_l2228_222847


namespace NUMINAMATH_CALUDE_goods_trade_scientific_notation_l2228_222804

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

/-- The value of one trillion -/
def trillion : ℝ := 1000000000000

theorem goods_trade_scientific_notation :
  to_scientific_notation (42.1 * trillion) =
    ScientificNotation.mk 4.21 13 (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_goods_trade_scientific_notation_l2228_222804


namespace NUMINAMATH_CALUDE_wheels_per_row_calculation_l2228_222869

/-- Calculates the number of wheels per row given the total number of wheels,
    number of trains, carriages per train, and rows of wheels per carriage. -/
def wheels_per_row (total_wheels : ℕ) (num_trains : ℕ) (carriages_per_train : ℕ) (rows_per_carriage : ℕ) : ℕ :=
  total_wheels / (num_trains * carriages_per_train * rows_per_carriage)

/-- Theorem stating that given 4 trains, 4 carriages per train, 3 rows of wheels per carriage,
    and 240 wheels in total, the number of wheels in each row is 5. -/
theorem wheels_per_row_calculation :
  wheels_per_row 240 4 4 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_wheels_per_row_calculation_l2228_222869


namespace NUMINAMATH_CALUDE_daves_sticks_l2228_222813

theorem daves_sticks (sticks_picked : ℕ) (sticks_left : ℕ) : 
  sticks_left = 4 → 
  sticks_picked - sticks_left = 10 → 
  sticks_picked = 14 := by
  sorry

end NUMINAMATH_CALUDE_daves_sticks_l2228_222813


namespace NUMINAMATH_CALUDE_coeff_bound_theorem_l2228_222885

/-- Represents a real polynomial -/
def RealPolynomial := ℝ → ℝ

/-- The degree of a polynomial -/
def degree (p : RealPolynomial) : ℕ := sorry

/-- The largest absolute value of the coefficients of a polynomial -/
def max_coeff (p : RealPolynomial) : ℝ := sorry

/-- Multiplication of polynomials -/
def poly_mul (p q : RealPolynomial) : RealPolynomial := sorry

/-- Addition of a constant to x -/
def add_const (a : ℝ) : RealPolynomial := sorry

theorem coeff_bound_theorem (p q : RealPolynomial) (a : ℝ) (n : ℕ) (h k : ℝ) :
  p = poly_mul (add_const a) q →
  degree p = n →
  max_coeff p = h →
  max_coeff q = k →
  k ≤ h * n := by sorry

end NUMINAMATH_CALUDE_coeff_bound_theorem_l2228_222885


namespace NUMINAMATH_CALUDE_expression_simplification_l2228_222870

theorem expression_simplification (y : ℝ) :
  3 * y - 5 * y^2 + 10 - (8 - 3 * y + 5 * y^2 - y^3) = y^3 - 10 * y^2 + 6 * y + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2228_222870


namespace NUMINAMATH_CALUDE_red_light_probability_is_two_fifths_l2228_222839

/-- Represents the duration of each light color in seconds -/
structure LightDuration where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the total cycle time of the traffic light -/
def totalCycleTime (d : LightDuration) : ℕ :=
  d.red + d.yellow + d.green

/-- Calculates the probability of seeing a red light -/
def redLightProbability (d : LightDuration) : ℚ :=
  d.red / totalCycleTime d

/-- Theorem: The probability of seeing a red light is 2/5 given the specified light durations -/
theorem red_light_probability_is_two_fifths (d : LightDuration) 
    (h1 : d.red = 30)
    (h2 : d.yellow = 5)
    (h3 : d.green = 40) : 
  redLightProbability d = 2/5 := by
  sorry

#eval redLightProbability ⟨30, 5, 40⟩

end NUMINAMATH_CALUDE_red_light_probability_is_two_fifths_l2228_222839


namespace NUMINAMATH_CALUDE_yellow_balls_count_l2228_222877

/-- The number of yellow balls in a bag, given the total number of balls,
    the number of balls of each color (except yellow), and the probability
    of choosing a ball that is neither red nor purple. -/
def yellow_balls (total : ℕ) (white green red purple : ℕ) (prob_not_red_purple : ℚ) : ℕ :=
  total - white - green - red - purple

/-- Theorem stating that the number of yellow balls is 5 under the given conditions. -/
theorem yellow_balls_count :
  yellow_balls 60 22 18 6 9 (3/4) = 5 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l2228_222877


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2228_222845

def solution_set : Set (ℕ × ℕ) :=
  {(5, 20), (6, 12), (8, 8), (12, 6), (20, 5)}

theorem diophantine_equation_solution :
  ∀ x y : ℕ, x > 0 ∧ y > 0 →
    (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 4 ↔ (x, y) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2228_222845


namespace NUMINAMATH_CALUDE_office_printer_paper_duration_l2228_222818

/-- The number of days printer paper will last given the number of packs, sheets per pack, and daily usage. -/
def printer_paper_duration (packs : ℕ) (sheets_per_pack : ℕ) (daily_usage : ℕ) : ℕ :=
  (packs * sheets_per_pack) / daily_usage

/-- Theorem stating that under the given conditions, the printer paper will last 6 days. -/
theorem office_printer_paper_duration :
  let packs : ℕ := 2
  let sheets_per_pack : ℕ := 240
  let daily_usage : ℕ := 80
  printer_paper_duration packs sheets_per_pack daily_usage = 6 := by
  sorry


end NUMINAMATH_CALUDE_office_printer_paper_duration_l2228_222818


namespace NUMINAMATH_CALUDE_cost_of_72_tulips_l2228_222835

/-- Represents the cost of tulips given the number of tulips -/
def tulip_cost (n : ℕ) : ℚ :=
  let base_cost := (36 : ℚ) * n / 18
  if n > 50 then base_cost * (1 - 1/5) else base_cost

/-- Theorem stating the cost of 72 tulips -/
theorem cost_of_72_tulips : tulip_cost 72 = 115.2 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_72_tulips_l2228_222835


namespace NUMINAMATH_CALUDE_bella_max_number_l2228_222849

theorem bella_max_number : 
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 → 3 * (250 - n) ≤ 720 ∧ 
  ∃ m : ℕ, 10 ≤ m ∧ m ≤ 99 ∧ 3 * (250 - m) = 720 :=
by sorry

end NUMINAMATH_CALUDE_bella_max_number_l2228_222849


namespace NUMINAMATH_CALUDE_smallest_power_congruence_l2228_222836

/-- For any integer r ≥ 3, the smallest positive integer d₀ such that 7^d₀ ≡ 1 (mod 2^r) is 2^(r-2) -/
theorem smallest_power_congruence (r : ℕ) (hr : r ≥ 3) :
  (∃ (d₀ : ℕ), d₀ > 0 ∧ 7^d₀ ≡ 1 [MOD 2^r] ∧
    ∀ (d : ℕ), d > 0 → 7^d ≡ 1 [MOD 2^r] → d₀ ≤ d) ∧
  (∀ (d₀ : ℕ), d₀ > 0 → 7^d₀ ≡ 1 [MOD 2^r] ∧
    (∀ (d : ℕ), d > 0 → 7^d ≡ 1 [MOD 2^r] → d₀ ≤ d) →
    d₀ = 2^(r-2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_power_congruence_l2228_222836


namespace NUMINAMATH_CALUDE_remaining_cube_volume_l2228_222884

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem remaining_cube_volume (cube_side : ℝ) (cylinder_radius : ℝ) : 
  cube_side = 5 → cylinder_radius = 1.5 → 
  cube_side^3 - π * cylinder_radius^2 * cube_side = 125 - 11.25 * π := by
  sorry

#check remaining_cube_volume

end NUMINAMATH_CALUDE_remaining_cube_volume_l2228_222884


namespace NUMINAMATH_CALUDE_tree_height_after_two_years_l2228_222887

/-- The height of a tree after a given number of years, given that it triples its height every year -/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

/-- Theorem stating that a tree tripling its height yearly reaches 9 feet after 2 years if it's 81 feet after 4 years -/
theorem tree_height_after_two_years 
  (h : tree_height (tree_height h₀ 2) 2 = 81) : 
  tree_height h₀ 2 = 9 :=
by
  sorry

#check tree_height_after_two_years

end NUMINAMATH_CALUDE_tree_height_after_two_years_l2228_222887


namespace NUMINAMATH_CALUDE_negative_integer_squared_plus_self_equals_twelve_l2228_222827

theorem negative_integer_squared_plus_self_equals_twelve (N : ℤ) : 
  N < 0 → N^2 + N = 12 → N = -4 := by sorry

end NUMINAMATH_CALUDE_negative_integer_squared_plus_self_equals_twelve_l2228_222827


namespace NUMINAMATH_CALUDE_ruby_apples_l2228_222873

/-- The number of apples Ruby has initially -/
def initial_apples : ℕ := 63

/-- The number of apples Emily takes away -/
def apples_taken : ℕ := 55

/-- The number of apples Ruby has after Emily takes some away -/
def remaining_apples : ℕ := initial_apples - apples_taken

theorem ruby_apples : remaining_apples = 8 := by
  sorry

end NUMINAMATH_CALUDE_ruby_apples_l2228_222873


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l2228_222861

/-- A hyperbola with equation mx^2 + y^2 = 1 where the length of its imaginary axis
    is twice the length of its real axis -/
structure Hyperbola (m : ℝ) where
  equation : ∀ (x y : ℝ), m * x^2 + y^2 = 1
  axis_ratio : (imaginary_axis_length : ℝ) = 2 * (real_axis_length : ℝ)

/-- The value of m for a hyperbola with the given properties is -1/4 -/
theorem hyperbola_m_value (h : Hyperbola m) : m = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l2228_222861


namespace NUMINAMATH_CALUDE_cookfire_logs_proof_l2228_222893

/-- The number of logs burned per hour -/
def logs_burned_per_hour : ℕ := 3

/-- The number of logs added at the end of each hour -/
def logs_added_per_hour : ℕ := 2

/-- The number of hours the cookfire burns -/
def burn_duration : ℕ := 3

/-- The number of logs left after the burn duration -/
def logs_remaining : ℕ := 3

/-- The initial number of logs in the cookfire -/
def initial_logs : ℕ := 6

theorem cookfire_logs_proof :
  initial_logs - burn_duration * logs_burned_per_hour + (burn_duration - 1) * logs_added_per_hour = logs_remaining :=
by
  sorry

end NUMINAMATH_CALUDE_cookfire_logs_proof_l2228_222893


namespace NUMINAMATH_CALUDE_min_blue_chips_l2228_222846

theorem min_blue_chips (r w b : ℕ) : 
  r ≥ 2 * w →
  r ≤ 2 * b / 3 →
  r + w ≥ 72 →
  ∀ b' : ℕ, (∃ r' w' : ℕ, r' ≥ 2 * w' ∧ r' ≤ 2 * b' / 3 ∧ r' + w' ≥ 72) → b' ≥ 72 :=
by sorry

end NUMINAMATH_CALUDE_min_blue_chips_l2228_222846


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_one_l2228_222874

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 4*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 4

-- Theorem statement
theorem tangent_slope_at_point_one :
  f 1 = -3 ∧ f' 1 = -1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_one_l2228_222874


namespace NUMINAMATH_CALUDE_smallest_m_for_inequality_l2228_222803

theorem smallest_m_for_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_one : a + b + c = 1) :
  ∀ m : ℝ, (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 1 → 
    m * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1) → 
  m ≥ 27 :=
sorry

end NUMINAMATH_CALUDE_smallest_m_for_inequality_l2228_222803


namespace NUMINAMATH_CALUDE_film_festival_theorem_l2228_222814

theorem film_festival_theorem (n : ℕ) (m : ℕ) : 
  -- Total number of films
  n > 0 →
  -- Total number of viewers (2m, where m is the number of men/women)
  m > 0 →
  -- Each film is liked by exactly 8 viewers
  -- Each viewer likes the same number of films
  -- The total number of "likes" is 8n
  8 * n = 2 * m * (8 * n / (2 * m)) →
  -- At least 3/7 of the films are liked by at least two men
  ∃ (k : ℕ), k ≥ (3 * n + 6) / 7 ∧ 
    (∀ (i : ℕ), i < k → ∃ (male_viewers : ℕ), male_viewers ≥ 2 ∧ male_viewers ≤ 8) :=
by
  sorry

end NUMINAMATH_CALUDE_film_festival_theorem_l2228_222814


namespace NUMINAMATH_CALUDE_initial_sheep_count_l2228_222825

theorem initial_sheep_count (horses : ℕ) (chickens : ℕ) (goats : ℕ) (male_animals : ℕ) :
  horses = 100 →
  chickens = 9 →
  goats = 37 →
  male_animals = 53 →
  ∃ (sheep : ℕ), 
    (((horses + sheep + chickens) / 2 : ℚ) + goats : ℚ) = (2 * male_animals : ℚ) ∧
    sheep = 29 :=
by sorry

end NUMINAMATH_CALUDE_initial_sheep_count_l2228_222825


namespace NUMINAMATH_CALUDE_smallest_number_proof_l2228_222880

theorem smallest_number_proof (a b c d : ℝ) : 
  b = 4 * a →
  c = 2 * a →
  d = a + b + c →
  (a + b + c + d) / 4 = 77 →
  a = 22 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l2228_222880


namespace NUMINAMATH_CALUDE_least_possible_difference_l2228_222878

theorem least_possible_difference (x y z : ℤ) 
  (h_x_even : Even x)
  (h_y_odd : Odd y)
  (h_z_odd : Odd z)
  (h_order : x < y ∧ y < z)
  (h_diff : y - x > 5) :
  ∀ w, w = z - x → w ≥ 9 ∧ ∃ (a b c : ℤ), a - b = 9 ∧ Even b ∧ Odd a ∧ Odd c ∧ b < c ∧ c < a ∧ c - b > 5 := by
  sorry

end NUMINAMATH_CALUDE_least_possible_difference_l2228_222878


namespace NUMINAMATH_CALUDE_email_sending_ways_l2228_222856

theorem email_sending_ways (email_addresses : ℕ) (emails_to_send : ℕ) : 
  email_addresses = 3 → emails_to_send = 5 → email_addresses ^ emails_to_send = 243 := by
  sorry

end NUMINAMATH_CALUDE_email_sending_ways_l2228_222856


namespace NUMINAMATH_CALUDE_two_colonies_reach_limit_same_time_l2228_222823

/-- Represents the growth of a bacteria colony -/
structure BacteriaColony where
  growthRate : ℕ → ℕ
  limitDay : ℕ

/-- The number of days it takes for two colonies to reach the habitat's limit -/
def daysToLimitTwoColonies (colony : BacteriaColony) : ℕ := sorry

theorem two_colonies_reach_limit_same_time (colony : BacteriaColony) 
  (h1 : ∀ n : ℕ, colony.growthRate n = 2 * colony.growthRate (n - 1))
  (h2 : colony.limitDay = 16) :
  daysToLimitTwoColonies colony = colony.limitDay := by sorry

end NUMINAMATH_CALUDE_two_colonies_reach_limit_same_time_l2228_222823


namespace NUMINAMATH_CALUDE_gear_speed_ratio_l2228_222897

/-- Represents a gear with a number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  speed : ℝ

/-- Proves that for three interconnected gears, if the product of teeth and speed is equal,
    then the ratio of their speeds is proportional to the product of the other two gears' teeth -/
theorem gear_speed_ratio
  (G H I : Gear)
  (h : G.teeth * G.speed = H.teeth * H.speed ∧ H.teeth * H.speed = I.teeth * I.speed) :
  ∃ (k : ℝ), G.speed = k * (H.teeth * I.teeth) ∧
             H.speed = k * (G.teeth * I.teeth) ∧
             I.speed = k * (G.teeth * H.teeth) := by
  sorry

end NUMINAMATH_CALUDE_gear_speed_ratio_l2228_222897


namespace NUMINAMATH_CALUDE_fruit_purchase_cost_is_correct_l2228_222808

/-- Calculates the total cost in dollars for a fruit purchase with given conditions -/
def fruitPurchaseCost (grapeKg : ℝ) (grapeRate : ℝ) (mangoKg : ℝ) (mangoRate : ℝ)
                      (appleKg : ℝ) (appleRate : ℝ) (orangeKg : ℝ) (orangeRate : ℝ)
                      (grapeMangoeDiscountRate : ℝ) (appleOrangeFixedDiscount : ℝ)
                      (salesTaxRate : ℝ) (fixedTax : ℝ) (exchangeRate : ℝ) : ℝ :=
  let grapeCost := grapeKg * grapeRate
  let mangoCost := mangoKg * mangoRate
  let appleCost := appleKg * appleRate
  let orangeCost := orangeKg * orangeRate
  let grapeMangoeTotal := grapeCost + mangoCost
  let appleOrangeTotal := appleCost + orangeCost
  let grapeMangoeDiscount := grapeMangoeTotal * grapeMangoeDiscountRate
  let discountedGrapeMangoe := grapeMangoeTotal - grapeMangoeDiscount
  let discountedAppleOrange := appleOrangeTotal - appleOrangeFixedDiscount
  let totalDiscountedCost := discountedGrapeMangoe + discountedAppleOrange
  let salesTax := totalDiscountedCost * salesTaxRate
  let totalTax := salesTax + fixedTax
  let totalAmount := totalDiscountedCost + totalTax
  totalAmount * exchangeRate

/-- Theorem stating that the fruit purchase cost under given conditions is $323.79 -/
theorem fruit_purchase_cost_is_correct :
  fruitPurchaseCost 7 68 9 48 5 55 4 38 0.1 25 0.05 15 0.25 = 323.79 := by
  sorry


end NUMINAMATH_CALUDE_fruit_purchase_cost_is_correct_l2228_222808


namespace NUMINAMATH_CALUDE_min_sum_of_reciprocal_squares_l2228_222850

theorem min_sum_of_reciprocal_squares (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a + b + c = 1) : 
  27 ≤ (1/a^2 + 1/b^2 + 1/c^2) := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_reciprocal_squares_l2228_222850
