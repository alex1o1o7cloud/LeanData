import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_root_value_l1012_101231

theorem quadratic_root_value (p q : ℝ) : 
  3 * p^2 - 5 * p - 8 = 0 →
  3 * q^2 - 5 * q - 8 = 0 →
  p ≠ q →
  (9 * p^4 - 9 * q^4) / (p - q) = 365 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l1012_101231


namespace NUMINAMATH_CALUDE_election_votes_theorem_l1012_101271

theorem election_votes_theorem (total_votes : ℕ) : 
  (∃ (winner_votes loser_votes : ℕ),
    winner_votes + loser_votes = total_votes ∧
    winner_votes = (70 * total_votes) / 100 ∧
    winner_votes - loser_votes = 320) →
  total_votes = 800 := by
sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l1012_101271


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1012_101214

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The arithmetic sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The problem statement -/
theorem arithmetic_sequence_problem 
  (seq : ArithmeticSequence) 
  (h1 : seq.a 6 = seq.S 3) 
  (h2 : seq.a 6 = 12) : 
  seq.a 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1012_101214


namespace NUMINAMATH_CALUDE_chord_intersection_ratio_l1012_101223

-- Define a circle
variable (circle : Set ℝ × ℝ)

-- Define points E, F, G, H, Q
variable (E F G H Q : ℝ × ℝ)

-- Define that EF and GH are chords of the circle
variable (chord_EF : Set (ℝ × ℝ))
variable (chord_GH : Set (ℝ × ℝ))

-- Define that Q is the intersection point of EF and GH
variable (intersect_Q : Q ∈ chord_EF ∩ chord_GH)

-- Define lengths
def EQ : ℝ := sorry
def FQ : ℝ := sorry
def GQ : ℝ := sorry
def HQ : ℝ := sorry

-- State the theorem
theorem chord_intersection_ratio 
  (h1 : EQ = 4) 
  (h2 : GQ = 10) : 
  FQ / HQ = 5 / 2 := by sorry

end NUMINAMATH_CALUDE_chord_intersection_ratio_l1012_101223


namespace NUMINAMATH_CALUDE_expected_value_3X_plus_2_l1012_101259

/-- Probability distribution for random variable X -/
def prob_dist : List (ℝ × ℝ) :=
  [(1, 0.1), (2, 0.3), (3, 0.4), (4, 0.1), (5, 0.1)]

/-- Expected value of X -/
def E (X : List (ℝ × ℝ)) : ℝ :=
  (X.map (fun (x, p) => x * p)).sum

/-- Theorem: Expected value of 3X+2 is 10.4 -/
theorem expected_value_3X_plus_2 :
  E (prob_dist.map (fun (x, p) => (3 * x + 2, p))) = 10.4 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_3X_plus_2_l1012_101259


namespace NUMINAMATH_CALUDE_optimal_advertising_plan_l1012_101221

/-- Represents the advertising plan for a company --/
structure AdvertisingPlan where
  timeA : ℝ  -- Time allocated to TV station A in minutes
  timeB : ℝ  -- Time allocated to TV station B in minutes

/-- Calculates the total advertising time for a given plan --/
def totalTime (plan : AdvertisingPlan) : ℝ :=
  plan.timeA + plan.timeB

/-- Calculates the total advertising cost for a given plan --/
def totalCost (plan : AdvertisingPlan) : ℝ :=
  500 * plan.timeA + 200 * plan.timeB

/-- Calculates the total revenue for a given plan --/
def totalRevenue (plan : AdvertisingPlan) : ℝ :=
  0.3 * plan.timeA + 0.2 * plan.timeB

/-- Theorem stating the optimal advertising plan and maximum revenue --/
theorem optimal_advertising_plan :
  ∃ (plan : AdvertisingPlan),
    totalTime plan ≤ 300 ∧
    totalCost plan ≤ 90000 ∧
    plan.timeA = 100 ∧
    plan.timeB = 200 ∧
    totalRevenue plan = 70 ∧
    ∀ (other : AdvertisingPlan),
      totalTime other ≤ 300 →
      totalCost other ≤ 90000 →
      totalRevenue other ≤ totalRevenue plan :=
by
  sorry


end NUMINAMATH_CALUDE_optimal_advertising_plan_l1012_101221


namespace NUMINAMATH_CALUDE_bothMiss_mutually_exclusive_with_hitAtLeastOnce_bothMiss_complement_of_hitAtLeastOnce_l1012_101246

-- Define the sample space for two shots
inductive ShotOutcome
| Hit
| Miss

-- Define the type for a two-shot experiment
def TwoShots := (ShotOutcome × ShotOutcome)

-- Define the event "hitting the target at least once"
def hitAtLeastOnce (outcome : TwoShots) : Prop :=
  outcome.1 = ShotOutcome.Hit ∨ outcome.2 = ShotOutcome.Hit

-- Define the event "both shots miss"
def bothMiss (outcome : TwoShots) : Prop :=
  outcome.1 = ShotOutcome.Miss ∧ outcome.2 = ShotOutcome.Miss

-- Theorem stating that "both shots miss" is mutually exclusive with "hitting at least once"
theorem bothMiss_mutually_exclusive_with_hitAtLeastOnce :
  ∀ (outcome : TwoShots), ¬(hitAtLeastOnce outcome ∧ bothMiss outcome) :=
sorry

-- Theorem stating that "both shots miss" is the complement of "hitting at least once"
theorem bothMiss_complement_of_hitAtLeastOnce :
  ∀ (outcome : TwoShots), hitAtLeastOnce outcome ↔ ¬(bothMiss outcome) :=
sorry

end NUMINAMATH_CALUDE_bothMiss_mutually_exclusive_with_hitAtLeastOnce_bothMiss_complement_of_hitAtLeastOnce_l1012_101246


namespace NUMINAMATH_CALUDE_M_subset_N_l1012_101298

-- Define set M
def M : Set ℝ := {x | ∃ k : ℤ, x = k / 2 + 1 / 4}

-- Define set N
def N : Set ℝ := {x | ∃ k : ℤ, x = k / 4 + 1 / 2}

-- Theorem statement
theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l1012_101298


namespace NUMINAMATH_CALUDE_golden_section_steel_l1012_101235

theorem golden_section_steel (m : ℝ) : 
  (1000 + (m - 1000) * 0.618 = 1618 ∨ m - (m - 1000) * 0.618 = 1618) → 
  (m = 2000 ∨ m = 2618) := by
sorry

end NUMINAMATH_CALUDE_golden_section_steel_l1012_101235


namespace NUMINAMATH_CALUDE_cartesian_polar_equivalence_l1012_101234

-- Define the set of points in Cartesian coordinates
def cartesian_set : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 9}

-- Define the set of points in polar coordinates
def polar_set : Set (ℝ × ℝ) := {p | Real.sqrt (p.1^2 + p.2^2) = 3}

-- Theorem stating the equivalence of the two sets
theorem cartesian_polar_equivalence : cartesian_set = polar_set := by sorry

end NUMINAMATH_CALUDE_cartesian_polar_equivalence_l1012_101234


namespace NUMINAMATH_CALUDE_cuboid_height_from_cube_l1012_101233

/-- The length of wire needed to make a cube with given edge length -/
def cube_wire_length (edge : ℝ) : ℝ := 12 * edge

/-- The length of wire needed to make a cuboid with given dimensions -/
def cuboid_wire_length (length width height : ℝ) : ℝ :=
  4 * (length + width + height)

theorem cuboid_height_from_cube (cube_edge length width : ℝ) 
  (h_cube_edge : cube_edge = 10)
  (h_length : length = 8)
  (h_width : width = 5) :
  ∃ (height : ℝ), 
    cube_wire_length cube_edge = cuboid_wire_length length width height ∧ 
    height = 17 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_height_from_cube_l1012_101233


namespace NUMINAMATH_CALUDE_equation_solution_l1012_101277

theorem equation_solution : ∃ x : ℤ, 45 - (x - (37 - (15 - 17))) = 56 ∧ x = 28 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1012_101277


namespace NUMINAMATH_CALUDE_small_paintings_completed_l1012_101265

/-- Represents the number of ounces of paint used for a large canvas --/
def paint_per_large_canvas : ℕ := 3

/-- Represents the number of ounces of paint used for a small canvas --/
def paint_per_small_canvas : ℕ := 2

/-- Represents the number of large paintings completed --/
def large_paintings_completed : ℕ := 3

/-- Represents the total amount of paint used in ounces --/
def total_paint_used : ℕ := 17

/-- Proves that the number of small paintings completed is 4 --/
theorem small_paintings_completed :
  (total_paint_used - large_paintings_completed * paint_per_large_canvas) / paint_per_small_canvas = 4 :=
by sorry

end NUMINAMATH_CALUDE_small_paintings_completed_l1012_101265


namespace NUMINAMATH_CALUDE_smallest_angle_when_largest_is_120_l1012_101285

/-- Represents a trapezoid with angles in arithmetic sequence -/
structure ArithmeticTrapezoid where
  /-- The smallest angle of the trapezoid -/
  smallest_angle : ℝ
  /-- The common difference between consecutive angles -/
  angle_difference : ℝ
  /-- The sum of all angles in the trapezoid is 360° -/
  angle_sum : smallest_angle + (smallest_angle + angle_difference) + 
              (smallest_angle + 2 * angle_difference) + 
              (smallest_angle + 3 * angle_difference) = 360

theorem smallest_angle_when_largest_is_120 (t : ArithmeticTrapezoid) 
  (h : t.smallest_angle + 3 * t.angle_difference = 120) : 
  t.smallest_angle = 60 := by
  sorry

#check smallest_angle_when_largest_is_120

end NUMINAMATH_CALUDE_smallest_angle_when_largest_is_120_l1012_101285


namespace NUMINAMATH_CALUDE_circle_properties_l1012_101252

/-- A circle passing through two points with its center on a line -/
def circle_equation (x y : ℝ) : Prop :=
  (x - 1/2)^2 + (y - 1)^2 = 5/4

theorem circle_properties :
  (circle_equation 1 0) ∧
  (circle_equation 0 2) ∧
  (∃ (a : ℝ), circle_equation a (2*a)) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l1012_101252


namespace NUMINAMATH_CALUDE_intersection_condition_coincidence_condition_l1012_101217

/-- Two lines in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ∀ x y : ℝ, a * x + b * y + c = 0

/-- Definition of line l₁ -/
def l₁ (m : ℝ) : Line where
  a := 1
  b := m
  c := 6
  eq := by sorry

/-- Definition of line l₂ -/
def l₂ (m : ℝ) : Line where
  a := m - 2
  b := 3
  c := 2 * m
  eq := by sorry

/-- Two lines intersect if they are not parallel -/
def intersect (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b ≠ l₁.b * l₂.a

/-- Two lines coincide if they are equivalent -/
def coincide (l₁ l₂ : Line) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ l₁.a = k * l₂.a ∧ l₁.b = k * l₂.b ∧ l₁.c = k * l₂.c

/-- Main theorem for intersection -/
theorem intersection_condition (m : ℝ) :
  intersect (l₁ m) (l₂ m) ↔ m ≠ -1 ∧ m ≠ 3 := by sorry

/-- Main theorem for coincidence -/
theorem coincidence_condition (m : ℝ) :
  coincide (l₁ m) (l₂ m) ↔ m = 3 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_coincidence_condition_l1012_101217


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1012_101249

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1012_101249


namespace NUMINAMATH_CALUDE_toms_average_strokes_l1012_101203

/-- Represents the number of rounds Tom plays -/
def rounds : ℕ := 9

/-- Represents the par value per hole -/
def par_per_hole : ℕ := 3

/-- Represents the number of strokes Tom was over par -/
def strokes_over_par : ℕ := 9

/-- Calculates Tom's average number of strokes per hole -/
def average_strokes_per_hole : ℚ :=
  (rounds * par_per_hole + strokes_over_par) / rounds

/-- Theorem stating that Tom's average number of strokes per hole is 4 -/
theorem toms_average_strokes :
  average_strokes_per_hole = 4 := by sorry

end NUMINAMATH_CALUDE_toms_average_strokes_l1012_101203


namespace NUMINAMATH_CALUDE_isosceles_triangle_l1012_101212

theorem isosceles_triangle (A B C : ℝ) (a b c : ℝ) : 
  b * Real.cos C = c * Real.cos B → B = C := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l1012_101212


namespace NUMINAMATH_CALUDE_plant_growth_thirty_percent_plant_growth_day_l1012_101211

/-- Represents the growth of a plant over time. -/
structure PlantGrowth where
  initialLength : ℝ
  dailyGrowth : ℝ

/-- Calculates the length of the plant on a given day. -/
def plantLengthOnDay (p : PlantGrowth) (day : ℕ) : ℝ :=
  p.initialLength + p.dailyGrowth * (day - 1 : ℝ)

/-- Theorem: The plant grows 30% between the 4th day and the 10th day. -/
theorem plant_growth_thirty_percent (p : PlantGrowth) 
  (h1 : p.initialLength = 11)
  (h2 : p.dailyGrowth = 0.6875) :
  plantLengthOnDay p 10 = plantLengthOnDay p 4 * 1.3 := by
  sorry

/-- Corollary: The 10th day is the first day when the plant's length is at least 30% greater than on the 4th day. -/
theorem plant_growth_day (p : PlantGrowth)
  (h1 : p.initialLength = 11)
  (h2 : p.dailyGrowth = 0.6875) :
  (∀ d : ℕ, d < 10 → plantLengthOnDay p d < plantLengthOnDay p 4 * 1.3) ∧
  plantLengthOnDay p 10 ≥ plantLengthOnDay p 4 * 1.3 := by
  sorry

end NUMINAMATH_CALUDE_plant_growth_thirty_percent_plant_growth_day_l1012_101211


namespace NUMINAMATH_CALUDE_inequality_cube_l1012_101270

theorem inequality_cube (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  (a - c)^3 > (b - c)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_cube_l1012_101270


namespace NUMINAMATH_CALUDE_vasyas_numbers_l1012_101251

theorem vasyas_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = (1 : ℝ) / 2 ∧ y = -(1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_vasyas_numbers_l1012_101251


namespace NUMINAMATH_CALUDE_ideal_complex_condition_l1012_101276

def is_ideal_complex (z : ℂ) : Prop :=
  z.re = -z.im

theorem ideal_complex_condition (a b : ℝ) :
  let z : ℂ := (a / (1 - 2*I)) + b*I
  is_ideal_complex z → 3*a + 5*b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ideal_complex_condition_l1012_101276


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1012_101210

-- Problem 1
theorem problem_1 : (-7) - (-8) + (-9) - 14 = -22 := by sorry

-- Problem 2
theorem problem_2 : (-4) * (-3)^2 - 14 / (-7) = -34 := by sorry

-- Problem 3
theorem problem_3 : (3/10 - 1/4 + 4/5) * (-20) = -17 := by sorry

-- Problem 4
theorem problem_4 : (-2)^2 / |1-3| + 3 * (1/2 - 1) = 1/2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1012_101210


namespace NUMINAMATH_CALUDE_remainder_zero_l1012_101262

theorem remainder_zero (k α : ℕ+) (h : 10 * k.val - α.val > 0) :
  (8^(10 * k.val + α.val) + 6^(10 * k.val - α.val) - 
   7^(10 * k.val - α.val) - 2^(10 * k.val + α.val)) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_zero_l1012_101262


namespace NUMINAMATH_CALUDE_other_number_proof_l1012_101260

theorem other_number_proof (a b : ℕ+) (h1 : Nat.lcm a b = 2310) (h2 : Nat.gcd a b = 26) (h3 : a = 210) : b = 286 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l1012_101260


namespace NUMINAMATH_CALUDE_double_fraction_value_l1012_101218

theorem double_fraction_value (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2*x * 2*y) / (2*x + 2*y) = 2 * (x*y / (x + y)) :=
by sorry

end NUMINAMATH_CALUDE_double_fraction_value_l1012_101218


namespace NUMINAMATH_CALUDE_tangent_line_at_P_l1012_101201

-- Define the curve
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the point of tangency
def P : ℝ × ℝ := (2, 0)

-- Define the proposed tangent line
def tangentLine (x : ℝ) : ℝ := 2*x - 4

theorem tangent_line_at_P :
  (∀ x : ℝ, HasDerivAt f (tangentLine P.1) P.1) ∧
  f P.1 = tangentLine P.1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_P_l1012_101201


namespace NUMINAMATH_CALUDE_solve_system_l1012_101215

theorem solve_system (x y : ℝ) 
  (eq1 : 3 * x - 2 * y = 18) 
  (eq2 : x + y = 20) : 
  y = 8.4 := by sorry

end NUMINAMATH_CALUDE_solve_system_l1012_101215


namespace NUMINAMATH_CALUDE_five_dice_not_same_probability_l1012_101204

theorem five_dice_not_same_probability :
  let n_faces : ℕ := 6
  let n_dice : ℕ := 5
  let total_outcomes : ℕ := n_faces ^ n_dice
  let same_number_outcomes : ℕ := n_faces
  (1 : ℚ) - (same_number_outcomes : ℚ) / total_outcomes = 1295 / 1296 :=
by sorry

end NUMINAMATH_CALUDE_five_dice_not_same_probability_l1012_101204


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1012_101281

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 2 + a 3 + a 10 + a 11 = 48 → a 6 + a 7 = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1012_101281


namespace NUMINAMATH_CALUDE_binary_10001000_to_octal_l1012_101225

def binary_to_octal (b : ℕ) : ℕ := sorry

theorem binary_10001000_to_octal :
  binary_to_octal 0b10001000 = 0o210 := by sorry

end NUMINAMATH_CALUDE_binary_10001000_to_octal_l1012_101225


namespace NUMINAMATH_CALUDE_expression_evaluation_l1012_101237

theorem expression_evaluation :
  let a : ℚ := 2
  let b : ℚ := 1/2
  2 * (a^2 - 2*a*b) - 3 * (a^2 - a*b - 4*b^2) = -2 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1012_101237


namespace NUMINAMATH_CALUDE_total_CDs_is_448_l1012_101200

/-- The number of shelves in Store A -/
def store_A_shelves : ℕ := 5

/-- The number of CD racks per shelf in Store A -/
def store_A_racks_per_shelf : ℕ := 7

/-- The number of CDs per rack in Store A -/
def store_A_CDs_per_rack : ℕ := 8

/-- The number of shelves in Store B -/
def store_B_shelves : ℕ := 4

/-- The number of CD racks per shelf in Store B -/
def store_B_racks_per_shelf : ℕ := 6

/-- The number of CDs per rack in Store B -/
def store_B_CDs_per_rack : ℕ := 7

/-- The total number of CDs that can be held in Store A and Store B together -/
def total_CDs : ℕ := 
  (store_A_shelves * store_A_racks_per_shelf * store_A_CDs_per_rack) +
  (store_B_shelves * store_B_racks_per_shelf * store_B_CDs_per_rack)

/-- Theorem stating that the total number of CDs that can be held in Store A and Store B together is 448 -/
theorem total_CDs_is_448 : total_CDs = 448 := by
  sorry

end NUMINAMATH_CALUDE_total_CDs_is_448_l1012_101200


namespace NUMINAMATH_CALUDE_diamond_with_zero_not_always_double_l1012_101247

def diamond (x y : ℝ) : ℝ := x + y - |x - y|

theorem diamond_with_zero_not_always_double :
  ¬ (∀ x : ℝ, diamond x 0 = 2 * x) := by
  sorry

end NUMINAMATH_CALUDE_diamond_with_zero_not_always_double_l1012_101247


namespace NUMINAMATH_CALUDE_xy_unique_values_l1012_101297

def X : Finset ℤ := {2, 3, 7}
def Y : Finset ℤ := {-31, -24, 4}

theorem xy_unique_values : 
  Finset.card (Finset.image (λ (p : ℤ × ℤ) => p.1 * p.2) (X.product Y)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_xy_unique_values_l1012_101297


namespace NUMINAMATH_CALUDE_table_pattern_l1012_101293

/-- Represents the number at position (row, column) in the table -/
def tableEntry (row : ℕ) (column : ℕ) : ℕ := sorry

/-- The first number of each row is equal to the row number -/
axiom first_number (n : ℕ) : tableEntry (n + 1) 1 = n + 1

/-- Each row forms an arithmetic sequence with common difference 1 -/
axiom arithmetic_sequence (n m : ℕ) : 
  tableEntry (n + 1) (m + 1) = tableEntry (n + 1) m + 1

/-- The number at the intersection of the (n+1)th row and the mth column is m + n -/
theorem table_pattern (n m : ℕ) : tableEntry (n + 1) m = m + n := by
  sorry

end NUMINAMATH_CALUDE_table_pattern_l1012_101293


namespace NUMINAMATH_CALUDE_game_result_l1012_101299

def f (n : ℕ) : ℕ :=
  if n % 2 = 0 ∧ n % 3 = 0 then 7
  else if n % 2 = 0 then 3
  else if Nat.Prime n then 5
  else 0

def allie_rolls : List ℕ := [2, 3, 4, 5, 6]
def betty_rolls : List ℕ := [6, 3, 4, 2, 1]

theorem game_result :
  (List.sum (List.map f allie_rolls)) * (List.sum (List.map f betty_rolls)) = 500 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l1012_101299


namespace NUMINAMATH_CALUDE_circle_area_with_radius_3_l1012_101295

theorem circle_area_with_radius_3 :
  ∀ (π : ℝ), π > 0 →
  let r : ℝ := 3
  let area := π * r^2
  area = 9 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_radius_3_l1012_101295


namespace NUMINAMATH_CALUDE_sms_is_fraudulent_l1012_101230

/-- Represents an SMS message -/
structure SMS where
  claims_prize : Bool
  requests_payment : Bool
  recipient_participated : Bool

/-- Represents characteristics of a legitimate contest -/
structure LegitimateContest where
  requires_payment : Bool

/-- Determines if an SMS is fraudulent based on given conditions -/
def is_fraudulent (sms : SMS) (contest : LegitimateContest) : Prop :=
  sms.claims_prize ∧ 
  sms.requests_payment ∧ 
  ¬sms.recipient_participated ∧
  ¬contest.requires_payment

/-- Theorem stating that an SMS with specific characteristics is fraudulent -/
theorem sms_is_fraudulent (sms : SMS) (contest : LegitimateContest) :
  sms.claims_prize = true →
  sms.requests_payment = true →
  sms.recipient_participated = false →
  contest.requires_payment = false →
  is_fraudulent sms contest := by
  sorry

#check sms_is_fraudulent

end NUMINAMATH_CALUDE_sms_is_fraudulent_l1012_101230


namespace NUMINAMATH_CALUDE_recipe_change_l1012_101282

/-- Represents the recipe for the apple-grape drink -/
structure Recipe where
  apple_proportion : ℚ  -- Proportion of an apple juice container used per can
  grape_proportion : ℚ  -- Proportion of a grape juice container used per can

/-- The total volume of juice per can -/
def total_volume (r : Recipe) : ℚ :=
  r.apple_proportion + r.grape_proportion

theorem recipe_change (old_recipe new_recipe : Recipe) :
  old_recipe.apple_proportion = 1/6 →
  old_recipe.grape_proportion = 1/10 →
  new_recipe.apple_proportion = 1/5 →
  total_volume old_recipe = total_volume new_recipe →
  new_recipe.grape_proportion = 1/15 := by
  sorry

end NUMINAMATH_CALUDE_recipe_change_l1012_101282


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1012_101220

/-- A regular polygon is a polygon with all sides of equal length. -/
structure RegularPolygon where
  side_length : ℝ
  num_sides : ℕ
  perimeter : ℝ
  perimeter_eq : perimeter = side_length * num_sides

/-- Theorem: A regular polygon with side length 16 cm and perimeter 80 cm has 5 sides. -/
theorem regular_polygon_sides (p : RegularPolygon) 
  (h1 : p.side_length = 16) 
  (h2 : p.perimeter = 80) : 
  p.num_sides = 5 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1012_101220


namespace NUMINAMATH_CALUDE_max_value_theorem_l1012_101291

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
  (h2 : a + b + c = 3) (h3 : a = 1) :
  (∀ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z → x + y + z = 3 → x = 1 →
    (a*b)/(a+b) + (a*c)/(a+c) + (b*c)/(b+c) ≥ (x*y)/(x+y) + (x*z)/(x+z) + (y*z)/(y+z)) ∧
  (∃ b' c' : ℝ, 0 ≤ b' ∧ 0 ≤ c' ∧ a + b' + c' = 3 ∧
    (a*b')/(a+b') + (a*c')/(a+c') + (b'*c')/(b'+c') = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1012_101291


namespace NUMINAMATH_CALUDE_arithmetic_equalities_l1012_101286

theorem arithmetic_equalities : 
  (Real.sqrt 27 + 3 * Real.sqrt (1/3) - Real.sqrt 24 * Real.sqrt 2 = 0) ∧
  ((Real.sqrt 5 - 2) * (2 + Real.sqrt 5) - (Real.sqrt 3 - 1)^2 = -3 + 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equalities_l1012_101286


namespace NUMINAMATH_CALUDE_sum_f_91_and_neg_91_l1012_101268

/-- A polynomial function of degree 6 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^6 + b * x^4 - c * x^2 + 3

/-- Theorem: Given f(x) = ax^6 + bx^4 - cx^2 + 3 and f(91) = 1, prove f(91) + f(-91) = 2 -/
theorem sum_f_91_and_neg_91 (a b c : ℝ) (h : f a b c 91 = 1) : f a b c 91 + f a b c (-91) = 2 := by
  sorry

#check sum_f_91_and_neg_91

end NUMINAMATH_CALUDE_sum_f_91_and_neg_91_l1012_101268


namespace NUMINAMATH_CALUDE_tenths_vs_thousandths_l1012_101222

def number : ℚ := 85247.2048

theorem tenths_vs_thousandths :
  (number - number.floor) * 10 % 1 * 10 = 
  100 * ((number - number.floor) * 1000 % 10) / 10 := by
  sorry

end NUMINAMATH_CALUDE_tenths_vs_thousandths_l1012_101222


namespace NUMINAMATH_CALUDE_shortest_altitude_right_triangle_l1012_101275

theorem shortest_altitude_right_triangle :
  ∀ (a b c h : ℝ),
  a = 8 ∧ b = 15 ∧ c = 17 →
  a^2 + b^2 = c^2 →
  h = (2 * (a * b) / 2) / c →
  h = 120 / 17 ∧ 
  (∀ h' : ℝ, (h' = a ∨ h' = b ∨ h' = h) → h ≤ h') :=
by sorry

end NUMINAMATH_CALUDE_shortest_altitude_right_triangle_l1012_101275


namespace NUMINAMATH_CALUDE_cabbage_price_calculation_l1012_101267

/-- Represents the price of the cabbage in Janet's grocery purchase. -/
def cabbage_price : ℝ := sorry

/-- Represents Janet's total grocery budget. -/
def total_budget : ℝ := sorry

theorem cabbage_price_calculation :
  let broccoli_cost : ℝ := 3 * 4
  let oranges_cost : ℝ := 3 * 0.75
  let bacon_cost : ℝ := 1 * 3
  let chicken_cost : ℝ := 2 * 3
  let meat_cost : ℝ := bacon_cost + chicken_cost
  let known_items_cost : ℝ := broccoli_cost + oranges_cost + bacon_cost + chicken_cost
  meat_cost = 0.33 * total_budget ∧
  cabbage_price = total_budget - known_items_cost →
  cabbage_price = 4.02 := by sorry

end NUMINAMATH_CALUDE_cabbage_price_calculation_l1012_101267


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l1012_101206

theorem fourth_root_equation_solutions :
  ∀ x : ℝ, (x^(1/4) = 12 / (7 - x^(1/4))) ↔ (x = 81 ∨ x = 256) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l1012_101206


namespace NUMINAMATH_CALUDE_function_properties_l1012_101226

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 + x^2 + b * x

-- Define the function g
def g (a b x : ℝ) : ℝ := f a b x + (3 * a * x^2 + 2 * x + b)

-- Main theorem
theorem function_properties (a b : ℝ) :
  (∀ x, g a b x = -g a b (-x)) →
  (∃ f_simplified : ℝ → ℝ, 
    (∀ x, f a b x = f_simplified x) ∧
    (∀ x, f_simplified x = x^2 - x) ∧
    (∀ x ∈ Set.Icc 1 2, HasDerivAt (g a b) ((2 : ℝ) * x + 1) x) ∧
    (g a b 1 = 1) ∧
    (g a b 2 = 5) ∧
    (∀ x ∈ Set.Icc 1 2, g a b x ≥ 1 ∧ g a b x ≤ 5)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1012_101226


namespace NUMINAMATH_CALUDE_rainfall_difference_l1012_101278

/-- Calculates the difference between the average rainfall and the actual rainfall for the first three days of May. -/
theorem rainfall_difference (day1 day2 day3 avg : ℝ) 
  (h1 : day1 = 26)
  (h2 : day2 = 34)
  (h3 : day3 = day2 - 12)
  (h4 : avg = 140) :
  avg - (day1 + day2 + day3) = 58 := by
sorry

end NUMINAMATH_CALUDE_rainfall_difference_l1012_101278


namespace NUMINAMATH_CALUDE_wall_width_l1012_101216

/-- Theorem: Width of a wall with specific proportions and volume --/
theorem wall_width (w h l : ℝ) (volume : ℝ) : 
  h = 6 * w →
  l = 7 * h →
  volume = w * h * l →
  volume = 129024 →
  w = 8 := by
  sorry

end NUMINAMATH_CALUDE_wall_width_l1012_101216


namespace NUMINAMATH_CALUDE_extreme_value_cubic_l1012_101228

/-- Given a cubic function f(x) = ax³ + 3x² + 3x + 3,
    if f has an extreme value at x = 1, then a = -3 -/
theorem extreme_value_cubic (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 + 3 * x^2 + 3 * x + 3
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  a = -3 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_cubic_l1012_101228


namespace NUMINAMATH_CALUDE_shuttlecock_mass_probability_l1012_101202

variable (ξ : ℝ)

-- Define the probabilities given in the problem
def P_less_than_4_8 : ℝ := 0.3
def P_not_less_than_4_85 : ℝ := 0.32

-- Define the probability we want to prove
def P_between_4_8_and_4_85 : ℝ := 1 - P_less_than_4_8 - P_not_less_than_4_85

-- Theorem statement
theorem shuttlecock_mass_probability :
  P_between_4_8_and_4_85 = 0.38 := by sorry

end NUMINAMATH_CALUDE_shuttlecock_mass_probability_l1012_101202


namespace NUMINAMATH_CALUDE_grains_in_cup_is_480_l1012_101239

/-- Represents the number of grains of rice in one cup -/
def grains_in_cup : ℕ :=
  let half_cup_tablespoons : ℕ := 8
  let teaspoons_per_tablespoon : ℕ := 3
  let grains_per_teaspoon : ℕ := 10
  2 * (half_cup_tablespoons * teaspoons_per_tablespoon * grains_per_teaspoon)

/-- Theorem stating that there are 480 grains of rice in one cup -/
theorem grains_in_cup_is_480 : grains_in_cup = 480 := by
  sorry

end NUMINAMATH_CALUDE_grains_in_cup_is_480_l1012_101239


namespace NUMINAMATH_CALUDE_pencils_per_group_l1012_101257

theorem pencils_per_group (total_pencils : ℕ) (num_groups : ℕ) 
  (h1 : total_pencils = 154) (h2 : num_groups = 14) :
  total_pencils / num_groups = 11 := by
sorry

end NUMINAMATH_CALUDE_pencils_per_group_l1012_101257


namespace NUMINAMATH_CALUDE_push_up_difference_l1012_101296

theorem push_up_difference (zachary_pushups david_pushups : ℕ) 
  (h1 : zachary_pushups = 19)
  (h2 : david_pushups = 58) :
  david_pushups - zachary_pushups = 39 := by
  sorry

end NUMINAMATH_CALUDE_push_up_difference_l1012_101296


namespace NUMINAMATH_CALUDE_equal_cell_squares_count_l1012_101284

/-- Represents a cell in the grid -/
inductive Cell
| White
| Black

/-- Represents the 5x5 grid configuration -/
def Grid : Type := Fin 5 → Fin 5 → Cell

/-- The specific grid configuration given in the problem -/
def problem_grid : Grid := sorry

/-- A square in the grid -/
structure Square where
  top_left : Fin 5 × Fin 5
  size : Nat

/-- Checks if a square has equal number of black and white cells -/
def has_equal_cells (g : Grid) (s : Square) : Bool := sorry

/-- Counts the number of squares with equal black and white cells -/
def count_equal_squares (g : Grid) : Nat := sorry

/-- The main theorem -/
theorem equal_cell_squares_count :
  count_equal_squares problem_grid = 16 := by sorry

end NUMINAMATH_CALUDE_equal_cell_squares_count_l1012_101284


namespace NUMINAMATH_CALUDE_min_four_dollar_frisbees_min_four_dollar_frisbees_proof_l1012_101283

/-- Given 60 frisbees sold at either $3 or $4 each, with total receipts of $204,
    the minimum number of $4 frisbees sold is 24. -/
theorem min_four_dollar_frisbees : ℕ :=
  let total_frisbees : ℕ := 60
  let total_receipts : ℕ := 204
  let price_low : ℕ := 3
  let price_high : ℕ := 4
  24

/-- Proof that the minimum number of $4 frisbees sold is indeed 24. -/
theorem min_four_dollar_frisbees_proof :
  let total_frisbees : ℕ := 60
  let total_receipts : ℕ := 204
  let price_low : ℕ := 3
  let price_high : ℕ := 4
  let min_high_price_frisbees := min_four_dollar_frisbees
  (∃ (low_price_frisbees : ℕ),
    low_price_frisbees + min_high_price_frisbees = total_frisbees ∧
    low_price_frisbees * price_low + min_high_price_frisbees * price_high = total_receipts) ∧
  (∀ (high_price_frisbees : ℕ),
    high_price_frisbees < min_high_price_frisbees →
    ¬∃ (low_price_frisbees : ℕ),
      low_price_frisbees + high_price_frisbees = total_frisbees ∧
      low_price_frisbees * price_low + high_price_frisbees * price_high = total_receipts) :=
by
  sorry

#check min_four_dollar_frisbees
#check min_four_dollar_frisbees_proof

end NUMINAMATH_CALUDE_min_four_dollar_frisbees_min_four_dollar_frisbees_proof_l1012_101283


namespace NUMINAMATH_CALUDE_count_special_numbers_eq_266_l1012_101273

/-- The number of natural numbers from 1 to 1992 that are multiples of 3, but not multiples of 2 or 5 -/
def count_special_numbers : ℕ := 
  (Nat.floor (1992 / 3) : ℕ) - 
  (Nat.floor (1992 / 6) : ℕ) - 
  (Nat.floor (1992 / 15) : ℕ) + 
  (Nat.floor (1992 / 30) : ℕ)

theorem count_special_numbers_eq_266 : count_special_numbers = 266 := by
  sorry

end NUMINAMATH_CALUDE_count_special_numbers_eq_266_l1012_101273


namespace NUMINAMATH_CALUDE_sunzi_car_problem_l1012_101242

theorem sunzi_car_problem (x : ℕ) : 
  (x / 4 : ℚ) + 1 = (x - 9 : ℚ) / 3 ↔ 
  (∃ (cars : ℕ), 
    (x / 4 + 1 = cars) ∧ 
    ((x - 9) / 3 = cars - 1)) :=
by sorry

end NUMINAMATH_CALUDE_sunzi_car_problem_l1012_101242


namespace NUMINAMATH_CALUDE_binary_multiplication_example_l1012_101227

/-- Represents a binary number as a list of booleans, where true represents 1 and false represents 0 -/
def BinaryNumber := List Bool

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : BinaryNumber) : Nat :=
  b.foldl (fun acc digit => 2 * acc + if digit then 1 else 0) 0

/-- Multiplies two binary numbers -/
def binary_multiply (a b : BinaryNumber) : BinaryNumber :=
  sorry

theorem binary_multiplication_example :
  let a : BinaryNumber := [true, true, false, true]  -- 1101₂
  let b : BinaryNumber := [true, true, true]         -- 111₂
  let result : BinaryNumber := [true, false, false, true, true, true, true]  -- 1001111₂
  binary_multiply a b = result :=
sorry

end NUMINAMATH_CALUDE_binary_multiplication_example_l1012_101227


namespace NUMINAMATH_CALUDE_range_of_a_l1012_101207

theorem range_of_a (a : ℝ) :
  (((∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0) ∨ 
    (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0)) ∧
   ¬((∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0) ∧ 
     (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0))) →
  (a > 1 ∨ (-2 < a ∧ a < 1)) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1012_101207


namespace NUMINAMATH_CALUDE_coin_and_die_probability_l1012_101219

theorem coin_and_die_probability :
  let coin_outcomes : ℕ := 2  -- Fair coin has 2 possible outcomes
  let die_outcomes : ℕ := 8   -- Eight-sided die has 8 possible outcomes
  let total_outcomes : ℕ := coin_outcomes * die_outcomes
  let successful_outcomes : ℕ := 1  -- Only one successful outcome (Tails and 5)
  
  (successful_outcomes : ℚ) / total_outcomes = 1 / 16 :=
by
  sorry


end NUMINAMATH_CALUDE_coin_and_die_probability_l1012_101219


namespace NUMINAMATH_CALUDE_seven_digit_number_count_l1012_101224

def SevenDigitNumber := Fin 7 → Fin 7

def IsAscending (n : SevenDigitNumber) (start fin : Fin 7) : Prop :=
  ∀ i j, start ≤ i ∧ i < j ∧ j ≤ fin → n i < n j

def IsDescending (n : SevenDigitNumber) (start fin : Fin 7) : Prop :=
  ∀ i j, start ≤ i ∧ i < j ∧ j ≤ fin → n i > n j

def IsValidNumber (n : SevenDigitNumber) : Prop :=
  ∀ i j : Fin 7, i ≠ j → n i ≠ n j

theorem seven_digit_number_count :
  (∃ (S : Finset SevenDigitNumber),
    (∀ n ∈ S, IsValidNumber n ∧ IsAscending n 0 5 ∧ IsDescending n 5 6) ∧
    S.card = 6) ∧
  (∃ (T : Finset SevenDigitNumber),
    (∀ n ∈ T, IsValidNumber n ∧ IsAscending n 0 4 ∧ IsDescending n 4 6) ∧
    T.card = 15) := by sorry

end NUMINAMATH_CALUDE_seven_digit_number_count_l1012_101224


namespace NUMINAMATH_CALUDE_total_weight_calculation_l1012_101250

theorem total_weight_calculation (a b c d : ℝ) 
  (h1 : a + b = 156)
  (h2 : c + d = 195)
  (h3 : a + c = 174)
  (h4 : b + d = 186) :
  a + b + c + d = 355.5 := by
sorry

end NUMINAMATH_CALUDE_total_weight_calculation_l1012_101250


namespace NUMINAMATH_CALUDE_painted_face_probability_for_specific_prism_l1012_101258

/-- Represents a rectangular prism -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the total number of cubes in the prism -/
def total_cubes (p : RectangularPrism) : ℕ :=
  p.length * p.width * p.height

/-- Calculates the number of corner cubes -/
def corner_cubes : ℕ := 8

/-- Calculates the number of edge cubes -/
def edge_cubes (p : RectangularPrism) : ℕ :=
  4 * (p.length - 2) + 8 * (p.height - 2)

/-- Calculates the number of face cubes -/
def face_cubes (p : RectangularPrism) : ℕ :=
  2 * (p.length * p.height) - edge_cubes p - corner_cubes

/-- Calculates the probability of a randomly chosen cube showing a painted face when rolled -/
def painted_face_probability (p : RectangularPrism) : ℚ :=
  (3 * corner_cubes + 2 * edge_cubes p + face_cubes p) / (6 * total_cubes p)

theorem painted_face_probability_for_specific_prism :
  let p : RectangularPrism := ⟨20, 1, 7⟩
  painted_face_probability p = 9 / 35 := by
  sorry

end NUMINAMATH_CALUDE_painted_face_probability_for_specific_prism_l1012_101258


namespace NUMINAMATH_CALUDE_mirror_frame_areas_l1012_101266

/-- Represents the dimensions and properties of a rectangular mirror frame -/
structure MirrorFrame where
  outer_width : ℝ
  outer_length : ℝ
  frame_width : ℝ

/-- Calculates the area of the frame alone -/
def frame_area (frame : MirrorFrame) : ℝ :=
  frame.outer_width * frame.outer_length - (frame.outer_width - 2 * frame.frame_width) * (frame.outer_length - 2 * frame.frame_width)

/-- Calculates the area of the mirror inside the frame -/
def mirror_area (frame : MirrorFrame) : ℝ :=
  (frame.outer_width - 2 * frame.frame_width) * (frame.outer_length - 2 * frame.frame_width)

theorem mirror_frame_areas (frame : MirrorFrame) 
  (h1 : frame.outer_width = 100)
  (h2 : frame.outer_length = 120)
  (h3 : frame.frame_width = 15) :
  frame_area frame = 5700 ∧ mirror_area frame = 6300 := by
  sorry

end NUMINAMATH_CALUDE_mirror_frame_areas_l1012_101266


namespace NUMINAMATH_CALUDE_rectangle_squares_l1012_101287

theorem rectangle_squares (N : ℕ) : 
  (∃ x y : ℕ, N = x * (x + 9) ∧ N = y * (y + 6)) → N = 112 := by
sorry

end NUMINAMATH_CALUDE_rectangle_squares_l1012_101287


namespace NUMINAMATH_CALUDE_cube_volume_doubling_l1012_101244

theorem cube_volume_doubling (original_volume : ℝ) (new_volume : ℝ) : 
  original_volume = 216 →
  new_volume = (2 * original_volume^(1/3))^3 →
  new_volume = 1728 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_doubling_l1012_101244


namespace NUMINAMATH_CALUDE_sophie_gave_one_box_to_mom_l1012_101255

/-- Represents the number of donuts in a box --/
def donuts_per_box : ℕ := 12

/-- Represents the number of boxes Sophie bought --/
def boxes_bought : ℕ := 4

/-- Represents the number of donuts Sophie gave to her sister --/
def donuts_to_sister : ℕ := 6

/-- Represents the number of donuts Sophie had left for herself --/
def donuts_left_for_sophie : ℕ := 30

/-- Calculates the number of boxes Sophie gave to her mom --/
def boxes_to_mom : ℕ :=
  (boxes_bought * donuts_per_box - donuts_to_sister - donuts_left_for_sophie) / donuts_per_box

theorem sophie_gave_one_box_to_mom :
  boxes_to_mom = 1 :=
sorry

end NUMINAMATH_CALUDE_sophie_gave_one_box_to_mom_l1012_101255


namespace NUMINAMATH_CALUDE_max_t_is_one_l1012_101288

/-- The function f(x) = x^2 - ax + a - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + a - 1

/-- The theorem stating that the maximum value of t is 1 -/
theorem max_t_is_one (t : ℝ) :
  (∀ a ∈ Set.Ioo 0 4, ∃ x₀ ∈ Set.Icc 0 2, t ≤ |f a x₀|) →
  t ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_t_is_one_l1012_101288


namespace NUMINAMATH_CALUDE_largest_common_divisor_l1012_101274

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def product_function (n : ℕ) : ℕ := (n+2)*(n+4)*(n+6)*(n+8)*(n+10)

theorem largest_common_divisor (n : ℕ) (h : is_odd n) :
  (∀ m : ℕ, m > 8 → ¬(m ∣ product_function n)) ∧
  (8 ∣ product_function n) :=
sorry

end NUMINAMATH_CALUDE_largest_common_divisor_l1012_101274


namespace NUMINAMATH_CALUDE_expression_equality_l1012_101261

theorem expression_equality : 150 * (150 - 8) - (150 * 150 - 8) = -1192 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1012_101261


namespace NUMINAMATH_CALUDE_february_first_is_monday_l1012_101208

/-- Represents the days of the week -/
inductive Weekday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents the month of February in a specific year -/
structure February where
  /-- The day of the week that February 1 falls on -/
  first_day : Weekday
  /-- The number of days in February -/
  num_days : Nat
  /-- The number of Mondays in February -/
  num_mondays : Nat
  /-- The number of Thursdays in February -/
  num_thursdays : Nat

/-- Theorem stating that if February has exactly four Mondays and four Thursdays,
    then February 1 must fall on a Monday -/
theorem february_first_is_monday (feb : February) 
  (h1 : feb.num_mondays = 4)
  (h2 : feb.num_thursdays = 4) : 
  feb.first_day = Weekday.Monday := by
  sorry


end NUMINAMATH_CALUDE_february_first_is_monday_l1012_101208


namespace NUMINAMATH_CALUDE_sum_of_five_integers_l1012_101269

theorem sum_of_five_integers (C y M A : ℕ) : 
  C > 0 → y > 0 → M > 0 → A > 0 →
  C ≠ y → C ≠ M → C ≠ A → y ≠ M → y ≠ A → M ≠ A →
  C + y + M + M + A = 11 →
  M = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_five_integers_l1012_101269


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1012_101290

def arithmetic_sequence (a₁ d n : ℕ) := 
  (fun i => a₁ + (i - 1) * d)

theorem arithmetic_sequence_length : 
  ∃ n : ℕ, n > 0 ∧ arithmetic_sequence 15 4 n (n) = 95 ∧ n = 21 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1012_101290


namespace NUMINAMATH_CALUDE_valid_configuration_exists_l1012_101256

/-- Represents a point in a 2D grid --/
structure Point where
  x : Int
  y : Int

/-- Represents a line containing 4 points --/
structure Line where
  points : Fin 4 → Point

/-- The configuration of ships --/
structure ShipConfiguration where
  ships : Fin 10 → Point
  lines : Fin 5 → Line

/-- Checks if a line contains 4 distinct points from the given set of points --/
def Line.isValidLine (l : Line) (points : Fin 10 → Point) : Prop :=
  ∃ (indices : Fin 4 → Fin 10), (∀ i j, i ≠ j → indices i ≠ indices j) ∧
    (∀ i, l.points i = points (indices i))

/-- Checks if a configuration is valid --/
def ShipConfiguration.isValid (config : ShipConfiguration) : Prop :=
  ∀ l, config.lines l |>.isValidLine config.ships

/-- The theorem stating that a valid configuration exists --/
theorem valid_configuration_exists : ∃ (config : ShipConfiguration), config.isValid := by
  sorry


end NUMINAMATH_CALUDE_valid_configuration_exists_l1012_101256


namespace NUMINAMATH_CALUDE_fourth_root_equivalence_l1012_101289

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) : 
  (x^3 * (x^2)^(1/2))^(1/4) = x := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equivalence_l1012_101289


namespace NUMINAMATH_CALUDE_semi_circle_perimeter_l1012_101292

/-- The perimeter of a semi-circle with radius 6.4 cm is π * 6.4 + 12.8 -/
theorem semi_circle_perimeter :
  let r : ℝ := 6.4
  (2 * r * Real.pi / 2) + 2 * r = r * Real.pi + 2 * r := by
  sorry

end NUMINAMATH_CALUDE_semi_circle_perimeter_l1012_101292


namespace NUMINAMATH_CALUDE_train_length_l1012_101253

/-- Given a train that crosses a signal post in 40 seconds and takes 600 seconds
    to cross a 9000-meter long bridge at a constant speed, prove that the length
    of the train is 642.857142857... meters. -/
theorem train_length (signal_time : ℝ) (bridge_time : ℝ) (bridge_length : ℝ) :
  signal_time = 40 →
  bridge_time = 600 →
  bridge_length = 9000 →
  ∃ (train_length : ℝ), train_length = 360000 / 560 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1012_101253


namespace NUMINAMATH_CALUDE_sum_of_odds_is_even_product_zero_implies_factor_zero_exists_even_prime_l1012_101280

-- Definition of odd integer
def IsOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2*k + 1

-- Definition of even integer
def IsEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2*k

-- Definition of prime number
def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem sum_of_odds_is_even (x y : ℤ) (hx : IsOdd x) (hy : IsOdd y) : 
  IsEven (x + y) := by sorry

theorem product_zero_implies_factor_zero (x y : ℝ) :
  x * y = 0 → x = 0 ∨ y = 0 := by sorry

theorem exists_even_prime : 
  ∃ n : ℕ, IsPrime n ∧ ¬IsOdd n := by sorry

end NUMINAMATH_CALUDE_sum_of_odds_is_even_product_zero_implies_factor_zero_exists_even_prime_l1012_101280


namespace NUMINAMATH_CALUDE_root_square_relation_l1012_101279

theorem root_square_relation (b c : ℝ) : 
  (∃ r s : ℝ, r^2 + s^2 = -b ∧ r^2 * s^2 = c ∧ 
   r + s = 5 ∧ r * s = 2) → 
  c / b = -4 / 21 :=
by sorry

end NUMINAMATH_CALUDE_root_square_relation_l1012_101279


namespace NUMINAMATH_CALUDE_set_operations_and_range_l1012_101241

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 1 < x ∧ x < 4}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem set_operations_and_range :
  (∀ a : ℝ,
    (A ∩ B = {x | 1 < x ∧ x ≤ 3}) ∧
    (A ∪ B = {x | -1 ≤ x ∧ x < 4}) ∧
    ((Aᶜ ∩ Bᶜ) = {x | x < -1 ∨ 4 ≤ x}) ∧
    ((B ∩ C a = B) → a ≥ 4)) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_range_l1012_101241


namespace NUMINAMATH_CALUDE_piggy_bank_dimes_l1012_101243

/-- Proves that given $5.55 in dimes and quarters, with three more dimes than quarters, the number of dimes is 18 -/
theorem piggy_bank_dimes (total : ℚ) (dimes quarters : ℕ) : 
  total = (5 : ℚ) + (55 : ℚ) / 100 →
  dimes = quarters + 3 →
  (10 : ℚ) * dimes + (25 : ℚ) * quarters = total * 100 →
  dimes = 18 := by
sorry

end NUMINAMATH_CALUDE_piggy_bank_dimes_l1012_101243


namespace NUMINAMATH_CALUDE_min_value_parallel_vectors_l1012_101294

theorem min_value_parallel_vectors (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let a : Fin 2 → ℝ := ![3, 2]
  let b : Fin 2 → ℝ := ![x, 1 - y]
  (∃ (k : ℝ), ∀ i, a i = k * b i) →
  (3 / x + 2 / y) ≥ 8 ∧ ∃ x₀ y₀, x₀ > 0 ∧ y₀ > 0 ∧ 3 / x₀ + 2 / y₀ = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_parallel_vectors_l1012_101294


namespace NUMINAMATH_CALUDE_students_AD_combined_prove_students_AD_combined_l1012_101229

/-- The number of students in classes A and B combined -/
def students_AB : ℕ := 83

/-- The number of students in classes B and C combined -/
def students_BC : ℕ := 86

/-- The number of students in classes C and D combined -/
def students_CD : ℕ := 88

/-- Theorem stating that the number of students in classes A and D combined is 85 -/
theorem students_AD_combined : ℕ := 85

/-- Proof of the theorem -/
theorem prove_students_AD_combined : students_AD_combined = 85 := by
  sorry

end NUMINAMATH_CALUDE_students_AD_combined_prove_students_AD_combined_l1012_101229


namespace NUMINAMATH_CALUDE_polynomial_consecutive_integers_l1012_101238

/-- A polynomial P(n) = (n^5 + a) / b takes integer values for three consecutive integers
    if and only if (a, b) = (k, 1) or (11k ± 1, 11) for some integer k. -/
theorem polynomial_consecutive_integers (a b : ℕ+) :
  (∃ n : ℤ, ∀ i ∈ ({0, 1, 2} : Set ℤ), ∃ k : ℤ, (n + i)^5 + a = b * k) ↔
  (∃ k : ℤ, (a = k ∧ b = 1) ∨ (a = 11 * k + 1 ∧ b = 11) ∨ (a = 11 * k - 1 ∧ b = 11)) :=
sorry

end NUMINAMATH_CALUDE_polynomial_consecutive_integers_l1012_101238


namespace NUMINAMATH_CALUDE_water_container_percentage_l1012_101264

theorem water_container_percentage (initial_water : ℝ) (capacity : ℝ) (added_water : ℝ) :
  capacity = 40 →
  added_water = 14 →
  (initial_water + added_water) / capacity = 3/4 →
  initial_water / capacity = 2/5 :=
by sorry

end NUMINAMATH_CALUDE_water_container_percentage_l1012_101264


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l1012_101240

theorem inverse_proportion_ratio {x₁ x₂ y₁ y₂ : ℝ} (hx : x₁ ≠ 0 ∧ x₂ ≠ 0) (hy : y₁ ≠ 0 ∧ y₂ ≠ 0)
  (h_inv_prop : ∃ k : ℝ, k ≠ 0 ∧ x₁ * y₁ = k ∧ x₂ * y₂ = k)
  (h_x_ratio : x₁ / x₂ = 3 / 5) :
  y₁ / y₂ = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l1012_101240


namespace NUMINAMATH_CALUDE_cookies_in_bags_l1012_101236

def total_cookies : ℕ := 75
def cookies_per_bag : ℕ := 3

theorem cookies_in_bags : total_cookies / cookies_per_bag = 25 := by
  sorry

end NUMINAMATH_CALUDE_cookies_in_bags_l1012_101236


namespace NUMINAMATH_CALUDE_coin_distribution_l1012_101254

theorem coin_distribution (a b c d e : ℚ) : 
  a + b + c + d + e = 5 →  -- Total 5 coins
  a + b = c + d + e →  -- Sum condition
  b - a = c - b ∧ c - b = d - c ∧ d - c = e - d →  -- Arithmetic sequence
  e = 2/3 := by sorry

end NUMINAMATH_CALUDE_coin_distribution_l1012_101254


namespace NUMINAMATH_CALUDE_altitude_triangle_min_side_range_l1012_101213

/-- A triangle with side lengths a, b, c, perimeter 1, and altitudes that form a new triangle -/
structure AltitudeTriangle where
  a : Real
  b : Real
  c : Real
  perimeter_one : a + b + c = 1
  altitudes_form_triangle : 1/a + 1/b > 1/c ∧ 1/b + 1/c > 1/a ∧ 1/c + 1/a > 1/b
  a_smallest : a ≤ b ∧ a ≤ c

theorem altitude_triangle_min_side_range (t : AltitudeTriangle) :
  (3 - Real.sqrt 5) / 4 < t.a ∧ t.a ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_altitude_triangle_min_side_range_l1012_101213


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1012_101209

theorem sphere_surface_area (R : ℝ) (r₁ r₂ d : ℝ) : 
  r₁ = 24 → r₂ = 15 → d = 27 → 
  R^2 = r₁^2 + x^2 → 
  R^2 = r₂^2 + (d - x)^2 → 
  4 * π * R^2 = 2500 * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1012_101209


namespace NUMINAMATH_CALUDE_linear_system_existence_l1012_101232

theorem linear_system_existence :
  ∃ m : ℝ, ∀ x y : ℝ, (m - 1) * x - y = 1 ∧ m ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_existence_l1012_101232


namespace NUMINAMATH_CALUDE_min_sum_and_min_product_l1012_101205

/-- An arithmetic sequence with sum S_n -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- S_n is the sum of first n terms
  is_arithmetic : ∀ n : ℕ, S (n + 2) - S (n + 1) = S (n + 1) - S n

/-- The specific arithmetic sequence satisfying given conditions -/
def special_sequence (a : ArithmeticSequence) : Prop :=
  a.S 10 = 0 ∧ a.S 15 = 25

theorem min_sum_and_min_product (a : ArithmeticSequence) 
  (h : special_sequence a) :
  (∀ n : ℕ, n > 0 → a.S n ≥ a.S 5) ∧ 
  (∀ n : ℕ, n > 0 → n * (a.S n) ≥ -49) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_and_min_product_l1012_101205


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1012_101245

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying 4a_3 + a_11 - 3a_5 = 10, prove that 1/5 * a_4 = 1 -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h : 4 * seq.a 3 + seq.a 11 - 3 * seq.a 5 = 10) : 
  1/5 * seq.a 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1012_101245


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l1012_101248

theorem sum_of_x_and_y (x y m : ℝ) 
  (eq1 : x + m = 4) 
  (eq2 : y - 3 = m) : 
  x + y = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l1012_101248


namespace NUMINAMATH_CALUDE_umbrella_probability_l1012_101272

theorem umbrella_probability (p_forget : ℚ) (h1 : p_forget = 5/8) :
  1 - p_forget = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_umbrella_probability_l1012_101272


namespace NUMINAMATH_CALUDE_tenth_term_of_specific_geometric_sequence_l1012_101263

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r ^ (n - 1)

theorem tenth_term_of_specific_geometric_sequence :
  geometric_sequence 9 (1/3) 10 = 1/2187 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_specific_geometric_sequence_l1012_101263
