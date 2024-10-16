import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l581_58168

theorem polynomial_division_remainder : ∀ (z : ℝ),
  ∃ (r : ℝ),
    3 * z^3 - 4 * z^2 - 14 * z + 3 = (3 * z + 5) * (z^2 - 3 * z + 1/3) + r ∧
    r = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l581_58168


namespace NUMINAMATH_CALUDE_original_ratio_l581_58167

theorem original_ratio (x y : ℕ) (h1 : y = 48) (h2 : (x + 12) / y = 1/2) : x / y = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_original_ratio_l581_58167


namespace NUMINAMATH_CALUDE_chord_cosine_theorem_l581_58135

theorem chord_cosine_theorem (r : ℝ) (φ ψ θ : ℝ) 
  (h1 : φ + ψ + θ < π)
  (h2 : 3^2 = 2*r^2 - 2*r^2*Real.cos φ)
  (h3 : 4^2 = 2*r^2 - 2*r^2*Real.cos ψ)
  (h4 : 5^2 = 2*r^2 - 2*r^2*Real.cos θ)
  (h5 : 12^2 = 2*r^2 - 2*r^2*Real.cos (φ + ψ + θ))
  (h6 : r > 0) :
  Real.cos φ = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_chord_cosine_theorem_l581_58135


namespace NUMINAMATH_CALUDE_range_of_a_l581_58179

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 6) * (x - (2 * a + 5)) > 0}
def B (a : ℝ) : Set ℝ := {x | ((a^2 + 2) - x) * (2 * a - x) < 0}

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, 
    a > 1/2 → 
    (∀ x : ℝ, x ∈ B a → x ∈ A a) →
    (∃ x : ℝ, x ∈ A a ∧ x ∉ B a) →
    a > 1/2 ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l581_58179


namespace NUMINAMATH_CALUDE_t_range_l581_58154

-- Define the propositions p and q as functions of t
def p (t : ℝ) : Prop := ∀ x, x^2 + 2*x + 2*t - 4 ≠ 0

def q (t : ℝ) : Prop := 2 < t ∧ t < 3

-- Define the main theorem
theorem t_range (t : ℝ) : 
  (p t ∨ q t) ∧ ¬(p t ∧ q t) → (2 < t ∧ t ≤ 5/2) ∨ t ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_t_range_l581_58154


namespace NUMINAMATH_CALUDE_john_video_release_l581_58108

/-- The number of videos John releases per day -/
def videos_per_day : ℕ := 3

/-- The length of a short video in minutes -/
def short_video_length : ℕ := 2

/-- The number of short videos released per day -/
def short_videos_per_day : ℕ := 2

/-- The factor by which the long video is longer than the short videos -/
def long_video_factor : ℕ := 6

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Calculates the total minutes of video John releases per week -/
def total_minutes_per_week : ℕ :=
  days_per_week * (
    short_videos_per_day * short_video_length +
    (videos_per_day - short_videos_per_day) * (long_video_factor * short_video_length)
  )

theorem john_video_release :
  total_minutes_per_week = 112 := by
  sorry

end NUMINAMATH_CALUDE_john_video_release_l581_58108


namespace NUMINAMATH_CALUDE_average_of_21_numbers_l581_58191

theorem average_of_21_numbers (first_11_avg : ℝ) (last_11_avg : ℝ) (eleventh_num : ℝ) :
  first_11_avg = 48 →
  last_11_avg = 41 →
  eleventh_num = 55 →
  (11 * first_11_avg + 11 * last_11_avg - eleventh_num) / 21 = 44 := by
  sorry

end NUMINAMATH_CALUDE_average_of_21_numbers_l581_58191


namespace NUMINAMATH_CALUDE_base_difference_not_divisible_by_three_l581_58122

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- The difference of (2021)_b and (221)_b in base 10 -/
def baseDifference (b : Nat) : Nat :=
  toBase10 [2, 0, 2, 1] b - toBase10 [2, 2, 1] b

theorem base_difference_not_divisible_by_three (b : Nat) :
  b > 0 → (baseDifference b % 3 ≠ 0 ↔ b % 3 = 2) := by
  sorry

end NUMINAMATH_CALUDE_base_difference_not_divisible_by_three_l581_58122


namespace NUMINAMATH_CALUDE_total_earnings_is_5500_l581_58113

/-- Grant's earnings as a freelance math worker over three months -/
def grant_earnings : ℕ → ℕ
| 1 => 350  -- First month earnings
| 2 => 2 * grant_earnings 1 + 50  -- Second month earnings
| 3 => 4 * (grant_earnings 1 + grant_earnings 2)  -- Third month earnings
| _ => 0  -- For any other month (not needed for this problem)

/-- The total earnings for the first three months -/
def total_earnings : ℕ := grant_earnings 1 + grant_earnings 2 + grant_earnings 3

/-- Theorem stating that the total earnings for the first three months is $5500 -/
theorem total_earnings_is_5500 : total_earnings = 5500 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_is_5500_l581_58113


namespace NUMINAMATH_CALUDE_arithmetic_sequence_min_sum_l581_58173

/-- An arithmetic sequence with common difference d, first term a₁, and sum function S_n -/
structure ArithmeticSequence where
  d : ℝ
  a₁ : ℝ
  S_n : ℕ → ℝ

/-- The sum of an arithmetic sequence reaches its minimum -/
def sum_reaches_minimum (seq : ArithmeticSequence) (n : ℕ) : Prop :=
  ∀ k : ℕ, seq.S_n k ≥ seq.S_n n

/-- Theorem: For an arithmetic sequence with non-zero common difference,
    negative first term, and S₇ = S₁₃, the sum reaches its minimum when n = 10 -/
theorem arithmetic_sequence_min_sum
  (seq : ArithmeticSequence)
  (h_d : seq.d ≠ 0)
  (h_a₁ : seq.a₁ < 0)
  (h_S : seq.S_n 7 = seq.S_n 13) :
  sum_reaches_minimum seq 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_min_sum_l581_58173


namespace NUMINAMATH_CALUDE_quadratic_root_expression_l581_58146

theorem quadratic_root_expression (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ = 0 → 
  x₂^2 - 2*x₂ = 0 → 
  (x₁ * x₂) / (x₁^2 + x₂^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_expression_l581_58146


namespace NUMINAMATH_CALUDE_parabola_directrix_l581_58110

/-- The equation of a parabola -/
def parabola_eq (x y : ℝ) : Prop := y^2 = -12*x

/-- The equation of the directrix -/
def directrix_eq (x : ℝ) : Prop := x = 3

/-- Theorem: The directrix of the parabola y^2 = -12x is x = 3 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_eq x y → directrix_eq x :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l581_58110


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l581_58126

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 5 ways to distribute 6 indistinguishable balls into 3 indistinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute_balls 6 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l581_58126


namespace NUMINAMATH_CALUDE_special_sequence_tenth_term_l581_58133

/-- A sequence satisfying the given condition -/
def SpecialSequence (a : ℕ+ → ℤ) : Prop :=
  ∀ m n : ℕ+, a m + a n = a (m + n) - 2 * (m.val * n.val)

/-- The theorem to be proved -/
theorem special_sequence_tenth_term (a : ℕ+ → ℤ) 
  (h : SpecialSequence a) (h1 : a 1 = 1) : a 10 = 100 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_tenth_term_l581_58133


namespace NUMINAMATH_CALUDE_exists_odd_power_function_l581_58199

/-- A function satisfying the given conditions -/
def special_function (f : ℕ → ℕ) : Prop :=
  (∀ m n : ℕ, f (m * n) = f m * f n) ∧
  (∀ m n : ℕ, (m + n) ∣ (f m + f n))

/-- The main theorem -/
theorem exists_odd_power_function (f : ℕ → ℕ) (hf : special_function f) :
  ∃ k : ℕ, Odd k ∧ ∀ n : ℕ, f n = n^k :=
sorry

end NUMINAMATH_CALUDE_exists_odd_power_function_l581_58199


namespace NUMINAMATH_CALUDE_sum_a_d_equals_six_l581_58112

theorem sum_a_d_equals_six (a b c d : ℝ) 
  (eq1 : a + b = 12) 
  (eq2 : b + c = 9) 
  (eq3 : c + d = 3) : 
  a + d = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_d_equals_six_l581_58112


namespace NUMINAMATH_CALUDE_oxford_high_school_classes_l581_58184

/-- Represents the structure of Oxford High School -/
structure OxfordHighSchool where
  teachers : ℕ
  principal : ℕ
  students_per_class : ℕ
  total_people : ℕ

/-- Calculates the number of classes in Oxford High School -/
def number_of_classes (school : OxfordHighSchool) : ℕ :=
  let total_students := school.total_people - school.teachers - school.principal
  total_students / school.students_per_class

/-- Theorem stating that Oxford High School has 15 classes -/
theorem oxford_high_school_classes :
  let school : OxfordHighSchool := {
    teachers := 48,
    principal := 1,
    students_per_class := 20,
    total_people := 349
  }
  number_of_classes school = 15 := by
  sorry


end NUMINAMATH_CALUDE_oxford_high_school_classes_l581_58184


namespace NUMINAMATH_CALUDE_dennis_floor_l581_58152

/-- Given the floor arrangements of Frank, Charlie, Bob, and Dennis, prove that Dennis lives on the 6th floor. -/
theorem dennis_floor :
  ∀ (frank_floor charlie_floor bob_floor dennis_floor : ℕ),
    frank_floor = 16 →
    charlie_floor = frank_floor / 4 →
    bob_floor + 1 = charlie_floor →
    dennis_floor = charlie_floor + 2 →
    dennis_floor = 6 := by
  sorry

end NUMINAMATH_CALUDE_dennis_floor_l581_58152


namespace NUMINAMATH_CALUDE_parabola_properties_l581_58143

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the parabola D
def parabolaD (x y : ℝ) : Prop := y^2 = 4*x

-- Define point P
def P : ℝ × ℝ := (4, 0)

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 4)

-- Define the midpoint of PQ
def midpoint_PQ (Q : ℝ × ℝ) : Prop := (0, 0) = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the angle equality
def angle_equality (A B Q : ℝ × ℝ) : Prop :=
  (A.2 - Q.2) / (A.1 - Q.1) = -(B.2 - Q.2) / (B.1 - Q.1)

-- Define the line m
def line_m (x : ℝ) : Prop := x = 3

-- Theorem statement
theorem parabola_properties :
  ∀ (A B Q : ℝ × ℝ) (k : ℝ),
  parabolaD A.1 A.2 ∧ parabolaD B.1 B.2 ∧
  line_l k A.1 A.2 ∧ line_l k B.1 B.2 ∧
  midpoint_PQ Q →
  (∀ (x y : ℝ), parabolaD x y ↔ y^2 = 4*x) ∧
  angle_equality A B Q ∧
  (∃ (x : ℝ), line_m x ∧
    ∀ (A : ℝ × ℝ), parabolaD A.1 A.2 →
    ∃ (c : ℝ), ∀ (y : ℝ), 
      (x - (A.1 + 4) / 2)^2 + (y - A.2 / 2)^2 = ((A.1 - 4)^2 + A.2^2) / 4 →
      (x - 3)^2 + y^2 = c) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l581_58143


namespace NUMINAMATH_CALUDE_factors_180_multiples_15_l581_58186

/-- A function that returns the number of positive integers that are both factors of n and multiples of m -/
def count_common_factors_multiples (n m : ℕ) : ℕ :=
  (Finset.filter (λ x => n % x = 0 ∧ x % m = 0) (Finset.range n)).card

/-- Theorem stating that the number of positive integers that are both factors of 180 and multiples of 15 is 6 -/
theorem factors_180_multiples_15 : count_common_factors_multiples 180 15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_factors_180_multiples_15_l581_58186


namespace NUMINAMATH_CALUDE_dvd_pack_cost_l581_58120

/-- If 10 packs of DVDs cost 110 dollars, then the cost of one pack is 11 dollars -/
theorem dvd_pack_cost (total_cost : ℝ) (num_packs : ℕ) (h1 : total_cost = 110) (h2 : num_packs = 10) :
  total_cost / num_packs = 11 := by
  sorry

end NUMINAMATH_CALUDE_dvd_pack_cost_l581_58120


namespace NUMINAMATH_CALUDE_inscribed_right_triangle_perimeter_l581_58197

/-- A right-angled triangle inscribed in a circle -/
structure InscribedRightTriangle where
  /-- The diameter of the circle in which the triangle is inscribed -/
  outerDiameter : ℝ
  /-- The diameter of the circle inscribed in the triangle -/
  innerDiameter : ℝ

/-- The perimeter of an inscribed right-angled triangle -/
def perimeter (t : InscribedRightTriangle) : ℝ := sorry

/-- Theorem: The perimeter of a right-angled triangle inscribed in a circle 
    with diameter 18 cm is 43.2 cm -/
theorem inscribed_right_triangle_perimeter :
  ∀ (t : InscribedRightTriangle), 
  t.outerDiameter = 18 → 
  t.innerDiameter = 6 → 
  perimeter t = 43.2 := by sorry

end NUMINAMATH_CALUDE_inscribed_right_triangle_perimeter_l581_58197


namespace NUMINAMATH_CALUDE_inequality_problem_l581_58123

theorem inequality_problem (a b c d : ℝ) 
  (h1 : a > 0) (h2 : 0 > b) (h3 : b > -a) 
  (h4 : c < d) (h5 : d < 0) : 
  (a / d + b / c < 0) ∧ 
  (a - c > b - d) ∧ 
  (a * (d - c) > b * (d - c)) := by
sorry

end NUMINAMATH_CALUDE_inequality_problem_l581_58123


namespace NUMINAMATH_CALUDE_apex_to_center_distance_for_specific_pyramid_l581_58181

/-- Represents a rectangular pyramid with a parallel cut -/
structure CutPyramid where
  base_length : ℝ
  base_width : ℝ
  height : ℝ
  volume_ratio : ℝ

/-- The distance between the apex and the center of the circumsphere of the frustum -/
noncomputable def apex_to_center_distance (p : CutPyramid) : ℝ :=
  sorry

/-- Theorem stating the relationship between the pyramid's properties and the apex-to-center distance -/
theorem apex_to_center_distance_for_specific_pyramid :
  let p : CutPyramid := {
    base_length := 15,
    base_width := 20,
    height := 30,
    volume_ratio := 6
  }
  apex_to_center_distance p = 5 * (36 ^ (1/3 : ℝ)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_apex_to_center_distance_for_specific_pyramid_l581_58181


namespace NUMINAMATH_CALUDE_composite_and_three_factors_l581_58111

theorem composite_and_three_factors (n : ℕ) (h : n > 10) :
  let N := n^4 - 90*n^2 - 91*n - 90
  (∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ N = a * b) ∧
  (∃ (x y z : ℕ), x > 1 ∧ y > 1 ∧ z > 1 ∧ N = x * y * z) :=
by sorry

end NUMINAMATH_CALUDE_composite_and_three_factors_l581_58111


namespace NUMINAMATH_CALUDE_complement_of_union_l581_58102

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset Nat := {1, 2}

-- Define set N
def N : Finset Nat := {3, 4}

-- Theorem statement
theorem complement_of_union (u : Finset Nat) (m n : Finset Nat) 
  (hU : u = U) (hM : m = M) (hN : n = N) : 
  u \ (m ∪ n) = {5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l581_58102


namespace NUMINAMATH_CALUDE_highway_length_is_105_l581_58164

/-- The length of a highway where two cars meet after traveling from opposite ends -/
def highway_length (speed1 speed2 time : ℝ) : ℝ :=
  speed1 * time + speed2 * time

/-- Theorem: The highway length is 105 miles given the specific conditions -/
theorem highway_length_is_105 :
  highway_length 15 20 3 = 105 := by
  sorry

end NUMINAMATH_CALUDE_highway_length_is_105_l581_58164


namespace NUMINAMATH_CALUDE_fundraiser_problem_l581_58100

/-- Fundraiser Problem -/
theorem fundraiser_problem 
  (total_promised : ℕ)
  (amount_received : ℕ)
  (sally_owed : ℕ)
  (carl_owed : ℕ)
  (h_total : total_promised = 400)
  (h_received : amount_received = 285)
  (h_sally : sally_owed = 35)
  (h_carl : carl_owed = 35)
  : ∃ (amy_owed : ℕ) (derek_owed : ℕ),
    amy_owed = 30 ∧ 
    derek_owed = amy_owed / 2 ∧
    total_promised = amount_received + sally_owed + carl_owed + amy_owed + derek_owed :=
by sorry

end NUMINAMATH_CALUDE_fundraiser_problem_l581_58100


namespace NUMINAMATH_CALUDE_smallest_common_multiple_l581_58170

theorem smallest_common_multiple : ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > 0 ∧ 6 ∣ m ∧ 8 ∣ m ∧ 12 ∣ m → n ≤ m) ∧ 
  6 ∣ n ∧ 8 ∣ n ∧ 12 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_l581_58170


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l581_58124

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  12 * x - 6 * y + 3 * y - 24 * x = -12 * x - 3 * y := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) :
  3/2 * (a^2 * b - 2 * a * b^2) - 1/2 * (a * b^2 - 4 * a^2 * b) + 1/2 * a * b^2 =
  7/2 * a^2 * b - 3 * a * b^2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l581_58124


namespace NUMINAMATH_CALUDE_aaron_erasers_l581_58137

/-- The number of erasers Aaron gives away -/
def erasers_given : ℕ := 34

/-- The number of erasers Aaron ends with -/
def erasers_left : ℕ := 47

/-- The initial number of erasers Aaron had -/
def initial_erasers : ℕ := erasers_given + erasers_left

theorem aaron_erasers : initial_erasers = 81 := by
  sorry

end NUMINAMATH_CALUDE_aaron_erasers_l581_58137


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l581_58134

/-- If a rectangle's width is halved and its area increases by 30.000000000000004%,
    then the length of the rectangle increases by 160%. -/
theorem rectangle_dimension_change (L W : ℝ) (L' W' : ℝ) (h1 : W' = W / 2) 
  (h2 : L' * W' = 1.30000000000000004 * L * W) : L' = 2.6 * L := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l581_58134


namespace NUMINAMATH_CALUDE_equal_grid_values_l581_58156

/-- Represents a point in the infinite square grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents an admissible polygon on the grid --/
structure AdmissiblePolygon where
  vertices : List GridPoint
  area : ℕ
  area_gt_two : area > 2

/-- The grid of natural numbers --/
def Grid := GridPoint → ℕ

/-- The value of an admissible polygon --/
def value (grid : Grid) (polygon : AdmissiblePolygon) : ℕ := sorry

/-- Two polygons are congruent --/
def congruent (p1 p2 : AdmissiblePolygon) : Prop := sorry

/-- Main theorem --/
theorem equal_grid_values (grid : Grid) :
  (∀ p1 p2 : AdmissiblePolygon, congruent p1 p2 → value grid p1 = value grid p2) →
  (∀ p1 p2 : GridPoint, grid p1 = grid p2) := by sorry

end NUMINAMATH_CALUDE_equal_grid_values_l581_58156


namespace NUMINAMATH_CALUDE_negative_sum_bound_l581_58190

theorem negative_sum_bound (a b c : ℝ) (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  min (a + 1/b) (min (b + 1/c) (c + 1/a)) ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_sum_bound_l581_58190


namespace NUMINAMATH_CALUDE_partnership_investment_l581_58136

/-- Represents a partnership with three partners -/
structure Partnership where
  investmentA : ℝ
  investmentB : ℝ
  investmentC : ℝ
  totalProfit : ℝ
  cProfit : ℝ

/-- Calculates the total investment of the partnership -/
def totalInvestment (p : Partnership) : ℝ :=
  p.investmentA + p.investmentB + p.investmentC

/-- Theorem stating that if the given conditions are met, 
    then Partner C's investment is 36000 -/
theorem partnership_investment 
  (p : Partnership) 
  (h1 : p.investmentA = 24000)
  (h2 : p.investmentB = 32000)
  (h3 : p.totalProfit = 92000)
  (h4 : p.cProfit = 36000)
  (h5 : p.cProfit / p.totalProfit = p.investmentC / totalInvestment p) :
  p.investmentC = 36000 :=
sorry

end NUMINAMATH_CALUDE_partnership_investment_l581_58136


namespace NUMINAMATH_CALUDE_intersection_point_a_l581_58119

-- Define the function f
def f (b : ℤ) (x : ℝ) : ℝ := 2 * x + b

-- Define the theorem
theorem intersection_point_a (b : ℤ) (a : ℤ) :
  (∃ (x : ℝ), f b x = a ∧ f b (-4) = a) →  -- f and f^(-1) intersect at (-4, a)
  a = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_point_a_l581_58119


namespace NUMINAMATH_CALUDE_no_integer_solution_for_trig_equation_l581_58161

theorem no_integer_solution_for_trig_equation : 
  ¬ ∃ (a b : ℤ), Real.sqrt (4 - 3 * Real.sin (30 * π / 180)) = a + b * (1 / Real.sin (30 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_trig_equation_l581_58161


namespace NUMINAMATH_CALUDE_remainder_of_3_600_mod_19_l581_58118

theorem remainder_of_3_600_mod_19 : 3^600 % 19 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_3_600_mod_19_l581_58118


namespace NUMINAMATH_CALUDE_root_conditions_l581_58103

/-- The equation x^4 + px^2 + q = 0 has real roots satisfying x₂/x₁ = x₃/x₂ = x₄/x₃ 
    if and only if p < 0 and q = p^2/4 -/
theorem root_conditions (p q : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₃ ≠ 0 ∧
    x₁^4 + p*x₁^2 + q = 0 ∧
    x₂^4 + p*x₂^2 + q = 0 ∧
    x₃^4 + p*x₃^2 + q = 0 ∧
    x₄^4 + p*x₄^2 + q = 0 ∧
    x₂/x₁ = x₃/x₂ ∧ x₃/x₂ = x₄/x₃) ↔
  (p < 0 ∧ q = p^2/4) :=
by sorry


end NUMINAMATH_CALUDE_root_conditions_l581_58103


namespace NUMINAMATH_CALUDE_tangent_line_equations_l581_58104

-- Define the function
def f (x : ℝ) : ℝ := -x^3 - 1

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := -3 * x^2

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Define the tangent line equation
def tangent_line (m : ℝ) (x : ℝ) : ℝ := 
  -m^3 - 1 + (f' m) * (x - m)

-- Theorem statement
theorem tangent_line_equations :
  ∃ (m₁ m₂ : ℝ), 
    m₁ ≠ m₂ ∧
    tangent_line m₁ P.1 = P.2 ∧
    tangent_line m₂ P.1 = P.2 ∧
    (∀ (x : ℝ), tangent_line m₁ x = -3*x - 1) ∧
    (∀ (x : ℝ), tangent_line m₂ x = -(3*x + 5) / 4) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equations_l581_58104


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l581_58176

/-- Hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : 0 < a ∧ 0 < b

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Eccentricity of a specific hyperbola -/
theorem hyperbola_eccentricity (C : Hyperbola) (P F₁ F₂ Q O : Point) : 
  (P.x^2 / C.a^2 - P.y^2 / C.b^2 = 1) →  -- P is on the hyperbola
  (F₁.x < 0 ∧ F₁.y = 0 ∧ F₂.x > 0 ∧ F₂.y = 0) →  -- F₁ and F₂ are left and right foci
  ((P.x - F₂.x) * (F₁.x - F₂.x) + (P.y - F₂.y) * (F₁.y - F₂.y) = 0) →  -- PF₂ ⟂ F₁F₂
  (∃ t : ℝ, Q.x = 0 ∧ Q.y = t * P.y + (1 - t) * F₁.y) →  -- PF₁ intersects y-axis at Q
  (O.x = 0 ∧ O.y = 0) →  -- O is the origin
  (∃ M : Point, ∃ r : ℝ, 
    (M.x - O.x)^2 + (M.y - O.y)^2 = r^2 ∧
    (M.x - F₂.x)^2 + (M.y - F₂.y)^2 = r^2 ∧
    (M.x - P.x)^2 + (M.y - P.y)^2 = r^2 ∧
    (M.x - Q.x)^2 + (M.y - Q.y)^2 = r^2) →  -- OF₂PQ has an inscribed circle
  (F₂.x^2 - F₁.x^2) / C.a^2 = 4  -- Eccentricity is 2
:= by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l581_58176


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l581_58149

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  1/a + 1/b ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l581_58149


namespace NUMINAMATH_CALUDE_regular_ngon_smallest_area_and_perimeter_l581_58180

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- An n-gon circumscribed about a circle --/
structure CircumscribedNGon where
  n : ℕ
  circle : Circle
  vertices : Fin n → ℝ × ℝ

/-- Checks if an n-gon is regular --/
def is_regular (ngon : CircumscribedNGon) : Prop :=
  sorry

/-- Calculates the area of an n-gon --/
def area (ngon : CircumscribedNGon) : ℝ :=
  sorry

/-- Calculates the perimeter of an n-gon --/
def perimeter (ngon : CircumscribedNGon) : ℝ :=
  sorry

/-- Theorem: The regular n-gon has the smallest area and perimeter among all n-gons circumscribed about a given circle --/
theorem regular_ngon_smallest_area_and_perimeter (n : ℕ) (c : Circle) :
  ∀ (ngon : CircumscribedNGon), ngon.n = n ∧ ngon.circle = c →
    ∃ (reg_ngon : CircumscribedNGon), 
      reg_ngon.n = n ∧ reg_ngon.circle = c ∧ is_regular reg_ngon ∧
      area reg_ngon ≤ area ngon ∧ perimeter reg_ngon ≤ perimeter ngon :=
  sorry

end NUMINAMATH_CALUDE_regular_ngon_smallest_area_and_perimeter_l581_58180


namespace NUMINAMATH_CALUDE_sqrt_seven_identities_l581_58148

theorem sqrt_seven_identities (a b : ℝ) (ha : a = Real.sqrt 7 + 2) (hb : b = Real.sqrt 7 - 2) :
  (a * b = 3) ∧ (a^2 + b^2 - a * b = 19) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_identities_l581_58148


namespace NUMINAMATH_CALUDE_statutory_capital_scientific_notation_l581_58169

/-- The statutory capital of the Asian Infrastructure Investment Bank in U.S. dollars -/
def statutory_capital : ℝ := 100000000000

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Theorem stating that the statutory capital in scientific notation is 1 × 10^11 -/
theorem statutory_capital_scientific_notation :
  ∃ (sn : ScientificNotation),
    sn.coefficient = 1 ∧
    sn.exponent = 11 ∧
    statutory_capital = sn.coefficient * (10 : ℝ) ^ sn.exponent :=
by sorry

end NUMINAMATH_CALUDE_statutory_capital_scientific_notation_l581_58169


namespace NUMINAMATH_CALUDE_isosceles_triangles_l581_58192

/-- A circle with two equal chords that extend to intersect -/
structure CircleWithIntersectingChords where
  /-- The circle -/
  circle : Set (ℝ × ℝ)
  /-- The center of the circle -/
  center : ℝ × ℝ
  /-- First chord endpoint -/
  A : ℝ × ℝ
  /-- Second chord endpoint -/
  B : ℝ × ℝ
  /-- Third chord endpoint -/
  C : ℝ × ℝ
  /-- Fourth chord endpoint -/
  D : ℝ × ℝ
  /-- Intersection point of extended chords -/
  P : ℝ × ℝ
  /-- A and B are on the circle -/
  hAB : A ∈ circle ∧ B ∈ circle
  /-- C and D are on the circle -/
  hCD : C ∈ circle ∧ D ∈ circle
  /-- AB and CD are equal chords -/
  hEqualChords : dist A B = dist C D
  /-- P is on the extension of AB beyond B -/
  hPAB : ∃ t > 1, P = A + t • (B - A)
  /-- P is on the extension of CD beyond C -/
  hPCD : ∃ t > 1, P = D + t • (C - D)

/-- The main theorem: triangles APD and BPC are isosceles -/
theorem isosceles_triangles (cfg : CircleWithIntersectingChords) :
  dist cfg.A cfg.P = dist cfg.D cfg.P ∧ dist cfg.B cfg.P = dist cfg.C cfg.P := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangles_l581_58192


namespace NUMINAMATH_CALUDE_sixteen_pow_six_mod_nine_l581_58131

theorem sixteen_pow_six_mod_nine : 16^6 ≡ 1 [ZMOD 9] := by
  sorry

end NUMINAMATH_CALUDE_sixteen_pow_six_mod_nine_l581_58131


namespace NUMINAMATH_CALUDE_min_value_fraction_l581_58128

theorem min_value_fraction (x : ℝ) (h : x > 8) : 
  x^2 / (x - 8)^2 ≥ 1 ∧ ∀ ε > 0, ∃ y > 8, y^2 / (y - 8)^2 < 1 + ε :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l581_58128


namespace NUMINAMATH_CALUDE_lottery_probability_l581_58162

theorem lottery_probability : 
  (1 : ℚ) / 30 * (1 / 50 * 1 / 49 * 1 / 48 * 1 / 47 * 1 / 46) = 1 / 7627536000 := by
  sorry

end NUMINAMATH_CALUDE_lottery_probability_l581_58162


namespace NUMINAMATH_CALUDE_special_function_is_zero_l581_58147

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x ≥ 0) ∧
  (∀ x, DifferentiableAt ℝ f x) ∧
  (∀ x, deriv f x ≥ 0) ∧
  (∀ n : ℤ, f n = 0)

/-- Theorem stating that any function satisfying the conditions must be identically zero -/
theorem special_function_is_zero (f : ℝ → ℝ) (hf : SpecialFunction f) : 
  ∀ x, f x = 0 := by sorry

end NUMINAMATH_CALUDE_special_function_is_zero_l581_58147


namespace NUMINAMATH_CALUDE_marble_problem_l581_58193

theorem marble_problem (total initial_marbles : ℕ) 
  (white red blue : ℕ) 
  (h1 : total = 50)
  (h2 : red = blue)
  (h3 : white + red + blue = total)
  (h4 : total - (2 * (white - blue)) = 40) :
  white = 5 := by sorry

end NUMINAMATH_CALUDE_marble_problem_l581_58193


namespace NUMINAMATH_CALUDE_unique_five_digit_number_l581_58115

theorem unique_five_digit_number : ∀ N : ℕ,
  (10000 ≤ N ∧ N < 100000) →
  let P := 200000 + N
  let Q := 10 * N + 2
  Q = 3 * P →
  N = 85714 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_five_digit_number_l581_58115


namespace NUMINAMATH_CALUDE_not_perfect_squares_l581_58155

theorem not_perfect_squares : 
  (∃ x : ℝ, (6 : ℝ)^3032 = x^2) ∧ 
  (∀ x : ℝ, (7 : ℝ)^3033 ≠ x^2) ∧ 
  (∃ x : ℝ, (8 : ℝ)^3034 = x^2) ∧ 
  (∀ x : ℝ, (9 : ℝ)^3035 ≠ x^2) ∧ 
  (∃ x : ℝ, (10 : ℝ)^3036 = x^2) := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_squares_l581_58155


namespace NUMINAMATH_CALUDE_oranges_from_joyce_calculation_l581_58142

/-- Represents the number of oranges Clarence has initially. -/
def initial_oranges : ℕ := 5

/-- Represents the total number of oranges Clarence has after receiving some from Joyce. -/
def total_oranges : ℕ := 8

/-- Represents the number of oranges Joyce gave to Clarence. -/
def oranges_from_joyce : ℕ := total_oranges - initial_oranges

/-- Proves that the number of oranges Joyce gave to Clarence is equal to the difference
    between Clarence's total oranges after receiving from Joyce and Clarence's initial oranges. -/
theorem oranges_from_joyce_calculation :
  oranges_from_joyce = total_oranges - initial_oranges :=
by sorry

end NUMINAMATH_CALUDE_oranges_from_joyce_calculation_l581_58142


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l581_58171

theorem quadratic_equation_roots (m : ℝ) : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - m*x₁ - 1 = 0 ∧ x₂^2 - m*x₂ - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l581_58171


namespace NUMINAMATH_CALUDE_trapezoid_perimeters_l581_58138

/-- A trapezoid with given measurements -/
structure Trapezoid where
  longerBase : ℝ
  height : ℝ
  leg1 : ℝ
  leg2 : ℝ

/-- The set of possible perimeters for a given trapezoid -/
def possiblePerimeters (t : Trapezoid) : Set ℝ :=
  {p | ∃ shorterBase : ℝ, 
    p = t.longerBase + t.leg1 + t.leg2 + shorterBase ∧
    shorterBase > 0 ∧
    (shorterBase = t.longerBase - Real.sqrt (t.leg1^2 - t.height^2) - Real.sqrt (t.leg2^2 - t.height^2) ∨
     shorterBase = t.longerBase + Real.sqrt (t.leg1^2 - t.height^2) - Real.sqrt (t.leg2^2 - t.height^2))}

/-- The theorem to be proved -/
theorem trapezoid_perimeters (t : Trapezoid) 
  (h1 : t.longerBase = 30)
  (h2 : t.height = 24)
  (h3 : t.leg1 = 25)
  (h4 : t.leg2 = 30) :
  possiblePerimeters t = {90, 104} := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeters_l581_58138


namespace NUMINAMATH_CALUDE_horner_method_first_step_l581_58101

def f (x : ℝ) : ℝ := 7 * x^6 + 6 * x^5 + 3 * x^2 + 2

def horner_first_step (a₆ a₅ : ℝ) (x : ℝ) : ℝ := a₆ * x + a₅

theorem horner_method_first_step :
  horner_first_step 7 6 4 = 34 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_first_step_l581_58101


namespace NUMINAMATH_CALUDE_g_equals_zero_at_negative_one_l581_58182

/-- Given a function g(x) = 3x^4 + 2x^3 - x^2 - 4x + s, prove that g(-1) = 0 when s = -4 -/
theorem g_equals_zero_at_negative_one (s : ℝ) : 
  let g : ℝ → ℝ := λ x => 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s
  g (-1) = 0 ↔ s = -4 := by
  sorry

end NUMINAMATH_CALUDE_g_equals_zero_at_negative_one_l581_58182


namespace NUMINAMATH_CALUDE_division_with_remainder_l581_58150

theorem division_with_remainder : ∃ (q r : ℤ), 1234567 = 131 * q + r ∧ 0 ≤ r ∧ r < 131 ∧ r = 36 := by
  sorry

end NUMINAMATH_CALUDE_division_with_remainder_l581_58150


namespace NUMINAMATH_CALUDE_domain_of_f_l581_58159

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.log (2 * x + 1) / Real.log (1/2))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -1/2 < x ∧ x ≠ 0} :=
by sorry

end NUMINAMATH_CALUDE_domain_of_f_l581_58159


namespace NUMINAMATH_CALUDE_triangle_circle_intersection_l581_58198

theorem triangle_circle_intersection (DE DF EY FY : ℕ) (EF : ℝ) : 
  DE = 65 →
  DF = 104 →
  EY + FY = EF →
  FY * EF = 39 * 169 →
  EF = 169 :=
by sorry

end NUMINAMATH_CALUDE_triangle_circle_intersection_l581_58198


namespace NUMINAMATH_CALUDE_expected_population_after_increase_l581_58157

def current_population : ℝ := 1.75
def percentage_increase : ℝ := 325

theorem expected_population_after_increase :
  let increase_factor := 1 + percentage_increase / 100
  let expected_population := current_population * increase_factor
  expected_population = 7.4375 := by sorry

end NUMINAMATH_CALUDE_expected_population_after_increase_l581_58157


namespace NUMINAMATH_CALUDE_triangle_median_inequality_l581_58121

theorem triangle_median_inequality (a b c ma mb mc : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hma : ma > 0) (hmb : mb > 0) (hmc : mc > 0)
  (h_ma : 4 * ma^2 = 2 * (b^2 + c^2) - a^2)
  (h_mb : 4 * mb^2 = 2 * (c^2 + a^2) - b^2)
  (h_mc : 4 * mc^2 = 2 * (a^2 + b^2) - c^2) :
  ma^2 / a^2 + mb^2 / b^2 + mc^2 / c^2 ≥ 9/4 := by
sorry

end NUMINAMATH_CALUDE_triangle_median_inequality_l581_58121


namespace NUMINAMATH_CALUDE_correct_number_of_pitbulls_l581_58187

/-- Represents the number of pitbulls James has -/
def num_pitbulls : ℕ := 2

/-- Represents the number of huskies James has -/
def num_huskies : ℕ := 5

/-- Represents the number of golden retrievers James has -/
def num_golden_retrievers : ℕ := 4

/-- Represents the number of pups each husky and pitbull has -/
def pups_per_husky_pitbull : ℕ := 3

/-- Represents the additional number of pups each golden retriever has compared to huskies -/
def additional_pups_golden : ℕ := 2

/-- Represents the difference between total pups and adult dogs -/
def pup_adult_difference : ℕ := 30

theorem correct_number_of_pitbulls :
  (num_huskies * pups_per_husky_pitbull) +
  (num_golden_retrievers * (pups_per_husky_pitbull + additional_pups_golden)) +
  (num_pitbulls * pups_per_husky_pitbull) =
  (num_huskies + num_golden_retrievers + num_pitbulls) + pup_adult_difference := by
  sorry

end NUMINAMATH_CALUDE_correct_number_of_pitbulls_l581_58187


namespace NUMINAMATH_CALUDE_red_peaches_count_l581_58107

/-- Given a basket of peaches, calculate the number of red peaches. -/
def red_peaches (total_peaches green_peaches : ℕ) : ℕ :=
  total_peaches - green_peaches

/-- Theorem: The number of red peaches in the basket is 4. -/
theorem red_peaches_count : red_peaches 7 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_red_peaches_count_l581_58107


namespace NUMINAMATH_CALUDE_price_restoration_l581_58175

theorem price_restoration (original_price : ℝ) (original_price_pos : 0 < original_price) : 
  let price_after_increases := original_price * (1 + 0.1) * (1 + 0.1) * (1 + 0.05)
  let price_after_decrease := price_after_increases * (1 - 0.22)
  price_after_decrease = original_price := by sorry

end NUMINAMATH_CALUDE_price_restoration_l581_58175


namespace NUMINAMATH_CALUDE_time_between_flashes_l581_58158

/-- Represents the number of flashes in 3/4 of an hour -/
def flashes_per_three_quarters_hour : ℕ := 300

/-- Represents 3/4 of an hour in seconds -/
def three_quarters_hour_in_seconds : ℕ := 45 * 60

/-- Theorem stating that the time between flashes is 9 seconds -/
theorem time_between_flashes :
  three_quarters_hour_in_seconds / flashes_per_three_quarters_hour = 9 := by
  sorry

end NUMINAMATH_CALUDE_time_between_flashes_l581_58158


namespace NUMINAMATH_CALUDE_faye_finished_problems_l581_58132

theorem faye_finished_problems (math_problems science_problems left_for_homework : ℕ)
  (h1 : math_problems = 46)
  (h2 : science_problems = 9)
  (h3 : left_for_homework = 15) :
  math_problems + science_problems - left_for_homework = 40 := by
  sorry

end NUMINAMATH_CALUDE_faye_finished_problems_l581_58132


namespace NUMINAMATH_CALUDE_projection_of_a_onto_b_l581_58139

noncomputable section

/-- Given two non-zero vectors a and b in ℝ², prove that under certain conditions,
    the projection of a onto b is (1/4) * b. -/
theorem projection_of_a_onto_b (a b : ℝ × ℝ) : 
  a ≠ (0, 0) → 
  b = (Real.sqrt 3, 1) → 
  a.1 * b.1 + a.2 * b.2 = π / 3 → 
  (a.1 - b.1) * a.1 + (a.2 - b.2) * a.2 = 0 → 
  ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b = (1/4) • b := by
  sorry

end

end NUMINAMATH_CALUDE_projection_of_a_onto_b_l581_58139


namespace NUMINAMATH_CALUDE_triangle_with_lattice_point_is_equilateral_l581_58127

/-- A triangle in a plane --/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- The perimeter of a triangle --/
def perimeter (t : Triangle) : ℝ := sorry

/-- Whether a point is a lattice point --/
def is_lattice_point (p : ℝ × ℝ) : Prop := sorry

/-- Whether a point is on or inside a triangle --/
def point_in_triangle (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Whether two triangles are congruent --/
def congruent (t1 t2 : Triangle) : Prop := sorry

/-- Whether a triangle is equilateral --/
def is_equilateral (t : Triangle) : Prop := sorry

theorem triangle_with_lattice_point_is_equilateral (t : Triangle) :
  perimeter t = 3 + 2 * Real.sqrt 3 →
  (∀ t' : Triangle, congruent t t' → ∃ p : ℝ × ℝ, is_lattice_point p ∧ point_in_triangle p t') →
  is_equilateral t :=
sorry

end NUMINAMATH_CALUDE_triangle_with_lattice_point_is_equilateral_l581_58127


namespace NUMINAMATH_CALUDE_cost_of_one_juice_and_sandwich_janice_purchase_l581_58106

/-- Given the cost of multiple juices and sandwiches, calculate the cost of one juice and one sandwich. -/
theorem cost_of_one_juice_and_sandwich 
  (total_juice_cost : ℝ) 
  (juice_quantity : ℕ) 
  (total_sandwich_cost : ℝ) 
  (sandwich_quantity : ℕ) : 
  total_juice_cost / juice_quantity + total_sandwich_cost / sandwich_quantity = 5 :=
by
  sorry

/-- Specific instance of the theorem with given values -/
theorem janice_purchase : 
  (10 : ℝ) / 5 + (6 : ℝ) / 2 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_cost_of_one_juice_and_sandwich_janice_purchase_l581_58106


namespace NUMINAMATH_CALUDE_column_compression_strength_l581_58105

theorem column_compression_strength (T H L : ℚ) : 
  T = 3 → H = 6 → L = (15 * T^5) / H^3 → L = 55 / 13 := by sorry

end NUMINAMATH_CALUDE_column_compression_strength_l581_58105


namespace NUMINAMATH_CALUDE_water_speed_in_swimming_problem_l581_58130

/-- Proves that the speed of water is 4 km/h given the conditions of the swimming problem -/
theorem water_speed_in_swimming_problem : 
  ∀ (still_water_speed : ℝ) (distance : ℝ) (time : ℝ) (water_speed : ℝ),
    still_water_speed = 8 →
    distance = 8 →
    time = 2 →
    distance = (still_water_speed - water_speed) * time →
    water_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_water_speed_in_swimming_problem_l581_58130


namespace NUMINAMATH_CALUDE_circular_arrangement_theorem_l581_58129

/-- Represents a circular arrangement of people -/
structure CircularArrangement where
  n : ℕ  -- Total number of people
  dist : ℕ → ℕ → ℕ  -- Distance function between two positions

/-- The main theorem -/
theorem circular_arrangement_theorem (c : CircularArrangement) :
  (c.dist 31 7 = c.dist 31 14) → c.n = 41 := by
  sorry

/-- Helper function to calculate clockwise distance -/
def clockwise_distance (n : ℕ) (a b : ℕ) : ℕ :=
  if b ≥ a then b - a else n - a + b

/-- Axiom: The distance function in CircularArrangement is defined by clockwise_distance -/
axiom distance_defined (c : CircularArrangement) :
  ∀ a b, c.dist a b = clockwise_distance c.n a b

/-- Axiom: The arrangement is circular, so the distance from a to b equals the distance from b to a -/
axiom circular_symmetry (c : CircularArrangement) :
  ∀ a b, c.dist a b = c.dist b a

end NUMINAMATH_CALUDE_circular_arrangement_theorem_l581_58129


namespace NUMINAMATH_CALUDE_problem_solution_l581_58160

theorem problem_solution : 
  (1 + 3/4 - 3/8 + 5/6) / (-1/24) = -53 ∧ 
  -2^2 + (-4) / 2 * (1/2) + |(-3)| = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l581_58160


namespace NUMINAMATH_CALUDE_product_price_relationship_l581_58165

/-- Proves the relationship between fall and spring prices of a product given specific conditions -/
theorem product_price_relationship (fall_amount : ℝ) (total_cost : ℝ) (spring_difference : ℝ) : 
  fall_amount = 550 ∧ 
  total_cost = 825 ∧ 
  spring_difference = 220 →
  ∃ (spring_price : ℝ),
    spring_price = total_cost / (fall_amount - spring_difference) ∧
    spring_price = total_cost / fall_amount + 1 ∧
    spring_price = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_product_price_relationship_l581_58165


namespace NUMINAMATH_CALUDE_mall_promotion_max_purchase_l581_58177

/-- Calculates the maximum value of goods that can be purchased given a cashback rule and initial amount --/
def max_purchase_value (cashback_amount : ℕ) (cashback_threshold : ℕ) (initial_amount : ℕ) : ℕ :=
  sorry

/-- The maximum value of goods that can be purchased given the specific conditions --/
theorem mall_promotion_max_purchase :
  max_purchase_value 40 200 650 = 770 :=
sorry

end NUMINAMATH_CALUDE_mall_promotion_max_purchase_l581_58177


namespace NUMINAMATH_CALUDE_equation_has_one_solution_l581_58144

-- Define the functions f and p
def f (x : ℝ) : ℝ := |x + 1|

def p (x a : ℝ) : ℝ := |x - 4| + a

-- Define the domain of f
def domain_f (x : ℝ) : Prop := x ≠ -4 ∧ x ≠ 1

-- Define the set of values for a
def valid_a (a : ℝ) : Prop := (a > -5 ∧ a < -1) ∨ (a > -1 ∧ a < 5)

-- Theorem statement
theorem equation_has_one_solution (a : ℝ) :
  valid_a a →
  ∃! x : ℝ, domain_f x ∧ f x = p x a :=
by sorry

end NUMINAMATH_CALUDE_equation_has_one_solution_l581_58144


namespace NUMINAMATH_CALUDE_not_pythagorean_triple_l581_58185

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem not_pythagorean_triple : ¬ is_pythagorean_triple 7 25 26 := by
  sorry

end NUMINAMATH_CALUDE_not_pythagorean_triple_l581_58185


namespace NUMINAMATH_CALUDE_constant_term_is_60_l581_58125

/-- The constant term in the expansion of (√x - 2/x)^6 -/
def constantTerm : ℕ :=
  -- We define the constant term without using the solution steps
  -- This definition should be completed in the proof
  sorry

/-- Proof that the constant term in the expansion of (√x - 2/x)^6 is 60 -/
theorem constant_term_is_60 : constantTerm = 60 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_is_60_l581_58125


namespace NUMINAMATH_CALUDE_max_value_of_expression_l581_58195

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the theorem
theorem max_value_of_expression (hf : ∀ x, f x ∈ Set.Icc (-3) 5) 
                                 (hg : ∀ x, g x ∈ Set.Icc (-4) 2) :
  ∃ d, d = 45 ∧ ∀ x, 2 * f x * g x + f x ≤ d ∧ 
  ∃ y, 2 * f y * g y + f y = d :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l581_58195


namespace NUMINAMATH_CALUDE_inverse_value_l581_58163

noncomputable section

variables (f g : ℝ → ℝ)

-- f⁻¹(g(x)) = x^4 - 1
axiom inverse_composition (x : ℝ) : f⁻¹ (g x) = x^4 - 1

-- g has an inverse
axiom g_has_inverse : Function.Bijective g

theorem inverse_value : g⁻¹ (f 10) = (11 : ℝ)^(1/4) := by sorry

end NUMINAMATH_CALUDE_inverse_value_l581_58163


namespace NUMINAMATH_CALUDE_pizza_slice_angle_l581_58145

theorem pizza_slice_angle (p : ℝ) (h1 : p > 0) (h2 : p < 1) (h3 : p = 1/8) :
  let angle := p * 360
  angle = 45 := by sorry

end NUMINAMATH_CALUDE_pizza_slice_angle_l581_58145


namespace NUMINAMATH_CALUDE_trig_identity_proof_l581_58140

theorem trig_identity_proof : 
  Real.cos (70 * π / 180) * Real.sin (80 * π / 180) + 
  Real.cos (20 * π / 180) * Real.sin (10 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l581_58140


namespace NUMINAMATH_CALUDE_factorial_equation_sum_of_digits_l581_58194

/-- The factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- The theorem statement -/
theorem factorial_equation_sum_of_digits :
  ∃ (n : ℕ), n > 0 ∧ 
  (factorial (n + 1) + 2 * factorial (n + 2) = factorial n * 871) ∧
  (sumOfDigits n = 10) := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_sum_of_digits_l581_58194


namespace NUMINAMATH_CALUDE_parkway_elementary_girls_not_soccer_l581_58183

theorem parkway_elementary_girls_not_soccer (total_students : ℕ) (boys : ℕ) (soccer_players : ℕ) 
  (h1 : total_students = 420)
  (h2 : boys = 296)
  (h3 : soccer_players = 250)
  (h4 : (86 : ℚ) / 100 * soccer_players = ↑(boys_playing_soccer))
  (boys_playing_soccer : ℕ) :
  total_students - soccer_players - (boys - boys_playing_soccer) = 89 := by
  sorry

#check parkway_elementary_girls_not_soccer

end NUMINAMATH_CALUDE_parkway_elementary_girls_not_soccer_l581_58183


namespace NUMINAMATH_CALUDE_problem_solution_l581_58141

theorem problem_solution : 18 * ((150 / 3) + (40 / 5) + (16 / 32) + 2) = 1089 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l581_58141


namespace NUMINAMATH_CALUDE_perfect_cubes_with_special_property_l581_58174

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def last_three_digits (n : ℕ) : ℕ := n % 1000

def erase_last_three_digits (n : ℕ) : ℕ := n / 1000

theorem perfect_cubes_with_special_property :
  ∀ n : ℕ,
    n > 0 ∧
    is_perfect_cube n ∧
    n % 10 ≠ 0 ∧
    is_perfect_cube (erase_last_three_digits n) →
    n = 1331 ∨ n = 1728 :=
by sorry

end NUMINAMATH_CALUDE_perfect_cubes_with_special_property_l581_58174


namespace NUMINAMATH_CALUDE_zero_exponent_l581_58172

theorem zero_exponent (x : ℝ) (hx : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_l581_58172


namespace NUMINAMATH_CALUDE_final_milk_water_ratio_l581_58114

/- Given conditions -/
def initial_ratio : Rat := 1 / 5
def can_capacity : ℝ := 8
def additional_milk : ℝ := 2

/- Theorem to prove -/
theorem final_milk_water_ratio :
  let initial_mixture := can_capacity - additional_milk
  let initial_milk := initial_mixture * (initial_ratio / (1 + initial_ratio))
  let initial_water := initial_mixture * (1 / (1 + initial_ratio))
  let final_milk := initial_milk + additional_milk
  let final_water := initial_water
  (final_milk / final_water) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_final_milk_water_ratio_l581_58114


namespace NUMINAMATH_CALUDE_monkey_climb_theorem_l581_58116

/-- Calculates the time for a monkey to climb a tree given the tree height,
    hop distance, slip distance, and net climb rate per hour. -/
def monkey_climb_time (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) (net_climb_rate : ℕ) : ℕ :=
  (tree_height - hop_distance) / net_climb_rate + 1

theorem monkey_climb_theorem (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) :
  tree_height = 22 →
  hop_distance = 3 →
  slip_distance = 2 →
  monkey_climb_time tree_height hop_distance slip_distance (hop_distance - slip_distance) = 20 := by
  sorry

#eval monkey_climb_time 22 3 2 1

end NUMINAMATH_CALUDE_monkey_climb_theorem_l581_58116


namespace NUMINAMATH_CALUDE_G_equals_3F_l581_58109

noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

noncomputable def G (x : ℝ) : ℝ := Real.log ((1 + (3 * x + x^3) / (1 + 3 * x^2)) / (1 - (3 * x + x^3) / (1 + 3 * x^2)))

theorem G_equals_3F : ∀ x : ℝ, x ≠ 1 → x ≠ -1 → G x = 3 * F x := by sorry

end NUMINAMATH_CALUDE_G_equals_3F_l581_58109


namespace NUMINAMATH_CALUDE_total_covered_area_is_72_l581_58196

/-- Represents a rectangular strip with length and width -/
structure Strip where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular strip -/
def Strip.area (s : Strip) : ℝ := s.length * s.width

/-- Calculates the area of overlap between two perpendicular strips -/
def overlap_area (s : Strip) : ℝ := s.width * s.width

/-- The setup of the problem with four strips and overlaps -/
structure StripSetup where
  strips : Fin 4 → Strip
  num_overlaps : ℕ

/-- Theorem: The total area covered by four strips with given dimensions and overlaps is 72 -/
theorem total_covered_area_is_72 (setup : StripSetup) 
  (h1 : ∀ i, (setup.strips i).length = 12)
  (h2 : ∀ i, (setup.strips i).width = 2)
  (h3 : setup.num_overlaps = 6) :
  (Finset.sum Finset.univ (λ i => (setup.strips i).area)) - 
  (setup.num_overlaps : ℝ) * overlap_area (setup.strips 0) = 72 := by
  sorry


end NUMINAMATH_CALUDE_total_covered_area_is_72_l581_58196


namespace NUMINAMATH_CALUDE_common_chord_equation_l581_58117

/-- The equation of the line where the common chord of two circles lies -/
theorem common_chord_equation (r : ℝ) (h : r > 0) :
  ∃ (ρ θ : ℝ), (ρ = r ∨ ρ = -2 * r * Real.sin (θ + π/4)) →
  Real.sqrt 2 * ρ * (Real.sin θ + Real.cos θ) = -r :=
sorry

end NUMINAMATH_CALUDE_common_chord_equation_l581_58117


namespace NUMINAMATH_CALUDE_matrix_power_1000_l581_58153

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 0; 2, 1]

theorem matrix_power_1000 : A ^ 1000 = !![1, 0; 2000, 1] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_1000_l581_58153


namespace NUMINAMATH_CALUDE_solution_of_functional_equation_l581_58151

def f (x : ℝ) := x^2 + 2*x - 5

theorem solution_of_functional_equation :
  let s1 := (-1 + Real.sqrt 21) / 2
  let s2 := (-1 - Real.sqrt 21) / 2
  let s3 := (-3 + Real.sqrt 17) / 2
  let s4 := (-3 - Real.sqrt 17) / 2
  (∀ x : ℝ, f (f x) = x ↔ x = s1 ∨ x = s2 ∨ x = s3 ∨ x = s4) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_functional_equation_l581_58151


namespace NUMINAMATH_CALUDE_cos_double_angle_unit_circle_l581_58188

theorem cos_double_angle_unit_circle (α : Real) :
  (Real.cos α = -Real.sqrt 3 / 2 ∧ Real.sin α = 1 / 2) →
  Real.cos (2 * α) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_double_angle_unit_circle_l581_58188


namespace NUMINAMATH_CALUDE_equation_equivalence_l581_58178

/-- An equation is homogeneous if for any solution (x, y), (rx, ry) is also a solution for any non-zero scalar r. -/
def IsHomogeneous (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ (x y r : ℝ), r ≠ 0 → f x y = 0 → f (r * x) (r * y) = 0

/-- The original equation -/
def OriginalEq (x y : ℝ) : ℝ := x^3 - 2*x^2*y + x*y^2 - 2*y^3

/-- The equivalent equation -/
def EquivalentEq (x y : ℝ) : Prop := x = 2*y

theorem equation_equivalence :
  IsHomogeneous OriginalEq →
  (∀ x y : ℝ, OriginalEq x y = 0 ↔ EquivalentEq x y) :=
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l581_58178


namespace NUMINAMATH_CALUDE_product_of_integers_l581_58166

theorem product_of_integers (A B C D : ℕ+) : 
  A + B + C + D = 51 →
  A = 2 * C - 3 →
  B = 2 * C + 3 →
  D = 5 * C + 1 →
  A * B * C * D = 14910 := by
sorry

end NUMINAMATH_CALUDE_product_of_integers_l581_58166


namespace NUMINAMATH_CALUDE_clock_time_l581_58189

/-- Represents a clock with a specific ticking pattern -/
structure Clock where
  ticks_at_hour : ℕ
  time_between_first_and_last : ℝ
  time_at_12 : ℝ

/-- The number of ticks at 12 o'clock -/
def ticks_at_12 : ℕ := 12

theorem clock_time (c : Clock) (h1 : c.ticks_at_hour = 6) 
  (h2 : c.time_between_first_and_last = 25) 
  (h3 : c.time_at_12 = 55) : 
  c.ticks_at_hour = 6 := by sorry

end NUMINAMATH_CALUDE_clock_time_l581_58189
