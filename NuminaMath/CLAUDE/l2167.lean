import Mathlib

namespace NUMINAMATH_CALUDE_inequality_solution_range_l2167_216729

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + 3 > a^2 - 2*a - 1) ↔ (-1 < a ∧ a < 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2167_216729


namespace NUMINAMATH_CALUDE_solution_exists_l2167_216714

theorem solution_exists (N : ℝ) : ∃ x₁ x₂ x₃ x₄ : ℤ, 
  (x₁ > ⌊N⌋) ∧ (x₂ > ⌊N⌋) ∧ (x₃ > ⌊N⌋) ∧ (x₄ > ⌊N⌋) ∧
  (x₁^2 + x₂^2 + x₃^2 + x₄^2 : ℤ) = x₁*x₂*x₃ + x₁*x₂*x₄ + x₁*x₃*x₄ + x₂*x₃*x₄ :=
by
  sorry

end NUMINAMATH_CALUDE_solution_exists_l2167_216714


namespace NUMINAMATH_CALUDE_rectangle_90_42_cut_result_l2167_216711

/-- Represents the dimensions of a rectangle in centimeters -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Represents the result of cutting a rectangle into squares -/
structure CutResult where
  squareCount : ℕ
  totalPerimeter : ℕ

/-- Cuts a rectangle into the maximum number of equal-sized squares -/
def cutIntoSquares (rect : Rectangle) : CutResult :=
  sorry

/-- Theorem stating the correct result for a 90cm × 42cm rectangle -/
theorem rectangle_90_42_cut_result :
  let rect : Rectangle := { length := 90, width := 42 }
  let result : CutResult := cutIntoSquares rect
  result.squareCount = 105 ∧ result.totalPerimeter = 2520 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_90_42_cut_result_l2167_216711


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l2167_216742

theorem absolute_value_simplification : |-4^2 - 6| = 22 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l2167_216742


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2167_216731

theorem cube_root_equation_solution :
  ∃! x : ℝ, (3 - x)^(1/3 : ℝ) = -(3/2) ∧ x = 51/8 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2167_216731


namespace NUMINAMATH_CALUDE_julia_tag_game_l2167_216710

theorem julia_tag_game (total : ℕ) (monday : ℕ) (tuesday : ℕ) 
  (h1 : total = 18) 
  (h2 : monday = 4) 
  (h3 : total = monday + tuesday) : 
  tuesday = 14 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_game_l2167_216710


namespace NUMINAMATH_CALUDE_frustum_volume_l2167_216768

/-- The volume of a frustum with specific conditions --/
theorem frustum_volume (r₁ r₂ : ℝ) (h : ℝ) : 
  r₁ = Real.sqrt 3 →
  r₂ = 3 * Real.sqrt 3 →
  h = 6 →
  (1/3 : ℝ) * (π * r₁^2 + π * r₂^2 + Real.sqrt (π^2 * r₁^2 * r₂^2)) * h = 78 * π := by
  sorry

#check frustum_volume

end NUMINAMATH_CALUDE_frustum_volume_l2167_216768


namespace NUMINAMATH_CALUDE_least_positive_angle_theta_l2167_216746

theorem least_positive_angle_theta (θ : Real) : 
  (θ > 0 ∧ 
   Real.cos (15 * Real.pi / 180) = Real.sin (35 * Real.pi / 180) + Real.sin θ ∧
   ∀ φ, φ > 0 ∧ Real.cos (15 * Real.pi / 180) = Real.sin (35 * Real.pi / 180) + Real.sin φ → θ ≤ φ) →
  θ = 70 * Real.pi / 180 := by
sorry

end NUMINAMATH_CALUDE_least_positive_angle_theta_l2167_216746


namespace NUMINAMATH_CALUDE_ernesto_age_proof_l2167_216771

/-- Ernesto's current age -/
def ernesto_age : ℕ := 11

/-- Jayden's current age -/
def jayden_age : ℕ := 4

/-- The number of years in the future when the age comparison is made -/
def years_future : ℕ := 3

theorem ernesto_age_proof :
  ernesto_age = 11 ∧
  jayden_age = 4 ∧
  jayden_age + years_future = (ernesto_age + years_future) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ernesto_age_proof_l2167_216771


namespace NUMINAMATH_CALUDE_prob_two_or_more_fail_ge_0_9_l2167_216799

/-- The probability of failure for a single device -/
def p : ℝ := 0.2

/-- The probability of success for a single device -/
def q : ℝ := 1 - p

/-- The number of devices to be tested -/
def n : ℕ := 18

/-- The probability of at least two devices failing out of n tested devices -/
def prob_at_least_two_fail (n : ℕ) : ℝ :=
  1 - (q ^ n + n * p * q ^ (n - 1))

/-- Theorem stating that testing 18 devices ensures a probability of at least 0.9 
    that two or more devices will fail -/
theorem prob_two_or_more_fail_ge_0_9 : prob_at_least_two_fail n ≥ 0.9 := by
  sorry


end NUMINAMATH_CALUDE_prob_two_or_more_fail_ge_0_9_l2167_216799


namespace NUMINAMATH_CALUDE_ralph_peanuts_l2167_216775

def initial_peanuts : ℕ := 74
def lost_peanuts : ℕ := 59

theorem ralph_peanuts : initial_peanuts - lost_peanuts = 15 := by
  sorry

end NUMINAMATH_CALUDE_ralph_peanuts_l2167_216775


namespace NUMINAMATH_CALUDE_initial_distance_proof_l2167_216796

/-- The initial distance between Fred and Sam -/
def initial_distance : ℝ := 35

/-- Fred's walking speed in miles per hour -/
def fred_speed : ℝ := 2

/-- Sam's walking speed in miles per hour -/
def sam_speed : ℝ := 5

/-- The distance Sam walks before they meet -/
def sam_distance : ℝ := 25

theorem initial_distance_proof :
  initial_distance = sam_distance + (sam_distance * fred_speed) / sam_speed :=
by sorry

end NUMINAMATH_CALUDE_initial_distance_proof_l2167_216796


namespace NUMINAMATH_CALUDE_max_trailing_zeros_1003_sum_l2167_216792

/-- Given three natural numbers that sum to 1003, the maximum number of trailing zeros in their product is 7. -/
theorem max_trailing_zeros_1003_sum (a b c : ℕ) (h_sum : a + b + c = 1003) :
  ∃ (n : ℕ), n ≤ 7 ∧
  ∀ (m : ℕ), (a * b * c) % (10^m) = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_max_trailing_zeros_1003_sum_l2167_216792


namespace NUMINAMATH_CALUDE_shell_difference_l2167_216778

theorem shell_difference (perfect_shells broken_shells non_spiral_perfect : ℕ) 
  (h1 : perfect_shells = 17)
  (h2 : broken_shells = 52)
  (h3 : non_spiral_perfect = 12) :
  broken_shells / 2 - (perfect_shells - non_spiral_perfect) = 21 := by
  sorry

end NUMINAMATH_CALUDE_shell_difference_l2167_216778


namespace NUMINAMATH_CALUDE_line_through_focus_iff_b_eq_neg_one_l2167_216764

/-- The parabola y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The line y = x + b -/
def line (x y b : ℝ) : Prop := y = x + b

/-- The focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- The line passes through the focus -/
def line_passes_through_focus (b : ℝ) : Prop :=
  line (focus.1) (focus.2) b

theorem line_through_focus_iff_b_eq_neg_one :
  ∀ b : ℝ, line_passes_through_focus b ↔ b = -1 :=
sorry

end NUMINAMATH_CALUDE_line_through_focus_iff_b_eq_neg_one_l2167_216764


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2167_216760

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →  -- Angles are in (0, π)
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Sides are positive
  A + B + C = π →  -- Angle sum in a triangle
  (2*c - a) * Real.cos B = b * Real.cos A →  -- Given equation
  b = 6 →  -- Given condition
  c = 2*a →  -- Given condition
  B = π/3 ∧ (1/2) * a * c * Real.sin B = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2167_216760


namespace NUMINAMATH_CALUDE_gcd_problem_l2167_216797

theorem gcd_problem (b : ℤ) (h : 345 ∣ b) :
  Nat.gcd (5*b^3 + 2*b^2 + 7*b + 69).natAbs b.natAbs = 69 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2167_216797


namespace NUMINAMATH_CALUDE_negative_option_l2167_216716

theorem negative_option : ∀ (x : ℝ), 
  (x = -(-3) ∨ x = -|5| ∨ x = 1/2 ∨ x = 0) → 
  (x < 0 ↔ x = -|5|) := by
sorry

end NUMINAMATH_CALUDE_negative_option_l2167_216716


namespace NUMINAMATH_CALUDE_simplify_fraction_l2167_216720

theorem simplify_fraction : (121 / 9801 : ℚ) * 22 = 22 / 81 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2167_216720


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l2167_216793

/-- An isosceles trapezoid with given base lengths and perpendicular diagonals -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  perpendicular_diagonals : Prop

/-- The area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ := sorry

/-- Theorem: The area of an isosceles trapezoid with bases 40 and 24, 
    and mutually perpendicular diagonals, is 1024 -/
theorem isosceles_trapezoid_area : 
  ∀ (t : IsoscelesTrapezoid), 
  t.base1 = 40 ∧ t.base2 = 24 ∧ t.perpendicular_diagonals → 
  area t = 1024 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l2167_216793


namespace NUMINAMATH_CALUDE_circle_problem_l2167_216745

/-- Given two circles centered at the origin, with P(3,4) on the larger circle,
    S(0,k) on the smaller circle, and QR = 5, prove that k = 0. -/
theorem circle_problem (k : ℝ) : 
  (∃ (r R : ℝ), r ≥ 0 ∧ R ≥ 0 ∧  -- Two circles with non-negative radii
    3^2 + 4^2 = R^2 ∧             -- P(3,4) lies on the larger circle
    0^2 + k^2 = r^2 ∧             -- S(0,k) lies on the smaller circle
    R - r = 5) →                  -- QR = 5
  k = 0 := by
sorry

end NUMINAMATH_CALUDE_circle_problem_l2167_216745


namespace NUMINAMATH_CALUDE_student_count_proof_l2167_216791

def total_students (group1 group2 group3 group4 : ℕ) : ℕ :=
  group1 + group2 + group3 + group4

theorem student_count_proof :
  let group1 : ℕ := 5
  let group2 : ℕ := 8
  let group3 : ℕ := 7
  let group4 : ℕ := 4
  total_students group1 group2 group3 group4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_student_count_proof_l2167_216791


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2167_216737

theorem sum_of_cubes (a b c : ℕ+) :
  (a + b + c)^3 - a^3 - b^3 - c^3 = 294 →
  a + b + c = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2167_216737


namespace NUMINAMATH_CALUDE_angle_increase_in_equilateral_triangle_l2167_216728

/-- 
Given an equilateral triangle where each angle initially measures 60 degrees,
if one angle is increased by 40 degrees, the resulting measure of that angle is 100 degrees.
-/
theorem angle_increase_in_equilateral_triangle :
  ∀ (A B C : ℝ),
  A = 60 ∧ B = 60 ∧ C = 60 →  -- Initially equilateral triangle
  (C + 40 : ℝ) = 100 :=
by sorry

end NUMINAMATH_CALUDE_angle_increase_in_equilateral_triangle_l2167_216728


namespace NUMINAMATH_CALUDE_proportion_solution_l2167_216774

theorem proportion_solution (x : ℝ) : (0.75 / x = 10 / 8) → x = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l2167_216774


namespace NUMINAMATH_CALUDE_min_value_sin_squares_l2167_216782

theorem min_value_sin_squares (α β : Real) 
  (h : -5 * Real.sin α ^ 2 + Real.sin β ^ 2 = 3 * Real.sin α) :
  ∃ (y : Real), y = Real.sin α ^ 2 + Real.sin β ^ 2 ∧ 
  (∀ (z : Real), z = Real.sin α ^ 2 + Real.sin β ^ 2 → y ≤ z) ∧
  y = 0 := by
sorry

end NUMINAMATH_CALUDE_min_value_sin_squares_l2167_216782


namespace NUMINAMATH_CALUDE_average_of_combined_results_l2167_216708

theorem average_of_combined_results (n1 n2 : ℕ) (avg1 avg2 : ℚ) 
  (h1 : n1 = 45) (h2 : n2 = 25) (h3 : avg1 = 25) (h4 : avg2 = 45) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = 2250 / 70 := by
  sorry

end NUMINAMATH_CALUDE_average_of_combined_results_l2167_216708


namespace NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l2167_216767

theorem cryptarithmetic_puzzle (F I V E N : ℕ) : 
  F = 8 → 
  E % 3 = 0 →
  E % 2 = 0 →
  E > 0 →
  E + E ≡ E [ZMOD 10] →
  I + I ≡ N [ZMOD 10] →
  F + F = 10 + N →
  N = 1 →
  (F * 1000 + I * 100 + V * 10 + E) + (F * 1000 + I * 100 + V * 10 + E) = N * 1000 + I * 100 + N * 10 + E →
  I = 5 := by
sorry

end NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l2167_216767


namespace NUMINAMATH_CALUDE_saucers_per_pitcher_l2167_216701

/-- The weight of a cup -/
def cup_weight : ℝ := sorry

/-- The weight of a pitcher -/
def pitcher_weight : ℝ := sorry

/-- The weight of a saucer -/
def saucer_weight : ℝ := sorry

/-- Two cups and two pitchers weigh the same as 14 saucers -/
axiom weight_equation : 2 * cup_weight + 2 * pitcher_weight = 14 * saucer_weight

/-- One pitcher weighs the same as one cup and one saucer -/
axiom pitcher_cup_saucer : pitcher_weight = cup_weight + saucer_weight

/-- The number of saucers that balance with a pitcher is 4 -/
theorem saucers_per_pitcher : pitcher_weight = 4 * saucer_weight := by sorry

end NUMINAMATH_CALUDE_saucers_per_pitcher_l2167_216701


namespace NUMINAMATH_CALUDE_smallest_class_size_l2167_216739

theorem smallest_class_size (b g : ℕ) : 
  b > 0 → g > 0 → 
  (3 * b) % 5 = 0 → 
  (2 * g) % 3 = 0 → 
  3 * b / 5 = 2 * (2 * g / 3) → 
  29 ≤ b + g ∧ 
  (∀ b' g' : ℕ, b' > 0 → g' > 0 → 
    (3 * b') % 5 = 0 → 
    (2 * g') % 3 = 0 → 
    3 * b' / 5 = 2 * (2 * g' / 3) → 
    b' + g' ≥ 29) :=
by sorry

#check smallest_class_size

end NUMINAMATH_CALUDE_smallest_class_size_l2167_216739


namespace NUMINAMATH_CALUDE_incandescent_bulbs_count_l2167_216777

/-- Represents the waterfall system and power consumption --/
structure WaterfallSystem where
  water_flow : ℝ  -- m³/s
  waterfall_height : ℝ  -- m
  turbine_efficiency : ℝ
  dynamo_efficiency : ℝ
  transmission_efficiency : ℝ
  num_motors : ℕ
  power_per_motor : ℝ  -- horsepower
  motor_efficiency : ℝ
  num_arc_lamps : ℕ
  arc_lamp_voltage : ℝ  -- V
  arc_lamp_current : ℝ  -- A
  incandescent_bulb_power : ℝ  -- W

/-- Calculates the number of incandescent bulbs that can be powered --/
def calculate_incandescent_bulbs (system : WaterfallSystem) : ℕ :=
  sorry

/-- Theorem stating the number of incandescent bulbs that can be powered --/
theorem incandescent_bulbs_count (system : WaterfallSystem) 
  (h1 : system.water_flow = 8)
  (h2 : system.waterfall_height = 5)
  (h3 : system.turbine_efficiency = 0.8)
  (h4 : system.dynamo_efficiency = 0.9)
  (h5 : system.transmission_efficiency = 0.95)
  (h6 : system.num_motors = 5)
  (h7 : system.power_per_motor = 10)
  (h8 : system.motor_efficiency = 0.85)
  (h9 : system.num_arc_lamps = 24)
  (h10 : system.arc_lamp_voltage = 40)
  (h11 : system.arc_lamp_current = 10)
  (h12 : system.incandescent_bulb_power = 55) :
  calculate_incandescent_bulbs system = 3920 :=
sorry

end NUMINAMATH_CALUDE_incandescent_bulbs_count_l2167_216777


namespace NUMINAMATH_CALUDE_highest_score_is_143_l2167_216784

/-- Represents a batsman's performance in a cricket tournament --/
structure BatsmanPerformance where
  totalInnings : ℕ
  averageRuns : ℚ
  highestScore : ℕ
  lowestScore : ℕ
  centuryCount : ℕ

/-- Theorem stating the highest score of the batsman given the conditions --/
theorem highest_score_is_143 (b : BatsmanPerformance) : 
  b.totalInnings = 46 ∧
  b.averageRuns = 58 ∧
  b.highestScore - b.lowestScore = 150 ∧
  (b.totalInnings * b.averageRuns - b.highestScore - b.lowestScore) / (b.totalInnings - 2) = b.averageRuns ∧
  b.centuryCount = 5 ∧
  ∀ score, score ≠ b.highestScore → score < 100 →
  b.highestScore = 143 := by
  sorry


end NUMINAMATH_CALUDE_highest_score_is_143_l2167_216784


namespace NUMINAMATH_CALUDE_division_remainder_l2167_216755

theorem division_remainder : ∃ (A B : ℕ), 26 = 4 * A + B ∧ B < 4 ∧ B = 2 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l2167_216755


namespace NUMINAMATH_CALUDE_sin_90_plus_alpha_eq_neg_half_l2167_216738

/-- Given that α is an angle in the second quadrant and tan α = -√3, prove that sin(90° + α) = -1/2 -/
theorem sin_90_plus_alpha_eq_neg_half (α : Real) 
  (h1 : π/2 < α ∧ α < π)  -- α is in the second quadrant
  (h2 : Real.tan α = -Real.sqrt 3) : -- tan α = -√3
  Real.sin (π/2 + α) = -1/2 := by sorry

end NUMINAMATH_CALUDE_sin_90_plus_alpha_eq_neg_half_l2167_216738


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l2167_216736

/-- Given a line with slope -8 passing through (4, -3), prove that m + b = 21 in y = mx + b -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = -8 → 
  -3 = m * 4 + b → 
  m + b = 21 := by
sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l2167_216736


namespace NUMINAMATH_CALUDE_second_derivative_of_f_l2167_216762

/-- Given a function f(x) = α² - cos x, prove that its second derivative at α is sin α -/
theorem second_derivative_of_f (α : ℝ) : 
  let f : ℝ → ℝ := λ x => α^2 - Real.cos x
  (deriv (deriv f)) α = Real.sin α := by sorry

end NUMINAMATH_CALUDE_second_derivative_of_f_l2167_216762


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2167_216757

theorem perfect_square_trinomial (a b : ℝ) : 9*a^2 - 24*a*b + 16*b^2 = (3*a + 4*b)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2167_216757


namespace NUMINAMATH_CALUDE_agency_A_more_cost_effective_l2167_216759

/-- Represents the cost calculation for travel agencies A and B -/
def travel_cost (num_students : ℕ) : ℚ × ℚ :=
  let full_price : ℚ := 40
  let num_parents : ℕ := 10
  let cost_A : ℚ := full_price * num_parents.cast + (full_price / 2) * num_students.cast
  let cost_B : ℚ := full_price * (1 - 0.4) * (num_parents + num_students).cast
  (cost_A, cost_B)

/-- Theorem stating when travel agency A is more cost-effective -/
theorem agency_A_more_cost_effective (num_students : ℕ) :
  num_students > 40 → (travel_cost num_students).1 < (travel_cost num_students).2 := by
  sorry

#check agency_A_more_cost_effective

end NUMINAMATH_CALUDE_agency_A_more_cost_effective_l2167_216759


namespace NUMINAMATH_CALUDE_luke_coin_count_l2167_216783

def total_coins (quarter_piles dime_piles coins_per_pile : ℕ) : ℕ :=
  (quarter_piles + dime_piles) * coins_per_pile

theorem luke_coin_count :
  let quarter_piles : ℕ := 5
  let dime_piles : ℕ := 5
  let coins_per_pile : ℕ := 3
  total_coins quarter_piles dime_piles coins_per_pile = 30 := by
  sorry

end NUMINAMATH_CALUDE_luke_coin_count_l2167_216783


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l2167_216706

theorem min_value_of_sum_of_squares (x y : ℝ) (h : 4 * x^2 + 4 * x * y + 7 * y^2 = 3) :
  ∃ (m : ℝ), m = 3/8 ∧ x^2 + y^2 ≥ m ∧ ∃ (x₀ y₀ : ℝ), 4 * x₀^2 + 4 * x₀ * y₀ + 7 * y₀^2 = 3 ∧ x₀^2 + y₀^2 = m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l2167_216706


namespace NUMINAMATH_CALUDE_range_of_a_l2167_216770

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 / (a * Real.log (x + 1))

/-- The function g(x) defined in the problem -/
def g (x : ℝ) : ℝ := x^2 * (x - 1)^2

/-- The helper function h(x) used in the proof -/
noncomputable def h (x : ℝ) : ℝ := x^2 / Real.log x

theorem range_of_a :
  ∃ (a : ℝ), ∀ (x₁ x₂ : ℝ),
    (Real.exp (1/4) - 1 < x₁ ∧ x₁ < Real.exp 1 - 1) →
    x₂ < 0 →
    x₂ = -x₁ →
    (f a x₁) * (-x₁) + (g x₂) * x₁ = 0 →
    2 * Real.exp 1 ≤ a ∧ a < Real.exp 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2167_216770


namespace NUMINAMATH_CALUDE_left_handed_or_throwers_count_l2167_216756

/-- Represents a football team with specific player distributions -/
structure FootballTeam where
  total_players : Nat
  throwers : Nat
  left_handed : Nat
  right_handed : Nat

/-- Calculates the number of players who are either left-handed or throwers -/
def left_handed_or_throwers (team : FootballTeam) : Nat :=
  team.left_handed + team.throwers

/-- Theorem stating the number of players who are either left-handed or throwers in the given scenario -/
theorem left_handed_or_throwers_count (team : FootballTeam) :
  team.total_players = 70 →
  team.throwers = 34 →
  team.left_handed = (team.total_players - team.throwers) / 3 →
  team.right_handed = team.total_players - team.throwers - team.left_handed + team.throwers →
  left_handed_or_throwers team = 46 := by
  sorry

end NUMINAMATH_CALUDE_left_handed_or_throwers_count_l2167_216756


namespace NUMINAMATH_CALUDE_first_year_interest_rate_l2167_216727

/-- Given an initial amount, time period, interest rates, and final amount,
    calculate the interest rate for the first year. -/
theorem first_year_interest_rate
  (initial_amount : ℝ)
  (time_period : ℕ)
  (second_year_rate : ℝ)
  (final_amount : ℝ)
  (h1 : initial_amount = 5000)
  (h2 : time_period = 2)
  (h3 : second_year_rate = 0.25)
  (h4 : final_amount = 7500) :
  ∃ (first_year_rate : ℝ),
    first_year_rate = 0.20 ∧
    final_amount = initial_amount * (1 + first_year_rate) * (1 + second_year_rate) :=
by sorry

end NUMINAMATH_CALUDE_first_year_interest_rate_l2167_216727


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l2167_216741

/-- A rectangular plot with length thrice its breadth and area 675 sq m has a breadth of 15 m -/
theorem rectangular_plot_breadth : 
  ∀ (length breadth : ℝ),
  length = 3 * breadth →
  length * breadth = 675 →
  breadth = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_rectangular_plot_breadth_l2167_216741


namespace NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_neg_seven_l2167_216758

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The first line l1 -/
def l1 (m : ℝ) : Line :=
  { a := 3 + m, b := 4, c := 5 - 3*m }

/-- The second line l2 -/
def l2 (m : ℝ) : Line :=
  { a := 2, b := 5 + m, c := 8 }

/-- The theorem stating that l1 and l2 are parallel iff m = -7 -/
theorem lines_parallel_iff_m_eq_neg_seven :
  ∀ m : ℝ, parallel (l1 m) (l2 m) ↔ m = -7 :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_neg_seven_l2167_216758


namespace NUMINAMATH_CALUDE_square_sum_diff_product_l2167_216754

theorem square_sum_diff_product (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ((a + b)^2 - (a - b)^2)^2 / (a * b)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_diff_product_l2167_216754


namespace NUMINAMATH_CALUDE_harmonio_theorem_l2167_216704

/-- Represents the student population at Harmonio Middle School -/
structure School where
  total : ℝ
  enjoy_singing : ℝ
  admit_liking : ℝ
  dislike_consistent : ℝ

/-- Conditions for Harmonio Middle School -/
def harmonio_conditions (s : School) : Prop :=
  s.total > 0 ∧
  s.enjoy_singing = 0.7 * s.total ∧
  s.admit_liking = 0.75 * s.enjoy_singing ∧
  s.dislike_consistent = 0.8 * (s.total - s.enjoy_singing)

/-- Theorem statement for the problem -/
theorem harmonio_theorem (s : School) (h : harmonio_conditions s) :
  let claim_dislike := s.dislike_consistent + (s.enjoy_singing - s.admit_liking)
  (s.enjoy_singing - s.admit_liking) / claim_dislike = 0.4217 := by
  sorry


end NUMINAMATH_CALUDE_harmonio_theorem_l2167_216704


namespace NUMINAMATH_CALUDE_cubic_equation_root_l2167_216780

theorem cubic_equation_root : ∃ x : ℝ, x^3 + 6*x^2 + 12*x + 35 = 0 :=
  by
    use -5
    -- Proof goes here
    sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l2167_216780


namespace NUMINAMATH_CALUDE_verandah_area_is_124_l2167_216786

/-- Calculates the area of a verandah surrounding a rectangular room. -/
def verandahArea (roomLength : ℝ) (roomWidth : ℝ) (verandahWidth : ℝ) : ℝ :=
  (roomLength + 2 * verandahWidth) * (roomWidth + 2 * verandahWidth) - roomLength * roomWidth

/-- Theorem: The area of the verandah is 124 square meters. -/
theorem verandah_area_is_124 :
  verandahArea 15 12 2 = 124 := by
  sorry

#eval verandahArea 15 12 2

end NUMINAMATH_CALUDE_verandah_area_is_124_l2167_216786


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l2167_216750

theorem complex_magnitude_equation (m : ℝ) (h : m > 0) : 
  Complex.abs (5 + m * Complex.I) = 5 * Real.sqrt 26 → m = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l2167_216750


namespace NUMINAMATH_CALUDE_max_leap_years_in_period_l2167_216702

/-- Represents the number of years in a period -/
def period : ℕ := 200

/-- Represents the frequency of leap years -/
def leapYearFrequency : ℕ := 5

/-- Calculates the maximum number of leap years in the given period -/
def maxLeapYears : ℕ := period / leapYearFrequency

/-- Theorem: The maximum number of leap years in a 200-year period
    with leap years occurring every five years is 40 -/
theorem max_leap_years_in_period :
  maxLeapYears = 40 := by sorry

end NUMINAMATH_CALUDE_max_leap_years_in_period_l2167_216702


namespace NUMINAMATH_CALUDE_gym_monthly_income_l2167_216700

-- Define the gym's charging structure
def twice_monthly_charge : ℕ := 18

-- Define the number of members
def number_of_members : ℕ := 300

-- Define the monthly income
def monthly_income : ℕ := twice_monthly_charge * 2 * number_of_members

-- Theorem statement
theorem gym_monthly_income :
  monthly_income = 10800 :=
by sorry

end NUMINAMATH_CALUDE_gym_monthly_income_l2167_216700


namespace NUMINAMATH_CALUDE_a_over_two_plus_a_is_fraction_l2167_216749

/-- Definition of a fraction -/
def is_fraction (x y : ℝ) : Prop := ∃ (a b : ℝ), x = a ∧ y = b ∧ b ≠ 0

/-- The expression a / (2 + a) is a fraction -/
theorem a_over_two_plus_a_is_fraction (a : ℝ) : is_fraction a (2 + a) := by
  sorry

end NUMINAMATH_CALUDE_a_over_two_plus_a_is_fraction_l2167_216749


namespace NUMINAMATH_CALUDE_john_walks_to_school_l2167_216748

/-- The distance Nina walks to school in miles -/
def nina_distance : ℝ := 0.4

/-- The additional distance John walks compared to Nina in miles -/
def additional_distance : ℝ := 0.3

/-- John's distance to school in miles -/
def john_distance : ℝ := nina_distance + additional_distance

/-- Theorem stating that John walks 0.7 miles to school -/
theorem john_walks_to_school : john_distance = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_john_walks_to_school_l2167_216748


namespace NUMINAMATH_CALUDE_factorial_quotient_l2167_216790

theorem factorial_quotient : Nat.factorial 50 / Nat.factorial 47 = 117600 := by
  sorry

end NUMINAMATH_CALUDE_factorial_quotient_l2167_216790


namespace NUMINAMATH_CALUDE_cubic_function_extrema_difference_l2167_216798

/-- A cubic function with specific properties -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x + c

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*b

theorem cubic_function_extrema_difference (a b c : ℝ) :
  (f' a b 2 = 0) →  -- Extremum at x = 2
  (f' a b 1 = -3) →  -- Tangent at x = 1 is parallel to 6x + 2y + 5 = 0
  ∃ (x_max x_min : ℝ), 
    (∀ x, f a b c x ≤ f a b c x_max) ∧
    (∀ x, f a b c x_min ≤ f a b c x) ∧
    (f a b c x_max - f a b c x_min = 4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_difference_l2167_216798


namespace NUMINAMATH_CALUDE_point_in_intersection_l2167_216753

def U : Set (ℝ × ℝ) := Set.univ

def A (m : ℝ) : Set (ℝ × ℝ) := U \ {p | p.1 + p.2 > m}

def B (n : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ n}

def C_U_A (m : ℝ) : Set (ℝ × ℝ) := {p | p.1 + p.2 ≤ m}

theorem point_in_intersection (m n : ℝ) :
  (1, 2) ∈ (C_U_A m ∩ B n) ↔ m ≥ 3 ∧ n ≥ 5 := by sorry

end NUMINAMATH_CALUDE_point_in_intersection_l2167_216753


namespace NUMINAMATH_CALUDE_min_area_bounded_by_curve_and_lines_l2167_216735

noncomputable section

-- Define the curve C
def f (x : ℝ) : ℝ := 1 / (1 + x^2)

-- Define the area function T(α)
def T (α : ℝ) : ℝ :=
  Real.arctan α + Real.arctan (1 / α) - α / (1 + α^2)

-- Theorem statement
theorem min_area_bounded_by_curve_and_lines (α : ℝ) (h : α > 0) :
  ∃ (min_area : ℝ), min_area = π / 2 - 1 / 2 ∧
  ∀ β > 0, T β ≥ min_area :=
sorry

end NUMINAMATH_CALUDE_min_area_bounded_by_curve_and_lines_l2167_216735


namespace NUMINAMATH_CALUDE_lisa_candy_weeks_l2167_216722

/-- The number of weeks it takes Lisa to eat all her candies -/
def weeks_to_eat_candies (total_candies : ℕ) (candies_mon_wed : ℕ) (candies_other_days : ℕ) : ℕ :=
  total_candies / (2 * candies_mon_wed + 5 * candies_other_days)

/-- Theorem stating that it takes 4 weeks for Lisa to eat all her candies -/
theorem lisa_candy_weeks : weeks_to_eat_candies 36 2 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_lisa_candy_weeks_l2167_216722


namespace NUMINAMATH_CALUDE_set_B_equivalence_l2167_216761

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 - a*x + b

-- Define set A
def A (a b : ℝ) : Set ℝ := {x | f a b x - x = 0}

-- Define set B
def B (a b : ℝ) : Set ℝ := {x | f a b x - a*x = 0}

-- State the theorem
theorem set_B_equivalence (a b : ℝ) : 
  A a b = {1, -3} → B a b = {-2 - Real.sqrt 7, -2 + Real.sqrt 7} := by
  sorry

end NUMINAMATH_CALUDE_set_B_equivalence_l2167_216761


namespace NUMINAMATH_CALUDE_stanleys_distance_difference_l2167_216769

theorem stanleys_distance_difference (run_distance walk_distance : ℝ) : 
  run_distance = 0.4 → walk_distance = 0.2 → run_distance - walk_distance = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_stanleys_distance_difference_l2167_216769


namespace NUMINAMATH_CALUDE_largest_three_digit_divisible_by_two_digit_parts_l2167_216763

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def AB (n : ℕ) : ℕ := n / 10

def BC (n : ℕ) : ℕ := n % 100

theorem largest_three_digit_divisible_by_two_digit_parts :
  ∀ n : ℕ,
    is_three_digit n →
    is_two_digit (AB n) →
    is_two_digit (BC n) →
    n % (AB n) = 0 →
    n % (BC n) = 0 →
    n ≤ 990 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_divisible_by_two_digit_parts_l2167_216763


namespace NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l2167_216765

/-- Given that the total marks in physics, chemistry, and mathematics is 130 more than
    the marks in physics, prove that the average mark in chemistry and mathematics is 65. -/
theorem average_marks_chemistry_mathematics (P C M : ℕ) : 
  P + C + M = P + 130 → (C + M) / 2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l2167_216765


namespace NUMINAMATH_CALUDE_boys_walking_speed_l2167_216734

/-- 
Given two boys walking in the same direction for 7 hours, with one boy walking at 5.5 km/h 
and ending up 10.5 km apart, prove that the speed of the other boy is 7 km/h.
-/
theorem boys_walking_speed 
  (time : ℝ) 
  (distance_apart : ℝ) 
  (speed_second_boy : ℝ) 
  (speed_first_boy : ℝ) 
  (h1 : time = 7) 
  (h2 : distance_apart = 10.5) 
  (h3 : speed_second_boy = 5.5) 
  (h4 : distance_apart = (speed_first_boy - speed_second_boy) * time) : 
  speed_first_boy = 7 := by
  sorry

end NUMINAMATH_CALUDE_boys_walking_speed_l2167_216734


namespace NUMINAMATH_CALUDE_expression_evaluation_l2167_216713

theorem expression_evaluation :
  let a : ℚ := -2
  let b : ℚ := 1/2
  (2*a + b) * (2*a - b) + (3*a - b)^2 - ((12*a*b^2 - 16*a^2*b + 4*b) / (2*b)) = 104 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2167_216713


namespace NUMINAMATH_CALUDE_linear_equation_implies_m_eq_neg_three_l2167_216776

/-- Given that the equation (|m|-3)x^2 + (-m+3)x - 4 = 0 is linear in x with respect to m, prove that m = -3 -/
theorem linear_equation_implies_m_eq_neg_three (m : ℝ) 
  (h1 : ∀ x, (|m| - 3) * x^2 + (-m + 3) * x - 4 = 0 → (|m| - 3 = 0 ∧ -m + 3 ≠ 0)) : 
  m = -3 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_implies_m_eq_neg_three_l2167_216776


namespace NUMINAMATH_CALUDE_mountain_dew_to_coke_ratio_l2167_216773

/-- Represents the composition of a drink -/
structure DrinkComposition where
  coke : ℝ
  sprite : ℝ
  mountainDew : ℝ

/-- Proves that the ratio of Mountain Dew to Coke is 3:2 given the conditions -/
theorem mountain_dew_to_coke_ratio 
  (drink : DrinkComposition)
  (coke_sprite_ratio : drink.coke = 2 * drink.sprite)
  (coke_amount : drink.coke = 6)
  (total_amount : drink.coke + drink.sprite + drink.mountainDew = 18) :
  drink.mountainDew / drink.coke = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_mountain_dew_to_coke_ratio_l2167_216773


namespace NUMINAMATH_CALUDE_smallest_n_sum_squares_over_n_is_square_l2167_216703

/-- Sum of squares from 1 to n -/
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- Predicate to check if a number is a perfect square -/
def is_square (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

/-- Predicate to check if the sum of squares divided by n is a square -/
def is_sum_of_squares_over_n_square (n : ℕ) : Prop :=
  is_square (sum_of_squares n / n)

theorem smallest_n_sum_squares_over_n_is_square :
  (∀ m : ℕ, m > 1 ∧ m < 337 → ¬is_sum_of_squares_over_n_square m) ∧
  is_sum_of_squares_over_n_square 337 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_sum_squares_over_n_is_square_l2167_216703


namespace NUMINAMATH_CALUDE_hannah_dog_food_l2167_216789

/-- The amount of dog food Hannah needs to prepare for her three dogs in a day -/
def total_dog_food (first_dog_food : ℝ) (second_dog_multiplier : ℝ) (third_dog_extra : ℝ) : ℝ :=
  first_dog_food + 
  (first_dog_food * second_dog_multiplier) + 
  (first_dog_food * second_dog_multiplier + third_dog_extra)

/-- Theorem stating that Hannah needs to prepare 10 cups of dog food for her three dogs -/
theorem hannah_dog_food : total_dog_food 1.5 2 2.5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_hannah_dog_food_l2167_216789


namespace NUMINAMATH_CALUDE_modulus_of_complex_l2167_216733

def i : ℂ := Complex.I

theorem modulus_of_complex (z : ℂ) : z = (2 + i) / (1 - i) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_l2167_216733


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2167_216726

theorem train_speed_calculation (v : ℝ) : 
  v > 0 → -- The speed is positive
  (v + 42) * (5 / 18) * 9 = 280 → -- Equation derived from the problem
  v = 70 := by
sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2167_216726


namespace NUMINAMATH_CALUDE_bicycle_usage_theorem_l2167_216795

/-- Represents the number of times a student used a shared bicycle -/
inductive Usage : Type
  | zero | one | two | three | four | five

/-- Represents the data of student bicycle usage -/
structure UsageData where
  usage : Usage
  count : Nat

def sample : List UsageData := [
  ⟨Usage.zero, 22⟩,
  ⟨Usage.one, 14⟩,
  ⟨Usage.two, 24⟩,
  ⟨Usage.three, 27⟩,
  ⟨Usage.four, 8⟩,
  ⟨Usage.five, 5⟩
]

def totalStudents : Nat := (sample.map UsageData.count).sum

def median (data : List UsageData) : Nat := sorry

def mode (data : List UsageData) : Nat := sorry

def average (data : List UsageData) : Rat := sorry

def estimateUsage (data : List UsageData) (totalPopulation : Nat) : Nat := sorry

theorem bicycle_usage_theorem :
  median sample = 2 ∧
  mode sample = 3 ∧
  average sample = 2 ∧
  estimateUsage sample 1500 = 600 := by sorry

end NUMINAMATH_CALUDE_bicycle_usage_theorem_l2167_216795


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2167_216721

def complexI : ℂ := Complex.I

theorem imaginary_part_of_z (z : ℂ) (h : z / (1 + complexI) = 2 - 3 * complexI) :
  z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2167_216721


namespace NUMINAMATH_CALUDE_integer_solutions_count_l2167_216772

theorem integer_solutions_count (m : ℤ) : 
  (∃! (s : Finset ℤ), s.card = 2 ∧ ∀ x ∈ s, x - m < 0 ∧ 5 - 2*x ≤ 1) ↔ m = 4 := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_count_l2167_216772


namespace NUMINAMATH_CALUDE_train_overtake_time_l2167_216730

/-- The time taken for a train to overtake a motorbike -/
theorem train_overtake_time (train_speed motorbike_speed : ℝ) (train_length : ℝ) :
  train_speed = 100 →
  motorbike_speed = 64 →
  train_length = 400.032 →
  (train_length / ((train_speed - motorbike_speed) * (1000 / 3600))) = 40.0032 := by
  sorry

end NUMINAMATH_CALUDE_train_overtake_time_l2167_216730


namespace NUMINAMATH_CALUDE_equation_solution_l2167_216794

theorem equation_solution : ∃ x : ℚ, (1 / 4 : ℚ) + 1 / x = 7 / 8 ∧ x = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2167_216794


namespace NUMINAMATH_CALUDE_power_division_equality_l2167_216788

theorem power_division_equality (a : ℝ) : (-2 * a^2)^3 / (2 * a^2) = -4 * a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equality_l2167_216788


namespace NUMINAMATH_CALUDE_luke_fish_catch_l2167_216752

theorem luke_fish_catch (days : ℕ) (fillets_per_fish : ℕ) (total_fillets : ℕ) 
  (h1 : days = 30)
  (h2 : fillets_per_fish = 2)
  (h3 : total_fillets = 120) :
  total_fillets / (days * fillets_per_fish) = 2 :=
by sorry

end NUMINAMATH_CALUDE_luke_fish_catch_l2167_216752


namespace NUMINAMATH_CALUDE_cookie_calorie_count_l2167_216715

/-- The number of calories in each cracker -/
def cracker_calories : ℕ := 15

/-- The number of cookies Jimmy eats -/
def cookies_eaten : ℕ := 7

/-- The number of crackers Jimmy eats -/
def crackers_eaten : ℕ := 10

/-- The total number of calories Jimmy consumes -/
def total_calories : ℕ := 500

/-- The number of calories in each cookie -/
def cookie_calories : ℕ := 50

theorem cookie_calorie_count :
  cookie_calories * cookies_eaten + cracker_calories * crackers_eaten = total_calories :=
by sorry

end NUMINAMATH_CALUDE_cookie_calorie_count_l2167_216715


namespace NUMINAMATH_CALUDE_cos_72_minus_cos_144_l2167_216719

theorem cos_72_minus_cos_144 : Real.cos (72 * π / 180) - Real.cos (144 * π / 180) = 1.117962 := by
  sorry

end NUMINAMATH_CALUDE_cos_72_minus_cos_144_l2167_216719


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2167_216709

theorem negation_of_proposition (a b : ℝ) :
  ¬(a + b = 1 → a^2 + b^2 ≥ 1/2) ↔ (a + b ≠ 1 → a^2 + b^2 < 1/2) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2167_216709


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l2167_216724

theorem quadratic_root_implies_m_value (m : ℝ) :
  (3^2 - 2*3 + m = 0) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l2167_216724


namespace NUMINAMATH_CALUDE_chess_pieces_present_l2167_216787

/-- The number of pieces in a standard chess set -/
def standard_chess_pieces : ℕ := 32

/-- The number of missing chess pieces -/
def missing_pieces : ℕ := 8

/-- Theorem: The number of chess pieces present is 24 -/
theorem chess_pieces_present : 
  standard_chess_pieces - missing_pieces = 24 := by
  sorry

end NUMINAMATH_CALUDE_chess_pieces_present_l2167_216787


namespace NUMINAMATH_CALUDE_second_stop_count_l2167_216723

/-- The number of students who got on the bus during the first stop -/
def first_stop_students : ℕ := 39

/-- The total number of students on the bus after the second stop -/
def total_students : ℕ := 68

/-- The number of students who got on the bus during the second stop -/
def second_stop_students : ℕ := total_students - first_stop_students

theorem second_stop_count : second_stop_students = 29 := by
  sorry

end NUMINAMATH_CALUDE_second_stop_count_l2167_216723


namespace NUMINAMATH_CALUDE_am_gm_inequality_and_equality_condition_l2167_216705

theorem am_gm_inequality_and_equality_condition (x : ℝ) (h : x > 0) :
  x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by sorry

end NUMINAMATH_CALUDE_am_gm_inequality_and_equality_condition_l2167_216705


namespace NUMINAMATH_CALUDE_fifteen_is_zero_l2167_216740

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ (∀ x, f (x + 2) = -f x)

/-- Theorem stating that any function satisfying the conditions has f(15) = 0 -/
theorem fifteen_is_zero (f : ℝ → ℝ) (h : special_function f) : f 15 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_is_zero_l2167_216740


namespace NUMINAMATH_CALUDE_intersection_A_B_l2167_216766

def A : Set ℤ := {1, 2}
def B : Set ℤ := {x : ℤ | |x| < 2}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2167_216766


namespace NUMINAMATH_CALUDE_chess_pawns_remaining_l2167_216744

theorem chess_pawns_remaining (initial_pawns : ℕ) 
  (sophia_lost : ℕ) (chloe_lost : ℕ) : 
  initial_pawns = 8 → 
  sophia_lost = 5 → 
  chloe_lost = 1 → 
  (initial_pawns - sophia_lost) + (initial_pawns - chloe_lost) = 10 := by
  sorry

end NUMINAMATH_CALUDE_chess_pawns_remaining_l2167_216744


namespace NUMINAMATH_CALUDE_decimal_between_996_998_l2167_216785

theorem decimal_between_996_998 :
  ∃ x y : ℝ, x ≠ y ∧ 0.996 < x ∧ x < 0.998 ∧ 0.996 < y ∧ y < 0.998 :=
sorry

end NUMINAMATH_CALUDE_decimal_between_996_998_l2167_216785


namespace NUMINAMATH_CALUDE_salary_percentage_is_120_percent_l2167_216779

/-- The percentage of one employee's salary compared to another -/
def salary_percentage (total_salary n_salary : ℚ) : ℚ :=
  ((total_salary - n_salary) / n_salary) * 100

/-- Proof that the salary percentage is 120% given the conditions -/
theorem salary_percentage_is_120_percent 
  (total_salary : ℚ) 
  (n_salary : ℚ) 
  (h1 : total_salary = 594) 
  (h2 : n_salary = 270) : 
  salary_percentage total_salary n_salary = 120 := by
  sorry

end NUMINAMATH_CALUDE_salary_percentage_is_120_percent_l2167_216779


namespace NUMINAMATH_CALUDE_hyperbola_theorem_l2167_216781

def hyperbola (a b h k : ℝ) (x y : ℝ) : Prop :=
  (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1

def asymptote1 (x y : ℝ) : Prop := y = 3 * x + 6
def asymptote2 (x y : ℝ) : Prop := y = -3 * x + 2

theorem hyperbola_theorem (a b h k : ℝ) :
  a > 0 → b > 0 →
  (∀ x y, asymptote1 x y ∨ asymptote2 x y → hyperbola a b h k x y) →
  hyperbola a b h k 1 5 →
  a + h = 6 * Real.sqrt 2 - 2/3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_theorem_l2167_216781


namespace NUMINAMATH_CALUDE_sum_x_2y_equals_5_l2167_216751

theorem sum_x_2y_equals_5 (x y : ℕ+) 
  (h : x^3 + 3*x^2*y + 8*x*y^2 + 6*y^3 = 87) : 
  x + 2*y = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_2y_equals_5_l2167_216751


namespace NUMINAMATH_CALUDE_equation_solution_l2167_216718

theorem equation_solution : ∃ n : ℝ, (1 / (n + 2) + 2 / (n + 2) + n / (n + 2) = 2) ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2167_216718


namespace NUMINAMATH_CALUDE_at_op_difference_l2167_216717

/-- Definition of the @ operation -/
def at_op (x y : ℤ) : ℤ := 3 * x * y - 2 * x + y

/-- Theorem stating that (6@4) - (4@6) = -6 -/
theorem at_op_difference : at_op 6 4 - at_op 4 6 = -6 := by
  sorry

end NUMINAMATH_CALUDE_at_op_difference_l2167_216717


namespace NUMINAMATH_CALUDE_f_range_upper_bound_l2167_216712

def f (x : ℝ) : ℝ := -x^2 + 2*x

theorem f_range_upper_bound (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 2, a < f x) → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_f_range_upper_bound_l2167_216712


namespace NUMINAMATH_CALUDE_sin_2x_value_l2167_216707

theorem sin_2x_value (x : Real) (h : Real.sin (π / 4 - x) = 3 / 5) : 
  Real.sin (2 * x) = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_value_l2167_216707


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l2167_216732

/-- Given vectors a and b, prove that the magnitude of a - 2b is √3 -/
theorem vector_magnitude_proof (a b : ℝ × ℝ) :
  a.1 = Real.cos (15 * π / 180) ∧
  a.2 = Real.sin (15 * π / 180) ∧
  b.1 = Real.cos (75 * π / 180) ∧
  b.2 = Real.sin (75 * π / 180) →
  Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l2167_216732


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2167_216743

theorem functional_equation_solution (f g : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y * g x) = g x + x * f y) →
  (f = id ∧ g = id) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2167_216743


namespace NUMINAMATH_CALUDE_greatest_base9_digit_sum_l2167_216725

/-- Represents a positive integer in base 9 --/
structure Base9Int where
  digits : List Nat
  positive : digits ≠ []
  valid : ∀ d ∈ digits, d < 9

/-- Converts a Base9Int to its decimal (base 10) representation --/
def toDecimal (n : Base9Int) : Nat :=
  n.digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- Calculates the sum of digits of a Base9Int --/
def digitSum (n : Base9Int) : Nat :=
  n.digits.sum

/-- The main theorem to be proved --/
theorem greatest_base9_digit_sum :
  ∃ (max : Nat), 
    (∀ (n : Base9Int), toDecimal n < 3000 → digitSum n ≤ max) ∧ 
    (∃ (n : Base9Int), toDecimal n < 3000 ∧ digitSum n = max) ∧
    max = 24 := by
  sorry

end NUMINAMATH_CALUDE_greatest_base9_digit_sum_l2167_216725


namespace NUMINAMATH_CALUDE_vector_magnitude_l2167_216747

/-- Given vectors a and b in ℝ², where a · b = 0, prove that |b| = √5 -/
theorem vector_magnitude (a b : ℝ × ℝ) (h : a.1 * b.1 + a.2 * b.2 = 0) 
  (ha : a = (1, 2)) (hb : b.1 = 2) : Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 5 := by
  sorry

#check vector_magnitude

end NUMINAMATH_CALUDE_vector_magnitude_l2167_216747
