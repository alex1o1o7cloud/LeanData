import Mathlib

namespace NUMINAMATH_CALUDE_park_trees_l3082_308297

/-- The number of trees in a rectangular park -/
def num_trees (length width tree_density : ℕ) : ℕ :=
  (length * width) / tree_density

/-- Proof that a park with given dimensions and tree density has 100,000 trees -/
theorem park_trees : num_trees 1000 2000 20 = 100000 := by
  sorry

end NUMINAMATH_CALUDE_park_trees_l3082_308297


namespace NUMINAMATH_CALUDE_set_difference_empty_implies_subset_l3082_308275

theorem set_difference_empty_implies_subset (A B : Set α) : 
  (A \ B = ∅) → (A ⊆ B) := by
  sorry

end NUMINAMATH_CALUDE_set_difference_empty_implies_subset_l3082_308275


namespace NUMINAMATH_CALUDE_soccer_club_girls_l3082_308259

theorem soccer_club_girls (total : ℕ) (attended : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 30 →
  attended = 18 →
  boys + girls = total →
  boys + (girls / 3) = attended →
  boys + girls = total →
  girls = 18 := by
sorry

end NUMINAMATH_CALUDE_soccer_club_girls_l3082_308259


namespace NUMINAMATH_CALUDE_wheel_speed_calculation_l3082_308219

/-- Prove that given a wheel with a circumference of 8 feet, if reducing the time
    for a complete rotation by 0.5 seconds increases the speed by 6 miles per hour,
    then the original speed of the wheel is 9 miles per hour. -/
theorem wheel_speed_calculation (r : ℝ) : 
  let circumference : ℝ := 8 / 5280  -- circumference in miles
  let t : ℝ := circumference * 3600 / r  -- time for one rotation in seconds
  let new_t : ℝ := t - 0.5  -- new time after reduction
  let new_r : ℝ := r + 6  -- new speed after increase
  (new_r * new_t / 3600 = circumference) →  -- equation for new speed and time
  r = 9 := by
sorry

end NUMINAMATH_CALUDE_wheel_speed_calculation_l3082_308219


namespace NUMINAMATH_CALUDE_equation_solution_l3082_308211

theorem equation_solution :
  let f (x : ℂ) := (3 * x^2 - 1) / (4 * x - 4)
  ∀ x : ℂ, f x = 2/3 ↔ x = 8/18 + (Complex.I * Real.sqrt 116)/18 ∨ x = 8/18 - (Complex.I * Real.sqrt 116)/18 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3082_308211


namespace NUMINAMATH_CALUDE_ellipse_intersection_slope_product_l3082_308266

/-- Given a line l passing through (-2,0) with slope k1 (k1 ≠ 0) intersecting 
    the ellipse x^2 + 2y^2 = 4 at points P1 and P2, and P being the midpoint of P1P2, 
    if k2 is the slope of OP, then k1 * k2 = -1/2 -/
theorem ellipse_intersection_slope_product (k1 : ℝ) (h1 : k1 ≠ 0) : 
  ∃ (P1 P2 P : ℝ × ℝ) (k2 : ℝ),
    (P1.1^2 + 2*P1.2^2 = 4) ∧ 
    (P2.1^2 + 2*P2.2^2 = 4) ∧
    (P1.2 = k1 * (P1.1 + 2)) ∧ 
    (P2.2 = k1 * (P2.1 + 2)) ∧
    (P = ((P1.1 + P2.1)/2, (P1.2 + P2.2)/2)) ∧
    (k2 = P.2 / P.1) →
    k1 * k2 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_intersection_slope_product_l3082_308266


namespace NUMINAMATH_CALUDE_school_camp_buses_l3082_308223

theorem school_camp_buses (B : ℕ) (S : ℕ) : 
  B ≤ 18 ∧                           -- No more than 18 buses
  S = 22 * B + 3 ∧                   -- Initial distribution with 3 left out
  ∃ (n : ℕ), n ≤ 36 ∧                -- Each bus can hold up to 36 people
  S = n * (B - 1) ∧                  -- Even distribution after one bus leaves
  n = (22 * B + 3) / (B - 1) →       -- Relationship between n, B, and S
  S = 355 :=
by sorry

end NUMINAMATH_CALUDE_school_camp_buses_l3082_308223


namespace NUMINAMATH_CALUDE_colorings_theorem_l3082_308206

/-- The number of ways to color five cells in a 5x5 grid with one colored cell in each row and column. -/
def total_colorings : ℕ := 120

/-- The number of ways to color five cells in a 5x5 grid without one corner cell, 
    with one colored cell in each row and column. -/
def colorings_without_one_corner : ℕ := 96

/-- The number of ways to color five cells in a 5x5 grid without two corner cells, 
    with one colored cell in each row and column. -/
def colorings_without_two_corners : ℕ := 78

theorem colorings_theorem : 
  colorings_without_two_corners = total_colorings - 2 * (total_colorings - colorings_without_one_corner) + 6 :=
by sorry

end NUMINAMATH_CALUDE_colorings_theorem_l3082_308206


namespace NUMINAMATH_CALUDE_sequence_term_16_l3082_308269

theorem sequence_term_16 (a : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → a n = (Real.sqrt 2) ^ (n - 1)) →
  ∃ n : ℕ, n > 0 ∧ a n = 16 ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_sequence_term_16_l3082_308269


namespace NUMINAMATH_CALUDE_surface_area_of_specific_solid_l3082_308225

/-- A right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  base_side : ℝ

/-- A solid formed by slicing off the top of the prism -/
structure SlicedSolid where
  prism : RightPrism

/-- The surface area of the sliced solid -/
def surface_area (solid : SlicedSolid) : ℝ :=
  sorry

/-- Theorem stating the surface area of the specific sliced solid -/
theorem surface_area_of_specific_solid :
  let prism := RightPrism.mk 20 10
  let solid := SlicedSolid.mk prism
  surface_area solid = 50 + (25 * Real.sqrt 3) / 4 + (5 * Real.sqrt 118.75) / 2 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_specific_solid_l3082_308225


namespace NUMINAMATH_CALUDE_sqrt_nine_minus_half_inverse_equals_one_l3082_308245

theorem sqrt_nine_minus_half_inverse_equals_one :
  Real.sqrt 9 - (1/2)⁻¹ = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_minus_half_inverse_equals_one_l3082_308245


namespace NUMINAMATH_CALUDE_inequality_proof_l3082_308239

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) :
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) ≥ 
  2 / (1 + a) + 2 / (1 + b) + 2 / (1 + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3082_308239


namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l3082_308247

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l3082_308247


namespace NUMINAMATH_CALUDE_factor_expression_l3082_308272

theorem factor_expression (x : ℝ) : 92 * x^3 - 184 * x^6 = 92 * x^3 * (1 - 2 * x^3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3082_308272


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_increase_l3082_308255

theorem rectangular_prism_volume_increase (a b c : ℝ) : 
  (a * b * c = 8) → 
  ((a + 1) * (b + 1) * (c + 1) = 27) → 
  ((a + 2) * (b + 2) * (c + 2) = 64) := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_increase_l3082_308255


namespace NUMINAMATH_CALUDE_k_range_oa_perpendicular_ob_l3082_308288

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := y^2 = -x
def line (k x y : ℝ) : Prop := y = k * (x + 1)

-- Define the intersection points A and B
def intersection_points (k : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), 
    parabola x1 y1 ∧ line k x1 y1 ∧
    parabola x2 y2 ∧ line k x2 y2 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2)

-- Define the vertex O of the parabola
def vertex : ℝ × ℝ := (0, 0)

-- Theorem for the range of k
theorem k_range : 
  ∀ k : ℝ, intersection_points k ↔ k ≠ 0 :=
sorry

-- Define perpendicularity
def perpendicular (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1) = 0

-- Theorem for perpendicularity of OA and OB
theorem oa_perpendicular_ob (k : ℝ) :
  k ≠ 0 → 
  ∃ (x1 y1 x2 y2 : ℝ),
    parabola x1 y1 ∧ line k x1 y1 ∧
    parabola x2 y2 ∧ line k x2 y2 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) →
    perpendicular vertex (x1, y1) (x2, y2) :=
sorry

end NUMINAMATH_CALUDE_k_range_oa_perpendicular_ob_l3082_308288


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_60_l3082_308232

theorem sqrt_sum_equals_sqrt_60 :
  Real.sqrt (25 - 10 * Real.sqrt 6) + Real.sqrt (25 + 10 * Real.sqrt 6) = Real.sqrt 60 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_60_l3082_308232


namespace NUMINAMATH_CALUDE_boxes_with_neither_l3082_308271

theorem boxes_with_neither (total : ℕ) (stickers : ℕ) (stamps : ℕ) (both : ℕ) 
  (h1 : total = 15)
  (h2 : stickers = 9)
  (h3 : stamps = 5)
  (h4 : both = 3) : 
  total - (stickers + stamps - both) = 4 := by
  sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l3082_308271


namespace NUMINAMATH_CALUDE_mean_of_scores_l3082_308227

def scores : List ℝ := [69, 68, 70, 61, 74, 62, 65, 74]

theorem mean_of_scores :
  (scores.sum / scores.length : ℝ) = 67.875 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_scores_l3082_308227


namespace NUMINAMATH_CALUDE_alpha_value_l3082_308242

theorem alpha_value (α β γ : Real) 
  (h1 : 0 < α ∧ α < π)
  (h2 : α + β + γ = π)
  (h3 : 2 * Real.sin α + Real.tan β + Real.tan γ = 2 * Real.sin α * Real.tan β * Real.tan γ) :
  α = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l3082_308242


namespace NUMINAMATH_CALUDE_job_completion_time_l3082_308298

/-- Given that Sylvia can complete a job in 45 minutes and Carla can complete
    the same job in 30 minutes, prove that together they can complete the job
    in 18 minutes. -/
theorem job_completion_time (sylvia_time carla_time : ℝ) 
    (h_sylvia : sylvia_time = 45)
    (h_carla : carla_time = 30) :
    1 / (1 / sylvia_time + 1 / carla_time) = 18 := by
  sorry


end NUMINAMATH_CALUDE_job_completion_time_l3082_308298


namespace NUMINAMATH_CALUDE_intersection_implies_a_zero_l3082_308215

def A : Set ℝ := {0, 1, 3}
def B (a : ℝ) : Set ℝ := {a + 1, a^2 + 2}

theorem intersection_implies_a_zero (a : ℝ) : A ∩ B a = {1} → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_zero_l3082_308215


namespace NUMINAMATH_CALUDE_coin_weighing_strategy_exists_l3082_308228

/-- Represents the possible weights of a coin type -/
inductive CoinWeight
  | Five
  | Six
  | Seven
  | Eight

/-- Represents the result of a weighing -/
inductive WeighingResult
  | LeftHeavier
  | RightHeavier
  | Equal

/-- Represents a weighing strategy -/
structure WeighingStrategy :=
  (firstWeighing : WeighingResult → Option WeighingResult)

/-- Represents the coin set -/
structure CoinSet :=
  (doubloonWeight : CoinWeight)
  (crownWeight : CoinWeight)

/-- Determines if a weighing strategy can identify the exact weights -/
def canIdentifyWeights (strategy : WeighingStrategy) (coins : CoinSet) : Prop :=
  ∃ (result1 : WeighingResult) (result2 : Option WeighingResult),
    (result2 = strategy.firstWeighing result1) ∧
    (∀ (otherCoins : CoinSet),
      (otherCoins ≠ coins) →
      (∃ (otherResult1 : WeighingResult) (otherResult2 : Option WeighingResult),
        (otherResult2 = strategy.firstWeighing otherResult1) ∧
        ((otherResult1 ≠ result1) ∨ (otherResult2 ≠ result2))))

theorem coin_weighing_strategy_exists :
  ∃ (strategy : WeighingStrategy),
    ∀ (coins : CoinSet),
      (coins.doubloonWeight = CoinWeight.Five ∨ coins.doubloonWeight = CoinWeight.Six) →
      (coins.crownWeight = CoinWeight.Seven ∨ coins.crownWeight = CoinWeight.Eight) →
      canIdentifyWeights strategy coins :=
by sorry


end NUMINAMATH_CALUDE_coin_weighing_strategy_exists_l3082_308228


namespace NUMINAMATH_CALUDE_smaller_number_problem_l3082_308222

theorem smaller_number_problem (x y : ℕ) 
  (h1 : y - x = 1365)
  (h2 : y = 6 * x + 15) : 
  x = 270 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l3082_308222


namespace NUMINAMATH_CALUDE_probability_multiple_of_three_in_eight_rolls_l3082_308273

theorem probability_multiple_of_three_in_eight_rolls : 
  let p : ℚ := 1 - (2/3)^8
  p = 6305/6561 := by sorry

end NUMINAMATH_CALUDE_probability_multiple_of_three_in_eight_rolls_l3082_308273


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l3082_308296

theorem binomial_expansion_sum (n : ℕ) : 
  (∃ P S : ℕ, (P = (3 + 1)^n) ∧ (S = 2^n) ∧ (P + S = 272)) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l3082_308296


namespace NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l3082_308263

/-- Represents a box with a square base -/
structure Box where
  base_length : ℝ
  height : ℝ

/-- Calculates the area of wrapping paper needed for a given box -/
def wrapping_paper_area (box : Box) : ℝ :=
  2 * (box.base_length + box.height)^2

/-- Theorem stating that the area of the wrapping paper is 2(w+h)^2 -/
theorem wrapping_paper_area_theorem (box : Box) :
  wrapping_paper_area box = 2 * (box.base_length + box.height)^2 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l3082_308263


namespace NUMINAMATH_CALUDE_odd_divisors_of_square_plus_one_l3082_308287

theorem odd_divisors_of_square_plus_one (x : ℤ) (d : ℤ) (h : d ∣ x^2 + 1) (hodd : Odd d) :
  ∃ (k : ℤ), d = 4 * k + 1 := by
sorry

end NUMINAMATH_CALUDE_odd_divisors_of_square_plus_one_l3082_308287


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3082_308231

theorem partial_fraction_decomposition (x A B C : ℝ) :
  (x + 2) / (x^3 - 9*x^2 + 14*x + 24) = A / (x - 4) + B / (x - 3) + C / ((x + 2)^2) →
  A = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3082_308231


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_ratio_l3082_308229

theorem geometric_sequence_minimum_ratio :
  ∀ (a : ℕ → ℕ) (q : ℚ),
  (∀ n : ℕ, 1 ≤ n → n < 2016 → a (n + 1) = a n * q) →
  (1 < q ∧ q < 2) →
  (∀ r : ℚ, 1 < r ∧ r < 2 → a 2016 ≤ a 1 * r^2015) →
  q = 6/5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_ratio_l3082_308229


namespace NUMINAMATH_CALUDE_minimize_distance_sum_l3082_308205

/-- Given points P, Q, and R in a coordinate plane, prove that the value of m 
    that minimizes the sum of distances PR + QR is 7/2, under specific conditions. -/
theorem minimize_distance_sum (P Q R : ℝ × ℝ) (x m : ℝ) : 
  P = (7, 7) →
  Q = (3, 2) →
  R = (x, m) →
  ((-7 : ℝ), 7) ∈ {(x, y) | y = 3*x - 4} →
  (∀ m' : ℝ, 
    Real.sqrt ((7 - x)^2 + (7 - m')^2) + Real.sqrt ((3 - x)^2 + (2 - m')^2) ≥ 
    Real.sqrt ((7 - x)^2 + (7 - m)^2) + Real.sqrt ((3 - x)^2 + (2 - m)^2)) →
  m = 7/2 := by
sorry

end NUMINAMATH_CALUDE_minimize_distance_sum_l3082_308205


namespace NUMINAMATH_CALUDE_biology_homework_wednesday_l3082_308265

def homework_monday : ℚ := 3/5
def remaining_after_monday : ℚ := 1 - homework_monday
def homework_tuesday : ℚ := (1/3) * remaining_after_monday

theorem biology_homework_wednesday :
  1 - homework_monday - homework_tuesday = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_biology_homework_wednesday_l3082_308265


namespace NUMINAMATH_CALUDE_infinitely_many_linear_combinations_l3082_308226

/-- An infinite sequence of positive integers with strictly increasing terms. -/
def StrictlyIncreasingSequence (a : ℕ → ℕ) : Prop :=
  ∀ k, 0 < a k ∧ a k < a (k + 1)

/-- The property that infinitely many terms can be expressed as a linear combination of two earlier terms. -/
def InfinitelyManyLinearCombinations (a : ℕ → ℕ) : Prop :=
  ∀ N, ∃ m p q x y, N < m ∧ p ≠ q ∧ 0 < x ∧ 0 < y ∧ a m = x * a p + y * a q

/-- The main theorem: any strictly increasing sequence of positive integers has infinitely many terms
    that can be expressed as a linear combination of two earlier terms. -/
theorem infinitely_many_linear_combinations
  (a : ℕ → ℕ) (h : StrictlyIncreasingSequence a) :
  InfinitelyManyLinearCombinations a :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_linear_combinations_l3082_308226


namespace NUMINAMATH_CALUDE_cubic_difference_evaluation_l3082_308285

theorem cubic_difference_evaluation : 
  2010^3 - 2007 * 2010^2 - 2007^2 * 2010 + 2007^3 = 36153 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_evaluation_l3082_308285


namespace NUMINAMATH_CALUDE_correct_fraction_proof_l3082_308235

theorem correct_fraction_proof (x y : ℚ) : 
  (5 : ℚ) / 6 * 384 = x / y * 384 + 200 → x / y = (5 : ℚ) / 16 := by
sorry

end NUMINAMATH_CALUDE_correct_fraction_proof_l3082_308235


namespace NUMINAMATH_CALUDE_bd_equals_twelve_l3082_308209

/-- Represents a quadrilateral ABCD with given side lengths and diagonal BD --/
structure Quadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  BD : ℤ

/-- Theorem stating that BD = 12 is a valid solution for the given quadrilateral --/
theorem bd_equals_twelve (q : Quadrilateral) 
  (h1 : q.AB = 6)
  (h2 : q.BC = 12)
  (h3 : q.CD = 6)
  (h4 : q.DA = 8) :
  q.BD = 12 → 
  (q.AB + q.BD > q.DA) ∧ 
  (q.BC + q.CD > q.BD) ∧ 
  (q.DA + q.BD > q.AB) ∧ 
  (q.BD + q.CD > q.BC) ∧ 
  (q.BD > 6) ∧ 
  (q.BD < 14) := by
  sorry

#check bd_equals_twelve

end NUMINAMATH_CALUDE_bd_equals_twelve_l3082_308209


namespace NUMINAMATH_CALUDE_yuna_weekly_problems_l3082_308257

/-- The number of English problems Yuna solves per day -/
def problems_per_day : ℕ := 8

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of English problems Yuna solves in one week -/
def problems_per_week : ℕ := problems_per_day * days_in_week

theorem yuna_weekly_problems : problems_per_week = 56 := by
  sorry

end NUMINAMATH_CALUDE_yuna_weekly_problems_l3082_308257


namespace NUMINAMATH_CALUDE_function_inequality_l3082_308270

theorem function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
    (h_cond : ∀ x, (x - 1) * deriv f x ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3082_308270


namespace NUMINAMATH_CALUDE_line_translation_theorem_l3082_308299

/-- A line in the xy-plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line vertically -/
def translate_line (l : Line) (units : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + units }

theorem line_translation_theorem (original : Line) (units : ℝ) :
  original.slope = 1/2 ∧ original.intercept = -2 ∧ units = 3 →
  translate_line original units = Line.mk (1/2) 1 :=
by sorry

end NUMINAMATH_CALUDE_line_translation_theorem_l3082_308299


namespace NUMINAMATH_CALUDE_ball_box_problem_l3082_308212

theorem ball_box_problem (num_balls : ℕ) (X : ℕ) (h1 : num_balls = 25) 
  (h2 : num_balls - 20 = X - num_balls) : X = 30 := by
  sorry

end NUMINAMATH_CALUDE_ball_box_problem_l3082_308212


namespace NUMINAMATH_CALUDE_unique_three_digit_square_l3082_308203

theorem unique_three_digit_square (a b c : Nat) : 
  a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  a^2 < 10 ∧ b^2 < 10 ∧ c^2 < 10 ∧
  (100*a + 10*b + c)^2 = 1000*100*a + 1000*10*b + 1000*c + 100*a + 10*b + c →
  a = 2 ∧ b = 3 ∧ c = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_three_digit_square_l3082_308203


namespace NUMINAMATH_CALUDE_parabola_above_line_l3082_308280

theorem parabola_above_line (p : ℝ) : 
  (∀ x : ℝ, x^2 - 2*p*x + p + 1 ≥ -12*x + 5) ↔ (5 ≤ p ∧ p ≤ 8) := by
  sorry

end NUMINAMATH_CALUDE_parabola_above_line_l3082_308280


namespace NUMINAMATH_CALUDE_ngon_recovery_l3082_308278

/-- Represents a point in the plane with an associated number -/
structure MarkedPoint where
  x : ℝ
  y : ℝ
  number : ℕ

/-- Represents a regular n-gon with its center -/
structure RegularNGon where
  n : ℕ
  center : MarkedPoint
  vertices : Fin n → MarkedPoint

/-- Represents a triangle formed by two adjacent vertices and the center -/
structure Triangle where
  a : MarkedPoint
  b : MarkedPoint
  c : MarkedPoint

/-- Function to generate the list of triangles from a regular n-gon -/
def generateTriangles (ngon : RegularNGon) : List Triangle := sorry

/-- Function to get the multiset of numbers from a triangle -/
def getTriangleNumbers (triangle : Triangle) : Multiset ℕ := sorry

/-- Predicate to check if the original numbers can be uniquely recovered -/
def canRecover (ngon : RegularNGon) : Prop := sorry

theorem ngon_recovery (n : ℕ) :
  ∀ (ngon : RegularNGon),
    ngon.n = n →
    canRecover ngon ↔ Odd n :=
  sorry

end NUMINAMATH_CALUDE_ngon_recovery_l3082_308278


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3082_308224

theorem quadratic_equation_roots (p : ℝ) : 
  (∃ x₁ x₂ : ℂ, x₁ ≠ x₂ ∧ 
   (∀ x : ℂ, x^2 - p*x + 1 = 0 ↔ (x = x₁ ∨ x = x₂)) ∧
   x₁.im ≠ 0 ∧ x₂.im ≠ 0 ∧
   Complex.abs (x₁ - x₂) = 1) →
  p = Real.sqrt 3 ∨ p = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3082_308224


namespace NUMINAMATH_CALUDE_b_97_mod_81_l3082_308260

def b (n : ℕ) : ℕ := 7^n + 9^n

theorem b_97_mod_81 : b 97 ≡ 52 [MOD 81] := by sorry

end NUMINAMATH_CALUDE_b_97_mod_81_l3082_308260


namespace NUMINAMATH_CALUDE_marathon_time_l3082_308207

/-- Calculates the total time to complete a marathon given specific conditions -/
theorem marathon_time (total_distance : ℝ) (initial_distance : ℝ) (initial_time : ℝ) (remaining_pace_factor : ℝ) :
  total_distance = 26 →
  initial_distance = 10 →
  initial_time = 1 →
  remaining_pace_factor = 0.8 →
  let initial_pace := initial_distance / initial_time
  let remaining_distance := total_distance - initial_distance
  let remaining_pace := initial_pace * remaining_pace_factor
  let remaining_time := remaining_distance / remaining_pace
  initial_time + remaining_time = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_marathon_time_l3082_308207


namespace NUMINAMATH_CALUDE_rental_van_cost_increase_l3082_308294

theorem rental_van_cost_increase (C : ℝ) : 
  C / 8 - C / 9 = C / 72 := by sorry

end NUMINAMATH_CALUDE_rental_van_cost_increase_l3082_308294


namespace NUMINAMATH_CALUDE_jill_earnings_l3082_308233

def first_month_daily_wage : ℕ := 10
def days_per_month : ℕ := 30

def second_month_daily_wage : ℕ := 2 * first_month_daily_wage
def third_month_working_days : ℕ := days_per_month / 2

def first_month_earnings : ℕ := first_month_daily_wage * days_per_month
def second_month_earnings : ℕ := second_month_daily_wage * days_per_month
def third_month_earnings : ℕ := second_month_daily_wage * third_month_working_days

def total_earnings : ℕ := first_month_earnings + second_month_earnings + third_month_earnings

theorem jill_earnings : total_earnings = 1200 := by
  sorry

end NUMINAMATH_CALUDE_jill_earnings_l3082_308233


namespace NUMINAMATH_CALUDE_box_filling_theorem_l3082_308286

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of identical cubes that can fill a box completely -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  sorry

/-- The theorem stating that for a box with dimensions 36x45x18 inches, 
    the smallest number of identical cubes that can fill it is 40 -/
theorem box_filling_theorem : 
  let box : BoxDimensions := { length := 36, width := 45, depth := 18 }
  smallestNumberOfCubes box = 40 := by
  sorry

end NUMINAMATH_CALUDE_box_filling_theorem_l3082_308286


namespace NUMINAMATH_CALUDE_sandy_token_difference_l3082_308250

/-- Represents the number of Safe Moon tokens Sandy bought -/
def total_tokens : ℕ := 1000000

/-- Represents the number of Sandy's siblings -/
def num_siblings : ℕ := 4

/-- Calculates the number of tokens Sandy keeps for herself -/
def sandy_tokens : ℕ := total_tokens / 2

/-- Calculates the number of tokens each sibling receives -/
def sibling_tokens : ℕ := (total_tokens - sandy_tokens) / num_siblings

/-- Proves that Sandy has 375,000 more tokens than any of her siblings -/
theorem sandy_token_difference : sandy_tokens - sibling_tokens = 375000 := by
  sorry

end NUMINAMATH_CALUDE_sandy_token_difference_l3082_308250


namespace NUMINAMATH_CALUDE_car_oil_problem_l3082_308284

/-- Represents the relationship between remaining oil and distance traveled for a car -/
def oil_remaining (x : ℝ) : ℝ := 56 - 0.08 * x

/-- The initial amount of oil in the tank -/
def initial_oil : ℝ := 56

/-- The rate of oil consumption per kilometer -/
def consumption_rate : ℝ := 0.08

theorem car_oil_problem :
  (∀ x : ℝ, oil_remaining x = 56 - 0.08 * x) ∧
  (oil_remaining 350 = 28) ∧
  (∃ x : ℝ, oil_remaining x = 8 ∧ x = 600) := by
  sorry


end NUMINAMATH_CALUDE_car_oil_problem_l3082_308284


namespace NUMINAMATH_CALUDE_permutation_combination_equality_l3082_308295

/-- Given that A_n^2 = C_n^(n-3), prove that n = 8 --/
theorem permutation_combination_equality (n : ℕ) : 
  (n.factorial / (n - 2).factorial) = (n.factorial / ((3).factorial * (n - 3).factorial)) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_permutation_combination_equality_l3082_308295


namespace NUMINAMATH_CALUDE_g_of_3_eq_38_div_5_l3082_308217

noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

noncomputable def g (x : ℝ) : ℝ := 1 / (f.invFun x) + 7

theorem g_of_3_eq_38_div_5 : g 3 = 38 / 5 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_eq_38_div_5_l3082_308217


namespace NUMINAMATH_CALUDE_complement_intersection_equals_five_l3082_308243

open Set

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 4}
def N : Set Nat := {2, 3}

theorem complement_intersection_equals_five :
  (U \ M) ∩ (U \ N) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_five_l3082_308243


namespace NUMINAMATH_CALUDE_rest_time_calculation_l3082_308252

theorem rest_time_calculation (walking_rate : ℝ) (total_distance : ℝ) (total_time : ℝ) 
  (rest_interval : ℝ) (h1 : walking_rate = 10) (h2 : total_distance = 50) 
  (h3 : total_time = 320) (h4 : rest_interval = 10) : 
  (total_time - (total_distance / walking_rate) * 60) / ((total_distance / rest_interval) - 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_rest_time_calculation_l3082_308252


namespace NUMINAMATH_CALUDE_odd_function_property_l3082_308277

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_even : is_even_function (fun x ↦ f (x + 1)))
  (h_f_neg_one : f (-1) = -1) :
  f 2018 + f 2019 = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l3082_308277


namespace NUMINAMATH_CALUDE_polynomial_inequality_range_l3082_308254

theorem polynomial_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → a * x^3 - x^2 + 4*x + 3 ≥ 0) →
  a ∈ Set.Icc (-6) (-2) := by
sorry

end NUMINAMATH_CALUDE_polynomial_inequality_range_l3082_308254


namespace NUMINAMATH_CALUDE_sixDigitPermutations_eq_90_l3082_308238

/-- The number of different positive, six-digit integers that can be formed using the digits 1, 1, 3, 3, 7, and 7 -/
def sixDigitPermutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of such permutations is 90 -/
theorem sixDigitPermutations_eq_90 : sixDigitPermutations = 90 := by
  sorry

end NUMINAMATH_CALUDE_sixDigitPermutations_eq_90_l3082_308238


namespace NUMINAMATH_CALUDE_max_diff_reversed_digits_l3082_308214

/-- Given two three-digit positive integers with the same digits in reverse order,
    prove that their maximum difference less than 300 is 297. -/
theorem max_diff_reversed_digits (q r : ℕ) : 
  (100 ≤ q) ∧ (q < 1000) ∧  -- q is a three-digit number
  (100 ≤ r) ∧ (r < 1000) ∧  -- r is a three-digit number
  (∃ a b c : ℕ, q = 100*a + 10*b + c ∧ r = 100*c + 10*b + a) ∧  -- q and r have reversed digits
  (q > r) ∧  -- ensure q is greater than r
  (q - r < 300) →  -- difference is less than 300
  (q - r ≤ 297) ∧ (∃ q' r' : ℕ, q' - r' = 297 ∧ 
    (100 ≤ q') ∧ (q' < 1000) ∧ (100 ≤ r') ∧ (r' < 1000) ∧
    (∃ a b c : ℕ, q' = 100*a + 10*b + c ∧ r' = 100*c + 10*b + a) ∧
    (q' > r') ∧ (q' - r' < 300)) := by
  sorry

end NUMINAMATH_CALUDE_max_diff_reversed_digits_l3082_308214


namespace NUMINAMATH_CALUDE_total_pencils_l3082_308248

-- Define the number of pencils in each set
def pencils_set_a : ℕ := 10
def pencils_set_b : ℕ := 20
def pencils_set_c : ℕ := 30

-- Define the number of friends who bought each set
def friends_set_a : ℕ := 3
def friends_set_b : ℕ := 2
def friends_set_c : ℕ := 2

-- Define Chloe's purchase
def chloe_sets : ℕ := 1

-- Theorem statement
theorem total_pencils :
  (friends_set_a * pencils_set_a + 
   friends_set_b * pencils_set_b + 
   friends_set_c * pencils_set_c) +
  (chloe_sets * (pencils_set_a + pencils_set_b + pencils_set_c)) = 190 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l3082_308248


namespace NUMINAMATH_CALUDE_choose_cooks_l3082_308291

theorem choose_cooks (n m : ℕ) (h1 : n = 10) (h2 : m = 3) :
  Nat.choose n m = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_cooks_l3082_308291


namespace NUMINAMATH_CALUDE_guppy_ratio_l3082_308208

/-- The number of guppies Haylee has -/
def hayleeGuppies : ℕ := 36

/-- The number of guppies Jose has -/
def joseGuppies : ℕ := hayleeGuppies / 2

/-- The number of guppies Charliz has -/
def charlizGuppies : ℕ := 6

/-- The number of guppies Nicolai has -/
def nicolaiGuppies : ℕ := 4 * charlizGuppies

/-- The total number of guppies all four friends have -/
def totalGuppies : ℕ := 84

/-- Theorem stating that the ratio of Charliz's guppies to Jose's guppies is 1:3 -/
theorem guppy_ratio :
  charlizGuppies * 3 = joseGuppies ∧
  hayleeGuppies + joseGuppies + charlizGuppies + nicolaiGuppies = totalGuppies :=
by sorry

end NUMINAMATH_CALUDE_guppy_ratio_l3082_308208


namespace NUMINAMATH_CALUDE_rectangle_y_value_l3082_308244

/-- Rectangle EFGH with vertices E(0, 0), F(0, 5), G(y, 5), and H(y, 0) -/
structure Rectangle where
  y : ℝ
  h_positive : y > 0

/-- The area of the rectangle -/
def area (r : Rectangle) : ℝ := 5 * r.y

theorem rectangle_y_value (r : Rectangle) (h_area : area r = 35) : r.y = 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l3082_308244


namespace NUMINAMATH_CALUDE_equation_graph_is_axes_l3082_308268

/-- The set of points satisfying (x+y)^2 = x^2 + y^2 is equivalent to the union of the x-axis and y-axis -/
theorem equation_graph_is_axes (x y : ℝ) : 
  (x + y)^2 = x^2 + y^2 ↔ x = 0 ∨ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_graph_is_axes_l3082_308268


namespace NUMINAMATH_CALUDE_cells_after_three_divisions_l3082_308201

/-- The number of cells after n divisions, starting with 1 cell -/
def cells_after_divisions (n : ℕ) : ℕ := 2^n

/-- Theorem: After 3 divisions, the number of cells is 8 -/
theorem cells_after_three_divisions : cells_after_divisions 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cells_after_three_divisions_l3082_308201


namespace NUMINAMATH_CALUDE_max_seated_people_is_14_l3082_308274

/-- Represents the state of the break room --/
structure BreakRoom where
  totalTables : Nat
  maxSeatsPerTable : Nat
  maxSeatsPerTableWithDistancing : Nat
  occupiedTables : List Nat
  totalChairs : Nat

/-- Calculates the maximum number of people that can be seated in the break room --/
def maxSeatedPeople (room : BreakRoom) : Nat :=
  sorry

/-- Theorem stating that the maximum number of people that can be seated is 14 --/
theorem max_seated_people_is_14 (room : BreakRoom) : 
  room.totalTables = 7 ∧ 
  room.maxSeatsPerTable = 6 ∧ 
  room.maxSeatsPerTableWithDistancing = 3 ∧ 
  room.occupiedTables = [2, 1, 1, 3] ∧ 
  room.totalChairs = 14 →
  maxSeatedPeople room = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_seated_people_is_14_l3082_308274


namespace NUMINAMATH_CALUDE_adam_books_theorem_l3082_308237

def initial_books : ℕ := 67
def sold_fraction : ℚ := 2/3
def reinvestment_fraction : ℚ := 3/4
def new_book_price : ℕ := 3

def books_after_transactions : ℕ := 56

theorem adam_books_theorem :
  let sold_books := (initial_books * sold_fraction).floor
  let money_earned := sold_books * new_book_price
  let money_for_new_books := (money_earned : ℚ) * reinvestment_fraction
  let new_books := (money_for_new_books / new_book_price).floor
  initial_books - sold_books + new_books = books_after_transactions := by
  sorry

end NUMINAMATH_CALUDE_adam_books_theorem_l3082_308237


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_11_l3082_308216

theorem smallest_positive_integer_ending_in_3_divisible_by_11 : 
  ∃ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 11 = 0 ∧ ∀ m : ℕ, m > 0 ∧ m % 10 = 3 ∧ m % 11 = 0 → n ≤ m :=
by
  -- The proof would go here
  sorry

#check smallest_positive_integer_ending_in_3_divisible_by_11

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_11_l3082_308216


namespace NUMINAMATH_CALUDE_joes_lifts_l3082_308290

theorem joes_lifts (total_weight first_lift : ℕ) 
  (h1 : total_weight = 1500)
  (h2 : first_lift = 600) : 
  2 * first_lift - (total_weight - first_lift) = 300 := by
  sorry

end NUMINAMATH_CALUDE_joes_lifts_l3082_308290


namespace NUMINAMATH_CALUDE_greatest_good_and_smallest_bad_l3082_308276

/-- Definition of a GOOD number -/
def isGood (M : ℕ) : Prop :=
  ∃ a b c d : ℕ, M ≤ a ∧ a < b ∧ b ≤ c ∧ c < d ∧ d ≤ M + 49 ∧ a * d = b * c

/-- Definition of a BAD number -/
def isBad (M : ℕ) : Prop := ¬(isGood M)

/-- The greatest GOOD number -/
def greatestGood : ℕ := 576

/-- The smallest BAD number -/
def smallestBad : ℕ := 443

/-- Theorem stating that 576 is the greatest GOOD number and 443 is the smallest BAD number -/
theorem greatest_good_and_smallest_bad :
  (∀ M : ℕ, M > greatestGood → isBad M) ∧
  (∀ M : ℕ, M < smallestBad → isGood M) ∧
  isGood greatestGood ∧
  isBad smallestBad :=
sorry

end NUMINAMATH_CALUDE_greatest_good_and_smallest_bad_l3082_308276


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l3082_308241

theorem purely_imaginary_condition (θ : ℝ) : 
  (∃ (y : ℝ), Complex.mk (Real.sin (2 * θ) - 1) (Real.sqrt 2 * Real.cos θ + 1) = Complex.I * y) ↔ 
  (∃ (k : ℤ), θ = 2 * k * Real.pi + Real.pi / 4) :=
sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l3082_308241


namespace NUMINAMATH_CALUDE_coin_flip_problem_l3082_308283

theorem coin_flip_problem (n : ℕ) 
  (p_tails : ℚ) 
  (p_event : ℚ) : 
  p_tails = 1/2 → 
  p_event = 3125/100000 → 
  p_event = (p_tails^2) * ((1 - p_tails)^3) → 
  n ≥ 5 → 
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_coin_flip_problem_l3082_308283


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3082_308262

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {2, 4, 6}

theorem complement_of_A_in_U :
  U \ A = {1, 3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3082_308262


namespace NUMINAMATH_CALUDE_sue_votes_l3082_308292

theorem sue_votes (total_votes : ℕ) (candidate1_percent : ℚ) (candidate2_percent : ℚ)
  (h_total : total_votes = 1000)
  (h_cand1 : candidate1_percent = 20 / 100)
  (h_cand2 : candidate2_percent = 45 / 100) :
  (1 - (candidate1_percent + candidate2_percent)) * total_votes = 350 :=
by sorry

end NUMINAMATH_CALUDE_sue_votes_l3082_308292


namespace NUMINAMATH_CALUDE_number_of_sailors_l3082_308236

theorem number_of_sailors (W : ℝ) (n : ℕ) : 
  (W + 64 - 56) / n = W / n + 1 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_of_sailors_l3082_308236


namespace NUMINAMATH_CALUDE_trig_sum_simplification_l3082_308264

theorem trig_sum_simplification :
  (Real.sin (30 * π / 180) + Real.sin (50 * π / 180) + Real.sin (70 * π / 180) + Real.sin (90 * π / 180) +
   Real.sin (110 * π / 180) + Real.sin (130 * π / 180) + Real.sin (150 * π / 180) + Real.sin (170 * π / 180)) /
  (Real.cos (15 * π / 180) * Real.cos (25 * π / 180) * Real.cos (50 * π / 180)) =
  (8 * Real.sin (80 * π / 180) * Real.cos (40 * π / 180) * Real.cos (20 * π / 180)) /
  (Real.cos (15 * π / 180) * Real.cos (25 * π / 180) * Real.cos (50 * π / 180)) :=
by
  sorry

end NUMINAMATH_CALUDE_trig_sum_simplification_l3082_308264


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_quarter_l3082_308289

theorem sin_cos_sum_equals_quarter :
  Real.sin (20 * π / 180) * Real.cos (70 * π / 180) +
  Real.sin (10 * π / 180) * Real.sin (50 * π / 180) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_quarter_l3082_308289


namespace NUMINAMATH_CALUDE_root_difference_ratio_l3082_308249

theorem root_difference_ratio (a b : ℝ) : 
  a^4 - 7*a - 3 = 0 → 
  b^4 - 7*b - 3 = 0 → 
  a > b → 
  (a - b) / (a^4 - b^4) = 1/7 := by
sorry

end NUMINAMATH_CALUDE_root_difference_ratio_l3082_308249


namespace NUMINAMATH_CALUDE_line_circle_separation_l3082_308202

theorem line_circle_separation (α β : ℝ) : 
  let m : ℝ × ℝ := (2 * Real.cos α, 2 * Real.sin α)
  let n : ℝ × ℝ := (3 * Real.cos β, 3 * Real.sin β)
  let angle_between := Real.arccos ((m.1 * n.1 + m.2 * n.2) / (Real.sqrt (m.1^2 + m.2^2) * Real.sqrt (n.1^2 + n.2^2)))
  let line_eq (x y : ℝ) := x * Real.cos α - y * Real.sin α + 1/2
  let circle_center : ℝ × ℝ := (Real.cos β, -Real.sin β)
  let circle_radius : ℝ := Real.sqrt 2 / 2
  let distance_to_line := |line_eq circle_center.1 circle_center.2|
  angle_between = π/3 → distance_to_line > circle_radius :=
by sorry

end NUMINAMATH_CALUDE_line_circle_separation_l3082_308202


namespace NUMINAMATH_CALUDE_card_area_theorem_l3082_308246

/-- Represents the dimensions of a rectangular card -/
structure Card where
  length : ℝ
  width : ℝ

/-- Calculates the area of a card -/
def area (c : Card) : ℝ := c.length * c.width

/-- Theorem: Given a 5x7 card, if shortening one side by 2 inches results in 
    an area of 21 square inches, then shortening the other side by 2 inches 
    will result in an area of 25 square inches -/
theorem card_area_theorem (c : Card) 
    (h1 : c.length = 5 ∧ c.width = 7)
    (h2 : ∃ (shortened_card : Card), 
      (shortened_card.length = c.length - 2 ∧ shortened_card.width = c.width) ∨
      (shortened_card.length = c.length ∧ shortened_card.width = c.width - 2))
    (h3 : ∃ (shortened_card : Card), 
      area shortened_card = 21 ∧
      ((shortened_card.length = c.length - 2 ∧ shortened_card.width = c.width) ∨
       (shortened_card.length = c.length ∧ shortened_card.width = c.width - 2)))
    : ∃ (other_shortened_card : Card), 
      area other_shortened_card = 25 ∧
      ((other_shortened_card.length = c.length - 2 ∧ other_shortened_card.width = c.width) ∨
       (other_shortened_card.length = c.length ∧ other_shortened_card.width = c.width - 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_card_area_theorem_l3082_308246


namespace NUMINAMATH_CALUDE_unique_triple_solution_l3082_308256

theorem unique_triple_solution : 
  ∃! (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b * c = 2010 ∧ 
    b + c * a = 250 ∧
    a = 3 ∧ b = 223 ∧ c = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l3082_308256


namespace NUMINAMATH_CALUDE_unique_prime_triple_l3082_308210

theorem unique_prime_triple : ∃! (I M C : ℕ),
  (Nat.Prime I ∧ Nat.Prime M ∧ Nat.Prime C) ∧
  (I ≤ M ∧ M ≤ C) ∧
  (I * M * C = I + M + C + 1007) ∧
  I = 2 ∧ M = 2 ∧ C = 337 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_triple_l3082_308210


namespace NUMINAMATH_CALUDE_incident_ray_slope_l3082_308261

/-- Given a circle with center (2, -1) and a point P(-1, -3), prove that the slope
    of the line passing through P and the reflection of the circle's center
    across the x-axis is 4/3. -/
theorem incident_ray_slope (P : ℝ × ℝ) (C : ℝ × ℝ) :
  P = (-1, -3) →
  C = (2, -1) →
  let D : ℝ × ℝ := (C.1, -C.2)  -- Reflection of C across x-axis
  (D.2 - P.2) / (D.1 - P.1) = 4/3 := by
sorry

end NUMINAMATH_CALUDE_incident_ray_slope_l3082_308261


namespace NUMINAMATH_CALUDE_product_sequence_sum_l3082_308253

theorem product_sequence_sum (c d : ℕ) (h1 : c / 3 = 12) (h2 : d = c - 1) : c + d = 71 := by
  sorry

end NUMINAMATH_CALUDE_product_sequence_sum_l3082_308253


namespace NUMINAMATH_CALUDE_proportion_solution_l3082_308282

theorem proportion_solution (x : ℚ) : (3/4 : ℚ) / x = 7/8 → x = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l3082_308282


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3082_308204

/-- Given a right circular cylinder with radius 2 intersected by a plane forming an ellipse,
    if the major axis of the ellipse is 25% longer than the minor axis,
    then the length of the major axis is 5. -/
theorem ellipse_major_axis_length (cylinder_radius : ℝ) (minor_axis : ℝ) (major_axis : ℝ) :
  cylinder_radius = 2 →
  minor_axis = 2 * cylinder_radius →
  major_axis = 1.25 * minor_axis →
  major_axis = 5 := by
sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3082_308204


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3082_308218

theorem inequality_solution_set (x : ℝ) : (x - 2) * (3 - x) > 0 ↔ 2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3082_308218


namespace NUMINAMATH_CALUDE_shara_shell_count_l3082_308200

/-- Calculates the total number of shells Shara has after her vacation. -/
def total_shells (initial_shells : ℕ) (shells_per_day : ℕ) (days : ℕ) (fourth_day_shells : ℕ) : ℕ :=
  initial_shells + shells_per_day * days + fourth_day_shells

/-- Theorem stating that Shara has 41 shells after her vacation. -/
theorem shara_shell_count : 
  total_shells 20 5 3 6 = 41 := by
  sorry

end NUMINAMATH_CALUDE_shara_shell_count_l3082_308200


namespace NUMINAMATH_CALUDE_charcoal_drawings_count_l3082_308234

theorem charcoal_drawings_count (total : ℕ) (colored_pencil : ℕ) (blending_marker : ℕ) 
  (h1 : total = 25)
  (h2 : colored_pencil = 14)
  (h3 : blending_marker = 7) :
  total - (colored_pencil + blending_marker) = 4 := by
  sorry

end NUMINAMATH_CALUDE_charcoal_drawings_count_l3082_308234


namespace NUMINAMATH_CALUDE_shop_e_tv_sets_l3082_308279

theorem shop_e_tv_sets (shops : Fin 5 → ℕ)
  (ha : shops 0 = 20)
  (hb : shops 1 = 30)
  (hc : shops 2 = 60)
  (hd : shops 3 = 80)
  (havg : (shops 0 + shops 1 + shops 2 + shops 3 + shops 4) / 5 = 48) :
  shops 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_shop_e_tv_sets_l3082_308279


namespace NUMINAMATH_CALUDE_youth_palace_participants_l3082_308293

theorem youth_palace_participants (last_year this_year : ℕ) :
  this_year = last_year + 41 →
  this_year = 3 * last_year - 35 →
  this_year = 79 ∧ last_year = 38 := by
  sorry

end NUMINAMATH_CALUDE_youth_palace_participants_l3082_308293


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3082_308281

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3082_308281


namespace NUMINAMATH_CALUDE_upstream_speed_calculation_l3082_308258

/-- Represents the speed of a man rowing in different conditions. -/
structure RowingSpeed where
  stillWater : ℝ
  downstream : ℝ

/-- Calculates the upstream speed of a man given his rowing speeds in still water and downstream. -/
def upstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.downstream

/-- Theorem stating that given a man's speed in still water is 35 kmph and his downstream speed is 45 kmph, his upstream speed is 25 kmph. -/
theorem upstream_speed_calculation (s : RowingSpeed) 
  (h1 : s.stillWater = 35) 
  (h2 : s.downstream = 45) : 
  upstreamSpeed s = 25 := by
  sorry

#eval upstreamSpeed { stillWater := 35, downstream := 45 }

end NUMINAMATH_CALUDE_upstream_speed_calculation_l3082_308258


namespace NUMINAMATH_CALUDE_limit_implies_a_range_l3082_308213

/-- If the limit of 3^n / (3^(n+1) + (a+1)^n) as n approaches infinity is 1/3, 
    then a is in the open interval (-4, 2) -/
theorem limit_implies_a_range (a : ℝ) :
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |3^n / (3^(n+1) + (a+1)^n) - 1/3| < ε) →
  a ∈ Set.Ioo (-4 : ℝ) 2 :=
by sorry

end NUMINAMATH_CALUDE_limit_implies_a_range_l3082_308213


namespace NUMINAMATH_CALUDE_initial_workers_l3082_308240

/-- Proves that the initial number of workers is 14, given the problem conditions --/
theorem initial_workers (total_toys : ℕ) (initial_days : ℕ) (added_workers : ℕ) (remaining_days : ℕ) :
  total_toys = 1400 →
  initial_days = 5 →
  added_workers = 14 →
  remaining_days = 2 →
  ∃ (initial_workers : ℕ),
    (initial_workers * initial_days + (initial_workers + added_workers) * remaining_days) * total_toys / 
    (initial_days + remaining_days) = total_toys ∧
    initial_workers = 14 := by
  sorry


end NUMINAMATH_CALUDE_initial_workers_l3082_308240


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l3082_308267

theorem smallest_integer_solution (x : ℤ) : (∀ y : ℤ, 7 + 3 * y < 26 → x ≤ y) ↔ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l3082_308267


namespace NUMINAMATH_CALUDE_convex_quad_polyhedron_16v_14f_l3082_308230

/-- A convex polyhedron with quadrilateral faces -/
structure ConvexQuadPolyhedron where
  vertices : ℕ
  faces : ℕ
  edges : ℕ
  convex : Bool
  all_faces_quadrilateral : Bool
  euler : vertices + faces - edges = 2
  quad_face_edge_relation : edges = 2 * faces

/-- Theorem: A convex polyhedron with 16 vertices and all quadrilateral faces has 14 faces -/
theorem convex_quad_polyhedron_16v_14f :
  ∀ (P : ConvexQuadPolyhedron), 
    P.vertices = 16 ∧ P.convex ∧ P.all_faces_quadrilateral → P.faces = 14 :=
by sorry

end NUMINAMATH_CALUDE_convex_quad_polyhedron_16v_14f_l3082_308230


namespace NUMINAMATH_CALUDE_sanhat_integers_l3082_308251

theorem sanhat_integers (x y : ℤ) (h1 : 3 * x + 2 * y = 160) (h2 : x = 36 ∨ y = 36) :
  (x = 36 ∧ y = 26) ∨ (y = 36 ∧ x = 26) :=
sorry

end NUMINAMATH_CALUDE_sanhat_integers_l3082_308251


namespace NUMINAMATH_CALUDE_proposition_two_l3082_308220

theorem proposition_two (a b : ℝ) : a > b → ((1 / a < 1 / b) ↔ (a * b > 0)) := by
  sorry

end NUMINAMATH_CALUDE_proposition_two_l3082_308220


namespace NUMINAMATH_CALUDE_angle_from_coordinates_l3082_308221

theorem angle_from_coordinates (α : Real) 
  (h1 : α > 0) (h2 : α < 2 * Real.pi)
  (h3 : ∃ (x y : Real), x = Real.sin (Real.pi / 6) ∧ 
                        y = Real.cos (5 * Real.pi / 6) ∧
                        x = Real.sin α ∧
                        y = Real.cos α) :
  α = 5 * Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_angle_from_coordinates_l3082_308221
