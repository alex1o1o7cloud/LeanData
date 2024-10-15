import Mathlib

namespace NUMINAMATH_CALUDE_three_m_minus_n_l3225_322500

theorem three_m_minus_n (m n : ℝ) (h : m + 1 = (n - 2) / 3) : 3 * m - n = -5 := by
  sorry

end NUMINAMATH_CALUDE_three_m_minus_n_l3225_322500


namespace NUMINAMATH_CALUDE_regression_lines_intersection_l3225_322504

/-- A linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The point where a regression line passes through -/
def passes_through (l : RegressionLine) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem regression_lines_intersection
  (l₁ l₂ : RegressionLine)
  (s t : ℝ)
  (h₁ : passes_through l₁ s t)
  (h₂ : passes_through l₂ s t) :
  ∃ (x y : ℝ), passes_through l₁ x y ∧ passes_through l₂ x y ∧ x = s ∧ y = t :=
sorry

end NUMINAMATH_CALUDE_regression_lines_intersection_l3225_322504


namespace NUMINAMATH_CALUDE_base_10_144_equals_base_12_100_l3225_322591

def base_10_to_12 (n : ℕ) : List ℕ := sorry

theorem base_10_144_equals_base_12_100 :
  base_10_to_12 144 = [1, 0, 0] :=
sorry

end NUMINAMATH_CALUDE_base_10_144_equals_base_12_100_l3225_322591


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3225_322524

theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) : 
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  (∀ n, S n = a 1 * (1 - q^n) / (1 - q)) →  -- sum formula
  q = 1/2 →  -- given ratio
  S 4 / a 4 = 15 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3225_322524


namespace NUMINAMATH_CALUDE_second_largest_is_seven_l3225_322505

def numbers : Finset ℕ := {5, 8, 4, 3, 7}

theorem second_largest_is_seven :
  ∃ (x : ℕ), x ∈ numbers ∧ x > 7 ∧ ∀ y ∈ numbers, y ≠ x → y ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_second_largest_is_seven_l3225_322505


namespace NUMINAMATH_CALUDE_quiz_correct_answers_l3225_322528

theorem quiz_correct_answers (total : ℕ) (difference : ℕ) (sang_hyeon : ℕ) : 
  total = sang_hyeon + (sang_hyeon + difference) → 
  difference = 5 → 
  total = 43 → 
  sang_hyeon = 19 := by sorry

end NUMINAMATH_CALUDE_quiz_correct_answers_l3225_322528


namespace NUMINAMATH_CALUDE_classics_section_books_l3225_322574

/-- The number of classic authors in Jack's collection -/
def num_authors : ℕ := 6

/-- The number of books per author -/
def books_per_author : ℕ := 33

/-- The total number of books in Jack's classics section -/
def total_books : ℕ := num_authors * books_per_author

theorem classics_section_books :
  total_books = 198 :=
by sorry

end NUMINAMATH_CALUDE_classics_section_books_l3225_322574


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3225_322522

theorem sufficient_not_necessary_condition : 
  (∀ x : ℝ, x > 3 → x^2 > 4) ∧ 
  (∃ x : ℝ, x^2 > 4 ∧ ¬(x > 3)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3225_322522


namespace NUMINAMATH_CALUDE_rug_overlap_problem_l3225_322538

/-- Given three rugs with total area A, overlapped to cover floor area F,
    with S2 area covered by exactly two layers, prove the area S3
    covered by three layers. -/
theorem rug_overlap_problem (A F S2 : ℝ) (hA : A = 200) (hF : F = 138) (hS2 : S2 = 24) :
  ∃ S1 S3 : ℝ,
    S1 + S2 + S3 = F ∧
    S1 + 2 * S2 + 3 * S3 = A ∧
    S3 = 19 := by
  sorry

end NUMINAMATH_CALUDE_rug_overlap_problem_l3225_322538


namespace NUMINAMATH_CALUDE_equation_equivalence_l3225_322523

theorem equation_equivalence : ∀ x y : ℝ, (5 * x - y = 6) ↔ (y = 5 * x - 6) := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3225_322523


namespace NUMINAMATH_CALUDE_unique_m_opens_downwards_l3225_322509

/-- A function f(x) = (m + 1)x^(|m|) that opens downwards -/
def opens_downwards (m : ℝ) : Prop :=
  (abs m = 2) ∧ (m + 1 < 0)

/-- The unique value of m for which the function opens downwards is -2 -/
theorem unique_m_opens_downwards :
  ∃! m : ℝ, opens_downwards m :=
sorry

end NUMINAMATH_CALUDE_unique_m_opens_downwards_l3225_322509


namespace NUMINAMATH_CALUDE_valid_three_digit_numbers_count_l3225_322594

/-- The count of three-digit numbers where either all digits are the same or the first and last digits are different -/
def validThreeDigitNumbers : ℕ :=
  -- Total three-digit numbers
  let totalThreeDigitNumbers := 999 - 100 + 1
  -- Numbers to exclude (ABA form where A ≠ B and B ≠ 0)
  let excludedNumbers := 10 * 9
  -- Calculation
  totalThreeDigitNumbers - excludedNumbers

/-- Theorem stating that the count of valid three-digit numbers is 810 -/
theorem valid_three_digit_numbers_count : validThreeDigitNumbers = 810 := by
  sorry


end NUMINAMATH_CALUDE_valid_three_digit_numbers_count_l3225_322594


namespace NUMINAMATH_CALUDE_choose_team_with_smaller_variance_l3225_322565

-- Define the teams
inductive Team
  | A
  | B

-- Define the properties of the teams
def average_height : ℝ := 1.72
def variance (t : Team) : ℝ :=
  match t with
  | Team.A => 1.2
  | Team.B => 5.6

-- Define a function to determine which team has more uniform heights
def more_uniform_heights (t1 t2 : Team) : Prop :=
  variance t1 < variance t2

-- Theorem statement
theorem choose_team_with_smaller_variance :
  more_uniform_heights Team.A Team.B :=
sorry

end NUMINAMATH_CALUDE_choose_team_with_smaller_variance_l3225_322565


namespace NUMINAMATH_CALUDE_clock_coincidences_l3225_322568

/-- Represents a clock with minute and hour hands -/
structure Clock :=
  (minuteRotations : ℕ) -- Number of full rotations of minute hand in 12 hours
  (hourRotations : ℕ)   -- Number of full rotations of hour hand in 12 hours

/-- The standard 12-hour clock -/
def standardClock : Clock :=
  { minuteRotations := 12,
    hourRotations := 1 }

/-- Number of coincidences between minute and hour hands in 12 hours -/
def coincidences (c : Clock) : ℕ :=
  c.minuteRotations - c.hourRotations

/-- Interval between coincidences in minutes -/
def coincidenceInterval (c : Clock) : ℚ :=
  (12 * 60) / (coincidences c)

theorem clock_coincidences (c : Clock) :
  c = standardClock →
  coincidences c = 11 ∧
  coincidenceInterval c = 65 + 5/11 :=
sorry

end NUMINAMATH_CALUDE_clock_coincidences_l3225_322568


namespace NUMINAMATH_CALUDE_cubic_equation_root_l3225_322583

theorem cubic_equation_root (a b : ℚ) : 
  (2 + Real.sqrt 3 : ℝ) ^ 3 + a * (2 + Real.sqrt 3 : ℝ) ^ 2 + b * (2 + Real.sqrt 3 : ℝ) - 15 = 0 → 
  b = -44 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l3225_322583


namespace NUMINAMATH_CALUDE_origin_outside_circle_iff_a_in_range_l3225_322572

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 2*y + a - 2 = 0

/-- A point is outside a circle if the left side of the equation is positive when substituting the point's coordinates -/
def point_outside_circle (x y a : ℝ) : Prop :=
  x^2 + y^2 + 2*y + a - 2 > 0

theorem origin_outside_circle_iff_a_in_range (a : ℝ) :
  point_outside_circle 0 0 a ↔ 2 < a ∧ a < 3 :=
sorry

end NUMINAMATH_CALUDE_origin_outside_circle_iff_a_in_range_l3225_322572


namespace NUMINAMATH_CALUDE_six_solutions_l3225_322559

/-- The number of ordered pairs of positive integers (m,n) satisfying 6/m + 3/n = 1 -/
def solution_count : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    let (m, n) := p
    m > 0 ∧ n > 0 ∧ 6 * n + 3 * m = m * n) (Finset.product (Finset.range 25) (Finset.range 22))).card

/-- The theorem stating that there are exactly 6 solutions -/
theorem six_solutions : solution_count = 6 := by
  sorry


end NUMINAMATH_CALUDE_six_solutions_l3225_322559


namespace NUMINAMATH_CALUDE_min_value_theorem_l3225_322564

theorem min_value_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3225_322564


namespace NUMINAMATH_CALUDE_triangle_area_change_l3225_322521

theorem triangle_area_change (base height : ℝ) (base_new height_new area area_new : ℝ) 
  (h1 : base_new = 1.10 * base) 
  (h2 : height_new = 0.95 * height) 
  (h3 : area = (base * height) / 2) 
  (h4 : area_new = (base_new * height_new) / 2) :
  area_new = 1.045 * area := by
  sorry

#check triangle_area_change

end NUMINAMATH_CALUDE_triangle_area_change_l3225_322521


namespace NUMINAMATH_CALUDE_puzzle_pieces_count_l3225_322518

theorem puzzle_pieces_count :
  let border_pieces : ℕ := 75
  let trevor_pieces : ℕ := 105
  let joe_pieces : ℕ := 3 * trevor_pieces
  let missing_pieces : ℕ := 5
  let total_pieces : ℕ := border_pieces + trevor_pieces + joe_pieces + missing_pieces
  total_pieces = 500 := by
sorry

end NUMINAMATH_CALUDE_puzzle_pieces_count_l3225_322518


namespace NUMINAMATH_CALUDE_zoo_revenue_calculation_l3225_322515

/-- Calculates the total revenue for a zoo over two days with given attendance and pricing information --/
def zoo_revenue (
  monday_children monday_adults monday_seniors : ℕ)
  (tuesday_children tuesday_adults tuesday_seniors : ℕ)
  (monday_child_price monday_adult_price monday_senior_price : ℚ)
  (tuesday_child_price tuesday_adult_price tuesday_senior_price : ℚ)
  (tuesday_discount : ℚ) : ℚ :=
  let monday_total := 
    monday_children * monday_child_price + 
    monday_adults * monday_adult_price + 
    monday_seniors * monday_senior_price
  let tuesday_total := 
    tuesday_children * tuesday_child_price + 
    tuesday_adults * tuesday_adult_price + 
    tuesday_seniors * tuesday_senior_price
  let tuesday_discounted := tuesday_total * (1 - tuesday_discount)
  monday_total + tuesday_discounted

theorem zoo_revenue_calculation : 
  zoo_revenue 7 5 3 9 6 2 3 4 3 4 5 3 (1/10) = 114.8 := by
  sorry

end NUMINAMATH_CALUDE_zoo_revenue_calculation_l3225_322515


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_diagonal_intersection_property_l3225_322553

/-- A point in a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A circle in a 2D plane -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- A quadrilateral in a 2D plane -/
structure Quadrilateral :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Checks if a quadrilateral is cyclic -/
def is_cyclic (q : Quadrilateral) (c : Circle) : Prop :=
  -- Definition omitted for brevity
  sorry

/-- Calculates the intersection point of two line segments -/
def intersection (p1 p2 p3 p4 : Point) : Point :=
  -- Definition omitted for brevity
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  -- Definition omitted for brevity
  sorry

/-- The main theorem -/
theorem cyclic_quadrilateral_diagonal_intersection_property
  (ABCD : Quadrilateral) (c : Circle) (X : Point) :
  is_cyclic ABCD c →
  X = intersection ABCD.A ABCD.C ABCD.B ABCD.D →
  distance X ABCD.A * distance X ABCD.C = distance X ABCD.B * distance X ABCD.D :=
by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_diagonal_intersection_property_l3225_322553


namespace NUMINAMATH_CALUDE_simplify_expression_l3225_322589

theorem simplify_expression (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 3) :
  (1 - 2 / (x - 1)) * ((x^2 - x) / (x^2 - 6*x + 9)) = x / (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3225_322589


namespace NUMINAMATH_CALUDE_expression_value_l3225_322510

theorem expression_value : -25 + 5 * (4^2 / 2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3225_322510


namespace NUMINAMATH_CALUDE_amusement_park_problem_l3225_322531

/-- The number of children in the group satisfies the given conditions -/
theorem amusement_park_problem (C : ℕ) : C = 5 ↔ 
  15 + 3 * C + 16 * C = 110 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_problem_l3225_322531


namespace NUMINAMATH_CALUDE_coin_count_theorem_l3225_322556

/-- Represents the types of coins --/
inductive CoinType
  | Penny
  | Nickel
  | Dime
  | Quarter
  | Dollar

/-- The value of each coin type in cents --/
def coinValue (c : CoinType) : ℕ :=
  match c with
  | .Penny => 1
  | .Nickel => 5
  | .Dime => 10
  | .Quarter => 25
  | .Dollar => 100

/-- The set of all coin types --/
def allCoinTypes : List CoinType := [.Penny, .Nickel, .Dime, .Quarter, .Dollar]

theorem coin_count_theorem (n : ℕ) 
    (h1 : n > 0)
    (h2 : (List.sum (List.map (fun c => coinValue c * n) allCoinTypes)) = 351) :
    List.length allCoinTypes * n = 15 := by
  sorry

end NUMINAMATH_CALUDE_coin_count_theorem_l3225_322556


namespace NUMINAMATH_CALUDE_prime_factor_sum_l3225_322537

theorem prime_factor_sum (w x y z : ℕ) : 
  2^w * 3^x * 5^y * 7^z = 2450 → 3*w + 2*x + 7*y + 5*z = 27 := by
  sorry

end NUMINAMATH_CALUDE_prime_factor_sum_l3225_322537


namespace NUMINAMATH_CALUDE_negation_of_existence_tan_equals_one_l3225_322598

theorem negation_of_existence_tan_equals_one :
  (¬ ∃ x : ℝ, Real.tan x = 1) ↔ (∀ x : ℝ, Real.tan x ≠ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_tan_equals_one_l3225_322598


namespace NUMINAMATH_CALUDE_volunteer_assignment_count_l3225_322543

theorem volunteer_assignment_count : 
  (volunteers : ℕ) → 
  (pavilions : ℕ) → 
  volunteers = 5 → 
  pavilions = 4 → 
  (arrangements : ℕ) → 
  arrangements = 240 := by sorry

end NUMINAMATH_CALUDE_volunteer_assignment_count_l3225_322543


namespace NUMINAMATH_CALUDE_total_profit_is_840_l3225_322551

/-- Represents the investment and profit details of a business partnership --/
structure BusinessPartnership where
  initial_investment_A : ℕ
  initial_investment_B : ℕ
  withdrawal_A : ℕ
  addition_B : ℕ
  months_before_change : ℕ
  total_months : ℕ
  profit_share_A : ℕ

/-- Calculates the total profit given the business partnership details --/
def calculate_total_profit (bp : BusinessPartnership) : ℕ :=
  sorry

/-- Theorem stating that given the specific investment pattern, if A's profit share is 320,
    then the total profit is 840 --/
theorem total_profit_is_840 (bp : BusinessPartnership)
  (h1 : bp.initial_investment_A = 3000)
  (h2 : bp.initial_investment_B = 4000)
  (h3 : bp.withdrawal_A = 1000)
  (h4 : bp.addition_B = 1000)
  (h5 : bp.months_before_change = 8)
  (h6 : bp.total_months = 12)
  (h7 : bp.profit_share_A = 320) :
  calculate_total_profit bp = 840 :=
  sorry

end NUMINAMATH_CALUDE_total_profit_is_840_l3225_322551


namespace NUMINAMATH_CALUDE_parabola_equation_l3225_322508

-- Define the parabola
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define the line
def line (x y : ℝ) : Prop := 3 * x - 4 * y - 24 = 0

-- Theorem statement
theorem parabola_equation (p : Parabola) :
  (∀ x y, p.equation x y ↔ x^2 = 2 * y) →  -- Standard form of parabola with vertex at origin and y-axis as axis of symmetry
  (∃ x y, p.equation x y ∧ line x y) →     -- Focus lies on the given line
  (∀ x y, p.equation x y ↔ x^2 = -24 * y)  -- Conclusion: The standard equation is x² = -24y
  := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l3225_322508


namespace NUMINAMATH_CALUDE_hat_shoppe_pricing_l3225_322552

theorem hat_shoppe_pricing (x : ℝ) (h : x > 0) : 
  0.75 * (1.3 * x) = 0.975 * x := by
  sorry

end NUMINAMATH_CALUDE_hat_shoppe_pricing_l3225_322552


namespace NUMINAMATH_CALUDE_cube_plane_difference_l3225_322578

/-- Represents a cube with points placed on each face -/
structure MarkedCube where
  -- Add necessary fields

/-- Represents a plane intersecting the cube -/
structure IntersectingPlane where
  -- Add necessary fields

/-- Represents a segment on the surface of the cube -/
structure SurfaceSegment where
  -- Add necessary fields

/-- The maximum number of planes required to create all possible segments -/
def max_planes (cube : MarkedCube) : ℕ := sorry

/-- The minimum number of planes required to create all possible segments -/
def min_planes (cube : MarkedCube) : ℕ := sorry

/-- All possible segments on the surface of the cube -/
def all_segments (cube : MarkedCube) : Set SurfaceSegment := sorry

/-- The set of segments created by a given set of planes -/
def segments_from_planes (cube : MarkedCube) (planes : Set IntersectingPlane) : Set SurfaceSegment := sorry

theorem cube_plane_difference (cube : MarkedCube) :
  max_planes cube - min_planes cube = 24 :=
sorry

end NUMINAMATH_CALUDE_cube_plane_difference_l3225_322578


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sine_problem_l3225_322558

theorem arithmetic_sequence_sine_problem (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 5 + a 6 = 10 * Real.pi / 3 →                    -- given condition
  Real.sin (a 4 + a 7) = -Real.sqrt 3 / 2 :=        -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sine_problem_l3225_322558


namespace NUMINAMATH_CALUDE_charlie_metal_purchase_l3225_322561

/-- Given that Charlie needs a total amount of metal and has some in storage,
    this function calculates the additional amount he needs to buy. -/
def additional_metal_needed (total_needed : ℕ) (in_storage : ℕ) : ℕ :=
  total_needed - in_storage

/-- Theorem stating that given Charlie's specific situation, 
    he needs to buy 359 lbs of additional metal. -/
theorem charlie_metal_purchase : 
  additional_metal_needed 635 276 = 359 := by sorry

end NUMINAMATH_CALUDE_charlie_metal_purchase_l3225_322561


namespace NUMINAMATH_CALUDE_ratio_of_repeating_decimals_l3225_322557

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ := (10 * a + b : ℚ) / 99

/-- The main theorem stating that the ratio of 0.overline{63} to 0.overline{21} is 3 -/
theorem ratio_of_repeating_decimals : 
  (RepeatingDecimal 6 3) / (RepeatingDecimal 2 1) = 3 := by sorry

end NUMINAMATH_CALUDE_ratio_of_repeating_decimals_l3225_322557


namespace NUMINAMATH_CALUDE_sphere_radius_l3225_322554

theorem sphere_radius (A : ℝ) (h : A = 64 * Real.pi) :
  ∃ (r : ℝ), A = 4 * Real.pi * r^2 ∧ r = 4 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_l3225_322554


namespace NUMINAMATH_CALUDE_fog_sum_l3225_322539

theorem fog_sum (f o g : Nat) : 
  f < 10 → o < 10 → g < 10 →
  (100 * f + 10 * o + g) * 4 = 1464 →
  f + o + g = 15 := by
sorry

end NUMINAMATH_CALUDE_fog_sum_l3225_322539


namespace NUMINAMATH_CALUDE_plane_equation_is_correct_l3225_322573

/-- The line passing through (2,4,-3) with direction vector (4,-1,5) -/
def line (t : ℝ) : ℝ × ℝ × ℝ := (2 + 4*t, 4 - t, -3 + 5*t)

/-- The point that the plane passes through -/
def point : ℝ × ℝ × ℝ := (1, 6, -8)

/-- The coefficients of the plane equation -/
def plane_coeff : ℤ × ℤ × ℤ × ℤ := (5, 15, -7, 151)

theorem plane_equation_is_correct :
  let (A, B, C, D) := plane_coeff
  (A > 0) ∧ 
  (Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1) ∧
  (∀ (x y z : ℝ), A * x + B * y + C * z - D = 0 ↔ 
    (∃ (t : ℝ), (x, y, z) = line t) ∨ (x, y, z) = point) := by sorry

end NUMINAMATH_CALUDE_plane_equation_is_correct_l3225_322573


namespace NUMINAMATH_CALUDE_point_on_line_l3225_322546

/-- A line in the xy-plane with slope m and y-intercept b -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

/-- Theorem: For a line with slope 4 and y-intercept 4, 
    the point (199, 800) lies on this line -/
theorem point_on_line : 
  let l : Line := { m := 4, b := 4 }
  let p : Point := { x := 199, y := 800 }
  p.onLine l := by sorry

end NUMINAMATH_CALUDE_point_on_line_l3225_322546


namespace NUMINAMATH_CALUDE_derivative_at_alpha_l3225_322586

open Real

theorem derivative_at_alpha (α : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 2 * cos α - sin x
  HasDerivAt f (-cos α) α := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_alpha_l3225_322586


namespace NUMINAMATH_CALUDE_four_heads_before_three_tails_l3225_322567

/-- The probability of encountering 4 consecutive heads before 3 consecutive tails in repeated fair coin flips -/
def q : ℚ := 16/23

/-- A fair coin has equal probability of heads and tails -/
def fair_coin (p : ℚ → Prop) : Prop := p (1/2)

theorem four_heads_before_three_tails (fair_coin : (ℚ → Prop) → Prop) : q = 16/23 := by
  sorry

end NUMINAMATH_CALUDE_four_heads_before_three_tails_l3225_322567


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3225_322590

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^(4*x+2) * (4 : ℝ)^(2*x+3) = (8 : ℝ)^(3*x+4) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3225_322590


namespace NUMINAMATH_CALUDE_pigeonhole_sum_to_ten_l3225_322580

theorem pigeonhole_sum_to_ten :
  ∀ (S : Finset ℕ), 
    (∀ x ∈ S, 1 ≤ x ∧ x ≤ 10) → 
    S.card ≥ 7 → 
    ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a + b = 10 :=
by sorry

end NUMINAMATH_CALUDE_pigeonhole_sum_to_ten_l3225_322580


namespace NUMINAMATH_CALUDE_arccos_sin_one_l3225_322533

theorem arccos_sin_one : Real.arccos (Real.sin 1) = π / 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sin_one_l3225_322533


namespace NUMINAMATH_CALUDE_teairra_clothing_count_l3225_322513

/-- The number of shirts and pants Teairra has which are neither plaid nor purple -/
def non_plaid_purple_count (total_shirts : ℕ) (total_pants : ℕ) (plaid_shirts : ℕ) (purple_pants : ℕ) : ℕ :=
  (total_shirts - plaid_shirts) + (total_pants - purple_pants)

/-- Theorem stating that Teairra has 21 items that are neither plaid nor purple -/
theorem teairra_clothing_count : non_plaid_purple_count 5 24 3 5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_teairra_clothing_count_l3225_322513


namespace NUMINAMATH_CALUDE_delores_initial_amount_l3225_322532

/-- The amount of money Delores had initially -/
def initial_amount : ℕ := sorry

/-- The cost of the computer -/
def computer_cost : ℕ := 400

/-- The cost of the printer -/
def printer_cost : ℕ := 40

/-- The amount of money Delores had left after purchases -/
def remaining_amount : ℕ := 10

/-- Theorem stating that Delores' initial amount was $450 -/
theorem delores_initial_amount : 
  initial_amount = computer_cost + printer_cost + remaining_amount := by sorry

end NUMINAMATH_CALUDE_delores_initial_amount_l3225_322532


namespace NUMINAMATH_CALUDE_smallest_product_l3225_322547

def S : Finset Int := {-7, -5, -1, 1, 3}

theorem smallest_product (a b : Int) (ha : a ∈ S) (hb : b ∈ S) :
  ∃ (x y : Int) (hx : x ∈ S) (hy : y ∈ S), x * y = -21 ∧ ∀ (c d : Int), c ∈ S → d ∈ S → x * y ≤ c * d :=
sorry

end NUMINAMATH_CALUDE_smallest_product_l3225_322547


namespace NUMINAMATH_CALUDE_age_problem_solution_l3225_322549

/-- Represents the ages of James and Joe -/
structure Ages where
  james : ℕ
  joe : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.joe = ages.james + 10 ∧
  2 * (ages.joe + 8) = 3 * (ages.james + 8)

/-- The theorem to prove -/
theorem age_problem_solution :
  ∃ (ages : Ages), satisfiesConditions ages ∧ ages.james = 12 ∧ ages.joe = 22 := by
  sorry


end NUMINAMATH_CALUDE_age_problem_solution_l3225_322549


namespace NUMINAMATH_CALUDE_basic_astrophysics_degrees_l3225_322581

/-- Represents the allocation of a research and development budget in a circle graph -/
structure BudgetAllocation where
  microphotonics : ℝ
  home_electronics : ℝ
  food_additives : ℝ
  genetically_modified_microorganisms : ℝ
  industrial_lubricants : ℝ

/-- The theorem stating that the remaining sector (basic astrophysics) occupies 90 degrees of the circle -/
theorem basic_astrophysics_degrees (budget : BudgetAllocation) : 
  budget.microphotonics = 14 ∧ 
  budget.home_electronics = 19 ∧ 
  budget.food_additives = 10 ∧ 
  budget.genetically_modified_microorganisms = 24 ∧ 
  budget.industrial_lubricants = 8 → 
  (100 - (budget.microphotonics + budget.home_electronics + budget.food_additives + 
          budget.genetically_modified_microorganisms + budget.industrial_lubricants)) / 100 * 360 = 90 :=
by sorry

end NUMINAMATH_CALUDE_basic_astrophysics_degrees_l3225_322581


namespace NUMINAMATH_CALUDE_banana_permutations_l3225_322534

/-- The number of distinct permutations of letters in a word with repeated letters -/
def distinctPermutations (totalLetters : ℕ) (repeatedLetters : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repeatedLetters.map Nat.factorial).prod

/-- The number of distinct permutations of the letters in "BANANA" -/
theorem banana_permutations :
  distinctPermutations 6 [3, 2] = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_l3225_322534


namespace NUMINAMATH_CALUDE_f_4_1981_l3225_322530

/-- A function satisfying the given recursive properties -/
def f : ℕ → ℕ → ℕ
| 0, y => y + 1
| x + 1, 0 => f x 1
| x + 1, y + 1 => f x (f (x + 1) y)

/-- Power tower of 2 with given height -/
def power_tower_2 : ℕ → ℕ
| 0 => 1
| n + 1 => 2^(power_tower_2 n)

/-- The main theorem to prove -/
theorem f_4_1981 : f 4 1981 = power_tower_2 1984 - 3 := by
  sorry

end NUMINAMATH_CALUDE_f_4_1981_l3225_322530


namespace NUMINAMATH_CALUDE_last_day_is_monday_l3225_322596

/-- 
Given a year with 365 days, if the 15th day falls on a Monday,
then the 365th day also falls on a Monday.
-/
theorem last_day_is_monday (year : ℕ) : 
  year % 7 = 1 → -- Assuming Monday is represented by 1
  (365 % 7 = year % 7) → -- The last day falls on the same day as the first
  (15 % 7 = 1) → -- The 15th day is a Monday
  (365 % 7 = 1) -- The 365th day is also a Monday
:= by sorry

end NUMINAMATH_CALUDE_last_day_is_monday_l3225_322596


namespace NUMINAMATH_CALUDE_number_solution_l3225_322587

theorem number_solution : ∃ x : ℝ, (45 - 3 * x = 18) ∧ (x = 9) := by sorry

end NUMINAMATH_CALUDE_number_solution_l3225_322587


namespace NUMINAMATH_CALUDE_power_of_three_mod_eight_l3225_322541

theorem power_of_three_mod_eight : 3^2023 % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_eight_l3225_322541


namespace NUMINAMATH_CALUDE_cubic_sum_given_sum_and_product_l3225_322576

theorem cubic_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 10) (h2 : x * y = 16) : x^3 + y^3 = 520 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_given_sum_and_product_l3225_322576


namespace NUMINAMATH_CALUDE_spade_calculation_l3225_322592

def spade (a b : ℝ) : ℝ := |a - b|

theorem spade_calculation : spade (spade 3 5) (spade 6 9) = 1 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l3225_322592


namespace NUMINAMATH_CALUDE_quadratic_polynomial_integer_root_exists_l3225_322588

/-- Represents a quadratic polynomial ax^2 + bx + c -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Evaluates a quadratic polynomial at x = -1 -/
def evalAtNegativeOne (p : QuadraticPolynomial) : ℤ :=
  p.a + -p.b + p.c

/-- Represents a single step change in the polynomial -/
inductive PolynomialStep
  | ChangeX : (δ : ℤ) → PolynomialStep
  | ChangeConstant : (δ : ℤ) → PolynomialStep

/-- Applies a step to a polynomial -/
def applyStep (p : QuadraticPolynomial) (step : PolynomialStep) : QuadraticPolynomial :=
  match step with
  | PolynomialStep.ChangeX δ => ⟨p.a, p.b + δ, p.c⟩
  | PolynomialStep.ChangeConstant δ => ⟨p.a, p.b, p.c + δ⟩

theorem quadratic_polynomial_integer_root_exists 
  (initial : QuadraticPolynomial)
  (final : QuadraticPolynomial)
  (h_initial : initial = ⟨1, 10, 20⟩)
  (h_final : final = ⟨1, 20, 10⟩)
  (steps : List PolynomialStep)
  (h_steps : ∀ step ∈ steps, 
    (∃ δ, step = PolynomialStep.ChangeX δ ∧ (δ = 1 ∨ δ = -1)) ∨
    (∃ δ, step = PolynomialStep.ChangeConstant δ ∧ (δ = 1 ∨ δ = -1)))
  (h_transform : final = steps.foldl applyStep initial) :
  ∃ p : QuadraticPolynomial, p ∈ initial :: (List.scanl applyStep initial steps) ∧ 
    ∃ x : ℤ, p.a * x^2 + p.b * x + p.c = 0 :=
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_integer_root_exists_l3225_322588


namespace NUMINAMATH_CALUDE_optimal_stamp_combination_l3225_322512

/-- The minimum number of stamps needed to make 50 cents using only 5-cent and 7-cent stamps -/
def min_stamps : ℕ := 8

/-- The number of 5-cent stamps used in the optimal solution -/
def num_5cent : ℕ := 3

/-- The number of 7-cent stamps used in the optimal solution -/
def num_7cent : ℕ := 5

theorem optimal_stamp_combination :
  (∀ x y : ℕ, 5 * x + 7 * y = 50 → x + y ≥ min_stamps) ∧
  5 * num_5cent + 7 * num_7cent = 50 ∧
  num_5cent + num_7cent = min_stamps := by
  sorry

end NUMINAMATH_CALUDE_optimal_stamp_combination_l3225_322512


namespace NUMINAMATH_CALUDE_custom_op_result_l3225_322545

/-- Define the custom operation * -/
def custom_op (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y + 1

/-- Theorem stating the result of the custom operation given the conditions -/
theorem custom_op_result (a b : ℝ) :
  (custom_op a b 3 5 = 15) →
  (custom_op a b 4 7 = 28) →
  (custom_op a b 1 1 = -11) := by
  sorry

end NUMINAMATH_CALUDE_custom_op_result_l3225_322545


namespace NUMINAMATH_CALUDE_max_volume_triangular_pyramid_l3225_322516

/-- A triangular pyramid P-ABC with given side lengths -/
structure TriangularPyramid where
  PA : ℝ
  PB : ℝ
  AB : ℝ
  BC : ℝ
  CA : ℝ

/-- The volume of a triangular pyramid -/
def volume (t : TriangularPyramid) : ℝ := sorry

/-- The maximum volume of a triangular pyramid with specific side lengths -/
theorem max_volume_triangular_pyramid :
  ∀ t : TriangularPyramid,
  t.PA = 3 ∧ t.PB = 3 ∧ t.AB = 2 ∧ t.BC = 2 ∧ t.CA = 2 →
  volume t ≤ 2 * Real.sqrt 6 / 3 :=
sorry

end NUMINAMATH_CALUDE_max_volume_triangular_pyramid_l3225_322516


namespace NUMINAMATH_CALUDE_product_remainder_l3225_322599

theorem product_remainder (n : ℤ) : (12 - 2*n) * (n + 5) ≡ -2*n^2 + 2*n + 5 [ZMOD 11] := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l3225_322599


namespace NUMINAMATH_CALUDE_square_of_8y_minus_2_l3225_322540

theorem square_of_8y_minus_2 (y : ℝ) (h : 4 * y^2 + 7 = 6 * y + 12) :
  (8 * y - 2)^2 = 248 := by
  sorry

end NUMINAMATH_CALUDE_square_of_8y_minus_2_l3225_322540


namespace NUMINAMATH_CALUDE_followers_exceed_thousand_l3225_322582

/-- 
Given that Daniel starts with 5 followers on Sunday and his followers triple each day,
this theorem proves that Saturday (6 days after Sunday) is the first day 
when Daniel has more than 1000 followers.
-/
theorem followers_exceed_thousand (k : ℕ) : 
  (∀ n < k, 5 * 3^n ≤ 1000) ∧ 5 * 3^k > 1000 → k = 6 := by
  sorry

end NUMINAMATH_CALUDE_followers_exceed_thousand_l3225_322582


namespace NUMINAMATH_CALUDE_michaels_book_purchase_l3225_322519

theorem michaels_book_purchase (m : ℚ) : 
  (∃ (n : ℚ), (1 / 3 : ℚ) * m = (1 / 2 : ℚ) * n * ((1 / 3 : ℚ) * m / ((1 / 2 : ℚ) * n))) →
  (5 : ℚ) = (1 / 15 : ℚ) * m →
  m - ((2 / 3 : ℚ) * m + (1 / 15 : ℚ) * m) = (4 / 15 : ℚ) * m :=
by sorry

end NUMINAMATH_CALUDE_michaels_book_purchase_l3225_322519


namespace NUMINAMATH_CALUDE_large_triangle_toothpicks_l3225_322544

/-- The number of small triangles in the base row of the large equilateral triangle -/
def base_triangles : ℕ := 100

/-- The total number of small triangles in the large equilateral triangle -/
def total_triangles : ℕ := (base_triangles * (base_triangles + 1)) / 2

/-- The number of toothpicks required to assemble the large equilateral triangle -/
def toothpicks_required : ℕ := ((3 * total_triangles) / 2) + (3 * base_triangles)

theorem large_triangle_toothpicks :
  toothpicks_required = 7875 := by sorry

end NUMINAMATH_CALUDE_large_triangle_toothpicks_l3225_322544


namespace NUMINAMATH_CALUDE_bicycle_oil_requirement_l3225_322597

/-- The amount of oil needed to fix a bicycle -/
theorem bicycle_oil_requirement (wheel_count : ℕ) (oil_per_wheel : ℕ) (oil_for_rest : ℕ) : 
  wheel_count = 2 → oil_per_wheel = 10 → oil_for_rest = 5 →
  wheel_count * oil_per_wheel + oil_for_rest = 25 := by
  sorry

#check bicycle_oil_requirement

end NUMINAMATH_CALUDE_bicycle_oil_requirement_l3225_322597


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3225_322579

theorem quadratic_factorization (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3225_322579


namespace NUMINAMATH_CALUDE_f_geq_a_iff_a_in_range_l3225_322502

/-- The function f(x) = x^2 - 2ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

/-- The domain of x -/
def domain : Set ℝ := {x | x ≥ -1}

/-- The theorem stating the condition for a -/
theorem f_geq_a_iff_a_in_range (a : ℝ) : 
  (∀ x ∈ domain, f a x ≥ a) ↔ -3 ≤ a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_f_geq_a_iff_a_in_range_l3225_322502


namespace NUMINAMATH_CALUDE_parabola_focus_focus_of_specific_parabola_l3225_322570

/-- The focus of a parabola y = ax^2 + k is at (0, 1/(4a) + k) -/
theorem parabola_focus (a k : ℝ) (h : a ≠ 0) :
  let f : ℝ × ℝ := (0, 1/(4*a) + k)
  ∀ x y : ℝ, y = a * x^2 + k → (x - f.1)^2 + (y - f.2)^2 = (y - k + 1/(4*a))^2 :=
sorry

/-- The focus of the parabola y = 9x^2 + 6 is at (0, 217/36) -/
theorem focus_of_specific_parabola :
  let f : ℝ × ℝ := (0, 217/36)
  ∀ x y : ℝ, y = 9 * x^2 + 6 → (x - f.1)^2 + (y - f.2)^2 = (y - 6 + 1/36)^2 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_focus_of_specific_parabola_l3225_322570


namespace NUMINAMATH_CALUDE_equation_solution_l3225_322511

theorem equation_solution :
  ∃ x : ℤ, 45 - (x - (37 - (15 - 18))) = 57 ∧ x = 28 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3225_322511


namespace NUMINAMATH_CALUDE_minimum_nickels_needed_l3225_322595

/-- The cost of the sneakers in dollars -/
def sneaker_cost : ℚ := 45.5

/-- The number of $10 bills Chloe has -/
def ten_dollar_bills : ℕ := 4

/-- The number of quarters Chloe has -/
def quarters : ℕ := 5

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The minimum number of nickels needed -/
def min_nickels : ℕ := 85

theorem minimum_nickels_needed :
  ∀ n : ℕ,
  (n : ℚ) * nickel_value + (ten_dollar_bills * 10 : ℚ) + (quarters * 0.25 : ℚ) ≥ sneaker_cost →
  n ≥ min_nickels :=
by sorry

end NUMINAMATH_CALUDE_minimum_nickels_needed_l3225_322595


namespace NUMINAMATH_CALUDE_quadratic_function_problem_l3225_322562

/-- Given a quadratic function f(x) = x^2 + ax + b, if f(f(x) + x) / f(x) = x^2 + 2023x + 3000,
    then a = 2021 and b = 979. -/
theorem quadratic_function_problem (a b : ℝ) : 
  (let f := fun x => x^2 + a*x + b
   (∀ x, (f (f x + x)) / (f x) = x^2 + 2023*x + 3000)) → 
  (a = 2021 ∧ b = 979) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_problem_l3225_322562


namespace NUMINAMATH_CALUDE_book_cost_problem_l3225_322584

theorem book_cost_problem (cost_of_three : ℝ) (h : cost_of_three = 45) :
  let cost_of_one : ℝ := cost_of_three / 3
  let cost_of_seven : ℝ := 7 * cost_of_one
  cost_of_seven = 105 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_problem_l3225_322584


namespace NUMINAMATH_CALUDE_sum_of_ages_matt_age_relation_l3225_322563

/-- Given Matt's age and John's age, prove the sum of their ages -/
theorem sum_of_ages (matt_age john_age : ℕ) (h1 : matt_age = 41) (h2 : john_age = 11) :
  matt_age + john_age = 52 := by
  sorry

/-- Matt's age in relation to John's -/
theorem matt_age_relation (matt_age john_age : ℕ) (h1 : matt_age = 41) (h2 : john_age = 11) :
  matt_age = 4 * john_age - 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_matt_age_relation_l3225_322563


namespace NUMINAMATH_CALUDE_diana_reading_time_l3225_322542

/-- The number of hours Diana read this week -/
def hours_read : ℝ := 12

/-- The initial reward rate in minutes per hour -/
def initial_rate : ℝ := 30

/-- The percentage increase in the reward rate -/
def rate_increase : ℝ := 0.2

/-- The total increase in video game time due to the raise in minutes -/
def total_increase : ℝ := 72

theorem diana_reading_time :
  hours_read * initial_rate * rate_increase = total_increase := by
  sorry

end NUMINAMATH_CALUDE_diana_reading_time_l3225_322542


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3225_322507

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  a/(a-1) + 4*b/(b-1) ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 1/a₀ + 1/b₀ = 1 ∧ a₀/(a₀-1) + 4*b₀/(b₀-1) = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3225_322507


namespace NUMINAMATH_CALUDE_potato_sale_revenue_l3225_322517

/-- Calculates the revenue from selling potatoes given the total weight, damaged weight, bag size, and price per bag. -/
def potato_revenue (total_weight damaged_weight bag_size price_per_bag : ℕ) : ℕ :=
  let sellable_weight := total_weight - damaged_weight
  let num_bags := sellable_weight / bag_size
  num_bags * price_per_bag

/-- Theorem stating that the revenue from selling potatoes under given conditions is $9144. -/
theorem potato_sale_revenue :
  potato_revenue 6500 150 50 72 = 9144 := by
  sorry

end NUMINAMATH_CALUDE_potato_sale_revenue_l3225_322517


namespace NUMINAMATH_CALUDE_apple_basket_problem_l3225_322501

def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem apple_basket_problem (total_apples : ℕ) (first_basket : ℕ) (increment : ℕ) :
  total_apples = 495 →
  first_basket = 25 →
  increment = 2 →
  ∃ x : ℕ, x = 13 ∧ arithmetic_sum first_basket increment x = total_apples :=
by sorry

end NUMINAMATH_CALUDE_apple_basket_problem_l3225_322501


namespace NUMINAMATH_CALUDE_fraction_simplification_l3225_322585

theorem fraction_simplification :
  (1 / 2 + 1 / 3) / (3 / 7 - 1 / 5) = 175 / 48 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3225_322585


namespace NUMINAMATH_CALUDE_unique_factorization_l3225_322566

/-- A factorization of 2210 into a two-digit and a three-digit number -/
structure Factorization :=
  (a : ℕ) (b : ℕ)
  (h1 : 10 ≤ a ∧ a ≤ 99)
  (h2 : 100 ≤ b ∧ b ≤ 999)
  (h3 : a * b = 2210)

/-- Two factorizations are considered equal if they have the same factors (regardless of order) -/
def factorization_eq (f1 f2 : Factorization) : Prop :=
  (f1.a = f2.a ∧ f1.b = f2.b) ∨ (f1.a = f2.b ∧ f1.b = f2.a)

/-- The set of all valid factorizations of 2210 -/
def factorizations : Set Factorization :=
  {f : Factorization | true}

theorem unique_factorization : ∃! (f : Factorization), f ∈ factorizations :=
sorry

end NUMINAMATH_CALUDE_unique_factorization_l3225_322566


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l3225_322535

theorem quadratic_roots_sum (α β : ℝ) : 
  (α^2 - 3*α - 1 = 0) → 
  (β^2 - 3*β - 1 = 0) → 
  7 * α^4 + 10 * β^3 = 1093 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l3225_322535


namespace NUMINAMATH_CALUDE_heather_total_distance_l3225_322593

/-- The distance Heather bicycled per day in kilometers -/
def distance_per_day : ℝ := 40.0

/-- The number of days Heather bicycled -/
def number_of_days : ℝ := 8.0

/-- The total distance Heather bicycled -/
def total_distance : ℝ := distance_per_day * number_of_days

theorem heather_total_distance : total_distance = 320.0 := by
  sorry

end NUMINAMATH_CALUDE_heather_total_distance_l3225_322593


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3225_322571

/-- A hyperbola with center at the origin, transverse axis on the y-axis, and one focus at (0, 6) -/
structure Hyperbola where
  center : ℝ × ℝ
  transverse_axis : ℝ → ℝ × ℝ
  focus : ℝ × ℝ
  h_center : center = (0, 0)
  h_transverse : ∀ x, transverse_axis x = (0, x)
  h_focus : focus = (0, 6)

/-- The equation of the hyperbola is y^2 - x^2 = 18 -/
theorem hyperbola_equation (h : Hyperbola) : 
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | y^2 - x^2 = 18} ↔ 
  ∃ t : ℝ, h.transverse_axis t = (x, y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3225_322571


namespace NUMINAMATH_CALUDE_max_mn_and_min_4m2_plus_n2_l3225_322526

theorem max_mn_and_min_4m2_plus_n2 (m n : ℝ) 
  (h1 : m > 0) (h2 : n > 0) (h3 : 2 * m + n = 1) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 → m * n ≥ x * y) ∧
  (m * n = 1/8) ∧
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 → 4 * m^2 + n^2 ≤ 4 * x^2 + y^2) ∧
  (4 * m^2 + n^2 = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_max_mn_and_min_4m2_plus_n2_l3225_322526


namespace NUMINAMATH_CALUDE_equation_solution_l3225_322555

theorem equation_solution (x y A : ℝ) : 
  (x + y)^3 - x*y*(x + y) = (x + y) * A → A = x^2 + x*y + y^2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3225_322555


namespace NUMINAMATH_CALUDE_rectangle_area_l3225_322575

theorem rectangle_area (width : ℝ) (length : ℝ) (area : ℝ) : 
  width = 5 →
  length = 2 * width →
  area = length * width →
  area = 50 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3225_322575


namespace NUMINAMATH_CALUDE_betty_age_l3225_322569

/-- Represents the ages of Albert, Mary, and Betty --/
structure Ages where
  albert : ℕ
  mary : ℕ
  betty : ℕ

/-- The conditions given in the problem --/
def age_conditions (ages : Ages) : Prop :=
  ages.albert = 2 * ages.mary ∧
  ages.albert = 4 * ages.betty ∧
  ages.mary = ages.albert - 22

/-- The theorem stating Betty's age --/
theorem betty_age (ages : Ages) (h : age_conditions ages) : ages.betty = 11 := by
  sorry

end NUMINAMATH_CALUDE_betty_age_l3225_322569


namespace NUMINAMATH_CALUDE_simplify_negative_cube_squared_l3225_322529

theorem simplify_negative_cube_squared (a : ℝ) : (-a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_negative_cube_squared_l3225_322529


namespace NUMINAMATH_CALUDE_tournament_probability_l3225_322525

/-- Represents a soccer tournament with the given conditions -/
structure SoccerTournament where
  num_teams : ℕ
  games_per_team : ℕ
  win_probability : ℝ

/-- Calculates the probability of team A finishing with more points than team B -/
def probability_A_beats_B (tournament : SoccerTournament) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem tournament_probability : 
  let tournament := SoccerTournament.mk 7 6 (1/2)
  probability_A_beats_B tournament = 319/512 := by sorry

end NUMINAMATH_CALUDE_tournament_probability_l3225_322525


namespace NUMINAMATH_CALUDE_nucleic_acid_test_is_comprehensive_l3225_322550

/-- Represents a survey method -/
inductive SurveyMethod
| BallpointPenRefills
| FoodProducts
| CarCrashResistance
| NucleicAcidTest

/-- Predicate to determine if a survey method destroys its subjects -/
def destroysSubjects (method : SurveyMethod) : Prop :=
  match method with
  | SurveyMethod.BallpointPenRefills => true
  | SurveyMethod.FoodProducts => true
  | SurveyMethod.CarCrashResistance => true
  | SurveyMethod.NucleicAcidTest => false

/-- Definition of a comprehensive survey -/
def isComprehensiveSurvey (method : SurveyMethod) : Prop :=
  ¬(destroysSubjects method)

/-- Theorem: Nucleic Acid Test is suitable for a comprehensive survey -/
theorem nucleic_acid_test_is_comprehensive :
  isComprehensiveSurvey SurveyMethod.NucleicAcidTest :=
by
  sorry

#check nucleic_acid_test_is_comprehensive

end NUMINAMATH_CALUDE_nucleic_acid_test_is_comprehensive_l3225_322550


namespace NUMINAMATH_CALUDE_max_abc_value_l3225_322560

theorem max_abc_value (a b c : ℕ+) 
  (h1 : a * b + b * c = 518)
  (h2 : a * b - a * c = 360) :
  ∀ x y z : ℕ+, x * y * z ≤ a * b * c → x * y + y * z = 518 → x * y - x * z = 360 → 
  a * b * c = 1008 := by
sorry

end NUMINAMATH_CALUDE_max_abc_value_l3225_322560


namespace NUMINAMATH_CALUDE_system_solution_l3225_322577

theorem system_solution : ∃ (a b c : ℝ), 
  (a^2 * b^2 - a^2 - a*b + 1 = 0) ∧ 
  (a^2 * c - a*b - a - c = 0) ∧ 
  (a*b*c = -1) ∧ 
  (a = -1) ∧ (b = -1) ∧ (c = -1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3225_322577


namespace NUMINAMATH_CALUDE_max_profit_l3225_322506

/-- Profit function for location A -/
def L₁ (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2

/-- Profit function for location B -/
def L₂ (x : ℝ) : ℝ := 2 * x

/-- Total number of cars sold across both locations -/
def total_cars : ℝ := 15

/-- Total profit function -/
def L (x : ℝ) : ℝ := L₁ x + L₂ (total_cars - x)

theorem max_profit :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ total_cars ∧
  ∀ y : ℝ, 0 ≤ y ∧ y ≤ total_cars → L y ≤ L x ∧ L x = 45.6 :=
sorry

end NUMINAMATH_CALUDE_max_profit_l3225_322506


namespace NUMINAMATH_CALUDE_rectangles_combinable_l3225_322536

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents a square divided into four rectangles -/
structure DividedSquare where
  side : ℝ
  r1 : Rectangle
  r2 : Rectangle
  r3 : Rectangle
  r4 : Rectangle

/-- Assumption that the sum of areas of two non-adjacent rectangles equals the sum of areas of the other two -/
def equal_area_pairs (s : DividedSquare) : Prop :=
  area s.r1 + area s.r3 = area s.r2 + area s.r4

/-- The theorem to be proved -/
theorem rectangles_combinable (s : DividedSquare) (h : equal_area_pairs s) :
  (s.r1.width = s.r3.width ∨ s.r1.height = s.r3.height) :=
sorry

end NUMINAMATH_CALUDE_rectangles_combinable_l3225_322536


namespace NUMINAMATH_CALUDE_legos_lost_l3225_322527

def initial_legos : ℕ := 380
def given_to_sister : ℕ := 24
def current_legos : ℕ := 299

theorem legos_lost : initial_legos - given_to_sister - current_legos = 57 := by
  sorry

end NUMINAMATH_CALUDE_legos_lost_l3225_322527


namespace NUMINAMATH_CALUDE_cary_needs_14_weekends_l3225_322520

/-- Calculates the number of weekends Cary needs to mow lawns to afford discounted shoes --/
def weekends_needed (
  normal_cost : ℚ
  ) (discount_percent : ℚ
  ) (saved : ℚ
  ) (bus_expense : ℚ
  ) (earnings_per_lawn : ℚ
  ) (lawns_per_weekend : ℕ
  ) : ℕ :=
  sorry

/-- Theorem stating that Cary needs 14 weekends to afford the discounted shoes --/
theorem cary_needs_14_weekends :
  weekends_needed 120 20 30 10 5 3 = 14 :=
  sorry

end NUMINAMATH_CALUDE_cary_needs_14_weekends_l3225_322520


namespace NUMINAMATH_CALUDE_trapezoid_pq_length_l3225_322548

/-- Represents a trapezoid ABCD with a parallel line PQ intersecting diagonals -/
structure Trapezoid :=
  (a : ℝ) -- Length of base BC
  (b : ℝ) -- Length of base AD
  (pl : ℝ) -- Length of PL
  (lr : ℝ) -- Length of LR

/-- The main theorem about the length of PQ in a trapezoid -/
theorem trapezoid_pq_length (t : Trapezoid) (h : t.pl = t.lr) :
  ∃ (pq : ℝ), pq = (3 * t.a * t.b) / (2 * t.a + t.b) ∨ pq = (3 * t.a * t.b) / (t.a + 2 * t.b) :=
sorry

end NUMINAMATH_CALUDE_trapezoid_pq_length_l3225_322548


namespace NUMINAMATH_CALUDE_quadratic_roots_and_isosceles_triangle_l3225_322503

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 - (2*k + 1)*x + k^2 + k

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ := (2*k + 1)^2 - 4*(k^2 + k)

-- Define a function to check if three sides form an isosceles triangle
def is_isosceles (a b c : ℝ) : Prop := (a = b ∧ c ≠ a) ∨ (b = c ∧ a ≠ b) ∨ (c = a ∧ b ≠ c)

-- Statement of the theorem
theorem quadratic_roots_and_isosceles_triangle :
  (∀ k : ℝ, discriminant k > 0) ∧
  (∀ k : ℝ, (∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0 ∧
    is_isosceles x y 4) → (k = 3 ∨ k = 4)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_isosceles_triangle_l3225_322503


namespace NUMINAMATH_CALUDE_regular_polygon_with_140_degree_interior_angles_is_nonagon_l3225_322514

theorem regular_polygon_with_140_degree_interior_angles_is_nonagon :
  ∀ (n : ℕ) (interior_angle : ℝ),
    n ≥ 3 →
    interior_angle = 140 →
    (n - 2) * 180 = n * interior_angle →
    n = 9 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_with_140_degree_interior_angles_is_nonagon_l3225_322514
