import Mathlib

namespace NUMINAMATH_CALUDE_composite_numbers_1991_l1136_113698

theorem composite_numbers_1991 : 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = 1991^1991 + 1) ∧ 
  (∃ c d : ℕ, c > 1 ∧ d > 1 ∧ c * d = 1991^1991 - 1) := by
  sorry

end NUMINAMATH_CALUDE_composite_numbers_1991_l1136_113698


namespace NUMINAMATH_CALUDE_percent_within_one_std_dev_l1136_113642

/-- A symmetric distribution about a mean -/
structure SymmetricDistribution where
  mean : ℝ
  std_dev : ℝ
  is_symmetric : Bool
  percent_less_than_mean_plus_std : ℝ

/-- Theorem: In a symmetric distribution where 80% is less than mean + std_dev,
    60% lies within one standard deviation of the mean -/
theorem percent_within_one_std_dev
  (dist : SymmetricDistribution)
  (h_symmetric : dist.is_symmetric = true)
  (h_eighty_percent : dist.percent_less_than_mean_plus_std = 80) :
  ∃ (percent_within : ℝ), percent_within = 60 :=
sorry

end NUMINAMATH_CALUDE_percent_within_one_std_dev_l1136_113642


namespace NUMINAMATH_CALUDE_smallest_two_digit_switch_add_five_l1136_113690

def digit_switch (n : ℕ) : ℕ := 
  (n % 10) * 10 + (n / 10)

theorem smallest_two_digit_switch_add_five : 
  ∀ n : ℕ, 
    10 ≤ n → n < 100 → 
    (∀ m : ℕ, 10 ≤ m → m < n → digit_switch m + 5 ≠ 3 * m) → 
    digit_switch n + 5 = 3 * n → 
    n = 34 := by
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_switch_add_five_l1136_113690


namespace NUMINAMATH_CALUDE_function_domain_l1136_113607

/-- The function f(x) = √(2-2^x) + 1/ln(x) is defined if and only if x ∈ (0,1) -/
theorem function_domain (x : ℝ) : 
  (∃ (y : ℝ), y = Real.sqrt (2 - 2^x) + 1 / Real.log x) ↔ 0 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_function_domain_l1136_113607


namespace NUMINAMATH_CALUDE_james_work_hours_l1136_113633

/-- Calculates the number of hours James works at his main job --/
theorem james_work_hours (main_rate : ℝ) (second_rate_reduction : ℝ) (total_earnings : ℝ) :
  main_rate = 20 →
  second_rate_reduction = 0.2 →
  total_earnings = 840 →
  ∃ h : ℝ, h = 30 ∧ 
    main_rate * h + (main_rate * (1 - second_rate_reduction)) * (h / 2) = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_james_work_hours_l1136_113633


namespace NUMINAMATH_CALUDE_smallest_n_equal_l1136_113647

/-- Geometric series C_n -/
def C (n : ℕ) : ℚ :=
  352 * (1 - (1/2)^n) / (1 - 1/2)

/-- Geometric series D_n -/
def D (n : ℕ) : ℚ :=
  992 * (1 - (1/(-2))^n) / (1 + 1/2)

/-- The smallest n ≥ 1 for which C_n = D_n is 1 -/
theorem smallest_n_equal (n : ℕ) (h : n ≥ 1) : (C n = D n) → n = 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_equal_l1136_113647


namespace NUMINAMATH_CALUDE_units_digit_of_T_is_zero_l1136_113630

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def T : ℕ := (List.range 99).foldl (λ acc i => acc + factorial (i + 3)) 0

theorem units_digit_of_T_is_zero : T % 10 = 0 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_T_is_zero_l1136_113630


namespace NUMINAMATH_CALUDE_distance_between_points_l1136_113687

theorem distance_between_points : ∀ (x1 y1 x2 y2 : ℝ), 
  x1 = 0 ∧ y1 = 6 ∧ x2 = 8 ∧ y2 = 0 → 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1136_113687


namespace NUMINAMATH_CALUDE_fraction_simplification_l1136_113637

theorem fraction_simplification :
  (240 : ℚ) / 18 * 9 / 135 * 7 / 4 = 14 / 9 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1136_113637


namespace NUMINAMATH_CALUDE_sqrt2_not_all_zeros_in_range_l1136_113675

/-- The nth decimal digit of √2 -/
def d (n : ℕ) : ℕ := sorry

/-- The range of n we're considering -/
def n_range : Set ℕ := {n | 1000001 ≤ n ∧ n ≤ 3000000}

theorem sqrt2_not_all_zeros_in_range : 
  ¬ (∀ n ∈ n_range, d n = 0) := by sorry

end NUMINAMATH_CALUDE_sqrt2_not_all_zeros_in_range_l1136_113675


namespace NUMINAMATH_CALUDE_power_of_three_equality_l1136_113604

theorem power_of_three_equality (m : ℕ) : 3^m = 27 * 81^4 * 243^3 → m = 34 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_equality_l1136_113604


namespace NUMINAMATH_CALUDE_prob_one_pilot_hits_l1136_113674

/-- The probability that exactly one of two independent events occurs,
    given their individual probabilities of occurrence. -/
def prob_exactly_one (p_a p_b : ℝ) : ℝ := p_a * (1 - p_b) + (1 - p_a) * p_b

/-- The probability of pilot A hitting the target -/
def p_a : ℝ := 0.4

/-- The probability of pilot B hitting the target -/
def p_b : ℝ := 0.5

/-- Theorem: The probability that exactly one pilot hits the target is 0.5 -/
theorem prob_one_pilot_hits : prob_exactly_one p_a p_b = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_pilot_hits_l1136_113674


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1136_113655

theorem unique_solution_for_equation :
  ∃! (x y : ℝ), (x - 10)^2 + (y - 11)^2 + (x - y)^2 = 1/3 ∧ 
  x = 10 + 1/3 ∧ y = 10 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1136_113655


namespace NUMINAMATH_CALUDE_gcd_problem_l1136_113611

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 953) :
  Int.gcd (3 * b^2 + 17 * b + 23) (b + 19) = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1136_113611


namespace NUMINAMATH_CALUDE_francie_savings_l1136_113626

/-- Calculates Francie's remaining money after saving and spending --/
def franciesRemainingMoney (
  initialWeeklyAllowance : ℕ) 
  (initialWeeks : ℕ)
  (raisedWeeklyAllowance : ℕ)
  (raisedWeeks : ℕ)
  (videoGameCost : ℕ) : ℕ :=
  let totalSavings := initialWeeklyAllowance * initialWeeks + raisedWeeklyAllowance * raisedWeeks
  let remainingAfterClothes := totalSavings / 2
  remainingAfterClothes - videoGameCost

theorem francie_savings : franciesRemainingMoney 5 8 6 6 35 = 3 := by
  sorry

end NUMINAMATH_CALUDE_francie_savings_l1136_113626


namespace NUMINAMATH_CALUDE_scientific_notation_of_248000_l1136_113616

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_248000 :
  toScientificNotation 248000 = ScientificNotation.mk 2.48 5 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_248000_l1136_113616


namespace NUMINAMATH_CALUDE_unique_integer_with_properties_l1136_113608

theorem unique_integer_with_properties : ∃! n : ℕ+, 
  (∃ k : ℕ, n = 18 * k) ∧ 
  (30 < Real.sqrt n.val) ∧ 
  (Real.sqrt n.val < 30.5) := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_with_properties_l1136_113608


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1136_113672

theorem circle_area_ratio (C D : Real) (hC : C > 0) (hD : D > 0) : 
  (60 / 360 * (2 * Real.pi * C) = 40 / 360 * (2 * Real.pi * D)) → 
  (Real.pi * C^2) / (Real.pi * D^2) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1136_113672


namespace NUMINAMATH_CALUDE_base_eight_solution_l1136_113658

theorem base_eight_solution : ∃! (b : ℕ), b > 1 ∧ (3 * b + 2)^2 = b^3 + b + 4 :=
by sorry

end NUMINAMATH_CALUDE_base_eight_solution_l1136_113658


namespace NUMINAMATH_CALUDE_marked_price_calculation_jobber_pricing_l1136_113656

theorem marked_price_calculation (original_price : ℝ) (discount_percent : ℝ) 
  (gain_percent : ℝ) (final_discount_percent : ℝ) : ℝ :=
  let purchase_price := original_price * (1 - discount_percent / 100)
  let selling_price := purchase_price * (1 + gain_percent / 100)
  let marked_price := selling_price / (1 - final_discount_percent / 100)
  marked_price

theorem jobber_pricing : 
  marked_price_calculation 30 15 50 25 = 51 := by
  sorry

end NUMINAMATH_CALUDE_marked_price_calculation_jobber_pricing_l1136_113656


namespace NUMINAMATH_CALUDE_sqrt_5_simplest_l1136_113678

def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → x = Real.sqrt y → ¬∃ (a b : ℝ), b ≠ 1 ∧ y = a * b^2

theorem sqrt_5_simplest :
  is_simplest_sqrt (Real.sqrt 5) ∧
  ¬is_simplest_sqrt (Real.sqrt 9) ∧
  ¬is_simplest_sqrt (Real.sqrt 18) ∧
  ¬is_simplest_sqrt (Real.sqrt (1/2)) :=
sorry

end NUMINAMATH_CALUDE_sqrt_5_simplest_l1136_113678


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1136_113662

def polynomial (x : ℝ) : ℝ := 5*x^4 - 12*x^3 + 3*x^2 - 5*x + 15

def divisor (x : ℝ) : ℝ := 3*x - 9

theorem polynomial_remainder : 
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * (q x) + 108 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1136_113662


namespace NUMINAMATH_CALUDE_point_coordinates_l1136_113693

/-- A point in the two-dimensional plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of the two-dimensional plane. -/
def fourthQuadrant (p : Point) : Prop := p.x > 0 ∧ p.y < 0

/-- The distance from a point to the x-axis. -/
def distanceToXAxis (p : Point) : ℝ := |p.y|

/-- The distance from a point to the y-axis. -/
def distanceToYAxis (p : Point) : ℝ := |p.x|

/-- Theorem: If a point P is in the fourth quadrant, its distance to the x-axis is 4,
    and its distance to the y-axis is 2, then its coordinates are (2, -4). -/
theorem point_coordinates (P : Point) 
  (h1 : fourthQuadrant P) 
  (h2 : distanceToXAxis P = 4) 
  (h3 : distanceToYAxis P = 2) : 
  P.x = 2 ∧ P.y = -4 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1136_113693


namespace NUMINAMATH_CALUDE_final_order_exact_points_total_games_l1136_113677

-- Define the structure for a team's game outcomes
structure TeamOutcome where
  name : String
  wins : Nat
  losses : Nat
  draws : Nat
  bonusWins : Nat
  extraBonus : Nat

-- Define the point system
def regularWinPoints : Nat := 3
def regularLossPoints : Nat := 0
def regularDrawPoints : Nat := 1
def bonusWinPoints : Nat := 2
def extraBonusPoints : Nat := 1

-- Calculate total points for a team
def calculatePoints (team : TeamOutcome) : Nat :=
  team.wins * regularWinPoints +
  team.losses * regularLossPoints +
  team.draws * regularDrawPoints +
  team.bonusWins * bonusWinPoints +
  team.extraBonus * extraBonusPoints

-- Define the teams
def soccerStars : TeamOutcome := ⟨"Team Soccer Stars", 18, 5, 7, 6, 4⟩
def lightningStrikers : TeamOutcome := ⟨"Lightning Strikers", 15, 8, 7, 5, 3⟩
def goalGrabbers : TeamOutcome := ⟨"Goal Grabbers", 21, 5, 4, 4, 9⟩
def cleverKickers : TeamOutcome := ⟨"Clever Kickers", 11, 10, 9, 2, 1⟩

-- Theorem to prove the final order of teams
theorem final_order :
  calculatePoints goalGrabbers > calculatePoints soccerStars ∧
  calculatePoints soccerStars > calculatePoints lightningStrikers ∧
  calculatePoints lightningStrikers > calculatePoints cleverKickers :=
by sorry

-- Theorem to prove the exact points for each team
theorem exact_points :
  calculatePoints goalGrabbers = 84 ∧
  calculatePoints soccerStars = 77 ∧
  calculatePoints lightningStrikers = 65 ∧
  calculatePoints cleverKickers = 47 :=
by sorry

-- Theorem to prove that each team played exactly 30 games
theorem total_games (team : TeamOutcome) :
  team.wins + team.losses + team.draws = 30 :=
by sorry

end NUMINAMATH_CALUDE_final_order_exact_points_total_games_l1136_113677


namespace NUMINAMATH_CALUDE_circular_track_length_l1136_113668

/-- The length of a circular track given specific overtaking conditions -/
theorem circular_track_length : ∃ (x : ℝ), x > 0 ∧ 279 < x ∧ x < 281 ∧
  ∃ (v_fast v_slow : ℝ), v_fast > v_slow ∧ v_fast > 0 ∧ v_slow > 0 ∧
  (150 / (x - 150) = (x + 100) / (x + 50)) := by
  sorry

end NUMINAMATH_CALUDE_circular_track_length_l1136_113668


namespace NUMINAMATH_CALUDE_triangle_nabla_equality_l1136_113691

-- Define the triangle operation
def triangle (a b : ℤ) : ℤ := 3 * a + 2 * b

-- Define the nabla operation
def nabla (a b : ℤ) : ℤ := 2 * a + 3 * b

-- Theorem to prove
theorem triangle_nabla_equality : triangle 2 (nabla 3 4) = 42 := by
  sorry

end NUMINAMATH_CALUDE_triangle_nabla_equality_l1136_113691


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_smallest_solution_is_3_minus_sqrt_3_l1136_113639

theorem smallest_solution_of_equation (x : ℝ) :
  (1 / (x - 2) + 1 / (x - 4) = 3 / (x - 3)) ↔ (x = 3 - Real.sqrt 3 ∨ x = 3 + Real.sqrt 3) :=
by sorry

theorem smallest_solution_is_3_minus_sqrt_3 :
  ∃ x : ℝ, (1 / (x - 2) + 1 / (x - 4) = 3 / (x - 3)) ∧
           (∀ y : ℝ, (1 / (y - 2) + 1 / (y - 4) = 3 / (y - 3)) → x ≤ y) ∧
           x = 3 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_smallest_solution_is_3_minus_sqrt_3_l1136_113639


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_range_l1136_113660

theorem rectangular_prism_volume_range (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b + c = 9 → 
  a * b + b * c + a * c = 24 → 
  16 ≤ a * b * c ∧ a * b * c ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_range_l1136_113660


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l1136_113631

/-- Given a hyperbola with equation x²/25 - y²/9 = 1, 
    prove that the distance between its foci is 2√34 -/
theorem hyperbola_foci_distance (x y : ℝ) :
  (x^2 / 25 - y^2 / 9 = 1) → 
  (∃ (f₁ f₂ : ℝ × ℝ), (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = (2 * Real.sqrt 34)^2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l1136_113631


namespace NUMINAMATH_CALUDE_angle_ABC_is_60_l1136_113635

-- Define the angles as real numbers
def angle_ABC : ℝ := sorry
def angle_ABD : ℝ := 30
def angle_CBD : ℝ := 90

-- State the theorem
theorem angle_ABC_is_60 :
  -- Condition: The sum of angles around point B is 180°
  angle_ABC + angle_ABD + angle_CBD = 180 →
  -- Conclusion: The measure of ∠ABC is 60°
  angle_ABC = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_ABC_is_60_l1136_113635


namespace NUMINAMATH_CALUDE_equal_perimeter_lines_concurrent_l1136_113638

open Real

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a line through a vertex
structure VertexLine :=
  (vertex : ℝ × ℝ)
  (point : ℝ × ℝ)

-- Function to check if a line divides a triangle into two triangles with equal perimeter
def divides_equal_perimeter (t : Triangle) (l : VertexLine) : Prop :=
  sorry

-- Function to check if three lines are concurrent
def are_concurrent (l1 l2 l3 : VertexLine) : Prop :=
  sorry

-- Theorem statement
theorem equal_perimeter_lines_concurrent (t : Triangle) :
  ∀ (l1 l2 l3 : VertexLine),
    (divides_equal_perimeter t l1 ∧ 
     divides_equal_perimeter t l2 ∧ 
     divides_equal_perimeter t l3) →
    are_concurrent l1 l2 l3 :=
sorry

end NUMINAMATH_CALUDE_equal_perimeter_lines_concurrent_l1136_113638


namespace NUMINAMATH_CALUDE_jacks_card_collection_l1136_113618

theorem jacks_card_collection :
  ∀ (football_cards baseball_cards total_cards : ℕ),
    baseball_cards = 3 * football_cards + 5 →
    baseball_cards = 95 →
    total_cards = baseball_cards + football_cards →
    total_cards = 125 := by
  sorry

end NUMINAMATH_CALUDE_jacks_card_collection_l1136_113618


namespace NUMINAMATH_CALUDE_min_shaded_triangles_theorem_l1136_113696

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℕ

/-- Represents the division of a large equilateral triangle into smaller ones -/
structure TriangleDivision where
  largeSideLength : ℕ
  smallSideLength : ℕ

/-- Calculates the number of intersection points in a triangle division -/
def intersectionPoints (d : TriangleDivision) : ℕ :=
  let n : ℕ := d.largeSideLength / d.smallSideLength + 1
  n * (n + 1) / 2

/-- Calculates the minimum number of smaller triangles needed to be shaded -/
def minShadedTriangles (d : TriangleDivision) : ℕ :=
  (intersectionPoints d + 2) / 3

/-- The main theorem to prove -/
theorem min_shaded_triangles_theorem (t : EquilateralTriangle) (d : TriangleDivision) :
  t.sideLength = 8 →
  d.largeSideLength = 8 →
  d.smallSideLength = 1 →
  minShadedTriangles d = 15 := by
  sorry

end NUMINAMATH_CALUDE_min_shaded_triangles_theorem_l1136_113696


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l1136_113685

theorem rectangle_dimension_change 
  (L B : ℝ) -- Original length and breadth
  (L' : ℝ) -- New length
  (h1 : L > 0 ∧ B > 0) -- Positive dimensions
  (h2 : L' * (3 * B) = (3/2) * (L * B)) -- Area increased by 50% and breadth tripled
  : L' = L / 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l1136_113685


namespace NUMINAMATH_CALUDE_panthers_scored_17_points_l1136_113627

-- Define the points scored by the Wildcats
def wildcats_points : ℕ := 36

-- Define the difference in points between Wildcats and Panthers
def point_difference : ℕ := 19

-- Define the points scored by the Panthers
def panthers_points : ℕ := wildcats_points - point_difference

-- Theorem to prove
theorem panthers_scored_17_points : panthers_points = 17 := by
  sorry

end NUMINAMATH_CALUDE_panthers_scored_17_points_l1136_113627


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_multiplier_l1136_113621

theorem consecutive_odd_integers_multiplier :
  ∀ (n : ℤ),
  (n + 4 = 15) →
  (∃ k : ℚ, 3 * n = k * (n + 4) + 3) →
  (∃ k : ℚ, 3 * n = k * (n + 4) + 3 ∧ k = 2) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_multiplier_l1136_113621


namespace NUMINAMATH_CALUDE_expand_squared_difference_product_expand_linear_factors_l1136_113617

/-- Theorem for the expansion of (2a-b)^2 * (2a+b)^2 -/
theorem expand_squared_difference_product (a b : ℝ) :
  (2*a - b)^2 * (2*a + b)^2 = 16*a^4 - 8*a^2*b^2 + b^4 := by sorry

/-- Theorem for the expansion of (3a+b-2)(3a-b+2) -/
theorem expand_linear_factors (a b : ℝ) :
  (3*a + b - 2) * (3*a - b + 2) = 9*a^2 - b^2 + 4*b - 4 := by sorry

end NUMINAMATH_CALUDE_expand_squared_difference_product_expand_linear_factors_l1136_113617


namespace NUMINAMATH_CALUDE_gcd_lcm_examples_l1136_113652

theorem gcd_lcm_examples : 
  (Nat.gcd 17 51 = 17) ∧ 
  (Nat.lcm 17 51 = 51) ∧ 
  (Nat.gcd 6 8 = 2) ∧ 
  (Nat.lcm 8 9 = 72) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_examples_l1136_113652


namespace NUMINAMATH_CALUDE_floor_plus_one_l1136_113682

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the ceiling function
noncomputable def ceil (x : ℝ) : ℤ :=
  -Int.floor (-x)

-- Statement to prove
theorem floor_plus_one (x : ℝ) : floor (x + 1) = floor x + 1 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_one_l1136_113682


namespace NUMINAMATH_CALUDE_expression_bounds_l1136_113684

theorem expression_bounds (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  1/2 ≤ |2*a - b| / (|a| + |b|) ∧ |2*a - b| / (|a| + |b|) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l1136_113684


namespace NUMINAMATH_CALUDE_number_of_paths_equals_combination_l1136_113628

def grid_width : ℕ := 7
def grid_height : ℕ := 4

def total_steps : ℕ := grid_width + grid_height - 2
def up_steps : ℕ := grid_height - 1

theorem number_of_paths_equals_combination :
  (Nat.choose total_steps up_steps) = 84 := by
  sorry

end NUMINAMATH_CALUDE_number_of_paths_equals_combination_l1136_113628


namespace NUMINAMATH_CALUDE_cupcakes_baked_and_iced_l1136_113667

/-- Represents the number of cups of sugar in a bag -/
def sugar_per_bag : ℕ := 6

/-- Represents the number of bags of sugar bought -/
def bags_bought : ℕ := 2

/-- Represents the number of cups of sugar Lillian has at home -/
def sugar_at_home : ℕ := 3

/-- Represents the number of cups of sugar needed for batter per dozen cupcakes -/
def sugar_for_batter : ℕ := 1

/-- Represents the number of cups of sugar needed for frosting per dozen cupcakes -/
def sugar_for_frosting : ℕ := 2

/-- Theorem stating that Lillian can bake and ice 5 dozen cupcakes -/
theorem cupcakes_baked_and_iced : ℕ := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_baked_and_iced_l1136_113667


namespace NUMINAMATH_CALUDE_sum_even_digits_1_to_200_l1136_113614

/-- E(n) represents the sum of even digits in the number n -/
def E (n : ℕ) : ℕ := sorry

/-- The sum of E(n) for n from 1 to 200 -/
def sumE : ℕ := (Finset.range 200).sum E + E 200

theorem sum_even_digits_1_to_200 : sumE = 800 := by sorry

end NUMINAMATH_CALUDE_sum_even_digits_1_to_200_l1136_113614


namespace NUMINAMATH_CALUDE_regression_line_not_necessarily_through_sample_point_l1136_113641

/-- Given a set of sample data points and a regression line, 
    prove that the line doesn't necessarily pass through any sample point. -/
theorem regression_line_not_necessarily_through_sample_point 
  (n : ℕ) 
  (x y : Fin n → ℝ) 
  (a b : ℝ) : 
  ¬ (∀ (ε : ℝ), ε > 0 → 
    ∃ (i : Fin n), |y i - (b * x i + a)| < ε) :=
sorry

end NUMINAMATH_CALUDE_regression_line_not_necessarily_through_sample_point_l1136_113641


namespace NUMINAMATH_CALUDE_smallest_factorial_with_1987_zeros_l1136_113602

/-- Count the number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

/-- The smallest natural number n such that n! ends with exactly 1987 zeros -/
def smallestFactorialWith1987Zeros : ℕ := 7960

theorem smallest_factorial_with_1987_zeros :
  (∀ m < smallestFactorialWith1987Zeros, trailingZeros m < 1987) ∧
  trailingZeros smallestFactorialWith1987Zeros = 1987 := by
  sorry

end NUMINAMATH_CALUDE_smallest_factorial_with_1987_zeros_l1136_113602


namespace NUMINAMATH_CALUDE_real_solutions_quadratic_l1136_113666

theorem real_solutions_quadratic (x : ℝ) : 
  (∃ y : ℝ, 4 * y^2 + 4 * x * y + x + 6 = 0) ↔ x ≤ -2 ∨ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_real_solutions_quadratic_l1136_113666


namespace NUMINAMATH_CALUDE_function_fits_data_l1136_113699

def f (x : ℝ) : ℝ := 210 - 10*x - x^2 - 2*x^3

theorem function_fits_data : 
  (f 0 = 210) ∧ 
  (f 2 = 170) ∧ 
  (f 4 = 110) ∧ 
  (f 6 = 30) ∧ 
  (f 8 = -70) := by
  sorry

end NUMINAMATH_CALUDE_function_fits_data_l1136_113699


namespace NUMINAMATH_CALUDE_charles_pictures_l1136_113680

theorem charles_pictures (total_papers : ℕ) (today_pictures : ℕ) (yesterday_before_work : ℕ) (papers_left : ℕ) 
  (h1 : total_papers = 20)
  (h2 : today_pictures = 6)
  (h3 : yesterday_before_work = 6)
  (h4 : papers_left = 2) :
  total_papers - (today_pictures + yesterday_before_work) - papers_left = 6 := by
  sorry

end NUMINAMATH_CALUDE_charles_pictures_l1136_113680


namespace NUMINAMATH_CALUDE_correct_assignment_count_l1136_113694

/-- The number of ways to assign 5 friends to 5 rooms with at most 2 friends per room -/
def assignmentWays : ℕ := 1620

/-- A function that calculates the number of ways to assign n friends to m rooms with at most k friends per room -/
def calculateAssignmentWays (n m k : ℕ) : ℕ :=
  sorry  -- The actual implementation is not provided

theorem correct_assignment_count :
  calculateAssignmentWays 5 5 2 = assignmentWays :=
by sorry

end NUMINAMATH_CALUDE_correct_assignment_count_l1136_113694


namespace NUMINAMATH_CALUDE_twentyseven_binary_l1136_113689

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

/-- Convert a list of booleans to a natural number in binary -/
def fromBinary (l : List Bool) : ℕ :=
  l.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem twentyseven_binary :
  toBinary 27 = [true, true, false, true, true] :=
sorry

end NUMINAMATH_CALUDE_twentyseven_binary_l1136_113689


namespace NUMINAMATH_CALUDE_total_dress_designs_l1136_113605

/-- The number of fabric colors available. -/
def num_colors : ℕ := 4

/-- The number of patterns available. -/
def num_patterns : ℕ := 5

/-- Each dress design requires exactly one color and one pattern. -/
axiom dress_design_requirement : True

/-- The total number of different dress designs. -/
def total_designs : ℕ := num_colors * num_patterns

/-- Theorem stating that the total number of different dress designs is 20. -/
theorem total_dress_designs : total_designs = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_dress_designs_l1136_113605


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1136_113644

theorem inequality_system_solution (k : ℝ) : 
  (∀ x : ℤ, (x^2 - x - 2 > 0 ∧ 2*x^2 + (2*k+5)*x + 5*k < 0) ↔ x = -2) →
  -3 ≤ k ∧ k < 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1136_113644


namespace NUMINAMATH_CALUDE_debby_candy_l1136_113613

def initial_candy : ℕ → ℕ → ℕ
  | remaining, eaten => remaining + eaten

theorem debby_candy (remaining eaten : ℕ) 
  (h1 : remaining = 3) 
  (h2 : eaten = 9) : 
  initial_candy remaining eaten = 12 := by
  sorry

end NUMINAMATH_CALUDE_debby_candy_l1136_113613


namespace NUMINAMATH_CALUDE_problem_statement_l1136_113683

theorem problem_statement (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -1) :
  2 * a + 2 * b - 3 * a * b = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1136_113683


namespace NUMINAMATH_CALUDE_sin_A_in_special_triangle_l1136_113646

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem sin_A_in_special_triangle (t : Triangle) (h1 : t.a = 8) (h2 : t.b = 7) (h3 : t.B = 30 * π / 180) :
  Real.sin t.A = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sin_A_in_special_triangle_l1136_113646


namespace NUMINAMATH_CALUDE_voting_difference_l1136_113673

/-- Represents the voting results for a company policy -/
structure VotingResults where
  total_employees : ℕ
  initial_for : ℕ
  initial_against : ℕ
  second_for : ℕ
  second_against : ℕ

/-- Conditions for the voting scenario -/
def voting_conditions (v : VotingResults) : Prop :=
  v.total_employees = 450 ∧
  v.initial_for + v.initial_against = v.total_employees ∧
  v.second_for + v.second_against = v.total_employees ∧
  v.initial_against > v.initial_for ∧
  v.second_for > v.second_against ∧
  (v.second_for - v.second_against) = 3 * (v.initial_against - v.initial_for) ∧
  v.second_for = (10 * v.initial_against) / 9

theorem voting_difference (v : VotingResults) 
  (h : voting_conditions v) : v.second_for - v.initial_for = 52 := by
  sorry

end NUMINAMATH_CALUDE_voting_difference_l1136_113673


namespace NUMINAMATH_CALUDE_smallest_n_for_divisible_by_20_l1136_113606

theorem smallest_n_for_divisible_by_20 :
  ∃ (n : ℕ), n = 7 ∧ n ≥ 4 ∧
  (∀ (S : Finset ℤ), S.card = n →
    ∃ (a b c d : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    20 ∣ (a + b - c - d)) ∧
  (∀ (m : ℕ), m < 7 → m ≥ 4 →
    ∃ (T : Finset ℤ), T.card = m ∧
      ∀ (a b c d : ℤ), a ∈ T → b ∈ T → c ∈ T → d ∈ T →
      a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
      ¬(20 ∣ (a + b - c - d))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_divisible_by_20_l1136_113606


namespace NUMINAMATH_CALUDE_unique_denomination_l1136_113695

/-- Given unlimited supply of stamps of denominations 7, n, and n+2 cents,
    120 cents is the greatest postage that cannot be formed -/
def is_valid_denomination (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 120 → ∃ (a b c : ℕ), k = 7 * a + n * b + (n + 2) * c

/-- 120 cents cannot be formed using stamps of denominations 7, n, and n+2 cents -/
def cannot_form_120 (n : ℕ) : Prop :=
  ¬∃ (a b c : ℕ), 120 = 7 * a + n * b + (n + 2) * c

theorem unique_denomination :
  ∃! n : ℕ, n > 0 ∧ is_valid_denomination n ∧ cannot_form_120 n :=
by sorry

end NUMINAMATH_CALUDE_unique_denomination_l1136_113695


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l1136_113650

theorem sqrt_x_div_sqrt_y (x y : ℝ) :
  (((1 : ℝ) / 3) ^ 2 + ((1 : ℝ) / 4) ^ 2) / (((1 : ℝ) / 5) ^ 2 + ((1 : ℝ) / 6) ^ 2) = 25 * x / (53 * y) →
  Real.sqrt x / Real.sqrt y = 150 / 239 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l1136_113650


namespace NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l1136_113636

theorem cos_sum_of_complex_exponentials (α β : ℝ) :
  Complex.exp (α * Complex.I) = (4 / 5 : ℂ) - (3 / 5 : ℂ) * Complex.I →
  Complex.exp (β * Complex.I) = (5 / 13 : ℂ) + (12 / 13 : ℂ) * Complex.I →
  Real.cos (α + β) = -16 / 65 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l1136_113636


namespace NUMINAMATH_CALUDE_paddle_prices_and_cost_effective_solution_l1136_113651

/-- Represents the price of a pair of straight paddles in yuan -/
def straight_paddle_price : ℝ := sorry

/-- Represents the price of a pair of horizontal paddles in yuan -/
def horizontal_paddle_price : ℝ := sorry

/-- Cost of table tennis balls per pair of paddles -/
def ball_cost : ℝ := 20

/-- Total cost for 20 pairs of straight paddles and 15 pairs of horizontal paddles -/
def total_cost_35_pairs : ℝ := 9000

/-- Difference in cost between 10 pairs of horizontal paddles and 5 pairs of straight paddles -/
def cost_difference : ℝ := 1600

/-- Theorem stating the prices of paddles and the cost-effective solution -/
theorem paddle_prices_and_cost_effective_solution :
  (straight_paddle_price = 220 ∧ horizontal_paddle_price = 260) ∧
  (∀ m : ℕ, m ≤ 40 → m ≤ 3 * (40 - m) →
    m * (straight_paddle_price + ball_cost) + (40 - m) * (horizontal_paddle_price + ball_cost) ≥ 10000) ∧
  (30 * (straight_paddle_price + ball_cost) + 10 * (horizontal_paddle_price + ball_cost) = 10000) :=
by sorry

end NUMINAMATH_CALUDE_paddle_prices_and_cost_effective_solution_l1136_113651


namespace NUMINAMATH_CALUDE_star_calculation_star_equation_solutions_l1136_113648

-- Define the ☆ operation
noncomputable def star (x y : ℤ) : ℤ :=
  if x = 0 then |y|
  else if y = 0 then |x|
  else if (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) then |x| + |y|
  else -(|x| + |y|)

-- Theorem for the first part of the problem
theorem star_calculation : star 11 (star 0 (-12)) = 23 := by sorry

-- Theorem for the second part of the problem
theorem star_equation_solutions :
  {a : ℤ | 2 * (star 2 a) - 1 = 3 * a} = {3, -5} := by sorry

end NUMINAMATH_CALUDE_star_calculation_star_equation_solutions_l1136_113648


namespace NUMINAMATH_CALUDE_power_sum_constant_implies_zero_or_one_l1136_113679

/-- Given a natural number n > 1 and a list of real numbers x,
    if the sum of the k-th powers of these numbers is constant for k from 1 to n+1,
    then each number in the list is either 0 or 1. -/
theorem power_sum_constant_implies_zero_or_one (n : ℕ) (x : List ℝ) :
  n > 1 →
  x.length = n →
  (∀ k : ℕ, k ≥ 1 → k ≤ n + 1 →
    (List.sum (List.map (fun xi => xi ^ k) x)) = (List.sum (List.map (fun xi => xi ^ 1) x))) →
  ∀ xi ∈ x, xi = 0 ∨ xi = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_constant_implies_zero_or_one_l1136_113679


namespace NUMINAMATH_CALUDE_inequality_proof_l1136_113688

theorem inequality_proof (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a * b / (c - 1) + b * c / (a - 1) + c * a / (b - 1) ≥ 12) ∧
  (a * b / (c - 1) + b * c / (a - 1) + c * a / (b - 1) = 12 ↔ a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1136_113688


namespace NUMINAMATH_CALUDE_popping_corn_probability_l1136_113619

theorem popping_corn_probability (total : ℝ) (h_total : total > 0) :
  let white := (3 / 4 : ℝ) * total
  let yellow := (1 / 4 : ℝ) * total
  let white_pop_prob := (3 / 5 : ℝ)
  let yellow_pop_prob := (1 / 2 : ℝ)
  let white_popped := white * white_pop_prob
  let yellow_popped := yellow * yellow_pop_prob
  let total_popped := white_popped + yellow_popped
  (white_popped / total_popped) = (18 / 23 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_popping_corn_probability_l1136_113619


namespace NUMINAMATH_CALUDE_median_mode_difference_l1136_113663

def data : List ℕ := [21, 23, 23, 24, 24, 33, 33, 33, 33, 42, 42, 47, 48, 51, 52, 53, 54, 62, 67, 68]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem median_mode_difference (h : data.length = 20) : 
  |median data - (mode data : ℚ)| = 0 := by sorry

end NUMINAMATH_CALUDE_median_mode_difference_l1136_113663


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1136_113697

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (4 * x + 15 - 6) = 12 → x = 33.75 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1136_113697


namespace NUMINAMATH_CALUDE_special_ellipse_equation_l1136_113640

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  -- The ellipse equation in the form (x²/a²) + (y²/b²) = 1
  a : ℝ
  b : ℝ
  -- Center at origin
  center_origin : True
  -- Foci on coordinate axis
  foci_on_axis : True
  -- Line y = x + 1 intersects the ellipse
  intersects_line : True
  -- OP ⊥ OQ where P and Q are intersection points
  op_perp_oq : True
  -- |PQ| = √10/2
  pq_length : True

/-- The theorem stating the possible equations of the special ellipse -/
theorem special_ellipse_equation (e : SpecialEllipse) :
  (∀ x y, x^2 + 3*y^2 = 2) ∨ (∀ x y, 3*x^2 + y^2 = 2) :=
sorry

end NUMINAMATH_CALUDE_special_ellipse_equation_l1136_113640


namespace NUMINAMATH_CALUDE_extremum_at_negative_three_l1136_113653

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 5*x^2 + a*x

-- State the theorem
theorem extremum_at_negative_three (a : ℝ) : 
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x, x ≠ -3 ∧ |x + 3| < ε → f a x ≤ f a (-3)) ∨
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x, x ≠ -3 ∧ |x + 3| < ε → f a x ≥ f a (-3)) →
  a = 3 := by
sorry


end NUMINAMATH_CALUDE_extremum_at_negative_three_l1136_113653


namespace NUMINAMATH_CALUDE_complex_absolute_value_l1136_113622

def i : ℂ := Complex.I

theorem complex_absolute_value : 
  Complex.abs ((1 : ℂ) / (1 - i) - i) = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l1136_113622


namespace NUMINAMATH_CALUDE_tamika_always_wins_l1136_113681

def tamika_set : Finset Nat := {6, 7, 8}
def carlos_set : Finset Nat := {2, 3, 5}

theorem tamika_always_wins :
  ∀ (t1 t2 : Nat) (c1 c2 : Nat),
    t1 ∈ tamika_set → t2 ∈ tamika_set → t1 ≠ t2 →
    c1 ∈ carlos_set → c2 ∈ carlos_set → c1 ≠ c2 →
    t1 * t2 > c1 * c2 := by
  sorry

#check tamika_always_wins

end NUMINAMATH_CALUDE_tamika_always_wins_l1136_113681


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_cubed_less_than_800_l1136_113669

theorem greatest_multiple_of_four_cubed_less_than_800 :
  ∃ (x : ℕ), x = 8 ∧ 
  (∀ (y : ℕ), y > 0 ∧ 4 ∣ y ∧ y^3 < 800 → y ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_cubed_less_than_800_l1136_113669


namespace NUMINAMATH_CALUDE_inscribed_circle_max_radius_l1136_113659

/-- Given a triangle ABC with side lengths a, b, c, and area A,
    and an inscribed circle with radius r, 
    the radius r is at most (2 * A) / (a + b + c) --/
theorem inscribed_circle_max_radius 
  (a b c : ℝ) 
  (A : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hA : A > 0) 
  (h_triangle : A = a * b * c / (4 * (a * b + b * c + c * a - a * a - b * b - c * c).sqrt)) 
  (r : ℝ) 
  (hr : r > 0) 
  (h_inscribed : r * (a + b + c) ≤ 2 * A) :
  r ≤ 2 * A / (a + b + c) ∧ 
  (r = 2 * A / (a + b + c) ↔ r * (a + b + c) = 2 * A) :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_max_radius_l1136_113659


namespace NUMINAMATH_CALUDE_diesel_rates_indeterminable_l1136_113649

/-- Represents the diesel purchase data for a company over 4 years -/
structure DieselPurchaseData where
  /-- The diesel rates for each of the 4 years (in dollars per gallon) -/
  rates : Fin 4 → ℝ
  /-- The amount spent on diesel each year (in dollars) -/
  annual_spend : ℝ
  /-- The mean cost of diesel over the 4-year period (in dollars per gallon) -/
  mean_cost : ℝ

/-- Theorem stating that given the conditions, the individual yearly rates cannot be uniquely determined -/
theorem diesel_rates_indeterminable (data : DieselPurchaseData) : 
  data.mean_cost = 1.52 → 
  (∀ (i j : Fin 4), i ≠ j → data.rates i ≠ data.rates j) →
  (∀ (i : Fin 4), data.annual_spend / data.rates i = data.annual_spend / data.rates 0) →
  ¬∃! (rates : Fin 4 → ℝ), rates = data.rates :=
sorry


end NUMINAMATH_CALUDE_diesel_rates_indeterminable_l1136_113649


namespace NUMINAMATH_CALUDE_tower_construction_modulo_l1136_113692

/-- Represents the number of towers that can be built using cubes up to size n -/
def T : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 2
| (n + 1) => if n ≥ 2 then 4 * T n else 3 * T n

/-- The problem statement -/
theorem tower_construction_modulo :
  T 10 % 1000 = 304 := by
  sorry

end NUMINAMATH_CALUDE_tower_construction_modulo_l1136_113692


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1136_113670

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_theorem : 
  (U \ (M ∪ N)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1136_113670


namespace NUMINAMATH_CALUDE_race_time_proof_l1136_113624

/-- Represents the time taken by runner A to complete the race -/
def time_A : ℝ := 235

/-- Represents the length of the race in meters -/
def race_length : ℝ := 1000

/-- Represents the distance by which A beats B in meters -/
def distance_difference : ℝ := 60

/-- Represents the time difference by which A beats B in seconds -/
def time_difference : ℝ := 15

/-- Theorem stating that given the race conditions, runner A completes the race in 235 seconds -/
theorem race_time_proof :
  (race_length / time_A = (race_length - distance_difference) / time_A) ∧
  (race_length / time_A = race_length / (time_A + time_difference)) →
  time_A = 235 := by
  sorry

end NUMINAMATH_CALUDE_race_time_proof_l1136_113624


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l1136_113615

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We only need to define two angles, as the third is determined by these two
  base_angle : Real
  vertex_angle : Real
  -- Condition that the sum of angles in a triangle is 180°
  angle_sum : base_angle * 2 + vertex_angle = 180

-- Theorem statement
theorem isosceles_triangle_vertex_angle 
  (triangle : IsoscelesTriangle) 
  (h : triangle.base_angle = 80 ∨ triangle.vertex_angle = 80) :
  triangle.vertex_angle = 80 ∨ triangle.vertex_angle = 20 := by
sorry


end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l1136_113615


namespace NUMINAMATH_CALUDE_number_multiplying_a_l1136_113676

theorem number_multiplying_a (a b : ℝ) (h1 : a * b ≠ 0) (h2 : a / 4 = b / 3) :
  ∃ x : ℝ, x * a = 4 * b ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplying_a_l1136_113676


namespace NUMINAMATH_CALUDE_both_heads_prob_l1136_113632

/-- Represents the outcome of flipping two coins simultaneously -/
inductive CoinFlip
| HH -- Both heads
| HT -- First head, second tail
| TH -- First tail, second head
| TT -- Both tails

/-- The probability of getting a specific outcome when flipping two fair coins -/
def flip_prob : CoinFlip → ℚ
| CoinFlip.HH => 1/4
| CoinFlip.HT => 1/4
| CoinFlip.TH => 1/4
| CoinFlip.TT => 1/4

/-- The process of flipping coins until at least one head appears -/
def flip_until_head : ℕ → ℚ
| 0 => flip_prob CoinFlip.HH
| (n+1) => flip_prob CoinFlip.TT * flip_until_head n

/-- The theorem stating the probability of both coins showing heads when the process stops -/
theorem both_heads_prob : (∑' n, flip_until_head n) = 1/3 :=
sorry


end NUMINAMATH_CALUDE_both_heads_prob_l1136_113632


namespace NUMINAMATH_CALUDE_pattern_equation_l1136_113620

theorem pattern_equation (n : ℕ) (h : n ≥ 2) :
  n * Real.sqrt (n / (n^2 - 1)) = Real.sqrt (n + n / (n^2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_pattern_equation_l1136_113620


namespace NUMINAMATH_CALUDE_inverse_of_composed_linear_functions_l1136_113654

/-- Given two functions p and q, we define r as their composition and prove its inverse -/
theorem inverse_of_composed_linear_functions 
  (p q r : ℝ → ℝ)
  (hp : ∀ x, p x = 4 * x - 7)
  (hq : ∀ x, q x = 3 * x + 2)
  (hr : ∀ x, r x = p (q x))
  : (∀ x, r x = 12 * x + 1) ∧ 
    (∀ x, Function.invFun r x = (x - 1) / 12) := by
  sorry


end NUMINAMATH_CALUDE_inverse_of_composed_linear_functions_l1136_113654


namespace NUMINAMATH_CALUDE_parabola_vertex_l1136_113609

/-- The vertex of the parabola y = -2(x-2)^2 - 5 is at the point (2, -5) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -2 * (x - 2)^2 - 5 → (2, -5) = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1136_113609


namespace NUMINAMATH_CALUDE_rational_inequality_solution_l1136_113610

theorem rational_inequality_solution (x : ℝ) : 
  x ≠ -5 → ((x - 2) / (x + 5) ≥ 0 ↔ x < -5 ∨ x ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_rational_inequality_solution_l1136_113610


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l1136_113601

/-- For non-zero real numbers a, b, c, if they form a geometric sequence,
    then their reciprocals and their squares also form geometric sequences. -/
theorem geometric_sequence_properties (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
    (h_geometric : b^2 = a * c) : 
  (1 / b)^2 = (1 / a) * (1 / c) ∧ (b^2)^2 = a^2 * c^2 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_properties_l1136_113601


namespace NUMINAMATH_CALUDE_expected_black_pairs_60_30_l1136_113603

/-- The expected number of adjacent black card pairs in a circular arrangement -/
def expected_black_pairs (total_cards : ℕ) (black_cards : ℕ) : ℚ :=
  (black_cards : ℚ) * ((black_cards - 1 : ℚ) / (total_cards - 1 : ℚ))

/-- Theorem: Expected number of adjacent black pairs in a 60-card deck with 30 black cards -/
theorem expected_black_pairs_60_30 :
  expected_black_pairs 60 30 = 870 / 59 := by
  sorry

end NUMINAMATH_CALUDE_expected_black_pairs_60_30_l1136_113603


namespace NUMINAMATH_CALUDE_computation_proof_l1136_113643

theorem computation_proof : 8 * (250 / 3 + 50 / 6 + 16 / 32 + 2) = 2260 / 3 := by
  sorry

end NUMINAMATH_CALUDE_computation_proof_l1136_113643


namespace NUMINAMATH_CALUDE_smallest_n_for_fraction_inequality_l1136_113629

theorem smallest_n_for_fraction_inequality : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℤ), 0 < m → m < 2001 → 
    ∃ (k : ℤ), (m : ℚ) / 2001 < (k : ℚ) / n ∧ (k : ℚ) / n < ((m + 1) : ℚ) / 2002) ∧
  (∀ (n' : ℕ), 0 < n' → n' < n → 
    ∃ (m : ℤ), 0 < m ∧ m < 2001 ∧
      ∀ (k : ℤ), ¬((m : ℚ) / 2001 < (k : ℚ) / n' ∧ (k : ℚ) / n' < ((m + 1) : ℚ) / 2002)) ∧
  n = 4003 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_fraction_inequality_l1136_113629


namespace NUMINAMATH_CALUDE_smallest_k_and_digit_sum_l1136_113625

-- Define the function to count digits
def countDigits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + countDigits (n / 10)

-- Define the function to sum digits
def sumDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumDigits (n / 10)

-- Theorem statement
theorem smallest_k_and_digit_sum :
  ∃ k : ℕ, 
    (k > 0) ∧
    (∀ j : ℕ, j > 0 → j < k → countDigits ((2^j) * (5^300)) < 303) ∧
    (countDigits ((2^k) * (5^300)) = 303) ∧
    (k = 307) ∧
    (sumDigits ((2^k) * (5^300)) = 11) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_and_digit_sum_l1136_113625


namespace NUMINAMATH_CALUDE_paul_crayons_left_l1136_113657

/-- The number of crayons Paul had at the end of the school year -/
def crayons_left (initial_crayons lost_crayons : ℕ) : ℕ :=
  initial_crayons - lost_crayons

/-- Theorem: Paul had 291 crayons left at the end of the school year -/
theorem paul_crayons_left : crayons_left 606 315 = 291 := by
  sorry

end NUMINAMATH_CALUDE_paul_crayons_left_l1136_113657


namespace NUMINAMATH_CALUDE_exists_unique_max_N_l1136_113665

/-- The number of positive divisors of a positive integer -/
def d (n : ℕ+) : ℕ+ := sorry

/-- The function f(n) = d(n) / (n^(1/3)) -/
noncomputable def f (n : ℕ+) : ℝ := (d n : ℝ) / n.val ^ (1/3 : ℝ)

/-- The sum of digits of a positive integer -/
def sum_of_digits (n : ℕ+) : ℕ := sorry

/-- The theorem stating the existence of a unique N maximizing f(n) -/
theorem exists_unique_max_N : ∃! N : ℕ+, (∀ n : ℕ+, n ≠ N → f N > f n) ∧ sum_of_digits N = 6 := by sorry

end NUMINAMATH_CALUDE_exists_unique_max_N_l1136_113665


namespace NUMINAMATH_CALUDE_dans_initial_amount_l1136_113671

/-- Dan's initial amount of money -/
def initial_amount : ℝ := 4

/-- The cost of the candy bar -/
def candy_cost : ℝ := 1

/-- The amount Dan had left after buying the candy bar -/
def remaining_amount : ℝ := 3

/-- Theorem stating that Dan's initial amount equals the sum of the remaining amount and the candy cost -/
theorem dans_initial_amount : initial_amount = remaining_amount + candy_cost := by
  sorry

end NUMINAMATH_CALUDE_dans_initial_amount_l1136_113671


namespace NUMINAMATH_CALUDE_complex_modulus_product_l1136_113686

theorem complex_modulus_product : Complex.abs ((10 - 6*I) * (7 + 24*I)) = 25 * Real.sqrt 136 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l1136_113686


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l1136_113612

theorem painted_cube_theorem (n : ℕ) (h1 : n > 2) :
  (n - 2)^3 = 6 * (n - 2)^2 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l1136_113612


namespace NUMINAMATH_CALUDE_expression_value_l1136_113645

theorem expression_value (a b c : ℝ) (h : a^2 + b = b^2 + c ∧ b^2 + c = c^2 + a) :
  a * (a^2 - b^2) + b * (b^2 - c^2) + c * (c^2 - a^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1136_113645


namespace NUMINAMATH_CALUDE_spherical_caps_ratio_l1136_113661

/-- 
Given a sphere of radius 1 cut by a plane into two spherical caps, 
if the combined surface area of the caps is 25% greater than the 
surface area of the original sphere, then the ratio of the surface 
areas of the larger cap to the smaller cap is (5 + 2√2) : (5 - 2√2).
-/
theorem spherical_caps_ratio (m₁ m₂ : ℝ) (ρ : ℝ) : 
  (0 < m₁) → (0 < m₂) → (0 < ρ) →
  (m₁ + m₂ = 2) →
  (2 * π * m₁ + π * ρ^2 + 2 * π * m₂ + π * ρ^2 = 5 * π) →
  (ρ^2 = 1 - (1 - m₁)^2) →
  (ρ^2 = 1 - (1 - m₂)^2) →
  ((2 * π * m₁ + π * ρ^2) / (2 * π * m₂ + π * ρ^2) = (5 + 2 * Real.sqrt 2) / (5 - 2 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_spherical_caps_ratio_l1136_113661


namespace NUMINAMATH_CALUDE_min_value_of_quadratic_l1136_113623

theorem min_value_of_quadratic (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) :
  ∃ (z : ℝ), z = x^2 + 4*y^2 ∧ z ≥ 4 ∧ ∀ (w : ℝ), w = x^2 + 4*y^2 → w ≥ z := by
sorry

end NUMINAMATH_CALUDE_min_value_of_quadratic_l1136_113623


namespace NUMINAMATH_CALUDE_units_digit_of_large_product_l1136_113634

theorem units_digit_of_large_product : ∃ n : ℕ, n < 10 ∧ 2^1007 * 6^1008 * 14^1009 ≡ n [ZMOD 10] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_large_product_l1136_113634


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1136_113600

theorem inscribed_circle_radius (PQ PR QR : ℝ) (h_PQ : PQ = 30) (h_PR : PR = 26) (h_QR : QR = 28) :
  let s := (PQ + PR + QR) / 2
  let area := Real.sqrt (s * (s - PQ) * (s - PR) * (s - QR))
  area / s = 8 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1136_113600


namespace NUMINAMATH_CALUDE_cos_75_degrees_l1136_113664

theorem cos_75_degrees :
  Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_degrees_l1136_113664
