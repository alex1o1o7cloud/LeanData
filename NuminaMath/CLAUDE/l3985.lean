import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3985_398541

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), 
  (x₁ = -3 ∧ x₂ = -1) ∧ 
  (∀ x : ℝ, x^2 + 4*x + 3 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3985_398541


namespace NUMINAMATH_CALUDE_position_selection_count_l3985_398517

/-- The number of people in the group --/
def group_size : ℕ := 6

/-- The number of positions to be filled --/
def num_positions : ℕ := 3

/-- Theorem: The number of ways to choose a President, Vice-President, and Secretary
    from a group of 6 people, where all positions must be held by different individuals,
    is equal to 120. --/
theorem position_selection_count :
  (group_size.factorial) / ((group_size - num_positions).factorial) = 120 := by
  sorry

end NUMINAMATH_CALUDE_position_selection_count_l3985_398517


namespace NUMINAMATH_CALUDE_faster_train_speed_l3985_398531

/-- Calculates the speed of a faster train given the conditions of the problem -/
theorem faster_train_speed
  (train_length : ℝ)
  (slower_speed : ℝ)
  (passing_time : ℝ)
  (h1 : train_length = 100)  -- Length of each train in meters
  (h2 : slower_speed = 36)   -- Speed of slower train in km/hr
  (h3 : passing_time = 72)   -- Time to pass in seconds
  : ∃ (faster_speed : ℝ), faster_speed = 86 :=
by
  sorry

#check faster_train_speed

end NUMINAMATH_CALUDE_faster_train_speed_l3985_398531


namespace NUMINAMATH_CALUDE_elder_age_problem_l3985_398598

theorem elder_age_problem (y e : ℕ) : 
  e = y + 20 →                 -- The ages differ by 20 years
  e - 4 = 5 * (y - 4) →        -- 4 years ago, elder was 5 times younger's age
  e = 29                       -- Elder's present age is 29
  := by sorry

end NUMINAMATH_CALUDE_elder_age_problem_l3985_398598


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3985_398529

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The theorem to prove -/
theorem point_in_fourth_quadrant :
  let P : Point := ⟨3, -3⟩
  is_in_fourth_quadrant P := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3985_398529


namespace NUMINAMATH_CALUDE_rectangle_90_42_cut_result_l3985_398550

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

end NUMINAMATH_CALUDE_rectangle_90_42_cut_result_l3985_398550


namespace NUMINAMATH_CALUDE_min_jumps_to_visit_all_l3985_398558

/-- Represents a jump on the circle -/
inductive Jump
| Two  : Jump  -- Jump of 2 points
| Three : Jump -- Jump of 3 points

/-- The number of points on the circle -/
def numPoints : ℕ := 2016

/-- Function to calculate the total distance covered by a sequence of jumps -/
def totalDistance (jumps : List Jump) : ℕ :=
  jumps.foldl (fun acc jump => acc + match jump with
    | Jump.Two => 2
    | Jump.Three => 3) 0

/-- Predicate to check if a sequence of jumps visits all points -/
def visitsAllPoints (jumps : List Jump) : Prop :=
  totalDistance jumps % numPoints = 0 ∧ 
  jumps.length ≥ numPoints

/-- The main theorem stating the minimum number of jumps required -/
theorem min_jumps_to_visit_all : 
  ∃ (jumps : List Jump), visitsAllPoints jumps ∧ 
    jumps.length = 2017 ∧ 
    (∀ (other_jumps : List Jump), visitsAllPoints other_jumps → 
      other_jumps.length ≥ 2017) := by
  sorry

end NUMINAMATH_CALUDE_min_jumps_to_visit_all_l3985_398558


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l3985_398570

theorem no_positive_integer_solution :
  ¬ ∃ (a b c : ℕ+), a^2 + b^2 = 4 * c + 3 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l3985_398570


namespace NUMINAMATH_CALUDE_line_inclination_angle_l3985_398547

theorem line_inclination_angle (x y : ℝ) :
  let line_equation := (2 * x - 2 * y - 1 = 0)
  let slope := (2 : ℝ) / 2
  let angle_of_inclination := Real.arctan slope
  line_equation → angle_of_inclination = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l3985_398547


namespace NUMINAMATH_CALUDE_limit_of_sequence_l3985_398525

/-- The sum of the first n multiples of 3 -/
def S (n : ℕ) : ℕ := 3 * n * (n + 1) / 2

/-- The sequence we're interested in -/
def a (n : ℕ) : ℚ := (S n : ℚ) / (n^2 + 4 : ℚ)

theorem limit_of_sequence :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - 3/2| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_sequence_l3985_398525


namespace NUMINAMATH_CALUDE_cookie_earnings_proof_l3985_398522

/-- The amount earned by girl scouts from selling cookies -/
def cookie_earnings : ℝ := 30

/-- The cost per person to go to the pool -/
def pool_cost_per_person : ℝ := 2.5

/-- The number of people going to the pool -/
def number_of_people : ℕ := 10

/-- The amount left after paying for the pool -/
def amount_left : ℝ := 5

/-- Theorem stating that the cookie earnings equal $30 -/
theorem cookie_earnings_proof :
  cookie_earnings = pool_cost_per_person * number_of_people + amount_left :=
by sorry

end NUMINAMATH_CALUDE_cookie_earnings_proof_l3985_398522


namespace NUMINAMATH_CALUDE_intersection_of_lines_l3985_398556

theorem intersection_of_lines : ∃! p : ℚ × ℚ, 
  8 * p.1 - 3 * p.2 = 24 ∧ 5 * p.1 + 2 * p.2 = 17 ∧ p = (99/31, 16/31) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l3985_398556


namespace NUMINAMATH_CALUDE_parabola_intersection_count_l3985_398542

/-- The parabola is defined by the function f(x) = 2x^2 - 4x + 1 --/
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 1

/-- The number of intersection points between the parabola and the coordinate axes --/
def intersection_count : ℕ := 3

/-- Theorem stating that the parabola intersects the coordinate axes at exactly 3 points --/
theorem parabola_intersection_count :
  (∃! y, y = f 0) ∧ 
  (∃! x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  intersection_count = 3 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_count_l3985_398542


namespace NUMINAMATH_CALUDE_pentagon_sum_problem_l3985_398572

theorem pentagon_sum_problem : ∃ (a b c d e : ℝ),
  a + b = 1 ∧
  b + c = 2 ∧
  c + d = 3 ∧
  d + e = 4 ∧
  e + a = 5 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_sum_problem_l3985_398572


namespace NUMINAMATH_CALUDE_muscovy_duck_percentage_l3985_398520

theorem muscovy_duck_percentage (total_ducks : ℕ) (female_muscovy : ℕ) 
  (h1 : total_ducks = 40)
  (h2 : female_muscovy = 6)
  (h3 : (female_muscovy : ℝ) / ((total_ducks : ℝ) * 0.5) = 0.3) :
  (total_ducks : ℝ) * 0.5 = (total_ducks : ℝ) * 0.5 := by
  sorry

#check muscovy_duck_percentage

end NUMINAMATH_CALUDE_muscovy_duck_percentage_l3985_398520


namespace NUMINAMATH_CALUDE_x_value_proof_l3985_398585

theorem x_value_proof (x : ℝ) : x = 2 * (1/x) * (-x) - 5 → x = -7 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3985_398585


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_six_l3985_398509

theorem largest_four_digit_divisible_by_six : 
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 6 = 0 → n ≤ 9996 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_six_l3985_398509


namespace NUMINAMATH_CALUDE_units_digit_of_product_l3985_398599

theorem units_digit_of_product (n : ℕ) : 
  (2^2023 * 5^2024 * 11^2025) % 10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l3985_398599


namespace NUMINAMATH_CALUDE_units_digit_theorem_l3985_398507

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the property we want to prove
def propertyHolds (n : ℕ) : Prop :=
  n > 0 → unitsDigit ((35 ^ n) + (93 ^ 45)) = 8

-- The theorem statement
theorem units_digit_theorem :
  ∀ n : ℕ, propertyHolds n :=
sorry

end NUMINAMATH_CALUDE_units_digit_theorem_l3985_398507


namespace NUMINAMATH_CALUDE_jane_has_66_robots_l3985_398567

/-- The number of car robots each person has -/
structure CarRobots where
  tom : ℕ
  michael : ℕ
  bob : ℕ
  sarah : ℕ
  jane : ℕ

/-- The conditions of the car robot collections -/
def satisfiesConditions (c : CarRobots) : Prop :=
  c.tom = 15 ∧
  c.michael = 3 * c.tom - 5 ∧
  c.bob = 8 * (c.tom + c.michael) ∧
  c.sarah = c.bob / 2 - 7 ∧
  c.jane = (c.sarah - c.tom) / 3

/-- Theorem stating that Jane has 66 car robots -/
theorem jane_has_66_robots (c : CarRobots) (h : satisfiesConditions c) : c.jane = 66 := by
  sorry

end NUMINAMATH_CALUDE_jane_has_66_robots_l3985_398567


namespace NUMINAMATH_CALUDE_simplify_sqrt_500_l3985_398524

theorem simplify_sqrt_500 : Real.sqrt 500 = 10 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_500_l3985_398524


namespace NUMINAMATH_CALUDE_election_winner_votes_l3985_398508

theorem election_winner_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) :
  winner_percentage = 60 / 100 →
  vote_difference = 288 →
  winner_percentage * total_votes - (1 - winner_percentage) * total_votes = vote_difference →
  winner_percentage * total_votes = 864 :=
by
  sorry

end NUMINAMATH_CALUDE_election_winner_votes_l3985_398508


namespace NUMINAMATH_CALUDE_election_win_percentage_l3985_398557

theorem election_win_percentage 
  (total_votes : ℕ) 
  (geoff_percentage : ℚ) 
  (additional_votes_needed : ℕ) 
  (h1 : total_votes = 6000)
  (h2 : geoff_percentage = 1 / 100)
  (h3 : additional_votes_needed = 3000) : 
  ∃ (x : ℚ), x > 51 / 100 ∧ 
    x * total_votes ≤ (geoff_percentage * total_votes + additional_votes_needed) ∧ 
    ∀ (y : ℚ), y < x → y * total_votes < (geoff_percentage * total_votes + additional_votes_needed) :=
by
  sorry

end NUMINAMATH_CALUDE_election_win_percentage_l3985_398557


namespace NUMINAMATH_CALUDE_smallest_multiple_l3985_398564

theorem smallest_multiple (n : ℕ) : n = 544 ↔ 
  (∃ k : ℕ, n = 17 * k) ∧ 
  (∃ m : ℕ, n = 53 * m + 7) ∧ 
  (∀ x : ℕ, x < n → ¬(∃ k m : ℕ, x = 17 * k ∧ x = 53 * m + 7)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l3985_398564


namespace NUMINAMATH_CALUDE_candy_distribution_l3985_398573

theorem candy_distribution (initial_candies : ℕ) (additional_candies : ℕ) (num_friends : ℕ) 
  (h1 : initial_candies = 10)
  (h2 : additional_candies = 4)
  (h3 : num_friends = 7)
  (h4 : num_friends > 0) :
  (initial_candies + additional_candies) / num_friends = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3985_398573


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l3985_398532

/-- Triangle ABC with given side lengths and angle -/
structure TriangleABC where
  AB : ℝ
  BC : ℝ
  cosC : ℝ
  h_AB : AB = Real.sqrt 2
  h_BC : BC = 1
  h_cosC : cosC = 3/4

/-- The main theorem about TriangleABC -/
theorem triangle_ABC_properties (t : TriangleABC) :
  let sinA := Real.sqrt (14) / 8
  let dot_product := -(3/2 : ℝ)
  (∃ (CA : ℝ), sinA = Real.sqrt (1 - t.cosC^2) * t.BC / t.AB) ∧
  (∃ (CA : ℝ), dot_product = t.BC * CA * (-t.cosC)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l3985_398532


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3985_398554

/-- The equation of a hyperbola sharing foci with an ellipse and passing through a point -/
theorem hyperbola_equation (x y : ℝ) : 
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    (∀ (x y : ℝ), x^2 / 4 + y^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
    c^2 = a^2 - b^2 ∧
    (x - 2)^2 / a^2 - (y - 1)^2 / b^2 = 1) →
  x^2 / 2 - y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3985_398554


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3985_398506

theorem inequality_solution_set :
  {x : ℝ | x * (x - 1) > 0} = {x : ℝ | x < 0 ∨ x > 1} := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3985_398506


namespace NUMINAMATH_CALUDE_probability_theorem_l3985_398589

def is_valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 100 ∧ 1 ≤ b ∧ b ≤ 100 ∧ a ≠ b

def satisfies_condition (a b : ℕ) : Prop :=
  ∃ k : ℕ, a * b + a + b = 7 * k - 2

def total_pairs : ℕ := Nat.choose 100 2

def valid_pairs : ℕ := 105

theorem probability_theorem :
  (valid_pairs : ℚ) / total_pairs = 7 / 330 := by sorry

end NUMINAMATH_CALUDE_probability_theorem_l3985_398589


namespace NUMINAMATH_CALUDE_log_equation_solution_l3985_398575

theorem log_equation_solution (x : ℝ) (h : Real.log x / Real.log 3 * Real.log 3 / Real.log 4 = 4) : x = 256 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3985_398575


namespace NUMINAMATH_CALUDE_tablet_cash_savings_l3985_398512

/-- Represents the savings when buying a tablet in cash versus installments -/
def tablet_savings (cash_price : ℕ) (down_payment : ℕ) 
  (first_4_months : ℕ) (next_4_months : ℕ) (last_4_months : ℕ) : ℕ :=
  (down_payment + 4 * first_4_months + 4 * next_4_months + 4 * last_4_months) - cash_price

/-- Theorem stating the savings when buying the tablet in cash -/
theorem tablet_cash_savings : 
  tablet_savings 450 100 40 35 30 = 70 := by
  sorry

end NUMINAMATH_CALUDE_tablet_cash_savings_l3985_398512


namespace NUMINAMATH_CALUDE_ab_range_l3985_398505

theorem ab_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a * b = a + b) : a * b ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ab_range_l3985_398505


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l3985_398516

theorem absolute_value_simplification : |-4^2 - 6| = 22 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l3985_398516


namespace NUMINAMATH_CALUDE_sphere_radius_in_cone_l3985_398510

/-- A right circular cone with four congruent spheres inside -/
structure ConeSpheres where
  base_radius : ℝ
  height : ℝ
  sphere_radius : ℝ
  is_right_circular : base_radius > 0 ∧ height > 0
  spheres_tangent : sphere_radius > 0
  spheres_fit : sphere_radius < base_radius ∧ sphere_radius < height

/-- The theorem stating the radius of each sphere in the specific configuration -/
theorem sphere_radius_in_cone (cs : ConeSpheres) 
  (h1 : cs.base_radius = 8)
  (h2 : cs.height = 15) :
  cs.sphere_radius = 8 * Real.sqrt 3 / 17 := by
  sorry


end NUMINAMATH_CALUDE_sphere_radius_in_cone_l3985_398510


namespace NUMINAMATH_CALUDE_associate_prof_charts_l3985_398503

theorem associate_prof_charts (total_people : ℕ) (total_pencils : ℕ) (total_charts : ℕ)
  (h1 : total_people = 8)
  (h2 : total_pencils = 10)
  (h3 : total_charts = 14) :
  ∃ (assoc_prof : ℕ) (asst_prof : ℕ) (charts_per_assoc : ℕ),
    assoc_prof + asst_prof = total_people ∧
    2 * assoc_prof + asst_prof = total_pencils ∧
    charts_per_assoc * assoc_prof + 2 * asst_prof = total_charts ∧
    charts_per_assoc = 1 :=
by sorry

end NUMINAMATH_CALUDE_associate_prof_charts_l3985_398503


namespace NUMINAMATH_CALUDE_system1_solution_system2_solution_l3985_398568

-- System 1
theorem system1_solution :
  ∃ (x y : ℝ), 2*x + 3*y = -1 ∧ y = 4*x - 5 ∧ x = 1 ∧ y = -1 := by
  sorry

-- System 2
theorem system2_solution :
  ∃ (x y : ℝ), 3*x + 2*y = 20 ∧ 4*x - 5*y = 19 ∧ x = 6 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_system1_solution_system2_solution_l3985_398568


namespace NUMINAMATH_CALUDE_apples_to_eat_raw_l3985_398502

theorem apples_to_eat_raw (total : ℕ) (wormy : ℕ) (bruised : ℕ) : 
  total = 85 →
  wormy = total / 5 →
  bruised = wormy + 9 →
  total - bruised - wormy = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_apples_to_eat_raw_l3985_398502


namespace NUMINAMATH_CALUDE_intersection_A_B_l3985_398545

open Set Real

-- Define sets A and B
def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B : Set ℝ := Ioo 0 3

-- State the theorem
theorem intersection_A_B : A ∩ B = Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3985_398545


namespace NUMINAMATH_CALUDE_income_ratio_proof_l3985_398528

def uma_income : ℕ := 20000
def bala_income : ℕ := 15000
def uma_savings : ℕ := 5000
def bala_savings : ℕ := 5000
def expenditure_ratio : Rat := 3 / 2

theorem income_ratio_proof :
  let uma_expenditure := uma_income - uma_savings
  let bala_expenditure := bala_income - bala_savings
  (uma_expenditure : Rat) / bala_expenditure = expenditure_ratio →
  (uma_income : Rat) / bala_income = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_income_ratio_proof_l3985_398528


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_negative_one_l3985_398546

open Real

theorem sin_cos_sum_equals_negative_one :
  sin (200 * π / 180) * cos (110 * π / 180) + cos (160 * π / 180) * sin (70 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_negative_one_l3985_398546


namespace NUMINAMATH_CALUDE_special_ellipse_eccentricity_l3985_398574

/-- An ellipse with a special point P -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  P : ℝ × ℝ
  h_on_ellipse : (P.1^2 / a^2) + (P.2^2 / b^2) = 1
  h_PF1_perpendicular : P.1 = -((a^2 - b^2).sqrt)
  h_PF2_parallel : P.2 / (P.1 + ((a^2 - b^2).sqrt)) = -b / a

/-- The eccentricity of an ellipse with a special point P is √5/5 -/
theorem special_ellipse_eccentricity (E : SpecialEllipse) :
  ((E.a^2 - E.b^2) / E.a^2).sqrt = (5 : ℝ).sqrt / 5 := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_eccentricity_l3985_398574


namespace NUMINAMATH_CALUDE_vertical_dominoes_even_l3985_398504

/-- A grid with even rows colored white and odd rows colored black -/
structure ColoredGrid where
  rows : ℕ
  cols : ℕ

/-- A domino placement on a colored grid -/
structure DominoPlacement (grid : ColoredGrid) where
  horizontal : Finset (ℕ × ℕ)  -- Set of starting positions for horizontal dominoes
  vertical : Finset (ℕ × ℕ)    -- Set of starting positions for vertical dominoes

/-- Predicate to check if a domino placement is valid -/
def is_valid_placement (grid : ColoredGrid) (placement : DominoPlacement grid) : Prop :=
  ∀ (i j : ℕ), i < grid.rows ∧ j < grid.cols →
    ((i, j) ∈ placement.horizontal → j + 1 < grid.cols) ∧
    ((i, j) ∈ placement.vertical → i + 1 < grid.rows)

/-- The main theorem: The number of vertically placed dominoes is even -/
theorem vertical_dominoes_even (grid : ColoredGrid) (placement : DominoPlacement grid)
  (h_valid : is_valid_placement grid placement) :
  Even placement.vertical.card :=
sorry

end NUMINAMATH_CALUDE_vertical_dominoes_even_l3985_398504


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3985_398518

theorem simplify_and_rationalize :
  (Real.sqrt 6 / Real.sqrt 7) * (Real.sqrt 6 / Real.sqrt 14) * (Real.sqrt 9 / Real.sqrt 21) = Real.sqrt 2058 / 114 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3985_398518


namespace NUMINAMATH_CALUDE_line_transformation_l3985_398582

-- Define the original line
def original_line (x : ℝ) : ℝ := x

-- Define rotation by 90 degrees counterclockwise
def rotate_90 (x y : ℝ) : ℝ × ℝ := (-y, x)

-- Define vertical shift by 1 unit
def shift_up (y : ℝ) : ℝ := y + 1

-- Theorem statement
theorem line_transformation :
  ∀ x : ℝ, 
  let (x', y') := rotate_90 x (original_line x)
  shift_up y' = -x' + 1 := by
  sorry

end NUMINAMATH_CALUDE_line_transformation_l3985_398582


namespace NUMINAMATH_CALUDE_verandah_area_is_124_l3985_398500

/-- Calculates the area of a verandah surrounding a rectangular room. -/
def verandahArea (roomLength : ℝ) (roomWidth : ℝ) (verandahWidth : ℝ) : ℝ :=
  (roomLength + 2 * verandahWidth) * (roomWidth + 2 * verandahWidth) - roomLength * roomWidth

/-- Theorem: The area of the verandah is 124 square meters. -/
theorem verandah_area_is_124 :
  verandahArea 15 12 2 = 124 := by
  sorry

#eval verandahArea 15 12 2

end NUMINAMATH_CALUDE_verandah_area_is_124_l3985_398500


namespace NUMINAMATH_CALUDE_wrapping_paper_area_formula_l3985_398523

/-- The area of square wrapping paper required to wrap a rectangular box -/
def wrapping_paper_area (l w h : ℝ) : ℝ :=
  (l + 4 + 2 * h) ^ 2

/-- Theorem stating the formula for the area of wrapping paper -/
theorem wrapping_paper_area_formula (l w h : ℝ) (hl : l > 0) (hw : w > 0) (hh : h > 0) :
  wrapping_paper_area l w h = l^2 + 8*l + 16 + 4*l*h + 16*h + 4*h^2 := by
  sorry

#check wrapping_paper_area_formula

end NUMINAMATH_CALUDE_wrapping_paper_area_formula_l3985_398523


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3985_398595

theorem polynomial_factorization (x : ℝ) (h : x^3 ≠ 1) :
  x^12 + x^6 + 1 = (x^6 + x^3 + 1) * (x^6 - x^3 + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3985_398595


namespace NUMINAMATH_CALUDE_solve_for_b_l3985_398592

theorem solve_for_b (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 35 * 45 * b) : b = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l3985_398592


namespace NUMINAMATH_CALUDE_midpoint_property_l3985_398560

/-- Given two points A and B in the plane, if C is their midpoint,
    then 3 times the x-coordinate of C minus 5 times the y-coordinate of C equals 6. -/
theorem midpoint_property (A B : ℝ × ℝ) (h : A = (20, 10) ∧ B = (4, 2)) :
  let C : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  3 * C.1 - 5 * C.2 = 6 := by sorry

end NUMINAMATH_CALUDE_midpoint_property_l3985_398560


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3985_398584

theorem solution_set_inequality (x : ℝ) :
  (x * (x + 2) < 3) ↔ (-3 < x ∧ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3985_398584


namespace NUMINAMATH_CALUDE_initial_leaves_count_l3985_398530

/-- The number of leaves Mikey had initially -/
def initial_leaves : ℕ := sorry

/-- The number of leaves that blew away -/
def blown_leaves : ℕ := 244

/-- The number of leaves left -/
def remaining_leaves : ℕ := 112

/-- Theorem stating that the initial number of leaves is 356 -/
theorem initial_leaves_count : initial_leaves = 356 := by
  sorry

end NUMINAMATH_CALUDE_initial_leaves_count_l3985_398530


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3985_398590

theorem fractional_equation_solution : 
  ∃ x : ℝ, (1 / x = 2 / (x + 3)) ∧ (x = 3) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3985_398590


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3985_398540

theorem algebraic_expression_value (x : ℝ) :
  12 * x - 8 * x^2 = -1 → 4 * x^2 - 6 * x + 5 = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3985_398540


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3985_398569

theorem negation_of_proposition (a b : ℝ) :
  ¬(a + b = 1 → a^2 + b^2 ≥ 1/2) ↔ (a + b ≠ 1 → a^2 + b^2 < 1/2) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3985_398569


namespace NUMINAMATH_CALUDE_chess_pieces_present_l3985_398501

/-- The number of pieces in a standard chess set -/
def standard_chess_pieces : ℕ := 32

/-- The number of missing chess pieces -/
def missing_pieces : ℕ := 8

/-- Theorem: The number of chess pieces present is 24 -/
theorem chess_pieces_present : 
  standard_chess_pieces - missing_pieces = 24 := by
  sorry

end NUMINAMATH_CALUDE_chess_pieces_present_l3985_398501


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3985_398533

theorem trigonometric_identities (θ : Real) 
  (h : Real.tan θ + (Real.tan θ)⁻¹ = 2) : 
  (Real.sin θ * Real.cos θ = 1/2) ∧ 
  ((Real.sin θ + Real.cos θ)^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3985_398533


namespace NUMINAMATH_CALUDE_largest_odd_sum_288_largest_odd_sum_288_is_43_l3985_398565

/-- Sum of first n consecutive odd integers -/
def sum_n_odd (n : ℕ) : ℕ := n^2

/-- Sum of odd integers from a to b inclusive -/
def sum_odd_range (a b : ℕ) : ℕ := 
  (sum_n_odd ((b - a) / 2 + 1)) - (sum_n_odd ((a - 1) / 2))

/-- The largest odd integer x such that the sum of all odd integers 
    from 13 to x inclusive is 288 -/
theorem largest_odd_sum_288 : 
  ∃ x : ℕ, x % 2 = 1 ∧ sum_odd_range 13 x = 288 ∧ 
  ∀ y : ℕ, y > x → y % 2 = 1 → sum_odd_range 13 y > 288 :=
sorry
 
/-- The largest odd integer x such that the sum of all odd integers 
    from 13 to x inclusive is 288 is equal to 43 -/
theorem largest_odd_sum_288_is_43 : 
  ∃! x : ℕ, x % 2 = 1 ∧ sum_odd_range 13 x = 288 ∧ 
  ∀ y : ℕ, y > x → y % 2 = 1 → sum_odd_range 13 y > 288 ∧ x = 43 :=
sorry

end NUMINAMATH_CALUDE_largest_odd_sum_288_largest_odd_sum_288_is_43_l3985_398565


namespace NUMINAMATH_CALUDE_square_side_ratio_l3985_398539

theorem square_side_ratio (area_ratio : ℚ) (h : area_ratio = 72 / 98) :
  ∃ (a b c : ℕ), 
    (a : ℚ) * Real.sqrt b / c = Real.sqrt (area_ratio) ∧ 
    a = 6 ∧ 
    b = 1 ∧ 
    c = 7 ∧ 
    a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_square_side_ratio_l3985_398539


namespace NUMINAMATH_CALUDE_quadratic_root_transformations_l3985_398521

/-- Given a quadratic equation x^2 + px + q = 0, this theorem proves the equations
    with roots differing by sign and reciprocal roots. -/
theorem quadratic_root_transformations (p q : ℝ) :
  let original := fun x : ℝ => x^2 + p*x + q
  let opposite_sign := fun x : ℝ => x^2 - p*x + q
  let reciprocal := fun x : ℝ => q*x^2 + p*x + 1
  (∀ x, original x = 0 → ∃ y, opposite_sign y = 0 ∧ y = -x) ∧
  (∀ x, original x = 0 → x ≠ 0 → ∃ y, reciprocal y = 0 ∧ y = 1/x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_transformations_l3985_398521


namespace NUMINAMATH_CALUDE_dorothy_score_l3985_398580

theorem dorothy_score (tatuya ivanna dorothy : ℚ) 
  (h1 : tatuya = 2 * ivanna)
  (h2 : ivanna = (3/5) * dorothy)
  (h3 : (tatuya + ivanna + dorothy) / 3 = 84) :
  dorothy = 90 := by
  sorry

end NUMINAMATH_CALUDE_dorothy_score_l3985_398580


namespace NUMINAMATH_CALUDE_block_stacks_ratio_l3985_398514

theorem block_stacks_ratio : 
  ∀ (stack1 stack2 stack3 stack4 stack5 : ℕ),
  stack1 = 7 →
  stack2 = stack1 + 3 →
  stack3 = stack2 - 6 →
  stack4 = stack3 + 10 →
  stack1 + stack2 + stack3 + stack4 + stack5 = 55 →
  stack5 / stack2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_block_stacks_ratio_l3985_398514


namespace NUMINAMATH_CALUDE_carwash_donation_percentage_l3985_398576

/-- Proves that the percentage of carwash proceeds donated is 90%, given the conditions of Hank's fundraising activities. -/
theorem carwash_donation_percentage
  (carwash_amount : ℝ)
  (bake_sale_amount : ℝ)
  (lawn_mowing_amount : ℝ)
  (bake_sale_donation_percentage : ℝ)
  (lawn_mowing_donation_percentage : ℝ)
  (total_donation : ℝ)
  (h1 : carwash_amount = 100)
  (h2 : bake_sale_amount = 80)
  (h3 : lawn_mowing_amount = 50)
  (h4 : bake_sale_donation_percentage = 0.75)
  (h5 : lawn_mowing_donation_percentage = 1)
  (h6 : total_donation = 200)
  (h7 : total_donation = carwash_amount * x + bake_sale_amount * bake_sale_donation_percentage + lawn_mowing_amount * lawn_mowing_donation_percentage)
  : x = 0.9 := by
  sorry

#check carwash_donation_percentage

end NUMINAMATH_CALUDE_carwash_donation_percentage_l3985_398576


namespace NUMINAMATH_CALUDE_inverse_proportion_l3985_398535

theorem inverse_proportion (a b : ℝ → ℝ) (k : ℝ) :
  (∀ x, a x * b x = k) →  -- a and b are inversely proportional
  (a 5 = 40) →            -- a = 40 when b = 5
  (b 5 = 5) →             -- explicitly stating b = 5
  (b 10 = 10) →           -- explicitly stating b = 10
  (a 10 = 20) :=          -- a = 20 when b = 10
by
  sorry


end NUMINAMATH_CALUDE_inverse_proportion_l3985_398535


namespace NUMINAMATH_CALUDE_cricket_bat_profit_l3985_398587

/-- Calculates the profit amount for a cricket bat sale -/
theorem cricket_bat_profit (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 850 →
  profit_percentage = 42.857142857142854 →
  ∃ (cost_price : ℝ) (profit : ℝ),
    cost_price > 0 ∧
    profit > 0 ∧
    selling_price = cost_price + profit ∧
    profit_percentage = (profit / cost_price) * 100 ∧
    profit = 255 := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_profit_l3985_398587


namespace NUMINAMATH_CALUDE_percentage_difference_l3985_398581

theorem percentage_difference (X : ℝ) (h : X > 0) : 
  let first_number := 0.70 * X
  let second_number := 0.63 * X
  (first_number - second_number) / first_number * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l3985_398581


namespace NUMINAMATH_CALUDE_exists_large_configuration_l3985_398555

/-- A configuration in the plane is a finite set of points where each point
    has at least k other points at a distance of exactly 1 unit. -/
def IsConfiguration (S : Set (ℝ × ℝ)) (k : ℕ) : Prop :=
  S.Finite ∧ 
  ∀ P ∈ S, ∃ T ⊆ S, T.ncard ≥ k ∧ ∀ Q ∈ T, Q ≠ P ∧ dist P Q = 1

/-- There exists a configuration of 3^1000 points where each point
    has at least 2000 other points at a distance of 1 unit. -/
theorem exists_large_configuration :
  ∃ S : Set (ℝ × ℝ), IsConfiguration S 2000 ∧ S.ncard = 3^1000 := by
  sorry


end NUMINAMATH_CALUDE_exists_large_configuration_l3985_398555


namespace NUMINAMATH_CALUDE_circle_properties_l3985_398562

-- Define the circle C
def circle_C (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the conditions
def center_on_line (a : ℝ) : ℝ × ℝ := (a, -2*a)
def point_A : ℝ × ℝ := (2, -1)
def tangent_line (p : ℝ × ℝ) : Prop := p.1 + p.2 = 1

-- Theorem statement
theorem circle_properties (a : ℝ) :
  let center := center_on_line a
  let C := circle_C center (|a - 2*a - 1| / Real.sqrt 2)
  point_A ∈ C ∧ (∃ p, p ∈ C ∧ tangent_line p) →
  (C = circle_C (1, -2) (Real.sqrt 2)) ∧
  (Set.Icc (-3 : ℝ) (-1) ⊆ {y | (0, y) ∈ C}) ∧
  (Set.Ioo (-3 : ℝ) (-1) ⊆ {y | (0, y) ∉ C}) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l3985_398562


namespace NUMINAMATH_CALUDE_smallest_a1_l3985_398583

def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = 7 * a (n - 1) - n

theorem smallest_a1 (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  sequence_property a →
  ∀ a1 : ℝ, (a 1 = a1 ∧ ∀ n, a n > 0) → a1 ≥ 13/36 :=
sorry

end NUMINAMATH_CALUDE_smallest_a1_l3985_398583


namespace NUMINAMATH_CALUDE_square_equation_solution_l3985_398538

theorem square_equation_solution : 
  ∃! y : ℤ, (2010 + y)^2 = y^2 :=
by
  -- The unique solution is y = -1005
  use -1005
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l3985_398538


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3985_398561

theorem least_subtraction_for_divisibility (n : ℕ) : 
  (∃ (x : ℕ), x = 46 ∧ 
   (∀ (y : ℕ), y < x → ¬(5 ∣ (9671 - y) ∧ 7 ∣ (9671 - y) ∧ 11 ∣ (9671 - y))) ∧
   (5 ∣ (9671 - x) ∧ 7 ∣ (9671 - x) ∧ 11 ∣ (9671 - x))) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3985_398561


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3985_398536

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔ (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3985_398536


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3985_398534

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + 3 > a^2 - 2*a - 1) ↔ (-1 < a ∧ a < 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3985_398534


namespace NUMINAMATH_CALUDE_unique_solution_l3985_398571

-- Define the equation
def equation (x : ℝ) : Prop := Real.rpow (5 - x) (1/3) + Real.sqrt (x + 2) = 3

-- Theorem statement
theorem unique_solution :
  ∃! x : ℝ, equation x ∧ x = 4 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l3985_398571


namespace NUMINAMATH_CALUDE_marbles_per_pack_l3985_398559

theorem marbles_per_pack (total_marbles : ℕ) (total_packs : ℕ) 
  (leo_packs manny_packs neil_packs : ℕ) : 
  total_marbles = 400 →
  leo_packs = 25 →
  manny_packs = total_packs / 4 →
  neil_packs = total_packs / 8 →
  leo_packs + manny_packs + neil_packs = total_packs →
  total_marbles / total_packs = 10 := by
  sorry

end NUMINAMATH_CALUDE_marbles_per_pack_l3985_398559


namespace NUMINAMATH_CALUDE_julia_tag_game_l3985_398549

theorem julia_tag_game (total : ℕ) (monday : ℕ) (tuesday : ℕ) 
  (h1 : total = 18) 
  (h2 : monday = 4) 
  (h3 : total = monday + tuesday) : 
  tuesday = 14 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_game_l3985_398549


namespace NUMINAMATH_CALUDE_f_range_upper_bound_l3985_398511

def f (x : ℝ) : ℝ := -x^2 + 2*x

theorem f_range_upper_bound (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 2, a < f x) → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_f_range_upper_bound_l3985_398511


namespace NUMINAMATH_CALUDE_total_savings_after_three_months_l3985_398544

def savings (n : ℕ) : ℕ := 10 + 30 * n

theorem total_savings_after_three_months : 
  savings 0 + savings 1 + savings 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_after_three_months_l3985_398544


namespace NUMINAMATH_CALUDE_cookie_calorie_count_l3985_398552

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

end NUMINAMATH_CALUDE_cookie_calorie_count_l3985_398552


namespace NUMINAMATH_CALUDE_sqrt_difference_square_l3985_398597

theorem sqrt_difference_square : (Real.sqrt 7 + Real.sqrt 6) * (Real.sqrt 7 - Real.sqrt 6) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_square_l3985_398597


namespace NUMINAMATH_CALUDE_line_always_intersects_ellipse_l3985_398579

/-- The range of m for which the line y = kx + 1 always intersects the ellipse x²/5 + y²/m = 1 -/
theorem line_always_intersects_ellipse (k : ℝ) (m : ℝ) :
  (∀ x y, y = k * x + 1 → x^2 / 5 + y^2 / m = 1) ↔ m ≥ 1 ∧ m ≠ 5 :=
sorry

end NUMINAMATH_CALUDE_line_always_intersects_ellipse_l3985_398579


namespace NUMINAMATH_CALUDE_jake_fewer_peaches_indeterminate_peach_difference_l3985_398593

-- Define the number of apples and peaches for Steven
def steven_apples : ℕ := 52
def steven_peaches : ℕ := 13

-- Define Jake's apples in terms of Steven's
def jake_apples : ℕ := steven_apples + 84

-- Define a variable for Jake's peaches (unknown, but less than Steven's)
variable (jake_peaches : ℕ)

-- Theorem stating that Jake's peaches are fewer than Steven's
theorem jake_fewer_peaches : jake_peaches < steven_peaches := by sorry

-- Theorem stating that the exact difference in peaches cannot be determined
theorem indeterminate_peach_difference :
  ¬ ∃ (diff : ℕ), ∀ (jake_peaches : ℕ), jake_peaches < steven_peaches →
    steven_peaches - jake_peaches = diff := by sorry

end NUMINAMATH_CALUDE_jake_fewer_peaches_indeterminate_peach_difference_l3985_398593


namespace NUMINAMATH_CALUDE_prescription_final_cost_l3985_398513

/-- Calculates the final cost of a prescription after cashback and rebate --/
theorem prescription_final_cost 
  (original_price : ℝ) 
  (cashback_percent : ℝ) 
  (rebate : ℝ) 
  (h1 : original_price = 150)
  (h2 : cashback_percent = 0.1)
  (h3 : rebate = 25) :
  original_price - (cashback_percent * original_price) - rebate = 110 := by
  sorry

#check prescription_final_cost

end NUMINAMATH_CALUDE_prescription_final_cost_l3985_398513


namespace NUMINAMATH_CALUDE_savings_over_three_years_l3985_398543

def multi_tariff_meter_cost : ℕ := 3500
def installation_cost : ℕ := 1100
def monthly_consumption : ℕ := 300
def night_consumption : ℕ := 230
def day_consumption : ℕ := monthly_consumption - night_consumption
def multi_tariff_day_rate : ℚ := 52/10
def multi_tariff_night_rate : ℚ := 34/10
def standard_rate : ℚ := 46/10

def monthly_cost_multi_tariff : ℚ :=
  (night_consumption : ℚ) * multi_tariff_night_rate + (day_consumption : ℚ) * multi_tariff_day_rate

def monthly_cost_standard : ℚ :=
  (monthly_consumption : ℚ) * standard_rate

def total_cost_multi_tariff (months : ℕ) : ℚ :=
  (multi_tariff_meter_cost : ℚ) + (installation_cost : ℚ) + monthly_cost_multi_tariff * (months : ℚ)

def total_cost_standard (months : ℕ) : ℚ :=
  monthly_cost_standard * (months : ℚ)

theorem savings_over_three_years :
  total_cost_standard 36 - total_cost_multi_tariff 36 = 3824 := by sorry

end NUMINAMATH_CALUDE_savings_over_three_years_l3985_398543


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3985_398548

theorem quadratic_equal_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + 1 = 0 ∧ (∀ y : ℝ, y^2 - a*y + 1 = 0 → y = x)) → 
  a = 2 ∨ a = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3985_398548


namespace NUMINAMATH_CALUDE_francisFamily_violins_l3985_398553

theorem francisFamily_violins :
  let ukuleles : ℕ := 2
  let guitars : ℕ := 4
  let ukuleleStrings : ℕ := 4
  let guitarStrings : ℕ := 6
  let violinStrings : ℕ := 4
  let totalStrings : ℕ := 40
  
  ∃ violins : ℕ,
    violins * violinStrings + ukuleles * ukuleleStrings + guitars * guitarStrings = totalStrings ∧
    violins = 2 :=
by sorry

end NUMINAMATH_CALUDE_francisFamily_violins_l3985_398553


namespace NUMINAMATH_CALUDE_solution_exists_l3985_398551

theorem solution_exists (N : ℝ) : ∃ x₁ x₂ x₃ x₄ : ℤ, 
  (x₁ > ⌊N⌋) ∧ (x₂ > ⌊N⌋) ∧ (x₃ > ⌊N⌋) ∧ (x₄ > ⌊N⌋) ∧
  (x₁^2 + x₂^2 + x₃^2 + x₄^2 : ℤ) = x₁*x₂*x₃ + x₁*x₂*x₄ + x₁*x₃*x₄ + x₂*x₃*x₄ :=
by
  sorry

end NUMINAMATH_CALUDE_solution_exists_l3985_398551


namespace NUMINAMATH_CALUDE_parallelogram_EFGH_area_l3985_398577

-- Define the parallelogram EFGH
def E : ℝ × ℝ := (1, 3)
def F : ℝ × ℝ := (5, 3)
def G : ℝ × ℝ := (6, 1)
def H : ℝ × ℝ := (2, 1)

-- Define the area function for a parallelogram
def parallelogram_area (a b c d : ℝ × ℝ) : ℝ :=
  let base := abs (b.1 - a.1)
  let height := abs (a.2 - d.2)
  base * height

-- Theorem statement
theorem parallelogram_EFGH_area :
  parallelogram_area E F G H = 8 := by sorry

end NUMINAMATH_CALUDE_parallelogram_EFGH_area_l3985_398577


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3985_398527

/-- An arithmetic sequence with given conditions -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a_4 : a 4 = 70
  a_21 : a 21 = -100

/-- Main theorem about the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (seq.a 1 = 100) ∧ 
  (∀ n, seq.a (n + 1) - seq.a n = -10) ∧
  (∀ n, seq.a n = -10 * n + 110) ∧
  (Finset.filter (fun n => -18 ≤ seq.a n ∧ seq.a n ≤ 18) (Finset.range 100)).card = 3 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3985_398527


namespace NUMINAMATH_CALUDE_min_winning_set_size_l3985_398526

/-- The set of allowed digits -/
def AllowedDigits : Finset Nat := {1, 2, 3, 4}

/-- A type representing a three-digit number using only allowed digits -/
structure ThreeDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  h1 : d1 ∈ AllowedDigits
  h2 : d2 ∈ AllowedDigits
  h3 : d3 ∈ AllowedDigits

/-- Function to count how many digits differ between two ThreeDigitNumbers -/
def diffCount (n1 n2 : ThreeDigitNumber) : Nat :=
  (if n1.d1 ≠ n2.d1 then 1 else 0) +
  (if n1.d2 ≠ n2.d2 then 1 else 0) +
  (if n1.d3 ≠ n2.d3 then 1 else 0)

/-- A set of ThreeDigitNumbers is winning if for any other ThreeDigitNumber,
    at least one number in the set differs from it by at most one digit -/
def isWinningSet (s : Finset ThreeDigitNumber) : Prop :=
  ∀ n : ThreeDigitNumber, ∃ m ∈ s, diffCount n m ≤ 1

/-- The main theorem: The minimum size of a winning set is 8 -/
theorem min_winning_set_size :
  (∃ s : Finset ThreeDigitNumber, isWinningSet s ∧ s.card = 8) ∧
  (∀ s : Finset ThreeDigitNumber, isWinningSet s → s.card ≥ 8) :=
sorry

end NUMINAMATH_CALUDE_min_winning_set_size_l3985_398526


namespace NUMINAMATH_CALUDE_robotics_club_enrollment_l3985_398566

theorem robotics_club_enrollment (total : ℕ) (cs : ℕ) (electronics : ℕ) (both : ℕ) 
  (h1 : total = 60)
  (h2 : cs = 45)
  (h3 : electronics = 33)
  (h4 : both = 25) :
  total - (cs + electronics - both) = 7 := by
  sorry

end NUMINAMATH_CALUDE_robotics_club_enrollment_l3985_398566


namespace NUMINAMATH_CALUDE_star_property_counterexample_l3985_398519

/-- Definition of the star operation -/
def star (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

/-- Theorem stating that 2(x ★ y) ≠ (2x) ★ (2y) for some real x and y -/
theorem star_property_counterexample : ∃ x y : ℝ, 2 * (star x y) ≠ star (2*x) (2*y) := by
  sorry

end NUMINAMATH_CALUDE_star_property_counterexample_l3985_398519


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l3985_398515

/-- A rectangular plot with length thrice its breadth and area 675 sq m has a breadth of 15 m -/
theorem rectangular_plot_breadth : 
  ∀ (length breadth : ℝ),
  length = 3 * breadth →
  length * breadth = 675 →
  breadth = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_rectangular_plot_breadth_l3985_398515


namespace NUMINAMATH_CALUDE_abs_ln_equal_implies_product_one_l3985_398586

theorem abs_ln_equal_implies_product_one (a b : ℝ) (h1 : a ≠ b) (h2 : |Real.log a| = |Real.log b|) : a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_ln_equal_implies_product_one_l3985_398586


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l3985_398594

/-- Given an arithmetic sequence where the first term is 5 and the common difference is 2,
    prove that the 15th term is equal to 33. -/
theorem fifteenth_term_of_sequence (a : ℕ → ℕ) :
  a 1 = 5 →
  (∀ n : ℕ, a (n + 1) = a n + 2) →
  a 15 = 33 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l3985_398594


namespace NUMINAMATH_CALUDE_frank_candy_count_l3985_398537

/-- Given a number of bags, pieces per bag, and leftover pieces, 
    calculates the total number of candy pieces. -/
def total_candy (bags : ℕ) (pieces_per_bag : ℕ) (leftover : ℕ) : ℕ :=
  bags * pieces_per_bag + leftover

/-- Proves that with 37 bags of 46 pieces each and 5 leftover pieces, 
    the total number of candy pieces is 1707. -/
theorem frank_candy_count : total_candy 37 46 5 = 1707 := by
  sorry

end NUMINAMATH_CALUDE_frank_candy_count_l3985_398537


namespace NUMINAMATH_CALUDE_circles_tangent_implies_a_value_l3985_398578

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 16

/-- Definition of circle C₂ -/
def C₂ (a x y : ℝ) : Prop := (x - a)^2 + y^2 = 1

/-- Two circles are tangent if the distance between their centers equals the sum or difference of their radii -/
def are_tangent (a : ℝ) : Prop := |a| = 5 ∨ |a| = 3

/-- Theorem stating that if C₁ and C₂ are tangent, then |a| = 5 or |a| = 3 -/
theorem circles_tangent_implies_a_value (a : ℝ) :
  (∃ x y : ℝ, C₁ x y ∧ C₂ a x y) → are_tangent a :=
sorry

end NUMINAMATH_CALUDE_circles_tangent_implies_a_value_l3985_398578


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l3985_398588

theorem smallest_common_multiple_of_6_and_15 :
  ∃ b : ℕ+, (∀ n : ℕ+, (6 ∣ n) → (15 ∣ n) → b ≤ n) ∧ (6 ∣ b) ∧ (15 ∣ b) ∧ b = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l3985_398588


namespace NUMINAMATH_CALUDE_base_8_to_7_conversion_l3985_398596

def base_8_to_10 (n : ℕ) : ℕ := 
  5 * 8^2 + 3 * 8^1 + 6 * 8^0

def base_10_to_7 (n : ℕ) : ℕ := 
  1 * 7^3 + 0 * 7^2 + 1 * 7^1 + 0 * 7^0

theorem base_8_to_7_conversion : 
  base_10_to_7 (base_8_to_10 536) = 1010 := by
  sorry

end NUMINAMATH_CALUDE_base_8_to_7_conversion_l3985_398596


namespace NUMINAMATH_CALUDE_unique_integer_proof_l3985_398563

theorem unique_integer_proof : ∃! n : ℕ+, 
  (24 ∣ n) ∧ 
  (8 < (n : ℝ) ^ (1/3)) ∧ 
  ((n : ℝ) ^ (1/3) < 8.2) ∧ 
  n = 528 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_proof_l3985_398563


namespace NUMINAMATH_CALUDE_unique_solution_system_l3985_398591

theorem unique_solution_system (x y z : ℝ) : 
  (x^2 + 25*y + 19*z = -471) ∧
  (y^2 + 23*x + 21*z = -397) ∧
  (z^2 + 21*x + 21*y = -545) ↔
  (x = -22 ∧ y = -23 ∧ z = -20) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3985_398591
