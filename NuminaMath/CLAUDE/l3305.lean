import Mathlib

namespace least_candies_to_remove_daniel_candy_problem_l3305_330543

theorem least_candies_to_remove (total_candies : Nat) (sisters : Nat) : Nat :=
  let remainder := total_candies % sisters
  if remainder = 0 then 0 else sisters - remainder

theorem daniel_candy_problem :
  least_candies_to_remove 25 4 = 1 := by
  sorry

end least_candies_to_remove_daniel_candy_problem_l3305_330543


namespace binomial_cube_example_l3305_330527

theorem binomial_cube_example : 4^3 + 3*(4^2)*2 + 3*4*(2^2) + 2^3 = 216 := by
  sorry

end binomial_cube_example_l3305_330527


namespace sum_inequality_l3305_330525

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + b^2) + b / Real.sqrt (b^2 + c^2) + c / Real.sqrt (c^2 + a^2)) ≤ 3 * Real.sqrt 2 / 2 :=
sorry

end sum_inequality_l3305_330525


namespace fourth_term_coefficient_implies_a_equals_one_l3305_330518

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

theorem fourth_term_coefficient_implies_a_equals_one (x a : ℝ) :
  (binomial 9 3 : ℝ) * a^3 = 84 → a = 1 := by sorry

end fourth_term_coefficient_implies_a_equals_one_l3305_330518


namespace ten_thousands_representation_l3305_330581

def ten_thousands : ℕ := 10000

def three_thousand_nine_hundred_seventy_six : ℕ := 3976

theorem ten_thousands_representation :
  three_thousand_nine_hundred_seventy_six * ten_thousands = 39760000 ∧
  three_thousand_nine_hundred_seventy_six = 3976 :=
by sorry

end ten_thousands_representation_l3305_330581


namespace M_characterization_a_range_l3305_330560

-- Define the set M
def M : Set ℝ := {m | ∃ x : ℝ, -1 < x ∧ x < 1 ∧ x^2 - x - m = 0}

-- Define the set N
def N (a : ℝ) : Set ℝ := {x | (x - a) * (x + a - 2) < 0}

-- Statement 1
theorem M_characterization : M = {m | -1/4 ≤ m ∧ m < 2} := by sorry

-- Statement 2
theorem a_range (h : ∀ m ∈ M, ∃ x ∈ N a, x^2 - x - m = 0) : 
  a ∈ Set.Iic (-1/4) ∪ Set.Ioi (9/4) := by sorry

end M_characterization_a_range_l3305_330560


namespace haleys_marbles_l3305_330538

/-- The number of boys in Haley's class who love to play marbles. -/
def num_boys : ℕ := 5

/-- The number of marbles each boy would receive. -/
def marbles_per_boy : ℕ := 7

/-- The total number of marbles Haley has. -/
def total_marbles : ℕ := num_boys * marbles_per_boy

/-- Theorem stating that the total number of marbles Haley has is equal to
    the product of the number of boys and the number of marbles each boy would receive. -/
theorem haleys_marbles : total_marbles = num_boys * marbles_per_boy := by
  sorry

end haleys_marbles_l3305_330538


namespace smallest_positive_integer_with_given_remainders_l3305_330599

theorem smallest_positive_integer_with_given_remainders : ∃ x : ℕ, 
  x > 0 ∧ 
  x % 6 = 3 ∧ 
  x % 8 = 5 ∧
  (∀ y : ℕ, y > 0 ∧ y % 6 = 3 ∧ y % 8 = 5 → x ≤ y) ∧
  x = 21 := by
  sorry

end smallest_positive_integer_with_given_remainders_l3305_330599


namespace sqrt_equation_implies_power_l3305_330507

theorem sqrt_equation_implies_power (x y : ℝ) : 
  Real.sqrt (2 - x) + Real.sqrt (x - 2) + y = 4 → x^y = 16 := by
  sorry

end sqrt_equation_implies_power_l3305_330507


namespace leo_balloon_distribution_l3305_330547

theorem leo_balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) 
  (h1 : total_balloons = 144) 
  (h2 : num_friends = 9) :
  total_balloons % num_friends = 0 := by
  sorry

end leo_balloon_distribution_l3305_330547


namespace gym_membership_cost_l3305_330513

/-- Calculates the total cost of gym memberships for the first year -/
theorem gym_membership_cost (cheap_monthly_fee : ℕ) (cheap_signup_fee : ℕ) 
  (expensive_monthly_multiplier : ℕ) (expensive_signup_months : ℕ) (months_per_year : ℕ) : 
  cheap_monthly_fee = 10 →
  cheap_signup_fee = 50 →
  expensive_monthly_multiplier = 3 →
  expensive_signup_months = 4 →
  months_per_year = 12 →
  (cheap_monthly_fee * months_per_year + cheap_signup_fee) + 
  (cheap_monthly_fee * expensive_monthly_multiplier * months_per_year + 
   cheap_monthly_fee * expensive_monthly_multiplier * expensive_signup_months) = 650 := by
  sorry

#check gym_membership_cost

end gym_membership_cost_l3305_330513


namespace intersection_of_A_and_B_l3305_330517

def A : Set ℝ := {x | |x - 1| < 2}

def B : Set ℝ := {y | ∃ x ∈ Set.Icc 0 2, y = 2^x}

theorem intersection_of_A_and_B : A ∩ B = Set.Ico 1 3 := by sorry

end intersection_of_A_and_B_l3305_330517


namespace badge_exchange_l3305_330595

theorem badge_exchange (x : ℕ) : 
  (x + 5 - (6 * (x + 5)) / 25 + x / 5 = x - x / 5 + (6 * (x + 5)) / 25 - 1) → 
  (x = 45 ∧ x + 5 = 50) := by
  sorry

end badge_exchange_l3305_330595


namespace balance_condition1_balance_condition2_triangular_weight_is_60_l3305_330572

/-- The weight of a rectangular weight in grams -/
def rectangular_weight : ℝ := 90

/-- The weight of a round weight in grams -/
def round_weight : ℝ := 30

/-- The weight of a triangular weight in grams -/
def triangular_weight : ℝ := 60

/-- First balance condition: 1 round + 1 triangular = 3 round -/
theorem balance_condition1 : round_weight + triangular_weight = 3 * round_weight := by sorry

/-- Second balance condition: 4 round + 1 triangular = 1 triangular + 1 round + 1 rectangular -/
theorem balance_condition2 : 4 * round_weight + triangular_weight = triangular_weight + round_weight + rectangular_weight := by sorry

/-- Proof that the triangular weight is 60 grams -/
theorem triangular_weight_is_60 : triangular_weight = 60 := by sorry

end balance_condition1_balance_condition2_triangular_weight_is_60_l3305_330572


namespace gold_bar_ratio_l3305_330535

theorem gold_bar_ratio (initial_bars : ℕ) (tax_percent : ℚ) (final_bars : ℕ) : 
  initial_bars = 60 →
  tax_percent = 1/10 →
  final_bars = 27 →
  (initial_bars - initial_bars * tax_percent - final_bars) / (initial_bars - initial_bars * tax_percent) = 1/2 := by
sorry

end gold_bar_ratio_l3305_330535


namespace min_value_reciprocal_sum_l3305_330530

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  1/a + 1/b ≥ 3 + 2*Real.sqrt 2 := by
sorry

end min_value_reciprocal_sum_l3305_330530


namespace train_platform_crossing_time_l3305_330531

/-- Given a train of length 1200 m that crosses a tree in 120 sec,
    prove that it takes 190 sec to pass a platform of length 700 m. -/
theorem train_platform_crossing_time :
  let train_length : ℝ := 1200
  let tree_crossing_time : ℝ := 120
  let platform_length : ℝ := 700
  let train_speed : ℝ := train_length / tree_crossing_time
  let total_distance : ℝ := train_length + platform_length
  let platform_crossing_time : ℝ := total_distance / train_speed
  platform_crossing_time = 190 := by
  sorry

end train_platform_crossing_time_l3305_330531


namespace solution_set_is_open_interval_l3305_330590

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - 2 * |x - 1|

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | f x > 1}

-- Theorem statement
theorem solution_set_is_open_interval :
  solution_set = Set.Ioo (2/3) 2 :=
sorry

end solution_set_is_open_interval_l3305_330590


namespace yellow_highlighters_count_l3305_330575

/-- The number of yellow highlighters in Kaya's teacher's desk -/
def yellow_highlighters (total pink blue : ℕ) : ℕ := total - pink - blue

/-- Theorem stating the number of yellow highlighters -/
theorem yellow_highlighters_count :
  yellow_highlighters 22 9 5 = 8 := by
  sorry

end yellow_highlighters_count_l3305_330575


namespace sqrt45_same_type_as_sqrt5_l3305_330505

-- Define the property of being "of the same type as √5"
def same_type_as_sqrt5 (x : ℝ) : Prop :=
  ∃ (k : ℝ), x = k * Real.sqrt 5

-- State the theorem
theorem sqrt45_same_type_as_sqrt5 :
  same_type_as_sqrt5 (Real.sqrt 45) :=
by
  -- The proof goes here
  sorry

end sqrt45_same_type_as_sqrt5_l3305_330505


namespace sum_of_solutions_quadratic_l3305_330578

theorem sum_of_solutions_quadratic (a b c d e : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  let g : ℝ → ℝ := λ x => d * x + e
  (∀ x, f x = g x) →
  (-(b - d) / (2 * a)) = 7 :=
by
  sorry

end sum_of_solutions_quadratic_l3305_330578


namespace largest_n_for_inequality_l3305_330591

theorem largest_n_for_inequality : ∃ (n : ℕ), n = 24 ∧ 
  (∀ (a b c d : ℝ), 
    (↑n + 2) * Real.sqrt (a^2 + b^2) + 
    (↑n + 1) * Real.sqrt (a^2 + c^2) + 
    (↑n + 1) * Real.sqrt (a^2 + d^2) ≥ 
    ↑n * (a + b + c + d)) ∧
  (∀ (m : ℕ), m > n → 
    ∃ (a b c d : ℝ), 
      (↑m + 2) * Real.sqrt (a^2 + b^2) + 
      (↑m + 1) * Real.sqrt (a^2 + c^2) + 
      (↑m + 1) * Real.sqrt (a^2 + d^2) < 
      ↑m * (a + b + c + d)) :=
by sorry


end largest_n_for_inequality_l3305_330591


namespace area_inequality_l3305_330549

/-- A point with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- A convex quadrilateral with integer vertices -/
structure ConvexQuadrilateral where
  A : IntPoint
  B : IntPoint
  C : IntPoint
  D : IntPoint
  convex : Bool  -- Assume this is true for a convex quadrilateral

/-- The intersection point of the diagonals -/
def diagonalIntersection (q : ConvexQuadrilateral) : IntPoint :=
  sorry  -- Definition of diagonal intersection

/-- The area of a shape -/
class HasArea (α : Type) where
  area : α → ℝ

instance : HasArea ConvexQuadrilateral where
  area := sorry  -- Definition of quadrilateral area

instance : HasArea (IntPoint × IntPoint × IntPoint) where
  area := sorry  -- Definition of triangle area

theorem area_inequality (q : ConvexQuadrilateral) :
  let S := diagonalIntersection q
  let P := HasArea.area q
  let P₁ := HasArea.area (q.A, q.B, S)
  Real.sqrt P ≥ Real.sqrt P₁ + Real.sqrt 2 / 2 := by
  sorry

end area_inequality_l3305_330549


namespace max_tiles_on_floor_l3305_330536

theorem max_tiles_on_floor (floor_length floor_width tile_length tile_width : ℕ) 
  (h1 : floor_length = 560)
  (h2 : floor_width = 240)
  (h3 : tile_length = 60)
  (h4 : tile_width = 56) : 
  (floor_length / tile_length) * (floor_width / tile_width) ≤ 40 ∧ 
  (floor_length / tile_width) * (floor_width / tile_length) ≤ 40 ∧
  ((floor_length / tile_length) * (floor_width / tile_width) = 40 ∨
   (floor_length / tile_width) * (floor_width / tile_length) = 40) :=
by sorry

end max_tiles_on_floor_l3305_330536


namespace blue_pigment_percentage_l3305_330567

/-- Represents the composition of the brown paint mixture --/
structure BrownPaint where
  total_weight : ℝ
  blue_percentage : ℝ
  red_weight : ℝ

/-- Represents the composition of the dark blue paint --/
structure DarkBluePaint where
  blue_percentage : ℝ
  red_percentage : ℝ

/-- Represents the composition of the green paint --/
structure GreenPaint where
  blue_percentage : ℝ
  yellow_percentage : ℝ

/-- Theorem stating the percentage of blue pigment in dark blue and green paints --/
theorem blue_pigment_percentage
  (brown : BrownPaint)
  (dark_blue : DarkBluePaint)
  (green : GreenPaint)
  (h1 : brown.total_weight = 10)
  (h2 : brown.blue_percentage = 0.4)
  (h3 : brown.red_weight = 3)
  (h4 : dark_blue.red_percentage = 0.6)
  (h5 : green.yellow_percentage = 0.6)
  (h6 : dark_blue.blue_percentage = green.blue_percentage) :
  dark_blue.blue_percentage = 0.2 :=
sorry


end blue_pigment_percentage_l3305_330567


namespace expression_simplification_l3305_330555

theorem expression_simplification 
  (x y z w : ℝ) 
  (hx : x ≠ 2) 
  (hy : y ≠ 3) 
  (hz : z ≠ 4) 
  (hw : w ≠ 5) 
  (h1 : y ≠ 6) 
  (h2 : w ≠ 4) 
  (h3 : z ≠ 6) :
  (x - 2) / (6 - y) * (y - 3) / (2 - x) * (z - 4) / (3 - y) * 
  (6 - z) / (4 - w) * (w - 5) / (z - 6) * (x + 1) / (5 - w) = 1 := by
  sorry

#check expression_simplification

end expression_simplification_l3305_330555


namespace polynomial_divisibility_l3305_330556

theorem polynomial_divisibility : ∃ z : Polynomial ℤ, 
  X^44 + X^33 + X^22 + X^11 + 1 = (X^4 + X^3 + X^2 + X + 1) * z :=
sorry

end polynomial_divisibility_l3305_330556


namespace sin_negative_2055_degrees_l3305_330577

theorem sin_negative_2055_degrees : 
  Real.sin ((-2055 : ℝ) * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end sin_negative_2055_degrees_l3305_330577


namespace range_of_a_l3305_330561

def A (a : ℝ) : Set ℝ := {x | (a * x - 1) * (a - x) > 0}

theorem range_of_a :
  ∀ a : ℝ, (2 ∈ A a ∧ 3 ∉ A a) ↔ a ∈ (Set.Ioo 2 3 ∪ Set.Ico (1/3) (1/2)) :=
by sorry

end range_of_a_l3305_330561


namespace analogical_reasoning_example_l3305_330594

/-- Represents different types of reasoning -/
inductive ReasoningType
  | Deductive
  | Inductive
  | Analogical
  | Other

/-- Determines the type of reasoning for a given statement -/
def determineReasoningType (statement : String) : ReasoningType :=
  match statement with
  | "Inferring the properties of a spatial quadrilateral from the properties of a plane triangle" => ReasoningType.Analogical
  | _ => ReasoningType.Other

/-- Theorem stating that the given statement is an example of analogical reasoning -/
theorem analogical_reasoning_example :
  determineReasoningType "Inferring the properties of a spatial quadrilateral from the properties of a plane triangle" = ReasoningType.Analogical := by
  sorry


end analogical_reasoning_example_l3305_330594


namespace log_xy_value_l3305_330566

theorem log_xy_value (x y : ℝ) (h1 : Real.log (x * y^5) = 2) (h2 : Real.log (x^3 * y) = 2) :
  Real.log (x * y) = 6/7 := by sorry

end log_xy_value_l3305_330566


namespace original_number_proof_l3305_330545

theorem original_number_proof (x : ℚ) : (1 / x) - 2 = 5 / 2 → x = 2 / 9 := by
  sorry

end original_number_proof_l3305_330545


namespace smallest_n_square_and_cube_l3305_330541

theorem smallest_n_square_and_cube : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (m : ℕ), 3 * n = m^3) ∧ 
  (∀ (x : ℕ), x > 0 ∧ (∃ (y : ℕ), 4 * x = y^2) ∧ (∃ (z : ℕ), 3 * x = z^3) → x ≥ n) ∧
  n = 18 :=
sorry

end smallest_n_square_and_cube_l3305_330541


namespace rectangle_dimensions_l3305_330598

/-- Proves that a rectangle with perimeter 150 cm and length 15 cm greater than width has width 30 cm and length 45 cm -/
theorem rectangle_dimensions (w l : ℝ) 
  (h_perimeter : 2 * w + 2 * l = 150)
  (h_length_width : l = w + 15) :
  w = 30 ∧ l = 45 := by
  sorry

end rectangle_dimensions_l3305_330598


namespace intersection_distance_squared_l3305_330520

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the squared distance between two points in 2D space -/
def squaredDistance (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Theorem: The squared distance between intersection points of two specific circles -/
theorem intersection_distance_squared (c1 c2 : Circle)
  (h1 : c1 = ⟨(1, -2), 5⟩)
  (h2 : c2 = ⟨(1, 4), 3⟩) :
  ∃ (p1 p2 : ℝ × ℝ),
    squaredDistance p1 c1.center = c1.radius^2 ∧
    squaredDistance p1 c2.center = c2.radius^2 ∧
    squaredDistance p2 c1.center = c1.radius^2 ∧
    squaredDistance p2 c2.center = c2.radius^2 ∧
    squaredDistance p1 p2 = 224/9 := by
  sorry

end intersection_distance_squared_l3305_330520


namespace yellow_leaves_count_l3305_330551

theorem yellow_leaves_count (thursday_leaves friday_leaves saturday_leaves : ℕ)
  (thursday_brown_percent thursday_green_percent : ℚ)
  (friday_brown_percent friday_green_percent : ℚ)
  (saturday_brown_percent saturday_green_percent : ℚ)
  (h1 : thursday_leaves = 15)
  (h2 : friday_leaves = 22)
  (h3 : saturday_leaves = 30)
  (h4 : thursday_brown_percent = 25/100)
  (h5 : thursday_green_percent = 40/100)
  (h6 : friday_brown_percent = 30/100)
  (h7 : friday_green_percent = 20/100)
  (h8 : saturday_brown_percent = 15/100)
  (h9 : saturday_green_percent = 50/100) :
  ⌊thursday_leaves * (1 - thursday_brown_percent - thursday_green_percent)⌋ +
  ⌊friday_leaves * (1 - friday_brown_percent - friday_green_percent)⌋ +
  ⌊saturday_leaves * (1 - saturday_brown_percent - saturday_green_percent)⌋ = 26 := by
sorry

end yellow_leaves_count_l3305_330551


namespace sum_of_squared_sums_of_roots_l3305_330515

theorem sum_of_squared_sums_of_roots (p q r : ℝ) : 
  (p^3 - 15*p^2 + 25*p - 10 = 0) → 
  (q^3 - 15*q^2 + 25*q - 10 = 0) → 
  (r^3 - 15*r^2 + 25*r - 10 = 0) → 
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 := by
sorry

end sum_of_squared_sums_of_roots_l3305_330515


namespace coupon_redemption_schedule_l3305_330503

-- Define the days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday    => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday   => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday  => DayOfWeek.Friday
  | DayOfWeek.Friday    => DayOfWeek.Saturday
  | DayOfWeek.Saturday  => DayOfWeek.Sunday
  | DayOfWeek.Sunday    => DayOfWeek.Monday

def advance_days (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advance_days (next_day d) n

def is_saturday (d : DayOfWeek) : Prop :=
  d = DayOfWeek.Saturday

theorem coupon_redemption_schedule :
  let start_day := DayOfWeek.Monday
  let days_between_redemptions := 15
  let num_coupons := 7
  ∀ i, i < num_coupons →
    ¬(is_saturday (advance_days start_day (i * days_between_redemptions))) :=
by sorry

end coupon_redemption_schedule_l3305_330503


namespace triangle_circle_radii_relation_l3305_330508

/-- Given a triangle with sides of consecutive natural numbers, 
    the radius of its circumcircle (R) and the radius of its incircle (r) 
    satisfy the equation R = 2r + 1/(2r) -/
theorem triangle_circle_radii_relation (n : ℕ) (R r : ℝ) 
    (h1 : n > 1) 
    (h2 : R = (n^2 - 1) / (6 * r)) 
    (h3 : r^2 = (n^2 - 4) / 12) : 
  R = 2*r + 1/(2*r) := by
  sorry

end triangle_circle_radii_relation_l3305_330508


namespace town_population_problem_l3305_330563

theorem town_population_problem (original_population : ℕ) : 
  (((original_population + 1500) * 85 / 100) : ℕ) = original_population - 45 →
  original_population = 8800 := by
sorry

end town_population_problem_l3305_330563


namespace paths_in_7x7_grid_l3305_330585

/-- The number of paths in a square grid from bottom left to top right -/
def num_paths (n : ℕ) : ℕ := Nat.choose (2 * n) n

/-- The theorem stating that the number of paths in a 7x7 grid is 3432 -/
theorem paths_in_7x7_grid : num_paths 7 = 3432 := by
  sorry

end paths_in_7x7_grid_l3305_330585


namespace school_sections_l3305_330501

theorem school_sections (boys girls : ℕ) (h1 : boys = 408) (h2 : girls = 312) :
  let section_size := Nat.gcd boys girls
  let boy_sections := boys / section_size
  let girl_sections := girls / section_size
  boy_sections + girl_sections = 30 := by
sorry

end school_sections_l3305_330501


namespace perpendicular_tangents_intersection_y_coord_l3305_330533

/-- The y-coordinate of the intersection point of perpendicular tangents on a parabola -/
theorem perpendicular_tangents_intersection_y_coord 
  (a b : ℝ) 
  (h_parabola : ∀ x : ℝ, (x = a ∨ x = b) → 4 * x^2 = (4 * x^2))
  (h_perpendicular : (8 * a) * (8 * b) = -1) :
  ∃ P : ℝ × ℝ, 
    (P.1 = (a + b) / 2) ∧ 
    (P.2 = 4 * a * b) ∧ 
    (P.2 = -1/8) := by
  sorry

end perpendicular_tangents_intersection_y_coord_l3305_330533


namespace point_distance_and_reflection_l3305_330532

/-- Point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Distance from a point to the y-axis -/
def distanceToYAxis (p : Point2D) : ℝ := |p.x|

/-- Reflection of a point across the x-axis -/
def reflectAcrossXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

theorem point_distance_and_reflection :
  let P : Point2D := { x := 4, y := -2 }
  (distanceToYAxis P = 4) ∧
  (reflectAcrossXAxis P = { x := 4, y := 2 }) := by
  sorry

end point_distance_and_reflection_l3305_330532


namespace secret_spread_l3305_330548

/-- Represents the number of people each person tells the secret to on a given day -/
def tell_count (day : Nat) : Nat :=
  match day with
  | 1 => 1  -- Monday: Jessica tells 1 friend
  | 2 => 2  -- Tuesday
  | 3 => 2  -- Wednesday
  | 4 => 1  -- Thursday
  | _ => 2  -- Friday to Monday

/-- Calculates the total number of people knowing the secret after a given number of days -/
def total_knowing (days : Nat) : Nat :=
  match days with
  | 0 => 1  -- Only Jessica knows on day 0
  | n + 1 => total_knowing n + (total_knowing n - total_knowing (n - 1)) * tell_count (n + 1)

/-- The theorem stating that after 8 days, 132 people will know the secret -/
theorem secret_spread : total_knowing 8 = 132 := by
  sorry


end secret_spread_l3305_330548


namespace inverse_function_range_l3305_330557

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a

-- State the theorem
theorem inverse_function_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 3, Function.Injective (fun x => f a x)) ∧ 
  (|a - 1| + |a - 3| ≤ 4) →
  a ∈ Set.Icc 0 1 ∪ Set.Icc 3 4 :=
sorry

end inverse_function_range_l3305_330557


namespace f_bijection_l3305_330529

def f (n : ℤ) : ℤ := 2 * n

theorem f_bijection : Function.Bijective f := by sorry

end f_bijection_l3305_330529


namespace satellite_has_24_units_l3305_330584

/-- Represents a satellite with modular units and sensors. -/
structure Satellite where
  units : ℕ
  non_upgraded_per_unit : ℕ
  total_upgraded : ℕ

/-- The conditions of the satellite problem. -/
def satellite_conditions (s : Satellite) : Prop :=
  -- Condition 2: non-upgraded sensors per unit is 1/6 of total upgraded
  s.non_upgraded_per_unit = s.total_upgraded / 6 ∧
  -- Condition 3: 20% of all sensors are upgraded
  s.total_upgraded = (s.total_upgraded + s.units * s.non_upgraded_per_unit) / 5

/-- The theorem stating that a satellite satisfying the given conditions has 24 units. -/
theorem satellite_has_24_units (s : Satellite) (h : satellite_conditions s) : s.units = 24 := by
  sorry


end satellite_has_24_units_l3305_330584


namespace card_stack_problem_l3305_330512

theorem card_stack_problem (n : ℕ) : 
  let total_cards := 2 * n
  let pile_A := n
  let pile_B := n
  let card_80_position := 80
  (card_80_position ≤ pile_A) →
  (card_80_position % 2 = 1) →
  (∃ (new_position : ℕ), new_position = card_80_position ∧ 
    new_position = pile_B + (card_80_position + 1) / 2) →
  total_cards = 240 := by
sorry

end card_stack_problem_l3305_330512


namespace range_of_m_l3305_330550

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (|1 - (x-1)/2| ≤ 3 → x^2 - 2*x + 1 - m^2 ≤ 0) ∧ 
  (∃ x : ℝ, x^2 - 2*x + 1 - m^2 ≤ 0 ∧ |1 - (x-1)/2| > 3)) ∧ 
  m > 0 → 
  m ≥ 8 :=
by sorry

end range_of_m_l3305_330550


namespace count_perfect_squares_l3305_330510

theorem count_perfect_squares (n : ℕ) : 
  (Finset.filter (fun k => 16 * k * k < 5000) (Finset.range n)).card = 17 :=
sorry

end count_perfect_squares_l3305_330510


namespace hotel_occupancy_and_profit_l3305_330534

/-- Represents a hotel with its pricing and occupancy characteristics -/
structure Hotel where
  totalRooms : ℕ
  originalPrice : ℕ
  fullBookingPrice : ℕ
  costPerRoom : ℕ
  vacancyRate : ℚ
  maxPriceMultiplier : ℚ

/-- Calculates the number of occupied rooms given a price increase -/
def occupiedRooms (h : Hotel) (priceIncrease : ℕ) : ℚ :=
  h.totalRooms - priceIncrease * h.vacancyRate

/-- Calculates the profit given a price increase -/
def profit (h : Hotel) (priceIncrease : ℕ) : ℚ :=
  (h.fullBookingPrice + priceIncrease - h.costPerRoom) * occupiedRooms h priceIncrease

/-- The hotel in the problem -/
def problemHotel : Hotel := {
  totalRooms := 50
  originalPrice := 190
  fullBookingPrice := 180
  costPerRoom := 20
  vacancyRate := 1/10
  maxPriceMultiplier := 3/2
}

theorem hotel_occupancy_and_profit :
  (occupiedRooms problemHotel 50 = 45) ∧
  (profit problemHotel 50 = 9450) := by sorry

end hotel_occupancy_and_profit_l3305_330534


namespace li_point_parabola_range_l3305_330582

/-- A point (x, y) is a "Li point" if x and y have opposite signs -/
def is_li_point (x y : ℝ) : Prop := x * y < 0

/-- Parabola equation -/
def parabola (a c x : ℝ) : ℝ := a * x^2 - 7 * x + c

theorem li_point_parabola_range (a c : ℝ) :
  a > 1 →
  (∃! x : ℝ, is_li_point x (parabola a c x)) →
  0 < c ∧ c < 9 :=
by sorry

end li_point_parabola_range_l3305_330582


namespace equation_solutions_l3305_330509

theorem equation_solutions :
  (∃ x : ℝ, 8 * (x + 1)^3 = 64 ∧ x = 1) ∧
  (∃ x y : ℝ, (x + 1)^2 = 100 ∧ (y + 1)^2 = 100 ∧ x = 9 ∧ y = -11) := by
  sorry

end equation_solutions_l3305_330509


namespace fixed_point_of_exponential_function_l3305_330528

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 3 + a^(x - 1)
  f 1 = 4 := by
sorry

end fixed_point_of_exponential_function_l3305_330528


namespace files_deleted_l3305_330564

theorem files_deleted (initial_files initial_apps final_files final_apps : ℕ) 
  (h1 : initial_files = 24)
  (h2 : initial_apps = 13)
  (h3 : final_files = 21)
  (h4 : final_apps = 17) :
  initial_files - final_files = 3 := by
  sorry

end files_deleted_l3305_330564


namespace sum_six_consecutive_integers_l3305_330553

theorem sum_six_consecutive_integers (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 6 * n + 15 := by
  sorry

end sum_six_consecutive_integers_l3305_330553


namespace josanna_minimum_score_l3305_330559

def josanna_scores : List ℕ := [82, 76, 91, 65, 87, 78]

def current_average : ℚ := (josanna_scores.sum : ℚ) / josanna_scores.length

def target_average : ℚ := current_average + 5

def minimum_score : ℕ := 116

theorem josanna_minimum_score :
  ∀ (new_score : ℕ),
    ((josanna_scores.sum + new_score : ℚ) / (josanna_scores.length + 1) ≥ target_average) →
    (new_score ≥ minimum_score) := by sorry

end josanna_minimum_score_l3305_330559


namespace ellipse_eccentricity_l3305_330576

theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := c / a
  (a^2 * b^2 = (a^2 - c^2) * b^2) →  -- Ellipse equation
  (c^2 + b^2 = a^2) →               -- Right triangle condition
  e = (Real.sqrt 5 - 1) / 2 := by
sorry

end ellipse_eccentricity_l3305_330576


namespace bijective_function_property_l3305_330539

variable {V : Type*} [Fintype V]
variable (f g : V → V)
variable (S T : Set V)

def is_bijective (h : V → V) : Prop :=
  Function.Injective h ∧ Function.Surjective h

theorem bijective_function_property
  (hf : is_bijective f)
  (hg : is_bijective g)
  (hS : S = {w : V | f (f w) = g (g w)})
  (hT : T = {w : V | f (g w) = g (f w)})
  (hST : S ∪ T = Set.univ) :
  ∀ w : V, f w ∈ S ↔ g w ∈ S :=
by sorry

end bijective_function_property_l3305_330539


namespace rectangle_lcm_gcd_product_l3305_330544

theorem rectangle_lcm_gcd_product : 
  let a : ℕ := 24
  let b : ℕ := 36
  Nat.lcm a b * Nat.gcd a b = 864 := by
  sorry

end rectangle_lcm_gcd_product_l3305_330544


namespace sin_seventeen_pi_sixths_l3305_330597

theorem sin_seventeen_pi_sixths : Real.sin (17 * π / 6) = 1 / 2 := by
  sorry

end sin_seventeen_pi_sixths_l3305_330597


namespace brick_width_proof_l3305_330580

/-- The width of a brick that satisfies the given conditions --/
def brick_width : ℝ := 11.25

theorem brick_width_proof (wall_volume : ℝ) (brick_length : ℝ) (brick_height : ℝ) (num_bricks : ℕ) 
  (h1 : wall_volume = 800 * 600 * 22.5)
  (h2 : ∀ w, brick_length * w * brick_height * num_bricks = wall_volume)
  (h3 : brick_length = 50)
  (h4 : brick_height = 6)
  (h5 : num_bricks = 3200) :
  brick_width = 11.25 := by
sorry

end brick_width_proof_l3305_330580


namespace cost_of_potting_soil_l3305_330554

def cost_of_seeds : ℝ := 2.00
def number_of_plants : ℕ := 20
def price_per_plant : ℝ := 5.00
def net_profit : ℝ := 90.00

theorem cost_of_potting_soil :
  ∃ (cost : ℝ), cost = (number_of_plants : ℝ) * price_per_plant - cost_of_seeds - net_profit :=
by sorry

end cost_of_potting_soil_l3305_330554


namespace quadratic_not_always_positive_l3305_330500

theorem quadratic_not_always_positive : ¬ (∀ x : ℝ, x^2 + 3*x + 1 > 0) := by
  sorry

end quadratic_not_always_positive_l3305_330500


namespace total_pages_calculation_l3305_330573

/-- The number of pages in each booklet -/
def pages_per_booklet : ℕ := 9

/-- The number of booklets in the short story section -/
def number_of_booklets : ℕ := 49

/-- The total number of pages in all booklets -/
def total_pages : ℕ := pages_per_booklet * number_of_booklets

theorem total_pages_calculation :
  total_pages = 441 :=
by sorry

end total_pages_calculation_l3305_330573


namespace cucumber_count_l3305_330522

theorem cucumber_count (total : ℕ) (ratio : ℕ) (h1 : total = 420) (h2 : ratio = 4) :
  ∃ (cucumbers : ℕ), cucumbers * (ratio + 1) = total ∧ cucumbers = 84 := by
  sorry

end cucumber_count_l3305_330522


namespace triangle_properties_l3305_330583

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * Real.sin t.A * Real.cos t.B = 2 * Real.sin t.C - Real.sin t.B)
  (h2 : t.a = 4 * Real.sqrt 3)
  (h3 : t.b + t.c = 8) :
  t.A = π / 3 ∧ (1/2 * t.b * t.c * Real.sin t.A = 4 * Real.sqrt 3 / 3) := by
  sorry

end triangle_properties_l3305_330583


namespace max_value_of_sum_of_squares_l3305_330502

theorem max_value_of_sum_of_squares (x y : ℝ) (h : 3 * x^2 + 2 * y^2 = 6 * x) :
  ∃ (max : ℝ), max = 1 ∧ ∀ (a b : ℝ), 3 * a^2 + 2 * b^2 = 6 * a → a^2 + b^2 ≤ max :=
by sorry

end max_value_of_sum_of_squares_l3305_330502


namespace binomial_expansion_largest_coeff_l3305_330558

theorem binomial_expansion_largest_coeff (n : ℕ+) :
  (∀ k : ℕ, k ≠ 5 → Nat.choose n 5 ≥ Nat.choose n k) → n = 10 := by
  sorry

end binomial_expansion_largest_coeff_l3305_330558


namespace pond_fish_problem_l3305_330574

/-- Represents the number of fish in a pond -/
def total_fish : ℕ := 500

/-- Represents the number of fish initially tagged -/
def tagged_fish : ℕ := 50

/-- Represents the number of fish caught in the second catch -/
def second_catch : ℕ := 50

/-- Represents the number of tagged fish found in the second catch -/
def tagged_in_second_catch : ℕ := 5

theorem pond_fish_problem :
  (tagged_in_second_catch : ℚ) / second_catch = tagged_fish / total_fish :=
sorry

end pond_fish_problem_l3305_330574


namespace projectile_meeting_distance_l3305_330571

theorem projectile_meeting_distance
  (speed1 : ℝ)
  (speed2 : ℝ)
  (meeting_time_minutes : ℝ)
  (h1 : speed1 = 444)
  (h2 : speed2 = 555)
  (h3 : meeting_time_minutes = 120) :
  speed1 * (meeting_time_minutes / 60) + speed2 * (meeting_time_minutes / 60) = 1998 :=
by
  sorry

end projectile_meeting_distance_l3305_330571


namespace lucia_outfits_l3305_330569

/-- Represents the number of different outfits Lucia can create -/
def outfits (shoes dresses hats : ℕ) : ℕ := shoes * dresses * hats

/-- Proves that Lucia can create 60 different outfits -/
theorem lucia_outfits :
  outfits 3 5 4 = 60 := by
  sorry

end lucia_outfits_l3305_330569


namespace song_listens_theorem_l3305_330506

def calculate_total_listens (initial_listens : ℕ) (months : ℕ) : ℕ :=
  let doubling_factor := 2 ^ months
  initial_listens * (doubling_factor - 1) + initial_listens

theorem song_listens_theorem (initial_listens : ℕ) (months : ℕ) 
  (h1 : initial_listens = 60000) (h2 : months = 3) :
  calculate_total_listens initial_listens months = 900000 := by
  sorry

#eval calculate_total_listens 60000 3

end song_listens_theorem_l3305_330506


namespace rectangles_in_5x4_grid_l3305_330540

/-- The number of rectangles in a row of length n -/
def rectangles_in_row (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of rectangles in a grid of width w and height h -/
def rectangles_in_grid (w h : ℕ) : ℕ :=
  w * rectangles_in_row h + h * rectangles_in_row w - w * h

theorem rectangles_in_5x4_grid :
  rectangles_in_grid 5 4 = 24 := by
  sorry

end rectangles_in_5x4_grid_l3305_330540


namespace inverse_of_A_squared_l3305_330586

theorem inverse_of_A_squared (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A⁻¹ = ![![3, -2], ![1, 1]]) : 
  (A^2)⁻¹ = ![![7, -8], ![4, -1]] := by
  sorry

end inverse_of_A_squared_l3305_330586


namespace inequality_proof_l3305_330592

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  a^3 + b^3 + c^3 + (a*b)/(a^2 + b^2) + (b*c)/(b^2 + c^2) + (c*a)/(c^2 + a^2) ≥ 9/2 := by
  sorry

end inequality_proof_l3305_330592


namespace least_possible_b_l3305_330579

-- Define Fibonacci sequence
def isFibonacci (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (((1 + Real.sqrt 5) / 2) ^ k - ((1 - Real.sqrt 5) / 2) ^ k) / Real.sqrt 5

-- Define the problem
theorem least_possible_b (a b : ℕ) : 
  (a + b = 90) →  -- Sum of acute angles in a right triangle
  (a > b) →       -- a is greater than b
  isFibonacci a → -- a is a Fibonacci number
  isFibonacci b → -- b is a Fibonacci number
  (∀ c : ℕ, c < b → (c + a ≠ 90 ∨ ¬isFibonacci c ∨ ¬isFibonacci a)) →
  b = 1 :=
by sorry

end least_possible_b_l3305_330579


namespace largest_angle_in_triangle_l3305_330589

theorem largest_angle_in_triangle (x y z : ℝ) : 
  x = 30 ∧ y = 45 ∧ x + y + z = 180 → max x (max y z) = 105 :=
by sorry

end largest_angle_in_triangle_l3305_330589


namespace probability_knows_cpp_l3305_330596

/-- Given a software company with the following characteristics:
  * 600 total employees
  * 3/10 of employees know C++
  * 5/12 of employees know Java
  * 4/15 of employees know Python
  * 2/25 of employees know neither C++, Java, nor Python
  Prove that the probability of a randomly selected employee knowing C++ is 3/10. -/
theorem probability_knows_cpp (total_employees : ℕ) 
  (fraction_cpp : ℚ) (fraction_java : ℚ) (fraction_python : ℚ) (fraction_none : ℚ)
  (h1 : total_employees = 600)
  (h2 : fraction_cpp = 3 / 10)
  (h3 : fraction_java = 5 / 12)
  (h4 : fraction_python = 4 / 15)
  (h5 : fraction_none = 2 / 25) :
  (↑total_employees * fraction_cpp : ℚ) / total_employees = 3 / 10 := by
  sorry

end probability_knows_cpp_l3305_330596


namespace cost_price_per_meter_is_58_l3305_330504

/-- Calculates the cost price per meter of cloth given the total length,
    total selling price, and profit per meter. -/
def costPricePerMeter (totalLength : ℕ) (totalSellingPrice : ℕ) (profitPerMeter : ℕ) : ℕ :=
  (totalSellingPrice - totalLength * profitPerMeter) / totalLength

/-- Proves that the cost price per meter of cloth is 58 rupees given the
    specified conditions. -/
theorem cost_price_per_meter_is_58 :
  costPricePerMeter 78 6788 29 = 58 := by sorry

end cost_price_per_meter_is_58_l3305_330504


namespace total_pears_picked_l3305_330565

def sara_pears : ℕ := 6
def tim_pears : ℕ := 5

theorem total_pears_picked : sara_pears + tim_pears = 11 := by
  sorry

end total_pears_picked_l3305_330565


namespace digit_counting_theorem_l3305_330523

/-- The set of available digits -/
def availableDigits : Finset ℕ := {0, 1, 2, 3, 5, 9}

/-- Count of four-digit numbers -/
def countFourDigit : ℕ := 300

/-- Count of four-digit odd numbers -/
def countFourDigitOdd : ℕ := 192

/-- Count of four-digit even numbers -/
def countFourDigitEven : ℕ := 108

/-- Total count of natural numbers -/
def countNaturalNumbers : ℕ := 1631

/-- Main theorem stating the counting results -/
theorem digit_counting_theorem :
  (∀ d ∈ availableDigits, d < 10) ∧
  (countFourDigit = 300) ∧
  (countFourDigitOdd = 192) ∧
  (countFourDigitEven = 108) ∧
  (countNaturalNumbers = 1631) := by
  sorry

end digit_counting_theorem_l3305_330523


namespace remainder_theorem_l3305_330524

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 60 * k - 1) :
  (n^2 + 2*n + 3) % 60 = 2 := by sorry

end remainder_theorem_l3305_330524


namespace polynomial_factors_imply_relation_l3305_330511

theorem polynomial_factors_imply_relation (h k : ℝ) : 
  (∃ a : ℝ, 2 * x^3 - h * x + k = (x + 2) * (x - 1) * a) → 
  2 * h - 3 * k = 0 := by
  sorry

end polynomial_factors_imply_relation_l3305_330511


namespace power_division_result_l3305_330562

theorem power_division_result : 3^12 / 27^2 = 729 := by
  -- Define 27 as 3^3
  have h1 : 27 = 3^3 := by sorry
  
  -- Rewrite the division using the definition of 27
  have h2 : 3^12 / 27^2 = 3^12 / (3^3)^2 := by sorry
  
  -- Simplify the exponents
  have h3 : 3^12 / (3^3)^2 = 3^(12 - 3*2) := by sorry
  
  -- Evaluate the final result
  have h4 : 3^(12 - 3*2) = 3^6 := by sorry
  have h5 : 3^6 = 729 := by sorry
  
  -- Combine all steps
  sorry

#check power_division_result

end power_division_result_l3305_330562


namespace accurate_estimation_l3305_330546

/-- Represents a scale reading on a measuring device --/
structure ScaleReading where
  min : Float
  max : Float
  reading : Float
  min_le_reading : min ≤ reading
  reading_le_max : reading ≤ max

/-- The most accurate estimation for a scale reading --/
def mostAccurateEstimation (s : ScaleReading) : Float :=
  15.9

/-- Theorem stating that 15.9 is the most accurate estimation for the given scale reading --/
theorem accurate_estimation (s : ScaleReading) 
  (h1 : s.min = 15.75) 
  (h2 : s.max = 16.0) : 
  mostAccurateEstimation s = 15.9 := by
  sorry

end accurate_estimation_l3305_330546


namespace sequence_property_l3305_330588

theorem sequence_property (a : ℕ → ℕ) (p : ℕ) : 
  (∀ (m n : ℕ), m ≥ n → a (m + n) + a (m - n) + 2 * m - 2 * n - 1 = (a (2 * m) + a (2 * n)) / 2) →
  a 1 = 0 →
  a p = 2019 * 2019 →
  p = 2020 := by
  sorry

end sequence_property_l3305_330588


namespace right_triangle_hypotenuse_l3305_330537

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right-angled triangle condition
  b = 2 * a →        -- one leg is twice the other
  a^2 + b^2 + c^2 = 1450 →  -- sum of squares of sides
  c = 5 * Real.sqrt 29 :=
by
  sorry

end right_triangle_hypotenuse_l3305_330537


namespace parabola_equation_l3305_330514

/-- The standard equation of a parabola with vertex (0,0) and focus (3,0) -/
theorem parabola_equation (x y : ℝ) : 
  (∃ (p : ℝ), p > 0 ∧ y^2 = 4 * p * x ∧ 3 = p) → y^2 = 12 * x := by
  sorry

end parabola_equation_l3305_330514


namespace inequality_proof_l3305_330516

theorem inequality_proof (a b c d : ℝ) 
  (h1 : b + Real.sin a > d + Real.sin c) 
  (h2 : a + Real.sin b > c + Real.sin d) : 
  a + b > c + d := by
  sorry

end inequality_proof_l3305_330516


namespace candy_problem_l3305_330526

theorem candy_problem (initial_candies : ℕ) : 
  let day1_remaining := initial_candies / 2
  let day2_remaining := day1_remaining / 3 * 2
  let day3_remaining := day2_remaining / 4 * 3
  let day4_remaining := day3_remaining / 5 * 4
  let day5_remaining := day4_remaining / 6 * 5
  day5_remaining = 1 → initial_candies = 720 :=
by sorry

end candy_problem_l3305_330526


namespace set_conditions_equivalence_l3305_330542

theorem set_conditions_equivalence (m : ℝ) :
  let A := {x : ℝ | 0 < x - m ∧ x - m < 2}
  let B := {x : ℝ | -x^2 + 3*x ≤ 0}
  (A ∩ B = ∅ ∧ A ∪ B = B) ↔ (m ≤ -2 ∨ m ≥ 3) :=
sorry

end set_conditions_equivalence_l3305_330542


namespace one_third_percent_of_150_l3305_330552

theorem one_third_percent_of_150 : (1 / 3 : ℚ) / 100 * 150 = 0.5 := by sorry

end one_third_percent_of_150_l3305_330552


namespace ms_cole_total_students_l3305_330519

/-- The number of students in Ms. Cole's sixth-level math class -/
def S6 : ℕ := 40

/-- The number of students in Ms. Cole's fourth-level math class -/
def S4 : ℕ := 4 * S6

/-- The number of students in Ms. Cole's seventh-level math class -/
def S7 : ℕ := 2 * S4

/-- The total number of math students Ms. Cole teaches -/
def total_students : ℕ := S6 + S4 + S7

/-- Theorem stating that Ms. Cole teaches 520 math students in total -/
theorem ms_cole_total_students : total_students = 520 := by
  sorry

end ms_cole_total_students_l3305_330519


namespace sine_cosine_inequality_l3305_330587

theorem sine_cosine_inequality (α β : ℝ) (h : 0 < α + β ∧ α + β ≤ π) :
  (Real.sin α - Real.sin β) * (Real.cos α - Real.cos β) ≤ 0 := by
  sorry

end sine_cosine_inequality_l3305_330587


namespace light_path_in_cube_l3305_330568

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- Represents a light beam in the cube -/
structure LightBeam where
  start : Point3D
  reflectionPoint : Point3D

/-- The length of the light path in the cube -/
def lightPathLength (c : Cube) (lb : LightBeam) : ℝ :=
  sorry

theorem light_path_in_cube (c : Cube) (lb : LightBeam) :
  c.sideLength = 10 ∧
  lb.start = Point3D.mk 0 0 0 ∧
  lb.reflectionPoint = Point3D.mk 6 4 10 →
  lightPathLength c lb = 10 * Real.sqrt 152 :=
sorry

end light_path_in_cube_l3305_330568


namespace smallest_angle_is_76_l3305_330570

/-- A pentagon with angles in arithmetic sequence -/
structure ArithmeticPentagon where
  -- The common difference between consecutive angles
  d : ℝ
  -- The smallest angle
  a : ℝ
  -- The sum of all angles is 540°
  sum_constraint : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 540
  -- The largest angle is 140°
  max_angle : a + 4*d = 140

/-- 
If a pentagon has angles in arithmetic sequence and its largest angle is 140°,
then its smallest angle is 76°.
-/
theorem smallest_angle_is_76 (p : ArithmeticPentagon) : p.a = 76 := by
  sorry

end smallest_angle_is_76_l3305_330570


namespace hexagon_count_l3305_330593

theorem hexagon_count (initial_sheets : ℕ) (cuts : ℕ) (initial_sides_per_sheet : ℕ) :
  initial_sheets = 15 →
  cuts = 60 →
  initial_sides_per_sheet = 4 →
  let final_sheets := initial_sheets + cuts
  let total_sides := initial_sheets * initial_sides_per_sheet + cuts * 4
  let hexagon_count := (total_sides - 3 * final_sheets) / 3
  hexagon_count = 25 :=
by sorry

end hexagon_count_l3305_330593


namespace calculation_proof_l3305_330521

theorem calculation_proof : 61 + 5 * 12 / (180 / 3) = 62 := by
  sorry

end calculation_proof_l3305_330521
