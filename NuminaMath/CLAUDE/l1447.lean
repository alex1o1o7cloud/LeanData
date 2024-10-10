import Mathlib

namespace sequence_first_term_l1447_144742

theorem sequence_first_term (a : ℕ → ℚ) :
  (∀ n, a (n + 1) = 1 / (1 - a n)) →
  a 2 = 2 →
  a 1 = 1/2 := by
sorry

end sequence_first_term_l1447_144742


namespace intersection_when_a_is_one_union_equals_reals_iff_l1447_144792

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a : ℝ) : Set ℝ := {x | x - 4*a ≤ 0}

-- Part I
theorem intersection_when_a_is_one :
  A ∩ B 1 = {x : ℝ | x < -1 ∨ (3 < x ∧ x ≤ 4)} := by sorry

-- Part II
theorem union_equals_reals_iff (a : ℝ) :
  A ∪ B a = Set.univ ↔ a ≥ 3/4 := by sorry

end intersection_when_a_is_one_union_equals_reals_iff_l1447_144792


namespace common_number_in_overlapping_lists_l1447_144783

theorem common_number_in_overlapping_lists (numbers : List ℝ) : 
  numbers.length = 8 →
  (numbers.take 5).sum / 5 = 7 →
  (numbers.drop 5).sum / 3 = 10 →
  numbers.sum / 8 = 8 →
  ∃ x ∈ numbers.take 5 ∩ numbers.drop 5, x = 1 :=
by
  sorry


end common_number_in_overlapping_lists_l1447_144783


namespace prob_13_11_is_quarter_l1447_144731

/-- Represents a table tennis game with specific scoring probabilities -/
structure TableTennisGame where
  /-- Probability of player A scoring when A serves -/
  prob_a_scores_on_a_serve : ℝ
  /-- Probability of player A scoring when B serves -/
  prob_a_scores_on_b_serve : ℝ

/-- Calculates the probability of reaching a 13:11 score from a 10:10 tie -/
def prob_13_11 (game : TableTennisGame) : ℝ :=
  sorry

/-- The main theorem stating the probability of reaching 13:11 is 1/4 -/
theorem prob_13_11_is_quarter (game : TableTennisGame) 
  (h1 : game.prob_a_scores_on_a_serve = 2/3)
  (h2 : game.prob_a_scores_on_b_serve = 1/2) :
  prob_13_11 game = 1/4 := by
  sorry

end prob_13_11_is_quarter_l1447_144731


namespace reservoir_crossing_time_l1447_144737

/-- The time it takes to cross a reservoir under specific conditions -/
theorem reservoir_crossing_time
  (b : ℝ)  -- width of the reservoir in km
  (v : ℝ)  -- swimming speed of A and C in km/h
  (h1 : 0 < b)  -- reservoir width is positive
  (h2 : 0 < v)  -- swimming speed is positive
  : ∃ (t : ℝ), t = (31 * b) / (130 * v) ∧ 
    (∃ (x d : ℝ),
      0 < x ∧ 0 < d ∧
      x = (9 * b) / 13 ∧
      d = (b - x) / 2 ∧
      2 * d + x = b ∧
      (b + 3 * x) / 2 / (10 * v) = d / v ∧
      t = ((b + 2 * x) / (10 * v))) :=
sorry

end reservoir_crossing_time_l1447_144737


namespace integer_sqrt_15_l1447_144724

theorem integer_sqrt_15 (a : ℝ) : 
  (∃ m n : ℤ, (a + Real.sqrt 15 = m) ∧ (1 / (a - Real.sqrt 15) = n)) →
  (a = 4 + Real.sqrt 15 ∨ a = -(4 + Real.sqrt 15)) :=
by sorry

end integer_sqrt_15_l1447_144724


namespace max_value_implies_t_equals_one_l1447_144770

def f (t : ℝ) (x : ℝ) : ℝ := |x^2 - 2*x - t|

theorem max_value_implies_t_equals_one (t : ℝ) :
  (∀ x ∈ Set.Icc 0 3, f t x ≤ 2) ∧ (∃ x ∈ Set.Icc 0 3, f t x = 2) →
  t = 1 :=
by sorry

end max_value_implies_t_equals_one_l1447_144770


namespace theater_revenue_l1447_144776

/-- The total number of tickets sold -/
def total_tickets : ℕ := 800

/-- The price of an advanced ticket in cents -/
def advanced_price : ℕ := 1450

/-- The price of a door ticket in cents -/
def door_price : ℕ := 2200

/-- The number of tickets sold at the door -/
def door_tickets : ℕ := 672

/-- The total money taken in cents -/
def total_money : ℕ := 1664000

theorem theater_revenue :
  total_money = 
    (total_tickets - door_tickets) * advanced_price +
    door_tickets * door_price :=
by sorry

end theater_revenue_l1447_144776


namespace car_trip_duration_l1447_144759

theorem car_trip_duration :
  ∀ (total_time : ℝ) (second_part_time : ℝ),
    total_time > 0 →
    second_part_time ≥ 0 →
    total_time = 5 + second_part_time →
    (30 * 5 + 42 * second_part_time) / total_time = 34 →
    total_time = 7.5 := by
  sorry

end car_trip_duration_l1447_144759


namespace arithmetic_sequence_common_difference_l1447_144771

/-- An arithmetic sequence with given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_a5 : a 5 = 8)
  (h_a9 : a 9 = 24) :
  ∃ (d : ℝ), d = 4 ∧ ∀ n, a n = a 1 + (n - 1) * d :=
sorry

end arithmetic_sequence_common_difference_l1447_144771


namespace two_true_propositions_l1447_144781

theorem two_true_propositions :
  let P : ℝ → Prop := λ a => a > -3
  let Q : ℝ → Prop := λ a => a > -6
  let original := ∀ a, P a → Q a
  let converse := ∀ a, Q a → P a
  let inverse := ∀ a, ¬(P a) → ¬(Q a)
  let contrapositive := ∀ a, ¬(Q a) → ¬(P a)
  (original ∧ ¬converse ∧ ¬inverse ∧ contrapositive) :=
by sorry

end two_true_propositions_l1447_144781


namespace solution_set_of_sin_equation_l1447_144706

theorem solution_set_of_sin_equation :
  let S : Set ℝ := {x | 2 * Real.sin ((2/3) * x) = 1}
  S = {x | ∃ k : ℤ, x = 3 * k * Real.pi + Real.pi / 4 ∨ x = 3 * k * Real.pi + 5 * Real.pi / 4} := by
  sorry

end solution_set_of_sin_equation_l1447_144706


namespace triangle_perimeter_l1447_144746

/-- The perimeter of a right-angled triangle formed by the difference between two squares -/
theorem triangle_perimeter (x y : ℝ) (h : 0 < x ∧ x < y) :
  let side := (y - x) / 2
  let hypotenuse := (y - x) / Real.sqrt 2
  2 * side + hypotenuse = (y - x) * (1 + Real.sqrt 2) / Real.sqrt 2 :=
by sorry

end triangle_perimeter_l1447_144746


namespace solution_set_is_closed_interval_l1447_144748

-- Define the function representing the left side of the inequality
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the solution set
def solution_set : Set ℝ := {x | f x ≥ 0}

-- Theorem statement
theorem solution_set_is_closed_interval :
  solution_set = Set.Icc (-1) 3 := by sorry

end solution_set_is_closed_interval_l1447_144748


namespace brandon_squirrel_count_l1447_144700

/-- The number of squirrels Brandon can catch in an hour -/
def S : ℕ := sorry

/-- The number of rabbits Brandon can catch in an hour -/
def R : ℕ := 2

/-- The calorie content of a squirrel -/
def squirrel_calories : ℕ := 300

/-- The calorie content of a rabbit -/
def rabbit_calories : ℕ := 800

/-- The additional calories Brandon gets from catching squirrels instead of rabbits -/
def additional_calories : ℕ := 200

theorem brandon_squirrel_count :
  S * squirrel_calories = R * rabbit_calories + additional_calories ∧ S = 6 := by
  sorry

end brandon_squirrel_count_l1447_144700


namespace tangent_lines_to_circle_l1447_144738

/-- A line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a line is tangent to a circle -/
def tangent_to_circle (l : Line) (c : Circle) : Prop :=
  (abs (l.a * c.h + l.b * c.k + l.c) / Real.sqrt (l.a^2 + l.b^2)) = c.r

theorem tangent_lines_to_circle (given_line : Line) (c : Circle) :
  given_line = Line.mk 1 2 1 →
  c = Circle.mk 0 0 (Real.sqrt 5) →
  ∃ (l1 l2 : Line),
    parallel l1 given_line ∧
    parallel l2 given_line ∧
    tangent_to_circle l1 c ∧
    tangent_to_circle l2 c ∧
    ((l1 = Line.mk 1 2 5 ∧ l2 = Line.mk 1 2 (-5)) ∨
     (l1 = Line.mk 1 2 (-5) ∧ l2 = Line.mk 1 2 5)) :=
by sorry

end tangent_lines_to_circle_l1447_144738


namespace eight_divided_by_point_three_repeating_l1447_144789

theorem eight_divided_by_point_three_repeating (x : ℚ) : x = 1/3 → 8 / x = 24 := by
  sorry

end eight_divided_by_point_three_repeating_l1447_144789


namespace altitude_sum_of_specific_triangle_l1447_144760

/-- The sum of altitudes of a triangle formed by the line 10x + 8y = 80 and coordinate axes --/
theorem altitude_sum_of_specific_triangle : 
  let line : ℝ → ℝ → Prop := λ x y => 10 * x + 8 * y = 80
  let triangle := {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ line p.1 p.2}
  let altitudes := 
    [10, 8, (40 : ℝ) / Real.sqrt 41]  -- altitudes to y-axis, x-axis, and hypotenuse
  (altitudes.sum : ℝ) = 18 + 40 / Real.sqrt 41 := by
  sorry

end altitude_sum_of_specific_triangle_l1447_144760


namespace steve_monday_pounds_l1447_144761

/-- The amount of money Steve wants to make in total -/
def total_money : ℕ := 100

/-- The pay rate per pound of lingonberries -/
def pay_rate : ℕ := 2

/-- The number of pounds Steve picked on Thursday -/
def thursday_pounds : ℕ := 18

/-- The factor by which Tuesday's harvest was greater than Monday's -/
def tuesday_factor : ℕ := 3

theorem steve_monday_pounds : 
  ∃ (monday_pounds : ℕ), 
    monday_pounds + tuesday_factor * monday_pounds + thursday_pounds = total_money / pay_rate ∧ 
    monday_pounds = 8 := by
  sorry

end steve_monday_pounds_l1447_144761


namespace median_name_length_is_five_l1447_144702

/-- Represents the distribution of name lengths -/
structure NameLengthDistribution where
  fourLetters : Nat
  fiveLetters : Nat
  sixLetters : Nat
  sevenLetters : Nat

/-- Calculates the median of a list of numbers -/
def median (list : List Nat) : Rat :=
  sorry

/-- Generates a list of name lengths based on the distribution -/
def generateNameLengthList (dist : NameLengthDistribution) : List Nat :=
  sorry

theorem median_name_length_is_five (dist : NameLengthDistribution) : 
  dist.fourLetters = 9 ∧ 
  dist.fiveLetters = 6 ∧ 
  dist.sixLetters = 2 ∧ 
  dist.sevenLetters = 7 → 
  median (generateNameLengthList dist) = 5 := by
  sorry

end median_name_length_is_five_l1447_144702


namespace product_of_roots_quadratic_l1447_144774

theorem product_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*x₁ - 4 = 0) → (x₂^2 + 2*x₂ - 4 = 0) → x₁ * x₂ = -4 := by
  sorry

end product_of_roots_quadratic_l1447_144774


namespace maria_water_bottles_l1447_144773

theorem maria_water_bottles (initial_bottles : ℝ) (sister_drank : ℝ) (bottles_left : ℝ) :
  initial_bottles = 45.0 →
  sister_drank = 8.0 →
  bottles_left = 23 →
  initial_bottles - sister_drank - bottles_left = 14.0 :=
by
  sorry

end maria_water_bottles_l1447_144773


namespace equal_roots_quadratic_l1447_144764

/-- 
If the quadratic equation 2x^2 - 5x + m = 0 has two equal real roots,
then m = 25/8.
-/
theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - 5 * x + m = 0 ∧ 
   ∀ y : ℝ, 2 * y^2 - 5 * y + m = 0 → y = x) → 
  m = 25/8 := by
sorry

end equal_roots_quadratic_l1447_144764


namespace remainder_2n_mod_4_l1447_144707

theorem remainder_2n_mod_4 (n : ℕ) (h : n % 4 = 3) : (2 * n) % 4 = 2 := by
  sorry

end remainder_2n_mod_4_l1447_144707


namespace triangle_properties_l1447_144708

/-- Triangle ABC with vertices A(-1,-1), B(3,2), C(7,-7) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The specific triangle ABC given in the problem -/
def triangleABC : Triangle :=
  { A := (-1, -1)
    B := (3, 2)
    C := (7, -7) }

/-- Altitude from a point to a line -/
def altitude (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ → ℝ := sorry

/-- Area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Main theorem stating the properties of triangle ABC -/
theorem triangle_properties (t : Triangle) (h : t = triangleABC) :
  (altitude t.C (λ x => (3/4) * x - 5/4) = λ x => (-4/3) * x + 19/3) ∧
  triangleArea t = 24 := by sorry

end triangle_properties_l1447_144708


namespace solution_inequality_l1447_144703

theorem solution_inequality (x : ℝ) : 
  x > 0 → x * Real.sqrt (15 - x) + Real.sqrt (15 * x - x^3) ≥ 15 → x = 5 := by
  sorry

end solution_inequality_l1447_144703


namespace game_configurations_l1447_144766

/-- Represents the game state -/
structure GameState where
  blackPosition : Nat
  whiteCheckers : Nat

/-- The game rules -/
def gameRules : Prop :=
  ∃ (initialState : GameState),
    initialState.blackPosition = 3 ∧
    initialState.whiteCheckers = 2 ∧
    ∀ (moves : Nat),
      moves ≤ 2008 →
      ∃ (finalState : GameState),
        finalState.blackPosition = initialState.blackPosition + moves ∧
        finalState.whiteCheckers ≥ initialState.whiteCheckers ∧
        finalState.whiteCheckers ≤ initialState.whiteCheckers + moves

/-- The theorem to be proved -/
theorem game_configurations (rules : gameRules) :
  ∃ (finalConfigurations : Nat),
    finalConfigurations = 2009 ∧
    ∀ (state : GameState),
      state.blackPosition = 2011 →
      state.whiteCheckers ≥ 2 ∧
      state.whiteCheckers ≤ 2010 :=
sorry

end game_configurations_l1447_144766


namespace is_center_of_hyperbola_l1447_144725

/-- The equation of a hyperbola in the form ((4y-8)^2 / 7^2) - ((2x+6)^2 / 9^2) = 1 -/
def hyperbola_equation (x y : ℝ) : Prop :=
  (4 * y - 8)^2 / 7^2 - (2 * x + 6)^2 / 9^2 = 1

/-- The center of a hyperbola -/
def hyperbola_center : ℝ × ℝ := (-3, 2)

/-- Theorem stating that the given point is the center of the hyperbola -/
theorem is_center_of_hyperbola :
  ∀ x y : ℝ, hyperbola_equation x y ↔ hyperbola_equation (x - hyperbola_center.1) (y - hyperbola_center.2) :=
by sorry

end is_center_of_hyperbola_l1447_144725


namespace park_outer_diameter_l1447_144705

/-- Represents the dimensions of a circular park -/
structure CircularPark where
  pond_diameter : ℝ
  garden_width : ℝ
  path_width : ℝ

/-- Calculates the diameter of the outer boundary of a circular park -/
def outer_boundary_diameter (park : CircularPark) : ℝ :=
  park.pond_diameter + 2 * (park.garden_width + park.path_width)

/-- Theorem stating that for a park with given dimensions, the outer boundary diameter is 60 feet -/
theorem park_outer_diameter (park : CircularPark) 
  (h1 : park.pond_diameter = 16)
  (h2 : park.garden_width = 12)
  (h3 : park.path_width = 10) : 
  outer_boundary_diameter park = 60 := by sorry

end park_outer_diameter_l1447_144705


namespace ps_length_approx_l1447_144791

/-- A quadrilateral with given side and diagonal segment lengths -/
structure Quadrilateral :=
  (QT TS PT TR PQ : ℝ)

/-- The length of PS in the quadrilateral -/
noncomputable def lengthPS (q : Quadrilateral) : ℝ :=
  Real.sqrt (q.PT^2 + q.TS^2 - 2 * q.PT * q.TS * (-((q.PQ^2 - q.PT^2 - q.QT^2) / (2 * q.PT * q.QT))))

/-- Theorem stating that for a quadrilateral with given measurements, PS ≈ 19.9 -/
theorem ps_length_approx (q : Quadrilateral) 
  (h1 : q.QT = 5) (h2 : q.TS = 7) (h3 : q.PT = 9) (h4 : q.TR = 4) (h5 : q.PQ = 7) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |lengthPS q - 19.9| < ε :=
sorry

end ps_length_approx_l1447_144791


namespace oranges_count_l1447_144780

/-- The number of oranges initially in Tom's fruit bowl -/
def initial_oranges : ℕ := 3

/-- The number of lemons initially in Tom's fruit bowl -/
def initial_lemons : ℕ := 6

/-- The number of fruits Tom eats -/
def fruits_eaten : ℕ := 3

/-- The number of fruits remaining after Tom eats -/
def remaining_fruits : ℕ := 6

/-- Theorem stating that the number of oranges initially in the fruit bowl is 3 -/
theorem oranges_count : initial_oranges = 3 :=
  by
    have h1 : initial_oranges + initial_lemons = remaining_fruits + fruits_eaten :=
      by sorry
    have h2 : initial_oranges + 6 = 6 + 3 :=
      by sorry
    show initial_oranges = 3
    sorry

#check oranges_count

end oranges_count_l1447_144780


namespace four_tangent_lines_l1447_144787

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Counts the number of common tangent lines to two circles -/
def countCommonTangents (c1 c2 : Circle) : ℕ := sorry

/-- The main theorem -/
theorem four_tangent_lines (c1 c2 : Circle) 
  (h1 : c1.radius = 5)
  (h2 : c2.radius = 2)
  (h3 : Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2) = 10) :
  countCommonTangents c1 c2 = 4 := by sorry

end four_tangent_lines_l1447_144787


namespace flower_beds_count_l1447_144778

/-- Given that 10 seeds are put in each flower bed and 60 seeds were planted altogether,
    prove that the number of flower beds is 6. -/
theorem flower_beds_count (seeds_per_bed : ℕ) (total_seeds : ℕ) (num_beds : ℕ)
  (h1 : seeds_per_bed = 10)
  (h2 : total_seeds = 60)
  (h3 : num_beds * seeds_per_bed = total_seeds) :
  num_beds = 6 := by
  sorry

end flower_beds_count_l1447_144778


namespace ellipse_arithmetic_sequence_eccentricity_l1447_144762

/-- An ellipse with focal length, minor axis length, and major axis length in arithmetic sequence has eccentricity 3/5 -/
theorem ellipse_arithmetic_sequence_eccentricity :
  ∀ (a b c : ℝ),
    a > b ∧ b > 0 →  -- Conditions for a valid ellipse
    b = (a + c) / 2 →  -- Arithmetic sequence condition
    c^2 = a^2 - b^2 →  -- Relation between focal length and axes lengths
    c / a = 3 / 5 :=
by sorry


end ellipse_arithmetic_sequence_eccentricity_l1447_144762


namespace total_toys_given_l1447_144718

theorem total_toys_given (toy_cars : ℕ) (dolls : ℕ) (board_games : ℕ)
  (h1 : toy_cars = 134)
  (h2 : dolls = 269)
  (h3 : board_games = 87) :
  toy_cars + dolls + board_games = 490 := by
  sorry

end total_toys_given_l1447_144718


namespace class_fraction_proof_l1447_144777

theorem class_fraction_proof (G : ℚ) (B : ℚ) (T : ℚ) (F : ℚ) :
  B / G = 7 / 3 →
  T = B + G →
  (2 / 3) * G = F * T →
  F = 1 / 5 :=
by sorry

end class_fraction_proof_l1447_144777


namespace right_triangle_side_length_l1447_144755

theorem right_triangle_side_length (A B C : Real) (tanA : Real) (AC : Real) :
  tanA = 3 / 5 →
  AC = 10 →
  A^2 + B^2 = C^2 →
  A / C = tanA →
  B = 2 * Real.sqrt 34 :=
by
  sorry

end right_triangle_side_length_l1447_144755


namespace math_club_team_selection_l1447_144733

def mathClubSize : ℕ := 15
def teamSize : ℕ := 5

theorem math_club_team_selection :
  Nat.choose mathClubSize teamSize = 3003 := by
  sorry

end math_club_team_selection_l1447_144733


namespace marble_175_is_white_l1447_144799

/-- Represents the color of a marble -/
inductive MarbleColor
| Gray
| White
| Black

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : MarbleColor :=
  match n % 12 with
  | 0 | 1 | 2 | 3 | 4 => MarbleColor.Gray
  | 5 | 6 | 7 | 8 => MarbleColor.White
  | _ => MarbleColor.Black

/-- Theorem stating that the 175th marble is white -/
theorem marble_175_is_white : marbleColor 175 = MarbleColor.White := by
  sorry

end marble_175_is_white_l1447_144799


namespace line_property_l1447_144714

/-- Given a line passing through points (1, -1) and (3, 7), 
    prove that 3m - b = 17, where m is the slope and b is the y-intercept. -/
theorem line_property (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b) →  -- Line equation
  (1 : ℝ) * m + b = -1 →        -- Point (1, -1) satisfies the equation
  (3 : ℝ) * m + b = 7 →         -- Point (3, 7) satisfies the equation
  3 * m - b = 17 := by
sorry

end line_property_l1447_144714


namespace all_numbers_equal_l1447_144720

/-- Represents a grid of positive integers -/
def Grid := ℕ → ℕ → ℕ+

/-- Checks if two polygons are congruent -/
def CongruentPolygons (p q : Set (ℕ × ℕ)) : Prop := sorry

/-- Calculates the sum of numbers in a polygon -/
def PolygonSum (g : Grid) (p : Set (ℕ × ℕ)) : ℕ := sorry

/-- Calculates the area of a polygon -/
def PolygonArea (p : Set (ℕ × ℕ)) : ℕ := sorry

/-- Main theorem -/
theorem all_numbers_equal (g : Grid) (n : ℕ) (h_n : n > 2) :
  (∀ p q : Set (ℕ × ℕ), CongruentPolygons p q → PolygonArea p = n → PolygonArea q = n →
    PolygonSum g p = PolygonSum g q) →
  ∀ i j k l : ℕ, g i j = g k l :=
sorry

end all_numbers_equal_l1447_144720


namespace smallest_gcd_multiple_l1447_144736

theorem smallest_gcd_multiple (m n : ℕ+) (h : Nat.gcd m n = 18) :
  ∃ (k : ℕ+), k = Nat.gcd (12 * m) (20 * n) ∧ 
  (∀ (l : ℕ+), l = Nat.gcd (12 * m) (20 * n) → k ≤ l) ∧
  k = 72 := by
  sorry

end smallest_gcd_multiple_l1447_144736


namespace max_min_values_of_f_l1447_144753

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x^2 - 4 * x - 6)

theorem max_min_values_of_f :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 3, f x = max) ∧
    (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 3, f x = min) ∧
    max = 1 ∧
    min = 1 / Real.exp 8 :=
by sorry

end max_min_values_of_f_l1447_144753


namespace fence_cost_square_plot_l1447_144717

/-- The cost of building a fence around a square plot -/
theorem fence_cost_square_plot (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 54) :
  let side_length : ℝ := Real.sqrt area
  let perimeter : ℝ := 4 * side_length
  let total_cost : ℝ := perimeter * price_per_foot
  total_cost = 3672 := by
sorry


end fence_cost_square_plot_l1447_144717


namespace intersection_one_element_l1447_144790

theorem intersection_one_element (a : ℝ) : 
  let A : Set ℝ := {1, a, 5}
  let B : Set ℝ := {2, a^2 + 1}
  (∃! x, x ∈ A ∩ B) → (a = 0 ∨ a = -2) :=
by sorry

end intersection_one_element_l1447_144790


namespace quadratic_roots_real_and_equal_l1447_144723

/-- The quadratic equation x^2 + 2x√3 + 3 = 0 has real and equal roots given that its discriminant is zero -/
theorem quadratic_roots_real_and_equal :
  let a : ℝ := 1
  let b : ℝ := 2 * Real.sqrt 3
  let c : ℝ := 3
  let discriminant := b^2 - 4*a*c
  discriminant = 0 →
  ∃! x : ℝ, a * x^2 + b * x + c = 0 :=
by sorry

end quadratic_roots_real_and_equal_l1447_144723


namespace new_mixture_ratio_l1447_144704

/-- Represents a mixture of alcohol and water -/
structure Mixture where
  alcohol : ℚ
  water : ℚ

/-- Calculates the ratio of alcohol to water in a mixture -/
def ratioAlcoholToWater (m : Mixture) : ℚ := m.alcohol / m.water

/-- First jar with 3:1 ratio and 4 liters total -/
def jar1 : Mixture := { alcohol := 3, water := 1 }

/-- Second jar with 2:1 ratio and 6 liters total -/
def jar2 : Mixture := { alcohol := 4, water := 2 }

/-- Amount taken from first jar -/
def amount1 : ℚ := 1

/-- Amount taken from second jar -/
def amount2 : ℚ := 2

/-- New mixture created from combining portions of jar1 and jar2 -/
def newMixture : Mixture := {
  alcohol := amount1 * (jar1.alcohol / (jar1.alcohol + jar1.water)) + 
             amount2 * (jar2.alcohol / (jar2.alcohol + jar2.water)),
  water := amount1 * (jar1.water / (jar1.alcohol + jar1.water)) + 
           amount2 * (jar2.water / (jar2.alcohol + jar2.water))
}

theorem new_mixture_ratio : 
  ratioAlcoholToWater newMixture = 41 / 19 := by sorry

end new_mixture_ratio_l1447_144704


namespace one_minus_repeating_decimal_l1447_144793

/-- The value of the repeating decimal 0.123123... -/
def repeating_decimal : ℚ := 41 / 333

/-- Theorem: 1 - 0.123123... = 292/333 -/
theorem one_minus_repeating_decimal :
  1 - repeating_decimal = 292 / 333 := by
  sorry

end one_minus_repeating_decimal_l1447_144793


namespace device_improvement_l1447_144785

/-- Represents the sample mean and variance of a device's measurements -/
structure DeviceStats where
  mean : ℝ
  variance : ℝ

/-- Determines if there's a significant improvement between two devices -/
def significantImprovement (old new : DeviceStats) : Prop :=
  new.mean - old.mean ≥ 2 * Real.sqrt ((old.variance + new.variance) / 10)

theorem device_improvement (old new : DeviceStats) 
  (h_old : old = ⟨10, 0.036⟩) 
  (h_new : new = ⟨10.3, 0.04⟩) : 
  significantImprovement old new := by
  sorry

#check device_improvement

end device_improvement_l1447_144785


namespace banana_pear_weight_equivalence_l1447_144757

theorem banana_pear_weight_equivalence (banana_weight pear_weight : ℝ) 
  (h1 : 9 * banana_weight = 6 * pear_weight) :
  36 * banana_weight = 24 * pear_weight := by
  sorry

end banana_pear_weight_equivalence_l1447_144757


namespace abc_sum_l1447_144784

/-- Given prime numbers a, b, c satisfying abc + a = 851, prove a + b + c = 50 -/
theorem abc_sum (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c)
  (heq : a * b * c + a = 851) : a + b + c = 50 := by
  sorry

end abc_sum_l1447_144784


namespace addition_problem_l1447_144711

theorem addition_problem : ∃! x : ℝ, 8 + x = -5 ∧ x = -13 := by sorry

end addition_problem_l1447_144711


namespace emily_took_55_apples_l1447_144782

/-- The number of apples Ruby initially had -/
def initial_apples : ℕ := 63

/-- The number of apples Ruby has left -/
def remaining_apples : ℕ := 8

/-- The number of apples Emily took -/
def emily_took : ℕ := initial_apples - remaining_apples

/-- Theorem stating that Emily took 55 apples -/
theorem emily_took_55_apples : emily_took = 55 := by
  sorry

end emily_took_55_apples_l1447_144782


namespace largest_number_with_conditions_l1447_144741

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 3

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_number_with_conditions :
  ∀ n : ℕ,
    is_valid_number n ∧
    digit_sum n = 14 →
    n ≤ 333322 :=
by sorry

end largest_number_with_conditions_l1447_144741


namespace library_books_before_grant_l1447_144763

/-- The number of books purchased with the grant -/
def books_purchased : ℕ := 2647

/-- The total number of books after the purchase -/
def total_books : ℕ := 8582

/-- The number of books before the grant -/
def books_before : ℕ := total_books - books_purchased

theorem library_books_before_grant :
  books_before = 5935 :=
sorry

end library_books_before_grant_l1447_144763


namespace evaluate_expression_l1447_144721

theorem evaluate_expression : (827 * 827) - (826 * 828) + 2 = 3 := by
  sorry

end evaluate_expression_l1447_144721


namespace geometric_sequence_first_term_l1447_144719

theorem geometric_sequence_first_term 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_positive : q > 0) 
  (h_geometric : ∀ n, a (n + 1) = a n * q) 
  (h_condition : a 3 * a 9 = 2 * (a 5)^2) 
  (h_second_term : a 2 = 1) : 
  a 1 = Real.sqrt 2 / 2 := by
sorry

end geometric_sequence_first_term_l1447_144719


namespace scissors_count_l1447_144769

theorem scissors_count (initial : Nat) (added : Nat) (total : Nat) : 
  initial = 54 → added = 22 → total = initial + added → total = 76 := by
  sorry

end scissors_count_l1447_144769


namespace fraction_geq_one_iff_x_in_range_l1447_144739

theorem fraction_geq_one_iff_x_in_range (x : ℝ) : 2 / x ≥ 1 ↔ 0 < x ∧ x ≤ 2 :=
sorry

end fraction_geq_one_iff_x_in_range_l1447_144739


namespace order_of_numbers_l1447_144728

theorem order_of_numbers : ∀ (a b c : ℝ), 
  a = 6^(1/2) → b = (1/2)^6 → c = Real.log 6 / Real.log (1/2) →
  c < b ∧ b < a :=
by
  sorry

end order_of_numbers_l1447_144728


namespace fabulous_iff_not_power_of_two_l1447_144796

def is_fabulous (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ a : ℕ, 2 ≤ a ∧ a ≤ n - 1 ∧ (n ∣ a^n - a)

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

theorem fabulous_iff_not_power_of_two (n : ℕ) :
  is_fabulous n ↔ (¬ is_power_of_two n) :=
sorry

end fabulous_iff_not_power_of_two_l1447_144796


namespace a_range_l1447_144772

-- Define the statements p and q
def p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

-- Theorem statement
theorem a_range (a : ℝ) : 
  (a > 0) → 
  (∀ x, q x → p x a) → 
  (∃ x, p x a ∧ ¬q x) → 
  1 ≤ a ∧ a ≤ 2 := by
  sorry

end a_range_l1447_144772


namespace rose_bushes_count_l1447_144710

/-- The number of rose bushes in the park after planting -/
def final_roses : ℕ := 6

/-- The number of new rose bushes to be planted -/
def new_roses : ℕ := 4

/-- The number of rose bushes currently in the park -/
def current_roses : ℕ := final_roses - new_roses

theorem rose_bushes_count : current_roses = 2 := by
  sorry

end rose_bushes_count_l1447_144710


namespace function_eq_zero_l1447_144744

theorem function_eq_zero (f : ℝ → ℝ) 
  (h1 : ∀ u v : ℝ, f (2 * u) = f (u + v) * f (v - u) + f (u - v) * f (-u - v))
  (h2 : ∀ u : ℝ, f u ≥ 0) :
  ∀ x : ℝ, f x = 0 := by
sorry

end function_eq_zero_l1447_144744


namespace A_characterization_l1447_144779

/-- The set A defined by the quadratic equation kx^2 - 3x + 2 = 0 -/
def A (k : ℝ) : Set ℝ := {x | k * x^2 - 3 * x + 2 = 0}

/-- Theorem stating the conditions for A to be empty or contain exactly one element -/
theorem A_characterization (k : ℝ) :
  (A k = ∅ ↔ k > 9/8) ∧
  (∃! x, x ∈ A k ↔ k = 0 ∨ k = 9/8) ∧
  (k = 0 → A k = {2/3}) ∧
  (k = 9/8 → A k = {4/3}) :=
sorry

end A_characterization_l1447_144779


namespace only_first_is_prime_one_prime_in_sequence_l1447_144797

/-- Generates the nth number in the sequence starting with 47 and repeating 47 sequentially -/
def sequenceNumber (n : ℕ) : ℕ :=
  if n = 0 then 47 else
  (sequenceNumber (n - 1)) * 100 + 47

/-- Theorem stating that only the first number in the sequence is prime -/
theorem only_first_is_prime :
  ∀ n : ℕ, n > 0 → ¬ Nat.Prime (sequenceNumber n) :=
by
  sorry

/-- Corollary stating that there is exactly one prime number in the sequence -/
theorem one_prime_in_sequence :
  (∃! k : ℕ, Nat.Prime (sequenceNumber k)) :=
by
  sorry

end only_first_is_prime_one_prime_in_sequence_l1447_144797


namespace iced_tea_price_l1447_144709

/-- The cost of a beverage order --/
def order_cost (cappuccino_price : ℚ) (latte_price : ℚ) (espresso_price : ℚ) (iced_tea_price : ℚ) : ℚ :=
  3 * cappuccino_price + 2 * iced_tea_price + 2 * latte_price + 2 * espresso_price

theorem iced_tea_price (cappuccino_price latte_price espresso_price : ℚ)
  (h1 : cappuccino_price = 2)
  (h2 : latte_price = 3/2)
  (h3 : espresso_price = 1)
  (h4 : ∃ (x : ℚ), order_cost cappuccino_price latte_price espresso_price x = 17) :
  ∃ (x : ℚ), x = 3 ∧ order_cost cappuccino_price latte_price espresso_price x = 17 :=
sorry

end iced_tea_price_l1447_144709


namespace inequality_equivalence_l1447_144747

def solution_set (x y : ℝ) : Prop :=
  (x ≤ -1 ∧ y ≤ x + 2 ∧ y ≥ -x - 2) ∨
  (-1 < x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1) ∨
  (x > 1 ∧ y ≤ 2 - x ∧ y ≥ x - 2)

theorem inequality_equivalence (x y : ℝ) :
  |x - 1| + |x + 1| + |2 * y| ≤ 4 ↔ solution_set x y := by
  sorry

end inequality_equivalence_l1447_144747


namespace divisibility_implies_unit_l1447_144752

theorem divisibility_implies_unit (a b c d : ℤ) 
  (h1 : (ab - cd) ∣ a) 
  (h2 : (ab - cd) ∣ b) 
  (h3 : (ab - cd) ∣ c) 
  (h4 : (ab - cd) ∣ d) : 
  ab - cd = 1 ∨ ab - cd = -1 :=
by
  sorry

end divisibility_implies_unit_l1447_144752


namespace mixed_fruit_juice_cost_l1447_144740

/-- The cost per litre of the superfruit juice cocktail -/
def superfruit_cost : ℝ := 1399.45

/-- The cost per litre of the açaí berry juice -/
def acai_cost : ℝ := 3104.35

/-- The volume of mixed fruit juice used -/
def mixed_fruit_volume : ℝ := 33

/-- The volume of açaí berry juice used -/
def acai_volume : ℝ := 22

/-- The cost per litre of the mixed fruit juice -/
def mixed_fruit_cost : ℝ := 256.79

theorem mixed_fruit_juice_cost : 
  mixed_fruit_volume * mixed_fruit_cost + acai_volume * acai_cost = 
  (mixed_fruit_volume + acai_volume) * superfruit_cost :=
by sorry

end mixed_fruit_juice_cost_l1447_144740


namespace sqrt_280_between_16_and_17_l1447_144732

theorem sqrt_280_between_16_and_17 : 16 < Real.sqrt 280 ∧ Real.sqrt 280 < 17 := by
  sorry

end sqrt_280_between_16_and_17_l1447_144732


namespace max_students_per_classroom_l1447_144716

/-- Theorem: Maximum students per classroom with equal gender distribution -/
theorem max_students_per_classroom 
  (num_classrooms : ℕ) 
  (num_boys : ℕ) 
  (num_girls : ℕ) 
  (h1 : num_classrooms = 7)
  (h2 : num_boys = 68)
  (h3 : num_girls = 53) :
  ∃ (students_per_classroom : ℕ),
    students_per_classroom = 14 ∧
    students_per_classroom ≤ min num_boys num_girls ∧
    students_per_classroom % 2 = 0 ∧
    (students_per_classroom / 2) * num_classrooms ≤ min num_boys num_girls :=
by
  sorry

#check max_students_per_classroom

end max_students_per_classroom_l1447_144716


namespace max_value_sqrt_sum_l1447_144786

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 8) :
  Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2) ≤ 3 * Real.sqrt 10 := by
sorry

end max_value_sqrt_sum_l1447_144786


namespace shirt_price_l1447_144765

/-- Given a shirt and coat with a total cost of 600 dollars, where the shirt costs
    one-third the price of the coat, prove that the shirt costs 150 dollars. -/
theorem shirt_price (total_cost : ℝ) (shirt_price : ℝ) (coat_price : ℝ) 
  (h1 : total_cost = 600)
  (h2 : shirt_price + coat_price = total_cost)
  (h3 : shirt_price = (1/3) * coat_price) :
  shirt_price = 150 := by
  sorry

end shirt_price_l1447_144765


namespace necessary_not_sufficient_cube_condition_nonzero_sum_of_squares_iff_not_both_zero_l1447_144734

-- Statement 1
theorem necessary_not_sufficient_cube_condition (x : ℝ) :
  (x^3 = -27 → x^2 = 9) ∧ ¬(x^2 = 9 → x^3 = -27) :=
sorry

-- Statement 2
theorem nonzero_sum_of_squares_iff_not_both_zero (a b : ℝ) :
  a^2 + b^2 ≠ 0 ↔ ¬(a = 0 ∧ b = 0) :=
sorry

end necessary_not_sufficient_cube_condition_nonzero_sum_of_squares_iff_not_both_zero_l1447_144734


namespace unique_solution_modular_equation_l1447_144756

theorem unique_solution_modular_equation :
  ∃! n : ℕ, n < 103 ∧ (100 * n) % 103 = 65 % 103 ∧ n = 68 := by sorry

end unique_solution_modular_equation_l1447_144756


namespace stratified_sampling_correct_l1447_144798

/-- Represents the number of employees in each age group -/
structure EmployeeGroups where
  over50 : ℕ
  between35and49 : ℕ
  under35 : ℕ

/-- Represents the sampling results for each age group -/
structure SamplingResult where
  over50 : ℕ
  between35and49 : ℕ
  under35 : ℕ

/-- Calculates the correct stratified sampling for given employee groups and sample size -/
def stratifiedSampling (groups : EmployeeGroups) (sampleSize : ℕ) : SamplingResult :=
  sorry

/-- The theorem statement for the stratified sampling problem -/
theorem stratified_sampling_correct 
  (groups : EmployeeGroups)
  (h1 : groups.over50 = 15)
  (h2 : groups.between35and49 = 45)
  (h3 : groups.under35 = 90)
  (h4 : groups.over50 + groups.between35and49 + groups.under35 = 150)
  (sampleSize : ℕ)
  (h5 : sampleSize = 30) :
  stratifiedSampling groups sampleSize = SamplingResult.mk 3 9 18 :=
by sorry

end stratified_sampling_correct_l1447_144798


namespace jeanette_juggling_progress_l1447_144775

/-- Calculates the number of objects Jeanette can juggle after a given number of weeks -/
def juggle_objects (initial_objects : ℕ) (weekly_increase : ℕ) (sessions_per_week : ℕ) (session_increase : ℕ) (weeks : ℕ) : ℕ :=
  initial_objects + weeks * (weekly_increase + sessions_per_week * session_increase)

/-- Proves that Jeanette can juggle 21 objects by the end of the 5th week -/
theorem jeanette_juggling_progress : 
  juggle_objects 3 2 3 1 5 = 21 := by
  sorry

#eval juggle_objects 3 2 3 1 5

end jeanette_juggling_progress_l1447_144775


namespace probability_two_heads_and_three_l1447_144767

def coin_outcomes : ℕ := 2
def die_outcomes : ℕ := 6

def total_outcomes : ℕ := coin_outcomes * coin_outcomes * die_outcomes

def favorable_outcome : ℕ := 1

theorem probability_two_heads_and_three : 
  (favorable_outcome : ℚ) / total_outcomes = 1 / 24 := by sorry

end probability_two_heads_and_three_l1447_144767


namespace gwen_book_collection_total_l1447_144745

/-- Represents Gwen's book collection --/
structure BookCollection where
  mystery_shelves : ℕ
  mystery_books_per_shelf : ℕ
  picture_shelves : ℕ
  picture_books_per_shelf : ℕ
  scifi_shelves : ℕ
  scifi_books_per_shelf : ℕ
  nonfiction_shelves : ℕ
  nonfiction_books_per_shelf : ℕ
  mystery_books_lent : ℕ
  scifi_books_lent : ℕ
  picture_books_borrowed : ℕ

/-- Calculates the total number of books in Gwen's collection --/
def total_books (collection : BookCollection) : ℕ :=
  (collection.mystery_shelves * collection.mystery_books_per_shelf - collection.mystery_books_lent) +
  (collection.picture_shelves * collection.picture_books_per_shelf) +
  (collection.scifi_shelves * collection.scifi_books_per_shelf - collection.scifi_books_lent) +
  (collection.nonfiction_shelves * collection.nonfiction_books_per_shelf)

/-- Theorem stating that Gwen's book collection contains 106 books --/
theorem gwen_book_collection_total :
  ∃ (collection : BookCollection),
    collection.mystery_shelves = 8 ∧
    collection.mystery_books_per_shelf = 6 ∧
    collection.picture_shelves = 5 ∧
    collection.picture_books_per_shelf = 4 ∧
    collection.scifi_shelves = 4 ∧
    collection.scifi_books_per_shelf = 7 ∧
    collection.nonfiction_shelves = 3 ∧
    collection.nonfiction_books_per_shelf = 5 ∧
    collection.mystery_books_lent = 2 ∧
    collection.scifi_books_lent = 3 ∧
    collection.picture_books_borrowed = 5 ∧
    total_books collection = 106 :=
by sorry


end gwen_book_collection_total_l1447_144745


namespace calculation_proof_l1447_144726

theorem calculation_proof : 3 * 15 + 3 * 16 + 3 * 19 + 11 = 161 := by
  sorry

end calculation_proof_l1447_144726


namespace line_circle_orthogonality_l1447_144794

/-- Given a line and a circle, prove that a specific value of 'a' ensures orthogonality of OA and OB -/
theorem line_circle_orthogonality (a : ℝ) (A B : ℝ × ℝ) :
  (∀ (x y : ℝ), x - 2*y + a = 0 → x^2 + y^2 = 2) →  -- Line intersects circle
  (A.1 - 2*A.2 + a = 0 ∧ A.1^2 + A.2^2 = 2) →        -- A satisfies both equations
  (B.1 - 2*B.2 + a = 0 ∧ B.1^2 + B.2^2 = 2) →        -- B satisfies both equations
  a = Real.sqrt 5 →                                  -- Specific value of a
  A.1 * B.1 + A.2 * B.2 = 0                          -- OA · OB = 0
  := by sorry

end line_circle_orthogonality_l1447_144794


namespace student_minimum_earnings_l1447_144729

/-- Represents the student's work situation -/
structure WorkSituation where
  library_rate : ℝ
  construction_rate : ℝ
  total_hours : ℝ
  library_hours : ℝ

/-- Calculates the minimum weekly earnings for the student -/
def minimum_weekly_earnings (w : WorkSituation) : ℝ :=
  w.library_rate * w.library_hours + 
  w.construction_rate * (w.total_hours - w.library_hours)

/-- Theorem stating the minimum weekly earnings for the given work situation -/
theorem student_minimum_earnings : 
  let w : WorkSituation := {
    library_rate := 8,
    construction_rate := 15,
    total_hours := 25,
    library_hours := 10
  }
  minimum_weekly_earnings w = 305 := by sorry

end student_minimum_earnings_l1447_144729


namespace sqrt_last_digit_exists_l1447_144727

/-- A p-adic number -/
structure PAdicNumber (p : ℕ) where
  digits : ℕ → ℕ
  last_digit : ℕ

/-- The concept of square root in p-arithmetic -/
def has_sqrt_p_adic (α : PAdicNumber p) : Prop :=
  ∃ β : PAdicNumber p, β.digits 0 ^ 2 ≡ α.digits 0 [MOD p]

/-- The main theorem -/
theorem sqrt_last_digit_exists (p : ℕ) (α : PAdicNumber p) :
  has_sqrt_p_adic α → ∃ x : ℕ, x ^ 2 ≡ α.last_digit [MOD p] :=
sorry

end sqrt_last_digit_exists_l1447_144727


namespace multiplication_properties_l1447_144795

theorem multiplication_properties (m n : ℕ) :
  let a := 6 * m + 1
  let b := 6 * n + 1
  let c := 6 * m + 5
  let d := 6 * n + 5
  (∃ k : ℕ, a * b = 6 * k + 1) ∧
  (∃ k : ℕ, c * d = 6 * k + 1) ∧
  (∃ k : ℕ, a * d = 6 * k + 5) :=
by sorry

end multiplication_properties_l1447_144795


namespace congruent_integers_count_l1447_144730

theorem congruent_integers_count : 
  (Finset.filter (fun n => n > 0 ∧ n < 2000 ∧ n % 13 = 6) (Finset.range 2000)).card = 154 :=
by sorry

end congruent_integers_count_l1447_144730


namespace salary_increase_after_three_years_l1447_144749

theorem salary_increase_after_three_years (annual_raise : Real) (years : Nat) : 
  annual_raise = 0.15 → years = 3 → 
  ((1 + annual_raise) ^ years - 1) * 100 = 52.0875 := by
  sorry

end salary_increase_after_three_years_l1447_144749


namespace square_area_from_smaller_squares_l1447_144751

/-- The area of a square composed of smaller squares -/
theorem square_area_from_smaller_squares
  (n : ℕ) -- number of smaller squares
  (side_length : ℝ) -- side length of each smaller square
  (h_n : n = 8) -- there are 8 smaller squares
  (h_side : side_length = 2) -- side length of each smaller square is 2 cm
  : n * side_length^2 = 32 := by
  sorry

end square_area_from_smaller_squares_l1447_144751


namespace infinite_primes_satisfying_conditions_l1447_144788

/-- The set of odd prime numbers -/
def OddPrimes : Set Nat := {p | Nat.Prime p ∧ p % 2 = 1}

/-- The remainder of the Euclidean division of n by p -/
def d_p (p n : Nat) : Nat := n % p

/-- A p-sequence is a sequence where a_{n+1} = a_n + d_p(a_n) -/
def IsPSequence (p : Nat) (a : Nat → Nat) : Prop :=
  ∀ n, a (n + 1) = a n + d_p p (a n)

/-- The set of primes satisfying the first condition -/
def PrimesCondition1 : Set Nat :=
  {p ∈ OddPrimes | ∃ a b : Nat → Nat,
    IsPSequence p a ∧ IsPSequence p b ∧
    (∃ S1 S2 : Set Nat, S1.Infinite ∧ S2.Infinite ∧
      (∀ n ∈ S1, a n > b n) ∧ (∀ n ∈ S2, a n < b n))}

/-- The set of primes satisfying the second condition -/
def PrimesCondition2 : Set Nat :=
  {p ∈ OddPrimes | ∃ a b : Nat → Nat,
    IsPSequence p a ∧ IsPSequence p b ∧
    a 0 < b 0 ∧ (∀ n ≥ 1, a n > b n)}

theorem infinite_primes_satisfying_conditions :
  PrimesCondition1.Infinite ∧ PrimesCondition2.Infinite := by
  sorry

end infinite_primes_satisfying_conditions_l1447_144788


namespace line_intersects_circle_l1447_144735

/-- A circle with center O and radius r -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance from a point to a line -/
def distancePointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  sorry

/-- Relationship between a line and a circle -/
inductive LineCircleRelation
  | Disjoint
  | Tangent
  | Intersect

theorem line_intersects_circle (O : Circle) (l : Line) :
  O.radius = 4 →
  distancePointToLine O.center l = 3 →
  LineCircleRelation.Intersect = 
    match O.radius, distancePointToLine O.center l with
    | r, d => if r > d then LineCircleRelation.Intersect
              else if r = d then LineCircleRelation.Tangent
              else LineCircleRelation.Disjoint :=
by
  sorry

end line_intersects_circle_l1447_144735


namespace student_marks_l1447_144722

theorem student_marks (M P C : ℕ) : 
  C = P + 20 →
  (M + C) / 2 = 25 →
  M + P = 30 :=
by sorry

end student_marks_l1447_144722


namespace animal_arrangement_count_l1447_144754

/-- The number of ways to arrange animals in cages -/
def arrange_animals (num_chickens num_dogs num_cats num_rabbits : ℕ) : ℕ :=
  Nat.factorial 4 * 
  Nat.factorial num_chickens * 
  Nat.factorial num_dogs * 
  Nat.factorial num_cats * 
  Nat.factorial num_rabbits

/-- Theorem stating the number of arrangements for the given animals -/
theorem animal_arrangement_count :
  arrange_animals 3 3 4 2 = 41472 := by
  sorry

end animal_arrangement_count_l1447_144754


namespace incorrect_fraction_equality_l1447_144713

theorem incorrect_fraction_equality (a b : ℝ) (h : 0.7 * a ≠ b) :
  (0.2 * a + b) / (0.7 * a - b) ≠ (2 * a + b) / (7 * a - b) :=
sorry

end incorrect_fraction_equality_l1447_144713


namespace minimal_intercept_line_properties_l1447_144743

/-- A line that passes through (1, 4) with positive intercepts and minimal sum of intercepts -/
def minimal_intercept_line (x y : ℝ) : Prop :=
  x + y = 5

theorem minimal_intercept_line_properties :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  minimal_intercept_line 1 4 ∧
  (∀ x y, minimal_intercept_line x y → x = 0 ∨ y = 0 → x = a ∨ y = b) ∧
  (∀ c d : ℝ, c > 0 → d > 0 →
    (∃ x y, x + y = c + d ∧ (x = 0 ∨ y = 0)) →
    a + b ≤ c + d) :=
by sorry

end minimal_intercept_line_properties_l1447_144743


namespace salary_calculation_l1447_144750

theorem salary_calculation (salary : ℚ) : 
  (salary * (1 - 0.2) * (1 - 0.1) * (1 - 0.1) = 1377) → 
  salary = 2125 := by
sorry

end salary_calculation_l1447_144750


namespace probability_exactly_one_instrument_l1447_144758

/-- The probability of playing exactly one instrument in a group -/
theorem probability_exactly_one_instrument 
  (total_people : ℕ) 
  (at_least_one_fraction : ℚ) 
  (two_or_more : ℕ) 
  (h1 : total_people = 800) 
  (h2 : at_least_one_fraction = 1 / 5) 
  (h3 : two_or_more = 64) : 
  (↑((at_least_one_fraction * ↑total_people).num - two_or_more) / ↑total_people : ℚ) = 3 / 25 := by
  sorry

end probability_exactly_one_instrument_l1447_144758


namespace paper_edge_length_l1447_144768

theorem paper_edge_length (cube_edge : ℝ) (num_papers : ℕ) :
  cube_edge = 12 →
  num_papers = 54 →
  ∃ (paper_edge : ℝ),
    paper_edge^2 * num_papers = 6 * cube_edge^2 ∧
    paper_edge = 4 := by
  sorry

end paper_edge_length_l1447_144768


namespace cube_volume_problem_l1447_144715

theorem cube_volume_problem (a : ℝ) : 
  a > 0 → 
  (a - 2) * a * (a + 2) = a^3 - 16 → 
  a^3 = 64 := by
sorry

end cube_volume_problem_l1447_144715


namespace equation_solution_l1447_144712

theorem equation_solution : ∃! x : ℝ, 
  x ≠ 2 ∧ x ≠ 3 ∧ (x^3 - 4*x^2)/(x^2 - 5*x + 6) - x = 9 := by
  sorry

end equation_solution_l1447_144712


namespace angle_sum_is_pi_half_l1447_144701

theorem angle_sum_is_pi_half (α β : Real) (h_acute_α : 0 < α ∧ α < π/2) (h_acute_β : 0 < β ∧ β < π/2) 
  (h_eq1 : 3 * Real.sin α ^ 2 + 2 * Real.sin β ^ 2 = 1) 
  (h_eq2 : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) : 
  α + 2 * β = π/2 := by
sorry

end angle_sum_is_pi_half_l1447_144701
