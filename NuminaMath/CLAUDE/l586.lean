import Mathlib

namespace NUMINAMATH_CALUDE_mario_garden_after_two_weeks_l586_58660

/-- Calculates the number of flowers on a plant after a given number of weeks -/
def flowers_after_weeks (initial : ℕ) (growth_rate : ℕ) (weeks : ℕ) : ℕ :=
  initial + growth_rate * weeks

/-- Calculates the number of flowers on a plant that doubles each week -/
def flowers_doubling (initial : ℕ) (weeks : ℕ) : ℕ :=
  initial * (2^weeks)

/-- Represents Mario's garden and calculates the total number of blossoms -/
def mario_garden (weeks : ℕ) : ℕ :=
  let hibiscus1 := flowers_after_weeks 2 3 weeks
  let hibiscus2 := flowers_after_weeks 4 4 weeks
  let hibiscus3 := flowers_after_weeks 16 5 weeks
  let rose1 := flowers_after_weeks 3 2 weeks
  let rose2 := flowers_after_weeks 5 3 weeks
  let sunflower := flowers_doubling 6 weeks
  hibiscus1 + hibiscus2 + hibiscus3 + rose1 + rose2 + sunflower

theorem mario_garden_after_two_weeks :
  mario_garden 2 = 88 := by
  sorry

end NUMINAMATH_CALUDE_mario_garden_after_two_weeks_l586_58660


namespace NUMINAMATH_CALUDE_tan_sum_product_equality_l586_58688

theorem tan_sum_product_equality (α β γ : ℝ) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) 
  (h_sum : α + β + γ = π / 2) : 
  Real.tan α * Real.tan β + Real.tan α * Real.tan γ + Real.tan β * Real.tan γ = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_product_equality_l586_58688


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l586_58667

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 9 → y = 53 / 3 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l586_58667


namespace NUMINAMATH_CALUDE_negative_cube_squared_l586_58646

theorem negative_cube_squared (a : ℝ) : (-a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l586_58646


namespace NUMINAMATH_CALUDE_water_added_to_tank_l586_58662

theorem water_added_to_tank (tank_capacity : ℚ) 
  (h1 : tank_capacity = 56)
  (initial_fraction : ℚ) (final_fraction : ℚ)
  (h2 : initial_fraction = 3/4)
  (h3 : final_fraction = 7/8) :
  final_fraction * tank_capacity - initial_fraction * tank_capacity = 7 := by
  sorry

end NUMINAMATH_CALUDE_water_added_to_tank_l586_58662


namespace NUMINAMATH_CALUDE_mario_garden_flowers_l586_58655

/-- The number of flowers on Mario's first hibiscus plant -/
def first_plant : ℕ := 2

/-- The number of flowers on Mario's second hibiscus plant -/
def second_plant : ℕ := 2 * first_plant

/-- The number of flowers on Mario's third hibiscus plant -/
def third_plant : ℕ := 4 * second_plant

/-- The total number of flowers in Mario's garden -/
def total_flowers : ℕ := first_plant + second_plant + third_plant

theorem mario_garden_flowers : total_flowers = 22 := by
  sorry

end NUMINAMATH_CALUDE_mario_garden_flowers_l586_58655


namespace NUMINAMATH_CALUDE_average_monthly_production_l586_58600

/-- Calculates the average monthly salt production for a year given the initial production and monthly increase. -/
theorem average_monthly_production
  (initial_production : ℕ)
  (monthly_increase : ℕ)
  (months : ℕ)
  (h1 : initial_production = 1000)
  (h2 : monthly_increase = 100)
  (h3 : months = 12) :
  (initial_production + (initial_production + monthly_increase * (months - 1)) * (months - 1) / 2) / months = 9800 / 12 := by
  sorry

end NUMINAMATH_CALUDE_average_monthly_production_l586_58600


namespace NUMINAMATH_CALUDE_oldest_child_age_l586_58617

theorem oldest_child_age (age1 age2 : ℕ) (avg : ℚ) :
  age1 = 6 →
  age2 = 9 →
  avg = 10 →
  (age1 + age2 + (3 * avg - age1 - age2 : ℚ) : ℚ) / 3 = avg →
  3 * avg - age1 - age2 = 15 :=
by sorry

end NUMINAMATH_CALUDE_oldest_child_age_l586_58617


namespace NUMINAMATH_CALUDE_deepak_age_l586_58637

theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 2 = 26 →
  deepak_age = 18 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l586_58637


namespace NUMINAMATH_CALUDE_sqrt_product_difference_of_squares_l586_58613

-- Problem 1
theorem sqrt_product : Real.sqrt 2 * Real.sqrt 5 = Real.sqrt 10 := by sorry

-- Problem 2
theorem difference_of_squares : (3 + Real.sqrt 6) * (3 - Real.sqrt 6) = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_product_difference_of_squares_l586_58613


namespace NUMINAMATH_CALUDE_complex_number_properties_l586_58649

theorem complex_number_properties (i : ℂ) (h : i^2 = -1) :
  let z₁ : ℂ := 2 / (-1 + i)
  z₁^4 = -4 ∧ Complex.abs z₁ = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_number_properties_l586_58649


namespace NUMINAMATH_CALUDE_orange_apple_ratio_l586_58630

/-- Represents the contents of a shopping cart with apples, oranges, and pears. -/
structure ShoppingCart where
  apples : ℕ
  oranges : ℕ
  pears : ℕ

/-- Checks if the shopping cart satisfies the given conditions. -/
def satisfiesConditions (cart : ShoppingCart) : Prop :=
  cart.pears = 4 * cart.oranges ∧
  cart.apples = (1 / 12 : ℚ) * cart.pears

/-- The main theorem stating the relationship between oranges and apples. -/
theorem orange_apple_ratio (cart : ShoppingCart) 
  (h : satisfiesConditions cart) (h_nonzero : cart.apples > 0) : 
  cart.oranges = 3 * cart.apples := by
  sorry


end NUMINAMATH_CALUDE_orange_apple_ratio_l586_58630


namespace NUMINAMATH_CALUDE_no_parallel_axes_in_bounded_figures_parallel_axes_in_unbounded_figures_intersecting_axes_in_all_figures_l586_58632

-- Define a spatial geometric figure
structure SpatialFigure where
  isBounded : Bool

-- Define an axis of symmetry
structure SymmetryAxis where
  figure : SpatialFigure

-- Define a relation for parallel axes
def areParallel (a1 a2 : SymmetryAxis) : Prop :=
  sorry

-- Define a relation for intersecting axes
def areIntersecting (a1 a2 : SymmetryAxis) : Prop :=
  sorry

-- Theorem 1: Bounded figures cannot have parallel axes of symmetry
theorem no_parallel_axes_in_bounded_figures (f : SpatialFigure) (h : f.isBounded) :
  ¬∃ (a1 a2 : SymmetryAxis), a1.figure = f ∧ a2.figure = f ∧ areParallel a1 a2 :=
sorry

-- Theorem 2: Unbounded figures can have parallel axes of symmetry
theorem parallel_axes_in_unbounded_figures :
  ∃ (f : SpatialFigure) (a1 a2 : SymmetryAxis), 
    ¬f.isBounded ∧ a1.figure = f ∧ a2.figure = f ∧ areParallel a1 a2 :=
sorry

-- Theorem 3: Both bounded and unbounded figures can have intersecting axes of symmetry
theorem intersecting_axes_in_all_figures :
  (∃ (f : SpatialFigure) (a1 a2 : SymmetryAxis), 
    f.isBounded ∧ a1.figure = f ∧ a2.figure = f ∧ areIntersecting a1 a2) ∧
  (∃ (f : SpatialFigure) (a1 a2 : SymmetryAxis), 
    ¬f.isBounded ∧ a1.figure = f ∧ a2.figure = f ∧ areIntersecting a1 a2) :=
sorry

end NUMINAMATH_CALUDE_no_parallel_axes_in_bounded_figures_parallel_axes_in_unbounded_figures_intersecting_axes_in_all_figures_l586_58632


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l586_58608

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific arithmetic sequence from the problem
def specific_sequence (a : ℕ → ℝ) : Prop :=
  is_arithmetic_sequence a ∧ a 3 = 7 ∧ a 7 = 3

-- Theorem statement
theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) (h : specific_sequence a) : 
  ∀ n : ℕ, a n = -n + 10 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l586_58608


namespace NUMINAMATH_CALUDE_interest_rate_equation_l586_58675

/-- Given the following conditions:
  - Manoj borrowed Rs. 3900 from Anwar
  - The loan is for 3 years
  - Manoj lent Rs. 5655 to Ramu for 3 years at 9% p.a. simple interest
  - Manoj gains Rs. 824.85 from the whole transaction
Prove that the interest rate r at which Manoj borrowed from Anwar satisfies the equation:
5655 * 0.09 * 3 - 3900 * (r / 100) * 3 = 824.85 -/
theorem interest_rate_equation (borrowed : ℝ) (lent : ℝ) (duration : ℝ) (ramu_rate : ℝ) (gain : ℝ) (r : ℝ) 
    (h1 : borrowed = 3900)
    (h2 : lent = 5655)
    (h3 : duration = 3)
    (h4 : ramu_rate = 0.09)
    (h5 : gain = 824.85) :
  lent * ramu_rate * duration - borrowed * (r / 100) * duration = gain := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_equation_l586_58675


namespace NUMINAMATH_CALUDE_inverse_function_point_and_sum_l586_58618

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem inverse_function_point_and_sum :
  (f 2 = 9) →  -- This captures the condition that (2,3) is on y = f(x)/3
  (∃ (x : ℝ), f x = 9 ∧ f⁻¹ 9 = 2) ∧  -- This states that (9, 2/3) is on y = f^(-1)(x)/3
  (9 + 2/3 = 29/3) :=  -- This is the sum of coordinates
by sorry

end NUMINAMATH_CALUDE_inverse_function_point_and_sum_l586_58618


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l586_58698

theorem geometric_progression_ratio (a : ℝ) (r : ℝ) :
  a > 0 ∧ r > 0 ∧ 
  (∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)) →
  r = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l586_58698


namespace NUMINAMATH_CALUDE_optimal_price_and_profit_l586_58694

/-- Represents the daily sales volume as a function of price -/
def sales_volume (x : ℝ) : ℝ := -20 * x + 1600

/-- Represents the daily profit as a function of price -/
def daily_profit (x : ℝ) : ℝ := (x - 40) * (sales_volume x)

/-- The selling price must not be less than 45 yuan -/
def min_price : ℝ := 45

/-- Theorem stating the optimal price and maximum profit -/
theorem optimal_price_and_profit :
  ∃ (x : ℝ), x ≥ min_price ∧
  (∀ y : ℝ, y ≥ min_price → daily_profit y ≤ daily_profit x) ∧
  x = 60 ∧ daily_profit x = 8000 := by
  sorry

#check optimal_price_and_profit

end NUMINAMATH_CALUDE_optimal_price_and_profit_l586_58694


namespace NUMINAMATH_CALUDE_hcf_of_36_and_84_l586_58652

theorem hcf_of_36_and_84 : Nat.gcd 36 84 = 12 := by
  sorry

end NUMINAMATH_CALUDE_hcf_of_36_and_84_l586_58652


namespace NUMINAMATH_CALUDE_contest_score_difference_l586_58605

def score_65_percent : ℝ := 0.15
def score_85_percent : ℝ := 0.20
def score_95_percent : ℝ := 0.40
def score_110_percent : ℝ := 1 - (score_65_percent + score_85_percent + score_95_percent)

def score_65 : ℝ := 65
def score_85 : ℝ := 85
def score_95 : ℝ := 95
def score_110 : ℝ := 110

def mean_score : ℝ := 
  score_65_percent * score_65 + 
  score_85_percent * score_85 + 
  score_95_percent * score_95 + 
  score_110_percent * score_110

def median_score : ℝ := score_95

theorem contest_score_difference : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.25 ∧ |median_score - mean_score - 3| < ε :=
sorry

end NUMINAMATH_CALUDE_contest_score_difference_l586_58605


namespace NUMINAMATH_CALUDE_grey_eyed_black_hair_count_l586_58689

/-- Represents the number of students with a specific hair color and eye color combination -/
structure StudentCount where
  redHairGreenEyes : ℕ
  redHairGreyEyes : ℕ
  blackHairGreenEyes : ℕ
  blackHairGreyEyes : ℕ

/-- Theorem stating the number of grey-eyed students with black hair -/
theorem grey_eyed_black_hair_count (s : StudentCount) : s.blackHairGreyEyes = 20 :=
  by
  have total_students : s.redHairGreenEyes + s.redHairGreyEyes + s.blackHairGreenEyes + s.blackHairGreyEyes = 60 := by sorry
  have green_eyed_red_hair : s.redHairGreenEyes = 20 := by sorry
  have black_hair_total : s.blackHairGreenEyes + s.blackHairGreyEyes = 36 := by sorry
  have grey_eyes_total : s.redHairGreyEyes + s.blackHairGreyEyes = 24 := by sorry
  sorry

#check grey_eyed_black_hair_count

end NUMINAMATH_CALUDE_grey_eyed_black_hair_count_l586_58689


namespace NUMINAMATH_CALUDE_bunny_burrow_exits_l586_58636

-- Define the rate at which a bunny comes out of its burrow
def bunny_rate : ℕ := 3

-- Define the number of bunnies
def num_bunnies : ℕ := 20

-- Define the time period in hours
def time_period : ℕ := 10

-- Define the number of minutes in an hour
def minutes_per_hour : ℕ := 60

-- Theorem statement
theorem bunny_burrow_exits :
  bunny_rate * minutes_per_hour * time_period * num_bunnies = 36000 := by
  sorry

end NUMINAMATH_CALUDE_bunny_burrow_exits_l586_58636


namespace NUMINAMATH_CALUDE_area_of_XYZ_main_theorem_l586_58664

/-- Triangle ABC with given side lengths --/
structure Triangle :=
  (AB BC CA : ℝ)

/-- Points on the triangle --/
structure TrianglePoints :=
  (A B C D P Q X Y Z : ℝ × ℝ)

/-- Given triangle ABC with altitude AD and inscribed circle tangent points --/
def given_triangle : Triangle := { AB := 13, BC := 14, CA := 15 }

/-- Theorem: Area of triangle XYZ is 25/4 --/
theorem area_of_XYZ (t : Triangle) (tp : TrianglePoints) : ℝ :=
  let triangle := given_triangle
  25 / 4

/-- Main theorem --/
theorem main_theorem (t : Triangle) (tp : TrianglePoints) : 
  t = given_triangle → 
  tp.D.1 = tp.B.1 ∨ tp.D.1 = tp.C.1 →  -- D is on BC
  (tp.A.1 - tp.D.1)^2 + (tp.A.2 - tp.D.2)^2 = (tp.B.1 - tp.D.1)^2 + (tp.B.2 - tp.D.2)^2 →  -- AD perpendicular to BC
  (tp.P.1 - tp.A.1) * (tp.D.1 - tp.A.1) + (tp.P.2 - tp.A.2) * (tp.D.2 - tp.A.2) = 0 →  -- P on AD
  (tp.Q.1 - tp.A.1) * (tp.D.1 - tp.A.1) + (tp.Q.2 - tp.A.2) * (tp.D.2 - tp.A.2) = 0 →  -- Q on AD
  tp.X.1 = tp.B.1 ∨ tp.X.1 = tp.C.1 →  -- X on BC
  tp.Y.1 = tp.B.1 ∨ tp.Y.1 = tp.C.1 →  -- Y on BC
  (tp.Z.1 - tp.P.1) * (tp.X.1 - tp.P.1) + (tp.Z.2 - tp.P.2) * (tp.X.2 - tp.P.2) = 0 →  -- Z on PX
  (tp.Z.1 - tp.Q.1) * (tp.Y.1 - tp.Q.1) + (tp.Z.2 - tp.Q.2) * (tp.Y.2 - tp.Q.2) = 0 →  -- Z on QY
  area_of_XYZ t tp = 25 / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_of_XYZ_main_theorem_l586_58664


namespace NUMINAMATH_CALUDE_special_isosceles_triangle_base_length_l586_58696

/-- An isosceles triangle with side length a and a specific height property -/
structure SpecialIsoscelesTriangle (a : ℝ) where
  -- The triangle is isosceles with side length a
  side_length : ℝ
  is_isosceles : side_length = a
  -- The height dropped onto the base is equal to the segment connecting
  -- the midpoint of the base with the midpoint of the side
  height_property : ℝ → Prop

/-- The base length of the special isosceles triangle is a√3 -/
theorem special_isosceles_triangle_base_length 
  {a : ℝ} (t : SpecialIsoscelesTriangle a) : 
  ∃ (base : ℝ), base = a * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_special_isosceles_triangle_base_length_l586_58696


namespace NUMINAMATH_CALUDE_complex_product_one_plus_i_one_minus_i_l586_58621

theorem complex_product_one_plus_i_one_minus_i : (1 + Complex.I) * (1 - Complex.I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_one_plus_i_one_minus_i_l586_58621


namespace NUMINAMATH_CALUDE_mary_max_earnings_l586_58666

/-- Calculates the maximum weekly earnings for a worker under specific pay conditions -/
def maxWeeklyEarnings (maxHours : ℕ) (regularRate : ℚ) (overtime1Multiplier : ℚ) (overtime2Multiplier : ℚ) : ℚ :=
  let regularHours := min maxHours 40
  let overtime1Hours := min (maxHours - regularHours) 20
  let overtime2Hours := maxHours - regularHours - overtime1Hours
  let regularPay := regularRate * regularHours
  let overtime1Pay := regularRate * overtime1Multiplier * overtime1Hours
  let overtime2Pay := regularRate * overtime2Multiplier * overtime2Hours
  regularPay + overtime1Pay + overtime2Pay

theorem mary_max_earnings :
  maxWeeklyEarnings 80 15 1.6 2 = 1680 := by
  sorry

#eval maxWeeklyEarnings 80 15 1.6 2

end NUMINAMATH_CALUDE_mary_max_earnings_l586_58666


namespace NUMINAMATH_CALUDE_election_winner_votes_l586_58634

theorem election_winner_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) : 
  winner_percentage = 62 / 100 →
  vote_difference = 324 →
  (winner_percentage * total_votes).num = (winner_percentage * total_votes).den * 837 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_votes_l586_58634


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l586_58693

theorem sufficient_not_necessary_condition (x y : ℝ) : 
  (∀ y, x = 0 → x * y = 0) ∧ (∃ x y, x * y = 0 ∧ x ≠ 0) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l586_58693


namespace NUMINAMATH_CALUDE_f_difference_l586_58640

/-- The function f(x) = x^4 + 2x^3 + 3x^2 + 7x -/
def f (x : ℝ) : ℝ := x^4 + 2*x^3 + 3*x^2 + 7*x

/-- Theorem: f(3) - f(-3) = 150 -/
theorem f_difference : f 3 - f (-3) = 150 := by sorry

end NUMINAMATH_CALUDE_f_difference_l586_58640


namespace NUMINAMATH_CALUDE_sequence_sum_l586_58641

theorem sequence_sum : 
  let seq := [3, 15, 27, 53, 65, 17, 29, 41, 71, 83]
  List.sum seq = 404 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l586_58641


namespace NUMINAMATH_CALUDE_roots_of_quadratic_sum_of_eighth_powers_l586_58633

theorem roots_of_quadratic_sum_of_eighth_powers (a b : ℂ) : 
  (a^2 - 2*a + 5 = 0) → (b^2 - 2*b + 5 = 0) → Complex.abs (a^8 + b^8) = 1054 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_sum_of_eighth_powers_l586_58633


namespace NUMINAMATH_CALUDE_count_multiples_of_seven_between_squares_l586_58610

theorem count_multiples_of_seven_between_squares : 
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, n % 7 = 0 ∧ (17 : ℝ) < Real.sqrt n ∧ Real.sqrt n < 18) ∧
    (∀ n : ℕ, n % 7 = 0 ∧ (17 : ℝ) < Real.sqrt n ∧ Real.sqrt n < 18 → n ∈ s) ∧
    Finset.card s = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_of_seven_between_squares_l586_58610


namespace NUMINAMATH_CALUDE_clock_angle_at_8_l586_58619

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees per hour on a clock face -/
def degrees_per_hour : ℕ := 360 / clock_hours

/-- The position of the minute hand at 8:00 in degrees -/
def minute_hand_position : ℕ := 0

/-- The position of the hour hand at 8:00 in degrees -/
def hour_hand_position : ℕ := 8 * degrees_per_hour

/-- The smaller angle between the hour and minute hands at 8:00 -/
def smaller_angle : ℕ := min (hour_hand_position - minute_hand_position) (360 - (hour_hand_position - minute_hand_position))

theorem clock_angle_at_8 : smaller_angle = 120 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_8_l586_58619


namespace NUMINAMATH_CALUDE_rain_probability_three_days_l586_58615

def prob_rain_friday : ℝ := 0.4
def prob_rain_saturday : ℝ := 0.7
def prob_rain_sunday : ℝ := 0.3

theorem rain_probability_three_days :
  let prob_all_days := prob_rain_friday * prob_rain_saturday * prob_rain_sunday
  prob_all_days = 0.084 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_three_days_l586_58615


namespace NUMINAMATH_CALUDE_candidate_B_votes_l586_58674

/-- Represents a candidate in the election --/
inductive Candidate
  | A | B | C | D | E

/-- The total number of people in the class --/
def totalVotes : Nat := 46

/-- The number of votes received by candidate A --/
def votesForA : Nat := 25

/-- The number of votes received by candidate E --/
def votesForE : Nat := 4

/-- The voting results satisfy the given conditions --/
def validVotingResult (votes : Candidate → Nat) : Prop :=
  votes Candidate.A = votesForA ∧
  votes Candidate.E = votesForE ∧
  votes Candidate.B > votes Candidate.E ∧
  votes Candidate.B < votes Candidate.A ∧
  votes Candidate.C = votes Candidate.D ∧
  votes Candidate.A + votes Candidate.B + votes Candidate.C + votes Candidate.D + votes Candidate.E = totalVotes

theorem candidate_B_votes (votes : Candidate → Nat) 
  (h : validVotingResult votes) : votes Candidate.B = 7 := by
  sorry

end NUMINAMATH_CALUDE_candidate_B_votes_l586_58674


namespace NUMINAMATH_CALUDE_coopers_fence_bricks_l586_58665

/-- Calculates the number of bricks needed for a fence with given dimensions. -/
def bricks_needed (num_walls length height depth : ℕ) : ℕ :=
  num_walls * length * height * depth

/-- Theorem stating the number of bricks needed for Cooper's fence. -/
theorem coopers_fence_bricks : 
  bricks_needed 4 20 5 2 = 800 := by
  sorry

end NUMINAMATH_CALUDE_coopers_fence_bricks_l586_58665


namespace NUMINAMATH_CALUDE_hyperbola_equation_l586_58670

theorem hyperbola_equation (ellipse : Real → Real → Prop)
  (hyperbola : Real → Real → Prop)
  (h1 : ∀ x y, ellipse x y ↔ x^2/27 + y^2/36 = 1)
  (h2 : ∃ x, hyperbola x 4 ∧ ellipse x 4)
  (h3 : ∀ x y, hyperbola x y → (x = 0 → y^2 = 9) ∧ (y = 0 → x^2 = 9)) :
  ∀ x y, hyperbola x y ↔ y^2/4 - x^2/5 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l586_58670


namespace NUMINAMATH_CALUDE_unique_complex_pair_l586_58695

theorem unique_complex_pair : 
  ∃! (a b : ℂ), (a^4 * b^3 = 1) ∧ (a^6 * b^7 = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_complex_pair_l586_58695


namespace NUMINAMATH_CALUDE_top_square_is_14_l586_58622

/-- Represents a position in the 4x4 grid -/
structure Position :=
  (row : Fin 4)
  (col : Fin 4)

/-- Represents the 4x4 grid -/
def Grid := Fin 4 → Fin 4 → ℕ

/-- Initial configuration of the grid -/
def initialGrid : Grid :=
  λ i j => i.val * 4 + j.val + 1

/-- Fold right half over left half -/
def foldRight (g : Grid) : Grid :=
  λ i j => g i (3 - j)

/-- Fold bottom half over top half -/
def foldBottom (g : Grid) : Grid :=
  λ i j => g (3 - i) j

/-- Apply all folding operations -/
def applyFolds (g : Grid) : Grid :=
  foldRight (foldBottom (foldRight g))

/-- The position of the top square after folding -/
def topPosition : Position :=
  ⟨0, 0⟩

/-- Theorem: After folding, the top square was originally numbered 14 -/
theorem top_square_is_14 :
  applyFolds initialGrid topPosition.row topPosition.col = 14 := by
  sorry

end NUMINAMATH_CALUDE_top_square_is_14_l586_58622


namespace NUMINAMATH_CALUDE_cow_count_is_seven_l586_58680

/-- Represents the number of animals in the group -/
structure AnimalCount where
  cows : ℕ
  chickens : ℕ

/-- The total number of legs in the group -/
def totalLegs (ac : AnimalCount) : ℕ := 4 * ac.cows + 2 * ac.chickens

/-- The total number of heads in the group -/
def totalHeads (ac : AnimalCount) : ℕ := ac.cows + ac.chickens

/-- The main theorem stating that if the number of legs is 14 more than twice the number of heads,
    then the number of cows is 7 -/
theorem cow_count_is_seven (ac : AnimalCount) :
  totalLegs ac = 2 * totalHeads ac + 14 → ac.cows = 7 := by
  sorry


end NUMINAMATH_CALUDE_cow_count_is_seven_l586_58680


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l586_58609

/-- A trinomial ax^2 + bx + c is a perfect square if there exists a real number k such that
    ax^2 + bx + c = (x + k)^2 for all x. -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (x + k)^2

theorem perfect_square_trinomial_m_value :
  ∀ m : ℝ, IsPerfectSquareTrinomial 1 m 9 → m = 6 ∨ m = -6 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l586_58609


namespace NUMINAMATH_CALUDE_sqrt_two_sqrt_three_minus_three_equals_fourth_root_difference_l586_58651

theorem sqrt_two_sqrt_three_minus_three_equals_fourth_root_difference : 
  Real.sqrt (2 * Real.sqrt 3 - 3) = (27/4)^(1/4) - (3/4)^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_sqrt_three_minus_three_equals_fourth_root_difference_l586_58651


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l586_58629

theorem square_plus_reciprocal_square (x : ℝ) (hx : x ≠ 0) 
  (h : x + 1/x = Real.sqrt 2019) : x^2 + 1/x^2 = 2017 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l586_58629


namespace NUMINAMATH_CALUDE_logarithmic_function_properties_l586_58669

-- Define the logarithmic function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Theorem statement
theorem logarithmic_function_properties :
  -- f(x) passes through (8,3)
  f 8 = 3 →
  -- f(x) is a logarithmic function (this is implied by its definition)
  -- Prove the following:
  (-- 1. f(x) = log₂(x) (this is true by definition of f)
   -- 2. The domain of f(x) is (0, +∞)
   (∀ x : ℝ, x > 0 ↔ f x ≠ 0) ∧
   -- 3. For f(1-x) > f(1+x), x ∈ (-1, 0)
   (∀ x : ℝ, f (1 - x) > f (1 + x) ↔ -1 < x ∧ x < 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_logarithmic_function_properties_l586_58669


namespace NUMINAMATH_CALUDE_intersection_complement_A_with_B_l586_58684

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 4)}

-- Define set B
def B : Set ℝ := {x | -1 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 0}

-- Define the complement of A in real numbers
def complementA : Set ℝ := {x | x ∉ A}

-- Theorem statement
theorem intersection_complement_A_with_B : 
  complementA ∩ B = Set.Icc 0 (1/2) := by sorry

end NUMINAMATH_CALUDE_intersection_complement_A_with_B_l586_58684


namespace NUMINAMATH_CALUDE_train_crossing_time_l586_58628

/-- The time taken for a train to cross a man running in the same direction --/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 450 →
  train_speed = 60 * (1000 / 3600) →
  man_speed = 6 * (1000 / 3600) →
  (train_length / (train_speed - man_speed)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l586_58628


namespace NUMINAMATH_CALUDE_delegation_selection_ways_l586_58668

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of men in the brigade -/
def num_men : ℕ := 10

/-- The number of women in the brigade -/
def num_women : ℕ := 8

/-- The number of men to be selected for the delegation -/
def men_in_delegation : ℕ := 3

/-- The number of women to be selected for the delegation -/
def women_in_delegation : ℕ := 2

/-- The theorem stating the number of ways to select the delegation -/
theorem delegation_selection_ways :
  (choose num_men men_in_delegation) * (choose num_women women_in_delegation) = 3360 := by
  sorry

end NUMINAMATH_CALUDE_delegation_selection_ways_l586_58668


namespace NUMINAMATH_CALUDE_conceived_number_is_seven_l586_58627

theorem conceived_number_is_seven :
  ∃! (x : ℕ+), (10 * x.val + 7 - x.val ^ 2) / 4 - x.val = 0 :=
by sorry

end NUMINAMATH_CALUDE_conceived_number_is_seven_l586_58627


namespace NUMINAMATH_CALUDE_waiter_customer_count_l586_58699

theorem waiter_customer_count (initial : Float) (lunch_rush : Float) (later : Float) :
  initial = 29.0 →
  lunch_rush = 20.0 →
  later = 34.0 →
  initial + lunch_rush + later = 83.0 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customer_count_l586_58699


namespace NUMINAMATH_CALUDE_fencing_cost_is_1950_l586_58642

/-- A rectangular plot with specific dimensions and fencing cost. -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  fencing_rate : ℝ
  length_width_relation : length = width + 10
  perimeter_constraint : perimeter = 2 * (length + width)
  perimeter_value : perimeter = 300
  fencing_rate_value : fencing_rate = 6.5

/-- The cost of fencing the rectangular plot. -/
def fencing_cost (plot : RectangularPlot) : ℝ :=
  plot.perimeter * plot.fencing_rate

/-- Theorem stating the fencing cost for the given rectangular plot. -/
theorem fencing_cost_is_1950 (plot : RectangularPlot) : fencing_cost plot = 1950 := by
  sorry


end NUMINAMATH_CALUDE_fencing_cost_is_1950_l586_58642


namespace NUMINAMATH_CALUDE_fermat_500_units_digit_l586_58645

def fermat (n : ℕ) : ℕ := 2^(2^n) + 1

theorem fermat_500_units_digit :
  fermat 500 % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_fermat_500_units_digit_l586_58645


namespace NUMINAMATH_CALUDE_inequality_for_positive_integers_l586_58631

theorem inequality_for_positive_integers (n : ℕ) (h : n > 0) : 2 * n - 1 < (n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_for_positive_integers_l586_58631


namespace NUMINAMATH_CALUDE_inequality_holds_iff_l586_58690

theorem inequality_holds_iff (a : ℝ) :
  (∀ x ∈ Set.Icc 2 3, a * x^2 - (a + 2) * x + 2 < 0) ↔ a < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_l586_58690


namespace NUMINAMATH_CALUDE_y_value_theorem_l586_58659

theorem y_value_theorem (y : ℝ) :
  (y / 5) / 3 = 5 / (y / 3) → y = 15 ∨ y = -15 := by
  sorry

end NUMINAMATH_CALUDE_y_value_theorem_l586_58659


namespace NUMINAMATH_CALUDE_bookstore_sales_ratio_l586_58602

theorem bookstore_sales_ratio :
  let tuesday_sales : ℕ := 7
  let wednesday_sales : ℕ := 3 * tuesday_sales
  let total_sales : ℕ := 91
  let thursday_sales : ℕ := total_sales - (tuesday_sales + wednesday_sales)
  (thursday_sales : ℚ) / wednesday_sales = 3 / 1 :=
by sorry

end NUMINAMATH_CALUDE_bookstore_sales_ratio_l586_58602


namespace NUMINAMATH_CALUDE_pet_shop_birds_l586_58656

theorem pet_shop_birds (total : ℕ) (kittens : ℕ) (hamsters : ℕ) (birds : ℕ) : 
  total = 77 → kittens = 32 → hamsters = 15 → birds = total - kittens - hamsters → birds = 30 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_birds_l586_58656


namespace NUMINAMATH_CALUDE_calculate_expression_l586_58661

theorem calculate_expression : 8 * (5 + 2/5) - 3 = 40.2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l586_58661


namespace NUMINAMATH_CALUDE_max_sum_abc_l586_58607

theorem max_sum_abc (a b c : ℤ) 
  (h1 : a + b = 2006) 
  (h2 : c - a = 2005) 
  (h3 : a < b) : 
  ∃ (m : ℤ), m = 5013 ∧ a + b + c ≤ m ∧ ∃ (a' b' c' : ℤ), a' + b' = 2006 ∧ c' - a' = 2005 ∧ a' < b' ∧ a' + b' + c' = m :=
sorry

end NUMINAMATH_CALUDE_max_sum_abc_l586_58607


namespace NUMINAMATH_CALUDE_triangle_inequality_l586_58639

/-- Given three line segments of lengths a, 2, and 6, they can form a triangle if and only if 4 < a < 8 -/
theorem triangle_inequality (a : ℝ) : 
  (∃ (x y z : ℝ), x = a ∧ y = 2 ∧ z = 6 ∧ x + y > z ∧ y + z > x ∧ z + x > y) ↔ 4 < a ∧ a < 8 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l586_58639


namespace NUMINAMATH_CALUDE_vertical_distance_to_charlie_l586_58614

/-- The vertical distance between the midpoint of the line segment connecting
    (8, -15) and (2, 10), and the point (5, 3) is 5.5 units. -/
theorem vertical_distance_to_charlie : 
  let annie : ℝ × ℝ := (8, -15)
  let barbara : ℝ × ℝ := (2, 10)
  let charlie : ℝ × ℝ := (5, 3)
  let midpoint : ℝ × ℝ := ((annie.1 + barbara.1) / 2, (annie.2 + barbara.2) / 2)
  charlie.2 - midpoint.2 = 5.5 := by sorry

end NUMINAMATH_CALUDE_vertical_distance_to_charlie_l586_58614


namespace NUMINAMATH_CALUDE_q_must_be_false_l586_58681

theorem q_must_be_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q := by
  sorry

end NUMINAMATH_CALUDE_q_must_be_false_l586_58681


namespace NUMINAMATH_CALUDE_larger_number_problem_l586_58691

theorem larger_number_problem (a b : ℝ) (h1 : a > b) (h2 : (a + b) + (a - b) = 68) : a = 34 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l586_58691


namespace NUMINAMATH_CALUDE_cyclic_sum_divisibility_l586_58672

theorem cyclic_sum_divisibility (x y z : ℤ) (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) :
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (x - y) * (y - z) * (z - x)) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_divisibility_l586_58672


namespace NUMINAMATH_CALUDE_workers_per_team_lead_is_ten_l586_58616

/-- Represents the hierarchical structure of a company -/
structure CompanyStructure where
  supervisors : ℕ
  workers : ℕ
  team_leads_per_supervisor : ℕ
  workers_per_team_lead : ℕ

/-- Calculates the number of workers per team lead given a company structure -/
def calculate_workers_per_team_lead (c : CompanyStructure) : ℕ :=
  c.workers / (c.supervisors * c.team_leads_per_supervisor)

/-- Theorem stating that for the given company structure, there are 10 workers per team lead -/
theorem workers_per_team_lead_is_ten :
  let c := CompanyStructure.mk 13 390 3 10
  calculate_workers_per_team_lead c = 10 := by
  sorry


end NUMINAMATH_CALUDE_workers_per_team_lead_is_ten_l586_58616


namespace NUMINAMATH_CALUDE_a_3_value_l586_58620

def S (n : ℕ+) : ℕ := 5 * n.val ^ 2 + 10 * n.val

theorem a_3_value : ∃ (a : ℕ+ → ℕ), a 3 = 35 :=
  sorry

end NUMINAMATH_CALUDE_a_3_value_l586_58620


namespace NUMINAMATH_CALUDE_largest_c_value_l586_58643

theorem largest_c_value (c : ℝ) : 
  (3 * c + 4) * (c - 2) = 7 * c →
  c ≤ (9 + Real.sqrt 177) / 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_c_value_l586_58643


namespace NUMINAMATH_CALUDE_no_tie_in_total_hr_l586_58687

/-- Represents the months of the baseball season -/
inductive Month
| Mar
| Apr
| May
| Jun
| Jul
| Aug
| Sep

/-- Returns the number of home runs hit by Johnson in a given month -/
def johnson_hr (m : Month) : ℕ :=
  match m with
  | Month.Mar => 2
  | Month.Apr => 11
  | Month.May => 15
  | Month.Jun => 9
  | Month.Jul => 7
  | Month.Aug => 12
  | Month.Sep => 14

/-- Returns the number of home runs hit by Carter in a given month -/
def carter_hr (m : Month) : ℕ :=
  match m with
  | Month.Mar => 0
  | Month.Apr => 5
  | Month.May => 8
  | Month.Jun => 18
  | Month.Jul => 6
  | Month.Aug => 15
  | Month.Sep => 10

/-- Calculates the cumulative home runs for a player up to and including a given month -/
def cumulative_hr (hr_func : Month → ℕ) (m : Month) : ℕ :=
  match m with
  | Month.Mar => hr_func Month.Mar
  | Month.Apr => hr_func Month.Mar + hr_func Month.Apr
  | Month.May => hr_func Month.Mar + hr_func Month.Apr + hr_func Month.May
  | Month.Jun => hr_func Month.Mar + hr_func Month.Apr + hr_func Month.May + hr_func Month.Jun
  | Month.Jul => hr_func Month.Mar + hr_func Month.Apr + hr_func Month.May + hr_func Month.Jun + hr_func Month.Jul
  | Month.Aug => hr_func Month.Mar + hr_func Month.Apr + hr_func Month.May + hr_func Month.Jun + hr_func Month.Jul + hr_func Month.Aug
  | Month.Sep => hr_func Month.Mar + hr_func Month.Apr + hr_func Month.May + hr_func Month.Jun + hr_func Month.Jul + hr_func Month.Aug + hr_func Month.Sep

theorem no_tie_in_total_hr : ∀ m : Month, cumulative_hr johnson_hr m ≠ cumulative_hr carter_hr m := by
  sorry

end NUMINAMATH_CALUDE_no_tie_in_total_hr_l586_58687


namespace NUMINAMATH_CALUDE_problem_solution_l586_58654

theorem problem_solution (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + 2 = d + (a + b + c - d)^(1/3)) : 
  d = 1/2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l586_58654


namespace NUMINAMATH_CALUDE_characterize_equal_prime_factors_l586_58601

/-- The set of prime factors of a positive integer n -/
def primeDivisors (n : ℕ) : Set ℕ := sorry

theorem characterize_equal_prime_factors :
  ∀ (a m n : ℕ),
    a > 1 →
    m < n →
    (primeDivisors (a^m - 1) = primeDivisors (a^n - 1)) ↔
    (∃ l : ℕ, l ≥ 2 ∧ a = 2^l - 1 ∧ m = 1 ∧ n = 2) :=
by sorry

end NUMINAMATH_CALUDE_characterize_equal_prime_factors_l586_58601


namespace NUMINAMATH_CALUDE_triangle_angle_C_l586_58682

open Real

theorem triangle_angle_C (A B C : ℝ) (a b c : ℝ) : 
  A = π/3 → a = 3 → c = Real.sqrt 6 → 
  (sin C = sin (π/4) ∨ sin C = sin (3*π/4)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l586_58682


namespace NUMINAMATH_CALUDE_oil_percentage_in_mixtureA_l586_58635

/-- Represents the composition of a mixture --/
structure Mixture where
  oil : ℝ
  materialB : ℝ

/-- The original mixture A --/
def mixtureA : Mixture := sorry

/-- The weight of the original mixture A in kilograms --/
def originalWeight : ℝ := 8

/-- The weight of oil added to mixture A in kilograms --/
def addedOil : ℝ := 2

/-- The weight of mixture A added to the new mixture in kilograms --/
def addedMixtureA : ℝ := 6

/-- The percentage of material B in the final mixture --/
def finalMaterialBPercentage : ℝ := 70

/-- Theorem stating that the percentage of oil in the original mixture A is 20% --/
theorem oil_percentage_in_mixtureA : mixtureA.oil / (mixtureA.oil + mixtureA.materialB) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_oil_percentage_in_mixtureA_l586_58635


namespace NUMINAMATH_CALUDE_short_sleeve_shirts_count_l586_58683

theorem short_sleeve_shirts_count :
  ∀ (total_shirts long_sleeve_shirts washed_shirts unwashed_shirts : ℕ),
    long_sleeve_shirts = 47 →
    washed_shirts = 20 →
    unwashed_shirts = 66 →
    total_shirts = washed_shirts + unwashed_shirts →
    total_shirts = (total_shirts - long_sleeve_shirts) + long_sleeve_shirts →
    (total_shirts - long_sleeve_shirts) = 39 :=
by sorry

end NUMINAMATH_CALUDE_short_sleeve_shirts_count_l586_58683


namespace NUMINAMATH_CALUDE_sin_five_pi_six_plus_two_alpha_l586_58603

theorem sin_five_pi_six_plus_two_alpha (α : Real) 
  (h : Real.cos (α + π/6) = 1/3) : 
  Real.sin (5*π/6 + 2*α) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_five_pi_six_plus_two_alpha_l586_58603


namespace NUMINAMATH_CALUDE_intersection_properties_l586_58612

-- Define the line l: ax - y + 2 - 2a = 0
def line_equation (a x y : ℝ) : Prop := a * x - y + 2 - 2 * a = 0

-- Define the circle C: (x - 4)² + (y - 1)² = r²
def circle_equation (x y r : ℝ) : Prop := (x - 4)^2 + (y - 1)^2 = r^2

-- Define the intersection condition
def intersects_at_two_points (a r : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    line_equation a x₁ y₁ ∧ line_equation a x₂ y₂ ∧
    circle_equation x₁ y₁ r ∧ circle_equation x₂ y₂ r

theorem intersection_properties (a r : ℝ) (hr : r > 0) 
  (h_intersect : intersects_at_two_points a r) :
  -- 1. The line passes through (2, 2)
  (line_equation a 2 2) ∧
  -- 2. r > √5
  (r > Real.sqrt 5) ∧
  -- 3. When r = 3, the chord length is between 4 and 6
  (r = 3 → ∃ (l : ℝ), 4 ≤ l ∧ l ≤ 6 ∧
    ∀ (x₁ y₁ x₂ y₂ : ℝ), 
      line_equation a x₁ y₁ ∧ line_equation a x₂ y₂ ∧
      circle_equation x₁ y₁ 3 ∧ circle_equation x₂ y₂ 3 →
      l = Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)) ∧
  -- 4. When r = 5, the minimum dot product is -25
  (r = 5 → ∃ (min_dot_product : ℝ), min_dot_product = -25 ∧
    ∀ (x₁ y₁ x₂ y₂ : ℝ), 
      line_equation a x₁ y₁ ∧ line_equation a x₂ y₂ ∧
      circle_equation x₁ y₁ 5 ∧ circle_equation x₂ y₂ 5 →
      ((x₁ - 4) * (x₂ - 4) + (y₁ - 1) * (y₂ - 1)) ≥ min_dot_product) :=
by sorry

end NUMINAMATH_CALUDE_intersection_properties_l586_58612


namespace NUMINAMATH_CALUDE_fraction_denominator_proof_l586_58673

theorem fraction_denominator_proof (y : ℝ) (x : ℝ) (h1 : y > 0) 
  (h2 : (9 * y) / 20 + (3 * y) / x = 0.75 * y) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_denominator_proof_l586_58673


namespace NUMINAMATH_CALUDE_triangle_side_length_l586_58671

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →  -- acute triangle
  a = 4 →
  b = 5 →
  (1/2) * a * b * Real.sin C = 5 * Real.sqrt 3 →  -- area condition
  c = Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l586_58671


namespace NUMINAMATH_CALUDE_matrix_identity_l586_58647

variable {n : Type*} [DecidableEq n] [Fintype n]

theorem matrix_identity (A : Matrix n n ℝ) (h_inv : IsUnit A) 
  (h_eq : (A - 3 • (1 : Matrix n n ℝ)) * (A - 5 • (1 : Matrix n n ℝ)) = 0) :
  A + 8 • A⁻¹ = (7 • A + 64 • (1 : Matrix n n ℝ)) / 15 := by
  sorry

end NUMINAMATH_CALUDE_matrix_identity_l586_58647


namespace NUMINAMATH_CALUDE_sqrt_three_squared_l586_58650

theorem sqrt_three_squared : (Real.sqrt 3) ^ 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_squared_l586_58650


namespace NUMINAMATH_CALUDE_three_zeros_condition_l586_58638

/-- The cubic function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 2

/-- Theorem stating the condition for f to have exactly 3 real zeros -/
theorem three_zeros_condition (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧
    ∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) ↔ 
  a < -3 :=
sorry

end NUMINAMATH_CALUDE_three_zeros_condition_l586_58638


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_l586_58644

theorem complex_purely_imaginary (a b : ℝ) :
  (∃ (z : ℂ), z = Complex.I * a + b ∧ z.re = 0 ∧ z.im ≠ 0) ↔ (a ≠ 0 ∧ b = 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_l586_58644


namespace NUMINAMATH_CALUDE_min_value_fraction_l586_58625

theorem min_value_fraction (x y : ℝ) (h : (x + 2)^2 + y^2 = 1) :
  ∃ k : ℝ, k = (y - 1) / (x - 2) ∧ k ≥ 0 ∧ ∀ m : ℝ, m = (y - 1) / (x - 2) → m ≥ k :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l586_58625


namespace NUMINAMATH_CALUDE_negation_of_zero_product_l586_58611

theorem negation_of_zero_product (a b : ℝ) :
  ¬(a * b = 0 → a = 0 ∨ b = 0) ↔ (a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_zero_product_l586_58611


namespace NUMINAMATH_CALUDE_power_equation_solution_l586_58663

theorem power_equation_solution (n : ℕ) : 3^n = 3 * 27^3 * 243^2 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l586_58663


namespace NUMINAMATH_CALUDE_number_thought_of_l586_58653

theorem number_thought_of (x : ℝ) : (x / 6 + 5 = 17) → x = 72 := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l586_58653


namespace NUMINAMATH_CALUDE_line_parameterization_l586_58678

/-- Given a line y = 2x - 30 parameterized by (x,y) = (g(t), 12t - 10), 
    prove that g(t) = 6t + 10 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ t : ℝ, 12 * t - 10 = 2 * g t - 30) → 
  (∀ t : ℝ, g t = 6 * t + 10) := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l586_58678


namespace NUMINAMATH_CALUDE_ken_snow_days_l586_58692

/-- Represents the cycling scenario for Ken in a week -/
structure CyclingWeek where
  rain_speed : ℕ  -- miles per 20 minutes when raining
  snow_speed : ℕ  -- miles per 20 minutes when snowing
  rain_days : ℕ   -- number of rainy days
  total_miles : ℕ -- total miles cycled in the week
  hours_per_day : ℕ -- hours cycled per day

/-- Calculates the number of snowy days in a week -/
def snow_days (w : CyclingWeek) : ℕ :=
  ((w.total_miles - w.rain_days * w.rain_speed * 3) / (w.snow_speed * 3))

/-- Theorem stating the number of snowy days in Ken's cycling week -/
theorem ken_snow_days :
  let w : CyclingWeek := {
    rain_speed := 30,
    snow_speed := 10,
    rain_days := 3,
    total_miles := 390,
    hours_per_day := 1
  }
  snow_days w = 4 := by sorry

end NUMINAMATH_CALUDE_ken_snow_days_l586_58692


namespace NUMINAMATH_CALUDE_wolf_sheep_problem_l586_58624

theorem wolf_sheep_problem (x : ℕ) : 
  (∃ y : ℕ, y = 3 * x + 2 ∧ y = 8 * x - 8) → x = 2 :=
by sorry

end NUMINAMATH_CALUDE_wolf_sheep_problem_l586_58624


namespace NUMINAMATH_CALUDE_cos_pi_minus_alpha_l586_58677

theorem cos_pi_minus_alpha (α : Real) (h : Real.sin (α / 2) = 2 / 3) : 
  Real.cos (Real.pi - α) = -1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_minus_alpha_l586_58677


namespace NUMINAMATH_CALUDE_bryce_raisins_l586_58686

theorem bryce_raisins (bryce carter : ℕ) : 
  bryce = carter + 8 →
  carter = bryce / 3 →
  bryce + carter = 44 →
  bryce = 33 := by
sorry

end NUMINAMATH_CALUDE_bryce_raisins_l586_58686


namespace NUMINAMATH_CALUDE_difference_zero_for_sqrt_three_l586_58648

-- Define the custom operation
def custom_op (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- State the theorem
theorem difference_zero_for_sqrt_three :
  let x : ℝ := Real.sqrt 3
  let y : ℝ := Real.sqrt 3
  x - y = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_difference_zero_for_sqrt_three_l586_58648


namespace NUMINAMATH_CALUDE_income_growth_and_projection_l586_58676

/-- Represents the annual growth rate as a real number between 0 and 1 -/
def AnnualGrowthRate := { r : ℝ // 0 < r ∧ r < 1 }

/-- Calculates the future value given initial value, growth rate, and number of years -/
def futureValue (initialValue : ℝ) (rate : AnnualGrowthRate) (years : ℕ) : ℝ :=
  initialValue * (1 + rate.val) ^ years

theorem income_growth_and_projection (initialIncome : ℝ) (finalIncome : ℝ) (years : ℕ) 
  (h1 : initialIncome = 2500)
  (h2 : finalIncome = 3600)
  (h3 : years = 2) :
  ∃ (rate : AnnualGrowthRate),
    (futureValue initialIncome rate years = finalIncome) ∧ 
    (rate.val = 0.2) ∧
    (futureValue finalIncome rate 1 > 4200) := by
  sorry

#check income_growth_and_projection

end NUMINAMATH_CALUDE_income_growth_and_projection_l586_58676


namespace NUMINAMATH_CALUDE_intersection_point_l586_58658

/-- The slope of the given line -/
def m : ℝ := 2

/-- The y-intercept of the given line -/
def b : ℝ := 5

/-- The x-coordinate of the point on the perpendicular line -/
def x₀ : ℝ := 5

/-- The y-coordinate of the point on the perpendicular line -/
def y₀ : ℝ := 5

/-- The x-coordinate of the claimed intersection point -/
def x_int : ℝ := 1

/-- The y-coordinate of the claimed intersection point -/
def y_int : ℝ := 7

/-- Theorem stating that (x_int, y_int) is the intersection point of the given line
    and its perpendicular line passing through (x₀, y₀) -/
theorem intersection_point :
  (y_int = m * x_int + b) ∧
  (y_int - y₀ = -(1/m) * (x_int - x₀)) ∧
  (∀ x y : ℝ, (y = m * x + b) ∧ (y - y₀ = -(1/m) * (x - x₀)) → x = x_int ∧ y = y_int) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_l586_58658


namespace NUMINAMATH_CALUDE_vector_equations_l586_58657

def A : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (3, -1)
def C : ℝ × ℝ := (-3, -4)

def a : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def b : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
def c : ℝ × ℝ := (A.1 - C.1, A.2 - C.2)

theorem vector_equations :
  (3 * a.1 + b.1, 3 * a.2 + b.2) = (9, -18) ∧
  a = (-b.1 - c.1, -b.2 - c.2) :=
sorry

end NUMINAMATH_CALUDE_vector_equations_l586_58657


namespace NUMINAMATH_CALUDE_school_classrooms_problem_l586_58606

theorem school_classrooms_problem (original_desks new_desks new_classrooms : ℕ) 
  (h1 : original_desks = 539)
  (h2 : new_desks = 1080)
  (h3 : new_classrooms = 9)
  (h4 : ∃ (original_classrooms : ℕ), original_classrooms > 0 ∧ original_desks % original_classrooms = 0)
  (h5 : ∃ (current_classrooms : ℕ), current_classrooms = original_classrooms + new_classrooms)
  (h6 : ∃ (new_desks_per_classroom : ℕ), new_desks_per_classroom > 0 ∧ new_desks % current_classrooms = 0)
  (h7 : ∀ (original_desks_per_classroom : ℕ), 
    original_desks_per_classroom > 0 → 
    original_desks = original_classrooms * original_desks_per_classroom → 
    new_desks_per_classroom > original_desks_per_classroom) :
  current_classrooms = 20 :=
sorry

end NUMINAMATH_CALUDE_school_classrooms_problem_l586_58606


namespace NUMINAMATH_CALUDE_function_and_inequality_properties_l586_58679

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| - |2*x - 1|

-- Define the theorem
theorem function_and_inequality_properties :
  ∀ (a b : ℝ), a ≠ 0 →
  (∀ (x m : ℝ), |b + 2*a| - |2*b - a| ≥ |a| * (|x + 1| + |x - m|)) →
  (∀ (x : ℝ), f x > -5 ↔ x ∈ Set.Ioo (-2) 8) ∧
  (∀ (m : ℝ), m ∈ Set.Icc (-7/2) (3/2)) :=
by sorry

end NUMINAMATH_CALUDE_function_and_inequality_properties_l586_58679


namespace NUMINAMATH_CALUDE_division_problem_l586_58604

theorem division_problem (n : ℕ) : 
  (n / 20 = 9) ∧ (n % 20 = 1) → n = 181 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l586_58604


namespace NUMINAMATH_CALUDE_escalator_standing_time_l586_58626

/-- Represents the time it takes Clea to ride an escalator in different scenarios -/
structure EscalatorRide where
  nonOperatingWalkTime : ℝ
  operatingWalkTime : ℝ
  standingTime : ℝ

/-- Proves that given the conditions, the standing time on the operating escalator is 80 seconds -/
theorem escalator_standing_time (ride : EscalatorRide) 
  (h1 : ride.nonOperatingWalkTime = 120)
  (h2 : ride.operatingWalkTime = 48) :
  ride.standingTime = 80 := by
  sorry

#check escalator_standing_time

end NUMINAMATH_CALUDE_escalator_standing_time_l586_58626


namespace NUMINAMATH_CALUDE_jack_morning_emails_l586_58623

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := sorry

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 8

/-- The difference between afternoon and morning emails -/
def email_difference : ℕ := 2

theorem jack_morning_emails : 
  morning_emails = 6 ∧ 
  afternoon_emails = morning_emails + email_difference := by
  sorry

end NUMINAMATH_CALUDE_jack_morning_emails_l586_58623


namespace NUMINAMATH_CALUDE_composite_expression_l586_58685

theorem composite_expression (x y : ℕ) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 2022*x^2 + 349*x + 72*x*y + 12*y + 2 = a * b := by
  sorry

end NUMINAMATH_CALUDE_composite_expression_l586_58685


namespace NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l586_58697

theorem quadratic_equation_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 2 * x - 1 = 0 ∧ k * y^2 - 2 * y - 1 = 0) ↔ 
  (k > -1 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l586_58697
