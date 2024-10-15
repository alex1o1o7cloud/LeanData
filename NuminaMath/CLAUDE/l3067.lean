import Mathlib

namespace NUMINAMATH_CALUDE_divisible_by_seven_pair_l3067_306714

theorem divisible_by_seven_pair : ∃! (x y : ℕ), x < 10 ∧ y < 10 ∧
  (1000 + 100 * x + 10 * y + 2) % 7 = 0 ∧
  (1000 * x + 120 + y) % 7 = 0 ∧
  x = 6 ∧ y = 5 := by sorry

end NUMINAMATH_CALUDE_divisible_by_seven_pair_l3067_306714


namespace NUMINAMATH_CALUDE_num_unique_heights_equals_multiples_of_five_l3067_306709

/-- Represents the dimensions of a brick in inches -/
structure BrickDimensions where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the configuration of a tower of bricks -/
def TowerConfiguration := List Nat

/-- The number of bricks in the tower -/
def numBricks : Nat := 80

/-- The dimensions of each brick -/
def brickDimensions : BrickDimensions := { small := 3, medium := 8, large := 18 }

/-- Calculate the height of a tower given its configuration -/
def towerHeight (config : TowerConfiguration) : Nat :=
  config.sum

/-- Generate all possible tower configurations -/
def allConfigurations : List TowerConfiguration :=
  sorry

/-- Calculate the number of unique tower heights -/
def numUniqueHeights : Nat :=
  (allConfigurations.map towerHeight).toFinset.card

/-- The main theorem to prove -/
theorem num_unique_heights_equals_multiples_of_five :
  numUniqueHeights = (((numBricks * brickDimensions.large) - (numBricks * brickDimensions.small)) / 5 + 1) := by
  sorry

end NUMINAMATH_CALUDE_num_unique_heights_equals_multiples_of_five_l3067_306709


namespace NUMINAMATH_CALUDE_min_comparisons_for_max_l3067_306788

/-- Represents a list of n pairwise distinct numbers -/
def DistinctNumbers (n : ℕ) := { l : List ℝ // l.length = n ∧ l.Pairwise (· ≠ ·) }

/-- Represents a comparison between two numbers -/
def Comparison := ℝ × ℝ

/-- A function that finds the maximum number in a list using pairwise comparisons -/
def FindMax (n : ℕ) (numbers : DistinctNumbers n) : 
  { comparisons : List Comparison // comparisons.length = n - 1 ∧ 
    ∃ max, max ∈ numbers.val ∧ ∀ x ∈ numbers.val, x ≤ max } :=
sorry

theorem min_comparisons_for_max (n : ℕ) (numbers : DistinctNumbers n) :
  (∀ comparisons : List Comparison, 
    (∃ max, max ∈ numbers.val ∧ ∀ x ∈ numbers.val, x ≤ max) → 
    comparisons.length ≥ n - 1) ∧
  (∃ comparisons : List Comparison, 
    comparisons.length = n - 1 ∧ 
    ∃ max, max ∈ numbers.val ∧ ∀ x ∈ numbers.val, x ≤ max) :=
sorry

end NUMINAMATH_CALUDE_min_comparisons_for_max_l3067_306788


namespace NUMINAMATH_CALUDE_commodity_trade_fair_companies_l3067_306744

theorem commodity_trade_fair_companies : ∃ (n : ℕ), n > 0 ∧ n * (n - 1) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_commodity_trade_fair_companies_l3067_306744


namespace NUMINAMATH_CALUDE_complex_number_location_l3067_306775

theorem complex_number_location (z : ℂ) (h : (2 + 3*I)*z = 1 + I) :
  (z.re > 0) ∧ (z.im < 0) :=
sorry

end NUMINAMATH_CALUDE_complex_number_location_l3067_306775


namespace NUMINAMATH_CALUDE_triangle_area_l3067_306712

/-- The area of a triangle with side lengths 15, 36, and 39 is 270 -/
theorem triangle_area (a b c : ℝ) (h1 : a = 15) (h2 : b = 36) (h3 : c = 39) :
  (1/2 : ℝ) * a * b = 270 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3067_306712


namespace NUMINAMATH_CALUDE_sulfuric_acid_mixture_l3067_306785

/-- Proves that mixing 42 liters of 2% sulfuric acid solution with 18 liters of 12% sulfuric acid solution results in a 60-liter solution containing 5% sulfuric acid. -/
theorem sulfuric_acid_mixture :
  let solution1_volume : ℝ := 42
  let solution1_concentration : ℝ := 0.02
  let solution2_volume : ℝ := 18
  let solution2_concentration : ℝ := 0.12
  let total_volume : ℝ := solution1_volume + solution2_volume
  let total_acid : ℝ := solution1_volume * solution1_concentration + solution2_volume * solution2_concentration
  let final_concentration : ℝ := total_acid / total_volume
  total_volume = 60 ∧ final_concentration = 0.05 := by
  sorry

#check sulfuric_acid_mixture

end NUMINAMATH_CALUDE_sulfuric_acid_mixture_l3067_306785


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3067_306751

theorem polynomial_evaluation : 
  ∃ x : ℝ, x > 0 ∧ x^2 - 3*x - 10 = 0 ∧ x^3 - 3*x^2 - 9*x + 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3067_306751


namespace NUMINAMATH_CALUDE_square_plot_area_l3067_306795

/-- Given a square plot with a perimeter that costs a certain amount to fence at a given price per foot, 
    this theorem proves that the area of the plot is as calculated. -/
theorem square_plot_area 
  (perimeter_cost : ℝ) 
  (price_per_foot : ℝ) 
  (perimeter_cost_positive : perimeter_cost > 0)
  (price_per_foot_positive : price_per_foot > 0)
  (h_cost : perimeter_cost = 3944)
  (h_price : price_per_foot = 58) : 
  (perimeter_cost / (4 * price_per_foot))^2 = 289 := by
  sorry

#eval (3944 / (4 * 58))^2  -- Should evaluate to 289.0

end NUMINAMATH_CALUDE_square_plot_area_l3067_306795


namespace NUMINAMATH_CALUDE_book_club_groups_l3067_306761

theorem book_club_groups (n m : ℕ) (hn : n = 7) (hm : m = 4) :
  Nat.choose n m = 35 := by
  sorry

end NUMINAMATH_CALUDE_book_club_groups_l3067_306761


namespace NUMINAMATH_CALUDE_volleyballs_left_l3067_306739

theorem volleyballs_left (total : ℕ) (lent : ℕ) (left : ℕ) : 
  total = 9 → lent = 5 → left = total - lent → left = 4 := by sorry

end NUMINAMATH_CALUDE_volleyballs_left_l3067_306739


namespace NUMINAMATH_CALUDE_weekly_earnings_correct_l3067_306778

/-- Represents the weekly earnings of Jake, Jacob, and Jim --/
structure WeeklyEarnings where
  jacob : ℕ
  jake : ℕ
  jim : ℕ

/-- Calculates the weekly earnings based on the given conditions --/
def calculateWeeklyEarnings : WeeklyEarnings :=
  let jacobWeekdayRate := 6
  let jacobWeekendRate := 8
  let weekdayHours := 8
  let weekendHours := 5
  let weekdays := 5
  let weekendDays := 2

  let jacobWeekdayEarnings := jacobWeekdayRate * weekdayHours * weekdays
  let jacobWeekendEarnings := jacobWeekendRate * weekendHours * weekendDays
  let jacobTotal := jacobWeekdayEarnings + jacobWeekendEarnings

  let jakeWeekdayRate := 3 * jacobWeekdayRate
  let jakeWeekdayEarnings := jakeWeekdayRate * weekdayHours * weekdays
  let jakeWeekendEarnings := jacobWeekendEarnings
  let jakeTotal := jakeWeekdayEarnings + jakeWeekendEarnings

  let jimWeekdayRate := 2 * jakeWeekdayRate
  let jimWeekdayEarnings := jimWeekdayRate * weekdayHours * weekdays
  let jimWeekendEarnings := jacobWeekendEarnings
  let jimTotal := jimWeekdayEarnings + jimWeekendEarnings

  { jacob := jacobTotal, jake := jakeTotal, jim := jimTotal }

/-- Theorem stating that the calculated weekly earnings match the expected values --/
theorem weekly_earnings_correct : 
  let earnings := calculateWeeklyEarnings
  earnings.jacob = 320 ∧ earnings.jake = 800 ∧ earnings.jim = 1520 := by
  sorry

end NUMINAMATH_CALUDE_weekly_earnings_correct_l3067_306778


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l3067_306792

theorem complex_equation_solutions : 
  ∃ (S : Finset ℂ), (∀ z ∈ S, Complex.abs z < 24 ∧ Complex.exp z = (z - 2) / (z + 2)) ∧ 
                    Finset.card S = 8 ∧
                    ∀ z, Complex.abs z < 24 → Complex.exp z = (z - 2) / (z + 2) → z ∈ S := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l3067_306792


namespace NUMINAMATH_CALUDE_house_profit_percentage_l3067_306738

/-- Proves that given two houses sold at $10,000 each, with a 10% loss on the second house
    and a 17% net profit overall, the profit percentage on the first house is approximately 67.15%. -/
theorem house_profit_percentage (selling_price : ℝ) (loss_percentage : ℝ) (net_profit_percentage : ℝ) :
  selling_price = 10000 →
  loss_percentage = 0.10 →
  net_profit_percentage = 0.17 →
  ∃ (profit_percentage : ℝ), abs (profit_percentage - 0.6715) < 0.0001 :=
by sorry

end NUMINAMATH_CALUDE_house_profit_percentage_l3067_306738


namespace NUMINAMATH_CALUDE_total_students_olympiad_l3067_306703

/-- Represents a mathematics teacher at Archimedes Academy -/
inductive Teacher
| Euler
| Fibonacci
| Gauss
| Noether

/-- Returns the number of students taking the Math Olympiad for a given teacher -/
def students_in_class (t : Teacher) : Nat :=
  match t with
  | Teacher.Euler => 15
  | Teacher.Fibonacci => 10
  | Teacher.Gauss => 12
  | Teacher.Noether => 7

/-- The list of all teachers at Archimedes Academy -/
def all_teachers : List Teacher :=
  [Teacher.Euler, Teacher.Fibonacci, Teacher.Gauss, Teacher.Noether]

/-- Theorem stating that the total number of students taking the Math Olympiad is 44 -/
theorem total_students_olympiad :
  (all_teachers.map students_in_class).sum = 44 := by
  sorry

end NUMINAMATH_CALUDE_total_students_olympiad_l3067_306703


namespace NUMINAMATH_CALUDE_cubic_polynomial_value_at_5_l3067_306715

/-- A cubic polynomial satisfying specific conditions -/
def cubic_polynomial (p : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, p x = a*x^3 + b*x^2 + c*x + d) ∧
  (p 1 = 1) ∧
  (p 2 = 1/8) ∧
  (p 3 = 1/27) ∧
  (p 4 = 1/64)

/-- The main theorem -/
theorem cubic_polynomial_value_at_5 (p : ℝ → ℝ) (h : cubic_polynomial p) :
  p 5 = -76/375 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_value_at_5_l3067_306715


namespace NUMINAMATH_CALUDE_lloyd_excess_rate_multiple_l3067_306720

/-- Represents Lloyd's work information --/
structure WorkInfo where
  regularHours : Float
  regularRate : Float
  totalHours : Float
  totalEarnings : Float

/-- Calculates the multiple of regular rate for excess hours --/
def excessRateMultiple (info : WorkInfo) : Float :=
  let regularEarnings := info.regularHours * info.regularRate
  let excessHours := info.totalHours - info.regularHours
  let excessEarnings := info.totalEarnings - regularEarnings
  let excessRate := excessEarnings / excessHours
  excessRate / info.regularRate

/-- Theorem stating that the multiple of regular rate for excess hours is 1.5 --/
theorem lloyd_excess_rate_multiple :
  let lloyd : WorkInfo := {
    regularHours := 7.5,
    regularRate := 3.5,
    totalHours := 10.5,
    totalEarnings := 42
  }
  excessRateMultiple lloyd = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_lloyd_excess_rate_multiple_l3067_306720


namespace NUMINAMATH_CALUDE_binary_representation_1023_l3067_306730

/-- Represents a binary expansion of a natural number -/
def BinaryExpansion (n : ℕ) : List Bool :=
  sorry

/-- Counts the number of true values in a list of booleans -/
def countOnes (l : List Bool) : ℕ :=
  sorry

/-- Calculates the sum of indices where the value is true -/
def sumIndices (l : List Bool) : ℕ :=
  sorry

theorem binary_representation_1023 :
  let binary := BinaryExpansion 1023
  (sumIndices binary = 45) ∧ (countOnes binary = 10) :=
sorry

end NUMINAMATH_CALUDE_binary_representation_1023_l3067_306730


namespace NUMINAMATH_CALUDE_ski_trips_theorem_l3067_306762

/-- Represents the ski lift problem -/
structure SkiLiftProblem where
  lift_time : ℕ  -- Time to ride the lift up (in minutes)
  ski_time : ℕ   -- Time to ski down (in minutes)
  known_trips : ℕ  -- Known number of trips in 2 hours
  known_hours : ℕ  -- Known number of hours for known_trips

/-- Calculates the number of ski trips possible in a given number of hours -/
def ski_trips (problem : SkiLiftProblem) (hours : ℕ) : ℕ :=
  3 * hours

/-- Theorem stating the relationship between hours and number of ski trips -/
theorem ski_trips_theorem (problem : SkiLiftProblem) (hours : ℕ) :
  problem.lift_time = 15 →
  problem.ski_time = 5 →
  problem.known_trips = 6 →
  problem.known_hours = 2 →
  ski_trips problem hours = 3 * hours :=
by
  sorry

#check ski_trips_theorem

end NUMINAMATH_CALUDE_ski_trips_theorem_l3067_306762


namespace NUMINAMATH_CALUDE_remainder_sum_l3067_306736

theorem remainder_sum (n : ℤ) : n % 20 = 13 → (n % 4 + n % 5 = 4) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l3067_306736


namespace NUMINAMATH_CALUDE_smaller_circle_area_smaller_circle_radius_l3067_306735

/-- Two externally tangent circles with common tangents -/
structure TangentCircles where
  R : ℝ  -- radius of larger circle
  r : ℝ  -- radius of smaller circle
  tangent_length : ℝ  -- length of common tangent segment (PA and AB)
  circles_tangent : R = 2 * r  -- condition for external tangency
  common_tangent : tangent_length = 4  -- given PA = AB = 4

/-- The area of the smaller circle in a TangentCircles configuration is 2π -/
theorem smaller_circle_area (tc : TangentCircles) : 
  Real.pi * tc.r^2 = 2 * Real.pi := by
  sorry

/-- Alternative formulation using Real.sqrt -/
theorem smaller_circle_radius (tc : TangentCircles) :
  tc.r = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_smaller_circle_area_smaller_circle_radius_l3067_306735


namespace NUMINAMATH_CALUDE_uniform_probability_diff_colors_l3067_306725

def shorts_colors := Fin 3
def jersey_colors := Fin 3

def total_combinations : ℕ := 9

def matching_combinations : ℕ := 2

theorem uniform_probability_diff_colors :
  (total_combinations - matching_combinations) / total_combinations = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_uniform_probability_diff_colors_l3067_306725


namespace NUMINAMATH_CALUDE_correct_division_l3067_306748

theorem correct_division (x : ℤ) : x + 4 = 40 → x / 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_correct_division_l3067_306748


namespace NUMINAMATH_CALUDE_billion_scientific_notation_l3067_306700

/-- Represents 1.2 billion in decimal form -/
def billion : ℝ := 1200000000

/-- Represents 1.2 × 10^8 in scientific notation -/
def scientific_notation : ℝ := 1.2 * (10^8)

/-- Theorem stating that 1.2 billion is equal to 1.2 × 10^8 in scientific notation -/
theorem billion_scientific_notation : billion = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_billion_scientific_notation_l3067_306700


namespace NUMINAMATH_CALUDE_range_of_m_l3067_306777

-- Define the sets M and N
def M (m : ℝ) : Set ℝ := {x : ℝ | x + m ≥ 0}
def N : Set ℝ := {x : ℝ | x^2 - 2*x - 8 < 0}

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Theorem statement
theorem range_of_m (m : ℝ) : (Set.compl (M m) ∩ N = ∅) → m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3067_306777


namespace NUMINAMATH_CALUDE_apple_cost_for_two_weeks_l3067_306750

/-- Represents the cost of apples for Irene and her dog for 2 weeks -/
def appleCost (daysPerWeek : ℕ) (weeks : ℕ) (appleWeight : ℚ) (pricePerPound : ℚ) : ℚ :=
  let totalDays : ℕ := daysPerWeek * weeks
  let totalApples : ℕ := totalDays
  let totalWeight : ℚ := appleWeight * totalApples
  totalWeight * pricePerPound

/-- Theorem stating that the cost of apples for 2 weeks is $7.00 -/
theorem apple_cost_for_two_weeks :
  appleCost 7 2 (1/4) 2 = 7 :=
sorry

end NUMINAMATH_CALUDE_apple_cost_for_two_weeks_l3067_306750


namespace NUMINAMATH_CALUDE_scientific_notation_10374_billion_l3067_306701

/-- Converts a number to scientific notation with a specified number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ℝ × ℤ :=
  sorry

/-- Checks if two scientific notation representations are equal -/
def scientificNotationEqual (a : ℝ) (b : ℤ) (c : ℝ) (d : ℤ) : Prop :=
  sorry

theorem scientific_notation_10374_billion :
  let result := toScientificNotation (10374 * 1000000000) 3
  scientificNotationEqual result.1 result.2 1.04 13 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_10374_billion_l3067_306701


namespace NUMINAMATH_CALUDE_last_ball_black_prob_specific_box_l3067_306769

/-- A box containing black and white balls -/
structure Box where
  black : ℕ
  white : ℕ

/-- The probability of the last ball being black in a drawing process -/
def last_ball_black_prob (b : Box) : ℚ :=
  b.white / (b.black + b.white)

/-- The theorem stating the probability of the last ball being black for a specific box -/
theorem last_ball_black_prob_specific_box :
  let b : Box := { black := 3, white := 4 }
  last_ball_black_prob b = 4/7 := by
  sorry

#check last_ball_black_prob_specific_box

end NUMINAMATH_CALUDE_last_ball_black_prob_specific_box_l3067_306769


namespace NUMINAMATH_CALUDE_total_muffins_for_sale_l3067_306763

theorem total_muffins_for_sale : 
  let num_boys : ℕ := 3
  let num_girls : ℕ := 2
  let muffins_per_boy : ℕ := 12
  let muffins_per_girl : ℕ := 20
  let total_muffins : ℕ := num_boys * muffins_per_boy + num_girls * muffins_per_girl
  total_muffins = 76 :=
by
  sorry

end NUMINAMATH_CALUDE_total_muffins_for_sale_l3067_306763


namespace NUMINAMATH_CALUDE_price_reduction_equation_l3067_306724

/-- Represents the price reduction scenario for an item -/
structure PriceReduction where
  initial_price : ℝ
  final_price : ℝ
  reduction_percentage : ℝ
  num_reductions : ℕ

/-- Theorem stating the relationship between initial price, final price, and reduction percentage -/
theorem price_reduction_equation (pr : PriceReduction) 
  (h1 : pr.initial_price = 150)
  (h2 : pr.final_price = 96)
  (h3 : pr.num_reductions = 2) :
  pr.initial_price * (1 - pr.reduction_percentage)^pr.num_reductions = pr.final_price := by
  sorry

#check price_reduction_equation

end NUMINAMATH_CALUDE_price_reduction_equation_l3067_306724


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l3067_306760

/-- Represents an ellipse -/
structure Ellipse where
  center : ℝ × ℝ
  passes_through : ℝ × ℝ
  a_b_ratio : ℝ

/-- Checks if the given equation represents the standard form of the ellipse -/
def is_standard_equation (e : Ellipse) (eq : ℝ → ℝ → Bool) : Prop :=
  (eq 3 0 = true) ∧ 
  (∀ x y, eq x y ↔ (x^2 / 9 + y^2 = 1 ∨ y^2 / 81 + x^2 / 9 = 1))

/-- Theorem: Given the conditions, the ellipse has one of the two standard equations -/
theorem ellipse_standard_equation (e : Ellipse) 
  (h1 : e.center = (0, 0))
  (h2 : e.passes_through = (3, 0))
  (h3 : e.a_b_ratio = 3) :
  ∃ eq : ℝ → ℝ → Bool, is_standard_equation e eq := by
  sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l3067_306760


namespace NUMINAMATH_CALUDE_log_exponent_sum_l3067_306747

theorem log_exponent_sum (a b : ℝ) (h1 : a = Real.log 25) (h2 : b = Real.log 36) :
  (5 : ℝ) ^ (a / b) + (6 : ℝ) ^ (b / a) = 11 := by
  sorry

end NUMINAMATH_CALUDE_log_exponent_sum_l3067_306747


namespace NUMINAMATH_CALUDE_granger_age_multiple_l3067_306789

/-- The multiple of Mr. Granger's son's age last year that Mr. Granger's age last year was 4 years less than -/
def multiple_last_year (grangers_age : ℕ) (sons_age : ℕ) : ℚ :=
  (grangers_age - 1) / (sons_age - 1)

/-- Mr. Granger's current age -/
def grangers_age : ℕ := 42

/-- Mr. Granger's son's current age -/
def sons_age : ℕ := 16

theorem granger_age_multiple : multiple_last_year grangers_age sons_age = 3 := by
  sorry

end NUMINAMATH_CALUDE_granger_age_multiple_l3067_306789


namespace NUMINAMATH_CALUDE_asafa_arrives_5_min_after_florence_l3067_306756

/-- Represents a point in the route -/
inductive Point | P | Q | R | S

/-- Represents a runner -/
inductive Runner | Asafa | Florence

/-- The speed of a runner in km/h -/
def speed (r : Runner) : ℝ :=
  match r with
  | Runner.Asafa => 21
  | Runner.Florence => 16.8  -- This is derived, not given directly

/-- The distance between two points in km -/
def distance (p1 p2 : Point) : ℝ :=
  match p1, p2 with
  | Point.P, Point.Q => 8
  | Point.Q, Point.R => 15
  | Point.R, Point.S => 7
  | Point.P, Point.R => 17  -- This is derived, not given directly
  | _, _ => 0  -- For all other combinations

/-- The time difference in minutes between Florence and Asafa arriving at point R -/
def time_difference_at_R : ℝ := 5

/-- The theorem to be proved -/
theorem asafa_arrives_5_min_after_florence :
  let total_distance_asafa := distance Point.P Point.Q + distance Point.Q Point.R + distance Point.R Point.S
  let total_distance_florence := distance Point.P Point.R + distance Point.R Point.S
  let total_time := total_distance_asafa / speed Runner.Asafa
  let time_asafa_RS := distance Point.R Point.S / speed Runner.Asafa
  let time_florence_RS := distance Point.R Point.S / speed Runner.Florence
  time_florence_RS - time_asafa_RS = time_difference_at_R / 60 := by
  sorry

end NUMINAMATH_CALUDE_asafa_arrives_5_min_after_florence_l3067_306756


namespace NUMINAMATH_CALUDE_ellipse_x_intersection_l3067_306759

/-- Definition of the ellipse based on the given conditions -/
def ellipse (P : ℝ × ℝ) : Prop :=
  let F₁ : ℝ × ℝ := (0, 1)
  let F₂ : ℝ × ℝ := (4, 0)
  let d : ℝ := Real.sqrt 2 + 3
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) +
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = d

/-- The theorem stating the other intersection point of the ellipse with the x-axis -/
theorem ellipse_x_intersection :
  ellipse (1, 0) →
  ∃ x : ℝ, x ≠ 1 ∧ ellipse (x, 0) ∧ x = 3 * Real.sqrt 2 / 4 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_x_intersection_l3067_306759


namespace NUMINAMATH_CALUDE_sin_negative_45_degrees_l3067_306773

theorem sin_negative_45_degrees : Real.sin (-(π / 4)) = -(Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_45_degrees_l3067_306773


namespace NUMINAMATH_CALUDE_age_relation_l3067_306716

/-- Given that p is currently 3 times as old as q and p was 30 years old 3 years ago,
    prove that p will be twice as old as q in 11 years. -/
theorem age_relation (p q : ℕ) (x : ℕ) : 
  p = 3 * q →  -- p is 3 times as old as q
  p = 30 + 3 →  -- p was 30 years old 3 years ago
  p + x = 2 * (q + x) →  -- in x years, p will be twice as old as q
  x = 11 := by
sorry

end NUMINAMATH_CALUDE_age_relation_l3067_306716


namespace NUMINAMATH_CALUDE_expression_evaluation_l3067_306791

theorem expression_evaluation : 2197 + 180 / 60 * 3 - 197 = 2009 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3067_306791


namespace NUMINAMATH_CALUDE_tammy_orange_earnings_l3067_306702

/-- Calculates Tammy's earnings from selling oranges over 3 weeks --/
def tammys_earnings (num_trees : ℕ) (oranges_per_tree : ℕ) (oranges_per_pack : ℕ) 
  (price_per_pack : ℚ) (days : ℕ) : ℚ :=
  let oranges_per_day := num_trees * oranges_per_tree
  let packs_per_day := oranges_per_day / oranges_per_pack
  let packs_in_period := packs_per_day * days
  packs_in_period * price_per_pack

/-- Proves that Tammy's earnings after 3 weeks will be $840 --/
theorem tammy_orange_earnings : 
  tammys_earnings 10 12 6 2 21 = 840 := by sorry

end NUMINAMATH_CALUDE_tammy_orange_earnings_l3067_306702


namespace NUMINAMATH_CALUDE_prob_at_least_two_same_l3067_306732

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The probability of at least two dice showing the same number when rolling four fair 8-sided dice -/
theorem prob_at_least_two_same (num_sides : ℕ) (num_dice : ℕ) : 
  num_sides = 8 → num_dice = 4 → 
  (1 - (num_sides * (num_sides - 1) * (num_sides - 2) * (num_sides - 3)) / (num_sides ^ num_dice)) = 151/256 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_same_l3067_306732


namespace NUMINAMATH_CALUDE_cubic_symmetry_extrema_l3067_306737

/-- A cubic function that is symmetric about the origin -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + (a - 1) * x^2 + 48 * (b - 3) * x + b

/-- The derivative of f -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * (a - 1) * x + 144

/-- The discriminant of f' = 0 -/
def discriminant (a : ℝ) : ℝ := 4 * (a^2 - 434 * a + 1)

theorem cubic_symmetry_extrema (a b : ℝ) :
  (∀ x, f a b x = f a b (-x)) →  -- symmetry about the origin
  (∃ x_min, ∀ x, f a b x_min ≤ f a b x) ∧ 
  (∃ x_max, ∀ x, f a b x ≤ f a b x_max) := by
  sorry

end NUMINAMATH_CALUDE_cubic_symmetry_extrema_l3067_306737


namespace NUMINAMATH_CALUDE_M_divisible_by_40_l3067_306722

/-- M is the number formed by concatenating integers from 1 to 39 -/
def M : ℕ := sorry

/-- Theorem stating that M is divisible by 40 -/
theorem M_divisible_by_40 : 40 ∣ M := by sorry

end NUMINAMATH_CALUDE_M_divisible_by_40_l3067_306722


namespace NUMINAMATH_CALUDE_max_sundays_in_45_days_l3067_306740

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a year with a starting day -/
structure Year where
  startDay : DayOfWeek

/-- Counts the number of Sundays in the first n days of a year -/
def countSundays (y : Year) (n : ℕ) : ℕ :=
  sorry

/-- The maximum number of Sundays in the first 45 days of a year is 7 -/
theorem max_sundays_in_45_days :
  ∀ y : Year, countSundays y 45 ≤ 7 ∧ ∃ y' : Year, countSundays y' 45 = 7 :=
sorry

end NUMINAMATH_CALUDE_max_sundays_in_45_days_l3067_306740


namespace NUMINAMATH_CALUDE_stationery_prices_l3067_306719

-- Define the variables
variable (x : ℝ) -- Price of one notebook
variable (y : ℝ) -- Price of one pen

-- Define the theorem
theorem stationery_prices : 
  (3 * x + 5 * y = 30) ∧ 
  (30 - (3 * x + 5 * y + 2 * y) = -0.4) ∧ 
  (30 - (3 * x + 5 * y + 2 * x) = -2) → 
  (x = 3.6 ∧ y = 2.8) :=
by sorry

end NUMINAMATH_CALUDE_stationery_prices_l3067_306719


namespace NUMINAMATH_CALUDE_factorization_proof_l3067_306764

theorem factorization_proof (x y : ℝ) : x^2 - 2*x^2*y + x*y^2 = x*(x - 2*x*y + y^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3067_306764


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l3067_306755

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem composition_of_even_is_even (f : ℝ → ℝ) (h : IsEven f) :
  IsEven (fun x ↦ f (f x)) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l3067_306755


namespace NUMINAMATH_CALUDE_arrange_crosses_and_zeros_theorem_l3067_306710

def arrange_crosses_and_zeros (n : ℕ) : ℕ :=
  if n = 27 then 14
  else if n = 26 then 105
  else 0

theorem arrange_crosses_and_zeros_theorem :
  (arrange_crosses_and_zeros 27 = 14) ∧
  (arrange_crosses_and_zeros 26 = 105) :=
sorry

end NUMINAMATH_CALUDE_arrange_crosses_and_zeros_theorem_l3067_306710


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3067_306784

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 24 →
  b = 18 →
  c^2 = a^2 + b^2 →
  c = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3067_306784


namespace NUMINAMATH_CALUDE_value_of_p_l3067_306783

theorem value_of_p (p q r : ℝ) 
  (sum_eq : p + q + r = 70)
  (p_eq : p = 2 * q)
  (q_eq : q = 3 * r) : 
  p = 42 := by
sorry

end NUMINAMATH_CALUDE_value_of_p_l3067_306783


namespace NUMINAMATH_CALUDE_forty_eggs_not_eaten_l3067_306734

/-- Represents the number of eggs in a problem about weekly egg consumption --/
structure EggProblem where
  trays_per_week : ℕ
  eggs_per_tray : ℕ
  children_eggs_per_day : ℕ
  parents_eggs_per_day : ℕ
  days_per_week : ℕ

/-- Calculates the number of eggs not eaten in a week --/
def eggs_not_eaten (p : EggProblem) : ℕ :=
  p.trays_per_week * p.eggs_per_tray - 
  (p.children_eggs_per_day + p.parents_eggs_per_day) * p.days_per_week

/-- Theorem stating that given the problem conditions, 40 eggs are not eaten in a week --/
theorem forty_eggs_not_eaten (p : EggProblem) 
  (h1 : p.trays_per_week = 2)
  (h2 : p.eggs_per_tray = 24)
  (h3 : p.children_eggs_per_day = 4)
  (h4 : p.parents_eggs_per_day = 4)
  (h5 : p.days_per_week = 7) :
  eggs_not_eaten p = 40 := by
  sorry

end NUMINAMATH_CALUDE_forty_eggs_not_eaten_l3067_306734


namespace NUMINAMATH_CALUDE_stadium_length_feet_l3067_306728

/-- Conversion factor from yards to feet -/
def yards_to_feet : ℕ → ℕ := λ x => 3 * x

/-- Length of the stadium in yards -/
def stadium_length_yards : ℕ := 80

/-- Theorem stating that the stadium length in feet is 240 -/
theorem stadium_length_feet : yards_to_feet stadium_length_yards = 240 := by
  sorry

end NUMINAMATH_CALUDE_stadium_length_feet_l3067_306728


namespace NUMINAMATH_CALUDE_right_triangle_area_l3067_306752

theorem right_triangle_area (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_hypotenuse : c = 50) (h_sum_legs : a + b = 70) : 
  (1/2) * a * b = 300 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3067_306752


namespace NUMINAMATH_CALUDE_point_A_outside_circle_l3067_306708

/-- The position of point A on the number line after t seconds -/
def position_A (t : ℝ) : ℝ := 2 * t

/-- The center of circle B on the number line -/
def center_B : ℝ := 16

/-- The radius of circle B -/
def radius_B : ℝ := 4

/-- Predicate for point A being outside circle B -/
def is_outside_circle (t : ℝ) : Prop :=
  position_A t < center_B - radius_B ∨ position_A t > center_B + radius_B

theorem point_A_outside_circle (t : ℝ) :
  is_outside_circle t ↔ t < 6 ∨ t > 10 := by sorry

end NUMINAMATH_CALUDE_point_A_outside_circle_l3067_306708


namespace NUMINAMATH_CALUDE_range_of_a_theorem_l3067_306768

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, 0 < a * x^2 - x + 1/16 * a

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (a - 3/2)^y < (a - 3/2)^x

-- Define the range of a
def range_of_a (a : ℝ) : Prop := (3/2 < a ∧ a ≤ 2) ∨ a ≥ 5/2

-- State the theorem
theorem range_of_a_theorem (a : ℝ) :
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → range_of_a a :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_theorem_l3067_306768


namespace NUMINAMATH_CALUDE_teacher_worksheets_l3067_306731

theorem teacher_worksheets :
  ∀ (total_worksheets : ℕ) 
    (problems_per_worksheet : ℕ) 
    (graded_worksheets : ℕ) 
    (remaining_problems : ℕ),
  problems_per_worksheet = 7 →
  graded_worksheets = 8 →
  remaining_problems = 63 →
  problems_per_worksheet * (total_worksheets - graded_worksheets) = remaining_problems →
  total_worksheets = 17 := by
sorry

end NUMINAMATH_CALUDE_teacher_worksheets_l3067_306731


namespace NUMINAMATH_CALUDE_monomial_properties_l3067_306779

-- Define the structure of a monomial
structure Monomial (α : Type*) [Field α] where
  coeff : α
  x_exp : ℕ
  y_exp : ℕ

-- Define the given monomial
def given_monomial : Monomial ℚ := {
  coeff := -1/7,
  x_exp := 2,
  y_exp := 1
}

-- Define the coefficient of a monomial
def coefficient (m : Monomial ℚ) : ℚ := m.coeff

-- Define the degree of a monomial
def degree (m : Monomial ℚ) : ℕ := m.x_exp + m.y_exp

-- Theorem statement
theorem monomial_properties :
  coefficient given_monomial = -1/7 ∧ degree given_monomial = 3 := by
  sorry

end NUMINAMATH_CALUDE_monomial_properties_l3067_306779


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l3067_306711

theorem binomial_coefficient_ratio (n : ℕ) : 
  (∃ r : ℕ, r + 2 ≤ n ∧ 
    (n.choose r : ℚ) / (n.choose (r + 1)) = 1 / 2 ∧
    (n.choose (r + 1) : ℚ) / (n.choose (r + 2)) = 2 / 3) → 
  n = 14 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l3067_306711


namespace NUMINAMATH_CALUDE_bret_reading_time_l3067_306793

/-- The time Bret spends reading a book during a train ride -/
def time_reading_book (total_time dinner_time movie_time nap_time : ℕ) : ℕ :=
  total_time - (dinner_time + movie_time + nap_time)

/-- Theorem: Bret spends 2 hours reading a book during his train ride -/
theorem bret_reading_time :
  time_reading_book 9 1 3 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_bret_reading_time_l3067_306793


namespace NUMINAMATH_CALUDE_gcf_lcm_60_72_l3067_306799

theorem gcf_lcm_60_72 : 
  (Nat.gcd 60 72 = 12) ∧ (Nat.lcm 60 72 = 360) := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_60_72_l3067_306799


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3067_306786

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_ratio
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : d ≠ 0)
  (h2 : arithmetic_sequence a d)
  (h3 : a 3 ^ 2 = a 1 * a 9) :
  a 3 / a 6 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3067_306786


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3067_306723

theorem quadratic_inequality_solution_set (x : ℝ) :
  {x | x^2 - 4*x > 44} = {x | x < -4 ∨ x > 11} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3067_306723


namespace NUMINAMATH_CALUDE_cos_pi_minus_2alpha_l3067_306749

theorem cos_pi_minus_2alpha (α : Real) (h : Real.sin α = 2/3) : 
  Real.cos (Real.pi - 2*α) = -1/9 := by sorry

end NUMINAMATH_CALUDE_cos_pi_minus_2alpha_l3067_306749


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l3067_306721

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and subset relations
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (l : Line) (m : Line) (α β : Plane) 
  (h1 : l ≠ m) 
  (h2 : α ≠ β) 
  (h3 : subset l α) 
  (h4 : subset m β) :
  perpendicular l β → plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l3067_306721


namespace NUMINAMATH_CALUDE_hash_difference_six_four_l3067_306771

-- Define the # operation
def hash (x y : ℤ) : ℤ := x * y - 3 * x

-- State the theorem
theorem hash_difference_six_four : hash 6 4 - hash 4 6 = -6 := by
  sorry

end NUMINAMATH_CALUDE_hash_difference_six_four_l3067_306771


namespace NUMINAMATH_CALUDE_plane_ticket_price_is_800_l3067_306733

/-- Represents the luggage and ticket pricing scenario -/
structure LuggagePricing where
  totalWeight : ℕ
  freeAllowance : ℕ
  excessChargeRate : ℚ
  luggageTicketPrice : ℕ

/-- Calculates the plane ticket price based on the given luggage pricing scenario -/
def planeTicketPrice (scenario : LuggagePricing) : ℕ :=
  sorry

/-- Theorem stating that the plane ticket price is 800 yuan for the given scenario -/
theorem plane_ticket_price_is_800 :
  planeTicketPrice ⟨30, 20, 3/200, 120⟩ = 800 :=
sorry

end NUMINAMATH_CALUDE_plane_ticket_price_is_800_l3067_306733


namespace NUMINAMATH_CALUDE_otimes_four_two_l3067_306757

def otimes (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem otimes_four_two : otimes 4 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_otimes_four_two_l3067_306757


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3067_306770

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 2 + Real.sqrt 7 ∧ x₁^2 - 4*x₁ - 3 = 0) ∧
  (x₂ = 2 - Real.sqrt 7 ∧ x₂^2 - 4*x₂ - 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3067_306770


namespace NUMINAMATH_CALUDE_expression_equals_twenty_times_ten_to_1234_l3067_306776

theorem expression_equals_twenty_times_ten_to_1234 :
  (2^1234 + 5^1235)^2 - (2^1234 - 5^1235)^2 = 20 * 10^1234 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_twenty_times_ten_to_1234_l3067_306776


namespace NUMINAMATH_CALUDE_stephanies_age_to_jobs_age_ratio_l3067_306774

/-- Given the ages of Freddy, Stephanie, and Job, prove the ratio of Stephanie's age to Job's age -/
theorem stephanies_age_to_jobs_age_ratio :
  ∀ (freddy_age stephanie_age job_age : ℕ),
  freddy_age = 18 →
  stephanie_age = freddy_age + 2 →
  job_age = 5 →
  (stephanie_age : ℚ) / (job_age : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_stephanies_age_to_jobs_age_ratio_l3067_306774


namespace NUMINAMATH_CALUDE_no_basic_operation_satisfies_equation_l3067_306754

def basic_operations := [Int.add, Int.sub, Int.mul, Int.div]

theorem no_basic_operation_satisfies_equation :
  ∀ op ∈ basic_operations, (op 8 2) - 5 + 7 - (3^2 - 4) ≠ 6 := by
  sorry

end NUMINAMATH_CALUDE_no_basic_operation_satisfies_equation_l3067_306754


namespace NUMINAMATH_CALUDE_regular_polygon_140_degrees_has_9_sides_l3067_306798

/-- A regular polygon with interior angles of 140 degrees has 9 sides -/
theorem regular_polygon_140_degrees_has_9_sides :
  ∀ n : ℕ,
  n > 2 →
  (∀ angle : ℝ, angle = 140 →
    (180 * (n - 2) : ℝ) = n * angle) →
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_140_degrees_has_9_sides_l3067_306798


namespace NUMINAMATH_CALUDE_solution_range_l3067_306746

theorem solution_range (x : ℝ) : 
  x ≥ 1 → 
  Real.sqrt (x + 2 - 2 * Real.sqrt (x - 1)) + Real.sqrt (x + 5 - 3 * Real.sqrt (x - 1)) = 2 → 
  2 ≤ x ∧ x ≤ 5 := by sorry

end NUMINAMATH_CALUDE_solution_range_l3067_306746


namespace NUMINAMATH_CALUDE_three_integers_problem_l3067_306741

theorem three_integers_problem :
  ∃ (a b c : ℕ) (k : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    k > 0 ∧
    a + b + c = 93 ∧
    a * b * c = 3375 ∧
    b = k * a ∧
    c = k^2 * a ∧
    a = 3 ∧ b = 15 ∧ c = 75 :=
by sorry

end NUMINAMATH_CALUDE_three_integers_problem_l3067_306741


namespace NUMINAMATH_CALUDE_intersection_value_l3067_306753

/-- The value of k for which the lines -3x + y = k and 2x + y = 8 intersect when x = -6 -/
theorem intersection_value : ∃ k : ℝ, 
  (∀ x y : ℝ, -3*x + y = k ∧ 2*x + y = 8 → x = -6) → k = 38 := by
  sorry

end NUMINAMATH_CALUDE_intersection_value_l3067_306753


namespace NUMINAMATH_CALUDE_decimal_51_to_binary_l3067_306704

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

/-- Checks if a list of booleans represents the given binary number -/
def isBinaryRepresentation (bits : List Bool) (binaryStr : String) : Prop :=
  bits.reverse.map (fun b => if b then '1' else '0') = binaryStr.toList

theorem decimal_51_to_binary :
  isBinaryRepresentation (toBinary 51) "110011" := by
  sorry

#eval toBinary 51

end NUMINAMATH_CALUDE_decimal_51_to_binary_l3067_306704


namespace NUMINAMATH_CALUDE_yulin_school_sampling_l3067_306742

/-- Systematic sampling function that calculates the number of elements to be removed -/
def systematicSamplingRemoval (populationSize sampleSize : ℕ) : ℕ :=
  populationSize % sampleSize

/-- Theorem stating that for the given population and sample size, 
    the number of students to be removed is 2 -/
theorem yulin_school_sampling :
  systematicSamplingRemoval 254 42 = 2 := by
  sorry

end NUMINAMATH_CALUDE_yulin_school_sampling_l3067_306742


namespace NUMINAMATH_CALUDE_part_one_part_two_l3067_306745

/-- The quadratic function y in terms of x and a -/
def y (x a : ℝ) : ℝ := x^2 - (a + 2) * x + 4

/-- Part 1 of the theorem -/
theorem part_one (a b : ℝ) (h1 : b > 1) 
  (h2 : ∀ x, y x a < 0 ↔ 1 < x ∧ x < b) : 
  a = 3 ∧ b = 4 := by sorry

/-- Part 2 of the theorem -/
theorem part_two (a : ℝ) 
  (h : ∀ x, 1 ≤ x → x ≤ 4 → y x a ≥ -a - 1) : 
  a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3067_306745


namespace NUMINAMATH_CALUDE_smallest_perimeter_cross_section_area_is_sqrt_6_l3067_306707

/-- Represents a quadrilateral pyramid with a square base -/
structure QuadPyramid where
  base_side : ℝ
  lateral_height : ℝ

/-- Represents a cross-section of the pyramid -/
structure CrossSection where
  pyramid : QuadPyramid
  point_on_base : ℝ  -- Distance from A to the point on AB

/-- The area of the cross-section with the smallest perimeter -/
def smallest_perimeter_cross_section_area (p : QuadPyramid) : ℝ := sorry

/-- The theorem stating that the area of the smallest perimeter cross-section is √6 -/
theorem smallest_perimeter_cross_section_area_is_sqrt_6 
  (p : QuadPyramid) 
  (h1 : p.base_side = 2) 
  (h2 : p.lateral_height = 2) : 
  smallest_perimeter_cross_section_area p = Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_smallest_perimeter_cross_section_area_is_sqrt_6_l3067_306707


namespace NUMINAMATH_CALUDE_tangerine_cost_theorem_l3067_306766

/-- The cost of tangerines bought by Dong-jin -/
def tangerine_cost (original_money : ℚ) : ℚ :=
  original_money / 2

/-- The amount of money Dong-jin has after buying tangerines and giving some to his brother -/
def remaining_money (original_money : ℚ) : ℚ :=
  original_money / 2 * (1 - 3/8)

/-- Theorem stating the cost of tangerines given the conditions -/
theorem tangerine_cost_theorem (original_money : ℚ) :
  remaining_money original_money = 2500 →
  tangerine_cost original_money = 4000 :=
by
  sorry

end NUMINAMATH_CALUDE_tangerine_cost_theorem_l3067_306766


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l3067_306718

/-- If f(x) = kx - ln x is monotonically increasing on (1, +∞), then k ≥ 1 -/
theorem monotone_increasing_condition (k : ℝ) : 
  (∀ x > 1, Monotone (fun x => k * x - Real.log x)) → k ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l3067_306718


namespace NUMINAMATH_CALUDE_fraction_proof_l3067_306713

theorem fraction_proof (N : ℝ) (h1 : N = 150) (h2 : N - (3/5) * N = 60) : (3 : ℝ) / 5 = (3 : ℝ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_proof_l3067_306713


namespace NUMINAMATH_CALUDE_circle_passes_through_point_l3067_306758

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line
def line (x : ℝ) : Prop := x + 2 = 0

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define a circle
structure Circle where
  center : PointOnParabola
  radius : ℝ
  tangent_to_line : radius = center.x + 2

-- Theorem to prove
theorem circle_passes_through_point :
  ∀ (c : Circle), (c.center.x - 2)^2 + c.center.y^2 = c.radius^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_point_l3067_306758


namespace NUMINAMATH_CALUDE_movie_theater_problem_l3067_306767

theorem movie_theater_problem (adult_price child_price : ℚ) 
  (total_people : ℕ) (total_paid : ℚ) : 
  adult_price = 9.5 → 
  child_price = 6.5 → 
  total_people = 7 → 
  total_paid = 54.5 → 
  ∃ (adults : ℕ), 
    adults ≤ total_people ∧ 
    (adult_price * adults + child_price * (total_people - adults) = total_paid) ∧
    adults = 3 :=
by sorry

end NUMINAMATH_CALUDE_movie_theater_problem_l3067_306767


namespace NUMINAMATH_CALUDE_ages_sum_l3067_306780

theorem ages_sum (a b c : ℕ+) (h1 : a = b) (h2 : a > c) (h3 : a * b * c = 72) : a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_ages_sum_l3067_306780


namespace NUMINAMATH_CALUDE_inequality_proof_l3067_306782

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / b + b^2 / c + c^2 / a) ≥ 3 * (a^3 + b^3 + c^3) / (a^2 + b^2 + c^2) := by
  sorry


end NUMINAMATH_CALUDE_inequality_proof_l3067_306782


namespace NUMINAMATH_CALUDE_questionnaire_C_count_l3067_306796

/-- Represents the total population size -/
def population_size : ℕ := 1000

/-- Represents the sample size -/
def sample_size : ℕ := 50

/-- Represents the first number drawn in the systematic sample -/
def first_number : ℕ := 8

/-- Represents the lower bound of the interval for questionnaire C -/
def lower_bound : ℕ := 751

/-- Represents the upper bound of the interval for questionnaire C -/
def upper_bound : ℕ := 1000

/-- Theorem stating that the number of people taking questionnaire C is 12 -/
theorem questionnaire_C_count :
  (Finset.filter (fun n => lower_bound ≤ (first_number + (n - 1) * (population_size / sample_size)) ∧
                           (first_number + (n - 1) * (population_size / sample_size)) ≤ upper_bound)
                 (Finset.range sample_size)).card = 12 :=
by sorry

end NUMINAMATH_CALUDE_questionnaire_C_count_l3067_306796


namespace NUMINAMATH_CALUDE_common_remainder_difference_l3067_306772

theorem common_remainder_difference (d r : ℕ) : 
  d.Prime → 
  d > 1 → 
  r < d → 
  1274 % d = r → 
  1841 % d = r → 
  2866 % d = r → 
  d - r = 6 := by sorry

end NUMINAMATH_CALUDE_common_remainder_difference_l3067_306772


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l3067_306787

theorem at_least_one_greater_than_one (a b : ℝ) : a + b > 2 → max a b > 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l3067_306787


namespace NUMINAMATH_CALUDE_profit_function_satisfies_conditions_max_profit_at_45_profit_function_is_quadratic_l3067_306797

/-- The profit function for a toy store -/
def profit_function (x : ℝ) : ℝ := -2 * (x - 30) * (x - 60)

/-- The theorem stating that the profit function satisfies all required conditions -/
theorem profit_function_satisfies_conditions :
  (profit_function 30 = 0) ∧ 
  (∃ (max_profit : ℝ), max_profit = profit_function 45 ∧ 
    ∀ (x : ℝ), profit_function x ≤ max_profit) ∧
  (profit_function 45 = 450) ∧
  (profit_function 60 = 0) := by
  sorry

/-- The maximum profit occurs at x = 45 -/
theorem max_profit_at_45 :
  ∀ (x : ℝ), profit_function x ≤ profit_function 45 := by
  sorry

/-- The profit function is a quadratic function -/
theorem profit_function_is_quadratic :
  ∃ (a b c : ℝ), ∀ (x : ℝ), profit_function x = a * x^2 + b * x + c := by
  sorry

end NUMINAMATH_CALUDE_profit_function_satisfies_conditions_max_profit_at_45_profit_function_is_quadratic_l3067_306797


namespace NUMINAMATH_CALUDE_point_on_graph_l3067_306706

/-- A linear function passing through (0, -3) with slope 2 -/
def f (x : ℝ) : ℝ := 2 * x - 3

/-- The point (2, 1) lies on the graph of f -/
theorem point_on_graph : f 2 = 1 := by sorry

end NUMINAMATH_CALUDE_point_on_graph_l3067_306706


namespace NUMINAMATH_CALUDE_jackson_charity_collection_l3067_306717

/-- Represents the problem of Jackson's charity collection --/
theorem jackson_charity_collection 
  (total_days : ℕ) 
  (goal : ℕ) 
  (monday_earning : ℕ) 
  (tuesday_earning : ℕ) 
  (houses_per_bundle : ℕ) 
  (earning_per_bundle : ℕ) 
  (h1 : total_days = 5)
  (h2 : goal = 1000)
  (h3 : monday_earning = 300)
  (h4 : tuesday_earning = 40)
  (h5 : houses_per_bundle = 4)
  (h6 : earning_per_bundle = 10) :
  ∃ (houses_per_day : ℕ), 
    houses_per_day = 88 ∧ 
    (goal - monday_earning - tuesday_earning) = 
      (total_days - 2) * houses_per_day * (earning_per_bundle / houses_per_bundle) :=
sorry

end NUMINAMATH_CALUDE_jackson_charity_collection_l3067_306717


namespace NUMINAMATH_CALUDE_fraction_to_seventh_power_l3067_306781

theorem fraction_to_seventh_power : (2 / 5 : ℚ) ^ 7 = 128 / 78125 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_seventh_power_l3067_306781


namespace NUMINAMATH_CALUDE_abc_product_l3067_306726

theorem abc_product (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * (b + c) = 198)
  (h2 : b * (c + a) = 210)
  (h3 : c * (a + b) = 222) :
  a * b * c = 1069 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l3067_306726


namespace NUMINAMATH_CALUDE_total_cars_l3067_306794

/-- The number of cars owned by five people given specific relationships between their car counts -/
theorem total_cars (tommy : ℕ) (jessie : ℕ) : 
  tommy = 7 →
  jessie = 9 →
  (tommy + jessie + (jessie + 2) + (tommy - 3) + 2 * (jessie + 2)) = 53 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_l3067_306794


namespace NUMINAMATH_CALUDE_deck_size_l3067_306729

theorem deck_size (r b : ℕ) : 
  (r : ℚ) / (r + b) = 2/5 →
  (r : ℚ) / (r + b + 7) = 1/3 →
  r + b = 35 := by
sorry

end NUMINAMATH_CALUDE_deck_size_l3067_306729


namespace NUMINAMATH_CALUDE_no_integer_solution_l3067_306790

theorem no_integer_solution : ¬ ∃ (a b c : ℤ), a^2 + b^2 - 8*c = 6 := by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3067_306790


namespace NUMINAMATH_CALUDE_min_value_of_complex_sum_l3067_306765

theorem min_value_of_complex_sum (z : ℂ) (h : Complex.abs (z + Complex.I) + Complex.abs (z - Complex.I) = 2) :
  ∃ (min_val : ℝ), min_val = 1 ∧ ∀ w : ℂ, Complex.abs (w + Complex.I) + Complex.abs (w - Complex.I) = 2 →
    Complex.abs (z + Complex.I + 1) ≤ Complex.abs (w + Complex.I + 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_complex_sum_l3067_306765


namespace NUMINAMATH_CALUDE_sum_of_parts_l3067_306727

theorem sum_of_parts (x y : ℝ) (h1 : x + y = 56) (h2 : y = 37.66666666666667) :
  10 * x + 22 * y = 1012 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_parts_l3067_306727


namespace NUMINAMATH_CALUDE_domain_of_shifted_function_l3067_306743

-- Define the function f with domain [0,2]
def f : Set ℝ := Set.Icc 0 2

-- Define the function g(x) = f(x+1)
def g (x : ℝ) : Prop := ∃ y ∈ f, y = x + 1

-- Theorem statement
theorem domain_of_shifted_function :
  Set.Icc (-1) 1 = {x | g x} := by sorry

end NUMINAMATH_CALUDE_domain_of_shifted_function_l3067_306743


namespace NUMINAMATH_CALUDE_two_trees_remain_l3067_306705

/-- The number of walnut trees remaining after removal -/
def remaining_trees (initial : ℕ) (removed : ℕ) : ℕ :=
  initial - removed

/-- Theorem stating that 2 trees remain after removing 4 from 6 -/
theorem two_trees_remain :
  remaining_trees 6 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_trees_remain_l3067_306705
