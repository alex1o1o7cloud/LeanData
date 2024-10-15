import Mathlib

namespace NUMINAMATH_CALUDE_acme_profit_l1330_133087

/-- Calculates the profit for Acme's horseshoe manufacturing --/
def calculate_profit (initial_outlay : ℝ) (cost_per_set : ℝ) (price_per_set : ℝ) (num_sets : ℕ) : ℝ :=
  let revenue := price_per_set * num_sets
  let total_cost := initial_outlay + cost_per_set * num_sets
  revenue - total_cost

/-- Theorem stating that Acme's profit is $15,337.50 --/
theorem acme_profit :
  calculate_profit 12450 20.75 50 950 = 15337.50 := by
  sorry

end NUMINAMATH_CALUDE_acme_profit_l1330_133087


namespace NUMINAMATH_CALUDE_fraction_decimal_digits_l1330_133059

-- Define the fraction
def fraction : ℚ := 987654321 / (2^30 * 5^5)

-- Define the function to calculate the minimum number of decimal digits
def min_decimal_digits (q : ℚ) : ℕ := sorry

-- Theorem statement
theorem fraction_decimal_digits :
  min_decimal_digits fraction = 30 := by sorry

end NUMINAMATH_CALUDE_fraction_decimal_digits_l1330_133059


namespace NUMINAMATH_CALUDE_patio_tile_count_l1330_133004

/-- Represents a square patio with red tiles along its diagonals -/
structure SquarePatio where
  side_length : ℕ
  red_tiles : ℕ

/-- The number of red tiles on a square patio with given side length -/
def red_tiles_count (s : ℕ) : ℕ := 2 * s - 1

/-- The total number of tiles on a square patio with given side length -/
def total_tiles_count (s : ℕ) : ℕ := s * s

/-- Theorem stating that if a square patio has 61 red tiles, it has 961 total tiles -/
theorem patio_tile_count (p : SquarePatio) (h : p.red_tiles = 61) :
  total_tiles_count p.side_length = 961 := by
  sorry

end NUMINAMATH_CALUDE_patio_tile_count_l1330_133004


namespace NUMINAMATH_CALUDE_successive_integers_product_l1330_133050

theorem successive_integers_product (n : ℤ) : 
  n * (n + 1) = 7832 → n = 88 := by
  sorry

end NUMINAMATH_CALUDE_successive_integers_product_l1330_133050


namespace NUMINAMATH_CALUDE_no_right_triangle_with_integer_side_l1330_133013

theorem no_right_triangle_with_integer_side : 
  ¬ ∃ (x : ℤ), 
    (12 < x ∧ x < 30) ∧ 
    (x^2 = 12^2 + 30^2 ∨ 30^2 = 12^2 + x^2 ∨ 12^2 = 30^2 + x^2) :=
by sorry

#check no_right_triangle_with_integer_side

end NUMINAMATH_CALUDE_no_right_triangle_with_integer_side_l1330_133013


namespace NUMINAMATH_CALUDE_translation_result_l1330_133063

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a translation in 2D space
structure Translation2D where
  dx : ℝ
  dy : ℝ

-- Define a function to apply a translation to a point
def applyTranslation (p : Point2D) (t : Translation2D) : Point2D :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_result :
  let A : Point2D := { x := -3, y := 2 }
  let right_translation : Translation2D := { dx := 4, dy := 0 }
  let down_translation : Translation2D := { dx := 0, dy := -3 }
  let A' := applyTranslation (applyTranslation A right_translation) down_translation
  A'.x = 1 ∧ A'.y = -1 := by
sorry

end NUMINAMATH_CALUDE_translation_result_l1330_133063


namespace NUMINAMATH_CALUDE_michaels_brother_money_l1330_133073

theorem michaels_brother_money (michael_initial : ℕ) (brother_initial : ℕ) (candy_cost : ℕ) : 
  michael_initial = 42 →
  brother_initial = 17 →
  candy_cost = 3 →
  brother_initial + michael_initial / 2 - candy_cost = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_michaels_brother_money_l1330_133073


namespace NUMINAMATH_CALUDE_brett_marbles_l1330_133053

theorem brett_marbles (red : ℕ) (blue : ℕ) : 
  blue = red + 24 → 
  blue = 5 * red → 
  red = 6 := by
  sorry

end NUMINAMATH_CALUDE_brett_marbles_l1330_133053


namespace NUMINAMATH_CALUDE_train_problem_l1330_133067

/-- The length of the longer train given the conditions of the problem -/
def longer_train_length : ℝ := 319.96

theorem train_problem (train1_length train1_speed train2_speed clearing_time : ℝ) 
  (h1 : train1_length = 160)
  (h2 : train1_speed = 42)
  (h3 : train2_speed = 30)
  (h4 : clearing_time = 23.998) : 
  longer_train_length = 319.96 := by
  sorry

#check train_problem

end NUMINAMATH_CALUDE_train_problem_l1330_133067


namespace NUMINAMATH_CALUDE_find_m_l1330_133064

def U : Set ℕ := {0, 1, 2, 3}

def A (m : ℝ) : Set ℕ := {x ∈ U | x^2 + m * x = 0}

theorem find_m : ∃ m : ℝ, (U \ A m = {1, 2}) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l1330_133064


namespace NUMINAMATH_CALUDE_circular_seating_pairs_l1330_133028

/-- The number of adjacent pairs in a circular seating arrangement --/
def adjacentPairs (n : ℕ) : ℕ := n

/-- Theorem: In a circular seating arrangement with n people,
    the number of different sets of two people sitting next to each other is n --/
theorem circular_seating_pairs (n : ℕ) (h : n > 0) :
  adjacentPairs n = n := by
  sorry

end NUMINAMATH_CALUDE_circular_seating_pairs_l1330_133028


namespace NUMINAMATH_CALUDE_friday_occurs_five_times_in_september_l1330_133076

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents months of the year -/
inductive Month
  | January
  | February
  | March
  | April
  | May
  | June
  | July
  | August
  | September
  | October
  | November
  | December

/-- Structure representing a year with specific properties -/
structure Year where
  julySundayCount : Nat
  februaryLeap : Bool
  septemberDayCount : Nat

/-- Function to determine the day that occurs five times in September -/
def dayOccurringFiveTimesInSeptember (y : Year) : DayOfWeek :=
  sorry

/-- Theorem stating that Friday occurs five times in September under given conditions -/
theorem friday_occurs_five_times_in_september (y : Year) 
    (h1 : y.julySundayCount = 5)
    (h2 : y.februaryLeap = true)
    (h3 : y.septemberDayCount = 30) :
    dayOccurringFiveTimesInSeptember y = DayOfWeek.Friday := by
  sorry


end NUMINAMATH_CALUDE_friday_occurs_five_times_in_september_l1330_133076


namespace NUMINAMATH_CALUDE_andrew_brought_40_chicken_nuggets_l1330_133020

/-- Represents the number of appetizer portions Andrew brought -/
def total_appetizers : ℕ := 90

/-- Represents the number of hotdogs on sticks Andrew brought -/
def hotdogs : ℕ := 30

/-- Represents the number of bite-sized cheese pops Andrew brought -/
def cheese_pops : ℕ := 20

/-- Represents the number of chicken nuggets Andrew brought -/
def chicken_nuggets : ℕ := total_appetizers - hotdogs - cheese_pops

/-- Theorem stating that Andrew brought 40 pieces of chicken nuggets -/
theorem andrew_brought_40_chicken_nuggets : chicken_nuggets = 40 := by
  sorry

end NUMINAMATH_CALUDE_andrew_brought_40_chicken_nuggets_l1330_133020


namespace NUMINAMATH_CALUDE_max_gcd_of_three_digit_numbers_l1330_133022

theorem max_gcd_of_three_digit_numbers :
  ∀ a b : ℕ,
  a ≠ b →
  a < 10 →
  b < 10 →
  (∃ (x y : ℕ), x = 100 * a + 11 * b ∧ y = 101 * b + 10 * a ∧ Nat.gcd x y ≤ 45) ∧
  (∃ (a' b' : ℕ), a' ≠ b' ∧ a' < 10 ∧ b' < 10 ∧
    Nat.gcd (100 * a' + 11 * b') (101 * b' + 10 * a') = 45) :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_of_three_digit_numbers_l1330_133022


namespace NUMINAMATH_CALUDE_simplify_rational_expression_l1330_133052

theorem simplify_rational_expression (x : ℝ) 
  (h1 : x ≠ 3) (h2 : x ≠ 4) (h3 : x ≠ 2) (h4 : x ≠ 5) :
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = 
  ((x - 1) * (x - 5)) / ((x - 4) * (x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_rational_expression_l1330_133052


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l1330_133018

theorem sqrt_equation_solutions (x : ℝ) : 
  (Real.sqrt (9 * x - 4) + 18 / Real.sqrt (9 * x - 4) = 10) ↔ (x = 85 / 9 ∨ x = 8 / 9) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l1330_133018


namespace NUMINAMATH_CALUDE_tangent_parallel_x_axis_tangent_parallel_line_l1330_133019

-- Define the curve
def x (t : ℝ) : ℝ := t - 1
def y (t : ℝ) : ℝ := t^3 - 12*t + 1

-- Define the derivative of y with respect to x
def dy_dx (t : ℝ) : ℝ := 3*t^2 - 12

-- Define the slope of the line 9x + y + 3 = 0
def m : ℝ := -9

-- Theorem for points where tangent is parallel to x-axis
theorem tangent_parallel_x_axis :
  ∃ t₁ t₂ : ℝ, 
    t₁ ≠ t₂ ∧
    dy_dx t₁ = 0 ∧ dy_dx t₂ = 0 ∧
    x t₁ = 1 ∧ y t₁ = -15 ∧
    x t₂ = -3 ∧ y t₂ = 17 :=
sorry

-- Theorem for points where tangent is parallel to 9x + y + 3 = 0
theorem tangent_parallel_line :
  ∃ t₁ t₂ : ℝ,
    t₁ ≠ t₂ ∧
    dy_dx t₁ = m ∧ dy_dx t₂ = m ∧
    x t₁ = 0 ∧ y t₁ = -10 ∧
    x t₂ = -2 ∧ y t₂ = 12 :=
sorry

end NUMINAMATH_CALUDE_tangent_parallel_x_axis_tangent_parallel_line_l1330_133019


namespace NUMINAMATH_CALUDE_reading_time_difference_l1330_133036

/-- Proves that given Xanthia's and Molly's reading speeds and a book's page count,
    the difference in reading time is 240 minutes. -/
theorem reading_time_difference
  (xanthia_speed : ℕ)
  (molly_speed : ℕ)
  (book_pages : ℕ)
  (h1 : xanthia_speed = 80)
  (h2 : molly_speed = 40)
  (h3 : book_pages = 320) :
  (book_pages / molly_speed - book_pages / xanthia_speed) * 60 = 240 :=
by sorry

end NUMINAMATH_CALUDE_reading_time_difference_l1330_133036


namespace NUMINAMATH_CALUDE_taxi_fare_formula_l1330_133069

/-- Represents the taxi fare function for distances greater than 3 km -/
def taxiFare (x : ℝ) : ℝ :=
  10 + 2 * (x - 3)

/-- Theorem stating that the taxi fare function is equivalent to 2x + 4 for x > 3 -/
theorem taxi_fare_formula (x : ℝ) (h : x > 3) :
  taxiFare x = 2 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_formula_l1330_133069


namespace NUMINAMATH_CALUDE_weeks_to_afford_laptop_l1330_133000

/-- The minimum number of whole weeks needed to afford a laptop -/
def weeks_needed (laptop_cost birthday_money weekly_earnings : ℕ) : ℕ :=
  (laptop_cost - birthday_money + weekly_earnings - 1) / weekly_earnings

/-- Proof that 34 weeks are needed to afford the laptop -/
theorem weeks_to_afford_laptop :
  weeks_needed 800 125 20 = 34 := by
  sorry

end NUMINAMATH_CALUDE_weeks_to_afford_laptop_l1330_133000


namespace NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_3_l1330_133088

theorem simplify_fraction_with_sqrt_3 :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_3_l1330_133088


namespace NUMINAMATH_CALUDE_simplify_fraction_l1330_133080

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1330_133080


namespace NUMINAMATH_CALUDE_sum_of_xyz_l1330_133025

theorem sum_of_xyz (x y z : ℕ+) 
  (h1 : x * y = 18)
  (h2 : x * z = 3)
  (h3 : y * z = 6) :
  x + y + z = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l1330_133025


namespace NUMINAMATH_CALUDE_teacher_worked_six_months_l1330_133097

/-- Calculates the number of months a teacher has worked based on given conditions -/
def teacher_months_worked (periods_per_day : ℕ) (days_per_month : ℕ) (pay_per_period : ℕ) (total_earned : ℕ) : ℕ :=
  let daily_earnings := periods_per_day * pay_per_period
  let monthly_earnings := daily_earnings * days_per_month
  total_earned / monthly_earnings

/-- Theorem stating that the teacher has worked for 6 months given the specified conditions -/
theorem teacher_worked_six_months :
  teacher_months_worked 5 24 5 3600 = 6 := by
  sorry

end NUMINAMATH_CALUDE_teacher_worked_six_months_l1330_133097


namespace NUMINAMATH_CALUDE_total_situps_is_110_l1330_133070

/-- The number of situps Diana did -/
def diana_situps : ℕ := 40

/-- The rate at which Diana did situps (situps per minute) -/
def diana_rate : ℕ := 4

/-- The difference in situps per minute between Hani and Diana -/
def hani_extra_rate : ℕ := 3

/-- Theorem stating that the total number of situps Hani and Diana did together is 110 -/
theorem total_situps_is_110 : 
  diana_situps + (diana_rate + hani_extra_rate) * (diana_situps / diana_rate) = 110 := by
  sorry

end NUMINAMATH_CALUDE_total_situps_is_110_l1330_133070


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l1330_133034

theorem simplify_sqrt_expression :
  (Real.sqrt 726 / Real.sqrt 72) - (Real.sqrt 294 / Real.sqrt 98) = Real.sqrt 10 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l1330_133034


namespace NUMINAMATH_CALUDE_kenneth_rowing_speed_l1330_133026

/-- Calculates the rowing speed of Kenneth given the race conditions -/
theorem kenneth_rowing_speed 
  (race_distance : ℝ) 
  (biff_speed : ℝ) 
  (kenneth_extra_distance : ℝ) 
  (h1 : race_distance = 500) 
  (h2 : biff_speed = 50) 
  (h3 : kenneth_extra_distance = 10) : 
  (race_distance + kenneth_extra_distance) / (race_distance / biff_speed) = 51 := by
  sorry

end NUMINAMATH_CALUDE_kenneth_rowing_speed_l1330_133026


namespace NUMINAMATH_CALUDE_assignment_count_is_correct_l1330_133042

/-- The number of ways to assign 4 people to 3 offices with at least one person in each office -/
def assignmentCount : ℕ := 36

/-- The number of people to be assigned -/
def numPeople : ℕ := 4

/-- The number of offices -/
def numOffices : ℕ := 3

theorem assignment_count_is_correct :
  assignmentCount = (numPeople.choose 2) * numOffices * 2 :=
sorry

end NUMINAMATH_CALUDE_assignment_count_is_correct_l1330_133042


namespace NUMINAMATH_CALUDE_domain_of_g_l1330_133095

-- Define the function f with domain (-1, 0)
def f : Set ℝ := { x : ℝ | -1 < x ∧ x < 0 }

-- Define the function g(x) = f(2x+1)
def g : Set ℝ := { x : ℝ | (2*x + 1) ∈ f }

-- Theorem statement
theorem domain_of_g : g = { x : ℝ | -1 < x ∧ x < -1/2 } := by sorry

end NUMINAMATH_CALUDE_domain_of_g_l1330_133095


namespace NUMINAMATH_CALUDE_vector_relations_l1330_133083

def vector_a : Fin 2 → ℝ := ![2, 3]
def vector_b (x : ℝ) : Fin 2 → ℝ := ![x, -6]

def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (∀ i, u i = k * v i)

def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) = 0

theorem vector_relations :
  (∃ x : ℝ, parallel vector_a (vector_b x) ↔ x = -4) ∧
  (∃ x : ℝ, perpendicular vector_a (vector_b x) ↔ x = 9) := by
  sorry

end NUMINAMATH_CALUDE_vector_relations_l1330_133083


namespace NUMINAMATH_CALUDE_gold_coins_percentage_l1330_133045

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  beads_percent : ℝ
  papers_percent : ℝ
  silver_coins_percent : ℝ
  gold_coins_percent : ℝ

/-- Theorem stating the percentage of gold coins in the urn -/
theorem gold_coins_percentage (u : UrnComposition) 
  (h1 : u.beads_percent = 15)
  (h2 : u.papers_percent = 10)
  (h3 : u.silver_coins_percent + u.gold_coins_percent = 75)
  (h4 : u.silver_coins_percent = 0.3 * 75) :
  u.gold_coins_percent = 52.5 := by
  sorry

#check gold_coins_percentage

end NUMINAMATH_CALUDE_gold_coins_percentage_l1330_133045


namespace NUMINAMATH_CALUDE_vector_BA_l1330_133054

def complex_vector (a b : ℂ) : ℂ := a - b

theorem vector_BA (OA OB : ℂ) :
  OA = 2 - 3*I ∧ OB = -3 + 2*I →
  complex_vector OA OB = 5 - 5*I :=
by sorry

end NUMINAMATH_CALUDE_vector_BA_l1330_133054


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l1330_133084

theorem polygon_sides_from_angle_sum (n : ℕ) (angle_sum : ℝ) : 
  angle_sum = 900 → (n - 2) * 180 = angle_sum → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l1330_133084


namespace NUMINAMATH_CALUDE_fraction_simplification_implies_even_difference_l1330_133041

theorem fraction_simplification_implies_even_difference 
  (a b c d : ℕ) (h1 : ∀ n : ℕ, c * n + d ≠ 0) 
  (h2 : ∀ n : ℕ, ∃ k : ℕ, a * n + b = 2 * k ∧ c * n + d = 2 * k) : 
  Even (a * d - b * c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_implies_even_difference_l1330_133041


namespace NUMINAMATH_CALUDE_homework_completion_l1330_133016

/-- Fraction of homework done on Monday night -/
def monday_fraction : ℚ := sorry

/-- Fraction of homework done on Tuesday night -/
def tuesday_fraction (x : ℚ) : ℚ := (1 - x) / 3

/-- Fraction of homework done on Wednesday night -/
def wednesday_fraction : ℚ := 4 / 15

theorem homework_completion (x : ℚ) :
  x + tuesday_fraction x + wednesday_fraction = 1 →
  x = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_homework_completion_l1330_133016


namespace NUMINAMATH_CALUDE_CH4_required_for_CCl4_l1330_133047

-- Define the chemical species as real numbers (representing moles)
variable (CH4 CH2Cl2 CCl4 CHCl3 HCl Cl2 CH3Cl : ℝ)

-- Define the equilibrium constants
def K1 : ℝ := 1.2 * 10^2
def K2 : ℝ := 1.5 * 10^3
def K3 : ℝ := 3.4 * 10^4

-- Define the initial amounts of species
def initial_CH2Cl2 : ℝ := 2.5
def initial_CHCl3 : ℝ := 1.5
def initial_HCl : ℝ := 0.5
def initial_Cl2 : ℝ := 10
def initial_CH3Cl : ℝ := 0.2

-- Define the target amount of CCl4
def target_CCl4 : ℝ := 5

-- Theorem statement
theorem CH4_required_for_CCl4 :
  ∃ (required_CH4 : ℝ),
    required_CH4 = 2.5 ∧
    required_CH4 + initial_CH2Cl2 = target_CCl4 :=
sorry

end NUMINAMATH_CALUDE_CH4_required_for_CCl4_l1330_133047


namespace NUMINAMATH_CALUDE_range_of_m_l1330_133017

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (2*x + 5)/3 - 1 ≤ 2 - x → 3*(x - 1) + 5 > 5*x + 2*(m + x)) → 
  m < -3/5 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1330_133017


namespace NUMINAMATH_CALUDE_standard_deviation_proof_l1330_133038

/-- The standard deviation of a test score distribution. -/
def standard_deviation : ℝ := 20

/-- The mean score of the test. -/
def mean_score : ℝ := 60

/-- The lowest possible score within 2 standard deviations of the mean. -/
def lowest_score : ℝ := 20

/-- Theorem stating that the standard deviation is correct given the conditions. -/
theorem standard_deviation_proof :
  lowest_score = mean_score - 2 * standard_deviation :=
by sorry

end NUMINAMATH_CALUDE_standard_deviation_proof_l1330_133038


namespace NUMINAMATH_CALUDE_shortest_distance_parabola_line_l1330_133074

/-- The shortest distance between a point on the parabola y = x^2 - 6x + 15 
    and a point on the line y = 2x - 7 -/
theorem shortest_distance_parabola_line : 
  let parabola := fun x : ℝ => x^2 - 6*x + 15
  let line := fun x : ℝ => 2*x - 7
  ∃ (min_dist : ℝ), 
    (∀ (p q : ℝ × ℝ), 
      (p.2 = parabola p.1) → 
      (q.2 = line q.1) → 
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ min_dist) ∧
    (∃ (p q : ℝ × ℝ), 
      (p.2 = parabola p.1) ∧ 
      (q.2 = line q.1) ∧ 
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = min_dist) ∧
    min_dist = 6 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_shortest_distance_parabola_line_l1330_133074


namespace NUMINAMATH_CALUDE_unique_positive_integer_l1330_133001

theorem unique_positive_integer : ∃! (x : ℕ), x > 0 ∧ (4 * x)^2 - 2 * x = 2652 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_l1330_133001


namespace NUMINAMATH_CALUDE_total_chips_count_l1330_133014

def plain_chips : ℕ := 4
def bbq_chips : ℕ := 5
def probability_3_bbq : ℚ := 5/42

theorem total_chips_count : 
  let total_chips := plain_chips + bbq_chips
  (Nat.choose bbq_chips 3 : ℚ) / (Nat.choose total_chips 3 : ℚ) = probability_3_bbq →
  total_chips = 9 := by sorry

end NUMINAMATH_CALUDE_total_chips_count_l1330_133014


namespace NUMINAMATH_CALUDE_study_group_probability_l1330_133094

/-- Given a study group where 70% of members are women and 40% of women are lawyers,
    the probability of randomly selecting a woman lawyer is 0.28. -/
theorem study_group_probability (total : ℕ) (women : ℕ) (women_lawyers : ℕ)
    (h1 : women = (70 : ℕ) * total / 100)
    (h2 : women_lawyers = (40 : ℕ) * women / 100) :
    (women_lawyers : ℚ) / total = 28 / 100 := by
  sorry

end NUMINAMATH_CALUDE_study_group_probability_l1330_133094


namespace NUMINAMATH_CALUDE_power_zero_eq_one_l1330_133048

theorem power_zero_eq_one (a b : ℝ) (h : a - b ≠ 0) : (a - b)^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_l1330_133048


namespace NUMINAMATH_CALUDE_total_hockey_games_l1330_133092

/-- The number of hockey games in a season -/
def hockey_season_games (games_per_month : ℕ) (months_in_season : ℕ) : ℕ :=
  games_per_month * months_in_season

/-- Theorem: The total number of hockey games in the season is 182 -/
theorem total_hockey_games : hockey_season_games 13 14 = 182 := by
  sorry

end NUMINAMATH_CALUDE_total_hockey_games_l1330_133092


namespace NUMINAMATH_CALUDE_percentage_students_taking_music_l1330_133030

/-- The percentage of students taking music, given the total number of students
    and the number of students taking dance and art. -/
theorem percentage_students_taking_music
  (total_students : ℕ)
  (dance_students : ℕ)
  (art_students : ℕ)
  (h1 : total_students = 400)
  (h2 : dance_students = 120)
  (h3 : art_students = 200) :
  (((total_students - dance_students - art_students) : ℚ) / total_students) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_students_taking_music_l1330_133030


namespace NUMINAMATH_CALUDE_simplify_cube_root_exponent_sum_l1330_133033

theorem simplify_cube_root_exponent_sum (a b c : ℝ) : 
  ∃ (k : ℝ) (x y z : ℕ), 
    (∀ t : ℝ, t > 0 → (k * a^x * b^y * c^z)^3 * t = 40 * a^6 * b^9 * c^14) ∧ 
    x + y + z = 7 :=
sorry

end NUMINAMATH_CALUDE_simplify_cube_root_exponent_sum_l1330_133033


namespace NUMINAMATH_CALUDE_optimal_plan_is_best_three_valid_plans_l1330_133061

/-- Represents a purchasing plan for machines --/
structure PurchasePlan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a purchase plan is valid according to the given conditions --/
def isValidPlan (p : PurchasePlan) : Prop :=
  p.typeA + p.typeB = 6 ∧
  7 * p.typeA + 5 * p.typeB ≤ 34 ∧
  100 * p.typeA + 60 * p.typeB ≥ 380

/-- Calculates the total cost of a purchase plan --/
def totalCost (p : PurchasePlan) : ℕ :=
  7 * p.typeA + 5 * p.typeB

/-- The optimal purchase plan --/
def optimalPlan : PurchasePlan :=
  { typeA := 1, typeB := 5 }

/-- Theorem stating that the optimal plan is valid and minimizes cost --/
theorem optimal_plan_is_best :
  isValidPlan optimalPlan ∧
  ∀ p : PurchasePlan, isValidPlan p → totalCost optimalPlan ≤ totalCost p :=
sorry

/-- Theorem stating that there are exactly 3 valid purchase plans --/
theorem three_valid_plans :
  ∃! (plans : List PurchasePlan),
    plans.length = 3 ∧
    ∀ p : PurchasePlan, isValidPlan p ↔ p ∈ plans :=
sorry

end NUMINAMATH_CALUDE_optimal_plan_is_best_three_valid_plans_l1330_133061


namespace NUMINAMATH_CALUDE_record_storage_cost_l1330_133003

-- Define the box dimensions
def box_length : ℝ := 15
def box_width : ℝ := 12
def box_height : ℝ := 10

-- Define the total occupied space in cubic inches
def total_space : ℝ := 1080000

-- Define the storage cost per box per month
def cost_per_box : ℝ := 0.5

-- Theorem to prove
theorem record_storage_cost :
  let box_volume : ℝ := box_length * box_width * box_height
  let num_boxes : ℝ := total_space / box_volume
  let total_cost : ℝ := num_boxes * cost_per_box
  total_cost = 300 := by
sorry


end NUMINAMATH_CALUDE_record_storage_cost_l1330_133003


namespace NUMINAMATH_CALUDE_fifteen_students_in_neither_l1330_133086

/-- Represents the number of students in different categories of a robotics club. -/
structure RoboticsClub where
  total : ℕ
  cs : ℕ
  electronics : ℕ
  both : ℕ

/-- Calculates the number of students taking neither computer science nor electronics. -/
def studentsInNeither (club : RoboticsClub) : ℕ :=
  club.total - (club.cs + club.electronics - club.both)

/-- Theorem stating that 15 students take neither computer science nor electronics. -/
theorem fifteen_students_in_neither (club : RoboticsClub)
  (h1 : club.total = 80)
  (h2 : club.cs = 52)
  (h3 : club.electronics = 38)
  (h4 : club.both = 25) :
  studentsInNeither club = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_students_in_neither_l1330_133086


namespace NUMINAMATH_CALUDE_last_two_digits_of_2006_factorial_l1330_133039

theorem last_two_digits_of_2006_factorial (n : ℕ) (h : n = 2006) : n! % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_2006_factorial_l1330_133039


namespace NUMINAMATH_CALUDE_special_line_equation_l1330_133007

/-- A line passing through a point and intersecting coordinate axes at points with negative reciprocal intercepts -/
structure SpecialLine where
  a : ℝ × ℝ  -- The point A that the line passes through
  eq : ℝ → ℝ → Prop  -- The equation of the line

/-- The condition for the line to have negative reciprocal intercepts -/
def hasNegativeReciprocalIntercepts (l : SpecialLine) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (l.eq k 0 ∧ l.eq 0 (-k) ∨ l.eq (-k) 0 ∧ l.eq 0 k)

/-- The main theorem stating the equation of the special line -/
theorem special_line_equation (l : SpecialLine) 
    (h1 : l.a = (5, 2))
    (h2 : hasNegativeReciprocalIntercepts l) :
    (∀ x y, l.eq x y ↔ 2*x - 5*y = -8) ∨
    (∀ x y, l.eq x y ↔ x - y = 3) := by
  sorry


end NUMINAMATH_CALUDE_special_line_equation_l1330_133007


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1330_133002

theorem inequality_system_solution :
  {x : ℝ | (5 * x + 3 > 3 * (x - 1)) ∧ ((8 * x + 2) / 9 > x)} = {x : ℝ | -3 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1330_133002


namespace NUMINAMATH_CALUDE_unbroken_seashells_l1330_133024

theorem unbroken_seashells (total : ℕ) (broken : ℕ) (h1 : total = 7) (h2 : broken = 4) :
  total - broken = 3 := by
  sorry

end NUMINAMATH_CALUDE_unbroken_seashells_l1330_133024


namespace NUMINAMATH_CALUDE_catch_up_time_l1330_133068

/-- Two people walk in opposite directions at the same speed for 10 minutes,
    then one increases speed by 5 times and chases the other. -/
theorem catch_up_time (s : ℝ) (h : s > 0) : 
  let initial_distance := 2 * 10 * s
  let relative_speed := 5 * s - s
  initial_distance / relative_speed = 5 :=
by sorry

end NUMINAMATH_CALUDE_catch_up_time_l1330_133068


namespace NUMINAMATH_CALUDE_two_digit_number_sum_l1330_133008

theorem two_digit_number_sum (a b : ℕ) : 
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  (10 * a + b) - (10 * b + a) = 3 * (a + b) →
  (10 * a + b) + (10 * b + a) = 33 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_sum_l1330_133008


namespace NUMINAMATH_CALUDE_chemists_sons_ages_l1330_133081

theorem chemists_sons_ages (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive integers
  a * b * c = 36 →  -- product is 36
  a + b + c = 13 →  -- sum is 13
  (a ≥ b ∧ a ≥ c) ∨ (b ≥ a ∧ b ≥ c) ∨ (c ≥ a ∧ c ≥ b) →  -- unique oldest son
  (a = 2 ∧ b = 2 ∧ c = 9) ∨ (a = 2 ∧ b = 9 ∧ c = 2) ∨ (a = 9 ∧ b = 2 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_chemists_sons_ages_l1330_133081


namespace NUMINAMATH_CALUDE_race_outcomes_eq_210_l1330_133072

/-- The number of participants in the race -/
def num_participants : ℕ := 7

/-- The number of podium positions (1st, 2nd, 3rd) -/
def podium_positions : ℕ := 3

/-- Calculates the number of permutations of k elements from n elements -/
def permutations (n k : ℕ) : ℕ :=
  if k > n then 0
  else List.range k |>.foldl (fun acc i => acc * (n - i)) 1

/-- The number of different 1st-2nd-3rd place outcomes in a race with no ties -/
def race_outcomes : ℕ := permutations num_participants podium_positions

/-- Theorem: The number of different 1st-2nd-3rd place outcomes in a race
    with 7 participants and no ties is equal to 210 -/
theorem race_outcomes_eq_210 : race_outcomes = 210 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_eq_210_l1330_133072


namespace NUMINAMATH_CALUDE_f_2002_equals_96_l1330_133065

/-- A function satisfying the given property -/
def special_function (f : ℕ → ℝ) : Prop :=
  ∀ (a b n : ℕ), a > 0 → b > 0 → n > 0 → a + b = 2^n → f a + f b = n^2

/-- The theorem to be proved -/
theorem f_2002_equals_96 (f : ℕ → ℝ) (h : special_function f) : f 2002 = 96 := by
  sorry

end NUMINAMATH_CALUDE_f_2002_equals_96_l1330_133065


namespace NUMINAMATH_CALUDE_system_solution_l1330_133062

theorem system_solution : 
  ∃! (x y z : ℝ), 
    x * (y + z) * (x + y + z) = 1170 ∧ 
    y * (z + x) * (x + y + z) = 1008 ∧ 
    z * (x + y) * (x + y + z) = 1458 ∧ 
    x = 5 ∧ y = 4 ∧ z = 9 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1330_133062


namespace NUMINAMATH_CALUDE_sector_central_angle_l1330_133040

/-- Given a circular sector with circumference 10 and area 4, 
    prove that its central angle in radians is 1/2 -/
theorem sector_central_angle (r l : ℝ) : 
  (2 * r + l = 10) →  -- circumference condition
  ((1 / 2) * l * r = 4) →  -- area condition
  (l / r = 1 / 2) :=  -- central angle in radians
by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1330_133040


namespace NUMINAMATH_CALUDE_modulus_of_fraction_l1330_133077

def z : ℂ := -1 + Complex.I

theorem modulus_of_fraction : Complex.abs ((z + 3) / (z + 2)) = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_fraction_l1330_133077


namespace NUMINAMATH_CALUDE_min_mines_is_23_l1330_133057

/-- Represents the state of a square in the minesweeper grid -/
inductive SquareState
  | Unopened
  | Opened (n : Nat)

/-- Represents the minesweeper grid -/
def MinesweeperGrid := Matrix (Fin 11) (Fin 13) SquareState

/-- Checks if a given position is valid on the grid -/
def isValidPosition (row : Fin 11) (col : Fin 13) : Bool := true

/-- Returns the number of mines in the neighboring squares -/
def neighboringMines (grid : MinesweeperGrid) (row : Fin 11) (col : Fin 13) : Nat :=
  sorry

/-- Checks if the grid satisfies all opened square conditions -/
def satisfiesConditions (grid : MinesweeperGrid) : Prop :=
  sorry

/-- Counts the total number of mines in the grid -/
def countMines (grid : MinesweeperGrid) : Nat :=
  sorry

/-- The specific minesweeper grid layout from the problem -/
def problemGrid : MinesweeperGrid :=
  sorry

/-- Theorem stating that the minimum number of mines is 23 -/
theorem min_mines_is_23 :
  ∀ (grid : MinesweeperGrid),
    satisfiesConditions grid →
    grid = problemGrid →
    countMines grid ≥ 23 ∧
    ∃ (minGrid : MinesweeperGrid),
      satisfiesConditions minGrid ∧
      minGrid = problemGrid ∧
      countMines minGrid = 23 :=
sorry

end NUMINAMATH_CALUDE_min_mines_is_23_l1330_133057


namespace NUMINAMATH_CALUDE_larger_group_size_l1330_133011

/-- Given that 36 men can complete a piece of work in 18 days, and a larger group
    of men can complete the same work in 6 days, prove that the larger group
    consists of 108 men. -/
theorem larger_group_size (work : ℕ) (small_group : ℕ) (large_group : ℕ)
    (small_days : ℕ) (large_days : ℕ)
    (h1 : small_group = 36)
    (h2 : small_days = 18)
    (h3 : large_days = 6)
    (h4 : small_group * small_days = work)
    (h5 : large_group * large_days = work) :
    large_group = 108 := by
  sorry

#check larger_group_size

end NUMINAMATH_CALUDE_larger_group_size_l1330_133011


namespace NUMINAMATH_CALUDE_half_radius_circle_y_l1330_133006

-- Define the circles
def circle_x : Real → Prop := λ r => 2 * Real.pi * r = 14 * Real.pi
def circle_y : Real → Prop := λ r => True  -- We don't have specific information about y's circumference

-- Theorem statement
theorem half_radius_circle_y : 
  ∃ (rx ry : Real), 
    circle_x rx ∧ 
    circle_y ry ∧ 
    (Real.pi * rx^2 = Real.pi * ry^2) ∧  -- Same area
    (ry / 2 = 3.5) := by
  sorry

end NUMINAMATH_CALUDE_half_radius_circle_y_l1330_133006


namespace NUMINAMATH_CALUDE_units_digit_G_1000_l1330_133046

def G (n : ℕ) : ℕ := 2^(3^n) + 1

theorem units_digit_G_1000 : G 1000 % 10 = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_G_1000_l1330_133046


namespace NUMINAMATH_CALUDE_consecutive_naturals_properties_l1330_133058

theorem consecutive_naturals_properties (n k : ℕ) (h : k > 0) :
  (∃ m ∈ Finset.range k, 2 ∣ (n + m)) ∧ 
  (k % 2 = 0 → 2 ∣ (k * n + k * (k - 1) / 2)) :=
sorry

end NUMINAMATH_CALUDE_consecutive_naturals_properties_l1330_133058


namespace NUMINAMATH_CALUDE_scientific_notation_3080000_l1330_133032

theorem scientific_notation_3080000 :
  (3080000 : ℝ) = 3.08 * (10 ^ 6) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_3080000_l1330_133032


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1330_133071

/-- The complex number z = (2-i)/(1+i) is located in the fourth quadrant of the complex plane. -/
theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (2 - Complex.I) / (1 + Complex.I)
  (z.re > 0) ∧ (z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1330_133071


namespace NUMINAMATH_CALUDE_problem_solution_l1330_133085

theorem problem_solution (a b c d x y : ℝ) 
  (h1 : a = -b)
  (h2 : c * d = 1)
  (h3 : (x + 3)^2 + |y - 2| = 0) :
  2*(a + b) - 2*(c*d)^4 + (x + y)^2022 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1330_133085


namespace NUMINAMATH_CALUDE_probability_is_half_l1330_133098

/-- The probability of drawing either a red or blue marble from a bag -/
def probability_red_or_blue (red : ℕ) (blue : ℕ) (yellow : ℕ) : ℚ :=
  (red + blue : ℚ) / (red + blue + yellow)

/-- Theorem: The probability of drawing either a red or blue marble
    from a bag containing 3 red, 2 blue, and 5 yellow marbles is 1/2 -/
theorem probability_is_half :
  probability_red_or_blue 3 2 5 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_half_l1330_133098


namespace NUMINAMATH_CALUDE_bicycle_profit_percentage_l1330_133096

/-- Profit percentage calculation for bicycle sale --/
theorem bicycle_profit_percentage
  (cost_price_A : ℝ)
  (profit_percentage_A : ℝ)
  (final_price : ℝ)
  (h1 : cost_price_A = 144)
  (h2 : profit_percentage_A = 25)
  (h3 : final_price = 225) :
  let selling_price_A := cost_price_A * (1 + profit_percentage_A / 100)
  let profit_B := final_price - selling_price_A
  let profit_percentage_B := (profit_B / selling_price_A) * 100
  profit_percentage_B = 25 := by sorry

end NUMINAMATH_CALUDE_bicycle_profit_percentage_l1330_133096


namespace NUMINAMATH_CALUDE_subtract_fractions_l1330_133056

theorem subtract_fractions : (3 : ℚ) / 4 - (1 : ℚ) / 6 = (7 : ℚ) / 12 := by sorry

end NUMINAMATH_CALUDE_subtract_fractions_l1330_133056


namespace NUMINAMATH_CALUDE_line_graph_most_suitable_l1330_133090

/-- Represents different types of statistical graphs -/
inductive StatisticalGraph
| BarGraph
| PieChart
| LineGraph
| FrequencyDistributionGraph

/-- Characteristics of a statistical graph -/
structure GraphCharacteristics where
  showsTrend : Bool
  showsTimeProgression : Bool
  comparesCategories : Bool
  showsProportions : Bool
  showsFrequency : Bool

/-- Define the characteristics of each graph type -/
def graphProperties : StatisticalGraph → GraphCharacteristics
| StatisticalGraph.BarGraph => {
    showsTrend := false,
    showsTimeProgression := false,
    comparesCategories := true,
    showsProportions := false,
    showsFrequency := false
  }
| StatisticalGraph.PieChart => {
    showsTrend := false,
    showsTimeProgression := false,
    comparesCategories := false,
    showsProportions := true,
    showsFrequency := false
  }
| StatisticalGraph.LineGraph => {
    showsTrend := true,
    showsTimeProgression := true,
    comparesCategories := false,
    showsProportions := false,
    showsFrequency := false
  }
| StatisticalGraph.FrequencyDistributionGraph => {
    showsTrend := false,
    showsTimeProgression := false,
    comparesCategories := false,
    showsProportions := false,
    showsFrequency := true
  }

/-- Defines the requirements for a graph to show temperature trends over a week -/
def suitableForTemperatureTrend (g : GraphCharacteristics) : Prop :=
  g.showsTrend ∧ g.showsTimeProgression

/-- Theorem stating that a line graph is the most suitable for showing temperature trends over a week -/
theorem line_graph_most_suitable :
  ∀ (g : StatisticalGraph), 
    suitableForTemperatureTrend (graphProperties g) → g = StatisticalGraph.LineGraph := by
  sorry

end NUMINAMATH_CALUDE_line_graph_most_suitable_l1330_133090


namespace NUMINAMATH_CALUDE_symmetric_point_complex_l1330_133044

def symmetric_about_imaginary_axis (z : ℂ) : ℂ := -Complex.re z + Complex.im z * Complex.I

theorem symmetric_point_complex : 
  let A : ℂ := 2 + Complex.I
  let B : ℂ := symmetric_about_imaginary_axis A
  B = -2 + Complex.I := by
sorry

end NUMINAMATH_CALUDE_symmetric_point_complex_l1330_133044


namespace NUMINAMATH_CALUDE_monomial_sum_implies_m_plus_n_eq_3_l1330_133089

/-- Two algebraic expressions form a monomial when added together if they have the same powers for each variable -/
def forms_monomial (expr1 expr2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ (x y : ℕ), expr1 x y ≠ 0 ∧ expr2 x y ≠ 0 → x = y

/-- The first algebraic expression: 3a^m * b^2 -/
def expr1 (m : ℕ) (a b : ℕ) : ℚ := 3 * (a^m) * (b^2)

/-- The second algebraic expression: -2a^2 * b^(n+1) -/
def expr2 (n : ℕ) (a b : ℕ) : ℚ := -2 * (a^2) * (b^(n+1))

theorem monomial_sum_implies_m_plus_n_eq_3 (m n : ℕ) :
  forms_monomial (expr1 m) (expr2 n) → m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_monomial_sum_implies_m_plus_n_eq_3_l1330_133089


namespace NUMINAMATH_CALUDE_min_value_a_l1330_133082

theorem min_value_a (a b c d : ℕ+) 
  (h1 : a > b ∧ b > c ∧ c > d)
  (h2 : a + b + c + d = 2004)
  (h3 : a^2 - b^2 + c^2 - d^2 = 2004) :
  a ≥ 503 ∧ ∃ (a₀ b₀ c₀ d₀ : ℕ+), 
    a₀ = 503 ∧ 
    a₀ > b₀ ∧ b₀ > c₀ ∧ c₀ > d₀ ∧
    a₀ + b₀ + c₀ + d₀ = 2004 ∧
    a₀^2 - b₀^2 + c₀^2 - d₀^2 = 2004 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l1330_133082


namespace NUMINAMATH_CALUDE_divisibility_of_square_l1330_133010

theorem divisibility_of_square (n : ℕ) (h1 : n > 0) (h2 : ∀ d : ℕ, d > 0 → d ∣ n → d ≤ 30) :
  900 ∣ n^2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_square_l1330_133010


namespace NUMINAMATH_CALUDE_cube_root_sum_l1330_133037

theorem cube_root_sum (u v w : ℝ) : 
  (∃ x y z : ℝ, x^3 = 8 ∧ y^3 = 27 ∧ z^3 = 64 ∧
   (u - x) * (u - y) * (u - z) = 1/2 ∧
   (v - x) * (v - y) * (v - z) = 1/2 ∧
   (w - x) * (w - y) * (w - z) = 1/2 ∧
   u ≠ v ∧ u ≠ w ∧ v ≠ w) →
  u^3 + v^3 + w^3 = -42 := by
sorry

end NUMINAMATH_CALUDE_cube_root_sum_l1330_133037


namespace NUMINAMATH_CALUDE_negation_unique_solution_equivalence_l1330_133049

theorem negation_unique_solution_equivalence :
  ¬(∀ a : ℝ, ∃! x : ℝ, a * x + 1 = 0) ↔
  (∃ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ a * x + 1 = 0 ∧ a * y + 1 = 0) ∨ (∀ x : ℝ, a * x + 1 ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_unique_solution_equivalence_l1330_133049


namespace NUMINAMATH_CALUDE_greatest_common_divisor_630_90_under_35_l1330_133021

theorem greatest_common_divisor_630_90_under_35 : 
  ∀ n : ℕ, n ∣ 630 ∧ n < 35 ∧ n ∣ 90 → n ≤ 30 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_630_90_under_35_l1330_133021


namespace NUMINAMATH_CALUDE_algorithm_output_l1330_133043

def algorithm (n : ℕ) : ℤ :=
  let init := (0 : ℤ)
  init - 3 * n

theorem algorithm_output : algorithm 3 = -9 := by
  sorry

end NUMINAMATH_CALUDE_algorithm_output_l1330_133043


namespace NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l1330_133009

theorem only_one_divides_power_minus_one : 
  ∀ n : ℕ, n > 0 → n ∣ (2^n - 1) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l1330_133009


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l1330_133023

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A * B * C = 1764 →
  A + B + C ≤ 33 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l1330_133023


namespace NUMINAMATH_CALUDE_acute_triangle_contains_grid_point_l1330_133005

-- Define a point on a graph paper grid
structure GridPoint where
  x : ℤ
  y : ℤ

-- Define a triangle on a graph paper grid
structure GridTriangle where
  A : GridPoint
  B : GridPoint
  C : GridPoint

-- Define what it means for a triangle to be acute
def isAcute (t : GridTriangle) : Prop :=
  sorry -- Definition of acute triangle on a grid

-- Define what it means for a point to be inside or on the sides of a triangle
def isInsideOrOnSides (p : GridPoint) (t : GridTriangle) : Prop :=
  sorry -- Definition of a point being inside or on the sides of a triangle

-- The main theorem
theorem acute_triangle_contains_grid_point (t : GridTriangle) :
  isAcute t →
  ∃ p : GridPoint, p ≠ t.A ∧ p ≠ t.B ∧ p ≠ t.C ∧ isInsideOrOnSides p t :=
sorry

end NUMINAMATH_CALUDE_acute_triangle_contains_grid_point_l1330_133005


namespace NUMINAMATH_CALUDE_set_operations_l1330_133075

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | -2 < x ∧ x < 3}

def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

theorem set_operations :
  (A ∩ B = {x | -2 < x ∧ x ≤ 2}) ∧
  ((Set.compl A) ∪ B = {x | x ≤ 2 ∨ x ≥ 3}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l1330_133075


namespace NUMINAMATH_CALUDE_chord_count_l1330_133029

/-- The number of chords formed by connecting any two of n points on a circle's circumference -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- There are 9 points on the circumference of a circle -/
def num_points : ℕ := 9

theorem chord_count : num_chords num_points = 36 := by
  sorry

end NUMINAMATH_CALUDE_chord_count_l1330_133029


namespace NUMINAMATH_CALUDE_percentage_of_percentage_l1330_133055

theorem percentage_of_percentage (y : ℝ) (h : y ≠ 0) :
  (30 / 100) * (60 / 100) * y = (18 / 100) * y :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_percentage_l1330_133055


namespace NUMINAMATH_CALUDE_largest_negative_congruent_to_two_mod_seventeen_l1330_133060

theorem largest_negative_congruent_to_two_mod_seventeen :
  ∀ n : ℤ, -9999 ≤ n ∧ n < -999 ∧ n ≡ 2 [ZMOD 17] → n ≤ -1001 :=
by sorry

end NUMINAMATH_CALUDE_largest_negative_congruent_to_two_mod_seventeen_l1330_133060


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1330_133035

theorem sphere_surface_area (C : ℝ) (h : C = 4 * Real.pi) :
  ∃ (S : ℝ), S = 16 * Real.pi ∧ S = 4 * Real.pi * (C / (2 * Real.pi))^2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1330_133035


namespace NUMINAMATH_CALUDE_toys_remaining_l1330_133078

theorem toys_remaining (initial_stock : ℕ) (sold_week1 : ℕ) (sold_week2 : ℕ) 
  (h1 : initial_stock = 83) 
  (h2 : sold_week1 = 38) 
  (h3 : sold_week2 = 26) :
  initial_stock - (sold_week1 + sold_week2) = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_toys_remaining_l1330_133078


namespace NUMINAMATH_CALUDE_election_winner_votes_l1330_133079

theorem election_winner_votes (total_votes : ℝ) (winner_votes : ℝ) : 
  (winner_votes = 0.62 * total_votes) →
  (winner_votes - (total_votes - winner_votes) = 384) →
  (winner_votes = 992) :=
by
  sorry

end NUMINAMATH_CALUDE_election_winner_votes_l1330_133079


namespace NUMINAMATH_CALUDE_expression_evaluation_l1330_133051

theorem expression_evaluation : (1 + 2 + 3) * (1 + 1/2 + 1/3) = 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1330_133051


namespace NUMINAMATH_CALUDE_unique_solution_3x_4y_5z_l1330_133091

theorem unique_solution_3x_4y_5z :
  ∀ x y z : ℕ+, 3^(x : ℕ) + 4^(y : ℕ) = 5^(z : ℕ) → x = 2 ∧ y = 2 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_3x_4y_5z_l1330_133091


namespace NUMINAMATH_CALUDE_spinner_direction_l1330_133015

-- Define the four cardinal directions
inductive Direction
  | North
  | East
  | South
  | West

-- Define a function to rotate a direction
def rotate (d : Direction) (revolutions : ℚ) : Direction :=
  match (revolutions % 1).num.mod 4 with
  | 0 => d
  | 1 => match d with
    | Direction.North => Direction.East
    | Direction.East => Direction.South
    | Direction.South => Direction.West
    | Direction.West => Direction.North
  | 2 => match d with
    | Direction.North => Direction.South
    | Direction.East => Direction.West
    | Direction.South => Direction.North
    | Direction.West => Direction.East
  | 3 => match d with
    | Direction.North => Direction.West
    | Direction.East => Direction.North
    | Direction.South => Direction.East
    | Direction.West => Direction.South
  | _ => d  -- This case should never occur due to mod 4

theorem spinner_direction :
  let initial_direction := Direction.North
  let clockwise_rotation := 7/2  -- 3.5 revolutions
  let counterclockwise_rotation := 7/4  -- 1.75 revolutions
  let final_direction := rotate (rotate initial_direction clockwise_rotation) (-counterclockwise_rotation)
  final_direction = Direction.West := by
  sorry

end NUMINAMATH_CALUDE_spinner_direction_l1330_133015


namespace NUMINAMATH_CALUDE_tangent_circle_center_slope_l1330_133093

-- Define the circles u₁ and u₂
def u₁ (x y : ℝ) : Prop := x^2 + y^2 + 8*x - 20*y - 32 = 0
def u₂ (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 20*y + 128 = 0

-- Define the condition for a point (x, y) to be on the line y = bx
def on_line (x y b : ℝ) : Prop := y = b * x

-- Define the condition for a circle to be externally tangent to u₁
def externally_tangent_u₁ (x y r : ℝ) : Prop :=
  r + 12 = Real.sqrt ((x + 4)^2 + (y - 10)^2)

-- Define the condition for a circle to be internally tangent to u₂
def internally_tangent_u₂ (x y r : ℝ) : Prop :=
  8 - r = Real.sqrt ((x - 4)^2 + (y - 10)^2)

-- State the theorem
theorem tangent_circle_center_slope :
  ∃ n : ℝ, 
    (∀ b : ℝ, b > 0 → 
      (∃ x y r : ℝ, 
        on_line x y b ∧ 
        externally_tangent_u₁ x y r ∧ 
        internally_tangent_u₂ x y r) → 
      n ≤ b) ∧
    n^2 = 69/25 := by sorry

end NUMINAMATH_CALUDE_tangent_circle_center_slope_l1330_133093


namespace NUMINAMATH_CALUDE_rectangular_garden_width_l1330_133066

theorem rectangular_garden_width (width length area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 675 →
  width = 15 := by
sorry

end NUMINAMATH_CALUDE_rectangular_garden_width_l1330_133066


namespace NUMINAMATH_CALUDE_mika_stickers_problem_l1330_133099

/-- The number of stickers Mika gave to her sister -/
def stickers_given_to_sister (initial bought birthday used left : ℕ) : ℕ :=
  initial + bought + birthday - used - left

theorem mika_stickers_problem (initial bought birthday used left : ℕ) 
  (h1 : initial = 20)
  (h2 : bought = 26)
  (h3 : birthday = 20)
  (h4 : used = 58)
  (h5 : left = 2) :
  stickers_given_to_sister initial bought birthday used left = 6 := by
sorry

end NUMINAMATH_CALUDE_mika_stickers_problem_l1330_133099


namespace NUMINAMATH_CALUDE_base_eight_31_equals_25_l1330_133012

/-- Converts a two-digit base-eight number to base-ten -/
def base_eight_to_ten (tens : Nat) (ones : Nat) : Nat :=
  tens * 8 + ones

/-- The base-eight number 31 is equal to the base-ten number 25 -/
theorem base_eight_31_equals_25 : base_eight_to_ten 3 1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_31_equals_25_l1330_133012


namespace NUMINAMATH_CALUDE_prism_faces_count_l1330_133027

/-- Represents a polygonal prism -/
structure Prism where
  base_sides : ℕ
  edges : ℕ := 3 * base_sides
  faces : ℕ := 2 + base_sides

/-- Represents a polygonal pyramid -/
structure Pyramid where
  base_sides : ℕ
  edges : ℕ := 2 * base_sides

/-- Theorem stating that a prism has 8 faces given the conditions -/
theorem prism_faces_count (p : Prism) (py : Pyramid) 
  (h1 : p.base_sides = py.base_sides) 
  (h2 : p.edges + py.edges = 30) : p.faces = 8 := by
  sorry

end NUMINAMATH_CALUDE_prism_faces_count_l1330_133027


namespace NUMINAMATH_CALUDE_cube_edge_length_l1330_133031

/-- Given a cube with surface area 216 cm², prove that the length of its edge is 6 cm. -/
theorem cube_edge_length (surface_area : ℝ) (edge_length : ℝ) 
  (h1 : surface_area = 216)
  (h2 : surface_area = 6 * edge_length^2) : 
  edge_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l1330_133031
